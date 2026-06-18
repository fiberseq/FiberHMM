#!/usr/bin/env python3
"""
FiberHMM stats_utils.py
Shared statistics and plotting utilities for probability generation scripts.

Contains:
- generate_probability_stats: Generate statistics report and plots for probability tables
"""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterator

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from fiberhmm.probabilities.context_counter import ContextCounter


def _modification_rate(counter: 'ContextCounter') -> float:
    return counter.total_modified / max(1, counter.total_positions)


def _fold_enrichment(accessible_rate: float, inaccessible_rate: float,
                     floor: float = 0.001) -> float:
    return accessible_rate / max(floor, inaccessible_rate)


def _probability_context_label(context_size: int) -> str:
    return f"k={context_size}, {2*context_size+1}-mer"


def _context_observation_totals(
    prob_table, hit_col: str = 'hit', nohit_col: str = 'nohit',
):
    return prob_table[hit_col] + prob_table[nohit_col]


def _probability_ratios_with_data(prob_table) -> np.ndarray:
    ratios = prob_table['ratio'].values
    return ratios[_context_observation_totals(prob_table) > 0]


@dataclass(frozen=True)
class _CounterRateSummary:
    total_positions: int
    total_modified: int
    rate: float
    unique_contexts: int


def _counter_rate_summary(counter: 'ContextCounter') -> _CounterRateSummary:
    rate = _modification_rate(counter)
    return _CounterRateSummary(
        total_positions=counter.total_positions,
        total_modified=counter.total_modified,
        rate=rate,
        unique_contexts=len(counter.counts),
    )


def _write_counter_rate_summary(
    handle,
    label: str,
    summary: _CounterRateSummary,
) -> None:
    handle.write(f"\n{label}:\n")
    handle.write(f"  Total positions:     {summary.total_positions:,}\n")
    handle.write(f"  Modified positions:  {summary.total_modified:,}\n")
    rate = summary.rate
    handle.write(f"  Modification rate:   {rate:.4f} ({rate*100:.2f}%)\n")
    handle.write(f"  Unique contexts:     {summary.unique_contexts:,}\n")


def _merged_probability_table(acc_probs, inacc_probs):
    merged = acc_probs[['context', 'ratio', 'hit', 'nohit']].merge(
        inacc_probs[['context', 'ratio', 'hit', 'nohit']],
        on='context', suffixes=('_acc', '_inacc')
    )
    merged['total_acc'] = _context_observation_totals(
        merged, 'hit_acc', 'nohit_acc',
    )
    merged['total_inacc'] = _context_observation_totals(
        merged, 'hit_inacc', 'nohit_inacc',
    )
    return merged


@dataclass(frozen=True)
class _ProbabilityTablesForBase:
    accessible: 'pd.DataFrame'
    inaccessible: 'pd.DataFrame'


@dataclass(frozen=True)
class _BaseProbabilityContext:
    base: str
    accessible_counter: 'ContextCounter'
    inaccessible_counter: 'ContextCounter'
    probability_tables: _ProbabilityTablesForBase


def _probability_tables_for_base(
    accessible_counters,
    inaccessible_counters,
    base: str,
    context_size: int,
) -> _ProbabilityTablesForBase:
    _, acc_probs = accessible_counters[base].get_encoding_table(context_size)
    _, inacc_probs = inaccessible_counters[base].get_encoding_table(context_size)
    return _ProbabilityTablesForBase(acc_probs, inacc_probs)


def _base_probability_contexts(
    accessible_counters: Dict[str, 'ContextCounter'],
    inaccessible_counters: Dict[str, 'ContextCounter'],
    context_size: int,
) -> Iterator[_BaseProbabilityContext]:
    for base in accessible_counters.keys():
        yield _BaseProbabilityContext(
            base=base,
            accessible_counter=accessible_counters[base],
            inaccessible_counter=inaccessible_counters[base],
            probability_tables=_probability_tables_for_base(
                accessible_counters, inaccessible_counters, base, context_size,
            ),
        )


def _filtered_log_odds(merged, eps: float = 0.001) -> np.ndarray:
    if len(merged) == 0:
        return np.array([])
    acc_clipped = np.clip(merged['ratio_acc'].values, eps, 1 - eps)
    inacc_clipped = np.clip(merged['ratio_inacc'].values, eps, 1 - eps)
    log_odds = np.log2(acc_clipped / inacc_clipped)
    return log_odds[np.isfinite(log_odds)]


def _top_differentiating_contexts(merged, n: int = 15):
    if len(merged) == 0:
        return merged.copy()
    ranked = merged.copy()
    ranked['diff'] = ranked['ratio_acc'] - ranked['ratio_inacc']
    return ranked.nlargest(n, 'diff')


def _contexts_with_min_observations(merged, min_total: int):
    return (
        (merged['total_acc'] > min_total)
        & (merged['total_inacc'] > min_total)
    )


def _cumulative_observation_percentages(totals) -> np.ndarray:
    sorted_totals = np.sort(np.asarray(totals))[::-1]
    total = np.sum(sorted_totals)
    if total <= 0:
        return np.array([])
    return np.cumsum(sorted_totals) / total * 100


def _positive_log10_observations(totals) -> np.ndarray:
    totals = np.asarray(totals)
    positive = totals[totals > 0]
    return np.log10(positive + 1)


def _positive_median(values):
    values = np.asarray(values)
    positive = values[values > 0]
    if len(positive) == 0:
        return None
    return np.median(positive)


def _plot_probability_ratio_histograms(
    ax,
    acc_ratios,
    inacc_ratios,
    accessible_label: str = 'Accessible',
    inaccessible_label: str = 'Inaccessible',
    show_positive_medians: bool = True,
) -> None:
    bins = np.linspace(0, 1, 51)
    ax.hist(
        acc_ratios,
        bins=bins,
        alpha=0.6,
        label=accessible_label,
        color='forestgreen',
        edgecolor='white',
    )
    ax.hist(
        inacc_ratios,
        bins=bins,
        alpha=0.6,
        label=inaccessible_label,
        color='firebrick',
        edgecolor='white',
    )

    if not show_positive_medians:
        return

    acc_median = _positive_median(acc_ratios)
    if acc_median is not None:
        ax.axvline(acc_median, color='green', linestyle='--', linewidth=2)

    inacc_median = _positive_median(inacc_ratios)
    if inacc_median is not None:
        ax.axvline(inacc_median, color='red', linestyle='--', linewidth=2)


def _plot_accessible_inaccessible_probability_scatter(ax, merged) -> None:
    mask = _contexts_with_min_observations(merged, 10)
    n_contexts = int(mask.sum())
    if n_contexts > 0:
        ax.scatter(
            merged.loc[mask, 'ratio_inacc'],
            merged.loc[mask, 'ratio_acc'],
            alpha=0.5,
            s=10,
            c='steelblue',
        )
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('P(m | inaccessible)')
    ax.set_ylabel('P(m | accessible)')
    ax.set_title(f'Accessible vs Inaccessible ({n_contexts:,} contexts)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()


def _plot_log_odds_distribution(ax, merged) -> None:
    log_odds_filtered = _filtered_log_odds(merged)
    if len(log_odds_filtered) == 0:
        return

    bins = np.linspace(-5, 10, 51)
    median_log_odds = np.median(log_odds_filtered)
    ax.hist(
        log_odds_filtered,
        bins=bins,
        color='purple',
        alpha=0.7,
        edgecolor='white',
    )
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(
        median_log_odds,
        color='red',
        linestyle='--',
        label=f'Median: {median_log_odds:.2f}',
    )
    ax.set_xlabel('Log2(P_accessible / P_inaccessible)')
    ax.set_ylabel('Number of contexts')
    ax.set_title('Separation Score (Log-Odds)')
    ax.legend()


def _plot_top_differentiating_contexts(ax, merged) -> None:
    if len(merged) == 0:
        return

    top_contexts = _top_differentiating_contexts(merged)
    contexts = top_contexts['context'].values
    acc_vals = top_contexts['ratio_acc'].values
    inacc_vals = top_contexts['ratio_inacc'].values

    y_pos = np.arange(len(contexts))
    width = 0.35

    ax.barh(
        y_pos - width / 2,
        inacc_vals,
        width,
        label='Inaccessible',
        color='firebrick',
        alpha=0.8,
    )
    ax.barh(
        y_pos + width / 2,
        acc_vals,
        width,
        label='Accessible',
        color='forestgreen',
        alpha=0.8,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(contexts, fontsize=7, fontfamily='monospace')
    ax.set_xlabel('P(methylation)')
    ax.set_title('Top Differentiating Contexts')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)


def _plot_observations_per_context(ax, acc_total, inacc_total) -> None:
    ax.hist(
        _positive_log10_observations(acc_total),
        bins=50,
        alpha=0.6,
        label='Accessible',
        color='forestgreen',
    )
    ax.hist(
        _positive_log10_observations(inacc_total),
        bins=50,
        alpha=0.6,
        label='Inaccessible',
        color='firebrick',
    )
    ax.set_xlabel('Log10(observations + 1)')
    ax.set_ylabel('Number of contexts')
    ax.set_title('Observations per Context')
    ax.legend()


def _plot_context_coverage(ax, acc_total, inacc_total) -> None:
    acc_cumsum = _cumulative_observation_percentages(acc_total)
    inacc_cumsum = _cumulative_observation_percentages(inacc_total)
    if len(acc_cumsum) > 0:
        ax.plot(
            range(len(acc_cumsum)),
            acc_cumsum,
            label='Accessible',
            color='forestgreen',
        )
    if len(inacc_cumsum) > 0:
        ax.plot(
            range(len(inacc_cumsum)),
            inacc_cumsum,
            label='Inaccessible',
            color='firebrick',
        )
    ax.axhline(90, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of contexts (ranked)')
    ax.set_ylabel('Cumulative % of observations')
    ax.set_title('Context Coverage (Lorenz-like)')
    ax.legend()


def _plot_probability_vs_coverage(ax, merged) -> None:
    if len(merged) == 0:
        return

    mask = _contexts_with_min_observations(merged, 100)
    if np.sum(mask) == 0:
        return

    ax.scatter(
        merged.loc[mask, 'total_acc'],
        merged.loc[mask, 'ratio_acc'],
        alpha=0.5,
        s=10,
        c='forestgreen',
        label='Accessible',
    )
    ax.scatter(
        merged.loc[mask, 'total_inacc'],
        merged.loc[mask, 'ratio_inacc'],
        alpha=0.5,
        s=10,
        c='firebrick',
        label='Inaccessible',
    )
    ax.set_xscale('log')
    ax.set_xlabel('Total observations (log scale)')
    ax.set_ylabel('Methylation probability')
    ax.set_title('Probability vs Coverage')
    ax.legend()


def _plot_context_frequency_comparison(ax, merged) -> None:
    if len(merged) == 0:
        return

    mask = _contexts_with_min_observations(merged, 0)
    if np.sum(mask) == 0:
        return

    ax.scatter(
        merged.loc[mask, 'total_inacc'],
        merged.loc[mask, 'total_acc'],
        alpha=0.5,
        s=10,
        c='steelblue',
    )
    max_val = max(
        merged.loc[mask, 'total_acc'].max(),
        merged.loc[mask, 'total_inacc'].max(),
    )
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
    ax.set_xlabel('Inaccessible observations')
    ax.set_ylabel('Accessible observations')
    ax.set_title('Context Frequency Comparison')
    ax.set_xscale('log')
    ax.set_yscale('log')


def _plot_probability_distribution_png_axis(
    ax,
    acc: 'ContextCounter',
    inacc: 'ContextCounter',
    acc_probs,
    inacc_probs,
    base: str,
    context_size: int,
) -> None:
    _plot_probability_ratio_histograms(
        ax,
        acc_probs['ratio'].values,
        inacc_probs['ratio'].values,
        accessible_label=f'Accessible (n={acc.total_positions:,})',
        inaccessible_label=f'Inaccessible (n={inacc.total_positions:,})',
        show_positive_medians=False,
    )

    ax.set_xlabel('P(methylation | context)', fontsize=12)
    ax.set_ylabel('Number of contexts', fontsize=12)
    ax.set_title(
        f'{base}-centered Emission Probability Distributions '
        f'(k={context_size})',
        fontsize=14,
    )
    ax.legend(fontsize=11)


def _write_probability_ratio_summary(handle, label: str, ratios: np.ndarray) -> None:
    if len(ratios) == 0:
        return
    handle.write(f"  {label} contexts with data: {len(ratios)}\n")
    handle.write(
        f"    Prob range:  {np.min(ratios):.4f} - "
        f"{np.max(ratios):.4f}\n"
    )
    handle.write(f"    Prob median: {np.median(ratios):.4f}\n")
    handle.write(f"    Prob mean:   {np.mean(ratios):.4f}\n")
    handle.write(f"    Prob std:    {np.std(ratios):.4f}\n")


def _write_base_probability_summary(
    handle,
    base: str,
    acc: 'ContextCounter',
    inacc: 'ContextCounter',
    acc_probs,
    inacc_probs,
    context_size: int,
) -> None:
    acc_summary = _counter_rate_summary(acc)
    inacc_summary = _counter_rate_summary(inacc)
    acc_rate = acc_summary.rate
    inacc_rate = inacc_summary.rate

    handle.write(f"{base}-centered Contexts\n")
    handle.write("-" * 40 + "\n")
    _write_counter_rate_summary(handle, "Accessible", acc_summary)
    _write_counter_rate_summary(handle, "Inaccessible", inacc_summary)

    handle.write("\nSeparation:\n")
    handle.write(f"  Rate difference:     {acc_rate - inacc_rate:.4f}\n")
    handle.write(f"  Fold enrichment:     {_fold_enrichment(acc_rate, inacc_rate):.2f}x\n")

    acc_with_data = _probability_ratios_with_data(acc_probs)
    inacc_with_data = _probability_ratios_with_data(inacc_probs)

    handle.write(
        f"\nPer-context statistics "
        f"({_probability_context_label(context_size)}):\n"
    )
    _write_probability_ratio_summary(handle, "Accessible", acc_with_data)
    _write_probability_ratio_summary(handle, "Inaccessible", inacc_with_data)

    handle.write("\n")


def _write_probability_stats_summary(summary_file: str,
                                     accessible_counters: Dict[str, 'ContextCounter'],
                                     inaccessible_counters: Dict[str, 'ContextCounter'],
                                     context_size: int,
                                     title_prefix: str) -> None:
    with open(summary_file, 'w') as f:
        f.write(
            f"{title_prefix} Emission Probability Statistics "
            f"({_probability_context_label(context_size)})\n"
        )
        f.write("=" * 60 + "\n\n")

        for base_context in _base_probability_contexts(
            accessible_counters, inaccessible_counters, context_size,
        ):
            probability_tables = base_context.probability_tables
            _write_base_probability_summary(
                f,
                base_context.base,
                base_context.accessible_counter,
                base_context.inaccessible_counter,
                probability_tables.accessible,
                probability_tables.inaccessible,
                context_size,
            )


def _probability_stats_output_path(
    plots_dir: str,
    base_name: str,
    context_size: int,
    extension: str,
) -> str:
    return os.path.join(
        plots_dir,
        f"{base_name}_k{context_size}_stats.{extension}",
    )


def _probability_distribution_plot_path(
    plots_dir: str,
    base_name: str,
    base: str,
    context_size: int,
) -> str:
    return os.path.join(
        plots_dir,
        f"{base_name}_{base}_k{context_size}_distribution.png",
    )


def _write_probability_distribution_pdf_page(
    plt,
    pdf,
    base: str,
    context_size: int,
    acc_probs,
    inacc_probs,
):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    try:
        fig.suptitle(
            f'{base}-centered Context Statistics (k={context_size})',
            fontsize=14,
            fontweight='bold',
        )

        ax = axes[0, 0]
        acc_ratios = acc_probs['ratio'].values
        inacc_ratios = inacc_probs['ratio'].values
        _plot_probability_ratio_histograms(ax, acc_ratios, inacc_ratios)
        ax.set_xlabel('P(methylation | context)')
        ax.set_ylabel('Number of contexts')
        ax.set_title('Emission Probability Distributions')
        ax.legend()

        merged = _merged_probability_table(acc_probs, inacc_probs)
        _plot_accessible_inaccessible_probability_scatter(axes[0, 1], merged)
        _plot_log_odds_distribution(axes[1, 0], merged)
        _plot_top_differentiating_contexts(axes[1, 1], merged)

        plt.tight_layout()
        pdf.savefig(fig)
    finally:
        plt.close(fig)


def _write_probability_counts_pdf_page(
    plt,
    pdf,
    base: str,
    acc_probs,
    inacc_probs,
):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    try:
        fig.suptitle(
            f'{base}-centered Context Counts',
            fontsize=14,
            fontweight='bold',
        )

        acc_total = _context_observation_totals(acc_probs)
        inacc_total = _context_observation_totals(inacc_probs)
        merged = _merged_probability_table(acc_probs, inacc_probs)

        _plot_observations_per_context(axes[0, 0], acc_total, inacc_total)
        _plot_context_coverage(axes[0, 1], acc_total.values, inacc_total.values)
        _plot_probability_vs_coverage(axes[1, 0], merged)
        _plot_context_frequency_comparison(axes[1, 1], merged)

        plt.tight_layout()
        pdf.savefig(fig)
    finally:
        plt.close(fig)


def _save_probability_distribution_png(
    plt,
    plots_dir: str,
    base_name: str,
    base: str,
    context_size: int,
    acc,
    inacc,
    acc_probs,
    inacc_probs,
) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    png_path = _probability_distribution_plot_path(
        plots_dir, base_name, base, context_size,
    )
    try:
        _plot_probability_distribution_png_axis(
            ax, acc, inacc, acc_probs, inacc_probs, base, context_size,
        )

        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
    finally:
        plt.close(fig)
    print(f"  Plot: {png_path}")
    return png_path


def _write_probability_stats_pdf(
    plt,
    pdf_pages_factory,
    pdf_path: str,
    accessible_counters: Dict[str, 'ContextCounter'],
    inaccessible_counters: Dict[str, 'ContextCounter'],
    context_size: int,
) -> str:
    with pdf_pages_factory(pdf_path) as pdf:
        for base_context in _base_probability_contexts(
            accessible_counters, inaccessible_counters, context_size,
        ):
            probability_tables = base_context.probability_tables

            _write_probability_distribution_pdf_page(
                plt, pdf, base_context.base, context_size,
                probability_tables.accessible,
                probability_tables.inaccessible,
            )
            _write_probability_counts_pdf_page(
                plt, pdf, base_context.base,
                probability_tables.accessible,
                probability_tables.inaccessible,
            )
    return pdf_path


def _save_probability_distribution_pngs(
    plt,
    plots_dir: str,
    base_name: str,
    accessible_counters: Dict[str, 'ContextCounter'],
    inaccessible_counters: Dict[str, 'ContextCounter'],
    context_size: int,
) -> list[str]:
    png_paths = []
    for base_context in _base_probability_contexts(
        accessible_counters, inaccessible_counters, context_size,
    ):
        probability_tables = base_context.probability_tables

        png_paths.append(
            _save_probability_distribution_png(
                plt, plots_dir, base_name, base_context.base, context_size,
                base_context.accessible_counter,
                base_context.inaccessible_counter,
                probability_tables.accessible,
                probability_tables.inaccessible,
            )
        )
    return png_paths


def _write_probability_plot_outputs(
    plt,
    PdfPages,
    plots_dir: str,
    base_name: str,
    accessible_counters: Dict[str, 'ContextCounter'],
    inaccessible_counters: Dict[str, 'ContextCounter'],
    context_size: int,
) -> str:
    pdf_path = _probability_stats_output_path(
        plots_dir, base_name, context_size, "pdf",
    )
    _write_probability_stats_pdf(
        plt,
        PdfPages,
        pdf_path,
        accessible_counters,
        inaccessible_counters,
        context_size,
    )
    print(f"  Plots: {pdf_path}")

    _save_probability_distribution_pngs(
        plt,
        plots_dir,
        base_name,
        accessible_counters,
        inaccessible_counters,
        context_size,
    )
    return pdf_path


def generate_probability_stats(accessible_counters: Dict[str, 'ContextCounter'],
                                inaccessible_counters: Dict[str, 'ContextCounter'],
                                plots_dir: str, base_name: str, context_size: int = 3,
                                title_prefix: str = "FiberHMM"):
    """
    Generate summary statistics and QC plots for emission probability tables.

    This is the shared implementation used by generate_probs.py and bootstrap_probs.py.

    Args:
        accessible_counters: Dict mapping base -> ContextCounter for accessible regions
        inaccessible_counters: Dict mapping base -> ContextCounter for inaccessible regions
        plots_dir: Directory to write plots and stats
        base_name: Base name for output files
        context_size: Context size k for statistics
        title_prefix: Prefix for plot titles (e.g., "FiberHMM" or "FiberHMM Bootstrapped")
    """

    # Write text summary (includes k in filename)
    summary_file = _probability_stats_output_path(
        plots_dir, base_name, context_size, "txt",
    )
    _write_probability_stats_summary(
        summary_file, accessible_counters, inaccessible_counters,
        context_size, title_prefix,
    )

    print(f"  Summary: {summary_file}")

    # Generate plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print("  Warning: matplotlib not installed. Skipping plots.")
        print("  Install with: pip install matplotlib")
        return

    _write_probability_plot_outputs(
        plt,
        PdfPages,
        plots_dir,
        base_name,
        accessible_counters,
        inaccessible_counters,
        context_size,
    )
