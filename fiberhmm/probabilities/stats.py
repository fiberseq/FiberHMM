#!/usr/bin/env python3
"""
FiberHMM stats_utils.py
Shared statistics and plotting utilities for probability generation scripts.

Contains:
- generate_probability_stats: Generate statistics report and plots for probability tables
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from fiberhmm.probabilities.context_counter import ContextCounter


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
    summary_file = os.path.join(plots_dir, f"{base_name}_k{context_size}_stats.txt")
    with open(summary_file, 'w') as f:
        f.write(f"{title_prefix} Emission Probability Statistics (k={context_size}, {2*context_size+1}-mer)\n")
        f.write("=" * 60 + "\n\n")

        for base in accessible_counters.keys():
            acc = accessible_counters[base]
            inacc = inaccessible_counters[base]

            acc_rate = acc.total_modified / max(1, acc.total_positions)
            inacc_rate = inacc.total_modified / max(1, inacc.total_positions)

            f.write(f"{base}-centered Contexts\n")
            f.write("-" * 40 + "\n")
            f.write(f"\nAccessible:\n")
            f.write(f"  Total positions:     {acc.total_positions:,}\n")
            f.write(f"  Modified positions:  {acc.total_modified:,}\n")
            f.write(f"  Modification rate:   {acc_rate:.4f} ({acc_rate*100:.2f}%)\n")
            f.write(f"  Unique contexts:     {len(acc.counts):,}\n")

            f.write(f"\nInaccessible:\n")
            f.write(f"  Total positions:     {inacc.total_positions:,}\n")
            f.write(f"  Modified positions:  {inacc.total_modified:,}\n")
            f.write(f"  Modification rate:   {inacc_rate:.4f} ({inacc_rate*100:.2f}%)\n")
            f.write(f"  Unique contexts:     {len(inacc.counts):,}\n")

            f.write(f"\nSeparation:\n")
            f.write(f"  Rate difference:     {acc_rate - inacc_rate:.4f}\n")
            f.write(f"  Fold enrichment:     {acc_rate / max(0.001, inacc_rate):.2f}x\n")

            # Per-context stats
            _, acc_probs = acc.get_encoding_table(context_size)
            _, inacc_probs = inacc.get_encoding_table(context_size)

            acc_ratios = acc_probs['ratio'].values
            inacc_ratios = inacc_probs['ratio'].values

            # Filter to contexts with data
            acc_with_data = acc_ratios[acc_probs['hit'] + acc_probs['nohit'] > 0]
            inacc_with_data = inacc_ratios[inacc_probs['hit'] + inacc_probs['nohit'] > 0]

            f.write(f"\nPer-context statistics (k={context_size}, {2*context_size+1}-mer):\n")
            if len(acc_with_data) > 0:
                f.write(f"  Accessible contexts with data: {len(acc_with_data)}\n")
                f.write(f"    Prob range:  {np.min(acc_with_data):.4f} - {np.max(acc_with_data):.4f}\n")
                f.write(f"    Prob median: {np.median(acc_with_data):.4f}\n")
                f.write(f"    Prob mean:   {np.mean(acc_with_data):.4f}\n")
                f.write(f"    Prob std:    {np.std(acc_with_data):.4f}\n")

            if len(inacc_with_data) > 0:
                f.write(f"  Inaccessible contexts with data: {len(inacc_with_data)}\n")
                f.write(f"    Prob range:  {np.min(inacc_with_data):.4f} - {np.max(inacc_with_data):.4f}\n")
                f.write(f"    Prob median: {np.median(inacc_with_data):.4f}\n")
                f.write(f"    Prob mean:   {np.mean(inacc_with_data):.4f}\n")
                f.write(f"    Prob std:    {np.std(inacc_with_data):.4f}\n")

            f.write("\n")

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

    pdf_path = os.path.join(plots_dir, f"{base_name}_k{context_size}_stats.pdf")

    with PdfPages(pdf_path) as pdf:
        for base in accessible_counters.keys():
            acc = accessible_counters[base]
            inacc = inaccessible_counters[base]

            _, acc_probs = acc.get_encoding_table(context_size)
            _, inacc_probs = inacc.get_encoding_table(context_size)

            # Page: Distribution comparison
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.suptitle(f'{base}-centered Context Statistics (k={context_size})',
                        fontsize=14, fontweight='bold')

            # 1. Histogram of methylation probabilities
            ax = axes[0, 0]
            acc_ratios = acc_probs['ratio'].values
            inacc_ratios = inacc_probs['ratio'].values

            bins = np.linspace(0, 1, 51)
            ax.hist(acc_ratios, bins=bins, alpha=0.6, label='Accessible', color='forestgreen', edgecolor='white')
            ax.hist(inacc_ratios, bins=bins, alpha=0.6, label='Inaccessible', color='firebrick', edgecolor='white')
            if len(acc_ratios[acc_ratios > 0]) > 0:
                ax.axvline(np.median(acc_ratios[acc_ratios > 0]), color='green', linestyle='--', linewidth=2)
            if len(inacc_ratios[inacc_ratios > 0]) > 0:
                ax.axvline(np.median(inacc_ratios[inacc_ratios > 0]), color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('P(methylation | context)')
            ax.set_ylabel('Number of contexts')
            ax.set_title('Emission Probability Distributions')
            ax.legend()

            # 2. Scatter plot: accessible vs inaccessible
            ax = axes[0, 1]
            # Merge by context to ensure alignment
            merged = acc_probs[['context', 'ratio', 'hit', 'nohit']].merge(
                inacc_probs[['context', 'ratio', 'hit', 'nohit']],
                on='context', suffixes=('_acc', '_inacc')
            )
            merged['total_acc'] = merged['hit_acc'] + merged['nohit_acc']
            merged['total_inacc'] = merged['hit_inacc'] + merged['nohit_inacc']
            # Filter to contexts with data in both
            mask = (merged['total_acc'] > 10) & (merged['total_inacc'] > 10)
            if mask.sum() > 0:
                ax.scatter(merged.loc[mask, 'ratio_inacc'], merged.loc[mask, 'ratio_acc'],
                          alpha=0.5, s=10, c='steelblue')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
            ax.set_xlabel('P(m | inaccessible)')
            ax.set_ylabel('P(m | accessible)')
            ax.set_title(f'Accessible vs Inaccessible ({mask.sum():,} contexts)')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend()

            # 3. Log-odds (separation score) - use merged data
            ax = axes[1, 0]
            # Calculate log-odds for each context (use merged for alignment)
            eps = 0.001
            if len(merged) > 0:
                acc_clipped = np.clip(merged['ratio_acc'].values, eps, 1 - eps)
                inacc_clipped = np.clip(merged['ratio_inacc'].values, eps, 1 - eps)
                log_odds = np.log2(acc_clipped / inacc_clipped)

                # Filter extreme values
                log_odds_filtered = log_odds[np.isfinite(log_odds)]
                if len(log_odds_filtered) > 0:
                    bins = np.linspace(-5, 10, 51)
                    ax.hist(log_odds_filtered, bins=bins, color='purple', alpha=0.7, edgecolor='white')
                    ax.axvline(0, color='black', linestyle='-', linewidth=1)
                    ax.axvline(np.median(log_odds_filtered), color='red', linestyle='--',
                              label=f'Median: {np.median(log_odds_filtered):.2f}')
                    ax.set_xlabel('Log2(P_accessible / P_inaccessible)')
                    ax.set_ylabel('Number of contexts')
                    ax.set_title('Separation Score (Log-Odds)')
                    ax.legend()

            # 4. Top differentiating contexts (use merged)
            ax = axes[1, 1]
            if len(merged) > 0:
                # Find contexts with best separation
                merged['diff'] = merged['ratio_acc'] - merged['ratio_inacc']
                top_contexts = merged.nlargest(15, 'diff')

                contexts = top_contexts['context'].values
                acc_vals = top_contexts['ratio_acc'].values
                inacc_vals = top_contexts['ratio_inacc'].values

                y_pos = np.arange(len(contexts))
                width = 0.35

                ax.barh(y_pos - width/2, inacc_vals, width, label='Inaccessible', color='firebrick', alpha=0.8)
                ax.barh(y_pos + width/2, acc_vals, width, label='Accessible', color='forestgreen', alpha=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(contexts, fontsize=7, fontfamily='monospace')
                ax.set_xlabel('P(methylation)')
                ax.set_title('Top Differentiating Contexts')
                ax.legend(loc='lower right')
                ax.set_xlim(0, 1)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # Page 2: Count distributions
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.suptitle(f'{base}-centered Context Counts', fontsize=14, fontweight='bold')

            # Total observations per context
            ax = axes[0, 0]
            acc_total = acc_probs['hit'] + acc_probs['nohit']
            inacc_total = inacc_probs['hit'] + inacc_probs['nohit']

            ax.hist(np.log10(acc_total[acc_total > 0] + 1), bins=50, alpha=0.6,
                   label='Accessible', color='forestgreen')
            ax.hist(np.log10(inacc_total[inacc_total > 0] + 1), bins=50, alpha=0.6,
                   label='Inaccessible', color='firebrick')
            ax.set_xlabel('Log10(observations + 1)')
            ax.set_ylabel('Number of contexts')
            ax.set_title('Observations per Context')
            ax.legend()

            # Cumulative coverage
            ax = axes[0, 1]
            acc_sorted = np.sort(acc_total.values)[::-1]
            inacc_sorted = np.sort(inacc_total.values)[::-1]

            if np.sum(acc_sorted) > 0:
                acc_cumsum = np.cumsum(acc_sorted) / np.sum(acc_sorted) * 100
                ax.plot(range(len(acc_cumsum)), acc_cumsum, label='Accessible', color='forestgreen')
            if np.sum(inacc_sorted) > 0:
                inacc_cumsum = np.cumsum(inacc_sorted) / np.sum(inacc_sorted) * 100
                ax.plot(range(len(inacc_cumsum)), inacc_cumsum, label='Inaccessible', color='firebrick')
            ax.axhline(90, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Number of contexts (ranked)')
            ax.set_ylabel('Cumulative % of observations')
            ax.set_title('Context Coverage (Lorenz-like)')
            ax.legend()

            # Hit rate comparison - use merged data for alignment
            ax = axes[1, 0]
            if len(merged) > 0:
                mask = (merged['total_acc'] > 100) & (merged['total_inacc'] > 100)
                if np.sum(mask) > 0:
                    ax.scatter(merged.loc[mask, 'total_acc'], merged.loc[mask, 'ratio_acc'],
                              alpha=0.5, s=10, c='forestgreen', label='Accessible')
                    ax.scatter(merged.loc[mask, 'total_inacc'], merged.loc[mask, 'ratio_inacc'],
                              alpha=0.5, s=10, c='firebrick', label='Inaccessible')
                    ax.set_xscale('log')
                    ax.set_xlabel('Total observations (log scale)')
                    ax.set_ylabel('Methylation probability')
                    ax.set_title('Probability vs Coverage')
                    ax.legend()

            # Context frequency comparison - use merged data
            ax = axes[1, 1]
            if len(merged) > 0:
                mask = (merged['total_acc'] > 0) & (merged['total_inacc'] > 0)
                if np.sum(mask) > 0:
                    ax.scatter(merged.loc[mask, 'total_inacc'], merged.loc[mask, 'total_acc'],
                              alpha=0.5, s=10, c='steelblue')
                    max_val = max(merged.loc[mask, 'total_acc'].max(), merged.loc[mask, 'total_inacc'].max())
                    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
                    ax.set_xlabel('Inaccessible observations')
                    ax.set_ylabel('Accessible observations')
                    ax.set_title('Context Frequency Comparison')
                    ax.set_xscale('log')
                    ax.set_yscale('log')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"  Plots: {pdf_path}")

    # Also save a simple PNG of the key distribution
    for base in accessible_counters.keys():
        acc = accessible_counters[base]
        inacc = inaccessible_counters[base]

        _, acc_probs = acc.get_encoding_table(context_size)
        _, inacc_probs = inacc.get_encoding_table(context_size)

        fig, ax = plt.subplots(figsize=(8, 5))

        bins = np.linspace(0, 1, 51)
        ax.hist(acc_probs['ratio'].values, bins=bins, alpha=0.6,
               label=f'Accessible (n={acc.total_positions:,})',
               color='forestgreen', edgecolor='white')
        ax.hist(inacc_probs['ratio'].values, bins=bins, alpha=0.6,
               label=f'Inaccessible (n={inacc.total_positions:,})',
               color='firebrick', edgecolor='white')

        ax.set_xlabel('P(methylation | context)', fontsize=12)
        ax.set_ylabel('Number of contexts', fontsize=12)
        ax.set_title(f'{base}-centered Emission Probability Distributions (k={context_size})', fontsize=14)
        ax.legend(fontsize=11)

        plt.tight_layout()
        png_path = os.path.join(plots_dir, f"{base_name}_{base}_k{context_size}_distribution.png")
        plt.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"  Plot: {png_path}")
