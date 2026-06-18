#!/usr/bin/env python3
"""
FiberHMM utilities: convert, inspect, transfer, adjust.

Consolidated utility script for model and emission probability management.

Usage:
    fiberhmm-utils convert input.pickle output.json
    fiberhmm-utils inspect model.json
    fiberhmm-utils transfer --target daf.bam --reference-bam fiber.bam -o out/
    fiberhmm-utils adjust model.json --state accessible --scale 1.1 -o adjusted.json
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from fiberhmm.cli.common import MODE_CHOICES
from fiberhmm.core.model_io import load_model_with_metadata, save_model
from fiberhmm.inference.read_filters import is_primary_mapped_alignment
from fiberhmm.io.ma_tags import flip_intervals_to_seq
from fiberhmm.io.path_status import path_is_regular_file
from fiberhmm.probabilities.output_paths import (
    combined_probability_table_path as _combined_probability_table_path,
)
from fiberhmm.probabilities.output_paths import (
    probability_table_path as _probability_table_path,
)
from fiberhmm.probabilities.utils import get_base_name, setup_output_dirs


@dataclass(frozen=True)
class _EmissionEstimate:
    p_acc: float
    p_inacc: float
    diagnostics: dict


@dataclass(frozen=True)
class _RegressionInputs:
    merged: pd.DataFrame
    diagnostics: dict


@dataclass(frozen=True)
class _EmissionStats:
    minimum: float
    maximum: float
    mean: float


@dataclass(frozen=True)
class _EmissionScalingResult:
    adjusted: np.ndarray
    before: _EmissionStats
    after: _EmissionStats


@dataclass(frozen=True)
class _TransferAccessibilityInputs:
    counters: Optional[dict]
    priors: Optional[pd.DataFrame]


# =============================================================================
# convert subcommand
# =============================================================================

def _load_pickle_model_raw(filepath):
    """Load model from pickle file (raw, without FiberHMM wrapper)."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        model = data.get('model', data)
        context_size = data.get('context_size', 3)
        mode = data.get('mode', 'pacbio-fiber')

        if hasattr(model, 'startprob_'):
            return {
                'startprob': np.array(model.startprob_),
                'transmat': np.array(model.transmat_),
                'emissionprob': np.array(model.emissionprob_),
                'n_states': getattr(model, 'n_states', 2),
                'context_size': context_size,
                'mode': mode,
            }
        else:
            return {
                'startprob': np.array(data.get('startprob', data.get('startprob_'))),
                'transmat': np.array(data.get('transmat', data.get('transmat_'))),
                'emissionprob': np.array(data.get('emissionprob', data.get('emissionprob_'))),
                'n_states': data.get('n_states', 2),
                'context_size': data.get('context_size', 3),
                'mode': data.get('mode', 'pacbio-fiber'),
            }
    else:
        context_size = getattr(data, 'context_size', 3)
        mode = getattr(data, 'mode', 'pacbio-fiber')
        return {
            'startprob': np.array(data.startprob_),
            'transmat': np.array(data.transmat_),
            'emissionprob': np.array(data.emissionprob_),
            'n_states': getattr(data, 'n_states', 2),
            'context_size': context_size,
            'mode': mode,
        }


def _load_npz_model_raw(filepath):
    """Load model from NPZ file (raw dict)."""
    with np.load(filepath, allow_pickle=True) as data:
        return {
            'startprob': data['startprob'],
            'transmat': data['transmat'],
            'emissionprob': data['emissionprob'],
            'n_states': int(data.get('n_states', 2)),
            'context_size': int(data.get('context_size', 3)),
            'mode': str(data.get('mode', 'pacbio-fiber')),
        }


def _load_raw_model_by_suffix(input_path: Path):
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()
    if suffix == '.npz':
        return _load_npz_model_raw(input_path)
    if suffix in ['.pickle', '.pkl']:
        return _load_pickle_model_raw(input_path)

    try:
        return _load_pickle_model_raw(input_path)
    except Exception:
        return _load_npz_model_raw(input_path)


def _raw_model_parameter_arrays(data: dict):
    return (
        np.array(data['startprob']),
        np.array(data['transmat']),
        np.array(data['emissionprob']),
    )


def _converted_model_json_payload(
    data: dict,
    startprob: np.ndarray,
    transmat: np.ndarray,
    emissionprob: np.ndarray,
) -> dict:
    context_size = data.get('context_size', 3)
    if context_size is None:
        context_size = 3
    mode = data.get('mode', 'pacbio-fiber')
    mode = str(mode).strip() if mode is not None else 'pacbio-fiber'
    return {
        'model_type': 'FiberHMM',
        'version': '2.0',
        'n_states': int(data['n_states']),
        'startprob': startprob.tolist(),
        'transmat': transmat.tolist(),
        'emissionprob': emissionprob.tolist(),
        'context_size': int(context_size),
        'mode': mode or 'pacbio-fiber',
    }


def _exit_if_missing_or_non_file(
    path,
    *,
    missing_prefix: str,
    non_file_prefix: str,
) -> None:
    path = Path(path)
    if not path.exists():
        print(f"Error: {missing_prefix}: {path}", file=sys.stderr)
        sys.exit(1)
    if not path_is_regular_file(path):
        print(f"Error: {non_file_prefix}: {path}", file=sys.stderr)
        sys.exit(1)


def cmd_convert(args):
    """Convert pickle/NPZ model to JSON."""
    input_path = Path(args.input)
    output_path = Path(args.output)

    _exit_if_missing_or_non_file(
        input_path,
        missing_prefix="Input file not found",
        non_file_prefix="Input path is not a file",
    )

    print(f"Loading model from {input_path}...")

    try:
        data = _load_raw_model_by_suffix(input_path)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    startprob, transmat, emissionprob = _raw_model_parameter_arrays(data)

    print(f"  States: {data['n_states']}")
    print(f"  Context size: k={data['context_size']} ({2*data['context_size']+1}-mer)")
    print(f"  Mode: {data['mode']}")
    print(f"  Start probs: {startprob}")
    print(f"  Transition matrix shape: {transmat.shape}")
    print(f"  Emission prob shape: {emissionprob.shape}")

    json_data = _converted_model_json_payload(
        data, startprob, transmat, emissionprob,
    )

    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    # Verify
    file_size = output_path.stat().st_size
    print(f"Done! Output size: {file_size:,} bytes")

    with open(output_path) as f:
        verify = json.load(f)
    assert len(verify['startprob']) == data['n_states']
    assert len(verify['transmat']) == data['n_states']
    assert len(verify['emissionprob']) == data['n_states']
    print("Verification passed!")


# =============================================================================
# inspect subcommand
# =============================================================================

def cmd_inspect(args):
    """Inspect a model file: print metadata, parameters, emission summary."""
    filepath = args.model

    _exit_if_missing_or_non_file(
        filepath,
        missing_prefix="File not found",
        non_file_prefix="Path is not a file",
    )

    model, context_size, mode = load_model_with_metadata(filepath, normalize=False)

    print(f"Model: {filepath}")
    print(f"  Format: {Path(filepath).suffix}")
    print(f"  Mode: {mode}")
    print(f"  Context size: k={context_size} ({2*context_size+1}-mer)")
    print(f"  States: {model.n_states}")
    print()

    print("Start probabilities:")
    for i, p in enumerate(model.startprob_):
        label = "inaccessible" if i == 0 else "accessible"
        print(f"  State {i} ({label}): {p:.6f}")
    print()

    print("Transition matrix:")
    labels = ["inacc", "acc"]
    header = "         " + "  ".join(f"{label:>10s}" for label in labels)
    print(header)
    for i, row in enumerate(model.transmat_):
        row_str = "  ".join(f"{v:10.6f}" for v in row)
        print(f"  {labels[i]:>5s}  {row_str}")
    print()

    emissionprob = model.emissionprob_
    n_states, n_symbols = emissionprob.shape
    print(f"Emission probabilities: {n_states} states x {n_symbols} symbols")
    for i in range(emissionprob.shape[0]):
        label = "inaccessible" if i == 0 else "accessible"
        row = emissionprob[i]
        print(f"  State {i} ({label}):")
        print(f"    min={row.min():.6f}  max={row.max():.6f}  "
              f"mean={row.mean():.6f}  std={row.std():.6f}")

    if args.full:
        print()
        print("Full emission table:")
        for i in range(emissionprob.shape[0]):
            label = "inaccessible" if i == 0 else "accessible"
            print(f"\n  State {i} ({label}):")
            row = emissionprob[i]
            for j, val in enumerate(row):
                print(f"    Symbol {j:4d}: {val:.8f}")


# =============================================================================
# transfer subcommand
# =============================================================================


def _target_bases_for_transfer_mode(mode):
    if mode == 'daf':
        return ['C', 'G']
    return ['A']


def _trim_accessibility_context(full_context: str, trim: int) -> str:
    if trim > 0:
        return full_context[trim : len(full_context) - trim]
    return full_context


def _aggregate_accessibility_counts(
    counts,
    max_context: int,
    context_size: int,
) -> dict:
    aggregated = defaultdict(lambda: [0, 0])
    trim = max_context - context_size

    for full_context, context_counts in counts.items():
        small_context = _trim_accessibility_context(full_context, trim)
        aggregated[small_context][0] += context_counts[0]
        aggregated[small_context][1] += context_counts[1]

    return dict(aggregated)


def _accessibility_prior_row(context: str, counts) -> dict:
    acc, total = counts
    p_acc = acc / total if total > 0 else 0.5
    return {
        'context': context,
        'accessible_bp': int(acc),
        'total_bp': int(total),
        'p_accessible': p_acc,
    }


def _accessibility_priors_dataframe(aggregated: dict) -> pd.DataFrame:
    rows = [
        _accessibility_prior_row(context, counts)
        for context, counts in sorted(aggregated.items())
    ]
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values('context').reset_index(drop=True)
    return df


class AccessibilityCounter:
    """Count bases in accessible vs inaccessible regions per context."""

    def __init__(self, max_context, center_base):
        self.max_context = max_context
        self.center_base = center_base.upper()
        self.counts = defaultdict(lambda: [0, 0])
        self.total_accessible = 0
        self.total_positions = 0

    def _context_at(self, seq_upper, position):
        seq_len = len(seq_upper)
        k = self.max_context
        if position < k or position >= seq_len - k:
            return None
        if seq_upper[position] != self.center_base:
            return None
        context = seq_upper[position - k : position + k + 1]
        if len(context) != 2 * k + 1:
            return None
        if any(b not in 'ACGT' for b in context):
            return None
        return context

    def _record_accessibility(self, context, is_accessible):
        self.counts[context][1] += 1
        if is_accessible:
            self.counts[context][0] += 1
            self.total_accessible += 1
        self.total_positions += 1

    def process_read_with_footprints(self, sequence, footprint_mask, edge_trim=10):
        seq_len = len(sequence)
        seq_upper = sequence.upper()

        for i in range(edge_trim, seq_len - edge_trim):
            if i >= len(footprint_mask):
                continue

            context = self._context_at(seq_upper, i)
            if context is None:
                continue

            is_accessible = not footprint_mask[i]
            self._record_accessibility(context, is_accessible)

    def get_accessibility_priors(self, context_size=None):
        if context_size is None:
            context_size = self.max_context

        aggregated = _aggregate_accessibility_counts(
            self.counts,
            self.max_context,
            context_size,
        )
        return _accessibility_priors_dataframe(aggregated)


def _parse_footprints_from_bam_read(read):
    """Parse footprint positions from BAM read tags (ns/nl)."""
    seq_len = len(read.query_sequence) if read.query_sequence else 0
    if seq_len == 0:
        return None

    footprint_mask = np.zeros(seq_len, dtype=bool)

    ns_tag = nl_tag = None
    try:
        if read.has_tag('ns'):
            ns_tag = read.get_tag('ns')
        if read.has_tag('nl'):
            nl_tag = read.get_tag('nl')
    except KeyError:
        pass

    if ns_tag is not None and nl_tag is not None:
        starts = list(ns_tag) if hasattr(ns_tag, '__iter__') else [ns_tag]
        lengths = list(nl_tag) if hasattr(nl_tag, '__iter__') else [nl_tag]
        # molecular frame -> SEQ/query coords for the per-base scan below
        starts, lengths = flip_intervals_to_seq(starts, lengths, read)

        for start, length in zip(starts, lengths):
            end = min(start + length, seq_len)
            if 0 <= start < seq_len:
                footprint_mask[start:end] = True

        return footprint_mask

    return None


def _prepare_regression_inputs(target_rates, accessibility_priors, min_observations):
    merged = target_rates.merge(accessibility_priors, on='context', how='inner')
    merged = merged[(merged['total'] >= min_observations) &
                    (merged['total_bp'] >= min_observations)]

    diagnostics = {
        'n_contexts': len(merged),
        'total_target_obs': merged['total'].sum() if len(merged) > 0 else 0,
        'total_ref_obs': merged['total_bp'].sum() if len(merged) > 0 else 0,
    }
    return _RegressionInputs(merged=merged, diagnostics=diagnostics)


def _estimate_emission_probs(target_rates, accessibility_priors, min_observations=100):
    """Estimate P(m|acc) and P(m|inacc) using weighted linear regression."""
    inputs = _prepare_regression_inputs(
        target_rates,
        accessibility_priors,
        min_observations,
    )
    merged = inputs.merged
    diagnostics = inputs.diagnostics

    if len(merged) < 10:
        print(f"  Warning: Only {len(merged)} contexts with sufficient data")
        return _EmissionEstimate(0.5, 0.1, diagnostics)

    x = merged['p_accessible'].values
    y = merged['ratio'].values
    w = np.sqrt(merged['total'].values)

    diagnostics['x'] = x
    diagnostics['y'] = y
    diagnostics['w'] = w

    X = np.column_stack([np.ones_like(x), x])
    W = np.diag(w)

    try:
        XtW = X.T @ W
        coeffs = np.linalg.solve(XtW @ X, XtW @ y)

        p_inacc = coeffs[0]
        p_acc = coeffs[0] + coeffs[1]

        y_pred = coeffs[0] + coeffs[1] * x
        ss_res = np.sum(w * (y - y_pred) ** 2)
        ss_tot = np.sum(w * (y - np.average(y, weights=w)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        diagnostics['r_squared'] = r_squared
        diagnostics['intercept'] = coeffs[0]
        diagnostics['slope'] = coeffs[1]

        p_inacc = np.clip(p_inacc, 0.001, 0.99)
        p_acc = np.clip(p_acc, 0.001, 0.99)

        if p_acc < p_inacc:
            p_acc, p_inacc = p_inacc, p_acc
            diagnostics['swapped'] = True

        return _EmissionEstimate(float(p_acc), float(p_inacc), diagnostics)

    except np.linalg.LinAlgError:
        print("  Warning: Regression failed, using fallback")
        return _EmissionEstimate(0.5, 0.1, diagnostics)


def _passes_transfer_base_filters(read, min_mapq: int) -> bool:
    if not is_primary_mapped_alignment(read):
        return False
    if read.mapping_quality < min_mapq:
        return False
    return read.query_sequence is not None


def _passes_transfer_target_filters(read, min_mapq: int,
                                    min_read_length: int) -> bool:
    if not _passes_transfer_base_filters(read, min_mapq):
        return False
    if read.reference_end is None or read.reference_start is None:
        return False
    return (read.reference_end - read.reference_start) >= min_read_length


def _target_mod_positions_from_bam_read(read, prob_threshold: int, mode: str):
    from fiberhmm.core.bam_reader import _has_mm_ml_inputs, parse_mm_tag_query_positions
    from fiberhmm.core.tag_access import compact_ml_value, get_preferred_tag

    mm_tag = get_preferred_tag(read, 'MM', 'Mm')
    ml_raw = get_preferred_tag(read, 'ML', 'Ml')

    if mm_tag is None:
        return None
    if not _has_mm_ml_inputs(mm_tag, ml_raw):
        return set()

    ml_tag = compact_ml_value(ml_raw)
    return parse_mm_tag_query_positions(
        mm_tag, ml_tag, read.query_sequence,
        read.is_reverse, prob_threshold, mode=mode,
    )


def _transfer_progress_postfix(total_reads: int, reads_with_footprints=None) -> dict:
    postfix = {'reads': f'{total_reads:,}'}
    if reads_with_footprints is not None:
        postfix['w/footprints'] = f'{reads_with_footprints:,}'
    return postfix


def _transfer_read_limit_reached(total_reads: int, max_reads: int) -> bool:
    return max_reads > 0 and total_reads >= max_reads


def _process_reference_bam(bam_path, max_context, args):
    """Process reference BAM with footprint tags to get accessibility priors."""
    import pysam

    target_bases = ['A', 'C', 'G', 'T']
    counters = {base: AccessibilityCounter(max_context, base) for base in target_bases}
    total_reads = 0
    reads_with_footprints = 0

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        pbar = tqdm(bam.fetch(), desc="Processing reference BAM")

        for read in pbar:
            if not _passes_transfer_base_filters(read, args.min_mapq):
                continue

            total_reads += 1

            footprint_mask = _parse_footprints_from_bam_read(read)
            if footprint_mask is None:
                continue

            reads_with_footprints += 1

            for base in target_bases:
                counters[base].process_read_with_footprints(
                    read.query_sequence, footprint_mask, args.edge_trim
                )

            if total_reads % 5000 == 0:
                pbar.set_postfix(
                    _transfer_progress_postfix(total_reads, reads_with_footprints)
                )

            if _transfer_read_limit_reached(total_reads, args.max_reads):
                break

    print(f"  Processed {total_reads:,} reads, {reads_with_footprints:,} with footprint tags")
    return counters


def _process_target_bam(bam_path, mode, max_context, args):
    """Process target BAM to get modification rates per context."""
    import pysam

    from fiberhmm.probabilities.context_counter import ContextCounter
    from fiberhmm.probabilities.utils import detect_strand_and_base

    target_bases = _target_bases_for_transfer_mode(mode)

    counters = {base: ContextCounter(max_context, base) for base in target_bases}
    total_reads = 0

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        pbar = tqdm(bam.fetch(), desc="Processing target BAM")

        for read in pbar:
            if not _passes_transfer_target_filters(
                read, args.min_mapq, args.min_read_length,
            ):
                continue

            mod_positions = _target_mod_positions_from_bam_read(
                read, args.prob_threshold, mode,
            )
            if mod_positions is None:
                continue

            strand, target_base = detect_strand_and_base(
                read.query_sequence, mod_positions, mode
            )

            if target_base in counters:
                counters[target_base].process_read(
                    read.query_sequence, mod_positions, args.edge_trim
                )

            total_reads += 1
            if total_reads % 5000 == 0:
                pbar.set_postfix(_transfer_progress_postfix(total_reads))

            if _transfer_read_limit_reached(total_reads, args.max_reads):
                break

    print(f"  Processed {total_reads:,} reads")
    return counters


def _regression_stats_summary_path(plots_dir, base_name, context_size):
    return os.path.join(
        plots_dir,
        f"{base_name}_k{context_size}_regression_stats.txt",
    )


def _regression_diagnostic_plot_path(plots_dir, base_name, base, context_size):
    return os.path.join(
        plots_dir,
        f"{base_name}_{base}_k{context_size}_regression.png",
    )


def _save_regression_diagnostic_plot(
    plt,
    plots_dir,
    base_name,
    base,
    context_size,
    data,
):
    x = data['x']
    y = data['y']
    w = data['w']
    diag = data['diagnostics']
    png_path = _regression_diagnostic_plot_path(
        plots_dir, base_name, base, context_size,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        ax.scatter(x, y, alpha=0.5, s=np.clip(w, 1, 50), c='steelblue')
        if 'intercept' in diag and 'slope' in diag:
            x_line = np.array([0, 1])
            y_line = diag['intercept'] + diag['slope'] * x_line
            ax.plot(
                x_line,
                y_line,
                'r-',
                linewidth=2,
                label=(
                    f'y = {diag["intercept"]:.3f} + {diag["slope"]:.3f}x '
                    f'(R2={diag["r_squared"]:.3f})'
                ),
            )
            ax.scatter(
                [0, 1],
                [diag['intercept'], diag['intercept'] + diag['slope']],
                color='red',
                s=100,
                zorder=5,
                marker='o',
            )

        ax.set_xlabel('P(accessible|context) from fiber-seq', fontsize=12)
        ax.set_ylabel('P(methylation|context) from target', fontsize=12)
        ax.set_title(
            f'{base}-centered Emission Probability Transfer (k={context_size})',
            fontsize=14,
        )
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
    finally:
        plt.close(fig)
    return png_path


def _write_regression_stats_summary(
    regression_data,
    summary_file,
    context_size,
) -> None:
    with open(summary_file, 'w') as f:
        f.write(f"FiberHMM Transfer Learning Regression Statistics (k={context_size})\n")
        f.write("=" * 60 + "\n\n")
        f.write("Model: P(m|context) = P(m|inacc) + [P(m|acc) - P(m|inacc)] * P(acc|context)\n")
        f.write("       y = intercept + slope * x\n\n")

        for base, data in regression_data.items():
            diag = data['diagnostics']
            f.write(f"{base}-centered Contexts\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Contexts with sufficient data: {diag['n_contexts']}\n")
            f.write(f"  Target observations:           {diag['total_target_obs']:,}\n")
            f.write(f"  Reference observations:        {diag['total_ref_obs']:,}\n")

            if 'r_squared' in diag:
                f.write("\n  Regression results:\n")
                f.write(f"    R-squared: {diag['r_squared']:.4f}\n")
                f.write(f"    Intercept: {diag['intercept']:.4f} (= P(m|inaccessible))\n")
                f.write(f"    Slope:     {diag['slope']:.4f} (= P(m|acc) - P(m|inacc))\n")
                f.write("\n  Estimated emission probabilities:\n")
                f.write(f"    P(m|accessible):   {data['p_acc']:.4f}\n")
                f.write(f"    P(m|inaccessible): {data['p_inacc']:.4f}\n")
                enrichment_ratio = data['p_acc'] / max(0.001, data['p_inacc'])
                f.write(f"    Enrichment ratio:  {enrichment_ratio:.1f}x\n")
                if diag.get('swapped'):
                    f.write("    Warning: Values were swapped (acc < inacc before swap)\n")
            f.write("\n")


def _generate_regression_stats(regression_data, plots_dir, base_name, context_size):
    """Generate statistics and plots for the regression-based emission estimation."""
    summary_file = _regression_stats_summary_path(
        plots_dir,
        base_name,
        context_size,
    )
    _write_regression_stats_summary(regression_data, summary_file, context_size)
    print(f"    Summary: {summary_file}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("    Warning: matplotlib not installed. Skipping plots.")
        return

    for base, data in regression_data.items():
        if 'x' not in data or len(data['x']) == 0:
            continue

        png_path = _save_regression_diagnostic_plot(
            plt,
            plots_dir,
            base_name,
            base,
            context_size,
            data,
        )
        print(f"    Plot: {png_path}")


def _emission_stats(values):
    return _EmissionStats(
        minimum=values.min(),
        maximum=values.max(),
        mean=values.mean(),
    )


def _scale_emission_probabilities(emissionprob, target_state, scale):
    adjusted = emissionprob.copy()
    if target_state is not None:
        before_stats = _emission_stats(adjusted[target_state])
        adjusted[target_state] *= scale
        adjusted[target_state] = np.clip(adjusted[target_state], 0, 1)
        after_stats = _emission_stats(adjusted[target_state])
    else:
        before_stats = _emission_stats(adjusted)
        adjusted *= scale
        adjusted = np.clip(adjusted, 0, 1)
        after_stats = _emission_stats(adjusted)
    return _EmissionScalingResult(
        adjusted=adjusted,
        before=before_stats,
        after=after_stats,
    )


def _accessibility_priors_for_base(base: str, context_size: int,
                                   accessibility_priors_df,
                                   accessibility_counters):
    if accessibility_priors_df is not None:
        center_idx = context_size
        return accessibility_priors_df[
            accessibility_priors_df['context'].apply(
                lambda c: len(c) > center_idx and c[center_idx] == base
            )
        ].copy()

    if accessibility_counters is not None and base in accessibility_counters:
        return accessibility_counters[base].get_accessibility_priors(context_size)

    return None


def _transfer_context_index(target_rates: pd.DataFrame) -> pd.DataFrame:
    output_df = target_rates[['context']].copy()
    output_df = output_df.sort_values('context').reset_index(drop=True)
    output_df['encode'] = range(len(output_df))
    return output_df[['encode', 'context']]


def _transfer_probability_frame(
    context_index: pd.DataFrame,
    ratio: float,
) -> pd.DataFrame:
    output_df = context_index.copy()
    output_df['ratio'] = ratio
    return output_df[['encode', 'context', 'ratio']]


def _write_transfer_probability_tables(
    tables_dir: str,
    base_name: str,
    base: str,
    context_size: int,
    target_rates: pd.DataFrame,
    p_acc: float,
    p_inacc: float,
) -> str:
    output_df = _transfer_context_index(target_rates)

    acc_df = _transfer_probability_frame(output_df, p_acc)
    acc_file = _probability_table_path(
        tables_dir,
        base_name,
        "accessible",
        base,
        context_size,
    )
    acc_df.to_csv(acc_file, sep='\t', index=False)

    inacc_df = _transfer_probability_frame(output_df, p_inacc)
    inacc_file = _probability_table_path(
        tables_dir,
        base_name,
        "inaccessible",
        base,
        context_size,
    )
    inacc_df.to_csv(inacc_file, sep='\t', index=False)

    combined = pd.DataFrame({
        'encode': range(len(output_df)),
        'context': output_df['context'].values,
        'accessible_prob': p_acc,
        'inaccessible_prob': p_inacc
    })
    combined_file = _combined_probability_table_path(
        tables_dir,
        base_name,
        base,
        context_size,
    )
    combined.to_csv(combined_file, sep='\t', index=False)
    return combined_file


def _save_accessibility_priors(
    tables_dir: str,
    base_name: str,
    max_context: int,
    accessibility_counters,
) -> list[str]:
    written = []
    for base in ['A', 'C', 'G', 'T']:
        if base not in accessibility_counters:
            continue
        priors = accessibility_counters[base].get_accessibility_priors(max_context)
        priors_file = os.path.join(
            tables_dir,
            f"{base_name}_accessibility_priors_{base}_k{max_context}.tsv",
        )
        priors.to_csv(priors_file, sep='\t', index=False)
        print(f"  {priors_file}")
        written.append(priors_file)
    return written


def _print_transfer_header(args, output_dir: str, target_bases: list[str]) -> None:
    print("=" * 60)
    print("FiberHMM Transfer Learning - Emission Probability Estimator")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Target data: {args.target}")
    print(f"Target mode: {args.mode}")
    print(f"Context sizes: {args.context_sizes}")
    print(f"Target bases: {', '.join(target_bases)}")


def _print_transfer_accessibility_summary(target_bases: list[str],
                                          accessibility_counters) -> None:
    for base in target_bases:
        if base not in accessibility_counters:
            continue
        counter = accessibility_counters[base]
        acc_rate = counter.total_accessible / max(1, counter.total_positions)
        print(
            f"  {base}: {counter.total_positions:,} positions, "
            f"{acc_rate:.1%} accessible"
        )


def _load_transfer_accessibility_inputs(args, max_context: int,
                                        target_bases: list[str]):
    if args.accessibility_priors:
        print(f"\nLoading accessibility priors from: {args.accessibility_priors}")
        accessibility_priors_df = pd.read_csv(args.accessibility_priors, sep='\t')
        print(f"  Loaded {len(accessibility_priors_df)} contexts")
        return _TransferAccessibilityInputs(counters=None, priors=accessibility_priors_df)

    if args.reference_bam:
        print(f"\nComputing accessibility priors from: {args.reference_bam}")
        accessibility_counters = _process_reference_bam(
            args.reference_bam,
            max_context,
            args,
        )
        _print_transfer_accessibility_summary(target_bases, accessibility_counters)
        return _TransferAccessibilityInputs(counters=accessibility_counters, priors=None)

    print("\nError: Must provide one of:")
    print("  --reference-bam (fiber-seq BAM with footprint tags ns/nl)")
    print("  --accessibility-priors (pre-computed TSV)")
    sys.exit(1)


def _transfer_regression_record(diag, p_acc: float, p_inacc: float) -> dict:
    return {
        'x': diag.get('x', np.array([])),
        'y': diag.get('y', np.array([])),
        'w': diag.get('w', np.array([])),
        'diagnostics': diag,
        'p_acc': p_acc,
        'p_inacc': p_inacc,
    }


def _print_transfer_emission_summary(base: str, diag: dict, p_acc: float,
                                     p_inacc: float) -> None:
    print(f"\n  {base}-centered:")
    print(f"    Contexts with sufficient data: {diag['n_contexts']}")
    if 'r_squared' in diag:
        print(f"    Regression R-squared: {diag['r_squared']:.3f}")
    print(f"    Estimated P(m|accessible):   {p_acc:.4f}")
    print(f"    Estimated P(m|inaccessible): {p_inacc:.4f}")
    print(f"    Enrichment ratio: {p_acc/max(0.001, p_inacc):.1f}x")


def _estimate_transfer_context_size(
    context_size: int,
    args,
    tables_dir: str,
    base_name: str,
    target_bases: list[str],
    target_counters,
    accessibility_priors_df,
    accessibility_counters,
) -> dict:
    print(f"\nContext size k={context_size} ({2*context_size+1}-mer):")
    regression_data = {}

    for base in target_bases:
        target_rates = target_counters[base].get_probabilities(context_size)
        priors = _accessibility_priors_for_base(
            base,
            context_size,
            accessibility_priors_df,
            accessibility_counters,
        )
        if priors is None:
            print(f"  Warning: No accessibility priors for {base}")
            continue

        estimate = _estimate_emission_probs(
            target_rates,
            priors,
            args.min_observations,
        )

        regression_data[base] = _transfer_regression_record(
            estimate.diagnostics,
            estimate.p_acc,
            estimate.p_inacc,
        )
        _print_transfer_emission_summary(
            base,
            estimate.diagnostics,
            estimate.p_acc,
            estimate.p_inacc,
        )

        combined_file = _write_transfer_probability_tables(
            tables_dir,
            base_name,
            base,
            context_size,
            target_rates,
            estimate.p_acc,
            estimate.p_inacc,
        )
        print(f"    Output: {combined_file}")

    return regression_data


def _estimate_transfer_emissions(
    args,
    tables_dir: str,
    base_name: str,
    target_bases: list[str],
    target_counters,
    accessibility_priors_df,
    accessibility_counters,
) -> dict:
    print("\nEstimating emission probabilities...")
    return {
        context_size: _estimate_transfer_context_size(
            context_size,
            args,
            tables_dir,
            base_name,
            target_bases,
            target_counters,
            accessibility_priors_df,
            accessibility_counters,
        )
        for context_size in args.context_sizes
    }


def _maybe_save_transfer_accessibility_priors(
    tables_dir: str,
    base_name: str,
    max_context: int,
    accessibility_counters,
) -> None:
    if accessibility_counters is None:
        return

    print("\nSaving accessibility priors for reuse:")
    _save_accessibility_priors(tables_dir, base_name, max_context, accessibility_counters)


def _maybe_generate_transfer_stats(
    args,
    all_regression_data: dict,
    plots_dir: str,
    base_name: str,
) -> None:
    if not args.stats:
        return

    print("\nGenerating statistics and plots:")
    for context_size in args.context_sizes:
        if context_size in all_regression_data and all_regression_data[context_size]:
            print(f"\n  k={context_size} ({2*context_size+1}-mer):")
            _generate_regression_stats(
                all_regression_data[context_size],
                plots_dir,
                base_name,
                context_size,
            )


def cmd_transfer(args):
    """Transfer emission probs between modalities."""
    max_context = max(args.context_sizes)

    output_dir = args.output
    _, tables_dir_path, plots_dir_path = setup_output_dirs(output_dir)
    tables_dir = str(tables_dir_path)
    plots_dir = str(plots_dir_path)
    base_name = get_base_name(output_dir, default="transfer")

    target_bases = _target_bases_for_transfer_mode(args.mode)
    _print_transfer_header(args, output_dir, target_bases)
    accessibility_inputs = _load_transfer_accessibility_inputs(
        args, max_context, target_bases,
    )

    # Step 2: Get target modification rates
    print("\nProcessing target BAM to get modification rates...")
    target_counters = _process_target_bam(args.target, args.mode, max_context, args)

    # Step 3: Estimate emission probs for each context size
    all_regression_data = _estimate_transfer_emissions(
        args,
        tables_dir,
        base_name,
        target_bases,
        target_counters,
        accessibility_inputs.priors,
        accessibility_inputs.counters,
    )

    # Save accessibility priors if computed
    _maybe_save_transfer_accessibility_priors(
        tables_dir,
        base_name,
        max_context,
        accessibility_inputs.counters,
    )

    # Generate stats if requested
    _maybe_generate_transfer_stats(args, all_regression_data, plots_dir, base_name)

    print("\nDone!")
    print("Note: This estimates GLOBAL emission probs (same for all contexts).")
    print("For context-specific probs, use generate_probs.py with proper controls.")


# =============================================================================
# adjust subcommand
# =============================================================================

def cmd_adjust(args):
    """Adjust emission probabilities in a model."""
    filepath = args.model

    _exit_if_missing_or_non_file(
        filepath,
        missing_prefix="File not found",
        non_file_prefix="Path is not a file",
    )

    model, context_size, mode = load_model_with_metadata(filepath, normalize=False)

    print(f"Loading model: {filepath}")
    print(f"  Mode: {mode}, k={context_size}, states={model.n_states}")

    state_map = {'inaccessible': 0, 'accessible': 1, 'both': None}
    target_state = state_map[args.state]
    scale = args.scale

    print("\nAdjusting emission probabilities:")
    print(f"  Target: {args.state} (state {'all' if target_state is None else target_state})")
    print(f"  Scale factor: {scale}")

    scaling_result = _scale_emission_probabilities(
        model.emissionprob_,
        target_state,
        scale,
    )
    model.emissionprob_ = scaling_result.adjusted

    print(
        f"  Before: min={scaling_result.before.minimum:.6f}, "
        f"max={scaling_result.before.maximum:.6f}, "
        f"mean={scaling_result.before.mean:.6f}",
    )
    print(
        f"  After:  min={scaling_result.after.minimum:.6f}, "
        f"max={scaling_result.after.maximum:.6f}, "
        f"mean={scaling_result.after.mean:.6f}",
    )

    output_path = args.output
    save_model(model, output_path, context_size=context_size, mode=mode)
    print(f"\nSaved adjusted model to: {output_path}")


# =============================================================================
# main: argument parsing with subcommands
# =============================================================================

def _utils_parser_epilog() -> str:
    return """
Subcommands:
  convert   Convert pickle/NPZ models to JSON format
  inspect   Print model metadata, parameters, and emission statistics
  transfer  Transfer emission probs between modalities (e.g., fiber-seq to DAF-seq)
  adjust    Apply scaling to emission probabilities

Examples:
  fiberhmm-utils convert old_model.pickle new_model.json
  fiberhmm-utils inspect model.json
  fiberhmm-utils inspect model.json --full
  fiberhmm-utils transfer --target daf.bam --reference-bam fiber.bam -o probs/
  fiberhmm-utils adjust model.json --state accessible --scale 1.1 -o adjusted.json
        """


def _add_utils_convert_parser(subparsers) -> None:
    p_convert = subparsers.add_parser(
        'convert',
        help='Convert pickle/NPZ model to JSON',
        description='Convert legacy pickle or NPZ model files to the portable JSON format.'
    )
    p_convert.add_argument('input', help='Input model file (.pickle, .pkl, or .npz)')
    p_convert.add_argument('output', help='Output JSON file')


def _add_utils_inspect_parser(subparsers) -> None:
    p_inspect = subparsers.add_parser(
        'inspect',
        help='Inspect a model file',
        description='Print model metadata, parameters, and emission probability statistics.'
    )
    p_inspect.add_argument('model', help='Model file to inspect (.json, .npz, .pickle)')
    p_inspect.add_argument('--full', action='store_true',
                          help='Print full emission probability table')


def _add_utils_transfer_parser(subparsers) -> None:
    p_transfer = subparsers.add_parser(
        'transfer',
        help='Transfer emission probs between modalities',
        description='Estimate emission probabilities for a new sequencing technology '
                    'using accessibility priors from a matched cell type.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p_transfer.add_argument('--target', '-t', required=True,
                           help='Target BAM file (e.g., DAF-seq)')
    p_transfer.add_argument('-o', '--output', required=True,
                           help='Output directory')
    p_transfer.add_argument('--mode', choices=MODE_CHOICES,
                           default='daf', help='Analysis mode for target data')
    p_transfer.add_argument('--reference-bam', '-rb',
                           help='Reference BAM with footprint tags (ns/nl)')
    p_transfer.add_argument('--accessibility-priors', '-ap',
                           help='Pre-computed P(accessible|context) TSV')
    p_transfer.add_argument('-k', '--context-sizes', type=int, nargs='+',
                           default=[3, 4, 5, 6], help='Context size(s)')
    p_transfer.add_argument('-n', '--max-reads', type=int, default=100000,
                           help='Max reads to process (0 = all)')
    p_transfer.add_argument('-q', '--min-mapq', type=int, default=20,
                           help='Min mapping quality')
    p_transfer.add_argument('-p', '--prob-threshold', type=int, default=128,
                           help='Min ML probability for modification call')
    p_transfer.add_argument('--min-read-length', type=int, default=1000,
                           help='Min aligned read length')
    p_transfer.add_argument('-e', '--edge-trim', type=int, default=10,
                           help='Bases to exclude at read edges')
    p_transfer.add_argument('--min-observations', type=int, default=100,
                           help='Min observations per context for regression')
    p_transfer.add_argument('--stats', action='store_true',
                           help='Generate diagnostic plots')


def _add_utils_adjust_parser(subparsers) -> None:
    p_adjust = subparsers.add_parser(
        'adjust',
        help='Adjust emission probabilities in a model',
        description='Apply a scaling factor to emission probabilities, '
                    'clamped to [0, 1].'
    )
    p_adjust.add_argument('model', help='Input model file (.json)')
    p_adjust.add_argument('--state', required=True,
                         choices=['accessible', 'inaccessible', 'both'],
                         help='Which state(s) to adjust')
    p_adjust.add_argument('--scale', type=float, required=True,
                         help='Multiplier for emission probabilities')
    p_adjust.add_argument('-o', '--output', required=True,
                         help='Output model file (.json)')


def _build_utils_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='fiberhmm-utils',
        description='FiberHMM utilities: model conversion, inspection, and probability transfer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_utils_parser_epilog(),
    )
    subparsers = parser.add_subparsers(dest='command')

    _add_utils_convert_parser(subparsers)
    _add_utils_inspect_parser(subparsers)
    _add_utils_transfer_parser(subparsers)
    _add_utils_adjust_parser(subparsers)
    return parser


def _dispatch_utils_command(args, parser) -> None:
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        'convert': cmd_convert,
        'inspect': cmd_inspect,
        'transfer': cmd_transfer,
        'adjust': cmd_adjust,
    }
    dispatch[args.command](args)


def main(argv=None):
    parser = _build_utils_parser()
    args = parser.parse_args(argv)
    _dispatch_utils_command(args, parser)


if __name__ == '__main__':
    main()
