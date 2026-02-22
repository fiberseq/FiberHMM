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
from pathlib import Path

import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from fiberhmm.core.model_io import load_model, save_model, load_model_with_metadata


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
    data = np.load(filepath, allow_pickle=True)
    return {
        'startprob': data['startprob'],
        'transmat': data['transmat'],
        'emissionprob': data['emissionprob'],
        'n_states': int(data.get('n_states', 2)),
        'context_size': int(data.get('context_size', 3)),
        'mode': str(data.get('mode', 'pacbio-fiber')),
    }


def cmd_convert(args):
    """Convert pickle/NPZ model to JSON."""
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    suffix = input_path.suffix.lower()
    print(f"Loading model from {input_path}...")

    try:
        if suffix == '.npz':
            data = _load_npz_model_raw(input_path)
        elif suffix in ['.pickle', '.pkl']:
            data = _load_pickle_model_raw(input_path)
        else:
            try:
                data = _load_pickle_model_raw(input_path)
            except Exception:
                data = _load_npz_model_raw(input_path)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    startprob = np.array(data['startprob'])
    transmat = np.array(data['transmat'])
    emissionprob = np.array(data['emissionprob'])

    print(f"  States: {data['n_states']}")
    print(f"  Context size: k={data['context_size']} ({2*data['context_size']+1}-mer)")
    print(f"  Mode: {data['mode']}")
    print(f"  Start probs: {startprob}")
    print(f"  Transition matrix shape: {transmat.shape}")
    print(f"  Emission prob shape: {emissionprob.shape}")

    # Save as JSON
    json_data = {
        'model_type': 'FiberHMM',
        'version': '2.0',
        'n_states': int(data['n_states']),
        'startprob': startprob.tolist(),
        'transmat': transmat.tolist(),
        'emissionprob': emissionprob.tolist(),
        'context_size': int(data['context_size']),
        'mode': str(data['mode']),
    }

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

    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

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
    header = "         " + "  ".join(f"{l:>10s}" for l in labels)
    print(header)
    for i, row in enumerate(model.transmat_):
        row_str = "  ".join(f"{v:10.6f}" for v in row)
        print(f"  {labels[i]:>5s}  {row_str}")
    print()

    emissionprob = model.emissionprob_
    print(f"Emission probabilities: {emissionprob.shape[0]} states x {emissionprob.shape[1]} symbols")
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

class AccessibilityCounter:
    """Count bases in accessible vs inaccessible regions per context."""

    def __init__(self, max_context, center_base):
        self.max_context = max_context
        self.center_base = center_base.upper()
        self.counts = defaultdict(lambda: [0, 0])
        self.total_accessible = 0
        self.total_positions = 0

    def process_read_with_footprints(self, sequence, footprint_mask, edge_trim=10):
        seq_len = len(sequence)
        seq_upper = sequence.upper()
        k = self.max_context

        for i in range(edge_trim, seq_len - edge_trim):
            if seq_upper[i] != self.center_base:
                continue
            if i < k or i >= seq_len - k:
                continue
            if i >= len(footprint_mask):
                continue

            context = seq_upper[i - k : i + k + 1]
            if len(context) != 2 * k + 1:
                continue
            if any(b not in 'ACGT' for b in context):
                continue

            is_accessible = not footprint_mask[i]
            self.counts[context][1] += 1
            if is_accessible:
                self.counts[context][0] += 1
                self.total_accessible += 1
            self.total_positions += 1

    def get_accessibility_priors(self, context_size=None):
        if context_size is None:
            context_size = self.max_context

        aggregated = defaultdict(lambda: [0, 0])
        trim = self.max_context - context_size

        for full_context, counts in self.counts.items():
            if trim > 0:
                small_context = full_context[trim : len(full_context) - trim]
            else:
                small_context = full_context
            aggregated[small_context][0] += counts[0]
            aggregated[small_context][1] += counts[1]

        rows = []
        for context, (acc, total) in sorted(aggregated.items()):
            p_acc = acc / total if total > 0 else 0.5
            rows.append({
                'context': context,
                'accessible_bp': int(acc),
                'total_bp': int(total),
                'p_accessible': p_acc
            })

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values('context').reset_index(drop=True)
        return df


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

        for start, length in zip(starts, lengths):
            end = min(start + length, seq_len)
            if 0 <= start < seq_len:
                footprint_mask[start:end] = True

        return footprint_mask

    return None


def _estimate_emission_probs(target_rates, accessibility_priors, min_observations=100):
    """Estimate P(m|acc) and P(m|inacc) using weighted linear regression."""
    merged = target_rates.merge(accessibility_priors, on='context', how='inner')
    merged = merged[(merged['total'] >= min_observations) &
                    (merged['total_bp'] >= min_observations)]

    diagnostics = {
        'n_contexts': len(merged),
        'total_target_obs': merged['total'].sum() if len(merged) > 0 else 0,
        'total_ref_obs': merged['total_bp'].sum() if len(merged) > 0 else 0,
    }

    if len(merged) < 10:
        print(f"  Warning: Only {len(merged)} contexts with sufficient data")
        return 0.5, 0.1, diagnostics

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

        return float(p_acc), float(p_inacc), diagnostics

    except np.linalg.LinAlgError:
        print("  Warning: Regression failed, using fallback")
        return 0.5, 0.1, diagnostics


def _process_reference_bam(bam_path, max_context, args):
    """Process reference BAM with footprint tags to get accessibility priors."""
    import pysam

    target_bases = ['A', 'C', 'G', 'T']
    counters = {base: AccessibilityCounter(max_context, base) for base in target_bases}
    total_reads = 0
    reads_with_footprints = 0

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        pbar = tqdm(bam.fetch(), desc="Processing reference BAM")

        for read in pbar:
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.mapping_quality < args.min_mapq:
                continue
            if read.query_sequence is None:
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
                pbar.set_postfix({
                    'reads': f'{total_reads:,}',
                    'w/footprints': f'{reads_with_footprints:,}'
                })

            if args.max_reads > 0 and total_reads >= args.max_reads:
                break

    print(f"  Processed {total_reads:,} reads, {reads_with_footprints:,} with footprint tags")
    return counters


def _process_target_bam(bam_path, mode, max_context, args):
    """Process target BAM to get modification rates per context."""
    import pysam
    from fiberhmm.core.bam_reader import parse_mm_tag_query_positions
    from fiberhmm.probabilities.context_counter import ContextCounter
    from fiberhmm.probabilities.utils import detect_strand_and_base

    if mode in ('pacbio-fiber', 'nanopore-fiber'):
        target_bases = ['A']
    elif mode == 'daf':
        target_bases = ['C', 'G']
    else:
        target_bases = ['A']

    counters = {base: ContextCounter(max_context, base) for base in target_bases}
    total_reads = 0

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        pbar = tqdm(bam.fetch(), desc="Processing target BAM")

        for read in pbar:
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.mapping_quality < args.min_mapq:
                continue
            if read.query_sequence is None:
                continue
            if read.reference_end is None or read.reference_start is None:
                continue
            if read.reference_end - read.reference_start < args.min_read_length:
                continue

            mm_tag = ml_tag = None
            try:
                if read.has_tag('MM'):
                    mm_tag = read.get_tag('MM')
                elif read.has_tag('Mm'):
                    mm_tag = read.get_tag('Mm')
                if read.has_tag('ML'):
                    ml_tag = list(read.get_tag('ML'))
                elif read.has_tag('Ml'):
                    ml_tag = list(read.get_tag('Ml'))
            except KeyError:
                continue

            if mm_tag is None:
                continue

            mod_positions = parse_mm_tag_query_positions(
                mm_tag, ml_tag, read.query_sequence,
                read.is_reverse, args.prob_threshold, mode=mode
            )

            strand, target_base = detect_strand_and_base(
                read.query_sequence, mod_positions, mode
            )

            if target_base in counters:
                counters[target_base].process_read(
                    read.query_sequence, mod_positions, args.edge_trim
                )

            total_reads += 1
            if total_reads % 5000 == 0:
                pbar.set_postfix({'reads': f'{total_reads:,}'})

            if args.max_reads > 0 and total_reads >= args.max_reads:
                break

    print(f"  Processed {total_reads:,} reads")
    return counters


def _generate_regression_stats(regression_data, plots_dir, base_name, context_size):
    """Generate statistics and plots for the regression-based emission estimation."""
    summary_file = os.path.join(plots_dir, f"{base_name}_k{context_size}_regression_stats.txt")
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
                f.write(f"\n  Regression results:\n")
                f.write(f"    R-squared: {diag['r_squared']:.4f}\n")
                f.write(f"    Intercept: {diag['intercept']:.4f} (= P(m|inaccessible))\n")
                f.write(f"    Slope:     {diag['slope']:.4f} (= P(m|acc) - P(m|inacc))\n")
                f.write(f"\n  Estimated emission probabilities:\n")
                f.write(f"    P(m|accessible):   {data['p_acc']:.4f}\n")
                f.write(f"    P(m|inaccessible): {data['p_inacc']:.4f}\n")
                f.write(f"    Enrichment ratio:  {data['p_acc']/max(0.001, data['p_inacc']):.1f}x\n")
                if diag.get('swapped'):
                    f.write(f"    Warning: Values were swapped (acc < inacc before swap)\n")
            f.write("\n")

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

        x = data['x']
        y = data['y']
        w = data['w']
        diag = data['diagnostics']

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x, y, alpha=0.5, s=np.clip(w, 1, 50), c='steelblue')
        if 'intercept' in diag and 'slope' in diag:
            x_line = np.array([0, 1])
            y_line = diag['intercept'] + diag['slope'] * x_line
            ax.plot(x_line, y_line, 'r-', linewidth=2,
                   label=f'y = {diag["intercept"]:.3f} + {diag["slope"]:.3f}x '
                         f'(R2={diag["r_squared"]:.3f})')
            ax.scatter([0, 1], [diag['intercept'], diag['intercept'] + diag['slope']],
                      color='red', s=100, zorder=5, marker='o')

        ax.set_xlabel('P(accessible|context) from fiber-seq', fontsize=12)
        ax.set_ylabel('P(methylation|context) from target', fontsize=12)
        ax.set_title(f'{base}-centered Emission Probability Transfer (k={context_size})',
                    fontsize=14)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        png_path = os.path.join(plots_dir,
                               f"{base_name}_{base}_k{context_size}_regression.png")
        plt.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"    Plot: {png_path}")


def cmd_transfer(args):
    """Transfer emission probs between modalities."""
    max_context = max(args.context_sizes)

    output_dir = args.output
    tables_dir = os.path.join(output_dir, "tables")
    plots_dir = os.path.join(output_dir, "plots")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    base_name = os.path.basename(output_dir.rstrip('/'))
    if not base_name:
        base_name = "transfer"

    print("=" * 60)
    print("FiberHMM Transfer Learning - Emission Probability Estimator")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Target data: {args.target}")
    print(f"Target mode: {args.mode}")
    print(f"Context sizes: {args.context_sizes}")

    if args.mode in ('pacbio-fiber', 'nanopore-fiber'):
        target_bases = ['A']
    elif args.mode == 'daf':
        target_bases = ['C', 'G']
    else:
        target_bases = ['A']

    print(f"Target bases: {', '.join(target_bases)}")

    # Step 1: Get accessibility priors
    accessibility_counters = None
    accessibility_priors_df = None

    if args.accessibility_priors:
        print(f"\nLoading accessibility priors from: {args.accessibility_priors}")
        accessibility_priors_df = pd.read_csv(args.accessibility_priors, sep='\t')
        print(f"  Loaded {len(accessibility_priors_df)} contexts")
    elif args.reference_bam:
        print(f"\nComputing accessibility priors from: {args.reference_bam}")
        accessibility_counters = _process_reference_bam(
            args.reference_bam, max_context, args
        )
        for base in target_bases:
            if base in accessibility_counters:
                counter = accessibility_counters[base]
                acc_rate = counter.total_accessible / max(1, counter.total_positions)
                print(f"  {base}: {counter.total_positions:,} positions, "
                      f"{acc_rate:.1%} accessible")
    else:
        print("\nError: Must provide one of:")
        print("  --reference-bam (fiber-seq BAM with footprint tags ns/nl)")
        print("  --accessibility-priors (pre-computed TSV)")
        sys.exit(1)

    # Step 2: Get target modification rates
    print(f"\nProcessing target BAM to get modification rates...")
    target_counters = _process_target_bam(args.target, args.mode, max_context, args)

    # Step 3: Estimate emission probs for each context size
    print(f"\nEstimating emission probabilities...")

    all_regression_data = {}

    for k in args.context_sizes:
        print(f"\nContext size k={k} ({2*k+1}-mer):")
        regression_data_k = {}

        for base in target_bases:
            target_rates = target_counters[base].get_probabilities(k)

            if accessibility_priors_df is not None:
                center_idx = k
                priors = accessibility_priors_df[
                    accessibility_priors_df['context'].apply(
                        lambda c: len(c) > center_idx and c[center_idx] == base
                    )
                ].copy()
            elif accessibility_counters is not None and base in accessibility_counters:
                priors = accessibility_counters[base].get_accessibility_priors(k)
            else:
                print(f"  Warning: No accessibility priors for {base}")
                continue

            p_acc, p_inacc, diag = _estimate_emission_probs(
                target_rates, priors, args.min_observations
            )

            regression_data_k[base] = {
                'x': diag.get('x', np.array([])),
                'y': diag.get('y', np.array([])),
                'w': diag.get('w', np.array([])),
                'diagnostics': diag,
                'p_acc': p_acc,
                'p_inacc': p_inacc
            }

            print(f"\n  {base}-centered:")
            print(f"    Contexts with sufficient data: {diag['n_contexts']}")
            if 'r_squared' in diag:
                print(f"    Regression R-squared: {diag['r_squared']:.3f}")
            print(f"    Estimated P(m|accessible):   {p_acc:.4f}")
            print(f"    Estimated P(m|inaccessible): {p_inacc:.4f}")
            print(f"    Enrichment ratio: {p_acc/max(0.001, p_inacc):.1f}x")

            # Create output files
            output_df = target_rates[['context']].copy()
            output_df = output_df.sort_values('context').reset_index(drop=True)
            output_df['encode'] = range(len(output_df))

            acc_df = output_df.copy()
            acc_df['ratio'] = p_acc
            acc_file = os.path.join(tables_dir, f"{base_name}_accessible_{base}_k{k}.tsv")
            acc_df[['encode', 'context', 'ratio']].to_csv(acc_file, sep='\t', index=False)

            inacc_df = output_df.copy()
            inacc_df['ratio'] = p_inacc
            inacc_file = os.path.join(tables_dir,
                                      f"{base_name}_inaccessible_{base}_k{k}.tsv")
            inacc_df[['encode', 'context', 'ratio']].to_csv(inacc_file, sep='\t', index=False)

            combined = pd.DataFrame({
                'encode': range(len(output_df)),
                'context': output_df['context'].values,
                'accessible_prob': p_acc,
                'inaccessible_prob': p_inacc
            })
            combined_file = os.path.join(tables_dir, f"{base_name}_{base}_k{k}_probs.tsv")
            combined.to_csv(combined_file, sep='\t', index=False)
            print(f"    Output: {combined_file}")

        all_regression_data[k] = regression_data_k

    # Save accessibility priors if computed
    if accessibility_counters is not None:
        print(f"\nSaving accessibility priors for reuse:")
        for base in ['A', 'C', 'G', 'T']:
            if base in accessibility_counters:
                priors = accessibility_counters[base].get_accessibility_priors(max_context)
                priors_file = os.path.join(
                    tables_dir,
                    f"{base_name}_accessibility_priors_{base}_k{max_context}.tsv"
                )
                priors.to_csv(priors_file, sep='\t', index=False)
                print(f"  {priors_file}")

    # Generate stats if requested
    if args.stats:
        print(f"\nGenerating statistics and plots:")
        for k in args.context_sizes:
            if k in all_regression_data and all_regression_data[k]:
                print(f"\n  k={k} ({2*k+1}-mer):")
                _generate_regression_stats(
                    all_regression_data[k], plots_dir, base_name, k
                )

    print(f"\nDone!")
    print("Note: This estimates GLOBAL emission probs (same for all contexts).")
    print("For context-specific probs, use generate_probs.py with proper controls.")


# =============================================================================
# adjust subcommand
# =============================================================================

def cmd_adjust(args):
    """Adjust emission probabilities in a model."""
    filepath = args.model

    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    model, context_size, mode = load_model_with_metadata(filepath, normalize=False)

    print(f"Loading model: {filepath}")
    print(f"  Mode: {mode}, k={context_size}, states={model.n_states}")

    state_map = {'inaccessible': 0, 'accessible': 1, 'both': None}
    target_state = state_map[args.state]
    scale = args.scale

    print(f"\nAdjusting emission probabilities:")
    print(f"  Target: {args.state} (state {'all' if target_state is None else target_state})")
    print(f"  Scale factor: {scale}")

    emissionprob = model.emissionprob_.copy()

    if target_state is not None:
        before_stats = (emissionprob[target_state].min(),
                       emissionprob[target_state].max(),
                       emissionprob[target_state].mean())
        emissionprob[target_state] *= scale
        emissionprob[target_state] = np.clip(emissionprob[target_state], 0, 1)
        after_stats = (emissionprob[target_state].min(),
                      emissionprob[target_state].max(),
                      emissionprob[target_state].mean())
    else:
        before_stats = (emissionprob.min(), emissionprob.max(), emissionprob.mean())
        emissionprob *= scale
        emissionprob = np.clip(emissionprob, 0, 1)
        after_stats = (emissionprob.min(), emissionprob.max(), emissionprob.mean())

    model.emissionprob_ = emissionprob

    print(f"  Before: min={before_stats[0]:.6f}, max={before_stats[1]:.6f}, "
          f"mean={before_stats[2]:.6f}")
    print(f"  After:  min={after_stats[0]:.6f}, max={after_stats[1]:.6f}, "
          f"mean={after_stats[2]:.6f}")

    output_path = args.output
    save_model(model, output_path, context_size=context_size, mode=mode)
    print(f"\nSaved adjusted model to: {output_path}")


# =============================================================================
# main: argument parsing with subcommands
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog='fiberhmm-utils',
        description='FiberHMM utilities: model conversion, inspection, and probability transfer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
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
    )
    subparsers = parser.add_subparsers(dest='command')

    # --- convert ---
    p_convert = subparsers.add_parser(
        'convert',
        help='Convert pickle/NPZ model to JSON',
        description='Convert legacy pickle or NPZ model files to the portable JSON format.'
    )
    p_convert.add_argument('input', help='Input model file (.pickle, .pkl, or .npz)')
    p_convert.add_argument('output', help='Output JSON file')

    # --- inspect ---
    p_inspect = subparsers.add_parser(
        'inspect',
        help='Inspect a model file',
        description='Print model metadata, parameters, and emission probability statistics.'
    )
    p_inspect.add_argument('model', help='Model file to inspect (.json, .npz, .pickle)')
    p_inspect.add_argument('--full', action='store_true',
                          help='Print full emission probability table')

    # --- transfer ---
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
    p_transfer.add_argument('--mode', choices=['pacbio-fiber', 'nanopore-fiber', 'daf'],
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

    # --- adjust ---
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

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == 'convert':
        cmd_convert(args)
    elif args.command == 'inspect':
        cmd_inspect(args)
    elif args.command == 'transfer':
        cmd_transfer(args)
    elif args.command == 'adjust':
        cmd_adjust(args)


if __name__ == '__main__':
    main()
