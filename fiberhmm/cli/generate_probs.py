#!/usr/bin/env python3
"""
FiberHMM generate_probs.py
Generate emission probability tables from control BAM files.

Requires TWO types of control samples:
1. Accessible/naked DNA: Dechromatinized or naked DNA treated with MTase
   - Estimates P(methylation | accessible) - should be HIGH

2. Inaccessible/untreated: Native chromatin (untreated) or nucleosome-bound DNA
   - Estimates P(methylation | inaccessible) - should be LOW (background rate)

These two distributions define the HMM emission probabilities for the
accessible (State 1) and inaccessible/footprint (State 0) states.

Features:
- Direct BAM input (no BED preprocessing)
- Flexible context sizes (default 3 = hexamer/7-mer, up to 10 = 21-mer)
- Stores all context sizes in single file for easy querying
- Supports pacbio-fiber, nanopore-fiber, and daf modes
"""

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, MutableMapping, Tuple

import numpy as np
import pandas as pd
import pysam
from tqdm import tqdm

# Package imports
from fiberhmm.core.bam_reader import parse_mm_tag_query_positions
from fiberhmm.core.tag_access import get_preferred_tag
from fiberhmm.probabilities.context_counter import ContextCounter
from fiberhmm.probabilities.output_paths import (
    combined_probability_table_path as _combined_probability_table_path,
    probability_counter_path as _probability_counter_path,
    probability_table_path as _probability_table_path,
)
from fiberhmm.probabilities.stats import generate_probability_stats
from fiberhmm.probabilities.utils import (
    detect_strand_and_base,
    get_base_name,
    setup_output_dirs,
)


FILTER_STAT_KEYS = (
    'scanned',
    'unmapped',
    'secondary',
    'supplementary',
    'low_mapq',
    'no_sequence',
    'short_read',
    'no_mm_tag',
    'no_ml_tag',
    'processed',
)
SAMPLE_TYPES = ('accessible', 'inaccessible')
PROBABILITY_TSV_COLUMNS = ('encode', 'context', 'hit', 'nohit', 'ratio')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate emission probability tables from control BAM files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Two required inputs for the two states
    parser.add_argument('--accessible', '-a', required=True, nargs='+',
                        help='BAM file(s) from accessible/naked DNA (dechromatinized, MTase-treated)')
    parser.add_argument('--inaccessible', '-u', required=True, nargs='+',
                        help='BAM file(s) from inaccessible/untreated samples (native chromatin)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output file prefix (will create _accessible.tsv and _inaccessible.tsv)')

    # Context settings
    parser.add_argument('-k', '--context-sizes', type=int, nargs='+', default=[3, 4, 5, 6],
                        help='Context size(s) to compute (bases on each side: 3=7mer, 6=13mer). '
                             'Single value or list. Default: 3 4 5 6')
    parser.add_argument('--mode', choices=['pacbio-fiber', 'nanopore-fiber', 'daf'], default='pacbio-fiber',
                        help='Analysis mode: pacbio-fiber (PacBio), nanopore-fiber (Nanopore), daf (DAF-seq)')

    # Sampling
    parser.add_argument('-n', '--max-reads', type=int, default=100000,
                        help='Maximum reads to process per sample type (0 = all)')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='Random seed for sampling')

    # Filtering
    parser.add_argument('-q', '--min-mapq', type=int, default=20,
                        help='Minimum mapping quality')
    parser.add_argument('-p', '--prob-threshold', type=int, default=128,
                        help='Minimum ML probability for modification call (0-255)')
    parser.add_argument('--min-read-length', type=int, default=1000,
                        help='Minimum aligned read length')
    parser.add_argument('-e', '--edge-trim', type=int, default=10,
                        help='Bases to exclude at read edges')

    # Output
    parser.add_argument('--save-interval', type=int, default=10000,
                        help='Save intermediate results every N reads')
    parser.add_argument('--stats', action='store_true',
                        help='Generate summary statistics and QC plots')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed filter statistics per BAM file')

    return parser.parse_args()


def _new_filter_stats() -> Dict[str, int]:
    return dict.fromkeys(FILTER_STAT_KEYS, 0)


def _record_mm_tag_types(mm_tag: str, mm_tag_types: MutableMapping[str, int]) -> None:
    for mod_spec in mm_tag.split(';'):
        if not mod_spec:
            continue
        base_mod = mod_spec.split(',')[0] if ',' in mod_spec else mod_spec
        mm_tag_types[base_mod] += 1


def _safe_percent(numerator: int, denominator: int) -> float:
    return 100 * numerator / max(1, denominator)


def _print_filter_stats(filter_stats: Dict[str, int], min_mapq: int, min_read_length: int) -> None:
    print("\n    Filter Statistics:")
    print(f"      Total scanned:      {filter_stats['scanned']:>10,}")
    print(f"      Passed all filters: {filter_stats['processed']:>10,} ({_safe_percent(filter_stats['processed'], filter_stats['scanned']):.1f}%)")
    print("      ─────────────────────────────────")
    print(f"      Unmapped:           {filter_stats['unmapped']:>10,}")
    print(f"      Secondary:          {filter_stats['secondary']:>10,}")
    print(f"      Supplementary:      {filter_stats['supplementary']:>10,}")
    print(f"      Low MAPQ (<{min_mapq}):    {filter_stats['low_mapq']:>10,}")
    print(f"      No sequence:        {filter_stats['no_sequence']:>10,}")
    print(f"      Short (<{min_read_length}bp):      {filter_stats['short_read']:>10,}")
    print(f"      No MM tag:          {filter_stats['no_mm_tag']:>10,}")
    print(f"      No ML tag:          {filter_stats['no_ml_tag']:>10,}")


def _read_reference_span(read):
    if read.reference_end is None or read.reference_start is None:
        return None
    return read.reference_end - read.reference_start


def _generate_probs_skip_reason(read, min_mapq: int, min_read_length: int):
    if read.is_unmapped:
        return 'unmapped'
    if read.is_secondary:
        return 'secondary'
    if read.is_supplementary:
        return 'supplementary'
    if read.mapping_quality < min_mapq:
        return 'low_mapq'
    if read.query_sequence is None:
        return 'no_sequence'
    reference_span = _read_reference_span(read)
    if reference_span is None:
        return 'no_sequence'
    if reference_span < min_read_length:
        return 'short_read'
    return None


def _read_mm_ml_tags_or_skip(read):
    mm_tag = get_preferred_tag(read, 'MM', 'Mm')
    if mm_tag is None:
        return None, None, 'no_mm_tag'

    ml_raw = get_preferred_tag(read, 'ML', 'Ml')
    if ml_raw is None:
        return mm_tag, None, 'no_ml_tag'

    return mm_tag, list(ml_raw), None


def _target_bases_for_mode(mode: str) -> List[str]:
    if mode == 'daf':
        return ['C']
    return ['A']


def _context_size_label(context_sizes: List[int], include_mer_span: bool = True) -> str:
    if len(context_sizes) == 1:
        k = context_sizes[0]
        label = f"k={k}"
        if include_mer_span:
            label += f" ({2*k + 1}-mer)"
        return label
    min_k = min(context_sizes)
    max_k = max(context_sizes)
    label = f"k={min_k} to k={max_k}"
    if include_mer_span:
        label += f" ({2*min_k + 1}-mer to {2*max_k + 1}-mer)"
    return label


def _context_probability_frame(probs: pd.DataFrame, column_name: str) -> pd.DataFrame:
    if len(probs) == 0:
        return pd.DataFrame(columns=['context', column_name])
    return probs[['context', 'ratio']].rename(columns={'ratio': column_name})


def _combined_probability_frame(
    accessible_probs: pd.DataFrame,
    inaccessible_probs: pd.DataFrame,
) -> pd.DataFrame:
    accessible = _context_probability_frame(accessible_probs, 'accessible_prob')
    inaccessible = _context_probability_frame(inaccessible_probs, 'inaccessible_prob')
    combined = accessible.merge(inaccessible, on='context', how='outer')
    combined = combined.fillna(0.0)
    combined = combined.sort_values('context').reset_index(drop=True)
    combined['encode'] = range(len(combined))
    return combined[['encode', 'context', 'accessible_prob', 'inaccessible_prob']]


def _write_probability_table(probs: pd.DataFrame, output_path: str) -> None:
    probs[list(PROBABILITY_TSV_COLUMNS)].to_csv(
        output_path, sep='\t', index=False,
    )


def _probability_counter_summary(counter: ContextCounter) -> dict:
    return {
        'positions': counter.total_positions,
        'modified': counter.total_modified,
        'rate': counter.total_modified / max(1, counter.total_positions),
        'unique_contexts': len(counter.counts),
    }


def _max_reads_per_file(max_reads: int, n_files: int) -> int:
    return max_reads // n_files if max_reads > 0 else 0


def _accumulate_filter_stats(combined_stats, filter_stats: Dict[str, int]) -> None:
    for key, value in filter_stats.items():
        combined_stats[key] += value


def _remove_temporary_probability_counters(
    output_dir: str,
    base_name: str,
    target_bases: List[str],
) -> None:
    for base in target_bases:
        for sample in SAMPLE_TYPES:
            tmp_file = _probability_counter_path(
                output_dir,
                base_name,
                sample,
                base,
                temporary=True,
            )
            if os.path.exists(tmp_file):
                os.remove(tmp_file)


def _save_temporary_probability_counters(
    counters: Dict[str, ContextCounter],
    output_dir: str,
    base_name: str,
    sample_name: str,
) -> None:
    for base, counter in counters.items():
        counter.save(
            _probability_counter_path(
                output_dir,
                base_name,
                sample_name,
                base,
                temporary=True,
            )
        )


def _record_filter_skip(filter_stats: Dict[str, int], skip_reason) -> bool:
    if skip_reason is None:
        return False
    filter_stats[skip_reason] += 1
    return True


def _probability_read_tags_or_skip(
    read,
    filter_stats: Dict[str, int],
    min_mapq: int,
    min_read_length: int,
):
    skip_reason = _generate_probs_skip_reason(read, min_mapq, min_read_length)
    if _record_filter_skip(filter_stats, skip_reason):
        return None

    mm_tag, ml_tag, tag_skip_reason = _read_mm_ml_tags_or_skip(read)
    if _record_filter_skip(filter_stats, tag_skip_reason):
        return None

    return mm_tag, ml_tag


def _process_probability_read(
    read,
    counters: Dict[str, ContextCounter],
    mode: str,
    prob_threshold: int,
    edge_trim: int,
    mm_tag: str,
    ml_tag,
    mm_tag_types: MutableMapping[str, int],
    strand_assignments: MutableMapping[str, int],
) -> None:
    _record_mm_tag_types(mm_tag, mm_tag_types)

    mod_positions = parse_mm_tag_query_positions(
        mm_tag, ml_tag, read.query_sequence,
        read.is_reverse, prob_threshold, mode=mode,
    )
    strand, target_base = detect_strand_and_base(
        read.query_sequence, mod_positions, mode,
    )
    strand_assignments[f"{strand}:{target_base}"] += 1

    if mode == 'daf':
        if 'C' in counters:
            counters['C'].process_read_daf(
                read.query_sequence, mod_positions, strand, edge_trim,
            )
    elif target_base in counters:
        counters[target_base].process_read(
            read.query_sequence, mod_positions, edge_trim,
        )


def _process_probability_read_or_skip(
    read,
    counters: Dict[str, ContextCounter],
    mode: str,
    args,
    filter_stats: Dict[str, int],
    mm_tag_types: MutableMapping[str, int],
    strand_assignments: MutableMapping[str, int],
) -> bool:
    tags = _probability_read_tags_or_skip(
        read, filter_stats, args.min_mapq, args.min_read_length,
    )
    if tags is None:
        return False

    mm_tag, ml_tag = tags
    _process_probability_read(
        read, counters, mode, args.prob_threshold, args.edge_trim,
        mm_tag, ml_tag, mm_tag_types, strand_assignments,
    )
    return True


def _progress_postfix(reads_processed: int, reads_scanned: int) -> Dict[str, str]:
    return {
        'processed': f'{reads_processed:,}',
        'scanned': f'{reads_scanned:,}',
        'rate': f'{_safe_percent(reads_processed, reads_scanned):.1f}%'
    }


def _maybe_update_probability_progress(
    pbar,
    reads_processed: int,
    reads_scanned: int,
) -> None:
    if reads_scanned > 0 and reads_scanned % 5000 == 0:
        pbar.set_postfix(_progress_postfix(reads_processed, reads_scanned))


def _count_items_desc(counts):
    return sorted(counts.items(), key=lambda x: -x[1])


def _print_daf_diagnostics(mm_tag_types, strand_assignments) -> None:
    print("\n    MM tag modification types found:")
    for tag_type, count in _count_items_desc(mm_tag_types):
        print(f"      {tag_type}: {count:,}")
    print("\n    Strand assignments:")
    for assignment, count in _count_items_desc(strand_assignments):
        print(f"      {assignment}: {count:,}")


def _read_limit_reached(max_reads: int, reads_processed: int) -> bool:
    return max_reads > 0 and reads_processed >= max_reads


def _finalize_probability_bam_run(
    *,
    mode: str,
    args,
    filter_stats: Dict[str, int],
    mm_tag_types,
    strand_assignments,
    verbose: bool,
) -> None:
    if verbose:
        _print_filter_stats(filter_stats, args.min_mapq, args.min_read_length)

    if mode == 'daf':
        _print_daf_diagnostics(mm_tag_types, strand_assignments)


def process_bam(bam_path: str, counters: Dict[str, ContextCounter],
                mode: str, args, max_reads: int = 0, verbose: bool = False) -> Tuple[int, dict]:
    """
    Process a BAM file and update counters.

    Args:
        bam_path: Path to BAM file
        counters: Dict of center_base -> ContextCounter
        mode: Analysis mode
        args: Command line arguments
        max_reads: Max reads to process (0 = all)
        verbose: Show detailed filter statistics

    Returns:
        (reads_processed, filter_stats dict)
    """
    reads_processed = 0
    reads_scanned = 0

    # Track filter failures
    filter_stats = _new_filter_stats()

    # Track MM tag types seen (for diagnostics)
    mm_tag_types = defaultdict(int)
    strand_assignments = defaultdict(int)

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        pbar = tqdm(bam.fetch(), desc=f"Processing {os.path.basename(bam_path)}")
        for read in pbar:
            reads_scanned += 1
            filter_stats['scanned'] += 1

            # Update progress bar periodically
            _maybe_update_probability_progress(
                pbar, reads_processed, reads_scanned,
            )

            processed = _process_probability_read_or_skip(
                read, counters, mode, args, filter_stats,
                mm_tag_types, strand_assignments,
            )
            if not processed:
                continue

            reads_processed += 1
            filter_stats['processed'] += 1

            # Check max reads (limit by PROCESSED count for consistent sample size)
            if _read_limit_reached(max_reads, reads_processed):
                break

    _finalize_probability_bam_run(
        mode=mode,
        args=args,
        filter_stats=filter_stats,
        mm_tag_types=mm_tag_types,
        strand_assignments=strand_assignments,
        verbose=verbose,
    )

    return reads_processed, filter_stats


def process_sample_set(bam_files: List[str], counters: Dict[str, ContextCounter],
                       mode: str, args, sample_name: str, output_dir: str, base_name: str) -> Tuple[int, int, dict]:
    """Process a set of BAM files for one sample type.

    Returns:
        (total_processed, total_scanned, combined_filter_stats)
    """
    total_reads = 0
    total_scanned = 0
    combined_stats = defaultdict(int)

    max_per_file = _max_reads_per_file(args.max_reads, len(bam_files))

    for bam_file in bam_files:
        print(f"\n  Processing: {bam_file}")
        reads, filter_stats = process_bam(bam_file, counters, mode, args, max_per_file,
                                          verbose=getattr(args, 'verbose', False))
        total_reads += reads
        total_scanned += filter_stats['scanned']

        _accumulate_filter_stats(combined_stats, filter_stats)

        print(f"    Processed {reads:,} reads (scanned {filter_stats['scanned']:,})")

        # Save intermediate
        if args.save_interval > 0:
            _save_temporary_probability_counters(
                counters, output_dir, base_name, sample_name,
            )

        if _read_limit_reached(args.max_reads, total_reads):
            break

    return total_reads, total_scanned, dict(combined_stats)


def _print_probability_generation_header(args, output_dir: str, tables_dir: str,
                                         plots_dir: str) -> None:
    print("=" * 60)
    print("FiberHMM Emission Probability Generator")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"  Tables: {tables_dir}/")
    print(f"  Plots:  {plots_dir}/")
    print(f"\nMode: {args.mode}")
    context_label = _context_size_label(args.context_sizes)
    if len(args.context_sizes) == 1:
        print(f"Context size: {context_label}")
    else:
        print(f"Context sizes: {context_label}")
    print(f"Max reads per sample: {args.max_reads if args.max_reads > 0 else 'all'}")

    print("\nFilter thresholds:")
    print(f"  Min MAPQ:           {args.min_mapq}")
    print(f"  Min read length:    {args.min_read_length} bp")
    print(f"  Edge trim:          {args.edge_trim} bp")
    print(
        f"  ML prob threshold:  {args.prob_threshold}/255 "
        f"({100 * args.prob_threshold / 255:.1f}%)"
    )

    print(f"\nAccessible samples (naked/dechromatinized): {len(args.accessible)} files")
    for f in args.accessible:
        print(f"  - {f}")

    print(f"\nInaccessible samples (untreated/native): {len(args.inaccessible)} files")
    for f in args.inaccessible:
        print(f"  - {f}")


def _new_probability_counters(target_bases: list, max_context: int) -> dict:
    return {base: ContextCounter(max_context, base) for base in target_bases}


def _process_probability_sample_group(
    section_label: str,
    sample_description: str,
    estimate_description: str,
    bam_files: List[str],
    target_bases: list,
    max_context: int,
    mode: str,
    args,
    sample_name: str,
    output_dir: str,
    base_name: str,
) -> tuple:
    print("\n" + "=" * 60)
    print(f"Processing {section_label} samples ({sample_description})")
    print(f"  This estimates {estimate_description}")
    print("=" * 60)

    counters = _new_probability_counters(target_bases, max_context)
    reads, scanned, stats = process_sample_set(
        bam_files, counters, mode, args, sample_name, output_dir, base_name,
    )
    return counters, reads, scanned, stats


def _print_probability_results_summary(
    accessible_reads: int,
    accessible_scanned: int,
    inaccessible_reads: int,
    inaccessible_scanned: int,
) -> None:
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    print("\nAccessible (naked DNA):")
    print(
        f"  Reads processed: {accessible_reads:,} "
        f"(scanned {accessible_scanned:,}, "
        f"{100 * accessible_reads / max(1, accessible_scanned):.1f}% pass rate)"
    )
    print("Inaccessible (native):")
    print(
        f"  Reads processed: {inaccessible_reads:,} "
        f"(scanned {inaccessible_scanned:,}, "
        f"{100 * inaccessible_reads / max(1, inaccessible_scanned):.1f}% pass rate)"
    )


def _print_probability_base_summary(
    base: str,
    accessible_counter: ContextCounter,
    inaccessible_counter: ContextCounter,
) -> None:
    acc_summary = _probability_counter_summary(accessible_counter)
    inacc_summary = _probability_counter_summary(inaccessible_counter)

    print(f"\n{base}-centered contexts:")
    print("  Accessible:")
    print(f"    Positions: {acc_summary['positions']:,}")
    print(f"    Modified: {acc_summary['modified']:,}")
    print(f"    Rate: {acc_summary['rate']:.4f}")
    print(f"    Unique contexts: {acc_summary['unique_contexts']:,}")

    print("  Inaccessible:")
    print(f"    Positions: {inacc_summary['positions']:,}")
    print(f"    Modified: {inacc_summary['modified']:,}")
    print(f"    Rate: {inacc_summary['rate']:.4f}")
    print(f"    Unique contexts: {inacc_summary['unique_contexts']:,}")


def _write_probability_tables_for_base(
    tables_dir: str,
    base_name: str,
    base: str,
    context_sizes: List[int],
    accessible_counter: ContextCounter,
    inaccessible_counter: ContextCounter,
) -> None:
    tsv_context_label = _context_size_label(
        context_sizes, include_mer_span=False,
    )
    print(f"\n  Generating TSV files for {tsv_context_label}...")
    for ctx_size in context_sizes:
        _, acc_probs = accessible_counter.get_encoding_table(ctx_size)
        acc_tsv = _probability_table_path(
            tables_dir,
            base_name,
            "accessible",
            base,
            ctx_size,
        )
        _write_probability_table(acc_probs, acc_tsv)

        _, inacc_probs = inaccessible_counter.get_encoding_table(ctx_size)
        inacc_tsv = _probability_table_path(
            tables_dir,
            base_name,
            "inaccessible",
            base,
            ctx_size,
        )
        _write_probability_table(inacc_probs, inacc_tsv)

        n_acc = len(acc_probs)
        n_inacc = len(inacc_probs)
        print(
            f"    k={ctx_size} ({2*ctx_size + 1}-mer): "
            f"{n_acc} accessible, {n_inacc} inaccessible contexts"
        )


def _write_combined_probability_tables(
    tables_dir: str,
    base_name: str,
    target_bases: List[str],
    context_sizes: List[int],
    accessible_counters: Dict[str, ContextCounter],
    inaccessible_counters: Dict[str, ContextCounter],
) -> None:
    print("\n" + "-" * 60)
    print("Creating combined probability files for train_model.py:")

    for ctx_size in context_sizes:
        for base in target_bases:
            accessible_counter = accessible_counters[base]
            inaccessible_counter = inaccessible_counters[base]
            if (
                accessible_counter.total_positions == 0
                and inaccessible_counter.total_positions == 0
            ):
                continue

            _, acc_probs = accessible_counter.get_encoding_table(ctx_size)
            _, inacc_probs = inaccessible_counter.get_encoding_table(ctx_size)

            if len(acc_probs) == 0 and len(inacc_probs) == 0:
                continue

            combined = _combined_probability_frame(acc_probs, inacc_probs)
            combined_file = _combined_probability_table_path(
                tables_dir,
                base_name,
                base,
                ctx_size,
            )
            combined.to_csv(combined_file, sep='\t', index=False)
            print(f"  {combined_file} ({len(combined)} contexts)")


def _probability_counters_have_data(
    accessible_counter: ContextCounter,
    inaccessible_counter: ContextCounter,
) -> bool:
    return (
        accessible_counter.total_positions > 0
        or inaccessible_counter.total_positions > 0
    )


def _save_probability_outputs_for_base(
    output_dir: str,
    tables_dir: str,
    base_name: str,
    base: str,
    context_sizes: List[int],
    accessible_counter: ContextCounter,
    inaccessible_counter: ContextCounter,
) -> bool:
    _print_probability_base_summary(
        base, accessible_counter, inaccessible_counter,
    )

    accessible_counter.save(
        _probability_counter_path(output_dir, base_name, "accessible", base)
    )
    inaccessible_counter.save(
        _probability_counter_path(output_dir, base_name, "inaccessible", base)
    )

    if not _probability_counters_have_data(
        accessible_counter, inaccessible_counter,
    ):
        print(f"\n  WARNING: No data for {base}-centered contexts, skipping TSV generation")
        print(f"           This may indicate the MM tags don't contain {base} modifications")
        return False

    _write_probability_tables_for_base(
        tables_dir,
        base_name,
        base,
        context_sizes,
        accessible_counter,
        inaccessible_counter,
    )
    return True


def _generate_probability_stats_for_contexts(
    context_sizes: List[int],
    accessible_counters: Dict[str, ContextCounter],
    inaccessible_counters: Dict[str, ContextCounter],
    plots_dir: str,
    base_name: str,
) -> None:
    print("\n" + "-" * 60)
    print("Generating statistics and plots:")
    for ctx_size in context_sizes:
        print(f"\n  k={ctx_size} ({2*ctx_size+1}-mer):")
        generate_probability_stats(
            accessible_counters,
            inaccessible_counters,
            plots_dir,
            base_name,
            context_size=ctx_size,
        )


def _print_probability_completion_message() -> None:
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review the modification rates - accessible should be HIGH,")
    print("     inaccessible should be LOW (background miscall rate)")
    print("  2. Use the *_probs.tsv files with train_model.py to build HMM")


@dataclass(frozen=True)
class _ProbabilityRunContext:
    max_context: int
    output_dir: str
    tables_dir: str
    plots_dir: str
    base_name: str
    target_bases: List[str]


def _setup_probability_run(args) -> _ProbabilityRunContext:
    np.random.seed(args.seed)

    max_context = max(args.context_sizes)
    output_dir = args.output
    _, tables_dir_path, plots_dir_path = setup_output_dirs(output_dir)
    tables_dir = str(tables_dir_path)
    plots_dir = str(plots_dir_path)
    base_name = get_base_name(output_dir)

    _print_probability_generation_header(args, output_dir, tables_dir, plots_dir)

    # Determine which bases to track based on mode
    target_bases = _target_bases_for_mode(args.mode)

    print(f"\nTarget bases: {', '.join(target_bases)}")
    if args.mode == 'daf':
        print("  (G-strand reads will be reverse complemented to C-centered contexts)")

    return _ProbabilityRunContext(
        max_context=max_context,
        output_dir=output_dir,
        tables_dir=tables_dir,
        plots_dir=plots_dir,
        base_name=base_name,
        target_bases=target_bases,
    )


def _process_probability_control_groups(args, run: _ProbabilityRunContext):
    accessible_counters, accessible_reads, accessible_scanned, accessible_stats = (
        _process_probability_sample_group(
            "ACCESSIBLE",
            "naked/dechromatinized DNA",
            "P(methylation | accessible)",
            args.accessible,
            run.target_bases,
            run.max_context,
            args.mode,
            args,
            "accessible",
            run.output_dir,
            run.base_name,
        )
    )

    inaccessible_counters, inaccessible_reads, inaccessible_scanned, inaccessible_stats = (
        _process_probability_sample_group(
            "INACCESSIBLE",
            "untreated/native chromatin",
            "P(methylation | inaccessible) = background rate",
            args.inaccessible,
            run.target_bases,
            run.max_context,
            args.mode,
            args,
            "inaccessible",
            run.output_dir,
            run.base_name,
        )
    )

    return (
        (accessible_counters, accessible_reads, accessible_scanned, accessible_stats),
        (
            inaccessible_counters,
            inaccessible_reads,
            inaccessible_scanned,
            inaccessible_stats,
        ),
    )


def _save_probability_run_outputs(
    args,
    run: _ProbabilityRunContext,
    accessible_result,
    inaccessible_result,
) -> None:
    accessible_counters, accessible_reads, accessible_scanned, _ = accessible_result
    (
        inaccessible_counters,
        inaccessible_reads,
        inaccessible_scanned,
        _,
    ) = inaccessible_result

    _print_probability_results_summary(
        accessible_reads,
        accessible_scanned,
        inaccessible_reads,
        inaccessible_scanned,
    )

    for base in run.target_bases:
        _save_probability_outputs_for_base(
            run.output_dir,
            run.tables_dir,
            run.base_name,
            base,
            args.context_sizes,
            accessible_counters[base],
            inaccessible_counters[base],
        )

    _write_combined_probability_tables(
        run.tables_dir,
        run.base_name,
        run.target_bases,
        args.context_sizes,
        accessible_counters,
        inaccessible_counters,
    )

    _remove_temporary_probability_counters(
        run.output_dir,
        run.base_name,
        run.target_bases,
    )

    if args.stats:
        _generate_probability_stats_for_contexts(
            args.context_sizes,
            accessible_counters,
            inaccessible_counters,
            run.plots_dir,
            run.base_name,
        )

    _print_probability_completion_message()


def main():
    args = parse_args()
    run = _setup_probability_run(args)
    accessible_result, inaccessible_result = _process_probability_control_groups(
        args,
        run,
    )
    _save_probability_run_outputs(args, run, accessible_result, inaccessible_result)


if __name__ == '__main__':
    main()
