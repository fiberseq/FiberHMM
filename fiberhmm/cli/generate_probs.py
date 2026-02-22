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

import pandas as pd
import numpy as np
import argparse
import sys
import os
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, Set, Tuple, List
import pysam

# Package imports
from fiberhmm.core.bam_reader import parse_mm_tag_query_positions
from fiberhmm.probabilities.context_counter import ContextCounter
from fiberhmm.probabilities.utils import detect_strand_and_base, setup_output_dirs, get_base_name
from fiberhmm.probabilities.stats import generate_probability_stats


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
    filter_stats = {
        'scanned': 0,
        'unmapped': 0,
        'secondary': 0,
        'supplementary': 0,
        'low_mapq': 0,
        'no_sequence': 0,
        'short_read': 0,
        'no_mm_tag': 0,
        'no_ml_tag': 0,
        'processed': 0,
    }

    # Track MM tag types seen (for diagnostics)
    mm_tag_types = defaultdict(int)
    strand_assignments = defaultdict(int)

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        pbar = tqdm(bam.fetch(), desc=f"Processing {os.path.basename(bam_path)}")
        for read in pbar:
            reads_scanned += 1
            filter_stats['scanned'] += 1

            # Update progress bar periodically
            if reads_scanned % 5000 == 0:
                pbar.set_postfix({
                    'processed': f'{reads_processed:,}',
                    'scanned': f'{reads_scanned:,}',
                    'rate': f'{100*reads_processed/max(1,reads_scanned):.1f}%'
                })

            # Filters with tracking
            if read.is_unmapped:
                filter_stats['unmapped'] += 1
                continue
            if read.is_secondary:
                filter_stats['secondary'] += 1
                continue
            if read.is_supplementary:
                filter_stats['supplementary'] += 1
                continue
            if read.mapping_quality < args.min_mapq:
                filter_stats['low_mapq'] += 1
                continue
            if read.query_sequence is None:
                filter_stats['no_sequence'] += 1
                continue
            if read.reference_end is None or read.reference_start is None:
                filter_stats['no_sequence'] += 1
                continue
            if read.reference_end - read.reference_start < args.min_read_length:
                filter_stats['short_read'] += 1
                continue

            # Get MM/ML tags
            mm_tag = None
            ml_tag = None

            try:
                if read.has_tag('MM'):
                    mm_tag = read.get_tag('MM')
                elif read.has_tag('Mm'):
                    mm_tag = read.get_tag('Mm')
            except KeyError:
                pass

            if mm_tag is None:
                filter_stats['no_mm_tag'] += 1
                continue

            # Track MM tag types (for diagnostics)
            for mod_spec in mm_tag.split(';'):
                if mod_spec:
                    base_mod = mod_spec.split(',')[0] if ',' in mod_spec else mod_spec
                    mm_tag_types[base_mod] += 1

            try:
                if read.has_tag('ML'):
                    ml_tag = list(read.get_tag('ML'))
                elif read.has_tag('Ml'):
                    ml_tag = list(read.get_tag('Ml'))
            except KeyError:
                pass

            if ml_tag is None:
                filter_stats['no_ml_tag'] += 1
                continue

            # Parse modifications
            mod_positions = parse_mm_tag_query_positions(
                mm_tag, ml_tag, read.query_sequence,
                read.is_reverse, args.prob_threshold, mode=mode
            )

            # Determine strand and target base
            strand, target_base = detect_strand_and_base(
                read.query_sequence, mod_positions, mode
            )
            strand_assignments[f"{strand}:{target_base}"] += 1

            # Process read with appropriate counter
            if mode == 'daf':
                # DAF-seq: always use C counter (G contexts are RC'd to C internally)
                if 'C' in counters:
                    counters['C'].process_read_daf(
                        read.query_sequence, mod_positions, strand, args.edge_trim
                    )
            elif target_base in counters:
                counters[target_base].process_read(
                    read.query_sequence, mod_positions, args.edge_trim
                )

            reads_processed += 1
            filter_stats['processed'] += 1

            # Check max reads (limit by PROCESSED count for consistent sample size)
            if max_reads > 0 and reads_processed >= max_reads:
                break

    # Print filter stats in verbose mode
    if verbose:
        print(f"\n    Filter Statistics:")
        print(f"      Total scanned:      {filter_stats['scanned']:>10,}")
        print(f"      Passed all filters: {filter_stats['processed']:>10,} ({100*filter_stats['processed']/max(1,filter_stats['scanned']):.1f}%)")
        print(f"      ─────────────────────────────────")
        print(f"      Unmapped:           {filter_stats['unmapped']:>10,}")
        print(f"      Secondary:          {filter_stats['secondary']:>10,}")
        print(f"      Supplementary:      {filter_stats['supplementary']:>10,}")
        print(f"      Low MAPQ (<{args.min_mapq}):    {filter_stats['low_mapq']:>10,}")
        print(f"      No sequence:        {filter_stats['no_sequence']:>10,}")
        print(f"      Short (<{args.min_read_length}bp):      {filter_stats['short_read']:>10,}")
        print(f"      No MM tag:          {filter_stats['no_mm_tag']:>10,}")
        print(f"      No ML tag:          {filter_stats['no_ml_tag']:>10,}")

    # Print MM tag diagnostics for daf mode (to show C vs G strand detection)
    if mode == 'daf':
        print(f"\n    MM tag modification types found:")
        for tag_type, count in sorted(mm_tag_types.items(), key=lambda x: -x[1]):
            print(f"      {tag_type}: {count:,}")
        print(f"\n    Strand assignments:")
        for assignment, count in sorted(strand_assignments.items(), key=lambda x: -x[1]):
            print(f"      {assignment}: {count:,}")

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

    max_per_file = args.max_reads // len(bam_files) if args.max_reads > 0 else 0

    for bam_file in bam_files:
        print(f"\n  Processing: {bam_file}")
        reads, filter_stats = process_bam(bam_file, counters, mode, args, max_per_file,
                                          verbose=getattr(args, 'verbose', False))
        total_reads += reads
        total_scanned += filter_stats['scanned']

        # Accumulate filter stats
        for k, v in filter_stats.items():
            combined_stats[k] += v

        print(f"    Processed {reads:,} reads (scanned {filter_stats['scanned']:,})")

        # Save intermediate
        if args.save_interval > 0:
            for base, counter in counters.items():
                counter.save(os.path.join(output_dir, f"{base_name}_{sample_name}_{base}.probs.pkl.tmp"))

        if args.max_reads > 0 and total_reads >= args.max_reads:
            break

    return total_reads, total_scanned, dict(combined_stats)


def main():
    args = parse_args()

    np.random.seed(args.seed)

    # max_context for internal storage is the max of requested sizes
    max_context = max(args.context_sizes)

    # Set up output directories
    output_dir = args.output
    tables_dir = os.path.join(output_dir, "tables")
    plots_dir = os.path.join(output_dir, "plots")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Base name for files (last component of output path)
    base_name = os.path.basename(output_dir.rstrip('/'))
    if not base_name:
        base_name = "probs"

    print("=" * 60)
    print("FiberHMM Emission Probability Generator")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"  Tables: {tables_dir}/")
    print(f"  Plots:  {plots_dir}/")
    print(f"\nMode: {args.mode}")
    if len(args.context_sizes) == 1:
        k = args.context_sizes[0]
        print(f"Context size: k={k} ({2*k + 1}-mer)")
    else:
        print(f"Context sizes: k={min(args.context_sizes)} to k={max(args.context_sizes)} "
              f"({2*min(args.context_sizes) + 1}-mer to {2*max(args.context_sizes) + 1}-mer)")
    print(f"Max reads per sample: {args.max_reads if args.max_reads > 0 else 'all'}")

    print(f"\nFilter thresholds:")
    print(f"  Min MAPQ:           {args.min_mapq}")
    print(f"  Min read length:    {args.min_read_length} bp")
    print(f"  Edge trim:          {args.edge_trim} bp")
    print(f"  ML prob threshold:  {args.prob_threshold}/255 ({100*args.prob_threshold/255:.1f}%)")

    print(f"\nAccessible samples (naked/dechromatinized): {len(args.accessible)} files")
    for f in args.accessible:
        print(f"  - {f}")

    print(f"\nInaccessible samples (untreated/native): {len(args.inaccessible)} files")
    for f in args.inaccessible:
        print(f"  - {f}")

    # Determine which bases to track based on mode
    if args.mode in ('pacbio-fiber', 'nanopore-fiber'):
        target_bases = ['A']
    elif args.mode == 'daf':
        # DAF mode uses only C-centered contexts; G contexts are reverse complemented to C
        target_bases = ['C']
    else:
        target_bases = ['A']

    print(f"\nTarget bases: {', '.join(target_bases)}")
    if args.mode == 'daf':
        print("  (G-strand reads will be reverse complemented to C-centered contexts)")

    # Process ACCESSIBLE samples (naked DNA)
    print("\n" + "=" * 60)
    print("Processing ACCESSIBLE samples (naked/dechromatinized DNA)")
    print("  This estimates P(methylation | accessible)")
    print("=" * 60)

    accessible_counters = {base: ContextCounter(max_context, base) for base in target_bases}
    accessible_reads, accessible_scanned, accessible_stats = process_sample_set(
        args.accessible, accessible_counters, args.mode, args, "accessible", output_dir, base_name)

    # Process INACCESSIBLE samples (untreated/native)
    print("\n" + "=" * 60)
    print("Processing INACCESSIBLE samples (untreated/native chromatin)")
    print("  This estimates P(methylation | inaccessible) = background rate")
    print("=" * 60)

    inaccessible_counters = {base: ContextCounter(max_context, base) for base in target_bases}
    inaccessible_reads, inaccessible_scanned, inaccessible_stats = process_sample_set(
        args.inaccessible, inaccessible_counters, args.mode, args, "inaccessible", output_dir, base_name)

    # Report and save results
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    print(f"\nAccessible (naked DNA):")
    print(f"  Reads processed: {accessible_reads:,} (scanned {accessible_scanned:,}, "
          f"{100*accessible_reads/max(1,accessible_scanned):.1f}% pass rate)")
    print(f"Inaccessible (native):")
    print(f"  Reads processed: {inaccessible_reads:,} (scanned {inaccessible_scanned:,}, "
          f"{100*inaccessible_reads/max(1,inaccessible_scanned):.1f}% pass rate)")

    for base in target_bases:
        acc = accessible_counters[base]
        inacc = inaccessible_counters[base]

        print(f"\n{base}-centered contexts:")
        print(f"  Accessible:")
        print(f"    Positions: {acc.total_positions:,}")
        print(f"    Modified: {acc.total_modified:,}")
        print(f"    Rate: {acc.total_modified / max(1, acc.total_positions):.4f}")
        print(f"    Unique contexts: {len(acc.counts):,}")

        print(f"  Inaccessible:")
        print(f"    Positions: {inacc.total_positions:,}")
        print(f"    Modified: {inacc.total_modified:,}")
        print(f"    Rate: {inacc.total_modified / max(1, inacc.total_positions):.4f}")
        print(f"    Unique contexts: {len(inacc.counts):,}")

        # Save full counters (PKL files in output root)
        acc.save(os.path.join(output_dir, f"{base_name}_accessible_{base}.probs.pkl"))
        inacc.save(os.path.join(output_dir, f"{base_name}_inaccessible_{base}.probs.pkl"))

        # Skip if no data for this base
        if acc.total_positions == 0 and inacc.total_positions == 0:
            print(f"\n  WARNING: No data for {base}-centered contexts, skipping TSV generation")
            print(f"           This may indicate the MM tags don't contain {base} modifications")
            continue

        # Save TSV files for requested context sizes
        if len(args.context_sizes) == 1:
            print(f"\n  Generating TSV files for k={args.context_sizes[0]}...")
        else:
            print(f"\n  Generating TSV files for k={min(args.context_sizes)} to k={max(args.context_sizes)}...")
        for ctx_size in args.context_sizes:
            # Accessible probabilities
            _, acc_probs = acc.get_encoding_table(ctx_size)
            acc_tsv = os.path.join(tables_dir, f"{base_name}_accessible_{base}_k{ctx_size}.tsv")
            acc_probs[['encode', 'context', 'hit', 'nohit', 'ratio']].to_csv(
                acc_tsv, sep='\t', index=False
            )

            # Inaccessible probabilities
            _, inacc_probs = inacc.get_encoding_table(ctx_size)
            inacc_tsv = os.path.join(tables_dir, f"{base_name}_inaccessible_{base}_k{ctx_size}.tsv")
            inacc_probs[['encode', 'context', 'hit', 'nohit', 'ratio']].to_csv(
                inacc_tsv, sep='\t', index=False
            )

            n_acc = len(acc_probs)
            n_inacc = len(inacc_probs)
            print(f"    k={ctx_size} ({2*ctx_size + 1}-mer): {n_acc} accessible, {n_inacc} inaccessible contexts")

    # Also create combined probability files for direct use with train_model.py
    print("\n" + "-" * 60)
    print("Creating combined probability files for train_model.py:")

    for ctx_size in args.context_sizes:
        for base in target_bases:
            # Skip if no data for this base
            if accessible_counters[base].total_positions == 0 and \
               inaccessible_counters[base].total_positions == 0:
                continue

            _, acc_probs = accessible_counters[base].get_encoding_table(ctx_size)
            _, inacc_probs = inaccessible_counters[base].get_encoding_table(ctx_size)

            # Skip if both are empty
            if len(acc_probs) == 0 and len(inacc_probs) == 0:
                continue

            # Use context as key (more robust than encode which may differ)
            acc_df = acc_probs[['context', 'ratio']].rename(columns={'ratio': 'accessible_prob'}) if len(acc_probs) > 0 else pd.DataFrame(columns=['context', 'accessible_prob'])
            inacc_df = inacc_probs[['context', 'ratio']].rename(columns={'ratio': 'inaccessible_prob'}) if len(inacc_probs) > 0 else pd.DataFrame(columns=['context', 'inaccessible_prob'])

            # Outer merge to include all observed contexts from both samples
            combined = acc_df.merge(inacc_df, on='context', how='outer')
            combined = combined.fillna(0.0)  # Missing contexts get 0 probability
            combined = combined.sort_values('context').reset_index(drop=True)
            combined['encode'] = range(len(combined))

            # Reorder columns
            combined = combined[['encode', 'context', 'accessible_prob', 'inaccessible_prob']]

            combined_file = os.path.join(tables_dir, f"{base_name}_{base}_k{ctx_size}_probs.tsv")
            combined.to_csv(combined_file, sep='\t', index=False)
            print(f"  {combined_file} ({len(combined)} contexts)")

    # Clean up temp files
    for base in target_bases:
        for sample in ['accessible', 'inaccessible']:
            tmp_file = os.path.join(output_dir, f"{base_name}_{sample}_{base}.probs.pkl.tmp")
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    # Generate stats if requested (for each context size)
    if args.stats:
        print("\n" + "-" * 60)
        print("Generating statistics and plots:")
        for ctx_size in args.context_sizes:
            print(f"\n  k={ctx_size} ({2*ctx_size+1}-mer):")
            generate_probability_stats(accessible_counters, inaccessible_counters,
                                       plots_dir, base_name, context_size=ctx_size)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review the modification rates - accessible should be HIGH,")
    print("     inaccessible should be LOW (background miscall rate)")
    print("  2. Use the *_probs.tsv files with train_model.py to build HMM")


if __name__ == '__main__':
    main()
