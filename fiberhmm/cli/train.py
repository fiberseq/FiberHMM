#!/usr/bin/env python3
"""
FiberHMM train_model.py v2
Train HMM from PacBio BAM files.

Modes:
- m6a (default): Fiber-seq m6A methylation footprinting (PacBio, A-centered with RC)
- nanopore: Nanopore fiber-seq (A-centered, no RC)
- daf: DAF-seq deamination footprinting (strand-specific C/G)

Supports variable context sizes (default k=3 for 7-mer hexamers).

No genome context H5 file needed - uses read sequences directly.
Uses native HMM implementation (no hmmlearn dependency).
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os
import pickle
import json
from tqdm import tqdm

from fiberhmm.core.bam_reader import (read_bam, FiberRead, encode_from_query_sequence,
                                       detect_daf_strand, ContextEncoder)
from fiberhmm.core.hmm import FiberHMM, train_model as train_hmm_models
from fiberhmm.core.model_io import save_model, load_model

pd.options.mode.chained_assignment = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train FiberHMM model from PacBio BAM files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-i', '--input', nargs='+',
                        help='Input BAM file(s) for training (not required with --base-model)')
    parser.add_argument('-p', '--probs', required=True, nargs=2,
                        metavar=('ACC', 'INACC'),
                        help='Accessible and inaccessible probability files (.tsv or .probs.pkl)')
    parser.add_argument('--base-model', type=str, default=None,
                        metavar='MODEL',
                        help='Use transitions from existing model with new emissions (skip training)')
    parser.add_argument('-o', '--outdir', required=True,
                        help='Output directory')

    # Mode selection
    parser.add_argument('--mode', choices=['pacbio-fiber', 'nanopore-fiber', 'daf'], default='pacbio-fiber',
                        help='Analysis mode: pacbio-fiber (PacBio), nanopore-fiber (Nanopore), daf (DAF-seq)')

    # Context size
    parser.add_argument('-k', '--context-size', type=int, default=3,
                        help='Context size (bases on each side): 3=7mer, 5=11mer, etc.')

    parser.add_argument('-c', '--iterations', type=int, default=10,
                        help='Training iterations (random initializations)')
    parser.add_argument('-r', '--read-count', type=int, default=500,
                        help='Total reads to sample for training')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('-e', '--edge-trim', type=int, default=10,
                        help='Edge masking')
    parser.add_argument('-q', '--min-mapq', type=int, default=20,
                        help='Min mapping quality')
    parser.add_argument('--prob-threshold', type=int, default=125,
                        help='Min ML probability (default matches ft-extract)')
    parser.add_argument('--min-read-length', type=int, default=1000,
                        help='Min aligned length')
    parser.add_argument('-a', '--prob-adjust', type=float, default=1.0,
                        help='Accessible probability adjustment factor')
    parser.add_argument('--use-hmmlearn', action='store_true',
                        help='Use hmmlearn instead of native implementation (for legacy compatibility)')
    parser.add_argument('--stats', action='store_true',
                        help='Generate training statistics and example plots')
    parser.add_argument('--n-examples', type=int, default=5,
                        help='Number of example reads to plot (with --stats)')

    return parser.parse_args()


def load_probability_file(filepath: str, context_size: int = 3) -> pd.DataFrame:
    """
    Load probability file - supports both legacy TSV and new .probs.pkl format.

    Args:
        filepath: Path to probability file
        context_size: Desired context size (for .probs.pkl files)

    Returns:
        DataFrame with columns: encode, ratio (and optionally hit, nohit)
    """
    if filepath.endswith('.probs.pkl') or filepath.endswith('.pkl'):
        # New format - ContextCounter pickle
        from fiberhmm.probabilities.context_counter import ContextCounter
        counter = ContextCounter.load(filepath)
        _, probs = counter.get_encoding_table(context_size)
        return probs
    else:
        # Legacy TSV format
        # Expected columns: encode (or index), hit, nohit, ratio
        probs = pd.read_csv(filepath, sep='\t')

        # Handle different column naming conventions
        if 'encode' not in probs.columns:
            if probs.columns[0] in ['Unnamed: 0', '']:
                probs = probs.rename(columns={probs.columns[0]: 'encode'})
            elif probs.index.name == 'encode' or probs.index.dtype == np.int64:
                probs = probs.reset_index()
                probs = probs.rename(columns={probs.columns[0]: 'encode'})

        probs['encode'] = probs['encode'].astype(int)

        # Ensure ratio column exists
        if 'ratio' not in probs.columns:
            if 'hit' in probs.columns and 'nohit' in probs.columns:
                total = probs['hit'] + probs['nohit']
                probs['ratio'] = probs['hit'] / total.replace(0, 1)
            else:
                raise ValueError(f"Cannot find probability values in {filepath}")

        return probs


def make_emission_probs(acc_file: str, inacc_file: str,
                        context_size: int = 3, prob_adjust: float = 1.0):
    """
    Generate emission probability matrix from probability files.

    Returns 2D array: [state] x [observation]

    Note: State assignment during training is arbitrary - normalize_states()
    is called on model load to ensure canonical order:
    - State 0: Footprint (low methylation probability)
    - State 1: Accessible (high methylation probability)

    Observations (for context_size k):
    - 0 to 4^(2k)-1: Modified base with context
    - 4^(2k): Non-target position (modified version)
    - 4^(2k)+1 to 2*4^(2k): Unmodified base with context
    - 2*4^(2k)+1: Non-target position (unmodified version)
    """
    # Load probability files
    acc = load_probability_file(acc_file, context_size)
    inacc = load_probability_file(inacc_file, context_size)

    # Rename columns for merging
    acc = acc[['encode', 'ratio']].rename(columns={'ratio': 'prob_acc'})
    inacc = inacc[['encode', 'ratio']].rename(columns={'ratio': 'prob_inacc'})

    # Apply probability adjustment
    acc['prob_acc'] = (acc['prob_acc'].astype(float) * prob_adjust).clip(0, 1)
    inacc['prob_inacc'] = inacc['prob_inacc'].astype(float)

    # Merge
    hexamer_probs = acc.merge(inacc, on='encode', how='outer')

    # Calculate expected number of codes
    n_codes = ContextEncoder.get_n_codes(context_size)  # 4^(2k)

    # Fill missing encodings
    all_encodes = np.arange(n_codes + 1)  # 0 to 4^(2k) including non-target
    missing = np.setdiff1d(all_encodes, hexamer_probs['encode'].values)
    if len(missing) > 0:
        missing_df = pd.DataFrame({
            'encode': missing,
            'prob_acc': [0.0] * len(missing),
            'prob_inacc': [0.0] * len(missing)
        })
        hexamer_probs = pd.concat([hexamer_probs, missing_df])

    hexamer_probs = hexamer_probs.sort_values('encode').reset_index(drop=True)
    hexamer_probs = hexamer_probs.fillna(0)

    # Build emission matrix
    # Row order here is arbitrary - normalize_states() fixes it on load
    prob_acc = hexamer_probs['prob_acc'].values
    prob_inacc = hexamer_probs['prob_inacc'].values

    emission_probs = np.array([
        np.concatenate([prob_acc, 1 - prob_acc]),      # Higher meth probs
        np.concatenate([prob_inacc, 1 - prob_inacc])   # Lower meth probs
    ])

    return emission_probs


def sample_reads_indexed(bam_path: str, n_samples: int, seed: int,
                         mode: str = 'pacbio-fiber', min_mapq: int = 20,
                         prob_threshold: int = 125, min_read_length: int = 1000) -> list:
    """
    Sample reads using BAM index for efficient random access.

    Instead of reading the entire BAM, picks random genomic coordinates
    and fetches reads from those positions using the index.
    """
    import pysam

    np.random.seed(seed)
    sampled = []
    seen_read_ids = set()

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        # Get reference lengths from header
        ref_lengths = dict(zip(bam.references, bam.lengths))

        # Filter to main chromosomes (skip scaffolds/contigs for efficiency)
        main_chroms = {}
        for ref, length in ref_lengths.items():
            # Skip very short references and obvious scaffolds
            if length < 100000:
                continue
            # Skip common scaffold patterns
            if any(x in ref.lower() for x in ['random', 'un_', 'chrun', '_alt', 'hap', 'scaffold']):
                continue
            main_chroms[ref] = length

        if not main_chroms:
            main_chroms = ref_lengths  # Fall back to all if filtering removed everything

        # Calculate cumulative lengths for weighted random selection
        chroms = list(main_chroms.keys())
        lengths = np.array([main_chroms[c] for c in chroms])
        cum_lengths = np.cumsum(lengths)
        total_length = cum_lengths[-1]

        # Over-sample positions since not all will yield valid reads
        n_attempts = n_samples * 20
        max_attempts = n_samples * 100
        attempts = 0

        while len(sampled) < n_samples and attempts < max_attempts:
            # Generate batch of random positions
            batch_size = min(n_attempts, n_samples * 5)
            random_positions = np.random.randint(0, total_length, size=batch_size)

            for genome_pos in random_positions:
                if len(sampled) >= n_samples:
                    break

                attempts += 1

                # Find which chromosome this position falls in
                chrom_idx = np.searchsorted(cum_lengths, genome_pos, side='right')
                chrom = chroms[chrom_idx]

                # Get position within chromosome
                if chrom_idx > 0:
                    pos = genome_pos - cum_lengths[chrom_idx - 1]
                else:
                    pos = genome_pos

                # Fetch reads overlapping this position
                try:
                    for read in bam.fetch(chrom, max(0, pos - 100), pos + 100):
                        # Apply filters
                        if read.is_unmapped or read.is_secondary or read.is_supplementary:
                            continue
                        if read.mapping_quality < min_mapq:
                            continue
                        if read.query_sequence is None:
                            continue
                        aligned_length = read.reference_end - read.reference_start
                        if aligned_length < min_read_length:
                            continue

                        # Skip if we've already sampled this read
                        if read.query_name in seen_read_ids:
                            continue

                        # Get modifications
                        from fiberhmm.core.bam_reader import get_modified_positions_pysam, parse_mm_tag_query_positions, FiberRead

                        mod_query_pos = get_modified_positions_pysam(read, prob_threshold, mode)

                        if not mod_query_pos:
                            try:
                                mm_tag = read.get_tag('MM') if read.has_tag('MM') else (
                                    read.get_tag('Mm') if read.has_tag('Mm') else None)
                                ml_tag = list(read.get_tag('ML')) if read.has_tag('ML') else (
                                    list(read.get_tag('Ml')) if read.has_tag('Ml') else None)
                            except KeyError:
                                mm_tag = ml_tag = None

                            if mm_tag and ml_tag:
                                mod_query_pos = parse_mm_tag_query_positions(
                                    mm_tag, ml_tag, read.query_sequence,
                                    read.is_reverse, prob_threshold, mode=mode
                                )

                        if not mod_query_pos:
                            continue

                        # Build query-to-ref map as list
                        query_to_ref = [None] * len(read.query_sequence)
                        for query_pos, ref_pos in read.get_aligned_pairs():
                            if query_pos is not None and ref_pos is not None:
                                query_to_ref[query_pos] = ref_pos

                        fiber_read = FiberRead(
                            read_id=read.query_name,
                            chrom=read.reference_name,
                            ref_start=read.reference_start,
                            ref_end=read.reference_end,
                            strand='-' if read.is_reverse else '+',
                            query_sequence=read.query_sequence,
                            m6a_query_positions=set(mod_query_pos),
                            query_to_ref=query_to_ref
                        )

                        sampled.append(fiber_read)
                        seen_read_ids.add(read.query_name)
                        break  # Got one read from this position

                except ValueError:
                    # Invalid region
                    continue

    return sampled


def sample_reads(bam_files: list, read_count: int, seed: int,
                 mode: str = 'pacbio-fiber', **kwargs) -> list:
    """Sample reads from BAM files for training using index-based sampling."""
    np.random.seed(seed)

    all_reads = []
    reads_per_file = max(1, read_count // len(bam_files))

    for bam_file in tqdm(bam_files, desc="Sampling reads"):
        file_reads = sample_reads_indexed(
            bam_file, reads_per_file, seed,
            mode=mode, **kwargs
        )
        all_reads.extend(file_reads)
        print(f"  Sampled {len(file_reads)} from {os.path.basename(bam_file)}")

    # Shuffle and truncate
    np.random.shuffle(all_reads)

    # If we need more reads and have some, sample with replacement
    if len(all_reads) < read_count and len(all_reads) > 0:
        extra_needed = read_count - len(all_reads)
        extra_indices = np.random.choice(len(all_reads), extra_needed, replace=True)
        all_reads.extend([all_reads[i] for i in extra_indices])

    return all_reads[:read_count]


def generate_training_arrays(reads: list, edge_trim: int,
                             n_iterations: int, mode: str = 'pacbio-fiber',
                             context_size: int = 3) -> tuple:
    """
    Generate training arrays from sampled reads.
    Returns (dict of training arrays, list of read IDs, list of encoded reads).
    """
    print(f"Encoding {len(reads)} reads (context size k={context_size})...")

    encoded_reads = []
    train_rids = []
    valid_reads = []  # Keep track of which reads were successfully encoded

    for fiber_read in tqdm(reads, desc="Encoding"):
        # Detect strand based on mode
        if mode == 'daf':
            strand = detect_daf_strand(fiber_read.query_sequence,
                                       fiber_read.m6a_query_positions)
        elif mode == 'nanopore-fiber':
            strand = '.'  # No strand detection for nanopore
        else:  # m6a mode
            strand = '.'

        encoded = encode_from_query_sequence(
            fiber_read.query_sequence,
            fiber_read.m6a_query_positions,
            edge_trim,
            mode=mode,
            strand=strand,
            context_size=context_size
        )

        if len(encoded) > 0:
            encoded_reads.append(encoded)
            train_rids.append(fiber_read.read_id)
            valid_reads.append(fiber_read)

    print(f"Successfully encoded {len(encoded_reads)} reads")

    if len(encoded_reads) == 0:
        raise ValueError("No reads successfully encoded! Check input BAM files.")

    # Create shuffled training arrays
    train_arrays = {}
    for i in range(n_iterations):
        np.random.seed(i)
        indices = np.random.permutation(len(encoded_reads))
        shuffled = [encoded_reads[j] for j in indices]
        train_arrays[i] = np.concatenate(shuffled).astype(int)

    return train_arrays, train_rids, encoded_reads, valid_reads


def train_hmm(emission_probs: np.ndarray, train_arrays: dict,
              use_legacy: bool = False) -> tuple:
    """
    Train HMM models and return best one.
    Returns (best_model, all_models).
    """
    print(f"Training with {'hmmlearn (legacy)' if use_legacy else 'native'} implementation")

    best_model, models = train_hmm_models(
        emission_probs,
        train_arrays,
        n_iterations=len(train_arrays),
        use_legacy=use_legacy
    )

    print(f"\nBest model selected")
    print(f"Start probabilities: {best_model.startprob_}")
    print(f"Transition matrix:\n{best_model.transmat_}")

    return best_model, models


def generate_training_stats(model: FiberHMM, sampled_reads: list, encoded_reads: list,
                            emission_probs: np.ndarray, output_dir: str,
                            n_examples: int = 5, mode: str = 'pacbio-fiber'):
    """
    Generate training statistics and example plots.

    Args:
        model: Trained FiberHMM model
        sampled_reads: List of FiberRead objects
        encoded_reads: List of encoded arrays
        emission_probs: Emission probability matrix
        output_dir: Output directory for plots
        n_examples: Number of example reads to plot
        mode: Analysis mode
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection
    except ImportError:
        print("  Warning: matplotlib not installed. Skipping stats plots.")
        return

    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    pdf_path = os.path.join(plots_dir, 'training_stats.pdf')

    # Run Viterbi on all encoded reads to get footprint statistics
    print("  Running Viterbi on training reads...")
    all_footprint_sizes = []
    all_msp_sizes = []
    all_states = []

    for encoded in encoded_reads:
        if len(encoded) == 0:
            continue
        states = model.predict(encoded)
        all_states.append(states)

        # Extract footprint and MSP sizes
        current_state = states[0]
        current_length = 1

        for i in range(1, len(states)):
            if states[i] == current_state:
                current_length += 1
            else:
                if current_state == 0:  # Footprint
                    all_footprint_sizes.append(current_length)
                else:  # MSP
                    all_msp_sizes.append(current_length)
                current_state = states[i]
                current_length = 1

        # Don't forget the last segment
        if current_state == 0:
            all_footprint_sizes.append(current_length)
        else:
            all_msp_sizes.append(current_length)

    with PdfPages(pdf_path) as pdf:
        # Page 1: Model parameters
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('FiberHMM Training Results', fontsize=14, fontweight='bold')

        # 1. Transition matrix
        ax = axes[0, 0]
        trans = model.transmat_
        im = ax.imshow(trans, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Footprint', 'Accessible'])
        ax.set_yticklabels(['Footprint', 'Accessible'])
        ax.set_xlabel('To State')
        ax.set_ylabel('From State')
        ax.set_title('Transition Probabilities')

        # Add text annotations
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{trans[i, j]:.4f}', ha='center', va='center',
                       color='white' if trans[i, j] > 0.5 else 'black', fontsize=12)

        # Calculate expected durations
        fp_stay = trans[0, 0]
        msp_stay = trans[1, 1]
        expected_fp = 1 / (1 - fp_stay) if fp_stay < 1 else float('inf')
        expected_msp = 1 / (1 - msp_stay) if msp_stay < 1 else float('inf')
        ax.text(0.5, -0.3, f'Expected durations: Footprint={expected_fp:.0f}bp, MSP={expected_msp:.0f}bp',
               transform=ax.transAxes, ha='center', fontsize=10)

        # 2. Emission probability distribution
        ax = axes[0, 1]
        # Get emission probs - matrix is (2, n_codes): row 0=footprint, row 1=accessible
        fp_emissions = emission_probs[0, :]
        msp_emissions = emission_probs[1, :]

        # Filter to observed contexts (non-zero)
        fp_nonzero = fp_emissions[fp_emissions > 0]
        msp_nonzero = msp_emissions[msp_emissions > 0]

        bins = np.linspace(0, 1, 51)
        ax.hist(fp_nonzero, bins=bins, alpha=0.6, label=f'Footprint (n={len(fp_nonzero):,})',
               color='firebrick')
        ax.hist(msp_nonzero, bins=bins, alpha=0.6, label=f'Accessible (n={len(msp_nonzero):,})',
               color='forestgreen')
        ax.set_xlabel('P(methylation | state, context)')
        ax.set_ylabel('Number of contexts')
        ax.set_title('Emission Probability Distribution')
        ax.legend()

        # 3. Footprint size distribution
        ax = axes[1, 0]
        if len(all_footprint_sizes) > 0:
            bins = np.arange(0, min(500, max(all_footprint_sizes) + 10), 10)
            ax.hist(all_footprint_sizes, bins=bins, color='firebrick', alpha=0.7, edgecolor='white')
            ax.axvline(np.median(all_footprint_sizes), color='black', linestyle='--',
                      label=f'Median: {np.median(all_footprint_sizes):.0f}bp')
            ax.axvline(147, color='blue', linestyle=':', alpha=0.7, label='Nucleosome (147bp)')
            ax.set_xlabel('Footprint Size (bp)')
            ax.set_ylabel('Count')
            ax.set_title(f'Footprint Sizes (n={len(all_footprint_sizes):,})')
            ax.legend()
            ax.set_xlim(0, 500)

        # 4. MSP size distribution
        ax = axes[1, 1]
        if len(all_msp_sizes) > 0:
            bins = np.arange(0, min(500, max(all_msp_sizes) + 10), 10)
            ax.hist(all_msp_sizes, bins=bins, color='forestgreen', alpha=0.7, edgecolor='white')
            ax.axvline(np.median(all_msp_sizes), color='black', linestyle='--',
                      label=f'Median: {np.median(all_msp_sizes):.0f}bp')
            ax.set_xlabel('MSP Size (bp)')
            ax.set_ylabel('Count')
            ax.set_title(f'MSP Sizes (n={len(all_msp_sizes):,})')
            ax.legend()
            ax.set_xlim(0, 500)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2+: Example reads with overview and zoom panels
        n_to_plot = min(n_examples, len(sampled_reads), len(all_states))

        for idx in range(n_to_plot):
            read = sampled_reads[idx]
            states = all_states[idx]
            encoded = encoded_reads[idx]

            seq_len = len(read.query_sequence)
            m6a_positions = sorted(read.m6a_query_positions)

            # Get posterior probabilities
            posteriors = model.predict_proba(encoded)  # Shape: (T, 2)
            footprint_prob = posteriors[:, 0]  # P(footprint)

            # Helper function to draw state blocks
            def draw_state_blocks(ax, region_states, region_len):
                """Draw footprint (green) and accessible (white) blocks."""
                patches = []
                colors_list = []
                current_state = region_states[0]
                block_start = 0

                for i in range(1, len(region_states)):
                    if region_states[i] != current_state:
                        width = i - block_start
                        rect = Rectangle((block_start, 0), width, 1)
                        patches.append(rect)
                        # Green for footprint (state 0), white for accessible (state 1)
                        colors_list.append('forestgreen' if current_state == 0 else 'white')
                        current_state = region_states[i]
                        block_start = i

                # Last block
                width = len(region_states) - block_start
                rect = Rectangle((block_start, 0), width, 1)
                patches.append(rect)
                colors_list.append('forestgreen' if current_state == 0 else 'white')

                collection = PatchCollection(patches, facecolors=colors_list,
                                            edgecolors='lightgray', linewidths=0.3)
                ax.add_collection(collection)
                ax.set_xlim(0, region_len)
                ax.set_ylim(0, 1)

            # Create figure: 4 rows (overview) + 3 rows per zoom (3 zooms = 9 rows) = 13 rows
            # But let's use GridSpec for better control
            fig = plt.figure(figsize=(14, 16))

            # Use gridspec for layout
            gs = fig.add_gridspec(6, 3, height_ratios=[0.8, 1.2, 0.8, 0.8, 1.2, 0.8],
                                 hspace=0.35, wspace=0.25,
                                 left=0.06, right=0.98, top=0.93, bottom=0.04)

            # Title
            fig.suptitle(f'Example Read {idx+1}: {read.read_id[:50]}...\n'
                        f'Length: {seq_len:,}bp | Chromosome: {read.chrom}:{read.ref_start:,}-{read.ref_end:,} | '
                        f'm6A calls: {len(m6a_positions):,}',
                        fontsize=11, y=0.98)

            # === OVERVIEW PANELS (top row, span all 3 columns) ===

            # Overview: m6A positions (purple)
            ax_overview_m6a = fig.add_subplot(gs[0, :])
            if len(m6a_positions) > 0:
                ax_overview_m6a.eventplot([m6a_positions], colors='purple', lineoffsets=0.5,
                                         linelengths=0.8, linewidths=0.3)
            ax_overview_m6a.set_xlim(0, seq_len)
            ax_overview_m6a.set_ylim(0, 1)
            ax_overview_m6a.set_ylabel('m6A', fontsize=9)
            ax_overview_m6a.set_yticks([])
            ax_overview_m6a.set_title('Overview: m6A Positions', fontsize=10, loc='left')
            ax_overview_m6a.set_xticklabels([])

            # Overview: Footprints
            ax_overview_fp = fig.add_subplot(gs[1, :])
            draw_state_blocks(ax_overview_fp, states, seq_len)
            ax_overview_fp.set_ylabel('State', fontsize=9)
            ax_overview_fp.set_yticks([])
            ax_overview_fp.set_title('Overview: Footprints (green) vs Accessible (white)', fontsize=10, loc='left')
            ax_overview_fp.set_xticklabels([])
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='forestgreen', edgecolor='gray', label='Footprint'),
                             Patch(facecolor='white', edgecolor='gray', label='Accessible')]
            ax_overview_fp.legend(handles=legend_elements, loc='upper right', fontsize=8)

            # Overview: Probability
            ax_overview_prob = fig.add_subplot(gs[2, :])
            ax_overview_prob.fill_between(range(len(footprint_prob)), 0, footprint_prob,
                                         color='forestgreen', alpha=0.4, step='mid')
            ax_overview_prob.axhline(0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            ax_overview_prob.set_xlim(0, seq_len)
            ax_overview_prob.set_ylim(0, 1)
            ax_overview_prob.set_ylabel('P(FP)', fontsize=9)
            ax_overview_prob.set_yticks([0, 0.5, 1])
            ax_overview_prob.tick_params(axis='y', labelsize=7)
            ax_overview_prob.set_xlabel('Position (bp)', fontsize=9)
            ax_overview_prob.set_title('Overview: Footprint Probability', fontsize=10, loc='left')

            # === ZOOM PANELS (3 random 1kb windows) ===

            # Select 3 random windows
            np.random.seed(idx)  # Reproducible per read
            window_size = 1000
            n_windows = 3

            # Ensure windows don't overlap and fit within read
            if seq_len >= window_size * n_windows:
                # Divide read into regions and pick one window from each
                region_size = seq_len // n_windows
                window_starts = []
                for w in range(n_windows):
                    region_start = w * region_size
                    region_end = min((w + 1) * region_size, seq_len) - window_size
                    if region_end > region_start:
                        start = np.random.randint(region_start, region_end)
                    else:
                        start = region_start
                    window_starts.append(start)
            else:
                # Read is short, just use evenly spaced windows
                window_starts = [int(seq_len * i / (n_windows + 1)) for i in range(1, n_windows + 1)]
                window_starts = [max(0, min(s, seq_len - window_size)) for s in window_starts]

            # Plot each zoom window
            for w_idx, w_start in enumerate(window_starts):
                w_end = min(w_start + window_size, seq_len)
                w_len = w_end - w_start

                # Row indices in grid: 3, 4, 5 for zoom panels
                # m6A
                ax_zoom_m6a = fig.add_subplot(gs[3, w_idx])
                m6a_in_window = [p - w_start for p in m6a_positions if w_start <= p < w_end]
                if len(m6a_in_window) > 0:
                    ax_zoom_m6a.eventplot([m6a_in_window], colors='purple', lineoffsets=0.5,
                                         linelengths=0.8, linewidths=1.0)
                ax_zoom_m6a.set_xlim(0, w_len)
                ax_zoom_m6a.set_ylim(0, 1)
                ax_zoom_m6a.set_ylabel('m6A', fontsize=8)
                ax_zoom_m6a.set_yticks([])
                ax_zoom_m6a.set_xticklabels([])
                ax_zoom_m6a.set_title(f'Zoom {w_idx+1}: {w_start:,}-{w_end:,}bp', fontsize=9)

                # Mark this window on overview
                ax_overview_m6a.axvspan(w_start, w_end, alpha=0.15, color=['red', 'blue', 'orange'][w_idx])
                ax_overview_fp.axvspan(w_start, w_end, alpha=0.15, color=['red', 'blue', 'orange'][w_idx])
                ax_overview_prob.axvspan(w_start, w_end, alpha=0.15, color=['red', 'blue', 'orange'][w_idx])

                # Footprints
                ax_zoom_fp = fig.add_subplot(gs[4, w_idx])
                window_states = states[w_start:w_end]
                draw_state_blocks(ax_zoom_fp, window_states, w_len)
                ax_zoom_fp.set_ylabel('State', fontsize=8)
                ax_zoom_fp.set_yticks([])
                ax_zoom_fp.set_xticklabels([])

                # Probability
                ax_zoom_prob = fig.add_subplot(gs[5, w_idx])
                window_prob = footprint_prob[w_start:w_end]
                ax_zoom_prob.fill_between(range(len(window_prob)), 0, window_prob,
                                         color='forestgreen', alpha=0.4, step='mid')
                ax_zoom_prob.plot(range(len(window_prob)), window_prob,
                                 color='forestgreen', linewidth=0.5, alpha=0.8)
                ax_zoom_prob.axhline(0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                ax_zoom_prob.set_xlim(0, w_len)
                ax_zoom_prob.set_ylim(0, 1)
                ax_zoom_prob.set_ylabel('P(FP)', fontsize=8)
                ax_zoom_prob.set_yticks([0, 0.5, 1])
                ax_zoom_prob.tick_params(axis='y', labelsize=7)
                ax_zoom_prob.set_xlabel('Position (bp)', fontsize=8)

            pdf.savefig(fig)
            plt.close(fig)

    print(f"  Saved: {pdf_path}")

    # Also save a standalone PNG of the first example (simplified version)
    if len(sampled_reads) > 0 and len(all_states) > 0:
        read = sampled_reads[0]
        states = all_states[0]
        encoded = encoded_reads[0]

        seq_len = len(read.query_sequence)
        m6a_positions = sorted(read.m6a_query_positions)
        posteriors = model.predict_proba(encoded)
        footprint_prob = posteriors[:, 0]

        fig, axes = plt.subplots(3, 1, figsize=(14, 6),
                                gridspec_kw={'height_ratios': [1, 1.5, 1]})

        # m6A
        ax = axes[0]
        if len(m6a_positions) > 0:
            ax.eventplot([m6a_positions], colors='purple', lineoffsets=0.5,
                        linelengths=0.8, linewidths=0.3)
        ax.set_xlim(0, seq_len)
        ax.set_ylim(0, 1)
        ax.set_ylabel('m6A')
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_title(f'Example: {read.read_id[:40]}... ({seq_len:,}bp)', fontsize=10)

        # Footprints
        ax = axes[1]
        patches = []
        colors = []
        current_state = states[0]
        start_pos = 0
        for i in range(1, len(states)):
            if states[i] != current_state:
                width = i - start_pos
                rect = Rectangle((start_pos, 0), width, 1)
                patches.append(rect)
                colors.append('forestgreen' if current_state == 0 else 'white')
                current_state = states[i]
                start_pos = i
        width = len(states) - start_pos
        rect = Rectangle((start_pos, 0), width, 1)
        patches.append(rect)
        colors.append('forestgreen' if current_state == 0 else 'white')
        collection = PatchCollection(patches, facecolors=colors, edgecolors='lightgray', linewidths=0.3)
        ax.add_collection(collection)
        ax.set_xlim(0, seq_len)
        ax.set_ylim(0, 1)
        ax.set_ylabel('State')
        ax.set_yticks([])
        ax.set_xticklabels([])

        # Probability
        ax = axes[2]
        ax.fill_between(range(len(footprint_prob)), 0, footprint_prob,
                       color='forestgreen', alpha=0.4, step='mid')
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlim(0, seq_len)
        ax.set_ylim(0, 1)
        ax.set_ylabel('P(FP)')
        ax.set_xlabel('Position (bp)')

        plt.tight_layout()
        png_path = os.path.join(plots_dir, 'example_read.png')
        plt.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {png_path}")

    # Write text summary
    summary_path = os.path.join(plots_dir, 'training_stats.txt')
    with open(summary_path, 'w') as f:
        f.write("FiberHMM Training Statistics\n")
        f.write("=" * 60 + "\n\n")

        f.write("Transition Matrix:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Footprint → Footprint: {model.transmat_[0, 0]:.6f}\n")
        f.write(f"  Footprint → Accessible: {model.transmat_[0, 1]:.6f}\n")
        f.write(f"  Accessible → Footprint: {model.transmat_[1, 0]:.6f}\n")
        f.write(f"  Accessible → Accessible: {model.transmat_[1, 1]:.6f}\n")

        expected_fp = 1 / (1 - model.transmat_[0, 0]) if model.transmat_[0, 0] < 1 else float('inf')
        expected_msp = 1 / (1 - model.transmat_[1, 1]) if model.transmat_[1, 1] < 1 else float('inf')
        f.write(f"\n  Expected footprint duration: {expected_fp:.1f} bp\n")
        f.write(f"  Expected MSP duration: {expected_msp:.1f} bp\n")

        f.write("\nEmission Probabilities:\n")
        f.write("-" * 40 + "\n")
        fp_nonzero = emission_probs[0, :][emission_probs[0, :] > 0]
        msp_nonzero = emission_probs[1, :][emission_probs[1, :] > 0]
        f.write(f"  Footprint contexts: {len(fp_nonzero):,} (mean={np.mean(fp_nonzero):.4f}, median={np.median(fp_nonzero):.4f})\n")
        f.write(f"  Accessible contexts: {len(msp_nonzero):,} (mean={np.mean(msp_nonzero):.4f}, median={np.median(msp_nonzero):.4f})\n")

        f.write("\nTraining Data Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Reads used: {len(sampled_reads)}\n")
        if len(all_footprint_sizes) > 0:
            f.write(f"  Total footprints: {len(all_footprint_sizes):,}\n")
            f.write(f"  Footprint sizes: mean={np.mean(all_footprint_sizes):.1f}, "
                   f"median={np.median(all_footprint_sizes):.1f}, "
                   f"range={min(all_footprint_sizes)}-{max(all_footprint_sizes)}\n")
        if len(all_msp_sizes) > 0:
            f.write(f"  Total MSPs: {len(all_msp_sizes):,}\n")
            f.write(f"  MSP sizes: mean={np.mean(all_msp_sizes):.1f}, "
                   f"median={np.median(all_msp_sizes):.1f}, "
                   f"range={min(all_msp_sizes)}-{max(all_msp_sizes)}\n")

    print(f"  Saved: {summary_path}")


def main():
    args = parse_args()

    # Validate arguments
    if not args.base_model and not args.input:
        print("Error: --input is required unless using --base-model")
        sys.exit(1)

    mode_descs = {
        'pacbio-fiber': 'PacBio fiber-seq (A-centered)',
        'nanopore-fiber': 'Nanopore fiber-seq (A-centered)',
        'daf': 'DAF-seq deamination (C/G-centered)'
    }
    mode_desc = mode_descs.get(args.mode, args.mode)

    context_mer = 2 * args.context_size + 1
    n_codes = ContextEncoder.get_n_codes(args.context_size)

    if args.base_model:
        print("FiberHMM Model Builder (using base model)")
        print(f"  Mode: {args.mode} ({mode_desc})")
        print(f"  Context: k={args.context_size} ({context_mer}-mer, {n_codes:,} codes)")
        print(f"  Base model: {args.base_model}")
        print(f"  (Transitions from base model, new emissions from prob files)")
    else:
        print("FiberHMM Model Training v2")
        print(f"  Mode: {args.mode} ({mode_desc})")
        print(f"  Context: k={args.context_size} ({context_mer}-mer, {n_codes:,} codes)")
        print(f"  Input: {args.input}")
        print(f"  Iterations: {args.iterations}")
        print(f"  Reads: {args.read_count}")
        print(f"  Seed: {args.seed}")
        print("  (No genome context file needed)")

    os.makedirs(args.outdir, exist_ok=True)

    # Generate emission probabilities
    print(f"\nLoading emission probabilities...")
    emission_probs = make_emission_probs(
        args.probs[0], args.probs[1],
        context_size=args.context_size,
        prob_adjust=args.prob_adjust
    )
    print(f"Emission matrix: {emission_probs.shape}")

    # Initialize variables for stats generation
    valid_reads = []
    encoded_reads = []
    train_rids = []

    if args.base_model:
        # Use transitions from base model with new emissions
        print(f"\nLoading base model: {args.base_model}")
        base_model = load_model(args.base_model, normalize=False)

        # Validate emission dimensions match
        if base_model.emissionprob_.shape[1] != emission_probs.shape[1]:
            print(f"Error: Emission size mismatch!")
            print(f"  Base model: {base_model.emissionprob_.shape[1]} observations")
            print(f"  New emissions: {emission_probs.shape[1]} observations")
            print(f"  Check that context_size matches (k={args.context_size})")
            sys.exit(1)

        # Create new model with base model's transitions and new emissions
        best_model = FiberHMM(n_states=2)
        best_model.startprob_ = base_model.startprob_.copy()
        best_model.transmat_ = base_model.transmat_.copy()
        best_model.emissionprob_ = emission_probs

        print(f"  Kept startprob: {best_model.startprob_}")
        print(f"  Kept transmat:\n{best_model.transmat_}")
        print(f"  Replaced emissions: {emission_probs.shape}")

        all_models = [best_model]
    else:
        # Normal training path
        # Sample reads
        print(f"\nSampling reads from {len(args.input)} BAM file(s)...")
        sampled = sample_reads(
            args.input, args.read_count, args.seed,
            mode=args.mode,
            min_mapq=args.min_mapq,
            prob_threshold=args.prob_threshold,
            min_read_length=args.min_read_length
        )
        print(f"Total sampled: {len(sampled)} reads")

        # Generate training arrays
        train_arrays, train_rids, encoded_reads, valid_reads = generate_training_arrays(
            sampled, args.edge_trim, args.iterations, args.mode, args.context_size
        )

        # Train
        print(f"\nTraining HMM ({args.iterations} iterations)...")
        best_model, all_models = train_hmm(emission_probs, train_arrays, args.use_hmmlearn)

    # Save
    print(f"\nSaving to {args.outdir}")

    # Save best model in JSON format (recommended - portable, human-readable)
    save_model(
        best_model,
        os.path.join(args.outdir, 'best-model.json'),
        context_size=args.context_size,
        mode=args.mode
    )
    print(f"  Saved: best-model.json (recommended)")

    # Also save in NPZ for backwards compatibility
    save_model(
        best_model,
        os.path.join(args.outdir, 'best-model.npz'),
        context_size=args.context_size,
        mode=args.mode
    )
    print(f"  Saved: best-model.npz (numpy format)")

    # Save all models as JSON list
    all_models_data = []
    for m in all_models:
        all_models_data.append({
            'n_states': m.n_states,
            'startprob': m.startprob_.tolist(),
            'transmat': m.transmat_.tolist(),
            'emissionprob': m.emissionprob_.tolist()
        })
    with open(os.path.join(args.outdir, 'all_models.json'), 'w') as f:
        json.dump(all_models_data, f)

    # Save training reads (only if we did training)
    if train_rids:
        pd.DataFrame({'rid': train_rids}).to_csv(
            os.path.join(args.outdir, 'training-reads.tsv'),
            sep='\t', index=False
        )

    # Save config (JSON - human readable)
    config = {
        'context_size': args.context_size,
        'mode': args.mode,
        'edge_trim': args.edge_trim,
        'prob_adjust': args.prob_adjust
    }
    if args.base_model:
        config['base_model'] = args.base_model
    with open(os.path.join(args.outdir, 'model_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Generate stats if requested (only if we have training data)
    if args.stats and valid_reads and encoded_reads:
        print(f"\nGenerating training statistics...")
        generate_training_stats(
            best_model, valid_reads, encoded_reads, emission_probs,
            args.outdir, n_examples=args.n_examples, mode=args.mode
        )
    elif args.stats and args.base_model:
        print(f"\nNote: --stats skipped (no training data with --base-model)")

    print("Done!")


if __name__ == '__main__':
    main()
