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

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from fiberhmm.cli.common import MODE_CHOICES, mode_description
from fiberhmm.core.bam_reader import (
    ContextEncoder,
    detect_daf_strand,
    encode_from_query_sequence,
    get_reference_positions,
)
from fiberhmm.core.hmm import FiberHMM
from fiberhmm.core.hmm import train_model as train_hmm_models
from fiberhmm.core.model_io import load_model, save_model
from fiberhmm.core.tag_access import get_preferred_tag
from fiberhmm.inference.read_filters import is_primary_mapped_alignment

pd.options.mode.chained_assignment = None

_TRAINING_ZOOM_COLORS = ('red', 'blue', 'orange')


@dataclass(frozen=True)
class _TrainingArrays:
    train_arrays: dict
    train_rids: list
    encoded_reads: list
    valid_reads: list


@dataclass(frozen=True)
class _TrainingRunResult:
    best_model: object
    all_models: list
    train_rids: list
    valid_reads: list
    encoded_reads: list


@dataclass(frozen=True)
class _TrainingSamplingIndex:
    chroms: list
    cum_lengths: np.ndarray
    total_length: int


@dataclass(frozen=True)
class _ViterbiStateSizeStats:
    footprint_sizes: list
    msp_sizes: list
    states: list


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
    parser.add_argument(
        '--base-model', type=str, default=None,
        metavar='MODEL',
        help='Use transitions from existing model with new emissions (skip training)',
    )
    parser.add_argument('-o', '--outdir', required=True,
                        help='Output directory')

    # Mode selection
    parser.add_argument(
        '--mode', choices=MODE_CHOICES,
        default='pacbio-fiber',
        help='Analysis mode: pacbio-fiber (PacBio), nanopore-fiber (Nanopore), daf (DAF-seq)',
    )

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
    parser.add_argument(
        '--use-hmmlearn', action='store_true',
        help='Use hmmlearn instead of native implementation (for legacy compatibility)',
    )
    parser.add_argument('--stats', action='store_true',
                        help='Generate training statistics and example plots')
    parser.add_argument('--n-examples', type=int, default=5,
                        help='Number of example reads to plot (with --stats)')

    return parser.parse_args()


def _probability_table_with_encode_column(probs: pd.DataFrame) -> pd.DataFrame:
    if 'encode' in probs.columns:
        return probs

    if probs.columns[0] in ['Unnamed: 0', '']:
        return probs.rename(columns={probs.columns[0]: 'encode'})

    if probs.index.name == 'encode' or probs.index.dtype == np.int64:
        probs = probs.reset_index()
        return probs.rename(columns={probs.columns[0]: 'encode'})

    return probs


def _probability_table_with_ratio(probs: pd.DataFrame, filepath: str) -> pd.DataFrame:
    if 'ratio' in probs.columns:
        return probs

    if 'hit' not in probs.columns or 'nohit' not in probs.columns:
        raise ValueError(f"Cannot find probability values in {filepath}")

    total = probs['hit'] + probs['nohit']
    probs['ratio'] = probs['hit'] / total.replace(0, 1)
    return probs


def _load_legacy_probability_table(filepath: str) -> pd.DataFrame:
    # Expected columns: encode (or index), hit, nohit, ratio
    probs = pd.read_csv(filepath, sep='\t')
    probs = _probability_table_with_encode_column(probs)
    probs['encode'] = probs['encode'].astype(int)
    return _probability_table_with_ratio(probs, filepath)


def _is_probability_counter_path(filepath: str) -> bool:
    filepath = os.fspath(filepath)
    filepath_lower = filepath.lower()
    return filepath_lower.endswith('.probs.pkl') or filepath_lower.endswith('.pkl')


def load_probability_file(filepath: str, context_size: int = 3) -> pd.DataFrame:
    """
    Load probability file - supports both legacy TSV and new .probs.pkl format.

    Args:
        filepath: Path to probability file
        context_size: Desired context size (for .probs.pkl files)

    Returns:
        DataFrame with columns: encode, ratio (and optionally hit, nohit)
    """
    if _is_probability_counter_path(filepath):
        # New format - ContextCounter pickle
        from fiberhmm.probabilities.context_counter import ContextCounter
        counter = ContextCounter.load(filepath)
        _, probs = counter.get_encoding_table(context_size)
        return probs
    else:
        return _load_legacy_probability_table(filepath)


def _complete_emission_probability_table(
    hexamer_probs: pd.DataFrame,
    n_codes: int,
) -> pd.DataFrame:
    all_encodes = np.arange(n_codes + 1)  # Include non-target code.
    missing = np.setdiff1d(all_encodes, hexamer_probs['encode'].values)
    if len(missing) > 0:
        missing_df = pd.DataFrame({
            'encode': missing,
            'prob_acc': [0.0] * len(missing),
            'prob_inacc': [0.0] * len(missing)
        })
        hexamer_probs = pd.concat([hexamer_probs, missing_df])

    return hexamer_probs.sort_values('encode').reset_index(drop=True).fillna(0)


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

    hexamer_probs = _complete_emission_probability_table(hexamer_probs, n_codes)

    # Build emission matrix
    # Row order here is arbitrary - normalize_states() fixes it on load
    prob_acc = hexamer_probs['prob_acc'].values
    prob_inacc = hexamer_probs['prob_inacc'].values

    emission_probs = np.array([
        np.concatenate([prob_acc, 1 - prob_acc]),      # Higher meth probs
        np.concatenate([prob_inacc, 1 - prob_inacc])   # Lower meth probs
    ])

    return emission_probs


def _query_to_ref_positions(read) -> list:
    try:
        return get_reference_positions(read)
    except AttributeError:
        query_to_ref = [None] * len(read.query_sequence)
        for query_pos, ref_pos in read.get_aligned_pairs():
            if query_pos is not None and ref_pos is not None:
                query_to_ref[query_pos] = ref_pos
        return query_to_ref


_TRAINING_SKIP_CHROM_PATTERNS = (
    'random', 'un_', 'chrun', '_alt', 'hap', 'scaffold',
)


def _training_chrom_is_sampleable(ref: str, length: int) -> bool:
    if length < 100000:
        return False
    ref_lower = ref.lower()
    return not any(
        pattern in ref_lower for pattern in _TRAINING_SKIP_CHROM_PATTERNS
    )


def _training_sampling_chrom_lengths(ref_lengths: dict) -> dict:
    main_chroms = {}
    for ref, length in ref_lengths.items():
        if _training_chrom_is_sampleable(ref, length):
            main_chroms[ref] = length
    return main_chroms or ref_lengths


def _chrom_pos_from_genome_offset(chroms: list, cum_lengths: np.ndarray,
                                  genome_pos: int):
    chrom_idx = np.searchsorted(cum_lengths, genome_pos, side='right')
    chrom = chroms[chrom_idx]
    if chrom_idx > 0:
        pos = genome_pos - cum_lengths[chrom_idx - 1]
    else:
        pos = genome_pos
    return chrom, int(pos)


def _training_sampling_index(ref_lengths: dict) -> _TrainingSamplingIndex:
    main_chroms = _training_sampling_chrom_lengths(ref_lengths)
    chroms = list(main_chroms.keys())
    lengths = np.array([main_chroms[c] for c in chroms])
    cum_lengths = np.cumsum(lengths)
    return _TrainingSamplingIndex(
        chroms=chroms,
        cum_lengths=cum_lengths,
        total_length=int(cum_lengths[-1]),
    )


def _training_reference_span(read):
    if read.reference_end is None or read.reference_start is None:
        return None
    return read.reference_end - read.reference_start


def _passes_training_sample_filters(read, min_mapq: int,
                                    min_read_length: int) -> bool:
    if not is_primary_mapped_alignment(read):
        return False
    if read.mapping_quality < min_mapq:
        return False
    if read.query_sequence is None:
        return False
    reference_span = _training_reference_span(read)
    return reference_span is not None and reference_span >= min_read_length


def _training_mod_query_positions(read, prob_threshold: int, mode: str) -> set:
    from fiberhmm.core.bam_reader import (
        get_modified_positions_pysam,
    )

    mod_query_pos = get_modified_positions_pysam(read, prob_threshold, mode)
    if mod_query_pos:
        return set(mod_query_pos)

    return _training_mm_ml_query_positions(read, prob_threshold, mode)


def _training_mm_ml_query_positions(read, prob_threshold: int, mode: str) -> set:
    from fiberhmm.core.bam_reader import _has_mm_ml_inputs, parse_mm_tag_query_positions

    mm_tag = get_preferred_tag(read, 'MM', 'Mm')
    ml_tag = get_preferred_tag(read, 'ML', 'Ml')
    if _has_mm_ml_inputs(mm_tag, ml_tag):
        return set(parse_mm_tag_query_positions(
            mm_tag, ml_tag, read.query_sequence,
            read.is_reverse, prob_threshold, mode=mode,
        ))
    return set()


def _training_fiber_read_from_segment(read, mod_query_pos: set):
    from fiberhmm.core.bam_reader import FiberRead

    return FiberRead(
        read_id=read.query_name,
        chrom=read.reference_name,
        ref_start=read.reference_start,
        ref_end=read.reference_end,
        strand='-' if read.is_reverse else '+',
        query_sequence=read.query_sequence,
        m6a_query_positions=set(mod_query_pos),
        query_to_ref=_query_to_ref_positions(read),
        is_reverse=read.is_reverse,
    )


def _training_sample_candidate(
    read,
    seen_read_ids: set,
    min_mapq: int,
    min_read_length: int,
    prob_threshold: int,
    mode: str,
):
    if not _passes_training_sample_filters(read, min_mapq, min_read_length):
        return None
    if read.query_name in seen_read_ids:
        return None

    mod_query_pos = _training_mod_query_positions(read, prob_threshold, mode)
    if not mod_query_pos:
        return None

    return _training_fiber_read_from_segment(read, mod_query_pos)


def _reads_per_training_file(read_count: int, n_files: int) -> int:
    return max(1, read_count // n_files)


def _sample_training_read_at_position(
    bam,
    chrom: str,
    pos: int,
    seen_read_ids: set,
    min_mapq: int,
    min_read_length: int,
    prob_threshold: int,
    mode: str,
):
    try:
        reads = bam.fetch(chrom, max(0, pos - 100), pos + 100)
    except ValueError:
        return None

    for read in reads:
        fiber_read = _training_sample_candidate(
            read,
            seen_read_ids,
            min_mapq,
            min_read_length,
            prob_threshold,
            mode,
        )
        if fiber_read is None:
            continue

        seen_read_ids.add(read.query_name)
        return fiber_read

    return None


def _sample_training_reads_for_positions(
    bam,
    random_positions: np.ndarray,
    sampled: list,
    n_samples: int,
    chroms: list,
    cum_lengths: np.ndarray,
    seen_read_ids: set,
    min_mapq: int,
    min_read_length: int,
    prob_threshold: int,
    mode: str,
) -> int:
    attempts = 0
    for genome_pos in random_positions:
        if len(sampled) >= n_samples:
            break

        attempts += 1

        chrom, pos = _chrom_pos_from_genome_offset(
            chroms, cum_lengths, int(genome_pos),
        )

        fiber_read = _sample_training_read_at_position(
            bam,
            chrom,
            pos,
            seen_read_ids,
            min_mapq,
            min_read_length,
            prob_threshold,
            mode,
        )
        if fiber_read is not None:
            sampled.append(fiber_read)

    return attempts


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

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        # Get reference lengths from header
        ref_lengths = dict(zip(bam.references, bam.lengths))
        sampling_index = _training_sampling_index(ref_lengths)

        # Over-sample positions since not all will yield valid reads
        n_attempts = n_samples * 20
        max_attempts = n_samples * 100
        attempts = 0

        while len(sampled) < n_samples and attempts < max_attempts:
            # Generate batch of random positions
            batch_size = min(n_attempts, n_samples * 5)
            random_positions = np.random.randint(
                0, sampling_index.total_length, size=batch_size,
            )

            attempts += _sample_training_reads_for_positions(
                bam,
                random_positions,
                sampled,
                n_samples,
                sampling_index.chroms,
                sampling_index.cum_lengths,
                seen_read_ids,
                min_mapq,
                min_read_length,
                prob_threshold,
                mode,
            )

    return sampled


def _sample_training_reads_from_file(
    bam_file: str,
    reads_per_file: int,
    seed: int,
    mode: str,
    sample_kwargs: dict,
) -> list:
    file_reads = sample_reads_indexed(
        bam_file, reads_per_file, seed,
        mode=mode, **sample_kwargs
    )
    print(f"  Sampled {len(file_reads)} from {os.path.basename(bam_file)}")
    return file_reads


def sample_reads(bam_files: list, read_count: int, seed: int,
                 mode: str = 'pacbio-fiber', **kwargs) -> list:
    """Sample reads from BAM files for training using index-based sampling."""
    np.random.seed(seed)

    all_reads = []
    reads_per_file = _reads_per_training_file(read_count, len(bam_files))

    for bam_file in tqdm(bam_files, desc="Sampling reads"):
        file_reads = _sample_training_reads_from_file(
            bam_file, reads_per_file, seed, mode, kwargs,
        )
        all_reads.extend(file_reads)

    # Shuffle and truncate
    np.random.shuffle(all_reads)
    return _top_up_training_reads(all_reads, read_count)


def _top_up_training_reads(reads: list, read_count: int) -> list:
    # If we need more reads and have some, sample with replacement.
    if len(reads) < read_count and len(reads) > 0:
        extra_needed = read_count - len(reads)
        extra_indices = np.random.choice(len(reads), extra_needed, replace=True)
        reads.extend([reads[i] for i in extra_indices])
    return reads[:read_count]


def _shuffled_training_arrays(encoded_reads: list, n_iterations: int) -> dict:
    train_arrays = {}
    for i in range(n_iterations):
        np.random.seed(i)
        indices = np.random.permutation(len(encoded_reads))
        shuffled = [encoded_reads[j] for j in indices]
        train_arrays[i] = np.concatenate(shuffled).astype(int)
    return train_arrays


def _training_strand_for_read(fiber_read, mode: str) -> str:
    if mode == 'daf':
        return detect_daf_strand(
            fiber_read.query_sequence,
            fiber_read.m6a_query_positions,
        )
    return '.'


def _encode_training_read(
    fiber_read,
    edge_trim: int,
    mode: str,
    context_size: int,
) -> np.ndarray:
    strand = _training_strand_for_read(fiber_read, mode)
    return encode_from_query_sequence(
        fiber_read.query_sequence,
        fiber_read.m6a_query_positions,
        edge_trim,
        mode=mode,
        strand=strand,
        context_size=context_size,
        is_reverse=getattr(fiber_read, 'is_reverse', False),
    )


def generate_training_arrays(reads: list, edge_trim: int,
                             n_iterations: int, mode: str = 'pacbio-fiber',
                             context_size: int = 3) -> _TrainingArrays:
    """
    Generate training arrays from sampled reads.
    Returns encoded training arrays plus read IDs and the reads they came from.
    """
    print(f"Encoding {len(reads)} reads (context size k={context_size})...")

    encoded_reads = []
    train_rids = []
    valid_reads = []  # Keep track of which reads were successfully encoded

    for fiber_read in tqdm(reads, desc="Encoding"):
        encoded = _encode_training_read(
            fiber_read,
            edge_trim,
            mode,
            context_size,
        )

        if len(encoded) > 0:
            encoded_reads.append(encoded)
            train_rids.append(fiber_read.read_id)
            valid_reads.append(fiber_read)

    print(f"Successfully encoded {len(encoded_reads)} reads")

    if len(encoded_reads) == 0:
        raise ValueError("No reads successfully encoded! Check input BAM files.")

    # Create shuffled training arrays
    train_arrays = _shuffled_training_arrays(encoded_reads, n_iterations)

    return _TrainingArrays(
        train_arrays=train_arrays,
        train_rids=train_rids,
        encoded_reads=encoded_reads,
        valid_reads=valid_reads,
    )


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

    print("\nBest model selected")
    print(f"Start probabilities: {best_model.startprob_}")
    print(f"Transition matrix:\n{best_model.transmat_}")

    return best_model, models


def _state_runs(states):
    """Yield ``(start, end, state)`` runs from a Viterbi state sequence."""
    if len(states) == 0:
        return []

    runs = []
    current_state = states[0]
    start = 0
    for i in range(1, len(states)):
        if states[i] != current_state:
            runs.append((start, i, current_state))
            current_state = states[i]
            start = i
    runs.append((start, len(states), current_state))
    return runs


def _state_run_lengths(states):
    footprint_sizes = []
    msp_sizes = []
    for start, end, state in _state_runs(states):
        length = end - start
        if state == 0:
            footprint_sizes.append(length)
        else:
            msp_sizes.append(length)
    return footprint_sizes, msp_sizes


def _state_block_specs(states) -> list:
    return [
        (start, end - start, 'forestgreen' if state == 0 else 'white')
        for start, end, state in _state_runs(states)
    ]


def _expected_state_duration(stay_prob: float) -> float:
    if stay_prob >= 1:
        return float('inf')
    return 1 / (1 - stay_prob)


def _expected_model_durations(transmat: np.ndarray) -> tuple:
    return (
        _expected_state_duration(float(transmat[0, 0])),
        _expected_state_duration(float(transmat[1, 1])),
    )


def _viterbi_state_size_stats(
    model: FiberHMM,
    encoded_reads: list,
) -> _ViterbiStateSizeStats:
    all_footprint_sizes = []
    all_msp_sizes = []
    all_states = []

    for encoded in encoded_reads:
        if len(encoded) == 0:
            continue
        states = model.predict(encoded)
        all_states.append(states)
        fp_sizes, msp_sizes = _state_run_lengths(states)
        all_footprint_sizes.extend(fp_sizes)
        all_msp_sizes.extend(msp_sizes)

    return _ViterbiStateSizeStats(
        all_footprint_sizes,
        all_msp_sizes,
        all_states,
    )


def _nonzero_emissions_by_state(emission_probs: np.ndarray) -> tuple:
    return (
        emission_probs[0, :][emission_probs[0, :] > 0],
        emission_probs[1, :][emission_probs[1, :] > 0],
    )


def _training_zoom_window_starts(seq_len: int, window_size: int,
                                 n_windows: int, seed: int) -> list:
    np.random.seed(seed)

    if seq_len >= window_size * n_windows:
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
        return window_starts

    window_starts = [
        int(seq_len * i / (n_windows + 1))
        for i in range(1, n_windows + 1)
    ]
    return [max(0, min(start, seq_len - window_size)) for start in window_starts]


def _training_zoom_window_bounds(seq_len: int, window_size: int,
                                 n_windows: int, seed: int) -> list:
    return [
        (
            start,
            min(start + window_size, seq_len),
            min(window_size, seq_len - start),
        )
        for start in _training_zoom_window_starts(
            seq_len, window_size, n_windows, seed,
        )
    ]


def _relative_positions_in_window(positions: list, start: int, end: int) -> list:
    return [position - start for position in positions if start <= position < end]


def _training_size_summary(total_label: str, size_label: str, sizes: list) -> str:
    if len(sizes) == 0:
        return ''
    return (
        f"  Total {total_label}: {len(sizes):,}\n"
        f"  {size_label}: mean={np.mean(sizes):.1f}, "
        f"median={np.median(sizes):.1f}, "
        f"range={min(sizes)}-{max(sizes)}\n"
    )


def _training_example_plot_data(model, read, encoded):
    seq_len = len(read.query_sequence)
    m6a_positions = sorted(read.m6a_query_positions)
    posteriors = model.predict_proba(encoded)
    footprint_prob = posteriors[:, 0]
    return seq_len, m6a_positions, footprint_prob


def _training_example_title(idx: int, read, seq_len: int,
                            m6a_positions: list) -> str:
    return (
        f'Example Read {idx + 1}: {read.read_id[:50]}...\n'
        f'Length: {seq_len:,}bp | '
        f'Chromosome: {read.chrom}:{read.ref_start:,}-{read.ref_end:,} | '
        f'm6A calls: {len(m6a_positions):,}'
    )


def _add_training_zoom_highlight(overview_axes: tuple, start: int, end: int,
                                 color: str) -> None:
    for ax in overview_axes:
        ax.axvspan(start, end, alpha=0.15, color=color)


def _plot_training_probability_area(
    ax,
    footprint_prob,
    *,
    show_line: bool = False,
    threshold_alpha: Optional[float] = None,
) -> None:
    ax.fill_between(
        range(len(footprint_prob)),
        0,
        footprint_prob,
        color='forestgreen',
        alpha=0.4,
        step='mid',
    )
    if show_line:
        ax.plot(
            range(len(footprint_prob)),
            footprint_prob,
            color='forestgreen',
            linewidth=0.5,
            alpha=0.8,
        )
    threshold_kwargs = {
        'color': 'gray',
        'linestyle': '--',
        'linewidth': 0.5,
    }
    if threshold_alpha is not None:
        threshold_kwargs['alpha'] = threshold_alpha
    ax.axhline(0.5, **threshold_kwargs)


def _add_training_state_blocks(
    ax,
    region_states,
    region_len: int,
    rectangle_cls,
    patch_collection_cls,
) -> None:
    patches = []
    colors = []
    for start, width, color in _state_block_specs(region_states):
        patches.append(rectangle_cls((start, 0), width, 1))
        colors.append(color)

    if patches:
        collection = patch_collection_cls(
            patches,
            facecolors=colors,
            edgecolors='lightgray',
            linewidths=0.3,
        )
        ax.add_collection(collection)
    ax.set_xlim(0, region_len)
    ax.set_ylim(0, 1)


def _plot_training_size_distribution(
    ax,
    sizes: list,
    color: str,
    x_label: str,
    title_prefix: str,
    include_nucleosome_marker: bool = False,
) -> None:
    if len(sizes) == 0:
        return

    bins = np.arange(0, min(500, max(sizes) + 10), 10)
    median_size = np.median(sizes)
    ax.hist(sizes, bins=bins, color=color, alpha=0.7, edgecolor='white')
    ax.axvline(
        median_size,
        color='black',
        linestyle='--',
        label=f'Median: {median_size:.0f}bp',
    )
    if include_nucleosome_marker:
        ax.axvline(
            147,
            color='blue',
            linestyle=':',
            alpha=0.7,
            label='Nucleosome (147bp)',
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel('Count')
    ax.set_title(f'{title_prefix} (n={len(sizes):,})')
    ax.legend()
    ax.set_xlim(0, 500)


def _plot_training_emission_distribution(ax, emission_probs: np.ndarray) -> None:
    fp_nonzero, msp_nonzero = _nonzero_emissions_by_state(emission_probs)

    bins = np.linspace(0, 1, 51)
    ax.hist(
        fp_nonzero,
        bins=bins,
        alpha=0.6,
        label=f'Footprint (n={len(fp_nonzero):,})',
        color='firebrick',
    )
    ax.hist(
        msp_nonzero,
        bins=bins,
        alpha=0.6,
        label=f'Accessible (n={len(msp_nonzero):,})',
        color='forestgreen',
    )
    ax.set_xlabel('P(methylation | state, context)')
    ax.set_ylabel('Number of contexts')
    ax.set_title('Emission Probability Distribution')
    ax.legend()


def _plot_training_transition_matrix(ax, trans: np.ndarray) -> None:
    ax.imshow(trans, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Footprint', 'Accessible'])
    ax.set_yticklabels(['Footprint', 'Accessible'])
    ax.set_xlabel('To State')
    ax.set_ylabel('From State')
    ax.set_title('Transition Probabilities')

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f'{trans[i, j]:.4f}',
                ha='center',
                va='center',
                color='white' if trans[i, j] > 0.5 else 'black',
                fontsize=12,
            )

    expected_fp, expected_msp = _expected_model_durations(trans)
    ax.text(
        0.5,
        -0.3,
        (
            f'Expected durations: Footprint={expected_fp:.0f}bp, '
            f'MSP={expected_msp:.0f}bp'
        ),
        transform=ax.transAxes,
        ha='center',
        fontsize=10,
    )


def _training_stats_paths(output_dir: str) -> dict:
    plots_dir = os.path.join(output_dir, 'plots')
    return {
        'plots_dir': plots_dir,
        'pdf': os.path.join(plots_dir, 'training_stats.pdf'),
        'summary': os.path.join(plots_dir, 'training_stats.txt'),
    }


def _write_training_stats_summary(summary_path: str, model: FiberHMM,
                                  emission_probs: np.ndarray,
                                  sampled_reads: list,
                                  all_footprint_sizes: list,
                                  all_msp_sizes: list) -> None:
    with open(summary_path, 'w') as f:
        f.write("FiberHMM Training Statistics\n")
        f.write("=" * 60 + "\n\n")

        f.write("Transition Matrix:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Footprint \u2192 Footprint: {model.transmat_[0, 0]:.6f}\n")
        f.write(f"  Footprint \u2192 Accessible: {model.transmat_[0, 1]:.6f}\n")
        f.write(f"  Accessible \u2192 Footprint: {model.transmat_[1, 0]:.6f}\n")
        f.write(f"  Accessible \u2192 Accessible: {model.transmat_[1, 1]:.6f}\n")

        expected_fp, expected_msp = _expected_model_durations(model.transmat_)
        f.write(f"\n  Expected footprint duration: {expected_fp:.1f} bp\n")
        f.write(f"  Expected MSP duration: {expected_msp:.1f} bp\n")

        f.write("\nEmission Probabilities:\n")
        f.write("-" * 40 + "\n")
        fp_nonzero, msp_nonzero = _nonzero_emissions_by_state(emission_probs)
        f.write(
            f"  Footprint contexts: {len(fp_nonzero):,} "
            f"(mean={np.mean(fp_nonzero):.4f}, "
            f"median={np.median(fp_nonzero):.4f})\n"
        )
        f.write(
            f"  Accessible contexts: {len(msp_nonzero):,} "
            f"(mean={np.mean(msp_nonzero):.4f}, "
            f"median={np.median(msp_nonzero):.4f})\n"
        )

        f.write("\nTraining Data Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Reads used: {len(sampled_reads)}\n")
        f.write(
            _training_size_summary(
                'footprints', 'Footprint sizes', all_footprint_sizes,
            )
        )
        f.write(_training_size_summary('MSPs', 'MSP sizes', all_msp_sizes))


def _save_training_model_parameter_page(
    plt,
    pdf,
    model: FiberHMM,
    emission_probs: np.ndarray,
    all_footprint_sizes: list,
    all_msp_sizes: list,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    try:
        fig.suptitle('FiberHMM Training Results', fontsize=14, fontweight='bold')

        _plot_training_transition_matrix(axes[0, 0], model.transmat_)
        _plot_training_emission_distribution(axes[0, 1], emission_probs)
        _plot_training_size_distribution(
            axes[1, 0],
            all_footprint_sizes,
            color='firebrick',
            x_label='Footprint Size (bp)',
            title_prefix='Footprint Sizes',
            include_nucleosome_marker=True,
        )
        _plot_training_size_distribution(
            axes[1, 1],
            all_msp_sizes,
            color='forestgreen',
            x_label='MSP Size (bp)',
            title_prefix='MSP Sizes',
        )

        plt.tight_layout()
        pdf.savefig(fig)
    finally:
        plt.close(fig)


def _save_training_example_png(
    plt,
    plots_dir: str,
    model: FiberHMM,
    read,
    states,
    encoded,
    rectangle_cls,
    patch_collection_cls,
) -> str:
    seq_len, m6a_positions, footprint_prob = _training_example_plot_data(
        model, read, encoded,
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 6),
        gridspec_kw={'height_ratios': [1, 1.5, 1]},
    )
    png_path = os.path.join(plots_dir, 'example_read.png')
    try:
        ax = axes[0]
        if len(m6a_positions) > 0:
            ax.eventplot(
                [m6a_positions],
                colors='purple',
                lineoffsets=0.5,
                linelengths=0.8,
                linewidths=0.3,
            )
        ax.set_xlim(0, seq_len)
        ax.set_ylim(0, 1)
        ax.set_ylabel('m6A')
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_title(f'Example: {read.read_id[:40]}... ({seq_len:,}bp)', fontsize=10)

        ax = axes[1]
        _add_training_state_blocks(
            ax, states, seq_len, rectangle_cls, patch_collection_cls,
        )
        ax.set_ylabel('State')
        ax.set_yticks([])
        ax.set_xticklabels([])

        ax = axes[2]
        _plot_training_probability_area(ax, footprint_prob)
        ax.set_xlim(0, seq_len)
        ax.set_ylim(0, 1)
        ax.set_ylabel('P(FP)')
        ax.set_xlabel('Position (bp)')

        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
    finally:
        plt.close(fig)
    return png_path


def _training_example_pdf_layout(plt, idx: int, read, seq_len, m6a_positions):
    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(
        6,
        3,
        height_ratios=[0.8, 1.2, 0.8, 0.8, 1.2, 0.8],
        hspace=0.35,
        wspace=0.25,
        left=0.06,
        right=0.98,
        top=0.93,
        bottom=0.04,
    )
    fig.suptitle(
        _training_example_title(idx, read, seq_len, m6a_positions),
        fontsize=11,
        y=0.98,
    )
    return fig, gs


def _add_training_example_overview_pdf_panels(
    fig,
    gs,
    seq_len,
    m6a_positions,
    footprint_prob,
    states,
    rectangle_cls,
    patch_collection_cls,
    patch_cls,
):
    ax_overview_m6a = fig.add_subplot(gs[0, :])
    if len(m6a_positions) > 0:
        ax_overview_m6a.eventplot(
            [m6a_positions],
            colors='purple',
            lineoffsets=0.5,
            linelengths=0.8,
            linewidths=0.3,
        )
    ax_overview_m6a.set_xlim(0, seq_len)
    ax_overview_m6a.set_ylim(0, 1)
    ax_overview_m6a.set_ylabel('m6A', fontsize=9)
    ax_overview_m6a.set_yticks([])
    ax_overview_m6a.set_title('Overview: m6A Positions', fontsize=10, loc='left')
    ax_overview_m6a.set_xticklabels([])

    ax_overview_fp = fig.add_subplot(gs[1, :])
    _add_training_state_blocks(
        ax_overview_fp, states, seq_len, rectangle_cls, patch_collection_cls,
    )
    ax_overview_fp.set_ylabel('State', fontsize=9)
    ax_overview_fp.set_yticks([])
    ax_overview_fp.set_title(
        'Overview: Footprints (green) vs Accessible (white)',
        fontsize=10,
        loc='left',
    )
    ax_overview_fp.set_xticklabels([])
    legend_elements = [
        patch_cls(facecolor='forestgreen', edgecolor='gray', label='Footprint'),
        patch_cls(facecolor='white', edgecolor='gray', label='Accessible'),
    ]
    ax_overview_fp.legend(handles=legend_elements, loc='upper right', fontsize=8)

    ax_overview_prob = fig.add_subplot(gs[2, :])
    _plot_training_probability_area(
        ax_overview_prob, footprint_prob, threshold_alpha=0.5,
    )
    ax_overview_prob.set_xlim(0, seq_len)
    ax_overview_prob.set_ylim(0, 1)
    ax_overview_prob.set_ylabel('P(FP)', fontsize=9)
    ax_overview_prob.set_yticks([0, 0.5, 1])
    ax_overview_prob.tick_params(axis='y', labelsize=7)
    ax_overview_prob.set_xlabel('Position (bp)', fontsize=9)
    ax_overview_prob.set_title(
        'Overview: Footprint Probability', fontsize=10, loc='left',
    )

    return ax_overview_m6a, ax_overview_fp, ax_overview_prob


def _add_training_example_zoom_pdf_panel(
    fig,
    gs,
    w_idx: int,
    window,
    m6a_positions,
    states,
    footprint_prob,
    overview_axes,
    rectangle_cls,
    patch_collection_cls,
):
    w_start, w_end, w_len = window
    ax_zoom_m6a = fig.add_subplot(gs[3, w_idx])
    m6a_in_window = _relative_positions_in_window(m6a_positions, w_start, w_end)
    if len(m6a_in_window) > 0:
        ax_zoom_m6a.eventplot(
            [m6a_in_window],
            colors='purple',
            lineoffsets=0.5,
            linelengths=0.8,
            linewidths=1.0,
        )
    ax_zoom_m6a.set_xlim(0, w_len)
    ax_zoom_m6a.set_ylim(0, 1)
    ax_zoom_m6a.set_ylabel('m6A', fontsize=8)
    ax_zoom_m6a.set_yticks([])
    ax_zoom_m6a.set_xticklabels([])
    ax_zoom_m6a.set_title(
        f'Zoom {w_idx + 1}: {w_start:,}-{w_end:,}bp', fontsize=9,
    )

    _add_training_zoom_highlight(
        overview_axes,
        w_start,
        w_end,
        _TRAINING_ZOOM_COLORS[w_idx],
    )

    ax_zoom_fp = fig.add_subplot(gs[4, w_idx])
    window_states = states[w_start:w_end]
    _add_training_state_blocks(
        ax_zoom_fp, window_states, w_len, rectangle_cls, patch_collection_cls,
    )
    ax_zoom_fp.set_ylabel('State', fontsize=8)
    ax_zoom_fp.set_yticks([])
    ax_zoom_fp.set_xticklabels([])

    ax_zoom_prob = fig.add_subplot(gs[5, w_idx])
    window_prob = footprint_prob[w_start:w_end]
    _plot_training_probability_area(
        ax_zoom_prob,
        window_prob,
        show_line=True,
        threshold_alpha=0.5,
    )
    ax_zoom_prob.set_xlim(0, w_len)
    ax_zoom_prob.set_ylim(0, 1)
    ax_zoom_prob.set_ylabel('P(FP)', fontsize=8)
    ax_zoom_prob.set_yticks([0, 0.5, 1])
    ax_zoom_prob.tick_params(axis='y', labelsize=7)
    ax_zoom_prob.set_xlabel('Position (bp)', fontsize=8)


def _save_training_example_pdf_page(
    plt,
    pdf,
    model: FiberHMM,
    read,
    states,
    encoded,
    idx: int,
    rectangle_cls,
    patch_collection_cls,
    patch_cls,
) -> None:
    seq_len, m6a_positions, footprint_prob = _training_example_plot_data(
        model, read, encoded,
    )

    fig, gs = _training_example_pdf_layout(
        plt, idx, read, seq_len, m6a_positions,
    )
    try:
        overview_axes = _add_training_example_overview_pdf_panels(
            fig,
            gs,
            seq_len,
            m6a_positions,
            footprint_prob,
            states,
            rectangle_cls,
            patch_collection_cls,
            patch_cls,
        )

        zoom_windows = _training_zoom_window_bounds(
            seq_len, window_size=1000, n_windows=3, seed=idx,
        )

        for w_idx, window in enumerate(zoom_windows):
            _add_training_example_zoom_pdf_panel(
                fig,
                gs,
                w_idx,
                window,
                m6a_positions,
                states,
                footprint_prob,
                overview_axes,
                rectangle_cls,
                patch_collection_cls,
            )

        pdf.savefig(fig)
    finally:
        plt.close(fig)


def _save_training_stats_pdf_pages(
    plt,
    pdf,
    model: FiberHMM,
    emission_probs: np.ndarray,
    sampled_reads: list,
    encoded_reads: list,
    all_states: list,
    all_footprint_sizes: list,
    all_msp_sizes: list,
    n_examples: int,
    rectangle_cls,
    patch_collection_cls,
    patch_cls,
) -> int:
    _save_training_model_parameter_page(
        plt,
        pdf,
        model,
        emission_probs,
        all_footprint_sizes,
        all_msp_sizes,
    )

    n_to_plot = min(n_examples, len(sampled_reads), len(all_states))
    for idx in range(n_to_plot):
        _save_training_example_pdf_page(
            plt,
            pdf,
            model,
            sampled_reads[idx],
            all_states[idx],
            encoded_reads[idx],
            idx,
            rectangle_cls,
            patch_collection_cls,
            patch_cls,
        )
    return n_to_plot


def _save_training_stats_example_png(
    plt,
    plots_dir: str,
    model: FiberHMM,
    sampled_reads: list,
    encoded_reads: list,
    all_states: list,
    rectangle_cls,
    patch_collection_cls,
) -> None:
    if len(sampled_reads) == 0 or len(all_states) == 0:
        return

    png_path = _save_training_example_png(
        plt,
        plots_dir,
        model,
        sampled_reads[0],
        all_states[0],
        encoded_reads[0],
        rectangle_cls,
        patch_collection_cls,
    )
    print(f"  Saved: {png_path}")


def _write_training_stats_summary_report(
    summary_path: str,
    model: FiberHMM,
    emission_probs: np.ndarray,
    sampled_reads: list,
    all_footprint_sizes: list,
    all_msp_sizes: list,
) -> None:
    _write_training_stats_summary(
        summary_path,
        model,
        emission_probs,
        sampled_reads,
        all_footprint_sizes,
        all_msp_sizes,
    )
    print(f"  Saved: {summary_path}")


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
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Patch, Rectangle
    except ImportError:
        print("  Warning: matplotlib not installed. Skipping stats plots.")
        return

    paths = _training_stats_paths(output_dir)
    plots_dir = paths['plots_dir']
    os.makedirs(plots_dir, exist_ok=True)

    pdf_path = paths['pdf']

    # Run Viterbi on all encoded reads to get footprint statistics
    print("  Running Viterbi on training reads...")
    viterbi_stats = _viterbi_state_size_stats(model, encoded_reads)

    with PdfPages(pdf_path) as pdf:
        _save_training_stats_pdf_pages(
            plt,
            pdf,
            model,
            emission_probs,
            sampled_reads,
            encoded_reads,
            viterbi_stats.states,
            viterbi_stats.footprint_sizes,
            viterbi_stats.msp_sizes,
            n_examples,
            Rectangle,
            PatchCollection,
            Patch,
        )

    print(f"  Saved: {pdf_path}")

    # Also save a standalone PNG of the first example (simplified version)
    _save_training_stats_example_png(
        plt,
        plots_dir,
        model,
        sampled_reads,
        encoded_reads,
        viterbi_stats.states,
        Rectangle,
        PatchCollection,
    )

    # Write text summary
    _write_training_stats_summary_report(
        paths['summary'],
        model,
        emission_probs,
        sampled_reads,
        viterbi_stats.footprint_sizes,
        viterbi_stats.msp_sizes,
    )


def _model_json_record(model):
    return {
        'n_states': model.n_states,
        'startprob': model.startprob_.tolist(),
        'transmat': model.transmat_.tolist(),
        'emissionprob': model.emissionprob_.tolist(),
    }


def _training_config(args):
    config = {
        'context_size': args.context_size,
        'mode': args.mode,
        'edge_trim': args.edge_trim,
        'prob_adjust': args.prob_adjust,
    }
    if args.base_model:
        config['base_model'] = args.base_model
    return config


def _training_output_paths(outdir: str) -> dict:
    return {
        'best_json': os.path.join(outdir, 'best-model.json'),
        'best_npz': os.path.join(outdir, 'best-model.npz'),
        'all_models': os.path.join(outdir, 'all_models.json'),
        'train_reads': os.path.join(outdir, 'training-reads.tsv'),
        'config': os.path.join(outdir, 'model_config.json'),
    }


def _save_training_outputs(best_model, all_models, args, train_rids) -> None:
    print(f"\nSaving to {args.outdir}")
    paths = _training_output_paths(args.outdir)

    # Save best model in JSON format (recommended - portable, human-readable)
    save_model(
        best_model,
        paths['best_json'],
        context_size=args.context_size,
        mode=args.mode
    )
    print("  Saved: best-model.json (recommended)")

    # Also save in NPZ for backwards compatibility
    save_model(
        best_model,
        paths['best_npz'],
        context_size=args.context_size,
        mode=args.mode
    )
    print("  Saved: best-model.npz (numpy format)")

    # Save all models as JSON list
    all_models_data = [_model_json_record(m) for m in all_models]
    with open(paths['all_models'], 'w') as f:
        json.dump(all_models_data, f)

    # Save training reads (only if we did training)
    if train_rids:
        pd.DataFrame({'rid': train_rids}).to_csv(
            paths['train_reads'],
            sep='\t', index=False
        )

    # Save config (JSON - human readable)
    with open(paths['config'], 'w') as f:
        json.dump(_training_config(args), f, indent=2)


def _build_model_from_base(base_model_path: str, emission_probs: np.ndarray,
                           context_size: int):
    print(f"\nLoading base model: {base_model_path}")
    base_model = load_model(base_model_path, normalize=False)

    # Validate emission dimensions match
    if base_model.emissionprob_.shape[1] != emission_probs.shape[1]:
        print("Error: Emission size mismatch!")
        print(f"  Base model: {base_model.emissionprob_.shape[1]} observations")
        print(f"  New emissions: {emission_probs.shape[1]} observations")
        print(f"  Check that context_size matches (k={context_size})")
        sys.exit(1)

    # Create new model with base model's transitions and new emissions
    best_model = FiberHMM(n_states=2)
    best_model.startprob_ = base_model.startprob_.copy()
    best_model.transmat_ = base_model.transmat_.copy()
    best_model.emissionprob_ = emission_probs

    print(f"  Kept startprob: {best_model.startprob_}")
    print(f"  Kept transmat:\n{best_model.transmat_}")
    print(f"  Replaced emissions: {emission_probs.shape}")

    return best_model, [best_model]


def _print_training_header(args) -> None:
    mode_desc = mode_description(args.mode)
    context_mer = 2 * args.context_size + 1
    n_codes = ContextEncoder.get_n_codes(args.context_size)

    if args.base_model:
        print("FiberHMM Model Builder (using base model)")
        print(f"  Mode: {args.mode} ({mode_desc})")
        print(f"  Context: k={args.context_size} ({context_mer}-mer, {n_codes:,} codes)")
        print(f"  Base model: {args.base_model}")
        print("  (Transitions from base model, new emissions from prob files)")
        return

    print("FiberHMM Model Training v2")
    print(f"  Mode: {args.mode} ({mode_desc})")
    print(f"  Context: k={args.context_size} ({context_mer}-mer, {n_codes:,} codes)")
    print(f"  Input: {args.input}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Reads: {args.read_count}")
    print(f"  Seed: {args.seed}")
    print("  (No genome context file needed)")


def _load_training_emission_probs(args) -> np.ndarray:
    print("\nLoading emission probabilities...")
    emission_probs = make_emission_probs(
        args.probs[0],
        args.probs[1],
        context_size=args.context_size,
        prob_adjust=args.prob_adjust,
    )
    print(f"Emission matrix: {emission_probs.shape}")
    return emission_probs


def _run_training_or_base_model(
    args,
    emission_probs: np.ndarray,
) -> _TrainingRunResult:
    if args.base_model:
        best_model, all_models = _build_model_from_base(
            args.base_model,
            emission_probs,
            args.context_size,
        )
        return _TrainingRunResult(
            best_model=best_model,
            all_models=all_models,
            train_rids=[],
            valid_reads=[],
            encoded_reads=[],
        )

    print(f"\nSampling reads from {len(args.input)} BAM file(s)...")
    sampled = sample_reads(
        args.input,
        args.read_count,
        args.seed,
        mode=args.mode,
        min_mapq=args.min_mapq,
        prob_threshold=args.prob_threshold,
        min_read_length=args.min_read_length,
    )
    print(f"Total sampled: {len(sampled)} reads")

    arrays = generate_training_arrays(
        sampled, args.edge_trim, args.iterations, args.mode, args.context_size
    )

    print(f"\nTraining HMM ({args.iterations} iterations)...")
    best_model, all_models = train_hmm(
        emission_probs,
        arrays.train_arrays,
        args.use_hmmlearn,
    )
    return _TrainingRunResult(
        best_model=best_model,
        all_models=all_models,
        train_rids=arrays.train_rids,
        valid_reads=arrays.valid_reads,
        encoded_reads=arrays.encoded_reads,
    )


def _maybe_generate_training_stats(args, best_model, valid_reads, encoded_reads,
                                   emission_probs: np.ndarray) -> None:
    if args.stats and valid_reads and encoded_reads:
        print("\nGenerating training statistics...")
        generate_training_stats(
            best_model,
            valid_reads,
            encoded_reads,
            emission_probs,
            args.outdir,
            n_examples=args.n_examples,
            mode=args.mode,
        )
    elif args.stats and args.base_model:
        print("\nNote: --stats skipped (no training data with --base-model)")


def main():
    args = parse_args()

    # Validate arguments
    if not args.base_model and not args.input:
        print("Error: --input is required unless using --base-model")
        sys.exit(1)

    _print_training_header(args)
    os.makedirs(args.outdir, exist_ok=True)

    # Generate emission probabilities
    emission_probs = _load_training_emission_probs(args)
    training_result = _run_training_or_base_model(args, emission_probs)

    _save_training_outputs(
        training_result.best_model,
        training_result.all_models,
        args,
        training_result.train_rids,
    )

    # Generate stats if requested (only if we have training data)
    _maybe_generate_training_stats(
        args,
        training_result.best_model,
        training_result.valid_reads,
        training_result.encoded_reads,
        emission_probs,
    )

    print("Done!")


if __name__ == '__main__':
    main()
