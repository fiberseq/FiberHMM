"""
Shared utilities for probability generation.

Contains helper functions used by generate_probs, bootstrap_probs, and transfer_probs.
"""

import os
from pathlib import Path
from typing import NamedTuple, Set, Tuple

# Reverse complement lookup
_RC_TABLE = str.maketrans('ACGT', 'TGCA')


def reverse_complement(seq: str) -> str:
    """Return reverse complement of a DNA sequence."""
    return seq.translate(_RC_TABLE)[::-1]


def detect_strand_and_base(sequence: str, mod_positions: Set[int], mode: str) -> Tuple[str, str]:
    """
    Detect strand and target base based on mode.

    Args:
        sequence: Read sequence
        mod_positions: Set of query positions with modifications
        mode: Analysis mode ('pacbio-fiber', 'nanopore-fiber', 'daf')

    Returns:
        (strand, target_base)

    For pacbio-fiber and nanopore-fiber modes, returns ('.', 'A') - A-centered, no strand.
    For daf mode, detects strand by whether modifications are at T or A positions:
        - + strand: C→T deamination, MM tag marks T positions → target_base 'C'
        - - strand: G→A deamination, MM tag marks A positions → target_base 'G'
    """
    if mode in ('pacbio-fiber', 'nanopore-fiber'):
        return '.', 'A'

    seq_upper = sequence.upper()

    if mode == 'daf':
        t_count = sum(1 for p in mod_positions if p < len(seq_upper) and seq_upper[p] == 'T')
        a_count = sum(1 for p in mod_positions if p < len(seq_upper) and seq_upper[p] == 'A')

        if t_count > a_count:
            return '+', 'C'
        elif a_count > t_count:
            return '-', 'G'
        else:
            return '.', 'C'

    return '.', 'A'


def setup_output_dirs(output_path: str) -> Tuple[Path, Path, Path]:
    """
    Create standard output directory structure (tables/, plots/).

    Args:
        output_path: Base output directory path

    Returns:
        (output_dir, tables_dir, plots_dir) as Path objects
    """
    output_dir = Path(output_path)
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    return output_dir, tables_dir, plots_dir


def get_base_name(output_path: str, default: str = "probs") -> str:
    """
    Extract base name from output path for file naming.

    Args:
        output_path: Output directory path
        default: Default name if path is empty

    Returns:
        Base name string for output files
    """
    base_name = os.path.basename(output_path.rstrip('/'))
    return base_name if base_name else default


# ---------------------------------------------------------------------------
# Shared per-read evidence extraction for the table-building / training tools
# ---------------------------------------------------------------------------

class TrainingRead(NamedTuple):
    """Modification evidence of one read, as the inference encoder sees it.

    ``sequence`` is the SEQ-frame sequence the counters/encoder consume (R/Y
    already decoded to T/A for DAF). ``strand`` is the DAF conversion strand
    (``'+'`` C->T, ``'-'`` G->A, ``'.'`` unknown) and ``'.'`` otherwise.
    ``unknown_positions`` are target bases an MM ``?`` entry left unlisted.
    """
    sequence: str
    mod_positions: Set[int]
    strand: str
    unknown_positions: Set[int]


def extract_training_read(read, mode: str, prob_threshold: int):
    """Extract a read's evidence exactly as ``fiberhmm-call`` does.

    DAF: the call engine's R/Y -> MD -> MM/ML precedence, including its
    strand-swap chimera filter. Other modes: the MM/ML parser with the mode's
    mod codes. MM/ML is only walked when ``mm_applicable(read)`` (no hard clip
    without a matching ``MN``).

    Returns a :class:`TrainingRead`, or a skip-reason string: ``'chimera'``,
    ``'no_modifications'``, ``'no_mm_tag'``, ``'no_ml_tag'``,
    ``'mm_not_applicable'``.
    """
    from fiberhmm.core.bam_reader import mm_applicable, parse_mm_tag_query_calls

    if mode == 'daf':
        from fiberhmm.inference.engine import (
            CHIMERA_SKIP,
            _extract_fiber_read_from_pysam,
        )
        fr = _extract_fiber_read_from_pysam(read, 'daf', prob_threshold)
        if fr is CHIMERA_SKIP:
            return 'chimera'
        if fr is None:
            return 'no_modifications'
        strand = fr.get('_daf_strand')
        if strand is None:
            # MM/ML-native deamination calls (no R/Y, no usable MD).
            if not mm_applicable(read):
                return 'mm_not_applicable'
            strand = detect_strand_and_base(
                fr['query_sequence'], fr['m6a_query_positions'], 'daf')[0]
        return TrainingRead(fr['query_sequence'], set(fr['m6a_query_positions']),
                            strand, set(fr.get('unknown_query_positions') or ()))

    mm_tag = ml_tag = None
    try:
        if read.has_tag('MM'):
            mm_tag = read.get_tag('MM')
        elif read.has_tag('Mm'):
            mm_tag = read.get_tag('Mm')
        if read.has_tag('ML'):
            ml_tag = read.get_tag('ML')
        elif read.has_tag('Ml'):
            ml_tag = read.get_tag('Ml')
    except KeyError:
        pass
    if mm_tag is None:
        return 'no_mm_tag'
    if ml_tag is None:
        return 'no_ml_tag'
    if not mm_applicable(read):
        return 'mm_not_applicable'
    mods, unknown = parse_mm_tag_query_calls(
        mm_tag, list(ml_tag), read.query_sequence, read.is_reverse,
        prob_threshold, mode=mode)
    return TrainingRead(read.query_sequence, mods, '.', unknown)
