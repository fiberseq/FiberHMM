"""
Shared utilities for probability generation.

Contains helper functions used by generate_probs, bootstrap_probs, and transfer_probs.
"""

import os
from pathlib import Path
from typing import Set, Tuple


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
