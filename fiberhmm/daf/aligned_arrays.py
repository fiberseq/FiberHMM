"""Vectorised matched-base arrays for DAF mismatch scans.

Several DAF stages (the SNP screen, ``get_daf_positions`` and the dedup
fingerprint) walk ``read.get_aligned_pairs(with_seq=True)`` in Python and
compare each reference base with the query base.  On amplicon/plasmid runs
that loop dominates the wall time: one tuple per aligned base, several
``str.upper`` calls per tuple.

:func:`matched_base_arrays` returns the same information as numpy arrays.
pysam builds the ``with_seq`` reference base of every M/=/X pair as
``get_reference_sequence()[r_idx]``, where ``r_idx`` advances over M/=/X and
D operations only (see ``AlignedSegment.get_aligned_pairs`` in pysam's
``libcalignedsegment.pyx``).  This module reproduces that indexing from the
CIGAR with numpy, so the arrays are element-for-element identical to the
``(query_position, reference_position, reference_base)`` triples whose
positions are both non-None, in the same order.

Whenever the slow path could behave differently (no MD tag, malformed MD or
CIGAR, unusual CIGAR operations, sequence shorter than the CIGAR, non-ASCII
bases, duck-typed reads, no matched bases) the helper returns ``None`` and
callers run their original per-pair code unchanged, so exception handling and
reference-FASTA fallbacks are exactly as before.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np
import pysam

# Per-operation lookup tables indexed by BAM CIGAR code (0..9).  P (6) and
# B (9) are left to the slow path.
def _op_table(codes):
    table = np.zeros(10, dtype=bool)
    table[list(codes)] = True
    return table


_SUPPORTED_OPS = _op_table((0, 1, 2, 3, 4, 5, 7, 8))
_MATCH_OPS = _op_table((0, 7, 8))            # M, =, X
_QUERY_OPS = _op_table((0, 1, 4, 7, 8))      # M, I, S, =, X
_REF_POS_OPS = _op_table((0, 2, 3, 7, 8))    # M, D, N, =, X
_REF_SEQ_OPS = _op_table((0, 2, 7, 8))       # M, D, =, X (not N)

BASE_A = ord("A")
BASE_C = ord("C")
BASE_G = ord("G")
BASE_T = ord("T")
BASE_R = ord("R")
BASE_Y = ord("Y")


_MD_DIGIT_RUNS = re.compile(r"[0-9]+")
_MD_LETTERS = re.compile(r"[A-Za-z]")


def md_reference_length(md) -> Optional[int]:
    """Reference bases described by an ASCII MD string; ``None`` otherwise.

    Digit runs are matched bases; every letter consumes one reference base,
    whether a mismatch or part of a ``^`` deletion run; any other character
    consumes none (the same accounting as
    :func:`fiberhmm.daf.encoder._md_tag_ref_length`).
    """
    if not isinstance(md, str) or not md.isascii():
        return None
    return sum(map(int, _MD_DIGIT_RUNS.findall(md))) + len(_MD_LETTERS.findall(md))


def matched_base_arrays(
    read, sequence: Optional[str] = None
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return ``(qpos, rpos, ref_codes, query_codes)`` for matched bases.

    ``qpos``/``rpos`` are int64 query and reference positions of every
    M/=/X aligned base, in alignment order.  ``ref_codes`` and
    ``query_codes`` are uint8 ASCII codes of the upper-cased reference base
    (as ``get_aligned_pairs(with_seq=True)`` reports it) and the upper-cased
    query base at ``qpos``.

    Returns ``None`` when the caller must use its original slow path.
    """
    if not isinstance(read, pysam.AlignedSegment):
        return None
    if sequence is None:
        sequence = read.query_sequence
    if not sequence or not isinstance(sequence, str) or not sequence.isascii():
        return None
    cigar = read.cigartuples
    if not cigar:
        return None
    table = np.asarray(cigar, dtype=np.int64)
    ops = table[:, 0]
    lengths = table[:, 1]
    if int(ops.min()) < 0 or int(ops.max()) >= _SUPPORTED_OPS.size:
        return None
    if not _SUPPORTED_OPS[ops].all():
        return None

    query_lengths = np.where(_QUERY_OPS[ops], lengths, 0)
    ref_pos_lengths = np.where(_REF_POS_OPS[ops], lengths, 0)
    ref_seq_lengths = np.where(_REF_SEQ_OPS[ops], lengths, 0)
    ref_seq_total = int(ref_seq_lengths.sum())

    # Only an MD tag that describes exactly the CIGAR's M/=/X/D bases gives a
    # well-defined reference string.  pysam does not reject a short MD: its
    # reference string then runs past the parsed MD into undefined memory, so
    # such reads are left to the (unchanged) slow path.
    try:
        md = read.get_tag("MD")
    except KeyError:
        return None
    if md_reference_length(md) != ref_seq_total:
        return None
    try:
        reference = read.get_reference_sequence()
    except Exception:
        return None
    if not isinstance(reference, str) or not reference.isascii():
        return None
    # pysam indexes the reference string for every M/=/X and D base; running
    # past its end raises IndexError there, so leave that case to it.
    if ref_seq_total > len(reference):
        return None

    query_starts = np.cumsum(query_lengths) - query_lengths
    ref_starts = np.cumsum(ref_pos_lengths) - ref_pos_lengths + int(read.reference_start)
    ref_seq_starts = np.cumsum(ref_seq_lengths) - ref_seq_lengths

    match = _MATCH_OPS[ops] & (lengths > 0)
    block_lengths = lengths[match]
    total = int(block_lengths.sum())
    if total == 0:
        return None
    block_query_starts = query_starts[match]
    if int((block_query_starts + block_lengths).max()) > len(sequence):
        return None

    offsets = np.arange(total, dtype=np.int64) - np.repeat(
        np.cumsum(block_lengths) - block_lengths, block_lengths
    )
    qpos = np.repeat(block_query_starts, block_lengths) + offsets
    rpos = np.repeat(ref_starts[match], block_lengths) + offsets
    ref_index = np.repeat(ref_seq_starts[match], block_lengths) + offsets

    ref_codes = np.frombuffer(reference.upper().encode("ascii"), dtype=np.uint8)[ref_index]
    query_codes = np.frombuffer(sequence.upper().encode("ascii"), dtype=np.uint8)[qpos]
    return qpos, rpos, ref_codes, query_codes


__all__ = [
    "BASE_A",
    "BASE_C",
    "BASE_G",
    "BASE_R",
    "BASE_T",
    "BASE_Y",
    "matched_base_arrays",
    "md_reference_length",
]
