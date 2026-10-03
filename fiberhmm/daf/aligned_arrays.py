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

try:
    from numba import njit as _numba_njit
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - numba is a core dependency
    _HAS_NUMBA = False

    def _numba_njit(*args, **kwargs):  # type: ignore[misc]
        def _wrap(fn):
            return fn
        return _wrap

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


@_numba_njit(cache=True)
def _md_walk_ends_short(md, ops, lengths):
    """pysam's ``build_alignment_sequence`` MD walk over the CIGAR layout.

    True when the walk stops before the last M/=/X/D base, so that
    ``build_reference_sequence`` indexes past the string it built.
    """
    size = 0
    for k in range(ops.size):
        op = ops[k]
        if op == 0 or op == 1 or op == 2 or op == 6 or op == 7 or op == 8:
            size += lengths[k]
    inserted = np.zeros(size, dtype=np.bool_)
    position = 0
    last_reference = -1
    for k in range(ops.size):
        op = ops[k]
        length = lengths[k]
        if op == 0 or op == 2 or op == 7 or op == 8:
            if length > 0:
                last_reference = position + length - 1
            position += length
        elif op == 1 or op == 6:
            for j in range(position, position + length):
                inserted[j] = True
            position += length
    position = 0
    matches = 0
    index = 0
    n = md.size
    while index < n:
        char = md[index]
        if 48 <= char <= 57:
            matches = matches * 10 + (char - 48)
            index += 1
            continue
        for _ in range(matches):
            while position < size and inserted[position]:
                position += 1
            position += 1
        while position < size and inserted[position]:
            position += 1
        matches = 0
        index += 1
        if char == 94:
            while index < n and 65 <= md[index] <= 90:
                position += 1
                index += 1
        else:
            position += 1
    for _ in range(matches):
        while position < size and inserted[position]:
            position += 1
        position += 1
    while position < size and inserted[position]:
        position += 1
    return position <= last_reference


def md_deletion_spans_insertion(md, ops, lengths) -> bool:
    """True when an MD ``^`` deletion run covers a CIGAR insertion.

    pysam lays out M/=/X/D/I/P bases, then walks MD over them, skipping
    insertions before every MD token but not inside a ``^`` deletion run. A
    deletion run of length L starting at reference offset o therefore
    swallows an I/P operation that sits strictly between o and o + L (its
    reference offset counts the M/=/X/D bases before it); the walk ends short
    and pysam builds the rest of the reference from undefined memory, even
    though the MD length matches the CIGAR. MD is read as pysam reads it: a
    ``^`` run is the upper-case letters after it; every other non-digit
    character (except ``^``) is one reference base. ``ops``/``lengths`` are
    the CIGAR codes and lengths; a non-ASCII or non-string MD returns False
    (callers reject those on length).
    """
    if not isinstance(md, str) or "^" not in md or not md.isascii():
        return False
    ops = np.asarray(ops, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    inserted = ((ops == 1) | (ops == 6)) & (lengths > 0)
    if not inserted.any():
        return False
    md_codes = np.frombuffer(md.encode("ascii"), dtype=np.uint8)
    if _HAS_NUMBA:
        return bool(_md_walk_ends_short(md_codes, ops, lengths))
    return _deletion_run_covers(md_codes, ops, lengths, inserted)


def _deletion_run_covers(chars, ops, lengths, inserted) -> bool:
    """Vectorised form of the check (used when numba is unavailable)."""
    ref_seq = np.where(_REF_SEQ_OPS.take(ops, mode="clip"), lengths, 0)
    insertion_offsets = (np.cumsum(ref_seq) - ref_seq)[inserted]

    index = np.arange(chars.size)
    digit = (chars - 48) < 10          # uint8 arithmetic wraps below '0'
    upper = (chars - 65) < 26
    caret = chars == 94
    # A ^ run is the upper-case letters right after a '^'.
    previous_other = np.maximum.accumulate(np.where(upper, -1, index))
    deleted = upper & (previous_other >= 0) & caret[previous_other]
    if not deleted.any():
        return False
    # Reference bases consumed per character: each digit contributes its
    # place value within its run; '^' none; any other character one.
    run_end = np.minimum.accumulate(np.where(digit, chars.size, index)[::-1])[::-1]
    place = np.power(10, np.maximum(run_end - index - 1, 0), dtype=np.int64)
    consumed = np.where(digit, (chars.astype(np.int64) - 48) * place,
                        np.where(caret, 0, 1))
    offsets = np.cumsum(consumed) - consumed
    first = deleted.copy()
    first[1:] &= ~deleted[:-1]
    last = deleted.copy()
    last[:-1] &= ~deleted[1:]
    starts = offsets[first]
    ends = offsets[last] + 1
    before = np.searchsorted(starts, insertion_offsets, side="left") - 1
    return bool(((before >= 0) & (insertion_offsets < ends[np.maximum(before, 0)])).any())


def md_disagrees_with_cigar(read) -> bool:
    """True when ``read`` has an MD tag that does not describe its CIGAR.

    pysam builds the ``get_aligned_pairs(with_seq=True)`` /
    ``get_reference_sequence()`` reference from MD+CIGAR without checking
    that MD covers every M/=/X/D base: for an MD shorter than that span it
    copies whatever bytes follow the parsed MD in memory, so the "reference"
    bases of the rest of the read change from run to run (and sometimes fail
    to decode). An MD longer than the span raises instead. Callers that want
    MD-derived reference bases must treat such reads as having no usable MD
    (FASTA fallback, or skip), as ``get_daf_positions`` and the dedup
    fingerprint already do. The same holds when an MD deletion run covers a
    CIGAR insertion (:func:`md_deletion_spans_insertion`). Reads without MD,
    or objects that are not alignments, return False (the caller's own path
    handles them).
    """
    try:
        md = read.get_tag("MD")
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    try:
        cigar = read.cigartuples
    except (AttributeError, TypeError, ValueError):
        return False
    if not cigar:
        return False
    span = sum(length for op, length in cigar if op in (0, 2, 7, 8))
    if md_reference_length(md) != span:
        return True
    table = np.asarray(cigar, dtype=np.int64)
    return md_deletion_spans_insertion(md, table[:, 0], table[:, 1])


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
    if md_deletion_spans_insertion(md, ops, lengths):
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


def unaligned_query_positions(cigartuples, query_length: Optional[int] = None) -> set:
    """SEQ positions with no reference counterpart: CIGAR I and S bases.

    DAF deamination evidence is a read-versus-reference comparison, so these
    bases can never carry a mark; callers treat them as no evidence. A record
    without a CIGAR (unmapped) returns an empty set: nothing is aligned, and
    unmapped DAF input keeps its previous handling. Hard clips (H) are not in
    SEQ and are skipped. ``query_length`` bounds the result when given.
    """
    out: set = set()
    if not cigartuples:
        return out
    q = 0
    for op, length in cigartuples:
        if op == 1 or op == 4:
            out.update(range(q, q + length))
        if _QUERY_OPS[op] if 0 <= op < 10 else False:
            q += length
    if query_length is not None and q > query_length:
        out = {p for p in out if p < query_length}
    return out


__all__ = [
    "BASE_A",
    "BASE_C",
    "BASE_G",
    "BASE_R",
    "BASE_T",
    "BASE_Y",
    "matched_base_arrays",
    "md_deletion_spans_insertion",
    "md_disagrees_with_cigar",
    "md_reference_length",
    "unaligned_query_positions",
]
