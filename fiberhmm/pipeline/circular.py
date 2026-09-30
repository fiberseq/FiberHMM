"""Alignments across the origin of circular references, and hard clipping.

A plasmid is circular but its FASTA is linear, so a read that runs through
the arbitrary position 0 comes back from minimap2 as two pieces: one ending at
the last base of the contig and one starting at its first base (a primary and
a supplementary record). Keeping only the primary piece, as a linear pipeline
does, throws away the rest of the molecule; for a Plasmidsaurus library about
a third of the reads cross the origin. Rotating the reference only moves the
cut.

The SAM specification (section 1.4, "Circular reference sequences") gives a
preferred representation instead: on a contig declared ``@SQ TP:circular``,
POS stays within ``1..LN`` and the alignment may run past ``LN``; a position
``p > LN`` means ``((p - 1) mod LN) + 1``. :func:`merge_origin_pieces` joins
the two pieces of an origin-spanning read into one such record:

* the pieces must be on the same strand, one must end within ``tolerance``
  bp of the contig end and the other start within ``tolerance`` bp of its
  start, and they must be consecutive on the read (query gap or overlap of at
  most ``tolerance`` bp);
* an overlap at the junction is removed from the supplementary piece and a
  small gap is filled with aligned (``M``) plus ``I``/``D`` operations; ``MD``
  and ``NM`` are recomputed against the reference, so the join itself is
  scored honestly;
* one molecule covers at most one full circle: the supplementary piece is
  trimmed at its far end so the merged span is at most ``LN`` (the rest is
  concatemer sequence, which the hard-clip step then removes, as for any
  other soft-clipped arm).

The merged record keeps the primary's flags, MAPQ, read group and full
``SEQ``; the supplementary record is dropped. Calling then sees the whole
molecule, and FiberBrowser draws it continuously across the origin in a
circular view.
"""
from __future__ import annotations

from typing import Optional, Sequence

import pysam

M, INS, DEL, REF_SKIP, SOFT, HARD, PAD, EQ, DIFF = range(9)
_QUERY_OPS = {M, INS, SOFT, EQ, DIFF}
_REF_OPS = {M, DEL, REF_SKIP, EQ, DIFF}

# Tags that describe minimap2's original alignment and are wrong after a merge
# or clip. MD/NM are recomputed; MM/ML are kept unless SEQ is trimmed.
_ALIGNMENT_TAGS = ("SA", "MD", "NM", "AS", "ms", "nn", "tp", "cm", "s1", "s2",
                   "de", "dv", "rl", "zd", "cs")


def aligned_ops(cigar: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """CIGAR without clipping operations."""
    return [(op, n) for op, n in cigar if op not in (SOFT, HARD) and n > 0]


def _normalise(cigar: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for op, n in cigar:
        if n <= 0:
            continue
        if out and out[-1][0] == op:
            out[-1] = (op, out[-1][1] + n)
        else:
            out.append((op, n))
    return out


def trim_front(ops: list[tuple[int, int]], min_query: int, min_ref: int
               ) -> tuple[list[tuple[int, int]], int, int]:
    """Drop leading bases until at least ``min_query`` query and ``min_ref``
    reference bases were removed, ending on an aligned base boundary.

    Returns ``(ops, query_removed, ref_removed)``.
    """
    ops = list(ops)
    q = r = 0
    while ops and (q < min_query or r < min_ref or ops[0][0] in (INS, DEL, REF_SKIP)):
        op, n = ops[0]
        if op in (M, EQ, DIFF):
            need = max(min_query - q, min_ref - r, 1)
            take = min(n, need)
            q += take
            r += take
            ops[0] = (op, n - take)
        elif op == INS:
            q += n
            ops[0] = (op, 0)
        else:  # DEL / REF_SKIP
            r += n
            ops[0] = (op, 0)
        if ops[0][1] == 0:
            ops.pop(0)
    return ops, q, r


def trim_back(ops: list[tuple[int, int]], min_query: int, min_ref: int
              ) -> tuple[list[tuple[int, int]], int, int]:
    rev, q, r = trim_front(list(reversed(ops)), min_query, min_ref)
    return list(reversed(rev)), q, r


def _span(ops) -> tuple[int, int]:
    q = sum(n for op, n in ops if op in _QUERY_OPS)
    r = sum(n for op, n in ops if op in _REF_OPS)
    return q, r


def compute_md_nm(query: str, cigar: Sequence[tuple[int, int]], ref: str, pos: int
                  ) -> tuple[str, int]:
    """``MD``/``NM`` of an alignment, reading the reference circularly.

    ``pos`` is the 0-based reference start; positions past ``len(ref)`` wrap.
    """
    length = len(ref)
    md: list[str] = []
    matches = 0
    nm = 0
    q = 0
    r = pos
    for op, n in cigar:
        if op in (M, EQ, DIFF):
            for i in range(n):
                rb = ref[(r + i) % length]
                qb = query[q + i]
                if rb == qb and rb != "N":
                    matches += 1
                else:
                    md.append(str(matches))
                    md.append(rb)
                    matches = 0
                    nm += 1
            q += n
            r += n
        elif op == INS:
            q += n
            nm += n
        elif op == DEL:
            md.append(str(matches))
            md.append("^" + "".join(ref[(r + i) % length] for i in range(n)))
            matches = 0
            r += n
            nm += n
        elif op == REF_SKIP:
            r += n
        elif op == SOFT:
            q += n
    md.append(str(matches))
    return "".join(md), nm


def _query_window(read: pysam.AlignedSegment) -> tuple[int, int, int, int]:
    """(leading soft clip, aligned query start, aligned query end, SEQ length)."""
    cigar = read.cigartuples or []
    lead = cigar[0][1] if cigar and cigar[0][0] == SOFT else 0
    hard_lead = cigar[0][1] if cigar and cigar[0][0] == HARD else 0
    q_len, _ = _span(aligned_ops(cigar))
    start = hard_lead + lead
    total = sum(n for op, n in cigar if op in _QUERY_OPS or op == HARD)
    return lead, start, start + q_len, total


def merge_origin_pieces(primary: pysam.AlignedSegment,
                        supplementary: pysam.AlignedSegment,
                        reference: str,
                        tolerance: int = 100) -> Optional[pysam.AlignedSegment]:
    """Join the two pieces of an origin-spanning read (see the module docstring).

    Returns a new record anchored on the piece that ends at the contig end, or
    ``None`` when the two records are not the pieces of one crossing.
    """
    length = len(reference)
    if (primary.is_unmapped or supplementary.is_unmapped
            or primary.reference_id != supplementary.reference_id
            or primary.is_reverse != supplementary.is_reverse
            or primary.query_sequence is None):
        return None
    if primary.reference_end is None or supplementary.reference_end is None:
        return None
    # A = the piece at the end of the contig, B = the piece at its start.
    if (primary.reference_end >= length - tolerance
            and supplementary.reference_start <= tolerance):
        a, b, a_is_primary = primary, supplementary, True
    elif (supplementary.reference_end >= length - tolerance
          and primary.reference_start <= tolerance):
        a, b, a_is_primary = supplementary, primary, False
    else:
        return None

    seq = primary.query_sequence
    _, a_q0, a_q1, a_total = _query_window(a)
    _, b_q0, b_q1, b_total = _query_window(b)
    if a_total != len(seq) or b_total != len(seq):
        return None  # hard-clipped piece: query coordinates are not comparable
    # Consecutive on the read, in SEQ (reference-forward) orientation.
    if b_q0 - a_q1 > tolerance or a_q1 - b_q0 > tolerance:
        return None
    a_ops = aligned_ops(a.cigartuples)
    b_ops = aligned_ops(b.cigartuples)
    a_r0, a_r1 = a.reference_start, a.reference_end
    b_r0, b_r1 = b.reference_start + length, b.reference_end + length  # unrolled

    # Remove any junction overlap from the supplementary piece.
    q_over = max(0, a_q1 - b_q0)
    r_over = max(0, a_r1 - b_r0)
    if q_over or r_over:
        if a_is_primary:
            b_ops, dq, dr = trim_front(b_ops, q_over, r_over)
            b_q0 += dq
            b_r0 += dr
        else:
            a_ops, dq, dr = trim_back(a_ops, q_over, r_over)
            a_q1 -= dq
            a_r1 -= dr
    if not a_ops or not b_ops:
        return None
    # One molecule covers at most one circle: trim the supplementary far end.
    excess = (b_r1 - a_r0) - length
    if excess > 0:
        if a_is_primary:
            b_ops, dq, dr = trim_back(b_ops, 0, excess)
            b_q1 -= dq
            b_r1 -= dr
        else:
            a_ops, dq, dr = trim_front(a_ops, 0, excess)
            a_q0 += dq
            a_r0 += dr
    if not a_ops or not b_ops:
        return None
    gap_q = b_q0 - a_q1
    gap_r = b_r0 - a_r1
    if gap_q < 0 or gap_r < 0:
        return None
    filled = min(gap_q, gap_r)
    junction = [(M, filled), (INS, gap_q - filled), (DEL, gap_r - filled)]
    cigar = _normalise([(SOFT, a_q0)] + a_ops + junction + b_ops
                       + [(SOFT, len(seq) - b_q1)])
    # A deletion or insertion cannot start or end the aligned part.
    if cigar and cigar[0][0] == SOFT and len(cigar) > 1 and cigar[1][0] in (INS, DEL):
        return None

    merged = pysam.AlignedSegment(primary.header)
    merged.query_name = primary.query_name
    merged.flag = primary.flag & ~0x800
    merged.reference_id = primary.reference_id
    merged.reference_start = a_r0
    merged.mapping_quality = primary.mapping_quality
    merged.query_sequence = seq
    merged.query_qualities = primary.query_qualities
    merged.cigartuples = cigar
    merged.next_reference_id = -1
    merged.next_reference_start = -1
    merged.template_length = 0
    tags = [(tag, value, kind) for tag, value, kind in primary.get_tags(with_value_type=True)
            if tag not in _ALIGNMENT_TAGS]
    merged.set_tags(tags)
    md, nm = compute_md_nm(seq, cigar, reference, a_r0)
    merged.set_tag("NM", nm, "i")
    merged.set_tag("MD", md, "Z")
    return merged


def hard_clip(read: pysam.AlignedSegment) -> int:
    """Turn leading/trailing soft clips into hard clips; return bases removed.

    ``SEQ``/``QUAL`` are trimmed to the aligned part. ``POS``, ``MD`` and ``NM``
    do not change. ``MM``/``ML`` (and ``MN``) would no longer match ``SEQ`` and
    are removed.
    """
    cigar = read.cigartuples
    if not cigar:
        return 0
    lead = cigar[0][1] if cigar[0][0] == SOFT else 0
    trail = cigar[-1][1] if len(cigar) > 1 and cigar[-1][0] == SOFT else 0
    if not lead and not trail:
        return 0
    seq = read.query_sequence
    quals = read.query_qualities
    end = len(seq) - trail
    new_cigar = list(cigar)
    if lead:
        new_cigar[0] = (HARD, lead)
    if trail:
        new_cigar[-1] = (HARD, trail)
    # Merge with an existing hard clip outside the soft clip.
    new_cigar = _normalise(new_cigar)
    read.query_sequence = seq[lead:end]
    if quals is not None:
        read.query_qualities = quals[lead:end]
    read.cigartuples = new_cigar
    for tag in ("MM", "Mm", "ML", "Ml", "MN"):
        if read.has_tag(tag):
            read.set_tag(tag, None)
    return lead + trail
