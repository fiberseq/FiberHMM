"""Cut a DAF strand-swap chimera into two single-strand alignments.

Used by ``fiberhmm-call --chimeras split`` (an option, not the default).
Each piece keeps its own-strand events up to the breakpoint gap: piece 0
is the query from its start to the last left-strand event of the left
segment, piece 1 from the first right-strand event of the right segment to
its end. The gap between them, where the template switch happened at an
unknown point, is in neither piece, so neither piece ends in a stretch
without deamination created by the cut.

Pieces are new records named ``<qname>/c0`` and ``<qname>/c1`` with
``xc:Z:split``, ``xi:i`` (piece index), ``xn:Z`` (original name) and
``xb:B:i`` (the gap on the record's reference, 0-based half-open
``[start, end)``). CIGAR, SEQ, QUAL and MD are cut together and NM is
recomputed. Read-level tags that a cut cannot invalidate are copied
(:data:`KEPT_TAGS`). Base modifications are kept: ``MM``/``ML`` (and the
legacy ``Mm``/``Ml``) are re-indexed to the piece (:func:`cut_modifications`)
and ``MN``, if present, is set to the piece's SEQ length. They are dropped for
a piece whose record is hard-clipped or whose MM/ML do not parse or do not
match SEQ. Every other position-dependent tag (SA, the DAF strand tag ``st``,
earlier FiberHMM calls) is dropped.
"""
from __future__ import annotations

from array import array
from bisect import bisect_left

import pysam

SPLIT_REASON = 'split'
PIECE_TAG = 'xi'
ORIGINAL_NAME_TAG = 'xn'
GAP_TAG = 'xb'

# Read-level tags that stay true for any part of the read: read group,
# FiberHMM deduplication cluster, and basecaller read metadata.
KEPT_TAGS = ('RG', 'ds', 'di', 'qs', 'du', 'ch', 'rn', 'fn')

# (MM, ML) tag pairs: the SAM names and the pre-standard names.
MODIFICATION_TAGS = (('MM', 'ML'), ('Mm', 'Ml'))

_MAX_QNAME = 254
_COMPLEMENT = str.maketrans('ACGTN', 'TGCAN')
_ALIGNED = (0, 7, 8)   # M, =, X


def _cut_cigar(read, qs, qe):
    """CIGAR ops of query ``[qs, qe)`` and the piece's reference start."""
    qlen = read.query_length
    ops = []
    qpos = 0
    rpos = read.reference_start
    ref_start = None
    for op, length in read.cigartuples:
        if op in _ALIGNED:
            a, b = max(qpos, qs), min(qpos + length, qe)
            if a < b:
                if ref_start is None:
                    ref_start = rpos + (a - qpos)
                ops.append((op, b - a))
            qpos += length
            rpos += length
        elif op in (1, 4):          # I, S
            a, b = max(qpos, qs), min(qpos + length, qe)
            if a < b:
                ops.append((op, b - a))
            qpos += length
        elif op in (2, 3):          # D, N: only strictly inside the piece
            if ref_start is not None and qs < qpos < qe:
                ops.append((op, length))
            rpos += length
        elif op == 5:               # H: only at an outer end that is kept
            if (qpos == 0 and qs == 0 and not ops) or (qpos == qlen and qe == qlen):
                ops.append((op, length))
    # An inner edge is always an aligned event base; drop any I/D next to it
    # defensively so the piece starts and ends on an aligned base.
    while ops and ops[-1][0] in (1, 2, 3) and qe < qlen:
        ops.pop()
    lead = 0
    while lead < len(ops) and ops[lead][0] in (4, 5):
        lead += 1
    while lead < len(ops) and ops[lead][0] in (1, 2, 3) and qs > 0:
        ops.pop(lead)
    merged = []
    for op, length in ops:
        if merged and merged[-1][0] == op:
            merged[-1] = (op, merged[-1][1] + length)
        else:
            merged.append((op, length))
    return merged, ref_start


def _cut_md(read, qs, qe, ref_start, ref_end):
    """MD and NM of the piece from the original's MD, or (None, None)."""
    from fiberhmm.daf.encoder import md_matches_cigar
    if not read.has_tag('MD') or not md_matches_cigar(read):
        return None, None
    if any(op == 3 for op, _ in read.cigartuples):
        return None, None
    try:
        pairs = read.get_aligned_pairs(with_seq=True)
    except Exception:
        return None, None
    md, run, nm, in_del = [], 0, 0, False
    for q, r, base in pairs:
        if q is not None and r is None:          # insertion / soft clip (NM below)
            continue
        if q is None:                            # deletion
            if r is None or not ref_start <= r < ref_end:
                continue
            if not in_del:
                md.append(str(run))
                md.append('^')
                run = 0
                in_del = True
            md.append(base.upper())
            nm += 1
            continue
        if not qs <= q < qe:
            continue
        in_del = False
        if base is not None and base.islower():
            md.append(str(run))
            md.append(base.upper())
            run = 0
            nm += 1
        else:
            run += 1
    md.append(str(run))
    return ''.join(md), nm


def _reference_of(read, qpos):
    for q, r in read.get_aligned_pairs(matches_only=True):
        if q == qpos:
            return r
    return None


def gap_interval(read, split):
    """The breakpoint gap on the reference, 0-based half-open, or None."""
    left = _reference_of(read, split['left_last'])
    right = _reference_of(read, split['right_first'])
    if left is None or right is None:
        return None
    return left + 1, right


def _parse_mm(mm):
    """``[(head, base, n_codes, skips)]`` of an MM string, or None.

    ``head`` is the entry text before its skip counts (for example ``C+mh?``),
    kept verbatim so the codes and the ``.``/``?`` flag are written back as
    they were. ``n_codes`` is the number of ML values per called position.
    """
    if not isinstance(mm, str):
        return None
    entries = []
    for item in mm.split(';'):
        if not item:
            continue
        head, *skips = item.split(',')
        if len(head) < 3 or head[0] not in 'ACGTN' or head[1] not in '+-':
            return None
        codes = head[2:-1] if head[-1] in '.?' else head[2:]
        if codes.isdigit():
            n_codes = 1                       # one ChEBI code
        elif codes.isalpha():
            n_codes = len(codes)              # one letter per code
        else:
            return None
        try:
            skips = [int(skip) for skip in skips]
        except ValueError:
            return None
        if any(skip < 0 for skip in skips):
            return None
        entries.append((head, head[0], n_codes, skips))
    return entries


def cut_modifications(read, qs, qe, mm_tag='MM', ml_tag='ML'):
    """``(mm, ml)`` of SEQ ``[qs, qe)`` of ``read``, or None (drop them).

    MM skip counts run over the entry's base (``N``: any base) in the read's
    original orientation, SEQ reverse-complemented for a reverse-strand
    record, for ``+`` and ``-`` entries alike. In that orientation the piece
    is ``[qs, qe)`` (forward) or ``[L - qe, L - qs)`` (reverse). The calls
    inside it are kept, their skip counts recomputed over the piece and their
    ML values (interleaved per position for multi-code entries) carried. Each
    entry and its ``.``/``?`` flag stay, also with no call left. None when the
    record is hard-clipped, MM/ML do not parse, a skip runs past SEQ, the ML
    length disagrees with MM or MN disagrees with SEQ.
    """
    seq = read.query_sequence
    mm = read.get_tag(mm_tag)
    ml = read.get_tag(ml_tag) if read.has_tag(ml_tag) else []
    entries = _parse_mm(mm)
    if (not seq or entries is None
            or any(op == 5 for op, _ in (read.cigartuples or ()))
            or (read.has_tag('MN') and read.get_tag('MN') != len(seq))):
        return None
    try:
        ml = [int(value) for value in ml]
    except (TypeError, ValueError):
        return None
    length = len(seq)
    original = seq.upper()
    if read.is_reverse:
        original = original.translate(_COMPLEMENT)[::-1]
    start = qs if not read.is_reverse else length - qe
    end = start + (qe - qs)
    texts, values, offset = [], [], 0
    for head, base, n_codes, skips in entries:
        occurrences = (range(length) if base == 'N'
                       else [i for i, b in enumerate(original) if b == base])
        first = bisect_left(occurrences, start)
        last = bisect_left(occurrences, end)
        text, previous, index = [head], first - 1, -1
        for skip in skips:
            index += skip + 1
            if index >= len(occurrences):
                return None
            if first <= index < last:
                text.append(f',{index - previous - 1}')
                previous = index
                values.extend(ml[offset:offset + n_codes])
            offset += n_codes
        if offset > len(ml):
            return None
        texts.append(''.join(text) + ';')
    if offset != len(ml):
        return None
    return ''.join(texts), array('B', values)


def make_piece(read, header, qs, qe, index, gap):
    """One piece of ``read`` (query ``[qs, qe)``), or None if it cannot be cut."""
    seq = read.query_sequence
    if not seq or qe <= qs:
        return None
    ops, ref_start = _cut_cigar(read, qs, qe)
    if ref_start is None:
        return None
    ref_len = sum(length for op, length in ops if op in (0, 2, 3, 7, 8))
    piece = pysam.AlignedSegment(header)
    piece.query_name = f'{read.query_name}/c{index}'
    piece.flag = read.flag
    piece.reference_id = read.reference_id
    piece.reference_start = ref_start
    piece.mapping_quality = read.mapping_quality
    piece.cigartuples = ops
    quals = read.query_qualities
    # Setting SEQ clears QUAL, so QUAL is set second.
    piece.query_sequence = seq[qs:qe]
    if quals is not None:
        piece.query_qualities = quals[qs:qe]
    piece.next_reference_id = -1
    piece.next_reference_start = -1
    piece.template_length = 0
    for tag in KEPT_TAGS:
        if read.has_tag(tag):
            value, kind = read.get_tag(tag, with_value_type=True)
            piece.set_tag(tag, value, value_type=kind)
    kept_modifications = False
    for mm_tag, ml_tag in MODIFICATION_TAGS:
        if read.has_tag(mm_tag):
            cut = cut_modifications(read, qs, qe, mm_tag, ml_tag)
            if cut is not None:
                piece.set_tag(mm_tag, cut[0], value_type='Z')
                if len(cut[1]) or read.has_tag(ml_tag):
                    piece.set_tag(ml_tag, cut[1])
                kept_modifications = True
    if kept_modifications and read.has_tag('MN'):
        piece.set_tag('MN', qe - qs, value_type='i')
    md, nm = _cut_md(read, qs, qe, ref_start, ref_start + ref_len)
    if md is not None:
        inserted = sum(length for op, length in ops if op == 1)
        piece.set_tag('NM', nm + inserted, value_type='i')
        piece.set_tag('MD', md, value_type='Z')
    piece.set_tag('xc', SPLIT_REASON, value_type='Z')
    piece.set_tag(PIECE_TAG, index, value_type='i')
    piece.set_tag(ORIGINAL_NAME_TAG, read.query_name, value_type='Z')
    if gap is not None:
        piece.set_tag(GAP_TAG, array('i', gap))
    return piece


def split_record(read, header, split):
    """``(pieces, gap)`` for a record and its :func:`daf_chimera_split`.

    ``pieces`` is ``[piece0, piece1]`` with None for a piece that cannot be
    cut; length filtering is the caller's.
    """
    name = read.query_name
    if not name or name == '*' or len(name) + 3 > _MAX_QNAME:
        return [None, None], None
    gap = gap_interval(read, split)
    qlen = read.query_length
    return [make_piece(read, header, 0, split['left_last'] + 1, 0, gap),
            make_piece(read, header, split['right_first'], qlen, 1, gap)], gap
