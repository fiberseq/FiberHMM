"""No-call blocks: long stretches of a DAF read that carry no evidence.

DAF deamination evidence is a read-versus-reference comparison, so query bases
without a reference counterpart (CIGAR insertions and soft clips) are encoded
as no evidence (see ``engine.daf_unaligned_query_positions``). Over a short
gap the HMM bridges them like any run of non-target bases. Over a long one its
path only follows the transition prior, and Viterbi fills the whole stretch
with one state: a block-long "MSP" or "nucleosome" that no observation
supports. Calls are therefore removed from unaligned stretches of at least
:data:`NO_CALL_MIN_BLOCK` bp (the conventional structural-variant size):

* nucleosomes and MSPs are trimmed to the part outside the block (a call that
  spans a whole block becomes two pieces); a trimmed nucleosome edge gets edge
  byte 0, the same "not molecule-resolved" mark a censored edge gets;
* TF footprints overlapping a block are dropped.

The block is then annotated with nothing: neither protected nor accessible.
Per-call quality arrays stay aligned with their intervals, so the tags remain
well formed. Linear reads only: circular molecules keep the mask but not the
trimming.

The same trimming keeps a supplementary record's calls to its aligned bases
in every mode: its soft clips are the primary record's sequence (minimap2
``-Y``), so calling them again would annotate the same bases twice.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

NO_CALL_MIN_BLOCK = 50

_QUERY_OPS = (0, 1, 4, 7, 8)


def unaligned_blocks(cigartuples, min_length: int = NO_CALL_MIN_BLOCK,
                     query_length: Optional[int] = None) -> List[Tuple[int, int]]:
    """``[(start, end), ...]`` SEQ spans of consecutive CIGAR I/S bases at
    least ``min_length`` long (adjacent I and S operations are merged)."""
    blocks: List[Tuple[int, int]] = []
    if not cigartuples:
        return blocks
    q = 0
    run_start = None
    for op, length in cigartuples:
        if op in (1, 4):
            if run_start is None:
                run_start = q
        elif op in (0, 7, 8):
            if run_start is not None:
                blocks.append((run_start, q))
                run_start = None
        if op in _QUERY_OPS:
            q += length
    if run_start is not None:
        blocks.append((run_start, q))
    if query_length is not None:
        blocks = [(s, min(e, query_length)) for s, e in blocks if s < query_length]
    return [(s, e) for s, e in blocks if e - s >= min_length]


def supplementary_clip_blocks(read) -> List[Tuple[int, int]]:
    """SEQ spans of a record's leading and trailing soft clips (any length)."""
    cigar = getattr(read, 'cigartuples', None)
    if not cigar:
        return []
    q_len = sum(n for op, n in cigar if op in _QUERY_OPS)
    out = []
    if cigar[0][0] == 4:
        out.append((0, cigar[0][1]))
    elif len(cigar) > 1 and cigar[0][0] == 5 and cigar[1][0] == 4:
        out.append((0, cigar[1][1]))
    if len(cigar) > 1 and cigar[-1][0] == 4:
        out.append((q_len - cigar[-1][1], q_len))
    elif len(cigar) > 2 and cigar[-1][0] == 5 and cigar[-2][0] == 4:
        out.append((q_len - cigar[-2][1], q_len))
    return [(s, e) for s, e in out if e > s]


def merge_blocks(blocks) -> List[Tuple[int, int]]:
    """Sort and merge overlapping or touching spans."""
    out: List[Tuple[int, int]] = []
    for s, e in sorted((int(a), int(b)) for a, b in blocks):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _pieces(start: int, end: int, blocks: Sequence[Tuple[int, int]]):
    """``[(s, e, left_cut, right_cut)]``: parts of [start, end) outside blocks."""
    out = []
    cur = start
    left_cut = False
    for bs, be in blocks:
        if be <= cur or bs >= end:
            continue
        if bs > cur:
            out.append((cur, bs, left_cut, True))
        cur = max(cur, be)
        left_cut = True
        if cur >= end:
            break
    if cur < end:
        out.append((cur, end, left_cut, False))
    return out


def _overlaps(start: int, end: int, blocks) -> bool:
    return any(bs < end and be > start for bs, be in blocks)


def _trim(starts, lengths, blocks, per_call: Iterable[Optional[Sequence]] = ()):
    """Trim intervals; returns (starts, lengths, [per-call lists...], cuts)."""
    per_call = list(per_call)
    out_s, out_l, cuts = [], [], []
    out_extra = [[] if arr is not None else None for arr in per_call]
    for i, (s, length) in enumerate(zip(starts, lengths)):
        s = int(s)
        e = s + int(length)
        for ps, pe, lc, rc in _pieces(s, e, blocks):
            out_s.append(ps)
            out_l.append(pe - ps)
            cuts.append((lc, rc))
            for k, arr in enumerate(per_call):
                if arr is not None:
                    out_extra[k].append(arr[i])
    return out_s, out_l, out_extra, cuts


def _like(template, values, dtype=np.int32):
    if isinstance(template, np.ndarray):
        return np.asarray(values, dtype=template.dtype if values else dtype)
    return list(values)


def _like_scores(template, values):
    if template is None:
        return None
    if isinstance(template, np.ndarray):
        return np.asarray(values, dtype=template.dtype)
    return list(values)


def suppress_calls_in_blocks(result: dict, blocks) -> dict:
    """Remove calls from ``blocks`` in an apply or fused-recall result dict.

    Handles ``ns``/``nl`` (with ``ns_scores``, ``nq_for_kept_nucs``,
    ``nuc_el_for_kept``, ``nuc_er_for_kept``), ``as``/``al`` (with
    ``as_scores``) and ``tf_calls``. Circular results are returned unchanged.
    """
    if not blocks or result is None or result.get('circular'):
        return result
    blocks = sorted((int(s), int(e)) for s, e in blocks)

    if 'ns' in result and 'nl' in result:
        ns, nl = result['ns'], result['nl']
        names = ('ns_scores', 'nq_for_kept_nucs', 'nuc_el_for_kept', 'nuc_er_for_kept')
        per_call = [result.get(n) for n in names]
        per_call = [p if p is not None and len(p) == len(ns) else None
                    for p in per_call]
        s, l, extra, cuts = _trim(ns, nl, blocks, per_call)
        el, er = extra[2], extra[3]
        if el is not None:
            el = [0 if lc else v for v, (lc, _rc) in zip(el, cuts)]
        if er is not None:
            er = [0 if rc else v for v, (_lc, rc) in zip(er, cuts)]
        result['ns'] = _like(ns, s)
        result['nl'] = _like(nl, l)
        for name, values in zip(names, (extra[0], extra[1], el, er)):
            if values is not None:
                result[name] = (_like_scores(result[name], values)
                                if name == 'ns_scores' else list(values))

    if 'as' in result and 'al' in result:
        a_s, a_l = result['as'], result['al']
        scores = result.get('as_scores')
        if scores is not None and len(scores) != len(a_s):
            scores = None
        s, l, extra, _cuts = _trim(a_s, a_l, blocks, [scores])
        result['as'] = _like(a_s, s)
        result['al'] = _like(a_l, l)
        if extra[0] is not None:
            result['as_scores'] = _like_scores(result['as_scores'], extra[0])

    if result.get('tf_calls'):
        result['tf_calls'] = [
            call for call in result['tf_calls']
            if not _overlaps(int(call.start), int(call.start) + int(call.length), blocks)
        ]
    return result
