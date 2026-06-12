"""Writer + parser for the fiberseq Molecular Annotation BAM tag spec.

Spec: https://github.com/fiberseq/Molecular-annotation-spec

Annotation types emitted by FiberHMM:
  - ``nuc+Q``    nucleosomes (>= unify_threshold bp). Quality byte = ``nq``
                 (per-call posterior mean, 0-255).
  - ``msp+``     methylase-sensitive patches (no quality).
  - ``tf+QQQ``   recaller TF footprints. Quality bytes per call:
                 (``tq``, ``el``, ``er``) = (LLR-derived score,
                 left-edge sharpness, right-edge sharpness).

tq encoding
-----------

``tq = min(255, round(LLR * 10))``. LLR is the cumulative log-likelihood
ratio (nats) of the protected vs accessible hypothesis under the trained
emission table.

  - tq =  50  -> LLR =  5  nats (likelihood ratio ~ 148:1)
  - tq = 100  -> LLR = 10  nats (LR ~ 22,000:1)
  - tq = 255  -> LLR >= 25.5 nats (LR >= 1.2e11, saturated)

Mnemonic: every 23 tq points adds one order of magnitude to the
likelihood ratio. tq >= 50 is the soft floor (recommended default
emission threshold); tq >= 100 is "high confidence".

el / er (edge sharpness) encoding
---------------------------------

For each call boundary, the recaller emits the **conservative** boundary
(immediately past the last informative miss). The true boundary may
extend up to the terminating hit (the bp after the call). The gap
between these two positions is the edge ambiguity:

  ambiguity = (terminating_hit_bp) - (conservative_boundary_bp)

Encoded as:

  edge_q = round(255 * max(0, 1 - ambiguity / EDGE_AMBIGUITY_SAT))

where ``EDGE_AMBIGUITY_SAT = 30`` bp. Sharp edges (hit immediately
adjacent to the call boundary) score 255. Calls whose terminating hit
is >= 30 bp away from the conservative boundary score 0 (size estimate
is ambiguous to the upper end of that 30 bp range).
"""
from __future__ import annotations

import array
from typing import List, Optional, Sequence, Tuple

TQ_SCALE = 10.0          # tq = round(LLR * TQ_SCALE); saturates at LLR=25.5 nats
EDGE_AMBIGUITY_SAT = 30  # bp; el/er = 0 at ambiguity >= this


def llr_to_tq(llr: float) -> int:
    """Encode an LLR (nats) into the tq quality byte."""
    return max(0, min(255, int(round(llr * TQ_SCALE))))


def tq_to_llr(tq: int) -> float:
    """Inverse of llr_to_tq."""
    return tq / TQ_SCALE


def ambiguity_to_edge(ambiguity_bp: int) -> int:
    """Encode an edge-ambiguity (bp) into the el/er quality byte."""
    if ambiguity_bp <= 0:
        return 255
    if ambiguity_bp >= EDGE_AMBIGUITY_SAT:
        return 0
    return int(round(255 * (1 - ambiguity_bp / EDGE_AMBIGUITY_SAT)))


def split_circular_interval(start: int, length: int,
                            read_length: int) -> List[Tuple[int, int]]:
    """Split a molecular interval into MA-valid linear pieces.

    Input intervals are 0-based start + length on a circular molecule. Output
    pieces are 0-based start + length with every piece contained in
    ``[0, read_length)``. Non-wrapping intervals return one piece; wrapping
    intervals return the start-of-read piece first, then the end-of-read piece.
    """
    start = int(start)
    length = int(length)
    read_length = int(read_length)
    if length <= 0 or read_length <= 0:
        return []
    if length >= read_length:
        return [(0, read_length)]

    start %= read_length
    end = start + length
    if end <= read_length:
        return [(start, length)]

    return [(0, end - read_length), (start, read_length - start)]


def interval_wraps(start: int, length: int, read_length: int) -> bool:
    """Return whether an interval crosses the circular read origin."""
    if int(length) <= 0 or int(read_length) <= 0:
        return False
    if int(length) >= int(read_length):
        return False
    return (int(start) % int(read_length)) + int(length) > int(read_length)


def format_ma_tag(read_length: int,
                  nuc_intervals: Sequence[Tuple[int, int]],
                  msp_intervals: Sequence[Tuple[int, int]],
                  tf_intervals: Sequence[Tuple[int, int]] = (),
                  nuc_qual_spec: str = 'Q',
                  tf_qual_spec: str = 'QQQ') -> str:
    """Build the MA:Z string per the fiberseq Molecular-annotation spec.

    Coordinates are converted from 0-based (internal) to 1-based (spec).
    """
    parts = [str(int(read_length))]
    if nuc_intervals:
        nucs = ','.join(f'{int(s) + 1}-{int(end)}' for s, end in nuc_intervals)
        parts.append(f'nuc+{nuc_qual_spec}:{nucs}' if nuc_qual_spec
                     else f'nuc+:{nucs}')
    if msp_intervals:
        msps = ','.join(f'{int(s) + 1}-{int(end)}' for s, end in msp_intervals)
        parts.append(f'msp+:{msps}')
    if tf_intervals:
        tfs = ','.join(f'{int(s) + 1}-{int(end)}' for s, end in tf_intervals)
        parts.append(f'tf+{tf_qual_spec}:{tfs}' if tf_qual_spec
                     else f'tf+:{tfs}')
    return ';'.join(parts)


def format_an_tag(annotation_names: Sequence[Optional[str]]) -> str:
    """Build an AN:Z tag aligned to MA annotation order.

    FiberHMM uses AN to give both clipped pieces of a circularly wrapped
    annotation the same feature ID. Non-wrapped annotations receive unique IDs
    whenever AN is emitted so the tag has one unambiguous name per annotation.
    """
    return ','.join(str(name or '.') for name in annotation_names)


def parse_an_tag(an_string: str) -> List[str]:
    """Parse an AN:Z string into names aligned with MA annotations."""
    if not an_string:
        return []
    return [tok if tok != '.' else '' for tok in an_string.split(',')]


def format_aq_array(nq_values: Sequence[int],
                    tf_q_values: Sequence[int] = (),
                    tf_lq_values: Sequence[int] = (),
                    tf_rq_values: Sequence[int] = (),
                    nuc_lq_values: Sequence[int] = (),
                    nuc_rq_values: Sequence[int] = ()) -> array.array:
    """Build the AQ:B:C array, interleaved per annotation in MA order.

    Layout for the FiberHMM default schema (nuc+Q, msp+, tf+QQQ):
      - One byte per nuc:    (nq,)
      - MSPs:                (no bytes)
      - Three bytes per TF:  (tq, el, er)

    When ``nuc_lq_values``/``nuc_rq_values`` are supplied (the nuc recaller's
    nuc+QQQ schema), each nucleosome instead emits three bytes (nq, el, er),
    mirroring the TF layout. The three nuc arrays must then be equal length.
    """
    def clamp(v):
        return max(0, min(255, int(v)))

    out = array.array('B')
    if nuc_lq_values or nuc_rq_values:
        if not (len(nq_values) == len(nuc_lq_values) == len(nuc_rq_values)):
            raise ValueError("nuc quality arrays must have equal length")
        for nq, el, er in zip(nq_values, nuc_lq_values, nuc_rq_values):
            out.append(clamp(nq))
            out.append(clamp(el))
            out.append(clamp(er))
    else:
        for nq in nq_values:
            out.append(clamp(nq))
    if tf_q_values:
        if not (len(tf_q_values) == len(tf_lq_values) == len(tf_rq_values)):
            raise ValueError("TF quality arrays must have equal length")
        for tq, el, er in zip(tf_q_values, tf_lq_values, tf_rq_values):
            out.append(clamp(tq))
            out.append(clamp(el))
            out.append(clamp(er))
    return out


def parse_ma_tag(ma_string: str) -> dict:
    """Parse a MA:Z string into a dict.

    Returns:
        {
            'read_length': int,
            'nuc': [(start_0based, length), ...],
            'msp': [(start_0based, length), ...],
            'tf':  [(start_0based, length), ...],
            'raw_types': [(name, strand, qual_spec, [(s,l), ...]), ...],
        }

    Coordinates are 0-based (converted from the 1-based spec format).
    """
    if not ma_string:
        raise ValueError('empty MA tag')
    pieces = ma_string.split(';')
    try:
        read_length = int(pieces[0])
    except ValueError:
        raise ValueError(f'MA tag must start with read length; got {pieces[0]!r}')
    out = {'read_length': read_length, 'nuc': [], 'msp': [], 'tf': [],
           'raw_types': []}
    for chunk in pieces[1:]:
        if not chunk:
            continue
        if ':' not in chunk:
            raise ValueError(f'MA chunk missing colon: {chunk!r}')
        head, data = chunk.split(':', 1)
        for i, c in enumerate(head):
            if c in '+-.':
                name = head[:i]
                strand = head[i]
                qual_spec = head[i + 1:]
                break
        else:
            raise ValueError(f'MA head missing strand: {head!r}')
        intervals: List[Tuple[int, int]] = []
        for tok in data.split(','):
            if not tok:
                continue
            if '-' not in tok:
                raise ValueError(f'MA interval missing dash: {tok!r}')
            s_str, l_str = tok.split('-', 1)
            intervals.append((int(s_str) - 1, int(l_str)))
        out['raw_types'].append((name, strand, qual_spec, intervals))
        if name in out:
            out[name].extend(intervals)
    return out


def parse_aq_array(aq, qual_spec_per_type: Sequence[str],
                   n_annotations_per_type: Sequence[int]) -> List[List[int]]:
    """Parse the flat AQ array into per-annotation quality lists.

    Walks the AQ bytes consuming ``len(qual_spec)`` bytes per annotation,
    in the same order as the MA string.
    """
    result: List[List[int]] = []
    idx = 0
    aq_values = aq if aq is not None else ()
    if not (hasattr(aq_values, "__len__") and hasattr(aq_values, "__getitem__")):
        aq_values = tuple(aq_values)
    aq_len = len(aq_values)
    for spec, n in zip(qual_spec_per_type, n_annotations_per_type):
        n_q = len(spec)
        for _ in range(n):
            if n_q == 0:
                result.append([])
            else:
                end = min(idx + n_q, aq_len)
                result.append([int(aq_values[i]) for i in range(idx, end)])
                idx += n_q
    return result
