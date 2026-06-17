"""Writer + parser for the fiberseq Molecular Annotation BAM tag spec.

Spec: https://github.com/fiberseq/Molecular-annotation-spec

Annotation types emitted by FiberHMM. All use the ``.`` (unknown) strand
field, matching fibertools: nucleosomes, MSPs and TF footprints are
strand-agnostic molecular features. (The parser also accepts ``+``/``-`` so
tags written by FiberHMM <= 2.13.1, which used ``+``, still read back.)
  - ``nuc.Q``    nucleosomes (>= unify_threshold bp). Quality byte = ``nq``
                 (per-call posterior mean, 0-255). The recaller emits
                 ``nuc.QQQ`` (nq, el, er) instead.
  - ``msp.``     methylase-sensitive patches (no quality).
  - ``tf.QQQ``   recaller TF footprints. Quality bytes per call:
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


def _clamp_byte(value: int) -> int:
    return max(0, min(255, int(value)))


def _append_quality_rows(out: array.array, label: str, *columns: Sequence[int]) -> None:
    if not columns:
        return
    n_rows = len(columns[0])
    if any(len(column) != n_rows for column in columns[1:]):
        raise ValueError(f"{label} quality arrays must have equal length")
    for row in zip(*columns):
        for value in row:
            out.append(_clamp_byte(value))


def _nuc_aq_has_edge_qualities(
    nuc_lq_values: Sequence[int],
    nuc_rq_values: Sequence[int],
) -> bool:
    return bool(nuc_lq_values or nuc_rq_values)


def _format_interval_list(intervals: Sequence[Tuple[int, int]]) -> str:
    return ','.join(f'{int(start) + 1}-{int(length)}'
                    for start, length in intervals)


def _format_ma_annotation_part(
    name: str,
    intervals: Sequence[Tuple[int, int]],
    qual_spec: str,
) -> Optional[str]:
    if not intervals:
        return None
    formatted = _format_interval_list(intervals)
    return f'{name}.{qual_spec}:{formatted}' if qual_spec else f'{name}.:{formatted}'


def _parse_ma_head(head: str) -> Tuple[str, str, str]:
    for i, c in enumerate(head):
        if c in '+-.':
            return head[:i], c, head[i + 1:]
    raise ValueError(f'MA head missing strand: {head!r}')


def _parse_ma_interval(token: str) -> Tuple[int, int]:
    if '-' not in token:
        raise ValueError(f'MA interval missing dash: {token!r}')
    start_str, length_str = token.split('-', 1)
    return int(start_str) - 1, int(length_str)


def _parse_ma_interval_list(data: str) -> List[Tuple[int, int]]:
    intervals: List[Tuple[int, int]] = []
    for tok in data.split(','):
        if not tok:
            continue
        intervals.append(_parse_ma_interval(tok))
    return intervals


def _parse_ma_read_length(token: str) -> int:
    try:
        return int(token)
    except ValueError:
        raise ValueError(f'MA tag must start with read length; got {token!r}')


def flip_interval_frame(start: int, length: int, read_length: int) -> Tuple[int, int]:
    """Convert a (start, length) interval between the SEQ (forward-reference) and
    molecular (original-fiber) coordinate frames of a reverse-mapped read.

    pysam ``query_sequence`` / FiberHMM internals work in SEQ frame (the BAM SEQ
    field, forward-reference). fibertools and the Molecular-annotation spec store
    ``ns/nl/as/al`` and ``MA`` in molecular frame (original read, 5'->3' as
    sequenced). For a reverse-mapped read those frames are reverse complements,
    so an interval ``[s, s+l)`` in one frame is ``[L-(s+l), L-s)`` in the other.

    This mapping is its own inverse (SEQ<->molecular), so the same call converts
    either direction. Forward reads need no conversion (frames coincide).
    """
    return (int(read_length) - (int(start) + int(length)), int(length))


def _read_length_of(read) -> int:
    """Best-effort read (query) length across pysam reads and lightweight stubs."""
    n = getattr(read, 'query_length', None)
    if n:
        return int(n)
    seq = getattr(read, 'query_sequence', None)
    if seq:
        return len(seq)
    infer = getattr(read, 'infer_read_length', None)
    if callable(infer):
        try:
            v = infer()
            if v:
                return int(v)
        except Exception:
            pass
    return 0


def flip_intervals_to_seq(starts, lengths, read) -> Tuple[List[int], List[int]]:
    """Flip molecular-frame ``ns/nl`` (or ``as/al``) read off a FiberHMM-tagged
    BAM back into SEQ (query_sequence) frame for internal processing / reference
    mapping. Coordinates only -- order is preserved, so any aligned per-interval
    quality array (``nq``/``aq``) read in the same order stays aligned. No-op for
    forward reads (the frames coincide)."""
    s = [int(x) for x in starts]
    length_list = [int(x) for x in lengths]
    if not s or not getattr(read, 'is_reverse', False):
        return s, length_list
    read_length = _read_length_of(read)
    if not read_length:
        return s, length_list
    flipped = [flip_interval_frame(a, b, read_length)
               for a, b in zip(s, length_list)]
    return [f[0] for f in flipped], [f[1] for f in flipped]


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
    # Strand field is '.' (unknown): nuc/msp/tf are strand-agnostic molecular
    # features, matching fibertools. (Versions <= 2.13.1 wrote '+'; the parser
    # still accepts it.)
    parts = [str(int(read_length))]
    for part in (
        _format_ma_annotation_part('nuc', nuc_intervals, nuc_qual_spec),
        _format_ma_annotation_part('msp', msp_intervals, ''),
        _format_ma_annotation_part('tf', tf_intervals, tf_qual_spec),
    ):
        if part is not None:
            parts.append(part)
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

    Layout for the FiberHMM default schema (nuc.Q, msp., tf.QQQ):
      - One byte per nuc:    (nq,)
      - MSPs:                (no bytes)
      - Three bytes per TF:  (tq, el, er)

    When ``nuc_lq_values``/``nuc_rq_values`` are supplied (the nuc recaller's
    nuc+QQQ schema), each nucleosome instead emits three bytes (nq, el, er),
    mirroring the TF layout. The three nuc arrays must then be equal length.
    """
    out = array.array('B')
    if _nuc_aq_has_edge_qualities(nuc_lq_values, nuc_rq_values):
        _append_quality_rows(out, "nuc", nq_values, nuc_lq_values, nuc_rq_values)
    else:
        _append_quality_rows(out, "nuc", nq_values)
    if tf_q_values:
        _append_quality_rows(out, "TF", tf_q_values, tf_lq_values, tf_rq_values)
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
    read_length = _parse_ma_read_length(pieces[0])
    out = {'read_length': read_length, 'nuc': [], 'msp': [], 'tf': [],
           'raw_types': []}
    for chunk in pieces[1:]:
        if not chunk:
            continue
        if ':' not in chunk:
            raise ValueError(f'MA chunk missing colon: {chunk!r}')
        head, data = chunk.split(':', 1)
        name, strand, qual_spec = _parse_ma_head(head)
        intervals = _parse_ma_interval_list(data)
        out['raw_types'].append((name, strand, qual_spec, intervals))
        if name in out:
            out[name].extend(intervals)
    return out


def _aq_values_sequence(aq):
    aq_values = aq if aq is not None else ()
    if not (hasattr(aq_values, "__len__") and hasattr(aq_values, "__getitem__")):
        aq_values = tuple(aq_values)
    return aq_values


def parse_aq_array(aq, qual_spec_per_type: Sequence[str],
                   n_annotations_per_type: Sequence[int]) -> List[List[int]]:
    """Parse the flat AQ array into per-annotation quality lists.

    Walks the AQ bytes consuming ``len(qual_spec)`` bytes per annotation,
    in the same order as the MA string.
    """
    result: List[List[int]] = []
    idx = 0
    aq_values = _aq_values_sequence(aq)
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
