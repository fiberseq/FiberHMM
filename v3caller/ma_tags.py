"""Writer for the fiberseq Molecular Annotation BAM tag spec.

Spec: https://github.com/fiberseq/Molecular-annotation-spec

We implement the subset we need: `nuc+QQ` and `msp+` annotation types
on the forward strand. Coordinates in the tag string are 1-based
closed (start, length) — our caller_v8 output is 0-based half-open
query coords (ns = start, ns + nl = end_exclusive), so we add 1 to
starts and keep lengths unchanged.

Quality convention: the MA spec's `Q` modifier stores one linear
0-255 value per annotation; `QQ` stores two. For nucleosomes we use
`nuc+QQ` with:
  - first value  = nq (core tightness, same as the legacy nq tag)
  - second value = mq (merge quality, minimum Poisson-interval
    center distance across internal gaps fused into this call;
    255 = pure Pass-1 atom with no merges)

MSPs get `msp+` (no quality).

This module is pure-Python and has no dependency on the Rust
`molecular-annotation` package. Output is byte-identical to what
that library would produce for the same inputs, so downstream
tools parsing the official spec will handle our BAMs correctly.
"""

from __future__ import annotations

import array
from typing import Iterable, List, Sequence, Tuple


def format_ma_tag(read_length: int,
                    nuc_intervals: Sequence[Tuple[int, int]],
                    msp_intervals: Sequence[Tuple[int, int]],
                    tf_intervals: Sequence[Tuple[int, int]] = (),
                    v2_intervals: Sequence[Tuple[int, int]] = (),
                    nuc_qual_spec: str = 'QQQ',
                    tf_qual_spec: str = 'QQQ') -> str:
    """Build the MA:Z string per the fiberseq Molecular-annotation
    spec (https://github.com/fiberseq/Molecular-annotation-spec).

    Emitted annotation types:
      - `nuc+{nuc_qual_spec}`: v3 nucleosome calls, default QQQ =
        (nq, lq, rq). Drop mq from MA (kept in the legacy mq tag
        for merge diagnostics).
      - `msp+`: MSPs between non-merged nucs, no quality.
      - `tf+{tf_qual_spec}`: v3 TF footprints, default QQQ =
        (tq, el, er).
      - `fp_v2+`: v2 HMM all-footprints overlay (from the input
        BAM's ns/nl tags), no quality. This is a CUSTOM type per
        the spec's allowance for user-defined annotations.

    Returns:
        MA tag string like
            "4521;nuc+QQQ:43-147,216-155;msp+:1-42;tf+QQQ:50-20;fp_v2+:40-200"
    """
    parts = [str(int(read_length))]
    if nuc_intervals:
        nucs = ','.join(f'{int(s) + 1}-{int(l)}' for s, l in nuc_intervals)
        parts.append(f'nuc+{nuc_qual_spec}:{nucs}' if nuc_qual_spec
                      else f'nuc+:{nucs}')
    if msp_intervals:
        msps = ','.join(f'{int(s) + 1}-{int(l)}' for s, l in msp_intervals)
        parts.append(f'msp+:{msps}')
    if tf_intervals:
        tfs = ','.join(f'{int(s) + 1}-{int(l)}' for s, l in tf_intervals)
        parts.append(f'tf+{tf_qual_spec}:{tfs}' if tf_qual_spec
                      else f'tf+:{tfs}')
    if v2_intervals:
        v2s = ','.join(f'{int(s) + 1}-{int(l)}' for s, l in v2_intervals)
        parts.append(f'fp_v2+:{v2s}')
    return ';'.join(parts)


def format_aq_array(nq_values: Sequence[int],
                     lq_values: Sequence[int],
                     rq_values: Sequence[int],
                     tf_q_values: Sequence[int] = (),
                     tf_lq_values: Sequence[int] = (),
                     tf_rq_values: Sequence[int] = ()) -> array.array:
    """Build the AQ:B:C array, interleaved per annotation.

    Layout matches the default MA spec emitted by format_ma_tag:
      - For each nuc (nuc+QQQ): (nq, lq, rq)
      - MSPs contribute nothing (unqualified)
      - For each TF (tf+QQQ): (tq, el, er)
      - fp_v2 contributes nothing (unqualified)

    Lengths must match — format_aq_array trusts the caller has
    already aligned the arrays.
    """
    def clamp(v):
        return max(0, min(255, int(v)))

    out = array.array('B')
    assert len(nq_values) == len(lq_values) == len(rq_values)
    for nq, lq, rq in zip(nq_values, lq_values, rq_values):
        out.append(clamp(nq)); out.append(clamp(lq)); out.append(clamp(rq))
    if tf_q_values:
        assert len(tf_q_values) == len(tf_lq_values) == len(tf_rq_values)
        for tq, el, er in zip(tf_q_values, tf_lq_values, tf_rq_values):
            out.append(clamp(tq)); out.append(clamp(el)); out.append(clamp(er))
    return out


# -------------------------------------------------------------------
# Parser (for tests / validation)
# -------------------------------------------------------------------

def parse_ma_tag(ma_string: str) -> dict:
    """Parse a MA:Z string into a dict.

    Returns:
        {
            'read_length': int,
            'nuc': [(start_0based, length), ...],   # from nuc+QQ or nuc+
            'msp': [(start_0based, length), ...],   # from msp+ or msp+Q
            'raw_types': [(type_name, strand, qual_spec,
                            [(start, length), ...]), ...],
        }

    Start positions are converted from 1-based (tag format) to 0-based
    (our internal convention).
    """
    if not ma_string:
        raise ValueError('empty MA tag')
    pieces = ma_string.split(';')
    try:
        read_length = int(pieces[0])
    except ValueError:
        raise ValueError(f'MA tag must start with read length; got {pieces[0]!r}')
    out = {
        'read_length': read_length,
        'nuc': [],
        'msp': [],
        'raw_types': [],
    }
    for chunk in pieces[1:]:
        if not chunk:
            continue
        # chunk = "name+Q:s-l,s-l" or similar
        if ':' not in chunk:
            raise ValueError(f'MA annotation chunk missing colon: {chunk!r}')
        head, data = chunk.split(':', 1)
        # head = "nuc+QQ" or "msp+" or "fire.PQ"
        if len(head) < 2:
            raise ValueError(f'MA annotation head too short: {head!r}')
        # Strand is at position after the trailing alphanumeric name
        # Simplest: find the first '+', '-', '.' character
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
            s_1 = int(s_str)
            l = int(l_str)
            intervals.append((s_1 - 1, l))  # convert 1-based -> 0-based
        out['raw_types'].append((name, strand, qual_spec, intervals))
        if name == 'nuc':
            out['nuc'].extend(intervals)
        elif name == 'msp':
            out['msp'].extend(intervals)
    return out


def parse_aq_array(aq, qual_spec_per_annotation: Sequence[str],
                     n_annotations_per_type: Sequence[int]):
    """Parse the flat AQ array into per-annotation quality tuples.

    Args:
        aq: u8 array from the BAM AQ tag.
        qual_spec_per_annotation: e.g. ['QQ', ''] — one string per
            annotation TYPE in MA order (after the read length).
        n_annotations_per_type: parallel to qual_spec_per_annotation;
            how many annotations each type has.

    Returns:
        List-of-lists: one sublist per annotation (flattened in MA
        order). Each sublist has len == len(qual_spec_for_its_type).

    Example: MA has nuc+QQ (3 annotations) + msp+ (2), aq array is
    [40, 80, 50, 90, 30, 70]. We return
        [[40, 80], [50, 90], [30, 70], [], []]
    """
    result: List[List[int]] = []
    idx = 0
    aq_list = list(aq) if aq is not None else []
    for qspec, n in zip(qual_spec_per_annotation, n_annotations_per_type):
        per_ann = len(qspec)
        for _ in range(n):
            vals = aq_list[idx:idx + per_ann]
            if len(vals) < per_ann:
                raise ValueError(
                    f'AQ array shorter than expected: need {per_ann} more '
                    f'values for {qspec} annotation')
            result.append(vals)
            idx += per_ann
    if idx != len(aq_list):
        raise ValueError(
            f'AQ array has {len(aq_list)} values but MA required {idx}')
    return result
