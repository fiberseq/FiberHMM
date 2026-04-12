#!/usr/bin/env python3
"""Reusable helpers for parsing v3-caller output BAMs with MA tags.

Extracts v2 footprints (`fp_v2+`), v3 nucs (`nuc+QQQ`), v3 MSPs
(`msp+`), and v3 TFs (`tf+QQQ`) into per-read dicts with start/
length/quality arrays.

Coordinates are converted from 1-based (MA) to 0-based (consumer).
"""

from __future__ import annotations

import pysam
from typing import Iterable


def parse_ma(ma_string: str) -> dict:
    """Parse MA:Z string → dict of annotation type → list of (start, length).
    Returns {'read_length': int, 'nuc': [...], 'msp': [...], 'tf': [...],
    'fp_v2': [...], 'raw': [(name, strand, qspec, [...])]}."""
    pieces = ma_string.split(';')
    read_len = int(pieces[0])
    out = {
        'read_length': read_len,
        'nuc': [], 'msp': [], 'tf': [], 'fp_v2': [],
        'raw': [],
    }
    for chunk in pieces[1:]:
        if not chunk or ':' not in chunk:
            continue
        head, data = chunk.split(':', 1)
        # head like "nuc+QQQ" or "msp+" or "fp_v2+" — find strand char
        strand_idx = None
        for i, c in enumerate(head):
            if c in '+-.':
                strand_idx = i; break
        if strand_idx is None:
            continue
        name = head[:strand_idx]
        strand = head[strand_idx]
        qspec = head[strand_idx + 1:]
        intervals = []
        for tok in data.split(','):
            if not tok or '-' not in tok:
                continue
            s_str, l_str = tok.split('-', 1)
            try:
                # 1-based MA → 0-based
                s = int(s_str) - 1
                l = int(l_str)
            except ValueError:
                continue
            intervals.append((s, l))
        out['raw'].append((name, strand, qspec, intervals))
        if name in out:
            out[name].extend(intervals)
    return out


def split_aq(aq_array, qspec_per_type: list[tuple[str, int]]):
    """Split flat AQ:B:C array into per-annotation quality tuples.

    qspec_per_type: [(qspec_string, n_annotations), ...] in MA order.
    Returns list-of-tuples, one tuple per annotation (flattened in MA order).
    """
    result = []
    idx = 0
    aq_list = list(aq_array) if aq_array is not None else []
    for qspec, n in qspec_per_type:
        per = len(qspec)
        for _ in range(n):
            result.append(tuple(aq_list[idx:idx + per]))
            idx += per
    return result


def iter_reads(bam_path: str) -> Iterable[dict]:
    """Yield one dict per mapped primary read: {
        'name', 'chrom', 'ref_start', 'ref_end', 'read_len',
        'nuc': [(s, l, quality_tuple_or_None)], ...
        'msp': [(s, l)], 'tf': [(s, l, q)], 'fp_v2': [(s, l)],
    }.
    """
    bam = pysam.AlignmentFile(bam_path, 'rb', check_sq=False)
    for r in bam.fetch(until_eof=True):
        if r.is_unmapped or r.is_secondary or r.is_supplementary:
            continue
        if not r.has_tag('MA'):
            continue
        ma_str = r.get_tag('MA')
        parsed = parse_ma(ma_str)

        # Build qspec_per_type in MA order for AQ splitting
        qspec_per_type = [(qspec, len(intervals))
                           for name, strand, qspec, intervals in parsed['raw']]
        aq = r.get_tag('AQ') if r.has_tag('AQ') else []
        per_ann_q = split_aq(aq, qspec_per_type)

        # Re-walk raw to attach qualities per annotation
        nuc_out, tf_out = [], []
        q_idx = 0
        for name, strand, qspec, intervals in parsed['raw']:
            for s, l in intervals:
                q = per_ann_q[q_idx] if qspec else None
                q_idx += 1
                if name == 'nuc':
                    nuc_out.append((s, l, q))
                elif name == 'tf':
                    tf_out.append((s, l, q))

        yield {
            'name': r.query_name,
            'chrom': r.reference_name,
            'ref_start': r.reference_start,
            'ref_end': r.reference_end,
            'read_len': parsed['read_length'],
            'nuc': nuc_out,
            'msp': parsed['msp'],
            'tf': tf_out,
            'fp_v2': parsed['fp_v2'],
        }
    bam.close()
