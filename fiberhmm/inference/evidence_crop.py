"""Crop one molecule's loaded evidence to a reference span (window +/- pad).

Long reads (15-30 kb) carry their complete call geometry (nucleosomes, MSPs, TF
calls, aligned blocks, m6A fingerprint) although a regional analysis reads only
the window. Kept whole-read, that is ~280 KB of Python objects per molecule, and
consensus holds several copies of each unit. These helpers keep only what can
influence a window:

* **Calls crossing the pad boundary are kept whole, never clipped or dropped.**
  An interval is kept when it overlaps ``[lo, hi)``; the kept span is then
  widened to the union of the kept intervals (``effective_span``), so aligned
  blocks and fingerprint positions under a kept call survive too. A short pad
  therefore never loses a call that reaches into the padded window (the failure
  mode of pre-trimming a BAM, which drops calls crossing the clip).
* The cached SAM record used for native replay is not cropped: replay re-decodes
  the actual query observations of the whole record, so every replayed call that
  overlaps the span is identical to the full-read run, and its results are
  cropped by the same rule afterwards (``crop_unit_intervals``). The record is
  dropped from the cache right after replay.

Shared by ``fiberhmm-consensus`` (``--evidence-pad``) and any other regional
loader built on ``strand_rescue.load_region_evidence``.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

# Interval-list fields of a consensus evidence unit (genomic [start, end) pairs).
UNIT_INTERVAL_FIELDS = ('raw_tf_intervals', 'representative_raw_tf_intervals', 'raw_nuc_intervals', 'msp_intervals',
                        'aligned_blocks', 'original_bam_tf_intervals', 'original_bam_nuc_intervals',
                        'original_bam_msp_intervals', 'native_multi_interval_tf_intervals')
_RECALL_INTERVALS = ('nucleosomes', 'msps', 'original_tag_msps_absent_from_loaded_scaffold')
_RECALL_CALLS = ('calls', 'excluded_calls')


def parse_pad(value) -> Optional[int]:
    """CLI/parameter value -> pad in bp, or None for the whole read ('full', -1)."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ('full', 'none', 'whole', '-1'):
            return None
        value = int(text)
    value = int(value)
    if value < 0:
        return None
    return value


def pad_span(start: int, end: int, pad: Optional[int]) -> Optional[Tuple[int, int]]:
    return None if pad is None else (int(start) - int(pad), int(end) + int(pad))


def _overlaps(a, b, lo, hi):
    return a < hi and b > lo


def effective_span(lo: int, hi: int, intervals: Iterable[Tuple[int, int]]) -> Tuple[int, int]:
    """[lo, hi) widened to cover every interval that overlaps it (kept whole)."""
    elo, ehi = int(lo), int(hi)
    for a, b in intervals:
        if a < hi and b > lo:
            elo = min(elo, int(a)); ehi = max(ehi, int(b))
    return elo, ehi


def crop_calls(calls: Sequence, lo: int, hi: int) -> list:
    """IntervalCall-like objects (``.start``/``.end``) overlapping [lo, hi), unchanged."""
    return [c for c in calls if c.start < hi and c.end > lo]


def crop_blocks(blocks: Optional[Sequence[Tuple[int, int]]], lo: int, hi: int):
    if blocks is None:
        return None
    return tuple((int(a), int(b)) for a, b in blocks if a < hi and b > lo)


def crop_positions(positions: Optional[np.ndarray], lo: int, hi: int):
    if positions is None:
        return None
    positions = np.asarray(positions)
    return np.ascontiguousarray(positions[(positions >= lo) & (positions < hi)])


def cropped_unit(unit: dict, lo: int, hi: int) -> dict:
    """A shallow copy of ``unit`` cropped like ``crop_unit_intervals``; ``unit`` itself is untouched.

    For the analysis copy of saved (possibly whole-read) evidence: the payload, its digest and
    evidence.json.gz keep what was loaded."""
    u = dict(unit)
    if u.get('upstream_nuc_tf_recall'):
        u['upstream_nuc_tf_recall'] = dict(u['upstream_nuc_tf_recall'])
    return crop_unit_intervals(u, lo, hi)


def analysis_units(units, start: int, end: int, pad: Optional[int]):
    """Units for analysis: cropped copies to [start - pad, end + pad), or the units themselves when pad is None."""
    if pad is None:
        return units
    return [cropped_unit(u, int(start) - int(pad), int(end) + int(pad)) for u in units]


def crop_unit_intervals(unit: dict, lo: int, hi: int) -> dict:
    """In place: keep only intervals of a consensus evidence unit that overlap [lo, hi), whole.

    Covers the loader's interval lists, native replay calls and the Hia5 upstream
    recall result. Lattice positions are already limited to the window."""
    keep = lambda iv: iv[0] < hi and iv[1] > lo
    for key in UNIT_INTERVAL_FIELDS:
        if unit.get(key) is not None:
            unit[key] = [iv for iv in unit[key] if keep(iv)]
    if unit.get('native_multi_interval_calls') is not None:
        unit['native_multi_interval_calls'] = [c for c in unit['native_multi_interval_calls'] if keep(c['interval'])]
    recall = unit.get('upstream_nuc_tf_recall')
    if recall:
        for key in _RECALL_INTERVALS:
            if recall.get(key) is not None:
                recall[key] = [iv for iv in recall[key] if keep(iv)]
        for key in _RECALL_CALLS:
            if recall.get(key) is not None:
                recall[key] = [c for c in recall[key] if 'interval' not in c or keep(c['interval'])]
    return unit
