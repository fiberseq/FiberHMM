"""Crop whole-molecule interval lists to the analysis region plus a flank.

Each evidence unit carries the molecule's complete call geometry (nucleosomes,
MSPs, TF calls, native replay calls, the upstream recall and aligned blocks)
across the whole long read, although lattice positions are already limited
to the region. For a 15-30 kb read that is ~200 KB of Python objects per unit,
about 50 times the region's own evidence, and the analysis deep-copies it.

``crop_unit`` keeps every interval that overlaps ``[start - flank, end + flank]``
(whole, never clipped) and drops the rest. It is applied to the analysis copy
only (``prepare_sources``/``_pool``), after the payload's digest and
``evidence.json.gz`` are taken, so input digests, family tokens and saved
evidence are unchanged. The default flank (``compute.evidence_flank_bp`` = 500)
is far wider than any context the engines read outside a window: discovery
keeps lattice sites within 60 bp of a tile, classes are scored over their edge
boxes plus 25 bp, calls are at most 100 bp (identity grid 150 bp), and any
interval overlapping the region is kept whole whatever the flank.
"""
from __future__ import annotations

INTERVAL_FIELDS = ('raw_tf_intervals', 'representative_raw_tf_intervals', 'raw_nuc_intervals', 'msp_intervals',
                   'aligned_blocks', 'original_bam_tf_intervals', 'original_bam_nuc_intervals',
                   'original_bam_msp_intervals', 'native_multi_interval_tf_intervals')
RECALL_INTERVALS = ('nucleosomes', 'msps', 'original_tag_msps_absent_from_loaded_scaffold')
RECALL_CALLS = ('calls', 'excluded_calls')


def crop_unit(unit: dict, start: int, end: int, flank: int) -> dict:
    """A shallow copy of ``unit`` whose interval lists keep only intervals
    overlapping ``[start - flank, end + flank)``; ``flank < 0`` returns ``unit``."""
    if flank is None or flank < 0:
        return unit
    lo, hi = start - flank, end + flank
    keep = lambda iv: iv[0] < hi and iv[1] > lo
    u = dict(unit)
    for key in INTERVAL_FIELDS:
        if key in u and u[key] is not None:
            u[key] = [iv for iv in u[key] if keep(iv)]
    if 'native_multi_interval_calls' in u:
        u['native_multi_interval_calls'] = [c for c in u['native_multi_interval_calls'] if keep(c['interval'])]
    recall = u.get('upstream_nuc_tf_recall')
    if recall:
        recall = dict(recall)
        for key in RECALL_INTERVALS:
            if key in recall:
                recall[key] = [iv for iv in recall[key] if keep(iv)]
        for key in RECALL_CALLS:
            if key in recall:
                recall[key] = [c for c in recall[key] if 'interval' not in c or keep(c['interval'])]
        u['upstream_nuc_tf_recall'] = recall
    return u
