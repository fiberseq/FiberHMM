"""``peaks``: accessibility peaks with tiers, ported from FiberBrowser's peak scan.

Each unit counts, per base of its zone and per dataset, the reads whose
compact tiling covers the base (``den``) and those whose tile there is a gap
or a sub-90 bp footprint (``num``), with FiberBrowser's loader rules
(:mod:`fiberhmm.scan.peaks_reads`). ``stitch`` joins the zones of a contig,
pools the datasets, smooths and calls tiers (:mod:`fiberhmm.scan.peaks_core`).
Per-base counts are position-local, so per-position scores equal the browser's
for any view; at a territory edge the smoothing pads with zeros as the
browser does at a view edge.

Tier rules (``tier_rule``): ``window`` is FiberBrowser's rare tier (it uses the
window's read count, so it is accepted only when every unit is one BED/region
unit, i.e. the browser's view); ``local`` needs ``min_reads`` molecules at each
base; ``auto`` (default) picks ``window`` for BED scans and ``local`` for grid
scans. The effective rule is recorded in ``peaks/extra/calibration.json`` and
in every hit.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from . import ScanModuleBase
from ..peaks_core import (DEFAULTS, TILING_SOURCES, compute_scan_counts, scan_peaks, scan_signal_from_counts,
                          smooth_scan_signal)
from ..types import FeatureTypeDecl, Hit, ScoreDecl, StitchResult, UnitResult, Value

TIER_LABELS = ('Strong', 'Medium', 'Weak', 'Rare')
READ_OPTIONS = dict(hide_terminal_gaps=True, terminal_gap_threshold=500, short_terminal_footprint_bp=90,
                    include_duplicates=False)


class PeaksModule(ScanModuleBase):
    name = 'peaks'
    version = '1.0'
    experimental = False
    reconcile = 'stitch'
    max_support_bp = 2_000
    wants_molecules = False

    def parameter_schema(self) -> dict:
        schema = {k: {'default': v, 'type': type(v).__name__, 'help': f'FiberBrowser peak scan {k}'}
                  for k, v in DEFAULTS.items()}
        schema['signal_source'] = {'default': 'accessibility', 'type': 'str',
                                   'help': "accessibility (gaps and sub-90 bp footprints among all tiles) or "
                                           "tf_density (20-64 bp footprints among footprint tiles)"}
        schema['tier_rule'] = {'default': 'auto', 'type': 'str',
                               'help': "rare tier: window (FiberBrowser's: max(threshold_rare, min_reads / reads in "
                                       "the window); BED/region units only), local (min_reads molecules at each "
                                       "base) or auto (window for BED scans, local for grid scans)"}
        for k, v in READ_OPTIONS.items():
            schema[k] = {'default': v, 'type': type(v).__name__,
                         'help': f'FiberBrowser read loading option {k} (its working-set default)'}
        schema['row_rule'] = {'default': 'auto', 'type': 'str',
                              'help': "view (FiberBrowser's rows: a supplementary counts only when its primary is "
                                      "out of the view, so it depends on the view's extent; BED/region units only), "
                                      "records (every primary and supplementary record) or auto (view for BED "
                                      "scans, records for grid scans)"}
        return schema

    def resolve(self, params):
        out = super().resolve(params)
        src = out['signal_source']
        if src not in TILING_SOURCES:
            raise ValueError(f"peaks: signal_source {src!r}: use {' or '.join(TILING_SOURCES)} (FIRE and "
                             "layer:<name> peaks stay in FiberBrowser for now)")
        if out['tier_rule'] not in ('auto', 'window', 'local'):
            raise ValueError("peaks: tier_rule must be auto, window or local")
        if out['row_rule'] not in ('auto', 'view', 'records'):
            raise ValueError("peaks: row_rule must be auto, view or records")
        for k in ('min_width', 'min_reads', 'smooth_width', 'max_peak_width', 'terminal_gap_threshold',
                  'short_terminal_footprint_bp'):
            if int(out[k]) != out[k] or out[k] < 0:
                raise ValueError(f'peaks: {k} must be a non-negative integer')
            out[k] = int(out[k])
        if out['smooth_width'] < 1:
            raise ValueError('peaks: smooth_width must be >= 1')
        for k in ('threshold_strong', 'threshold_medium', 'threshold_weak', 'threshold_rare'):
            out[k] = float(out[k])
        return out

    def feature_types(self):
        cols = ('tier', 'tier_label', 'threshold', 'width', 'summit', 'mean_reads', 'tier_rule', 'row_rule')
        scores = (ScoreDecl('score', 'Peak score', 'fraction', 'high', True),
                  ScoreDecl('tier', 'Tier', 'code', 'low', False))
        return [FeatureTypeDecl('peaks.peak', 'peaks', 'Accessibility peak', 'Accessibility peaks', ('hit',), 'summit',
                                scores=scores, columns=cols),
                FeatureTypeDecl('peaks.broad', 'peaks', 'Broad accessible stretch', 'Broad accessible stretches',
                                ('hit',), 'summit', scores=scores, columns=cols)]

    def load_policy(self, settings):
        return dict(loader='fiberhmm.scan.peaks_reads.compact_rows (FiberBrowser compact reads, BAM path)',
                    **{k: settings[k] for k in READ_OPTIONS}, spanning_only=False, min_fp_quality=0,
                    rows='one per primary record; a supplementary only when its primary is not in view')

    def calibrate(self, inputs, units, settings, work, progress):
        single = all(u.mode == 'bed' and u.left is None and u.right is None for u in units)
        rule = settings['tier_rule']
        if rule == 'auto':
            rule = 'window' if single else 'local'
        if rule == 'window' and not single:
            raise ValueError("peaks: tier_rule 'window' uses the window's read count, so it needs every unit to be "
                             "one BED/region unit (a browser view); use tier_rule 'local' for grid scans")
        rows = settings['row_rule']
        if rows == 'auto':
            rows = 'view' if single else 'records'
        if rows == 'view' and not single:
            raise ValueError("peaks: row_rule 'view' depends on the view's extent, so it needs every unit to be one "
                             "BED/region unit; use row_rule 'records' for grid scans")
        return dict(tier_rule=rule, row_rule=rows, datasets={str(d.ordinal): d.dataset_id for d in inputs.datasets})

    def run_unit(self, unit, inputs, settings, calibration, upstream, work, progress):
        import pysam
        from ..peaks_reads import compact_rows
        lo, hi = unit.zone_start, unit.zone_end
        tracks: Dict[str, Any] = {}
        n_reads = 0
        for ds in inputs.datasets:
            num = np.zeros(hi - lo, dtype=np.int32)
            den = np.zeros(hi - lo, dtype=np.int32)
            for path in ds.paths:
                with pysam.AlignmentFile(path, 'rb') as bam:
                    if unit.chrom not in bam.references:
                        continue
                    rows = compact_rows(bam, unit.chrom, lo, hi, hide_terminal_gaps=settings['hide_terminal_gaps'],
                                        terminal_gap_threshold=settings['terminal_gap_threshold'],
                                        short_terminal_footprint_bp=settings['short_terminal_footprint_bp'],
                                        include_duplicates=settings['include_duplicates'],
                                        row_rule=(calibration or {}).get('row_rule', 'view'))
                    n_reads += len(rows)
                    a, b = compute_scan_counts(rows, settings['signal_source'], hi - lo)
                    num += a
                    den += b
            tracks[f'num.{ds.ordinal}'] = num
            tracks[f'den.{ds.ordinal}'] = den
        if (calibration or {}).get('tier_rule') == 'window':
            tracks['nreads'] = np.full(hi - lo, n_reads, dtype=np.int64)
        return UnitResult(tracks=tracks, extra=dict(n_reads=n_reads))

    def stitch(self, chrom, territory, tracks, settings, calibration):
        rule = (calibration or {}).get('tier_rule', 'local')
        out = StitchResult()
        ordinals = sorted({int(k.split('.', 1)[1]) for k in tracks if k.startswith('num.')})
        params = {k: settings[k] for k in DEFAULTS}
        for i, (start, end, left_open, right_open) in enumerate(territory):
            nums = {o: tracks[f'num.{o}'][i] for o in ordinals}
            dens = {o: tracks[f'den.{o}'][i] for o in ordinals}
            num = np.sum(np.stack([nums[o] for o in ordinals]), axis=0, dtype=np.int64).astype(np.int32) \
                if len(ordinals) > 1 else nums[ordinals[0]]
            den = np.sum(np.stack([dens[o] for o in ordinals]), axis=0, dtype=np.int64).astype(np.int32) \
                if len(ordinals) > 1 else dens[ordinals[0]]
            n_reads = int(tracks['nreads'][i][0]) if rule == 'window' else 0
            res = scan_peaks(num, den, n_reads=n_reads, rule=rule, params=params)
            names = (calibration or {}).get('datasets') or {}
            per_ds = {}
            if len(ordinals) > 1:
                for o in ordinals:
                    per_ds[o] = smooth_scan_signal(scan_signal_from_counts(nums[o], dens[o]), settings['smooth_width'])
            for kind, items in (('peaks.peak', res['peaks']), ('peaks.broad', res['broad'])):
                for p in items:
                    s, e = int(p['start']), int(p['end'])
                    seg = res['smoothed'][s:e]
                    summit = s + int(np.argmax(seg)) if len(seg) else s
                    flags = []
                    if (s == 0 and left_open) or (e == len(den) and right_open):
                        flags.append('open_end')
                    a, b = start + s, start + e
                    key = f'{kind}:{a}-{b}'
                    tier = int(p['layer'])
                    out.hits.append(Hit(kind, key, chrom, a, b, start + summit, score=float(p['score']),
                                        label=TIER_LABELS[tier], flags=tuple(flags),
                                        columns=dict(tier=tier, tier_label=TIER_LABELS[tier],
                                                     threshold=float(p['threshold']), width=e - s,
                                                     summit=start + summit,
                                                     mean_reads=float(den[s:e].mean()) if e > s else 0.0,
                                                     tier_rule=rule,
                                                     row_rule=(calibration or {}).get('row_rule', 'view'))))
                    out.values.append(Value(key, 'all', 'all', 'score', float(p['score']),
                                            n_molecules=int(den[s:e].max()) if e > s else 0))
                    out.values.append(Value(key, 'all', 'all', 'mean_signal', float(seg.mean()) if len(seg) else None))
                    for o, sm in per_ds.items():
                        ds_seg = sm[s:e]
                        out.values.append(Value(key, names.get(str(o), str(o)), 'dataset', 'score',
                                                float(ds_seg.max()) if len(ds_seg) else None,
                                                n_molecules=int(dens[o][s:e].max()) if e > s else 0))
                        out.values.append(Value(key, names.get(str(o), str(o)), 'dataset', 'mean_signal',
                                                float(ds_seg.mean()) if len(ds_seg) else None))
        return out
