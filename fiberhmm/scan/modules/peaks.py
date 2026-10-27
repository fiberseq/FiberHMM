"""``peaks``: accessibility peaks with tiers, ported from FiberBrowser's peak scan.

Each unit counts, per base of its zone and per dataset, the reads whose
compact tiling covers the base (``den``) and those whose tile there is a gap
or a sub-90 bp footprint (``num``), with FiberBrowser's loader rules
(:mod:`fiberhmm.scan.peaks_reads`). ``stitch`` joins the zones of a contig,
pools the datasets, smooths and calls tiers (:mod:`fiberhmm.scan.peaks_core`).
Per-base counts are position-local, so per-position scores equal the browser's
for any view; at a territory edge the smoothing pads with zeros as the
browser does at a view edge.

Tier rules (``tier_rule``): ``local`` (default; owner decision 26 Oct 2026)
needs ``min_reads`` molecules at each base, for every scan; ``window`` is
FiberBrowser's earlier rare tier (it uses the window's read count, so it is
accepted only when every unit is one BED/region unit, i.e. the browser's view);
``auto`` is the pre-decision default, kept for one release: ``window`` for
scans of lone BED/region units, ``local`` for grid scans. Row rules
(``row_rule``) likewise: ``records`` (default; every primary and supplementary
record), ``view`` (BED/region units only) or ``auto``. The effective rules are
recorded in ``peaks/extra/calibration.json`` and in every hit.
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
    version = '1.1'
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
        schema['tier_rule'] = {'default': 'local', 'type': 'str',
                               'help': "rare tier: local (default: min_reads molecules at each base), window "
                                       "(FiberBrowser's earlier rule: max(threshold_rare, min_reads / reads in the "
                                       "window); BED/region units only) or auto (the earlier default, kept for one "
                                       "release: window for scans of lone BED/region units, local for grid scans)"}
        for k, v in READ_OPTIONS.items():
            schema[k] = {'default': v, 'type': type(v).__name__,
                         'help': f'FiberBrowser read loading option {k} (its working-set default)'}
        schema['row_rule'] = {'default': 'records', 'type': 'str',
                              'help': "records (default: every primary and supplementary record), view "
                                      "(FiberBrowser's earlier rows: a supplementary counts only when its primary "
                                      "is out of the view, so it depends on the view's extent; BED/region units "
                                      "only) or auto (the earlier default, kept for one release: view for scans of "
                                      "lone BED/region units, records for grid scans)"}
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
        rows = {'records': 'one per primary and supplementary record',
                'view': 'one per primary record; a supplementary only when its primary is not in view',
                'auto': 'view rule for scans of lone BED/region units, records rule for grid scans'}
        return dict(loader='fiberhmm.scan.peaks_reads.compact_rows (FiberBrowser compact reads, BAM path)',
                    **{k: settings[k] for k in READ_OPTIONS}, spanning_only=False, min_fp_quality=0,
                    rows=rows[settings['row_rule']])

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
        # Circular contigs (@SQ TP:circular): FiberBrowser joins records that run past the contig end with
        # their wrapped tails; that is not ported, so these contigs are not scanned for peaks.
        circular = set()
        import pysam
        for d in inputs.datasets:
            for path in d.paths:
                with pysam.AlignmentFile(path, 'rb', check_sq=False) as bam:
                    circular.update(sq['SN'] for sq in bam.header.to_dict().get('SQ', [])
                                    if str(sq.get('TP', '')).lower() == 'circular')
        # every circular contig of the headers, not only the plan's: the record does not depend on the plan, so
        # a scan extended to other contigs keeps its calibration (and its unit keys)
        return dict(tier_rule=rule, row_rule=rows, datasets={str(d.ordinal): d.dataset_id for d in inputs.datasets},
                    circular_contigs_skipped=sorted(circular))

    def run_unit(self, unit, inputs, settings, calibration, upstream, work, progress):
        import pysam
        from ..peaks_reads import compact_rows
        lo, hi = unit.zone_start, unit.zone_end
        if unit.chrom in (calibration or {}).get('circular_contigs_skipped', ()):
            return UnitResult(tracks=None, extra=dict(skipped='circular contig'))
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

    def stitch_support_bp(self, settings):
        # the smoothing kernel's reach and length: runs padded by it compute as one whole-contig array
        return int(settings['smooth_width']) + int(settings['min_width'])

    def stitch(self, chrom, territory, tracks, settings, calibration):
        rule = (calibration or {}).get('tier_rule', 'local')
        out = StitchResult()
        ordinals = sorted({int(k.split('.', 1)[1]) for k in tracks if k.startswith('num.')})
        params = {k: settings[k] for k in DEFAULTS}
        # an open end (a planned zone without a result, or the edge of a --zones plan) zero-pads the smoothing
        # there; a hit that may depend on it (edge_dependent_regions: through the smoothing and the tiers'
        # masks), or that lies within the stitch support of it, is flagged open_end
        reach = int(self.stitch_support_bp(settings))
        for i, (start, end, left_open, right_open) in enumerate(territory):
            nums = {o: tracks[f'num.{o}'][i] for o in ordinals}
            dens = {o: tracks[f'den.{o}'][i] for o in ordinals}
            num = np.sum(np.stack([nums[o] for o in ordinals]), axis=0, dtype=np.int64).astype(np.int32) \
                if len(ordinals) > 1 else nums[ordinals[0]]
            den = np.sum(np.stack([dens[o] for o in ordinals]), axis=0, dtype=np.int64).astype(np.int32) \
                if len(ordinals) > 1 else dens[ordinals[0]]
            n_reads = int(tracks['nreads'][i][0]) if rule == 'window' else 0
            res = scan_peaks(num, den, n_reads=n_reads, rule=rule, params=params)
            dependent = edge_dependent_regions(res, den, params, rule, left_open, right_open) \
                if (left_open or right_open) else set()
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
                    if (left_open and s < reach) or (right_open and e > len(den) - reach) or \
                            (s, e, int(p['layer'])) in dependent:
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


def edge_dependent_regions(res, total, params, rule, left_open, right_open):
    """(start, end, layer) of every tier region (peak or broad stretch) of one territory run that may differ
    from the whole contig's because the run has an open end.

    The smoothed signal can differ within ``smooth_width`` of an open end (all of it when the run is shorter
    than the kernel, which is then clamped); per-base totals never differ. Outside the uncertain bases the
    signal and the masks are the same as on the whole contig, so a region of a tier can differ only if it, or
    a base next to it, is uncertain. Such a region's mask (``smooth_width // 4`` on each side) is uncertain
    in the next tiers, as is the band ``smooth_width // 4`` around every uncertain base (a region the whole
    contig has there and this run does not). The loop replays ``peaks_core.detect_threshold_layers``."""
    from ..peaks_core import _find_threshold_regions
    smoothed = np.asarray(res['smoothed'])
    L = len(smoothed)
    sw = int(params['smooth_width'])
    sw4 = sw // 4
    unsure = np.zeros(L, dtype=bool)
    if L <= sw:
        unsure[:] = True
    else:
        if left_open:
            unsure[:sw] = True
        if right_open:
            unsure[max(0, L - sw):] = True
    masked = np.zeros(L, dtype=bool)
    local_ok = (np.asarray(total) >= int(params['min_reads'])) if rule == 'local' else None
    max_width = int(params.get('max_peak_width') or 0)
    out = set()

    def dilate(mask, k):
        if k <= 0 or not mask.any():
            return mask.copy()
        idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
        grown = np.zeros(L, dtype=bool)
        for a, b in zip(idx[::2], idx[1::2]):
            grown[max(0, a - k):min(L, b + k)] = True
        return grown
    for layer, threshold, _label in res['thresholds']:
        starts, ends = _find_threshold_regions(smoothed, threshold, masked, local_ok if layer == 3 else None)
        new_unsure = dilate(unsure, sw4)
        for a, b in zip(starts, ends):
            a, b = int(a), int(b)
            if unsure[max(0, a - 1):min(L, b + 1)].any():
                out.add((a, b, layer))
                new_unsure[max(0, a - sw4):min(L, b + sw4)] = True
            width = b - a
            if width < int(params['min_width']):
                continue
            if max_width > 0 and width > max_width:
                masked[max(0, a - sw4):min(L, b + sw4)] = True
                continue
            if layer == 3:
                pc = np.asarray(total)[a:b]
                if (int(pc.mean()) if pc.size > 0 else 0) < int(params['min_reads']):
                    continue
            masked[max(0, a - sw4):min(L, b + sw4)] = True
        unsure = new_unsure
    return out
