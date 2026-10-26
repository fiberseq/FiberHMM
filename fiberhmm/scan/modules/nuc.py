"""``nuc``: nucleosome sizes, segments and array patches as a scan module.

Wraps :mod:`fiberhmm.nuc` (NUC design; feature-hub CONTRACTS 11):

- ``calibrate``: once per scan, per dataset, on a declared sample of units
  (stable unit-id hash), by thinning (default) or self calibration. The
  record is ``{'schema': 'fiberhmm.nuc.scan_calibration.v1', 'datasets':
  {dataset_id: NucCalibration dict}}``.
- ``molecule_outputs``: per owned source record, from the complete record:
  ``nucsz``/``nucseg`` marks (molecular intervals with their reference
  envelopes) and the ``nucmol`` scalars.
- ``run_unit``: per-base core tracks over the unit's zone, per observation
  group (``all``, each dataset, each dataset haplotype): molecules with a
  regular / irregular / low-information / sub-rich segment at each base, the
  measured denominator (aligned and not low-information) and the summed
  q-hat of regular segments; plus phased-window statistics on a fixed genome
  grid (``phase_window_bp`` windows every ``phase_step_bp``), each written by
  the zone holding the window start, from molecular distances to the window
  start (indels never stretch a phase). Molecules below the calibration's
  low-information floor, too short to segment, or with nothing to segment
  are not measured. Nothing depends on the tiling.
- ``stitch``: patches from the concatenated tracks: per-base fractions
  smoothed over ``patch_smooth_bp`` and thresholded (regular 0.5, irregular
  0.35, sub-rich 0.25 by default), at least ``patch_min_bp`` long with at
  least ``patch_min_measured`` measured molecules; phased patches where a
  window's resultant exceeds its null by ``phase_min_excess`` with
  p < ``phase_max_p``.

Phased windows use the large-sample form of the molecule-level phase
randomisation null: with V_m the molecule sums, E_null|S| = sqrt(pi/4 sum
|V_m|^2) and P(|S| >= s) = exp(-s^2 / sum |V_m|^2). It is deterministic, so
stitching does not depend on random draws.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import ScanModuleBase
from ..types import (FeatureTypeDecl, Hit, MAGroupDecl, Mark, MoleculeResult, ScalarCodec, ScoreDecl,
                     StitchResult, UnitResult, Value)

CAL_SCHEMA = 'fiberhmm.nuc.scan_calibration.v1'
SEG_METRICS = ('reg', 'irr', 'low', 'sub')
PATCH_TYPES = (('regular', 'reg', 'patch_tau_regular'), ('irregular', 'irr', 'patch_tau_irregular'),
               ('sub_rich', 'sub', 'patch_tau_sub_rich'))


def _p(default, typ, help_):
    return {'default': default, 'type': typ, 'help': help_}


class NucModule(ScanModuleBase):
    name = 'nuc'
    version = '1.0'
    experimental = True
    reconcile = 'stitch'
    max_support_bp = 12_500
    wants_molecules = True

    def __init__(self):
        self._cals: Dict[Tuple[int, str], Any] = {}
        self._ctx: Dict[Tuple[str, int], Any] = {}

    # ------------------------------------------------------------------ declarations
    def parameter_schema(self) -> dict:
        return {
            'params': _p({}, 'dict', 'fiberhmm.nuc.NucParams overrides (see fiberhmm.nuc.params)'),
            'calibration': _p('thin', 'str', "dataset calibration: 'thin' (thinning, NUC 4.1) or 'self'"),
            'calibration_population': _p(2000, 'int', 'molecules per dataset in the calibration population'),
            'calibration_seeds': _p(5, 'int', 'thinning repeats'),
            'calibration_targets': _p(8, 'int', 'thinning target lambda values'),
            'calibration_units': _p(40, 'int', 'units sampled (stable hash) for the calibration population'),
            'groups': _p('dataset+haplotype', 'str', "observation groups: 'dataset' or 'dataset+haplotype'"),
            'patch_smooth_bp': _p(200, 'int', 'smoothing window for per-base fractions'),
            'patch_tau_regular': _p(0.5, 'float', 'regular patch threshold'),
            'patch_tau_irregular': _p(0.35, 'float', 'irregular patch threshold'),
            'patch_tau_sub_rich': _p(0.25, 'float', 'sub-rich patch threshold'),
            'patch_min_bp': _p(800, 'int', 'minimum patch length'),
            'patch_min_measured': _p(10, 'int', 'minimum measured molecules'),
            'phase_window_bp': _p(1200, 'int', 'phased-window length on the fixed grid'),
            'phase_step_bp': _p(200, 'int', 'phased-window step'),
            'phase_min_excess': _p(0.15, 'float', 'R minus null for a phased window'),
            'phase_max_p': _p(0.01, 'float', 'null p for a phased window'),
        }

    def resolve(self, params: Mapping[str, Any]) -> dict:
        from fiberhmm.nuc.params import NucParams
        out = super().resolve(params)
        out['params'] = NucParams.from_dict(out.get('params') or {}).to_dict()
        out['params'].pop('extra', None)
        if out['calibration'] not in ('thin', 'self'):
            raise ValueError("nuc: calibration must be 'thin' or 'self'")
        if out['groups'] not in ('dataset', 'dataset+haplotype'):
            raise ValueError("nuc: groups must be 'dataset' or 'dataset+haplotype'")
        return out

    def feature_types(self) -> List[FeatureTypeDecl]:
        post = ScoreDecl('posterior', 'Posterior', 'fraction', 'high', False)
        frac = ScoreDecl('fraction', 'Fraction of measured molecules', 'fraction', 'high', True)
        qh = ScoreDecl('q_hat', 'Repeat length', 'bp', 'none', False)
        cols = ('q_hat', 'P', 'width', 'open_end')
        types = [
            FeatureTypeDecl('nuc.size.coarse', 'nuc', 'Footprint size class', 'Footprint size classes', ('mark',),
                            'centre', scores=(post, ScoreDecl('z', 'Density-matched z', 'z', 'none', False))),
            FeatureTypeDecl('nuc.size.named', 'nuc', 'Named size class', 'Named size classes', ('mark',),
                            'centre', scores=(post,)),
            FeatureTypeDecl('nuc.seg.regular', 'nuc', 'Regular array segment', 'Regular array segments',
                            ('mark',), 'none', scores=(post, qh)),
            FeatureTypeDecl('nuc.seg.irregular', 'nuc', 'Irregular segment', 'Irregular segments', ('mark',),
                            'none', scores=(post,)),
            FeatureTypeDecl('nuc.seg.low_information', 'nuc', 'Low-information segment',
                            'Low-information segments', ('mark',), 'none', scores=(post,)),
            FeatureTypeDecl('nuc.seg.sub_rich', 'nuc', 'Sub-rich segment', 'Sub-rich segments', ('mark',),
                            'none', scores=(post,)),
            FeatureTypeDecl('nuc.metric', 'nuc', 'Nucleosome metric', 'Nucleosome metrics', ('scalar',), 'none'),
        ]
        for t, _m, _k in PATCH_TYPES:
            types.append(FeatureTypeDecl(f'nuc.patch.{t}', 'nuc', f'{t.replace("_", "-")} patch',
                                         f'{t.replace("_", "-")} patches', ('hit',), 'centre',
                                         scores=(frac, qh) if t == 'regular' else (frac,), columns=cols))
        types.append(FeatureTypeDecl('nuc.patch.phased', 'nuc', 'Phased patch', 'Phased patches', ('hit',),
                                     'centre', scores=(ScoreDecl('R_minus_null', 'Resultant above null',
                                                                 'fraction', 'high', True),
                                                       ScoreDecl('P', 'Period', 'bp', 'none', False)),
                                     columns=cols))
        return types

    def ma_groups(self) -> List[MAGroupDecl]:
        from fiberhmm.nuc.tags import ma_group_declarations
        out = []
        for d in ma_group_declarations():
            out.append(MAGroupDecl(d['group'], d['group_version'], d['kind'], tuple(d['quality_names']),
                                   tuple(sorted(d['state_codes'].items())),
                                   tuple(ScalarCodec(c['name'], c['nbytes'], c['scale'], c['offset'], c['unit'])
                                         for c in d['codecs']),
                                   an=d['an']))
        return out

    def load_policy(self, settings: Mapping[str, Any]) -> dict:
        return {'reader': 'fiberhmm.nuc full-alignment records', 'molecules': 'all',
                'tracks': 'primary and soft-clipped records overlapping the zone (supplementaries skipped)'}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _params(settings):
        from fiberhmm.nuc.params import NucParams
        return NucParams.from_dict(settings.get('params') or {})

    def _cal(self, calibration, dataset_id):
        from fiberhmm.nuc.calibration import NucCalibration
        if calibration is None:
            raise ValueError('nuc: no calibration record')
        key = (id(calibration), dataset_id)
        cal = self._cals.get(key)
        if cal is None:
            d = calibration.get('datasets', {}).get(dataset_id)
            if d is None:
                raise ValueError(f'nuc: calibration has no dataset {dataset_id!r}')
            cal = NucCalibration.from_dict(d)
            self._cals[key] = cal
        return cal

    def _context(self, dataset_id, input_index, header, params):
        from fiberhmm.nuc.records import DecodeContext
        key = (dataset_id, int(input_index))
        ctx = self._ctx.get(key)
        if ctx is None:
            ctx = DecodeContext(header, params)
            self._ctx[key] = ctx
        return ctx

    # ------------------------------------------------------------------ calibration
    def calibrate(self, inputs, units, settings, work: Path, progress) -> Optional[dict]:
        from fiberhmm.nuc.calibration import calibrate_self
        from fiberhmm.nuc.thinning import calibrate_thin, load_population
        params = self._params(settings)
        n_units = int(settings['calibration_units'])
        ranked = sorted(units, key=lambda u: hashlib.sha256(u.unit_id.encode()).hexdigest())
        sample = sorted(ranked[:n_units] if n_units else ranked, key=lambda u: (u.chrom, u.start))
        regions = [(u.chrom, u.start, u.end) for u in sample]
        out = {'schema': CAL_SCHEMA, 'source': settings['calibration'],
               'units': [u.unit_id for u in sample], 'datasets': {}}
        for ds in inputs.datasets:
            progress.report('nuc', f'calibrating {ds.dataset_id} ({settings["calibration"]})')
            if settings['calibration'] == 'thin':
                cal = calibrate_thin(list(ds.paths), params, label=ds.dataset_id,
                                     n_population=int(settings['calibration_population']),
                                     seeds=int(settings['calibration_seeds']),
                                     n_targets=int(settings['calibration_targets']),
                                     cores=max(1, int(inputs.cores)), regions=regions)
            else:
                mols, _frs, _s, chem = load_population(list(ds.paths), params,
                                                       int(settings['calibration_population']), regions)
                cal = calibrate_self(mols, params, label=ds.dataset_id, chemistry=chem,
                                     max_molecules=int(settings['calibration_population']))
            out['datasets'][ds.dataset_id] = cal.to_dict()
        return out

    # ------------------------------------------------------------------ molecule-wide outputs
    def molecule_outputs(self, molecule, settings, calibration) -> Optional[MoleculeResult]:
        from fiberhmm.nuc.molecule import molecule_outputs as nuc_outputs
        from fiberhmm.nuc.records import decode_record
        from fiberhmm.nuc.tags import NUCMOL, encode_nucmol, group_rows  # noqa: F401
        params = self._params(settings)
        rec = molecule.record
        read = molecule.read
        cal = self._cal(calibration, rec.dataset_id)
        ctx = self._context(rec.dataset_id, rec.input_index, read.header, params)
        mol, _why = decode_record(read, ctx)
        if mol is None:
            return None
        res = nuc_outputs(mol, params, cal)
        marks: List[Mark] = []
        for group, (spec, rows) in group_rows(res).items():
            if group == NUCMOL:
                continue
            for s, n, q, name in rows:
                a, b = _ref_envelope(mol, s, s + n)
                marks.append(Mark(rec, group, rec.chrom, a, b, tuple(int(x) for x in q), name, (int(s), int(n))))
        scalars = {k: (None if v is None or not np.isfinite(v) else float(v)) for k, v in res.scalars.items()}
        from fiberhmm.nuc.tags import FLAG_BITS
        scalars['flags'] = float(sum(1 << i for i, k in enumerate(FLAG_BITS) if res.flags.get(k)))
        from fiberhmm.nuc.tags import digest_pieces
        scalars['calls_digest_hi'], scalars['calls_digest_lo'] = digest_pieces(res.digest)
        return MoleculeResult(rec, scalars, marks)

    # ------------------------------------------------------------------ unit tracks
    def run_unit(self, unit, inputs, settings, calibration, upstream, work: Path, progress) -> UnitResult:
        import pysam
        from fiberhmm.nuc.molecule import molecule_outputs as nuc_outputs
        from fiberhmm.nuc.records import decode_record
        from fiberhmm.nuc.segment import IRREGULAR, LOWINFO, REGULAR
        from fiberhmm.scan.reads import iter_records
        params = self._params(settings)
        z0, z1 = unit.zone_start, unit.zone_end
        n = z1 - z0
        W, step = int(settings['phase_window_bp']), int(settings['phase_step_bp'])
        haps = settings['groups'] == 'dataset+haplotype'
        filters = inputs.read_policy['filters']
        tracks: Dict[str, np.ndarray] = {}

        def arr(name):
            a = tracks.get(name)
            if a is None:
                a = np.zeros(n, dtype=np.float64)
                tracks[name] = a
            return a
        win_starts = [w for w in range((z0 + step - 1) // step * step, z1, step)]
        phase_mols: Dict[str, List[Tuple[np.ndarray, int, int]]] = {}
        # a fixed group (and so track) set for every unit: territory runs
        # concatenate tracks by name, zone by zone
        groups_seen = ['all']
        for ds in inputs.datasets:
            groups_seen.append(ds.dataset_id)
            if haps:
                groups_seen += [f'{ds.dataset_id}:HP1', f'{ds.dataset_id}:HP2']
        for ds in inputs.datasets:
            cal = self._cal(calibration, ds.dataset_id)
            for i, path in enumerate(ds.paths):
                with pysam.AlignmentFile(path, 'rb') as bam:
                    if unit.chrom not in bam.references:
                        continue
                    ctx = self._context(ds.dataset_id, i, bam.header, params)
                    for read, _sha, _occ in iter_records(bam, unit.chrom, z0, z1 + W, filters):
                        if read.is_supplementary:
                            continue
                        mol, _why = decode_record(read, ctx)
                        if mol is None:
                            continue
                        res = nuc_outputs(mol, params, cal)
                        if res.flags.get('low_information') or not res.segments:
                            # below the floor (D8), too short or nothing to segment:
                            # not measured anywhere
                            continue
                        grp = ['all', ds.dataset_id]
                        if haps and mol.hp in (1, 2):
                            grp.append(f'{ds.dataset_id}:HP{mol.hp}')
                        # measured = aligned bases minus low-information segments
                        cov = np.zeros(n, dtype=bool)
                        if mol.blocks is not None:
                            for _q0, r0, ln in mol.blocks:
                                a, b = max(int(r0), z0), min(int(r0 + ln), z1)
                                if b > a:
                                    cov[a - z0:b - z0] = True
                        segmask = {k: np.zeros(n, dtype=bool) for k in SEG_METRICS}
                        qsum = np.zeros(n)
                        for sgm in res.segments:
                            env = mol.mol_interval_to_ref(sgm.start, sgm.end)
                            if env is None:
                                continue
                            a, b = max(env[0], z0), min(env[1], z1)
                            if b <= a:
                                continue
                            key = {REGULAR: 'reg', IRREGULAR: 'irr', LOWINFO: 'low'}[sgm.state]
                            segmask[key][a - z0:b - z0] = True
                            if sgm.sub_rich:
                                segmask['sub'][a - z0:b - z0] = True
                            if sgm.state == REGULAR and sgm.q_hat:
                                qsum[a - z0:b - z0] = sgm.q_hat
                        meas = cov & ~segmask['low']
                        for g in grp:
                            arr(f'measured:{g}')[meas] += 1
                            for k in SEG_METRICS:
                                arr(f'{k}:{g}')[segmask[k] & cov] += 1
                            arr(f'qsum:{g}')[segmask['reg'] & cov] += qsum[segmask['reg'] & cov]
                        # dyads for phased windows: molecular positions of observed mono
                        # dyads (distances are measured along the molecule; the
                        # reference position only decides window membership)
                        fp = res.footprints
                        if fp is not None and not res.flags.get('low_information'):
                            from fiberhmm.nuc.sizes import MONO
                            sel = (fp['coarse'] == MONO) & ~fp['is_tf'] & ~fp['truncated']
                            c = ((fp['start'][sel] + fp['end'][sel]) / 2.0)
                            ref = mol.mol_to_ref(np.round(c).astype(np.int64)) if len(c) else np.zeros(0, np.int64)
                            ok = ref >= 0
                            for g in grp:
                                phase_mols.setdefault(g, []).append((mol, c[ok], ref[ok]))
        # fixed-grid phased windows written by the zone holding the window start
        qgrid = np.arange(165.0, 216.0, 1.0)
        for g in groups_seen:
            Rt, Nt, nullt, logpt, Pt = (np.full(n, np.nan) for _ in range(5))
            mols = phase_mols.get(g, [])
            for w in win_starts:
                xs = []
                for mol, c, ref in mols:
                    if not (mol.ref_start <= w and mol.ref_end >= w + W):
                        continue
                    inw = (ref >= w) & (ref < w + W)
                    if not inw.any():
                        continue
                    a = int(mol.ref_to_mol([w])[0])
                    if a < 0:
                        continue
                    sgn = -1.0 if mol.is_reverse else 1.0
                    xs.append(sgn * (c[inw] - a))
                xs = [x for x in xs if len(x)]
                if len(xs) < int(settings['patch_min_measured']):
                    continue
                # P maximising the mean per-molecule resultant (phase-free)
                best_p, best = None, -1.0
                for P in qgrid:
                    r = np.mean([abs(np.exp(2j * np.pi * x / P).sum()) / len(x) for x in xs])
                    if r > best:
                        best, best_p = r, P
                V = np.array([np.exp(2j * np.pi * x / best_p).sum() for x in xs])
                N = float(sum(len(x) for x in xs))
                S = abs(V.sum())
                var = float(np.sum(np.abs(V) ** 2))
                k = w - z0
                Rt[k] = S / N
                nullt[k] = np.sqrt(np.pi / 4.0 * var) / N
                logpt[k] = -(S * S) / var if var > 0 else 0.0
                Nt[k] = len(xs)
                Pt[k] = best_p
            tracks[f'phase_R:{g}'] = Rt
            tracks[f'phase_null:{g}'] = nullt
            tracks[f'phase_logp:{g}'] = logpt
            tracks[f'phase_n:{g}'] = Nt
            tracks[f'phase_P:{g}'] = Pt
        for g in groups_seen:
            arr(f'measured:{g}')
            for k in SEG_METRICS + ('qsum',):
                arr(f'{k}:{g}')
        return UnitResult(status='ok', tracks=tracks, extra={'groups': groups_seen})

    # ------------------------------------------------------------------ stitching
    def stitch(self, chrom, territory, tracks, settings, calibration) -> StitchResult:
        out = StitchResult()
        groups = sorted({name.split(':', 1)[1] for name in tracks if name.startswith('measured:')})
        if 'all' not in groups:
            return out
        smooth = int(settings['patch_smooth_bp'])
        min_bp = int(settings['patch_min_bp'])
        min_meas = float(settings['patch_min_measured'])
        W = int(settings['phase_window_bp'])
        for run_index, (t0, t1, left_open, right_open) in enumerate(territory):
            def get(name):
                arrs = tracks.get(name)
                if arrs is None or arrs[run_index] is None:
                    return None
                return np.asarray(arrs[run_index], dtype=float)
            den_all = get('measured:all')
            if den_all is None:
                continue
            sden = _smooth(den_all, smooth)
            for ptype, metric, tau_key in PATCH_TYPES:
                num = get(f'{metric}:all')
                snum = _smooth(num, smooth)
                with np.errstate(invalid='ignore', divide='ignore'):
                    frac = np.where(sden > 0, snum / sden, 0.0)
                ok = (frac >= float(settings[tau_key])) & (den_all >= min_meas)
                for a, b in _runs(ok):
                    if b - a < min_bp:
                        continue
                    s, e = t0 + a, t0 + b
                    open_end = (a == 0 and left_open) or (b == len(ok) and right_open)
                    vals = []
                    for g in groups:
                        if get(f'measured:{g}') is None:
                            continue
                        dg, ng = get(f'measured:{g}')[a:b], get(f'{metric}:{g}')[a:b]
                        fg = float(ng.sum() / dg.sum()) if dg.sum() > 0 else None
                        kind = 'all' if g == 'all' else ('haplotype' if ':HP' in g else 'dataset')
                        vals.append((g, kind, fg, int(np.median(dg)) if len(dg) else 0))
                    qh = None
                    if metric == 'reg':
                        qs, rn = get('qsum:all')[a:b].sum(), get('reg:all')[a:b].sum()
                        qh = float(qs / rn) if rn > 0 else None
                    score = vals[[v[0] for v in vals].index('all')][2]
                    key = f'{ptype}:{s}'
                    out.hits.append(Hit(f'nuc.patch.{ptype}', key, chrom, s, e, (s + e) // 2, score=score,
                                        flags=('open_end',) if open_end else (),
                                        columns={'q_hat': '' if qh is None else round(qh, 1), 'P': '',
                                                 'width': e - s, 'open_end': int(open_end)}))
                    for g, kind, fg, nm in vals:
                        out.values.append(Value(key, g, kind, 'fraction', fg, n_molecules=nm))
                    if qh is not None:
                        out.values.append(Value(key, 'all', 'all', 'q_hat', qh))
            # phased patches: union of qualifying fixed-grid windows
            R, null, logp, P = get('phase_R:all'), get('phase_null:all'), get('phase_logp:all'), get('phase_P:all')
            if R is None:
                continue
            good = np.zeros(len(R), dtype=bool)
            starts = np.flatnonzero(np.isfinite(R) & ((R - null) >= float(settings['phase_min_excess']))
                                    & (logp <= np.log(float(settings['phase_max_p']))))
            for k_ in starts:
                good[k_:min(k_ + W, len(good))] = True
            for a, b in _runs(good):
                s, e = t0 + a, t0 + b
                inwin = [k_ for k_ in starts if a <= k_ < b]
                key = f'phased:{s}'
                open_end = (a == 0 and left_open) or (b == len(good) and right_open)
                ex = float(np.max(R[inwin] - null[inwin]))
                Pm = float(np.median(P[inwin]))
                out.hits.append(Hit('nuc.patch.phased', key, chrom, s, e, (s + e) // 2, score=ex,
                                    flags=('open_end',) if open_end else (),
                                    columns={'q_hat': '', 'P': round(Pm, 1), 'width': e - s,
                                             'open_end': int(open_end)}))
                for g in groups:
                    Rg, ng = get(f'phase_R:{g}'), get(f'phase_null:{g}')
                    if Rg is None:
                        continue
                    vals = [Rg[k_] - ng[k_] for k_ in inwin if np.isfinite(Rg[k_])]
                    kind = 'all' if g == 'all' else ('haplotype' if ':HP' in g else 'dataset')
                    out.values.append(Value(key, g, kind, 'R_minus_null', float(np.max(vals)) if vals else None))
                out.values.append(Value(key, 'all', 'all', 'P', Pm))
        return out

    def scalar_names(self) -> List[str]:
        from fiberhmm.nuc.tags import NUCMOL_CODECS
        return [c[0] for c in NUCMOL_CODECS]


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Centred moving average of window ``w`` with exactly ``len(x)`` values
    (zero padding outside the territory), also when ``len(x) < w``."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or w <= 1:
        return x.copy()
    full = np.convolve(x, np.ones(w) / w, mode='full')
    a = (w - 1) // 2
    return full[a:a + len(x)]


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    m = np.asarray(mask, dtype=np.int8)
    if not len(m):
        return []
    d = np.diff(np.concatenate([[0], m, [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _ref_envelope(mol, s: int, e: int) -> Tuple[int, int]:
    """Reference envelope of molecular [s, e); zero-width at the nearest
    aligned base (by alignment-block boundaries) when none of it is aligned
    (inside a clip or an insertion)."""
    env = mol.mol_interval_to_ref(s, e)
    if env is not None:
        return env
    blocks = mol.blocks
    if blocks is None or len(blocks) == 0:
        p = mol.ref_start if mol.ref_start >= 0 else 0
        return p, p
    if mol.is_reverse:
        ss, se = mol.length - e, mol.length - s
    else:
        ss, se = s, e
    mid = (ss + se) / 2.0
    q0, r0, n = blocks[:, 0], blocks[:, 1], blocks[:, 2]
    cand_q = np.concatenate([q0, q0 + n - 1])
    cand_r = np.concatenate([r0, r0 + n - 1])
    k = int(np.argmin(np.abs(cand_q - mid)))
    p = int(cand_r[k])
    return p, p
