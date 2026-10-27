"""``nuc``: nucleosome sizes, segments and array patches as a scan module.

Wraps :mod:`fiberhmm.nuc` (NUC design; feature-hub CONTRACTS 11):

- ``calibrate``: once per scan, per dataset, on a declared sample of fixed
  50 kb windows of absolute coordinates taken in a stable hash order, records
  passing the read policy, by thinning (default) or self calibration. With
  ``calibration_scope = 'data'`` (default) the windows are drawn from every
  contig of the reference that holds the dataset's records, so the record
  depends on the inputs and settings only, never on the plan: a view scan, its
  extensions and a genome scan share it (and their unit keys). ``'territory'``
  draws them from the scanned territory (the earlier rule). The
  record is ``{'schema': 'fiberhmm.nuc.scan_calibration.v1', 'datasets':
  {dataset_id: NucCalibration dict}}``.
- ``molecule_outputs``: per owned source record, from the complete record:
  ``nucsz``/``nucseg`` marks (molecular intervals with their reference
  envelopes) and the ``nucmol`` scalars.
- ``run_unit``: per-base core tracks over the unit's zone, per observation
  group (``all``, each dataset, each dataset haplotype): molecules with a
  regular / disordered / undetermined / low-information segment and with an
  undersized run or a missing nucleosome at each base, the measured
  denominator (aligned, inside a decoded segment, not low-information: read
  ends beyond the outer dyads and NFR-length MSPs are not measured) and the
  summed q-hat of regular segments. Two fixed genome grids, each window written by the zone
  holding its start: phased windows (``phase_window_bp`` every
  ``phase_step_bp``, from molecular distances to the window start, so indels
  never stretch a phase; resultant, null, p, period and the coherence of the
  molecules' phases) and recurrence windows (``enrich_window_bp`` every
  ``enrich_step_bp``; the exact shift-within-read null of
  :mod:`fiberhmm.nuc.enrichment` per kind, pooled over every measured
  molecule). Molecules below the calibration's low-information floor, too
  short to segment, or with nothing to segment are not measured. Nothing
  depends on the tiling.
- ``stitch``: recurrence patches per kind (regular, disordered, undersized,
  missing): Benjamini-Hochberg at ``enrich_fdr`` over every tested window of
  the scan (``enrich_fdr_scope = 'scan'``, default: a first pass over all
  contigs gives the threshold, recorded in the manifest as
  ``scan_statistics``) or of the contig (``'contig'``), significant windows
  merged; phased patches where a window's resultant
  exceeds its null by ``phase_min_excess`` with p < ``phase_max_p``; fuzzy
  patches where a phased window starts inside a recurrent regular patch but
  its molecules share no phase (null p > ``fuzzy_min_p``, excess below
  ``phase_min_excess``), scored by the positional dispersion
  P / (2 pi) sqrt(-2 ln coherence).

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
#: per-base tracks: segment states and disruption events
SEG_METRICS = ('reg', 'dis', 'und', 'low', 'uds', 'mis')
#: kinds tested for recurrence, and their patch types
ENRICH_KINDS = ('reg', 'dis', 'uds', 'mis')
PATCH_KINDS = (('reg', 'regular'), ('dis', 'disordered'), ('uds', 'undersized'), ('mis', 'missing'))


def _p(default, typ, help_):
    return {'default': default, 'type': typ, 'help': help_}


class NucModule(ScanModuleBase):
    name = 'nuc'
    version = '2.2'
    experimental = True
    reconcile = 'stitch'
    max_support_bp = 12_500
    wants_molecules = True
    # fiberhmm.nuc.records decodes each annotation's frame from the record's header (DecodeContext), so it gets
    # the stored record: an orchestrator carry would be mirrored a second time on reverse reads
    molecule_frame = 'source'

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
            'calibration_units': _p(40, 'int', '50 kb windows that hold records, taken in stable-hash order of '
                                       'the absolute window index, for the calibration population (per '
                                       'dataset); 0 = all'),
            'calibration_scope': _p('data', 'str', "where calibration windows come from: 'data' (every contig "
                                    "of the reference with the dataset's records; the record does not depend "
                                    "on the plan) or 'territory' (the scanned territory)"),
            'groups': _p('dataset+haplotype', 'str', "observation groups: 'dataset' or 'dataset+haplotype'"),
            'enrich_window_bp': _p(500, 'int', 'recurrence-test window on the fixed grid'),
            'enrich_step_bp': _p(250, 'int', 'recurrence-test window step'),
            'enrich_fdr': _p(0.1, 'float', 'Benjamini-Hochberg FDR of recurrence patches, per kind'),
            'enrich_fdr_scope': _p('scan', 'str', "windows one Benjamini-Hochberg runs over, per kind: 'scan' "
                                   "(every tested window of the scan: the FDR holds for the scan's patch "
                                   "list) or 'contig' (each contig on its own)"),
            'enrich_levels': _p(10, 'int', 'levels of the per-molecule window score'),
            'enrich_min_molecules': _p(3, 'int', 'minimum molecules in a tested window'),
            'patch_min_measured': _p(10, 'int', 'minimum molecules in a phased window'),
            'fuzzy_min_p': _p(0.2, 'float', 'phase null p above which a window in a regular patch is fuzzy'),
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
        if out['calibration_scope'] not in ('data', 'territory'):
            raise ValueError("nuc: calibration_scope must be 'data' or 'territory'")
        if out['enrich_fdr_scope'] not in ('scan', 'contig'):
            raise ValueError("nuc: enrich_fdr_scope must be 'scan' or 'contig'")
        if not 0.0 < float(out['enrich_fdr']) < 1.0:
            raise ValueError('nuc: enrich_fdr must be between 0 and 1')
        return out

    def calibration_uses_plan(self, settings) -> bool:
        return settings['calibration_scope'] == 'territory'

    def feature_types(self) -> List[FeatureTypeDecl]:
        post = ScoreDecl('posterior', 'Posterior', 'fraction', 'high', False)
        frac = ScoreDecl('fraction', 'Fraction of measured molecules', 'fraction', 'high', True)
        qh = ScoreDecl('q_hat', 'Repeat length', 'bp', 'none', False)
        llr = ScoreDecl('llr', 'Log-likelihood ratio', 'nats', 'high', False)
        cols = ('q_hat', 'P', 'width', 'open_end')
        types = [
            FeatureTypeDecl('nuc.size.coarse', 'nuc', 'Footprint size class', 'Footprint size classes', ('mark',),
                            'centre', scores=(post, ScoreDecl('z', 'Density-matched z', 'z', 'none', False))),
            FeatureTypeDecl('nuc.size.named', 'nuc', 'Named size class', 'Named size classes', ('mark',),
                            'centre', scores=(post,)),
            FeatureTypeDecl('nuc.seg.regular', 'nuc', 'Regular array segment', 'Regular array segments',
                            ('mark',), 'none', scores=(post, qh)),
            FeatureTypeDecl('nuc.seg.disordered', 'nuc', 'Disordered spacing', 'Disordered spacing',
                            ('mark',), 'none', scores=(post,)),
            FeatureTypeDecl('nuc.seg.undetermined', 'nuc', 'Undetermined spacing', 'Undetermined spacing',
                            ('mark',), 'none', scores=(post,)),
            FeatureTypeDecl('nuc.seg.low_information', 'nuc', 'Low-information segment',
                            'Low-information segments', ('mark',), 'none', scores=(post,)),
            FeatureTypeDecl('nuc.event.undersized', 'nuc', 'Undersized footprint run', 'Undersized footprint runs',
                            ('mark',), 'none', scores=(llr,)),
            FeatureTypeDecl('nuc.event.missing', 'nuc', 'Missing nucleosome', 'Missing nucleosomes',
                            ('mark',), 'centre', scores=(llr,)),
            FeatureTypeDecl('nuc.metric', 'nuc', 'Nucleosome metric', 'Nucleosome metrics', ('scalar',), 'none'),
        ]
        for _k, t in PATCH_KINDS:
            types.append(FeatureTypeDecl(f'nuc.patch.{t}', 'nuc', f'Recurrent {t} patch',
                                         f'Recurrent {t} patches', ('hit',), 'centre',
                                         scores=(frac, qh) if t == 'regular' else (frac,), columns=cols))
        types.append(FeatureTypeDecl('nuc.patch.phased', 'nuc', 'Phased patch', 'Phased patches', ('hit',),
                                     'centre', scores=(ScoreDecl('R_minus_null', 'Resultant above null',
                                                                 'fraction', 'high', True),
                                                       ScoreDecl('P', 'Period', 'bp', 'none', False)),
                                     columns=cols))
        types.append(FeatureTypeDecl('nuc.patch.fuzzy', 'nuc', 'Fuzzy array patch', 'Fuzzy array patches',
                                     ('hit',), 'centre',
                                     scores=(ScoreDecl('dispersion_bp', 'Positional dispersion', 'bp', 'high',
                                                       False), ScoreDecl('P', 'Period', 'bp', 'none', False)),
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
        from fiberhmm.scan.reads import passes
        params = self._params(settings)
        filters = inputs.read_policy['filters']

        def read_filter(r):
            return passes(r, filters)
        scope = settings['calibration_scope']
        out = {'schema': CAL_SCHEMA, 'source': settings['calibration'], 'scope': scope, 'regions': {},
               'datasets': {}}
        for ds in inputs.datasets:
            progress.report('nuc', f'calibrating {ds.dataset_id} ({settings["calibration"]}, {scope} windows)')
            with _RecordProbe(ds.paths, read_filter) as probe:
                if scope == 'data':
                    regions = data_calibration_regions(inputs.reference, ds.paths,
                                                       int(settings['calibration_units']), has_records=probe)
                else:
                    regions = calibration_regions(units, int(settings['calibration_units']), has_records=probe)
            if not regions:
                raise ValueError(f'nuc: no records passing the read policy for {ds.dataset_id} '
                                 + ('in the reference' if scope == 'data' else 'in the scanned territory'))
            out['regions'][ds.dataset_id] = [f'{c}:{a}-{b}' for c, a, b in regions]
            if settings['calibration'] == 'thin':
                cal = calibrate_thin(list(ds.paths), params, label=ds.dataset_id,
                                     n_population=int(settings['calibration_population']),
                                     seeds=int(settings['calibration_seeds']),
                                     n_targets=int(settings['calibration_targets']),
                                     cores=max(1, int(inputs.cores)), regions=regions,
                                     read_filter=read_filter)
            else:
                mols, _frs, _s, chem = load_population(list(ds.paths), params,
                                                       int(settings['calibration_population']), regions,
                                                       read_filter=read_filter, fiber_reads=False)
                if not mols:
                    raise ValueError(f'nuc: no usable molecules in the calibration sample of {ds.dataset_id}')
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
        from fiberhmm.nuc.enrichment import MeasuredTrack, grid_windows, window_test
        from fiberhmm.nuc.molecule import molecule_outputs as nuc_outputs
        from fiberhmm.nuc.records import decode_record
        from fiberhmm.nuc.segment import DISORDERED, LOWINFO, REGULAR, UNDETERMINED
        from fiberhmm.scan.reads import iter_records
        params = self._params(settings)
        z0, z1 = unit.zone_start, unit.zone_end
        n = z1 - z0
        W, step = int(settings['phase_window_bp']), int(settings['phase_step_bp'])
        EW, Estep = int(settings['enrich_window_bp']), int(settings['enrich_step_bp'])
        reach = max(W, EW)
        haps = settings['groups'] == 'dataset+haplotype'
        filters = inputs.read_policy['filters']
        tracks: Dict[str, np.ndarray] = {}

        def arr(name):
            a = tracks.get(name)
            if a is None:
                a = np.zeros(n, dtype=np.float64)
                tracks[name] = a
            return a

        def mask(ivs):
            mk = np.zeros(n, dtype=bool)
            for a, b in ivs:
                a, b = max(int(a), z0), min(int(b), z1)
                if b > a:
                    mk[a - z0:b - z0] = True
            return mk
        win_starts = [w for w in range((z0 + step - 1) // step * step, z1, step)]
        enr_starts = grid_windows(z0, z1, Estep)
        phase_mols: Dict[str, List[Tuple[object, np.ndarray, np.ndarray]]] = {}
        enr_tracks: List[MeasuredTrack] = []
        # a fixed group (and so track) set for every unit: territory runs
        # concatenate tracks by name, zone by zone
        groups_seen = ['all']
        for ds in inputs.datasets:
            groups_seen.append(ds.dataset_id)
            if haps:
                groups_seen += [f'{ds.dataset_id}:HP1', f'{ds.dataset_id}:HP2']
        state_key = {REGULAR: 'reg', DISORDERED: 'dis', UNDETERMINED: 'und', LOWINFO: 'low'}
        for ds in inputs.datasets:
            cal = self._cal(calibration, ds.dataset_id)
            for i, path in enumerate(ds.paths):
                with pysam.AlignmentFile(path, 'rb') as bam:
                    if unit.chrom not in bam.references:
                        continue
                    ctx = self._context(ds.dataset_id, i, bam.header, params)
                    for read, _sha, _occ in iter_records(bam, unit.chrom, z0, z1 + reach, filters):
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
                        # reference intervals: aligned blocks; segment and event envelopes
                        blocks = [(int(r0), int(r0 + ln)) for _q0, r0, ln in mol.blocks] \
                            if mol.blocks is not None else []
                        kind_ref: Dict[str, List[Tuple[int, int]]] = {k: [] for k in SEG_METRICS}
                        q_ref: List[Tuple[int, int, float]] = []
                        for sgm in res.segments:
                            env = mol.mol_interval_to_ref(sgm.start, sgm.end)
                            if env is None or env[1] <= env[0]:
                                continue
                            kind_ref[state_key[sgm.state]].append((int(env[0]), int(env[1])))
                            if sgm.state == REGULAR and sgm.q_hat:
                                q_ref.append((int(env[0]), int(env[1]), float(sgm.q_hat)))
                        for ev in res.events:
                            env = mol.mol_interval_to_ref(ev.start, ev.end)
                            if env is not None and env[1] > env[0]:
                                kind_ref['uds' if ev.kind == 16 else 'mis'].append((int(env[0]), int(env[1])))
                        # measured: aligned bases inside decoded segments, outside
                        # low information (read ends beyond the outer dyads and
                        # NFR-length MSPs are not segmented, so not measured)
                        seg_cov = _union_iv(kind_ref['reg'] + kind_ref['dis'] + kind_ref['und'])
                        measured = _intersect_iv(_union_iv(blocks), seg_cov)
                        measured = _subtract_iv(measured, kind_ref['low'])
                        cov = mask(blocks)
                        meas = mask(measured)
                        kmask = {k: mask(kind_ref[k]) for k in SEG_METRICS}
                        qsum = np.zeros(n)
                        for a, b, q in q_ref:
                            a, b = max(a, z0), min(b, z1)
                            if b > a:
                                qsum[a - z0:b - z0] = q
                        for g in grp:
                            arr(f'measured:{g}')[meas] += 1
                            for k in SEG_METRICS:
                                base = cov if k == 'low' else meas
                                arr(f'{k}:{g}')[kmask[k] & base] += 1
                            arr(f'qsum:{g}')[kmask['reg'] & meas] += qsum[kmask['reg'] & meas]
                        # recurrence test: the molecule's measured bases and kinds
                        t = MeasuredTrack(measured)
                        for k in ENRICH_KINDS:
                            t.set_kind(k, kind_ref[k])
                        enr_tracks.append(t)
                        # dyads for phased windows: molecular positions of observed mono
                        # dyads (distances are measured along the molecule; the
                        # reference position only decides window membership)
                        fp = res.footprints
                        if fp is not None:
                            from fiberhmm.nuc.sizes import MONO
                            sel = (fp['coarse'] == MONO) & ~fp['is_tf'] & ~fp['truncated']
                            c = ((fp['start'][sel] + fp['end'][sel]) / 2.0)
                            ref = mol.mol_to_ref(np.round(c).astype(np.int64)) if len(c) else np.zeros(0, np.int64)
                            ok = ref >= 0
                            for g in grp:
                                phase_mols.setdefault(g, []).append((mol, c[ok], ref[ok]))
        # fixed-grid phased windows written by the zone holding the window start
        qgrid = np.arange(165.0, 216.0, 1.0)
        from fiberhmm.nuc.phasing import uniform_expectation
        for g in groups_seen:
            Rt, Nt, nullt, logpt, Pt, Ct = (np.full(n, np.nan) for _ in range(6))
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
                # the window is a fixed W bp, generally not whole periods: remove
                # the uniform expectation over [0, W) per dyad
                best_p, best = None, -1.0
                for P in qgrid:
                    eu = uniform_expectation(0.0, float(W), P)
                    r = np.mean([abs(np.exp(2j * np.pi * x / P).sum() - eu * len(x)) / len(x) for x in xs])
                    if r > best:
                        best, best_p = r, P
                eu = uniform_expectation(0.0, float(W), best_p)
                V = np.array([np.exp(2j * np.pi * x / best_p).sum() - eu * len(x) for x in xs])
                N = float(sum(len(x) for x in xs))
                S = abs(V.sum())
                var = float(np.sum(np.abs(V) ** 2))
                k = w - z0
                Rt[k] = S / N
                nullt[k] = np.sqrt(np.pi / 4.0 * var) / N
                logpt[k] = -(S * S) / var if var > 0 else 0.0
                Nt[k] = len(xs)
                Pt[k] = best_p
                # coherence of the molecules' phases: 1 all in phase, ~1/sqrt(n) random
                tot = float(np.abs(V).sum())
                Ct[k] = S / tot if tot > 0 else np.nan
            tracks[f'phase_R:{g}'] = Rt
            tracks[f'phase_null:{g}'] = nullt
            tracks[f'phase_logp:{g}'] = logpt
            tracks[f'phase_n:{g}'] = Nt
            tracks[f'phase_P:{g}'] = Pt
            tracks[f'phase_C:{g}'] = Ct
        # recurrence windows (pooled over every measured molecule): exact
        # shift-null p per window start, written by the zone holding the start
        K = int(settings['enrich_levels'])
        min_mol = int(settings['enrich_min_molecules'])
        for kind in ENRICH_KINDS:
            Pv, Ov, Ev, Nv = (np.full(n, np.nan) for _ in range(4))
            for w in enr_starts:
                r = window_test(enr_tracks, kind, w, w + EW, K, min_mol)
                if r is None:
                    continue
                k = w - z0
                Pv[k], Ov[k], Ev[k], Nv[k] = r['p'], r['obs'], r['exp'], r['n']
            tracks[f'enr_p:{kind}'] = Pv
            tracks[f'enr_obs:{kind}'] = Ov
            tracks[f'enr_exp:{kind}'] = Ev
            tracks[f'enr_n:{kind}'] = Nv
        for g in groups_seen:
            arr(f'measured:{g}')
            for k in SEG_METRICS + ('qsum',):
                arr(f'{k}:{g}')
        return UnitResult(status='ok', tracks=tracks, extra={'groups': groups_seen})

    # ------------------------------------------------------------------ stitching
    def stitch_support_bp(self, settings) -> int:
        """Zero context a stitch run needs: the longest window that extends past
        its start (a phased window or a recurrence window)."""
        return max(int(settings['phase_window_bp']), int(settings['enrich_window_bp']))

    # ---- scan-wide Benjamini-Hochberg (enrich_fdr_scope = 'scan')
    def scan_statistics_tracks(self, settings):
        if settings['enrich_fdr_scope'] != 'scan':
            return None
        return [f'{p}:{k}' for k in ENRICH_KINDS for p in ('enr_p', 'enr_n')]

    def _tested(self, territory, tracks, settings, kind):
        """(run_index, offset, p) of every tested window of ``kind`` (empty zones are zero-padded: n = 0)."""
        out = []
        for run_index in range(len(territory)):
            pv, nv = (tracks.get(f'{n}:{kind}') for n in ('enr_p', 'enr_n'))
            if pv is None or nv is None or pv[run_index] is None or nv[run_index] is None:
                continue
            pa, na = np.asarray(pv[run_index], dtype=float), np.asarray(nv[run_index], dtype=float)
            valid = np.isfinite(pa) & np.isfinite(na) & (na >= int(settings['enrich_min_molecules']))
            for off in np.flatnonzero(valid):
                out.append((run_index, int(off), float(pa[off])))
        return out

    def stitch_statistics(self, chrom, territory, tracks, settings, calibration):
        from fiberhmm.nuc.enrichment import bh_candidate_bound
        bound = bh_candidate_bound(float(settings['enrich_fdr']))
        res = {}
        for kind in ENRICH_KINDS:
            tested = self._tested(territory, tracks, settings, kind)
            res[kind] = (len(tested), [p for _r, _o, p in tested if p <= bound])
        return res

    def combine_statistics(self, per_contig, settings, calibration):
        from fiberhmm.nuc.enrichment import bh_threshold
        q = float(settings['enrich_fdr'])
        kinds = {}
        for kind in ENRICH_KINDS:
            m = sum(per_contig[c][kind][0] for c in per_contig)
            small = [p for c in sorted(per_contig) for p in per_contig[c][kind][1]]
            thr = bh_threshold(small, m, q)
            kinds[kind] = {'windows_tested': int(m), 'p_threshold': thr,
                           'windows_rejected': int(sum(1 for p in small if thr is not None and p <= thr))}
        return {'fdr_scope': 'scan', 'enrich_fdr': q, 'kinds': kinds}

    def stitch(self, chrom, territory, tracks, settings, calibration, scan_statistics=None) -> StitchResult:
        from fiberhmm.nuc.enrichment import bh, merge_windows
        out = StitchResult()
        groups = sorted({name.split(':', 1)[1] for name in tracks if name.startswith('measured:')})
        if 'all' not in groups:
            return out
        W = int(settings['phase_window_bp'])
        EW = int(settings['enrich_window_bp'])
        q = float(settings['enrich_fdr'])

        def getter(run_index):
            def get(name):
                arrs = tracks.get(name)
                if arrs is None or arrs[run_index] is None:
                    return None
                return np.asarray(arrs[run_index], dtype=float)
            return get

        def kind_of(g):
            return 'all' if g == 'all' else ('haplotype' if ':HP' in g else 'dataset')
        regular_patches: List[Tuple[int, int]] = []
        # --- recurrence patches: BH over the scan's (or the contig's) windows, per kind ---
        scan_fdr = settings['enrich_fdr_scope'] == 'scan'
        if scan_fdr and scan_statistics is None:
            raise ValueError("nuc: enrich_fdr_scope 'scan' needs the scan-wide statistics (run through "
                             "fiberhmm.scan.run_units, or set enrich_fdr_scope 'contig')")
        for kind, ptype in PATCH_KINDS:
            rows = []                          # (run_index, offset, p, obs, exp, n)
            for run_index, (t0, t1, _lo, _ro) in enumerate(territory):
                get = getter(run_index)
                pv = get(f'enr_p:{kind}')
                if pv is None:
                    continue
                ov, ev, nv = get(f'enr_obs:{kind}'), get(f'enr_exp:{kind}'), get(f'enr_n:{kind}')
                if ov is None or ev is None or nv is None:
                    continue
                # tested windows only (empty zones are zero-padded: n = 0 there)
                valid = np.isfinite(pv) & np.isfinite(nv) & (nv >= int(settings['enrich_min_molecules']))
                for off in np.flatnonzero(valid):
                    rows.append((run_index, int(off), float(pv[off]), float(ov[off]), float(ev[off]), int(nv[off])))
            if not rows:
                continue
            if scan_fdr:
                rec = scan_statistics['kinds'][kind]
                thr = rec['p_threshold']
                rej = [thr is not None and r[2] <= thr for r in rows]
                n_tests = int(rec['windows_tested'])
            else:
                rej = bh([r[2] for r in rows], q)
                n_tests = len(rows)
            for run_index, (t0, t1, left_open, right_open) in enumerate(territory):
                get = getter(run_index)
                sig = [r for r, ok in zip(rows, rej) if ok and r[0] == run_index]
                if not sig:
                    continue
                den_all = get('measured:all')
                L = len(den_all)
                for a, b in merge_windows([r[1] for r in sig], EW):
                    b = min(b, L)
                    inside = [r for r in sig if a <= r[1] < b]
                    s, e = t0 + a, t0 + b
                    open_end = (a == 0 and left_open) or (b == L and right_open)
                    vals = []
                    for g in groups:
                        dg, ng = get(f'measured:{g}'), get(f'{kind}:{g}')
                        if dg is None or ng is None:
                            continue
                        dg, ng = dg[a:b], ng[a:b]
                        fg = float(ng.sum() / dg.sum()) if dg.sum() > 0 else None
                        vals.append((g, kind_of(g), fg, int(np.median(dg)) if len(dg) else 0))
                    qh = None
                    if kind == 'reg':
                        qs, rn = get('qsum:all')[a:b].sum(), get('reg:all')[a:b].sum()
                        qh = float(qs / rn) if rn > 0 else None
                        regular_patches.append((s, e))
                    p_min = min(r[2] for r in inside)
                    score = vals[[v[0] for v in vals].index('all')][2]
                    key = f'{ptype}:{s}'
                    out.hits.append(Hit(f'nuc.patch.{ptype}', key, chrom, s, e, (s + e) // 2, score=score,
                                        flags=('open_end',) if open_end else (),
                                        columns={'q_hat': '' if qh is None else round(qh, 1), 'P': '',
                                                 'width': e - s, 'open_end': int(open_end)}))
                    for g, kd, fg, nm in vals:
                        out.values.append(Value(key, g, kd, 'fraction', fg, n_molecules=nm))
                    out.values.append(Value(key, 'all', 'all', 'p_min', p_min))
                    out.values.append(Value(key, 'all', 'all', 'expected_fraction',
                                            float(np.mean([r[4] for r in inside]))))
                    out.values.append(Value(key, 'all', 'all', 'observed_fraction',
                                            float(np.mean([r[3] for r in inside]))))
                    out.values.append(Value(key, 'all', 'all', 'windows_tested', float(n_tests)))
                    if qh is not None:
                        out.values.append(Value(key, 'all', 'all', 'q_hat', qh))
        # --- phased and fuzzy windows on the fixed grid --------------------------------
        for run_index, (t0, t1, left_open, right_open) in enumerate(territory):
            get = getter(run_index)
            R, null, logp, P = get('phase_R:all'), get('phase_null:all'), get('phase_logp:all'), get('phase_P:all')
            C = get('phase_C:all')
            if R is None:
                continue
            excess = R - null
            Nw = get('phase_n:all')
            nw = np.nan_to_num(Nw) if Nw is not None else np.zeros(len(R))
            # computed windows only (empty zones are zero-padded: n = 0 there)
            fin = np.isfinite(R) & (nw >= int(settings['patch_min_measured']))
            phased = fin & (excess >= float(settings['phase_min_excess'])) & \
                (logp <= np.log(float(settings['phase_max_p'])))
            # fuzzy: a window starting inside a recurrent regular patch whose
            # molecules do not share a phase (coherence at the null)
            in_reg = np.zeros(len(R), dtype=bool)
            for s_, e_ in regular_patches:
                a_, b_ = max(s_ - t0, 0), min(e_ - t0, len(R))
                if b_ > a_:
                    in_reg[a_:b_] = True
            fuzzy = fin & in_reg & (logp > np.log(float(settings['fuzzy_min_p']))) & \
                (excess < float(settings['phase_min_excess']))
            for ptype, starts_mask in (('phased', phased), ('fuzzy', fuzzy)):
                starts = np.flatnonzero(starts_mask)
                if not len(starts):
                    continue
                good = np.zeros(len(R), dtype=bool)
                for k_ in starts:
                    good[k_:min(k_ + W, len(good))] = True
                for a, b in _runs(good):
                    s, e = t0 + a, t0 + b
                    inwin = [k_ for k_ in starts if a <= k_ < b]
                    key = f'{ptype}:{s}'
                    open_end = (a == 0 and left_open) or (b == len(good) and right_open)
                    Pm = float(np.median(P[inwin]))
                    if ptype == 'phased':
                        score = float(np.max(excess[inwin]))
                    else:
                        cc = C[inwin] if C is not None else np.array([np.nan])
                        cm = float(np.nanmedian(cc)) if np.isfinite(cc).any() else float('nan')
                        score = float(Pm / (2 * np.pi) * np.sqrt(-2 * np.log(cm))) if 0 < cm < 1 else float('nan')
                    out.hits.append(Hit(f'nuc.patch.{ptype}', key, chrom, s, e, (s + e) // 2, score=score,
                                        flags=('open_end',) if open_end else (),
                                        columns={'q_hat': '', 'P': round(Pm, 1), 'width': e - s,
                                                 'open_end': int(open_end)}))
                    for g in groups:
                        Rg, ng, Cg = get(f'phase_R:{g}'), get(f'phase_null:{g}'), get(f'phase_C:{g}')
                        if Rg is None:
                            continue
                        vals = [Rg[k_] - ng[k_] for k_ in inwin if np.isfinite(Rg[k_])]
                        out.values.append(Value(key, g, kind_of(g), 'R_minus_null',
                                                float(np.max(vals)) if vals else None))
                        if Cg is not None:
                            cs = [Cg[k_] for k_ in inwin if np.isfinite(Cg[k_])]
                            out.values.append(Value(key, g, kind_of(g), 'coherence',
                                                    float(np.median(cs)) if cs else None))
                    out.values.append(Value(key, 'all', 'all', 'P', Pm))
                    if ptype == 'fuzzy':
                        out.values.append(Value(key, 'all', 'all', 'dispersion_bp', score))
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


CAL_WINDOW_BP = 50_000


class _RecordProbe:
    """``probe(chrom, start, end)``: does any BAM hold a primary, annotated
    record there that passes the read filter?"""

    def __init__(self, paths, read_filter=None):
        self.paths, self.read_filter, self.bams = list(paths), read_filter, []

    def __enter__(self):
        import pysam
        self.bams = [pysam.AlignmentFile(p, 'rb') for p in self.paths]
        return self

    def __exit__(self, *exc):
        for b in self.bams:
            b.close()
        self.bams = []

    def __call__(self, chrom, start, end) -> bool:
        from fiberhmm.scan.reads import has_annotations
        for bam in self.bams:
            if chrom not in bam.references:
                continue
            for r in bam.fetch(chrom, start, end):
                if r.is_unmapped or r.is_secondary or r.is_supplementary:
                    continue
                if (self.read_filter is None or self.read_filter(r)) and has_annotations(r):
                    return True
        return False


def calibration_regions(units, n_windows: int, has_records=None) -> List[Tuple[str, int, int]]:
    """Calibration sample independent of the tiling: the scanned territory
    (union of unit spans) cut on fixed 50 kb windows of absolute coordinates,
    taken in order of a stable hash of (contig, window index); the first
    ``n_windows`` (all windows when 0). With ``has_records`` only windows for
    which it returns True count, so a genome scan of targeted data samples the
    targeted windows."""
    spans: Dict[str, List[Tuple[int, int]]] = {}
    for u in units:
        spans.setdefault(u.chrom, []).append((int(u.start), int(u.end)))
    windows = []
    for chrom, iv in spans.items():
        iv.sort()
        merged = []
        for a, b in iv:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        for a, b in merged:
            for k in range(a // CAL_WINDOW_BP, (b - 1) // CAL_WINDOW_BP + 1):
                s0, e0 = max(a, k * CAL_WINDOW_BP), min(b, (k + 1) * CAL_WINDOW_BP)
                if e0 > s0:
                    windows.append((hashlib.sha256(f'{chrom}:{k}'.encode()).hexdigest(), chrom, s0, e0))
    windows.sort()
    if has_records is not None:
        keep = []
        for w in windows:
            if has_records(w[1], w[2], w[3]):
                keep.append(w)
                if n_windows and len(keep) >= n_windows:
                    break
        windows = keep
    elif n_windows:
        windows = windows[:n_windows]
    return sorted((c, s0, e0) for _h, c, s0, e0 in windows)


def _contigs_with_records(paths, reference) -> List[str]:
    """Contigs of ``reference`` (in its order) that hold a record in any of ``paths``: the index's mapped
    count when it is positive, else one indexed fetch (an index may carry no count metadata, and then
    reports zero for contigs that hold records)."""
    import pysam
    found = set()
    bams = [pysam.AlignmentFile(p, 'rb') for p in paths]
    try:
        for bam in bams:
            try:
                for st in bam.get_index_statistics():
                    if int(st.mapped) > 0:
                        found.add(st.contig)
            except (ValueError, OSError, AttributeError):
                pass
        for chrom in reference:
            if chrom in found:
                continue
            for bam in bams:
                if chrom in bam.references and next(iter(bam.fetch(chrom)), None) is not None:
                    found.add(chrom)
                    break
    finally:
        for bam in bams:
            bam.close()
    return [c for c in reference if c in found]


def data_calibration_regions(reference, paths, n_windows: int, has_records=None) -> List[Tuple[str, int, int]]:
    """Calibration sample that depends on the data only: fixed 50 kb windows of absolute coordinates over every
    contig of ``reference`` that holds records of ``paths`` (index statistics), taken in the same stable hash
    order of (contig, window index) as :func:`calibration_regions`; the first ``n_windows`` (all when 0) for
    which ``has_records`` returns True. A whole-genome scan with ``calibration_scope = 'territory'`` draws the
    same windows."""
    windows = []
    for chrom in _contigs_with_records(paths, reference):
        length = int(reference[chrom])
        for k in range(0, (length - 1) // CAL_WINDOW_BP + 1):
            s0, e0 = k * CAL_WINDOW_BP, min(length, (k + 1) * CAL_WINDOW_BP)
            if e0 > s0:
                windows.append((hashlib.sha256(f'{chrom}:{k}'.encode()).hexdigest(), chrom, s0, e0))
    windows.sort()
    keep = []
    for w in windows:
        if has_records is None or has_records(w[1], w[2], w[3]):
            keep.append(w)
            if n_windows and len(keep) >= n_windows:
                break
    return sorted((c, s0, e0) for _h, c, s0, e0 in keep)


def _union_iv(iv) -> List[Tuple[int, int]]:
    out: List[List[int]] = []
    for a, b in sorted((int(a), int(b)) for a, b in iv if b > a):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _intersect_iv(a_iv, b_iv) -> List[Tuple[int, int]]:
    """Intersection of two sorted disjoint interval lists."""
    out, i, j = [], 0, 0
    while i < len(a_iv) and j < len(b_iv):
        a0, a1 = a_iv[i]
        b0, b1 = b_iv[j]
        lo, hi = max(a0, b0), min(a1, b1)
        if hi > lo:
            out.append((lo, hi))
        if a1 < b1:
            i += 1
        else:
            j += 1
    return out


def _subtract_iv(iv, cut) -> List[Tuple[int, int]]:
    """Intervals ``iv`` minus the union of ``cut`` (both half-open)."""
    cut = sorted(cut)
    out = []
    for a, b in sorted(iv):
        cur = a
        for c0, c1 in cut:
            if c1 <= cur or c0 >= b:
                continue
            if c0 > cur:
                out.append((cur, c0))
            cur = max(cur, c1)
        if cur < b:
            out.append((cur, b))
    return out


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
