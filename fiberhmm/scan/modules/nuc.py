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
  envelopes), the ``nucmol`` scalars and, unless ``nucreg`` is null, the
  ``nucreg`` regularity tiles (:mod:`fiberhmm.nuc.regularity`; the header
  contract records their parameters) and, with ``nucpatch`` set, the
  ``nucpatch`` regularity patches (:mod:`fiberhmm.nuc.patches`) read against
  the dataset's random-packing reference median, computed in ``calibrate``
  (``nucreg_reference``) unless ``nucpatch.ref_median`` gives it.
- ``run_unit`` (BED units, with ``nucpatch`` and ``nucgeo`` set): shared patch
  geometries of the unit's molecules (:mod:`fiberhmm.nuc.geometries`) as
  ``nuc.geometry`` hits anchored in the unit, a share per dataset and one
  state per (geometry, record): 1 carries, 0 absent, . no call or not spanned.
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
  molecule). Molecules with too few marks for spacing evidence, too
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
from ..types import (FeatureTypeDecl, Hit, MAGroupDecl, MoleculeResult, ScalarCodec, ScoreDecl,
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
    version = '2.4'
    #: r2: per-width reference medians placed at their classes' sampled widths, truncated-reference and geometry
    #: support/cohort fixes (resumed scans recompute calibrations and units made before them)
    implementation = 'r2'
    experimental = True
    reconcile = 'stitch'
    max_support_bp = 12_500
    wants_molecules = True
    _lazy_stitch_tracks = True
    # fiberhmm.nuc.records decodes each annotation's frame from the record's header (DecodeContext), so it gets
    # the stored record: an orchestrator carry would be mirrored a second time on reverse reads
    molecule_frame = 'source'
    state_groups = {'nuc.geometry': 'nucgeo'}

    def __init__(self):
        self._cals: Dict[Tuple[int, str], Any] = {}
        self._ctx: Dict[Tuple[str, int], Any] = {}
        self._record_cache = {}
        self._cache_scope = None

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
            'nucreg': _p({}, 'dict', 'sliding-window regularity (MA nucreg): fiberhmm.nuc.regularity.'
                         'RegularityParams overrides; null = off'),
            'nucpatch': _p(None, 'dict', 'regularity patches (MA nucpatch): fiberhmm.nuc.patches.PatchParams '
                           'overrides plus ref_median (log2; default: the random-packing reference of the '
                           'calibration population, reference_windows windows, reference_seed; with nucreg ends '
                           'truncated also per window-width class, reference_cut_windows per class, classes with '
                           'at least reference_min_cut scores; a ref_median override shifts those deltas); '
                           'null = off (default); needs nucreg'),
            'nucgeo': _p({}, 'dict', 'shared patch geometries per BED unit (nuc.geometry hits and states): '
                         'fiberhmm.nuc.geometries.GeometryParams overrides plus cohort (overlapping: records '
                         'overlapping the unit, default, as the browser panel; spanning: records spanning it); '
                         'one record per physical molecule (mol:v1) per dataset, the first primary record in '
                         'file and coordinate order; null = off; needs nucpatch'),
        }

    def resolve(self, params: Mapping[str, Any]) -> dict:
        from fiberhmm.nuc.params import NucParams
        out = super().resolve(params)
        out['params'] = NucParams.from_dict(out.get('params') or {}).to_dict()
        out['params'].pop('extra', None)
        if out.get('nucreg') is not None:
            from fiberhmm.nuc.regularity import RegularityParams
            out['nucreg'] = RegularityParams.from_dict(out['nucreg']).to_dict()
        if out.get('nucpatch') is not None:
            if out.get('nucreg') is None:
                raise ValueError('nuc: nucpatch needs nucreg (the regularity track)')
            out['nucpatch'] = _resolve_patch_settings(out['nucpatch'])
        if out.get('nucpatch') is None:
            # Patches off: neither key enters the settings, so the settings hash, calibration and unit keys stay
            # those of a scan without patch support (geometries need patches).
            out.pop('nucpatch', None)
            out.pop('nucgeo', None)
        elif out.get('nucgeo') is not None:
            out['nucgeo'] = _resolve_geo_settings(out['nucgeo'])
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
        types.append(FeatureTypeDecl('nuc.geometry', 'nuc', 'Shared regularity geometry', 'Shared regularity geometries',
                                     ('hit', 'state'), 'centre',
                                     scores=(ScoreDecl('share', 'Share of molecules carrying it', 'fraction',
                                                       'high', True),
                                             ScoreDecl('share_dm', 'Share of callable molecules carrying it, m6A '
                                                       'density matched across datasets (one dataset or all: '
                                                       'the plain callable share)', 'fraction', 'high', True)),
                                     state_rule='1: the molecule has a member patch; 0: its tiles cover the span '
                                                'with calls and no member patch; .: no call (over half the '
                                                'tiles there unscored) or not spanned',
                                     columns=('geometry', 'open_end', 'P', 'width', 'molecules', 'patches',
                                              'start_mad', 'end_mad', 'edges', 'fold', 'parent', 'kind')))
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
        out.append(MAGroupDecl('nucgeo', 1, 'state', ('state', 'reason'),
                               state_codes=(('1', 'carries'), ('2', 'absent'), ('3', 'no call or not spanned')),
                               an='hit_id'))
        return out

    def state_columns(self) -> List[str]:
        return ['reason', 'patch_start', 'patch_end']

    def load_policy(self, settings: Mapping[str, Any]) -> dict:
        return {'reader': 'fiberhmm.nuc full-alignment records', 'molecules': 'all',
                'tracks': 'primary and soft-clipped records overlapping the zone (supplementaries skipped)'}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _params(settings):
        from fiberhmm.nuc.params import NucParams
        return NucParams.from_dict(settings.get('params') or {})

    @staticmethod
    def _regularity(settings):
        from fiberhmm.nuc.regularity import RegularityParams
        value = settings.get('nucreg', {})
        return None if value is None else RegularityParams.from_dict(value)

    @staticmethod
    def _patches(settings):
        from fiberhmm.nuc.patches import PatchParams
        value = settings.get('nucpatch')
        if value is None:
            return None
        return PatchParams.from_dict({k: v for k, v in value.items() if k not in PATCH_EXTRA})

    def ma_group_provenance(self, settings) -> Dict[str, dict]:
        """Extra header-contract fields per group: the nucreg and nucpatch parameters."""
        from fiberhmm.nuc.tags import NUCPATCH, NUCREG, nucpatch_declaration, nucreg_declaration
        reg = self._regularity(settings)
        d = nucreg_declaration(reg)
        extra = {k: d[k] for k in ('score_byte', 'period_offset', 'params', 'params_digest') if k in d}
        pp = self._patches(settings)
        out = {NUCREG: extra if reg is not None else {'params': None}}
        if pp is None:
            out[NUCPATCH] = {'params': None}
        else:
            ps = settings['nucpatch']
            ref = ps['ref_median'] if ps.get('ref_median') is not None else \
                {'source': 'calibration nucreg_reference', 'windows': ps['reference_windows'],
                 'seed': ps['reference_seed']}
            if reg is not None and reg.ends == 'truncated':
                ref = {'source': 'calibration nucreg_reference', 'override': ps.get('ref_median'),
                       'windows': ps['reference_windows'], 'seed': ps['reference_seed'],
                       'per_width': {'cut_windows': ps.get('reference_cut_windows', CUT_WINDOWS_DEFAULT),
                                     'min_cut': ps.get('reference_min_cut', MIN_CUT_DEFAULT)}}
            dp = nucpatch_declaration(pp, ref)
            out[NUCPATCH] = {k: dp[k] for k in ('score_byte', 'period_offset', 'flag_bits', 'params', 'reference')}
        return out

    def _ref_median(self, settings, calibration, dataset_id):
        return self._reference(settings, calibration, dataset_id)[0]

    def _reference(self, settings, calibration, dataset_id):
        """(reference median, per-width deltas or None) the dataset's patches are read against. An override
        (``nucpatch.ref_median``) replaces the median; the calibration's per-width deltas (ends 'truncated') still
        apply on top of it."""
        from fiberhmm.nuc.regularity import reference_widths
        ps = settings.get('nucpatch')
        if ps is None:
            return None, None
        ref = (calibration or {}).get('nucreg_reference', {}).get(dataset_id)
        widths = reference_widths(ref)
        if ps.get('ref_median') is not None:
            return float(ps['ref_median']), widths
        if ref is None or ref.get('median') is None:
            raise ValueError(f'nuc: nucpatch needs a reference median for {dataset_id!r} (none in the calibration '
                             'record, or its calibration population had no window to score; set nucpatch.ref_median)')
        return float(ref['median']), widths

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
            ps = settings.get('nucpatch')
            reg = self._regularity(settings)
            if ps is not None and (ps.get('ref_median') is None or (reg is not None and reg.ends == 'truncated')):
                # the random-packing reference the patches are read against (only with nucpatch on, so the
                # calibration record of scans without patches is unchanged; with an override only for ends
                # 'truncated', whose per-width deltas still apply)
                progress.report('nuc', f'regularity reference for {ds.dataset_id}')
                ref = nucreg_reference(list(ds.paths), params, reg, regions,
                                       n_population=int(settings['calibration_population']),
                                       n_windows=int(ps['reference_windows']), seed=int(ps['reference_seed']),
                                       read_filter=read_filter,
                                       cut_windows=int(ps.get('reference_cut_windows', CUT_WINDOWS_DEFAULT)),
                                       min_cut_scores=int(ps.get('reference_min_cut', MIN_CUT_DEFAULT)))
                if ref is None:
                    raise ValueError(f'nuc: no usable molecules for the regularity reference of {ds.dataset_id}')
                out.setdefault('nucreg_reference', {})[ds.dataset_id] = ref.to_dict()
        return out

    # ------------------------------------------------------------------ molecule-wide outputs
    def _record_outputs(self, read, record, settings, calibration):
        from fiberhmm.nuc.scan import record_outputs
        scope = self._cache_scope
        if scope is None or scope[0] is not settings or scope[1] is not calibration:
            self._record_cache.clear()
            self._cache_scope = (settings, calibration)
        key = (record.dataset_id, record.input_index, record.record)
        if key not in self._record_cache:
            params = self._params(settings)
            ctx = self._context(record.dataset_id, record.input_index, read.header, params)
            pp = self._patches(settings)
            ref_median, ref_widths = self._reference(settings, calibration, record.dataset_id) \
                if pp is not None else (None, None)
            value = record_outputs(read, record, self._cal(calibration, record.dataset_id),
                                   params=params, context=ctx, regularity=self._regularity(settings),
                                   patches=pp, ref_median=ref_median, ref_widths=ref_widths)
            if len(self._record_cache) >= 512:
                self._record_cache.pop(next(iter(self._record_cache)))
            self._record_cache[key] = value
        return self._record_cache[key]

    def molecule_outputs(self, molecule, settings, calibration) -> Optional[MoleculeResult]:
        return self._record_outputs(molecule.read, molecule.record, settings, calibration)[2]

    # ------------------------------------------------------------------ unit tracks
    def run_unit(self, unit, inputs, settings, calibration, upstream, work: Path, progress) -> UnitResult:
        import pysam
        from fiberhmm.nuc.enrichment import MeasuredTrack, grid_windows, window_test
        from fiberhmm.nuc.segment import REGULAR
        from fiberhmm.scan.identity import molecule_id
        from fiberhmm.scan.reads import iter_records, record_ref
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
        geo_on = unit.mode == 'bed' and settings.get('nucpatch') is not None and settings.get('nucgeo') is not None
        geo_mols = []                      # (record, molecule layout): one per physical molecule and dataset
        geo_rep: Dict[Tuple[str, str], Optional[int]] = {}   # (dataset, mol:v1) -> geo_mols index (None: none)
        geo_alias = []                     # (record, geo_mols index): the molecule's other records
        for ds in inputs.datasets:
            cal = self._cal(calibration, ds.dataset_id)
            for i, path in enumerate(ds.paths):
                with pysam.AlignmentFile(path, 'rb') as bam:
                    if unit.chrom not in bam.references:
                        continue
                    for read, _sha, _occ in iter_records(bam, unit.chrom, z0, z1 + reach, filters):
                        if read.is_supplementary:
                            continue
                        record = record_ref(read, ds, i, _sha, _occ, unit.chrom)
                        mol, res, _mr = self._record_outputs(read, record, settings, calibration)
                        if mol is None:
                            continue
                        if geo_on and res.patches is not None and mol.ref_end > unit.start \
                                and mol.ref_start < unit.end:
                            # one record per physical molecule (mol:v1) and dataset, the first primary record in
                            # file and coordinate order, as the browser's cohort; its other records take its state
                            mk = (ds.dataset_id, record.molecule or molecule_id([read.query_name]))
                            if mk in geo_rep:
                                if geo_rep[mk] is not None:
                                    geo_alias.append((record, geo_rep[mk]))
                            else:
                                lay = _geometry_layout(mol, res, ds.dataset_id, unit, settings['nucgeo'],
                                                       bool(settings['nucgeo'].get('snapped', True)))
                                geo_rep[mk] = len(geo_mols) if lay is not None else None
                                if lay is not None:
                                    geo_mols.append((record, lay))
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
                        from fiberhmm.nuc.scan import measured_track
                        measured, kind_ref = measured_track(mol, res)
                        q_ref = []
                        for sgm in res.segments:
                            env = mol.mol_interval_to_ref(sgm.start, sgm.end)
                            if env is not None and sgm.state == REGULAR and sgm.q_hat:
                                q_ref.append((int(env[0]), int(env[1]), float(sgm.q_hat)))
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
                        # dyads for phased windows: molecular positions of observed eligible
                        # dyads (distances are measured along the molecule; the
                        # reference position only decides window membership)
                        fp = res.footprints
                        if fp is not None:
                            from fiberhmm.nuc.spacing import footprint_dyad_mask
                            sel = footprint_dyad_mask(fp, cal.nrl_pooled, params)
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
        out = UnitResult(status='ok', tracks=tracks, extra={'groups': groups_seen})
        if geo_on:
            _unit_geometries(out, unit, geo_mols, settings['nucgeo'], [d.dataset_id for d in inputs.datasets],
                             geo_alias)
        return out

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
            def get(name, selection=None):
                arrs = tracks.get(name)
                if arrs is None or arrs[run_index] is None:
                    return None
                arr = arrs[run_index]
                return np.asarray(arr if selection is None else arr[selection], dtype=float)
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
                L = t1-t0
                for a, b in merge_windows([r[1] for r in sig], EW):
                    b = min(b, L)
                    inside = [r for r in sig if a <= r[1] < b]
                    s, e = t0 + a, t0 + b
                    open_end = (a == 0 and left_open) or (b == L and right_open)
                    vals = []
                    for g in groups:
                        dg, ng = get(f'measured:{g}', slice(a, b)), get(f'{kind}:{g}', slice(a, b))
                        if dg is None or ng is None:
                            continue
                        fg = float(ng.sum() / dg.sum()) if dg.sum() > 0 else None
                        vals.append((g, kind_of(g), fg, int(np.median(dg)) if len(dg) else 0))
                    qh = None
                    if kind == 'reg':
                        qs, rn = get('qsum:all', slice(a, b)).sum(), get('reg:all', slice(a, b)).sum()
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
                        Rg, ng, Cg = (get(f'{prefix}:{g}', slice(a, b)) for prefix in
                                      ('phase_R', 'phase_null', 'phase_C'))
                        if Rg is None:
                            continue
                        vals = [Rg[k_-a] - ng[k_-a] for k_ in inwin if np.isfinite(Rg[k_-a])]
                        out.values.append(Value(key, g, kind_of(g), 'R_minus_null',
                                                float(np.max(vals)) if vals else None))
                        if Cg is not None:
                            cs = [Cg[k_-a] for k_ in inwin if np.isfinite(Cg[k_-a])]
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
                from fiberhmm.inference.chimeras import is_marked_chimera
                if is_marked_chimera(r):
                    continue
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


def _ref_envelopes(mol, starts, ends) -> List[Tuple[int, int]]:
    """``[_ref_envelope(mol, s, e) for s, e in zip(starts, ends)]`` in one pass.

    With alignment blocks ordered and disjoint on both the molecule and the
    reference (every CIGAR-derived block list; checked here), the blocks a
    molecular interval overlaps are one contiguous run, the envelope starts in
    its first block and ends in its last one. Anything else (and an interval
    with no aligned base) goes through ``_ref_envelope``."""
    starts = [int(x) for x in starts]
    ends = [int(x) for x in ends]
    blocks = mol.blocks
    if not starts:
        return []
    if blocks is None or len(blocks) == 0:
        return [_ref_envelope(mol, a, b) for a, b in zip(starts, ends)]
    q0, r0, n = blocks[:, 0], blocks[:, 1], blocks[:, 2]
    qe, re_ = q0 + n, r0 + n
    if not (bool((n > 0).all()) and bool((q0[1:] >= qe[:-1]).all()) and bool((r0[1:] >= re_[:-1]).all())):
        return [_ref_envelope(mol, a, b) for a, b in zip(starts, ends)]
    s = np.asarray(starts, dtype=np.int64)
    e = np.asarray(ends, dtype=np.int64)
    if mol.is_reverse:
        ss, se = mol.length - e, mol.length - s
    else:
        ss, se = s, e
    first = np.searchsorted(qe, ss, side='right')       # first block ending after the interval's start
    stop = np.searchsorted(q0, se, side='left')         # blocks before this one start before its end
    ok = (stop > first) & (se > ss)
    f = np.minimum(first, len(q0) - 1)
    l_ = np.clip(stop - 1, 0, len(q0) - 1)
    rs = r0[f] + (np.maximum(q0[f], ss) - q0[f])
    rend = r0[l_] + (np.minimum(qe[l_], se) - q0[l_])
    out = []
    for i, good in enumerate(ok.tolist()):
        out.append((int(rs[i]), int(rend[i])) if good else _ref_envelope(mol, starts[i], ends[i]))
    return out


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


# ---------------------------------------------------------------------- regularity patches and geometries
#: nucpatch settings that are not PatchParams fields
PATCH_EXTRA = ('ref_median', 'reference_windows', 'reference_seed', 'reference_cut_windows', 'reference_min_cut')
#: ends 'truncated': reference windows simulated per width class and the fewest scores a class needs
#: (fiberhmm.nuc.regularity.CUT_WINDOWS_PER_CLASS / MIN_CUT_SCORES)
CUT_WINDOWS_DEFAULT, MIN_CUT_DEFAULT = 500, 100
GEO_COHORTS = ('overlapping', 'spanning')
#: the provisional default geometry cohort (the browser panel's default too)
DEFAULT_GEO_COHORT = 'overlapping'
_REASON = {'carries': 1, 'absent': 2, 'no_call': 3, 'not_spanned': 4, 'partial': 5}


def nucreg_reference(paths, params, regularity, regions, *, n_population: int, n_windows: int, seed: int,
                     read_filter=None, cut_windows: int = CUT_WINDOWS_DEFAULT, min_cut_scores: int = MIN_CUT_DEFAULT,
                     with_molecules: bool = False):
    """The random-packing regularity reference (``fiberhmm.nuc.regularity.reference_scores``) of one dataset's
    calibration population: the hash-ranked primary records of ``paths`` in the calibration ``regions`` passing
    ``read_filter`` (``fiberhmm.nuc.thinning.load_population``). What ``calibrate`` records as ``nucreg_reference``;
    FiberBrowser's live panel computes its reference with the same call, so the two agree. None without molecules.
    ``with_molecules``: (reference, the population's molecules) instead."""
    from fiberhmm.nuc.regularity import reference_scores
    from fiberhmm.nuc.thinning import load_population
    mols, _frs, _s, _chem = load_population(list(paths), params, int(n_population), regions,
                                            read_filter=read_filter, fiber_reads=False)
    if not mols:
        return (None, []) if with_molecules else None
    ref = reference_scores(mols, regularity, n_windows=int(n_windows), seed=int(seed),
                           cut_windows=int(cut_windows), min_cut_scores=int(min_cut_scores))
    return (ref, mols) if with_molecules else ref


def _resolve_patch_settings(value) -> dict:
    from fiberhmm.nuc.patches import PatchParams
    value = dict(value or {})
    extra = {'ref_median': value.pop('ref_median', None),
             'reference_windows': int(value.pop('reference_windows', 3000)),
             'reference_seed': int(value.pop('reference_seed', 1))}
    if extra['ref_median'] is not None:
        extra['ref_median'] = float(extra['ref_median'])
    if extra['reference_windows'] < 100:
        raise ValueError('nuc: nucpatch reference_windows must be at least 100')
    # the per-width reference settings enter the settings only when they differ from the defaults, so the settings
    # hash (and the calibration and unit keys) of scans made before they existed stay the same
    for key, default in (('reference_cut_windows', CUT_WINDOWS_DEFAULT), ('reference_min_cut', MIN_CUT_DEFAULT)):
        v = int(value.pop(key, default))
        if v < 1:
            raise ValueError(f'nuc: nucpatch {key} must be positive')
        if v != default:
            extra[key] = v
    return dict(PatchParams.from_dict(value).to_dict(), **extra)


def _resolve_geo_settings(value) -> dict:
    from fiberhmm.nuc.geometries import GeometryParams
    value = dict(value or {})
    cohort = value.pop('cohort', DEFAULT_GEO_COHORT)
    if cohort not in GEO_COHORTS:
        raise ValueError(f'nuc: nucgeo cohort must be one of {GEO_COHORTS}')
    return dict(GeometryParams.from_dict(value).to_dict(), cohort=cohort)


def _geometry_layout(mol, res, dataset_id, unit, geo, snapped):
    """The molecule's layout clipped to the unit, or None when it is not in the unit's cohort."""
    from fiberhmm.nuc.geometries import molecule_layout
    if geo['cohort'] == 'spanning':
        if not (mol.ref_start <= unit.start and mol.ref_end >= unit.end):
            return None
    elif mol.ref_end <= unit.start or mol.ref_start >= unit.end:
        return None
    lay = molecule_layout(mol, res.regularity, res.patches, dataset_id, snapped=snapped,
                          view=(unit.start, unit.end))
    return lay if len(lay.tile_lo) else None


def _unit_geometries(out, unit, geo_mols, geo, dataset_ids, aliases=()):
    """Geometries of one BED unit's cohort as hits, values and per-record states (in place on ``out``).
    ``geo_mols`` holds one record per physical molecule; ``aliases`` ((record, geo_mols index)) are the molecules'
    other records, which take their molecule's state (support and shares count molecules)."""
    from fiberhmm.nuc.geometries import (SCAN_STATE, GeometryParams, assign_states, density_matched_shares,
                                         find_all_geometries, support)
    from fiberhmm.scan.types import State
    if not geo_mols:
        return
    gp = GeometryParams.from_dict({k: v for k, v in geo.items() if k != 'cohort'})
    lays = [lay for _r, lay in geo_mols]
    # high-regularity geometries; low-regularity ones (found on their own) when the patches hold any
    kinds = ('high', 'low') if any(p.kind == 'low' for lay in lays for p in lay.patches) else ('high',)
    geoms = find_all_geometries(lays, gp, chrom=unit.chrom, kinds=kinds)
    states = assign_states(lays, geoms)
    seen_keys = set()
    for g in geoms:
        key = f'geometry:{g.start}-{g.end}' if g.kind == 'high' else f'geometry-low:{g.start}-{g.end}'
        if key in seen_keys:
            # two geometries of one kind with the same consensus span (different members): one key each
            key = f'{key}:{g.geometry_id.split("_", 1)[0]}'
        seen_keys.add(key)
        sup_all = support(lays, states, g.geometry_id)
        open_end = g.open_left or g.open_right
        cols = {'geometry': g.geometry_id, 'open_end': int(open_end),
                'P': g.period if g.period > 0 else '', 'width': g.width,
                'molecules': g.n_molecules, 'patches': g.n_patches,
                'start_mad': round(g.start_mad, 1), 'end_mad': round(g.end_mad, 1),
                'edges': 'snapped' if gp.snapped else 'grid',
                'fold': gp.fold,
                'parent': g.parent_id.split('_', 1)[0] if g.parent_id else ''}
        cols['kind'] = g.kind
        out.hits.append(Hit('nuc.geometry', key, unit.chrom, g.start, g.end, (g.start + g.end) // 2,
                            score=sup_all['share'], label=g.geometry_id.split('_', 1)[0],
                            flags=('open_end',) if open_end else (), columns=cols))
        # share_dm: carriers among callable molecules, per dataset standardised to the pooled m6A/kb distribution
        # (scores rise with marking density, so datasets compare density matched; FiberBrowser's panel shows the
        # same). One dataset, and the 'all' group: the plain callable share (same denominator, nothing to match).
        present = [d for d in dataset_ids if any(lay.dataset == d for lay in lays)]
        dms = density_matched_shares(lays, states, g.geometry_id, present) if len(present) > 1 else {}
        for grp, kind, ds in [('all', 'all', None)] + [(d, 'dataset', d) for d in dataset_ids]:
            sp = sup_all if ds is None else support(lays, states, g.geometry_id, ds)
            dm = sp['share_callable'] if ds is None or len(present) <= 1 else dms.get(ds)
            if dm is not None:
                out.values.append(Value(key, grp, kind, 'share_dm', dm, n_molecules=sp['molecules'],
                                        n_callable=sp['callable']))
            out.values.append(Value(key, grp, kind, 'share', sp['share'], n_molecules=sp['molecules'],
                                    n_callable=sp['callable']))
            out.values.append(Value(key, grp, kind, 'share_callable', sp['share_callable'],
                                    n_molecules=sp['molecules'], n_callable=sp['callable']))
            out.values.append(Value(key, grp, kind, 'share_partial', sp['share_partial'],
                                    n_molecules=sp['molecules'], n_callable=sp['callable']))
        out.values.append(Value(key, 'all', 'all', 'mean_score', g.mean_score))
        member: Dict[int, List[int]] = {}
        for i, k in g.members:
            member.setdefault(i, []).append(k)
        for i, (rec, lay) in enumerate(geo_mols):
            st = states[i][g.geometry_id]
            cols = {'reason': st, 'patch_start': '', 'patch_end': ''}
            interval = None
            if i in member:
                # every member patch of the molecule (two patches in one box: their hull)
                ps = [lay.patches[k] for k in member[i]]
                a, b = min(p.start for p in ps), max(p.end for p in ps)
                cols.update(patch_start=a, patch_end=b)
                interval = (a, b)
            out.states.append(State(key, rec, SCAN_STATE[st], (_REASON[st],), cols, interval))
        for rec, i in aliases:
            st = states[i][g.geometry_id]
            out.states.append(State(key, rec, SCAN_STATE[st], (_REASON[st],),
                                    {'reason': st, 'patch_start': '', 'patch_end': ''}, None))
