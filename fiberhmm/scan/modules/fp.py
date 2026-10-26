"""``fp``: footprint classes (the lattice recaller) as a scan module.

Each unit runs exactly what ``fiberhmm-consensus --bam ... --region <unit>``
runs: the same loader (``consensus.bam.load_bam_payload``, its own sampling
and duplicate collapse), the same sparse parameters and seeds, and
``consensus.workflow.run_analysis``. The engine's ``classes.tsv`` rows become
``fp/classes.tsv.bgz`` verbatim; its ``molecules.tsv.gz`` rows become states
(MA group ``fpcls`` in annotated mode, ``fp/states.tsv.bgz`` otherwise).
"""
from __future__ import annotations

import copy
import csv
import gzip
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from . import ScanModuleBase
from ..identity import molecule_id
from ..types import EvidenceMember, FeatureTypeDecl, Hit, MAGroupDecl, RecordRef, ScoreDecl, State, UnitResult, Value

# consensus parameter groups accepted under "fp"
GROUPS = ('input', 'sr', 'cr', 'rescue', 'cross', 'comparability', 'split', 'compute', 'families', 'recaller')
# execution-only controls: never change a row, so they are left out of the settings hash
EXECUTION_ONLY = {'compute': ('cores', 'fit_cache_dir', 'require_native_cache', 'maximum_matrix_mb', 'batch_size',
                              'accelerator_mb')}
CAP_NAMES = ('max_discovery_molecules', 'max_quantification_molecules', 'sample_seed')
STATE = {'member': '1', 'non_member': '0', 'abstain': '.'}
TIERS = {'': 0, 'core': 1, 'edge': 2, 'loose': 3}
VALUE_FIELDS = ('prevalence_edge', 'prevalence_loose', 'broader', 'other_shape', 'accessible', 'support_gain_nats',
                'resolution_nats', 'unknown_accessible_fraction', 'efficiency')


def _num(text) -> Optional[float]:
    if text is None or text == '':
        return None
    try:
        v = float(text)
    except ValueError:
        return None
    return None if math.isnan(v) else v


def _int(text) -> Optional[int]:
    v = _num(text)
    return None if v is None else int(v)


def member_records(unit: Mapping[str, Any], dataset, chrom: str):
    """RecordRefs of an evidence unit's member source records, sharing one molecule id."""
    paths = {str(Path(p).resolve()): i for i, p in enumerate(dataset.paths)}
    members = unit.get('source_members') or []
    names = [unit.get('read_name', '')] + [m.get('read_name', '') for m in members] + \
        list(unit.get('physical_source_names') or [])
    mol = molecule_id(names)
    out = []
    for m in members:
        sha = m.get('record_sha256')
        if not sha:
            continue
        lib = m.get('library_id') or ''
        idx = paths.get(str(Path(lib).resolve())) if lib else None
        if idx is None:
            idx = 0 if len(dataset.paths) == 1 else -1
        if idx < 0:
            raise ValueError(f'cannot identify the source BAM of an evidence unit member (library {lib!r})')
        out.append(RecordRef(dataset.dataset_id, idx, str(m.get('read_name', '')), str(sha),
                             int(m.get('alignment_occurrence', 0)), chrom, int(m.get('reference_start', 0)),
                             int(m.get('reference_end', 0)), mol))
    return out


def _read_tsv(path: Path, compressed=False) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    opener = (lambda p: gzip.open(p, 'rt', newline='')) if compressed else (lambda p: open(p, newline=''))
    with opener(path) as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


class FpModule(ScanModuleBase):
    name = 'fp'
    version = '1.0'
    experimental = False
    reconcile = 'anchor'
    max_support_bp = 12_500
    wants_molecules = False
    native_tables = ('classes',)
    state_groups = {'fp.class': 'fpcls'}

    def parameter_schema(self) -> dict:
        from fiberhmm.inference.consensus.parameters import parameter_schema
        schema = parameter_schema()
        return {g: {'default': {}, 'type': 'group', 'help': f'fiberhmm-consensus "{g}" parameter group',
                    'parameters': schema.get(g, schema)} for g in GROUPS}

    def resolve(self, params: Mapping[str, Any]) -> dict:
        from fiberhmm.inference.consensus.parameters import options_dict, parse_options
        params = copy.deepcopy(dict(params or {}))
        caps = params.pop('caps', {}) or {}
        unknown = sorted(set(params) - set(GROUPS))
        if unknown:
            raise ValueError(f'fp: unknown parameter group(s) {", ".join(unknown)} (use fiberhmm-consensus groups: '
                             f'{", ".join(GROUPS)})')
        values = {g: dict(v) for g, v in params.items() if v}
        engine = values.get('cr', {}).get('engine', 'lattice_recaller')
        if engine != 'lattice_recaller':
            raise ValueError('fp runs the lattice recaller (cr.engine = lattice_recaller)')
        values.setdefault('cr', {})['engine'] = 'lattice_recaller'
        # Explicit caps (the cap flags) override the parameter groups, as in fiberhmm-consensus.
        for name in CAP_NAMES:
            if caps.get(name) is not None:
                values.setdefault('recaller', {})[name] = int(caps[name])
        for group, names in EXECUTION_ONLY.items():
            for n in names:
                values.get(group, {}).pop(n, None)
        values = {g: v for g, v in values.items() if v}
        resolved = options_dict(parse_options(dict(values, compute=dict(values.get('compute', {}),
                                                                         require_native_cache=False))))
        for group, names in EXECUTION_ONLY.items():
            for n in names:
                resolved.get(group, {}).pop(n, None)
        return dict(parameters=values, resolved=resolved)

    def feature_types(self):
        scores = (ScoreDecl('prevalence', 'Prevalence', 'fraction', 'high', True),
                  ScoreDecl('support_gain_nats', 'Support gain', 'nats', 'high', True))
        return [FeatureTypeDecl('fp.class', 'fp', 'Footprint class', 'Footprint classes', ('hit', 'state'), 'centre',
                                scores=scores,
                                state_rule='1: the molecule is a class member (posterior odds over prior >= '
                                           'recaller.bf_threshold); 0: non-member; .: abstain or not scored',
                                columns=('class_id', 'group', 'start_exact', 'end_exact', 'L0', 'L1', 'R0', 'R1',
                                         'calls', 'stability', 'status', 'supported_channels'))]

    def ma_groups(self):
        return [MAGroupDecl('fpcls', 1, 'state', ('state', 'p', 'tier'),
                            state_codes=(('1', 'member'), ('2', 'non-member'), ('3', 'abstain')), an='hit_id')]

    def state_columns(self):
        return ['channel', 'posterior', 'log_bf', 'tier', 'call_start', 'call_end']

    def load_policy(self, settings):
        rec = settings['resolved'].get('recaller', {})
        return dict(loader='fiberhmm.inference.consensus.bam.load_bam_payload', mode='recaller',
                    caps={n: rec.get(n, 0) for n in CAP_NAMES if n in rec},
                    minimum_mapq=settings['resolved'].get('input', {}).get('minimum_mapq'))

    def run_unit(self, unit, inputs, settings, calibration, upstream, work, progress):
        from fiberhmm.inference.consensus.bam import load_bam_payload
        from fiberhmm.inference.consensus.parameters import parse_options
        from fiberhmm.inference.consensus.workflow import run_analysis
        values = copy.deepcopy(settings['parameters'])
        values.setdefault('compute', {})['cores'] = max(1, int(inputs.cores))
        values['compute']['require_native_cache'] = False
        datasets = [dict(dataset_id=d.dataset_id, paths=list(d.paths), chemistry=d.chemistry) for d in inputs.datasets]
        payload = load_bam_payload(datasets, dict(chrom=unit.chrom, start=unit.start, end=unit.end),
                                   parse_options(values), progress)
        if not any(s['units'] for s in payload['strata']):
            return UnitResult(extra=dict(note='no evidence units in the unit'))
        result = run_analysis(payload, values, work, progress=progress)
        return self.unit_result(unit, inputs, payload, result, Path(work))

    # ---- conversion of the engine's outputs
    def unit_result(self, unit, inputs, payload, result, work: Path) -> UnitResult:
        by_ds = {d.dataset_id: d for d in inputs.datasets}
        rows = _read_tsv(work / 'classes.tsv')
        mols = _read_tsv(work / 'molecules.tsv.gz', compressed=True)
        status = {c['id']: c for c in (result.get('recaller') or {}).get('classes', [])}
        res = UnitResult()
        by_class: Dict[str, List[Dict[str, str]]] = {}
        for r in rows:
            by_class.setdefault(r['class_id'], []).append(r)
        for cid, crows in by_class.items():
            s, e = float(crows[0]['start']), float(crows[0]['end'])
            prev = [v for v in (_num(r.get('prevalence')) for r in crows) if v is not None]
            info = status.get(cid, {})
            res.hits.append(Hit('fp.class', cid, unit.chrom, int(math.floor(s)), int(math.ceil(e)),
                                int(math.floor((s + e) / 2)), score=max(prev) if prev else None, label=cid,
                                columns=dict(class_id=cid, group=_int(crows[0].get('group')), start_exact=s,
                                             end_exact=e, L0=crows[0].get('L0'), L1=crows[0].get('L1'),
                                             R0=crows[0].get('R0'), R1=crows[0].get('R1'),
                                             calls=_int(crows[0].get('calls')),
                                             stability=_num(crows[0].get('stability')),
                                             status=info.get('status', ''),
                                             supported_channels=info.get('supported_channels'))))
            for r in crows:
                n = _int(r.get('molecules'))
                res.values.append(Value(cid, r['channel'], 'channel', 'prevalence', _num(r.get('prevalence')),
                                        lo=_num(r.get('prevalence_lower_bound')), n_molecules=n, n_callable=n))
                for f in VALUE_FIELDS:
                    res.values.append(Value(cid, r['channel'], 'channel', f, _num(r.get(f)), n_molecules=n,
                                            n_callable=n))
                res.native.setdefault('classes', []).append(dict(hit_key=cid, **{k: (v if v is not None else '')
                                                                                  for k, v in r.items()}))
        # evidence units -> member records
        units = {}
        for stratum in payload['strata']:
            ds = by_ds[stratum['dataset_id']]
            for u in stratum['units']:
                recs = member_records(u, ds, unit.chrom)
                units[u['unit_id']] = recs
                for rec in recs:
                    res.evidence.append(EvidenceMember(u['unit_id'], rec))
        spans = {h.key: (h.start, h.end) for h in res.hits}
        for m in mols:
            cid = m['class_id']
            if cid not in spans:
                continue
            state = STATE.get(m.get('label', ''), '.')
            a, b = _num(m.get('start')), _num(m.get('end'))
            interval = (int(math.floor(a)), int(math.ceil(b))) if (state == '1' and a is not None and b is not None) \
                else spans[cid]
            post = _num(m.get('posterior')) or 0.0
            q = (int(round(255 * min(1.0, max(0.0, post)))), TIERS.get(m.get('tier', ''), 0))
            cols = dict(channel=m.get('channel'), posterior=_num(m.get('posterior')), log_bf=_num(m.get('log_bf')),
                        tier=m.get('tier', ''), call_start=a, call_end=b)
            for rec in units.get(m['unit_id'], ()):
                res.states.append(State(cid, rec, state, q, cols, interval))
        return res
