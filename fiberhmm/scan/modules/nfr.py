"""``nfr`` (EXPERIMENTAL, as ``fiberhmm-nfr``): NFR regions and variants as a scan module.

Each unit runs what ``fiberhmm-nfr --bam ... --region <unit>`` runs with
automatic NFR detection: the consensus loader with ``mode='nfr'`` sampling,
then ``accessibility.workflow.run_accessibility``. The engine's
``variants.tsv`` rows become ``nfr/variants.tsv.bgz`` verbatim; per-molecule
memberships become states (MA group ``nfrvar`` or ``nfr/states.tsv.bgz``).
Molecule-wide scalars (``nfrmol``) come from each record's own >= 90 bp
nucleosome calls.
"""
from __future__ import annotations

import copy
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from . import ScanModuleBase
from .fp import member_records
from ..types import (EvidenceMember, FeatureTypeDecl, Hit, MAGroupDecl, MoleculeResult, ScalarCodec, ScoreDecl,
                     State, UnitResult, Value)

LOADER_GROUPS = ('input', 'compute', 'families', 'cr', 'recaller')
CAP_NAMES = ('max_discovery_molecules', 'max_quantification_molecules', 'sample_seed')
NUC_MIN_BP = 90
NFR_MIN_BP = 175
REGION_COLUMNS = ('nfr', 'status', 'source', 'callable', 'reads', 'censored', 'overflow_reads', 'profile_peak', 'k',
                  'closed')
VARIANT_COLUMNS = ('name', 'L', 'R', 'Lsd', 'Rsd', 'width', 'depth', 'relation', 'prevalence_strict', 'prevalence_em',
                   'ci_lo', 'ci_hi', 'only', 'with_ref', 'gain_nats', 'stability', 'exp_reads', 'robust')


def nucleosome_runs(read) -> List[tuple]:
    """(start, length) of the record's nucleosome calls (MA, else fibertools Ma, else ns/nl), any frame:
    lengths and gaps are the same in SEQ and molecular frame."""
    from fiberhmm.io.ma_tags import parse_ma_tag
    for tag in ('MA', 'Ma'):
        if read.has_tag(tag):
            try:
                return sorted(parse_ma_tag(str(read.get_tag(tag)))['nuc'])
            except ValueError:
                return []
    if read.has_tag('ns') and read.has_tag('nl'):
        return sorted(zip((int(x) for x in read.get_tag('ns')), (int(x) for x in read.get_tag('nl'))))
    return []


def molecule_scalars(read) -> Dict[str, Optional[float]]:
    """nfrmol scalars: openness, n_nfr and longest_linker over the >= 90 bp nucleosome-bounded extent."""
    nucs = [(s, s + n) for s, n in nucleosome_runs(read) if n >= NUC_MIN_BP]
    if len(nucs) < 2:
        return dict(openness=None, n_nfr=None, longest_linker=None)
    lo, hi = nucs[0][0], max(e for _s, e in nucs)
    covered = 0
    cur_s, cur_e = nucs[0]
    gaps = []
    for s, e in nucs[1:]:
        if s > cur_e:
            covered += cur_e - cur_s
            gaps.append(s - cur_e)
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    covered += cur_e - cur_s
    extent = hi - lo
    return dict(openness=(extent - covered) / extent if extent > 0 else None,
                n_nfr=float(sum(1 for g in gaps if g >= NFR_MIN_BP)),
                longest_linker=float(max(gaps)) if gaps else 0.0)


def _num(v) -> Optional[float]:
    if v is None or v == '':
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


class NfrModule(ScanModuleBase):
    name = 'nfr'
    version = '1.0'
    experimental = True
    reconcile = 'anchor'
    max_support_bp = 12_500
    wants_molecules = True
    native_tables = ('variants',)
    state_groups = {'nfr.region': 'nfrvar', 'nfr.variant': 'nfrvar'}

    def parameter_schema(self) -> dict:
        from fiberhmm.inference.accessibility.workflow import NFROptions
        schema = {'nfr': {'default': {}, 'type': 'group', 'help': 'fiberhmm-nfr engine parameters (see fiberhmm-nfr '
                                                                   '--schema)', 'parameters': NFROptions().as_dict()}}
        for g in LOADER_GROUPS:
            schema[g] = {'default': {}, 'type': 'group', 'help': f'loader group "{g}" as for fiberhmm-consensus'}
        schema['use_fp_classes'] = {'default': False, 'type': 'bool',
                                    'help': 'use the fp module\'s classes as co-accessibility elements (makes nfr need fp)'}
        return schema

    def resolve(self, params: Mapping[str, Any]) -> dict:
        from fiberhmm.inference.accessibility.workflow import NFROptions
        from fiberhmm.inference.consensus.parameters import parse_options
        params = copy.deepcopy(dict(params or {}))
        caps = params.pop('caps', {}) or {}
        use_fp = bool(params.pop('use_fp_classes', False))
        unknown = sorted(set(params) - set(LOADER_GROUPS) - {'nfr'})
        if unknown:
            raise ValueError(f'nfr: unknown parameter group(s) {", ".join(unknown)}')
        nfr = dict(params.pop('nfr', {}) or {})
        if nfr.get('nfr_regions'):
            raise ValueError('nfr: a scan detects NFR regions per unit; nfr_regions cannot be given')
        for name in CAP_NAMES:
            if name in caps and name not in nfr and caps[name] not in (None, 0, 1 if name == 'sample_seed' else 0):
                nfr[name] = int(caps[name])
        opt = NFROptions.from_params(nfr)
        loader = {g: dict(v) for g, v in params.items() if v}
        loader.get('compute', {}).pop('cores', None)
        loader.setdefault('cr', {}).setdefault('engine', 'lattice_recaller')
        parse_options(loader)                                     # validates the loader groups
        return dict(nfr=opt.as_dict(), loader=loader, use_fp_classes=use_fp)

    def needs(self, settings):
        return {'fp'} if settings.get('use_fp_classes') else set()

    def feature_types(self):
        return [
            FeatureTypeDecl('nfr.region', 'nfr', 'NFR', 'NFRs', ('hit', 'state'), 'centre',
                            scores=(ScoreDecl('closed', 'Closed share', 'fraction', 'none', False),),
                            state_rule='1: the molecule has an opening (>= min_gap_bp) in the NFR region; 0: closed; '
                                       '.: not callable (no nucleosome on both sides)',
                            transferable_state=True, columns=REGION_COLUMNS),
            FeatureTypeDecl('nfr.variant', 'nfr', 'NFR variant', 'NFR variants', ('hit', 'state'), 'centre',
                            scores=(ScoreDecl('prevalence', 'Prevalence (EM)', 'fraction', 'high', True),
                                    ScoreDecl('gain_nats', 'Support gain', 'nats', 'high', False)),
                            state_rule='1: P(variant in the molecule\'s configuration) >= open_threshold; 0: below; '
                                       '.: not callable', transferable_state=False,
                            columns=('nfr',) + VARIANT_COLUMNS),
        ]

    def ma_groups(self):
        return [
            MAGroupDecl('nfrvar', 1, 'state', ('state', 'p'), state_codes=(('1', 'open'), ('2', 'closed')),
                        an='hit_id'),
            MAGroupDecl('nfrmol', 1, 'scalar', ('openness', 'n_nfr', 'longest_linker_hi', 'longest_linker_lo'),
                        codecs=(ScalarCodec('openness', 1, 0.004, 0.0, 'fraction'),
                                ScalarCodec('n_nfr', 1, 1.0, 0.0, 'count'),
                                ScalarCodec('longest_linker', 2, 1.0, 0.0, 'bp')),
                        an='group'),
        ]

    def state_columns(self):
        return ['map', 'map_posterior', 'p']

    def load_policy(self, settings):
        n = settings['nfr']
        return dict(loader='fiberhmm.inference.consensus.bam.load_bam_payload', mode='nfr',
                    caps={k: n.get(k) for k in CAP_NAMES})

    def run_unit(self, unit, inputs, settings, calibration, upstream, work, progress):
        from fiberhmm.inference.accessibility.workflow import NFROptions, classes_from_rows, run_accessibility
        from fiberhmm.inference.consensus.bam import load_bam_payload
        from fiberhmm.inference.consensus.parameters import parse_options
        opt = NFROptions.from_params(settings['nfr'])
        values = copy.deepcopy(settings['loader'])
        values.setdefault('compute', {})['cores'] = max(1, int(inputs.cores))
        options = parse_options(values)
        caps = [opt.max_quantification_molecules, opt.max_discovery_molecules]
        sampling = dict(cap=max(caps) if all(caps) else 0, discovery_cap=opt.max_discovery_molecules,
                        seed=opt.sample_seed, mode='nfr', windows=None, nfr_options=opt)
        datasets = [dict(dataset_id=d.dataset_id, paths=list(d.paths), chemistry=d.chemistry) for d in inputs.datasets]
        payload = load_bam_payload(datasets, dict(chrom=unit.chrom, start=unit.start, end=unit.end), options,
                                   progress, loader_sampling=sampling)
        if not any(s['units'] for s in payload['strata']):
            return UnitResult(extra=dict(note='no evidence units in the unit'))
        classes = None
        if settings.get('use_fp_classes') and 'fp' in upstream:
            classes = self._classes_from_fp(upstream['fp'], opt)
        prov = dict(datasets=datasets, evidence=None, loader_parameters={'input': _asdict(options['input'])},
                    classes=[])
        result = run_accessibility(payload, opt, Path(work), progress, classes=classes or None, inputs=prov)
        return self.unit_result(unit, inputs, payload, result, opt, Path(work))

    def _classes_from_fp(self, fp_result, opt):
        from fiberhmm.inference.accessibility.workflow import classes_from_rows
        sup, state = {}, {}
        for row in fp_result.native.get('classes', []):
            try:
                prev = float(row.get('prevalence') or 'nan')
            except ValueError:
                prev = float('nan')
            if str(row.get('supported')) == 'True' and prev >= opt.class_min_prevalence:
                sup[(row['class_id'], row['channel'])] = row
        # fp and nfr load the same records into the same unit_ids (consensus.adapter.unit_identity)
        unit_of = {e.record.key: e.evidence_unit for e in fp_result.evidence}
        for s in fp_result.states:
            ch = (s.columns or {}).get('channel')
            if s.state in ('1', '0') and (s.hit_key, ch) in sup and s.record.key in unit_of:
                state.setdefault(s.hit_key, {})[unit_of[s.record.key]] = (int(s.state == '1'), ch)
        return classes_from_rows(sup, state) if sup else None

    def unit_result(self, unit, inputs, payload, result, opt, work: Path) -> UnitResult:
        by_ds = {d.dataset_id: d for d in inputs.datasets}
        res = UnitResult()
        variant_rows = {}
        with open(work / 'variants.tsv', newline='') as handle:
            for r in csv.DictReader(handle, delimiter='\t'):
                variant_rows.setdefault(r['id'], r)
        for n in result['nfrs']:
            nid = n['id']
            res.hits.append(Hit('nfr.region', nid, unit.chrom, int(n['start']), int(n['end']),
                                (int(n['start']) + int(n['end'])) // 2, score=_num(n.get('closed')), label=nid,
                                columns=dict(nfr=nid, status=n.get('status'), source=n.get('source'),
                                             callable=n.get('callable'), reads=n.get('reads'),
                                             censored=n.get('censored'), overflow_reads=n.get('overflow_reads'),
                                             profile_peak=n.get('profile_peak'), k=n.get('k'),
                                             closed=n.get('closed'))))
            res.values.append(Value(nid, 'all', 'all', 'closed', _num(n.get('closed')), n_molecules=n.get('reads'),
                                    n_callable=n.get('callable')))
            other = n.get('other') or {}
            res.values.append(Value(nid, 'all', 'all', 'other_em', _num(other.get('em')),
                                    lo=_num((other.get('ci') or [None, None])[0]),
                                    hi=_num((other.get('ci') or [None, None])[1]),
                                    n_molecules=n.get('reads'), n_callable=n.get('callable')))
            for v in list(n.get('variants') or []) + list(n.get('depth_states') or []):
                vid = v['id']
                L = _num(v.get('L')) if v.get('L') is not None else float(n['start'])
                R = _num(v.get('R')) if v.get('R') is not None else float(n['end'])
                cols = {c: v.get(c) for c in VARIANT_COLUMNS if c in v}
                cols.update(nfr=nid, prevalence_em=v.get('prevalence'), prevalence_strict=v.get('strict'),
                            ci_lo=(v.get('ci') or [None, None])[0], ci_hi=(v.get('ci') or [None, None])[1])
                res.hits.append(Hit('nfr.variant', vid, unit.chrom, int(math.floor(L)), int(math.ceil(R)),
                                    int(math.floor((L + R) / 2)), score=_num(v.get('prevalence')),
                                    label=f"{nid} {v.get('name', '')}".strip(), parent=nid, columns=cols))
                ci = v.get('ci') or [None, None]
                res.values.append(Value(vid, 'all', 'all', 'prevalence_em', _num(v.get('prevalence')),
                                        lo=_num(ci[0]), hi=_num(ci[1]), n_molecules=n.get('reads'),
                                        n_callable=n.get('callable')))
                res.values.append(Value(vid, 'all', 'all', 'prevalence_strict', _num(v.get('strict')),
                                        n_molecules=n.get('reads'), n_callable=n.get('callable')))
                for ch, rec in sorted((v.get('by_channel') or {}).items()):
                    res.values.append(Value(vid, ch, 'channel', 'em', _num(rec.get('em')), n_molecules=rec.get('n'),
                                            n_callable=rec.get('n')))
                    res.values.append(Value(vid, ch, 'channel', 'strict', _num(rec.get('strict')),
                                            n_molecules=rec.get('n'), n_callable=rec.get('n')))
                row = variant_rows.get(vid)
                if row is not None:
                    res.native.setdefault('variants', []).append(dict(hit_key=vid, **row))
        # evidence units -> member records; per-molecule states
        units = {}
        for stratum in payload['strata']:
            ds = by_ds[stratum['dataset_id']]
            for u in stratum['units']:
                recs = member_records(u, ds, unit.chrom)
                units[u['unit_id']] = recs
                for rec in recs:
                    res.evidence.append(EvidenceMember(u['unit_id'], rec))
        variants_of = {n['id']: [v for v in (n.get('variants') or [])] for n in result['nfrs']}
        for uid, m in sorted(result['molecules'].items()):
            for nid, rec in sorted((m.get('nfr') or {}).items()):
                if rec.get('status') != 'callable':
                    continue
                is_open = bool(rec.get('gaps'))
                cols = dict(map=rec.get('map'), map_posterior=rec.get('map_posterior'),
                            p=';'.join(f'{k}={v}' for k, v in (rec.get('p') or {}).items()))
                for r in units.get(uid, ()):
                    res.states.append(State(nid, r, '1' if is_open else '0', (255 if is_open else 0,), cols))
                for v in variants_of.get(nid, []):
                    p = (rec.get('p') or {}).get(v['name'])
                    if p is None:
                        continue
                    state = '1' if p >= opt.open_threshold else '0'
                    q = (int(round(255 * min(1.0, max(0.0, float(p))))),)
                    for r in units.get(uid, ()):
                        res.states.append(State(v['id'], r, state, q, dict(cols, p=p)))
        return res

    def molecule_outputs(self, molecule, settings, calibration):
        return MoleculeResult(molecule.record, scalars=molecule_scalars(molecule.read))


def _asdict(value):
    import dataclasses
    return dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value
