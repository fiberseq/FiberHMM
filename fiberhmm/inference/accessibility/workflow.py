"""NFR variants + element co-accessibility for one region (EXPERIMENTAL preview).

``run_accessibility(payload, params, output_dir, progress, classes=None, clusters=None)`` takes a FiberHMM consensus
payload (``consensus.bam.load_bam_payload``; the same evidence units, unit_ids, duplicate collapse and DAF channels
as the lattice recaller, so NFR and footprint-class results join by unit_id) and returns a JSON-able result. With an
``output_dir`` it also writes variants.tsv, configurations.tsv, molecules.tsv.gz, coaccess.tsv, combos.tsv,
result.json and manifest.json. Outputs are deterministic: no timestamps, sorted keys, gzip mtime 0.
"""
from __future__ import annotations

import csv
import dataclasses
import gzip
import hashlib
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import coaccess as C
from . import gaps as G
from . import variants as V

SCHEMA = 'fiberhmm.accessibility.preview.v0'
EXPERIMENTAL_NOTE = ('EXPERIMENTAL preview: NFR variants and element co-accessibility. Outputs, parameters and formats '
                     'may change without notice.')


@dataclass
class NFROptions:
    # discovery (the lattice recaller's recipe on gap edges)
    stringency: float = 0.9          # prediction strength needed for k: lower admits finer variants (0.85 = finer)
    kmax: int = 8
    splits: int = 6
    seed: int = 1
    min_gap_bp: int = 60             # smallest opening counted (linkers below it are ignored)
    min_reads: float = 10.0          # expected member reads a variant needs
    identity_nats: float = 5.0       # merge overlapping candidates while held-out gain < this
    support_nats: float = 5.0        # keep a variant only if dropping it costs >= this (held-out)
    folds: int = 3
    max_gaps: int = 3                # gaps per read inside an NFR region (more: flagged as overflow)
    bootstrap: int = 200             # read bootstrap for prevalence intervals (conditional on the catalogue)
    shift_bp: int = 25               # relation-label tolerance (cosmetic)
    robust: int = 0                  # n reordered discovery reruns (0 = off)
    mode: str = 'variants'           # 'variants' (discovered) | 'depth' (Timer 175/300/500-bp width states)
    # NFR regions
    nfr_regions: tuple = ()          # ((start, end), ...); empty: detected from the gap >= 175-bp profile
    detect_threshold: float = 0.15
    # elements and co-accessibility
    elements: str = 'variants'       # 'variants' (each variant an element) | 'nfrs' (any opening in the NFR)
    open_threshold: float = 0.5      # a read is open for a variant when P(variant in configuration) >= this
    internal_footprint_labels: bool = True   # split whose protection matches a supported class -> internal footprint
    internal_footprint_tolerance_bp: int = 25
    pairs: str = 'nfr'               # 'nfr' (pairs with at least one NFR element) | 'all' (also class x class)
    per_bin: int = 50                # reads per openness stratum
    openness_pad_bp: int = 0         # extra bp masked around both tested elements when measuring per-read openness
    min_spanning: int = 50
    min_marginal: int = 10
    n_perm: int = 500                # display-only permutations (null centre of the pooled OR)
    q_max: float = 0.1
    class_min_prevalence: float = 0.03
    within_clusters: int = 0         # k of the masked k-means within-cluster check (0 = off)
    combo_elements: tuple = ()       # element ids (3-8); empty: automatic (non-overlapping NFR elements)
    combo_max: int = 6
    combo_samples: int = 300

    @classmethod
    def from_params(cls, params=None):
        params = dict(params or {})
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(params) - names)
        if unknown:
            raise ValueError(f'Unknown NFR parameter(s): {", ".join(unknown)}')
        opt = cls(**params)
        opt.validate()
        return opt

    def validate(self):
        if not 0 < self.stringency <= 1:
            raise ValueError('stringency must be in (0, 1]')
        if self.mode not in ('variants', 'depth'):
            raise ValueError("mode must be 'variants' or 'depth'")
        if self.elements not in ('variants', 'nfrs'):
            raise ValueError("elements must be 'variants' or 'nfrs'")
        if self.pairs not in ('nfr', 'all'):
            raise ValueError("pairs must be 'nfr' or 'all'")
        if self.min_gap_bp < 1 or self.max_gaps < 1 or self.kmax < 1 or self.folds < 2:
            raise ValueError('min_gap_bp, max_gaps and kmax must be >= 1 and folds >= 2')
        if self.bootstrap < 0 or self.robust < 0 or self.within_clusters < 0:
            raise ValueError('bootstrap, robust and within_clusters must be >= 0')
        regions = []
        for r in self.nfr_regions or ():
            a, b = (int(r[0]), int(r[1])) if not isinstance(r, dict) else (int(r['start']), int(r['end']))
            if b <= a:
                raise ValueError(f'NFR region {a}-{b}: end must be greater than start')
            regions.append((a, b))
        self.nfr_regions = tuple(sorted(regions))
        self.combo_elements = tuple(self.combo_elements or ())
        if self.combo_elements and not 3 <= len(self.combo_elements) <= 8:
            raise ValueError('combo_elements: choose 3-8 elements')

    def as_dict(self):
        d = dataclasses.asdict(self)
        d['nfr_regions'] = [list(r) for r in self.nfr_regions]
        d['combo_elements'] = list(self.combo_elements)
        return d


def _report(progress, stage, message, **work):
    if progress is None:
        return
    fn = getattr(progress, 'report', progress)
    try:
        fn(stage, message, **work)
    except TypeError:
        fn(stage, message)


def _r(x, nd=4):
    if x is None:
        return None
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, nd)


# ------------------------------------------------------------------ footprint classes (lattice recaller output)
def load_recaller_classes(directory, min_prevalence=0.03, prefix=''):
    """Supported footprint classes of a lattice-recaller run directory (classes.tsv + molecules.tsv.gz) as elements.

    A read is a member or non-member only on a channel where the class is supported; abstentions are not spanning.
    ``prefix`` is prepended to class ids (to combine several runs)."""
    d = Path(directory)
    with open(d/'classes.tsv', newline='') as fh:
        rows = list(csv.DictReader(fh, delimiter='\t'))
    sup = {}
    for r in rows:
        try:
            prev = float(r.get('prevalence') or 'nan')
        except ValueError:
            prev = float('nan')
        if str(r.get('supported')) == 'True' and prev >= min_prevalence:
            sup[(r['class_id'], r['channel'])] = r
    state = {}
    with gzip.open(d/'molecules.tsv.gz', 'rt', newline='') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if (r['class_id'], r['channel']) in sup and r['label'] in ('member', 'non_member'):
                state.setdefault(r['class_id'], {})[r['unit_id']] = (int(r['label'] == 'member'), r['channel'])
    return classes_from_rows(sup, state, prefix)


def classes_from_rows(sup, state, prefix=''):
    """sup: {(class_id, channel): classes.tsv row (start, end, prevalence)}; state: {class_id: {unit_id: (0/1, channel)}}."""
    out = []
    for cid in sorted({c for c, _ in sup}):
        chans = sorted(ch for c, ch in sup if c == cid)
        r = sup[(cid, chans[0])]
        prev = float(np.mean([float(sup[(cid, ch)]['prevalence']) for ch in chans]))
        st = state.get(cid, {})
        out.append(dict(id=f'{prefix}{cid}', class_id=cid, kind='tf', subtype='class', start=float(r['start']), end=float(r['end']),
                        label=f'{prefix}class {cid.rsplit("_", 1)[-1]}', prevalence=prev, channels=chans,
                        state={u: s for u, (s, _ch) in st.items()}, channel={u: ch for u, (_s, ch) in st.items()}))
    return out


# ------------------------------------------------------------------ the run
def _nfr_regions(units, region, opt):
    if opt.nfr_regions:
        return [dict(start=a, end=b, source='given') for a, b in opt.nfr_regions]
    return [dict(r, source='detected') for r in G.detect_nfrs(units, region['start'], region['end'], opt.detect_threshold)]


def _variant_row(nfr_id, v, by_channel, robust_frac):
    return dict(id=f"{nfr_id}:{v['name']}", nfr=nfr_id, name=v['name'], start=_r(v['L'], 1), end=_r(v['R'], 1),
                L=_r(v['L'], 1), R=_r(v['R'], 1), Lsd=_r(v['Lsd'], 1), Rsd=_r(v['Rsd'], 1), width=_r(v['width'], 1),
                depth=v['depth'], relation=v['relation'], prevalence=_r(v['prevalence']), strict=_r(v['strict']),
                ci=[_r(v['ci'][0]), _r(v['ci'][1])] if v.get('ci') else None, only=_r(v['only']), with_ref=_r(v['with_ref']),
                gain_nats=_r(v['gain'], 1), stability=_r(v['stability'], 3), exp_reads=_r(v['exp_reads'], 1),
                robust=None if robust_frac is None else _r(robust_frac, 3), by_channel=by_channel)


def _by_channel(reads, Pv, j):
    out = {}
    chans = np.array([r['ch'] for r in reads])
    for ch in sorted(set(chans)):
        m = chans == ch
        out[ch] = dict(n=int(m.sum()), em=_r(Pv[m, j].mean()) if m.any() else None,
                       strict=_r(np.mean(Pv[m, j] >= 0.9)) if m.any() else None)
    return out


def _internal_footprints(nfr, classes, opt):
    """Split configurations (two variants on one read) whose protection between them matches a supported class."""
    if not classes or not opt.internal_footprint_labels:
        return
    vs = {v['name']: v for v in nfr['variants']}
    tol = opt.internal_footprint_tolerance_bp
    for cfg in nfr['configurations']:
        names = [x for x in cfg['label'].split('+') if x in vs]
        if len(names) < 2 or cfg['label'] in ('closed',):
            continue
        hits = []
        for a, b in zip(names, names[1:]):
            p0, p1 = vs[a]['R'], vs[b]['L']
            if p1 <= p0:
                continue
            for c in classes:
                w = c['end'] - c['start']
                if c['start'] >= p0 - tol and c['end'] <= p1 + tol and w >= 0.5*(p1 - p0):
                    hits.append(c['id'])
        if hits:
            cfg['internal_footprint'] = sorted(set(hits))
            cfg['display_label'] = f"{cfg['label']}: full NFR with internal footprint ({', '.join(sorted(set(hits)))})"


def _masked_clusters(units, cov, els, k, region):
    """The within-cluster check: k-means (fixed seed) of locus accessibility with the two tested elements' bins
    (+-50 bp) masked, so clustering cannot condition on the outcome (EXPERIMENTAL diagnostic)."""
    from sklearn.cluster import KMeans
    by = {u['unit_id']: u for u in units}
    uid = [u for u in sorted(cov) if u in by]
    e_lo = max(region['start'], min(e['start'] for e in els) - 150)
    e_hi = min(region['end'], max(e['end'] for e in els) + 150)
    x, M = G.access_matrix([by[u] for u in uid], e_lo, e_hi)
    full = {u for u in uid if cov[u]['x'][0] <= e_lo + 10 and cov[u]['x'][-1] >= e_hi - 20}
    row_of = {u: i for i, u in enumerate(uid)}

    def clusters(uids, ex):
        keep = np.ones(len(x), bool)
        for a, b in ex:
            keep &= ~((x >= a - 50) & (x < b + 50))
        us = [u for u in uids if u in full]
        if len(us) < 50 or keep.sum() < 2:
            return {}
        lab = KMeans(k, n_init=4, random_state=42).fit_predict(M[[row_of[u] for u in us]][:, keep])
        return dict(zip(us, lab.tolist()))
    clusters.n = len(full)
    return clusters


def _auto_combo(els, opt):
    nf = [e for e in els if e['kind'] == 'nfr' and e.get('prevalence') is not None]
    nf.sort(key=lambda e: (abs(e['prevalence'] - .5), e['id']))
    chosen = []
    for e in nf:
        if len(chosen) >= opt.combo_max:
            break
        if all(not C.overlapping(e, o) for o in chosen):
            chosen.append(e)
    chosen.sort(key=lambda e: (e['start'], e['end']))
    return chosen if len(chosen) >= 3 else []


def run_accessibility(payload, params=None, output_dir=None, progress=None, classes=None, clusters=None, inputs=None):
    """NFR variants, membership, prevalence and element co-accessibility for the payload's region.

    params: NFROptions fields (dict). classes: footprint-class elements (``load_recaller_classes``) or None.
    clusters: optional {unit_id: label} for the within-cluster check (else ``within_clusters`` k-means, masked).
    inputs: provenance for the manifest (input files, loader parameters)."""
    opt = params if isinstance(params, NFROptions) else NFROptions.from_params(params)
    region = dict(payload['region'])
    units = G.units_of(payload)
    warnings = []
    _report(progress, 'nfr', f'{len(units)} molecules in {region["chrom"]}:{region["start"]}-{region["end"]}')
    regions = _nfr_regions(units, region, opt)
    if not regions:
        warnings.append('No NFR region found: no stretch where more than '
                        f'{opt.detect_threshold:.0%} of callable reads are inside a >= 175-bp gap. Give the NFR region explicitly.')
    nfrs, elements, gaps_by_uid, molecules = [], [], {}, {}
    for ni, reg in enumerate(regions):
        nfr_id = f'N{ni + 1}'
        nreg = (reg['start'], reg['end'])
        _report(progress, 'nfr', f'{nfr_id} {nreg[0]}-{nreg[1]}: per-read gaps', completed=ni, total=len(regions))
        reads = G.collect(units, nreg, opt.min_gap_bp, opt.max_gaps)
        n_call = sum(r['callable'] for r in reads)
        for r in reads:
            if r['callable']:
                gaps_by_uid.setdefault(r['uid'], []).extend((g['g0'], g['g1']) for g in r['gaps'])
        nfr = dict(id=nfr_id, start=nreg[0], end=nreg[1], source=reg['source'], profile_peak=reg.get('peak'),
                   reads=len(reads), callable=n_call, censored=len(reads) - n_call,
                   overflow_reads=sum(1 for r in reads if r['overflow']), mode=opt.mode, variants=[], configurations=[],
                   depth_states=[], status='ok', message='')
        for r in reads:
            molecules.setdefault(r['uid'], dict(read_name=r['read_name'], dataset=r['dataset'], strand=r['strand'],
                                                members=r['members'], nfr={}))
            molecules[r['uid']]['nfr'][nfr_id] = dict(status='callable' if r['callable'] else 'censored',
                                                      gaps=[[g['g0'], g['g1']] for g in r['gaps']])
        if n_call == 0:
            nfr.update(status='not_analysable', message='No read has a nucleosome on both sides of this region.')
            nfrs.append(nfr); continue
        if opt.mode == 'depth':
            cr, widest, states = V.depth_states(reads)
            for s in states:
                prev = float(s['open'].mean())
                nfr['depth_states'].append(dict(id=f"{nfr_id}:{s['name']}", name=s['name'], depth=s['depth'], threshold=s['threshold'],
                                                prevalence=_r(prev), n=len(cr)))
                elements.append(dict(id=f"{nfr_id}:{s['name']}", kind='nfr', subtype='depth', nfr=nfr_id, start=float(nreg[0]),
                                     end=float(nreg[1]), label=f"{nfr_id} {s['depth']} ({s['name']} bp)", prevalence=prev,
                                     state={r['uid']: int(o) for r, o in zip(cr, s['open'])}, channel=None))
            for r, w in zip(cr, widest):
                molecules[r['uid']]['nfr'][nfr_id].update(widest=int(w), map='closed' if not r['gaps'] else f'widest {int(w)} bp')
            nfrs.append(nfr); continue

        def ps_progress(k, kmax, _id=nfr_id):
            _report(progress, 'nfr', f'{_id}: prediction strength k={k}/{kmax}', completed=k, total=kmax)
        vs, diag, es = V.discover(reads, nreg, opt, progress=ps_progress)
        nfr.update(k=diag['k'], ps_curve=[[k, ps] for k, ps in diag['ps_curve']], gaps=diag['gaps'],
                   candidates=diag['candidates'], merges=[list(m) for m in diag['merges']], dropped=diag['dropped'])
        _report(progress, 'nfr', f'{nfr_id}: prevalence (EM, {opt.bootstrap} bootstrap replicates)')
        q = V.quantify(reads, vs, es, opt)
        rob = V.robust(reads, nreg, vs, opt, n=opt.robust) if (vs and opt.robust) else None
        Pv = q['Pv']
        for j, v in enumerate(vs):
            nfr['variants'].append(_variant_row(nfr_id, v, _by_channel(q['reads'], Pv, j), None if rob is None else rob[j]))
        nfr['closed'] = _r(q['closed']); nfr['other'] = dict(em=_r(q['other_prev']), strict=_r(q['other_strict']),
                                                           ci=[_r(q['other_ci'][0]), _r(q['other_ci'][1])] if q['other_ci'] else None)
        order = np.argsort(-q['weights'], kind='stable')
        nfr['configurations'] = [dict(label=q['labels'][i], display_label=q['labels'][i], weight=_r(q['weights'][i]),
                                      internal_footprint=None) for i in order if q['weights'][i] >= 5e-4]
        if not vs:
            nfr.update(status='no_variants', message='No supported variant at this stringency (all gaps are "other shape" or the NFR is closed).')
        # per-read membership
        map_idx = np.argmax(q['P'], 1) if len(q['P']) else []
        for i, r in enumerate(q['reads']):
            rec = molecules[r['uid']]['nfr'][nfr_id]
            rec.update(map=q['labels'][int(map_idx[i])], map_posterior=_r(q['P'][i, int(map_idx[i])]),
                       p={vs[j]['name']: _r(Pv[i, j]) for j in range(len(vs))}, p_other=_r(Pv[i, len(vs)]))
        # elements
        if opt.elements == 'variants':
            for j, v in enumerate(vs):
                elements.append(dict(id=f"{nfr_id}:{v['name']}", kind='nfr', subtype='variant', nfr=nfr_id, start=v['L'], end=v['R'],
                                     label=f"{nfr_id} {v['name']} {v['relation']}", prevalence=v['prevalence'],
                                     state={r['uid']: int(Pv[i, j] >= opt.open_threshold) for i, r in enumerate(q['reads'])}, channel=None))
        else:
            elements.append(dict(id=f'{nfr_id}:open', kind='nfr', subtype='nfr', nfr=nfr_id, start=float(nreg[0]), end=float(nreg[1]),
                                 label=f'{nfr_id} any opening', prevalence=float(np.mean([bool(r['gaps']) for r in q['reads']])),
                                 state={r['uid']: int(bool(r['gaps'])) for r in q['reads']}, channel=None))
        nfrs.append(nfr)
    # footprint classes (lattice recaller) overlapping the window
    tf_els = []
    class_join = None
    if classes:
        uids = {u['unit_id'] for u in units}
        tf_els = [c for c in classes if c['end'] > region['start'] and c['start'] < region['end']]
        class_units = set().union(*[set(c['state']) for c in tf_els]) if tf_els else set()
        class_join = dict(classes=len(tf_els), class_molecules=len(class_units), joined_molecules=len(class_units & uids),
                          molecules=len(uids))
        if class_units and not class_units & uids:
            warnings.append('No footprint-class molecule matches a molecule of this run (unit_id join). Classes come from a run on '
                            'different datasets or a different dataset order; variant x class pairs are not tested.')
        for nfr in nfrs:
            _internal_footprints(nfr, tf_els, opt)
    all_els = sorted(elements + tf_els, key=lambda e: (e['start'], e['end'], e['id']))
    # co-accessibility
    _report(progress, 'coaccess', 'Per-read openness')
    cov = G.read_covariates(units, region['start'], region['end'])
    cl = None
    if clusters is not None:
        cl = clusters
    elif opt.within_clusters and len(all_els) >= 2:
        cl = _masked_clusters(units, cov, all_els, opt.within_clusters, region)

    def pair_progress(done, total):
        _report(progress, 'coaccess', f'Pair tests {done}/{total}', completed=done, total=total)
    pairs, skipped = C.pair_table(all_els, gaps_by_uid, cov, scope=opt.pairs, clusters=cl, n_perm=opt.n_perm,
                                  min_reads=opt.min_spanning, min_marginal=opt.min_marginal, per_bin=opt.per_bin,
                                  q_max=opt.q_max, pad_bp=opt.openness_pad_bp, progress=pair_progress) if len(all_els) >= 2 else ([], [])
    # combinations
    combo = None
    ids = {e['id']: e for e in all_els}
    chosen = [ids[i] for i in opt.combo_elements if i in ids] if opt.combo_elements else _auto_combo(all_els, opt)
    if opt.combo_elements and len(chosen) != len(opt.combo_elements):
        warnings.append('combo_elements: ' + ', '.join(sorted(set(opt.combo_elements) - set(ids))) + ' not found')
    if len(chosen) >= 3:
        _report(progress, 'combinations', f'Combination patterns of {len(chosen)} elements')
        try:
            combo = C.combinations(chosen, gaps_by_uid, n_samples=opt.combo_samples, q_max=opt.q_max)
        except ValueError as error:
            warnings.append(str(error))
    result = dict(schema=SCHEMA, experimental=True, note=EXPERIMENTAL_NOTE, region=region,
                  datasets=[dict(dataset_id=s['dataset_id'], chemistry=s.get('chemistry'), molecules=len(s['units'])) for s in payload['strata']],
                  parameters=opt.as_dict(), nfrs=nfrs,
                  elements=[{k: (_r(v, 1) if k in ('start', 'end') else _r(v) if k == 'prevalence' else v)
                             for k, v in e.items() if k not in ('state', 'channel')} | dict(n=len(e['state'])) for e in all_els],
                  pairs=[_pair_out(p) for p in pairs], family=dict(tested=len(pairs), scope=opt.pairs, skipped=skipped,
                                                                   bh='all tested pairs of this run'),
                  combos=combo, class_join=class_join,
                  within_clusters=dict(mode='labels' if clusters is not None else 'masked_kmeans', k=opt.within_clusters,
                                       molecules=getattr(cl, 'n', None)) if cl is not None else None,
                  warnings=warnings)
    result['molecules'] = molecules
    if output_dir is not None:
        write_outputs(result, Path(output_dir), payload, inputs)
    return result


def _pair_out(p):
    out = {}
    for k, v in p.items():
        if isinstance(v, float):
            # p and q values keep 4 significant digits (they span many orders of magnitude)
            out[k] = (float(f'{v:.4g}') if math.isfinite(v) else None) if k in ('fisher_p', 'p_exact', 'q') else _r(v, 4)
        else:
            out[k] = v
    return out


# ------------------------------------------------------------------ outputs
VARIANT_FIELDS = ['nfr', 'nfr_start', 'nfr_end', 'id', 'name', 'L', 'R', 'Lsd', 'Rsd', 'width', 'depth', 'relation', 'prevalence_strict',
                  'prevalence_em', 'ci_lo', 'ci_hi', 'only', 'with_ref', 'gain_nats', 'stability', 'exp_reads', 'robust',
                  'callable', 'reads', 'by_channel']
CONFIG_FIELDS = ['nfr', 'label', 'weight', 'internal_footprint', 'display_label']
MOLECULE_FIELDS = ['nfr', 'unit_id', 'read_name', 'dataset', 'strand', 'status', 'map', 'map_posterior', 'p', 'p_other', 'gaps']
PAIR_FIELDS = ['a', 'b', 'kind_a', 'kind_b', 'n', 'shared', 'n11', 'n10', 'n01', 'n00', 'log2or', 'lo', 'hi', 'fisher_p', 'mh', 'mh_lo',
               'mh_hi', 'p_exact', 'q', 'class', 'adjust', 'null_median', 'null_lo', 'null_hi', 'exp11', 'nested', 'dist', 'cluster', 'cluster_n']
COMBO_FIELDS = ['pattern', 'elements', 'obs', 'exp_indep', 'null_med', 'null_lo', 'null_hi', 'p', 'q', 'pinned', 'n']


def _cell(v):
    if v is None:
        return ''
    if isinstance(v, bool):
        return 'True' if v else 'False'
    if isinstance(v, float):
        return '' if math.isnan(v) else repr(v)
    if isinstance(v, (list, tuple)):
        return ','.join(map(str, v))
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True, separators=(',', ':'))
    return str(v)


def _tsv(rows, fields):
    buf = io.StringIO()
    w = csv.writer(buf, delimiter='\t', lineterminator='\n')
    w.writerow(fields)
    for r in rows:
        w.writerow([_cell(r.get(f)) for f in fields])
    return buf.getvalue()


def _gzip_bytes(text):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0, filename='') as fh:
        fh.write(text.encode())
    return buf.getvalue()


def variant_rows(result):
    rows = []
    for n in result['nfrs']:
        for v in n['variants']:
            rows.append(dict(nfr=n['id'], nfr_start=n['start'], nfr_end=n['end'], id=v['id'], name=v['name'], L=v['L'], R=v['R'],
                             Lsd=v['Lsd'], Rsd=v['Rsd'], width=v['width'], depth=v['depth'], relation=v['relation'],
                             prevalence_strict=v['strict'], prevalence_em=v['prevalence'], ci_lo=(v['ci'] or [None])[0],
                             ci_hi=(v['ci'] or [None, None])[1], only=v['only'], with_ref=v['with_ref'], gain_nats=v['gain_nats'],
                             stability=v['stability'], exp_reads=v['exp_reads'], robust=v['robust'], callable=n['callable'],
                             reads=n['reads'], by_channel=v['by_channel']))
        for s in n.get('depth_states') or ():
            rows.append(dict(nfr=n['id'], nfr_start=n['start'], nfr_end=n['end'], id=s['id'], name=s['name'], depth=s['depth'],
                             relation='depth state', prevalence_strict=s['prevalence'], prevalence_em=s['prevalence'],
                             callable=n['callable'], reads=n['reads']))
    return rows


def write_outputs(result, out, payload=None, inputs=None):
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    files['variants.tsv'] = _tsv(variant_rows(result), VARIANT_FIELDS).encode()
    cfg_rows = [dict(nfr=n['id'], **c) for n in result['nfrs'] for c in n['configurations']]
    files['configurations.tsv'] = _tsv(cfg_rows, CONFIG_FIELDS).encode()
    mol_rows = []
    for uid in sorted(result['molecules']):
        m = result['molecules'][uid]
        for nid in sorted(m['nfr']):
            rec = m['nfr'][nid]
            mol_rows.append(dict(nfr=nid, unit_id=uid, read_name=m['read_name'], dataset=m['dataset'], strand=m['strand'],
                                 status=rec['status'], map=rec.get('map'), map_posterior=rec.get('map_posterior'),
                                 p=';'.join(f'{k}={v}' for k, v in (rec.get('p') or {}).items()), p_other=rec.get('p_other'),
                                 gaps=';'.join(f'{a}-{b}' for a, b in rec.get('gaps') or ())))
    files['molecules.tsv.gz'] = _gzip_bytes(_tsv(mol_rows, MOLECULE_FIELDS))
    prs = [dict(p, n11=p['table'][0], n10=p['table'][1], n01=p['table'][2], n00=p['table'][3]) for p in result['pairs']]
    files['coaccess.tsv'] = _tsv(prs, PAIR_FIELDS).encode()
    combo = result.get('combos')
    crow = [dict(o, elements='+'.join(combo['elements']), n=combo['n']) for o in (combo or {}).get('patterns', [])]
    files['combos.tsv'] = _tsv(crow, COMBO_FIELDS).encode()
    slim = {k: v for k, v in result.items() if k != 'molecules'}
    files['result.json'] = (json.dumps(slim, sort_keys=True, indent=1) + '\n').encode()
    for name, data in files.items():
        (out/name).write_bytes(data)
    manifest = dict(schema=SCHEMA, experimental=True, note=EXPERIMENTAL_NOTE, tool='fiberhmm-nfr', fiberhmm=_version(),
                    region=result['region'], parameters=result['parameters'], datasets=result['datasets'],
                    inputs=_inputs(payload, inputs), nfr_regions=[dict(id=n['id'], start=n['start'], end=n['end'], source=n['source'])
                                                               for n in result['nfrs']],
                    outputs={name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())},
                    definitions=dict(nfr='gap between consecutive >= 90-bp nucleosome calls (Timer preprint); internal factor-sized '
                                         'protections do not split it',
                                     callable='nucleosome-bounded read coverage spans the NFR region (both edges observed)',
                                     prevalence='range: strict (membership >= 0.9) to EM; bootstrap interval conditional on the catalogue',
                                     coaccess='spanning reads only; Timer shared rule; exact test stratified by openness x channel; '
                                              'Mantel-Haenszel OR; BH over all tested pairs'))
    (out/'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=1) + '\n')
    return files


def _version():
    from fiberhmm import __version__
    info = dict(version=__version__)
    root = Path(__file__).resolve().parents[3]
    try:
        import subprocess
        sha = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=5)
        if sha.returncode == 0:
            info['git'] = sha.stdout.strip()
    except Exception:
        pass
    return info


def _inputs(payload, inputs):
    out = dict(inputs or {})
    if payload is not None and payload.get('input_files'):
        out.setdefault('files', payload['input_files'])
    if payload is not None:
        out.setdefault('chemistry', {s['dataset_id']: s.get('chemistry') for s in payload['strata']})
    return out
