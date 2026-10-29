"""Association calibration on confounded nulls and power on planted effects.

usage: calibrate_assoc.py MODE N_NULL N_POWER OUT.json [--workers 2] [--scenarios a,b]
Every scenario draws its nuisance settings per replicate (confounding strength, depth, latent molecule factors,
sizes), so a pass covers a family of confounded nulls, not one point. Gate: FPR at 0.05 within 0.04-0.06 and a
KS test of the p-values against Uniform(0, 1) (permutation p-values are discrete at 1/(N + 1)).
"""
import json
import hashlib
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import kstest

from fiberhmm.assoc.simulate import simulate_m1, simulate_m2, simulate_m3


def _rs(seed):
    return np.random.default_rng(10_000_019 * (seed + 1))


# ------------------------------------------------------------------ M1
def m1(seed, scen):
    from fiberhmm.assoc.cooccur import cooccurrence
    r = _rs(seed)
    conf = float(r.uniform(0.5, 2.0)); cov = float(r.choice([15, 30, 60])); lat = float(r.choice([0.5, 0.8, 1.5]))
    na, nb = int(r.integers(40, 81)), int(r.integers(60, 121))
    kw = dict(confound=conf, coverage=cov, latent=lat, n_a=na, n_b=nb)
    ck = dict(n_boot=0, per_pair=False, effect=False)
    if scen == 'null':
        sim = simulate_m1(seed, **kw); res = cooccurrence(sim['a'], sim['b'], sim['molecules'], **ck)
        return {'p': res['pooled']['p']}
    if scen == 'null_pairs':
        sim = simulate_m1(seed, **kw)
        res = cooccurrence(sim['a'], sim['b'], sim['molecules'], n_boot=0, per_pair=True, min_molecules=10)
        return {'p_pairs': [x['p_exact'] for x in res.get('pairs', [])]}
    if scen == 'null_strata':
        sim = simulate_m1(seed, **kw); res = cooccurrence(sim['a'], sim['b'], sim['molecules'], test='strata', **ck)
        return {'p': res['pooled']['p']}
    if scen == 'control_unadjusted':
        sim = simulate_m1(seed, **kw)
        res = cooccurrence(sim['a'], sim['b'], sim['molecules'], basis='none', **ck)
        return {'p': res['pooled']['p']}
    if scen == 'null_small':
        sim = simulate_m1(seed, confound=conf, coverage=15, latent=lat, chrom_len=150_000, n_a=20, n_b=30)
        res = cooccurrence(sim['a'], sim['b'], sim['molecules'], **ck)
        return {'p': res['pooled']['p'], 'signflip': 't(' in res['pooled'].get('test', ''),
                'n_mol': res['pooled'].get('n_molecules')}
    if scen == 'null_interaction':
        e = float(r.uniform(0.3, 0.9))
        sim = simulate_m1(seed, n_datasets=2, effect=e, **kw)
        res = cooccurrence(sim['a'], sim['b'], sim['molecules'], group='dataset', n_boot=300, per_pair=False,
                           seed=seed)
        return {'p': res['interaction']['p'], 'p_main': res['pooled']['p']}
    if scen == 'null_haplotype':
        e = float(r.uniform(0.3, 0.9))
        sim = simulate_m1(seed, unphased_fraction=0.2, effect=e, **kw)
        res = cooccurrence(sim['a'], sim['b'], sim['molecules'], group='haplotype', group_kind='haplotype',
                           n_boot=300, per_pair=False, seed=seed)
        return {'p': res['interaction']['p'], 'unphased_reported': 'unphased' in res}
    if scen == 'null_ci':
        sim = simulate_m1(seed, **kw)
        res = cooccurrence(sim['a'], sim['b'], sim['molecules'], n_boot=300, per_pair=False, seed=seed)
        b = res['pooled'].get('bootstrap', {})
        return {'covers0': bool(b.get('lo', np.nan) <= 0 <= b.get('hi', np.nan))}
    if scen.startswith('power') and not scen.startswith('power_interaction'):
        e = float(scen.split(':')[1])
        sim = simulate_m1(seed, effect=e, confound=1.0, coverage=30, latent=0.8)
        res = cooccurrence(sim['a'], sim['b'], sim['molecules'], **ck)
        res2 = cooccurrence(sim['a'], sim['b'], sim['molecules'], test='strata', **ck)
        return {'p': res['pooled']['p'], 'p_strata': res2['pooled']['p']}
    if scen.startswith('power_interaction'):
        e = float(scen.split(':')[1])
        sim = simulate_m1(seed, n_datasets=2, effect_by_dataset={0: 0.0, 1: e}, confound=1.0, coverage=30)
        res = cooccurrence(sim['a'], sim['b'], sim['molecules'], group='dataset', n_boot=300, per_pair=False,
                           seed=seed)
        return {'p': res['interaction']['p']}
    raise ValueError(scen)


# ------------------------------------------------------------------ M2
def m2(seed, scen):
    from fiberhmm.assoc.condition import condition
    r = _rs(seed)
    conf = float(r.uniform(0.5, 2.0)); cov = float(r.choice([15, 25, 40])); na = int(r.integers(40, 101))
    kw = dict(confound=conf, coverage=cov, n_anchors=na)
    ck = dict(n_boot=0, per_anchor=False, seed=seed)

    def run(sim, **extra):
        if extra.get('group'):
            sim['app'].stats = sim['app'].stats[:, :1, :]
            sim['app'].mask &= 1
            sim['bins'] = sim['bins'][:1]
        return condition(sim['app'], sim['loci'], sim['metric'], bins=sim['bins'], **dict(ck, **extra))
    def contrast(res):
        inter = res['bins'][0].get('interaction', {})
        difference = next(iter(inter.get('differences', [])), {})
        return {'p': inter.get('p', np.nan), 'statistic': inter.get('statistic'),
                'clusters': inter.get('bootstrap_support', {}).get('clusters'),
                'difference': difference.get('difference'), 'se': difference.get('se'),
                'df': difference.get('df'),
                'group_clusters': inter.get('bootstrap_support', {}).get('group_clusters'),
                'n_sets': {g: values.get('n_sets') for g, values in res['bins'][0].get('groups', {}).items()},
                'unphased_reported': 'unphased' in res}

    if scen == 'null_anchor':
        res = run(simulate_m2(seed, self_seed=0.0, **kw), n_role=0)
        return {'p': res['bins'][0]['p_within_anchor'], 'p_bins': [b['p_within_anchor'] for b in res['bins']]}
    if scen == 'null_site':
        res = run(simulate_m2(seed, self_seed=float(r.uniform(0, 0.6)), **kw), n_perm=0)
        return {'p': res['bins'][0].get('p_site', np.nan), 'p_bins': [b.get('p_site', np.nan) for b in res['bins']]}
    if scen == 'control_unbalanced':
        res = run(simulate_m2(seed, self_seed=0.0, **kw), n_role=0, balance=False)
        return {'p': res['bins'][0]['p_within_anchor']}
    if scen == 'null_interaction':
        s = float(r.uniform(0.2, 0.5))
        sim = simulate_m2(seed, self_seed=0.3, site_effect=s, n_datasets=2, group_effect={0: s, 1: s}, **kw)
        res = run(sim, n_perm=0, n_role=0, n_boot=1000, group='dataset')
        return contrast(res)
    if scen == 'null_haplotype':
        s = float(r.uniform(0.2, 0.5))
        sim = simulate_m2(seed, self_seed=0.3, site_effect=s, unphased_fraction=0.2, **kw)
        res = run(sim, n_perm=0, n_role=0, n_boot=1000, group='haplotype', group_kind='haplotype')
        return contrast(res)
    if scen == 'null_ci':
        res = run(simulate_m2(seed, self_seed=0.3, **kw), n_perm=0, n_role=0, n_boot=300)
        b = res['bins'][0].get('delta_site_ci', {})
        return {'covers0': bool(b.get('lo', np.nan) <= 0 <= b.get('hi', np.nan))}
    if scen.startswith(('power_interaction:', 'power_haplotype:')):
        e = float(scen.split(':')[1])
        haplotype = scen.startswith('power_haplotype:')
        sim = simulate_m2(seed, self_seed=0.3, site_effect=0.0, confound=1.0, coverage=25, n_anchors=80,
                          n_datasets=1 if haplotype else 2, unphased_fraction=0.2 if haplotype else 0.0,
                          group_effect=None if haplotype else {0: 0.0, 1: e})
        if haplotype:
            app = sim['app']
            planted = (app.labels['haplotype'] == 2) & sim['loci'].is_anchor[app.locus] & (app.state == 1)
            measured = ((app.mask[:, None] >> np.arange(app.n_bins)[None, :]) & 1)
            app.stats[:, :, 0] += e * planted[:, None] * measured
        res = run(sim, n_perm=0, n_role=0, n_boot=1000,
                  group='haplotype' if haplotype else 'dataset', group_kind='haplotype' if haplotype else 'label')
        return contrast(res)
    if scen.startswith('power_site'):
        e = float(scen.split(':')[1])
        res = run(simulate_m2(seed, self_seed=0.3, site_effect=e, confound=1.0, coverage=25, n_anchors=80),
                  n_perm=0)
        return {'p': res['bins'][0].get('p_site', np.nan)}
    if scen.startswith('power_anchor'):
        e = float(scen.split(':')[1])
        res = run(simulate_m2(seed, self_seed=e, confound=1.0, coverage=25, n_anchors=80), n_role=0)
        return {'p': res['bins'][0]['p_within_anchor']}
    raise ValueError(scen)


# ------------------------------------------------------------------ M3
def m3(seed, scen):
    from fiberhmm.assoc.rate import edge_association, rate_ratio
    r = _rs(seed)
    conf = float(r.uniform(0.5, 2.0)); nm = int(r.integers(150, 600)); cl = bool(r.random() < 0.7)
    kw = dict(confound=conf, n_molecules=nm, clustered=cl)
    if scen == 'null':
        res = rate_ratio(simulate_m3(seed, **kw), n_boot=0, seed=seed)
        return {'p': res['p_shift']}
    if scen == 'null_edge':
        res = edge_association(simulate_m3(seed, **kw), seed=seed)
        return {'p': res['p_two_sided']}
    if scen == 'null_interaction':
        b = float(r.uniform(-0.8, -0.2))
        res = rate_ratio(simulate_m3(seed, beta=b, n_datasets=2, **kw), n_shift=0, n_boot=300, group='dataset',
                         seed=seed)
        return {'p': res['interaction']['p']}
    if scen == 'null_haplotype':
        b = float(r.uniform(-0.8, -0.2))
        res = rate_ratio(simulate_m3(seed, beta=b, unphased_fraction=0.2, **kw), n_shift=0, n_boot=300,
                         group='haplotype', group_kind='haplotype', seed=seed)
        return {'p': res['interaction']['p']}
    if scen == 'null_ci':
        res = rate_ratio(simulate_m3(seed, **kw), n_shift=0, n_boot=300, seed=seed)
        b = res['log_rate_ratio_ci']
        return {'covers0': bool(b['lo'] <= 0 <= b['hi'])}
    if scen == 'control_crude':
        # crude (unconditional) rate ratio Poisson test ignoring molecules: shows the confounding
        from scipy.stats import binomtest
        d = simulate_m3(seed, **kw)
        res = rate_ratio(d, n_shift=0, n_boot=0)
        n = int(res['O_in'] + res['O_out'])
        p0 = res['E_in'] / (res['E_in'] + res['E_out'])
        return {'p': binomtest(int(res['O_in']), n, p0).pvalue}
    if scen.startswith('power_interaction'):
        e = float(scen.split(':')[1])
        res = rate_ratio(simulate_m3(seed, beta_by_dataset={0: 0.0, 1: e}, n_datasets=2, n_molecules=400),
                         n_shift=0, n_boot=300, group='dataset', seed=seed)
        return {'p': res['interaction']['p']}
    if scen.startswith('power'):
        e = float(scen.split(':')[1])
        res = rate_ratio(simulate_m3(seed, beta=e, n_molecules=300), n_boot=0, seed=seed)
        return {'p': res['p_shift']}
    raise ValueError(scen)


def m4(seed, scen):
    from fiberhmm.assoc.overlap import overlap_enrichment, simulate_overlap
    if scen == 'null':
        r = np.random.default_rng(900_000 + seed)
        na = int(r.integers(200, 2001))
        nb = int(np.exp(r.uniform(np.log(500), np.log(10_000))))
        sim = simulate_overlap(10_000 + seed, n_a=na, n_b=nb, n_groups=2)
    elif scen.startswith('power:'):
        effect = float(scen.split(':')[1])
        seed -= 100_000
        sim = simulate_overlap(20_000 + seed + int(effect * 1e6), n_a=1000, n_b=2000, planted=effect)
    else:
        raise ValueError(scen)
    res = overlap_enrichment(sim['a'], sim['b'], sim['territory'], sim['strata'], n_relocations=1000, seed=seed)
    return {'p': res.get('global_p', np.nan), 'status': res['status']}


MODES = {'M1': m1, 'M2': m2, 'M3': m3, 'M4': m4}
NULLS = {'M1': ['null', 'null_strata', 'null_small', 'null_interaction', 'null_haplotype', 'null_pairs',
                'control_unadjusted', 'null_ci'],
         'M2': ['null_anchor', 'null_site', 'null_interaction', 'null_haplotype', 'control_unbalanced', 'null_ci'],
         'M3': ['null', 'null_edge', 'null_interaction', 'null_haplotype', 'control_crude', 'null_ci'],
         'M4': ['null']}
POWER = {'M1': ['power:0.3', 'power:0.5', 'power:0.8', 'power_interaction:0.8'],
         'M2': ['power_site:0.2', 'power_site:0.4', 'power_site:0.6', 'power_anchor:0.15', 'power_anchor:0.3',
                'power_interaction:0.4', 'power_interaction:0.8', 'power_haplotype:0.4', 'power_haplotype:0.8'],
         'M3': ['power:-0.15', 'power:-0.3', 'power:-0.5', 'power_interaction:-0.5'],
         'M4': ['power:0.1', 'power:0.2', 'power:0.4']}
# scenarios with heavier per-replicate cost run with fewer replicates
LIGHT = {'null_ci', 'null_pairs', 'control_unadjusted', 'control_unbalanced', 'control_crude', 'null_strata'}


def job(args):
    mode, scen, seed = args
    t = time.time()
    out = MODES[mode](seed, scen)
    out['t'] = time.time() - t
    return mode, scen, seed, out


def summarise(vals, key='p'):
    p = np.array([v.get(key, np.nan) for v in vals], float)
    p = p[np.isfinite(p)]
    if not len(p):
        return {'n': 0}
    f = float(np.mean(p < 0.05))
    ks = kstest(p, 'uniform')
    return {'n': int(len(p)), 'fpr_05': f, 'fpr_se': float(np.sqrt(f * (1 - f) / len(p))),
            'fpr_01': float(np.mean(p < 0.01)), 'ks_D': float(ks.statistic), 'ks_p': float(ks.pvalue),
            'within_4_6': bool(0.04 <= f <= 0.06)}


def main():
    mode, n_null, n_power, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    workers = 2
    scen_sel = None
    seed_start = 0
    for i, a in enumerate(sys.argv):
        if a == '--workers':
            workers = int(sys.argv[i + 1])
        if a == '--seed-start':
            seed_start = int(sys.argv[i + 1])
        if a == '--scenarios':
            scen_sel = sys.argv[i + 1].split(',')
    jobs = []
    for s in NULLS[mode]:
        if scen_sel and s not in scen_sel:
            continue
        n = n_null if s not in LIGHT else max(200, n_null // 5)
        jobs += [(mode, s, seed) for seed in range(seed_start, seed_start + n)]
    for s in POWER[mode]:
        if scen_sel and s not in scen_sel:
            continue
        jobs += [(mode, s, 100_000 + seed) for seed in range(seed_start, seed_start + n_power)]
    if not 1 <= workers <= 4:
        raise ValueError('workers must be between 1 and 4')
    t0 = time.time()
    res = []
    with Pool(workers) as pool, open(out + '.jsonl', 'w') as raw:
        for result in pool.imap_unordered(job, jobs, chunksize=2):
            res.append(result)
            raw.write(json.dumps(result) + '\n')
            raw.flush()
            if len(res) % 100 == 0:
                print(f'{len(res)}/{len(jobs)} done ({time.time() - t0:.0f}s)', flush=True)
    res.sort(key=lambda x: (x[1], x[2]))
    by = {}
    for m, s, seed, o in res:
        by.setdefault(s, []).append(o)
    summary = {'mode': mode, 'n_null': n_null, 'n_power': n_power, 'workers': workers, 'seed_start': seed_start,
               'wall_clock_s': time.time() - t0, 'scenarios': {}}
    root = Path(__file__).resolve().parents[2]
    summary['source_sha256'] = {name: hashlib.sha256((root / 'fiberhmm' / 'assoc' / name).read_bytes()).hexdigest()
                                for name in ('condition.py', 'groups.py', 'resample.py', 'simulate.py')}
    for s, vals in by.items():
        row = {'mean_replicate_s': float(np.mean([v['t'] for v in vals]))}
        if s == 'null_pairs':
            ps = [p for v in vals for p in v['p_pairs']]
            row.update(summarise([{'p': p} for p in ps]))
            row['note'] = 'per-pair exact conditional tests pooled over replicates (discrete, conservative)'
        elif s == 'null_ci':
            c = np.array([v['covers0'] for v in vals])
            row.update({'n': int(len(c)), 'coverage_95': float(c.mean()), 'coverage_se': float(np.sqrt(c.mean() * (1 - c.mean()) / len(c)))})
        else:
            row.update(summarise(vals))
            if 'statistic' in vals[0]:
                row['by_statistic'] = {key: summarise([v for v in vals if v.get('statistic') == key])
                                       for key in sorted({v['statistic'] for v in vals if v.get('statistic')})}
            if s.startswith('power') and 'p_strata' in vals[0]:
                row['power_strata_test'] = summarise(vals, 'p_strata').get('fpr_05')
            if 'p_bins' in vals[0]:
                pb = np.array([v['p_bins'] for v in vals], float)
                row['fpr_05_by_bin'] = [float(np.mean(pb[np.isfinite(pb[:, k]), k] < 0.05)) for k in range(pb.shape[1])]
            if 'signflip' in vals[0]:
                row['share_signflip'] = float(np.mean([v['signflip'] for v in vals]))
            if 'unphased_reported' in vals[0]:
                row['unphased_reported'] = float(np.mean([v['unphased_reported'] for v in vals]))
        if s.startswith('power'):
            row['power_05'] = row.get('fpr_05')
        summary['scenarios'][s] = row
    with open(out, 'w') as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    for s, row in summary['scenarios'].items():
        print(s, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
    print('wall', summary['wall_clock_s'])


if __name__ == '__main__':
    main()
