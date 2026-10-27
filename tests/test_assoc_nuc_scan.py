"""fiberhmm.assoc nucleosome analyses (NUC 3.8/3.9) and `fiberhmm-scan assoc` on scan directories.

Synthetic data only: planted nucleosome arrays and barriers (fiberhmm.assoc.simulate.simulate_nuc), a hand-built
table-mode scan directory, and a nuc scan of a synthetic BAM.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pysam
import pytest

from nuc_synth import array_molecule, flat_calibration, header, make_record, molecule_sequence

from fiberhmm.assoc import nucleosome as N
from fiberhmm.assoc.simulate import simulate_nuc


# ---------------------------------------------------------------- state conditioning (NUC 3.8)
def test_nuc_state_rules():
    from fiberhmm.nuc.records import make_molecule
    m = make_molecule(2000, nucs=[(100, 247), (700, 847)], msps=[(247, 700)], name='m/1/ccs')
    assert N.nuc_state(m, 400) == '1'          # an MSP of 453 bp covers 400 +/- 30
    assert N.nuc_state(m, 150) == '0'          # inside a nucleosome
    assert N.nuc_state(m, 1500) == '.'


def test_condition_stats_finds_a_planted_anchor_frame_site_effect_and_not_a_null():
    sim = simulate_nuc(2, site_effect=0.7, n_anchors=40)
    res = N.condition_stats(sim['molecules'], sim['anchors'], backgrounds=sim['candidates'], n_perm=200,
                            n_role=400, n_boot=0, metrics=('R',), per_anchor=False, seed=1)
    a = res['frames']['anchor']['metrics']['R']['bins'][-1]
    assert a['p_site'] < 0.01 and a['delta_site'] > 0.1 and a['site_state1'] > 0.1
    assert res['frames']['molecule']['appearances']['appearances'] > 0
    null = simulate_nuc(3, site_effect=0.0, n_anchors=40)
    r0 = N.condition_stats(null['molecules'], null['anchors'], backgrounds=null['candidates'], n_perm=0, n_role=400,
                           n_boot=0, metrics=('R',), per_anchor=False, seed=1)
    assert r0['frames']['anchor']['metrics']['R']['bins'][-1]['p_site'] > 0.01


def test_resultant_metric_null_adjustment_and_oscillation_fit():
    s = np.array([3.0, 4.0, 0.0, 0.0, 25.0, 0.0, 10.0])
    assert N.resultant_metric()(s) == pytest.approx(0.5)
    assert N.resultant_metric(True)(s) == pytest.approx(0.5 - np.sqrt(np.pi / 4 * 25) / 10)
    osc = N.oscillation_metric(150, 150 + 5 * 190, 190.0, ell=1e9)
    xc = (osc.edges[:-1] + osc.edges[1:]) / 2
    counts = 10 + 3 * np.cos(2 * np.pi * xc / 190.0)
    assert osc(counts) == pytest.approx(0.3, abs=1e-6)


def test_pooled_msp_anchors_and_held_out_cross_fit():
    sim = simulate_nuc(4, site_effect=0.5, n_anchors=30, coverage=30)
    anchors = N.pooled_msp_anchors(sim['molecules'], min_frac=0.2)
    assert anchors and all(b[1] - a[1] >= 3000 for a, b in zip(anchors, anchors[1:]))
    out = N.held_out(sim['molecules'], lambda ms: N.pooled_msp_anchors(ms, min_frac=0.2),
                     lambda ms, an: N.condition_stats(ms, an, backgrounds='random', k=2, n_perm=50, n_role=50,
                                                      n_boot=0, metrics=('R',), per_anchor=False,
                                                      return_null=True, frames=('anchor',)))
    d0, d1 = out['directions']
    assert d0['tested_on'] == 'half 1' and d1['tested_on'] == 'half 0'
    rows = out['cross_fit']['anchor/R']
    assert rows and 'delta_anchor' in rows[-1] and 0 < rows[-1]['p_within_anchor'] <= 1
    keys = [N.molecule_key(m) for m in sim['molecules']]
    assert len(set(keys)) == len(keys)
    halves = {N.hash_half(m) for m in sim['molecules']}
    assert halves == {0, 1}


def test_reverse_association_cmh_and_conditional_logistic_agree():
    rng = np.random.default_rng(5)
    rows = []
    for locus in range(60):
        base = rng.normal()
        for m in range(30):
            lam = rng.normal()
            x = int(rng.random() < 1 / (1 + np.exp(-(0.5 * lam))))
            y = int(rng.random() < 1 / (1 + np.exp(-(base + 1.0 * x + 0.7 * lam))))
            rows.append((locus, locus * 100 + m, y, x, lam))
    loc, mol, y, x, lam = map(np.array, zip(*rows))
    res = N.reverse_association(loc, mol, y, x, lam, n_boot=200)
    assert 0.6 < res['clogit']['log_or'] < 1.5 and res['clogit']['p'] < 1e-3
    assert res['cmh']['or'] > 1.5 and res['cmh']['lo'] < res['cmh']['or'] < res['cmh']['hi']


def test_association_stats_on_informative_sites_detects_tf_depletion_in_arrays():
    rng = np.random.default_rng(6)
    cal = flat_calibration(nrl=190.0)
    mols = []
    for i in range(80):
        m = array_molecule([('irr', 8, 0), ('reg', 14, 190), ('irr', 8, 0), ('reg', 14, 190), ('irr', 8, 0)],
                           rng, name=f'm/{i}/ccs', chrom='chrS', ref_start=10_000 * i, ref_end=10_000 * i + 9000)
        mols.append(m)
    res0 = N.association_stats(mols, cal=cal, n_shift=200, n_boot=0, censor_bp=200, edges=False)
    assert res0['coordinate'].startswith('informative sites')
    # TF calls only in the irregular (linker-rich) stretches: depleted inside regular segments

    def tf_outside(m):
        from fiberhmm.nuc.molecule import molecule_outputs
        segs = [(s.start, s.end) for s in molecule_outputs(m, cal=cal).segments if s.state == 1]
        cand = rng.integers(250, m.length - 250, 40)
        return np.array([c for c in cand if not any(a <= c < b for a, b in segs)])
    res = N.association_stats(mols, cal=cal, y=tf_outside, n_shift=200, n_boot=100, censor_bp=200)
    if res.get('n_molecules', 0) >= 10:
        assert res['rate_ratio'] < 0.5 and res['p_depleted'] < 0.05
        assert 'edges' in res


# ---------------------------------------------------------------- scan directories and the CLI
def _write_scan(tmp: Path) -> Path:
    """A hand-built table-mode scan directory: one contig, nfr.region and fp.class hits with states, nfr molecule
    openness, callable runs and done units."""
    from fiberhmm.scan.tables import TableWriter
    from fiberhmm.scan.types import SCHEMA
    rng = np.random.default_rng(7)
    out = tmp / 'scan'
    order = {'chrT': 0}
    tables = {}
    common = ['chrom', 'start', 'end', 'idx_start', 'idx_end']
    hits = {'nfr': [(1000 + 2000 * i, 1200 + 2000 * i, 'nfr.region') for i in range(20)],
            'fp': [(1500 + 2000 * i, 1520 + 2000 * i, 'fp.class') for i in range(20)]}
    mols = [f'mol:v1 {k:016x}' for k in range(150)]
    open_ = rng.random(150)
    for mod, hs in hits.items():
        w = TableWriter(out / mod / 'hits.tsv.bgz', common + ['hit_id', 'type', 'anchor', 'strand', 'unit_id', 'score',
                                                              'label', 'flags', 'parent_id'], order)
        st = TableWriter(out / mod / 'states.tsv.bgz', common + ['hit_id', 'dataset_id', 'input_index', 'qname',
                                                                 'record', 'molecule', 'state'], order)
        for i, (s, e, t) in enumerate(hs):
            hid = f'{mod}:{t.split(".")[1]}:chrT:{(s + e) // 2}:{i:08x}'
            w.add('chrT', s, e, [hid, t, (s + e) // 2, '.', 'chrT:0-50000', 0.5, '', '', ''])
            for k in rng.choice(150, 40, replace=False):
                p = 0.2 + 0.6 * open_[k]
                state = '1' if rng.random() < p else '0'
                st.add('chrT', s, e, [hid, 'd1', 0, f'm1/{k}/ccs', f'{k:064x}:0', mols[k], state])
        tables[f'{mod}/hits.tsv.bgz'] = w.close()
        tables[f'{mod}/states.tsv.bgz'] = st.close()
    m = TableWriter(out / 'nfr' / 'molecules.tsv.bgz', common + ['dataset_id', 'input_index', 'qname', 'record',
                                                                 'molecule', 'openness'], order)
    for k in range(150):
        m.add('chrT', 0, 45_000, ['d1', 0, f'm1/{k}/ccs', f'{k:064x}:0', mols[k], float(open_[k])])
    tables['nfr/molecules.tsv.bgz'] = m.close()
    c = TableWriter(out / 'callable.tsv.bgz', common + ['dataset_id', 'min_molecules'], order)
    c.add('chrT', 0, 45_000, ['d1', 5])
    tables['callable.tsv.bgz'] = c.close()
    u = TableWriter(out / 'units.tsv.bgz', common + ['unit_id', 'zone_start', 'zone_end', 'mode', 'status',
                                                     'records'], order)
    u.add('chrT', 0, 50_000, ['chrT:0-50000', 0, 50_000, 'bed', 'done', 150])
    tables['units.tsv.bgz'] = u.close()
    (out / 'manifest.json').write_text(json.dumps({
        'schema': SCHEMA, 'scan_id': '0' * 16, 'datasets': [{'dataset_id': 'd1', 'ordinal': 0, 'files': []}],
        'modules': {'nfr': {}, 'fp': {}}, 'outputs': {'annotate': False, 'tables': tables}}))
    return out


def test_cli_cooccur_condition_and_overlap_on_a_scan_directory(tmp_path):
    from fiberhmm.scan.cli import main
    scan = _write_scan(tmp_path)
    assert main(['assoc', str(scan), '--mode', 'cooccur', '--a', 'nfr.region', '--b', 'fp.class', '--n-boot', '50',
                 '-o', str(tmp_path / 'm1')]) == 0
    r = json.loads((tmp_path / 'm1' / 'result.json').read_text())
    assert r['mode'] == 'M1' and r['n_pairs_tested'] >= 20 and np.isfinite(r['pooled']['p'])
    assert r['input']['a'] == 'nfr.region' and (tmp_path / 'm1' / 'pairs.tsv').exists()
    bed = tmp_path / 'anchors.bed'
    bed.write_text('chrT\t0\t10000\n')
    assert main(['assoc', str(scan), '--mode', 'condition', '--a', 'nfr.region', '--response', 'nfr:openness',
                 '--bed', str(bed), '--n-perm', '50', '--n-boot', '20', '-o', str(tmp_path / 'm2')]) == 0
    r2 = json.loads((tmp_path / 'm2' / 'result.json').read_text())
    assert r2['mode'] == 'M2' and r2['bins'][0]['n_anchors'] >= 1
    assert main(['assoc', str(scan), '--mode', 'overlap', '--a', 'nfr.region', '--b', 'fp.class', '--n-perm', '50',
                 '-o', str(tmp_path / 'm4')]) == 0
    r4 = json.loads((tmp_path / 'm4' / 'result.json').read_text())
    assert r4['A->B']['status'] in ('ok', 'refused') and r4['territory_bp'] == 45_000


def test_cli_refuses_m1_on_an_annotated_scan_with_the_reason(tmp_path, capsys):
    from fiberhmm.scan.cli import main
    scan = _write_scan(tmp_path)
    man = json.loads((scan / 'manifest.json').read_text())
    man['outputs']['annotate'] = True
    for rel in ('nfr/states.tsv.bgz', 'fp/states.tsv.bgz', 'nfr/molecules.tsv.bgz'):
        man['outputs']['tables'].pop(rel)
    (scan / 'manifest.json').write_text(json.dumps(man))
    with pytest.raises(SystemExit) as e:
        main(['assoc', str(scan), '--mode', 'cooccur', '--a', 'nfr.region', '--b', 'fp.class', '-o',
              str(tmp_path / 'x')])
    assert e.value.code == 2
    assert '--no-annotate' in capsys.readouterr().err


@pytest.fixture(scope='module')
def nuc_scan(tmp_path_factory):
    """A table-mode nuc scan of a synthetic BAM: regular arrays in one half, TF calls in linkers of the other."""
    from fiberhmm.scan import run_units
    from fiberhmm.scan.units import plan
    tmp = tmp_path_factory.mktemp('assoc_nuc')
    rng = np.random.default_rng(8)
    LENGTH = 60_000
    genome = molecule_sequence(LENGTH, rng)
    path = str(tmp / 'chromatin.bam')
    recs = []
    for k, start in enumerate(range(0, LENGTH - 9000, 600)):
        L = 8000
        seq = genome[start:start + L]
        nucs, tfs, pos = [], [], 150.0
        while pos < L - 300:
            regular = (start + pos) < LENGTH / 2
            nucs.append((int(pos), int(pos + 147)))
            if not regular and rng.random() < 0.5:
                tfs.append((int(pos + 160), int(pos + 180)))
            pos += 190 + rng.normal(0, 3) if regular else rng.uniform(200, 300)
        occ = np.zeros(L, bool)
        for s, e in nucs + tfs:
            occ[s:e] = True
        a = np.array([b == 'A' for b in seq])
        marks = np.flatnonzero(a & ~occ & (rng.random(L) < 0.55))
        marks = marks[(marks >= 12) & (marks < L - 12)]
        msps, p = [], 0
        for s, e in sorted(nucs + tfs):
            if s > p:
                msps.append((p, s))
            p = max(p, e)
        recs.append((start, f'm1/{7000 + k}/ccs', seq, nucs, msps, tfs, marks, 1 + k % 2))
    with pysam.AlignmentFile(path, 'wb', header=header(contig='chrS', length=LENGTH)) as out:
        for start, name, seq, nucs, msps, tfs, marks, hp in recs:
            r = make_record(out, name, seq, start, nucs, msps, tfs, marks)
            r.set_tag('HP', hp)
            out.write(r)
    pysam.index(path)
    units = plan({'chrS': LENGTH}, genome=True, contigs=['chrS'])
    out = tmp / 'scan'
    run_units([{'dataset_id': 'synth', 'paths': [path]}], units, modules=['nuc'],
              params={'nuc': {'calibration': 'self', 'calibration_population': 200}}, out=out, annotate=False)
    return out


def test_cli_rate_from_nuc_marks_with_haplotypes(nuc_scan, tmp_path):
    from fiberhmm.scan.cli import main
    assert main(['assoc', str(nuc_scan), '--mode', 'rate', '--a', 'nuc.seg.regular', '--b', 'tf', '--group',
                 'haplotype', '--n-perm', '100', '--n-boot', '50', '-o', str(tmp_path / 'm3')]) == 0
    r = json.loads((tmp_path / 'm3' / 'result.json').read_text())
    assert r['mode'] == 'M3' and r['n_molecules'] > 0
    assert r['group_info']['groups'] == ['1', '2']
    if np.isfinite(r.get('rate_ratio', np.nan)):
        assert r['rate_ratio'] < 1.0
