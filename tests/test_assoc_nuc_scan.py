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
                     lambda ms, an, seed: N.condition_stats(ms, an, backgrounds='random', k=2, n_perm=50, n_role=50,
                                                            n_boot=0, metrics=('R',), per_anchor=False,
                                                            return_null=True, frames=('anchor',), seed=seed))
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
    assert r['coordinate'].startswith('admissible reference bp')        # C1: the exposure is labelled
    assert r['group_info']['groups'] == ['1', '2']
    if np.isfinite(r.get('rate_ratio', np.nan)):
        assert r['rate_ratio'] < 1.0


# ---------------------------------------------------------------- review regressions (Codex round 1, group C)
def test_masks_follow_measured_geometry_and_partial_spans_count():
    """C2: a molecule with no informative sites in its flanks gets no appearance; one that reaches the first bin
    but not the summary span appears with a partial mask."""
    from fiberhmm.assoc.condition import LocusTable
    from fiberhmm.nuc.records import make_molecule
    nucs = [(int(d - 73), int(d + 74)) for d in np.arange(300, 20_000, 190)]
    msps = [(nucs[i][1], nucs[i + 1][0]) for i in range(len(nucs) - 1)]

    def mol(name, start, L, informative=True):
        nn = [(s - start, e - start) for s, e in nucs if s >= start and e <= start + L]
        mm = [(s - start, e - start) for s, e in msps if s >= start and e <= start + L]
        inf = np.ones(L, bool) if informative else np.zeros(L, bool)
        return make_molecule(L, nucs=nn, msps=mm, informative=inf, name=name, chrom='c', ref_start=start,
                             ref_end=start + L, blocks=np.array([[0, start, L]]))
    anchor = 9990                             # a nucleosome dyad: closed
    full = mol('m/1/ccs', 5000, 10_000)
    blind = mol('m/2/ccs', 5000, 10_000, informative=False)
    short = mol('m/3/ccs', anchor - 700, 1400)         # reaches 150 + 2 x 190 = 530 bp, not 1100 bp
    loci = LocusTable(['a'], ['c'], [anchor], [True], [-1])
    app, bins, info = N.anchor_appearances([full, blind, short], loci, P=190.0)
    assert len(app.locus) == 2 and info['no_bin_measured'] == 1
    full_mask = (1 << len(bins)) - 1
    masks = sorted(app.mask.tolist())
    assert masks[-1] == full_mask and 0 < masks[0] < full_mask and masks[0] & 1


def test_one_appearance_per_physical_molecule_and_one_record_in_m3():
    """C3: two records of one physical molecule give one M2 appearance; overlapping scan records of one molecule
    count once in M3."""
    from fiberhmm.assoc.condition import LocusTable
    sim = simulate_nuc(9, n_anchors=5, coverage=10)
    m = sim['molecules'][0]
    import copy
    twin = copy.deepcopy(m)
    twin.name = m.name.rsplit('/', 1)[0] + '/ccs_fwd'        # same movie/zmw, another record
    c, p = m.chrom, (m.ref_start + m.ref_end) // 2
    loci = LocusTable(['a'], [c], [p], [True], [-1])
    app, _bins, info = N.anchor_appearances([m, twin], loci, P=189.0, state_fn=lambda mol, a, l: '1')
    assert len(app.locus) == 1 and info.get('duplicate_records') == 1


def test_held_out_gives_each_direction_its_own_seed():
    """C4: the two directions draw independent permutations."""
    sim = simulate_nuc(10, n_anchors=10, coverage=15)
    seeds = []

    def analyse(ms, an, seed):
        seeds.append(seed)
        return N.condition_stats(ms, an, backgrounds=None, n_perm=20, n_role=0, n_boot=0, metrics=('R',),
                                 per_anchor=False, return_null=True, frames=('anchor',), seed=seed)
    N.held_out(sim['molecules'], lambda ms: sim['anchors'], analyse, seed=5)
    assert len(seeds) == 2 and seeds[0] != seeds[1]


def test_association_stats_drops_censored_points_before_mapping():
    """C5: a TF centre inside the censored end does not enter the informative coordinate."""
    rng = np.random.default_rng(11)
    cal = flat_calibration(nrl=190.0)
    m = array_molecule([('reg', 20, 190), ('irr', 10, 0)], rng, name='m/1/ccs', chrom='c', ref_start=0,
                       ref_end=7000)
    res = N.association_stats([m], cal=cal, y=lambda mol: np.array([100, 450, mol.length // 2]), n_shift=10,
                              n_boot=0, censor_bp=500, edges=False)
    if res.get('n_molecules'):
        assert res['O_in'] + res['O_out'] == 1


def test_molecule_index_is_per_dataset_and_territory_needs_every_dataset(tmp_path):
    """C6, C8."""
    from fiberhmm.assoc import scanio as S
    mi = S.MoleculeIndex()
    assert mi.add('mol:v1 aa', 'd1', 'q') != mi.add('mol:v1 aa', 'd2', 'q') and mi.n == 2
    scan = _write_scan(tmp_path)
    man = json.loads((scan / 'manifest.json').read_text())
    man['datasets'].append({'dataset_id': 'd2', 'ordinal': 1, 'files': []})
    (scan / 'manifest.json').write_text(json.dumps(man))
    assert S.territory(S.open_scan(scan)) == []
    assert S.territory(S.open_scan(scan), datasets=['d1'], regions=[('chrT', 100, 200)]) == [('chrT', 100, 200)]


def test_haplotypes_are_read_within_each_dataset(tmp_path):
    """C7: one QNAME in two datasets with HP 1 and HP 2 gives each molecule its own haplotype."""
    from fiberhmm.assoc import scanio as S
    paths = {}
    for ds, hp in (('d1', 1), ('d2', 2)):
        path = str(tmp_path / f'{ds}.bam')
        with pysam.AlignmentFile(path, 'wb', header={'HD': {'VN': '1.6', 'SO': 'coordinate'},
                                                     'SQ': [{'SN': 'chrT', 'LN': 10_000}]}) as out:
            r = pysam.AlignedSegment(out.header)
            r.query_name, r.reference_id, r.reference_start, r.cigarstring = 'm1/5/ccs', 0, 100, '50M'
            r.query_sequence, r.mapping_quality = 'A' * 50, 60
            r.set_tag('HP', hp)
            out.write(r)
        pysam.index(path)
        paths[ds] = path

    class FakeScan:
        manifest = {'datasets': [{'dataset_id': d, 'files': [{'path': p}]} for d, p in paths.items()]}
    mi = S.MoleculeIndex()
    mi.add('mol:v1 x', 'd1', 'm1/5/ccs')
    mi.add('mol:v1 x', 'd2', 'm1/5/ccs')
    assert list(S.haplotypes_from_bams(FakeScan(), mi)) == [1, 2]


def test_cli_condition_with_custom_labels_and_overlap_region(tmp_path):
    """C9, C10."""
    from fiberhmm.scan.cli import main
    scan = _write_scan(tmp_path)
    lab = tmp_path / 'labels.tsv'
    lab.write_text('molecule\ttreatment\n' + ''.join(f'mol:v1 {k:016x}\t{"x" if k % 2 else "y"}\n'
                                                     for k in range(150)))
    bed = tmp_path / 'anchors.bed'
    bed.write_text('chrT\t0\t20000\n')
    assert main(['assoc', str(scan), '--mode', 'condition', '--a', 'nfr.region', '--response', 'nfr:openness',
                 '--bed', str(bed), '--labels', str(lab), '--group-label', 'treatment', '--n-perm', '30',
                 '--n-boot', '20', '-o', str(tmp_path / 'm2')]) == 0
    r2 = json.loads((tmp_path / 'm2' / 'result.json').read_text())
    assert r2['group_info']['groups'] == ['x', 'y']
    assert main(['assoc', str(scan), '--mode', 'overlap', '--a', 'nfr.region', '--b', 'fp.class', '--region',
                 'chrT:0-20000', '--n-perm', '30', '-o', str(tmp_path / 'm4')]) == 0
    r4 = json.loads((tmp_path / 'm4' / 'result.json').read_text())
    assert r4['territory_bp'] == 20_000


# ---------------------------------------------------------------- review regressions (Codex round 2, group C)
def test_flank_eligibility_is_molecular_when_insertions_lengthen_the_molecule():
    from fiberhmm.assoc.condition import LocusTable
    from fiberhmm.nuc.records import make_molecule
    L = 4000
    blocks = np.array([[0, 0, 400], [1600, 400, 800], [3600, 1200, 400]])   # two 1.2 kb insertions
    inf = np.zeros(L, bool)
    for q, _r, n in blocks:
        inf[q:q + n] = True
    inf[400:1600] = True
    inf[2400:3600] = True
    nucs = [(int(d - 73), int(d + 74)) for d in np.arange(200, L - 100, 190)]
    m = make_molecule(L, nucs=nucs, informative=inf, name='m/1/ccs', chrom='c', ref_start=0, ref_end=1600,
                      blocks=blocks)
    a_ref = 800                                   # molecular 2000, inside a nucleosome near the middle
    loci = LocusTable(['a'], ['c'], [a_ref], [True], [-1])
    app, _bins, info = N.anchor_appearances([m], loci, P=190.0, state_fn=lambda mol, a, l: '0')
    assert len(app.locus) == 1 and app.mask[0] > 0


def test_conflicting_records_of_one_molecule_are_dropped_and_m3_counts_one_record():
    import copy
    from fiberhmm.assoc.condition import LocusTable
    sim = simulate_nuc(12, n_anchors=5, coverage=10)
    m = sim['molecules'][0]
    twin = copy.deepcopy(m)
    twin.name = m.name.rsplit('/', 1)[0] + '/ccs_rev'
    p = (m.ref_start + m.ref_end) // 2
    loci = LocusTable(['a'], [m.chrom], [p], [True], [-1])
    states = {id(m): '1', id(twin): '0'}
    for order in ([m, twin], [twin, m]):
        app, _b, info = N.anchor_appearances(order, loci, P=189.0, state_fn=lambda mol, a, l: states[id(mol)])
        assert len(app.locus) == 0 and info['conflicting_records'] == 1
    rng = np.random.default_rng(13)
    cal = flat_calibration(nrl=190.0)
    mol = array_molecule([('reg', 20, 190), ('irr', 10, 0)], rng, name='m/7/ccs', chrom='c', ref_start=0,
                         ref_end=7000)
    dup = copy.deepcopy(mol)
    one = N.association_stats([mol], cal=cal, y=lambda x: np.array([1500, 3000, 5000]), n_shift=10, n_boot=0,
                              censor_bp=200, edges=False)
    two = N.association_stats([mol, dup], cal=cal, y=lambda x: np.array([1500, 3000, 5000]), n_shift=10,
                              n_boot=0, censor_bp=200, edges=False)
    assert two['skipped']['duplicate_records'] == 1 and two.get('n_molecules') == one.get('n_molecules')


def _write_marks_scan(tmp: Path, records) -> Path:
    """A table-mode scan with nuc marks only. records: [(record, molecule, [(start, end, q1)], [tf centres])]."""
    from fiberhmm.scan.tables import TableWriter
    from fiberhmm.scan.types import SCHEMA
    out = tmp / 'mscan'
    common = ['chrom', 'start', 'end', 'idx_start', 'idx_end']
    w = TableWriter(out / 'nuc' / 'marks.tsv.bgz', common + ['group', 'dataset_id', 'input_index', 'qname', 'record',
                                                             'molecule', 'mol_start', 'mol_len', 'q', 'name'],
                    {'chrT': 0})
    for rec, molid, segs, tfs in records:
        for s, e, q1 in segs:
            w.add('chrT', s, e, ['nucseg', 'd1', 0, f'q{rec}', rec, molid, s, e - s, f'{q1},200,0', '.'])
        for c in tfs:
            w.add('chrT', c - 10, c + 10, ['nucsz', 'd1', 0, f'q{rec}', rec, molid, c - 10, 20, '18,0,255,128', '.'])
    tables = {'nuc/marks.tsv.bgz': w.close()}
    (out / 'manifest.json').write_text(json.dumps({'schema': SCHEMA, 'datasets': [{'dataset_id': 'd1', 'files': []}],
                                                   'modules': {'nuc': {}},
                                                   'outputs': {'annotate': False, 'tables': tables}}))
    return out


def test_scan_m3_merges_records_of_one_molecule_and_clips_regions(tmp_path):
    from fiberhmm.assoc import scanio as S
    from fiberhmm.assoc.rate import rate_ratio
    # one record: regular [0,500), disordered [500,1000); TF centres 300, 600, 700, 800; region [200, 800)
    scan = _write_marks_scan(tmp_path, [('r1', 'mol:v1 a', [(0, 500, 1), (500, 1000, 2)], [300, 600, 700, 800])])
    d = S.rate_data_from_marks(S.open_scan(scan), 'nuc.seg.regular', 'tf', regions=[('chrT', 200, 800),
                                                                                     ('chrT', 300, 700)])
    r = rate_ratio(d, n_shift=0, n_boot=0)
    assert (r['E_in'], r['E_out'], r['O_in'], r['O_out']) == (300, 300, 1, 2)
    # two overlapping records of one molecule: unique geometry and calls are kept, shared ones counted once
    scan2 = _write_marks_scan(tmp_path / 'b', [
        ('r1', 'mol:v1 a', [(0, 900, 1), (900, 1000, 2)], [100, 200, 950]),
        ('r2', 'mol:v1 a', [(900, 1800, 2)], [950, 1200, 1500])])
    d2 = S.rate_data_from_marks(S.open_scan(scan2), 'nuc.seg.regular', 'tf')
    assert d2.n == 1
    r2 = rate_ratio(d2, n_shift=0, n_boot=0)
    assert (r2['E_in'], r2['E_out'], r2['O_in'], r2['O_out']) == (900, 900, 2, 3)


def test_reverse_association_p_uses_the_cluster_t_reference():
    from scipy.stats import t as tdist
    rng = np.random.default_rng(5)
    rows = []
    for locus in range(5):
        for m in range(30):
            lam = rng.normal()
            x = int(rng.random() < 0.5)
            y = int(rng.random() < 1 / (1 + np.exp(-(0.4 * x + 0.5 * lam))))
            rows.append((locus, locus * 100 + m, y, x, lam))
    loc, mol, y, x, lam = map(np.array, zip(*rows))
    res = N.reverse_association(loc, mol, y, x, lam, n_boot=200)
    c = res['clogit']
    if np.isfinite(c['df']):
        z = c['log_or'] / c['se_boot']
        assert c['p'] == pytest.approx(2 * tdist.sf(abs(z), c['df']))


# ---------------------------------------------------------------- regressions (Codex round 3, group C; unreviewed fixes)
def test_an_ineligible_longer_record_does_not_hide_an_eligible_one():
    import copy
    from fiberhmm.assoc.condition import LocusTable
    from fiberhmm.nuc.records import make_molecule
    nucs = [(int(d - 73), int(d + 74)) for d in np.arange(200, 5000, 190)]
    good = make_molecule(4000, nucs=[n for n in nucs if n[1] <= 4000], name='m/9/ccs', chrom='c', ref_start=0,
                         ref_end=4000, blocks=np.array([[0, 0, 4000]]))
    blind = make_molecule(5000, nucs=nucs, informative=np.zeros(5000, bool), name='m/9/ccs2', chrom='c',
                          ref_start=0, ref_end=5000, blocks=np.array([[0, 0, 5000]]))
    loci = LocusTable(['a'], ['c'], [1910], [True], [-1])
    for order in ([good, blind], [blind, good]):
        app, _b, _i = N.anchor_appearances(order, loci, P=190.0, state_fn=lambda mol, a, l: '0')
        assert len(app.locus) == 1


def test_merged_scan_molecule_keeps_every_qname(tmp_path):
    from fiberhmm.assoc import scanio as S
    scan = _write_marks_scan(tmp_path, [('r1', 'mol:v1 a', [(0, 500, 1)], [100]),
                                        ('r2', 'mol:v1 a', [(2000, 2500, 2)], [2100])])
    d = S.rate_data_from_marks(S.open_scan(scan), 'nuc.seg.regular', 'tf')
    assert d.n == 1 and set(d.labels['qname'][0]) == {'qr1', 'qr2'}


def test_scanio_cross_scan_states_and_explicit_empty_regions(tmp_path):
    from fiberhmm.assoc import scanio as S
    from fiberhmm.scan.tables import TableWriter
    marks = _write_marks_scan(tmp_path / 'a', [('r0', 'mol0', [(100, 900, 1)], [])])
    other = tmp_path / 'b'
    other.mkdir()
    common = ['chrom', 'start', 'end', 'idx_start', 'idx_end']
    writer = TableWriter(other / 'fp' / 'hits.tsv.bgz', common + ['hit_id', 'type'], {'chrT': 0})
    writer.add('chrT', 300, 350, ['h0', 'fp.class'])
    hits = writer.close()
    writer = TableWriter(other / 'fp' / 'states.tsv.bgz', common + [
        'hit_id', 'dataset_id', 'input_index', 'qname', 'record', 'molecule', 'state'], {'chrT': 0})
    writer.add('chrT', 300, 350, ['h0', 'd1', 0, 'q0', 'r0', 'mol0', '1'])
    states = writer.close()
    manifest = json.loads((marks / 'manifest.json').read_text())
    manifest['outputs']['tables'] = {'fp/hits.tsv.bgz': hits, 'fp/states.tsv.bgz': states}
    (other / 'manifest.json').write_text(json.dumps(manifest))
    a, b = S.open_scan(marks), S.open_scan(other)
    data = S.rate_data_from_marks(a, 'nuc.seg.regular', 'fp.class', b_scan=b)
    assert data.n == 1 and len(data.points[0]) == 1
    assert S.rate_data_from_marks(a, 'nuc.seg.regular', 'fp.class', b_scan=b, regions=[]).n == 0
    assert S.territory(a, regions=[]) == []
    assert list(S._rows(a, 'nuc/marks.tsv.bgz', [])) == []
    manifest['frame'] = {'kind': 'lifted'}
    (other / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(S.ScanInputError, match='coordinate frame'):
        S.rate_data_from_marks(a, 'nuc.seg.regular', 'fp.class', b_scan=S.open_scan(other))
