"""EXPERIMENTAL NFR variants + element co-accessibility (fiberhmm.inference.accessibility, preview).

Synthetic fibers: every molecule is a nucleosome array whose open stretches are planted; positions/hits follow
the accessible (p=0.6) / nucleosome (p=0.03) pattern so edge ranges come from the first/last mark as in real data.
"""
import gzip
import json
import random

import numpy as np
import pytest

from fiberhmm.inference.accessibility import NFROptions, run_accessibility
from fiberhmm.inference.accessibility import coaccess as C
from fiberhmm.inference.accessibility import gaps as G
from fiberhmm.inference.accessibility.workflow import classes_from_rows

FAST = dict(bootstrap=30, kmax=6)


def _fill(a, b, linker, rng=None, nuc=147):
    """Nucleosomes tiling [a, b): 147-bp nucleosomes from ``a`` with linkers of ``linker`` bp (+- 8 bp per linker
    when ``rng`` is given, clipped to 8-58 bp so no linker is an NFR); the last nucleosome ends at ``b``."""
    out, x = [], int(a)
    while b - x > 0:
        if b - x <= nuc + 90 + 60:
            out.append((x, int(b))); break
        out.append((x, x + nuc))
        x += nuc + (int(np.clip(linker + rng.normal(0, 8), 8, 58)) if rng is not None else linker)
    return out


def fiber(uid, opens, rng, *, lo=100, hi=3000, linker=30, strand='CT', ds='d', jitter=5., step=4, window=(400, 2800), tfs=()):
    """One synthetic molecule: ``opens`` are planted open stretches (each edge jittered); the rest is nucleosomes."""
    opens = sorted((a + rng.normal(0, jitter), b + rng.normal(0, jitter)) for a, b in opens)
    nucs, x = [], lo
    for a, b in opens:
        nucs += _fill(x, int(a), linker, rng); x = int(b)
    nucs += _fill(x, hi, linker, rng)
    pos = np.arange(window[0] + (0 if strand == 'CT' else 2), window[1], step)
    prot = np.zeros(len(pos), bool)
    for a, b in nucs:
        prot |= (pos >= a) & (pos < b)
    hit = np.where(prot, rng.random(len(pos)) < .03, rng.random(len(pos)) < .6)
    return dict(unit_id=f'unit_{uid:05d}', read_name=f'read{uid:05d}', strand=strand, reference_start=lo, reference_end=hi,
                positions=pos.tolist(), hits=hit.astype(int).tolist(), raw_nuc_intervals=[list(v) for v in nucs],
                raw_tf_intervals=[list(t) for t in tfs], msp_intervals=[], source_members=[dict(read_name=f'read{uid:05d}')])


def payload(units, window=(400, 2800), ds='d', chemistry='ddda'):
    return dict(region=dict(chrom='chrT', start=window[0], end=window[1]),
                strata=[dict(dataset_id=ds, stratum_id=ds, chemistry=chemistry, units=units)])


def planted(states, n=400, seed=3, **kw):
    """states: [(weight, [open stretches])]; returns units with the planted state per read."""
    rng = np.random.default_rng(seed)
    w = np.array([s[0] for s in states], float); w /= w.sum()
    pick = rng.choice(len(states), n, p=w)
    units = [fiber(i, states[j][1], rng, strand='CT' if i % 2 == 0 else 'GA', **kw) for i, j in enumerate(pick)]
    return units, pick


def close(v, a, b, tol=15):
    return abs(v['L'] - a) <= tol and abs(v['R'] - b) <= tol


# ---------------------------------------------------------------- per-read gaps (Timer definition)
def test_gap_is_between_nucleosomes_of_at_least_90_bp_and_footprints_do_not_split_it():
    u = dict(unit_id='u', strand='CT', dataset='d', positions=[1000, 1100, 1200, 1300], hits=[1, 1, 1, 1],
             raw_nuc_intervals=[[700, 847], [1050, 1110], [1400, 1547]], raw_tf_intervals=[[1150, 1180]])
    ok, gaps = G.read_gaps(u, (900, 1350), min_gap_bp=60)
    assert ok and len(gaps) == 1
    g = gaps[0]
    assert (g['g0'], g['g1']) == (847, 1400)          # the 60-bp protection is not a nucleosome: one NFR
    assert g['tfs'] == [(1150, 1180)]                  # kept as an annotation only
    assert g['lr'] == (847, 877) and g['rr'] == (1370, 1400)   # edge ranges capped at 30 bp


def test_read_without_a_nucleosome_on_both_sides_is_censored():
    u = dict(unit_id='u', strand='CT', dataset='d', positions=[], hits=[], raw_nuc_intervals=[[1400, 1547], [1700, 1847]])
    assert G.read_gaps(u, (1300, 1600), 60) == (False, [])


# ---------------------------------------------------------------- planted variants
@pytest.fixture(scope='module')
def split_locus():
    states = [(.35, [(1400, 1700)]), (.20, [(1480, 1620)]), (.15, [(1400, 1520), (1620, 1700)]), (.30, [])]
    units, pick = planted(states, n=500, seed=5)
    res = run_accessibility(payload(units), dict(FAST, nfr_regions=[(1380, 1720)]))
    return res, pick


def test_full_core_and_split_variants_are_recovered_with_their_prevalence(split_locus):
    res, pick = split_locus
    n, = res['nfrs']
    assert n['status'] == 'ok' and n['callable'] == n['reads'] == 500
    vs = n['variants']
    assert len(vs) == 4, [(v['L'], v['R']) for v in vs]
    full = next(v for v in vs if close(v, 1400, 1700)); core = next(v for v in vs if close(v, 1480, 1620))
    left = next(v for v in vs if close(v, 1400, 1520)); right = next(v for v in vs if close(v, 1620, 1700))
    truth = np.bincount(pick, minlength=4)/len(pick)
    assert abs(full['prevalence'] - truth[0]) < .04 and abs(core['prevalence'] - truth[1]) < .04
    assert abs(left['prevalence'] - truth[2]) < .04 and abs(right['prevalence'] - truth[2]) < .04
    assert full['strict'] <= full['prevalence'] and full['ci'][0] <= full['prevalence'] <= full['ci'][1]
    assert core['relation'] == 'core'
    assert full['relation'] == f"merged {left['name']}+{right['name']}"      # one opening covering both registers
    split = next(c for c in n['configurations'] if c['label'] == f"{left['name']}+{right['name']}")
    assert abs(split['weight'] - truth[2]) < .04
    assert abs(n['closed'] - truth[3]) < .03


def test_shifted_variant_is_separated_from_the_full_one():
    units, pick = planted([(.5, [(1400, 1700)]), (.3, [(1470, 1770)]), (.2, [])], n=400, seed=8)
    res = run_accessibility(payload(units), dict(FAST, nfr_regions=[(1380, 1790)]))
    vs = res['nfrs'][0]['variants']
    assert len(vs) == 2
    full = next(v for v in vs if close(v, 1400, 1700)); shifted = next(v for v in vs if close(v, 1470, 1770))
    assert full['relation'] == 'full' and shifted['relation'] == 'shifted-right'
    assert abs(shifted['prevalence'] - np.mean(pick == 1)) < .04


def test_membership_and_map_configuration_per_read(split_locus):
    res, pick = split_locus
    ok = 0
    vs = {(round(v['L'], -1), round(v['R'], -1)): v['name'] for v in res['nfrs'][0]['variants']}
    for i, (uid, m) in enumerate(sorted(res['molecules'].items())):
        rec = m['nfr']['N1']
        assert set(rec['p']) == set(vs.values()) and 0 <= rec['map_posterior'] <= 1
        if pick[i] == 3:
            ok += rec['map'] == 'closed'
        if pick[i] == 2:
            ok += '+' in rec['map']
    assert ok >= .9*((pick == 3).sum() + (pick == 2).sum())


def test_stringency_is_a_knob_and_depth_mode_uses_timer_widths():
    units, _ = planted([(.6, [(1400, 1760)]), (.4, [(1450, 1560)])], n=300, seed=4)
    p = payload(units)
    res = run_accessibility(p, dict(FAST, mode='depth', nfr_regions=[(1380, 1780)]))
    st = {s['name']: s['prevalence'] for s in res['nfrs'][0]['depth_states']}
    assert st['>=175'] == pytest.approx(st['>=300'], abs=1e-9) and st['>=500'] == 0
    assert 0.5 < st['>=175'] < .7
    assert res['nfrs'][0]['variants'] == [] and all(e['subtype'] == 'depth' for e in res['elements'])
    with pytest.raises(ValueError):
        NFROptions.from_params(dict(stringency=1.5))
    with pytest.raises(ValueError):
        NFROptions.from_params(dict(colour='red'))


def test_no_callable_read_is_not_analysable_and_no_region_is_a_warning():
    rng = np.random.default_rng(1)
    units = [fiber(i, [(1400, 1700)], rng, lo=1450) for i in range(40)]
    res = run_accessibility(payload(units), dict(FAST, nfr_regions=[(1380, 1720)]))
    assert res['nfrs'][0]['status'] == 'not_analysable' and res['nfrs'][0]['callable'] == 0
    closed = [fiber(i, [], rng) for i in range(40)]
    res = run_accessibility(payload(closed), dict(FAST))
    assert res['nfrs'] == [] and 'No NFR region found' in res['warnings'][0]


def test_nfr_regions_are_detected_from_the_profile():
    units, _ = planted([(.7, [(1400, 1700)]), (.3, [])], n=200, seed=2)
    res = run_accessibility(payload(units), dict(FAST))
    n, = res['nfrs']
    assert n['source'] == 'detected' and abs(n['start'] - 1400) <= 20 and abs(n['end'] - 1700) <= 20


# ---------------------------------------------------------------- determinism
def test_outputs_do_not_depend_on_read_order_and_are_byte_identical(tmp_path):
    states = [(.3, [(800, 1000), (1400, 1700), (2100, 2300)]), (.2, [(1480, 1620), (2100, 2300)]), (.2, [(800, 1000)]), (.3, [])]
    units, _ = planted(states, n=250, seed=6)
    shuffled = list(units); random.Random(0).shuffle(shuffled)
    params = dict(FAST, nfr_regions=[(780, 1020), (1380, 1720), (2080, 2320)], elements='nfrs',
                  max_discovery_molecules=0, max_quantification_molecules=0)
    a = run_accessibility(payload(units), params, tmp_path/'a')
    b = run_accessibility(payload(shuffled), params, tmp_path/'b')
    assert len(a['pairs']) == 3 and a['combos']['status'] == 'ok'          # the pair and combination files are not empty
    assert len(a['combos']['patterns']) == 8 and sum(p['obs'] for p in a['combos']['patterns']) == a['combos']['n']
    for name in ('variants.tsv', 'configurations.tsv', 'molecules.tsv.gz', 'coaccess.tsv', 'combos.tsv', 'result.json'):
        assert (tmp_path/'a'/name).read_bytes() == (tmp_path/'b'/name).read_bytes(), name
    ma = json.loads((tmp_path/'a'/'manifest.json').read_text())
    assert ma['schema'] == 'fiberhmm.accessibility.preview.v1' and ma['experimental'] is True
    assert a['element_states'] == b['element_states'] and a['molecules'] == b['molecules']   # v1 additions too
    assert ma['outputs'] == json.loads((tmp_path/'b'/'manifest.json').read_text())['outputs']
    assert a['nfrs'] == b['nfrs'] and len((tmp_path/'a'/'coaccess.tsv').read_text().splitlines()) == 4
    with gzip.open(tmp_path/'a'/'molecules.tsv.gz', 'rt') as fh:
        rows = fh.read().splitlines()
    assert rows[0].split('\t')[:3] == ['nfr', 'unit_id', 'read_name'] and len(rows) == 1 + 3*250   # one row per molecule and NFR


# ---------------------------------------------------------------- co-accessibility
def two_nfr_units(n, seed, p_open, window=(400, 2800)):
    """Two NFRs (N1 at 1000-1250, N2 at 2000-2250); p_open(rng, i) -> (open1, open2, linker)."""
    rng = np.random.default_rng(seed); units = []
    for i in range(n):
        o1, o2, linker = p_open(rng, i)
        opens = ([(1000, 1250)] if o1 else []) + ([(2000, 2250)] if o2 else [])
        units.append(fiber(i, opens, rng, linker=linker, strand='CT' if i % 2 == 0 else 'GA', window=window))
    return units


def pair_of(res, a, b):
    return next(p for p in res['pairs'] if {p['a'], p['b']} == {a, b})


def test_planted_co_accessibility_is_called():
    def p_open(rng, i):
        o1 = rng.random() < .5
        return o1, rng.random() < (.8 if o1 else .2), 30
    res = run_accessibility(payload(two_nfr_units(600, 11, p_open)), dict(FAST, nfr_regions=[(980, 1270), (1980, 2270)]))
    p = pair_of(res, 'N1:V1', 'N2:V1')
    assert p['class'] == 'co-accessible' and p['q'] < 1e-6 and p['mh'] > 2 and p['shared'] == 0
    assert p['adjust'] == 'openness x channel'


def test_openness_only_confound_is_not_called_by_the_stratified_test():
    """Both NFRs depend only on how open the whole fiber is (linker length): pooled Fisher calls them co-accessible,
    the openness-stratified exact test must not."""
    def p_open(rng, i):
        z = rng.random() < .5
        return rng.random() < (.85 if z else .15), rng.random() < (.85 if z else .15), (50 if z else 14)
    res = run_accessibility(payload(two_nfr_units(800, 12, p_open)), dict(FAST, nfr_regions=[(980, 1270), (1980, 2270)]))
    p = pair_of(res, 'N1:V1', 'N2:V1')
    assert p['fisher_p'] < 1e-20 and p['log2or'] > 2          # what the pooled table says
    assert p['class'] != 'co-accessible' and p['p_exact'] > .01
    assert abs(p['mh']) < 1 and p['null_median'] > 2           # the openness-preserving null centre absorbs it


def test_shared_openings_are_excluded_and_overlapping_variants_not_tested(split_locus):
    res, _ = split_locus
    names = {v['name']: v for v in res['nfrs'][0]['variants']}
    left = next(v for v in names.values() if close(v, 1400, 1520)); right = next(v for v in names.values() if close(v, 1620, 1700))
    p = pair_of(res, left['id'], right['id'])
    assert p['shared'] > 0                         # full-width reads cover both centres: excluded (Timer shared rule)
    assert p['n'] + p['shared'] <= 500
    skipped = {(s['a'], s['b']) for s in res['family']['skipped'] if s['reason'] == 'overlap'}
    full = next(v for v in names.values() if close(v, 1400, 1700))
    assert any(full['id'] in pair for pair in skipped)


def test_variant_x_class_pairs_internal_footprint_label_and_join():
    states = [(.4, [(1400, 1700)]), (.3, [(1400, 1520), (1620, 1700)]), (.3, [])]
    units, pick = planted(states, n=400, seed=9)
    # a supported 80-bp class inside the split's protection, bound on split reads, rarely elsewhere
    rng = np.random.default_rng(0)
    member = {u['unit_id']: int(rng.random() < (.9 if j == 1 else .05)) for u, j in zip(units, pick)}
    sup = {('class_007', 'd::CT'): dict(start=1530, end=1610, prevalence=.3),
           ('class_007', 'd::GA'): dict(start=1530, end=1610, prevalence=.3)}
    st = {'class_007': {u['unit_id']: (member[u['unit_id']], f"d::{u['strand']}") for u in units}}
    classes = classes_from_rows(sup, st)
    res = run_accessibility(payload(units), dict(FAST, nfr_regions=[(1380, 1720)]), classes=classes)
    assert res['class_join'] == dict(classes=1, class_molecules=400, joined_molecules=400, molecules=400)
    n = res['nfrs'][0]
    split = next(c for c in n['configurations'] if c['label'].count('+') == 1 and 'other' not in c['label'])
    assert split['internal_footprint'] == ['class_007'] and 'full NFR with internal footprint' in split['display_label']
    left = next(v for v in n['variants'] if close(v, 1400, 1520))
    p = pair_of(res, left['id'], 'class_007')
    assert p['class'] == 'co-accessible' and p['nested'] == ''
    full = next(v for v in n['variants'] if close(v, 1400, 1700))
    q = pair_of(res, full['id'], 'class_007')
    assert q['class'] == 'anti' and q['nested'] == 'inside'
    off = run_accessibility(payload(units), dict(FAST, nfr_regions=[(1380, 1720)], internal_footprint_labels=False), classes=classes)
    assert all(c['internal_footprint'] is None for c in off['nfrs'][0]['configurations'])
    other = classes_from_rows(sup, {'class_007': {f'x{i}': (1, 'd::CT') for i in range(50)}})
    res = run_accessibility(payload(units), dict(FAST, nfr_regions=[(1380, 1720)]), classes=other)
    assert res['class_join']['joined_molecules'] == 0 and any('unit_id join' in w for w in res['warnings'])
    assert all(c['internal_footprint'] is None for c in res['nfrs'][0]['configurations'])   # geometry alone does not label


def test_within_cluster_check_and_combinations():
    def p_open(rng, i):
        o1 = rng.random() < .5
        return o1, rng.random() < (.8 if o1 else .2), 30
    units = two_nfr_units(400, 13, p_open)
    res = run_accessibility(payload(units), dict(FAST, nfr_regions=[(980, 1270), (1980, 2270)], within_clusters=3))
    p = pair_of(res, 'N1:V1', 'N2:V1')
    assert p['cluster_n'] == 400 and 'cluster' in p       # None when the clusters separate the states (not estimable)
    assert res['within_clusters'] == dict(mode='masked_kmeans', k=3, molecules=400)
    # caller-supplied labels (e.g. the browser's read clusters): the effect within them
    labels = {u['unit_id']: i % 3 for i, u in enumerate(units)}
    res = run_accessibility(payload(units), dict(FAST, nfr_regions=[(980, 1270), (1980, 2270)]), clusters=labels)
    p = pair_of(res, 'N1:V1', 'N2:V1')
    assert p['cluster'] > 1 and res['within_clusters']['mode'] == 'labels'
    # three non-overlapping elements -> automatic combinations
    units3, _ = planted([(.5, [(1000, 1250)]), (.5, [(1000, 1250), (2000, 2250)])], n=300, seed=14)
    res = run_accessibility(payload(units3), dict(FAST, nfr_regions=[(980, 1270), (1980, 2270)], elements='nfrs',
                                                  combo_elements=('N1:open', 'N2:open', 'N1:open')))
    assert res['combos'] is None and any('overlap' in w for w in res['warnings'])


# ---------------------------------------------------------------- statistics
def test_exact_stratified_test_is_fisher_with_one_stratum_and_mh_is_the_common_or():
    from scipy.stats import fisher_exact
    rng = np.random.default_rng(0)
    x = (rng.random(300) < .4).astype(int); y = ((rng.random(300) < .3) | (x & (rng.random(300) < .3))).astype(int)
    p, _, _ = C.exact_stratified(x, y, np.zeros(300, int).astype(str))
    t = C.table(x, y)
    assert p == pytest.approx(fisher_exact([[t[0], t[1]], [t[2], t[3]]])[1], rel=1e-6)
    s = np.array(['a']*150 + ['b']*150)
    xx = np.r_[np.ones(50), np.zeros(100), np.ones(50), np.zeros(100)].astype(int)
    yy = np.r_[np.ones(40), np.zeros(10), np.ones(20), np.zeros(80), np.ones(40), np.zeros(10), np.ones(20), np.zeros(80)].astype(int)
    mh, lo, hi = C.mantel_haenszel(xx, yy, s)
    assert mh == pytest.approx(np.log2(40*80/(10*20)), rel=1e-9) and lo < mh < hi
    assert C.bh([.01, .04, .03])[0] == pytest.approx(.03)


def test_curveball_preserves_row_and_column_sums():
    rng = np.random.default_rng(1)
    M = (rng.random((60, 4)) < .4).astype(np.int8)
    for X in C.curveball(M, 5, np.random.default_rng(2)):
        assert (X.sum(0) == M.sum(0)).all() and (X.sum(1) == M.sum(1)).all()


# ---------------------------------------------------------------- CLI on a synthetic BAM
def nfr_bam(path, n=160, seed=21):
    """DddA BAM (both strands) with nucleosome annotations: an NFR at chr1:1400-1700 open on 60% of molecules,
    a core register 1480-1620 on 20%, closed on 20%."""
    import pysam
    from fiberhmm.io.bam_header import append_chemistry
    rng = np.random.default_rng(seed)
    ref0, length = 200, 2600
    reference = ''.join(rng.choice(list('ACGT'), length))
    header = pysam.AlignmentHeader.from_dict(dict(HD={'VN': '1.6', 'SO': 'coordinate'}, SQ=[{'SN': 'chr1', 'LN': 4000}]))
    header = append_chemistry(header, dict(assay='daf', enzyme='ddda', platform='pacbio', mode='daf'))
    with pysam.AlignmentFile(str(path), 'wb', header=header) as out:
        for i in range(n):
            strand = 'CT' if i % 2 == 0 else 'GA'; target, mark = ('C', 'Y') if strand == 'CT' else ('G', 'R')
            r_ = rng.random()
            opens = [(1400, 1700)] if r_ < .6 else [(1480, 1620)] if r_ < .8 else []
            opens = [(int(a + rng.integers(-4, 5)), int(b + rng.integers(-4, 5))) for a, b in opens]
            nucs, x = [], ref0
            for a, b in opens:
                nucs += _fill(x, a, 30); x = b
            nucs += _fill(x, ref0 + length, 30)
            prot = np.zeros(length, bool)
            for a, b in nucs:
                prot[a - ref0:b - ref0] = True
            seq = ''.join(mark if base == target and rng.random() < (.02 if prot[j] else .6) else base for j, base in enumerate(reference))
            rec = pysam.AlignedSegment(header)
            rec.query_name = f'm{i:04d}'; rec.query_sequence = seq
            rec.reference_id = 0; rec.reference_start = ref0; rec.mapping_quality = 60; rec.cigarstring = f'{length}M'
            rec.set_tag('st', strand)
            msps = [(b0, a1) for (_a0, b0), (a1, _b1) in zip(nucs, nucs[1:]) if a1 > b0]
            rec.set_tag('MA', f'{length};nuc.:' + ','.join(f'{a - ref0 + 1}-{b - a}' for a, b in nucs)
                        + ';msp.:' + ','.join(f'{a - ref0 + 1}-{b - a}' for a, b in msps))
            out.write(rec)
    pysam.index(str(path))
    return path


def test_cli_writes_every_output_deterministically(tmp_path, capsys):
    from fiberhmm.inference.accessibility.cli import main
    bam = nfr_bam(tmp_path/'nfr.bam')
    args = ['--bam', str(bam), '--region', 'chr1:700-2400', '--nfr', '1380-1720', '--bootstrap', '20', '--cores', '1']
    main(args + ['--output', str(tmp_path/'a')])
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary['status'] == 'complete' and summary['experimental'] is True
    n, = summary['nfrs']
    assert n['status'] == 'ok' and n['callable'] == 160 and len(n['variants']) >= 2
    main(args + ['--output', str(tmp_path/'b')])
    for name in ('variants.tsv', 'configurations.tsv', 'molecules.tsv.gz', 'coaccess.tsv', 'combos.tsv', 'result.json'):
        assert (tmp_path/'a'/name).read_bytes() == (tmp_path/'b'/name).read_bytes(), name
    manifest = json.loads((tmp_path/'a'/'manifest.json').read_text())
    assert manifest['tool'] == 'fiberhmm-nfr' and manifest['fiberhmm']['version']
    assert manifest['inputs']['files'][0]['path'].endswith('nfr.bam') and manifest['nfr_regions'][0]['source'] == 'given'
    with pytest.raises(SystemExit):
        main(args + ['--output', str(tmp_path/'a')])           # never overwrites
    assert 'must be empty' in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(['--bam', str(bam), '--region', 'chr1:700-2400', '--nfr', '2500-2600', '--output', str(tmp_path/'c')])
    assert 'outside the analysis window' in capsys.readouterr().err
    main(['--schema'])
    assert json.loads(capsys.readouterr().out)['stringency'] == .9


# ---------------------------------------------------------------- Codex review regressions (nfr-preview)
def test_stringency_only_lowers_or_keeps_k():
    units, _ = planted([(.5, [(1400, 1700)]), (.3, [(1470, 1770)]), (.2, [])], n=300, seed=8)
    ks = [run_accessibility(payload(units), dict(FAST, bootstrap=0, stringency=s, nfr_regions=[(1380, 1790)]))['nfrs'][0]
          for s in (.5, .9, 1.)]
    assert ks[0]['ps_curve'] == ks[1]['ps_curve'] == ks[2]['ps_curve']
    assert ks[0]['k'] >= ks[1]['k'] >= ks[2]['k']


def test_depth_mode_uses_every_gap_of_an_overflow_read():
    u = dict(unit_id='u', strand='CT', dataset='d', positions=[], hits=[],
             raw_nuc_intervals=[[0, 100], [200, 300], [400, 500], [600, 700], [1300, 1400]])
    reads = G.collect([u], (50, 1350), 60, 3)
    assert reads[0]['overflow'] == 1 and reads[0]['widest'] == 600
    from fiberhmm.inference.accessibility.variants import depth_states
    _, widest, states = depth_states(reads)
    assert widest[0] == 600 and [int(s['open'][0]) for s in states] == [1, 1, 1]


def test_a_read_without_background_is_dropped_not_the_adjustment():
    def p_open(rng, i):
        o1 = rng.random() < .5
        return o1, rng.random() < (.8 if o1 else .2), 30
    units = two_nfr_units(300, 11, p_open)
    units[0]['raw_nuc_intervals'] = [[950, 1000], [1000, 1090], [1260, 1360]]   # covers only N1: no background bins
    res = run_accessibility(payload(units), dict(FAST, nfr_regions=[(980, 1270), (1980, 2270)]))
    p = pair_of(res, 'N1:V1', 'N2:V1')
    assert p['adjust'] == 'openness x channel'


def test_separation_is_classified_from_the_exact_test():
    x = np.r_[np.ones(50), np.zeros(50)].astype(int)
    els = [dict(id='A', kind='nfr', start=100., end=200., state={f'u{i}': int(v) for i, v in enumerate(x)}),
           dict(id='B', kind='nfr', start=900., end=1000., state={f'u{i}': int(v) for i, v in enumerate(x)})]
    rng = np.random.default_rng(0)
    cov = {f'u{i}': dict(x=np.arange(0, 1200, 10), closed=rng.random(120) < .5, ch='d::CT') for i in range(100)}
    rows, _ = C.pair_table(els, {}, cov, min_reads=10)
    r, = rows
    assert r['separation'] and np.isnan(r['mh']) and r['class'] == 'co-accessible' and r['q'] < 1e-10


def test_kmax_one_caps_discovery_and_depth_honours_whole_nfr_elements():
    units, _ = planted([(.5, [(1400, 1700)]), (.3, [(1470, 1770)]), (.2, [])], n=200, seed=8)
    res = run_accessibility(payload(units), dict(FAST, kmax=1, nfr_regions=[(1380, 1790)]))
    assert res['nfrs'][0]['k'] == 1 and [k for k, _ in res['nfrs'][0]['ps_curve']] == [1]
    res = run_accessibility(payload(units), dict(FAST, mode='depth', elements='nfrs', nfr_regions=[(1380, 1790)]))
    assert [e['id'] for e in res['elements']] == ['N1:open'] and len(res['nfrs'][0]['depth_states']) == 3


def test_detected_regions_stay_inside_the_window_and_bad_parameters_are_refused():
    units = [dict(unit_id=f'u{i}', strand='CT', dataset='d', raw_nuc_intervals=[[0, 100], [1000, 1100]]) for i in range(20)]
    runs = G.detect_nfrs(units, 400, 703)
    assert runs and runs[-1]['end'] <= 703
    for bad in (dict(min_reads=.5), dict(n_perm=0), dict(per_bin=0), dict(splits=0), dict(within_clusters=1),
                dict(kmax=2.5), dict(q_max=float('nan'))):
        with pytest.raises(ValueError):
            NFROptions.from_params(bad)
    assert NFROptions.from_params(dict(kmax=3.0)).kmax == 3


def test_load_recaller_classes_reads_recaller_artifacts(tmp_path):
    from fiberhmm.inference.accessibility import load_recaller_classes
    (tmp_path/'classes.tsv').write_text('class_id\tchannel\tstart\tend\tprevalence\tsupported\n'
                                        'class_001\td::CT\t100.5\t130\t0.4\tTrue\nclass_001\td::GA\t100.5\t130\t0.2\tFalse\n'
                                        'class_002\td::CT\t300\t330\t0.01\tTrue\n')
    with gzip.open(tmp_path/'molecules.tsv.gz', 'wt') as fh:
        fh.write('class_id\tchannel\tunit_id\tposterior\tlabel\n')
        fh.write('class_001\td::CT\tu1\t0.9\tmember\nclass_001\td::CT\tu2\t0.1\tnon_member\n'
                 'class_001\td::CT\tu3\t0.5\tabstain\nclass_001\td::GA\tu4\t0.9\tmember\n')
    c, = load_recaller_classes(tmp_path, prefix='R1:')     # class_002 below the minimum prevalence; GA unsupported
    assert c['id'] == 'R1:class_001' and c['state'] == {'u1': 1, 'u2': 0} and c['channels'] == ['d::CT']
    assert c['start'] == 100.5 and c['kind'] == 'tf'


def test_k_is_chosen_on_full_precision_prediction_strength(monkeypatch):
    """A k whose strength only reaches the stringency after rounding is not chosen (as in the recaller)."""
    from fiberhmm.inference.accessibility import variants as V
    real = V.prediction_strength

    def just_below(X, groups, k, seed, splits):
        ps, detail = real(X, groups, k, seed, splits)
        return (0.89996 if k == 2 else (1.0 if k == 1 else 0.0)), detail
    monkeypatch.setattr(V, 'prediction_strength', just_below)
    units, _ = planted([(.5, [(1400, 1700)]), (.5, [(1470, 1770)])], n=200, seed=8)
    nfr = run_accessibility(payload(units), dict(FAST, bootstrap=0, stringency=.9, nfr_regions=[(1380, 1790)]))['nfrs'][0]
    assert dict(map(tuple, nfr['ps_curve']))[2] == .9      # the diagnostics show it rounded
    assert nfr['k'] == 1


# ---------------------------------------------------------------- NFR region detection (enrichment rule)
def background_locus(n, states, seed, *, missed=.5, span=(200, 2900)):
    """``n`` molecules of planted ``states`` plus, on a share ``missed`` of them, one missed nucleosome somewhere
    on the molecule (a 200-260-bp gap at a random place): the low, flat background of long gaps real data have."""
    rng = np.random.default_rng(seed)
    w = np.array([s[0] for s in states], float); w /= w.sum()
    units = []
    for i in range(n):
        opens = list(states[rng.choice(len(states), p=w)][1])
        if rng.random() < missed:
            a = int(rng.integers(span[0], span[1] - 260)); b = a + int(rng.integers(200, 261))
            if all(b + 60 < c or a > d + 60 for c, d in opens):
                opens.append((a, b))
        units.append(fiber(i, opens, rng, strand='CT' if i % 2 == 0 else 'GA', jitter=3.))
    return units


def test_a_rare_nfr_is_found_at_depth_and_the_fixed_rule_misses_it():
    units = background_locus(1500, [(.5, [(800, 1050)]), (.06, [(1900, 2150)]), (.44, [])], seed=11)
    lo, hi = 400, 2800
    fixed = G.detect_nfrs(units, lo, hi)
    assert [abs(r['start'] - 800) <= 30 for r in fixed] == [True]       # 6% open (+ background) < 15%: missed
    regs, det = G.detect_nfrs_enriched(units, lo, hi)
    assert det['enrichment_test'] and det['note'] == '' and 0.005 <= det['background'] < .05
    assert len(regs) == 2, regs
    common, rare = regs
    assert abs(common['start'] - 800) <= 30 and abs(common['end'] - 1050) <= 30
    assert abs(rare['start'] - 1900) <= 40 and abs(rare['end'] - 2150) <= 40
    assert .06 < rare['peak'] < .15 and rare['fold'] >= 2 and rare['q'] <= .01 and rare['open_reads'] >= 10
    assert rare['background'] == det['background'] and rare['peak_callable'] >= 20 and lo <= rare['peak_at'] < hi
    # the run reports it, with the detection
    res = run_accessibility(payload(units), dict(FAST, bootstrap=0, kmax=2, detect_method='enrichment'))
    assert [n['source'] for n in res['nfrs']] == ['detected', 'detected']
    assert res['detection']['method'] == 'enrichment' and res['nfrs'][1]['detection']['fold'] == rare['fold']
    assert res['nfrs'][1]['profile_peak'] == rare['peak']
    # the fixed rule stays reachable, and a saved detect_threshold alone still means it
    old = run_accessibility(payload(units), dict(FAST, bootstrap=0, kmax=2, detect_threshold=.15))
    assert old['parameters']['detect_method'] == 'fixed' and len(old['nfrs']) == 1 and old['detection']['method'] == 'fixed'


def test_the_rarest_share_and_the_depth_gate_detection():
    units = background_locus(1500, [(.5, [(800, 1050)]), (.06, [(1900, 2150)]), (.44, [])], seed=11)
    regs, _ = G.detect_nfrs_enriched(units, 400, 2800, min_fraction=.2)
    assert len(regs) == 1                                                      # the rare NFR is below the slider
    # the same rare NFR in 80 reads: a handful of open reads is not enough (min_open_reads, the test)
    shallow = background_locus(80, [(.5, [(800, 1050)]), (.06, [(1900, 2150)]), (.44, [])], seed=4)
    regs, _ = G.detect_nfrs_enriched(shallow, 400, 2800)
    assert any(abs(r['start'] - 800) <= 40 for r in regs)                     # the common NFR is still found
    assert all(not (r['start'] < 2150 and r['end'] > 1900) for r in regs)


def test_uniformly_closed_chromatin_calls_nothing():
    units = background_locus(1500, [(1., [])], seed=12, missed=.6)
    regs, det = G.detect_nfrs_enriched(units, 400, 2800)
    assert det['enrichment_test'] and regs == [], regs
    no_gaps = [fiber(i, [], np.random.default_rng(i)) for i in range(200)]
    regs, det = G.detect_nfrs_enriched(no_gaps, 400, 2800)
    assert regs == [] and det['background'] == G.BACKGROUND_FLOOR
    res = run_accessibility(payload(units), dict(FAST, detect_method='enrichment'))
    assert res['nfrs'] == [] and res['warnings'][0].startswith('No NFR region found') and 'background' in res['warnings'][0]


def test_a_mostly_open_window_says_so_and_falls_back_to_the_share():
    units = background_locus(800, [(.8, [(1300, 1900)]), (.2, [])], seed=13, missed=0)
    regs, det = G.detect_nfrs_enriched(units, 1350, 1850)                # the window sits inside the NFR
    assert det['background'] is None and not det['enrichment_test'] and det['background_raw'] > G.BACKGROUND_MAX
    assert 'most of this window is open' in det['note']
    assert len(regs) == 1 and regs[0]['fold'] is None and regs[0]['peak'] > .7          # found, not hidden
    res = run_accessibility(payload(units, window=(1350, 1850)), dict(FAST, bootstrap=0, kmax=2))
    assert len(res['nfrs']) == 1 and any('Background not estimated' in w for w in res['warnings'])
    # a window too short to estimate a background says that instead
    _regs, det = G.detect_nfrs_enriched(units, 1500, 1600)
    assert det['background'] is None and 'positions' in det['note']


def nuc_units(groups, n, lo=0, hi=1400):
    """``n`` molecules tiled with 147-bp nucleosomes and 30-bp linkers over [lo, hi); ``groups`` = [(count, (a, b))]:
    that many molecules (different ones per group) have one nucleosome-free gap [a, b)."""
    units, uid = [], 0
    for count, gap in list(groups) + [(n - sum(c for c, _ in groups), None)]:
        for _ in range(count):
            nucs = _fill(lo, gap[0], 30) + _fill(gap[1], hi, 30) if gap else _fill(lo, hi, 30)
            units.append(dict(unit_id=f'u{uid:04d}', strand='CT', dataset='d', raw_nuc_intervals=[list(v) for v in nucs]))
            uid += 1
    return units


@pytest.mark.parametrize('rare, common', [(20, 40), (40, 20)])
def test_one_continuous_run_of_two_neighbouring_openings_is_split_where_the_openings_change(rare, common):
    # one qualifying run 400-800 made of two populations of openings on different reads (a flat step, no dip)
    units = nuc_units([(rare, (400, 600)), (common, (600, 800))], 200)
    regs, det = G.detect_nfrs_enriched(units, 0, 1400)
    assert det['enrichment_test'] and det['background'] == G.BACKGROUND_FLOOR
    assert [(r['start'], r['end']) for r in regs] == [(400, 600), (600, 800)], regs
    unsplit, _ = G.detect_nfrs_enriched(units, 0, 1400, opening_share=0)
    assert [(r['start'], r['end']) for r in unsplit] == [(400, 800)]          # what the split undoes


def test_a_flat_run_of_three_opening_populations_is_split_into_three():
    units = nuc_units([(20, (200, 400)), (20, (400, 600)), (20, (600, 800))], 200)
    regs, _ = G.detect_nfrs_enriched(units, 0, 1400)
    assert [(r['start'], r['end']) for r in regs] == [(200, 400), (400, 600), (600, 800)], regs


def test_a_shoulder_too_narrow_where_the_window_clips_it_stays_attached():
    units = nuc_units([(40, (200, 400)), (20, (400, 600))], 200)
    regs, _ = G.detect_nfrs_enriched(units, 0, 455)                 # the rare opening is cut to 55 bp by the window
    assert [(r['start'], r['end']) for r in regs] == [(200, 455)], regs
    regs, _ = G.detect_nfrs_enriched(units, 0, 470)                 # 70 bp: its own region
    assert [(r['start'], r['end']) for r in regs] == [(200, 400), (400, 470)], regs


def test_a_common_nfrs_wider_openings_stay_part_of_it():
    units = nuc_units([(60, (400, 600)), (20, (400, 800))], 200)
    regs, _ = G.detect_nfrs_enriched(units, 0, 1400)
    assert [(r['start'], r['end']) for r in regs] == [(400, 800)], regs
    # and with a background of missed nucleosomes
    units = background_locus(1500, [(.5, [(800, 1050)]), (.15, [(800, 1300)]), (.35, [])], seed=15, missed=.3)
    regs, _ = G.detect_nfrs_enriched(units, 400, 2800)
    assert len(regs) == 1 and abs(regs[0]['start'] - 800) <= 30 and abs(regs[0]['end'] - 1300) <= 40, regs


def test_the_fixed_rule_keeps_its_rounding():
    units = nuc_units([(7, (400, 600))], 30)
    r, = G.detect_nfrs(units, 0, 1400)
    assert r['peak'] == 0.233 and r['background'] is None and r['fold'] is None


def test_detection_parameters_are_validated():
    for bad in (dict(detect_method='peaks'), dict(min_fraction=0), dict(min_fraction=1), dict(min_fraction='0.1'),
                dict(min_fraction=True), dict(detect_q=0), dict(detect_q=1.5), dict(min_fold=.5), dict(min_open_reads=0),
                dict(min_open_reads=2.5), dict(min_fold=float('inf'))):
        with pytest.raises(ValueError):
            NFROptions.from_params(bad)
    opt = NFROptions.from_params(dict(min_open_reads=5.0, min_fraction=.02))
    assert opt.detect_method == 'raw' and opt.min_open_reads == 5 and opt.min_fraction == .02
    assert NFROptions.from_params(dict(detect_threshold=.2)).detect_method == 'fixed'
    assert NFROptions.from_params(dict(detect_threshold=.2, detect_method='enrichment')).detect_method == 'enrichment'
    d = NFROptions().as_dict()
    assert (d['detect_method'], d['min_fraction'], d['detect_q'], d['min_fold'], d['min_open_reads']) == ('raw', .10, .01, 2., 10)


def test_cli_detection_flags(tmp_path, capsys):
    from fiberhmm.inference.accessibility.cli import main
    bam = nfr_bam(tmp_path/'nfr.bam')
    base = ['--bam', str(bam), '--region', 'chr1:700-2400', '--bootstrap', '0', '--cores', '1']
    main(base + ['--detect-method', 'enrichment', '--min-fraction', '0.04', '--min-fold', '3', '--detect-q', '0.05', '--min-open-reads', '5', '--output', str(tmp_path/'a')])
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary['detection']['method'] == 'enrichment' and summary['nfrs'] and summary['nfrs'][0]['peak'] > .5
    p = json.loads((tmp_path/'a'/'result.json').read_text())['parameters']
    assert (p['detect_method'], p['min_fraction'], p['min_fold'], p['detect_q'], p['min_open_reads']) == ('enrichment', .04, 3., .05, 5)
    assert json.loads((tmp_path/'a'/'manifest.json').read_text())['detection']['method'] == 'enrichment'
    main(base + ['--detect-threshold', '0.3', '--output', str(tmp_path/'b')])
    capsys.readouterr()
    assert json.loads((tmp_path/'b'/'result.json').read_text())['parameters']['detect_method'] == 'fixed'
    for bad, said in ((['--min-fraction', '0'], 'min_fraction must be in (0, 1)'), (['--min-fold', '0.5'], 'min_fold >= 1'),
                      (['--detect-method', 'peaks'], 'invalid choice')):
        with pytest.raises(SystemExit):
            main(base + bad + ['--output', str(tmp_path/'c')])
        assert said in capsys.readouterr().err
    main(['--schema'])
    assert json.loads(capsys.readouterr().out)['min_fraction'] == .10


def test_a_long_run_of_many_shoulders_does_not_exhaust_the_recursion_limit():
    # two groups of reads with 180-bp gaps out of phase: one continuous run of ~1000 alternating opening populations
    units = [dict(unit_id=f'u{r}', strand='CT', dataset='d',
                  raw_nuc_intervals=[[x, x + 180] for x in range(0 if r < 20 else 180, 180_000, 360)]) for r in range(40)]
    regs, _ = G.detect_nfrs_enriched(units, 0, 180_000)
    assert len(regs) > 900 and all(a['end'] <= b['start'] for a, b in zip(regs, regs[1:]))


@pytest.mark.parametrize('count,n,expected', [(20, 200, True), (19, 200, False), (9, 80, False), (10, 19, False)])
def test_raw_rate_and_count_floors(count, n, expected):
    units = nuc_units([(count, (400, 600))], n)
    regs, det = G.detect_nfrs_raw(units, 0, 1400, min_fold=1000, q=1e-100)
    assert bool(regs) == expected
    assert det['method'] == 'raw' and not det['enrichment_gates']
    if regs:
        assert regs[0]['fold'] < 1000 and regs[0]['q'] > 1e-100
        assert regs[0]['background'] == det['background']
    assert G.detect_nfrs_enriched(units, 0, 1400, min_fold=1000, q=1e-100)[0] == []


def test_raw_splits_and_width_floor():
    units = nuc_units([(20, (400, 600)), (40, (600, 800))], 200)
    regs, _ = G.detect_nfrs_raw(units, 0, 1400)
    assert [(r['start'], r['end']) for r in regs] == [(400, 600), (600, 800)]
    assert G.detect_nfrs_raw(units, 400, 450)[0] == []


@pytest.mark.parametrize('enabled', [False, True])
def test_gap_assignment_before_quantification_and_overflow(enabled):
    units = nuc_units([(20, (400, 800))], 40)
    regions = [(400, 550), (550, 800)]
    res = run_accessibility(payload(units, window=(0, 1400)), dict(
        nfr_regions=regions, clip_gaps_to_region=enabled, mode='depth', bootstrap=0))
    assert res['parameters']['clip_gaps_to_region'] is enabled
    gap_counts = [sum(bool(m['nfr'][n['id']]['gaps']) for m in res['molecules'].values()) for n in res['nfrs']]
    assert gap_counts == ([0, 20] if enabled else [20, 20])
    assigned = G.collect(units, regions[0], 60, 1, regions if enabled else None)
    assert assigned[0]['widest'] == (0 if enabled else 400)
    assert assigned[0]['overflow'] == 0
    # equal overlaps always go to the leftmost region, independent of input order
    tied = [(600, 800), (400, 600)]
    assert G.collect(units, tied[0], 60, 3, tied)[0]['gaps'] == []
    assert len(G.collect(units, tied[1], 60, 3, tied)[0]['gaps']) == 1


@pytest.mark.parametrize('chemistry,strand', [('ddda', 'CT'), ('dddb', 'GA'), ('hia5-pacbio', '+')])
def test_opportunity_diagnostic_counts_unmarked_sites_and_skips_missing(chemistry, strand):
    units = nuc_units([(1, (400, 600)), (1, (800, 1000)), (1, (200, 400)), (1, (1000, 1200))], 4)
    for u in units:
        u.update(chemistry=chemistry, strand=strand)
    units[0].update(positions=[400, 450, 599, 600], hits=[0, 0, 0, 1])
    units[1].update(positions=list(range(800, 1000, 5)), hits=[0]*40)
    units[2].update(positions=[], hits=[])
    det = G.opportunity_diagnostic(units, 300, 1400, [dict(start=800, end=1000)])
    assert det['background'] == dict(gaps=1, median_opportunities_per_100bp=1.5, sparse_share=1.)
    assert det['nfr'] == dict(gaps=1, median_opportunities_per_100bp=20., sparse_share=0.)
    assert det['partial_gaps'] == 1 and det['missing_opportunity_gaps'] == 1
    units[0]['positions'] = []
    assert G.opportunity_diagnostic(units[:1], 0, 1400, [])['background']['median_opportunities_per_100bp'] == 0
    del units[0]['positions']
    assert not G.opportunity_diagnostic(units[:1], 0, 1400, [])['available']


def test_raw_cli_and_saved_enrichment_default(tmp_path, capsys):
    from fiberhmm.inference.accessibility.cli import main
    assert NFROptions.from_params(dict(detect_method='enrichment')).min_fraction == .05
    assert NFROptions(detect_method='enrichment').min_fraction == .05
    with pytest.raises(ValueError, match='boolean'):
        NFROptions.from_params(dict(clip_gaps_to_region='false'))
    units = nuc_units([(20, (400, 800))], 40)
    evidence = tmp_path/'evidence.json'
    evidence.write_text(json.dumps(payload(units, window=(0, 1400))))
    main(['--evidence', str(evidence), '--output', str(tmp_path/'out'), '--mode', 'depth',
          '--clip-gaps-to-region', '--bootstrap', '0'])
    result = json.loads((tmp_path/'out'/'result.json').read_text())
    assert result['parameters']['detect_method'] == 'raw'
    assert result['parameters']['min_fraction'] == .10
    assert result['parameters']['clip_gaps_to_region'] is True
    assert 'opportunity_diagnostic' in result['detection']


def test_gap_assignment_is_saved_for_frozen_transfer():
    from fiberhmm.inference.accessibility import analysis as A
    units, _ = planted([(0.5, [(400, 800)]), (0.5, [])], n=100, lo=0, hi=1400, window=(0, 1400))
    data = payload(units, window=(0, 1400))
    result = run_accessibility(data, dict(nfr_regions=[(400, 550), (550, 800)], clip_gaps_to_region=True,
                                          bootstrap=0, kmax=2))
    left, right = result['nfrs']
    assert not left['variants'] and len(right['variants']) == 1
    assert right['catalogue']['assignment_regions'] == [(400, 550), (550, 800)]
    # JSON roundtrip reproduces assignment when the catalogue is used on another payload.
    result = json.loads(json.dumps(result))
    transferred = A.transfer(data, result, 'N2', bootstrap=0)
    assert transferred['datasets']['d']['variants']['V1']['prevalence'] == right['variants'][0]['prevalence']
    assert A.transfer(data, result, 'N1', bootstrap=0)['datasets']['d']['closed'] == 1.
