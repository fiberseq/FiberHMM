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
    params = dict(FAST, nfr_regions=[(780, 1020), (1380, 1720), (2080, 2320)], elements='nfrs')
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
