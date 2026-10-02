"""EXPERIMENTAL NFR preview: the analysis views of a finished run (fiberhmm.inference.accessibility.analysis).

The read split of a pair must be the pair test's 2x2 table (same eligibility code), per-read openness rebuilt from the
stored nucleosome calls must be the run's, the V-plot matrices follow their definitions, and the frozen catalogue
reproduces the run's prevalence.
"""
import copy

import numpy as np
import pytest

from fiberhmm.inference.accessibility import analysis as A
from fiberhmm.inference.accessibility import run_accessibility
from fiberhmm.inference.accessibility.workflow import classes_from_rows

from test_accessibility_nfr import FAST, fiber, payload, planted

WINDOW = (1500, 2700)


@pytest.fixture(scope='module')
def two_nfrs_and_a_class():
    """Reads 100-4200 bp (well past the 1500-2700 window, so stored nucleosome calls are clipped), two NFRs that open
    together, a footprint class bound mostly when N2 is open, DAF strands alternating."""
    rng = np.random.default_rng(31)
    units, states = [], []
    for i in range(500):
        z = rng.random() < .5
        o1 = rng.random() < (.75 if z else .3)
        o2 = rng.random() < (.8 if o1 else .25)
        linker = 45 if z else 20
        opens = ([(1700, 1950)] if o1 else []) + ([(2300, 2550)] if o2 else [])
        units.append(fiber(i, opens, rng, lo=100, hi=4200, linker=linker, strand='CT' if i % 2 == 0 else 'GA', window=WINDOW))
        states.append((o1, o2))
    member = {u['unit_id']: int(rng.random() < (.85 if s[1] else .1)) for u, s in zip(units, states)}
    sup = {('class_003', 'd::CT'): dict(start=2580, end=2610, prevalence=.4), ('class_003', 'd::GA'): dict(start=2580, end=2610, prevalence=.4)}
    st = {'class_003': {u['unit_id']: (member[u['unit_id']], f"d::{u['strand']}") for u in units[:460]}}   # 40 abstain
    res = run_accessibility(payload(units, window=WINDOW), dict(FAST, nfr_regions=[(1680, 1970), (2280, 2570)], pairs='all'),
                            classes=classes_from_rows(sup, st))
    return res, units


def test_schema_v1_stores_what_the_views_need(two_nfrs_and_a_class):
    res, units = two_nfrs_and_a_class
    assert res['schema'] == 'fiberhmm.accessibility.preview.v1' and A.has_context(res)
    m = res['molecules'][units[0]['unit_id']]
    assert m['span'] == [100, 4200] and m['cov'][0] >= WINDOW[0] and m['cov'][1] <= WINDOW[1]
    assert all(b > WINDOW[0] - 1000 and a < WINDOW[1] + 1000 for a, b in m['nucs'])
    assert min(a for a, _ in m['nucs']) < WINDOW[0] - 500       # calls kept past the window (phasing)
    n1 = res['nfrs'][0]
    assert n1['catalogue']['variants'][0]['name'] == 'V1' and len(n1['catalogue']['variants'][0]['cov']) == 2


def test_split_is_the_pair_tables_reads_for_every_pair(two_nfrs_and_a_class):
    res, units = two_nfrs_and_a_class
    tested = res['pairs']
    assert any(p['kind_b'] == 'tf' or p['kind_a'] == 'tf' for p in tested)
    for p in tested:
        s = A.pair_split(res, p['b'], p['a'])           # either order: oriented like the tested row
        assert (s['a'], s['b']) == (p['a'], p['b'])
        assert s['table'] == p['table'], (p['a'], p['b'])
        everyone = set().union(*map(set, s['cells'].values())) | set(s['not_informative'])
        assert everyone == set(res['molecules']) and sum(s['table']) + len(s['not_informative']) == len(res['molecules'])
        # openness rebuilt from the stored nucleosome calls gives the run's strata, hence its MH effect
        if p['mh'] is not None:
            assert s['recomputed']['mh'] == pytest.approx(p['mh'], abs=1e-3)
        assert sum(r['n'] for r in s['strata']) == p['n']
        assert sum(sum(r['table']) for r in s['strata']) == p['n']
    p = next(p for p in tested if p['kind_a'] == 'tf' or p['kind_b'] == 'tf')
    s = A.pair_split(res, p['a'], p['b'])
    assert len(s['excluded'].get('not_spanning_a' if p['kind_a'] == 'tf' else 'not_spanning_b', [])) >= 40   # class abstentions
    assert s['spacing']['kind'] == 'class'


def test_split_of_two_nfrs_counts_shared_reads_and_the_spacing(two_nfrs_and_a_class):
    res, _ = two_nfrs_and_a_class
    p = A.find_pair(res, 'N1:V1', 'N2:V1')
    s = A.pair_split(res, 'N1:V1', 'N2:V1')
    assert len(s['excluded'].get('shared', [])) == p['shared']
    sp = s['spacing']
    assert sp['kind'] == 'between' and 330 <= sp['median'] <= 370      # 1950 -> 2300: about two nucleosomes
    assert 'nucleosome' in sp['note']
    prof = s['profiles']
    i = min(range(len(prof['11']['x'])), key=lambda j: abs(prof['11']['x'][j] - 1825))
    assert prof['11']['accessibility'][i] > .9 and prof['01']['accessibility'][i] < .2


def test_skipped_and_overlapping_pairs():
    units, _ = planted([(.4, [(1400, 1700)]), (.3, [(1400, 1520), (1620, 1700)]), (.3, [])], n=300, seed=5)
    res = run_accessibility(payload(units), dict(FAST, nfr_regions=[(1380, 1720)]))
    vs = {(round(v['L'], -1), round(v['R'], -1)): v['id'] for v in res['nfrs'][0]['variants']}
    full = next(i for (a, b), i in vs.items() if b - a > 250)
    left = next(i for (a, b), i in vs.items() if b - a < 150 and a < 1450)
    with pytest.raises(ValueError, match='overlap'):
        A.pair_split(res, full, left)


def test_combo_split_matches_the_combination_table():
    units, _ = planted([(.4, [(1000, 1250)]), (.3, [(1000, 1250), (2000, 2250)]), (.3, [(1500, 1650), (2000, 2250)])], n=400, seed=14)
    res = run_accessibility(payload(units), dict(FAST, nfr_regions=[(980, 1270), (1480, 1670), (1980, 2270)], elements='nfrs'))
    combos = res['combos']
    assert combos and combos['status'] == 'ok'
    s = A.combo_split(res, combos['elements'])
    assert s['n'] == combos['n']
    obs = {p['pattern']: p['obs'] for p in combos['patterns']}
    for g in s['patterns']:
        assert len(g['reads']) == obs[g['pattern']]


def test_coverage_vplot_adds_the_opening_at_every_position_it_covers():
    c, cv = A.vplot_matrices([(1003, 1218)], 900, 1400, 60, 400, 10, 10)
    row = (215 - 60)//10
    assert c.sum() == 1 and c[row, (1110 - 900)//10] == 1          # centre 1110.5 -> bin 21
    assert cv[row].sum() == pytest.approx(21.5) and cv.sum() == pytest.approx(21.5)   # width / step
    assert cv[row, (1003 - 900)//10] == pytest.approx(.7) and cv[row, (1100 - 900)//10] == 1
    assert c.shape == cv.shape == (34, 50)
    c, cv = A.vplot_matrices([(1003, 1218), (950, 1000), (1300, 1800)], 900, 1400, 60, 400, 10, 10)
    assert c.sum() == 1 and cv.sum() == pytest.approx(21.5)          # 50 bp: below the size range; 500 bp: above


def test_size_shape_edges_and_boundary_nucleosomes():
    units, pick = planted([(.5, [(1400, 1700)]), (.5, [(1470, 1770)])], n=300, seed=8)
    res = run_accessibility(payload(units), dict(FAST, nfr_regions=[(1380, 1790)]))
    vs = res['nfrs'][0]['variants']
    shifted = next(v for v in vs if v['L'] > 1440)
    s = A.size_shape(res, shifted['id'])
    assert s['edges']['total'] == sum(len(m['nfr']['N1']['gaps']) for m in res['molecules'].values() if m['nfr']['N1']['status'] == 'callable')
    assert sum(map(sum, s['edges']['density']['counts'])) == s['edges']['total']
    w = s['sizes']['variants'][shifted['name']]['summary']
    assert 280 <= w['median'] <= 320                                     # same width: a shift, not an expansion
    b = s['boundary'][shifted['name']]
    assert abs(b['minus1']['median'] - (1470 - 73.5)) < 15 and abs(b['plus1']['median'] - (1770 + 73.5)) < 15
    assert 140 <= b['minus1_size']['median'] <= 150


def test_vplots_of_a_variant_and_of_all_openings():
    units, _ = planted([(.5, [(1400, 1700)]), (.5, [(1480, 1620)])], n=200, seed=8)
    res = run_accessibility(payload(units), dict(FAST, nfr_regions=[(1380, 1720)]))
    core = next(v for v in res['nfrs'][0]['variants'] if v['width'] < 200)
    vp = A.vplots(res, core['id'])
    allc = np.array(vp['all']['centre']); mine = np.array(vp['own']['centre'])
    assert vp['variant'] == core['id']
    assert allc.sum() == vp['counts']['all'] and mine.sum() == vp['counts']['own'] < vp['counts']['all']
    assert np.array(vp['own']['coverage']).shape == allc.shape
    rows = np.where(mine.sum(1) > 0)[0]
    sizes = vp['size'][0] + rows*vp['size_step']
    assert sizes.min() >= 110 and sizes.max() <= 170


def test_frozen_catalogue_reproduces_the_run_and_splits_by_group():
    states = [(.5, [(1400, 1700)]), (.5, [])]
    units, pick = planted(states, n=400, seed=12)
    res = run_accessibility(payload(units), dict(FAST, nfr_regions=[(1380, 1720)]))
    v = res['nfrs'][0]['variants'][0]
    uid = {u['unit_id']: j for u, j in zip(units, pick)}
    g = A.group_prevalence(res, v['id'], {'open': [u for u, j in uid.items() if j == 0], 'closed': [u for u, j in uid.items() if j == 1]},
                           bootstrap=FAST['bootstrap'])
    rows = {r['group']: r for r in g['groups']}
    assert rows['all']['prevalence'] == pytest.approx(v['prevalence'], abs=1e-4)
    assert rows['all']['ci'] == pytest.approx(v['ci'], abs=1e-4)          # same EM, same bootstrap seed
    assert rows['open']['prevalence'] > .95 and rows['closed']['prevalence'] < .05
    assert rows['open']['callable'] + rows['closed']['callable'] == rows['all']['callable']
    t = A.transfer(payload(units), res, 'N1', bootstrap=FAST['bootstrap'])
    assert t['datasets']['d']['variants'][v['name']]['prevalence'] == pytest.approx(v['prevalence'], abs=1e-4)


def test_phasing_and_profiles():
    units, _ = planted([(.5, [(1400, 1700)]), (.5, [])], n=300, seed=12)
    res = run_accessibility(payload(units), dict(FAST, nfr_regions=[(1380, 1720)]))
    v = res['nfrs'][0]['variants'][0]
    ph = A.phasing(res, v['id'], flank=400)
    x = np.array(ph['members']['x']); mid = np.argmin(abs(x - 1550))
    assert ph['members']['occupancy'][mid] < .05 and ph['others']['occupancy'][mid] > .6
    assert ph['members']['reads'] + ph['others']['reads'] == res['nfrs'][0]['callable']
    pr = A.variant_profiles(res, 'N1')
    i = np.argmin(abs(np.array(pr['x']) - 1550))
    assert pr['variants'][v['name']]['profile'][i] > .95 and .3 < pr['all']['profile'][i] < .7


def test_v0_results_still_give_the_views_that_need_only_gaps():
    units, _ = planted([(.5, [(1400, 1700)]), (.5, [])], n=200, seed=12)
    res = run_accessibility(payload(units), dict(FAST, nfr_regions=[(1380, 1720)]))
    old = copy.deepcopy(res)
    old['schema'] = 'fiberhmm.accessibility.preview.v0'
    old.pop('element_states')
    for n in old['nfrs']:
        n.pop('catalogue', None)
    for m in old['molecules'].values():
        for k in ('span', 'nucs', 'cov'):
            m.pop(k, None)
        for rec in m['nfr'].values():
            rec.pop('edges', None)
    vid = old['nfrs'][0]['variants'][0]['id']
    assert A.variant_profiles(old, 'N1')['all']['n'] == 200
    assert A.vplots(old, vid)['counts']['all'] > 0
    assert A.size_shape(old, vid)['boundary'] is None
    for fn in (lambda: A.pair_split(old, vid, vid), lambda: A.phasing(old, vid), lambda: A.group_prevalence(old, vid, {})):
        with pytest.raises(A.NeedsRerun):
            fn()
