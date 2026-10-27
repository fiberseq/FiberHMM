"""M4 pooled overlap and distance enrichment (fiberhmm.assoc.overlap): synthetic data only."""
import json

import numpy as np
import pytest

from fiberhmm.assoc.overlap import (HitSet, _Prep, intersect_territories, overlap_contrast, overlap_enrichment,
                                    simulate_overlap, tercile_strata)


def hs(rows, strand=None):
    chrom = [r[0] for r in rows]
    return HitSet(np.array(chrom), np.array([r[1] for r in rows]), np.array([r[2] for r in rows]),
                  None if strand is None else np.array(strand))


def row(res, d):
    return next(r for r in res['rows'] if r['d'] == d)


# ------------------------------------------------------------------ hand-computed cases

TERR = [('c1', 0, 20000)]
A_ROWS = [('c1', 1000, 1010),      # B1 100 bp downstream: within d only for d > 100
          ('c1', 5000, 5000),      # zero-width -> [5000, 5001); B2 abuts it (gap 0)
          ('c1', 10000, 10100),    # B3 overlaps it
          ('c1', 14000, 14010),    # '-' strand; B4 300 bp downstream in genome coordinates
          ('c1', 19900, 19950),    # 50 bp from the territory end, no B within 5 kb
          ('c1', 19990, 20010)]    # crosses the territory end: dropped
A_STRAND = ['+', '-', '.', '-', '+', '+']
B_ROWS = [('c1', 1110, 1200), ('c1', 5001, 5050), ('c1', 10050, 10060), ('c1', 14310, 14400),
          ('c1', 25000, 25100),    # outside the territory: dropped
          ('c2', 100, 200)]        # contig without territory: dropped


def hand(**kw):
    return overlap_enrichment(hs(A_ROWS, A_STRAND), hs(B_ROWS), TERR, None, distances=(0, 100, 250, 500),
                              n_relocations=50, seed=1, **kw)


def test_hand_fractions_censoring_zero_width_within_d():
    r = hand()
    assert r['status'] == 'ok'
    assert (r['n_a_used'], r['n_a_dropped_outside']) == (5, 1)
    assert (r['n_b_used'], r['n_b_dropped_outside']) == (4, 2)
    assert r['n_clusters'] == 5 and r['n_unplaced'] == 0
    assert r['n_a_zero_width_used'] == 1
    # d = 0: only the overlap counts (abutting is not overlap)
    assert (row(r, 0)['n_hits'], row(r, 0)['n_used'], row(r, 0)['n_censored']) == (1, 5, 0)
    # d = 100: the abutting zero-width hit counts; gap 100 does not (strict); the edge hit is censored
    assert (row(r, 100)['n_hits'], row(r, 100)['n_used'], row(r, 100)['n_censored']) == (2, 4, 1)
    assert row(r, 100)['observed'] == pytest.approx(0.5)
    assert (row(r, 250)['n_hits'], row(r, 250)['n_used']) == (3, 4)
    assert (row(r, 500)['n_hits'], row(r, 500)['n_used']) == (4, 4)
    assert row(r, 500)['observed'] == pytest.approx(1.0)
    json.dumps(r, allow_nan=False)


def test_hand_signed_distance_histogram_strand():
    r = hand(return_hits=True)
    dist = {h['start']: h['signed_distance'] for h in r['hits']}
    assert dist[1000] == 100          # nearest B downstream
    assert dist[10000] == 0           # overlap
    assert dist[14000] == -300        # '-' strand: genome-downstream is upstream
    assert dist[19900] == -(19900 - 14400)
    assert dist[5000] == 0            # abutting
    assert next(h for h in r['hits'] if h['start'] == 5000)['zero_width'] is True
    h = r['histogram']
    edges = np.array(h['edges'])
    obs = np.array(h['observed'])
    assert obs[np.searchsorted(edges, -300, 'right') - 1] == 1
    assert obs[np.searchsorted(edges, 100, 'right') - 1] == 3   # +100, two zeros
    assert h['observed_underflow'] == 1 and h['observed_overflow'] == 0
    r2 = overlap_enrichment(hs(A_ROWS, A_STRAND), hs(B_ROWS), TERR, None, distances=(0,), n_relocations=10,
                            strand_aware=False, return_hits=True)
    assert {h['start']: h['signed_distance'] for h in r2['hits']}[14000] == 300


# ------------------------------------------------------------------ relocation geometry

def test_clusters_move_rigidly_inside_stratum_and_territory():
    sim = simulate_overlap(3, n_contigs=2, contig_len=500_000, n_a=300, n_b=1000)
    prep = _Prep(sim['a'], sim['territory'], sim['strata'], (0, 1000), 1000, True)
    prep.keep()
    rng = np.random.default_rng(0)
    rel0 = prep.a0 - prep.c0[prep.cid]
    n_checked = 0
    for a0, a1, room in prep.relocations(rng, 30):            # kept apart (default)
        s = a0[:, prep.c_first]
        e = s + prep.span[None, :]
        # rigid: every hit keeps its offset from the cluster start and its length
        assert np.array_equal(a0 - s[:, prep.cid], np.broadcast_to(rel0, a0.shape))
        assert np.array_equal(a1 - a0, np.broadcast_to(prep.a1 - prep.a0, a0.shape))
        for c in range(prep.nk):
            k = np.searchsorted(prep.r0, s[:, c], 'right') - 1
            assert np.all(k >= 0)
            assert np.all(e[:, c] <= prep.r1[k])                    # whole span inside one maximal run
            assert np.all(prep.rlab[k] == prep.clabel[c])           # of the cluster's own stratum
            t = prep.rtid[k]
            assert np.all((s[:, c] >= prep.frame.t0[t]) & (e[:, c] <= prep.frame.t1[t]))
            assert np.all(prep.frame.tcontig[t] == prep.ccontig[c])  # same contig
            n_checked += len(k)
        assert np.all(room >= 0)
        # kept apart: in every relocation, each cluster starts more than cluster_gap after all earlier ones end
        o = np.argsort(s, axis=1)
        ss, ee = np.take_along_axis(s, o, 1), np.take_along_axis(e, o, 1)
        assert np.all(ss[:, 1:] - np.maximum.accumulate(ee, axis=1)[:, :-1] > 1000)
    assert n_checked == 30 * prep.nk
    assert prep.n_capped == 0


def test_kept_apart_cap_refuses_and_can_be_turned_off():
    terr = [('c1', 0, 20_000)]
    a = hs([('c1', k * 3000, k * 3000 + 1000) for k in range(6)])      # six 1-kb clusters, crowded territory
    b = hs([('c1', 500, 520), ('c1', 9000, 9020)])
    r = overlap_enrichment(a, b, terr, None, distances=(0, 500), n_relocations=100, max_redraw_rounds=0)
    assert r['status'] == 'refused' and r['global_p'] is None and r['rows'] == []
    assert r['n_relocations_capped'] > 1 and 'kept_apart=False' in r['message']
    assert r['settings']['kept_apart'] is True
    c = overlap_contrast(a, {'x': b, 'y': b}, terr, None, d=0, n_relocations=100, n_boot=50, max_redraw_rounds=0)
    assert c['status'] == 'refused' and c['contrasts'] == []
    off = overlap_enrichment(a, b, terr, None, distances=(0, 500), n_relocations=100, kept_apart=False)
    assert off['status'] == 'ok' and off['n_relocations_capped'] == 0 and off['settings']['kept_apart'] is False
    ok = overlap_enrichment(a, b, terr, None, distances=(0, 500), n_relocations=100)
    assert ok['status'] == 'ok' and ok['n_relocations_capped'] <= 1


def test_relocation_draws_are_uniform_over_admissible_starts():
    from scipy.stats import chisquare
    terr = [('c1', 0, 100_000)]
    strata = [('c1', 0, 1000, 0), ('c1', 1000, 10_000, 1), ('c1', 10_000, 13_000, 0), ('c1', 13_000, 50_000, 1),
              ('c1', 50_000, 60_000, 0), ('c1', 60_000, 100_000, 1)]
    prep = _Prep(hs([('c1', 50_100, 50_600)]), terr, strata, (0,), 1000, True)   # one cluster, span 500, stratum 0
    prep.keep()
    s, _ = prep.draw(np.random.default_rng(0), np.zeros((30_000, 1), np.int64))
    s = s[:, 0] - prep.frame.offset['c1']
    runs = [(0, 1000), (10_000, 13_000), (50_000, 60_000)]
    inside = [(s >= a) & (s <= b - 500) for a, b in runs]
    assert np.all(np.any(inside, axis=0))                                    # every start admissible
    counts = np.array([m.sum() for m in inside])
    expect = np.array([b - a - 500 + 1 for a, b in runs], float)
    assert chisquare(counts, expect / expect.sum() * counts.sum()).pvalue > 1e-3
    big = s[inside[2]] - 50_000                                               # uniform within the longest run
    assert chisquare(np.histogram(big, bins=10, range=(0, 9501))[0]).pvalue > 1e-3
    assert big.min() >= 0 and big.max() <= 9500


def test_too_many_unplaced_clusters_refuses():
    terr = [('c1', 0, 100_000)]
    strata = [('c1', s, s + 500, (s // 500) % 2) for s in range(0, 100_000, 500)]
    rows = []
    for s in range(2000, 90_000, 5000):          # 2-hit clusters spanning 2 kb; no run is longer than 500 bp
        rows += [('c1', s, s + 10), ('c1', s + 990, s + 1000), ('c1', s + 1990, s + 2000)]
    r = overlap_enrichment(hs(rows), hs([('c1', 50, 60)]), terr, strata, n_relocations=20)
    assert r['status'] == 'refused'
    assert r['global_p'] is None and r['rows'] == []
    assert str(r['n_unplaced']) in r['message'] and 'strata=None' in r['message']
    assert r['suggestions']


def test_few_unplaced_clusters_are_excluded_and_counted():
    terr = [('c1', 0, 100_000)]
    strata = [('c1', 0, 50_000, 0), ('c1', 50_000, 52_000, 1), ('c1', 52_000, 100_000, 0)]
    rows = [('c1', s, s + 50) for s in list(range(1000, 48_000, 2500)) + list(range(55_000, 99_000, 4000))]
    rows += [('c1', 50_000, 50_100), ('c1', 51_000, 51_100), ('c1', 51_900, 52_100)]   # stratum 1, span 2.1 kb
    b = hs([('c1', s, s + 20) for s in range(500, 100_000, 3000)])
    r = overlap_enrichment(hs(rows), b, terr, strata, n_relocations=50)
    assert r['status'] == 'ok'
    assert r['n_unplaced'] == 1 and r['n_clusters'] == len(rows) - 2
    assert r['n_a_excluded_unplaced'] == 3
    assert r['n_a_used'] == len(rows) - 3
    assert 'excluded' in r['message']
    assert r['global_p'] is not None


# ------------------------------------------------------------------ determinism and calibration sanity

def test_deterministic_for_a_seed():
    sim = simulate_overlap(4, n_contigs=2, contig_len=500_000, n_a=200, n_b=1500)
    args = (sim['a'], sim['b'], sim['territory'], sim['strata'])
    r1 = overlap_enrichment(*args, n_relocations=100, seed=7)
    r2 = overlap_enrichment(*args, n_relocations=100, seed=7)
    r3 = overlap_enrichment(*args, n_relocations=100, seed=8)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    assert [x['null_mean'] for x in r1['rows']] != [x['null_mean'] for x in r3['rows']]
    r4 = overlap_enrichment(*args, n_relocations=100, seed=7, kept_apart=False)
    assert [x['null_mean'] for x in r1['rows']] != [x['null_mean'] for x in r4['rows']]
    c1 = overlap_contrast(sim['a'], {'g': sim['b'], 'h': sim['b']}, sim['territory'], sim['strata'], d=500,
                          n_relocations=50, n_boot=50, seed=3)
    c2 = overlap_contrast(sim['a'], {'g': sim['b'], 'h': sim['b']}, sim['territory'], sim['strata'], d=500,
                          n_relocations=50, n_boot=50, seed=3)
    assert json.dumps(c1, sort_keys=True) == json.dumps(c2, sort_keys=True)
    s2 = simulate_overlap(4, n_contigs=2, contig_len=500_000, n_a=200, n_b=1500)
    assert np.array_equal(s2['a'].start, sim['a'].start)


def test_simulator_separated_clusters_are_recovered_by_linkage():
    sim = simulate_overlap(5, n_contigs=2, contig_len=500_000, n_a=600, n_b=500, min_separation=1000)
    r = overlap_enrichment(sim['a'], sim['b'], sim['territory'], sim['strata'], n_relocations=10)
    assert r['n_a_dropped_outside'] == 0
    assert r['n_clusters'] == sim['truth']['n_a_clusters']
    free = simulate_overlap(5, n_contigs=2, contig_len=500_000, n_a=600, n_b=500)
    rf = overlap_enrichment(free['a'], free['b'], free['territory'], free['strata'], n_relocations=10)
    assert rf['n_clusters'] < free['truth']['n_a_clusters']      # independent placement merges some by chance


def test_planted_effect_gives_small_global_p():
    sim = simulate_overlap(21, n_contigs=2, contig_len=1_000_000, n_a=500, n_b=300, planted=0.5)
    r = overlap_enrichment(sim['a'], sim['b'], sim['territory'], sim['strata'], n_relocations=200, seed=0)
    assert r['global_p'] <= 1 / 201 + 1e-12
    assert row(r, 500)['o_over_e'] > 1.3


def test_null_simulation_seed_31_has_non_extreme_p():
    sim = simulate_overlap(31, n_contigs=2, contig_len=1_000_000, n_a=500, n_b=2000)
    r = overlap_enrichment(sim['a'], sim['b'], sim['territory'], sim['strata'], n_relocations=200, seed=0)
    assert r['global_p'] > 0.01
    # the same confounded null analysed without strata is far off
    r0 = overlap_enrichment(sim['a'], sim['b'], sim['territory'], None, n_relocations=200, seed=0)
    assert r0['global_p'] < r['global_p']


def test_direction_a_to_b_differs_from_b_to_a():
    terr = [('c1', 0, 2_000_000)]
    a_rows = [('c1', s, s + 50) for s in range(10_000, 1_000_000, 50_000)]
    b_rows = [('c1', r[1] + 20, r[1] + 40) for r in a_rows]
    b_rows += [('c1', s, s + 30) for s in range(1_000_000, 1_990_000, 2_000)]
    ab = overlap_enrichment(hs(a_rows), hs(b_rows), terr, None, distances=(0,), n_relocations=100)
    ba = overlap_enrichment(hs(b_rows), hs(a_rows), terr, None, distances=(0,), n_relocations=100)
    assert ab['direction'] == 'A->B'
    assert row(ab, 0)['observed'] == pytest.approx(1.0)
    assert row(ba, 0)['observed'] == pytest.approx(len(a_rows) / len(b_rows))
    assert ab['n_a_used'] == len(a_rows) and ba['n_a_used'] == len(b_rows)


# ------------------------------------------------------------------ group contrast

def test_contrast_detects_planted_group_difference():
    sim = simulate_overlap(41, n_contigs=4, contig_len=1_000_000, n_a=800, n_b=400, n_groups=2,
                           group_planted=0.8)
    c = overlap_contrast(sim['a'], sim['b_by_group'], sim['territory'], sim['strata'], d=(0, 500),
                         n_relocations=200, n_boot=400, seed=0)
    assert c['status'] == 'ok' and c['groups'] == ['g0', 'g1']
    r = next(x for x in c['contrasts'] if x['d'] == 500)
    assert r['group'] == 'g1' and r['reference'] == 'g0'
    assert r['log_oe_diff'] > 0 and r['ci_low'] > 0 and r['p'] < 0.01
    json.dumps(c, allow_nan=False)


def test_contrast_equal_groups_seed_42_not_significant():
    sim = simulate_overlap(42, n_contigs=4, contig_len=1_000_000, n_a=800, n_b=400, n_groups=2)
    c = overlap_contrast(sim['a'], sim['b_by_group'], sim['territory'], sim['strata'], d=500,
                         n_relocations=200, n_boot=400, seed=0)
    r = c['contrasts'][0]
    assert r['p'] > 0.05 and r['ci_low'] < 0 < r['ci_high']


def test_contrast_all_pairs():
    sim = simulate_overlap(43, n_contigs=1, contig_len=1_000_000, n_a=300, n_b=1000, n_groups=3)
    c = overlap_contrast(sim['a'], sim['b_by_group'], sim['territory'], sim['strata'], d=0,
                         n_relocations=50, n_boot=100, seed=0, pairs='all')
    assert [(x['reference'], x['group']) for x in c['contrasts']] == [('g0', 'g1'), ('g0', 'g2'), ('g1', 'g2')]


# ------------------------------------------------------------------ inputs and helpers

def test_hitset_from_scan_rows():
    rows = [dict(chrom='chr1', start='100', end='150', strand='-', hit_id='h1', type='x'),
            dict(chrom='chr1', start='300', end='300', strand='', hit_id='h2', type='x'),
            dict(chrom='chr2', start='5', end='9', strand='+', hit_id='h3', type='x')]
    h = HitSet.from_rows(rows)
    assert h.start.dtype == np.int64 and list(h.start) == [100, 300, 5] and list(h.end) == [150, 300, 9]
    assert list(h.strand) == ['-', '.', '+'] and list(h.ids) == ['h1', 'h2', 'h3']
    assert list(h.chrom) == ['chr1', 'chr1', 'chr2']
    h2 = HitSet.from_rows([dict(chrom='c', start=1, end=2)])
    assert h2.strand is None and h2.ids is None


def test_intersect_territories_and_terciles():
    t = intersect_territories([('c1', 0, 100), ('c1', 100, 200), ('c2', 0, 50)], [('c1', 50, 150), ('c3', 0, 9)])
    assert t == [('c1', 50, 150)]
    terr = [('c1', 0, 90_000)]
    cov = [('c1', i * 10_000, (i + 1) * 10_000, float(i)) for i in range(9)]
    den = [('c1', i * 10_000, (i + 1) * 10_000, float(i % 3)) for i in range(9)]
    runs = tercile_strata(terr, cov, den, bin_bp=10_000)
    assert [r[3] for r in runs] == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert runs[0][:3] == ('c1', 0, 10_000)
    coarse = tercile_strata(terr, cov, None, bin_bp=10_000)
    assert [(r[1], r[2], r[3]) for r in coarse] == [(0, 30_000, 0), (30_000, 60_000, 1), (60_000, 90_000, 2)]
    flat = tercile_strata(terr, [('c1', 0, 90_000, 1.0)], None, bin_bp=10_000)   # ties: one tercile
    assert flat == [('c1', 0, 90_000, 0)]


# ------------------------------------------------------------------ Codex round 1 regressions

def test_contrast_keeps_infinite_bootstrap_draws():
    terr = [('c', 0, 100_000)]
    a = hs([('c', 2000 + 9000 * i, 2010 + 9000 * i) for i in range(10)])
    groups = {'ref': hs([('c', 0, 100_000)]), 'grp': hs([('c', 1900, 2900)])}    # only one A cluster near grp's B
    c = overlap_contrast(a, groups, terr, None, d=0, n_relocations=200, n_boot=500, seed=0)
    r = c['contrasts'][0]
    assert r['p'] > 0.3                              # dropping the -inf draws gave p ~ 0.003
    assert r['p'] == r['p_percentile'] and r['ci_low'] == '-inf'
    assert r['n_boot_infinite'] > 50                 # resamples without the one associated cluster
    assert r['p_normal'] is None                     # bootstrap variance undefined
    json.dumps(c, allow_nan=False)


def test_contrast_normal_interval():
    h = HitSet(np.array(['c', 'c']), np.array([100, 3000]), np.array([110, 3010]))
    c = overlap_contrast(h, {'x': h, 'y': h}, [('c', 0, 10_000)], d=0, n_relocations=20, n_boot=20,
                         interval='normal')
    r = c['contrasts'][0]
    assert r['p'] is None and r['p_normal'] is None and 'undefined' in r['note']    # zero bootstrap SD
    sim = simulate_overlap(41, n_contigs=4, contig_len=1_000_000, n_a=800, n_b=400, n_groups=2, group_planted=0.8)
    c = overlap_contrast(sim['a'], sim['b_by_group'], sim['territory'], sim['strata'], d=500, n_relocations=100,
                         n_boot=200, seed=0, interval='normal')
    r = c['contrasts'][0]
    assert r['p'] == r['p_normal'] < 0.01
    assert r['ci_high'] - r['log_oe_diff'] == pytest.approx(r['log_oe_diff'] - r['ci_low'])     # symmetric
    assert 0 < r['ci_low'] < r['log_oe_diff'] < r['ci_high']


@pytest.mark.parametrize('strata', [[('c', 10_000, 20_000, 0)], []])
def test_no_admissible_stratum_run_refuses(strata):
    h = hs([('c', 100, 110)])
    r = overlap_enrichment(h, h, [('c', 0, 10_000)], strata, n_relocations=10)
    assert r['status'] == 'refused' and r['n_unplaced'] == 1 and r['global_p'] is None
    c = overlap_contrast(h, {'x': h, 'y': h}, [('c', 0, 10_000)], strata, d=0, n_relocations=10, n_boot=10)
    assert c['status'] == 'refused'


def test_nearest_distance_stays_on_the_contig():
    terr = [('c1', 0, 10_000), ('c2', 0, 10_000)]
    r = overlap_enrichment(hs([('c2', 100, 110)]), hs([('c1', 100, 110)]), terr, None, distances=(0,),
                           n_relocations=10, return_hits=True)
    assert r['hits'][0]['signed_distance'] is None
    assert r['histogram']['observed_overflow'] == 1 and r['histogram']['observed_underflow'] == 0
    assert r['histogram']['null_mean_underflow'] == 0


def test_capped_relocations_are_left_out_of_the_null():
    terr = [('c', 0, 100_000)]
    a = hs([('c', 100, 200), ('c', 5000, 5100)])
    b = hs([('c', 100, 2000)])
    prep = _Prep(a, terr, None, (0,), 1000, True)
    prep.keep()
    n_rows = 0
    for a0, _, _ in prep.relocations(np.random.default_rng(0), 100, True, 0):
        s = a0[:, prep.c_first]
        o = np.argsort(s, axis=1)
        ss, ee = np.take_along_axis(s, o, 1), np.take_along_axis(s + prep.span[None, :], o, 1)
        assert np.all(ss[:, 1:] - np.maximum.accumulate(ee, axis=1)[:, :-1] > 1000)   # no capped draw yielded
        n_rows += len(a0)
    assert n_rows == 100 - prep.n_capped
    r = overlap_enrichment(a, b, terr, None, distances=(0,), seed=0, n_relocations=100, max_redraw_rounds=0)
    assert r['status'] == 'ok' and r['n_relocations_capped'] == 1
    assert r['n_relocations_effective'] == 99
    k = r['global_p'] * (r['n_relocations_effective'] + 1)
    assert k == pytest.approx(round(k))              # p on the grid of the effective N
