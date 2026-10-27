"""NUC 13.1: the regular default (posterior OR rule), disordered versus
undetermined, undersized runs, missing nucleosomes, the recurrence test and
dephasing at anchors. Synthetic molecules only."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from nuc_synth import array_molecule, complement_msps, flat_calibration

from fiberhmm.nuc.disrupted import (MISSING, UNDERSIZED, maximal_segments, missing_nucleosomes,
                                    rule_runs, undersized_runs)
from fiberhmm.nuc.enrichment import MeasuredTrack, bh, merge_windows, window_test
from fiberhmm.nuc.molecule import molecule_outputs
from fiberhmm.nuc.params import NucParams
from fiberhmm.nuc.records import make_molecule
from fiberhmm.nuc.segment import DISORDERED, LOWINFO, REGULAR, UNDETERMINED


def _partition_ok(segs):
    segs = sorted(segs, key=lambda s: (s.start, s.end))
    return all(a.end <= b.start for a, b in zip(segs, segs[1:]))


# --- maximal segments ---------------------------------------------------------------

def test_maximal_segments_examples_and_kadane():
    assert maximal_segments([2, -1, 2, -5, 3]) == [(0, 3, 3.0), (4, 5, 3.0)]
    assert maximal_segments([-1, 4, -2, -2, 5, -10, 1]) == [(1, 2, 4.0), (4, 5, 5.0), (6, 7, 1.0)]
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = rng.normal(-0.3, 1.0, size=int(rng.integers(1, 30)))
        segs = maximal_segments(x.tolist())
        best = max((x[i:j].sum() for i in range(len(x)) for j in range(i + 1, len(x) + 1)), default=0.0)
        if best > 0:
            assert max(s for _a, _b, s in segs) == pytest.approx(best)
        assert all(a < b and s > 0 for a, b, s in segs)
        assert all(s1[1] <= s2[0] for s1, s2 in zip(segs, segs[1:]))


# --- regular default ---------------------------------------------------------------------

def test_rule_alone_calls_regular_and_segments_stay_a_partition():
    rng = np.random.default_rng(4)
    m = array_molecule([('irr', 12, 0), ('reg', 14, 190), ('irr', 12, 0)], rng)
    cal = flat_calibration()
    never = NucParams(seg_posterior_min=1.01)            # the decoder never says regular
    r = molecule_outputs(m, never, cal)
    regs = [s for s in r.segments if s.state == REGULAR]
    assert regs and all(s.source == 'rule' for s in regs)
    assert _partition_ok(r.segments)
    assert abs(regs[0].q_hat - 190) < 6 and np.isfinite(regs[0].posterior)
    _dy, runs = rule_runs(m, r.footprints, cal.nrl_pooled, never)
    assert runs and all(any(s.start <= a and s.end >= b for s in regs) for a, b, _q, _n in runs)
    off = molecule_outputs(m, NucParams(seg_posterior_min=1.01, seg_rule=False), cal)
    assert not [s for s in off.segments if s.state == REGULAR]


def test_viterbi_option_and_default_agree_on_a_clean_array():
    rng = np.random.default_rng(5)
    m = array_molecule([('irr', 10, 0), ('reg', 25, 185), ('irr', 10, 0)], rng)
    cal = flat_calibration()
    vit = molecule_outputs(m, NucParams(seg_decode='viterbi', seg_rule=False), cal)
    dflt = molecule_outputs(m, cal=cal)
    rv = [s for s in vit.segments if s.state == REGULAR]
    rd = [s for s in dflt.segments if s.state == REGULAR]
    assert len(rv) == 1 and rv[0].source == 'hmm' and len(rd) >= 1
    assert sum(s.end - s.start for s in rd) >= rv[0].end - rv[0].start - 200
    with pytest.raises(ValueError):
        molecule_outputs(m, NucParams(seg_decode='bogus'), cal)


def test_disorder_needs_a_margin_else_undetermined():
    rng = np.random.default_rng(6)
    m = array_molecule([('reg', 20, 190), ('irr', 25, 0), ('reg', 20, 190)], rng)
    cal = flat_calibration()
    r = molecule_outputs(m, cal=cal)
    assert DISORDERED in {s.state for s in r.segments}
    strict = molecule_outputs(m, NucParams(seg_disorder_llr=1e9), cal)
    assert DISORDERED not in {s.state for s in strict.segments}
    assert UNDETERMINED in {s.state for s in strict.segments}
    assert _partition_ok(strict.segments)


# --- undersized runs ---------------------------------------------------------------

def test_undersized_run_against_the_density_matched_background():
    rng = np.random.default_rng(7)
    m = array_molecule([('reg', 15, 190), ('sub', 24, 60), ('reg', 15, 190)], rng)
    cal = flat_calibration()
    r = molecule_outputs(m, cal=cal)
    runs = [e for e in r.events if e.kind == UNDERSIZED]
    assert len(runs) == 1 and runs[0].count >= 10
    # the same footprints over a background where undersized is already the rule
    rich = flat_calibration(disruption={'undersized': {'log_lambda_edges': [], 'rate': [0.45], 'n': [1000]}})
    r2 = molecule_outputs(m, NucParams(undersized_min_llr=40.0), rich)
    assert not [e for e in r2.events if e.kind == UNDERSIZED]


def test_undersized_false_runs_do_not_follow_the_background_rate():
    """Chance runs of independent undersized calls at background p0: the
    score is a log-likelihood ratio, so the false-run rate stays put as p0
    changes six-fold."""
    rng = np.random.default_rng(8)
    rates = {}
    for p0 in (0.05, 0.3):
        p1 = max(0.5, 2 * p0)
        hits = 0
        n_seq = 3000
        min_llr = NucParams().undersized_min_llr
        for _ in range(n_seq):
            u = rng.random(100) < p0
            s = np.where(u, np.log(p1 / p0), np.log((1 - p1) / (1 - p0)))
            hits += any(sc >= min_llr and u[a:b].sum() >= 2 for a, b, sc in maximal_segments(s.tolist()))
        rates[p0] = hits / n_seq
    assert rates[0.05] < 0.03 and rates[0.3] < 0.03
    assert max(rates.values()) < 3 * max(min(rates.values()), 0.003)


# --- missing nucleosomes ----------------------------------------------------------------

def _array_with_gap(n_left, n_right, gap_cores, rng, q=190, rate=0.5, unmarked_gap=False):
    nucs, pos = [], 300
    for _ in range(n_left):
        nucs.append((pos, pos + 147))
        pos += q
    gap0 = pos
    pos += q * gap_cores
    for _ in range(n_right):
        nucs.append((pos, pos + 147))
        pos += q
    L = pos + 300
    occ = np.zeros(L, bool)
    for s, e in nucs:
        occ[s:e] = True
    hits = np.flatnonzero(~occ & (rng.random(L) < rate))
    if unmarked_gap:
        hits = hits[(hits < gap0 - 43) | (hits >= gap0 + q * gap_cores)]
    return make_molecule(L, nucs=nucs, msps=complement_msps(L, nucs), hits=hits), gap0


def test_missing_nucleosome_found_where_the_grid_predicts_one():
    rng = np.random.default_rng(9)
    cal = flat_calibration()
    m, gap0 = _array_with_gap(10, 10, 1, rng)
    r = molecule_outputs(m, cal=cal)
    miss = [e for e in r.events if e.kind == MISSING]
    assert len(miss) == 1 and miss[0].start < gap0 + 147 and miss[0].end > gap0
    assert r.scalars['n_missing'] == 1
    # two cores gone (an NFR-length opening) and an unmarked stretch: no call
    m2, _ = _array_with_gap(10, 10, 2, rng)
    assert not [e for e in molecule_outputs(m2, cal=cal).events if e.kind == MISSING]
    m3, _ = _array_with_gap(10, 10, 1, rng, unmarked_gap=True)
    assert not [e for e in molecule_outputs(m3, cal=cal).events if e.kind == MISSING]


# --- recurrence test -------------------------------------------------------------------

def test_measured_track_compresses_and_maps_windows():
    t = MeasuredTrack([(100, 110), (200, 205), (108, 120)])
    assert t.length == 25 and t.index(99) == 0 and t.index(115) == 15 and t.index(150) == 20
    assert t.index(203) == 23 and t.index(500) == 25
    t.set_kind('reg', [(105, 202)])
    assert t.x['reg'].tolist() == [0] * 5 + [1] * 17 + [0] * 3
    assert t.window(110, 202) == (10, 22)


def test_window_test_is_exact_against_enumeration():
    rng = np.random.default_rng(10)
    tracks = []
    for _ in range(3):
        t = MeasuredTrack([(0, 23)])
        x = (rng.random(23) < 0.35).astype(int)
        t.x['k'] = x
        tracks.append(t)
    w0, w1, K = 5, 13, 4
    r = window_test(tracks, 'k', w0, w1, K=K, min_molecules=1)
    obs = sum(int(np.floor(K * t.x['k'][w0:w1].sum() / 8 + 1e-9)) for t in tracks)
    dist = []
    for shifts in itertools.product(range(23), repeat=3):
        s = 0
        for t, sh in zip(tracks, shifts):
            xx = np.roll(t.x['k'], sh)
            s += int(np.floor(K * xx[w0:w1].sum() / 8 + 1e-9))
        dist.append(s)
    exact = float(np.mean(np.asarray(dist) >= obs))
    exp = float(np.mean(dist))
    if obs > exp:
        assert r['p'] == pytest.approx(exact)
    else:
        assert r['p'] == 1.0 and exact >= 0.2
    assert r['n'] == 3


def test_bh_and_window_merge():
    p = [0.001, 0.2, 0.01, np.nan, 0.04, 0.9]
    assert bh(p, 0.1).tolist() == [True, False, True, False, True, False]
    assert bh([0.5, 0.6], 0.1).tolist() == [False, False]
    assert merge_windows([0, 250, 1000], 500) == [(0, 750), (1000, 1500)]


def test_recurrence_null_holds_and_planted_recurrence_is_found():
    """Under shift-within-read, BH at 0.1 over a contig rarely rejects; a
    locus where most reads carry the kind is found."""
    rng = np.random.default_rng(12)

    def reads(planted):
        out = []
        for i in range(60):
            s = int(rng.integers(0, 40_000))
            L = int(rng.integers(8_000, 14_000))
            t = MeasuredTrack([(s, s + L)])
            iv = []
            pos = s + int(rng.integers(0, 3000))
            while pos < s + L:
                n = int(rng.integers(800, 3000))
                iv.append((pos, pos + n))
                pos += n + int(rng.integers(2000, 6000))
            if planted and s < 30_000 < s + L - 1500 and rng.random() < 0.8:
                iv.append((30_000, 31_500))
            t.set_kind('reg', iv)
            out.append(t)
        return out
    starts = list(range(0, 54_000, 250))
    any_null = 0
    for _ in range(5):
        tr = reads(False)
        p = [(window_test(tr, 'reg', w, w + 500) or {'p': np.nan})['p'] for w in starts]
        any_null += int(bh(p, 0.1).any())
    assert any_null <= 2
    tr = reads(True)
    p = [(window_test(tr, 'reg', w, w + 500) or {'p': np.nan})['p'] for w in starts]
    rej = bh(p, 0.1)
    hit = [w for w, ok in zip(starts, rej) if ok]
    assert any(29_500 <= w <= 31_000 for w in hit)


# --- dephasing --------------------------------------------------------------------------

def _anchor_molecules(n, rng, phased, chrom='c1', anchor=50_000, q=190, L=6_000):
    mols = []
    for i in range(n):
        off = 0.0 if phased else rng.uniform(0, q)
        start = anchor - L // 2
        # dyads on the anchor's grid (phased) or on a grid shifted per molecule (fuzzy)
        dy = anchor + off + q * np.arange(-(L // 2 - 200) // q, (L // 2 - 200) // q + 1) + rng.normal(0, 4, size=1)
        nucs = [(int(d - 73), int(d + 74)) for d in dy]
        occ = np.zeros(L, bool)
        for s, e in nucs:
            occ[s - start:e - start] = True
        hits = np.flatnonzero(~occ & (rng.random(L) < 0.5))
        nn = [(s - start, e - start) for s, e in nucs]
        m = make_molecule(L, nucs=nn, msps=complement_msps(L, nn), hits=hits, name=f'{chrom}{anchor}{i}',
                          chrom=chrom, ref_start=start, ref_end=start + L,
                          blocks=np.array([[0, start, L]]))
        mols.append(m)
    return mols


def test_fuzzy_locus_keeps_periodicity_but_loses_coherence():
    from fiberhmm.nuc.dephasing import anchor_phase, dephasing_test, matched_percentile
    rng = np.random.default_rng(13)
    cal = flat_calibration()
    phased_sets, fuzzy_sets = [], []
    for k in range(8):
        a = 50_000 + 20_000 * k
        phased_sets.append((_anchor_molecules(12, rng, True, anchor=a), a))
        fuzzy_sets.append((_anchor_molecules(12, rng, False, anchor=a + 10_000), a + 10_000))
    ph = [anchor_phase(m, ('c1', a, 0), cal) for m, a in phased_sets]
    fz = [anchor_phase(m, ('c1', a, 0), cal) for m, a in fuzzy_sets]
    assert min(p.coherence for p in ph) > 0.8 and max(f.coherence for f in fz) < 0.7
    assert abs(np.mean([p.rbar for p in ph]) - np.mean([f.rbar for f in fz])) < 0.15
    assert np.mean([f.dispersion_bp for f in fz]) > 3 * np.mean([p.dispersion_bp for p in ph])
    t = dephasing_test(fz, ph, n_perm=500)
    assert t['effect'] < 0 and t['p_dephased'] < 0.01 and t['p_more_phased'] > 0.5
    mp = matched_percentile(fz[0], ph)
    assert mp['p_dephased'] <= 1 / 9 + 1e-9
