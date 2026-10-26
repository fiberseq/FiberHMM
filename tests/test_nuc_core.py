"""fiberhmm.nuc core (NUC P-A): molecular frame, density, dyads, spacing,
mark autocorrelation, phasing, reference curve, window mode, self calibration.
Synthetic fixtures only."""
from __future__ import annotations

import numpy as np
import pysam
import pytest

from nuc_synth import (array_molecule, complement_msps, flat_calibration, header, make_record,
                       planted_record_molecule)

from fiberhmm.nuc.calibration import NucCalibration, calibrate_self
from fiberhmm.nuc.density import (BetaPrior, accessible_counts, edge_informativeness,
                                  edge_truncated, fit_beta_prior, footprint_lambdas, rho_read)
from fiberhmm.nuc.markacf import mark_acf
from fiberhmm.nuc.params import NucParams
from fiberhmm.nuc.phasing import period_bins, phasing, uniform_expectation
from fiberhmm.nuc.records import DecodeContext, decode_record, iter_molecules, make_molecule
from fiberhmm.nuc.sizes import fit_reference_curve, kde_mode
from fiberhmm.nuc.spacing import bootstrap_nrl_pooled, eligible_dyads, molecule_spacing
from fiberhmm.nuc.window import Observation, window_metrics


# --- records ------------------------------------------------------------------

def _write(tmp_path, records):
    path = str(tmp_path / 'm.bam')
    with pysam.AlignmentFile(path, 'wb', header=header()) as out:
        for build in sorted(records, key=lambda b: b[0]):
            out.write(build[1](out))
    pysam.index(path)
    return path


def test_forward_and_reverse_records_decode_to_one_molecular_frame(tmp_path):
    rng = np.random.default_rng(3)
    seq, nucs, msps, marks = planted_record_molecule(rng)
    tfs = [(nucs[3][1] + 5, nucs[3][1] + 30)]
    path = _write(tmp_path, [
        (1000, lambda o: make_record(o, 'fwd', seq, 1000, nucs, msps, tfs, marks)),
        (9000, lambda o: make_record(o, 'rev', seq, 9000, nucs, msps, tfs, marks, reverse=True)),
    ])
    mols = {m.name: m for m in iter_molecules(path)}
    f, r = mols['fwd'], mols['rev']
    assert f.length == r.length == len(seq)
    np.testing.assert_array_equal(f.nucs, np.array(nucs))
    np.testing.assert_array_equal(r.nucs, np.array(nucs))
    np.testing.assert_array_equal(f.tfs, r.tfs)
    # marks land on the planted molecular positions in both orientations
    assert set(np.flatnonzero(f.hits)) == set(int(m) for m in marks)
    np.testing.assert_array_equal(f.hits, r.hits)
    np.testing.assert_array_equal(f.informative, r.informative)
    # informative = A/T away from the encoder's edge trim
    at = np.array([b in 'AT' for b in seq])
    assert f.informative[20:-20].sum() == at[20:-20].sum()
    # projection: molecular 0 is the alignment start (forward) or end (reverse)
    assert f.mol_to_ref([0])[0] == 1000
    assert r.mol_to_ref([0])[0] == 9000 + len(seq) - 1
    assert r.ref_to_mol([9000 + len(seq) - 1])[0] == 0
    s, e = nucs[2]
    assert f.mol_interval_to_ref(s, e) == (1000 + s, 1000 + e)
    assert r.mol_interval_to_ref(s, e) == (9000 + len(seq) - e, 9000 + len(seq) - s)
    assert r.ref_interval_to_mol(9000 + len(seq) - e, 9000 + len(seq) - s)[:2] == (s, e)


def test_hard_clipped_record_with_whole_molecule_tags_is_skipped(tmp_path):
    rng = np.random.default_rng(4)
    seq, nucs, msps, marks = planted_record_molecule(rng, n_nuc=4)
    path = str(tmp_path / 'h.bam')
    with pysam.AlignmentFile(path, 'wb', header=header()) as out:
        r = make_record(out, 'clip', seq, 500, nucs, msps, (), marks)
        # store only the second half, hard-clipping the first, but keep whole-molecule MA
        half = len(seq) // 2
        r.query_sequence, r.cigarstring = seq[half:], f'{half}H{len(seq) - half}M'
        r.flag |= 0x800
        out.write(r)
    with pysam.AlignmentFile(path) as bam:
        ctx = DecodeContext(bam.header)
        mol, why = decode_record(next(bam.fetch(until_eof=True)), ctx)
    assert mol is None and why == 'length_mismatch'


# --- density --------------------------------------------------------------------

def test_lambda_acc_drops_msp_boundary_marks():
    L = 1000
    # MSP [100, 110) with marks on both boundary sites and nowhere else inside
    m = make_molecule(L, nucs=[(0, 100), (110, 1000)], msps=[(100, 110)], hits=[100, 109])
    k0, n0 = accessible_counts(m, NucParams(msp_boundary_sites=0))
    k1, n1 = accessible_counts(m)
    assert (k0, n0) == (2, 10)
    assert (k1, n1) == (0, 8)
    # anchor exclusion removes the window's sites
    assert accessible_counts(m, exclude=(100, 105)) == (0, 4)


def test_footprint_lambda_is_local_with_fallback():
    L = 4000
    hits = list(range(1000, 1300, 2))           # dense marks in one MSP only
    m = make_molecule(L, nucs=[(800, 1000), (1300, 1500), (3300, 3500)],
                      msps=[(1000, 1300), (3000, 3300)], hits=hits)
    lam = footprint_lambdas(m, m.nucs, 0.1, NucParams(lambda_local_min_sites=20))
    assert lam[0] > 0.4 and lam[1] > 0.4         # next to the dense MSP
    assert lam[2] < 0.1                          # MSP without marks: local rate shrinks to ~0
    lam2 = footprint_lambdas(m, m.nucs, 0.1, NucParams(lambda_local_min_sites=10_000))
    np.testing.assert_allclose(lam2, 0.1)        # below the site floor: lambda_acc


def test_beta_prior_shrinks_towards_dataset():
    rng = np.random.default_rng(0)
    p = rng.beta(20, 30, 500)
    n = rng.integers(200, 2000, 500)
    k = rng.binomial(n, p)
    prior = fit_beta_prior(k, n)
    assert abs(prior.alpha / (prior.alpha + prior.beta) - 0.4) < 0.02
    assert 20 < prior.alpha + prior.beta < 120
    assert abs(float(prior.shrink(0, 0)) - 0.4) < 0.02


def test_edge_truncation_and_informativeness():
    L = 2000
    m = make_molecule(L, nucs=[(0, 150), (300, 450), (1850, 2000)], msps=[(150, 300), (450, 1850)],
                      hits=[200, 290, 500, 1800])
    assert edge_truncated(m, m.nucs).tolist() == [True, False, True]
    nl, nr = edge_informativeness(m, m.nucs)
    assert nl[1] == 20 and nr[1] == 20 and nl[0] == 0


# --- spacing ----------------------------------------------------------------------

def test_regular_array_spacing():
    rng = np.random.default_rng(1)
    m = array_molecule([('reg', 40, 190)], rng)
    s, d = molecule_spacing(m, 190.0)
    assert abs(s.nrl - 190) <= 3
    assert s.regularity > 0.3
    assert abs(s.tau - 190) <= 5
    assert s.spectral_snr > 5
    assert s.gap_cv < 0.1
    assert not (s.too_short_nrl or s.too_short_regularity or s.too_short_snr)
    irr = array_molecule([('irr', 40, 0)], rng)
    si, _ = molecule_spacing(irr, 190.0)
    assert si.regularity < s.regularity - 0.15
    assert si.gap_cv > 0.15


def test_minimum_windows_mark_too_short():
    rng = np.random.default_rng(2)
    m = array_molecule([('reg', 4, 190)], rng)          # 3 gaps
    s, _ = molecule_spacing(m, 190.0)
    assert s.too_short_nrl and np.isnan(s.nrl)
    m6 = array_molecule([('reg', 6, 190)], rng)         # 5 gaps: NRL yes, S no
    s6, _ = molecule_spacing(m6, 190.0)
    assert not s6.too_short_nrl and s6.too_short_regularity


def test_gaps_skip_nonadjacent_truncated_and_long_msp():
    L = 3000
    nucs = [(0, 147), (200, 347), (390, 537), (580, 760), (800, 1400), (1450, 1597), (2000, 2147)]
    msps = complement_msps(L, nucs)
    hits = [s - 3 for s, _ in nucs[1:]] + [e + 3 for _, e in nucs]
    m = make_molecule(L, nucs=nucs, msps=msps, hits=hits)
    d = eligible_dyads(m, 190.0)
    # first footprint touches the read start (truncated); 600 bp footprint is not mono
    assert not d.eligible_mask[0] and not d.eligible_mask[4]
    # gaps: 1-2, 2-3 adjacent; 3->5 spans the 600 bp call; 5->6 spans a 403 bp MSP
    np.testing.assert_allclose(sorted(d.gaps), [190.0, 206.5])


def test_bootstrap_nrl_pooled_recovers_planted_repeat():
    rng = np.random.default_rng(5)
    mols = [array_molecule([('reg', 25, 186)], rng) for _ in range(30)]
    p0, p1 = bootstrap_nrl_pooled(mols)
    assert abs(p0 - 186) <= 3 and abs(p1 - 186) <= 3


def test_mark_acf_finds_the_repeat_without_calls():
    rng = np.random.default_rng(6)
    mols = [array_molecule([('reg', 60, 192)], rng, mark_rate=0.3) for _ in range(40)]
    r = mark_acf(mols, min_marks_per_bp=0.01)
    assert abs(r.peak_lag - 192) <= 6
    assert r.n_molecules == 40


# --- phasing -------------------------------------------------------------------------

def _phased_mols(rng, n, phase_fixed, P=190.0, anchor=50_000, reverse_every=0):
    mols = []
    for i in range(n):
        L = 4000
        offset = 0.0 if phase_fixed else rng.uniform(0, P)
        a_mol = 2000
        # symmetric about the anchor so an unoriented (folded) analysis sees one register
        dy = [a_mol + side * (150 + offset + k * P) + rng.normal(0, 6)
              for k in range(12) for side in (1, -1)]
        dy = sorted(dy)
        dy = [d for d in dy if 80 < d < L - 80]
        nucs = [(int(d - 73), int(d + 74)) for d in dy]
        rev = bool(reverse_every and i % reverse_every == 0)
        if rev:   # mirror so the reference-side dyads are the same
            nucs = [(L - e, L - s) for s, e in nucs]
        hits = [s - 2 for s, _ in nucs] + [e + 2 for _, e in nucs]
        ref_start = anchor - (L - 1 - a_mol if rev else a_mol)
        m = make_molecule(L, nucs=nucs, msps=complement_msps(L, nucs), hits=hits,
                          chrom='chrS', ref_start=ref_start, ref_end=ref_start + L,
                          blocks=np.array([[0, ref_start, L]]), is_reverse=rev, name=f'm{i}')
        mols.append(m)
    return mols


def test_phasing_fixed_register_beats_null_and_random_does_not():
    rng = np.random.default_rng(7)
    fixed = _phased_mols(rng, 60, True)
    rand = _phased_mols(rng, 60, False)
    pf = phasing(fixed, [('chrS', 50_000, 0)], P=190.0, require_span=False, seed=1)
    pr = phasing(rand, [('chrS', 50_000, 0)], P=190.0, require_span=False, seed=1)
    assert pf.summary.R > 0.5 and pf.summary.null_p < 0.01
    assert pf.summary.R_minus_null > 0.4
    assert pr.summary.R_minus_null < 0.1
    # per-molecule phase is reliable and close to the pooled phase when fixed
    rel = [pm for pm in pf.per_molecule if pm['reliable']]
    assert len(rel) > 50 and np.median([pm['f_bp'] for pm in rel]) < 20
    assert pf.dispersion_bp < 40 < (pr.dispersion_bp if np.isfinite(pr.dispersion_bp) else 1e9)


def test_whole_period_bins_have_zero_uniform_expectation():
    P = 189.0
    for lo, hi in period_bins(P):
        assert abs(uniform_expectation(lo, hi, P)) < 1e-9
    assert abs(uniform_expectation(150, 1200, P)) > 0.03     # 5.56 periods: biased


def test_oriented_anchor_uses_reference_direction_on_reverse_reads():
    rng = np.random.default_rng(8)
    mols = _phased_mols(rng, 40, True, reverse_every=2)
    down = phasing(mols, [('chrS', 50_000, 1)], P=190.0, side='down', require_span=False)
    both = phasing(mols, [('chrS', 50_000, 0)], P=190.0, require_span=False)
    assert down.summary.R > 0.5            # forward and mirrored reverse reads agree
    assert both.summary.n_dyads > down.summary.n_dyads


def test_molecule_frame_uses_the_barrier_msp():
    L = 3000
    nfr = (1400, 1700)
    nucs = []
    for k in range(6):
        nucs.append((nfr[1] + 40 + k * 190, nfr[1] + 40 + k * 190 + 147))
        nucs.append((nfr[0] - 187 - k * 190, nfr[0] - 40 - k * 190))
    nucs = [(s, e) for s, e in nucs if s > 0 and e < L]
    m = make_molecule(L, nucs=nucs, msps=complement_msps(L, nucs), hits=[s - 2 for s, _ in nucs],
                      chrom='chrS', ref_start=0, ref_end=L, blocks=np.array([[0, 0, L]]))
    ph = phasing([m] * 5, [('chrS', 1550, 0)], frame='molecule', P=190.0, require_span=False)
    assert ph.summary.R > 0.9 and not ph.excluded


# --- reference curve -------------------------------------------------------------------

def test_reference_curve_tracks_density_and_is_monotone():
    rng = np.random.default_rng(9)
    lam = np.exp(rng.uniform(np.log(0.05), np.log(0.8), 20_000))
    true_mode = 185 - 40 * (np.log(lam) - np.log(0.05)) / (np.log(0.8) - np.log(0.05))
    L = rng.normal(true_mode, 12)
    c = fit_reference_curve(L, lam, 190.0)
    assert all(np.diff(c.mu) <= 1e-9)
    assert abs(float(c.mu_at(0.06)) - 183) < 8 and abs(float(c.mu_at(0.7)) - 147) < 8
    assert 8 < float(c.sigma_at(0.2)) < 20
    z = c.z([float(c.mu_at(0.3))], [0.3])
    assert abs(z[0]) < 1e-6


def test_kde_mode():
    rng = np.random.default_rng(10)
    assert abs(kde_mode(rng.normal(150, 10, 5000), 90, 230) - 150) <= 2


# --- window mode and observations ----------------------------------------------------------

def test_window_metrics_per_observation():
    rng = np.random.default_rng(11)
    mols = []
    for i in range(20):
        hp = 1 + i % 2
        m = array_molecule([('reg', 30, 180 if hp == 1 else 200)], rng, chrom='chrS', hp=hp,
                           name=f'w{i}')
        m.ref_start, m.ref_end = 10_000, 10_000 + m.length
        m.blocks = np.array([[0, 10_000, m.length]])
        mols.append(m)
    cal = flat_calibration(nrl=190.0)
    obs = [Observation('HP1', haplotype=1), Observation('HP2', haplotype=2)]
    w = window_metrics(mols, ('chrS', 11_000, 15_000), anchor=13_000, cal=cal, observations=obs)
    q1 = w.groups['HP1']['nrl_quantiles'][2]
    q2 = w.groups['HP2']['nrl_quantiles'][2]
    assert abs(q1 - 180) <= 4 and abs(q2 - 200) <= 4
    assert w.groups['HP1']['n_molecules'] == 10
    assert not w.groups['HP1']['insufficient']
    assert sum(w.groups['HP1']['state_counts'].values()) == 10
    assert 'anchor' in w.groups['HP1']['phasing']
    # a window the molecules do not cover yields no rows
    w2 = window_metrics(mols, ('chrS', 1, 2000), cal=cal, observations=obs)
    assert w2.groups['HP1']['n_molecules'] == 0
    assert Observation.from_dict(obs[0].to_dict()) == obs[0]


# --- self calibration ---------------------------------------------------------------------

def test_calibrate_self_round_trip():
    rng = np.random.default_rng(12)
    mols = [array_molecule([('reg', 20, 188), ('irr', 5, 0)], rng, name=f'c{i}') for i in range(40)]
    cal = calibrate_self(mols, label='synthetic')
    assert abs(cal.nrl_pooled - 188) <= 4
    assert cal.curve is not None and cal.population and len(cal.population) == 40
    again = NucCalibration.from_dict(cal.to_dict())
    assert again.digest() == cal.digest()
    assert calibrate_self(list(reversed(mols)), label='synthetic').digest() == cal.digest()
