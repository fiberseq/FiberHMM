"""Round-2 audit regressions: the conditional logistic DP no longer underflows on large balanced strata (tilted
recurrence), a fit that does not converge is never reported as an estimate, fits leave the process's BLAS threads
alone, per-width reference deltas are placed at their classes' sampled widths, and the scan's cache keys carry the
nuc module's implementation revision. Synthetic data only."""
from __future__ import annotations

import numpy as np
import pytest

from test_nuc_refmath import _random_packed


# ---------------------------------------------------------------- clogit DP underflow
def _independent_optimum(n_half: int, l2: float = 1e-6) -> float:
    """The penalised optimum of one stratum: n_half rows x=0 (all cases), n_half rows x=-1, by log-sum-exp over
    the case count split (exact, independent of the DP)."""
    from scipy.optimize import minimize_scalar
    from scipy.special import gammaln, logsumexp

    def lc(n, k):
        return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    j = np.arange(n_half + 1)

    def nll(b):
        return logsumexp(lc(n_half, j) + lc(n_half, n_half - j) - (n_half - j) * b) + 0.5 * l2 * b * b
    return float(minimize_scalar(nll, bounds=(0, 80), method='bounded', options=dict(xatol=1e-10)).x)


def test_clogit_dp_reaches_the_optimum_of_a_large_balanced_separated_stratum():
    from fiberhmm.assoc.clogit import clogit, fit_ok
    n = 1500
    x = np.r_[np.zeros(n), -np.ones(n)]
    y = np.r_[np.ones(n), np.zeros(n)].astype(int)
    f = clogit(x[:, None], y, np.zeros(2 * n, int))
    assert f['converged'] and fit_ok(f)
    assert f['beta'][0] == pytest.approx(_independent_optimum(n), abs=1e-4)
    # the score at the returned coefficient is (penalised) zero
    assert np.nansum(f['pi'][:n]) == pytest.approx(n, abs=1e-3)


def test_tilted_dp_moments_equal_the_dft_route():
    from fiberhmm.assoc import _clogit_kernels as K
    from fiberhmm.assoc import clogit as CL
    rng = np.random.default_rng(2)
    n, k, p = 400, 150, 3
    Xf = rng.normal(size=(n, p))
    beta = np.array([0.4, -0.2, 0.1])
    eta = Xf @ beta
    offs = np.array([0, n], np.int64)
    kk = np.array([k], np.int64)
    a = np.zeros(1)
    M = np.zeros(1, np.int64)
    K.tilt(eta, offs, kk, CL._DFT_T0, CL._DFT_T1, a, M)
    loge, mean, cov = np.zeros(1), np.zeros((1, p)), np.zeros((1, p, p))
    K.dp_strata(Xf, eta, offs, kk, np.array([0], np.int64), True, loge, mean, cov, a)
    Mt = max(int(M[0]), 60)
    Qr, Qi = np.empty((n, Mt + 1)), np.empty((n, Mt + 1))
    le, pr, wv, c = K.dft_stratum(eta, 0, n, k, float(a[0]), Mt, True, Qr, Qi)
    m2, c2 = np.zeros(p), np.zeros((p, p))
    K.dft_products(Xf, pr, wv, c, Qr, Qi, k, True, m2, c2)
    assert loge[0] == pytest.approx(le, rel=1e-10)
    assert np.allclose(mean[0], m2, atol=1e-8) and np.allclose(cov[0], c2, atol=1e-8)
    pi = np.zeros(n)
    K.dp_case_probabilities(eta, offs, kk, np.array([0], np.int64), pi, a)
    assert np.allclose(pi, pr, atol=1e-9) and pi.sum() == pytest.approx(k)


def _stalled(fit_fn):
    """A clogit that stops after one Newton step (not converged)."""
    def run(X, y, strata, **kw):
        return fit_fn(X, y, strata, max_iter=1)
    return run


def test_a_fit_that_did_not_converge_is_not_reported(monkeypatch):
    from fiberhmm.assoc import clogit as CL
    from fiberhmm.assoc import cooccur as CO
    from fiberhmm.assoc.simulate import simulate_m1
    sim = simulate_m1(3, effect=1.2)
    ok = CO.cooccurrence(sim['a'], sim['b'], sim['molecules'], n_boot=20)
    assert np.isfinite(ok['pooled']['log_or']) and ok['pooled']['clogit_converged']
    monkeypatch.setattr(CO, 'clogit', _stalled(CL.clogit))
    bad = CO.cooccurrence(sim['a'], sim['b'], sim['molecules'], n_boot=20)
    pooled = bad['pooled']
    assert pooled['clogit_converged'] is False and 'did not converge' in pooled['effect_note']
    assert np.isnan(pooled['log_or']) and np.isnan(pooled['or']) and np.isnan(pooled['log_or_se_model'])
    assert np.isnan(pooled.get('or_lo', np.nan)) and np.isnan(pooled.get('or_hi', np.nan))
    # the test itself (score test) is unaffected
    assert pooled['p'] == ok['pooled']['p']


def test_reverse_association_does_not_report_a_stalled_fit(monkeypatch):
    from fiberhmm.assoc import clogit as CL
    from fiberhmm.assoc.nucleosome import reverse_association
    rng = np.random.default_rng(4)
    n = 600
    locus = rng.integers(0, 30, n)
    mol = np.arange(n)
    x = rng.integers(0, 2, n)
    y = (rng.random(n) < np.where(x == 1, 0.6, 0.3)).astype(int)
    ll = rng.normal(size=n)
    good = reverse_association(locus, mol, y, x, ll, n_boot=20)
    assert np.isfinite(good['clogit']['log_or']) and good['clogit']['converged']
    monkeypatch.setattr(CL, 'clogit', _stalled(CL.clogit))
    bad = reverse_association(locus, mol, y, x, ll, n_boot=20)
    assert bad['clogit']['converged'] is False and np.isnan(bad['clogit']['log_or'])
    assert np.isnan(bad['clogit']['se_model']) and np.isnan(bad['clogit']['lo'])
    assert bad['cmh']['log_or'] == good['cmh']['log_or']


def test_a_fit_leaves_the_process_blas_threads_alone(monkeypatch):
    from fiberhmm.assoc import clogit as CL
    assert CL.LIMIT_BLAS is False

    def boom():
        raise AssertionError('single_thread_blas used')
    monkeypatch.setattr(CL, 'single_thread_blas', boom)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4000, 3))
    y = (rng.random(4000) < 0.3).astype(int)
    f = CL.clogit(X, y, np.repeat(np.arange(4), 1000))
    assert f['converged']


# ---------------------------------------------------------------- width-class placement
def test_width_deltas_are_placed_at_the_classes_sampled_widths():
    from fiberhmm.nuc.regularity import RegularityParams, reference_scores, reference_widths, tile_reference
    trunc = RegularityParams(ends='truncated')
    rng = np.random.default_rng(7)
    mols = [_random_packed(5000, rng, name=f'm{i}') for i in range(80)]
    ref = reference_scores(mols, trunc, n_windows=300, seed=1, cut_windows=300, min_cut_scores=50)
    assert ref.width_deltas and set(ref.width_centres) == set(ref.width_deltas)
    d = ref.to_dict()
    assert 'width_centres' in d
    widths = reference_widths(d)
    for c, centre in ref.width_centres.items():
        assert c <= centre < c + ref.width_step
        # a tile of exactly the class's sampled width reads against the class's own median
        got = tile_reference(np.array([centre]), ref.median, widths)[0]
        assert got == pytest.approx(ref.median + ref.width_deltas[c], abs=1e-5)
    # 5 kb molecules on the default grid: the cut windows sit at the grid's own widths (the classes' lower edges)
    assert any(centre == float(c) for c, centre in ref.width_centres.items())


def test_records_without_centres_keep_the_class_midpoints():
    from fiberhmm.nuc.regularity import tile_reference
    w = {'deltas': {1000: -0.6, 1500: -0.2}, 'step': 250, 'window_bp': 2000}
    assert tile_reference(np.array([1125.0]), 3.0, w)[0] == pytest.approx(2.4)
    w2 = dict(w, centres={1000: 1000.0, 1500: 1500.0})
    assert tile_reference(np.array([1000.0]), 3.0, w2)[0] == pytest.approx(2.4)
    from fiberhmm.nuc.patches import DEFAULT_PATCHES, patch_digest
    assert patch_digest('abcdef12', DEFAULT_PATCHES, 3.0, w) != patch_digest('abcdef12', DEFAULT_PATCHES, 3.0, w2)


# ---------------------------------------------------------------- scan cache keys
def test_scan_cache_keys_carry_the_nuc_implementation():
    from fiberhmm.scan.modules import cache_version
    from fiberhmm.scan.modules.nuc import NucModule

    class Old:
        version = NucModule.version
    assert NucModule.implementation
    assert cache_version(NucModule) != cache_version(Old) == NucModule.version
    import inspect
    from fiberhmm.scan import runner
    src = inspect.getsource(runner.run_scan if hasattr(runner, 'run_scan') else runner)
    assert 'cal_key = I.digest([base, name, cache_version(mod)' in src
    assert '_unit_key(base, name, cache_version(mod)' in src
