"""The NFR engine's fast paths reproduce their reference implementations bit for bit (EXPERIMENTAL preview).

Each optimisation keeps a reference (``_em_dense``, uncached MinCovDet, ``element_open_excluding`` per read, the
per-read "shared" rule, ``_stratified_perm_reference``, ``_exact_stratified_reference``). A whole run with every fast
path must write byte-identical files to a run with every reference, so the outputs do not depend on which is used.
"""
import numpy as np
import pytest

from fiberhmm.inference.accessibility import coaccess as C
from fiberhmm.inference.accessibility import variants as V
from fiberhmm.inference.accessibility import run_accessibility
from fiberhmm.inference.accessibility.workflow import classes_from_rows

from test_accessibility_nfr import FAST, payload, planted

FILES = ('variants.tsv', 'configurations.tsv', 'molecules.tsv.gz', 'coaccess.tsv', 'combos.tsv', 'result.json',
         'context.json.gz')


def _references(monkeypatch):
    """Every fast path replaced by its reference."""
    monkeypatch.setattr(V, '_em', V._em_dense)
    monkeypatch.setattr(V, 'robust_geometry', lambda Xc, seed, cache=None: V._robust_geometry(Xc, seed))
    monkeypatch.setattr(C, 'stratified_perm', C._stratified_perm_reference)
    monkeypatch.setattr(C, 'exact_stratified', C._exact_stratified_reference)
    monkeypatch.setattr(C._OpennessIndex, 'openness',
                        lambda self, uids, ex: {u: C.element_open_excluding(self.cov[u], ex) for u in uids})
    monkeypatch.setattr(C._GapIndex, 'covering', lambda self, lo, hi: {
        u for u in self.uids if any(g0 <= lo and g1 >= hi for g0, g1 in self.gaps[u])})


def _locus():
    states = [(.3, [(800, 1000), (1400, 1700), (2100, 2300)]), (.2, [(1400, 1520), (1620, 1700), (2100, 2300)]),
              (.15, [(1480, 1620)]), (.15, [(800, 1000)]), (.2, [])]
    units, pick = planted(states, n=320, seed=31)
    rng = np.random.default_rng(5)
    member = {u['unit_id']: int(rng.random() < (.85 if j == 1 else .1)) for u, j in zip(units, pick)}
    sup = {('class_003', ch): dict(start=1530, end=1610, prevalence=.25) for ch in ('d::CT', 'd::GA')}
    classes = classes_from_rows(sup, {'class_003': {u['unit_id']: (member[u['unit_id']], f"d::{u['strand']}") for u in units}})
    return units, classes


@pytest.mark.parametrize('params', [
    dict(FAST, nfr_regions=[(780, 1020), (1380, 1720), (2080, 2320)], robust=2, within_clusters=3),
    dict(FAST),                                            # detected regions, variants as elements
    dict(FAST, nfr_regions=[(780, 1020), (1380, 1720), (2080, 2320)], elements='nfrs', openness_pad_bp=15),
])
def test_fast_paths_write_byte_identical_outputs(tmp_path, monkeypatch, params):
    units, classes = _locus()
    fast = run_accessibility(payload(units), dict(params), tmp_path/'fast', classes=classes)
    assert any(n['variants'] for n in fast['nfrs']) and fast['pairs']
    with monkeypatch.context() as m:
        _references(m)
        run_accessibility(payload(units), dict(params), tmp_path/'reference', classes=classes)
    for name in FILES:
        assert (tmp_path/'fast'/name).read_bytes() == (tmp_path/'reference'/name).read_bytes(), name


def _block_ll(rng, n, spans):
    """Read x configuration log-likelihoods: each row finite on one span of columns, -inf elsewhere."""
    c = spans[-1][1]
    LL = np.full((n, c), -np.inf)
    for i in range(n):
        a, b = spans[rng.integers(len(spans))]
        LL[i, a:b] = rng.normal(-20, 6, b - a)
        LL[i, a + rng.integers(b - a)] = -np.inf if b - a > 2 and rng.random() < .2 else LL[i, a]
    return LL


def test_em_is_bit_identical_to_the_dense_em():
    rng = np.random.default_rng(0)
    for trial in range(40):
        spans = [(0, 1), (1, 1 + int(rng.integers(1, 6))), ]
        spans.append((spans[-1][1], spans[-1][1] + int(rng.integers(1, 30))))
        LL = _block_ll(rng, int(rng.integers(5, 400)), spans)
        LL[rng.integers(len(LL), size=len(LL)//3)] = LL[0]               # duplicated rows (closed reads, bootstrap)
        if trial % 5 == 1:
            LL[2] = np.nan                                               # a NaN row poisons the dense EM; so here
        if trial % 5 == 2:
            LL[3] = -np.inf                                              # a read with no possible configuration
        w0 = None if trial % 2 else rng.dirichlet(np.ones(LL.shape[1]))
        for iters in (1, 7, 2000):
            wf, pf = V._em(LL, iters, w0)
            wd, pd = V._em_dense(LL, iters, w0)
            assert wf.tobytes() == wd.tobytes() and pf.tobytes() == pd.tobytes()
    one = np.zeros((9, 1))
    assert V._em(one)[0].tobytes() == V._em_dense(one)[0].tobytes()


def test_robust_geometry_cache_returns_the_same_values_and_copies():
    X = np.random.default_rng(2).normal(1000, 30, (80, 2))
    cache = {}
    a = V.robust_geometry(X, 1, cache)
    b = V.robust_geometry(X.copy(), 1, cache)
    ref = V._robust_geometry(X, 1)
    assert len(cache) == 1 and all(x.tobytes() == y.tobytes() for x, y in zip(a, ref))
    assert all(x.tobytes() == y.tobytes() for x, y in zip(b, ref)) and a[0] is not b[0]
    assert len(V.robust_geometry(X[::-1], 1, cache)) == 2 and len(cache) == 2      # order is part of the key


def test_openness_index_matches_element_open_excluding():
    rng = np.random.default_rng(4)
    cov = {}
    for i in range(300):
        s = int(rng.integers(0, 3000)); e = s + int(rng.integers(0, 1500))
        x = np.arange(s, e, 10)
        cov[f'u{i}'] = dict(x=x, closed=rng.random(len(x)) < rng.random())
    cov['float_grid'] = dict(x=np.arange(100., 900., 10.), closed=np.zeros(80, bool))
    cov['uneven'] = dict(x=np.array([0, 10, 25, 30] + list(range(40, 400, 10))), closed=np.ones(40, bool))
    index = C._OpennessIndex(cov)
    assert 'float_grid' not in index.row and 'uneven' not in index.row
    uids = sorted(cov)
    for _ in range(200):
        a = rng.uniform(-100, 4600); b = a + rng.uniform(-50, 600)
        c = rng.uniform(-100, 4600); d = c + rng.uniform(0, 600)
        ex = [(round(a, int(rng.integers(0, 3))), b), (c, round(d))]
        if rng.random() < .1:
            ex = [(a, b), (a + 5, b - 5)]                                    # nested
        got = index.openness(uids, ex)
        want = {u: C.element_open_excluding(cov[u], ex) for u in uids}
        assert np.array_equal(np.array([got[u] for u in uids]), np.array([want[u] for u in uids]), equal_nan=True)
    ex = [(float('nan'), 50.), (100., 200.)]
    assert np.array_equal(np.array(list(index.openness(uids, ex).values())),
                          np.array([C.element_open_excluding(cov[u], ex) for u in uids]), equal_nan=True)


def test_permutation_null_and_exact_test_match_their_references():
    rng = np.random.default_rng(6)
    for _ in range(30):
        n = int(rng.integers(20, 900))
        x = (rng.random(n) < rng.random()).astype(int); y = np.where(rng.random(n) < .5, x, (rng.random(n) < .3).astype(int))
        strata = C.openness_strata(rng.random(n), np.array(['d::CT', 'd::GA'])[rng.integers(0, 2, n)], int(rng.integers(5, 60)))
        assert (C.stratified_perm(x, y, strata, 200, np.random.default_rng(7)).tobytes()
                == C._stratified_perm_reference(x, y, strata, 200, np.random.default_rng(7)).tobytes())
        assert repr(C.exact_stratified(x, y, strata)) == repr(C._exact_stratified_reference(x, y, strata))


def test_nfr_progress_is_one_monotonic_bar_within_its_total():
    units, classes = _locus()
    seen = []

    def progress(stage, message=None, **work):
        seen.append((stage, message, work.get('completed'), work.get('total')))
    run_accessibility(payload(units), dict(FAST, nfr_regions=[(780, 1020), (1380, 1720), (2080, 2320)], robust=1),
                      None, progress, classes=classes)
    bar = [(c, t, m) for s, m, c, t in seen if s == 'nfr' and t is not None]
    assert len({t for _, t, _ in bar}) == 1 and len(bar) > 3*(FAST['kmax'] + 3)
    done = [c for c, _, _ in bar]
    assert all(isinstance(c, int) and 0 <= c <= bar[0][1] for c in done) and done == sorted(done) and done[-1] == bar[0][1]
    assert any('support test' in m for _, _, m in bar) and any('bootstrap' in m for _, _, m in bar)
    assert any('robustness rerun' in m for _, _, m in bar)


def test_edge_inputs_take_the_reference_paths():
    """Inputs the workflow does not generate still match the references (Codex review of the speedups)."""
    rng = np.random.default_rng(8)
    LL = _block_ll(rng, 100, [(0, 1), (1, 4), (4, 12)])
    for M in (np.asfortranarray(LL), LL[::2]):                         # other memory orders and strides
        for iters in (0, 1, 50):
            wf, pf = V._em(M, iters); wd, pd = V._em_dense(M, iters)
            assert wf.tobytes() == wd.tobytes() and (pf is pd is None or pf.tobytes() == pd.tobytes())
    plus = LL.copy(); plus[5, 2] = np.inf                                # +inf: the whole row is dense
    assert all(a.tobytes() == b.tobytes() for a, b in zip(V._em(plus, 30), V._em_dense(plus, 30)))
    # a small-integer grid that wrapped around is not a constant-step grid
    x = np.arange(120, 320, 10).astype(np.int8)
    cov = {'u': dict(x=x, closed=np.arange(20) % 2 == 0)}
    ex = [(-120, -90), (100, 110)]
    assert C._OpennessIndex(cov).openness(['u'], ex)['u'] == C.element_open_excluding(cov['u'], ex)
    # a stratum label that selects nothing (NaN) is left unpermuted, as the reference does
    x = np.array([1, 0, 1, 1, 0, 0]); y = np.array([1, 0, 0, 1, 1, 0]); strata = np.array([np.nan, 0, 0, 1, 1, 1])
    assert (C.stratified_perm(x, y, strata, 50, np.random.default_rng(1)).tobytes()
            == C._stratified_perm_reference(x, y, strata, 50, np.random.default_rng(1)).tobytes())
    # only an integer seed is cached
    cache = {}
    V.robust_geometry(np.random.default_rng(2).normal(0, 1, (30, 2)), None, cache)
    assert cache == {}
