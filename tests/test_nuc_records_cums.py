"""NucMolecule's cumulative count arrays are never handed out half-built (a molecule shared between threads, or
slimmed while another thread counts)."""
import numpy as np

from fiberhmm.nuc.records import NucMolecule


def _mol(L=2000):
    rng = np.random.default_rng(0)
    hits = rng.random(L) < 0.1
    informative = hits | (rng.random(L) < 0.5)
    empty = np.zeros((0, 2), dtype=np.int64)
    return NucMolecule('m', L, False, empty, empty, empty, hits, informative)


def test_counts_with_a_half_published_cumulative_pair():
    mol = _mol()
    s, e = np.arange(0, 1900, 50), np.arange(100, 2000, 50)
    want = mol.counts(s, e)
    # another thread has published the hit array but not yet the informative one, or a slim cleared one
    mol._cum_hit, mol._cum_inf = np.zeros(mol.length + 1, dtype=np.int64), None
    got = mol.counts(s, e)
    assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])
    mol._cum_hit, mol._cum_inf = None, None
    assert np.array_equal(mol.counts(s, e)[1], want[1])
