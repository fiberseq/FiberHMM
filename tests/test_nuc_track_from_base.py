"""track_from_base: a molecule's 'extend' / 'truncated' track derived from its full-window ('blank') track is
identical, array for array, to sliding_regularity with those parameters (so a caller keeping base tracks changes the
read-end mode without computing the full windows again)."""
from __future__ import annotations

import numpy as np
import pytest

from fiberhmm.nuc.records import make_molecule
from fiberhmm.nuc.regularity import RegularityParams, base_params, sliding_regularity, track_from_base


def _mol(length, rng, name):
    starts = np.sort(rng.choice(np.arange(50, max(60, length - 200)), size=max(1, length // 400), replace=False))
    nucs, last = [], -1
    for s in starts:
        if s > last:
            nucs.append((int(s), int(min(length, s + 147))))
            last = s + 147 + int(rng.integers(5, 80))
    occ = np.zeros(length, bool)
    for a, b in nucs:
        occ[a:b] = True
    hits = ~occ & (rng.random(length) < rng.uniform(0.05, 0.5))
    return make_molecule(length, nucs=nucs, hits=hits, name=name)


FIELDS = ('centres', 'tile_starts', 'tile_ends', 'score', 'period', 'n_calls')


@pytest.mark.parametrize('extra', [{}, {'mask_open_bp': 300}, {'thin_per_kb': 25.0}, {'signal': 'occupancy'},
                                   {'step_bp': 200, 'min_calls': 5}])
@pytest.mark.parametrize('ends', ['blank', 'extend', 'truncated'])
def test_derived_tracks_equal_computed_ones(ends, extra):
    rng = np.random.default_rng(7)
    params = RegularityParams.from_dict(dict(extra, ends=ends))
    for i, length in enumerate([600, 1500, 1999, 2000, 2001, 2250, 4321, 9000, 15_503]):
        mol = _mol(length, rng, f'm{i}')
        want = sliding_regularity(mol, params)
        got = track_from_base(mol, sliding_regularity(mol, base_params(params)), params)
        for f in FIELDS:
            a, b = getattr(want, f), getattr(got, f)
            assert a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True), (length, f)
        assert (want.length, want.thinned, want.params_digest) == (got.length, got.thinned, got.params_digest)
        assert want.density_per_kb == got.density_per_kb or (np.isnan(want.density_per_kb) and np.isnan(got.density_per_kb))
        if want.window_widths is None:
            assert got.window_widths is None
        else:
            assert np.array_equal(want.window_widths, got.window_widths)


def test_a_base_track_of_other_parameters_is_refused():
    rng = np.random.default_rng(1)
    mol = _mol(5000, rng, 'x')
    base = sliding_regularity(mol, RegularityParams(step_bp=200))
    with pytest.raises(ValueError):
        track_from_base(mol, base, RegularityParams(ends='extend'))
