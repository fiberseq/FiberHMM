"""Round-2 audit (second report) regressions in FiberHMM: patch edge clamping, the per-width reference with few
full windows, near folding, scan geometry over tiled BED intervals. Synthetic data only."""
from __future__ import annotations

import numpy as np
import pytest

from fiberhmm.nuc.patches import (DEFAULT_LOW_PATCHES, IN_LOW_PATCH, LOW, OUTSIDE, Patch, PatchParams, _clamp_low,
                                  _clamp_neighbours, call_patches)
from fiberhmm.nuc.records import make_molecule
from fiberhmm.nuc.regularity import RegularityParams, sliding_regularity


def _random_packed(length, rng, core=147, n_per_kb=5.0, rate=0.5, name='mol'):
    n = max(1, int(length / 1000 * n_per_kb))
    free = max(0, (length - 200) - n * core)
    cuts = np.sort(rng.integers(0, free + 1, n))
    starts = 100 + cuts + np.arange(n) * core
    nucs = [(int(s), int(s + core)) for s in starts if s + core < length]
    occ = np.zeros(length, bool)
    for a, b in nucs:
        occ[a:b] = True
    hits = ~occ & (rng.random(length) < rate)
    return make_molecule(length, nucs=nucs, hits=hits, name=name)


def _valid(ps):
    for p in ps:
        assert p.start < p.end, p
    ps = sorted(ps, key=lambda p: p.start)
    for i, p in enumerate(ps):
        for q in ps[i + 1:]:
            assert p.end <= q.start, (p, q)


# ---------------------------------------------------------------- H5: clamping
def test_a_start_reset_to_its_grid_never_inverts_the_patch():
    p = Patch(50, 1060, 500, 1000, 2, 3, 5.0, 190, snapped_start=True, snapped_end=True)
    q = Patch(950, 1450, 1500, 2000, 6, 7, 5.0, 190, snapped_start=True, snapped_end=True)   # both edges left
    ps = [p, q]
    _clamp_neighbours(ps)
    _valid(ps)
    assert (ps[1].start, ps[1].end) == (1500, 2000) and not ps[1].snapped_start and not ps[1].snapped_end


def test_a_reset_that_reopens_an_earlier_pair_is_clamped_too():
    p = Patch(60, 1800, 0, 500, 0, 1, 5.0, 190, snapped_start=True, snapped_end=True)
    q = Patch(1900, 2200, 1000, 1500, 4, 5, 5.0, 190, snapped_start=True, snapped_end=True)
    r = Patch(1750, 2250, 2000, 2500, 8, 9, 5.0, 190, snapped_start=True, snapped_end=True)
    ps = [p, q, r]
    _clamp_neighbours(ps)
    _valid(ps)
    got = {x.grid_start: (x.start, x.end) for x in ps}
    assert got == {0: (60, 500), 1000: (1000, 1500), 2000: (1750, 2250)}


def test_a_low_patch_covered_by_a_high_patch_is_dropped():
    h = Patch(0, 1597, 0, 500, 0, 1, 5.0, 190, snapped_end=True)
    q = Patch(1000, 1997, 1000, 1500, 4, 5, -2.0, -1, snapped_end=True, kind=LOW)
    state = np.full(10, OUTSIDE, dtype=np.int8)
    state[4:6] = IN_LOW_PATCH
    assert _clamp_low([q], [h], state) == []
    assert (state[4:6] == OUTSIDE).all()


def test_a_low_patch_clipped_on_one_side_keeps_its_free_grid():
    h = Patch(0, 1100, 0, 500, 0, 1, 5.0, 190, snapped_end=True)
    # start clipped to h's end; its end had snapped left of that point
    q = Patch(700, 1050, 1000, 1500, 4, 5, -2.0, -1, snapped_start=True, snapped_end=True, kind=LOW)
    state = np.full(10, OUTSIDE, dtype=np.int8)
    kept = _clamp_low([q], [h], state)
    assert len(kept) == 1 and (kept[0].start, kept[0].end) == (1100, 1500)


@pytest.mark.parametrize('rule', ['open', 'period'])
def test_called_patches_are_valid_and_low_patches_hold_their_own_tiles(rule):
    rng = np.random.default_rng(11)
    params = PatchParams(enter=0.4, extend=0.1, search_bp=1500, snap_rule=rule, open_bp=150,
                         low=DEFAULT_LOW_PATCHES)
    n = 0
    for i in range(60):
        mol = _random_packed(14000, rng, n_per_kb=rng.uniform(3.5, 6.0), name=f'r{i}')
        track = sliding_regularity(mol, RegularityParams())
        mp = call_patches(track, mol.nucs, ref_median=float(np.nanmedian(track.score)), params=params)
        _valid(mp.patches)
        for p in mp.patches:
            assert p.start < p.grid_end and p.end > p.grid_start, p
        n += len(mp.patches)
    assert n > 40


# ---------------------------------------------------------------- M15: near folding into another eligible root
def test_a_geometry_folds_into_an_eligible_root_when_its_widest_parent_folded_elsewhere():
    from fiberhmm.nuc.geometries import Geometry, GeometryParams, fold_geometries

    def g(gid, a, b):
        return Geometry(gid, a, b, False, False, 10, 10, [], 0.0, 0.0, 3.0, 190)
    C, B, D, A = g('C', 0, 2000), g('B', 600, 2000), g('D', 200, 1400), g('A', 600, 1400)
    geoms = [C, B, D, A]
    fold_geometries(geoms, GeometryParams(fold='near', tol_bp=100, near_max_bp=700))
    assert B.parent_id == 'C'                # B's own fold
    assert D.parent is None and C.parent is None
    assert A.parent_id == 'D'                # A is not near C; its widest parent B folded into C, D is a root


# ---------------------------------------------------------------- H11: a few long molecules among short ones
def test_a_few_full_windows_among_short_molecules_do_not_anchor_the_reference():
    from fiberhmm.nuc.regularity import reference_scores
    trunc = RegularityParams(ends='truncated')
    rng = np.random.default_rng(5)
    short = [_random_packed(1500, rng, n_per_kb=5.5, rate=0.6, name=f's{i}') for i in range(120)]
    longer = [_random_packed(2600, rng, n_per_kb=5.5, rate=0.6, name=f'l{i}') for i in range(2)]
    alone = reference_scores(short, trunc, n_windows=300, seed=1, cut_windows=300, min_cut_scores=50)
    mixed = reference_scores(short + longer, trunc, n_windows=300, seed=1, cut_windows=300, min_cut_scores=50)
    assert alone.usable() and 0 < mixed.n < 100
    assert mixed.usable()
    # the widest kept width class anchors, as with no full window at all (not the median of a couple of windows)
    top = max(mixed.width_deltas)
    assert mixed.width_deltas[top] == 0.0
    assert abs(mixed.median - alone.median) < 0.5


# ---------------------------------------------------------------- L21: geometries over BED intervals wider than a unit
def test_bed_intervals_tiled_into_grid_units_are_reported_for_geometries(tmp_path):
    from fiberhmm.scan.cli import tiled_bed_geometry_note
    from fiberhmm.scan.units import plan
    bed = tmp_path / 'loci.bed'
    bed.write_text('c1\t1000\t21000\tnarrow\nc1\t100000\t160000\twide\n')
    planned = plan({'c1': 1_000_000}, bed=str(bed))
    assert {u.mode for u in planned[0]} == {'bed', 'grid'}
    note = tiled_bed_geometry_note(planned, ['nuc'], {'nuc': {'nucpatch': {}}})
    assert '1 BED interval' in note and 'c1:100000-160000' in note and 'nuc.geometry' in note
    assert tiled_bed_geometry_note(planned, ['nuc'], {'nuc': {'nucpatch': {}, 'nucgeo': None}}) == ''
    assert tiled_bed_geometry_note(planned, ['nuc'], {'nuc': {}}) == ''
    assert tiled_bed_geometry_note(planned, ['fp'], {'nuc': {'nucpatch': {}}}) == ''
    small = tmp_path / 'small.bed'
    small.write_text('c1\t1000\t21000\tnarrow\n')
    assert tiled_bed_geometry_note(plan({'c1': 1_000_000}, bed=str(small)), ['nuc'], {'nuc': {'nucpatch': {}}}) == ''


# ---------------------------------------------------------------- M18: cached arrays read once
def test_cached_count_arrays_are_read_once_while_another_thread_slims_the_molecule():
    from fiberhmm.nuc.records import NucMolecule

    class Slimmed(NucMolecule):
        """The cached arrays vanish right after their first read (a slim on another thread)."""
        def __getattribute__(self, name):
            v = object.__getattribute__(self, name)
            if name in ('_cum_hit', '_cum_inf') and v is not None:
                object.__setattr__(self, name, None)
            return v
    rng = np.random.default_rng(0)
    L = 500
    hits = rng.random(L) < 0.1
    inf = hits | (rng.random(L) < 0.5)
    empty = np.zeros((0, 2), dtype=np.int64)
    mol = Slimmed('m', L, False, empty, empty, empty, hits, inf)
    for attr in ('n_hits', 'n_informative'):
        mol._cums()
        assert getattr(mol, attr) == int((hits & inf).sum() if attr == 'n_hits' else inf.sum())
    mol._cums()
    assert int(mol.cum_informative()[-1]) == int(inf.sum())
