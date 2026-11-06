"""Regressions for the regularity reference, patch edges, geometry folding, scan cohorts and clogit numerics
(audit fixes): per-window-width reference for ends 'truncated', usable-reference rule, digests of full-window
references unchanged, snapped patches never overlap, near folding checked against the root, the scan's cohort
default, share_dm's denominator, physical-molecule dedup in scan geometries, the reference-counted BLAS limit and a
non-finite likelihood in the line search."""
from __future__ import annotations

import threading

import numpy as np
import pytest

from fiberhmm.nuc.records import make_molecule
from fiberhmm.nuc.regularity import (RegularityParams, ReferenceScores, reference_scores, reference_widths,
                                     sliding_regularity, tile_reference, track_from_rows, track_rows)


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


TRUNC = RegularityParams(ends='truncated')


# ---------------------------------------------------------------- 1: per-width reference
def test_blank_reference_is_unchanged_and_has_no_width_classes():
    rng = np.random.default_rng(3)
    mols = [_random_packed(6000, rng, name=f'm{i}') for i in range(30)]
    blank = reference_scores(mols, RegularityParams(), n_windows=300, seed=5)
    trunc = reference_scores(mols, TRUNC, n_windows=300, seed=5, min_cut_scores=20)
    # the full-width sample does not depend on the read-end mode
    assert blank.median == trunc.median and np.array_equal(blank.scores, trunc.scores)
    assert not blank.width_deltas and 'width_deltas' not in blank.to_dict()
    assert trunc.width_deltas and set(trunc.to_dict()) >= {'width_deltas', 'width_n', 'width_step', 'window_bp'}


def test_cut_windows_have_a_lower_null_and_tiles_read_against_their_width():
    rng = np.random.default_rng(4)
    mols = [_random_packed(7000, rng, name=f'm{i}') for i in range(80)]
    ref = reference_scores(mols, TRUNC, n_windows=400, seed=1, cut_windows=300, min_cut_scores=50)
    assert ref.usable()
    d = ref.width_deltas
    # narrower windows: fewer band bins, a lower random-packing median
    assert min(d) < 1250 and d[min(d)] < -0.2
    assert all(v < 0.15 for v in d.values())
    widths = reference_widths(ref.to_dict())
    per = tile_reference(np.array([2000, 1999, min(d) + 125, 1000]), ref.median, widths)
    assert per[0] == pytest.approx(ref.median)
    assert per[3] < per[1] and per[3] < ref.median - 0.2
    # random-packed molecules: cut tiles sit about as far below the full median as their own null does
    track = sliding_regularity(mols[0], TRUNC)
    cut = track.window_widths < 2000
    assert cut.any() and (~cut).any()
    ex_full = track.score[~cut] - ref.median
    ex_cut = track.score[cut] - tile_reference(track.window_widths, ref.median, widths)[cut]
    assert np.isfinite(ex_cut).any()
    # without the per-width reference, cut tiles would read as much less regular than full ones
    raw_cut = track.score[cut] - ref.median
    assert np.nanmedian(raw_cut) < np.nanmedian(ex_cut)
    assert abs(np.nanmedian(ex_cut) - np.nanmedian(ex_full)) < abs(np.nanmedian(raw_cut) - np.nanmedian(ex_full)) + 1e-9


def test_short_molecules_get_a_finite_reference_in_truncated_mode():
    rng = np.random.default_rng(5)
    mols = [_random_packed(1500, rng, n_per_kb=5.5, rate=0.6, name=f's{i}') for i in range(120)]
    blank = reference_scores(mols, RegularityParams(), n_windows=300, seed=1)
    assert blank.n == 0 and not np.isfinite(blank.median) and not blank.usable()
    ref = reference_scores(mols, TRUNC, n_windows=300, seed=1, cut_windows=300, min_cut_scores=50)
    assert np.isfinite(ref.median) and ref.usable()
    track = sliding_regularity(mols[0], TRUNC)
    assert np.isfinite(track.score).any()
    from fiberhmm.nuc.patches import NO_CALL, call_patches
    mp = call_patches(track, mols[0].nucs, ref_median=ref.median, ref_widths=reference_widths(ref.to_dict()))
    assert (mp.tile_state != NO_CALL).sum() == np.isfinite(track.score).sum()


def test_usable_needs_enough_finite_scores():
    few = ReferenceScores(np.full(20, 3.0), 3.0)
    assert not few.usable() and few.usable(10)
    assert not ReferenceScores(np.full(500, 3.0), float('nan')).usable()


def test_rebuilt_tracks_carry_window_widths():
    rng = np.random.default_rng(6)
    mol = _random_packed(5300, rng)
    t = sliding_regularity(mol, TRUNC)
    rows = track_rows(t)
    r = track_from_rows(mol.length, [(a, n) for a, n, _q in rows], [q for _a, _n, q in rows], TRUNC)
    assert np.array_equal(r.window_widths, t.window_widths)
    assert sliding_regularity(mol, RegularityParams()).window_widths is None


def test_patch_digest_changes_only_with_width_deltas():
    from fiberhmm.nuc.patches import DEFAULT_PATCHES, patch_digest
    base = patch_digest('abcdef12', DEFAULT_PATCHES, 3.0)
    assert patch_digest('abcdef12', DEFAULT_PATCHES, 3.0, None) == base
    w = {'deltas': {1000: -0.6}, 'step': 250, 'window_bp': 2000}
    assert patch_digest('abcdef12', DEFAULT_PATCHES, 3.0, w) != base


# ---------------------------------------------------------------- 9: snapped edges never overlap
def test_snapped_edges_are_clamped_at_neighbours():
    from fiberhmm.nuc.patches import LOW, Patch, _clamp_low, _clamp_neighbours
    a = Patch(1000, 3400, 1000, 2500, 0, 5, 5.0, 190, snapped_end=True)          # end snapped past b's start
    b = Patch(2600, 5000, 3000, 5000, 8, 15, 5.0, 190, snapped_start=True)       # start snapped into the gap
    ps = [a, b]
    _clamp_neighbours(ps)
    assert ps[0].end <= ps[1].start
    assert not ps[0].snapped_end and ps[0].end == ps[0].grid_end
    state = np.zeros(30, dtype=np.int8)
    lo = Patch(4800, 7000, 5500, 7000, 18, 24, 1.0, -1, snapped_start=True, kind=LOW)   # snapped into b
    kept = _clamp_low([lo], ps, state)
    assert kept and kept[0].start >= ps[1].end and not kept[0].snapped_start


def test_called_patches_never_overlap():
    from fiberhmm.nuc.patches import DEFAULT_LOW_PATCHES, PatchParams, call_patches
    rng = np.random.default_rng(8)
    params = PatchParams(enter=0.5, extend=0.2, search_bp=1000, low=DEFAULT_LOW_PATCHES)
    n = 0
    for i in range(40):
        mol = _random_packed(12000, rng, name=f'r{i}')
        track = sliding_regularity(mol, RegularityParams())
        mp = call_patches(track, mol.nucs, ref_median=float(np.nanmedian(track.score)), params=params)
        ps = sorted(mp.patches, key=lambda p: p.start)
        n += len(ps)
        for p, q in zip(ps, ps[1:]):
            assert p.end <= q.start
    assert n > 20


# ---------------------------------------------------------------- 4, 5: scan cohort default and share_dm
def test_scan_geometry_cohort_defaults_to_overlapping():
    from fiberhmm.scan.modules.nuc import NucModule
    s = NucModule().resolve({'nucpatch': {}})
    assert s['nucgeo']['cohort'] == 'overlapping'
    # per-width reference settings enter the settings only when not default (settings hash unchanged)
    assert 'reference_cut_windows' not in s['nucpatch']
    assert NucModule().resolve({'nucpatch': {'reference_cut_windows': 200}})['nucpatch']['reference_cut_windows'] == 200


def _layout(uid, ds, patches, span=(0, 10000), density=20.0):
    from fiberhmm.nuc.geometries import MoleculeLayout, RefPatch
    lo = np.arange(span[0], span[1], 250, dtype=np.int64)
    pts = [RefPatch(a, b, False, False, 5.0, 190) for a, b in patches]
    st = np.zeros(len(lo), dtype=np.int8)
    for a, b in patches:
        st[(lo >= a) & (lo < b)] = 1
    return MoleculeLayout(uid, ds, pts, lo, lo + 250, st, density)


class _Out:
    def __init__(self):
        self.hits, self.values, self.states = [], [], []


class _Unit:
    chrom, start, end = 'chrS', 0, 10000


def _rec(name, ds='a'):
    from fiberhmm.scan.types import RecordRef
    return RecordRef(ds, 0, name, name, 0, 'chrS', 0, 10000, name)


def test_share_dm_is_the_callable_share_for_one_dataset():
    from fiberhmm.scan.modules.nuc import _resolve_geo_settings, _unit_geometries
    geo = _resolve_geo_settings({'min_share': 0.05})
    mols = []
    for i in range(10):          # carriers
        mols.append((_rec(f'c{i}'), _layout(f'c{i}', 'a', [(2000, 5000)])))
    for i in range(10):          # absent
        mols.append((_rec(f'x{i}'), _layout(f'x{i}', 'a', [])))
    for i in range(20):          # not spanned
        mols.append((_rec(f'n{i}'), _layout(f'n{i}', 'a', [], span=(6000, 10000))))
    out = _Out()
    _unit_geometries(out, _Unit(), mols, geo, ['a'])
    v = {(x.group, x.value_name): x.value for x in out.values}
    assert v[('a', 'share')] == pytest.approx(10 / 40)
    assert v[('a', 'share_callable')] == pytest.approx(0.5)
    assert v[('a', 'share_dm')] == pytest.approx(0.5) and v[('all', 'share_dm')] == pytest.approx(0.5)


# ---------------------------------------------------------------- 6: scan support counts physical molecules
def test_scan_geometries_count_physical_molecules(tmp_path):
    """Three records of one PacBio ZMW (one mol:v1) carrying the same patch are one molecule: below the minimum
    of three molecules, no geometry; their states follow the representative record."""
    from fiberhmm.scan.modules.nuc import _resolve_geo_settings, _unit_geometries
    from fiberhmm.scan.identity import molecule_id
    names = ['mv/7/ccs/fwd', 'mv/7/ccs/rev', 'mv/7/ccs']
    assert len({molecule_id([n]) for n in names}) == 1
    geo = _resolve_geo_settings({'min_share': 0.0, 'min_molecules': 3})
    rep = (_rec(names[0]), _layout('rep', 'a', [(2000, 5000)]))
    others = [(_rec(f'o{i}'), _layout(f'o{i}', 'a', [])) for i in range(5)]
    out = _Out()
    _unit_geometries(out, _Unit(), [rep] + others, geo, ['a'], [(_rec(names[1]), 0), (_rec(names[2]), 0)])
    assert not out.hits
    two = (_rec('mv/9/ccs'), _layout('two', 'a', [(2010, 5010)]))
    three = (_rec('mv/11/ccs'), _layout('three', 'a', [(1990, 4990)]))
    out = _Out()
    _unit_geometries(out, _Unit(), [rep, two, three] + others, geo, ['a'], [(_rec(names[1]), 0), (_rec(names[2]), 0)])
    assert len(out.hits) == 1 and out.hits[0].columns['molecules'] == 3
    st = {s.record.qname: s.state for s in out.states}
    assert st[names[1]] == st[names[2]] == st[names[0]] == '1'


# ---------------------------------------------------------------- 7, 8: clogit
def test_overlapping_fits_restore_the_blas_thread_count():
    tp = pytest.importorskip('threadpoolctl')
    from fiberhmm.assoc.clogit import single_thread_blas

    def blas():
        return [p['num_threads'] for p in tp.threadpool_info() if p.get('user_api') == 'blas']
    before = blas()
    if not before:
        pytest.skip('no BLAS thread pool')
    a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()

    def fit_a():
        with single_thread_blas():
            a_in.set(); b_in.wait(5)
        a_out.set()

    def fit_b():
        a_in.wait(5)
        with single_thread_blas():
            b_in.set(); a_out.wait(5)
    ta, tb = threading.Thread(target=fit_a), threading.Thread(target=fit_b)
    ta.start(); tb.start(); ta.join(); tb.join()
    assert blas() == before


def test_line_search_never_accepts_a_non_finite_likelihood():
    from fiberhmm.assoc.clogit import clogit
    rng = np.random.default_rng(0)
    n, k = 3000, 1500
    x = np.r_[np.zeros(n // 2), np.full(n // 2, -1.0)]
    y = np.zeros(n, dtype=np.int64)
    y[np.argsort(x + rng.normal(0, 1e-3, n))[-k:]] = 1        # cases all in the x = 0 half: nearly separated
    fit = clogit(x[:, None], y, np.zeros(n, dtype=np.int64), max_iter=40)
    assert np.all(np.isfinite(fit['beta'])) and np.isfinite(fit['loglik'])
