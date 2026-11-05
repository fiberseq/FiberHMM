"""Geometry folding (none / near / all), partial carriers, rebuilt tracks and regularity read-end modes.

- nested pairs follow the definition (inside with tolerance slack, one end shared, the other moved > tol);
- fold 'none' keeps every geometry separate, 'near' folds only moves <= near_max_bp, 'all' folds every
  nested pair transitively into the longest parent; folding never changes the grouping;
- a molecule carrying only a folded child is a partial carrier of the parent; shares are reported apart;
- ``track_from_rows`` rebuilds the track from nucreg rows, so patches from tags equal patches from the track;
- RegularityParams 'blank' (default) is byte-identical to the parameters before read-end modes existed;
  'extend' and 'truncated' cover the read ends and short reads; patch ends at a read end stay open.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from fiberhmm.nuc.geometries import (ABSENT, CARRIES, PARTIAL, SCAN_STATE, GeometryParams, MoleculeLayout,
                                     RefPatch, assign_states, density_matched_shares, find_geometries,
                                     fold_geometries, nested_pairs, support)
from fiberhmm.nuc.patches import IRREGULAR, PatchParams, call_patches
from fiberhmm.nuc.records import make_molecule
from fiberhmm.nuc.regularity import (RegularityParams, sliding_regularity, track_from_rows, track_rows,
                                     window_grid, window_reach)


def _layout(uid, spans, ds='a', lo=-5000, hi=5000, density=50.0):
    t = np.arange(lo, hi, 250, dtype=np.int64)
    pts = [RefPatch(a, b, False, False, 4.0, 190) for a, b in spans]
    return MoleculeLayout(uid, ds, pts, t, t + 250, np.full(len(t), IRREGULAR, dtype=np.int8), density)


def _cohort(groups, n_plain=20):
    """``groups``: [(start, end, n_molecules)]: every molecule of a group carries that one patch."""
    lays, k = [], 0
    for a, b, n in groups:
        for i in range(n):
            j = (i % 3 - 1) * 50                      # a little edge jitter, well within the tolerance
            lays.append(_layout(f'm{k}', [(a + j, b - j)], ds='a' if k % 2 else 'b'))
            k += 1
    for i in range(n_plain):
        lays.append(_layout(f'p{i}', [], ds='a' if i % 2 else 'b'))
    return lays


# long array 0..4000; nested near child 0..3000 (end moved 1000); nested far child 0..1500 (moved 2500);
# a separate array -4000..-2000
GROUPS = [(0, 4000, 30), (0, 3000, 20), (0, 1500, 15), (-4000, -2000, 25)]


def _spans(geoms):
    return [(g.start, g.end) for g in geoms]


def test_nested_pairs_follow_the_definition():
    geoms = find_geometries(_cohort(GROUPS), GeometryParams(fold='none'))
    assert sorted(_spans(geoms)) == sorted((a, b) for a, b, _n in GROUPS)
    idx = {(g.start, g.end): i for i, g in enumerate(geoms)}
    pairs = {(geoms[z.child].start, geoms[z.child].end, geoms[z.parent].start, geoms[z.parent].end, z.shared,
              z.moved) for z in nested_pairs(geoms, 500)}
    assert pairs == {(0, 3000, 0, 4000, 'start', 1000), (0, 1500, 0, 4000, 'start', 2500),
                     (0, 1500, 0, 3000, 'start', 1500)}
    # an end moved by no more than the tolerance is not nested (the grouping would have joined them)
    assert not nested_pairs(geoms, 2600) or all(z.moved > 2600 for z in nested_pairs(geoms, 2600))
    assert idx


@pytest.mark.parametrize('fold, expect', [
    ('none', {}),
    ('near', {(0, 3000): (0, 4000)}),                     # 1500 -> 3000 moves 1500 > 1250: not folded
    ('all', {(0, 3000): (0, 4000), (0, 1500): (0, 4000)}),
])
def test_fold_modes(fold, expect):
    lays = _cohort(GROUPS)
    geoms = find_geometries(lays, GeometryParams(fold=fold))
    base = find_geometries(lays, GeometryParams(fold='none'))
    assert _spans(geoms) == _spans(base)                  # folding never changes the grouping
    got = {(g.start, g.end): (geoms[g.parent].start, geoms[g.parent].end) for g in geoms if g.parent is not None}
    assert got == expect
    for g in geoms:
        if g.parent is not None:
            assert geoms[g.parent].parent is None         # transitive: always the root
            assert g.parent_id == geoms[g.parent].geometry_id


def test_near_max_move_is_a_parameter():
    lays = _cohort(GROUPS)
    geoms = find_geometries(lays, GeometryParams(fold='near', near_max_bp=1500))
    got = {(g.start, g.end): (geoms[g.parent].start, geoms[g.parent].end) for g in geoms if g.parent is not None}
    # 0..1500 is nested in 0..3000 (moved 1500) and 0..4000 (moved 2500): near 1500 folds it into 0..3000, which
    # itself folds into 0..4000 -> root 0..4000
    assert got == {(0, 3000): (0, 4000), (0, 1500): (0, 4000)}
    geoms = find_geometries(lays, GeometryParams(fold='near', near_max_bp=900))
    assert all(g.parent is None for g in geoms)


def test_partial_carriers_and_shares():
    lays = _cohort(GROUPS)
    geoms = find_geometries(lays, GeometryParams(fold='all'))
    states = assign_states(lays, geoms)
    root = next(g for g in geoms if (g.start, g.end) == (0, 4000))
    sp = support(lays, states, root.geometry_id)
    assert sp['carriers'] == 30 and sp['partial'] == 35
    assert sp['share'] == pytest.approx(30 / len(lays)) and sp['share_partial'] == pytest.approx(35 / len(lays))
    assert sum(s[root.geometry_id] == PARTIAL for s in states) == 35
    child = next(g for g in geoms if (g.start, g.end) == (0, 3000))
    assert support(lays, states, child.geometry_id)['partial'] == 0      # only roots have partial carriers
    none = assign_states(lays, find_geometries(lays, GeometryParams(fold='none')))
    assert not any(v == PARTIAL for s in none for v in s.values())
    assert SCAN_STATE[PARTIAL] == '0'
    dm = density_matched_shares(lays, states, root.geometry_id, ['a', 'b'])
    assert set(dm) == {'a', 'b'}


def test_geometry_params_validation():
    for bad in (dict(fold='some'), dict(near_max_bp=-1), dict(tol_bp=-5), dict(unknown=1)):
        with pytest.raises(ValueError):
            GeometryParams.from_dict(bad)
    assert GeometryParams().fold == 'near' and GeometryParams().near_max_bp == 1250
    assert GeometryParams.from_dict(GeometryParams(fold='all').to_dict()).fold == 'all'


def test_fold_geometries_resets_on_refold():
    lays = _cohort(GROUPS)
    geoms = find_geometries(lays, GeometryParams(fold='all'))
    assert any(g.parent is not None for g in geoms)
    fold_geometries(geoms, GeometryParams(fold='none'))
    assert all(g.parent is None and g.parent_id is None for g in geoms)


# ---------------------------------------------------------------- tracks from rows
def _array(length, rng, repeat=190, core=147, rate=0.5, name='mol', irregular=()):
    starts = [s for s in np.arange(100, length - core - 100, repeat)]
    nucs = []
    for s in starts:
        if any(a <= s < b for a, b in irregular):
            s = int(s + rng.integers(-60, 60))
        nucs.append((int(s), int(s + core)))
    nucs = sorted(nucs)
    occ = np.zeros(length, bool)
    for a, b in nucs:
        occ[max(0, a):b] = True
    hits = ~occ & (rng.random(length) < rate)
    return make_molecule(length, nucs=nucs, hits=hits, name=name)


@pytest.mark.parametrize('ends', ['blank', 'extend', 'truncated'])
def test_patches_from_rebuilt_track_equal_patches_from_the_track(ends):
    rng = np.random.default_rng(3)
    p = RegularityParams(ends=ends)
    for i in range(6):
        mol = _array(int(rng.integers(5000, 12000)), rng, rate=0.06, name=f'r{i}', irregular=[(3000, 5000)])
        tr = sliding_regularity(mol, p)
        rows = track_rows(tr)
        rb = track_from_rows(mol.length, [(s, n) for s, n, _q in rows], [q for _s, _n, q in rows], p)
        assert np.array_equal(rb.tile_starts, tr.tile_starts) and np.array_equal(rb.tile_ends, tr.tile_ends)
        assert np.array_equal(np.isfinite(rb.score), np.isfinite(tr.score))
        assert np.nanmax(np.abs(rb.score - np.clip(tr.score, -2.0, 8.0))) <= 10 / 254 / 2 + 1e-9   # byte range
        ref = float(np.nanmedian(rb.score)) - 1.0
        assert np.nanmax(tr.score) < 8.0
        a = call_patches(tr, mol.nucs, ref_median=ref)
        b = call_patches(rb, mol.nucs, ref_median=ref)
        assert [(x.start, x.end, x.open_start, x.open_end) for x in a.patches] == \
            [(x.start, x.end, x.open_start, x.open_end) for x in b.patches]


# ---------------------------------------------------------------- read-end modes
_OLD_FIELDS = ('window_bp', 'step_bp', 'signal', 'band_bp', 'ref_bp', 'min_calls', 'mask_open_bp', 'thin_per_kb',
               'nfft')


def test_blank_ends_are_byte_identical_to_the_parameters_before_read_end_modes():
    p = RegularityParams()
    old = {k: (list(v) if isinstance(v, tuple) else v) for k, v in
           ((f, getattr(p, f)) for f in _OLD_FIELDS)}
    blob = json.dumps(old, sort_keys=True, separators=(',', ':'))
    assert p.digest() == hashlib.sha256(blob.encode()).hexdigest()
    assert p.to_dict() == old
    assert RegularityParams(ends='blank', end_min_bp=500).digest() == p.digest()
    assert RegularityParams(ends='extend').digest() != p.digest()
    assert RegularityParams(ends='truncated').digest() != RegularityParams(ends='truncated', end_min_bp=1500).digest()
    rng = np.random.default_rng(1)
    mol = _array(9_731, rng)
    a, b = sliding_regularity(mol), sliding_regularity(mol, RegularityParams.from_dict(old))
    assert track_rows(a) == track_rows(b)
    for bad in (dict(ends='both'), dict(end_min_bp=10), dict(short_min_bp=5000)):
        with pytest.raises(ValueError):
            RegularityParams(**bad)


def test_extend_paints_the_outer_tiles_to_the_read_ends():
    rng = np.random.default_rng(2)
    mol = _array(9_731, rng)
    blank = sliding_regularity(mol)
    ext = sliding_regularity(mol, RegularityParams(ends='extend'))
    assert ext.tile_starts[0] == 0 and ext.tile_ends[-1] == mol.length
    assert np.array_equal(ext.tile_starts[1:], blank.tile_starts[1:])
    assert np.array_equal(ext.score[np.isfinite(blank.score)], blank.score[np.isfinite(blank.score)])
    assert window_reach(mol.length, RegularityParams(ends='extend')) == (0, mol.length)


def test_truncated_ends_cover_the_read_and_short_reads():
    rng = np.random.default_rng(4)
    p = RegularityParams(ends='truncated')
    mol = _array(9_731, rng)
    tr = sliding_regularity(mol, p)
    blank = sliding_regularity(mol)
    assert tr.tile_starts[0] == 0 and tr.tile_ends[-1] == mol.length
    assert np.array_equal(tr.tile_ends[:-1], tr.tile_starts[1:])          # still abutting
    # full windows score exactly as in 'blank'
    full = np.isin(tr.centres, blank.centres)
    assert np.array_equal(tr.score[full], blank.score, equal_nan=True)
    assert np.isfinite(tr.score[~full]).any()
    # a short read: no tile in 'blank', tiles in 'truncated' down to short_min_bp
    short = _array(1_600, rng, name='short')
    assert len(sliding_regularity(short)) == 0
    st = sliding_regularity(short, p)
    assert len(st) and st.tile_starts[0] == 0 and st.tile_ends[-1] == 1_600
    assert len(sliding_regularity(short, RegularityParams(ends='truncated', short_min_bp=1700))) == 0
    # a larger end_min_bp leaves more of the end out of reach
    reach = window_reach(mol.length, RegularityParams(ends='truncated', end_min_bp=1500))
    assert 0 < reach[0] < 875 and mol.length - 875 < reach[1] < mol.length
    c, ts, te, ws, we = window_grid(mol.length, p)
    assert ((we - ws) >= 1000).all() and ((ts <= c) | (ts == 0)).all()


def test_patch_ends_at_a_read_end_stay_open_in_every_mode():
    rng = np.random.default_rng(5)
    mol = _array(8_000, rng, rate=0.6)
    for ends in ('blank', 'extend', 'truncated'):
        tr = sliding_regularity(mol, RegularityParams(ends=ends))
        mp = call_patches(tr, mol.nucs, ref_median=float(np.nanmin(tr.score)) - 3.0)
        assert mp.patches, ends
        p0 = mp.patches[0]
        assert p0.open_start and p0.open_end, ends
        if ends != 'blank':
            assert p0.start == 0 and p0.end == mol.length


def test_cli_folds_the_new_flags():
    from fiberhmm.scan.cli import _nucpatch_params, _nucreg_params, _parser
    p = _parser()
    params = {}
    args = p.parse_args(['--nucreg-ends', 'truncated', '--nucreg-end-min', '1200', '--nucreg-short-min', '800',
                         '--nucreg-min-calls', '10', '--nucgeo-fold', 'all', '--nucgeo-near-max-bp', '900',
                         '--nucpatch-search-bp', '600', '--nucpatch-tol-frac', '0.3', '--nucpatch-nuc-min-bp', '100'])
    _nucreg_params(p, args, ['nuc'], params)
    _nucpatch_params(p, args, ['nuc'], params)
    reg = RegularityParams.from_dict(params['nuc']['nucreg'])
    assert (reg.ends, reg.end_min_bp, reg.short_min_bp, reg.min_calls) == ('truncated', 1200, 800, 10)
    geo = params['nuc']['nucgeo']
    assert geo['fold'] == 'all' and geo['near_max_bp'] == 900
    pat = params['nuc']['nucpatch']
    assert (pat['search_bp'], pat['tol_frac'], pat['nuc_min_bp']) == (600, 0.3, 100)
    PatchParams.from_dict({k: v for k, v in pat.items() if k in PatchParams.__dataclass_fields__})
    assert CARRIES and ABSENT


# ---------------------------------------------------------------- incremental grouping == the plain greedy rule
def _reference_groups(layouts, params):
    """The prototype's greedy rule, recounting every candidate in every round (the original implementation)."""
    import math
    rows = [(i, k, p.start, p.end) for i, lay in enumerate(layouts) for k, p in enumerate(lay.patches)]
    if not rows:
        return []
    R = np.asarray(rows, dtype=np.int64)
    need = max(int(params.min_molecules), int(math.ceil(params.min_share * len(layouts) - 1e-9)))
    tol = int(params.tol_bp)
    left = np.ones(len(R), dtype=bool)
    out = []
    while left.any() and (params.max_geometries is None or len(out) < params.max_geometries):
        L = R[left]
        cands, reps = np.unique(L[:, 2:4], axis=0, return_counts=True)
        best = None
        for j, c in enumerate(cands):
            box = (np.abs(L[:, 2] - c[0]) <= tol) & (np.abs(L[:, 3] - c[1]) <= tol)
            key = (len(np.unique(L[box, 0])), int(reps[j]), -int(c[0]), -int(c[1]))
            if best is None or key > best[0]:
                best = (key, c)
        if best is None or best[0][0] < need:
            break
        a, b = best[1]
        idx = np.flatnonzero(left)
        mem = idx[(np.abs(R[idx, 2] - a) <= tol) & (np.abs(R[idx, 3] - b) <= tol)]
        left[mem] = False
        out.append(sorted((int(i), int(k)) for i, k in R[mem, :2]))
    return out


@pytest.mark.parametrize('seed', range(6))
def test_incremental_grouping_equals_the_plain_greedy_rule(seed):
    rng = np.random.default_rng(seed)
    centres = [(int(rng.integers(-4000, 2000) // 250 * 250), int(rng.integers(500, 4000) // 250 * 250)) for _ in range(6)]
    lays = []
    for m in range(400):
        spans, pos = [], -5000
        for _ in range(int(rng.integers(0, 4))):
            a, w = centres[int(rng.integers(len(centres)))]
            a = max(pos, a + int(rng.integers(-3, 4)) * int(rng.choice([1, 125, 250])))
            if a + w > 5000:
                break
            # short patches too: a molecule may hold two patches inside one box
            w = w if rng.random() > 0.15 else int(rng.integers(2, 4)) * 250
            spans.append((a, a + w))
            pos = a + w + int(rng.integers(0, 3)) * 250
        lays.append(_layout(f'm{m}', spans))
    for tol in (250, 500, 1000):
        gp = GeometryParams(tol_bp=tol, min_share=0.01, fold='none')
        got = [sorted(g.members) for g in find_geometries(lays, gp)]
        assert got == _reference_groups(lays, gp)


def test_patches_off_leave_the_settings_and_their_hash_as_without_patch_support():
    from fiberhmm.scan.identity import settings_hash
    from fiberhmm.scan.modules.nuc import NucModule
    m = NucModule()
    off = m.resolve({})
    assert 'nucpatch' not in off and 'nucgeo' not in off
    assert settings_hash(m.resolve({'nucgeo': {'fold': 'all'}})) == settings_hash(off)   # geometries need patches
    on = m.resolve({'nucpatch': {}})
    assert on['nucgeo']['fold'] == 'near' and on['nucgeo']['near_max_bp'] == 1250
    assert settings_hash(on) != settings_hash(off)
    assert m.resolve({'nucpatch': {}, 'nucgeo': None})['nucgeo'] is None
