"""Low-regularity patches (fiberhmm.nuc.patches LowPatchParams) and per-kind shared geometries."""
from __future__ import annotations

import json

import numpy as np
import pytest

from test_nuc_patches import _lay, _scan_bam, _track

from fiberhmm.nuc.geometries import (CARRIES, GeometryParams, RefPatch, assign_states, find_all_geometries,
                                     find_geometries)
from fiberhmm.nuc.patches import (HIGH, IN_LOW_PATCH, IN_PATCH, LOW, LOW_KIND, OUTSIDE, LowPatchParams,
                                  PatchParams, call_patches, decode_patch_rows, patch_rows)
from fiberhmm.nuc.tags import NUCPATCH, nucpatch_declaration

SCORES = [0, 2.5, 1.5, None, 1.3, 0.2, -1.8, -2.6, -1.2, -1.5, 0.1, 2.2, 2.4]


def test_high_only_output_is_byte_identical_to_the_high_only_calls():
    # pinned on the commit before low calling existed (2072bb5)
    t = _track(SCORES)
    mp = call_patches(t, None, ref_median=0.0, params=PatchParams(snap=False))
    assert patch_rows(mp) == [(1125, 1000, [97, 90, 0]), (3625, 500, [110, 90, 2])]
    assert mp.params_digest == '4b83db16'
    assert mp.tile_state.tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]
    assert PatchParams().digest(0.5) == '021d22045ecf9e37f91f7c2937a9bc50af740f81ef47099cb07688ffefa548b0'
    assert 'low' not in PatchParams().to_dict()
    assert 'low' not in nucpatch_declaration(PatchParams())['flag_bits']


def test_low_patches_mirror_the_hysteresis_and_carry_bit_4():
    t = _track(SCORES)
    pp = PatchParams(snap=False, low=LowPatchParams(snap=False))
    mp = call_patches(t, None, ref_median=0.0, params=pp)
    rows = patch_rows(mp)
    assert [r[2][2] & LOW_KIND for r in rows] == [0, LOW_KIND, 0]
    assert [p.kind for p in mp.patches] == [HIGH, LOW, HIGH]
    low = mp.patches[1]
    assert (low.first_tile, low.last_tile) == (6, 9)
    assert mp.tile_state.tolist() == [OUTSIDE, 1, 1, 1, 1, OUTSIDE, 2, 2, 2, 2, OUTSIDE, 1, 1]
    assert IN_PATCH == 1 and IN_LOW_PATCH == 2
    # the high rows are those of high-only calling; the digest names the low parameters
    assert [r for r in rows if not r[2][2] & LOW_KIND] == patch_rows(call_patches(t, None, ref_median=0.0,
                                                                                  params=PatchParams(snap=False)))
    assert mp.params_digest != '4b83db16'
    dec = decode_patch_rows([(s, n) for s, n, _q in rows], [q for _s, _n, q in rows])
    assert [d['kind'] for d in dec] == [HIGH, LOW, HIGH]
    # enter must be reached: a dip to -1.2 only is no low patch
    t2 = _track([0, -1.0, -1.2, -0.9, 0])
    assert not call_patches(t2, None, ref_median=0.0, params=pp).patches
    # min_tiles
    t3 = _track([0, -2.0, 0, 0])
    assert not call_patches(t3, None, ref_median=0.0, params=pp).patches
    assert call_patches(t3, None, ref_median=0.0,
                        params=PatchParams(snap=False, low=LowPatchParams(snap=False, min_tiles=1))).patches[0].kind == LOW


def test_low_params_validation_and_round_trip():
    with pytest.raises(ValueError):
        LowPatchParams(enter=-0.5, extend=-1.0)        # mirrored: enter <= extend
    with pytest.raises(ValueError):
        PatchParams(extend=1.0, low=LowPatchParams(enter=-1, extend=1.0))   # low extend must lie below high extend
    with pytest.raises(ValueError):
        PatchParams.from_dict({'low': {'low': {}}})
    pp = PatchParams.from_dict({'enter': 2.5, 'low': {'enter': -2.0}})
    assert pp.low == LowPatchParams(enter=-2.0) and pp.kinds == (HIGH, LOW)
    assert PatchParams.from_dict(pp.to_dict()) == pp
    assert nucpatch_declaration(pp)['flag_bits']['low'] == 16
    assert nucpatch_declaration(pp)['params']['low']['enter'] == -2.0


def test_geometries_are_found_per_kind():
    lays = []
    for i in range(20):
        lays.append(_lay(f'h{i}', 'x', [(1000, 4000)]))
    for i in range(15):
        lay = _lay(f'l{i}', 'y', [])
        lay.patches = [RefPatch(1000, 4000, False, False, 1.0, -1, LOW)]   # same span, low kind
        lays.append(lay)
    hi = find_geometries(lays, GeometryParams())
    lo = find_geometries(lays, GeometryParams(), kind=LOW)
    assert [g.n_molecules for g in hi] == [20] and [g.n_molecules for g in lo] == [15]
    assert hi[0].geometry_id.startswith('G1_') and lo[0].geometry_id.startswith('L1_')
    assert lo[0].kind == LOW and hi[0].kind == HIGH
    G = find_all_geometries(lays, GeometryParams())
    assert [g.geometry_id.split('_')[0] for g in G] == ['G1', 'L1']
    st = assign_states(lays, G)
    assert st[0][G[0].geometry_id] == CARRIES and st[0][G[1].geometry_id] != CARRIES
    assert st[25][G[1].geometry_id] == CARRIES
    with pytest.raises(ValueError):
        find_geometries(lays, GeometryParams(), kind='medium')


def test_cli_low_flags():
    from fiberhmm.scan.cli import _nucpatch_params, _parser
    p = _parser()

    def fold(argv):
        params = {}
        _nucpatch_params(p, p.parse_args(argv), ['nuc'], params)
        return params
    assert 'low' not in fold(['--nucpatch'])['nuc']['nucpatch']
    got = fold(['--nucpatch-low'])['nuc']['nucpatch']['low']
    assert got == LowPatchParams().to_dict()
    got = fold(['--nucpatch-low-enter', '-2', '--nucpatch-low-extend', '-1', '--nucpatch-low-snap', 'off'])
    low = got['nuc']['nucpatch']['low']
    assert (low['enter'], low['extend'], low['snap']) == (-2.0, -1.0, False)
    with pytest.raises(SystemExit):
        fold(['--nucpatch-low-enter', '-0.5', '--nucpatch-low-extend', '-1'])


def test_scan_with_low_patches_keeps_the_high_rows(tmp_path):
    from fiberhmm.scan import run_units
    from fiberhmm.scan.tables import read_all
    from fiberhmm.scan.units import plan
    bam, length = _scan_bam(tmp_path)
    units, _info = plan({'chrS': length}, regions=[('chrS', 14_000, 20_000)])
    base = {'calibration': 'self', 'calibration_population': 300}
    ds = [{'dataset_id': 'synth', 'paths': [bam]}]

    def marks(out, low):
        nuc = dict(base, nucpatch={'low': {}} if low else {}, nucgeo={'min_share': 0.05})
        run_units(ds, units, modules=['nuc'], params={'nuc': nuc}, out=tmp_path / out, annotate=False)
        rows = [m for m in read_all(tmp_path / out / 'nuc' / 'marks.tsv.bgz') if m['group'] == NUCPATCH]
        hits = [h for h in read_all(tmp_path / out / 'nuc' / 'hits.tsv.bgz') if h['type'] == 'nuc.geometry']
        return rows, hits
    hi_rows, hi_hits = marks('hi', False)
    both_rows, both_hits = marks('both', True)
    flag = lambda m: int(m['q'].split(',')[2])            # noqa: E731
    high_of_both = sorted((m['qname'], m['mol_start'], m['mol_len'], m['q']) for m in both_rows
                          if not flag(m) & LOW_KIND)
    assert high_of_both == sorted((m['qname'], m['mol_start'], m['mol_len'], m['q']) for m in hi_rows)
    assert all(h.get('kind', '') in ('', None) for h in hi_hits)
    span = lambda h: (h['start'], h['end'], h['label'])   # noqa: E731
    assert {span(h) for h in hi_hits} == {span(h) for h in both_hits if not str(h['label']).startswith('L')}
    json.dumps(both_hits, default=str)
