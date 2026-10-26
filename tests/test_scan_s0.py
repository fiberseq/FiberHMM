"""fiberhmm-scan S0: unit planning, identities, tables, seam reconciliation and stitching.

Gates (scan format reference, S0):
- planted fixture with merge/split cases at seams: each feature is reported exactly once;
- a stitched region is identical for two grid offsets;
- a byte-identical rerun;
- zero-width rows are fetched at p.
"""
from __future__ import annotations

import filecmp
import json
from pathlib import Path

import pytest

pysam = pytest.importorskip('pysam')

from fiberhmm.scan import identity as I
from fiberhmm.scan.seams import reconcile_anchor, territory_runs
from fiberhmm.scan.tables import Table, TableWriter, read_all
from fiberhmm.scan.types import Hit, ScanUnit
from fiberhmm.scan.units import PlanError, UnitIndex, default_annotate, merge_intervals, plan

REG = {'plant': 'fiberhmm.scan.testing:PlantedAnchorModule',
       'stitchy': 'fiberhmm.scan.testing:PlantedStitchModule',
       'molly': 'fiberhmm.scan.testing:PlantedMoleculeModule'}


@pytest.fixture(scope='module')
def synth_bam(tmp_path_factory):
    from fiberhmm.scan.testing import write_synthetic_bam
    d = tmp_path_factory.mktemp('synth')
    return write_synthetic_bam(d / 'synth.bam', length=200_000)


# ------------------------------------------------------------------ planning
def test_grid_zones_partition_and_context():
    units, info = plan({'c1': 120_000, 'c2': 30_000}, genome=True)
    assert info['mode'] == 'grid' and default_annotate(info)
    for chrom, length in (('c1', 120_000), ('c2', 30_000)):
        us = [u for u in units if u.chrom == chrom]
        assert us[0].zone_start == 0 and us[-1].zone_end == length
        for a, b in zip(us, us[1:]):
            assert a.zone_end == b.zone_start and a.right == b.unit_id and b.left == a.unit_id
        for u in us:
            assert u.start == max(0, u.zone_start - 12_500) and u.end == min(length, u.zone_end + 12_500)
            assert u.end - u.start <= 50_000
    assert [u.index for u in units] == list(range(len(units)))


def test_grid_offset_and_budget():
    units, _ = plan({'c1': 100_000}, genome=True, grid_offset=5_000)
    assert [(u.zone_start, u.zone_end) for u in units][:3] == [(0, 5_000), (5_000, 30_000), (30_000, 55_000)]
    with pytest.raises(PlanError):
        plan({'c1': 100_000}, genome=True, unit_bp=60_000)
    with pytest.raises(PlanError):
        plan({'c1': 100_000}, genome=True, unit_bp=50_100)
    with pytest.raises(PlanError):
        plan({'c1': 100_000}, genome=True, unit_bp=25_000, stride_bp=50_000)


def test_bed_merge_overlap_and_bookended(tmp_path):
    bed = tmp_path / 'u.bed'
    bed.write_text('c1\t100\t200\ta\nc1\t150\t300\tb\nc1\t300\t400\tc\nc1\t1000\t1100\td\n')
    units, info = plan({'c1': 200_000}, bed=bed)
    assert [(u.start, u.end, u.name) for u in units] == [(100, 400, 'a;b;c'), (1000, 1100, 'd')]
    assert info['mode'] == 'bed' and not default_annotate(info)
    assert info['merged'][0]['parts'] == [[100, 200], [150, 300], [300, 400]]
    merged, _ = merge_intervals([('c1', 0, 10, ''), ('c1', 10, 20, '')])
    assert merged == [('c1', 0, 20, '')]


def test_long_region_is_grid_planned_restricted():
    units, info = plan({'c1': 500_000}, regions=[('c1', 100_000, 180_000)])
    assert info['mode'] == 'grid' and default_annotate(info)
    assert units[0].start == 100_000 and units[-1].end == 180_000
    assert units[0].zone_start == 100_000 and units[-1].zone_end == 180_000
    short, info2 = plan({'c1': 500_000}, regions=[('c1', 100_000, 140_000)])
    assert len(short) == 1 and short[0].mode == 'bed' and info2['mode'] == 'bed'


def test_unit_index_ownership():
    units, _ = plan({'c1': 100_000}, genome=True)
    idx = UnitIndex(units)
    assert idx.zone_unit('c1', 0).zone_start == 0
    assert idx.zone_unit('c1', 25_000).zone_start == 25_000
    assert idx.owner_of_record('c1', 49_999, 70_000).zone_start == 25_000


# ------------------------------------------------------------------ identities
def test_identities_are_deterministic(synth_bam, tmp_path):
    cid = I.content_id(synth_bam)
    assert cid.startswith('cid:v1:') and cid == I.content_id(synth_bam)
    copy = tmp_path / 'copy.bam'
    copy.write_bytes(Path(synth_bam).read_bytes())
    assert I.content_id(copy) == cid                        # a copy is the same content
    assert I.canonical_order([['b'], ['a'], ['c']]) == [1, 0, 2]
    assert I.molecule_key('m64/123/ccs') == 'pacbio/m64/123'
    assert I.molecule_key('m64/123/ccs/fwd') == 'pacbio/m64/123'
    assert I.molecule_key('read7') == 'read/read7'
    assert I.molecule_id(['m64/123/ccs/rev', 'm64/123/ccs']) == I.molecule_id(['m64/123/ccs'])
    assert I.molecule_id(['x']).startswith('mol:v1 ')
    h1 = I.hit_id('fp', 'fp.class', 'c1', 150, 100, 200, 'class_001', 'c1:0-50000', '1.0', 'abc')
    assert h1.startswith('fp:class:c1:150:') and len(h1.rsplit(':', 1)[1]) == 8
    assert h1 != I.hit_id('fp', 'fp.class', 'c1', 150, 100, 201, 'class_001', 'c1:0-50000', '1.0', 'abc')


# ------------------------------------------------------------------ tables
def test_tables_zero_width_and_sorting(tmp_path):
    order = {'c1': 0, 'c2': 1}
    w = TableWriter(tmp_path / 't.tsv.bgz', ['chrom', 'start', 'end', 'idx_start', 'idx_end', 'name', 'x'], order)
    w.add('c2', 5, 9, ['late', 1.5])
    w.add('c1', 50, 50, ['zero', None])
    w.add('c1', 10, 20, ['first', True])
    rec = w.close()
    assert rec['rows'] == 3
    rows = read_all(tmp_path / 't.tsv.bgz')
    assert [r['name'] for r in rows] == ['first', 'zero', 'late']
    assert rows[1]['idx_start'] == '50' and rows[1]['idx_end'] == '51' and rows[1]['x'] == ''
    t = Table(tmp_path / 't.tsv.bgz')
    assert [r['name'] for r in t.fetch('c1', 50, 51)] == ['zero']      # fetched at p
    assert [r['name'] for r in t.fetch('c1', 45, 55)] == ['zero']
    assert [r['name'] for r in t.fetch('c1', 51, 60)] == []
    assert list(t.fetch('cX', 0, 10)) == []


# ------------------------------------------------------------------ seams (unit level)
def _units3():
    units, _ = plan({'c1': 100_000}, genome=True)
    return units


def test_reconcile_merge_split_component_resolved_once():
    u0, u1, u2, u3 = _units3()[:4]          # zones 0-25k, 25-50k, 50-75k, 75-100k
    b = u1.zone_end
    merged = Hit('t', 'm', 'c1', b - 500, b + 100, b - 200)
    left_part = Hit('t', 'p1', 'c1', b - 500, b - 300, b - 400)
    right_part = Hit('t', 'p2', 'c1', b - 200, b + 100, b - 50)
    kept, comps, open_s = reconcile_anchor([(u1, [merged]), (u2, [left_part, right_part])])
    assert comps == 1 and open_s == 0
    assert [(u.unit_id, h.key) for u, h in kept] == [(u1.unit_id, 'm')]


def test_reconcile_identical_hit_once_and_children_follow():
    us = _units3()
    u1, u2 = us[1], us[2]
    b = u1.zone_end
    a1 = Hit('nfr.region', 'N1', 'c1', b + 400, b + 900, b + 650)
    a1c = Hit('nfr.variant', 'N1:V1', 'c1', b + 420, b + 880, b + 650, parent='N1')
    a2 = Hit('nfr.region', 'N7', 'c1', b + 400, b + 900, b + 650)
    a2c = Hit('nfr.variant', 'N7:V1', 'c1', b + 410, b + 890, b + 650, parent='N7')
    kept, comps, _ = reconcile_anchor([(u1, [a1, a1c]), (u2, [a2, a2c])])
    keys = sorted((u.unit_id, h.key) for u, h in kept)
    assert keys == [(u2.unit_id, 'N7'), (u2.unit_id, 'N7:V1')]


def test_reconcile_anchor_rule_away_from_boundaries_and_open_seam():
    us = _units3()
    u1, u2 = us[1], us[2]
    inside = Hit('t', 'in', 'c1', 30_000, 30_100, 30_050)      # u1 zone, far from boundaries
    outside = Hit('t', 'out', 'c1', 20_000, 20_100, 20_050)    # u1 span but u0 zone (no u0 result)
    kept, _c, open_s = reconcile_anchor([(u1, [inside, outside])], missing=[us[0].unit_id, u2.unit_id])
    assert [h.key for _u, h in kept] == ['in']
    assert open_s == 1


def test_territory_runs():
    import numpy as np
    us = _units3()
    t = lambda u: {'x': np.full(u.zone_end - u.zone_start, 1)}
    runs = territory_runs([(us[0], t(us[0]), 'ok'), (us[1], None, 'missing'), (us[2], t(us[2]), 'ok'),
                           (us[3], t(us[3]), 'ok')])
    assert [(s, e, lo, ro) for s, e, _a, lo, ro in runs] == [(0, 25_000, False, True), (50_000, 100_000, True, False)]
    runs = territory_runs([(us[0], t(us[0]), 'ok'), (us[1], None, 'empty'), (us[2], t(us[2]), 'ok')])
    assert [(s, e, lo, ro) for s, e, _a, lo, ro in runs] == [(0, 75_000, False, False)]     # zeros between
    assert runs[0][2]['x'][25_000:50_000].sum() == 0 and runs[0][2]['x'][:25_000].sum() == 25_000
    runs = territory_runs([(us[0], None, 'empty'), (us[1], t(us[1]), 'ok'), (us[2], None, 'empty'),
                           (us[3], None, 'empty')])
    assert [(s, e) for s, e, _a, _l, _r in runs] == [(0, 75_000)]                   # one empty zone padded each side


# ------------------------------------------------------------------ the planted fixture through run_units
FEATURES = [
    ['A', 10_000, 10_300, 'both'],                 # far from seams
    ['B', 24_500, 24_700, 'merge_left'],           # boundary 25k: left unit merges B+C, right unit splits
    ['C', 24_800, 25_200, 'both'],
    ['D', 49_900, 50_400, 'both'],                 # straddles 50k, both units report it
    ['E', 74_200, 74_900, 'both'],
    ['E1', 74_300, 74_500, 'child:E'],
    ['Z', 120_000, 120_000, 'both'],               # zero width
    ['F', 150_000, 150_600, 'both'],
]


def _run(out, bam, *, modules=('plant',), params=None, offset=0, annotate=None, jobs=0, cont=False,
         contigs=('chrS',)):
    from fiberhmm.scan import run_units
    units = plan({'chrS': 200_000}, genome=True, contigs=list(contigs), grid_offset=offset)
    p = {'plant': {'features': FEATURES, 'boundaries': list(range(offset, 200_001, 25_000)) + [25_000]},
         'stitchy': {'runs': [[24_000, 26_500], [49_990, 50_010], [99_000, 101_000], [160_000, 160_250]]}}
    p.update(params or {})
    p = {k: v for k, v in p.items() if k in modules}
    return run_units([{'dataset_id': 'synth', 'paths': [bam]}], units, modules=list(modules), params=p,
                     out=out, annotate=annotate, registry=REG, cores=2 if jobs else 1, window_jobs=jobs,
                     continue_run=cont)


def _covering(rows, s, e):
    return [r for r in rows if int(r['start']) <= s and e <= max(int(r['end']), int(r['start']) + 1)]


def test_planted_fixture_each_feature_once(tmp_path, synth_bam):
    _run(tmp_path / 's', synth_bam, annotate=False)
    rows = read_all(tmp_path / 's' / 'plant' / 'hits.tsv.bgz')
    roots = [r for r in rows if r['type'] == 'plant.feature']
    for key, s, e, mode in FEATURES:
        if mode.startswith('child:'):
            continue
        cover = _covering(roots, s, e)
        assert len(cover) == 1, (key, cover)
    assert any(r['label'] == '' and r['planted'] == 'B+C' for r in roots)       # the left unit's merge won
    child = [r for r in rows if r['type'] == 'plant.child']
    assert len(child) == 1 and child[0]['parent_id'] == next(r['hit_id'] for r in roots if r['planted'] == 'E')
    z = [r for r in roots if r['planted'] == 'Z'][0]
    assert z['start'] == z['end'] == '120000' and 'zero_width' in z['flags']
    t = Table(tmp_path / 's' / 'plant' / 'hits.tsv.bgz')
    assert [r['planted'] for r in t.fetch('chrS', 120_000, 120_001)] == ['Z']
    native = read_all(tmp_path / 's' / 'plant' / 'planted.tsv.bgz')
    assert sorted(r['hit_id'] for r in native) == sorted(r['hit_id'] for r in rows)
    m = json.loads((tmp_path / 's' / 'manifest.json').read_text())
    assert m['status']['seam_components']['plant'] >= 3 and m['status']['units_failed'] == []
    assert m['outputs']['annotate'] is False


def test_stitched_region_identical_for_two_grid_offsets(tmp_path, synth_bam):
    _run(tmp_path / 'o0', synth_bam, modules=('stitchy',), annotate=False)
    _run(tmp_path / 'o1', synth_bam, modules=('stitchy',), offset=5_000, annotate=False)
    a = (tmp_path / 'o0' / 'stitchy' / 'hits.tsv.bgz').read_bytes()
    b = (tmp_path / 'o1' / 'stitchy' / 'hits.tsv.bgz').read_bytes()
    assert a == b
    rows = read_all(tmp_path / 'o0' / 'stitchy' / 'hits.tsv.bgz')
    assert [(int(r['start']), int(r['end'])) for r in rows] == [(24_000, 26_500), (49_990, 50_010),
                                                                (99_000, 101_000), (160_000, 160_250)]


def _scan_files(d: Path):
    return sorted(p.relative_to(d).as_posix() for p in d.rglob('*')
                  if p.is_file() and 'work' not in p.relative_to(d).parts and p.name not in ('run.json', '.scan.lock')
                  and not p.name.endswith('.csi'))


def test_byte_identical_rerun_and_parallel(tmp_path, synth_bam):
    _run(tmp_path / 'r1', synth_bam, modules=('plant', 'stitchy', 'molly'), annotate=True)
    _run(tmp_path / 'r2', synth_bam, modules=('plant', 'stitchy', 'molly'), annotate=True, jobs=2)
    f1, f2 = _scan_files(tmp_path / 'r1'), _scan_files(tmp_path / 'r2')
    assert f1 == f2 and 'manifest.json' in f1 and any(f.endswith('.scan.bam') for f in f1)
    for f in f1:
        assert (tmp_path / 'r1' / f).read_bytes() == (tmp_path / 'r2' / f).read_bytes(), f


def test_continue_reuses_units_and_refuses_changes(tmp_path, synth_bam):
    from fiberhmm.scan.runner import ScanRefused
    out = tmp_path / 'c'
    _run(out, synth_bam, annotate=False)
    first = (out / 'plant' / 'hits.tsv.bgz').read_bytes()
    victim = sorted((out / 'work' / 'plant').iterdir())[2]
    (victim / 'marker.json').unlink()
    _run(out, synth_bam, annotate=False, cont=True)
    assert (out / 'plant' / 'hits.tsv.bgz').read_bytes() == first
    run = json.loads((out / 'run.json').read_text())
    assert run['attempts'][-1]['units_run'] == 1
    with pytest.raises(ScanRefused):
        _run(out, synth_bam, annotate=False)                      # not empty, no --continue
    with pytest.raises(ScanRefused):
        _run(out, synth_bam, annotate=False, cont=True, params={'plant': {'features': FEATURES[:2], 'boundaries': []}})


def test_scalar_group_in_annotated_bam(tmp_path, synth_bam):
    from fiberhmm.io.scan_ma import scan_contracts, scan_groups_of
    _run(tmp_path / 'a', synth_bam, modules=('molly',), annotate=True)
    m = json.loads((tmp_path / 'a' / 'manifest.json').read_text())
    path = tmp_path / 'a' / m['outputs']['annotated'][0]['path']
    with pysam.AlignmentFile(str(path)) as bam:
        contracts = scan_contracts(bam.header)
        assert set(contracts) == {'mollymol'} and contracts['mollymol']['kind'] == 'scalar'
        n = 0
        for read in bam.fetch(until_eof=True):
            g = scan_groups_of(read, contracts)['mollymol']
            assert len(g) == 1 and g[0]['start'] == 0 and g[0]['length'] == read.query_length
            assert g[0]['scalars']['length'] == read.query_length
            n += 1
    assert n == m['outputs']['annotated'][0]['records'] == m['outputs']['annotated'][0]['marked_records']


def test_tsv_fallback_molecules_table(tmp_path, synth_bam):
    _run(tmp_path / 't', synth_bam, modules=('molly',), annotate=False)
    rows = read_all(tmp_path / 't' / 'molly' / 'molecules.tsv.bgz')
    assert rows and all(r['molecule'].startswith('mol:v1 ') for r in rows)
    assert len({r['record'] for r in rows}) == len(rows)            # each record exactly once
    with pysam.AlignmentFile(str(synth_bam)) as bam:
        assert len(rows) == sum(1 for _ in bam.fetch(until_eof=True))


# ------------------------------------------------------------------ review findings (round 1)
def test_unit_ids_unique_when_spans_repeat():
    units, _ = plan({'c': 10_000}, genome=True, grid_offset=5_000)
    assert [(u.zone_start, u.zone_end) for u in units] == [(0, 5_000), (5_000, 10_000)]
    assert len({u.unit_id for u in units}) == len(units) == 2
    assert units[0].start == units[1].start == 0 and units[0].end == units[1].end == 10_000
    h = Hit('t', 'x', 'c', 1000, 1100, 1050)
    kept, _c, _o = reconcile_anchor([(units[0], [h]), (units[1], [Hit('t', 'y', 'c', 1000, 1100, 1050)])])
    assert len(kept) == 1


def test_overlapping_version_outside_the_window_is_decided_with_the_component():
    us = _units3()
    u1, u2 = us[1], us[2]
    a = Hit('t', 'f', 'c1', 47_000, 47_900, 47_450)      # meets the 50 kb boundary window
    b = Hit('t', 'f', 'c1', 47_800, 52_300, 50_050)
    kept, comps, _ = reconcile_anchor([(u1, [a]), (u2, [b])])
    assert len(kept) == 1 and comps == 1


def test_reannotation_replaces_older_marks(tmp_path, synth_bam):
    from fiberhmm.io.scan_ma import scan_contracts, scan_groups_of
    from fiberhmm.scan import run_units
    units = plan({'chrS': 200_000}, genome=True)
    p1 = {'plant': {'features': [['A', 10_000, 10_300, 'both']], 'boundaries': [], 'states': True}}
    run_units([{'dataset_id': 'synth', 'paths': [synth_bam]}], units, modules=['plant'], params=p1,
              out=tmp_path / 'one', annotate=True, registry=REG)
    first = next((tmp_path / 'one' / 'annotated').glob('*.scan.bam'))
    p2 = {'plant': {'features': [['B', 150_000, 150_300, 'both']], 'boundaries': [], 'states': True}}
    run_units([{'dataset_id': 'synth', 'paths': [str(first)]}], units, modules=['plant'], params=p2,
              out=tmp_path / 'two', annotate=True, registry=REG)
    second = next((tmp_path / 'two' / 'annotated').glob('*.scan.bam'))
    with pysam.AlignmentFile(str(second)) as bam:
        contracts = scan_contracts(bam.header)
        names = {g['name'] for r in bam.fetch(until_eof=True)
                 for g in scan_groups_of(r, contracts).get('plantst', [])}
    assert names and all(':150150:' in n for n in names)              # only the second scan's hit ids


def test_relabelled_datasets_refuse_continue_and_extend_recomputes(tmp_path, synth_bam):
    from fiberhmm.scan import run_units
    from fiberhmm.scan.runner import ScanRefused
    units = plan({'chrS': 200_000}, regions=[('chrS', 0, 100_000)])
    feats = {'plant': {'features': FEATURES[:4], 'boundaries': [25_000, 50_000, 75_000]}}
    out = tmp_path / 's'
    run_units([{'dataset_id': 'x', 'paths': [synth_bam]}], units, modules=['plant'], params=feats, out=out,
              annotate=False, registry=REG)
    with pytest.raises(ScanRefused):
        run_units([{'dataset_id': 'y', 'paths': [synth_bam]}], units, modules=['plant'], params=feats, out=out,
                  annotate=False, registry=REG, continue_run=True)


def test_extend_reuses_units_of_the_earlier_plan(tmp_path, synth_bam):
    from fiberhmm.scan import run_units
    ds = [{'dataset_id': 'synth', 'paths': [synth_bam]}]
    feats = {'plant': {'features': FEATURES, 'boundaries': list(range(0, 200_001, 25_000))}}
    small = plan({'chrS': 200_000}, regions=[('chrS', 0, 100_000)])
    both = plan({'chrS': 200_000}, regions=[('chrS', 0, 100_000), ('chrS', 120_000, 200_000)])
    out = tmp_path / 'ext'
    run_units(ds, small, modules=['plant'], params=feats, out=out, annotate=False, registry=REG)
    run_units(ds, both, modules=['plant'], params=feats, out=out, annotate=False, registry=REG,
              continue_run=True, extend=True)
    run = json.loads((out / 'run.json').read_text())
    assert run['attempts'][-1]['units_reused'] == len(small[0])
    run_units(ds, both, modules=['plant'], params=feats, out=tmp_path / 'fresh', annotate=False, registry=REG)
    assert (out / 'plant' / 'hits.tsv.bgz').read_bytes() == (tmp_path / 'fresh' / 'plant' / 'hits.tsv.bgz').read_bytes()


# ------------------------------------------------------------------ review findings (round 2)
def test_extend_keeps_one_molecule_result_per_record(tmp_path, synth_bam):
    from fiberhmm.scan import run_units
    ds = [{'dataset_id': 'synth', 'paths': [synth_bam]}]
    out = tmp_path / 'm'
    run_units(ds, plan({'chrS': 200_000}, regions=[('chrS', 20_000, 21_000)]), modules=['molly'], out=out,
              annotate=False, registry=REG)
    both = plan({'chrS': 200_000}, regions=[('chrS', 10_000, 11_000), ('chrS', 20_000, 21_000)])
    run_units(ds, both, modules=['molly'], out=out, annotate=False, registry=REG, continue_run=True, extend=True)
    rows = read_all(out / 'molly' / 'molecules.tsv.bgz')
    assert len({r['record'] for r in rows}) == len(rows)
    run_units(ds, both, modules=['molly'], out=tmp_path / 'fresh', annotate=False, registry=REG)
    assert (out / 'molly' / 'molecules.tsv.bgz').read_bytes() == \
        (tmp_path / 'fresh' / 'molly' / 'molecules.tsv.bgz').read_bytes()


def test_unit_directories_are_injective():
    assert I.unit_dir_name('c#1:0-10000') != I.unit_dir_name('c_1:0-10000')


def test_feature_reaching_several_seams_is_kept_once():
    units, _ = plan({'c': 10_000}, genome=True, unit_bp=5_000, stride_bp=250)
    by_idx = {u.index: u for u in units}
    spans = {10: (2_600, 2_800), 11: (2_500, 2_950), 12: (2_700, 3_600)}
    results = [(u, [Hit('fp.class', 'k', 'c', *spans[u.index], (spans[u.index][0] + spans[u.index][1]) // 2)]
                if u.index in spans else []) for u in units]
    kept, _c, _o = reconcile_anchor(results)
    kept_spans = [(h.start, h.end) for _u, h in kept]
    for a in kept_spans:
        for b in kept_spans:
            if a != b:
                assert not (a[0] < b[1] and b[0] < a[1])
    assert len(kept) >= 1


def test_scan_groups_keep_legacy_only_calls_readable():
    import array
    from fiberhmm.io.scan_ma import replace_scan_groups
    from fiberhmm.io.ma_tags import parse_ma_tag
    h = pysam.AlignmentHeader.from_dict({'HD': {'VN': '1.6'}, 'SQ': [{'SN': 'c', 'LN': 5000}]})
    r = pysam.AlignedSegment(h)
    r.query_name = 'm/1/ccs'
    r.query_sequence = 'A' * 1000
    r.reference_id = 0
    r.reference_start = 0
    r.cigarstring = '1000M'
    r.set_tag('ns', array.array('I', [100, 500]))
    r.set_tag('nl', array.array('I', [100, 100]))
    replace_scan_groups(r, {'nfrmol': ('QQQQ', [(0, 1000, [150, 1, 1, 44], 'nfrmol')])})
    parsed = parse_ma_tag(r.get_tag('MA'))
    assert parsed['nuc'] == [(100, 100), (500, 100)]
    assert [t[0] for t in parsed['raw_types']] == ['nuc', 'nfrmol']


def test_repeated_contigs_and_large_coordinates(tmp_path):
    units, _ = plan({'c': 60_000}, genome=True, contigs=['c', 'c'])
    assert len({u.unit_id for u in units}) == len(units) == 3
    with pytest.raises(PlanError):
        plan({'#c': 60_000}, genome=True)
    w = TableWriter(tmp_path / 'big.tsv.bgz', ['chrom', 'start', 'end', 'idx_start', 'idx_end', 'x'], {'c': 0})
    w.add('c', 600_000_000, 600_000_010, ['far'])
    w.close()
    assert (tmp_path / 'big.tsv.bgz.csi').is_file() and not (tmp_path / 'big.tsv.bgz.tbi').exists()
    assert [r['x'] for r in Table(tmp_path / 'big.tsv.bgz').fetch('c', 600_000_005, 600_000_006)] == ['far']
