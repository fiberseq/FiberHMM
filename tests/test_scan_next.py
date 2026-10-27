"""fiberhmm-scan follow-ups requested by the browser hub (contracts v1.9).

- ``input_index`` in the per-record tables (states, marks, molecules, evidence units);
- ``--zones``: a subset of the genome grid, units unchanged, results reusable by and from a genome scan;
- dataset labels by canonical order: a scan with the BAMs listed in another order is byte-identical;
- the unit survey is keyed by the unit, not by its place in the plan (extensions reuse it).
"""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

pytest.importorskip('pysam')

from fiberhmm.scan.tables import read_all
from fiberhmm.scan.testing import (PlantedAnchorModule, PlantedMoleculeModule, PlantedStitchModule,
                                   write_synthetic_bam)
from fiberhmm.scan.units import PlanError, plan

REG = {'plant': PlantedAnchorModule, 'stitchy': PlantedStitchModule, 'molly': PlantedMoleculeModule}
REF = {'chrS': 200_000}
FEATURES = [['A', 10_000, 10_300, 'both'], ['D', 49_900, 50_400, 'both'], ['G', 74_000, 76_500, 'both'],
            ['H', 99_800, 100_300, 'both'], ['K', 124_400, 124_700, 'both'], ['M', 150_000, 150_600, 'both']]
RUNS = [[24_000, 26_500], [49_990, 50_010], [61_000, 63_000], [99_000, 101_000], [160_000, 160_250]]
PARAMS = {'plant': {'features': FEATURES, 'boundaries': list(range(0, 200_001, 25_000)), 'states': True},
          'stitchy': {'runs': RUNS}}


@pytest.fixture(scope='module')
def bams(tmp_path_factory):
    d = tmp_path_factory.mktemp('next_bams')
    a = write_synthetic_bam(d / 'a.bam', seed=1)
    b = write_synthetic_bam(d / 'b.bam', seed=2, step=2_500)
    return a, b


def _run(out, datasets, units, *, modules=('plant', 'stitchy', 'molly'), annotate=False, cont=False, extend=False):
    from fiberhmm.scan import run_units
    return run_units(datasets, units, modules=list(modules), params={k: v for k, v in PARAMS.items() if k in modules},
                     out=out, annotate=annotate, registry=REG, continue_run=cont, extend=extend)


def _rows(path):
    return [dict(r) for r in read_all(path)]


# ------------------------------------------------------------------ input_index
def test_per_record_tables_name_the_input_file(tmp_path, bams):
    a, b = bams
    units = plan(REF, regions=[('chrS', 40_000, 110_000)])
    _run(tmp_path / 's', [{'dataset_id': 'pooled', 'paths': [a, b]}], units)
    import pysam
    from fiberhmm.scan.identity import record_sha256
    sha_of = []
    for p in (a, b):
        with pysam.AlignmentFile(p, 'rb') as bam:
            sha_of.append({record_sha256(r) for r in bam.fetch('chrS')})
    for table in ('plant/states.tsv.bgz', 'molly/molecules.tsv.bgz'):
        rows = _rows(tmp_path / 's' / table)
        assert rows and {r['input_index'] for r in rows} == {'0', '1'}, table
        cols = list(rows[0])
        assert cols.index('input_index') == cols.index('dataset_id') + 1
        for r in rows:
            assert r['record'].split(':')[0] in sha_of[int(r['input_index'])]
    m = json.loads((tmp_path / 's' / 'manifest.json').read_text())
    assert [f['input_index'] for f in m['datasets'][0]['files']] == [0, 1]


# ------------------------------------------------------------------ labels by canonical order
def _scan_cli(argv):
    from fiberhmm.scan.cli import main
    with redirect_stdout(io.StringIO()) as out:
        assert main(argv) == 0
    return json.loads(out.getvalue().strip().splitlines()[-1])


def _outputs(d: Path):
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob('*'))
            if p.is_file() and 'work' not in p.relative_to(d).parts and p.name not in ('run.json', '.scan.lock')
            and not p.name.endswith('.csi')}


def test_bam_order_does_not_change_labels_keys_or_rows(tmp_path, bams):
    a, b = bams
    common = ['--region', 'chrS:40000-90000', '--modules', 'peaks', '--no-annotate']
    s1 = _scan_cli(['--bam', a, '--bam', b] + common + ['-o', str(tmp_path / 'ab')])
    s2 = _scan_cli(['--bam', b, '--bam', a] + common + ['-o', str(tmp_path / 'ba')])
    assert s1['scan_id'] == s2['scan_id']
    assert _outputs(tmp_path / 'ab') == _outputs(tmp_path / 'ba')
    m = json.loads((tmp_path / 'ab' / 'manifest.json').read_text())
    assert [(d['dataset_id'], d['ordinal']) for d in m['datasets']] == [('dataset_1', 0), ('dataset_2', 1)]
    # the API labels unlabelled datasets the same way, in any order; explicit labels are kept
    units = plan(REF, regions=[('chrS', 40_000, 90_000)])
    _run(tmp_path / 'api_ab', [{'paths': [a]}, {'paths': [b]}], units, modules=('plant',))
    _run(tmp_path / 'api_ba', [{'paths': [b]}, {'paths': [a]}], units, modules=('plant',))
    assert _outputs(tmp_path / 'api_ab') == _outputs(tmp_path / 'api_ba')
    from fiberhmm.scan.runner import ScanRefused, resolve_datasets
    resolved, _ref = resolve_datasets([{'paths': [a]}, {'paths': [b]}])
    first, second = resolved[0].paths[0], resolved[1].paths[0]          # ordinals 0 and 1
    again, _ref = resolve_datasets([{'paths': [second]}, {'dataset_id': 'mine', 'paths': [first]}])
    assert [(d.dataset_id, d.ordinal) for d in again] == [('mine', 0), ('dataset_2', 1)]
    with pytest.raises(ScanRefused):
        resolve_datasets([{'paths': [first]}, {'dataset_id': 'dataset_1', 'paths': [second]}])


# ------------------------------------------------------------------ --zones
def test_zones_plan_is_a_subset_of_the_genome_grid():
    units, info = plan(REF, zones=[('chrS', 60_000, 90_000)])
    genome, _ = plan(REF, genome=True)
    by_id = {u.unit_id: u for u in genome}
    assert [u.unit_id for u in units] == ['chrS:50000-75000', 'chrS:75000-100000']
    for i, u in enumerate(units):
        g = by_id[u.unit_id]
        assert (u.start, u.end, u.zone_start, u.zone_end, u.left, u.right, u.mode) == \
               (g.start, g.end, g.zone_start, g.zone_end, g.left, g.right, 'grid') and u.index == i
    assert info['mode'] == 'grid' and info['zones'] == [['chrS', 60_000, 90_000]] and info['contigs'] is None
    # touching a zone boundary does not select the neighbour; overlapping windows select each unit once
    u2, _ = plan(REF, zones=[('chrS', 75_000, 75_001), ('chrS', 70_000, 80_000)])
    assert [u.unit_id for u in u2] == ['chrS:50000-75000', 'chrS:75000-100000']
    with pytest.raises(PlanError):
        plan(REF, zones=[('chrS', 0, 300_000)])
    with pytest.raises(PlanError):
        plan(REF, zones=[('chrS', 0, 10)], contigs=['chrS'])


def test_zones_scan_reuses_and_equals_the_genome_scan(tmp_path, bams):
    """A --zones scan computes the genome scan's units: an extension of a genome scan's directory reuses every
    one of them, and its rows equal the genome scan's away from the subset's edges (units whose two grid
    neighbours are planned). At the edges anchor hits fall back to anchor-in-zone and stitched hits that reach
    the edge are flagged open_end."""
    a, _b = bams
    ds = [{'paths': [a]}]
    genome, _ = plan(REF, genome=True)
    zones = plan(REF, zones=[('chrS', 40_000, 135_000)])
    _run(tmp_path / 'g', ds, genome)
    _run(tmp_path / 'z', ds, zones)
    import shutil
    shutil.copytree(tmp_path / 'g', tmp_path / 'gz')
    _run(tmp_path / 'gz', ds, zones, cont=True, extend=True)
    att = json.loads((tmp_path / 'gz' / 'run.json').read_text())['attempts'][-1]
    assert att['units_run'] == 0 and att['units_reused'] == len(zones[0])
    assert _outputs(tmp_path / 'gz') == _outputs(tmp_path / 'z')      # same plan, same bytes
    zu = zones[0]
    planned = {u.unit_id for u in zu}
    interior = {u.unit_id for u in zu if u.left in planned and u.right in planned}
    owned = [(u.zone_start, u.zone_end) for u in zu if u.unit_id in interior]
    g_hits = [r for r in _rows(tmp_path / 'g' / 'plant' / 'hits.tsv.bgz') if r['unit_id'] in interior]
    z_hits = [r for r in _rows(tmp_path / 'z' / 'plant' / 'hits.tsv.bgz') if r['unit_id'] in interior]
    assert g_hits and g_hits == z_hits
    for table in ('plant/states.tsv.bgz', 'plant/values.tsv.bgz'):
        ids = {r['hit_id'] for r in g_hits}
        assert [r for r in _rows(tmp_path / 'g' / table) if r['hit_id'] in ids] == \
               [r for r in _rows(tmp_path / 'z' / table) if r['hit_id'] in ids]

    def inner(r):
        return any(a0 <= int(r['start']) and int(r['end']) <= z0 for a0, z0 in owned)
    g_mol = [r for r in _rows(tmp_path / 'g' / 'molly' / 'molecules.tsv.bgz') if inner(r)]
    z_mol = [r for r in _rows(tmp_path / 'z' / 'molly' / 'molecules.tsv.bgz') if inner(r)]
    assert g_mol and g_mol == z_mol
    g_st = _rows(tmp_path / 'g' / 'stitchy' / 'hits.tsv.bgz')
    z_st = _rows(tmp_path / 'z' / 'stitchy' / 'hits.tsv.bgz')
    lo, hi = zu[0].zone_start, zu[-1].zone_end
    for r in z_st:
        if int(r['start']) > lo and int(r['end']) < hi:
            assert r in g_st and 'open_end' not in r['flags']
    m = json.loads((tmp_path / 'z' / 'manifest.json').read_text())
    assert m['units']['zones'] == [['chrS', 40_000, 135_000]] and m['units']['contigs'] is None


def test_zones_plan_edge_flags_stitched_hits_open(tmp_path, bams):
    a, _b = bams
    zones = plan(REF, zones=[('chrS', 62_000, 70_000)])           # one unit, zone 50000-75000
    _run(tmp_path / 'z', [{'paths': [a]}], zones, modules=('stitchy',))
    rows = _rows(tmp_path / 'z' / 'stitchy' / 'hits.tsv.bgz')
    flags = {(int(r['start']), int(r['end'])): r['flags'] for r in rows}
    assert sorted(flags) == [(50_000, 50_010), (61_000, 63_000)]
    assert 'open_end' in flags[(50_000, 50_010)] and 'open_end' not in flags[(61_000, 63_000)]
    edge = plan(REF, zones=[('chrS', 30_000, 40_000)])            # zone 25000-50000: run [24000, 26500) reaches out
    _run(tmp_path / 'e', [{'paths': [a]}], edge, modules=('stitchy',))
    rows = _rows(tmp_path / 'e' / 'stitchy' / 'hits.tsv.bgz')
    starts = {(int(r['start']), int(r['end'])): r['flags'] for r in rows}
    assert 'open_end' in starts[(25_000, 26_500)]                  # clipped at the plan edge, flagged
    assert 'open_end' in starts[(49_990, 50_000)]


def test_zones_cli_line(tmp_path, bams):
    a, _b = bams
    s = _scan_cli(['--bam', a, '--zones', 'chrS:60000-90000', '--modules', 'peaks', '--no-annotate',
                   '-o', str(tmp_path / 'z1')])
    t = _scan_cli(['--bam', a, '--genome', '--zones', 'chrS:60000-90000', '--modules', 'peaks', '--no-annotate',
                   '-o', str(tmp_path / 'z2')])
    assert s['units'] == t['units'] == 2 and _outputs(tmp_path / 'z1') == _outputs(tmp_path / 'z2')
    from fiberhmm.scan.cli import main
    with pytest.raises(SystemExit):
        with redirect_stdout(io.StringIO()):
            main(['--bam', a, '--zones', 'chrS:1-10', '--region', 'chrS:1-10', '-o', str(tmp_path / 'x')])


# ------------------------------------------------------------------ survey keyed by the unit
def test_extension_reuses_surveys_and_unit_results(tmp_path, bams, monkeypatch):
    a, _b = bams
    from fiberhmm.scan import runner
    calls = []
    real = runner.unit_survey

    def counting(*args, **kw):
        calls.append(args[1].unit_id)
        return real(*args, **kw)
    monkeypatch.setattr(runner, 'unit_survey', counting)
    first = plan(REF, zones=[('chrS', 60_000, 90_000)])
    _run(tmp_path / 's', [{'paths': [a]}], first, modules=('plant',))
    assert len(calls) == 2
    calls.clear()
    second = plan(REF, zones=[('chrS', 10_000, 120_000)])         # indices and neighbours change
    _run(tmp_path / 's', [{'paths': [a]}], second, modules=('plant',), cont=True, extend=True)
    assert sorted(calls) == sorted(u.unit_id for u in second[0] if u.unit_id not in {x.unit_id for x in first[0]})
    att = json.loads((tmp_path / 's' / 'run.json').read_text())['attempts'][-1]
    assert att['units_reused'] == 2 and att['units_run'] == len(second[0]) - 2
