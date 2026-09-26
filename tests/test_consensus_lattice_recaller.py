"""Lattice recaller engine: model invariants, planted-class recovery, outputs, options and frozen-reference regression."""
import copy
import gzip
import io
import json
import math
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

from fiberhmm.inference.consensus.lattice_recaller import discovery as D, model as Mo
from fiberhmm.inference.consensus.parameters import RecallerOptions, parse_options
from fiberhmm.inference.consensus.workflow import run_workflow

PA, PP = 0.8, 0.02


def lattice(pos, hits):
    pos = np.asarray(pos, np.int64); hit = np.asarray(hits, bool)
    pa = np.full(len(pos), PA); pp = np.full(len(pos), PP)
    d = np.where(hit, np.log(pp) - np.log(pa), np.log1p(-pp) - np.log1p(-pa))
    return dict(uid='u', pos=pos, hit=hit, pa=pa, pp=pp, d=d)


CLASS = dict(L=[1176, 1184], R=[1206, 1214], span=(1180., 1210.))
SITES = np.arange(1100, 1300, 4)


def scores(protected=(), marked_inside=()):
    hits = [(not any(a <= p < b for a, b in protected)) or p in marked_inside for p in SITES]
    return Mo.joint(lattice(SITES, hits), [CLASS], 0.5, RecallerOptions())


def test_all_marked_molecule_is_accessible():
    s = scores()
    assert int(np.argmax(s)) == 3            # [class, broader, other, accessible]


def test_class_shaped_gap_with_marked_linkers_is_the_class():
    s = scores(protected=[(1180, 1210)])
    assert int(np.argmax(s)) == 0 and s[0] > s[1] + 2 and s[0] > s[3] + 5


def test_unmarked_run_through_both_edge_boxes_is_broader():
    s = scores(protected=[(1120, 1280)])
    assert s[1] > s[0] and s[1] > s[3]


def test_marked_core_is_not_the_class():
    s = scores(protected=[(1180, 1210)], marked_inside={1188, 1192, 1196, 1200})
    assert s[0] < max(s[2], s[3])


def test_molecule_not_spanning_the_group_is_skipped():
    short = lattice(np.arange(1170, 1200, 4), [True]*8)
    assert Mo.joint(short, [CLASS], 0.5, RecallerOptions()) is None


def test_absent_class_earns_no_held_out_support():
    rng = np.random.default_rng(3); units = []
    for i in range(80):
        u = lattice(SITES, rng.random(len(SITES)) < PA); u['uid'] = f'm{i}'; units.append(u)
    res = Mo.fit_channel(units, [CLASS], 0.5, RecallerOptions(learned_spots=False))
    assert res['w'][0] < 0.02 and res['support_gain'][0] < 5.


def test_core_width_uses_core_rule_boxes():
    g = dict(L=[100, 120], R=[110, 140], span=(105., 125.))
    assert D.core_width(g) == -10
    assert D.core_width(dict(g, core_bp=4)) == 4


def test_overlap_groups_join_nested_and_partial_classes():
    cls = [dict(span=(0., 20.)), dict(span=(10., 30.)), dict(span=(50., 60.)), dict(span=(5., 12.))]
    groups = sorted(sorted(g) for g in D.overlap_groups(cls))
    assert groups == [[0, 1, 3], [2]]


def test_recaller_options_validate():
    opts = parse_options({'cr': {'engine': 'lattice_recaller'}})
    assert opts['recaller'].stringency == 0.9 and opts['recaller'].linker == 'both'
    with pytest.raises(ValueError):
        parse_options({'cr': {'engine': 'lattice_recaller'}, 'recaller': {'core_quantile_low': 50., 'core_quantile_high': 50.}})
    with pytest.raises(ValueError):
        parse_options({'cr': {'engine': 'lattice_recaller'}, 'recaller': {'linker': 'none'}})


def test_cli_schema_defaults_to_lattice_recaller():
    from fiberhmm.inference.consensus.cli import main
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(['--schema'])
    schema = json.loads(buf.getvalue())
    assert next(c for c in schema['cr'] if c['name'] == 'engine')['default'] == 'lattice_recaller'
    assert any(c['name'] == 'stringency' for c in schema['recaller'])


# ---------------------------------------------------------------- planted class, end to end
def planted_units(n, footprint, always_marked=(), occupancy=.5, seed=3):
    rng = np.random.default_rng(seed); pos = np.arange(1100, 1300, 4); out = []
    for i in range(n):
        prot = (rng.random() < occupancy) & (pos >= footprint[0]) & (pos < footprint[1])
        hit = np.where(prot, rng.random(len(pos)) < PP, rng.random(len(pos)) < PA)
        for a in always_marked:
            hit[pos == a] = rng.random() < .99
        u = lattice(pos, hit); u['uid'] = f'm{i}'; out.append(u)
    return out


def test_edge_contraction_moves_a_box_past_an_always_marked_site_only_where_one_exists():
    import dataclasses
    g = dict(L=[1170, 1190], R=[1206, 1214], span=(1180., 1210.))
    opt = dataclasses.replace(RecallerOptions(), edge_contraction=True, learned_spots=False)
    narrow = Mo.fit_channel(planted_units(400, (1188, 1210), always_marked=(1184,)), [g], .5, opt)
    assert narrow['gs'][0]['L'] == [1185, 1188] and narrow['gs'][0]['R'] == [1206, 1214]
    assert narrow['edges'][0].startswith('left:1170-1190>1185-1188') and abs(narrow['w'][0] - .5) < .06
    full = Mo.fit_channel(planted_units(400, (1180, 1210), seed=4), [g], .5, opt)
    assert full['gs'][0]['L'] == [1170, 1190] and full['edges'][0] == ''
    off = Mo.fit_channel(planted_units(400, (1188, 1210), always_marked=(1184,)), [g], .5, dataclasses.replace(opt, edge_contraction=False))
    assert off['gs'][0]['L'] == [1170, 1190] and off['edges'][0] == ''


def planted_payload(n=160, occupancy=0.4, seed=7):
    """DAF-like dataset, two strands, one 30-bp class at 1180-1210 bound in `occupancy` of molecules."""
    rng = np.random.default_rng(seed); units = []
    for i in range(n):
        strand = 'CT' if i % 2 == 0 else 'GA'; off = 0 if strand == 'CT' else 2
        pos = np.arange(900 + off, 1500, 4); bound = rng.random() < occupancy
        jl, jr = rng.integers(-1, 2, size=2); a, b = 1180 + jl, 1210 + jr
        prot = bound & (pos >= a) & (pos < b)
        hit = np.where(prot, rng.random(len(pos)) < PP, rng.random(len(pos)) < PA)
        calls = [dict(interval=[int(a), int(b)], llr=12.0)] if bound else []
        units.append(dict(unit_id=f'mol{i:04d}', read_name=f'read{i:04d}', strand=strand, reference_start=880, reference_end=1520,
                          positions=pos.tolist(), hits=hit.astype(int).tolist(), p_accessible=[PA]*len(pos), p_protected=[PP]*len(pos),
                          native_multi_interval_calls=calls, native_multi_interval_tf_intervals=[c['interval'] for c in calls],
                          raw_tf_intervals=[c['interval'] for c in calls], representative_raw_tf_intervals=[c['interval'] for c in calls],
                          raw_nuc_intervals=[], msp_intervals=[[880, 1520]], contexts=[0]*len(pos)))
    return dict(region=dict(chrom='chrT', start=1050, end=1350), strata=[dict(dataset_id='planted', stratum_id='planted', chemistry='ddda',
                                                                            model_manifest=dict(preset='ddda'), units=units)])


def test_planted_class_is_recovered_with_all_outputs(tmp_path):
    res = run_workflow(planted_payload(), {'cr': {'engine': 'lattice_recaller'}, 'recaller': {'learned_spots': False}}, tmp_path)
    assert res['cr_mode'] == 'lattice_recaller'
    for name in ('classes.tsv', 'molecules.tsv.gz', 'result.json.gz', 'manifest.json', 'evidence.json.gz', 'families.tsv', 'calls.tsv'):
        assert (tmp_path/name).exists(), name
    rows = res['recaller']['rows']; classes = res['recaller']['classes']
    hit = [c for c in classes if abs(c['start'] - 1180) <= 3 and abs(c['end'] - 1210) <= 3]
    assert len(hit) == 1
    est = {r['strand']: r['prevalence'] for r in rows if r['class_id'] == hit[0]['id']}
    assert set(est) == {'CT', 'GA'} and all(0.28 <= v <= 0.52 for v in est.values())
    assert all(r['supported'] for r in rows if r['class_id'] == hit[0]['id'])
    catalog = res['datasets']['planted']['cr']['catalog']
    assert any(f['family'] == hit[0]['id'] and set(f['classification_counts']) == {'CT', 'GA'} for f in catalog)
    assigned = [p for r in res['datasets']['planted']['cr']['records'] for p in r['proposals'] if p['family'] == hit[0]['id']]
    assert len(assigned) > 30
    with gzip.open(tmp_path/'molecules.tsv.gz', 'rt') as fh:
        header = fh.readline().rstrip('\n').split('\t')
    assert header == ['class_id', 'channel', 'unit_id', 'posterior', 'log_bf', 'label', 'start', 'end']
    assert (tmp_path/'broader.tsv.gz').exists()
    # The recaller's own calls: every member molecule of the class, with its own edges near the planted footprint.
    rc = [c for r in res['datasets']['planted']['cr']['records'] for c in r.get('recaller_calls', []) if c['family'] == hit[0]['id']]
    assert len(rc) > 30 and all(c['kind'] == 'class' for c in rc)
    assert all(abs(c['interval'][0] - 1180) <= 8 and abs(c['interval'][1] - 1210) <= 8 for c in rc)
    assert all(c['consensus_interval'] == [round(hit[0]['start']), round(hit[0]['end'])] for c in rc)


def test_core_rule_can_drop_every_class(tmp_path):
    res = run_workflow(planted_payload(), {'cr': {'engine': 'lattice_recaller'}, 'recaller': {'learned_spots': False, 'minimum_core_bp': 1000}}, tmp_path)
    assert res['recaller']['classes'] == [] and res['manifest']['recaller']['dropped_by_core_rule']


def test_payload_is_not_mutated(tmp_path):
    payload = planted_payload(n=80); before = copy.deepcopy(payload)
    run_workflow(payload, {'cr': {'engine': 'lattice_recaller'}, 'recaller': {'learned_spots': False}}, tmp_path)
    assert payload == before


# ---------------------------------------------------------------- frozen prototype reference (local data only)
REFERENCE = Path.home()/'fiberhmm_work/cr_recaller_reference_20260925'
NAPA = Path.home()/'fiberhmm_work/napa_keepone_20260923/napa_full_keepone/evidence.json.gz'


@pytest.mark.skipif(not (REFERENCE/'napa_N1.classes.tsv').exists() or not NAPA.exists(), reason='frozen reference data not on this machine')
def test_matches_frozen_prototype_reference_napa(tmp_path):
    import csv
    payload = json.load(gzip.open(NAPA)); payload['region'] = dict(chrom='chr19', start=47514980, end=47515330)
    res = run_workflow(payload, {'cr': {'engine': 'lattice_recaller'}, 'families': {'recall_hia5_nucleosomes': False},
                                 'recaller': {'core_quantile_low': 10., 'core_quantile_high': 90.}}, tmp_path)
    strand = {'DAF_CT': 'CT', 'DAF_GA': 'GA', 'Hia5': 'pooled'}
    with open(REFERENCE/'napa_N1.classes.tsv') as fh:
        ref = list(csv.DictReader(fh, delimiter='\t'))
    got = {(round(r['start'], 1), round(r['end'], 1), r['strand']): r for r in res['recaller']['rows']}
    assert len(got) == len(ref)
    for r in ref:
        g = got[(float(r['left']), float(r['right']), strand[r['channel']])]
        assert math.isclose(g['prevalence'], float(r['em']), abs_tol=1e-3)
        assert math.isclose(g['support_gain_nats'], float(r['support_gain']), abs_tol=0.05)
