"""Molecule budgets: stable priorities, balanced channels, frozen discovery and honest denominators."""
import json

import pytest

from fiberhmm.inference import depth_cap as C
from fiberhmm.inference.consensus.parameters import RecallerOptions, parse_options, options_dict, saved_parameters
from fiberhmm.inference.consensus.lattice_recaller import units as U, workflow as W
from fiberhmm.inference.consensus.workflow import run_workflow
from fiberhmm.inference.accessibility.workflow import NFROptions, run_accessibility
from test_consensus_lattice_recaller import planted_payload
from test_accessibility_nfr import planted, payload


def test_cap_is_order_independent_balanced_and_zero_is_exact_noop():
    us = [dict(uid=f'{ch}:{i}', ch=ch) for ch, n in [('deep::CT', 100), ('deep::GA', 50), ('shallow::CT', 3)] for i in range(n)]
    select = lambda rows, cap, seed=1: C.select(rows, cap, seed, lambda u: u['ch'], C.identity)
    one = select(us, 10)
    assert {u['uid'] for u in one} == {u['uid'] for u in select(list(reversed(us)), 10)}
    assert [sum(u['ch'] == ch for u in one) for ch in ('deep::CT', 'deep::GA', 'shallow::CT')] == [10, 10, 3]
    assert select(us, 0) is us
    assert select(us, 1000) == us
    assert one != select(us, 10, 2)
    # Equal names in separate libraries use the stable unit id to break ties, never the input order.
    tied = [dict(u, read_name='same') for u in us]
    assert {u['uid'] for u in select(tied, 10)} == {u['uid'] for u in select(tied[::-1], 10)}


def test_overlapping_tiles_use_the_same_priorities_and_pool_hia5():
    source = planted_payload()['strata'][0]
    ids = lambda rows: {u['uid'] for u in rows}
    a = U.tile_units([source], 1050, 1300, cap=15, seed=7)
    b = U.tile_units([source], 1100, 1350, cap=15, seed=7)
    assert ids(a) == ids(b) and len(a) == 30
    source = dict(source, chemistry='hia5-pacbio')
    assert len(U.tile_units([source], 1050, 1300, cap=15)) == 15
    # Removing candidates cannot evict a retained molecule that is still eligible.
    source = dict(source, units=source['units'][20:])
    reduced = ids(U.tile_units([source], 1050, 1300, cap=15, seed=7))
    original = ids(U.tile_units([dict(source, units=planted_payload()['strata'][0]['units'])], 1050, 1300, cap=15, seed=7))
    assert original & {u['unit_id'] for u in source['units']} <= reduced


def test_recaller_reports_per_phase_counts_and_quantification_keeps_catalogue(tmp_path):
    p = planted_payload(240, occupancy=.7)
    params = dict(compute=dict(cores=1), recaller=dict(max_discovery_molecules=60, max_quantification_molecules=0,
                                                     learned_spots=False, sample_seed=5))
    full = run_workflow(p, params, tmp_path/'full')
    params['recaller']['max_quantification_molecules'] = 35
    capped = run_workflow(p, params, tmp_path/'capped')
    geometry = lambda r: [(c['id'], c['start'], c['end'], c['L'], c['R']) for c in r['recaller']['classes']]
    assert geometry(full) and geometry(full) == geometry(capped)
    records = capped['manifest']['molecule_sampling']
    assert all(s['available'] == 120 for r in records for s in r['strata'])
    assert all(s[r['phase']] == (60 if r['phase'] == 'discovery' else 35) for r in records for s in r['strata'])
    assert all(r['molecules'] == 35 for r in capped['recaller']['rows'])
    assert not capped['manifest']['all_units']
    assert any('60 of 120' in w for w in capped['manifest']['data_warnings'])
    assert json.loads((tmp_path/'capped'/'manifest.json').read_text())['molecule_sampling'] == records


def test_zero_cap_omits_new_receipt_fields_and_saved_replay_stays_uncapped(tmp_path):
    values = dict(compute=dict(cores=1), recaller=dict(max_discovery_molecules=0, max_quantification_molecules=0, learned_spots=False))
    out = run_workflow(planted_payload(80), values, tmp_path)
    assert 'molecule_sampling' not in out['manifest']
    params = out['manifest']['parameters']
    assert not set(C.NAMES) & params['recaller'].keys()
    restored = parse_options(saved_parameters(params))
    assert restored['recaller'].max_discovery_molecules == restored['recaller'].max_quantification_molecules == 0
    assert options_dict(restored) == params
    nfr = NFROptions(max_discovery_molecules=0, max_quantification_molecules=0)
    assert NFROptions.from_params(nfr.as_dict()).max_discovery_molecules == 0
    assert not set(C.NAMES) & nfr.as_dict(receipt=True).keys()


def test_nfr_run_caps_discovery_and_quantification_per_strand(tmp_path):
    units, _ = planted([(0.6, [(950, 1250)]), (.4, [])], n=180)
    params = dict(nfr_regions=[(920, 1290)], bootstrap=0, kmax=2, splits=2, min_reads=5,
                  max_discovery_molecules=40, max_quantification_molecules=25, sample_seed=9)
    result = run_accessibility(payload(units), params, tmp_path)
    records = result['detection']['molecule_sampling']
    counts = records[-1]['strata']
    assert [(s['available'], s['discovery'], s['quantification']) for s in counts] == [(90, 40, 25)]*2
    assert result['nfrs'][0]['callable'] == 50
    assert result['nfrs'][0]['variants']
    assert any('25 of 90' in w for w in result['warnings'])
    manifest = json.loads((tmp_path/'manifest.json').read_text())
    assert manifest['detection']['molecule_sampling'] == records
    uncapped = run_accessibility(payload(units), dict(params, max_quantification_molecules=0))
    geom = lambda r: [(v['L'], v['R'], v['Lsd'], v['Rsd']) for v in r['nfrs'][0]['variants']]
    assert geom(result) == geom(uncapped)
    assert uncapped['nfrs'][0]['callable'] == 180


@pytest.mark.parametrize('name', C.NAMES)
@pytest.mark.parametrize('bad', [-1, True, 1.5])
def test_sampling_options_reject_invalid_values(name, bad):
    with pytest.raises(ValueError):
        parse_options({'recaller': {name: bad}})
    with pytest.raises(ValueError):
        NFROptions.from_params({name: bad})


def test_cli_flags_and_schema():
    from tools.gen_cli_reference import capture_parser
    from fiberhmm.inference.accessibility.cli import _parser
    from fiberhmm.inference.consensus.parameters import parameter_schema
    parser = capture_parser('fiberhmm.inference.consensus.cli', 'main')
    flags = ['--max-discovery-molecules', '17', '--max-quantification-molecules', '9', '--sample-seed', '3']
    for p in (parser, _parser()):
        args = p.parse_args(flags)
        assert [getattr(args, n) for n in C.NAMES] == [17, 9, 3]
    fields = {f['name']: f for f in parameter_schema()['recaller']}
    assert [fields[n]['default'] for n in C.NAMES] == [0, 0, 1]


def test_caps_filter_eligibility_before_selecting():
    source = planted_payload()['strata'][0]
    original = source['units']
    # These have the best hashes but cannot span the scoring window.
    ordered = sorted(original, key=lambda u: C.rank(C.identity(u), 1))
    short = [dict(u, positions=[1100, 1104, 1108], hits=[0]*3, p_accessible=[.8]*3, p_protected=[.02]*3)
             for u in ordered[:30]]
    source = dict(source, units=short + ordered[30:])
    counts = []
    chosen = U.tile_units([source], 1050, 1350, cap=20, counts=counts,
                          scoring_bounds={'planted::CT': (1150, 1240), 'planted::GA': (1150, 1240)})
    assert len(chosen) == 40
    assert not {u['uid'] for u in chosen} & {u['unit_id'] for u in short}
    assert len(counts[0]) == 130


def test_frozen_transfer_allows_budget_override_without_refitting(tmp_path, monkeypatch):
    from fiberhmm.inference.consensus.lattice_recaller import frozen as F
    p = planted_payload(160, occupancy=.7)
    run_workflow(p, dict(compute=dict(cores=1), recaller=dict(max_discovery_molecules=0, max_quantification_molecules=0,
                                                         learned_spots=False)), tmp_path/'source')
    catalog = F.freeze_run(tmp_path/'source', tmp_path/'catalog.json.gz')
    before = json.dumps(catalog, sort_keys=True)
    def forbidden(*args, **kwargs):
        raise AssertionError('Frozen transfer rediscovered or refitted classes')
    monkeypatch.setattr(W, 'discover', forbidden)
    monkeypatch.setattr(W.Mo, 'fit_channel', forbidden)
    r, _ = F.apply_catalog(catalog, p, tmp_path/'target', include_training=True, cores=1,
                           parameters={'recaller': {'max_quantification_molecules': 25, 'sample_seed': 4}})
    assert r['recaller']['rows'] and all(row['molecules'] == 25 for row in r['recaller']['rows'])
    assert before == json.dumps(catalog, sort_keys=True)
    assert r['manifest']['parameters']['recaller']['max_discovery_molecules'] == 0
    assert all(row['phase'] == 'quantification' for row in r['manifest']['molecule_sampling'])


def test_cli_caps_reach_both_engines(tmp_path):
    from fiberhmm.inference.consensus.artifacts import write_json, read_json
    from fiberhmm.inference.consensus.cli import main as consensus
    from fiberhmm.inference.accessibility.cli import main as nfr
    flags = ['--max-discovery-molecules', '35', '--max-quantification-molecules', '25', '--sample-seed', '8', '--cores', '1']
    evidence = tmp_path/'evidence.json.gz'
    write_json(evidence, planted_payload(100, occupancy=.7))
    consensus(['--evidence', str(evidence), '--output', str(tmp_path/'consensus'), '--no-bam', *flags])
    manifest = read_json(tmp_path/'consensus'/'manifest.json')
    assert [manifest['parameters']['recaller'][n] for n in C.NAMES] == [35, 25, 8]
    assert manifest['molecule_sampling'][0]['strata'][0]['discovery'] == 35
    units, _ = planted([(1., [(950, 1250)])], n=100)
    write_json(evidence, payload(units))
    nfr(['--evidence', str(evidence), '--output', str(tmp_path/'nfr'), '--nfr', '920-1290', '--bootstrap', '0', *flags])
    manifest = read_json(tmp_path/'nfr'/'manifest.json')
    assert [manifest['parameters'][n] for n in C.NAMES] == [35, 25, 8]
    assert manifest['detection']['molecule_sampling'][-1]['strata'][0]['quantification'] == 25


def test_nfr_depth_cap_counts_only_quantified_callable_molecules():
    units, _ = planted([(1., [(950, 1250)])], n=80)
    result = run_accessibility(payload(units), dict(mode='depth', nfr_regions=[(920, 1290)],
                                                   max_discovery_molecules=20, max_quantification_molecules=10))
    assert result['nfrs'][0]['callable'] == 20
    assert all(row['n'] == 20 for row in result['nfrs'][0]['depth_states'])
    assert all(s['discovery'] == 0 and s['quantification'] == 10
               for s in result['detection']['molecule_sampling'][-1]['strata'])


def test_caps_default_off_and_loader_does_not_scan(monkeypatch):
    from fiberhmm.inference.consensus import bam, loader_sampling
    assert RecallerOptions().max_discovery_molecules == NFROptions().max_discovery_molecules == 0
    assert RecallerOptions().max_quantification_molecules == NFROptions().max_quantification_molecules == 0
    sentinel = object()
    monkeypatch.setattr(bam, '_load_payload_once', lambda *args: sentinel)
    def forbidden(*args, **kwargs):
        raise AssertionError('Uncapped loading entered the hash sampler')
    monkeypatch.setattr(loader_sampling, 'load_sampled', forbidden)
    assert bam._load_payload(None, {}, parse_options({}), None) is sentinel
