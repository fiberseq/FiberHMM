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
    assert all(r['prevalence'] <= r['prevalence_edge'] <= r['prevalence_loose'] <= 1 for r in rows)
    catalog = res['datasets']['planted']['cr']['catalog']
    assert any(f['family'] == hit[0]['id'] and set(f['classification_counts']) == {'CT', 'GA'} for f in catalog)
    assigned = [p for r in res['datasets']['planted']['cr']['records'] for p in r['proposals'] if p['family'] == hit[0]['id']]
    assert len(assigned) > 30
    with gzip.open(tmp_path/'molecules.tsv.gz', 'rt') as fh:
        header = fh.readline().rstrip('\n').split('\t'); first = dict(zip(header, fh.readline().rstrip('\n').split('\t')))
    # edge_range used to be dropped by the TSV writer (audit 2026-09-29); it is 'l0-l1,r0-r1' (empty without a call).
    assert header == ['class_id', 'channel', 'unit_id', 'posterior', 'log_bf', 'label', 'tier', 'start', 'end', 'edge_range']
    assert first['edge_range'] == '' or len(first['edge_range'].split(',')) == 2
    with open(tmp_path/'classes.tsv') as fh:
        cheader = fh.readline().rstrip('\n').split('\t')
    assert 'status' in cheader and 'unscored_reason' in cheader
    assert res['manifest']['mode'] == 'CR' and res['manifest']['recaller']['unscored_classes'] == []
    assert next(c for c in classes if c['id'] == hit[0]['id'])['status'] == 'supported'
    # Labelled native calls carry the class posterior as q0; DAF classes carry a recaller strand-trust verdict.
    labelled = [p for r in res['datasets']['planted']['cr']['records'] for p in r['proposals'] if p['family']]
    assert labelled and all(1 <= p['q0'] <= 255 and p['member_q0'][p['family']] == p['q0'] for p in labelled)
    verdict = next(f for f in catalog if f['family'] == hit[0]['id'])['strand_resolution']
    assert verdict['trusted_strand'] == 'both' and verdict['supported_strands'] == ['CT', 'GA']
    assert (tmp_path/'broader.tsv.gz').exists()
    # The recaller's own calls: every member molecule of the class, with its own edges near the planted footprint.
    rc = [c for r in res['datasets']['planted']['cr']['records'] for c in r.get('recaller_calls', []) if c['family'] == hit[0]['id']]
    core = [c for c in rc if c['tier'] == 'core']
    assert len(core) > 30 and all(c['kind'] == 'class' for c in rc) and {c['tier'] for c in rc} <= {'core', 'edge', 'loose'}
    # Members with a native call for the class take its edges; the rest keep their lattice edges.
    assert {c['edge_source'] for c in core} <= {'native', 'lattice'} and any(c['edge_source'] == 'native' for c in core)
    rc = core
    assert all(abs(c['interval'][0] - 1180) <= 8 and abs(c['interval'][1] - 1210) <= 8 for c in rc)
    assert all(c['consensus_interval'] == [round(hit[0]['start']), round(hit[0]['end'])] for c in rc)
    assert all(c['edge_range'][0][0] <= c['lattice_interval'][0] <= c['edge_range'][0][1] and c['edge_range'][1][0] <= c['lattice_interval'][1] <= c['edge_range'][1][1] for c in rc)


def test_core_rule_can_drop_every_class(tmp_path):
    res = run_workflow(planted_payload(), {'cr': {'engine': 'lattice_recaller'}, 'recaller': {'learned_spots': False, 'minimum_core_bp': 1000}}, tmp_path)
    assert res['recaller']['classes'] == [] and res['manifest']['recaller']['dropped_by_core_rule']


def test_payload_is_not_mutated(tmp_path):
    payload = planted_payload(n=80); before = copy.deepcopy(payload)
    run_workflow(payload, {'cr': {'engine': 'lattice_recaller'}, 'recaller': {'learned_spots': False}}, tmp_path)
    assert payload == before


# ---------------------------------------------------------------- frozen prototype reference (external data)
# FIBERHMM_CR_REFERENCE_DIR: directory holding napa_N1.classes.tsv (frozen prototype output).
# FIBERHMM_NAPA_EVIDENCE: the matching NAPA keep-one evidence.json.gz. Skipped unless both are set.
import os  # noqa: E402  (kept local to this optional block)

REFERENCE = Path(os.environ.get('FIBERHMM_CR_REFERENCE_DIR') or '/nonexistent/cr_recaller_reference')
NAPA = Path(os.environ.get('FIBERHMM_NAPA_EVIDENCE') or '/nonexistent/napa_evidence.json.gz')


@pytest.mark.skipif(not (REFERENCE/'napa_N1.classes.tsv').exists() or not NAPA.exists(), reason='set FIBERHMM_CR_REFERENCE_DIR and FIBERHMM_NAPA_EVIDENCE')
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


# ---------------------------------------------------------------- release audit 2026-09-29
def test_looser_tiers_are_a_coherent_union_and_match_molecule_labels():
    """90 class-shaped + 10 wider molecules: the loose tier used to add a full 1/n for every sub-0.5 molecule whose
    posterior was already in core, giving 1.0296 (codex probe_tier_fit.py). Each tier must equal
    mean_i [P_i + (1 - P_i) * 1{molecule i is labelled into the tier}] and never exceed 1."""
    units = []
    for i in range(100):
        w = 30 if i < 90 else 60
        u = lattice(SITES, ~((SITES >= 1180 - (w - 30)//2) & (SITES < 1210 + (w - 30)//2))); u['uid'] = f'u{i}'; units.append(u)
    opt = RecallerOptions(learned_spots=False)
    r = Mo.fit_channel(units, [CLASS], .5, opt)
    tiers = r['tiers'][0]; P = r['P'][:, 0]
    assert tiers['core'] <= tiers['edge'] <= tiers['loose'] <= 1 + 1e-9
    labels = [Mo.label(p, r['w'][0], opt.bf_threshold)[0] for p in P]
    per = [None if lab == 'member' else (call['tiers'][0] or (None,))[0] for lab, call in zip(labels, r['calls'])]
    assert any(t == 'loose' for t in per)
    expect_edge = np.mean([p + (1 - p)*(t == 'edge') for p, t in zip(P, per)])
    expect_loose = np.mean([p + (1 - p)*(t in ('edge', 'loose')) for p, t in zip(P, per)])
    assert math.isclose(tiers['edge'], expect_edge, abs_tol=1e-6) and math.isclose(tiers['loose'], expect_loose, abs_tol=1e-6)


def test_edge_contraction_folds_choose_boxes_from_their_training_molecules(monkeypatch):
    """Each validation fold must score boxes proposed from its own training half, not from all molecules
    (codex audit: the held-out half chose the boxes it was scored on)."""
    import dataclasses
    g = dict(L=[1170, 1190], R=[1206, 1214], span=(1180., 1210.))
    opt = dataclasses.replace(RecallerOptions(), edge_contraction=True, learned_spots=False)
    units = planted_units(400, (1188, 1210), always_marked=(1184,))
    original_propose, original_rows = Mo.propose_edges, Mo._aligned_rows
    proposed, scored = [], []

    def propose(keep, P, gs, o):
        out = original_propose(keep, P, gs, o)
        tagged = [(dict(x, proposed_from=len(keep)) if x is not None else None, moves) for x, moves in out]
        proposed.append(len(keep)); return tagged

    def rows(us, sc_a, gs_b, f, o):
        scored.append((len(us), gs_b[0].get('proposed_from'))); return original_rows(us, sc_a, gs_b, f, o)

    monkeypatch.setattr(Mo, 'propose_edges', propose); monkeypatch.setattr(Mo, '_aligned_rows', rows)
    res = Mo.fit_channel(units, [g], .5, opt)
    assert proposed[0] == len(units) and len(proposed) == 3
    train = [n for n in proposed[1:]]
    # Per fold: the training rows and the held-out rows both use the geometry proposed from that fold's training set.
    assert [s[1] for s in scored] == [train[0], train[0], train[1], train[1]]
    assert all(s[1] != len(units) for s in scored)
    # The accepted geometry is the all-molecule proposal.
    assert res['gs'][0].get('proposed_from') == len(units) and res['gs'][0]['L'] == [1185, 1188]


def _units_for(n, fps, strands=('CT', 'GA'), span=(900, 1500), step=4, seed=7, ref=None):
    """Planted DAF molecules (from the audit's edgecases.py): each footprint (a, b, occupancy) bound at random."""
    rng = np.random.default_rng(seed); out = []
    for i in range(n):
        strand = strands[i % len(strands)]; off = 0 if strand == 'CT' else 2
        pos = np.arange(span[0] + off, span[1], step); hit = rng.random(len(pos)) < PA; calls = []
        for a0, b0, occ in fps:
            if rng.random() < occ:
                jl, jr = rng.integers(-1, 2, size=2); a, b = a0 + jl, b0 + jr; prot = (pos >= a) & (pos < b)
                hit = np.where(prot, rng.random(len(pos)) < PP, hit); calls.append(dict(interval=[int(a), int(b)], llr=12.))
        rs, re_ = ref or (span[0] - 20, span[1] + 20)
        out.append(dict(unit_id=f'mol{i:04d}', read_name=f'read{i:04d}', strand=strand, reference_start=rs, reference_end=re_,
                        positions=pos.tolist(), hits=hit.astype(int).tolist(), p_accessible=[PA]*len(pos), p_protected=[PP]*len(pos),
                        native_multi_interval_calls=calls, native_multi_interval_tf_intervals=[c['interval'] for c in calls],
                        raw_tf_intervals=[c['interval'] for c in calls], representative_raw_tf_intervals=[c['interval'] for c in calls],
                        raw_nuc_intervals=[], msp_intervals=[[rs, re_]], contexts=[0]*len(pos)))
    return out


def _payload(region, units):
    return dict(region=dict(chrom='chrT', start=region[0], end=region[1]),
                strata=[dict(dataset_id='d', stratum_id='d', chemistry='ddda', model_manifest=dict(preset='ddda'), units=units)])


def test_class_no_molecule_spans_is_unscored_not_unsupported(tmp_path):
    """A class at a data end (molecules stop at 1352) is discovered but cannot be scored: it must be reported as
    unscored with the reason, not hidden as unsupported (audit: edgecases.py dataend)."""
    import csv
    p = _payload((1050, 1350), _units_for(160, [(1300, 1330, .5)], span=(900, 1352), ref=(880, 1352)))
    res = run_workflow(p, {'cr': {'engine': 'lattice_recaller'}, 'recaller': {'learned_spots': False}}, tmp_path)
    rec = res['manifest']['recaller']; cid = res['recaller']['classes'][0]['id']
    assert rec['unscored_classes'] == [cid] and rec['unsupported_classes'] == []
    assert res['recaller']['classes'][0]['status'] == 'unscored' and res['recaller']['rows'] == []
    assert {u['channel'] for u in rec['unscored'] if u['class_id'] == cid} == {'d::CT', 'd::GA'}
    assert all(u['unscored_reason'].startswith('no_spanning_molecules') for u in rec['unscored'])
    with open(tmp_path/'classes.tsv') as fh:
        rows = list(csv.DictReader(fh, delimiter='\t'))
    assert {r['status'] for r in rows} == {'unscored'} and all(r['prevalence'] == '' for r in rows)


def test_wide_native_calls_of_members_do_not_inherit_a_small_class_label(tmp_path):
    """Class labels used to go to any overlapping native call of a member molecule (a 10.5-bp class labelled calls up
    to 189 bp). A call wider than recaller.call_max_bp, or whose censored edges miss the class edge boxes, keeps no
    label; the class's own calls still do."""
    p = planted_payload()
    for u in p['strata'][0]['units']:
        if u['native_multi_interval_calls']:
            u['native_multi_interval_calls'].append(dict(interval=[1100, 1215], llr=12.))
    res = run_workflow(p, {'cr': {'engine': 'lattice_recaller'}, 'recaller': {'learned_spots': False}}, tmp_path)
    props = [q for r in res['datasets']['planted']['cr']['records'] for q in r['proposals']]
    wide = [q for q in props if q['source_interval'] == [1100, 1215]]; own = [q for q in props if q['source_interval'][1] - q['source_interval'][0] < 40]
    assert len(wide) > 30 and not any(q['family'] for q in wide)
    assert sum(bool(q['family']) for q in own) > 30


def test_abutting_option_is_removed_but_old_manifests_still_parse():
    """recaller.abutting was removed in 3.0 (unnormalized configuration weights: +88 nats on null evidence).
    Manifests, frozen catalogs and sessions written before the removal carry abutting=false and must still load;
    a request for abutting=true is refused with the alternatives named."""
    from fiberhmm.inference.consensus.parameters import parse_options, parameter_schema
    assert not any(c['name'] == 'abutting' for c in parameter_schema()['recaller'])
    options = parse_options({'cr': {'engine': 'lattice_recaller'}, 'recaller': {'abutting': False, 'stringency': 0.9}})
    assert not hasattr(options['recaller'], 'abutting') and options['recaller'].stringency == 0.9
    with pytest.raises(ValueError, match=r'removed in FiberHMM 3\.0.*\+ edge.*linker=either'):
        parse_options({'cr': {'engine': 'lattice_recaller'}, 'recaller': {'abutting': True}})


def test_recaller_records_mode_cr_whatever_sr_and_cross_say(tmp_path):
    res = run_workflow(planted_payload(n=80), {'cr': {'engine': 'lattice_recaller'}, 'sr': {'enabled': True}, 'cross': {'enabled': True},
                                              'recaller': {'learned_spots': False}}, tmp_path)
    m = res['manifest']
    assert m['mode'] == m['display_mode'] == m['mode_realized'] == 'CR'
    assert not any('cross-dataset' in w or 'strand-shared' in w for w in m['data_warnings'])


def test_failed_task_stops_the_worker_pool_at_once():
    """A failing task used to surface only after every queued task had run (poolfail.py). time.sleep(-1) raises at
    once; the seven 2-s sleeps behind it must be cancelled and the workers killed."""
    import time
    from fiberhmm.inference.consensus.lattice_recaller.workflow import _run_all
    from fiberhmm.inference.consensus.execution import shared_worker_pool
    for shared in (False, True):
        started = time.monotonic()
        with pytest.raises(ValueError):
            if shared:
                with shared_worker_pool(2):
                    _run_all(time.sleep, [(-1,)] + [(2,)]*7, 2)
            else:
                _run_all(time.sleep, [(-1,)] + [(2,)]*7, 2)
        assert time.monotonic() - started < 5.5


def test_recaller_parameter_contract_rejects_ignored_and_unsafe_settings():
    base = {'cr': {'engine': 'lattice_recaller'}}
    assert parse_options({})['cr'].engine == 'lattice_recaller'
    for group, values, match in [('families', {'physical_radius_bp': 5}, 'families.physical_radius_bp is not used by the lattice recaller'),
                                 ('cr', {'seed': 5}, 'cr.seed is not used'),
                                 ('families', {'stop_after': 'native'}, 'stop_after'),
                                 ('input', {'correct_native': False}, 'correct_native=false is not supported'),
                                 ('compute', {'predictive_backend': 'auto'}, 'not used by the lattice recaller'),
                                 ('recaller', {'call_max_bp': 150}, 'must not exceed the tile overlap')]:
        with pytest.raises(ValueError, match=match):
            parse_options({**base, group: dict(base.get(group, {}), **values)})
    # Controls the recaller reads, and flags it records but ignores, stay accepted.
    parse_options({**base, 'families': {'recall_hia5_nucleosomes': False, 'nuc_split_minimum_llr': 3.}, 'input': {'minimum_mapq': 5},
                   'sr': {'enabled': False}, 'cross': {'enabled': True}, 'compute': {'cores': 2, 'fit_cache_dir': '/tmp/x'},
                   'recaller': {'call_max_bp': 150, 'tile_bp': 400}})
    # Other engines reject recaller settings instead of ignoring them.
    with pytest.raises(ValueError, match='recaller.stringency is not used by staged families'):
        parse_options({'cr': {'engine': 'staged_native_families'}, 'recaller': {'stringency': .5}})
    with pytest.raises(ValueError, match='recaller.stringency is not used by cr.engine=call_harmonization'):
        parse_options({'cr': {'engine': 'call_harmonization'}, 'recaller': {'stringency': .5}})


def test_cli_rejects_staged_stage_controls_before_loading_and_streams_windows(tmp_path, monkeypatch):
    from fiberhmm.inference.consensus import cli, bam
    events = []
    monkeypatch.setattr(bam, 'load_bam_payload', lambda d, w, o, p: events.append(('load', w['start'])) or dict(region=w, strata=[]))
    # x.bam is never written: the CLI's up-front chemistry check reads BAM headers, so stub it with the loader.
    monkeypatch.setattr(bam, 'check_dataset_chemistries', lambda datasets: None)
    monkeypatch.setattr(cli, 'run_analysis', lambda payload, v, folder, progress=None: events.append(('run', payload['region']['start'])) or
                        dict(manifest=dict(status='complete', seconds=0.)))
    common = ['--bam', str(tmp_path/'x.bam'), '--region', 'chrT:100-200', '--region', 'chrT:300-400']
    for extra in (['--stop-after', 'native'], ['--stop-after', 'parents'], ['--consolidation-bp', '5'], ['--cache', str(tmp_path/'c')]):
        with pytest.raises(SystemExit):
            cli.main(common + extra + ['--output', str(tmp_path/'bad')])
        assert events == []
    with pytest.raises(SystemExit):
        params = tmp_path/'p.json'; params.write_text(json.dumps({'recaller': {'call_max_bp': 200}}))
        cli.main(common + ['--parameters', str(params), '--output', str(tmp_path/'bad2')])
    assert events == []
    with redirect_stdout(io.StringIO()):
        # --window-jobs 1: since 3.0 independent windows run in parallel worker processes by default (where these
        # in-process stubs do not apply); one window at a time keeps the in-process streaming order under test.
        cli.main(common + ['--window-jobs', '1', '--output', str(tmp_path/'ok')])
    # Independent windows stream: each is loaded and analysed before the next is loaded.
    assert events == [('load', 100), ('run', 100), ('load', 300), ('run', 300)]
