"""Cheap loader frontiers preserve the post-collapse hash order and report shortages."""
from types import SimpleNamespace

import pytest

from fiberhmm.inference import depth_cap as C
from fiberhmm.inference.consensus import bam, loader_sampling as S
from fiberhmm.inference.consensus.parameters import parse_options


def options(d=10, q=10):
    return parse_options(dict(cr=dict(engine='lattice_recaller'),
                              recaller=dict(max_discovery_molecules=d, max_quantification_molecules=q)))


def exercise(monkeypatch, margin, *, chemistry='hia5-pacbio', n=100, invalid=22):
    rows = [(f'read{i}', 0, 1000) for i in range(n)]
    ordered = sorted((r[0] for r in rows), key=lambda name: C.rank(name, 1))
    bad = set(ordered[:invalid])
    units = [dict(read_name=name, unit_id=f'u{name}', strand='CT' if i % 2 else 'GA',
                  reference_start=lo, reference_end=hi, positions=[] if name in bad else [100, 200, 300],
                  raw_nuc_intervals=[[0, 95], [350, 500]]) for i, (name, lo, hi) in enumerate(rows)]
    monkeypatch.setattr(S, 'scan', lambda *args: (n, rows))
    calls = []
    def load(state, request, opt, progress):
        names = request['_read_names']['D']; calls.append(set(names))
        # Model filtering and collapse dropping unusable records.
        return dict(strata=[dict(dataset_id='D', chemistry=chemistry, load={},
                                 units=[u for u in units if u['read_name'] in names and u['positions']])])
    state = SimpleNamespace(get_dataset=lambda _: SimpleNamespace(paths=['unused']))
    result = S.load_sampled(state, dict(dataset_ids=['D'], chrom='chr1', start=100, end=350), options(),
                            lambda *a: None, dict(cap=10, seed=1, mode='recaller', margin=margin), load)
    source = result['strata'][0]
    channel = lambda u: 'pooled' if chemistry.startswith('hia5') else u['strand']
    choose = lambda us: C.select(us, 10, 1, channel, C.identity)
    assert choose(source['units']) == choose([u for u in units if u['positions']])
    return source, calls


@pytest.mark.parametrize('chemistry', ['hia5-pacbio', 'dddb'])
def test_margin_topup_and_same_postcollapse_selection(monkeypatch, chemistry):
    selections = []
    for margin in [1, 1.5, 3]:
        source, calls = exercise(monkeypatch, margin, chemistry=chemistry)
        receipt = source['load']['loader_sampling']
        assert receipt['passes'] >= 2
        assert all(a <= b for a, b in zip(calls, calls[1:]))
        assert receipt['reads_seen'] == 100
        assert receipt['reads_decoded'] == len(calls[-1]) < 100
        selections.append(C.select(source['units'], 10, 1,
            lambda u: 'pooled' if chemistry.startswith('hia5') else u['strand'], C.identity))
    assert selections[0] == selections[1] == selections[2]


def test_short_strata_exhaustion_reported(monkeypatch):
    source, calls = exercise(monkeypatch, 1.5, n=25, invalid=22)
    r = source['load']['loader_sampling']
    assert r['reads_seen'] == r['reads_decoded'] == 25
    assert r['usable_after_collapse'] == 3
    assert r['windows'][0]['strata'] == [dict(stratum='D::pooled', usable_after_collapse=3,
                                              used=3, short=True, exhausted=True)]


def test_zero_and_uncapped_phase_bypass_scan_exactly(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(bam, '_load_payload_once', lambda *a: sentinel)
    monkeypatch.setattr(S, 'scan', lambda *a: pytest.fail('uncapped loader scanned'))
    for d, q in [(0, 0), (2000, 0), (0, 2000)]:
        assert bam._load_payload(None, {}, options(d, q), None) is sentinel


def test_name_ties_and_tile_frontiers():
    rows = [('a', 0, 500), ('a', 5, 500), ('b', 0, 100), ('c', 400, 500)]
    left = S.Frontier(rows, (0, 100), 1, 1, 1)
    right = S.Frontier(rows, (400, 500), 1, 1, 1)
    assert len(left.names) == len(right.names) == 2
    assert 'c' not in left.names and 'b' not in right.names
    assert left.selected() == C.select(left.names, 1, 1, lambda _: '', lambda n: n)


def test_scan_never_decodes_tags_or_sequence(monkeypatch):
    class Read:
        query_name = 'a'; reference_start = 10; reference_end = 80
        is_unmapped = is_secondary = is_supplementary = False
        mapping_quality = 60
        def __getattr__(self, name):
            raise AssertionError(f'expensive read field {name}')
    class File:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def fetch(self, *args): return iter([Read()])
    import pysam
    monkeypatch.setattr(pysam, 'AlignmentFile', lambda *args: File())
    monkeypatch.setattr(bam, '_resolve_bam_fetch_region', lambda *args: ('chr1', 0, 100, 100))
    assert S.scan(['fake'], dict(chrom='chr1', start=0, end=100), 10) == (1, [('a', 10, 80)])


def test_real_bam_gate_precedes_tag_decode_and_native_replay(tmp_path, monkeypatch):
    import array
    import pysam
    from fiberhmm.inference import strand_rescue
    from fiberhmm.inference.consensus import adapter
    path = tmp_path/'deep.bam'
    header = pysam.AlignmentHeader.from_dict(dict(HD=dict(SO='coordinate'), SQ=[dict(SN='chr1', LN=2000)]))
    with pysam.AlignmentFile(path, 'wb', header=header) as out:
        for i in range(80):
            r = pysam.AlignedSegment(header)
            r.query_name = f'read{i}'; r.query_sequence = 'ACGT'*250
            r.reference_id = 0; r.reference_start = 0; r.cigarstring = '1000M'; r.mapping_quality = 60
            r.set_tag('MM', 'A+a,0,1,2;'); r.set_tag('ML', array.array('B', [255]*3))
            r.set_tag('MA', '1000;msp.:1-1000')
            out.write(r)
    pysam.index(str(path))
    decoded = []; replayed = []; fetches = []
    alignment_file = pysam.AlignmentFile
    class File:
        def __init__(self, *args, **kwargs): self.handle = alignment_file(*args, **kwargs)
        def __getattr__(self, name): return getattr(self.handle, name)
        def __enter__(self): return self
        def __exit__(self, *args): self.handle.close()
        def fetch(self, *args, **kwargs):
            fetches.append(1)
            return self.handle.fetch(*args, **kwargs)
    monkeypatch.setattr(pysam, 'AlignmentFile', File)
    hard = strand_rescue.hard_observations
    def decode(read, *args):
        decoded.append(read.query_name)
        return hard(read, *args)
    def replay(read, *args, **kwargs):
        replayed.append(read.query_name)
        return []
    monkeypatch.setattr(strand_rescue, 'hard_observations', decode)
    monkeypatch.setattr(adapter, 'replay_alignment', replay)
    opt = options(); opt['families'].recall_hia5_nucleosomes = False
    ds = [dict(dataset_id='D', paths=[str(path)], chemistry='hia5-pacbio')]
    region = dict(chrom='chr1', start=100, end=350)
    sampled = bam.load_bam_payload(ds, region, opt)
    assert len(decoded) == len(replayed) == 15
    assert len(fetches) == 2  # cheap scan + tag load; native replay reuses selected records
    assert set(decoded) == set(replayed)
    source = sampled['strata'][0]
    assert source['load']['loader_sampling']['reads_decoded'] == 15
    blocked = set(decoded)
    sampled.refine([((100, 350), lambda source, unit: unit['read_name'] not in blocked)])
    assert len(decoded) == len(set(decoded)) == len(replayed) == len(set(replayed))
    assert len(decoded) > 15
    assert len(fetches) == 3  # top-up decodes only the next tranche, without a replay scan
    decoded.clear(); replayed.clear(); fetches.clear()
    opt['recaller'].max_discovery_molecules = opt['recaller'].max_quantification_molecules = 0
    full = bam.load_bam_payload(ds, region, opt)
    assert len(decoded) == len(replayed) == 80
    assert len(fetches) == 2  # the original uncapped path
    choose = lambda p: C.select(p['strata'][0]['units'], 10, 1, lambda _: 'pooled', C.identity)
    assert choose(sampled) == choose(full)
    assert 'loader_sampling' not in full['strata'][0]['load']


def test_scoring_window_topup_is_margin_independent(monkeypatch):
    from fiberhmm.inference.consensus.lattice_recaller.units import eligible_unit
    names = sorted([f'r{i}' for i in range(100)], key=lambda n: C.rank(n, 1))
    rows = [(n, 0, 1000) for n in names]
    us = [dict(read_name=n, unit_id=n, strand='FWD', reference_start=0, reference_end=1000,
               positions=[150, 200, 250] if i < 22 else [95, 150, 200, 300, 360]) for i, n in enumerate(names)]
    monkeypatch.setattr(S, 'scan', lambda *a: (100, rows))
    state = SimpleNamespace(get_dataset=lambda _: SimpleNamespace(paths=['unused']))
    def load(state, req, opt, progress):
        return dict(strata=[dict(dataset_id='D', chemistry='hia5-pacbio', load={},
                                 units=[u for u in us if u['read_name'] in req['_read_names']['D']])])
    accepts = lambda source, unit: eligible_unit(unit, 100, 350, (120, 280))
    chosen = lambda units: C.select([u for u in units if accepts(None, u)], 10, 1, lambda _: '', C.identity)
    for margin in [1, 1.5, 3]:
        p = S.load_sampled(state, dict(dataset_ids=['D'], chrom='chr1', start=100, end=350), options(),
                           lambda *a: None, dict(cap=10, seed=1, margin=margin), load)
        p['input_files'] = ['preserved']
        decoded = p['strata'][0]['load']['loader_sampling']['reads_decoded']
        impossible = lambda source, unit: False
        impossible.loader_channels = set()
        p.refine([((100, 350), impossible)])
        assert p['strata'][0]['load']['loader_sampling']['reads_decoded'] == decoded
        assert p['strata'][0]['load']['loader_sampling']['windows'][-1]['strata'][0]['short']
        p.refine([((100, 350), accepts)])
        assert chosen(p['strata'][0]['units']) == chosen(us)
        assert p['input_files'] == ['preserved']
        assert p['strata'][0]['load']['loader_sampling']['reads_decoded'] < 100


def test_nfr_detected_windows_top_up_callable_units(monkeypatch):
    from fiberhmm.inference.accessibility.workflow import NFROptions
    from fiberhmm.inference.accessibility import workflow as N
    names = sorted([f'r{i}' for i in range(100)], key=lambda n: C.rank(n, 1))
    rows = [(n, 0, 1000) for n in names]
    us = [dict(read_name=n, unit_id=n, strand='FWD', reference_start=0, reference_end=1000,
               positions=[100, 200, 300], raw_nuc_intervals=[[150 if i < 22 else 0, 250], [300, 500]])
          for i, n in enumerate(names)]
    monkeypatch.setattr(S, 'scan', lambda *a: (100, rows))
    monkeypatch.setattr(N, '_nfr_regions', lambda *a: ([dict(start=120, end=280)], {}))
    state = SimpleNamespace(get_dataset=lambda _: SimpleNamespace(paths=['unused']))
    def load(state, req, opt, progress):
        return dict(strata=[dict(dataset_id='D', chemistry='hia5-pacbio', load={},
                                 units=[u for u in us if u['read_name'] in req['_read_names']['D']])])
    p = S.load_sampled(state, dict(dataset_ids=['D'], chrom='chr1', start=100, end=350), options(),
        lambda *a: None, dict(cap=10, seed=1, mode='nfr', nfr_options=NFROptions(max_discovery_molecules=10,
                            max_quantification_molecules=10)), load)
    rec = p['strata'][0]['load']['loader_sampling']
    assert rec['passes'] > 1 and rec['reads_decoded'] < 100
    nfr, = [r for r in rec['windows'] if r['purpose'] == 'nfr']
    assert nfr['window'] == [120, 280] and nfr['strata'][0]['used'] == 10


@pytest.mark.parametrize('strand', ['+', '-'])
def test_frozen_frame_and_training_exclusions_preserve_refinement(strand):
    from test_consensus_lattice_recaller import planted_payload
    from fiberhmm.inference.consensus.regions import pool_payloads
    from fiberhmm.inference.consensus.lattice_recaller.frozen import shift_payload, exclude_training
    from fiberhmm.inference.consensus.transfer import molecule_keys
    raw = planted_payload(4)
    seen = []
    def refine(requirements):
        for window, predicate in requirements:
            seen.append((window, [predicate(raw['strata'][0], u) for u in raw['strata'][0]['units']]))
        return raw
    live = S.SampledPayload(raw, refine)
    window = dict(raw['region'], name='target', strand=strand)
    pooled = pool_payloads([live], [window])
    width = window['end'] - window['start']
    target = shift_payload(pooled, 500, dict(chrom='oriented_BED', start=500, end=500 + width))
    original = target['strata'][0]['units']
    training = molecule_keys(original[0])
    filtered, ledger = exclude_training(target, dict(training_molecule_hashes=list(training)))
    expected = {u['read_name']: u['positions'] for u in original}
    predicate = lambda source, unit: unit['positions'] == expected[unit['read_name']]
    filtered.refine([((550, 650), predicate)])
    mapped = ((window['start'] + 50, window['start'] + 150) if strand == '+' else
              (window['end'] - 150, window['end'] - 50))
    assert seen[0][0] == mapped
    assert sum(seen[0][1]) == 3
    assert len(ledger) == 1 and len(filtered['strata'][0]['units']) == 3
    assert all(u['positions'] == expected[u['read_name']] for u in filtered['strata'][0]['units'])


def test_near_budget_avoids_decode_caches(monkeypatch):
    rows = [(f'r{i}', 0, 1000) for i in range(12)]
    monkeypatch.setattr(S, 'scan', lambda *a: (12, rows))
    state = SimpleNamespace(get_dataset=lambda _: SimpleNamespace(paths=['unused']))
    def load(state, req, opt, progress):
        assert not req['_decode_cache'] and not req['_replay_cache']
        return dict(strata=[dict(dataset_id='D', chemistry='hia5-pacbio', load={}, units=[
            dict(read_name=n, unit_id=n, strand='FWD', reference_start=0, reference_end=1000,
                 positions=[100, 200, 300]) for n, _, _ in rows])])
    p = S.load_sampled(state, dict(dataset_ids=['D'], chrom='chr1', start=100, end=350), options(),
                      lambda *a: None, dict(cap=10, seed=1), load)
    r = p['strata'][0]['load']['loader_sampling']
    assert r['strategy'] == 'all_candidates_near_budget' and r['reads_decoded'] == 12 and r['passes'] == 1


def test_daf_duplicate_collapse_topup_and_short_channel(tmp_path, monkeypatch):
    import pysam
    from fiberhmm.inference import strand_rescue
    from fiberhmm.inference.consensus import adapter
    path = tmp_path/'duplicates.bam'
    header = pysam.AlignmentHeader.from_dict(dict(HD=dict(SO='coordinate'), SQ=[dict(SN='chr1', LN=2000)]))
    with pysam.AlignmentFile(path, 'wb', header=header) as out:
        for i in range(40):
            read = pysam.AlignedSegment(header)
            read.query_name = f'duplicate{i}'; read.query_sequence = 'AYGT'*250
            read.reference_id = 0; read.reference_start = 0; read.cigarstring = '1000M'; read.mapping_quality = 60
            read.set_tag('st', 'CT'); read.set_tag('MA', '1000;msp.:1-1000'); out.write(read)
    pysam.index(str(path))
    decoded = []
    hard = strand_rescue.hard_observations
    def decode(read, *args):
        decoded.append(read.query_name)
        return hard(read, *args)
    monkeypatch.setattr(strand_rescue, 'hard_observations', decode)
    monkeypatch.setattr(adapter, 'replay_alignment', lambda *a, **kw: [])
    p = bam.load_bam_payload([dict(dataset_id='D', paths=[str(path)], chemistry='dddb')],
                            dict(chrom='chr1', start=100, end=350), options(5, 5))
    source = p['strata'][0]; receipt = source['load']['loader_sampling']
    assert len(decoded) == len(set(decoded)) == 40
    assert receipt['passes'] > 1
    assert source['evidence_units']['duplicate_reads_collapsed'] == 39
    assert receipt['usable_after_collapse'] == 1
    assert all(s['short'] and s['exhausted'] for r in receipt['windows'] for s in r['strata'])
    assert receipt['strata'][0]['reads_seen'] is None
    assert sum(s['reads_decoded'] for s in receipt['strata']) == 40


def test_sampled_recaller_matches_full_load_postcap_across_margins(tmp_path, monkeypatch):
    from copy import deepcopy
    from test_consensus_lattice_recaller import planted_payload
    from fiberhmm.inference.consensus.workflow import run_workflow
    full = planted_payload(240, occupancy=.7)
    source = full['strata'][0]; ds = source['dataset_id']
    rows = [(u['read_name'], u['reference_start'], u['reference_end']) for u in source['units']]
    monkeypatch.setattr(S, 'scan', lambda *a: (len(rows), rows))
    state = SimpleNamespace(get_dataset=lambda _: SimpleNamespace(paths=['unused'], family_scan_chemistry='ddda'))
    params = dict(compute=dict(cores=1), recaller=dict(max_discovery_molecules=40,
                  max_quantification_molecules=25, learned_spots=False))
    opt = parse_options(params)
    baseline = run_workflow(full, params, tmp_path/'full')
    def load(state, req, opt, progress):
        p = deepcopy(full)
        p['strata'][0]['units'] = [u for u in p['strata'][0]['units'] if u['read_name'] in req['_read_names'][ds]]
        p['strata'][0]['load'] = {}
        return p
    for margin in [1, 1.5, 3]:
        p = S.load_sampled(state, dict(full['region'], dataset_ids=[ds]), opt, lambda *a: None,
                           dict(S.policy(opt), margin=margin), load)
        result = run_workflow(p, params, tmp_path/f'margin{margin}')
        assert result['recaller']['classes'] == baseline['recaller']['classes']
        assert result['recaller']['rows'] == baseline['recaller']['rows']
        import gzip
        assert gzip.decompress((tmp_path/f'margin{margin}'/'molecules.tsv.gz').read_bytes()) == gzip.decompress(
            (tmp_path/'full'/'molecules.tsv.gz').read_bytes())


def test_unnamed_record_falls_back_to_full_loading(monkeypatch):
    monkeypatch.setattr(S, 'scan', lambda *a: (1, [(None, 0, 1000)]))
    state = SimpleNamespace(get_dataset=lambda _: SimpleNamespace(paths=['unused']))
    sentinel = object()
    def load(state, request, opt, progress):
        assert '_read_names' not in request
        return sentinel
    assert S.load_sampled(state, dict(dataset_ids=['D'], chrom='chr1', start=100, end=350), options(),
                          lambda *a: None, dict(cap=10, seed=1), load) is sentinel
