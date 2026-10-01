"""recaller.order_replicates / fiberhmm-consensus --robust: the optional read-order robustness check.

Off (the default) the outputs are exactly those of a run without the option; on, the class set is still the default
ordering's, and each class gains order_robustness (the fraction of the N+1 orderings that find it supported) and robust
(supported in at least recaller.order_robust_fraction of them; default every ordering)."""
import copy
import gzip
import hashlib
import json

import pytest

from fiberhmm.inference.consensus.lattice_recaller import workflow as W
from fiberhmm.inference.consensus.parameters import RecallerOptions, parse_options
from fiberhmm.inference.consensus.workflow import run_workflow

from test_consensus_lattice_recaller import planted_payload

BASE = {'cr': {'engine': 'lattice_recaller'}, 'recaller': {'learned_spots': False}}


def _params(**recaller):
    p = copy.deepcopy(BASE); p['recaller'].update(recaller); return p


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gz_sha(path):
    return hashlib.sha256(gzip.open(path).read()).hexdigest()


def _strip_timing(value):
    """result/manifest without wall-clock fields (seconds) and the recorded parameters."""
    if isinstance(value, dict):
        return {k: _strip_timing(v) for k, v in value.items() if k not in ('seconds', 'parameters')}
    if isinstance(value, list):
        return [_strip_timing(v) for v in value]
    return value


def test_option_defaults_off_and_validates():
    assert RecallerOptions().order_replicates == 0
    assert parse_options({'recaller': {'order_replicates': 3}})['recaller'].order_replicates == 3
    for bad in (-1, 21, 1.5, True):
        with pytest.raises(ValueError):
            parse_options({'recaller': {'order_replicates': bad}})
    # The staged engine does not read it: a non-default value is refused, not ignored.
    with pytest.raises(ValueError, match='order_replicates'):
        parse_options({'cr': {'engine': 'staged_native_families'}, 'recaller': {'order_replicates': 2}})


def test_salt_zero_is_the_input_and_other_salts_rename_and_reorder_only():
    sources = [dict(dataset_id='d', chemistry='ddda', units=[dict(unit_id=f'u{i}', strand='CT' if i % 2 else 'GA', x=i) for i in range(20)])]
    assert W.reordered_sources(sources, 0) is sources
    one, again, two = (W.reordered_sources(sources, s) for s in (1, 1, 2))
    assert one == again                                                      # deterministic
    ids = [[u['unit_id'] for u in s[0]['units']] for s in (one, two)]
    assert ids[0] != ids[1] and len(set(ids[0])) == 20
    assert {u['x'] for u in one[0]['units']} == set(range(20))               # the same molecules
    strands = [u['strand'] for u in one[0]['units']]
    assert strands == sorted(strands, key=lambda s: s != 'CT')               # DAF: CT before GA, as the loader sorts
    assert [u['unit_id'] for u in sources[0]['units']] == [f'u{i}' for i in range(20)]   # input untouched


def test_option_off_writes_exactly_the_default_outputs(tmp_path):
    """No new columns, keys or manifest block unless the check is asked for; order_replicates=0 equals omitting it."""
    a = run_workflow(planted_payload(), BASE, tmp_path/'a')
    b = run_workflow(planted_payload(), _params(order_replicates=0), tmp_path/'b')
    for name in ('classes.tsv', 'families.tsv', 'calls.tsv'):
        assert _sha(tmp_path/'a'/name) == _sha(tmp_path/'b'/name), name
    for name in ('molecules.tsv.gz', 'broader.tsv.gz'):
        assert _gz_sha(tmp_path/'a'/name) == _gz_sha(tmp_path/'b'/name), name
    assert _strip_timing(a) == _strip_timing(b)
    header = (tmp_path/'a'/'classes.tsv').read_text().split('\n', 1)[0].split('\t')
    assert header == W.CLASS_FIELDS and 'order_robustness' not in header
    assert 'order_robustness' not in a['manifest']['recaller']
    assert all('robust' not in c for c in a['recaller']['classes'])
    assert all('robust' not in f for f in a['datasets']['planted']['cr']['catalog'])


def test_option_on_keeps_the_default_classes_and_marks_the_planted_class_robust(tmp_path):
    off = run_workflow(planted_payload(), BASE, tmp_path/'off')
    on = run_workflow(planted_payload(), _params(order_replicates=2), tmp_path/'on')
    # The classes, estimates and labels are the default ordering's (salt 0); only the robustness annotation is added.
    drop = lambda rows: [{k: v for k, v in r.items() if k not in W.ROBUST_FIELDS} for r in rows]
    assert drop(on['recaller']['classes']) == drop(off['recaller']['classes'])
    assert drop(on['recaller']['rows']) == drop(off['recaller']['rows'])
    assert _gz_sha(tmp_path/'on'/'molecules.tsv.gz') == _gz_sha(tmp_path/'off'/'molecules.tsv.gz')
    planted = [c for c in on['recaller']['classes'] if abs(c['start'] - 1180) <= 3 and abs(c['end'] - 1210) <= 3]
    assert len(planted) == 1 and planted[0]['order_robustness'] == 1.0 and planted[0]['robust'] is True
    block = on['manifest']['recaller']['order_robustness']
    assert block['orderings'] == 3 and block['replicates'] == 2 and block['salts'] == [0, 1, 2]
    assert 'order_robust_fraction' in block['rule'] and block['robust_fraction'] == 1.0
    assert len(block['supported_per_ordering']) == 3 and planted[0]['id'] in block['robust_classes']
    with open(tmp_path/'on'/'classes.tsv') as fh:
        header = fh.readline().rstrip('\n').split('\t'); rows = [dict(zip(header, line.rstrip('\n').split('\t'))) for line in fh]
    assert header == W.CLASS_FIELDS + ['order_robustness', 'robust']
    assert {r['robust'] for r in rows if r['class_id'] == planted[0]['id']} == {'True'}
    fam = next(f for f in on['datasets']['planted']['cr']['catalog'] if f['family'] == planted[0]['id'])
    assert fam['order_robustness'] == 1.0 and fam['robust'] is True
    assert on['manifest']['parameters']['recaller']['order_replicates'] == 2


def test_option_on_is_deterministic(tmp_path):
    one = run_workflow(planted_payload(n=120, seed=11), _params(order_replicates=2), tmp_path/'1')
    two = run_workflow(planted_payload(n=120, seed=11), _params(order_replicates=2), tmp_path/'2')
    assert _sha(tmp_path/'1'/'classes.tsv') == _sha(tmp_path/'2'/'classes.tsv')
    assert _strip_timing(one['manifest']['recaller']) == _strip_timing(two['manifest']['recaller'])


def test_robustness_counts_supported_matches_and_needs_a_majority(monkeypatch):
    """Unit test of the counting: ordering 0 is the run's own verdict; other orderings match by geometry
    (discovery.same_class) and count only when supported; robust needs order_robust_fraction (default all) of them."""
    def cls(cid, a, b):
        return dict(id=cid, span=(float(a), float(b)))
    classes = [cls('A', 100, 130), cls('B', 200, 220), cls('C', 300, 320), cls('D', 400, 420)]
    rows = [dict(class_id=c, supported=c != 'D') for c in 'ABCD']
    # salt 1: A (centre 2 bp off), B unsupported, C. salt 2: A, D (supported there only), B's centre 5 bp off.
    found = {1: ([cls('x', 102, 132), cls('y', 200, 220), cls('z', 300, 320)], {'x', 'z'}),
             2: ([cls('p', 100, 130), cls('q', 400, 420), cls('r', 190, 240)], {'p', 'q', 'r'})}
    seen = []

    def discover(src, region, opt, progress, cores):
        assert src[0]['units'][0]['unit_id'] != 'u1'                         # a reordered copy, never the input
        seen.append(len(seen) + 1); return found[seen[-1]][0], [], [(0, 1000)], []

    def quantify(src, classes_, tiles, opt, progress, cores, frozen=None):
        sup = found[seen[-1]][1]
        return [dict(class_id=g['id'], supported=g['id'] in sup) for g in classes_], [], [], []

    monkeypatch.setattr(W, 'discover', discover); monkeypatch.setattr(W, 'quantify', quantify)
    sources = [dict(dataset_id='d', chemistry='hia5-pacbio', units=[dict(unit_id='u1', strand='+')])]
    run = lambda **o: W.order_robustness(sources, dict(start=0, end=1000), RecallerOptions(order_replicates=2, **o),
                                         lambda *a, **k: None, 1, classes, rows)
    block = run()
    got = {g['id']: (g['order_robustness'], g['robust']) for g in classes}
    assert got == {'A': (1.0, True), 'B': (round(1/3, 4), False), 'C': (round(2/3, 4), False), 'D': (round(1/3, 4), False)}
    assert seen == [1, 2] and block['supported_per_ordering'] == [3, 2, 3] and block['robust_classes'] == ['A']
    assert block['robust_supported_classes'] == 1 and block['supported_classes'] == 3 and block['robust_fraction'] == 1.0
    seen.clear(); block = run(order_robust_fraction=0.5)                     # half of the orderings: C (2 of 3) too
    assert block['robust_classes'] == ['A', 'C'] and block['robust_supported_classes'] == 2


def test_transfer_does_not_rerun_discovery_for_a_robust_source_run():
    from fiberhmm.inference.consensus.lattice_recaller.frozen import transfer_options
    from fiberhmm.inference.consensus.parameters import options_dict
    params = json.loads(json.dumps(options_dict(parse_options({'recaller': {'order_replicates': 3}}))))
    assert transfer_options(dict(parameters=params), [], cores=1)['recaller'].order_replicates == 0


def test_cli_robust_flag_sets_order_replicates(tmp_path, monkeypatch):
    from fiberhmm.inference.consensus import cli
    seen = {}

    def fake_run(payload, values, folder, progress=None):
        seen.update(values); raise SystemExit(0)
    monkeypatch.setattr(cli, 'run_analysis', fake_run)
    evidence = tmp_path/'evidence.json.gz'
    with gzip.open(evidence, 'wt') as fh:
        json.dump(planted_payload(n=40), fh)
    with pytest.raises(SystemExit):
        cli.main(['--evidence', str(evidence), '--robust', '2', '--output', str(tmp_path/'out'), '--no-bam'])
    assert seen['recaller']['order_replicates'] == 2
    with pytest.raises(SystemExit) as error:
        cli.main(['--evidence', str(evidence), '--robust', '50', '--output', str(tmp_path/'out2'), '--no-bam'])
    assert error.value.code == 2
