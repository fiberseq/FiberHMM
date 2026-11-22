"""fiberhmm-consensus per-read tables, cross-fitted discovery, role rules, NFR-state conditioning and bounded memory.

Synthetic DddA windows only (tests/test_consensus_continue.make_windows_bam); no real loci or sample names.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import sys
import tracemalloc
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

from fiberhmm.inference.consensus import batch
from fiberhmm.inference.consensus import read_table as RT
from fiberhmm.inference.consensus.cli import main
from fiberhmm.inference.consensus.parameters import options_dict, parse_options
from fiberhmm.io.folds import FoldSpec, fold_of

sys.path.insert(0, str(Path(__file__).parent))
from test_consensus_continue import make_windows_bam  # noqa: E402

SPEC = 'hash=md5,mod=10,discover=0-4'


def _rows(path):
    with gzip.open(path, 'rt', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


@pytest.fixture(scope='module')
def deep(tmp_path_factory):
    """Three windows with 80 reads each (enough for discovery on half the reads)."""
    root = tmp_path_factory.mktemp('deep')
    bam = make_windows_bam(root/'in.bam', n_windows=3, reads=80, seed=5)
    bed = root/'w.bed'
    bed.write_text('chr1\t1100\t1700\tw1\t0\t+\nchr1\t3100\t3700\tw2\t0\t-\nchr1\t5100\t5700\tw3\t0\t+\n')

    def run(output, *extra, bed_path=None, region=None):
        import os
        from fiberhmm.core import bam_reader
        from fiberhmm.inference import engine
        source = ['--region', region] if region else ['--bed', str(bed_path or bed)]
        # The module-scoped runs happen outside conftest's per-test isolation: give every call the same clean
        # process-wide DAF masks (an earlier test module may have configured them).
        with pytest.MonkeyPatch.context() as mp, redirect_stdout(io.StringIO()):
            for name, value in (('_DAF_RUN_MASK_MIN', None), ('_DAF_RUN_POLICY', None), ('_DAF_RUN_MASK_CONFIGURED', False)):
                mp.setattr(bam_reader, name, value)
            mp.setattr(engine, '_DAF_UNALIGNED_MASK', None)
            for env in (bam_reader._DAF_RUN_MASK_ENV, bam_reader._DAF_RUN_POLICY_ENV, engine._DAF_UNALIGNED_MASK_ENV):
                mp.delenv(env, raising=False)
            try:
                main(['--bam', str(bam), *source, '--no-bam', '--cores', '1', '--output', str(output), *map(str, extra)])
            finally:
                for env in (bam_reader._DAF_RUN_MASK_ENV, bam_reader._DAF_RUN_POLICY_ENV, engine._DAF_UNALIGNED_MASK_ENV):
                    os.environ.pop(env, None)
        return Path(output)
    return run, root


@pytest.fixture(scope='module')
def crossfit_run(deep):
    run, root = deep
    return run(root/'crossfit', '--read-table', '--discover-fold-spec', SPEC, '--window-jobs', '1')


# ------------------------------------------------------------------------------------------------- folds
def test_folds_are_md5_of_the_read_name_and_order_free():
    names = [f'm{i}/ccs' for i in range(200)]
    for name in names[:20]:
        assert fold_of(name) == int(hashlib.md5(name.encode()).hexdigest(), 16) % 10
    spec = FoldSpec.parse(SPEC)
    assert spec.canonical() == SPEC and FoldSpec.parse('discover=4+0-3').canonical() == SPEC
    from fiberhmm.inference.consensus.lattice_recaller.workflow import discovery_sources
    opt = parse_options({'recaller': {'discover_fold_spec': 'discover=0-4'}})['recaller']
    units = [dict(unit_id=f'u{i}', read_name=n) for i, n in enumerate(names)]
    a = discovery_sources([dict(dataset_id='d', units=units)], opt)[0]['units']
    b = discovery_sources([dict(dataset_id='d', units=units[::-1])], opt)[0]['units']
    assert {u['read_name'] for u in a} == {u['read_name'] for u in b} == {n for n in names if fold_of(n) < 5}
    assert 60 < len(a) < 140


def test_new_parameters_stay_out_of_receipts_unless_used():
    plain = options_dict(parse_options({}))['recaller']
    assert not {'discover_fold_spec', 'fold_weights', 'read_table'} & set(plain)
    used = options_dict(parse_options({'recaller': {'discover_fold_spec': 'mod=10,discover=0+1+2+3+4', 'read_table': True}}))['recaller']
    assert used['discover_fold_spec'] == SPEC and used['fold_weights'] == 'all' and used['read_table'] is True
    with pytest.raises(ValueError, match='heldout'):
        parse_options({'recaller': {'fold_weights': 'heldout'}})
    with pytest.raises(ValueError, match='heldout'):
        parse_options({'recaller': {'discover_fold_spec': 'mod=2,discover=0-1', 'fold_weights': 'heldout'}})
    with pytest.raises(ValueError):
        parse_options({'recaller': {'discover_fold_spec': 'mod=10,discover=12'}})


# --------------------------------------------------------------------------------------- cross-fit + table
def test_crossfit_discovers_on_discovery_folds_and_scores_every_read(crossfit_run):
    out = crossfit_run; with_classes = 0
    for i in (1, 2, 3):
        folder = out/f'window_{i:06d}'
        manifest = json.loads((folder/'manifest.json').read_text())
        cf = manifest['cross_fit']
        assert cf['discover_fold_spec'] == SPEC and cf['discovery_units'] > 0 and cf['heldout_units'] > 0
        assert manifest['parameters']['recaller']['discover_fold_spec'] == SPEC
        rows = _rows(folder/RT.READ_TABLE)
        if not manifest['recaller']['classes']:
            assert rows == []                                       # no class: header only
            continue
        with_classes += 1
        evidence = json.loads(gzip.decompress((folder/'evidence.json.gz').read_bytes()))
        names = {u['read_name'] for s in evidence['strata'] for u in s['units']}
        assert {r['read_name'] for r in rows} == names            # every read, both folds
        for r in rows:
            assert int(r['fold']) == fold_of(r['read_name']) and (r['discovery'] == 'True') == (int(r['fold']) < 5)
        assert any(r['scored'] == 'True' and r['discovery'] == 'False' for r in rows)
    assert with_classes >= 2


def test_crossfit_discovery_never_sees_heldout_reads(deep, monkeypatch):
    from fiberhmm.inference.consensus.lattice_recaller import workflow as W
    seen = []; real = W.Un.tile_units

    def spy(sources, *a, **k):
        if len(a) == 4:            # discovery's call (quantification also passes efficiency)
            seen.extend(u['read_name'] for s in sources for u in s['units'])
        return real(sources, *a, **k)
    monkeypatch.setattr(W.Un, 'tile_units', spy)
    run, root = deep
    run(root/'spy', '--discover-fold-spec', SPEC, region='chr1:1100-1700')
    assert seen and all(fold_of(n) < 5 for n in seen)


def test_posteriors_sum_to_one_per_read_and_group(crossfit_run):
    total = defaultdict(float); rest = {}
    rows = _rows(crossfit_run/'read_table'/RT.READ_TABLE)
    assert rows
    for r in rows:
        if r['scored'] != 'True':
            assert r['posterior'] == '' and r['loglik'] == ''
            continue
        key = (r['window'], r['unit_id'], r['group'])
        total[key] += float(r['posterior'])
        rest[key] = float(r['p_broader']) + float(r['p_other_shape']) + float(r['p_accessible'])
    assert total
    for key in total:
        assert abs(total[key] + rest[key] - 1) < 5e-4 * 6, key


def test_loglik_columns_reproduce_the_posteriors(crossfit_run):
    """posterior = softmax(loglik + log w) with the group's EM weights: the table carries the EM's own matrix."""
    folder = crossfit_run/'window_000001'
    rows = [r for r in _rows(folder/RT.READ_TABLE) if r['scored'] == 'True']
    classes = {r['class_id']: r for r in csv.DictReader(open(folder/'classes.tsv'), delimiter='\t') if r['status'] != 'unscored'}
    groups = defaultdict(dict)
    for r in rows:
        groups[(r['unit_id'], r['group'])][r['class_id']] = r
    checked = 0
    for (_, g), members in groups.items():
        ids = sorted(members)
        if any(c not in classes for c in ids):
            continue
        first = classes[ids[0]]
        w = np.array([float(classes[c]['prevalence']) for c in ids] + [float(first['broader']), float(first['other_shape']),
                                                                      float(first['accessible'])])
        from fiberhmm.inference.consensus.lattice_recaller.model import EM_WEIGHT_FLOOR
        w = np.maximum(w, EM_WEIGHT_FLOOR)     # the EM's floor (weights are rounded to 4 decimals in classes.tsv)
        r0 = members[ids[0]]
        M = np.array([float(members[c]['loglik']) for c in ids] + [float(r0['ll_broader']), float(r0['ll_other_shape']), float(r0['ll_accessible'])])
        A = M + np.log(w); P = np.exp(A - A.max()); P /= P.sum()
        assert np.allclose(P[:len(ids)], [float(members[c]['posterior']) for c in ids], atol=2e-2)
        checked += 1
    assert checked


def test_batch_equals_single_window_runs(deep, crossfit_run):
    run, root = deep
    for i, region in ((1, 'chr1:1100-1700'), (2, 'chr1:3100-3700')):
        single = run(root/f'single{i}', '--read-table', '--discover-fold-spec', SPEC, region=region)
        assert (single/RT.READ_TABLE).read_bytes() == (crossfit_run/f'window_{i:06d}'/RT.READ_TABLE).read_bytes()


def test_parallel_windows_give_identical_tables(deep, crossfit_run):
    run, root = deep
    out = run(root/'parallel', '--read-table', '--discover-fold-spec', SPEC, '--window-jobs', '3', '--cores', '3')
    for name in (RT.READ_TABLE, RT.CLASS_ROLES, RT.READ_ROLES):
        a, b = (out/'read_table'/name).read_bytes(), (crossfit_run/'read_table'/name).read_bytes()
        assert a == b, name


def test_heldout_fold_weights_fit_on_heldout_reads(deep):
    run, root = deep
    out = run(root/'heldout', '--read-table', '--discover-fold-spec', SPEC, '--fold-weights', 'heldout', region='chr1:1100-1700')
    manifest = json.loads((out/'manifest.json').read_text())
    held = manifest['cross_fit']['heldout_units']
    classes = [r for r in csv.DictReader(open(out/'classes.tsv'), delimiter='\t') if r['status'] != 'unscored']
    assert classes and all(int(r['molecules']) <= held for r in classes)
    rows = _rows(out/RT.READ_TABLE)
    assert any(r['scored'] == 'True' and r['discovery'] == 'True' for r in rows)   # discovery reads still scored


# ----------------------------------------------------------------------------------------------- roles
RULES = {'schema': RT.ROLE_SCHEMA, 'rules': [
    {'role': 'small_up', 'end': {'le': 0}, 'width': [None, 60]},
    {'role': 'down', 'start': {'gt': 0}},
    {'role': 'wide_over_down', 'component': 'broader', 'group_roles': ['down']}]}


def test_role_rules_are_strand_aware_first_match_and_validated():
    rules = RT.parse_role_rules(RULES)
    classes = [dict(id='a', start=960., end=990., group=1), dict(id='b', start=1050., end=1080., group=2)]
    roles, broader = RT.assign_roles(rules, classes, 1000, '+')
    assert roles['a'] == (-40., -10., 'small_up') and roles['b'][2] == 'down' and broader == {1: '', 2: 'wide_over_down'}
    roles, broader = RT.assign_roles(rules, classes, 1000, '-')       # minus strand: upstream is to the right
    assert roles['a'] == (10., 40., 'down') and roles['b'] == (-80., -50., 'small_up') and broader[1] == 'wide_over_down'
    assert RT.window_anchor(dict(start=100, end=1200, strand='+'), 300) == 400
    assert RT.window_anchor(dict(start=100, end=1200, strand='-'), 300) == 900
    for bad in ({'rules': [{'role': 'x', 'width': [5, 1]}]}, {'rules': [{'role': 'x', 'size': [1, 2]}]},
                {'rules': [{'role': 'x', 'start': {'above': 1}}]}, {'rules': [{'role': ''}]},
                {'rules': [{'role': 'x', 'group_roles': ['y']}]}, {'schema': 'other', 'rules': []}, {'rules': {}}):
        with pytest.raises(ValueError):
            RT.parse_role_rules(bad)


def test_role_rules_relabel_on_continue_without_recompute(deep, crossfit_run, tmp_path, monkeypatch):
    import shutil
    run, root = deep
    out = tmp_path/'relabel'; shutil.copytree(crossfit_run, out)
    rules = tmp_path/'roles.json'; rules.write_text(json.dumps(RULES))
    calls = []; real = batch.run_unit
    monkeypatch.setattr(batch, 'run_unit', lambda spec, **kw: calls.append(spec['name']) or real(spec, **kw))
    # (the copied run manifest records the original output path's inputs, not the directory: --continue accepts it)
    run(out, '--read-table', '--discover-fold-spec', SPEC, '--continue', '--role-rules', rules, '--anchor-offset', 300)
    assert calls == []
    summary = json.loads((out/'read_table'/'read_table.json').read_text())
    assert summary['roles'] == ['small_up', 'down', 'wide_over_down'] and summary['anchor_offset_bp'] == 300
    roles = list(csv.DictReader(open(out/'read_table'/RT.CLASS_ROLES), delimiter='\t'))
    assert roles and any(r['role'] for r in roles)
    w2 = [r for r in roles if r['window'] == 'w2']
    for r in w2:   # minus-strand window: anchor = end - 300, positions mirrored
        assert int(r['anchor']) == 3700 - 300
        assert float(r['class_rel_start']) == pytest.approx(3400 - float(r['class_end']), abs=.06)
    header = gzip.decompress((out/'read_table'/RT.READ_ROLES).read_bytes()).decode().splitlines()[0].split('\t')
    assert header[-3:] == ['p_small_up', 'p_down', 'p_wide_over_down']


def test_read_roles_sum_within_group_and_take_the_best_group(tmp_path):
    folder = tmp_path/'window_000001'; folder.mkdir()
    (folder/'classes.tsv').write_text('class_id\tchannel\tstart\tend\tstatus\tprevalence\tsupported\n'
                                      'c1\td::pooled\t1010.0\t1030.0\tsupported\t0.3\tTrue\n'
                                      'c2\td::pooled\t1015.0\t1040.0\tsupported\t0.4\tTrue\n'
                                      'c3\td::pooled\t1200.0\t1220.0\tunsupported\t0.1\tFalse\n')
    base = dict(fold=0, discovery=True, dataset='d', strand='pooled', unit_id='u', class_status='supported', scored=True,
                log_bf=0, label='abstain', p_other_shape=0, p_accessible=0, loglik=0, ll_broader=0, ll_other_shape=0, ll_accessible=0)
    rows = [dict(base, read_name='r', group=1, class_id='c1', class_start=1010., class_end=1030., posterior=.3, p_broader=.2),
            dict(base, read_name='r', group=1, class_id='c2', class_start=1015., class_end=1040., posterior=.4, p_broader=.2),
            dict(base, read_name='r', group=2, class_id='c3', class_start=1200., class_end=1220., posterior=.6, p_broader=.1)]
    RT.write_window_table(folder/RT.READ_TABLE, rows)
    rules = RT.parse_role_rules({'rules': [{'role': 'down', 'start': {'gt': 0}},
                                           {'role': 'wide', 'component': 'broader', 'group_roles': ['down']}]})
    spec = dict(index=0, folder=str(folder), window=dict(name='w', chrom='c', start=1000, end=2000, strand='+'))
    RT.aggregate(tmp_path, [spec], rules=rules, anchor_offset=0)
    roles = _rows(tmp_path/RT.READ_ROLES)
    assert len(roles) == 1 and roles[0]['groups_scored'] == '2'
    assert float(roles[0]['p_down']) == pytest.approx(.7) and float(roles[0]['p_wide']) == pytest.approx(.2)
    table = list(csv.DictReader(open(tmp_path/RT.CLASS_ROLES), delimiter='\t'))
    assert [(r['class_id'], r['group'], r['status'], r['supported_channels']) for r in table] == [
        ('c1', '1', 'supported', '1'), ('c2', '1', 'supported', '1'), ('c3', '2', 'unsupported', '0')]


def test_cli_validates_role_and_crossfit_flags(deep, tmp_path, capsys):
    run, root = deep
    bad = tmp_path/'bad.json'; bad.write_text('{"rules": [{"role": "x", "wdth": [1, 2]}]}')
    for extra in (['--role-rules', bad, '--read-table'], ['--role-rules', bad], ['--fold-weights', 'heldout'],
                  ['--discover-fold-spec', 'mod=10,discover=10']):
        with pytest.raises(SystemExit) as error:
            run(tmp_path/'x', *extra)
        assert error.value.code == 2


# ------------------------------------------------------------------------------------------- NFR states
def _fake_nfr(root, names, state_of, region=('chr1', 900, 1900)):
    root.mkdir(parents=True)
    (root/'result.json').write_text(json.dumps(dict(region=dict(chrom=region[0], start=region[1], end=region[2]))))
    (root/'variants.tsv').write_text('nfr\tnfr_start\tnfr_end\tid\nnfr_1\t1200\t1400\tV1\n')
    lines = ['nfr\tunit_id\tread_name\tdataset\tstrand\tstatus\tmap\tmap_posterior\tp\tp_other\tgaps']
    for n in names:
        lines.append(f'nfr_1\tu\t{n}\tdataset_1\tCT\tcallable\t{state_of(n)}\t0.9\t\t\t')
    lines.append(f'nfr_1\tu\textra_read\tdataset_1\tCT\tcensored\t\t\t\t\t')
    with gzip.open(root/'molecules.tsv.gz', 'wt') as handle:
        handle.write('\n'.join(lines)+'\n')


def test_nfr_states_condition_class_prevalence(deep, crossfit_run, tmp_path):
    import shutil
    run, root = deep
    rows = _rows(crossfit_run/'window_000001'/RT.READ_TABLE)
    names = sorted({r['read_name'] for r in rows})
    state = lambda n: 'open' if fold_of(n, 2) else 'closed'
    _fake_nfr(tmp_path/'nfr', names, state)
    out = tmp_path/'states'; shutil.copytree(crossfit_run, out)
    run(out, '--read-table', '--discover-fold-spec', SPEC, '--continue', '--nfr-states', tmp_path/'nfr')
    table = list(csv.DictReader(open(out/'read_table'/'class_by_nfr_state.tsv'), delimiter='\t'))
    assert {r['window'] for r in table} == {'w1'}                 # the NFR run overlaps window 1 only
    assert {r['state'] for r in table} == {'all', 'open', 'closed'}
    summary = json.loads((out/'read_table'/'read_table.json').read_text())['nfr_states']
    assert summary['joined'] == len(names)
    # em_prevalence is the EM over the state's reads with the table's log-likelihoods.
    scored = [r for r in rows if r['scored'] == 'True']
    one = next(r for r in table if r['state'] == 'open')
    g = [r for r in scored if r['group'] == one['group'] and r['strand'] == one['strand'] and state(r['read_name']) == 'open']
    ids = sorted({r['class_id'] for r in g}, key=lambda c: [x['class_id'] for x in g].index(c))
    by_read = defaultdict(dict)
    for r in g:
        by_read[r['read_name']][r['class_id']] = r
    M = np.array([[float(v[c]['loglik']) for c in ids] + [float(v[ids[0]][k]) for k in ('ll_broader', 'll_other_shape', 'll_accessible')]
                  for v in by_read.values()])
    from fiberhmm.inference.consensus.lattice_recaller.model import em
    w, _ = em(M)
    assert int(one['reads']) == len(by_read)
    assert float(one['em_prevalence']) == pytest.approx(float(w[ids.index(one['class_id'])]), abs=1e-4)


def test_nfr_states_without_any_joined_read_fail(deep, crossfit_run, tmp_path):
    import shutil
    run, root = deep
    _fake_nfr(tmp_path/'nfr', ['renamed|x'], lambda n: 'open')
    out = tmp_path/'states'; shutil.copytree(crossfit_run, out)
    with pytest.raises((ValueError, SystemExit)):
        run(out, '--read-table', '--discover-fold-spec', SPEC, '--continue', '--nfr-states', tmp_path/'nfr')


# ----------------------------------------------------------------------------------------------- memory
def test_window_memory_does_not_grow_with_window_count(tmp_path):
    """Peak traced Python allocation of an in-process batch (one window at a time, read tables on) is set by the
    largest window, not by how many windows the BED has; the merged tables stream one window at a time."""
    bam = make_windows_bam(tmp_path/'in.bam', n_windows=8, reads=30, seed=9)
    peaks = {}
    for n in (2, 8):
        bed = tmp_path/f'w{n}.bed'
        bed.write_text(''.join(f'chr1\t{1100+2000*w}\t{1700+2000*w}\tw{w+1}\n' for w in range(n)))
        tracemalloc.start()
        with redirect_stdout(io.StringIO()):
            main(['--bam', str(bam), '--bed', str(bed), '--no-bam', '--cores', '1', '--window-jobs', '1', '--read-table',
                  '--output', str(tmp_path/f'out{n}')])
        peaks[n] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    assert peaks[8] < 1.35*peaks[2], peaks


def test_max_memory_gb_lowers_window_concurrency(monkeypatch):
    from fiberhmm.inference.consensus import cli, pooled_sampling

    class Args:
        max_memory_gb = 2.0
    monkeypatch.setattr(pooled_sampling, 'scanned_molecules', lambda datasets, windows, options: 1000)
    reports = []

    class Progress:
        def report(self, stage, message, **_):
            reports.append(message)
    windows = [dict(chrom='c', start=i*2000, end=i*2000+1100, name=f'w{i}') for i in range(20)]
    cores, jobs = cli._window_memory_plan(cli.p_exit(Args), Args, windows, [], {}, 8, 0, Progress())
    assert 1 <= jobs < 8 and cores >= jobs
    from fiberhmm.inference.consensus.memory_estimate import estimate
    assert jobs*estimate(1000, 1100, cores//jobs, pad=250, read_kb=30)['total_gb'] <= 2.0
    Args.max_memory_gb = 0.01
    with pytest.raises(SystemExit):
        cli._window_memory_plan(cli.p_exit(Args), Args, windows, [], {}, 8, 0, Progress())


def test_keep_going_records_failed_windows_and_continue_retries_them(deep, tmp_path, monkeypatch, capsys):
    run, root = deep
    real = batch.run_unit; calls = []; fail = {'w2'}

    def flaky(spec, **kw):
        calls.append(spec['name'])
        if spec['name'] in fail:
            raise RuntimeError('simulated window failure')
        return real(spec, **kw)
    monkeypatch.setattr(batch, 'run_unit', flaky)
    out = tmp_path/'kg'
    with pytest.raises(SystemExit) as error:
        run(out, '--read-table', '--keep-going', '--window-jobs', '1')
    assert error.value.code == 1 and 'w2' in capsys.readouterr().err
    manifest = json.loads((out/batch.RUN_MANIFEST).read_text())
    assert manifest['status'] == 'incomplete' and [f['index'] for f in manifest['failed_windows']] == [1]
    regions = json.loads((out/'regions.json').read_text())
    assert [r['status'] for r in regions][1] == 'failed'
    assert {r['window'] for r in _rows(out/'read_table'/RT.READ_TABLE)} <= {'w1', 'w3'}
    calls.clear(); fail.clear()
    run(out, '--read-table', '--keep-going', '--continue', '--window-jobs', '1')
    assert calls == ['w2']
    manifest = json.loads((out/batch.RUN_MANIFEST).read_text())
    assert manifest['status'] == 'complete' and 'failed_windows' not in manifest
    # Without --keep-going the first failure still stops the run.
    calls.clear(); fail.add('w2')
    with pytest.raises((RuntimeError, SystemExit)):
        run(tmp_path/'stop', '--read-table', '--window-jobs', '1')
    assert calls == ['w1', 'w2']


def test_keep_going_in_parallel_with_memory_plan_and_a_bad_window(deep, tmp_path, capsys):
    """A BED row on a contig the BAM lacks fails in its own worker; the planner does not stop on it either."""
    run, root = deep
    bed = tmp_path/'w.bed'
    bed.write_text('chr1\t1100\t1700\tw1\nchrAbsent\t100\t700\tbad\nchr1\t3100\t3700\tw2\n')
    out = tmp_path/'par'
    with pytest.raises(SystemExit) as error:
        run(out, '--read-table', '--keep-going', '--window-jobs', '3', '--cores', '3', '--max-memory-gb', '50', bed_path=bed)
    err = capsys.readouterr().err
    assert error.value.code == 1 and 'bad' in err and 'not counted' in err
    manifest = json.loads((out/batch.RUN_MANIFEST).read_text())
    assert manifest['status'] == 'incomplete' and [f['index'] for f in manifest['failed_windows']] == [1]
    assert all((out/f'window_{i:06d}'/batch.UNIT_MARKER).is_file() for i in (1, 3))
    assert {r['window'] for r in _rows(out/'read_table'/RT.READ_TABLE)} == {'w1', 'w2'}
