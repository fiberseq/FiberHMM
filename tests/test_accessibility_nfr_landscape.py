"""fiberhmm-nfr landscape options (EXPERIMENTAL preview): skipped stages, inline --parameters, parallel regions and
datasets, the compact loader hook, cross-fitted discovery, edge consistency and restartable BED runs.

Every execution option must leave the kept outputs byte-identical to the plain sequential run.
"""
import gzip
import json
import os
import random
from pathlib import Path

import numpy as np
import pytest

from fiberhmm.inference.accessibility import run_accessibility
from fiberhmm.inference.accessibility import gaps as G
from fiberhmm.inference.accessibility import variants as V
from fiberhmm.inference.accessibility.cli import main
from fiberhmm.io.folds import FoldSpec, fold_of

from test_accessibility_nfr import FAST, nfr_bam, payload, planted

KEPT = ('variants.tsv', 'configurations.tsv', 'molecules.tsv.gz', 'context.json.gz')
REGIONS = [(780, 1020), (1380, 1720), (2080, 2320)]


def _units(n=320, seed=31):
    states = [(.3, [(800, 1000), (1400, 1700), (2100, 2300)]), (.2, [(1400, 1520), (1620, 1700), (2100, 2300)]),
              (.15, [(1480, 1620)]), (.15, [(800, 1000)]), (.2, [])]
    return planted(states, n=n, seed=seed)[0]


def _bytes(d, names=KEPT):
    return {n: (Path(d)/n).read_bytes() for n in names}


# ---------------------------------------------------------------- skipped stages
@pytest.mark.parametrize('skip', [('coaccess', 'combinations'), ('coaccess',), ('combinations',)])
def test_skipping_stages_keeps_the_other_outputs_byte_identical(tmp_path, skip):
    units = _units()
    run_accessibility(payload(units), dict(FAST, nfr_regions=REGIONS), tmp_path/'full')
    r = run_accessibility(payload(units), dict(FAST, nfr_regions=REGIONS, skip=list(skip)), tmp_path/'skip')
    assert _bytes(tmp_path/'full') == _bytes(tmp_path/'skip')
    assert r['skipped_stages'] == list(skip)
    assert (tmp_path/'skip'/'coaccess.tsv').exists() == ('coaccess' not in skip)
    assert (tmp_path/'skip'/'combos.tsv').exists() == ('combinations' not in skip)
    if 'coaccess' in skip:
        assert r['pairs'] == [] and r['family']['stage'] == 'skipped'
    manifest = json.loads((tmp_path/'skip'/'manifest.json').read_text())
    assert manifest['skipped_stages'] == list(skip) and manifest['parameters']['skip'] == list(skip)
    # a run that skips nothing records no skip parameter (its receipts keep their earlier bytes)
    assert 'skip' not in json.loads((tmp_path/'full'/'manifest.json').read_text())['parameters']


def test_unknown_stage_is_refused():
    from fiberhmm.inference.accessibility import NFROptions
    with pytest.raises(ValueError, match='unknown stage'):
        NFROptions.from_params(dict(skip=['pairs']))
    assert NFROptions.from_params(dict(skip='combinations,coaccess'.split(','))).skip == ('coaccess', 'combinations')


def test_cli_fast_landscape_and_inline_parameters(tmp_path, capsys):
    bam = nfr_bam(tmp_path/'nfr.bam')
    params = tmp_path/'p.json'
    params.write_text(json.dumps({'nfr': {'min_gap_bp': 80}}))
    common = ['--bam', str(bam), '--region', 'chr1:700-2400', '--bootstrap', '10', '--cores', '1']
    main(common + ['--parameters', str(params), '--output', str(tmp_path/'file')])
    main(common + ['--parameters', '{"nfr": {"min_gap_bp": 80}}', '--fast-landscape', '--output', str(tmp_path/'inline')])
    main(common + ['--parameters', ' {"nfr":{"min_gap_bp":80}}', '--skip', 'coaccess,combinations',
                   '--output', str(tmp_path/'skip')])
    capsys.readouterr()
    assert _bytes(tmp_path/'file') == _bytes(tmp_path/'inline') == _bytes(tmp_path/'skip')
    assert not (tmp_path/'inline'/'coaccess.tsv').exists() and not (tmp_path/'inline'/'combos.tsv').exists()
    assert json.loads((tmp_path/'inline'/'result.json').read_text())['parameters']['min_gap_bp'] == 80
    # the shell strips the quotes of an unquoted JSON argument: a clear error says to quote it
    with pytest.raises(SystemExit):
        main(common + ['--parameters', '{nfr:{min_gap_bp:150}}', '--output', str(tmp_path/'bad')])
    assert 'inline JSON does not parse' in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(common + ['--parameters', str(tmp_path/'missing.json'), '--output', str(tmp_path/'bad2')])
    assert 'neither a JSON file nor an inline JSON object' in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(common + ['--parameters', '[1, 2]', '--output', str(tmp_path/'bad3')])
    assert 'neither a JSON file' in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(common + ['--skip', 'pairs', '--output', str(tmp_path/'bad4')])
    assert 'unknown stage' in capsys.readouterr().err


# ---------------------------------------------------------------- parallel execution
def test_regions_fitted_in_worker_processes_match_the_sequential_run(tmp_path):
    units = _units()
    run_accessibility(payload(units), dict(FAST, nfr_regions=REGIONS), tmp_path/'one')
    run_accessibility(payload(units), dict(FAST, nfr_regions=REGIONS), tmp_path/'three', jobs=3)
    names = KEPT + ('coaccess.tsv', 'combos.tsv', 'result.json')
    assert _bytes(tmp_path/'one', names) == _bytes(tmp_path/'three', names)


def test_parallel_dataset_loading_and_compact_units_match(tmp_path, capsys):
    bam = nfr_bam(tmp_path/'nfr.bam')
    common = ['--bam', str(bam), '--bam', str(bam), '--region', 'chr1:700-2400', '--bootstrap', '10']
    main(common + ['--cores', '1', '--output', str(tmp_path/'serial')])
    main(common + ['--cores', '2', '--output', str(tmp_path/'parallel')])
    capsys.readouterr()
    names = KEPT + ('coaccess.tsv', 'combos.tsv', 'result.json')
    assert _bytes(tmp_path/'serial', names) == _bytes(tmp_path/'parallel', names)


def test_the_compact_unit_hook_does_not_change_the_analysis(tmp_path):
    from fiberhmm.inference.consensus.bam import load_bam_payload
    from fiberhmm.inference.consensus.parameters import parse_options
    bam = nfr_bam(tmp_path/'nfr.bam')
    datasets = [dict(dataset_id='d', paths=[str(bam)])]
    region = dict(chrom='chr1', start=700, end=2400)
    options = parse_options({'cr': {'engine': 'lattice_recaller'}})
    full = load_bam_payload(datasets, region, options)
    compact = load_bam_payload(datasets, region, options, unit_hook=G.compact_unit)
    u = compact['strata'][0]['units'][0]
    assert not set(G.LATTICE_ONLY_FIELDS) & set(u) and isinstance(u['positions'], np.ndarray)
    assert [x['unit_id'] for x in full['strata'][0]['units']] == [x['unit_id'] for x in compact['strata'][0]['units']]
    run_accessibility(full, dict(FAST), tmp_path/'full')
    run_accessibility(compact, dict(FAST), tmp_path/'compact')
    names = KEPT + ('coaccess.tsv', 'combos.tsv', 'result.json')
    assert _bytes(tmp_path/'full', names) == _bytes(tmp_path/'compact', names)


def test_upstream_recall_helpers_match_their_references():
    from fiberhmm.inference.consensus import upstream_recall as U
    from fiberhmm.inference.consensus.adapter import reference_gap_inside
    from fiberhmm.inference.consensus.geometry import merged
    from copy import deepcopy
    rng = random.Random(4)
    for trial in range(200):
        n = rng.randint(1, 300)
        refs = []
        x = rng.randint(0, 50)
        for _ in range(n):           # insertions (-1) and deletions (jumps)
            if rng.random() < .1:
                refs.append(-1)
            else:
                x += rng.choice((1, 1, 1, 2, 7)); refs.append(x)
        if trial % 10 == 0:
            rng.shuffle(refs)        # not monotone: the slicing path
        refs = np.asarray(refs, np.int64)
        project = U._projector(refs)
        for _ in range(30):
            a = rng.randint(-3, n + 3); b = rng.randint(-3, n + 5)
            vals = refs[a:b]; vals = vals[vals >= 0]
            want = [int(vals.min()), int(vals.max() + 1)] if len(vals) else None
            assert project(a, b) == want, (a, b)
        blocks = merged([[s, s + rng.randint(1, 40)] for s in sorted(rng.sample(range(0, 2000), rng.randint(1, 30)))])
        gap = U._gap_counter(blocks)
        ivs = [[rng.randint(0, 1000), rng.randint(0, 1000)] for _ in range(rng.randint(0, 25))]
        ivs = [[min(a, b), max(a, b) + 1] for a, b in ivs]
        ov = U._Overlaps(ivs)
        for _ in range(30):
            lo = rng.randint(-10, 2100); hi = lo + rng.randint(1, 300)
            assert gap(lo, hi) == reference_gap_inside(lo, hi, blocks)
            assert ov.any(lo, hi) == any(lo < b and hi > a for a, b in ivs)
    shared = [1, 2]
    tree = dict(calls=[dict(interval=shared, llr=1.5, flag=True)], excluded=[dict(interval=shared, reason='x')],
                nested=[[1, [2, (3, 4)]], None, 'text'])
    a, b = U._deepcopy(tree), deepcopy(tree)
    assert a == b and a['calls'][0]['interval'] is a['excluded'][0]['interval'] and a['calls'][0]['interval'] is not shared


# ---------------------------------------------------------------- cross-fitting
def test_all_folds_discovering_equals_a_run_without_folds(tmp_path):
    units = _units()
    run_accessibility(payload(units), dict(FAST), tmp_path/'plain')
    r = run_accessibility(payload(units), dict(FAST, discover_fold_spec='mod=3,discover=0-2'), tmp_path/'folds')
    for name in ('variants.tsv', 'configurations.tsv'):
        assert (tmp_path/'plain'/name).read_bytes() == (tmp_path/'folds'/name).read_bytes()
    plain = gzip.open(tmp_path/'plain'/'molecules.tsv.gz', 'rt').read().splitlines()
    folds = gzip.open(tmp_path/'folds'/'molecules.tsv.gz', 'rt').read().splitlines()
    assert folds[0].split('\t')[-2:] == ['fold', 'discovery']
    assert [ln.rsplit('\t', 2)[0] for ln in folds] == plain
    assert r['cross_fit']['heldout_molecules'] == 0


def test_cross_fitted_discovery_uses_only_discovery_folds_and_labels_every_read(tmp_path, monkeypatch):
    units = _units(n=480)
    spec = FoldSpec.parse('hash=md5,mod=10,discover=0-4')
    seen = []
    real = V.discover

    def spy(reads, *a, **k):
        seen.append({r['read_name'] for r in reads if r['callable']})
        return real(reads, *a, **k)
    monkeypatch.setattr(V, 'discover', spy)
    r = run_accessibility(payload(units), dict(FAST, nfr_regions=REGIONS, discover_fold_spec='discover=0-4'), tmp_path/'cf')
    assert seen and all(spec.discovers(name) for names in seen for name in names)
    heldout = {u['read_name'] for u in units if not spec.discovers(u['read_name'])}
    assert heldout and not heldout & set().union(*seen)
    rows = [ln.split('\t') for ln in gzip.open(tmp_path/'cf'/'molecules.tsv.gz', 'rt').read().splitlines()]
    head = rows[0]; rows = [dict(zip(head, x)) for x in rows[1:]]
    assert {x['read_name'] for x in rows} == {u['read_name'] for u in units}
    for x in rows:
        assert int(x['fold']) == fold_of(x['read_name']) and (x['discovery'] == 'True') == spec.discovers(x['read_name'])
    callable_heldout = [x for x in rows if x['status'] == 'callable' and x['discovery'] == 'False']
    assert callable_heldout and all(x['map'] for x in callable_heldout)        # held-out reads get membership too
    assert r['cross_fit']['spec'] == 'hash=md5,mod=10,discover=0-4'
    assert r['parameters']['discover_fold_spec'] == 'hash=md5,mod=10,discover=0-4'
    # held-out weights: a different fit, every read still labelled
    r2 = run_accessibility(payload(units), dict(FAST, nfr_regions=REGIONS, discover_fold_spec='discover=0-4',
                                                fold_weights='heldout'), tmp_path/'cfh')
    assert [v['name'] for n in r2['nfrs'] for v in n['variants']] == [v['name'] for n in r['nfrs'] for v in n['variants']]
    assert r2['cross_fit']['weights'] == 'heldout'
    with pytest.raises(ValueError):
        run_accessibility(payload(units), dict(FAST, fold_weights='heldout'))


def test_cli_discover_fold_spec(tmp_path, capsys):
    bam = nfr_bam(tmp_path/'nfr.bam')
    main(['--bam', str(bam), '--region', 'chr1:700-2400', '--bootstrap', '10', '--cores', '1', '--fast-landscape',
          '--discover-fold-spec', 'hash=md5,mod=4,discover=0+2', '--output', str(tmp_path/'cf')])
    capsys.readouterr()
    head = gzip.open(tmp_path/'cf'/'molecules.tsv.gz', 'rt').readline().rstrip('\n').split('\t')
    assert head[-2:] == ['fold', 'discovery']
    with pytest.raises(SystemExit):
        main(['--bam', str(bam), '--region', 'chr1:700-2400', '--discover-fold-spec', 'mod=4,discover=7',
              '--output', str(tmp_path/'bad')])
    assert 'folds must lie' in capsys.readouterr().err


# ---------------------------------------------------------------- edge consistency
def test_edge_consistency_separates_tight_from_scattered_edges():
    rng = np.random.default_rng(2)
    tight = [G.index_unit(dict(u, dataset='d')) for u in planted([(1, [(1400, 1700)])], n=200, seed=1, jitter=3.)[0]]
    loose = [G.index_unit(dict(u, dataset='d')) for u in planted([(1, [(1400, 1700)])], n=200, seed=1, jitter=60.)[0]]
    out = {}
    for name, units in (('tight', tight), ('loose', loose)):
        r = run_accessibility(payload(units), dict(FAST, nfr_regions=[(1300, 1800)], skip=['coaccess', 'combinations']))
        out[name] = r['nfrs'][0]['variants'][0]
    assert out['tight']['edge_within'] > 0.5 > out['loose']['edge_within']
    assert out['loose']['edge_dispersion'] > 3*out['tight']['edge_dispersion']
    assert out['tight']['member_gaps'] > 150
    assert rng is not None


# ---------------------------------------------------------------- restartable BED runs
def test_bed_run_resumes_isolates_empty_windows_and_combines_tables(tmp_path, capsys):
    bam = nfr_bam(tmp_path/'nfr.bam')
    bed = tmp_path/'w.bed'
    bed.write_text('chr1\t700\t2400\tlocus\nchr1\t3000\t3900\tempty\nchr1\t800\t2300\tinner\n')
    common = ['--bam', str(bam), '--bed', str(bed), '--bootstrap', '10', '--cores', '2', '--fast-landscape',
              '--output', str(tmp_path/'run')]
    main(common)
    capsys.readouterr()
    run = tmp_path/'run'
    summary = json.loads((run/'regions.json').read_text())
    assert [r['name'] for r in summary] == ['locus', 'empty', 'inner']
    assert [r['status'] for r in summary] == ['complete', 'no_molecules', 'complete']
    tsv = (run/'variants.tsv').read_text().splitlines()
    assert tsv[0].startswith('window\twindow_chrom\twindow_start\twindow_end\tnfr\t')
    assert {ln.split('\t')[0] for ln in tsv[1:]} == {'locus', 'inner'}
    assert (run/'regions.tsv').read_text().splitlines()[0].startswith('index\tname\tchrom')
    # the same window as a --region run: identical bytes
    main(['--bam', str(bam), '--region', 'chr1:700-2400', '--bootstrap', '10', '--cores', '1', '--fast-landscape',
          '--output', str(tmp_path/'single')])
    capsys.readouterr()
    assert _bytes(tmp_path/'single') == _bytes(run/'window_000001')
    # --continue redoes only what is missing or damaged
    stamp = (run/'window_000001'/'window_complete.json').stat().st_mtime_ns
    (run/'window_000003'/'variants.tsv').write_text('damaged\n')
    with pytest.raises(SystemExit):
        main(common)                                               # never overwrites without --continue
    assert '--continue' in capsys.readouterr().err
    main(common + ['--continue'])
    capsys.readouterr()
    assert (run/'window_000001'/'window_complete.json').stat().st_mtime_ns == stamp
    assert (run/'window_000003'/'variants.tsv').read_text() != 'damaged\n'
    with pytest.raises(SystemExit):
        main(common + ['--continue', '--edge-tolerance-bp', '10'])  # different parameters
    assert 'continue refused' in capsys.readouterr().err


def test_bed_failure_is_reported_per_window(tmp_path, capsys, monkeypatch):
    from fiberhmm.inference.accessibility import batch
    bam = nfr_bam(tmp_path/'nfr.bam')
    bed = tmp_path/'w.bed'
    bed.write_text('chr1\t700\t2400\tgood\nchr1\t800\t2300\tbad\n')
    real = batch.window_task

    def flaky(spec):
        if spec['name'] == 'bad' and not os.environ.get('NFR_TEST_FIXED'):
            spec = dict(spec, datasets=[dict(spec['datasets'][0], paths=[str(tmp_path/'gone.bam')])])
        return real(spec)
    monkeypatch.setattr(batch, 'window_task', flaky)
    common = ['--bam', str(bam), '--bed', str(bed), '--bootstrap', '10', '--cores', '1', '--fast-landscape',
              '--output', str(tmp_path/'run')]
    with pytest.raises(SystemExit) as error:
        main(common)
    assert error.value.code == 1
    summary = json.loads((tmp_path/'run'/'regions.json').read_text())
    assert [r['status'] for r in summary] == ['complete', 'failed'] and summary[1]['error']
    capsys.readouterr()
    monkeypatch.setenv('NFR_TEST_FIXED', '1')
    main(common + ['--continue'])
    assert [r['status'] for r in json.loads((tmp_path/'run'/'regions.json').read_text())] == ['complete', 'complete']


# ---------------------------------------------------------------- real data (opt-in)
@pytest.mark.skipif(not os.environ.get('FIBERHMM_NFR_SMOKE_BAM'), reason='set FIBERHMM_NFR_SMOKE_BAM (and _REGION)')
def test_real_bam_smoke(tmp_path, capsys):
    """Opt-in: FIBERHMM_NFR_SMOKE_BAM=<indexed called BAM> FIBERHMM_NFR_SMOKE_REGION=<chrom:start-end>."""
    bam, region = os.environ['FIBERHMM_NFR_SMOKE_BAM'], os.environ['FIBERHMM_NFR_SMOKE_REGION']
    common = ['--bam', bam, '--region', region, '--bootstrap', '5', '--parameters', '{"nfr":{"min_gap_bp":150}}']
    main(common + ['--cores', '1', '--output', str(tmp_path/'full')])
    main(common + ['--cores', '4', '--fast-landscape', '--output', str(tmp_path/'fast')])
    capsys.readouterr()
    assert _bytes(tmp_path/'full') == _bytes(tmp_path/'fast')
