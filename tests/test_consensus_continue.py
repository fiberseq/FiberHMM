"""fiberhmm-consensus multi-window runs: unit-level restart (--continue) and window parallelism.

An interrupted + continued run, a serial run and a parallel run must give the same outputs (timing fields,
execution-only compute.cores and absolute output paths aside); changed inputs or parameters are refused;
partial and stale units are redone.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import random
from contextlib import redirect_stdout
from pathlib import Path

import pysam
import pytest

from fiberhmm.inference.consensus import batch
from fiberhmm.inference.consensus.cli import main

TIMING = {'seconds', 'numerical_environment', 'native_timings', 'timings'}


def make_windows_bam(path, n_windows=3, reads=30, seed=2):
    """DddA reads over ``n_windows`` loci, each with shared protected footprints and matching native tf calls."""
    from fiberhmm.io.bam_header import append_chemistry
    rng = random.Random(seed)
    ref = ''.join(rng.choice('ACGT') for _ in range(20_000))
    header = pysam.AlignmentHeader.from_dict(dict(HD={'VN': '1.6', 'SO': 'coordinate'}, SQ=[{'SN': 'chr1', 'LN': 20_000}]))
    header = append_chemistry(header, dict(assay='daf', enzyme='ddda', platform='pacbio', mode='daf'))
    records = []
    for w in range(n_windows):
        base = 1000+w*2000
        for i in range(reads):
            start = base+rng.randint(0, 20); fp = rng.choice([(300, 345), (300, 345), (380, 420), None])
            seq = []
            for q in range(600):
                b = ref[start+q]
                if b == 'C' and not (fp and fp[0] <= start+q-base < fp[1]) and rng.random() < .85: b = 'Y'
                seq.append(b)
            r = pysam.AlignedSegment(header); r.query_name = f'w{w}_r{i}'; r.query_sequence = ''.join(seq)
            r.reference_id = 0; r.reference_start = start; r.mapping_quality = 60; r.cigarstring = '600M'
            r.set_tag('st', 'CT')
            r.set_tag('MA', '600;msp.:1-600'+(f';tf.:{fp[0]-(start-base)+1}-{fp[1]-fp[0]}' if fp else ''))
            records.append(r)
    records.sort(key=lambda r: r.reference_start)
    with pysam.AlignmentFile(str(path), 'wb', header=header) as handle:
        for r in records: handle.write(r)
    pysam.index(str(path))
    return path


@pytest.fixture
def windows_run(tmp_path):
    bam = make_windows_bam(tmp_path/'in.bam')
    bed = tmp_path/'w.bed'
    bed.write_text(''.join(f'chr1\t{1100+2000*w}\t{1700+2000*w}\tw{w+1}\n' for w in range(3)))
    def run(output, *extra):
        with redirect_stdout(io.StringIO()):
            main(['--bam', str(bam), '--bed', str(bed), '--output', str(output), *map(str, extra)])
    return run, bam, bed


def _norm(value):
    if isinstance(value, dict):
        out = {k: _norm(v) for k, v in value.items() if k not in TIMING}
        if isinstance(out.get('compute'), dict): out['compute'].pop('cores', None)
        return out
    if isinstance(value, list): return [_norm(v) for v in value]
    return value


def snapshot(root):
    """Every output file's content: BAM records, JSON without timing fields, other files as bytes."""
    root = Path(root); out = {}
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel.startswith('logs/') or rel == batch.RUN_MANIFEST or path.suffix in ('.csi', '.bai') or path.name == batch.UNIT_MARKER:
            continue
        if path.suffix == '.bam':
            with pysam.AlignmentFile(str(path), 'rb', check_sq=False) as handle:
                out[rel] = [r.to_string() for r in handle.fetch(until_eof=True)]
            continue
        data = path.read_bytes()
        if path.suffix == '.gz': data = gzip.decompress(data)
        data = data.decode(errors='replace').replace(str(root), '<OUT>')
        if path.name == 'report.html':      # carries "elapsed inference X s"
            import re; data = re.sub(r'elapsed inference [0-9.]+ s', 'elapsed inference - s', data)
        out[rel] = _norm(json.loads(data)) if '.json' in path.name else data
    return out


def test_parallel_windows_equal_serial(windows_run, tmp_path):
    run, _, _ = windows_run
    run(tmp_path/'serial', '--cores', 1, '--window-jobs', 1)
    run(tmp_path/'parallel', '--cores', 2, '--window-jobs', 2)
    serial, parallel = snapshot(tmp_path/'serial'), snapshot(tmp_path/'parallel')
    assert serial == parallel
    assert any('tf_consensus' in r for r in serial['bams/001_dataset_1.families.bam'])   # classes were exported
    assert len(list((tmp_path/'parallel'/'logs').glob('window_*.log'))) == 3


def test_interrupted_run_continues_to_identical_outputs(windows_run, tmp_path, monkeypatch):
    run, _, _ = windows_run
    run(tmp_path/'reference', '--cores', 1, '--window-jobs', 1)
    calls = []; real = batch.run_unit

    def interrupt_third(spec, **kw):
        calls.append(spec['name'])
        if len(calls) == 3: raise KeyboardInterrupt
        return real(spec, **kw)
    monkeypatch.setattr(batch, 'run_unit', interrupt_third)
    out = tmp_path/'run'
    with pytest.raises(KeyboardInterrupt):
        run(out, '--cores', 1, '--window-jobs', 1)
    assert (out/'window_000002'/batch.UNIT_MARKER).is_file() and not (out/'window_000003').exists()
    assert not (out/'regions.json').exists() and not (out/'bams').exists()
    # A partial third window (killed mid-write) and a damaged second window are redone; the first is kept.
    (out/'window_000003').mkdir(); (out/'window_000003'/'classes.tsv').write_text('partial')
    with (out/'window_000002'/'classes.tsv').open('a') as handle: handle.write('damaged\n')
    calls.clear(); monkeypatch.setattr(batch, 'run_unit', lambda spec, **kw: calls.append(spec['name']) or real(spec, **kw))
    with pytest.raises(SystemExit):              # a non-empty output directory needs --continue
        run(out, '--cores', 1)
    run(out, '--continue', '--cores', 2, '--window-jobs', 1)
    assert calls == ['w2', 'w3']
    assert snapshot(out) == snapshot(tmp_path/'reference')
    manifest = json.loads((out/batch.RUN_MANIFEST).read_text())
    assert manifest['status'] == 'complete' and len(manifest['attempts']) == 2
    # Continuing a complete run reruns nothing and rebuilds identical aggregates.
    calls.clear(); run(out, '--continue', '--cores', 1)
    assert calls == [] and snapshot(out) == snapshot(tmp_path/'reference')


def test_continue_refuses_changed_parameters_and_inputs(windows_run, tmp_path, capsys):
    run, bam, bed = windows_run
    out = tmp_path/'run'
    run(out, '--cores', 1, '--window-jobs', 1)
    params = tmp_path/'p.json'; params.write_text(json.dumps({'recaller': {'call_max_bp': 90}}))
    with pytest.raises(SystemExit) as error:
        run(out, '--continue', '--parameters', params)
    assert error.value.code == 2 and 'parameters.recaller' in capsys.readouterr().err
    os.utime(bam, ns=(bam.stat().st_atime_ns, bam.stat().st_mtime_ns+10**9))
    with pytest.raises(SystemExit):
        run(out, '--continue')
    assert 'datasets' in capsys.readouterr().err
    with pytest.raises(SystemExit):              # nothing to continue from
        run(tmp_path/'fresh', '--continue')
    with pytest.raises(SystemExit):              # pooled and evidence replays are one analysis
        run(out, '--continue', '--pool-loci')


def test_unit_marker_digest_and_files_are_checked(tmp_path):
    folder = tmp_path/'u'; folder.mkdir(); (folder/'a.tsv').write_text('x')
    from fiberhmm.inference.consensus.artifacts import write_json
    write_json(folder/batch.UNIT_MARKER, dict(schema=batch.UNIT_SCHEMA, digest='d', files={'a.tsv': 1}))
    assert batch.completed_unit(folder, 'd') is not None
    assert batch.completed_unit(folder, 'other') is None
    (folder/'a.tsv').write_text('xy')
    assert batch.completed_unit(folder, 'd') is None
    assert batch.plan_jobs(10, 8, 0) == (8, 1) and batch.plan_jobs(2, 8, 0) == (2, 4)
    assert batch.plan_jobs(10, 8, 3) == (3, 2) and batch.plan_jobs(1, 8, 0) == (1, 8)


def test_single_window_keeps_its_layout_and_continues_in_place(tmp_path, monkeypatch):
    bam = make_windows_bam(tmp_path/'in.bam', n_windows=1)
    def run(output, *extra):
        with redirect_stdout(io.StringIO()):
            main(['--bam', str(bam), '--region', 'chr1:1100-1700', '--cores', '1', '--output', str(output), *extra])
    run(tmp_path/'reference')
    out = tmp_path/'run'; run(out)
    # One window: the run files stay directly in the output directory, beside the run manifest and log.
    for name in ('manifest.json', 'result.json.gz', 'classes.tsv', batch.UNIT_MARKER, batch.RUN_MANIFEST, 'logs'):
        assert (out/name).exists(), name
    calls = []; real = batch.run_unit
    monkeypatch.setattr(batch, 'run_unit', lambda spec, **kw: calls.append(spec['name']) or real(spec, **kw))
    run(out, '--continue')
    assert calls == []
    with (out/'classes.tsv').open('a') as handle: handle.write('damaged\n')
    run(out, '--continue')
    assert calls == ['region_1'] and (out/batch.RUN_MANIFEST).is_file()
    assert snapshot(out) == snapshot(tmp_path/'reference')


def test_staged_engine_defaults_to_one_window_at_a_time(windows_run, tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(batch, 'run_windows', lambda *a, **kw: seen.update(kw) or ([], []))
    run, _, _ = windows_run
    run(tmp_path/'staged', '--engine', 'staged_native_families', '--cores', 4)
    assert seen['window_jobs'] == 1
    run(tmp_path/'lattice', '--cores', 4)
    assert seen['window_jobs'] == 0
