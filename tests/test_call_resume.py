"""fiberhmm-call --region-parallel: resumable work directory, --resume, progress JSON and signals.

An interrupted region-parallel run leaves a work directory of finished regions; --resume reuses them, reruns
missing/partial regions and publishes records identical to an uninterrupted run. Changed inputs or parameters are
refused; nothing is ever half-published.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pysam
import pytest

from fiberhmm.inference import region_resume

sys.path.insert(0, str(Path(__file__).parent))
from test_call_entrypoint_regressions import REPO_ROOT, make_region_test_bam  # noqa: E402

COMMON = ('--min-read-length', '0', '--prob-threshold', '0', '--no-qc', '--no-recall-nucs',
          '--region-size', '2000', '-c', '2', '--io-threads', '1')


def call(monkeypatch, bam, output, model, *extra, region_parallel=True):
    """Run fiberhmm-call in-process; returns the exit code (0 on success)."""
    from fiberhmm.cli import call as cli
    argv = ['fiberhmm-call', '-i', str(bam), '-o', str(output), '-m', model, *COMMON, *map(str, extra)]
    if region_parallel: argv.append('--region-parallel')
    monkeypatch.setattr(sys, 'argv', argv)
    try:
        cli.main()
    except SystemExit as exit:
        return exit.code or 0
    return 0


def records(path):
    with pysam.AlignmentFile(str(path), 'rb', check_sq=False) as handle:
        return [r.to_string() for r in handle.fetch(until_eof=True)]


def events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def work_dir(output):
    return region_resume.default_work_dir(output)


def interrupt_after(monkeypatch, count):
    real = region_resume.WorkDir.mark_done
    seen = []

    def mark_done(self, index, item, result):
        if len(seen) == count: raise KeyboardInterrupt
        seen.append(index)
        return real(self, index, item, result)
    monkeypatch.setattr(region_resume.WorkDir, 'mark_done', mark_done)
    return seen


@pytest.fixture
def region_bam(tmp_path):
    return make_region_test_bam(tmp_path/'in.bam')


def test_interrupted_run_resumes_to_identical_records(tmp_path, monkeypatch, region_bam, benchmark_model_path):
    reference = tmp_path/'reference.bam'
    assert call(monkeypatch, region_bam, reference, benchmark_model_path) == 0
    assert not work_dir(reference).exists()          # removed after a successful publish

    output = tmp_path/'out'/'run.bam'
    finished = interrupt_after(monkeypatch, 5)
    assert call(monkeypatch, region_bam, output, benchmark_model_path) == 130
    work = work_dir(output)
    assert not output.exists() and not list(output.parent.glob('*.tmp'))   # nothing half-published
    markers = sorted(work.glob('region_*.done.json'))
    assert len(markers) == len(finished) == 5
    # A region killed mid-write: its BAM exists without a marker, so it is rerun.
    unfinished = next(p for p in sorted(work.glob('region_*.bam'))
                      if not (work/(p.stem+'.done.json')).exists())
    unfinished.write_bytes(b'partial')
    # A stale marker whose BAM changed size is not trusted either.
    damaged = work/json.loads(markers[0].read_text())['bam']
    damaged.write_bytes(damaged.read_bytes()+b'x')

    monkeypatch.undo()
    progress = tmp_path/'progress.jsonl'
    assert call(monkeypatch, region_bam, output, benchmark_model_path, '--resume', '--progress-json', progress) == 0
    lines = events(progress)
    assert lines[0]['event'] == 'start' and lines[0]['regions_reused'] == 4
    assert lines[-1]['event'] == 'done' and lines[-1]['regions_reused'] == 4
    region_lines = [e for e in lines if e['event'] == 'region']
    assert len(region_lines) == lines[0]['regions_total']-4
    assert {'regions_done', 'regions_total', 'reads', 'reads_per_s', 'eta_s', 'elapsed_s'} <= set(region_lines[-1])
    assert records(output) == records(reference)
    assert not work.exists()
    with pysam.AlignmentFile(str(output), 'rb') as handle:
        pg = [p for p in handle.header.to_dict()['PG'] if p.get('PN') == 'fiberhmm-call'][-1]
    assert '--resume' not in pg['CL']                    # the header records the original command line


def test_resume_refuses_changed_parameters_and_input(tmp_path, monkeypatch, region_bam, benchmark_model_path, capsys):
    output = tmp_path/'run.bam'
    interrupt_after(monkeypatch, 3)
    assert call(monkeypatch, region_bam, output, benchmark_model_path) == 130
    monkeypatch.undo(); capsys.readouterr()
    # Without --resume an existing work directory is never silently discarded or mixed.
    assert call(monkeypatch, region_bam, output, benchmark_model_path) == 2
    assert '--resume' in capsys.readouterr().err
    assert call(monkeypatch, region_bam, output, benchmark_model_path, '--resume', '--min-mapq', '5') == 2
    assert 'region_pipeline.min_mapq' in capsys.readouterr().err
    # Identity is content, not metadata: touching the input is accepted by the digest...
    stat = os.stat(region_bam)
    os.utime(region_bam, ns=(stat.st_atime_ns, stat.st_mtime_ns+10**9))
    # ...but different records of the same size, with size and mtime preserved, are refused.
    same_size_variant(region_bam)
    assert call(monkeypatch, region_bam, output, benchmark_model_path, '--resume') == 2
    assert 'input.sha256' in capsys.readouterr().err
    assert not output.exists() and len(list(work_dir(output).glob('region_*.done.json'))) == 3


def same_size_variant(path):
    """Rewrite ``path`` with one QNAME character changed, same byte size, mtime restored (like cp -p)."""
    path = Path(path)
    stat = path.stat()
    with pysam.AlignmentFile(str(path), 'rb', check_sq=False) as handle:
        header = handle.header
        rows = list(handle.fetch(until_eof=True))
    candidate = path.with_name(path.name+'.variant')
    for index, row in enumerate(rows):
        original = row.query_name
        for position in range(len(original)):
            for char in 'abcdefghijklmnopqrstuvwxyz0123456789':
                if char == original[position]:
                    continue
                row.query_name = original[:position]+char+original[position+1:]
                with pysam.AlignmentFile(str(candidate), 'wb', header=header) as out:
                    for item in rows:
                        out.write(item)
                if candidate.stat().st_size == stat.st_size:
                    path.write_bytes(candidate.read_bytes())
                    candidate.unlink()
                    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                    return row.query_name
        row.query_name = original
    raise AssertionError('no same-size variant found')


def test_resume_mode_rules(tmp_path, monkeypatch, region_bam, benchmark_model_path, capsys):
    unindexed = make_region_test_bam(tmp_path/'unsorted.bam', sort=False)
    assert call(monkeypatch, unindexed, tmp_path/'a.bam', benchmark_model_path, '--resume', region_parallel=False) == 2
    assert 'streaming' in capsys.readouterr().err
    assert call(monkeypatch, region_bam, tmp_path/'b.bam', benchmark_model_path, '--work-dir', tmp_path/'w',
                region_parallel=False) == 2
    # --resume implies --region-parallel for an indexed input; with no work directory it simply starts.
    kept = tmp_path/'kept.bam'
    assert call(monkeypatch, region_bam, kept, benchmark_model_path, '--resume', '--keep-work-dir',
                region_parallel=False) == 0
    assert 'implies --region-parallel' in capsys.readouterr().err
    assert work_dir(kept).is_dir()
    first = records(kept)
    # Resuming a finished (kept) work directory reuses every region and republishes the same records.
    progress = tmp_path/'p.jsonl'
    assert call(monkeypatch, region_bam, kept, benchmark_model_path, '--resume', '--progress-json', progress) == 0
    start = events(progress)[0]
    assert start['regions_reused'] == start['regions_total'] and records(kept) == first
    assert not work_dir(kept).exists()


def test_sigterm_leaves_resumable_work_dir(tmp_path, benchmark_model_path):
    bam = make_region_test_bam(tmp_path/'in.bam', n_reads=160)
    output = tmp_path/'run.bam'; progress = tmp_path/'p.jsonl'
    env = dict(os.environ, FIBERHMM_NO_UPDATE_CHECK='1',
               PYTHONPATH=str(REPO_ROOT)+os.pathsep+os.environ.get('PYTHONPATH', ''))
    argv = [sys.executable, '-m', 'fiberhmm.cli.call', '-i', bam, '-o', str(output), '-m', benchmark_model_path,
            *COMMON[:-4], '-c', '1', '--io-threads', '1', '--region-parallel', '--progress-json', str(progress)]
    process = subprocess.Popen(argv, cwd=REPO_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    deadline = time.time()+120
    while time.time() < deadline and process.poll() is None:
        try:
            if progress.exists() and any(e['event'] == 'region' for e in events(progress)): break
        except ValueError:            # a line still being written
            pass
        time.sleep(.02)
    if process.poll() is not None:
        pytest.skip('run finished before it could be interrupted')
    process.send_signal(signal.SIGTERM)
    _, stderr = process.communicate(timeout=60)
    if process.returncode == 0:
        pytest.skip('run finished before the signal arrived')
    assert process.returncode == 128+signal.SIGTERM, stderr.decode(errors='replace')
    assert b'--resume' in stderr, stderr.decode(errors='replace')[-1500:]
    assert not output.exists() and not list(tmp_path.glob('*.tmp*'))
    assert events(progress)[-1]['event'] == 'stopped'
    assert any(work_dir(output).glob('region_*.done.json'))
    resumed = subprocess.run(argv+['--resume'], cwd=REPO_ROOT, env=env, capture_output=True, timeout=300)
    assert resumed.returncode == 0, resumed.stderr.decode(errors='replace')
    reference = tmp_path/'reference.bam'
    fresh = [reference if a == str(output) else a for a in argv]
    assert subprocess.run(fresh, cwd=REPO_ROOT, env=env, capture_output=True, timeout=300).returncode == 0
    assert records(output) == records(reference)
