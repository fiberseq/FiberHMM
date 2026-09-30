"""Resumable state is bound to content and owned by one process.

Regression tests for the 3.0 release review (Codex, 29 Sep): equal-size changes with preserved metadata, corrupted
kept artifacts, concurrent owners of one work directory, string dataset paths and the environment DAF run mask in
consensus --continue.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import pysam
import pytest

from fiberhmm.inference import region_resume
from fiberhmm.inference.consensus import batch
from fiberhmm.io.run_state import DigestMemo, DirectoryBusy, DirectoryLock, content_identity, stat_key

sys.path.insert(0, str(Path(__file__).parent))
from test_call_entrypoint_regressions import REPO_ROOT, make_region_test_bam  # noqa: E402
from test_call_resume import call, records, same_size_variant, work_dir  # noqa: E402
from test_consensus_continue import make_windows_bam  # noqa: E402


def _write(path, header, rows):
    with pysam.AlignmentFile(str(path), 'wb', header=header) as out:
        for row in rows:
            out.write(row)
    pysam.index(str(path))


def _record(header, seq, pos=100):
    r = pysam.AlignedSegment(header)
    r.query_name = 'origin'; r.reference_id = 0; r.reference_start = pos; r.mapping_quality = 60
    r.query_sequence = seq; r.cigarstring = f'{len(seq)}M'
    return r


# ---------------------------------------------------------------------------
# Content identity (the digest-reuse rule)
# ---------------------------------------------------------------------------

def test_digest_memo_is_not_fooled_by_preserved_metadata(tmp_path):
    path = tmp_path/'ref.fa'
    path.write_text('>p\nACGTACGT\n')
    memo = DigestMemo()
    first = content_identity(path, memo)
    st = path.stat()
    path.write_text('>p\nTGCAACGT\n')                       # same size
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))     # mtime restored; ctime cannot be
    second = content_identity(path, DigestMemo(memo.to_json()))
    assert first['size'] == second['size'] and first['sha256'] != second['sha256']
    # Unchanged files are not rehashed: the remembered digest is used as long as every stat field matches.
    entries = memo.to_json()
    entries[str(path.resolve())] = {'stat': stat_key(path.resolve()), 'sha256': 'remembered'}
    assert DigestMemo(entries).sha256(path) == 'remembered'


def test_call_input_identity_binds_record_content(tmp_path):
    h = pysam.AlignmentHeader.from_dict({'HD': {'SO': 'coordinate'}, 'SQ': [{'SN': 'p', 'LN': 2000}]})
    bam = tmp_path/'identity.bam'
    _write(bam, h, [_record(h, 'A'*100)])
    before = region_resume.input_identity(str(bam))
    consensus_before = batch.file_identity(bam)
    st = bam.stat()
    candidate = tmp_path/'identity2.bam'
    _write(candidate, h, [_record(h, 'T'*100)])
    assert candidate.stat().st_size == st.st_size
    bam.write_bytes(candidate.read_bytes())
    os.utime(bam, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert region_resume.input_identity(str(bam)) != before
    assert batch.file_identity(bam) != consensus_before


def test_reference_digest_cache_rehashes_changed_content(tmp_path):
    from fiberhmm.pipeline.reference import cached_fasta_digests, file_sha256, scan_fasta
    fasta = tmp_path/'cached.fa'; cache = tmp_path/'cache'
    fasta.write_text('>p\nACGTACGT\n')
    first = cached_fasta_digests(str(fasta), str(cache))
    st = fasta.stat()
    fasta.write_text('>p\nTGCAACGT\n')
    os.utime(fasta, ns=(st.st_atime_ns, st.st_mtime_ns))
    second = cached_fasta_digests(str(fasta), str(cache))
    assert second != first
    assert second[0] == file_sha256(str(fasta)) and second[1] == scan_fasta(str(fasta))


# ---------------------------------------------------------------------------
# Kept artifacts are verified by content
# ---------------------------------------------------------------------------

def test_equal_size_change_to_a_kept_region_bam_is_rerun(tmp_path, monkeypatch, benchmark_model_path):
    bam = make_region_test_bam(tmp_path/'in.bam', n_reads=8)
    reference = tmp_path/'reference.bam'
    assert call(monkeypatch, bam, reference, benchmark_model_path) == 0
    output = tmp_path/'run.bam'
    assert call(monkeypatch, bam, output, benchmark_model_path, '--keep-work-dir') == 0
    work = work_dir(output)
    altered = None
    for marker in sorted(work.glob('region_*.done.json')):
        region_bam = work/json.loads(marker.read_text())['bam']
        with pysam.AlignmentFile(str(region_bam), check_sq=False) as handle:
            if any(True for _ in handle.fetch(until_eof=True)):
                size = region_bam.stat().st_size
                altered = same_size_variant(region_bam)
                assert region_bam.stat().st_size == size
                break
    assert altered is not None
    assert call(monkeypatch, bam, output, benchmark_model_path, '--resume') == 0
    assert records(output) == records(reference)


# ---------------------------------------------------------------------------
# One owner per work directory
# ---------------------------------------------------------------------------

def test_work_dir_has_one_owner(tmp_path):
    path = tmp_path/'shared_work'
    one = region_resume.WorkDir(path, {'input': 'x'}).open()
    with pytest.raises(region_resume.ResumeRefused, match='in use by another running process'):
        region_resume.WorkDir(path, {'input': 'x'}, resume=True).open()
    one.release()
    two = region_resume.WorkDir(path, {'input': 'x'}, resume=True).open()   # a finished owner frees it
    two.remove()
    assert not path.exists()


def _claim(root, identity, barrier, queue):
    real = region_resume._atomic_json

    def gated(path, value):
        barrier.wait(timeout=5)
        return real(path, value)
    region_resume._atomic_json = gated
    try:
        # The owner holds the lock while its manifest write waits for the other
        # process, which can only get there by being refused.
        work = region_resume.WorkDir(root, {'parameter': identity}).open()
        queue.put(('owner', identity))
        work.release()
    except region_resume.ResumeRefused:
        queue.put(('refused', identity))
        barrier.wait(timeout=5)
    except Exception as exc:  # pragma: no cover - reported below
        queue.put(('error', repr(exc)))


@pytest.mark.skipif(sys.platform.startswith('win'), reason='fork')
def test_two_processes_cannot_both_claim_a_work_dir(tmp_path):
    ctx = mp.get_context('fork')
    barrier = ctx.Barrier(2); queue = ctx.Queue()
    root = tmp_path/'race'
    procs = [ctx.Process(target=_claim, args=(root, i, barrier, queue)) for i in range(2)]
    for p in procs: p.start()
    for p in procs: p.join(20)
    outcomes = sorted(queue.get(timeout=5)[0] for _ in procs)
    assert outcomes == ['owner', 'refused'] and queue.empty()
    assert [p.exitcode for p in procs] == [0, 0]
    stored = json.loads((root/'manifest.json').read_text())['identity']
    assert stored in ({'parameter': 0}, {'parameter': 1})


def test_lock_of_a_dead_owner_is_free(tmp_path):
    lock = tmp_path/'d'/'.lock'
    code = ('import os,sys; from fiberhmm.io.run_state import DirectoryLock;'
            f'DirectoryLock({str(lock)!r}).acquire(); os._exit(0)')   # dies holding it, without cleanup
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT)+os.pathsep+os.environ.get('PYTHONPATH', ''))
    assert subprocess.run([sys.executable, '-c', code], env=env).returncode == 0
    with DirectoryLock(lock):
        with pytest.raises(DirectoryBusy):
            DirectoryLock(lock).acquire()


# ---------------------------------------------------------------------------
# fiberhmm-consensus --continue
# ---------------------------------------------------------------------------

def test_consensus_string_dataset_paths_are_normalised(tmp_path, monkeypatch):
    bam = make_windows_bam(tmp_path/'split.bam', n_windows=1)
    absolute = batch.run_record([{'dataset_id': 'd', 'paths': str(bam), 'chemistry': 'ddda'}], [], {}, {})
    assert [p['path'] for p in absolute['datasets'][0]['paths']] == [str(bam.resolve())]
    monkeypatch.chdir(tmp_path)
    relative = batch.run_record([{'dataset_id': 'd', 'paths': 'split.bam', 'chemistry': 'ddda'}], [], {}, {})
    assert relative == absolute
    st = bam.stat(); os.utime(bam, ns=(st.st_atime_ns, st.st_mtime_ns+10**9))
    assert batch.run_record([{'dataset_id': 'd', 'paths': 'split.bam', 'chemistry': 'ddda'}], [], {}, {}) != relative


def _consensus(bam, bed, out, mask, *extra):
    env = dict(os.environ, FIBERHMM_DAF_RUN_POLICY='keep-one',
               PYTHONPATH=str(REPO_ROOT)+os.pathsep+os.environ.get('PYTHONPATH', ''))
    env.pop('FIBERHMM_DAF_RUN_MASK', None)
    if mask is not None:
        env['FIBERHMM_DAF_RUN_MASK'] = str(mask)
    return subprocess.run([sys.executable, '-m', 'fiberhmm.inference.consensus.cli', '--bam', str(bam), '--bed',
                           str(bed), '--output', str(out), '--cores', '1', '--window-jobs', '1', '--no-bam', *extra],
                          env=env, capture_output=True, text=True, timeout=300)


def test_consensus_continue_records_the_environment_daf_mask(tmp_path):
    bam = make_windows_bam(tmp_path/'in.bam', n_windows=2)
    bed = tmp_path/'w.bed'
    bed.write_text(''.join(f'chr1\t{1100+2000*w}\t{1700+2000*w}\tw{w+1}\n' for w in range(2)))
    out = tmp_path/'run'
    assert _consensus(bam, bed, out, 0).returncode == 0
    assert json.loads((out/batch.RUN_MANIFEST).read_text())['options']['daf_mask'] == [0, 'keep-one']
    changed = _consensus(bam, bed, out, 2, '--continue')
    assert changed.returncode == 2 and 'options.daf_mask' in changed.stderr
    unset = _consensus(bam, bed, out, None, '--continue')       # back to the chemistry default: also different
    assert unset.returncode == 2 and 'options.daf_mask' in unset.stderr
    same = _consensus(bam, bed, out, 0, '--continue')
    assert same.returncode == 0, same.stderr[-2000:]
