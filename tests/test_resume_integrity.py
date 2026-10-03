"""Resumable state is bound to content and owned by one process.

Regression tests for the 3.0 release review (Codex, 29 Sep): equal-size changes with preserved metadata, corrupted
kept artifacts, concurrent owners of one work directory, string dataset paths and the environment DAF run mask in
consensus --continue.
"""
from __future__ import annotations

import hashlib
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
from fiberhmm.io import run_state
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

def _freeze_stat_key(monkeypatch, path):
    """Make run_state.stat_key report ``path``'s current key from now on, as when a rewrite lands in the same
    filesystem timestamp tick (WSL2 ext4: every stat field unchanged after an immediate same-size rewrite), and pin
    the clock to that key's newest timestamp so the file stays "just written" however slowly the test runs."""
    from fiberhmm.io import run_state
    frozen, real_stat_key, target = run_state.stat_key(path), run_state.stat_key, os.path.realpath(path)
    monkeypatch.setattr(run_state, 'stat_key', lambda p: frozen if os.path.realpath(p) == target else real_stat_key(p))
    monkeypatch.setattr(run_state, '_now_ns', lambda: max(frozen[3], frozen[4]))
    return frozen


def _age_files(monkeypatch, seconds=3600):
    """Hash as if ``seconds`` had passed since every file was written (ctime cannot be set back on disk)."""
    from fiberhmm.io import run_state
    real_now = run_state._now_ns
    monkeypatch.setattr(run_state, '_now_ns', lambda: real_now() + seconds * 10**9)


def test_digest_memo_is_not_fooled_by_preserved_metadata(tmp_path, monkeypatch):
    path = tmp_path/'ref.fa'
    path.write_text('>p\nACGTACGT\n')
    memo = DigestMemo()
    first = content_identity(path, memo)
    st = path.stat()
    path.write_text('>p\nTGCAACGT\n')                       # same size
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))     # mtime restored; ctime cannot be
    second = content_identity(path, DigestMemo(memo.to_json()))
    assert first['size'] == second['size'] and first['sha256'] != second['sha256']
    # Unchanged old files are not rehashed: the remembered digest is used as long as every stat field matches and the
    # file's timestamps were safely older than the hash.
    _age_files(monkeypatch)
    key = stat_key(path.resolve())
    old = max(key[3], key[4]) + run_state.RACY_MARGIN_NS
    entries = {str(path.resolve()): {'stat': key, 'sha256': 'remembered', 'hashed_at_ns': old}}
    assert DigestMemo(entries).sha256(path) == 'remembered'
    # A trusted entry of an old file is dropped when only the ctime moved (the metadata-preserving rewrite above).
    earlier = key[:4] + [key[4] - 10**9]
    moved = {str(path.resolve()): {'stat': earlier, 'sha256': 'remembered', 'hashed_at_ns': old}}
    assert DigestMemo(moved).sha256(path) == second['sha256']
    # ... but not when it was hashed within the margin of its last change, nor when it predates the rule.
    entries[str(path.resolve())]['hashed_at_ns'] = old - 1
    assert DigestMemo(entries).sha256(path) == second['sha256']
    del entries[str(path.resolve())]['hashed_at_ns']
    assert DigestMemo(entries).sha256(path) == second['sha256']


def test_digest_memo_rehashes_a_rewrite_with_identical_stat(tmp_path, monkeypatch):
    """A same-size rewrite in the same timestamp tick leaves dev/inode/size/mtime/ctime identical; the digest of the
    just-written file was racily clean, so it is never trusted (deterministic version of the WSL2 failure)."""
    path = tmp_path/'table.json'
    path.write_bytes(b'AAAA')
    _freeze_stat_key(monkeypatch, path)
    memo = DigestMemo()
    assert memo.sha256(path) == hashlib.sha256(b'AAAA').hexdigest()
    assert memo.to_json() == {}                              # just written: not remembered
    path.write_bytes(b'TTTT')
    assert memo.sha256(path) == hashlib.sha256(b'TTTT').hexdigest()
    assert DigestMemo(memo.to_json()).sha256(path) == hashlib.sha256(b'TTTT').hexdigest()


def test_digest_memo_serves_old_files_without_rehashing(tmp_path, monkeypatch):
    path = tmp_path/'model.json'
    path.write_bytes(b'{"old": true}')
    calls = []
    real_sha = run_state.sha256_file
    monkeypatch.setattr(run_state, 'sha256_file', lambda p: calls.append(p) or real_sha(p))
    # Just written: hashed every time, never remembered.
    memo = DigestMemo()
    memo.sha256(path); memo.sha256(path)
    assert len(calls) == 2 and memo.to_json() == {}
    # Old (its timestamps safely precede the hash): hashed once, then served from the memo, also after a round trip.
    _age_files(monkeypatch)
    digest = memo.sha256(path)
    assert len(calls) == 3 and str(path.resolve()) in memo.to_json()
    assert memo.sha256(path) == digest and DigestMemo(memo.to_json()).sha256(path) == digest
    assert len(calls) == 3


def test_digest_memo_entry_stays_untrusted_when_hashing_outlasts_the_margin(tmp_path, monkeypatch):
    """The age test uses the time hashing *started*: a long read of a just-written file does not make it trusted."""
    path = tmp_path/'big.bin'
    path.write_bytes(b'x' * 64)
    key = stat_key(path)
    fresh = max(key[3], key[4])
    clock = iter([fresh, fresh + 10 * run_state.RACY_MARGIN_NS])
    monkeypatch.setattr(run_state, '_now_ns', lambda: next(clock))
    memo = DigestMemo()
    memo.sha256(path)
    assert memo.to_json() == {}


def test_digest_memo_future_timestamps_and_clock_rollback(tmp_path, monkeypatch):
    path = tmp_path/'skewed.bin'
    path.write_bytes(b'abc')
    key = stat_key(path)
    newest = max(key[3], key[4])
    # A file stamped in the future (relative to this clock) is never remembered.
    monkeypatch.setattr(run_state, '_now_ns', lambda: newest - 3600 * 10**9)
    memo = DigestMemo()
    memo.sha256(path)
    assert memo.to_json() == {}
    # A remembered entry is not returned once the clock has been set back to within the margin of the file's
    # timestamps: a new write could reproduce them.
    entries = {str(path.resolve()): {'stat': key, 'sha256': 'remembered', 'hashed_at_ns': newest + 3600 * 10**9}}
    monkeypatch.setattr(run_state, '_now_ns', lambda: newest + 3600 * 10**9)
    assert DigestMemo(entries).sha256(path) == 'remembered'
    monkeypatch.setattr(run_state, '_now_ns', lambda: newest)
    assert DigestMemo(entries).sha256(path) == hashlib.sha256(b'abc').hexdigest()


def test_digest_memo_rereads_a_file_that_changes_while_hashed(tmp_path, monkeypatch):
    path = tmp_path/'moving.bin'
    path.write_bytes(b'AAAA')
    _age_files(monkeypatch)
    real_sha, calls = run_state.sha256_file, []

    def racing_sha(p):
        digest = real_sha(p)
        calls.append(digest)
        if len(calls) == 1:  # a writer replaces the file while the first read is under way
            os.replace(_written(tmp_path/'next.bin', b'TTTTTT'), path)
        return digest
    monkeypatch.setattr(run_state, 'sha256_file', racing_sha)
    memo = DigestMemo()
    assert memo.sha256(path) == hashlib.sha256(b'TTTTTT').hexdigest() and len(calls) == 2


def _written(path, data):
    path.write_bytes(data)
    return path


def test_digest_is_trusted_requires_both_timestamps_to_be_old():
    margin = run_state.RACY_MARGIN_NS
    key = [1, 2, 3, 10**18, 10**18 + 5]
    assert run_state.digest_is_trusted(key, 10**18 + 5 + margin)
    assert not run_state.digest_is_trusted(key, 10**18 + 4 + margin)           # ctime too fresh
    assert not run_state.digest_is_trusted([1, 2, 3, 10**18 + 9, 10**18], 10**18 + 8 + margin)  # mtime too fresh
    assert not run_state.digest_is_trusted(key, None) and not run_state.digest_is_trusted(key, 'x')


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


def test_reference_digest_cache_rehashes_a_rewrite_with_identical_stat(tmp_path, monkeypatch):
    from fiberhmm.pipeline.reference import cached_fasta_digests, file_sha256, scan_fasta
    fasta = tmp_path/'cached.fa'; cache = tmp_path/'cache'
    fasta.write_text('>p\nACGTACGT\n')
    _freeze_stat_key(monkeypatch, fasta)
    first = cached_fasta_digests(str(fasta), str(cache))
    assert json.loads((cache/'reference_digests.json').read_text()) == {}   # just written: not remembered
    fasta.write_text('>p\nTGCAACGT\n')
    second = cached_fasta_digests(str(fasta), str(cache))
    assert second != first and second == (file_sha256(str(fasta)), scan_fasta(str(fasta)))


def test_reference_digest_cache_serves_old_files_without_rescanning(tmp_path, monkeypatch):
    from fiberhmm.pipeline import reference
    fasta = tmp_path/'old.fa'; cache = tmp_path/'cache'
    fasta.write_text('>p\nACGTACGT\n')
    calls = []
    real_scan = reference.scan_fasta
    monkeypatch.setattr(reference, 'scan_fasta', lambda p: calls.append(p) or real_scan(p))
    hashes = []
    real_sha = reference.file_sha256
    monkeypatch.setattr(reference, 'file_sha256', lambda p: hashes.append(p) or real_sha(p))
    _age_files(monkeypatch)
    first = reference.cached_fasta_digests(str(fasta), str(cache))
    assert reference.cached_fasta_digests(str(fasta), str(cache)) == first and len(calls) == len(hashes) == 1
    # An entry written before the racily-clean rule (no hash time) is not trusted, and is replaced.
    memo_path = cache/'reference_digests.json'
    memo = json.loads(memo_path.read_text())
    for entry in memo.values():
        del entry['hashed_at_ns']; entry['sha256'] = 'legacy'
    memo_path.write_text(json.dumps(memo))
    assert reference.cached_fasta_digests(str(fasta), str(cache)) == first and len(calls) == len(hashes) == 2
    assert all(isinstance(e.get('hashed_at_ns'), int) for e in json.loads(memo_path.read_text()).values())


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


def test_reference_index_is_rebuilt_when_its_contigs_disagree(tmp_path):
    """A .fai newer than its FASTA is reused only while it lists the FASTA's contigs (a same-tick or restored-mtime
    rewrite that renames or resizes a contig must not keep the old index)."""
    from fiberhmm.pipeline import reference
    fasta = tmp_path/'big.fa'
    fasta.write_text('>chrA\nACGTACGT\n>chrB\nACGT\n')
    fai = tmp_path/'big.fa.fai'
    fai.write_text('chrA\t8\t6\t8\t9\nchrZ\t4\t21\t4\t5\n')
    st = fasta.stat()
    os.utime(fai, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    records = reference.scan_fasta(str(fasta))
    reference._ensure_fai(str(fasta), records)
    assert reference._fai_contigs(str(fai)) == [('chrA', 8), ('chrB', 4)]
    stamped = fai.stat().st_mtime_ns
    reference._ensure_fai(str(fasta), records)                 # now consistent: reused
    assert fai.stat().st_mtime_ns == stamped


def test_targeted_strand_rescue_script_source_digests_follow_the_memo_rule(tmp_path, monkeypatch):
    import importlib.util
    script = Path(__file__).resolve().parents[1]/'scripts'/'run_targeted_strand_rescue.py'
    spec = importlib.util.spec_from_file_location('_targeted_sr_script', script)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    source = tmp_path/'source.py'
    source.write_bytes(b'x = 1\n')
    _freeze_stat_key(monkeypatch, source)
    first = module._source_fingerprint([source])[0]['sha256']
    source.write_bytes(b'x = 2\n')                             # same size, same (frozen) stat
    second = module._source_fingerprint([source])[0]['sha256']
    assert first != second == hashlib.sha256(b'x = 2\n').hexdigest()
