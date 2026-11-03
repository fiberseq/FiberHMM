"""Molecule-wide DAF chimera policy, QC provenance and consumer boundaries."""
import json
import subprocess
import sys

import pysam
import pytest

from fiberhmm.inference.chimeras import (
    clear_chimera_mark, is_marked_chimera, mark_chimera, prepare_chimeras,
)


@pytest.fixture(autouse=True)
def extraction_policy():
    from fiberhmm.inference import engine
    engine.configure_daf_chimera_filter()
    engine.configure_daf_snp_mask(None)
    engine.configure_daf_insert_evidence(None)
    yield
    engine.configure_daf_chimera_filter()
    engine.configure_daf_snp_mask(None)
    engine.configure_daf_insert_evidence(None)


def fixture_bam(path):
    header = pysam.AlignmentHeader.from_dict({
        'HD': {'VN': '1.6', 'SO': 'coordinate'},
        'SQ': [{'SN': 'chr1', 'LN': 10000}],
    })
    with pysam.AlignmentFile(path, 'wb', header=header) as out:
        # Non-primary records precede the diagnostic primary, and have clean
        # sequences themselves: propagation must follow QNAME, not record order.
        for name, flag, seq in [
            ('swap', 2048, 'ACGY' * 150),
            ('swap', 256, 'ACGY' * 150),
            ('swap', 0, 'ACGY' * 75 + 'ACRT' * 75),
            ('clean', 0, 'ACGY' * 150),
        ]:
            read = pysam.AlignedSegment(header)
            read.query_name = name
            read.flag = flag
            read.reference_id = 0
            read.reference_start = 100
            read.mapping_quality = 60
            read.query_sequence = seq
            read.cigarstring = '600M'
            read.set_tag('ns', [10])  # stale calls must disappear on skips
            read.set_tag('nl', [100])
            out.write(read)
    pysam.index(str(path))
    return path


def records(path):
    with pysam.AlignmentFile(path, 'rb') as bam:
        return list(bam.fetch(until_eof=True))


@pytest.mark.parametrize('mode', ['flag', 'remove', 'keep'])
def test_name_wide_policy(tmp_path, mode):
    source = fixture_bam(tmp_path / 'source.bam')
    target = tmp_path / 'out.bam'
    stats = prepare_chimeras(str(source), str(target), mode=mode)
    reads = records(target)
    swaps = [r for r in reads if r.query_name == 'swap']
    assert len(swaps) == (0 if mode == 'remove' else 3)
    assert stats['n_chimeric'] == (0 if mode == 'keep' else 1)
    assert stats['n_reads_checked'] == (0 if mode == 'keep' else 2)
    assert stats['fraction'] == (None if mode == 'keep' else 0.5)
    assert all(is_marked_chimera(r) == (mode == 'flag') for r in swaps)
    assert not is_marked_chimera(reads[-1])


def test_stale_marks_and_foreign_qcfail(tmp_path):
    source = fixture_bam(tmp_path / 'source.bam')
    stale = tmp_path / 'stale.bam'
    reads = records(source)
    mark_chimera(reads[-1])  # clean read incorrectly marked by a previous run
    reads[0].is_qcfail = True  # another tool's QC failure has no xc tag
    with pysam.AlignmentFile(stale, 'wb', header=reads[0].header) as out:
        for read in reads:
            out.write(read)
    target = tmp_path / 'out.bam'
    stats = prepare_chimeras(str(stale), str(target), mode='keep')
    current = records(target)
    assert stats['n_stale_marks_cleared'] == 1
    assert not current[-1].is_qcfail and not current[-1].has_tag('xc')
    assert current[0].is_qcfail and not current[0].has_tag('xc')
    clear_chimera_mark(current[0])
    assert current[0].is_qcfail


@pytest.mark.parametrize('mode', ['flag', 'remove', 'keep'])
@pytest.mark.parametrize('region_parallel', [False, True])
def test_call_cli_and_qc(tmp_path, mode, region_parallel):
    source = fixture_bam(tmp_path / 'source.bam')
    target = tmp_path / 'calls.bam'
    command = [sys.executable, '-m', 'fiberhmm.cli.call', '-i', str(source),
               '-o', str(target), '--enzyme', 'dddb', '-c', '1', '--io-threads', '1',
               '--no-dedup', '--no-daf-call-snps', '--daf-insert-consensus', 'off',
               '--min-read-length', '0', '--chimeras', mode, '--qc-sample-reads', '20']
    if region_parallel:
        command.append('--region-parallel')
    run = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    swaps = [r for r in records(target) if r.query_name == 'swap']
    assert len(swaps) == (0 if mode == 'remove' else 3)
    assert all(is_marked_chimera(r) == (mode == 'flag') for r in swaps)
    if mode == 'flag':
        assert all(not r.has_tag('ns') and not r.has_tag('MA') for r in swaps)
        assert 'chimera: 3' in run.stderr + run.stdout
    elif mode == 'remove':
        assert 'chimera: 3' in run.stderr + run.stdout
    payload = json.loads((tmp_path / 'qc/calls.qc.json').read_text())
    chim = payload['chimeras']
    assert chim['full_run'] is True
    assert chim['mode'] == mode
    assert chim['n_chimeric'] == (0 if mode == 'keep' else 1)
    assert chim['n_reads_checked'] == (0 if mode == 'keep' else 2)
    assert chim['min_seg'] == 5 and chim['purity'] == 0.8
    assert 'chimeras_n_chimeric' in (tmp_path / 'qc/calls.qc.tsv').read_text()
    with pysam.AlignmentFile(target, 'rb') as bam:
        ds = bam.header.to_dict()['PG'][-1]['DS']
        tokens = [item for item in ds.split() if '=' in item]
        keys = [item.split('=', 1)[0] for item in tokens]
        assert keys.count('chimeras') == keys.count('chimera_filter') == 1
        assert f'chimeras={mode}' in tokens
        assert f"chimera_filter={'off' if mode == 'keep' else 'on'}" in tokens
    # Standalone QC must recover exact pre-removal counts from the sidecar.
    if mode == 'remove':
        from fiberhmm.qc.core import run_qc
        result = run_qc(str(target), output_prefix=str(tmp_path / 'standalone'),
                        mode='daf', enzyme='dddb', sample_reads=10,
                        state_source='none', stream=None)
        assert result['chimeras']['n_chimeric'] == 1
        assert result['chimeras']['full_run']


def test_alias(monkeypatch):
    from fiberhmm.cli.call import parse_args
    monkeypatch.setattr(sys, 'argv', ['fiberhmm-call', '-i', 'a', '-o', 'b', '--keep-chimeras'])
    assert parse_args().chimeras == 'keep'


@pytest.mark.parametrize('enzyme,seq', [('hia5', 'pacbio'), ('hia5', 'nanopore'),
                                      ('dddb', 'nanopore')])
def test_call_bypasses_preparation_and_uses_original_input(tmp_path, monkeypatch,
                                                         enzyme, seq):
    from conftest import make_synthetic_bam
    from fiberhmm.cli import call
    from fiberhmm.inference import chimeras
    source = str(tmp_path / 'source.bam')
    make_synthetic_bam(source, n_reads=2, read_length=1500, n_chroms=1)
    if enzyme == 'dddb':
        fixture_bam(source)
    def forbidden(*args, **kwargs):
        pytest.fail('m6A/keep must never prepare or copy the input BAM')
    monkeypatch.setattr(chimeras, 'prepare_chimeras', forbidden)
    seen = []
    def caller(**kwargs):
        seen.append(kwargs)
        return 0, 0
    monkeypatch.setattr(call, '_process_bam_streaming_pipeline_fused', caller)
    monkeypatch.setattr(sys, 'argv', [
        'fiberhmm-call', '-i', source, '-o', str(tmp_path / 'out.bam'),
        '--enzyme', enzyme, '--seq', seq, '--force-seq', '--no-qc',
        '--no-dedup', '--no-daf-call-snps', '--daf-insert-consensus', 'off',
        '--chimeras', 'keep' if enzyme == 'dddb' else 'flag'])
    call._main(call.parse_args())
    assert len(seen) == 1 and seen[0]['input_bam'] == source
    assert seen[0]['precall_chimeric_names'] is None


@pytest.mark.parametrize('old,explicit,expected', [
    (True, None, 'flag'), (False, None, 'keep'),
    (True, 'remove', 'remove'), (False, 'flag', 'flag'),
])
def test_pipeline_modes_and_legacy_mapping(tmp_path, old, explicit, expected):
    from fiberhmm.pipeline.runner import Pipeline, PipelineConfig
    from fiberhmm.pipeline.progress import ProgressReporter
    from fiberhmm.cli.pipeline import parse_args, config_from_args
    argv = ['a.bam', '--reference', 'ref.fa', '--enzyme', 'dddb',
            '-o', str(tmp_path), '--chimera-filter' if old else '--no-chimera-filter']
    if explicit:
        argv += ['--chimeras', explicit]
    cfg = config_from_args(parse_args(argv))
    assert cfg.resolved_chimeras() == expected
    assert PipelineConfig(reads=['a'], reference='r', enzyme='dddb',
                          outdir=str(tmp_path), chimera_filter=old,
                          chimeras=explicit).resolved_chimeras() == expected
    assert 'chimeras' not in cfg.calling_settings()  # FiberBrowser 3.0.1 contract
    assert cfg.calling_settings()['chimera_filter'] == (expected != 'keep')
    p = Pipeline(cfg, ProgressReporter())
    p.aligned_bam = str(tmp_path / 'aligned.bam')
    p.called_bam = str(tmp_path / 'called.bam')
    try:
        cmd, _ = p.call_command(set())
        assert cmd[cmd.index('--chimeras') + 1] == expected
    finally:
        p.close()


def test_pipeline_qc_contains_chimeras(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from fiberhmm.pipeline.runner import Pipeline, PipelineConfig, qc_verdicts
    path = tmp_path / 'qc.json'
    chim = dict(mode='remove', n_chimeric=449, n_reads_checked=3343,
                fraction=449 / 3343, full_run=True)
    path.write_text(json.dumps({'deduplication': {'duplicate_fraction': 0.06},
                                'chimeras': chim}))
    result = qc_verdicts(str(path))
    assert result['duplicate_fraction'] == 0.06
    assert result['chimeras'] == chim
    cfg = PipelineConfig(reads=['a.bam'], reference='ref.fa', enzyme='dddb',
                         outdir=str(tmp_path), sample='test', chimeras='remove')
    p = Pipeline(cfg)
    p.reference = SimpleNamespace(fasta='ref.fa', plasmid_map=None, contigs=[])
    p.aligned_bam, p.called_bam = 'aligned.bam', 'called.bam'
    p.qc_outputs = {'json': str(path)}
    monkeypatch.setattr(p, 'open_region', lambda: None)
    monkeypatch.setattr(p, '_save_memo', lambda: None)
    outputs = p.write_outputs()
    assert outputs['qc']['verdicts']['chimeras'] == chim
    saved = json.loads((tmp_path / 'outputs.json').read_text())
    assert 'chimeras' not in saved['settings']
    assert saved['qc']['verdicts']['chimeras']['mode'] == 'remove'


def test_downstream_boundaries(tmp_path):
    from fiberhmm.cli.export_posteriors import extract_posteriors_from_read
    from fiberhmm.io.footprint_bam import load_footprint_molecules_from_bam
    source = fixture_bam(tmp_path / 'source.bam')
    target = tmp_path / 'marked.bam'
    prepare_chimeras(str(source), str(target))
    pysam.index(str(target))
    read = next(r for r in records(target) if r.query_name == 'swap' and r.flag == 512)
    stats = {}
    assert extract_posteriors_from_read(read, None, 'daf', 3, 10, stats=stats) is None
    assert stats['chimera'] == 1
    loaded = load_footprint_molecules_from_bam(str(target))
    assert loaded.diagnostics.skipped_chimera == 1


def test_non_daf_does_not_detect(tmp_path):
    source = fixture_bam(tmp_path / 'source.bam')
    target = tmp_path / 'out.bam'
    stats = prepare_chimeras(str(source), str(target), daf=False)
    assert stats['mode'] == 'n/a' and not stats['detected']
    assert stats['n_reads_checked'] == stats['n_chimeric'] == 0
    assert not any(is_marked_chimera(r) for r in records(target))


def test_decision_pass_is_read_only_and_does_not_extract(tmp_path, monkeypatch):
    from fiberhmm.inference import engine
    source = fixture_bam(tmp_path / 'source.bam')
    original = source.read_bytes()
    opened = []
    alignment_file = pysam.AlignmentFile
    def tracked(path, mode, **kwargs):
        opened.append((str(path), mode))
        return alignment_file(path, mode, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail('pass 1 must only decide chimeras, never extract call evidence')
    monkeypatch.setattr(pysam, 'AlignmentFile', tracked)
    monkeypatch.setattr(engine, '_extract_fiber_read_from_pysam', forbidden)
    names = set()
    stats = prepare_chimeras(str(source), chimeric_names=names)
    assert names == {'swap'} and stats['n_records_affected'] == 3
    assert opened == [(str(source), 'rb')]
    assert source.read_bytes() == original
    assert not list(tmp_path.glob('*.source.bam'))


def test_stdin_decisions_match_seekable_input(tmp_path):
    source = fixture_bam(tmp_path / 'source.bam')
    target = tmp_path / 'stream.bam'
    command = [sys.executable, '-m', 'fiberhmm.cli.call', '-i', '-',
               '-o', str(target), '--enzyme', 'dddb', '--seq', 'nanopore',
               '-c', '1', '--io-threads', '1', '--no-qc', '--no-dedup',
               '--no-daf-call-snps', '--daf-insert-consensus', 'off',
               '--min-read-length', '0']
    run = subprocess.run(command, input=source.read_bytes(), capture_output=True,
                         timeout=120)
    assert run.returncode == 0, run.stderr.decode()
    reads = records(target)
    assert len(reads) == 4
    assert all(is_marked_chimera(r) for r in reads[:3])
    assert all(not r.has_tag('ns') and not r.has_tag('MA') for r in reads[:3])
    summary = json.loads((tmp_path / 'qc/stream.chimeras.json').read_text())
    assert summary['statistics']['n_chimeric'] == 1
    assert not list(tmp_path.glob('.fiberhmm_chimeras_*'))


def test_caller_extracts_only_nonchimeras_once(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from fiberhmm.cli import call
    from fiberhmm.inference import engine, streaming_pipeline
    source = fixture_bam(tmp_path / 'source.bam')
    target = tmp_path / 'calls.bam'
    seen = []
    original = engine._extract_fiber_read_from_pysam
    policy = dict(engine._DAF_CHIMERA_CFG)
    def tracked(read, *args, **kwargs):
        seen.append(read.query_name)
        return original(read, *args, **kwargs)
    # Run the real worker in one thread so its extraction calls are visible
    # to the spy, preserving the actual payload/extraction/writing path.
    def executor(**kwargs):
        kwargs.pop('mp_context')
        return ThreadPoolExecutor(**kwargs)
    monkeypatch.setattr(engine, '_extract_fiber_read_from_pysam', tracked)
    monkeypatch.setattr(streaming_pipeline, 'ProcessPoolExecutor', executor)
    monkeypatch.setattr(sys, 'argv', [
        'fiberhmm-call', '-i', str(source), '-o', str(target), '--enzyme', 'dddb',
        '--seq', 'nanopore', '-c', '1', '--io-threads', '1', '--no-qc', '--no-dedup',
        '--no-daf-call-snps', '--daf-insert-consensus', 'off', '--min-read-length', '0'])
    call._main(call.parse_args())
    assert engine._DAF_CHIMERA_CFG == policy
    assert seen == ['clean']
    current = records(target)
    assert len(current) == 4 and all(is_marked_chimera(r) for r in current[:3])


@pytest.mark.parametrize('mode', ['flag', 'remove'])
def test_region_passthrough_propagates_qname_decisions(tmp_path, mode):
    source = fixture_bam(tmp_path / 'source.bam')
    original = records(source)
    header = original[0].header.to_dict()
    header['SQ'].append({'SN': 'chr2', 'LN': 10000})
    raw, mixed, target = (tmp_path / name for name in ('raw.bam', 'mixed.bam', 'out.bam'))
    with pysam.AlignmentFile(raw, 'wb', header=header) as bam:
        for read in original:
            bam.write(read)
        for name, refid, flag in [('swap', 1, 2048), ('stale', 1, 0), ('swap', -1, 4)]:
            read = pysam.AlignedSegment(bam.header)
            read.query_name, read.query_sequence = name, 'ACGY' * 150
            read.reference_id, read.reference_start = refid, 100 if refid >= 0 else -1
            read.flag, read.mapping_quality = flag, 60
            if refid >= 0:
                read.cigarstring = '600M'
            if name == 'stale':
                mark_chimera(read)
            bam.write(read)
    pysam.sort('-o', str(mixed), str(raw))
    pysam.index(str(mixed))
    command = [sys.executable, '-m', 'fiberhmm.cli.call', '-i', str(mixed),
               '-o', str(target), '--enzyme', 'dddb', '--seq', 'nanopore',
               '-c', '1', '--io-threads', '1', '--no-qc', '--no-dedup',
               '--no-daf-call-snps', '--daf-insert-consensus', 'off',
               '--min-read-length', '0', '--chimeras', mode,
               '--region-parallel', '--chroms', 'chr1']
    run = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    current = records(target)
    swaps = [r for r in current if r.query_name == 'swap']
    assert len(swaps) == (5 if mode == 'flag' else 0)
    assert all(is_marked_chimera(r) and not r.has_tag('ns') for r in swaps)
    stale = next(r for r in current if r.query_name == 'stale')
    assert not is_marked_chimera(stale) and not stale.is_qcfail


def test_flag_rerun_recomputes_and_preserves_foreign_qcfail(tmp_path):
    source = fixture_bam(tmp_path / 'source.bam')
    marked = tmp_path / 'marked.bam'
    prepare_chimeras(str(source), str(marked))
    old = records(marked)
    old[-1].is_qcfail = True
    with pysam.AlignmentFile(source, 'wb', header=old[0].header) as bam:
        for read in old:
            bam.write(read)
    target = tmp_path / 'rerun.bam'
    stats = prepare_chimeras(str(source), str(target), min_seg=1000)
    assert stats['n_stale_marks_cleared'] == 3
    assert stats['n_chimeric'] == 0
    current = records(target)
    assert not any(r.has_tag('xc') or r.is_qcfail for r in current[:-1])
    assert current[-1].is_qcfail


def test_extract_consensus_and_recall_skip(tmp_path):
    import numpy as np
    from fiberhmm.cli.extract_tags import _init_extract_worker, _extract_region_worker
    from fiberhmm.cli.recall_tfs import _make_payload, _process_payload_record
    from fiberhmm.inference.strand_rescue import load_region_evidence, hard_observations
    from fiberhmm.inference.tf_recaller import extract_modification_calls, N_CTX
    source = fixture_bam(tmp_path / 'source.bam')
    target = tmp_path / 'marked.bam'
    prepare_chimeras(str(source), str(target))
    pysam.index(str(target))
    bed = tmp_path / 'deam.bed'
    _init_extract_worker(dict(extract_types=['deam'], min_mapq=0,
                              prob_threshold=128, with_scores=False))
    _, n_reads, counts = _extract_region_worker((('chr1', 0, 1000), str(target),
                                                 {'deam': str(bed)}))
    assert n_reads == 1 and counts['chimeras_skipped'] == 1
    assert '\tswap\t' not in bed.read_text()
    read = next(r for r in records(target) if r.query_name == 'swap' and r.flag == 512)
    assert extract_modification_calls(read, 'daf') is None
    assert hard_observations(read, 'daf', 'daf', 3, 128) is None
    result, stats = _process_payload_record(_make_payload(read, 'daf'))
    assert result is None and stats['failed'] == 0
    diagnostics = {}
    loaded = load_region_evidence(str(target), 'chr1', 0, 1000,
                                  strand_mode='daf', mode='daf', context_size=3,
                                  prob_threshold=128, llr_hit=np.zeros(N_CTX),
                                  llr_miss=np.zeros(N_CTX), min_mapq=0,
                                  load_diagnostics=diagnostics)
    assert diagnostics['chimera_excluded_count'] == 3
    assert all(r.name != 'swap' for r in loaded)


def test_owned_marks_do_not_change_insert_evidence_identity(tmp_path):
    from fiberhmm.daf.insert_consensus import record_key
    from fiberhmm.inference import engine
    read = records(fixture_bam(tmp_path / 'source.bam'))[-1]
    key = record_key(read)
    engine.configure_daf_insert_evidence({key: {'evidence': 'retained'}})
    mark_chimera(read)
    assert record_key(read) == key
    assert engine.daf_insert_evidence(read) == {'evidence': 'retained'}
    clear_chimera_mark(read)
    assert engine.daf_insert_evidence(read) == {'evidence': 'retained'}
