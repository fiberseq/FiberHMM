"""Unavailable QNAMEs and stale chimera report provenance."""
import json
import subprocess
import sys

import pysam
import pytest

from test_chimera_flags import extraction_policy, fixture_bam, records
from fiberhmm.inference.chimeras import is_marked_chimera, mark_chimera, prepare_chimeras


def unnamed_bam(path):
    source = fixture_bam(path)
    reads = records(source)[-2:]  # diagnostic primary followed by a clean read
    for read in reads:
        read.query_name = '*'
    with pysam.AlignmentFile(source, 'wb', header=reads[0].header) as out:
        # Repeat identical unnamed records too: each must count separately.
        for read in reads * 2:
            out.write(read)
    pysam.index(str(source))
    return source


@pytest.mark.parametrize('mode', ['flag', 'remove', 'keep'])
def test_unnamed_records_are_independent(tmp_path, mode):
    source = unnamed_bam(tmp_path / 'source.bam')
    target = tmp_path / 'out.bam'
    stats = prepare_chimeras(str(source), str(target), mode=mode)
    assert stats['n_reads_checked'] == (0 if mode == 'keep' else 4)
    assert stats['n_chimeric'] == (0 if mode == 'keep' else 2)
    assert stats['n_records_affected'] == (0 if mode == 'keep' else 2)
    assert stats['fraction'] == (None if mode == 'keep' else 0.5)
    current = records(target)
    assert len(current) == (2 if mode == 'remove' else 4)
    assert [is_marked_chimera(r) for r in current] == (
        [True, False, True, False] if mode == 'flag' else [False] * len(current))


@pytest.mark.parametrize('region_parallel', [False, True])
@pytest.mark.parametrize('mode', ['flag', 'remove', 'keep'])
def test_cli_unnamed_decisions(tmp_path, region_parallel, mode):
    source = unnamed_bam(tmp_path / 'source.bam')
    target = tmp_path / 'out.bam'
    cmd = [sys.executable, '-m', 'fiberhmm.cli.call', '-i', str(source),
           '-o', str(target), '--enzyme', 'dddb', '-c', '1', '--io-threads', '1',
           '--min-read-length', '0', '--no-dedup', '--no-qc', '--no-daf-call-snps',
           '--daf-insert-consensus', 'off', '--chimeras', mode]
    if region_parallel:
        cmd.append('--region-parallel')
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    current = records(target)
    assert len(current) == (2 if mode == 'remove' else 4)
    assert [is_marked_chimera(r) for r in current] == (
        [True, False, True, False] if mode == 'flag' else [False] * len(current))
    for read in current:
        assert read.has_tag('MA') == (not is_marked_chimera(read))
    stats = json.loads((tmp_path / 'qc/out.chimeras.json').read_text())['statistics']
    assert stats['n_reads_checked'] == (0 if mode == 'keep' else 4)
    assert stats['n_chimeric'] == (0 if mode == 'keep' else 2)


def test_qc_sample_counts_unnamed_records(tmp_path):
    from fiberhmm.qc.core import analyze_sample
    reads = records(unnamed_bam(tmp_path / 'source.bam'))
    mark_chimera(reads[0])
    mark_chimera(reads[2])
    analysis, _ = analyze_sample(reads, 'daf', None)
    assert analysis['chimeras']['n_reads_checked'] == 4
    assert analysis['chimeras']['n_chimeric'] == 2
    assert analysis['chimeras']['fraction'] == 0.5


@pytest.mark.parametrize('mismatch', ['none', 'assay', 'enzyme', 'program_id',
                                      'command', 'run_id', 'legacy', 'non_daf'])
def test_qc_requires_matching_chimera_report(tmp_path, monkeypatch, mismatch):
    from fiberhmm.qc import core
    from fiberhmm.io.run_state import sha256_file
    source = fixture_bam(tmp_path / 'calls.bam')
    reads = records(source)[-1:]
    pg = {'ID': 'fiberhmm-call.2', 'PN': 'fiberhmm-call', 'CL': 'call --chimeras flag',
          'DS': 'mode=daf enzyme=dddb chimeras=flag run_id=current'}
    header = reads[0].header.to_dict()
    header['PG'] = [pg]
    if mismatch == 'non_daf':
        header['PG'] = [dict(pg, DS='mode=pacbio-fiber enzyme=hia5 run_id=current')]
    with pysam.AlignmentFile(source, 'wb', header=header) as out:
        out.write(reads[0])
    pysam.index(str(source))
    payload = {'schema_version': 1, 'assay': {'mode': 'daf', 'enzyme': 'dddb'},
               'output_sha256': sha256_file(source), 'statistics': {
                   'detected': True, 'mode': 'remove', 'full_run': True,
                   'n_reads_checked': 100, 'n_chimeric': 25, 'fraction': 0.25}}
    if mismatch == 'assay':
        payload['assay']['mode'] = 'pacbio-fiber'
    elif mismatch == 'enzyme':
        payload['assay']['enzyme'] = 'ddda'
    elif mismatch == 'program_id':
        pg['ID'] = 'fiberhmm-call'
    elif mismatch == 'command':
        pg['CL'] = 'call --chimeras remove'
    elif mismatch == 'run_id':
        pg['DS'] = pg['DS'].replace('current', 'previous')
    elif mismatch == 'legacy':
        payload.pop('assay')
        payload.pop('output_sha256')
    if mismatch in ('program_id', 'command', 'run_id'):
        with pysam.AlignmentFile(source, 'wb', header=header) as out:
            out.write(reads[0])
        pysam.index(str(source))
    report = tmp_path / 'qc/calls.chimeras.json'
    report.parent.mkdir()
    report.write_text(json.dumps(payload))
    monkeypatch.setattr(core, '_plot_qc', lambda *a, **k: False)
    result = core.run_qc(str(source), output_prefix=str(tmp_path / 'qc/result'),
                         sample_reads=10, state_source='none', stream=None)
    assert result['assay']['mode'] == ('pacbio-fiber' if mismatch == 'non_daf' else 'daf')
    chim = result['chimeras']
    assert chim['full_run'] == (mismatch == 'none')
    assert chim['n_chimeric'] == (25 if mismatch == 'none' else 0)
    assert ('full_run_statistics' in chim) == (mismatch == 'none')
    assert ('run_report_path' in chim) == (mismatch == 'none')


def test_non_daf_call_removes_old_chimera_report(tmp_path):
    source = fixture_bam(tmp_path / 'source.bam')
    target = tmp_path / 'out.bam'
    report = tmp_path / 'qc/out.chimeras.json'
    report.parent.mkdir()
    report.write_text('{"schema_version": 1, "statistics": {"mode": "remove"}}')
    cmd = [sys.executable, '-m', 'fiberhmm.cli.call', '-i', str(source),
           '-o', str(target), '--enzyme', 'hia5', '--seq', 'pacbio', '--force-seq',
           '-c', '1', '--io-threads', '1', '--min-read-length', '0', '--no-qc']
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    assert target.is_file()
    assert not report.exists()
