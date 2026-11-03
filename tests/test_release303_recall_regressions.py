"""Second release audit: current decisions override every stale chimera mark."""
import csv
import io
import json
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pysam
import pytest

from test_chimera_flags import extraction_policy, fixture_bam, records
from fiberhmm.inference.chimeras import clear_chimera_mark, mark_chimera


@pytest.mark.parametrize('prior_qcfail', [False, True])
@pytest.mark.parametrize('decision', ['clean', 'chimera'])
def test_nrl_recomputed_decisions_override_stale_marks(tmp_path, monkeypatch,
                                                      prior_qcfail, decision):
    from fiberhmm.inference import engine, nrl_estimate as ne
    source = fixture_bam(tmp_path / 'source.bam')
    read = records(source)[-1]
    read.is_qcfail = prior_qcfail
    mark_chimera(read)
    with pysam.AlignmentFile(source, 'wb', header=read.header) as bam:
        bam.write(read)
    original = source.read_bytes()
    engine.configure_daf_chimera_filter(False)  # caller's completed decision pass
    monkeypatch.setattr(ne, 'load_model', lambda *a: None)
    monkeypatch.setattr(ne, 'freeze_model_for_inference', lambda *a: None)
    monkeypatch.setattr(ne, 'load_model_with_metadata', lambda *a: (None, None, None))
    monkeypatch.setattr(ne, 'build_llr_tables', lambda *a: (None, None))
    extracted = []
    extract = ne._extract_fiber_read_from_pysam
    def tracked(read, *args):
        assert not read.has_tag('xc') and not read.has_tag('xq')
        assert read.is_qcfail == prior_qcfail
        extracted.append(read.query_name)
        return extract(read, *args)
    monkeypatch.setattr(ne, '_extract_fiber_read_from_pysam', tracked)
    monkeypatch.setattr(ne, 'run_hmm_apply_stage', lambda *a: {
        'ns': [0, 190], 'nl': [100, 100], 'encoded': [0] * 600})
    monkeypatch.setattr(ne, 'recall_nucs_in_read', lambda *a, **k: (
        [SimpleNamespace(start=0, length=100), SimpleNamespace(start=190, length=100)], None))
    names = {'clean'} if decision == 'chimera' else set()
    result = ne.estimate_phase_nrl(str(source), 'unused', min_pairs=1,
                                   precall_chimeric_names=names)
    assert result['n_pairs'] == (0 if names else 1)
    assert result['source'] == ('anchor' if names else 'estimated')
    assert extracted == ([] if names else ['clean'])
    assert source.read_bytes() == original
    # Standalone callers without a recomputed decision still respect the mark.
    assert ne.estimate_phase_nrl(str(source), 'unused', min_pairs=1)['n_reads'] == 0


@pytest.mark.parametrize('prior_qcfail', [False, True])
def test_insert_evidence_survives_owned_mark_cleanup(tmp_path, prior_qcfail):
    from fiberhmm.daf.insert_consensus import ReadInsertEvidence, record_key
    from fiberhmm.inference import engine
    read = records(fixture_bam(tmp_path / 'source.bam'))[-1]
    read.cigarstring = '100M50I450M'
    read.is_qcfail = prior_qcfail
    mark_chimera(read)
    key = record_key(read)  # consensus prepass sees the old mark
    evidence = ReadInsertEvidence(known=set(range(100, 150)))
    engine.configure_daf_insert_evidence({key: evidence})
    assert engine.daf_insert_evidence(read) is evidence
    clear_chimera_mark(read)  # clean under the new decision
    assert read.is_qcfail == prior_qcfail
    assert record_key(read) == key
    assert engine.daf_insert_evidence(read) is evidence
    assert not engine.daf_unaligned_without_evidence(read)


@pytest.mark.parametrize('option', [['--chimeras', 'keep'], ['--keep-chimeras']])
@pytest.mark.parametrize('region_parallel', [False, True])
def test_keep_recall_clears_marks_for_downstream_consumers(tmp_path, option, region_parallel):
    from pathlib import Path
    import fiberhmm
    from fiberhmm.cli.export_posteriors import extract_posteriors_from_read
    from fiberhmm.cli.extract_tags import _init_extract_worker, _extract_region_worker
    from fiberhmm.core.model_io import load_model
    from fiberhmm.inference.strand_rescue import hard_observations
    from fiberhmm.inference.tf_recaller import extract_modification_calls
    from fiberhmm.io.footprint_bam import load_footprint_molecules_from_bam
    source = fixture_bam(tmp_path / 'source.bam')
    reads = records(source)
    for i, read in enumerate(reads):
        read.is_qcfail = i == 3  # another tool's QC failure must survive
        mark_chimera(read)
    with pysam.AlignmentFile(source, 'wb', header=reads[0].header) as bam:
        for read in reads:
            bam.write(read)
    pysam.index(str(source))
    target = tmp_path / 'keep.bam'
    command = [sys.executable, '-m', 'fiberhmm.cli.call', '-i', str(source),
               '-o', str(target), '--enzyme', 'dddb', '--seq', 'nanopore',
               '-c', '1', '--io-threads', '1', '--min-read-length', '0',
               '--no-dedup', '--no-qc', '--no-daf-call-snps',
               '--daf-insert-consensus', 'off'] + option
    if region_parallel:
        command.append('--region-parallel')
    run = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    current = records(target)
    assert len(current) == len(reads)
    assert all(not r.has_tag('xc') and not r.has_tag('xq') for r in current)
    assert [r.is_qcfail for r in current] == [False, False, False, True]
    primary = next(r for r in current if r.query_name == 'swap' and r.flag == 0)
    assert primary.has_tag('MA') and primary.has_tag('ns')
    assert extract_modification_calls(primary, 'daf') is not None
    assert hard_observations(primary, 'daf', 'daf', 3, 128) is not None
    assert load_footprint_molecules_from_bam(str(target)).diagnostics.skipped_chimera == 0
    bed = tmp_path / 'deam.bed'
    _init_extract_worker(dict(extract_types=['deam'], min_mapq=0,
                              prob_threshold=128, with_scores=False))
    _, _, counts = _extract_region_worker((('chr1', 0, 1000), str(target),
                                           {'deam': str(bed)}))
    assert counts.get('chimeras_skipped', 0) == 0
    assert '\tswap\t' in bed.read_text()
    model = load_model(str(Path(fiberhmm.__file__).parent / 'models/dddb_nanopore.json'))
    stats = {}
    assert extract_posteriors_from_read(current[-1], model, 'daf', 3, 10, stats=stats) is not None
    assert stats.get('chimera', 0) == 0


@pytest.mark.parametrize('mode', ['flag', 'remove', 'keep'])
@pytest.mark.parametrize('extra', [[], ['--chimeras', 'remove'],
                                  ['--chimeras=keep'], ['--keep-chimeras']])
def test_browser_301_recall_preserves_exact_chimera_mode(mode, extra):
    from fiberhmm.pipeline.runner import PipelineConfig
    original = PipelineConfig(['a'], 'r', 'dddb', 'o', chimeras=mode, call_args=extra)
    settings = original.calling_settings()
    # FiberBrowser 3.0.1 carries only the boolean and shlex-joined call_args.
    carried = shlex.split(shlex.join(settings['call_args']))
    recalled = PipelineConfig(['a'], 'r', 'dddb', 'o',
                              chimera_filter=settings['chimera_filter'], call_args=carried)
    assert recalled.effective_chimeras() == original.effective_chimeras()
    assert carried[:len(extra)] == extra
    assert original.call_args == extra  # serializing must not mutate the config


def test_standalone_auto_qc_keeps_sampled_daf_chimera_policy(tmp_path):
    from fiberhmm.qc.core import run_qc
    source = fixture_bam(tmp_path / 'copied.bam')
    reads = records(source)
    header = reads[0].header.to_dict()
    header['PG'] = [{'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call',
                     'DS': 'mode=daf enzyme=dddb coord=molecular chimeras=flag '
                           'chimera_min_seg=7 chimera_purity=0.9'}]
    mark_chimera(reads[2])
    with pysam.AlignmentFile(source, 'wb', header=header) as bam:
        for read in reads:
            bam.write(read)
    pysam.index(str(source))
    prefix = tmp_path / 'standalone'
    run_qc(str(source), output_prefix=str(prefix), sample_reads=20,
           state_source='none', stream=io.StringIO())  # default mode=auto; no sidecar
    result = json.loads(prefix.with_suffix('.qc.json').read_text())
    assert result['assay']['mode'] == 'daf'
    chim = result['chimeras']
    assert chim['mode'] == 'flag'
    assert not chim['full_run']
    assert chim['n_reads_checked'] == 2 and chim['n_chimeric'] == 1
    assert chim['fraction'] == 0.5
    assert chim['min_seg'] == 7 and chim['purity'] == 0.9
    tsv = next(csv.DictReader(io.StringIO(prefix.with_suffix('.qc.tsv').read_text()), delimiter='\t'))
    assert tsv['chimeras_mode'] == 'flag' and float(tsv['chimeras_fraction']) == 0.5
    assert prefix.with_suffix('.qc.pdf').stat().st_size > 0
