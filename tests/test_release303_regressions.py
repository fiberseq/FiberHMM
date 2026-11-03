"""Release audit regressions: preserve 3.0.2 APIs and 3.0.1 consumers."""
import inspect
import sys
from concurrent.futures import Future
from dataclasses import fields
from types import SimpleNamespace

import pysam
import pytest

from test_chimera_flags import extraction_policy, fixture_bam, records
from fiberhmm.inference.chimeras import (
    _is_chimeric, clear_chimera_mark, mark_chimera, prepare_chimeras,
)


def test_302_footprint_diagnostics_constructors():
    from fiberhmm.io.footprint_bam import BamFootprintDiagnostics
    legacy = ('records_seen primary_mapped_records emitted_molecules skipped_unmapped '
              'skipped_secondary skipped_supplementary skipped_qcfail skipped_duplicate '
              'skipped_low_mapq skipped_missing_ma skipped_without_mapped_blocks '
              'duplicate_region_records reads_with_tag_stratum reads_with_alignment_stratum '
              'tf_annotations projected_tf_annotations geometry_eligible_tf_annotations '
              'unprojectable_tf_annotations out_of_scope_tf_annotations msp_annotations '
              'projected_msp_annotations incomplete_msp_annotations unprojectable_msp_annotations').split()
    values = dict(zip(legacy, range(len(legacy))))
    for result in (BamFootprintDiagnostics(*values.values()), BamFootprintDiagnostics(**values)):
        assert all(getattr(result, key) == value for key, value in values.items())
        assert result.skipped_chimera == 0


def test_302_qc_positional_arguments():
    from fiberhmm.qc.core import run_qc
    legacy = ('input_path output_prefix mode enzyme reference_profile reference_fasta '
              'sample_reads seed min_mapq prob_threshold min_opportunities snp_report_path '
              'snp_mask_path snp_preflight_summary dedup_run_summary stream return_arrays '
              'state_source min_msp_bp light_call_reads light_call_seconds use_m5c cpg_mask_policy').split()
    bound = inspect.signature(run_qc).bind(*range(len(legacy)))
    assert dict(bound.arguments) == dict(zip(legacy, range(len(legacy))))


def test_302_pipeline_positional_constructor():
    from fiberhmm.pipeline.runner import PipelineConfig
    legacy = ('reads reference enzyme outdir sample cores seq topology regions min_mapq '
              'min_read_length hard_clip origin_merge origin_tolerance dedup dedup_mode '
              'snp_screen snp_mask chimera_filter daf_mask_unaligned alignments prob_threshold '
              'use_m5c cpg_mask_policy qc call_args call_mode tracks aligner force_chemistry '
              'redo basecaller_info modbase_model dorado dorado_model dorado_modified_bases '
              'dorado_modbase_models dorado_device dorado_batchsize dorado_models_dir '
              'dorado_args verbose quiet').split()
    result = PipelineConfig(*range(len(legacy)))
    assert [getattr(result, key) for key in legacy] == list(range(len(legacy)))
    assert [f.name for f in fields(PipelineConfig)][:len(legacy)] == legacy
    assert result.chimeras is None


@pytest.mark.parametrize('enzyme', ['ddda', 'dddb', 'hia5'])
@pytest.mark.parametrize('mode', ['flag', 'remove', 'keep'])
def test_pipeline_settings_accepted_by_fiberbrowser_301(enzyme, mode):
    from fiberhmm.pipeline.runner import PipelineConfig
    # Frozen FiberBrowser 3.0.1 library_jobs.SETTING_KEYS contract.
    accepted = set(('enzyme seq min_mapq min_read_length dedup dedup_mode snp_screen '
                    'snp_mask chimera_filter primary_only prob_threshold use_m5c cpg_mask_policy '
                    'hard_clip origin_merge qc tracks call_args alignments daf_unaligned_mask '
                    'daf_insert_consensus basecaller basecaller_override').split())
    settings = PipelineConfig(['a'], 'r', enzyme, 'o', chimeras=mode).calling_settings()
    assert set(settings) <= accepted
    assert settings['chimera_filter'] == (mode != 'keep' if enzyme != 'hia5' else None)


@pytest.mark.parametrize('alignments,flag,expected', [
    ('primary', 256, False), ('primary', 2048, False),
    ('primary-supplementary', 256, False), ('primary-supplementary', 2048, True),
    ('all', 256, True), ('all', 2048, True),
])
def test_decision_alignment_eligibility(tmp_path, alignments, flag, expected):
    source = fixture_bam(tmp_path / 'source.bam')
    reads = records(source)
    reads = [reads[-1], reads[2]]
    reads[0].query_name = 'swap'
    reads[1].flag = flag
    with pysam.AlignmentFile(source, 'wb', header=reads[0].header) as out:
        for read in reads:
            out.write(read)
    names = set()
    prepare_chimeras(str(source), chimeric_names=names, alignments=alignments)
    assert names == ({'swap'} if expected else set())


@pytest.mark.parametrize('option', [['--alignments', 'all'], ['--no-primary']])
@pytest.mark.parametrize('mode', ['flag', 'remove'])
@pytest.mark.parametrize('region_parallel', [False, True])
def test_cli_secondary_only_evidence(tmp_path, option, mode, region_parallel):
    import subprocess
    source = fixture_bam(tmp_path / 'source.bam')
    reads = records(source)
    reads[2].flag = 256  # the sole diagnostic record is secondary
    reads[1].flag = 0
    with pysam.AlignmentFile(source, 'wb', header=reads[0].header) as out:
        for read in reads:
            out.write(read)
    pysam.index(str(source))
    target = tmp_path / 'out.bam'
    cmd = [sys.executable, '-m', 'fiberhmm.cli.call', '-i', str(source), '-o', str(target),
           '--enzyme', 'dddb', '-c', '1', '--io-threads', '1', '--min-read-length', '0',
           '--no-dedup', '--no-qc', '--no-daf-call-snps', '--daf-insert-consensus', 'off',
           '--chimeras', mode] + option + (['--region-parallel'] if region_parallel else [])
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    swaps = [read for read in records(target) if read.query_name == 'swap']
    assert len(swaps) == (3 if mode == 'flag' else 0)
    assert all(read.is_qcfail and read.get_tag('xc') == 'strand_swap'
               and not read.has_tag('MA') and not read.has_tag('ns') for read in swaps)


@pytest.mark.parametrize('reverse', [False, True])
def test_balanced_raw_chimera(tmp_path, reverse):
    from fiberhmm.daf.encoder import get_daf_positions, is_daf_chimera
    read = records(fixture_bam(tmp_path / 'source.bam'))[-1]
    read.query_sequence = 'T' * 10 + 'A' * 10
    read.cigarstring = '20M'
    read.flag = 16 if reverse else 0
    read.set_tag('MD', '0' + 'C0' * 10 + 'G0' * 10)
    assert get_daf_positions(read) is None  # ambiguous dominant strand
    ct, ga, _ = get_daf_positions(read, force_strand='CT')
    assert is_daf_chimera(ct, ga)
    assert _is_chimeric(read, None)
    source, target = tmp_path / 'raw.bam', tmp_path / 'out.bam'
    with pysam.AlignmentFile(source, 'wb', header=read.header) as out:
        out.write(read)
    stats = prepare_chimeras(str(source), str(target))
    assert stats['n_chimeric'] == 1
    assert records(target)[0].is_qcfail
    prepare_chimeras(str(source), str(target), mode='remove')
    assert records(target) == []


@pytest.mark.parametrize('initial_qcfail', [False, True])
def test_qcfail_ownership_roundtrip(tmp_path, initial_qcfail):
    read = records(fixture_bam(tmp_path / 'source.bam'))[-1]
    read.is_qcfail = initial_qcfail
    mark_chimera(read)
    mark_chimera(read)  # idempotence must preserve original ownership
    assert read.is_qcfail
    clear_chimera_mark(read)
    assert read.is_qcfail == initial_qcfail
    assert not read.has_tag('xc')


def test_foreign_xc_cleanup_and_counts(tmp_path):
    source = fixture_bam(tmp_path / 'source.bam')
    reads = records(source)
    read = reads[-1]
    read.is_qcfail = True
    read.set_tag('xc', 'foreign_reason')
    clear_chimera_mark(read)
    assert read.is_qcfail and read.get_tag('xc') == 'foreign_reason'
    with pysam.AlignmentFile(source, 'wb', header=read.header) as out:
        out.write(read)
    target = tmp_path / 'out.bam'
    stats = prepare_chimeras(str(source), str(target), mode='keep')
    assert stats['n_stale_marks_cleared'] == 0
    assert records(target)[0].get_tag('xc') == 'foreign_reason'
    assert records(target)[0].is_qcfail


def test_legacy_mark_without_ownership_preserves_qcfail(tmp_path):
    read = records(fixture_bam(tmp_path / 'source.bam'))[-1]
    read.is_qcfail = True
    read.set_tag('xc', 'strand_swap')
    clear_chimera_mark(read)
    assert read.is_qcfail  # original owner cannot be recovered from old marks
    assert not read.has_tag('xc')


@pytest.mark.parametrize('skip_kind', ['chimera', 'low_mapq', 'no_modifications'])
def test_streaming_skips_write_before_eof(tmp_path, monkeypatch, benchmark_model_path, skip_kind):
    from fiberhmm.inference import streaming_pipeline as sp
    source = fixture_bam(tmp_path / 'source.bam')
    original = records(source)[-1]
    with pysam.AlignmentFile(source, 'wb', header=original.header) as out:
        for i in range(1000):
            original.query_name = str(i)
            if skip_kind == 'no_modifications':
                original.query_sequence = 'ACGT' * 150
            out.write(original)
    written = []
    real_open = pysam.AlignmentFile

    class TrackedBam:
        def __init__(self, bam, writing):
            self.bam, self.writing = bam, writing
        def __getattr__(self, name):
            return getattr(self.bam, name)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.bam.close()
        def fetch(self, **kwargs):
            for i, read in enumerate(self.bam.fetch(**kwargs)):
                if i >= 20:
                    assert len(written) >= i - 16, 'skipped BAM records buffered without bound'
                yield read
        def write(self, read):
            written.append(read.query_name)
            return self.bam.write(read)

    class Executor:
        def __init__(self, **kwargs):
            pass
        def submit(self, *args):
            future = Future()
            future.set_result([])
            return future
        def shutdown(self, **kwargs):
            pass

    monkeypatch.setattr(sp, 'ProcessPoolExecutor', Executor)
    if skip_kind == 'no_modifications':
        monkeypatch.setattr(sp, 'make_apply_payload', lambda *a, **k: None)
    monkeypatch.setattr(pysam, 'AlignmentFile', lambda *a, **k: TrackedBam(real_open(*a, **k), a[1] == 'wb'))
    sp._process_bam_streaming_pipeline_fused(
        str(source), str(tmp_path / 'out.bam'), benchmark_model_path, benchmark_model_path,
        set(), 10, False, 'daf', 3, 0, 85, 61 if skip_kind == 'low_mapq' else 0,
        128, 0, False, 4., 3, 0, 0., False, False, 0, 1, 4, 1,
        precall_chimeric_names=set(map(str, range(1000))) if skip_kind == 'chimera' else set())
    assert written == list(map(str, range(1000)))


def test_auto_nrl_matches_302_exclusion(tmp_path, monkeypatch):
    """Use real DAF extraction; deterministic recalled spacings isolate NRL policy."""
    from fiberhmm.cli import call
    from fiberhmm.inference import engine, nrl_estimate as ne
    source = fixture_bam(tmp_path / 'source.bam')
    original = records(source)
    with pysam.AlignmentFile(source, 'wb', header=original[0].header) as out:
        for prefix, template, count in [('swap', original[2], 500), ('clean', original[-1], 320)]:
            for i in range(count):
                template.query_name = f'{prefix}{i}'
                out.write(template)
    monkeypatch.setattr(ne, 'load_model', lambda *a: None)
    monkeypatch.setattr(ne, 'freeze_model_for_inference', lambda *a: None)
    monkeypatch.setattr(ne, 'load_model_with_metadata', lambda *a: (None, None, None))
    monkeypatch.setattr(ne, 'build_llr_tables', lambda *a: (None, None))
    monkeypatch.setattr(ne, 'run_hmm_apply_stage', lambda fr, *a: {
        'ns': [0, 1], 'nl': [100, 100], 'encoded': [250 if fr['read_id'].startswith('swap') else 190]})
    monkeypatch.setattr(ne, 'recall_nucs_in_read', lambda encoded, *a, **k: (
        [SimpleNamespace(start=0, length=100), SimpleNamespace(start=encoded[0], length=100)], None))
    engine.configure_daf_chimera_filter(True)
    legacy = ne.estimate_phase_nrl(str(source), 'unused')
    assert legacy['source'] == 'estimated' and legacy['n_pairs'] == 320
    observed = []
    estimate = ne.estimate_phase_nrl
    def tracked(*a, **k):
        result = estimate(*a, **k)
        observed.append(result)
        return result
    monkeypatch.setattr(ne, 'estimate_phase_nrl', tracked)
    monkeypatch.setattr(call, '_process_bam_streaming_pipeline_fused', lambda **k: (0, 0))
    monkeypatch.setattr(sys, 'argv', [
        'fiberhmm-call', '-i', str(source), '-o', str(tmp_path / 'calls.bam'),
        '--enzyme', 'dddb', '--seq', 'nanopore', '--min-read-length', '0',
        '--no-qc', '--no-dedup', '--no-daf-call-snps', '--daf-insert-consensus', 'off',
        '--phase-nrl', 'auto'])
    call._main(call.parse_args())
    assert observed == [legacy]
