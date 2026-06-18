"""CLI characterization tests for `fiberhmm-call`."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pysam
import pytest
from conftest import make_synthetic_bam, make_synthetic_iupac_bam

from fiberhmm.cli.call import (
    _build_pg_record,
    _build_pg_record_from_request,
    _call_banner_settings,
    _call_banner_text,
    _call_banner_text_from_settings,
    _call_circular_label,
    _call_context_size_or_default,
    _call_enzyme_label,
    _call_fused_common_kwargs,
    _call_fused_common_kwargs_from_settings,
    _call_mode_label,
    _call_mode_or_default,
    _call_pg_description,
    _call_pg_description_from_request,
    _call_recall_model_label,
    _CallBannerSettings,
    _CallFusedCommonSettings,
    _CallPgDescriptionRequest,
    _CallPgRecordRequest,
    _CallPhaseNrlRequest,
    _CallPipelineRequest,
    _CallRuntimeRequest,
    _check_daf_inputs,
    _check_region_parallel_file_io,
    _chimera_filter_state,
    _daf_sources_available,
    _DafSniffResult,
    _index_streaming_call_output,
    _invalid_phase_nrl_message,
    _is_phase_nrl_off,
    _missing_daf_source_message,
    _new_daf_sniff_result,
    _normalize_phase_nrl_option,
    _parse_fixed_phase_nrl,
    _phase_nrl_estimate_message,
    _resolve_apply_model,
    _resolve_call_chroms,
    _resolve_call_mode_context,
    _resolve_call_runtime,
    _resolve_call_runtime_for_request,
    _resolve_phase_nrl,
    _resolve_phase_nrl_for_request,
    _resolve_recall_model,
    _resolve_recall_nucs,
    _run_call_pipeline,
    _run_call_pipeline_for_request,
    _should_check_daf_inputs,
    _should_warn_stale_daf_md,
    _sniff_daf_input_sources,
    _stale_daf_md_warning_message,
)


def test_fiberhmm_call_stdout_is_clean_bam_stream(benchmark_model_path, tmp_path):
    input_bam = str(tmp_path / "input.bam")
    make_synthetic_bam(
        input_bam,
        n_reads=4,
        read_length=1500,
        n_chroms=1,
        chrom_length=20_000,
        seed=321,
    )

    cmd = [
        sys.executable, "-m", "fiberhmm.cli.call",
        "-i", input_bam,
        "-o", "-",
        "-m", benchmark_model_path,
        "--mode", "pacbio-fiber",
        "--min-read-length", "0",
        "--prob-threshold", "0",
        "--min-llr", "1000",
        "--chunk-size", "2",
        "--io-threads", "1",
        "-c", "1",
        "--max-reads", "4",
    ]

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert b"fiberhmm-call" in result.stderr
    assert b"fiberhmm-call" not in result.stdout

    stdout_bam = tmp_path / "stdout.bam"
    stdout_bam.write_bytes(result.stdout)
    with pysam.AlignmentFile(stdout_bam, "rb", check_sq=False) as bam:
        reads = list(bam.fetch(until_eof=True))

    assert len(reads) == 4


def test_daf_input_sniff_accepts_iupac_encoding(tmp_path):
    input_bam = str(tmp_path / "iupac.bam")
    make_synthetic_iupac_bam(
        input_bam,
        n_reads=4,
        read_length=200,
        n_chroms=1,
        chrom_length=5_000,
        seed=11,
    )

    sniff = _sniff_daf_input_sources(input_bam, n_sniff=4)
    assert sniff.has_ry is True
    assert sniff.checked == 4

    _check_daf_inputs(input_bam, n_sniff=4)


def test_daf_input_sniff_accepts_reference_fallback(tmp_path):
    input_bam = str(tmp_path / "raw.bam")
    make_synthetic_bam(
        input_bam,
        n_reads=4,
        read_length=200,
        n_chroms=1,
        chrom_length=5_000,
        seed=12,
    )

    _check_daf_inputs(input_bam, reference="ref.fa", n_sniff=4)


def test_daf_input_sniff_rejects_missing_deamination_source(tmp_path, capsys):
    input_bam = str(tmp_path / "raw.bam")
    make_synthetic_bam(
        input_bam,
        n_reads=4,
        read_length=200,
        n_chroms=1,
        chrom_length=5_000,
        seed=13,
    )

    with pytest.raises(SystemExit) as exc:
        _check_daf_inputs(input_bam, n_sniff=4)

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--mode daf needs deamination calls" in err
    assert "--reference ref.fa" in err


def test_daf_input_predicates_cover_sources_and_md_warning():
    sniff_result = _new_daf_sniff_result()
    assert sniff_result == _DafSniffResult()
    sniff_result.has_ry = True
    assert _new_daf_sniff_result().has_ry is False

    sniff = _DafSniffResult()
    assert not _daf_sources_available(sniff, has_ref=False)
    assert _daf_sources_available(sniff, has_ref=True)

    sniff = _DafSniffResult(has_ry=True, md_bad=2)
    assert _daf_sources_available(sniff, has_ref=False)
    assert not _should_warn_stale_daf_md(sniff, has_ref=False)

    sniff = _DafSniffResult(has_md=True, md_bad=2)
    assert _daf_sources_available(sniff, has_ref=False)
    assert _should_warn_stale_daf_md(sniff, has_ref=False)
    assert not _should_warn_stale_daf_md(sniff, has_ref=True)


def test_missing_daf_source_message_mentions_all_recovery_paths():
    message = _missing_daf_source_message("input.bam", checked_reads=7)

    assert "first 7 mapped reads of input.bam" in message
    assert "R/Y IUPAC codes" in message
    assert "MD tags" in message
    assert "--reference ref.fa" in message


def test_stale_daf_md_warning_message_mentions_counts_and_recovery():
    message = _stale_daf_md_warning_message(
        _DafSniffResult(md_bad=2, md_total=5),
    )

    assert "2/5" in message
    assert "MD/CIGAR length mismatches" in message
    assert "samtools calmd" in message
    assert "--reference ref.fa" in message


def test_should_check_daf_inputs_only_for_daf_file_input():
    assert _should_check_daf_inputs("daf", "input.bam")
    assert not _should_check_daf_inputs("daf", "-")
    assert not _should_check_daf_inputs("pacbio-fiber", "input.bam")


def test_call_model_resolution_uses_custom_paths():
    args = SimpleNamespace(
        model="/tmp/custom_apply.json",
        recall_model="/tmp/custom_recall.json",
        enzyme=None,
        seq=None,
    )

    assert _resolve_apply_model(args) == "/tmp/custom_apply.json"
    assert _resolve_recall_model(args) == "/tmp/custom_recall.json"


def test_call_model_resolution_uses_separate_ddda_models():
    args = SimpleNamespace(model=None, recall_model=None, enzyme="ddda", seq=None)

    apply_model = _resolve_apply_model(args)
    recall_model = _resolve_recall_model(args)

    assert apply_model.endswith("ddda_nuc.json")
    assert recall_model.endswith("ddda_TF.json")
    assert apply_model != recall_model


def test_call_model_resolution_requires_model_or_enzyme(capsys):
    args = SimpleNamespace(model=None, recall_model=None, enzyme=None, seq=None)

    with pytest.raises(SystemExit) as exc:
        _resolve_apply_model(args)

    assert exc.value.code == 1
    assert "one of --model or --enzyme required" in capsys.readouterr().err


def test_call_mode_context_resolution_uses_overrides_metadata_and_defaults():
    assert _call_mode_or_default(" daf ", "pacbio-fiber") == "daf"
    assert _call_mode_or_default(None, "nanopore-fiber") == "nanopore-fiber"
    assert _call_mode_or_default("", " nanopore-fiber ") == "nanopore-fiber"
    assert _call_mode_or_default(None, None) == "pacbio-fiber"
    assert _call_context_size_or_default(5, 3) == 5
    assert _call_context_size_or_default("5", 3) == 5
    assert _call_context_size_or_default(0, 4) == 0
    assert _call_context_size_or_default(None, 4) == 4
    assert _call_context_size_or_default(None, None) == 3

    args = SimpleNamespace(mode="daf", context_size=5)
    context = _resolve_call_mode_context(args, model_k=3, model_mode="pacbio-fiber")
    assert context.mode == "daf"
    assert context.k == 5

    args = SimpleNamespace(mode=None, context_size=None)
    context = _resolve_call_mode_context(args, model_k=4, model_mode="nanopore-fiber")
    assert context.mode == "nanopore-fiber"
    assert context.k == 4
    context = _resolve_call_mode_context(args, model_k=None, model_mode=None)
    assert context.mode == "pacbio-fiber"
    assert context.k == 3


def test_call_region_parallel_helpers(capsys):
    assert _resolve_call_chroms(None) is None
    assert _resolve_call_chroms([]) is None
    assert _resolve_call_chroms(["chr2", "chr1", "chr2"]) == {"chr1", "chr2"}

    _check_region_parallel_file_io(SimpleNamespace(input="in.bam", output="out.bam"))

    with pytest.raises(SystemExit) as exc:
        _check_region_parallel_file_io(SimpleNamespace(input="-", output="out.bam"))

    assert exc.value.code == 1
    assert "--region-parallel requires file I/O" in capsys.readouterr().err


def test_call_phase_nrl_literal_helpers():
    assert _normalize_phase_nrl_option(" Auto ") == "auto"
    assert _is_phase_nrl_off("off")
    assert _is_phase_nrl_off("none")
    assert _is_phase_nrl_off("0")
    assert not _is_phase_nrl_off("185")
    assert _parse_fixed_phase_nrl("185") == 185
    assert _parse_fixed_phase_nrl("-10") == 0
    assert _parse_fixed_phase_nrl("auto") is None
    assert _invalid_phase_nrl_message("bad") == (
        "  WARNING: invalid --phase-nrl 'bad'; using off."
    )


def test_resolve_phase_nrl_request_handles_non_auto_paths(capsys):
    request = _CallPhaseNrlRequest(
        apply_model_path="apply.json",
        recall_model_path="recall.json",
        mode="daf",
        k=3,
        recall_nucs=True,
    )
    args = SimpleNamespace(phase_nrl="off")

    assert _resolve_phase_nrl_for_request(args, request) == 0

    args.phase_nrl = "185"
    assert _resolve_phase_nrl_for_request(args, request) == 185
    assert _resolve_phase_nrl(
        args,
        apply_model_path="apply.json",
        recall_model_path="recall.json",
        mode="daf",
        k=3,
        recall_nucs=True,
    ) == 185

    args.phase_nrl = "auto"
    disabled_request = _CallPhaseNrlRequest(
        apply_model_path="apply.json",
        recall_model_path="recall.json",
        mode="daf",
        k=3,
        recall_nucs=False,
    )
    assert _resolve_phase_nrl_for_request(args, disabled_request) == 0

    args.phase_nrl = "bad"
    assert _resolve_phase_nrl_for_request(args, request) == 0
    assert "invalid --phase-nrl 'bad'" in capsys.readouterr().err


def test_phase_nrl_estimate_message_formats_ci_and_counts():
    message = _phase_nrl_estimate_message({
        "nrl": 185,
        "source": "estimated",
        "n_pairs": 1234,
        "n_reads": 56,
        "ci": (170.2, 199.8),
    })

    assert message == (
        "  phase NRL: 185 bp (estimated, 1,234 pairs from 56 reads CI[170-200])"
    )


def test_phase_nrl_estimate_message_omits_missing_ci():
    message = _phase_nrl_estimate_message({
        "nrl": 185,
        "source": "fallback",
        "n_pairs": 0,
        "n_reads": 0,
        "ci": None,
    })

    assert message == "  phase NRL: 185 bp (fallback, 0 pairs from 0 reads)"


def test_call_recall_nucs_defaults_and_ddda_warnings(capsys):
    args = SimpleNamespace(enzyme="hia5", recall_nucs=None)
    assert _resolve_recall_nucs(args) is True
    assert capsys.readouterr().err == ""

    args = SimpleNamespace(enzyme="ddda", recall_nucs=None)
    assert _resolve_recall_nucs(args) is False
    assert "OFF by default for DddA" in capsys.readouterr().err

    args = SimpleNamespace(enzyme="ddda", recall_nucs=True)
    assert _resolve_recall_nucs(args) is True
    assert "--recall-nucs on DddA" in capsys.readouterr().err


def test_chimera_filter_state_is_daf_specific():
    assert _chimera_filter_state("daf", keep_chimeras=False) == "on"
    assert _chimera_filter_state("daf", keep_chimeras=True) == "off"
    assert _chimera_filter_state("pacbio-fiber", keep_chimeras=False) == "n/a"


def test_call_pg_record_documents_molecular_coordinates():
    request = _CallPgRecordRequest(
        mode="daf",
        recall_nucs=True,
        phase_nrl=185,
        keep_chimeras=False,
        argv=["fiberhmm-call", "-i", "in.bam"],
    )
    record = _build_pg_record_from_request(request)

    assert record["PN"] == "fiberhmm-call"
    assert record["CL"] == "fiberhmm-call -i in.bam"
    assert "coord=molecular" in record["DS"]
    assert "mode=daf" in record["DS"]
    assert "recall_nucs=True" in record["DS"]
    assert "phase_nrl=185" in record["DS"]
    assert "chimera_filter=on" in record["DS"]
    assert _build_pg_record(
        mode="daf",
        recall_nucs=True,
        phase_nrl=185,
        keep_chimeras=False,
        argv=["fiberhmm-call", "-i", "in.bam"],
    ) == record


def test_call_pg_description_documents_settings():
    request = _CallPgDescriptionRequest(
        mode="daf",
        recall_nucs=False,
        phase_nrl=0,
        keep_chimeras=True,
    )
    description = _call_pg_description_from_request(request)

    assert "coord=molecular" in description
    assert "mode=daf" in description
    assert "recall_nucs=False" in description
    assert "phase_nrl=0" in description
    assert "chimera_filter=off" in description
    assert _call_pg_description(
        mode="daf",
        recall_nucs=False,
        phase_nrl=0,
        keep_chimeras=True,
    ) == description


def test_call_banner_label_helpers():
    assert _call_mode_label(False) == "streaming"
    assert _call_mode_label(True) == "region-parallel"
    assert _call_recall_model_label(None) == "(reuse apply model)"
    assert _call_recall_model_label("recall.json") == "recall.json"
    assert _call_enzyme_label(None) == "custom"
    assert _call_enzyme_label("dddb") == "dddb"
    assert _call_circular_label(False) == ""
    assert _call_circular_label(True) == " circular=on"


def test_call_banner_text_formats_resolved_settings():
    settings = _CallBannerSettings(
        apply_model_path="apply.json",
        recall_model_path=None,
        mode="daf",
        k=4,
        enzyme=None,
        min_llr=1.5,
        min_opps=3,
        unify_threshold=120,
        uplift=0.25,
        cores=8,
        io_threads=2,
        circular=True,
        region_parallel=True,
    )
    banner = _call_banner_text_from_settings(settings)

    assert "fiberhmm-call [BETA]" in banner
    assert "fused apply + recall-tfs (region-parallel)" in banner
    assert "apply model:  apply.json" in banner
    assert "recall model: (reuse apply model)" in banner
    assert "mode=daf k=4 enzyme=custom" in banner
    assert "min_llr=1.5 min_opps=3 unify_threshold=120 uplift=0.25" in banner
    assert "cores=8 io-threads=2 circular=on" in banner
    assert _call_banner_text(
        apply_model_path="apply.json",
        recall_model_path=None,
        mode="daf",
        k=4,
        enzyme=None,
        min_llr=1.5,
        min_opps=3,
        unify_threshold=120,
        uplift=0.25,
        cores=8,
        io_threads=2,
        circular=True,
        region_parallel=True,
    ) == banner


def test_call_banner_settings_from_runtime_and_args():
    args = SimpleNamespace(
        enzyme="dddb",
        min_opps=3,
        unify_threshold=90,
        cores=4,
        io_threads=2,
        circular=False,
        region_parallel=True,
    )
    runtime = SimpleNamespace(
        apply_model_path="apply.json",
        recall_model_path="recall.json",
        mode="daf",
        k=3,
        min_llr=5.0,
        uplift=1.0,
    )

    assert _call_banner_settings(args, runtime) == _CallBannerSettings(
        apply_model_path="apply.json",
        recall_model_path="recall.json",
        mode="daf",
        k=3,
        enzyme="dddb",
        min_llr=5.0,
        min_opps=3,
        unify_threshold=90,
        uplift=1.0,
        cores=4,
        io_threads=2,
        circular=False,
        region_parallel=True,
    )


def test_call_fused_common_kwargs_preserve_shared_pipeline_arguments():
    pg_record = {"ID": "fiberhmm-call"}
    args = SimpleNamespace(
        input="in.bam",
        output="out.bam",
        edge_trim=10,
        circular=True,
        msp_min_size=0,
        nuc_min_size=85,
        min_mapq=20,
        prob_threshold=128,
        min_read_length=1000,
        with_scores=True,
        min_opps=3,
        unify_threshold=90,
        downstream_compat=False,
        cores=4,
        io_threads=2,
        primary=True,
        reference="ref.fa",
        split_min_llr=4.0,
        split_min_opps=3,
        keep_chimeras=False,
        chimera_min_seg=4,
        chimera_purity=0.85,
    )

    settings = _CallFusedCommonSettings(
        recall_model_path="recall.json",
        mode="daf",
        context_size=3,
        min_llr=5.0,
        emission_uplift=1.0,
        also_write_legacy=True,
        recall_nucs=True,
        phase_nrl=185,
        pg_record=pg_record,
    )
    kwargs = _call_fused_common_kwargs_from_settings(args, settings)

    assert kwargs["input_bam"] == "in.bam"
    assert kwargs["output_bam"] == "out.bam"
    assert kwargs["recall_model_path"] == "recall.json"
    assert kwargs["mode"] == "daf"
    assert kwargs["context_size"] == 3
    assert kwargs["min_llr"] == 5.0
    assert kwargs["n_cores"] == 4
    assert kwargs["filter_chimeras"] is True
    assert kwargs["phase_nrl"] == 185
    assert kwargs["pg_record"] is pg_record
    assert _call_fused_common_kwargs(
        args,
        recall_model_path="recall.json",
        mode="daf",
        context_size=3,
        min_llr=5.0,
        emission_uplift=1.0,
        also_write_legacy=True,
        recall_nucs=True,
        phase_nrl=185,
        pg_record=pg_record,
    ) == kwargs


def test_resolve_call_runtime_wires_setup_and_common_kwargs(monkeypatch):
    from fiberhmm.cli import call

    args = SimpleNamespace(
        input="in.bam",
        reference="ref.fa",
        keep_chimeras=False,
    )
    pg_record = {"ID": "fiberhmm-call"}
    calls = []

    monkeypatch.setattr(call, "_resolve_apply_model", lambda got_args: "apply.json")
    monkeypatch.setattr(call, "_resolve_recall_model", lambda got_args: "recall.json")
    monkeypatch.setattr(
        call,
        "load_model_with_metadata",
        lambda path: ("model", 4, "model-mode"),
    )
    monkeypatch.setattr(
        call,
        "_resolve_call_mode_context",
        lambda got_args, model_k, model_mode: call._CallModeContext("daf", 5),
    )
    monkeypatch.setattr(call, "resolve_recall_defaults", lambda got_args: (6.0, 1.2))
    monkeypatch.setattr(call, "_resolve_recall_nucs", lambda got_args: True)
    monkeypatch.setattr(call, "_should_check_daf_inputs", lambda mode, path: True)
    monkeypatch.setattr(
        call,
        "_check_daf_inputs",
        lambda path, reference: calls.append(("check", path, reference)),
    )
    monkeypatch.setattr(
        call,
        "_resolve_phase_nrl_for_request",
        lambda got_args, request: calls.append(
            ("phase", got_args, request)
        ) or 185,
    )
    monkeypatch.setattr(
        call,
        "_build_pg_record_from_request",
        lambda request: calls.append(("pg", request)) or pg_record,
    )
    monkeypatch.setattr(call, "should_write_legacy_tags", lambda got_args: False)
    monkeypatch.setattr(
        call,
        "_call_fused_common_kwargs_from_settings",
        lambda got_args, settings: calls.append(
            ("common", got_args, settings)
        ) or {"shared": True},
    )

    runtime = _resolve_call_runtime_for_request(
        _CallRuntimeRequest(args=args, argv=["fiberhmm-call", "-i", "in.bam"]),
    )

    assert runtime == call._CallRuntime(
        apply_model_path="apply.json",
        recall_model_path="recall.json",
        mode="daf",
        k=5,
        min_llr=6.0,
        uplift=1.2,
        recall_nucs=True,
        phase_nrl=185,
        pg_record=pg_record,
        also_write_legacy=False,
        common_kwargs={"shared": True},
    )
    assert calls[:3] == [
        ("check", "in.bam", "ref.fa"),
        (
            "phase",
            args,
            _CallPhaseNrlRequest(
                apply_model_path="apply.json",
                recall_model_path="recall.json",
                mode="daf",
                k=5,
                recall_nucs=True,
            ),
        ),
        (
            "pg",
            _CallPgRecordRequest(
                mode="daf",
                recall_nucs=True,
                phase_nrl=185,
                keep_chimeras=False,
                argv=["fiberhmm-call", "-i", "in.bam"],
            ),
        ),
    ]
    assert calls[3] == (
        "common",
        args,
        _CallFusedCommonSettings(
            recall_model_path="recall.json",
            mode="daf",
            context_size=5,
            min_llr=6.0,
            emission_uplift=1.2,
            also_write_legacy=False,
            recall_nucs=True,
            phase_nrl=185,
            pg_record=pg_record,
        ),
    )


def test_resolve_call_runtime_adapter_builds_request(monkeypatch):
    from fiberhmm.cli import call

    args = SimpleNamespace()
    argv = ["fiberhmm-call"]
    sentinel = object()
    calls = []

    monkeypatch.setattr(
        call,
        "_resolve_call_runtime_for_request",
        lambda request: calls.append(request) or sentinel,
    )

    assert _resolve_call_runtime(args, argv) is sentinel
    assert calls == [_CallRuntimeRequest(args=args, argv=argv)]


def test_run_call_pipeline_dispatches_streaming(monkeypatch):
    from fiberhmm.cli import call

    calls = []
    monkeypatch.setattr(
        call,
        "_process_bam_streaming_pipeline_fused",
        lambda **kwargs: calls.append(kwargs) or (10, 4),
    )
    args = SimpleNamespace(
        region_parallel=False,
        max_reads=100,
        chunk_size=500,
        process_unmapped=True,
    )
    common_kwargs = {"input_bam": "in.bam", "output_bam": "out.bam"}
    request = _CallPipelineRequest(
        args=args,
        apply_model_path="apply.json",
        common_kwargs=common_kwargs,
    )

    assert _run_call_pipeline_for_request(request) == (10, 4)
    assert calls == [{
        "model_path": "apply.json",
        "max_reads": 100,
        "chunk_size": 500,
        "process_unmapped": True,
        "input_bam": "in.bam",
        "output_bam": "out.bam",
    }]


def test_run_call_pipeline_dispatches_region(monkeypatch):
    from fiberhmm.cli import call

    calls = []
    checks = []
    monkeypatch.setattr(
        call,
        "_check_region_parallel_file_io",
        lambda args: checks.append(args),
    )
    monkeypatch.setattr(call, "_resolve_call_chroms", lambda chroms: {"chr2L"})
    monkeypatch.setattr(
        call,
        "_process_bam_region_parallel_fused",
        lambda **kwargs: calls.append(kwargs) or (12, 5),
    )
    args = SimpleNamespace(
        region_parallel=True,
        chroms=["chr2L"],
        region_size=123,
        skip_scaffolds=True,
    )
    common_kwargs = {"input_bam": "in.bam", "output_bam": "out.bam"}
    request = _CallPipelineRequest(
        args=args,
        apply_model_path="apply.json",
        common_kwargs=common_kwargs,
    )

    assert _run_call_pipeline_for_request(request) == (12, 5)
    assert checks == [args]
    assert calls == [{
        "apply_model_path": "apply.json",
        "region_size": 123,
        "skip_scaffolds": True,
        "chroms": {"chr2L"},
        "input_bam": "in.bam",
        "output_bam": "out.bam",
    }]


def test_run_call_pipeline_adapter_builds_request(monkeypatch):
    from fiberhmm.cli import call

    args = SimpleNamespace()
    common_kwargs = {"input_bam": "in.bam"}
    sentinel = object()
    calls = []

    monkeypatch.setattr(
        call,
        "_run_call_pipeline_for_request",
        lambda request: calls.append(request) or sentinel,
    )

    assert _run_call_pipeline(args, "apply.json", common_kwargs) is sentinel
    assert calls == [
        _CallPipelineRequest(
            args=args,
            apply_model_path="apply.json",
            common_kwargs=common_kwargs,
        ),
    ]


def test_index_streaming_call_output_only_indexes_streaming_files(monkeypatch):
    calls = []
    monkeypatch.setattr(pysam, "index", lambda path: calls.append(path))

    _index_streaming_call_output(
        SimpleNamespace(output="stream.bam", region_parallel=False),
        stdout_mode=False,
    )
    _index_streaming_call_output(
        SimpleNamespace(output="region.bam", region_parallel=True),
        stdout_mode=False,
    )
    _index_streaming_call_output(
        SimpleNamespace(output="-", region_parallel=False),
        stdout_mode=True,
    )

    assert calls == ["stream.bam"]
