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
    _call_banner_text,
    _call_context_size_or_default,
    _call_circular_label,
    _call_enzyme_label,
    _call_mode_or_default,
    _call_mode_label,
    _call_pg_description,
    _call_recall_model_label,
    _check_daf_inputs,
    _check_region_parallel_file_io,
    _chimera_filter_state,
    _daf_sources_available,
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
    _resolve_recall_nucs,
    _resolve_recall_model,
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
    assert sniff["has_ry"] is True
    assert sniff["checked"] == 4

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
    assert sniff_result == {
        "has_ry": False,
        "has_md": False,
        "md_bad": 0,
        "md_total": 0,
        "checked": 0,
    }
    sniff_result["has_ry"] = True
    assert _new_daf_sniff_result()["has_ry"] is False

    sniff = {"has_ry": False, "has_md": False, "md_bad": 0}
    assert not _daf_sources_available(sniff, has_ref=False)
    assert _daf_sources_available(sniff, has_ref=True)

    sniff = {"has_ry": True, "has_md": False, "md_bad": 2}
    assert _daf_sources_available(sniff, has_ref=False)
    assert not _should_warn_stale_daf_md(sniff, has_ref=False)

    sniff = {"has_ry": False, "has_md": True, "md_bad": 2}
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
    message = _stale_daf_md_warning_message({"md_bad": 2, "md_total": 5})

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
    assert _call_mode_or_default("daf", "pacbio-fiber") == "daf"
    assert _call_mode_or_default(None, "nanopore-fiber") == "nanopore-fiber"
    assert _call_mode_or_default(None, None) == "pacbio-fiber"
    assert _call_context_size_or_default(5, 3) == 5
    assert _call_context_size_or_default(None, 4) == 4
    assert _call_context_size_or_default(None, None) == 3

    args = SimpleNamespace(mode="daf", context_size=5)
    assert _resolve_call_mode_context(args, model_k=3, model_mode="pacbio-fiber") == (
        "daf",
        5,
    )

    args = SimpleNamespace(mode=None, context_size=None)
    assert _resolve_call_mode_context(args, model_k=4, model_mode="nanopore-fiber") == (
        "nanopore-fiber",
        4,
    )
    assert _resolve_call_mode_context(args, model_k=None, model_mode=None) == (
        "pacbio-fiber",
        3,
    )


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
    record = _build_pg_record(
        mode="daf",
        recall_nucs=True,
        phase_nrl=185,
        keep_chimeras=False,
        argv=["fiberhmm-call", "-i", "in.bam"],
    )

    assert record["PN"] == "fiberhmm-call"
    assert record["CL"] == "fiberhmm-call -i in.bam"
    assert "coord=molecular" in record["DS"]
    assert "mode=daf" in record["DS"]
    assert "recall_nucs=True" in record["DS"]
    assert "phase_nrl=185" in record["DS"]
    assert "chimera_filter=on" in record["DS"]


def test_call_pg_description_documents_settings():
    description = _call_pg_description(
        mode="daf",
        recall_nucs=False,
        phase_nrl=0,
        keep_chimeras=True,
    )

    assert "coord=molecular" in description
    assert "mode=daf" in description
    assert "recall_nucs=False" in description
    assert "phase_nrl=0" in description
    assert "chimera_filter=off" in description


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
    banner = _call_banner_text(
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

    assert "fiberhmm-call [BETA]" in banner
    assert "fused apply + recall-tfs (region-parallel)" in banner
    assert "apply model:  apply.json" in banner
    assert "recall model: (reuse apply model)" in banner
    assert "mode=daf k=4 enzyme=custom" in banner
    assert "min_llr=1.5 min_opps=3 unify_threshold=120 uplift=0.25" in banner
    assert "cores=8 io-threads=2 circular=on" in banner
