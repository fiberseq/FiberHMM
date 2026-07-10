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
    _check_daf_inputs,
    _configure_ddda_mcg,
    _resolve_apply_model,
    _resolve_recall_model,
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


def test_ddda_mode_surfaces_whole_genome_mcg_hint(capsys):
    args = SimpleNamespace(ddda_mcg=False, enzyme="ddda")
    assert _configure_ddda_mcg(args, "daf") is False
    message = capsys.readouterr().err
    assert "whole-genome DddA DAF-seq" in message
    assert "--ddda-mcg --reference ref.fa" in message
    assert "unnecessary for targeted/amplicon" in message


def test_integrated_ddda_mcg_guardrails(capsys, tmp_path):
    reference = tmp_path / "ref.fa"
    reference.write_text(">chr1\nACGTACGT\n")
    pysam.faidx(str(reference))
    valid = SimpleNamespace(
        ddda_mcg=True,
        enzyme="ddda",
        reference=str(reference),
        circular=False,
        downstream_compat=False,
        input="-",
    )
    assert _configure_ddda_mcg(valid, "daf") is True
    assert "integrated DddA mCG calling enabled" in capsys.readouterr().err

    invalid = SimpleNamespace(
        ddda_mcg=True,
        enzyme="dddb",
        reference=None,
        circular=True,
        downstream_compat=True,
        input="-",
    )
    with pytest.raises(SystemExit) as exc:
        _configure_ddda_mcg(invalid, "daf")
    assert exc.value.code == 2
    message = capsys.readouterr().err
    assert "requires --enzyme ddda" in message
    assert "requires --reference ref.fa" in message
    assert "does not support --circular" in message
    assert "cannot be combined with --downstream-compat" in message
