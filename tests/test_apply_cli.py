"""CLI characterization tests for `fiberhmm-apply` helpers."""

from types import SimpleNamespace

import pytest

from fiberhmm.cli.apply import (
    _print_processing_settings,
    _resolve_context_size,
    _resolve_mode,
    _resolve_model_path,
    _resolve_process_unmapped,
    _use_streaming_pipeline,
)


def test_apply_model_resolution_uses_custom_path():
    args = SimpleNamespace(model="/tmp/custom.json", enzyme=None, seq=None)

    assert _resolve_model_path(args) == "/tmp/custom.json"


def test_apply_model_resolution_requires_model_or_enzyme(capsys):
    args = SimpleNamespace(model=None, enzyme=None, seq=None)

    with pytest.raises(SystemExit) as exc:
        _resolve_model_path(args)

    assert exc.value.code == 1
    assert "one of --model or --enzyme must be provided" in capsys.readouterr().err


def test_apply_streaming_pipeline_selection():
    assert _use_streaming_pipeline("-", 1)
    assert _use_streaming_pipeline("input.bam", 2)
    assert not _use_streaming_pipeline("input.bam", 1)


def test_apply_process_unmapped_auto_enables_without_index(tmp_path, capsys):
    args = SimpleNamespace(
        input=str(tmp_path / "input.bam"),
        process_unmapped=False,
    )

    assert _resolve_process_unmapped(args, use_streaming=True)
    assert "no BAM index" in capsys.readouterr().out


def test_apply_process_unmapped_respects_explicit_true(tmp_path):
    args = SimpleNamespace(
        input=str(tmp_path / "input.bam"),
        process_unmapped=True,
    )

    assert _resolve_process_unmapped(args, use_streaming=False)


def test_apply_context_size_override_warns(capsys):
    args = SimpleNamespace(context_size=5)

    assert _resolve_context_size(args, model_context_size=3) == 5
    out = capsys.readouterr().out
    assert "Overriding model context size 3 with 5" in out
    assert "Context size: k=5" in out


def test_apply_mode_override_warns(capsys):
    args = SimpleNamespace(mode="daf")

    assert _resolve_mode(args, model_mode="pacbio-fiber") == "daf"
    out = capsys.readouterr().out
    assert "overrides model mode 'pacbio-fiber'" in out
    assert "Mode: daf" in out


def test_apply_mode_defaults_without_model_metadata(capsys):
    args = SimpleNamespace(mode=None)

    assert _resolve_mode(args, model_mode="unknown") == "pacbio-fiber"
    assert "defaulting to 'pacbio-fiber'" in capsys.readouterr().out


def test_apply_print_processing_settings_reports_mode_specific_options(capsys):
    args = SimpleNamespace(
        input="input.bam",
        outdir="out",
        edge_trim=10,
        min_mapq=0,
        prob_threshold=128,
        circular=True,
        scores_db=True,
        no_msps=False,
        stats=True,
    )

    _print_processing_settings(
        args,
        mode="daf",
        context_size=3,
        n_cores=4,
        msp_min_size=0,
        with_scores=True,
        db_path="scores.db",
    )

    out = capsys.readouterr().out
    assert "Mode: daf (DAF-seq deamination" in out
    assert "Circular mode: enabled" in out
    assert "Confidence scores: enabled" in out
    assert "Scores database: scores.db" in out
    assert "Strand detection: automatic" in out
