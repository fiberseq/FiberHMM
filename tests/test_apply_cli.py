"""CLI characterization tests for `fiberhmm-apply` helpers."""

import sqlite3
from types import SimpleNamespace

import pytest

from fiberhmm.cli.apply import (
    _apply_processing_kwargs,
    _context_size_message,
    _dataset_name,
    _ddda_notice_needed,
    _load_apply_model_with_summary,
    _load_training_read_ids,
    _finalize_apply_outputs,
    _print_apply_done,
    _print_numba_status,
    _print_processing_settings,
    _print_processing_result,
    _print_ddda_two_pass_notice,
    _print_region_filter_settings,
    _msp_output_message,
    _resolve_apply_cores,
    _resolve_apply_runtime,
    _resolve_chroms_set,
    _resolve_context_size,
    _resolve_mode,
    _resolve_model_path,
    _resolve_output_bam,
    _resolve_process_unmapped,
    _resolve_scores_db_path,
    _scores_enabled,
    _print_scores_db_summary,
    _prepare_apply_io,
    _processing_status_message,
    _run_apply_processing,
    _strand_detection_message,
    _use_streaming_pipeline,
    _stats_output_prefix,
    _scores_db_counts,
    _write_apply_stats,
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


def test_apply_load_model_with_summary_reports_parameters(monkeypatch, capsys):
    model = SimpleNamespace(startprob_=[0.4, 0.6], transmat_=[[0.9, 0.1]])

    monkeypatch.setattr(
        "fiberhmm.cli.apply.load_model_with_metadata",
        lambda path: (model, 3, "pacbio-fiber"),
    )

    assert _load_apply_model_with_summary("model.json") == (
        model, 3, "pacbio-fiber",
    )

    out = capsys.readouterr().out
    assert "Loading model from model.json" in out
    assert "Model loaded successfully" in out
    assert "Start probs: [0.4, 0.6]" in out
    assert "Transition matrix:" in out


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


def test_apply_resolve_cores_auto_and_clamps(capsys):
    assert _resolve_apply_cores(-3) == 1

    assert _resolve_apply_cores(0) >= 1
    assert "Auto-detected" in capsys.readouterr().out


def test_prepare_apply_io_resolves_cores_and_creates_output_dir(monkeypatch, tmp_path):
    from fiberhmm.cli import apply

    monkeypatch.setattr(apply, "_resolve_apply_cores", lambda cores: 7)
    outdir = tmp_path / "apply-out"

    stdout_mode, n_cores = _prepare_apply_io(
        SimpleNamespace(outdir=str(outdir), cores=0)
    )

    assert stdout_mode is False
    assert n_cores == 7
    assert outdir.is_dir()


def test_apply_dataset_name():
    assert _dataset_name("-") == "stdin"
    assert _dataset_name("/tmp/sample.bam") == "sample"
    assert _dataset_name("/tmp/sample.cram") == "sample.cram"


def test_apply_output_and_scores_paths():
    args = SimpleNamespace(outdir="/tmp/out", scores_db=True)

    assert _resolve_output_bam(args, "sample", stdout_mode=False) == (
        "/tmp/out/sample_footprints.bam"
    )
    assert _resolve_output_bam(args, "sample", stdout_mode=True) == "-"
    assert _resolve_scores_db_path(args, "sample") == "/tmp/out/sample_scores.db"

    args.scores_db = False
    assert _resolve_scores_db_path(args, "sample") is None
    assert _stats_output_prefix("/tmp/out", "sample") == "/tmp/out/sample_footprints"


@pytest.mark.parametrize(
    ("scores", "scores_db", "expected"),
    [
        (False, False, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ],
)
def test_apply_scores_enabled(scores, scores_db, expected):
    assert _scores_enabled(
        SimpleNamespace(scores=scores, scores_db=scores_db)
    ) is expected


def test_apply_ddda_notice_detection_and_output(capsys):
    assert _ddda_notice_needed("/models/ddda_nuc.json", None)
    assert _ddda_notice_needed("/models/custom.json", "ddda")
    assert not _ddda_notice_needed("/models/custom.json", "hia5")

    _print_ddda_two_pass_notice("/models/custom.json", "hia5")
    assert capsys.readouterr().err == ""

    _print_ddda_two_pass_notice("/models/ddda_nuc.json")
    err = capsys.readouterr().err
    assert "DddA model detected" in err
    assert "fiberhmm-recall-tfs" in err


def test_apply_context_size_override_warns(capsys):
    args = SimpleNamespace(context_size=5)

    assert _context_size_message(5) == "k=5 (11-mer)"
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


def test_apply_strand_detection_message_by_mode():
    assert _strand_detection_message("daf") == (
        "Strand detection: automatic (C=+, G=-)"
    )
    assert _strand_detection_message("nanopore-fiber") == (
        "Strand detection: none (A-centered only)"
    )
    assert _strand_detection_message("pacbio-fiber") is None


def test_apply_msp_output_message():
    assert _msp_output_message(False, 0) == "MSP min size: 0 bp"
    assert _msp_output_message(False, 75) == "MSP min size: 75 bp"
    assert _msp_output_message(True, 75) == "MSP output: disabled (--no-msps)"


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


def test_apply_load_training_read_ids(tmp_path, capsys):
    assert _load_training_read_ids(None) == set()

    train_reads = tmp_path / "train_reads.tsv"
    train_reads.write_text("rid\nread1\nread2\nread1\n")

    assert _load_training_read_ids(str(train_reads)) == {"read1", "read2"}
    assert "Excluding 2 training reads" in capsys.readouterr().out


def test_apply_chrom_filter_helpers(capsys):
    assert _resolve_chroms_set(None) is None
    assert _resolve_chroms_set([]) is None
    assert _resolve_chroms_set(["chr2", "chr1", "chr2"]) == {"chr1", "chr2"}

    args = SimpleNamespace(skip_scaffolds=True)
    _print_region_filter_settings(args, {"chr2", "chr1"})

    out = capsys.readouterr().out
    assert "Processing only chromosomes: chr1, chr2" in out
    assert "Skipping scaffold/contig chromosomes" in out


def test_apply_scores_db_summary_helpers(tmp_path, capsys):
    db_path = tmp_path / "scores.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE reads (id INTEGER)")
        conn.execute("CREATE TABLE footprints (id INTEGER)")
        conn.executemany("INSERT INTO reads VALUES (?)", [(1,), (2,)])
        conn.executemany("INSERT INTO footprints VALUES (?)", [(1,), (2,), (3,)])
        conn.commit()
    finally:
        conn.close()

    assert _scores_db_counts(str(db_path)) == (2, 3)

    _print_scores_db_summary(str(db_path))
    out = capsys.readouterr().out
    assert f"Scores DB: {db_path}" in out
    assert "2 reads, 3 footprints" in out

    _print_scores_db_summary(str(tmp_path / "missing.db"))
    assert capsys.readouterr().out == ""


def test_processing_status_message_formats_read_limit():
    assert _processing_status_message(None) == "Processing BAM..."
    assert _processing_status_message(0) == "Processing BAM..."
    assert _processing_status_message(12500) == (
        "Processing BAM (limited to 12,500 reads)..."
    )


def test_apply_reporting_helpers_route_messages(capsys):
    _print_numba_status(True)
    assert "Numba JIT: enabled" in capsys.readouterr().out

    _print_numba_status(False)
    assert "Numba JIT: disabled" in capsys.readouterr().out

    _print_processing_result(1200, 34, "out.bam", stdout_mode=False)
    out = capsys.readouterr().out
    assert "Processed 1,200 reads -> 34 with footprints" in out
    assert "BAM: out.bam" in out
    assert "BAM index: out.bam.bai" in out

    _print_processing_result(5, 2, "-", stdout_mode=True)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Processed 5 reads -> 2 with footprints" in captured.err

    _print_apply_done(False, "out.bam")
    out = capsys.readouterr().out
    assert "Done!" in out
    assert "fiberhmm-extract-tags -i out.bam" in out

    _print_apply_done(True, "-")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Done!" in captured.err


def test_apply_stats_writer_uses_expected_prefix(monkeypatch, capsys):
    calls = []

    class Stats:
        def write_summary(self, path):
            calls.append(("summary", path))

        def plot_distributions(self, prefix):
            calls.append(("plot", prefix))

    def fake_collect(output_bam, **kwargs):
        calls.append(("collect", output_bam, kwargs))
        return Stats()

    monkeypatch.setattr("fiberhmm.cli.apply.collect_stats_from_bam", fake_collect)
    args = SimpleNamespace(outdir="/tmp/out", stats_sample=100, stats_seed=7)

    _write_apply_stats("out.bam", args, "sample", with_scores=True)

    assert calls == [
        ("collect", "out.bam", {
            "n_samples": 100,
            "seed": 7,
            "with_scores": True,
        }),
        ("summary", "/tmp/out/sample_footprints_stats.txt"),
        ("plot", "/tmp/out/sample_footprints"),
    ]
    assert "Stats: /tmp/out/sample_footprints_stats.txt" in capsys.readouterr().out


def test_apply_processing_kwargs_preserve_pipeline_arguments():
    args = SimpleNamespace(
        input="input.bam",
        edge_trim=10,
        circular=True,
        nuc_min_size=85,
        min_mapq=20,
        prob_threshold=128,
        min_read_length=1000,
        max_reads=500,
        debug_timing=True,
        region_size=1_000_000,
        skip_scaffolds=True,
        primary=True,
        output_posteriors="post.h5",
        no_msps=False,
        io_threads=3,
        chunk_size=2000,
    )

    assert _apply_processing_kwargs(
        args,
        output_bam="out.bam",
        model_path="model.json",
        train_rids={"read-a"},
        mode="daf",
        context_size=3,
        msp_min_size=0,
        with_scores=True,
        n_cores=4,
        chroms_set={"chr1"},
        use_streaming=True,
        process_unmapped=True,
    ) == {
        "input_bam": "input.bam",
        "output_bam": "out.bam",
        "model_or_path": "model.json",
        "train_rids": {"read-a"},
        "edge_trim": 10,
        "circular": True,
        "mode": "daf",
        "context_size": 3,
        "msp_min_size": 0,
        "nuc_min_size": 85,
        "min_mapq": 20,
        "prob_threshold": 128,
        "min_read_length": 1000,
        "with_scores": True,
        "n_cores": 4,
        "max_reads": 500,
        "debug_timing": True,
        "region_parallel": False,
        "region_size": 1_000_000,
        "skip_scaffolds": True,
        "chroms": {"chr1"},
        "primary_only": True,
        "output_posteriors": "post.h5",
        "write_msps": True,
        "io_threads": 3,
        "streaming_pipeline": True,
        "chunk_size": 2000,
        "process_unmapped": True,
    }


def test_run_apply_processing_delegates_pipeline_and_reports_result(
    monkeypatch,
    capsys,
):
    from fiberhmm.cli import apply

    args = SimpleNamespace(
        input="input.bam",
        edge_trim=10,
        circular=True,
        nuc_min_size=85,
        min_mapq=20,
        prob_threshold=128,
        min_read_length=1000,
        max_reads=500,
        debug_timing=True,
        region_size=1_000_000,
        skip_scaffolds=True,
        primary=True,
        output_posteriors="post.h5",
        no_msps=False,
        io_threads=3,
        chunk_size=2000,
    )
    calls = []
    monkeypatch.setattr(
        apply,
        "process_bam_for_footprints",
        lambda **kwargs: calls.append(("process", kwargs)) or (10, 4),
    )
    monkeypatch.setattr(
        apply,
        "_print_processing_result",
        lambda *args: calls.append(("result", args)),
    )

    assert _run_apply_processing(
        args,
        output_bam="out.bam",
        model_path="model.json",
        train_rids={"read-a"},
        mode="daf",
        context_size=3,
        msp_min_size=0,
        with_scores=True,
        n_cores=4,
        chroms_set={"chr1"},
        use_streaming=True,
        process_unmapped=True,
        stdout_mode=False,
    ) == (10, 4)

    assert calls[0][0] == "process"
    assert calls[0][1]["input_bam"] == "input.bam"
    assert calls[0][1]["output_bam"] == "out.bam"
    assert calls[0][1]["process_unmapped"] is True
    assert calls[1] == ("result", (10, 4, "out.bam", False))
    assert "Processing BAM (limited to 500 reads)" in capsys.readouterr().out


def test_resolve_apply_runtime_wires_setup_outputs(monkeypatch):
    from fiberhmm.cli import apply

    args = SimpleNamespace(
        input="input.bam",
        msp_min_size=None,
        train_reads="train.txt",
        chroms=["chr1"],
        mode=None,
    )
    calls = []

    monkeypatch.setattr(apply, "_resolve_model_path", lambda got_args: "model.json")
    monkeypatch.setattr(
        apply,
        "_load_apply_model_with_summary",
        lambda path: ("model", 4, "model-mode"),
    )
    monkeypatch.setattr(
        apply,
        "_print_ddda_two_pass_notice",
        lambda model_path, enzyme: calls.append(("ddda", model_path, enzyme)),
    )
    monkeypatch.setattr(
        apply,
        "_print_numba_status",
        lambda has_numba: calls.append(("numba", has_numba)),
    )
    monkeypatch.setattr(apply, "_resolve_context_size", lambda got_args, k: 5)
    monkeypatch.setattr(apply, "_resolve_mode", lambda got_args, mode: "daf")
    monkeypatch.setattr(apply, "_load_training_read_ids", lambda path: {"read-a"})
    monkeypatch.setattr(apply, "_dataset_name", lambda path: "sample")
    monkeypatch.setattr(apply, "_scores_enabled", lambda got_args: True)
    monkeypatch.setattr(
        apply,
        "_resolve_scores_db_path",
        lambda got_args, dataset: "scores.db",
    )
    monkeypatch.setattr(
        apply,
        "_print_processing_settings",
        lambda *call_args: calls.append(("settings", call_args)),
    )
    monkeypatch.setattr(apply, "_resolve_chroms_set", lambda chroms: {"chr1"})
    monkeypatch.setattr(
        apply,
        "_print_region_filter_settings",
        lambda got_args, chroms: calls.append(("regions", chroms)),
    )
    monkeypatch.setattr(
        apply,
        "_use_streaming_pipeline",
        lambda input_bam, n_cores: True,
    )
    monkeypatch.setattr(
        apply,
        "_resolve_process_unmapped",
        lambda got_args, use_streaming: False,
    )
    monkeypatch.setattr(
        apply,
        "_resolve_output_bam",
        lambda got_args, dataset, stdout_mode: "out.bam",
    )

    runtime = _resolve_apply_runtime(args, n_cores=4, stdout_mode=False)

    assert runtime == {
        "model_path": "model.json",
        "train_rids": {"read-a"},
        "mode": "daf",
        "context_size": 5,
        "msp_min_size": 0,
        "with_scores": True,
        "dataset": "sample",
        "db_path": "scores.db",
        "chroms_set": {"chr1"},
        "use_streaming": True,
        "process_unmapped": False,
        "output_bam": "out.bam",
    }
    assert args.mode == "daf"
    assert calls[0] == ("ddda", "model.json", None)
    assert calls[2][0] == "settings"
    assert calls[3] == ("regions", {"chr1"})


def test_finalize_apply_outputs_writes_stats_only_for_file_outputs(monkeypatch):
    from fiberhmm.cli import apply

    calls = []
    monkeypatch.setattr(
        apply,
        "_write_apply_stats",
        lambda *args: calls.append(("stats", args)),
    )
    monkeypatch.setattr(
        apply,
        "_print_scores_db_summary",
        lambda db_path: calls.append(("scores", db_path)),
    )
    monkeypatch.setattr(
        apply,
        "_print_apply_done",
        lambda stdout_mode, output_bam: calls.append(
            ("done", stdout_mode, output_bam)
        ),
    )
    args = SimpleNamespace(stats=True)

    _finalize_apply_outputs(args, "out.bam", "sample", True, "scores.db", False)
    _finalize_apply_outputs(args, "-", "sample", True, "scores.db", True)

    assert calls == [
        ("stats", ("out.bam", args, "sample", True)),
        ("scores", "scores.db"),
        ("done", False, "out.bam"),
        ("scores", "scores.db"),
        ("done", True, "-"),
    ]
