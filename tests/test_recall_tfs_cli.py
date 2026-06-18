"""CLI stability tests for `fiberhmm-recall-tfs`."""

from __future__ import annotations

import array
import json
from collections import deque
from types import SimpleNamespace

import pytest

import fiberhmm.cli.recall_tfs as recall_tfs


def test_recall_tfs_make_payload_keeps_legacy_tag_arrays_compact():
    tags = {
        "MM": "A+a,0;",
        "ML": array.array("B", [200]),
        "ns": array.array("I", [10, 200]),
        "nl": array.array("I", [80, 120]),
        "as": array.array("I", [0]),
        "al": array.array("I", [50]),
        "nq": array.array("B", [180, 190]),
        "st": "CT",
    }

    class FakeRead:
        query_sequence = "A" * 300
        is_reverse = False

        def has_tag(self, tag):
            return tag in tags

        def get_tag(self, tag):
            return tags[tag]

    extracted_tags = recall_tfs._payload_tags(FakeRead())

    assert extracted_tags["ML"] == bytes(tags["ML"])
    for tag in ("ns", "nl", "as", "al", "nq"):
        assert extracted_tags[tag] is tags[tag]

    payload = recall_tfs._make_payload(FakeRead())
    assert payload["tags"]["ML"] == bytes(tags["ML"])
    for tag in ("ns", "nl", "as", "al", "nq"):
        assert payload["tags"][tag] is tags[tag]


def test_recall_tfs_daf_md_fallback_predicate():
    assert recall_tfs._needs_daf_md_result(
        "ACGT",
        {},
        "daf",
    ) is True
    assert recall_tfs._needs_daf_md_result(
        "ARYT",
        {},
        "daf",
    ) is False
    assert recall_tfs._needs_daf_md_result(
        "ACGT",
        {"MM": "C+m,0;", "ML": b"\xff"},
        "daf",
    ) is False
    assert recall_tfs._needs_daf_md_result(
        "ACGT",
        {},
        "pacbio-fiber",
    ) is False
    assert recall_tfs._needs_daf_md_result(
        "",
        {},
        "daf",
    ) is False


def test_recall_tfs_short_v2_nuc_count_uses_open_threshold():
    assert recall_tfs._short_v2_nuc_count({}, 90) == 0
    assert recall_tfs._short_v2_nuc_count({"ns": [1], "nl": [0, 1, 89, 90]}, 90) == 2


def test_recall_tfs_reverse_read_preserves_nq(monkeypatch):
    # A reverse read: stored ns/nl/nq are molecular, but recall_read returns kept
    # nucs in SEQ frame. The nq lookup must be built in SEQ frame too, else the
    # kept nuc misses its original score and gets 0 (regression guard).
    import numpy as np

    monkeypatch.setitem(recall_tfs._WORKER, "llr_hit", np.zeros(1))
    monkeypatch.setitem(recall_tfs._WORKER, "llr_miss", np.zeros(1))
    monkeypatch.setitem(recall_tfs._WORKER, "mode", "m6a")
    monkeypatch.setitem(recall_tfs._WORKER, "k", 3)
    monkeypatch.setitem(recall_tfs._WORKER, "min_llr", 5.0)
    monkeypatch.setitem(recall_tfs._WORKER, "min_opps", 3)
    monkeypatch.setitem(recall_tfs._WORKER, "unify_threshold", 85)

    L = 1000
    payload = {
        "seq": "A" * L,                       # no MM/ML -> recall_read pass-through
        "is_reverse": True,
        "tags": {"ns": [100], "nl": [50], "nq": [77]},   # molecular nuc (100,50)
    }
    (tf_calls, kept_nucs, msps, nq_for_kept), _ = recall_tfs._process_payload_record(payload)
    # molecular (100,50) flips to SEQ (850,50) on a 1000 bp reverse read
    assert kept_nucs == [(850, 50)]
    assert nq_for_kept == [77]


def test_recall_tfs_kept_nuc_nq_from_legacy_defaults_missing_matches():
    read = SimpleNamespace(is_reverse=False, query_sequence="A" * 200)

    assert recall_tfs._kept_nuc_nq_from_legacy(
        {"ns": [10], "nl": [20], "nq": [55]},
        read,
        [(10, 20), (100, 5)],
    ) == [55, 0]
    assert recall_tfs._kept_nuc_nq_from_legacy(
        {"ns": [10], "nl": [20]},
        read,
        [(10, 20)],
    ) is None


def test_recall_tfs_payload_chunk_counts_per_read_failures(monkeypatch):
    def fake_process(payload):
        if payload == "bad":
            raise RuntimeError("bad read")
        return payload.upper(), {"v2": 1, "tf": 2, "demoted": 3, "failed": 0}

    monkeypatch.setattr(recall_tfs, "_process_payload_record", fake_process)

    results, stats = recall_tfs._process_payload_chunk(["ok", "bad", "next"])

    assert results == ["OK", None, "NEXT"]
    assert stats == {"v2": 2, "tf": 4, "demoted": 6, "failed": 1}


def test_recall_tfs_stats_helpers_accumulate_summary():
    total = recall_tfs._new_stats()

    recall_tfs._add_stats(total, {"v2": 1, "tf": 2, "demoted": 3, "failed": 4})
    recall_tfs._add_stats(total, {"tf": 5})

    assert total == {"v2": 1, "tf": 7, "demoted": 3, "failed": 4}
    assert recall_tfs._stats_tuple(9, total) == (9, 1, 7, 3, 4)


def test_recall_tfs_record_recall_stats_counts_demoted_short_nucs():
    stats = recall_tfs._new_stats()

    recall_tfs._record_recall_stats(
        stats,
        tf_calls=["tf-a", "tf-b"],
        kept_nucs=[(0, 10), (100, 90)],
        v2_short_count=3,
        unify_threshold=90,
    )

    assert stats["tf"] == 2
    assert stats["demoted"] == 2


def test_recall_tfs_record_v2_input_stats_counts_short_nucs():
    stats = recall_tfs._new_stats()

    assert recall_tfs._record_v2_input_stats(stats, {}, 90) == 0
    assert stats["v2"] == 0

    assert recall_tfs._record_v2_input_stats(
        stats,
        {"ns": [1, 2, 3], "nl": [10, 90, 0]},
        90,
    ) == 1
    assert stats["v2"] == 1


def test_recall_tfs_write_recall_result_passes_through_null_result(monkeypatch):
    reads = [
        SimpleNamespace(query_name="annotated"),
        SimpleNamespace(query_name="failed"),
    ]
    written = []
    applied = []

    class FakeOut:
        def write(self, read):
            written.append(read.query_name)

    monkeypatch.setattr(
        recall_tfs,
        "_apply_result",
        lambda read, result, also_write_legacy, downstream_compat: applied.append(
            (read.query_name, result, also_write_legacy, downstream_compat)
        ),
    )

    out = FakeOut()
    recall_tfs._write_recall_result(reads[0], "result", out, True, False)
    recall_tfs._write_recall_result(reads[1], None, out, True, False)

    assert written == ["annotated", "failed"]
    assert applied == [("annotated", "result", True, False)]


def test_recall_tfs_submit_chunk_enqueues_worker_future():
    pending = deque()

    class FakePool:
        def apply_async(self, func, args):
            assert func is recall_tfs._process_payload_chunk
            assert args == (["payload"],)
            return "future"

    recall_tfs._submit_recall_chunk(
        FakePool(),
        pending,
        [SimpleNamespace(query_name="read")],
        ["payload"],
    )

    reads, future = pending.popleft()
    assert [read.query_name for read in reads] == ["read"]
    assert future == "future"


def test_recall_tfs_drain_chunk_writes_results_and_returns_stats(monkeypatch):
    reads = [SimpleNamespace(query_name="a"), SimpleNamespace(query_name="b")]
    stats = {"v2": 1, "tf": 2, "demoted": 3, "failed": 0}
    calls = []

    class FakeFuture:
        def get(self):
            return ["result-a", None], stats

    pending = deque([(reads, FakeFuture())])
    monkeypatch.setattr(
        recall_tfs,
        "_write_recall_result",
        lambda read, result, out, legacy, compat: calls.append(
            (read.query_name, result, out, legacy, compat)
        ),
    )

    assert recall_tfs._drain_recall_chunk(
        pending, "bam-out", True, False,
    ) == (2, stats)
    assert list(pending) == []
    assert calls == [
        ("a", "result-a", "bam-out", True, False),
        ("b", None, "bam-out", True, False),
    ]


def test_recall_tfs_read_sequence_length_handles_missing_sequence():
    assert recall_tfs._read_sequence_length(SimpleNamespace(query_sequence="ACGT")) == 4
    assert recall_tfs._read_sequence_length(SimpleNamespace(query_sequence=None)) == 0


def test_recall_tfs_model_resolution_uses_custom_path():
    args = SimpleNamespace(model="/tmp/custom.json", enzyme=None, seq=None)

    assert recall_tfs._resolve_model_path(args) == "/tmp/custom.json"


def test_recall_tfs_model_resolution_requires_model_or_enzyme(capsys):
    args = SimpleNamespace(model=None, enzyme=None, seq=None)

    with pytest.raises(SystemExit) as exc:
        recall_tfs._resolve_model_path(args)

    assert exc.value.code == 1
    assert "one of --model or --enzyme must be provided" in capsys.readouterr().err


def test_recall_tfs_setup_helpers_resolve_cores_defaults_and_legacy(monkeypatch):
    monkeypatch.setattr(recall_tfs.mp, "cpu_count", lambda: 12)
    monkeypatch.setattr(
        recall_tfs,
        "ENZYME_PRESETS",
        {"test-enzyme": {"min_llr": 2.5, "emission_uplift": 1.2}},
    )

    assert recall_tfs._resolve_cores(0) == 12
    assert recall_tfs._resolve_cores(-3) == 1
    assert recall_tfs._resolve_cores(4) == 4

    preset_args = SimpleNamespace(
        enzyme="test-enzyme",
        min_llr=None,
        emission_uplift=None,
        downstream_compat=False,
        no_legacy_tags=True,
    )
    assert recall_tfs._resolve_recall_defaults(preset_args) == (2.5, 1.2)
    assert recall_tfs._also_write_legacy(preset_args) is False

    override_args = SimpleNamespace(
        enzyme="test-enzyme",
        min_llr=8.0,
        emission_uplift=1.5,
        downstream_compat=True,
        no_legacy_tags=True,
    )
    assert recall_tfs._resolve_recall_defaults(override_args) == (8.0, 1.5)
    assert recall_tfs._also_write_legacy(override_args) is True


def test_recall_tfs_load_model_config_uses_metadata_and_overrides(monkeypatch):
    monkeypatch.setattr(
        recall_tfs,
        "load_model_with_metadata",
        lambda path: ("model", 5, "model-mode"),
    )

    args = SimpleNamespace(mode=None, context_size=None)

    assert recall_tfs._load_recall_model_config("/tmp/model.json", args) == (
        "model",
        "model-mode",
        5,
    )

    args = SimpleNamespace(mode=" arg-mode ", context_size="7")

    assert recall_tfs._load_recall_model_config("/tmp/model.json", args) == (
        "model",
        "arg-mode",
        7,
    )

    args = SimpleNamespace(mode=" ", context_size=0)
    assert recall_tfs._load_recall_model_config("/tmp/model.json", args) == (
        "model",
        "model-mode",
        0,
    )


def test_recall_tfs_load_model_config_falls_back_to_json_metadata(monkeypatch):
    monkeypatch.setattr(
        recall_tfs,
        "load_model_with_metadata",
        lambda path: ("model", None, None),
    )
    monkeypatch.setattr(
        recall_tfs,
        "_resolve_model_metadata",
        lambda path: ("fallback-mode", 3),
    )

    args = SimpleNamespace(mode=None, context_size=None)

    assert recall_tfs._load_recall_model_config("/tmp/model.json", args) == (
        "model",
        "fallback-mode",
        3,
    )


def test_recall_tfs_json_metadata_fallback_uses_case_insensitive_suffix(tmp_path):
    model_path = tmp_path / "model.JSON"
    model_path.write_text(
        json.dumps({"mode": "daf", "context_size": 5}),
        encoding="utf-8",
    )

    assert recall_tfs._resolve_model_metadata(model_path) == ("daf", 5)


def test_recall_tfs_json_metadata_fallback_keeps_defaults_for_nulls(tmp_path):
    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps({"mode": None, "context_size": None}),
        encoding="utf-8",
    )

    assert recall_tfs._resolve_model_metadata(str(model_path)) == ("pacbio-fiber", 3)


def test_recall_tfs_run_processing_dispatches_single_and_parallel(monkeypatch):
    args = SimpleNamespace(
        min_opps=3,
        unify_threshold=90,
        downstream_compat=False,
        max_reads=10,
        chunk_size=256,
    )
    calls = []

    def fake_single(*call_args):
        calls.append(("single", call_args))
        return (1, 2, 3, 4, 5)

    def fake_parallel(*call_args):
        calls.append(("parallel", call_args))
        return (6, 7, 8, 9, 10)

    monkeypatch.setattr(recall_tfs, "_single_thread_loop", fake_single)
    monkeypatch.setattr(recall_tfs, "_parallel_loop", fake_parallel)

    common_args = (
        "bam-in",
        "bam-out",
        "header",
        "hit",
        "miss",
        "daf",
        3,
        4.5,
        True,
    )
    assert recall_tfs._run_recall_processing(args, 1, *common_args) == (
        1, 2, 3, 4, 5,
    )
    assert recall_tfs._run_recall_processing(args, 4, *common_args) == (
        6, 7, 8, 9, 10,
    )
    assert calls[0] == (
        "single",
        (
            "bam-in", "bam-out", "header", "hit", "miss", "daf", 3,
            4.5, 3, 90, True, False, 10,
        ),
    )
    assert calls[1] == (
        "parallel",
        (
            "bam-in", "bam-out", "header", "hit", "miss", "daf", 3,
            4.5, 3, 90, True, False, 10, 4, 256,
        ),
    )


def test_recall_tfs_print_summary_reports_failures(capsys):
    recall_tfs._print_recall_summary(10, 4, 3, 2, 1)

    err = capsys.readouterr().err
    assert "processed 10 reads" in err
    assert "4 carried v2 tags" in err
    assert "3 TF calls emitted" in err
    assert "2 v2 short nucs demoted" in err
    assert "warning: 1 reads passed through unchanged" in err


def test_recall_tfs_single_thread_passes_failed_reads_through(monkeypatch):
    reads = [
        SimpleNamespace(query_name="ok", query_sequence="AAAA"),
        SimpleNamespace(query_name="bad", query_sequence="CCCC"),
    ]
    written = []
    applied = []

    class FakeOut:
        def write(self, read):
            written.append(read)

    def fake_process(payload):
        if payload == "bad":
            raise RuntimeError("bad read")
        return "result", {"v2": 1, "tf": 2, "demoted": 3, "failed": 0}

    monkeypatch.setattr(
        recall_tfs,
        "_make_payload",
        lambda read, mode=None: read.query_name,
    )
    monkeypatch.setattr(recall_tfs, "_process_payload_record", fake_process)
    monkeypatch.setattr(
        recall_tfs,
        "_apply_result",
        lambda read, result, also_write_legacy, downstream_compat: applied.append(
            (read.query_name, result)
        ),
    )

    assert recall_tfs._single_thread_loop(
        reads,
        FakeOut(),
        None,
        None,
        None,
        "pacbio-fiber",
        3,
        5.0,
        3,
        90,
        True,
        False,
        0,
    ) == (2, 1, 2, 3, 1)
    assert written == reads
    assert applied == [("ok", "result")]


def test_recall_tfs_closes_bams_when_processing_fails(monkeypatch):
    opened = []

    class FakeBam:
        def __init__(self, path, mode, **kwargs):
            self.path = path
            self.mode = mode
            self.header = {"HD": {"VN": "1.6"}}
            self.closed = False
            opened.append(self)

        def close(self):
            self.closed = True

    args = SimpleNamespace(
        in_bam="in.bam",
        out_bam="out.bam",
        model="/tmp/model.json",
        enzyme=None,
        seq=None,
        downstream_compat=False,
        cores=1,
        min_llr=None,
        emission_uplift=None,
        unify_threshold=90,
        no_legacy_tags=False,
        min_opps=3,
        io_threads=1,
        mode=None,
        context_size=None,
        max_reads=0,
        chunk_size=1024,
    )

    monkeypatch.setattr(recall_tfs, "parse_args", lambda: args)
    monkeypatch.setattr(recall_tfs.pysam, "AlignmentFile", FakeBam)
    monkeypatch.setattr(
        recall_tfs,
        "load_model_with_metadata",
        lambda path: (object(), 3, "pacbio-fiber"),
    )
    monkeypatch.setattr(
        recall_tfs,
        "_build_recall_llr_tables",
        lambda model, uplift: (object(), object()),
    )

    def fail_processing(*args, **kwargs):
        raise RuntimeError("processing failed")

    monkeypatch.setattr(recall_tfs, "_single_thread_loop", fail_processing)

    with pytest.raises(RuntimeError, match="processing failed"):
        recall_tfs.main()

    assert len(opened) == 2
    assert [bam.mode for bam in opened] == ["rb", "wb"]
    assert all(bam.closed for bam in opened)
