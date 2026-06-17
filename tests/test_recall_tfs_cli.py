"""CLI stability tests for `fiberhmm-recall-tfs`."""

from __future__ import annotations

import array
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

    payload = recall_tfs._make_payload(FakeRead())

    assert payload["tags"]["ML"] == bytes(tags["ML"])
    for tag in ("ns", "nl", "as", "al", "nq"):
        assert payload["tags"][tag] is tags[tag]


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


def test_recall_tfs_model_resolution_uses_custom_path():
    args = SimpleNamespace(model="/tmp/custom.json", enzyme=None, seq=None)

    assert recall_tfs._resolve_model_path(args) == "/tmp/custom.json"


def test_recall_tfs_model_resolution_requires_model_or_enzyme(capsys):
    args = SimpleNamespace(model=None, enzyme=None, seq=None)

    with pytest.raises(SystemExit) as exc:
        recall_tfs._resolve_model_path(args)

    assert exc.value.code == 1
    assert "one of --model or --enzyme must be provided" in capsys.readouterr().err


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
    monkeypatch.setattr(recall_tfs, "build_llr_tables", lambda model: (object(), object()))

    def fail_processing(*args, **kwargs):
        raise RuntimeError("processing failed")

    monkeypatch.setattr(recall_tfs, "_single_thread_loop", fail_processing)

    with pytest.raises(RuntimeError, match="processing failed"):
        recall_tfs.main()

    assert len(opened) == 2
    assert [bam.mode for bam in opened] == ["rb", "wb"]
    assert all(bam.closed for bam in opened)
