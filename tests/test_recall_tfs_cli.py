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


def test_recall_tfs_payload_chunk_counts_per_read_failures(monkeypatch):
    def fake_process(payload):
        if payload == "bad":
            raise RuntimeError("bad read")
        return payload.upper(), {"v2": 1, "tf": 2, "demoted": 3, "failed": 0}

    monkeypatch.setattr(recall_tfs, "_process_payload_record", fake_process)

    results, stats = recall_tfs._process_payload_chunk(["ok", "bad", "next"])

    assert results == ["OK", None, "NEXT"]
    assert stats == {"v2": 2, "tf": 4, "demoted": 6, "failed": 1}


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
