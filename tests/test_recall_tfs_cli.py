"""CLI stability tests for `fiberhmm-recall-tfs`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import fiberhmm.cli.recall_tfs as recall_tfs


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
