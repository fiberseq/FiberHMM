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


# --------------------------------------------------------------------------- #
#  input coordinate-frame auto-detection (legacy v1.0 query-frame BAMs)        #
# --------------------------------------------------------------------------- #


def test_resolve_input_molecular_frame_auto_detects_marker():
    mol = {"CO": ["fiberhmm:coord=molecular"]}
    fused = {"PG": [{
        "ID": "fiberhmm-call",
        "DS": "FiberHMM fused apply+recall; coord=molecular; mode=daf",
    }]}
    legacy = {"CO": []}
    R = recall_tfs._resolve_input_molecular_frame
    # auto: marker present -> molecular; absent -> legacy query-frame
    assert R(SimpleNamespace(input_frame="auto"), mol) is True
    assert R(SimpleNamespace(input_frame="auto"), fused) is True
    assert R(SimpleNamespace(input_frame="auto"), legacy) is False
    # explicit overrides win regardless of the marker
    assert R(SimpleNamespace(input_frame="molecular"), legacy) is True
    assert R(SimpleNamespace(input_frame="query"), mol) is False


def test_recall_query_frame_input_skips_reverse_flip(monkeypatch):
    # CORRECTNESS GUARD for legacy/v1.0 BAMs. ns/nl already in SEQ (query) frame
    # must NOT be flipped again on reverse reads -- otherwise the nuc lands on
    # accessible DNA (open promoters fill with spurious nucleosome).
    import numpy as np

    for key, val in {
        "llr_hit": np.zeros(1), "llr_miss": np.zeros(1), "mode": "m6a",
        "k": 3, "min_llr": 5.0, "min_opps": 3, "unify_threshold": 85,
        "nuc_cfg": None,
    }.items():
        monkeypatch.setitem(recall_tfs._WORKER, key, val)

    L = 1000
    payload = {
        "seq": "A" * L,                       # no MM/ML -> recall_read pass-through
        "is_reverse": True,
        "tags": {"ns": [100], "nl": [50], "nq": [77]},   # already SEQ-frame (v1.0)
    }

    # Molecular (default): reverse read flips (100,50) -> (850,50).
    monkeypatch.setitem(recall_tfs._WORKER, "input_molecular_frame", True)
    (_, kept_mol, _, _), _ = recall_tfs._process_payload_record(payload)
    assert kept_mol == [(850, 50)]

    # Query-frame input: no flip, the nuc stays put at (100,50).
    monkeypatch.setitem(recall_tfs._WORKER, "input_molecular_frame", False)
    (_, kept_query, _, nq_for_kept), _ = recall_tfs._process_payload_record(payload)
    assert kept_query == [(100, 50)]
    assert nq_for_kept == [77]   # nq lookup keyed in the same (seq) frame


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

    monkeypatch.setattr(recall_tfs, "parse_args",
                        lambda default_recall_nucs=False: args)
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


# --------------------------------------------------------------------------- #
#  --recall-nucs (nucleosome recall on an already apply-tagged BAM)            #
# --------------------------------------------------------------------------- #


def test_parse_phase_nrl_option_off_auto_fixed():
    assert recall_tfs._parse_phase_nrl_option("off") == ("off", 0)
    assert recall_tfs._parse_phase_nrl_option("0") == ("off", 0)
    assert recall_tfs._parse_phase_nrl_option("") == ("off", 0)
    assert recall_tfs._parse_phase_nrl_option("auto") == ("auto", 0)
    assert recall_tfs._parse_phase_nrl_option("AUTO") == ("auto", 0)
    assert recall_tfs._parse_phase_nrl_option("185") == ("fixed", 185)
    assert recall_tfs._parse_phase_nrl_option("banana") == ("auto", 0)


def test_resolve_phase_nrl_off_when_recall_nucs_disabled():
    args = SimpleNamespace(recall_nucs=False, phase_nrl="auto", in_bam="x.bam")
    assert recall_tfs._resolve_recall_nucs_phase_nrl(args) == 0


def test_resolve_phase_nrl_fixed_value():
    args = SimpleNamespace(recall_nucs=True, phase_nrl="200",
                           in_bam="x.bam", nuc_min_size=85)
    assert recall_tfs._resolve_recall_nucs_phase_nrl(args) == 200


def test_resolve_phase_nrl_auto_stdin_uses_anchor():
    args = SimpleNamespace(recall_nucs=True, phase_nrl="auto",
                           in_bam="-", nuc_min_size=85)
    assert recall_tfs._resolve_recall_nucs_phase_nrl(args) == 185


def test_resolve_phase_nrl_auto_estimates_from_tags(monkeypatch):
    args = SimpleNamespace(recall_nucs=True, phase_nrl="auto",
                           in_bam="sample.bam", nuc_min_size=85)
    seen = {}

    def fake_estimate(path, nuc_min_size, **kw):
        seen["path"] = path
        seen["nuc_min_size"] = nuc_min_size
        return {"nrl": 187, "source": "estimated", "n_pairs": 999, "n_reads": 50}

    monkeypatch.setattr(recall_tfs, "_estimate_phase_nrl_from_tags", fake_estimate)
    assert recall_tfs._resolve_recall_nucs_phase_nrl(args) == 187
    assert seen == {"path": "sample.bam", "nuc_min_size": 85}


def test_worker_init_sets_nuc_cfg():
    state = {}
    cfg = recall_tfs._NucCfg(
        recall_nucs=True, split_min_llr=4.0, split_min_opps=3,
        nuc_min_size=85, msp_min_size=0, phase_nrl=185,
    )
    import fiberhmm.cli.recall_tfs as rt
    saved = rt._WORKER
    rt._WORKER = state
    try:
        rt._worker_init("h", "m", "daf", 3, 5.0, 3, 90, cfg)
        assert state["nuc_cfg"] is cfg
        # default (TF-only) leaves nuc_cfg None
        rt._worker_init("h", "m", "daf", 3, 5.0, 3, 90)
        assert state["nuc_cfg"] is None
    finally:
        rt._WORKER = saved


def test_apply_result_routes_fused_5tuple(monkeypatch):
    calls = {}

    def fake_fused(read, read_length, result, also_write_legacy, downstream_compat):
        calls["fused"] = result

    def fake_ma(*a, **k):
        calls["ma"] = True

    monkeypatch.setattr(recall_tfs, "write_fused_recall_tags", fake_fused)
    monkeypatch.setattr(recall_tfs, "write_ma_tags", fake_ma)

    class R:
        query_sequence = "A" * 100

    # 5-tuple with fused dict -> write_fused_recall_tags
    recall_tfs._apply_result(R(), (None, None, None, None, {"tf_calls": []}),
                             True, False)
    assert calls.get("fused") == {"tf_calls": []}
    assert "ma" not in calls

    # 5-tuple with None fused -> nothing written
    calls.clear()
    recall_tfs._apply_result(R(), (None, None, None, None, None), True, False)
    assert calls == {}

    # 4-tuple -> write_ma_tags (TF-only path unchanged)
    calls.clear()
    recall_tfs._apply_result(R(), ([], [], [], None), True, False)
    assert calls.get("ma") is True


def test_main_recall_nucs_sets_default_on(monkeypatch):
    captured = {}

    def fake_main(default_recall_nucs=False):
        captured["default"] = default_recall_nucs

    monkeypatch.setattr(recall_tfs, "main", fake_main)
    recall_tfs.main_recall_nucs()
    assert captured["default"] is True
