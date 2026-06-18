"""Tests for region worker helpers."""

from types import SimpleNamespace

import pytest

import fiberhmm.inference.region_workers as region_workers
from fiberhmm.inference.engine import CHIMERA_SKIP
from fiberhmm.inference.read_filters import ReadFilterConfig
from fiberhmm.inference.region_types import RegionBamResult, RegionBamWorkItem
from fiberhmm.inference.region_workers import (
    _REGION_ROUTE_OUTSIDE,
    _REGION_ROUTE_PROCESS,
    _REGION_ROUTE_SKIP,
    _build_fused_region_recall_result,
    _extract_region_fiber_read,
    _extract_region_payload_fiber_read,
    _format_region_bed12_row,
    _fused_region_recall_config,
    _fused_region_worker_runtime,
    _new_region_skip_reasons,
    _open_region_posterior_tsv,
    _pad_region_bed12_to_read_span,
    _process_fused_region_bam_read,
    _process_region_bam_read,
    _process_region_bed_read,
    _record_skipped_region_read,
    _region_apply_config,
    _region_bam_output_config,
    _region_bam_result_from_counts,
    _region_bam_worker_runtime,
    _region_bed12_blocks,
    _region_bed12_row_from_read_result,
    _region_bed_block_components,
    _region_bed_read_filter_config,
    _region_bed_score_list,
    _region_fused_recall_options,
    _region_read_route,
    _region_result_ns_scores,
    _RegionBamWorkerCounts,
    _run_fused_region_apply_read,
    _run_region_apply_read,
    _write_footprinted_region_read,
    _write_region_posterior_record,
    _write_unfootprinted_region_read,
)


def _route_read(**overrides):
    attrs = {
        "is_unmapped": False,
        "is_secondary": False,
        "is_supplementary": False,
        "mapping_quality": 60,
        "query_alignment_length": 100,
        "query_length": 100,
        "query_name": "read1",
        "query_sequence": "A" * 100,
        "reference_start": 120,
    }
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def _apply_config(**overrides):
    params = {
        "edge_trim": 2,
        "circular": False,
        "mode": "pacbio-fiber",
        "context_size": 6,
        "msp_min_size": 21,
        "nuc_min_size": 90,
        "prob_threshold": 128,
        "with_scores": True,
        "io_threads": 1,
    }
    params.update(overrides)
    return _region_apply_config(params)


def test_region_apply_config_casts_worker_params_and_fused_score_default():
    params = {
        "edge_trim": "2",
        "circular": True,
        "mode": "pacbio-fiber",
        "context_size": "6",
        "msp_min_size": "21",
        "nuc_min_size": "90",
        "prob_threshold": "128",
        "with_scores": True,
        "io_threads": "3",
    }

    config = _region_apply_config(params)

    assert config.edge_trim == 2
    assert config.circular is True
    assert config.mode == "pacbio-fiber"
    assert config.context_size == 6
    assert config.msp_min_size == 21
    assert config.nuc_min_size == 90
    assert config.prob_threshold == 128
    assert config.with_scores is True
    assert config.io_threads == 3

    fused_config = _region_apply_config(
        {key: value for key, value in params.items() if key != "with_scores"},
        with_scores_default=False,
    )

    assert fused_config.with_scores is False


def test_region_bam_output_config_gates_posteriors_on_temp_path():
    assert _region_bam_output_config(
        {"return_posteriors": True, "write_msps": False},
        "region.tsv",
    ).return_posteriors is True
    assert _region_bam_output_config(
        {"return_posteriors": True},
        None,
    ).return_posteriors is False
    assert _region_bam_output_config({}, "region.tsv").write_msps is True


def test_region_bam_worker_runtime_builds_apply_output_and_filter_configs():
    train_rids = {"read1"}

    runtime = _region_bam_worker_runtime({
        "edge_trim": "2",
        "circular": False,
        "mode": "pacbio-fiber",
        "context_size": "6",
        "msp_min_size": "21",
        "nuc_min_size": "90",
        "prob_threshold": "128",
        "with_scores": True,
        "io_threads": "3",
        "return_posteriors": True,
        "write_msps": False,
        "min_mapq": "11",
        "min_read_length": "101",
        "primary_only": True,
        "train_rids": train_rids,
    }, "region.tsv")

    assert runtime.apply_config.with_scores is True
    assert runtime.apply_config.io_threads == 3
    assert runtime.output_config.return_posteriors is True
    assert runtime.output_config.write_msps is False
    assert runtime.filter_config == ReadFilterConfig(
        min_mapq=11,
        min_read_length=101,
        primary_only=True,
        process_unmapped=False,
        train_rids=train_rids,
    )


def test_new_region_skip_reasons_includes_region_extras():
    skip_reasons = _new_region_skip_reasons()

    assert skip_reasons["no_footprints"] == 0
    assert skip_reasons["chimera"] == 0
    assert skip_reasons["low_mapq"] == 0


def test_open_region_posterior_tsv_opens_only_when_enabled(tmp_path):
    disabled_config = _region_bam_output_config({"return_posteriors": False}, None)
    assert _open_region_posterior_tsv(disabled_config, None) == (None, False)

    tsv_path = tmp_path / "region.tsv"
    enabled_config = _region_bam_output_config(
        {"return_posteriors": True},
        str(tsv_path),
    )
    tsv_file, return_posteriors = _open_region_posterior_tsv(
        enabled_config,
        str(tsv_path),
    )

    try:
        assert return_posteriors is True
        assert tsv_file is not None
        tsv_file.write("ok\n")
    finally:
        tsv_file.close()

    assert tsv_path.read_text() == "ok\n"

    bad_file, bad_enabled = _open_region_posterior_tsv(
        enabled_config,
        str(tmp_path / "missing" / "region.tsv"),
    )
    assert bad_file is None
    assert bad_enabled is False


def test_process_region_to_bam_closes_tsv_once_when_fetch_region_missing(
    monkeypatch,
):
    class TsvFile:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    class AlignmentFile:
        def __init__(self, *args, **kwargs):
            self.header = {"HD": {"SO": "coordinate"}}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def fetch(self, *args):
            raise ValueError("missing region")

    tsv_file = TsvFile()
    params = {
        "edge_trim": 0,
        "circular": False,
        "mode": "pacbio-fiber",
        "context_size": 3,
        "msp_min_size": 10,
        "nuc_min_size": 85,
        "prob_threshold": 0,
        "with_scores": False,
        "io_threads": 1,
        "return_posteriors": True,
        "write_msps": True,
        "min_mapq": 0,
        "min_read_length": 0,
        "primary_only": False,
        "train_rids": set(),
    }

    monkeypatch.setattr(region_workers.pysam, "set_verbosity", lambda value: None)
    monkeypatch.setattr(region_workers.pysam, "AlignmentFile", AlignmentFile)
    monkeypatch.setattr(region_workers, "append_coord_marker", lambda header: header)
    monkeypatch.setattr(
        region_workers,
        "_open_region_posterior_tsv",
        lambda *args: (tsv_file, True),
    )
    monkeypatch.setattr(region_workers, "_worker_model", object())
    monkeypatch.setattr(region_workers, "_worker_region_params", params)

    result = region_workers._process_region_to_bam(
        RegionBamWorkItem(("missing", 0, 100), "input.bam", "region.bam", "post.tsv"),
    )

    assert result == RegionBamResult("region.bam", 0, 0, 0)
    assert tsv_file.closed == 1


def test_process_region_to_bam_preserves_work_item_coercion_error(capsys):
    with pytest.raises(TypeError, match="has no len"):
        region_workers._process_region_to_bam(object())

    assert "Worker error in region ?:?-?" in capsys.readouterr().out


def test_region_bam_result_from_counts_includes_tsv_only_when_written():
    counts = _RegionBamWorkerCounts(
        total_reads=10,
        reads_with_footprints=4,
        written=9,
        posteriors_written=2,
    )
    result = _region_bam_result_from_counts(
        "region.bam",
        counts,
        {"low_mapq": 1},
        "region.tsv",
        return_posteriors=True,
    )

    assert result == RegionBamResult(
        "region.bam",
        10,
        4,
        9,
        "region.tsv",
        {"low_mapq": 1},
    )

    no_tsv = _region_bam_result_from_counts(
        "region.bam",
        counts,
        {},
        "region.tsv",
        return_posteriors=False,
    )
    assert no_tsv.temp_tsv_path is None


def test_fused_region_recall_config_casts_worker_params():
    config = _fused_region_recall_config({
        "min_llr": "2.5",
        "min_opps": "4",
        "unify_threshold": "7",
        "also_write_legacy": False,
        "downstream_compat": True,
    })

    assert config.min_llr == 2.5
    assert config.min_opps == 4
    assert config.unify_threshold == 7
    assert config.also_write_legacy is False
    assert config.downstream_compat is True


def test_process_region_bam_read_counts_footprinted_read(monkeypatch):
    import fiberhmm.inference.region_workers as region_workers

    read = _route_read(reference_name="chr1", reference_end=180, is_reverse=False)
    outbam = SimpleNamespace(written=[])
    outbam.write = outbam.written.append
    model = object()
    fiber_read = {"query_sequence": "ACGT"}
    result = {"ns": [10], "nl": [5], "posteriors": [0.1]}
    tsv_file = object()
    calls = {}

    monkeypatch.setattr(
        region_workers,
        "_extract_region_fiber_read",
        lambda got_read, mode, threshold: (fiber_read, None),
    )

    def fake_process(
        got_fiber_read,
        got_model,
        edge_trim,
        circular,
        mode,
        context_size,
        msp_min_size,
        *,
        nuc_min_size,
        with_scores,
        return_posteriors,
    ):
        calls["process"] = (
            got_fiber_read,
            got_model,
            edge_trim,
            circular,
            mode,
            context_size,
            msp_min_size,
            nuc_min_size,
            with_scores,
            return_posteriors,
        )
        return result

    def fake_write(out, got_read, got_result, with_scores, write_msps, got_tsv):
        calls["write"] = (
            out, got_read, got_result, with_scores, write_msps, got_tsv
        )
        return 1, True

    monkeypatch.setattr(region_workers, "_process_single_read", fake_process)
    monkeypatch.setattr(region_workers, "_write_footprinted_region_read", fake_write)

    delta = _process_region_bam_read(
        read,
        outbam,
        model,
        _apply_config(),
        _region_bam_output_config({"write_msps": False}, None),
        ReadFilterConfig(min_mapq=10, min_read_length=50),
        start=100,
        end=200,
        skip_reasons={},
        return_posteriors=True,
        tsv_file=tsv_file,
    )

    assert delta.total_reads == 1
    assert delta.reads_with_footprints == 1
    assert delta.written == 1
    assert delta.posteriors_written == 1
    assert calls["process"] == (
        fiber_read, model, 2, False, "pacbio-fiber", 6, 21, 90, True, True
    )
    assert calls["write"] == (outbam, read, result, True, False, tsv_file)


def test_process_region_bed_read_writes_footprint_row(monkeypatch):
    import fiberhmm.inference.region_workers as region_workers

    read = _route_read(
        reference_name="chr1",
        reference_start=100,
        reference_end=200,
        query_name="read1",
        is_reverse=False,
    )
    bed_out = SimpleNamespace(lines=[])
    bed_out.write = bed_out.lines.append
    fiber_read = {"query_sequence": "ACGT"}
    result = {"ns": [120], "nl": [10], "ns_scores": [0.5]}

    monkeypatch.setattr(
        region_workers,
        "_extract_region_fiber_read",
        lambda got_read, mode, threshold: (fiber_read, None),
    )
    monkeypatch.setattr(region_workers, "_process_single_read", lambda *_, **__: result)

    delta = _process_region_bed_read(
        read,
        bed_out,
        model=object(),
        apply_config=_apply_config(),
        filter_config=ReadFilterConfig(min_mapq=10, min_read_length=50),
        start=100,
        end=200,
    )

    assert delta.total_reads == 1
    assert delta.reads_with_footprints == 1
    assert bed_out.lines == [
        "chr1\t100\t200\tread1\t0\t+\t100\t200\t0,0,0\t3\t1,10,1\t0,20,99\t0,500,0\n"
    ]


def test_process_fused_region_bam_read_writes_recalled_tags(monkeypatch):
    import fiberhmm.inference.region_workers as region_workers

    read = _route_read(reference_name="chr1", reference_end=180, is_reverse=False)
    outbam = SimpleNamespace(written=[])
    outbam.write = outbam.written.append
    model = object()
    payload = {"read": "payload"}
    fiber_read = {"query_sequence": "ACGT"}
    apply_result = {"ns": [10], "nl": [5]}
    fused_result = {"ma": "tag"}
    calls = {}

    monkeypatch.setattr(region_workers, "make_apply_payload", lambda *_, **__: payload)
    monkeypatch.setattr(
        region_workers,
        "_extract_region_payload_fiber_read",
        lambda got_payload, mode, threshold: (fiber_read, None),
    )
    def fake_apply(*args):
        calls["apply_args"] = args
        return apply_result

    monkeypatch.setattr(region_workers, "run_hmm_apply_stage", fake_apply)
    monkeypatch.setattr(region_workers, "apply_result_has_footprints", lambda _: True)

    def fake_build(*args, **kwargs):
        calls["build"] = (args, kwargs)
        return fused_result

    def fake_write_tags(got_read, **kwargs):
        calls["tags"] = (got_read, kwargs)

    monkeypatch.setattr(region_workers, "build_fused_recall_result", fake_build)
    monkeypatch.setattr(region_workers, "write_fused_recall_tags", fake_write_tags)

    apply_config = _region_apply_config({
        "edge_trim": 2,
        "circular": False,
        "mode": "pacbio-fiber",
        "context_size": 6,
        "msp_min_size": 21,
        "nuc_min_size": 90,
        "prob_threshold": 128,
        "io_threads": 1,
    }, with_scores_default=False)
    recall_config = _fused_region_recall_config({
        "min_llr": 2.5,
        "min_opps": 4,
        "unify_threshold": 7,
        "also_write_legacy": False,
        "downstream_compat": True,
    })
    recall_options = {
        "recall_nucs": True,
        "split_min_llr": 5.0,
        "split_min_opps": 3,
        "nuc_min_size": 90,
        "msp_min_size": 21,
        "phase_nrl": 185,
    }

    delta = _process_fused_region_bam_read(
        read,
        outbam,
        model,
        llr_hit=[1],
        llr_miss=[2],
        apply_config=apply_config,
        recall_config=recall_config,
        recall_options=recall_options,
        ref_fasta=None,
        filter_config=ReadFilterConfig(min_mapq=10, min_read_length=50),
        start=100,
        end=200,
        skip_reasons={},
    )

    assert delta.total_reads == 1
    assert delta.reads_with_footprints == 1
    assert delta.written == 1
    assert outbam.written == [read]
    assert calls["build"][0] == (
        fiber_read, apply_result, [1], [2], 2.5, 4, 7, False
    )
    assert calls["build"][1] == recall_options
    assert calls["tags"] == (
        read,
        {
            "read_length": 4,
            "result": fused_result,
            "also_write_legacy": False,
            "downstream_compat": True,
        },
    )


def test_region_read_route_preserves_skip_and_ownership_order():
    config = ReadFilterConfig(min_mapq=10, min_read_length=50)

    assert _region_read_route(_route_read(), 100, 200, config) == (
        _REGION_ROUTE_PROCESS,
        None,
    )
    assert _region_read_route(
        _route_read(reference_start=90), 100, 200, config
    ) == (_REGION_ROUTE_OUTSIDE, None)
    assert _region_read_route(
        _route_read(mapping_quality=0), 100, 200, config
    ) == (_REGION_ROUTE_SKIP, "low_mapq")
    assert _region_read_route(
        _route_read(is_unmapped=True, reference_start=90), 100, 200, config
    ) == (_REGION_ROUTE_SKIP, "unmapped")


def test_record_skipped_region_read_writes_and_updates_counters():
    read = _route_read()
    outbam = SimpleNamespace(written=[])
    outbam.write = outbam.written.append
    skip_reasons = {"low_mapq": 0}

    written, skipped = _record_skipped_region_read(
        outbam, read, skip_reasons, "low_mapq", written=3, skipped=2
    )

    assert outbam.written == [read]
    assert skip_reasons["low_mapq"] == 1
    assert (written, skipped) == (4, 3)


def test_write_unfootprinted_region_read_counts_no_footprints():
    read = _route_read()
    outbam = SimpleNamespace(written=[])
    outbam.write = outbam.written.append
    skip_reasons = {"no_footprints": 4}

    assert _write_unfootprinted_region_read(outbam, read, skip_reasons) == 1
    assert outbam.written == [read]
    assert skip_reasons["no_footprints"] == 5


def test_write_footprinted_region_read_tags_writes_and_streams_posterior(monkeypatch):
    import fiberhmm.inference.region_workers as region_workers

    read = _route_read()
    result = {"posteriors": [0.1], "ns": [10], "nl": [5]}
    outbam = SimpleNamespace(written=[])
    outbam.write = outbam.written.append
    tsv_file = SimpleNamespace()
    calls = {}

    def fake_set_tags(got_read, got_result, with_scores, write_msps):
        calls["set_tags"] = (got_read, got_result, with_scores, write_msps)

    def fake_write_posterior(got_tsv, got_read, got_result):
        calls["posterior"] = (got_tsv, got_read, got_result)
        return True

    monkeypatch.setattr(region_workers, "set_legacy_apply_tags", fake_set_tags)
    monkeypatch.setattr(
        region_workers, "_write_region_posterior_record", fake_write_posterior
    )

    written, posterior_written = _write_footprinted_region_read(
        outbam, read, result, with_scores=True, write_msps=False, tsv_file=tsv_file
    )

    assert (written, posterior_written) == (1, True)
    assert calls["set_tags"] == (read, result, True, False)
    assert calls["posterior"] == (tsv_file, read, result)
    assert outbam.written == [read]


def test_write_footprinted_region_read_skips_absent_posteriors(monkeypatch):
    import fiberhmm.inference.region_workers as region_workers

    read = _route_read()
    outbam = SimpleNamespace(written=[])
    outbam.write = outbam.written.append
    result = {"posteriors": None}
    posterior_calls = []

    monkeypatch.setattr(region_workers, "set_legacy_apply_tags", lambda *_: None)
    monkeypatch.setattr(
        region_workers,
        "_write_region_posterior_record",
        lambda *args: posterior_calls.append(args),
    )

    assert _write_footprinted_region_read(
        outbam, read, result, with_scores=False, write_msps=True, tsv_file=object()
    ) == (1, False)
    assert posterior_calls == []
    assert outbam.written == [read]


def test_extract_region_fiber_read_maps_skip_reasons(monkeypatch):
    read = _route_read()
    fiber_read = {"query_sequence": "ACGT"}

    def fake_extract(got_read, mode, prob_threshold):
        assert got_read is read
        assert mode == "pacbio-fiber"
        assert prob_threshold == 128
        return fiber_read

    import fiberhmm.inference.region_workers as region_workers

    monkeypatch.setattr(region_workers, "_extract_fiber_read_from_pysam", fake_extract)
    assert _extract_region_fiber_read(read, "pacbio-fiber", 128) == (fiber_read, None)

    monkeypatch.setattr(
        region_workers, "_extract_fiber_read_from_pysam", lambda *_: CHIMERA_SKIP
    )
    assert _extract_region_fiber_read(read, "pacbio-fiber", 128) == (None, "chimera")

    monkeypatch.setattr(
        region_workers, "_extract_fiber_read_from_pysam", lambda *_: None
    )
    assert _extract_region_fiber_read(read, "pacbio-fiber", 128) == (
        None,
        "no_modifications",
    )

    def fail_extract(*_):
        raise ValueError("bad read")

    monkeypatch.setattr(region_workers, "_extract_fiber_read_from_pysam", fail_extract)
    assert _extract_region_fiber_read(read, "pacbio-fiber", 128) == (
        None,
        "extraction_failed",
    )


def test_extract_region_payload_fiber_read_maps_skip_reasons(monkeypatch):
    payload = {"query_name": "read1"}
    fiber_read = {"query_sequence": "ACGT"}

    def fake_extract(got_payload, mode, prob_threshold):
        assert got_payload is payload
        assert mode == "pacbio-fiber"
        assert prob_threshold == 128
        return fiber_read

    import fiberhmm.inference.region_workers as region_workers

    monkeypatch.setattr(region_workers, "extract_fiber_read_from_payload", fake_extract)
    assert _extract_region_payload_fiber_read(payload, "pacbio-fiber", 128) == (
        fiber_read,
        None,
    )
    assert _extract_region_payload_fiber_read(None, "pacbio-fiber", 128) == (
        None,
        "no_modifications",
    )

    monkeypatch.setattr(
        region_workers, "extract_fiber_read_from_payload", lambda *_: CHIMERA_SKIP
    )
    assert _extract_region_payload_fiber_read(payload, "pacbio-fiber", 128) == (
        None,
        "chimera",
    )

    def fail_extract(*_):
        raise ValueError("bad payload")

    monkeypatch.setattr(region_workers, "extract_fiber_read_from_payload", fail_extract)
    assert _extract_region_payload_fiber_read(payload, "pacbio-fiber", 128) == (
        None,
        "extraction_failed",
    )


def test_run_region_apply_read_uses_apply_config(monkeypatch):
    import fiberhmm.inference.region_workers as region_workers

    calls = {}

    def fake_process(
        fiber_read,
        model,
        edge_trim,
        circular,
        mode,
        context_size,
        msp_min_size,
        *,
        nuc_min_size,
        with_scores,
        return_posteriors,
    ):
        calls["args"] = (
            fiber_read,
            model,
            edge_trim,
            circular,
            mode,
            context_size,
            msp_min_size,
            nuc_min_size,
            with_scores,
            return_posteriors,
        )
        return {"ns": [1]}

    monkeypatch.setattr(region_workers, "_process_single_read", fake_process)

    fiber_read = {"query_sequence": "ACGT"}
    model = object()
    config = _apply_config(
        edge_trim=3,
        circular=True,
        context_size=7,
        msp_min_size=22,
        nuc_min_size=91,
        with_scores=False,
    )

    result = _run_region_apply_read(
        fiber_read,
        model,
        config,
        return_posteriors=True,
    )

    assert result == {"ns": [1]}
    assert calls["args"] == (
        fiber_read,
        model,
        3,
        True,
        "pacbio-fiber",
        7,
        22,
        91,
        False,
        True,
    )


def test_run_fused_region_apply_read_uses_apply_config(monkeypatch):
    import fiberhmm.inference.region_workers as region_workers

    calls = {}

    def fake_run_hmm_apply_stage(
        fiber_read,
        model,
        edge_trim,
        circular,
        mode,
        context_size,
        msp_min_size,
        nuc_min_size,
        with_scores,
    ):
        calls["args"] = (
            fiber_read,
            model,
            edge_trim,
            circular,
            mode,
            context_size,
            msp_min_size,
            nuc_min_size,
            with_scores,
        )
        return {"ns": [1]}

    monkeypatch.setattr(
        region_workers,
        "run_hmm_apply_stage",
        fake_run_hmm_apply_stage,
    )

    fiber_read = {"query_sequence": "ACGT"}
    model = object()
    config = _apply_config(
        edge_trim=3,
        circular=True,
        context_size=7,
        msp_min_size=22,
        nuc_min_size=91,
        with_scores=False,
    )

    result = _run_fused_region_apply_read(fiber_read, model, config)

    assert result == {"ns": [1]}
    assert calls["args"] == (
        fiber_read,
        model,
        3,
        True,
        "pacbio-fiber",
        7,
        22,
        91,
        False,
    )


def test_build_fused_region_recall_result_uses_worker_configs(monkeypatch):
    import fiberhmm.inference.region_workers as region_workers

    calls = {}

    def fake_build_fused_recall_result(
        fiber_read,
        apply_result,
        llr_hit,
        llr_miss,
        min_llr,
        min_opps,
        unify_threshold,
        with_scores,
        **recall_options,
    ):
        calls["args"] = (
            fiber_read,
            apply_result,
            llr_hit,
            llr_miss,
            min_llr,
            min_opps,
            unify_threshold,
            with_scores,
            recall_options,
        )
        return {"ma": "tag"}

    monkeypatch.setattr(
        region_workers,
        "build_fused_recall_result",
        fake_build_fused_recall_result,
    )

    fiber_read = {"query_sequence": "ACGT"}
    apply_result = {"ns": [1]}
    apply_config = _apply_config(with_scores=False)
    recall_config = _fused_region_recall_config({
        "min_llr": "2.5",
        "min_opps": "4",
        "unify_threshold": "7",
        "also_write_legacy": False,
        "downstream_compat": True,
    })
    recall_options = {"recall_nucs": True, "phase_nrl": 190}

    result = _build_fused_region_recall_result(
        fiber_read,
        apply_result,
        "hit",
        "miss",
        apply_config,
        recall_config,
        recall_options,
    )

    assert result == {"ma": "tag"}
    assert calls["args"] == (
        fiber_read,
        apply_result,
        "hit",
        "miss",
        2.5,
        4,
        7,
        False,
        recall_options,
    )


def test_region_bed12_row_pads_blocks_and_scores():
    row = _format_region_bed12_row(
        ref_name="chr1",
        ref_start=100,
        ref_end=200,
        read_id="read1",
        strand="+",
        starts=[120, 180],
        lengths=[10, 5],
        scores=[0.5, 0.75],
    )

    assert row.split("\t") == [
        "chr1",
        "100",
        "200",
        "read1",
        "0",
        "+",
        "100",
        "200",
        "0,0,0",
        "4",
        "1,10,5,1",
        "0,20,80,99",
        "0,500,750,0",
    ]


def test_region_bed12_blocks_project_pad_and_scale_scores():
    block_starts, block_sizes, score_list = _region_bed12_blocks(
        ref_start=100,
        ref_end=200,
        starts=[120, 180],
        lengths=[10, 5],
        scores=[0.5, 0.75],
    )

    assert block_starts == [0, 20, 80, 99]
    assert block_sizes == [1, 10, 5, 1]
    assert score_list == [0, 500, 750, 0]

    block_starts, block_sizes, score_list = _region_bed12_blocks(
        ref_start=100,
        ref_end=130,
        starts=[100],
        lengths=[30],
    )

    assert block_starts == [0]
    assert block_sizes == [30]
    assert score_list is None


def test_region_bed_block_components_project_reference_offsets():
    block_starts, block_sizes = _region_bed_block_components(
        ref_start=100,
        starts=[120, 180],
        lengths=[10, 5],
    )

    assert block_starts == [20, 80]
    assert block_sizes == [10, 5]
    assert all(type(value) is int for value in block_starts + block_sizes)


def test_region_bed_score_list_scales_optional_scores():
    assert _region_bed_score_list(None) is None
    assert _region_bed_score_list([0.5, 0.75]) == [500, 750]


def test_pad_region_bed12_to_read_span_adds_edge_blocks_and_scores():
    block_starts = [20, 80]
    block_sizes = [10, 5]
    scores = [500, 750]

    _pad_region_bed12_to_read_span(block_starts, block_sizes, scores, read_length=100)

    assert block_starts == [0, 20, 80, 99]
    assert block_sizes == [1, 10, 5, 1]
    assert scores == [0, 500, 750, 0]


def test_region_bed12_row_omits_scores_when_absent():
    row = _format_region_bed12_row(
        ref_name="chr1",
        ref_start=100,
        ref_end=130,
        read_id="read1",
        strand="-",
        starts=[100],
        lengths=[30],
    )

    assert row.split("\t") == [
        "chr1",
        "100",
        "130",
        "read1",
        "0",
        "-",
        "100",
        "130",
        "0,0,0",
        "1",
        "30",
        "0",
    ]


def test_region_bed12_row_from_read_result_uses_read_fields_and_scores():
    read = SimpleNamespace(
        reference_name="chr2",
        reference_start=50,
        reference_end=80,
        query_name="read2",
        is_reverse=True,
    )
    result = {
        "ns": [55],
        "nl": [10],
        "ns_scores": [0.25],
    }

    row = _region_bed12_row_from_read_result(read, result, with_scores=True)

    assert row.split("\t") == [
        "chr2",
        "50",
        "80",
        "read2",
        "0",
        "-",
        "50",
        "80",
        "0,0,0",
        "3",
        "1,10,1",
        "0,5,29",
        "0,250,0",
    ]


def test_region_bed_read_filter_config_preserves_bed_policy():
    train_rids = {"training-read"}

    config = _region_bed_read_filter_config({
        "min_mapq": "7",
        "min_read_length": "101",
        "primary_only": False,
        "train_rids": train_rids,
    })

    assert config.min_mapq == 7
    assert config.min_read_length == 101
    assert config.primary_only is True
    assert config.process_unmapped is False
    assert config.train_rids is train_rids


def test_region_fused_recall_options_uses_defaults_and_casts_values():
    assert _region_fused_recall_options(
        {},
        nuc_min_size=85,
        msp_min_size=20,
    ) == {
        "recall_nucs": False,
        "split_min_llr": 4.0,
        "split_min_opps": 3,
        "nuc_min_size": 85,
        "msp_min_size": 20,
        "phase_nrl": 0,
    }
    assert _region_fused_recall_options(
        {
            "recall_nucs": 1,
            "split_min_llr": "5.5",
            "split_min_opps": "7",
            "phase_nrl": "185",
        },
        nuc_min_size=90,
        msp_min_size=25,
    ) == {
        "recall_nucs": True,
        "split_min_llr": 5.5,
        "split_min_opps": 7,
        "nuc_min_size": 90,
        "msp_min_size": 25,
        "phase_nrl": 185,
    }


def test_fused_region_worker_runtime_builds_worker_configs():
    ref_fasta = object()

    runtime = _fused_region_worker_runtime({
        "edge_trim": "2",
        "circular": True,
        "mode": "pacbio-fiber",
        "context_size": "6",
        "msp_min_size": "21",
        "nuc_min_size": "90",
        "prob_threshold": "128",
        "io_threads": "3",
        "min_llr": "2.5",
        "min_opps": "4",
        "unify_threshold": "7",
        "also_write_legacy": False,
        "downstream_compat": True,
        "recall_nucs": True,
        "split_min_llr": "5.5",
        "split_min_opps": "6",
        "phase_nrl": "185",
        "min_mapq": "11",
        "min_read_length": "101",
        "primary_only": True,
        "train_rids": {"read1"},
        "ref_fasta": ref_fasta,
    })

    assert runtime.apply_config.with_scores is False
    assert runtime.apply_config.io_threads == 3
    assert runtime.recall_config.min_llr == 2.5
    assert runtime.recall_config.min_opps == 4
    assert runtime.recall_options == {
        "recall_nucs": True,
        "split_min_llr": 5.5,
        "split_min_opps": 6,
        "nuc_min_size": 90,
        "msp_min_size": 21,
        "phase_nrl": 185,
    }
    assert runtime.filter_config == ReadFilterConfig(
        min_mapq=11,
        min_read_length=101,
        primary_only=True,
        process_unmapped=False,
        train_rids={"read1"},
    )
    assert runtime.ref_fasta is ref_fasta


def test_write_region_posterior_record_returns_success_status(monkeypatch):
    import fiberhmm.inference.region_workers as region_workers

    read = SimpleNamespace(
        query_name="read1",
        reference_name="chr1",
        reference_start=10,
        reference_end=20,
    )
    result = {
        "strand": "+",
        "posteriors": [0.1, 0.9],
        "ns": [1],
        "nl": [2],
    }
    seen = {}

    def fake_format(**kwargs):
        seen.update(kwargs)
        return "posterior-line\n"

    class Tsv:
        def __init__(self, fail=False):
            self.fail = fail
            self.lines = []

        def write(self, line):
            if self.fail:
                raise OSError("write failed")
            self.lines.append(line)

    monkeypatch.setattr(region_workers, "format_region_posterior_line", fake_format)
    tsv = Tsv()

    assert _write_region_posterior_record(tsv, read, result)
    assert tsv.lines == ["posterior-line\n"]
    assert seen == {
        "read_name": "read1",
        "chrom": "chr1",
        "ref_start": 10,
        "ref_end": 20,
        "strand": "+",
        "posteriors": [0.1, 0.9],
        "footprint_starts": [1],
        "footprint_sizes": [2],
    }
    assert not _write_region_posterior_record(Tsv(fail=True), read, result)


def test_region_result_ns_scores_respects_with_scores_flag():
    scores = [0.25, 0.75]

    assert _region_result_ns_scores({"ns_scores": scores}, with_scores=False) is None
    assert _region_result_ns_scores({"ns_scores": None}, with_scores=True) is None
    assert _region_result_ns_scores({"ns_scores": scores}, with_scores=True) is scores
