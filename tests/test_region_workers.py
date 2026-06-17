"""Tests for region worker helpers."""

from types import SimpleNamespace

from fiberhmm.inference.engine import CHIMERA_SKIP
from fiberhmm.inference.read_filters import ReadFilterConfig
from fiberhmm.inference.region_workers import (
    _REGION_ROUTE_OUTSIDE,
    _REGION_ROUTE_PROCESS,
    _REGION_ROUTE_SKIP,
    _extract_region_fiber_read,
    _extract_region_payload_fiber_read,
    _region_bed12_row_from_read_result,
    _format_region_bed12_row,
    _pad_region_bed12_to_read_span,
    _region_bed_block_components,
    _region_bed_read_filter_config,
    _region_bed_score_list,
    _region_bed12_blocks,
    _region_fused_recall_options,
    _region_read_route,
    _region_result_ns_scores,
    _record_skipped_region_read,
    _write_region_posterior_record,
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
