"""Tests for region worker helpers."""

from types import SimpleNamespace

from fiberhmm.inference.read_filters import ReadFilterConfig
from fiberhmm.inference.region_workers import (
    _REGION_ROUTE_OUTSIDE,
    _REGION_ROUTE_PROCESS,
    _REGION_ROUTE_SKIP,
    _format_region_bed12_row,
    _region_read_route,
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
