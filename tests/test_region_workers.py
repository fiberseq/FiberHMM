"""Tests for region worker helpers."""

from fiberhmm.inference.region_workers import _format_region_bed12_row


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
