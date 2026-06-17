"""Tests for region-parallel worker contract containers."""

from __future__ import annotations

from fiberhmm.inference.region_types import (
    RegionBamAggregation,
    RegionBamResult,
    RegionBamWorkItem,
    RegionBedAggregation,
    RegionBedResult,
    RegionBedWorkItem,
    _accumulate_skip_reasons,
)


def test_region_bam_work_item_coerces_legacy_three_tuple():
    item = RegionBamWorkItem.from_value((("chr1", "10", "20"), "in.bam", "tmp.bam"))

    assert item.region == ("chr1", 10, 20)
    assert item.input_bam == "in.bam"
    assert item.temp_bam_path == "tmp.bam"
    assert item.temp_tsv_path is None


def test_region_bam_work_item_coerces_legacy_four_tuple():
    item = RegionBamWorkItem.from_value((("chr2", 5, 9), "in.bam", "tmp.bam", "tmp.tsv"))

    assert item.region == ("chr2", 5, 9)
    assert item.temp_tsv_path == "tmp.tsv"


def test_region_bam_result_coerces_legacy_result_lengths():
    bare = RegionBamResult.from_value(("tmp.bam", 1, 2, 3))
    with_tsv = RegionBamResult.from_value(("tmp.bam", 1, 2, 3, "tmp.tsv"))
    with_skips = RegionBamResult.from_value(
        ("tmp.bam", 1, 2, 3, "tmp.tsv", {"low_mapq": 4})
    )

    assert bare.temp_tsv_path is None
    assert bare.skip_reasons == {}
    assert with_tsv.temp_tsv_path == "tmp.tsv"
    assert with_tsv.skip_reasons == {}
    assert with_skips.skip_reasons == {"low_mapq": 4}


def test_region_bed_work_item_and_result_coerce_legacy_tuples():
    item = RegionBedWorkItem.from_value((("chr3", "11", "12"), "in.bam", "tmp.bed"))
    result = RegionBedResult.from_value(("tmp.bed", 5, 4))

    assert item.region == ("chr3", 11, 12)
    assert item.temp_bed_path == "tmp.bed"
    assert result.temp_bed_path == "tmp.bed"
    assert result.total_reads == 5
    assert result.reads_with_footprints == 4


def test_region_bam_aggregation_accumulates_counts_paths_and_skips():
    aggregation = RegionBamAggregation()

    aggregation.add_result(
        2,
        RegionBamResult("region_2.bam", 10, 7, 10, "region_2.tsv", {"low_mapq": 2}),
        include_tsv=True,
    )
    aggregation.add_result(
        0,
        ("region_0.bam", 3, 1, 3, None, {"low_mapq": 1, "too_short": 4}),
    )

    assert aggregation.total_reads == 13
    assert aggregation.reads_with_footprints == 8
    assert aggregation.total_skipped == 7
    assert aggregation.skip_reasons == {"low_mapq": 3, "too_short": 4}
    assert aggregation.temp_bams == [(2, "region_2.bam"), (0, "region_0.bam")]
    assert aggregation.temp_tsvs == [(2, "region_2.tsv")]
    assert aggregation.completed == 2


def test_accumulate_skip_reasons_updates_target_and_returns_added_total():
    target = {"low_mapq": 2}

    added = _accumulate_skip_reasons(target, {"low_mapq": 1, "too_short": 4})

    assert added == 5
    assert target == {"low_mapq": 3, "too_short": 4}


def test_region_bed_aggregation_accumulates_counts_and_paths():
    aggregation = RegionBedAggregation()

    aggregation.add_result(1, RegionBedResult("region_1.bed", 4, 2))
    aggregation.add_result(0, ("region_0.bed", 3, 1))

    assert aggregation.total_reads == 7
    assert aggregation.reads_with_footprints == 3
    assert aggregation.temp_beds == [(1, "region_1.bed"), (0, "region_0.bed")]
    assert aggregation.completed == 2
