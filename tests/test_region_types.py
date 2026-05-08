"""Tests for region-parallel worker contract containers."""

from __future__ import annotations

from fiberhmm.inference.region_types import (
    RegionBamResult,
    RegionBamWorkItem,
    RegionBedResult,
    RegionBedWorkItem,
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
