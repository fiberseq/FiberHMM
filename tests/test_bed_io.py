"""Tests for shared BED formatting helpers."""

import numpy as np

from fiberhmm.io.bed import (
    _bed12_block_fields,
    _bed12_core_columns,
    _bed12_core_columns_from_record,
    _bed12_core_record,
    _bed12_row_from_record,
    _Bed12Record,
    bed12_row,
)


def test_bed12_block_fields_format_count_sizes_and_relative_starts():
    block_fields = _bed12_block_fields([(100, 110), (140, 150)], 100)

    assert block_fields.count == 2
    assert block_fields.sizes == "10,10"
    assert block_fields.starts == "0,40"


def test_bed12_core_columns_preserve_standard_field_order():
    record = _Bed12Record(
        ref_name="chr1",
        chrom_start=100,
        chrom_end=150,
        read_id="read1",
        score=42,
        strand="+",
        blocks=[(100, 110), (140, 150)],
        item_rgb="0,0,0",
    )

    assert _bed12_core_record(
        "chr1",
        100,
        150,
        "read1",
        42,
        "+",
        [(100, 110), (140, 150)],
        "0,0,0",
    ) == record
    assert _bed12_core_columns_from_record(record) == [
        "chr1",
        100,
        150,
        "read1",
        42,
        "+",
        100,
        150,
        "0,0,0",
        2,
        "10,10",
        "0,40",
    ]
    assert _bed12_core_columns(
        ref_name="chr1",
        chrom_start=100,
        chrom_end=150,
        read_id="read1",
        score=42,
        strand="+",
        blocks=[(100, 110), (140, 150)],
        item_rgb="0,0,0",
    ) == _bed12_core_columns_from_record(record)


def test_bed12_row_formats_standard_and_extra_columns():
    record = _Bed12Record(
        ref_name="chr1",
        chrom_start=100,
        chrom_end=150,
        read_id="read1",
        score=42,
        strand="+",
        blocks=[(100, 110), (140, 150)],
    )
    expected_row = (
        "chr1\t100\t150\tread1\t42\t+\t100\t150\t0\t2\t"
        "10,10\t0,40\t1,2\t3,4"
    )
    assert _bed12_row_from_record(record, extra_columns=("1,2", "3,4")) == expected_row
    assert bed12_row(
        ref_name="chr1",
        chrom_start=100,
        chrom_end=150,
        read_id="read1",
        score=42,
        strand="+",
        blocks=[(100, 110), (140, 150)],
        extra_columns=("1,2", "3,4"),
    ) == expected_row

    assert bed12_row(
        ref_name="chr1",
        chrom_start=100,
        chrom_end=110,
        read_id="read1",
        score=42,
        strand="+",
        blocks=[(100, 110)],
        extra_columns=np.asarray(["5,6", "7,8"]),
    ).endswith("\t5,6\t7,8")


def test_bed12_row_allows_item_rgb_override():
    row = bed12_row(
        ref_name="chr1",
        chrom_start=100,
        chrom_end=110,
        read_id="read1",
        score=0,
        strand="+",
        blocks=[(100, 110)],
        item_rgb="0,0,0",
    )

    assert row.split("\t")[8] == "0,0,0"
