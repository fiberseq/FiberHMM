"""Tests for shared BED formatting helpers."""

from fiberhmm.io.bed import bed12_row


def test_bed12_row_formats_standard_and_extra_columns():
    assert bed12_row(
        ref_name="chr1",
        chrom_start=100,
        chrom_end=150,
        read_id="read1",
        score=42,
        strand="+",
        blocks=[(100, 110), (140, 150)],
        extra_columns=("1,2", "3,4"),
    ) == (
        "chr1\t100\t150\tread1\t42\t+\t100\t150\t0\t2\t"
        "10,10\t0,40\t1,2\t3,4"
    )


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
