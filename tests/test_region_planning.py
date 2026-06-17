"""Tests for region-parallel genome planning helpers."""

from __future__ import annotations

import pysam

from fiberhmm.inference.region_planning import _chromosome_regions, _get_genome_regions


def _write_empty_bam(path):
    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [
            {"SN": "chr1", "LN": 250},
            {"SN": "chrUn_random", "LN": 100},
            {"SN": "chr2", "LN": 80},
        ],
    })
    with pysam.AlignmentFile(path, "wb", header=header):
        pass


def test_chromosome_regions_tiles_partial_last_region():
    assert _chromosome_regions("chr1", 250, 100) == [
        ("chr1", 0, 100),
        ("chr1", 100, 200),
        ("chr1", 200, 250),
    ]


def test_get_genome_regions_splits_and_filters(tmp_path):
    bam_path = str(tmp_path / "empty.bam")
    _write_empty_bam(bam_path)

    assert _get_genome_regions(bam_path, region_size=100) == [
        ("chr1", 0, 100),
        ("chr1", 100, 200),
        ("chr1", 200, 250),
        ("chrUn_random", 0, 100),
        ("chr2", 0, 80),
    ]
    assert _get_genome_regions(bam_path, region_size=100, skip_scaffolds=True) == [
        ("chr1", 0, 100),
        ("chr1", 100, 200),
        ("chr1", 200, 250),
        ("chr2", 0, 80),
    ]
    assert _get_genome_regions(bam_path, region_size=100, chroms={"chr2"}) == [
        ("chr2", 0, 80),
    ]
