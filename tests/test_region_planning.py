"""Tests for region-parallel genome planning helpers."""

from __future__ import annotations

import pysam

from fiberhmm.inference.region_planning import (
    _chromosome_regions,
    _get_genome_regions,
    _has_skip_chromosome_pattern,
    _include_chromosome,
    _is_c_elegans_chromosome,
    _is_drosophila_chromosome,
    _is_main_chromosome,
    _is_sex_or_mito_chromosome,
    _normalized_chromosome_name,
)


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


def test_normalized_chromosome_name_strips_chr_prefix_and_uppercases():
    assert _normalized_chromosome_name("chrM") == "M"
    assert _normalized_chromosome_name("2l") == "2L"


def test_has_skip_chromosome_pattern_detects_scaffold_markers():
    assert _has_skip_chromosome_pattern("chrUn_random")
    assert _has_skip_chromosome_pattern("chr1_KI270706v1_random")
    assert not _has_skip_chromosome_pattern("chr2L")


def test_main_chromosome_class_helpers_cover_named_chromosomes():
    assert _is_sex_or_mito_chromosome("MT")
    assert _is_drosophila_chromosome("2L")
    assert _is_c_elegans_chromosome("IV")
    assert _is_main_chromosome("chr3R")
    assert not _is_main_chromosome("chrEBV")


def test_include_chromosome_applies_allowlist_and_scaffold_filter():
    assert _include_chromosome("chr1", chroms=None, skip_scaffolds=False)
    assert _include_chromosome("chr1", chroms={"chr1"}, skip_scaffolds=True)
    assert not _include_chromosome("chr2", chroms={"chr1"}, skip_scaffolds=False)
    assert not _include_chromosome("chrUn_random", chroms=None, skip_scaffolds=True)


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
