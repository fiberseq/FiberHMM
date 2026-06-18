"""Genome region planning helpers for region-parallel inference."""

from __future__ import annotations

import re
from typing import Optional, Set

import pysam

_SKIP_CHROM_PATTERNS = (
    '_RANDOM', '_ALT', '_FIX', '_HAP',
    'CHRUN_', 'UN_', 'SCAFFOLD', 'CONTIG',
    '_GL', '_KI', '_JH', '_KB',  # Common GenBank accession prefixes
)
_DROSOPHILA_CHROM_RE = re.compile(r'^[234][LR]?$')


def _normalized_chromosome_name(chrom: str) -> str:
    chrom = chrom.upper()
    if chrom.startswith('CHR'):
        return chrom[3:]
    return chrom


def _has_skip_chromosome_pattern(chrom: str) -> bool:
    c = chrom.upper()
    return any(pattern in c for pattern in _SKIP_CHROM_PATTERNS)


def _is_sex_or_mito_chromosome(chrom: str) -> bool:
    return chrom in ('X', 'Y', 'M', 'MT', 'W', 'Z')


def _is_drosophila_chromosome(chrom: str) -> bool:
    return _DROSOPHILA_CHROM_RE.match(chrom) is not None


def _is_c_elegans_chromosome(chrom: str) -> bool:
    return chrom in ('I', 'II', 'III', 'IV', 'V', 'VI')


def _is_main_chromosome(chrom: str) -> bool:
    """
    Check if a chromosome name is a main chromosome (not a scaffold/contig).

    Returns True for:
    - chr1-chr22, chrX, chrY, chrM, chrMT (human with chr prefix)
    - 1-22, X, Y, M, MT (human without chr prefix)
    - 2L, 2R, 3L, 3R, 4, X, Y (Drosophila)

    Returns False for:
    - *_random, chrUn_*, scaffolds, contigs, etc.
    """

    c = chrom.upper()

    # Skip obvious scaffolds/contigs
    if _has_skip_chromosome_pattern(c):
        return False

    c = _normalized_chromosome_name(c)

    # Accept numbered chromosomes 1-22 (or more for other organisms)
    if c.isdigit():
        return True

    # Accept X, Y, M, MT, W, Z (sex chromosomes and mitochondrial)
    if _is_sex_or_mito_chromosome(c):
        return True

    # Accept Drosophila chromosomes: 2L, 2R, 3L, 3R, 4
    if _is_drosophila_chromosome(c):
        return True

    # Accept C. elegans chromosomes: I, II, III, IV, V, X
    if _is_c_elegans_chromosome(c):
        return True

    return False


def _normalize_region_size(region_size: int) -> int:
    region_size = int(region_size)
    if region_size <= 0:
        raise ValueError("region_size must be positive")
    return region_size


def _chromosome_regions(chrom: str, chrom_len: int, region_size: int) -> list[tuple[str, int, int]]:
    region_size = _normalize_region_size(region_size)
    return [
        (chrom, int(start), int(min(start + region_size, chrom_len)))
        for start in range(0, int(chrom_len), int(region_size))
    ]


def _include_chromosome(
    chrom: str,
    chroms: Optional[Set[str]],
    skip_scaffolds: bool,
) -> bool:
    if chroms is not None and chrom not in chroms:
        return False
    if skip_scaffolds and not _is_main_chromosome(chrom):
        return False
    return True


def _get_genome_regions(
    bam_path: str,
    region_size: int = 10_000_000,
    skip_scaffolds: bool = False,
    chroms: Optional[Set[str]] = None,
) -> list[tuple[str, int, int]]:
    """
    Split genome into regions for parallel processing.

    Args:
        bam_path: Path to indexed BAM file
        region_size: Target size of each region in bp (default 10MB)
        skip_scaffolds: If True, skip scaffold/contig chromosomes
        chroms: If provided, only include these chromosomes

    Returns:
        List of (chrom, start, end) tuples
    """
    regions = []
    region_size = _normalize_region_size(region_size)

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for chrom in bam.references:
            if not _include_chromosome(chrom, chroms, skip_scaffolds):
                continue

            chrom_len = int(bam.get_reference_length(chrom))
            regions.extend(_chromosome_regions(chrom, chrom_len, region_size))

    return regions
