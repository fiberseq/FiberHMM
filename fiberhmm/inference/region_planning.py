"""Genome region planning helpers for region-parallel inference."""

from __future__ import annotations

import re
from typing import List, NamedTuple, Optional, Set

import pysam

# Region name for the unplaced-unmapped reads at the end of a sorted BAM
# (``samtools view in.bam '*'``).
UNPLACED_REGION = "*"

# Roman numerals I..XXXIX (yeast chrI-chrXVI, C. elegans I-V). Letters
# beyond I/V/X are excluded so names such as chrC or chrL stay scaffolds.
_ROMAN_NUMERAL = re.compile(r"^X{0,3}(IX|IV|V?I{0,3})$")


class RegionPlanError(ValueError):
    """Region-parallel processing cannot run on this input/selection."""


class RegionPlanItem(NamedTuple):
    """One unit of region-parallel work, in output (coordinate) order.

    ``passthrough`` items cover whole contigs excluded by ``--chroms`` /
    ``--skip-scaffolds`` and the unplaced unmapped reads; their records are
    copied to the output unannotated so region-parallel output keeps every
    input record, like the streaming pipeline.
    """

    chrom: str
    start: int
    end: int
    passthrough: bool = False

    @property
    def region(self):
        return (self.chrom, self.start, self.end)


def _is_main_chromosome(chrom: str) -> bool:
    """
    Check if a chromosome name is a main chromosome (not a scaffold/contig).

    Returns True for:
    - chr1-chr22, chrX, chrY, chrM, chrMT (human with chr prefix)
    - 1-22, X, Y, M, MT (human without chr prefix)
    - 2L, 2R, 3L, 3R, 4, X, Y (Drosophila)
    - Roman-numeral chromosomes (S. cerevisiae chrI-chrXVI, C. elegans I-V)
    - RefSeq chromosome/organelle accessions (NC_000001.11, ...)
    - chrEBV (the EBV episome present in LCL samples)

    Returns False for:
    - *_random, chrUn_*, scaffolds, contigs, RefSeq NT_/NW_ scaffolds, etc.
    """

    # Normalize to uppercase for comparison
    c = chrom.upper()

    # RefSeq complete molecules (chromosomes, mitochondrion, plastids) are
    # NC_ accessions; NT_/NW_ are contigs/scaffolds.
    if re.match(r"^NC_\d+(\.\d+)?$", c):
        return True

    # Skip obvious scaffolds/contigs
    skip_patterns = [
        '_RANDOM', '_ALT', '_FIX', '_HAP',
        'CHRUN_', 'UN_', 'SCAFFOLD', 'CONTIG',
        '_GL', '_KI', '_JH', '_KB'  # Common GenBank accession prefixes
    ]
    for pattern in skip_patterns:
        if pattern in c:
            return False

    # Strip chr prefix if present
    if c.startswith('CHR'):
        c = c[3:]

    # Accept numbered chromosomes 1-22 (or more for other organisms)
    if c.isdigit():
        return True

    # Accept X, Y, M, MT, W, Z (sex chromosomes and mitochondrial) and EBV
    if c in ('X', 'Y', 'M', 'MT', 'W', 'Z', 'EBV'):
        return True

    # Accept Drosophila chromosomes: 2L, 2R, 3L, 3R, 4
    if re.match(r'^[234][LR]?$', c):
        return True

    # Accept Roman-numeral chromosomes (yeast I-XVI, C. elegans I-V).
    if c and _ROMAN_NUMERAL.match(c):
        return True

    return False


def _selected(chrom: str, skip_scaffolds: bool, chroms: Optional[Set[str]]) -> bool:
    if chroms is not None and chrom not in chroms:
        return False
    if skip_scaffolds and not _is_main_chromosome(chrom):
        return False
    return True


def _validate_selection(references, chroms: Optional[Set[str]]) -> None:
    if chroms is None:
        return
    unknown = sorted(set(chroms) - set(references))
    if unknown:
        raise RegionPlanError(
            "--chroms names contigs that are not in the BAM header: "
            + ", ".join(unknown)
        )


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

    Raises:
        RegionPlanError: ``chroms`` names an unknown contig, or no region is
        left to process (for example ``--skip-scaffolds`` removed everything).
    """
    return [item.region for item in plan_region_work(
        bam_path, region_size, skip_scaffolds, chroms, include_passthrough=False,
    )]


def plan_region_work(
    bam_path: str,
    region_size: int = 10_000_000,
    skip_scaffolds: bool = False,
    chroms: Optional[Set[str]] = None,
    include_passthrough: bool = True,
) -> List[RegionPlanItem]:
    """Plan region-parallel work in the BAM's coordinate order.

    Selected contigs are split into ``region_size`` windows. With
    ``include_passthrough``, every unselected contig becomes one pass-through
    item at its header position and the unplaced unmapped reads a final
    ``'*'`` item, so concatenating the per-item BAMs in plan order reproduces
    a coordinate-sorted copy of every input record.
    """
    region_size = int(region_size)
    if region_size <= 0:
        raise RegionPlanError("--region-size must be a positive number of bp")
    items: List[RegionPlanItem] = []
    n_processing = 0
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        references = list(bam.references)
        _validate_selection(references, chroms)
        for chrom in references:
            chrom_len = int(bam.get_reference_length(chrom))
            if not _selected(chrom, skip_scaffolds, chroms):
                if include_passthrough:
                    items.append(RegionPlanItem(chrom, 0, chrom_len, True))
                continue
            for start in range(0, chrom_len, region_size):
                end = min(start + region_size, chrom_len)
                items.append(RegionPlanItem(chrom, int(start), int(end)))
                n_processing += 1
    if n_processing == 0:
        detail = (
            "the BAM header lists no reference sequences (unaligned input?)"
            if not references
            else "--chroms/--skip-scaffolds excluded every contig"
        )
        raise RegionPlanError(f"no genomic regions to process: {detail}")
    if include_passthrough:
        items.append(RegionPlanItem(UNPLACED_REGION, 0, 0, True))
    return items


def require_indexed_bam(bam_path: str) -> None:
    """Raise :class:`RegionPlanError` unless ``bam_path`` has a usable index."""
    try:
        with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
            indexed = bam.has_index()
    except (OSError, ValueError) as exc:
        raise RegionPlanError(f"cannot open {bam_path}: {exc}") from exc
    if not indexed:
        raise RegionPlanError(
            f"--region-parallel needs a coordinate-sorted, indexed BAM, but "
            f"{bam_path} has no index (.bai/.csi). Run 'samtools index', or drop "
            "--region-parallel to use the streaming pipeline (which also "
            "handles unaligned and unindexed input)."
        )
