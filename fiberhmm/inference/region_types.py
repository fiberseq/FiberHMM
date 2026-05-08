"""Typed contracts for region-parallel worker inputs and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

GenomicRegion = Tuple[str, int, int]
SkipReasons = Dict[str, int]


@dataclass(frozen=True)
class RegionBamWorkItem:
    """Input payload for a region worker that writes a temporary BAM."""

    region: GenomicRegion
    input_bam: str
    temp_bam_path: str
    temp_tsv_path: Optional[str] = None

    @classmethod
    def from_value(
        cls, value: Union["RegionBamWorkItem", tuple]
    ) -> "RegionBamWorkItem":
        if isinstance(value, cls):
            return value
        if len(value) == 3:
            region, input_bam, temp_bam_path = value
            temp_tsv_path = None
        else:
            region, input_bam, temp_bam_path, temp_tsv_path = value
        return cls(_coerce_region(region), input_bam, temp_bam_path, temp_tsv_path)


@dataclass(frozen=True)
class RegionBamResult:
    """Output payload from a region worker that wrote a temporary BAM."""

    temp_bam_path: str
    total_reads: int
    reads_with_footprints: int
    written: int
    temp_tsv_path: Optional[str] = None
    skip_reasons: SkipReasons = field(default_factory=dict)

    @classmethod
    def from_value(
        cls, value: Union["RegionBamResult", tuple]
    ) -> "RegionBamResult":
        if isinstance(value, cls):
            return value
        if len(value) == 6:
            (
                temp_bam_path, total_reads, reads_with_footprints,
                written, temp_tsv_path, skip_reasons,
            ) = value
        elif len(value) == 5:
            (
                temp_bam_path, total_reads, reads_with_footprints,
                written, temp_tsv_path,
            ) = value
            skip_reasons = {}
        else:
            temp_bam_path, total_reads, reads_with_footprints, written = value
            temp_tsv_path = None
            skip_reasons = {}
        return cls(
            temp_bam_path=temp_bam_path,
            total_reads=total_reads,
            reads_with_footprints=reads_with_footprints,
            written=written,
            temp_tsv_path=temp_tsv_path,
            skip_reasons=skip_reasons or {},
        )


@dataclass(frozen=True)
class RegionBedWorkItem:
    """Input payload for a region worker that writes a temporary BED."""

    region: GenomicRegion
    input_bam: str
    temp_bed_path: str

    @classmethod
    def from_value(
        cls, value: Union["RegionBedWorkItem", tuple]
    ) -> "RegionBedWorkItem":
        if isinstance(value, cls):
            return value
        region, input_bam, temp_bed_path = value
        return cls(_coerce_region(region), input_bam, temp_bed_path)


@dataclass(frozen=True)
class RegionBedResult:
    """Output payload from a region worker that wrote a temporary BED."""

    temp_bed_path: str
    total_reads: int
    reads_with_footprints: int

    @classmethod
    def from_value(
        cls, value: Union["RegionBedResult", tuple]
    ) -> "RegionBedResult":
        if isinstance(value, cls):
            return value
        temp_bed_path, total_reads, reads_with_footprints = value
        return cls(temp_bed_path, total_reads, reads_with_footprints)


def _coerce_region(region: tuple) -> GenomicRegion:
    chrom, start, end = region
    return chrom, int(start), int(end)
