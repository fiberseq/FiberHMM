"""Typed contracts for region-parallel worker inputs and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

GenomicRegion = Tuple[str, int, int]
SkipReasons = Dict[str, int]
Metrics = Dict[str, int]


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
    metrics: Metrics = field(default_factory=dict)

    @classmethod
    def from_value(
        cls, value: Union["RegionBamResult", tuple]
    ) -> "RegionBamResult":
        if isinstance(value, cls):
            return value
        if len(value) == 7:
            (
                temp_bam_path, total_reads, reads_with_footprints,
                written, temp_tsv_path, skip_reasons, metrics,
            ) = value
        elif len(value) == 6:
            (
                temp_bam_path, total_reads, reads_with_footprints,
                written, temp_tsv_path, skip_reasons,
            ) = value
            metrics = {}
        elif len(value) == 5:
            (
                temp_bam_path, total_reads, reads_with_footprints,
                written, temp_tsv_path,
            ) = value
            skip_reasons = {}
            metrics = {}
        else:
            temp_bam_path, total_reads, reads_with_footprints, written = value
            temp_tsv_path = None
            skip_reasons = {}
            metrics = {}
        return cls(
            temp_bam_path=temp_bam_path,
            total_reads=total_reads,
            reads_with_footprints=reads_with_footprints,
            written=written,
            temp_tsv_path=temp_tsv_path,
            skip_reasons=skip_reasons or {},
            metrics=metrics or {},
        )


@dataclass
class RegionBamAggregation:
    """Running totals for region BAM worker results."""

    total_reads: int = 0
    reads_with_footprints: int = 0
    total_skipped: int = 0
    skip_reasons: SkipReasons = field(default_factory=dict)
    metrics: Metrics = field(default_factory=dict)
    temp_bams: List[Tuple[int, str]] = field(default_factory=list)
    temp_tsvs: List[Tuple[int, str]] = field(default_factory=list)

    @property
    def completed(self) -> int:
        return len(self.temp_bams)

    def add_result(
        self,
        region_index: int,
        result: Union[RegionBamResult, tuple],
        include_tsv: bool = False,
    ) -> RegionBamResult:
        result = RegionBamResult.from_value(result)
        self.total_reads += result.total_reads
        self.reads_with_footprints += result.reads_with_footprints
        self.temp_bams.append((region_index, result.temp_bam_path))

        for reason, count in result.skip_reasons.items():
            self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + count
            self.total_skipped += count

        for name, count in result.metrics.items():
            self.metrics[name] = self.metrics.get(name, 0) + count

        if include_tsv and result.temp_tsv_path:
            self.temp_tsvs.append((region_index, result.temp_tsv_path))

        return result


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


@dataclass
class RegionBedAggregation:
    """Running totals for region BED worker results."""

    total_reads: int = 0
    reads_with_footprints: int = 0
    temp_beds: List[Tuple[int, str]] = field(default_factory=list)

    @property
    def completed(self) -> int:
        return len(self.temp_beds)

    def add_result(
        self, region_index: int, result: Union[RegionBedResult, tuple]
    ) -> RegionBedResult:
        result = RegionBedResult.from_value(result)
        self.total_reads += result.total_reads
        self.reads_with_footprints += result.reads_with_footprints
        self.temp_beds.append((region_index, result.temp_bed_path))
        return result


def _coerce_region(region: tuple) -> GenomicRegion:
    chrom, start, end = region
    return chrom, int(start), int(end)
