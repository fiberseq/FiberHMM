"""Load data-derived footprint population model inputs from annotated BAMs.

The adapter deliberately consumes the ordinary ``tf`` and ``msp`` groups in
the Molecular Annotation (``MA``) tag.  A mapped record without ``MA`` did not
necessarily pass through footprint calling and is therefore not evidence for
an unoccupied site.  Conversely, a valid ``MA`` record with no TF annotations
is an essential denominator molecule and is retained.

MA coordinates are molecular (original-read) coordinates.  They are converted
to the stored BAM query frame before CIGAR projection.  Partially mapped TF
calls remain observations, but only calls with reliable projected boundaries
are allowed to teach model geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pysam

from fiberhmm.core.bam_reader import cigar_to_query_ref
from fiberhmm.inference.tf_sites import BaselineMolecule, TFObservation
from fiberhmm.io.ma_tags import parse_ma_tag


@dataclass(frozen=True)
class ReferenceRegion:
    """One zero-based, half-open BAM analysis interval."""

    contig: str
    start: int
    end: int

    @property
    def label(self) -> str:
        return f"{self.contig}:{self.start}-{self.end}"


@dataclass(frozen=True)
class BamFootprintDiagnostics:
    """Auditable BAM filtering and reference-projection counts."""

    records_seen: int
    primary_mapped_records: int
    emitted_molecules: int
    skipped_unmapped: int
    skipped_secondary: int
    skipped_supplementary: int
    skipped_qcfail: int
    skipped_duplicate: int
    skipped_low_mapq: int
    skipped_missing_ma: int
    skipped_without_mapped_blocks: int
    duplicate_region_records: int
    reads_with_tag_stratum: int
    reads_with_alignment_stratum: int
    tf_annotations: int
    projected_tf_annotations: int
    geometry_eligible_tf_annotations: int
    unprojectable_tf_annotations: int
    out_of_scope_tf_annotations: int
    msp_annotations: int
    projected_msp_annotations: int
    incomplete_msp_annotations: int
    unprojectable_msp_annotations: int

    def as_dict(self) -> Mapping[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class BamFootprintInput:
    """Materialized model input and its BAM provenance."""

    input_path: Path
    molecules: Tuple[BaselineMolecule, ...]
    diagnostics: BamFootprintDiagnostics
    regions: Tuple[ReferenceRegion, ...]
    references: Tuple[Tuple[str, int], ...]

    @property
    def region_labels(self) -> Tuple[str, ...]:
        return tuple(region.label for region in self.regions)


class BamFootprintInputError(ValueError):
    """Raised when a BAM cannot be interpreted without losing semantics."""


def parse_reference_region(
    value: str,
    reference_lengths: Mapping[str, int],
) -> ReferenceRegion:
    """Parse ``CONTIG[:START-END]`` as zero-based, half-open coordinates."""

    if not isinstance(value, str) or not value:
        raise BamFootprintInputError("region must be CONTIG or CONTIG:START-END")
    if ":" not in value:
        contig = value
        if contig not in reference_lengths:
            raise BamFootprintInputError(f"unknown BAM contig in region: {contig}")
        return ReferenceRegion(contig, 0, int(reference_lengths[contig]))

    contig, coordinates = value.rsplit(":", 1)
    try:
        start_text, end_text = coordinates.replace(",", "").split("-", 1)
        start, end = int(start_text), int(end_text)
    except (TypeError, ValueError) as error:
        raise BamFootprintInputError("region must be CONTIG or CONTIG:START-END") from error
    if contig not in reference_lengths:
        raise BamFootprintInputError(f"unknown BAM contig in region: {contig}")
    if start < 0 or end <= start:
        raise BamFootprintInputError("region must satisfy 0 <= START < END")
    reference_length = int(reference_lengths[contig])
    if end > reference_length:
        raise BamFootprintInputError(f"region end {end} exceeds {contig} length {reference_length}")
    return ReferenceRegion(contig, start, end)


def _merge_regions(
    regions: Iterable[ReferenceRegion],
    reference_order: Mapping[str, int],
) -> Tuple[ReferenceRegion, ...]:
    ordered = sorted(
        regions,
        key=lambda region: (
            reference_order[region.contig],
            region.start,
            region.end,
        ),
    )
    merged: List[ReferenceRegion] = []
    for region in ordered:
        if merged and merged[-1].contig == region.contig and region.start <= merged[-1].end:
            previous = merged[-1]
            merged[-1] = ReferenceRegion(
                previous.contig,
                previous.start,
                max(previous.end, region.end),
            )
        else:
            merged.append(region)
    return tuple(merged)


def _merged_reference_blocks(read) -> Tuple[Tuple[int, int], ...]:
    blocks: List[Tuple[int, int]] = []
    for raw_start, raw_end in read.get_blocks():
        start, end = int(raw_start), int(raw_end)
        if end <= start:
            continue
        if blocks and start <= blocks[-1][1]:
            blocks[-1] = (blocks[-1][0], max(blocks[-1][1], end))
        else:
            blocks.append((start, end))
    return tuple(blocks)


def _hard_clip_lengths(read) -> Tuple[int, int]:
    cigar = read.cigartuples or ()
    leading = int(cigar[0][1]) if cigar and int(cigar[0][0]) == 5 else 0
    trailing = int(cigar[-1][1]) if cigar and int(cigar[-1][0]) == 5 else 0
    return leading, trailing


def _ma_interval_to_query(
    *,
    start: int,
    length: int,
    ma_read_length: int,
    read,
) -> Tuple[int, int, bool]:
    """Return stored-query bounds and whether the full interval was retained."""

    if start < 0 or length <= 0 or start + length > ma_read_length:
        raise BamFootprintInputError(
            f"invalid MA interval {start}-{length} for read length {ma_read_length}"
        )
    query_length = int(read.query_length or 0)
    leading_hard, trailing_hard = _hard_clip_lengths(read)
    if ma_read_length == query_length:
        stored_start = 0
        stored_end = query_length
    elif ma_read_length == query_length + leading_hard + trailing_hard:
        stored_start = leading_hard
        stored_end = ma_read_length - trailing_hard
    else:
        raise BamFootprintInputError(
            f"MA read length {ma_read_length} does not match stored query length "
            f"{query_length} (plus {leading_hard}+{trailing_hard} hard-clipped bases)"
        )

    seq_start = ma_read_length - (start + length) if read.is_reverse else start
    seq_end = seq_start + length
    clipped_start = max(seq_start, stored_start)
    clipped_end = min(seq_end, stored_end)
    complete = clipped_start == seq_start and clipped_end == seq_end
    if clipped_end <= clipped_start:
        return 0, 0, False
    return clipped_start - stored_start, clipped_end - stored_start, complete


def _validate_ma_against_read(parsed_ma: Mapping[str, object], read) -> None:
    ma_read_length = int(parsed_ma["read_length"])
    query_length = int(read.query_length or 0)
    leading_hard, trailing_hard = _hard_clip_lengths(read)
    if ma_read_length not in {
        query_length,
        query_length + leading_hard + trailing_hard,
    }:
        raise BamFootprintInputError(
            f"MA read length {ma_read_length} does not match stored query length "
            f"{query_length} (plus {leading_hard}+{trailing_hard} hard-clipped bases)"
        )
    for name, _strand, _quality_spec, intervals in parsed_ma["raw_types"]:  # type: ignore[index]
        for raw_start, raw_length in intervals:
            start, length = int(raw_start), int(raw_length)
            if start < 0 or length <= 0 or start + length > ma_read_length:
                raise BamFootprintInputError(
                    f"invalid {name} MA interval {start}-{length} for read length {ma_read_length}"
                )


def _project_query_interval(
    query_to_ref: np.ndarray,
    query_start: int,
    query_end: int,
    original_length: int,
    complete_in_stored_query: bool,
) -> Tuple[Optional[Tuple[int, int]], float, bool]:
    if query_end <= query_start or original_length <= 0:
        return None, 0.0, False
    bounded_start = max(0, int(query_start))
    bounded_end = min(len(query_to_ref), int(query_end))
    if bounded_end <= bounded_start:
        return None, 0.0, False
    positions = query_to_ref[bounded_start:bounded_end]
    mapped = positions[positions >= 0]
    mapped_fraction = float(len(mapped)) / float(original_length)
    if len(mapped) == 0:
        return None, mapped_fraction, False
    endpoints_mapped = bool(
        complete_in_stored_query
        and int(query_to_ref[bounded_start]) >= 0
        and int(query_to_ref[bounded_end - 1]) >= 0
    )
    return (
        (int(mapped.min()), int(mapped.max()) + 1),
        mapped_fraction,
        endpoints_mapped,
    )


def _center_is_in_scope(
    contig: str,
    start: int,
    end: int,
    regions_by_contig: Mapping[str, Sequence[ReferenceRegion]],
) -> bool:
    if not regions_by_contig:
        return True
    doubled_center = start + end
    return any(
        2 * region.start <= doubled_center < 2 * region.end
        for region in regions_by_contig.get(contig, ())
    )


def _alignment_identity(read) -> Tuple[object, ...]:
    return (
        read.query_name,
        int(read.flag),
        int(read.reference_id),
        int(read.reference_start),
        int(read.reference_end or read.reference_start),
        read.cigarstring,
    )


def _iter_records(bam, regions: Sequence[ReferenceRegion]):
    if not regions:
        yield from bam.fetch(until_eof=True)
        return
    for region in regions:
        yield from bam.fetch(region.contig, region.start, region.end)


def load_footprint_molecules_from_bam(
    input_bam: Union[str, Path],
    *,
    regions: Sequence[str] = (),
    min_mapq: int = 0,
    include_duplicates: bool = False,
    minimum_projection_fraction: float = 0.95,
    invalid_stratum_tag_policy: str = "error",
) -> BamFootprintInput:
    """Load ordinary TF/MSP annotations and denominator molecules from BAM.

    Region coordinates are zero-based and half-open.  An indexed BAM is
    required only when one or more regions are supplied.  TF calls are retained
    when at least one base projects; ``geometry_eligible`` additionally requires
    the configured mapped fraction and both projected endpoints.  MSPs below
    the mapped-fraction threshold are not used for containment statistics.

    ``invalid_stratum_tag_policy='alignment'`` is an explicit compatibility
    escape hatch for legacy BAMs that used the optional ``st`` tag for an
    unrelated value (for example a timestamp).  The strict default remains an
    error.  DAF callers using the escape hatch should subsequently replace the
    provisional alignment stratum with one inferred from assay evidence.
    """

    if isinstance(min_mapq, bool) or not isinstance(min_mapq, int) or min_mapq < 0:
        raise BamFootprintInputError("min_mapq must be a non-negative integer")
    if not 0 < float(minimum_projection_fraction) <= 1:
        raise BamFootprintInputError("minimum_projection_fraction must be in (0, 1]")
    if invalid_stratum_tag_policy not in {"error", "alignment"}:
        raise BamFootprintInputError(
            "invalid_stratum_tag_policy must be 'error' or 'alignment'"
        )
    if isinstance(regions, (str, bytes)):
        raise BamFootprintInputError("regions must be a sequence of region strings")

    resolved = Path(input_bam).expanduser().resolve()
    if not resolved.is_file():
        raise BamFootprintInputError(f"input BAM does not exist: {resolved}")

    counts: Dict[str, int] = {field: 0 for field in BamFootprintDiagnostics.__dataclass_fields__}
    molecules: List[BaselineMolecule] = []
    with pysam.AlignmentFile(str(resolved), "rb", check_sq=False) as bam:
        references = tuple(
            (str(contig), int(length)) for contig, length in zip(bam.references, bam.lengths)
        )
        reference_lengths = dict(references)
        reference_order = {contig: index for index, (contig, _length) in enumerate(references)}
        parsed_regions = _merge_regions(
            (parse_reference_region(value, reference_lengths) for value in regions),
            reference_order,
        )
        if parsed_regions and not bam.has_index():
            raise BamFootprintInputError("--region requires an indexed BAM")

        regions_by_contig: Dict[str, List[ReferenceRegion]] = {}
        for region in parsed_regions:
            regions_by_contig.setdefault(region.contig, []).append(region)

        seen_alignments = set()
        seen_molecule_keys = set()
        for read in _iter_records(bam, parsed_regions):
            counts["records_seen"] += 1
            if len(parsed_regions) > 1:
                identity = _alignment_identity(read)
                if identity in seen_alignments:
                    counts["duplicate_region_records"] += 1
                    continue
                seen_alignments.add(identity)

            if read.is_unmapped:
                counts["skipped_unmapped"] += 1
                continue
            if read.is_secondary:
                counts["skipped_secondary"] += 1
                continue
            if read.is_supplementary:
                counts["skipped_supplementary"] += 1
                continue
            if read.is_qcfail:
                counts["skipped_qcfail"] += 1
                continue
            if read.is_duplicate and not include_duplicates:
                counts["skipped_duplicate"] += 1
                continue
            if int(read.mapping_quality) < min_mapq:
                counts["skipped_low_mapq"] += 1
                continue
            counts["primary_mapped_records"] += 1

            if not read.has_tag("MA"):
                counts["skipped_missing_ma"] += 1
                continue
            try:
                parsed_ma = parse_ma_tag(read.get_tag("MA"))
                _validate_ma_against_read(parsed_ma, read)
            except (TypeError, ValueError) as error:
                raise BamFootprintInputError(
                    f"malformed MA tag on read {read.query_name!r}: {error}"
                ) from error
            if read.has_tag("AN") and any(
                name in {"tf", "msp"} and intervals
                for name, _strand, _quality_spec, intervals in parsed_ma["raw_types"]
            ):
                raise BamFootprintInputError(
                    f"AN-linked ordinary tf/msp annotations on read "
                    f"{read.query_name!r} are not supported by the linear "
                    "footprint-model adapter"
                )

            mapped_blocks = _merged_reference_blocks(read)
            if not mapped_blocks:
                counts["skipped_without_mapped_blocks"] += 1
                continue
            contig = str(read.reference_name)
            molecule_id = str(read.query_name or "")
            if not molecule_id:
                raise BamFootprintInputError("primary BAM record has an empty query name")
            if read.has_tag("st"):
                stratum = str(read.get_tag("st")).upper()
                if stratum not in {"CT", "GA"}:
                    if invalid_stratum_tag_policy == "error":
                        raise BamFootprintInputError(
                            f"invalid st tag on read {molecule_id!r}: expected CT or GA, "
                            f"found {stratum!r}"
                        )
                    stratum = "REV" if read.is_reverse else "FWD"
                    counts["reads_with_alignment_stratum"] += 1
                else:
                    counts["reads_with_tag_stratum"] += 1
            else:
                stratum = "REV" if read.is_reverse else "FWD"
                counts["reads_with_alignment_stratum"] += 1
            molecule_key = (contig, molecule_id)
            if molecule_key in seen_molecule_keys:
                raise BamFootprintInputError(
                    f"multiple primary records share contig/read identity {contig}:{molecule_id}"
                )
            seen_molecule_keys.add(molecule_key)

            query_to_ref = cigar_to_query_ref(read)
            ma_read_length = int(parsed_ma["read_length"])
            tf_observations: List[TFObservation] = []
            msp_intervals: List[Tuple[int, int]] = []
            tf_ordinal = 0
            for name, _strand, _quality_spec, intervals in parsed_ma["raw_types"]:
                if name not in {"tf", "msp"}:
                    continue
                for raw_start, raw_length in intervals:
                    start, length = int(raw_start), int(raw_length)
                    try:
                        query_start, query_end, complete = _ma_interval_to_query(
                            start=start,
                            length=length,
                            ma_read_length=ma_read_length,
                            read=read,
                        )
                    except BamFootprintInputError as error:
                        raise BamFootprintInputError(
                            f"invalid {name} annotation on read {molecule_id!r}: {error}"
                        ) from error
                    projected, mapped_fraction, endpoints_mapped = _project_query_interval(
                        query_to_ref,
                        query_start,
                        query_end,
                        length,
                        complete,
                    )
                    if name == "tf":
                        counts["tf_annotations"] += 1
                        current_ordinal = tf_ordinal
                        tf_ordinal += 1
                        if projected is None:
                            counts["unprojectable_tf_annotations"] += 1
                            continue
                        ref_start, ref_end = projected
                        if not _center_is_in_scope(
                            contig,
                            ref_start,
                            ref_end,
                            regions_by_contig,
                        ):
                            counts["out_of_scope_tf_annotations"] += 1
                            continue
                        geometry_eligible = bool(
                            mapped_fraction >= minimum_projection_fraction and endpoints_mapped
                        )
                        tf_observations.append(
                            TFObservation(
                                call_id=f"ma.tf.{current_ordinal:06d}",
                                start=ref_start,
                                end=ref_end,
                                geometry_eligible=geometry_eligible,
                            )
                        )
                        counts["projected_tf_annotations"] += 1
                        if geometry_eligible:
                            counts["geometry_eligible_tf_annotations"] += 1
                    else:
                        counts["msp_annotations"] += 1
                        if projected is None:
                            counts["unprojectable_msp_annotations"] += 1
                            continue
                        if mapped_fraction < minimum_projection_fraction:
                            counts["incomplete_msp_annotations"] += 1
                            continue
                        msp_intervals.append(projected)
                        counts["projected_msp_annotations"] += 1

            molecules.append(
                BaselineMolecule(
                    molecule_id=molecule_id,
                    contig=contig,
                    mapped_blocks=mapped_blocks,
                    tfs=tuple(tf_observations),
                    msps=tuple(sorted(set(msp_intervals))),
                    stratum=stratum,
                )
            )
            counts["emitted_molecules"] += 1

    diagnostics = BamFootprintDiagnostics(**counts)
    return BamFootprintInput(
        input_path=resolved,
        molecules=tuple(molecules),
        diagnostics=diagnostics,
        regions=parsed_regions,
        references=references,
    )


__all__ = [
    "BamFootprintDiagnostics",
    "BamFootprintInput",
    "BamFootprintInputError",
    "ReferenceRegion",
    "load_footprint_molecules_from_bam",
    "parse_reference_region",
]
