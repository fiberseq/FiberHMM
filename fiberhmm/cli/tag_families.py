#!/usr/bin/env python3
"""Add compact TF-family identity and assignment confidence to ``tf_sr``.

The ordinary ``tf`` layer and every interval in the complete ``tf_sr`` shadow
layer are preserved.  ``tf_sr.QQQ`` becomes ``tf_sr.QQQQQ`` with two aligned
bytes appended per annotation:

``fi``
    Locally reusable family identifier.  Zero means unassigned.
``fq``
    ``round(255 * assignment_confidence)``. The producer declares calibration
    in the assignment artifact; this is not a biological occupancy posterior.

Assignments are supplied as an auditable TSV so family fitting remains
separate from BAM materialization and can be cross-fitted when appropriate.
"""
from __future__ import annotations

import argparse
import array
import csv
import hashlib
import json
import os
import shlex
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import pysam

from fiberhmm import __version__ as FIBERHMM_VERSION
from fiberhmm.cli.common import add_version_args
from fiberhmm.io.bam_header import append_pg_record, declared_ma_types
from fiberhmm.io.ma_tags import parse_aq_array, parse_ma_tag
from fiberhmm.core.bam_reader import cigar_to_query_ref
from fiberhmm.io.footprint_bam import _ma_interval_to_query, _project_query_interval
from fiberhmm.inference.tf_family_ids import DEFAULT_FAMILY_SEPARATION_BP


FAMILY_HEADER_PREFIX = "FIBERHMM-TF-FAMILY:v1:"
SUPPORTED_STRAND_RESCUE_PREFIXES = (
    "FIBERHMM-STRAND-RESCUE:v6:",
    "FIBERHMM-STRAND-RESCUE:v4:",
)
LEGACY_STRAND_RESCUE_PREFIXES = (
    "FIBERHMM-STRAND-RESCUE:v3:",
    "FIBERHMM-STRAND-RESCUE:v2:",
)
FAMILY_SEPARATION_BP = DEFAULT_FAMILY_SEPARATION_BP
SOURCE_QUALITY_SPEC = "QQQ"
FAMILY_QUALITY_SPEC = "QQQQQ"
ASSIGNMENT_FIELDS = (
    "read_name",
    "alignment_occurrence",
    "tf_sr_ordinal",
    "family_id",
    "assignment_probability",
    "family_key",
    "contig",
    "call_start",
    "call_end",
)
ASSIGNMENT_FIELDS_V2 = ASSIGNMENT_FIELDS + ("calibration_scope",)


@dataclass(frozen=True)
class FamilyAssignment:
    read_name: str
    alignment_occurrence: int
    tf_sr_ordinal: int
    family_id: int
    assignment_probability: float
    family_key: str
    contig: str
    call_start: int
    call_end: int
    calibration_scope: str = "legacy_unspecified"

    @property
    def confidence_q(self) -> int:
        return max(1, min(255, int(round(255.0 * self.assignment_probability))))

    @property
    def key(self) -> Tuple[str, int, int]:
        return self.read_name, self.alignment_occurrence, self.tf_sr_ordinal


def _nonnegative_int(value: object, context: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be an integer") from error
    if result < 0:
        raise ValueError(f"{context} must be nonnegative")
    return result


def load_family_assignments(path: str | Path) -> Dict[Tuple[str, int, int], FamilyAssignment]:
    """Load and strictly validate a family-call assignment TSV."""

    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise ValueError(f"missing assignment TSV: {candidate}")
    assignments: Dict[Tuple[str, int, int], FamilyAssignment] = {}
    family_contracts: dict[str, tuple[int, str]] = {}
    family_hulls: dict[tuple[int, str, str], list[int]] = {}
    with candidate.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        assignment_fields = tuple(reader.fieldnames or ())
        if assignment_fields not in (ASSIGNMENT_FIELDS, ASSIGNMENT_FIELDS_V2):
            raise ValueError(
                "assignment TSV fields differ from the v1/v2 schemas; expected "
                + ",".join(ASSIGNMENT_FIELDS)
                + " or "
                + ",".join(ASSIGNMENT_FIELDS_V2)
            )
        for line_number, row in enumerate(reader, start=2):
            read_name = str(row["read_name"] or "")
            family_key = str(row["family_key"] or "")
            contig = str(row["contig"] or "")
            if not read_name or not family_key or not contig:
                raise ValueError(f"assignment line {line_number} has an empty identifier")
            family_id = _nonnegative_int(row["family_id"], f"line {line_number} family_id")
            if not 1 <= family_id <= 255:
                raise ValueError(f"line {line_number} family_id must be in [1,255]")
            try:
                probability = float(row["assignment_probability"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"line {line_number} assignment_probability must be numeric"
                ) from error
            if not 0.0 < probability <= 1.0:
                raise ValueError(
                    f"line {line_number} assignment_probability must be in (0,1]"
                )
            calibration_scope = (
                str(row["calibration_scope"] or "")
                if assignment_fields == ASSIGNMENT_FIELDS_V2
                else "legacy_unspecified"
            )
            if not calibration_scope:
                raise ValueError(f"assignment line {line_number} has an empty calibration_scope")
            assignment = FamilyAssignment(
                read_name=read_name,
                alignment_occurrence=_nonnegative_int(
                    row["alignment_occurrence"],
                    f"line {line_number} alignment_occurrence",
                ),
                tf_sr_ordinal=_nonnegative_int(
                    row["tf_sr_ordinal"], f"line {line_number} tf_sr_ordinal"
                ),
                family_id=family_id,
                assignment_probability=probability,
                family_key=family_key,
                contig=contig,
                call_start=_nonnegative_int(row["call_start"], f"line {line_number} call_start"),
                call_end=_nonnegative_int(row["call_end"], f"line {line_number} call_end"),
                calibration_scope=calibration_scope,
            )
            if assignment.call_end <= assignment.call_start:
                raise ValueError(f"line {line_number} call interval is empty")
            if assignment.key in assignments:
                raise ValueError(f"duplicate assignment key on line {line_number}")
            prior_contract = family_contracts.get(family_key)
            current_contract = (family_id, contig)
            if prior_contract is not None and prior_contract != current_contract:
                raise ValueError(
                    f"family_key {family_key!r} maps to inconsistent family_id/contig"
                )
            family_contracts[family_key] = current_contract
            hull = family_hulls.setdefault(
                (family_id, contig, family_key),
                [assignment.call_start, assignment.call_end],
            )
            hull[0] = min(hull[0], assignment.call_start)
            hull[1] = max(hull[1], assignment.call_end)
            assignments[assignment.key] = assignment
    by_slot: defaultdict[tuple[int, str], list[tuple[str, int, int]]] = defaultdict(list)
    for (family_id, contig, family_key), (start, end) in family_hulls.items():
        by_slot[(family_id, contig)].append((family_key, start, end))
    for (family_id, contig), rows in by_slot.items():
        rows.sort(key=lambda value: (value[1], value[2], value[0]))
        for previous, current in zip(rows, rows[1:]):
            if current[1] < previous[2] + FAMILY_SEPARATION_BP:
                raise ValueError(
                    f"family_id {family_id} is reused by nearby family keys "
                    f"{previous[0]!r} and {current[0]!r} on {contig}"
                )
    return assignments


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_groups(
    read_length: int,
    groups: Sequence[
        Tuple[str, str, str, Sequence[Tuple[int, int]], Sequence[Sequence[int]]]
    ],
) -> Tuple[str, array.array]:
    parts = [str(int(read_length))]
    qualities = array.array("B")
    for name, strand, quality_spec, intervals, rows in groups:
        if len(intervals) != len(rows):
            raise ValueError(f"{name} interval/quality count mismatch")
        if not intervals:
            continue
        tokens = []
        for (start, length), row in zip(intervals, rows):
            if start < 0 or length <= 0 or start + length > read_length:
                raise ValueError(f"{name} interval lies outside the molecule")
            if len(row) != len(quality_spec):
                raise ValueError(f"{name} quality arity mismatch")
            tokens.append(f"{int(start) + 1}-{int(length)}")
            qualities.extend(max(0, min(255, int(value))) for value in row)
        parts.append(f"{name}{strand}{quality_spec}:" + ",".join(tokens))
    return ";".join(parts), qualities


def _family_header(header, *, assignment_sha256: str, command_line: str):
    header_dict = header.to_dict()
    comments = [
        str(comment)
        for comment in header_dict.get("CO", [])
        if not str(comment).startswith(FAMILY_HEADER_PREFIX)
    ]
    comments.append(
        FAMILY_HEADER_PREFIX
        + "layer=tf_sr;quality_spec=QQQQQ"
        + ";q0=sr_state_or_action_probability"
        + ";q1=molecular_left_edge_confidence"
        + ";q2=molecular_right_edge_confidence"
        + ";q3=fi_local_repeating_family_id_uint8"
        + ";fi_zero=unassigned"
        + ";fi_identity=reference_neighborhood_plus_fi"
        + ";fi_overlap_collision_forbidden=true"
        + f";fi_reuse_separation_bp={FAMILY_SEPARATION_BP}"
        + ";q4=fq_conditional_family_assignment_confidence"
        + ";fq_scale=round_255_times_confidence"
        + ";fq_producer_calibration=declared_in_assignment_artifact"
        + f";assignments_sha256={assignment_sha256}"
    )
    header_dict["CO"] = comments
    output = pysam.AlignmentHeader.from_dict(header_dict)
    from fiberhmm.io.annotation_frame import append_coord_to_ds, pass_through_frame
    return append_pg_record(
        output,
        {
            "PN": "fiberhmm-tag-consensus",
            "VN": FIBERHMM_VERSION,
            "CL": command_line,
            "DS": append_coord_to_ds(
                "BAM-native local TF-family IDs and conditional assignment "
                f"confidence; assignments_sha256={assignment_sha256}",
                pass_through_frame(output),
            ),
        },
    )


def tag_tf_families(
    input_bam: str | Path,
    output_bam: str | Path,
    assignment_tsv: str | Path,
    *,
    force: bool = False,
    command_line: str = "fiberhmm-tag-consensus",
) -> Mapping[str, object]:
    """Materialize assignment rows into ``tf_sr`` while preserving geometry."""

    source = Path(input_bam).expanduser().resolve()
    output = Path(output_bam).expanduser().resolve()
    assignment_path = Path(assignment_tsv).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"missing input BAM: {source}")
    if source == output:
        raise ValueError("input and output BAM paths must differ")
    if output.exists() and not force:
        raise ValueError(f"output exists; use --force to replace it: {output}")
    for index_suffix in (".bai", ".csi"):
        if Path(str(output) + index_suffix).exists() and not force:
            raise ValueError(f"output index exists; use --force to replace it")
    assignments = load_family_assignments(assignment_path)
    assignment_sha256 = _sha256(assignment_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    observed_occurrences: defaultdict[str, int] = defaultdict(int)
    applied: set[Tuple[str, int, int]] = set()
    temporary_path = None
    temporary_index = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".bam", dir=output.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        with pysam.AlignmentFile(source, "rb", check_sq=False) as input_handle:
            comments = [str(value) for value in input_handle.header.to_dict().get("CO", [])]
            if "tf_sr" not in declared_ma_types(input_handle.header):
                raise ValueError(
                    "input BAM does not declare the prerequisite tf_sr layer in "
                    "@CO MA-TYPES:v1; run strand-rescue annotation first"
                )
            modern_contracts = [
                comment for comment in comments
                if comment.startswith(SUPPORTED_STRAND_RESCUE_PREFIXES)
            ]
            legacy_contracts = [
                comment for comment in comments
                if comment.startswith(LEGACY_STRAND_RESCUE_PREFIXES)
            ]
            if len(modern_contracts) != 1:
                detail = "legacy v2/v3 contract detected" if legacy_contracts else "modern contract missing"
                raise ValueError(
                    "input BAM must carry exactly one v4/v6 strand-rescue contract; " + detail
                )
            input_has_family_contract = any(
                comment.startswith(FAMILY_HEADER_PREFIX) for comment in comments
            )
            output_header = _family_header(
                input_handle.header,
                assignment_sha256=assignment_sha256,
                command_line=command_line,
            )
            with pysam.AlignmentFile(temporary_path, "wb", header=output_header) as output_handle:
                for read in input_handle.fetch(until_eof=True):
                    counts["records"] += 1
                    occurrence = observed_occurrences[read.query_name]
                    observed_occurrences[read.query_name] += 1
                    if not read.has_tag("MA"):
                        output_handle.write(read)
                        continue
                    parsed = parse_ma_tag(read.get_tag("MA"))
                    raw_types = parsed["raw_types"]
                    aq = read.get_tag("AQ") if read.has_tag("AQ") else []
                    expected_aq = sum(
                        len(specification) * len(intervals)
                        for _name, _strand, specification, intervals in raw_types
                    )
                    if len(aq) != expected_aq:
                        raise ValueError(
                            f"read {read.query_name!r}: AQ has {len(aq)} bytes; "
                            f"MA requires {expected_aq}"
                        )
                    per_annotation = parse_aq_array(
                        aq,
                        [item[2] for item in raw_types],
                        [len(item[3]) for item in raw_types],
                    )
                    groups = []
                    cursor = 0
                    tf_sr_groups = sum(name == "tf_sr" for name, *_rest in raw_types)
                    if tf_sr_groups > 1:
                        raise ValueError(f"read {read.query_name!r}: multiple tf_sr groups")
                    query_to_ref = None
                    for name, strand, specification, intervals in raw_types:
                        rows = [list(row) for row in per_annotation[cursor:cursor + len(intervals)]]
                        cursor += len(intervals)
                        if name == "tf_sr":
                            if specification not in {SOURCE_QUALITY_SPEC, FAMILY_QUALITY_SPEC}:
                                raise ValueError(
                                    f"read {read.query_name!r}: tf_sr must use QQQ or QQQQQ"
                                )
                            if specification == FAMILY_QUALITY_SPEC and not input_has_family_contract:
                                raise ValueError(
                                    f"read {read.query_name!r}: tf_sr.QQQQQ belongs to a legacy "
                                    "strand-rescue contract and cannot be overwritten as fi/fq"
                                )
                            for ordinal, row in enumerate(rows):
                                key = (read.query_name, occurrence, ordinal)
                                assignment = assignments.get(key)
                                prefix = row[:3]
                                if assignment is None:
                                    rows[ordinal] = prefix + [0, 0]
                                else:
                                    if read.reference_name != assignment.contig:
                                        raise ValueError(
                                            f"assignment contig differs for read {read.query_name!r}"
                                        )
                                    if query_to_ref is None:
                                        query_to_ref = cigar_to_query_ref(read)
                                    molecular_start, molecular_length = intervals[ordinal]
                                    query_start, query_end, complete = _ma_interval_to_query(
                                        start=int(molecular_start),
                                        length=int(molecular_length),
                                        ma_read_length=int(parsed["read_length"]),
                                        read=read,
                                    )
                                    projected, _mapped_fraction, _endpoints_mapped = _project_query_interval(
                                        query_to_ref,
                                        query_start,
                                        query_end,
                                        int(molecular_length),
                                        complete,
                                    )
                                    if projected != (assignment.call_start, assignment.call_end):
                                        raise ValueError(
                                            f"assignment interval differs for read {read.query_name!r} "
                                            f"tf_sr ordinal {ordinal}: {projected} != "
                                            f"{(assignment.call_start, assignment.call_end)}"
                                        )
                                    rows[ordinal] = prefix + [
                                        assignment.family_id,
                                        assignment.confidence_q,
                                    ]
                                    applied.add(key)
                                    counts["assigned_tf_sr"] += 1
                            specification = FAMILY_QUALITY_SPEC
                            counts["tf_sr_annotations"] += len(intervals)
                        groups.append((name, strand, specification, intervals, rows))
                    ma_value, aq_value = _format_groups(int(parsed["read_length"]), groups)
                    read.set_tag("MA", ma_value, value_type="Z")
                    if aq_value:
                        read.set_tag("AQ", aq_value)
                    elif read.has_tag("AQ"):
                        read.set_tag("AQ", None)
                    output_handle.write(read)
        missing = sorted(set(assignments) - applied)
        if missing:
            example = missing[0]
            raise ValueError(
                f"{len(missing)} assignment rows did not match a tf_sr annotation; "
                f"first={example}"
            )
        pysam.index(str(temporary_path))
        index_candidates = [
            Path(str(temporary_path) + suffix)
            for suffix in (".bai", ".csi")
            if Path(str(temporary_path) + suffix).exists()
        ]
        if len(index_candidates) != 1:
            raise ValueError("indexing did not produce exactly one BAI/CSI index")
        temporary_index = index_candidates[0]
        index_suffix = temporary_index.suffix
        os.replace(temporary_path, output)
        os.replace(temporary_index, Path(str(output) + index_suffix))
        alternate_suffix = ".csi" if index_suffix == ".bai" else ".bai"
        alternate_index = Path(str(output) + alternate_suffix)
        if force and alternate_index.exists():
            alternate_index.unlink()
        temporary_path = None
        temporary_index = None
    finally:
        for candidate in (temporary_path, temporary_index):
            if candidate is not None and candidate.exists():
                candidate.unlink()

    return {
        "schema": "fiberhmm.tf_family_tagging.v1",
        "input_bam": str(source),
        "output_bam": str(output),
        "assignment_tsv": str(assignment_path),
        "assignment_sha256": assignment_sha256,
        "assignment_rows": len(assignments),
        **dict(sorted(counts.items())),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fiberhmm-tag-consensus",
        description=(
            "Append local TF-family ID and assignment-confidence bytes to the "
            "complete tf_sr Molecular Annotation layer."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Assignment TSV (-a): tab-separated with exactly this header row\n"
            "  read_name alignment_occurrence tf_sr_ordinal family_id\n"
            "  assignment_probability family_key contig call_start call_end\n"
            "and optionally a final calibration_scope column (v2). One row per\n"
            "assigned tf_sr call:\n"
            "  alignment_occurrence  0-based occurrence of read_name in file order\n"
            "  tf_sr_ordinal         0-based position of the call in the record's\n"
            "                        tf_sr group\n"
            "  family_id             1-255; written as fi (0 = unassigned)\n"
            "  assignment_probability  (0,1]; written as fq = round(255 x p)\n"
            "  family_key            stable family identifier (one family_id and\n"
            "                        contig per key)\n"
            "  contig, call_start, call_end  the call's reference interval,\n"
            "                        0-based half-open; must match the BAM exactly\n"
            "  calibration_scope     how the probability was calibrated (v2)\n"
            f"A family_id may be reused on one contig only by families at least\n"
            f"{FAMILY_SEPARATION_BP} bp apart. Every row must match a tf_sr call."
        ),
    )
    add_version_args(parser)
    parser.add_argument("-i", "--input", required=True, help="Input BAM with tf_sr.QQQ")
    parser.add_argument("-o", "--output", required=True, help="New sorted/indexed BAM")
    parser.add_argument(
        "-a", "--assignments", required=True, help="v1/v2 family assignment TSV"
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command_line = " ".join(
        ["fiberhmm-tag-consensus"]
        + [shlex.quote(str(value)) for value in (sys.argv[1:] if argv is None else argv)]
    )
    try:
        result = tag_tf_families(
            args.input,
            args.output,
            args.assignments,
            force=args.force,
            command_line=command_line,
        )
    except (OSError, ValueError, pysam.utils.SamtoolsError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ASSIGNMENT_FIELDS",
    "ASSIGNMENT_FIELDS_V2",
    "FAMILY_HEADER_PREFIX",
    "FAMILY_QUALITY_SPEC",
    "FamilyAssignment",
    "load_family_assignments",
    "tag_tf_families",
]
