#!/usr/bin/env python3
"""Audit manifests and adjudicate hierarchical cross-assay evidence."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pysam

try:
    from consensus_recaller_collab.prototype import PRESETS, parse_region
    from consensus_recaller_collab.validation.hierarchy import (
        AXES,
        adjudicate_all,
        load_json,
        validate_hierarchy,
    )
    from consensus_recaller_collab.validation.evidence import build_evidence
    from consensus_recaller_collab.validation.calibration import (
        calibrate_fine_tf,
        merge_summaries,
        write_proposal_tsv,
    )
    from consensus_recaller_collab.validation.summary import summarize_fine_tf
    from consensus_recaller_collab.validation.sensitivity import audit_molecule_collapse
except ModuleNotFoundError:  # Direct execution from this directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from consensus_recaller_collab.prototype import PRESETS, parse_region
    from consensus_recaller_collab.validation.hierarchy import (
        AXES,
        adjudicate_all,
        load_json,
        validate_hierarchy,
    )
    from consensus_recaller_collab.validation.evidence import build_evidence
    from consensus_recaller_collab.validation.calibration import (
        calibrate_fine_tf,
        merge_summaries,
        write_proposal_tsv,
    )
    from consensus_recaller_collab.validation.summary import summarize_fine_tf
    from consensus_recaller_collab.validation.sensitivity import audit_molecule_collapse


AVAILABILITY = ("available", "missing_local", "disabled")


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _resolve_bam(path: str, manifest_path: Optional[str]) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or manifest_path is None:
        return candidate
    if candidate.exists():
        return candidate.resolve()
    manifest_dir = Path(manifest_path).resolve().parent
    # A manifest may be nested several directories below the project root but
    # still use project-relative input paths.  Search from the supplied
    # manifest outward rather than from this module: after installation,
    # ``__file__`` lives in site-packages and has no relationship to the data.
    for base in (manifest_dir, *manifest_dir.parents):
        manifest_relative = base / candidate
        if manifest_relative.exists():
            return manifest_relative.resolve()
    # Preserve a deterministic, manifest-relative missing path for audit
    # diagnostics instead of misleadingly rebasing it into site-packages.
    return manifest_dir / candidate


def _locus_allowed(sample: Mapping[str, object], locus: Mapping[str, object]) -> bool:
    restricted_loci = sample.get("restrict_loci")
    if restricted_loci is not None and locus["locus_id"] not in restricted_loci:
        return False
    restricted_groups = sample.get("restrict_comparison_groups")
    if restricted_groups is not None and locus.get("comparison_group") not in restricted_groups:
        return False
    return sample.get("cohort_id") == locus.get("cohort_id")


def validate_manifest_structure(manifest: Mapping[str, object]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    cohorts = manifest.get("cohorts")
    loci = manifest.get("loci")
    samples = manifest.get("samples")
    if not isinstance(cohorts, list):
        errors.append("manifest.cohorts must be a list")
        cohorts = []
    if not isinstance(loci, list):
        errors.append("manifest.loci must be a list")
        loci = []
    if not isinstance(samples, list):
        errors.append("manifest.samples must be a list")
        samples = []

    def unique_ids(records, key):
        values = [str(record.get(key, "")) for record in records]
        if any(not value for value in values):
            errors.append(f"every {key} must be non-empty")
        if len(values) != len(set(values)):
            errors.append(f"duplicate {key} values")
        return set(values)

    cohort_ids = unique_ids(cohorts, "cohort_id")
    locus_ids = unique_ids(loci, "locus_id")
    unique_ids(samples, "sample_id")

    for locus in loci:
        if locus.get("cohort_id") not in cohort_ids:
            errors.append(f"{locus.get('locus_id')}: unknown cohort")
        try:
            parse_region(str(locus.get("region", "")))
        except (ValueError, argparse.ArgumentTypeError):
            errors.append(f"{locus.get('locus_id')}: invalid region")

    for sample in samples:
        sample_id = str(sample.get("sample_id", "<unknown>"))
        if sample.get("cohort_id") not in cohort_ids:
            errors.append(f"{sample_id}: unknown cohort")
        if sample.get("preset") not in PRESETS:
            errors.append(f"{sample_id}: unknown preset {sample.get('preset')}")
        availability = sample.get("availability", "available")
        if availability not in AVAILABILITY:
            errors.append(f"{sample_id}: unknown availability {availability}")
        bams = sample.get("bams")
        if not isinstance(bams, list) or not bams:
            errors.append(f"{sample_id}: bams must be a non-empty list")
        restricted = sample.get("restrict_loci", [])
        if any(locus_id not in locus_ids for locus_id in restricted):
            errors.append(f"{sample_id}: restrict_loci contains an unknown locus")
        permissions = sample.get("axis_permissions", {})
        for axis in AXES:
            if permissions.get(axis) not in ("anchor", "support", "geometry_only", "none"):
                errors.append(f"{sample_id}: invalid or missing permission for {axis}")
        if sample.get("ascertainment") == "positive_only" and sample.get(
            "negative_evidence_allowed", True
        ):
            errors.append(f"{sample_id}: positive-only data cannot provide negative evidence")
        if not sample.get("truth_vote", True):
            warnings.append(f"{sample_id}: excluded from default truth voting")
    return errors, warnings


def _bam_audit(path: Path, sample: Mapping[str, object], loci: Sequence[Mapping[str, object]]) -> dict:
    result = {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
        "indexed": False,
        "tag_sample": {},
        "locus_read_counts": {},
        "errors": [],
    }
    if not path.exists():
        return result
    try:
        with pysam.AlignmentFile(str(path), "rb") as bam:
            result["indexed"] = bool(bam.has_index())
            tag_counts: Dict[str, int] = {tag: 0 for tag in ("MA", "AQ", "MM", "ML")}
            sampled = 0
            for read in bam.fetch(until_eof=True):
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                sampled += 1
                for tag in tag_counts:
                    tag_counts[tag] += int(read.has_tag(tag))
                if sampled >= 25:
                    break
            result["tag_sample"] = {"reads": sampled, **tag_counts}
        if result["indexed"]:
            with pysam.AlignmentFile(str(path), "rb") as bam:
                references = set(bam.references)
                for locus in loci:
                    chrom, start, end = parse_region(str(locus["region"]))
                    if chrom not in references:
                        result["locus_read_counts"][str(locus["locus_id"])] = None
                        continue
                    result["locus_read_counts"][str(locus["locus_id"])] = bam.count(
                        chrom, start, end, read_callback="nofilter"
                    )
    except (OSError, ValueError) as error:
        result["errors"].append(str(error))
    return result


def audit_manifest(manifest: Mapping[str, object], manifest_path: Optional[str]) -> dict:
    errors, warnings = validate_manifest_structure(manifest)
    loci = manifest.get("loci", [])
    bam_audits = []
    for sample in manifest.get("samples", []):
        availability = sample.get("availability", "available")
        sample_loci = [locus for locus in loci if _locus_allowed(sample, locus)]
        for bam_path in sample.get("bams", []):
            path = _resolve_bam(str(bam_path), manifest_path)
            if availability == "available":
                audit = _bam_audit(path, sample, sample_loci)
                if not audit["exists"]:
                    errors.append(f"{sample['sample_id']}: missing BAM {path}")
                if audit["errors"]:
                    errors.extend(
                        f"{sample['sample_id']}: {message}" for message in audit["errors"]
                    )
                bam_audits.append({"sample_id": sample["sample_id"], **audit})
            elif not path.exists():
                warnings.append(f"{sample['sample_id']}: unavailable as declared: {path}")

    coverage = []
    for locus in loci:
        row = {
            "locus_id": locus["locus_id"],
            "cohort_id": locus["cohort_id"],
            "comparison_group": locus.get("comparison_group"),
            "default_truth_samples": [],
            "exploratory_samples": [],
        }
        for sample in manifest.get("samples", []):
            if _locus_allowed(sample, locus) and sample.get("availability") == "available":
                destination = (
                    "default_truth_samples" if sample.get("truth_vote", True)
                    else "exploratory_samples"
                )
                row[destination].append(sample["sample_id"])
            elif locus.get("comparison_group") in sample.get(
                "exploratory_comparison_groups", []
            ):
                row["exploratory_samples"].append(sample["sample_id"])
        coverage.append(row)
    return {
        "schema_version": 1,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "bam_audits": bam_audits,
        "coverage_matrix": coverage,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="validate paths, tags, and locus coverage")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("-o", "--output", required=True)

    adjudicate = subparsers.add_parser(
        "adjudicate", help="construct hierarchical leave-one-assay-out truth"
    )
    adjudicate.add_argument("--manifest", required=True)
    adjudicate.add_argument("--evidence", required=True)
    adjudicate.add_argument(
        "--hierarchy",
        default=str(Path(__file__).with_name("default_hierarchy.json")),
    )
    adjudicate.add_argument("--no-leave-one-assay-out", action="store_true")
    adjudicate.add_argument("-o", "--output", required=True)

    evidence = subparsers.add_parser(
        "build-evidence",
        help="discover anchor candidates and summarize raw cross-assay evidence",
    )
    evidence.add_argument("--manifest", required=True)
    evidence.add_argument(
        "--hierarchy",
        default=str(Path(__file__).with_name("default_hierarchy.json")),
    )
    evidence.add_argument("--locus", action="append")
    evidence.add_argument("--exclude-exploratory", action="store_true")
    evidence.add_argument("--min-mapq", type=int, default=20)
    evidence.add_argument("--max-reads", type=int, default=0)
    evidence.add_argument(
        "--controls-per-candidate",
        type=int,
        default=0,
        help="Add this many opportunity-matched shifted controls per fine-TF candidate.",
    )
    evidence.add_argument(
        "--control-radius", type=int, default=2000,
        help="Maximum absolute shifted-control distance in bp (default: 2000).",
    )
    evidence.add_argument(
        "--control-step", type=int, default=25,
        help="Shifted-control search grid in bp (default: 25).",
    )
    evidence.add_argument(
        "--max-control-opportunity-ratio", type=float, default=4.0,
        help="Reject controls outside this source-opportunity fold difference.",
    )
    evidence.add_argument("-o", "--output", required=True)

    summary = subparsers.add_parser(
        "summarize", help="summarize fine-TF support, controls, and optional truth"
    )
    summary.add_argument("--manifest", required=True)
    summary.add_argument("--evidence", required=True)
    summary.add_argument("--truth")
    summary.add_argument("--support-log-bf", type=float, default=5.0)
    summary.add_argument("-o", "--output", required=True)

    calibrate = subparsers.add_parser(
        "calibrate",
        help="calibrate matched-control evidence on held-out loci and assign tiers",
    )
    calibrate.add_argument("--manifest", required=True)
    calibrate.add_argument(
        "--evidence", action="append", required=True,
        help="Evidence JSON; repeat for independently built locus batches.",
    )
    calibrate.add_argument(
        "--policy",
        default=str(Path(__file__).with_name("default_proposal_policy.json")),
    )
    calibrate.add_argument("--support-log-bf", type=float, default=5.0)
    calibrate.add_argument("--tsv", help="Optional deterministic sample-level TSV.")
    calibrate.add_argument("-o", "--output", required=True)

    sensitivity = subparsers.add_parser(
        "dedup-sensitivity",
        help="audit amplified-DAF molecule counts across Jaccard thresholds",
    )
    sensitivity.add_argument("--manifest", required=True)
    sensitivity.add_argument("--locus", required=True)
    sensitivity.add_argument("--sample", required=True)
    sensitivity.add_argument(
        "--jaccard", action="append", type=float,
        help="Repeatable threshold (defaults: 0.90, 0.95, 0.98).",
    )
    sensitivity.add_argument("--min-mapq", type=int, default=20)
    sensitivity.add_argument("--max-reads", type=int, default=0)
    sensitivity.add_argument("-o", "--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_json(args.manifest)
    if args.command == "audit":
        report = audit_manifest(manifest, args.manifest)
    elif args.command == "adjudicate":
        hierarchy = load_json(args.hierarchy)
        hierarchy_errors = validate_hierarchy(hierarchy)
        manifest_errors, _ = validate_manifest_structure(manifest)
        if hierarchy_errors or manifest_errors:
            raise SystemExit("\n".join(hierarchy_errors + manifest_errors))
        evidence = load_json(args.evidence)
        report = adjudicate_all(
            evidence,
            manifest,
            hierarchy,
            leave_one_assay_out=not args.no_leave_one_assay_out,
        )
    elif args.command == "build-evidence":
        hierarchy = load_json(args.hierarchy)
        hierarchy_errors = validate_hierarchy(hierarchy)
        manifest_errors, _ = validate_manifest_structure(manifest)
        if hierarchy_errors or manifest_errors:
            raise SystemExit("\n".join(hierarchy_errors + manifest_errors))
        if args.control_radius < 50 or args.control_step <= 0:
            raise SystemExit("control radius must be >=50 and control step must be positive")
        control_offsets = tuple(
            offset for offset in range(
                -args.control_radius, args.control_radius + 1, args.control_step
            )
            if abs(offset) >= 50
        )
        report = build_evidence(
            manifest,
            hierarchy,
            locus_ids=args.locus,
            include_exploratory=not args.exclude_exploratory,
            min_mapq=args.min_mapq,
            max_reads=args.max_reads,
            controls_per_candidate=args.controls_per_candidate,
            control_offsets=control_offsets,
            max_control_opportunity_ratio=args.max_control_opportunity_ratio,
            manifest_base=str(Path(args.manifest).resolve().parent),
        )
    elif args.command == "summarize":
        evidence = load_json(args.evidence)
        truth = load_json(args.truth) if args.truth else None
        report = summarize_fine_tf(
            evidence,
            manifest,
            truth,
            support_log_bf=args.support_log_bf,
        )
    elif args.command == "calibrate":
        policy = load_json(args.policy)
        summaries = [
            summarize_fine_tf(
                load_json(path), manifest, support_log_bf=args.support_log_bf
            )
            for path in args.evidence
        ]
        report = calibrate_fine_tf(
            merge_summaries(summaries), manifest, policy
        )
        if args.tsv:
            write_proposal_tsv(report, args.tsv)
    else:
        report = audit_molecule_collapse(
            manifest,
            locus_id=args.locus,
            sample_id=args.sample,
            jaccards=(args.jaccard or (0.90, 0.95, 0.98)),
            min_mapq=args.min_mapq,
            max_reads=args.max_reads,
            manifest_base=str(Path(args.manifest).resolve().parent),
        )
    output = Path(args.output)
    _write_json_atomic(output, report)
    print(json.dumps({
        "output": str(output),
        "valid": report.get("valid"),
        "errors": len(report.get("errors", [])),
        "candidates": len(report.get("candidates", [])),
        "controls": len(report.get("controls", [])),
        "records": (
            report.get("n_candidate_folds")
            if report.get("n_candidate_folds") is not None
            else len(report.get("records", []))
        ),
    }))
    return int(bool(report.get("errors")))


if __name__ == "__main__":
    raise SystemExit(main())
