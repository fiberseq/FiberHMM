#!/usr/bin/env python3
"""Audit FiberHMM v3 paired-consensus MA/AQ/AN invariants."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple

import pysam

from consensus_recaller_collab.annotate import (
    LAYER_ORDER,
    PAIRED_CONSENSUS_HEADER_PREFIX,
    PAIRED_QUALITY_SPEC,
)
from fiberhmm.io.bam_header import declared_ma_types
from fiberhmm.io.ma_tags import parse_an_tag, parse_aq_array, parse_ma_tag


PAIRED_NAME_RE = re.compile(
    r"^(fh(?P<pass>cr|sr)_[0-9a-f]{16})_"
    r"(?P<role>N|(?P<access>A)?T(?P<index>[0-9]+))$"
)
THRESHOLDS = (0, 64, 128, 192, 255)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _header_contract(header) -> Optional[str]:
    return next(
        (
            str(comment)
            for comment in header.to_dict().get("CO", [])
            if str(comment).startswith(PAIRED_CONSENSUS_HEADER_PREFIX)
        ),
        None,
    )


def _annotation_rows(read) -> List[dict]:
    if not read.has_tag("MA"):
        return []
    parsed = parse_ma_tag(read.get_tag("MA"))
    raw_types = parsed["raw_types"]
    aq = read.get_tag("AQ") if read.has_tag("AQ") else []
    expected_aq = sum(
        len(quality_spec) * len(intervals)
        for _name, _strand, quality_spec, intervals in raw_types
    )
    if len(aq) != expected_aq:
        raise ValueError(
            f"AQ has {len(aq)} bytes but MA requires {expected_aq}"
        )
    quality_rows = parse_aq_array(
        aq,
        [item[2] for item in raw_types],
        [len(item[3]) for item in raw_types],
    )
    annotation_count = sum(len(item[3]) for item in raw_types)
    if read.has_tag("AN"):
        names = parse_an_tag(read.get_tag("AN"))
        if len(names) != annotation_count:
            raise ValueError(
                f"AN has {len(names)} fields but MA has {annotation_count} annotations"
            )
    else:
        names = [""] * annotation_count

    rows = []
    cursor = 0
    for name, strand, quality_spec, intervals in raw_types:
        for interval in intervals:
            rows.append({
                "type": name,
                "strand": strand,
                "quality_spec": quality_spec,
                "interval": tuple(int(value) for value in interval),
                "qualities": tuple(int(value) for value in quality_rows[cursor]),
                "annotation_name": names[cursor],
            })
            cursor += 1
    return rows


def audit_bam(path: str, *, max_errors: int = 100) -> dict:
    """Return a complete paired-invariant audit for one BAM."""
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise ValueError(f"missing BAM: {candidate}")
    if max_errors < 1:
        raise ValueError("max_errors must be positive")

    errors: List[dict] = []
    error_count = 0

    def record_error(read_name: str, decision: str, message: str) -> None:
        nonlocal error_count
        error_count += 1
        if len(errors) < max_errors:
            errors.append({
                "read": read_name,
                "decision": decision,
                "message": message,
            })

    counts: Counter = Counter()
    by_pass: Dict[str, Counter] = {"cr": Counter(), "sr": Counter()}
    tf_component_histogram: Counter = Counter()
    q0_tf_histogram: Counter = Counter()
    threshold_states = {
        str(threshold): {"TF": 0, "N": 0, "A": 0}
        for threshold in THRESHOLDS
    }

    try:
        pysam.quickcheck(str(candidate))
        quickcheck = True
    except pysam.utils.SamtoolsError:
        quickcheck = False
        record_error("", "", "samtools quickcheck failed")

    index_opened = False
    with pysam.AlignmentFile(candidate, "rb", check_sq=False) as bam:
        try:
            index_opened = bool(bam.has_index() and bam.check_index())
        except (AttributeError, OSError, ValueError):
            index_opened = False
        if not index_opened:
            record_error("", "", "BAM index is missing or cannot be opened")

        contract = _header_contract(bam.header)
        if contract is None:
            record_error("", "", "missing FIBERHMM-CONSENSUS:v3 header")
        else:
            for token in (
                "groups=" + ",".join(LAYER_ORDER),
                "semantics=paired_hypotheses",
                f"quality_spec={PAIRED_QUALITY_SPEC}",
                "q0=represented_state_posterior",
                "pair_sum=255",
                "pairing=AN_shared_prefix",
                "accessible_tf_role=ATn",
                "baseline_q0=255",
            ):
                if token not in contract:
                    record_error("", "", f"header contract lacks {token}")
        advertised = set(declared_ma_types(bam.header))
        missing_types = sorted(set(LAYER_ORDER) - advertised)
        if missing_types:
            record_error(
                "",
                "",
                "MA-TYPES header lacks " + ",".join(missing_types),
            )

        for read in bam.fetch(until_eof=True):
            counts["records"] += 1
            try:
                annotations = _annotation_rows(read)
            except (KeyError, TypeError, ValueError) as error:
                record_error(read.query_name, "", str(error))
                continue

            decisions: DefaultDict[str, List[dict]] = defaultdict(list)
            for annotation in annotations:
                match = PAIRED_NAME_RE.fullmatch(annotation["annotation_name"])
                if match is not None and annotation["type"] not in LAYER_ORDER:
                    record_error(
                        read.query_name,
                        match.group(1),
                        f"paired AN label occurs on non-consensus layer {annotation['type']}",
                    )
                    continue
                if annotation["type"] not in LAYER_ORDER:
                    continue

                counts["consensus_annotations"] += 1
                counts[f"annotations_{annotation['type']}"] += 1
                if annotation["quality_spec"] != PAIRED_QUALITY_SPEC:
                    record_error(
                        read.query_name,
                        match.group(1) if match else "",
                        f"{annotation['type']} uses {annotation['quality_spec']!r}, not QQQQQ",
                    )
                    continue
                if len(annotation["qualities"]) != 5:
                    record_error(
                        read.query_name,
                        match.group(1) if match else "",
                        f"{annotation['type']} does not have five AQ bytes",
                    )
                    continue
                if match is None:
                    counts["fixed_baseline_annotations"] += 1
                    if annotation["qualities"] != (255, 0, 0, 0, 0):
                        record_error(
                            read.query_name,
                            "",
                            "unpaired consensus annotation is not fixed baseline (255,0,0,0,0)",
                        )
                    continue

                annotation["decision"] = match.group(1)
                annotation["pass"] = match.group("pass")
                annotation["role"] = match.group("role")
                annotation["tf_index"] = (
                    None if match.group("index") is None
                    else int(match.group("index"))
                )
                annotation["accessible_origin"] = bool(match.group("access"))
                decisions[match.group(1)].append(annotation)

            if decisions:
                counts["records_with_paired_decisions"] += 1
            for decision_id, members in decisions.items():
                counts["decisions"] += 1
                pass_names = {member["pass"] for member in members}
                if len(pass_names) != 1:
                    record_error(
                        read.query_name, decision_id, "members disagree on pass"
                    )
                    continue
                pass_name = next(iter(pass_names))
                roles = [member["role"] for member in members]
                if len(set(roles)) != len(roles):
                    record_error(
                        read.query_name, decision_id, "duplicate member role"
                    )
                    continue
                n_members = [member for member in members if member["role"] == "N"]
                tf_members = [member for member in members if member["role"] != "N"]
                if len(n_members) > 1:
                    record_error(
                        read.query_name, decision_id, "more than one N member"
                    )
                if not tf_members:
                    record_error(
                        read.query_name, decision_id, "decision has no TF member"
                    )
                    continue

                expected_n_layer = f"nuc_{pass_name}"
                expected_tf_layer = f"tf_{pass_name}"
                for member in n_members:
                    if member["type"] != expected_n_layer:
                        record_error(
                            read.query_name,
                            decision_id,
                            f"N member is on {member['type']}, expected {expected_n_layer}",
                        )
                for member in tf_members:
                    if member["type"] != expected_tf_layer:
                        record_error(
                            read.query_name,
                            decision_id,
                            f"TF member is on {member['type']}, expected {expected_tf_layer}",
                        )

                tf_rows = {member["qualities"] for member in tf_members}
                if len(tf_rows) != 1:
                    record_error(
                        read.query_name,
                        decision_id,
                        "multi-TF components do not share one quality row",
                    )
                    continue
                tf_row = next(iter(tf_rows))
                q_tf = tf_row[0]
                q0_tf_histogram[str(q_tf)] += 1
                tf_component_histogram[str(len(tf_members))] += 1
                by_pass[pass_name]["decisions"] += 1
                by_pass[pass_name]["tf_components"] += len(tf_members)
                access_roles = {
                    bool(member["accessible_origin"]) for member in tf_members
                }
                if len(access_roles) != 1:
                    record_error(
                        read.query_name,
                        decision_id,
                        "decision mixes Tn and ATn member roles",
                    )
                    continue
                accessible_origin = next(iter(access_roles))

                if n_members:
                    if accessible_origin:
                        record_error(
                            read.query_name,
                            decision_id,
                            "ATn accessible-origin roles cannot have an N member",
                        )
                    n_row = n_members[0]["qualities"]
                    if n_row[1:] != tf_row[1:]:
                        record_error(
                            read.query_name,
                            decision_id,
                            "N and TF auxiliary qualities differ",
                        )
                    if n_row[0] + q_tf != 255:
                        record_error(
                            read.query_name,
                            decision_id,
                            f"q0 values sum to {n_row[0] + q_tf}, not 255",
                        )
                    counts["paired_n_vs_tf_decisions"] += 1
                    by_pass[pass_name]["paired_n_vs_tf"] += 1
                    current = "N"
                else:
                    counts["one_sided_access_vs_tf_decisions"] += 1
                    by_pass[pass_name]["one_sided_access_vs_tf"] += 1
                    current = "A"
                    if pass_name != "sr" or not accessible_origin:
                        record_error(
                            read.query_name,
                            decision_id,
                            "one-sided decision is not an SR ATn accessible-origin decision",
                        )

                for threshold in THRESHOLDS:
                    selected = "TF" if q_tf >= threshold else current
                    threshold_states[str(threshold)][selected] += 1

    index_paths = [
        Path(str(candidate) + suffix) for suffix in (".bai", ".csi")
        if Path(str(candidate) + suffix).is_file()
    ]
    return {
        "path": str(candidate),
        "sha256": _sha256(candidate),
        "size_bytes": candidate.stat().st_size,
        "index_paths": [str(path) for path in index_paths],
        "indexed": index_opened,
        "quickcheck": quickcheck,
        "valid": error_count == 0,
        "error_count": error_count,
        "errors": errors,
        "counts": dict(sorted(counts.items())),
        "by_pass": {
            key: dict(sorted(value.items())) for key, value in by_pass.items()
        },
        "tf_component_histogram": dict(sorted(tf_component_histogram.items())),
        "q0_tf_histogram": dict(
            sorted(q0_tf_histogram.items(), key=lambda item: int(item[0]))
        ),
        "threshold_states": threshold_states,
    }


def audit_bams(paths: Sequence[str], *, max_errors: int = 100) -> dict:
    files = [audit_bam(path, max_errors=max_errors) for path in paths]
    totals: Counter = Counter()
    for result in files:
        totals.update(result["counts"])
    return {
        "schema_version": 1,
        "producer": "fiberhmm-consensus-audit-pairs",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "quality_spec": PAIRED_QUALITY_SPEC,
        "thresholds": list(THRESHOLDS),
        "valid": all(result["valid"] for result in files),
        "file_count": len(files),
        "totals": dict(sorted(totals.items())),
        "files": files,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--bam", action="append", required=True)
    parser.add_argument("-o", "--output", help="Persistent JSON audit path")
    parser.add_argument("--max-errors", type=int, default=100)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = audit_bams(args.bam, max_errors=args.max_errors)
        if args.output:
            _atomic_json(Path(args.output).expanduser(), result)
    except (OSError, TypeError, ValueError, pysam.utils.SamtoolsError) as error:
        parser.error(str(error))
    print(json.dumps({
        "valid": result["valid"],
        "file_count": result["file_count"],
        "totals": result["totals"],
        "output": str(Path(args.output).expanduser().resolve()) if args.output else None,
    }, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
