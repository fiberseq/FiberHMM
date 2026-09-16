"""BED-driven orchestration for targeted site-consensus scans."""

from __future__ import annotations

import csv
import contextlib
from collections import defaultdict
import hashlib
import json
import multiprocessing
import os
import shlex
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pysam

from fiberhmm import __version__
from fiberhmm.inference.targeted_family_batch import (
    family_target_memberships,
    load_bed_targets,
    plan_targeted_family_work_units,
)
from fiberhmm.inference.targeted_families import CHEMISTRY_PROFILES
from fiberhmm.inference.composite_footprint_states import (
    nominate_composite_footprint_states,
)
from fiberhmm.inference.mp_context import _MP_CONTEXT
from fiberhmm.inference.tf_family_ids import (
    TFFamilyInterval,
    allocate_repeating_family_ids,
)


def add_batch_parser(subparsers) -> None:
    batch = subparsers.add_parser(
        "batch",
        help="Discover and quantify site-consensus states across BED-defined targets",
    )
    batch.add_argument("--bed", required=True)
    batch.add_argument("-i", "--input", action="append", required=True)
    batch.add_argument("-o", "--output-dir", required=True)
    batch.add_argument("--chemistry", choices=("ddda", "dddb", "hia5-pacbio", "hia5-nanopore"))
    batch.add_argument("-c", "--cores", type=int, default=1, help="Workers within each work unit; 0=all CPUs")
    batch.add_argument(
        "--unit-workers",
        type=int,
        default=0,
        help="Concurrent BED work units; 0 chooses a CPU-aware value",
    )
    batch.add_argument("--site-padding", type=int, default=500)
    batch.add_argument("--merge-gap", type=int, default=500)
    batch.add_argument("--max-work-unit-bp", type=int, default=25000)
    batch.add_argument("--resume", action="store_true")
    batch.add_argument(
        "--no-aggregate-molecule-assignments",
        action="store_true",
        help="Keep molecule assignments in unit directories without concatenating them",
    )

    batch.add_argument("--window-size", type=int, default=1000)
    batch.add_argument("--halo-size", type=int, default=200)
    batch.add_argument("--minimum-nfr-length", type=int, default=150)
    batch.add_argument("--minimum-informative-molecules", type=int, default=3)
    batch.add_argument("--minimum-informative-fraction", type=float, default=0.01)
    batch.add_argument("--discovery-reads", type=int, default=500)
    batch.add_argument(
        "--minimum-state-support",
        dest="minimum_family_support",
        type=int,
        default=3,
    )
    batch.add_argument(
        "--minimum-state-fraction",
        dest="minimum_family_fraction",
        type=float,
        default=0.05,
    )
    batch.add_argument("--min-mapq", type=int, default=20)
    batch.add_argument("--seed", default="fiberhmm-site-consensus-discovery-v1")
    batch.add_argument("--daf-minimum-jaccard", type=float, default=0.95)
    batch.add_argument("--daf-minimum-deaminations", type=int, default=10)
    batch.add_argument("--daf-already-deduplicated", action="store_true")
    batch.add_argument(
        "--skip-input-hash",
        action="store_true",
        help="Use cached BAM/header/index fast provenance instead of full BAM SHA-256",
    )

    batch.add_argument("--chunk-size", type=int, default=512)
    batch.add_argument(
        "--likelihood-backend", choices=("auto", "cpu", "cuda"), default="cpu"
    )
    batch.add_argument("--cuda-interval-chunk-size", type=int, default=1024)
    batch.add_argument("--cuda-read-chunk-size", type=int, default=0)
    batch.add_argument(
        "--cuda-state-batch-span-bp",
        dest="cuda_family_batch_span_bp",
        type=int,
        default=25000,
    )
    batch.add_argument("--cuda-replay-guard-nats", type=float, default=1e-8)
    batch.add_argument("--tf-layer", choices=("tf", "tf_sr"), default="tf_sr")
    batch.add_argument("--nuc-layer", choices=("nuc", "nuc_sr"), default="nuc_sr")
    batch.add_argument("--occupancy-pseudocount", type=float, default=0.5)
    batch.add_argument(
        "--minimum-assignment-standardized-posterior",
        dest="minimum_assignment_standardized_posterior",
        type=float,
        default=0.5,
    )
    batch.add_argument("--minimum-assignment-log-bayes-factor", type=float, default=0.0)
    batch.add_argument("--include-conditional-geometry", action="store_true")


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_table(path: Path, rows, fields) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _next_stage_directory(unit_directory: Path, stem: str) -> Path:
    first = unit_directory / stem
    if not first.exists():
        return first
    index = 1
    while True:
        candidate = unit_directory / f"{stem}.retry_{index:03d}"
        if not candidate.exists():
            return candidate
        index += 1


def _targeted_catalog(
    catalog_path: Path,
    output_path: Path,
    membership_path: Path,
    *,
    unit,
    targets,
    reference_lengths,
    padding: int,
):
    catalog = json.loads(catalog_path.read_text())
    unfiltered_family_count = len(catalog["families"])
    retained = []
    memberships = []
    for family in catalog["families"]:
        matched = family_target_memberships(
            family,
            unit,
            targets,
            reference_lengths,
            padding=padding,
        )
        if not matched:
            continue
        direct = family_target_memberships(
            family,
            unit,
            targets,
            reference_lengths,
            padding=0,
        )
        retained.append(family)
        memberships.append(
            {
                "family_id": family["family_id"],
                "target_ids": ",".join(target.target_id for target in matched),
                "target_names": ",".join(target.name for target in matched),
                "direct_target_ids": ",".join(
                    target.target_id for target in direct
                ),
                "direct_target_names": ",".join(target.name for target in direct),
            }
        )
    catalog["families"] = retained
    catalog["family_slots"] = {
        family["family_id"]: catalog["family_slots"][family["family_id"]]
        for family in retained
    }
    catalog.setdefault("contracts", {})["bed_target_filter"] = {
        "criterion": "family_center_inside_padded_target",
        "padding_bp": int(padding),
        "work_unit_id": unit.unit_id,
        "unfiltered_family_count": unfiltered_family_count,
        "retained_family_count": len(retained),
    }
    _write_json(output_path, catalog)
    _write_table(
        membership_path,
        memberships,
        (
            "family_id",
            "target_ids",
            "target_names",
            "direct_target_ids",
            "direct_target_names",
        ),
    )
    return catalog, {row["family_id"]: row for row in memberships}


def _stage_args(parent_parser, values, input_provenance):
    parsed = parent_parser.parse_args([str(value) for value in values])
    parsed._batch_input_provenance = input_provenance
    return parsed


def _stage_command_line(values) -> str:
    return " ".join(
        ["fiberhmm-site-consensus"]
        + [shlex.quote(str(value)) for value in values]
    )


def _run_discovery_stage(payload):
    values, input_provenance, log_path = payload
    from fiberhmm.cli import targeted_families as single

    parsed = _stage_args(single.build_parser(), values, input_provenance)
    try:
        with Path(log_path).open("a") as log_handle:
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(
                log_handle
            ):
                single._discover(parsed, _stage_command_line(values))
    except ValueError as error:
        if str(error) == "no eligible independent molecules":
            return {"status": "empty", "reason": str(error)}
        raise
    return {"status": "complete", "output_dir": str(parsed.output_dir)}


def _run_quantification_stage(payload):
    values, input_provenance, log_path = payload
    from fiberhmm.cli import targeted_families as single

    parsed = _stage_args(single.build_parser(), values, input_provenance)
    try:
        with Path(log_path).open("a") as log_handle:
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(
                log_handle
            ):
                single._quantify(parsed, _stage_command_line(values))
    except ValueError as error:
        message = str(error)
        if (
            message == "no independent full-cohort molecules"
            or message == "no catalog family has sufficient sequence opportunities to fit"
            or (message.startswith("family ") and message.endswith(
                " has no raw MSP-enriched evidence"
            ))
        ):
            return {"status": "empty", "reason": message}
        raise
    return {"status": "complete", "output_dir": str(parsed.output_dir)}


def _report_stage_progress(stage_name: str, completed: int, total: int, result) -> None:
    stride = max(1, (total + 19) // 20)
    if completed == 1 or completed == total or completed % stride == 0:
        print(
            f"BED {stage_name}: completed={completed}/{total} "
            f"status={result['status']}",
            file=sys.stderr,
            flush=True,
        )


def _execute_stage(
    payloads,
    worker,
    workers: int,
    stage_name: str,
    *,
    isolate_each: bool = False,
):
    if isolate_each:
        results = []
        spawn_context = multiprocessing.get_context("spawn")
        for completed, payload in enumerate(payloads, start=1):
            # A fresh process per device unit ensures that no process which has
            # initialized CUDA subsequently forks the evidence/fit CPU pools.
            with ProcessPoolExecutor(
                max_workers=1,
                mp_context=spawn_context,
            ) as executor:
                result = executor.submit(worker, payload).result()
            results.append(result)
            _report_stage_progress(stage_name, completed, len(payloads), result)
        return results
    if workers == 1 or len(payloads) <= 1:
        iterator = map(worker, payloads)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=min(workers, len(payloads)),
            mp_context=_MP_CONTEXT,
        )
        iterator = executor.map(worker, payloads)
    results = []
    try:
        for completed, result in enumerate(iterator, start=1):
            results.append(result)
            _report_stage_progress(stage_name, completed, len(payloads), result)
    finally:
        if executor is not None:
            executor.shutdown()
    return results


def _complete_quantification_matches(
    directory: Path,
    catalog_path: Path,
    *,
    expected_efficiency_exclusion_sha256: str | None = None,
) -> bool:
    """Validate the minimum restart contract for a completed unit fit."""

    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file() or not catalog_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return False
    if manifest.get("status") != "complete_unbiased_independent_family_screen":
        return False
    if manifest.get("catalog_sha256") != hashlib.sha256(
        catalog_path.read_bytes()
    ).hexdigest():
        return False
    if expected_efficiency_exclusion_sha256 is not None:
        fit_config = manifest.get("model_fit_config", {})
        if (
            fit_config.get("efficiency_exclusion_bed_sha256")
            != expected_efficiency_exclusion_sha256
            or fit_config.get("efficiency_exclusion_padding") != 0
            or fit_config.get("efficiency_evidence_scope") != "full-alignment"
        ):
            return False
    artifacts = manifest.get("artifacts", {})
    required = (
        artifacts.get("models"),
        artifacts.get("family_scores"),
        artifacts.get("molecule_family_scores"),
    )
    for relative in required:
        if not relative or not (directory / str(relative)).exists():
            return False
    return True


def _discover_values(args, unit, output_directory: Path):
    values = ["discover"]
    for path in args.input:
        values.extend(("-i", path))
    values.extend(
        (
            "--region", unit.region,
            "-o", str(output_directory),
            "-c", str(args.cores),
            "--window-size", str(args.window_size),
            "--halo-size", str(args.halo_size),
            "--minimum-nfr-length", str(args.minimum_nfr_length),
            "--minimum-informative-molecules", str(args.minimum_informative_molecules),
            "--minimum-informative-fraction", str(args.minimum_informative_fraction),
            "--discovery-reads", str(args.discovery_reads),
            "--minimum-state-support", str(args.minimum_family_support),
            "--minimum-state-fraction", str(args.minimum_family_fraction),
            "--min-mapq", str(args.min_mapq),
            "--seed", str(args.seed),
            "--daf-minimum-jaccard", str(args.daf_minimum_jaccard),
            "--daf-minimum-deaminations", str(args.daf_minimum_deaminations),
        )
    )
    if args.chemistry:
        values.extend(("--chemistry", args.chemistry))
    if args.skip_input_hash:
        values.append("--skip-input-hash")
    if args.daf_already_deduplicated:
        values.append("--daf-already-deduplicated")
    return values


def _quantify_values(
    args,
    catalog_path: Path,
    output_directory: Path,
    *,
    efficiency_exclusion_bed: Path,
):
    values = ["quantify", "--catalog", str(catalog_path)]
    for path in args.input:
        values.extend(("-i", path))
    values.extend(
        (
            "-o", str(output_directory),
            "-c", str(args.cores),
            "--chunk-size", str(args.chunk_size),
            "--likelihood-backend", args.likelihood_backend,
            "--cuda-interval-chunk-size", str(args.cuda_interval_chunk_size),
            "--cuda-read-chunk-size", str(args.cuda_read_chunk_size),
            "--cuda-state-batch-span-bp", str(args.cuda_family_batch_span_bp),
            "--cuda-replay-guard-nats", str(args.cuda_replay_guard_nats),
            "--min-mapq", str(args.min_mapq),
            "--tf-layer", args.tf_layer,
            "--nuc-layer", args.nuc_layer,
            "--occupancy-pseudocount", str(args.occupancy_pseudocount),
            "--minimum-assignment-standardized-posterior",
            str(args.minimum_assignment_standardized_posterior),
            "--minimum-assignment-log-bayes-factor",
            str(args.minimum_assignment_log_bayes_factor),
            "--efficiency-exclusion-bed", str(efficiency_exclusion_bed),
            "--efficiency-exclusion-padding", "0",
            "--efficiency-evidence-scope", "full-alignment",
        )
    )
    if args.skip_input_hash:
        values.append("--skip-input-hash-check")
    values.append("--compact-artifacts")
    if args.include_conditional_geometry:
        values.append("--include-conditional-geometry")
    return values


def _write_global_efficiency_exclusions(
    path: Path,
    *,
    targets,
    reference_lengths,
    site_padding: int,
    catalogs,
    chemistry: str,
    candidate_interval_builder,
) -> dict:
    """Freeze one batch-wide calibration mask after all discovery completes."""

    intervals_by_contig = {}
    for target in targets:
        intervals_by_contig.setdefault(target.contig, []).append(
            target.expanded(site_padding, int(reference_lengths[target.contig]))
        )
    candidate_intervals = 0
    for catalog in catalogs:
        families = list(catalog.get("families", ()))
        for start, end in candidate_interval_builder(families, chemistry):
            intervals_by_contig.setdefault(str(catalog["contig"]), []).append(
                (int(start), int(end))
            )
            candidate_intervals += 1
    rows = []
    for contig in sorted(intervals_by_contig):
        merged = []
        for start, end in sorted(intervals_by_contig[contig]):
            start = max(0, int(start))
            end = min(int(reference_lengths[contig]), int(end))
            if end <= start:
                continue
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        rows.extend((contig, start, end) for start, end in merged)
    with path.open("w") as handle:
        for contig, start, end in rows:
            handle.write(f"{contig}\t{start}\t{end}\n")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "target_intervals": len(targets),
        "family_candidate_union_intervals_before_global_merge": candidate_intervals,
        "merged_intervals": len(rows),
        "policy": "padded_targets_plus_all_retained_family_candidate_envelopes",
    }


def _resolve_backend_probe(requested: str) -> str:
    from fiberhmm.inference.cuda_likelihood import resolve_likelihood_backend

    return str(resolve_likelihood_backend(requested)[0])


def _resolve_batch_backend_without_parent_cuda(requested: str) -> str:
    if requested != "auto":
        return requested
    # The short-lived spawned probe may initialize CUDA; the orchestrating
    # parent remains CUDA-clean and can safely fork CPU discovery workers.
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        return executor.submit(_resolve_backend_probe, requested).result()


def _aggregate_outputs(
    output_directory: Path,
    unit_states,
    *,
    aggregate_molecules: bool,
    chemistry: str,
):
    aggregate_assignment_path = output_directory / "molecule_family_scores.tsv"
    if not aggregate_molecules and aggregate_assignment_path.is_file():
        aggregate_assignment_path.unlink()
    catalog_by_unit = {}
    global_intervals = []
    for state in unit_states:
        if int(state.get("retained_family_count", 0)) == 0:
            continue
        catalog = json.loads(Path(state["targeted_catalog"]).read_text())
        catalog_by_unit[state["unit_id"]] = catalog
        for family in catalog["families"]:
            batch_family_id = f"{state['unit_id']}:{family['family_id']}"
            global_intervals.append(
                TFFamilyInterval(
                    family_key=batch_family_id,
                    contig=str(family["contig"]),
                    start=int(family["start"]),
                    end=int(family["end"]),
                )
            )
    global_slots = allocate_repeating_family_ids(
        global_intervals,
        separation_bp=2 * CHEMISTRY_PROFILES[chemistry].maximum_boundary_delta,
    )
    family_fields = (
        "batch_family_id", "work_unit_id", "target_ids", "target_names",
        "direct_target_ids", "direct_target_names",
        "family_id", "family_slot", "local_family_slot", "contig", "start", "end", "width",
        "discovery_support_molecules", "discovery_denominator_molecules",
        "score_status",
    )
    score_fields = (
        "batch_family_id", "work_unit_id", "target_ids", "target_names",
        "direct_target_ids", "direct_target_names",
        "family_id", "family_slot", "local_family_slot", "eligible_molecules",
        "fitted_family_occupancy", "fitted_family_effective_support",
        "standardized_family_effective_support_equal_prior",
        "median_family_vs_null_log_bayes_factor", "assigned_molecules", "score_status",
    )
    assignment_fields = (
        "batch_family_id", "work_unit_id", "target_ids", "target_names",
        "direct_target_ids", "direct_target_names",
        "family_id", "family_slot", "local_family_slot", "score_status",
        "library_id", "read_name", "strand",
        "fitted_family_posterior", "standardized_family_posterior_equal_prior",
        "family_vs_null_log_bayes_factor", "map_start", "map_end",
        "map_interval_probability",
    )
    counts = {
        "families": 0,
        "scored_families": 0,
        "unscorable_families": 0,
        "assignments": 0,
        "units_with_families": 0,
    }
    with (output_directory / "families.tsv").open("w", newline="") as family_handle, (
        output_directory / "family_scores.tsv"
    ).open("w", newline="") as score_handle:
        family_writer = csv.DictWriter(
            family_handle, fieldnames=family_fields, delimiter="\t", lineterminator="\n"
        )
        score_writer = csv.DictWriter(
            score_handle, fieldnames=score_fields, delimiter="\t", lineterminator="\n"
        )
        family_writer.writeheader()
        score_writer.writeheader()
        assignment_handle = None
        assignment_writer = None
        if aggregate_molecules:
            assignment_handle = aggregate_assignment_path.open("w", newline="")
            assignment_writer = csv.DictWriter(
                assignment_handle,
                fieldnames=assignment_fields,
                delimiter="\t",
                lineterminator="\n",
            )
            assignment_writer.writeheader()
        try:
            for state in unit_states:
                if int(state.get("retained_family_count", 0)) == 0:
                    continue
                counts["units_with_families"] += 1
                catalog = catalog_by_unit[state["unit_id"]]
                with Path(state["family_memberships"]).open(
                    newline=""
                ) as membership_source:
                    membership = {
                        row["family_id"]: row
                        for row in csv.DictReader(
                            membership_source,
                            delimiter="\t",
                        )
                    }
                quantification = (
                    Path(state["quantification_directory"])
                    if state.get("quantification_directory")
                    else None
                )
                scores = {}
                if quantification is not None:
                    with (quantification / "family_scores.tsv").open(
                        newline=""
                    ) as score_source:
                        scores = {
                            row["family_id"]: row
                            for row in csv.DictReader(
                                score_source,
                                delimiter="\t",
                            )
                        }
                prefix_by_family = {}
                for family in catalog["families"]:
                    family_id = family["family_id"]
                    score = scores.get(family_id)
                    score_status = "scored" if score is not None else "unscorable"
                    batch_family_id = f"{state['unit_id']}:{family_id}"
                    prefix = {
                        "batch_family_id": batch_family_id,
                        "work_unit_id": state["unit_id"],
                        "family_slot": global_slots[batch_family_id],
                        "local_family_slot": catalog["family_slots"][family_id],
                        "score_status": score_status,
                        **membership[family_id],
                    }
                    prefix_by_family[family_id] = prefix
                    family_writer.writerow(
                        {
                            **prefix,
                            "family_id": family_id,
                            "contig": family["contig"],
                            "start": family["start"],
                            "end": family["end"],
                            "width": int(family["end"]) - int(family["start"]),
                            "discovery_support_molecules": family[
                                "discovery_support_molecules"
                            ],
                            "discovery_denominator_molecules": family[
                                "discovery_denominator_molecules"
                            ],
                        }
                    )
                    score_writer.writerow(
                        {
                            **(
                                score
                                if score is not None
                                else {
                                    "family_id": family_id,
                                    "eligible_molecules": "",
                                    "fitted_family_occupancy": "",
                                    "fitted_family_effective_support": "",
                                    "standardized_family_effective_support_equal_prior": "",
                                    "median_family_vs_null_log_bayes_factor": "",
                                    "assigned_molecules": 0,
                                }
                            ),
                            **prefix,
                        }
                    )
                    counts["families"] += 1
                    counts[f"{score_status}_families"] += 1
                if assignment_writer is not None and quantification is not None:
                    with (quantification / "molecule_family_scores.tsv").open(
                        newline=""
                    ) as source:
                        for row in csv.DictReader(source, delimiter="\t"):
                            assignment_writer.writerow(
                                {**row, **prefix_by_family[row["family_id"]]}
                            )
                            counts["assignments"] += 1
        finally:
            if assignment_handle is not None:
                assignment_handle.close()
    return counts


def _write_composite_state_outputs(output_directory: Path) -> dict:
    """Nominate broad-family/component relationships without changing calls."""

    family_path = output_directory / "families.tsv"
    score_path = output_directory / "family_scores.tsv"
    assignment_path = output_directory / "molecule_family_scores.tsv"
    with score_path.open(newline="") as handle:
        scores = {
            row["batch_family_id"]: row
            for row in csv.DictReader(handle, delimiter="\t")
        }
    families = []
    with family_path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            score = scores.get(row["batch_family_id"], {})
            families.append(
                {
                    **row,
                    "family_slot": int(row["family_slot"]),
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "working_support_molecules": int(
                        score.get("assigned_molecules") or 0
                    ),
                }
            )
    family_molecules: dict[str, set[str]] = defaultdict(set)
    family_molecules_by_dataset: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    if assignment_path.is_file():
        with assignment_path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                family_id = row["batch_family_id"]
                dataset_id = row["library_id"]
                molecule_id = "\x1f".join(
                    (dataset_id, row["read_name"], row["strand"])
                )
                family_molecules[family_id].add(molecule_id)
                family_molecules_by_dataset[family_id][dataset_id].add(
                    molecule_id
                )
    for family in families:
        family_id = family["batch_family_id"]
        family["working_support_by_dataset"] = {
            dataset_id: len(molecules)
            for dataset_id, molecules in sorted(
                family_molecules_by_dataset.get(family_id, {}).items()
            )
        }
    analysis = nominate_composite_footprint_states(
        families,
        family_molecules=family_molecules,
        family_molecules_by_dataset=family_molecules_by_dataset,
    )
    _write_json(output_directory / "composite_states.json", analysis)
    rows = []
    for nomination in analysis["nominations"]:
        envelope = nomination["envelope_family"]
        components = nomination["component_families"]
        support = nomination["support"]
        geometry = nomination["geometry"]
        rows.append(
            {
                "nomination_id": nomination["nomination_id"],
                "contig": envelope["contig"],
                "start": envelope["start"],
                "end": envelope["end"],
                "envelope_family_id": envelope["family_id"],
                "envelope_family_slot": envelope["family_slot"],
                "component_family_ids": ",".join(
                    family["family_id"] for family in components
                ),
                "component_family_slots": ",".join(
                    str(family["family_slot"]) for family in components
                ),
                "component_count": nomination["component_count"],
                "union_coverage_fraction": geometry[
                    "union_coverage_fraction"
                ],
                "internal_gap_bp": geometry["internal_gap_bp"],
                "geometry_score": geometry["geometry_score"],
                "envelope_molecules": support["envelope_molecules"],
                "component_molecules": ",".join(
                    str(value) for value in support["component_molecules"]
                ),
                "datasets_with_all_states": ",".join(
                    support["datasets_with_all_states"]
                ),
                "evidence_level": nomination["evidence_level"],
                "nomination_score": nomination["nomination_score"],
                "claim_boundary": nomination["claim_boundary"],
            }
        )
    fields = (
        "nomination_id",
        "contig",
        "start",
        "end",
        "envelope_family_id",
        "envelope_family_slot",
        "component_family_ids",
        "component_family_slots",
        "component_count",
        "union_coverage_fraction",
        "internal_gap_bp",
        "geometry_score",
        "envelope_molecules",
        "component_molecules",
        "datasets_with_all_states",
        "evidence_level",
        "nomination_score",
        "claim_boundary",
    )
    _write_table(output_directory / "composite_states.tsv", rows, fields)
    return dict(analysis["summary"])


def _write_oriented_target_family_outputs(output_directory: Path, targets) -> int:
    """Write a non-mutating target-centered view of direct parent families.

    For BED6 motifs, ``strand`` defines biological orientation.  BED3--BED5
    targets remain in genomic-forward orientation and are explicitly marked
    unstranded.  This artifact is a coordinate transform only: the canonical
    genomic interval and parent family ID are retained byte-for-byte.
    """

    targets_by_id = {target.target_id: target for target in targets}
    rows = []
    with (output_directory / "families.tsv").open(newline="") as handle:
        for family in csv.DictReader(handle, delimiter="\t"):
            direct_ids = [
                value for value in family["direct_target_ids"].split(",") if value
            ]
            for target_id in direct_ids:
                target = targets_by_id[target_id]
                start, end = int(family["start"]), int(family["end"])
                center = (target.start + target.end) / 2.0
                if target.strand == "-":
                    oriented_start = center - end
                    oriented_end = center - start
                    orientation = "motif_reverse"
                else:
                    oriented_start = start - center
                    oriented_end = end - center
                    orientation = (
                        "motif_forward" if target.strand == "+" else "genomic_unstranded"
                    )
                rows.append(
                    {
                        "target_id": target_id,
                        "target_name": target.name,
                        "target_contig": target.contig,
                        "target_start": target.start,
                        "target_end": target.end,
                        "target_center": center,
                        "target_score": target.score,
                        "target_strand": target.strand,
                        "orientation": orientation,
                        "batch_family_id": family["batch_family_id"],
                        "family_slot": family["family_slot"],
                        "family_contig": family["contig"],
                        "family_start": start,
                        "family_end": end,
                        "family_width": end - start,
                        "oriented_start": oriented_start,
                        "oriented_end": oriented_end,
                        "parent_calls_changed": False,
                    }
                )
    fields = (
        "target_id",
        "target_name",
        "target_contig",
        "target_start",
        "target_end",
        "target_center",
        "target_score",
        "target_strand",
        "orientation",
        "batch_family_id",
        "family_slot",
        "family_contig",
        "family_start",
        "family_end",
        "family_width",
        "oriented_start",
        "oriented_end",
        "parent_calls_changed",
    )
    _write_table(output_directory / "oriented_target_families.tsv", rows, fields)
    return len(rows)


def run_targeted_family_batch(args, command_line: str) -> int:
    from fiberhmm.cli import targeted_families as single

    started = time.perf_counter()
    if args.cores < 0:
        raise ValueError("cores must be non-negative")
    if args.unit_workers < 0:
        raise ValueError("unit-workers must be non-negative")
    if args.cores == 0 and args.unit_workers > 1:
        raise ValueError(
            "--cores 0 uses every CPU within a unit and cannot be combined "
            "with --unit-workers greater than 1"
        )
    inputs = tuple(Path(value).expanduser().resolve() for value in args.input)
    if len(set(inputs)) != len(inputs):
        raise ValueError("the same resolved BAM was supplied more than once")
    for path in inputs:
        if not path.is_file():
            raise ValueError(f"input BAM does not exist: {path}")
    with pysam.AlignmentFile(str(inputs[0]), "rb", check_sq=False) as handle:
        reference_lengths = {
            str(name): int(length)
            for name, length in zip(handle.references, handle.lengths)
        }
    bed_path = Path(args.bed).expanduser().resolve()
    targets = load_bed_targets(bed_path, reference_lengths)
    required_contigs = {target.contig for target in targets}
    for path in inputs[1:]:
        with pysam.AlignmentFile(str(path), "rb", check_sq=False) as handle:
            lengths = {
                contig: int(handle.get_reference_length(contig))
                for contig in required_contigs
                if contig in handle.references
            }
        if set(lengths) != required_contigs:
            missing = sorted(required_contigs - set(lengths))
            raise ValueError(f"BED contigs absent from {path}: {missing[:5]}")
        for contig in required_contigs:
            if lengths[contig] != reference_lengths[contig]:
                raise ValueError(f"reference length for {contig} differs across BAMs")
    chemistry, chemistry_records = single._resolve_chemistry(inputs, args.chemistry)
    args.chemistry = chemistry
    resolved_batch_backend = _resolve_batch_backend_without_parent_cuda(
        args.likelihood_backend
    )
    units = plan_targeted_family_work_units(
        targets,
        reference_lengths,
        padding=args.site_padding,
        merge_gap=args.merge_gap,
        maximum_work_unit_bp=args.max_work_unit_bp,
        grid_size=args.window_size,
    )
    signature_payload = {
        "schema": "fiberhmm.targeted_family_batch_plan.v2",
        "scientific_contract": "batch_global_family_candidate_calibration_mask_v1",
        "fiberhmm_version": __version__,
        "bed": str(bed_path),
        "bed_sha256": single._sha256(bed_path),
        "inputs": [str(path) for path in inputs],
        "input_provenance": [
            {
                "path": str(path),
                "sha256": None if args.skip_input_hash else single._bam_sha256(path),
                "sha256_status": (
                    "skipped_explicit_fast_regional_provenance"
                    if args.skip_input_hash
                    else "complete_file"
                ),
                "fast_provenance": single._fast_bam_provenance(path),
            }
            for path in inputs
        ],
        "chemistry": chemistry,
        "targets": [target.as_dict() for target in targets],
        "work_units": [unit.as_dict() for unit in units],
        "parameters": {
            name: value
            for name, value in vars(args).items()
            if name not in {
                "output_dir",
                "resume",
                "subcommand",
                "input",
                "bed",
                "cores",
                "unit_workers",
                "no_aggregate_molecule_assignments",
            }
        },
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    input_provenance = {
        record["path"]: record for record in signature_payload["input_provenance"]
    }
    output_directory = Path(args.output_dir).expanduser().resolve()
    manifest_path = output_directory / "manifest.json"
    previous_manifest = {}
    if output_directory.exists():
        if not args.resume:
            raise ValueError(f"output directory already exists: {output_directory}")
        if not manifest_path.is_file():
            raise ValueError("resume directory lacks a batch manifest")
        previous_manifest = json.loads(manifest_path.read_text())
        if previous_manifest.get("run_signature_sha256") != signature:
            raise ValueError("resume parameters, BED, BAM provenance, or version differ")
    else:
        output_directory.mkdir(parents=True)
        (output_directory / "units").mkdir()
        _write_table(
            output_directory / "targets.tsv",
            (target.as_dict() for target in targets),
            ("ordinal", "target_id", "contig", "start", "end", "name", "score", "strand"),
        )
        _write_table(
            output_directory / "work_units.tsv",
            (
                {
                    **unit.as_dict(),
                    "target_ordinals": ",".join(map(str, unit.target_ordinals)),
                }
                for unit in units
            ),
            (
                "ordinal", "unit_id", "contig", "start", "end", "region",
                "target_ordinals", "oversized_connected_component",
            ),
        )
    batch_manifest = {
        **signature_payload,
        "schema": "fiberhmm.targeted_family_batch.v1",
        "status": "running",
        "command_line": command_line,
        "run_signature_sha256": signature,
        "chemistry_resolution": chemistry_records,
        "completed_units": 0,
        "total_units": len(units),
        "execution_parameters": {
            "cores": int(args.cores),
            "unit_workers_requested": int(args.unit_workers),
            "aggregate_molecule_assignments": not bool(
                args.no_aggregate_molecule_assignments
            ),
        },
    }
    _write_json(manifest_path, batch_manifest)

    internal_workers = (
        max(1, os.cpu_count() or 1) if args.cores == 0 else max(1, args.cores)
    )
    automatic_unit_workers = min(
        len(units),
        16,
        max(1, (os.cpu_count() or 1) // internal_workers),
    )
    unit_workers = args.unit_workers or automatic_unit_workers
    records = []
    discovery_tasks = []
    for index, unit in enumerate(units, start=1):
        unit_directory = output_directory / "units" / unit.unit_id
        unit_directory.mkdir(exist_ok=True)
        state_path = unit_directory / "unit.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        if str(state.get("status", "")).startswith("complete"):
            required = []
            if state.get("targeted_catalog"):
                required.extend(
                    [
                        Path(state["targeted_catalog"]),
                        Path(state["family_memberships"]),
                    ]
                )
            quantification_valid = True
            if int(state.get("retained_family_count", 0)):
                if state.get("status") == "complete_unscorable_unit":
                    quantification_valid = bool(
                        state.get("quantification_skip_reason")
                    )
                else:
                    quantification_valid = bool(
                        state.get("quantification_directory")
                        and state.get("targeted_catalog")
                        and _complete_quantification_matches(
                            Path(state["quantification_directory"]),
                            Path(state["targeted_catalog"]),
                        )
                    )
            if all(path.is_file() for path in required) and quantification_valid:
                records.append(
                    {
                        "unit": unit,
                        "state": state,
                        "state_path": state_path,
                        "complete": True,
                    }
                )
                print(
                    f"BED batch {index}/{len(units)} resumed {unit.unit_id}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
        state = {
            "schema": "fiberhmm.targeted_family_batch_unit.v1",
            "status": "running",
            **unit.as_dict(),
        }
        _write_json(state_path, state)
        discovery_directory = None
        values = None
        for candidate in sorted(unit_directory.glob("discovery*")):
            if (candidate / "catalog.json").is_file():
                discovery_directory = candidate
                break
        if discovery_directory is None:
            discovery_directory = _next_stage_directory(unit_directory, "discovery")
            values = _discover_values(args, unit, discovery_directory)
        record = {
                "unit": unit,
                "state": state,
                "state_path": state_path,
                "discovery_directory": discovery_directory,
                "complete": False,
            }
        records.append(record)
        if values is not None and not (
            discovery_directory / "catalog.json"
        ).is_file():
            discovery_tasks.append(
                (
                    record,
                    (
                        values,
                        input_provenance,
                        unit_directory / f"{discovery_directory.name}.log",
                    ),
                )
            )
    if discovery_tasks:
        print(
            f"BED discovery: units={len(discovery_tasks)} "
            f"concurrent_units={unit_workers} internal_cores={args.cores}",
            file=sys.stderr,
            flush=True,
        )
        discovery_results = _execute_stage(
            [payload for _record, payload in discovery_tasks],
            _run_discovery_stage,
            unit_workers,
            "discovery",
        )
        for (record, _payload), result in zip(discovery_tasks, discovery_results):
            if result["status"] == "empty":
                state = record["state"]
                state.update(
                    {
                        "status": "complete_no_eligible_molecules",
                        "discovery_skip_reason": result["reason"],
                        "retained_family_count": 0,
                    }
                )
                _write_json(record["state_path"], state)
                record["skip_discovery"] = True

    targeted_catalogs = []
    for record in records:
        if record.get("skip_discovery"):
            continue
        state = record["state"]
        if record["complete"]:
            if state.get("targeted_catalog"):
                targeted_catalogs.append(
                    json.loads(Path(state["targeted_catalog"]).read_text())
                )
            continue
        unit = record["unit"]
        state_path = record["state_path"]
        discovery_directory = record["discovery_directory"]
        targeted_catalog_path = discovery_directory / "targeted_catalog.json"
        membership_path = discovery_directory / "targeted_family_memberships.tsv"
        catalog, _memberships = _targeted_catalog(
            discovery_directory / "catalog.json",
            targeted_catalog_path,
            membership_path,
            unit=unit,
            targets=targets,
            reference_lengths=reference_lengths,
            padding=args.site_padding,
        )
        targeted_catalogs.append(catalog)
        state.update(
            {
                "discovery_directory": str(discovery_directory),
                "targeted_catalog": str(targeted_catalog_path),
                "family_memberships": str(membership_path),
                "retained_family_count": len(catalog["families"]),
            }
        )
        _write_json(state_path, state)

    efficiency_exclusion_path = output_directory / "efficiency_exclusion_intervals.bed"
    efficiency_exclusion = _write_global_efficiency_exclusions(
        efficiency_exclusion_path,
        targets=targets,
        reference_lengths=reference_lengths,
        site_padding=args.site_padding,
        catalogs=targeted_catalogs,
        chemistry=chemistry,
        candidate_interval_builder=single._family_candidate_exclusion_intervals,
    )
    for record in records:
        state = record["state"]
        if (
            record["complete"]
            and int(state.get("retained_family_count", 0))
            and state.get("status") != "complete_unscorable_unit"
            and not _complete_quantification_matches(
                Path(state["quantification_directory"]),
                Path(state["targeted_catalog"]),
                expected_efficiency_exclusion_sha256=efficiency_exclusion["sha256"],
            )
        ):
            record["complete"] = False
            state["status"] = "running"
            _write_json(record["state_path"], state)
            print(
                f"BED resume reopened {state['unit_id']}: calibration mask changed",
                file=sys.stderr,
                flush=True,
            )

    quantification_tasks = []
    for record in records:
        if record["complete"] or record.get("skip_discovery"):
            continue
        state = record["state"]
        state_path = record["state_path"]
        unit_directory = state_path.parent
        if int(state.get("retained_family_count", 0)):
            targeted_catalog_path = Path(state["targeted_catalog"])
            quantification_directory = None
            for candidate in sorted(unit_directory.glob("quantification*")):
                if _complete_quantification_matches(
                    candidate,
                    targeted_catalog_path,
                    expected_efficiency_exclusion_sha256=efficiency_exclusion[
                        "sha256"
                    ],
                ):
                    quantification_directory = candidate
                    break
            if quantification_directory is None:
                quantification_directory = _next_stage_directory(
                    unit_directory, "quantification"
                )
                values = _quantify_values(
                    args,
                    targeted_catalog_path,
                    quantification_directory,
                    efficiency_exclusion_bed=efficiency_exclusion_path,
                )
                quantification_tasks.append(
                    (
                        record,
                        (
                            values,
                            input_provenance,
                            unit_directory / f"{quantification_directory.name}.log",
                        ),
                    )
                )
            state["quantification_directory"] = str(quantification_directory)
        _write_json(state_path, state)

    quantification_unit_workers = (
        unit_workers if resolved_batch_backend == "cpu" else 1
    )
    if quantification_tasks:
        print(
            f"BED quantification: units={len(quantification_tasks)} "
            f"concurrent_units={quantification_unit_workers} "
            f"backend={args.likelihood_backend} resolved={resolved_batch_backend}",
            file=sys.stderr,
            flush=True,
        )
        quantification_results = _execute_stage(
            [payload for _record, payload in quantification_tasks],
            _run_quantification_stage,
            quantification_unit_workers,
            "quantification",
            isolate_each=resolved_batch_backend != "cpu",
        )
        for (record, _payload), result in zip(
            quantification_tasks, quantification_results
        ):
            if result["status"] == "empty":
                state = record["state"]
                state.pop("quantification_directory", None)
                state.update(
                    {
                        "status": "complete_unscorable_unit",
                        "quantification_skip_reason": result["reason"],
                    }
                )
                _write_json(record["state_path"], state)
                record["skip_quantification"] = True

    unit_states = []
    for record in records:
        state = record["state"]
        state_path = record["state_path"]
        if state.get("status") == "running":
            state["status"] = "complete"
        _write_json(state_path, state)
        unit_states.append(state)
    unit_states.sort(key=lambda value: int(value["ordinal"]))
    batch_manifest["completed_units"] = len(unit_states)
    batch_manifest["parallelism"] = {
        "unit_workers": int(unit_workers),
        "internal_cores_per_unit": int(args.cores),
        "requested_likelihood_backend": args.likelihood_backend,
        "resolved_likelihood_backend": resolved_batch_backend,
        "discovery": "parallel_work_units",
        "quantification": (
            "parallel_work_units"
            if quantification_unit_workers > 1
            else "sequential_single_cuda_owner_or_single_cpu_unit"
        ),
    }
    batch_manifest["efficiency_exclusion"] = efficiency_exclusion
    previous_compute_seconds = float(
        previous_manifest.get("compute_elapsed_seconds", 0.0)
    )
    stage_work_ran = bool(discovery_tasks or quantification_tasks)
    batch_manifest["compute_elapsed_seconds"] = previous_compute_seconds + (
        time.perf_counter() - started if stage_work_ran else 0.0
    )
    _write_json(manifest_path, batch_manifest)

    aggregate_counts = _aggregate_outputs(
        output_directory,
        unit_states,
        aggregate_molecules=not args.no_aggregate_molecule_assignments,
        chemistry=chemistry,
    )
    composite_counts = _write_composite_state_outputs(output_directory)
    aggregate_counts["oriented_target_family_rows"] = (
        _write_oriented_target_family_outputs(output_directory, targets)
    )
    aggregate_counts["composite_state_nominations"] = int(
        composite_counts["nomination_count"]
    )
    aggregate_counts["composite_state_envelopes"] = int(
        composite_counts["nominated_envelopes"]
    )
    batch_manifest.update(
        {
            "status": "complete",
            "completed_units": len(unit_states),
            "aggregate_counts": aggregate_counts,
            "compute_elapsed_seconds": batch_manifest["compute_elapsed_seconds"],
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
            "contracts": {
                "work_units_non_overlapping": True,
                "overlapping_padded_targets_never_split": True,
                "gap_only_families_pruned_before_fit": True,
                "cuda_work_units_use_one_sequential_owner": True,
                "single_region_cli_code_path_unchanged": True,
                "efficiency_calibration_independent_of_work_unit_partition": True,
                "composite_states_preserve_parent_family_calls": True,
                "composite_states_are_not_dimer_or_factor_identity_claims": True,
                "oriented_target_families_are_coordinate_transforms_only": True,
            },
            "artifacts": {
                "targets": "targets.tsv",
                "work_units": "work_units.tsv",
                "efficiency_exclusion_intervals": "efficiency_exclusion_intervals.bed",
                "families": "families.tsv",
                "family_scores": "family_scores.tsv",
                "composite_states": "composite_states.tsv",
                "composite_state_analysis": "composite_states.json",
                "oriented_target_families": "oriented_target_families.tsv",
                "molecule_family_scores": (
                    None
                    if args.no_aggregate_molecule_assignments
                    else "molecule_family_scores.tsv"
                ),
                "units": "units/",
            },
        }
    )
    _write_json(manifest_path, batch_manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output_directory),
                "targets": len(targets),
                "work_units": len(units),
                **aggregate_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = ["add_batch_parser", "run_targeted_family_batch"]
