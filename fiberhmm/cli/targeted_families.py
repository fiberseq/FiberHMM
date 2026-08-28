#!/usr/bin/env python3
"""Offline targeted footprint-family discovery and application."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import sys
import time
from bisect import bisect_left, bisect_right
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pysam

from fiberhmm import __version__
from fiberhmm.cli.common import add_version_args
from fiberhmm.inference.targeted_families import (
    CHEMISTRY_PROFILES,
    TargetedFamilyDiscoveryConfig,
    collapse_daf_amplification_families,
    discover_targeted_families,
    score_boundary_families_on_unbiased_cohort,
)
from fiberhmm.inference.tf_family_ids import (
    TFFamilyInterval,
    allocate_repeating_family_ids,
)
from fiberhmm.inference.tf_sites import BaselineMolecule, TFObservation
from fiberhmm.inference.mp_context import _MP_CONTEXT
from fiberhmm.io.bam_header import declared_chemistries, infer_legacy_chemistry
from fiberhmm.io.footprint_bam import (
    BamFootprintInputError,
    load_footprint_molecules_from_bam,
    parse_reference_region,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


_BAM_SHA256_CACHE = {}
_BAM_FAST_PROVENANCE_CACHE = {}


def _bam_cache_key(path: Path) -> tuple:
    path = path.expanduser().resolve()
    stat = path.stat()
    return str(path), int(stat.st_size), int(stat.st_mtime_ns)


def _bam_sha256(path: Path) -> str:
    """Hash a stable BAM once per process, including across BED work units."""

    key = _bam_cache_key(path)
    digest = _BAM_SHA256_CACHE.get(key)
    if digest is None:
        digest = _sha256(path)
        _BAM_SHA256_CACHE[key] = digest
    return digest


def _progress_checkpoint(completed: int, total: int, maximum_updates: int = 100) -> bool:
    """Bound progress output while retaining first, last, and regular updates."""

    stride = max(1, int(math.ceil(max(1, total) / maximum_updates)))
    return completed == 1 or completed == total or completed % stride == 0


def _fast_bam_provenance(path: Path) -> dict:
    """Fingerprint BAM structure without rereading the complete data payload."""

    cache_key = _bam_cache_key(path)
    cached = _BAM_FAST_PROVENANCE_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    stat = path.stat()
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as handle:
        header_text = str(handle.header)
    candidates = (
        Path(str(path) + ".bai"),
        path.with_suffix(".bai"),
        Path(str(path) + ".csi"),
        path.with_suffix(".csi"),
    )
    index_path = next((value for value in candidates if value.is_file()), None)
    index = None
    if index_path is not None:
        index_stat = index_path.stat()
        index = {
            "name": index_path.name,
            "size": int(index_stat.st_size),
            "mtime_ns": int(index_stat.st_mtime_ns),
            "sha256": _sha256(index_path),
        }
    provenance = {
        "bam_size": int(stat.st_size),
        "bam_mtime_ns": int(stat.st_mtime_ns),
        "header_sha256": hashlib.sha256(header_text.encode("utf-8")).hexdigest(),
        "index": index,
    }
    _BAM_FAST_PROVENANCE_CACHE[cache_key] = provenance
    return dict(provenance)


def _profile_from_declaration(declaration: Mapping[str, str]) -> str | None:
    enzyme = str(declaration.get("enzyme", "")).lower()
    platform = str(declaration.get("platform", "")).lower()
    assay = str(declaration.get("assay", "")).lower()
    if enzyme in {"ddda", "dddb"}:
        return enzyme
    if enzyme == "hia5" or assay == "fiber-seq":
        if platform == "pacbio":
            return "hia5-pacbio"
        if platform in {"nanopore", "ont"}:
            return "hia5-nanopore"
    return None


def _bam_chemistry(path: Path) -> tuple[str | None, dict | None, str]:
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as handle:
        declarations = declared_chemistries(handle.header)
        if len(declarations) > 1:
            profiles = {_profile_from_declaration(value) for value in declarations}
            profiles.discard(None)
            if len(profiles) != 1:
                raise ValueError(f"ambiguous chemistry declarations in {path}")
        if declarations:
            declaration = declarations[-1]
            return _profile_from_declaration(declaration), declaration, "declared_v1"
        legacy = infer_legacy_chemistry(handle.header)
        return (
            _profile_from_declaration(legacy) if legacy else None,
            legacy,
            "legacy_pg_inference" if legacy else "missing",
        )


def _resolve_chemistry(paths: Sequence[Path], requested: str | None):
    records = []
    detected = set()
    for path in paths:
        profile, declaration, source = _bam_chemistry(path)
        records.append(
            {
                "path": str(path),
                "profile": profile,
                "source": source,
                "declaration": declaration,
            }
        )
        if profile is not None:
            detected.add(profile)
    if len(detected) > 1:
        raise ValueError(
            "inputs declare different chemistries; discover separate catalogs "
            "before cross-dataset geometry matching"
        )
    detected_profile = next(iter(detected), None)
    if requested is not None and detected_profile is not None and requested != detected_profile:
        raise ValueError(
            f"--chemistry {requested} conflicts with BAM chemistry {detected_profile}"
        )
    selected = requested or detected_profile
    if selected is None:
        raise ValueError(
            "chemistry is absent or ambiguous in the BAM header; provide --chemistry"
        )
    return selected, records


def _prefix_molecules(molecules, prefix: str):
    return tuple(
        BaselineMolecule(
            molecule_id=f"{prefix}\x1f{molecule.molecule_id}",
            contig=molecule.contig,
            mapped_blocks=molecule.mapped_blocks,
            tfs=tuple(
                TFObservation(
                    call_id=f"{prefix}.{call.call_id}",
                    start=call.start,
                    end=call.end,
                    geometry_eligible=call.geometry_eligible,
                )
                for call in molecule.tfs
            ),
            msps=molecule.msps,
            stratum=molecule.stratum,
        )
        for molecule in molecules
    )


def _write_tsv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _select_family_training_reads(
    reads,
    family,
    windows,
    *,
    maximum: int,
    seed: str,
    eligible_reads=None,
):
    source_ordinals = {int(value) for value in family["source_window_ordinals"]}
    cores = [
        (int(window["core_start"]), int(window["core_end"]))
        for window in windows
        if int(window["ordinal"]) in source_ordinals
    ]
    minimum_nfr_length = int(family["discovery_minimum_nfr_length"])
    if eligible_reads is None:
        eligible = []
        for read in reads:
            if not any(
                msp.end - msp.start >= minimum_nfr_length
                and any(msp.start < end and start < msp.end for start, end in cores)
                for msp in read.msps
            ):
                continue
            eligible.append(read)
    else:
        eligible = list(eligible_reads)
    groups = {}
    for read in eligible:
        groups.setdefault((str(read.library_id or ""), read.strand), []).append(read)
    if len(eligible) <= maximum:
        selected = eligible
    else:
        exact = {
            key: maximum * len(values) / len(eligible) for key, values in groups.items()
        }
        quotas = {key: int(math.floor(value)) for key, value in exact.items()}
        remaining = maximum - sum(quotas.values())
        for key in sorted(
            groups,
            key=lambda value: (-(exact[value] - math.floor(exact[value])), value),
        ):
            if remaining <= 0:
                break
            quotas[key] += 1
            remaining -= 1
        selected = []
        family_id = str(family["family_id"])
        for key in sorted(groups):
            ordered = sorted(
                groups[key],
                key=lambda read: (
                    hashlib.sha256(
                        "\x1f".join(
                            (
                                seed,
                                family_id,
                                str(read.library_id or ""),
                                read.name,
                                read.strand,
                            )
                        ).encode("utf-8")
                    ).digest(),
                    read.molecule_id,
                ),
            )
            selected.extend(ordered[: quotas[key]])
    selected.sort(key=lambda read: read.molecule_id)
    return tuple(selected), len(eligible)


def _index_family_training_reads(reads, windows, *, minimum_nfr_length: int):
    """Index long-MSP overlap once for every source-window core."""

    ordered_windows = sorted(
        (
            int(window["core_start"]),
            int(window["core_end"]),
            int(window["ordinal"]),
        )
        for window in windows
    )
    starts = [value[0] for value in ordered_windows]
    ends = [value[1] for value in ordered_windows]
    by_ordinal = {value[2]: [] for value in ordered_windows}
    for read in reads:
        matched = set()
        for msp in read.msps:
            if msp.end - msp.start < minimum_nfr_length:
                continue
            first = bisect_right(ends, int(msp.start))
            stop = bisect_left(starts, int(msp.end))
            for index in range(first, stop):
                start, end, ordinal = ordered_windows[index]
                if int(msp.start) < end and start < int(msp.end):
                    matched.add(ordinal)
        for ordinal in sorted(matched):
            by_ordinal[ordinal].append(read)
    return {ordinal: tuple(values) for ordinal, values in by_ordinal.items()}


def _fit_family_worker(payload):
    family, selected, chemistry = payload
    from fiberhmm.inference.strand_rescue import (
        fit_boundary_marginalized_tf_family_model,
    )

    profile = CHEMISTRY_PROFILES[chemistry]
    model = fit_boundary_marginalized_tf_family_model(
        selected,
        str(family["family_id"]),
        [tuple(int(value) for value in interval) for interval in family["seed_intervals"]],
        boundary_search_radius=profile.boundary_search_radius,
        minimum_molecule_opportunities=profile.minimum_molecule_opportunities,
        stratum_semantics=(
            "physical_complementary"
            if chemistry in {"ddda", "dddb"}
            else "diagnostic_partition"
        ),
        evidence_summation_mode="prefix",
    )
    return model


def _family_fit_eligible_reads(reads, family, chemistry: str):
    """Apply the fitter's exact envelope/opportunity eligibility before spawn."""

    profile = CHEMISTRY_PROFILES[chemistry]
    radius = int(profile.boundary_search_radius)
    seeds = [
        (int(interval[0]), int(interval[1]))
        for interval in family["seed_intervals"]
    ]
    # Mirrors fit_boundary_marginalized_tf_family_model's default candidate
    # expansion and 20-bp spatial-null padding.
    envelope_start = min(start for start, _end in seeds) - radius - 20
    envelope_end = max(end for _start, end in seeds) + radius + 20
    minimum = int(profile.minimum_molecule_opportunities)
    return tuple(
        read
        for read in reads
        if read.fully_maps(envelope_start, envelope_end)
        and read.interval_evidence(envelope_start, envelope_end)[1] >= minimum
    )


def _family_candidate_exclusion_intervals(families, chemistry: str):
    """Return the merged union of every plausible anchored-family geometry.

    Efficiency is calibrated from accessible MSP opportunities outside the
    model under test.  Excluding only the catalog median would allow shifted
    boundary candidates to teach the background efficiency, especially for
    lower-resolution DddB and Hia5 data.  The union envelope below is exactly
    the coordinate support generated by the symmetric boundary search for
    each family's frozen seeds; adjacent envelopes are merged only to make
    the per-read mask application cheaper.
    """

    if chemistry not in CHEMISTRY_PROFILES:
        raise ValueError(f"unsupported targeted-family chemistry: {chemistry}")
    radius = int(CHEMISTRY_PROFILES[chemistry].boundary_search_radius)
    intervals = []
    for family in families:
        seeds = [
            (int(interval[0]), int(interval[1]))
            for interval in family.get("seed_intervals", ())
        ]
        if not seeds:
            seeds = [(int(family["start"]), int(family["end"]))]
        start = min(value[0] for value in seeds) - radius
        end = max(value[1] for value in seeds) + radius
        if end <= start:
            raise ValueError(f"family {family.get('family_id', '')!r} has invalid geometry")
        intervals.append((start, end))
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _merge_interval_union(intervals):
    merged = []
    for start, end in sorted((int(start), int(end)) for start, end in intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _bed_efficiency_exclusion_intervals(
    path: Path,
    *,
    contig: str,
    contig_length: int,
    padding: int,
):
    intervals = []
    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith(("#", "track ", "browser ")):
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                raise ValueError(
                    f"efficiency-exclusion BED line {line_number} has fewer than three columns"
                )
            if fields[0] != contig:
                continue
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError as error:
                raise ValueError(
                    f"efficiency-exclusion BED line {line_number} has invalid coordinates"
                ) from error
            if start < 0 or end <= start or end > contig_length:
                raise ValueError(
                    f"efficiency-exclusion BED line {line_number} is out of bounds"
                )
            intervals.append(
                (max(0, start - padding), min(contig_length, end + padding))
            )
    return _merge_interval_union(intervals)


def _load_evidence_worker(payload):
    (
        path,
        contig,
        start,
        end,
        strand_mode,
        mode,
        context_size,
        probability_threshold,
        llr_hit,
        llr_miss,
        min_mapq,
        tf_layer,
        nuc_layer,
        input_index,
        required_read_names,
        evidence_scope,
    ) = payload
    from fiberhmm.inference.strand_rescue import load_region_evidence

    diagnostics = {}
    reads = load_region_evidence(
        str(path),
        contig,
        start,
        end,
        strand_mode=strand_mode,
        mode=mode,
        context_size=context_size,
        prob_threshold=probability_threshold,
        llr_hit=llr_hit,
        llr_miss=llr_miss,
        min_mapq=min_mapq,
        tf_layer=tf_layer,
        nuc_layer=nuc_layer,
        input_index=input_index,
        required_read_names=required_read_names,
        projection="targeted_family",
        evidence_scope=evidence_scope,
        load_diagnostics=diagnostics,
    )
    return reads, diagnostics


def _clip_calibrated_evidence_to_region(reads, start: int, end: int) -> dict:
    """Drop calibration-only flanking opportunities before family scoring."""

    before = 0
    after = 0
    for read in reads:
        positions = np.asarray(read.positions, dtype=np.int64)
        before += int(positions.size)
        retained = (positions >= int(start)) & (positions < int(end))
        read.positions = positions[retained]
        read.steps = np.asarray(read.steps)[retained]
        read.hits = np.asarray(read.hits)[retained]
        read.contexts = np.asarray(read.contexts)[retained]
        if read.nuc_steps is not None:
            read.nuc_steps = np.asarray(read.nuc_steps)[retained]
        after += int(np.sum(retained))
    return {
        "calibration_opportunities_before_local_projection": before,
        "scoring_opportunities_after_local_projection": after,
        "scoring_region": [int(start), int(end)],
    }


def _load_catalog(path_value: str | Path):
    path = Path(path_value).expanduser().resolve()
    if path.is_dir():
        path = path / "catalog.json"
    if not path.is_file():
        raise ValueError(f"catalog does not exist: {path}")
    catalog = json.loads(path.read_text())
    if catalog.get("schema") != "fiberhmm.targeted_family_discovery.v1":
        raise ValueError("--catalog is not a targeted-family discovery v1 artifact")
    if catalog.get("contracts", {}).get("occupancy_cohort") != "full_unbiased_required":
        raise ValueError("catalog lacks the full-unbiased occupancy contract")
    return path, catalog


def _catalog_independent_read_names(catalog_path: Path, catalog, input_count: int):
    relative = catalog.get("artifacts", {}).get("independent_molecules")
    if not relative:
        raise ValueError(
            "catalog predates the independent-molecule allowlist; rerun discovery"
        )
    table = catalog_path.parent / str(relative)
    if not table.is_file():
        raise ValueError(f"catalog independent-molecule table is missing: {table}")
    names = [set() for _index in range(input_count)]
    with table.open(newline="") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            input_index = int(record["input_index"])
            if not 0 <= input_index < input_count:
                raise ValueError("independent-molecule table has an invalid input index")
            read_name = str(record["read_name"])
            if read_name in names[input_index]:
                raise ValueError(
                    f"duplicate independent read name for input {input_index}: {read_name}"
                )
            names[input_index].add(read_name)
    expected = [
        int(value["retained_independent_molecules"]) for value in catalog["inputs"]
    ]
    observed = [len(value) for value in names]
    if observed != expected:
        raise ValueError(
            f"independent-molecule table counts differ: observed={observed}, expected={expected}"
        )
    return tuple(frozenset(value) for value in names), table


def _load_reused_family_models(
    source_directory: Path,
    *,
    catalog_path: Path,
    catalog: Mapping,
    inputs: Sequence[Path],
    chemistry: str,
    model_fit_config: Mapping[str, object],
):
    """Load a previously fitted model set after strict provenance checks."""

    source_directory = source_directory.expanduser().resolve()
    manifest_path = source_directory / "manifest.json"
    models_directory = source_directory / "models"
    models_jsonl = source_directory / "models.jsonl"
    if not manifest_path.is_file() or not (
        models_directory.is_dir() or models_jsonl.is_file()
    ):
        raise ValueError(
            "--reuse-models-from must contain manifest.json and either models/ "
            "or models.jsonl: "
            f"{source_directory}"
        )
    source_manifest = json.loads(manifest_path.read_text())
    if source_manifest.get("status") != "complete_unbiased_independent_family_screen":
        raise ValueError("reused model source is not a complete quantification")
    if source_manifest.get("catalog_sha256") != _sha256(catalog_path):
        raise ValueError("reused models were fitted from a different discovery catalog")
    if source_manifest.get("chemistry") != chemistry:
        raise ValueError("reused model chemistry differs from the discovery catalog")
    if tuple(source_manifest.get("inputs", ())) != tuple(str(path) for path in inputs):
        raise ValueError("reused model inputs or input order differ")
    if source_manifest.get("fiberhmm_version") != __version__:
        raise ValueError("reused models were fitted by a different FiberHMM version")

    source_fit_config = source_manifest.get("model_fit_config")
    fit_config_validation = "manifest_v1"
    if source_fit_config is None:
        # Development outputs predating model_fit_config still contain the
        # exact shell-quoted CLI. Recover only the cohort-shaping options; the
        # same-version requirement above guards code/model drift for this
        # compatibility path.
        source_tokens = shlex.split(str(source_manifest.get("command_line", "")))

        def command_option(name: str, default: str) -> str:
            for index, token in enumerate(source_tokens):
                if token == name and index + 1 < len(source_tokens):
                    return source_tokens[index + 1]
                if token.startswith(name + "="):
                    return token.split("=", 1)[1]
            return default

        source_fit_config = {
            "min_mapq": int(command_option("--min-mapq", "20")),
            "tf_layer": command_option("--tf-layer", "tf_sr"),
            "nuc_layer": command_option("--nuc-layer", "nuc_sr"),
            "likelihood_model_sha256": None,
            "efficiency_exclusion_bed_sha256": None,
            "efficiency_exclusion_padding": 0,
            "efficiency_evidence_scope": "region",
        }
        fit_config_validation = "legacy_command_line_plus_same_version"
    for name in (
        "min_mapq",
        "tf_layer",
        "nuc_layer",
        "efficiency_exclusion_bed_sha256",
        "efficiency_exclusion_padding",
        "efficiency_evidence_scope",
    ):
        if source_fit_config.get(name) != model_fit_config.get(name):
            raise ValueError(
                f"reused model {name} differs: source="
                f"{source_fit_config.get(name)!r}, requested="
                f"{model_fit_config.get(name)!r}"
            )
    source_model_sha = source_fit_config.get("likelihood_model_sha256")
    if (
        source_model_sha is not None
        and source_model_sha != model_fit_config.get("likelihood_model_sha256")
    ):
        raise ValueError("reused likelihood model SHA-256 differs")

    skipped = list(source_manifest.get("skipped_unscorable_families", ()))
    skipped_ids = {str(value["family_id"]) for value in skipped}
    catalog_ids = [str(value["family_id"]) for value in catalog["families"]]
    expected_ids = [value for value in catalog_ids if value not in skipped_ids]
    compact_models = None
    if models_directory.is_dir():
        available_ids = {path.stem for path in models_directory.glob("*.json")}
    else:
        compact_models = {}
        with models_jsonl.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                model = json.loads(line)
                family_id = str(model.get("family_id", ""))
                if not family_id or family_id in compact_models:
                    raise ValueError(
                        f"invalid or duplicate family in models.jsonl line {line_number}"
                    )
                compact_models[family_id] = model
        available_ids = set(compact_models)
    if available_ids != set(expected_ids):
        missing = sorted(set(expected_ids) - available_ids)
        extra = sorted(available_ids - set(expected_ids))
        raise ValueError(
            "reused model files do not exactly match the scorable catalog families: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    models = []
    for family_id in expected_ids:
        model = (
            compact_models[family_id]
            if compact_models is not None
            else json.loads((models_directory / f"{family_id}.json").read_text())
        )
        if str(model.get("family_id")) != family_id:
            raise ValueError(f"reused model identity mismatch: {family_id}")
        models.append(model)
    if len(models) != int(source_manifest.get("family_count", -1)):
        raise ValueError("reused model count differs from its source manifest")
    training_cohorts = list(source_manifest.get("training_cohorts", ()))
    training_artifact = source_manifest.get("artifacts", {}).get(
        "training_molecules"
    )
    if training_artifact:
        selected_by_family = {}
        with (source_directory / str(training_artifact)).open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                selected_by_family.setdefault(row["family_id"], []).append(
                    [row["library_id"], row["read_name"], row["strand"]]
                )
        training_cohorts = [
            {
                **record,
                "selected_molecule_ids": selected_by_family.get(
                    record["family_id"], []
                ),
            }
            for record in training_cohorts
        ]
    return (
        models,
        training_cohorts,
        skipped,
        {
            "source_directory": str(source_directory),
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": _sha256(manifest_path),
            "model_fit_config_validation": fit_config_validation,
            "source_model_fit_config": source_fit_config,
            "model_artifact_format": (
                "jsonl" if compact_models is not None else "per_family_json"
            ),
        },
    )


def _quantify(args, command_line: str) -> int:
    if args.efficiency_exclusion_padding < 0:
        raise ValueError("efficiency-exclusion padding must be non-negative")
    catalog_path, catalog = _load_catalog(args.catalog)
    inputs = tuple(Path(value).expanduser().resolve() for value in args.input)
    catalog_inputs = tuple(Path(value["path"]).resolve() for value in catalog["inputs"])
    if inputs != catalog_inputs:
        raise ValueError("quantification inputs and order must match the discovery catalog")
    batch_input_provenance = getattr(args, "_batch_input_provenance", {})
    for path, expected in zip(inputs, catalog["inputs"]):
        if not path.is_file():
            raise ValueError(f"input BAM does not exist: {path}")
        if path.stat().st_size != int(expected["size"]):
            raise ValueError(f"input BAM size changed since discovery: {path}")
        if args.skip_input_hash_check:
            expected_fast = expected.get("fast_provenance")
            observed_fast = batch_input_provenance.get(str(path), {}).get(
                "fast_provenance"
            ) or _fast_bam_provenance(path)
            if expected_fast is not None and observed_fast != expected_fast:
                raise ValueError(
                    f"input BAM/header/index metadata changed since discovery: {path}"
                )
        else:
            expected_sha256 = expected.get("sha256")
            if not expected_sha256:
                raise ValueError(
                    "catalog discovery explicitly skipped full BAM hashing; pass "
                    "--skip-input-hash-check to validate its fast regional provenance"
                )
            observed_sha256 = batch_input_provenance.get(str(path), {}).get(
                "sha256"
            ) or _bam_sha256(path)
            if observed_sha256 != expected_sha256:
                raise ValueError(f"input BAM SHA-256 changed since discovery: {path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    families = list(catalog["families"])
    if not families:
        raise ValueError("catalog contains no footprint families")
    chemistry = str(catalog["chemistry"])
    if chemistry not in CHEMISTRY_PROFILES:
        raise ValueError(f"unsupported catalog chemistry: {chemistry}")
    workers = args.cores or max(1, os.cpu_count() or 1)
    independent_names, independent_table = _catalog_independent_read_names(
        catalog_path, catalog, len(inputs)
    )
    contig = str(catalog["contig"])
    locus_start, locus_end = (int(value) for value in catalog["locus"])
    evidence_start, evidence_end = (
        int(value) for value in catalog["loaded_evidence_region"][1:]
    )

    from fiberhmm.core.model_io import load_model_with_metadata
    from fiberhmm.inference.strand_rescue import (
        PRESETS,
        calibrate_cohort_efficiency,
        resolve_resource_path,
    )
    from fiberhmm.inference.tf_recaller import build_llr_tables

    if chemistry in PRESETS:
        preset = PRESETS[chemistry]
        model_path = Path(resolve_resource_path(str(preset["model"])))
        strand_mode = str(preset["strand_mode"])
        probability_threshold = preset["prob_threshold"]
    elif chemistry == "hia5-pacbio":
        model_path = Path(resolve_resource_path("models/hia5_pacbio.json"))
        strand_mode = "alignment"
        probability_threshold = 125
    else:
        raise ValueError(f"no likelihood preset for {chemistry}")
    hmm, context_size, mode = load_model_with_metadata(str(model_path))
    llr_hit, llr_miss = build_llr_tables(hmm)
    efficiency_exclusion_bed = (
        Path(args.efficiency_exclusion_bed).expanduser().resolve()
        if args.efficiency_exclusion_bed
        else None
    )
    if efficiency_exclusion_bed is not None and not efficiency_exclusion_bed.is_file():
        raise ValueError(
            f"efficiency-exclusion BED does not exist: {efficiency_exclusion_bed}"
        )
    model_fit_config = {
        "min_mapq": int(args.min_mapq),
        "tf_layer": str(args.tf_layer),
        "nuc_layer": str(args.nuc_layer),
        "likelihood_model_sha256": _sha256(model_path),
        "efficiency_exclusion_bed_sha256": (
            _sha256(efficiency_exclusion_bed)
            if efficiency_exclusion_bed is not None
            else None
        ),
        "efficiency_exclusion_padding": int(args.efficiency_exclusion_padding),
        "efficiency_evidence_scope": str(args.efficiency_evidence_scope),
    }
    reused_model_bundle = None
    if args.reuse_models_from:
        reused_model_bundle = _load_reused_family_models(
            Path(args.reuse_models_from),
            catalog_path=catalog_path,
            catalog=catalog,
            inputs=inputs,
            chemistry=chemistry,
            model_fit_config=model_fit_config,
        )
    evidence = []
    load_diagnostics = []
    load_started = time.perf_counter()
    load_payloads = [
        (
            path,
            contig,
            evidence_start,
            evidence_end,
            strand_mode,
            mode,
            context_size,
            probability_threshold,
            llr_hit,
            llr_miss,
            args.min_mapq,
            args.tf_layer,
            args.nuc_layer,
            input_index,
            independent_names[input_index],
            args.efficiency_evidence_scope,
        )
        for input_index, path in enumerate(inputs)
    ]
    if workers == 1 or len(load_payloads) <= 1:
        load_iterator = map(_load_evidence_worker, load_payloads)
        load_executor = None
    else:
        load_executor = ProcessPoolExecutor(
            max_workers=min(workers, len(load_payloads)),
            mp_context=_MP_CONTEXT,
        )
        load_iterator = load_executor.map(_load_evidence_worker, load_payloads)
    try:
        for input_index, (current_reads, diagnostics) in enumerate(
            load_iterator
        ):
            evidence.extend(current_reads)
            load_diagnostics.append(diagnostics)
            print(
                f"full-cohort load {input_index + 1}/{len(inputs)} {inputs[input_index].name}: "
                f"eligible={diagnostics.get('eligible_read_count', 0)}",
                file=sys.stderr,
                flush=True,
            )
    finally:
        if load_executor is not None:
            load_executor.shutdown()
    collapse = {
        "mode": "reuse_exact_discovery_independent_molecule_allowlist",
        "table": str(independent_table),
        "table_sha256": _sha256(independent_table),
        "retained_reads": len(evidence),
        "per_input_discovery_collapse": [
            value["amplification_collapse"] for value in catalog["inputs"]
        ],
    }
    evidence.sort(key=lambda read: read.molecule_id)
    if not evidence:
        raise ValueError("no independent full-cohort molecules")
    candidate_intervals = _family_candidate_exclusion_intervals(
        families, chemistry
    )
    if efficiency_exclusion_bed is not None:
        with pysam.AlignmentFile(str(inputs[0]), "rb", check_sq=False) as handle:
            contig_length = int(handle.get_reference_length(contig))
        candidate_intervals = _merge_interval_union(
            (
                *candidate_intervals,
                *_bed_efficiency_exclusion_intervals(
                    efficiency_exclusion_bed,
                    contig=contig,
                    contig_length=contig_length,
                    padding=int(args.efficiency_exclusion_padding),
                ),
            )
        )
    efficiency = calibrate_cohort_efficiency(
        evidence,
        hmm,
        pseudo_count=20.0,
        min_opportunities=20,
        excluded_intervals=candidate_intervals,
    )
    efficiency["evidence_scope"] = args.efficiency_evidence_scope
    if args.efficiency_evidence_scope == "full-alignment":
        efficiency.update(
            _clip_calibrated_evidence_to_region(
                evidence,
                evidence_start,
                evidence_end,
            )
        )
    load_elapsed = time.perf_counter() - load_started
    print(
        f"full unbiased cohort ready: molecules={len(evidence)} elapsed={load_elapsed:.2f}s",
        file=sys.stderr,
        flush=True,
    )

    reused_model_provenance = None
    if reused_model_bundle is not None:
        (
            models,
            training_provenance,
            skipped_unscorable_families,
            reused_model_provenance,
        ) = reused_model_bundle
        selection_elapsed = 0.0
        fit_elapsed = 0.0
        print(
            f"reused {len(models)} frozen family models from "
            f"{reused_model_provenance['source_directory']}",
            file=sys.stderr,
            flush=True,
        )
    else:
        discovery_windows = list(catalog["windows"])
        discovery_config = catalog["config"]
        selection_started = time.perf_counter()
        training_reads_by_window = _index_family_training_reads(
            evidence,
            discovery_windows,
            minimum_nfr_length=int(discovery_config["minimum_nfr_length"]),
        )
        fit_payloads = []
        training_provenance = []
        skipped_unscorable_families = []
        for family in families:
            family_for_selection = {
                **family,
                "discovery_minimum_nfr_length": discovery_config[
                    "minimum_nfr_length"
                ],
            }
            eligible_by_id = {}
            for ordinal in family["source_window_ordinals"]:
                for read in training_reads_by_window.get(int(ordinal), ()):
                    eligible_by_id[read.molecule_id] = read
            eligible_reads = tuple(
                eligible_by_id[molecule_id]
                for molecule_id in sorted(eligible_by_id)
            )
            eligible_reads = _family_fit_eligible_reads(
                eligible_reads, family, chemistry
            )
            if not eligible_reads:
                skipped_unscorable_families.append(
                    {
                        "family_id": family["family_id"],
                        "reason": (
                            "no_full_envelope_molecule_with_minimum_"
                            "sequence_opportunities"
                        ),
                    }
                )
                training_provenance.append(
                    {
                        "family_id": family["family_id"],
                        "available_msp_enriched_molecules": 0,
                        "selected_molecules": 0,
                        "selected_molecule_ids": [],
                        "cohort_role": (
                            "unscorable_no_sequence_opportunity_support"
                        ),
                    }
                )
                continue
            selected, available = _select_family_training_reads(
                evidence,
                family_for_selection,
                discovery_windows,
                maximum=int(discovery_config["maximum_discovery_molecules"]),
                seed=str(discovery_config["seed"]),
                eligible_reads=eligible_reads,
            )
            if not selected:
                raise ValueError(
                    f"family {family['family_id']} has no raw MSP-enriched evidence"
                )
            fit_payloads.append((family, selected, chemistry))
            training_provenance.append(
                {
                    "family_id": family["family_id"],
                    "available_msp_enriched_molecules": available,
                    "selected_molecules": len(selected),
                    "selected_molecule_ids": [
                        list(read.molecule_id) for read in selected
                    ],
                    "cohort_role": "geometry_and_nuisance_fit_only_not_occupancy",
                }
            )
        selection_elapsed = time.perf_counter() - selection_started
        if not fit_payloads:
            raise ValueError(
                "no catalog family has sufficient sequence opportunities to fit"
            )
        models = []
        fit_started = time.perf_counter()
        if workers == 1 or len(fit_payloads) <= 1:
            model_iterator = map(_fit_family_worker, fit_payloads)
            executor = None
        else:
            executor = ProcessPoolExecutor(
                max_workers=min(workers, len(fit_payloads)),
                mp_context=_MP_CONTEXT,
            )
            model_iterator = executor.map(_fit_family_worker, fit_payloads)
        try:
            for index, model in enumerate(model_iterator, start=1):
                models.append(model)
                if _progress_checkpoint(index, len(fit_payloads)):
                    print(
                        f"family fits {index}/{len(fit_payloads)}: "
                        f"{model['family_id']} eligible={model['eligible_molecules']}",
                        file=sys.stderr,
                        flush=True,
                    )
        finally:
            if executor is not None:
                executor.shutdown()
        fit_elapsed = time.perf_counter() - fit_started

    score_started = time.perf_counter()
    print(
        f"full-cohort likelihood backend requested={args.likelihood_backend}",
        file=sys.stderr,
        flush=True,
    )
    def score_progress(completed, total):
        if _progress_checkpoint(completed, total):
            print(
                f"joint full-cohort scoring chunks {completed}/{total} "
                f"families={len(models)}",
                file=sys.stderr,
                flush=True,
            )

    scores = list(
        score_boundary_families_on_unbiased_cohort(
            evidence,
            models,
            chunk_size=args.chunk_size,
            workers=workers,
            occupancy_pseudocount=args.occupancy_pseudocount,
            minimum_assignment_standardized_posterior=(
                args.minimum_assignment_standardized_posterior
            ),
            minimum_assignment_log_bayes_factor=(
                args.minimum_assignment_log_bayes_factor
            ),
            include_conditional_geometry=args.include_conditional_geometry,
            likelihood_backend=args.likelihood_backend,
            cuda_interval_chunk_size=args.cuda_interval_chunk_size,
            cuda_read_chunk_size=args.cuda_read_chunk_size,
            cuda_family_batch_span_bp=args.cuda_family_batch_span_bp,
            cuda_replay_guard_nats=args.cuda_replay_guard_nats,
            progress=score_progress,
        )
    )
    score_elapsed = time.perf_counter() - score_started

    output_dir.mkdir(parents=True)
    if args.compact_artifacts:
        with (output_dir / "models.jsonl").open("w") as handle:
            for model in models:
                handle.write(json.dumps(model, sort_keys=True) + "\n")
        with (output_dir / "scores.jsonl").open("w") as handle:
            for score in scores:
                handle.write(json.dumps(score, sort_keys=True) + "\n")
    else:
        models_dir = output_dir / "models"
        scores_dir = output_dir / "scores"
        models_dir.mkdir()
        scores_dir.mkdir()
        for model, score in zip(models, scores):
            (models_dir / f"{model['family_id']}.json").write_text(
                json.dumps(model, indent=2, sort_keys=True) + "\n"
            )
            (scores_dir / f"{model['family_id']}.json").write_text(
                json.dumps(score, indent=2, sort_keys=True) + "\n"
            )
    score_rows = [
        {
            "family_id": score["family_id"],
            "family_slot": catalog["family_slots"][score["family_id"]],
            "eligible_molecules": score["eligible_molecules"],
            "fitted_family_occupancy": score["fitted_family_occupancy"],
            "fitted_family_effective_support": score.get(
                "fitted_family_effective_support", 0.0
            ),
            "standardized_family_effective_support_equal_prior": score.get(
                "standardized_family_effective_support_equal_prior", 0.0
            ),
            "median_family_vs_null_log_bayes_factor": score.get(
                "median_family_vs_null_log_bayes_factor"
            ),
            "assigned_molecules": len(score["molecules"]),
        }
        for score in scores
    ]
    _write_tsv(output_dir / "family_scores.tsv", score_rows, tuple(score_rows[0]))
    assignment_rows = [
        {
            "family_id": score["family_id"],
            "family_slot": catalog["family_slots"][score["family_id"]],
            "library_id": record["molecule_id"][0],
            "read_name": record["molecule_id"][1],
            "strand": record["molecule_id"][2],
            "fitted_family_posterior": record["fitted_family_posterior"],
            "standardized_family_posterior_equal_prior": record[
                "standardized_family_posterior_equal_prior"
            ],
            "family_vs_null_log_bayes_factor": record[
                "family_vs_null_log_bayes_factor"
            ],
            "map_start": (
                record["conditional_map_interval"][0]
                if "conditional_map_interval" in record else ""
            ),
            "map_end": (
                record["conditional_map_interval"][1]
                if "conditional_map_interval" in record else ""
            ),
            "map_interval_probability": record.get(
                "conditional_map_interval_probability", ""
            ),
        }
        for score in scores
        for record in score["molecules"]
    ]
    assignment_fields = (
        "family_id", "family_slot", "library_id", "read_name", "strand",
        "fitted_family_posterior", "standardized_family_posterior_equal_prior",
        "family_vs_null_log_bayes_factor", "map_start", "map_end",
        "map_interval_probability",
    )
    _write_tsv(output_dir / "molecule_family_scores.tsv", assignment_rows, assignment_fields)
    serialized_training_provenance = training_provenance
    if args.compact_artifacts:
        training_rows = [
            {
                "family_id": record["family_id"],
                "library_id": molecule_id[0],
                "read_name": molecule_id[1],
                "strand": molecule_id[2],
            }
            for record in training_provenance
            for molecule_id in record.get("selected_molecule_ids", ())
        ]
        _write_tsv(
            output_dir / "training_molecules.tsv",
            training_rows,
            ("family_id", "library_id", "read_name", "strand"),
        )
        serialized_training_provenance = [
            {
                key: value
                for key, value in record.items()
                if key != "selected_molecule_ids"
            }
            for record in training_provenance
        ]
    manifest = {
        "schema": "fiberhmm.targeted_family_quantification.v1",
        "status": "complete_unbiased_independent_family_screen",
        "fiberhmm_version": __version__,
        "command_line": command_line,
        "catalog": str(catalog_path),
        "catalog_sha256": _sha256(catalog_path),
        "inputs": [str(path) for path in inputs],
        "chemistry": chemistry,
        "region": [contig, locus_start, locus_end],
        "full_cohort_molecules": len(evidence),
        "load_diagnostics": load_diagnostics,
        "amplification_collapse": collapse,
        "efficiency_calibration": efficiency,
        "model_fit_config": model_fit_config,
        "training_cohorts": serialized_training_provenance,
        "reused_model_provenance": reused_model_provenance,
        "family_count": len(models),
        "catalog_family_count": len(families),
        "skipped_unscorable_family_count": len(skipped_unscorable_families),
        "skipped_unscorable_families": skipped_unscorable_families,
        "timing_seconds": {
            "full_cohort_load_and_calibration": load_elapsed,
            "training_cohort_index_and_selection": selection_elapsed,
            "parallel_family_fit": fit_elapsed,
            "joint_chunked_full_cohort_score": score_elapsed,
        },
        "parallelism": {
            "workers": workers,
            "chunk_size": args.chunk_size,
            "family_fit": "independent_family_process_tasks",
            "full_scoring": (
                "sparse_genomic_locality_batches_one_cuda_owner"
                if scores[0]["likelihood_backend"]["resolved"] == "cuda"
                else "one_deterministic_read_stream_all_families_per_worker"
            ),
        },
        "likelihood_backend": scores[0]["likelihood_backend"],
        "contracts": {
            "discovery_cohort_role": "geometry_only_msp_enriched",
            "quantification_cohort": "complete_unbiased_independent_molecules",
            "discovery_mixture_weights_used_for_quantification": False,
            "ordinary_tf_nuc_calls_mutated": False,
            "bam_materialized": False,
            "assignment_posterior": "standardized_equal_prior_family_vs_null",
            "conditional_geometry_materialized": bool(
                args.include_conditional_geometry
            ),
            "next_phase": "joint_family_and_nucleosome_hybrid_application",
        },
        "artifacts": {
            "models": "models.jsonl" if args.compact_artifacts else "models/",
            "scores": "scores.jsonl" if args.compact_artifacts else "scores/",
            "family_scores": "family_scores.tsv",
            "molecule_family_scores": "molecule_family_scores.tsv",
            "training_molecules": (
                "training_molecules.tsv" if args.compact_artifacts else None
            ),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "families": len(models), "molecules": len(evidence)}, indent=2, sort_keys=True))
    return 0


def _discover(args, command_line: str) -> int:
    inputs = tuple(Path(value).expanduser().resolve() for value in args.input)
    if len(set(inputs)) != len(inputs):
        raise ValueError("the same resolved BAM was supplied more than once")
    for path in inputs:
        if not path.is_file():
            raise ValueError(f"input BAM does not exist: {path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    chemistry, chemistry_records = _resolve_chemistry(inputs, args.chemistry)
    workers = args.cores or max(1, os.cpu_count() or 1)

    with pysam.AlignmentFile(str(inputs[0]), "rb", check_sq=False) as handle:
        reference_lengths = {
            str(name): int(length) for name, length in zip(handle.references, handle.lengths)
        }
    region = parse_reference_region(args.region, reference_lengths)
    for path in inputs[1:]:
        with pysam.AlignmentFile(str(path), "rb", check_sq=False) as handle:
            if region.contig not in handle.references:
                raise ValueError(f"{region.contig!r} is absent from {path}")
            if region.end > int(handle.get_reference_length(region.contig)):
                raise ValueError(f"region exceeds {region.contig} in {path}")

    config = TargetedFamilyDiscoveryConfig(
        core_size=args.window_size,
        halo_size=args.halo_size,
        minimum_nfr_length=args.minimum_nfr_length,
        minimum_informative_molecules=args.minimum_informative_molecules,
        minimum_informative_fraction=args.minimum_informative_fraction,
        maximum_discovery_molecules=args.discovery_reads,
        minimum_family_support=args.minimum_family_support,
        minimum_family_fraction=args.minimum_family_fraction,
        seed=args.seed,
    )
    evidence_start = max(0, region.start - config.halo_size)
    evidence_end = min(reference_lengths[region.contig], region.end + config.halo_size)
    evidence_region = f"{region.contig}:{evidence_start}-{evidence_end}"

    loaded = []
    input_records = []
    batch_input_provenance = getattr(args, "_batch_input_provenance", {})
    for input_index, path in enumerate(inputs):
        current = load_footprint_molecules_from_bam(
            path,
            regions=[evidence_region],
            min_mapq=args.min_mapq,
            invalid_stratum_tag_policy=(
                "alignment" if chemistry in {"ddda", "dddb"} else "error"
            ),
        )
        molecules = current.molecules
        collapse = {"enabled": False, "retained_molecules": len(molecules)}
        if chemistry in {"ddda", "dddb"} and not args.daf_already_deduplicated:
            molecules, collapse = collapse_daf_amplification_families(
                path,
                molecules,
                contig=region.contig,
                start=evidence_start,
                end=evidence_end,
                min_mapq=args.min_mapq,
                minimum_jaccard=args.daf_minimum_jaccard,
                minimum_deaminations=args.daf_minimum_deaminations,
            )
        elif chemistry not in {"ddda", "dddb"}:
            molecules = tuple(replace(molecule, stratum=".") for molecule in molecules)
        prefix = f"input{input_index:03d}_{hashlib.sha256(str(path).encode()).hexdigest()[:8]}"
        loaded.extend(_prefix_molecules(molecules, prefix))
        precomputed = batch_input_provenance.get(str(path), {})
        input_records.append(
            {
                "path": str(path),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": (
                    None
                    if args.skip_input_hash
                    else precomputed.get("sha256") or _bam_sha256(path)
                ),
                "sha256_status": (
                    "skipped_explicit_fast_regional_provenance"
                    if args.skip_input_hash
                    else "complete_file"
                ),
                "fast_provenance": precomputed.get("fast_provenance")
                or _fast_bam_provenance(path),
                "annotation_loader": current.diagnostics.as_dict(),
                "amplification_collapse": collapse,
                "retained_independent_molecules": len(molecules),
            }
        )
        print(
            f"loaded {input_index + 1}/{len(inputs)} {path.name}: "
            f"{len(molecules)} independent molecules",
            file=sys.stderr,
            flush=True,
        )
    loaded.sort(key=lambda molecule: molecule.molecule_id)
    if not loaded:
        raise ValueError("no eligible independent molecules")

    def progress(completed, total, result):
        if _progress_checkpoint(completed, total):
            print(
                f"discovery windows {completed}/{total}: {result.window.label}; "
                f"selected={len(result.selected_molecule_ids)} "
                f"families={len(result.families)} "
                f"elapsed={result.elapsed_seconds:.2f}s",
                file=sys.stderr,
                flush=True,
            )

    discovery = discover_targeted_families(
        tuple(loaded),
        contig=region.contig,
        locus_start=region.start,
        locus_end=region.end,
        chemistry=chemistry,
        config=config,
        workers=workers,
        progress=progress,
    )
    separation = 2 * CHEMISTRY_PROFILES[chemistry].maximum_boundary_delta
    slots = allocate_repeating_family_ids(
        (
            TFFamilyInterval(
                family_key=family.family_id,
                contig=family.contig,
                start=family.start,
                end=family.end,
            )
            for family in discovery.families
        ),
        separation_bp=separation,
    )

    output_dir.mkdir(parents=True)
    family_rows = [
        {
            "family_id": family.family_id,
            "family_slot": slots[family.family_id],
            "contig": family.contig,
            "start": family.start,
            "end": family.end,
            "width": family.width,
            "seed_intervals": json.dumps(family.seed_intervals, separators=(",", ":")),
            "member_site_ids": ",".join(family.member_site_ids),
            "discovery_support_molecules": family.discovery_support_molecules,
            "discovery_denominator_molecules": family.discovery_denominator_molecules,
            "source_window_ordinals": ",".join(map(str, family.source_window_ordinals)),
            "cohort_role": "geometry_only_msp_enriched",
        }
        for family in discovery.families
    ]
    family_fields = (
        "family_id", "family_slot", "contig", "start", "end", "width",
        "seed_intervals", "member_site_ids", "discovery_support_molecules",
        "discovery_denominator_molecules", "source_window_ordinals", "cohort_role",
    )
    _write_tsv(output_dir / "families.tsv", family_rows, family_fields)
    window_rows = [
        {
            "ordinal": window.ordinal,
            "contig": window.contig,
            "core_start": window.core_start,
            "core_end": window.core_end,
            "halo_start": window.halo_start,
            "halo_end": window.halo_end,
            "fully_mapped_molecules": window.fully_mapped_molecules,
            "long_msp_molecules": window.long_msp_molecules,
            "required_long_msp_molecules": window.required_long_msp_molecules,
            "informative": int(window.informative),
        }
        for window in discovery.windows
    ]
    _write_tsv(output_dir / "windows.tsv", window_rows, tuple(window_rows[0]))
    selected_rows = [
        {
            "window_ordinal": result.window.ordinal,
            "molecule_id": molecule_id,
            "cohort_role": "geometry_only_msp_enriched",
        }
        for result in discovery.window_results
        for molecule_id in result.selected_molecule_ids
    ]
    _write_tsv(
        output_dir / "discovery_molecules.tsv",
        selected_rows,
        ("window_ordinal", "molecule_id", "cohort_role"),
    )
    independent_rows = []
    for molecule in loaded:
        prefix, separator, read_name = molecule.molecule_id.partition("\x1f")
        if not separator or not prefix.startswith("input"):
            raise ValueError("internal prefixed molecule identity is malformed")
        independent_rows.append(
            {
                "input_index": int(prefix[5:8]),
                "read_name": read_name,
                "stratum": molecule.stratum,
                "prefixed_molecule_id": molecule.molecule_id,
            }
        )
    _write_tsv(
        output_dir / "independent_molecules.tsv",
        independent_rows,
        ("input_index", "read_name", "stratum", "prefixed_molecule_id"),
    )
    manifest = discovery.as_dict()
    manifest.update(
        {
            "fiberhmm_version": __version__,
            "command_line": command_line,
            "inputs": input_records,
            "chemistry_resolution": chemistry_records,
            "loaded_evidence_region": [region.contig, evidence_start, evidence_end],
            "family_slots": slots,
            "family_slot_separation_bp": separation,
            "artifacts": {
                "families": "families.tsv",
                "windows": "windows.tsv",
                "discovery_molecules": "discovery_molecules.tsv",
                "independent_molecules": "independent_molecules.tsv",
            },
        }
    )
    (output_dir / "catalog.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output_dir),
                "input_molecules": len(loaded),
                "informative_windows": sum(window.informative for window in discovery.windows),
                "families": len(discovery.families),
                "workers": workers,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fiberhmm-targeted-families",
        description=(
            "Discover chemistry-aware footprint families in short targeted windows. "
            "Then quantify frozen geometry on the complete unbiased locus cohort."
        ),
    )
    add_version_args(parser)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    discover = subparsers.add_parser(
        "discover",
        help="Build a frozen geometry catalog from recurrent long-MSP windows",
    )
    discover.add_argument("-i", "--input", action="append", required=True)
    discover.add_argument("--region", required=True, help="Zero-based CONTIG:START-END")
    discover.add_argument("-o", "--output-dir", required=True)
    discover.add_argument("--chemistry", choices=tuple(CHEMISTRY_PROFILES))
    discover.add_argument("-c", "--cores", type=int, default=1, help="Workers; 0=all CPUs")
    discover.add_argument("--window-size", type=int, default=1000)
    discover.add_argument("--halo-size", type=int, default=200)
    discover.add_argument("--minimum-nfr-length", type=int, default=150)
    discover.add_argument("--minimum-informative-molecules", type=int, default=3)
    discover.add_argument("--minimum-informative-fraction", type=float, default=0.01)
    discover.add_argument("--discovery-reads", type=int, default=500)
    discover.add_argument("--minimum-family-support", type=int, default=3)
    discover.add_argument("--minimum-family-fraction", type=float, default=0.05)
    discover.add_argument("--min-mapq", type=int, default=20)
    discover.add_argument("--seed", default="fiberhmm-targeted-family-discovery-v1")
    discover.add_argument("--daf-minimum-jaccard", type=float, default=0.95)
    discover.add_argument("--daf-minimum-deaminations", type=int, default=10)
    discover.add_argument(
        "--skip-input-hash",
        action="store_true",
        help=(
            "Skip complete BAM SHA-256 reads and record size/mtime plus BAM-header "
            "and index fingerprints; intended for indexed regional scans of very "
            "large local BAMs"
        ),
    )
    discover.add_argument(
        "--daf-already-deduplicated",
        action="store_true",
        help="Trust each eligible DAF record as an independent molecule",
    )
    quantify = subparsers.add_parser(
        "quantify",
        help="Fit boundary geometry and score every independent locus molecule",
    )
    quantify.add_argument("--catalog", required=True)
    quantify.add_argument("-i", "--input", action="append", required=True)
    quantify.add_argument("-o", "--output-dir", required=True)
    quantify.add_argument(
        "--reuse-models-from",
        help=(
            "Reuse the strictly provenance-matched frozen models from a complete "
            "quantification directory; useful for rescoring thresholds/backends"
        ),
    )
    quantify.add_argument("-c", "--cores", type=int, default=1, help="Workers; 0=all CPUs")
    quantify.add_argument("--chunk-size", type=int, default=512)
    quantify.add_argument(
        "--likelihood-backend",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help=(
            "Family/null likelihood backend. CPU is the unchanged reference; "
            "auto uses CUDA only when an accessible PyTorch CUDA device exists."
        ),
    )
    quantify.add_argument(
        "--cuda-interval-chunk-size",
        type=int,
        default=1024,
        help="Candidate intervals evaluated per resident CUDA tile",
    )
    quantify.add_argument(
        "--cuda-read-chunk-size",
        type=int,
        default=0,
        help=(
            "Molecule rows per resident CUDA batch; 0 chooses a VRAM-aware size "
            "targeting 70%% of currently free device memory"
        ),
    )
    quantify.add_argument(
        "--cuda-family-batch-span-bp",
        type=int,
        default=25000,
        help=(
            "Maximum genomic span of one sparse CUDA family/read batch; nearby "
            "families share resident fibers without scoring distant pairs"
        ),
    )
    quantify.add_argument(
        "--cuda-replay-guard-nats",
        type=float,
        default=1e-8,
        help="Replay CUDA assignments this close to a decision threshold on CPU",
    )
    quantify.add_argument("--min-mapq", type=int, default=20)
    quantify.add_argument("--tf-layer", choices=("tf", "tf_sr"), default="tf_sr")
    quantify.add_argument("--nuc-layer", choices=("nuc", "nuc_sr"), default="nuc_sr")
    quantify.add_argument("--occupancy-pseudocount", type=float, default=0.5)
    quantify.add_argument(
        "--minimum-assignment-standardized-posterior",
        "--minimum-assignment-posterior",
        dest="minimum_assignment_standardized_posterior",
        type=float,
        default=0.5,
        help=(
            "Equal-prior family-vs-null posterior threshold; the legacy shorter "
            "option name is retained as an alias"
        ),
    )
    quantify.add_argument(
        "--minimum-assignment-log-bayes-factor", type=float, default=0.0
    )
    quantify.add_argument(
        "--include-conditional-geometry",
        action="store_true",
        help=(
            "Materialize per-molecule MAP boundaries and credible envelopes; "
            "off by default because the exploratory screen does not require them"
        ),
    )
    quantify.add_argument("--skip-input-hash-check", action="store_true")
    quantify.add_argument(
        "--compact-artifacts",
        action="store_true",
        help=(
            "Write one models.jsonl and scores.jsonl instead of per-family JSON "
            "files; intended for large BED batches and supported by model reuse"
        ),
    )
    quantify.add_argument(
        "--efficiency-exclusion-bed",
        help=(
            "Exclude all intervals in this BED from cohort efficiency calibration; "
            "useful for boundary-invariant BED batch scans"
        ),
    )
    quantify.add_argument("--efficiency-exclusion-padding", type=int, default=0)
    quantify.add_argument(
        "--efficiency-evidence-scope",
        choices=("region", "full-alignment"),
        default="region",
        help=(
            "Opportunity scope used to estimate per-read chemistry efficiency; "
            "full-alignment is boundary-invariant for BED batches and is locally "
            "projected again before scoring"
        ),
    )
    from fiberhmm.cli.targeted_family_batch import add_batch_parser

    add_batch_parser(subparsers)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    values = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(values)
    command_line = " ".join(
        ["fiberhmm-targeted-families"] + [shlex.quote(str(value)) for value in values]
    )
    try:
        if args.subcommand == "discover":
            return _discover(args, command_line)
        if args.subcommand == "quantify":
            return _quantify(args, command_line)
        if args.subcommand == "batch":
            from fiberhmm.cli.targeted_family_batch import (
                run_targeted_family_batch,
            )

            return run_targeted_family_batch(args, command_line)
        raise ValueError(f"unsupported subcommand: {args.subcommand}")
    except (OSError, ValueError, BamFootprintInputError, pysam.utils.SamtoolsError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
