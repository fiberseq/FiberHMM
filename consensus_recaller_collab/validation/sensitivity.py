"""Read-only sensitivity audits for amplified DAF molecule collapse."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Mapping, Optional, Sequence

from consensus_recaller_collab.validation import VALIDATION_VERSION
from consensus_recaller_collab.validation.evidence import (
    EvidenceLoader,
    _canonical_sha256,
    _collapse_molecule_families,
    _input_file_metadata,
)


def audit_molecule_collapse(
    manifest: Mapping[str, object],
    *,
    locus_id: str,
    sample_id: str,
    jaccards: Sequence[float] = (0.90, 0.95, 0.98),
    min_mapq: int = 20,
    max_reads: int = 0,
    manifest_base: Optional[str] = None,
) -> dict:
    loci = {
        str(locus["locus_id"]): locus for locus in manifest.get("loci", [])
    }
    samples = {
        str(sample["sample_id"]): sample for sample in manifest.get("samples", [])
    }
    if locus_id not in loci:
        raise ValueError(f"unknown locus: {locus_id}")
    if sample_id not in samples:
        raise ValueError(f"unknown sample: {sample_id}")
    locus = loci[locus_id]
    sample = samples[sample_id]
    if sample.get("cohort_id") != locus.get("cohort_id"):
        raise ValueError("sample and locus cohorts differ")
    if sample.get("availability", "available") != "available":
        raise ValueError("sample is not locally available")
    restricted_loci = sample.get("restrict_loci")
    if restricted_loci is not None and locus_id not in restricted_loci:
        raise ValueError("sample is restricted away from this locus")
    restricted_groups = sample.get("restrict_comparison_groups")
    if (
        restricted_groups is not None
        and locus.get("comparison_group") not in restricted_groups
    ):
        raise ValueError("sample is restricted away from this comparison group")

    collapse = sample.get("molecule_collapse")
    if not isinstance(collapse, Mapping) or not bool(collapse.get("enabled", False)):
        raise ValueError("sample has no enabled molecule-collapse policy")
    thresholds = sorted({float(value) for value in jaccards})
    if not thresholds or any(not 0.0 < value <= 1.0 for value in thresholds):
        raise ValueError("Jaccard thresholds must lie in (0, 1]")

    raw_sample = copy.deepcopy(sample)
    raw_sample["molecule_collapse"] = {"enabled": False}
    base_dir = Path(manifest_base).resolve() if manifest_base else None
    loader = EvidenceLoader(
        min_mapq=min_mapq, max_reads=max_reads, base_dir=base_dir
    )
    reads = loader.load(raw_sample, locus)
    rows = []
    for threshold in thresholds:
        _, diagnostics = _collapse_molecule_families(
            reads,
            min_jaccard=threshold,
            min_deam=int(collapse.get("min_deam", 10)),
            ignore_strand=bool(collapse.get("ignore_strand", False)),
            num_hashes=int(collapse.get("num_hashes", 32)),
            bands=int(collapse.get("bands", 8)),
            seed=int(collapse.get("seed", 7)),
        )
        rows.append({"min_jaccard": threshold, **diagnostics})
    primary = float(collapse.get("min_jaccard", 0.95))
    primary_row = next(
        (row for row in rows if row["min_jaccard"] == primary), None
    )
    if primary_row is None:
        _, diagnostics = _collapse_molecule_families(
            reads,
            min_jaccard=primary,
            min_deam=int(collapse.get("min_deam", 10)),
            ignore_strand=bool(collapse.get("ignore_strand", False)),
            num_hashes=int(collapse.get("num_hashes", 32)),
            bands=int(collapse.get("bands", 8)),
            seed=int(collapse.get("seed", 7)),
        )
        primary_row = {"min_jaccard": primary, **diagnostics}
        rows.append(primary_row)
        rows.sort(key=lambda row: row["min_jaccard"])
    primary_count = int(primary_row["analyzed_molecules"])
    for row in rows:
        row["molecule_count_delta_from_primary"] = (
            int(row["analyzed_molecules"]) - primary_count
        )
        row["molecule_count_fractional_delta_from_primary"] = (
            (int(row["analyzed_molecules"]) - primary_count) / primary_count
            if primary_count else 0.0
        )
    return {
        "schema_version": 1,
        "producer": {
            "name": "fiberhmm-consensus-validation",
            "version": VALIDATION_VERSION,
            "manifest_sha256": _canonical_sha256(manifest),
        },
        "audit": "amplified_daf_molecule_collapse_sensitivity",
        "locus_id": locus_id,
        "sample_id": sample_id,
        "region": locus["region"],
        "primary_min_jaccard": primary,
        "min_deam": int(collapse.get("min_deam", 10)),
        "raw_reads": len(reads),
        "input_files": _input_file_metadata([sample], base_dir=base_dir),
        "thresholds": rows,
    }
