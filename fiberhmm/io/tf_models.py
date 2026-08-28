"""Portable exports for data-derived footprint population models.

The population files describe the consensus model catalog.  The FiberBrowser
sidecar preserves each assigned observation at its original reference
coordinates and links every block back to its stable ``site_id``.  Site IDs are
data entities, not UI layers: the sidecar always exposes one fixed derived
layer named ``footprint_model_assignments``.

BED12 requires non-overlapping blocks within a row.  A read can legitimately
carry overlapping assignments to alternative TF geometry families, so this
writer greedily partitions a read's intervals into non-overlapping physical
rows.  The manifest declares the ``duplicate_class_rows`` extension, which
instructs compatible consumers to concatenate those rows for the same
read/class rather than overwriting one with another.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from heapq import heappop, heappush
from itertools import chain, groupby
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from fiberhmm import __version__
from fiberhmm.inference.tf_sites import (
    TFCallAssignment,
    TFPopulationSite,
    TFSiteCatalog,
)

FOOTPRINT_MODEL_CLASS_ID = "footprint_model_assignments"
FOOTPRINT_MODEL_SOURCE_LAYER = "tf"
FOOTPRINT_MODEL_RULE_ID = "fiberhmm_footprint_model_v1"
FOOTPRINT_MODEL_COLOR = "123,50,148"
FOOTPRINT_MODEL_COLOR_HEX = "#7B3294"

# Compatibility names for prototypes created before the public terminology was
# settled.  A footprint population model contains TF-binding hypotheses; it is
# not itself a prior or an assertion of TF protein identity.
TF_MODEL_CLASS_ID = FOOTPRINT_MODEL_CLASS_ID
TF_MODEL_SOURCE_LAYER = FOOTPRINT_MODEL_SOURCE_LAYER
TF_MODEL_RULE_ID = FOOTPRINT_MODEL_RULE_ID
TF_MODEL_COLOR = FOOTPRINT_MODEL_COLOR
TF_MODEL_COLOR_HEX = FOOTPRINT_MODEL_COLOR_HEX


POPULATION_TSV_COLUMNS = (
    "contig",
    "start",
    "end",
    "site_id",
    "locus_id",
    "locus_summit",
    "summit",
    "family_index",
    "start_mad",
    "end_mad",
    "assigned_call_count",
    "geometry_call_count",
    "cluster_support_molecules",
    "n_fully_mapped",
    "n_tf",
    "n_msp",
    "n_tf_msp",
    "n_tf_no_msp",
    "n_no_tf_msp",
    "n_no_tf_no_msp",
    "occupancy_overall",
    "occupancy_given_msp",
    "geometry_ready",
    "population_ready",
)

STRATA_TSV_COLUMNS = (
    "site_id",
    "locus_id",
    "contig",
    "start",
    "end",
    "family_index",
    "stratum",
    "cluster_support_molecules",
    "n_fully_mapped",
    "n_tf",
    "n_msp",
    "n_tf_msp",
    "n_tf_no_msp",
    "n_no_tf_msp",
    "n_no_tf_no_msp",
    "occupancy_overall",
    "occupancy_given_msp",
    "median_start",
    "median_end",
    "start_mad",
    "end_mad",
)

ASSIGNMENTS_TSV_COLUMNS = (
    "call_id",
    "molecule_id",
    "contig",
    "stratum",
    "start",
    "end",
    "site_id",
    "locus_id",
    "family_index",
    "used_for_geometry",
    "fully_maps_site",
    "msp_contains_site",
)


FOOTPRINT_MODEL_POPULATION_AUTOSQL = """table fiberhmm_footprint_model
"FiberHMM footprint population model and data-derived TF-binding hypotheses"
    (
    string chrom;        "Reference chromosome / contig"
    uint chromStart;     "Consensus family start (0-based)"
    uint chromEnd;       "Consensus family end (exclusive)"
    string name;         "Stable TF geometry-family site ID"
    uint score;          "Population support (unique molecules), capped at 1000"
    char[1] strand;      "Unstranded consensus model (.)"
    string locusId;      "Stable parent binding-locus ID"
    uint familyIndex;    "Deterministic display ordinal within the parent locus"
    uint locusSummit;    "Smoothed center-density summit of the parent locus"
    uint summit;         "Center-density summit of this geometry family"
    uint assignedCallCount; "Number of source calls assigned to this family"
    uint geometryCallCount; "Number of eligible calls used to learn geometry"
    uint clusterSupportMolecules; "Unique molecules supporting learned geometry"
    uint nFullyMapped;   "Molecules fully mapping the complete family interval"
    uint nTf;            "Fully mapped molecules with an assigned TF call"
    uint nMsp;           "Fully mapped molecules whose MSP contains the family"
    uint nTfMsp;         "Molecules with both an assigned TF and containing MSP"
    uint nTfNoMsp;       "Molecules with TF and no containing MSP"
    uint nNoTfMsp;       "Molecules with containing MSP and no TF"
    uint nNoTfNoMsp;     "Molecules with neither TF nor containing MSP"
    float occupancyOverall; "nTf / nFullyMapped, or -1 when unavailable"
    float occupancyGivenMsp; "nTfMsp / nMsp, or -1 when unavailable"
    float startMad;      "Median absolute deviation of learned starts"
    float endMad;        "Median absolute deviation of learned ends"
    uint geometryReady;  "1 when geometry support reaches the reporting threshold"
    uint populationReady; "1 when TF-supporting population reaches the reporting threshold"
    )
"""


FOOTPRINT_MODEL_FIBERLAYERS_AUTOSQL = """table fiberbrowser_fiberlayers
"FiberHMM per-read assignments to footprint-model binding hypotheses"
    (
    string chrom;        "Chromosome"
    uint chromStart;     "Start position"
    uint chromEnd;       "End position"
    string name;         "Source read / molecule ID"
    uint score;          "Categorical assignment confidence (1000)"
    char[1] strand;      "Strand (.)"
    uint thickStart;     "Start position (same as chromStart)"
    uint thickEnd;       "End position (same as chromEnd)"
    uint reserved;       "Class color encoded as R,G,B in the BED input"
    int blockCount;      "Number of non-overlapping assignment blocks in this lane"
    int[blockCount] blockSizes;  "Observed TF-call interval sizes"
    int[blockCount] chromStarts; "Observed TF-call starts relative to chromStart"
    string class_id;     "Stable derived-layer class ID"
    string source_layer; "Native layer supplying the observations"
    string rule_id;      "Stable footprint-model assignment rule ID"
    lstring blockQuality; "Comma-separated categorical values (255), aligned to blocks"
    lstring blockSiteIds; "Comma-separated stable site IDs, aligned to blocks"
    )
"""

TF_MODEL_POPULATION_AUTOSQL = FOOTPRINT_MODEL_POPULATION_AUTOSQL
TF_MODEL_FIBERLAYERS_AUTOSQL = FOOTPRINT_MODEL_FIBERLAYERS_AUTOSQL


@dataclass(frozen=True)
class FootprintModelBundlePaths:
    """Paths written by :func:`write_footprint_model_bundle`."""

    population_tsv: Path
    population_bed: Path
    population_autosql: Path
    strata_tsv: Path
    assignments_tsv: Path
    fiberlayers_bed: Path
    fiberlayers_autosql: Path
    fiberlayers_manifest: Path

    def as_dict(self) -> Mapping[str, str]:
        """Return JSON-friendly string paths keyed by artifact role."""

        return {
            "population_tsv": str(self.population_tsv),
            "population_bed": str(self.population_bed),
            "population_autosql": str(self.population_autosql),
            "strata_tsv": str(self.strata_tsv),
            "assignments_tsv": str(self.assignments_tsv),
            "fiberlayers_bed": str(self.fiberlayers_bed),
            "fiberlayers_autosql": str(self.fiberlayers_autosql),
            "fiberlayers_manifest": str(self.fiberlayers_manifest),
        }


@dataclass(frozen=True)
class FootprintModelBigBedPaths:
    """Indexed tracks added to an existing TF-model bundle."""

    population_bigbed: Path
    fiberlayers_bigbed: Path


TFModelBundlePaths = FootprintModelBundlePaths
TFModelBigBedPaths = FootprintModelBigBedPaths


@dataclass(frozen=True)
class _FiberLayerRow:
    contig: str
    stratum: str
    molecule_id: str
    lane_index: int
    blocks: Tuple[TFCallAssignment, ...]

    @property
    def start(self) -> int:
        return min(block.start for block in self.blocks)

    @property
    def end(self) -> int:
        return max(block.end for block in self.blocks)

    def bed_fields(self) -> Tuple[str, ...]:
        row_start = self.start
        return (
            _bed_field(self.contig),
            str(row_start),
            str(self.end),
            _bed_field(self.molecule_id),
            "1000",
            ".",
            str(row_start),
            str(self.end),
            TF_MODEL_COLOR,
            str(len(self.blocks)),
            _comma_list(block.end - block.start for block in self.blocks),
            _comma_list(block.start - row_start for block in self.blocks),
            TF_MODEL_CLASS_ID,
            TF_MODEL_SOURCE_LAYER,
            TF_MODEL_RULE_ID,
            ",".join("255" for _block in self.blocks),
            ",".join(_bed_field(block.site_id) for block in self.blocks),
        )


def _site_sort_key(site: TFPopulationSite) -> Tuple[object, ...]:
    return (
        site.contig,
        site.start,
        site.end,
        site.locus_summit,
        site.family_index,
        site.site_id,
    )


def _assignment_sort_key(assignment: TFCallAssignment) -> Tuple[object, ...]:
    return (
        assignment.contig,
        assignment.stratum,
        assignment.molecule_id,
        assignment.start,
        assignment.end,
        assignment.call_id,
    )


def _number(value: Optional[float], *, missing: str = ".") -> str:
    if value is None:
        return missing
    return format(float(value), ".12g")


def _flag(value: bool) -> str:
    return "1" if value else "0"


def _bed_field(value: object) -> str:
    text = str(value).replace("\t", "_").replace("\n", "_").replace("\r", "_")
    return text or "."


def _portable_identifier(value: object, *, label: str, forbid_comma: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    if any(character in value for character in "\t\n\r"):
        raise ValueError(f"{label} must not contain tab or newline characters")
    if forbid_comma and "," in value:
        raise ValueError(f"{label} must not contain commas")
    return value


def _validate_catalog(catalog: TFSiteCatalog) -> None:
    locus_by_id = {}
    for locus in catalog.loci:
        locus_id = _portable_identifier(locus.locus_id, label="locus_id")
        if locus_id in locus_by_id:
            raise ValueError(f"duplicate locus_id in catalog: {locus_id}")
        _portable_identifier(locus.contig, label="locus contig")
        locus_by_id[locus_id] = locus

    site_by_id = {}
    sites_by_locus = {}
    for site in catalog.sites:
        site_id = _portable_identifier(site.site_id, label="site_id", forbid_comma=True)
        if site_id in site_by_id:
            raise ValueError(f"duplicate site_id in catalog: {site_id}")
        _portable_identifier(site.contig, label=f"contig for {site_id}")
        locus_id = _portable_identifier(site.locus_id, label=f"locus_id for {site_id}")
        locus = locus_by_id.get(locus_id)
        if locus is None:
            raise ValueError(f"site {site_id} references missing locus_id {locus_id}")
        if (site.contig, site.locus_summit) != (locus.contig, locus.summit):
            raise ValueError(f"site {site_id} conflicts with parent locus {locus_id}")
        site_by_id[site_id] = site
        sites_by_locus.setdefault(locus_id, []).append(site)

    for locus_id, locus in locus_by_id.items():
        ordered = sorted(sites_by_locus.get(locus_id, ()), key=lambda site: site.family_index)
        observed_site_ids = tuple(site.site_id for site in ordered)
        if observed_site_ids != locus.family_site_ids:
            raise ValueError(f"family_site_ids disagree for locus {locus_id}")

    for assignment in catalog.assignments:
        _portable_identifier(assignment.call_id, label="assignment call_id")
        _portable_identifier(assignment.molecule_id, label="assignment molecule_id")
        _portable_identifier(assignment.contig, label="assignment contig")
        _portable_identifier(assignment.stratum, label="assignment stratum")
        if assignment.site_id is None:
            continue
        site_id = _portable_identifier(
            assignment.site_id,
            label="assignment site_id",
            forbid_comma=True,
        )
        site = site_by_id.get(site_id)
        if site is None:
            raise ValueError(f"assignment references missing site_id {site_id}")
        if assignment.contig != site.contig:
            raise ValueError(f"assignment contig conflicts with site_id {site_id}")


def _tsv_field(value: object) -> str:
    return _bed_field(value)


def _comma_list(values: Iterable[int]) -> str:
    return ",".join(str(int(value)) for value in values) + ","


def _population_tsv_fields(site: TFPopulationSite) -> Tuple[str, ...]:
    return (
        _tsv_field(site.contig),
        str(site.start),
        str(site.end),
        _tsv_field(site.site_id),
        _tsv_field(site.locus_id),
        str(site.locus_summit),
        str(site.summit),
        str(site.family_index),
        _number(site.start_mad),
        _number(site.end_mad),
        str(site.assigned_call_count),
        str(site.geometry_call_count),
        str(site.cluster_support_molecules),
        str(site.n_fully_mapped),
        str(site.n_tf),
        str(site.n_msp),
        str(site.n_tf_msp),
        str(site.n_tf_no_msp),
        str(site.n_no_tf_msp),
        str(site.n_no_tf_no_msp),
        _number(site.occupancy_overall),
        _number(site.occupancy_given_msp),
        _flag(site.geometry_ready),
        _flag(site.population_ready),
    )


def _population_bed_fields(site: TFPopulationSite) -> Tuple[str, ...]:
    # The standard BED score ranks recurrence support, not occupancy.  A rare
    # but repeatedly observed population must outrank a singleton with a small
    # denominator.  The exact uncapped count remains in populationSupport.
    score = min(1000, site.population_support)
    return (
        _bed_field(site.contig),
        str(site.start),
        str(site.end),
        _bed_field(site.site_id),
        str(score),
        ".",
        _bed_field(site.locus_id),
        str(site.family_index),
        str(site.locus_summit),
        str(site.summit),
        str(site.assigned_call_count),
        str(site.geometry_call_count),
        str(site.cluster_support_molecules),
        str(site.n_fully_mapped),
        str(site.n_tf),
        str(site.n_msp),
        str(site.n_tf_msp),
        str(site.n_tf_no_msp),
        str(site.n_no_tf_msp),
        str(site.n_no_tf_no_msp),
        _number(site.occupancy_overall, missing="-1"),
        _number(site.occupancy_given_msp, missing="-1"),
        _number(site.start_mad),
        _number(site.end_mad),
        _flag(site.geometry_ready),
        _flag(site.population_ready),
    )


def _stratum_fields(site: TFPopulationSite, summary: object) -> Tuple[str, ...]:
    n_fully_mapped = int(getattr(summary, "n_fully_mapped"))
    n_tf = int(getattr(summary, "n_tf"))
    n_msp = int(getattr(summary, "n_msp"))
    n_tf_msp = int(getattr(summary, "n_tf_msp"))
    n_tf_no_msp = n_tf - n_tf_msp
    n_no_tf_msp = n_msp - n_tf_msp
    n_no_tf_no_msp = n_fully_mapped - n_tf - n_msp + n_tf_msp
    occupancy_overall = None if n_fully_mapped == 0 else n_tf / n_fully_mapped
    occupancy_given_msp = None if n_msp == 0 else n_tf_msp / n_msp
    return (
        _tsv_field(site.site_id),
        _tsv_field(site.locus_id),
        _tsv_field(site.contig),
        str(site.start),
        str(site.end),
        str(site.family_index),
        _tsv_field(getattr(summary, "stratum")),
        str(getattr(summary, "cluster_support_molecules")),
        str(n_fully_mapped),
        str(n_tf),
        str(n_msp),
        str(n_tf_msp),
        str(n_tf_no_msp),
        str(n_no_tf_msp),
        str(n_no_tf_no_msp),
        _number(occupancy_overall),
        _number(occupancy_given_msp),
        str(getattr(summary, "median_start")),
        str(getattr(summary, "median_end")),
        _number(float(getattr(summary, "start_mad"))),
        _number(float(getattr(summary, "end_mad"))),
    )


def _assignment_fields(
    assignment: TFCallAssignment,
    sites_by_id: Mapping[str, TFPopulationSite],
) -> Tuple[str, ...]:
    site = sites_by_id.get(assignment.site_id) if assignment.site_id is not None else None
    return (
        _tsv_field(assignment.call_id),
        _tsv_field(assignment.molecule_id),
        _tsv_field(assignment.contig),
        _tsv_field(assignment.stratum),
        str(assignment.start),
        str(assignment.end),
        "." if assignment.site_id is None else _tsv_field(assignment.site_id),
        "." if site is None else _tsv_field(site.locus_id),
        "." if site is None else str(site.family_index),
        _flag(assignment.used_for_geometry),
        _flag(assignment.fully_maps_site),
        _flag(assignment.msp_contains_site),
    )


def _lane_partition(
    blocks: Sequence[TFCallAssignment],
) -> Tuple[Tuple[TFCallAssignment, ...], ...]:
    """Partition intervals into stable non-overlapping lanes in O(n log n)."""

    lanes: List[List[TFCallAssignment]] = []
    active: List[Tuple[int, int]] = []
    available: List[int] = []
    for block in sorted(
        blocks,
        key=lambda value: (value.start, value.end, value.site_id, value.call_id),
    ):
        while active and active[0][0] <= block.start:
            _lane_end, lane_index = heappop(active)
            heappush(available, lane_index)
        if available:
            selected = heappop(available)
            lanes[selected].append(block)
        else:
            selected = len(lanes)
            lanes.append([block])
        heappush(active, (block.end, selected))
    return tuple(tuple(lane) for lane in lanes)


def _fiberlayer_rows(assignments: Sequence[TFCallAssignment]) -> Tuple[_FiberLayerRow, ...]:
    rows = []
    assigned = (assignment for assignment in assignments if assignment.site_id is not None)
    for (contig, stratum, molecule_id), grouped_assignments in groupby(
        assigned,
        key=lambda assignment: (
            assignment.contig,
            assignment.stratum,
            assignment.molecule_id,
        ),
    ):
        for lane_index, lane in enumerate(_lane_partition(tuple(grouped_assignments))):
            rows.append(
                _FiberLayerRow(
                    contig=contig,
                    stratum=stratum,
                    molecule_id=molecule_id,
                    lane_index=lane_index,
                    blocks=lane,
                )
            )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.contig,
                row.start,
                row.end,
                row.molecule_id,
                row.stratum,
                row.lane_index,
            ),
        )
    )


def footprint_model_bundle_paths(
    output_prefix: Union[str, os.PathLike],
) -> FootprintModelBundlePaths:
    """Return the artifact paths for an output prefix without writing them."""

    prefix = str(Path(output_prefix))
    return FootprintModelBundlePaths(
        population_tsv=Path(prefix + ".footprint-model.tsv"),
        population_bed=Path(prefix + ".footprint-model.bed"),
        population_autosql=Path(prefix + ".footprint-model.as"),
        strata_tsv=Path(prefix + ".footprint-model.strata.tsv"),
        assignments_tsv=Path(prefix + ".footprint-model.assignments.tsv"),
        fiberlayers_bed=Path(prefix + ".footprint-model.fiberlayers.bed"),
        fiberlayers_autosql=Path(prefix + ".footprint-model.fiberlayers.as"),
        fiberlayers_manifest=Path(prefix + ".footprint-model.fiberlayers.json"),
    )


_paths_for_prefix = footprint_model_bundle_paths


def _write_rows(path: Path, rows: Iterable[Sequence[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for fields in rows:
            handle.write("\t".join(fields))
            handle.write("\n")


def _write_text(path: Path, value: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        if value and not value.endswith("\n"):
            handle.write("\n")


def _json_compatible(value: object) -> object:
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _source_dataset_manifest(
    source_dataset: Optional[Union[str, os.PathLike, Mapping[str, object]]],
    genome: Optional[str],
) -> Mapping[str, object]:
    if source_dataset is None:
        result = {}
    elif isinstance(source_dataset, Mapping):
        result = dict(_json_compatible(source_dataset))
    else:
        result = {"path": os.fspath(source_dataset)}
    if genome is not None:
        result["genome"] = genome
    else:
        result.setdefault("genome", None)
    return result


def _created_at_value(value: Optional[Union[str, datetime]]) -> Optional[str]:
    if value is None:
        # Keeping the absent provenance explicit makes repeated exports of the
        # same catalog byte-for-byte deterministic.
        return None
    if isinstance(value, str):
        return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _manifest(
    catalog: TFSiteCatalog,
    paths: FootprintModelBundlePaths,
    rows: Sequence[_FiberLayerRow],
    *,
    source_dataset: Optional[Union[str, os.PathLike, Mapping[str, object]]],
    genome: Optional[str],
    created_at: Optional[Union[str, datetime]],
    genomewide: bool,
    regions: Sequence[str],
) -> Mapping[str, object]:
    assigned_count = sum(assignment.site_id is not None for assignment in catalog.assignments)
    read_keys = {
        (assignment.contig, assignment.stratum, assignment.molecule_id)
        for assignment in catalog.assignments
        if assignment.site_id is not None
    }
    contigs = sorted(
        {site.contig for site in catalog.sites}
        | {assignment.contig for assignment in catalog.assignments}
    )
    return {
        "schema": "fiberlayers.v1",
        "extensions": {
            "block_site_ids": {
                "schema": "fiberbrowser.block_site_ids.v1",
                "field": "blockSiteIds",
                "alignment": ["blockSizes", "chromStarts", "blockQuality"],
            },
            "duplicate_class_rows": {
                "schema": "fiberbrowser.duplicate_class_rows.v1",
                "merge_key": ["chrom", "name", "class_id"],
                "semantics": "concatenate blocks from every matching row",
                "reason": "non_overlapping_bed12_lane_partition",
            },
        },
        "files": {
            "bed": paths.fiberlayers_bed.name,
            "bigbed": None,
            "autosql": paths.fiberlayers_autosql.name,
            "population_tsv": paths.population_tsv.name,
            "population_bed": paths.population_bed.name,
            "population_autosql": paths.population_autosql.name,
            "strata_tsv": paths.strata_tsv.name,
            "assignments_tsv": paths.assignments_tsv.name,
        },
        "source_dataset": _source_dataset_manifest(source_dataset, genome),
        "classes": {
            TF_MODEL_CLASS_ID: {
                "id": TF_MODEL_CLASS_ID,
                "label": "Footprint model assignments",
                "source_layer": TF_MODEL_SOURCE_LAYER,
                "kind": "footprint_class",
                "color": TF_MODEL_COLOR_HEX,
                "rules": {
                    "type": "footprint_model_assignment",
                    "rule_id": TF_MODEL_RULE_ID,
                    "site_catalog": paths.population_tsv.name,
                    "recommended_site_filter": {
                        "all": [
                            {"field": "geometry_ready", "equals": 1},
                            {"field": "population_ready", "equals": 1},
                        ],
                    },
                },
                "version": 1,
            }
        },
        "scope": {
            "regions": list(regions),
            "contigs": contigs,
            "genomewide": genomewide,
            "spanning": False,
            "coordinate_system": "0-based-half-open",
        },
        "model": {
            "schema": "fiberhmm.footprint_population_model.v1",
            "object": "footprint_population_model",
            "row_object": "tf_binding_hypothesis",
            "config": asdict(catalog.config),
            "diagnostics": asdict(catalog.diagnostics),
            "support": {
                "population": {
                    "meaning": "unique_fully_mapped_molecules_assigned",
                    "api_property": "population_support",
                    "tsv_field": "n_tf",
                    "bigbed_field": "nTf",
                },
                "geometry": {
                    "meaning": "unique_eligible_molecules_establishing_geometry",
                    "tsv_field": "cluster_support_molecules",
                    "bigbed_field": "clusterSupportMolecules",
                },
                "analysis_ready": {
                    "meaning": "geometry_ready_and_population_ready",
                    "api_property": "analysis_ready",
                    "expression": "geometry_ready == 1 and population_ready == 1",
                },
                "bed_score": {
                    "meaning": "population_support_capped_at_1000",
                    "formula": "min(population_support, 1000)",
                },
                "null_background": None,
            },
        },
        "provenance": {
            "created_at": _created_at_value(created_at),
            "exporter": "FiberHMM",
            "exporter_version": __version__,
            "rule_id": TF_MODEL_RULE_ID,
        },
        "statistics": {
            "loci": len(getattr(catalog, "loci", ())),
            "sites": len(catalog.sites),
            "geometry_ready_sites": sum(site.geometry_ready for site in catalog.sites),
            "population_ready_sites": sum(site.population_ready for site in catalog.sites),
            "analysis_ready_sites": sum(site.analysis_ready for site in catalog.sites),
            "assigned_calls": assigned_count,
            "unassigned_calls": len(catalog.assignments) - assigned_count,
            "classes": {
                TF_MODEL_CLASS_ID: {
                    "reads": len(read_keys),
                    "rows": len(rows),
                    "intervals": assigned_count,
                }
            },
        },
    }


def write_footprint_model_bundle(
    catalog: TFSiteCatalog,
    output_prefix: Union[str, os.PathLike],
    *,
    source_dataset: Optional[Union[str, os.PathLike, Mapping[str, object]]] = None,
    genome: Optional[str] = None,
    created_at: Optional[Union[str, datetime]] = None,
    genomewide: bool = False,
    regions: Sequence[str] = (),
) -> FootprintModelBundlePaths:
    """Write a footprint population model and FiberBrowser overlay bundle.

    ``output_prefix`` is a filename prefix rather than a directory.  The
    returned dataclass names all eight emitted artifacts.  ``created_at`` is
    left as JSON ``null`` by default so otherwise identical exports remain
    byte-for-byte reproducible; callers can provide a string or ``datetime``
    when wall-clock provenance is required. Scope is conservative by default:
    set ``genomewide=True`` only when the input cohort was scanned genome-wide,
    or provide the analyzed ``regions`` for a targeted catalog.
    """

    paths = footprint_model_bundle_paths(output_prefix)
    if not isinstance(genomewide, bool):
        raise ValueError("genomewide must be a boolean")
    if isinstance(regions, (str, bytes)):
        raise ValueError("regions must be a sequence of region strings")
    normalized_regions = tuple(
        _portable_identifier(region, label="scope region") for region in regions
    )
    _validate_catalog(catalog)

    sites = tuple(sorted(catalog.sites, key=_site_sort_key))
    assignments = tuple(sorted(catalog.assignments, key=_assignment_sort_key))
    sites_by_id = {site.site_id: site for site in sites}
    rows = _fiberlayer_rows(assignments)

    manifest = _manifest(
        catalog,
        paths,
        rows,
        source_dataset=source_dataset,
        genome=genome,
        created_at=created_at,
        genomewide=genomewide,
        regions=normalized_regions,
    )
    # Serialize before creating any artifact so malformed provenance cannot
    # leave a seven-file partial bundle.
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"

    paths.population_tsv.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(
        paths.population_tsv,
        chain((POPULATION_TSV_COLUMNS,), (_population_tsv_fields(site) for site in sites)),
    )
    _write_rows(paths.population_bed, (_population_bed_fields(site) for site in sites))
    _write_text(paths.population_autosql, FOOTPRINT_MODEL_POPULATION_AUTOSQL)

    stratum_rows = (
        _stratum_fields(site, summary)
        for site in sites
        for summary in sorted(site.strata, key=lambda value: value.stratum)
    )
    _write_rows(paths.strata_tsv, chain((STRATA_TSV_COLUMNS,), stratum_rows))

    _write_rows(
        paths.assignments_tsv,
        chain(
            (ASSIGNMENTS_TSV_COLUMNS,),
            (_assignment_fields(assignment, sites_by_id) for assignment in assignments),
        ),
    )
    _write_rows(paths.fiberlayers_bed, (row.bed_fields() for row in rows))
    _write_text(paths.fiberlayers_autosql, FOOTPRINT_MODEL_FIBERLAYERS_AUTOSQL)
    _write_text(paths.fiberlayers_manifest, manifest_text)

    return paths


def convert_footprint_model_bundle_to_bigbed(
    paths: FootprintModelBundlePaths,
    chrom_sizes: Union[str, os.PathLike],
    *,
    bed_to_bigbed: Optional[Union[str, os.PathLike]] = None,
) -> FootprintModelBigBedPaths:
    """Create indexed population and FiberBrowser tracks and update the manifest.

    The plain-text bundle remains the portable baseline. This optional step
    requires UCSC ``bedToBigBed`` and a two-column chromosome-sizes file. Both
    BigBeds are staged before replacement, so converter failure does not publish
    a half-updated manifest.
    """

    chrom_sizes_path = Path(chrom_sizes)
    if not chrom_sizes_path.is_file():
        raise ValueError(f"chromosome sizes file does not exist: {chrom_sizes_path}")
    executable = (
        os.fspath(bed_to_bigbed) if bed_to_bigbed is not None else shutil.which("bedToBigBed")
    )
    if not executable:
        raise RuntimeError("bedToBigBed was not found on PATH")

    population_bigbed = paths.population_bed.with_suffix(".bb")
    fiberlayers_bigbed = paths.fiberlayers_bed.with_suffix(".bb")
    population_temporary = population_bigbed.with_suffix(population_bigbed.suffix + ".tmp")
    fiberlayers_temporary = fiberlayers_bigbed.with_suffix(fiberlayers_bigbed.suffix + ".tmp")
    manifest_temporary = paths.fiberlayers_manifest.with_suffix(
        paths.fiberlayers_manifest.suffix + ".tmp"
    )

    with paths.fiberlayers_manifest.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not str(manifest.get("schema", "")).startswith("fiberlayers."):
        raise ValueError("not a FiberLayers manifest")

    conversions = (
        (
            "bed6+20",
            paths.population_autosql,
            paths.population_bed,
            population_temporary,
        ),
        (
            "bed12+5",
            paths.fiberlayers_autosql,
            paths.fiberlayers_bed,
            fiberlayers_temporary,
        ),
    )
    try:
        for bed_type, autosql, bed, output in conversions:
            output.unlink(missing_ok=True)
            command = [
                executable,
                "-tab",
                f"-type={bed_type}",
                f"-as={autosql}",
                os.fspath(bed),
                os.fspath(chrom_sizes_path),
                os.fspath(output),
            ]
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except OSError as error:
                raise RuntimeError(f"bedToBigBed failed to run: {error}") from error
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "unknown error").strip()
                raise RuntimeError(f"bedToBigBed failed for {bed.name}: {detail}")

        manifest.setdefault("files", {})["bigbed"] = fiberlayers_bigbed.name
        manifest["files"]["population_bigbed"] = population_bigbed.name
        manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        _write_text(manifest_temporary, manifest_text)
        os.replace(population_temporary, population_bigbed)
        os.replace(fiberlayers_temporary, fiberlayers_bigbed)
        os.replace(manifest_temporary, paths.fiberlayers_manifest)
    finally:
        population_temporary.unlink(missing_ok=True)
        fiberlayers_temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)

    return FootprintModelBigBedPaths(
        population_bigbed=population_bigbed,
        fiberlayers_bigbed=fiberlayers_bigbed,
    )


# Backward-compatible spellings for the unreleased prototype API.
write_tf_model_bundle = write_footprint_model_bundle
convert_tf_model_bundle_to_bigbed = convert_footprint_model_bundle_to_bigbed


__all__ = [
    "ASSIGNMENTS_TSV_COLUMNS",
    "FOOTPRINT_MODEL_CLASS_ID",
    "FOOTPRINT_MODEL_FIBERLAYERS_AUTOSQL",
    "FOOTPRINT_MODEL_POPULATION_AUTOSQL",
    "FOOTPRINT_MODEL_RULE_ID",
    "FootprintModelBigBedPaths",
    "FootprintModelBundlePaths",
    "POPULATION_TSV_COLUMNS",
    "STRATA_TSV_COLUMNS",
    "TF_MODEL_CLASS_ID",
    "TF_MODEL_FIBERLAYERS_AUTOSQL",
    "TF_MODEL_POPULATION_AUTOSQL",
    "TF_MODEL_RULE_ID",
    "TFModelBigBedPaths",
    "TFModelBundlePaths",
    "convert_footprint_model_bundle_to_bigbed",
    "convert_tf_model_bundle_to_bigbed",
    "footprint_model_bundle_paths",
    "write_footprint_model_bundle",
    "write_tf_model_bundle",
]
