"""Build chemistry-aware evidence at hierarchically discovered candidates."""
from __future__ import annotations

import math
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from consensus_recaller_collab.prototype import (
    PRESETS,
    ReadEvidence,
    discover_sites,
    fit_mixture_weights,
    load_region_evidence,
    parse_region,
    resolve_resource_path,
)
from fiberhmm.core.model_io import load_model_with_metadata
from fiberhmm.cli.dedup import cluster_reads
from fiberhmm.inference.tf_recaller import build_llr_tables
from consensus_recaller_collab.validation import VALIDATION_VERSION


DEFAULT_CONTROL_OFFSETS = tuple(
    offset for offset in range(-2000, 2001, 25) if abs(offset) >= 50
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _input_file_metadata(
    samples: Sequence[Mapping[str, object]],
    *,
    base_dir: Optional[Path] = None,
) -> List[dict]:
    rows = []
    seen = set()
    for sample in samples:
        for bam_path in sample.get("bams", []):
            path = _resolve_bam(str(bam_path), base_dir=base_dir)
            key = (str(sample["sample_id"]), str(path))
            if key in seen:
                continue
            seen.add(key)
            stat = path.stat()
            rows.append({
                "sample_id": str(sample["sample_id"]),
                "path": str(path),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            })
    return sorted(rows, key=lambda row: (row["sample_id"], row["path"]))


@dataclass(frozen=True)
class Candidate:
    cohort_id: str
    locus_id: str
    candidate_id: str
    axis: str
    start: int
    end: int
    center: int
    geometry_tier: int
    source_samples: Tuple[str, ...]
    source_support: int
    geometry_source_samples: Tuple[str, ...] = ()
    source_diagnostics: Tuple[dict, ...] = ()

    @property
    def interval(self) -> List[int]:
        return [self.start, self.end]


def _resolve_bam(path: str, *, base_dir: Optional[Path] = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    if base_dir is not None:
        manifest_dir = base_dir.resolve()
        for base in (manifest_dir, *manifest_dir.parents):
            manifest_relative = base / candidate
            if manifest_relative.exists():
                return manifest_relative.resolve()
        return manifest_dir / candidate
    return Path(__file__).resolve().parents[2] / candidate


def _permission(sample: Mapping[str, object], axis: str, default: str) -> str:
    return str(sample.get("axis_permissions", {}).get(axis, default))


def _sample_applies(
    sample: Mapping[str, object],
    locus: Mapping[str, object],
    *,
    include_exploratory: bool,
) -> bool:
    if sample.get("availability", "available") != "available":
        return False
    if sample.get("cohort_id") == locus.get("cohort_id"):
        restricted_loci = sample.get("restrict_loci")
        if restricted_loci is not None and locus["locus_id"] not in restricted_loci:
            return False
        restricted_groups = sample.get("restrict_comparison_groups")
        if restricted_groups is not None and locus.get("comparison_group") not in restricted_groups:
            return False
        return True
    return bool(
        include_exploratory
        and locus.get("comparison_group")
        in sample.get("exploratory_comparison_groups", [])
    )


class EvidenceLoader:
    """Cache models while loading one small validation locus at a time."""

    def __init__(
        self,
        min_mapq: int = 20,
        max_reads: int = 0,
        base_dir: Optional[Path] = None,
    ):
        self.min_mapq = min_mapq
        self.max_reads = max_reads
        self.base_dir = base_dir
        self._models: Dict[str, tuple] = {}
        self.diagnostics: Dict[Tuple[str, str], dict] = {}

    def _model(self, preset_name: str):
        if preset_name not in self._models:
            preset = PRESETS[preset_name]
            model, context_size, mode = load_model_with_metadata(
                resolve_resource_path(str(preset["model"]))
            )
            llr_hit, llr_miss = build_llr_tables(model)
            self._models[preset_name] = (
                context_size,
                mode,
                llr_hit,
                llr_miss,
            )
        return self._models[preset_name]

    def load(
        self,
        sample: Mapping[str, object],
        locus: Mapping[str, object],
        *,
        flank: int = 75,
    ) -> List[ReadEvidence]:
        preset_name = str(sample["preset"])
        preset = PRESETS[preset_name]
        context_size, mode, llr_hit, llr_miss = self._model(preset_name)
        chrom, start, end = parse_region(str(locus["region"]))
        load_start = max(0, start - flank)
        load_end = end + flank
        threshold = sample.get("prob_threshold", preset["prob_threshold"])
        reads: List[ReadEvidence] = []
        for bam_path in sample["bams"]:
            remaining = (
                0 if not self.max_reads else max(0, self.max_reads - len(reads))
            )
            if self.max_reads and remaining == 0:
                break
            reads.extend(load_region_evidence(
                str(_resolve_bam(str(bam_path), base_dir=self.base_dir)),
                chrom,
                load_start,
                load_end,
                strand_mode=preset["strand_mode"],
                mode=mode,
                context_size=context_size,
                prob_threshold=(None if threshold is None else int(threshold)),
                llr_hit=llr_hit,
                llr_miss=llr_miss,
                min_mapq=self.min_mapq,
                max_reads=remaining,
            ))
        raw_reads = len(reads)
        molecule_config = sample.get("molecule_collapse")
        molecule_diagnostics = {
            "mode": "read",
            "raw_reads": raw_reads,
            "analyzed_molecules": raw_reads,
            "fingerprintable_reads": 0,
            "duplicate_reads_collapsed": 0,
        }
        if isinstance(molecule_config, Mapping) and bool(
            molecule_config.get("enabled", False)
        ):
            reads, molecule_diagnostics = _collapse_molecule_families(
                reads,
                min_jaccard=float(molecule_config.get("min_jaccard", 0.95)),
                min_deam=int(molecule_config.get("min_deam", 10)),
                ignore_strand=bool(molecule_config.get("ignore_strand", False)),
                num_hashes=int(molecule_config.get("num_hashes", 32)),
                bands=int(molecule_config.get("bands", 8)),
                seed=int(molecule_config.get("seed", 7)),
            )
        self.diagnostics[(str(sample["sample_id"]), str(locus["locus_id"]))] = (
            molecule_diagnostics
        )
        return reads


def _collapse_molecule_families(
    reads: Sequence[ReadEvidence],
    *,
    min_jaccard: float,
    min_deam: int,
    ignore_strand: bool,
    num_hashes: int,
    bands: int,
    seed: int,
) -> Tuple[List[ReadEvidence], dict]:
    """Collapse amplified DAF reads by their hard-call deamination pattern.

    This mirrors ``fiberhmm-dedup`` but operates only on the reads loaded for
    one validation locus.  It never rewrites the source BAM.  Calls are
    clustered on hard modified-reference positions, and one maximally
    informative representative is retained per family.  Reads with too few
    hard calls to fingerprint remain separate molecules.
    """
    if not reads:
        return [], {
            "mode": "deamination_fingerprint",
            "raw_reads": 0,
            "analyzed_molecules": 0,
            "fingerprintable_reads": 0,
            "duplicate_reads_collapsed": 0,
            "duplication_fraction": 0.0,
        }
    position_sets = []
    group_keys = []
    for read in reads:
        fingerprint = (
            read.fingerprint_positions
            if read.fingerprint_positions is not None
            else read.positions[read.hits]
        )
        modified = frozenset(int(value) for value in fingerprint)
        if len(modified) < min_deam:
            position_sets.append(None)
            group_keys.append(None)
            continue
        position_sets.append(modified)
        group_keys.append(("" if ignore_strand else read.strand,))
    labels = cluster_reads(
        position_sets,
        group_keys,
        min_jaccard,
        num_hashes,
        bands,
        seed,
    )
    best_by_cluster: Dict[int, int] = {}
    cluster_sizes: Counter = Counter()
    for index, label_value in enumerate(labels):
        label = int(label_value)
        if label < 0:
            continue
        cluster_sizes[label] += 1
        incumbent = best_by_cluster.get(label)
        quality = (
            len(reads[index].positions),
            reads[index].ref_end - reads[index].ref_start,
        )
        if incumbent is None:
            best_by_cluster[label] = index
        else:
            incumbent_quality = (
                len(reads[incumbent].positions),
                reads[incumbent].ref_end - reads[incumbent].ref_start,
            )
            if quality > incumbent_quality:
                best_by_cluster[label] = index
    keep = {
        index for index, label_value in enumerate(labels)
        if int(label_value) < 0
    } | set(best_by_cluster.values())
    collapsed = [read for index, read in enumerate(reads) if index in keep]
    fingerprintable = sum(int(label) >= 0 for label in labels)
    duplicates = fingerprintable - len(best_by_cluster)
    return collapsed, {
        "mode": "deamination_fingerprint",
        "raw_reads": len(reads),
        "analyzed_molecules": len(collapsed),
        "fingerprintable_reads": fingerprintable,
        "unfingerprintable_reads": len(reads) - fingerprintable,
        "duplicate_reads_collapsed": duplicates,
        "duplication_fraction": (
            duplicates / fingerprintable if fingerprintable else 0.0
        ),
        "largest_family": max(cluster_sizes.values(), default=0),
        "min_jaccard": min_jaccard,
        "min_deam": min_deam,
    }


def _weighted_median(values: Sequence[int], weights: Sequence[float]) -> int:
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    halfway = 0.5 * sum(weight for _, weight in ordered)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= halfway:
            return int(value)
    return int(ordered[-1][0])


def _cluster_source_candidates(
    proposals: Sequence[dict],
    *,
    center_radius: int,
    cohort_id: str,
    locus_id: str,
    axis: str,
) -> List[Candidate]:
    if not proposals:
        return []
    ordered = sorted(proposals, key=lambda proposal: proposal["center"])
    groups: List[List[dict]] = []
    for proposal in ordered:
        if groups:
            group_center = float(np.median([item["center"] for item in groups[-1]]))
        else:
            group_center = -math.inf
        if groups and proposal["center"] - group_center <= center_radius:
            groups[-1].append(proposal)
        else:
            groups.append([proposal])

    candidates = []
    for group in groups:
        best_tier = min(int(item["geometry_tier"]) for item in group)
        geometry = [item for item in group if int(item["geometry_tier"]) == best_tier]
        weights = [max(1.0, float(item["support"])) for item in geometry]
        start = _weighted_median([int(item["start"]) for item in geometry], weights)
        end = _weighted_median([int(item["end"]) for item in geometry], weights)
        if end <= start:
            continue
        center = int(round(float(np.median([item["center"] for item in geometry]))))
        source_diagnostics = tuple(sorted((
            ({
                key: value for key, value in item.items()
                if key not in {"sample_id"}
            } | {"sample_id": str(item["sample_id"])})
            for item in group
        ),
            key=lambda item: (
                int(item["geometry_tier"]), str(item["sample_id"]),
                int(item["start"]), int(item["end"]),
            ),
        ))
        candidates.append(Candidate(
            cohort_id=cohort_id,
            locus_id=locus_id,
            candidate_id="",
            axis=axis,
            start=start,
            end=end,
            center=center,
            geometry_tier=best_tier,
            source_samples=tuple(sorted({str(item["sample_id"]) for item in group})),
            source_support=int(sum(int(item["support"]) for item in group)),
            geometry_source_samples=tuple(sorted({
                str(item["sample_id"]) for item in geometry
            })),
            source_diagnostics=source_diagnostics,
        ))
    prefix = "tf" if axis == "fine_tf" else "nuc"
    return [Candidate(
        **{
            **asdict(candidate),
            "candidate_id": f"{locus_id}.{prefix}{index:03d}",
        }
    ) for index, candidate in enumerate(candidates, start=1)]


def _discover_fine_proposals(
    reads: Sequence[ReadEvidence],
    sample: Mapping[str, object],
    locus: Mapping[str, object],
    geometry_tier: int,
) -> List[dict]:
    _, start, end = parse_region(str(locus["region"]))
    discovery = sample.get("discovery", {})
    min_support = int(discovery.get("min_tf_support", 10))
    if min_support <= 0:
        return []
    sites = discover_sites(
        reads,
        start,
        end,
        min_tq=int(discovery.get("min_tq", 100)),
        min_support=min_support,
        center_radius=10,
        peak_distance=15,
        max_boundary_mad=12.0,
        min_local_enrichment=1.5,
        local_background_radius=250,
        max_auto_sites=100,
    )
    return [{
        "sample_id": sample["sample_id"],
        "start": site.start,
        "end": site.end,
        "center": site.center,
        "support": max(site.support.values(), default=0),
        "geometry_tier": geometry_tier,
        "support_by_strand": {
            str(key): int(value) for key, value in sorted(site.support.items())
        },
        "all_support_by_strand": {
            str(key): int(value) for key, value in sorted(site.all_support.items())
        },
        "median_tq_by_strand": {
            str(key): (None if value is None else float(value))
            for key, value in sorted(site.median_tq.items())
        },
        "start_mad": float(site.start_mad),
        "end_mad": float(site.end_mad),
        "local_enrichment": float(site.local_enrichment),
        "local_enrichment_by_strand": {
            str(key): float(value)
            for key, value in sorted(
                (site.local_enrichment_by_strand or {}).items()
            )
        },
    } for site in sites]


def _discover_nuc_proposals(
    reads: Sequence[ReadEvidence],
    sample: Mapping[str, object],
    locus: Mapping[str, object],
    geometry_tier: int,
) -> List[dict]:
    chrom, start, end = parse_region(str(locus["region"]))
    del chrom
    min_support = int(sample.get("discovery", {}).get("min_nuc_support", 10))
    if min_support <= 0 or end <= start:
        return []
    calls = [
        (call, read.name)
        for read in reads
        for call in read.nucs
        if 90 <= call.end - call.start <= 220
        and start <= call.center < end
    ]
    if not calls:
        return []
    histogram = np.zeros(end - start, dtype=np.float64)
    for call, _ in calls:
        index = int(round(call.center)) - start
        if 0 <= index < len(histogram):
            histogram[index] += 1.0
    sigma = 10.0
    smooth = gaussian_filter1d(histogram, sigma)
    minimum_height = 0.35 * min_support / (sigma * math.sqrt(2.0 * math.pi))
    peaks, _ = find_peaks(
        smooth,
        distance=70,
        height=minimum_height,
        prominence=0.25 * minimum_height,
    )
    proposals = []
    for peak in peaks:
        center = start + int(peak)
        nearby = [(call, name) for call, name in calls if abs(call.center - center) <= 25]
        support = len({name for _, name in nearby})
        if support < min_support:
            continue
        starts = [call.start for call, _ in nearby]
        ends = [call.end for call, _ in nearby]
        proposals.append({
            "sample_id": sample["sample_id"],
            "start": int(round(float(np.median(starts)))),
            "end": int(round(float(np.median(ends)))),
            "center": center,
            "support": support,
            "geometry_tier": geometry_tier,
            "start_mad": float(_median_absolute_deviation(starts)),
            "end_mad": float(_median_absolute_deviation(ends)),
        })
    return proposals


def _median_absolute_deviation(values: Sequence[int]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return math.nan
    return float(np.median(np.abs(array - np.median(array))))


def discover_candidates(
    manifest: Mapping[str, object],
    hierarchy: Mapping[str, object],
    locus: Mapping[str, object],
    loaded: Mapping[str, Sequence[ReadEvidence]],
) -> List[Candidate]:
    proposals = {"fine_tf": [], "broad_nucleosome": []}
    for sample in manifest["samples"]:
        sample_id = str(sample["sample_id"])
        if sample_id not in loaded or not bool(sample.get("truth_vote", True)):
            continue
        if sample.get("cohort_id") != locus.get("cohort_id"):
            continue
        family = str(sample["assay_family"])
        for axis in proposals:
            family_config = hierarchy["axes"][axis]["families"].get(family)
            if family_config is None:
                continue
            permission = _permission(sample, axis, str(family_config["role"]))
            discovery = sample.get("discovery", {})
            support_seed = (
                axis == "fine_tf"
                and permission == "support"
                and bool(discovery.get("allow_support_seed", False))
            )
            if permission != "anchor" and not support_seed:
                continue
            configured_tier = family_config.get("geometry_tier")
            if configured_tier is None:
                tier = int(discovery.get("candidate_geometry_tier", 99))
            else:
                tier = int(configured_tier)
            if axis == "fine_tf":
                proposals[axis].extend(_discover_fine_proposals(
                    loaded[sample_id], sample, locus, tier
                ))
            else:
                proposals[axis].extend(_discover_nuc_proposals(
                    loaded[sample_id], sample, locus, tier
                ))
    return (
        _cluster_source_candidates(
            proposals["fine_tf"],
            center_radius=10,
            cohort_id=str(locus["cohort_id"]),
            locus_id=str(locus["locus_id"]),
            axis="fine_tf",
        )
        + _cluster_source_candidates(
            proposals["broad_nucleosome"],
            center_radius=30,
            cohort_id=str(locus["cohort_id"]),
            locus_id=str(locus["locus_id"]),
            axis="broad_nucleosome",
        )
    )


def _softmax3(a: float, b: float, c: float) -> Tuple[float, float, float]:
    maximum = max(a, b, c)
    values = [math.exp(a - maximum), math.exp(b - maximum), math.exp(c - maximum)]
    total = sum(values)
    return values[0] / total, values[1] / total, values[2] / total


def _fine_read_posterior(
    read: ReadEvidence,
    candidate: Candidate,
    flank: int,
) -> Optional[float]:
    site_llr, site_opportunities, _ = read.interval_evidence(
        candidate.start, candidate.end
    )
    left_llr, left_opportunities, _ = read.interval_evidence(
        candidate.start - flank, candidate.start
    )
    right_llr, right_opportunities, _ = read.interval_evidence(
        candidate.end, candidate.end + flank
    )
    if site_opportunities == 0 or left_opportunities + right_opportunities == 0:
        return None
    flank_llr = left_llr + right_llr
    # Common accessible baseline: A protects nothing, TF protects the focal
    # interval, and local N protects the focal interval plus available flanks.
    _, tf_posterior, _ = _softmax3(0.0, site_llr, site_llr + flank_llr)
    return tf_posterior


def _fine_read_state_likelihoods(
    read: ReadEvidence,
    candidate: Candidate,
    flank: int,
) -> Optional[Tuple[np.ndarray, int, int]]:
    site_llr, site_opportunities, _ = read.interval_evidence(
        candidate.start, candidate.end
    )
    left_llr, left_opportunities, _ = read.interval_evidence(
        candidate.start - flank, candidate.start
    )
    right_llr, right_opportunities, _ = read.interval_evidence(
        candidate.end, candidate.end + flank
    )
    flank_opportunities = left_opportunities + right_opportunities
    if site_opportunities == 0 or flank_opportunities == 0:
        return None
    # Likelihoods relative to a fully accessible local window.  The null
    # permits accessible and locally nucleosomal molecules; the alternative
    # adds a focal TF state.  Missing flanks contribute no likelihood term.
    return (
        np.asarray(
            [0.0, site_llr, site_llr + left_llr + right_llr],
            dtype=np.float64,
        ),
        site_opportunities,
        flank_opportunities,
    )


def _nuc_read_posterior(read: ReadEvidence, candidate: Candidate) -> Optional[float]:
    llr, opportunities, _ = read.interval_evidence(candidate.start, candidate.end)
    if opportunities == 0:
        return None
    if llr >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(llr, 745.0)))
    exp_value = math.exp(max(llr, -745.0))
    return exp_value / (1.0 + exp_value)


def _nuc_read_state_likelihoods(
    read: ReadEvidence,
    candidate: Candidate,
) -> Optional[Tuple[np.ndarray, int, int]]:
    llr, opportunities, _ = read.interval_evidence(candidate.start, candidate.end)
    if opportunities == 0:
        return None
    return np.asarray([0.0, llr], dtype=np.float64), opportunities, 0


def _fit_state_model(
    log_likelihoods: Sequence[np.ndarray],
    *,
    null_columns: Sequence[int],
    positive_column: int,
) -> dict:
    """Fit nested population mixtures and return a BIC log Bayes factor.

    The statistic tests whether adding the focal state explains independent
    molecules better than the corresponding no-focal-state mixture.  BIC is
    used as a conservative Laplace approximation while the priors are being
    calibrated on held-out loci.
    """
    if not log_likelihoods:
        return {
            "informative": 0,
            "log_bayes_factor": 0.0,
            "expected_positive": 0.0,
            "hard_positive": 0,
            "state_weights": [],
            "null_state_weights": [],
            "alt_log_likelihood": 0.0,
            "null_log_likelihood": 0.0,
        }
    matrix = np.vstack(log_likelihoods)
    alt_weights, _ = fit_mixture_weights(matrix)
    null_matrix = matrix[:, list(null_columns)]
    null_weights, _ = fit_mixture_weights(null_matrix)

    alt_log_joint = matrix + np.log(np.maximum(alt_weights, 1e-300))[None, :]
    alt_log_norm = np.logaddexp.reduce(alt_log_joint, axis=1)
    null_log_joint = (
        null_matrix + np.log(np.maximum(null_weights, 1e-300))[None, :]
    )
    null_log_norm = np.logaddexp.reduce(null_log_joint, axis=1)
    alt_log_likelihood = float(np.sum(alt_log_norm))
    null_log_likelihood = float(np.sum(null_log_norm))
    added_parameters = (matrix.shape[1] - 1) - (null_matrix.shape[1] - 1)
    complexity_penalty = 0.5 * added_parameters * math.log(len(matrix))
    log_bayes_factor = (
        alt_log_likelihood - null_log_likelihood - complexity_penalty
    )
    responsibilities = np.exp(alt_log_joint - alt_log_norm[:, None])
    positive = responsibilities[:, positive_column]
    return {
        "informative": len(matrix),
        "log_bayes_factor": float(log_bayes_factor),
        "expected_positive": float(np.sum(positive)),
        "hard_positive": int(np.sum(positive >= 0.5)),
        "state_weights": [float(value) for value in alt_weights],
        "null_state_weights": [float(value) for value in null_weights],
        "alt_log_likelihood": alt_log_likelihood,
        "null_log_likelihood": null_log_likelihood,
        "complexity_penalty": float(complexity_penalty),
    }


def summarize_candidate(
    reads: Sequence[ReadEvidence],
    sample: Mapping[str, object],
    candidate: Candidate,
    *,
    flank: int = 25,
) -> dict:
    state_values = []
    strand_values: Dict[str, List[np.ndarray]] = {}
    library_values: Dict[str, List[np.ndarray]] = {}
    site_opportunities = 0
    flank_opportunities = 0
    strand_opportunities: Dict[str, List[int]] = {}
    library_opportunities: Dict[str, List[int]] = {}
    library_explicit_tf: Counter = Counter()
    library_explicit_nuc: Counter = Counter()
    explicit_tf = 0
    explicit_nuc = 0
    for read in reads:
        if candidate.axis == "fine_tf":
            state_evidence = _fine_read_state_likelihoods(read, candidate, flank)
            has_explicit_tf = int(any(
                abs(call.center - candidate.center) <= 10 and call.score >= 100
                for call in read.tfs
            ))
            explicit_tf += has_explicit_tf
            library_explicit_tf[str(read.library_id or "unspecified")] += has_explicit_tf
        else:
            state_evidence = _nuc_read_state_likelihoods(read, candidate)
            has_explicit_nuc = int(any(
                call.start <= candidate.center < call.end
                and 90 <= call.end - call.start <= 220
                for call in read.nucs
            ))
            explicit_nuc += has_explicit_nuc
            library_explicit_nuc[str(read.library_id or "unspecified")] += has_explicit_nuc
        if state_evidence is None:
            continue
        likelihoods, site_count, flank_count = state_evidence
        state_values.append(likelihoods)
        strand_values.setdefault(read.strand, []).append(likelihoods)
        library_id = str(read.library_id or "unspecified")
        library_values.setdefault(library_id, []).append(likelihoods)
        site_opportunities += site_count
        flank_opportunities += flank_count
        opportunity_summary = strand_opportunities.setdefault(read.strand, [0, 0])
        opportunity_summary[0] += site_count
        opportunity_summary[1] += flank_count
        library_summary = library_opportunities.setdefault(library_id, [0, 0])
        library_summary[0] += site_count
        library_summary[1] += flank_count
    if candidate.axis == "fine_tf":
        null_columns = (0, 2)
        positive_column = 1
        state_names = ["accessible", "tf", "local_nucleosome"]
        null_state_names = ["accessible", "local_nucleosome"]
    else:
        null_columns = (0,)
        positive_column = 1
        state_names = ["not_nucleosome", "nucleosome"]
        null_state_names = ["not_nucleosome"]
    fitted = _fit_state_model(
        state_values,
        null_columns=null_columns,
        positive_column=positive_column,
    )
    by_strand: Dict[str, dict] = {}
    for strand, values in strand_values.items():
        strand_fit = _fit_state_model(
            values,
            null_columns=null_columns,
            positive_column=positive_column,
        )
        by_strand[strand] = {
            "informative": strand_fit["informative"],
            "expected_positive": strand_fit["expected_positive"],
            "hard_positive": strand_fit["hard_positive"],
            "mean_posterior": (
                strand_fit["expected_positive"] / strand_fit["informative"]
                if strand_fit["informative"] else None
            ),
            "log_bayes_factor": strand_fit["log_bayes_factor"],
            "state_weights": dict(zip(state_names, strand_fit["state_weights"])),
            "site_opportunities": strand_opportunities[strand][0],
            "flank_opportunities": strand_opportunities[strand][1],
        }
    by_library: Dict[str, dict] = {}
    for library_id, values in sorted(library_values.items()):
        library_fit = _fit_state_model(
            values,
            null_columns=null_columns,
            positive_column=positive_column,
        )
        by_library[library_id] = {
            "informative": library_fit["informative"],
            "expected_positive": library_fit["expected_positive"],
            "hard_positive": library_fit["hard_positive"],
            "mean_posterior": (
                library_fit["expected_positive"] / library_fit["informative"]
                if library_fit["informative"] else None
            ),
            "log_bayes_factor": library_fit["log_bayes_factor"],
            "positive_log_bayes_factor": max(
                0.0, library_fit["log_bayes_factor"]
            ),
            "state_weights": dict(zip(state_names, library_fit["state_weights"])),
            "site_opportunities": library_opportunities[library_id][0],
            "flank_opportunities": library_opportunities[library_id][1],
            "explicit_tf_reads": int(library_explicit_tf[library_id]),
            "explicit_nuc_reads": int(library_explicit_nuc[library_id]),
        }
    geometry_calls = []
    if candidate.axis == "fine_tf":
        minimum_score = int(sample.get("discovery", {}).get("min_tq", 100))
        geometry_calls = [
            call for read in reads for call in read.tfs
            if call.score >= minimum_score
            and abs(call.center - candidate.center) <= 10
        ]
    else:
        geometry_calls = [
            call for read in reads for call in read.nucs
            if 90 <= call.end - call.start <= 220
            and abs(call.center - candidate.center) <= 30
        ]
    geometry_interval = None
    if geometry_calls:
        geometry_interval = [
            int(round(float(np.median([call.start for call in geometry_calls])))),
            int(round(float(np.median([call.end for call in geometry_calls])))),
        ]
    return {
        "cohort_id": candidate.cohort_id,
        "locus_id": candidate.locus_id,
        "candidate_id": candidate.candidate_id,
        "axis": candidate.axis,
        "sample_id": sample["sample_id"],
        "positive": fitted["expected_positive"],
        "informative": fitted["informative"],
        "hard_positive": fitted["hard_positive"],
        "mean_posterior": (
            fitted["expected_positive"] / fitted["informative"]
            if fitted["informative"] else None
        ),
        "log_bayes_factor": fitted["log_bayes_factor"],
        "positive_log_bayes_factor": float(sum(
            max(0.0, summary["log_bayes_factor"])
            for summary in by_strand.values()
        )),
        "state_weights": dict(zip(state_names, fitted["state_weights"])),
        "null_state_weights": dict(zip(
            null_state_names, fitted["null_state_weights"]
        )),
        "site_opportunities": site_opportunities,
        "flank_opportunities": flank_opportunities,
        "model": {
            "alternative": "+".join(state_names),
            "null": "+".join(null_state_names),
            "alt_log_likelihood": fitted["alt_log_likelihood"],
            "null_log_likelihood": fitted["null_log_likelihood"],
            "complexity_penalty": fitted.get("complexity_penalty", 0.0),
            "evidence_statistic": "BIC-approximated log Bayes factor",
        },
        "interval": candidate.interval,
        "geometry_interval": geometry_interval,
        "geometry_support": len(geometry_calls),
        "explicit_tf_reads": explicit_tf,
        "explicit_nuc_reads": explicit_nuc,
        "by_strand": by_strand,
        "by_library": by_library,
    }


def _opportunity_prefix(
    reads: Sequence[ReadEvidence],
    start: int,
    end: int,
) -> np.ndarray:
    width = end - start
    histogram = np.zeros(width, dtype=np.int64)
    for read in reads:
        if len(read.positions) == 0:
            continue
        lo = int(np.searchsorted(read.positions, start, side="left"))
        hi = int(np.searchsorted(read.positions, end, side="left"))
        if hi <= lo:
            continue
        indices = read.positions[lo:hi] - start
        histogram += np.bincount(indices, minlength=width)[:width]
    return np.concatenate(([0], np.cumsum(histogram, dtype=np.int64)))


def _prefix_interval_total(
    prefix: np.ndarray,
    start: int,
    end: int,
    locus_start: int,
) -> int:
    left = max(0, start - locus_start)
    right = min(len(prefix) - 1, end - locus_start)
    if right <= left:
        return 0
    return int(prefix[right] - prefix[left])


def build_shift_controls(
    candidates: Sequence[Candidate],
    locus: Mapping[str, object],
    loaded: Mapping[str, Sequence[ReadEvidence]],
    *,
    controls_per_candidate: int,
    offsets: Sequence[int],
    exclusion_buffer: int = 15,
    max_opportunity_log_mismatch: float = math.log(4.0),
) -> Tuple[List[Candidate], List[dict]]:
    """Choose local, opportunity-matched shifts away from known TF sites."""
    if controls_per_candidate <= 0:
        return [], []
    _, locus_start, locus_end = parse_region(str(locus["region"]))
    fine = [candidate for candidate in candidates if candidate.axis == "fine_tf"]
    prefixes = {
        sample_id: _opportunity_prefix(reads, locus_start, locus_end)
        for sample_id, reads in loaded.items()
    }
    controls: List[Candidate] = []
    metadata: List[dict] = []
    for parent in fine:
        source_ids = [
            sample_id for sample_id in parent.source_samples
            if sample_id in prefixes
        ]
        if not source_ids:
            source_ids = list(prefixes)
        target_opportunities = sum(
            _prefix_interval_total(
                prefixes[sample_id], parent.start, parent.end, locus_start
            )
            for sample_id in source_ids
        )
        proposals = []
        for offset in offsets:
            if offset == 0:
                continue
            start = parent.start + int(offset)
            end = parent.end + int(offset)
            if start < locus_start or end > locus_end:
                continue
            if any(
                start < other.end + exclusion_buffer
                and end > other.start - exclusion_buffer
                for other in fine
            ):
                continue
            opportunities = sum(
                _prefix_interval_total(
                    prefixes[sample_id], start, end, locus_start
                )
                for sample_id in source_ids
            )
            mismatch = abs(math.log(
                (opportunities + 1.0) / (target_opportunities + 1.0)
            ))
            if mismatch > max_opportunity_log_mismatch:
                continue
            proposals.append((mismatch, abs(offset), offset, start, end, opportunities))
        selected_proposals = []
        for proposal in sorted(proposals):
            _, _, _, start, end, _ = proposal
            if any(
                start < selected_end + exclusion_buffer
                and end > selected_start - exclusion_buffer
                for _, _, _, selected_start, selected_end, _ in selected_proposals
            ):
                continue
            selected_proposals.append(proposal)
            if len(selected_proposals) >= controls_per_candidate:
                break
        for control_index, proposal in enumerate(selected_proposals, start=1):
            mismatch, _, offset, start, end, opportunities = proposal
            control_id = f"{parent.candidate_id}.control{control_index:02d}"
            control = Candidate(
                cohort_id=parent.cohort_id,
                locus_id=parent.locus_id,
                candidate_id=control_id,
                axis="fine_tf",
                start=start,
                end=end,
                center=parent.center + int(offset),
                geometry_tier=999,
                source_samples=parent.source_samples,
                source_support=0,
                geometry_source_samples=(),
                source_diagnostics=(),
            )
            controls.append(control)
            metadata.append({
                **asdict(control),
                "parent_candidate_id": parent.candidate_id,
                "control_type": "opportunity_matched_nonoverlapping_shift",
                "offset": int(offset),
                "source_opportunities": opportunities,
                "parent_source_opportunities": target_opportunities,
                "opportunity_log_mismatch": float(mismatch),
            })
    return controls, metadata


def build_evidence(
    manifest: Mapping[str, object],
    hierarchy: Mapping[str, object],
    *,
    locus_ids: Optional[Sequence[str]] = None,
    include_exploratory: bool = True,
    min_mapq: int = 20,
    max_reads: int = 0,
    controls_per_candidate: int = 0,
    control_offsets: Sequence[int] = DEFAULT_CONTROL_OFFSETS,
    max_control_opportunity_ratio: float = 4.0,
    manifest_base: Optional[str] = None,
) -> dict:
    if controls_per_candidate < 0:
        raise ValueError("controls_per_candidate must be non-negative")
    if max_control_opportunity_ratio < 1.0:
        raise ValueError("max control opportunity ratio must be at least one")
    selected = set(locus_ids or [])
    loci = [
        locus for locus in manifest["loci"]
        if not selected or str(locus["locus_id"]) in selected
    ]
    missing = selected - {str(locus["locus_id"]) for locus in loci}
    if missing:
        raise ValueError(f"unknown loci: {','.join(sorted(missing))}")
    base_dir = Path(manifest_base).resolve() if manifest_base else None
    loader = EvidenceLoader(
        min_mapq=min_mapq, max_reads=max_reads, base_dir=base_dir
    )
    all_candidates = []
    all_records = []
    all_controls = []
    all_control_records = []
    locus_summaries = []
    applied_samples: Dict[str, Mapping[str, object]] = {}
    for locus in loci:
        applicable = [
            sample for sample in manifest["samples"]
            if _sample_applies(
                sample, locus, include_exploratory=include_exploratory
            )
        ]
        applied_samples.update({str(sample["sample_id"]): sample for sample in applicable})
        loaded = {
            str(sample["sample_id"]): loader.load(sample, locus)
            for sample in applicable
        }
        candidates = discover_candidates(manifest, hierarchy, locus, loaded)
        for candidate in candidates:
            all_candidates.append(asdict(candidate))
            for sample in applicable:
                all_records.append(summarize_candidate(
                    loaded[str(sample["sample_id"])], sample, candidate
                ))
        controls, control_metadata = build_shift_controls(
            candidates,
            locus,
            loaded,
            controls_per_candidate=controls_per_candidate,
            offsets=control_offsets,
            max_opportunity_log_mismatch=math.log(max_control_opportunity_ratio),
        )
        all_controls.extend(control_metadata)
        for control in controls:
            for sample in applicable:
                all_control_records.append(summarize_candidate(
                    loaded[str(sample["sample_id"])], sample, control
                ))
        locus_summaries.append({
            "locus_id": locus["locus_id"],
            "loaded_reads": {
                sample_id: len(reads) for sample_id, reads in loaded.items()
            },
            "molecule_diagnostics": {
                sample_id: loader.diagnostics[(sample_id, str(locus["locus_id"]))]
                for sample_id in loaded
            },
            "fine_tf_candidates": sum(
                candidate.axis == "fine_tf" for candidate in candidates
            ),
            "broad_nucleosome_candidates": sum(
                candidate.axis == "broad_nucleosome" for candidate in candidates
            ),
            "shift_controls": len(controls),
        })
    return {
        "schema_version": 2,
        "producer": {
            "name": "fiberhmm-consensus-validation",
            "version": VALIDATION_VERSION,
            "manifest_sha256": _canonical_sha256(manifest),
            "hierarchy_sha256": _canonical_sha256(hierarchy),
        },
        "parameters": {
            "locus_ids": sorted(selected),
            "include_exploratory": include_exploratory,
            "min_mapq": min_mapq,
            "max_reads": max_reads,
            "controls_per_candidate": controls_per_candidate,
            "control_offsets": [int(value) for value in control_offsets],
            "max_control_opportunity_ratio": max_control_opportunity_ratio,
        },
        "input_files": _input_file_metadata(
            list(applied_samples.values()), base_dir=base_dir
        ),
        "observation_source": "standard BAM sequence + hard MM/ML + MA diagnostics",
        "candidate_policy": "anchor assays only; best available geometry tier",
        "short_read_policy": "local focal and one-or-more flank opportunities; no locus-span requirement",
        "loci": locus_summaries,
        "candidates": all_candidates,
        "records": all_records,
        "control_policy": {
            "controls_per_fine_candidate": controls_per_candidate,
            "offsets": [int(value) for value in control_offsets],
            "known_site_exclusion_buffer": 15,
            "maximum_opportunity_ratio": max_control_opportunity_ratio,
            "selection": "closest hard-call opportunity count among nonoverlapping shifts",
        },
        "controls": all_controls,
        "control_records": all_control_records,
    }
