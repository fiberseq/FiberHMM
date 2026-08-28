#!/usr/bin/env python3
"""Report-only consensus recaller with two independent inference passes.

``strand_rescue`` uses the other physical strand to rescue weak focal TF
evidence in stranded DAF/Nanopore assays. ``composite_deconvolution`` asks
whether a current nuc-like protected block is better explained by a recurring
configuration of explicit TF footprints. PacBio molecules are duplex and are
never split by alignment orientation.

Only standard BAM sequence, MM/ML, and existing MA calls are read. No BAM is
modified. The installed command deliberately stops at atomic JSON/TSV
proposals so inference and future BAM mutation remain separate stages.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from consensus_recaller_collab.prototype import (
        PRESETS,
        MIN_MAPPED_ANNOTATION_FRACTION,
        DddaRadialNucScorer,
        IntervalCall,
        ReadEvidence,
        SiteTemplate,
        analyze_window,
        build_ddda_radial_nuc_scorer,
        calibrate_cohort_deamination,
        discover_sites,
        group_sites,
        load_region_evidence,
        match_direct_site_indices,
        parse_region,
        resolve_resource_path,
    )
    from consensus_recaller_collab.validation.evidence import (
        _collapse_molecule_families,
    )
    from consensus_recaller_collab.validation import VALIDATION_VERSION
except ModuleNotFoundError:  # Direct ``python path/to/revised_prototype.py``.
    from prototype import (  # type: ignore[no-redef]
        PRESETS,
        MIN_MAPPED_ANNOTATION_FRACTION,
        DddaRadialNucScorer,
        IntervalCall,
        ReadEvidence,
        SiteTemplate,
        analyze_window,
        build_ddda_radial_nuc_scorer,
        calibrate_cohort_deamination,
        discover_sites,
        group_sites,
        load_region_evidence,
        match_direct_site_indices,
        parse_region,
        resolve_resource_path,
    )
    from validation.evidence import _collapse_molecule_families  # type: ignore
    from validation import VALIDATION_VERSION  # type: ignore
from fiberhmm.core.model_io import load_model_with_metadata
from fiberhmm.inference.tf_recaller import N_CTX, UNMETH_OFFSET, build_llr_tables


MIN_COMPOSITE_NUC_SPAN = 90
# A TF union is only credible up to roughly one nucleosome's footprint.  Beyond
# that a block with no internal accessibility is most parsimoniously a
# nucleosome (or an over-merged di-nucleosome the recaller cannot split): it is
# not "two TFs that happen to sit on either end", and it is not a nucleosome
# plus an edge TF either -- both invent a composite the bases do not support.
MAX_COMPOSITE_NUC_SPAN = 220


def _file_metadata(path: str) -> dict:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


PROPOSAL_TSV_COLUMNS = (
    "pass", "proposal_id", "read", "library_id", "strand",
    "current_start", "current_end", "current_state", "proposed_state",
    "posterior", "log_bf", "tier", "source_prior",
    "nuc_prior_odds", "replacement_kind", "decomposition_posterior",
    "replacement_intervals",
)


def proposal_rows(report: dict) -> List[dict]:
    rows = []
    for window in report.get("strand_rescue", {}).get("windows", []):
        for target, result in sorted(window.get("cross_strand", {}).items()):
            source = result.get("source_prior")
            for scenario_key, scenario in sorted(result.get("scenarios", {}).items()):
                for proposal in scenario.get("proposals", []):
                    start, end = proposal["current_interval"]
                    rows.append({
                        "pass": "strand_rescue",
                        "proposal_id": proposal["proposal_id"],
                        "read": proposal["read"],
                        "library_id": proposal.get("library_id"),
                        "strand": target,
                        "current_start": start,
                        "current_end": end,
                        "current_state": proposal["current"],
                        "proposed_state": proposal["proposed"],
                        "posterior": proposal["posterior"],
                        "log_bf": proposal["log_bf_vs_current"],
                        "tier": proposal.get("proposal_tier", "strong"),
                        "source_prior": source,
                        "nuc_prior_odds": scenario_key,
                        "replacement_kind": "strand_rescued_tf",
                        "decomposition_posterior": "",
                        "replacement_intervals": json.dumps(
                            proposal.get("proposed_site_intervals", []),
                            separators=(",", ":"),
                        ),
                    })
    composite = report.get("composite_deconvolution", {})
    production_key = composite.get("production_scenario_key")
    production = composite.get("scenarios", {}).get(str(production_key), {})
    for proposal in production.get("proposals", []):
        start, end = proposal["current_interval"]
        rows.append({
            "pass": "composite_deconvolution",
            "proposal_id": proposal["proposal_id"],
            "read": proposal["read"],
            "library_id": proposal.get("library_id"),
            "strand": proposal["strand"],
            "current_start": start,
            "current_end": end,
            "current_state": "N",
            "proposed_state": (
                proposal["best_tf_state"]
                if proposal.get("decomposition_resolved", True)
                else "TF_COMPLEX"
            ),
            "posterior": proposal.get(
                "replacement_posterior", proposal["best_tf_posterior"]
            ),
            "log_bf": proposal.get(
                "complex_log_posterior_odds_vs_n",
                proposal["best_tf_log_posterior_odds_vs_n"],
            ),
            "tier": proposal["proposal_tier"],
            "source_prior": "population_configuration",
            "nuc_prior_odds": proposal["nuc_prior_odds"],
            "replacement_kind": proposal.get(
                "replacement_kind", "exact_configuration"
            ),
            "decomposition_posterior": proposal.get(
                "best_decomposition_posterior_given_complex", ""
            ),
            "replacement_intervals": json.dumps(
                proposal.get("replacement_intervals", []),
                separators=(",", ":"),
            ),
        })
    return sorted(rows, key=lambda row: (
        row["pass"], row["read"], row["current_start"], row["current_end"],
        row["proposed_state"],
    ))


def write_proposal_tsv(report: dict, path: str) -> None:
    rows = proposal_rows(report)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=PROPOSAL_TSV_COLUMNS, delimiter="\t"
    )
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write(Path(path), buffer.getvalue())


def parse_site_interval(value: str) -> Tuple[int, int]:
    """Parse a zero-based half-open ``START-END`` focal interval."""
    try:
        start_text, end_text = value.split("-", 1)
        start, end = int(start_text), int(end_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "site interval must be START-END"
        ) from error
    if start < 0 or end <= start:
        raise argparse.ArgumentTypeError(
            "site interval must have 0 <= START < END"
        )
    return start, end


def build_forced_site_template(
    reads: Sequence[ReadEvidence],
    interval: Tuple[int, int],
    *,
    site_id: str,
    min_tq: int,
    center_radius: int,
) -> SiteTemplate:
    """Describe an externally nominated site using source-call diagnostics.

    The nominated interval supplies only the geometry to test.  Support and
    configuration priors still come exclusively from explicit source calls;
    an external assay therefore cannot manufacture same-assay support.
    """
    start, end = interval
    center = int(round((start + end) / 2.0))
    strands = sorted({read.strand for read in reads}) or ["UNKNOWN"]
    support: Dict[str, int] = {}
    all_support: Dict[str, int] = {}
    median_tq: Dict[str, Optional[float]] = {}
    high_calls: List[IntervalCall] = []
    local_enrichment_by_strand: Dict[str, float] = {}
    for strand in strands:
        nearby = [
            (call, read.name)
            for read in reads if read.strand == strand
            for call in read.tfs
            if abs(call.center - center) <= center_radius
        ]
        high = [(call, name) for call, name in nearby if call.score >= min_tq]
        support[strand] = len({name for _, name in high})
        all_support[strand] = len({name for _, name in nearby})
        median_tq[strand] = (
            float(np.median([call.score for call, _ in nearby]))
            if nearby else None
        )
        high_calls.extend(call for call, _ in high)
        background_names = {
            read.name
            for read in reads if read.strand == strand
            for call in read.tfs
            if call.score >= min_tq
            and 2 * center_radius < abs(call.center - center) <= 250
        }
        background_width = max(1, 2 * (250 - 2 * center_radius))
        expected = (
            len(background_names) * (2 * center_radius + 1)
            / background_width
        )
        local_enrichment_by_strand[strand] = float(
            (support[strand] + 0.5) / (expected + 0.5)
        )

    def mad(values: Sequence[int]) -> float:
        if not values:
            return 0.0
        array = np.asarray(values, dtype=np.float64)
        return float(np.median(np.abs(array - np.median(array))))

    return SiteTemplate(
        site_id=site_id,
        start=start,
        end=end,
        center=center,
        support=support,
        all_support=all_support,
        median_tq=median_tq,
        start_mad=mad([call.start for call in high_calls]),
        end_mad=mad([call.end for call in high_calls]),
        # Geometry is external, but focality is always recomputed from this BAM.
        local_enrichment=max(local_enrichment_by_strand.values(), default=0.0),
        local_enrichment_by_strand=local_enrichment_by_strand,
    )


def merge_forced_sites(
    automatic: Sequence[SiteTemplate],
    forced_intervals: Sequence[Tuple[int, int]],
    reads: Sequence[ReadEvidence],
    *,
    min_tq: int,
    center_radius: int,
    forced_only: bool,
) -> List[SiteTemplate]:
    """Union forced geometries with automatic sites, preferring forced edges."""
    sites = [] if forced_only else list(automatic)
    for interval in forced_intervals:
        center = (interval[0] + interval[1]) / 2.0
        sites = [site for site in sites if abs(site.center - center) > center_radius]
        sites.append(build_forced_site_template(
            reads,
            interval,
            site_id="forced",
            min_tq=min_tq,
            center_radius=center_radius,
        ))
    sites.sort(key=lambda site: (site.start, site.end))
    return [
        SiteTemplate(
            site_id=f"site{index}",
            start=site.start,
            end=site.end,
            center=site.center,
            support=site.support,
            all_support=site.all_support,
            median_tq=site.median_tq,
            start_mad=site.start_mad,
            end_mad=site.end_mad,
            local_enrichment=site.local_enrichment,
            local_enrichment_by_strand=site.local_enrichment_by_strand,
        )
        for index, site in enumerate(sites, start=1)
    ]


def shift_site_templates(
    sites: Sequence[SiteTemplate], shift: int
) -> List[SiteTemplate]:
    """Shift target geometry while retaining the true source-strand prior."""
    shifted = copy.deepcopy(list(sites))
    for site in shifted:
        site.start += int(shift)
        site.end += int(shift)
        site.center += int(shift)
        if site.start < 0:
            raise ValueError("shifted site starts before coordinate zero")
    return shifted


@dataclass(frozen=True)
class DirectConfigurationRecord:
    """One source molecule's explicit high-confidence TF configuration."""

    read_name: str
    strand: str
    calls: Tuple[Tuple[int, IntervalCall], ...]
    library_id: Optional[str] = None
    ref_start: Optional[int] = None
    ref_end: Optional[int] = None

    @property
    def site_indices(self) -> Tuple[int, ...]:
        return tuple(index for index, _ in self.calls)

    @property
    def molecule_id(self) -> Tuple[str, str]:
        """Identity used for within-cohort leave-one-molecule-out inference."""
        return (self.read_name, self.strand)


@dataclass(frozen=True)
class ConfigurationLibraryEntry:
    """Empirical boundary distribution for one projected TF configuration."""

    site_indices: Tuple[int, ...]
    member_intervals: Tuple[Tuple[Tuple[int, int], ...], ...]
    member_molecule_ids: Tuple[Tuple[str, str], ...] = ()

    @property
    def support(self) -> int:
        return len(self.member_intervals)

    @property
    def hulls(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(
            (
                min(start for start, _ in intervals),
                max(end for _, end in intervals),
            )
            for intervals in self.member_intervals
        )


def read_molecule_id(read: ReadEvidence) -> Tuple[str, str]:
    """Return a BAM-independent molecule identity within an input cohort.

    Input BAM paths are deliberately absent: repeat ``--bam`` arguments are
    explicitly pooled as one inference cohort.  The path remains available as
    ``library_id`` only for routing output records and for per-input amplified
    DAF duplicate collapse.
    """
    return (str(read.name), str(read.strand))


def unmatched_target_molecule_ids(
    source_reads: Sequence[ReadEvidence],
    target_reads: Sequence[ReadEvidence],
) -> List[Tuple[str, str]]:
    """Identify target molecules that would import an external population."""
    source_ids = {read_molecule_id(read) for read in source_reads}
    return sorted({
        read_molecule_id(read) for read in target_reads
        if read_molecule_id(read) not in source_ids
    })


def collapse_amplified_cohort_by_input(
    reads: Sequence[ReadEvidence],
    *,
    min_jaccard: float,
    min_deam: int,
) -> Tuple[List[ReadEvidence], dict]:
    """Collapse PCR families per input BAM, then pool the representatives.

    Input BAMs may be compatible timepoints in one inference cohort, but their
    independent molecules must never be declared PCR duplicates of one another.
    """
    groups: Dict[Optional[str], List[ReadEvidence]] = {}
    for read in reads:
        groups.setdefault(read.library_id, []).append(read)
    collapsed: List[ReadEvidence] = []
    by_input = {}
    for input_id, group in groups.items():
        representatives, diagnostics = _collapse_molecule_families(
            group,
            min_jaccard=min_jaccard,
            min_deam=min_deam,
            ignore_strand=False,
            num_hashes=32,
            bands=8,
            seed=7,
        )
        collapsed.extend(representatives)
        by_input[str(input_id or "UNKNOWN")] = diagnostics
    sum_keys = (
        "raw_reads", "analyzed_molecules", "fingerprintable_reads",
        "unfingerprintable_reads", "duplicate_reads_collapsed",
    )
    totals = {
        key: sum(int(values.get(key, 0)) for values in by_input.values())
        for key in sum_keys
    }
    fingerprintable = totals["fingerprintable_reads"]
    return collapsed, {
        "mode": "deamination_fingerprint_per_input_bam",
        **totals,
        "duplication_fraction": (
            totals["duplicate_reads_collapsed"] / fingerprintable
            if fingerprintable else 0.0
        ),
        "largest_family": max(
            (int(values.get("largest_family", 0)) for values in by_input.values()),
            default=0,
        ),
        "min_jaccard": min_jaccard,
        "min_deam": min_deam,
        "by_input_bam": by_input,
    }


def exclude_configuration_molecule(
    entries: Sequence[ConfigurationLibraryEntry],
    molecule_id: Tuple[str, str],
    *,
    min_support: int,
) -> List[ConfigurationLibraryEntry]:
    """Remove one target molecule from empirical configuration geometry."""
    filtered = []
    for entry in entries:
        if not entry.member_molecule_ids:
            # Backward-compatible hand-built entries used by low-level callers
            # contain no source identities and therefore cannot be filtered.
            filtered.append(entry)
            continue
        keep = [
            index for index, member_id in enumerate(entry.member_molecule_ids)
            if member_id != molecule_id
        ]
        if len(keep) < min_support:
            continue
        filtered.append(ConfigurationLibraryEntry(
            site_indices=entry.site_indices,
            member_intervals=tuple(entry.member_intervals[index] for index in keep),
            member_molecule_ids=tuple(
                entry.member_molecule_ids[index] for index in keep
            ),
        ))
    return filtered


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    """One-sided conservative lower bound for a binomial proportion.

    High-confidence explicit TF configurations are treated as positive
    observations.  Uncalled spanning molecules remain in the denominator, so
    incomplete enzymatic saturation can only make this a lower bound on focal
    TF-complex occupancy rather than an optimistic occupancy estimate.
    """
    successes = int(successes)
    total = int(total)
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("binomial counts must satisfy 0 <= successes <= total")
    if z < 0.0:
        raise ValueError("Wilson z must be non-negative")
    if total == 0 or successes == 0:
        return 0.0
    fraction = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = fraction + z2 / (2.0 * total)
    radius = z * math.sqrt(
        fraction * (1.0 - fraction) / total + z2 / (4.0 * total * total)
    )
    return max(0.0, min(1.0, (center - radius) / denominator))


def estimate_local_complex_prior(
    source_reads: Sequence[ReadEvidence],
    records: Sequence[DirectConfigurationRecord],
    relevant_site_indices: Sequence[int],
    sites: Sequence[SiteTemplate],
    *,
    strand: Optional[str],
    exclude_molecule_id: Optional[Tuple[str, str]],
    z: float,
) -> dict:
    """Estimate a same-cohort lower bound on focal TF-complex occupancy.

    Only molecules spanning the full relevant template hull enter the
    denominator.  This is important for short Nanopore Hia5 reads: depth at one
    edge is not mistaken for evidence about a multi-site complex that the read
    cannot span.  All explicitly pooled input BAMs form one cohort.  The
    candidate molecule, rather than its entire input BAM, is excluded.
    """
    relevant = set(int(index) for index in relevant_site_indices)
    if not relevant:
        return {
            "spanning_molecules": 0,
            "explicit_complex_molecules": 0,
            "explicit_fraction": 0.0,
            "wilson_lower_bound": 0.0,
            "wilson_z": float(z),
            "inference_scope": "explicitly_pooled_input_cohort",
            "candidate_molecule_excluded": exclude_molecule_id is not None,
        }
    hull_start = min(sites[index].start for index in relevant)
    hull_end = max(sites[index].end for index in relevant)
    positives = {
        record.molecule_id
        for record in records
        if (strand is None or record.strand == strand)
        and record.molecule_id != exclude_molecule_id
        and relevant.intersection(record.site_indices)
    }
    spanning = {
        read_molecule_id(read)
        for read in source_reads
        if (strand is None or read.strand == strand)
        and read_molecule_id(read) != exclude_molecule_id
        and read.ref_start <= hull_start
        and read.ref_end >= hull_end
    }
    successes = len(spanning.intersection(positives))
    total = len(spanning)
    return {
        "template_hull": [int(hull_start), int(hull_end)],
        "spanning_molecules": total,
        "explicit_complex_molecules": successes,
        "explicit_fraction": (successes / total if total else 0.0),
        "wilson_lower_bound": wilson_lower_bound(successes, total, z),
        "wilson_z": float(z),
        "inference_scope": "explicitly_pooled_input_cohort",
        "candidate_molecule_excluded": exclude_molecule_id is not None,
    }


def _normal_logpdf(delta: np.ndarray, bandwidth: float) -> np.ndarray:
    return -0.5 * (delta / bandwidth) ** 2 - math.log(
        bandwidth * math.sqrt(2.0 * math.pi)
    )


def edge_log_density(
    call: IntervalCall,
    hulls: Sequence[Tuple[int, int]],
    bandwidth: float,
) -> float:
    """Two-dimensional Gaussian KDE for a protected block's outer edges."""
    if bandwidth <= 0:
        raise ValueError("edge bandwidth must be positive")
    if not hulls:
        return -math.inf
    array = np.asarray(hulls, dtype=np.float64)
    values = (
        _normal_logpdf(call.start - array[:, 0], bandwidth)
        + _normal_logpdf(call.end - array[:, 1], bandwidth)
    )
    return float(logsumexp(values) - math.log(len(values)))


def calibrate_edge_bandwidth(
    records: Sequence[DirectConfigurationRecord],
    bandwidths: Sequence[float],
    *,
    strand_specific: bool,
    fallback: float,
) -> dict:
    """Choose the MA-edge KDE width within the pooled inference cohort.

    Every explicit observation is scored against other molecules in the same
    explicitly pooled cohort. Input-BAM identity never chooses the training
    set. Ambiguous nuc calls never participate.
    """
    grid = sorted({float(value) for value in bandwidths})
    if any(value <= 0.0 for value in grid) or fallback <= 0.0:
        raise ValueError("edge bandwidths must be positive")
    groups: Dict[Tuple[Optional[str], Tuple[int, ...]], List[dict]] = {}
    for record in records:
        if not record.calls:
            continue
        hull = (
            min(call.start for _, call in record.calls),
            max(call.end for _, call in record.calls),
        )
        key = (
            record.strand if strand_specific else None,
            record.site_indices,
        )
        groups.setdefault(key, []).append({
            "hull": hull,
            "molecule_id": record.molecule_id,
        })
    holdout_mode = "leave_one_molecule_out"
    scores = []
    for bandwidth in grid:
        total = 0.0
        observations = 0
        for items in groups.values():
            if len(items) < 2:
                continue
            for item in items:
                training = [
                    other["hull"] for other in items
                    if other["molecule_id"] != item["molecule_id"]
                ]
                if not training:
                    continue
                total += edge_log_density(
                    IntervalCall(*item["hull"]), training, bandwidth
                )
                observations += 1
        scores.append({
            "bandwidth": bandwidth,
            "heldout_log_likelihood": total,
            "heldout_observations": observations,
        })
    eligible = [row for row in scores if row["heldout_observations"] > 0]
    selected = (
        max(
            eligible,
            key=lambda row: (
                row["heldout_log_likelihood"], row["bandwidth"]
            ),
        )["bandwidth"]
        if eligible else fallback
    )
    return {
        "selected_bandwidth": float(selected),
        "fallback_bandwidth": float(fallback),
        "selection": (
            "explicit_configuration_heldout_likelihood"
            if eligible else "fallback_no_eligible_configuration"
        ),
        "holdout_mode": holdout_mode,
        "scores": scores,
    }


def length_log_density(
    length: int,
    control_lengths: Sequence[int],
    bandwidth: float,
) -> float:
    """One-dimensional Gaussian KDE for background nucleosome lengths."""
    if bandwidth <= 0:
        raise ValueError("length bandwidth must be positive")
    if not control_lengths:
        # Weak fallback centered on a canonical 147 bp footprint.
        return float(_normal_logpdf(np.asarray(length - 147.0), 30.0))
    delta = length - np.asarray(control_lengths, dtype=np.float64)
    values = _normal_logpdf(delta, bandwidth)
    return float(logsumexp(values) - math.log(len(values)))


def accessible_hit_probabilities(model) -> np.ndarray:
    """Context-specific P(hard hit | accessible) from a two-state model."""
    emissions = np.asarray(model.emissionprob_, dtype=np.float64)
    hit = emissions[1, :N_CTX]
    miss = emissions[1, UNMETH_OFFSET:UNMETH_OFFSET + N_CTX]
    denominator = np.maximum(hit + miss, 1e-30)
    return np.clip(hit / denominator, 1e-9, 1.0 - 1e-9)


def interval_no_hit_probability(
    read: ReadEvidence,
    start: int,
    end: int,
    accessible_hit_probability: np.ndarray,
) -> Tuple[float, int, int]:
    """P(no hard accessible hit) for the actual opportunities in an interval."""
    lo = int(np.searchsorted(read.positions, start, side="left"))
    hi = int(np.searchsorted(read.positions, end, side="left"))
    if hi <= lo:
        return 1.0, 0, 0
    contexts = read.contexts[lo:hi]
    log_probability = float(np.sum(np.log1p(-accessible_hit_probability[contexts])))
    return (
        float(math.exp(max(log_probability, -745.0))),
        int(hi - lo),
        int(np.sum(read.hits[lo:hi])),
    )


def match_high_confidence_calls(
    read: ReadEvidence,
    sites: Sequence[SiteTemplate],
    *,
    min_tq: int,
    center_radius: int,
) -> Dict[int, IntervalCall]:
    """Map explicit high-TQ calls to one non-overlapping template each."""
    high_calls = [call for call in read.tfs if call.score >= min_tq]
    selected = match_direct_site_indices(
        high_calls, sites, center_radius=center_radius
    )
    result: Dict[int, IntervalCall] = {}
    for index in selected:
        site = sites[index]
        candidates = [
            call for call in high_calls
            if abs(call.center - site.center) <= center_radius
        ]
        if candidates:
            result[index] = min(
                candidates,
                key=lambda call: (
                    abs(call.center - site.center),
                    abs((call.end - call.start) - (site.end - site.start)),
                ),
            )
    return result


def build_direct_configuration_records(
    reads: Sequence[ReadEvidence],
    sites: Sequence[SiteTemplate],
    *,
    min_tq: int,
    center_radius: int,
) -> List[DirectConfigurationRecord]:
    """Freeze the pooled inference cohort using explicit calls only.

    Nuc-like and accessible molecules do not contribute a latent label and
    therefore cannot reinforce either TF or N priors.
    """
    records = []
    for read in reads:
        matched = match_high_confidence_calls(
            read, sites, min_tq=min_tq, center_radius=center_radius
        )
        if not matched:
            continue
        records.append(DirectConfigurationRecord(
            read_name=read.name,
            strand=read.strand,
            calls=tuple(sorted(matched.items())),
            library_id=read.library_id,
            ref_start=read.ref_start,
            ref_end=read.ref_end,
        ))
    return records


def project_configuration_library(
    records: Sequence[DirectConfigurationRecord],
    relevant_site_indices: Sequence[int],
    *,
    min_support: int,
    strand: Optional[str] = None,
    exclude_molecule_id: Optional[Tuple[str, str]] = None,
    sites: Optional[Sequence[SiteTemplate]] = None,
) -> List[ConfigurationLibraryEntry]:
    """Marginalize explicit source configurations onto one candidate block."""
    relevant = set(relevant_site_indices)
    template_hull = (
        (
            min(sites[index].start for index in relevant),
            max(sites[index].end for index in relevant),
        )
        if sites is not None and relevant else None
    )
    members_by_config: Dict[
        Tuple[int, ...], Dict[Tuple[str, str], Tuple[Tuple[int, int], ...]]
    ] = {}
    for record in records:
        if strand is not None and record.strand != strand:
            continue
        if record.molecule_id == exclude_molecule_id:
            continue
        if (
            template_hull is not None
            and record.ref_start is not None
            and record.ref_end is not None
            and not (
                record.ref_start <= template_hull[0]
                and record.ref_end >= template_hull[1]
            )
        ):
            continue
        projected = [(index, call) for index, call in record.calls if index in relevant]
        if not projected:
            continue
        config = tuple(index for index, _ in projected)
        member = tuple(
            (call.start, call.end)
            for _, call in projected
        )
        # The same molecule may occur in more than one input shard. It remains
        # one observation in the explicitly pooled inference cohort.
        members_by_config.setdefault(config, {}).setdefault(
            record.molecule_id, member
        )
    entries = []
    for config, members in members_by_config.items():
        if len(members) < min_support:
            continue
        ordered = sorted(members.items())
        entries.append(ConfigurationLibraryEntry(
            site_indices=config,
            member_intervals=tuple(member for _, member in ordered),
            member_molecule_ids=tuple(molecule_id for molecule_id, _ in ordered),
        ))
    return sorted(entries, key=lambda entry: (len(entry.site_indices), entry.site_indices))


def shift_configuration_library(
    entries: Sequence[ConfigurationLibraryEntry],
    shift: int,
) -> List[ConfigurationLibraryEntry]:
    """Create a coordinate-shifted boundary decoy while preserving support."""
    if shift == 0:
        return list(entries)
    return [
        ConfigurationLibraryEntry(
            entry.site_indices,
            tuple(
                tuple((start + shift, end + shift) for start, end in member)
                for member in entry.member_intervals
            ),
            entry.member_molecule_ids,
        )
        for entry in entries
    ]


def collect_background_nuc_lengths(
    reads: Sequence[ReadEvidence],
    sites: Sequence[SiteTemplate],
    *,
    exclusion_distance: int,
    min_length: int = 90,
    max_length: int = 220,
    strand: Optional[str] = None,
) -> List[int]:
    """Collect opportunity-matched N controls away from every focal template."""
    lengths = []
    for read in reads:
        if strand is not None and read.strand != strand:
            continue
        for call in read.nucs:
            length = call.end - call.start
            if not min_length <= length <= max_length:
                continue
            if any(abs(call.center - site.center) < exclusion_distance for site in sites):
                continue
            lengths.append(length)
    return lengths


def collect_local_nuc_geometries(
    reads: Sequence[ReadEvidence],
    relevant_site_indices: Sequence[int],
    sites: Sequence[SiteTemplate],
    *,
    min_length: int,
    max_length: int,
    strand: Optional[str] = None,
    exclude_molecule_id: Optional[Tuple[str, str]] = None,
) -> List[Tuple[int, int]]:
    """Source-observed N geometries covering the same focal sites.

    This is the exact N analogue of the empirical TF configuration library.
    The TF hypothesis draws its geometries from source molecules that carry an
    explicit TF call here; N must draw its geometries from source molecules
    that carry an explicit nuc call here.  Giving N a vague uniform prior over
    every arithmetically possible dyad position while TF gets a sharp fitted
    one is a model-selection artifact, not evidence: it charges N an Occam
    penalty that TF never pays and displaces the whole N:TF prior axis.
    """
    return [
        geometry for geometry, _ in collect_local_nuc_geometry_records(
            reads, relevant_site_indices, sites,
            min_length=min_length, max_length=max_length, strand=strand,
            exclude_molecule_id=exclude_molecule_id,
        )
    ]


def collect_local_nuc_geometry_records(
    reads: Sequence[ReadEvidence],
    relevant_site_indices: Sequence[int],
    sites: Sequence[SiteTemplate],
    *,
    min_length: int,
    max_length: int,
    strand: Optional[str] = None,
    exclude_molecule_id: Optional[Tuple[str, str]] = None,
) -> List[Tuple[Tuple[int, int], Tuple[str, str]]]:
    """As above, retaining each geometry's source molecule for holdout."""
    relevant = [int(index) for index in relevant_site_indices]
    if not relevant:
        return []
    left = min(sites[index].center for index in relevant)
    right = max(sites[index].center for index in relevant)
    records = []
    for read in reads:
        if strand is not None and read.strand != strand:
            continue
        molecule_id = read_molecule_id(read)
        if exclude_molecule_id is not None and molecule_id == exclude_molecule_id:
            continue
        for call in read.nucs:
            if not min_length <= call.end - call.start <= max_length:
                continue
            # Must be capable of covering every focal site, which is the same
            # eligibility rule the enumerated fallback applies.
            if call.start > left or call.end <= right:
                continue
            records.append(((int(call.start), int(call.end)), molecule_id))
    return records


def integrated_nuc_log_likelihood(
    read: ReadEvidence,
    call: IntervalCall,
    relevant_site_indices: Sequence[int],
    sites: Sequence[SiteTemplate],
    control_lengths: Sequence[int],
    edge_bandwidth: float,
    nuc_scorer: Optional[DddaRadialNucScorer],
    observed_geometries: Optional[Sequence[Tuple[int, int]]] = None,
    min_observed_geometries: int = 0,
) -> dict:
    """Marginalize N over its geometry prior, symmetrically with TF.

    When the source population supplies enough explicit nuc geometries at this
    locus, N is marginalized over them with the same ``-log(support)``
    normalization and the same Gaussian call-edge kernel that every empirical
    TF configuration uses.  Both hypotheses then carry comparably sharp,
    data-fitted geometry priors, so the N:TF Bayes factor reflects the
    molecule's bases rather than how vaguely each prior happened to be written
    down.  The uniform-enumeration path below is retained only as a fallback
    when the locus has too few observed nuc geometries to fit one.
    """
    if (
        observed_geometries is not None
        and len(observed_geometries) >= max(1, min_observed_geometries)
    ):
        array = np.asarray(observed_geometries, dtype=np.int64)
        if nuc_scorer is None:
            prefix = np.concatenate(
                ([0.0], np.cumsum(read.steps, dtype=np.float64))
            )
            lo = np.searchsorted(read.positions, array[:, 0], side="left")
            hi = np.searchsorted(read.positions, array[:, 1], side="left")
            values = prefix[hi] - prefix[lo]
        else:
            values = np.asarray([
                nuc_scorer.score(read, IntervalCall(int(start), int(end)))
                for start, end in array
            ], dtype=np.float64)
        normalizer = math.log(len(array))
        edge = values + (
            _normal_logpdf(call.start - array[:, 0], edge_bandwidth)
            + _normal_logpdf(call.end - array[:, 1], edge_bandwidth)
        )
        representative = int(np.argmax(edge))
        return {
            "raw_log_likelihood": float(logsumexp(values) - normalizer),
            "edge_conditioned_log_likelihood": float(
                logsumexp(edge) - normalizer
            ),
            "n_geometries": int(len(array)),
            "representative_interval": [
                int(array[representative, 0]), int(array[representative, 1])
            ],
            "representative_llr": float(values[representative]),
            "used_current_call_fallback": False,
            "geometry_prior": "source_observed_nuc_calls",
        }
    if control_lengths:
        unique, counts = np.unique(
            np.asarray(control_lengths, dtype=int), return_counts=True
        )
        length_counts = list(zip(unique.tolist(), counts.tolist()))
    else:
        length_counts = [(147, 1)]
    left_center = min(sites[index].center for index in relevant_site_indices)
    right_center = max(sites[index].center for index in relevant_site_indices)
    eligible = []
    for length, count in length_counts:
        if nuc_scorer is not None and not nuc_scorer.eligible(
            IntervalCall(0, length)
        ):
            continue
        start_min = max(
            read.ref_start,
            int(math.floor(right_center)) - length + 1,
        )
        start_max = min(
            int(math.floor(left_center)),
            read.ref_end - length,
        )
        if start_max < start_min:
            continue
        eligible.append((length, count, start_min, start_max))
    if not eligible:
        # A short read may not span any background-derived canonical length,
        # but its existing nuc call is itself a geometrically valid fallback.
        # Never turn absence of an eligible N control into posterior TF=1.
        fallback_llr = _nuc_base_llr(read, call, nuc_scorer)
        edge_term = 2.0 * float(
            _normal_logpdf(np.asarray(0.0), edge_bandwidth)
        )
        return {
            "raw_log_likelihood": fallback_llr,
            "edge_conditioned_log_likelihood": fallback_llr + edge_term,
            "n_geometries": 1,
            "representative_interval": [call.start, call.end],
            "representative_llr": fallback_llr,
            "used_current_call_fallback": True,
            "geometry_prior": "current_call_fallback",
        }

    total_length_count = float(sum(count for _, count, _, _ in eligible))
    raw_chunks = []
    edge_chunks = []
    start_chunks = []
    end_chunks = []
    llr_chunks = []
    prefix = np.concatenate(([0.0], np.cumsum(read.steps, dtype=np.float64)))
    for length, count, start_min, start_max in eligible:
        starts = np.arange(start_min, start_max + 1, dtype=np.int64)
        ends = starts + length
        n_starts = len(starts)
        log_geometry_prior = math.log(count / total_length_count) - math.log(n_starts)
        if nuc_scorer is None:
            lo = np.searchsorted(read.positions, starts, side="left")
            hi = np.searchsorted(read.positions, ends, side="left")
            values = prefix[hi] - prefix[lo]
        else:
            values = np.asarray([
                nuc_scorer.score(read, IntervalCall(int(start), int(end)))
                for start, end in zip(starts, ends)
            ], dtype=np.float64)
        raw = log_geometry_prior + values
        edge = raw + (
            _normal_logpdf(call.start - starts, edge_bandwidth)
            + _normal_logpdf(call.end - ends, edge_bandwidth)
        )
        raw_chunks.append(raw)
        edge_chunks.append(edge)
        start_chunks.append(starts)
        end_chunks.append(ends)
        llr_chunks.append(values)
    raw_log_terms = np.concatenate(raw_chunks)
    edge_log_terms = np.concatenate(edge_chunks)
    starts = np.concatenate(start_chunks)
    ends = np.concatenate(end_chunks)
    llrs = np.concatenate(llr_chunks)
    representative = int(np.argmax(edge_log_terms))
    return {
        "raw_log_likelihood": float(logsumexp(raw_log_terms)),
        "edge_conditioned_log_likelihood": float(logsumexp(edge_log_terms)),
        "n_geometries": len(raw_log_terms),
        "representative_interval": [
            int(starts[representative]),
            int(ends[representative]),
        ],
        "representative_llr": float(llrs[representative]),
        "used_current_call_fallback": False,
        "geometry_prior": "uniform_enumeration",
    }


def _nuc_base_llr(
    read: ReadEvidence,
    call: IntervalCall,
    nuc_scorer: Optional[DddaRadialNucScorer],
) -> float:
    if nuc_scorer is not None:
        return nuc_scorer.score(read, call)
    return read.interval_evidence(call.start, call.end)[0]


def configuration_base_evidence(
    read: ReadEvidence,
    call: IntervalCall,
    site_indices: Sequence[int],
    sites: Sequence[SiteTemplate],
) -> Tuple[float, List[dict]]:
    """Protected-vs-accessible LLR for every TF in a configuration."""
    total = 0.0
    evidence = []
    for index in site_indices:
        site = sites[index]
        start = max(call.start, site.start)
        end = min(call.end, site.end)
        llr, opportunities, hits = read.interval_evidence(start, end)
        total += llr
        evidence.append({
            "site": site.site_id,
            "interval": [start, end],
            "llr": llr,
            "opportunities": opportunities,
            "hits": hits,
        })
    return total, evidence


def configuration_member_evidence(
    read: ReadEvidence,
    call: IntervalCall,
    site_indices: Sequence[int],
    member_intervals: Sequence[Tuple[int, int]],
    sites: Sequence[SiteTemplate],
) -> Tuple[float, List[dict]]:
    """Target likelihood under one complete source-observed TF geometry."""
    total = 0.0
    evidence = []
    for index, (member_start, member_end) in zip(site_indices, member_intervals):
        start = member_start
        end = member_end
        llr, opportunities, hits = read.interval_evidence(start, end)
        total += llr
        evidence.append({
            "site": sites[index].site_id,
            "interval": [start, end],
            "source_interval": [member_start, member_end],
            "llr": llr,
            "opportunities": opportunities,
            "hits": hits,
        })
    return total, evidence


def composite_segment_evidence(
    read: ReadEvidence,
    call: IntervalCall,
    site_indices: Sequence[int],
    member_intervals: Sequence[Tuple[int, int]],
    sites: Sequence[SiteTemplate],
    protected_gap_prior: float,
    protected_flank_prior: float,
    nuc_scorer: Optional[DddaRadialNucScorer] = None,
) -> dict:
    """Log Bayes factor of a potential TF composite against one nucleosome.

    Both hypotheses explain the *same* bases -- every base of the candidate
    block -- so nothing is conditioned on the block's boundaries.  Those
    boundaries came from the upstream single-molecule segmentation, which is
    precisely the call under suspicion; it cannot also be the evidence.

    Under N every base is protected.  Under the composite the TF footprints are
    protected and each *flexible* segment (a bridge between footprints, or a
    flank between a footprint and the block edge) is marginalized over protected
    and accessible with an explicit prior.  A complex is allowed to protect more
    than the population happened to methylate around it: what is tested is the
    block's *potential* to be a composite, not the existence of one exact
    observed composite.

    Because the protected-vs-accessible LLR is additive over bases, the TF
    footprints cancel against N and the whole comparison collapses to a sum over
    flexible segments::

        log BF = sum_seg log( p_seg + (1 - p_seg) * exp(-LLR_seg) )

    An accessible segment drives this positive; a segment with no informative
    opportunities contributes exactly zero, so the decision falls back to the
    population prior rather than being penalized for enzyme dropout; and a
    protected segment costs at most ``log(p_seg)`` -- a bounded prior penalty,
    never an unbounded likelihood collapse.
    """
    if not 0.0 <= protected_gap_prior <= 1.0:
        raise ValueError("protected gap prior must lie in [0, 1]")
    if not 0.0 <= protected_flank_prior <= 1.0:
        raise ValueError("protected flank prior must lie in [0, 1]")
    ordered = sorted(
        zip(site_indices, member_intervals),
        key=lambda item: (item[1][0], item[1][1]),
    )
    ordered_indices = [index for index, _ in ordered]
    ordered_intervals = [
        (max(call.start, start), min(call.end, end))
        for _, (start, end) in ordered
    ]
    footprint_base, site_evidence = configuration_member_evidence(
        read, call, ordered_indices, ordered_intervals, sites
    )

    # Every base of the block that a TF footprint does not claim is flexible.
    segments: List[dict] = []
    cursor = call.start
    for start, end in ordered_intervals:
        if start > cursor:
            segments.append({"interval": (cursor, start), "kind": "flank"
                             if not segments and cursor == call.start
                             else "bridge"})
        cursor = max(cursor, end)
    if cursor < call.end:
        segments.append({"interval": (cursor, call.end), "kind": "flank"})
    # Interior gaps between two footprints are bridges; the outermost ones are
    # flanks.  Re-label using position rather than discovery order.
    if ordered_intervals:
        first_start = ordered_intervals[0][0]
        last_end = max(end for _, end in ordered_intervals)
        for segment in segments:
            lo, hi = segment["interval"]
            segment["kind"] = (
                "flank" if hi <= first_start or lo >= last_end else "bridge"
            )

    log_bf = 0.0
    for segment in segments:
        lo, hi = segment["interval"]
        llr, opportunities, hits = read.interval_evidence(lo, hi)
        prior = (
            protected_gap_prior if segment["kind"] == "bridge"
            else protected_flank_prior
        )
        # log( p + (1-p) * exp(-llr) ), computed stably.
        terms = []
        if prior > 0.0:
            terms.append(math.log(prior))
        if prior < 1.0:
            terms.append(math.log1p(-prior) - llr)
        contribution = float(logsumexp(np.asarray(terms))) if terms else 0.0
        posterior_protected = (
            float(math.exp(math.log(prior) - contribution))
            if prior > 0.0 and math.isfinite(contribution) else 0.0
        )
        segment.update({
            "interval": [int(lo), int(hi)],
            "llr_n_vs_accessible": float(llr),
            "opportunities": int(opportunities),
            "hits": int(hits),
            "prior_protected": float(prior),
            "log_bf_contribution": contribution,
            "posterior_protected": max(0.0, min(1.0, posterior_protected)),
            "informative": bool(opportunities > 0),
        })
        log_bf += contribution

    # DddA scores N with a radial dyad profile rather than a flat protected
    # state, so the footprint terms no longer cancel and must be carried.
    footprint_correction = 0.0
    if nuc_scorer is not None:
        nuc_base = nuc_scorer.score(read, call)
        flat_base = read.interval_evidence(call.start, call.end)[0]
        footprint_correction = flat_base - nuc_base
        log_bf += footprint_correction

    resolvable = all(
        segment["posterior_protected"] < 0.5 for segment in segments
    ) if segments else False
    informative = [s for s in segments if s["informative"]]
    return {
        "log_bf_vs_n": float(log_bf),
        "footprint_base_llr": float(footprint_base),
        "footprint_correction": float(footprint_correction),
        "site_evidence": site_evidence,
        "segments": segments,
        "flexible_segments": len(segments),
        "informative_segments": len(informative),
        "informative_opportunities": sum(
            s["opportunities"] for s in segments
        ),
        "informative_hits": sum(s["hits"] for s in segments),
        "all_segments_accessible": bool(resolvable),
        "effective_intervals": [list(interval) for interval in ordered_intervals],
    }


def member_protection_mixture(
    read: ReadEvidence,
    call: IntervalCall,
    site_indices: Sequence[int],
    member_intervals: Sequence[Tuple[int, int]],
    sites: Sequence[SiteTemplate],
    protected_gap_prior: float,
) -> dict:
    """Retained for the historical bridge-only model and its tests."""
    if not 0.0 <= protected_gap_prior <= 1.0:
        raise ValueError("protected gap prior must lie in [0, 1]")
    ordered = sorted(
        zip(site_indices, member_intervals),
        key=lambda item: (item[1][0], item[1][1]),
    )
    ordered_indices = [index for index, _ in ordered]
    ordered_intervals = [interval for _, interval in ordered]
    component_base, site_evidence = configuration_member_evidence(
        read, call, ordered_indices, ordered_intervals, sites
    )
    internal_gaps = []
    for left, right in zip(ordered_intervals, ordered_intervals[1:]):
        start = left[1]
        end = right[0]
        if end > start:
            llr, opportunities, hits = read.interval_evidence(start, end)
            internal_gaps.append({
                "interval": (start, end),
                "llr_n_vs_accessible": llr,
                "opportunities": opportunities,
                "hits": hits,
            })

    log_terms = []
    masks = []
    n_gaps = len(internal_gaps)
    for mask in range(1 << n_gaps):
        protected_count = int(mask.bit_count())
        accessible_count = n_gaps - protected_count
        if protected_gap_prior == 0.0 and protected_count:
            continue
        if protected_gap_prior == 1.0 and accessible_count:
            continue
        log_prior = 0.0
        if protected_count:
            log_prior += protected_count * math.log(protected_gap_prior)
        if accessible_count:
            log_prior += accessible_count * math.log1p(-protected_gap_prior)
        value = component_base + sum(
            gap["llr_n_vs_accessible"]
            for index, gap in enumerate(internal_gaps)
            if mask & (1 << index)
        )
        log_terms.append(log_prior + value)
        masks.append(mask)
    marginal = float(logsumexp(log_terms))
    best_position = int(np.argmax(log_terms))
    best_mask = masks[best_position]
    # A fully bridged complex protects every base the nucleosome would, so it
    # predicts exactly the same observations as N and cannot be evidence for
    # splitting anything.  Only the all-accessible-separator term is refutable
    # by the molecule's own bases, so it is kept separate from the bridged mass
    # and is what the strict tier is allowed to act on.
    resolvable = next(
        (value for value, mask in zip(log_terms, masks) if mask == 0),
        -math.inf,
    )

    effective_intervals: List[Tuple[int, int]] = []
    if ordered_intervals:
        current_start, current_end = ordered_intervals[0]
        gap_index = 0
        for next_start, next_end in ordered_intervals[1:]:
            if next_start <= current_end:
                current_end = max(current_end, next_end)
                continue
            is_bridged = bool(best_mask & (1 << gap_index))
            gap_index += 1
            if is_bridged:
                current_end = next_end
            else:
                effective_intervals.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        effective_intervals.append((current_start, current_end))

    return {
        "marginal_base_llr": marginal,
        "resolvable_base_llr": float(resolvable),
        "site_evidence": site_evidence,
        "internal_gaps": [
            {
                **gap,
                "interval": list(gap["interval"]),
                "representative_state": (
                    "protected_bridge"
                    if best_mask & (1 << index)
                    else "accessible_separator"
                ),
            }
            for index, gap in enumerate(internal_gaps)
        ],
        "representative_bridge_mask": best_mask,
        "representative_effective_intervals": effective_intervals,
    }


def gap_diagnostics(
    read: ReadEvidence,
    call: IntervalCall,
    site_indices: Sequence[int],
    sites: Sequence[SiteTemplate],
    accessible_hit_probability: np.ndarray,
    *,
    member_intervals: Optional[Sequence[Tuple[int, int]]] = None,
) -> dict:
    """Describe bases that distinguish a TF tiling from continuous N."""
    raw_intervals = (
        member_intervals
        if member_intervals is not None
        else [(sites[index].start, sites[index].end) for index in site_indices]
    )
    intervals = sorted(
        (max(call.start, start), min(call.end, end))
        for start, end in raw_intervals
    )
    gaps: List[Tuple[int, int, str]] = []
    cursor = call.start
    for start, end in intervals:
        if start > cursor:
            gaps.append((cursor, start, "outer" if cursor == call.start else "internal"))
        cursor = max(cursor, end)
    if cursor < call.end:
        gaps.append((cursor, call.end, "outer"))

    records = []
    joint_no_hit = 1.0
    total_opportunities = 0
    total_hits = 0
    total_llr = 0.0
    for start, end, kind in gaps:
        llr, opportunities, hits = read.interval_evidence(start, end)
        no_hit, _, _ = interval_no_hit_probability(
            read, start, end, accessible_hit_probability
        )
        joint_no_hit *= no_hit
        total_opportunities += opportunities
        total_hits += hits
        total_llr += llr
        records.append({
            "interval": [start, end],
            "kind": kind,
            "llr_n_vs_accessible": llr,
            "opportunities": opportunities,
            "hits": hits,
            "p_no_hit_if_accessible": no_hit,
        })
    return {
        "gaps": records,
        "outside_tf_opportunities": total_opportunities,
        "outside_tf_hits": total_hits,
        "outside_tf_llr_n_vs_accessible": total_llr,
        "p_no_hit_if_all_gaps_accessible": joint_no_hit,
    }


def reconstruct_nuc_call(
    read: ReadEvidence,
    call: IntervalCall,
    sites: Sequence[SiteTemplate],
    library: Sequence[ConfigurationLibraryEntry],
    background_lengths: Sequence[int],
    *,
    spanning_molecules: int,
    nuc_support: int,
    reconstruction_tolerance: int,
    length_bandwidth: float,
    position_window: float,
    nuc_prior_odds: float,
    prior_alpha: float,
) -> dict:
    """Propose alternative annotations for a nuc call, from consensus geometry.

    This pass does not try to prove the nuc call wrong.  It asks whether the
    block can be *reconstructed* as the union of TF footprints that the
    population actually shows at this site -- the ``-[]---`` plus ``---[]-``
    giving ``-[    ]-`` case.  The merged footprints leave no gap, so the two
    explanations predict the *same protected bases*: the molecule's own data can
    only ever veto a reconstruction (m6A inside a proposed footprint), never
    confirm it.  Support therefore comes from geometry and population frequency.

    For each configuration the population supports, the union hull of its
    observed instances is compared with the block's extent.  A reconstruction is
    proposed only when that union reproduces the extent within
    ``reconstruction_tolerance``.  Blocks no union reproduces retain N silently.
    """
    if reconstruction_tolerance < 0:
        raise ValueError("reconstruction tolerance must be non-negative")
    if position_window <= 0:
        raise ValueError("position window must be positive")

    length = call.end - call.start
    # A nucleosome may sit anywhere, so its extent costs a position term; a union
    # is pinned by its binding sites.  That asymmetry is the evidence, and it is
    # bounded by log(position_window) rather than an unbounded Gaussian.
    nuc_geometry = (
        length_log_density(length, background_lengths, length_bandwidth)
        - math.log(position_window)
    )

    proposals = []
    for entry in library:
        hulls = np.asarray(entry.hulls, dtype=np.float64)
        union_start = float(np.median(hulls[:, 0]))
        union_end = float(np.median(hulls[:, 1]))
        bandwidth = max(1.0, float(reconstruction_tolerance) / 2.0)

        # The TF footprints must reproduce the whole block's extent: this is the
        # -[]--- + ---[]- -> -[  ]- union.  A block the union cannot span is left
        # as N; we do not invent a nucleosome-plus-edge-TF to cover the excess.
        residual = abs(union_start - call.start) + abs(union_end - call.end)
        if residual > reconstruction_tolerance:
            continue

        # Density of the observed union hulls at this block's exact extent.
        union_geometry = edge_log_density(
            call, [tuple(hull) for hull in entry.hulls], bandwidth=bandwidth,
        )
        # The molecule may only VETO: accessibility inside a proposed footprint
        # contradicts that TF being bound.  Absence of m6A in the merged interior
        # is enzyme dropout and is never required.
        veto = 0.0
        footprints = []
        for index, interval in zip(
            entry.site_indices, entry.member_intervals[0]
        ):
            start = max(call.start, int(interval[0]))
            end = min(call.end, int(interval[1]))
            llr, opportunities, hits = read.interval_evidence(start, end)
            veto += min(0.0, llr)
            footprints.append({
                "site": sites[index].site_id,
                "interval": [start, end],
                "llr_protected": float(llr),
                "opportunities": int(opportunities),
                "hits": int(hits),
                "contradicted": bool(llr < 0.0),
            })
        frequency = entry.support / max(1, spanning_molecules)
        tf_label = ",".join(
            sites[index].site_id for index in entry.site_indices
        )
        proposals.append({
            "state": f"TF:{tf_label}",
            "site_indices": list(entry.site_indices),
            "union_interval": [int(round(union_start)), int(round(union_end))],
            "reconstruction_residual": float(residual),
            "source_support": entry.support,
            "population_frequency": float(frequency),
            "geometry_log_bf_vs_n": float(union_geometry - nuc_geometry),
            "molecule_veto_log_bf": float(veto),
            "log_support": float(
                math.log(entry.support + prior_alpha)
                + union_geometry - nuc_geometry + veto
            ),
            "footprints": footprints,
            "footprints_contradicted": sum(
                1 for item in footprints if item["contradicted"]
            ),
        })

    # N must be scored on the same footing: how often the population actually
    # shows a nucleosome here, not a bare prior.  Otherwise a configuration seen
    # 23 times wins on log(23) alone and every reconstruction reads as certain.
    nuc_log_score = (
        math.log(nuc_prior_odds) + math.log(max(nuc_support + prior_alpha, 1e-9))
    )
    log_scores = [nuc_log_score] + [item["log_support"] for item in proposals]
    posterior = _softmax(np.asarray(log_scores, dtype=np.float64))
    for index, item in enumerate(proposals, start=1):
        item["posterior"] = float(posterior[index])
    best = max(proposals, key=lambda item: item["posterior"], default=None)
    return {
        "read": read.name,
        "current_interval": [call.start, call.end],
        "current_length": length,
        "reconstructible": bool(proposals),
        "nuc_length_log_density": float(
            length_log_density(length, background_lengths, length_bandwidth)
        ),
        "nuc_support": int(nuc_support),
        "nuc_posterior": float(posterior[0]),
        "reconstruction_posterior": float(1.0 - posterior[0]),
        "best_reconstruction": best,
        "reconstructions": proposals,
    }


def _resolvable_mass(
    full_log_scores: Sequence[float],
    resolvable_log_scores: Sequence[float],
) -> float:
    """Posterior mass of TF states whose separators are actually accessible.

    Normalized against the *full* state distribution, so it is a strict subset
    of ``complex_posterior``.  The remainder is bridged mass, which predicts the
    same bases as N and can never be refuted by the molecule.
    """
    finite = [value for value in resolvable_log_scores if math.isfinite(value)]
    if not finite:
        return 0.0
    total = float(logsumexp(np.asarray(full_log_scores, dtype=np.float64)))
    resolvable = float(logsumexp(np.asarray(finite, dtype=np.float64)))
    return max(0.0, min(1.0, float(math.exp(resolvable - total))))


def score_composite_candidate(
    read: ReadEvidence,
    call: IntervalCall,
    relevant_site_indices: Sequence[int],
    sites: Sequence[SiteTemplate],
    library: Sequence[ConfigurationLibraryEntry],
    control_lengths: Sequence[int],
    *,
    accessible_hit_probability: np.ndarray,
    nuc_prior_odds: float,
    edge_bandwidth: float,
    length_bandwidth: float,
    boundary_null_width: float,
    prior_alpha: float,
    protected_gap_prior: float,
    nuc_scorer: Optional[DddaRadialNucScorer],
    nuc_reference: Optional[dict] = None,
    nuc_geometries: Optional[Sequence[Tuple[int, int]]] = None,
    min_nuc_geometries: int = 0,
    protected_flank_prior: float = 0.5,
) -> dict:
    """Posterior over N and the *potential* TF composites at this block.

    N is the reference hypothesis: every base of the block is protected.  Each
    supported TF configuration is scored by the flexible-segment Bayes factor in
    ``composite_segment_evidence``.  Nothing is conditioned on the block's
    boundaries, and neither hypothesis marginalizes over a geometry prior, so a
    block with no informative bases returns the prior exactly.
    """
    if nuc_prior_odds <= 0:
        raise ValueError("nuc prior odds must be positive")
    if not library:
        raise ValueError("configuration library is empty")

    nuc_reference = nuc_reference or {
        "current_nuc_base": float(_nuc_base_llr(read, call, nuc_scorer)),
        "call_length": int(call.end - call.start),
        "geometry_prior": "none_block_is_the_reference",
    }
    current_nuc_base = float(nuc_reference["current_nuc_base"])

    supports = np.asarray([entry.support for entry in library], dtype=np.float64)
    config_weights = supports + prior_alpha
    config_weights /= np.sum(config_weights)
    nuc_prior = nuc_prior_odds / (1.0 + nuc_prior_odds)
    tf_prior_mass = 1.0 / (1.0 + nuc_prior_odds)

    # N is the reference: its log Bayes factor against itself is zero.
    state_names = ["N"]
    log_scores = [math.log(nuc_prior)]
    details = []
    resolvable_log_scores = []
    for entry, weight in zip(library, config_weights):
        member_values = []
        member_evidence = []
        for intervals in entry.member_intervals:
            member_evidence.append(composite_segment_evidence(
                read, call, entry.site_indices, intervals, sites,
                protected_gap_prior, protected_flank_prior, nuc_scorer,
            ))
            member_values.append(member_evidence[-1]["log_bf_vs_n"])
        normalizer = math.log(entry.support)
        config_log_bf = float(logsumexp(member_values) - normalizer)
        representative = int(np.argmax(member_values))
        evidence = member_evidence[representative]
        # Mass whose separators/flanks are actually observed accessible, rather
        # than assumed protected.  Reported, never used to veto a split: absence
        # of m6A is enzyme dropout, not evidence for a nucleosome.
        resolvable_values = [
            value if item["all_segments_accessible"] else -math.inf
            for value, item in zip(member_values, member_evidence)
        ]
        # No member has all its segments observed accessible.  Floor rather than
        # -inf so the report stays JSON-serializable; exp(-745) underflows to 0.
        config_resolvable = (
            float(logsumexp(np.asarray(resolvable_values)) - normalizer)
            if any(math.isfinite(value) for value in resolvable_values)
            else -745.0
        )
        config_prior = tf_prior_mass * float(weight)
        state_name = "TF:" + ",".join(
            sites[index].site_id for index in entry.site_indices
        )
        state_names.append(state_name)
        log_scores.append(math.log(config_prior) + config_log_bf)
        resolvable_log_scores.append(math.log(config_prior) + config_resolvable)
        details.append({
            "state": state_name,
            "site_indices": list(entry.site_indices),
            "source_support": entry.support,
            "conditional_tf_weight": float(weight),
            "source_hull_examples": [list(hull) for hull in entry.hulls[:10]],
            "representative_source_intervals": [
                list(interval)
                for interval in entry.member_intervals[representative]
            ],
            "representative_edge_conditioned_intervals": (
                evidence["effective_intervals"]
            ),
            "prior": config_prior,
            "protected_gap_prior": protected_gap_prior,
            "protected_flank_prior": protected_flank_prior,
            "edge_conditioned_log_bf_vs_n": config_log_bf,
            "resolvable_edge_conditioned_log_bf_vs_n": config_resolvable,
            "current_call_base_log_bf_vs_n": config_log_bf,
            "site_evidence": evidence["site_evidence"],
            "internal_gap_states": evidence["segments"],
            "flexible_segments": evidence["flexible_segments"],
            "informative_segments": evidence["informative_segments"],
            "informative_opportunities": evidence["informative_opportunities"],
            "informative_hits": evidence["informative_hits"],
            "gap_diagnostics": gap_diagnostics(
                read, call, entry.site_indices, sites,
                accessible_hit_probability,
                member_intervals=[
                    tuple(interval)
                    for interval in evidence["effective_intervals"]
                ],
            ),
        })
    integrated_nuc = {
        "raw_log_likelihood": 0.0,
        "edge_conditioned_log_likelihood": 0.0,
        "n_geometries": 1,
        "representative_interval": [call.start, call.end],
        "representative_llr": current_nuc_base,
        "used_current_call_fallback": False,
        "geometry_prior": "none_block_is_the_reference",
    }

    posterior = _softmax(np.asarray(log_scores, dtype=np.float64))
    for index, detail in enumerate(details, start=1):
        detail["posterior"] = float(posterior[index])
        detail["log_posterior_odds_vs_n"] = float(
            log_scores[index] - log_scores[0]
        )
        detail["prior_log_odds_vs_n"] = float(
            math.log(detail["prior"] / nuc_prior)
        )
    best_index = int(np.argmax(posterior))
    best_config_detail = max(details, key=lambda detail: detail["posterior"])
    complex_posterior = max(0.0, min(1.0, float(np.sum(posterior[1:]))))
    complex_log_odds = float(logsumexp(log_scores[1:]) - log_scores[0])
    resolvable_posterior = _resolvable_mass(log_scores, resolvable_log_scores)
    decomposition_given_complex = (
        float(best_config_detail["posterior"] / complex_posterior)
        if complex_posterior > 0.0 else 0.0
    )
    return {
        "read": read.name,
        "strand": read.strand,
        "current_interval": [call.start, call.end],
        "current_length": call.end - call.start,
        "relevant_sites": [sites[index].site_id for index in relevant_site_indices],
        "nuc_prior_odds": nuc_prior_odds,
        "nuc_prior": nuc_prior,
        "current_nuc_base_llr_diagnostic": current_nuc_base,
        "integrated_nuc": integrated_nuc,
        "nuc_posterior": float(posterior[0]),
        "complex_posterior": complex_posterior,
        "resolvable_complex_posterior": resolvable_posterior,
        "bridged_complex_posterior": max(
            0.0, complex_posterior - resolvable_posterior
        ),
        "complex_log_posterior_odds_vs_n": complex_log_odds,
        "complex_top_state": "TF_COMPLEX" if complex_posterior > posterior[0] else "N",
        "top_state": state_names[best_index],
        "top_posterior": float(posterior[best_index]),
        "best_tf_state": best_config_detail["state"],
        "best_tf_posterior": best_config_detail["posterior"],
        "best_decomposition_posterior_given_complex": decomposition_given_complex,
        "best_tf": best_config_detail,
        "_tf_states": details,
        "_nuc_reference": nuc_reference,
    }


def rescore_candidate_nuc_prior(base_result: dict, nuc_prior_odds: float) -> dict:
    """Change global N:TF prior odds without recomputing any likelihood."""
    if nuc_prior_odds <= 0:
        raise ValueError("nuc prior odds must be positive")
    result = copy.deepcopy(base_result)
    details = result["_tf_states"]
    relative_scores = [0.0] + [
        detail["edge_conditioned_log_bf_vs_n"]
        + math.log(detail["conditional_tf_weight"] / nuc_prior_odds)
        for detail in details
    ]
    resolvable_scores = [
        detail["resolvable_edge_conditioned_log_bf_vs_n"]
        + math.log(detail["conditional_tf_weight"] / nuc_prior_odds)
        for detail in details
    ]
    posterior = _softmax(np.asarray(relative_scores, dtype=np.float64))
    nuc_prior = nuc_prior_odds / (1.0 + nuc_prior_odds)
    tf_prior_mass = 1.0 / (1.0 + nuc_prior_odds)
    result["nuc_prior_odds"] = nuc_prior_odds
    result["nuc_prior"] = nuc_prior
    result["nuc_posterior"] = float(posterior[0])
    state_names = ["N"]
    for index, detail in enumerate(details, start=1):
        state_names.append(detail["state"])
        detail["prior"] = tf_prior_mass * detail["conditional_tf_weight"]
        detail["posterior"] = float(posterior[index])
        detail["prior_log_odds_vs_n"] = math.log(
            detail["conditional_tf_weight"] / nuc_prior_odds
        )
        detail["log_posterior_odds_vs_n"] = float(relative_scores[index])
    top_index = int(np.argmax(posterior))
    best_tf = max(details, key=lambda detail: detail["posterior"])
    complex_posterior = max(0.0, min(1.0, float(np.sum(posterior[1:]))))
    result["top_state"] = state_names[top_index]
    result["top_posterior"] = float(posterior[top_index])
    result["best_tf_state"] = best_tf["state"]
    result["best_tf_posterior"] = best_tf["posterior"]
    result["complex_posterior"] = complex_posterior
    resolvable_posterior = _resolvable_mass(relative_scores, resolvable_scores)
    result["resolvable_complex_posterior"] = resolvable_posterior
    result["bridged_complex_posterior"] = max(
        0.0, complex_posterior - resolvable_posterior
    )
    result["complex_log_posterior_odds_vs_n"] = float(
        logsumexp(relative_scores[1:]) - relative_scores[0]
    )
    result["complex_top_state"] = (
        "TF_COMPLEX" if complex_posterior > posterior[0] else "N"
    )
    result["best_decomposition_posterior_given_complex"] = (
        float(best_tf["posterior"] / complex_posterior)
        if complex_posterior > 0.0 else 0.0
    )
    result["best_tf"] = best_tf
    return result


def choose_composite_replacement(
    result: dict,
    call: IntervalCall,
    min_decomposition_posterior: float,
) -> dict:
    """Resolve stage 2 without allowing it to veto stage-1 demotion.

    When no exact layout owns enough posterior conditional on the TF-complex
    class, the original protected envelope is retained as one unresolved
    TF-complex interval.  This records the biologically useful conclusion
    ("not a canonical nucleosome") without manufacturing a precise split.
    """
    if not 0.0 <= min_decomposition_posterior <= 1.0:
        raise ValueError("decomposition posterior must lie in [0, 1]")
    resolved = (
        result["best_decomposition_posterior_given_complex"]
        >= min_decomposition_posterior
    )
    return {
        "decomposition_resolved": resolved,
        "replacement_kind": (
            "exact_configuration" if resolved
            else "unresolved_tf_agglomeration"
        ),
        "replacement_intervals": (
            result["best_tf"].get(
                "representative_edge_conditioned_intervals", []
            )
            if resolved else [[call.start, call.end]]
        ),
        "replacement_posterior": (
            result["best_tf_posterior"]
            if resolved else result["complex_posterior"]
        ),
    }


def analyze_composite_deconvolution(
    source_reads: Sequence[ReadEvidence],
    target_reads: Sequence[ReadEvidence],
    sites: Sequence[SiteTemplate],
    *,
    min_tq: int,
    center_radius: int,
    min_config_support: int,
    min_nuc_geometries: int = 20,
    nuc_prior_odds_values: Sequence[float],
    edge_bandwidth: float,
    length_bandwidth: float,
    boundary_null_flank: int,
    background_exclusion: int,
    max_nuc_span: int,
    strict_posterior: float,
    review_posterior: float,
    prior_alpha: float,
    protected_gap_prior: float,
    protected_flank_prior: float = 0.5,
    configuration_shift: int = 0,
    max_examples: int,
    strand_specific_library: bool,
    nuc_scorer: Optional[DddaRadialNucScorer],
    accessible_hit_probability: np.ndarray,
    configuration_control_shifts: Sequence[int] = (),
    minimum_boundary_control_log_bf: float = 3.0,
    max_proposals: int = 10000,
    edge_bandwidth_grid: Sequence[float] = (),
    use_local_complex_prior: bool = True,
    local_prior_z: float = 1.96,
    min_decomposition_posterior: float = 0.8,
) -> dict:
    """Run composite TF deconvolution without learning from ambiguous nucs."""
    if not MIN_COMPOSITE_NUC_SPAN <= max_nuc_span <= MAX_COMPOSITE_NUC_SPAN:
        raise ValueError(
            "max_nuc_span must be between "
            f"{MIN_COMPOSITE_NUC_SPAN} and {MAX_COMPOSITE_NUC_SPAN} bp"
        )
    if minimum_boundary_control_log_bf < 0.0:
        raise ValueError("minimum boundary-control log BF must be non-negative")
    if max_proposals < 0:
        raise ValueError("max proposals must be non-negative")
    if local_prior_z < 0.0:
        raise ValueError("local-prior z must be non-negative")
    if not 0.0 <= min_decomposition_posterior <= 1.0:
        raise ValueError("decomposition posterior must lie in [0, 1]")
    control_shifts = sorted({
        int(shift) for shift in configuration_control_shifts if int(shift) != 0
    })
    records = build_direct_configuration_records(
        source_reads, sites, min_tq=min_tq, center_radius=center_radius
    )
    edge_calibration = calibrate_edge_bandwidth(
        records,
        edge_bandwidth_grid,
        strand_specific=strand_specific_library,
        fallback=edge_bandwidth,
    )
    edge_bandwidth = float(edge_calibration["selected_bandwidth"])
    cache: Dict[
        Tuple[Optional[str], Tuple[int, ...]],
        List[ConfigurationLibraryEntry],
    ] = {}
    control_cache: Dict[Optional[str], List[int]] = {}
    nuc_geometry_cache: Dict[
        Tuple[Optional[str], Tuple[int, ...]],
        List[Tuple[Tuple[int, int], Tuple[str, str]]],
    ] = {}
    local_prior_cache: Dict[
        Tuple[Optional[str], Tuple[int, ...]], dict
    ] = {}
    local_positive_cache: Dict[
        Tuple[Optional[str], Tuple[int, ...]], set[Tuple[str, str]]
    ] = {}
    source_configuration_input_ids = sorted({
        str(record.library_id) for record in records
        if record.library_id is not None
    })
    source_input_ids = sorted({
        str(read.library_id) for read in source_reads
        if read.library_id is not None
    })
    source_molecule_ids = {read_molecule_id(read) for read in source_reads}

    def cohort_library_for(read: ReadEvidence, relevant: Tuple[int, ...]):
        key_strand = read.strand if strand_specific_library else None
        key = (key_strand, relevant)
        if key not in cache:
            cache[key] = shift_configuration_library(
                project_configuration_library(
                    records, relevant, min_support=min_config_support,
                    strand=key_strand,
                    sites=sites,
                ),
                configuration_shift,
            )
        return cache[key]

    def library_for(read: ReadEvidence, relevant: Tuple[int, ...]):
        return exclude_configuration_molecule(
            cohort_library_for(read, relevant),
            read_molecule_id(read),
            min_support=min_config_support,
        )

    def local_prior_for(read: ReadEvidence, relevant: Tuple[int, ...]):
        key_strand = read.strand if strand_specific_library else None
        key = (key_strand, relevant)
        if key not in local_prior_cache:
            local_prior_cache[key] = estimate_local_complex_prior(
                source_reads,
                records,
                relevant,
                sites,
                strand=key_strand,
                exclude_molecule_id=None,
                z=local_prior_z,
            )
            relevant_set = set(relevant)
            local_positive_cache[key] = {
                record.molecule_id for record in records
                if (key_strand is None or record.strand == key_strand)
                and relevant_set.intersection(record.site_indices)
            }
        # Candidate states retain the exact leave-one-molecule-out count. The
        # subtraction is O(1); rescanning a deep amplified cohort for every
        # candidate would make the statistically correct model impractical.
        base = local_prior_cache[key]
        molecule_id = read_molecule_id(read)
        hull_start, hull_end = base.get("template_hull", (0, 0))
        spans = (
            molecule_id in source_molecule_ids
            and bool(base.get("template_hull"))
            and read.ref_start <= hull_start
            and read.ref_end >= hull_end
        )
        total = int(base["spanning_molecules"]) - int(spans)
        successes = int(base["explicit_complex_molecules"]) - int(
            spans and molecule_id in local_positive_cache[key]
        )
        if total < 0 or successes < 0 or successes > total:
            raise RuntimeError("invalid leave-one-molecule-out focal prior counts")
        return {
            **base,
            "spanning_molecules": total,
            "explicit_complex_molecules": successes,
            "explicit_fraction": (successes / total if total else 0.0),
            "wilson_lower_bound": wilson_lower_bound(
                successes, total, local_prior_z
            ),
            "candidate_molecule_excluded": molecule_id in source_molecule_ids,
        }

    def controls_for(strand: str):
        key_strand = strand if strand_specific_library else None
        if key_strand not in control_cache:
            control_cache[key_strand] = collect_background_nuc_lengths(
                source_reads, sites,
                exclusion_distance=background_exclusion,
                strand=key_strand,
            )
        return control_cache[key_strand]

    def nuc_geometries_for(read: ReadEvidence, relevant: Tuple[int, ...]):
        """Empirical N geometry prior, held out for the candidate molecule.

        Symmetric with ``library_for``: N draws its geometries from source
        molecules with an explicit nuc call over the same sites, exactly as TF
        draws its configurations from source molecules with explicit TF calls.
        """
        key_strand = read.strand if strand_specific_library else None
        key = (key_strand, relevant)
        if key not in nuc_geometry_cache:
            nuc_geometry_cache[key] = collect_local_nuc_geometry_records(
                source_reads, relevant, sites,
                min_length=MIN_COMPOSITE_NUC_SPAN,
                max_length=max_nuc_span,
                strand=key_strand,
            )
        molecule_id = read_molecule_id(read)
        return [
            geometry for geometry, owner in nuc_geometry_cache[key]
            if owner != molecule_id
        ]

    scenarios = {
        str(value): {
            "counts": {
                "candidate_nucs": 0,
                "without_supported_configuration": 0,
                "overlaps_existing_tf": 0,
                "nuc_geometry_fallback": 0,
                "top_tf": 0,
                "top_complex": 0,
                "local_prior_raised": 0,
                "unresolved_complex": 0,
                "review": 0,
                "strict": 0,
                "boundary_calibrated_review": 0,
                "boundary_calibrated_strong": 0,
            },
            "configuration_counts": {},
            "boundary_control_counts": {
                str(shift): {
                    "scored": 0,
                    "top_tf": 0,
                    "top_complex": 0,
                    "review": 0,
                    "strict": 0,
                    "expected_tf_mass": 0.0,
                }
                for shift in control_shifts
            },
            "expected_tf_mass": 0.0,
            "expected_n_mass": 0.0,
            "by_relevant_sites": {},
            "examples": [],
            # Compact, untruncated state summaries make the complete decision
            # surface inspectable in MA overlays.  Detailed diagnostics remain
            # limited to examples/proposals below.
            "candidate_states": [],
            "proposals": [],
        }
        for value in nuc_prior_odds_values
    }
    seen_candidates = set()
    skipped_above_max_span = set()
    no_template_overlap = 0
    for read in target_reads:
        for call in read.nucs:
            length = call.end - call.start
            if length < MIN_COMPOSITE_NUC_SPAN:
                continue
            # Every nuc block is considered.  Requiring the block to straddle a
            # template *centre* silently dropped a third to a half of them,
            # including every block a template only partially overlaps.  A
            # template is relevant if it overlaps the block at all.
            relevant = tuple(
                index for index, site in enumerate(sites)
                if call.start < site.end and site.start < call.end
            )
            if not relevant:
                no_template_overlap += 1
                continue
            candidate_key = (
                read.library_id, read.name, call.start, call.end
            )
            if length > max_nuc_span:
                skipped_above_max_span.add(candidate_key)
                continue
            if any(
                call.start < tf.end and tf.start < call.end
                for tf in read.tfs
            ):
                for odds in nuc_prior_odds_values:
                    scenarios[str(odds)]["counts"]["overlaps_existing_tf"] += 1
                continue
            if nuc_scorer is not None and not nuc_scorer.eligible(call):
                continue
            if candidate_key in seen_candidates:
                continue
            seen_candidates.add(candidate_key)
            library = library_for(read, relevant)
            local_prior = local_prior_for(read, relevant)
            null_width = max(
                1.0,
                max(sites[index].end for index in relevant)
                - min(sites[index].start for index in relevant)
                + 2.0 * boundary_null_flank,
            )
            controls = controls_for(read.strand)
            for odds in nuc_prior_odds_values:
                scenarios[str(odds)]["counts"]["candidate_nucs"] += 1
            if not library:
                for odds in nuc_prior_odds_values:
                    scenario = scenarios[str(odds)]
                    scenario["counts"]["without_supported_configuration"] += 1
                    scenario["expected_n_mass"] += 1.0
                continue
            base_result = score_composite_candidate(
                read, call, relevant, sites, library, controls,
                accessible_hit_probability=accessible_hit_probability,
                nuc_prior_odds=1.0,
                edge_bandwidth=edge_bandwidth,
                length_bandwidth=length_bandwidth,
                boundary_null_width=null_width,
                prior_alpha=prior_alpha,
                protected_gap_prior=protected_gap_prior,
                protected_flank_prior=protected_flank_prior,
                nuc_scorer=nuc_scorer,
            )
            control_base_results = []
            for shift in control_shifts:
                shifted_library = shift_configuration_library(library, shift)
                control_base_results.append((
                    shift,
                    score_composite_candidate(
                        read, call, relevant, sites, shifted_library, controls,
                        accessible_hit_probability=accessible_hit_probability,
                        nuc_prior_odds=1.0,
                        edge_bandwidth=edge_bandwidth,
                        length_bandwidth=length_bandwidth,
                        boundary_null_width=null_width,
                        prior_alpha=prior_alpha,
                        protected_gap_prior=protected_gap_prior,
                        protected_flank_prior=protected_flank_prior,
                        nuc_scorer=nuc_scorer,
                        nuc_reference=base_result["_nuc_reference"],
                    ),
                ))
            for odds in nuc_prior_odds_values:
                scenario = scenarios[str(odds)]
                counts = scenario["counts"]
                global_complex_prior = 1.0 / (1.0 + float(odds))
                local_complex_floor = (
                    float(local_prior["wilson_lower_bound"])
                    if use_local_complex_prior else 0.0
                )
                # The requested odds are a *skepticism multiplier* on the
                # site's own occupancy prior, not a competitor to it.  Taking
                # max(global, wilson) let the local floor swallow the requested
                # prior whole: once the floor exceeded the global TF prior the
                # 10:1 and 100:1 scenarios became bit-identical, so the
                # conservative end of the sweep was unreachable at exactly the
                # sites that make calls.  Multiplying keeps both meaningful --
                # the floor sets the baseline, the sweep always moves it.
                if local_complex_floor > 0.0:
                    local_nuc_odds = (
                        (1.0 - local_complex_floor) / local_complex_floor
                    )
                    effective_nuc_prior_odds = local_nuc_odds * float(odds)
                else:
                    effective_nuc_prior_odds = float(odds)
                effective_nuc_prior_odds = min(
                    max(effective_nuc_prior_odds, 1e-9), 1e9
                )
                effective_complex_prior = 1.0 / (
                    1.0 + effective_nuc_prior_odds
                )
                result = rescore_candidate_nuc_prior(
                    base_result, effective_nuc_prior_odds
                )
                result["requested_nuc_prior_odds"] = float(odds)
                result["effective_nuc_prior_odds"] = effective_nuc_prior_odds
                result["global_complex_prior"] = global_complex_prior
                result["local_complex_prior"] = {
                    **local_prior,
                    "enabled": bool(use_local_complex_prior),
                    "combination": "skepticism_multiplier",
                    "selected_complex_prior": effective_complex_prior,
                    "raised_over_global": (
                        effective_complex_prior > global_complex_prior + 1e-12
                    ),
                }
                result.pop("_tf_states", None)
                result.pop("_nuc_reference", None)
                boundary_controls = []
                for shift, control_base in control_base_results:
                    control_result = rescore_candidate_nuc_prior(
                        control_base, effective_nuc_prior_odds
                    )
                    control_count = scenario["boundary_control_counts"][
                        str(shift)
                    ]
                    control_count["scored"] += 1
                    control_count["top_tf"] += int(
                        control_result["top_state"] != "N"
                    )
                    control_count["top_complex"] += int(
                        control_result["complex_top_state"] != "N"
                    )
                    control_count["review"] += int(
                        control_result["complex_posterior"] >= review_posterior
                    )
                    control_count["strict"] += int(
                        control_result["complex_posterior"] >= strict_posterior
                    )
                    control_count["expected_tf_mass"] += (
                        1.0 - control_result["nuc_posterior"]
                    )
                    boundary_controls.append({
                        "shift": shift,
                        "top_state": control_result["top_state"],
                        "top_posterior": control_result["top_posterior"],
                        "complex_top_state": control_result[
                            "complex_top_state"
                        ],
                        "complex_posterior": control_result[
                            "complex_posterior"
                        ],
                        "complex_log_posterior_odds_vs_n": control_result[
                            "complex_log_posterior_odds_vs_n"
                        ],
                        "best_tf_state": control_result["best_tf_state"],
                        "best_tf_posterior": control_result[
                            "best_tf_posterior"
                        ],
                        "best_tf_log_posterior_odds_vs_n": control_result[
                            "best_tf"
                        ]["log_posterior_odds_vs_n"],
                    })
                true_log_odds = float(
                    result["complex_log_posterior_odds_vs_n"]
                )
                best_control_log_odds = (
                    max(
                        item["complex_log_posterior_odds_vs_n"]
                        for item in boundary_controls
                    )
                    if boundary_controls else None
                )
                boundary_delta = (
                    true_log_odds - best_control_log_odds
                    if best_control_log_odds is not None else None
                )
                # Absence of m6A in a separator is enzyme dropout, not evidence
                # for a nucleosome, so it must not veto a split -- the whole
                # point is the block's *potential* to be a composite.  The
                # resolvable/bridged split is reported so a consumer can demand
                # observed accessibility, but it never gates the tier here.
                if (
                    result["complex_posterior"] >= strict_posterior
                    and (
                        boundary_delta is None
                        or boundary_delta >= minimum_boundary_control_log_bf
                    )
                ):
                    proposal_tier = "strong"
                elif (
                    result["complex_posterior"] >= review_posterior
                    and (boundary_delta is None or boundary_delta > 0.0)
                ):
                    proposal_tier = "review"
                else:
                    proposal_tier = "retain_n"
                replacement = choose_composite_replacement(
                    result, call, min_decomposition_posterior
                )
                decomposition_resolved = replacement[
                    "decomposition_resolved"
                ]
                replacement_kind = replacement["replacement_kind"]
                replacement_intervals = replacement["replacement_intervals"]
                replacement_posterior = replacement["replacement_posterior"]
                result["boundary_controls"] = boundary_controls
                result["boundary_control_log_bf"] = boundary_delta
                result["proposal_tier"] = proposal_tier
                result["decomposition_resolved"] = decomposition_resolved
                result["replacement_kind"] = replacement_kind
                result["replacement_intervals"] = replacement_intervals
                result["replacement_posterior"] = replacement_posterior
                counts["nuc_geometry_fallback"] += int(
                    result["integrated_nuc"].get(
                        "used_current_call_fallback", False
                    )
                )
                if result["top_state"] != "N":
                    counts["top_tf"] += 1
                if result["complex_top_state"] != "N":
                    counts["top_complex"] += 1
                    counts["unresolved_complex"] += int(
                        not decomposition_resolved
                    )
                counts["local_prior_raised"] += int(
                    result["local_complex_prior"]["raised_over_global"]
                )
                if result["complex_posterior"] >= review_posterior:
                    counts["review"] += 1
                if result["complex_posterior"] >= strict_posterior:
                    counts["strict"] += 1
                    if decomposition_resolved:
                        key = result["best_tf_state"]
                        scenario["configuration_counts"][key] = (
                            scenario["configuration_counts"].get(key, 0) + 1
                        )
                counts["boundary_calibrated_review"] += int(
                    proposal_tier in {"strong", "review"}
                )
                counts["boundary_calibrated_strong"] += int(
                    proposal_tier == "strong"
                )
                tf_mass = 1.0 - result["nuc_posterior"]
                scenario["expected_tf_mass"] += tf_mass
                scenario["expected_n_mass"] += result["nuc_posterior"]
                relevant_key = ",".join(result["relevant_sites"])
                relevant_summary = scenario["by_relevant_sites"].setdefault(
                    relevant_key,
                    {
                        "candidates": 0,
                        "top_tf": 0,
                        "top_complex": 0,
                        "unresolved_complex": 0,
                        "review": 0,
                        "strict": 0,
                        "expected_tf_mass": 0.0,
                        "short_90_120": {
                            "candidates": 0,
                            "top_tf": 0,
                            "top_complex": 0,
                            "strict": 0,
                            "expected_tf_mass": 0.0,
                        },
                    },
                )
                relevant_summary["candidates"] += 1
                relevant_summary["top_tf"] += int(result["top_state"] != "N")
                relevant_summary["top_complex"] += int(
                    result["complex_top_state"] != "N"
                )
                relevant_summary["unresolved_complex"] += int(
                    result["complex_top_state"] != "N"
                    and not decomposition_resolved
                )
                relevant_summary["review"] += int(
                    result["complex_posterior"] >= review_posterior
                )
                relevant_summary["strict"] += int(
                    result["complex_posterior"] >= strict_posterior
                )
                relevant_summary["expected_tf_mass"] += tf_mass
                if 90 <= result["current_length"] <= 120:
                    short = relevant_summary["short_90_120"]
                    short["candidates"] += 1
                    short["top_tf"] += int(result["top_state"] != "N")
                    short["top_complex"] += int(
                        result["complex_top_state"] != "N"
                    )
                    short["strict"] += int(
                        result["complex_posterior"] >= strict_posterior
                    )
                    short["expected_tf_mass"] += tf_mass
                decision_id = (
                    f"{read.name}:{call.start}-{call.end}:"
                    f"{result['best_tf_state']}"
                )
                candidate_state = {
                    "decision_id": decision_id,
                    "read": read.name,
                    "library_id": read.library_id,
                    "strand": read.strand,
                    "current_interval": [call.start, call.end],
                    "current_length": length,
                    "relevant_sites": result["relevant_sites"],
                    "proposal_tier": proposal_tier,
                    "nuc_prior_odds": odds,
                    "effective_nuc_prior_odds": effective_nuc_prior_odds,
                    "local_complex_prior": result["local_complex_prior"],
                    "nuc_posterior": result["nuc_posterior"],
                    "complex_top_state": result["complex_top_state"],
                    "complex_posterior": result["complex_posterior"],
                    "complex_log_posterior_odds_vs_n": true_log_odds,
                    "top_state": result["top_state"],
                    "top_posterior": result["top_posterior"],
                    "best_tf_state": result["best_tf_state"],
                    "best_tf_posterior": result["best_tf_posterior"],
                    "best_tf_source_support_after_candidate_holdout": (
                        result["best_tf"]["source_support"]
                    ),
                    "best_tf_log_posterior_odds_vs_n": result["best_tf"][
                        "log_posterior_odds_vs_n"
                    ],
                    "best_tf_segment_log_bf_vs_n": result["best_tf"][
                        "edge_conditioned_log_bf_vs_n"
                    ],
                    "best_tf_informative_opportunities": result["best_tf"][
                        "informative_opportunities"
                    ],
                    "best_tf_informative_hits": result["best_tf"][
                        "informative_hits"
                    ],
                    "best_tf_edge_conditioned_log_bf_vs_n": result[
                        "best_tf"
                    ]["edge_conditioned_log_bf_vs_n"],
                    "best_decomposition_posterior_given_complex": result[
                        "best_decomposition_posterior_given_complex"
                    ],
                    "decomposition_resolved": decomposition_resolved,
                    "replacement_kind": replacement_kind,
                    "replacement_posterior": replacement_posterior,
                    "boundary_control_log_bf": boundary_delta,
                    "replacement_intervals": replacement_intervals,
                }
                scenario["candidate_states"].append(candidate_state)
                scenario["examples"].append(result)
                if proposal_tier != "retain_n":
                    scenario["proposals"].append({
                        "proposal_id": decision_id,
                        "read": read.name,
                        "library_id": read.library_id,
                        "strand": read.strand,
                        "current_interval": [call.start, call.end],
                        "current_length": length,
                        "relevant_sites": result["relevant_sites"],
                        "proposal_tier": proposal_tier,
                        "nuc_prior_odds": odds,
                        "effective_nuc_prior_odds": effective_nuc_prior_odds,
                        "local_complex_prior": result["local_complex_prior"],
                        "nuc_posterior": result["nuc_posterior"],
                        "complex_top_state": result["complex_top_state"],
                        "complex_posterior": result["complex_posterior"],
                        "complex_log_posterior_odds_vs_n": true_log_odds,
                        "top_state": result["top_state"],
                        "top_posterior": result["top_posterior"],
                        "best_tf_state": result["best_tf_state"],
                        "best_tf_posterior": result["best_tf_posterior"],
                        "best_tf_log_posterior_odds_vs_n": result["best_tf"][
                            "log_posterior_odds_vs_n"
                        ],
                        "best_decomposition_posterior_given_complex": result[
                            "best_decomposition_posterior_given_complex"
                        ],
                        "decomposition_resolved": decomposition_resolved,
                        "replacement_kind": replacement_kind,
                        "replacement_posterior": replacement_posterior,
                        "boundary_control_log_bf": boundary_delta,
                        "boundary_controls": boundary_controls,
                        "replacement_intervals": replacement_intervals,
                        "source_support": result["best_tf"].get(
                            "source_support", 0
                        ),
                        "site_evidence": result["best_tf"].get(
                            "site_evidence", []
                        ),
                        "gap_diagnostics": result["best_tf"].get(
                            "gap_diagnostics", {}
                        ),
                    })

    for scenario in scenarios.values():
        scenario["candidate_states"] = sorted(
            scenario["candidate_states"],
            key=lambda state: (
                state["library_id"] or "", state["read"],
                state["current_interval"][0], state["current_interval"][1],
                state["best_tf_state"],
            ),
        )
        scenario["candidate_state_count"] = len(
            scenario["candidate_states"]
        )
        scenario["examples"] = sorted(
            scenario["examples"],
            key=lambda result: (
                -result["complex_posterior"], result["read"],
                result["current_interval"][0], result["current_interval"][1],
            ),
        )[:max_examples]
        ordered_proposals = sorted(
            scenario["proposals"],
            key=lambda proposal: (
                0 if proposal["proposal_tier"] == "strong" else 1,
                -proposal["replacement_posterior"],
                proposal["read"], proposal["current_interval"][0],
                proposal["current_interval"][1], proposal["best_tf_state"],
            ),
        )
        scenario["proposal_count_before_limit"] = len(ordered_proposals)
        scenario["proposals_truncated"] = (
            max_proposals > 0 and len(ordered_proposals) > max_proposals
        )
        scenario["proposals"] = (
            ordered_proposals[:max_proposals] if max_proposals > 0
            else ordered_proposals
        )
    return {
        "n_source_explicit_configuration_reads": len(records),
        "n_source_configuration_input_bams": len(
            source_configuration_input_ids
        ),
        "n_source_input_bams": len(source_input_ids),
        "inference_cohort": {
            "mode": "explicitly_pooled_input_bams",
            "input_bam_ids": source_input_ids,
            "input_bam_identity_used_as_prior_partition": False,
            "external_libraries_used": False,
            "candidate_holdout": "leave_one_molecule_out",
        },
        "configuration_control_shifts": control_shifts,
        "minimum_boundary_control_log_bf": minimum_boundary_control_log_bf,
        "hierarchical_decision": {
            "stage_1": "canonical_nucleosome_vs_tf_agglomeration",
            "stage_2": "exact_configuration_conditional_on_tf_agglomeration",
            "local_complex_prior_enabled": bool(use_local_complex_prior),
            "local_prior_wilson_z": float(local_prior_z),
            "configuration_sources_require_full_template_hull": True,
            "minimum_decomposition_posterior_given_complex": float(
                min_decomposition_posterior
            ),
            "unresolved_replacement": "original_block_as_tf_complex",
        },
        "edge_bandwidth_calibration": edge_calibration,
        "strand_specific_library": strand_specific_library,
        "span_filter": {
            "minimum_bp": MIN_COMPOSITE_NUC_SPAN,
            "maximum_bp": max_nuc_span,
            "hard_ceiling_bp": MAX_COMPOSITE_NUC_SPAN,
            "skipped_focal_blocks_above_maximum": len(skipped_above_max_span),
        },
        "background_nuc_control_counts": {
            (strand if strand is not None else "POOLED"): len(lengths)
            for strand, lengths in control_cache.items()
        },
        "library_cache": {
            (
                f"{strand or 'POOLED'}:"
                f"{','.join(sites[index].site_id for index in relevant)}:"
                "cohort=POOLED_INPUTS"
            ): [
                {
                    "configuration": [sites[index].site_id for index in entry.site_indices],
                    "support": entry.support,
                }
                for entry in entries
            ]
            for (strand, relevant), entries in cache.items()
        },
        "local_complex_prior_cache": {
            (
                f"{strand or 'POOLED'}:"
                f"{','.join(sites[index].site_id for index in relevant)}:"
                "cohort=POOLED_INPUTS_BEFORE_CANDIDATE_HOLDOUT"
            ): values
            for (strand, relevant), values
            in local_prior_cache.items()
        },
        "scenarios": scenarios,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--bam", required=True, action="append",
        help=(
            "Inference BAM; repeat to explicitly pool shards/timepoints into "
            "one self-contained cohort. BAM identity never partitions priors."
        ),
    )
    parser.add_argument(
        "--target-bam", action="append",
        help=(
            "Optional derivative BAM containing a subset of the same cohort "
            "molecules. External libraries are rejected; use validation instead."
        ),
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--region", required=True, type=parse_region)
    parser.add_argument("--model")
    parser.add_argument("--prob-threshold", type=int)
    parser.add_argument("--control-flank", type=int, default=2000)
    parser.add_argument("--min-mapq", type=int, default=20)
    parser.add_argument("--min-tq", type=int, default=100)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--min-config-support", type=int, default=10)
    parser.add_argument("--center-radius", type=int, default=10)
    parser.add_argument("--peak-distance", type=int, default=15)
    parser.add_argument("--max-boundary-mad", type=float, default=12.0)
    parser.add_argument("--min-local-enrichment", type=float, default=2.0)
    parser.add_argument("--local-background-radius", type=int, default=250)
    parser.add_argument("--max-auto-sites", type=int, default=12)
    parser.add_argument(
        "--site",
        action="append",
        type=parse_site_interval,
        default=[],
        help=(
            "Externally nominate a zero-based half-open START-END focal site; "
            "repeatable. Geometry is tested, but same-assay explicit calls "
            "still determine support."
        ),
    )
    parser.add_argument(
        "--forced-sites-only",
        action="store_true",
        help="Test only --site intervals rather than unioning automatic sites.",
    )
    parser.add_argument(
        "--strand-window-gap", type=int, default=MAX_COMPOSITE_NUC_SPAN,
        help=(
            "Maximum gap between sites modeled in one strand-rescue window. "
            "The 220-bp default prevents one eligible nuc from being emitted "
            "as independent decisions in adjacent windows."
        ),
    )
    parser.add_argument("--strand-max-sites", type=int, default=6)
    parser.add_argument("--strand-posterior", type=float, default=0.95)
    parser.add_argument(
        "--strand-review-posterior", type=float, default=0.5,
        help="Posterior floor for non-actionable strand-rescue review proposals.",
    )
    parser.add_argument(
        "--strand-min-source-support",
        type=int,
        help=(
            "Minimum explicit source-strand calls before its raw fitted prior "
            "can rescue the other strand (default: --min-support). Set to 0 "
            "only for externally anchored --site validation."
        ),
    )
    parser.add_argument(
        "--strand-control-shifts", default="",
        help=(
            "Optional comma-separated target-geometry shifts. Source priors "
            "remain fitted at the true sites; only compact control counts are retained."
        ),
    )
    parser.add_argument(
        "--strand-min-source-enrichment", type=float, default=1.5,
        help="Minimum source-strand focal-call enrichment over local background.",
    )
    parser.add_argument("--skip-strand-rescue", action="store_true")
    parser.add_argument("--edge-bandwidth", type=float, default=7.5)
    parser.add_argument(
        "--edge-bandwidth-grid", default="3,5,7.5,10,15,20",
        help=(
            "Comma-separated within-cohort leave-one-molecule-out KDE widths; "
            "empty uses the fixed --edge-bandwidth."
        ),
    )
    parser.add_argument("--length-bandwidth", type=float, default=10.0)
    parser.add_argument("--boundary-null-flank", type=int, default=75)
    parser.add_argument("--background-exclusion", type=int, default=500)
    parser.add_argument(
        "--max-nuc-span", type=int, default=MAX_COMPOSITE_NUC_SPAN,
        help=(
            "Largest block tested (may be lowered; hard ceiling: "
            f"{MAX_COMPOSITE_NUC_SPAN} bp)"
        ),
    )
    parser.add_argument("--nuc-prior-odds", default="1,10,100")
    parser.add_argument(
        "--production-nuc-prior-odds", type=float, default=10.0,
        help=(
            "Conservative scenario designated for actionable proposals "
            "(default: N:TF = 10:1 for a greater-than-90-percent nucleosome prior)."
        ),
    )
    parser.add_argument(
        "--no-local-complex-prior",
        action="store_true",
        help=(
            "Disable the same-cohort Wilson lower bound on focal TF-complex "
            "occupancy; retain only the global --production-nuc-prior-odds."
        ),
    )
    parser.add_argument(
        "--local-prior-z", type=float, default=1.96,
        help=(
            "Wilson-bound z for explicit TF-complex occupancy (default 1.96). "
            "Larger values make the focal prior more conservative."
        ),
    )
    parser.add_argument("--strict-posterior", type=float, default=0.95)
    parser.add_argument("--review-posterior", type=float, default=0.5)
    parser.add_argument(
        "--min-decomposition-posterior", type=float, default=0.8,
        help=(
            "Minimum posterior of the best exact TF layout conditional on the "
            "TF-complex state. Below this, demote to an unresolved TF complex "
            "rather than forcing an exact split."
        ),
    )
    parser.add_argument("--prior-alpha", type=float, default=0.5)
    parser.add_argument(
        "--protected-gap-prior", type=float, default=0.25,
        help=(
            "Prior probability that each internal gap in a multi-TF "
            "configuration is continuously protected rather than accessible"
        ),
    )
    parser.add_argument(
        "--configuration-shift", type=int, default=0,
        help="Shift every source TF geometry by this many bp as a boundary decoy",
    )
    parser.add_argument(
        "--configuration-control-shifts", default="-50,-25,25,50",
        help="Comma-separated boundary-decoy shifts used for proposal tiers.",
    )
    parser.add_argument(
        "--minimum-boundary-control-log-bf", type=float, default=3.0,
        help="Minimum true-vs-best-shifted log BF for a strong proposal.",
    )
    parser.add_argument(
        "--max-proposals", type=int, default=10000,
        help="Maximum review/strong proposals retained per N-prior scenario; 0 is unlimited.",
    )
    parser.add_argument("--skip-composite-deconvolution", action="store_true")
    parser.add_argument("--max-reads", type=int, default=0)
    parser.add_argument(
        "--molecule-collapse", choices=("auto", "on", "off"), default="auto",
        help=(
            "Collapse amplified DAF reads by full-read hard-deamination "
            "fingerprint. Auto enables this for DddA/DddB and disables it "
            "for PacBio/Nanopore."
        ),
    )
    parser.add_argument("--molecule-min-jaccard", type=float, default=0.95)
    parser.add_argument("--molecule-min-deam", type=int, default=10)
    parser.add_argument(
        "--per-molecule-deamination",
        dest="per_molecule_deamination",
        action="store_true",
        default=True,
        help=(
            "Rescale each molecule's LLRs by its own deamination efficiency, "
            "estimated from its own accessible (msp) calls (default)."
        ),
    )
    parser.add_argument(
        "--global-deamination",
        dest="per_molecule_deamination",
        action="store_false",
        help=(
            "Score every molecule at the single global accessible hit rate. "
            "Reproduces pre-calibration behaviour; protected evidence on a "
            "poorly deaminated molecule is then systematically overcredited."
        ),
    )
    parser.add_argument(
        "--min-nuc-geometries", type=int, default=20,
        help=(
            "Source-observed nuc calls required before N uses an empirical "
            "geometry prior symmetric with the TF configuration library. "
            "Below this the vague uniform enumeration is used as a fallback."
        ),
    )
    parser.add_argument(
        "--protected-flank-prior", type=float, default=0.5,
        help=(
            "Prior that a block base outside every TF footprint is still "
            "protected, i.e. the complex extends beyond the footprints the "
            "population happened to methylate around. Sweepable."
        ),
    )
    parser.add_argument("--deamination-pseudo-count", type=float, default=20.0)
    parser.add_argument(
        "--deamination-min-opportunities", type=int, default=20
    )
    parser.add_argument("--max-examples", type=int, default=25)
    parser.add_argument("--ddda-nuc-profile")
    parser.add_argument("--ddda-max-nuc-size", type=int, default=220)
    parser.add_argument("--ddda-dyad-slack", type=int, default=20)
    parser.add_argument(
        "--proposal-tsv",
        help="Optional deterministic TSV of actionable/review proposals.",
    )
    parser.add_argument("-o", "--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.forced_sites_only and not args.site:
        parser.error("--forced-sites-only requires at least one --site")
    if (
        args.strand_min_source_support is not None
        and args.strand_min_source_support < 0
    ):
        parser.error("--strand-min-source-support must be non-negative")
    if args.strand_min_source_support == 0 and not args.site:
        parser.error(
            "--strand-min-source-support 0 requires an externally anchored --site"
        )
    if not MIN_COMPOSITE_NUC_SPAN <= args.max_nuc_span <= MAX_COMPOSITE_NUC_SPAN:
        parser.error(
            "--max-nuc-span must be between "
            f"{MIN_COMPOSITE_NUC_SPAN} and {MAX_COMPOSITE_NUC_SPAN} bp"
        )
    if not 0.0 < args.molecule_min_jaccard <= 1.0:
        parser.error("--molecule-min-jaccard must lie in (0, 1]")
    if args.molecule_min_deam < 1:
        parser.error("--molecule-min-deam must be positive")
    if args.deamination_pseudo_count < 0.0:
        parser.error("--deamination-pseudo-count must be non-negative")
    if args.deamination_min_opportunities < 1:
        parser.error("--deamination-min-opportunities must be positive")
    if args.local_prior_z < 0.0:
        parser.error("--local-prior-z must be non-negative")
    if not 0.0 <= args.min_decomposition_posterior <= 1.0:
        parser.error("--min-decomposition-posterior must lie in [0, 1]")
    if not 0.0 <= args.strand_review_posterior <= args.strand_posterior <= 1.0:
        parser.error(
            "strand thresholds must satisfy 0 <= review <= strong <= 1"
        )
    if args.strand_min_source_enrichment < 0.0:
        parser.error("--strand-min-source-enrichment must be non-negative")
    try:
        strand_control_shifts = sorted({
            int(value) for value in args.strand_control_shifts.split(",")
            if value and int(value) != 0
        })
    except ValueError:
        parser.error("--strand-control-shifts must be comma-separated integers")
    preset = PRESETS[args.preset]
    model_path = resolve_resource_path(args.model or preset["model"])
    missing_inputs = [
        path for path in [*args.bam, *(args.target_bam or []), model_path]
        if not Path(path).exists()
    ]
    if missing_inputs:
        parser.error(
            "missing input file(s): " + ", ".join(missing_inputs)
        )
    model, context_size, mode = load_model_with_metadata(model_path)
    llr_hit, llr_miss = build_llr_tables(model)
    accessible_probability = accessible_hit_probabilities(model)
    chrom, focal_start, focal_end = args.region
    load_start = max(0, focal_start - args.control_flank)
    load_end = focal_end + args.control_flank
    probability_threshold = (
        args.prob_threshold
        if args.prob_threshold is not None
        else preset["prob_threshold"]
    )

    nuc_scorer = None
    nuc_profile_path = None
    if preset["nuc_likelihood"] == "ddda-radial":
        nuc_profile_path = resolve_resource_path(
            args.ddda_nuc_profile or preset["nuc_profile"]
        )
        nuc_scorer = build_ddda_radial_nuc_scorer(
            model, nuc_profile_path,
            max_call_size=args.ddda_max_nuc_size,
            dyad_slack=args.ddda_dyad_slack,
        )

    def load_bams(paths: Sequence[str]) -> List[ReadEvidence]:
        loaded = []
        for bam_path in paths:
            remaining = 0 if not args.max_reads else max(0, args.max_reads - len(loaded))
            if args.max_reads and remaining == 0:
                break
            loaded.extend(load_region_evidence(
                bam_path, chrom, load_start, load_end,
                strand_mode=preset["strand_mode"],
                mode=mode,
                context_size=context_size,
                prob_threshold=probability_threshold,
                llr_hit=llr_hit,
                llr_miss=llr_miss,
                min_mapq=args.min_mapq,
                max_reads=remaining,
            ))
        return loaded

    def calibrate_deamination(reads: List[ReadEvidence]) -> dict:
        """Rescale each molecule's LLRs by its own deamination efficiency.

        Protected evidence is absence of deamination, so a single global
        accessible hit rate credits a poorly deaminated molecule with
        protection it never demonstrated.  This is applied before any state
        model, prior, or proposal is fit, so both passes see the same LLRs.
        """
        if not args.per_molecule_deamination:
            return {"enabled": False, "reads": len(reads)}
        return calibrate_cohort_deamination(
            reads, model,
            pseudo_count=args.deamination_pseudo_count,
            min_opportunities=args.deamination_min_opportunities,
        )

    raw_source_reads = load_bams(args.bam)
    deamination_diagnostics = calibrate_deamination(raw_source_reads)
    collapse_enabled = (
        args.molecule_collapse == "on"
        or (
            args.molecule_collapse == "auto"
            and args.preset in {"ddda", "dddb"}
        )
    )

    def collapse_reads(reads: Sequence[ReadEvidence]):
        if not collapse_enabled:
            return list(reads), {
                "mode": "read",
                "raw_reads": len(reads),
                "analyzed_molecules": len(reads),
                "duplicate_reads_collapsed": 0,
            }
        return collapse_amplified_cohort_by_input(
            reads,
            min_jaccard=args.molecule_min_jaccard,
            min_deam=args.molecule_min_deam,
        )

    source_reads, source_molecule_diagnostics = collapse_reads(raw_source_reads)
    target_cohort_membership_verified = True
    if args.target_bam:
        raw_target_reads = load_bams(args.target_bam)
        calibrate_deamination(raw_target_reads)
        external_target_ids = unmatched_target_molecule_ids(
            raw_source_reads, raw_target_reads
        )
        if external_target_ids:
            parser.error(
                "--target-bam contains molecules absent from the explicitly "
                "pooled --bam cohort; external libraries are validation-only "
                f"(first unmatched molecule: {external_target_ids[0]})"
            )
        target_reads, target_molecule_diagnostics = collapse_reads(raw_target_reads)
    else:
        target_reads = source_reads
        target_molecule_diagnostics = source_molecule_diagnostics
    automatic_sites = discover_sites(
        source_reads, focal_start, focal_end,
        min_tq=args.min_tq,
        min_support=args.min_support,
        center_radius=args.center_radius,
        peak_distance=args.peak_distance,
        max_boundary_mad=args.max_boundary_mad,
        min_local_enrichment=args.min_local_enrichment,
        local_background_radius=args.local_background_radius,
        max_auto_sites=args.max_auto_sites,
    )
    sites = merge_forced_sites(
        automatic_sites,
        args.site,
        source_reads,
        min_tq=args.min_tq,
        center_radius=args.center_radius,
        forced_only=args.forced_sites_only,
    )
    strand_rescue = {
        "chemistry_applicable": preset["consensus_mode"] == "cross-strand",
        "enabled": not args.skip_strand_rescue,
        "applicable": (
            preset["consensus_mode"] == "cross-strand"
            and not args.skip_strand_rescue
        ),
        "windows": [],
        "coordinate_controls": {},
        "coordinate_control_interpretation": (
            "counterfactual target-geometry stress test with the true focal "
            "source prior preserved; not an empirical false-discovery null"
        ),
    }
    if strand_rescue["applicable"]:
        strand_min_source_support = (
            args.min_support
            if args.strand_min_source_support is None
            else args.strand_min_source_support
        )
        strand_groups = group_sites(
            sites, window_gap=args.strand_window_gap,
            max_sites=args.strand_max_sites,
        )
        strand_rescue["windows"] = [
            analyze_window(
                source_reads, group,
                target_reads=target_reads,
                flank=35,
                posterior_threshold=args.strand_posterior,
                nuc_multipliers=[1.0],
                max_examples=args.max_examples,
                allow_nuc_rescue=True,
                include_nucleosome_state=True,
                min_source_support=strand_min_source_support,
                max_nuc_span=args.max_nuc_span,
                nuc_scorer=None,
                consensus_mode="cross-strand",
                max_proposals=args.max_proposals,
                review_posterior_threshold=args.strand_review_posterior,
                min_source_local_enrichment=args.strand_min_source_enrichment,
            )
            for group in strand_groups
        ]
        for shift in strand_control_shifts:
            control_windows = []
            for group in strand_groups:
                control = analyze_window(
                    source_reads,
                    group,
                    target_reads=target_reads,
                    target_sites=shift_site_templates(group, shift),
                    flank=35,
                    posterior_threshold=args.strand_posterior,
                    nuc_multipliers=[1.0],
                    max_examples=0,
                    allow_nuc_rescue=True,
                    include_nucleosome_state=True,
                    min_source_support=strand_min_source_support,
                    max_nuc_span=args.max_nuc_span,
                    nuc_scorer=None,
                    consensus_mode="cross-strand",
                    max_proposals=1,
                    review_posterior_threshold=args.strand_review_posterior,
                    min_source_local_enrichment=args.strand_min_source_enrichment,
                )
                control_windows.append({
                    "source_sites": [[site.start, site.end] for site in group],
                    "target_sites": [
                        [site["start"], site["end"]]
                        for site in control["target_sites"]
                    ],
                    "cross_strand": {
                        target: {
                            "source_prior": target_result["source_prior"],
                            "scenarios": {
                                scenario: {
                                    "counts": values["counts"],
                                    "proposal_counts": values["proposal_counts"],
                                    "proposal_site_counts": values[
                                        "proposal_site_counts"
                                    ],
                                }
                                for scenario, values in target_result[
                                    "scenarios"
                                ].items()
                            },
                        }
                        for target, target_result in control[
                            "cross_strand"
                        ].items()
                    },
                })
            strand_rescue["coordinate_controls"][str(shift)] = control_windows

    odds_values = [float(value) for value in args.nuc_prior_odds.split(",") if value]
    if args.production_nuc_prior_odds <= 0.0:
        parser.error("--production-nuc-prior-odds must be positive")
    if args.production_nuc_prior_odds not in odds_values:
        odds_values.append(args.production_nuc_prior_odds)
    edge_bandwidth_grid = [
        float(value) for value in args.edge_bandwidth_grid.split(",") if value
    ]
    control_shifts = [
        int(value) for value in args.configuration_control_shifts.split(",")
        if value
    ]
    if args.skip_composite_deconvolution:
        composite = {
            "enabled": False,
            "scenarios": {},
            "skip_reason": "requested by --skip-composite-deconvolution",
        }
    else:
        composite = analyze_composite_deconvolution(
            source_reads, target_reads, sites,
            min_tq=args.min_tq,
            center_radius=args.center_radius,
            min_config_support=args.min_config_support,
            min_nuc_geometries=args.min_nuc_geometries,
            nuc_prior_odds_values=odds_values,
            edge_bandwidth=args.edge_bandwidth,
            length_bandwidth=args.length_bandwidth,
            boundary_null_flank=args.boundary_null_flank,
            background_exclusion=args.background_exclusion,
            max_nuc_span=args.max_nuc_span,
            strict_posterior=args.strict_posterior,
            review_posterior=args.review_posterior,
            prior_alpha=args.prior_alpha,
            protected_gap_prior=args.protected_gap_prior,
            protected_flank_prior=args.protected_flank_prior,
            configuration_shift=args.configuration_shift,
            max_examples=args.max_examples,
            strand_specific_library=(preset["consensus_mode"] == "cross-strand"),
            nuc_scorer=nuc_scorer,
            accessible_hit_probability=accessible_probability,
            configuration_control_shifts=control_shifts,
            minimum_boundary_control_log_bf=args.minimum_boundary_control_log_bf,
            max_proposals=args.max_proposals,
            edge_bandwidth_grid=edge_bandwidth_grid,
            use_local_complex_prior=not args.no_local_complex_prior,
            local_prior_z=args.local_prior_z,
            min_decomposition_posterior=args.min_decomposition_posterior,
        )
        composite["enabled"] = True
        composite["production_scenario_nuc_prior_odds"] = (
            args.production_nuc_prior_odds
        )
        composite["production_scenario_key"] = str(
            args.production_nuc_prior_odds
        )

    report = {
        "schema_version": 6,
        "producer": {
            "name": "fiberhmm-consensus-recaller-report",
            "version": VALIDATION_VERSION,
            "model_sha256": _sha256_file(model_path),
        },
        "engine": "focal-consensus-recaller-v4-paired-candidates",
        # Retained for compatibility with the development reports.
        "prototype": "consensus-recaller-v7-paired-candidate-report",
        "input": {
            "bams": [str(Path(path).resolve()) for path in args.bam],
            "target_bams": [str(Path(path).resolve()) for path in args.target_bam or []],
            "inference_cohort": {
                "mode": "explicitly_pooled_input_bams",
                "input_bams_are_one_inference_population": True,
                "bam_identity_used_only_for_output_and_dedup_strata": True,
                "external_libraries_used": False,
                "target_membership_verified": target_cohort_membership_verified,
            },
            "preset": args.preset,
            "model": str(Path(model_path).resolve()),
            "mode": mode,
            "prob_threshold": probability_threshold,
            "focal_region": [chrom, focal_start, focal_end],
            "loaded_region": [chrom, load_start, load_end],
            "files": [
                _file_metadata(path)
                for path in dict.fromkeys([
                    *args.bam, *(args.target_bam or [])
                ])
            ],
        },
        "parameters": {
            "min_tq": args.min_tq,
            "min_support": args.min_support,
            "min_config_support": args.min_config_support,
            "nuc_prior_odds": odds_values,
            "production_nuc_prior_odds": args.production_nuc_prior_odds,
            "local_complex_prior": not args.no_local_complex_prior,
            "local_prior_z": args.local_prior_z,
            "min_decomposition_posterior": args.min_decomposition_posterior,
            "edge_bandwidth": args.edge_bandwidth,
            "edge_bandwidth_grid": edge_bandwidth_grid,
            "length_bandwidth": args.length_bandwidth,
            "strict_posterior": args.strict_posterior,
            "review_posterior": args.review_posterior,
            "protected_gap_prior": args.protected_gap_prior,
            "configuration_shift": args.configuration_shift,
            "configuration_control_shifts": control_shifts,
            "minimum_boundary_control_log_bf": (
                args.minimum_boundary_control_log_bf
            ),
            "max_proposals": args.max_proposals,
            "max_nuc_span": args.max_nuc_span,
            "inference_cohort": "explicitly_pooled_input_bams",
            "candidate_holdout": "leave_one_molecule_out",
            "forced_site_intervals": [list(interval) for interval in args.site],
            "forced_sites_only": args.forced_sites_only,
            "strand_min_source_support": (
                args.min_support
                if args.strand_min_source_support is None
                else args.strand_min_source_support
            ),
            "strand_posterior": args.strand_posterior,
            "strand_review_posterior": args.strand_review_posterior,
            "strand_control_shifts": strand_control_shifts,
            "strand_min_source_enrichment": args.strand_min_source_enrichment,
            "molecule_collapse": args.molecule_collapse,
            "molecule_min_jaccard": args.molecule_min_jaccard,
            "molecule_min_deam": args.molecule_min_deam,
            "per_molecule_deamination": args.per_molecule_deamination,
            "deamination_pseudo_count": args.deamination_pseudo_count,
            "deamination_min_opportunities": (
                args.deamination_min_opportunities
            ),
            "minimum_mapped_annotation_fraction": (
                MIN_MAPPED_ANNOTATION_FRACTION
            ),
        },
        "n_source_reads": len(source_reads),
        "n_target_reads": len(target_reads),
        "molecule_diagnostics": {
            "source": source_molecule_diagnostics,
            "target": target_molecule_diagnostics,
        },
        "deamination_calibration": deamination_diagnostics,
        "sites": [asdict(site) for site in sites],
        "strand_rescue": strand_rescue,
        "composite_deconvolution": composite,
        "guardrails": {
            "ambiguous_nucs_train_tf_prior": False,
            "ambiguous_nucs_train_n_prior": False,
            "pacbio_alignment_orientation_used_as_strand": False,
            "writes_bam": False,
            "forced_site_geometry_is_external_nomination": bool(args.site),
            "forced_site_support_recomputed_from_source_bam": True,
            "raw_strand_prior_without_explicit_source_calls": (
                (
                    args.min_support
                    if args.strand_min_source_support is None
                    else args.strand_min_source_support
                ) == 0
            ),
            "source_prior_requires_focal_high_tq_calls": (
                (
                    args.min_support
                    if args.strand_min_source_support is None
                    else args.strand_min_source_support
                ) > 0
                and args.strand_min_source_enrichment > 0.0
            ),
            "strand_coordinate_controls_preserve_true_source_prior": True,
            "strand_rescue_scores_nucleosome_targets": True,
            "strand_rescue_skips_nucleosome_targets_above_max_span": True,
            "existing_tf_overwrites_conflicting_nucleosome_state": True,
            "aggressive_candidate_tables_are_untruncated": True,
            "aggressive_output_is_secondary_to_baseline_calls": True,
            "amplified_daf_counts_molecule_families": collapse_enabled,
            "amplified_daf_collapse_crosses_input_bams": False,
            "input_bam_identity_partitions_inference_prior": False,
            "external_libraries_used_for_inference": False,
            "target_bam_membership_in_source_cohort_verified": (
                target_cohort_membership_verified
            ),
            "composite_candidate_molecule_trains_its_focal_prior": False,
            "composite_candidate_molecule_trains_configuration_geometry": False,
            "global_composite_n_prior_odds_at_least_10": (
                args.production_nuc_prior_odds >= 10.0
            ),
            "local_complex_prior_uses_explicit_calls_only": True,
            "local_complex_prior_counts_only_spanning_molecules": True,
            "configuration_sources_require_full_template_hull": True,
            "unresolved_complex_does_not_force_exact_split": True,
            "substantially_clipped_ma_annotations_train_or_test_model": False,
        },
    }
    output = Path(args.output)
    _atomic_write(
        output, json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    if args.proposal_tsv:
        write_proposal_tsv(report, args.proposal_tsv)
    print(json.dumps({
        "output": str(output),
        "source_reads": len(source_reads),
        "target_reads": len(target_reads),
        "sites": len(sites),
        "strand_rescue_applicable": strand_rescue["applicable"],
        "composite_candidates": (
            {
                odds: data["counts"]["candidate_nucs"]
                for odds, data in composite["scenarios"].items()
            }
            if composite["enabled"] else {}
        ),
        "composite_skipped_above_max_span": (
            composite["span_filter"]["skipped_focal_blocks_above_maximum"]
            if composite["enabled"] else None
        ),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
