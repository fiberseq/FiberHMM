#!/usr/bin/env python3
"""Shared hard-call inference core for focal consensus recall.

This is the shared inference core for the report-only consensus command. It
uses existing MA calls to propose focal TF templates, then recomputes every
likelihood from hard per-base observations. PacBio Fiber-seq uses the A and T
opportunities present together on every HiFi molecule; alignment orientation
is never treated as an independent biochemical strand.  No BAM tags are
changed.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pysam
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from fiberhmm.cli.extract_tags import _parse_ma_annotations
from fiberhmm.core.bam_reader import (
    encode_from_query_sequence,
    parse_mm_tag_query_positions,
)
from fiberhmm.core.model_io import load_model_with_metadata
from fiberhmm.inference.tf_recaller import (
    N_CTX,
    UNMETH_OFFSET,
    build_llr_tables,
    extract_modifications,
)
from fiberhmm.inference.nuc_recaller import NucProfile, load_nuc_profile


PRESETS = {
    "dddb": {
        "model": "fiberhmm/models/dddb_nanopore.json",
        "strand_mode": "daf",
        "consensus_mode": "cross-strand",
        "prob_threshold": None,
        "nuc_rescue": True,
        "nuc_likelihood": "flat",
    },
    "ddda": {
        "model": "fiberhmm/models/ddda_TF.json",
        "strand_mode": "daf",
        "consensus_mode": "cross-strand",
        "prob_threshold": None,
        "nuc_rescue": True,
        "nuc_likelihood": "ddda-radial",
        "nuc_profile": "fiberhmm/models/ddda_nuc_profile.json",
    },
    "hia5-nanopore": {
        "model": "fiberhmm/models/hia5_nanopore.json",
        "strand_mode": "alignment",
        "consensus_mode": "cross-strand",
        "prob_threshold": 248,
        "nuc_rescue": True,
        "nuc_likelihood": "flat",
    },
    "hia5-pacbio": {
        "model": "fiberhmm/models/hia5_pacbio.json",
        "strand_mode": "pacbio-duplex",
        "consensus_mode": "population",
        "prob_threshold": 125,
        "nuc_rescue": True,
        "nuc_likelihood": "flat",
    },
}


# A molecular annotation that is visible only as a small alignment fragment
# cannot be assigned honest reference boundaries or length.  In particular, a
# >220 bp MA block extending through a soft clip must not masquerade as a
# 90--220 bp composite candidate after reference projection.
MIN_MAPPED_ANNOTATION_FRACTION = 0.8


def resolve_resource_path(path: str) -> str:
    """Resolve a user path or a resource relative to the installed package root."""
    candidate = Path(path).expanduser()
    if candidate.exists() or candidate.is_absolute():
        return str(candidate)
    package_root = Path(__file__).resolve().parent.parent
    packaged = package_root / candidate
    return str(packaged if packaged.exists() else candidate)


@dataclass(frozen=True)
class IntervalCall:
    start: int
    end: int
    score: int = 0

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2.0


@dataclass
class ReadEvidence:
    name: str
    strand: str
    ref_start: int
    ref_end: int
    positions: np.ndarray
    steps: np.ndarray
    hits: np.ndarray
    contexts: np.ndarray
    tfs: List[IntervalCall]
    nucs: List[IntervalCall]
    msps: List[IntervalCall]
    fingerprint_positions: Optional[np.ndarray] = None
    library_id: Optional[str] = None

    def spans(self, start: int, end: int) -> bool:
        return self.ref_start <= start and self.ref_end >= end

    def interval_evidence(self, start: int, end: int) -> Tuple[float, int, int]:
        lo = int(np.searchsorted(self.positions, start, side="left"))
        hi = int(np.searchsorted(self.positions, end, side="left"))
        if hi <= lo:
            return 0.0, 0, 0
        return (
            float(np.sum(self.steps[lo:hi])),
            int(hi - lo),
            int(np.sum(self.hits[lo:hi])),
        )


@dataclass
class SiteTemplate:
    site_id: str
    start: int
    end: int
    center: int
    support: Dict[str, int]
    all_support: Dict[str, int]
    median_tq: Dict[str, Optional[float]]
    start_mad: float
    end_mad: float
    local_enrichment: float
    local_enrichment_by_strand: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class Configuration:
    name: str
    site_indices: Tuple[int, ...]
    is_nucleosome: bool = False


@dataclass(frozen=True)
class DddaRadialNucScorer:
    """DddA nucleosome-vs-accessible LLR from the empirical dyad profile."""
    profile: NucProfile
    context_llr: np.ndarray
    accessible_hit_probability: np.ndarray
    max_call_size: int = 220
    dyad_slack: int = 20

    def eligible(self, call: IntervalCall) -> bool:
        return 85 <= call.end - call.start <= self.max_call_size

    def score(self, read: ReadEvidence, call: IntervalCall) -> float:
        if not self.eligible(call) or len(read.positions) == 0:
            return 0.0
        lo = int(np.searchsorted(read.positions, call.start, side="left"))
        hi = int(np.searchsorted(read.positions, call.end, side="left"))
        if hi <= lo:
            return 0.0
        positions = read.positions[lo:hi]
        hits = read.hits[lo:hi]
        contexts = read.contexts[lo:hi]
        radial = np.asarray(self.profile.radial, dtype=np.float64)
        midpoint = (call.start + call.end - 1) / 2.0
        dyads = midpoint + np.arange(-self.dyad_slack, self.dyad_slack + 1)
        offsets = np.rint(np.abs(
            dyads[:, None] - positions.astype(np.float64)[None, :]
        )).astype(int)
        keep = offsets <= self.profile.half
        rates = radial[np.minimum(offsets, len(radial) - 1)]
        # The profile has finite-depth zero bins; regularize those rather than
        # assigning an impossible likelihood to an internal hit.
        rates = np.clip(
            np.nan_to_num(rates, nan=0.05), 1e-3, 1.0 - 1e-3)
        accessible = np.clip(
            self.accessible_hit_probability[contexts], 1e-6, 1.0 - 1e-6)
        context_term = self.context_llr[contexts]
        values = np.where(
            hits[None, :],
            context_term[None, :] + np.log(rates) - np.log(accessible)[None, :],
            context_term[None, :] + np.log1p(-rates) - np.log1p(-accessible)[None, :],
        )
        dyad_scores = np.sum(np.where(keep, values, 0.0), axis=1)
        # The N state is a uniform mixture over plausible nearby dyads, not a
        # best-position scan.  This preserves the Occam penalty for uncertainty.
        scores = np.asarray(dyad_scores, dtype=np.float64)
        return float(_logsumexp(scores) - math.log(len(scores)))


def build_ddda_radial_nuc_scorer(
    model,
    profile_path: str,
    *,
    max_call_size: int,
    dyad_slack: int,
) -> DddaRadialNucScorer:
    """Build a radial DddA N-vs-accessible scorer from existing models.

    The empirical profile supplies ``P(hit | nucleosome, distance to dyad)``.
    The TF model supplies the context-specific accessible hit probability and
    the same context-marginal term used by FiberHMM's ordinary TF LLRs.
    """
    if max_call_size < 85:
        raise ValueError("DddA max nuc size must be at least 85 bp")
    if dyad_slack < 0:
        raise ValueError("DddA dyad slack must be non-negative")
    emissions = np.asarray(model.emissionprob_, dtype=np.float64)
    if emissions.ndim != 2 or emissions.shape[0] != 2:
        raise ValueError("DddA radial scoring requires a two-state model")
    if emissions.shape[1] < UNMETH_OFFSET + N_CTX:
        raise ValueError("DddA model has an incomplete emission table")
    eps = 1e-30
    hit_protected = emissions[0, :N_CTX]
    hit_accessible = emissions[1, :N_CTX]
    miss_protected = emissions[0, UNMETH_OFFSET:UNMETH_OFFSET + N_CTX]
    miss_accessible = emissions[1, UNMETH_OFFSET:UNMETH_OFFSET + N_CTX]
    context_protected = np.clip(hit_protected + miss_protected, eps, None)
    context_accessible = np.clip(hit_accessible + miss_accessible, eps, None)
    context_llr = np.log(context_protected) - np.log(context_accessible)
    accessible_hit_probability = np.clip(
        hit_accessible / context_accessible, eps, 1.0 - eps)
    return DddaRadialNucScorer(
        profile=load_nuc_profile(profile_path),
        context_llr=context_llr,
        accessible_hit_probability=accessible_hit_probability,
        max_call_size=max_call_size,
        dyad_slack=dyad_slack,
    )


def _logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def fit_mixture_weights(
    log_likelihoods: np.ndarray,
    *,
    max_iter: int = 500,
    tol: float = 1e-9,
    min_weight: float = 1e-9,
) -> Tuple[np.ndarray, int]:
    """Maximum-likelihood mixture weights for fixed per-read state likelihoods."""
    ll = np.asarray(log_likelihoods, dtype=np.float64)
    if ll.ndim != 2 or ll.shape[0] == 0 or ll.shape[1] == 0:
        raise ValueError("log_likelihoods must be a non-empty 2D array")
    weights = np.full(ll.shape[1], 1.0 / ll.shape[1], dtype=np.float64)
    for iteration in range(1, max_iter + 1):
        log_resp = ll + np.log(np.maximum(weights, min_weight))[None, :]
        log_resp -= _logsumexp(log_resp, axis=1)[:, None]
        updated = np.mean(np.exp(log_resp), axis=0)
        updated = np.maximum(updated, min_weight)
        updated /= np.sum(updated)
        if float(np.max(np.abs(updated - weights))) < tol:
            return updated, iteration
        weights = updated
    return weights, max_iter


def enumerate_configurations(
    sites: Sequence[SiteTemplate],
    *,
    include_nucleosome: bool = True,
) -> List[Configuration]:
    """Return accessible, every non-overlapping TF subset, and one nuc state."""
    configs = [Configuration("A", ())]
    n = len(sites)
    for mask in range(1, 1 << n):
        indices = tuple(i for i in range(n) if mask & (1 << i))
        ordered = sorted(indices, key=lambda i: (sites[i].start, sites[i].end))
        valid = all(sites[a].end <= sites[b].start for a, b in zip(ordered, ordered[1:]))
        if not valid:
            continue
        configs.append(Configuration(
            "TF:" + ",".join(sites[i].site_id for i in ordered),
            tuple(ordered),
        ))
    if include_nucleosome:
        configs.append(Configuration("N", (), is_nucleosome=True))
    return configs


def match_direct_site_indices(
    calls: Sequence[IntervalCall],
    sites: Sequence[SiteTemplate],
    *,
    center_radius: int = 10,
) -> set[int]:
    """Map each existing TF call to one best non-overlapping site template."""
    selected: set[int] = set()
    for call in sorted(calls, key=lambda item: (item.start, item.end)):
        candidates = sorted(
            (
                index for index, site in enumerate(sites)
                if abs(call.center - site.center) <= center_radius
            ),
            key=lambda index: (
                abs(call.center - sites[index].center),
                abs((call.end - call.start) - (sites[index].end - sites[index].start)),
            ),
        )
        for index in candidates:
            site = sites[index]
            overlaps_selected = any(
                site.start < sites[other].end and sites[other].start < site.end
                for other in selected
            )
            if not overlaps_selected:
                selected.add(index)
                break
    return selected


def configuration_log_likelihoods(
    read: ReadEvidence,
    sites: Sequence[SiteTemplate],
    configurations: Sequence[Configuration],
    window_start: int,
    window_end: int,
    nuc_scorer: Optional[DddaRadialNucScorer] = None,
) -> Tuple[np.ndarray, List[Tuple[float, int, int]]]:
    """LLRs relative to a fully accessible window plus per-site evidence."""
    site_evidence = [read.interval_evidence(site.start, site.end) for site in sites]
    nuc_call = IntervalCall(window_start, window_end)
    if nuc_scorer is None:
        nuc_llr, _, _ = read.interval_evidence(window_start, window_end)
    else:
        nuc_llr = nuc_scorer.score(read, nuc_call)
    values = []
    for config in configurations:
        if config.is_nucleosome:
            values.append(nuc_llr)
        else:
            values.append(sum(site_evidence[i][0] for i in config.site_indices))
    return np.asarray(values, dtype=np.float64), site_evidence


def posterior_with_prior_multiplier(
    log_likelihoods: np.ndarray,
    prior: np.ndarray,
    configurations: Sequence[Configuration],
    nuc_multiplier: float,
) -> np.ndarray:
    adjusted = np.asarray(prior, dtype=np.float64).copy()
    for i, config in enumerate(configurations):
        if config.is_nucleosome:
            adjusted[i] *= float(nuc_multiplier)
    adjusted = np.maximum(adjusted, 1e-12)
    adjusted /= np.sum(adjusted)
    return _softmax(np.log(adjusted) + np.asarray(log_likelihoods, dtype=np.float64))


def _mapped_annotations(
    read,
    target: str,
    reference_positions: Sequence[Optional[int]],
) -> List[IntervalCall]:
    result: List[IntervalCall] = []
    for ann in _parse_ma_annotations(read, target) or []:
        q_start = max(0, int(ann["start"]))
        q_end = min(len(reference_positions), q_start + int(ann["length"]))
        interval_positions = reference_positions[q_start:q_end]
        mapped = [p for p in interval_positions if p is not None]
        mapped_fraction = (
            len(mapped) / len(interval_positions) if interval_positions else 0.0
        )
        if (
            not mapped
            or mapped_fraction < MIN_MAPPED_ANNOTATION_FRACTION
        ):
            continue
        quals = ann.get("quals", [])
        result.append(IntervalCall(min(mapped), max(mapped) + 1, int(quals[0]) if quals else 0))
    return result


def _hard_observations(
    read,
    strand_mode: str,
    mode: str,
    context_size: int,
    prob_threshold: Optional[int],
):
    sequence = read.query_sequence
    if not sequence:
        return None
    if strand_mode == "daf":
        extracted = extract_modifications(read, "daf", context_size)
        if extracted is None:
            return None
        mod_positions, symbol, sequence = extracted
        strand = "CT" if symbol == "+" else "GA"
    else:
        if not read.has_tag("MM") or not read.has_tag("ML"):
            return None
        mod_positions = parse_mm_tag_query_positions(
            read.get_tag("MM"),
            read.get_tag("ML"),
            sequence,
            bool(read.is_reverse),
            prob_threshold=int(prob_threshold if prob_threshold is not None else 125),
            mode=mode,
        )
        # A PacBio Fiber-seq MM tag contains both A+a and T-a blocks on the
        # same HiFi molecule.  The alignment flag only orients that duplex
        # molecule against the reference; it does not define two evidence
        # populations.  The parser/encoder above still uses is_reverse to put
        # MM positions and contexts in the correct coordinate frame.
        strand = (
            "BOTH" if strand_mode == "pacbio-duplex"
            else "REV" if read.is_reverse else "FWD"
        )
    obs = encode_from_query_sequence(
        sequence,
        mod_positions,
        edge_trim=10,
        mode=mode,
        strand=(symbol if strand_mode == "daf" else "."),
        context_size=context_size,
        is_reverse=bool(read.is_reverse),
    )
    return obs, strand


def load_region_evidence(
    bam_path: str,
    chrom: str,
    start: int,
    end: int,
    *,
    strand_mode: str,
    mode: str,
    context_size: int,
    prob_threshold: Optional[int],
    llr_hit: np.ndarray,
    llr_miss: np.ndarray,
    min_mapq: int,
    max_reads: int,
) -> List[ReadEvidence]:
    reads: List[ReadEvidence] = []
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(chrom, start, end):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.mapping_quality < min_mapq:
                continue
            extracted = _hard_observations(read, strand_mode, mode, context_size, prob_threshold)
            if extracted is None:
                continue
            obs, strand = extracted
            ref_positions = read.get_reference_positions(full_length=True)
            positions: List[int] = []
            steps: List[float] = []
            hits: List[bool] = []
            contexts: List[int] = []
            fingerprint_positions: List[int] = []
            for q_pos, ref_pos in enumerate(ref_positions):
                if ref_pos is None:
                    continue
                code = int(obs[q_pos])
                if 0 <= code < N_CTX:
                    fingerprint_positions.append(int(ref_pos))
                if ref_pos < start or ref_pos >= end:
                    continue
                if 0 <= code < N_CTX:
                    positions.append(int(ref_pos))
                    steps.append(float(llr_hit[code]))
                    hits.append(True)
                    contexts.append(code)
                elif UNMETH_OFFSET <= code < UNMETH_OFFSET + N_CTX:
                    context = code - UNMETH_OFFSET
                    positions.append(int(ref_pos))
                    steps.append(float(llr_miss[context]))
                    hits.append(False)
                    contexts.append(context)
            order = (
                np.argsort(np.asarray(positions, dtype=np.int64))
                if positions else np.array([], dtype=int)
            )
            pos_arr = np.asarray(positions, dtype=np.int64)[order]
            step_arr = np.asarray(steps, dtype=np.float64)[order]
            hit_arr = np.asarray(hits, dtype=bool)[order]
            context_arr = np.asarray(contexts, dtype=np.int64)[order]
            reads.append(ReadEvidence(
                name=read.query_name,
                strand=strand,
                ref_start=int(read.reference_start),
                ref_end=int(read.reference_end or read.reference_start),
                positions=pos_arr,
                steps=step_arr,
                hits=hit_arr,
                contexts=context_arr,
                tfs=_mapped_annotations(read, "tf", ref_positions),
                nucs=_mapped_annotations(read, "nuc", ref_positions),
                msps=_mapped_annotations(read, "msp", ref_positions),
                fingerprint_positions=np.asarray(
                    fingerprint_positions, dtype=np.int64
                ),
                library_id=str(bam_path),
            ))
            if max_reads and len(reads) >= max_reads:
                break
    return reads


def conditional_hit_probabilities(model) -> Tuple[np.ndarray, np.ndarray]:
    """Return context-wise P(hard hit | protected) and P(hard hit | accessible).

    The emission table stores joint mass over (context, hit/miss), so each
    state's hit probability is renormalized within its own context.  The
    context marginal is a property of the sequence and cancels in every LLR.
    """
    emissions = np.asarray(model.emissionprob_, dtype=np.float64)
    hit_prot = emissions[0, :N_CTX]
    miss_prot = emissions[0, UNMETH_OFFSET:UNMETH_OFFSET + N_CTX]
    hit_acc = emissions[1, :N_CTX]
    miss_acc = emissions[1, UNMETH_OFFSET:UNMETH_OFFSET + N_CTX]
    protected = hit_prot / np.maximum(hit_prot + miss_prot, 1e-30)
    accessible = hit_acc / np.maximum(hit_acc + miss_acc, 1e-30)
    return (
        np.clip(protected, 1e-6, 1.0 - 1e-6),
        np.clip(accessible, 1e-6, 1.0 - 1e-6),
    )


def _llr_tables_for_accessible_rate(
    protected_hit: np.ndarray,
    accessible_hit: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """(llr_hit, llr_miss) for one context-wise accessible hit probability."""
    return (
        np.log(protected_hit) - np.log(accessible_hit),
        np.log1p(-protected_hit) - np.log1p(-accessible_hit),
    )


def calibrate_read_deamination(
    read: ReadEvidence,
    protected_hit: np.ndarray,
    accessible_hit: np.ndarray,
    *,
    pseudo_count: float,
    min_opportunities: int,
    min_factor: float,
    max_factor: float,
) -> dict:
    """Rescale one molecule's LLRs by its own deamination efficiency.

    A protected call is evidenced by *absence* of deamination, so its LLR is a
    direct function of P(hit | accessible).  A single global value credits a
    poorly deaminated molecule with protection it never demonstrated.  The
    molecule's own accessible (``msp``) calls supply an independent estimate of
    that rate; it is shrunk toward the model expectation so shallow molecules
    cannot swing far on a handful of bases.
    """
    if pseudo_count < 0.0:
        raise ValueError("pseudo-count must be non-negative")
    if not 0.0 < min_factor <= 1.0 <= max_factor:
        raise ValueError("factor bounds must bracket 1.0")
    selected = np.zeros(read.positions.size, dtype=bool)
    for call in read.msps:
        lo = int(np.searchsorted(read.positions, call.start, side="left"))
        hi = int(np.searchsorted(read.positions, call.end, side="left"))
        selected[lo:hi] = True
    opportunities = int(selected.sum())
    # Expected accessible hit rate for *this* molecule's context composition,
    # so the factor measures enzyme efficiency rather than sequence content.
    expected = (
        float(np.mean(accessible_hit[read.contexts[selected]]))
        if opportunities else float("nan")
    )
    if opportunities < min_opportunities or not expected > 0.0:
        return {
            "calibrated": False,
            "accessible_opportunities": opportunities,
            "factor": 1.0,
        }
    hits = int(read.hits[selected].sum())
    shrunk = (hits + pseudo_count * expected) / (opportunities + pseudo_count)
    factor = float(np.clip(shrunk / expected, min_factor, max_factor))
    scaled = np.clip(accessible_hit * factor, 1e-6, 1.0 - 1e-6)
    llr_hit, llr_miss = _llr_tables_for_accessible_rate(protected_hit, scaled)
    read.steps = np.where(
        read.hits, llr_hit[read.contexts], llr_miss[read.contexts]
    ).astype(np.float64)
    return {
        "calibrated": True,
        "accessible_opportunities": opportunities,
        "accessible_hits": hits,
        "observed_rate": hits / opportunities,
        "expected_rate": expected,
        "shrunk_rate": float(shrunk),
        "factor": factor,
    }


def calibrate_cohort_deamination(
    reads: Sequence[ReadEvidence],
    model,
    *,
    pseudo_count: float = 20.0,
    min_opportunities: int = 20,
    min_factor: float = 0.2,
    max_factor: float = 2.0,
) -> dict:
    """Apply per-molecule deamination calibration across a cohort."""
    protected_hit, accessible_hit = conditional_hit_probabilities(model)
    factors = []
    calibrated = 0
    for read in reads:
        record = calibrate_read_deamination(
            read, protected_hit, accessible_hit,
            pseudo_count=pseudo_count,
            min_opportunities=min_opportunities,
            min_factor=min_factor,
            max_factor=max_factor,
        )
        if record["calibrated"]:
            calibrated += 1
            factors.append(record["factor"])
    array = np.asarray(factors, dtype=np.float64)
    quantiles = (
        [float(value) for value in np.percentile(array, [5, 25, 50, 75, 95])]
        if array.size else None
    )
    return {
        "enabled": True,
        "reads": len(reads),
        "calibrated_reads": calibrated,
        "uncalibrated_reads": len(reads) - calibrated,
        "pseudo_count": float(pseudo_count),
        "min_accessible_opportunities": int(min_opportunities),
        "factor_bounds": [float(min_factor), float(max_factor)],
        "global_accessible_hit_probability": float(np.mean(accessible_hit)),
        "factor_quantiles_5_25_50_75_95": quantiles,
    }


def _mad(values: Sequence[int]) -> float:
    if not values:
        return math.nan
    array = np.asarray(values, dtype=np.float64)
    return float(np.median(np.abs(array - np.median(array))))


def discover_sites(
    reads: Sequence[ReadEvidence],
    chrom_start: int,
    chrom_end: int,
    *,
    min_tq: int,
    min_support: int,
    center_radius: int,
    peak_distance: int,
    max_boundary_mad: float,
    min_local_enrichment: float,
    local_background_radius: int,
    max_auto_sites: int,
) -> List[SiteTemplate]:
    strands = sorted({read.strand for read in reads})
    calls_by_strand: Dict[str, List[Tuple[IntervalCall, str]]] = {s: [] for s in strands}
    all_calls_by_strand: Dict[str, List[Tuple[IntervalCall, str]]] = {s: [] for s in strands}
    for read in reads:
        for call in read.tfs:
            if call.end <= chrom_start or call.start >= chrom_end:
                continue
            all_calls_by_strand[read.strand].append((call, read.name))
            if call.score >= min_tq:
                calls_by_strand[read.strand].append((call, read.name))

    proposed: List[Tuple[int, List[IntervalCall]]] = []
    width = chrom_end - chrom_start
    for strand, calls in calls_by_strand.items():
        if not calls:
            continue
        hist = np.zeros(width, dtype=np.float64)
        for call, _ in calls:
            index = int(round(call.center)) - chrom_start
            if 0 <= index < width:
                hist[index] += 1.0
        smooth = gaussian_filter1d(hist, 3.0)
        peaks, _ = find_peaks(smooth, distance=peak_distance, prominence=0.25, height=0.18)
        for peak in peaks:
            center = chrom_start + int(peak)
            nearby = [call for call, _ in calls if abs(call.center - center) <= center_radius]
            support = len({
                name for call, name in calls
                if abs(call.center - center) <= center_radius
            })
            if support >= min_support:
                proposed.append((center, nearby))

    proposed.sort(key=lambda item: item[0])
    merged: List[Tuple[List[int], List[IntervalCall]]] = []
    for center, calls in proposed:
        if merged and center - int(np.median(merged[-1][0])) <= center_radius:
            merged[-1][0].append(center)
            merged[-1][1].extend(calls)
        else:
            merged.append(([center], list(calls)))

    sites: List[SiteTemplate] = []
    for centers, seed_calls in merged:
        center = int(round(float(np.median(centers))))
        starts = [call.start for call in seed_calls]
        ends = [call.end for call in seed_calls]
        start = int(round(float(np.median(starts))))
        end = int(round(float(np.median(ends))))
        if end <= start:
            continue
        start_mad = _mad(starts)
        end_mad = _mad(ends)
        if max(start_mad, end_mad) > max_boundary_mad:
            continue
        support: Dict[str, int] = {}
        all_support: Dict[str, int] = {}
        median_tq: Dict[str, Optional[float]] = {}
        for strand in strands:
            high = [(call, name) for call, name in calls_by_strand[strand]
                    if abs(call.center - center) <= center_radius]
            all_near = [(call, name) for call, name in all_calls_by_strand[strand]
                        if abs(call.center - center) <= center_radius]
            support[strand] = len({name for _, name in high})
            all_support[strand] = len({name for _, name in all_near})
            median_tq[strand] = (
                float(np.median([call.score for call, _ in all_near])) if all_near else None
            )
        if max(support.values(), default=0) < min_support:
            continue
        strongest = max(strands, key=lambda strand: support.get(strand, 0))
        inner = max(1, 2 * center_radius)
        background_width = max(1, 2 * (local_background_radius - inner))
        local_enrichment_by_strand = {}
        for strand in strands:
            background = [
                (call, name) for call, name in calls_by_strand[strand]
                if inner < abs(call.center - center) <= local_background_radius
            ]
            expected = (
                len(background) * (2 * center_radius + 1) / background_width
            )
            local_enrichment_by_strand[strand] = float(
                (support[strand] + 0.5) / (expected + 0.5)
            )
        local_enrichment = local_enrichment_by_strand[strongest]
        if local_enrichment < min_local_enrichment:
            continue
        sites.append(SiteTemplate(
            site_id=f"site{len(sites) + 1}",
            start=start,
            end=end,
            center=center,
            support=support,
            all_support=all_support,
            median_tq=median_tq,
            start_mad=start_mad,
            end_mad=end_mad,
            local_enrichment=float(local_enrichment),
            local_enrichment_by_strand=local_enrichment_by_strand,
        ))
    sites.sort(
        key=lambda site: (
            max(site.support.values(), default=0) * site.local_enrichment
            / (1.0 + site.start_mad + site.end_mad)
        ),
        reverse=True,
    )
    if max_auto_sites > 0:
        sites = sites[:max_auto_sites]
    return sorted(sites, key=lambda site: (site.start, site.end))


def group_sites(
    sites: Sequence[SiteTemplate],
    *,
    window_gap: int,
    max_sites: int,
) -> List[List[SiteTemplate]]:
    ordered = sorted(sites, key=lambda site: (site.start, site.end))
    groups: List[List[SiteTemplate]] = []
    current: List[SiteTemplate] = []
    current_end = -1
    for site in ordered:
        if current and (site.start - current_end > window_gap or len(current) >= max_sites):
            groups.append(current)
            current = []
        current.append(site)
        current_end = max(current_end, site.end) if len(current) > 1 else site.end
    if current:
        groups.append(current)
    return groups


def marginalize_configuration_prior(
    global_configurations: Sequence[Configuration],
    global_prior: np.ndarray,
    relevant_global_indices: Sequence[int],
    local_configurations: Sequence[Configuration],
) -> np.ndarray:
    """Marginalize a window prior onto the sites inside one nuc/MSP call."""
    global_to_local = {global_index: local_index
                       for local_index, global_index in enumerate(relevant_global_indices)}
    local_lookup = {
        (config.is_nucleosome, tuple(config.site_indices)): index
        for index, config in enumerate(local_configurations)
    }
    result = np.zeros(len(local_configurations), dtype=np.float64)
    for config, weight in zip(global_configurations, global_prior):
        if config.is_nucleosome:
            index = local_lookup.get((True, ()))
        else:
            projected = tuple(sorted(
                global_to_local[index] for index in config.site_indices
                if index in global_to_local
            ))
            index = local_lookup.get((False, projected))
        if index is not None:
            result[index] += float(weight)
    result = np.maximum(result, 1e-12)
    result /= np.sum(result)
    return result


def local_call_likelihoods(
    read: ReadEvidence,
    local_sites: Sequence[SiteTemplate],
    configurations: Sequence[Configuration],
    call: IntervalCall,
    nuc_scorer: Optional[DddaRadialNucScorer] = None,
) -> Tuple[np.ndarray, List[Tuple[float, int, int]]]:
    evidence = [
        read.interval_evidence(max(call.start, site.start), min(call.end, site.end))
        for site in local_sites
    ]
    if nuc_scorer is None:
        nuc_llr, _, _ = read.interval_evidence(call.start, call.end)
    else:
        nuc_llr = nuc_scorer.score(read, call)
    values = []
    for config in configurations:
        if config.is_nucleosome:
            values.append(nuc_llr)
        else:
            values.append(sum(evidence[index][0] for index in config.site_indices))
    return np.asarray(values, dtype=np.float64), evidence


def fit_site_state_model(
    reads: Sequence[ReadEvidence],
    site: SiteTemplate,
    *,
    flank: int,
    include_nucleosome: bool,
    nuc_scorer: Optional[DddaRadialNucScorer] = None,
) -> dict:
    """Fit A/TF/(optional N) weights using every read spanning one focal site."""
    configurations = enumerate_configurations(
        [site], include_nucleosome=include_nucleosome)
    if nuc_scorer is None:
        window_start = max(0, site.start - flank)
        window_end = site.end + flank
    else:
        # A source molecule need only span the focal TF.  Its available bases
        # within this canonical nucleosome window contribute to N; absent
        # flanks are marginalized by simply contributing no likelihood term.
        dyad = int(round(site.center))
        window_start = max(0, dyad - nuc_scorer.profile.half)
        window_end = dyad + nuc_scorer.profile.half + 1
    eligible = [read for read in reads if read.spans(site.start, site.end)]
    matrix = []
    for read in eligible:
        values, _ = configuration_log_likelihoods(
            read, [site], configurations, window_start, window_end,
            nuc_scorer=nuc_scorer)
        matrix.append(values)
    if not matrix:
        weights = np.full(len(configurations), 1.0 / len(configurations))
        iterations = 0
    else:
        weights, iterations = fit_mixture_weights(np.vstack(matrix))
    by_state = {
        ("N" if config.is_nucleosome else "TF" if config.site_indices else "A"):
        float(weights[index])
        for index, config in enumerate(configurations)
    }
    return {
        "coverage": len(eligible),
        "iterations": iterations,
        "weights": by_state,
    }


def fit_joint_configuration_model(
    reads: Sequence[ReadEvidence],
    sites: Sequence[SiteTemplate],
    configurations: Sequence[Configuration],
    *,
    flank: int,
    nuc_scorer: Optional[DddaRadialNucScorer] = None,
) -> Tuple[np.ndarray, dict]:
    """Fit a joint focal occupancy prior from molecules spanning all sites.

    This is the PacBio population-consensus model.  Each eligible molecule
    contributes its combined A- and T-opportunity likelihood: both biochemical
    strands are already encoded on that molecule.  Reads are never separated
    by genomic alignment orientation.
    """
    window_start = max(0, min(site.start for site in sites) - flank)
    window_end = max(site.end for site in sites) + flank
    core_start = min(site.start for site in sites)
    core_end = max(site.end for site in sites)
    eligible = [read for read in reads if read.spans(core_start, core_end)]
    matrix = []
    for read in eligible:
        values, _ = configuration_log_likelihoods(
            read, sites, configurations, window_start, window_end,
            nuc_scorer=nuc_scorer,
        )
        matrix.append(values)
    if matrix:
        weights, iterations = fit_mixture_weights(np.vstack(matrix))
    else:
        weights = np.full(len(configurations), 1.0 / len(configurations))
        iterations = 0
    return weights, {
        "coverage": len(eligible),
        "iterations": iterations,
        "weights": {
            config.name: float(weights[index])
            for index, config in enumerate(configurations)
        },
    }


def factorized_configuration_prior(
    local_configurations: Sequence[Configuration],
    relevant_global_indices: Sequence[int],
    source_site_models: Dict[int, dict],
) -> np.ndarray:
    """Compose arbitrary local TF subsets from short-read-aware site marginals."""
    tf_probabilities = []
    nuc_probabilities = []
    for global_index in relevant_global_indices:
        weights = source_site_models[global_index]["weights"]
        accessible = float(weights.get("A", 0.0))
        tf = float(weights.get("TF", 0.0))
        denominator = accessible + tf
        tf_probabilities.append(tf / denominator if denominator > 0 else 0.5)
        if "N" in weights:
            nuc_probabilities.append(float(weights["N"]))
    rho_nuc = float(np.mean(nuc_probabilities)) if nuc_probabilities else 0.0

    prior = np.zeros(len(local_configurations), dtype=np.float64)
    non_nuc_indices = []
    for config_index, config in enumerate(local_configurations):
        if config.is_nucleosome:
            prior[config_index] = rho_nuc
            continue
        occupied = set(config.site_indices)
        weight = 1.0
        for site_index, probability in enumerate(tf_probabilities):
            weight *= probability if site_index in occupied else (1.0 - probability)
        prior[config_index] = weight
        non_nuc_indices.append(config_index)
    non_nuc_total = float(np.sum(prior[non_nuc_indices])) if non_nuc_indices else 0.0
    if non_nuc_total > 0:
        non_nuc_mass = 1.0 - rho_nuc if any(c.is_nucleosome for c in local_configurations) else 1.0
        prior[non_nuc_indices] *= non_nuc_mass / non_nuc_total
    prior = np.maximum(prior, 1e-12)
    prior /= np.sum(prior)
    return prior


def analyze_window(
    reads: Sequence[ReadEvidence],
    sites: Sequence[SiteTemplate],
    *,
    target_reads: Optional[Sequence[ReadEvidence]] = None,
    flank: int,
    posterior_threshold: float,
    nuc_multipliers: Sequence[float],
    max_examples: int,
    allow_nuc_rescue: bool,
    include_nucleosome_state: Optional[bool] = None,
    min_source_support: int,
    max_nuc_span: int = 220,
    nuc_scorer: Optional[DddaRadialNucScorer] = None,
    consensus_mode: str = "cross-strand",
    review_log_bf_band: float = 2.0,
    review_min_prior_odds: float = 0.5,
    max_proposals: int = 10000,
    review_posterior_threshold: float = 0.5,
    target_sites: Optional[Sequence[SiteTemplate]] = None,
    min_source_local_enrichment: float = 1.5,
) -> dict:
    if consensus_mode not in {"cross-strand", "population"}:
        raise ValueError(f"unknown consensus mode: {consensus_mode}")
    if max_proposals < 0:
        raise ValueError("max proposals must be non-negative")
    if not 0.0 <= review_posterior_threshold <= posterior_threshold <= 1.0:
        raise ValueError(
            "posterior thresholds must satisfy 0 <= review <= strong <= 1"
        )
    if min_source_local_enrichment < 0.0:
        raise ValueError("minimum source local enrichment must be non-negative")
    if not 90 <= max_nuc_span <= 220:
        raise ValueError("strand-rescue max nuc span must be between 90 and 220 bp")
    if include_nucleosome_state is None:
        include_nucleosome_state = allow_nuc_rescue
    target_site_list = list(sites if target_sites is None else target_sites)
    if len(target_site_list) != len(sites):
        raise ValueError("source and target site lists must have equal length")
    window_start = max(
        0,
        min(
            min(site.start for site in sites),
            min(site.start for site in target_site_list),
        ) - flank,
    )
    window_end = max(
        max(site.end for site in sites),
        max(site.end for site in target_site_list),
    ) + flank
    configurations = enumerate_configurations(
        sites, include_nucleosome=include_nucleosome_state
    )
    strands = sorted({read.strand for read in reads})
    by_strand = {
        strand: [read for read in reads if read.strand == strand]
        for strand in strands
    }
    target_pool = reads if target_reads is None else target_reads
    target_by_strand = {
        strand: [read for read in target_pool if read.strand == strand]
        for strand in strands
    }
    site_models: Dict[str, Dict[int, dict]] = {}
    for strand, strand_reads in by_strand.items():
        site_models[strand] = {
            index: fit_site_state_model(
                strand_reads, site, flank=flank,
                include_nucleosome=include_nucleosome_state,
                nuc_scorer=nuc_scorer)
            for index, site in enumerate(sites)
        }

    joint_models: Dict[str, dict] = {}
    joint_priors: Dict[str, np.ndarray] = {}
    if consensus_mode == "population":
        if strands != ["BOTH"]:
            raise ValueError(
                "PacBio population consensus requires one BOTH evidence group; "
                "alignment orientations must not be split"
            )
        prior, model = fit_joint_configuration_model(
            by_strand["BOTH"], sites, configurations, flank=flank,
            nuc_scorer=nuc_scorer,
        )
        joint_priors["BOTH"] = prior
        joint_models["BOTH"] = model

    cross_results = {}
    population_results = {}
    comparisons = []
    if consensus_mode == "population":
        comparisons.append((population_results, "BOTH", "BOTH", list(target_pool)))
    elif len(strands) == 2 and all(strand in site_models for strand in strands):
        for target in strands:
            source = strands[1] if target == strands[0] else strands[0]
            comparisons.append(
                (cross_results, target, source, target_by_strand[target])
            )

    for result_bucket, target, source, target_strand_reads in comparisons:
            scenarios = {}
            for multiplier in nuc_multipliers:
                changes = []
                candidate_states = []
                nuc_reviews = []
                proposal_counts: Dict[str, int] = {}
                proposal_site_counts: Dict[str, int] = {}
                configurations_before: Dict[str, int] = {}
                configurations_after: Dict[str, int] = {}
                configuration_reads = 0
                counts = {
                    "eligible": 0,
                    "eligible_nuc": 0,
                    "eligible_access": 0,
                    "top_without_tf": 0,
                    "nonpositive_bf": 0,
                    "positive_bf": 0,
                    "insufficient_local_evidence": 0,
                    "below_posterior": 0,
                    "review_proposed": 0,
                    "proposed": 0,
                    "from_nuc": 0,
                    "from_access": 0,
                    "nuc_likelihood_ambiguous": 0,
                    "nuc_consensus_review": 0,
                    "aggressive_candidates": 0,
                    "nuc_above_max_span": 0,
                    "nuc_overlaps_existing_tf": 0,
                }
                for read in target_strand_reads:
                    directly_called = match_direct_site_indices(
                        read.tfs, target_site_list
                    )
                    blocked_by_direct = {
                        index for index, site in enumerate(target_site_list)
                        if any(
                            site.start < target_site_list[direct].end
                            and target_site_list[direct].start < site.end
                            for direct in directly_called
                        )
                    }
                    rescued_sites = set()
                    spans_group = read.spans(
                        min(site.start for site in target_site_list),
                        max(site.end for site in target_site_list),
                    )
                    source_supported = {
                        index for index, site in enumerate(sites)
                        if site.support.get(source, 0) >= min_source_support
                        and (
                            (
                                site.local_enrichment_by_strand or {}
                            ).get(source, site.local_enrichment)
                            >= min_source_local_enrichment
                        )
                        and (
                            consensus_mode == "population"
                            or site.support.get(source, 0) >= site.support.get(target, 0)
                        )
                    }
                    if not source_supported:
                        if spans_group:
                            key = (
                                ",".join(sites[index].site_id for index in sorted(directly_called))
                                or "A"
                            )
                            configurations_before[key] = configurations_before.get(key, 0) + 1
                            configurations_after[key] = configurations_after.get(key, 0) + 1
                            configuration_reads += 1
                        continue
                    candidate_calls: List[Tuple[str, IntervalCall, List[int]]] = []
                    if allow_nuc_rescue:
                        for call in read.nucs:
                            call_length = call.end - call.start
                            if call_length < 90:
                                continue
                            relevant = [
                                index for index, site in enumerate(target_site_list)
                                if index in source_supported
                                and index not in blocked_by_direct
                                and call.start <= site.center < call.end
                            ]
                            if not relevant:
                                continue
                            if call_length > max_nuc_span:
                                counts["nuc_above_max_span"] += 1
                                continue
                            if any(
                                call.start < tf.end and tf.start < call.end
                                for tf in read.tfs
                            ):
                                counts["nuc_overlaps_existing_tf"] += 1
                                continue
                            if nuc_scorer is not None and not nuc_scorer.eligible(call):
                                continue
                            candidate_calls.append(("N", call, relevant))
                    for call in read.msps:
                        relevant = [
                            index for index, site in enumerate(target_site_list)
                            if index in source_supported
                            and index not in blocked_by_direct
                            and call.start <= site.center < call.end
                        ]
                        if relevant:
                            candidate_calls.append(("A", call, relevant))

                    for current, call, relevant in candidate_calls:
                        local_sites = [target_site_list[index] for index in relevant]
                        local_configs = enumerate_configurations(
                            local_sites, include_nucleosome=(current == "N"))
                        if consensus_mode == "population":
                            local_prior = marginalize_configuration_prior(
                                configurations, joint_priors[source], relevant,
                                local_configs,
                            )
                        else:
                            local_prior = factorized_configuration_prior(
                                local_configs, relevant, site_models[source])
                        values, per_site = local_call_likelihoods(
                            read, local_sites, local_configs, call,
                            nuc_scorer=nuc_scorer)
                        posterior = posterior_with_prior_multiplier(
                            values, local_prior, local_configs,
                            multiplier if current == "N" else 1.0)
                        counts["eligible"] += 1
                        eligible_key = "eligible_nuc" if current == "N" else "eligible_access"
                        counts[eligible_key] += 1
                        nuc_index = next(
                            (i for i, config in enumerate(local_configs)
                             if config.is_nucleosome),
                            None,
                        )
                        supported_tf_indices = [
                            index for index, config in enumerate(local_configs)
                            if config.site_indices
                            and all(
                                per_site[site_index][1] >= 1
                                and per_site[site_index][0] > 0.0
                                for site_index in config.site_indices
                            )
                        ]
                        if current == "N" and nuc_index is not None and supported_tf_indices:
                            neutral_posterior = posterior_with_prior_multiplier(
                                values, local_prior, local_configs, 1.0
                            )
                            best_tf_index = max(
                                supported_tf_indices,
                                key=lambda index: float(neutral_posterior[index]),
                            )
                            best_tf = local_configs[best_tf_index]
                            log_bf = float(values[best_tf_index] - values[nuc_index])
                            prior_odds = float(
                                local_prior[best_tf_index] / local_prior[nuc_index]
                            )
                            # The likelihood band asks whether this molecule's
                            # hard A/T (or deamination) calls distinguish the
                            # alternatives.  The prior-odds floor separately
                            # requires the population-consensus TF tiling to be
                            # a genuinely plausible competitor to N.
                            if abs(log_bf) <= review_log_bf_band:
                                counts["nuc_likelihood_ambiguous"] += 1
                            if (
                                abs(log_bf) <= review_log_bf_band
                                and prior_odds >= review_min_prior_odds
                            ):
                                counts["nuc_consensus_review"] += 1
                                call_llr, call_opportunities, call_hits = (
                                    read.interval_evidence(call.start, call.end)
                                )
                                selected_llr = float(sum(
                                    per_site[index][0]
                                    for index in best_tf.site_indices
                                ))
                                selected_opportunities = int(sum(
                                    per_site[index][1]
                                    for index in best_tf.site_indices
                                ))
                                selected_hits = int(sum(
                                    per_site[index][2]
                                    for index in best_tf.site_indices
                                ))
                                nuc_reviews.append({
                                    "read": read.name,
                                    "current_interval": [call.start, call.end],
                                    "current_length": call.end - call.start,
                                    "candidate": best_tf.name,
                                    "candidate_posterior": float(
                                        neutral_posterior[best_tf_index]
                                    ),
                                    "nuc_posterior": float(
                                        neutral_posterior[nuc_index]
                                    ),
                                    "scenario_candidate_posterior": float(
                                        posterior[best_tf_index]
                                    ),
                                    "scenario_nuc_posterior": float(
                                        posterior[nuc_index]
                                    ),
                                    "log_bf_candidate_vs_nuc": log_bf,
                                    "candidate_prior": float(local_prior[best_tf_index]),
                                    "nuc_prior": float(local_prior[nuc_index]),
                                    "candidate_prior_odds_vs_nuc": prior_odds,
                                    "call_opportunities": call_opportunities,
                                    "call_hits": call_hits,
                                    "selected_tf_opportunities": selected_opportunities,
                                    "selected_tf_hits": selected_hits,
                                    "outside_tf_opportunities": max(
                                        0, call_opportunities - selected_opportunities
                                    ),
                                    "outside_tf_hits": max(
                                        0, call_hits - selected_hits
                                    ),
                                    "outside_tf_llr": (
                                        float(call_llr - selected_llr)
                                        if nuc_scorer is None else None
                                    ),
                                    "opportunity_density": (
                                        call_opportunities / (call.end - call.start)
                                    ),
                                    "site_evidence": [
                                        {
                                            "site": local_sites[index].site_id,
                                            "llr": per_site[index][0],
                                            "opportunities": per_site[index][1],
                                            "hits": per_site[index][2],
                                        }
                                        for index in best_tf.site_indices
                                    ],
                                })

                        # Preserve the complete locally supported TF-vs-current
                        # decision surface before the conservative proposal
                        # gates below.  The paired MA writer uses this table so
                        # a browser threshold can move a molecule between its
                        # current state and the best TF configuration without
                        # rerunning inference.  Unsupported configurations
                        # (no positive target-strand evidence) remain excluded.
                        candidate_state = None
                        current_index = nuc_index if current == "N" else 0
                        if current_index is not None and supported_tf_indices:
                            best_supported_index = max(
                                supported_tf_indices,
                                key=lambda index: float(posterior[index]),
                            )
                            best_supported = local_configs[best_supported_index]
                            tf_mass = float(sum(
                                posterior[index]
                                for index in supported_tf_indices
                            ))
                            current_mass = float(posterior[current_index])
                            pair_mass = tf_mass + current_mass
                            paired_tf_posterior = (
                                tf_mass / pair_mass if pair_mass > 0.0 else 0.5
                            )
                            paired_current_posterior = 1.0 - paired_tf_posterior
                            best_given_tf = (
                                float(posterior[best_supported_index]) / tf_mass
                                if tf_mass > 0.0 else 0.0
                            )
                            tf_prior_mass = float(sum(
                                local_prior[index]
                                for index in supported_tf_indices
                            ))
                            current_prior = float(local_prior[current_index])
                            pair_prior_mass = tf_prior_mass + current_prior
                            tf_prior_probability = (
                                tf_prior_mass / pair_prior_mass
                                if pair_prior_mass > 0.0 else 0.5
                            )
                            source_evidence = [
                                {
                                    "site": sites[relevant[i]].site_id,
                                    "source_strand": source,
                                    "explicit_high_tq_support": sites[
                                        relevant[i]
                                    ].support.get(source, 0),
                                    "local_enrichment": (
                                        sites[relevant[i]].local_enrichment_by_strand
                                        or {}
                                    ).get(
                                        source,
                                        sites[relevant[i]].local_enrichment,
                                    ),
                                    "state_model": site_models[source][
                                        relevant[i]
                                    ],
                                }
                                for i in best_supported.site_indices
                            ]
                            support_fractions = [
                                min(
                                    1.0,
                                    float(item["explicit_high_tq_support"])
                                    / max(
                                        1,
                                        int(item["state_model"]["coverage"]),
                                    ),
                                )
                                for item in source_evidence
                            ]
                            decision_id = (
                                f"{read.name}:{call.start}-{call.end}:"
                                f"{current}->{best_supported.name}"
                            )
                            candidate_state = {
                                "decision_id": decision_id,
                                "read": read.name,
                                "library_id": read.library_id,
                                "target_strand": target,
                                "source_prior_strand": source,
                                "proposal_tier": "retain_current",
                                "current": current,
                                "current_interval": [call.start, call.end],
                                "proposed": best_supported.name,
                                "proposed_site_intervals": [
                                    [local_sites[i].start, local_sites[i].end]
                                    for i in best_supported.site_indices
                                ],
                                "posterior": paired_tf_posterior,
                                "current_posterior": paired_current_posterior,
                                "raw_tf_class_posterior": tf_mass,
                                "raw_current_posterior": current_mass,
                                "best_tf_posterior": float(
                                    posterior[best_supported_index]
                                ),
                                "best_configuration_posterior_given_tf": (
                                    best_given_tf
                                ),
                                "nuc_posterior": (
                                    float(posterior[nuc_index])
                                    if nuc_index is not None else None
                                ),
                                "proposed_log_likelihood": float(
                                    values[best_supported_index]
                                ),
                                "current_log_likelihood": float(
                                    values[current_index]
                                ),
                                "log_bf_vs_current": float(
                                    values[best_supported_index]
                                    - values[current_index]
                                ),
                                "tf_prior_probability_vs_current": (
                                    tf_prior_probability
                                ),
                                "proposed_prior": float(
                                    local_prior[best_supported_index]
                                ),
                                "current_prior": current_prior,
                                "source_support_fraction": (
                                    min(support_fractions)
                                    if support_fractions else None
                                ),
                                "site_evidence": [
                                    {
                                        "site": local_sites[i].site_id,
                                        "llr": per_site[i][0],
                                        "opportunities": per_site[i][1],
                                        "hits": per_site[i][2],
                                    }
                                    for i in best_supported.site_indices
                                ],
                                "source_prior_evidence": source_evidence,
                            }
                            candidate_states.append(candidate_state)
                            counts["aggressive_candidates"] += 1
                        top_index = int(np.argmax(posterior))
                        top = local_configs[top_index]
                        if not top.site_indices:
                            counts["top_without_tf"] += 1
                            continue
                        current_index = nuc_index if current == "N" else 0
                        if current_index is None or values[top_index] <= values[current_index]:
                            # Population depth may promote weak positive evidence,
                            # but it may not overturn molecule-level evidence that
                            # favors the existing call.
                            counts["nonpositive_bf"] += 1
                            continue
                        counts["positive_bf"] += 1
                        locally_supported = all(
                            per_site[index][1] >= 1 and per_site[index][0] > 0.0
                            for index in top.site_indices
                        )
                        if not locally_supported:
                            counts["insufficient_local_evidence"] += 1
                            continue
                        posterior_value = float(posterior[top_index])
                        if posterior_value < posterior_threshold:
                            counts["below_posterior"] += 1
                        if posterior_value < review_posterior_threshold:
                            continue
                        proposal_tier = (
                            "strong"
                            if posterior_value >= posterior_threshold
                            else "review"
                        )
                        if candidate_state is not None:
                            candidate_state["proposal_tier"] = proposal_tier
                        proposal_record = {
                            "proposal_id": (
                                f"{read.name}:{call.start}-{call.end}:"
                                f"{current}->{top.name}"
                            ),
                            "read": read.name,
                            "library_id": read.library_id,
                            "target_strand": target,
                            "source_prior_strand": source,
                            "proposal_tier": proposal_tier,
                            "current": current,
                            "current_interval": [call.start, call.end],
                            "proposed": top.name,
                            "proposed_site_intervals": [
                                [local_sites[i].start, local_sites[i].end]
                                for i in top.site_indices
                            ],
                            "posterior": posterior_value,
                            "nuc_posterior": (
                                float(posterior[nuc_index]) if nuc_index is not None else None
                            ),
                            "proposed_log_likelihood": float(values[top_index]),
                            "current_log_likelihood": float(
                                values[nuc_index] if nuc_index is not None else values[0]
                            ),
                            "log_bf_vs_current": float(
                                values[top_index]
                                - (values[nuc_index] if nuc_index is not None else values[0])
                            ),
                            "proposed_prior": float(local_prior[top_index]),
                            "current_prior": float(
                                local_prior[nuc_index]
                                if nuc_index is not None else local_prior[0]
                            ),
                            "site_evidence": [
                                {"site": local_sites[i].site_id,
                                 "llr": per_site[i][0],
                                 "opportunities": per_site[i][1],
                                 "hits": per_site[i][2]}
                                for i in top.site_indices
                            ],
                            "source_prior_evidence": [
                                {
                                    "site": sites[relevant[i]].site_id,
                                    "source_strand": source,
                                    "explicit_high_tq_support": sites[
                                        relevant[i]
                                    ].support.get(source, 0),
                                    "local_enrichment": (
                                        sites[relevant[i]].local_enrichment_by_strand
                                        or {}
                                    ).get(
                                        source,
                                        sites[relevant[i]].local_enrichment,
                                    ),
                                    "state_model": site_models[source][
                                        relevant[i]
                                    ],
                                }
                                for i in top.site_indices
                            ],
                        }
                        changes.append(proposal_record)
                        if proposal_tier == "review":
                            counts["review_proposed"] += 1
                            continue
                        counts["proposed"] += 1
                        if current == "N":
                            counts["from_nuc"] += 1
                        else:
                            counts["from_access"] += 1
                        proposal_key = f"{current}->{top.name}"
                        proposal_counts[proposal_key] = proposal_counts.get(proposal_key, 0) + 1
                        selected_global_indices = {
                            relevant[local_index] for local_index in top.site_indices
                        }
                        rescued_sites.update(selected_global_indices)
                        for global_index in selected_global_indices:
                            site_id = sites[global_index].site_id
                            proposal_site_counts[site_id] = (
                                proposal_site_counts.get(site_id, 0) + 1
                            )
                    if spans_group:
                        before_key = (
                            ",".join(
                                sites[index].site_id for index in sorted(directly_called)
                            ) or "A"
                        )
                        after_indices = directly_called | rescued_sites
                        after_key = (
                            ",".join(
                                sites[index].site_id for index in sorted(after_indices)
                            ) or "A"
                        )
                        configurations_before[before_key] = (
                            configurations_before.get(before_key, 0) + 1
                        )
                        configurations_after[after_key] = (
                            configurations_after.get(after_key, 0) + 1
                        )
                        configuration_reads += 1
                ordered_changes = sorted(
                    changes,
                    key=lambda item: (
                        -item["posterior"], -item["log_bf_vs_current"],
                        item["read"], item["current_interval"][0],
                        item["current_interval"][1], item["proposed"],
                    ),
                )
                ordered_reviews = sorted(
                    nuc_reviews,
                    key=lambda item: (
                        -item["candidate_posterior"],
                        -item["log_bf_candidate_vs_nuc"],
                        item["read"], item["current_interval"][0],
                        item["current_interval"][1], item["candidate"],
                    ),
                )
                ordered_candidate_states = sorted(
                    candidate_states,
                    key=lambda item: (
                        item["library_id"] or "", item["read"],
                        item["current_interval"][0],
                        item["current_interval"][1], item["proposed"],
                    ),
                )
                scenarios[str(multiplier)] = {
                    "counts": counts,
                    "proposal_counts": proposal_counts,
                    "proposal_site_counts": proposal_site_counts,
                    "configuration_reads": configuration_reads,
                    "configurations_before": configurations_before,
                    "configurations_after": configurations_after,
                    "proposal_count_before_limit": len(ordered_changes),
                    "proposals_truncated": (
                        max_proposals > 0
                        and len(ordered_changes) > max_proposals
                    ),
                    "proposals": (
                        ordered_changes[:max_proposals]
                        if max_proposals > 0 else ordered_changes
                    ),
                    "candidate_state_count": len(ordered_candidate_states),
                    "candidate_states": ordered_candidate_states,
                    "examples": ordered_changes[:max_examples],
                    "nuc_review_count_before_limit": len(ordered_reviews),
                    "nuc_reviews_truncated": (
                        max_proposals > 0
                        and len(ordered_reviews) > max_proposals
                    ),
                    "nuc_review_proposals": (
                        ordered_reviews[:max_proposals]
                        if max_proposals > 0 else ordered_reviews
                    ),
                    "nuc_review_examples": ordered_reviews[:max_examples],
                }
            result_bucket[target] = {"source_prior": source, "scenarios": scenarios}

    return {
        "window": [window_start, window_end],
        "sites": [asdict(site) for site in sites],
        "target_sites": [asdict(site) for site in target_site_list],
        "site_coverage": {
            strand: {
                sites[index].site_id: model["coverage"]
                for index, model in models.items()
            }
            for strand, models in site_models.items()
        },
        "states": [config.name for config in configurations],
        "site_models": {
            strand: {
                sites[index].site_id: model
                for index, model in models.items()
            }
            for strand, models in site_models.items()
        },
        "consensus_mode": consensus_mode,
        "joint_models": joint_models,
        "cross_strand": cross_results,
        "population_consensus": population_results,
    }


def parse_region(value: str) -> Tuple[str, int, int]:
    try:
        chrom, span = value.rsplit(":", 1)
        start, end = span.replace(",", "").split("-", 1)
        start_i, end_i = int(start), int(end)
    except Exception as error:
        raise argparse.ArgumentTypeError("region must be chrom:start-end") from error
    if start_i < 0 or end_i <= start_i:
        raise argparse.ArgumentTypeError("region must have 0 <= start < end")
    return chrom, start_i, end_i


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--bam", required=True, action="append",
        help=(
            "Prior/discovery post-nuc/post-TF BAM; repeat -i to pool without merging"
        ),
    )
    parser.add_argument(
        "--target-bam", action="append",
        help=(
            "Optional held-out BAM to score; repeat to pool held-out targets. "
            "Its calls never influence site discovery or source priors"
        ),
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--region", required=True, type=parse_region)
    parser.add_argument("--model", help="Override the preset model JSON")
    parser.add_argument("--prob-threshold", type=int, help="Hard ML threshold for Hia5")
    parser.add_argument(
        "--ddda-nuc-profile",
        help="Override the DddA radial nucleosome-profile JSON",
    )
    parser.add_argument(
        "--ddda-max-nuc-size", type=int, default=220,
        help="Largest current DddA nuc scored as one radial nucleosome",
    )
    parser.add_argument(
        "--ddda-dyad-slack", type=int, default=20,
        help="Half-width of the DddA dyad-position prior in bp",
    )
    parser.add_argument("--min-mapq", type=int, default=20)
    parser.add_argument("--min-tq", type=int, default=100)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--center-radius", type=int, default=10)
    parser.add_argument("--peak-distance", type=int, default=15)
    parser.add_argument("--max-boundary-mad", type=float, default=12.0)
    parser.add_argument("--min-local-enrichment", type=float, default=2.0)
    parser.add_argument("--local-background-radius", type=int, default=250)
    parser.add_argument("--max-auto-sites", type=int, default=12)
    parser.add_argument("--window-gap", type=int, default=90)
    parser.add_argument("--window-flank", type=int, default=35)
    parser.add_argument("--max-sites", type=int, default=6)
    parser.add_argument("--posterior-threshold", type=float, default=0.95)
    parser.add_argument("--nuc-prior-multipliers", default="1,10,100")
    parser.add_argument(
        "--review-log-bf-band", type=float, default=2.0,
        help="Flag nuc alternatives whose hard-call log BF is within +/- this value",
    )
    parser.add_argument(
        "--review-min-prior-odds", type=float, default=0.5,
        help="Minimum empirical TF-tiling:N prior odds for the nuc review tier",
    )
    parser.add_argument("--max-reads", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("-o", "--output", required=True, help="JSON report path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    preset = PRESETS[args.preset]
    model_path = resolve_resource_path(args.model or preset["model"])
    model, context_size, mode = load_model_with_metadata(model_path)
    llr_hit, llr_miss = build_llr_tables(model)
    chrom, start, end = args.region
    prob_threshold = args.prob_threshold
    if prob_threshold is None:
        prob_threshold = preset["prob_threshold"]
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
        loaded: List[ReadEvidence] = []
        for bam_path in paths:
            remaining = 0 if not args.max_reads else max(
                0, args.max_reads - len(loaded))
            if args.max_reads and remaining == 0:
                break
            loaded.extend(load_region_evidence(
                bam_path, chrom, start, end,
                strand_mode=preset["strand_mode"],
                mode=mode,
                context_size=context_size,
                prob_threshold=prob_threshold,
                llr_hit=llr_hit,
                llr_miss=llr_miss,
                min_mapq=args.min_mapq,
                max_reads=remaining,
            ))
        return loaded

    reads = load_bams(args.bam)
    held_out_reads = load_bams(args.target_bam) if args.target_bam else None
    sites = discover_sites(
        reads, start, end,
        min_tq=args.min_tq,
        min_support=args.min_support,
        center_radius=args.center_radius,
        peak_distance=args.peak_distance,
        max_boundary_mad=args.max_boundary_mad,
        min_local_enrichment=args.min_local_enrichment,
        local_background_radius=args.local_background_radius,
        max_auto_sites=args.max_auto_sites,
    )
    groups = group_sites(sites, window_gap=args.window_gap, max_sites=args.max_sites)
    multipliers = [float(value) for value in args.nuc_prior_multipliers.split(",") if value]
    windows = [
        analyze_window(
            reads, group,
            target_reads=held_out_reads,
            flank=args.window_flank,
            posterior_threshold=args.posterior_threshold,
            nuc_multipliers=multipliers,
            max_examples=args.max_examples,
            allow_nuc_rescue=bool(preset["nuc_rescue"]),
            min_source_support=args.min_support,
            nuc_scorer=nuc_scorer,
            consensus_mode=preset["consensus_mode"],
            review_log_bf_band=args.review_log_bf_band,
            review_min_prior_odds=args.review_min_prior_odds,
        )
        for group in groups
    ]
    report = {
        "prototype": "focal-consensus-v0",
        "input": {
            "bams": [str(Path(path).resolve()) for path in args.bam],
            "target_bams": (
                [str(Path(path).resolve()) for path in args.target_bam]
                if args.target_bam else []
            ),
            "preset": args.preset,
            "model": str(Path(model_path).resolve()),
            "mode": mode,
            "consensus_mode": preset["consensus_mode"],
            "prob_threshold": prob_threshold,
            "nuc_likelihood": preset["nuc_likelihood"],
            "nuc_profile": (
                str(Path(nuc_profile_path).resolve()) if nuc_profile_path else None
            ),
            "region": [chrom, start, end],
        },
        "parameters": {
            "min_tq": args.min_tq,
            "min_support": args.min_support,
            "posterior_threshold": args.posterior_threshold,
            "nuc_prior_multipliers": multipliers,
            "review_log_bf_band": args.review_log_bf_band,
            "review_min_prior_odds": args.review_min_prior_odds,
            "allow_nuc_rescue": bool(preset["nuc_rescue"]),
            "ddda_max_nuc_size": (
                args.ddda_max_nuc_size if nuc_scorer is not None else None
            ),
            "ddda_dyad_slack": (
                args.ddda_dyad_slack if nuc_scorer is not None else None
            ),
        },
        "n_reads": len(reads),
        "n_prior_reads": len(reads),
        "n_target_reads": (
            len(held_out_reads) if held_out_reads is not None else len(reads)
        ),
        "evidence_group_counts": {
            strand: sum(read.strand == strand for read in reads)
            for strand in sorted({read.strand for read in reads})
        },
        "target_evidence_group_counts": {
            strand: sum(read.strand == strand for read in (
                held_out_reads if held_out_reads is not None else reads
            ))
            for strand in sorted({
                read.strand for read in (
                    held_out_reads if held_out_reads is not None else reads
                )
            })
        },
        "n_sites": len(sites),
        "n_windows": len(windows),
        "windows": windows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "prior_reads": len(reads),
        "target_reads": (
            len(held_out_reads) if held_out_reads is not None else len(reads)
        ),
        "sites": len(sites),
        "windows": len(windows),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
