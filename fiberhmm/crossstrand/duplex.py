"""Sequence-identity-free CT/GA duplex pairing for scDAF.

The scorer combines the existing FiberHMM nucleosome lattice with two views of
the DddA protection profile.  It deliberately does not inspect A/T alleles,
haplotypes, TF likelihoods, or any other sequence-identity feature.  The model
was frozen on GRCh38 chr1:[0, 40.1 Mb) and replicated on the disjoint interval
chr1:[40.5, 80.5 Mb) before it was added to FiberHMM.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from fiberhmm.cli.extract_tags import _build_query_to_ref, _deam_positions_list
from fiberhmm.crossstrand.pairing import (
    FLAVOR_CT,
    FLAVOR_GA,
    PairParams,
    ReadFeat,
    _gaussian_kernel,
    nuc_dyads_ref,
    read_flavor,
    score_pair,
)

MODEL_FILENAME = "ddda_duplex_v1.json"
ROTATIONAL_MODEL_FILENAME = "ddda_duplex_rotational_v1.json"
STATUS_PAIRED = "P"
STATUS_UNRESOLVED = "U"
STATUS_NONE = "."


@dataclass(frozen=True)
class DuplexParams:
    """Fixed feature geometry and configurable reciprocal-selection gate."""

    grid_bp: int = 10
    dyad_sigma_bp: float = 30.0
    max_lag_bp: int = 60
    min_overlap_bp: int = 1500
    min_nucs: int = 4
    profile_bin_bp: int = 20
    profile_sigma_bp: int = 20
    min_profile_bins: int = 20
    min_opportunities_per_bin: float = 2.0
    min_margin: float = 1.0
    null_floor: float = 0.0

    def pair_params(self) -> PairParams:
        return PairParams(
            grid_bp=self.grid_bp,
            sigma_bp=self.dyad_sigma_bp,
            max_lag_bp=self.max_lag_bp,
            min_overlap_bp=self.min_overlap_bp,
            min_nucs=self.min_nucs,
        )


@dataclass(frozen=True)
class ProtectionProfile:
    """Smoothed non-CpG DddA deamination profile in reference bins."""

    name: str
    flavor: int
    start_bin: int
    end_bin: int
    rate: np.ndarray
    weight: np.ndarray
    opportunities: int
    hits: int


@dataclass(frozen=True)
class DuplexModel:
    model_id: str
    feature_names: Tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    metadata: Mapping[str, object]

    def decision_function(self, values: Mapping[str, float]) -> float:
        vector = np.asarray([values[name] for name in self.feature_names], dtype=float)
        if not np.all(np.isfinite(vector)):
            return float("nan")
        standardized = (vector - self.mean) / self.scale
        return float(self.intercept + np.dot(self.coefficients, standardized))


@dataclass(frozen=True)
class EdgeEvidence:
    ct_index: int
    ga_index: int
    score: float
    features: Mapping[str, float]


@dataclass
class DuplexResult:
    partner: Dict[int, int]
    score: Dict[int, float]
    margin: Dict[int, float]
    status: Dict[int, str]
    evidence: Dict[Tuple[int, int], EdgeEvidence]
    geometric_edges: int
    scored_edges: int


def infer_call_layer(header) -> str:
    """Infer which bundled calibration matches the BAM's nucleosome calls."""
    data = header.to_dict() if hasattr(header, "to_dict") else dict(header)
    provenance = [str(comment) for comment in data.get("CO", [])]
    provenance.extend(str(program.get("DS", "")) for program in data.get("PG", []))
    if any("ddda_phase_posterior_v1" in text for text in provenance):
        return "rotational-recall"
    return "input-ma"


def load_duplex_model(path: Optional[str] = None,
                      call_layer: str = "input-ma") -> DuplexModel:
    if path:
        model_path = Path(path)
    else:
        filename = (ROTATIONAL_MODEL_FILENAME
                    if call_layer == "rotational-recall" else MODEL_FILENAME)
        model_path = Path(__file__).parents[1] / "models" / filename
    with model_path.open() as handle:
        data = json.load(handle)
    names = tuple(data["feature_names"])
    mean = np.asarray(data["standard_scaler_mean"], dtype=float)
    scale = np.asarray(data["standard_scaler_scale"], dtype=float)
    coefficients = np.asarray(data["logistic_coefficients"], dtype=float)
    if not (len(names) == len(mean) == len(scale) == len(coefficients)):
        raise ValueError(f"invalid duplex model dimensions in {model_path}")
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError(f"invalid duplex model scales in {model_path}")
    return DuplexModel(
        model_id=str(data["model_id"]),
        feature_names=names,
        mean=mean,
        scale=scale,
        coefficients=coefficients,
        intercept=float(data["logistic_intercept"]),
        metadata=data,
    )


def build_pattern_feature(read, index: int, params: DuplexParams,
                          prob_threshold: int = 0) -> Optional[ReadFeat]:
    """Build a dyad feature without deriving any sequence-identity signature."""
    if read.is_unmapped or read.is_secondary or read.is_supplementary:
        return None
    if read.query_sequence is None or not read.has_tag("MA"):
        return None
    flavor = read_flavor(read, prob_threshold)
    if flavor is None:
        return None
    dyads = nuc_dyads_ref(read)
    grid = params.grid_bp
    grid0 = int(read.reference_start) // grid
    grid1 = int(read.reference_end) // grid
    kernel = _gaussian_kernel(params.dyad_sigma_bp, grid)
    radius = len(kernel) // 2
    signal = np.zeros(grid1 - grid0 + 1, dtype=np.float32)
    for center in dyads:
        center_bin = int(center) // grid - grid0
        lo = max(0, center_bin - radius)
        hi = min(len(signal), center_bin + radius + 1)
        if hi > lo:
            kernel_lo = lo - (center_bin - radius)
            signal[lo:hi] += kernel[kernel_lo:kernel_lo + hi - lo]
    return ReadFeat(
        index=index,
        name=read.query_name,
        flavor=flavor,
        ref_start=int(read.reference_start),
        ref_end=int(read.reference_end),
        dyads=dyads,
        grid0=grid0,
        signal=signal,
        sequence_pos=None,
        sequence_base=None,
    )


def _profile_kernel(params: DuplexParams) -> np.ndarray:
    width = params.profile_sigma_bp / params.profile_bin_bp
    radius = max(1, int(round(3 * width)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / width) ** 2)
    return kernel / kernel.sum()


def build_protection_profile(read, reference: np.ndarray, params: DuplexParams,
                             prob_threshold: int = 0,
                             flavor: Optional[int] = None) -> ProtectionProfile:
    """Build the validated 20-bp non-CpG DddA profile for one read.

    The reference is used only to enumerate C/G enzyme opportunities.  Query
    A/T allele agreement is neither derived nor retained.
    """
    query_to_ref = _build_query_to_ref(read)
    calls = _deam_positions_list(read, query_to_ref, prob_threshold)
    if flavor is None:
        ct = sum(call_flavor == FLAVOR_CT for _, call_flavor in calls)
        ga = len(calls) - ct
        if ct == ga:
            raise ValueError(f"read {read.query_name!r} has no dominant DddA flavor")
        flavor = FLAVOR_CT if ct > ga else FLAVOR_GA
    valid = (query_to_ref >= 0) & (query_to_ref < len(reference))
    positions = query_to_ref[valid]
    if flavor == FLAVOR_CT:
        opportunity = (reference[positions] == ord("C")) & (positions + 1 < len(reference))
        opportunity &= reference[np.minimum(positions + 1, len(reference) - 1)] != ord("G")
    else:
        opportunity = (reference[positions] == ord("G")) & (positions > 0)
        opportunity &= reference[np.maximum(positions - 1, 0)] != ord("C")
    opportunities = np.unique(positions[opportunity])
    hits = np.asarray(
        [position for position, call_flavor in calls if call_flavor == flavor],
        dtype=np.int64,
    )
    hits = np.intersect1d(hits, opportunities, assume_unique=False)
    bin_bp = params.profile_bin_bp
    start_bin = int(read.reference_start) // bin_bp
    end_bin = (int(read.reference_end) + bin_bp - 1) // bin_bp
    length = end_bin - start_bin + 1
    opportunity_counts = np.bincount(
        opportunities // bin_bp - start_bin, minlength=length,
    ).astype(np.float32)
    hit_counts = np.bincount(
        hits // bin_bp - start_bin, minlength=length,
    ).astype(np.float32)
    kernel = _profile_kernel(params)
    weight = np.convolve(opportunity_counts, kernel, "same").astype(np.float32)
    numerator = np.convolve(hit_counts, kernel, "same")
    rate = (numerator / np.maximum(weight, 1e-5)).astype(np.float32)
    return ProtectionProfile(
        name=read.query_name,
        flavor=int(flavor),
        start_bin=start_bin,
        end_bin=end_bin,
        rate=rate,
        weight=weight,
        opportunities=len(opportunities),
        hits=len(hits),
    )


def component_residual_profiles(
    profiles: Mapping[int, ProtectionProfile],
) -> Dict[int, ProtectionProfile]:
    """Subtract the opportunity-weighted profile common to a component."""
    if not profiles:
        return {}
    lo = min(profile.start_bin for profile in profiles.values())
    hi = max(profile.end_bin for profile in profiles.values()) + 1
    numerator = np.zeros(hi - lo, dtype=np.float64)
    weight = np.zeros(hi - lo, dtype=np.float64)
    for profile in profiles.values():
        start = profile.start_bin - lo
        stop = start + len(profile.rate)
        numerator[start:stop] += profile.rate * profile.weight
        weight[start:stop] += profile.weight
    mean = numerator / np.maximum(weight, 1e-5)
    residual = {}
    for index, profile in profiles.items():
        start = profile.start_bin - lo
        stop = start + len(profile.rate)
        residual[index] = ProtectionProfile(
            name=profile.name,
            flavor=profile.flavor,
            start_bin=profile.start_bin,
            end_bin=profile.end_bin,
            rate=(profile.rate - mean[start:stop]).astype(np.float32),
            weight=profile.weight,
            opportunities=profile.opportunities,
            hits=profile.hits,
        )
    return residual


def profile_correlation(a: ProtectionProfile, b: ProtectionProfile,
                        params: DuplexParams) -> Optional[float]:
    lo = max(a.start_bin, b.start_bin)
    hi = min(a.end_bin + 1, b.end_bin + 1)
    if (hi - lo) * params.profile_bin_bp < params.min_overlap_bp:
        return None
    a_start, b_start = lo - a.start_bin, lo - b.start_bin
    ar = a.rate[a_start:a_start + hi - lo]
    br = b.rate[b_start:b_start + hi - lo]
    aw = a.weight[a_start:a_start + hi - lo]
    bw = b.weight[b_start:b_start + hi - lo]
    valid = ((aw >= params.min_opportunities_per_bin) &
             (bw >= params.min_opportunities_per_bin))
    if int(valid.sum()) < params.min_profile_bins:
        return None
    x, y = ar[valid].astype(float), br[valid].astype(float)
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denominator <= 1e-7:
        return None
    return float(np.dot(x, y) / denominator)


def _matched(a: np.ndarray, b: np.ndarray, tolerance: int, shift: int) -> int:
    i = j = hits = 0
    while i < len(a) and j < len(b):
        delta = int(a[i]) - int(b[j]) - shift
        if abs(delta) <= tolerance:
            hits += 1
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return hits


def _anchor_f1(a: np.ndarray, b: np.ndarray, tolerance: int,
               max_shift: int) -> float:
    choices = [
        (_matched(a, b, tolerance, shift), shift)
        for shift in range(-max_shift, max_shift + 1, 10)
    ]
    hits, _ = max(choices, key=lambda item: (item[0], -abs(item[1])))
    return 2.0 * hits / (len(a) + len(b))


def edge_features(a: ReadFeat, b: ReadFeat,
                  raw_a: ProtectionProfile, raw_b: ProtectionProfile,
                  residual_a: ProtectionProfile, residual_b: ProtectionProfile,
                  params: DuplexParams) -> Optional[Dict[str, float]]:
    pair_params = params.pair_params()
    baseline = score_pair(a, b, pair_params)
    if baseline is None:
        return None
    lo = max(a.grid0, b.grid0) * params.grid_bp
    hi = min(a.grid0 + len(a.signal), b.grid0 + len(b.signal)) * params.grid_bp
    dyads_a = a.dyads[(a.dyads >= lo) & (a.dyads < hi)]
    dyads_b = b.dyads[(b.dyads >= lo) & (b.dyads < hi)]
    overlap = hi - lo
    min_dyads = min(len(dyads_a), len(dyads_b))
    raw = profile_correlation(raw_a, raw_b, params)
    residual = profile_correlation(residual_a, residual_b, params)
    if raw is None or residual is None:
        return None
    span = max(0, min(a.ref_end, b.ref_end) - max(a.ref_start, b.ref_start))
    union = max(a.ref_end, b.ref_end) - min(a.ref_start, b.ref_start)
    return {
        "baseline_score": float(baseline),
        "anchor_f1_tol15_shift0": _anchor_f1(dyads_a, dyads_b, 15, 0),
        "anchor_f1_tol25_shift30": _anchor_f1(dyads_a, dyads_b, 25, 30),
        "anchor_f1_tol40_shift0": _anchor_f1(dyads_a, dyads_b, 40, 0),
        "log_overlap": float(np.log(overlap)),
        "log_min_dyads": float(np.log(min_dyads)),
        "dyad_balance": min_dyads / max(len(dyads_a), len(dyads_b)),
        "span_jaccard": span / union,
        "raw_protection_sigma20_lag0": raw,
        "residual_all_reads_sigma20_lag0": residual,
        "overlap_bp": float(overlap),
        "ct_dyads": float(len(dyads_a) if a.flavor == FLAVOR_CT else len(dyads_b)),
        "ga_dyads": float(len(dyads_b) if b.flavor == FLAVOR_GA else len(dyads_a)),
    }


def _opposite_edges(features: Sequence[ReadFeat]) -> Iterable[Tuple[ReadFeat, ReadFeat]]:
    order = sorted(features, key=lambda feature: (feature.ref_start, feature.name))
    for ai, a in enumerate(order):
        for b in order[ai + 1:]:
            if b.ref_start >= a.ref_end:
                break
            if a.flavor != b.flavor:
                yield a, b


def assign_duplex_pairs(features: Sequence[ReadFeat],
                        profiles: Mapping[int, ProtectionProfile],
                        model: DuplexModel, params: DuplexParams) -> DuplexResult:
    """Score a complete overlap component and select reciprocal-best pairs."""
    residual = component_residual_profiles(profiles)
    geometric_nodes: Set[int] = set()
    offers: Dict[int, List[Tuple[float, str, int]]] = {
        feature.index: [] for feature in features
    }
    evidence: Dict[Tuple[int, int], EdgeEvidence] = {}
    geometric_edges = 0
    for a, b in _opposite_edges(features):
        overlap = min(a.ref_end, b.ref_end) - max(a.ref_start, b.ref_start)
        if overlap < params.min_overlap_bp:
            continue
        geometric_edges += 1
        geometric_nodes.update((a.index, b.index))
        values = edge_features(
            a, b, profiles[a.index], profiles[b.index],
            residual[a.index], residual[b.index], params,
        )
        if values is None:
            continue
        score = model.decision_function(values)
        if not np.isfinite(score):
            continue
        ct, ga = (a, b) if a.flavor == FLAVOR_CT else (b, a)
        edge = EdgeEvidence(ct.index, ga.index, score, values)
        evidence[(ct.index, ga.index)] = edge
        offers[a.index].append((score, b.name, b.index))
        offers[b.index].append((score, a.name, a.index))

    best: Dict[int, Tuple[float, str, int]] = {}
    second: Dict[int, float] = {}
    for index, choices in offers.items():
        choices.sort(key=lambda item: (-item[0], item[1]))
        if choices:
            best[index] = choices[0]
            second[index] = choices[1][0] if len(choices) > 1 else float("-inf")

    partner: Dict[int, int] = {}
    selected_score: Dict[int, float] = {}
    margin: Dict[int, float] = {}
    for index in sorted(best):
        score, _, mate = best[index]
        if mate not in best or best[mate][2] != index:
            continue
        gap = min(
            score - max(second[index], params.null_floor),
            score - max(second[mate], params.null_floor),
        )
        if gap < params.min_margin:
            continue
        partner[index] = mate
        selected_score[index] = score
        margin[index] = gap

    status = {}
    for feature in features:
        if feature.index in partner:
            status[feature.index] = STATUS_PAIRED
        elif feature.index in geometric_nodes:
            status[feature.index] = STATUS_UNRESOLVED
        else:
            status[feature.index] = STATUS_NONE
    return DuplexResult(
        partner=partner,
        score=selected_score,
        margin=margin,
        status=status,
        evidence=evidence,
        geometric_edges=geometric_edges,
        scored_edges=len(evidence),
    )
