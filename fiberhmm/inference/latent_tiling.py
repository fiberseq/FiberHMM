"""Posterior scoring for locus-specific protected-block tilings.

The ordinary FiberHMM call layers describe one molecule at a time.  At deep
targeted coverage, recurrent TF-family boundaries can additionally explain a
continuous protected block as several adjacent biological objects (for
example, a site-consensus state abutting a nucleosome).  Those explanations can be
chemically indistinguishable on one molecule when no accessible linker lies
between them.  This module deliberately keeps that distinction explicit:
raw chemistry scores the union of protected intervals, while frozen
population/topology terms score how that union is partitioned.

The functions here do not modify calls or evidence.  Candidate construction
and prior fitting are separate so they can be cross-fitted by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Callable, Hashable, Iterable, Mapping, Sequence

import numpy as np

from fiberhmm.inference.strand_rescue import ReadEvidence
from fiberhmm.inference.nuc_recaller import NucProfile


PROTECTED_KINDS = frozenset({"nuc", "tf_family", "unanchored_tf"})


@dataclass(frozen=True)
class TilingSegment:
    """One labeled interval in a candidate protected-block explanation."""

    kind: str
    start: int
    end: int
    family_id: str | None = None
    dyad: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {
            "nuc", "tf_family", "unanchored_tf", "accessible"
        }:
            raise ValueError(f"unsupported tiling segment kind: {self.kind!r}")
        if self.end <= self.start:
            raise ValueError("tiling segments must have positive width")
        if self.kind == "tf_family" and not self.family_id:
            raise ValueError("TF-family segments require a family id")
        if self.kind != "tf_family" and self.family_id is not None:
            raise ValueError("only TF-family segments can carry a family id")
        if self.kind != "nuc" and self.dyad is not None:
            raise ValueError("only nucleosome segments can carry a dyad")
        if self.dyad is not None and not self.start <= self.dyad < self.end:
            raise ValueError("nucleosome dyad must lie inside its protected span")

    @property
    def protected(self) -> bool:
        return self.kind in PROTECTED_KINDS


@dataclass(frozen=True)
class TilingConfiguration:
    """One complete candidate explanation and its frozen structural term.

    ``log_structural_weight`` may contain cross-fitted family occupancy,
    boundary, nucleosome-length, linker, and topology terms.  It need not be
    normalized across configurations; normalization occurs during scoring.
    """

    configuration_id: str
    segments: tuple[TilingSegment, ...]
    log_structural_weight: float = 0.0
    provenance: str = "unspecified"

    def __post_init__(self) -> None:
        if not self.configuration_id:
            raise ValueError("tiling configurations require a non-empty id")
        if not self.segments:
            raise ValueError("tiling configurations require at least one segment")
        if not math.isfinite(self.log_structural_weight):
            raise ValueError("tiling structural weights must be finite")
        ordered = sorted(self.segments, key=lambda value: (value.start, value.end))
        protected = [segment for segment in ordered if segment.protected]
        if any(left.end > right.start for left, right in zip(protected, protected[1:])):
            raise ValueError("protected segments in one tiling may not overlap")


@dataclass(frozen=True)
class DddARadialChemistryScorer:
    """DddA chemistry scorer with TF protection and radial nucleosomes.

    The accessible/linker state is the likelihood-ratio baseline.  TF-family
    segments use FiberHMM's context-aware protected LLR already stored in
    ``ReadEvidence.steps``.  Nucleosome segments use the empirical radial
    deamination profile against either FiberHMM's context-aware accessible
    rates (preferred) or the profile linker rate, with both numerator and
    denominator multiplied by the molecule's calibrated
    ``efficiency_factor``.  Candidate construction is responsible for
    marginalizing alternative dyads as separate, properly weighted
    configurations.
    """

    profile: NucProfile
    accessible_hit_rates: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not 0.0 < float(self.profile.linker) < 1.0:
            raise ValueError("DddA linker rate must lie in (0, 1)")
        radial = np.asarray(self.profile.radial, dtype=np.float64)
        finite = radial[np.isfinite(radial)]
        if finite.size == 0 or np.any((finite < 0.0) | (finite > 1.0)):
            raise ValueError(
                "finite DddA radial rates must lie in [0, 1]"
            )
        if self.accessible_hit_rates is not None:
            accessible = np.asarray(self.accessible_hit_rates, dtype=np.float64)
            if (
                accessible.ndim != 1
                or accessible.size == 0
                or np.any(~np.isfinite(accessible))
                or np.any((accessible <= 0.0) | (accessible >= 1.0))
            ):
                raise ValueError(
                    "context-aware accessible hit rates must be a finite 1D "
                    "array in (0, 1)"
                )

    def _prepared_probabilities(
        self, read: ReadEvidence
    ) -> tuple[float, np.ndarray]:
        efficiency = float(getattr(read, "efficiency_factor", 1.0))
        if not math.isfinite(efficiency) or efficiency <= 0.0:
            raise ValueError("read efficiency factor must be finite and positive")
        # The empirical radial template and linker rate live on the same raw
        # deamination-probability scale.  Apply the same molecule-specific
        # detection factor used to rebuild ``read.steps`` so TF and
        # nucleosome alternatives remain comparable within a molecule.
        fixed_linker = float(self.profile.linker) * efficiency
        if self.accessible_hit_rates is None and not 0.0 < fixed_linker < 1.0:
            raise ValueError(
                "efficiency-adjusted DddA linker rate must lie in (0, 1)"
            )
        # Match the production radial recaller: uncovered/sparse offsets in
        # the empirical template use a conservative protected rate of 0.05.
        base_radial = np.clip(
            np.nan_to_num(
                np.asarray(self.profile.radial, dtype=np.float64), nan=0.05
            ),
            0.01,
            0.6,
        )
        radial = base_radial * efficiency
        if np.any((radial <= 0.0) | (radial >= 1.0)):
            raise ValueError(
                "efficiency-adjusted DddA radial rates must lie in (0, 1)"
            )
        return fixed_linker, radial

    def score_segment(
        self,
        read: ReadEvidence,
        segment: TilingSegment,
    ) -> tuple[float, int, Hashable]:
        """Score one segment so callers can cache repeated tiling components."""
        if not segment.protected:
            return 0.0, 0, ("accessible", segment.start, segment.end)
        fixed_linker, radial = self._prepared_probabilities(read)
        efficiency = float(getattr(read, "efficiency_factor", 1.0))
        lo = int(np.searchsorted(read.positions, segment.start, side="left"))
        hi = int(np.searchsorted(read.positions, segment.end, side="left"))
        opportunities = hi - lo
        if segment.kind in {"tf_family", "unanchored_tf"}:
            return (
                float(np.sum(read.steps[lo:hi])),
                opportunities,
                ("tf", segment.start, segment.end),
            )
        # The called protected span and the nucleosome dyad are distinct
        # quantities.  In particular, a broad baseline block can remain fully
        # protected while candidate dyads are marginalized within it.
        center = (
            float(segment.dyad)
            if segment.dyad is not None
            else (segment.start + segment.end - 1) / 2.0
        )
        offsets = np.rint(np.abs(read.positions[lo:hi] - center)).astype(
            np.int64
        )
        rates = radial[np.minimum(offsets, radial.size - 1)]
        hits = read.hits[lo:hi]
        if self.accessible_hit_rates is None:
            baseline = np.full(rates.size, fixed_linker, dtype=np.float64)
        else:
            if np.any(read.contexts[lo:hi] >= self.accessible_hit_rates.size):
                raise ValueError(
                    "read context index exceeds accessible-hit rate table"
                )
            baseline = np.clip(
                np.asarray(
                    self.accessible_hit_rates[read.contexts[lo:hi]],
                    dtype=np.float64,
                ) * efficiency,
                1e-6,
                1.0 - 1e-6,
            )
        score = 0.0
        if np.any(hits):
            score += float(np.sum(np.log(rates[hits] / baseline[hits])))
        if np.any(~hits):
            score += float(
                np.sum(
                    np.log(
                        (1.0 - rates[~hits]) / (1.0 - baseline[~hits])
                    )
                )
            )
        return (
            score,
            opportunities,
            ("nuc_radial", segment.start, segment.end, center),
        )

    def __call__(
        self,
        read: ReadEvidence,
        configuration: TilingConfiguration,
    ) -> tuple[float, int, Hashable]:
        score = 0.0
        opportunities = 0
        equality_segments = []
        for segment in sorted(
            configuration.segments,
            key=lambda value: (value.start, value.end, value.kind),
        ):
            if not segment.protected:
                continue
            segment_score, segment_opportunities, equality_key = (
                self.score_segment(read, segment)
            )
            score += segment_score
            opportunities += segment_opportunities
            equality_segments.append(equality_key)
        return score, opportunities, tuple(equality_segments)


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    ordered = sorted({(int(start), int(end)) for start, end in intervals})
    if not ordered:
        return ()
    merged: list[list[int]] = []
    for start, end in ordered:
        if end <= start:
            raise ValueError("protected intervals must have positive width")
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((start, end) for start, end in merged)


def _chemistry_key(configuration: TilingConfiguration) -> tuple[tuple[int, int], ...]:
    return _merge_intervals(
        (segment.start, segment.end)
        for segment in configuration.segments
        if segment.protected
    )


def _stable_softmax(log_values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(log_values))
    shifted = np.exp(log_values - maximum)
    return shifted / float(np.sum(shifted))


def normalize_configuration_priors(
    configurations: Sequence[TilingConfiguration],
) -> tuple[tuple[TilingConfiguration, ...], float]:
    """Normalize one hypothesis group's retained structural prior mass.

    Candidate builders commonly discard impossible placements after assigning
    grid weights.  Normalizing *after* those removals prevents the number of
    invalid placements or the proposal-grid resolution from silently changing
    the top-level hypothesis prior.
    """
    if not configurations:
        raise ValueError("cannot normalize an empty configuration group")
    raw = np.asarray(
        [configuration.log_structural_weight for configuration in configurations],
        dtype=np.float64,
    )
    maximum = float(np.max(raw))
    log_normalizer = maximum + math.log(float(np.sum(np.exp(raw - maximum))))
    return (
        tuple(
            TilingConfiguration(
                configuration.configuration_id,
                configuration.segments,
                log_structural_weight=(
                    configuration.log_structural_weight - log_normalizer
                ),
                provenance=configuration.provenance,
            )
            for configuration in configurations
        ),
        log_normalizer,
    )


def score_tiling_groups(
    read: ReadEvidence,
    groups: Mapping[str, Sequence[TilingConfiguration]],
    *,
    chemistry_scorer: Callable[
        [ReadEvidence, TilingConfiguration], tuple[float, int, Hashable]
    ]
    | None = None,
) -> dict:
    """Stream marginal evidence for normalized hypothesis groups.

    Unlike :func:`score_tiling_configurations`, this path does not materialize
    a record and posterior for every placement.  It is intended for large
    nuisance grids where only group-level evidence and the MAP placement are
    needed.  Call :func:`normalize_configuration_priors` on each group first
    when equal top-level group priors are intended.
    """
    if not groups or any(not configurations for configurations in groups.values()):
        raise ValueError("tiling groups must be non-empty and contain configurations")
    envelope_start = min(
        segment.start
        for configurations in groups.values()
        for configuration in configurations
        for segment in configuration.segments
    )
    envelope_end = max(
        segment.end
        for configurations in groups.values()
        for configuration in configurations
        for segment in configuration.segments
    )
    if not read.fully_maps(envelope_start, envelope_end):
        return {
            "schema": "fiberhmm.latent_protected_tiling_group_score.v1",
            "status": "ineligible_incomplete_mapping",
            "molecule_id": list(read.molecule_id),
            "envelope": [envelope_start, envelope_end],
            "groups": {},
            "raw_calls_modified": False,
        }

    records = {}
    group_log_evidence = []
    group_names = list(groups)
    segment_scorer = (
        getattr(chemistry_scorer, "score_segment", None)
        if chemistry_scorer is not None else None
    )
    segment_cache: dict[TilingSegment, tuple[float, int, Hashable]] = {}
    for group_name in group_names:
        log_evidence = -math.inf
        best_configuration = None
        best_log_joint = -math.inf
        best_chemistry = -math.inf
        best_opportunities = 0
        for configuration in groups[group_name]:
            if chemistry_scorer is None:
                chemistry = 0.0
                opportunities = 0
                for start, end in _chemistry_key(configuration):
                    value, count, _hits = read.interval_evidence(start, end)
                    chemistry += value
                    opportunities += count
            elif callable(segment_scorer):
                chemistry = 0.0
                opportunities = 0
                equality_parts = []
                for segment in sorted(
                    configuration.segments,
                    key=lambda value: (value.start, value.end, value.kind),
                ):
                    if not segment.protected:
                        continue
                    if segment not in segment_cache:
                        segment_cache[segment] = segment_scorer(read, segment)
                    value, count, equality_part = segment_cache[segment]
                    chemistry += float(value)
                    opportunities += int(count)
                    equality_parts.append(equality_part)
                _equality_key = tuple(equality_parts)
            else:
                chemistry, opportunities, _equality_key = chemistry_scorer(
                    read, configuration
                )
            if not math.isfinite(float(chemistry)) or int(opportunities) < 0:
                raise ValueError("chemistry scorer returned invalid values")
            log_joint = float(chemistry) + configuration.log_structural_weight
            log_evidence = float(np.logaddexp(log_evidence, log_joint))
            if (
                log_joint > best_log_joint
                or (
                    log_joint == best_log_joint
                    and best_configuration is not None
                    and configuration.configuration_id
                    < best_configuration.configuration_id
                )
            ):
                best_configuration = configuration
                best_log_joint = log_joint
                best_chemistry = float(chemistry)
                best_opportunities = int(opportunities)
        assert best_configuration is not None
        group_log_evidence.append(log_evidence)
        records[group_name] = {
            "configuration_count": len(groups[group_name]),
            "log_evidence": log_evidence,
            "map_configuration_id": best_configuration.configuration_id,
            "map_configuration_posterior_within_group": float(
                math.exp(best_log_joint - log_evidence)
            ),
            "map_chemistry_log_likelihood": best_chemistry,
            "map_chemistry_opportunities": best_opportunities,
            "map_segments": [
                {
                    "kind": segment.kind,
                    "start": int(segment.start),
                    "end": int(segment.end),
                    "family_id": segment.family_id,
                    "dyad": (
                        int(segment.dyad) if segment.dyad is not None else None
                    ),
                }
                for segment in best_configuration.segments
            ],
        }
    posteriors = _stable_softmax(np.asarray(group_log_evidence, dtype=np.float64))
    for group_name, posterior in zip(group_names, posteriors):
        records[group_name]["posterior"] = float(posterior)
    return {
        "schema": "fiberhmm.latent_protected_tiling_group_score.v1",
        "status": "scored_normalized_groups",
        "molecule_id": list(read.molecule_id),
        "envelope": [envelope_start, envelope_end],
        "groups": records,
        "raw_calls_modified": False,
    }


def _chemistry_class_id(equality_key: Hashable) -> str:
    encoded = repr(equality_key)
    return "chemistry_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def score_tiling_configurations(
    read: ReadEvidence,
    configurations: Sequence[TilingConfiguration],
    *,
    chemistry_scorer: Callable[
        [ReadEvidence, TilingConfiguration], tuple[float, int, Hashable]
    ]
    | None = None,
    minimum_resolved_posterior: float = 0.9,
    minimum_runner_up_log_odds: float = math.log(10.0),
) -> dict:
    """Score frozen tilings on one molecule without rewriting its raw calls.

    By default, chemistry is the protected-versus-accessible log likelihood
    already carried by :class:`ReadEvidence`.  Adjacent protected segments are
    scored once through their union.  A caller may instead supply a frozen
    ``chemistry_scorer`` (for example, one that uses the DddA radial
    nucleosome profile).  Its third return value is an equality key: two
    configurations sharing that key explicitly make the same chemistry
    prediction and differ only through population/topology terms.
    """
    if not configurations:
        raise ValueError("at least one tiling configuration is required")
    if not 0.0 < minimum_resolved_posterior <= 1.0:
        raise ValueError("minimum resolved posterior must lie in (0, 1]")
    if minimum_runner_up_log_odds < 0.0 or not math.isfinite(
        minimum_runner_up_log_odds
    ):
        raise ValueError("minimum runner-up log odds must be finite and non-negative")
    ids = [configuration.configuration_id for configuration in configurations]
    if len(set(ids)) != len(ids):
        raise ValueError("tiling configuration ids must be unique")

    envelope_start = min(
        segment.start
        for configuration in configurations
        for segment in configuration.segments
    )
    envelope_end = max(
        segment.end
        for configuration in configurations
        for segment in configuration.segments
    )
    if not read.fully_maps(envelope_start, envelope_end):
        return {
            "schema": "fiberhmm.latent_protected_tiling_score.v1",
            "status": "ineligible_incomplete_mapping",
            "molecule_id": list(read.molecule_id),
            "envelope": [envelope_start, envelope_end],
            "configurations": [],
            "family_marginal_posteriors": {},
            "decision": "ineligible",
            "decision_basis": "incomplete_mapping",
        }

    protected_unions = [_chemistry_key(configuration) for configuration in configurations]
    chemistry_keys: list[Hashable] = []
    chemistry_score_values: list[float] = []
    chemistry_opportunity_values: list[int] = []
    if chemistry_scorer is None:
        default_scores: dict[tuple[tuple[int, int], ...], float] = {}
        default_opportunities: dict[tuple[tuple[int, int], ...], int] = {}
        for key in set(protected_unions):
            score = 0.0
            opportunities = 0
            for start, end in key:
                interval_score, interval_opportunities, _hits = read.interval_evidence(
                    start, end
                )
                score += interval_score
                opportunities += interval_opportunities
            default_scores[key] = score
            default_opportunities[key] = opportunities
        chemistry_keys = list(protected_unions)
        chemistry_score_values = [default_scores[key] for key in protected_unions]
        chemistry_opportunity_values = [
            default_opportunities[key] for key in protected_unions
        ]
    else:
        for configuration in configurations:
            score, opportunities, equality_key = chemistry_scorer(
                read, configuration
            )
            if not math.isfinite(float(score)) or int(opportunities) < 0:
                raise ValueError("custom chemistry scorer returned invalid values")
            try:
                hash(equality_key)
            except TypeError as error:
                raise ValueError(
                    "custom chemistry equality keys must be hashable"
                ) from error
            chemistry_keys.append(equality_key)
            chemistry_score_values.append(float(score))
            chemistry_opportunity_values.append(int(opportunities))

    log_joint = np.asarray(
        [
            chemistry_score + configuration.log_structural_weight
            for chemistry_score, configuration in zip(
                chemistry_score_values, configurations
            )
        ],
        dtype=np.float64,
    )
    posteriors = _stable_softmax(log_joint)
    order = sorted(
        range(len(configurations)),
        key=lambda index: (-float(posteriors[index]), configurations[index].configuration_id),
    )
    best_index = order[0]
    runner_up_index = order[1] if len(order) > 1 else None
    best_posterior = float(posteriors[best_index])
    runner_up_log_odds = (
        float(log_joint[best_index] - log_joint[runner_up_index])
        if runner_up_index is not None
        else math.inf
    )
    resolved = (
        best_posterior >= minimum_resolved_posterior
        and runner_up_log_odds >= minimum_runner_up_log_odds
    )
    same_chemistry_as_runner_up = bool(
        runner_up_index is not None
        and chemistry_keys[best_index] == chemistry_keys[runner_up_index]
    )
    if not resolved:
        decision = "unresolved"
        decision_basis = "posterior_or_runner_up_margin_below_threshold"
    elif same_chemistry_as_runner_up:
        decision = configurations[best_index].configuration_id
        decision_basis = "population_topology_with_chemistry_equivalent_runner_up"
    else:
        decision = configurations[best_index].configuration_id
        decision_basis = "chemistry_and_population_topology"

    families = sorted(
        {
            segment.family_id
            for configuration in configurations
            for segment in configuration.segments
            if segment.family_id is not None
        }
    )
    family_marginals = {
        family_id: float(
            sum(
                posteriors[index]
                for index, configuration in enumerate(configurations)
                if any(
                    segment.family_id == family_id
                    for segment in configuration.segments
                )
            )
        )
        for family_id in families
    }
    records = []
    for index, (configuration, key, protected_union) in enumerate(
        zip(configurations, chemistry_keys, protected_unions)
    ):
        records.append(
            {
                "configuration_id": configuration.configuration_id,
                "segments": [
                    {
                        "kind": segment.kind,
                        "start": int(segment.start),
                        "end": int(segment.end),
                        "family_id": segment.family_id,
                    }
                    for segment in configuration.segments
                ],
                "chemistry_equivalence_class": _chemistry_class_id(key),
                "protected_union": [list(interval) for interval in protected_union],
                "chemistry_log_likelihood": float(
                    chemistry_score_values[index]
                ),
                "chemistry_opportunities": int(
                    chemistry_opportunity_values[index]
                ),
                "log_structural_weight": float(configuration.log_structural_weight),
                "log_joint": float(log_joint[index]),
                "posterior": float(posteriors[index]),
                "provenance": configuration.provenance,
            }
        )
    records.sort(key=lambda record: (-record["posterior"], record["configuration_id"]))
    positive = posteriors[posteriors > 0.0]
    return {
        "schema": "fiberhmm.latent_protected_tiling_score.v1",
        "status": "scored_frozen_configurations",
        "molecule_id": list(read.molecule_id),
        "envelope": [envelope_start, envelope_end],
        "configurations": records,
        "family_marginal_posteriors": family_marginals,
        "maximum_configuration_posterior": best_posterior,
        "runner_up_log_odds": runner_up_log_odds,
        "configuration_entropy_nats": float(-np.sum(positive * np.log(positive))),
        "decision": decision,
        "decision_basis": decision_basis,
        "minimum_resolved_posterior": float(minimum_resolved_posterior),
        "minimum_runner_up_log_odds": float(minimum_runner_up_log_odds),
        "raw_calls_modified": False,
    }


def family_quantification_weights(
    scores: Sequence[Mapping[str, object]],
    *,
    minimum_marginal_posterior: float = 0.9,
) -> dict[str, dict[str, float | int]]:
    """Summarize soft and conservative family occupancy across molecules."""
    if not 0.0 < minimum_marginal_posterior <= 1.0:
        raise ValueError("minimum marginal posterior must lie in (0, 1]")
    values: dict[str, list[float]] = {}
    for score in scores:
        marginals = score.get("family_marginal_posteriors", {})
        if not isinstance(marginals, Mapping):
            raise ValueError("family marginal posteriors must be a mapping")
        for family_id, raw_value in marginals.items():
            value = float(raw_value)
            if not 0.0 <= value <= 1.0 or not math.isfinite(value):
                raise ValueError("family marginal posteriors must lie in [0, 1]")
            values.setdefault(str(family_id), []).append(value)
    return {
        family_id: {
            "eligible_molecules": len(posteriors),
            "posterior_weighted_occupancy": float(sum(posteriors)),
            "conservative_occupied_molecules": sum(
                posterior >= minimum_marginal_posterior
                for posterior in posteriors
            ),
            "minimum_marginal_posterior": float(minimum_marginal_posterior),
        }
        for family_id, posteriors in sorted(values.items())
    }
