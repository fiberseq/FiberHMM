#!/usr/bin/env python3
"""Materialize consensus-recaller decisions as non-destructive MA callsets.

The ordinary ``nuc``/``tf`` annotations and their qualities remain unchanged.
This writer adds four optional, posterior-scored logical MA feature groups:

``nuc_cr.Q`` / ``tf_cr.Q``
    Complete nucleosome and TF callsets after the population composite
    recaller.  A recalled nucleosome is removed from ``nuc_cr`` and its
    selected replacement is added to ``tf_cr``; the two are not simultaneous
    alternative hypotheses. Any carried or new TF also overwrites an
    overlapping shadow nucleosome.

``nuc_sr.Q`` / ``tf_sr.Q``
    Complete nucleosome and TF callsets after physical strand rescue.

The legacy selected-state modes use one posterior Q byte.  ``--paired`` is the
aggressive focal mode: every locally supported decision is represented by
linked current/TF alternatives with ``QQQQQ`` qualities and stable ``AN``
labels.  The first qualities are exactly complementary and let FiberBrowser
move the molecule between the two states with one threshold.  Input BAMs are
never edited; the command writes small, indexed regional derivative BAMs for
visualization.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import shlex
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pysam

from consensus_recaller_collab.prototype import parse_region
from consensus_recaller_collab.validation import VALIDATION_VERSION
from fiberhmm import __version__ as FIBERHMM_VERSION
from fiberhmm.io.bam_header import append_ma_types, append_pg_record
from fiberhmm.io.ma_tags import flip_interval_frame, parse_an_tag, parse_aq_array, parse_ma_tag


LAYER_ORDER = ("nuc_cr", "tf_cr", "nuc_sr", "tf_sr")
CONSENSUS_HEADER_PREFIX = "FIBERHMM-CONSENSUS:v2:"
PAIRED_CONSENSUS_HEADER_PREFIX = "FIBERHMM-CONSENSUS:v3:"
PAIRED_QUALITY_SPEC = "QQQQQ"
PASS_LAYERS = {
    "cr": ("nuc_cr", "tf_cr"),
    "sr": ("nuc_sr", "tf_sr"),
}
TIER_ORDER = {"strong": 0, "review": 1, "retain_n": 2}


@dataclass(frozen=True)
class OverlayProposal:
    pass_name: str
    proposal_id: str
    read_name: str
    library_id: str
    tier: str
    current_interval: Tuple[int, int]
    tf_intervals: Tuple[Tuple[int, int], ...]
    tf_posterior: float
    nuc_posterior: Optional[float]
    current_state: str = "N"
    replacement_selected: bool = True


@dataclass(frozen=True)
class PairedDecision:
    """One aggressive current-vs-TF decision for linked MA rendering."""

    pass_name: str
    proposal_id: str
    read_name: str
    library_id: str
    tier: str
    current_state: str
    current_interval: Tuple[int, int]
    tf_intervals: Tuple[Tuple[int, int], ...]
    tf_posterior: float
    current_posterior: float
    configuration_posterior: float
    molecule_probability: float
    population_probability: float
    specificity_probability: float


ConsensusDecision = Union[OverlayProposal, PairedDecision]


def posterior_to_q(value: float) -> int:
    """Encode a posterior probability as the FiberHMM-style linear byte."""
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"posterior outside [0, 1]: {value!r}")
    return max(0, min(255, int(round(255.0 * float(value)))))


def log_odds_to_probability(value: Optional[float]) -> float:
    """Map a finite log odds/Bayes factor to an equal-odds probability."""
    if value is None:
        return 0.5
    value = float(value)
    if math.isnan(value):
        return 0.5
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(value, 745.0)))
    exp_value = math.exp(max(value, -745.0))
    return exp_value / (1.0 + exp_value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_path(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _selected_tier(
    tier: str,
    *,
    include_review: bool,
    include_retain_n: bool,
) -> bool:
    normalized = str(tier or "strong")
    if normalized not in TIER_ORDER:
        return False
    if normalized == "strong":
        return True
    if normalized == "review":
        return include_review or include_retain_n
    return include_retain_n


def _choose_scenario(
    scenarios: Mapping[str, object], report: Mapping[str, object]
) -> Optional[Mapping[str, object]]:
    if not scenarios:
        return None
    composite = report.get("composite_deconvolution", {})
    preferred = str(
        composite.get(
            "production_scenario_key",
            report.get("parameters", {}).get("production_nuc_prior_odds", ""),
        )
    )
    if preferred in scenarios:
        value = scenarios[preferred]
        return value if isinstance(value, Mapping) else None
    if len(scenarios) == 1:
        value = next(iter(scenarios.values()))
        return value if isinstance(value, Mapping) else None
    try:
        target = float(preferred)
        key = min(scenarios, key=lambda item: abs(float(item) - target))
    except (TypeError, ValueError):
        key = sorted(scenarios)[0]
    value = scenarios[key]
    return value if isinstance(value, Mapping) else None


def collect_overlay_proposals(
    report: Mapping[str, object],
    *,
    include_review: bool = True,
    include_retain_n: bool = False,
    allow_truncated: bool = False,
) -> List[OverlayProposal]:
    """Extract one production-scenario proposal record per read/state change."""
    proposals: Dict[Tuple[str, str, str], OverlayProposal] = {}

    composite = report.get("composite_deconvolution", {})
    if isinstance(composite, Mapping) and composite.get("enabled", False):
        scenario = _choose_scenario(composite.get("scenarios", {}), report)
        if scenario is not None:
            if (
                not include_retain_n
                and scenario.get("proposals_truncated")
                and not allow_truncated
            ):
                raise ValueError(
                    "composite proposal list was truncated; rerun with "
                    "--max-proposals 0 or pass --allow-truncated-report"
                )
            if include_retain_n:
                raw_records = scenario.get("candidate_states")
                if raw_records is None:
                    raise ValueError(
                        "report predates complete candidate-state output; rerun "
                        "fiberhmm-consensus-recall before using --all-tested"
                    )
            else:
                raw_records = scenario.get("proposals", [])
            for raw in raw_records:
                tier = str(raw.get("proposal_tier", "strong"))
                if not _selected_tier(
                    tier,
                    include_review=include_review,
                    include_retain_n=include_retain_n,
                ):
                    continue
                proposal = OverlayProposal(
                    pass_name="cr",
                    proposal_id=str(
                        raw.get("proposal_id") or raw["decision_id"]
                    ),
                    read_name=str(raw["read"]),
                    library_id=str(raw.get("library_id") or ""),
                    tier=tier,
                    current_state="N",
                    replacement_selected=(
                        str(raw.get(
                            "complex_top_state", raw.get("top_state", "TF")
                        )) != "N"
                    ),
                    current_interval=tuple(int(x) for x in raw["current_interval"]),
                    tf_intervals=tuple(
                        tuple(int(x) for x in interval)
                        for interval in raw.get("replacement_intervals", [])
                    ),
                    tf_posterior=float(raw.get(
                        "replacement_posterior", raw["best_tf_posterior"]
                    )),
                    nuc_posterior=float(raw["nuc_posterior"]),
                )
                proposals[
                    (proposal.pass_name, proposal.library_id, proposal.proposal_id)
                ] = proposal

    strand = report.get("strand_rescue", {})
    if isinstance(strand, Mapping) and strand.get("enabled", False):
        for window in strand.get("windows", []):
            for result in window.get("cross_strand", {}).values():
                scenario = _choose_scenario(result.get("scenarios", {}), report)
                if scenario is None:
                    continue
                if scenario.get("proposals_truncated") and not allow_truncated:
                    raise ValueError(
                        "strand-rescue proposal list was truncated; rerun with "
                        "--max-proposals 0 or pass --allow-truncated-report"
                    )
                for raw in scenario.get("proposals", []):
                    tier = str(raw.get("proposal_tier", "strong"))
                    if not _selected_tier(
                        tier,
                        include_review=include_review,
                        include_retain_n=False,
                    ):
                        continue
                    nuc_posterior = raw.get("nuc_posterior")
                    proposal = OverlayProposal(
                        pass_name="sr",
                        proposal_id=str(raw["proposal_id"]),
                        read_name=str(raw["read"]),
                        library_id=str(raw.get("library_id") or ""),
                        tier=tier,
                        current_state=str(raw.get("current", "A")),
                        replacement_selected=True,
                        current_interval=tuple(
                            int(x) for x in raw["current_interval"]
                        ),
                        tf_intervals=tuple(
                            tuple(int(x) for x in interval)
                            for interval in raw.get("proposed_site_intervals", [])
                        ),
                        tf_posterior=float(raw["posterior"]),
                        nuc_posterior=(
                            None if nuc_posterior is None else float(nuc_posterior)
                        ),
                    )
                    proposals[
                        (proposal.pass_name, proposal.library_id, proposal.proposal_id)
                    ] = proposal

    return sorted(
        proposals.values(),
        key=lambda item: (
            item.library_id,
            item.read_name,
            item.pass_name,
            item.current_interval,
            item.proposal_id,
        ),
    )


def _bounded_probability(value: object, *, default: float = 0.5) -> float:
    if value is None:
        return default
    result = float(value)
    if not math.isfinite(result):
        return default
    return max(0.0, min(1.0, result))


def collect_paired_decisions(
    report: Mapping[str, object],
) -> List[PairedDecision]:
    """Extract every aggressive CR/SR alternative from complete state tables."""
    decisions: Dict[Tuple[str, str, str], PairedDecision] = {}

    composite = report.get("composite_deconvolution", {})
    if isinstance(composite, Mapping) and composite.get("enabled", False):
        scenario = _choose_scenario(composite.get("scenarios", {}), report)
        if scenario is not None:
            raw_records = scenario.get("candidate_states")
            if raw_records is None:
                raise ValueError(
                    "report predates complete CR candidate-state output; rerun "
                    "fiberhmm-consensus-recall before using --paired"
                )
            for raw in raw_records:
                tf_intervals = tuple(
                    tuple(int(x) for x in interval)
                    for interval in raw.get("replacement_intervals", [])
                )
                if not tf_intervals:
                    continue
                tf_probability = _bounded_probability(
                    raw.get(
                        "complex_posterior",
                        1.0 - float(raw.get("nuc_posterior", 1.0)),
                    )
                )
                current_probability = _bounded_probability(
                    raw.get("nuc_posterior", 1.0 - tf_probability)
                )
                pair_total = tf_probability + current_probability
                if pair_total > 0.0:
                    tf_probability /= pair_total
                    current_probability = 1.0 - tf_probability
                local_prior = raw.get("local_complex_prior", {})
                population_probability = _bounded_probability(
                    local_prior.get(
                        "selected_complex_prior",
                        local_prior.get("wilson_lower_bound"),
                    ) if isinstance(local_prior, Mapping) else None
                )
                decision = PairedDecision(
                    pass_name="cr",
                    proposal_id=str(raw["decision_id"]),
                    read_name=str(raw["read"]),
                    library_id=str(raw.get("library_id") or ""),
                    tier=str(raw.get("proposal_tier", "retain_n")),
                    current_state="N",
                    current_interval=tuple(
                        int(x) for x in raw["current_interval"]
                    ),
                    tf_intervals=tf_intervals,
                    tf_posterior=tf_probability,
                    current_posterior=current_probability,
                    configuration_posterior=_bounded_probability(
                        raw.get(
                            "best_decomposition_posterior_given_complex"
                        ),
                        default=0.0,
                    ),
                    molecule_probability=log_odds_to_probability(
                        raw.get("best_tf_segment_log_bf_vs_n",
                                raw.get("best_tf_integrated_raw_log_bf_vs_n"))
                    ),
                    population_probability=population_probability,
                    specificity_probability=log_odds_to_probability(
                        raw.get("boundary_control_log_bf")
                    ),
                )
                decisions[(
                    decision.pass_name,
                    decision.library_id,
                    decision.proposal_id,
                )] = decision

    strand = report.get("strand_rescue", {})
    if isinstance(strand, Mapping) and strand.get("enabled", False):
        for window in strand.get("windows", []):
            for result in window.get("cross_strand", {}).values():
                scenario = _choose_scenario(result.get("scenarios", {}), report)
                if scenario is None:
                    continue
                raw_records = scenario.get("candidate_states")
                if raw_records is None:
                    raise ValueError(
                        "report predates complete SR candidate-state output; rerun "
                        "fiberhmm-consensus-recall before using --paired"
                    )
                for raw in raw_records:
                    tf_intervals = tuple(
                        tuple(int(x) for x in interval)
                        for interval in raw.get("proposed_site_intervals", [])
                    )
                    if not tf_intervals:
                        continue
                    tf_probability = _bounded_probability(raw.get("posterior"))
                    current_probability = _bounded_probability(
                        raw.get("current_posterior", 1.0 - tf_probability)
                    )
                    pair_total = tf_probability + current_probability
                    if pair_total > 0.0:
                        tf_probability /= pair_total
                        current_probability = 1.0 - tf_probability
                    decision = PairedDecision(
                        pass_name="sr",
                        proposal_id=str(raw["decision_id"]),
                        read_name=str(raw["read"]),
                        library_id=str(raw.get("library_id") or ""),
                        tier=str(raw.get("proposal_tier", "retain_current")),
                        current_state=str(raw.get("current", "A")),
                        current_interval=tuple(
                            int(x) for x in raw["current_interval"]
                        ),
                        tf_intervals=tf_intervals,
                        tf_posterior=tf_probability,
                        current_posterior=current_probability,
                        configuration_posterior=_bounded_probability(
                            raw.get("best_configuration_posterior_given_tf"),
                            default=0.0,
                        ),
                        molecule_probability=log_odds_to_probability(
                            raw.get("log_bf_vs_current")
                        ),
                        population_probability=_bounded_probability(
                            raw.get("tf_prior_probability_vs_current")
                        ),
                        specificity_probability=_bounded_probability(
                            raw.get("source_support_fraction")
                        ),
                    )
                    decisions[(
                        decision.pass_name,
                        decision.library_id,
                        decision.proposal_id,
                    )] = decision

    ordered = sorted(
        decisions.values(),
        key=lambda item: (
            item.library_id,
            item.read_name,
            item.pass_name,
            item.current_interval,
            item.proposal_id,
        ),
    )
    current_nuc_decisions = {}
    for decision in ordered:
        if decision.current_state != "N":
            continue
        key = (
            decision.pass_name,
            decision.library_id,
            decision.read_name,
            decision.current_interval,
        )
        previous = current_nuc_decisions.get(key)
        if previous is not None:
            raise ValueError(
                "one current nucleosome has multiple paired decisions "
                f"({previous.proposal_id!r}, {decision.proposal_id!r}); "
                "increase strand-window grouping or revise the focal site set"
            )
        current_nuc_decisions[key] = decision
    return ordered


def _read_length(read) -> int:
    length = read.query_length
    if not length and read.query_sequence:
        length = len(read.query_sequence)
    if not length and read.has_tag("MA"):
        try:
            length = int(str(read.get_tag("MA")).split(";", 1)[0])
        except (TypeError, ValueError):
            length = 0
    if not length:
        infer = getattr(read, "infer_read_length", None)
        if callable(infer):
            length = infer()
    if not length:
        raise ValueError(f"read {read.query_name!r} has no query length")
    return int(length)


def reference_interval_to_molecular(
    read, start: int, end: int
) -> Optional[Tuple[int, int]]:
    """Project a reference half-open interval into molecular MA coordinates."""
    if end <= start:
        raise ValueError(f"invalid reference interval: {start}-{end}")
    reference_positions = read.get_reference_positions(full_length=True)
    query_positions = [
        index
        for index, ref_pos in enumerate(reference_positions)
        if ref_pos is not None and start <= int(ref_pos) < end
    ]
    if not query_positions:
        return None
    seq_start = min(query_positions)
    length = max(query_positions) + 1 - seq_start
    if read.is_reverse:
        return flip_interval_frame(seq_start, length, _read_length(read))
    return int(seq_start), int(length)


def _format_ma(
    read_length: int,
    groups: Sequence[
        Tuple[str, str, str, Sequence[Tuple[int, int]], Sequence[Sequence[int]]]
    ],
) -> Tuple[str, array.array]:
    parts = [str(int(read_length))]
    qualities = array.array("B")
    for name, strand, quality_spec, intervals, quality_rows in groups:
        if not intervals:
            continue
        if len(intervals) != len(quality_rows):
            raise ValueError(f"{name} interval/quality count mismatch")
        tokens = []
        for (start, length), row in zip(intervals, quality_rows):
            if start < 0 or length <= 0 or start + length > read_length:
                raise ValueError(
                    f"{name} interval outside read: start={start}, length={length}, "
                    f"read_length={read_length}"
                )
            if len(row) != len(quality_spec):
                raise ValueError(f"{name} quality arity mismatch")
            tokens.append(f"{int(start) + 1}-{int(length)}")
            qualities.extend(max(0, min(255, int(value))) for value in row)
        parts.append(f"{name}{strand}{quality_spec}:" + ",".join(tokens))
    return ";".join(parts), qualities


def add_overlay_groups(
    read,
    proposals: Sequence[OverlayProposal],
    *,
    active_passes: Optional[Sequence[str]] = None,
    diagnostics: Optional[DefaultDict[str, int]] = None,
) -> Dict[str, int]:
    """Replace stale groups and append complete recalled shadow callsets."""
    passes = set(active_passes or (item.pass_name for item in proposals))
    unknown_passes = passes.difference(PASS_LAYERS)
    if unknown_passes:
        raise ValueError(
            "unknown consensus pass(es): " + ", ".join(sorted(unknown_passes))
        )
    if not proposals and not passes:
        if not read.has_tag("MA"):
            return {}
        ma_value = str(read.get_tag("MA"))
        observed_names = {
            chunk.split("+", 1)[0].split("-", 1)[0].split(".", 1)[0]
            for chunk in ma_value.split(";")[1:]
        }
        if not observed_names.intersection(LAYER_ORDER):
            return {}
    read_length = _read_length(read)
    preserved_groups = []
    preserved_names: List[str] = []
    original_calls: Dict[str, List[Tuple[Tuple[int, int], Sequence[int]]]] = {
        "nuc": [], "tf": [],
    }
    had_an = read.has_tag("AN")

    if read.has_tag("MA"):
        parsed = parse_ma_tag(read.get_tag("MA"))
        if int(parsed["read_length"]) != read_length:
            raise ValueError(
                f"read {read.query_name!r}: MA length {parsed['read_length']} "
                f"does not match query length {read_length}"
            )
        raw_types = parsed["raw_types"]
        aq = read.get_tag("AQ") if read.has_tag("AQ") else []
        expected_aq = sum(len(spec) * len(intervals) for _, _, spec, intervals in raw_types)
        if len(aq) != expected_aq:
            raise ValueError(
                f"read {read.query_name!r}: AQ has {len(aq)} bytes; "
                f"MA requires {expected_aq}"
            )
        per_annotation = parse_aq_array(
            aq,
            [item[2] for item in raw_types],
            [len(item[3]) for item in raw_types],
        )
        old_names = (
            parse_an_tag(read.get_tag("AN")) if had_an else []
        )
        cursor = 0
        for name, strand, quality_spec, intervals in raw_types:
            count = len(intervals)
            rows = per_annotation[cursor:cursor + count]
            names = old_names[cursor:cursor + count]
            cursor += count
            if name in original_calls:
                original_calls[name].extend(zip(intervals, rows))
            if name in LAYER_ORDER:
                continue
            preserved_groups.append(
                (name, strand, quality_spec, list(intervals), rows)
            )
            preserved_names.extend(names + [""] * (count - len(names)))

    layer_values: Dict[str, Dict[Tuple[int, int], Tuple[int, str]]] = {
        name: {} for name in LAYER_ORDER
    }

    # A recalled layer is a full shadow callset, not a list of competing
    # hypotheses. Calls not challenged by a pass are retained with Q=255 by
    # construction. Their native nq/tq remains on the original layer.
    for pass_name in sorted(passes):
        nuc_name, tf_name = PASS_LAYERS[pass_name]
        for index, (interval, _row) in enumerate(original_calls["nuc"]):
            layer_values[nuc_name][interval] = (
                255, f"baseline:{pass_name}:nuc:{index}:{interval}"
            )
        for index, (interval, _row) in enumerate(original_calls["tf"]):
            layer_values[tf_name][interval] = (
                255, f"baseline:{pass_name}:tf:{index}:{interval}"
            )

    def matching_nuc(
        values: Mapping[Tuple[int, int], Tuple[int, str]],
        projected: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        if projected in values:
            return projected
        projected_start, projected_length = projected
        projected_end = projected_start + projected_length
        candidates = []
        for interval in values:
            start, length = interval
            end = start + length
            overlap = max(0, min(end, projected_end) - max(start, projected_start))
            union = max(end, projected_end) - min(start, projected_start)
            if overlap:
                candidates.append((overlap / union, overlap, interval))
        if not candidates:
            return None
        jaccard, _overlap, interval = max(candidates)
        return interval if jaccard >= 0.8 else None

    for proposal in proposals:
        if diagnostics is not None:
            diagnostics["decisions_seen"] += 1
        nuc_name, tf_name = PASS_LAYERS[proposal.pass_name]
        current_interval = reference_interval_to_molecular(
            read, *proposal.current_interval
        )
        matched_nuc = (
            matching_nuc(layer_values[nuc_name], current_interval)
            if current_interval is not None and proposal.current_state == "N"
            else None
        )
        if diagnostics is not None and proposal.current_state == "N":
            diagnostics[
                "nuc_current_matched" if matched_nuc is not None
                else "nuc_current_unmatched"
            ] += 1
        if proposal.replacement_selected:
            if matched_nuc is not None:
                del layer_values[nuc_name][matched_nuc]
                if diagnostics is not None:
                    diagnostics["nuc_removed"] += 1
        elif matched_nuc is not None and proposal.nuc_posterior is not None:
            layer_values[nuc_name][matched_nuc] = (
                posterior_to_q(proposal.nuc_posterior), proposal.proposal_id
            )
            continue

        if not proposal.replacement_selected:
            continue
        tf_q = posterior_to_q(proposal.tf_posterior)
        added = 0
        for ref_interval in proposal.tf_intervals:
            interval = reference_interval_to_molecular(read, *ref_interval)
            if interval is None:
                continue
            value = (tf_q, proposal.proposal_id)
            if interval not in layer_values[tf_name] or tf_q > layer_values[tf_name][interval][0]:
                layer_values[tf_name][interval] = value
                added += 1
        if diagnostics is not None:
            diagnostics["replacement_intervals_added"] += added

    # Enforce the shadow-callset invariant globally. Some legacy inputs retain
    # an HMM nuc underneath an ordinary post-TF call; copying both would make
    # the supposedly final shadow state internally contradictory. TF calls
    # therefore overwrite every overlapping shadow nuc, whether the TF was
    # carried from baseline or added by this pass.
    for pass_name in sorted(passes):
        nuc_name, tf_name = PASS_LAYERS[pass_name]
        tf_intervals = list(layer_values[tf_name])
        overwritten = []
        for nuc_interval in layer_values[nuc_name]:
            nuc_start, nuc_length = nuc_interval
            nuc_end = nuc_start + nuc_length
            if any(
                nuc_start < tf_start + tf_length
                and tf_start < nuc_end
                for tf_start, tf_length in tf_intervals
            ):
                overwritten.append(nuc_interval)
        for interval in overwritten:
            del layer_values[nuc_name][interval]
        if diagnostics is not None:
            diagnostics["nuc_removed_by_tf_overlap"] += len(overwritten)

    new_names: List[str] = []
    counts: Dict[str, int] = {}
    for layer_name in LAYER_ORDER:
        records = sorted(layer_values[layer_name].items())
        if not records:
            continue
        intervals = [interval for interval, _ in records]
        rows = [[value[0]] for _, value in records]
        preserved_groups.append((layer_name, ".", "Q", intervals, rows))
        counts[layer_name] = len(records)
        for _, (_, proposal_id) in records:
            token = hashlib.sha256(proposal_id.encode("utf-8")).hexdigest()[:16]
            new_names.append(f"fh_{layer_name}_{token}")

    if not preserved_groups:
        for tag in ("MA", "AQ", "AN"):
            if read.has_tag(tag):
                read.set_tag(tag, None)
        return counts

    ma_value, aq_value = _format_ma(read_length, preserved_groups)
    read.set_tag("MA", ma_value, value_type="Z")
    if aq_value:
        read.set_tag("AQ", aq_value)
    elif read.has_tag("AQ"):
        read.set_tag("AQ", None)
    if had_an:
        names = preserved_names + new_names
        read.set_tag(
            "AN", ",".join(name if name else "." for name in names), value_type="Z"
        )
    return counts


def paired_quality_rows(
    decision: PairedDecision,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return exactly complementary current/TF q0 rows for one decision."""
    tf_q = posterior_to_q(decision.tf_posterior)
    current_q = 255 - tf_q
    auxiliary = (
        posterior_to_q(decision.configuration_posterior),
        posterior_to_q(decision.molecule_probability),
        posterior_to_q(decision.population_probability),
        posterior_to_q(decision.specificity_probability),
    )
    return (current_q, *auxiliary), (tf_q, *auxiliary)


def add_paired_overlay_groups(
    read,
    decisions: Sequence[PairedDecision],
    *,
    active_passes: Optional[Sequence[str]] = None,
    diagnostics: Optional[DefaultDict[str, int]] = None,
    applied_proposal_ids: Optional[set] = None,
) -> Dict[str, int]:
    """Append aggressive, linked current/TF hypotheses as MA-valid groups."""
    passes = set(active_passes or (item.pass_name for item in decisions))
    unknown_passes = passes.difference(PASS_LAYERS)
    if unknown_passes:
        raise ValueError(
            "unknown consensus pass(es): " + ", ".join(sorted(unknown_passes))
        )
    if not read.has_tag("MA"):
        return {}

    read_length = _read_length(read)
    parsed = parse_ma_tag(read.get_tag("MA"))
    if int(parsed["read_length"]) != read_length:
        raise ValueError(
            f"read {read.query_name!r}: MA length {parsed['read_length']} "
            f"does not match query length {read_length}"
        )
    raw_types = parsed["raw_types"]
    aq = read.get_tag("AQ") if read.has_tag("AQ") else []
    expected_aq = sum(
        len(spec) * len(intervals)
        for _, _, spec, intervals in raw_types
    )
    if len(aq) != expected_aq:
        raise ValueError(
            f"read {read.query_name!r}: AQ has {len(aq)} bytes; "
            f"MA requires {expected_aq}"
        )
    per_annotation = parse_aq_array(
        aq,
        [item[2] for item in raw_types],
        [len(item[3]) for item in raw_types],
    )
    had_an = read.has_tag("AN")
    old_names = parse_an_tag(read.get_tag("AN")) if had_an else []
    annotation_total = sum(len(item[3]) for item in raw_types)
    if had_an and len(old_names) != annotation_total:
        raise ValueError(
            f"read {read.query_name!r}: AN has {len(old_names)} names; "
            f"MA contains {annotation_total} annotations"
        )

    preserved_groups = []
    preserved_names: List[str] = []
    original_calls: Dict[
        str, List[Tuple[Tuple[int, int], Sequence[int]]]
    ] = {"nuc": [], "tf": [], "msp": []}
    cursor = 0
    for name, strand, quality_spec, intervals in raw_types:
        count = len(intervals)
        rows = per_annotation[cursor:cursor + count]
        names = old_names[cursor:cursor + count]
        cursor += count
        if name in original_calls:
            original_calls[name].extend(zip(intervals, rows))
        if name in LAYER_ORDER:
            continue
        preserved_groups.append(
            (name, strand, quality_spec, list(intervals), rows)
        )
        preserved_names.extend(names + [""] * (count - len(names)))

    # value = (quality row, AN label, provenance kind)
    layer_values: Dict[
        str,
        Dict[Tuple[int, int], Tuple[Tuple[int, ...], str, str]],
    ] = {name: {} for name in LAYER_ORDER}
    baseline_row = (255, 0, 0, 0, 0)
    for pass_name in sorted(passes):
        nuc_name, tf_name = PASS_LAYERS[pass_name]
        for interval, _row in original_calls["nuc"]:
            layer_values[nuc_name][interval] = (
                baseline_row, "", "baseline"
            )
        for interval, _row in original_calls["tf"]:
            layer_values[tf_name][interval] = (
                baseline_row, "", "baseline"
            )

        # A direct baseline TF remains authoritative over a contradictory
        # baseline nuc. Paired TF alternatives are added only after this fixed
        # callset cleanup and therefore do not erase their paired N state.
        fixed_tfs = list(layer_values[tf_name])
        overwritten = []
        for nuc_interval in layer_values[nuc_name]:
            nuc_start, nuc_length = nuc_interval
            nuc_end = nuc_start + nuc_length
            if any(
                nuc_start < tf_start + tf_length and tf_start < nuc_end
                for tf_start, tf_length in fixed_tfs
            ):
                overwritten.append(nuc_interval)
        for interval in overwritten:
            del layer_values[nuc_name][interval]
        if diagnostics is not None:
            diagnostics["paired_baseline_nuc_removed_by_tf_overlap"] += len(
                overwritten
            )

    def matching_interval(
        intervals: Sequence[Tuple[int, int]],
        projected: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        if projected in intervals:
            return projected
        projected_start, projected_length = projected
        projected_end = projected_start + projected_length
        candidates = []
        for interval in intervals:
            start, length = interval
            end = start + length
            overlap = max(
                0, min(end, projected_end) - max(start, projected_start)
            )
            union = max(end, projected_end) - min(start, projected_start)
            if overlap:
                candidates.append((overlap / union, overlap, interval))
        if not candidates:
            return None
        jaccard, _overlap, interval = max(candidates)
        return interval if jaccard >= 0.8 else None

    for decision in decisions:
        if diagnostics is not None:
            diagnostics["paired_decisions_seen"] += 1
        current_interval = reference_interval_to_molecular(
            read, *decision.current_interval
        )
        if current_interval is None:
            if diagnostics is not None:
                diagnostics["paired_current_unprojectable"] += 1
            continue
        nuc_name, tf_name = PASS_LAYERS[decision.pass_name]
        matched_current = None
        if decision.current_state == "N":
            matched_current = matching_interval(
                list(layer_values[nuc_name]), current_interval
            )
        elif decision.current_state == "A":
            matched_current = matching_interval(
                [interval for interval, _row in original_calls["msp"]],
                current_interval,
            )
        else:
            raise ValueError(
                f"unsupported paired current state: {decision.current_state!r}"
            )
        if matched_current is None:
            if diagnostics is not None:
                diagnostics[
                    "paired_nuc_current_unmatched"
                    if decision.current_state == "N"
                    else "paired_access_current_unmatched"
                ] += 1
            continue

        projected_tfs = []
        for ref_interval in decision.tf_intervals:
            interval = reference_interval_to_molecular(read, *ref_interval)
            if interval is not None and interval not in projected_tfs:
                projected_tfs.append(interval)
        if not projected_tfs:
            if diagnostics is not None:
                diagnostics["paired_tf_unprojectable"] += 1
            continue

        current_row, tf_row = paired_quality_rows(decision)
        token = hashlib.sha256(
            (
                f"{decision.pass_name}:{decision.library_id}:"
                f"{decision.proposal_id}"
            ).encode("utf-8")
        ).hexdigest()[:16]
        prefix = f"fh{decision.pass_name}_{token}"
        previous_nuc_value = None
        if decision.current_state == "N":
            previous_nuc_value = layer_values[nuc_name][matched_current]
            layer_values[nuc_name][matched_current] = (
                current_row, f"{prefix}_N", "paired"
            )
            if diagnostics is not None:
                diagnostics["paired_nuc_linked"] += 1

        tf_intervals_written = 0
        for index, interval in enumerate(projected_tfs):
            tf_role = (
                f"AT{index}" if decision.current_state == "A" else f"T{index}"
            )
            existing = layer_values[tf_name].get(interval)
            if (
                existing is None
                or existing[2] == "baseline"
                or tf_row[0] > existing[0][0]
            ):
                layer_values[tf_name][interval] = (
                    tf_row, f"{prefix}_{tf_role}", "paired"
                )
                tf_intervals_written += 1
                if diagnostics is not None:
                    diagnostics["paired_tf_intervals_added"] += 1
        if tf_intervals_written:
            if applied_proposal_ids is not None:
                applied_proposal_ids.add(decision.proposal_id)
        elif decision.current_state == "N":
            # Do not leave an orphan N member if a conflicting TF decision
            # already owns every proposed interval.
            layer_values[nuc_name][matched_current] = previous_nuc_value
            if diagnostics is not None:
                diagnostics["paired_decision_conflict_not_applied"] += 1

    new_names: List[str] = []
    counts: Dict[str, int] = {}
    for layer_name in LAYER_ORDER:
        records = sorted(layer_values[layer_name].items())
        if not records:
            continue
        intervals = [interval for interval, _ in records]
        rows = [list(value[0]) for _, value in records]
        preserved_groups.append(
            (layer_name, ".", PAIRED_QUALITY_SPEC, intervals, rows)
        )
        new_names.extend(value[1] for _, value in records)
        counts[layer_name] = len(records)

    if not preserved_groups:
        for tag in ("MA", "AQ", "AN"):
            if read.has_tag(tag):
                read.set_tag(tag, None)
        return counts

    ma_value, aq_value = _format_ma(read_length, preserved_groups)
    read.set_tag("MA", ma_value, value_type="Z")
    if aq_value:
        read.set_tag("AQ", aq_value)
    elif read.has_tag("AQ"):
        read.set_tag("AQ", None)

    names = preserved_names + new_names
    if had_an or any(names):
        # The MA spec uses empty positional fields for unnamed annotations.
        read.set_tag("AN", ",".join(names), value_type="Z")
    elif read.has_tag("AN"):
        read.set_tag("AN", None)
    return counts


def _report_inputs(report: Mapping[str, object]) -> List[str]:
    input_block = report.get("input", {})
    paths = [
        *input_block.get("bams", []),
        *input_block.get("target_bams", []),
    ]
    return list(dict.fromkeys(_canonical_path(str(path)) for path in paths))


def _input_metadata(report: Mapping[str, object]) -> Dict[str, Mapping[str, object]]:
    values = {}
    for record in report.get("input", {}).get("files", []):
        values[_canonical_path(str(record["path"]))] = record
    return values


def _validate_input_file(
    path: str,
    metadata: Mapping[str, Mapping[str, object]],
    *,
    allow_drift: bool,
) -> None:
    candidate = Path(path)
    if not candidate.is_file():
        raise ValueError(f"missing input BAM: {candidate}")
    expected = metadata.get(_canonical_path(path))
    if expected is None:
        return
    stat = candidate.stat()
    changed = (
        int(expected.get("size_bytes", stat.st_size)) != int(stat.st_size)
        or int(expected.get("mtime_ns", stat.st_mtime_ns)) != int(stat.st_mtime_ns)
    )
    if changed and not allow_drift:
        raise ValueError(
            f"input BAM changed since report generation: {candidate}; "
            "pass --allow-input-drift only after verifying provenance"
        )


def _header_with_provenance(
    header,
    *,
    report_sha256: str,
    include_review: bool,
    include_retain_n: bool,
    paired: bool,
    command_line: str,
):
    tier_mode = (
        "paired-aggressive"
        if paired
        else (
            "all-tested" if include_retain_n
            else ("strong+review" if include_review else "strong")
        )
    )
    output = append_ma_types(header, LAYER_ORDER)
    header_dict = output.to_dict()
    comments = list(header_dict.get("CO", []))
    if paired:
        comments.append(
            PAIRED_CONSENSUS_HEADER_PREFIX
            + "groups=" + ",".join(LAYER_ORDER)
            + ";semantics=paired_hypotheses"
            + f";quality_spec={PAIRED_QUALITY_SPEC}"
            + ";q_scale=linear_probability"
            + ";q0=represented_state_posterior"
            + ";pair_sum=255;pairing=AN_shared_prefix"
            + ";tf_if=q0_tf>=T;nuc_if=q0_nuc>=256-T"
            + ";accessible_current=ordinary_msp"
            + ";accessible_tf_role=ATn"
            + ";aux=q1_configuration,q2_molecule,q3_population,q4_specificity"
            + ";baseline_q0=255;baseline_aux=0"
            + f";tiers={tier_mode}"
            + f";report_sha256={report_sha256}"
        )
    else:
        comments.append(
            CONSENSUS_HEADER_PREFIX
            + "groups=" + ",".join(LAYER_ORDER)
            + ";semantics=complete_shadow_callsets"
            + ";q=round(255*posterior);q_scale=linear_probability"
            + ";unchallenged_q=255"
            + f";tiers={tier_mode}"
            + f";report_sha256={report_sha256}"
        )
    header_dict["CO"] = comments
    output = pysam.AlignmentHeader.from_dict(header_dict)
    return append_pg_record(
        output,
        {
            "PN": "fiberhmm-consensus-annotate",
            "VN": FIBERHMM_VERSION,
            "CL": command_line,
            "DS": (
                (
                    "linked aggressive current/TF MA hypotheses; QQQQQ; "
                    "q0 pair sums to 255; AN prefix links states; "
                    if paired else
                    "complete recalled MA shadow callsets; "
                    "tested Q=round(255*posterior); unchallenged Q=255; "
                )
                + "original nuc/tf preserved; "
                + f"tiers={tier_mode}; "
                f"report_sha256={report_sha256}; consensus_schema={VALIDATION_VERSION}"
            ),
        },
    )


def _proposal_index(
    proposals: Sequence[ConsensusDecision],
    report_inputs: Sequence[str],
) -> Dict[str, DefaultDict[str, List[ConsensusDecision]]]:
    result: Dict[str, DefaultDict[str, List[ConsensusDecision]]] = {
        path: defaultdict(list) for path in report_inputs
    }
    basenames: DefaultDict[str, List[str]] = defaultdict(list)
    for path in report_inputs:
        basenames[Path(path).name].append(path)
    for proposal in proposals:
        if proposal.library_id:
            resolved = _canonical_path(proposal.library_id)
            if resolved not in result:
                matches = basenames.get(Path(proposal.library_id).name, [])
                if len(matches) != 1:
                    raise ValueError(
                        f"proposal library is not uniquely present in report inputs: "
                        f"{proposal.library_id}"
                    )
                resolved = matches[0]
        elif len(report_inputs) == 1:
            resolved = report_inputs[0]
        else:
            raise ValueError(
                f"proposal {proposal.proposal_id} has no library_id in a multi-BAM report"
            )
        result[resolved][proposal.read_name].append(proposal)
    return result


def _default_output_path(output_dir: Path, input_path: str, used: set[str]) -> Path:
    source = Path(input_path)
    stem = source.name[:-4] if source.name.endswith(".bam") else source.name
    name = f"{stem}.consensus-overlay.bam"
    if name in used:
        token = hashlib.sha256(input_path.encode("utf-8")).hexdigest()[:8]
        name = f"{stem}.{token}.consensus-overlay.bam"
    used.add(name)
    return output_dir / name


def write_overlay_bam(
    input_path: str,
    output_path: Path,
    *,
    region: Tuple[str, int, int],
    proposals_by_read: Mapping[str, Sequence[ConsensusDecision]],
    report_sha256: str,
    include_review: bool,
    include_retain_n: bool = False,
    paired: bool = False,
    active_passes: Optional[Sequence[str]] = None,
    command_line: str,
    io_threads: int = 1,
    protected_paths: Sequence[str] = (),
) -> dict:
    """Write one coordinate-sorted, indexed regional visualization BAM."""
    if output_path.suffix != ".bam":
        raise ValueError(f"output must end in .bam: {output_path}")
    # Every BAM this run may read is protected, not just the one being
    # processed.  ``--output`` naming a *different* cohort BAM would otherwise
    # pass this guard and be replaced atomically, destroying a source input.
    # Indices are protected alongside their BAM.  This runs before any
    # filesystem mutation, so a rejected path never creates directories.
    forbidden = {_canonical_path(input_path)}
    forbidden.update(_canonical_path(path) for path in protected_paths)
    forbidden.update(f"{path}.bai" for path in tuple(forbidden))
    for candidate in (str(output_path), f"{output_path}.bai"):
        if _canonical_path(candidate) in forbidden:
            raise ValueError(
                "refusing to overwrite a BAM this run reads from: "
                f"{candidate}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".bam", dir=str(output_path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary_index = Path(str(temporary) + ".bai")
    output_index = Path(str(output_path) + ".bai")
    chrom, start, end = region
    counts = {name: 0 for name in LAYER_ORDER}
    seen_proposals = set()
    reads_written = 0
    reads_with_overlays = 0
    application_diagnostics: DefaultDict[str, int] = defaultdict(int)
    try:
        with pysam.AlignmentFile(
            input_path, "rb", check_sq=False, threads=io_threads
        ) as source:
            header = _header_with_provenance(
                source.header,
                report_sha256=report_sha256,
                include_review=include_review,
                include_retain_n=include_retain_n,
                paired=paired,
                command_line=command_line,
            )
            with pysam.AlignmentFile(
                str(temporary), "wb", header=header, threads=io_threads
            ) as destination:
                for read in source.fetch(chrom, start, end):
                    read_proposals = (
                        ()
                        if read.is_unmapped or read.is_secondary or read.is_supplementary
                        else proposals_by_read.get(read.query_name, ())
                    )
                    if paired:
                        layer_counts = add_paired_overlay_groups(
                            read,
                            read_proposals,
                            active_passes=active_passes,
                            diagnostics=application_diagnostics,
                            applied_proposal_ids=seen_proposals,
                        )
                    else:
                        layer_counts = add_overlay_groups(
                            read,
                            read_proposals,
                            active_passes=active_passes,
                            diagnostics=application_diagnostics,
                        )
                    if layer_counts:
                        reads_with_overlays += 1
                        for name, value in layer_counts.items():
                            counts[name] += value
                        if not paired:
                            seen_proposals.update(
                                item.proposal_id for item in read_proposals
                            )
                    destination.write(read)
                    reads_written += 1
        pysam.index(str(temporary))
        os.replace(temporary, output_path)
        os.replace(temporary_index, output_index)
    except BaseException:
        temporary.unlink(missing_ok=True)
        temporary_index.unlink(missing_ok=True)
        raise

    expected = {
        item.proposal_id
        for values in proposals_by_read.values()
        for item in values
    }
    return {
        "input": _canonical_path(input_path),
        "output": str(output_path.resolve()),
        "index": str(output_index.resolve()),
        "region": [chrom, start, end],
        "reads_written": reads_written,
        "reads_with_overlays": reads_with_overlays,
        "annotations": counts,
        "proposals_expected": len(expected),
        "proposals_matched": len(seen_proposals),
        "proposals_unmatched": len(expected - seen_proposals),
        "application": dict(sorted(application_diagnostics.items())),
        "paired_hypotheses": bool(paired),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write non-destructive nuc_cr/tf_cr and nuc_sr/tf_sr MA layers "
            "from a consensus-recaller JSON report. Original nuc/tf "
            "annotations remain unchanged. --paired emits aggressive linked "
            "hypotheses for a FiberBrowser split threshold; legacy modes emit "
            "selected regional shadow callsets."
        )
    )
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "-i", "--bam", action="append",
        help="Report input BAM to materialize; repeatable (default: all report BAMs).",
    )
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument(
        "-o", "--output",
        help="Output BAM; valid only when exactly one input BAM is selected.",
    )
    outputs.add_argument(
        "--output-dir",
        help="Directory receiving one regional overlay BAM per selected input.",
    )
    parser.add_argument(
        "--region", type=parse_region,
        help="Override the report loaded region (CHROM:START-END).",
    )
    tiers = parser.add_mutually_exclusive_group()
    tiers.add_argument(
        "--strong-only", action="store_true",
        help="Exclude review-tier overlays (default writes strong and review).",
    )
    tiers.add_argument(
        "--all-tested", action="store_true",
        help=(
            "Apply the MAP state for every tested CR candidate, including "
            "retain-N decisions. This never writes simultaneous N/TF alternatives. "
            "Requires a current report."
        ),
    )
    tiers.add_argument(
        "--paired", action="store_true",
        help=(
            "Emit every locally supported CR/SR candidate as linked paired "
            "hypotheses with QQQQQ qualities. q0 values are complementary, "
            "and AN prefixes define atomic decisions for one browser slider."
        ),
    )
    parser.add_argument("--allow-truncated-report", action="store_true")
    parser.add_argument("--allow-input-drift", action="store_true")
    parser.add_argument("--io-threads", type=int, default=1)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.io_threads < 1:
        parser.error("--io-threads must be positive")
    report_path = Path(args.report).expanduser().resolve()
    if not report_path.is_file():
        parser.error(f"missing report: {report_path}")
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"cannot read report: {error}")
    if report.get("producer", {}).get("name") != "fiberhmm-consensus-recaller-report":
        parser.error("input is not a fiberhmm consensus-recaller report")

    report_inputs = _report_inputs(report)
    selected = (
        [_canonical_path(path) for path in args.bam]
        if args.bam else report_inputs
    )
    unknown = sorted(set(selected) - set(report_inputs))
    if unknown:
        parser.error("BAM is not an input recorded by the report: " + ", ".join(unknown))
    if not selected:
        parser.error("report contains no BAM inputs")
    if args.output and len(selected) != 1:
        parser.error("--output requires exactly one selected BAM; use --output-dir")

    loaded_region = report.get("input", {}).get("loaded_region")
    if args.region is not None:
        region = args.region
    elif (
        isinstance(loaded_region, list)
        and len(loaded_region) == 3
    ):
        region = (
            str(loaded_region[0]), int(loaded_region[1]), int(loaded_region[2])
        )
    else:
        parser.error("report has no valid loaded_region; pass --region")

    try:
        proposals = (
            collect_paired_decisions(report)
            if args.paired
            else collect_overlay_proposals(
                report,
                include_review=not args.strong_only,
                include_retain_n=args.all_tested,
                allow_truncated=args.allow_truncated_report,
            )
        )
        metadata = _input_metadata(report)
        for path in selected:
            _validate_input_file(
                path, metadata, allow_drift=args.allow_input_drift
            )
        index = _proposal_index(proposals, report_inputs)
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))

    report_sha256 = _sha256_file(report_path)
    active_passes = []
    composite = report.get("composite_deconvolution", {})
    if isinstance(composite, Mapping) and composite.get("enabled", False):
        active_passes.append("cr")
    strand = report.get("strand_rescue", {})
    if isinstance(strand, Mapping) and strand.get("applicable", False):
        active_passes.append("sr")
    command_line = " ".join(
        shlex.quote(value)
        for value in (["fiberhmm-consensus-annotate"] + list(argv or sys.argv[1:]))
    )
    used_names: set[str] = set()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    summaries = []
    try:
        for path in selected:
            output = (
                Path(args.output).expanduser()
                if args.output
                else _default_output_path(output_dir, path, used_names)
            )
            summaries.append(write_overlay_bam(
                path,
                output,
                region=region,
                proposals_by_read=index[path],
                report_sha256=report_sha256,
                include_review=not args.strong_only,
                include_retain_n=args.all_tested,
                paired=args.paired,
                active_passes=active_passes,
                command_line=command_line,
                io_threads=args.io_threads,
                protected_paths=tuple(report_inputs) + tuple(selected),
            ))
    except (OSError, ValueError, pysam.utils.SamtoolsError) as error:
        parser.error(str(error))

    result = {
        "report": str(report_path),
        "report_sha256": report_sha256,
        "callset_semantics": (
            "linked aggressive current/TF hypotheses; FiberBrowser selects "
            "one paired state by AN decision prefix"
            if args.paired else
            "complete shadow callsets; selected replacements remove their prior state"
        ),
        "q_encoding": (
            "QQQQQ linear bytes; q0 is represented-state posterior; paired "
            "q0 values sum to 255; q1 configuration, q2 molecule, q3 "
            "population, q4 specificity"
            if args.paired else
            "tested calls: round(255 * posterior), linear probability; "
            "unchallenged baseline calls: 255"
        ),
        "fiberbrowser_threshold": (
            "TF when q0_tf >= T; N when q0_nuc >= 256-T; group by shared "
            "AN prefix and render exactly one state"
            if args.paired else None
        ),
        "tiers": (
            "paired-aggressive" if args.paired
            else "all-tested" if args.all_tested
            else ("strong" if args.strong_only else "strong+review")
        ),
        "outputs": summaries,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
