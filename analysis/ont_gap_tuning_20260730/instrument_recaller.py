#!/usr/bin/env python3
"""Audit nucleosome recall one decision at a time on raw Fiber-seq reads.

This intentionally stops before TF promotion/unification.  Its purpose is to
measure which *nucleosome-recaller* decision changes HMM-protected sequence into
reported MSP/NFR sequence, and to compare alternative edge interpretations:

``conservative``
    Current behavior: use the first/last evidence-supported protected base.
``loose``
    Expand each protected call through its measured ambiguity, up to (but not
    including) the nearest accessible hit or the fragment boundary.
``fragment``
    Keep the full post-split fragment.  This is a diagnostic upper bound.

The loose interpretation is not a fitted size expansion: it uses ambiguity
already returned by the recaller's evidence scan.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pysam

from fiberhmm.core.model_io import freeze_model_for_inference, load_model
from fiberhmm.inference.engine import (
    CHIMERA_SKIP,
    _extract_fiber_read_from_pysam,
)
from fiberhmm.inference.fused_stages import run_hmm_apply_stage
from fiberhmm.inference.nuc_recaller import (
    NucCall,
    assemble_nuc_msp_tiling,
)
from fiberhmm.inference.tf_recaller import (
    build_llr_tables,
    call_tfs_in_interval,
)
from fiberhmm.io.ma_tags import ambiguity_to_edge, llr_to_tq
from fiberhmm.models import get_model_path


@dataclass
class Geometry:
    reads: int = 0
    query_bp: int = 0
    nuc_bp: int = 0
    nuc_lengths: list[int] = field(default_factory=list)
    nfr_bp: int = 0
    nfr_lengths: list[int] = field(default_factory=list)

    def add(
        self,
        read_length: int,
        nucs: list[NucCall],
        nfrs: list[tuple[int, int]],
    ) -> None:
        self.reads += 1
        self.query_bp += read_length
        self.nuc_bp += sum(n.length for n in nucs)
        self.nuc_lengths.extend(n.length for n in nucs)
        self.nfr_bp += sum(length for _, length in nfrs)
        self.nfr_lengths.extend(length for _, length in nfrs)


def _q(values: list[int], quantile: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values), quantile))


def _span(apply_result: dict, read_length: int) -> tuple[int, int]:
    starts: list[int] = []
    ends: list[int] = []
    for start_key, length_key in (("ns", "nl"), ("as", "al")):
        for start, length in zip(
            apply_result.get(start_key, ()), apply_result.get(length_key, ())
        ):
            starts.append(int(start))
            ends.append(int(start) + int(length))
    return (
        min(starts) if starts else 0,
        max(ends) if ends else read_length,
    )


def _fragments(
    obs: np.ndarray,
    start: int,
    end: int,
    llr_hit: np.ndarray,
    llr_miss: np.ndarray,
    split_min_llr: float,
    split_min_opps: int,
):
    cuts = sorted(
        call_tfs_in_interval(
            obs,
            start,
            end,
            -llr_hit,
            -llr_miss,
            split_min_llr,
            split_min_opps,
        ),
        key=lambda call: call.start,
    )
    fragments: list[tuple[int, int]] = []
    cursor = start
    for cut in cuts:
        cut_start = int(cut.start)
        cut_end = int(cut.start + cut.length)
        if cut_start > cursor:
            fragments.append((cursor, cut_start))
        cursor = max(cursor, cut_end)
    if cursor < end:
        fragments.append((cursor, end))
    return cuts, fragments


def _topology_cuts(cuts, start: int, end: int, nuc_min_size: int):
    """Maximum-evidence cut chain leaving a nucleosome-sized piece on both sides.

    A nucleosome *split* should separate two possible nucleosomes.  Calls near a
    footprint edge, or pairs of calls that leave a sub-floor island between
    them, are edge evidence rather than evidence for multiple nucleosomes.
    """
    eligible = [
        call
        for call in cuts
        if (
            int(call.start) - start >= nuc_min_size
            and end - int(call.start + call.length) >= nuc_min_size
        )
    ]
    if not eligible:
        return []
    best_score: list[float] = []
    predecessor: list[int | None] = []
    for i, call in enumerate(eligible):
        score = float(call.llr)
        pred = None
        for j in range(i):
            if (
                int(call.start)
                - int(eligible[j].start + eligible[j].length)
                >= nuc_min_size
                and best_score[j] + float(call.llr) > score
            ):
                score = best_score[j] + float(call.llr)
                pred = j
        best_score.append(score)
        predecessor.append(pred)
    cursor = int(np.argmax(np.asarray(best_score)))
    selected = []
    while cursor is not None:
        selected.append(eligible[cursor])
        cursor = predecessor[cursor]
    return list(reversed(selected))


def _protected_calls(
    obs: np.ndarray,
    start: int,
    end: int,
    llr_hit: np.ndarray,
    llr_miss: np.ndarray,
    edge_min_llr: float,
    edge_min_opps: int,
):
    return sorted(
        call_tfs_in_interval(
            obs,
            start,
            end,
            llr_hit,
            llr_miss,
            edge_min_llr,
            edge_min_opps,
        ),
        key=lambda call: call.start,
    )


def _nuc_from_bounds(calls, start: int, end: int) -> NucCall:
    total_llr = sum(call.llr for call in calls)
    return NucCall(
        start=start,
        length=end - start,
        nq=llr_to_tq(total_llr),
        el=ambiguity_to_edge(calls[0].left_ambiguity),
        er=ambiguity_to_edge(calls[-1].right_ambiguity),
    )


def audit_read(
    apply_result: dict,
    read_length: int,
    llr_hit: np.ndarray,
    llr_miss: np.ndarray,
    *,
    split_min_llr: float,
    split_min_opps: int,
    edge_min_llr: float,
    edge_min_opps: int,
    nuc_min_size: int,
    msp_min_size: int,
):
    obs = apply_result["encoded"]
    stage = Counter()
    lengths: dict[str, list[int]] = defaultdict(list)
    by_policy: dict[str, list[NucCall]] = {
        "raw_hmm": [],
        "conservative": [],
        "loose": [],
        "fragment": [],
        "topology_fragment": [],
    }

    for raw_start, raw_length in zip(apply_result["ns"], apply_result["nl"]):
        start = int(raw_start)
        end = min(read_length, start + int(raw_length))
        if end <= start:
            continue
        raw_len = end - start
        stage["raw_footprint_count"] += 1
        stage["raw_footprint_bp"] += raw_len
        lengths["raw_footprint"].append(raw_len)
        if raw_len >= nuc_min_size:
            by_policy["raw_hmm"].append(NucCall(start, raw_len, 0, 0, 0))
            stage["raw_nuc_count"] += 1
            stage["raw_nuc_bp"] += raw_len
        else:
            stage["raw_subfloor_count"] += 1
            stage["raw_subfloor_bp"] += raw_len

        cuts, fragments = _fragments(
            obs,
            start,
            end,
            llr_hit,
            llr_miss,
            split_min_llr,
            split_min_opps,
        )
        for cut in cuts:
            stage["cut_count"] += 1
            stage["cut_bp"] += int(cut.length)
            lengths["cut"].append(int(cut.length))

        selected_cuts = _topology_cuts(cuts, start, end, nuc_min_size)
        topology_fragments: list[tuple[int, int]] = []
        cursor = start
        for cut in selected_cuts:
            cut_start = int(cut.start)
            cut_end = int(cut.start + cut.length)
            topology_fragments.append((cursor, cut_start))
            cursor = cut_end
            stage["topology_cut_count"] += 1
            stage["topology_cut_bp"] += int(cut.length)
        topology_fragments.append((cursor, end))
        for frag_start, frag_end in topology_fragments:
            if frag_end - frag_start >= nuc_min_size:
                by_policy["topology_fragment"].append(
                    NucCall(frag_start, frag_end - frag_start, 0, 0, 0)
                )

        for frag_start, frag_end in fragments:
            frag_len = frag_end - frag_start
            stage["fragment_count"] += 1
            stage["fragment_bp"] += frag_len
            lengths["fragment"].append(frag_len)
            if frag_len < nuc_min_size:
                stage["postsplit_subfloor_count"] += 1
                stage["postsplit_subfloor_bp"] += frag_len
                lengths["postsplit_subfloor"].append(frag_len)
                continue

            by_policy["fragment"].append(
                NucCall(frag_start, frag_len, 0, 0, 0)
            )
            calls = _protected_calls(
                obs,
                frag_start,
                frag_end,
                llr_hit,
                llr_miss,
                edge_min_llr,
                edge_min_opps,
            )
            if not calls:
                stage["signal_desert_count"] += 1
                stage["signal_desert_bp"] += frag_len
                raw_nuc = NucCall(frag_start, frag_len, 0, 0, 0)
                by_policy["conservative"].append(raw_nuc)
                by_policy["loose"].append(raw_nuc)
                continue

            first = calls[0]
            last = calls[-1]
            core_start = int(first.start)
            core_end = int(last.start + last.length)
            core_len = core_end - core_start
            loose_start = max(
                frag_start, core_start - int(first.left_ambiguity)
            )
            loose_end = min(
                frag_end, core_end + int(last.right_ambiguity)
            )
            loose_len = loose_end - loose_start

            lengths["protected_core"].append(core_len)
            lengths["protected_loose"].append(loose_len)
            stage["edge_candidate_count"] += 1
            stage["edge_candidate_bp"] += frag_len
            stage["conservative_trim_bp"] += frag_len - core_len
            stage["loose_trim_bp"] += frag_len - loose_len

            if core_len >= nuc_min_size:
                by_policy["conservative"].append(
                    _nuc_from_bounds(calls, core_start, core_end)
                )
                stage["conservative_kept_count"] += 1
                stage["conservative_kept_bp"] += core_len
            else:
                stage["core_below_floor_count"] += 1
                stage["core_below_floor_fragment_bp"] += frag_len
                stage["core_below_floor_core_bp"] += core_len
                lengths["core_below_floor_fragment"].append(frag_len)

            if loose_len >= nuc_min_size:
                by_policy["loose"].append(
                    _nuc_from_bounds(calls, loose_start, loose_end)
                )
                stage["loose_kept_count"] += 1
                stage["loose_kept_bp"] += loose_len
                if core_len < nuc_min_size:
                    stage["loose_rescued_count"] += 1
                    stage["loose_rescued_fragment_bp"] += frag_len
                    stage["loose_rescued_nuc_bp"] += loose_len
            else:
                stage["loose_below_floor_count"] += 1
                stage["loose_below_floor_fragment_bp"] += frag_len

    span_lo, span_hi = _span(apply_result, read_length)
    tiled = {}
    for policy, nucs in by_policy.items():
        kept, nfrs = assemble_nuc_msp_tiling(
            nucs,
            span_lo,
            span_hi,
            msp_min_size,
            nuc_min_size,
        )
        tiled[policy] = (kept, nfrs)
    return stage, lengths, tiled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bam", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seq", choices=("nanopore", "pacbio"), required=True)
    parser.add_argument("--prob-threshold", type=int, default=248)
    parser.add_argument("--max-reads", type=int, default=2000)
    parser.add_argument("--min-read-length", type=int, default=1000)
    parser.add_argument("--split-min-llr", type=float, default=4.0)
    parser.add_argument("--split-min-opps", type=int, default=3)
    parser.add_argument("--edge-min-llr", type=float, default=2.0)
    parser.add_argument("--edge-min-opps", type=int, default=2)
    parser.add_argument("--nuc-min-size", type=int, default=85)
    parser.add_argument("--msp-min-size", type=int, default=60)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    model_path = get_model_path("hia5", tool="apply", seq=args.seq)
    model = freeze_model_for_inference(load_model(model_path))
    llr_hit, llr_miss = build_llr_tables(model)
    mode = "nanopore-fiber" if args.seq == "nanopore" else "pacbio-fiber"

    aggregate = Counter()
    aggregate_lengths: dict[str, list[int]] = defaultdict(list)
    geometry: dict[str, Geometry] = defaultdict(Geometry)
    seen = 0

    with pysam.AlignmentFile(args.bam, "rb", check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            if (
                read.is_unmapped
                or read.is_secondary
                or read.is_supplementary
                or int(read.query_length or 0) < args.min_read_length
            ):
                continue
            fiber_read = _extract_fiber_read_from_pysam(
                read, mode, args.prob_threshold
            )
            if fiber_read is None or fiber_read is CHIMERA_SKIP:
                continue
            result = run_hmm_apply_stage(
                fiber_read,
                model,
                edge_trim=10,
                circular=False,
                mode=mode,
                context_size=3,
                msp_min_size=args.msp_min_size,
                nuc_min_size=args.nuc_min_size,
                with_scores=False,
            )
            if result is None:
                continue
            stage, length_groups, tiled = audit_read(
                result,
                int(read.query_length),
                llr_hit,
                llr_miss,
                split_min_llr=args.split_min_llr,
                split_min_opps=args.split_min_opps,
                edge_min_llr=args.edge_min_llr,
                edge_min_opps=args.edge_min_opps,
                nuc_min_size=args.nuc_min_size,
                msp_min_size=args.msp_min_size,
            )
            aggregate.update(stage)
            for name, values in length_groups.items():
                aggregate_lengths[name].extend(values)
            for policy, (nucs, nfrs) in tiled.items():
                geometry[policy].add(int(read.query_length), nucs, nfrs)
            seen += 1
            if seen >= args.max_reads:
                break

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    stage_path = args.output_prefix.with_suffix(".stages.tsv")
    with stage_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("label", "metric", "value"))
        for metric, value in sorted(aggregate.items()):
            writer.writerow((args.label, metric, value))

    lengths_path = args.output_prefix.with_suffix(".lengths.tsv")
    with lengths_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ("label", "stage", "count", "p10", "p25", "median", "p75", "p90")
        )
        for stage, values in sorted(aggregate_lengths.items()):
            writer.writerow(
                (
                    args.label,
                    stage,
                    len(values),
                    _q(values, 0.10),
                    _q(values, 0.25),
                    _q(values, 0.50),
                    _q(values, 0.75),
                    _q(values, 0.90),
                )
            )

    geometry_path = args.output_prefix.with_suffix(".geometry.tsv")
    with geometry_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "label",
                "policy",
                "reads",
                "query_bp",
                "nuc_fraction",
                "nuc_per_kb",
                "nuc_length_p25",
                "nuc_length_median",
                "nuc_length_p75",
                "nfr_fraction_ge60",
                "nfr_per_kb",
                "nfr_length_p25",
                "nfr_length_median",
                "nfr_length_p75",
                "nfr_length_p90",
            )
        )
        for policy, values in sorted(geometry.items()):
            query_bp = max(1, values.query_bp)
            writer.writerow(
                (
                    args.label,
                    policy,
                    values.reads,
                    values.query_bp,
                    values.nuc_bp / query_bp,
                    1000.0 * len(values.nuc_lengths) / query_bp,
                    _q(values.nuc_lengths, 0.25),
                    _q(values.nuc_lengths, 0.50),
                    _q(values.nuc_lengths, 0.75),
                    values.nfr_bp / query_bp,
                    1000.0 * len(values.nfr_lengths) / query_bp,
                    _q(values.nfr_lengths, 0.25),
                    _q(values.nfr_lengths, 0.50),
                    _q(values.nfr_lengths, 0.75),
                    _q(values.nfr_lengths, 0.90),
                )
            )

    print(
        f"{args.label}: audited {seen} reads; wrote {stage_path}, "
        f"{lengths_path}, and {geometry_path}"
    )


if __name__ == "__main__":
    main()
