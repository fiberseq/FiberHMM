#!/usr/bin/env python3
"""Compare molecule-level NFR populations around the Drosophila eve locus.

Each raw read is rerun through the current HMM at the requested modification
threshold, then evaluated under three nucleosome interpretations:

* ``raw_hmm``: no nucleosome recall;
* ``conservative``: the current recaller;
* ``topology_fragment``: proposed topology-constrained, ambiguity-preserving
  recaller.

The BAM's pre-existing ``as/al`` calls are also retained as ``supplied``.  Calls
are projected from query coordinates to dm6, so the output captures NFR length,
per-molecule NFR density, and positional occupancy rather than only a global
accessible fraction.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pysam

from fiberhmm.core.model_io import freeze_model_for_inference, load_model
from fiberhmm.inference.engine import (
    CHIMERA_SKIP,
    _extract_fiber_read_from_pysam,
)
from fiberhmm.inference.fused_stages import run_hmm_apply_stage
from fiberhmm.inference.nuc_recaller import NucCall, assemble_nuc_msp_tiling
from fiberhmm.inference.tf_recaller import build_llr_tables
from fiberhmm.io.ma_tags import flip_intervals_to_seq
from fiberhmm.models import get_model_path

from instrument_recaller import audit_read


EVE_START = 9_979_318  # dm6 GTF, converted from 1-based inclusive
EVE_END = 9_980_795
EVE_CENTER = (EVE_START + EVE_END) // 2
WINDOW_START = 9_954_000
WINDOW_END = 10_006_000
PLOT_START = EVE_CENTER - 12_000
PLOT_END = EVE_CENTER + 12_000
BIN_SIZE = 100
POLICIES = (
    "supplied",
    "supplied_nuc_complement",
    "raw_hmm",
    "conservative",
    "topology_fragment",
)


@dataclass(frozen=True)
class InputSpec:
    label: str
    seq: str
    frame: str
    path: Path


def parse_input(value: str) -> InputSpec:
    try:
        metadata, path_text = value.split("=", 1)
        label, seq, frame = metadata.split(":", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--input must be LABEL:SEQ:FRAME=PATH"
        ) from exc
    if seq not in {"nanopore", "pacbio"}:
        raise argparse.ArgumentTypeError("SEQ must be nanopore or pacbio")
    if frame not in {"molecular", "query"}:
        raise argparse.ArgumentTypeError("FRAME must be molecular or query")
    return InputSpec(label, seq, frame, Path(path_text))


def _union_bp(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _supplied_nfrs(read, frame: str, floor: int):
    if not (read.has_tag("as") and read.has_tag("al")):
        return []
    starts = read.get_tag("as")
    lengths = read.get_tag("al")
    if frame == "molecular":
        starts, lengths = flip_intervals_to_seq(starts, lengths, read)
    return [
        (int(start), int(length))
        for start, length in zip(starts, lengths)
        if int(length) >= floor
    ]


def _supplied_nuc_complement(
    read,
    frame: str,
    nuc_floor: int,
    nfr_floor: int,
):
    """Derive actual inter-nucleosome gaps independent of legacy as/al semantics."""
    if not (read.has_tag("ns") and read.has_tag("nl")):
        return []
    starts = read.get_tag("ns")
    lengths = read.get_tag("nl")
    if frame == "molecular":
        starts, lengths = flip_intervals_to_seq(starts, lengths, read)
    nucs = [
        NucCall(int(start), int(length), 0, 0, 0)
        for start, length in zip(starts, lengths)
        if int(length) >= nuc_floor
    ]
    _, nfrs = assemble_nuc_msp_tiling(
        nucs,
        10,
        max(10, int(read.query_length or 0) - 10),
        nfr_floor,
        nuc_floor,
    )
    return nfrs


def _project_interval(
    reference_positions: list[int | None],
    start: int,
    length: int,
) -> tuple[int, int] | None:
    query_start = max(0, int(start))
    query_end = min(len(reference_positions), query_start + int(length))
    if query_end <= query_start:
        return None
    mapped = [
        int(position)
        for position in reference_positions[query_start:query_end]
        if position is not None
    ]
    if len(mapped) < max(1, int(0.50 * (query_end - query_start))):
        return None
    ref_start = min(mapped)
    ref_end = max(mapped) + 1
    if ref_end <= WINDOW_START or ref_start >= WINDOW_END:
        return None
    return max(WINDOW_START, ref_start), min(WINDOW_END, ref_end)


def _covered_bins(read) -> range:
    start = max(WINDOW_START, int(read.reference_start))
    end = min(WINDOW_END, int(read.reference_end or start))
    if end <= start:
        return range(0)
    first = max(0, (start - WINDOW_START) // BIN_SIZE)
    last = min(
        (WINDOW_END - WINDOW_START) // BIN_SIZE,
        (end - 1 - WINDOW_START) // BIN_SIZE,
    )
    return range(first, last + 1)


def _nfr_bins(intervals: list[tuple[int, int]]) -> set[int]:
    bins: set[int] = set()
    n_bins = (WINDOW_END - WINDOW_START) // BIN_SIZE
    for start, end in intervals:
        first = max(0, (start - WINDOW_START) // BIN_SIZE)
        last = min(n_bins - 1, (end - 1 - WINDOW_START) // BIN_SIZE)
        bins.update(range(first, last + 1))
    return bins


def _q(values, quantile: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values), quantile))


def _ecdf(values):
    ordered = np.sort(np.asarray(values, dtype=float))
    if ordered.size == 0:
        return ordered, ordered
    return ordered, np.arange(1, ordered.size + 1) / ordered.size


def _quantile_mae(left, right) -> float:
    if not left or not right:
        return float("nan")
    quantiles = np.linspace(0.01, 0.99, 99)
    return float(
        np.mean(
            np.abs(
                np.quantile(np.asarray(left), quantiles)
                - np.quantile(np.asarray(right), quantiles)
            )
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=parse_input,
        metavar="LABEL:SEQ:FRAME=PATH",
    )
    parser.add_argument("--prob-threshold", type=int, default=248)
    parser.add_argument("--nuc-min-size", type=int, default=85)
    parser.add_argument("--msp-min-size", type=int, default=60)
    parser.add_argument("--split-min-llr", type=float, default=4.0)
    parser.add_argument("--split-min-opps", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    call_rows: list[dict] = []
    read_rows: list[dict] = []
    denominator: dict[str, np.ndarray] = {}
    numerator: dict[tuple[str, str], np.ndarray] = {}
    n_bins = (WINDOW_END - WINDOW_START) // BIN_SIZE

    for spec in args.input:
        model_path = get_model_path("hia5", tool="apply", seq=spec.seq)
        model = freeze_model_for_inference(load_model(model_path))
        llr_hit, llr_miss = build_llr_tables(model)
        mode = "nanopore-fiber" if spec.seq == "nanopore" else "pacbio-fiber"
        denominator[spec.label] = np.zeros(n_bins, dtype=np.int64)
        for policy in POLICIES:
            numerator[(spec.label, policy)] = np.zeros(n_bins, dtype=np.int64)

        with pysam.AlignmentFile(spec.path, "rb", check_sq=False) as bam:
            for read in bam.fetch("chr2R", WINDOW_START, WINDOW_END):
                if (
                    read.is_unmapped
                    or read.is_secondary
                    or read.is_supplementary
                    or int(read.query_length or 0) < 1000
                ):
                    continue
                fiber_read = _extract_fiber_read_from_pysam(
                    read, mode, args.prob_threshold
                )
                if fiber_read is None or fiber_read is CHIMERA_SKIP:
                    continue
                apply_result = run_hmm_apply_stage(
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
                if apply_result is None:
                    continue
                _, _, tiled = audit_read(
                    apply_result,
                    int(read.query_length),
                    llr_hit,
                    llr_miss,
                    split_min_llr=args.split_min_llr,
                    split_min_opps=args.split_min_opps,
                    edge_min_llr=2.0,
                    edge_min_opps=2,
                    nuc_min_size=args.nuc_min_size,
                    msp_min_size=args.msp_min_size,
                )
                policy_nfrs = {
                    "supplied": _supplied_nfrs(
                        read, spec.frame, args.msp_min_size
                    ),
                    "supplied_nuc_complement": _supplied_nuc_complement(
                        read,
                        spec.frame,
                        args.nuc_min_size,
                        args.msp_min_size,
                    ),
                }
                for policy in POLICIES[2:]:
                    policy_nfrs[policy] = tiled[policy][1]

                covered_bins = list(_covered_bins(read))
                denominator[spec.label][covered_bins] += 1
                aligned_start = max(WINDOW_START, int(read.reference_start))
                aligned_end = min(
                    WINDOW_END, int(read.reference_end or aligned_start)
                )
                aligned_bp = max(0, aligned_end - aligned_start)
                reference_positions = read.get_reference_positions(
                    full_length=True
                )

                for policy, nfrs in policy_nfrs.items():
                    projected = []
                    query_lengths = []
                    for start, length in nfrs:
                        interval = _project_interval(
                            reference_positions, start, length
                        )
                        if interval is None:
                            continue
                        projected.append(interval)
                        query_lengths.append(int(length))
                        call_rows.append(
                            {
                                "label": spec.label,
                                "policy": policy,
                                "read_id": read.query_name,
                                "query_start": int(start),
                                "query_length": int(length),
                                "ref_start": interval[0],
                                "ref_end": interval[1],
                                "ref_length": interval[1] - interval[0],
                                "ref_center": (interval[0] + interval[1]) / 2,
                            }
                        )
                    numerator[(spec.label, policy)][
                        list(_nfr_bins(projected))
                    ] += 1
                    clipped = [
                        (
                            max(aligned_start, start),
                            min(aligned_end, end),
                        )
                        for start, end in projected
                        if min(aligned_end, end) > max(aligned_start, start)
                    ]
                    nfr_bp = _union_bp(clipped)
                    read_rows.append(
                        {
                            "label": spec.label,
                            "policy": policy,
                            "read_id": read.query_name,
                            "aligned_bp": aligned_bp,
                            "nfr_count": len(projected),
                            "nfr_bp": nfr_bp,
                            "nfr_fraction": (
                                nfr_bp / aligned_bp if aligned_bp else float("nan")
                            ),
                            "nfr_per_10kb": (
                                10_000 * len(projected) / aligned_bp
                                if aligned_bp
                                else float("nan")
                            ),
                            "median_nfr_length": (
                                float(np.median(query_lengths))
                                if query_lengths
                                else float("nan")
                            ),
                        }
                    )

    def write_rows(name: str, rows: list[dict]) -> None:
        path = args.output_dir / name
        opener = gzip.open if path.suffix == ".gz" else path.open
        with opener(path, "wt", newline="") if path.suffix == ".gz" else opener(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0]), delimiter="\t"
            )
            writer.writeheader()
            writer.writerows(rows)

    write_rows("eve_nfr_calls.tsv.gz", call_rows)
    write_rows("eve_per_read.tsv.gz", read_rows)

    occupancy_rows = []
    for spec in args.input:
        den = denominator[spec.label]
        for policy in POLICIES:
            num = numerator[(spec.label, policy)]
            for index in range(n_bins):
                bin_start = WINDOW_START + index * BIN_SIZE
                occupancy_rows.append(
                    {
                        "label": spec.label,
                        "policy": policy,
                        "bin_start": bin_start,
                        "bin_end": bin_start + BIN_SIZE,
                        "covered_reads": int(den[index]),
                        "nfr_reads": int(num[index]),
                        "nfr_occupancy": (
                            float(num[index] / den[index])
                            if den[index]
                            else float("nan")
                        ),
                    }
                )
    write_rows("eve_position_occupancy.tsv", occupancy_rows)

    summary_rows = []
    for spec in args.input:
        for policy in POLICIES:
            calls = [
                row
                for row in call_rows
                if row["label"] == spec.label and row["policy"] == policy
            ]
            reads = [
                row
                for row in read_rows
                if row["label"] == spec.label and row["policy"] == policy
            ]
            lengths = [row["query_length"] for row in calls]
            densities = [
                row["nfr_per_10kb"]
                for row in reads
                if np.isfinite(row["nfr_per_10kb"])
            ]
            fractions = [
                row["nfr_fraction"]
                for row in reads
                if np.isfinite(row["nfr_fraction"])
            ]
            summary_rows.append(
                {
                    "label": spec.label,
                    "policy": policy,
                    "reads": len(reads),
                    "nfr_calls": len(calls),
                    "nfr_length_p25": _q(lengths, 0.25),
                    "nfr_length_median": _q(lengths, 0.50),
                    "nfr_length_p75": _q(lengths, 0.75),
                    "nfr_length_p90": _q(lengths, 0.90),
                    "per_read_nfr_per_10kb_median": _q(densities, 0.50),
                    "per_read_nfr_per_10kb_p75": _q(densities, 0.75),
                    "per_read_nfr_fraction_median": _q(fractions, 0.50),
                    "aggregate_nfr_fraction": (
                        sum(row["nfr_bp"] for row in reads)
                        / max(1, sum(row["aligned_bp"] for row in reads))
                    ),
                }
            )
    write_rows("eve_summary.tsv", summary_rows)

    reference_policy = "supplied_nuc_complement"
    reference_lengths = [
        row["query_length"]
        for row in call_rows
        if (
            row["label"] == "pacbio_2_4"
            and row["policy"] == reference_policy
        )
    ]
    reference_density = [
        row["nfr_per_10kb"]
        for row in read_rows
        if (
            row["label"] == "pacbio_2_4"
            and row["policy"] == reference_policy
            and np.isfinite(row["nfr_per_10kb"])
        )
    ]
    reference_position = {
        row["bin_start"]: row["nfr_occupancy"]
        for row in occupancy_rows
        if (
            row["label"] == "pacbio_2_4"
            and row["policy"] == reference_policy
            and PLOT_START <= row["bin_start"] < PLOT_END
        )
    }
    aggregate_fraction = {
        (row["label"], row["policy"]): row["aggregate_nfr_fraction"]
        for row in summary_rows
    }
    comparison_rows = []
    for spec in args.input:
        for policy in POLICIES:
            lengths = [
                row["query_length"]
                for row in call_rows
                if row["label"] == spec.label and row["policy"] == policy
            ]
            density = [
                row["nfr_per_10kb"]
                for row in read_rows
                if (
                    row["label"] == spec.label
                    and row["policy"] == policy
                    and np.isfinite(row["nfr_per_10kb"])
                )
            ]
            position = {
                row["bin_start"]: row["nfr_occupancy"]
                for row in occupancy_rows
                if (
                    row["label"] == spec.label
                    and row["policy"] == policy
                    and PLOT_START <= row["bin_start"] < PLOT_END
                )
            }
            common = sorted(set(reference_position) & set(position))
            reference_values = np.asarray(
                [reference_position[key] for key in common], dtype=float
            )
            position_values = np.asarray(
                [position[key] for key in common], dtype=float
            )
            finite = np.isfinite(reference_values) & np.isfinite(position_values)
            reference_values = reference_values[finite]
            position_values = position_values[finite]
            position_rmse = (
                float(np.sqrt(np.mean((position_values - reference_values) ** 2)))
                if reference_values.size
                else float("nan")
            )
            position_correlation = (
                float(np.corrcoef(reference_values, position_values)[0, 1])
                if reference_values.size > 1
                and np.std(reference_values) > 0
                and np.std(position_values) > 0
                else float("nan")
            )
            comparison_rows.append(
                {
                    "label": spec.label,
                    "policy": policy,
                    "reference": "pacbio_2_4:supplied_nuc_complement",
                    "nfr_length_quantile_mae_bp": _quantile_mae(
                        lengths, reference_lengths
                    ),
                    "nfr_density_quantile_mae_per_10kb": _quantile_mae(
                        density, reference_density
                    ),
                    "position_occupancy_rmse": position_rmse,
                    "position_occupancy_correlation": position_correlation,
                    "aggregate_nfr_fraction_abs_diff": abs(
                        aggregate_fraction[(spec.label, policy)]
                        - aggregate_fraction[
                            ("pacbio_2_4", reference_policy)
                        ]
                    ),
                }
            )
    write_rows("eve_comparison_to_pacbio.tsv", comparison_rows)

    # Main population figure: supplied ONT calls show the failure. Proposed ONT
    # calls are compared with inter-nucleosome gaps re-derived from the existing
    # PacBio ns/nl, avoiding the old-vs-current as/al semantic difference.
    colors = {
        "pacbio_2_4": "#222222",
        "ont_early": "#1772b8",
        "ont_late": "#d95f02",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    ax_len, ax_density, ax_pos, ax_delta = axes.ravel()

    for spec in args.input:
        display_policies = (
            ("supplied_nuc_complement", "-", 1.0)
            if spec.seq == "pacbio"
            else (
                ("supplied", "--", 0.55),
                ("topology_fragment", "-", 1.0),
            )
        )
        if spec.seq == "pacbio":
            display_policies = (display_policies,)
        for policy, linestyle, alpha in display_policies:
            values = [
                row["query_length"]
                for row in call_rows
                if row["label"] == spec.label and row["policy"] == policy
            ]
            x, y = _ecdf(values)
            ax_len.plot(
                x,
                y,
                color=colors.get(spec.label),
                linestyle=linestyle,
                alpha=alpha,
                label=f"{spec.label}: {policy}",
            )
            density = [
                row["nfr_per_10kb"]
                for row in read_rows
                if (
                    row["label"] == spec.label
                    and row["policy"] == policy
                    and np.isfinite(row["nfr_per_10kb"])
                )
            ]
            x, y = _ecdf(density)
            ax_density.plot(
                x,
                y,
                color=colors.get(spec.label),
                linestyle=linestyle,
                alpha=alpha,
            )

    for spec in args.input:
        policy = (
            "supplied_nuc_complement"
            if spec.seq == "pacbio"
            else "topology_fragment"
        )
        rows = [
            row
            for row in occupancy_rows
            if (
                row["label"] == spec.label
                and row["policy"] == policy
                and PLOT_START <= row["bin_start"] < PLOT_END
            )
        ]
        x = [
            ((row["bin_start"] + row["bin_end"]) / 2 - EVE_CENTER) / 1000
            for row in rows
        ]
        y = [row["nfr_occupancy"] for row in rows]
        ax_pos.plot(
            x,
            y,
            color=colors.get(spec.label),
            label=f"{spec.label}: {policy}",
        )

    pb_rows = {
        row["bin_start"]: row["nfr_occupancy"]
        for row in occupancy_rows
        if (
            row["label"] == "pacbio_2_4"
            and row["policy"] == "supplied_nuc_complement"
        )
    }
    for label in ("ont_early", "ont_late"):
        rows = [
            row
            for row in occupancy_rows
            if (
                row["label"] == label
                and row["policy"] == "topology_fragment"
                and PLOT_START <= row["bin_start"] < PLOT_END
            )
        ]
        x = [
            ((row["bin_start"] + row["bin_end"]) / 2 - EVE_CENTER) / 1000
            for row in rows
        ]
        y = [
            row["nfr_occupancy"] - pb_rows.get(row["bin_start"], np.nan)
            for row in rows
        ]
        ax_delta.plot(x, y, color=colors[label], label=label)

    ax_len.set_xlim(60, 600)
    ax_len.set_xlabel("NFR length on molecule (bp)")
    ax_len.set_ylabel("Cumulative fraction")
    ax_len.set_title("eve-window NFR size population")
    ax_len.legend(fontsize=7, ncol=2)
    ax_density.set_xlim(0, 30)
    ax_density.set_xlabel("NFR calls per 10 kb per molecule")
    ax_density.set_ylabel("Cumulative fraction")
    ax_density.set_title("Per-molecule NFR number")
    ax_pos.axvspan(
        (EVE_START - EVE_CENTER) / 1000,
        (EVE_END - EVE_CENTER) / 1000,
        color="#777777",
        alpha=0.15,
        label="eve gene",
    )
    ax_pos.set_xlabel("Position relative to eve center (kb)")
    ax_pos.set_ylabel("Molecules with an NFR")
    ax_pos.set_title("Positional NFR occupancy")
    ax_pos.legend(fontsize=8)
    ax_delta.axhline(0, color="#555555", linewidth=0.8)
    ax_delta.axvspan(
        (EVE_START - EVE_CENTER) / 1000,
        (EVE_END - EVE_CENTER) / 1000,
        color="#777777",
        alpha=0.15,
    )
    ax_delta.set_xlabel("Position relative to eve center (kb)")
    ax_delta.set_ylabel("ONT proposed − PacBio supplied")
    ax_delta.set_title("Residual positional difference")
    ax_delta.legend(fontsize=8)
    for axis in axes.ravel():
        axis.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(args.output_dir / "eve_nfr_population.png", dpi=180)
    fig.savefig(args.output_dir / "eve_nfr_population.pdf")
    plt.close(fig)

    print(
        f"Wrote eve population benchmark for {len(args.input)} cohorts to "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
