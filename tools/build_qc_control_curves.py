#!/usr/bin/env python3
"""Build a QC profile's packaged control curves, exemplars and grades.

Every input BAM is sampled exactly as ``fiberhmm-qc`` samples it (2,000
reads, seed 20260824, MAPQ >= 20) and measured with the same
``analyze_sample`` path a user's BAM takes. The profile's entries are then
replaced in

* ``control_curves.json``: the pooled per-read rate ECDF (probabilities
  0.01-0.99), the phasogram of the pooled pair-distance histogram (lags
  60-800 bp) and the nucleosome / TF footprint-size histograms;
* ``control_examples.json``: rate-stratified, coordinate-free molecules,
  one group per input;
* ``references.json``: ``rate`` (median, IQR = PASS, 5th-95th = WARN) and
  ``periodicity`` (reference NRL and strength; PASS / WARN strength at
  ``--pass-strength-fraction`` / ``--warn-strength-fraction`` of the
  reference strength). ``state_rates`` is left alone
  (``tools/build_qc_state_references.py``).

Inputs should be called by this release's ``fiberhmm-call`` defaults, so
the footprint sizes are those a user's call produces.

    python tools/build_qc_control_curves.py --profile ddda \\
        --input "NAPA amplicon=napa.called.bam" \\
        --input "UBA1 amplicon=uba1.called.bam" \\
        --label "Human DddA DAF-seq controls" --write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from fiberhmm.qc.core import (
    DEFAULT_SAMPLE_READS,
    DEFAULT_SEED,
    MAX_LAG,
    PERIOD_BAND,
    PLOT_MAX_LAG,
    _rate_stratified_examples,
    analyze_sample,
    default_qc_prob_threshold,
    phasogram_metrics,
    sample_bam_reads,
)

QC_DIR = Path(__file__).resolve().parents[1] / "fiberhmm" / "qc"
_PROFILE_ASSAY = {
    "dddb": ("daf", "dddb"),
    "ddda": ("daf", "ddda"),
    "hia5_pacbio": ("pacbio-fiber", "hia5"),
    "hia5_nanopore": ("nanopore-fiber", "hia5"),
}
ECDF_PROBABILITIES = [round(0.01 * step, 2) for step in range(1, 100)]
RATE_PROBABILITIES = [0.05, 0.25, 0.5, 0.75, 0.95]
MIN_MAPQ = 20
MIN_OPPORTUNITIES = 200


def _size_histogram(values, step: int, stop: int, label: str):
    if not len(values):
        return None
    values = np.asarray(values, dtype=float)
    edges = np.arange(0, stop + step, step)
    counts, _ = np.histogram(values, bins=edges)
    return {
        "bin_edges_bp": edges.astype(int).tolist(),
        "fraction_per_bin": (counts / len(values)).tolist(),
        "median_bp": float(np.median(values)),
        "q25_bp": float(np.quantile(values, 0.25)),
        "q75_bp": float(np.quantile(values, 0.75)),
        "n_calls_total": int(len(values)),
        "n_calls_in_range": int(counts.sum()),
        "overflow_calls": int(np.sum(values >= stop)),
        "source_label": label,
    }


def _exemplar(example: dict) -> dict:
    start = int(example["span_start"])
    return {
        "signal_positions_bp": sorted(int(p) - start for p in example["positions"]),
        "signal_rate": float(example["rate"]),
        "span_bp": int(example["span_end"]) - start,
    }


def build(profile: str, inputs: list[tuple[str, str]], label: str,
          examples_per_input: int, pass_fraction: float, warn_fraction: float):
    mode, _enzyme = _PROFILE_ASSAY[profile]
    threshold = default_qc_prob_threshold(mode, _enzyme)
    rates, nuc, tf, accounting, groups = [], [], [], [], []
    histogram = np.zeros(MAX_LAG + 1)
    for group_label, path in inputs:
        sampled = sample_bam_reads(path, DEFAULT_SAMPLE_READS, DEFAULT_SEED, MIN_MAPQ)
        result, arrays = analyze_sample(
            sampled.reads, mode, None, prob_threshold=threshold,
            min_opportunities=MIN_OPPORTUNITIES)
        rates.append(arrays["rates"])
        nuc.append(arrays["nuc_lengths"])
        tf.append(arrays["tf_lengths"])
        histogram += arrays["histogram"]
        source = {
            "label": f"{label} — {group_label}",
            "records_examined": int(sampled.records_examined),
            "sampled_reads": int(len(sampled.reads)),
            "strategy": sampled.strategy,
        }
        accounting.append(source)
        chosen = _rate_stratified_examples(arrays["read_examples"], examples_per_input)
        groups.append({
            "label": source["label"],
            "reads": [_exemplar(item) for item in chosen],
            "source_accounting": {
                "bounded_sample_reads": source["sampled_reads"],
                "eligible_visual_exemplars": len(arrays["read_examples"]),
                "records_examined": source["records_examined"],
                "sampling_strategy": source["strategy"],
                "selected_exemplars": len(chosen),
                "source_label": source["label"],
            },
        })
        _c, nrl, strength = phasogram_metrics(arrays["histogram"])
        print(f"{profile} {group_label}: {len(sampled.reads)} reads, median rate "
              f"{result['signal']['median_per_read_rate']:.4f}, NRL {nrl}, "
              f"strength {strength:.4f}, nuc median "
              f"{result['footprints']['median_nucleosome_bp']}, TF median "
              f"{result['footprints']['median_tf_footprint_bp']}", file=sys.stderr)

    rate_values = np.concatenate(rates)
    curve, nrl, strength = phasogram_metrics(histogram)
    lags = list(range(60, PLOT_MAX_LAG + 1))
    nuc_values, tf_values = np.concatenate(nuc), np.concatenate(tf)
    curves = {
        "control_label": label,
        "footprint_sizes": {
            "contains_individual_read_data": False,
            "nucleosome": _size_histogram(nuc_values, 10, 2000, label),
            "source_accounting": [
                {key: item[key] for key in ("records_examined", "sampled_reads", "strategy")}
                for item in accounting
            ],
            "tf": _size_histogram(tf_values, 5, 500, label),
        },
        "minimum_mapq": MIN_MAPQ,
        "minimum_opportunities_per_read": MIN_OPPORTUNITIES,
        "mode": mode,
        "phasogram": {
            "detrended_pair_frequency": [float(curve[lag]) for lag in lags],
            "lags_bp": lags,
        },
        "probability_threshold": threshold,
        "rate_ecdf": {
            "probabilities": ECDF_PROBABILITIES,
            "rates": np.quantile(rate_values, ECDF_PROBABILITIES).tolist(),
        },
        "seed": DEFAULT_SEED,
        "source_accounting": accounting,
    }
    quantiles = np.quantile(rate_values, RATE_PROBABILITIES).tolist()
    rate = {
        "metric": "median_per_read",
        "pass_interval": [quantiles[1], quantiles[3]],
        "reference_iqr": [quantiles[1], quantiles[3]],
        "reference_median": quantiles[2],
        "reference_quantile_probabilities": RATE_PROBABILITIES,
        "reference_quantiles": quantiles,
        "warn_interval": [quantiles[0], quantiles[4]],
    }
    periodicity = {
        "pass_strength": round(pass_fraction * strength, 4),
        "reference_nrl_bp": nrl,
        "reference_strength": strength,
        "search_interval_bp": list(PERIOD_BAND),
        "warn_strength": round(warn_fraction * strength, 4),
    }
    print(f"{profile} pooled: {len(rate_values)} rate reads, median "
          f"{quantiles[2]:.4f}, NRL {nrl}, strength {strength:.4f}", file=sys.stderr)
    return curves, groups, rate, periodicity


def _update(path: Path, mutate) -> None:
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", required=True, choices=sorted(_PROFILE_ASSAY))
    parser.add_argument("--input", action="append", required=True,
                        help="GROUP_LABEL=BAM (called by this release's fiberhmm-call)")
    parser.add_argument("--label", required=True, help="control label shown in reports")
    parser.add_argument("--examples-per-input", type=int, default=8)
    parser.add_argument("--pass-strength-fraction", type=float, default=0.35)
    parser.add_argument("--warn-strength-fraction", type=float, default=0.13)
    parser.add_argument("--write", action="store_true",
                        help="update the packaged files in fiberhmm/qc/ in place")
    args = parser.parse_args(argv)
    inputs = [tuple(item.split("=", 1)) for item in args.input]
    curves, groups, rate, periodicity = build(
        args.profile, inputs, args.label, args.examples_per_input,
        args.pass_strength_fraction, args.warn_strength_fraction)
    if not args.write:
        json.dump({"curves": curves, "examples": groups, "rate": rate,
                   "periodicity": periodicity}, sys.stdout, indent=2)
        return 0

    def curves_update(payload):
        payload["profiles"][args.profile] = curves

    def examples_update(payload):
        payload["profiles"][args.profile] = {"groups": groups}

    def references_update(payload):
        entry = payload["profiles"][args.profile]
        entry["rate"].update(rate)
        entry["periodicity"].update(periodicity)

    _update(QC_DIR / "control_curves.json", curves_update)
    _update(QC_DIR / "control_examples.json", examples_update)
    _update(QC_DIR / "references.json", references_update)
    return 0


if __name__ == "__main__":
    sys.exit(main())
