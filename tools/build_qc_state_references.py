#!/usr/bin/env python3
"""Build the state-aware (in-MSP / outside-MSP) QC references.

For each QC profile, every input BAM is sampled exactly as ``fiberhmm-qc``
samples it (2,000 reads, seed 20260824, MAPQ >= 20) and its per-read in-MSP
and outside-MSP rates are computed twice under the packaged definition:

* ``tags``: from the BAM's FiberHMM calls. Inputs must be called by this
  release's ``fiberhmm-call`` defaults, so a user's called BAM is compared
  with reference calls of the same kind.
* ``light_call``: from the QC light call (bundled apply HMM only) on every
  sampled read, the path an uncalled BAM takes.

Per-read rates of all inputs of a profile are pooled; the packaged entry
keeps only aggregate quantiles and counts (no read data).

    python tools/build_qc_state_references.py \\
        --profile dddb:dddb_reference.called.bam \\
        --profile ddda:napa.called.bam,uba1.called.bam \\
        --profile hia5_pacbio:hia5_pacbio.called.bam \\
        --disable hia5_nanopore:"source control not available locally" \\
        --write fiberhmm/qc/references.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pysam

from fiberhmm.qc.core import (
    DEFAULT_SAMPLE_READS,
    DEFAULT_SEED,
    default_qc_prob_threshold,
    load_references,
    sample_bam_reads,
)
from fiberhmm.qc.states import DEFAULT_GRADING, compute_state_rates

PROBABILITIES = [0.05, 0.25, 0.5, 0.75, 0.95]
_PROFILE_ASSAY = {
    "dddb": ("daf", "dddb"),
    "ddda": ("daf", "ddda"),
    "hia5_pacbio": ("pacbio-fiber", "hia5"),
    "hia5_nanopore": ("nanopore-fiber", "hia5"),
}


def _source_entry(blocks_and_arrays) -> dict:
    def compartment(key, array_key):
        values = np.concatenate([arrays[array_key] for _b, arrays in blocks_and_arrays])
        events = sum(block[key]["n_events"] for block, _a in blocks_and_arrays)
        opportunities = sum(block[key]["n_opportunities"] for block, _a in blocks_and_arrays)
        return {
            "probabilities": PROBABILITIES,
            "quantiles": np.quantile(values, PROBABILITIES).tolist(),
            "n_rate_reads": int(len(values)),
            "n_events": int(events),
            "n_opportunities": int(opportunities),
            "aggregate_rate": float(events / opportunities),
        }

    msp = compartment("msp", "msp_rates")
    outside = compartment("outside_msp", "outside_rates")
    all_events = sum(b["all_states"]["n_events"] for b, _a in blocks_and_arrays)
    all_opps = sum(b["all_states"]["n_opportunities"] for b, _a in blocks_and_arrays)
    msp_bp = np.concatenate([a["msp_length_fractions"] for _b, a in blocks_and_arrays])
    return {
        "msp": msp,
        "outside_msp": outside,
        "all_states_rate": float(all_events / all_opps),
        "msp_to_outside_ratio": float(msp["aggregate_rate"] / outside["aggregate_rate"]),
        "median_per_read_msp_length_fraction": float(np.median(msp_bp)),
        "n_reads_used": int(sum(b["reads"]["reads_used"] for b, _a in blocks_and_arrays)),
    }


def build_profile(profile: str, paths: list[str]) -> dict:
    mode, enzyme = _PROFILE_ASSAY[profile]
    threshold = default_qc_prob_threshold(mode, enzyme)
    by_source: dict = {}
    definition = None
    for source, option in (("tags", "tags"), ("light_call", "light-call")):
        results = []
        for path in paths:
            sampled = sample_bam_reads(path, DEFAULT_SAMPLE_READS, DEFAULT_SEED, 20)
            with pysam.AlignmentFile(path, "rb", check_sq=False) as handle:
                header = handle.header
            block, arrays = compute_state_rates(
                sampled.reads, mode, enzyme, header=header, prob_threshold=threshold,
                state_source=option, light_call_reads=len(sampled.reads),
                light_call_seconds=float("inf"),
            )
            if not block.get("available"):
                raise SystemExit(f"{profile}/{source}: {path}: {block.get('note')}")
            results.append((block, arrays))
            definition = block["definition"]
            print(f"{profile:14s} {source:10s} {Path(path).name}: "
                  f"in-MSP {block['msp']['median_per_read_rate']:.4f} "
                  f"outside {block['outside_msp']['median_per_read_rate']:.4f} "
                  f"({block['reads']['reads_used']} reads)", file=sys.stderr)
        by_source[source] = _source_entry(results)
    definition = {key: definition[key] for key in (
        "min_msp_bp", "edge_trim_bp", "terminal_segments",
        "min_state_opportunities_per_read", "probability_threshold", "daf_run_mask")}
    return {
        "scoring_enabled": True,
        "definition": definition,
        "grading": DEFAULT_GRADING,
        "by_source": by_source,
        "inputs": [Path(path).name for path in paths],
        "thresholds": (
            "efficiency (median per-read in-MSP rate): PASS while less than "
            "grading.efficiency.pass_relative below the reference median, WARN "
            "down to warn_relative below, FAIL further; background (median "
            "per-read outside-MSP rate): the same fractions above the median; "
            "medians of the reference's per-read rates under the sample's own "
            "state source (tags or light_call)"
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", action="append", default=[],
                        help="PROFILE:BAM[,BAM...] (called by this release's fiberhmm-call)")
    parser.add_argument("--disable", action="append", default=[],
                        help="PROFILE:NOTE - ship the profile without state calibration")
    parser.add_argument("--write", default=None, help="references.json to update in place")
    args = parser.parse_args(argv)
    references = load_references()
    for item in args.profile:
        profile, _sep, paths = item.partition(":")
        references["profiles"][profile]["state_rates"] = build_profile(
            profile, [path for path in paths.split(",") if path])
    for item in args.disable:
        profile, _sep, note = item.partition(":")
        references["profiles"][profile]["state_rates"] = {
            "scoring_enabled": False, "calibration_note": note}
    if not args.write:
        sys.stdout.write(json.dumps(references, indent=2) + "\n")
        return 0
    path = Path(args.write)
    original = path.read_text()
    if '"state_rates"' in original:
        path.write_text(json.dumps(references, indent=2) + "\n")
        return 0
    # First build: insert each entry after its profile's label line, leaving
    # the hand-formatted rest of the file untouched.
    lines = original.splitlines(keepends=True)
    output = []
    for line in lines:
        output.append(line)
        stripped = line.strip()
        for profile, entry in references["profiles"].items():
            if "state_rates" in entry and stripped == (
                    f'"label": {json.dumps(entry["label"])},'):
                indent = line[: len(line) - len(line.lstrip())]
                body = json.dumps(entry["state_rates"], indent=2).splitlines()
                output.append(f'{indent}"state_rates": {body[0]}\n')
                output.extend(f"{indent}{item}\n" for item in body[1:-1])
                output.append(f"{indent}{body[-1]},\n")
    text = "".join(output)
    if json.loads(text) != references:
        raise SystemExit("internal error: inserted references do not round-trip")
    path.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
