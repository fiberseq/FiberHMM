#!/usr/bin/env python3
"""Measure how molecule-specific DAF m5C spans change TF recall."""
from __future__ import annotations

import argparse
import glob
import importlib.util
import re
import sys
from itertools import chain
from pathlib import Path

import numpy as np
import pysam

from fiberhmm.core.model_io import load_model_with_metadata
from fiberhmm.daf.m5c import (
    DDDA_FIVE_PRIME_FACTORS,
    add_m5c_ma_tag,
    call_read_m5c,
    collect_read_observations,
)
from fiberhmm.inference.tf_recaller import (
    build_llr_tables,
    build_m5c_llr_tables,
    recall_read,
)
from fiberhmm.io.bam_header import header_has_coord_marker
from fiberhmm.io.ma_tags import llr_to_tq


DEFAULT_MODEL = "fiberhmm/models/ddda_TF.json"
PRIMARY_HAPLOTYPE = re.compile(
    r"^chr(?:[1-9]|1[0-9]|2[0-2]|X|Y)_(?:MATERNAL|PATERNAL)$"
)


def load_truth_module(path):
    spec = importlib.util.spec_from_file_location("m5c_truth_external", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bam-glob", default="",
                        help="Input BAM glob; alternatively repeat --bam")
    parser.add_argument("--bam", action="append", default=[],
                        help="Input BAM; repeat to override --bam-glob")
    parser.add_argument("--reference", required=True,
                        help="Indexed diploid reference FASTA")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--truth-module", default="",
                        help="Python module providing load_truth(); required unless --whole-genome")
    parser.add_argument("--chrom", default="chr1_MATERNAL")
    parser.add_argument("--start", type=int, default=30_000_000)
    parser.add_argument("--end", type=int, default=33_000_000)
    parser.add_argument("--min-llr", type=float, default=5.0)
    parser.add_argument("--min-opps", type=int, default=3)
    parser.add_argument("--unify-threshold", type=int, default=90)
    parser.add_argument("--run-bp", type=float, default=5000)
    parser.add_argument("--posterior", type=float, default=0.99)
    parser.add_argument("--baseline-radius", type=int, default=250)
    parser.add_argument("--whole-genome", action="store_true",
                        help="Score primary HG002 haplotype contigs; omit regional truth details")
    parser.add_argument("--output", default="",
                        help="Optional metrics TSV (default: stdout only)")
    parser.add_argument("--progress-every", type=int, default=100_000)
    args = parser.parse_args()

    if not args.bam and not args.bam_glob:
        parser.error("supply --bam at least once or provide --bam-glob")
    if not args.whole_genome and not args.truth_module:
        parser.error("--truth-module is required unless --whole-genome is used")

    model, context_size, mode = load_model_with_metadata(args.model)
    llr_hit, llr_miss = build_llr_tables(model)
    m5c_hit, m5c_miss = build_m5c_llr_tables(model)
    factors = DDDA_FIVE_PRIME_FACTORS / DDDA_FIVE_PRIME_FACTORS.mean()
    truth_loader = None if args.whole_genome else load_truth_module(args.truth_module)
    stats = {
        "reads": 0, "eligible_reads": 0, "tagged_reads": 0, "spans": 0,
        "called_cpgs": 0, "tf_off": 0, "tf_on": 0, "removed": 0,
        "added": 0, "changed_reads": 0, "removed_overlapping_m5c": 0,
        "tf_off_overlapping_m5c": 0, "tf_on_overlapping_m5c": 0,
        "matched_overlapping_m5c": 0, "matched_llr_changed": 0,
        "matched_tq_changed": 0, "matched_llr_increased": 0,
        "matched_llr_decreased": 0, "matched_llr_delta_sum": 0.0,
        "matched_llr_abs_delta_sum": 0.0, "score_changed_reads": 0,
    }
    removed_truth = []
    added_truth = []
    with pysam.FastaFile(args.reference) as fasta:
        truth = ({} if args.whole_genome else truth_loader.load_truth(
            fasta, args.chrom, args.start, args.end, verbose=True,
        ))
        paths = args.bam or sorted(glob.glob(args.bam_glob))
        for path in paths:
            with pysam.AlignmentFile(path, "rb") as bam:
                molecular_frame = header_has_coord_marker(bam.header)
                if args.whole_genome:
                    # The diploid reference also contains alternate/unplaced
                    # contigs and an unmapped tail. Fetching the 46 requested
                    # primary haplotypes directly avoids streaming gigabytes
                    # of records that are immediately discarded.
                    source = chain.from_iterable(
                        bam.fetch(chrom) for chrom in bam.references
                        if PRIMARY_HAPLOTYPE.match(chrom)
                    )
                else:
                    source = bam.fetch(args.chrom, args.start, args.end)
                path_reads = 0
                for read in source:
                    if args.whole_genome and (
                        read.is_unmapped or read.is_secondary or
                        read.is_supplementary or
                        not PRIMARY_HAPLOTYPE.match(read.reference_name or "")
                    ):
                        continue
                    stats["reads"] += 1
                    path_reads += 1
                    if args.progress_every and path_reads % args.progress_every == 0:
                        print(
                            f"[m5c_tf_effect] {Path(path).name}: "
                            f"{path_reads:,} primary reads",
                            file=sys.stderr, flush=True,
                        )
                    observations = collect_read_observations(
                        read, fasta,
                        None if args.whole_genome else args.chrom,
                        None if args.whole_genome else args.start,
                        None if args.whole_genome else args.end,
                        input_molecular_frame=molecular_frame,
                    )
                    if observations:
                        stats["eligible_reads"] += 1
                        result = call_read_m5c(
                            observations, factors,
                            expected_run_bp=args.run_bp,
                            posterior_threshold=args.posterior,
                            baseline_radius=args.baseline_radius,
                            min_other=10, min_call_cpg=2,
                        )
                        spans = [(call.start, call.end) for call in result.calls]
                        stats["spans"] += len(spans)
                        stats["called_cpgs"] += sum(
                            call.n_cpg for call in result.calls
                        )
                    else:
                        spans = []
                    add_m5c_ma_tag(read, spans)
                    stats["tagged_reads"] += int(bool(spans))
                    off, _nuc_off, _msp_off = recall_read(
                        read, llr_hit, llr_miss, mode, context_size,
                        args.min_llr, args.min_opps, args.unify_threshold,
                        input_molecular_frame=molecular_frame,
                    )
                    # With no m5c span the corrected tables are never selected,
                    # so the on result is exactly the off result. Avoid a second
                    # full TF scan for the majority of reads.
                    if spans:
                        on, _nuc_on, _msp_on = recall_read(
                            read, llr_hit, llr_miss, mode, context_size,
                            args.min_llr, args.min_opps, args.unify_threshold,
                            input_molecular_frame=molecular_frame,
                            m5c_llr_hit=m5c_hit, m5c_llr_miss=m5c_miss,
                        )
                    else:
                        on = off
                    off_by_interval = {(call.start, call.length): call for call in off}
                    on_by_interval = {(call.start, call.length): call for call in on}
                    removed = set(off_by_interval) - set(on_by_interval)
                    added = set(on_by_interval) - set(off_by_interval)
                    matched = set(off_by_interval) & set(on_by_interval)
                    stats["tf_off"] += len(off)
                    stats["tf_on"] += len(on)
                    stats["tf_off_overlapping_m5c"] += sum(
                        any(lo < call.start + call.length and hi > call.start
                            for lo, hi in spans)
                        for call in off
                    )
                    stats["tf_on_overlapping_m5c"] += sum(
                        any(lo < call.start + call.length and hi > call.start
                            for lo, hi in spans)
                        for call in on
                    )
                    stats["removed"] += len(removed)
                    stats["added"] += len(added)
                    stats["changed_reads"] += int(bool(removed or added))
                    read_score_changed = False
                    for interval in matched:
                        start, length = interval
                        end = start + length
                        if not any(lo < end and hi > start for lo, hi in spans):
                            continue
                        stats["matched_overlapping_m5c"] += 1
                        off_call = off_by_interval[interval]
                        on_call = on_by_interval[interval]
                        delta = float(on_call.llr - off_call.llr)
                        if abs(delta) > 1e-12:
                            stats["matched_llr_changed"] += 1
                            stats["matched_llr_increased"] += int(delta > 0)
                            stats["matched_llr_decreased"] += int(delta < 0)
                            stats["matched_llr_delta_sum"] += delta
                            stats["matched_llr_abs_delta_sum"] += abs(delta)
                            stats["matched_tq_changed"] += int(
                                llr_to_tq(on_call.llr) != llr_to_tq(off_call.llr)
                            )
                            read_score_changed = True
                    stats["score_changed_reads"] += int(read_score_changed)
                    for start, length in removed:
                        end = start + length
                        overlap = any(lo < end and hi > start for lo, hi in spans)
                        stats["removed_overlapping_m5c"] += int(overlap)
                        seen = set()
                        for obs in observations:
                            if (obs.is_cpg and start <= obs.query_pos < end and
                                    obs.reference_pos not in seen):
                                beta = truth.get(obs.reference_pos)
                                if beta is not None:
                                    removed_truth.append(beta)
                                    seen.add(obs.reference_pos)
                    for start, length in added:
                        end = start + length
                        seen = set()
                        for obs in observations:
                            if (obs.is_cpg and start <= obs.query_pos < end and
                                    obs.reference_pos not in seen):
                                beta = truth.get(obs.reference_pos)
                                if beta is not None:
                                    added_truth.append(beta)
                                    seen.add(obs.reference_pos)

            print(
                f"[m5c_tf_effect] finished {Path(path).name}: "
                f"{path_reads:,} primary reads",
                file=sys.stderr, flush=True,
            )

    rows = [(key, value) for key, value in stats.items()]
    for label, values in (("removed", removed_truth), ("added", added_truth)):
        values = np.asarray(values, dtype=float)
        rows.extend([
            (f"{label}_truth_cpgs", len(values)),
            (f"{label}_truth_mean", np.mean(values) if len(values) else np.nan),
            (f"{label}_truth_gt_0.5",
             np.mean(values > 0.5) if len(values) else np.nan),
        ])
    text = "".join(f"{key}\t{value}\n" for key, value in rows)
    print(text, end="")
    if args.output:
        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
