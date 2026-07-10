#!/usr/bin/env python3
"""Chunked genome-wide aggregate DAF m5C caller and truth validator."""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pysam
from scipy.stats import rankdata, spearmanr

from fiberhmm.daf.m5c import (
    BETA_METH,
    BETA_UNMETH,
    M5CDomain,
    call_domains_from_emissions,
    collect_bam_observations,
    make_windows,
    window_log_likelihood,
    write_bed,
)

DEFAULT_FACTORS = np.array([0.865, 1.045, 0.944, 1.146])
PRIMARY_HAPLOTYPE = re.compile(
    r"^chr(?:[1-9]|1[0-9]|2[0-2]|X|Y)_(?:MATERNAL|PATERNAL)$"
)


def auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    if not labels.any() or labels.all():
        return np.nan
    ranks = rankdata(scores)
    n1, n0 = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _best_exact_lag(positions, reference_mask, reference_positions, previous_lag,
                    local_radius=100, broad_radius=10_000):
    """Track a coordinate lag, broadening when a local assembly jump occurs."""
    def score(lags):
        shifted = positions[:, None] + lags[None, :]
        valid = (shifted >= 0) & (shifted < len(reference_mask))
        hits = np.zeros_like(valid)
        hits[valid] = reference_mask[shifted[valid]]
        values = hits.sum(axis=0)
        best = values.max()
        tied = np.flatnonzero(values == best)
        chosen = tied[np.argmin(np.abs(lags[tied] - previous_lag))]
        return int(lags[chosen]), int(best)

    local = np.arange(previous_lag - local_radius,
                      previous_lag + local_radius + 1, dtype=np.int64)
    lag, matches = score(local)
    if matches >= max(20, int(0.8 * len(positions))):
        return lag, matches

    # Candidate lags come from exact CpG-to-CpG differences near the expected
    # coordinate. Correct assembly offsets recur across most sites in the bin;
    # random nearby CpGs do not.
    candidates = [lag]
    for pos in positions[::max(1, len(positions) // 25)]:
        center = int(pos + previous_lag)
        left = np.searchsorted(reference_positions, center - broad_radius)
        right = np.searchsorted(reference_positions, center + broad_radius)
        candidates.extend((reference_positions[left:right] - pos).tolist())
    candidates = np.unique(np.asarray(candidates, dtype=np.int64))
    candidates = candidates[
        (candidates >= previous_lag - broad_radius) &
        (candidates <= previous_lag + broad_radius)
    ]
    return score(candidates)


def fast_lifted_truth(bigwig, fasta, chrom, start, end, bin_size=20_000,
                      max_lag=60):
    """Load older-build truth by tracking exact CpG-coordinate alignment."""
    import pyBigWig

    pad = max(max_lag + 10, 10_100)
    fetch_start = max(0, start - pad)
    fasta_end = min(fasta.get_reference_length(chrom), end + pad)
    seq = fasta.fetch(chrom, fetch_start, fasta_end).upper()
    bases = np.frombuffer(seq.encode(), dtype="S1")
    ref = np.flatnonzero((bases[:-1] == b"C") & (bases[1:] == b"G")) + fetch_start
    reference_mask = np.zeros(fasta.get_reference_length(chrom), dtype=bool)
    reference_mask[ref] = True
    with pyBigWig.open(bigwig) as bw:
        if chrom not in bw.chroms():
            return {}, 0, 0
        truth_end = min(end + pad, bw.chroms(chrom))
        intervals = bw.intervals(chrom, fetch_start, truth_end) or []
    if not intervals:
        return {}, 0, 0
    truth_pos = np.asarray([x[0] for x in intervals], dtype=np.int64)
    truth_val = np.asarray([x[2] / 100.0 for x in intervals], dtype=float)
    out = {}
    matched = total = 0
    previous_lag = 0
    first_bin = (start // bin_size) * bin_size
    for left in range(first_bin, truth_end, bin_size):
        selected = (truth_pos >= left) & (truth_pos < left + bin_size)
        positions = truth_pos[selected]
        values = truth_val[selected]
        if len(positions) < 20:
            continue
        lag, best = _best_exact_lag(
            positions, reference_mask, ref, previous_lag,
        )
        previous_lag = lag
        shifted = positions + lag
        for pos, value in zip(shifted, values):
            if 0 <= pos < len(reference_mask) and reference_mask[pos] and start <= pos < end:
                out[int(pos)] = float(value)
        matched += best
        total += len(positions)
    return out, matched, total


def call_contig(bams, fasta, chrom, length, chunk_bp, window,
                factors, min_other, min_cpg, posterior_threshold, max_gap,
                output_dir, truth_bigwig=None):
    starts_parts, n_parts, emission_parts = [], [], []
    opened_bams = [pysam.AlignmentFile(path, "rb") for path in bams]
    try:
        for chunk_start in range(0, length, chunk_bp):
            chunk_end = min(length, chunk_start + chunk_bp)
            observations = []
            offset = 0
            for bam in opened_bams:
                part, offset = collect_bam_observations(
                    bam, fasta, chrom, chunk_start, chunk_end,
                    molecule_offset=offset,
                )
                observations.extend(part)
            windows = make_windows(
                observations, chunk_start, chunk_end, window_size=window,
                min_other=min_other, five_prime_factors=factors,
            )
            starts_parts.append(np.asarray([w.start for w in windows], dtype=np.int64))
            n_parts.append(np.asarray([w.n_cpg for w in windows], dtype=np.int32))
            emission_parts.append(np.asarray([
                [window_log_likelihood(w, BETA_UNMETH),
                 window_log_likelihood(w, BETA_METH)] for w in windows
            ]))
            print(f"  {chrom} {chunk_start:,}-{chunk_end:,}: "
                  f"{len(observations):,} obs, "
                  f"{sum(w.n_cpg >= min_cpg for w in windows):,} windows",
                  file=sys.stderr, flush=True)
    finally:
        for bam in opened_bams:
            bam.close()
    starts = np.concatenate(starts_parts)
    n_cpg = np.concatenate(n_parts)
    emission = np.concatenate(emission_parts)
    domains, posterior = call_domains_from_emissions(
        emission, starts, n_cpg, chrom, window_size=window,
        posterior_threshold=posterior_threshold, min_cpg=min_cpg,
        max_gap=max_gap,
    )
    domains = [M5CDomain(d.chrom, d.start, min(d.end, length),
                         d.methylated, d.posterior) for d in domains]
    np.savez_compressed(
        output_dir / f"{chrom}.m5c.npz", starts=starts, n_cpg=n_cpg,
        log_emission=emission, posterior=posterior,
    )
    with (output_dir / f"{chrom}.m5c.bed").open("w") as handle:
        write_bed(domains, handle)
    metrics = {"chrom": chrom, "length": length, "windows": len(starts),
               "informative": int((n_cpg >= min_cpg).sum()),
               "domains": len(domains)}
    metrics.update(validate_track(
        starts, n_cpg, posterior, fasta, chrom, length, window,
        min_cpg, truth_bigwig, posterior_threshold,
    ))
    return domains, metrics


def validate_track(starts, n_cpg, posterior, fasta, chrom, length, window,
                   min_cpg, truth_bigwig=None, posterior_threshold=0.99):
    metrics = {}
    if truth_bigwig:
        truth, matched, total = fast_lifted_truth(
            truth_bigwig, fasta, chrom, 0, length,
        )
        if total == 0 or not truth:
            raise ValueError(f"truth track has no usable records on {chrom}")
        if matched / total < 0.90:
            raise ValueError(
                f"older-build truth lift matched only {matched}/{total} "
                f"records on {chrom}; refusing misleading validation"
            )
        pos = np.asarray(sorted(truth), dtype=np.int64)
        beta = np.asarray([truth[int(p)] for p in pos])
        if len(beta) < 100 or np.ptp(beta) < 0.05:
            raise ValueError(
                f"truth track is sparse or degenerate on {chrom}: "
                f"n={len(beta)}, range={np.ptp(beta) if len(beta) else 0:.4g}"
            )
        window_index = np.minimum(pos // window, len(starts) - 1)
        sums = np.bincount(window_index, weights=beta, minlength=len(starts))
        counts = np.bincount(window_index, minlength=len(starts))
        truth_window = np.divide(sums, counts, out=np.full(len(starts), np.nan),
                                 where=counts > 0)
        good = (counts >= 3) & (n_cpg >= min_cpg)
        p_meth = posterior[:, 1]
        extreme = good & ((truth_window <= 0.1) | (truth_window >= 0.9))
        called_meth = good & (p_meth >= posterior_threshold)
        called_unmeth = good & (p_meth <= 1.0 - posterior_threshold)
        metrics.update({
            "truth_records": total, "truth_matched": matched,
            "truth_match_fraction": matched / max(total, 1),
            "truth_windows": int(good.sum()),
            "rho": float(spearmanr(p_meth[good], truth_window[good]).statistic),
            "auc_extreme": auc(truth_window[extreme] >= 0.9, p_meth[extreme]),
            "methylated_windows": int(called_meth.sum()),
            "methylated_truth_mean": float(np.mean(truth_window[called_meth])),
            "methylated_precision": float(np.mean(truth_window[called_meth] > 0.5)),
            "unmethylated_windows": int(called_unmeth.sum()),
            "unmethylated_truth_mean": float(np.mean(truth_window[called_unmeth])),
            "unmethylated_precision": float(np.mean(truth_window[called_unmeth] < 0.5)),
        })
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bam-glob", required=True,
                        help="Glob for input DddA FiberHMM BAMs")
    parser.add_argument("--reference", required=True,
                        help="Indexed diploid reference FASTA")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--contigs", default="",
                        help="Comma-separated contigs; default primary MATERNAL/PATERNAL")
    parser.add_argument("--chunk-bp", type=int, default=5_000_000)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--min-other", type=int, default=10)
    parser.add_argument("--min-cpg", type=int, default=10)
    parser.add_argument("--posterior", type=float, default=0.99)
    parser.add_argument("--max-gap", type=int, default=1000)
    parser.add_argument("--five-prime-factors", default="0.865,1.045,0.944,1.146")
    parser.add_argument("--truth-bigwig", default="")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Reuse per-contig NPZ/BED files and recompute validation")
    args = parser.parse_args()
    if args.chunk_bp % args.window:
        raise SystemExit("--chunk-bp must be a multiple of --window")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bams = sorted(glob.glob(args.bam_glob))
    factors = np.asarray([float(x) for x in args.five_prime_factors.split(",")])
    factors /= factors.mean()
    with pysam.FastaFile(args.reference) as fasta:
        if args.contigs:
            contigs = args.contigs.split(",")
        else:
            contigs = [c for c in fasta.references if PRIMARY_HAPLOTYPE.match(c)]
        metrics = []
        for index, chrom in enumerate(contigs, 1):
            print(f"[{index}/{len(contigs)}] {chrom}", file=sys.stderr, flush=True)
            length = fasta.get_reference_length(chrom)
            npz_path = output_dir / f"{chrom}.m5c.npz"
            bed_path = output_dir / f"{chrom}.m5c.bed"
            if args.skip_existing and npz_path.exists() and bed_path.exists():
                data = np.load(npz_path)
                starts, n_cpg, posterior = data["starts"], data["n_cpg"], data["posterior"]
                result = {
                    "chrom": chrom, "length": length, "windows": len(starts),
                    "informative": int((n_cpg >= args.min_cpg).sum()),
                    "domains": sum(1 for line in bed_path.open() if line.strip()),
                }
                result.update(validate_track(
                    starts, n_cpg, posterior, fasta, chrom, length,
                    args.window, args.min_cpg, args.truth_bigwig or None,
                    args.posterior,
                ))
            else:
                _domains, result = call_contig(
                    bams, fasta, chrom, length,
                    args.chunk_bp, args.window, factors, args.min_other,
                    args.min_cpg, args.posterior, args.max_gap, output_dir,
                    args.truth_bigwig or None,
                )
            metrics.append(result)
            print("  " + " ".join(f"{k}={v}" for k, v in result.items()),
                  file=sys.stderr, flush=True)
    with (output_dir / "genome.m5c.bed").open("w") as handle:
        for chrom in contigs:
            with (output_dir / f"{chrom}.m5c.bed").open() as source:
                for line in source:
                    handle.write(line)
    keys = sorted({key for row in metrics for key in row})
    with (output_dir / "metrics.tsv").open("w") as handle:
        handle.write("\t".join(keys) + "\n")
        for row in metrics:
            handle.write("\t".join(str(row.get(key, "")) for key in keys) + "\n")


if __name__ == "__main__":
    main()
