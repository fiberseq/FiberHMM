#!/usr/bin/env python3
"""
Plot log10 deamination rates (C>T and G>A) for a BAM file.
"""

import pysam
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import argparse


def count_deaminations(bam_path, max_reads=None):
    """
    Count C>T and G>A deaminations per read.

    Returns:
        ct_rates: list of C>T deamination rates per read
        ga_rates: list of G>A deamination rates per read
    """
    ct_rates = []
    ga_rates = []

    bam = pysam.AlignmentFile(bam_path, "rb")

    read_count = 0
    for read in tqdm(bam.fetch(), desc="Processing reads"):
        if read.is_secondary or read.is_supplementary:
            continue
        if read.is_unmapped:
            continue

        seq = read.query_sequence
        if seq is None:
            continue

        # Get aligned pairs with reference sequence
        pair = read.get_aligned_pairs(matches_only=False, with_seq=True)

        # Count bases and deaminations
        c_total = 0
        g_total = 0
        ct_count = 0
        ga_count = 0

        for pos in pair:
            if pos[0] is None or pos[1] is None or pos[2] is None:
                continue  # indel

            qi = pos[0]
            ref_base = pos[2].upper()
            query_base = seq[qi].upper()

            # Count reference C and G
            if ref_base == 'C':
                c_total += 1
                if query_base == 'T':
                    ct_count += 1
            elif ref_base == 'G':
                g_total += 1
                if query_base == 'A':
                    ga_count += 1

        # Calculate rates
        if c_total > 0:
            ct_rates.append(ct_count / c_total)
        if g_total > 0:
            ga_rates.append(ga_count / g_total)

        read_count += 1
        if max_reads and read_count >= max_reads:
            break

    bam.close()
    return ct_rates, ga_rates


def plot_deamination_rates(ct_rates, ga_rates, output_path=None, title="Deamination Rates"):
    """
    Plot log10 histograms of deamination rates.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Convert to log10 (add small epsilon to avoid log(0))
    epsilon = 1e-10
    ct_log = np.log10(np.array(ct_rates) + epsilon)
    ga_log = np.log10(np.array(ga_rates) + epsilon)

    # Filter out zero rates for better visualization
    ct_nonzero = [r for r in ct_rates if r > 0]
    ga_nonzero = [r for r in ga_rates if r > 0]
    ct_log_nonzero = np.log10(np.array(ct_nonzero)) if ct_nonzero else []
    ga_log_nonzero = np.log10(np.array(ga_nonzero)) if ga_nonzero else []

    # C>T plot
    ax1 = axes[0]
    if len(ct_log_nonzero) > 0:
        ax1.hist(ct_log_nonzero, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('log10(C>T rate)')
    ax1.set_ylabel('Number of reads')
    ax1.set_title(f'C>T Deamination Rate\n(n={len(ct_rates):,}, non-zero={len(ct_nonzero):,})')
    ax1.axvline(np.median(ct_log_nonzero) if len(ct_log_nonzero) > 0 else 0,
                color='red', linestyle='--', label=f'Median: {np.median(ct_log_nonzero):.2f}' if len(ct_log_nonzero) > 0 else 'N/A')
    ax1.legend()

    # G>A plot
    ax2 = axes[1]
    if len(ga_log_nonzero) > 0:
        ax2.hist(ga_log_nonzero, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('log10(G>A rate)')
    ax2.set_ylabel('Number of reads')
    ax2.set_title(f'G>A Deamination Rate\n(n={len(ga_rates):,}, non-zero={len(ga_nonzero):,})')
    ax2.axvline(np.median(ga_log_nonzero) if len(ga_log_nonzero) > 0 else 0,
                color='red', linestyle='--', label=f'Median: {np.median(ga_log_nonzero):.2f}' if len(ga_log_nonzero) > 0 else 'N/A')
    ax2.legend()

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")

    plt.show()

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"\nC>T Deamination:")
    print(f"  Total reads: {len(ct_rates):,}")
    print(f"  Reads with deamination: {len(ct_nonzero):,} ({100*len(ct_nonzero)/len(ct_rates):.1f}%)")
    if ct_nonzero:
        print(f"  Mean rate (non-zero): {np.mean(ct_nonzero):.4f}")
        print(f"  Median rate (non-zero): {np.median(ct_nonzero):.4f}")
        print(f"  Mean log10 rate: {np.mean(ct_log_nonzero):.2f}")
        print(f"  Median log10 rate: {np.median(ct_log_nonzero):.2f}")

    print(f"\nG>A Deamination:")
    print(f"  Total reads: {len(ga_rates):,}")
    print(f"  Reads with deamination: {len(ga_nonzero):,} ({100*len(ga_nonzero)/len(ga_rates):.1f}%)")
    if ga_nonzero:
        print(f"  Mean rate (non-zero): {np.mean(ga_nonzero):.4f}")
        print(f"  Median rate (non-zero): {np.median(ga_nonzero):.4f}")
        print(f"  Mean log10 rate: {np.mean(ga_log_nonzero):.2f}")
        print(f"  Median log10 rate: {np.median(ga_log_nonzero):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot deamination rates from BAM file")
    parser.add_argument("-i", "--input", required=True, help="Input BAM file")
    parser.add_argument("-o", "--output", help="Output plot file (PNG)")
    parser.add_argument("-n", "--max-reads", type=int, help="Maximum reads to process")
    parser.add_argument("-t", "--title", default="Deamination Rates", help="Plot title")
    args = parser.parse_args()

    ct_rates, ga_rates = count_deaminations(args.input, args.max_reads)
    plot_deamination_rates(ct_rates, ga_rates, args.output, args.title)
