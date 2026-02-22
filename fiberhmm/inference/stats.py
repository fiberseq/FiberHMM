"""FiberHMM footprint statistics and QC plotting."""

import os
import sys
import numpy as np
import pysam
from typing import Optional


class FootprintStats:
    """Collects footprint statistics from sampled reads."""

    def __init__(self):
        self.footprint_sizes = []
        self.gap_sizes = []  # gaps between footprints (accessible regions)
        self.footprints_per_read = []
        self.msps_per_read = []
        self.msp_sizes = []
        self.footprint_scores = []
        self.msp_scores = []
        self.read_lengths = []
        self.footprint_coverage = []  # fraction of read covered by footprints
        self.methylation_in_footprints = []  # methylation density in footprints
        self.methylation_in_accessible = []  # methylation density in accessible
        self.total_reads_sampled = 0
        self.reads_with_footprints = 0

    def add_read(self, read_length: int, ns: np.ndarray, nl: np.ndarray,
                 as_starts: np.ndarray = None, al_lengths: np.ndarray = None,
                 ns_scores: np.ndarray = None, as_scores: np.ndarray = None,
                 meth_positions: np.ndarray = None):
        """Add statistics from a single read."""
        self.total_reads_sampled += 1
        self.read_lengths.append(read_length)

        n_footprints = len(ns) if ns is not None else 0
        self.footprints_per_read.append(n_footprints)

        if n_footprints > 0:
            self.reads_with_footprints += 1

            # Footprint sizes
            self.footprint_sizes.extend(nl.tolist())

            # Footprint scores
            if ns_scores is not None:
                self.footprint_scores.extend(ns_scores.tolist())

            # Footprint coverage
            total_fp_bases = np.sum(nl)
            self.footprint_coverage.append(total_fp_bases / read_length if read_length > 0 else 0)

            # Gap sizes (accessible regions between footprints)
            if n_footprints > 1:
                # Sort by start position
                sorted_idx = np.argsort(ns)
                sorted_starts = ns[sorted_idx]
                sorted_lengths = nl[sorted_idx]

                for i in range(len(sorted_starts) - 1):
                    gap_start = sorted_starts[i] + sorted_lengths[i]
                    gap_end = sorted_starts[i + 1]
                    gap = gap_end - gap_start
                    if gap > 0:
                        self.gap_sizes.append(gap)

        # MSP stats
        n_msps = len(as_starts) if as_starts is not None else 0
        self.msps_per_read.append(n_msps)

        if n_msps > 0:
            self.msp_sizes.extend(al_lengths.tolist())
            if as_scores is not None:
                self.msp_scores.extend(as_scores.tolist())

    def get_summary(self) -> dict:
        """Generate summary statistics."""
        summary = {
            'total_reads_sampled': self.total_reads_sampled,
            'reads_with_footprints': self.reads_with_footprints,
            'pct_reads_with_footprints': 100 * self.reads_with_footprints / self.total_reads_sampled if self.total_reads_sampled > 0 else 0,
        }

        if self.read_lengths:
            summary['read_length_median'] = np.median(self.read_lengths)
            summary['read_length_mean'] = np.mean(self.read_lengths)
            summary['read_length_std'] = np.std(self.read_lengths)

        if self.footprint_sizes:
            summary['total_footprints'] = len(self.footprint_sizes)
            summary['footprint_size_median'] = np.median(self.footprint_sizes)
            summary['footprint_size_mean'] = np.mean(self.footprint_sizes)
            summary['footprint_size_std'] = np.std(self.footprint_sizes)
            summary['footprint_size_min'] = np.min(self.footprint_sizes)
            summary['footprint_size_max'] = np.max(self.footprint_sizes)
            summary['footprint_size_q25'] = np.percentile(self.footprint_sizes, 25)
            summary['footprint_size_q75'] = np.percentile(self.footprint_sizes, 75)

        if self.footprints_per_read:
            fp_per_read = [x for x in self.footprints_per_read if x > 0]
            if fp_per_read:
                summary['footprints_per_read_median'] = np.median(fp_per_read)
                summary['footprints_per_read_mean'] = np.mean(fp_per_read)

        if self.gap_sizes:
            summary['total_gaps'] = len(self.gap_sizes)
            summary['gap_size_median'] = np.median(self.gap_sizes)
            summary['gap_size_mean'] = np.mean(self.gap_sizes)
            summary['gap_size_std'] = np.std(self.gap_sizes)

        if self.msp_sizes:
            summary['total_msps'] = len(self.msp_sizes)
            summary['msp_size_median'] = np.median(self.msp_sizes)
            summary['msp_size_mean'] = np.mean(self.msp_sizes)

        if self.footprint_coverage:
            summary['footprint_coverage_median'] = np.median(self.footprint_coverage)
            summary['footprint_coverage_mean'] = np.mean(self.footprint_coverage)

        if self.footprint_scores:
            summary['footprint_score_median'] = np.median(self.footprint_scores)
            summary['footprint_score_mean'] = np.mean(self.footprint_scores)
            summary['footprint_score_std'] = np.std(self.footprint_scores)

        return summary

    def write_summary(self, filepath: str):
        """Write summary statistics to a text file."""
        summary = self.get_summary()

        with open(filepath, 'w') as f:
            f.write("FiberHMM Footprint Statistics\n")
            f.write("=" * 50 + "\n\n")

            f.write("Read Statistics\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total reads sampled:        {summary.get('total_reads_sampled', 0):,}\n")
            f.write(f"Reads with footprints:      {summary.get('reads_with_footprints', 0):,} ({summary.get('pct_reads_with_footprints', 0):.1f}%)\n")
            if 'read_length_median' in summary:
                f.write(f"Read length (median):       {summary['read_length_median']:.0f} bp\n")
                f.write(f"Read length (mean ± std):   {summary['read_length_mean']:.0f} ± {summary['read_length_std']:.0f} bp\n")
            f.write("\n")

            f.write("Footprint Statistics\n")
            f.write("-" * 30 + "\n")
            if 'total_footprints' in summary:
                f.write(f"Total footprints:           {summary['total_footprints']:,}\n")
                f.write(f"Size (median):              {summary['footprint_size_median']:.0f} bp\n")
                f.write(f"Size (mean ± std):          {summary['footprint_size_mean']:.1f} ± {summary['footprint_size_std']:.1f} bp\n")
                f.write(f"Size (range):               {summary['footprint_size_min']:.0f} - {summary['footprint_size_max']:.0f} bp\n")
                f.write(f"Size (IQR):                 {summary['footprint_size_q25']:.0f} - {summary['footprint_size_q75']:.0f} bp\n")
            if 'footprints_per_read_median' in summary:
                f.write(f"Per read (median):          {summary['footprints_per_read_median']:.1f}\n")
                f.write(f"Per read (mean):            {summary['footprints_per_read_mean']:.1f}\n")
            if 'footprint_coverage_median' in summary:
                f.write(f"Read coverage (median):     {summary['footprint_coverage_median']*100:.1f}%\n")
            f.write("\n")

            if 'total_gaps' in summary:
                f.write("Gap (Accessible Region) Statistics\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total gaps:                 {summary['total_gaps']:,}\n")
                f.write(f"Gap size (median):          {summary['gap_size_median']:.0f} bp\n")
                f.write(f"Gap size (mean ± std):      {summary['gap_size_mean']:.1f} ± {summary['gap_size_std']:.1f} bp\n")
                f.write("\n")

            if 'total_msps' in summary:
                f.write("MSP (Large Accessible) Statistics\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total MSPs:                 {summary['total_msps']:,}\n")
                f.write(f"MSP size (median):          {summary['msp_size_median']:.0f} bp\n")
                f.write(f"MSP size (mean):            {summary['msp_size_mean']:.1f} bp\n")
                f.write("\n")

            if 'footprint_score_median' in summary:
                f.write("Footprint Quality Scores\n")
                f.write("-" * 30 + "\n")
                f.write(f"Score (median):             {summary['footprint_score_median']:.3f}\n")
                f.write(f"Score (mean ± std):         {summary['footprint_score_mean']:.3f} ± {summary['footprint_score_std']:.3f}\n")

    def plot_distributions(self, output_prefix: str):
        """Generate distribution plots."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
        except ImportError:
            print("Warning: matplotlib not installed. Skipping plots.")
            print("Install with: pip install matplotlib")
            return

        # Create multi-page PDF
        pdf_path = f"{output_prefix}_stats.pdf"

        with PdfPages(pdf_path) as pdf:
            # Page 1: Footprint size distribution
            if self.footprint_sizes:
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                fig.suptitle('FiberHMM Footprint Statistics', fontsize=14, fontweight='bold')

                # Footprint size histogram
                ax = axes[0, 0]
                sizes = np.array(self.footprint_sizes)
                ax.hist(sizes, bins=50, range=(0, min(500, np.percentile(sizes, 99))),
                       color='steelblue', edgecolor='white', alpha=0.8)
                ax.axvline(np.median(sizes), color='red', linestyle='--',
                          label=f'Median: {np.median(sizes):.0f} bp')
                ax.set_xlabel('Footprint Size (bp)')
                ax.set_ylabel('Count')
                ax.set_title('Footprint Size Distribution')
                ax.legend()

                # Gap size histogram
                ax = axes[0, 1]
                if self.gap_sizes:
                    gaps = np.array(self.gap_sizes)
                    ax.hist(gaps, bins=50, range=(0, min(500, np.percentile(gaps, 99))),
                           color='coral', edgecolor='white', alpha=0.8)
                    ax.axvline(np.median(gaps), color='red', linestyle='--',
                              label=f'Median: {np.median(gaps):.0f} bp')
                    ax.set_xlabel('Gap Size (bp)')
                    ax.set_ylabel('Count')
                    ax.set_title('Gap (Accessible) Size Distribution')
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, 'No gap data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Gap Size Distribution')

                # Footprints per read
                ax = axes[1, 0]
                fp_per_read = [x for x in self.footprints_per_read if x > 0]
                if fp_per_read:
                    ax.hist(fp_per_read, bins=range(0, min(50, max(fp_per_read)+2)),
                           color='forestgreen', edgecolor='white', alpha=0.8)
                    ax.axvline(np.median(fp_per_read), color='red', linestyle='--',
                              label=f'Median: {np.median(fp_per_read):.1f}')
                    ax.set_xlabel('Footprints per Read')
                    ax.set_ylabel('Count')
                    ax.set_title('Footprints per Read')
                    ax.legend()

                # Footprint coverage
                ax = axes[1, 1]
                if self.footprint_coverage:
                    coverage = np.array(self.footprint_coverage) * 100
                    ax.hist(coverage, bins=50, range=(0, 100),
                           color='purple', edgecolor='white', alpha=0.8)
                    ax.axvline(np.median(coverage), color='red', linestyle='--',
                              label=f'Median: {np.median(coverage):.1f}%')
                    ax.set_xlabel('Footprint Coverage (%)')
                    ax.set_ylabel('Count')
                    ax.set_title('Read Coverage by Footprints')
                    ax.legend()

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            # Page 2: Quality scores and MSPs
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle('FiberHMM Quality and MSP Statistics', fontsize=14, fontweight='bold')

            # Footprint score distribution
            ax = axes[0, 0]
            if self.footprint_scores:
                scores = np.array(self.footprint_scores)
                ax.hist(scores, bins=50, range=(0, 1),
                       color='gold', edgecolor='white', alpha=0.8)
                ax.axvline(np.median(scores), color='red', linestyle='--',
                          label=f'Median: {np.median(scores):.3f}')
                ax.set_xlabel('Footprint Confidence Score')
                ax.set_ylabel('Count')
                ax.set_title('Footprint Quality Distribution')
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No score data\n(use --scores flag)',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Footprint Quality Distribution')

            # MSP size distribution
            ax = axes[0, 1]
            if self.msp_sizes:
                msp_sizes = np.array(self.msp_sizes)
                ax.hist(msp_sizes, bins=50, range=(0, min(2000, np.percentile(msp_sizes, 99))),
                       color='teal', edgecolor='white', alpha=0.8)
                ax.axvline(np.median(msp_sizes), color='red', linestyle='--',
                          label=f'Median: {np.median(msp_sizes):.0f} bp')
                ax.set_xlabel('MSP Size (bp)')
                ax.set_ylabel('Count')
                ax.set_title('MSP Size Distribution')
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No MSP data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('MSP Size Distribution')

            # Read length distribution
            ax = axes[1, 0]
            if self.read_lengths:
                lengths = np.array(self.read_lengths)
                ax.hist(lengths, bins=50,
                       color='slategray', edgecolor='white', alpha=0.8)
                ax.axvline(np.median(lengths), color='red', linestyle='--',
                          label=f'Median: {np.median(lengths):.0f} bp')
                ax.set_xlabel('Read Length (bp)')
                ax.set_ylabel('Count')
                ax.set_title('Read Length Distribution')
                ax.legend()

            # Footprint size vs count (2D histogram)
            ax = axes[1, 1]
            if len(self.footprint_sizes) > 100:
                # Create size bins
                sizes = np.array(self.footprint_sizes)
                bins = [0, 20, 40, 60, 80, 100, 150, 200, 300, 500, 10000]
                labels = ['0-20', '20-40', '40-60', '60-80', '80-100',
                         '100-150', '150-200', '200-300', '300-500', '500+']
                counts, _ = np.histogram(sizes, bins=bins)

                ax.barh(range(len(labels)), counts, color='steelblue', alpha=0.8)
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels)
                ax.set_xlabel('Count')
                ax.set_ylabel('Size Range (bp)')
                ax.set_title('Footprint Size Bins')
            else:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Footprint Size Bins')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        print(f"QC plots saved to: {pdf_path}")

        # Also save individual PNGs for convenience
        if self.footprint_sizes:
            fig, ax = plt.subplots(figsize=(8, 5))
            sizes = np.array(self.footprint_sizes)
            ax.hist(sizes, bins=100, range=(0, min(500, np.percentile(sizes, 99))),
                   color='steelblue', edgecolor='white', alpha=0.8)
            ax.axvline(np.median(sizes), color='red', linestyle='--', linewidth=2,
                      label=f'Median: {np.median(sizes):.0f} bp')
            ax.set_xlabel('Footprint Size (bp)', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Footprint Size Distribution', fontsize=14)
            ax.legend(fontsize=11)
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_footprint_sizes.png", dpi=150)
            plt.close(fig)


def collect_stats_from_bam(bam_path: str, n_samples: int = 10000,
                           seed: int = 42, with_scores: bool = False) -> FootprintStats:
    """
    Collect statistics from a processed BAM file with footprint tags.

    Uses reservoir sampling to get a uniform sample of reads.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    stats = FootprintStats()

    # First pass: count reads
    bam = pysam.AlignmentFile(bam_path, "rb")
    total_reads = 0
    for read in bam:
        if not read.is_unmapped and not read.is_secondary and not read.is_supplementary:
            total_reads += 1
    bam.close()

    # Calculate sampling probability
    if total_reads <= n_samples:
        sample_prob = 1.0
    else:
        sample_prob = n_samples / total_reads

    print(f"  Sampling ~{min(n_samples, total_reads):,} reads from {total_reads:,} total ({sample_prob*100:.1f}%)")

    # Second pass: collect stats from sampled reads
    bam = pysam.AlignmentFile(bam_path, "rb")
    sampled = 0

    for read in bam:
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue

        # Reservoir sampling
        if random.random() > sample_prob:
            continue

        if sampled >= n_samples:
            break

        # Get footprint tags
        try:
            ns = np.array(read.get_tag('ns'))
            nl = np.array(read.get_tag('nl'))
        except KeyError:
            ns = np.array([])
            nl = np.array([])

        # Get MSP tags
        try:
            as_starts = np.array(read.get_tag('as'))
            al_lengths = np.array(read.get_tag('al'))
        except KeyError:
            as_starts = None
            al_lengths = None

        # Get scores
        ns_scores = None
        as_scores = None
        if with_scores:
            try:
                nq = np.array(read.get_tag('nq'))
                ns_scores = nq / 255.0  # Scale back to 0-1
            except KeyError:
                pass
            try:
                aq = np.array(read.get_tag('aq'))
                as_scores = aq / 255.0
            except KeyError:
                pass

        read_length = read.query_length or 0
        stats.add_read(read_length, ns, nl, as_starts, al_lengths, ns_scores, as_scores)
        sampled += 1

    bam.close()
    return stats
