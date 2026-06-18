"""FiberHMM footprint statistics and QC plotting."""


from dataclasses import dataclass

import numpy as np
import pysam

from fiberhmm.inference.read_filters import is_primary_mapped_alignment
from fiberhmm.io.ma_tags import flip_intervals_to_seq

_FOOTPRINT_SIZE_BINS = [0, 20, 40, 60, 80, 100, 150, 200, 300, 500, 10000]
_FOOTPRINT_SIZE_BIN_LABELS = [
    '0-20', '20-40', '40-60', '60-80', '80-100',
    '100-150', '150-200', '200-300', '300-500', '500+',
]


@dataclass(frozen=True)
class _IntervalTagArrays:
    starts: object
    lengths: object


@dataclass(frozen=True)
class _StatsReadSignalArrays:
    nuc_starts: np.ndarray
    nuc_lengths: np.ndarray
    msp_starts: object
    msp_lengths: object
    nuc_scores: object
    msp_scores: object


def _positive_gaps_between_intervals(starts: np.ndarray, lengths: np.ndarray) -> list:
    if len(starts) <= 1:
        return []

    sorted_idx = np.argsort(starts)
    sorted_starts = starts[sorted_idx]
    sorted_lengths = lengths[sorted_idx]

    gaps = []
    for i in range(len(sorted_starts) - 1):
        gap_start = sorted_starts[i] + sorted_lengths[i]
        gap_end = sorted_starts[i + 1]
        gap = gap_end - gap_start
        if gap > 0:
            gaps.append(gap)
    return gaps


def _flipped_interval_tag_arrays(read, start_tag: str, length_tag: str, default):
    try:
        starts, lengths = flip_intervals_to_seq(
            read.get_tag(start_tag), read.get_tag(length_tag), read,
        )
        return _IntervalTagArrays(np.array(starts), np.array(lengths))
    except KeyError:
        return _IntervalTagArrays(default[0], default[1])


def _scaled_score_tag(read, tag: str):
    try:
        return np.array(read.get_tag(tag)) / 255.0
    except KeyError:
        return None


def _footprint_size_bin_counts(sizes) -> tuple:
    counts, _ = np.histogram(np.array(sizes), bins=_FOOTPRINT_SIZE_BINS)
    return _FOOTPRINT_SIZE_BIN_LABELS, counts


def _positive_counts(values) -> list:
    return [x for x in values if x > 0]


def _has_values(values) -> bool:
    return values is not None and len(values) > 0


def _footprint_coverage_fraction(total_fp_bases, read_length: int) -> float:
    return total_fp_bases / read_length if read_length > 0 else 0


def _initial_footprint_summary(
    total_reads_sampled: int,
    reads_with_footprints: int,
) -> dict:
    pct_reads_with_footprints = (
        100 * reads_with_footprints / total_reads_sampled
        if total_reads_sampled > 0
        else 0
    )
    return {
        'total_reads_sampled': total_reads_sampled,
        'reads_with_footprints': reads_with_footprints,
        'pct_reads_with_footprints': pct_reads_with_footprints,
    }


def _add_numeric_summary(
    summary: dict,
    prefix: str,
    values,
    *,
    total_key: str = None,
    include_std: bool = False,
    include_minmax: bool = False,
    include_iqr: bool = False,
) -> None:
    if len(values) == 0:
        return
    if total_key:
        summary[total_key] = len(values)
    summary[f'{prefix}_median'] = np.median(values)
    summary[f'{prefix}_mean'] = np.mean(values)
    if include_std:
        summary[f'{prefix}_std'] = np.std(values)
    if include_minmax:
        summary[f'{prefix}_min'] = np.min(values)
        summary[f'{prefix}_max'] = np.max(values)
    if include_iqr:
        summary[f'{prefix}_q25'] = np.percentile(values, 25)
        summary[f'{prefix}_q75'] = np.percentile(values, 75)


def _add_positive_count_summary(summary: dict, prefix: str, values) -> None:
    positive_values = _positive_counts(values)
    if not positive_values:
        return
    summary[f'{prefix}_median'] = np.median(positive_values)
    summary[f'{prefix}_mean'] = np.mean(positive_values)


def _write_read_stats_section(handle, summary: dict) -> None:
    handle.write("Read Statistics\n")
    handle.write("-" * 30 + "\n")
    handle.write(
        f"Total reads sampled:        "
        f"{summary.get('total_reads_sampled', 0):,}\n"
    )
    handle.write(
        f"Reads with footprints:      "
        f"{summary.get('reads_with_footprints', 0):,} "
        f"({summary.get('pct_reads_with_footprints', 0):.1f}%)\n"
    )
    if 'read_length_median' in summary:
        handle.write(
            f"Read length (median):       "
            f"{summary['read_length_median']:.0f} bp\n"
        )
        handle.write(
            f"Read length (mean ± std):   "
            f"{summary['read_length_mean']:.0f} ± "
            f"{summary['read_length_std']:.0f} bp\n"
        )
    handle.write("\n")


def _write_footprint_stats_section(handle, summary: dict) -> None:
    handle.write("Footprint Statistics\n")
    handle.write("-" * 30 + "\n")
    if 'total_footprints' in summary:
        handle.write(f"Total footprints:           {summary['total_footprints']:,}\n")
        handle.write(
            f"Size (median):              "
            f"{summary['footprint_size_median']:.0f} bp\n"
        )
        handle.write(
            f"Size (mean ± std):          "
            f"{summary['footprint_size_mean']:.1f} ± "
            f"{summary['footprint_size_std']:.1f} bp\n"
        )
        handle.write(
            f"Size (range):               "
            f"{summary['footprint_size_min']:.0f} - "
            f"{summary['footprint_size_max']:.0f} bp\n"
        )
        handle.write(
            f"Size (IQR):                 "
            f"{summary['footprint_size_q25']:.0f} - "
            f"{summary['footprint_size_q75']:.0f} bp\n"
        )
    if 'footprints_per_read_median' in summary:
        handle.write(
            f"Per read (median):          "
            f"{summary['footprints_per_read_median']:.1f}\n"
        )
        handle.write(
            f"Per read (mean):            "
            f"{summary['footprints_per_read_mean']:.1f}\n"
        )
    if 'footprint_coverage_median' in summary:
        handle.write(
            f"Read coverage (median):     "
            f"{summary['footprint_coverage_median']*100:.1f}%\n"
        )
    handle.write("\n")


def _write_gap_stats_section(handle, summary: dict) -> None:
    if 'total_gaps' not in summary:
        return

    handle.write("Gap (Accessible Region) Statistics\n")
    handle.write("-" * 30 + "\n")
    handle.write(f"Total gaps:                 {summary['total_gaps']:,}\n")
    handle.write(f"Gap size (median):          {summary['gap_size_median']:.0f} bp\n")
    handle.write(
        f"Gap size (mean ± std):      "
        f"{summary['gap_size_mean']:.1f} ± {summary['gap_size_std']:.1f} bp\n"
    )
    handle.write("\n")


def _write_msp_stats_section(handle, summary: dict) -> None:
    if 'total_msps' not in summary:
        return

    handle.write("MSP (Large Accessible) Statistics\n")
    handle.write("-" * 30 + "\n")
    handle.write(f"Total MSPs:                 {summary['total_msps']:,}\n")
    handle.write(f"MSP size (median):          {summary['msp_size_median']:.0f} bp\n")
    handle.write(f"MSP size (mean):            {summary['msp_size_mean']:.1f} bp\n")
    handle.write("\n")


def _write_footprint_quality_section(handle, summary: dict) -> None:
    if 'footprint_score_median' not in summary:
        return

    handle.write("Footprint Quality Scores\n")
    handle.write("-" * 30 + "\n")
    handle.write(
        f"Score (median):             "
        f"{summary['footprint_score_median']:.3f}\n"
    )
    handle.write(
        f"Score (mean ± std):         "
        f"{summary['footprint_score_mean']:.3f} ± "
        f"{summary['footprint_score_std']:.3f}\n"
    )


def _plot_median_histogram(
    ax,
    values,
    *,
    bins,
    hist_range=None,
    color: str,
    xlabel: str,
    title: str,
    median_format: str,
    median_suffix: str = '',
) -> None:
    values = np.array(values)
    hist_kwargs = {
        'bins': bins,
        'color': color,
        'edgecolor': 'white',
        'alpha': 0.8,
    }
    if hist_range is not None:
        hist_kwargs['range'] = hist_range

    median = np.median(values)
    ax.hist(values, **hist_kwargs)
    ax.axvline(
        median,
        color='red',
        linestyle='--',
        label=f'Median: {median:{median_format}}{median_suffix}',
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()


def _plot_no_data_message(ax, message: str, title: str) -> None:
    ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes)
    ax.set_title(title)


def _plot_footprint_size_bins(ax, footprint_sizes) -> None:
    if len(footprint_sizes) <= 100:
        _plot_no_data_message(ax, 'Insufficient data', 'Footprint Size Bins')
        return

    labels, counts = _footprint_size_bin_counts(footprint_sizes)
    ax.barh(range(len(labels)), counts, color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Count')
    ax.set_ylabel('Size Range (bp)')
    ax.set_title('Footprint Size Bins')


def _plot_footprint_size_png_axis(ax, footprint_sizes) -> None:
    sizes = np.array(footprint_sizes)
    ax.hist(
        sizes,
        bins=100,
        range=(0, min(500, np.percentile(sizes, 99))),
        color='steelblue',
        edgecolor='white',
        alpha=0.8,
    )
    ax.axvline(
        np.median(sizes),
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Median: {np.median(sizes):.0f} bp',
    )
    ax.set_xlabel('Footprint Size (bp)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Footprint Size Distribution', fontsize=14)
    ax.legend(fontsize=11)


def _stats_sampling_probability(total_reads: int, n_samples: int) -> float:
    if total_reads <= n_samples:
        return 1.0
    return n_samples / total_reads


def _count_primary_mapped_reads(bam_path: str) -> int:
    total_reads = 0
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for read in bam:
            if is_primary_mapped_alignment(read):
                total_reads += 1
    return total_reads


def _stats_read_signal_arrays(read, with_scores: bool) -> _StatsReadSignalArrays:
    nuc_tags = _flipped_interval_tag_arrays(
        read, 'ns', 'nl', (np.array([]), np.array([])),
    )
    msp_tags = _flipped_interval_tag_arrays(
        read, 'as', 'al', (None, None),
    )

    ns_scores = None
    as_scores = None
    if with_scores:
        ns_scores = _scaled_score_tag(read, 'nq')
        as_scores = _scaled_score_tag(read, 'aq')

    return _StatsReadSignalArrays(
        nuc_starts=nuc_tags.starts,
        nuc_lengths=nuc_tags.lengths,
        msp_starts=msp_tags.starts,
        msp_lengths=msp_tags.lengths,
        nuc_scores=ns_scores,
        msp_scores=as_scores,
    )


def _add_read_to_footprint_stats(
    stats: "FootprintStats",
    read,
    with_scores: bool,
) -> None:
    signals = _stats_read_signal_arrays(read, with_scores)

    read_length = read.query_length or 0
    stats.add_read(
        read_length,
        signals.nuc_starts,
        signals.nuc_lengths,
        signals.msp_starts,
        signals.msp_lengths,
        signals.nuc_scores,
        signals.msp_scores,
    )


def _plot_footprint_size_pdf_panel(ax, footprint_sizes) -> None:
    sizes = np.array(footprint_sizes)
    _plot_median_histogram(
        ax,
        sizes,
        bins=50,
        hist_range=(0, min(500, np.percentile(sizes, 99))),
        color='steelblue',
        xlabel='Footprint Size (bp)',
        title='Footprint Size Distribution',
        median_format='.0f',
        median_suffix=' bp',
    )


def _plot_gap_size_pdf_panel(ax, gap_sizes) -> None:
    if _has_values(gap_sizes):
        gaps = np.array(gap_sizes)
        _plot_median_histogram(
            ax,
            gaps,
            bins=50,
            hist_range=(0, min(500, np.percentile(gaps, 99))),
            color='coral',
            xlabel='Gap Size (bp)',
            title='Gap (Accessible) Size Distribution',
            median_format='.0f',
            median_suffix=' bp',
        )
    else:
        _plot_no_data_message(ax, 'No gap data', 'Gap Size Distribution')


def _plot_footprints_per_read_pdf_panel(ax, footprints_per_read) -> None:
    fp_per_read = _positive_counts(footprints_per_read)
    if fp_per_read:
        _plot_median_histogram(
            ax,
            fp_per_read,
            bins=range(0, min(50, max(fp_per_read)+2)),
            color='forestgreen',
            xlabel='Footprints per Read',
            title='Footprints per Read',
            median_format='.1f',
        )


def _plot_footprint_coverage_pdf_panel(ax, footprint_coverage) -> None:
    if _has_values(footprint_coverage):
        coverage = np.array(footprint_coverage) * 100
        _plot_median_histogram(
            ax,
            coverage,
            bins=50,
            hist_range=(0, 100),
            color='purple',
            xlabel='Footprint Coverage (%)',
            title='Read Coverage by Footprints',
            median_format='.1f',
            median_suffix='%',
        )


def _plot_footprint_overview_pdf_page(stats: "FootprintStats", plt, pdf) -> None:
    if not _has_values(stats.footprint_sizes):
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    try:
        fig.suptitle(
            'FiberHMM Footprint Statistics', fontsize=14, fontweight='bold',
        )

        _plot_footprint_size_pdf_panel(axes[0, 0], stats.footprint_sizes)
        _plot_gap_size_pdf_panel(axes[0, 1], stats.gap_sizes)
        _plot_footprints_per_read_pdf_panel(
            axes[1, 0], stats.footprints_per_read,
        )
        _plot_footprint_coverage_pdf_panel(
            axes[1, 1], stats.footprint_coverage,
        )

        plt.tight_layout()
        pdf.savefig(fig)
    finally:
        plt.close(fig)


def _plot_footprint_quality_pdf_panel(ax, footprint_scores) -> None:
    if _has_values(footprint_scores):
        scores = np.array(footprint_scores)
        _plot_median_histogram(
            ax,
            scores,
            bins=50,
            hist_range=(0, 1),
            color='gold',
            xlabel='Footprint Confidence Score',
            title='Footprint Quality Distribution',
            median_format='.3f',
        )
    else:
        _plot_no_data_message(
            ax,
            'No score data\n(use --scores flag)',
            'Footprint Quality Distribution',
        )


def _plot_msp_size_pdf_panel(ax, msp_size_values) -> None:
    if _has_values(msp_size_values):
        msp_sizes = np.array(msp_size_values)
        _plot_median_histogram(
            ax,
            msp_sizes,
            bins=50,
            hist_range=(0, min(2000, np.percentile(msp_sizes, 99))),
            color='teal',
            xlabel='MSP Size (bp)',
            title='MSP Size Distribution',
            median_format='.0f',
            median_suffix=' bp',
        )
    else:
        _plot_no_data_message(ax, 'No MSP data', 'MSP Size Distribution')


def _plot_read_length_pdf_panel(ax, read_lengths) -> None:
    if _has_values(read_lengths):
        lengths = np.array(read_lengths)
        _plot_median_histogram(
            ax,
            lengths,
            bins=50,
            color='slategray',
            xlabel='Read Length (bp)',
            title='Read Length Distribution',
            median_format='.0f',
            median_suffix=' bp',
        )


def _plot_quality_msp_pdf_page(stats: "FootprintStats", plt, pdf) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    try:
        fig.suptitle(
            'FiberHMM Quality and MSP Statistics', fontsize=14, fontweight='bold',
        )

        _plot_footprint_quality_pdf_panel(axes[0, 0], stats.footprint_scores)
        _plot_msp_size_pdf_panel(axes[0, 1], stats.msp_sizes)
        _plot_read_length_pdf_panel(axes[1, 0], stats.read_lengths)
        _plot_footprint_size_bins(axes[1, 1], stats.footprint_sizes)

        plt.tight_layout()
        pdf.savefig(fig)
    finally:
        plt.close(fig)


def _save_footprint_size_png(stats: "FootprintStats", plt, output_prefix: str) -> bool:
    if not _has_values(stats.footprint_sizes):
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        _plot_footprint_size_png_axis(ax, stats.footprint_sizes)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_footprint_sizes.png", dpi=150)
    finally:
        plt.close(fig)
    return True


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
            self.footprint_coverage.append(
                _footprint_coverage_fraction(total_fp_bases, read_length)
            )

            # Gap sizes (accessible regions between footprints)
            if n_footprints > 1:
                self.gap_sizes.extend(_positive_gaps_between_intervals(ns, nl))

        # MSP stats
        n_msps = len(as_starts) if as_starts is not None else 0
        self.msps_per_read.append(n_msps)

        if n_msps > 0:
            self.msp_sizes.extend(al_lengths.tolist())
            if as_scores is not None:
                self.msp_scores.extend(as_scores.tolist())

    def get_summary(self) -> dict:
        """Generate summary statistics."""
        summary = _initial_footprint_summary(
            self.total_reads_sampled,
            self.reads_with_footprints,
        )

        _add_numeric_summary(
            summary,
            'read_length',
            self.read_lengths,
            include_std=True,
        )

        _add_numeric_summary(
            summary,
            'footprint_size',
            self.footprint_sizes,
            total_key='total_footprints',
            include_std=True,
            include_minmax=True,
            include_iqr=True,
        )

        if self.footprints_per_read:
            _add_positive_count_summary(
                summary, 'footprints_per_read', self.footprints_per_read,
            )

        _add_numeric_summary(
            summary,
            'gap_size',
            self.gap_sizes,
            total_key='total_gaps',
            include_std=True,
        )

        _add_numeric_summary(
            summary,
            'msp_size',
            self.msp_sizes,
            total_key='total_msps',
        )

        _add_numeric_summary(
            summary,
            'footprint_coverage',
            self.footprint_coverage,
        )

        _add_numeric_summary(
            summary,
            'footprint_score',
            self.footprint_scores,
            include_std=True,
        )

        return summary

    def write_summary(self, filepath: str):
        """Write summary statistics to a text file."""
        summary = self.get_summary()

        with open(filepath, 'w') as f:
            f.write("FiberHMM Footprint Statistics\n")
            f.write("=" * 50 + "\n\n")

            _write_read_stats_section(f, summary)
            _write_footprint_stats_section(f, summary)
            _write_gap_stats_section(f, summary)
            _write_msp_stats_section(f, summary)
            _write_footprint_quality_section(f, summary)

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
            _plot_footprint_overview_pdf_page(self, plt, pdf)

            _plot_quality_msp_pdf_page(self, plt, pdf)

        print(f"QC plots saved to: {pdf_path}")

        _save_footprint_size_png(self, plt, output_prefix)


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
    total_reads = _count_primary_mapped_reads(bam_path)

    sample_prob = _stats_sampling_probability(total_reads, n_samples)

    print(
        f"  Sampling ~{min(n_samples, total_reads):,} reads from "
        f"{total_reads:,} total ({sample_prob * 100:.1f}%)"
    )

    # Second pass: collect stats from sampled reads
    sampled = 0

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for read in bam:
            if not is_primary_mapped_alignment(read):
                continue

            # Reservoir sampling
            if random.random() > sample_prob:
                continue

            if sampled >= n_samples:
                break

            _add_read_to_footprint_stats(stats, read, with_scores)
            sampled += 1

    return stats
