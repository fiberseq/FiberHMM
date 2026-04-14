#!/usr/bin/env python3
"""fiberhmm-consensus-tfs — per-locus TF consensus size and occupancy BED.

Sweeps a recalled BAM (output of fiberhmm-recall-tfs) and outputs one line
per TF locus with consensus footprint size, size variability, and
single-molecule occupancy statistics.  Feeds directly into MEME for motif
discovery and footprint-size v-plots.

Output columns (tab-separated, 0-based BED coordinates):
  chr  start  end  consensus_len  MAD  read_count  spanning_reads  fwd_reads  rev_reads

  chr            Reference sequence name
  start/end      Locus boundaries derived from consensus_len anchored at KDE peak center
  consensus_len  Median footprint length across all TF-calling reads at this locus (bp)
  MAD            Median absolute deviation of footprint length (bp); low = tight size
  read_count     Reads with a TF call at this locus (only loci >= min_cov appear)
  spanning_reads All valid mapped reads covering the locus center — occupancy denominator
  fwd_reads      TF-calling reads from the CT (forward) DAF-seq strand
  rev_reads      TF-calling reads from the GA (reverse) DAF-seq strand

Single-molecule occupancy = read_count / spanning_reads.
For Hia5 (no st:Z tag), fwd_reads and rev_reads will both be 0.

Examples:
  fiberhmm-consensus-tfs -i recalled.bam -o consensus_tfs.bed
  fiberhmm-consensus-tfs -i recalled.bam -o - --min-tq 50 --min-cov 10
  fiberhmm-consensus-tfs -i recalled.bam -o tfs.bed -q 20 --min-tq 80
"""
from __future__ import annotations

import argparse
import heapq
import sys
from collections import defaultdict
from typing import Generator, Tuple

import numpy as np
import pysam
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from fiberhmm.io.ma_tags import parse_ma_tag, parse_aq_array


# ── Query → reference coordinate conversion ──────────────────────────────────

def _ref_interval(read: pysam.AlignedSegment,
                  q_start: int, length: int) -> Tuple[int, int] | Tuple[None, None]:
    """Return (ref_start, ref_end) in half-open BED format for a query interval.

    Handles indels: slices the full reference-position map across the footprint
    and takes min/max of non-None entries.  Returns (None, None) if the entire
    footprint falls inside an insertion relative to the reference (rare).
    """
    ref_map = read.get_reference_positions(full_length=True)
    mapped = [p for p in ref_map[q_start: q_start + length] if p is not None]
    if not mapped:
        return None, None
    return mapped[0], mapped[-1] + 1   # half-open [start, end)


# ── TF call extraction ────────────────────────────────────────────────────────

def _iter_tf_calls(read: pysam.AlignedSegment,
                   min_tq: int) -> Generator[Tuple[int, int, str, int], None, None]:
    """Yield (ref_start, ref_end, strand, tq) for each TF call on this read.

    strand: DAF-seq chemistry strand from the st:Z tag ('CT', 'GA', or '.' if absent).
    Silently skips calls that land entirely in insertions relative to the reference.
    """
    try:
        ma_str = read.get_tag('MA')
    except KeyError:
        return
    try:
        aq = list(read.get_tag('AQ'))
    except KeyError:
        aq = []

    try:
        parsed = parse_ma_tag(ma_str)
    except ValueError:
        return

    strand = read.get_tag('st') if read.has_tag('st') else '.'

    qual_specs = [rt[2] for rt in parsed['raw_types']]
    n_per_type  = [len(rt[3]) for rt in parsed['raw_types']]
    per_ann     = parse_aq_array(aq, qual_specs, n_per_type)

    # ann_idx must increment for ALL annotations (nuc, msp, tf) to stay
    # in sync with the flat per_ann list built by parse_aq_array.
    ann_idx = 0
    for name, _strand_field, _qspec, intervals in parsed['raw_types']:
        for (q_start, length) in intervals:
            quals = per_ann[ann_idx] if ann_idx < len(per_ann) else []
            ann_idx += 1
            if name != 'tf':
                continue
            tq = int(quals[0]) if len(quals) >= 1 else 0
            if tq < min_tq:
                continue
            ref_start, ref_end = _ref_interval(read, q_start, length)
            if ref_start is None:
                continue
            yield ref_start, ref_end, strand, tq


# ── Sweep-line spanning-read count ───────────────────────────────────────────

def _spanning_at_peaks(peaks: np.ndarray, read_spans: list) -> dict:
    """Count spanning reads at each KDE peak center using an O(N log N) sweep line.

    BED coordinates are half-open, so a read with ref_end == p does NOT cover
    base p.  We therefore pop reads where heap[0] <= p (not < p).

    The heap state persists across peaks because peaks are sorted — we never
    need to rewind.  Returns {peak_position: spanning_count}.
    """
    read_spans.sort(key=lambda x: x[0])
    heap: list[int] = []        # min-heap of ref_end values
    read_idx = 0
    result: dict[int, int] = {}

    for p in peaks:             # find_peaks returns peaks in sorted order
        while read_idx < len(read_spans) and read_spans[read_idx][0] <= p:
            heapq.heappush(heap, read_spans[read_idx][1])
            read_idx += 1
        while heap and heap[0] <= p:   # half-open: ref_end == p → not covering p
            heapq.heappop(heap)
        result[int(p)] = len(heap)

    return result


# ── Per-chromosome processing ─────────────────────────────────────────────────

def _process_chrom(chrom: str, chrom_len: int,
                   tf_calls: list, read_spans: list,
                   min_cov: int, kde_sigma: float, peak_distance: int,
                   bed_out) -> int:
    """Cluster TF calls by KDE, compute consensus stats, write BED lines.

    Returns the number of loci written.
    """
    if not tf_calls:
        return 0

    centers_float = np.array([c[0] for c in tf_calls], dtype=np.float64)
    centers_int   = np.round(centers_float).astype(np.int64)

    # Clamp out-of-range centers (alignment edge-cases near telomeres can push
    # a footprint center slightly beyond the chromosome boundary).
    valid_mask = (centers_int >= 0) & (centers_int < chrom_len)
    if not valid_mask.any():
        return 0
    centers_valid = centers_int[valid_mask]

    # Fast KDE: np.bincount (pure C, ~20x faster than np.add.at) + float32
    # (halves RAM vs float64 — 1 GB vs 2 GB for chr1 at ~250 Mbp).
    signal   = np.bincount(centers_valid, minlength=chrom_len).astype(np.float32)
    smoothed = gaussian_filter1d(signal, sigma=kde_sigma)

    # Peak finding — NO height= filter here.
    # Gaussian filtering conserves area, not peak height.  A locus with 5
    # perfectly stacked reads has a smoothed peak height of only ~0.67 (not 5).
    # Enforcing height=MIN_COV would silently require >37 reads before any peak
    # is detected.  Enforce coverage based on exact per-peak counts below.
    peaks_arr, _ = find_peaks(smoothed, distance=peak_distance)
    if len(peaks_arr) == 0:
        return 0

    # Spanning read counts at each peak (sweep line runs after peaks are known)
    spanning_dict = _spanning_at_peaks(peaks_arr, read_spans)

    # Assign each TF call to its nearest KDE peak.
    # np.searchsorted gives the INSERTION index, which is not always the nearest
    # neighbour — we check both idx-1 and idx and take the closer one.
    # We also enforce a capture radius (peak_distance) to drop orphaned noise TFs.
    peak_pos = peaks_arr.astype(np.float64)
    per_peak: dict[int, list] = defaultdict(list)

    for center, ref_len, strand, tq in tf_calls:
        c_int = round(center)
        if not (0 <= c_int < chrom_len):
            continue
        idx = int(np.searchsorted(peak_pos, center))
        candidates: list[tuple[int, int]] = []
        if idx > 0:              candidates.append((int(peaks_arr[idx - 1]), idx - 1))
        if idx < len(peaks_arr): candidates.append((int(peaks_arr[idx]),     idx))
        if not candidates:
            continue
        closest_val, closest_idx = min(candidates, key=lambda x: abs(x[0] - center))
        if abs(closest_val - center) > peak_distance:
            continue                  # orphaned TF, too far from any structural peak
        per_peak[closest_idx].append((ref_len, strand, tq))

    n_written = 0
    for peak_idx, calls in sorted(per_peak.items()):
        lengths    = np.array([c[0] for c in calls], dtype=np.float64)
        read_count = len(lengths)
        if read_count < min_cov:
            continue                  # below minimum coverage floor

        med_len       = float(np.median(lengths))
        consensus_len = int(round(med_len))
        if consensus_len < 1:
            continue
        mad       = float(np.median(np.abs(lengths - med_len)))
        fwd_reads = sum(1 for _, s, _ in calls if s == 'CT')
        rev_reads = sum(1 for _, s, _ in calls if s == 'GA')

        peak_center = float(peaks_arr[peak_idx])
        # Anchor start, then add consensus_len to avoid Python banker's rounding
        # shrinking the BED feature by 1 bp when peak_center ± half-len is
        # exactly a 0.5 value.
        bed_start = max(0, int(round(peak_center - consensus_len / 2.0)))
        bed_end   = bed_start + consensus_len

        spanning = spanning_dict.get(int(peaks_arr[peak_idx]), 0)

        bed_out.write(
            f"{chrom}\t{bed_start}\t{bed_end}\t"
            f"{consensus_len}\t{mad:.1f}\t{read_count}\t"
            f"{spanning}\t{fwd_reads}\t{rev_reads}\n"
        )
        n_written += 1

    return n_written


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('-i', '--input', required=True,
                   help='Input recalled BAM (sorted + indexed, from fiberhmm-recall-tfs). '
                        'Must have MA/AQ tags.')
    p.add_argument('-o', '--output', required=True,
                   help='Output BED file. Use "-" for stdout.')
    p.add_argument('-q', '--min-mapq', type=int, default=20,
                   help='Min mapping quality (default: 20). Use >=20 to exclude '
                        'multi-mapping reads at repetitive loci such as tRNA genes.')
    p.add_argument('--min-tq', type=int, default=0,
                   help='Min TF quality score tq (0-255, default: 0). '
                        'tq=50 ≈ LLR 5 nats (~150:1); tq=100 ≈ LLR 10 nats (~22,000:1).')
    p.add_argument('--min-cov', type=int, default=5,
                   help='Min TF-calling reads per locus to emit a BED line (default: 5). '
                        'Loci with fewer reads are silently dropped.')
    p.add_argument('--kde-sigma', type=float, default=3.0,
                   help='Gaussian smoothing sigma for footprint-center KDE (bp, default: 3.0). '
                        'Smooths single-bp edge wobble; larger values merge closely-spaced TFs.')
    p.add_argument('--peak-distance', type=int, default=15,
                   help='Min distance between adjacent TF locus peaks (bp, default: 15). '
                        'Roughly the steric exclusion radius of a single TF. '
                        'Also used as the capture radius for assigning TF calls to peaks.')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.output == '-':
        bed_out = sys.stdout
    else:
        bed_out = open(args.output, 'w')

    try:
        _banner = lambda msg: print(f"[fiberhmm-consensus-tfs] {msg}", file=sys.stderr)
        _banner(f"input:         {args.input}")
        _banner(f"min-mapq:      {args.min_mapq}")
        _banner(f"min-tq:        {args.min_tq}")
        _banner(f"min-cov:       {args.min_cov}")
        _banner(f"kde-sigma:     {args.kde_sigma}")
        _banner(f"peak-distance: {args.peak_distance}")

        bed_out.write(
            "#chr\tstart\tend\tconsensus_len\tMAD\t"
            "read_count\tspanning_reads\tfwd_reads\trev_reads\n"
        )

        total_loci = 0

        with pysam.AlignmentFile(args.input, 'rb') as bam_fh:
            references = bam_fh.references
            if not references:
                _banner("warning: BAM has no reference sequences in header")
                return

            for chrom in references:
                chrom_len  = bam_fh.get_reference_length(chrom)
                tf_calls:   list = []   # (center, ref_len, strand, tq)
                read_spans: list = []   # (ref_start, ref_end) — ALL valid mapped reads

                try:
                    read_iter = bam_fh.fetch(chrom)
                except ValueError:
                    continue   # no BAM index or chrom absent from BAM

                for read in read_iter:
                    if (read.is_unmapped or read.is_secondary or
                            read.is_supplementary or read.is_duplicate):
                        continue
                    if read.mapping_quality < args.min_mapq:
                        continue

                    # Record span for EVERY valid read — denominator for occupancy.
                    # Using only TF-calling reads would inflate fractions toward 100%.
                    read_spans.append((read.reference_start, read.reference_end))

                    for ref_start, ref_end, strand, tq in _iter_tf_calls(read, args.min_tq):
                        center  = (ref_start + ref_end) / 2.0
                        ref_len = ref_end - ref_start
                        tf_calls.append((center, ref_len, strand, tq))

                n = _process_chrom(
                    chrom, chrom_len,
                    tf_calls, read_spans,
                    args.min_cov, args.kde_sigma, args.peak_distance,
                    bed_out,
                )
                if n:
                    _banner(f"{chrom}: {n} loci")
                total_loci += n

        _banner(f"total loci: {total_loci}")

    finally:
        if args.output != '-':
            bed_out.close()


if __name__ == '__main__':
    main()
