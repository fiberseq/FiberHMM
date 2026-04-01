"""DAF-seq deamination encoding: call C→T / G→A mismatches and encode as IUPAC R/Y.

Reads a plain aligned BAM (e.g. from minimap2), identifies deamination
mismatches via the MD tag (or reference FASTA fallback), encodes them as
Y (C or T) / R (A or G) in the query sequence, adds an st:Z tag, and
writes a new BAM ready for ``fiberhmm-apply --mode daf``.
"""

import os
import sys
import time

import pysam
from tqdm import tqdm

from fiberhmm.inference.bam_output import _sort_and_index_bam


# ---------------------------------------------------------------------------
# Per-read encoding
# ---------------------------------------------------------------------------

def encode_read_daf(read, force_strand=None, ref_fasta=None):
    """Identify deamination mismatches and encode as IUPAC R/Y.

    Parameters
    ----------
    read : pysam.AlignedSegment
        Aligned read from a BAM file.
    force_strand : str or None
        ``"CT"``, ``"GA"``, or ``None`` (auto-detect per read).
    ref_fasta : pysam.FastaFile or None
        Opened reference FASTA, used as fallback when the MD tag is absent.

    Returns
    -------
    tuple or None
        ``(new_sequence, st_tag, n_deaminations)`` on success.
        ``None`` if the read should be skipped (unmapped, secondary,
        supplementary, no mismatches, or ambiguous strand).
    """
    # Skip unmapped / secondary / supplementary
    if read.is_unmapped or read.is_secondary or read.is_supplementary:
        return None

    seq = read.query_sequence
    if seq is None:
        return None

    # Get aligned pairs with reference bases
    try:
        pairs = read.get_aligned_pairs(with_seq=True)
    except Exception:
        # MD tag missing and no way to get ref bases without it
        if ref_fasta is None:
            return None
        pairs = None

    # Fallback: build pairs from reference FASTA when MD tag is absent
    if pairs is None or all(p[2] is None for p in pairs if p[0] is not None and p[1] is not None):
        if ref_fasta is None:
            return None
        try:
            pairs = _aligned_pairs_from_fasta(read, ref_fasta)
        except Exception:
            return None

    # Collect mismatch positions
    ct_positions = []  # C→T (+ strand deamination)
    ga_positions = []  # G→A (− strand deamination)

    for query_pos, ref_pos, ref_base in pairs:
        if query_pos is None or ref_pos is None or ref_base is None:
            continue
        ref_base = ref_base.upper()
        query_base = seq[query_pos].upper()
        if ref_base == "C" and query_base == "T":
            ct_positions.append(query_pos)
        elif ref_base == "G" and query_base == "A":
            ga_positions.append(query_pos)

    n_ct = len(ct_positions)
    n_ga = len(ga_positions)

    # Determine conversion strand
    if force_strand is not None:
        strand = force_strand.upper()
    else:
        if n_ct == 0 and n_ga == 0:
            return None
        if n_ct > n_ga:
            strand = "CT"
        elif n_ga > n_ct:
            strand = "GA"
        else:
            # Equal and nonzero – ambiguous, skip
            return None

    # Build new sequence with IUPAC encoding
    seq_list = list(seq)
    if strand == "CT":
        for pos in ct_positions:
            seq_list[pos] = "Y"  # Y = C or T
        n_deam = n_ct
    else:  # GA
        for pos in ga_positions:
            seq_list[pos] = "R"  # R = A or G
        n_deam = n_ga

    new_seq = "".join(seq_list)
    st_tag = strand  # "CT" or "GA"
    return (new_seq, st_tag, n_deam)


def _aligned_pairs_from_fasta(read, ref_fasta):
    """Build (query_pos, ref_pos, ref_base) tuples using the reference FASTA."""
    chrom = read.reference_name
    pairs_no_seq = read.get_aligned_pairs()
    result = []
    for query_pos, ref_pos in pairs_no_seq:
        if ref_pos is not None:
            ref_base = ref_fasta.fetch(chrom, ref_pos, ref_pos + 1)
        else:
            ref_base = None
        result.append((query_pos, ref_pos, ref_base))
    return result


# ---------------------------------------------------------------------------
# BAM-level processing
# ---------------------------------------------------------------------------

def process_bam_daf_encode(
    input_bam,
    output_bam,
    reference=None,
    min_mapq=20,
    min_read_length=1000,
    io_threads=4,
    force_strand=None,
):
    """Stream-encode a BAM with IUPAC R/Y deamination annotations.

    Parameters
    ----------
    input_bam : str
        Input BAM path, or ``"-"`` for stdin.
    output_bam : str
        Output BAM path, or ``"-"`` for stdout.
    reference : str or None
        Path to reference FASTA (fallback when MD tag is missing).
    min_mapq : int
        Minimum mapping quality.
    min_read_length : int
        Minimum aligned read length (bp).
    io_threads : int
        htslib I/O threads.
    force_strand : str or None
        ``"CT"``, ``"GA"``, or ``None`` for auto.

    Returns
    -------
    dict
        Summary statistics: total, encoded, ct, ga, skipped, mean_deam_rate.
    """
    # When writing to stdout, all logging goes to stderr
    _log = sys.stderr if output_bam == "-" else sys.stderr

    ref_fasta = None
    if reference:
        ref_fasta = pysam.FastaFile(reference)

    # Open input
    inbam = pysam.AlignmentFile(
        input_bam, "rb", threads=io_threads, check_sq=False
    )

    # Check MD tag on the first mapped read
    _check_md_tag(inbam, ref_fasta, _log)

    # Open output — handle stdout specially
    _output_target = output_bam
    if output_bam == "-":
        _output_target = os.fdopen(1, "wb", closefd=False)

    outbam = pysam.AlignmentFile(
        _output_target, "wb", header=inbam.header, threads=io_threads
    )

    # Counters
    total = 0
    encoded = 0
    ct_count = 0
    ga_count = 0
    skipped = 0
    total_deam = 0
    total_bases = 0

    start_time = time.time()
    last_progress = start_time

    pbar = tqdm(
        desc="Encoding",
        unit=" reads",
        file=_log,
        mininterval=2.0,
        disable=(output_bam == "-"),
    )

    try:
        for read in inbam.fetch(until_eof=True):
            total += 1

            # Filter: unmapped / secondary / supplementary pass through
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                outbam.write(read)
                skipped += 1
                pbar.update(1)
                continue

            # MAPQ filter
            if read.mapping_quality < min_mapq:
                outbam.write(read)
                skipped += 1
                pbar.update(1)
                continue

            # Length filter
            read_len = read.query_alignment_length
            if read_len is not None and read_len < min_read_length:
                outbam.write(read)
                skipped += 1
                pbar.update(1)
                continue

            # Encode
            result = encode_read_daf(read, force_strand=force_strand, ref_fasta=ref_fasta)
            if result is None:
                outbam.write(read)
                skipped += 1
                pbar.update(1)
                continue

            new_seq, st_tag, n_deam = result

            # Update the read: replace sequence (preserving qualities)
            quals = read.query_qualities
            read.query_sequence = new_seq
            read.query_qualities = quals

            # Set st:Z tag
            read.set_tag("st", st_tag, value_type="Z")

            outbam.write(read)
            encoded += 1
            total_deam += n_deam
            if read.query_length:
                total_bases += read.query_length

            if st_tag == "CT":
                ct_count += 1
            else:
                ga_count += 1

            pbar.update(1)

            # Progress to stderr every 10k reads
            if total % 10000 == 0:
                now = time.time()
                elapsed = now - last_progress
                if elapsed > 0:
                    rate = 10000 / elapsed
                    print(
                        f"  [{total:,} reads] {rate:,.0f} reads/sec "
                        f"(CT={ct_count:,} GA={ga_count:,} skip={skipped:,})",
                        file=_log,
                    )
                    _log.flush()
                last_progress = now

    finally:
        pbar.close()
        outbam.close()
        inbam.close()
        if ref_fasta:
            ref_fasta.close()

    elapsed = time.time() - start_time
    mean_deam_rate = total_deam / total_bases if total_bases > 0 else 0.0

    # Summary
    summary = {
        "total": total,
        "encoded": encoded,
        "ct": ct_count,
        "ga": ga_count,
        "skipped": skipped,
        "mean_deam_rate": mean_deam_rate,
        "elapsed": elapsed,
    }

    print(f"\n{'=' * 60}", file=_log)
    print(f"fiberhmm-daf-encode summary", file=_log)
    print(f"{'=' * 60}", file=_log)
    print(f"  Total reads:       {total:>12,}", file=_log)
    print(f"  Encoded:           {encoded:>12,}", file=_log)
    print(f"    CT (+ strand):   {ct_count:>12,}", file=_log)
    print(f"    GA (- strand):   {ga_count:>12,}", file=_log)
    print(f"  Skipped:           {skipped:>12,}", file=_log)
    print(f"  Mean deam. rate:   {mean_deam_rate:>12.4f}", file=_log)
    print(f"  Elapsed:           {elapsed:>12.1f}s", file=_log)
    if elapsed > 0:
        print(f"  Throughput:        {total / elapsed:>12,.0f} reads/sec", file=_log)
    print(f"{'=' * 60}", file=_log)
    _log.flush()

    # Sort + index if writing to a file (not stdout)
    if output_bam != "-" and os.path.isfile(output_bam):
        print(f"\nFinalizing output BAM...", file=_log)
        _sort_and_index_bam(output_bam, verbose=True, threads=io_threads)

    return summary


def _check_md_tag(inbam, ref_fasta, _log):
    """Peek at the first few mapped reads to check for the MD tag.

    Resets the file iterator after peeking.  Prints a warning or error.
    """
    # Save position and peek
    checked = 0
    has_md = False
    for read in inbam.fetch(until_eof=True):
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        if read.has_tag("MD"):
            has_md = True
            break
        checked += 1
        if checked >= 10:
            break

    # Reset iterator
    inbam.reset()

    if not has_md:
        if ref_fasta is not None:
            print(
                "WARNING: MD tag not found in first reads. "
                "Using reference FASTA as fallback (slower).\n"
                "  Tip: re-align with 'minimap2 --MD' or run 'samtools calmd' "
                "to add MD tags for better performance.",
                file=_log,
            )
        else:
            print(
                "ERROR: MD tag not found in first reads and no --reference provided.\n"
                "  Either:\n"
                "    1. Re-align with 'minimap2 --MD ...' to include MD tags, or\n"
                "    2. Run 'samtools calmd -b in.bam ref.fa > out.bam' to add MD tags, or\n"
                "    3. Use '--reference ref.fa' to supply the reference FASTA.\n",
                file=_log,
            )
            _log.flush()
            sys.exit(1)
