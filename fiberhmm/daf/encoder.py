"""DAF-seq deamination encoding: call C→T / G→A mismatches and encode as IUPAC R/Y.

Reads a plain aligned BAM (e.g. from minimap2), identifies deamination
mismatches via the MD tag (or reference FASTA fallback), encodes them as
Y (C or T) / R (A or G) in the query sequence, adds an st:Z tag, and
writes a new BAM ready for ``fiberhmm-apply --mode daf``.
"""

import os
import sys
import time
from dataclasses import dataclass

import pysam
from tqdm import tqdm

from fiberhmm.inference.bam_output import _sort_and_index_bam
from fiberhmm.inference.read_filters import is_primary_mapped_alignment

# CIGAR op codes (from the BAM spec) that consume reference:
#   M=0, D=2, N=3, =7, X=8. Of these, MD describes M/=/X/D only
#   (N = skipped reference, no MD coverage), so we exclude N when
#   comparing against MD.
_CIGAR_REF_CONSUMING_FOR_MD = {0, 2, 7, 8}
_DAF_ENCODE_STAT_FIELDS = (
    "encoded",
    "skipped",
    "ct",
    "ga",
    "total_deam",
    "total_bases",
)


@dataclass(frozen=True)
class _DafChimeraSegmentCounts:
    right_total: int
    ga_left: int
    ct_right: int
    ga_right: int


@dataclass(frozen=True)
class _DafMismatchPositions:
    ct: list[int]
    ga: list[int]


@dataclass(frozen=True)
class _EncodedDafSequence:
    sequence: str
    n_deaminations: int


@dataclass(frozen=True)
class _EncodedDafRead:
    sequence: str
    st_tag: str
    n_deaminations: int

    def as_tuple(self):
        return self.sequence, self.st_tag, self.n_deaminations


@dataclass(frozen=True)
class _DafEncodeHandles:
    ref_fasta: object
    inbam: object
    outbam: object
    pbar: object


@dataclass(frozen=True)
class _DafEncodeReadStats:
    encoded: int
    skipped: int
    ct: int
    ga: int
    total_deam: int
    total_bases: int


@dataclass
class _DafEncodeCounts:
    total: int = 0
    encoded: int = 0
    skipped: int = 0
    ct: int = 0
    ga: int = 0
    total_deam: int = 0
    total_bases: int = 0


def _md_tag_ref_length(md_string: str) -> int:
    """Return the reference length encoded by an MD tag.

    MD is a sequence of: runs of digits (ref positions matched), single-base
    mismatches (one ref base consumed each), and ``^<SEQ>`` deletions
    (len(<SEQ>) ref bases consumed). See SAM spec section 1.4.11.
    """
    n = 0
    i = 0
    L = len(md_string)
    while i < L:
        c = md_string[i]
        if c.isdigit():
            j = i
            while j < L and md_string[j].isdigit():
                j += 1
            n += int(md_string[i:j])
            i = j
        elif c == '^':
            # ^<seq> deletion; each letter is one ref base consumed.
            j = i + 1
            while j < L and md_string[j].isalpha():
                j += 1
            n += j - (i + 1)
            i = j
        elif c.isalpha():
            # Single-base mismatch, one ref base consumed.
            n += 1
            i += 1
        else:
            # Skip any stray punctuation (the spec doesn't allow it but
            # be permissive on read-side).
            i += 1
    return n


def md_matches_cigar(read) -> bool:
    """Cheap pre-validation: does the MD tag's encoded reference length
    match the CIGAR's reference-consuming operations?

    Returns True if the read has no MD tag (caller should handle that
    case separately), True if the lengths match, False if they disagree.

    Motivation: ``pysam.AlignedSegment.get_aligned_pairs(with_seq=True)``
    raises AssertionError on mismatch AND, in at least some pysam
    versions, corrupts internal malloc state before the exception
    propagates — which then manifests as ``malloc(): invalid size``
    somewhere later in the worker. Skipping the call on obviously-bad
    MD avoids triggering the crash path at all.
    """
    try:
        md = read.get_tag('MD') if read.has_tag('MD') else None
    except Exception:
        return True
    if md is None:
        return True
    try:
        md_len = _md_tag_ref_length(md)
    except Exception:
        return False
    cigar = read.cigartuples
    if cigar is None:
        return True
    cigar_ref_len = sum(length for op, length in cigar
                        if op in _CIGAR_REF_CONSUMING_FOR_MD)
    return md_len == cigar_ref_len


def _select_daf_strand(n_ct: int, n_ga: int, force_strand=None):
    if force_strand is not None:
        return force_strand.upper()
    if n_ct == 0 and n_ga == 0:
        return None
    if n_ct > n_ga:
        return "CT"
    if n_ga > n_ct:
        return "GA"
    return None


def _daf_mismatch_positions_from_pairs(pairs, seq: str):
    ct_positions = []  # C->T (+ strand deamination)
    ga_positions = []  # G->A (- strand deamination)

    for query_pos, ref_pos, ref_base in pairs:
        if query_pos is None or ref_pos is None or ref_base is None:
            continue
        ref_base = ref_base.upper()
        query_base = seq[query_pos].upper()
        if ref_base == "C" and query_base == "T":
            ct_positions.append(query_pos)
        elif ref_base == "G" and query_base == "A":
            ga_positions.append(query_pos)
    return _DafMismatchPositions(ct_positions, ga_positions)


def _needs_fasta_pair_fallback(pairs) -> bool:
    return pairs is None or all(
        p[2] is None for p in pairs if p[0] is not None and p[1] is not None
    )


def _aligned_pairs_with_reference_bases(read, ref_fasta=None):
    # Pre-validate MD vs CIGAR to avoid pysam's AssertionError path on
    # malformed BAMs — that path can corrupt malloc state in some pysam
    # versions, crashing the worker later with "malloc(): invalid size".
    pairs = None
    if md_matches_cigar(read):
        try:
            pairs = read.get_aligned_pairs(with_seq=True)
        except Exception:
            if ref_fasta is None:
                return None
            pairs = None
    elif ref_fasta is None:
        return None

    if not _needs_fasta_pair_fallback(pairs):
        return pairs

    if ref_fasta is None:
        return None
    try:
        return _aligned_pairs_from_fasta(read, ref_fasta)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-read encoding
# ---------------------------------------------------------------------------

def get_daf_positions(read, force_strand=None, ref_fasta=None):
    """Collect C->T and G->A mismatch positions for a DAF-seq read.

    Pure position-collection helper used by both ``encode_read_daf``
    (which rewrites the query sequence with R/Y IUPAC codes) and the
    in-memory fallback inside ``fiberhmm-call --mode daf`` (which
    consumes positions directly without touching the stored sequence).

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
        ``(ct_positions, ga_positions, strand)`` where ``strand`` is
        ``"CT"`` or ``"GA"`` and the two lists are the full sets of
        C->T and G->A query-position mismatches (both populated even
        though only the selected-strand list is used downstream -- the
        other is returned for diagnostics and future use).

        ``None`` if the read should be skipped (unmapped, secondary,
        supplementary, no mismatches, or ambiguous strand with
        no ``force_strand``).
    """
    # Skip unmapped / secondary / supplementary
    if not is_primary_mapped_alignment(read):
        return None

    seq = read.query_sequence
    if seq is None:
        return None

    pairs = _aligned_pairs_with_reference_bases(read, ref_fasta)
    if pairs is None:
        return None

    mismatch_positions = _daf_mismatch_positions_from_pairs(pairs, seq)

    n_ct = len(mismatch_positions.ct)
    n_ga = len(mismatch_positions.ga)

    strand = _select_daf_strand(n_ct, n_ga, force_strand)
    if strand is None:
        return None

    return (mismatch_positions.ct, mismatch_positions.ga, strand)


def _daf_chimera_events(ct_positions, ga_positions):
    return sorted([(int(p), 1) for p in ct_positions]
                  + [(int(p), -1) for p in ga_positions])


def _has_min_daf_chimera_strand_counts(
    nct: int,
    nga: int,
    min_seg_events: int,
) -> bool:
    return min(nct, nga) >= min_seg_events


def _has_min_daf_chimera_events(n_events: int, min_seg_events: int) -> bool:
    return n_events >= 2 * min_seg_events


def _is_pure_daf_segment(dominant_count, segment_size,
                         min_seg_events: int, purity: float) -> bool:
    return (
        dominant_count >= min_seg_events
        and dominant_count >= purity * segment_size
    )


def _daf_chimera_segment_counts(ct_left, left_n, nct, nga):
    right_n = nct + nga - left_n
    ga_left = left_n - ct_left
    ct_right = nct - ct_left
    ga_right = nga - ga_left
    return _DafChimeraSegmentCounts(
        right_total=right_n,
        ga_left=ga_left,
        ct_right=ct_right,
        ga_right=ga_right,
    )


def _daf_chimera_breakpoint_matches(ct_left, left_n, nct, nga,
                                    min_seg_events: int,
                                    purity: float) -> bool:
    segment_counts = _daf_chimera_segment_counts(
        ct_left, left_n, nct, nga,
    )

    ct_then_ga = (
        _is_pure_daf_segment(ct_left, left_n, min_seg_events, purity)
        and _is_pure_daf_segment(
            segment_counts.ga_right,
            segment_counts.right_total,
            min_seg_events,
            purity,
        )
    )
    ga_then_ct = (
        _is_pure_daf_segment(
            segment_counts.ga_left,
            left_n,
            min_seg_events,
            purity,
        )
        and _is_pure_daf_segment(
            segment_counts.ct_right,
            segment_counts.right_total,
            min_seg_events,
            purity,
        )
    )
    return ct_then_ga or ga_then_ct


def is_daf_chimera(ct_positions, ga_positions,
                   min_seg_events: int = 5, purity: float = 0.8) -> bool:
    """Detect a DAF-seq strand-swap chimera.

    A clean DAF read is deaminated on ONE strand (C->T XOR G->A). A chimera
    (template switch / merged molecule) "swaps" mid-read: a contiguous segment
    of C->T deamination and a separate contiguous segment of G->A deamination.

    Spatial test: order all deamination events by query position (CT=+1, GA=-1);
    a chimera has a breakpoint splitting them into two segments that are each
    >= ``purity`` pure for OPPOSITE strands, with >= ``min_seg_events`` dominant
    events on each side. Scattered minority events (SNPs / sequencing error) do
    not form a pure segment, so they are not flagged -- robust to the low
    deamination rate of under-deaminating enzymes like DddB.

    Returns True if the read looks chimeric.
    """
    nct = len(ct_positions)
    nga = len(ga_positions)
    if not _has_min_daf_chimera_strand_counts(nct, nga, min_seg_events):
        return False

    events = _daf_chimera_events(ct_positions, ga_positions)
    n = len(events)
    if not _has_min_daf_chimera_events(n, min_seg_events):
        return False

    ct_left = 0
    for k in range(1, n):                       # breakpoint: left=[0,k), right=[k,n)
        if events[k - 1][1] == 1:
            ct_left += 1
        left_n = k
        right_n = n - k
        if left_n < min_seg_events or right_n < min_seg_events:
            continue
        if _daf_chimera_breakpoint_matches(
            ct_left,
            left_n,
            nct,
            nga,
            min_seg_events,
            purity,
        ):
            return True
    return False


def _mark_iupac_positions(sequence: str, positions, code: str) -> str:
    seq_list = list(sequence)
    for pos in positions:
        seq_list[int(pos)] = code
    return "".join(seq_list)


def _encoded_daf_sequence(seq: str, ct_positions, ga_positions, strand: str):
    if strand == "CT":
        return _EncodedDafSequence(
            _mark_iupac_positions(seq, ct_positions, "Y"),
            len(ct_positions),
        )
    return _EncodedDafSequence(
        _mark_iupac_positions(seq, ga_positions, "R"),
        len(ga_positions),
    )


def _encode_read_daf_record(read, force_strand=None, ref_fasta=None):
    res = get_daf_positions(read, force_strand=force_strand, ref_fasta=ref_fasta)
    if res is None:
        return None
    ct_positions, ga_positions, strand = res

    seq = read.query_sequence
    encoded = _encoded_daf_sequence(
        seq, ct_positions, ga_positions, strand,
    )

    st_tag = strand  # "CT" or "GA"
    return _EncodedDafRead(encoded.sequence, st_tag, encoded.n_deaminations)


def encode_read_daf(read, force_strand=None, ref_fasta=None):
    """Identify deamination mismatches and encode as IUPAC R/Y.

    Thin wrapper over ``get_daf_positions`` that adds the IUPAC
    sequence rewrite. Preserves the original ``encode_read_daf`` return
    contract exactly so ``fiberhmm-daf-encode`` output is bit-for-bit
    unchanged after the refactor.

    Returns
    -------
    tuple or None
        ``(new_sequence, st_tag, n_deaminations)`` on success.
        ``None`` if the read should be skipped.
    """
    encoded = _encode_read_daf_record(
        read,
        force_strand=force_strand,
        ref_fasta=ref_fasta,
    )
    return None if encoded is None else encoded.as_tuple()


def _aligned_pairs_from_fasta(read, ref_fasta):
    """Build (query_pos, ref_pos, ref_base) tuples using the reference FASTA."""
    chrom = read.reference_name
    pairs_no_seq = read.get_aligned_pairs()
    ref_start = read.reference_start
    ref_end = read.reference_end
    ref_seq = None
    if ref_start is not None and ref_end is not None and ref_end > ref_start:
        ref_seq = ref_fasta.fetch(chrom, ref_start, ref_end)

    result = []
    for query_pos, ref_pos in pairs_no_seq:
        if ref_pos is not None:
            if ref_seq is not None and ref_start <= ref_pos < ref_end:
                ref_base = ref_seq[ref_pos - ref_start]
            else:
                ref_base = ref_fasta.fetch(chrom, ref_pos, ref_pos + 1)
        else:
            ref_base = None
        result.append((query_pos, ref_pos, ref_base))
    return result


# ---------------------------------------------------------------------------
# BAM-level processing
# ---------------------------------------------------------------------------

def _write_skipped_daf_read(outbam, pbar, read) -> int:
    outbam.write(read)
    pbar.update(1)
    return 1


def _daf_encode_skip_reason(read, min_mapq: int, min_read_length: int):
    if not is_primary_mapped_alignment(read):
        return "not_primary_mapped"
    if read.mapping_quality < min_mapq:
        return "low_mapq"
    read_len = read.query_alignment_length
    if read_len is not None and read_len < min_read_length:
        return "too_short"
    return None


def _daf_encode_summary(total: int, encoded: int, ct_count: int, ga_count: int,
                        skipped: int, total_deam: int, total_bases: int,
                        elapsed: float) -> dict:
    mean_deam_rate = total_deam / total_bases if total_bases > 0 else 0.0
    return {
        "total": total,
        "encoded": encoded,
        "ct": ct_count,
        "ga": ga_count,
        "skipped": skipped,
        "mean_deam_rate": mean_deam_rate,
        "elapsed": elapsed,
    }


def _new_daf_encode_counts() -> _DafEncodeCounts:
    return _DafEncodeCounts()


def _accumulate_daf_read_stats(
    counts: _DafEncodeCounts,
    read_stats: _DafEncodeReadStats,
) -> None:
    counts.total += 1
    for field in _DAF_ENCODE_STAT_FIELDS:
        setattr(counts, field, getattr(counts, field) + getattr(read_stats, field))


def _daf_encode_summary_from_counts(
    counts: _DafEncodeCounts,
    elapsed: float,
) -> dict:
    return _daf_encode_summary(
        counts.total,
        counts.encoded,
        counts.ct,
        counts.ga,
        counts.skipped,
        counts.total_deam,
        counts.total_bases,
        elapsed,
    )


def _daf_encode_throughput(summary: dict):
    elapsed = summary['elapsed']
    if elapsed <= 0:
        return None
    return summary['total'] / elapsed


def _daf_progress_rate(reads_processed: int, elapsed: float):
    if elapsed <= 0:
        return None
    return reads_processed / elapsed


def _print_daf_progress(
    total: int,
    reads_processed: int,
    elapsed: float,
    ct_count: int,
    ga_count: int,
    skipped: int,
    log,
) -> None:
    rate = _daf_progress_rate(reads_processed, elapsed)
    if rate is None:
        return
    print(
        f"  [{total:,} reads] {rate:,.0f} reads/sec "
        f"(CT={ct_count:,} GA={ga_count:,} skip={skipped:,})",
        file=log,
    )
    log.flush()


def _print_daf_encode_summary(summary: dict, log) -> None:
    print(f"\n{'=' * 60}", file=log)
    print("fiberhmm-daf-encode summary", file=log)
    print(f"{'=' * 60}", file=log)
    print(f"  Total reads:       {summary['total']:>12,}", file=log)
    print(f"  Encoded:           {summary['encoded']:>12,}", file=log)
    print(f"    CT (+ strand):   {summary['ct']:>12,}", file=log)
    print(f"    GA (- strand):   {summary['ga']:>12,}", file=log)
    print(f"  Skipped:           {summary['skipped']:>12,}", file=log)
    print(f"  Mean deam. rate:   {summary['mean_deam_rate']:>12.4f}", file=log)
    print(f"  Elapsed:           {summary['elapsed']:>12.1f}s", file=log)
    throughput = _daf_encode_throughput(summary)
    if throughput is not None:
        print(f"  Throughput:        {throughput:>12,.0f} reads/sec", file=log)
    print(f"{'=' * 60}", file=log)
    log.flush()


def _apply_daf_encoding_to_read(read, new_seq: str, st_tag: str) -> None:
    quals = read.query_qualities
    read.query_sequence = new_seq
    read.query_qualities = quals
    read.set_tag("st", st_tag, value_type="Z")


def _daf_encoded_read_stats(
    read,
    st_tag: str,
    n_deam: int,
) -> _DafEncodeReadStats:
    return _DafEncodeReadStats(
        encoded=1,
        skipped=0,
        ct=1 if st_tag == "CT" else 0,
        ga=0 if st_tag == "CT" else 1,
        total_deam=n_deam,
        total_bases=read.query_length or 0,
    )


def _daf_skipped_read_stats() -> _DafEncodeReadStats:
    return _DafEncodeReadStats(
        encoded=0,
        skipped=1,
        ct=0,
        ga=0,
        total_deam=0,
        total_bases=0,
    )


def _process_daf_encode_read(
    outbam,
    pbar,
    read,
    min_mapq: int,
    min_read_length: int,
    force_strand=None,
    ref_fasta=None,
) -> _DafEncodeReadStats:
    skip_reason = _daf_encode_skip_reason(read, min_mapq, min_read_length)
    if skip_reason:
        _write_skipped_daf_read(outbam, pbar, read)
        return _daf_skipped_read_stats()

    encoded = _encode_read_daf_record(
        read,
        force_strand=force_strand,
        ref_fasta=ref_fasta,
    )
    if encoded is None:
        _write_skipped_daf_read(outbam, pbar, read)
        return _daf_skipped_read_stats()

    _apply_daf_encoding_to_read(read, encoded.sequence, encoded.st_tag)

    outbam.write(read)
    pbar.update(1)
    return _daf_encoded_read_stats(read, encoded.st_tag, encoded.n_deaminations)


def _open_daf_reference(reference):
    if reference:
        return pysam.FastaFile(reference)
    return None


def _open_daf_input_bam(input_bam, io_threads: int):
    return pysam.AlignmentFile(
        input_bam,
        "rb",
        threads=io_threads,
        check_sq=False,
    )


def _daf_output_target(output_bam):
    if output_bam == "-":
        return os.fdopen(1, "wb", closefd=False)
    return output_bam


def _open_daf_output_bam(output_bam, inbam, io_threads: int):
    return pysam.AlignmentFile(
        _daf_output_target(output_bam),
        "wb",
        header=inbam.header,
        threads=io_threads,
    )


def _new_daf_encode_progress(output_bam, log):
    return tqdm(
        desc="Encoding",
        unit=" reads",
        file=log,
        mininterval=2.0,
        disable=(output_bam == "-"),
    )


def _maybe_print_daf_encode_progress(
    counts: _DafEncodeCounts,
    last_progress: float,
    log,
):
    if counts.total == 0 or counts.total % 10000 != 0:
        return last_progress

    now = time.time()
    elapsed = now - last_progress
    _print_daf_progress(
        counts.total,
        10000,
        elapsed,
        counts.ct,
        counts.ga,
        counts.skipped,
        log,
    )
    return now


def _stream_daf_encode_reads(
    inbam,
    outbam,
    pbar,
    counts: _DafEncodeCounts,
    min_mapq: int,
    min_read_length: int,
    force_strand,
    ref_fasta,
    log,
    last_progress: float,
) -> float:
    for read in inbam.fetch(until_eof=True):
        read_stats = _process_daf_encode_read(
            outbam,
            pbar,
            read,
            min_mapq,
            min_read_length,
            force_strand=force_strand,
            ref_fasta=ref_fasta,
        )
        _accumulate_daf_read_stats(counts, read_stats)
        last_progress = _maybe_print_daf_encode_progress(
            counts,
            last_progress,
            log,
        )

    return last_progress


def _close_daf_encode_handles(pbar, outbam, inbam, ref_fasta) -> None:
    close_error = None
    for handle in (pbar, outbam, inbam, ref_fasta):
        if handle is None:
            continue
        try:
            handle.close()
        except Exception as error:
            if close_error is None:
                close_error = error

    if close_error is not None:
        raise close_error


def _maybe_finalize_daf_output(output_bam, io_threads: int, log) -> None:
    if output_bam == "-" or not os.path.isfile(output_bam):
        return

    print("\nFinalizing output BAM...", file=log)
    _sort_and_index_bam(output_bam, verbose=True, threads=io_threads)


def _open_daf_encode_handles(input_bam, output_bam, reference, io_threads: int, log):
    ref_fasta = None
    inbam = None
    outbam = None
    pbar = None
    try:
        ref_fasta = _open_daf_reference(reference)
        inbam = _open_daf_input_bam(input_bam, io_threads)
        _check_md_tag(inbam, ref_fasta, log)
        outbam = _open_daf_output_bam(output_bam, inbam, io_threads)
        pbar = _new_daf_encode_progress(output_bam, log)
        return _DafEncodeHandles(ref_fasta, inbam, outbam, pbar)
    except BaseException:
        _close_daf_encode_handles(pbar, outbam, inbam, ref_fasta)
        raise


def _finalize_daf_encode_run(
    counts: _DafEncodeCounts,
    start_time: float,
    output_bam,
    io_threads: int,
    log,
) -> dict:
    summary = _daf_encode_summary_from_counts(counts, time.time() - start_time)
    _print_daf_encode_summary(summary, log)
    _maybe_finalize_daf_output(output_bam, io_threads, log)
    return summary


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
    # Keep all progress/log output off stdout so "-" remains a clean BAM stream.
    _log = sys.stderr

    counts = _new_daf_encode_counts()

    start_time = time.time()
    last_progress = start_time

    handles = None

    try:
        handles = _open_daf_encode_handles(
            input_bam, output_bam, reference, io_threads, _log,
        )

        last_progress = _stream_daf_encode_reads(
            handles.inbam,
            handles.outbam,
            handles.pbar,
            counts,
            min_mapq,
            min_read_length,
            force_strand,
            handles.ref_fasta,
            _log,
            last_progress,
        )

    finally:
        if handles is not None:
            _close_daf_encode_handles(
                handles.pbar,
                handles.outbam,
                handles.inbam,
                handles.ref_fasta,
            )

    return _finalize_daf_encode_run(counts, start_time, output_bam, io_threads, _log)


def _first_mapped_reads_have_md_tag(inbam, max_reads: int = 10) -> bool:
    checked = 0
    for read in inbam.fetch(until_eof=True):
        if not is_primary_mapped_alignment(read):
            continue
        if read.has_tag("MD"):
            return True
        checked += 1
        if checked >= max_reads:
            break
    return False


def _check_md_tag(inbam, ref_fasta, _log):
    """Peek at the first few mapped reads to check for the MD tag.

    Resets the file iterator after peeking.  Prints a warning or error.
    """
    has_md = _first_mapped_reads_have_md_tag(inbam)

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
