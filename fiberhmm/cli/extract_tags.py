#!/usr/bin/env python3
"""
extract_tags.py - Extract tags from FiberHMM-tagged BAMs to BED12/bigBed.

Supported annotation types:
  - footprint: Nucleosome footprints (ns/nl) -> BED12 (one line per read)
  - msp: Methylase-sensitive patches (as/al) -> BED12 (one line per read)
  - tf:  TF/Pol II footprints (MA/AQ tf+QQQ) -> BED12 (one line per read).
         Quality filter: --min-tq (default 50 = LLR >= 5 nats,
         matches fiberhmm-recall-tfs's default emission floor).
  - m6a:  m6A modification positions (MM/ML) -> BED12 (one line per read)
  - m5c:  5mC modification positions (for DAF-seq) -> BED12
  - deam: DAF-seq deamination calls from R/Y IUPAC codes in the query
          sequence (written by fiberhmm-daf-encode) -> BED12 (one line per
          read). The DAF analogue of --m6a for IUPAC-encoded BAMs that
          don't carry MM/ML.

Extracts various tag types from tagged BAM files using region-parallel processing:
  - footprint: Nucleosome footprints (ns/nl tags) -> BED12 (one line per read)
  - msp: Methylase-sensitive patches (as/al tags) -> BED12 (one line per read)
  - m6a: m6A modification positions (from MM/ML tags) -> BED12 (one line per read)
  - m5c: 5mC modification positions (for DAF-seq) -> BED12 (one line per read)

Output files are named: {dataset}_{type}.bb (bigBed by default)

Usage:
    # Default: extract all types to bigBed in same directory as input
    python extract_tags.py -i tagged.bam

    # Extract only footprints
    python extract_tags.py -i tagged.bam --footprint

    # Extract to specific directory with 8 cores
    python extract_tags.py -i tagged.bam -o output/ -c 8

    # Keep BED files (in addition to bigBed)
    python extract_tags.py -i tagged.bam --keep-bed
"""

import argparse
import os
import sys
import time
import tempfile
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple, Set
import pysam
import numpy as np

from fiberhmm.inference.parallel import _get_genome_regions
from fiberhmm.core.bam_reader import cigar_to_query_ref


def get_chrom_sizes(bam_path: str) -> Dict[str, int]:
    """Extract chromosome sizes from BAM header."""
    if not os.path.exists(bam_path):
        raise FileNotFoundError(f"BAM file not found: {bam_path}")
    if os.path.getsize(bam_path) == 0:
        raise ValueError(
            f"BAM file is empty: {bam_path}\n"
            f"  This usually means the upstream tool (e.g. fiberhmm-apply) "
            f"failed mid-stream. Re-run with -c 1 to surface the underlying error."
        )
    try:
        with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
            return {sq['SN']: sq['LN'] for sq in bam.header['SQ']}
    except ValueError as e:
        raise ValueError(
            f"Could not read BAM {bam_path}: {e}\n"
            f"  The file may be truncated or header-only. If it was produced "
            f"by fiberhmm-apply, that run likely failed mid-stream — re-run "
            f"with -c 1 to surface the underlying error."
        ) from e


# Global worker state
_worker_params = None

def _init_extract_worker(params: dict):
    """Initialize worker with parameters."""
    global _worker_params
    _worker_params = params


def _build_query_to_ref(read):
    """Build query->reference position lookup from CIGAR (numpy int64 array).

    Returns a numpy int64 array indexed by query position.  ``-1`` means
    the query position has no reference mapping (insertion, soft-clip,
    or out-of-range).  Computed once per read and shared across all
    extraction types.

    Use ``_q2r_lookup(arr, qpos)`` to do bounds-checked lookup returning
    None for unmapped positions (drop-in for the old dict.get()).

    ~10× faster than get_aligned_pairs() + dict construction on long reads:
    avoids the 20k+ Python tuple allocations pysam emits per read.
    """
    return cigar_to_query_ref(read)


def _q2r_lookup(query_to_ref, qpos):
    """Dict.get()-compatible lookup on the numpy array returned by
    _build_query_to_ref.  Returns None for unmapped / out-of-range."""
    if 0 <= qpos < len(query_to_ref):
        r = int(query_to_ref[qpos])
        return r if r >= 0 else None
    return None


def _extract_region_worker(args) -> Tuple[dict, int, dict]:
    """Multi-type region worker.

    Iterates through reads in one region, computes the query->ref mapping
    ONCE per read, and dispatches to every requested extractor using the
    cached mapping.  Each extractor writes to its own per-type temp BED.

    This is 2-5× faster than the old single-type worker when multiple
    types are requested (which is the default) because:
      - BAM iteration happens once instead of N times
      - aligned_pairs is computed once per read instead of N times
      - ProcessPool startup cost is paid once instead of N times

    Args: (region, input_bam, {extract_type: temp_bed_path})
    Returns: ({extract_type: temp_bed_path}, n_reads, {extract_type: n_features})
    """
    global _worker_params

    try:
        (chrom, start, end), input_bam, temp_bed_paths = args
        start = int(start)
        end = int(end)

        params = _worker_params
        extract_types = params['extract_types']
        min_tq = int(params.get('min_tq', 50))
        min_mapq = params['min_mapq']
        prob_threshold = params['prob_threshold']
        with_scores = params['with_scores']
        block_scores = bool(params.get('block_scores', False))

        n_reads = 0
        n_features = {t: 0 for t in extract_types}

        pysam.set_verbosity(0)

        # Open per-type output files
        bed_outs = {t: open(temp_bed_paths[t], 'w') for t in extract_types}

        try:
            with pysam.AlignmentFile(input_bam, "rb", check_sq=False) as inbam:
                try:
                    read_iter = inbam.fetch(chrom, start, end)
                except ValueError:
                    return (temp_bed_paths, 0, n_features)

                # Which extract types need aligned_pairs?  All of them except
                # pure tag-presence checks — footprint, msp, tf, m6a, m5c, deam
                # all need query->ref mapping.
                need_mapping = any(t in extract_types for t in
                                   ('footprint', 'msp', 'tf', 'm6a', 'm5c', 'deam'))

                for read in read_iter:
                    if read.is_unmapped or read.is_secondary or read.is_supplementary:
                        continue
                    if read.reference_start < start or read.reference_start >= end:
                        continue
                    if read.mapping_quality < min_mapq:
                        continue

                    n_reads += 1

                    # Build query->ref mapping once, pass to all extractors.
                    # Lazy: only build if at least one selected type needs it
                    # AND the read has any relevant tags (skip for reads with
                    # no annotations at all).
                    query_to_ref = None

                    if 'footprint' in extract_types:
                        if query_to_ref is None:
                            query_to_ref = _build_query_to_ref(read)
                        n_features['footprint'] += _extract_footprints(
                            read, bed_outs['footprint'], with_scores, query_to_ref,
                            block_scores=block_scores)
                    if 'msp' in extract_types:
                        if query_to_ref is None:
                            query_to_ref = _build_query_to_ref(read)
                        n_features['msp'] += _extract_msps(
                            read, bed_outs['msp'], with_scores, query_to_ref,
                            block_scores=block_scores)
                    if 'tf' in extract_types:
                        if query_to_ref is None:
                            query_to_ref = _build_query_to_ref(read)
                        n_features['tf'] += _extract_tfs(
                            read, bed_outs['tf'], with_scores, min_tq, query_to_ref,
                            block_scores=block_scores)
                    if 'm6a' in extract_types:
                        if query_to_ref is None:
                            query_to_ref = _build_query_to_ref(read)
                        n_features['m6a'] += _extract_m6a(
                            read, bed_outs['m6a'], prob_threshold, query_to_ref,
                            block_scores=block_scores)
                    if 'm5c' in extract_types:
                        if query_to_ref is None:
                            query_to_ref = _build_query_to_ref(read)
                        n_features['m5c'] += _extract_m5c(
                            read, bed_outs['m5c'], prob_threshold, query_to_ref,
                            block_scores=block_scores)
                    if 'deam' in extract_types:
                        if query_to_ref is None:
                            query_to_ref = _build_query_to_ref(read)
                        n_features['deam'] += _extract_deam(
                            read, bed_outs['deam'], query_to_ref,
                            block_scores=block_scores,
                            prob_threshold=prob_threshold)
        finally:
            for f in bed_outs.values():
                f.close()

        return (temp_bed_paths, n_reads, n_features)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (temp_bed_paths, 0, {t: 0 for t in (params or {}).get('extract_types', [])})


def _extract_footprints(read, bed_out, with_scores: bool,
                        query_to_ref: Optional[dict] = None,
                        block_scores: bool = False) -> int:
    """Extract footprint intervals from ns/nl tags as BED12 (one line per read).

    When ``block_scores=True``, appends a 13th column of comma-separated
    per-block nq values (int[blockCount] blockNq in the autoSQL schema).
    """
    try:
        ns = read.get_tag('ns')  # Footprint starts (query coords)
        nl = read.get_tag('nl')  # Footprint lengths
    except KeyError:
        return 0

    if len(ns) == 0:
        return 0

    # Need per-block nq if we're emitting the blockNq column even when
    # with_scores is False (the mean-score column 5 still uses 0).
    scores = None
    if with_scores or block_scores:
        try:
            scores = read.get_tag('nq')
        except KeyError:
            pass

    ref_name = read.reference_name
    strand = '-' if read.is_reverse else '+'
    read_id = read.query_name

    if query_to_ref is None:
        query_to_ref = _build_query_to_ref(read)

    blocks = []  # list of (ref_start, ref_end, score)
    for i, (qstart, length) in enumerate(zip(ns, nl)):
        qend = qstart + length

        ref_start = _q2r_lookup(query_to_ref, qstart)
        ref_end = _q2r_lookup(query_to_ref, qend - 1)

        if ref_start is None or ref_end is None:
            continue

        ref_start, ref_end = min(ref_start, ref_end), max(ref_start, ref_end) + 1

        score = 0
        if scores is not None and i < len(scores):
            score = int(scores[i])

        blocks.append((ref_start, ref_end, score))

    if not blocks:
        return 0

    blocks.sort(key=lambda x: x[0])

    chrom_start = blocks[0][0]
    chrom_end = blocks[-1][1]
    block_count = len(blocks)
    block_sizes = ','.join(str(e - s) for s, e, _ in blocks)
    block_starts = ','.join(str(s - chrom_start) for s, _, _ in blocks)

    mean_score = int(sum(sc for _, _, sc in blocks) / len(blocks)) if with_scores else 0

    row = (f"{ref_name}\t{chrom_start}\t{chrom_end}\t{read_id}\t{mean_score}\t{strand}\t"
           f"{chrom_start}\t{chrom_end}\t0\t{block_count}\t{block_sizes}\t{block_starts}")
    if block_scores:
        block_nq = ','.join(str(sc) for _, _, sc in blocks)
        row += f"\t{block_nq}"
    bed_out.write(row + "\n")

    return len(blocks)


def _extract_tfs(read, bed_out, with_scores: bool, min_tq: int,
                 query_to_ref: Optional[dict] = None,
                 block_scores: bool = False) -> int:
    """Extract TF footprints from MA/AQ tags (tf+QQQ annotations) as BED12.

    Reads the spec-compliant MA/AQ tags written by ``fiberhmm-recall-tfs``
    in default (spec) mode. One BED12 row per read with all TF calls as
    blocks; BED score = mean tq across the read's TFs.

    Filters out calls with ``tq < min_tq``. Default ``min_tq = 50`` (LLR
    >= 5 nats) matches ``fiberhmm-recall-tfs``'s default emission floor;
    set to 0 to extract every call, or 100+ for high-confidence only.

    When ``block_scores=True``, appends three columns after blockStarts:
    blockTq, blockEl, blockEr -- the full tf+QQQ quality triplet per call.
    """
    from fiberhmm.io.ma_tags import parse_ma_tag, parse_aq_array

    try:
        ma_str = read.get_tag('MA')
    except KeyError:
        return 0
    try:
        aq = list(read.get_tag('AQ'))
    except KeyError:
        aq = []

    try:
        parsed = parse_ma_tag(ma_str)
    except ValueError:
        return 0

    qual_specs = [rt[2] for rt in parsed['raw_types']]
    n_per_type = [len(rt[3]) for rt in parsed['raw_types']]
    per_annotation = parse_aq_array(aq, qual_specs, n_per_type)

    # Grab full tf+QQQ triplet (tq, el, er) per call.
    tfs_with_quality = []  # (start_0based, length, tq, el, er)
    idx = 0
    for name, _strand, qspec, intervals in parsed['raw_types']:
        for i, (s, l) in enumerate(intervals):
            quals = per_annotation[idx]
            idx += 1
            if name != 'tf':
                continue
            tq = int(quals[0]) if len(quals) >= 1 else 0
            el = int(quals[1]) if len(quals) >= 2 else 0
            er = int(quals[2]) if len(quals) >= 3 else 0
            if tq < min_tq:
                continue
            tfs_with_quality.append((s, l, tq, el, er))

    if not tfs_with_quality:
        return 0

    ref_name = read.reference_name
    strand = '-' if read.is_reverse else '+'
    read_id = read.query_name

    if query_to_ref is None:
        query_to_ref = _build_query_to_ref(read)

    blocks = []  # (ref_start, ref_end, tq, el, er)
    for qstart, length, tq, el, er in tfs_with_quality:
        qend = qstart + length
        ref_start = _q2r_lookup(query_to_ref, qstart)
        ref_end = _q2r_lookup(query_to_ref, qend - 1)
        if ref_start is None or ref_end is None:
            continue
        ref_start, ref_end = min(ref_start, ref_end), max(ref_start, ref_end) + 1
        blocks.append((ref_start, ref_end, tq, el, er))

    if not blocks:
        return 0

    blocks.sort(key=lambda x: x[0])

    chrom_start = blocks[0][0]
    chrom_end = blocks[-1][1]
    block_count = len(blocks)
    block_sizes = ','.join(str(e - s) for s, e, _, _, _ in blocks)
    block_starts = ','.join(str(s - chrom_start) for s, _, _, _, _ in blocks)
    mean_score = int(sum(tq for _, _, tq, _, _ in blocks) / len(blocks)) if with_scores else 0

    row = (f"{ref_name}\t{chrom_start}\t{chrom_end}\t{read_id}\t{mean_score}\t{strand}\t"
           f"{chrom_start}\t{chrom_end}\t0\t{block_count}\t{block_sizes}\t{block_starts}")
    if block_scores:
        block_tq = ','.join(str(b[2]) for b in blocks)
        block_el = ','.join(str(b[3]) for b in blocks)
        block_er = ','.join(str(b[4]) for b in blocks)
        row += f"\t{block_tq}\t{block_el}\t{block_er}"
    bed_out.write(row + "\n")
    return len(blocks)


def _extract_msps(read, bed_out, with_scores: bool,
                  query_to_ref: Optional[dict] = None,
                  block_scores: bool = False) -> int:
    """Extract MSP intervals from as/al tags as BED12 (one line per read).

    When ``block_scores=True``, appends a 13th column of comma-separated
    per-block aq values (int[blockCount] blockAq).
    """
    try:
        as_starts = read.get_tag('as')  # MSP starts (query coords)
        al_lengths = read.get_tag('al')  # MSP lengths
    except KeyError:
        return 0

    if len(as_starts) == 0:
        return 0

    scores = None
    if with_scores or block_scores:
        try:
            scores = read.get_tag('aq')
        except KeyError:
            pass

    ref_name = read.reference_name
    strand = '-' if read.is_reverse else '+'
    read_id = read.query_name

    if query_to_ref is None:
        query_to_ref = _build_query_to_ref(read)

    blocks = []  # list of (ref_start, ref_end, score)
    for i, (qstart, length) in enumerate(zip(as_starts, al_lengths)):
        qend = qstart + length

        ref_start = _q2r_lookup(query_to_ref, qstart)
        ref_end = _q2r_lookup(query_to_ref, qend - 1)

        if ref_start is None or ref_end is None:
            continue

        ref_start, ref_end = min(ref_start, ref_end), max(ref_start, ref_end) + 1

        score = 0
        if scores is not None and i < len(scores):
            score = int(scores[i])

        blocks.append((ref_start, ref_end, score))

    if not blocks:
        return 0

    blocks.sort(key=lambda x: x[0])

    chrom_start = blocks[0][0]
    chrom_end = blocks[-1][1]
    block_count = len(blocks)
    block_sizes = ','.join(str(e - s) for s, e, _ in blocks)
    block_starts = ','.join(str(s - chrom_start) for s, _, _ in blocks)

    mean_score = int(sum(sc for _, _, sc in blocks) / len(blocks)) if with_scores else 0

    row = (f"{ref_name}\t{chrom_start}\t{chrom_end}\t{read_id}\t{mean_score}\t{strand}\t"
           f"{chrom_start}\t{chrom_end}\t0\t{block_count}\t{block_sizes}\t{block_starts}")
    if block_scores:
        block_aq = ','.join(str(sc) for _, _, sc in blocks)
        row += f"\t{block_aq}"
    bed_out.write(row + "\n")

    return len(blocks)


def _extract_m6a(read, bed_out, prob_threshold: int, query_to_ref=None,
                 block_scores: bool = False) -> int:
    """Extract m6A positions from MM/ML tags as BED12 (one line per read).

    When ``block_scores=True``, appends a 13th column of comma-separated
    per-position ML values (int[blockCount] blockMl).
    """
    try:
        if read.has_tag('ML') and len(read.get_tag('ML')) == 0:
            return 0
    except Exception:
        pass
    try:
        mod_bases = read.modified_bases
        if not mod_bases:
            return 0
    except (KeyError, TypeError):
        return 0

    ref_name = read.reference_name
    strand = '-' if read.is_reverse else '+'
    read_id = read.query_name

    aligned_pairs = query_to_ref if query_to_ref is not None else _build_query_to_ref(read)
    n = len(aligned_pairs)

    positions_list = []  # list of (ref_pos, score)
    for (base, strand_code, mod_type), positions in mod_bases.items():
        if mod_type != 'a':
            continue
        for query_pos, prob in positions:
            if prob < prob_threshold:
                continue
            if query_pos < 0 or query_pos >= n:
                continue
            ref_pos = int(aligned_pairs[query_pos])
            if ref_pos < 0:
                continue
            positions_list.append((ref_pos, int(prob)))

    if not positions_list:
        return 0

    positions_list.sort(key=lambda x: x[0])

    chrom_start = positions_list[0][0]
    chrom_end = positions_list[-1][0] + 1
    block_count = len(positions_list)
    block_sizes = ','.join('1' for _ in positions_list)
    block_starts = ','.join(str(pos - chrom_start) for pos, _ in positions_list)

    mean_score = int(sum(sc for _, sc in positions_list) / len(positions_list))

    row = (f"{ref_name}\t{chrom_start}\t{chrom_end}\t{read_id}\t{mean_score}\t{strand}\t"
           f"{chrom_start}\t{chrom_end}\t0\t{block_count}\t{block_sizes}\t{block_starts}")
    if block_scores:
        block_ml = ','.join(str(sc) for _, sc in positions_list)
        row += f"\t{block_ml}"
    bed_out.write(row + "\n")

    return len(positions_list)


def _extract_m5c(read, bed_out, prob_threshold: int, query_to_ref=None,
                 block_scores: bool = False) -> int:
    """Extract 5mC positions from MM/ML tags as BED12 (one line per read).

    When ``block_scores=True``, appends a 13th column of comma-separated
    per-position ML values (int[blockCount] blockMl).
    """
    try:
        if read.has_tag('ML') and len(read.get_tag('ML')) == 0:
            return 0
    except Exception:
        pass
    try:
        mod_bases = read.modified_bases
        if not mod_bases:
            return 0
    except (KeyError, TypeError):
        return 0

    ref_name = read.reference_name
    strand = '-' if read.is_reverse else '+'
    read_id = read.query_name

    aligned_pairs = query_to_ref if query_to_ref is not None else _build_query_to_ref(read)
    n = len(aligned_pairs)

    positions_list = []
    for (base, strand_code, mod_type), positions in mod_bases.items():
        if mod_type != 'm':
            continue
        for query_pos, prob in positions:
            if prob < prob_threshold:
                continue
            if query_pos < 0 or query_pos >= n:
                continue
            ref_pos = int(aligned_pairs[query_pos])
            if ref_pos < 0:
                continue
            positions_list.append((ref_pos, int(prob)))

    if not positions_list:
        return 0

    positions_list.sort(key=lambda x: x[0])

    chrom_start = positions_list[0][0]
    chrom_end = positions_list[-1][0] + 1
    block_count = len(positions_list)
    block_sizes = ','.join('1' for _ in positions_list)
    block_starts = ','.join(str(pos - chrom_start) for pos, _ in positions_list)

    mean_score = int(sum(sc for _, sc in positions_list) / len(positions_list))

    row = (f"{ref_name}\t{chrom_start}\t{chrom_end}\t{read_id}\t{mean_score}\t{strand}\t"
           f"{chrom_start}\t{chrom_end}\t0\t{block_count}\t{block_sizes}\t{block_starts}")
    if block_scores:
        block_ml = ','.join(str(sc) for _, sc in positions_list)
        row += f"\t{block_ml}"
    bed_out.write(row + "\n")

    return len(positions_list)


def _extract_deam(read, bed_out, query_to_ref=None,
                  block_scores: bool = False,
                  prob_threshold: int = 0) -> int:
    """Extract DAF-seq deamination calls as BED12 (one row per read).

    Matches FiberBrowser's BAM parsing priority: first the MM/ML-native
    ``u`` (single-letter dU) or ChEBI 55797 numeric mod code, then the
    IUPAC R/Y fallback written by ``fiberhmm-daf-encode``. First
    non-empty source wins so events never get double-counted.

    ``blockMod`` flavor convention (matches FiberBrowser):
      0 = R (G -> U on the GA strand)
      1 = Y (C -> U on the CT strand)

    For the MM/ML-native path, ``prob_threshold`` (default 0 = accept
    everything) filters by the per-call ML probability, same knob as
    ``-p/--prob-threshold`` on m6a/m5c.

    BED score column 5 is constant ``255`` (IUPAC calls are deterministic;
    MM/ML calls can be filtered up front via prob_threshold).
    """
    ref_name = read.reference_name
    strand = '-' if read.is_reverse else '+'
    read_id = read.query_name

    aligned_pairs = query_to_ref if query_to_ref is not None else _build_query_to_ref(read)
    n = len(aligned_pairs)

    positions_list = []  # (ref_pos, code) where code: 0=R, 1=Y

    # --- Priority 1: MM/ML-native 'u' or ChEBI 55797 --------------------
    try:
        if read.has_tag('ML') and len(read.get_tag('ML')) == 0:
            mod_bases = {}
        else:
            mod_bases = read.modified_bases or {}
    except (KeyError, TypeError, Exception):
        mod_bases = {}

    for (base, _strand_code, mod_type), positions in mod_bases.items():
        # Single-letter 'u' (SAM spec) OR numeric 55797 (ChEBI for dU).
        if mod_type != 'u' and mod_type != 55797:
            continue
        # Flavor from the canonical base: C->U = 1 (Y/CT-dea),
        # G->U = 0 (R/GA-dea).  pysam uppercases; be defensive anyway.
        b = base.upper() if isinstance(base, str) else chr(base).upper()
        if b == 'C':
            flavor = 1
        elif b == 'G':
            flavor = 0
        else:
            continue
        for query_pos, prob in positions:
            if prob < prob_threshold:
                continue
            if query_pos < 0 or query_pos >= n:
                continue
            ref_pos = int(aligned_pairs[query_pos])
            if ref_pos < 0:
                continue
            positions_list.append((ref_pos, flavor))

    # --- Priority 2: IUPAC R/Y in the query sequence --------------------
    if not positions_list:
        seq = read.query_sequence
        if seq:
            arr = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
            r_mask = (arr == ord('R'))
            y_mask = (arr == ord('Y'))
            mod_mask = r_mask | y_mask
            if mod_mask.any():
                for q_pos in np.where(mod_mask)[0]:
                    qi = int(q_pos)
                    if qi >= n:
                        continue
                    ref_pos = int(aligned_pairs[qi])
                    if ref_pos < 0:
                        continue
                    flavor = 1 if y_mask[qi] else 0
                    positions_list.append((ref_pos, flavor))

    # --- Priority 3: ref mismatch via MD tag ----------------------------
    # Fallback for raw DAF BAMs that were neither MM/ML-tagged nor IUPAC-
    # encoded. C->T mismatch on the ref = flavor 1 (Y/CT-dea),
    # G->A mismatch = flavor 0 (R/GA-dea). Slow (scans whole alignment)
    # so we only run it when the cheaper paths are empty and the read
    # actually carries an MD tag.
    if not positions_list and read.has_tag('MD'):
        seq = read.query_sequence
        if seq:
            try:
                pairs = read.get_aligned_pairs(with_seq=True)
            except Exception:
                # pysam raises AssertionError (not ValueError) on malformed
                # MD/CIGAR mismatches, plus ValueError on bad MD strings. Skip
                # the read entirely instead of crashing the worker.
                pairs = None
            if pairs:
                for qpos, rpos, ref_base in pairs:
                    if qpos is None or rpos is None or ref_base is None:
                        continue
                    ref_up = ref_base.upper()
                    q_up = seq[qpos].upper() if qpos < len(seq) else ''
                    # Only deamination-direction mismatches qualify.
                    if ref_up == 'C' and q_up == 'T':
                        positions_list.append((int(rpos), 1))
                    elif ref_up == 'G' and q_up == 'A':
                        positions_list.append((int(rpos), 0))

    if not positions_list:
        return 0

    positions_list.sort(key=lambda x: x[0])

    chrom_start = positions_list[0][0]
    chrom_end = positions_list[-1][0] + 1
    block_count = len(positions_list)
    block_sizes = ','.join('1' for _ in positions_list)
    block_starts = ','.join(str(pos - chrom_start) for pos, _ in positions_list)

    row = (f"{ref_name}\t{chrom_start}\t{chrom_end}\t{read_id}\t255\t{strand}\t"
           f"{chrom_start}\t{chrom_end}\t0\t{block_count}\t{block_sizes}\t{block_starts}")
    if block_scores:
        block_mod = ','.join(str(code) for _, code in positions_list)
        row += f"\t{block_mod}"
    bed_out.write(row + "\n")

    return len(positions_list)


def extract_tags_parallel(input_bam: str, output_beds, extract_types,
                          n_cores: int = 1, region_size: int = 10_000_000,
                          min_mapq: int = 0, prob_threshold: int = 125,
                          with_scores: bool = True,
                          min_tq: int = 50,
                          block_scores: bool = False,
                          skip_scaffolds: bool = False,
                          chroms: Optional[Set[str]] = None):
    """Extract one or more tag types from BAM in a single region-parallel pass.

    All requested extract types share a single BAM traversal + a single
    query->ref mapping computation per read, which is 2-5× faster than
    running extract N times when N types are requested (the default).

    Args:
        output_beds: dict mapping extract_type -> output bed path.  The old
            signature (single str for one type) is also accepted for back
            compat — in that case extract_types is a string.
        extract_types: list of extract types to process, or a single string.

    Returns: (n_reads_processed, {extract_type: n_features})
    """
    start_time = time.time()

    # Back-compat: accept single-type string args
    if isinstance(extract_types, str):
        extract_types = [extract_types]
    if isinstance(output_beds, str):
        if len(extract_types) != 1:
            raise ValueError("output_beds as string requires exactly one extract_type")
        output_beds = {extract_types[0]: output_beds}
    extract_types = list(extract_types)

    # Check BAM index
    if not os.path.exists(input_bam + '.bai') and not os.path.exists(input_bam.replace('.bam', '.bai')):
        print("Indexing input BAM...")
        pysam.index(input_bam)

    # Get regions
    regions = _get_genome_regions(input_bam, region_size, skip_scaffolds, chroms)
    print(f"Processing {len(regions)} regions with {n_cores} cores "
          f"(types: {', '.join(extract_types)})...")

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix='extract_tags_')

    try:
        params = {
            'extract_types': extract_types,
            'min_mapq': min_mapq,
            'prob_threshold': prob_threshold,
            'with_scores': with_scores,
            'min_tq': min_tq,
            'block_scores': block_scores,
        }

        # Work items: per region, a dict of {type: temp_bed_path}
        work_items = []
        for i, region in enumerate(regions):
            per_type_beds = {
                t: os.path.join(temp_dir, f'region_{i:06d}_{t}.bed')
                for t in extract_types
            }
            work_items.append((region, input_bam, per_type_beds))

        total_reads = 0
        total_features = {t: 0 for t in extract_types}
        # {extract_type: [(region_idx, temp_bed_path)]}
        temp_beds_by_type = {t: [] for t in extract_types}
        completed = 0

        with ProcessPoolExecutor(
            max_workers=n_cores,
            initializer=_init_extract_worker,
            initargs=(params,)
        ) as executor:
            futures = {executor.submit(_extract_region_worker, item): i
                      for i, item in enumerate(work_items)}

            for future in as_completed(futures):
                completed += 1
                try:
                    temp_bed_paths, n_reads, n_feats = future.result()
                    total_reads += n_reads
                    for t in extract_types:
                        total_features[t] += n_feats.get(t, 0)
                        tb = temp_bed_paths.get(t)
                        if tb and os.path.exists(tb) and os.path.getsize(tb) > 0:
                            temp_beds_by_type[t].append((futures[future], tb))
                except Exception as e:
                    print(f"Worker error: {e}")

                elapsed = time.time() - start_time
                rate = total_reads / elapsed if elapsed > 0 else 0
                feat_str = ' '.join(f"{t}={total_features[t]:,}" for t in extract_types)
                print(f"\r  Regions: {completed}/{len(regions)} | Reads: {total_reads:,} | {feat_str} | {rate:.0f} reads/s", end='')
        print()

        # Concatenate + sort per type
        for t in extract_types:
            beds = sorted(temp_beds_by_type[t], key=lambda x: x[0])
            out_path = output_beds[t]
            print(f"  [{t}] concatenating {len(beds)} region BEDs...")
            with open(out_path, 'w') as outf:
                for _, tb in beds:
                    with open(tb, 'r') as inf:
                        shutil.copyfileobj(inf, outf)
            if os.path.getsize(out_path) > 0:
                print(f"  [{t}] sorting BED...")
                sorted_bed = out_path + '.sorted'
                subprocess.run(['sort', '-k1,1', '-k2,2n', out_path, '-o', sorted_bed],
                               check=True)
                os.replace(sorted_bed, out_path)

        elapsed = time.time() - start_time
        feat_summary = ', '.join(f"{t}: {total_features[t]:,}" for t in extract_types)
        print(f"Completed in {elapsed:.1f}s: {total_reads:,} reads -> {feat_summary}")

        return total_reads, total_features

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def bed_to_bigbed(bed_path: str, bigbed_path: str, chrom_sizes: Dict[str, int],
                   bed_type: str = 'bed12',
                   extract_type: Optional[str] = None,
                   block_scores: bool = False,
                   sample_name: Optional[str] = None) -> bool:
    """Convert BED to bigBed using bedToBigBed.

    If ``extract_type`` matches a FiberHMM autoSQL schema
    (footprint/msp/tf/m6a/m5c), embed the schema via ``-as=`` so the
    bigBed is self-identifying -- browsers will see the table name
    (e.g. ``fiberhmm_tf``) and description when the file is loaded.

    When ``block_scores=True`` the matching BED12+N schema is used and
    the bedToBigBed ``-type`` flag is set to ``bed12+N`` (N depends on
    extract_type; see fiberhmm.io.autosql.EXTRA_FIELD_COUNTS).

    Args:
        bed_path: Input BED file
        bigbed_path: Output bigBed file
        chrom_sizes: Dict of chromosome sizes
        bed_type: 'bed12' for all FiberHMM outputs
        extract_type: One of footprint/msp/tf/m6a/m5c to embed the
            matching autoSQL schema; None to skip schema embedding.
        block_scores: If True, expect BED12+N input (per-block quality
            columns appended) and use the matching autoSQL variant.
    """
    from fiberhmm.io.autosql import write_autosql_for, EXTRA_FIELD_COUNTS

    sizes_file = bed_path + '.sizes'
    with open(sizes_file, 'w') as f:
        for chrom, size in sorted(chrom_sizes.items()):
            f.write(f"{chrom}\t{size}\n")

    as_file = (write_autosql_for(extract_type, block_scores=block_scores,
                                  sample_name=sample_name)
               if extract_type else None)

    n_extra = EXTRA_FIELD_COUNTS.get(extract_type, 0) if block_scores else 0

    try:
        cmd = ['bedToBigBed']
        if as_file:
            cmd.append(f'-as={as_file}')
        if bed_type == 'bed12':
            type_flag = f'-type=bed12+{n_extra}' if n_extra > 0 else '-type=bed12'
            cmd.append(type_flag)
        cmd.extend([bed_path, sizes_file, bigbed_path])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"bedToBigBed error: {result.stderr}")
            return False

        return True

    except FileNotFoundError:
        print("Warning: bedToBigBed not found in PATH. Skipping bigBed conversion.")
        print("Install with: conda install -c bioconda ucsc-bedtobigbed")
        return False

    finally:
        if as_file and os.path.exists(as_file):
            try: os.remove(as_file)
            except Exception: pass
        if os.path.exists(sizes_file):
            try:
                os.remove(sizes_file)
            except (PermissionError, OSError) as e:
                print(f"  Warning: Could not remove temp file {sizes_file}: {e}")


def diagnose_bam_tags(input_bam: str, n_reads: int = 20) -> Dict[str, object]:
    """Sniff the first N mapped reads and report which sources each
    extract type can work from. Reports per-source presence counts
    plus a ``summary`` string for immediate user feedback.

    Matches FiberBrowser's diagnose_bam_tags detection priority so
    the user can predict which tracks will populate before the heavy
    pass runs.
    """
    import re
    counts = {
        'reads_scanned': 0,
        # Footprint / MSP / TF tags
        'has_ns_nl': 0, 'has_as_al': 0, 'has_MA_AQ': 0,
        # Modified-base tags
        'has_MM': 0, 'has_ML': 0,
        'mm_subtypes': set(),
        # DAF-specific
        'has_ry_in_seq': 0, 'has_md_only': 0,
    }
    mm_re = re.compile(r'([ACGTUN])([+-])([a-z0-9]+|\d+)', re.IGNORECASE)

    try:
        with pysam.AlignmentFile(input_bam, 'rb', check_sq=False) as bam:
            for read in bam:
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                counts['reads_scanned'] += 1

                if read.has_tag('ns') and read.has_tag('nl'):
                    counts['has_ns_nl'] += 1
                if read.has_tag('as') and read.has_tag('al'):
                    counts['has_as_al'] += 1
                if read.has_tag('MA') and read.has_tag('AQ'):
                    counts['has_MA_AQ'] += 1

                mm_present = read.has_tag('MM') or read.has_tag('Mm')
                ml_present = read.has_tag('ML') or read.has_tag('Ml')
                if mm_present:
                    counts['has_MM'] += 1
                    try:
                        mm_str = (read.get_tag('MM') if read.has_tag('MM')
                                  else read.get_tag('Mm'))
                        for m in mm_re.finditer(mm_str):
                            counts['mm_subtypes'].add(
                                f"{m.group(1).upper()}{m.group(2)}{m.group(3)}")
                    except Exception:
                        pass
                if ml_present:
                    counts['has_ML'] += 1

                seq = read.query_sequence
                if seq and ('R' in seq or 'Y' in seq):
                    counts['has_ry_in_seq'] += 1

                has_md = read.has_tag('MD')
                has_mod_source = mm_present or (seq and ('R' in seq or 'Y' in seq))
                if has_md and not has_mod_source:
                    counts['has_md_only'] += 1

                if counts['reads_scanned'] >= n_reads:
                    break
    except (ValueError, OSError):
        pass

    n = counts['reads_scanned']
    counts['mm_subtypes'] = sorted(counts['mm_subtypes'])

    if n == 0:
        counts['summary'] = "no mapped reads in first sniff — extract will be empty"
        return counts

    def frac(k):
        return f"{k}/{n}"

    parts = []
    if counts['has_ns_nl']:
        parts.append(f"ns/nl={frac(counts['has_ns_nl'])}")
    if counts['has_as_al']:
        parts.append(f"as/al={frac(counts['has_as_al'])}")
    if counts['has_MA_AQ']:
        parts.append(f"MA/AQ={frac(counts['has_MA_AQ'])}")
    if counts['has_MM']:
        sub = (' [' + ','.join(counts['mm_subtypes']) + ']'
               if counts['mm_subtypes'] else '')
        parts.append(f"MM/ML={frac(counts['has_MM'])}{sub}")
    if counts['has_ry_in_seq']:
        parts.append(f"R/Y-in-seq={frac(counts['has_ry_in_seq'])}")
    if counts['has_md_only']:
        parts.append(f"MD-only={frac(counts['has_md_only'])}")

    counts['summary'] = (', '.join(parts) if parts
                         else 'no FiberHMM / modification tags detected '
                              f'in first {n} mapped reads')
    return counts


def _print_tag_diagnostic(diag: Dict[str, object], extract_types: list) -> None:
    """Print a short summary of detected tags + the tracks they'll populate."""
    print(f"Tag scan (first {diag['reads_scanned']} mapped reads): {diag['summary']}")

    # Predict which tracks will have content.
    predicted = {}
    if diag['has_ns_nl']: predicted['footprint'] = 'ns/nl'
    if diag['has_as_al']: predicted['msp'] = 'as/al'
    if diag['has_MA_AQ']: predicted['tf'] = 'MA/AQ tf+ annotations'
    if diag['has_MM'] and any('a' in s.lower() for s in diag['mm_subtypes']):
        predicted['m6a'] = 'MM/ML (A+a)'
    if diag['has_MM'] and any('+m' in s.lower() or '-m' in s.lower()
                               for s in diag['mm_subtypes']):
        predicted['m5c'] = 'MM/ML (C+m)'
    # deam: any of MM/ML u, R/Y in seq, or MD-only
    mm_has_u = any(('+u' in s.lower() or '+55797' in s)
                   for s in diag['mm_subtypes'])
    if mm_has_u:
        predicted['deam'] = 'MM/ML (u / 55797)'
    elif diag['has_ry_in_seq']:
        predicted['deam'] = 'R/Y in sequence'
    elif diag['has_md_only']:
        predicted['deam'] = 'MD-only (path-3 fallback)'

    expected = []
    empty = []
    for t in extract_types:
        if t in predicted:
            expected.append(f"{t}[{predicted[t]}]")
        else:
            empty.append(t)
    if expected:
        print(f"  will populate: {', '.join(expected)}")
    if empty:
        print(f"  will be empty: {', '.join(empty)}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Extract tags from FiberHMM-tagged BAMs to BED12/bigBed',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: extract all types to bigBed in same directory as input
    python extract_tags.py -i tagged.bam

    # Extract only footprints
    python extract_tags.py -i tagged.bam --footprint

    # Extract to specific directory with 8 cores
    python extract_tags.py -i tagged.bam -o output/ -c 8

    # Keep BED files (in addition to bigBed)
    python extract_tags.py -i tagged.bam --keep-bed
        """
    )

    parser.add_argument('-i', '--input', required=True, help='Input tagged BAM file')
    parser.add_argument('-o', '--outdir', default=None, help='Output directory (default: same as input)')
    parser.add_argument('-c', '--cores', type=int, default=1, help='Number of CPU cores')

    # Tag types (default: all)
    parser.add_argument('--footprint', action='store_true', help='Extract footprints (ns/nl tags)')
    parser.add_argument('--msp', action='store_true', help='Extract MSPs (as/al tags)')
    parser.add_argument('--tf', action='store_true',
                        help='Extract TF/Pol II footprints from MA/AQ tag (tf+QQQ). '
                             'Requires a BAM produced by fiberhmm-recall-tfs in '
                             'default (spec) mode. See --min-tq for the quality floor.')
    parser.add_argument('--min-tq', type=int, default=50,
                        help='Minimum TF quality (tq) to extract. 0-255 scale where '
                             'tq = round(LLR * 10). Default 50 (LLR >= 5 nats, ~148:1 '
                             'likelihood ratio) matches fiberhmm-recall-tfs\'s default '
                             'emission floor. Set to 0 for every call, 100+ for '
                             'high-confidence only.')
    parser.add_argument('--m6a', action='store_true', help='Extract m6A positions')
    parser.add_argument('--m5c', action='store_true', help='Extract 5mC positions (DAF-seq)')
    parser.add_argument('--deam', action='store_true',
                        help='Extract DAF-seq deamination calls. Priority: '
                             '(1) MM/ML-native dU calls (mod code "u" or ChEBI 55797); '
                             '(2) IUPAC R/Y codes in the query sequence '
                             '(fiberhmm-daf-encode output); '
                             '(3) MD-tag ref mismatches as a fallback for raw DAF BAMs. '
                             'First non-empty source wins per read. blockMod: '
                             '0 = R/GA-dea, 1 = Y/CT-dea, matching FiberBrowser flavor codes.')
    parser.add_argument('--all', action='store_true', help='Extract all tag types (default if none specified)')

    # Output options (default: bigbed)
    parser.add_argument('--bed-only', action='store_true', help='Output BED only (no bigBed)')
    parser.add_argument('--keep-bed', action='store_true', help='Keep BED files when creating bigBed')
    # Legacy flag for compatibility
    parser.add_argument('--bigbed', action='store_true', help=argparse.SUPPRESS)

    # Filtering
    parser.add_argument('-q', '--min-mapq', type=int, default=0, help='Min mapping quality (default: 0, no filtering)')
    parser.add_argument('-p', '--prob-threshold', type=int, default=125,
                        help='Min probability for m6a/m5c (0-255)')
    parser.add_argument('--no-scores', action='store_true', help='Omit scores from output')
    parser.add_argument('--block-scores', action='store_true',
                        help='Append per-block quality as extra BED column(s) (BED12+N). '
                             'footprint -> blockNq, msp -> blockAq, m6a/m5c -> blockMl, '
                             'tf -> blockTq/blockEl/blockEr. bigBed uses -type=bed12+N and '
                             'the matching autoSQL schema so FiberBrowser/UCSC can surface '
                             'per-feature quality without a sidecar database.')
    parser.add_argument('--sample-name', default=None,
                        help='Sample/dataset identifier to embed in the autoSQL '
                             'description of every output bigBed ("Sample: <name>. ..."). '
                             'Default: BAM basename stem. Lets downstream tools match '
                             'a bigBed to its source without filename parsing.')

    # Region options
    parser.add_argument('--region-size', type=int, default=10_000_000, help='Region size for parallel')
    parser.add_argument('--skip-scaffolds', action='store_true', help='Skip scaffold chromosomes')
    parser.add_argument('--chroms', type=str, help='Comma-separated chromosomes to process')

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Default output directory to input file's directory
    if args.outdir is None:
        args.outdir = os.path.dirname(os.path.abspath(args.input))
        if not args.outdir:
            args.outdir = '.'

    # Determine what to extract (default: all)
    extract_types = []
    any_selected = (args.footprint or args.msp or args.tf or
                    args.m6a or args.m5c or args.deam)
    if args.all or not any_selected:
        extract_types = ['footprint', 'msp', 'tf', 'm6a', 'm5c', 'deam']
    else:
        if args.footprint:
            extract_types.append('footprint')
        if args.msp:
            extract_types.append('msp')
        if args.tf:
            extract_types.append('tf')
        if args.m6a:
            extract_types.append('m6a')
        if args.m5c:
            extract_types.append('m5c')
        if args.deam:
            extract_types.append('deam')

    # Default to bigbed unless --bed-only specified
    make_bigbed = not args.bed_only

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Parse chromosome filter
    chroms = None
    if args.chroms:
        chroms = set(args.chroms.split(','))

    # Get dataset name (file prefix + default sample_name).
    dataset = os.path.basename(args.input).replace('.bam', '').replace('_footprints', '')
    sample_name = args.sample_name or dataset

    # Get chrom sizes — only required for bigBed conversion
    chrom_sizes = get_chrom_sizes(args.input) if make_bigbed else {}

    print(f"Input: {args.input}")
    print(f"Output: {args.outdir}")
    print(f"Extract types: {', '.join(extract_types)}")
    print(f"Cores: {args.cores}")
    print()

    # Quick tag sniff so the user knows which tracks will populate.
    diag = diagnose_bam_tags(args.input, n_reads=20)
    _print_tag_diagnostic(diag, extract_types)

    # Single region-parallel pass for ALL requested types.  Each worker
    # opens the BAM once, iterates once, builds the query->ref mapping
    # once per read, and writes to N per-type output BEDs.  N× faster
    # than running the old one-type-at-a-time loop.
    output_beds = {
        t: os.path.join(args.outdir, f"{dataset}_{t}.bed") for t in extract_types
    }
    bb_paths = {
        t: os.path.join(args.outdir, f"{dataset}_{t}.bb") for t in extract_types
    }
    print(f"=== Extracting {', '.join(extract_types)} (single pass) ===")
    n_reads, n_features = extract_tags_parallel(
        input_bam=args.input,
        output_beds=output_beds,
        extract_types=extract_types,
        n_cores=args.cores,
        region_size=args.region_size,
        min_mapq=args.min_mapq,
        prob_threshold=args.prob_threshold,
        with_scores=not args.no_scores,
        min_tq=args.min_tq,
        block_scores=args.block_scores,
        skip_scaffolds=args.skip_scaffolds,
        chroms=chroms,
    )

    print()
    for extract_type in extract_types:
        feats = n_features.get(extract_type, 0)
        bed_path = output_beds[extract_type]
        bb_path = bb_paths[extract_type]

        if feats == 0:
            print(f"  [{extract_type}] no features, skipping")
            if os.path.exists(bed_path):
                try:
                    os.remove(bed_path)
                except (PermissionError, OSError):
                    pass
            continue

        print(f"  [{extract_type}] BED: {bed_path}")

        if make_bigbed:
            print(f"  [{extract_type}] converting to bigBed...")
            if bed_to_bigbed(bed_path, bb_path, chrom_sizes, 'bed12',
                              extract_type=extract_type,
                              block_scores=args.block_scores,
                              sample_name=sample_name):
                print(f"  [{extract_type}] bigBed: {bb_path}")
                if not args.keep_bed:
                    try:
                        os.remove(bed_path)
                    except (PermissionError, OSError) as e:
                        print(f"  Warning: Could not remove BED file: {e}")

    print("\nDone!")


if __name__ == '__main__':
    main()
