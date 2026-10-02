"""Fast, bounded QC for FiberHMM-compatible BAM files.

The sampler never walks an entire large BAM. Coordinate-indexed files are
sampled through deterministic random genomic windows; unindexed files use a
reservoir capped at a fixed number of examined records.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, TextIO

import numpy as np
import pysam

from fiberhmm.core.bam_reader import (
    extract_daf_iupac_positions,
    parse_mm_tag_query_calls,
    parse_mm_tag_query_positions,
)
from fiberhmm.io.ma_tags import parse_ma_tag
from fiberhmm.qc.states import (
    DEFAULT_LIGHT_CALL_READS,
    DEFAULT_LIGHT_CALL_SECONDS,
    DEFAULT_MIN_MSP_BP,
    compute_state_rates,
    grade_state_rates,
)

MAX_LAG = 1000
PLOT_MAX_LAG = 800
PERIOD_BAND = (160, 220)
DEFAULT_SAMPLE_READS = 2000
DEFAULT_SEED = 20260824
QC_MODES = ("daf", "pacbio-fiber", "nanopore-fiber")
QC_ENZYMES = ("hia5", "ecogii", "ddda", "dddb", "sssi")


class QCInputError(ValueError):
    """A user-facing QC input problem (unreadable BAM, incompatible assay or
    reference profile): command-line tools report it in one line, exit 2."""


@dataclass
class SampledReads:
    reads: list
    strategy: str
    records_examined: int
    windows_examined: int = 0
    # Stream sampling only: unmapped primary records and aligned records seen,
    # and whether the scan reached the end of the file.
    unmapped_primary: int = 0
    aligned_records: int = 0
    reached_eof: bool = False


def _eligible(read, min_mapq: int, allow_unmapped: bool = False) -> bool:
    """Primary record with a sequence; aligned ones need ``min_mapq``.

    Unmapped records qualify only with ``allow_unmapped`` (a BAM with no
    aligned records, e.g. calls made on an unaligned BAM): the labelling and
    periodicity metrics work in read coordinates, and an unmapped record has
    no meaningful MAPQ.
    """
    if read.is_secondary or read.is_supplementary or read.query_sequence is None:
        return False
    if read.is_unmapped:
        return allow_unmapped
    return read.mapping_quality >= min_mapq


def _read_key(read) -> str:
    return "|".join(
        (
            str(read.query_name),
            str(read.flag),
            str(read.reference_id),
            str(read.reference_start),
            str(read.cigarstring),
        )
    )


def _stable_rank(key: str, seed: int) -> int:
    digest = hashlib.blake2b(
        f"{seed}|{key}".encode("utf-8", errors="replace"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def _bounded_stream_sample(
    path: str,
    sample_reads: int,
    seed: int,
    min_mapq: int,
    scan_limit: int,
    allow_unmapped: bool = False,
) -> SampledReads:
    """Reservoir-sample a bounded prefix of an unindexed/unsorted BAM."""
    rng = np.random.default_rng(seed)
    reservoir: list = []
    eligible_seen = 0
    records_examined = 0
    unmapped_primary = 0
    aligned_records = 0
    reached_eof = True
    with pysam.AlignmentFile(path, "rb", check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            if records_examined >= scan_limit:
                reached_eof = False
                break
            records_examined += 1
            if not read.is_unmapped:
                aligned_records += 1
            elif not (read.is_secondary or read.is_supplementary):
                unmapped_primary += 1
            if _eligible(read, min_mapq, allow_unmapped):
                eligible_seen += 1
                if len(reservoir) < sample_reads:
                    reservoir.append(read)
                else:
                    index = int(rng.integers(0, eligible_seen))
                    if index < sample_reads:
                        reservoir[index] = read
    return SampledReads(
        reads=reservoir,
        strategy=f"bounded reservoir (first <= {scan_limit:,} records)",
        records_examined=records_examined,
        unmapped_primary=unmapped_primary,
        aligned_records=aligned_records,
        reached_eof=reached_eof,
    )


def _no_aligned_records(sampled: SampledReads) -> bool:
    """The whole file was read and holds unmapped primaries, no aligned record."""
    return (sampled.reached_eof and sampled.unmapped_primary > 0
            and sampled.aligned_records == 0)


def _unaligned_stream_sample(path, sample_reads, seed, scan_limit) -> SampledReads:
    sampled = _bounded_stream_sample(
        path, sample_reads, seed, 0, scan_limit, allow_unmapped=True
    )
    sampled.strategy += "; unaligned reads (the BAM has no aligned records)"
    return sampled


def sample_bam_reads(
    path: str,
    sample_reads: int = DEFAULT_SAMPLE_READS,
    seed: int = DEFAULT_SEED,
    min_mapq: int = 20,
    scan_limit: Optional[int] = None,
) -> SampledReads:
    """Take a deterministic, bounded read sample without a whole-file pass."""
    sample_reads = max(1, int(sample_reads))
    scan_limit = int(scan_limit or max(10_000, sample_reads * 10))

    try:
        with pysam.AlignmentFile(path, "rb", check_sq=False) as bam:
            indexed = bool(bam.has_index())
            references = list(bam.references)
            lengths = np.asarray(bam.lengths, dtype=np.int64)
            index_says_unaligned = False
            if indexed:
                try:
                    index_says_unaligned = bam.mapped == 0 and bam.unmapped > 0
                except (ValueError, AttributeError):
                    index_says_unaligned = False
    except (OSError, ValueError) as exc:
        raise QCInputError(f"cannot open BAM/CRAM {path!r}: {exc}") from exc

    if not references or index_says_unaligned:
        # No @SQ, or an index counting no aligned record: an unaligned BAM
        # (e.g. fiberhmm-call on a uBAM).
        return _unaligned_stream_sample(path, sample_reads, seed, scan_limit)
    usable = lengths > 0
    if not indexed or not np.any(usable):
        sampled = _bounded_stream_sample(
            path, sample_reads, seed, min_mapq, scan_limit
        )
        if not sampled.reads and _no_aligned_records(sampled):
            return _unaligned_stream_sample(path, sample_reads, seed, scan_limit)
        return sampled

    references = [name for name, keep in zip(references, usable) if keep]
    lengths = lengths[usable]
    cumulative = np.cumsum(lengths)
    genome_size = int(cumulative[-1])
    rng = np.random.default_rng(seed)
    per_window = 24
    window_bp = 20_000
    n_windows = min(512, max(64, int(math.ceil(sample_reads / per_window)) * 2))
    candidates: dict[str, tuple[int, object]] = {}
    records_examined = 0
    windows_examined = 0

    with pysam.AlignmentFile(path, "rb", check_sq=False) as bam:
        for _ in range(n_windows):
            genome_pos = int(rng.integers(0, genome_size))
            ref_index = int(np.searchsorted(cumulative, genome_pos, side="right"))
            previous = int(cumulative[ref_index - 1]) if ref_index else 0
            local = genome_pos - previous
            chrom = references[ref_index]
            chrom_length = int(lengths[ref_index])
            start = min(max(0, local), max(0, chrom_length - 1))
            end = min(chrom_length, start + window_bp)
            window_candidates: list[tuple[int, str, object]] = []
            windows_examined += 1
            try:
                iterator = bam.fetch(chrom, start, end)
                for read in iterator:
                    records_examined += 1
                    # Sampling by alignment start avoids preferentially selecting
                    # longer molecules merely because they overlap more windows.
                    if not (start <= read.reference_start < end):
                        continue
                    if not _eligible(read, min_mapq):
                        continue
                    key = _read_key(read)
                    rank = _stable_rank(key, seed)
                    window_candidates.append((rank, key, read))
                    # Bound work in extremely deep/repetitive windows.
                    if len(window_candidates) >= 512:
                        break
            except (OSError, ValueError):
                continue
            window_candidates.sort(key=lambda item: (item[0], item[1]))
            for rank, key, read in window_candidates[:per_window]:
                old = candidates.get(key)
                if old is None or rank < old[0]:
                    candidates[key] = (rank, read)

    selected = sorted(
        ((rank, key, read) for key, (rank, read) in candidates.items()),
        key=lambda item: (item[0], item[1]),
    )[:sample_reads]
    reads = [read for _rank, _key, read in selected]
    strategy = f"indexed random windows ({windows_examined} x {window_bp:,} bp)"

    # Very sparse indexes may not yield the requested sample. Fill from a
    # bounded stream without ever turning this into a whole-BAM scan.
    if len(reads) < sample_reads:
        fill = _bounded_stream_sample(
            path, sample_reads, seed + 1, min_mapq, scan_limit
        )
        by_key = {_read_key(read): read for read in reads}
        for read in fill.reads:
            by_key.setdefault(_read_key(read), read)
        ordered = sorted(
            by_key.items(), key=lambda item: (_stable_rank(item[0], seed), item[0])
        )[:sample_reads]
        reads = [read for _key, read in ordered]
        records_examined += fill.records_examined
        strategy += " + bounded fill"
        if not reads and _no_aligned_records(fill):
            return _unaligned_stream_sample(path, sample_reads, seed, scan_limit)

    return SampledReads(
        reads=reads,
        strategy=strategy,
        records_examined=records_examined,
        windows_examined=windows_examined,
    )


_CALLING_PROGRAMS = (
    "fiberhmm-call", "fiberhmm-apply", "fiberhmm-recall-tfs", "fiberhmm-recall-nucs",
)
_DS_MODE_RE = re.compile(r"(?:^|[\s;(])mode=([a-z0-9_-]+)")
_DS_ENZYME_RE = re.compile(r"(?:^|[\s;(])enzyme=([a-z0-9_-]+)")
_CL_ENZYME_RE = re.compile(r"(?:^|\s)--enzyme(?:=|\s+)([a-z0-9_-]+)")
_CL_SEQ_RE = re.compile(r"(?:^|\s)--seq(?:=|\s+)(pacbio|nanopore)(?:\s|$)")
_CL_MODE_RE = re.compile(r"(?:^|\s)--mode(?:=|\s+)([a-z0-9_-]+)")


def _calling_program_assay(header) -> tuple[Optional[str], Optional[str]]:
    """``(mode, enzyme)`` from the newest FiberHMM calling ``@PG`` record.

    Only fiberhmm-call/-apply/-recall-tfs/-recall-nucs records are read, and
    mode and enzyme come from the same record: other programs' ``DS``/``CL``
    text (``fiberhmm-dedup ... mode=flag``, an aligner command line) never
    decides the assay.
    """
    payload = header.to_dict() if hasattr(header, "to_dict") else dict(header)
    for program in reversed(payload.get("PG", [])):
        name = str(program.get("PN") or program.get("ID") or "").lower()
        name = name.split(".", 1)[0]
        if name not in _CALLING_PROGRAMS:
            continue
        description = str(program.get("DS", "")).lower()
        command = str(program.get("CL", "")).lower()
        mode = None
        match = _DS_MODE_RE.search(description) or _CL_MODE_RE.search(command)
        if match and match.group(1) in QC_MODES:
            mode = match.group(1)
        enzyme = None
        match = _DS_ENZYME_RE.search(description) or _CL_ENZYME_RE.search(command)
        if match and match.group(1) in QC_ENZYMES:
            enzyme = match.group(1)
        if mode is None and enzyme is not None:
            if enzyme in ("ddda", "dddb"):
                mode = "daf"
            else:
                seq = _CL_SEQ_RE.search(command)
                if seq:
                    mode = f"{seq.group(1)}-fiber"
        if mode is None and enzyme is None:
            continue
        return mode, enzyme
    return None, None


def _declared_assay(header) -> tuple[Optional[str], Optional[str]]:
    """``(mode, enzyme)`` from the header's FIBERHMM-CHEMISTRY declaration."""
    from fiberhmm.io.bam_header import declared_chemistries

    for declaration in reversed(declared_chemistries(header)):
        mode = str(declaration.get("mode", "")).lower()
        enzyme = str(declaration.get("enzyme", "")).lower()
        mode = mode if mode in QC_MODES else None
        enzyme = enzyme if enzyme in QC_ENZYMES else None
        if mode is None and enzyme in ("ddda", "dddb"):
            mode = "daf"
        if mode or enzyme:
            return mode, enzyme
    return None, None


_MINIMAP2_PRESET_RE = re.compile(
    r"(?:(?:^|\s)-[A-Za-z]*x\s*|preset=)(map-ont|lr:hq|map-hifi|map-pb)(?=\s|$)")


def _aligner_platform(header) -> Optional[str]:
    """Platform named by @RG PL, by a known basecaller/aligner @PG, or by the
    preset of a minimap2 @PG (never by other programs' command lines)."""
    from fiberhmm.cli.common import _ONT_PROGRAMS, _PACBIO_PROGRAMS

    payload = header.to_dict() if hasattr(header, "to_dict") else dict(header)
    votes = set()
    for group in payload.get("RG", []):
        platform = str(group.get("PL", "")).upper()
        if platform in {"PACBIO", "PACBIO_SMRT"}:
            votes.add("pacbio")
        elif platform in {"ONT", "NANOPORE", "OXFORD_NANOPORE"}:
            votes.add("nanopore")
    for program in payload.get("PG", []):
        name = str(program.get("PN") or program.get("ID") or "").lower().split(".", 1)[0]
        if name in _PACBIO_PROGRAMS:
            votes.add("pacbio")
        elif name in _ONT_PROGRAMS:
            votes.add("nanopore")
        elif name in ("minimap2", "mappy"):
            match = _MINIMAP2_PRESET_RE.search(str(program.get("CL", "")))
            if match:
                votes.add("nanopore" if match.group(1) in ("map-ont", "lr:hq") else "pacbio")
    return next(iter(votes)) if len(votes) == 1 else None


def reference_profile_for_assay(mode: str, enzyme: Optional[str]) -> str:
    """Return the one valid built-in reference for an assay combination."""
    if mode == "daf":
        if enzyme in ("ddda", "dddb"):
            return str(enzyme)
        if enzyme == "hia5":
            raise QCInputError("incompatible QC assay: DAF mode cannot use Hia5")
        return ""
    if mode in ("pacbio-fiber", "nanopore-fiber"):
        if enzyme in ("ddda", "dddb"):
            raise QCInputError(
                f"incompatible QC assay: {mode} requires an m6A fiber enzyme, "
                f"not {enzyme}"
            )
        # EcoGII uses the same m6A observation machinery, but its rate
        # distribution is not interchangeable with the bundled Hia5 QC
        # controls.  Keep descriptive QC enabled without assigning a
        # misleading Hia5 reference score.
        if enzyme == "ecogii":
            return ""
        return "hia5_nanopore" if mode == "nanopore-fiber" else "hia5_pacbio"
    return ""


QC_PROB_THRESHOLD = 125


def default_qc_prob_threshold(mode: str, enzyme: Optional[str]) -> int:
    """ML threshold QC uses when none is given: the assay's chemistry preset.

    A Nanopore m6A assay (``nanopore-fiber``; Hia5 unless another enzyme is
    recorded) reads ML at 248, the threshold its bundled QC reference was
    calibrated at; every other assay keeps 125.
    """
    from fiberhmm.models import default_prob_threshold

    if mode == "nanopore-fiber":
        return default_prob_threshold(enzyme or "hia5", "nanopore", QC_PROB_THRESHOLD)
    return QC_PROB_THRESHOLD


def infer_assay(
    path: str,
    reads: Sequence,
    mode: str = "auto",
    enzyme: str = "auto",
) -> tuple[str, Optional[str], str]:
    """Infer observation mode, enzyme, and built-in reference profile.

    Evidence, strongest first: the ``FIBERHMM-CHEMISTRY`` declaration; the
    newest FiberHMM calling ``@PG`` record (mode and enzyme from that one
    record); then the reads and non-FiberHMM records (R/Y-encoded bases mean
    DAF; ``@RG PL``/aligner presets name the platform; PacBio otherwise).
    """
    with pysam.AlignmentFile(path, "rb", check_sq=False) as bam:
        header = bam.header
        declared_mode, declared_enzyme = _declared_assay(header)
        program_mode, program_enzyme = _calling_program_assay(header)
        platform = _aligner_platform(header)

    resolved_enzyme: Optional[str] = None if enzyme == "auto" else enzyme
    if resolved_enzyme is None:
        resolved_enzyme = declared_enzyme or (
            None if declared_mode else program_enzyme)

    resolved_mode = mode
    if resolved_mode == "auto":
        resolved_mode = declared_mode or (None if declared_enzyme else program_mode)
        if resolved_mode is None and resolved_enzyme in ("ddda", "dddb"):
            resolved_mode = "daf"
    if resolved_mode in (None, "auto"):
        if any("R" in (read.query_sequence or "").upper() or
               "Y" in (read.query_sequence or "").upper() for read in reads[:20]):
            resolved_mode = "daf"
        elif platform == "nanopore":
            resolved_mode = "nanopore-fiber"
        else:
            resolved_mode = "pacbio-fiber"

    profile = reference_profile_for_assay(resolved_mode, resolved_enzyme)
    return resolved_mode, resolved_enzyme, profile


def _mm_ml(read) -> tuple[str, Sequence[int]]:
    mm = ""
    ml: Sequence[int] = ()
    for tag in ("MM", "Mm"):
        if read.has_tag(tag):
            mm = read.get_tag(tag)
            break
    for tag in ("ML", "Ml"):
        if read.has_tag(tag):
            ml = read.get_tag(tag)
            break
    return mm, ml


_NO_UNKNOWN: frozenset = frozenset()


def _mm_calls(read, mode: str, prob_threshold: int):
    """``(modified, unknown)`` query positions (SEQ frame) from MM/ML, or None.

    ``unknown`` holds the target bases an MM ``?`` entry leaves unlisted: no
    call was made there, so they are not opportunities (SAM spec). Entries
    flagged ``.`` (or unflagged) list every base, so reads without a ``?``
    entry take the historical parse unchanged and ``unknown`` is empty.
    Returns None when the read carries no usable MM/ML.
    """
    sequence = (read.query_sequence or "").upper()
    mm, ml = _mm_ml(read)
    if mm and "?" in mm:
        # Even with an empty ML, a '?' entry says which bases were observed.
        modified, unknown = parse_mm_tag_query_calls(
            mm,
            ml,
            sequence,
            read.is_reverse,
            prob_threshold=prob_threshold,
            mode=mode,
        )
        return np.asarray(sorted(modified), dtype=np.int64), unknown
    if not mm or not len(ml):
        return None
    positions = parse_mm_tag_query_positions(
        mm,
        ml,
        sequence,
        read.is_reverse,
        prob_threshold=prob_threshold,
        mode=mode,
    )
    return np.asarray(sorted(positions), dtype=np.int64), _NO_UNKNOWN


def _daf_marks(read, reference_handle=None, prob_threshold: int = 125):
    """DAF event query positions and the MM ``?``-unknown query positions."""
    sequence = (read.query_sequence or "").upper()
    if "R" in sequence or "Y" in sequence:
        positions, _strand, _converted = extract_daf_iupac_positions(
            sequence, read.get_tag("st") if read.has_tag("st") else None
        )
        return np.asarray(sorted(positions), dtype=np.int64), _NO_UNKNOWN

    calls = _mm_calls(read, "daf", prob_threshold)
    if calls is not None:
        return calls

    try:
        from fiberhmm.daf.encoder import get_daf_positions

        result = get_daf_positions(read, ref_fasta=reference_handle)
    except Exception:
        result = None
    if result is None:
        return np.empty(0, dtype=np.int64), _NO_UNKNOWN
    ct_positions, ga_positions, strand = result
    positions = ct_positions if strand == "CT" else ga_positions
    return np.asarray(sorted(positions), dtype=np.int64), _NO_UNKNOWN


def _aligned_reference_pairs(read, reference_handle=None):
    """Return aligned query/reference/base triples without requiring a FASTA.

    An MD tag that does not describe the CIGAR gives no defined reference
    bases (pysam reads past a short MD into undefined memory), so such a read
    uses the FASTA when one is given, like a read without MD.
    """
    from fiberhmm.daf.aligned_arrays import md_disagrees_with_cigar

    if not md_disagrees_with_cigar(read):
        try:
            return read.get_aligned_pairs(with_seq=True)
        except (AttributeError, ValueError, TypeError, IndexError, AssertionError):
            pass
    if reference_handle is None or read.reference_name is None:
        return None
    try:
        start = int(read.reference_start)
        end = int(read.reference_end)
        reference = reference_handle.fetch(read.reference_name, start, end).upper()
        return [
            (query_position, reference_position, reference[reference_position - start])
            for query_position, reference_position in read.get_aligned_pairs(
                matches_only=True
            )
            if start <= reference_position < end
        ]
    except (ValueError, TypeError, IndexError, OSError):
        return None


def _daf_signal_profile(
    read,
    reference_handle=None,
    prob_threshold: int = 125,
    excluded_reference_positions=None,
) -> tuple[np.ndarray, int, int, int]:
    """Return MD/reference-conditioned DAF events, opportunities, and span.

    Raw and IUPAC-encoded DAF reads are conditioned on every aligned reference
    C/G, matching the single-embryo QC definition. Coordinates are reference
    relative, so indels do not distort the phasogram or molecule hatchmarks.
    """
    sequence = (read.query_sequence or "").upper()
    pairs = _aligned_reference_pairs(read, reference_handle)
    if pairs is not None:
        opportunities = 0
        opportunity_queries: list[int] = []
        positions: list[int] = []
        aligned_positions: list[int] = []
        query_to_reference: dict[int, int] = {}
        excluded_reference_positions = excluded_reference_positions or set()
        for query_position, reference_position, reference_base in pairs:
            if query_position is None or reference_position is None:
                continue
            aligned_positions.append(int(reference_position))
            query_to_reference[int(query_position)] = int(reference_position)
            if reference_position in excluded_reference_positions:
                continue
            if reference_base is None:
                continue
            reference_base = reference_base.upper()
            if reference_base not in ("C", "G"):
                continue
            opportunities += 1
            opportunity_queries.append(int(query_position))
            query_base = sequence[query_position]
            if (
                reference_base == "C" and query_base in ("T", "Y")
            ) or (
                reference_base == "G" and query_base in ("A", "R")
            ):
                positions.append(int(reference_position))

        # Some DAF representations carry calls only in MM/ML. Translate those
        # query coordinates onto the same reference axis when possible.
        if not positions:
            fallback, unknown = _daf_marks(read, reference_handle, prob_threshold)
            if unknown:
                # Bases an MM '?' entry leaves unlisted were never observed.
                opportunities -= sum(
                    1 for query_position in opportunity_queries
                    if query_position in unknown
                )
            positions = sorted(
                {
                    query_to_reference[int(position)]
                    for position in fallback
                    if int(position) in query_to_reference
                    and query_to_reference[int(position)] not in excluded_reference_positions
                }
            )
        if opportunities:
            start = min(aligned_positions) if aligned_positions else int(read.reference_start)
            end = (
                max(aligned_positions) + 1
                if aligned_positions
                else int(read.reference_end or read.reference_start + len(sequence))
            )
            return np.asarray(positions, dtype=np.int64), opportunities, start, end

    # Fallback for unaligned/MD-less encoded DAF BAMs. This is less exact but
    # preserves support for files that have no recoverable reference bases.
    positions, unknown = _daf_marks(read, reference_handle, prob_threshold)
    # Insertion and soft-clip bases have no reference counterpart, so they
    # carry no DAF evidence (as in calling): neither events nor opportunities.
    from fiberhmm.daf.aligned_arrays import unaligned_query_positions
    try:
        cigar = read.cigartuples
    except (AttributeError, ValueError, TypeError):
        cigar = None
    unaligned = unaligned_query_positions(cigar, len(sequence))
    if unaligned:
        positions = np.asarray(
            [p for p in positions if int(p) not in unaligned], dtype=np.int64)
    opportunities = _opportunities(sequence, positions, "daf")
    no_evidence = set(int(p) for p in unknown) | unaligned if (unknown or unaligned) else ()
    if no_evidence:
        # Unlisted '?' and unaligned bases counted above (C/G/R/Y; events
        # are never unknown).
        opportunities -= sum(
            1 for position in no_evidence
            if 0 <= int(position) < len(sequence)
            and sequence[int(position)] in "CGRY"
        )
    return positions, opportunities, 0, len(sequence)


def _fiber_marks(read, mode: str, prob_threshold: int):
    """Modified query positions and the MM ``?``-unknown query positions."""
    calls = _mm_calls(read, mode, prob_threshold)
    if calls is None:
        return np.empty(0, dtype=np.int64), _NO_UNKNOWN
    return calls


def _fiber_opportunities(
    sequence: str, mode: str, is_reverse: bool, unknown=_NO_UNKNOWN
) -> int:
    """Fiber-seq target bases the MM entries made a call at.

    Counted on the strand the MM parser (and the encoder) reads: Nanopore
    Hia5 calls basecalled-forward A only, which is SEQ T on a reverse-aligned
    read; PacBio calls A on both strands (SEQ A + T in either orientation).
    ``unknown`` (bases a ``?`` entry leaves unlisted, reported by the parser
    in that same frame) are not opportunities.
    """
    sequence = sequence.upper()
    if mode == "nanopore-fiber":
        targets = sequence.count("T" if is_reverse else "A")
    else:
        targets = sequence.count("A") + sequence.count("T")
    return targets - len(unknown)


def _opportunities(sequence: str, positions: np.ndarray, mode: str) -> int:
    """DAF opportunities of an unaligned/MD-less read (Fiber-seq modes:
    :func:`_fiber_opportunities`, which needs the read orientation)."""
    sequence = sequence.upper()
    if mode != "daf":
        raise ValueError(f"_opportunities is DAF-only, not {mode!r}")
    total = sum(sequence.count(base) for base in "CGRY")
    # Raw mismatch and MM/ML representations place converted targets at
    # A/T; add only events not already represented by C/G/R/Y.
    total += sum(
        1 for position in positions
        if 0 <= int(position) < len(sequence)
        and sequence[int(position)] not in "CGRY"
    )
    return total


def _signal_profile(
    read,
    mode: str,
    reference_handle=None,
    prob_threshold: int = 125,
    snp_mask: Optional[dict[str, set[int]]] = None,
) -> tuple[np.ndarray, int, int, int]:
    if mode == "daf":
        excluded = (snp_mask or {}).get(getattr(read, "reference_name", None), set())
        if excluded:
            from fiberhmm.daf.snps import wrapped_reference_sites
            excluded = wrapped_reference_sites(read, excluded)
        return _daf_signal_profile(
            read,
            reference_handle,
            prob_threshold,
            excluded_reference_positions=excluded,
        )
    sequence = read.query_sequence or ""
    positions, unknown = _fiber_marks(read, mode, prob_threshold)
    opportunities = _fiber_opportunities(
        sequence, mode, bool(getattr(read, "is_reverse", False)), unknown)
    return positions, opportunities, 0, len(sequence)


def _footprint_lengths(read) -> tuple[list[int], list[int], bool]:
    if read.has_tag("MA"):
        try:
            parsed = parse_ma_tag(read.get_tag("MA"))
            return (
                [int(length) for _start, length in parsed.get("nuc", [])],
                [int(length) for _start, length in parsed.get("tf", [])],
                True,
            )
        except (ValueError, TypeError, KeyError):
            return [], [], True
    if read.has_tag("nl"):
        try:
            return [int(length) for length in read.get_tag("nl")], [], True
        except (ValueError, TypeError):
            pass
    return [], [], False


def pair_distance_histogram(position_sets: Iterable[np.ndarray]) -> np.ndarray:
    histogram = np.zeros(MAX_LAG + 1, dtype=np.float64)
    for raw in position_sets:
        positions = np.sort(np.asarray(raw, dtype=np.int64))
        if len(positions) < 2:
            continue
        for index in range(len(positions) - 1):
            stop = int(
                np.searchsorted(positions, positions[index] + MAX_LAG, side="right")
            )
            distances = positions[index + 1 : stop] - positions[index]
            if len(distances):
                histogram += np.bincount(
                    distances, minlength=MAX_LAG + 1
                )[: MAX_LAG + 1]
    return histogram


def phasogram_metrics(histogram: np.ndarray) -> tuple[np.ndarray, Optional[int], float]:
    baseline = np.convolve(histogram, np.ones(121) / 121, mode="same")
    curve = np.divide(
        histogram,
        baseline,
        out=np.zeros_like(histogram, dtype=float),
        where=baseline > 0,
    )
    segment = curve[60:MAX_LAG]
    if histogram.sum() <= 0 or np.std(segment) <= 0:
        return curve, None, 0.0
    segment = segment - segment.mean()
    autocorrelation = np.correlate(segment, segment, mode="full")[len(segment) - 1 :]
    if autocorrelation[0] <= 0:
        return curve, None, 0.0
    autocorrelation /= autocorrelation[0]
    lower, upper = PERIOD_BAND
    band = autocorrelation[lower : upper + 1]
    best = int(np.argmax(band))
    return curve, lower + best, float(band[best])


def load_references() -> dict:
    path = Path(__file__).with_name("references.json")
    return json.loads(path.read_text())


def load_control_curves() -> dict:
    """Load aggregate packaged curves (never per-read control data)."""
    path = Path(__file__).with_name("control_curves.json")
    return json.loads(path.read_text())


def load_control_examples() -> dict:
    """Load anonymized, coordinate-free control molecule exemplars."""
    path = Path(__file__).with_name("control_examples.json")
    if not path.exists():
        return {"schema_version": 1, "profiles": {}}
    return json.loads(path.read_text())


def _curve_comparison(
    sample_curve: np.ndarray,
    control_curve: Optional[dict],
) -> tuple[Optional[float], Optional[float]]:
    """Return shape correlation and matched periodic amplitude versus control.

    The projection coefficient is the amplitude of the control-shaped
    component in the sample after identical smoothing and mean centering. A
    flat curve can therefore have a moderately high correlation but cannot
    receive a high periodicity score.
    """
    if not control_curve:
        return None, None
    lags = np.asarray(control_curve["phasogram"]["lags_bp"], dtype=int)
    control = np.asarray(
        control_curve["phasogram"]["detrended_pair_frequency"], dtype=float
    )
    valid = (lags >= 60) & (lags <= PLOT_MAX_LAG) & (lags < len(sample_curve))
    if valid.sum() < 20:
        return None, None
    sample = np.asarray(sample_curve[lags[valid]], dtype=float)
    control = control[valid]
    kernel = np.ones(11, dtype=float) / 11
    sample = np.convolve(sample, kernel, mode="same")
    control = np.convolve(control, kernel, mode="same")
    if np.std(sample) <= 0 or np.std(control) <= 0:
        return None, None
    sample = sample - sample.mean()
    control = control - control.mean()
    denominator = float(np.dot(control, control))
    if denominator <= 0:
        return None, None
    return (
        float(np.corrcoef(sample, control)[0, 1]),
        float(max(0.0, np.dot(sample, control) / denominator)),
    )


def _curve_correlation(
    sample_curve: np.ndarray,
    control_curve: Optional[dict],
) -> Optional[float]:
    """Backward-compatible scalar accessor for reference-curve correlation."""
    return _curve_comparison(sample_curve, control_curve)[0]


def _range_score(value: float, rate_reference: dict) -> float:
    pass_lo, pass_hi = map(float, rate_reference["pass_interval"])
    warn_lo, warn_hi = map(float, rate_reference["warn_interval"])
    center = float(rate_reference["reference_median"])
    if pass_lo <= value <= pass_hi:
        if value <= center:
            fraction = (value - pass_lo) / max(center - pass_lo, 1e-12)
        else:
            fraction = (pass_hi - value) / max(pass_hi - center, 1e-12)
        return float(np.clip(70 + 30 * fraction, 70, 100))
    if warn_lo <= value < pass_lo:
        return float(
            35 + 34 * (value - warn_lo) / max(pass_lo - warn_lo, 1e-12)
        )
    if pass_hi < value <= warn_hi:
        return float(
            35 + 34 * (warn_hi - value) / max(warn_hi - pass_hi, 1e-12)
        )
    if value < warn_lo:
        return float(34 * np.clip(value / max(warn_lo, 1e-12), 0, 1))
    return float(
        34
        * np.clip(
            1 - (value - warn_hi) / max(warn_hi - pass_hi, 1e-12),
            0,
            1,
        )
    )


def _high_is_good_score(
    value: float,
    warn_threshold: float,
    pass_threshold: float,
    reference_value: float = 1.0,
) -> float:
    """Strict FAIL/WARN/PASS score for a higher-is-better diagnostic."""
    if value >= pass_threshold:
        return float(
            np.clip(
                70
                + 30
                * (value - pass_threshold)
                / max(reference_value - pass_threshold, 1e-12),
                70,
                100,
            )
        )
    if value >= warn_threshold:
        return float(
            35
            + 34
            * (value - warn_threshold)
            / max(pass_threshold - warn_threshold, 1e-12)
        )
    return float(
        34 * np.clip(value / max(warn_threshold, 1e-12), 0, 1)
    )


def _periodicity_score(
    nrl: Optional[int],
    strength: float,
    reference: dict,
    curve_correlation: Optional[float] = None,
    reference_projection: Optional[float] = None,
) -> float:
    if nrl is None:
        return 0.0
    ref_strength = float(reference["reference_strength"])
    pass_strength = float(reference["pass_strength"])
    warn_strength = float(reference["warn_strength"])
    strength_score = _high_is_good_score(
        strength,
        warn_strength,
        pass_strength,
        ref_strength,
    )
    nrl_score = max(0.0, 100.0 - 2.0 * abs(nrl - float(reference["reference_nrl_bp"])))
    components = [strength_score, nrl_score]
    if curve_correlation is not None:
        components.append(
            _high_is_good_score(
                max(0.0, curve_correlation),
                float(reference.get("warn_curve_correlation", 0.60)),
                float(reference.get("pass_curve_correlation", 0.75)),
            )
        )
    if reference_projection is not None:
        components.append(
            _high_is_good_score(
                reference_projection,
                float(reference.get("warn_reference_projection", 0.50)),
                float(reference.get("pass_reference_projection", 0.60)),
            )
        )
    # All diagnostics must support the classification. Taking the limiting
    # component prevents a random in-band peak or a low-amplitude flat curve
    # from being rescued by an otherwise plausible NRL.
    return float(np.clip(min(components), 0, 100))


def _status(score: Optional[float]) -> str:
    if score is None:
        return "INSUFFICIENT"
    if score >= 70:
        return "PASS"
    if score >= 35:
        return "WARN"
    return "FAIL"


def analyze_sample(
    reads: Sequence,
    mode: str,
    reference_profile: Optional[dict],
    reference_fasta: Optional[str] = None,
    prob_threshold: int = 125,
    min_opportunities: int = 200,
    snp_mask: Optional[dict[str, set[int]]] = None,
    control_curve: Optional[dict] = None,
) -> tuple[dict, dict]:
    rates: list[float] = []
    events = 0
    opportunities = 0
    position_sets: list[np.ndarray] = []
    nuc_lengths: list[int] = []
    tf_lengths: list[int] = []
    tagged_reads = 0
    signal_reads = 0
    read_examples: list[dict] = []
    duplicate_flagged = 0
    dedup_tagged = 0
    dedup_cluster_sizes: dict[int, int] = {}
    reference_handle = pysam.FastaFile(reference_fasta) if reference_fasta else None
    try:
        for read in reads:
            is_duplicate = bool(getattr(read, "is_duplicate", False))
            duplicate_flagged += int(is_duplicate)
            if read.has_tag("ds"):
                try:
                    cluster_size = int(read.get_tag("ds"))
                    cluster_id = (
                        int(read.get_tag("di"))
                        if read.has_tag("di")
                        else -(dedup_tagged + 1)
                    )
                    dedup_cluster_sizes[cluster_id] = cluster_size
                    dedup_tagged += 1
                except (ValueError, TypeError):
                    pass
            # Nondestructive mark/retain dedup keeps every read in the BAM,
            # but aggregate rate, phase, footprint, and example panels should
            # represent original molecules rather than PCR copy number.
            if is_duplicate:
                continue
            positions, n_opportunities, span_start, span_end = _signal_profile(
                read,
                mode,
                reference_handle=reference_handle,
                prob_threshold=prob_threshold,
                snp_mask=snp_mask,
            )
            if n_opportunities >= min_opportunities:
                rate = len(positions) / n_opportunities
                rates.append(rate)
                events += len(positions)
                opportunities += n_opportunities
                if span_end > span_start:
                    read_examples.append(
                        {
                            "span_start": int(span_start),
                            "span_end": int(span_end),
                            "positions": positions.copy(),
                            "rate": float(rate),
                        }
                    )
            if len(positions) >= 3:
                signal_reads += 1
                position_sets.append(positions)
            nuc, tf, tagged = _footprint_lengths(read)
            nuc_lengths.extend(nuc)
            tf_lengths.extend(tf)
            tagged_reads += int(tagged)
    finally:
        if reference_handle is not None:
            reference_handle.close()

    histogram = pair_distance_histogram(position_sets)
    curve, nrl, strength = phasogram_metrics(histogram)
    curve_correlation, reference_projection = _curve_comparison(
        curve,
        control_curve,
    )
    rate_values = np.asarray(rates, dtype=float)
    rate_score: Optional[float] = None
    periodicity_score: Optional[float] = None
    rate_note = ""
    scoring_enabled = bool(
        reference_profile and reference_profile.get("scoring_enabled", True)
    )
    if reference_profile and not scoring_enabled:
        rate_note = reference_profile.get(
            "calibration_note", "reference calibration is pending"
        )
    if scoring_enabled and len(rate_values) >= 20 and events >= 50:
        rate_score = _range_score(float(np.median(rate_values)), reference_profile["rate"])
        reference_threshold = reference_profile["rate"].get("probability_threshold")
        if reference_threshold is not None and abs(prob_threshold - int(reference_threshold)) > 5:
            rate_note = (
                f"reference was calibrated at ML threshold {reference_threshold}; "
                f"this run used {prob_threshold}"
            )
            rate_score = min(rate_score, 69.0)
    if (
        scoring_enabled
        and signal_reads >= 50
        and events >= 500
        and histogram.sum() >= 5_000
    ):
        periodicity_score = _periodicity_score(
            nrl,
            strength,
            reference_profile["periodicity"],
            curve_correlation=curve_correlation,
            reference_projection=reference_projection,
        )

    component_statuses = [_status(rate_score), _status(periodicity_score)]
    scored = [score for score in (rate_score, periodicity_score) if score is not None]
    overall_score = float(np.mean(scored)) if scored else None
    if not scored:
        overall_status = "INSUFFICIENT"
    elif "FAIL" in component_statuses:
        overall_status = "FAIL"
    elif "WARN" in component_statuses or "INSUFFICIENT" in component_statuses:
        overall_status = "WARN"
    else:
        overall_status = "PASS"

    quantiles = (
        np.quantile(rate_values, (0.05, 0.25, 0.5, 0.75, 0.95)).tolist()
        if len(rate_values)
        else [None] * 5
    )
    result = {
        "signal": {
            "label": (
                reference_profile.get("signal_label", "signal")
                if reference_profile
                else ("deamination" if mode == "daf" else "m6A labeling")
            ),
            "n_rate_reads": int(len(rate_values)),
            "n_signal_reads": int(signal_reads),
            "n_events": int(events),
            "n_opportunities": int(opportunities),
            "aggregate_rate": float(events / opportunities) if opportunities else None,
            "mean_per_read_rate": float(np.mean(rate_values)) if len(rate_values) else None,
            "p05_per_read_rate": quantiles[0],
            "q25_per_read_rate": quantiles[1],
            "median_per_read_rate": quantiles[2],
            "q75_per_read_rate": quantiles[3],
            "p95_per_read_rate": quantiles[4],
            "score": rate_score,
            "status": _status(rate_score),
            "note": rate_note,
        },
        "periodicity": {
            "n_informative_reads": int(signal_reads),
            "pair_distance_observations_le_1000": int(histogram.sum()),
            "search_interval_bp": list(PERIOD_BAND),
            "nrl_bp": nrl,
            "autocorrelation_strength": strength,
            "reference_curve_correlation": curve_correlation,
            "reference_pattern_amplitude_ratio": reference_projection,
            "score": periodicity_score,
            "status": _status(periodicity_score),
        },
        "footprints": {
            "n_tagged_reads": int(tagged_reads),
            "n_nucleosomes": len(nuc_lengths),
            "median_nucleosome_bp": float(np.median(nuc_lengths)) if nuc_lengths else None,
            "q25_nucleosome_bp": float(np.quantile(nuc_lengths, 0.25)) if nuc_lengths else None,
            "q75_nucleosome_bp": float(np.quantile(nuc_lengths, 0.75)) if nuc_lengths else None,
            "fraction_nucleosome_85_250_bp": (
                float(np.mean((np.asarray(nuc_lengths) >= 85) & (np.asarray(nuc_lengths) <= 250)))
                if nuc_lengths else None
            ),
            "fraction_nucleosome_over_300_bp": (
                float(np.mean(np.asarray(nuc_lengths) > 300)) if nuc_lengths else None
            ),
            "fraction_nucleosome_over_1000_bp": (
                float(np.mean(np.asarray(nuc_lengths) > 1000)) if nuc_lengths else None
            ),
            "n_tf_footprints": len(tf_lengths),
            "median_tf_footprint_bp": float(np.median(tf_lengths)) if tf_lengths else None,
            "q25_tf_footprint_bp": float(np.quantile(tf_lengths, 0.25)) if tf_lengths else None,
            "q75_tf_footprint_bp": float(np.quantile(tf_lengths, 0.75)) if tf_lengths else None,
        },
        "deduplication": _summarize_deduplication(
            len(reads),
            duplicate_flagged,
            dedup_tagged,
            dedup_cluster_sizes,
        ),
        "overall": {"score": overall_score, "status": overall_status},
    }
    arrays = {
        "rates": rate_values,
        "phasogram": curve,
        "histogram": histogram,
        "nuc_lengths": np.asarray(nuc_lengths, dtype=float),
        "tf_lengths": np.asarray(tf_lengths, dtype=float),
        "read_examples": read_examples,
    }
    return result, arrays


def _summarize_deduplication(
    sampled_reads: int,
    duplicate_flagged: int,
    tagged_reads: int,
    cluster_sizes: dict[int, int],
) -> dict:
    detected = bool(duplicate_flagged or tagged_reads)
    histogram: dict[str, int] = {}
    for size in cluster_sizes.values():
        key = str(size)
        histogram[key] = histogram.get(key, 0) + 1
    if duplicate_flagged:
        duplicate_fraction = duplicate_flagged / sampled_reads if sampled_reads else None
        basis = "SAM duplicate flags in bounded sample"
        mode = "flagged"
        represented = sum(cluster_sizes.values())
    elif cluster_sizes:
        reconstructed_duplicates = sum(size - 1 for size in cluster_sizes.values())
        reconstructed_input = sampled_reads + reconstructed_duplicates
        duplicate_fraction = (
            reconstructed_duplicates / reconstructed_input
            if reconstructed_input
            else None
        )
        basis = "cluster-size tags on retained representatives in bounded sample"
        mode = "collapsed"
        represented = len(cluster_sizes)
    else:
        duplicate_fraction = None
        basis = "not detected; QC did not run deduplication"
        mode = "not_run"
        represented = 0
    if detected:
        inferred_singletons = max(0, sampled_reads - represented)
        histogram["1"] = histogram.get("1", 0) + inferred_singletons
    return {
        "detected": detected,
        "mode": mode,
        "method": "fiberhmm_deamination_fingerprint" if tagged_reads else (
            "sam_duplicate_flag" if duplicate_flagged else None
        ),
        "sampled_reads": int(sampled_reads),
        "cluster_tagged_reads": int(tagged_reads),
        "duplicate_flagged_reads": int(duplicate_flagged),
        "duplicate_fraction": duplicate_fraction,
        "duplicate_fraction_basis": basis,
        "observed_duplicate_cluster_size_histogram": histogram,
        "molecule_cluster_size_histogram": histogram,
        "full_run": False,
    }


def _merge_full_dedup_summary(sample_summary: dict, run_summary: Optional[dict]) -> dict:
    """Prefer exact pre-call dedup statistics while retaining sample context."""
    if not run_summary:
        return sample_summary
    fingerprintable = int(run_summary.get("n_fingerprintable", 0))
    duplicates = int(run_summary.get("n_duplicates", 0))
    histogram = {
        str(size): int(count)
        for size, count in run_summary.get("cluster_size_histogram", {}).items()
    }
    mode = str(run_summary.get("mode") or "collapsed")
    return {
        **sample_summary,
        "detected": True,
        "mode": mode,
        "method": "fiberhmm_deamination_fingerprint",
        "duplicate_fraction": duplicates / fingerprintable if fingerprintable else None,
        "duplicate_fraction_basis": "full pre-call PCR deduplication run",
        "molecule_cluster_size_histogram": histogram,
        "full_run": True,
        "full_run_statistics": {
            key: run_summary.get(key)
            for key in (
                "n_total",
                "n_fingerprintable",
                "n_clusters",
                "n_duplicates",
                "n_written",
                "n_below_min_deam",
                "n_unmapped",
                "n_singleton_molecules",
                "largest_cluster",
            )
        },
    }


def _load_variant_masking(
    snp_report_path: Optional[str],
    snp_mask_path: Optional[str],
) -> tuple[dict, list[dict], list[dict]]:
    calls: list[dict] = []
    site_distribution: list[dict] = []
    amplicons: list[dict] = []
    report = None
    if snp_report_path and Path(snp_report_path).exists():
        report = json.loads(Path(snp_report_path).read_text())
        calls = list(report.get("calls", []))
        site_distribution = list(report.get("site_distribution", []))
        amplicons = list(report.get("amplicons", []))
    n_sites = 0
    if snp_mask_path:
        try:
            from fiberhmm.daf.snps import mask_summary

            n_sites = mask_summary(snp_mask_path)["n_sites"]
        except (OSError, ValueError):
            n_sites = 0
    if report is not None:
        n_sites = int(report.get("n_called_snps", len(calls)))
    fractions = np.asarray(
        [call["alternate_fraction"] for call in calls], dtype=float
    )
    depths = np.asarray(
        [call["opposite_direction_depth"] for call in calls], dtype=float
    )
    return (
        {
            "applied": bool(snp_mask_path),
            "method": report.get("method") if report else (
                "external_bed_mask" if snp_mask_path else None
            ),
            "n_masked_sites": n_sites,
            "mask_path": str(Path(snp_mask_path).resolve()) if snp_mask_path else None,
            "caller_report_path": (
                str(Path(snp_report_path).resolve()) if snp_report_path else None
            ),
            "parameters": report.get("parameters", {}) if report else {},
            "threshold_policy": report.get("threshold_policy") if report else None,
            "n_profiled_sites": int(report.get("n_profiled_sites", 0)) if report else 0,
            "dominant_amplicon": report.get("dominant_amplicon") if report else None,
            "n_discovered_amplicons": len(amplicons),
            "amplicons": amplicons,
            "median_alternate_fraction": (
                float(np.median(fractions)) if len(fractions) else None
            ),
            "median_opposite_direction_depth": (
                float(np.median(depths)) if len(depths) else None
            ),
            "note": (
                "MD tags preserved; masked sites were excluded only from DAF observations"
                if snp_mask_path
                else "not run"
            ),
        },
        calls,
        site_distribution,
    )


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _fmt_score(value) -> str:
    return "NA" if value is None else f"{float(value):.0f}"


def _pct(value, digits: int = 2) -> str:
    return "NA" if value is None else f"{100 * float(value):.{digits}f}%"


def _state_terminal_lines(result: dict) -> list[str]:
    block = result.get("state_rates")
    if not block:
        return []
    if not block.get("available"):
        return [f"  state-aware rates: unavailable ({block.get('note') or 'no states'})"]
    source = block.get("source")
    if source == "light_call":
        light = block.get("light_call", {})
        stop = f", stopped by {light['stopped_by']}" if light.get("stopped_by") else ""
        source_text = (f"light call of {light.get('reads_called', 0):,} sampled reads "
                       f"with {light.get('model')} ({light.get('elapsed_seconds', 0):.1f} s"
                       f"{stop}; apply HMM only)")
    else:
        source_text = "the BAM's FiberHMM calls"
    efficiency = result.get("efficiency", {})
    background = result.get("background", {})
    ratio = block.get("msp_to_outside_ratio")
    lines = [
        f"  states from {source_text}; MSP >= {block['definition']['min_msp_bp']} bp",
        f"  in-MSP rate (efficiency): {_pct(efficiency.get('value'))} median, "
        f"{_pct(block['msp']['aggregate_rate'])} pooled  "
        f"[{efficiency.get('status', 'NA')} {_fmt_score(efficiency.get('score'))}/100]",
        f"  outside-MSP rate (background): {_pct(background.get('value'))} median, "
        f"{_pct(block['outside_msp']['aggregate_rate'])} pooled  "
        f"[{background.get('status', 'NA')} {_fmt_score(background.get('score'))}/100]",
        f"  in-MSP/outside-MSP = {'NA' if ratio is None else f'{ratio:.1f}x'}; "
        f"MSP = {_pct(block.get('msp_length_fraction'), 1)} of read length; "
        f"all-state rate {_pct(block['all_states']['aggregate_rate'])}",
    ]
    note = efficiency.get("note") or background.get("note")
    if note:
        lines.append(f"  state note: {note}")
    basis = result.get("overall", {}).get("verdict_basis")
    if basis == "state-aware":
        lines.append("  verdict uses in-MSP efficiency + outside-MSP background "
                     "(the signal-rate grade is reported but excluded from the verdict)")
    return lines


def format_terminal(result: dict) -> str:
    overall = result["overall"]
    signal = result["signal"]
    periodicity = result["periodicity"]
    footprints = result["footprints"]
    accounting = result["sampling"]
    median_rate = signal["median_per_read_rate"]
    rate_text = "NA" if median_rate is None else f"{100 * median_rate:.2f}%"
    nrl = periodicity["nrl_bp"]
    nrl_text = "NA" if nrl is None else f"{nrl} bp"
    curve_correlation = periodicity.get("reference_curve_correlation")
    amplitude_ratio = periodicity.get("reference_pattern_amplitude_ratio")
    correlation_text = (
        "NA" if curve_correlation is None else f"{curve_correlation:.3f}"
    )
    amplitude_text = (
        "NA" if amplitude_ratio is None else f"{amplitude_ratio:.2f}x"
    )
    lines = [
        "",
        "========================================================================",
        f"  FiberHMM QC: {overall['status']}  score={_fmt_score(overall['score'])}/100",
        f"  sample: {accounting['sampled_reads']:,} reads via {accounting['strategy']}",
        f"  {signal['label']}: {rate_text} median  "
        f"[{signal['status']} {_fmt_score(signal['score'])}/100]",
        f"  nucleosome periodicity: NRL={nrl_text}, "
        f"AC={periodicity['autocorrelation_strength']:.3f}  "
        f"[{periodicity['status']} {_fmt_score(periodicity['score'])}/100]",
        f"  reference phasogram agreement: r={correlation_text}, "
        f"matched amplitude={amplitude_text}",
    ]
    lines.extend(_state_terminal_lines(result))
    if footprints["n_tagged_reads"]:
        lines.append(
            f"  footprint tags: nuc n={footprints['n_nucleosomes']:,}, "
            f"median={footprints['median_nucleosome_bp'] or 0:.0f} bp; "
            f"TF n={footprints['n_tf_footprints']:,}, "
            f"median={footprints['median_tf_footprint_bp'] or 0:.0f} bp"
        )
        long_fraction = footprints.get("fraction_nucleosome_over_300_bp")
        very_long_fraction = footprints.get("fraction_nucleosome_over_1000_bp")
        if long_fraction is not None:
            lines.append(
                f"  nuc-tagged long spans: {100 * long_fraction:.1f}% >300 bp; "
                f"{100 * very_long_fraction:.1f}% >1 kb"
            )
    else:
        lines.append("  footprint sizes: unavailable (no MA or nl tags in sample)")
    deduplication = result["deduplication"]
    if deduplication["detected"]:
        duplicate_fraction = deduplication["duplicate_fraction"]
        fraction_text = (
            "NA"
            if duplicate_fraction is None
            else f"{100 * duplicate_fraction:.1f}%"
        )
        qualifier = "full run" if deduplication.get("full_run") else "bounded sample"
        lines.append(
            f"  PCR deduplication: {deduplication['mode']}, "
            f"duplicates={fraction_text} ({qualifier})"
        )
    else:
        lines.append("  PCR deduplication: not run")
    variant_masking = result.get("variant_masking", {})
    if variant_masking.get("applied"):
        policy = variant_masking.get("threshold_policy") or {}
        lines.append(
            f"  DAF SNP mask: {variant_masking['n_masked_sites']:,} sites applied; "
            "MD preserved"
        )
        if policy.get("name"):
            lines.append(f"  DAF SNP threshold policy: {policy['name']}")
        amplicons = variant_masking.get("amplicons", [])
        if amplicons:
            minimum = variant_masking.get("parameters", {}).get(
                "min_amplicon_reads", "?"
            )
            lines.append(
                f"  amplicons: {len(amplicons):,} discovered "
                f"(minimum {minimum} aligned reads)"
            )
            for amplicon in amplicons[:12]:
                lines.append(
                    f"    {amplicon['amplicon_id']}: {amplicon['chrom']}:"
                    f"{amplicon['consensus_start_0based'] + 1:,}-"
                    f"{amplicon['consensus_end_0based_exclusive']:,}; "
                    f"{amplicon['total_aligned_reads']:,} reads; "
                    f"{amplicon['n_called_snps']:,} SNPs"
                )
            if len(amplicons) > 12:
                lines.append(f"    ... {len(amplicons) - 12:,} additional amplicons")
            mapped_snps = sum(
                int(amplicon.get("n_called_snps", 0)) for amplicon in amplicons
            )
            total_snps = int(variant_masking.get("n_masked_sites", 0))
            if mapped_snps != total_snps:
                lines.append(
                    f"  SNP-to-amplicon assignment: {mapped_snps:,}/{total_snps:,} "
                    "within reported consensus spans; "
                    f"{total_snps - mapped_snps:,} outside"
                )
    elif result["assay"]["mode"] == "daf":
        preflight = variant_masking.get("screening_preflight")
        if preflight and preflight.get("run") is False:
            lines.append(
                "  DAF SNP screen: automatic low-coverage skip; "
                f"estimated {preflight.get('estimated_genome_coverage', 0):.2f}x, "
                f"sampled depth peak {preflight.get('max_local_depth', 0):,}, "
                f"targeted-bin fraction "
                f"{100 * preflight.get('supported_start_bin_fraction', 0):.1f}%"
            )
        else:
            lines.append("  DAF SNP mask: not run (optional)")
    if signal.get("note"):
        lines.append(f"  note: {signal['note']}")
    lines.extend(
        (
            f"  report: {result['outputs']['json']}",
            f"  plot:   {result['outputs'].get('plot') or 'not written (install matplotlib)'}",
            f"  vector PDF: {result['outputs'].get('pdf') or 'not written (install matplotlib)'}",
            "========================================================================",
        )
    )
    return "\n".join(lines)


def _plot_footprint_distribution(
    axis,
    sample_values: np.ndarray,
    control_curve: Optional[dict],
    footprint_key: str,
    sample_color: str,
    title: str,
    xlabel: str,
) -> None:
    """Plot sample calls against the packaged empirical control histogram."""
    control = ((control_curve or {}).get("footprint_sizes") or {}).get(footprint_key)
    step = 10 if footprint_key == "nucleosome" else 5
    stop = 2000 if footprint_key == "nucleosome" else 500
    edges = (
        np.asarray(control["bin_edges_bp"], dtype=float)
        if control
        else np.arange(0, stop + step, step, dtype=float)
    )
    sample_values = np.asarray(sample_values, dtype=float)
    if len(sample_values):
        axis.hist(
            sample_values,
            bins=edges,
            weights=np.full(len(sample_values), 1.0 / len(sample_values)),
            color=sample_color,
            alpha=0.68,
            label=f"sample (n={len(sample_values):,})",
        )
        sample_median = float(np.median(sample_values))
        axis.axvline(
            sample_median,
            color=sample_color,
            linestyle="--",
            linewidth=1.2,
            label=f"sample median={sample_median:.0f} bp",
        )
        if footprint_key == "nucleosome":
            long_fraction = float(np.mean(sample_values > 300))
            very_long_fraction = float(np.mean(sample_values > 1000))
            overflow_fraction = float(np.mean(sample_values >= edges[-1]))
            axis.text(
                0.98,
                0.96,
                f"{100 * long_fraction:.1f}% >300 bp\n"
                f"{100 * very_long_fraction:.1f}% >1 kb\n"
                f"{100 * overflow_fraction:.1f}% beyond {edges[-1]:.0f}-bp plot",
                ha="right",
                va="top",
                transform=axis.transAxes,
                fontsize=7.5,
                color="#555555",
            )
    else:
        axis.text(
            0.98, 0.94, "sample tags unavailable", ha="right", va="top",
            transform=axis.transAxes, fontsize=8, color="#666666",
        )
    if control:
        fractions = np.asarray(control["fraction_per_bin"], dtype=float)
        centers = (edges[:-1] + edges[1:]) / 2
        axis.step(
            centers,
            fractions,
            where="mid",
            color="#222222",
            linewidth=1.5,
            label=control.get("source_label", "matched control"),
        )
        control_median = float(control["median_bp"])
        axis.axvline(
            control_median,
            color="#222222",
            linestyle=":",
            linewidth=1.1,
            label=f"control median={control_median:.0f} bp",
        )
    else:
        axis.text(
            0.98, 0.80, "matched control distribution unavailable",
            ha="right", va="top", transform=axis.transAxes,
            fontsize=8, color="#666666",
        )
    visible_values = []
    if len(sample_values):
        visible_values.append(float(np.quantile(sample_values, 0.99)))
    if control:
        cdf = np.cumsum(np.asarray(control["fraction_per_bin"], dtype=float))
        if len(cdf) and cdf[-1] > 0:
            index = min(int(np.searchsorted(cdf, 0.99)), len(edges) - 2)
            visible_values.append(float(edges[index + 1]))
    minimum = 320 if footprint_key == "nucleosome" else 120
    upper = min(stop, max([minimum, *visible_values]))
    upper = step * math.ceil(upper / step)
    axis.set_xlim(0, upper)
    axis.set(
        xlabel=xlabel,
        ylabel=f"fraction of calls / {step} bp",
        title=title,
    )
    if len(sample_values) or control:
        axis.legend(frameon=False, fontsize=7.2)


def _plot_deduplication(axis, deduplication: dict) -> None:
    histogram = deduplication.get("molecule_cluster_size_histogram", {})
    if histogram:
        sizes = np.asarray(sorted(map(int, histogram)), dtype=int)
        counts = np.asarray([histogram[str(size)] for size in sizes], dtype=float)
        percentages = 100 * counts / counts.sum()
        axis.bar(sizes, percentages, color="#7a5195", alpha=0.85)
        axis.set(xlabel="copies per original molecule", ylabel="molecules (%)")
        if len(sizes) > 12 or (len(sizes) and sizes[-1] > 20):
            axis.set_xscale("log")
        statistics = deduplication.get("full_run_statistics") or {}
        if statistics:
            axis.text(
                0.98,
                0.96,
                f"{int(statistics.get('n_fingerprintable') or 0):,} fingerprintable reads\n"
                f"{int(statistics.get('n_clusters') or 0):,} original molecules",
                ha="right",
                va="top",
                transform=axis.transAxes,
                fontsize=8,
            )
    elif deduplication["detected"]:
        unique = deduplication["sampled_reads"] - deduplication["duplicate_flagged_reads"]
        axis.bar(
            ["retained", "duplicate-flagged"],
            [unique, deduplication["duplicate_flagged_reads"]],
            color=["#4c78a8", "#d1495b"],
        )
        axis.set_ylabel("sampled reads")
    else:
        axis.text(
            0.5, 0.5, "PCR deduplication not run", ha="center", va="center",
            transform=axis.transAxes,
        )
    duplicate_fraction = deduplication["duplicate_fraction"]
    duplicate_text = (
        "not run" if duplicate_fraction is None
        else f"{100 * duplicate_fraction:.1f}% duplicate reads"
    )
    basis = "full run" if deduplication.get("full_run") else "bounded sample"
    axis.set_title(f"PCR duplication — {duplicate_text} ({basis})")


def _plot_snp_landscape(figure, axis, result: dict, arrays: dict) -> None:
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    axis.set_box_aspect(1)
    variant = result["variant_masking"]
    sites = arrays.get("snp_site_distribution", [])
    if not sites:
        preflight = variant.get("screening_preflight")
        message = (
            f"{variant['n_masked_sites']:,} SNP sites masked\nbackground site profile unavailable"
            if variant["applied"]
            else (
                "Automatic DAF SNP screen skipped\n"
                "insufficient global or targeted coverage"
                if preflight and preflight.get("run") is False
                else "DAF SNP masking not run\n(optional: --daf-call-snps)"
            )
        )
        axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
        axis.set_title("DAF mismatch landscape")
        return
    usable = [
        site for site in sites
        if site.get("expected_direction_mismatch_fraction") is not None
        and site.get("opposite_direction_mismatch_fraction") is not None
    ]
    if not usable:
        axis.text(0.5, 0.5, "insufficient bidirectional depth", ha="center", va="center", transform=axis.transAxes)
        axis.set_title("DAF mismatch landscape")
        return
    depths = np.asarray([max(1, site["total_dominant_fiber_depth"]) for site in usable])
    vmin = max(1, int(depths.min()))
    norm = LogNorm(vmin=vmin, vmax=max(vmin * 1.01, float(depths.max())))
    plotted = None
    for change, marker, label in (
        ("C>T", "o", "C→T positions"),
        ("G>A", "^", "G→A positions"),
    ):
        group = [
            site for site in usable
            if f"{site['reference']}>{site['alternate']}" == change
        ]
        if not group:
            continue
        plotted = axis.scatter(
            [100 * site["expected_direction_mismatch_fraction"] for site in group],
            [100 * site["opposite_direction_mismatch_fraction"] for site in group],
            c=[site["total_dominant_fiber_depth"] for site in group],
            cmap="viridis",
            norm=norm,
            marker=marker,
            s=15,
            alpha=0.38,
            linewidths=0,
            label=label,
        )
    called = [site for site in usable if site.get("called_as_snp")]
    if called:
        axis.scatter(
            [100 * site["expected_direction_mismatch_fraction"] for site in called],
            [100 * site["opposite_direction_mismatch_fraction"] for site in called],
            facecolors="none",
            edgecolors="#d01c8b",
            linewidths=1.25,
            s=55,
            zorder=5,
        )
    axis.plot([0, 100], [0, 100], color="#bdbdbd", linestyle=":", linewidth=0.8)
    parameters = variant.get("parameters", {})
    if parameters.get("min_fraction") is not None:
        axis.axhline(
            100 * parameters["min_fraction"],
            color="#555555",
            linestyle="--",
            linewidth=0.9,
        )
    axis.set(
        xlim=(-2, 104),
        ylim=(-2, 106),
        xlabel="mismatch on expected-conversion fibers (%)",
        ylabel="mismatch on opposite-conversion fibers (%)",
        title=(
            f"DAF mismatch landscape — {variant['n_masked_sites']:,} SNPs / "
            f"{len(usable):,} profiled positions"
        ),
    )
    if plotted is not None:
        colorbar_axis = inset_axes(
            axis, width="2.3%", height="42%", loc="center right", borderpad=1.2
        )
        colorbar = figure.colorbar(plotted, cax=colorbar_axis)
        colorbar.set_label("fiber depth", fontsize=7, labelpad=2)
        colorbar.ax.yaxis.set_ticks_position("left")
        colorbar.ax.yaxis.set_label_position("left")
        colorbar.ax.tick_params(labelsize=7)
    handles, labels = axis.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0], [0], marker="o", color="none", markeredgecolor="#d01c8b",
            markerfacecolor="none", label="called SNP", markersize=7,
        )
    )
    labels.append("called SNP")
    axis.legend(handles, labels, frameon=False, fontsize=7.2, loc="upper left")


def _plot_snp_map(axis, result: dict, arrays: dict) -> None:
    from matplotlib.lines import Line2D

    variant = result["variant_masking"]
    amplicons = variant.get("amplicons", [])
    if not amplicons:
        preflight = variant.get("screening_preflight")
        axis.text(
            0.5,
            0.5,
            (
                "amplicon discovery not attempted\nlow-coverage SNP preflight"
                if preflight and preflight.get("run") is False
                else "amplicon discovery unavailable"
            ),
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_title("Discovered amplicons and SNP locations")
        return

    coverages = np.asarray(
        [max(1, int(amplicon["total_aligned_reads"])) for amplicon in amplicons],
        dtype=float,
    )
    log_coverages = np.log10(coverages)
    if len(amplicons) == 1 or np.ptp(log_coverages) == 0:
        coverage_scale = np.ones(len(amplicons))
    else:
        coverage_scale = (log_coverages - log_coverages.min()) / np.ptp(log_coverages)
    labels = []
    n_snps = 0
    for row, (amplicon, scaled_coverage) in enumerate(
        zip(amplicons, coverage_scale)
    ):
        start = int(amplicon["consensus_start_0based"])
        end = int(amplicon["consensus_end_0based_exclusive"])
        span = max(1, end - start)
        axis.hlines(
            row,
            0,
            100,
            color="#4c78a8",
            alpha=0.38 + 0.47 * scaled_coverage,
            linewidth=1.4 + 4.0 * scaled_coverage,
            zorder=1,
        )
        for change, offset, marker, color in (
            ("C>T", -0.18, "o", "#c51b7d"),
            ("G>A", 0.18, "^", "#2b8cbe"),
        ):
            sites = [
                site
                for site in amplicon.get("snp_positions", [])
                if site["change"] == change
            ]
            if not sites:
                continue
            x = np.asarray(
                [
                    100
                    * float(
                        site.get(
                            "relative_position_fraction",
                            float(site["relative_position_bp"]) / span,
                        )
                    )
                    for site in sites
                ]
            )
            heights = row + offset
            axis.vlines(x, row, heights, color=color, linewidth=0.9, alpha=0.8)
            axis.scatter(
                x,
                np.full(len(x), heights),
                marker=marker,
                color=color,
                s=26,
                zorder=3,
            )
            n_snps += len(sites)
        labels.append(
            amplicon["amplicon_id"].replace("_", " ").title()
        )

    minimum = variant.get("parameters", {}).get("min_amplicon_reads", "?")
    total_snps = int(variant.get("n_masked_sites", n_snps))
    axis.set(
        xlim=(-1.5, 101.5),
        ylim=(len(amplicons) - 0.45, -0.55),
        xlabel="position along each amplicon consensus (%)",
        title=(
            f"Discovered amplicons and SNP map — {len(amplicons):,} amplicons "
            f"≥{minimum} reads; {n_snps:,}/{total_snps:,} SNPs within consensus spans"
        ),
    )
    axis.set_yticks(range(len(amplicons)), labels, fontsize=max(5.5, 8.0 - 0.08 * len(amplicons)))
    axis.legend(
        handles=(
            Line2D([0], [0], color="#4c78a8", linewidth=4, label="amplicon (line weight ∝ log coverage)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#c51b7d", label="C→T SNP"),
            Line2D([0], [0], marker="^", color="none", markerfacecolor="#2b8cbe", label="G→A SNP"),
        ),
        frameon=False,
        fontsize=7.2,
        loc="lower right",
    )


def _plot_amplicon_table(axis, result: dict) -> None:
    """Show exact amplicon metadata without crowding the SNP-map axis."""
    axis.set_axis_off()
    variant = result["variant_masking"]
    amplicons = variant.get("amplicons", [])
    if not amplicons:
        preflight = variant.get("screening_preflight")
        if preflight and preflight.get("run") is False:
            # The map panel already carries the explicit low-coverage reason;
            # leave the table area blank rather than repeat it below the plot.
            return
        axis.text(
            0.5,
            0.5,
            "No amplicon inventory available",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        return
    rows = []
    for amplicon in amplicons:
        start = int(amplicon["consensus_start_0based"])
        end = int(amplicon["consensus_end_0based_exclusive"])
        rows.append(
            (
                amplicon["amplicon_id"].replace("_", " ").title(),
                f"{amplicon['chrom']}:{start + 1:,}–{end:,}",
                f"{int(amplicon['consensus_length_bp']):,} bp",
                f"{int(amplicon['total_aligned_reads']):,}",
                f"{int(amplicon['n_called_snps']):,}",
            )
        )
    table = axis.table(
        cellText=rows,
        colLabels=("Amplicon", "Consensus coordinates", "Length", "Reads", "SNPs"),
        cellLoc="left",
        colLoc="left",
        colWidths=(0.16, 0.38, 0.16, 0.14, 0.10),
        bbox=(0.0, 0.0, 1.0, 0.93),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(max(5.4, 8.0 - 0.09 * len(amplicons)))
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#edf3f8")
            cell.set_text_props(weight="bold", color="#333333")
        elif row % 2 == 0:
            cell.set_facecolor("#f8f8f8")
    minimum = variant.get("parameters", {}).get("min_amplicon_reads", "?")
    axis.set_title(
        f"Amplicon inventory — primary MAPQ-filtered coverage ≥{minimum} reads",
        fontsize=9,
        pad=3,
    )


def _state_plot_text(result: dict) -> str:
    block = result.get("state_rates") or {}
    if not block.get("available"):
        return ""
    efficiency = result.get("efficiency") or {}
    background = result.get("background") or {}
    ratio = block.get("msp_to_outside_ratio")
    source = "calls" if block.get("source") == "tags" else "light call"
    return "\n".join((
        f"states ({source}); MSP >= {block['definition']['min_msp_bp']} bp",
        f"in-MSP (efficiency) {_pct(efficiency.get('value'))} "
        f"[{efficiency.get('status')}]",
        f"outside-MSP (background) {_pct(background.get('value'))} "
        f"[{background.get('status')}]",
        f"ratio {'NA' if ratio is None else f'{ratio:.1f}x'}; "
        f"MSP {_pct(block.get('msp_length_fraction'), 1)} of length",
    ))


def _plot_qc(
    result: dict,
    arrays: dict,
    reference: Optional[dict],
    control_curve: Optional[dict],
    control_examples: Optional[dict],
    png_path: Path,
    pdf_path: Path,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        # TrueType (Type 42) font embedding keeps text selectable/editable in
        # Adobe Illustrator; the default Type 3 glyphs are much less useful
        # when assembling publication figures.
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    status_colors = {"PASS": "#2ca25f", "WARN": "#f0a202", "FAIL": "#d1495b", "INSUFFICIENT": "#777777"}
    sample_color = status_colors[result["overall"]["status"]]
    figure = plt.figure(figsize=(17.0, 19.5), constrained_layout=False)
    grid = figure.add_gridspec(
        5, 6, height_ratios=(1.0, 0.9, 0.82, 0.82, 1.22)
    )
    axes = np.asarray(
        [
            [figure.add_subplot(grid[0, :3]), figure.add_subplot(grid[0, 3:])],
            [figure.add_subplot(grid[1, :2]), figure.add_subplot(grid[1, 2:4])],
        ]
    )
    dedup_axis = figure.add_subplot(grid[1, 4:])
    snp_axis = figure.add_subplot(grid[2:4, :3])
    snp_right_grid = grid[2:4, 3:].subgridspec(
        2, 1, height_ratios=(1.05, 0.85), hspace=0.25
    )
    snp_map_axis = figure.add_subplot(snp_right_grid[0])
    amplicon_table_axis = figure.add_subplot(snp_right_grid[1])
    molecule_axis = figure.add_subplot(grid[4, :])

    rates = np.sort(arrays["rates"] * 100)
    if len(rates):
        axes[0, 0].step(rates, np.arange(1, len(rates) + 1) / len(rates), where="post", color=sample_color, linewidth=2.2, label=f"sample (n={len(rates):,})")
    if reference and reference.get("scoring_enabled", True):
        rr = reference["rate"]
        if control_curve:
            probs = np.asarray(
                control_curve["rate_ecdf"]["probabilities"], dtype=float
            )
            values = np.asarray(control_curve["rate_ecdf"]["rates"], dtype=float) * 100
        else:
            probs = np.asarray(rr["reference_quantile_probabilities"], dtype=float)
            values = np.asarray(rr["reference_quantiles"], dtype=float) * 100
        axes[0, 0].plot(values, probs, color="#222222", linestyle="--", linewidth=1.5, marker="o", markersize=3, label=reference["label"])
        warn_lo, warn_hi = np.asarray(rr["warn_interval"]) * 100
        q25, q75 = np.asarray(rr["reference_iqr"]) * 100
        axes[0, 0].axvspan(
            warn_lo,
            warn_hi,
            color="#777777",
            alpha=0.055,
            label="reference 5th–95th (WARN)",
        )
        axes[0, 0].axvspan(
            q25,
            q75,
            color="#777777",
            alpha=0.14,
            label="reference IQR (PASS)",
        )
        axes[0, 0].axvline(100 * rr["reference_median"], color="#555555", linestyle=":", linewidth=1)
        sample_median = result["signal"]["median_per_read_rate"]
        if sample_median is not None:
            sample_median *= 100
            reference_median = 100 * rr["reference_median"]
            axes[0, 0].plot(
                [sample_median, reference_median],
                [0.5, 0.5],
                color="#777777",
                linewidth=1.1,
                zorder=5,
            )
            axes[0, 0].scatter(
                [sample_median, reference_median],
                [0.5, 0.5],
                color=[sample_color, "#222222"],
                marker="D",
                s=38,
                edgecolor="white",
                linewidth=0.6,
                zorder=6,
            )
            axes[0, 0].annotate(
                f"sample median={sample_median:.2f}% | "
                f"control={reference_median:.2f}% | "
                f"Δ={sample_median - reference_median:+.2f} pp",
                ((sample_median + reference_median) / 2, 0.5),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#444444",
            )
    state_text = _state_plot_text(result)
    if state_text:
        axes[0, 0].text(
            0.98, 0.04, state_text, ha="right", va="bottom",
            transform=axes[0, 0].transAxes, fontsize=8, color="#333333",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                  "edgecolor": "#cccccc", "alpha": 0.9},
        )
    axes[0, 0].set(xlabel=f"per-read {result['signal']['label']} rate (%)", ylabel="fraction of fibers <= rate", title=(f"Signal rate [{result['signal']['status']}"
                   + (", not in verdict]"
                      if result["overall"].get("verdict_basis") == "state-aware" else "]")))
    axes[0, 0].set_ylim(0, 1.01)
    axes[0, 0].legend(frameon=False, fontsize=8)

    lag = np.arange(len(arrays["phasogram"]))
    axes[0, 1].plot(lag[60:PLOT_MAX_LAG + 1], arrays["phasogram"][60:PLOT_MAX_LAG + 1], color=sample_color, linewidth=1.5, label="sample")
    if reference and reference.get("scoring_enabled", True):
        if control_curve:
            control_lags = np.asarray(
                control_curve["phasogram"]["lags_bp"], dtype=int
            )
            control_values = np.asarray(
                control_curve["phasogram"]["detrended_pair_frequency"],
                dtype=float,
            )
            axes[0, 1].plot(
                control_lags,
                control_values,
                color="#222222",
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
                label=reference["label"],
            )
        ref_nrl = int(reference["periodicity"]["reference_nrl_bp"])
        for multiple in range(1, 1 + PLOT_MAX_LAG // ref_nrl):
            axes[0, 1].axvline(ref_nrl * multiple, color="#555555", linestyle=":" if multiple > 1 else "--", linewidth=0.9, alpha=0.8)
        lo, hi = reference["periodicity"]["search_interval_bp"]
        axes[0, 1].axvspan(lo, hi, color="#4c78a8", alpha=0.08, label=f"reference NRL={ref_nrl} bp")
    elif reference:
        axes[0, 1].text(
            0.98,
            0.95,
            "reference calibration pending",
            ha="right",
            va="top",
            transform=axes[0, 1].transAxes,
            color="#777777",
            fontsize=8,
        )
    correlation = result["periodicity"].get("reference_curve_correlation")
    amplitude = result["periodicity"].get("reference_pattern_amplitude_ratio")
    comparison = ""
    if correlation is not None and amplitude is not None:
        comparison = f" — reference r={correlation:.2f}, matched amplitude={amplitude:.2f}x"
    axes[0, 1].set(
        xlabel="within-read signal-pair distance (bp)",
        ylabel="detrended pair frequency",
        title=(
            f"Nucleosome-scale periodicity [{result['periodicity']['status']}]"
            f"{comparison}"
        ),
    )
    axes[0, 1].legend(frameon=False, fontsize=8)

    _plot_footprint_distribution(
        axes[1, 0], arrays["nuc_lengths"], control_curve, "nucleosome",
        "#4c78a8", "Nucleosome-tagged protected-span distribution",
        "nucleosome-tagged span size (bp)",
    )
    _plot_footprint_distribution(
        axes[1, 1], arrays["tf_lengths"], control_curve, "tf",
        "#e45756", "TF footprint-size distribution",
        "TF footprint size (bp)",
    )
    _plot_deduplication(dedup_axis, result["deduplication"])
    _plot_snp_landscape(figure, snp_axis, result, arrays)
    _plot_snp_map(snp_map_axis, result, arrays)
    _plot_amplicon_table(amplicon_table_axis, result)

    _plot_read_examples(
        molecule_axis,
        arrays.get("read_examples", []),
        control_examples,
        sample_color,
        result["signal"]["label"],
    )

    for axis in (*axes.flat, dedup_axis, snp_axis, snp_map_axis, molecule_axis):
        axis.grid(color="#ececec", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"FiberHMM QC — {result['overall']['status']} ({_fmt_score(result['overall']['score'])}/100)",
        fontsize=15,
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.045,
        top=0.945,
        hspace=0.52,
        wspace=0.38,
    )
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    figure.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": "FiberHMM QC",
            "Creator": "FiberHMM",
            "Subject": "Assay-aware FiberHMM quality-control report",
        },
    )
    plt.close(figure)
    return True


def _rate_stratified_examples(examples: Sequence[dict], maximum: int) -> list[dict]:
    ordered = sorted(examples, key=lambda item: (item.get("rate", 0.0), item["span_end"] - item["span_start"]))
    if len(ordered) <= maximum:
        return ordered
    indices = np.rint(np.linspace(0, len(ordered) - 1, maximum)).astype(int)
    return [ordered[index] for index in indices]


def _plot_read_examples(
    axis,
    sample_examples: Sequence[dict],
    control_examples: Optional[dict],
    sample_color: str,
    signal_label: str,
) -> None:
    """Plot centered, anonymous molecule hatchmarks without writing read data."""
    from matplotlib.lines import Line2D

    sample_rows = _rate_stratified_examples(sample_examples, 32)
    control_groups = (control_examples or {}).get("groups", [])
    rows: list[tuple[dict, str, str]] = [
        (example, sample_color, "sample") for example in sample_rows
    ]
    group_starts: list[tuple[int, str]] = []
    if rows and control_groups:
        rows.append(({}, "", "gap"))
        rows.append(({}, "", "gap"))
    for group in control_groups:
        group_starts.append((len(rows), group["label"]))
        for read in group.get("reads", []):
            rows.append(
                (
                    {
                        "span_start": 0,
                        "span_end": int(read["span_bp"]),
                        "positions": np.asarray(
                            read["signal_positions_bp"], dtype=np.int64
                        ),
                        "rate": float(read.get("signal_rate", 0.0)),
                    },
                    "#222222",
                    "control",
                )
            )
    if not rows:
        axis.text(
            0.5,
            0.5,
            "representative signal patterns unavailable",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_title("Representative individual molecules")
        return

    plotted_y = []
    half_spans = []
    for y, (example, color, source) in enumerate(rows):
        if source == "gap":
            continue
        start = int(example["span_start"])
        end = int(example["span_end"])
        center = (start + end) / 2
        half_spans.append((end - start) / 2000)
        plotted_y.append(y)
        axis.hlines(
            y,
            (start - center) / 1000,
            (end - center) / 1000,
            color="#cfcfcf",
            linewidth=0.75,
        )
        positions = np.asarray(example["positions"], dtype=float)
        if len(positions):
            axis.vlines(
                (positions - center) / 1000,
                y - 0.38,
                y + 0.38,
                color=color,
                linewidth=0.45,
                alpha=0.9,
            )
    if sample_rows:
        axis.text(
            0.005,
            0,
            f"sample: {len(sample_rows)} rate-stratified molecules",
            transform=axis.get_yaxis_transform(),
            va="bottom",
            color=sample_color,
            fontsize=8.5,
            fontweight="bold",
        )
    for start, label in group_starts:
        axis.text(
            0.005,
            start - 0.7,
            f"control: {label}",
            transform=axis.get_yaxis_transform(),
            va="bottom",
            color="#222222",
            fontsize=8.5,
            fontweight="bold",
        )
    if sample_rows and control_groups:
        axis.axhline(
            len(sample_rows) + 0.5,
            color="#888888",
            linewidth=0.8,
            linestyle=":",
        )
    half_span = max(half_spans, default=1.0)
    axis.set_xlim(-half_span * 1.04, half_span * 1.04)
    axis.set_ylim(max(plotted_y, default=0) + 0.7, -2.6)
    axis.set_yticks([])
    axis.axvline(0, color="#eeeeee", linewidth=0.8, zorder=0)
    axis.set_xlabel("molecule span relative to midpoint (kb)")
    axis.set_title(
        "Representative individual molecules — sample versus matched controls"
    )
    axis.legend(
        handles=(
            Line2D([0], [0], color="#cfcfcf", linewidth=2, label="molecule"),
            Line2D([0], [0], color=sample_color, linewidth=1, label=f"sample {signal_label}"),
            Line2D([0], [0], color="#222222", linewidth=1, label=f"control {signal_label}"),
        ),
        frameon=False,
        ncol=3,
        fontsize=8,
        loc="upper center",
    )


STATE_TSV_FIELDS = (
    "state_source",
    "efficiency_status",
    "efficiency_score",
    "background_status",
    "background_score",
    "median_msp_rate",
    "median_outside_msp_rate",
    "aggregate_msp_rate",
    "aggregate_outside_msp_rate",
    "msp_events",
    "msp_opportunities",
    "outside_msp_events",
    "outside_msp_opportunities",
    "all_states_rate",
    "msp_to_outside_ratio",
    "msp_length_fraction",
    "verdict_basis",
)


def _state_tsv_fields(result: dict) -> dict:
    block = result.get("state_rates") or {}
    msp = block.get("msp") or {}
    outside = block.get("outside_msp") or {}
    efficiency = result.get("efficiency") or {}
    background = result.get("background") or {}
    return {
        "state_source": block.get("source"),
        "efficiency_status": efficiency.get("status"),
        "efficiency_score": efficiency.get("score"),
        "background_status": background.get("status"),
        "background_score": background.get("score"),
        "median_msp_rate": msp.get("median_per_read_rate"),
        "median_outside_msp_rate": outside.get("median_per_read_rate"),
        "aggregate_msp_rate": msp.get("aggregate_rate"),
        "aggregate_outside_msp_rate": outside.get("aggregate_rate"),
        "msp_events": msp.get("n_events"),
        "msp_opportunities": msp.get("n_opportunities"),
        "outside_msp_events": outside.get("n_events"),
        "outside_msp_opportunities": outside.get("n_opportunities"),
        "all_states_rate": (block.get("all_states") or {}).get("aggregate_rate"),
        "msp_to_outside_ratio": block.get("msp_to_outside_ratio"),
        "msp_length_fraction": block.get("msp_length_fraction"),
        "verdict_basis": (result.get("overall") or {}).get("verdict_basis"),
    }


def _write_tsv(result: dict, path: Path) -> None:
    fields = {
        "input": result["input"],
        "mode": result["assay"]["mode"],
        "enzyme": result["assay"]["enzyme"] or "",
        "reference_profile": result["assay"]["reference_profile"] or "",
        "sampled_reads": result["sampling"]["sampled_reads"],
        "overall_status": result["overall"]["status"],
        "overall_score": result["overall"]["score"],
        "signal_status": result["signal"]["status"],
        "signal_score": result["signal"]["score"],
        "median_signal_rate": result["signal"]["median_per_read_rate"],
        "periodicity_status": result["periodicity"]["status"],
        "periodicity_score": result["periodicity"]["score"],
        "nrl_bp": result["periodicity"]["nrl_bp"],
        "periodicity_ac": result["periodicity"]["autocorrelation_strength"],
        "periodicity_reference_correlation": result["periodicity"].get(
            "reference_curve_correlation"
        ),
        "periodicity_reference_pattern_amplitude_ratio": result["periodicity"].get(
            "reference_pattern_amplitude_ratio"
        ),
        "n_nucleosomes": result["footprints"]["n_nucleosomes"],
        "median_nucleosome_bp": result["footprints"]["median_nucleosome_bp"],
        "fraction_nucleosome_85_250_bp": result["footprints"].get(
            "fraction_nucleosome_85_250_bp"
        ),
        "fraction_nucleosome_over_300_bp": result["footprints"].get(
            "fraction_nucleosome_over_300_bp"
        ),
        "fraction_nucleosome_over_1000_bp": result["footprints"].get(
            "fraction_nucleosome_over_1000_bp"
        ),
        "n_tf_footprints": result["footprints"]["n_tf_footprints"],
        "median_tf_footprint_bp": result["footprints"]["median_tf_footprint_bp"],
        "deduplication_detected": result["deduplication"]["detected"],
        "deduplication_mode": result["deduplication"]["mode"],
        "duplicate_fraction": result["deduplication"]["duplicate_fraction"],
        "snp_mask_applied": result["variant_masking"]["applied"],
        "n_masked_snp_sites": result["variant_masking"]["n_masked_sites"],
        "n_discovered_amplicons": result["variant_masking"].get(
            "n_discovered_amplicons", 0
        ),
        **_state_tsv_fields(result),
    }
    with path.open("w") as handle:
        handle.write("\t".join(fields) + "\n")
        handle.write("\t".join("" if value is None else str(value) for value in fields.values()) + "\n")


QC_CURVES_SCHEMA = "fiberhmm.qc.curves.v1"


def _histogram_payload(values: np.ndarray, control: Optional[dict], step: int,
                       stop: int) -> dict:
    edges = (np.asarray(control["bin_edges_bp"], dtype=float) if control
             else np.arange(0, stop + step, step, dtype=float))
    values = np.asarray(values, dtype=float)
    counts, _ = np.histogram(values, bins=edges)
    payload = {
        "bin_edges_bp": edges.tolist(),
        "sample_count_per_bin": counts.astype(int).tolist(),
        "sample_fraction_per_bin": (counts / len(values)).tolist() if len(values) else [],
        "sample_n": int(len(values)),
        "sample_overflow": int(np.sum(values >= edges[-1])) if len(values) else 0,
        "reference_fraction_per_bin": list(control["fraction_per_bin"]) if control else None,
    }
    return payload


def _state_curves(result: dict, arrays: dict) -> Optional[dict]:
    """Per-read in-MSP/outside-MSP rates and the reference quantiles."""
    block = result.get("state_rates") or {}
    state_arrays = arrays.get("state_rates") or {}
    if not block.get("available"):
        return None
    reference = block.get("reference") or {}
    return {
        "source": block.get("source"),
        "min_msp_bp": (block.get("definition") or {}).get("min_msp_bp"),
        "msp_per_read_rates": np.sort(state_arrays.get("msp_rates", [])).tolist(),
        "outside_msp_per_read_rates":
            np.sort(state_arrays.get("outside_rates", [])).tolist(),
        "reference": {
            key: {"quantiles": (reference.get(key) or {}).get("quantiles"),
                  "probabilities": (reference.get(key) or {}).get("probabilities")}
            for key in ("msp", "outside_msp")
        } if reference else None,
    }


def qc_curves(result: dict, arrays: dict, profile: Optional[dict],
              control_curve: Optional[dict]) -> dict:
    """The curves behind the QC plot, as JSON (``<prefix>.qc.curves.json``).

    Schema ``fiberhmm.qc.curves.v1``: the per-read signal-rate distribution
    and its ECDF with the bundled reference ECDF, the detrended phasogram
    with the reference curve, nucleosome/TF footprint-size histograms with
    the reference fractions, the duplicate cluster-size histogram, and the
    verdicts. Rates are fractions (not percent); the reference parts are
    ``null`` when the assay has no bundled reference.
    """
    rates = np.sort(np.asarray(arrays.get("rates", []), dtype=float))
    reference_ecdf = None
    rate_reference = (profile or {}).get("rate") if profile else None
    if control_curve and control_curve.get("rate_ecdf"):
        reference_ecdf = {
            "rates": list(control_curve["rate_ecdf"]["rates"]),
            "probabilities": list(control_curve["rate_ecdf"]["probabilities"]),
            "source": "packaged control curve",
        }
    elif rate_reference:
        reference_ecdf = {
            "rates": list(rate_reference.get("reference_quantiles", [])),
            "probabilities": list(rate_reference.get("reference_quantile_probabilities", [])),
            "source": "reference quantiles",
        }
    curve = np.asarray(arrays.get("phasogram", []), dtype=float)
    raw = np.asarray(arrays.get("histogram", []), dtype=float)
    reference_phasogram = None
    if control_curve and control_curve.get("phasogram"):
        reference_phasogram = {
            "lags_bp": list(control_curve["phasogram"]["lags_bp"]),
            "detrended_pair_frequency":
                list(control_curve["phasogram"]["detrended_pair_frequency"]),
        }
    sizes = (control_curve or {}).get("footprint_sizes") or {}
    return {
        "schema": QC_CURVES_SCHEMA,
        "input": result.get("input"),
        "assay": result.get("assay"),
        "verdicts": {
            "overall": result["overall"]["status"],
            "overall_score": result["overall"]["score"],
            "signal": result["signal"]["status"],
            "signal_score": result["signal"]["score"],
            "periodicity": result["periodicity"]["status"],
            "periodicity_score": result["periodicity"]["score"],
            # Additive (QC report schema 1.1): state-aware verdict parts.
            "efficiency": (result.get("efficiency") or {}).get("status"),
            "efficiency_score": (result.get("efficiency") or {}).get("score"),
            "background": (result.get("background") or {}).get("status"),
            "background_score": (result.get("background") or {}).get("score"),
            "verdict_basis": (result.get("overall") or {}).get("verdict_basis"),
        },
        "state_rates": _state_curves(result, arrays),
        "signal_rate": {
            "label": result["signal"]["label"],
            "per_read_rates": rates.tolist(),
            "ecdf": ((np.arange(1, len(rates) + 1) / len(rates)).tolist()
                     if len(rates) else []),
            "reference_ecdf": reference_ecdf,
            "warn_interval": list(rate_reference["warn_interval"])
            if rate_reference and "warn_interval" in rate_reference else None,
            "reference_iqr": list(rate_reference["reference_iqr"])
            if rate_reference and "reference_iqr" in rate_reference else None,
        },
        "phasogram": {
            "lags_bp": list(range(len(curve))),
            "detrended_pair_frequency": curve.tolist(),
            "pair_distance_counts": raw.astype(int).tolist(),
            "nrl_bp": result["periodicity"]["nrl_bp"],
            "reference": reference_phasogram,
        },
        "footprint_sizes": {
            "nucleosome": _histogram_payload(arrays.get("nuc_lengths", []),
                                             sizes.get("nucleosome"), 10, 2000),
            "tf": _histogram_payload(arrays.get("tf_lengths", []), sizes.get("tf"), 5, 500),
        },
        "duplicates": {
            "duplicate_fraction": result.get("deduplication", {}).get("duplicate_fraction"),
            "molecule_cluster_size_histogram":
                result.get("deduplication", {}).get("molecule_cluster_size_histogram"),
        },
    }


QC_SCHEMA_MINOR_VERSION = 1


def _state_rates(input_path, reads, mode, enzyme, **options) -> tuple[dict, dict]:
    """State-aware rates; a failure is reported in the block, never raised."""
    if options.get("state_source") in ("none", "", None):
        return {"available": False, "source": None,
                "note": "state-aware rates disabled (--state-source none)"}, {}
    try:
        with pysam.AlignmentFile(input_path, "rb", check_sq=False) as handle:
            header = handle.header
    except (OSError, ValueError):
        header = None
    try:
        return compute_state_rates(reads, mode, enzyme, header=header, **options)
    except Exception as exc:  # QC must still report everything else
        return {"available": False, "source": None,
                "note": f"state-aware rates failed: {type(exc).__name__}: {exc}"}, {}


def _overall_verdict(analysis: dict) -> dict:
    """Overall status from the graded components.

    When the in-MSP efficiency and/or outside-MSP background are graded they
    replace the overall-rate grade (which stays in ``signal`` as a reported
    metric): the overall rate also reflects how much of the DNA is accessible,
    so it confounds enzyme efficiency with the chromatin of the sample.
    """
    graded_states = [name for name in ("efficiency", "background")
                     if analysis[name]["score"] is not None]
    state_graded = bool(graded_states)
    # A state component without evidence is reported, not counted as a WARN,
    # when the other one is graded; every graded failure is kept.
    names = ((*graded_states, "periodicity") if state_graded
             else ("signal", "periodicity"))
    statuses = [analysis[name]["status"] for name in names]
    scored = [analysis[name]["score"] for name in names
              if analysis[name]["score"] is not None]
    if not scored:
        status = "INSUFFICIENT"
    elif "FAIL" in statuses:
        status = "FAIL"
    elif "WARN" in statuses or "INSUFFICIENT" in statuses:
        status = "WARN"
    else:
        status = "PASS"
    return {
        "score": float(np.mean(scored)) if scored else None,
        "status": status,
        "components": list(names),
        "verdict_basis": "state-aware" if state_graded else "overall-rate",
    }


def run_qc(
    input_path: str,
    output_prefix: Optional[str] = None,
    mode: str = "auto",
    enzyme: str = "auto",
    reference_profile: str = "auto",
    reference_fasta: Optional[str] = None,
    sample_reads: int = DEFAULT_SAMPLE_READS,
    seed: int = DEFAULT_SEED,
    min_mapq: int = 20,
    prob_threshold: Optional[int] = None,
    min_opportunities: int = 200,
    snp_report_path: Optional[str] = None,
    snp_mask_path: Optional[str] = None,
    snp_preflight_summary: Optional[dict] = None,
    dedup_run_summary: Optional[dict] = None,
    stream: Optional[TextIO] = sys.stderr,
    return_arrays: bool = False,
    state_source: str = "auto",
    min_msp_bp: int = DEFAULT_MIN_MSP_BP,
    light_call_reads: int = DEFAULT_LIGHT_CALL_READS,
    light_call_seconds: float = DEFAULT_LIGHT_CALL_SECONDS,
) -> dict:
    """Run bounded QC, write machine-readable output/plot, and print a scorecard.

    ``state_source`` selects where the in-MSP/outside-MSP split comes from:
    ``auto`` (the BAM's FiberHMM calls when the sample carries them, else a
    bounded light call), ``tags``, ``light-call`` or ``none`` (skip).
    """
    sampled = sample_bam_reads(
        input_path,
        sample_reads=sample_reads,
        seed=seed,
        min_mapq=min_mapq,
    )
    resolved_mode, resolved_enzyme, inferred_profile = infer_assay(
        input_path, sampled.reads, mode=mode, enzyme=enzyme
    )
    if prob_threshold is None:
        prob_threshold = default_qc_prob_threshold(resolved_mode, resolved_enzyme)
    profile_key = inferred_profile if reference_profile == "auto" else reference_profile
    if (
        reference_profile not in ("auto", "none", "", None)
        and inferred_profile
        and reference_profile != inferred_profile
    ):
        raise QCInputError(
            f"QC reference {reference_profile!r} is incompatible with "
            f"mode={resolved_mode!r}, enzyme={resolved_enzyme!r}; "
            f"use {inferred_profile!r} (or 'none')"
        )
    if profile_key in ("", "none", None):
        profile_key = ""
        profile = None
    else:
        profiles = load_references()["profiles"]
        if profile_key not in profiles:
            raise QCInputError(
                f"unknown QC reference profile {profile_key!r}; "
                f"choose one of {', '.join(sorted(profiles))} or 'none'"
            )
        profile = profiles[profile_key]
    curves = load_control_curves()
    control_curve = (
        curves.get("profiles", {}).get(profile_key) if profile else None
    )

    # Integrated fiberhmm-call writes aggregate sidecars into its qc/
    # directory. Discover only exact stem matches so standalone multi-BAM QC
    # can faithfully reproduce the SNP and full-run dedup panels.
    input_object = Path(input_path)
    input_stem = input_object.with_suffix("").name
    sidecar_dir = input_object.resolve().parent / "qc"
    if snp_report_path is None:
        candidate = sidecar_dir / f"{input_stem}.daf_snps.json"
        if candidate.exists():
            snp_report_path = str(candidate)
    if snp_mask_path is None and snp_report_path and Path(snp_report_path).exists():
        try:
            report_outputs = json.loads(Path(snp_report_path).read_text()).get("outputs", {})
            candidate = report_outputs.get("bed")
            if candidate and Path(candidate).exists():
                snp_mask_path = candidate
        except (OSError, ValueError, TypeError):
            pass
    dedup_report_path = None
    candidate = sidecar_dir / f"{input_stem}.dedup.json"
    if candidate.exists():
        dedup_report_path = str(candidate)
        if dedup_run_summary is None:
            try:
                dedup_payload = json.loads(candidate.read_text())
                dedup_run_summary = dedup_payload.get("statistics")
            except (OSError, ValueError, TypeError):
                pass

    if snp_mask_path:
        from fiberhmm.daf.snps import load_snp_mask

        snp_mask = load_snp_mask(snp_mask_path)
    else:
        snp_mask = None
    analysis, arrays = analyze_sample(
        sampled.reads,
        resolved_mode,
        profile,
        control_curve=control_curve,
        reference_fasta=reference_fasta,
        prob_threshold=prob_threshold,
        min_opportunities=min_opportunities,
        snp_mask=snp_mask,
    )
    state_block, state_arrays = _state_rates(
        input_path, sampled.reads, resolved_mode, resolved_enzyme,
        prob_threshold=prob_threshold, reference_fasta=reference_fasta,
        snp_mask=snp_mask, state_source=state_source, seed=seed,
        min_msp_bp=min_msp_bp, light_call_reads=light_call_reads,
        light_call_seconds=light_call_seconds,
    )
    efficiency, background = grade_state_rates(state_block, profile, prob_threshold)
    reference_states = ((profile or {}).get("state_rates") or {}).get("by_source", {})
    if state_block.get("source") in reference_states:
        state_block["reference"] = reference_states[state_block["source"]]
    analysis["state_rates"] = state_block
    analysis["efficiency"] = efficiency
    analysis["background"] = background
    analysis["overall"] = _overall_verdict(analysis)
    arrays["state_rates"] = state_arrays
    prefix = Path(output_prefix) if output_prefix else Path(str(input_path)).with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(str(prefix) + ".qc.json")
    tsv_path = Path(str(prefix) + ".qc.tsv")
    plot_path = Path(str(prefix) + ".qc.png")
    pdf_path = Path(str(prefix) + ".qc.pdf")
    from fiberhmm import __version__ as fiberhmm_version

    result = {
        "schema_version": 1,
        # Additive revisions of schema 1: 1 = state_rates/efficiency/background.
        "schema_minor_version": QC_SCHEMA_MINOR_VERSION,
        # Read by fiberhmm.advisories (re-run QC after a fix); since 3.0.
        "fiberhmm_version": fiberhmm_version,
        "input": str(Path(input_path).resolve()),
        "assay": {
            "mode": resolved_mode,
            "enzyme": resolved_enzyme,
            "reference_profile": profile_key or None,
            "reference_label": profile.get("label") if profile else None,
            "reference_sources": profile.get("sources", []) if profile else [],
            "probability_threshold": int(prob_threshold),
        },
        "sampling": {
            "requested_reads": int(sample_reads),
            "sampled_reads": len(sampled.reads),
            "strategy": sampled.strategy,
            "records_examined": sampled.records_examined,
            "windows_examined": sampled.windows_examined,
            "seed": int(seed),
            "whole_bam_scanned": False,
        },
        **analysis,
        "outputs": {
            "json": str(json_path.resolve()),
            "tsv": str(tsv_path.resolve()),
            "plot": None,
            "pdf": None,
        },
    }
    result["deduplication"] = _merge_full_dedup_summary(
        result["deduplication"], dedup_run_summary
    )
    if dedup_report_path:
        result["deduplication"]["run_report_path"] = dedup_report_path
    variant_masking, snp_calls, snp_site_distribution = _load_variant_masking(
        snp_report_path,
        snp_mask_path,
    )
    result["variant_masking"] = variant_masking
    if snp_preflight_summary is not None:
        result["variant_masking"]["screening_preflight"] = dict(
            snp_preflight_summary
        )
        if (
            snp_preflight_summary.get("run") is False
            and not result["variant_masking"]["applied"]
        ):
            result["variant_masking"]["note"] = (
                "automatic screen skipped after bounded low-coverage preflight"
            )
    arrays["snp_calls"] = snp_calls
    arrays["snp_site_distribution"] = snp_site_distribution
    examples = load_control_examples()
    control_examples = (
        examples.get("profiles", {}).get(profile_key) if profile else None
    )
    result["assay"]["packaged_control_curve"] = control_curve is not None
    result["assay"]["control_curve_schema_version"] = (
        curves.get("schema_version") if control_curve is not None else None
    )
    plotted = _plot_qc(
        result,
        arrays,
        profile,
        control_curve,
        control_examples,
        plot_path,
        pdf_path,
    )
    if plotted:
        result["outputs"]["plot"] = str(plot_path.resolve())
        result["outputs"]["pdf"] = str(pdf_path.resolve())
    curves_path = Path(str(prefix) + ".qc.curves.json")
    _atomic_json(qc_curves(result, arrays, profile, control_curve), curves_path)
    result["outputs"]["curves"] = str(curves_path.resolve())
    _atomic_json(result, json_path)
    _write_tsv(result, tsv_path)
    if stream is not None:
        print(format_terminal(result), file=stream)
    if return_arrays:
        # Used only transiently by run_multi_qc. These arrays are added after
        # JSON serialization and are never written into a report.
        result["_plot_arrays"] = arrays
    return result


def _combined_status(results: Sequence[dict]) -> tuple[str, Optional[float]]:
    statuses = [result["overall"]["status"] for result in results]
    scores = [
        float(result["overall"]["score"])
        for result in results
        if result["overall"]["score"] is not None
    ]
    if "FAIL" in statuses:
        status = "FAIL"
    elif "WARN" in statuses:
        status = "WARN"
    elif statuses and all(item == "PASS" for item in statuses):
        status = "PASS"
    else:
        status = "INSUFFICIENT"
    return status, (float(np.mean(scores)) if scores else None)


def _plot_combined_qc(
    results: Sequence[dict],
    png_path: Path,
    pdf_path: Path,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    n_samples = len(results)
    labels = []
    for result in results:
        label = Path(result["input"]).name.removesuffix(".bam").removesuffix(".cram")
        for suffix in (".aligned.sorted", ".aligned_footprints", ".encoded_footprints"):
            label = label.removesuffix(suffix)
        # Hide FiberHMM processing/provenance suffixes in user-facing plots;
        # the full filenames remain in JSON/TSV for traceability.
        if ".fiberhmm" in label:
            label = label.split(".fiberhmm", 1)[0]
        labels.append(label)
    display_labels = [
        f"{label} [{result['overall']['status']}]"
        for label, result in zip(labels, results)
    ]
    y = np.arange(n_samples)
    colors = plt.get_cmap("tab10")(np.arange(n_samples) % 10)
    figure, axes = plt.subplots(
        4,
        2,
        figsize=(15.5, max(15.0, 0.4 * n_samples + 13)),
        constrained_layout=True,
    )

    # Signal-rate comparison: each sample's median and IQR, with either one
    # shared reference band or per-row reference medians for mixed assays.
    rate_axis = axes[0, 0]
    for index, (result, color) in enumerate(zip(results, colors)):
        signal = result["signal"]
        median = signal["median_per_read_rate"]
        q25 = signal["q25_per_read_rate"]
        q75 = signal["q75_per_read_rate"]
        if median is None or q25 is None or q75 is None:
            continue
        rate_axis.errorbar(
            100 * median,
            index,
            xerr=[[100 * (median - q25)], [100 * (q75 - median)]],
            fmt="o",
            color=color,
            capsize=3,
            markersize=6,
        )
        rate_axis.annotate(
            f"{100 * median:.2f}%",
            (100 * median, index),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color="#444444",
        )
    references = load_references()["profiles"]
    profile_keys = [result["assay"]["reference_profile"] for result in results]
    unique_profiles = sorted({key for key in profile_keys if key in references})
    if len(unique_profiles) == 1:
        reference = references[unique_profiles[0]]["rate"]
        warn_lo, warn_hi = reference["warn_interval"]
        q25, q75 = reference["reference_iqr"]
        rate_axis.axvspan(
            100 * warn_lo,
            100 * warn_hi,
            color="#777777",
            alpha=0.055,
            label="control 5th–95th (WARN)",
        )
        rate_axis.axvspan(
            100 * q25,
            100 * q75,
            color="#777777",
            alpha=0.14,
            label="control IQR (PASS)",
        )
        rate_axis.axvline(100 * reference["reference_median"], color="#333333", linestyle="--", linewidth=1.3, label="control median")
    elif unique_profiles:
        for index, profile_key in enumerate(profile_keys):
            if profile_key in references:
                reference_median = references[profile_key]["rate"]["reference_median"]
                rate_axis.plot(100 * reference_median, index, marker="x", color="#222222", markersize=7)
    rate_axis.set_yticks(y, display_labels)
    rate_axis.invert_yaxis()
    rate_axis.set_xlabel("median per-read signal rate (%) with IQR")
    rate_axis.set_title("Signal-rate comparison")
    if unique_profiles:
        rate_axis.legend(frameon=False, fontsize=8)

    # A compact phase-quality score is more comparable than overplotting raw
    # phasograms. The curve correlation annotation captures reference-shape
    # agreement independently of NRL and autocorrelation strength.
    score_axis = axes[0, 1]
    score_axis.axvspan(0, 35, color="#d1495b", alpha=0.07)
    score_axis.axvspan(35, 70, color="#f0a202", alpha=0.07)
    score_axis.axvspan(70, 100, color="#2ca25f", alpha=0.07)
    for index, (result, color) in enumerate(zip(results, colors)):
        score = result["periodicity"]["score"]
        if score is None:
            continue
        score_axis.plot(score, index, "o", color=color, markersize=6)
        correlation = result["periodicity"].get("reference_curve_correlation")
        amplitude = result["periodicity"].get(
            "reference_pattern_amplitude_ratio"
        )
        if correlation is not None and amplitude is not None:
            score_axis.annotate(
                f"r={correlation:.2f}, amp={amplitude:.2f}x",
                (score, index),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.5,
                color="#444444",
            )
    score_axis.axvline(70, color="#666666", linewidth=0.8, linestyle="--")
    score_axis.set_xlim(0, 112)
    score_axis.set_yticks(y, display_labels)
    score_axis.invert_yaxis()
    score_axis.set(
        xlabel="periodicity quality score (0–100)",
        title="Periodicity score, reference shape, and matched amplitude",
    )

    nrl_axis = axes[1, 0]
    for index, (result, color) in enumerate(zip(results, colors)):
        nrl = result["periodicity"]["nrl_bp"]
        if nrl is not None:
            nrl_axis.plot(nrl, index, "o", color=color, markersize=6)
    if len(unique_profiles) == 1:
        period_reference = references[unique_profiles[0]]["periodicity"]
        lower, upper = period_reference["search_interval_bp"]
        nrl_axis.axvspan(lower, upper, color="#777777", alpha=0.08)
        nrl_axis.axvline(
            period_reference["reference_nrl_bp"],
            color="#333333",
            linestyle="--",
            linewidth=1.3,
            label="control NRL",
        )
        nrl_axis.legend(frameon=False, fontsize=8)
    else:
        for index, profile_key in enumerate(profile_keys):
            if profile_key in references:
                nrl_axis.plot(
                    references[profile_key]["periodicity"]["reference_nrl_bp"],
                    index,
                    marker="x",
                    color="#222222",
                    markersize=7,
                )
    nrl_axis.set_yticks(y, display_labels)
    nrl_axis.invert_yaxis()
    nrl_axis.set(xlabel="best nucleosome repeat length (bp)", title="NRL comparison")

    strength_axis = axes[1, 1]
    for index, (result, color) in enumerate(zip(results, colors)):
        strength_axis.plot(
            result["periodicity"]["autocorrelation_strength"],
            index,
            "o",
            color=color,
            markersize=6,
        )
    if len(unique_profiles) == 1:
        period_reference = references[unique_profiles[0]]["periodicity"]
        strength_axis.axvline(
            period_reference["reference_strength"],
            color="#333333",
            linestyle="--",
            linewidth=1.3,
            label="control strength",
        )
        strength_axis.axvline(
            period_reference["pass_strength"],
            color="#2ca25f",
            linestyle=":",
            linewidth=1.0,
            label="pass threshold",
        )
        strength_axis.axvline(
            period_reference["warn_strength"],
            color="#f0a202",
            linestyle=":",
            linewidth=1.0,
            label="warn threshold",
        )
        strength_axis.legend(frameon=False, fontsize=8)
    else:
        for index, profile_key in enumerate(profile_keys):
            if profile_key in references:
                strength_axis.plot(
                    references[profile_key]["periodicity"]["reference_strength"],
                    index,
                    marker="x",
                    color="#222222",
                    markersize=7,
                )
    strength_axis.set_yticks(y, display_labels)
    strength_axis.invert_yaxis()
    strength_axis.set(
        xlabel="max phasogram autocorrelation, 160–220 bp",
        title="Periodicity strength comparison",
    )

    controls = load_control_curves().get("profiles", {})

    def footprint_panel(
        axis, key: str, control_key: str, title: str, xlabel: str
    ):
        found = False
        for index, (result, color) in enumerate(zip(results, colors)):
            summary = result["footprints"]
            median = summary[f"median_{key}_bp"]
            q25 = summary[f"q25_{key}_bp"]
            q75 = summary[f"q75_{key}_bp"]
            if median is None or q25 is None or q75 is None:
                continue
            found = True
            axis.errorbar(
                median,
                index,
                xerr=[[median - q25], [q75 - median]],
                fmt="o",
                color=color,
                capsize=3,
                markersize=6,
            )
            if key == "nucleosome":
                long_fraction = summary.get("fraction_nucleosome_over_300_bp")
                if long_fraction is not None:
                    axis.annotate(
                        f"{100 * long_fraction:.1f}% >300 bp",
                        (median, index),
                        xytext=(6, 10 if index == len(results) - 1 else -10),
                        textcoords="offset points",
                        va="center",
                        fontsize=7.3,
                        color="#555555",
                    )
        if len(unique_profiles) == 1:
            distribution = (
                controls.get(unique_profiles[0], {})
                .get("footprint_sizes", {})
                .get(control_key)
            )
            if distribution:
                axis.axvspan(
                    distribution["q25_bp"],
                    distribution["q75_bp"],
                    color="#777777",
                    alpha=0.12,
                    label="control IQR",
                )
                axis.axvline(
                    distribution["median_bp"],
                    color="#333333",
                    linestyle="--",
                    linewidth=1.3,
                    label="control median",
                )
                axis.legend(frameon=False, fontsize=8)
        elif unique_profiles:
            for index, profile_key in enumerate(profile_keys):
                distribution = (
                    controls.get(profile_key, {})
                    .get("footprint_sizes", {})
                    .get(control_key)
                )
                if not distribution:
                    continue
                median = float(distribution["median_bp"])
                axis.errorbar(
                    median,
                    index,
                    xerr=[
                        [median - float(distribution["q25_bp"])],
                        [float(distribution["q75_bp"]) - median],
                    ],
                    fmt="x",
                    color="#222222",
                    capsize=2,
                    markersize=6,
                )
        axis.set_yticks(y, display_labels)
        axis.invert_yaxis()
        axis.set(xlabel=xlabel, title=title)
        if not found:
            axis.text(0.5, 0.5, "footprint tags unavailable", ha="center", va="center", transform=axis.transAxes)

    footprint_panel(
        axes[2, 0],
        "nucleosome",
        "nucleosome",
        "Nucleosome-tagged protected-span comparison",
        "median nucleosome-tagged span (bp) with IQR",
    )
    footprint_panel(
        axes[2, 1],
        "tf_footprint",
        "tf",
        "TF footprint-size comparison",
        "median TF footprint size (bp) with IQR",
    )

    dedup_axis = axes[3, 0]
    for index, (result, color) in enumerate(zip(results, colors)):
        fraction = result["deduplication"].get("duplicate_fraction")
        if fraction is not None:
            dedup_axis.plot(100 * fraction, index, "o", color=color, markersize=6)
            dedup_axis.annotate(
                "full run" if result["deduplication"].get("full_run") else "sample",
                (100 * fraction, index),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.5,
                color="#555555",
            )
    dedup_axis.set_yticks(y, display_labels)
    dedup_axis.invert_yaxis()
    dedup_axis.set(
        xlabel="duplicate reads among fingerprintable reads (%)",
        title="PCR duplication comparison",
    )

    snp_axis = axes[3, 1]
    for index, (result, color) in enumerate(zip(results, colors)):
        masking = result["variant_masking"]
        n_sites = int(masking.get("n_masked_sites", 0))
        n_profiled = int(masking.get("n_profiled_sites", 0))
        if n_profiled:
            percentage = 100 * n_sites / n_profiled
            snp_axis.plot(percentage, index, "o", color=color, markersize=6)
            snp_axis.annotate(
                f"{n_sites:,} calls / {n_profiled:,} sites",
                (percentage, index),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.5,
                color="#555555",
            )
    snp_axis.set_yticks(y, display_labels)
    snp_axis.invert_yaxis()
    snp_axis.set(
        xlabel="called recurrent SNPs / profiled C/G positions (%)",
        title="Opposite-direction SNP comparison",
    )

    for axis in axes.flat:
        axis.grid(color="#ececec", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    status, score = _combined_status(results)
    counts = {item: sum(result["overall"]["status"] == item for result in results) for item in ("PASS", "WARN", "FAIL", "INSUFFICIENT")}
    figure.suptitle(
        f"FiberHMM multi-sample QC — {status} ({_fmt_score(score)}/100) | "
        f"PASS {counts['PASS']}  WARN {counts['WARN']}  FAIL {counts['FAIL']}  INSUFFICIENT {counts['INSUFFICIENT']}",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    figure.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": "FiberHMM multi-sample QC",
            "Creator": "FiberHMM",
            "Subject": "Multi-sample FiberHMM quality-control report",
        },
    )
    plt.close(figure)
    return True


def _write_combined_tsv(results: Sequence[dict], path: Path) -> None:
    fields = (
        "sample",
        "input",
        "mode",
        "enzyme",
        "reference_profile",
        "sampled_reads",
        "overall_status",
        "overall_score",
        "signal_status",
        "median_signal_rate",
        "periodicity_status",
        "periodicity_score",
        "nrl_bp",
        "periodicity_ac",
        "periodicity_reference_correlation",
        "periodicity_reference_pattern_amplitude_ratio",
        "n_nucleosomes",
        "median_nucleosome_bp",
        "fraction_nucleosome_85_250_bp",
        "fraction_nucleosome_over_300_bp",
        "fraction_nucleosome_over_1000_bp",
        "n_tf_footprints",
        "median_tf_footprint_bp",
        "deduplication_detected",
        "duplicate_fraction",
        "snp_mask_applied",
        "n_masked_snp_sites",
        "n_discovered_amplicons",
        *STATE_TSV_FIELDS,
    )
    with path.open("w") as handle:
        handle.write("\t".join(fields) + "\n")
        for result in results:
            values = {
                "sample": Path(result["input"]).name,
                "input": result["input"],
                "mode": result["assay"]["mode"],
                "enzyme": result["assay"]["enzyme"],
                "reference_profile": result["assay"]["reference_profile"],
                "sampled_reads": result["sampling"]["sampled_reads"],
                "overall_status": result["overall"]["status"],
                "overall_score": result["overall"]["score"],
                "signal_status": result["signal"]["status"],
                "median_signal_rate": result["signal"]["median_per_read_rate"],
                "periodicity_status": result["periodicity"]["status"],
                "periodicity_score": result["periodicity"]["score"],
                "nrl_bp": result["periodicity"]["nrl_bp"],
                "periodicity_ac": result["periodicity"]["autocorrelation_strength"],
                "periodicity_reference_correlation": result["periodicity"].get(
                    "reference_curve_correlation"
                ),
                "periodicity_reference_pattern_amplitude_ratio": result[
                    "periodicity"
                ].get("reference_pattern_amplitude_ratio"),
                "n_nucleosomes": result["footprints"]["n_nucleosomes"],
                "median_nucleosome_bp": result["footprints"]["median_nucleosome_bp"],
                "fraction_nucleosome_85_250_bp": result["footprints"].get(
                    "fraction_nucleosome_85_250_bp"
                ),
                "fraction_nucleosome_over_300_bp": result["footprints"].get(
                    "fraction_nucleosome_over_300_bp"
                ),
                "fraction_nucleosome_over_1000_bp": result["footprints"].get(
                    "fraction_nucleosome_over_1000_bp"
                ),
                "n_tf_footprints": result["footprints"]["n_tf_footprints"],
                "median_tf_footprint_bp": result["footprints"]["median_tf_footprint_bp"],
                "deduplication_detected": result["deduplication"]["detected"],
                "duplicate_fraction": result["deduplication"]["duplicate_fraction"],
                "snp_mask_applied": result["variant_masking"]["applied"],
                "n_masked_snp_sites": result["variant_masking"]["n_masked_sites"],
                "n_discovered_amplicons": result["variant_masking"].get(
                    "n_discovered_amplicons", 0
                ),
                **_state_tsv_fields(result),
            }
            handle.write("\t".join("" if values[field] is None else str(values[field]) for field in fields) + "\n")


def _write_combined_html(payload: dict, path: Path) -> None:
    rows = []
    panels = []
    for result in payload["samples"]:
        sample = Path(result["input"]).name
        if ".fiberhmm" in sample:
            sample = sample.split(".fiberhmm", 1)[0]
        signal_rate = result["signal"]["median_per_read_rate"]
        curve_correlation = result["periodicity"].get(
            "reference_curve_correlation"
        )
        amplitude_ratio = result["periodicity"].get(
            "reference_pattern_amplitude_ratio"
        )
        duplicate_fraction = result["deduplication"].get("duplicate_fraction")
        duplicate_text = (
            "NA" if duplicate_fraction is None else f"{100 * duplicate_fraction:.2f}%"
        )
        long_nuc_fraction = result["footprints"].get(
            "fraction_nucleosome_over_300_bp"
        )
        plot = result["outputs"].get("plot")
        state_fields = _state_tsv_fields(result)
        rows.append(
            "<tr>"
            f"<td>{html.escape(sample)}</td>"
            f"<td>{html.escape(result['overall']['status'])}</td>"
            f"<td>{_fmt_score(result['overall']['score'])}</td>"
            f"<td>{'NA' if signal_rate is None else f'{100 * signal_rate:.2f}%'}</td>"
            f"<td>{_pct(state_fields['median_msp_rate'])} "
            f"{html.escape(str(state_fields['efficiency_status'] or ''))}</td>"
            f"<td>{_pct(state_fields['median_outside_msp_rate'])} "
            f"{html.escape(str(state_fields['background_status'] or ''))}</td>"
            f"<td>{_pct(state_fields['msp_length_fraction'], 1)}</td>"
            f"<td>{result['periodicity']['nrl_bp'] or 'NA'}</td>"
            f"<td>{result['periodicity']['autocorrelation_strength']:.3f}</td>"
            f"<td>{'NA' if curve_correlation is None else f'{curve_correlation:.3f}'}</td>"
            f"<td>{'NA' if amplitude_ratio is None else f'{amplitude_ratio:.2f}x'}</td>"
            f"<td>{duplicate_text}</td>"
            f"<td>{'NA' if long_nuc_fraction is None else f'{100 * long_nuc_fraction:.1f}%'}</td>"
            f"<td>{result['variant_masking']['n_masked_sites']}</td>"
            f"<td>{result['variant_masking'].get('n_discovered_amplicons', 0)}</td>"
            "</tr>"
        )
        if plot:
            panels.append(
                f"<h2>{html.escape(sample)}</h2>"
                f"<img src=\"{html.escape(Path(plot).name)}\" alt=\"{html.escape(sample)} QC\">"
            )
    overview = Path(payload["outputs"]["plot"]).name if payload["outputs"].get("plot") else ""
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>FiberHMM multi-sample QC</title>
<style>body{{font:15px system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse}}th,td{{padding:.45rem .7rem;border:1px solid #ccc;text-align:right}}th:first-child,td:first-child{{text-align:left}}</style>
</head><body><h1>FiberHMM multi-sample QC — {html.escape(payload['overall']['status'])}</h1>
<p>Bounded assay-aware QC. Standalone QC does not perform deduplication; matching integrated-call sidecars provide exact full-run duplication and SNP summaries when available.</p>
{f'<img src="{html.escape(overview)}" alt="combined QC overview">' if overview else ''}
<h2>Scorecard</h2><table><thead><tr><th>Sample</th><th>Status</th><th>Score</th><th>Signal rate</th><th>In-MSP rate</th><th>Outside-MSP rate</th><th>MSP length</th><th>NRL (bp)</th><th>Periodicity AC</th><th>Reference curve r</th><th>Matched amplitude</th><th>Duplicate reads</th><th>Nuc-tagged &gt;300 bp</th><th>SNP calls</th><th>Amplicons</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
{''.join(panels)}</body></html>"""
    path.write_text(document)


def run_multi_qc(
    input_paths: Sequence[str],
    output_dir: str,
    mode: str = "auto",
    enzyme: str = "auto",
    reference_profile: str = "auto",
    reference_fasta: Optional[str] = None,
    sample_reads: int = DEFAULT_SAMPLE_READS,
    seed: int = DEFAULT_SEED,
    min_mapq: int = 20,
    prob_threshold: Optional[int] = None,
    min_opportunities: int = 200,
    stream: Optional[TextIO] = sys.stderr,
    state_source: str = "auto",
    min_msp_bp: int = DEFAULT_MIN_MSP_BP,
    light_call_reads: int = DEFAULT_LIGHT_CALL_READS,
    light_call_seconds: float = DEFAULT_LIGHT_CALL_SECONDS,
) -> dict:
    """Run per-BAM QC once, then build aggregate comparison artifacts."""
    if not input_paths:
        raise ValueError("run_multi_qc requires at least one input BAM/CRAM")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    used_names: set[str] = set()
    for input_path in input_paths:
        stem = Path(input_path).name
        for suffix in (".bam", ".cram"):
            if stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
        base = stem
        counter = 2
        while stem in used_names:
            stem = f"{base}_{counter}"
            counter += 1
        used_names.add(stem)
        result = run_qc(
            input_path=input_path,
            output_prefix=str(directory / stem),
            mode=mode,
            enzyme=enzyme,
            reference_profile=reference_profile,
            reference_fasta=reference_fasta,
            sample_reads=sample_reads,
            seed=seed,
            min_mapq=min_mapq,
            prob_threshold=prob_threshold,
            min_opportunities=min_opportunities,
            stream=stream,
            state_source=state_source,
            min_msp_bp=min_msp_bp,
            light_call_reads=light_call_reads,
            light_call_seconds=light_call_seconds,
        )
        results.append(result)

    status, score = _combined_status(results)
    json_path = directory / "combined.qc.json"
    tsv_path = directory / "combined.qc.tsv"
    plot_path = directory / "combined.qc.png"
    pdf_path = directory / "combined.qc.pdf"
    html_path = directory / "combined.qc.html"
    payload = {
        "schema_version": 1,
        "schema_minor_version": QC_SCHEMA_MINOR_VERSION,
        "report_type": "fiberhmm_multi_sample_qc",
        "overall": {"status": status, "score": score},
        "n_samples": len(results),
        "samples": results,
        "outputs": {
            "json": str(json_path.resolve()),
            "tsv": str(tsv_path.resolve()),
            "plot": None,
            "pdf": None,
            "html": str(html_path.resolve()),
        },
    }
    if _plot_combined_qc(results, plot_path, pdf_path):
        payload["outputs"]["plot"] = str(plot_path.resolve())
        payload["outputs"]["pdf"] = str(pdf_path.resolve())
    _write_combined_tsv(results, tsv_path)
    _write_combined_html(payload, html_path)
    _atomic_json(payload, json_path)
    if stream is not None:
        print(
            f"\nFiberHMM multi-sample QC: {status} "
            f"score={_fmt_score(score)}/100 ({len(results)} samples)\n"
            f"  overview: {payload['outputs']['plot'] or 'not written'}\n"
            f"  vector PDF: {payload['outputs']['pdf'] or 'not written'}\n"
            f"  report:   {payload['outputs']['html']}",
            file=stream,
        )
    return payload


__all__ = [
    "DEFAULT_SAMPLE_READS",
    "DEFAULT_SEED",
    "analyze_sample",
    "format_terminal",
    "infer_assay",
    "load_control_curves",
    "load_references",
    "pair_distance_histogram",
    "phasogram_metrics",
    "reference_profile_for_assay",
    "run_qc",
    "run_multi_qc",
    "sample_bam_reads",
]
