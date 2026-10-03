"""DAF evidence inside insertions from a local, deamination-aware consensus.

A DAF deamination is a read-versus-reference mismatch, so bases a read has
no reference counterpart for (a CIGAR insertion) carry no evidence and are
masked (``engine.daf_unaligned_query_positions``). When many reads carry the
same insertion (an amplicon: hundreds to thousands), the inserted sequence can
be inferred from the carriers themselves and used as a local reference:

* On C->T (CT) reads a C column reads C in some molecules and T in the
  deaminated ones; a true T always reads T. On G->A (GA) reads the same holds
  for G/A. The variability of a column, not its majority, decides: an open C
  deaminated in 70% of molecules is still a C.
* The strands cross-check each other: CT reads show A and G unconverted, GA
  reads show C and T unconverted.

This is the logic of the DAF SNP screen (``fiberhmm.daf.snps``: a site whose
base converts in one strand class while the other class reads it unconverted
is a deaminating target, not a variant), applied to columns of an insert.

Each column is called by a profile likelihood: for every candidate base (or a
gap) the deamination rate of that column on each strand is fitted (capped at
:data:`MAX_DEAMINATION`, so an all-T column on CT reads is a T), sequencing
errors are uniform, and the column's quality is the Phred-scaled posterior of
the best candidate. Carriers are aligned to a star consensus (the median-length
insert, polished over rounds) with a banded global alignment whose match
function accepts a deamination on the read's strand (T read over a consensus
C on CT reads, A over G on GA reads).

Each carrier's inserted bases are then re-encoded against the consensus:
on a CT read, a base aligned to a confident consensus C is a deamination (T)
or an unconverted target (C); a non-C read base over a non-C column is a
non-target whatever the column's exact base; everything else (bases over
unconfident C columns, a read C over a non-C column, unaligned read bases)
stays masked. A carrier aligning below 85% identity gets no evidence, and a
column whose observations one base cannot explain (two alleles) is never
confident. The result is per-read
evidence that the calling engine merges with the read's own (see
``engine.configure_daf_insert_evidence``).

Scope: CIGAR insertions of at least ``min_length`` bp on mapped, primary or
supplementary records. Soft-clipped arms are not grouped here: with
supplementary calling (3.0 default) a clipped arm that aligns elsewhere (a TE
copy) already gets real evidence against that copy; unaligned arms stay
masked.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from numba import njit as _njit
except ImportError:  # pragma: no cover - numba is a core dependency
    def _njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn

# Base codes: A C G T, N (anything else), and the gap symbol for counting.
A, C, G, T, N, GAP = 0, 1, 2, 3, 4, 4
_ENC = np.full(256, N, dtype=np.uint8)
for _i, _b in enumerate("ACGT"):
    _ENC[ord(_b)] = _i
    _ENC[ord(_b.lower())] = _i
# R/Y input: Y is a deaminated C (reads as T), R a deaminated G (reads as A).
_ENC[ord("Y")] = T
_ENC[ord("y")] = T
_ENC[ord("R")] = A
_ENC[ord("r")] = A
_DEC = "ACGTN"

STRAND_CT, STRAND_GA = 0, 1

MIN_INSERT = 50            # bp: as the no-call blocks
BREAKPOINT_TOLERANCE = 30  # bp between carriers' insertion positions
LENGTH_TOLERANCE = 0.25    # relative insert-length difference within a cluster
MIN_CARRIERS = 20          # carriers needed to build a consensus
MIN_QUALITY = 20           # Phred quality for a column to be used as reference
SEQ_ERROR = 0.04           # per-base sequencing error (uniform over 4 others)
MAX_DEAMINATION = 0.95     # cap on a column's fitted deamination rate
POLISH_ROUNDS = 3
MAX_POLISH_READS = 200     # carriers used to build (all are re-encoded)
MIN_CARRIER_IDENTITY = 0.85
MAX_DISCORDANT = 0.15      # share of bases a column's call cannot explain  # alignment identity a carrier needs to be re-encoded

OP_PAIR, OP_READ, OP_CONS = 0, 1, 2  # read base vs consensus base, read only, consensus only
_NEG = -(1 << 28)


def encode(seq: str) -> np.ndarray:
    return _ENC[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]


@_njit(cache=True, nogil=True)
def _banded_align(read, cons, w, strand):  # pragma: no cover - compiled
    """Global alignment of ``read`` to ``cons`` in a band of half-width ``w``
    around the scaled diagonal. A deamination on ``strand`` (0 CT: read T over
    C; 1 GA: read A over G) scores as a match. Returns forward ops."""
    n = read.shape[0]
    m = cons.shape[0]
    width = 2 * w + 1
    score = np.full((n + 1, width), _NEG, dtype=np.int32)
    trace = np.zeros((n + 1, width), dtype=np.uint8)
    centers = np.empty(n + 1, dtype=np.int64)
    for i in range(n + 1):
        centers[i] = (i * m) // n if n > 0 else 0
    MATCH, MISMATCH, GAPC = 2, -3, -4
    for i in range(n + 1):
        lo = centers[i] - w
        for k in range(width):
            j = lo + k
            if j < 0 or j > m:
                continue
            if i == 0 and j == 0:
                score[0, k] = 0
                continue
            best = _NEG
            op = 0
            if i > 0 and j > 0:
                kk = (j - 1) - (centers[i - 1] - w)
                if 0 <= kk < width and score[i - 1, kk] > _NEG:
                    r = read[i - 1]
                    c = cons[j - 1]
                    same = r == c or (strand == 0 and c == 1 and r == 3) \
                        or (strand == 1 and c == 2 and r == 0)
                    s = score[i - 1, kk] + (MATCH if same and r < 4 else MISMATCH)
                    if s > best:
                        best = s
                        op = 0
            if i > 0:
                kk = j - (centers[i - 1] - w)
                if 0 <= kk < width and score[i - 1, kk] > _NEG:
                    s = score[i - 1, kk] + GAPC
                    if s > best:
                        best = s
                        op = 1
            if j > 0 and k > 0 and score[i, k - 1] > _NEG:
                s = score[i, k - 1] + GAPC
                if s > best:
                    best = s
                    op = 2
            score[i, k] = best
            trace[i, k] = op
    # traceback
    ops = np.empty(n + m, dtype=np.uint8)
    t = 0
    i = n
    j = m
    while i > 0 or j > 0:
        k = j - (centers[i] - w)
        if k < 0 or k >= width:
            # outside the band (only when w is too small): fall back to gaps
            if i > 0:
                ops[t] = 1
                i -= 1
            else:
                ops[t] = 2
                j -= 1
            t += 1
            continue
        if i == 0:
            op = 2
        elif j == 0:
            op = 1
        else:
            op = trace[i, k]
        ops[t] = op
        t += 1
        if op == 0:
            i -= 1
            j -= 1
        elif op == 1:
            i -= 1
        else:
            j -= 1
    return ops[:t][::-1].copy()


def align(read: np.ndarray, cons: np.ndarray, strand: int) -> np.ndarray:
    w = 48 + abs(len(read) - len(cons)) + max(len(read), len(cons)) // 40
    return _banded_align(read, cons, int(w), int(strand))


# ---------------------------------------------------------------------------
# Column calling
# ---------------------------------------------------------------------------

_D_GRID = np.linspace(0.0, MAX_DEAMINATION, 20)


def _strand_loglik(counts: np.ndarray, base: int, strand: int,
                   error: float = SEQ_ERROR) -> float:
    """max over the deamination rate of log P(counts | true base, strand).

    ``counts`` holds A, C, G, T and gap observations of one strand class.
    """
    if counts.sum() == 0:
        return 0.0
    off = error / 4.0
    convertible = (base == C and strand == STRAND_CT) or (base == G and strand == STRAND_GA)
    product = T if base == C else A
    if not convertible:
        p = np.full(5, off)
        p[base] = 1.0 - error
        return float((counts * np.log(p)).sum())
    best = -math.inf
    for d in _D_GRID:
        p = np.full(5, off)
        # effective base: true base (1 - d) or its deamination product (d)
        p[base] = (1.0 - d) * (1.0 - error) + d * off
        p[product] = d * (1.0 - error) + (1.0 - d) * off
        value = float((counts * np.log(p)).sum())
        if value > best:
            best = value
    return best


def call_column(ct_counts: np.ndarray, ga_counts: np.ndarray,
                error: float = SEQ_ERROR) -> Tuple[int, float]:
    """(called symbol 0-3 or GAP, Phred quality) for one consensus column."""
    logl = np.empty(5)
    for b in range(5):
        logl[b] = (_strand_loglik(ct_counts, b, STRAND_CT, error)
                   + _strand_loglik(ga_counts, b, STRAND_GA, error))
    order = np.argsort(logl)[::-1]
    best = int(order[0])
    # posterior of the best candidate, flat prior
    rel = np.exp(logl - logl[best])
    p_err = 1.0 - 1.0 / rel.sum()
    q = 60.0 if p_err <= 1e-6 else min(60.0, -10.0 * math.log10(p_err))
    # Two alleles (or misassigned carriers) in one column: the candidates
    # compare one-base hypotheses only, so check the observations fit the
    # call. Bases the call cannot produce on a strand (anything but the base,
    # or its deamination product where it converts) beyond sequencing error
    # mean the column is not one base: no confidence.
    if best != GAP:
        for strand, counts in ((STRAND_CT, ct_counts), (STRAND_GA, ga_counts)):
            total = float(counts[:4].sum())
            if total < 4:
                continue
            allowed = counts[best]
            if (best == C and strand == STRAND_CT) or (best == G and strand == STRAND_GA):
                allowed += counts[T if best == C else A]
            if 1.0 - allowed / total > max(MAX_DISCORDANT, 3 * error):
                q = 0.0
    return best, q


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------

@dataclass
class Carrier:
    key: tuple            # (query_name, flag, reference_id, reference_start)
    query_start: int      # insert start in SEQ
    seq: str              # inserted bases as stored (R/Y allowed)
    strand: int           # STRAND_CT or STRAND_GA
    duplicate: bool = False  # PCR duplicate (0x400): re-encoded, not counted


@dataclass
class InsertConsensus:
    sequence: str
    quality: np.ndarray            # Phred per column
    depth_ct: np.ndarray
    depth_ga: np.ndarray
    n_carriers: int


def _evenly(items: list, limit: int) -> list:
    if len(items) <= limit:
        return list(items)
    step = len(items) / float(limit)
    return [items[int(i * step)] for i in range(limit)]


def _pileup(backbone: np.ndarray, carriers: Sequence[Carrier]):
    m = len(backbone)
    counts = np.zeros((2, m, 5), dtype=np.int64)
    inserts: List[Counter] = [Counter() for _ in range(m + 1)]
    for carrier in carriers:
        read = encode(carrier.seq)
        ops = align(read, backbone, carrier.strand)
        i = j = 0
        pending: List[int] = []
        for op in ops:
            if op == OP_PAIR:
                if pending:
                    inserts[j][tuple(pending)] += 1
                    pending = []
                counts[carrier.strand, j, read[i]] += 1
                i += 1
                j += 1
            elif op == OP_READ:
                pending.append(int(read[i]))
                i += 1
            else:
                if pending:
                    inserts[j][tuple(pending)] += 1
                    pending = []
                counts[carrier.strand, j, GAP] += 1
                j += 1
        if pending:
            inserts[j][tuple(pending)] += 1
    return counts, inserts


def build_consensus(carriers: Sequence[Carrier], rounds: int = POLISH_ROUNDS,
                    max_reads: int = MAX_POLISH_READS,
                    error: float = SEQ_ERROR) -> Optional[InsertConsensus]:
    if not carriers:
        return None
    lengths = sorted(len(c.seq) for c in carriers)
    median = lengths[len(lengths) // 2]
    seed = min(carriers, key=lambda c: abs(len(c.seq) - median))
    # A seed from a GA read has its G->A deaminations; any seed is only a
    # starting point, the rounds re-call every column.
    backbone = encode(seed.seq)
    used = _evenly(sorted(carriers, key=lambda c: c.key), max_reads)
    quality = depth_ct = depth_ga = None
    for _round in range(rounds):
        counts, inserts = _pileup(backbone, used)
        n_used = len(used)
        new: List[int] = []
        quals: List[float] = []
        dct: List[int] = []
        dga: List[int] = []
        for j in range(len(backbone) + 1):
            votes = inserts[j]
            if votes:
                total = sum(votes.values())
                if total * 2 > n_used:
                    # most carriers carry extra bases here: keep the commonest
                    run, _n = votes.most_common(1)[0]
                    new.extend(run)
                    quals.extend([0.0] * len(run))
                    dct.extend([0] * len(run))
                    dga.extend([0] * len(run))
            if j == len(backbone):
                break
            base, q = call_column(counts[STRAND_CT, j], counts[STRAND_GA, j], error)
            if base == GAP:
                continue
            new.append(base)
            quals.append(q)
            dct.append(int(counts[STRAND_CT, j, :4].sum()))
            dga.append(int(counts[STRAND_GA, j, :4].sum()))
        backbone = np.asarray(new, dtype=np.uint8)
        quality = np.asarray(quals)
        depth_ct = np.asarray(dct)
        depth_ga = np.asarray(dga)
    # Final qualities: one more pileup against the polished consensus.
    counts, _inserts = _pileup(backbone, used)
    quals = []
    for j in range(len(backbone)):
        base, q = call_column(counts[STRAND_CT, j], counts[STRAND_GA, j], error)
        quals.append(q if base == backbone[j] else 0.0)
    quality = np.asarray(quals)
    depth_ct = counts[STRAND_CT, :, :4].sum(axis=1)
    depth_ga = counts[STRAND_GA, :, :4].sum(axis=1)
    return InsertConsensus(
        sequence="".join(_DEC[b] for b in backbone),
        quality=quality, depth_ct=depth_ct, depth_ga=depth_ga,
        n_carriers=len(carriers),
    )


@dataclass
class ReadInsertEvidence:
    mods: set = field(default_factory=set)    # SEQ positions: deaminated targets
    known: set = field(default_factory=set)   # SEQ positions with evidence (mods incl.)
    strand: int = STRAND_CT                   # the read's deamination strand


def reencode_carrier(carrier: Carrier, consensus: InsertConsensus,
                     min_quality: float = MIN_QUALITY) -> ReadInsertEvidence:
    """Deaminations and known bases of one carrier's insert against the
    consensus (SEQ positions of the read)."""
    read = encode(carrier.seq)
    cons = encode(consensus.sequence)
    ops = align(read, cons, carrier.strand)
    target, product = (C, T) if carrier.strand == STRAND_CT else (G, A)
    out = ReadInsertEvidence(strand=carrier.strand)
    # A carrier that does not match the consensus (another allele at the same
    # breakpoint, a misassigned cluster member) gets no evidence.
    pairs = same = 0
    i = j = 0
    for op in ops:
        if op == OP_PAIR:
            pairs += 1
            r, c = int(read[i]), int(cons[j])
            same += r == c or (c == target and r == product)
            i += 1
            j += 1
        elif op == OP_READ:
            i += 1
        else:
            j += 1
    if len(ops) == 0 or same < MIN_CARRIER_IDENTITY * len(ops):
        return out
    i = j = 0
    for op in ops:
        if op == OP_PAIR:
            r = int(read[i])
            c = int(cons[j])
            pos = carrier.query_start + i
            if c == target:
                # A target column is evidence only when it is confident: the
                # read's product is a deamination, its target base is not.
                if consensus.quality[j] >= min_quality:
                    if r == product:
                        out.mods.add(pos)
                        out.known.add(pos)
                    elif r == target:
                        out.known.add(pos)
            elif r != target:
                # Any other column is a non-target for this read whatever its
                # exact base; only a read target base over a non-target
                # column (an error, or a column called wrong) stays unknown.
                out.known.add(pos)
            i += 1
            j += 1
        elif op == OP_READ:
            i += 1
        else:
            j += 1
    return out


# ---------------------------------------------------------------------------
# Carriers from a BAM
# ---------------------------------------------------------------------------

def _read_strand(read, ref_fasta=None) -> Optional[int]:
    seq = read.query_sequence or ""
    if read.has_tag("st"):
        st = str(read.get_tag("st")).upper()
        if st in ("CT", "GA"):
            return STRAND_CT if st == "CT" else STRAND_GA
    if "Y" in seq or "R" in seq:
        return STRAND_CT if seq.count("Y") >= seq.count("R") else STRAND_GA
    if read.has_tag("MM") or read.has_tag("Mm"):
        # MM/ML-native deamination calls: T marks are C->T, A marks G->A.
        from fiberhmm.core.bam_reader import (
            detect_daf_strand,
            parse_mm_tag_query_positions,
        )
        try:
            mm = read.get_tag("MM") if read.has_tag("MM") else read.get_tag("Mm")
            ml = read.get_tag("ML") if read.has_tag("ML") else read.get_tag("Ml")
            marks = parse_mm_tag_query_positions(mm, ml, seq, read.is_reverse,
                                                 mode="daf")
        except Exception:
            marks = set()
        strand = detect_daf_strand(seq, marks)
        if strand in ("+", "-"):
            return STRAND_CT if strand == "+" else STRAND_GA
    from fiberhmm.daf.encoder import get_daf_positions
    try:
        res = get_daf_positions(read, ref_fasta=ref_fasta)
    except Exception:
        return None
    if res is None:
        return None
    return STRAND_CT if res[2] == "CT" else STRAND_GA


def record_key(read) -> tuple:
    return (read.query_name, int(read.flag) & ~0x400, int(read.reference_id),
            int(read.reference_start))


def inserted_query_positions(cigartuples) -> set:
    """SEQ positions of every CIGAR insertion (any length)."""
    out: set = set()
    q = 0
    for op, length in cigartuples or ():
        if op == 1:
            out.update(range(q, q + length))
        if op in (0, 1, 4, 7, 8):
            q += length
    return out


def insertion_events(read, min_length: int = MIN_INSERT):
    """(reference position, query start, length) of each CIGAR insertion."""
    out = []
    q = 0
    r = read.reference_start
    for op, length in read.cigartuples or ():
        if op == 1:
            if length >= min_length:
                out.append((r, q, length))
            q += length
        elif op in (0, 7, 8):
            q += length
            r += length
        elif op == 4:
            q += length
        elif op in (2, 3):
            r += length
    return out


def collect_carriers(reads: Iterable, min_mapq: int = 0, min_length: int = MIN_INSERT,
                     ref_fasta=None):
    """Insertion events of eligible records: ``[(contig_id, ref_pos, Carrier)]``."""
    events = []
    for read in reads:
        if read.is_unmapped or read.is_secondary or read.mapping_quality < min_mapq:
            continue
        if not read.cigartuples or not any(op == 1 and n >= min_length
                                           for op, n in read.cigartuples):
            continue
        strand = _read_strand(read, ref_fasta)
        if strand is None:
            continue
        seq = read.query_sequence
        key = record_key(read)
        for ref_pos, q, length in insertion_events(read, min_length):
            events.append((int(read.reference_id), int(ref_pos),
                           Carrier(key, q, seq[q:q + length], strand,
                                   bool(read.is_duplicate))))
    return events


def cluster_events(events, tolerance: int = BREAKPOINT_TOLERANCE,
                   length_tolerance: float = LENGTH_TOLERANCE):
    """Group insertion events by contig, breakpoint and length."""
    clusters: List[list] = []
    by_contig: Dict[int, list] = defaultdict(list)
    for contig, pos, carrier in events:
        by_contig[contig].append((pos, carrier))
    for contig, items in by_contig.items():
        items.sort(key=lambda x: (x[0], len(x[1].seq)))
        open_clusters: List[dict] = []
        for pos, carrier in items:
            placed = False
            for cl in open_clusters:
                if abs(pos - cl["pos"]) <= tolerance and \
                        abs(len(carrier.seq) - cl["length"]) <= length_tolerance * cl["length"]:
                    cl["members"].append(carrier)
                    placed = True
                    break
            if not placed:
                open_clusters.append({"contig": contig, "pos": pos,
                                      "length": len(carrier.seq), "members": [carrier]})
            # retire clusters left far behind
            still = []
            for cl in open_clusters:
                if pos - cl["pos"] > tolerance:
                    clusters.append(cl)
                else:
                    still.append(cl)
            open_clusters = still
        clusters.extend(open_clusters)
    return clusters


def build_insert_evidence(reads: Iterable, *, min_carriers: int = MIN_CARRIERS,
                          min_quality: float = MIN_QUALITY, min_mapq: int = 0,
                          min_length: int = MIN_INSERT, ref_fasta=None,
                          error: float = SEQ_ERROR):
    """Per-record insert evidence and a cluster report.

    Returns ``(evidence, report)``: ``evidence`` maps :func:`record_key` to a
    :class:`ReadInsertEvidence`; ``report`` lists every cluster (position,
    length, carriers per strand, consensus length and confident fraction,
    and whether it was used).
    """
    events = collect_carriers(reads, min_mapq, min_length, ref_fasta)
    evidence: Dict[tuple, ReadInsertEvidence] = {}
    report = []
    for cl in cluster_events(events):
        members = cl["members"]
        # PCR duplicates are copies of one molecule: they neither count as
        # carriers nor vote in the consensus, but are re-encoded against it.
        independent = [c for c in members if not c.duplicate]
        n_ct = sum(1 for c in independent if c.strand == STRAND_CT)
        entry = {"contig_id": cl["contig"], "pos": cl["pos"], "length": cl["length"],
                 "carriers": len(independent), "carriers_ct": n_ct,
                 "carriers_ga": len(independent) - n_ct,
                 "duplicate_records": len(members) - len(independent), "used": False}
        if len(independent) >= min_carriers:
            cons = build_consensus(independent, error=error)
            if cons is not None and len(cons.sequence):
                confident = float(np.mean(cons.quality >= min_quality))
                entry.update(consensus=cons.sequence,
                             consensus_length=len(cons.sequence),
                             confident_fraction=confident, used=True)
                for carrier in members:
                    ev = reencode_carrier(carrier, cons, min_quality)
                    have = evidence.get(carrier.key)
                    if have is None:
                        evidence[carrier.key] = ev
                    else:
                        have.mods |= ev.mods
                        have.known |= ev.known
        report.append(entry)
    return evidence, report


def run_insert_consensus_prepass(bam_path: str, evidence_dir: str, *,
                                 min_carriers: int = MIN_CARRIERS,
                                 min_mapq: int = 0,
                                 min_quality: float = MIN_QUALITY,
                                 reference: Optional[str] = None,
                                 report_path: Optional[str] = None) -> dict:
    """Calling pre-pass: build insert evidence for ``bam_path``.

    Writes ``<evidence_dir>/insert_evidence.pkl`` (per-record evidence for
    ``engine.configure_daf_insert_evidence``) when any cluster had enough
    carriers, and the cluster report to ``report_path`` when given (callers
    publish it with their outputs: :func:`write_report`). Returns the summary.
    """
    import os
    import pickle

    import pysam

    fasta = pysam.FastaFile(reference) if reference else None
    try:
        with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
            names = list(bam.references)
            evidence, report = build_insert_evidence(
                bam.fetch(until_eof=True), min_carriers=min_carriers,
                min_quality=min_quality, min_mapq=min_mapq, ref_fasta=fasta)
    finally:
        if fasta is not None:
            fasta.close()
    for entry in report:
        cid = entry.pop("contig_id")
        entry["contig"] = names[cid] if 0 <= cid < len(names) else None
    evidence_path = None
    if evidence:
        os.makedirs(evidence_dir, exist_ok=True)
        evidence_path = os.path.join(evidence_dir, "insert_evidence.pkl")
        with open(evidence_path, "wb") as handle:
            pickle.dump(evidence, handle, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "schema": "fiberhmm.daf.insert_consensus.v1",
        "input": os.path.abspath(bam_path),
        "min_carriers": int(min_carriers),
        "min_quality": float(min_quality),
        "min_length": MIN_INSERT,
        "breakpoint_tolerance": BREAKPOINT_TOLERANCE,
        "clusters": len(report),
        "clusters_used": sum(1 for e in report if e["used"]),
        "records_with_evidence": len(evidence),
        "evidence_path": evidence_path,
        "report_path": report_path,
        "insertions": report,
    }
    if report_path:
        write_report(summary, report_path)
    return summary


def write_report(summary: dict, report_path: str) -> str:
    """Publish the cluster report (JSON) atomically."""
    import json
    import os

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    payload = {k: v for k, v in summary.items() if k != "evidence_path"}
    payload["report_path"] = report_path
    tmp = report_path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=1)
    os.replace(tmp, report_path)
    return report_path
