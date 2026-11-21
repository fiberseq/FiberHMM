"""Variant discovery for ``fiberhmm-extract --variants discover``.

The ``var`` layer needs a VCF. ``discover`` makes one from the reads, so a
single ``fiberhmm-extract`` run gives the var (and sv) bigBeds:

* DAF-seq (DddA/DddB): the DAF SNP screen
  (:func:`fiberhmm.daf.snps.call_opposite_conversion_snps`, ``all_snvs``),
  which requires support on fibers of both conversion directions, so a C->T
  or G->A deamination is never called as a SNV.
* Fiber-seq (Hia5 / PacBio / Nanopore): a conservative pileup SNV screen
  (:func:`pileup_chunk`). It counts high-quality bases of well-mapped reads
  and keeps sites with enough depth, recurrent alternate reads and an
  alternate fraction; strand bias, homopolymer context (Nanopore), nearby
  deletions and m6A-associated mismatches become VCF ``FILTER`` values. It
  calls SNVs only (no indels, no MNVs) and estimates GT from the alternate
  fraction. It is a screen for the browser, not a genotyper: use DeepVariant
  or Clair3 for publication-grade genotypes and pass their VCF instead.

Discovery runs once per pooled sample (the input BAM plus any pool BAMs,
optionally restricted to read groups), in chunks (``--cores`` in parallel;
finished chunks are kept, so an interrupted run resumes). The result is a
bgzipped, tabix-indexed VCF in the cache directory whose name carries a key
over everything that decides it (input identities, regions, caller,
parameters, FiberHMM version); a later run with the same key reuses it.
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
import re
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pysam

from fiberhmm import defaults as D

CACHE_SCHEMA = 1
VCF_SUFFIX = ".fhvar.vcf.gz"
CALLERS = ("auto", "daf", "pileup")
HOMOPOLYMER_MODES = ("auto", "on", "off")

# Reference context read past each chunk end (homopolymer runs, ref votes).
_MARGIN = 32

_BASE_INDEX = np.full(256, 4, dtype=np.int8)
for _i, _b in enumerate("ACGT"):
    _BASE_INDEX[ord(_b)] = _i
    _BASE_INDEX[ord(_b.lower())] = _i
_BASES = "ACGT"

# CIGAR op codes
_M, _I, _D, _N, _S, _H, _P, _EQ, _X = 0, 1, 2, 3, 4, 5, 6, 7, 8


class VariantDiscoveryError(RuntimeError):
    """Discovery cannot run on this input (message says how to fix it)."""


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PileupParams:
    """Thresholds of the pileup SNV screen (defaults: fiberhmm.defaults)."""
    min_depth: int = D.PILEUP_MIN_DEPTH
    min_alt_reads: int = D.PILEUP_MIN_ALT_READS
    min_alt_fraction: float = D.PILEUP_MIN_ALT_FRACTION
    min_baseq: int = D.PILEUP_MIN_BASEQ
    min_mapq: int = D.PILEUP_MIN_MAPQ
    strand_bias_p: float = D.PILEUP_STRAND_BIAS_P
    homopolymer: str = D.PILEUP_HOMOPOLYMER
    homopolymer_len: int = D.PILEUP_HOMOPOLYMER_LEN
    max_indel_fraction: float = D.PILEUP_MAX_INDEL_FRACTION
    mod_filter: bool = True
    mod_min_methylation: float = D.PILEUP_MOD_MIN_METHYLATION
    mod_max_alt_fraction: float = D.PILEUP_MOD_MAX_ALT_FRACTION
    mod_prob_threshold: int = D.REREAD_PROB_THRESHOLD
    hom_alt_fraction: float = D.PILEUP_HOM_ALT_FRACTION

    def validate(self) -> None:
        if self.min_depth < 1 or self.min_alt_reads < 1:
            raise ValueError("pileup depth / alternate-read minima must be >= 1")
        for name in ("min_alt_fraction", "mod_min_methylation",
                     "mod_max_alt_fraction", "hom_alt_fraction",
                     "max_indel_fraction"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0 <= self.strand_bias_p <= 1:
            raise ValueError("strand_bias_p must be in [0, 1]")
        if self.homopolymer not in HOMOPOLYMER_MODES:
            raise ValueError(f"homopolymer must be one of {HOMOPOLYMER_MODES}")
        if self.homopolymer_len < 2:
            raise ValueError("homopolymer_len must be >= 2")


@dataclass(frozen=True)
class DafScreenParams:
    """Thresholds passed to the DAF SNP screen (its validated defaults)."""
    min_fraction: float = D.DAF_SNP_MIN_FRACTION
    min_depth: int = D.DAF_SNP_MIN_DEPTH
    min_alt_fibers: int = D.DAF_SNP_MIN_ALT_FIBERS
    min_mapq: int = D.MIN_MAPQ


@dataclass(frozen=True)
class DiscoveryOptions:
    """Settings of one discovery run (picklable).

    ``pool``: further BAMs of the same sample, pooled with the input BAM
    (ignored with ``per_bam``). ``read_groups`` / ``rg_sample``: only reads
    of these read-group IDs / of read groups whose ``SM`` is ``rg_sample``.
    ``cache_dir``: where the VCF is kept (None: the caller's output folder).
    ``reference``: indexed FASTA; needed for the pileup caller when reads
    carry neither MD tags nor ``=``/``X`` CIGARs.
    """
    caller: str = D.VARIANT_CALLER
    pool: Tuple[str, ...] = ()
    per_bam: bool = False
    read_groups: Tuple[str, ...] = ()
    rg_sample: Optional[str] = None
    sample_name: Optional[str] = None
    cache_dir: Optional[str] = None
    reference: Optional[str] = None
    chunk_bp: int = D.VARIANT_CHUNK_BP
    cores: int = 1
    pileup: PileupParams = field(default_factory=PileupParams)
    daf: DafScreenParams = field(default_factory=DafScreenParams)

    def validate(self) -> None:
        if self.caller not in CALLERS:
            raise ValueError(f"caller must be one of {CALLERS}")
        if self.chunk_bp < 1000:
            raise ValueError("chunk_bp must be >= 1000")
        self.pileup.validate()


@dataclass(frozen=True)
class DiscoveryResult:
    vcf: str               # bgzipped + tabix-indexed VCF in the cache
    key: str               # full sha256 cache key
    caller: str            # 'daf' | 'pileup'
    caller_reason: str
    reused: bool
    sidecar: str           # JSON next to the VCF (key material + summary)
    n_records: int
    n_pass: int


# ---------------------------------------------------------------------------
# inputs, caller choice
# ---------------------------------------------------------------------------

def pooled_inputs(input_bam: str, options: DiscoveryOptions) -> List[str]:
    """The BAMs discovery reads: the input plus the pool (deduplicated)."""
    paths = [input_bam] if options.per_bam else [input_bam, *options.pool]
    out: List[str] = []
    seen = set()
    for p in paths:
        real = os.path.realpath(p)
        if real not in seen:
            seen.add(real)
            out.append(p)
    return out


def check_pool_headers(bams: Sequence[str]) -> List[Tuple[str, int]]:
    """``[(contig, length)]`` shared by every pooled BAM; raises when the
    BAMs are on different references (pooling needs one coordinate system)."""
    first = None
    for path in bams:
        if not os.path.isfile(path):
            raise VariantDiscoveryError(f"pool BAM not found: {path}")
        if os.path.getsize(path) == 0:
            raise VariantDiscoveryError(
                f"pool BAM is empty (0 bytes; an online-only placeholder?): {path}")
        with pysam.AlignmentFile(path, "rb", check_sq=False) as bam:
            contigs = list(zip(bam.references, bam.lengths))
        if first is None:
            first = contigs
        elif contigs != first:
            raise VariantDiscoveryError(
                f"{os.path.basename(path)} has different @SQ contigs from "
                f"{os.path.basename(bams[0])}; pooled BAMs must share a reference")
    return list(first or [])


def resolve_read_groups(bams: Sequence[str], options: DiscoveryOptions
                        ) -> Optional[Tuple[frozenset, ...]]:
    """Per BAM (in ``bams`` order), the read-group IDs discovery keeps; None
    when every read is kept. RG IDs are local to a BAM, so ``rg_sample`` is
    resolved against each BAM's own @RG lines."""
    if not options.read_groups and options.rg_sample is None:
        return None
    wanted = set(options.read_groups)
    out = []
    for path in bams:
        with pysam.AlignmentFile(path, "rb", check_sq=False) as bam:
            rgs = bam.header.to_dict().get("RG", []) or []
        ids = {str(rg.get("ID")) for rg in rgs}
        if options.rg_sample is not None:
            ids = {str(rg.get("ID")) for rg in rgs if str(rg.get("SM")) == options.rg_sample}
        if wanted:
            ids &= wanted
        out.append(frozenset(ids))
    if not any(out):
        what = []
        if options.rg_sample is not None:
            what.append(f"SM:{options.rg_sample}")
        if wanted:
            what.append("ID " + ",".join(sorted(wanted)))
        raise VariantDiscoveryError(
            f"no @RG with {' and '.join(what)} in the pooled BAMs")
    return tuple(out)


def _rg_of(read_groups, k: int) -> Optional[frozenset]:
    return None if read_groups is None else read_groups[k]


def _sniff_daf_reads(path: str, n: int = 200) -> bool:
    """DAF evidence in the reads: R/Y IUPAC bases, an ``st`` tag or an MM dU code."""
    seen = 0
    with pysam.AlignmentFile(path, "rb", check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            seq = read.query_sequence or ""
            if "R" in seq or "Y" in seq or read.has_tag("st"):
                return True
            if read.has_tag("MM"):
                mm = str(read.get_tag("MM"))
                if re.search(r"[CG][+-]u|55797", mm):
                    return True
            seen += 1
            if seen >= n:
                break
    return False


def detect_caller(bams: Sequence[str], requested: str = "auto") -> Tuple[str, str]:
    """``(caller, reason)``: ``daf`` for DAF data (header or reads), else ``pileup``."""
    if requested in ("daf", "pileup"):
        return requested, f"--variant-caller {requested}"
    from fiberhmm.io.read_layers import bam_looks_daf
    for path in bams:
        if bam_looks_daf(path):
            return "daf", f"{os.path.basename(path)} header declares DAF"
    for path in bams:
        if _sniff_daf_reads(path):
            return "daf", f"{os.path.basename(path)} reads carry DAF marks (R/Y, st or dU)"
    return "pileup", "no DAF evidence (Fiber-seq): pileup SNV screen"


def detect_platform(bam_path: str) -> str:
    """``'nanopore'``, ``'pacbio'`` or ``'unknown'`` from the header."""
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        header = bam.header
        hd = header.to_dict()
    try:
        from fiberhmm.models import declared_prob_threshold_chemistry
        _enzyme, platform = declared_prob_threshold_chemistry(header)
    except Exception:
        platform = None
    if platform:
        p = platform.lower()
        if "nanopore" in p or p == "ont":
            return "nanopore"
        if "pacbio" in p:
            return "pacbio"
    text = json.dumps({k: hd.get(k) for k in ("RG", "PG")}).lower()
    if re.search(r'"pl": "(ont|nanopore)"|dorado|guppy|minknow', text):
        return "nanopore"
    if re.search(r'"pl": "pacbio"|readtype=ccs|pbmm2|"pn": "ccs"|jasmine', text):
        return "pacbio"
    return "unknown"


def _circular_lengths(bam_path: str) -> Dict[str, int]:
    """``{contig: length}`` of the header's ``TP:circular`` contigs."""
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        sqs = bam.header.to_dict().get("SQ", []) or []
    return {sq["SN"]: int(sq["LN"]) for sq in sqs
            if str(sq.get("TP", "")).lower() == "circular"}


def homopolymer_active(params: PileupParams, platform: str) -> bool:
    if params.homopolymer == "on":
        return True
    if params.homopolymer == "off":
        return False
    return platform != "pacbio"


# ---------------------------------------------------------------------------
# regions
# ---------------------------------------------------------------------------

def merge_windows(windows: Iterable[Tuple[str, int, int]],
                  contig_order: Sequence[str]) -> List[Tuple[str, int, int]]:
    """Sort (in header order) and merge overlapping / touching windows."""
    order = {c: i for i, c in enumerate(contig_order)}
    items = sorted(((c, int(s), int(e)) for c, s, e in windows if int(e) > int(s)),
                   key=lambda w: (order.get(w[0], 1 << 30), w[1], w[2]))
    out: List[List] = []
    for c, s, e in items:
        if out and out[-1][0] == c and s <= out[-1][2]:
            out[-1][2] = max(out[-1][2], e)
        else:
            out.append([c, s, e])
    return [tuple(w) for w in out]


def discovery_chunks(windows: Sequence[Tuple[str, int, int]], chunk_bp: int) -> List[Tuple[str, int, int]]:
    chunks = []
    for c, s, e in windows:
        for a in range(s, e, chunk_bp):
            chunks.append((c, a, min(e, a + chunk_bp)))
    return chunks


# ---------------------------------------------------------------------------
# cache key
# ---------------------------------------------------------------------------

def _file_identity(path: str, header_md5: bool = True) -> dict:
    st = os.stat(path)
    ident = {"name": os.path.basename(path), "size": int(st.st_size),
             "mtime_ns": int(st.st_mtime_ns)}
    if header_md5:
        with pysam.AlignmentFile(path, "rb", check_sq=False) as bam:
            text = str(bam.header)
        ident["header_md5"] = hashlib.md5(text.encode()).hexdigest()
    return ident


def cache_material(bams: Sequence[str], windows: Sequence[Tuple[str, int, int]],
                   caller: str, options: DiscoveryOptions,
                   read_groups, platform: str) -> dict:
    """Everything that decides the discovered VCF (hashed into the key)."""
    from fiberhmm import __version__
    params = (dataclasses.asdict(options.pileup) if caller == "pileup"
              else dataclasses.asdict(options.daf))
    if caller == "pileup":
        params["homopolymer_active"] = homopolymer_active(options.pileup, platform)
    reference = None
    if options.reference:
        reference = _file_identity(options.reference, header_md5=False)
    return {
        "schema": CACHE_SCHEMA,
        "fiberhmm_version": __version__,
        "caller": caller,
        "parameters": params,
        # Each input with the read groups kept from it (RG IDs are per BAM).
        "inputs": sorted(({**_file_identity(p),
                           "read_groups": (sorted(read_groups[k]) if read_groups is not None
                                           else None)}
                          for k, p in enumerate(bams)),
                         key=lambda d: json.dumps(d, sort_keys=True)),
        "regions": [list(w) for w in windows],
        "reference": reference,
        "chunk_bp": int(options.chunk_bp),
    }


def cache_key(material: dict) -> str:
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def _safe_token(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_+-]+", "_", str(name)).strip("_") or "sample"


def cache_paths(cache_dir: str, sample: str, key: str) -> Tuple[str, str, str]:
    """``(vcf, sidecar json, parts dir)`` for a sample and key."""
    stem = f"{_safe_token(sample)}.{key[:12]}"
    vcf = os.path.join(cache_dir, stem + VCF_SUFFIX)
    return vcf, vcf[:-len(".vcf.gz")] + ".json", os.path.join(cache_dir, f".{stem}.fhvar.parts")


def usable_cached_vcf(vcf: str, sidecar: str, key: str) -> bool:
    """A cached VCF is reused only when its sidecar has the key and the VCF
    and its index both open."""
    try:
        with open(sidecar) as handle:
            meta = json.load(handle)
        if meta.get("cache_key") != key:
            return False
        if not os.path.exists(vcf + ".tbi"):
            return False
        with pysam.VariantFile(vcf) as handle:
            if handle.index is None:
                return False
            for _ in handle:
                break
        return True
    except (OSError, ValueError, KeyError):
        return False


# ---------------------------------------------------------------------------
# pileup caller
# ---------------------------------------------------------------------------

def _aligned_arrays(read):
    """``(qpos, rpos, is_eq, del_rpos)`` for the M/=/X bases and D bases."""
    cig = read.cigartuples
    if not cig:
        return None
    table = np.asarray(cig, dtype=np.int64)
    ops, lens = table[:, 0], table[:, 1]
    q_adv = np.isin(ops, (_M, _I, _S, _EQ, _X))
    r_adv = np.isin(ops, (_M, _D, _N, _EQ, _X))
    q_start = np.cumsum(np.where(q_adv, lens, 0)) - np.where(q_adv, lens, 0)
    r_start = (np.cumsum(np.where(r_adv, lens, 0)) - np.where(r_adv, lens, 0)
               + int(read.reference_start))
    aligned = np.isin(ops, (_M, _EQ, _X))
    al = lens[aligned]
    total = int(al.sum())
    if total == 0:
        return None
    offs = np.arange(total) - np.repeat(np.cumsum(al) - al, al)
    qpos = np.repeat(q_start[aligned], al) + offs
    rpos = np.repeat(r_start[aligned], al) + offs
    is_eq = np.repeat(ops[aligned] == _EQ, al)
    dl = lens[ops == _D]
    if dl.size:
        doffs = np.arange(int(dl.sum())) - np.repeat(np.cumsum(dl) - dl, dl)
        del_rpos = np.repeat(r_start[ops == _D], dl) + doffs
    else:
        del_rpos = np.zeros(0, dtype=np.int64)
    return qpos, rpos, is_eq, del_rpos


def _keep_read(read, min_mapq: int, read_groups: Optional[frozenset]) -> bool:
    if (read.is_unmapped or read.is_secondary or read.is_supplementary
            or read.is_duplicate or read.mapping_quality < min_mapq):
        return False
    if read.is_qcfail:  # QC-fail, chimera marks included
        return False
    if read_groups is not None:
        try:
            rg = str(read.get_tag("RG"))
        except KeyError:
            return False
        if rg not in read_groups:
            return False
    return True


def _fisher_p(a: int, b: int, c: int, d: int) -> float:
    if min(a + b, c + d, a + c, b + d) == 0:
        return 1.0
    from scipy.stats import fisher_exact
    return float(fisher_exact([[a, b], [c, d]])[1])


def _run_length_at(ref: np.ndarray, i: int) -> int:
    """Length of the homopolymer run of ``ref`` containing index ``i`` (4 = N)."""
    b = ref[i]
    if b > 3:
        return 0
    lo = i
    while lo > 0 and ref[lo - 1] == b:
        lo -= 1
    hi = i
    while hi + 1 < len(ref) and ref[hi + 1] == b:
        hi += 1
    return hi - lo + 1


def _homopolymer_context(ref: np.ndarray, i: int) -> int:
    """Longest run at or next to index ``i``."""
    best = 0
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(ref):
            best = max(best, _run_length_at(ref, j))
    return best


def _m6a_query_positions(read, threshold: int) -> set:
    """Query positions with an m6A call at or above ``threshold`` (A+a / T-a)."""
    out = set()
    try:
        mods = read.modified_bases
    except Exception:
        return out
    if not mods:
        return out
    for (base, _strand, code), calls in mods.items():
        if code not in ("a", 21839) or base not in ("A", "T"):
            continue
        for qpos, prob in calls:
            if prob >= threshold:
                out.add(int(qpos))
    return out


def _fold(positions: np.ndarray, length: int):
    """Positions folded onto a circular contig, first occurrence of each
    (a read that runs round the origin counts a position once); returns
    ``(folded, index into the input)``."""
    folded = positions % length
    _u, first = np.unique(folded, return_index=True)
    first.sort()
    return folded[first], first


def pileup_chunk(chrom: str, lo: int, hi: int, bams: Sequence[str],
                 params: PileupParams, read_groups=None,
                 reference: Optional[str] = None,
                 homopolymer: bool = False,
                 circular_length: Optional[int] = None) -> Tuple[List[dict], dict]:
    """Candidate SNVs at ``[lo, hi)`` of ``chrom`` from every read that
    overlaps the chunk; ``(records, stats)``.

    ``read_groups``: per-BAM RG sets (:func:`resolve_read_groups`) or None.
    On a circular contig of ``circular_length`` every read of the contig is
    read and its coordinates are folded onto the contig, each position
    counted once per read."""
    w_lo = max(0, lo - _MARGIN)
    w_hi = hi + _MARGIN
    L = w_hi - w_lo
    counts = np.zeros(L * 8, dtype=np.int64)     # pos, base(4), strand(2)
    dels = np.zeros(L, dtype=np.int64)
    votes = np.zeros(L * 4, dtype=np.int64)      # reference base votes
    stats = {"reads": 0, "reads_no_bases": 0, "reads_no_quals": 0}
    batch_c: List[np.ndarray] = []
    batch_v: List[np.ndarray] = []
    batch_n = 0
    from fiberhmm.daf.aligned_arrays import matched_base_arrays

    def flush():
        nonlocal batch_n
        if batch_c:
            counts[:] += np.bincount(np.concatenate(batch_c), minlength=L * 8)
            batch_c.clear()
        if batch_v:
            votes[:] += np.bincount(np.concatenate(batch_v), minlength=L * 4)
            batch_v.clear()
        batch_n = 0

    for k, path in enumerate(bams):
        rg = _rg_of(read_groups, k)
        with pysam.AlignmentFile(path, "rb") as bam:
            try:
                reads = (bam.fetch(chrom) if circular_length
                         else bam.fetch(chrom, lo, hi))
            except ValueError:
                continue
            for read in reads:
                if not _keep_read(read, params.min_mapq, rg):
                    continue
                seq = read.query_sequence
                quals = read.query_qualities
                arrays = _aligned_arrays(read) if seq else None
                if arrays is None:
                    stats["reads_no_bases"] += 1
                    continue
                has_quals = quals is not None and len(quals) == len(seq)
                if not has_quals and params.min_baseq > 0:
                    # No qualities: no base can pass a quality floor.
                    stats["reads_no_quals"] += 1
                    continue
                stats["reads"] += 1
                qpos, rpos, is_eq, del_rpos = arrays
                if circular_length:
                    rpos, keep = _fold(rpos, circular_length)
                    qpos, is_eq = qpos[keep], is_eq[keep]
                    del_rpos = _fold(del_rpos, circular_length)[0]
                inside = (rpos >= w_lo) & (rpos < w_hi)
                qpos, rpos, is_eq = qpos[inside], rpos[inside], is_eq[inside]
                codes = _BASE_INDEX[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)[qpos]]
                if reference is None:
                    if read.has_tag("MD"):
                        m = matched_base_arrays(read, seq)
                        if m is not None:
                            _mq, mr, mref, _mquery = m
                            if circular_length:
                                mr, keep = _fold(mr, circular_length)
                                mref = mref[keep]
                            sel = (mr >= w_lo) & (mr < w_hi)
                            rc = _BASE_INDEX[mref[sel]]
                            ok = rc < 4
                            batch_v.append((mr[sel][ok] - w_lo) * 4 + rc[ok])
                    elif is_eq.any():
                        eqc = codes[is_eq]
                        ok = eqc < 4
                        batch_v.append((rpos[is_eq][ok] - w_lo) * 4 + eqc[ok])
                good = codes < 4
                if has_quals:
                    good &= np.asarray(quals, dtype=np.int64)[qpos] >= params.min_baseq
                strand = 1 if read.is_reverse else 0
                batch_c.append((rpos[good] - w_lo) * 8 + codes[good].astype(np.int64) * 2 + strand)
                d = del_rpos[(del_rpos >= w_lo) & (del_rpos < w_hi)]
                if d.size:
                    np.add.at(dels, d - w_lo, 1)
                batch_n += 1
                if batch_n >= 256:
                    flush()
    flush()

    cube = counts.reshape(L, 4, 2)
    if reference is not None:
        with pysam.FastaFile(reference) as fasta:
            try:
                clen = fasta.get_reference_length(chrom)
            except KeyError:
                raise VariantDiscoveryError(
                    f"contig {chrom} is not in the reference {reference}")
            text = fasta.fetch(chrom, w_lo, min(w_hi, clen)).upper()
        ref = np.full(L, 4, dtype=np.int8)
        ref[:len(text)] = _BASE_INDEX[np.frombuffer(text.encode(), dtype=np.uint8)]
    else:
        v = votes.reshape(L, 4)
        ref = np.where(v.sum(axis=1) > 0, v.argmax(axis=1), 4).astype(np.int8)
    depth_all = cube.sum(axis=(1, 2))
    stats["depth_sites"] = int(np.count_nonzero(depth_all[lo - w_lo:hi - w_lo] >= params.min_depth))
    unknown = (ref == 4) & (depth_all >= params.min_depth)
    stats["ref_unknown_sites"] = int(np.count_nonzero(unknown[lo - w_lo:hi - w_lo]))

    records: List[dict] = []
    per_base = cube.sum(axis=2)                       # L x 4
    idx = np.arange(lo - w_lo, hi - w_lo)
    idx = idx[(depth_all[idx] >= params.min_depth) & (ref[idx] < 4)]
    if idx.size:
        alt_counts = per_base[idx].copy()
        alt_counts[np.arange(idx.size), ref[idx]] = -1
        best = alt_counts.argmax(axis=1)
        ac = alt_counts[np.arange(idx.size), best]
        dp = depth_all[idx]
        keep = (ac >= params.min_alt_reads) & (ac >= params.min_alt_fraction * dp)
        for i, alt, n_alt, n_dp in zip(idx[keep].tolist(), best[keep].tolist(),
                                       ac[keep].tolist(), dp[keep].tolist()):
            r = int(ref[i])
            ref_f, ref_r = int(cube[i, r, 0]), int(cube[i, r, 1])
            alt_f, alt_r = int(cube[i, alt, 0]), int(cube[i, alt, 1])
            af = n_alt / n_dp
            filters = []
            sbp = _fisher_p(ref_f, ref_r, alt_f, alt_r)
            if sbp < params.strand_bias_p:
                filters.append("StrandBias")
            hp = _homopolymer_context(ref, i)
            if homopolymer and hp >= params.homopolymer_len:
                filters.append("Homopolymer")
            del_frac = int(dels[i]) / (n_dp + int(dels[i]))
            if del_frac >= params.max_indel_fraction:
                filters.append("IndelNear")
            records.append({
                "chrom": chrom, "pos0": w_lo + i, "ref": _BASES[r], "alt": _BASES[alt],
                "filters": filters,
                "info": {"DP": n_dp, "AC": n_alt, "AF": round(af, 6),
                         "SB": f"{ref_f},{ref_r},{alt_f},{alt_r}",
                         "SBP": float(f"{sbp:.4g}"), "HPL": hp,
                         "DF": round(del_frac, 4)},
                "gt": "1/1" if af >= params.hom_alt_fraction else "0/1",
                "ad": (ref_f + ref_r, n_alt), "dp": n_dp,
            })
    if params.mod_filter and records:
        _mod_pass(records, chrom, bams, params, read_groups, circular_length)
    stats["candidates"] = len(records)
    return records, stats


def _mod_pass(records: List[dict], chrom: str, bams: Sequence[str],
              params: PileupParams, read_groups=None,
              circular_length: Optional[int] = None) -> None:
    """Methylation fraction of REF reads at A/T candidates; FILTER
    ModArtifact where it is high and the alternate fraction low. Uses the
    same read and base-quality rules as the counting pass."""
    targets = {rec["pos0"]: rec for rec in records if rec["ref"] in "AT"}
    if not targets:
        return
    lo, hi = min(targets), max(targets) + 1
    ref_reads = dict.fromkeys(targets, 0)
    called = dict.fromkeys(targets, 0)
    any_mm = False
    positions = np.array(sorted(targets), dtype=np.int64)
    for k, path in enumerate(bams):
        rg = _rg_of(read_groups, k)
        with pysam.AlignmentFile(path, "rb") as bam:
            try:
                reads = bam.fetch(chrom) if circular_length else bam.fetch(chrom, lo, hi)
            except ValueError:
                continue
            for read in reads:
                if not _keep_read(read, params.min_mapq, rg):
                    continue
                if not read.has_tag("MM") and not read.has_tag("Mm"):
                    continue
                seq = read.query_sequence
                arrays = _aligned_arrays(read) if seq else None
                if arrays is None:
                    continue
                quals = read.query_qualities
                has_quals = quals is not None and len(quals) == len(seq)
                if not has_quals and params.min_baseq > 0:
                    continue
                qpos, rpos = arrays[0], arrays[1]
                if circular_length:
                    rpos, keep = _fold(rpos, circular_length)
                    qpos = qpos[keep]
                hit = np.isin(rpos, positions)
                if has_quals:
                    hit &= np.asarray(quals, dtype=np.int64)[qpos] >= params.min_baseq
                if not hit.any():
                    continue
                any_mm = True
                mods = _m6a_query_positions(read, params.mod_prob_threshold)
                for q, p in zip(qpos[hit].tolist(), rpos[hit].tolist()):
                    if seq[q].upper() == targets[p]["ref"]:
                        ref_reads[p] += 1
                        if q in mods:
                            called[p] += 1
    if not any_mm:
        return
    for p, rec in targets.items():
        n = ref_reads[p]
        mr = called[p] / n if n else 0.0
        rec["info"]["MR"] = round(mr, 4)
        if (n and mr >= params.mod_min_methylation
                and rec["info"]["AF"] < params.mod_max_alt_fraction):
            rec["filters"].append("ModArtifact")


# ---------------------------------------------------------------------------
# DAF screen per chunk
# ---------------------------------------------------------------------------

def daf_chunk(chrom: str, lo: int, hi: int, bams: Sequence[str], params: DafScreenParams,
              read_groups, reference: Optional[str],
              work_dir: str, circular_length: Optional[int] = None) -> Tuple[List[dict], dict]:
    """DAF SNP screen (``all_snvs``) on the reads overlapping the chunk;
    calls at ``[lo, hi)`` only. On a circular contig every read of the
    contig is screened (the screen folds reads through the origin)."""
    from fiberhmm.daf.snps import call_opposite_conversion_snps
    os.makedirs(work_dir, exist_ok=True)
    fd, tmp_bam = tempfile.mkstemp(prefix=f"chunk_{_safe_token(chrom)}_{lo}_", suffix=".bam",
                                   dir=work_dir)
    os.close(fd)
    n = 0
    try:
        with pysam.AlignmentFile(bams[0], "rb") as first:
            header = first.header
        with pysam.AlignmentFile(tmp_bam, "wb", header=header) as out:
            for k, path in enumerate(bams):
                rg = _rg_of(read_groups, k)
                with pysam.AlignmentFile(path, "rb") as bam:
                    try:
                        reads = (bam.fetch(chrom) if circular_length
                                 else bam.fetch(chrom, lo, hi))
                    except ValueError:
                        continue
                    for read in reads:
                        if not _keep_read(read, params.min_mapq, rg):
                            continue
                        out.write(read)
                        n += 1
        if n == 0:
            return [], {"reads": 0, "candidates": 0}
        payload = call_opposite_conversion_snps(
            tmp_bam, min_fraction=params.min_fraction, min_depth=params.min_depth,
            min_alt_fibers=params.min_alt_fibers, min_mapq=params.min_mapq,
            reference_fasta=reference, max_profile_sites=0, all_snvs=True)
    finally:
        try:
            os.remove(tmp_bam)
        except OSError:
            pass
    records = []
    for call in payload.get("calls", []):
        p = int(call["position_0based"])
        if not lo <= p < hi:
            continue
        dp = int(call["opposite_direction_depth"])
        ac = int(call["alternate_fibers"])
        af = float(call["alternate_fraction"])
        records.append({
            "chrom": chrom, "pos0": p, "ref": call["reference"], "alt": call["alternate"],
            "filters": [],
            "info": {"DP": dp, "AC": ac, "AF": round(af, 6), "SCREEN": "conversion",
                     "ED": int(call["expected_direction_depth"]),
                     "EA": int(call["expected_direction_mismatch_fibers"]),
                     "OD": call["opposite_dominant_direction"]},
            "gt": "1/1" if af >= D.PILEUP_HOM_ALT_FRACTION else "0/1",
            "ad": (max(0, dp - ac), ac), "dp": dp,
        })
    for call in payload.get("other_snv_calls", []):
        p = int(call["position_0based"])
        if not lo <= p < hi:
            continue
        dp, ac, af = int(call["depth"]), int(call["alternate_fibers"]), float(call["alternate_fraction"])
        records.append({
            "chrom": chrom, "pos0": p, "ref": call["reference"], "alt": call["alternate"],
            "filters": [],
            "info": {"DP": dp, "AC": ac, "AF": round(af, 6), "SCREEN": "all_snvs",
                     "CTD": int(call["ct_direction_depth"]),
                     "CTA": int(call["ct_direction_alternate_fibers"]),
                     "GAD": int(call["ga_direction_depth"]),
                     "GAA": int(call["ga_direction_alternate_fibers"])},
            "gt": "1/1" if af >= D.PILEUP_HOM_ALT_FRACTION else "0/1",
            "ad": (max(0, dp - ac), ac), "dp": dp,
        })
    return records, {"reads": n, "candidates": len(records)}


# ---------------------------------------------------------------------------
# chunk worker, VCF writer
# ---------------------------------------------------------------------------

def _chunk_name(chunk: Tuple[str, int, int]) -> str:
    """Collision-free part name (contig names may differ only in characters
    a file name cannot keep)."""
    c, s, e = chunk
    digest = hashlib.sha1(f"{c}\t{s}\t{e}".encode()).hexdigest()[:16]
    return f"{_safe_token(c)[:40]}_{s}_{e}_{digest}"


def _part_path(parts_dir: str, chunk) -> str:
    return os.path.join(parts_dir, _chunk_name(chunk) + ".jsonl")


def _run_chunk(task) -> Tuple[Tuple[str, int, int], dict]:
    chunk, caller, bams, options, read_groups, homopolymer, parts_dir, circ = task
    chrom, lo, hi = chunk
    pysam.set_verbosity(0)
    if caller == "pileup":
        records, stats = pileup_chunk(chrom, lo, hi, bams, options.pileup, read_groups,
                                      options.reference, homopolymer, circ)
    else:
        records, stats = daf_chunk(chrom, lo, hi, bams, options.daf, read_groups,
                                   options.reference, parts_dir, circ)
    path = _part_path(parts_dir, chunk)
    fd, tmp = tempfile.mkstemp(prefix=".part_", suffix=".tmp", dir=parts_dir)
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps({"chunk": list(chunk), "stats": stats}) + "\n")
        for rec in records:
            handle.write(json.dumps(rec) + "\n")
    os.replace(tmp, path)
    return chunk, stats


def _read_part(path: str) -> Tuple[dict, List[dict]]:
    with open(path) as handle:
        head = json.loads(handle.readline())
        return head, [json.loads(line) for line in handle if line.strip()]


def _part_done(parts_dir: str, chunk) -> bool:
    """A finished part whose recorded chunk is this chunk."""
    try:
        with open(_part_path(parts_dir, chunk)) as handle:
            head = json.loads(handle.readline())
        return head.get("chunk") == list(chunk)
    except (OSError, ValueError):
        return False


_INFO_HEADERS = {
    "DP": '##INFO=<ID=DP,Number=1,Type=Integer,Description="Depth used for the call (pileup: high-quality bases; DAF conversion: opposite-direction fibers; DAF all_snvs: both directions)">',
    "AC": '##INFO=<ID=AC,Number=A,Type=Integer,Description="Reads/fibers carrying ALT">',
    "AF": '##INFO=<ID=AF,Number=A,Type=Float,Description="ALT fraction (AC/DP)">',
    "SB": '##INFO=<ID=SB,Number=4,Type=Integer,Description="Pileup: REF forward, REF reverse, ALT forward, ALT reverse reads">',
    "SBP": '##INFO=<ID=SBP,Number=1,Type=Float,Description="Pileup: two-sided Fisher p-value of strand balance">',
    "HPL": '##INFO=<ID=HPL,Number=1,Type=Integer,Description="Pileup: longest reference homopolymer at or next to the site">',
    "DF": '##INFO=<ID=DF,Number=1,Type=Float,Description="Pileup: fraction of reads deleting the site">',
    "MR": '##INFO=<ID=MR,Number=1,Type=Float,Description="Pileup: fraction of REF reads with an m6A call at the site (A/T sites, BAMs with MM/ML)">',
    "SCREEN": '##INFO=<ID=SCREEN,Number=1,Type=String,Description="DAF screen: conversion (C>T/G>A seen against both conversion directions) or all_snvs">',
    "ED": '##INFO=<ID=ED,Number=1,Type=Integer,Description="DAF: expected-direction fiber depth">',
    "EA": '##INFO=<ID=EA,Number=1,Type=Integer,Description="DAF: expected-direction mismatch fibers">',
    "OD": '##INFO=<ID=OD,Number=1,Type=String,Description="DAF: dominant direction on which the event is unexpected">',
    "CTD": '##INFO=<ID=CTD,Number=1,Type=Integer,Description="DAF: C->T-direction fiber depth">',
    "CTA": '##INFO=<ID=CTA,Number=1,Type=Integer,Description="DAF: C->T-direction ALT fibers">',
    "GAD": '##INFO=<ID=GAD,Number=1,Type=Integer,Description="DAF: G->A-direction fiber depth">',
    "GAA": '##INFO=<ID=GAA,Number=1,Type=Integer,Description="DAF: G->A-direction ALT fibers">',
}
_FILTER_HEADERS = [
    '##FILTER=<ID=PASS,Description="All filters passed">',
    '##FILTER=<ID=StrandBias,Description="Pileup: ALT strand balance differs from REF (Fisher p below the threshold)">',
    '##FILTER=<ID=Homopolymer,Description="Pileup: in or next to a reference homopolymer (Nanopore or unknown platform)">',
    '##FILTER=<ID=IndelNear,Description="Pileup: many reads delete the site">',
    '##FILTER=<ID=ModArtifact,Description="Pileup: A/T site whose REF reads are often m6A-called and whose ALT fraction is low (possible modification basecalling artefact)">',
]


def write_vcf(records: List[dict], path: str, contigs: Sequence[Tuple[str, int]],
              sample: str, caller: str, key: str, reason: str) -> int:
    """Write sorted records as a bgzipped VCF with a tabix index (atomic)."""
    from fiberhmm import __version__
    order = {c: i for i, (c, _l) in enumerate(contigs)}
    records = sorted(records, key=lambda r: (order.get(r["chrom"], 1 << 30), r["pos0"], r["alt"]))
    used_info = []
    for rec in records:
        for k in rec["info"]:
            if k not in used_info:
                used_info.append(k)
    lines = ["##fileformat=VCFv4.3",
             f"##source=FiberHMM-{__version__} fiberhmm-extract --variants discover",
             f"##fiberhmm_variant_caller={caller}",
             f"##fiberhmm_variant_caller_reason={reason}",
             f"##fiberhmm_cache_key={key}"]
    lines += [f"##contig=<ID={c},length={n}>" for c, n in contigs]
    lines += _FILTER_HEADERS
    lines += [_INFO_HEADERS[k] for k in _INFO_HEADERS if k in used_info]
    lines.append('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype estimated from AF (screen estimate, not a genotyper call)">')
    lines.append('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">')
    lines.append('##FORMAT=<ID=AD,Number=R,Type=Integer,Description="REF and ALT depth">')
    # Any VCF sample name but whitespace (column separators) is kept as given.
    lines.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                 + (re.sub(r"\s+", "_", str(sample)) or "sample"))
    body = []
    for rec in records:
        info = ";".join(f"{k}={v}" for k, v in rec["info"].items())
        flt = ";".join(rec["filters"]) or "PASS"
        ad = ",".join(str(x) for x in rec["ad"])
        body.append(f"{rec['chrom']}\t{rec['pos0'] + 1}\t.\t{rec['ref']}\t{rec['alt']}\t.\t"
                    f"{flt}\t{info}\tGT:DP:AD\t{rec['gt']}:{rec['dp']}:{ad}")
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, plain = tempfile.mkstemp(prefix=".fhvar_", suffix=".vcf", dir=directory)
    os.close(fd)
    gz = plain + ".gz"
    try:
        with open(plain, "w") as handle:
            handle.write("\n".join(lines + body) + "\n")
        pysam.tabix_compress(plain, gz, force=True)
        pysam.tabix_index(gz, preset="vcf", force=True)
        os.replace(gz + ".tbi", path + ".tbi")
        os.replace(gz, path)
    finally:
        for p in (plain, gz, gz + ".tbi"):
            if os.path.exists(p):
                os.remove(p)
    return len(records)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def discover_variants(input_bam: str, options: DiscoveryOptions,
                      windows: Optional[Sequence[Tuple[str, int, int]]] = None,
                      default_cache_dir: Optional[str] = None,
                      log: Callable[[str], None] = print) -> DiscoveryResult:
    """Discover SNVs for the pooled sample of ``input_bam`` (cached).

    ``windows``: ``(chrom, start, end)`` to discover in (merged); None = every
    contig of the BAM header. Returns the cached VCF and how it was made.
    """
    options.validate()
    bams = pooled_inputs(input_bam, options)
    contigs = check_pool_headers(bams)
    contig_names = [c for c, _ in contigs]
    if windows is None:
        windows = [(c, 0, n) for c, n in contigs]
    windows = merge_windows(windows, contig_names)
    if not windows:
        raise VariantDiscoveryError("no regions to discover variants in")
    read_groups = resolve_read_groups(bams, options)
    caller, reason = detect_caller(bams, options.caller)
    platform = detect_platform(bams[0])
    homopolymer = caller == "pileup" and homopolymer_active(options.pileup, platform)
    for path in bams:
        with pysam.AlignmentFile(path, "rb", check_sq=False) as bam:
            if not bam.has_index():
                raise VariantDiscoveryError(
                    f"variant discovery needs an indexed BAM: samtools index {path}")
    if options.reference and not os.path.exists(options.reference + ".fai"):
        raise VariantDiscoveryError(
            f"--reference needs a FASTA index: samtools faidx {options.reference}")

    material = cache_material(bams, windows, caller, options, read_groups, platform)
    key = cache_key(material)
    def stem(path):
        name = os.path.basename(path)
        return name[:-4] if name.endswith(".bam") else name
    if options.sample_name:
        sample = options.sample_name
    elif len(bams) > 1:
        # The same for every BAM of the pool, whichever one is -i.
        sample = min(stem(p) for p in bams) + "_pool"
    else:
        sample = stem(input_bam)
    cache_dir = options.cache_dir or default_cache_dir or os.path.dirname(os.path.abspath(input_bam))
    os.makedirs(cache_dir, exist_ok=True)
    vcf, sidecar, parts_dir = cache_paths(cache_dir, sample, key)

    def reuse():
        with open(sidecar) as handle:
            meta = json.load(handle)
        log(f"[variants] reusing cached VCF {vcf} (key {key[:12]}, caller={caller})")
        return DiscoveryResult(vcf, key, caller, reason, True, sidecar,
                               int(meta.get("n_records", 0)), int(meta.get("n_pass", 0)))

    if usable_cached_vcf(vcf, sidecar, key):
        return reuse()
    lock = parts_dir + ".lock"
    _acquire_lock(lock, log)
    try:
        # Another run with the same key may have finished while we waited.
        if usable_cached_vcf(vcf, sidecar, key):
            return reuse()
        return _build(input_bam, options, bams, contigs, contig_names, windows, read_groups,
                      caller, reason, platform, homopolymer, material, key, sample,
                      vcf, sidecar, parts_dir, log)
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _acquire_lock(lock: str, log, poll: float = 2.0) -> None:
    """One discovery per key at a time: wait for a live holder, take over a
    stale lock (holder process gone on this host)."""
    import socket
    import time
    host = socket.gethostname()
    warned = False
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                with open(lock) as handle:
                    holder_host, _, holder_pid = handle.read().strip().partition(":")
                stale = holder_host == host and not _pid_alive(int(holder_pid))
            except (OSError, ValueError):
                stale = False  # being written: wait
            if stale:
                log(f"[variants] removing stale lock {lock}")
                try:
                    os.remove(lock)
                except OSError:
                    pass
                continue
            if not warned:
                log(f"[variants] another run is discovering with this key ({lock}); waiting")
                warned = True
            time.sleep(poll)
            continue
        with os.fdopen(fd, "w") as handle:
            handle.write(f"{host}:{os.getpid()}")
        return


def _build(input_bam, options, bams, contigs, contig_names, windows, read_groups, caller,
           reason, platform, homopolymer, material, key, sample, vcf, sidecar, parts_dir,
           log) -> DiscoveryResult:
    log(f"[variants] building VCF (key {key[:12]}): caller={caller} ({reason}); "
        f"{len(bams)} BAM(s), {len(windows)} window(s)"
        + (f", homopolymer filter on (platform {platform})" if homopolymer else
           (f", platform {platform}" if caller == "pileup" else "")))
    os.makedirs(parts_dir, exist_ok=True)
    with open(os.path.join(parts_dir, "key.json"), "w") as handle:
        json.dump({"cache_key": key, "material": material}, handle, sort_keys=True)
    # Contigs without mapped reads in any pooled BAM have nothing to call.
    occupied = set()
    for path in bams:
        with pysam.AlignmentFile(path, "rb") as bam:
            try:
                occupied.update(s.contig for s in bam.get_index_statistics() if s.mapped)
            except (ValueError, AttributeError):
                occupied.update(contig_names)
    chunks = [c for c in discovery_chunks(windows, options.chunk_bp) if c[0] in occupied]
    step = max(1, len(chunks) // 10)
    todo = [chunk for chunk in chunks if not _part_done(parts_dir, chunk)]
    if len(todo) < len(chunks):
        log(f"[variants] resuming: {len(chunks) - len(todo)}/{len(chunks)} chunks already done")
    circular = _circular_lengths(bams[0])
    tasks = [(c, caller, tuple(bams), options, read_groups, homopolymer, parts_dir,
              circular.get(c[0])) for c in todo]
    done = len(chunks) - len(todo)
    if tasks:
        if options.cores > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=min(options.cores, len(tasks))) as pool:
                for _chunk, _stats in pool.map(_run_chunk, tasks):
                    done += 1
                    if done % step == 0 or done == len(chunks):
                        log(f"[variants] chunks {done}/{len(chunks)}")
        else:
            for task in tasks:
                _run_chunk(task)
                done += 1
                if done % step == 0 or done == len(chunks):
                    log(f"[variants] chunks {done}/{len(chunks)}")

    records: List[dict] = []
    totals: Dict[str, int] = {}
    for chunk in chunks:
        head, recs = _read_part(_part_path(parts_dir, chunk))
        if head.get("chunk") != list(chunk):
            raise VariantDiscoveryError(f"discovery part for {chunk} is not that chunk; "
                                        f"remove {parts_dir} and rerun")
        records.extend(recs)
        for k, v in (head.get("stats") or {}).items():
            totals[k] = totals.get(k, 0) + int(v)
    if caller == "pileup" and totals.get("reads", 0) and totals.get("depth_sites", 0) \
            and totals.get("ref_unknown_sites", 0) >= totals["depth_sites"]:
        raise VariantDiscoveryError(
            "the pileup caller cannot tell the reference base: the reads carry no MD "
            "tag and no =/X CIGAR; pass --reference FASTA (indexed)")
    if caller == "pileup" and not options.reference and totals.get("ref_unknown_sites", 0):
        log(f"[variants] WARNING: {totals['ref_unknown_sites']:,} covered positions have no "
            "REF evidence (no MD tag and no '=' read there: typically homozygous-ALT sites "
            "in =/X CIGAR BAMs); they are not called. Pass --reference FASTA to call them.")
    n = write_vcf(records, vcf, contigs, sample, caller, key, reason)
    n_pass = sum(1 for r in records if not r["filters"])
    meta = {
        "cache_key": key,
        "vcf": os.path.basename(vcf),
        "caller": caller,
        "caller_reason": reason,
        "platform": platform,
        "homopolymer_filter": homopolymer,
        "sample": sample,
        "n_records": n,
        "n_pass": n_pass,
        "stats": totals,
        "material": material,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    fd, tmp = tempfile.mkstemp(prefix=".fhvar_", suffix=".json.tmp",
                               dir=os.path.dirname(os.path.abspath(sidecar)))
    with os.fdopen(fd, "w") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, sidecar)
    shutil.rmtree(parts_dir, ignore_errors=True)
    log(f"[variants] wrote {vcf}: {n:,} records, {n_pass:,} PASS")
    return DiscoveryResult(vcf, key, caller, reason, False, sidecar, n, n_pass)


def place_beside(vcf: str, out_dir: str) -> str:
    """The VCF (and its index) in ``out_dir`` under the same name, copied
    when the cache is elsewhere, so a viewer finds it next to the var bigBed."""
    target = os.path.join(out_dir, os.path.basename(vcf))
    if os.path.abspath(target) == os.path.abspath(vcf):
        return vcf
    # Data first, index second (copy2 keeps mtimes: the index stays newer).
    for src, dst in ((vcf, target), (vcf + ".tbi", target + ".tbi")):
        fd, tmp = tempfile.mkstemp(prefix=".fhvar_", suffix=".tmp", dir=out_dir)
        os.close(fd)
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    return target


__all__ = [
    "DafScreenParams", "DiscoveryOptions", "DiscoveryResult", "PileupParams",
    "VariantDiscoveryError", "cache_key", "cache_material", "detect_caller",
    "detect_platform", "discover_variants", "merge_windows", "pileup_chunk",
    "place_beside", "write_vcf",
]
