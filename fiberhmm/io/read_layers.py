"""Per-read variant alleles and structural events for ``fiberhmm-extract``.

Two opt-in BED12 layers that let a project keep compact bigBeds instead of
the BAM for two questions the BAM is usually kept for:

``var`` (``--variants VCF``)
    For every read, the allele it carries at each overlapping site of a VCF
    or BCF (DeepVariant / Clair3 calls, ``vg deconstruct`` output, the DAF
    SNP screen VCF, ...). The read is genotyped pileup-style, as WhatsHap /
    longphase haplotag do: an SNV from the aligned base, an MNV or a small
    indel from the read's bases over the REF span (CIGAR insertions after
    each REF base included). No realignment: an indel the aligner placed
    differently from the VCF's normalised position (homopolymers, tandem
    repeats) is reported as ``-1``.

``sv`` (``--sv``)
    CIGAR deletions and insertions of at least ``--sv-min-len`` bp, soft or
    hard clips of at least ``--sv-min-clip`` bp at either end, and the read's
    ``SA`` tag: the per-read signatures long-read SV callers (Sniffles2,
    cuteSV) start from. Supplementary records are written as their own rows.

Both layers write one row per read (per alignment record for ``sv``) with
one block per site / event. bigBed blocks must be ascending and must not
overlap, so a read whose sites or events overlap (overlapping VCF records, an
insertion next to a deletion, a clip next to an indel) is written as more than
one row with the same name, each row holding non-overlapping blocks.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
from bisect import bisect_left
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pysam

# --- allele codes (``blockAllele``) -----------------------------------------
# k >= 0 is the VCF allele index (0 = REF, 1.. = ALT).
ALLELE_OTHER = -1          # the read carries a sequence matching no allele
ALLELE_DELETED = -2        # every REF base of the site is deleted in the read
ALLELE_DAF_AMBIGUOUS = -3  # several alleles fit once DAF conversion is allowed

# --- event types (``blockType``) --------------------------------------------
SV_DEL = 0
SV_INS = 1
SV_CLIP_LEFT = 2
SV_CLIP_RIGHT = 3

DEFAULT_MAX_ALLELE_LEN = 50
DEFAULT_SV_MIN_LEN = 30
DEFAULT_SV_MIN_CLIP = 100
SA_MAX_CHARS = 4000

_BASES = frozenset('ACGTN')
# CIGAR operation codes
_M, _I, _D, _N, _S, _H, _P, _EQ, _X = 0, 1, 2, 3, 4, 5, 6, 7, 8


# ---------------------------------------------------------------------------
# VCF handling
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VariantSite:
    """A VCF record reduced to what genotyping needs.

    ``alleles[k]`` is the uppercase sequence of allele ``k`` or ``None`` for
    an allele that is never matched (symbolic, ``*``, longer than the
    allowed length). ``indel`` is True when some usable allele changes the
    length, which makes the insertion after the last REF base part of the
    compared read sequence.
    """
    start: int
    end: int
    alleles: Tuple[Optional[str], ...]
    indel: bool


@dataclass(frozen=True)
class VariantOptions:
    """Picklable settings for the ``var`` workers."""
    path: str                      # indexed VCF/BCF the workers fetch from
    source_path: str               # the file the user gave (for messages)
    pass_only: bool = True
    sample: Optional[str] = None
    het_only: bool = False
    max_allele_len: int = DEFAULT_MAX_ALLELE_LEN
    daf: str = 'auto'              # auto | on | off
    bam_is_daf: bool = False       # BAM-level DAF evidence (header/sniff)


def _usable_allele(allele: Optional[str], max_len: int) -> Optional[str]:
    if allele is None:
        return None
    seq = str(allele).upper()
    if not seq or not set(seq) <= _BASES:
        return None  # symbolic <DEL>, breakends, '*', '.', IUPAC
    if len(seq) > max_len + 1:  # + the VCF anchor base
        return None
    return seq


def record_to_site(rec, options: VariantOptions) -> Optional[VariantSite]:
    """The :class:`VariantSite` of a pysam VariantRecord, or None when the
    record is filtered out (not PASS, sample not carrying an ALT, no usable
    allele)."""
    if options.pass_only:
        filters = list(rec.filter.keys())
        if filters and filters != ['PASS']:
            return None
    ref = _usable_allele(rec.ref, options.max_allele_len)
    if ref is None:
        return None
    alts = rec.alts or ()
    if options.sample is not None:
        try:
            gt = rec.samples[options.sample].get('GT')
        except KeyError:
            gt = None
        called = [a for a in (gt or ()) if a is not None]
        if not any(a > 0 for a in called):
            return None
        if options.het_only and len(set(called)) < 2:
            return None
    usable = [_usable_allele(a, options.max_allele_len) for a in alts]
    if not any(a is not None for a in usable):
        return None
    alleles = (ref,) + tuple(usable)
    indel = any(a is not None and len(a) != len(ref) for a in usable)
    start = int(rec.start)
    return VariantSite(start, start + len(ref), alleles, indel)


class SiteWindow:
    """Sites of one contig fetched in growing chunks while reads (sorted by
    start) advance through a region; ``sites_in(start, end)`` returns the
    sites whose REF span lies inside ``[start, end)``."""

    CHUNK = 1_000_000

    def __init__(self, vcf, chrom: str, options: VariantOptions, origin: int):
        self.vcf = vcf
        self.chrom = chrom
        self.options = options
        self.fetched_end = int(origin)
        self.starts: List[int] = []
        self.sites: List[VariantSite] = []
        self.missing_contig = False

    def _extend(self, end: int) -> None:
        if self.missing_contig:
            return
        stop = max(int(end), self.fetched_end + self.CHUNK)
        try:
            records = self.vcf.fetch(self.chrom, self.fetched_end, stop)
        except ValueError:  # contig not in the VCF index
            self.missing_contig = True
            return
        lo = self.fetched_end
        new = []
        for rec in records:
            if not lo <= rec.start < stop:
                continue  # fetched earlier, or (multi-chunk) by its overlap
            site = record_to_site(rec, self.options)
            if site is not None:
                new.append(site)
        new.sort(key=lambda s: (s.start, s.end))
        self.sites.extend(new)
        self.starts.extend(s.start for s in new)
        self.fetched_end = stop

    def sites_in(self, start: int, end: int) -> List[VariantSite]:
        if end > self.fetched_end:
            self._extend(end)
        i = bisect_left(self.starts, start)
        out = []
        while i < len(self.sites) and self.sites[i].start < end:
            site = self.sites[i]
            if site.end <= end:
                out.append(site)
            i += 1
        return out


def vcf_md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_indexed_vcf(path: str, work_dir: str) -> str:
    """Path of an indexed copy of ``path`` (``path`` itself when it already
    has a tabix/CSI index). Plain VCF is bgzipped and indexed into
    ``work_dir``; bgzipped VCF and BCF are indexed there."""
    with pysam.VariantFile(path) as vcf:
        if vcf.index is not None:
            return path
        is_bcf = vcf.is_bcf if hasattr(vcf, 'is_bcf') else path.endswith('.bcf')
        compression = getattr(vcf, 'compression', None)
    base = os.path.basename(path)
    if is_bcf:
        target = os.path.join(work_dir, base)
        shutil.copyfile(path, target)
        pysam.bcftools.index('-c', target)
        return target
    if compression == 'BGZF':
        target = os.path.join(work_dir, base)
        shutil.copyfile(path, target)
        pysam.tabix_index(target, preset='vcf', force=True, keep_original=True)
        return target
    plain = path
    if compression == 'GZIP':  # gzip, not bgzip: decompress first
        import gzip
        plain = os.path.join(work_dir, re.sub(r'\.gz$', '', base) or 'variants.vcf')
        with gzip.open(path, 'rb') as src, open(plain, 'wb') as dst:
            shutil.copyfileobj(src, dst)
    target = os.path.join(work_dir, re.sub(r'\.vcf$', '', os.path.basename(plain)) + '.vcf.gz')
    pysam.tabix_compress(plain, target, force=True)
    pysam.tabix_index(target, preset='vcf', force=True)
    return target


def vcf_contigs(path: str) -> Optional[set]:
    """Contigs an indexed VCF has records on (``None`` if unknown)."""
    try:
        with pysam.VariantFile(path) as vcf:
            if vcf.index is not None:
                return set(vcf.index.keys())
    except (OSError, ValueError):
        pass
    return None


def vcf_samples(path: str) -> List[str]:
    with pysam.VariantFile(path) as vcf:
        return list(vcf.header.samples)


# ---------------------------------------------------------------------------
# DAF variant-file discovery
# ---------------------------------------------------------------------------

def _cl_value(tokens: Sequence[str], flag: str) -> Optional[str]:
    for i, token in enumerate(tokens):
        if token == flag and i + 1 < len(tokens):
            return tokens[i + 1]
        if token.startswith(flag + '='):
            return token.split('=', 1)[1]
    return None


def find_daf_snp_vcf(bam_path: str, header=None) -> Tuple[Optional[str], List[str]]:
    """The DAF SNP-screen VCF written for ``bam_path``, and the candidate
    paths looked at.

    Looked for, in order: the prefix or mask a ``fiberhmm-call`` @PG command
    line recorded (``--daf-snp-output-prefix P`` -> ``P.vcf``,
    ``--daf-snp-mask M.bed`` -> ``M.vcf``; relative paths are tried against
    the BAM's directory), then ``qc/<BAM stem>.daf_snps.vcf`` (fiberhmm-call's
    default) and ``<BAM stem>.daf_snps.vcf`` (fiberhmm-daf-snps' default)
    next to the BAM.
    """
    bam_dir = os.path.dirname(os.path.abspath(bam_path))
    stem = os.path.basename(bam_path)
    if stem.endswith('.bam'):
        stem = stem[:-4]
    candidates: List[str] = []
    if header is None:
        try:
            with pysam.AlignmentFile(bam_path, 'rb', check_sq=False) as bam:
                header = bam.header.to_dict()
        except (OSError, ValueError):
            header = {}
    elif hasattr(header, 'to_dict'):
        header = header.to_dict()
    for pg in reversed((header or {}).get('PG', []) or []):
        program = str(pg.get('PN') or pg.get('ID') or '')
        if not program.startswith('fiberhmm-call'):
            continue
        tokens = str(pg.get('CL', '')).split()
        prefix = _cl_value(tokens, '--daf-snp-output-prefix')
        if prefix:
            candidates.append(prefix + '.vcf')
        mask = _cl_value(tokens, '--daf-snp-mask')
        if mask:
            candidates.append(re.sub(r'\.bed$', '', mask) + '.vcf')
    candidates.append(os.path.join(bam_dir, 'qc', f'{stem}.daf_snps.vcf'))
    candidates.append(os.path.join(bam_dir, f'{stem}.daf_snps.vcf'))
    tried = []
    for candidate in candidates:
        options = [candidate]
        if not os.path.isabs(candidate):
            options.append(os.path.join(bam_dir, candidate))
        for path in options:
            if path in tried:
                continue
            tried.append(path)
            if os.path.isfile(path):
                return path, tried
    return None, tried


def bam_looks_daf(bam_path: str, header=None) -> bool:
    """BAM-level DAF evidence: a fiberhmm-daf-encode @PG, a fiberhmm-call
    @PG declaring ``mode=daf``, or the engine's MM/R-Y sniff."""
    if header is None:
        with pysam.AlignmentFile(bam_path, 'rb', check_sq=False) as bam:
            header = bam.header.to_dict()
    elif hasattr(header, 'to_dict'):
        header = header.to_dict()
    for pg in (header or {}).get('PG', []) or []:
        program = str(pg.get('PN') or pg.get('ID') or '')
        if program.startswith('fiberhmm-daf-encode'):
            return True
        if program.startswith('fiberhmm-call') and 'mode=daf' in str(pg.get('DS', '')):
            return True
    try:
        from fiberhmm.inference.engine import detect_mode_from_bam
        return detect_mode_from_bam(bam_path, n_sample=50) == 'daf'
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Per-read genotyping
# ---------------------------------------------------------------------------

def _read_layout(read):
    """``(r2q, insert_after, ref_start)`` for an aligned read.

    ``r2q[p - ref_start]`` is the query index aligned to reference ``p``,
    ``-1`` for a deleted base and ``-2`` for an ``N`` skip.
    ``insert_after[p]`` is the ``(qstart, qend)`` of the insertion that
    follows reference base ``p``.
    """
    ref_start = int(read.reference_start)
    ref_end = int(read.reference_end)
    r2q = np.full(max(0, ref_end - ref_start), -1, dtype=np.int64)
    insert_after: Dict[int, Tuple[int, int]] = {}
    r = ref_start
    q = 0
    for op, n in read.cigartuples or ():
        if op in (_M, _EQ, _X):
            r2q[r - ref_start:r - ref_start + n] = np.arange(q, q + n)
            r += n
            q += n
        elif op == _I:
            if r > ref_start:
                insert_after[r - 1] = (q, q + n)
            q += n
        elif op == _D:
            r += n
        elif op == _N:
            r2q[r - ref_start:r - ref_start + n] = -2
            r += n
        elif op == _S:
            q += n
    return r2q, insert_after, ref_start


def daf_conversion(read, options: VariantOptions) -> Optional[str]:
    """The conversion a DAF read may carry: ``'CT'``, ``'GA'``, ``'both'``
    (a DAF read whose strand is undecided) or ``None`` (not DAF / no
    ambiguity handling). Uses FiberHMM's DAF strand rule (``st`` tag on R/Y
    input, else the reads' R/Y marks or reference mismatches)."""
    if options.daf == 'off':
        return None
    seq = read.query_sequence or ''
    has_ry = ('R' in seq) or ('Y' in seq)
    if options.daf == 'auto' and not (options.bam_is_daf or has_ry or read.has_tag('st')):
        return None
    from fiberhmm.daf.insert_consensus import STRAND_CT, _read_strand
    try:
        strand = _read_strand(read, md_first=True)
    except Exception:
        strand = None
    if strand is None:
        return 'both'
    return 'CT' if strand == STRAND_CT else 'GA'


def _fits(observed: str, allele: str, conversion: Optional[str]) -> bool:
    """Whether the read's ``observed`` bases can come from ``allele`` given
    the read's DAF conversion (T may be a converted C on C->T reads, A a
    converted G on G->A reads; R/Y were rewritten to A/T already)."""
    if len(observed) != len(allele):
        return False
    if conversion is None:
        return observed == allele
    ct = conversion in ('CT', 'both')
    ga = conversion in ('GA', 'both')
    for o, a in zip(observed, allele):
        if o == a:
            continue
        if ct and o == 'T' and a == 'C':
            continue
        if ga and o == 'A' and a == 'G':
            continue
        return False
    return True


_RY_TO_BASE = str.maketrans({'R': 'A', 'Y': 'T', 'r': 'a', 'y': 't'})


def genotype_read(read, sites: Iterable[VariantSite], conversion: Optional[str] = None,
                  layout=None) -> List[Tuple[int, int, int, int]]:
    """``[(start, end, allele, bq), ...]`` for each site the read covers.

    A site is covered when its REF span lies inside the alignment and does
    not run into an ``N`` skip; an indel site also needs one aligned base
    after it, so an insertion after its last REF base is visible. ``bq`` is
    the base quality of an SNV base, the minimum over the compared bases of
    a longer site, ``-1`` without qualities or bases.
    """
    seq = (read.query_sequence or '').upper().translate(_RY_TO_BASE)
    quals = read.query_qualities
    has_quals = quals is not None and len(quals) == len(seq) and (len(quals) == 0 or quals[0] != 255)
    r2q, insert_after, ref_start = layout or _read_layout(read)
    ref_end = ref_start + len(r2q)
    out = []
    for site in sites:
        s, e = site.start, site.end
        if s < ref_start or e > ref_end or (site.indel and e >= ref_end):
            continue
        idx = r2q[s - ref_start:e - ref_start]
        if (idx == -2).any():
            continue
        if (idx == -1).all():
            out.append((s, e, ALLELE_DELETED, -1))
            continue
        if not site.indel and e - s == 1:
            q = int(idx[0])
            observed = seq[q]
            bq = int(quals[q]) if has_quals else -1
        else:
            parts = []
            qs: List[int] = []
            last = e - 1 if site.indel else e - 2
            for offset, q in enumerate(idx.tolist()):
                if q >= 0:
                    parts.append(seq[q])
                    qs.append(q)
                p = s + offset
                if p <= last and p in insert_after:
                    a, b = insert_after[p]
                    parts.append(seq[a:b])
                    qs.extend(range(a, b))
            observed = ''.join(parts)
            bq = int(min(quals[q] for q in qs)) if (has_quals and qs) else -1
        exact = [k for k, allele in enumerate(site.alleles)
                 if allele is not None and allele == observed]
        if conversion is None:
            allele = exact[0] if exact else ALLELE_OTHER
        else:
            fits = [k for k, allele in enumerate(site.alleles)
                    if allele is not None and _fits(observed, allele, conversion)]
            if len(fits) == 1:
                allele = fits[0]
            elif len(fits) > 1:
                allele = ALLELE_DAF_AMBIGUOUS
            else:
                allele = ALLELE_OTHER
        out.append((s, e, allele, bq))
    return out


# ---------------------------------------------------------------------------
# Structural events
# ---------------------------------------------------------------------------

def sv_events(read, min_len: int = DEFAULT_SV_MIN_LEN,
              min_clip: int = DEFAULT_SV_MIN_CLIP) -> List[Tuple[int, int, int, int]]:
    """``[(start, end, type, length), ...]`` of a read's structural events.

    Deletion: the deleted reference span. Insertion: the 1 bp reference base
    before it (the VCF anchor convention; the first aligned base for an
    insertion at the alignment start). Clips: the first / last aligned base,
    ``length`` = soft plus hard clipped bases at that end (left/right in
    reference orientation). ``N`` skips are not deletions.
    """
    cigar = read.cigartuples or ()
    ref_start = int(read.reference_start)
    ref_end = int(read.reference_end)
    events = []
    left = 0
    i = 0
    while i < len(cigar) and cigar[i][0] in (_S, _H):
        left += cigar[i][1]
        i += 1
    right = 0
    j = len(cigar) - 1
    while j >= i and cigar[j][0] in (_S, _H):
        right += cigar[j][1]
        j -= 1
    if left >= min_clip:
        events.append((ref_start, ref_start + 1, SV_CLIP_LEFT, left))
    r = ref_start
    for op, n in cigar[i:j + 1]:
        if op in (_M, _EQ, _X, _N):
            r += n
        elif op == _D:
            if n >= min_len:
                events.append((r, r + n, SV_DEL, n))
            r += n
        elif op == _I:
            if n >= min_len:
                anchor = r - 1 if r > ref_start else r
                events.append((anchor, anchor + 1, SV_INS, n))
    if right >= min_clip:
        events.append((ref_end - 1, ref_end, SV_CLIP_RIGHT, right))
    return events


def sa_column(read) -> str:
    """The ``SA`` tag verbatim, ``.`` when absent; a tag longer than
    :data:`SA_MAX_CHARS` keeps its leading whole entries and ends in
    ``...``."""
    try:
        sa = str(read.get_tag('SA'))
    except KeyError:
        return '.'
    if not sa:
        return '.'
    sa = sa.replace('\t', ' ')
    if len(sa) <= SA_MAX_CHARS:
        return sa
    cut = sa.rfind(';', 0, SA_MAX_CHARS - 3)
    return (sa[:cut + 1] if cut >= 0 else '') + '...'


def full_read_length(read) -> int:
    """Read length including soft and hard clips."""
    try:
        return int(read.infer_read_length() or 0)
    except Exception:
        return int(read.query_length or 0)


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def pack_blocks(blocks: Sequence[tuple]) -> List[List[tuple]]:
    """Split ``(start, end, ...)`` blocks into rows of ascending,
    non-overlapping blocks (first fit, in start order)."""
    rows: List[List[tuple]] = []
    ends: List[int] = []
    for block in sorted(blocks, key=lambda b: (b[0], b[1])):
        for k, last in enumerate(ends):
            if block[0] >= last:
                rows[k].append(block)
                ends[k] = block[1]
                break
        else:
            rows.append([block])
            ends.append(block[1])
    return rows


__all__ = [
    'ALLELE_DAF_AMBIGUOUS', 'ALLELE_DELETED', 'ALLELE_OTHER',
    'SV_CLIP_LEFT', 'SV_CLIP_RIGHT', 'SV_DEL', 'SV_INS',
    'SiteWindow', 'VariantOptions', 'VariantSite',
    'bam_looks_daf', 'daf_conversion', 'find_daf_snp_vcf', 'genotype_read',
    'pack_blocks', 'prepare_indexed_vcf', 'record_to_site', 'sa_column',
    'sv_events', 'vcf_md5',
]
