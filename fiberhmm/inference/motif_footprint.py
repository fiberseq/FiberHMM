"""Motif-anchored footprint scoring from FiberHMM calls (``fiberhmm-footprint``).

The FiberHMM analogue of fibertools ``ft footprint``. For every molecule x motif
it reads the calls already in the BAM (no re-decoding) and decides:

* **spanning**: the alignment spans the motif (ft's rule: alignment start <
  motif start and alignment end >= motif end, reference coordinates).
* **eligible**: an MSP of the molecule spans the motif, with the same ft rule
  applied to the MSP's reference projection. MSPs come from the BAM's own
  FiberHMM accessibility calls (MA ``msp``, else fibertools ``Ma``, else legacy
  ``as/al``), or from ``msp_bam`` when given.
* **footprinted**: one FiberHMM TF/sub-nucleosomal call (MA ``tf`` layer) with
  ``tq >= 10 * min_llr`` whose bounds contain the complete motif
  (``start <= motif start`` and ``end >= motif end``), and whose conservative
  reference width is within ``[min_width, max_width]`` (inclusive; 0 = no bound). Bounds are the outer edges
  by default: the conservative call extended by the edge ambiguity stored in its
  ``el``/``er`` bytes (``ma_tags.ambiguity_to_edge``), i.e. up to but excluding
  the terminating informative position on each side. ``outer_bound='inclusive'``
  extends one more base each side so the terminating position itself is
  included. Optionally (``require_unmodified``) no m6A call (ML >= ``min_ml``)
  may fall in the motif.

Per-molecule codes mirror ft's footprint codes for a one-module motif:
bit 0 = MSP spans the motif, bit 1 = footprinted. Code 3 is a footprinted,
eligible molecule; only code 3 counts as footprinted in the per-motif table.
Molecules that overlap but do not span the motif get code -1.
"""
from __future__ import annotations

import gzip
import os
import sys
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from fiberhmm.io.ma_tags import EDGE_AMBIGUITY_SAT, flip_interval_frame, parse_aq_array, parse_ma_tag

# Label-free width cap: the 95th percentile (74 bp) of the conservative widths
# of 168,943 FiberHMM TF calls (LLR >= 5) whose outer bounds overlap a benchmark
# motif in an MSP-eligible molecule, pooled over CTCF, NRF1, REST and USF1
# motifs in five cell lines; no ChIP labels used (90th = 61 bp, 99th = 87 bp).
DEFAULT_MAX_WIDTH = 74

SPAN_NOT = -1          # overlaps the motif but does not span it
M6A_CODES = ('a', 21839)

# Per-molecule status strings
FOOTPRINTED = 'footprinted'
NOT_FOOTPRINTED = 'not_footprinted'
NO_MSP = 'no_msp'
UNANNOTATED = 'unannotated'
NOT_SPANNING = 'not_spanning'

# Reasons a spanning molecule is not footprinted (also the fpneg_ quality byte)
REASONS = {1: 'no_call', 2: 'partial_overlap', 3: 'too_wide', 4: 'modified', 5: 'too_narrow'}
REASON_CODE = {v: k for k, v in REASONS.items()}

FP_PREFIX = 'fp_'
MAX_LAYERS_WITHOUT_CONFIRMATION = 20
FPNEG_PREFIX = 'fpneg_'
CONTRACT = 'FIBERHMM-FOOTPRINT-MA:v1:'


# --------------------------------------------------------------------------- motifs

def sanitize_label(name: str) -> str:
    """A motif label usable inside an MA type name.

    The Molecular-annotation spec's type string is ``name`` + strand
    (``+``/``-``/``.``) + quality spec (``Q``...), so the name itself may not
    contain ``+``, ``-`` or ``.``; FiberHMM's grammar is ``[A-Za-z0-9_]+``.
    Every other character run becomes ``_``; leading/trailing ``_`` are
    dropped; an empty result becomes ``motif``.
    """
    label = re.sub(r'[^A-Za-z0-9_]+', '_', str(name)).strip('_')
    if label.startswith(FP_PREFIX):
        label = label[len(FP_PREFIX):]
    return label or 'motif'


def check_layer_count(labels, allow_many: bool, layer_name=None) -> None:
    """Refuse a run that would write more than MAX_LAYERS_WITHOUT_CONFIRMATION
    layer pairs (e.g. a multi-TF bigBed without --motif-name) unless allowed."""
    if layer_name or allow_many or len(labels) <= MAX_LAYERS_WITHOUT_CONFIRMATION:
        return
    shown = ', '.join(sorted(labels)[:5])
    raise ValueError(f'more than {MAX_LAYERS_WITHOUT_CONFIRMATION} motif sets ({shown}, ...): each becomes its own MA '
                     'layer pair. Pick factors with --motif-name, merge them with --layer-name, or pass '
                     '--allow-many-layers')


def layer_names(label: str) -> Tuple[str, str]:
    """(footprinted layer, eligible-not-footprinted layer) for a sanitized label."""
    return FP_PREFIX + label, FPNEG_PREFIX + label


@dataclass(frozen=True)
class Motif:
    index: int
    chrom: str
    start: int
    end: int
    strand: str
    name: str
    label: str

    @property
    def motif_id(self) -> str:
        return f'{self.chrom}:{self.start}-{self.end}:{self.strand}'


def load_motif_bed(path, layer_name: Optional[str] = None, motif_filter=None) -> List[Motif]:
    """Motif BED (BED3+; BED6 strand honoured). Same input conventions as the
    consensus BED loader (gzip, ``#``/track/browser lines skipped, 0-based
    half-open), except that repeated names are expected: the name column is the
    motif set, and each distinct name becomes one MA layer unless
    ``layer_name`` overrides it."""
    opener = gzip.open if str(path).endswith('.gz') else open
    motifs = []
    with opener(path, 'rt') as handle:
        for n, line in enumerate(handle, 1):
            if not line.strip() or line.startswith(('#', 'track ', 'browser ')):
                continue
            f = line.rstrip('\n').split('\t') if '\t' in line else line.split()
            if len(f) < 3:
                raise ValueError(f'BED line {n}: expected at least 3 fields')
            try:
                a, b = int(f[1]), int(f[2])
            except ValueError:
                raise ValueError(f'BED line {n}: start/end must be integers') from None
            if a < 0 or b <= a:
                raise ValueError(f'BED line {n}: invalid half-open coordinates {a}-{b}')
            name = f[3] if len(f) >= 4 and f[3] not in ('', '.') else 'motif'
            if len(f) >= 6 and f[5] not in ('+', '-', '.'):
                raise ValueError(f'BED line {n}: strand must be +, - or .')
            strand = '-' if len(f) >= 6 and f[5] == '-' else '+'
            if motif_filter is not None and not _bed_row_passes(motif_filter, name, f[4] if len(f) >= 5 else None):
                continue
            label = sanitize_label(layer_name if layer_name else name)
            motifs.append(Motif(len(motifs), f[0], a, b, strand, name, label))
    if not motifs:
        raise ValueError('BED contains no motifs' + (' passing the filters' if motif_filter is not None else ''))
    return motifs


def _bed_row_passes(filt, name, score) -> bool:
    names = getattr(filt, '_names', None)
    if names is not None and not ({name.lower(), name.split('.')[0].lower()} & names):
        return False
    if filt.min_score is not None:
        try:
            return float(score) >= filt.min_score
        except (TypeError, ValueError):
            return False
    return True


class BigBedMotifs:
    """Motifs streamed from a bigBed, region by region, in genomic order.

    Only regions the BAM has reads in are queried (or the ``regions`` given),
    in ``chunk_bp`` pieces; a motif belongs to the chunk holding its start, so
    none is repeated. Labels come from a TF-name field when the file has one
    (JASPAR's ``TFName``), else from the name field, unless ``layer_name``."""

    def __init__(self, path, bam_path=None, regions=None, motif_filter=None, layer_name=None,
                 chunk_bp: int = 1_000_000, backend=None):
        from fiberhmm.io.bigbed_motifs import BigBedReader, MotifFilter, RowInterpreter
        self.path, self.bam_path, self.layer_name, self.chunk_bp = str(path), bam_path, layer_name, chunk_bp
        self.reader = BigBedReader(path, backend=backend)
        self.filter = motif_filter or MotifFilter()
        self.rows = RowInterpreter(self.reader.fields, self.reader.descriptions, self.filter)
        self.regions = regions
        self._cache = {}
        self._next_index = 0
        self.max_len = 0

    def _query_regions(self):
        import pysam
        from fiberhmm.io.bigbed_motifs import parse_region
        bb_chroms = self.reader.chroms()
        if self.regions:
            return [parse_region(r, bb_chroms) for r in self.regions]
        with pysam.AlignmentFile(self.bam_path) as bam:
            lengths = dict(zip(bam.references, bam.lengths))
            try:
                covered = [s.contig for s in bam.get_index_statistics() if s.mapped]
            except (ValueError, AttributeError):
                covered = list(bam.references)
            out = []
            for chrom in covered:
                if bb_chroms is not None and chrom not in bb_chroms:
                    continue
                for a in range(0, lengths[chrom], self.chunk_bp):
                    b = min(a + self.chunk_bp, lengths[chrom])
                    if next(bam.fetch(chrom, a, b), None) is not None:
                        out.append((chrom, a, b))
        return out

    def _chunk(self, chrom, a, b, index_from=None):
        """Motifs whose start is in [a, b), sorted."""
        out = []
        for c, s, e, rest in self.reader.entries(chrom, a, b):
            if not (a <= s < b) or e <= s:
                continue
            got = self.rows.interpret(rest)
            if got is None:
                continue
            name, label_source, strand = got
            out.append((s, e, strand, name, label_source))
        out.sort()
        motifs = []
        for s, e, strand, name, label_source in out:
            label = sanitize_label(self.layer_name if self.layer_name else label_source)
            motifs.append(Motif(-1, chrom, s, e, strand, name, label))
            self.max_len = max(self.max_len, e - s)
        return motifs

    def __iter__(self):
        """All motifs in genomic order, numbered as they are produced."""
        for chrom, a, b in self._query_regions():
            for c0 in range(a, b, self.chunk_bp):
                for m in self._chunk(chrom, c0, min(b, c0 + self.chunk_bp)):
                    yield Motif(self._next_index, m.chrom, m.start, m.end, m.strand, m.name, m.label)
                    self._next_index += 1

    # --- lookups for the annotated-BAM pass ---
    def _chunk_cached(self, chrom, k):
        key = (chrom, k)
        if key not in self._cache:
            if len(self._cache) >= 16:
                self._cache.pop(next(iter(self._cache)))
            ms = self._chunk(chrom, k * self.chunk_bp, (k + 1) * self.chunk_bp)
            self._cache[key] = [Motif(hash((chrom, m.start, m.end, m.strand, m.name)) & 0x7FFFFFFFFFFF, m.chrom,
                                      m.start, m.end, m.strand, m.name, m.label) for m in ms]
        return self._cache[key]

    def overlapping(self, chrom, a, b):
        if self.regions:
            from fiberhmm.io.bigbed_motifs import parse_region
            spans = [parse_region(r) for r in self.regions]
        else:
            spans = None
        pad = max(self.max_len, 1000)
        out = []
        for k in range(max(0, a - pad) // self.chunk_bp, (b - 1) // self.chunk_bp + 1):
            for m in self._chunk_cached(chrom, k):
                if m.start < b and m.end > a and (spans is None or any(
                        c == chrom and s <= m.start < e for c, s, e in spans)):
                    out.append(m)
        return out

    def cluster_of(self, m):
        k = m.start // self.chunk_bp
        return (m.chrom, k), m.chrom, k * self.chunk_bp, (k + 1) * self.chunk_bp


def iter_clusters(motifs: Iterable[Motif], max_gap: int = 2000, max_span: int = 200_000):
    """Group nearby motifs (input in genomic order) so each group's reads are
    fetched and parsed once; lazy, so a streamed motif source stays streamed."""
    last: List[Motif] = []
    c_end = -1
    for m in motifs:
        if last and last[0].chrom == m.chrom and m.start - c_end <= max_gap and m.end - last[0].start <= max_span:
            last.append(m)
            c_end = max(c_end, m.end)
            continue
        if last:
            yield last
        last, c_end = [m], m.end
    if last:
        yield last


def motif_clusters(motifs: Sequence[Motif], max_gap: int = 2000, max_span: int = 200_000) -> List[List[Motif]]:
    """Group nearby motifs so each group's reads are fetched and parsed once."""
    return list(iter_clusters(sorted(motifs, key=lambda m: (m.chrom, m.start, m.end, m.index)), max_gap, max_span))


# --------------------------------------------------------------------------- options

@dataclass(frozen=True)
class FootprintOptions:
    min_llr: float = 5.0
    edges: str = 'outer'               # outer | conservative
    outer_bound: str = 'exclusive'     # exclusive | inclusive (of the terminating informative position)
    max_width: int = DEFAULT_MAX_WIDTH  # conservative reference width cap, bp; 0 = off
    min_width: int = 0                  # conservative reference width floor, bp (inclusive); 0 = off
    require_unmodified: bool = False
    min_ml: int = 125
    flag_filter: int = 0x904           # unmapped | secondary | supplementary
    min_mapq: int = 0

    def __post_init__(self):
        if self.edges not in ('outer', 'conservative'):
            raise ValueError('edges must be outer or conservative')
        if self.outer_bound not in ('exclusive', 'inclusive'):
            raise ValueError('outer_bound must be exclusive or inclusive')
        if self.max_width < 0:
            raise ValueError('max_width must be >= 0 (0 disables the cap)')
        if self.min_width < 0:
            raise ValueError('min_width must be >= 0 (0 disables it)')
        if self.max_width and self.min_width > self.max_width:
            raise ValueError('min_width must not exceed max_width')
        if self.min_llr < 0:
            raise ValueError('min_llr must be >= 0')

    @property
    def min_tq(self) -> int:
        return int(round(self.min_llr * 10))


# --------------------------------------------------------------------------- molecule parsing

def decode_ambiguity(q: int) -> int:
    """Inverse of ``ma_tags.ambiguity_to_edge``: bp of non-informative sequence
    between the conservative boundary and the terminating informative position.
    A saturated byte (0) is read as ``EDGE_AMBIGUITY_SAT`` (a lower bound)."""
    q = int(q)
    if q <= 0:
        return EDGE_AMBIGUITY_SAT
    return int(round(EDGE_AMBIGUITY_SAT * (1 - q / 255)))


@dataclass
class TFCall:
    q0: int        # SEQ-frame query start (conservative)
    q1: int        # SEQ-frame query end (conservative, exclusive)
    amb_l: int     # SEQ-frame left ambiguity (bp)
    amb_r: int     # SEQ-frame right ambiguity (bp)
    tq: int
    x: int = -1    # reference projection of the conservative call
    y: int = -1
    ox: int = -1   # reference projection of the outer bounds
    oy: int = -1

    @property
    def width(self) -> int:
        return self.y - self.x


class _RefMap:
    """Query -> reference projection: an interval's reference span is the first
    and last aligned base inside it (clipped to the read)."""

    def __init__(self, read):
        # Same as get_reference_positions(full_length=True), built from the CIGAR.
        refs = np.full(int(read.query_length or 0), -1, dtype=np.int64)
        q, r = 0, int(read.reference_start)
        for op, n in read.cigartuples or ():
            if op in (0, 7, 8):          # M, =, X
                refs[q:q + n] = np.arange(r, r + n)
                q += n
                r += n
            elif op in (1, 4):           # I, S
                q += n
            elif op in (2, 3):           # D, N
                r += n
        self.refs = refs
        n = len(refs)
        idx = np.arange(n)
        aligned = refs >= 0
        # next aligned query index at or after i; previous aligned at or before i
        nxt = np.where(aligned, idx, n)
        self.next_idx = np.minimum.accumulate(nxt[::-1])[::-1] if n else nxt
        prv = np.where(aligned, idx, -1)
        self.prev_idx = np.maximum.accumulate(prv) if n else prv
        self.n = n

    def project(self, a: int, b: int) -> Optional[Tuple[int, int]]:
        a = max(int(a), 0)
        b = min(int(b), self.n)
        if b <= a:
            return None
        i = self.next_idx[a]
        j = self.prev_idx[b - 1]
        if i >= self.n or j < 0 or i > j:
            return None
        return int(self.refs[i]), int(self.refs[j]) + 1

    def query_span(self, ref_start: int, ref_end: int) -> Optional[Tuple[int, int]]:
        hit = np.flatnonzero((self.refs >= ref_start) & (self.refs < ref_end))
        if not len(hit):
            return None
        return int(hit[0]), int(hit[-1]) + 1


def _group_is_molecular(frame, name, strand):
    return frame.is_molecular(name, strand)


def parse_tf_and_nuc(read, group_frame) -> Tuple[Optional[List[TFCall]], List[Tuple[int, int]]]:
    """TF calls and nucleosomes of a read's FiberHMM ``MA`` tag, in SEQ frame.

    Returns ``(None, [])`` when the read has no ``MA`` (no FiberHMM calls).
    Reverse reads in a molecular-frame group are flipped and their ``el``/``er``
    bytes swapped (molecular left = genomic right)."""
    if not read.has_tag('MA'):
        return None, []
    parsed = parse_ma_tag(str(read.get_tag('MA')))
    types = parsed['raw_types']
    quals = parse_aq_array(read.get_tag('AQ') if read.has_tag('AQ') else None,
                           [t[2] for t in types], [len(t[3]) for t in types])
    L = int(parsed['read_length'])
    calls: List[TFCall] = []
    nucs: List[Tuple[int, int]] = []
    k = 0
    for name, strand, spec, intervals in types:
        flip = bool(read.is_reverse) and _group_is_molecular(group_frame, name, strand)
        for s, length in intervals:
            q = quals[k]
            k += 1
            if name not in ('tf', 'nuc'):
                continue
            if flip:
                s, length = flip_interval_frame(s, length, L)
            if name == 'nuc':
                nucs.append((s, s + length))
                continue
            if len(q) < 3:
                # tf without edge bytes: no ambiguity known, tq if present
                tq = q[0] if q else 0
                calls.append(TFCall(s, s + length, 0, 0, tq))
                continue
            tq, el, er = q[0], q[1], q[2]
            if flip:
                el, er = er, el
            calls.append(TFCall(s, s + length, decode_ambiguity(el), decode_ambiguity(er), tq))
    return calls, nucs


def ref_msps(read, frames, refmap: _RefMap) -> Optional[List[Tuple[int, int]]]:
    """Reference projections of the read's MSPs (``None`` = no accessibility calls)."""
    from fiberhmm.qc.states import msp_intervals_from_tags
    msps = msp_intervals_from_tags(read, frames)
    if msps is None:
        return None
    out = []
    for s, length, _feature in msps:
        p = refmap.project(s, s + length)
        if p:
            out.append(p)
    return out


def m6a_ref_positions(read, refs: np.ndarray, min_ml: int) -> np.ndarray:
    pos = []
    for (base, strand, code), calls in (read.modified_bases or {}).items():
        if code not in M6A_CODES:
            continue
        for q, ml in calls:
            if ml >= min_ml and 0 <= q < len(refs) and refs[q] >= 0:
                pos.append(refs[q])
    return np.array(sorted(pos), dtype=np.int64)


@dataclass
class Molecule:
    key: tuple
    name: str
    ref_start: int
    ref_end: int
    is_reverse: bool
    read_length: int
    hp: str
    calls: Optional[List[TFCall]]
    nucs: List[Tuple[int, int]]
    msps: Optional[List[Tuple[int, int]]]
    refmap: Optional[_RefMap]
    m6a: Optional[np.ndarray] = None
    duplicate: bool = False


def read_key(read) -> tuple:
    return (read.query_name, int(read.flag), int(read.reference_id), int(read.reference_start))


def parse_molecule(read, group_frame, frames, options: FootprintOptions, msp_override=None) -> Molecule:
    refmap = _RefMap(read)
    calls, nucs_q = parse_tf_and_nuc(read, group_frame)
    inc = 1 if options.outer_bound == 'inclusive' else 0
    proj_calls = []
    for c in calls or ():
        pc = refmap.project(c.q0, c.q1)
        if pc is None:
            continue
        po = refmap.project(c.q0 - c.amb_l - inc, c.q1 + c.amb_r + inc)
        c.x, c.y = pc
        c.ox, c.oy = po if po else pc
        proj_calls.append(c)
    nucs = [p for p in (refmap.project(a, b) for a, b in nucs_q) if p]
    if msp_override is not None:
        msps = msp_override.get(read.query_name)
    else:
        msps = ref_msps(read, frames, refmap)
    m6a = m6a_ref_positions(read, refmap.refs, options.min_ml) if options.require_unmodified else None
    return Molecule(read_key(read), read.query_name, int(read.reference_start), int(read.reference_end),
                    bool(read.is_reverse), int(read.query_length or len(refmap.refs)),
                    str(read.get_tag('HP')) if read.has_tag('HP') else 'UNK',
                    None if calls is None else proj_calls, nucs, msps, refmap, m6a, bool(read.is_duplicate))


# --------------------------------------------------------------------------- scoring

def ft_spans(a: int, b: int, m0: int, m1: int) -> bool:
    """fibertools ``ReferenceMotif::spans`` (motif end inclusive): a < m0 and b >= m1."""
    return a < m0 and b >= m1


@dataclass
class Observation:
    motif_index: int
    read_name: str
    key: tuple
    code: int
    status: str
    reason: str = ''
    call: Optional[TFCall] = None
    edge: str = ''           # conservative | outer (which bounds contained the motif)
    nuc_overlap: bool = False
    hp: str = 'UNK'
    query_span: Optional[Tuple[int, int]] = None   # SEQ-frame query span of the motif (for --out-bam)
    is_reverse: bool = False
    read_length: int = 0
    read_end: int = 0
    duplicate: bool = False
    # Conservative widths of every call that would footprint this eligible molecule under any width
    # window (LLR, containment and, with require_unmodified, no m6A): used by width calibration.
    widths: Tuple[int, ...] = ()


def score(mol: Molecule, motif: Motif, options: FootprintOptions, want_span: bool = False) -> Observation:
    m0, m1 = motif.start, motif.end
    nuc = any(a < m1 and b > m0 for a, b in mol.nucs)
    base = dict(motif_index=motif.index, read_name=mol.name, key=mol.key, nuc_overlap=nuc, hp=mol.hp,
                is_reverse=mol.is_reverse, read_length=mol.read_length, read_end=mol.ref_end,
                duplicate=mol.duplicate)
    if not ft_spans(mol.ref_start, mol.ref_end, m0, m1):
        return Observation(code=SPAN_NOT, status=NOT_SPANNING, **base)
    eligible = mol.msps is not None and any(ft_spans(a, b, m0, m1) for a, b in mol.msps)
    calls = [c for c in (mol.calls or ()) if c.tq >= options.min_tq or c.tq < 0]   # tq < 0: unknown (bigBed)
    outer = options.edges == 'outer'

    def bounds(c):
        return (c.ox, c.oy) if outer else (c.x, c.y)

    containing = [c for c in calls if bounds(c)[0] <= m0 and bounds(c)[1] >= m1]
    narrow = [c for c in containing if (options.max_width == 0 or c.width <= options.max_width)
              and c.width >= options.min_width]
    chosen, reason = None, ''
    if narrow:
        chosen = min(narrow, key=lambda c: (c.width, -c.tq, c.x))
        if options.require_unmodified and mol.m6a is not None and len(mol.m6a) and \
                np.any((mol.m6a >= m0) & (mol.m6a < m1)):
            reason = 'modified'
    elif containing:
        # No containing call inside [min_width, max_width]: too narrow when every one is below the floor.
        chosen = min(containing, key=lambda c: (c.width, -c.tq, c.x))
        reason = 'too_narrow' if all(c.width < options.min_width for c in containing) else 'too_wide'
    elif any(bounds(c)[0] < m1 and bounds(c)[1] > m0 for c in calls):
        reason = 'partial_overlap'
    else:
        reason = 'no_call'
    fp = chosen is not None and not reason
    code = (1 if eligible else 0) | (2 if fp else 0)
    if mol.calls is None and mol.msps is None:
        status = UNANNOTATED
    elif not eligible:
        status = NO_MSP
    else:
        status = FOOTPRINTED if fp else NOT_FOOTPRINTED
    edge = ''
    if chosen is not None:
        edge = 'conservative' if (chosen.x <= m0 and chosen.y >= m1) else 'outer'
    span = mol.refmap.query_span(m0, m1) if (want_span and eligible) else None
    widths = ()
    if eligible and reason != 'modified' and not (options.require_unmodified and mol.m6a is not None and len(mol.m6a)
                                                  and np.any((mol.m6a >= m0) & (mol.m6a < m1))):
        widths = tuple(sorted({c.width for c in containing}))
    return Observation(code=code, status=status, reason='' if fp else reason, call=chosen, edge=edge,
                       query_span=span, widths=widths, **base)


# --------------------------------------------------------------------------- region worker

def _msp_index(path, chrom, start, end, options: FootprintOptions):
    """read name -> reference MSPs from a separate BAM (e.g. the BAM ft read)."""
    import pysam
    from fiberhmm.qc.states import tag_frames
    out = {}
    with pysam.AlignmentFile(path) as bam:
        frames = tag_frames(bam.header)
        for read in bam.fetch(chrom, max(0, start), end):
            if read.flag & options.flag_filter or read.mapping_quality < options.min_mapq:
                continue
            refmap = _RefMap(read)
            msps = ref_msps(read, frames, refmap)
            if msps is not None:
                out.setdefault(read.query_name, []).extend(msps)
    return out


class _MotifIndex:
    """Motifs of one chromosome sorted by start, for 'which motifs does this read overlap'."""

    def __init__(self, motifs: Sequence[Motif]):
        self.motifs = sorted(motifs, key=lambda m: (m.start, m.index))
        self.starts = [m.start for m in self.motifs]
        self.max_len = max((m.end - m.start for m in self.motifs), default=0)

    def overlapping(self, a: int, b: int) -> List[Motif]:
        from bisect import bisect_left, bisect_right
        lo = bisect_left(self.starts, a - self.max_len)
        hi = bisect_right(self.starts, b - 1)
        return [m for m in self.motifs[lo:hi] if m.start < b and m.end > a]


def score_read(read, index: _MotifIndex, header_frames, options: FootprintOptions, override=None,
               want_spans=False) -> List[Observation]:
    """Observations of one alignment against the motifs it overlaps (none when filtered).

    The read is parsed, scored and released here, so memory does not grow with
    the depth or span of a region: only the small observations are kept."""
    if read.flag & options.flag_filter or read.mapping_quality < options.min_mapq:
        return []
    motifs = index.overlapping(int(read.reference_start), int(read.reference_end or read.reference_start))
    if not motifs:
        return []
    group_frame, frames = header_frames
    mol = parse_molecule(read, group_frame, frames, options, override)
    return [score(mol, m, options, want_span=want_spans) for m in motifs]


def score_cluster(bam, header_frames, cluster: Sequence[Motif], options: FootprintOptions,
                  msp_bam_path=None, want_spans=False) -> List[Observation]:
    """All observations of one motif cluster, per motif in BED order then fetch order."""
    chrom = cluster[0].chrom
    c0 = min(m.start for m in cluster)
    c1 = max(m.end for m in cluster)
    if chrom not in bam.references:
        return []
    override = _msp_index(msp_bam_path, chrom, c0, c1, options) if msp_bam_path else None
    index = _MotifIndex(cluster)
    per_motif = defaultdict(list)
    for read in bam.fetch(chrom, c0, c1):
        for o in score_read(read, index, header_frames, options, override, want_spans):
            per_motif[o.motif_index].append(o)
    return [o for i in sorted(per_motif) for o in per_motif[i]]


_WORKER = {}


def _init_worker(source, options_dict, msp_bam_path, want_spans):
    options = FootprintOptions(**options_dict)
    if isinstance(source, dict):                 # FiberHMM call bigBeds
        from fiberhmm.io.call_bigbeds import CallBigBeds
        _WORKER.update(bb=CallBigBeds(**source), bam=None, options=options, spans=False, msp=None)
        return
    import pysam
    from fiberhmm.io.annotation_frame import MAGroupFrame
    from fiberhmm.qc.states import tag_frames
    bam = pysam.AlignmentFile(source)
    _WORKER.update(bam=bam, bb=None, frames=(MAGroupFrame(bam.header), tag_frames(bam.header)),
                   options=options, msp=msp_bam_path, spans=want_spans)


def score_cluster_bigbed(calls, cluster: Sequence[Motif], options: FootprintOptions) -> List[Observation]:
    """:func:`score_cluster` for molecules read from FiberHMM call bigBeds."""
    index = _MotifIndex(cluster)
    per_motif = defaultdict(list)
    c0, c1 = min(m.start for m in cluster), max(m.end for m in cluster)
    for mol in calls.molecules(cluster[0].chrom, c0, c1, options):
        for m in index.overlapping(mol.ref_start, mol.ref_end):
            per_motif[m.index].append(score(mol, m, options))
    return [o for i in sorted(per_motif) for o in per_motif[i]]


def _compact(o: Observation) -> Observation:
    o.call = None if o.call is None or isinstance(o.call, tuple) else (o.call.x, o.call.y, o.call.ox, o.call.oy, o.call.tq)
    return o


def _work(cluster):
    w = _WORKER
    if w.get('bb') is not None:
        return [_compact(o) for o in score_cluster_bigbed(w['bb'], cluster, w['options'])]
    return [_compact(o) for o in score_cluster(w['bam'], w['frames'], cluster, w['options'], w['msp'], w['spans'])]


def iter_footprint(bam_path, motifs, options: FootprintOptions, *, cores: int = 1,
                   msp_bam_path=None, want_spans=False, progress=None):
    """Yield ``(motif, observations)`` for every motif, cluster by cluster in
    genomic order. ``motifs`` is a list (sorted here) or a stream already in
    genomic order (e.g. :class:`BigBedMotifs`), consumed lazily with a bounded
    look-ahead. Each cluster's observations are released once yielded, so a
    caller that writes and drops them runs in memory bounded by a few clusters
    per worker. ``call`` is ``(x, y, ox, oy, tq)`` or None."""
    from itertools import islice
    if isinstance(motifs, (list, tuple)):
        clusters = iter(motif_clusters(motifs))
    else:
        clusters = iter_clusters(motifs)
    source = bam_path if isinstance(bam_path, dict) else str(bam_path)
    init = (source, asdict(options), str(msp_bam_path) if msp_bam_path else None, want_spans)

    def emit(cluster, obs):
        grouped = defaultdict(list)
        for o in obs:
            grouped[o.motif_index].append(o)
        for m in sorted(cluster, key=lambda m: m.index):
            yield m, grouped.get(m.index, [])

    done = 0
    if cores <= 1:
        _init_worker(*init)
        try:
            for c in clusters:
                yield from emit(c, _work(c))
                done += 1
                if progress:
                    progress(done, None)
        finally:
            if _WORKER.get('bam') is not None:
                _WORKER['bam'].close()
            _WORKER.clear()
        return
    from fiberhmm.inference.mp_context import _MP_CONTEXT
    with _MP_CONTEXT.Pool(cores, initializer=_init_worker, initargs=init) as pool:
        while True:
            batch = list(islice(clusters, cores * 16))   # bounded look-ahead: Pool.imap would drain a stream
            if not batch:
                break
            for c, obs in zip(batch, pool.imap(_work, batch, chunksize=1)):
                yield from emit(c, obs)
                done += 1
                if progress:
                    progress(done, None)


def run_footprint(bam_path, motifs: Sequence[Motif], options: FootprintOptions, *, cores: int = 1,
                  msp_bam_path=None, want_spans=False, progress=None) -> Dict[int, List[Observation]]:
    """Score every motif; returns motif index -> observations (overlapping molecules).

    Holds every observation; the CLI streams with :func:`iter_footprint` instead."""
    by_motif: Dict[int, List[Observation]] = defaultdict(list)
    for m, obs in iter_footprint(bam_path, motifs, options, cores=cores, msp_bam_path=msp_bam_path,
                                 want_spans=want_spans, progress=progress):
        by_motif[m.index].extend(obs)
    return by_motif


# --------------------------------------------------------------------------- tables

TABLE_COLUMNS = ['#chrom', 'start', 'end', 'strand', 'name', 'layer', 'n_spanning_fibers', 'n_spanning_msps',
                 'n_footprinted', 'fraction_footprinted', 'n_overlapping_nucs', 'n_not_spanning',
                 'footprint_codes', 'fiber_names', 'haplotype']
MOLECULE_COLUMNS = ['#chrom', 'start', 'end', 'strand', 'name', 'motif_id', 'read_name', 'status', 'code', 'reason',
                    'call_start', 'call_end', 'outer_start', 'outer_end', 'call_width', 'llr', 'edge',
                    'nuc_overlap', 'haplotype']


def motif_row(motif: Motif, obs: Sequence[Observation]) -> List[str]:
    spanning = [o for o in obs if o.code != SPAN_NOT]
    elig = sum(o.code & 1 for o in spanning)
    fp = sum(o.code == 3 for o in spanning)
    frac = f'{fp / elig:.6g}' if elig else 'NA'
    return [motif.chrom, str(motif.start), str(motif.end), motif.strand, motif.name, layer_names(motif.label)[0],
            str(len(spanning)), str(elig), str(fp), frac, str(sum(o.nuc_overlap for o in spanning)),
            str(len(obs) - len(spanning)),
            ','.join(str(o.code) for o in spanning) or '.', ','.join(o.read_name for o in spanning) or '.',
            ','.join(o.hp for o in spanning) or '.']


def molecule_row(motif: Motif, o: Observation) -> List[str]:
    c = o.call
    if c is not None and not isinstance(c, tuple):
        c = (c.x, c.y, c.ox, c.oy, c.tq)
    cells = [str(v) for v in c[:4]] + [str(c[1] - c[0]), f'{c[4] / 10:g}' if c[4] >= 0 else '.'] if c else ['.'] * 6
    return [motif.chrom, str(motif.start), str(motif.end), motif.strand, motif.name, motif.motif_id, o.read_name,
            o.status, str(o.code), o.reason or '.', *cells, o.edge or '.', str(int(o.nuc_overlap)), o.hp]


# --------------------------------------------------------------------------- bigBed output

class FootprintBigBedWriter:
    """Per-read BED12 tracks of footprinted motif instances, one bigBed per motif
    set: ``{prefix}_fp_{label}.bb``, with the ``fiberhmm_fp`` autoSQL (blockTq,
    blockCw, blockEf, isDuplicate) and a ``Sample:`` marker so viewers group it
    with the dataset's other FiberHMM tracks. Rows are streamed to temporary BED
    files as reads leave the current region, then sorted and converted."""

    def __init__(self, prefix, chrom_sizes: Dict[str, int], sample_name: Optional[str] = None):
        import tempfile
        self.prefix = str(prefix)
        self.sizes = chrom_sizes
        self.sample = sample_name or os.path.basename(self.prefix)
        out_dir = os.path.dirname(os.path.abspath(self.prefix))
        os.makedirs(out_dir, exist_ok=True)
        self.tmp = tempfile.mkdtemp(prefix='.fp_bb_', dir=out_dir)
        self.handles = {}
        self.open_rows = {}   # (label, read) -> dict(chrom, strand, dup, end, blocks)
        self.chrom = None

    def add(self, motif: Motif, observations):
        if motif.chrom != self.chrom:
            self.flush(); self.chrom = motif.chrom
        else:
            self.flush(before=motif.start)
        for o in observations:
            if o.code != 3:
                continue
            c = o.call if isinstance(o.call, tuple) else (o.call.x, o.call.y, o.call.ox, o.call.oy, o.call.tq)
            row = self.open_rows.setdefault((motif.label, o.read_name), dict(
                chrom=motif.chrom, strand='-' if o.is_reverse else '+', dup=int(o.duplicate), end=o.read_end, blocks=[]))
            row['blocks'].append((motif.start, motif.end, max(0, c[4]), min(255, c[1] - c[0]),
                                  1 if o.edge == 'conservative' else 2))

    def flush(self, before=None):
        done = [k for k, r in self.open_rows.items() if before is None or r['end'] <= before]
        for key in done:
            label, read = key
            r = self.open_rows.pop(key)
            blocks = sorted(set(r['blocks']))
            s0, e0 = blocks[0][0], max(b[1] for b in blocks)
            fh = self.handles.get(label)
            if fh is None:
                fh = self.handles[label] = open(os.path.join(self.tmp, f'{label}.bed'), 'w')
            fh.write('\t'.join(map(str, [
                r['chrom'], s0, e0, read, int(sum(b[2] for b in blocks) / len(blocks)), r['strand'], s0, e0, 0,
                len(blocks), ','.join(str(b[1] - b[0]) for b in blocks), ','.join(str(b[0] - s0) for b in blocks),
                ','.join(str(b[2]) for b in blocks), ','.join(str(b[3]) for b in blocks),
                ','.join(str(b[4]) for b in blocks), r['dup']])) + '\n')

    def close(self, labels=()) -> List[str]:
        import shutil
        import subprocess
        from fiberhmm.cli.extract_tags import bed_to_bigbed
        self.flush()
        for fh in self.handles.values():
            fh.close()
        outputs = []
        try:
            for label in sorted(set(labels) | set(self.handles)):
                bed = os.path.join(self.tmp, f'{label}.bed')
                if not os.path.exists(bed):
                    open(bed, 'w').close()
                srt = bed + '.sorted'
                subprocess.run(['sort', '-k1,1', '-k2,2n', '-o', srt, bed], check=True,
                               env=dict(os.environ, LC_ALL='C'))
                target = f'{self.prefix}_fp_{label}.bb'
                if os.path.getsize(srt) == 0:
                    print(f'fiberhmm-footprint: no footprinted molecules for {label}; {target} not written',
                          file=sys.stderr)
                    continue
                if not bed_to_bigbed(srt, target, self.sizes, extract_type='fp', block_scores=True,
                                     sample_name=self.sample):
                    raise ValueError(f'bedToBigBed failed for {target}')
                outputs.append(target)
        finally:
            shutil.rmtree(self.tmp, ignore_errors=True)
        return outputs


# --------------------------------------------------------------------------- MA annotation output

def _clear_layers(read, layers):
    """Drop the named MA layers (with their AQ bytes and AN names) from a read."""
    from fiberhmm.io.ma_tags import drop_ma_layers
    layers = set(layers)
    drop_ma_layers(read, layers.__contains__)


def annotate_read(read, rows, group_frame) -> int:
    """Append fp_/fpneg_ layers to one read. ``rows``: (layer, seq_start, seq_end, quals, name).

    Intervals are written in the frame the output header's rule gives a ``.``
    group of that name (``MAGroupFrame``), so readers place them as they place
    the read's own calls."""
    from fiberhmm.io.ma_tags import format_an_tag, parse_an_tag
    if read.has_tag('MA'):
        old = str(read.get_tag('MA'))
    elif read.has_tag('Ma'):
        return 0      # fibertools-only read: a new MA would hide its Ma calls
    else:
        old = str(read.query_length)
    parsed = parse_ma_tag(old)
    types = parsed['raw_types']
    L = int(parsed['read_length'])
    aq = list(read.get_tag('AQ')) if read.has_tag('AQ') else []
    count = sum(len(iv) for *_, iv in types)
    names = parse_an_tag(str(read.get_tag('AN'))) if read.has_tag('AN') else [''] * count
    if len(names) != count or len(aq) != sum(len(q) * len(iv) for _, _, q, iv in types):
        raise ValueError(f'{read.query_name}: MA/AQ/AN length mismatch')
    groups = defaultdict(dict)
    for layer, s, e, quals, name in rows:
        if group_frame.is_molecular(layer, '.') and read.is_reverse:
            s, length = flip_interval_frame(s, e - s, L)
        else:
            length = e - s
        groups[layer][(s, length, name)] = quals
    suffix = []
    for layer in sorted(groups):
        items = sorted(groups[layer].items())
        width = len(items[0][1])
        suffix.append(f'{layer}.{"Q" * width}:' + ','.join(f'{s + 1}-{n}' for (s, n, _), _q in items))
        for (_s, _n, name), q in items:
            aq.extend(max(0, min(255, int(v))) for v in q)
            names.append(name)
    if not suffix:
        return 0
    import array
    from fiberhmm.io.ma_tags import replace_read_tags
    # Replace in place so the read's other tags keep their order and bytes.
    replace_read_tags(read, {'MA': (';'.join([old, *suffix]), 'Z'),
                             'AQ': (array.array('B', aq), None) if aq else None,
                             'AN': (format_an_tag(names), 'Z')})
    return sum(len(v) for v in groups.values())


def annotation_row(motif: Motif, o: Observation):
    """``(layer, seq_start, seq_end, quals, motif_id)`` for an eligible observation, else None.

    fp_<label>.QQQ (footprinted): tq of the qualifying call, its conservative
    reference width (bp, saturated 255), and the edge flag (1 = the
    conservative bounds contain the motif, 2 = only the outer bounds do).
    fpneg_<label>.Q (eligible, not footprinted): reason code (1 no call,
    2 partial overlap, 3 containing call wider than the cap, 4 m6A in motif,
    5 every containing call narrower than the floor)."""
    if o.code not in (1, 3) or o.query_span is None:
        return None
    fp_layer, neg_layer = layer_names(motif.label)
    s, e = o.query_span
    if o.code == 3:
        c = o.call if isinstance(o.call, tuple) else (o.call.x, o.call.y, o.call.ox, o.call.oy, o.call.tq)
        x, y, _ox, _oy, tq = c
        return fp_layer, s, e, [tq, min(255, y - x), 1 if o.edge == 'conservative' else 2], motif.motif_id
    return neg_layer, s, e, [REASON_CODE.get(o.reason, 0)], motif.motif_id


def annotation_rows(motifs: Sequence[Motif], by_motif: Dict[int, List[Observation]]):
    """read key -> annotation rows (see :func:`annotation_row`) from stored observations."""
    rows = defaultdict(list)
    for motif in motifs:
        for o in by_motif.get(motif.index, ()):
            row = annotation_row(motif, o)
            if row:
                rows[o.key].append(row)
    return rows


class _ListProvider:
    """Motif lookups for a motif list (see :class:`BigBedMotifs` for streamed motifs)."""

    def __init__(self, motifs):
        self.cluster = {}
        by_chrom = defaultdict(list)
        for ci, cluster in enumerate(motif_clusters(motifs)):
            span = (ci, cluster[0].chrom, min(m.start for m in cluster), max(m.end for m in cluster))
            for m in cluster:
                self.cluster[m.index] = span
                by_chrom[m.chrom].append(m)
        self.index = {c: _MotifIndex(ms) for c, ms in by_chrom.items()}

    def overlapping(self, chrom, a, b):
        index = self.index.get(chrom)
        return index.overlapping(a, b) if index else []

    def cluster_of(self, m):
        return self.cluster[m.index]


class _StreamScorer:
    """Scores each alignment of a coordinate-ordered stream against the motifs it
    overlaps, for the annotated-BAM pass: nothing but the current read, a few
    motif chunks and a few clusters' MSP overrides (``msp_bam``) is held."""

    def __init__(self, provider, options, header, msp_bam_path=None):
        from fiberhmm.io.annotation_frame import MAGroupFrame
        from fiberhmm.qc.states import tag_frames
        self.options, self.msp = options, msp_bam_path
        self.frames = (MAGroupFrame(header), tag_frames(header))
        self.provider = provider if hasattr(provider, 'overlapping') else _ListProvider(provider)
        self.cache = {}

    def _override(self, span):
        if self.msp is None:
            return None
        key, chrom, a, b = span
        if key not in self.cache:
            if len(self.cache) >= 8:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = _msp_index(self.msp, chrom, a, b, self.options)
        return self.cache[key]

    def rows(self, read):
        if read.is_unmapped:
            return []
        motifs = self.provider.overlapping(read.reference_name, int(read.reference_start),
                                           int(read.reference_end or read.reference_start))
        groups = defaultdict(list)
        for m in motifs:
            groups[self.provider.cluster_of(m)].append(m)
        out = []
        for span, ms in groups.items():
            by_index = {m.index: m for m in ms}
            for o in score_read(read, _MotifIndex(ms), self.frames, self.options, self._override(span), want_spans=True):
                row = annotation_row(by_index[o.motif_index], o)
                if row:
                    out.append(row)
        return out


def contract_value(label: str, options: FootprintOptions, bed_path, msp_bam_path) -> dict:
    fp, neg = layer_names(label)
    return dict(layers=[fp, neg], label=label, program='fiberhmm-footprint', motif_bed=str(bed_path),
                msp_source=str(msp_bam_path) if msp_bam_path else 'input BAM (MA msp, Ma, or as/al)',
                parameters=asdict(options),
                intervals='motif coordinates projected onto the molecule (aligned bases inside the motif); '
                          'frame = the header rule for a "." group (as the read\'s own calls)',
                quality_names={fp: ['tq', 'cw', 'ef'], neg: ['reason']},
                tq='LLR x10 of the qualifying FiberHMM TF call (saturated 255)',
                cw='conservative reference width of that call, bp (saturated 255)',
                ef='1 = conservative bounds contain the motif; 2 = only the outer bounds do',
                reason='1 no qualifying call overlaps the motif; 2 a call overlaps but none contains it; '
                       '3 containing call wider than max_width; 4 m6A in the motif (require_unmodified); '
                       '5 every containing call narrower than min_width',
                annotation_names='AN = motif id chrom:start-end:strand')


def write_annotated_bam(bam_path, out_path, motifs, by_motif, options: FootprintOptions,
                        bed_path, msp_bam_path=None, argv=None, labels=None) -> dict:
    """Copy ``bam_path`` to ``out_path`` with this run's fp_/fpneg_ layers.

    ``by_motif`` = stored observations (with query spans), or None to score
    each alignment again as it is copied (constant memory; same rule, same
    result).

    Existing layers (nuc/msp/tf, consensus layers, earlier fp_* layers) are kept
    byte-for-byte; layers with the names written here are replaced in every read."""
    import os
    import pysam
    from fiberhmm import __version__
    from fiberhmm.io.annotation_frame import MAGroupFrame, append_coord_to_ds, pass_through_frame
    from fiberhmm.io.bam_header import append_ma_types, append_pg_record
    from fiberhmm.io.output_files import mkstemp_shared
    from pathlib import Path

    out_path = Path(out_path)
    if out_path.resolve() == Path(bam_path).resolve():
        raise ValueError('--out-bam must differ from the input BAM')
    labels = sorted(labels if labels is not None else {m.label for m in motifs})
    owned = [n for label in labels for n in layer_names(label)]
    rows = annotation_rows(motifs, by_motif) if by_motif is not None else None
    fd, tmp = mkstemp_shared(prefix='.fiberhmm-footprint-', suffix='.bam', dir=str(out_path.parent or '.'))
    os.close(fd)
    annotated = written = 0
    try:
        with pysam.AlignmentFile(str(bam_path)) as bam:
            header = append_ma_types(bam.header, owned)
            hd = header.to_dict()
            keep = []
            for c in hd.get('CO', []):
                if c.startswith(CONTRACT):
                    try:
                        if set(json.loads(c[len(CONTRACT):]).get('layers', [])) & set(owned):
                            continue
                    except ValueError:
                        pass
                keep.append(c)
            keep += [CONTRACT + json.dumps(contract_value(label, options, bed_path, msp_bam_path), sort_keys=True)
                     for label in labels]
            hd['CO'] = keep
            header = pysam.AlignmentHeader.from_dict(hd)
            ds = append_coord_to_ds('Motif-anchored footprint annotations (MA fp_*/fpneg_* layers)',
                                    pass_through_frame(bam.header))
            header = append_pg_record(header, dict(PN='fiberhmm-footprint', VN=__version__, DS=ds,
                                                   CL=' '.join(argv) if argv else 'fiberhmm-footprint'))
            group_frame = MAGroupFrame(header)
            scorer = _StreamScorer(motifs, options, bam.header, msp_bam_path) if rows is None else None
            with pysam.AlignmentFile(tmp, 'wb', header=header) as dest:
                for read in bam.fetch(until_eof=True):
                    if read.has_tag('MA'):
                        present = {t[0] for t in parse_ma_tag(str(read.get_tag('MA')))['raw_types']}
                        if present & set(owned):
                            _clear_layers(read, owned)
                    if scorer is not None:
                        r = scorer.rows(read)
                    else:
                        r = rows.get(read_key(read)) if not read.is_unmapped else None
                    if r:
                        annotated += annotate_read(read, r, group_frame)
                    dest.write(read)
                    written += 1
        pysam.index(tmp)
        os.replace(tmp, out_path)
        os.replace(tmp + '.bai', str(out_path) + '.bai')
    finally:
        for p in (tmp, tmp + '.bai'):
            if os.path.exists(p):
                os.unlink(p)
    return dict(bam=str(out_path), layers=owned, annotations=annotated, written_alignments=written)
