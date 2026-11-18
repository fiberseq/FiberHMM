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
  reference width is ``<= max_width`` (0 = no cap). Bounds are the outer edges
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
REASONS = {1: 'no_call', 2: 'partial_overlap', 3: 'too_wide', 4: 'modified'}
REASON_CODE = {v: k for k, v in REASONS.items()}

FP_PREFIX = 'fp_'
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


def load_motif_bed(path, layer_name: Optional[str] = None) -> List[Motif]:
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
            label = sanitize_label(layer_name if layer_name else name)
            motifs.append(Motif(len(motifs), f[0], a, b, strand, name, label))
    if not motifs:
        raise ValueError('BED contains no motifs')
    return motifs


def motif_clusters(motifs: Sequence[Motif], max_gap: int = 2000, max_span: int = 200_000) -> List[List[Motif]]:
    """Group nearby motifs so each group's reads are fetched and parsed once."""
    out: List[List[Motif]] = []
    for m in sorted(motifs, key=lambda m: (m.chrom, m.start, m.end, m.index)):
        if out:
            last = out[-1]
            c_start = last[0].start
            c_end = max(x.end for x in last)
            if last[0].chrom == m.chrom and m.start - c_end <= max_gap and m.end - c_start <= max_span:
                last.append(m)
                continue
        out.append([m])
    return out


# --------------------------------------------------------------------------- options

@dataclass(frozen=True)
class FootprintOptions:
    min_llr: float = 5.0
    edges: str = 'outer'               # outer | conservative
    outer_bound: str = 'exclusive'     # exclusive | inclusive (of the terminating informative position)
    max_width: int = DEFAULT_MAX_WIDTH  # conservative reference width cap, bp; 0 = off
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
    refmap: _RefMap
    m6a: Optional[np.ndarray] = None


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
                    None if calls is None else proj_calls, nucs, msps, refmap, m6a)


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


def score(mol: Molecule, motif: Motif, options: FootprintOptions, want_span: bool = False) -> Observation:
    m0, m1 = motif.start, motif.end
    nuc = any(a < m1 and b > m0 for a, b in mol.nucs)
    base = dict(motif_index=motif.index, read_name=mol.name, key=mol.key, nuc_overlap=nuc, hp=mol.hp,
                is_reverse=mol.is_reverse, read_length=mol.read_length)
    if not ft_spans(mol.ref_start, mol.ref_end, m0, m1):
        return Observation(code=SPAN_NOT, status=NOT_SPANNING, **base)
    eligible = mol.msps is not None and any(ft_spans(a, b, m0, m1) for a, b in mol.msps)
    calls = [c for c in (mol.calls or ()) if c.tq >= options.min_tq]
    outer = options.edges == 'outer'

    def bounds(c):
        return (c.ox, c.oy) if outer else (c.x, c.y)

    containing = [c for c in calls if bounds(c)[0] <= m0 and bounds(c)[1] >= m1]
    narrow = [c for c in containing if options.max_width == 0 or c.width <= options.max_width]
    chosen, reason = None, ''
    if narrow:
        chosen = min(narrow, key=lambda c: (c.width, -c.tq, c.x))
        if options.require_unmodified and mol.m6a is not None and len(mol.m6a) and \
                np.any((mol.m6a >= m0) & (mol.m6a < m1)):
            reason = 'modified'
    elif containing:
        chosen, reason = min(containing, key=lambda c: (c.width, -c.tq, c.x)), 'too_wide'
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
    return Observation(code=code, status=status, reason='' if fp else reason, call=chosen, edge=edge,
                       query_span=span, **base)


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


def _init_worker(bam_path, options_dict, msp_bam_path, want_spans):
    import pysam
    from fiberhmm.io.annotation_frame import MAGroupFrame
    from fiberhmm.qc.states import tag_frames
    bam = pysam.AlignmentFile(bam_path)
    _WORKER.update(bam=bam, frames=(MAGroupFrame(bam.header), tag_frames(bam.header)),
                   options=FootprintOptions(**options_dict), msp=msp_bam_path, spans=want_spans)


def _compact(o: Observation) -> Observation:
    o.call = None if o.call is None or isinstance(o.call, tuple) else (o.call.x, o.call.y, o.call.ox, o.call.oy, o.call.tq)
    return o


def _work(cluster):
    w = _WORKER
    return [_compact(o) for o in score_cluster(w['bam'], w['frames'], cluster, w['options'], w['msp'], w['spans'])]


def iter_footprint(bam_path, motifs: Sequence[Motif], options: FootprintOptions, *, cores: int = 1,
                   msp_bam_path=None, want_spans=False, progress=None):
    """Yield ``(motif, observations)`` for every motif, cluster by cluster in
    genomic order. Each cluster's observations are released once yielded, so a
    caller that writes and drops them runs in memory bounded by one cluster
    per worker. ``call`` is ``(x, y, ox, oy, tq)`` or None."""
    clusters = motif_clusters(motifs)
    init = (str(bam_path), asdict(options), str(msp_bam_path) if msp_bam_path else None, want_spans)

    def emit(cluster, obs):
        grouped = defaultdict(list)
        for o in obs:
            grouped[o.motif_index].append(o)
        for m in sorted(cluster, key=lambda m: m.index):
            yield m, grouped.get(m.index, [])

    if cores <= 1 or len(clusters) == 1:
        _init_worker(*init)
        try:
            for i, c in enumerate(clusters, 1):
                yield from emit(c, _work(c))
                if progress:
                    progress(i, len(clusters))
        finally:
            _WORKER['bam'].close()
            _WORKER.clear()
        return
    from fiberhmm.inference.mp_context import _MP_CONTEXT
    with _MP_CONTEXT.Pool(cores, initializer=_init_worker, initargs=init) as pool:
        for i, (c, obs) in enumerate(zip(clusters, pool.imap(_work, clusters, chunksize=1)), 1):
            yield from emit(c, obs)
            if progress:
                progress(i, len(clusters))


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
    cells = [str(v) for v in c[:4]] + [str(c[1] - c[0]), f'{c[4] / 10:g}'] if c else ['.'] * 6
    return [motif.chrom, str(motif.start), str(motif.end), motif.strand, motif.name, motif.motif_id, o.read_name,
            o.status, str(o.code), o.reason or '.', *cells, o.edge or '.', str(int(o.nuc_overlap)), o.hp]


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
    2 partial overlap, 3 containing call wider than the cap, 4 m6A in motif)."""
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


class _StreamScorer:
    """Scores each alignment of a coordinate-ordered stream against the motifs it
    overlaps, for the annotated-BAM pass: nothing but the current read and a few
    clusters' MSP overrides (``msp_bam``) is held."""

    def __init__(self, motifs, options, header, msp_bam_path=None):
        from fiberhmm.io.annotation_frame import MAGroupFrame
        from fiberhmm.qc.states import tag_frames
        self.options, self.msp = options, msp_bam_path
        self.frames = (MAGroupFrame(header), tag_frames(header))
        self.by_motif = {}
        by_chrom = defaultdict(list)
        for ci, cluster in enumerate(motif_clusters(motifs)):
            for m in cluster:
                self.by_motif[m.index] = (ci, cluster)
                by_chrom[m.chrom].append(m)
        self.index = {c: _MotifIndex(ms) for c, ms in by_chrom.items()}
        self.cache = {}

    def _override(self, ci, cluster):
        if self.msp is None:
            return None
        if ci not in self.cache:
            if len(self.cache) >= 8:
                self.cache.pop(next(iter(self.cache)))
            self.cache[ci] = _msp_index(self.msp, cluster[0].chrom, min(m.start for m in cluster),
                                        max(m.end for m in cluster), self.options)
        return self.cache[ci]

    def rows(self, read):
        index = self.index.get(read.reference_name) if not read.is_unmapped else None
        if index is None:
            return []
        groups = defaultdict(list)
        for m in index.overlapping(int(read.reference_start), int(read.reference_end or read.reference_start)):
            groups[self.by_motif[m.index][0]].append(m)
        out = []
        for ci, ms in groups.items():
            sub = _MotifIndex(ms)
            for o in score_read(read, sub, self.frames, self.options, self._override(ci, self.by_motif[ms[0].index][1]),
                                want_spans=True):
                row = annotation_row(next(m for m in ms if m.index == o.motif_index), o)
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
                       '3 containing call wider than max_width; 4 m6A in the motif (require_unmodified)',
                annotation_names='AN = motif id chrom:start-end:strand')


def write_annotated_bam(bam_path, out_path, motifs: Sequence[Motif], by_motif, options: FootprintOptions,
                        bed_path, msp_bam_path=None, argv=None) -> dict:
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
    labels = sorted({m.label for m in motifs})
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
