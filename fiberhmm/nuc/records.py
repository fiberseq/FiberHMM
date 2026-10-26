"""Full molecule records for the nucleosome module, in molecular coordinates.

A :class:`NucMolecule` is one BAM record decoded into what every ``nuc``
statistic needs: the footprint calls (nucleosomes, TF footprints, MSPs), the
marks and informative sites from the encoder, no-call blocks, and the
alignment blocks for projection. Everything is in **molecular coordinates**
(0-based query bp in the original molecule's orientation). Marks are flipped
once here for reverse reads; nothing downstream flips again.

Lengths, dyads and spacing are measured in molecular bp. Reference positions
are used only for display, pooling and anchors (:meth:`NucMolecule.mol_to_ref`,
:meth:`NucMolecule.ref_to_mol`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import numpy as np

from fiberhmm.nuc.params import DEFAULT_PARAMS, NucParams

N_CTX = 4096
UNMETH_OFFSET = 4097

_EMPTY = np.zeros((0, 2), dtype=np.int64)


def _as_intervals(pairs) -> np.ndarray:
    if pairs is None or len(pairs) == 0:
        return _EMPTY.copy()
    arr = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    arr = arr[arr[:, 1] > arr[:, 0]]
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


@dataclass
class NucMolecule:
    """One record in molecular coordinates.

    ``nucs``, ``tfs``, ``msps`` and ``no_call`` are ``(n, 2)`` int arrays of
    half-open ``[start, end)`` intervals, sorted by start. ``hits`` and
    ``informative`` are boolean arrays of length ``length``.
    """

    name: str
    length: int
    is_reverse: bool
    nucs: np.ndarray
    tfs: np.ndarray
    msps: np.ndarray
    hits: np.ndarray
    informative: np.ndarray
    no_call: np.ndarray = field(default_factory=lambda: _EMPTY.copy())
    nuc_edges: Optional[np.ndarray] = None   # (n, 2) edge sharpness bytes (left, right)
    chrom: Optional[str] = None
    ref_start: int = -1
    ref_end: int = -1
    flag: int = 0
    hp: Optional[int] = None
    blocks: Optional[np.ndarray] = None      # (b, 3): seq query start, ref start, length
    source: str = 'MA'
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.nucs = _as_intervals(self.nucs)
        self.tfs = _as_intervals(self.tfs)
        self.msps = _as_intervals(self.msps)
        self.no_call = _as_intervals(self.no_call)
        self.hits = np.asarray(self.hits, dtype=bool)
        self.informative = np.asarray(self.informative, dtype=bool)
        if self.hits.shape != (self.length,) or self.informative.shape != (self.length,):
            raise ValueError('hits/informative must have the molecule length')
        self.hits = self.hits & self.informative
        self._cum_hit = None
        self._cum_inf = None

    # --- counting ----------------------------------------------------------
    def _cums(self):
        if self._cum_hit is None:
            self._cum_hit = np.concatenate([[0], np.cumsum(self.hits, dtype=np.int64)])
            self._cum_inf = np.concatenate([[0], np.cumsum(self.informative, dtype=np.int64)])
        return self._cum_hit, self._cum_inf

    def counts(self, start, end):
        """(marks, informative sites) in ``[start, end)``; vectorised."""
        ch, ci = self._cums()
        s = np.clip(np.asarray(start), 0, self.length)
        e = np.clip(np.asarray(end), 0, self.length)
        e = np.maximum(e, s)
        return ch[e] - ch[s], ci[e] - ci[s]

    @property
    def n_hits(self) -> int:
        return int(self._cums()[0][-1])

    @property
    def n_informative(self) -> int:
        return int(self._cums()[1][-1])

    @property
    def is_supplementary(self) -> bool:
        return bool(self.flag & 0x800)

    @property
    def is_secondary(self) -> bool:
        return bool(self.flag & 0x100)

    # --- projection ----------------------------------------------------------
    def _seq_pos(self, mol_pos):
        mol_pos = np.asarray(mol_pos, dtype=np.int64)
        return (self.length - 1 - mol_pos) if self.is_reverse else mol_pos

    def mol_to_ref(self, mol_pos):
        """Reference position of each molecular position, -1 where unaligned."""
        pos = np.atleast_1d(self._seq_pos(mol_pos))
        out = np.full(pos.shape, -1, dtype=np.int64)
        if self.blocks is None or len(self.blocks) == 0:
            return out
        q0, r0, n = self.blocks[:, 0], self.blocks[:, 1], self.blocks[:, 2]
        idx = np.searchsorted(q0, pos, side='right') - 1
        ok = idx >= 0
        idx_c = np.clip(idx, 0, len(q0) - 1)
        inside = ok & (pos < q0[idx_c] + n[idx_c])
        out[inside] = r0[idx_c[inside]] + (pos[inside] - q0[idx_c[inside]])
        return out

    def ref_to_mol(self, ref_pos):
        """Molecular position of each reference position.

        A position inside a deletion maps to the next aligned base; positions
        outside the alignment map to -1.
        """
        ref = np.atleast_1d(np.asarray(ref_pos, dtype=np.int64))
        out = np.full(ref.shape, -1, dtype=np.int64)
        if self.blocks is None or len(self.blocks) == 0:
            return out
        q0, r0, n = self.blocks[:, 0], self.blocks[:, 1], self.blocks[:, 2]
        idx = np.searchsorted(r0, ref, side='right') - 1
        for k, (i, r) in enumerate(zip(idx, ref)):
            if i < 0:
                continue
            if r < r0[i] + n[i]:
                seq = q0[i] + (r - r0[i])
            elif i + 1 < len(r0):
                seq = q0[i + 1]
            else:
                continue
            out[k] = (self.length - 1 - seq) if self.is_reverse else seq
        return out

    def mol_interval_to_ref(self, start: int, end: int) -> Optional[Tuple[int, int]]:
        """Reference envelope of molecular ``[start, end)``; None if unaligned."""
        if end <= start:
            return None
        if self.is_reverse:
            s_seq, e_seq = self.length - end, self.length - start
        else:
            s_seq, e_seq = start, end
        if self.blocks is None or len(self.blocks) == 0:
            return None
        q0, r0, n = self.blocks[:, 0], self.blocks[:, 1], self.blocks[:, 2]
        lo = np.maximum(q0, s_seq)
        hi = np.minimum(q0 + n, e_seq)
        ok = hi > lo
        if not ok.any():
            return None
        rs = (r0 + (lo - q0))[ok]
        re_ = (r0 + (hi - q0))[ok]
        return int(rs.min()), int(re_.max())

    def ref_interval_to_mol(self, start: int, end: int) -> Optional[Tuple[int, int, int]]:
        """Molecular envelope of reference ``[start, end)``.

        Returns ``(mol_start, mol_end, covered_ref_bp)``; None when the
        molecule's alignment does not touch the interval.
        """
        if self.blocks is None or len(self.blocks) == 0 or end <= start:
            return None
        q0, r0, n = self.blocks[:, 0], self.blocks[:, 1], self.blocks[:, 2]
        lo = np.maximum(r0, start)
        hi = np.minimum(r0 + n, end)
        ok = hi > lo
        if not ok.any():
            return None
        qs = (q0 + (lo - r0))[ok]
        qe = (q0 + (hi - r0))[ok]
        s_seq, e_seq = int(qs.min()), int(qe.max())
        covered = int((hi - lo)[ok].sum())
        if self.is_reverse:
            return self.length - e_seq, self.length - s_seq, covered
        return s_seq, e_seq, covered

    def record_key(self) -> Tuple[str, int, Optional[str], int]:
        return (self.name, self.flag, self.chrom, self.ref_start)


# ---------------------------------------------------------------------------
# Decoding from pysam
# ---------------------------------------------------------------------------

_ENZYME_RE = re.compile(r'--enzyme[ =](\S+)')
_SEQ_RE = re.compile(r'--seq[ =](\S+)')


def header_chemistry(header) -> dict:
    """``{'mode', 'enzyme', 'platform'}`` from a BAM header (declared, else
    the fiberhmm-call @PG record); values may be None."""
    from fiberhmm.io.bam_header import declared_chemistries, infer_legacy_chemistry
    out = {'mode': None, 'enzyme': None, 'platform': None}
    declared = declared_chemistries(header)
    chem = declared[-1] if declared else infer_legacy_chemistry(header)
    if chem:
        out['mode'] = chem.get('mode') if chem.get('mode') not in (None, 'custom') else None
        out['enzyme'] = chem.get('enzyme') if chem.get('enzyme') not in (None, 'custom') else None
        out['platform'] = chem.get('platform') if chem.get('platform') not in (None, 'unknown') else None
    hd = header.to_dict() if hasattr(header, 'to_dict') else dict(header or {})
    for pg in reversed(hd.get('PG', [])):
        if 'fiberhmm-call' not in str(pg.get('PN') or pg.get('ID') or ''):
            continue
        cl = str(pg.get('CL', ''))
        m = _ENZYME_RE.search(cl)
        if m and not out['enzyme']:
            out['enzyme'] = m.group(1).lower()
        m = _SEQ_RE.search(cl)
        if m and not out['platform']:
            out['platform'] = m.group(1).lower()
        break
    if out['mode'] is None and out['enzyme']:
        try:
            from fiberhmm.models import get_observation_mode
            out['mode'] = get_observation_mode(out['enzyme'], out['platform'],
                                               warn_missing_seq=False)
        except Exception:
            pass
    return out


class DecodeContext:
    """Header-level decoding state, built once per BAM."""

    def __init__(self, header, params: NucParams = DEFAULT_PARAMS):
        from fiberhmm.io.annotation_frame import legacy_frame_of, ma_group_frame_of
        from fiberhmm.models import resolve_prob_threshold
        self.header = header
        self.params = params
        chem = header_chemistry(header)
        self.chemistry = chem
        self.mode = params.mode or chem['mode'] or 'pacbio-fiber'
        self.mark_threshold = resolve_prob_threshold(
            params.mark_threshold, chem['enzyme'], chem['platform'])
        self.ma_frame = ma_group_frame_of(header)
        self.legacy_frame = legacy_frame_of(header)


def _aligned_blocks(read) -> Optional[np.ndarray]:
    cig = read.cigartuples
    if not cig or read.is_unmapped:
        return None
    q = 0
    r = read.reference_start
    rows = []
    for op, n in cig:
        if op in (0, 7, 8):
            rows.append((q, r, n))
            q += n
            r += n
        elif op in (1, 4):
            q += n
        elif op in (2, 3):
            r += n
    if not rows:
        return None
    return np.asarray(rows, dtype=np.int64)


def _footprints_from_ma(read, ctx: DecodeContext, read_length: int):
    from fiberhmm.io.ma_tags import annotation_tags, parse_aq_array, parse_ma_tag
    tags = annotation_tags(read)
    if tags is None:
        return None
    ma, aq, _an, source = tags
    parsed = parse_ma_tag(str(ma))
    if not any(t[0] in ('nuc', 'msp', 'tf') and t[3] for t in parsed['raw_types']):
        return None          # no footprint groups: the calls (if any) are in ns/nl/as/al
    if parsed['read_length'] != read_length:
        return 'length_mismatch'
    specs = [t[2] for t in parsed['raw_types']]
    counts = [len(t[3]) for t in parsed['raw_types']]
    try:
        quals = parse_aq_array(aq, specs, counts) if aq is not None else [[] for _ in range(sum(counts))]
    except Exception:
        quals = [[] for _ in range(sum(counts))]
    out = {'nuc': [], 'tf': [], 'msp': [], 'nuc_edges': []}
    k = 0
    for name, strand, spec, intervals in parsed['raw_types']:
        molecular = source == 'Ma' or ctx.ma_frame.is_molecular(name, strand)
        flip = read.is_reverse and not molecular
        for (s, length) in intervals:
            q = quals[k] if k < len(quals) else []
            k += 1
            if name not in out:
                continue
            a, b = int(s), int(s) + int(length)
            el = q[1] if len(q) >= 3 else -1
            er = q[2] if len(q) >= 3 else -1
            if flip:
                a, b = read_length - b, read_length - a
                el, er = er, el
            out[name].append((a, b))
            if name == 'nuc':
                out['nuc_edges'].append((el, er))
    out['source'] = source
    return out


def _footprints_from_legacy(read, ctx: DecodeContext, read_length: int):
    def pairs(st, ln):
        if not (read.has_tag(st) and read.has_tag(ln)):
            return []
        return [(int(a), int(a) + int(b)) for a, b in zip(read.get_tag(st), read.get_tag(ln))]
    nucs = pairs('ns', 'nl')
    msps = pairs('as', 'al')
    if not nucs and not msps:
        return None
    frame = ctx.legacy_frame.for_read(read)
    if frame is None:
        return 'frame_undecided'
    if read.is_reverse and frame != 'molecular':
        nucs = [(read_length - b, read_length - a) for a, b in nucs]
        msps = [(read_length - b, read_length - a) for a, b in msps]
    return {'nuc': nucs, 'tf': [], 'msp': msps, 'nuc_edges': [], 'source': 'legacy'}


def decode_record(read, ctx: DecodeContext) -> Tuple[Optional[NucMolecule], str]:
    """Decode one pysam record; returns ``(molecule, reason)``.

    ``molecule`` is None when the record cannot be used, with ``reason`` one
    of ``unmapped``, ``secondary``, ``no_sequence``, ``no_calls``,
    ``length_mismatch`` (footprint tags describe a longer molecule than the
    stored SEQ: a hard-clipped record), ``frame_undecided`` or
    ``no_evidence``.
    """
    from fiberhmm.core.bam_reader import encode_from_query_sequence
    from fiberhmm.inference.engine import CHIMERA_SKIP, _extract_fiber_read_from_pysam

    if read.is_secondary:
        return None, 'secondary'
    seq = read.query_sequence
    if not seq:
        return None, 'no_sequence'
    L = len(seq)
    fp = _footprints_from_ma(read, ctx, L)
    if fp is None:
        fp = _footprints_from_legacy(read, ctx, L)
    if fp is None:
        return None, 'no_calls'
    if isinstance(fp, str):
        return None, fp

    fr = _extract_fiber_read_from_pysam(read, ctx.mode, ctx.mark_threshold)
    if fr is None or fr is CHIMERA_SKIP:
        return None, 'no_evidence'
    mode = ctx.mode
    strand = fr.get('_daf_strand') or '.'
    codes = encode_from_query_sequence(
        fr['query_sequence'], fr['m6a_query_positions'], ctx.params.edge_trim,
        mode=mode, strand=strand, context_size=3,
        is_reverse=bool(read.is_reverse),
        unknown_positions=fr.get('unknown_query_positions'))
    if len(codes) != L:
        return None, 'no_evidence'
    hits = codes < N_CTX
    inf = hits | ((codes >= UNMETH_OFFSET) & (codes < UNMETH_OFFSET + N_CTX))
    no_call = [tuple(b) for b in (fr.get('no_call_blocks') or [])]
    if read.is_reverse:
        hits = hits[::-1].copy()
        inf = inf[::-1].copy()
        no_call = [(L - b, L - a) for a, b in no_call]

    edges = np.asarray(fp['nuc_edges'], dtype=np.int64).reshape(-1, 2) if fp['nuc_edges'] else None
    nucs = np.asarray(fp['nuc'], dtype=np.int64).reshape(-1, 2) if fp['nuc'] else _EMPTY
    if edges is not None and len(edges) == len(nucs) and len(nucs):
        order = np.lexsort((nucs[:, 1], nucs[:, 0]))
        edges = edges[order]
    else:
        edges = None
    hp = int(read.get_tag('HP')) if read.has_tag('HP') else None
    mol = NucMolecule(
        name=read.query_name, length=L, is_reverse=bool(read.is_reverse),
        nucs=nucs, tfs=fp['tf'], msps=fp['msp'], hits=hits, informative=inf,
        no_call=no_call, nuc_edges=edges,
        chrom=None if read.is_unmapped else read.reference_name,
        ref_start=-1 if read.is_unmapped else int(read.reference_start),
        ref_end=-1 if read.is_unmapped else int(read.reference_end or -1),
        flag=int(read.flag), hp=hp, blocks=_aligned_blocks(read),
        source=fp['source'],
    )
    return mol, 'ok'


def iter_molecules(bam_path: str, region: Optional[str] = None,
                   params: NucParams = DEFAULT_PARAMS, *,
                   contig: Optional[str] = None, start: Optional[int] = None,
                   end: Optional[int] = None, primary_only: bool = True,
                   stats: Optional[dict] = None,
                   max_records: Optional[int] = None) -> Iterator[NucMolecule]:
    """Decode the records of a BAM (or a region of it) into molecules.

    ``primary_only`` skips supplementary records: molecule-wide outputs come
    from one record per molecule. ``stats`` collects skip reasons.
    """
    import pysam
    with pysam.AlignmentFile(bam_path, 'rb', check_sq=False) as bam:
        ctx = DecodeContext(bam.header, params)
        if region is not None:
            it = bam.fetch(region=region)
        elif contig is not None:
            it = bam.fetch(contig, start, end)
        else:
            it = bam.fetch(until_eof=True)
        n = 0
        for read in it:
            if primary_only and read.is_supplementary:
                if stats is not None:
                    stats['supplementary'] = stats.get('supplementary', 0) + 1
                continue
            mol, reason = decode_record(read, ctx)
            if stats is not None:
                stats[reason] = stats.get(reason, 0) + 1
            if mol is None:
                continue
            yield mol
            n += 1
            if max_records is not None and n >= max_records:
                return


def internal_footprints(mol: NucMolecule) -> np.ndarray:
    """Nucleosome footprints without the first and last on the molecule."""
    return mol.nucs[1:-1] if len(mol.nucs) > 2 else _EMPTY.copy()


def make_molecule(length: int, nucs=(), tfs=(), msps=(), hits=None,
                  informative=None, **kw) -> NucMolecule:
    """Synthetic molecule for tests: ``hits`` may be positions or a bool mask;
    ``informative`` defaults to every base."""
    inf = np.ones(length, dtype=bool) if informative is None else np.asarray(informative)
    if inf.dtype != bool:
        mask = np.zeros(length, dtype=bool)
        mask[np.asarray(inf, dtype=np.int64)] = True
        inf = mask
    if hits is None:
        h = np.zeros(length, dtype=bool)
    else:
        h = np.asarray(hits)
        if h.dtype != bool:
            mask = np.zeros(length, dtype=bool)
            mask[np.asarray(h, dtype=np.int64)] = True
            h = mask
    kw.setdefault('name', 'mol')
    return NucMolecule(length=length, is_reverse=kw.pop('is_reverse', False),
                       nucs=nucs, tfs=tfs, msps=msps, hits=h, informative=inf, **kw)


def molecules_list(it) -> List[NucMolecule]:
    return list(it)
