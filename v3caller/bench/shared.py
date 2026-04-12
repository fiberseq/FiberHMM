"""Shared helpers for the nucleosome caller benchmark.

Unified reference-frame extractor (wraps DAFExtractor + get_reference_sequence
so we don't need external FASTAs), metrics, BAM I/O.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pysam

# Make sibling modules importable
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # phase0
from enzyme_extractors import DAFExtractor  # noqa: E402


# -------------------------------------------------------------------
# Per-read bundle: the arrays every algorithm works on.
# -------------------------------------------------------------------

@dataclass
class ReadBundle:
    read_id: str
    L: int               # length in ref frame
    opp: np.ndarray      # int8 [L], 1 at opportunity positions
    hit: np.ndarray      # int8 [L], 1 at modified (hit) positions
    baseline: float      # overall hit rate = sum(hit)/sum(opp)
    qlen: int            # query length for MSP complement
    ref_to_q: np.ndarray # int32 [L], query pos or -1


def _build_ref_to_query_map(read, L):
    m = np.full(L, -1, dtype=np.int32)
    for qp, rp in read.get_aligned_pairs(matches_only=True):
        if rp is None:
            continue
        rr = rp - read.reference_start
        if 0 <= rr < L:
            m[rr] = qp
    return m


def bundle_from_read(read, extractor: DAFExtractor,
                     min_read_rate: float = 0.05,
                     min_opp: int = 50,
                     min_length: int = 500) -> Optional[ReadBundle]:
    """Build ReadBundle from a BAM record, reconstructing ref_seq from MD tags.

    Returns None if the read should be skipped.
    """
    if read.is_unmapped or read.is_secondary or read.is_supplementary:
        return None
    if read.reference_end is None or read.reference_start is None:
        return None
    L = read.reference_end - read.reference_start
    if L < min_length:
        return None
    # Reconstruct reference sequence from MD tag — avoids FASTA juggling.
    try:
        ref_seq = read.get_reference_sequence()
    except Exception:
        return None
    if ref_seq is None or len(ref_seq) < L:
        return None
    ref_seq = ref_seq.upper()

    opp, hit = extractor.read_to_arrays(read, ref_seq)
    n_opp = int(opp.sum())
    if n_opp < min_opp:
        return None
    baseline = float(hit.sum()) / n_opp
    if baseline < min_read_rate:
        return None

    ref_to_q = _build_ref_to_query_map(read, L)

    return ReadBundle(
        read_id=read.query_name or '',
        L=L,
        opp=opp,
        hit=hit,
        baseline=baseline,
        qlen=read.query_length or 0,
        ref_to_q=ref_to_q,
    )


# -------------------------------------------------------------------
# Ref-frame windowed rate (reused from caller_v7)
# -------------------------------------------------------------------

def windowed_rate(opp, hit, W):
    opp_cum = np.concatenate([[0], np.cumsum(opp, dtype=np.int32)])
    hit_cum = np.concatenate([[0], np.cumsum(hit, dtype=np.int32)])
    opp_win = opp_cum[W:] - opp_cum[:-W]
    hit_win = hit_cum[W:] - hit_cum[:-W]
    valid = opp_win >= 5
    rate = np.zeros_like(opp_win, dtype=np.float32)
    rate[valid] = hit_win[valid] / opp_win[valid]
    return rate, valid


# -------------------------------------------------------------------
# Atom filtering, structural merge, coord conversion
# -------------------------------------------------------------------

def clip_and_filter_atoms(atoms, L, min_footprint=80):
    out = []
    for s, e in atoms:
        s2 = max(0, int(s))
        e2 = min(L, int(e))
        if e2 - s2 >= min_footprint:
            out.append((s2, e2))
    return out


def structural_merge(atoms):
    """Merge atoms whose ref intervals touch or overlap."""
    if not atoms:
        return []
    atoms = sorted(atoms, key=lambda x: x[0])
    out = [atoms[0]]
    for s, e in atoms[1:]:
        ls, le = out[-1]
        if s <= le:
            out[-1] = (ls, max(le, e))
        else:
            out.append((s, e))
    return out


def atoms_to_query_intervals(atoms, bundle: ReadBundle):
    """Convert [s, e) ref-frame atoms to (qs, qe) query intervals."""
    result = []
    for s, e in atoms:
        if e <= s:
            continue
        sub = bundle.ref_to_q[s:e]
        mask = sub >= 0
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        qs = int(sub[idx[0]])
        qe = int(sub[idx[-1]]) + 1
        if qe > qs:
            result.append((qs, qe))
    # Coalesce overlapping query intervals
    result.sort(key=lambda x: x[0])
    merged = []
    for qs, qe in result:
        if merged and qs <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], qe))
        else:
            merged.append((qs, qe))
    return merged


def query_complement(intervals, qlen):
    """Accessible (MSP) regions in query coords."""
    if qlen <= 0:
        return []
    out = []
    cur = 0
    for qs, qe in intervals:
        if qs > cur:
            out.append((cur, qs))
        cur = max(cur, qe)
    if cur < qlen:
        out.append((cur, qlen))
    return out


# -------------------------------------------------------------------
# BAM writing with ns/nl/as/al tags
# -------------------------------------------------------------------

import array as _array

STALE_TAGS = ('ns', 'nl', 'nq', 'as', 'al', 'aq')


def set_array_tag(read, tag, values):
    if values:
        read.set_tag(tag, _array.array('I', [max(0, int(v)) for v in values]))


def clear_stale(read):
    for t in STALE_TAGS:
        if read.has_tag(t):
            read.set_tag(t, None)


def write_calls_to_bam(in_bam_path, out_bam_path, calls_by_id, max_reads=None):
    """Stream in_bam -> out_bam, writing ns/nl/as/al for reads in calls_by_id.

    calls_by_id maps read_id -> dict with keys 'ns', 'nl', 'as', 'al'.
    Reads not in calls_by_id are written unchanged (with stale tags cleared).
    """
    os.makedirs(os.path.dirname(out_bam_path), exist_ok=True)
    src = pysam.AlignmentFile(in_bam_path, 'rb', check_sq=False)
    dst = pysam.AlignmentFile(out_bam_path, 'wb', template=src)
    n = 0
    for read in src.fetch(until_eof=True):
        clear_stale(read)
        call = calls_by_id.get(read.query_name)
        if call is not None:
            set_array_tag(read, 'ns', call['ns'])
            set_array_tag(read, 'nl', call['nl'])
            set_array_tag(read, 'as', call['as'])
            set_array_tag(read, 'al', call['al'])
        dst.write(read)
        n += 1
        if max_reads and n >= max_reads:
            break
    dst.close()
    src.close()
    return n


# -------------------------------------------------------------------
# Metrics aggregation
# -------------------------------------------------------------------

@dataclass
class BenchResult:
    caller: str
    dataset: str
    n_reads_total: int = 0
    n_reads_called: int = 0
    runtime_s: float = 0.0
    nucs_per_read: list = field(default_factory=list)
    nl: list = field(default_factory=list)  # nucleosome lengths
    al: list = field(default_factory=list)  # accessible lengths
    spacing: list = field(default_factory=list)  # center-to-center

    def add_call(self, ns, nl, al):
        self.n_reads_called += 1
        self.nucs_per_read.append(len(ns))
        self.nl.extend(nl)
        self.al.extend(al)
        if len(ns) >= 2:
            centers = [s + l // 2 for s, l in zip(ns, nl)]
            for a, b in zip(centers[:-1], centers[1:]):
                self.spacing.append(b - a)

    def summary(self) -> dict:
        nl = np.asarray(self.nl, dtype=np.int32)
        al = np.asarray(self.al, dtype=np.int32)
        npr = np.asarray(self.nucs_per_read, dtype=np.int32)
        sp = np.asarray(self.spacing, dtype=np.int32)

        def _pct(x, lo, hi=None):
            if len(nl) == 0:
                return 0.0
            if hi is None:
                return 100.0 * float((nl >= lo).mean())
            return 100.0 * float(((nl >= lo) & (nl < hi)).mean())

        return {
            'caller': self.caller,
            'dataset': self.dataset,
            'n_reads_total': self.n_reads_total,
            'n_reads_called': self.n_reads_called,
            'call_rate': (self.n_reads_called / max(1, self.n_reads_total)),
            'runtime_s': round(self.runtime_s, 2),
            'mean_nucs_per_read': float(npr.mean()) if len(npr) else 0.0,
            'median_nucs_per_read': float(np.median(npr)) if len(npr) else 0.0,
            'n_footprints': int(len(nl)),
            'median_footprint': float(np.median(nl)) if len(nl) else 0.0,
            'mean_footprint': float(nl.mean()) if len(nl) else 0.0,
            'pct_sub_100': _pct(nl, 0, 100),
            'pct_mono_100_220': _pct(nl, 100, 220),
            'pct_di_220_360': _pct(nl, 220, 360),
            'pct_over_360': _pct(nl, 360),
            'median_accessible': float(np.median(al)) if len(al) else 0.0,
            'median_spacing': float(np.median(sp)) if len(sp) else 0.0,
            'mean_spacing': float(sp.mean()) if len(sp) else 0.0,
        }
