"""M4: pooled overlap and distance enrichment between two hit sets in one frame (HUB section 7.6).

For hit sets A and B:

* statistic: the fraction of A hits with a B hit within d, for each d of a grid (d = 0 is overlap), and the signed
  distance from each A hit to its nearest B (strand-aware when A has strand);
* territory: callable bases scanned for both sets. A hit is used only if its whole interval lies inside one
  territory interval; B hits outside are dropped. An A hit with no B within d counts as a miss only when its
  d-window ``[a0 - d, a1 + d)`` lies inside one territory interval; otherwise it is censored at that d;
* null: cluster-preserving matched relocation. A hits are grouped by single linkage (gap <= 1 kb); each cluster
  moves rigidly to a uniform admissible start on its contig, where its whole span lies inside one maximal run of
  territory and its own stratum. B is fixed. By default relocated clusters are also kept more than the linkage gap
  apart, the property linkage guarantees for the observed clusters (without it the null is conservative when A
  clusters are dense). Clusters that conflict are redrawn, one of each conflicting pair per round: this is close to
  uniform over separated configurations, not exactly uniform, and exact in the limit of sparse clusters;
* test: the maximum over d of |z_d| against the same maximum under the null; per-d p-values are descriptive;
* no silent fallback: more than ``max_unplaced`` clusters without an admissible placement refuses the analysis.

Group contrasts (HUB section 7.7): the difference in log O/E between groups of B for the same A, with a bootstrap
over A clusters. One relocation draw serves every group.

Coordinates are 0-based half-open. A zero-width hit (start == end) is treated as ``[p, p + 1)``.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

DISTANCES = (0, 100, 250, 500, 1000, 2500, 5000)
HIST_RANGE = 5000
HIST_STEP = 250
_BIG = np.int64(1) << np.int64(60)
Z95 = 1.959964
_CHUNK_ELEMENTS = 2_000_000


# ----------------------------------------------------------------------------------------------- hit sets

@dataclass
class HitSet:
    """One feature type's hits in one frame."""
    chrom: np.ndarray
    start: np.ndarray
    end: np.ndarray
    strand: Optional[np.ndarray] = None
    ids: Optional[np.ndarray] = None

    def __post_init__(self):
        self.chrom = np.asarray(self.chrom, dtype=str).reshape(-1)
        self.start = np.asarray(self.start, dtype=np.int64).reshape(-1)
        self.end = np.asarray(self.end, dtype=np.int64).reshape(-1)
        n = len(self.chrom)
        if len(self.start) != n or len(self.end) != n:
            raise ValueError('HitSet: chrom, start and end differ in length')
        if np.any(self.end < self.start):
            raise ValueError('HitSet: end < start')
        if self.strand is not None:
            self.strand = np.asarray(self.strand, dtype=str).reshape(-1)
            if len(self.strand) != n:
                raise ValueError('HitSet: strand differs in length')
        if self.ids is not None:
            self.ids = np.asarray(self.ids).reshape(-1)
            if len(self.ids) != n:
                raise ValueError('HitSet: ids differ in length')

    def __len__(self):
        return len(self.chrom)

    @classmethod
    def from_rows(cls, rows) -> 'HitSet':
        """Rows of dicts with chrom/start/end and optional strand and hit_id (values may be strings)."""
        rows = list(rows)
        chrom = [str(r['chrom']) for r in rows]
        start = [int(r['start']) for r in rows]
        end = [int(r['end']) for r in rows]
        strand = None
        if any('strand' in r for r in rows):
            strand = [(str(r.get('strand') or '.').strip() or '.') for r in rows]
        ids = None
        if any('hit_id' in r for r in rows):
            ids = np.array([str(r.get('hit_id', '')) for r in rows], dtype=object)
        return cls(np.array(chrom, dtype=str), np.array(start, dtype=np.int64), np.array(end, dtype=np.int64),
                   None if strand is None else np.array(strand, dtype=str), ids)


# ----------------------------------------------------------------------------------------------- intervals

def _chrom_key(c):
    return str(c)


def _label_key(x):
    if isinstance(x, (bool, np.bool_)):
        return (1, str(x))
    if isinstance(x, (int, float, np.integer, np.floating)):
        return (0, float(x), '')
    return (1, str(x))


def _merge(intervals):
    """{chrom: (starts, ends)} with overlapping and abutting intervals merged; empty intervals dropped."""
    by = {}
    for row in intervals:
        c, s, e = str(row[0]), int(row[1]), int(row[2])
        if e > s:
            by.setdefault(c, []).append((s, e))
    out = {}
    for c in sorted(by, key=_chrom_key):
        iv = sorted(by[c])
        ms, me = [], []
        for s, e in iv:
            if ms and s <= me[-1]:
                if e > me[-1]:
                    me[-1] = e
            else:
                ms.append(s)
                me.append(e)
        out[c] = (np.array(ms, np.int64), np.array(me, np.int64))
    return out


def _as_list(merged):
    return [(c, int(s), int(e)) for c in merged for s, e in zip(*merged[c])]


def intersect_territories(*territories) -> list:
    """Bases present in every territory (e.g. callable.bed and both scanned territories), merged and sorted."""
    if not territories:
        return []
    cur = _merge(territories[0])
    for other in territories[1:]:
        oth = _merge(other)
        nxt = {}
        for c in cur:
            if c not in oth:
                continue
            s1, e1 = cur[c]
            s2, e2 = oth[c]
            i = j = 0
            rs, re_ = [], []
            while i < len(s1) and j < len(s2):
                lo, hi = max(s1[i], s2[j]), min(e1[i], e2[j])
                if hi > lo:
                    rs.append(lo)
                    re_.append(hi)
                if e1[i] < e2[j]:
                    i += 1
                else:
                    j += 1
            if rs:
                nxt[c] = (np.array(rs, np.int64), np.array(re_, np.int64))
        cur = _merge(_as_list(nxt))
    return _as_list(cur)


# ----------------------------------------------------------------------------------------------- strata

def _track_means(track, chrom, starts, ends):
    """Base-weighted mean of a track over [starts, ends) on one contig; uncovered bases count as 0."""
    if callable(track):
        return np.asarray(track(chrom, starts, ends), dtype=float)
    ts, te, tv = track.get(chrom, (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)))
    if len(ts) == 0:
        return np.zeros(len(starts))
    cum = np.concatenate([[0.0], np.cumsum(tv * (te - ts))])

    def integral(x):
        k = np.searchsorted(ts, x, 'left') - 1
        kk = np.maximum(k, 0)
        part = tv[kk] * np.clip(x - ts[kk], 0, te[kk] - ts[kk])
        return np.where(k >= 0, cum[kk] + part, 0.0)

    width = np.maximum(ends - starts, 1)
    return (integral(ends) - integral(starts)) / width


def _track_index(track):
    if track is None or callable(track):
        return track
    by = {}
    for row in track:
        by.setdefault(str(row[0]), []).append((int(row[1]), int(row[2]), float(row[3])))
    out = {}
    for c, rows in by.items():
        rows.sort()
        a = np.array(rows, dtype=float)
        s, e = a[:, 0].astype(np.int64), a[:, 1].astype(np.int64)
        if np.any(s[1:] < e[:-1]):
            raise ValueError(f'tercile_strata: overlapping track intervals on {c}')
        out[c] = (s, e, a[:, 2])
    return out


def _terciles(v):
    q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
    return (v > q1).astype(np.int64) + (v > q2).astype(np.int64)


def tercile_strata(territory, coverage_track, density_track=None, *, bin_bp: int = 10_000) -> list:
    """Coverage tercile x site-density tercile labels (0-8; 0-2 without a density track) as runs over the territory.

    Bins are the territory cut at multiples of ``bin_bp``. A track is rows of (chrom, start, end, value) or a
    callable ``f(chrom, starts, ends) -> values``; a bin's value is the base-weighted mean (uncovered bases count
    as 0). Terciles pool territory bins across contigs; equal values fall in the same tercile. Adjacent bins with
    the same label merge into one run. Returns ``[(chrom, start, end, label)]``.
    """
    merged = _merge(territory)
    cov, den = _track_index(coverage_track), _track_index(density_track)
    bc, bs, be = [], [], []
    for c in merged:
        for s, e in zip(*merged[c]):
            cuts = np.arange((s // bin_bp + 1) * bin_bp, e, bin_bp, dtype=np.int64)
            edges = np.concatenate([[s], cuts, [e]])
            bs.append(edges[:-1])
            be.append(edges[1:])
            bc.append(np.full(len(edges) - 1, c, dtype=object))
    if not bs:
        return []
    bc, bs, be = np.concatenate(bc), np.concatenate(bs), np.concatenate(be)
    cv = np.zeros(len(bs))
    dv = np.zeros(len(bs))
    for c in merged:
        m = bc == c
        cv[m] = _track_means(cov, c, bs[m], be[m])
        if den is not None:
            dv[m] = _track_means(den, c, bs[m], be[m])
    lab = _terciles(cv) * 3 + _terciles(dv) if den is not None else _terciles(cv)
    runs = []
    for i in range(len(bs)):
        if runs and runs[-1][0] == bc[i] and runs[-1][2] == bs[i] and runs[-1][3] == int(lab[i]):
            runs[-1] = (runs[-1][0], runs[-1][1], int(be[i]), runs[-1][3])
        else:
            runs.append((str(bc[i]), int(bs[i]), int(be[i]), int(lab[i])))
    return runs


# ----------------------------------------------------------------------------------------------- frame

class _Frame:
    """The territory on one global axis (contigs laid end to end, separated by a pad)."""

    def __init__(self, territory, pad):
        self.merged = _merge(territory)
        self.contigs = list(self.merged)
        self.offset = {}
        off = 0
        t0, t1, tc, clo, chi = [], [], [], [], []
        for i, c in enumerate(self.contigs):
            s, e = self.merged[c]
            self.offset[c] = off
            t0.append(s + off)
            t1.append(e + off)
            tc.append(np.full(len(s), i, np.int64))
            clo.append(off)
            chi.append(off + int(e[-1]))
            off += int(e[-1]) + pad
        self.clo, self.chi = np.array(clo, np.int64), np.array(chi, np.int64)   # global bounds of each contig
        self.t0 = np.concatenate(t0) if t0 else np.zeros(0, np.int64)
        self.t1 = np.concatenate(t1) if t1 else np.zeros(0, np.int64)
        self.tcontig = np.concatenate(tc) if tc else np.zeros(0, np.int64)

    def place(self, hits: HitSet):
        """Global coordinates of hits ([p, p + 1) for zero-width) and the territory interval wholly containing
        each (-1 when none)."""
        n = len(hits)
        g0 = np.zeros(n, np.int64)
        g1 = np.zeros(n, np.int64)
        tid = np.full(n, -1, np.int64)
        if n == 0 or len(self.t0) == 0:
            return g0, g1, tid
        names, inv = np.unique(hits.chrom, return_inverse=True)
        offs = np.array([self.offset.get(str(c), -1) for c in names], np.int64)[inv]
        cidx = np.array([self.contigs.index(str(c)) if str(c) in self.offset else -1 for c in names],
                        np.int64)[inv]
        known = offs >= 0
        e_eff = np.maximum(hits.end, hits.start + 1)
        g0 = hits.start + np.where(known, offs, 0)
        g1 = e_eff + np.where(known, offs, 0)
        k = np.searchsorted(self.t0, g0, 'right') - 1
        kk = np.maximum(k, 0)
        # the containing territory interval must belong to the hit's own contig (a coordinate past one contig's
        # end must not land in the next contig's territory on the concatenated axis)
        ok = known & (k >= 0) & (g0 >= self.t0[kk]) & (g1 <= self.t1[kk]) & (self.tcontig[kk] == cidx)
        tid = np.where(ok, kk, -1)
        return g0, g1, tid


class _BIndex:
    """Used B hits sorted by start, with a prefix maximum of their ends."""

    def __init__(self, g0, g1):
        o = np.argsort(g0, kind='stable')
        self.b0 = g0[o]
        self.pm1 = np.maximum.accumulate(g1[o]) if len(o) else np.zeros(0, np.int64)
        self.n = len(o)

    def gaps(self, a0, a1, lo, hi):
        """(gl, gr): left gap (negative when some B overlaps) and right gap to the nearest B on the same contig
        (global bounds [lo, hi)); _BIG when there is none."""
        if self.n == 0:
            big = np.full(np.shape(a0), _BIG, np.int64)
            return big, big.copy()
        idx = np.searchsorted(self.b0, a1, 'left')
        right = self.b0[np.minimum(idx, self.n - 1)]
        gr = np.where((idx < self.n) & (right < hi), right - a1, _BIG)
        left = self.pm1[np.maximum(idx - 1, 0)]     # the furthest-reaching earlier B; on A's contig iff it ends past lo
        gl = np.where((idx > 0) & (left > lo), a0 - left, _BIG)
        return gl, gr


def _hist_edges():
    return np.arange(-HIST_RANGE, HIST_RANGE + HIST_STEP, HIST_STEP, dtype=np.int64)


def _signed_bins(gl, gr, minus):
    """Histogram bin of the signed distance to the nearest B: 0 = underflow, 1..nb = bins, nb + 1 = overflow.

    Signed distance: 0 when some B overlaps; +gap when the nearest B lies downstream in A's orientation
    (higher coordinates, or lower ones for a '-' hit), -gap when upstream; equidistant B counts as downstream.
    """
    edges = _hist_edges()
    nb = len(edges) - 1
    down = np.where(minus, gl, gr)
    up = np.where(minus, gr, gl)
    over = gl < 0
    s = np.where(over, 0, np.where(down <= up, down, -up))
    s = np.clip(s, -HIST_RANGE - 1, HIST_RANGE + 1)
    b = np.searchsorted(edges, s, 'right')          # 0 = underflow; nb + 1 = at or above the last edge
    b = np.where(s == edges[-1], nb, b)
    return b, nb


# ----------------------------------------------------------------------------------------------- preparation

class _Prep:
    """A hits on the frame: used hits, clusters, cluster strata and placement tables."""

    def __init__(self, a: HitSet, territory, strata, distances, cluster_gap, strand_aware):
        self.distances = np.array(sorted({int(d) for d in distances}), np.int64)
        if len(self.distances) == 0 or self.distances[0] < 0:
            raise ValueError('distances must be non-negative')
        pad = 4 * (int(self.distances[-1]) + HIST_RANGE) + 1_000_000 + 4 * int(cluster_gap)
        self.frame = fr = _Frame(territory, pad)
        self.cluster_gap = int(cluster_gap)
        g0, g1, tid = fr.place(a)
        used = tid >= 0
        self.n_a_total = len(a)
        self.n_a_dropped = int((~used).sum())
        idx = np.flatnonzero(used)
        o = np.lexsort((g1[idx], g0[idx]))
        idx = idx[o]
        self.src = idx                         # positions in the input HitSet
        self.a0, self.a1, self.tid = g0[idx], g1[idx], tid[idx]
        self.zero_width = (a.end[idx] == a.start[idx])
        if strand_aware and a.strand is not None:
            self.minus = a.strand[idx] == '-'
        else:
            self.minus = np.zeros(len(idx), bool)
        hc = fr.tcontig[self.tid] if len(idx) else np.zeros(0, np.int64)
        self.hlo, self.hhi = fr.clo[hc], fr.chi[hc]   # global bounds of each hit's contig
        # clusters: single linkage, gap to the running cluster end <= cluster_gap
        n = len(idx)
        if n:
            run_end = np.maximum.accumulate(self.a1)
            brk = np.ones(n, bool)
            brk[1:] = (self.a0[1:] - run_end[:-1]) > self.cluster_gap
            brk[1:] |= fr.tcontig[self.tid[1:]] != fr.tcontig[self.tid[:-1]]
            self.cid = np.cumsum(brk) - 1
            starts = np.flatnonzero(brk)
            self.c_first = starts
            self.c0 = self.a0[starts]
            self.c1 = np.maximum.reduceat(self.a1, starts)
            self.ccontig = fr.tcontig[self.tid[starts]]
            self.c_spans_gap = np.minimum.reduceat(self.tid, starts) != np.maximum.reduceat(self.tid, starts)
        else:
            self.cid = np.zeros(0, np.int64)
            self.c_first = self.c0 = self.c1 = self.ccontig = np.zeros(0, np.int64)
            self.c_spans_gap = np.zeros(0, bool)
        self.n_clusters = len(self.c0)
        self._strata(strata)

    def _strata(self, strata):
        fr = self.frame
        if strata is None:
            self.strata_mode = 'one stratum per contig'
            self.labels = [0]
            p0, p1, plab, ptid = fr.t0.copy(), fr.t1.copy(), np.zeros(len(fr.t0), np.int64), np.arange(len(fr.t0))
        else:
            self.strata_mode = 'given'
            rows = [(str(r[0]), int(r[1]), int(r[2]), r[3]) for r in strata if int(r[2]) > int(r[1])]
            self.labels = sorted({r[3] for r in rows}, key=_label_key)
            code = {lab: i for i, lab in enumerate(self.labels)}
            by = {}
            for c, s, e, lab in rows:
                if c in fr.offset:
                    by.setdefault(c, []).append((s, e, code[lab]))
            P0, P1, PL, PT = [], [], [], []
            for ci, c in enumerate(fr.contigs):
                runs = sorted(by.get(c, []))
                if not runs:
                    continue
                rs = np.array([r[0] for r in runs], np.int64) + fr.offset[c]
                re_ = np.array([r[1] for r in runs], np.int64) + fr.offset[c]
                rl = np.array([r[2] for r in runs], np.int64)
                if np.any(rs[1:] < re_[:-1]):
                    raise ValueError(f'strata runs overlap on {c}')
                tk = np.arange(np.searchsorted(fr.tcontig, ci, 'left'), np.searchsorted(fr.tcontig, ci, 'right'))
                ts, te = fr.t0[tk], fr.t1[tk]
                bp = np.unique(np.concatenate([ts, te, rs, re_]))
                lo, hi = bp[:-1], bp[1:]
                it = np.searchsorted(ts, lo, 'right') - 1
                ir = np.searchsorted(rs, lo, 'right') - 1
                ok = (it >= 0) & (ir >= 0)
                ok &= (lo < te[np.maximum(it, 0)]) & (lo < re_[np.maximum(ir, 0)])
                P0.append(lo[ok])
                P1.append(hi[ok])
                PL.append(rl[ir[ok]])
                PT.append(tk[it[ok]])
            cat = (lambda xs: np.concatenate(xs) if xs else np.zeros(0, np.int64))
            p0, p1, plab, ptid = cat(P0), cat(P1), cat(PL), cat(PT)
        # maximal runs: merge abutting pieces with the same label inside the same territory interval
        if len(p0):
            new = np.ones(len(p0), bool)
            new[1:] = ~((p0[1:] == p1[:-1]) & (plab[1:] == plab[:-1]) & (ptid[1:] == ptid[:-1]))
            st = np.flatnonzero(new)
            r0, r1, rlab, rtid = p0[st], np.maximum.reduceat(p1, st), plab[st], ptid[st]
        else:
            r0 = r1 = rlab = rtid = np.zeros(0, np.int64)
        self.r0, self.r1, self.rlab, self.rtid = r0, r1, rlab, rtid
        # cluster stratum: the label covering most in-territory bases of the span (ties: smallest label)
        nl = len(self.labels)
        cl = np.full(self.n_clusters, -1, np.int64)
        for c in range(self.n_clusters):
            s, e = self.c0[c], self.c1[c]
            i0 = np.searchsorted(r1, s, 'right')
            i1 = np.searchsorted(r0, e, 'left')
            if i1 <= i0:
                continue
            ov = np.minimum(r1[i0:i1], e) - np.maximum(r0[i0:i1], s)
            w = np.bincount(rlab[i0:i1], weights=np.maximum(ov, 0), minlength=nl)
            if w.max() > 0:
                cl[c] = int(np.argmax(w))
        self.clabel = cl
        # placement tables: runs of each (contig, label) sorted by length, longest first
        rcontig = fr.tcontig[rtid] if len(rtid) else np.zeros(0, np.int64)
        gkey = rcontig * nl + rlab
        rlen = r1 - r0
        o = np.lexsort((r0, -rlen, gkey))
        self.R0, self.LEN, self.RT = r0[o], rlen[o], rtid[o]
        gk = gkey[o]
        self.PEX = np.zeros(len(o), np.int64)
        goff = {}
        keys, first, counts = np.unique(gk, return_index=True, return_counts=True)
        csum = np.concatenate([[0], np.cumsum(self.LEN)])
        for key, f0, m in zip(keys, first, counts):
            goff[int(key)] = (int(f0), int(m))
            self.PEX[f0:f0 + m] = csum[f0:f0 + m] - csum[f0]
        span = self.c1 - self.c0
        self.span = span
        self.goff = np.zeros(self.n_clusters, np.int64)
        self.nelig = np.zeros(self.n_clusters, np.int64)
        for c in range(self.n_clusters):
            if cl[c] < 0:
                continue
            key = int(self.ccontig[c]) * nl + int(cl[c])
            if key not in goff:
                continue
            o0, m = goff[key]
            lens = self.LEN[o0:o0 + m]
            self.goff[c] = o0
            self.nelig[c] = int(np.searchsorted(-lens, -span[c], 'right'))
        self.total = np.zeros(self.n_clusters, np.int64)
        ok = self.nelig > 0
        if ok.any():
            last = self.goff[ok] + self.nelig[ok] - 1
            self.total[ok] = self.PEX[last] + self.LEN[last] - self.nelig[ok] * (span[ok] - 1)
        self.placeable = ok

    def keep(self):
        """Restrict to hits of placeable clusters (call once, after the refusal check)."""
        keepc = np.flatnonzero(self.placeable)
        remap = np.full(self.n_clusters, -1, np.int64)
        remap[keepc] = np.arange(len(keepc))
        h = self.placeable[self.cid]
        for name in ('src', 'a0', 'a1', 'tid', 'zero_width', 'minus', 'hlo', 'hhi'):
            setattr(self, name, getattr(self, name)[h])
        self.cid = remap[self.cid[h]]
        for name in ('c0', 'c1', 'ccontig', 'clabel', 'span', 'goff', 'nelig', 'total', 'c_spans_gap'):
            setattr(self, name, getattr(self, name)[keepc])
        self.c_first = np.flatnonzero(np.r_[True, self.cid[1:] != self.cid[:-1]]) if len(self.cid) else self.cid
        self.nk = len(keepc)

    def chunk(self):
        return max(1, _CHUNK_ELEMENTS // max(len(self.a0), self.nk, 1))

    def draw(self, rng, cidx):
        """Uniform admissible starts for clusters ``cidx`` (an index array of any shape): (starts, run territory)."""
        u = rng.integers(0, self.total[cidx])
        goff, lm1 = self.goff[cidx], self.span[cidx] - 1
        lo = np.zeros(np.shape(cidx), np.int64)
        hi = self.nelig[cidx] - 1
        while (lo < hi).any():
            mid = (lo + hi) >> 1
            pos = goff + mid
            left = self.PEX[pos] + self.LEN[pos] - (mid + 1) * lm1 > u
            hi = np.where(left, mid, hi)
            lo = np.where(left, lo, mid + 1)
        pos = goff + lo
        return self.R0[pos] + u - (self.PEX[pos] - lo * lm1), self.RT[pos]

    def separate(self, rng, s, rt, max_rounds):
        """Redraw clusters (in place) until, in every row, relocated clusters lie more than cluster_gap apart.

        Each round sorts the unfinished rows by start; a cluster within cluster_gap of an earlier one is paired with
        the earlier cluster that reaches furthest, and one cluster of each pair (random priority) is redrawn from
        its own admissible starts. Returns the indices of rows still in conflict after ``max_rounds`` rounds.
        """
        nrel, nk = s.shape
        if nk < 2:
            return np.zeros(0, np.int64)
        rows, pos = np.arange(nrel), np.arange(nk)
        for rnd in range(max_rounds + 1):
            sub = s[rows]
            order = np.argsort(sub, axis=1, kind='stable')
            ss = np.take_along_axis(sub, order, 1)
            ee = ss + self.span[order]
            reach = np.maximum.accumulate(ee, axis=1)
            who = np.maximum.accumulate(np.where(ee == reach, pos, 0), axis=1)
            conf = np.zeros(ss.shape, bool)
            conf[:, 1:] = ss[:, 1:] - reach[:, :-1] <= self.cluster_gap
            bad = conf.any(axis=1)
            rows, order, conf, who = rows[bad], order[bad], conf[bad], who[bad]
            if len(rows) == 0 or rnd == max_rounds:
                break
            ri, ki = np.nonzero(conf)
            late, early = order[ri, ki], order[ri, who[ri, ki - 1]]
            prio = rng.random((len(rows), nk))
            redo = np.zeros((len(rows), nk), bool)
            redo[ri, np.where(prio[ri, late] < prio[ri, early], late, early)] = True
            rr, cc = np.nonzero(redo)
            s[rows[rr], cc], rt[rows[rr], cc] = self.draw(rng, cc)
        return rows

    def relocations(self, rng, n, kept_apart=True, max_rounds=200):
        """Yield (a0, a1, room) arrays of shape (rows, n_hits) for n relocation draws, in deterministic chunks.

        ``kept_apart``: relocated clusters lie more than cluster_gap apart, as linkage guarantees for the observed
        clusters. Draws still in conflict after ``max_rounds`` redraw rounds are left out (rejection of invalid
        configurations) and counted in ``self.n_capped``; the rows yielded total n - n_capped.
        """
        step = self.chunk()
        fr = self.frame
        done = 0
        self.n_capped = 0
        while done < n:
            m = min(step, n - done)
            s, rt = self.draw(rng, np.broadcast_to(np.arange(self.nk), (m, self.nk)))
            if kept_apart:
                capped = self.separate(rng, s, rt, max_rounds)
                if len(capped):
                    self.n_capped += len(capped)
                    keep = np.ones(m, bool)
                    keep[capped] = False
                    s, rt = s[keep], rt[keep]
            off = (s - self.c0[None, :])[:, self.cid]
            a0 = self.a0[None, :] + off
            a1 = self.a1[None, :] + off
            rth = rt[:, self.cid]
            room = np.minimum(a0 - fr.t0[rth], fr.t1[rth] - a1)
            yield a0, a1, room
            done += m

    def observed(self):
        fr = self.frame
        room = np.minimum(self.a0 - fr.t0[self.tid], fr.t1[self.tid] - self.a1)
        return self.a0[None, :], self.a1[None, :], room[None, :]


def _evaluate(bidx, prep, a0, a1, room, want_hist=True):
    """Per row: hits and denominators per d; per hit: summed hits and denominators per d; histogram counts."""
    minus, distances = prep.minus, prep.distances
    gl, gr = bidx.gaps(a0, a1, prep.hlo, prep.hhi)
    gmin = np.minimum(gl, gr)
    nd = len(distances)
    rows_h = np.zeros((a0.shape[0], nd), np.int64)
    rows_n = np.zeros((a0.shape[0], nd), np.int64)
    hit_h = np.zeros((nd, a0.shape[1]), np.int64)
    hit_n = np.zeros((nd, a0.shape[1]), np.int64)
    for k, d in enumerate(distances):
        w = gmin < d
        den = w | (room >= d)
        rows_h[:, k] = w.sum(1)
        rows_n[:, k] = den.sum(1)
        hit_h[k] = w.sum(0)
        hit_n[k] = den.sum(0)
    hist = None
    if want_hist:
        b, nb = _signed_bins(gl, gr, np.broadcast_to(minus, a0.shape))
        hist = np.bincount(b.ravel(), minlength=nb + 2).astype(np.int64)
    return rows_h, rows_n, hit_h, hit_n, hist


# ----------------------------------------------------------------------------------------------- helpers

def _fx(x):
    """A float for JSON: None for undefined, the strings '-inf' / 'inf' for unbounded values."""
    if x is None or math.isnan(float(x)):
        return None
    x = float(x)
    return x if math.isfinite(x) else ('inf' if x > 0 else '-inf')


def _order_stat(x, q, upper):
    """The q-quantile of x without interpolation (the order statistic below, or above when upper)."""
    v = np.sort(x)
    pos = q * (len(v) - 1)
    return v[int(math.ceil(pos)) if upper else int(math.floor(pos))]


def _f(x):
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _seed(seed):
    return int(seed) if isinstance(seed, (int, np.integer)) else None if seed is None else str(seed)


def _place_b(frame, b: HitSet):
    g0, g1, tid = frame.place(b)
    ok = tid >= 0
    return _BIndex(g0[ok], g1[ok]), int(ok.sum()), int((~ok).sum()), int((b.end[ok] == b.start[ok]).sum())


def _common(prep, max_unplaced):
    n_unplaced = int((~prep.placeable).sum())
    by = {}
    for c in range(prep.n_clusters):
        lab = prep.labels[prep.clabel[c]] if prep.clabel[c] >= 0 else None
        key = 'none' if lab is None else str(lab)
        ent = by.setdefault(key, [0, 0])
        ent[0] += 1
        ent[1] += int(not prep.placeable[c])
    info = dict(n_a_used=int(len(prep.a0)), n_a_dropped_outside=prep.n_a_dropped, n_clusters=prep.n_clusters,
                n_unplaced=n_unplaced, n_clusters_spanning_gaps=int(prep.c_spans_gap.sum()),
                strata=prep.strata_mode, n_strata=len(prep.labels),
                clusters_by_stratum={k: dict(clusters=v[0], unplaced=v[1]) for k, v in sorted(by.items())})
    refused = prep.n_clusters > 0 and n_unplaced > max_unplaced * prep.n_clusters
    msg = None
    if refused:
        msg = (f'{n_unplaced} of {prep.n_clusters} A clusters ({100 * n_unplaced / prep.n_clusters:.1f}%) have no '
               f'admissible placement in their stratum (limit {100 * max_unplaced:.0f}%). Rerun with coarser strata: '
               f'strata=None (one stratum per contig) or coverage-only terciles '
               f'(tercile_strata(territory, coverage_track, None)).')
    return info, refused, msg


def _capped_message(n_capped, n, rounds, limit):
    return (f'{n_capped} of {n} relocations still had A clusters within cluster_gap of each other after {rounds} '
            f'redraw rounds (limit {100 * limit:.0f}%): some strata are too crowded to keep the clusters apart. '
            f'Rerun with coarser strata, or with kept_apart=False.')


SUGGESTIONS = ['strata=None (one stratum per contig)',
               'coverage-only terciles: tercile_strata(territory, coverage_track, density_track=None)']


# ----------------------------------------------------------------------------------------------- M4 test

def overlap_enrichment(a: HitSet, b: HitSet, territory, strata=None, *, distances=DISTANCES, cluster_gap=1000,
                       n_relocations=1000, seed=0, max_unplaced=0.10, strand_aware=True, kept_apart=True,
                       max_redraw_rounds=200, max_capped=0.01, return_hits=False) -> dict:
    """A->B overlap and distance enrichment against cluster-preserving matched relocation of A (B fixed).

    ``territory``: [(chrom, start, end)], merged first. ``strata``: [(chrom, start, end, label)] runs (see
    :func:`tercile_strata`); None = one stratum per contig. Returns a JSON-able dict; ``status`` is 'ok',
    'refused' (too many clusters without an admissible placement, or too many relocations that could not keep the
    clusters apart; no p-value) or 'empty'. B->A is a separate call with the arguments swapped.

    ``kept_apart``: relocated clusters lie more than ``cluster_gap`` apart (conflict-redraw, at most
    ``max_redraw_rounds`` rounds; close to uniform over separated configurations and exact for sparse clusters).
    Draws still in conflict are left out of the null (``n_relocations_capped``; p-values use
    ``n_relocations_effective``); more than ``max_capped`` of them refuses the analysis. Nearest-B distances look
    only at A's contig; with no B there, the signed distance is None and counts in the histogram overflow. ``return_hits`` adds per-hit rows
    (id, position, zero-width flag, signed distance).
    """
    if int(n_relocations) < 2:
        raise ValueError('n_relocations must be at least 2')
    prep = _Prep(a, territory, strata, distances, cluster_gap, strand_aware)
    D = prep.distances
    bidx, n_b, n_b_drop, n_b_zw = _place_b(prep.frame, b)
    info, refused, msg = _common(prep, max_unplaced)
    out = dict(direction='A->B', status='ok', message=None, **info, n_b_used=n_b, n_b_dropped_outside=n_b_drop,
               n_b_zero_width_used=n_b_zw,
               settings=dict(distances=[int(d) for d in D], cluster_gap=int(cluster_gap),
                             n_relocations=int(n_relocations), seed=_seed(seed), max_unplaced=float(max_unplaced),
                             strand_aware=bool(strand_aware), hist_range=HIST_RANGE, hist_step=HIST_STEP,
                             kept_apart=bool(kept_apart), max_redraw_rounds=int(max_redraw_rounds),
                             max_capped=float(max_capped)),
               n_relocations=int(n_relocations), n_relocations_capped=0, n_relocations_effective=0, rows=[],
               global_stat=None, global_p=None,
               skipped_d=[], histogram=None)
    if refused:
        out.update(status='refused', message=msg, suggestions=SUGGESTIONS)
        return out
    prep.keep()
    out['n_a_used'] = int(len(prep.a0))
    out['n_a_excluded_unplaced'] = int(info['n_a_used'] - len(prep.a0))
    out['n_a_zero_width_used'] = int(prep.zero_width.sum())
    if len(prep.a0) == 0 or n_b == 0:
        out.update(status='empty', message='no A hits' if len(prep.a0) == 0 else 'no B hits in the territory')
        return out
    if info['n_unplaced']:
        out['message'] = (f"{info['n_unplaced']} of {info['n_clusters']} A clusters had no admissible placement; "
                          f"their {out['n_a_excluded_unplaced']} hits are excluded from the observed and null values.")
    rng = np.random.default_rng(seed)
    oh, on, _, _, ohist = _evaluate(bidx, prep, *prep.observed())
    oh, on = oh[0], on[0]
    N = int(n_relocations)
    nh = np.zeros((N, len(D)), np.int64)
    nn = np.zeros((N, len(D)), np.int64)
    hsum = np.zeros_like(ohist)
    r = 0
    for a0, a1, room in prep.relocations(rng, N, kept_apart, max_redraw_rounds):
        rh, rn, _, _, hist = _evaluate(bidx, prep, a0, a1, room)
        nh[r:r + len(rh)] = rh
        nn[r:r + len(rh)] = rn
        hsum += hist
        r += len(rh)
    out['n_relocations_capped'] = int(prep.n_capped)
    out['n_relocations_effective'] = int(r)
    if prep.n_capped > max_capped * N:
        out.update(status='refused', message=_capped_message(prep.n_capped, N, max_redraw_rounds, max_capped),
                   suggestions=SUGGESTIONS + ['kept_apart=False (relocated clusters may overlap; conservative)'])
        return out
    N = r                                     # capped draws are left out of the null
    nh, nn = nh[:N], nn[:N]
    with np.errstate(divide='ignore', invalid='ignore'):
        obs = np.where(on > 0, oh / np.maximum(on, 1), np.nan)
        nul = np.where(nn > 0, nh / np.maximum(nn, 1), np.nan)
    valid = np.isfinite(obs) & np.isfinite(nul).all(axis=0)
    safe = np.where(np.isfinite(nul), nul, 0.0)
    mean = np.where(valid, safe.mean(axis=0), np.nan)
    sd = np.where(valid, safe.std(axis=0, ddof=1), np.nan)
    valid &= np.nan_to_num(sd, nan=0.0) > 0
    sdv = np.where(valid, sd, 1.0)
    z = np.where(valid, (obs - np.nan_to_num(mean)) / sdv, np.nan)
    zs = np.where(valid[None, :], (safe - np.nan_to_num(mean)) / sdv, 0.0)
    rows = []
    for k, d in enumerate(D):
        p = None
        if valid[k]:
            p = (1 + int((np.abs(zs[:, k]) >= abs(z[k]) - 1e-12).sum())) / (N + 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            oe = obs[k] / mean[k] if valid[k] and mean[k] > 0 else np.nan
        rows.append(dict(d=int(d), observed=_f(obs[k]), n_hits=int(oh[k]), n_used=int(on[k]),
                         n_censored=int(len(prep.a0) - on[k]), null_mean=_f(mean[k]), null_sd=_f(sd[k]),
                         o_over_e=_f(oe), z=_f(z[k]), p_descriptive=_f(p)))
    out['rows'] = rows
    out['skipped_d'] = [int(D[k]) for k in range(len(D)) if not valid[k]]
    if valid.any():
        T = float(np.max(np.abs(z[valid])))
        Tn = np.max(np.abs(zs[:, valid]), axis=1)
        out['global_stat'] = T
        out['global_p'] = (1 + int((Tn >= T - 1e-12).sum())) / (N + 1)
    else:
        out['message'] = ((out['message'] + ' ') if out['message'] else '') + \
            'No distance has a non-degenerate null; no global test.'
    edges = _hist_edges()
    out['histogram'] = dict(edges=[int(e) for e in edges], observed=[int(x) for x in ohist[1:-1]],
                            observed_underflow=int(ohist[0]), observed_overflow=int(ohist[-1]),
                            null_mean=[float(x) / max(N, 1) for x in hsum[1:-1]],
                            null_mean_underflow=float(hsum[0]) / max(N, 1),
                            null_mean_overflow=float(hsum[-1]) / max(N, 1),
                            strand_aware=bool(strand_aware and a.strand is not None),
                            note='signed gap to the nearest B (0 = overlap; + = downstream in A orientation); '
                                 'overflow includes A hits with no B on the contig')
    if return_hits:
        gl, gr = bidx.gaps(prep.a0, prep.a1, prep.hlo, prep.hhi)
        down = np.where(prep.minus, gl, gr)
        up = np.where(prep.minus, gr, gl)
        sd_ = np.where(gl < 0, 0, np.where(down <= up, down, -up))
        hits = []
        for i in range(len(prep.a0)):
            j = int(prep.src[i])
            dist = int(sd_[i]) if abs(int(sd_[i])) < _BIG // 2 else None
            hits.append(dict(index=j, id=None if a.ids is None else str(a.ids[j]), chrom=str(a.chrom[j]),
                             start=int(a.start[j]), end=int(a.end[j]), zero_width=bool(prep.zero_width[i]),
                             cluster=int(prep.cid[i]), signed_distance=dist))
        out['hits'] = hits
    return out


# ----------------------------------------------------------------------------------------------- M4 contrast

def _contrast_arrays(prep, bidx_list, N, rng, kept_apart=True, max_rounds=200):
    """Per group, d and A cluster: observed hits and denominators, and their means over one shared relocation draw.
    Also, per d and cluster, whether some hit's d-window lies inside the territory (outcome-independent)."""
    D, starts = prep.distances, prep.c_first
    G, nd, nk = len(bidx_list), len(D), prep.nk
    obs_h = np.zeros((G, nd, nk))
    obs_n = np.zeros((G, nd, nk))
    nul_h = np.zeros((G, nd, len(prep.a0)))
    nul_n = np.zeros((G, nd, len(prep.a0)))
    ob = prep.observed()
    for gi, bidx in enumerate(bidx_list):
        _, _, hh, hn, _ = _evaluate(bidx, prep, *ob, want_hist=False)
        obs_h[gi] = np.add.reduceat(hh, starts, axis=1)
        obs_n[gi] = np.add.reduceat(hn, starts, axis=1)
    n_eff = 0
    for a0, a1, room in prep.relocations(rng, N, kept_apart, max_rounds):
        n_eff += len(a0)
        for gi, bidx in enumerate(bidx_list):
            _, _, hh, hn, _ = _evaluate(bidx, prep, a0, a1, room, want_hist=False)
            nul_h[gi] += hh
            nul_n[gi] += hn
    nul_h = np.add.reduceat(nul_h, starts, axis=2) / max(n_eff, 1)       # capped draws are left out
    nul_n = np.add.reduceat(nul_n, starts, axis=2) / max(n_eff, 1)
    informative = np.maximum.reduceat(ob[2][0][None, :] >= D[:, None], starts, axis=1)      # (nd, nk)
    return dict(obs_h=obs_h, obs_n=obs_n, nul_h=nul_h, nul_n=nul_n, informative=informative, n_capped=prep.n_capped,
                n_effective=n_eff)


def _log_oe(Oh, On, Eh, En):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log((Oh / On) / (Eh / En))


def _contrast_rows(arr, groups, D, plist, n_boot, rng, interval='percentile', resample='all'):
    """Bootstrap over A clusters for every pair and d: log O/E difference, 95% CI and two-sided p.

    Each row carries p_percentile (min(1, 2 min(P*(diff <= 0), P*(diff >= 0))), P* = (1 + count) / (n + 1)) and
    p_normal (2 Phi(-|diff| / bootstrap SD)); ``interval`` picks which one is ``p`` and the CI. Every draw counts:
    signed infinities (a resample with no hits in a group) stay in the tails, a draw undefined in both groups
    counts as a tie, and unbounded interval ends are reported as '-inf' / 'inf'. The normal interval is left out
    (p_normal None) when any draw is non-finite or the bootstrap SD is 0.
    ``resample='informative'`` restricts estimate and bootstrap at each d to clusters with a hit whose d-window lies
    inside the territory.
    """
    G, nd, nk = arr['obs_h'].shape
    out = []
    for k, dd in enumerate(D):
        keep = np.arange(nk) if resample == 'all' else np.flatnonzero(arr['informative'][k])
        m_ = len(keep)
        X = np.stack([arr[n][:, k, keep] for n in ('obs_h', 'obs_n', 'nul_h', 'nul_n')], 0)     # (4, G, m)
        point = _log_oe(*X.sum(2))                                                              # (G,)
        boot = np.empty((int(n_boot), 4 * G))
        if m_:
            Xf = X.reshape(4 * G, m_).T
            step = max(1, _CHUNK_ELEMENTS // m_)
            done = 0
            while done < n_boot:
                m = min(step, n_boot - done)
                idx = rng.integers(0, m_, size=(m, m_))
                W = np.bincount((idx + (np.arange(m) * m_)[:, None]).ravel(), minlength=m * m_).reshape(m, m_)
                boot[done:done + m] = W @ Xf
                done += m
        blog = _log_oe(*boot.reshape(int(n_boot), 4, G).transpose(1, 0, 2))                    # (n_boot, G)
        for i, j in plist:
            with np.errstate(invalid='ignore'):
                diff = point[j] - point[i]
                bd = blog[:, j] - blog[:, i]
            und = np.isnan(bd)              # both groups undefined in a draw: counts as a tie
            row = dict(group=groups[j], reference=groups[i], d=int(dd), log_oe_diff=_fx(diff), ci_low=None,
                       ci_high=None, p=None, p_percentile=None, p_normal=None, n_boot=int(len(bd)),
                       n_boot_infinite=int(np.isinf(bd).sum()), n_boot_undefined=int(und.sum()),
                       n_clusters_used=int(m_))
            if m_ and not math.isnan(diff) and len(bd) > 1:
                le = (1 + int(((bd <= 0) | und).sum())) / (len(bd) + 1)
                ge = (1 + int(((bd >= 0) | und).sum())) / (len(bd) + 1)
                p_pct = min(1.0, 2 * min(le, ge))
                lo = _order_stat(np.where(und, -np.inf, bd), 0.025, False)
                hi = _order_stat(np.where(und, np.inf, bd), 0.975, True)
                se = float(np.std(bd, ddof=1)) if (np.isfinite(bd).all() and math.isfinite(diff)) else None
                p_nrm = math.erfc(abs(diff) / se / math.sqrt(2)) if se else None
                row.update(p_percentile=p_pct, p_normal=p_nrm)
                if interval == 'normal':
                    if p_nrm is None:
                        row['note'] = 'normal interval undefined: non-finite or zero-variance bootstrap'
                    else:
                        row.update(ci_low=_fx(diff - Z95 * se), ci_high=_fx(diff + Z95 * se), p=p_nrm)
                else:
                    row.update(ci_low=_fx(lo), ci_high=_fx(hi), p=p_pct)
            out.append(row)
    return out


def overlap_contrast(a: HitSet, b_by_group: Mapping[str, HitSet], territory, strata=None, *, d=DISTANCES,
                     n_relocations=1000, n_boot=1000, seed=0, pairs='first', cluster_gap=1000, max_unplaced=0.10,
                     strand_aware=True, kept_apart=True, max_redraw_rounds=200, max_capped=0.01,
                     interval='percentile', resample='all') -> dict:
    """Difference in log O/E between groups of B for the same A (HUB section 7.7, M4).

    O/E_g at each d is (sum of observed hits / sum of observed denominators) over (the same sums averaged over
    relocations), summed over A clusters; one relocation draw (kept apart as in overlap_enrichment) serves every
    group. The interval and p come from a bootstrap over A clusters (see _contrast_rows): ``interval`` 'percentile'
    (percentile 95% CI and p = min(1, 2 min(P*(diff <= 0), P*(diff >= 0))), P* = (1 + count) / (n_valid + 1)) or
    'normal' (diff +/- 1.96 bootstrap SD, normal p); both p-values are reported. Infinite bootstrap draws are kept
    and unbounded interval ends are reported as '-inf' / 'inf'. Relocation draws that could not be kept apart are
    left out (``n_relocations_effective``). ``pairs``: 'first' compares every
    group with the first; 'all' every pair. Note: this O/E is a ratio of sums; overlap_enrichment reports the mean
    of per-relocation fractions.
    """
    groups = list(b_by_group)
    if len(groups) < 2:
        raise ValueError('overlap_contrast needs at least two groups')
    if interval not in ('percentile', 'normal') or resample not in ('all', 'informative'):
        raise ValueError("interval must be 'percentile' or 'normal'; resample 'all' or 'informative'")
    if pairs == 'all':
        plist = [(i, j) for i in range(len(groups)) for j in range(i + 1, len(groups))]
    elif pairs == 'first':
        plist = [(0, j) for j in range(1, len(groups))]
    else:
        raise ValueError("pairs must be 'first' or 'all'")
    ds = (d,) if np.isscalar(d) else tuple(d)
    prep = _Prep(a, territory, strata, ds, cluster_gap, strand_aware)
    D = prep.distances
    info, refused, msg = _common(prep, max_unplaced)
    bs = {g: _place_b(prep.frame, b_by_group[g]) for g in groups}
    N = int(n_relocations)
    out = dict(direction='A->B', status='ok', message=None, **info, groups=groups,
               settings=dict(distances=[int(x) for x in D], cluster_gap=int(cluster_gap), n_relocations=N,
                             n_boot=int(n_boot), seed=_seed(seed), pairs=pairs, max_unplaced=float(max_unplaced),
                             kept_apart=bool(kept_apart), max_redraw_rounds=int(max_redraw_rounds),
                             max_capped=float(max_capped), interval=interval, resample=resample),
               n_relocations_capped=0, n_relocations_effective=0, per_group=[], contrasts=[])
    if refused:
        out.update(status='refused', message=msg, suggestions=SUGGESTIONS)
        return out
    prep.keep()
    out['n_a_used'] = int(len(prep.a0))
    out['n_a_excluded_unplaced'] = int(info['n_a_used'] - len(prep.a0))
    if len(prep.a0) == 0:
        out.update(status='empty', message='no A hits')
        return out
    rng = np.random.default_rng(seed)
    arr = _contrast_arrays(prep, [bs[g][0] for g in groups], N, rng, kept_apart, max_redraw_rounds)
    out['n_relocations_capped'] = int(arr['n_capped'])
    out['n_relocations_effective'] = int(arr['n_effective'])
    if arr['n_capped'] > max_capped * N:
        out.update(status='refused', message=_capped_message(arr['n_capped'], N, max_redraw_rounds, max_capped),
                   suggestions=SUGGESTIONS + ['kept_apart=False (relocated clusters may overlap; conservative)'])
        return out
    point = _log_oe(arr['obs_h'].sum(2), arr['obs_n'].sum(2), arr['nul_h'].sum(2), arr['nul_n'].sum(2))  # (G, nd)
    for gi, g in enumerate(groups):
        _, nb, nbd, _ = bs[g]
        rows = []
        for k, dd in enumerate(D):
            on, oh = arr['obs_n'][gi, k].sum(), arr['obs_h'][gi, k].sum()
            en, eh = arr['nul_n'][gi, k].sum(), arr['nul_h'][gi, k].sum()
            rows.append(dict(d=int(dd), observed=_f(oh / on if on else np.nan),
                             null_mean=_f(eh / en if en else np.nan), o_over_e=_f(np.exp(point[gi, k])),
                             log_o_over_e=_fx(point[gi, k])))
        out['per_group'].append(dict(group=g, n_b_used=nb, n_b_dropped_outside=nbd, rows=rows))
    rows = _contrast_rows(arr, groups, D, plist, n_boot, rng, interval, resample)
    out['contrasts'] = sorted(rows, key=lambda r: (groups.index(r['reference']), groups.index(r['group']), r['d']))
    if info['n_unplaced']:
        out['message'] = (f"{info['n_unplaced']} of {info['n_clusters']} A clusters had no admissible placement; "
                          f"their {out['n_a_excluded_unplaced']} hits are excluded.")
    return out


# ----------------------------------------------------------------------------------------------- simulator

def _runs_of(territory, strata):
    """Maximal runs of territory x stratum as arrays per contig: {chrom: (start, end, label)}."""
    prep_frame = _Frame(territory, 0)
    out = {}
    for c in prep_frame.contigs:
        ts, te = prep_frame.merged[c]
        rows = sorted((s, e, lab) for cc, s, e, lab in strata if cc == c)
        r0, r1, rl = [], [], []
        for t0, t1 in zip(ts, te):
            for s, e, lab in rows:
                lo, hi = max(s, t0), min(e, t1)
                if hi <= lo:
                    continue
                if r0 and r1[-1] == lo and rl[-1] == lab:
                    r1[-1] = hi
                else:
                    r0.append(lo)
                    r1.append(hi)
                    rl.append(lab)
        out[c] = (np.array(r0, np.int64), np.array(r1, np.int64), np.array(rl, np.int64))
    return out


def _cluster_shape(rng, max_hits=4, link=1000, zero_width_frac=0.05):
    k = int(rng.integers(1, max_hits + 1))
    rel0 = np.zeros(k, np.int64)
    for i in range(1, k):
        rel0[i] = rel0[i - 1] + int(rng.integers(0, link))
    w = rng.integers(1, 501, size=k)
    zw = rng.random(k) < zero_width_frac
    rel1 = rel0 + np.where(zw, 0, w)
    span = int(np.max(np.maximum(rel1, rel0 + 1)))
    return rel0, rel1, span


def simulate_overlap(seed=0, *, n_contigs=4, contig_len=2_000_000, n_a=1000, n_b=10_000, n_strata=9,
                     b_ratio=10.0, a_ratio=10.0, strata_len=(2_000, 50_000), callable_mean=200_000,
                     gap_mean=20_000, b_cluster_mean=2.0, outside_b_frac=0.03, zero_width_frac=0.05,
                     planted=0.0, planted_max=500, n_groups=1, group_planted=0.0, a_seed=None,
                     min_separation=None) -> dict:
    """Synthetic frame with stratum-confounded A and B (A independent of B given strata unless planted).

    Strata are piecewise-constant runs (log-uniform lengths in ``strata_len``) of labels 0..n_strata-1 over contigs
    with callable gaps. B density per base rises ``b_ratio``-fold from stratum 0 to the last; B comes in clusters.
    A clusters (1-4 hits within 1 kb, widths 1-500 bp, some zero-width) are placed uniformly over admissible
    starts in their stratum with a rate rising ``a_ratio``-fold the same way (the confounding). ``planted``: the
    fraction of A clusters placed 0-``planted_max`` bp from a B hit of group 0. ``n_groups`` independent B sets
    under the same rates; ``group_planted`` adds, for the last group, one B hit within 0-``planted_max`` bp of that
    fraction of A clusters, and to every other group as many B hits placed at the B rates. ``a_seed`` draws A
    (and the group planting) from its own stream, so the frame and B stay fixed while A varies.
    ``min_separation``: reject A cluster placements within this gap of an existing A cluster (rejection sampling,
    so placements stay uniform over the allowed starts); with ``cluster_gap`` here, linkage recovers exactly the
    generated clusters. None (default) places clusters independently.
    Returns dict(a, b, b_by_group, territory, strata, truth).
    """
    rng = np.random.default_rng(seed)
    contigs = [f'sim{i + 1}' for i in range(n_contigs)]
    territory, strata = [], []
    for c in contigs:
        pos = int(rng.integers(0, int(gap_mean) + 1))
        while pos < contig_len:
            ln = int(rng.exponential(callable_mean)) + 5_000
            territory.append((c, pos, min(pos + ln, contig_len)))
            pos += ln + int(rng.exponential(gap_mean)) + 500
        pos = 0
        lo, hi = np.log(strata_len[0]), np.log(strata_len[1])
        while pos < contig_len:
            ln = int(np.exp(rng.uniform(lo, hi)))
            strata.append((c, pos, min(pos + ln, contig_len), int(rng.integers(0, n_strata))))
            pos += ln
    runs = _runs_of(territory, strata)
    wb = b_ratio ** (np.arange(n_strata) / max(n_strata - 1, 1))
    wa = a_ratio ** (np.arange(n_strata) / max(n_strata - 1, 1))
    rc = np.concatenate([np.full(len(runs[c][0]), i) for i, c in enumerate(contigs)])
    r0 = np.concatenate([runs[c][0] for c in contigs])
    r1 = np.concatenate([runs[c][1] for c in contigs])
    rl = np.concatenate([runs[c][2] for c in contigs])
    rlen = r1 - r0

    def make_b(n_hits):
        ncl = max(1, int(round(n_hits * (1 - outside_b_frac) / b_cluster_mean)))
        p = rlen * wb[rl]
        pick = rng.choice(len(r0), size=ncl, p=p / p.sum())
        centre = r0[pick] + (rng.random(ncl) * rlen[pick]).astype(np.int64)
        sizes = 1 + rng.poisson(b_cluster_mean - 1, size=ncl)
        cc = np.repeat(rc[pick], sizes)
        st = np.repeat(centre, sizes) + rng.integers(-1000, 1001, size=sizes.sum())
        nout = int(round(n_hits * outside_b_frac))
        cc = np.concatenate([cc, rng.integers(0, n_contigs, size=nout)])
        st = np.concatenate([st, rng.integers(0, contig_len, size=nout)])
        st = np.maximum(st, 0)
        en = st + rng.integers(10, 301, size=len(st))
        return cc.astype(np.int64), st.astype(np.int64), en.astype(np.int64)

    def as_hits(cc, st, en, strand=None, prefix='b'):
        return HitSet(np.array([contigs[i] for i in cc], dtype=str), st, en, strand,
                      np.array([f'{prefix}{i}' for i in range(len(st))], dtype=object))

    braw = [make_b(n_b) for _ in range(n_groups)]
    # B hits of group 0 inside a run, for planting A near B
    b0c, b0s, b0e = braw[0]
    if a_seed is not None:
        rng = np.random.default_rng(a_seed)
    a_c, a_s, a_e = [], [], []
    n_planted = n_fallback = n_gen = n_rejected = 0
    placed_spans, longest = {}, [0]

    def clash(ci, s, span):
        """An existing A cluster on contig ci within min_separation of [s, s + span)."""
        if min_separation is None:
            return False
        lst = placed_spans.get(ci, [])
        for e0, e1 in lst[bisect.bisect_left(lst, (s - min_separation - longest[0],)):]:
            if e0 - (s + span) > min_separation:
                break
            if s - e1 <= min_separation:
                return True
        return False
    total = 0
    while total < n_a:
        rel0, rel1, span = _cluster_shape(rng, zero_width_frac=zero_width_frac)
        placed = None
        if planted > 0 and rng.random() < planted:
            for _ in range(100):
                j = int(rng.integers(0, len(b0s)))
                delta = int(rng.integers(0, planted_max + 1))
                s = int(b0e[j]) + delta if rng.random() < 0.5 else int(b0s[j]) - delta - span
                ci = int(b0c[j])
                m = np.flatnonzero((rc == ci) & (r0 <= s) & (r1 >= s + span))
                if len(m) and clash(ci, s, span):
                    n_rejected += 1
                    continue
                if len(m):
                    placed = (ci, s)
                    n_planted += 1
                    break
            if placed is None:
                n_fallback += 1
        if placed is None:
            cnt = np.maximum(rlen - span + 1, 0)
            p = cnt * wa[rl]
            if p.sum() <= 0:
                continue
            for _ in range(1000):
                k = int(rng.choice(len(r0), p=p / p.sum()))
                placed = (int(rc[k]), int(r0[k] + rng.integers(0, cnt[k])))
                if not clash(*placed, span):
                    break
                n_rejected += 1
                placed = None
            if placed is None:
                continue
        ci, s = placed
        n_gen += 1
        if min_separation is not None:
            bisect.insort(placed_spans.setdefault(ci, []), (s, s + span))
            longest[0] = max(longest[0], span)
        a_c.extend([ci] * len(rel0))
        a_s.extend((s + rel0).tolist())
        a_e.extend((s + rel1).tolist())
        total += len(rel0)
    a_c, a_s, a_e = np.array(a_c, np.int64), np.array(a_s, np.int64), np.array(a_e, np.int64)
    strand = np.where(rng.random(len(a_s)) < 0.5, '+', '-')
    a = as_hits(a_c, a_s, a_e, strand, 'a')
    if group_planted > 0 and n_groups >= 2:
        # cluster starts of A (single linkage on the generated hits)
        o = np.lexsort((a_s, a_c))
        cs, cc_, run_end = [], [], -1
        prev_c = -1
        for i in o:
            e_eff = max(a_e[i], a_s[i] + 1)
            if a_c[i] != prev_c or a_s[i] - run_end > 1000:
                cs.append((a_s[i], e_eff))
                cc_.append(a_c[i])
                run_end = e_eff
            else:
                cs[-1] = (cs[-1][0], max(cs[-1][1], e_eff))
                run_end = max(run_end, e_eff)
            prev_c = a_c[i]
        cs = np.array(cs, np.int64)
        cc_ = np.array(cc_, np.int64)
        pick = np.flatnonzero(rng.random(len(cs)) < group_planted)
        delta = rng.integers(0, planted_max + 1, size=len(pick))
        w = rng.integers(10, 301, size=len(pick))
        right = rng.random(len(pick)) < 0.5
        ps = np.where(right, cs[pick, 1] + delta, cs[pick, 0] - delta - w)
        extra = (cc_[pick], ps, ps + w)
        last = n_groups - 1
        bc, bs_, be = braw[last]
        braw[last] = (np.concatenate([bc, extra[0]]), np.concatenate([bs_, extra[1]]),
                      np.concatenate([be, extra[2]]))
        for g in range(last):
            p = rlen * wb[rl]
            k = rng.choice(len(r0), size=len(pick), p=p / p.sum())
            st = r0[k] + (rng.random(len(pick)) * rlen[k]).astype(np.int64)
            bc, bs_, be = braw[g]
            braw[g] = (np.concatenate([bc, rc[k]]), np.concatenate([bs_, st]), np.concatenate([be, st + w]))
    b_by_group = {f'g{g}': as_hits(*braw[g]) for g in range(n_groups)}
    return dict(a=a, b=b_by_group['g0'], b_by_group=b_by_group, territory=territory, strata=strata,
                truth=dict(n_planted=n_planted, n_planted_fallback=n_fallback, n_a=len(a_s), n_a_clusters=n_gen, n_rejected=n_rejected,
                           n_b=[len(v) for v in b_by_group.values()]))
