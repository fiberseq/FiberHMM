"""Cross-strand read pairing by nucleosome dyad-pattern cross-correlation.

Core (I/O-free) library behind ``fiberhmm-pair``. Given footprint-called DAF
reads, it:

1. extracts each read's deamination **flavor** (CT/GA strand) and its MA ``nuc``
   dyad centers in reference coordinates, and rasterizes the dyads into a
   Gaussian dyad-density signal on a fixed reference grid;
2. scores a candidate cross-strand pair by the lag-tolerant normalized
   cross-correlation of the two signals over their genomic overlap;
3. resolves pairs per read by **reciprocal best match with a margin gate**: a
   read pairs with its top-scoring opposite-strand partner only when that
   partner also ranks the read first *and* the score beats the read's
   second-best partner by ``min_margin`` on both sides. This is the
   whole-genome generalization of the local 2x2 assignment at a diploid locus
   (second-best == the competing wrong matching); low-margin loci are left
   unresolved rather than force-paired.

Everything here operates on lightweight feature objects so it is unit-testable
without a BAM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fiberhmm.cli.extract_tags import _build_query_to_ref, _deam_positions_list
from fiberhmm.io.ma_tags import flip_interval_frame, parse_ma_tag

# Deamination flavor codes (match _deam_positions_list: 1 = Y = C->U, 0 = R = G->U).
FLAVOR_CT = 1
FLAVOR_GA = 0
_FLAVOR_NAME = {FLAVOR_CT: 'CT', FLAVOR_GA: 'GA'}


@dataclass(frozen=True)
class PairParams:
    """Tunables for cross-strand pairing.

    grid_bp        : reference resolution (bp) of the dyad-density signal.
    sigma_bp       : Gaussian dyad width; absorbs cross-strand edge variance.
    max_lag_bp     : +/- register shift searched when correlating (a whole-array
                     phase offset should not be penalized).
    min_overlap_bp : minimum genomic overlap for a pair to be scorable.
    min_nucs       : minimum dyads *within the overlap* on each read.
    min_score      : minimum cross-correlation to accept a pair.
    min_margin     : minimum (best - second_best) on both reads to accept
                     (the ambiguity gate; below this the locus is unresolved).
    """
    grid_bp: int = 10
    sigma_bp: float = 30.0
    max_lag_bp: int = 60
    min_overlap_bp: int = 1500
    min_nucs: int = 4
    min_score: float = 0.5
    min_margin: float = 0.05


@dataclass
class ReadFeat:
    """Per-read features for pairing (one primary alignment)."""
    index: int              # position in the batch (stable id)
    name: str               # query_name
    flavor: int             # FLAVOR_CT / FLAVOR_GA
    ref_start: int          # 0-based, inclusive
    ref_end: int            # 0-based, exclusive
    dyads: np.ndarray       # sorted reference dyad-center positions (int64)
    grid0: int              # first grid bin index (ref_start // grid_bp)
    signal: np.ndarray      # float32 dyad-density over [grid0 .. ref_end//grid]

    @property
    def flavor_name(self) -> str:
        return _FLAVOR_NAME[self.flavor]


def _gaussian_kernel(sigma_bp: float, grid_bp: int) -> np.ndarray:
    radius = max(1, int(round(3 * sigma_bp / grid_bp)))
    x = np.arange(-radius, radius + 1) * grid_bp
    return np.exp(-(x * x) / (2 * sigma_bp * sigma_bp)).astype(np.float32)


def read_flavor(read, prob_threshold: int = 0) -> Optional[int]:
    """Dominant deamination flavor of a read, or None if it has no calls.

    DAF reads are ~100% one flavor; ties (empty) return None.
    """
    calls = _deam_positions_list(read, _build_query_to_ref(read), prob_threshold)
    if not calls:
        return None
    ct = sum(1 for _, f in calls if f == FLAVOR_CT)
    ga = len(calls) - ct
    if ct == 0 and ga == 0:
        return None
    return FLAVOR_CT if ct >= ga else FLAVOR_GA


def nuc_dyads_ref(read) -> np.ndarray:
    """Reference-coordinate nucleosome dyad centers from the MA ``nuc`` track.

    MA intervals are in the molecular frame; for reverse reads they are flipped
    to the SEQ (query) frame before the query->reference lookup, matching
    ``fiberhmm-extract``. Returns a sorted int64 array (possibly empty).
    """
    try:
        parsed = parse_ma_tag(read.get_tag('MA'))
    except (KeyError, ValueError):
        return np.empty(0, dtype=np.int64)
    nucs = parsed['nuc']
    if not nucs:
        return np.empty(0, dtype=np.int64)
    q2r = _build_query_to_ref(read)
    n = len(q2r)
    read_len = int(parsed['read_length'])
    is_rev = bool(read.is_reverse)
    out: List[int] = []
    for s, length in nucs:
        if is_rev:
            s, length = flip_interval_frame(int(s), int(length), read_len)
        qc = int(s) + int(length) // 2
        if 0 <= qc < n:
            r = int(q2r[qc])
            if r >= 0:
                out.append(r)
    out.sort()
    return np.asarray(out, dtype=np.int64)


def build_feature(read, index: int, params: PairParams,
                  prob_threshold: int = 0) -> Optional[ReadFeat]:
    """Build a :class:`ReadFeat` for a primary mapped read, or None if unusable.

    None when: unmapped/secondary/supplementary, no sequence, no deamination
    flavor, or fewer than 2 mappable nucleosome dyads (too little pattern).
    """
    if read.is_unmapped or read.is_secondary or read.is_supplementary:
        return None
    if read.query_sequence is None or not read.has_tag('MA'):
        return None
    flavor = read_flavor(read, prob_threshold)
    if flavor is None:
        return None
    dyads = nuc_dyads_ref(read)
    if dyads.size < 2:
        return None
    g = params.grid_bp
    grid0 = int(read.reference_start) // g
    grid1 = int(read.reference_end) // g
    kern = _gaussian_kernel(params.sigma_bp, g)
    krad = len(kern) // 2
    sig = np.zeros(grid1 - grid0 + 1, dtype=np.float32)
    for c in dyads:
        cb = int(c) // g - grid0
        lo = max(0, cb - krad)
        hi = min(len(sig), cb + krad + 1)
        if hi > lo:
            klo = lo - (cb - krad)
            sig[lo:hi] += kern[klo:klo + (hi - lo)]
    return ReadFeat(index=index, name=read.query_name, flavor=flavor,
                    ref_start=int(read.reference_start),
                    ref_end=int(read.reference_end),
                    dyads=dyads, grid0=grid0, signal=sig)


def score_pair(a: ReadFeat, b: ReadFeat, params: PairParams) -> Optional[float]:
    """Lag-tolerant normalized cross-correlation of two dyad signals over their
    genomic overlap, or None if the overlap is too small / under-covered.

    Returns a value in [-1, 1] (the max over lags in +/- max_lag_bp).
    """
    g = params.grid_bp
    lo = max(a.grid0, b.grid0)
    hi = min(a.grid0 + len(a.signal), b.grid0 + len(b.signal))
    if (hi - lo) * g < params.min_overlap_bp:
        return None
    lo_bp, hi_bp = lo * g, hi * g
    if int(np.sum((a.dyads >= lo_bp) & (a.dyads < hi_bp))) < params.min_nucs:
        return None
    if int(np.sum((b.dyads >= lo_bp) & (b.dyads < hi_bp))) < params.min_nucs:
        return None
    sa = a.signal[lo - a.grid0:hi - a.grid0]
    sb = b.signal[lo - b.grid0:hi - b.grid0]
    m = min(len(sa), len(sb))
    maxlag = params.max_lag_bp // g
    if m < 2 * maxlag + 5:
        return None
    sa = sa[:m] - sa[:m].mean()
    sb = sb[:m] - sb[:m].mean()
    na = float(np.sqrt(np.dot(sa, sa)))
    nb = float(np.sqrt(np.dot(sb, sb)))
    if na == 0.0 or nb == 0.0:
        return None
    best = -1.0
    for lag in range(-maxlag, maxlag + 1):
        if lag >= 0:
            x, y = sa[lag:], sb[:len(sb) - lag]
        else:
            x, y = sa[:len(sa) + lag], sb[-lag:]
        k = min(len(x), len(y))
        if k == 0:
            continue
        r = float(np.dot(x[:k], y[:k]) / (na * nb))
        if r > best:
            best = r
    return best


# Pairing status codes written to the ``mt:A`` tag.
STATUS_PAIRED = 'P'       # resolved reciprocal-best pair above score+margin
STATUS_UNRESOLVED = 'U'   # had candidate(s) but failed margin/score gate
STATUS_NONE = '.'         # no overlapping opposite-strand candidate


@dataclass
class PairResult:
    partner: Dict[int, int]        # read index -> mate index (resolved only)
    score: Dict[int, float]        # read index -> pair correlation
    margin: Dict[int, float]       # read index -> min(best-2nd over both reads)
    status: Dict[int, str]         # read index -> STATUS_*


def assign_pairs(feats: Sequence[ReadFeat], params: PairParams) -> PairResult:
    """Resolve cross-strand pairs within one batch (typically one chromosome).

    Reciprocal-best with a two-sided margin gate. ``feats`` may be in any order;
    scoring uses a start-sorted sweep so only genomically overlapping
    opposite-strand pairs are compared.
    """
    n = len(feats)
    by_idx = {f.index: f for f in feats}
    order = sorted(feats, key=lambda f: f.ref_start)
    # best[idx] = (score, partner_idx); second[idx] = score of runner-up
    best: Dict[int, Tuple[float, int]] = {f.index: (-2.0, -1) for f in feats}
    second: Dict[int, float] = {f.index: -2.0 for f in feats}

    def offer(i: int, val: float, j: int) -> None:
        bs, _bi = best[i]
        if val > bs:
            second[i] = bs
            best[i] = (val, j)
        elif val > second[i]:
            second[i] = val

    for ai in range(n):
        a = order[ai]
        a_end = a.ref_end
        for bi in range(ai + 1, n):
            b = order[bi]
            if b.ref_start >= a_end:
                break  # start-sorted: no further overlaps with a
            if b.flavor == a.flavor:
                continue
            s = score_pair(a, b, params)
            if s is None:
                continue
            offer(a.index, s, b.index)
            offer(b.index, s, a.index)

    partner: Dict[int, int] = {}
    score: Dict[int, float] = {}
    margin: Dict[int, float] = {}
    status: Dict[int, str] = {}
    for idx in by_idx:
        bs, bj = best[idx]
        if bj < 0:
            status[idx] = STATUS_NONE
            continue
        # reciprocal check
        recip = best[bj][1] == idx
        marg = min(bs - second[idx], best[bj][0] - second[bj])
        if recip and bs >= params.min_score and marg >= params.min_margin:
            partner[idx] = bj
            score[idx] = bs
            margin[idx] = marg
            status[idx] = STATUS_PAIRED
        else:
            score[idx] = bs
            margin[idx] = marg
            status[idx] = STATUS_UNRESOLVED
    return PairResult(partner=partner, score=score, margin=margin, status=status)
