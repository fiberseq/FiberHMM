"""Per-read nucleosome recaller.

Splits over-merged HMM footprints on accessible (m6a/deam) evidence, then refines
each resulting fragment's edges and quality. Reuses the TF recaller's Kadane kernel
with *inverted* emission tables for splitting and *non-inverted* tables for the
nucleosome edge + quality pass -- no new scoring code.

  SPLIT:  call_tfs_in_interval(obs, ..., -llr_hit, -llr_miss)  over a footprint
          interior -> accessible runs == cuts. Footprint is split at the cuts.
  EDGES:  call_tfs_in_interval(obs, ..., +llr_hit, +llr_miss)  over each resulting
          fragment -> protected call whose conservative start/length trims the
          Viterbi overshoot, whose cumulative LLR -> nq, whose left/right
          ambiguity -> el/er (conservative+loose edge convention, same as tf+QQQ).

Design notes: nuc_recaller_collab/DESIGN.md (esp. §7b). The split is evidence-only
(no size prior); DddB recovers ~20-30% of buried linkers, which is the accepted
floor for an under-deaminating enzyme. Fiber-seq / DddA give the kernel much more
signal per read.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from fiberhmm.inference.tf_recaller import (
    N_CTX,
    UNMETH_OFFSET,
    call_tfs_in_interval,
    merge_intervals,
)
from fiberhmm.io.ma_tags import ambiguity_to_edge, llr_to_tq

Interval = Tuple[int, int]


@dataclass
class NucCall:
    """A refined nucleosome before conversion to MA/AQ (nuc+QQQ) output."""
    start: int       # query coord, 0-based, inclusive (conservative edge)
    length: int      # bp (conservative span)
    nq: int          # quality byte from cumulative protected LLR (0-255)
    el: int          # left-edge sharpness byte (0-255; 255 = sharp)
    er: int          # right-edge sharpness byte


def _refine_fragment(obs, a, b, llr_hit, llr_miss,
                     nuc_min_size, edge_min_llr, edge_min_opps):
    """Edge-refine one protected fragment into a NucCall (or demote it).

    Returns ``(nuc_or_None, access_intervals)``. A fragment shorter than
    ``nuc_min_size``, with no protected evidence, or whose conservative core
    trims below the floor, is demoted: ``nuc`` is None (signal-desert keeps a
    quality-0 NucCall) and the residue goes to ``access``.
    """
    access: List[Interval] = []
    if b - a < nuc_min_size:
        access.append((a, b - a))
        return None, access
    prot = call_tfs_in_interval(obs, a, b, llr_hit, llr_miss,
                                edge_min_llr, edge_min_opps)
    if not prot:
        # signal-desert fragment: keep raw extent, unknown quality/edges
        return NucCall(a, b - a, nq=0, el=0, er=0), access
    prot = sorted(prot, key=lambda p: p.start)
    first, last = prot[0], prot[-1]
    cstart = first.start
    cend = last.start + last.length
    if cend - cstart < nuc_min_size:
        # Edge refinement trimmed the protected core below the floor (a sparse
        # protected island) -> not a nucleosome, demote the whole fragment.
        access.append((a, b - a))
        return None, access
    total_llr = 0.0
    for p in prot:
        total_llr += p.llr
    nuc = NucCall(
        start=cstart,
        length=cend - cstart,
        nq=llr_to_tq(total_llr),
        el=ambiguity_to_edge(first.left_ambiguity),
        er=ambiguity_to_edge(last.right_ambiguity),
    )
    if cstart > a:
        access.append((a, cstart - a))
    if b > cend:
        access.append((cend, b - cend))
    return nuc, access


def _phase_subfragments(obs, a, b, nhit, nmiss, nrl,
                        phase_min_llr, phase_min_opps, phase_window):
    """Evidence-gated periodicity split of a long protected fragment.

    A fragment of length L >= 1.5*nrl is assumed to hold ``n = round(L/nrl)``
    nucleosomes. At each predicted internal linker (evenly spaced to fit L), scan
    a +-``phase_window`` bp window for an accessible run with the LOWERED
    ``phase_min_llr`` threshold; the strongest qualifying run becomes a cut.
    Returns ``(subfragments, cut_intervals)``; with no qualifying cut the
    fragment is returned whole (never split into a signal-desert).
    """
    L = b - a
    if L < int(1.5 * nrl):
        return [(a, b)], []
    n = int(round(L / float(nrl)))
    if n < 2:
        return [(a, b)], []
    spacing = L / float(n)
    cut_pairs: List[Interval] = []
    for i in range(1, n):
        pred = a + int(round(i * spacing))
        lo = max(a, pred - phase_window)
        hi = min(b, pred + phase_window)
        if hi - lo < 2:
            continue
        found = call_tfs_in_interval(obs, lo, hi, nhit, nmiss,
                                     phase_min_llr, phase_min_opps)
        if found:
            best = max(found, key=lambda c: c.llr)
            cut_pairs.append((best.start, best.start + best.length))
    if not cut_pairs:
        return [(a, b)], []
    cut_pairs.sort()
    subs: List[Interval] = []
    cur = a
    for cs, ce in cut_pairs:
        if cs > cur:
            subs.append((cur, cs))
        cur = max(cur, ce)
    if cur < b:
        subs.append((cur, b))
    cut_intervals = [(cs, ce - cs) for cs, ce in cut_pairs]
    return subs, cut_intervals


# ===================================================================== #
#  DddA radial split (nuc_profile mode)                                  #
#                                                                        #
#  DddA deaminates *inside* nucleosomes, so the accessible-cut split     #
#  above shatters them. Instead, match-filter a single-nucleosome RADIAL #
#  template (deam rate vs distance-from-dyad) to place dyads, then put   #
#  edges at the protected->linker density transition. See                #
#  ddda_profile/ for the derivation. The rotational (~10.3 bp) signal is #
#  real but too small (+-1%) to aid boundaries, so it is not used.       #
# ===================================================================== #

@dataclass(frozen=True)
class NucProfile:
    """Empirical within-nucleosome deamination radial template (DddA mode)."""
    radial: np.ndarray       # deam rate vs |offset from dyad|, index 0..half
    linker: float            # flat linker deam rate
    half: int                # nucleosome footprint half-extent (bp)
    min_sep: int             # min dyad-dyad separation (bp)
    edge_frac: float         # threshold (x linker) for the edge crossing


def load_nuc_profile(path: str) -> NucProfile:
    import json
    d = json.load(open(path))
    return NucProfile(
        radial=np.asarray(d['dyad_rate'], dtype=np.float64),
        linker=float(d['linker']),
        half=int(d.get('half', 73)),
        min_sep=int(d.get('min_sep', 150)),
        edge_frac=float(d.get('edge_frac', 0.82)),
    )


def _obs_opp_deam(obs: np.ndarray):
    """Opportunity (target) and deaminated (accessible 'hit') masks from obs."""
    obs = np.asarray(obs)
    hit = (obs >= 0) & (obs < N_CTX)
    miss = (obs >= UNMETH_OFFSET) & (obs < UNMETH_OFFSET + N_CTX)
    return (hit | miss), hit


def _smoothed_deam_rate(opp, deam, win=21):
    """Per-bp deam rate over a centered window of opportunities (~2 turns, so the
    rotational ripple averages out)."""
    k = np.ones(win)
    num = np.convolve(deam.astype(float), k, 'same')
    den = np.convolve(opp.astype(float), k, 'same')
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def _profile_weights(profile: NucProfile):
    """Signed per-offset log-weights (index = offset + half) for the dyad LLR."""
    half = profile.half
    off = np.arange(-half, half + 1)
    r = np.array([profile.radial[min(abs(d), len(profile.radial) - 1)] for d in off])
    t = np.clip(np.nan_to_num(r, nan=0.05), 0.01, 0.6)
    w1 = np.log(t / profile.linker)
    w0 = np.log((1 - t) / (1 - profile.linker))
    return w1, w0


def _dyad_llr_full(opp, deam, w1, w0):
    """Per-position LLR that a nucleosome is centered there, for the whole read.

    LLR(c) = sum_d a[c+d]*w1[d+half] + b[c+d]*w0[d+half] over the +-half window,
    where a = deaminated opportunities, b = protected opportunities. That is a
    cross-correlation of (a, b) with the (w1, w0) template -- vectorized with
    np.correlate (zero-padded edges == the clipped window), ~100x faster than the
    per-candidate Python loop and numerically identical."""
    a = (opp & deam).astype(np.float64)
    b = (opp & ~deam).astype(np.float64)
    return np.correlate(a, w1, 'same') + np.correlate(b, w0, 'same')


def _dyad_llr_track(opp, deam, lo, hi, w1, w0, half):
    """Convenience wrapper: LLR track restricted to ``[lo, hi)``."""
    llr = _dyad_llr_full(opp, deam, w1, w0)
    return np.arange(lo, hi), llr[lo:hi]


def _place_dyads(cs, llr, min_sep):
    """Greedy peak picking: highest positive LLR first, min separation."""
    chosen: List[int] = []
    for idx in np.argsort(llr)[::-1]:
        if llr[idx] <= 0:
            break
        c = int(cs[idx])
        if all(abs(c - cc) >= min_sep for cc in chosen):
            chosen.append(c)
    return sorted(chosen)


def _find_density_edge(sr, c, direction, bound, profile: NucProfile):
    """Scan out from dyad c to the protected->linker crossing; return
    (edge_pos, ambiguity_bp). Ambiguity = transition-band width (0.30L..0.60L);
    for DddA this is the edge byte source (not 'last-miss/first-hit')."""
    L = profile.linker
    mid, lo_t, hi_t = profile.edge_frac * L, 0.30 * L, 0.60 * L
    x = edge = c
    while 0 <= x + direction < len(sr) and direction * (bound - x) > 0:
        x += direction
        if np.isfinite(sr[x]) and sr[x] >= mid:
            edge = x
            break
    else:
        edge = x
    lo = hi = edge
    while lo - direction >= 0 and direction * (lo - c) > 0 and \
            np.isfinite(sr[lo]) and sr[lo] > lo_t:
        lo -= direction
    while 0 <= hi + direction < len(sr) and np.isfinite(sr[hi]) and sr[hi] < hi_t:
        hi += direction
    return edge, abs(hi - lo)


def _radial_split_footprint(sr, s, e, profile, nuc_min_size, llr_full):
    """Place dyads in protected footprint [s,e) and return (NucCalls, access)."""
    half = profile.half
    lo, hi = max(s + 20, 0), e - 20
    cs = np.arange(lo, hi)
    llr = llr_full[lo:hi]
    if len(cs) == 0:
        return [], [(s, e - s)]
    dyads = _place_dyads(cs, llr, profile.min_sep)
    if not dyads:
        return [], [(s, e - s)]
    nucs: List[NucCall] = []
    covered: List[Interval] = []
    for i, c in enumerate(dyads):
        lb = (dyads[i - 1] + c) // 2 if i > 0 else c - (half + 37)
        rb = (dyads[i + 1] + c) // 2 if i + 1 < len(dyads) else c + (half + 37)
        eL, ambL = _find_density_edge(sr, c, -1, max(0, lb), profile)
        eR, ambR = _find_density_edge(sr, c, +1, min(len(sr) - 1, rb), profile)
        if eR - eL < nuc_min_size:
            continue                       # sub-floor -> stays accessible
        peak = float(llr[int(np.argmin(np.abs(cs - c)))])
        nucs.append(NucCall(eL, eR - eL, llr_to_tq(max(0.0, peak)),
                            ambiguity_to_edge(ambL), ambiguity_to_edge(ambR)))
        covered.append((eL, eR))
    # accessible = footprint minus the emitted nucleosomes
    access: List[Interval] = []
    cur = s
    for a, b in sorted(covered):
        if a > cur:
            access.append((cur, a - cur))
        cur = max(cur, b)
    if e > cur:
        access.append((cur, e - cur))
    return nucs, access


def radial_split_in_read(obs, ns, nl, read_length, profile, nuc_min_size):
    """DddA nucleosome recall: match-filter each HMM footprint into nucleosomes
    + accessible residue. Same return contract as ``recall_nucs_in_read``."""
    opp, deam = _obs_opp_deam(obs)
    sr = _smoothed_deam_rate(opp, deam)
    w1, w0 = _profile_weights(profile)
    llr_full = _dyad_llr_full(opp, deam, w1, w0)   # once per read (vectorized)
    nucs: List[NucCall] = []
    access: List[Interval] = []
    for s_raw, length_raw in zip(ns, nl):
        s = int(s_raw)
        length = int(length_raw)
        if length <= 0:
            continue
        e = min(s + length, read_length)
        if e - s < nuc_min_size:
            access.append((s, e - s))      # too short to be a nucleosome
            continue
        fn, fa = _radial_split_footprint(
            sr, s, e, profile, nuc_min_size, llr_full)
        nucs.extend(fn)
        access.extend(fa)
    return nucs, access


def recall_nucs_in_read(
    obs: np.ndarray,
    ns: Sequence[int],
    nl: Sequence[int],
    read_length: int,
    llr_hit: np.ndarray,
    llr_miss: np.ndarray,
    *,
    split_min_llr: float,
    split_min_opps: int,
    nuc_min_size: int,
    edge_min_llr: float = 2.0,
    edge_min_opps: int = 2,
    phase_nrl: int = 0,
    phase_min_llr: float = 1.0,
    phase_min_opps: int = 1,
    phase_window: int = 35,
    nuc_profile: NucProfile | None = None,
) -> Tuple[List[NucCall], List[Interval]]:
    """Split + edge-refine the footprints (``ns``/``nl``) of one read.

    Returns ``(nuc_calls, accessible_intervals)``:
      - ``nuc_calls``: refined nucleosomes (>= ``nuc_min_size``) with nq/el/er.
      - ``accessible_intervals``: (start, length) patches freed up by splitting
        (the cuts) or trimming (overshoot residue + sub-min-size fragments).
        These feed the MSP re-derivation.

    Pass 1 (``split_min_llr``/``split_min_opps``) is the evidence-driven split:
    accessible runs inside a footprint are cuts. Pass 2 (enabled when
    ``phase_nrl > 0``) is the evidence-gated periodicity prior: a footprint
    longer than ~1.5x the nucleosome repeat length is examined for cuts at
    phase-predicted linker positions using a LOWERED threshold
    (``phase_min_llr`` < ``split_min_llr``). The prior only lowers the bar near
    a predicted linker -- a cut still requires real local evidence there, so a
    signal-desert is never split.

    When ``nuc_profile`` is supplied (DddA mode), the accessible-cut split is
    replaced by a radial template match-filter -- see ``radial_split_in_read``.
    """
    if nuc_profile is not None:
        return radial_split_in_read(obs, ns, nl, read_length,
                                    nuc_profile, nuc_min_size)
    nhit = -llr_hit
    nmiss = -llr_miss
    nucs: List[NucCall] = []
    access: List[Interval] = []

    for s_raw, length_raw in zip(ns, nl):
        s = int(s_raw)
        length = int(length_raw)
        if length <= 0:
            continue
        e = min(s + length, read_length)
        if e <= s:
            continue

        # --- SPLIT: accessible runs inside the footprint are cuts ---
        cuts = call_tfs_in_interval(obs, s, e, nhit, nmiss,
                                    split_min_llr, split_min_opps)
        cuts = sorted(cuts, key=lambda c: c.start)
        for c in cuts:
            access.append((c.start, c.length))

        # --- fragments = footprint minus the cut spans ---
        frags: List[Interval] = []
        cur = s
        for c in cuts:
            cs = c.start
            ce = c.start + c.length
            if cs > cur:
                frags.append((cur, cs))
            cur = max(cur, ce)
        if cur < e:
            frags.append((cur, e))

        # --- Pass 2 (optional): phase-prior split of long fragments ---
        for a, b in frags:
            if phase_nrl > 0:
                subs, phase_cuts = _phase_subfragments(
                    obs, a, b, nhit, nmiss, phase_nrl,
                    phase_min_llr, phase_min_opps, phase_window)
                access.extend(phase_cuts)
            else:
                subs = [(a, b)]
            # --- EDGES + quality per (sub)fragment (protected Kadane, +llr) ---
            for sa, sb in subs:
                nuc, acc = _refine_fragment(
                    obs, sa, sb, llr_hit, llr_miss,
                    nuc_min_size, edge_min_llr, edge_min_opps)
                if nuc is not None:
                    nucs.append(nuc)
                access.extend(acc)

    return nucs, access


def rederive_msps(
    original_msps: Sequence[Interval],
    accessible_from_splits: Sequence[Interval],
    read_length: int,
    msp_min_size: int,
) -> List[Interval]:
    """Re-derive MSPs from the new nucleosome boundaries.

    MSPs after nuc-recall = the original HMM MSPs unioned with the accessible
    patches freed by splitting/trimming, merged, and filtered to
    ``>= msp_min_size``. Returns (start, length) intervals.
    """
    iv: List[Interval] = []
    for s_raw, length_raw in list(original_msps) + list(accessible_from_splits):
        s = int(s_raw)
        length = int(length_raw)
        if length <= 0:
            continue
        a = max(0, s)
        b = min(read_length, s + length)
        if b > a:
            iv.append((a, b))
    merged = merge_intervals(iv)
    floor = max(1, int(msp_min_size))
    return [(a, b - a) for a, b in merged if (b - a) >= floor]


def unify_nuc_calls_with_tf_calls(
    nuc_calls: Sequence[NucCall],
    tf_calls: Sequence,
    unify_threshold: int,
) -> List[NucCall]:
    """Drop short refined nucleosomes overlapped by a TF call (carry nq/el/er).

    Mirrors ``tagging.unify_nucs_with_tf_calls`` but operates on NucCall objects
    so the per-nuc quality bytes survive unification.
    """
    tf_intervals = [(c.start, c.start + c.length) for c in tf_calls]
    kept: List[NucCall] = []
    for nc in nuc_calls:
        if nc.length <= 0:
            continue
        keep = nc.length >= unify_threshold
        if not keep:
            nuc_end = nc.start + nc.length
            keep = not any(ts < nuc_end and te > nc.start
                           for ts, te in tf_intervals)
        if keep:
            kept.append(nc)
    return kept


def assemble_nuc_msp_tiling(nuc_calls, span_lo, span_hi, msp_min_size,
                            nuc_min_size=85):
    """Produce non-overlapping nucleosomes + complementary MSPs that TILE
    ``[span_lo, span_hi)``.

    Splitting, the phase prior and TF->nuc promotion can leave overlapping
    nucleosomes and stale MSPs, but fibertools / FIRE require nucleosomes
    (ns/nl) and MSPs (as/al) to be sorted, non-overlapping, and tiling. This
    clips overlaps and derives MSPs as the gaps between the final nucleosomes.

    Ordering/clipping rules:
      - sort by (start, -end) so the LONGER call at a given start wins; this
        keeps a promoted full-length nucleosome over a short same-start call
        (which would otherwise be clipped, splitting the promoted one back into
        sub-nucleosome pieces).
      - clip the left of an overlapping call to the previous end (zeroing the
        now-meaningless left edge byte), and
      - drop any call that falls below ``nuc_min_size`` after clipping (its span
        reverts to MSP), so no sub-nucleosome nuc+ calls leak out.
    Returns ``(kept_nucs, msp_intervals)``.
    """
    floor = max(1, int(msp_min_size))
    nfloor = max(1, int(nuc_min_size))
    ordered = sorted((n for n in nuc_calls if n.length > 0),
                     key=lambda n: (n.start, -(n.start + n.length)))
    kept = []
    last_end = span_lo
    for n in ordered:
        s = n.start
        e = n.start + n.length
        el = n.el
        if s < last_end:          # overlaps the previous nucleosome
            s = last_end
            el = 0                # clipped left edge is no longer meaningful
        if e - s < nfloor:
            continue              # swallowed, or clipped below the nuc floor
        kept.append(NucCall(s, e - s, n.nq, el, n.er))
        last_end = e

    msps = []
    cur = span_lo
    for k in kept:
        if k.start - cur >= floor:
            msps.append((cur, k.start - cur))
        cur = max(cur, k.start + k.length)
    if span_hi - cur >= floor:
        msps.append((cur, span_hi - cur))
    return kept, msps


def assemble_circular_nuc_msp_tiling(nuc_calls, read_length, msp_min_size,
                                     nuc_min_size=85):
    """Circular-aware ``assemble_nuc_msp_tiling``.

    On a circular molecule a nucleosome can wrap the origin
    (``start + length > read_length``). Running the linear tiler at a fixed
    origin would derive MSP gaps that overlap a wrapped nucleosome's tail (e.g.
    a nuc covering ``[95,100)+[0,15)`` plus a spurious MSP ``[0,95)`` overlapping
    ``[0,15)``). Instead rotate the circle to an origin, split any call that
    still wraps that origin into two linear pieces, tile linearly, then rotate
    the kept nucs and MSPs back. Returns ``(kept_nucs, msp_intervals)`` in
    molecular coordinates.

    Edge cases:
      - no nucleosomes -> the whole molecule is one accessible MSP;
      - fully covered (no uncovered cut point, e.g. overlapping nucs that tile
        the circle): the origin can fall inside a wrapped call, so straddling
        calls are split at the origin before the linear clip -- otherwise the
        tiler would emit overlapping/wrapped pieces.
    """
    rl = int(read_length)
    floor = max(1, int(msp_min_size))
    nfloor = max(1, int(nuc_min_size))
    calls = [n for n in nuc_calls if n.length > 0]
    if rl <= 0:
        return list(calls), []
    if not calls:
        # no nucleosomes -> the entire molecule tiles as one accessible MSP
        return [], ([(0, rl)] if rl >= floor else [])
    whole = next((n for n in calls if int(n.length) >= rl), None)
    if whole is not None:
        if rl >= nfloor:
            return [NucCall(0, rl, whole.nq, whole.el, whole.er)], []
        return [], ([(0, rl)] if rl >= floor else [])

    # Prefer an uncovered cut point (no call straddles it); fall back to 0 when
    # the circle is fully covered. Either way, straddlers are split below, so a
    # fallback origin landing inside a wrapped call is handled correctly.
    covered = np.zeros(rl, dtype=bool)
    for n in calls:
        s = n.start % rl
        span = min(n.length, rl)
        idx = (s + np.arange(span)) % rl
        covered[idx] = True
    uncovered = np.flatnonzero(~covered)
    cut = int(uncovered[0]) if uncovered.size else 0

    rotated = []
    for n in calls:
        rs = (n.start - cut) % rl
        length = min(n.length, rl)
        end = rs + length
        if end <= rl:
            rotated.append(NucCall(rs, length, n.nq, n.el, n.er))
        else:
            # wraps the rotated origin -> split into [rs, rl) and [0, end-rl);
            # each piece keeps its real outer edge, the cut edge byte is zeroed
            # (same convention as split_intervals_for_legacy on a wrapped nuc).
            rotated.append(NucCall(rs, rl - rs, n.nq, n.el, 0))
            rotated.append(NucCall(0, end - rl, n.nq, 0, n.er))
    kept_rot, msp_rot = assemble_nuc_msp_tiling(
        rotated, 0, rl, msp_min_size, nuc_min_size)

    kept = sorted(
        (NucCall((k.start + cut) % rl, k.length, k.nq, k.el, k.er) for k in kept_rot),
        key=lambda n: n.start)
    msps = sorted(((s + cut) % rl, length) for s, length in msp_rot)
    return kept, msps


def drop_short_nucs_overlapping_promoted(nuc_calls, promoted, unify_threshold):
    """Drop short (< ``unify_threshold``) nucleosomes that overlap a promoted one.

    Promotion moves a nucleosome-sized TF call into the nuc set and removes it
    from ``tf_calls``, so ``unify_nuc_calls_with_tf_calls`` no longer drops a
    short nuc that overlapped it. Apply the same rule here against the promoted
    intervals: a short call overlapping a real (promoted) nucleosome is spurious.
    Without this, the start-order tiling can keep the short call and clip/drop
    the promoted one. Returns the filtered nuc list.
    """
    if not promoted:
        return list(nuc_calls)
    pints = [(p.start, p.start + p.length) for p in promoted]
    out = []
    for n in nuc_calls:
        if n.length >= unify_threshold:
            out.append(n)
            continue
        n_end = n.start + n.length
        if any(ps < n_end and n.start < pe for ps, pe in pints):
            continue  # short nuc overlapping a promoted nucleosome -> drop
        out.append(n)
    return out


def promote_large_tf_calls(tf_calls, obs, llr_hit, llr_miss, threshold,
                           nuc_min_size, edge_min_llr=2.0, edge_min_opps=2):
    """Promote nucleosome-sized TF calls (length >= ``threshold``) to NucCalls.

    The TF recaller emits ANY protected run inside an MSP as ``tf+`` with no size
    cap, so a nucleosome the HMM mis-placed in an MSP leaks into the TF track. A
    protected run >= ``threshold`` (``unify_threshold``) is a nucleosome by
    default -- relabel it, computing proper conservative edges via the same
    protected-Kadane edge pass. Returns ``(remaining_tf_calls, promoted_nucs)``.
    """
    remaining = []
    promoted: List[NucCall] = []
    for c in tf_calls:
        if c.length >= threshold:
            nuc, _ = _refine_fragment(obs, c.start, c.start + c.length,
                                      llr_hit, llr_miss, nuc_min_size,
                                      edge_min_llr, edge_min_opps)
            if nuc is not None:
                promoted.append(nuc)
                continue
        remaining.append(c)
    return remaining, promoted


def unify_circular_nuc_calls_with_tf_calls(
    nuc_calls: Sequence[NucCall],
    tf_calls: Sequence,
    unify_threshold: int,
    read_length: int,
) -> List[NucCall]:
    """Circular counterpart of ``unify_nuc_calls_with_tf_calls``.

    Nuc calls and TF calls are in molecular (circular) coordinates; overlap is
    tested with circular-aware segment overlap.
    """
    from fiberhmm.inference.circular import circular_intervals_overlap

    tf_intervals = [(c.start, c.length) for c in tf_calls]
    kept: List[NucCall] = []
    for nc in nuc_calls:
        if nc.length <= 0:
            continue
        keep = nc.length >= unify_threshold
        if not keep:
            iv = (nc.start, nc.length)
            keep = not any(
                circular_intervals_overlap(iv, tfi, read_length)
                for tfi in tf_intervals
            )
        if keep:
            kept.append(nc)
    return kept
