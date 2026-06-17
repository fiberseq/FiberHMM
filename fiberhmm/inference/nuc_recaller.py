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
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from fiberhmm.inference.circular import circular_intervals_overlap
from fiberhmm.inference.tf_recaller import call_tfs_in_interval, merge_intervals
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


def _fragments_after_cuts(a: int, b: int,
                          cut_spans: Iterable[Tuple[int, int]]) -> List[Interval]:
    frags: List[Interval] = []
    cur = int(a)
    for cs, ce in sorted((int(cs), int(ce)) for cs, ce in cut_spans):
        if cs > cur:
            frags.append((cur, cs))
        cur = max(cur, ce)
    if cur < b:
        frags.append((cur, int(b)))
    return frags


def _call_interval(call) -> Tuple[int, int]:
    return int(call.start), int(call.start + call.length)


def _bounded_interval(start_raw, length_raw, read_length: int, *, clamp_start: bool):
    start = int(start_raw)
    length = int(length_raw)
    if length <= 0:
        return None
    a = max(0, start) if clamp_start else start
    b = min(int(read_length), start + length)
    if b <= a:
        return None
    return a, b


def _linear_intervals_overlap(a_start: int, a_end: int,
                              b_start: int, b_end: int) -> bool:
    return int(a_start) < int(b_end) and int(b_start) < int(a_end)


def _keep_nuc_against_linear_intervals(
    nuc: NucCall,
    intervals: Sequence[Tuple[int, int]],
    unify_threshold: int,
) -> bool:
    if nuc.length <= 0:
        return False
    if nuc.length >= unify_threshold:
        return True
    nuc_start, nuc_end = _call_interval(nuc)
    return not any(
        _linear_intervals_overlap(nuc_start, nuc_end, other_start, other_end)
        for other_start, other_end in intervals
    )


def _keep_nuc_against_circular_intervals(
    nuc: NucCall,
    intervals: Sequence[Tuple[int, int]],
    unify_threshold: int,
    read_length: int,
) -> bool:
    if nuc.length <= 0:
        return False
    if nuc.length >= unify_threshold:
        return True
    iv = (nuc.start, nuc.length)
    return not any(
        circular_intervals_overlap(iv, other, read_length)
        for other in intervals
    )


def _total_call_llr(calls) -> float:
    total_llr = 0.0
    for call in calls:
        total_llr += call.llr
    return total_llr


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
    nuc = NucCall(
        start=cstart,
        length=cend - cstart,
        nq=llr_to_tq(_total_call_llr(prot)),
        el=ambiguity_to_edge(first.left_ambiguity),
        er=ambiguity_to_edge(last.right_ambiguity),
    )
    if cstart > a:
        access.append((a, cstart - a))
    if b > cend:
        access.append((cend, b - cend))
    return nuc, access


def _split_on_accessible_cuts(obs, a, b, nhit, nmiss,
                              split_min_llr, split_min_opps):
    cuts = call_tfs_in_interval(obs, a, b, nhit, nmiss,
                                split_min_llr, split_min_opps)
    cuts = sorted(cuts, key=lambda c: c.start)
    access = [(c.start, c.length) for c in cuts]
    frags = _fragments_after_cuts(
        a, b, ((c.start, c.start + c.length) for c in cuts)
    )
    return frags, access


def _phase_cut_window(a: int, b: int, pred: int, phase_window: int):
    lo = max(a, pred - phase_window)
    hi = min(b, pred + phase_window)
    if hi - lo < 2:
        return None
    return lo, hi


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
        window = _phase_cut_window(a, b, pred, phase_window)
        if window is None:
            continue
        lo, hi = window
        found = call_tfs_in_interval(obs, lo, hi, nhit, nmiss,
                                     phase_min_llr, phase_min_opps)
        if found:
            best = max(found, key=lambda c: c.llr)
            cut_pairs.append((best.start, best.start + best.length))
    if not cut_pairs:
        return [(a, b)], []
    cut_pairs.sort()
    subs = _fragments_after_cuts(a, b, cut_pairs)
    cut_intervals = [(cs, ce - cs) for cs, ce in cut_pairs]
    return subs, cut_intervals


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
    """
    nhit = -llr_hit
    nmiss = -llr_miss
    nucs: List[NucCall] = []
    access: List[Interval] = []

    for s_raw, length_raw in zip(ns, nl):
        span = _bounded_interval(
            s_raw, length_raw, read_length, clamp_start=False,
        )
        if span is None:
            continue
        s, e = span

        # --- SPLIT: accessible runs inside the footprint are cuts ---
        frags, cut_access = _split_on_accessible_cuts(
            obs, s, e, nhit, nmiss, split_min_llr, split_min_opps,
        )
        access.extend(cut_access)

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
        span = _bounded_interval(
            s_raw, length_raw, read_length, clamp_start=True,
        )
        if span is not None:
            iv.append(span)
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
    tf_intervals = [_call_interval(c) for c in tf_calls]
    return [
        nc for nc in nuc_calls
        if _keep_nuc_against_linear_intervals(nc, tf_intervals, unify_threshold)
    ]


def _msp_gaps_between_nucs(
    kept_nucs: Sequence[NucCall],
    span_lo: int,
    span_hi: int,
    floor: int,
) -> List[Interval]:
    msps: List[Interval] = []
    cur = int(span_lo)
    for k in kept_nucs:
        if k.start - cur >= floor:
            msps.append((cur, k.start - cur))
        cur = max(cur, k.start + k.length)
    if int(span_hi) - cur >= floor:
        msps.append((cur, int(span_hi) - cur))
    return msps


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

    return kept, _msp_gaps_between_nucs(kept, span_lo, span_hi, floor)


def _circular_uncovered_cut(calls, read_length: int) -> int:
    covered = np.zeros(read_length, dtype=bool)
    for n in calls:
        s = n.start % read_length
        span = min(n.length, read_length)
        idx = (s + np.arange(span)) % read_length
        covered[idx] = True
    uncovered = np.flatnonzero(~covered)
    return int(uncovered[0]) if uncovered.size else 0


def _rotate_circular_nuc_calls(calls, cut: int, read_length: int) -> List[NucCall]:
    rotated: List[NucCall] = []
    for n in calls:
        rs = (n.start - cut) % read_length
        length = min(n.length, read_length)
        end = rs + length
        if end <= read_length:
            rotated.append(NucCall(rs, length, n.nq, n.el, n.er))
        else:
            # wraps the rotated origin -> split into [rs, rl) and [0, end-rl);
            # each piece keeps its real outer edge, the cut edge byte is zeroed
            # (same convention as split_intervals_for_legacy on a wrapped nuc).
            rotated.append(NucCall(rs, read_length - rs, n.nq, n.el, 0))
            rotated.append(NucCall(0, end - read_length, n.nq, 0, n.er))
    return rotated


def _whole_molecule_msp(read_length: int, floor: int) -> List[Interval]:
    return [(0, read_length)] if read_length >= floor else []


def _whole_molecule_nuc_call(call: NucCall, read_length: int) -> NucCall:
    return NucCall(0, read_length, call.nq, call.el, call.er)


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
        return [], _whole_molecule_msp(rl, floor)
    whole = next((n for n in calls if int(n.length) >= rl), None)
    if whole is not None:
        if rl >= nfloor:
            return [_whole_molecule_nuc_call(whole, rl)], []
        return [], _whole_molecule_msp(rl, floor)

    # Prefer an uncovered cut point (no call straddles it); fall back to 0 when
    # the circle is fully covered. Either way, straddlers are split below, so a
    # fallback origin landing inside a wrapped call is handled correctly.
    cut = _circular_uncovered_cut(calls, rl)
    rotated = _rotate_circular_nuc_calls(calls, cut, rl)
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
    promoted_intervals = [_call_interval(p) for p in promoted]
    return [
        n for n in nuc_calls
        if _keep_nuc_against_linear_intervals(n, promoted_intervals, unify_threshold)
    ]


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
    tf_intervals = [(c.start, c.length) for c in tf_calls]
    return [
        nc for nc in nuc_calls
        if _keep_nuc_against_circular_intervals(
            nc, tf_intervals, unify_threshold, read_length
        )
    ]
