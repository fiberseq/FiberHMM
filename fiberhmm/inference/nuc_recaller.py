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


@dataclass(frozen=True)
class _BoundedInterval:
    start: int
    end: int


@dataclass(frozen=True)
class _NucRecallParams:
    split_min_llr: float
    split_min_opps: int
    nuc_min_size: int
    edge_min_llr: float
    edge_min_opps: int
    phase_nrl: int
    phase_min_llr: float
    phase_min_opps: int
    phase_window: int


@dataclass(frozen=True)
class _NucRecallTables:
    nhit: np.ndarray
    nmiss: np.ndarray
    llr_hit: np.ndarray
    llr_miss: np.ndarray


@dataclass(frozen=True)
class _NucRecallResult:
    nucs: List[NucCall]
    access: List[Interval]


@dataclass(frozen=True)
class _RefinedFragment:
    nuc: NucCall | None
    access: List[Interval]


@dataclass(frozen=True)
class _AccessibleSplit:
    fragments: List[Interval]
    access: List[Interval]


@dataclass(frozen=True)
class _PhaseSplit:
    fragments: List[Interval]
    cuts: List[Interval]


@dataclass(frozen=True)
class _PhaseCutWindow:
    start: int
    end: int


@dataclass(frozen=True)
class _TilingFloors:
    msp: int
    nuc: int


@dataclass(frozen=True)
class _CircularTilingFrame:
    nucs: List[NucCall]
    msps: List[Interval]


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


def _bounded_interval(
    start_raw,
    length_raw,
    read_length: int,
    *,
    clamp_start: bool,
) -> _BoundedInterval | None:
    start = int(start_raw)
    length = int(length_raw)
    if length <= 0:
        return None
    a = max(0, start) if clamp_start else start
    b = min(int(read_length), start + length)
    if b <= a:
        return None
    return _BoundedInterval(start=a, end=b)


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


def _nuc_from_protected_calls(calls, nuc_min_size: int):
    ordered = sorted(calls, key=lambda p: p.start)
    first, last = ordered[0], ordered[-1]
    cstart = first.start
    cend = last.start + last.length
    if cend - cstart < nuc_min_size:
        return None
    return NucCall(
        start=cstart,
        length=cend - cstart,
        nq=llr_to_tq(_total_call_llr(ordered)),
        el=ambiguity_to_edge(first.left_ambiguity),
        er=ambiguity_to_edge(last.right_ambiguity),
    )


def _residue_intervals_around_nuc(a: int, b: int, nuc: NucCall) -> List[Interval]:
    residues: List[Interval] = []
    nuc_end = nuc.start + nuc.length
    if nuc.start > a:
        residues.append((a, nuc.start - a))
    if b > nuc_end:
        residues.append((nuc_end, b - nuc_end))
    return residues


def _refine_fragment(obs, a, b, llr_hit, llr_miss,
                     nuc_min_size, edge_min_llr,
                     edge_min_opps) -> _RefinedFragment:
    """Edge-refine one protected fragment into a NucCall (or demote it).

    Returns a refined fragment with ``nuc`` and ``access`` fields. A fragment shorter than
    ``nuc_min_size``, with no protected evidence, or whose conservative core
    trims below the floor, is demoted: ``nuc`` is None (signal-desert keeps a
    quality-0 NucCall) and the residue goes to ``access``.
    """
    access: List[Interval] = []
    if b - a < nuc_min_size:
        access.append((a, b - a))
        return _RefinedFragment(nuc=None, access=access)
    prot = call_tfs_in_interval(
        obs,
        a,
        b,
        llr_hit,
        llr_miss,
        edge_min_llr,
        edge_min_opps,
    )
    if not prot:
        # signal-desert fragment: keep raw extent, unknown quality/edges
        return _RefinedFragment(
            nuc=NucCall(
                a,
                b - a,
                nq=0,
                el=0,
                er=0,
            ),
            access=access,
        )
    nuc = _nuc_from_protected_calls(prot, nuc_min_size)
    if nuc is None:
        # Edge refinement trimmed the protected core below the floor (a sparse
        # protected island) -> not a nucleosome, demote the whole fragment.
        access.append((a, b - a))
        return _RefinedFragment(nuc=None, access=access)
    access.extend(_residue_intervals_around_nuc(a, b, nuc))
    return _RefinedFragment(nuc=nuc, access=access)


def _split_on_accessible_cuts(obs, a, b, nhit, nmiss,
                              split_min_llr, split_min_opps) -> _AccessibleSplit:
    cuts = call_tfs_in_interval(
        obs,
        a,
        b,
        nhit,
        nmiss,
        split_min_llr,
        split_min_opps,
    )
    cuts = sorted(cuts, key=lambda c: c.start)
    access = [(c.start, c.length) for c in cuts]
    frags = _fragments_after_cuts(
        a,
        b,
        ((c.start, c.start + c.length) for c in cuts),
    )
    return _AccessibleSplit(fragments=frags, access=access)


def _phase_cut_window(
    a: int,
    b: int,
    pred: int,
    phase_window: int,
) -> _PhaseCutWindow | None:
    lo = max(a, pred - phase_window)
    hi = min(b, pred + phase_window)
    if hi - lo < 2:
        return None
    return _PhaseCutWindow(start=lo, end=hi)


def _phase_subfragments(obs, a, b, nhit, nmiss, nrl,
                        phase_min_llr, phase_min_opps,
                        phase_window) -> _PhaseSplit:
    """Evidence-gated periodicity split of a long protected fragment.

    A fragment of length L >= 1.5*nrl is assumed to hold ``n = round(L/nrl)``
    nucleosomes. At each predicted internal linker (evenly spaced to fit L), scan
    a +-``phase_window`` bp window for an accessible run with the LOWERED
    ``phase_min_llr`` threshold; the strongest qualifying run becomes a cut.
    Returns phase-split fragments plus cut intervals; with no qualifying cut
    the fragment is returned whole (never split into a signal-desert).
    """
    L = b - a
    if L < int(1.5 * nrl):
        return _PhaseSplit(fragments=[(a, b)], cuts=[])
    n = int(round(L / float(nrl)))
    if n < 2:
        return _PhaseSplit(fragments=[(a, b)], cuts=[])
    spacing = L / float(n)
    cut_pairs: List[Interval] = []
    for i in range(1, n):
        pred = a + int(round(i * spacing))
        window = _phase_cut_window(
            a,
            b,
            pred,
            phase_window,
        )
        if window is None:
            continue
        found = call_tfs_in_interval(
            obs,
            window.start,
            window.end,
            nhit,
            nmiss,
            phase_min_llr,
            phase_min_opps,
        )
        if found:
            best = max(found, key=lambda c: c.llr)
            cut_pairs.append((best.start, best.start + best.length))
    if not cut_pairs:
        return _PhaseSplit(fragments=[(a, b)], cuts=[])
    cut_pairs.sort()
    subs = _fragments_after_cuts(a, b, cut_pairs)
    cut_intervals = [(cs, ce - cs) for cs, ce in cut_pairs]
    return _PhaseSplit(fragments=subs, cuts=cut_intervals)


def _phase_or_unsplit_subfragments(
    obs,
    a: int,
    b: int,
    nhit,
    nmiss,
    phase_nrl: int,
    phase_min_llr: float,
    phase_min_opps: int,
    phase_window: int,
) -> _PhaseSplit:
    if phase_nrl > 0:
        return _phase_subfragments(
            obs,
            a,
            b,
            nhit,
            nmiss,
            phase_nrl,
            phase_min_llr,
            phase_min_opps,
            phase_window,
        )
    return _PhaseSplit(fragments=[(a, b)], cuts=[])


def _recall_nuc_span(
    obs,
    s: int,
    e: int,
    tables: _NucRecallTables,
    params: _NucRecallParams,
) -> _NucRecallResult:
    nucs: List[NucCall] = []
    access: List[Interval] = []

    split = _split_on_accessible_cuts(
        obs,
        s,
        e,
        tables.nhit,
        tables.nmiss,
        params.split_min_llr,
        params.split_min_opps,
    )
    access.extend(split.access)

    for a, b in split.fragments:
        phase = _phase_or_unsplit_subfragments(
            obs,
            a,
            b,
            tables.nhit,
            tables.nmiss,
            params.phase_nrl,
            params.phase_min_llr,
            params.phase_min_opps,
            params.phase_window,
        )
        access.extend(phase.cuts)
        for sa, sb in phase.fragments:
            refined = _refine_fragment(
                obs,
                sa,
                sb,
                tables.llr_hit,
                tables.llr_miss,
                params.nuc_min_size,
                params.edge_min_llr,
                params.edge_min_opps,
            )
            if refined.nuc is not None:
                nucs.append(refined.nuc)
            access.extend(refined.access)

    return _NucRecallResult(nucs=nucs, access=access)


def _recall_nuc_params(
    *,
    split_min_llr: float,
    split_min_opps: int,
    nuc_min_size: int,
    edge_min_llr: float,
    edge_min_opps: int,
    phase_nrl: int,
    phase_min_llr: float,
    phase_min_opps: int,
    phase_window: int,
) -> _NucRecallParams:
    return _NucRecallParams(
        split_min_llr=split_min_llr,
        split_min_opps=split_min_opps,
        nuc_min_size=nuc_min_size,
        edge_min_llr=edge_min_llr,
        edge_min_opps=edge_min_opps,
        phase_nrl=phase_nrl,
        phase_min_llr=phase_min_llr,
        phase_min_opps=phase_min_opps,
        phase_window=phase_window,
    )


def _recall_nuc_tables(llr_hit: np.ndarray, llr_miss: np.ndarray) -> _NucRecallTables:
    return _NucRecallTables(
        nhit=-llr_hit,
        nmiss=-llr_miss,
        llr_hit=llr_hit,
        llr_miss=llr_miss,
    )


def _recall_bounded_nuc_spans(
    obs,
    ns: Sequence[int],
    nl: Sequence[int],
    read_length: int,
    tables: _NucRecallTables,
    params: _NucRecallParams,
) -> _NucRecallResult:
    nucs: List[NucCall] = []
    access: List[Interval] = []

    for s_raw, length_raw in zip(ns, nl):
        span = _bounded_interval(
            s_raw,
            length_raw,
            read_length,
            clamp_start=False,
        )
        if span is None:
            continue

        span_result = _recall_nuc_span(
            obs,
            span.start,
            span.end,
            tables,
            params,
        )
        nucs.extend(span_result.nucs)
        access.extend(span_result.access)

    return _NucRecallResult(nucs=nucs, access=access)


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
    tables = _recall_nuc_tables(llr_hit, llr_miss)
    params = _recall_nuc_params(
        split_min_llr=split_min_llr,
        split_min_opps=split_min_opps,
        nuc_min_size=nuc_min_size,
        edge_min_llr=edge_min_llr,
        edge_min_opps=edge_min_opps,
        phase_nrl=phase_nrl,
        phase_min_llr=phase_min_llr,
        phase_min_opps=phase_min_opps,
        phase_window=phase_window,
    )
    result = _recall_bounded_nuc_spans(
        obs,
        ns,
        nl,
        read_length,
        tables,
        params,
    )
    return result.nucs, result.access


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
    for s_raw, length_raw in (
        list(original_msps) + list(accessible_from_splits)
    ):
        span = _bounded_interval(
            s_raw,
            length_raw,
            read_length,
            clamp_start=True,
        )
        if span is not None:
            iv.append((span.start, span.end))
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


def _ordered_positive_nuc_calls(nuc_calls) -> List[NucCall]:
    return sorted((n for n in nuc_calls if n.length > 0),
                  key=lambda n: (n.start, -(n.start + n.length)))


def _tiling_floors(msp_min_size, nuc_min_size) -> _TilingFloors:
    return _TilingFloors(
        msp=max(1, int(msp_min_size)),
        nuc=max(1, int(nuc_min_size)),
    )


def _clip_ordered_nuc_calls_for_tiling(
    ordered: Sequence[NucCall],
    span_lo: int,
    nuc_floor: int,
) -> List[NucCall]:
    kept: List[NucCall] = []
    last_end = int(span_lo)
    for n in ordered:
        s = n.start
        e = n.start + n.length
        el = n.el
        if s < last_end:          # overlaps the previous nucleosome
            s = last_end
            el = 0                # clipped left edge is no longer meaningful
        if e - s < nuc_floor:
            continue              # swallowed, or clipped below the nuc floor
        kept.append(NucCall(s, e - s, n.nq, el, n.er))
        last_end = e
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
    floors = _tiling_floors(msp_min_size, nuc_min_size)
    ordered = _ordered_positive_nuc_calls(nuc_calls)
    kept = _clip_ordered_nuc_calls_for_tiling(
        ordered,
        span_lo,
        floors.nuc,
    )

    return kept, _msp_gaps_between_nucs(
        kept,
        span_lo,
        span_hi,
        floors.msp,
    )


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


def _whole_molecule_nuc_candidate(calls, read_length: int):
    return next((n for n in calls if int(n.length) >= int(read_length)), None)


def _restore_circular_tiling_frame(
    kept_rot,
    msp_rot,
    cut: int,
    read_length: int,
) -> _CircularTilingFrame:
    kept = sorted(
        (
            NucCall(
                (k.start + cut) % read_length,
                k.length,
                k.nq,
                k.el,
                k.er,
            )
            for k in kept_rot
        ),
        key=lambda n: n.start,
    )
    msps = sorted(((s + cut) % read_length, length) for s, length in msp_rot)
    return _CircularTilingFrame(nucs=kept, msps=msps)


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
    floors = _tiling_floors(msp_min_size, nuc_min_size)
    calls = [n for n in nuc_calls if n.length > 0]
    if rl <= 0:
        return list(calls), []
    if not calls:
        # no nucleosomes -> the entire molecule tiles as one accessible MSP
        return [], _whole_molecule_msp(rl, floors.msp)
    whole = _whole_molecule_nuc_candidate(calls, rl)
    if whole is not None:
        if rl >= floors.nuc:
            return [_whole_molecule_nuc_call(whole, rl)], []
        return [], _whole_molecule_msp(rl, floors.msp)

    # Prefer an uncovered cut point (no call straddles it); fall back to 0 when
    # the circle is fully covered. Either way, straddlers are split below, so a
    # fallback origin landing inside a wrapped call is handled correctly.
    cut = _circular_uncovered_cut(calls, rl)
    rotated = _rotate_circular_nuc_calls(calls, cut, rl)
    kept_rot, msp_rot = assemble_nuc_msp_tiling(
        rotated,
        span_lo=0,
        span_hi=rl,
        msp_min_size=msp_min_size,
        nuc_min_size=nuc_min_size,
    )

    restored = _restore_circular_tiling_frame(kept_rot, msp_rot, cut, rl)
    return restored.nucs, restored.msps


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


def _promoted_nuc_from_tf_call(
    call,
    obs,
    llr_hit,
    llr_miss,
    nuc_min_size,
    edge_min_llr,
    edge_min_opps,
):
    refined = _refine_fragment(
        obs,
        call.start,
        call.start + call.length,
        llr_hit,
        llr_miss,
        nuc_min_size,
        edge_min_llr,
        edge_min_opps,
    )
    return refined.nuc


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
            nuc = _promoted_nuc_from_tf_call(
                c,
                obs,
                llr_hit,
                llr_miss,
                nuc_min_size,
                edge_min_llr,
                edge_min_opps,
            )
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
