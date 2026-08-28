"""Per-read nucleosome recaller.

Splits over-merged HMM footprints on accessible (m6a/deam) evidence, then resolves
each resulting fragment's edges and quality. Reuses the TF recaller's Kadane kernel
with *inverted* emission tables for splitting and *non-inverted* tables for the
nucleosome edge + quality pass -- no new scoring code. The topology policy adds
an HMM-occupancy constraint for sparse single-strand evidence: cuts must separate
nucleosome-sized pieces, and unresolved edge ambiguity stays protected.

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
                     nuc_min_size, edge_min_llr, edge_min_opps,
                     preserve_fragment=False):
    """Edge-refine one protected fragment into a NucCall (or demote it).

    Returns ``(nuc_or_None, access_intervals)``. A fragment shorter than
    ``nuc_min_size`` is demoted. Under the historical conservative policy, a
    protected core that trims below the floor is also demoted; signal deserts
    retain a quality-0 NucCall. ``preserve_fragment`` instead keeps a qualifying
    HMM fragment and records unresolved edges.
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
    total_llr = sum(p.llr for p in prot)
    if preserve_fragment:
        # Sparse single-strand evidence does not identify conservative edges.
        # The HMM extent remains the occupancy prior; the protected scan still
        # supplies a quality score, while zero edge bytes honestly record that
        # the exact boundaries were not resolved by bracketing evidence.
        return NucCall(
            start=a,
            length=b - a,
            nq=llr_to_tq(total_llr),
            el=0,
            er=0,
        ), access
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
        nq=llr_to_tq(total_llr),
        el=ambiguity_to_edge(first.left_ambiguity),
        er=ambiguity_to_edge(last.right_ambiguity),
    )
    if cstart > a:
        access.append((a, cstart - a))
    if b > cend:
        access.append((cend, b - cend))
    return nuc, access


def _select_nucleosome_separating_cuts(
    cuts,
    start: int,
    end: int,
    nuc_min_size: int,
):
    """Select the maximum-evidence cut chain with nuc-sized pieces throughout.

    The accessible Kadane scan can find isolated ONT events inside a single
    HMM-protected footprint. Treating every such run as a split can shatter one
    nucleosome into sub-floor fragments, after which the old recaller labels the
    entire footprint accessible. A true *nucleosome split* must instead leave a
    possible nucleosome on both outer sides and between consecutive cuts.

    Dynamic programming maximizes retained cut LLR subject to that topology.
    All current call LLRs are positive, so an eligible non-conflicting cut is
    retained unless a stronger incompatible chain exists.
    """
    floor = max(1, int(nuc_min_size))
    eligible = [
        cut
        for cut in sorted(cuts, key=lambda call: call.start)
        if (
            int(cut.start) - int(start) >= floor
            and int(end) - int(cut.start + cut.length) >= floor
        )
    ]
    if not eligible:
        return []

    best_score: List[float] = []
    predecessor: List[int | None] = []
    for i, cut in enumerate(eligible):
        score = float(cut.llr)
        pred = None
        for j in range(i):
            gap = int(cut.start) - int(
                eligible[j].start + eligible[j].length
            )
            candidate = best_score[j] + float(cut.llr)
            if gap >= floor and candidate > score:
                score = candidate
                pred = j
        best_score.append(score)
        predecessor.append(pred)

    cursor: int | None = int(np.argmax(np.asarray(best_score)))
    selected = []
    while cursor is not None:
        selected.append(eligible[cursor])
        cursor = predecessor[cursor]
    return list(reversed(selected))


def _phase_subfragments(obs, a, b, nhit, nmiss, nrl,
                        phase_min_llr, phase_min_opps, phase_window,
                        min_fragment_size=0):
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
    cut_calls = []
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
            cut_calls.append(best)
    if min_fragment_size > 0:
        cut_calls = _select_nucleosome_separating_cuts(
            cut_calls, a, b, min_fragment_size)
    if not cut_calls:
        return [(a, b)], []
    cut_pairs = [
        (int(call.start), int(call.start + call.length))
        for call in cut_calls
    ]
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
    with open(path) as handle:
        d = json.load(handle)
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


def _subtract_calls_from_interval(start, end, calls):
    """Return portions of ``[start,end)`` not occupied by provisional calls."""
    covered = sorted(
        (
            max(int(start), int(call.start)),
            min(int(end), int(call.start) + int(call.length)),
        )
        for call in calls
        if int(call.start) < int(end)
        and int(call.start) + int(call.length) > int(start)
    )
    residue: List[Tuple[int, int]] = []
    cursor = int(start)
    for left, right in covered:
        if left > cursor:
            residue.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < int(end):
        residue.append((cursor, int(end)))
    return residue


def _accessible_configuration_evidence(
    obs,
    intervals,
    llr_hit,
    llr_miss,
    m5c_mask=None,
    m5c_llr_hit=None,
    m5c_llr_miss=None,
):
    """Return ``(opportunities, log BF)`` for accessible vs protected.

    The TF tables encode log P(protected) / P(accessible), so their negative is
    the desired evidence.  Intervals supplied here are the accessible residue
    after subtracting a provisional TF configuration from a radial gap.
    """
    observations = np.asarray(obs)
    hit_table = np.asarray(llr_hit)
    miss_table = np.asarray(llr_miss)
    use_m5c = (
        m5c_mask is not None
        and m5c_llr_hit is not None
        and m5c_llr_miss is not None
    )
    methylated = np.asarray(m5c_mask, dtype=bool) if use_m5c else None
    methylated_hit = np.asarray(m5c_llr_hit) if use_m5c else None
    methylated_miss = np.asarray(m5c_llr_miss) if use_m5c else None
    opportunities = 0
    log_bf = 0.0
    for start, end in intervals:
        lo = max(0, int(start))
        hi = min(len(observations), int(end))
        for position in range(lo, hi):
            code = int(observations[position])
            if 0 <= code < N_CTX:
                table = methylated_hit if use_m5c and methylated[position] else hit_table
                log_bf -= float(table[code])
                opportunities += 1
            elif UNMETH_OFFSET <= code < UNMETH_OFFSET + N_CTX:
                context = code - UNMETH_OFFSET
                table = methylated_miss if use_m5c and methylated[position] else miss_table
                log_bf -= float(table[context])
                opportunities += 1
    return opportunities, log_bf


def validate_radial_access_in_read(
    obs,
    original_ns,
    original_nl,
    radial_nucs,
    provisional_tf_calls,
    read_length,
    llr_hit,
    llr_miss,
    *,
    min_llr,
    min_opps,
    nuc_min_size,
    m5c_mask=None,
    m5c_llr_hit=None,
    m5c_llr_miss=None,
):
    """Validate DddA radial gaps as complete nuc/linker/TF configurations.

    A radial dyad match establishes protected occupancy; absence of radial
    coverage does *not* establish accessibility.  For every part of an
    original HMM nucleosome demoted by the radial pass, subtract provisional TF
    calls and require the remaining observations to support accessible rather
    than protected sequence.  Unsupported outer flanks are restored to the HMM
    boundary, radial calls separated by an unsupported internal boundary are
    merged back into one broad protected block, and an HMM footprint with no
    radial dyad is preserved whole.  Restored or merged calls receive unresolved
    quality where the molecule did not identify exact topology.

    Returns ``(validated_nucleosomes, supported_accessible_intervals)``.  The
    caller should rebuild scan space and rerun TF recall; repeating this pass is
    monotonic because it can only close unsupported gaps.
    """
    nucs = [
        NucCall(int(call.start), int(call.length), int(call.nq),
                int(call.el), int(call.er))
        for call in radial_nucs
        if int(call.length) > 0
    ]
    tf_calls = list(provisional_tf_calls)
    originals = []
    for start_raw, length_raw in zip(original_ns, original_nl):
        start = max(0, int(start_raw))
        end = min(int(read_length), int(start_raw) + int(length_raw))
        if end > start:
            originals.append((start, end))

    assigned = [[] for _ in originals]
    unassigned = []
    for index, call in enumerate(nucs):
        center = int(call.start) + int(call.length) // 2
        owner = next(
            (i for i, (start, end) in enumerate(originals)
             if start <= center < end),
            None,
        )
        if owner is None:
            unassigned.append(call)
        else:
            assigned[owner].append(call)

    validated: List[NucCall] = list(unassigned)
    supported_access: List[Interval] = []
    score_threshold = float(min_llr)
    opportunity_threshold = max(1, int(min_opps))

    def supports_access(start, end):
        if end <= start:
            return False, 0
        residue = _subtract_calls_from_interval(start, end, tf_calls)
        opportunities, log_bf = _accessible_configuration_evidence(
            obs,
            residue,
            llr_hit,
            llr_miss,
            m5c_mask=m5c_mask,
            m5c_llr_hit=m5c_llr_hit,
            m5c_llr_miss=m5c_llr_miss,
        )
        supported = (
            opportunities >= opportunity_threshold
            and log_bf >= score_threshold
        )
        return supported, (llr_to_tq(max(0.0, log_bf)) if supported else 0)

    for (original_start, original_end), local in zip(originals, assigned):
        local.sort(key=lambda call: (int(call.start), int(call.length)))
        if original_end - original_start < int(nuc_min_size):
            # A sub-nucleosomal HMM footprint was already part of the ordinary
            # TF scan contract.  It is not a failed radial nucleosome and must
            # not be closed merely because no dyad can fit inside it.
            supported_access.append((
                original_start,
                original_end - original_start,
            ))
            continue
        if not local:
            validated.append(NucCall(
                original_start,
                original_end - original_start,
                nq=0,
                el=0,
                er=0,
            ))
            continue

        raw_starts = [int(call.start) for call in local]
        raw_ends = [int(call.start) + int(call.length) for call in local]
        # The HMM-accessible scan space is a valid baseline.  Radial edge
        # refinement may subdivide an HMM nucleosome, but it must not expand a
        # recalled nucleosome into an original HMM MSP and erase ordinary TF
        # hypotheses there.
        segments = []
        for call, raw_start, raw_end in zip(local, raw_starts, raw_ends):
            start = max(original_start, raw_start)
            end = min(original_end, raw_end)
            if end - start < int(nuc_min_size):
                continue
            segments.append(NucCall(
                start,
                end - start,
                int(call.nq),
                0 if start != raw_start else int(call.el),
                0 if end != raw_end else int(call.er),
            ))
        if not segments:
            validated.append(NucCall(
                original_start,
                original_end - original_start,
                nq=0,
                el=0,
                er=0,
            ))
            continue

        left_gap_end = min(original_end, int(segments[0].start))
        if left_gap_end > original_start:
            left_supported, left_quality = supports_access(
                original_start, left_gap_end,
            )
            if left_supported:
                supported_access.append((original_start, left_gap_end - original_start))
                first = segments[0]
                segments[0] = NucCall(
                    int(first.start),
                    int(first.length),
                    min(int(first.nq), int(left_quality)),
                    int(first.el),
                    int(first.er),
                )
            else:
                first = segments[0]
                segments[0] = NucCall(
                    original_start,
                    int(first.start) + int(first.length) - original_start,
                    int(first.nq),
                    0,
                    int(first.er),
                )

        # A radial dyad match alone cannot identify a boundary.  Retain two
        # nucleosomes only when the complete intervening configuration has
        # direct linker evidence; otherwise merge them back into one unresolved
        # protected HMM block rather than inventing an adjacent boundary.
        merged_segments: List[NucCall] = []
        current = segments[0]
        for next_call in segments[1:]:
            current_end = int(current.start) + int(current.length)
            next_start = int(next_call.start)
            gap_start = max(original_start, current_end)
            gap_end = min(original_end, next_start)
            split_supported, split_quality = supports_access(gap_start, gap_end)
            if gap_end > gap_start and split_supported:
                supported_access.append((gap_start, gap_end - gap_start))
                merged_segments.append(NucCall(
                    int(current.start),
                    int(current.length),
                    min(int(current.nq), int(split_quality)),
                    int(current.el),
                    int(current.er),
                ))
                current = NucCall(
                    int(next_call.start),
                    int(next_call.length),
                    min(int(next_call.nq), int(split_quality)),
                    int(next_call.el),
                    int(next_call.er),
                )
            else:
                merged_start = min(int(current.start), int(next_call.start))
                merged_end = max(
                    int(current.start) + int(current.length),
                    int(next_call.start) + int(next_call.length),
                )
                current = NucCall(
                    merged_start,
                    merged_end - merged_start,
                    nq=0,
                    el=int(current.el),
                    er=int(next_call.er),
                )
        merged_segments.append(current)

        right_gap_start = max(
            original_start,
            int(merged_segments[-1].start) + int(merged_segments[-1].length),
        )
        if original_end > right_gap_start:
            right_supported, right_quality = supports_access(
                right_gap_start, original_end,
            )
            if right_supported:
                supported_access.append((right_gap_start, original_end - right_gap_start))
                last = merged_segments[-1]
                merged_segments[-1] = NucCall(
                    int(last.start),
                    int(last.length),
                    min(int(last.nq), int(right_quality)),
                    int(last.el),
                    int(last.er),
                )
            else:
                last = merged_segments[-1]
                merged_segments[-1] = NucCall(
                    int(last.start),
                    original_end - int(last.start),
                    int(last.nq),
                    int(last.el),
                    0,
                )

        validated.extend(merged_segments)

    validated.sort(key=lambda call: (int(call.start), int(call.length)))
    merged_access = merge_intervals([
        (int(start), int(start) + int(length))
        for start, length in supported_access
        if int(length) > 0
    ])
    return validated, [(start, end - start) for start, end in merged_access]


def exclude_nucleosomes_from_msps(msps, nuc_calls, msp_min_size):
    """Subtract called nucleosomes from candidate MSP scan intervals."""
    nuc_intervals = merge_intervals([
        (int(call.start), int(call.start) + int(call.length))
        for call in nuc_calls
        if int(call.length) > 0
    ])
    result: List[Interval] = []
    floor = max(1, int(msp_min_size))
    for start_raw, length_raw in msps:
        start = int(start_raw)
        end = start + int(length_raw)
        cursor = start
        for left, right in nuc_intervals:
            if right <= cursor:
                continue
            if left >= end:
                break
            if left > cursor and left - cursor >= floor:
                result.append((cursor, left - cursor))
            cursor = max(cursor, right)
            if cursor >= end:
                break
        if end - cursor >= floor:
            result.append((cursor, end - cursor))
    return result


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
    recall_policy: str = "conservative",
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

    ``recall_policy="conservative"`` preserves the historical behavior: every
    qualifying accessible run becomes a cut and protected evidence defines
    conservative nucleosome edges. ``recall_policy="topology"`` is intended for
    sparse single-strand evidence such as Nanopore m6A: a cut is accepted only
    when every resulting piece can contain a nucleosome, and post-cut fragments
    retain their HMM extent instead of converting unresolved edge ambiguity to
    accessibility.
    """
    if recall_policy not in {"conservative", "topology"}:
        raise ValueError(
            "recall_policy must be 'conservative' or 'topology', got "
            f"{recall_policy!r}"
        )
    if nuc_profile is not None:
        return radial_split_in_read(obs, ns, nl, read_length,
                                    nuc_profile, nuc_min_size)
    topology_policy = recall_policy == "topology"
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
        if topology_policy:
            cuts = _select_nucleosome_separating_cuts(
                cuts, s, e, nuc_min_size)
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
                    phase_min_llr, phase_min_opps, phase_window,
                    min_fragment_size=(
                        nuc_min_size if topology_policy else 0
                    ))
                access.extend(phase_cuts)
            else:
                subs = [(a, b)]
            # --- EDGES + quality per (sub)fragment (protected Kadane, +llr) ---
            for sa, sb in subs:
                nuc, acc = _refine_fragment(
                    obs, sa, sb, llr_hit, llr_miss,
                    nuc_min_size, edge_min_llr, edge_min_opps,
                    preserve_fragment=topology_policy)
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
                           nuc_min_size, edge_min_llr=2.0, edge_min_opps=2,
                           preserve_fragment=False):
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
                                      edge_min_llr, edge_min_opps,
                                      preserve_fragment=preserve_fragment)
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
