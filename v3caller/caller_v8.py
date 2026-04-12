#!/usr/bin/env python3
"""Nucleosome caller v8 — drop-in replacement for v7 with the merge
step replaced by `poisson_merge_v2_evidence`.

Why v8 instead of patching v7?
------------------------------
The 2026-04-11 overnight benchmark ran every published FiberHMM
caller architecture (v2-v7 + the undergrad profile-guided HMM) on 10
DddA datasets. The single dominant finding was that v7's
`poisson_merge_atoms` over-merges:

  - stock v7 (merge_alpha=0.05, gap_override_opp=10):
      mean median 358 bp, 47 % overmerge rate, score -24.3
  - protected_runs alone (no merge):
      mean median 161 bp, 11.6 % overmerge rate, score 61.4
  - v8 (this file, with `merge_v2_evidence`):
      mean median 174 bp, 11.5 % overmerge rate, **score 66.2**

v8's merge step is the only variant tested that strictly beats
protected_runs alone on every dataset. It does so by:

  1. Using a two-sided Poisson interval test (central 80 %) instead
     of the one-sided "significantly elevated" test. Merge only when
     the gap's hit count is both not unusually low (which would be a
     protected dip) AND not unusually high (which would be a real
     linker).

  2. Imposing a MERGE CASCADE GUARD at 250 bp. This is NOT a hard
     cap on output — Pass 1 is free to emit protected runs of any
     length (up to 2450 bp was observed in scDAF). It only prevents
     the MERGE STEP from creating new mega-atoms via the chain of
     locally-justifiable small merges that plagued v7.

  3. Short-gap structural merge (gap <= 8 bp): always merge, because
     this is shorter than any real biological linker.

  4. Low-opp gap: default to split, not merge. The opposite of v7's
     `gap_override_opp` biological prior, which over-merged on
     low-density data like the embryo DddA samples.

Iteration-7 + iter-8 robustness sweeps (`bench/sweep_v8_params.py`)
revised the gap_cdf defaults from `gap_radius=30, q=0.10/0.90` to
`gap_radius=10, q=0.05/0.95`. The combined optimum lifts the bench
score from 70 → 80 (10 datasets) by reducing oversplit (13 % → 7 %)
without adding overmerge (still 0.3 %). Median footprint lands
exactly on the 180 bp mono-nucleosome target.

All other v7 defaults (W, min_read_rate, min_footprint, the core
`find_pass1_atoms` + `windowed_rate` primitives) are unchanged. If you
want to keep the original v7 behavior, use caller_v7.py.

CLI is the same as v7. The `--merge-alpha`, `--gap-override-length`,
and `--gap-override-opp` flags are kept for compatibility but only
`--merge-alpha` is consulted (the gap-override flags are ignored by
the new merge function). Use `--max-merge-len` to adjust the cascade
guard.

See `phase0/bench/SUMMARY.md` for the full benchmark writeup and
`phase0/bench/output/figure_merge_variants.png` for the visual.
"""

import argparse
import array
import os
import sys

import numpy as np
import pysam
from scipy.stats import poisson

from enzyme_extractors import get_extractor
from caller_v7 import (
    windowed_rate,
    build_ref_to_query_map,
    ref_interval_to_query,
    find_pass1_atoms,
    set_array_tag,
    clear_stale_tags,
    strip_mod_tags,
    STALE_CALL_TAGS,
    MOD_TAGS,
)
from ma_tags import format_ma_tag, format_aq_array


# Extra BAM tags the caller may emit/clear under the new MA schema.
# Both MA and the legacy set can coexist in the same BAM — they are
# independent tag names — so dual output is legal and cheap.
MA_TAGS = ('MA', 'AQ', 'AN')
# New legacy tags we now emit alongside the v7 schema:
#   lq/rq    = per-nucleosome left/right edge-ambiguity quality
#   tn/tl/tq = TF footprint start / length / significance quality
#   mq       = per-nucleosome merge quality (was MA-only before iter-17)
#   el/er    = per-TF left/right edge sharpness quality
NEW_LEGACY_TAGS = ('mq', 'lq', 'rq', 'tn', 'tl', 'tq', 'el', 'er')


def clear_all_call_tags(read):
    """Clear legacy + MA tags before writing fresh calls."""
    for t in STALE_CALL_TAGS:
        if read.has_tag(t):
            read.set_tag(t, None)
    for t in NEW_LEGACY_TAGS:
        if read.has_tag(t):
            read.set_tag(t, None)
    for t in MA_TAGS:
        if read.has_tag(t):
            read.set_tag(t, None)


# -------------------------------------------------------------------
# Alternative Pass 1: simple gap-CDF first pass
# -------------------------------------------------------------------
#
# Original v7 Pass 1 (`find_pass1_atoms`) computes a windowed rate and
# finds runs where the rate dips below the per-read baseline. The 4-2
# generalization benchmark showed that an even simpler first pass —
# "any hit-free stretch >= gap_radius bp" — combined with v8's
# evidence-based merge produces a *slightly better* aggregate score
# than v7-Pass1 + v8 merge:
#
#     gap_cdf+v8       score 71.3   median 155   over 0.3 %
#     protected_runs+v8  score 67.5   median 167   over 11.8 %
#
# The trade-off: gap_cdf has a higher oversplit rate (13 % vs 6 %)
# but essentially zero overmerge. For users whose primary failure mode
# is "over-calling dinucleosomes," gap_cdf is the better Pass 1.
# Selectable via `--first-pass gap_cdf` (default remains
# protected_runs for v7 compatibility).


# -------------------------------------------------------------------
# Edge-ambiguity quality for nucleosome boundaries
# -------------------------------------------------------------------

def compute_edge_q(nuc_s, nuc_e, hit, breathing_window=50):
    """Return (left_q, right_q) ∈ [0, 255] for one ref-frame
    nucleosome interval [nuc_s, nuc_e).

    For each edge, the ambiguity is encoded as the distance from the
    called boundary to the nearest HIT strictly INSIDE the nucleosome:
      - distance 0 bp  → perfectly sharp edge (first hit is right at
        the boundary) → q = 255
      - distance >= breathing_window → q = 0 (deep breathing, the
        edge could be anywhere in that range)
      - linear in between

    We cap the search within the nucleosome to at most
    `breathing_window` bp on each side, so a very long all-protected
    stretch still returns q=0 (honest: "the edge could be anywhere
    in this protected zone").
    """
    L = len(hit)
    # Left edge: look for the first HIT in [nuc_s, nuc_s + window)
    lo_l = max(0, nuc_s)
    hi_l = min(L, nuc_s + breathing_window)
    if hi_l > lo_l:
        hits_left = np.where(hit[lo_l:hi_l] > 0)[0]
        if len(hits_left) > 0:
            left_breath = int(hits_left[0])  # 0..window-1
        else:
            left_breath = breathing_window  # no hit found in window -> max ambiguity
    else:
        left_breath = breathing_window
    # Right edge: look for the LAST HIT in [nuc_e - window, nuc_e)
    lo_r = max(0, nuc_e - breathing_window)
    hi_r = min(L, nuc_e)
    if hi_r > lo_r:
        hits_right = np.where(hit[lo_r:hi_r] > 0)[0]
        if len(hits_right) > 0:
            # Distance from the right edge backwards to the last hit
            right_breath = int((hi_r - lo_r - 1) - hits_right[-1])
        else:
            right_breath = breathing_window
    else:
        right_breath = breathing_window
    left_q = int(round(255 * max(0.0, 1.0 - left_breath / breathing_window)))
    right_q = int(round(255 * max(0.0, 1.0 - right_breath / breathing_window)))
    return max(0, min(255, left_q)), max(0, min(255, right_q))


# -------------------------------------------------------------------
# TF caller — overcall every uninterrupted MISS run, rely on per-call
# quality scores for downstream filtering.
# -------------------------------------------------------------------

# Scaling constant for the TF caller significance score:
#   tq = clip(255 * -log10(P) / _TFP_MAX_NEG_LOG10, 0, 255)
#
# Iter-16e value: 3.0  (saturates at P = 10^-3)
#
# At _TFP_MAX_NEG_LOG10 = 3.0:
#   NAPA baseline 0.30   →  p5=22  p50=36  p95=131  saturated ~0.3%
#   scDAF baseline 0.18  →  p5=13  p50=21  p95=73   saturated ~0.03%
#   ENH30 baseline 0.44  →  p5=30  p50=51  p95=162  saturated ~0.5%
#
# History: 6.0 was the initial guess (useless, almost nothing above
# 128); dropped to 2.0 in iter-16 which used the full 0-255 range but
# saturated ~2-8% of calls at the top, crushing ordering information
# for the strongest TFs. 3.0 is the iter-16e sweet spot — the top of
# the scale spreads across ~170-255 with single-digit saturation,
# while the bottom (2-miss floor) stays around tq ~15-22 so the
# resolution at the weak end is preserved.
#
# "Best guess" filter recommendations by baseline (also in
# best_guess.py):
#     baseline >= 0.2  →  tq >= 22 & el >= 128 & er >= 128
#     baseline <  0.2  →  tq >= 12 & el >= 128 & er >= 128
# Both thresholds correspond to ~2-3 consecutive missed deaminations
# in an MSP, which is the biological signal we want to catch.
_TFP_MAX_NEG_LOG10 = 3.0
# Distance (in bp) from the call edge to the nearest HIT at which
# the edge quality saturates to 0. Below this, linear.
_TF_EDGE_SATURATE = 10

# Rotational phase correction (iter-17, calibrated 2026-04-12 from
# scDAF edge-anchored profile, see project_rotational_calibration.md).
# DAF deaminases are blocked by the histone-contacting face of wrapped
# DNA, creating a 10.4 bp oscillation in hit rate near nuc edges.
# rate_profile(d) = 1 + _ROT_AMP * cos(2π*d/10.4) * exp(-d/_ROT_TAU)
# where d = distance from the nearest nuc edge in bp.
# Only applied to DAF enzymes; Hia5 path is unaffected.
_ROT_AMP = 0.35       # half the peak-trough swing (calibrated)
_ROT_PERIOD = 10.4    # DNA helical pitch in bp
_ROT_TAU = 15.0       # decay constant in bp (gone by d ≈ 30)
_ROT_CUTOFF = 30      # skip correction for opps farther than this


def call_tfs_overcall(msps, opp, hit, baseline, min_misses=2,
                       nuc_edges=None, nuc_edge_quals=None):
    """Emit every uninterrupted MISS run inside the supplied MSPs as
    a TF candidate. No hardcoded size or score thresholds — users
    filter post-hoc using the per-call quality scores.

    Each returned TF comes with three quality values (0-255):

      tfp  (significance): measure of how unlikely this run of
           consecutive misses is under the read's LOCAL hit rate,
           corrected for rotational face-phasing near nucleosomes.

           For each miss at ref position p, the local hit probability
           is baseline * rate_profile(d_nuc(p)), where d_nuc is the
           distance to the nearest nuc edge and rate_profile encodes
           the 10.4 bp helical oscillation calibrated from scDAF.

           P = ∏ (1 - baseline * rate_profile(d_nuc(p_i)))
           tfp = clip(255 * -log10(P) / _TFP_MAX_NEG_LOG10, 0, 255)

           Falls back to uniform baseline when nuc_edges is None
           (e.g. Hia5 path) or when all opps are >30 bp from any nuc.

      tel / ter (left / right edge sharpness): unchanged from iter-16.

    Args:
        msps: list of (ref_s, ref_e) tuples — accessible regions to scan.
        opp, hit: per-bp arrays.
        baseline: per-read hit rate.
        min_misses: skip runs shorter than this many MISSES.
        nuc_edges: sorted numpy int array of ALL nuc boundary positions
            (both starts and ends). Used for the rotational correction.
            Pass None to disable (e.g. for Hia5).

    Returns:
        List of tuples ((ref_s, ref_e), tfp, tel, ter) in read-local
        reference coordinates.
    """
    import math
    if baseline <= 0 or baseline >= 1:
        return []
    log_base = math.log10(max(1e-12, 1.0 - baseline))
    tfs = []
    for ms, me in msps:
        if me - ms < 2:
            continue
        sub_opp = opp[ms:me] > 0
        if sub_opp.sum() < min_misses:
            continue
        local_positions = np.where(sub_opp)[0] + ms
        local_is_hit = hit[local_positions] > 0

        in_run = False
        run_s_idx = 0
        for i, is_h in enumerate(local_is_hit):
            if is_h:
                if in_run:
                    _emit_tf(tfs, run_s_idx, i - 1, local_positions,
                              ms, me, local_is_hit, min_misses,
                              log_base, baseline, nuc_edges,
                              nuc_edge_quals)
                    in_run = False
            else:
                if not in_run:
                    run_s_idx = i
                    in_run = True
        if in_run:
            _emit_tf(tfs, run_s_idx, len(local_is_hit) - 1,
                      local_positions, ms, me, local_is_hit,
                      min_misses, log_base, baseline, nuc_edges,
                      nuc_edge_quals)
    return tfs


def _d_nuc_for_positions(positions, nuc_edges):
    """For each ref position in `positions`, return the distance to the
    nearest nuc edge via searchsorted on the sorted edge array."""
    if nuc_edges is None or len(nuc_edges) == 0:
        return None
    idx = np.searchsorted(nuc_edges, positions)
    idx_left = np.clip(idx - 1, 0, len(nuc_edges) - 1)
    idx_right = np.clip(idx, 0, len(nuc_edges) - 1)
    d_left = np.abs(positions - nuc_edges[idx_left])
    d_right = np.abs(nuc_edges[idx_right] - positions)
    return np.minimum(d_left, d_right)


def _nearest_edge_qual(positions, nuc_edges, nuc_edge_quals):
    """For each position, return the quality (0-1) of the nearest nuc edge.

    Used to dampen the rotational correction on ambiguous edges:
    if the nearest nuc edge has low quality (lq or rq < 128), the
    edge placement is uncertain and the cosine phase could be wrong.
    """
    if nuc_edge_quals is None or len(nuc_edge_quals) == 0:
        return None
    idx = np.searchsorted(nuc_edges, positions)
    idx_left = np.clip(idx - 1, 0, len(nuc_edges) - 1)
    idx_right = np.clip(idx, 0, len(nuc_edges) - 1)
    d_left = np.abs(positions - nuc_edges[idx_left])
    d_right = np.abs(nuc_edges[idx_right] - positions)
    left_closer = d_left <= d_right
    return np.where(left_closer,
                     nuc_edge_quals[idx_left],
                     nuc_edge_quals[idx_right])


def _emit_tf(tfs, run_s_idx, run_e_idx, local_positions,
              msp_s, msp_e, local_is_hit, min_misses,
              log_base, baseline, nuc_edges, nuc_edge_quals):
    """Compute quality scores for one candidate run and append to tfs."""
    import math
    n_misses = run_e_idx - run_s_idx + 1
    if n_misses < min_misses:
        return
    left_bp = int(local_positions[run_s_idx])
    right_bp = int(local_positions[run_e_idx]) + 1

    # Significance with rotational phase correction.
    # For each miss position, compute the local hit rate adjusted for
    # the 10.4 bp face-phasing near nucleosome edges.
    # The correction amplitude is DAMPENED by the edge quality of the
    # nearest nuc: sharp edges (lq/rq ≈ 255) get full correction,
    # ambiguous edges (lq/rq < 128) get reduced correction. This
    # prevents applying an out-of-phase cosine when the nuc boundary
    # is uncertain.
    run_positions = local_positions[run_s_idx:run_e_idx + 1]
    d_nucs = _d_nuc_for_positions(run_positions, nuc_edges)

    if d_nucs is not None and np.any(d_nucs <= _ROT_CUTOFF):
        # Per-opp corrected P-value
        rate_profile = np.ones(n_misses, dtype=np.float64)
        near_mask = d_nucs <= _ROT_CUTOFF
        d_near = d_nucs[near_mask].astype(np.float64)

        # Edge-quality dampening: scale amplitude by nearest edge's
        # quality (0-1). Sharp edge → full 0.35 amplitude; ambiguous
        # edge → amplitude → 0 (reverts to uniform baseline).
        edge_quals = _nearest_edge_qual(
            run_positions[near_mask], nuc_edges, nuc_edge_quals)
        if edge_quals is not None:
            amp = _ROT_AMP * edge_quals
        else:
            amp = _ROT_AMP

        rate_profile[near_mask] = (
            1.0 + amp
            * np.cos(2.0 * np.pi * d_near / _ROT_PERIOD)
            * np.exp(-d_near / _ROT_TAU)
        )
        local_rates = np.clip(baseline * rate_profile, 0.001, 0.999)
        log_p = np.sum(np.log10(1.0 - local_rates))
        neg_log10 = -log_p
    else:
        # Uniform baseline (no nuc nearby or no nuc_edges provided)
        neg_log10 = -n_misses * log_base

    tfp = int(round(255.0 * min(1.0, neg_log10 / _TFP_MAX_NEG_LOG10)))
    tfp = max(0, min(255, tfp))

    # Left edge quality: is there a HIT at the opp just to the left
    # of the run, and how close is it?
    if run_s_idx > 0:
        prev_hit_bp = int(local_positions[run_s_idx - 1])
        d_left = left_bp - prev_hit_bp - 1
        tel = int(round(255.0 * max(0.0, 1.0 - d_left / _TF_EDGE_SATURATE)))
    else:
        # Run starts at the first opp of the MSP → ambiguous left edge.
        # Value depends on how close the run start is to the MSP start:
        # if right at the start (run could extend further) → tel=0.
        d_to_msp_start = left_bp - msp_s
        if d_to_msp_start < 3:
            tel = 0
        else:
            tel = int(round(255.0 * max(0.0, 1.0 - d_to_msp_start / _TF_EDGE_SATURATE)))

    # Right edge quality: mirror of left.
    if run_e_idx < len(local_is_hit) - 1:
        next_hit_bp = int(local_positions[run_e_idx + 1])
        d_right = next_hit_bp - right_bp
        ter = int(round(255.0 * max(0.0, 1.0 - d_right / _TF_EDGE_SATURATE)))
    else:
        d_to_msp_end = msp_e - right_bp
        if d_to_msp_end < 3:
            ter = 0
        else:
            ter = int(round(255.0 * max(0.0, 1.0 - d_to_msp_end / _TF_EDGE_SATURATE)))

    tel = max(0, min(255, tel))
    ter = max(0, min(255, ter))
    tfs.append(((left_bp, right_bp), tfp, tel, ter))


# Backward-compat alias: older code calls this name with a min_score
# keyword. The new implementation ignores score thresholds entirely.
def call_tfs_strict_miss(msps, opp, hit, baseline, **kwargs):
    """Deprecated shim. The new TF caller emits every MISS run and
    relies on quality scores for filtering. Returns a list of
    ((s, e), tfp, tel, ter) tuples — note the expanded return format
    compared to the old (pair, q) tuples."""
    return call_tfs_overcall(msps, opp, hit, baseline,
                               min_misses=kwargs.get('min_misses', 2),
                               nuc_edges=kwargs.get('nuc_edges'))


def find_pass1_atoms_gap_cdf(opp, hit, L, gap_radius=10):
    """Hit-free stretches >= gap_radius become atoms.

    Operates on the same (opp, hit) arrays as the rate-windowed
    primitive, but cares only about hit positions, not opportunity
    density. Equivalent to caller_v2's gap-CDF approach with
    confidence threshold expressed as gap_radius.
    """
    hit_pos = np.where(hit > 0)[0]
    if len(hit_pos) == 0:
        # No hits at all -> whole read is "protected"; let downstream
        # filters decide (this is a saturated edge case).
        return [(0, L)] if L > 0 else []
    atoms = []
    # Before first hit
    first = int(hit_pos[0])
    if first >= gap_radius:
        atoms.append((0, first))
    # Between consecutive hits
    for a, b in zip(hit_pos[:-1], hit_pos[1:]):
        gap = int(b - a) - 1
        if gap >= gap_radius:
            atoms.append((int(a) + 1, int(b)))
    # After last hit
    last = int(hit_pos[-1])
    tail = L - (last + 1)
    if tail >= gap_radius:
        atoms.append((last + 1, L))
    # Structural merge of any touching/overlapping atoms (shouldn't
    # happen given the construction, but be safe)
    if not atoms:
        return []
    out = [atoms[0]]
    for s, e in atoms[1:]:
        ls, le = out[-1]
        if s <= le:
            out[-1] = (ls, max(le, e))
        else:
            out.append((s, e))
    return out


# -------------------------------------------------------------------
# v8 Poisson-evidence merge
# -------------------------------------------------------------------

def poisson_merge_evidence(atoms, opp, hit, baseline,
                            merge_alpha=0.10,
                            max_merge_len=250,
                            short_gap_bp=8,
                            min_gap_opp=5,
                            low_quantile=0.05,
                            high_quantile=0.95,
                            return_merge_quality=False,
                            return_gap_records=False,
                            fp_model=None,
                            query_seq=None):
    """Two-sided evidence-based merge.

    For each adjacent pair of atoms:

      - gap_len <= short_gap_bp: always merge (structural bridging).
      - proposed merged span > max_merge_len: refuse to merge
        (cascade guard; Pass 1 calls of any length still pass through
        unchanged elsewhere).
      - gap_opp < min_gap_opp: refuse to merge (low statistical power;
        default to split rather than v7's default to merge).
      - otherwise: compute lambda = gap_opp * baseline and reject both
        tails of the Poisson(lambda) distribution at
        [low_quantile, high_quantile]. Merge only if gap_hit lies
        inside the interval, meaning the observed hit count is
        compatible with the baseline within normal sampling noise.

    Returns list of (ref_start, ref_end_exclusive).

    `merge_alpha` is accepted for CLI compatibility with v7 but is
    not used by this function. The width of the two-sided interval
    is controlled by `low_quantile` and `high_quantile`.

    If `return_merge_quality=True`, also returns a parallel list of
    per-output-atom merge qualities (0-255). Pure Pass-1 atoms with
    no internal merges get mq=255. Merged atoms get mq equal to the
    MINIMUM across all internal gaps of
        round(255 * (1 - 2 * |CDF(gap_hit | lambda) - 0.5|))
    so 255 = gap hit count was at the Poisson median (high confidence
    the gap is baseline breathing), 0 = gap was at the tail.

    If `return_gap_records=True`, also returns a list of tuples
    (gap_s, gap_e, gap_mq, was_merged) describing every inter-atom
    gap the merge saw. Used downstream to compute "confident MSPs"
    for the legacy `as/al` tag (gaps inside high-mq merges are
    absorbed; low-mq merge gaps stay visible as MSPs).
    """
    if len(atoms) < 2:
        outs = [list(atoms)]
        if return_merge_quality:
            outs.append([255] * len(atoms))
        if return_gap_records:
            outs.append([])
        return tuple(outs) if len(outs) > 1 else outs[0]

    result = [atoms[0]]
    mq_lists = [[]]  # per-output-atom list of per-gap merge qualities
    gap_records = []  # (gap_s, gap_e, mq, was_merged) for every considered gap

    for s, e in atoms[1:]:
        ps, pe = result[-1]
        gap_s, gap_e = pe, s
        merged_len = e - ps

        def _record(mq, was_merged):
            gap_records.append((int(gap_s), int(gap_e), int(mq), bool(was_merged)))

        # Cascade guard: never CREATE a merged atom larger than this.
        # Pass 1 atoms larger than max_merge_len pass through anywhere
        # else in the code, so this is not an output size cap.
        if merged_len > max_merge_len:
            result.append((s, e))
            mq_lists.append([])
            _record(0, False)
            continue

        if gap_e <= gap_s:
            # Touching / overlapping -> merge (structural, perfect quality)
            result[-1] = (ps, max(pe, e))
            mq_lists[-1].append(255)
            _record(255, True)
            continue

        gap_len = gap_e - gap_s

        if gap_len <= short_gap_bp:
            # Too short to be a real linker — always merge, but the
            # merge quality must reflect the hit content of the gap.
            # A 1-bp gap with 1 hit is still a "bridge" we want to
            # cross structurally, but its mq cannot be 255 because
            # the merged atom absorbs a real hit. Otherwise the TF
            # caller downstream sees an mq=255 "pure" atom that in
            # fact contains internal hits (iter-16d bug).
            #
            # Linear decay: each absorbed hit costs 30 mq points.
            # 0 hits → 255, 1 hit → 225, 2 hits → 195, …, 8 hits → 15
            # This is GENTLER than the Poisson CDF approach because
            # the Poisson model at low gap_opp exaggerates the
            # significance of individual hits (a 1-bp 1-hit "gap" is
            # noise-level, not strong merge evidence), and we want
            # mq to honestly say "this merge absorbed a handful of
            # hits" rather than "this merge is terrible".
            gap_hit_count = int(hit[gap_s:gap_e].sum())
            if gap_hit_count == 0:
                mq = 255
            else:
                mq = max(0, 255 - 30 * gap_hit_count)
            result[-1] = (ps, max(pe, e))
            mq_lists[-1].append(mq)
            _record(mq, True)
            continue

        gap_opp = int(opp[gap_s:gap_e].sum())
        gap_hit = int(hit[gap_s:gap_e].sum())

        if gap_opp < min_gap_opp:
            # Not enough power — default to split. This is the
            # intentional inversion of v7's `gap_override_opp`.
            result.append((s, e))
            mq_lists.append([])
            _record(0, False)
            continue

        # Lambda for the Poisson test. If a per-context FP model is
        # available, use the SUM of per-position FP rates in the gap
        # (accounts for CpG-adjacent bias). Otherwise fall back to
        # the uniform baseline * gap_opp.
        if fp_model is not None and query_seq is not None:
            # Per-position FP rates for opp positions in the gap
            gap_opp_positions = np.where(opp[gap_s:gap_e] > 0)[0] + gap_s
            lam = fp_model.expected_fp_hits(gap_opp_positions, query_seq)
        else:
            lam = gap_opp * baseline
        lam = max(0.01, lam)  # floor to avoid degenerate Poisson at λ=0
        lo = float(poisson.ppf(low_quantile, lam))
        hi = float(poisson.ppf(high_quantile, lam))
        if lo <= gap_hit <= hi:
            # Hit count compatible with expected FP rate -> merge.
            cdf_val = float(poisson.cdf(gap_hit, lam))
            center_dist = abs(cdf_val - 0.5)
            mq = int(round(255 * (1 - 2 * center_dist)))
            if mq < 0: mq = 0
            if mq > 255: mq = 255
            result[-1] = (ps, max(pe, e))
            mq_lists[-1].append(mq)
            _record(mq, True)
        else:
            # Gap is either too protected or has too many hits to
            # be a single breathing region. Keep split.
            result.append((s, e))
            mq_lists.append([])
            _record(0, False)

    outs = [result]
    if return_merge_quality:
        # Reduce each output atom's per-gap list to the minimum; an
        # atom with no internal merges gets mq=255 (nothing to weaken it)
        mq_final = [min(q) if q else 255 for q in mq_lists]
        outs.append(mq_final)
    if return_gap_records:
        outs.append(gap_records)
    return tuple(outs) if len(outs) > 1 else outs[0]


# -------------------------------------------------------------------
# Per-read caller
# -------------------------------------------------------------------

def call_read(read, ref_seq, extractor, W,
              min_read_rate, merge_alpha,
              max_merge_len, min_footprint,
              first_pass='gap_cdf',
              gap_radius=10,
              min_tf_bp=5, min_tf_tq=10,
              fp_model=None):
    """Run v8 pipeline on one read; return dict of ns/nl/as/al/nq
    or None if no calls.

    `first_pass` selects which Pass 1 to run:
      - 'gap_cdf' (default, recommended): hit-free stretches >= gap_radius
        bp. Simpler, faster, lower overmerge. Benchmark winner.
      - 'protected_runs': v7's windowed-rate find_pass1_atoms.

    `min_tf_bp` / `min_tf_tq`: TF calls shorter than min_tf_bp or
    with tq < min_tf_tq are dropped before output. Set to 0 to emit
    everything (overcall mode). Default 5 bp / tq 10 cuts pure noise.
    """
    L = read.reference_end - read.reference_start
    if L < W + 100:
        return None

    opp, hit = extractor.read_to_arrays(read, ref_seq)
    n_opp = int(opp.sum())
    if n_opp < 50:
        return None
    baseline = int(hit.sum()) / n_opp
    if baseline < min_read_rate:
        return None

    if first_pass == 'gap_cdf':
        # Need rate/valid only for nq computation later; cheaper-but-
        # still-correct path: compute them on demand.
        rate, valid = windowed_rate(opp, hit, W)
        atoms = find_pass1_atoms_gap_cdf(opp, hit, L, gap_radius=gap_radius)
    else:
        rate, valid = windowed_rate(opp, hit, W)
        if valid.sum() < 100:
            return None
        atoms = find_pass1_atoms(rate, valid, W, baseline)
    if not atoms:
        return None

    # Pass 2: evidence-based merge (v8 change). We also ask for
    # per-atom merge quality (mq) AND per-gap records — the records
    # let us compute "confident" MSPs (gaps NOT absorbed by high-mq
    # merges) for the legacy as/al tag, while still running the TF
    # caller on the fully permissive pre-merge accessible surface.
    # Query sequence for per-context FP model (if available)
    q_seq = read.query_sequence if fp_model is not None else None

    merged, merged_mqs, gap_records = poisson_merge_evidence(
        atoms, opp, hit, baseline,
        merge_alpha=merge_alpha,
        max_merge_len=max_merge_len,
        return_merge_quality=True,
        return_gap_records=True,
        fp_model=fp_model,
        query_seq=q_seq,
    )
    if not merged:
        return None

    # Build the TF-input accessible surface — "relaxed cascade":
    #
    #   (1) complement of the final (user-selected first-pass) nucs
    #       after merge + min_footprint. This is the normal linker
    #       complement — it is critical to use the SAME Pass-1 here
    #       as the reported nucs, otherwise the two sets disagree on
    #       boundaries by dozens of bp and we get bogus overlaps.
    #
    #   (2) every merged-over gap from the user-selected first-pass.
    #       A merged atom is exactly the place where the caller is
    #       uncertain about the nucleosome identity. Running the TF
    #       caller inside a merged atom preserves an alt hypothesis:
    #       "this might not be one big nucleosome — it might be two
    #       closely positioned TFs". Pure Pass-1 atoms (mq=255, no
    #       internal gaps) never trigger this branch, so TFs still
    #       NEVER appear inside pure Pass-1 calls.
    #
    # Caveat: the "merged-over gap is Poisson-consistent with
    # baseline" intuition is only weak — the merge central interval
    # is (5%, 95%) wide, so a gap with ZERO hits and small gap_opp
    # can still merge and yet carry strong TF signal. That is the
    # kind of case we want downstream users to see.
    #
    # Future work: for TFs inside merges, layer in cross-strand
    # rescue and aggregate-read evidence at the same locus to test
    # the alt hypothesis more specifically.
    tf_nucs = [(s, e) for (s, e) in merged if e - s >= min_footprint]
    tf_input_msps_ref: list = []
    cur = 0
    for a_s, a_e in tf_nucs:
        if a_s > cur:
            tf_input_msps_ref.append((cur, a_s))
        cur = max(cur, a_e)
    if L > cur:
        tf_input_msps_ref.append((cur, L))
    # Add merged-over gaps (source 2). A "was_merged" record means
    # this gap sits inside a current output atom — exactly the case
    # we want the TF caller to look inside.
    for g_s, g_e, g_mq, was_merged in gap_records:
        if was_merged and g_e - g_s >= 3:
            tf_input_msps_ref.append((int(g_s), int(g_e)))
    # Dedupe + structural merge of any touching/overlapping MSPs.
    tf_input_msps_ref.sort()
    tf_merged_input: list = []
    for ms, me in tf_input_msps_ref:
        if tf_merged_input and ms <= tf_merged_input[-1][1]:
            tf_merged_input[-1] = (tf_merged_input[-1][0],
                                     max(tf_merged_input[-1][1], me))
        else:
            tf_merged_input.append((ms, me))

    # Build sorted nuc-edge array for the rotational phase correction.
    # Includes both start and end of every final nuc in ref coords,
    # plus the edge quality (lq at start, rq at end) for dampening
    # the correction amplitude on ambiguous edges.
    if tf_nucs:
        edge_pos_list = []
        for (s, e) in tf_nucs:
            lq_val, rq_val = compute_edge_q(s, e, hit,
                                              breathing_window=50)
            edge_pos_list.append((s, lq_val))
            edge_pos_list.append((e, rq_val))
        edge_pos_list.sort(key=lambda x: x[0])
        nuc_edges = np.array([p for p, _ in edge_pos_list],
                               dtype=np.int64)
        nuc_edge_quals = np.array([q for _, q in edge_pos_list],
                                    dtype=np.float64) / 255.0
    else:
        nuc_edges = None
        nuc_edge_quals = None

    # TF calls — overcall everything, rely on tfp / tel / ter for
    # downstream filtering. Each entry is ((ref_s, ref_e), tfp, tel, ter).
    tf_calls_ref = call_tfs_overcall(tf_merged_input, opp, hit, baseline,
                                       nuc_edges=nuc_edges,
                                       nuc_edge_quals=nuc_edge_quals)

    # Clip and filter — carry mq and compute edge-quality (lq/rq)
    CONFIDENT_MQ = 128  # gaps inside merges with mq >= this are absorbed
    BREATHING_WINDOW = 50  # bp over which we look for the first inner HIT

    merged_clipped = []
    clipped_mqs = []
    clipped_edge_q = []
    for (s, e), mq in zip(merged, merged_mqs):
        s2 = max(0, s)
        e2 = min(L, e)
        if e2 - s2 >= min_footprint:
            merged_clipped.append((s2, e2))
            clipped_mqs.append(mq)
            lq, rq = compute_edge_q(s2, e2, hit,
                                     breathing_window=BREATHING_WINDOW)
            clipped_edge_q.append((lq, rq))
    if not merged_clipped:
        return None

    ref_to_q = build_ref_to_query_map(read, L)

    nuc_intervals = []
    nuc_nq = []
    nuc_mq = []
    nuc_lq = []
    nuc_rq = []
    for (s, e), mq_val, (lq_val, rq_val) in zip(
            merged_clipped, clipped_mqs, clipped_edge_q):
        rate_s = max(0, s)
        rate_e = max(rate_s, min(len(rate), e - W + 1))
        nq_val = 0
        if rate_e > rate_s:
            sub = rate[rate_s:rate_e]
            sub_valid = valid[rate_s:rate_e]
            if sub_valid.any():
                min_rate = float(np.min(sub[sub_valid]))
                if baseline > 0:
                    protection = 1.0 - min_rate / baseline
                    protection = max(0.0, min(1.0, protection))
                    nq_val = int(round(protection * 255))
        qi = ref_interval_to_query(ref_to_q, s, e)
        if qi is None:
            continue
        nuc_intervals.append(qi)
        nuc_nq.append(nq_val)
        nuc_mq.append(mq_val)
        nuc_lq.append(lq_val)
        nuc_rq.append(rq_val)

    if not nuc_intervals:
        return None

    # Query-coord absorb (collapse overlapping atoms post-conversion).
    # When two ref atoms map to overlapping query intervals, take the
    # higher of each quality value.
    pairs = sorted(
        zip(nuc_intervals, nuc_nq, nuc_mq, nuc_lq, nuc_rq),
        key=lambda x: x[0][0])
    merged_intervals = [pairs[0][0]]
    merged_nq = [pairs[0][1]]
    merged_mq_q = [pairs[0][2]]
    merged_lq = [pairs[0][3]]
    merged_rq = [pairs[0][4]]
    for (qs, qe), nq_val, mq_val, lq_val, rq_val in pairs[1:]:
        last_qs, last_qe = merged_intervals[-1]
        if qs <= last_qe:
            merged_intervals[-1] = (last_qs, max(last_qe, qe))
            if nq_val > merged_nq[-1]:
                merged_nq[-1] = nq_val
            if mq_val > merged_mq_q[-1]:
                merged_mq_q[-1] = mq_val
            if lq_val > merged_lq[-1]:
                merged_lq[-1] = lq_val
            if rq_val > merged_rq[-1]:
                merged_rq[-1] = rq_val
        else:
            merged_intervals.append((qs, qe))
            merged_nq.append(nq_val)
            merged_mq_q.append(mq_val)
            merged_lq.append(lq_val)
            merged_rq.append(rq_val)

    ns = [qs for qs, _ in merged_intervals]
    nl = [qe - qs for qs, qe in merged_intervals]
    nq = merged_nq
    mq = merged_mq_q
    lq = merged_lq
    rq = merged_rq

    qlen = read.query_length or 0

    # -------------------------------------------------------------
    # Confident MSPs for the legacy as/al tag (and MA msp+)
    # -------------------------------------------------------------
    # Start with the complement of the final nucleosome list, then
    # ADD BACK any low-confidence merge gaps (those stay visible as
    # ambiguous accessible regions).
    #
    # Semantically: "the places you should scan for TFs if you trust
    # v8's nucleosome calls as an upper bound on protection"
    exposed_low_conf_gaps = []
    for g_s, g_e, g_mq, was_merged in gap_records:
        if was_merged and g_mq < CONFIDENT_MQ and g_e > g_s:
            qi = ref_interval_to_query(ref_to_q, g_s, g_e)
            if qi is not None:
                exposed_low_conf_gaps.append(qi)

    # Standard complement in query coords
    complement_msps = []
    cursor = 0
    for qs, qe in merged_intervals:
        if qs > cursor:
            complement_msps.append((cursor, qs))
        cursor = max(cursor, qe)
    if cursor < qlen:
        complement_msps.append((cursor, qlen))

    # Union + merge
    all_msps = sorted(complement_msps + exposed_low_conf_gaps)
    merged_msps = []
    for ms, me in all_msps:
        if merged_msps and ms <= merged_msps[-1][1]:
            merged_msps[-1] = (merged_msps[-1][0], max(merged_msps[-1][1], me))
        else:
            merged_msps.append((ms, me))
    as_list = [ms for ms, _ in merged_msps]
    al_list = [me - ms for ms, me in merged_msps]

    # -------------------------------------------------------------
    # TF calls -> query coords. Each call carries three qualities:
    #   tq  = tfp, significance (-log10(P) scaled to 0-255)
    #   el  = left edge sharpness
    #   er  = right edge sharpness
    # -------------------------------------------------------------
    tf_ns = []
    tf_nl = []
    tf_tq = []   # tfp
    tf_tel = []
    tf_ter = []
    for (ref_s, ref_e), tfp, tel_, ter_ in tf_calls_ref:
        # Floor filter: drop noise before BAM emission
        if ref_e - ref_s < min_tf_bp:
            continue
        if tfp < min_tf_tq:
            continue
        qi = ref_interval_to_query(ref_to_q, ref_s, ref_e)
        if qi is None:
            continue
        qs, qe = qi
        if qe - qs < min_tf_bp:
            continue
        tf_ns.append(qs)
        tf_nl.append(qe - qs)
        tf_tq.append(tfp)
        tf_tel.append(tel_)
        tf_ter.append(ter_)

    return {
        'ns': ns, 'nl': nl, 'nq': nq, 'mq': mq, 'lq': lq, 'rq': rq,
        'as': as_list, 'al': al_list,
        'tn': tf_ns, 'tl': tf_nl, 'tq': tf_tq,
        'el': tf_tel, 'er': tf_ter,
        'qlen': qlen,
    }


# -------------------------------------------------------------------
# Driver (mirrors v7)
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--in-bam', required=True)
    ap.add_argument('--out-bam', required=True)
    ap.add_argument('--fa', required=True)
    ap.add_argument('--enzyme', required=True, choices=['daf', 'hia5'])
    ap.add_argument('--W', type=int, default=40,
                    help='scan window width (default 40)')
    ap.add_argument('--min-read-rate', type=float, default=0.05,
                    help='reject reads with overall rate below this')
    ap.add_argument('--merge-alpha', type=float, default=0.10,
                    help='v7 compatibility flag; ignored by v8 merge '
                         '(width is set by quantile args below)')
    ap.add_argument('--max-merge-len', type=int, default=250,
                    help='cascade guard: refuse to CREATE merged atoms '
                         'larger than this (does NOT cap Pass 1 output; '
                         'default 250 bp = mono-nuc + breathing + '
                         'chromatosome)')
    ap.add_argument('--first-pass', default='gap_cdf',
                    choices=['protected_runs', 'gap_cdf'],
                    help='Pass 1 algorithm: gap_cdf (default, recommended '
                         '— hit-free stretches; lower overmerge, benchmark '
                         'winner) or protected_runs (v7-style windowed rate)')
    ap.add_argument('--gap-radius', type=int, default=10,
                    help='gap_cdf only: minimum hit-free stretch in bp '
                         'to count as a Pass 1 atom (default 10, '
                         'iteration-8 sweep optimum on 10 datasets)')
    ap.add_argument('--min-footprint', type=int, default=80,
                    help='reject merged runs shorter than this')
    ap.add_argument('--gap-override-length', type=int, default=0,
                    help='v7 compatibility flag; ignored by v8')
    ap.add_argument('--gap-override-opp', type=int, default=0,
                    help='v7 compatibility flag; ignored by v8')
    ap.add_argument('--max-reads', type=int, default=0,
                    help='0 = no limit')
    ap.add_argument('--min-tf-bp', type=int, default=5,
                    help='minimum TF length in bp to emit to BAM '
                         '(default 5 — smaller runs are pure noise '
                         'at typical DAF density). Set to 0 to emit '
                         'everything.')
    ap.add_argument('--min-tf-tq', type=int, default=10,
                    help='minimum tq to emit a TF to BAM (default 10 '
                         '— cuts the weakest ~5%% of candidates that '
                         'are indistinguishable from baseline noise). '
                         'Set to 0 to emit everything.')
    ap.add_argument('--fp-model', default=None,
                    help='path to a per-context FP model JSON (from '
                         'ContextFPModel.from_bam on an untreated '
                         'control). When provided, the merge step '
                         'uses per-position FP rates instead of a '
                         'flat baseline for the Poisson test. '
                         'Recommended for Hia5 / m6A enzymes.')
    ap.add_argument('--strip-mods', action='store_true',
                    help='drop MM/ML tags to shrink output')
    ap.add_argument('--ml-threshold', type=int, default=128,
                    help='Hia5 only: m6A ML threshold')
    ap.add_argument('--tags', default='both',
                    choices=['legacy', 'ma', 'both'],
                    help='which BAM tag schema to write: '
                         '`legacy` = ns/nl/nq/as/al (v7-compatible, '
                         'current default for drop-in migration); '
                         '`ma` = MA/AQ per fiberseq Molecular-'
                         'annotation-spec (nuc+QQ with nq and mq); '
                         '`both` = write both simultaneously (the '
                         'two sets are independent BAM tags so a '
                         'single file can carry both and legacy '
                         'tools see ns/nl while MA-aware tools see '
                         'MA/AQ). Default: both.')
    ap.add_argument('--progress-every', type=int, default=1000)
    args = ap.parse_args()

    extractor_kwargs = {}
    if args.enzyme == 'hia5':
        extractor_kwargs['ml_threshold'] = args.ml_threshold
    extractor = get_extractor(args.enzyme, **extractor_kwargs)

    # Load per-context FP model if provided
    fp_model = None
    if args.fp_model:
        from context_fp_model import ContextFPModel
        fp_model = ContextFPModel.load(args.fp_model)
        print(f'Loaded FP model: {fp_model.summary()}', flush=True)

    out_dir = os.path.dirname(args.out_bam) or '.'
    os.makedirs(out_dir, exist_ok=True)

    bam_in = pysam.AlignmentFile(args.in_bam, 'rb', check_sq=False)
    fa = pysam.FastaFile(args.fa) if args.fa and os.path.exists(args.fa) else None
    bam_out = pysam.AlignmentFile(args.out_bam, 'wb', template=bam_in)
    if fa is None:
        print('  (no FASTA provided — will reconstruct ref from MD tags)',
              flush=True)

    n_reads = n_called = n_skipped = n_passthrough = 0
    nuc_counts = []

    print(f'caller_v8  enzyme={args.enzyme}  W={args.W}  '
          f'max_merge_len={args.max_merge_len}  '
          f'min_footprint={args.min_footprint}  '
          f'min_read_rate={args.min_read_rate}', flush=True)

    for read in bam_in.fetch(until_eof=True):
        n_reads += 1
        if args.max_reads and n_reads > args.max_reads:
            break
        if read.is_secondary or read.is_supplementary or read.is_unmapped:
            bam_out.write(read)
            n_passthrough += 1
            continue
        ref_seq = None
        if fa is not None:
            try:
                ref_seq = fa.fetch(read.reference_name,
                                    read.reference_start,
                                    read.reference_end).upper()
            except Exception:
                ref_seq = None
            if ref_seq is not None and len(ref_seq) != read.reference_end - read.reference_start:
                ref_seq = None
        if ref_seq is None:
            # Fallback: reconstruct reference from MD tags. Works for
            # custom amplicon BAMs whose contigs don't appear in the
            # provided FASTA, and for inputs with --fa /dev/null.
            try:
                ref_seq = read.get_reference_sequence().upper()
            except Exception:
                ref_seq = None
            if ref_seq is not None and len(ref_seq) != read.reference_end - read.reference_start:
                ref_seq = None
            if ref_seq is None:
                if args.enzyme == 'hia5':
                    # Hia5Extractor uses MM/ML tags, not ref_seq —
                    # pass a dummy so the read isn't skipped.
                    ref_seq = 'N' * (read.reference_end - read.reference_start)
                else:
                    clear_stale_tags(read)
                    if args.strip_mods:
                        strip_mod_tags(read)
                    bam_out.write(read)
                    n_skipped += 1
                    continue

        result = call_read(read, ref_seq, extractor,
                            args.W, args.min_read_rate, args.merge_alpha,
                            args.max_merge_len, args.min_footprint,
                            first_pass=args.first_pass,
                            gap_radius=args.gap_radius,
                            min_tf_bp=args.min_tf_bp,
                            min_tf_tq=args.min_tf_tq,
                            fp_model=fp_model)

        # Clear BOTH legacy and MA tags whether or not we call —
        # ensures a re-run doesn't leave stale output from either
        # schema behind.
        clear_all_call_tags(read)
        if args.strip_mods:
            strip_mod_tags(read)

        if result is None:
            bam_out.write(read)
            n_skipped += 1
        else:
            want_legacy = args.tags in ('legacy', 'both')
            want_ma = args.tags in ('ma', 'both')
            if want_legacy:
                # Nucleosome legacy tags
                set_array_tag(read, 'ns', result['ns'])
                set_array_tag(read, 'nl', result['nl'])
                set_array_tag(read, 'nq', result['nq'])
                set_array_tag(read, 'mq', result['mq'])
                # New legacy nucleosome quality arrays
                set_array_tag(read, 'lq', result['lq'])
                set_array_tag(read, 'rq', result['rq'])
                # Confident MSPs
                set_array_tag(read, 'as', result['as'])
                set_array_tag(read, 'al', result['al'])
                # TF calls (may be empty)
                if result['tn']:
                    set_array_tag(read, 'tn', result['tn'])
                    set_array_tag(read, 'tl', result['tl'])
                    set_array_tag(read, 'tq', result['tq'])
                    set_array_tag(read, 'el', result['el'])
                    set_array_tag(read, 'er', result['er'])
            if want_ma:
                # Build MA:Z string + AQ:B:C array
                # nuc+QQQQ: (nq, mq, lq, rq) per nucleosome
                # msp+:    no quality
                # tf+QQQ:  (tfp, tel, ter) per TF call
                nuc_intervals = list(zip(result['ns'], result['nl']))
                msp_intervals = list(zip(result['as'], result['al']))
                tf_intervals = list(zip(result['tn'], result['tl']))
                ma_str = format_ma_tag(
                    result['qlen'],
                    nuc_intervals, msp_intervals,
                    tf_intervals=tf_intervals,
                    nuc_qual_spec='QQQQ',
                    tf_qual_spec='QQQ',
                )
                aq_arr = format_aq_array(
                    result['nq'], result['mq'],
                    lq_values=result['lq'], rq_values=result['rq'],
                    tf_q_values=result['tq'],
                    tf_lq_values=result['el'],
                    tf_rq_values=result['er'],
                )
                read.set_tag('MA', ma_str, value_type='Z')
                if len(aq_arr) > 0:
                    read.set_tag('AQ', aq_arr)
            bam_out.write(read)
            n_called += 1
            nuc_counts.append(len(result['ns']))

        if n_reads % args.progress_every == 0:
            print(f'  {n_reads} reads processed  '
                  f'called={n_called} skipped={n_skipped}',
                  flush=True)

    bam_in.close()
    bam_out.close()
    if fa is not None:
        fa.close()

    if nuc_counts:
        nc = np.asarray(nuc_counts)
        print(f'\nTotal reads: {n_reads}')
        print(f'Called:      {n_called}')
        print(f'Skipped:     {n_skipped}')
        print(f'Passthrough: {n_passthrough}')
        print(f'Nucs per read: mean={nc.mean():.1f} median={int(np.median(nc))}')


if __name__ == '__main__':
    main()
