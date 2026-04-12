"""Reference-frame reimplementations of each first-pass architecture.

Every first-pass has signature:
    first_pass(bundle: ReadBundle, **params) -> list[(int, int)]

Returning ref-frame atoms [(s, e_exclusive), ...]. Atoms are later
filtered by min_footprint and optionally piped through v7's Poisson merge.

The second pass, poisson_merge_atoms, is imported from caller_v7 and
applied uniformly across all first-passes in the '+merge' variants.

Intentional design choices:
  - No fixed size caps anywhere. Footprints can be any length.
  - All algorithms work on the same (opp, hit, baseline) triple so the
    comparison is fair. The only difference is how they turn that signal
    into atoms.
  - The profile used by `profile_guided` comes from
    `phase0/results/ddda_profile.json` (derived from NAPA_PS00626, 42k
    nucleosomes). No max_nuc hard cap — we disable the undergrad's 220bp
    ceiling.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import List, Tuple

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import poisson

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # phase0
from caller_v7 import find_pass1_atoms, poisson_merge_atoms  # noqa: E402
from bench.shared import ReadBundle, windowed_rate, structural_merge  # noqa: E402


Atom = Tuple[int, int]


# ===================================================================
# 1. gap_cdf  —  caller_v2 style
# ===================================================================
#
# Simplest baseline: anywhere with a gap between hit events >= radius is
# considered protected. The "CDF radius" is the 1D equivalent of v2's
# statistical gap test — but in our unified harness we operationalize it
# as: protected regions are contiguous runs of bases at or between hit
# positions where the local gap (from previous hit to next hit) exceeds
# gap_radius. Functionally this is v2's algorithm rewritten for
# reference-frame arrays.
# ===================================================================

def first_pass_gap_cdf(bundle: ReadBundle,
                        gap_radius: int = 30,
                        edge_trim: int = 0) -> List[Atom]:
    hit_pos = np.where(bundle.hit > 0)[0]
    L = bundle.L
    if len(hit_pos) == 0:
        # No hits — whole read is "protected" (probably saturated)
        return [(edge_trim, L - edge_trim)] if L > 2 * edge_trim else []

    atoms = []
    # Before first hit
    first = int(hit_pos[0])
    if first - edge_trim >= gap_radius:
        atoms.append((edge_trim, first))
    # Between consecutive hits
    for a, b in zip(hit_pos[:-1], hit_pos[1:]):
        gap = int(b - a) - 1  # bases strictly between
        if gap >= gap_radius:
            atoms.append((int(a) + 1, int(b)))
    # After last hit
    last = int(hit_pos[-1])
    tail = (L - edge_trim) - (last + 1)
    if tail >= gap_radius:
        atoms.append((last + 1, L - edge_trim))
    return structural_merge(atoms)


# ===================================================================
# 2. xdrop_llr  —  caller_v3 style
# ===================================================================
#
# Build per-opportunity LLR score (MISS = +1, HIT = -penalty). Walk the
# opportunity sequence left-to-right and collect maximum subarrays using
# an x-drop reset: when the running score drops more than x_drop below
# the best so far, close the current segment. Classic X-drop from BLAST.
# ===================================================================

def first_pass_xdrop(bundle: ReadBundle,
                      miss_score: float = 1.0,
                      hit_penalty: float = 5.0,
                      x_drop: float = 10.0,
                      min_score: float = 15.0) -> List[Atom]:
    opp = bundle.opp
    hit = bundle.hit
    # Indices of opportunity sites and their miss/hit signs
    opp_idx = np.where(opp > 0)[0]
    if len(opp_idx) < 3:
        return []
    is_hit = hit[opp_idx] > 0
    tok_score = np.where(is_hit, -hit_penalty, miss_score)

    atoms: List[Atom] = []
    best = 0.0
    best_end = -1
    cur = 0.0
    seg_start_idx = 0

    for i, s in enumerate(tok_score):
        if cur <= 0.0:
            cur = s
            seg_start_idx = i
            best = s
            best_end = i
        else:
            cur += s
            if cur > best:
                best = cur
                best_end = i
            # X-drop check
            if best - cur > x_drop:
                if best >= min_score:
                    s_bp = int(opp_idx[seg_start_idx])
                    e_bp = int(opp_idx[best_end]) + 1
                    atoms.append((s_bp, e_bp))
                cur = 0.0
                best = 0.0
                best_end = -1

    if cur > 0 and best >= min_score:
        s_bp = int(opp_idx[seg_start_idx])
        e_bp = int(opp_idx[best_end]) + 1
        atoms.append((s_bp, e_bp))

    return structural_merge(atoms)


# ===================================================================
# 3. hmm_2state  —  caller_v4 / caller_v5 style
# ===================================================================
#
# Run a 2-state Viterbi on the opportunity sequence with states
# PROTECTED (P) and ACCESSIBLE (A). Emissions:
#     P(HIT | PROTECTED)  = 0.02      (background deamination inside cores)
#     P(HIT | ACCESSIBLE) = baseline  (this fiber's overall rate)
# Transitions: expected dwell times of ~73 tokens for protected (≈147 bp
# nucleosome at typical C density) and ~25 tokens for accessible.
# Emit ref-frame atoms from each contiguous PROTECTED run.
# ===================================================================

def first_pass_hmm2(bundle: ReadBundle,
                     emit_protected: float = 0.02,
                     expect_prot_tokens: int = 73,
                     expect_acc_tokens: int = 25) -> List[Atom]:
    opp = bundle.opp
    hit = bundle.hit
    opp_idx = np.where(opp > 0)[0]
    if len(opp_idx) < 10:
        return []
    is_hit = hit[opp_idx] > 0

    baseline = max(bundle.baseline, 2 * emit_protected + 1e-3)
    p_miss_prot = 1.0 - emit_protected
    p_miss_acc = 1.0 - baseline
    p_hit_prot = emit_protected
    p_hit_acc = baseline

    def _log(x):
        return math.log(max(x, 1e-30))

    ep_miss, ep_hit = _log(p_miss_prot), _log(p_hit_prot)
    ea_miss, ea_hit = _log(p_miss_acc), _log(p_hit_acc)
    t_pp = _log(1 - 1.0 / expect_prot_tokens)
    t_pa = _log(1.0 / expect_prot_tokens)
    t_ap = _log(1.0 / expect_acc_tokens)
    t_aa = _log(1 - 1.0 / expect_acc_tokens)
    start_p = _log(0.5)

    n = len(opp_idx)
    dp = np.empty((n, 2), dtype=np.float64)
    bp = np.empty((n, 2), dtype=np.int8)
    em0 = ep_hit if is_hit[0] else ep_miss
    em1 = ea_hit if is_hit[0] else ea_miss
    dp[0, 0] = start_p + em0
    dp[0, 1] = start_p + em1
    bp[0, :] = -1
    for i in range(1, n):
        em_p = ep_hit if is_hit[i] else ep_miss
        em_a = ea_hit if is_hit[i] else ea_miss
        # To PROTECTED
        from_pp = dp[i - 1, 0] + t_pp
        from_ap = dp[i - 1, 1] + t_ap
        if from_pp >= from_ap:
            dp[i, 0] = from_pp + em_p
            bp[i, 0] = 0
        else:
            dp[i, 0] = from_ap + em_p
            bp[i, 0] = 1
        # To ACCESSIBLE
        from_pa = dp[i - 1, 0] + t_pa
        from_aa = dp[i - 1, 1] + t_aa
        if from_pa >= from_aa:
            dp[i, 1] = from_pa + em_a
            bp[i, 1] = 0
        else:
            dp[i, 1] = from_aa + em_a
            bp[i, 1] = 1

    # Backtrace
    states = np.empty(n, dtype=np.int8)
    states[-1] = 0 if dp[-1, 0] >= dp[-1, 1] else 1
    for i in range(n - 1, 0, -1):
        states[i - 1] = bp[i, states[i]]

    # Protected runs in token space -> ref atoms
    atoms: List[Atom] = []
    i = 0
    while i < n:
        if states[i] == 0:
            j = i
            while j < n and states[j] == 0:
                j += 1
            s_bp = int(opp_idx[i])
            e_bp = int(opp_idx[j - 1]) + 1
            atoms.append((s_bp, e_bp))
            i = j
        else:
            i += 1
    return structural_merge(atoms)


# ===================================================================
# 4. core_peaks  —  caller_v6 style
# ===================================================================
#
# Rolling W-bp rate, find_peaks on -rate with a threshold, grow ±flank.
# Direct port of caller_v6's algorithm into the unified harness.
# ===================================================================

def first_pass_core_peaks(bundle: ReadBundle,
                           W: int = 40,
                           core_ratio: float = 0.25,
                           min_sep: int = 160,
                           flank: int = 80) -> List[Atom]:
    rate, valid = windowed_rate(bundle.opp, bundle.hit, W)
    if not valid.any():
        return []
    baseline = bundle.baseline
    threshold = core_ratio * baseline
    # Peaks in -rate = valleys in rate (protected cores)
    signal = -rate.copy()
    # Mask invalid positions down
    signal[~valid] = -1e9
    peaks, _ = find_peaks(signal, height=-threshold, distance=min_sep)
    L = bundle.L
    atoms = []
    for p in peaks:
        # p is a rate-frame index (center of a W-bp window starting at p).
        # Convert to ref center at p + W//2, then grow ±flank.
        center = int(p) + W // 2
        s = max(0, center - flank)
        e = min(L, center + flank)
        if e > s:
            atoms.append((s, e))
    return structural_merge(atoms)


# ===================================================================
# 5. protected_runs  —  caller_v7 Pass 1
# ===================================================================

def first_pass_protected_runs(bundle: ReadBundle,
                                W: int = 40) -> List[Atom]:
    rate, valid = windowed_rate(bundle.opp, bundle.hit, W)
    return find_pass1_atoms(rate, valid, W, bundle.baseline)


# ===================================================================
# 6. profile_guided  —  position_aware_hmm (undergrad), capless
# ===================================================================
#
# Three-pass profile-guided. Pass 1: find hit-free stretches in ref
# coords >= scan_gap. Pass 2: merge adjacent candidates using Poisson
# LLR comparing the profile's position-aware rate to a uniform
# accessible rate. Pass 3: iterate until convergence. NO max_nuc cap.
# ===================================================================

_PROFILE_PATH = os.path.join(os.path.dirname(HERE),
                              'results', 'ddda_profile.json')
_PROFILE_CACHE = None


def _load_profile():
    global _PROFILE_CACHE
    if _PROFILE_CACHE is not None:
        return _PROFILE_CACHE
    with open(_PROFILE_PATH) as fh:
        spec = json.load(fh)
    knots = {int(k): float(v) for k, v in spec['knots'].items()}
    max_d = max(knots.keys())
    xs = sorted(knots.keys())
    ys = [knots[x] for x in xs]
    prof = np.interp(np.arange(max_d + 100),
                     xs, ys,
                     left=ys[0], right=float(spec['accessible_rate']))
    prof = prof.astype(np.float32)
    acc = float(spec['accessible_rate'])
    _PROFILE_CACHE = (prof, acc)
    return _PROFILE_CACHE


def _logpmf(k: int, lam: float) -> float:
    if lam <= 0.0:
        return 0.0 if k == 0 else -1e9
    return k * math.log(lam) - lam - math.lgamma(k + 1)


def first_pass_profile_guided(bundle: ReadBundle,
                                scan_gap: int = 20,
                                llr_threshold: float = 0.0) -> List[Atom]:
    prof, acc_rate = _load_profile()
    p_max = len(prof) - 1

    # Use the per-read baseline instead of the global profile's acc_rate
    # so under- and over-labeled fibers get a fair comparison.
    acc_rate_read = max(bundle.baseline, 0.01)

    hit_sorted = np.where(bundle.hit > 0)[0]
    L = bundle.L
    # Pass 1: candidate cores = hit-free stretches >= scan_gap
    cores = []
    prev = 0
    for p in hit_sorted.tolist():
        gap = p - prev
        if gap >= scan_gap:
            cores.append((prev, p))
        prev = p + 1
    if L - prev >= scan_gap:
        cores.append((prev, L))
    if not cores:
        return []

    # Build a per-base cumulative of the profile integral so any gap's
    # expected hits under the nucleosome model can be computed in O(1).
    # Because the profile is CENTERED on the proposed merged center and
    # the center depends on the current neighbor pair, we do single-sweep
    # merging and use the midpoint of (prev_s, curr_e) as the center.
    # This is the same trade-off caller_v7's poisson_merge_atoms makes.

    merged = [list(cores[0])]
    cum_hit = np.concatenate([[0], np.cumsum(bundle.hit, dtype=np.int32)])
    for s, e in cores[1:]:
        ps, pe = merged[-1]
        gap_s, gap_e = pe, s
        gap_len = gap_e - gap_s
        if gap_len <= 0:
            merged[-1][1] = max(pe, e)
            continue
        # Counts via cumulative hit array
        n_obs = int(cum_hit[gap_e] - cum_hit[gap_s])
        # Profile sum around the proposed merged center
        center = (ps + e) / 2.0
        rng = np.arange(gap_s, gap_e)
        dists = np.abs(rng - center).astype(np.int32)
        np.clip(dists, 0, p_max, out=dists)
        lam_nuc = float(prof[dists].sum())
        lam_acc = gap_len * acc_rate_read
        llr = _logpmf(n_obs, lam_nuc) - _logpmf(n_obs, lam_acc)
        if llr >= llr_threshold:
            merged[-1][1] = max(pe, e)
        else:
            merged.append([s, e])

    return [(int(s), int(e)) for s, e in merged]


# ===================================================================
# Registry
# ===================================================================

ALGORITHMS = {
    'gap_cdf':         first_pass_gap_cdf,
    'xdrop':           first_pass_xdrop,
    'hmm2':            first_pass_hmm2,
    'core_peaks':      first_pass_core_peaks,
    'protected_runs':  first_pass_protected_runs,
    'profile_guided':  first_pass_profile_guided,
}
