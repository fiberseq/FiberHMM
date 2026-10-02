"""Tests for the per-read nucleosome recaller (split + edge refine + nuc+QQQ)."""
from __future__ import annotations

import numpy as np
import pytest

from fiberhmm.inference.circular import project_center_nuc_calls
from fiberhmm.inference.nuc_recaller import (
    NucCall,
    NucProfile,
    _complete_radial_nuc_from_adjacent_tf,
    _find_density_edge,
    _radial_extension_evidence,
    _smoothed_deam_rate,
    assemble_circular_nuc_msp_tiling,
    assemble_nuc_msp_tiling,
    drop_short_nucs_overlapping_promoted,
    exclude_nucleosomes_from_msps,
    promote_large_tf_calls,
    recall_nucs_in_read,
    rederive_msps,
    unify_circular_nuc_calls_with_tf_calls,
    unify_nuc_calls_with_tf_calls,
    validate_radial_access_in_read,
)
from fiberhmm.inference.tf_recaller import TFCall, N_CTX, UNMETH_OFFSET
from fiberhmm.io.ma_tags import ambiguity_to_edge, format_aq_array, parse_aq_array

HIT = 0                 # ctx 0, modified (accessible evidence)
MISS = UNMETH_OFFSET    # ctx 0, unmodified (protected evidence)
NONTARGET = N_CTX       # code 4096: not an opportunity (e.g. non-C/G base)


def _llr_tables():
    """Protected-favoring tables: a miss favors protected, a hit favors accessible."""
    llr_hit = np.full(N_CTX, -3.0, dtype=np.float64)
    llr_miss = np.full(N_CTX, 0.3, dtype=np.float64)
    return llr_hit, llr_miss


def _obs(*runs):
    """Build an obs array from (code, count) runs."""
    parts = [np.full(n, code, dtype=np.int32) for code, n in runs]
    return np.concatenate(parts)


def test_split_separates_two_nucs_at_a_hit_cluster():
    # 60 misses | 6 hits | 60 misses -> the hit cluster is a cut between 2 nucs
    obs = _obs((MISS, 60), (HIT, 6), (MISS, 60))
    llr_hit, llr_miss = _llr_tables()
    nucs, access = recall_nucs_in_read(
        obs, ns=[0], nl=[len(obs)], read_length=len(obs),
        llr_hit=llr_hit, llr_miss=llr_miss,
        split_min_llr=4.0, split_min_opps=3, nuc_min_size=40,
    )
    assert len(nucs) == 2, [(n.start, n.length) for n in nucs]
    # a cut (accessible run) was recorded over the hit cluster
    assert any(60 <= s <= 66 for s, _ in access)
    # both nucleosomes carry quality + edge bytes
    for nc in nucs:
        assert nc.nq > 0
        assert 0 <= nc.el <= 255 and 0 <= nc.er <= 255


def test_no_split_when_no_accessible_evidence():
    obs = _obs((MISS, 150))
    llr_hit, llr_miss = _llr_tables()
    nucs, access = recall_nucs_in_read(
        obs, ns=[0], nl=[len(obs)], read_length=len(obs),
        llr_hit=llr_hit, llr_miss=llr_miss,
        split_min_llr=4.0, split_min_opps=3, nuc_min_size=40,
    )
    assert len(nucs) == 1
    assert nucs[0].length == 150
    assert access == []


def test_subnucleosome_fragment_demoted_to_accessible():
    # 20 misses | 6 hits | 100 misses -> left flank (20bp) is below nuc_min_size
    obs = _obs((MISS, 20), (HIT, 6), (MISS, 100))
    llr_hit, llr_miss = _llr_tables()
    nucs, access = recall_nucs_in_read(
        obs, ns=[0], nl=[len(obs)], read_length=len(obs),
        llr_hit=llr_hit, llr_miss=llr_miss,
        split_min_llr=4.0, split_min_opps=3, nuc_min_size=40,
    )
    assert len(nucs) == 1
    assert nucs[0].start >= 26  # only the long right flank survives as a nuc
    # the 20bp flank shows up as accessible residue
    assert any(s == 0 and length == 20 for s, length in access)


def test_refined_core_below_floor_is_demoted_not_emitted():
    # 100bp fragment, but only a 20bp protected island (the rest is non-target,
    # so nothing splits and no nucleosome-sized protected core exists).
    obs = _obs((NONTARGET, 40), (MISS, 20), (NONTARGET, 40))
    llr_hit, llr_miss = _llr_tables()
    nucs, access = recall_nucs_in_read(
        obs, ns=[0], nl=[len(obs)], read_length=len(obs),
        llr_hit=llr_hit, llr_miss=llr_miss,
        split_min_llr=4.0, split_min_opps=3, nuc_min_size=85,
    )
    # the 20bp core must NOT be emitted as a nuc; whole fragment -> accessible
    assert nucs == []
    assert (0, 100) in access


def test_topology_policy_preserves_ambiguous_hmm_nucleosome():
    # With sparse single-strand evidence, a short protected core plus neutral
    # flanks is unresolved, not evidence that the entire HMM footprint is open.
    obs = _obs((NONTARGET, 40), (MISS, 20), (NONTARGET, 40))
    llr_hit, llr_miss = _llr_tables()
    nucs, access = recall_nucs_in_read(
        obs, ns=[0], nl=[len(obs)], read_length=len(obs),
        llr_hit=llr_hit, llr_miss=llr_miss,
        split_min_llr=4.0, split_min_opps=3, nuc_min_size=85,
        recall_policy="topology",
    )
    assert [(n.start, n.length) for n in nucs] == [(0, 100)]
    assert nucs[0].el == 0 and nucs[0].er == 0
    assert access == []


def test_topology_policy_rejects_cut_that_shatters_one_nucleosome():
    # Both sides of the apparent cut are below the nucleosome floor. The old
    # policy demotes all 126 bp; topology-aware recall keeps the HMM occupancy.
    obs = _obs((MISS, 60), (HIT, 6), (MISS, 60))
    llr_hit, llr_miss = _llr_tables()
    nucs, access = recall_nucs_in_read(
        obs, ns=[0], nl=[len(obs)], read_length=len(obs),
        llr_hit=llr_hit, llr_miss=llr_miss,
        split_min_llr=4.0, split_min_opps=3, nuc_min_size=85,
        recall_policy="topology",
    )
    assert [(n.start, n.length) for n in nucs] == [(0, 126)]
    assert access == []


def test_topology_policy_still_splits_an_overmerged_pair():
    obs = _obs((MISS, 100), (HIT, 6), (MISS, 100))
    llr_hit, llr_miss = _llr_tables()
    nucs, access = recall_nucs_in_read(
        obs, ns=[0], nl=[len(obs)], read_length=len(obs),
        llr_hit=llr_hit, llr_miss=llr_miss,
        split_min_llr=4.0, split_min_opps=3, nuc_min_size=85,
        recall_policy="topology",
    )
    assert [(n.start, n.length) for n in nucs] == [(0, 100), (106, 100)]
    assert any(start == 100 and length == 6 for start, length in access)


def test_genuine_85bp_nuc_survives_edge_pass():
    obs = _obs((MISS, 85))
    llr_hit, llr_miss = _llr_tables()
    nucs, _ = recall_nucs_in_read(
        obs, ns=[0], nl=[len(obs)], read_length=len(obs),
        llr_hit=llr_hit, llr_miss=llr_miss,
        split_min_llr=4.0, split_min_opps=3, nuc_min_size=85,
    )
    assert len(nucs) == 1 and nucs[0].length >= 85


def test_phase_prior_splits_long_footprint_at_single_event():
    # 380bp footprint, all protected except ONE deamination near the predicted
    # linker (~190). Pass 1 (min_opps=3) can't split a single event; the phase
    # prior (nrl=185) lowers the bar there and splits into two ~nucleosomes.
    obs = _obs((MISS, 190), (HIT, 1), (MISS, 189))
    llr_hit, llr_miss = _llr_tables()
    kw = dict(ns=[0], nl=[len(obs)], read_length=len(obs),
              llr_hit=llr_hit, llr_miss=llr_miss,
              split_min_llr=4.0, split_min_opps=3, nuc_min_size=85)
    nucs_off, _ = recall_nucs_in_read(obs, phase_nrl=0, **kw)
    assert len(nucs_off) == 1
    nucs_on, _ = recall_nucs_in_read(obs, phase_nrl=185, **kw)
    assert len(nucs_on) == 2


def test_phase_prior_never_splits_signal_desert():
    # long fully-protected footprint with ZERO deamination -> no split even with
    # the phase prior on (the prior lowers the threshold but evidence is still
    # required at the predicted linker).
    obs = _obs((MISS, 380))
    llr_hit, llr_miss = _llr_tables()
    nucs, _ = recall_nucs_in_read(
        obs, ns=[0], nl=[len(obs)], read_length=len(obs),
        llr_hit=llr_hit, llr_miss=llr_miss,
        split_min_llr=4.0, split_min_opps=3, nuc_min_size=85, phase_nrl=185)
    assert len(nucs) == 1


def test_rederive_msps_merges_and_filters():
    msps = rederive_msps(
        original_msps=[(0, 10)],
        accessible_from_splits=[(10, 5), (200, 3)],
        read_length=300, msp_min_size=4,
    )
    # (0,10)+(10,5) merge into one 15bp MSP; the 3bp patch is filtered out
    assert (0, 15) in msps
    assert all(length >= 4 for _, length in msps)


def test_unify_drops_short_nuc_overlapping_tf():
    nucs = [NucCall(0, 50, 200, 255, 255), NucCall(100, 30, 150, 200, 200)]
    tf = [TFCall(start=105, length=20, llr=6.0, n_opps=4,
                 left_ambiguity=1, right_ambiguity=1)]
    kept = unify_nuc_calls_with_tf_calls(nucs, tf, unify_threshold=85)
    # the 30bp nuc overlapping the TF call is dropped; the 50bp one is kept
    # (both are < threshold, so overlap is what decides)
    assert [(k.start, k.length) for k in kept] == [(0, 50)]


def test_project_center_nuc_calls_keeps_quality_and_picks_center():
    n = 100
    calls = [
        NucCall(start=110, length=30, nq=200, el=255, er=128),  # center tile -> (10,30)
        NucCall(start=10, length=30, nq=1, el=1, er=1),         # first tile -> dropped
        NucCall(start=50, length=200, nq=150, el=64, er=64),    # covers middle -> (0,100)
    ]
    proj = project_center_nuc_calls(calls, n)
    by_start = {p.start: p for p in proj}
    assert set(by_start) == {10, 0}
    assert (by_start[10].length, by_start[10].nq, by_start[10].el, by_start[10].er) == (30, 200, 255, 128)
    assert by_start[0].length == 100  # whole-molecule projection


def test_unify_circular_drops_short_nuc_overlapping_wrapped_tf():
    n = 100
    # a wrapped TF call near the origin: covers [95,100) and [0,5)
    tf = [TFCall(start=95, length=10, llr=6.0, n_opps=4,
                 left_ambiguity=1, right_ambiguity=1)]
    nucs = [
        NucCall(start=2, length=20, nq=100, el=0, er=0),    # short, overlaps wrap -> drop
        NucCall(start=40, length=30, nq=100, el=0, er=0),   # short, no overlap -> keep
    ]
    kept = unify_circular_nuc_calls_with_tf_calls(nucs, tf, unify_threshold=85,
                                                  read_length=n)
    assert [(k.start, k.length) for k in kept] == [(40, 30)]


def test_promote_large_tf_to_nuc():
    # a nucleosome-sized protected TF call (>= threshold) is promoted to nuc+
    # with edges; a small TF stays in tf+.
    obs = _obs((MISS, 300))
    llr_hit, llr_miss = _llr_tables()
    tf = [
        TFCall(start=0, length=120, llr=10.0, n_opps=20,
               left_ambiguity=1, right_ambiguity=1),   # nucleosome-sized
        TFCall(start=200, length=20, llr=6.0, n_opps=4,
               left_ambiguity=1, right_ambiguity=1),   # real small footprint
    ]
    remaining, promoted = promote_large_tf_calls(
        tf, obs, llr_hit, llr_miss, threshold=90, nuc_min_size=85)
    assert len(promoted) == 1 and promoted[0].length >= 85
    assert [c.start for c in remaining] == [200]


def test_drop_short_nuc_overlapping_promoted():
    # Codex repro: a short (< unify_threshold) nuc starting slightly BEFORE a
    # promoted nucleosome must be dropped, so the start-order tiling does not
    # keep the short one and clip/discard the promoted one.
    promoted = [NucCall(5, 100, 200, 255, 255)]            # [5,105)
    short = [NucCall(0, 85, 100, 255, 255)]                # 85 < 90, overlaps
    assert drop_short_nucs_overlapping_promoted(short, promoted, 90) == []
    # full path: drop + add promoted, then tile -> promoted survives whole
    kept, _ = assemble_nuc_msp_tiling(
        drop_short_nucs_overlapping_promoted(short, promoted, 90) + promoted,
        span_lo=0, span_hi=300, msp_min_size=0, nuc_min_size=85)
    assert [(k.start, k.length) for k in kept] == [(5, 100)]
    # a long (>= threshold) nuc is NOT dropped; a non-overlapping short is kept
    long_nuc = [NucCall(0, 90, 100, 255, 255)]
    assert drop_short_nucs_overlapping_promoted(long_nuc, promoted, 90) == long_nuc
    far = [NucCall(300, 85, 100, 255, 255)]
    assert drop_short_nucs_overlapping_promoted(far, promoted, 90) == far


def test_circular_tiling_no_overlap_for_wrapped_nuc():
    # Codex repro (High): a nucleosome wrapping the origin must not get an MSP gap
    # derived linearly over its wrapped tail. A nuc [180,200)+[0,80) on a 200 bp
    # circle should leave a single MSP over the uncovered arc [80,180), with nucs
    # and MSPs tiling the circle exactly (no overlap, no gap).
    rl = 200
    nucs = [NucCall(180, 100, 200, 255, 255)]   # wraps: [180,200) + [0,80)
    kept, msps = assemble_circular_nuc_msp_tiling(
        nucs, rl, msp_min_size=1, nuc_min_size=85)
    assert [(k.start, k.length) for k in kept] == [(180, 100)]
    assert msps == [(80, 100)]
    cov = [0] * rl
    for s, length in [(k.start, k.length) for k in kept] + msps:
        for off in range(length):
            cov[(s + off) % rl] += 1
    assert all(c == 1 for c in cov)   # exact circular tiling: no overlap, no gap


def test_circular_tiling_empty_nucs_gives_whole_molecule_msp():
    # Codex repro (High): if circular unification drops all nucleosomes, the read
    # must tile as one accessible MSP over the whole molecule, not lose it.
    kept, msps = assemble_circular_nuc_msp_tiling([], 200, msp_min_size=0)
    assert kept == []
    assert msps == [(0, 200)]


def test_circular_tiling_fully_covered_overlap_tiles_exactly():
    # Codex repro (High): a fully-covered circle (no uncovered cut point) with
    # overlapping nucs must still tile exactly -- the origin can fall inside a
    # wrapped call, which previously produced double-covered bases.
    rl = 200
    nucs = [NucCall(0, 120, 200, 255, 255), NucCall(100, 120, 200, 255, 255)]
    kept, msps = assemble_circular_nuc_msp_tiling(nucs, rl, msp_min_size=0)
    cov = [0] * rl
    for s, length in [(k.start, k.length) for k in kept] + msps:
        for off in range(length):
            cov[(s + off) % rl] += 1
    assert all(c == 1 for c in cov)   # exact tiling: no base covered twice, none missed


def test_circular_tiling_whole_molecule_nuc_normalizes_start():
    # A projected center-copy run can become a whole-molecule nuc with nonzero
    # start. It must stay a full nuc, not split at the origin and demote the
    # short piece to MSP by the nuc_min_size floor.
    kept, msps = assemble_circular_nuc_msp_tiling(
        [NucCall(50, 200, 201, 123, 45)], 200, msp_min_size=0, nuc_min_size=85)
    assert [(k.start, k.length, k.nq, k.el, k.er) for k in kept] == [
        (0, 200, 201, 123, 45)
    ]
    assert msps == []


def test_nuc_qqq_aq_roundtrip():
    # nuc+QQQ (2 nucs) then tf+QQQ (1 tf): parse back to per-annotation triples
    aq = format_aq_array(
        nq_values=[255, 150], tf_q_values=[100], tf_lq_values=[255], tf_rq_values=[0],
        nuc_lq_values=[246, 200], nuc_rq_values=[255, 64],
    )
    parsed = parse_aq_array(aq, ["QQQ", "", "QQQ"], [2, 1, 1])
    assert parsed[0] == [255, 246, 255]   # nuc 0 (nq, el, er)
    assert parsed[1] == [150, 200, 64]    # nuc 1
    assert parsed[2] == []                # msp (no bytes)
    assert parsed[3] == [100, 255, 0]     # tf


def test_load_bundled_ddda_nuc_profile():
    import json
    from fiberhmm.inference.nuc_recaller import NucProfile, load_nuc_profile
    from fiberhmm.models import _bundled_model_path
    profile_path = _bundled_model_path('ddda_nuc_profile.json')
    prof = load_nuc_profile(profile_path)
    with open(profile_path) as handle:
        metadata = json.load(handle)
    assert isinstance(prof, NucProfile)
    assert metadata['kind'] == 'ddda_phase_posterior_v1'
    assert metadata['status'] == 'production'
    assert metadata['validation']['locked'] == '2026-08-30'
    assert 0.5 < prof.linker < 0.95          # DddA linker deam rate
    assert prof.radial[0] < prof.linker       # dyad core is protected
    assert prof.half >= 60
    assert prof.edge_prior_center > 0.0
    assert prof.edge_prior_sd > 0.0
    assert 9.0 < prof.rotation_period < 12.0
    assert prof.rotation_period_sd > 0.0
    assert prof.rotation_band_sd > 0.0
    assert prof.rotation_phase_bins > 0
    assert prof.rotation_fraction > 0.0
    assert prof.rotation_min_phase_information > 0.0
    assert prof.rotation_edge_break_min_surprisal > 0.0


def test_bundled_ddda_phase_posterior_parameters_are_locked():
    """Changing these values requires a new named profile and validation."""
    from fiberhmm.inference.nuc_recaller import load_nuc_profile
    from fiberhmm.models import _bundled_model_path

    profile = load_nuc_profile(_bundled_model_path('ddda_nuc_profile.json'))

    assert profile.half == 73
    assert profile.min_sep == 150
    assert profile.edge_frac == 0.82
    assert profile.edge_prior_center == 73.0
    assert profile.edge_prior_sd == 25.0
    assert profile.edge_likelihood_temperature == 1.0
    assert profile.rotation_period == 10.12
    assert profile.rotation_period_sd == 0.2
    assert profile.rotation_phase == 5.70
    assert profile.rotation_phase_sd == 0.8
    assert profile.rotation_phase_bins == 8
    assert profile.rotation_band_sd == 1.5
    assert profile.rotation_fraction == 1.0
    assert profile.rotation_edge_break_min_surprisal == 1.5
    assert profile.rotation_min_phase_information == 0.5


def test_bundled_ddda_phase_posterior_artifact_is_locked():
    """Any byte change requires a new named profile and fresh validation."""
    import hashlib
    from pathlib import Path

    from fiberhmm.models import _bundled_model_path

    profile_path = _bundled_model_path('ddda_nuc_profile.json')
    digest = hashlib.sha256(Path(profile_path).read_bytes()).hexdigest()

    assert digest == (
        'c86b05dc07e45392880e3460cf7f8880593ecad174e0a338d36ac53b7d0172d6'
    )


def test_radial_split_splits_dinucleosome_block():
    """DddA mode: an over-merged block (two low-deam nucs + a high-deam linker)
    is split into two nucleosomes; the linker becomes accessible."""
    from fiberhmm.inference.nuc_recaller import NucProfile, recall_nucs_in_read
    from fiberhmm.inference.tf_recaller import UNMETH_OFFSET
    rng = np.random.default_rng(0)
    HIT, MISS = 0, UNMETH_OFFSET
    L = 360
    obs = np.full(L, MISS, dtype=np.int64)

    def fill(a, b, rate):
        for x in range(a, b):
            obs[x] = HIT if rng.random() < rate else MISS

    fill(0, 150, 0.07)      # nucleosome 1 (protected interior)
    fill(150, 195, 0.74)    # linker (accessible)
    fill(195, 345, 0.07)    # nucleosome 2
    fill(345, 360, 0.74)

    radial = np.concatenate([np.full(30, 0.03), np.linspace(0.03, 0.12, 44)])
    prof = NucProfile(radial=radial, linker=0.74, half=73,
                      min_sep=150, edge_frac=0.82)

    nucs, access = recall_nucs_in_read(
        obs, [0], [L], L, None, None,
        split_min_llr=4.0, split_min_opps=3, nuc_min_size=85, nuc_profile=prof)

    assert len(nucs) == 2
    centers = sorted(n.start + n.length / 2 for n in nucs)
    assert 50 < centers[0] < 120 and 220 < centers[1] < 300
    # the linker region was freed up as accessible (-> MSP downstream)
    assert any(s <= 170 <= s + length for s, length in access)
    # edge-sharpness bytes are populated
    assert all(0 <= n.el <= 255 and 0 <= n.er <= 255 for n in nucs)


def test_ddda_density_rate_rejects_isolated_rotational_hit():
    opportunity = np.zeros(101, dtype=bool)
    deaminated = np.zeros(101, dtype=bool)
    opportunity[50] = True
    deaminated[50] = True

    rate = _smoothed_deam_rate(opportunity, deaminated)

    assert np.all(np.isnan(rate))


def test_ddda_density_rate_uses_wider_window_for_sparse_sequence():
    opportunity = np.zeros(101, dtype=bool)
    deaminated = np.zeros(101, dtype=bool)
    opportunity[[30, 40, 60, 70]] = True
    deaminated[[30, 70]] = True

    rate = _smoothed_deam_rate(opportunity, deaminated)

    assert np.isclose(rate[50], 0.41)


def test_ddda_density_prior_rejects_two_of_three_but_accepts_three_of_four():
    opportunity = np.zeros(101, dtype=bool)
    deaminated = np.zeros(101, dtype=bool)
    opportunity[[45, 50, 55]] = True
    deaminated[[45, 55]] = True
    two_of_three = _smoothed_deam_rate(
        opportunity, deaminated, sparse_win=None
    )[50]

    opportunity[60] = True
    deaminated[60] = True
    three_of_four = _smoothed_deam_rate(
        opportunity, deaminated, sparse_win=None
    )[50]

    threshold = 0.82 * 0.7356
    assert two_of_three < threshold < three_of_four


def test_ddda_density_edge_preserves_unresolved_topology():
    profile = NucProfile(
        radial=np.full(96, 0.05),
        linker=0.75,
        half=73,
        min_sep=150,
        edge_frac=0.82,
    )
    unsupported = np.full(240, np.nan)

    right, right_ambiguity = _find_density_edge(
        unsupported, 120, +1, 230, profile, fallback=198
    )
    left, left_ambiguity = _find_density_edge(
        unsupported, 120, -1, 10, profile, fallback=42
    )

    # With no molecule-supported edge, preserve the structural boundary and
    # state the uncertainty in the edge bytes; do not invent a +/-73-bp span.
    assert (left, right) == (42, 198)
    assert left_ambiguity >= 30 and right_ambiguity >= 30


def test_ddda_phase_edge_reports_prior_median_and_q0_on_complete_dropout():
    profile = NucProfile(
        radial=np.full(96, 0.05),
        linker=0.75,
        half=73,
        min_sep=150,
        edge_frac=0.82,
        edge_prior_center=73.0,
        edge_prior_sd=18.0,
        rotation_period=10.12,
        rotation_period_sd=0.4,
        rotation_phase_bins=8,
        rotation_band_sd=1.1,
        rotation_fraction=1.0,
        rotation_edge_break_min_surprisal=1.5,
        protected_hit=np.full(N_CTX, 0.02),
        accessible_hit=np.full(N_CTX, 0.75),
    )
    center = 120
    observations = np.full(280, NONTARGET, dtype=np.int32)
    opportunity = np.zeros(280, dtype=bool)
    deaminated = np.zeros(280, dtype=bool)
    llr_hit = np.full(N_CTX, -3.0, dtype=np.float64)
    llr_miss = np.full(N_CTX, 1.2, dtype=np.float64)

    edge, ambiguity = _find_density_edge(
        np.full(280, np.nan),
        center,
        +1,
        center + 110,
        profile,
        fallback=center + 90,
        opportunity=opportunity,
        deaminated=deaminated,
        observations=observations,
        llr_hit=llr_hit,
        llr_miss=llr_miss,
        phase_core_radius=42,
    )

    assert edge == center + 73
    assert ambiguity >= 30


def test_ddda_phase_edge_never_switches_to_topology_at_low_confidence():
    """A broad posterior changes Q, never the coordinate estimator.

    This guards the population-size cliff caused by sending low-confidence
    molecules to one of a few HMM/adjacent-dyad structural coordinates.
    """
    profile = NucProfile(
        radial=np.full(128, 0.05),
        linker=0.75,
        half=73,
        min_sep=150,
        edge_frac=0.82,
        edge_prior_center=73.0,
        edge_prior_sd=25.0,
        rotation_period=10.12,
        rotation_period_sd=0.4,
        rotation_phase_bins=8,
        rotation_band_sd=1.1,
        rotation_fraction=1.0,
        protected_hit=np.full(N_CTX, 0.02),
        accessible_hit=np.full(N_CTX, 0.75),
    )
    center = 140
    length = 360
    observations = np.full(length, NONTARGET, dtype=np.int32)
    opportunity = np.zeros(length, dtype=bool)
    deaminated = np.zeros(length, dtype=bool)
    # Deliberately sparse evidence leaves a broad posterior.
    for offset, code in ((24, MISS), (47, HIT), (82, MISS)):
        position = center + offset
        opportunity[position] = True
        deaminated[position] = code == HIT
        observations[position] = code
    llr_hit = np.full(N_CTX, -3.0, dtype=np.float64)
    llr_miss = np.full(N_CTX, 1.2, dtype=np.float64)

    results = [
        _find_density_edge(
            np.full(length, np.nan),
            center,
            +1,
            center + 120,
            profile,
            fallback=center + topology_offset,
            opportunity=opportunity,
            deaminated=deaminated,
            observations=observations,
            llr_hit=llr_hit,
            llr_miss=llr_miss,
            phase_core_radius=42,
        )
        for topology_offset in (75, 90, 110)
    ]

    assert len({edge for edge, _ambiguity in results}) == 1
    assert all(ambiguity >= 30 for _edge, ambiguity in results)


def test_ddda_density_edge_ignores_rotational_hits_inside_clean_nuc():
    length = 320
    center = 160
    true_right_edge = center + 73
    opportunity = np.zeros(length, dtype=bool)
    deaminated = np.zeros(length, dtype=bool)
    opportunity[::3] = True
    # Strong periodic internal hits should not become a linker transition.
    for position in range(center, true_right_edge, 10):
        nearest = position - (position % 3)
        deaminated[nearest] = True
    # Deterministic high linker rate after the true edge.
    linker_opportunities = np.flatnonzero(
        opportunity & (np.arange(length) >= true_right_edge)
    )
    deaminated[
        linker_opportunities[np.arange(len(linker_opportunities)) % 4 != 0]
    ] = True
    profile = NucProfile(
        radial=np.full(96, 0.05),
        linker=0.75,
        half=73,
        min_sep=150,
        edge_frac=0.82,
    )

    rate = _smoothed_deam_rate(opportunity, deaminated)
    edge, ambiguity = _find_density_edge(
        rate, center, +1, center + 110, profile
    )

    assert true_right_edge - 3 <= edge <= true_right_edge + 15
    assert ambiguity >= 0


def test_ddda_density_edge_requires_sustained_outward_linker():
    profile = NucProfile(
        radial=np.full(96, 0.05),
        linker=0.75,
        half=73,
        min_sep=150,
        edge_frac=0.82,
    )
    center = 120
    signal = np.full(260, np.nan)
    # A rotational band crosses the centered density threshold 23 bp before
    # the physical edge.  The real linker transition begins at +73.
    signal[center + 50] = 0.70
    signal[center + 73:] = 0.70
    opportunity = np.zeros(260, dtype=bool)
    deaminated = np.zeros(260, dtype=bool)
    opportunity[center + 52:center + 73:4] = True
    opportunity[center + 74:center + 105:4] = True
    deaminated[center + 74:center + 105:4] = True

    edge, _ambiguity = _find_density_edge(
        signal,
        center,
        +1,
        center + 110,
        profile,
        fallback=center + 80,
        opportunity=opportunity,
        deaminated=deaminated,
    )

    # No opportunity occurs exactly at +73, so the molecularly identified
    # boundary is the short interval between the last wrapped miss and first
    # linker opportunity rather than a forced single base.
    assert center + 71 <= edge <= center + 75


def test_ddda_density_edge_models_multiple_rotational_bands_before_linker():
    profile = NucProfile(
        radial=np.full(96, 0.08),
        linker=0.75,
        half=73,
        min_sep=150,
        edge_frac=0.82,
    )
    center = 120
    true_edge = center + 73
    opportunity = np.zeros(280, dtype=bool)
    opportunity[::3] = True
    deaminated = np.zeros(280, dtype=bool)
    # Three separate outward-facing turns are hit inside one wrapped particle.
    for offset in (42, 52, 62):
        local = np.flatnonzero(
            opportunity
            & (np.abs(np.arange(len(opportunity)) - (center + offset)) <= 2)
        )
        deaminated[local[0]] = True
    linker_sites = np.flatnonzero(
        opportunity & (np.arange(len(opportunity)) >= true_edge)
    )
    deaminated[linker_sites[np.arange(len(linker_sites)) % 4 != 0]] = True

    edge, ambiguity = _find_density_edge(
        _smoothed_deam_rate(opportunity, deaminated),
        center,
        +1,
        center + 110,
        profile,
        fallback=center + 80,
        opportunity=opportunity,
        deaminated=deaminated,
    )

    assert true_edge - 3 <= edge <= true_edge + 5
    assert ambiguity >= 0


def test_ddda_phase_edge_allows_jitter_and_skipped_turns():
    profile = NucProfile(
        radial=np.full(96, 0.08),
        linker=0.75,
        half=73,
        min_sep=150,
        edge_frac=0.82,
        rotation_period=10.12,
        rotation_period_sd=0.4,
        rotation_phase=5.7,
        rotation_phase_sd=0.8,
        rotation_phase_bins=8,
        rotation_band_sd=1.1,
        rotation_fraction=1.0,
        rotation_edge_break_min_surprisal=1.5,
        protected_hit=np.full(N_CTX, 0.02),
        accessible_hit=np.full(N_CTX, 0.75),
    )
    center = 120
    true_edge = center + 73
    opportunity = np.zeros(280, dtype=bool)
    deaminated = np.zeros(280, dtype=bool)
    observations = np.full(280, NONTARGET, dtype=np.int32)
    sites = np.arange(center + 1, center + 111, 2)
    opportunity[sites] = True
    observations[sites] = MISS

    # Same latent phase with 21- and 28-bp separations: individual turns were
    # skipped and the retained bands are not at exact 10-bp coordinates.
    for offset in (16, 37, 65):
        position = sites[int(np.argmin(np.abs(sites - (center + offset))))]
        deaminated[position] = True
        observations[position] = HIT
    linker_sites = sites[sites >= true_edge]
    linker_hits = linker_sites[np.arange(len(linker_sites)) % 4 != 0]
    deaminated[linker_hits] = True
    observations[linker_hits] = HIT
    llr_hit = np.full(N_CTX, -3.0, dtype=np.float64)
    llr_miss = np.full(N_CTX, 1.2, dtype=np.float64)

    edge, ambiguity = _find_density_edge(
        _smoothed_deam_rate(opportunity, deaminated),
        center,
        +1,
        center + 110,
        profile,
        fallback=center + 85,
        opportunity=opportunity,
        deaminated=deaminated,
        observations=observations,
        llr_hit=llr_hit,
        llr_miss=llr_miss,
    )

    assert true_edge - 3 <= edge <= true_edge + 5
    assert ambiguity < 30


def test_ddda_phase_edge_requires_direct_off_phase_break():
    profile = NucProfile(
        radial=np.full(128, 0.08),
        linker=0.75,
        half=73,
        min_sep=150,
        edge_frac=0.82,
        rotation_period=10.12,
        rotation_period_sd=0.4,
        rotation_phase=5.7,
        rotation_phase_sd=0.8,
        rotation_phase_bins=8,
        rotation_band_sd=1.1,
        rotation_fraction=1.0,
        rotation_edge_break_min_surprisal=1.5,
        protected_hit=np.full(N_CTX, 0.02),
        accessible_hit=np.full(N_CTX, 0.75),
    )
    center = 140
    length = 320
    opportunity = np.zeros(length, dtype=bool)
    deaminated = np.zeros(length, dtype=bool)
    observations = np.full(length, NONTARGET, dtype=np.int32)

    # Protected opportunities establish one particle and its latent phase.
    protected_offsets = np.arange(10, 71, 10)
    protected_sites = center + protected_offsets
    opportunity[protected_sites] = True
    observations[protected_sites] = MISS
    for offset in (16, 37, 65):
        position = center + offset
        opportunity[position] = True
        deaminated[position] = True
        observations[position] = HIT

    # A run of hits that continues the inferred rotational phase is not, by
    # itself, a directly observed linker edge even though its aggregate can
    # make all-linker beat all-wrapped.
    on_phase_offsets = np.asarray((76, 86, 96, 106, 116))
    on_phase_sites = center + on_phase_offsets
    opportunity[on_phase_sites] = True
    deaminated[on_phase_sites] = True
    observations[on_phase_sites] = HIT
    llr_hit = np.full(N_CTX, -3.0, dtype=np.float64)
    llr_miss = np.full(N_CTX, 1.2, dtype=np.float64)

    unresolved_edge, unresolved_ambiguity = _find_density_edge(
        _smoothed_deam_rate(opportunity, deaminated),
        center,
        +1,
        center + 120,
        profile,
        fallback=center + 90,
        opportunity=opportunity,
        deaminated=deaminated,
        observations=observations,
        llr_hit=llr_hit,
        llr_miss=llr_miss,
    )
    assert unresolved_ambiguity >= 30

    # A single half-turn-shifted hit now breaks that phase. Its exact offset is
    # not a rule: the context-conditioned posterior-predictive Bayes factor is.
    off_phase_site = center + 81
    opportunity[off_phase_site] = True
    deaminated[off_phase_site] = True
    observations[off_phase_site] = HIT
    resolved_edge, resolved_ambiguity = _find_density_edge(
        _smoothed_deam_rate(opportunity, deaminated),
        center,
        +1,
        center + 120,
        profile,
        fallback=center + 90,
        opportunity=opportunity,
        deaminated=deaminated,
        observations=observations,
        llr_hit=llr_hit,
        llr_miss=llr_miss,
    )
    assert center + 55 <= resolved_edge < off_phase_site
    assert resolved_ambiguity < unresolved_ambiguity


def test_ddda_extension_is_conditioned_on_particle_phase():
    profile = NucProfile(
        radial=np.full(96, 0.08),
        linker=0.75,
        half=73,
        min_sep=150,
        edge_frac=0.82,
        rotation_period=10.0,
        rotation_phase_bins=10,
        rotation_band_sd=1.3,
        rotation_fraction=1.0,
        protected_hit=np.full(N_CTX, 0.02),
        accessible_hit=np.full(N_CTX, 0.75),
    )
    dyad = 100
    core_start, core_end = 100, 160
    extension_start, extension_end = 160, 185
    llr_hit, llr_miss = _llr_tables()

    def evidence(extension_hits, extension_misses):
        obs = np.full(240, NONTARGET, dtype=np.int32)
        # The core learns one helical register from repeated hits, with
        # protected half-turn opportunities anchoring the opposite phase.
        for offset in (6, 16, 26, 36, 46, 56):
            obs[dyad + offset] = HIT
        for offset in (11, 21, 31, 41, 51):
            obs[dyad + offset] = MISS
        for offset in extension_hits:
            obs[dyad + offset] = HIT
        for offset in extension_misses:
            obs[dyad + offset] = MISS
        return _radial_extension_evidence(
            obs,
            extension_start,
            extension_end,
            dyad,
            profile,
            [],
            llr_hit,
            llr_miss,
            condition_start=core_start,
            condition_end=core_end,
        )

    on_phase = evidence((66, 76), (71, 81))
    broken_phase = evidence((71, 81), (66, 76))
    assert on_phase[1] > broken_phase[1] + 2.0
    assert on_phase[4] is not None and on_phase[4] < 1.5
    assert broken_phase[4] is not None and broken_phase[4] >= 1.5


def test_ddda_density_edge_does_not_rescue_without_linker_opportunities():
    profile = NucProfile(
        radial=np.full(96, 0.05),
        linker=0.75,
        half=73,
        min_sep=150,
        edge_frac=0.82,
    )
    center = 120
    signal = np.full(260, np.nan)
    signal[center + 50] = 0.70
    opportunity = np.zeros(260, dtype=bool)
    deaminated = np.zeros(260, dtype=bool)

    edge, ambiguity = _find_density_edge(
        signal,
        center,
        +1,
        center + 110,
        profile,
        fallback=center + 80,
        opportunity=opportunity,
        deaminated=deaminated,
    )

    # The unsupported rotational hit is rejected; the topology coordinate is
    # preserved and marked explicitly unresolved.
    assert edge == center + 80
    assert ambiguity >= 30


def test_radial_configuration_preserves_hmm_nuc_when_no_dyad_is_emitted():
    llr_hit, llr_miss = _llr_tables()
    nucs, access = validate_radial_access_in_read(
        _obs((MISS, 220)),
        original_ns=[10],
        original_nl=[180],
        radial_nucs=[],
        provisional_tf_calls=[],
        read_length=220,
        llr_hit=llr_hit,
        llr_miss=llr_miss,
        min_llr=4.0,
        min_opps=3,
        nuc_min_size=85,
    )

    assert [(call.start, call.length, call.nq, call.el, call.er)
            for call in nucs] == [(10, 180, 0, 0, 0)]
    assert access == []


def test_radial_configuration_preserves_short_hmm_tf_scan_space():
    llr_hit, llr_miss = _llr_tables()
    nucs, access = validate_radial_access_in_read(
        _obs((MISS, 120)),
        original_ns=[20],
        original_nl=[40],
        radial_nucs=[],
        provisional_tf_calls=[],
        read_length=120,
        llr_hit=llr_hit,
        llr_miss=llr_miss,
        min_llr=4.0,
        min_opps=3,
        nuc_min_size=85,
    )

    assert nucs == []
    assert access == [(20, 40)]


def test_radial_configuration_preserves_core_across_unsupported_outer_flanks():
    llr_hit, llr_miss = _llr_tables()
    nucs, access = validate_radial_access_in_read(
        _obs((MISS, 180)),
        original_ns=[0],
        original_nl=[180],
        radial_nucs=[NucCall(20, 140, 200, 240, 230)],
        provisional_tf_calls=[],
        read_length=180,
        llr_hit=llr_hit,
        llr_miss=llr_miss,
        min_llr=4.0,
        min_opps=3,
        nuc_min_size=85,
    )

    assert [(call.start, call.length, call.el, call.er)
            for call in nucs] == [(20, 140, 0, 0)]
    assert access == []


def test_radial_configuration_preserves_phase_cores_across_unsupported_gap():
    llr_hit, llr_miss = _llr_tables()
    nucs, access = validate_radial_access_in_read(
        _obs((MISS, 360)),
        original_ns=[0],
        original_nl=[360],
        radial_nucs=[
            NucCall(0, 145, 220, 255, 200),
            NucCall(215, 145, 210, 190, 255),
        ],
        provisional_tf_calls=[TFCall(170, 20, 8.0, 4, 5, 5)],
        read_length=360,
        llr_hit=llr_hit,
        llr_miss=llr_miss,
        min_llr=4.0,
        min_opps=3,
        nuc_min_size=85,
    )

    assert [(call.start, call.length, call.nq, call.el, call.er)
            for call in nucs] == [
                (0, 145, 220, 255, 0),
                (215, 145, 210, 0, 255),
            ]
    assert access == []


def test_radial_configuration_keeps_adjacent_dyads_without_inventing_linker():
    llr_hit, llr_miss = _llr_tables()
    nucs, access = validate_radial_access_in_read(
        _obs((MISS, 300)),
        original_ns=[0],
        original_nl=[300],
        radial_nucs=[
            NucCall(0, 150, 255, 255, 0),
            NucCall(150, 150, 255, 0, 255),
        ],
        provisional_tf_calls=[],
        read_length=300,
        llr_hit=llr_hit,
        llr_miss=llr_miss,
        min_llr=4.0,
        min_opps=3,
        nuc_min_size=85,
    )

    assert [(call.start, call.length, call.nq, call.el, call.er)
            for call in nucs] == [
                (0, 150, 255, 255, 0),
                (150, 150, 255, 0, 255),
            ]
    assert access == []


def test_radial_configuration_keeps_tf_gap_with_accessible_residue():
    llr_hit, llr_miss = _llr_tables()
    obs = _obs(
        (MISS, 145),
        (HIT, 25),
        (MISS, 20),
        (HIT, 25),
        (MISS, 145),
    )
    original = [
        NucCall(0, 145, 220, 255, 200),
        NucCall(215, 145, 210, 190, 255),
    ]
    nucs, access = validate_radial_access_in_read(
        obs,
        original_ns=[0],
        original_nl=[360],
        radial_nucs=original,
        provisional_tf_calls=[TFCall(170, 20, 8.0, 4, 5, 5)],
        read_length=360,
        llr_hit=llr_hit,
        llr_miss=llr_miss,
        min_llr=4.0,
        min_opps=3,
        nuc_min_size=85,
    )

    assert [(call.start, call.length) for call in nucs] == [(0, 145), (215, 145)]
    assert access == [(145, 70)]


def test_radial_configuration_rescues_sequence_supported_clipped_edges():
    llr_hit, llr_miss = _llr_tables()
    profile = NucProfile(
        radial=np.full(96, 0.05),
        linker=0.75,
        half=73,
        min_sep=150,
        edge_frac=0.82,
    )
    nucs, access = validate_radial_access_in_read(
        _obs((MISS, 200)),
        original_ns=[20],
        original_nl=[110],
        radial_nucs=[NucCall(0, 146, 220, 240, 230, dyad=73)],
        provisional_tf_calls=[],
        read_length=200,
        llr_hit=llr_hit,
        llr_miss=llr_miss,
        min_llr=4.0,
        min_opps=3,
        nuc_min_size=85,
        nuc_profile=profile,
    )

    assert [(call.start, call.length, call.el, call.er, call.dyad)
            for call in nucs] == [(0, 146, 240, 230, 73)]
    assert access == []


def test_radial_hmm_crossing_is_independent_of_edge_q_threshold():
    """Ambiguity 29 vs 30 may change Q, never the selected coordinates."""
    llr_hit, llr_miss = _llr_tables()
    profile = NucProfile(
        radial=np.full(96, 0.05),
        linker=0.75,
        half=73,
        min_sep=150,
        edge_frac=0.82,
    )

    def call_with(observations, ambiguity):
        resolved = ambiguity < 30
        edge_q = ambiguity_to_edge(ambiguity)
        nucs, _access = validate_radial_access_in_read(
            observations,
            original_ns=[20],
            original_nl=[110],
            radial_nucs=[NucCall(
                0,
                146,
                220,
                edge_q,
                edge_q,
                dyad=73,
                radial_start=0,
                radial_end=146,
                phase_resolved_left=resolved,
                phase_resolved_right=resolved,
            )],
            provisional_tf_calls=[],
            read_length=200,
            llr_hit=llr_hit,
            llr_miss=llr_miss,
            min_llr=4.0,
            min_opps=3,
            nuc_min_size=85,
            nuc_profile=profile,
        )
        return [(call.start, call.length) for call in nucs]

    protected = _obs((MISS, 200))
    accessible_flanks = _obs(
        (HIT, 20),
        (MISS, 110),
        (HIT, 16),
        (MISS, 54),
    )

    # Molecular configuration evidence, not the 29/30-bp Q boundary, decides
    # whether the posterior crossing can reclaim HMM-accessible sequence.
    assert call_with(protected, 29) == call_with(protected, 30) == [(0, 146)]
    assert call_with(accessible_flanks, 29) == call_with(
        accessible_flanks, 30,
    ) == [(20, 110)]


def test_radial_configuration_does_not_cross_supported_linker_before_tf():
    llr_hit, llr_miss = _llr_tables()
    profile = NucProfile(
        radial=np.full(96, 0.05),
        linker=0.75,
        half=73,
        min_sep=150,
        edge_frac=0.82,
    )
    obs = _obs((MISS, 130), (HIT, 6), (MISS, 64))
    nucs, access = validate_radial_access_in_read(
        obs,
        original_ns=[20],
        original_nl=[110],
        radial_nucs=[NucCall(20, 126, 220, 240, 230, dyad=73)],
        provisional_tf_calls=[TFCall(136, 10, 8.0, 4, 5, 5)],
        read_length=200,
        llr_hit=llr_hit,
        llr_miss=llr_miss,
        min_llr=4.0,
        min_opps=3,
        nuc_min_size=85,
        nuc_profile=profile,
    )

    assert [(call.start, call.length, call.el, call.er)
            for call in nucs] == [(20, 110, 240, 0)]
    assert access == []


def test_radial_configuration_preserves_stronger_abutting_tf_hypothesis():
    llr_hit = np.full(N_CTX, -3.0, dtype=np.float64)
    llr_miss = np.full(N_CTX, 2.0, dtype=np.float64)
    profile = NucProfile(
        radial=np.full(96, 0.05),
        linker=0.75,
        half=73,
        min_sep=150,
        edge_frac=0.82,
    )
    nucs, access = validate_radial_access_in_read(
        _obs((MISS, 200)),
        original_ns=[20],
        original_nl=[110],
        radial_nucs=[NucCall(20, 126, 220, 240, 230, dyad=73)],
        provisional_tf_calls=[TFCall(130, 16, 20.0, 8, 2, 2)],
        read_length=200,
        llr_hit=llr_hit,
        llr_miss=llr_miss,
        min_llr=4.0,
        min_opps=3,
        nuc_min_size=85,
        nuc_profile=profile,
    )

    assert [(call.start, call.length, call.er) for call in nucs] == [
        (20, 110, 0),
    ]
    assert access == []


def test_radial_configuration_uses_equivalent_adjacent_tf_as_edge_proposal():
    llr_hit = np.full(N_CTX, -3.0, dtype=np.float64)
    # Uniform TF protection is slightly preferred, but by <2 nats over the
    # whole flank: the one-nucleosome and nuc+TF models are indistinguishable.
    llr_miss = np.full(N_CTX, 1.36, dtype=np.float64)
    profile = NucProfile(
        radial=np.full(96, 0.05),
        linker=0.75,
        half=73,
        min_sep=150,
        edge_frac=0.82,
    )
    nucs, access = validate_radial_access_in_read(
        _obs((MISS, 200)),
        original_ns=[20],
        original_nl=[110],
        radial_nucs=[NucCall(
            20, 130, 220, 240, 230,
            dyad=73, radial_start=20, radial_end=150,
        )],
        provisional_tf_calls=[TFCall(130, 20, 20.0, 10, 2, 2)],
        read_length=200,
        llr_hit=llr_hit,
        llr_miss=llr_miss,
        min_llr=4.0,
        min_opps=3,
        nuc_min_size=85,
        nuc_profile=profile,
    )

    assert [(call.start, call.length, call.er) for call in nucs] == [
        (20, 130, 0),
    ]
    assert access == []


def test_radial_configuration_searches_past_truncated_radial_edge():
    llr_hit = np.full(N_CTX, -3.0, dtype=np.float64)
    llr_miss = np.full(N_CTX, 1.36, dtype=np.float64)
    profile = NucProfile(
        radial=np.full(96, 0.05),
        linker=0.75,
        half=73,
        min_sep=150,
        edge_frac=0.82,
    )
    nucs, access = validate_radial_access_in_read(
        _obs((MISS, 200)),
        original_ns=[20],
        original_nl=[100],
        radial_nucs=[NucCall(
            20, 100, 220, 240, 0,
            dyad=70, radial_start=20, radial_end=120,
        )],
        provisional_tf_calls=[TFCall(120, 20, 20.0, 10, 2, 2)],
        read_length=200,
        llr_hit=llr_hit,
        llr_miss=llr_miss,
        min_llr=4.0,
        min_opps=3,
        nuc_min_size=85,
        nuc_profile=profile,
    )

    # radial_end is the truncated emitted edge, not the search limit. The
    # adjacent TF edge is a valid alternative boundary inside the dyad search
    # envelope and is absorbed when the likelihoods are indistinguishable.
    assert [(call.start, call.length, call.er) for call in nucs] == [
        (20, 120, 0),
    ]
    assert access == []


def test_radial_completion_scores_multiple_tf_fragments_without_size_gate():
    llr_hit = np.full(N_CTX, -3.0, dtype=np.float64)
    llr_miss = np.full(N_CTX, 1.36, dtype=np.float64)
    profile = NucProfile(
        radial=np.full(96, 0.05),
        linker=0.75,
        half=73,
        min_sep=150,
        edge_frac=0.82,
    )
    call = NucCall(
        20, 100, 220, 240, 0,
        dyad=73, radial_start=20, radial_end=150,
    )

    completed = _complete_radial_nuc_from_adjacent_tf(
        call,
        _obs((MISS, 200)),
        [
            TFCall(120, 10, 10.0, 10, 2, 2),
            TFCall(135, 10, 10.0, 10, 2, 2),
        ],
        profile,
        left_limit=20,
        right_limit=150,
        nuc_min_size=85,
        llr_hit=llr_hit,
        llr_miss=llr_miss,
    )

    # The joint likelihood prefers the outer edge of the two-fragment chain.
    # The superseded >=130-bp acceptance rule would reject this 125-bp result.
    assert (completed.start, completed.length, completed.er) == (20, 125, 0)


def test_exclude_nucleosomes_from_msps_removes_expanded_overlap():
    result = exclude_nucleosomes_from_msps(
        [(0, 100), (120, 100)],
        [NucCall(80, 80, 0, 0, 0)],
        msp_min_size=5,
    )

    assert result == [(0, 80), (160, 60)]


@pytest.mark.parametrize("length", [1, 20, 30, 40])
def test_ddda_density_rate_on_reads_shorter_than_the_window(length):
    """Reads under 41 bp crashed the DddA radial recaller (shape mismatch)."""
    opportunity = np.ones(length, dtype=bool)
    deaminated = np.zeros(length, dtype=bool)
    deaminated[::2] = True

    rate = _smoothed_deam_rate(opportunity, deaminated)

    assert rate.shape == (length,)
    # The same molecule embedded in a long read with no evidence around it
    # gives the same windowed rate inside the molecule.
    pad = 60
    long_opp = np.zeros(length + 2 * pad, dtype=bool)
    long_deam = np.zeros(length + 2 * pad, dtype=bool)
    long_opp[pad:pad + length] = opportunity
    long_deam[pad:pad + length] = deaminated
    expected = _smoothed_deam_rate(long_opp, long_deam)[pad:pad + length]
    np.testing.assert_array_equal(np.isnan(rate), np.isnan(expected))
    np.testing.assert_allclose(rate[~np.isnan(rate)],
                               expected[~np.isnan(expected)])


def test_ddda_radial_recall_handles_a_30bp_read():
    from fiberhmm.core.model_io import load_model
    from fiberhmm.inference.nuc_recaller import (
        attach_nuc_profile_emissions,
        load_nuc_profile,
        radial_split_in_read,
        validate_radial_access_in_read,
    )
    from fiberhmm.inference.tf_recaller import (
        build_conditional_hit_tables,
        build_llr_tables,
    )
    from fiberhmm.models import _bundled_model_path, get_model_path

    model = load_model(get_model_path('ddda', 'nuc_refine'))
    hit, miss = build_llr_tables(model)
    protected_hit, accessible_hit = build_conditional_hit_tables(model)
    profile = attach_nuc_profile_emissions(
        load_nuc_profile(_bundled_model_path('ddda_nuc_profile.json')),
        protected_hit, accessible_hit)
    length = 30
    rng = np.random.default_rng(3)
    # Non-target 8193; target hits carry their context code (0-4095),
    # target misses context + 4097.
    obs = np.full(length, 8193, dtype=np.int32)
    targets = rng.random(length) < 0.4
    hits = targets & (rng.random(length) < 0.3)
    contexts = rng.integers(0, 4096, length).astype(np.int32)
    obs[targets] = contexts[targets] + 4097
    obs[hits] = contexts[hits]
    ns, nl = [0], [length]

    nucs, access = radial_split_in_read(obs, ns, nl, length, profile, 85, hit, miss)
    validated, accessible = validate_radial_access_in_read(
        obs, ns, nl, nucs, (), length, hit, miss, min_llr=4.0, min_opps=3,
        nuc_min_size=85, nuc_profile=profile)
    for call in list(validated):
        assert 0 <= call.start and call.start + call.length <= length
    for start, size in accessible:
        assert 0 <= start and start + size <= length
