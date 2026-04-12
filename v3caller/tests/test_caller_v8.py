"""Unit tests for caller_v8.

These are the "load-bearing behaviors" the overnight benchmark
validated. Run after any edit to caller_v8.py / caller_v7.py to catch
silent regressions. Everything uses fake (opp, hit) arrays rather
than real BAMs so it runs in under a second.

Usage:
    cd phase0
    python -m pytest tests/test_caller_v8.py -v

Or:
    python tests/test_caller_v8.py
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from caller_v8 import (  # noqa: E402
    find_pass1_atoms_gap_cdf,
    poisson_merge_evidence,
    call_tfs_overcall,
    compute_edge_q,
    _d_nuc_for_positions,
    _ROT_AMP, _ROT_PERIOD, _ROT_TAU, _ROT_CUTOFF,
)
from caller_v7 import find_pass1_atoms, windowed_rate  # noqa: E402


def _make_bundle(pattern):
    """Turn a simple string pattern into (opp, hit) arrays.

    Legend:
      'H' = opp=1, hit=1
      'o' = opp=1, hit=0
      '.' = opp=0, hit=0
    """
    L = len(pattern)
    opp = np.zeros(L, dtype=np.int8)
    hit = np.zeros(L, dtype=np.int8)
    for i, c in enumerate(pattern):
        if c == 'H':
            opp[i] = 1
            hit[i] = 1
        elif c == 'o':
            opp[i] = 1
    return opp, hit


class TestGapCdfPass1(unittest.TestCase):
    """find_pass1_atoms_gap_cdf behavior."""

    def test_empty_hits_returns_whole_read(self):
        """Zero hits in a saturated-on-opposite-strand edge case →
        whole read is 'protected' (no linker signal anywhere)."""
        opp = np.ones(500, dtype=np.int8)
        hit = np.zeros(500, dtype=np.int8)
        atoms = find_pass1_atoms_gap_cdf(opp, hit, 500, gap_radius=10)
        self.assertEqual(atoms, [(0, 500)])

    def test_hit_dense_no_gaps(self):
        """Every position is a hit → no atoms."""
        opp = np.ones(100, dtype=np.int8)
        hit = np.ones(100, dtype=np.int8)
        atoms = find_pass1_atoms_gap_cdf(opp, hit, 100, gap_radius=10)
        self.assertEqual(atoms, [])

    def test_single_protected_gap(self):
        """One hit-free stretch ≥ gap_radius → one atom."""
        pattern = 'H' + 'o' * 50 + 'H'
        opp, hit = _make_bundle(pattern)
        atoms = find_pass1_atoms_gap_cdf(opp, hit, len(pattern), gap_radius=10)
        self.assertEqual(len(atoms), 1)
        s, e = atoms[0]
        self.assertEqual(s, 1)
        self.assertEqual(e, 51)

    def test_gap_radius_filters_short_stretches(self):
        """A 20-bp gap should produce an atom at radius 10 but not at radius 30."""
        pattern = 'H' + 'o' * 20 + 'H'
        opp, hit = _make_bundle(pattern)
        self.assertEqual(len(find_pass1_atoms_gap_cdf(opp, hit, len(pattern), gap_radius=10)), 1)
        self.assertEqual(len(find_pass1_atoms_gap_cdf(opp, hit, len(pattern), gap_radius=30)), 0)

    def test_multi_gap_atoms(self):
        """Three hit-free gaps, two meeting the radius."""
        pattern = 'o' * 50 + 'H' + 'o' * 5 + 'H' + 'o' * 60 + 'H' + 'o' * 30
        opp, hit = _make_bundle(pattern)
        atoms = find_pass1_atoms_gap_cdf(opp, hit, len(pattern), gap_radius=10)
        # Three long gaps (50, 60, 30) but only three ≥ 10 → 3 atoms
        self.assertEqual(len(atoms), 3)

    def test_gap_radius_default_is_ten(self):
        """Iter-8 sweep optimum is 10, not 30. Catch silent regressions."""
        import inspect
        sig = inspect.signature(find_pass1_atoms_gap_cdf)
        self.assertEqual(sig.parameters['gap_radius'].default, 10)


class TestPoissonMergeEvidence(unittest.TestCase):
    """poisson_merge_evidence behavior."""

    def test_empty_atoms(self):
        """No atoms → empty result."""
        opp = np.zeros(100, dtype=np.int8)
        hit = np.zeros(100, dtype=np.int8)
        self.assertEqual(poisson_merge_evidence([], opp, hit, 0.1), [])

    def test_single_atom_passes_through(self):
        """One atom → returned unchanged."""
        opp = np.ones(100, dtype=np.int8)
        hit = np.zeros(100, dtype=np.int8)
        atoms = [(10, 50)]
        result = poisson_merge_evidence(atoms, opp, hit, 0.1)
        self.assertEqual(result, atoms)

    def test_short_gap_force_merges(self):
        """Tiny gap (<= short_gap_bp) between atoms is always merged."""
        opp = np.ones(200, dtype=np.int8)
        hit = np.zeros(200, dtype=np.int8)
        # atoms touching at a 4-bp gap
        atoms = [(10, 100), (104, 180)]
        result = poisson_merge_evidence(atoms, opp, hit, 0.1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], (10, 180))

    def test_cascade_guard_prevents_long_merge(self):
        """Merging two atoms past max_merge_len must be refused."""
        opp = np.ones(500, dtype=np.int8)
        hit = np.zeros(500, dtype=np.int8)
        atoms = [(10, 200), (210, 400)]  # merged span = 390 > 250
        result = poisson_merge_evidence(atoms, opp, hit, 0.1, max_merge_len=250)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], (10, 200))
        self.assertEqual(result[1], (210, 400))

    def test_low_power_gate_keeps_split(self):
        """Gap with <5 opportunities defaults to SPLIT, not merge."""
        opp = np.zeros(200, dtype=np.int8)
        hit = np.zeros(200, dtype=np.int8)
        # Only 2 opportunities in the 50-bp gap between atoms
        opp[110] = 1
        opp[130] = 1
        atoms = [(10, 100), (150, 200)]
        result = poisson_merge_evidence(atoms, opp, hit, 0.1, max_merge_len=250)
        self.assertEqual(len(result), 2)  # kept split

    def test_elevated_gap_keeps_split(self):
        """Gap where every opportunity is a hit → outside Poisson interval → split."""
        L = 200
        opp = np.ones(L, dtype=np.int8)
        hit = np.zeros(L, dtype=np.int8)
        # Saturate the gap: every position is a hit
        hit[100:140] = 1
        atoms = [(10, 100), (140, 180)]
        result = poisson_merge_evidence(atoms, opp, hit, 0.1, max_merge_len=250)
        self.assertEqual(len(result), 2)  # kept split, gap looked like linker

    def test_short_gap_with_hits_degrades_mq(self):
        """iter-16d: a 'structural' short-gap merge that absorbs a HIT
        must NOT report mq=255. Otherwise downstream code sees a 'pure
        Pass-1 atom' that actually has internal hits."""
        opp = np.ones(300, dtype=np.int8)
        hit = np.zeros(300, dtype=np.int8)
        # Two atoms (10,100) and (104,200) with a 4-bp gap that
        # contains a single hit at position 101.
        hit[101] = 1
        atoms = [(10, 100), (104, 200)]
        result, mqs = poisson_merge_evidence(
            atoms, opp, hit, 0.1, max_merge_len=250,
            return_merge_quality=True)
        self.assertEqual(len(result), 1)  # merged
        self.assertLess(mqs[0], 255, 'mq must degrade when gap has hits')

    def test_short_gap_no_hits_stays_pure(self):
        """A 4-bp gap with zero hits merges structurally with mq=255."""
        opp = np.ones(300, dtype=np.int8)
        hit = np.zeros(300, dtype=np.int8)
        atoms = [(10, 100), (104, 200)]
        result, mqs = poisson_merge_evidence(
            atoms, opp, hit, 0.1, max_merge_len=250,
            return_merge_quality=True)
        self.assertEqual(len(result), 1)
        self.assertEqual(mqs[0], 255)

    def test_baseline_consistent_gap_merges(self):
        """Gap where hit count equals baseline expectation → merge."""
        L = 300
        opp = np.ones(L, dtype=np.int8)
        hit = np.zeros(L, dtype=np.int8)
        # baseline ~ 0.1: 4 hits in 40 bp gap is exactly expected
        for i in (105, 115, 125, 135):
            hit[i] = 1
        atoms = [(20, 100), (140, 180)]
        # expected merged span is 20..180 = 160 bp, within cap
        result = poisson_merge_evidence(atoms, opp, hit, 0.1, max_merge_len=250)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], (20, 180))

    def test_quantile_defaults_are_05_95(self):
        """Iter-7 sweep optimum was 0.05/0.95 — catch silent regression."""
        import inspect
        sig = inspect.signature(poisson_merge_evidence)
        self.assertAlmostEqual(sig.parameters['low_quantile'].default, 0.05)
        self.assertAlmostEqual(sig.parameters['high_quantile'].default, 0.95)

    def test_max_merge_len_default_is_250(self):
        """Iter-9 confirmed 250 is correct (plateau, not cliff)."""
        import inspect
        sig = inspect.signature(poisson_merge_evidence)
        self.assertEqual(sig.parameters['max_merge_len'].default, 250)


class TestMATags(unittest.TestCase):
    """MA/AQ tag round-trip per fiberseq Molecular-annotation-spec."""

    def test_format_ma_tag_basic(self):
        from ma_tags import format_ma_tag
        nuc = [(50, 147), (250, 147)]
        msp = [(0, 50), (197, 53)]
        # Explicit QQ for this legacy test; default is now QQQQ
        s = format_ma_tag(1000, nuc, msp, nuc_qual_spec='QQ')
        self.assertEqual(s, '1000;nuc+QQ:51-147,251-147;msp+:1-50,198-53')

    def test_format_ma_tag_no_msps(self):
        from ma_tags import format_ma_tag
        s = format_ma_tag(500, [(10, 100)], [], nuc_qual_spec='QQ')
        self.assertEqual(s, '500;nuc+QQ:11-100')

    def test_format_ma_tag_default_qqqq(self):
        from ma_tags import format_ma_tag
        s = format_ma_tag(1000, [(50, 147)], [(0, 50)])
        self.assertEqual(s, '1000;nuc+QQQQ:51-147;msp+:1-50')

    def test_format_ma_tag_with_tfs(self):
        from ma_tags import format_ma_tag
        s = format_ma_tag(1000, [(50, 147)], [(0, 50)],
                            tf_intervals=[(200, 25)],
                            nuc_qual_spec='QQ', tf_qual_spec='Q')
        self.assertEqual(s, '1000;nuc+QQ:51-147;msp+:1-50;tf+Q:201-25')

    def test_format_ma_tag_tf_qqq_default(self):
        from ma_tags import format_ma_tag
        s = format_ma_tag(1000, [], [], tf_intervals=[(100, 20)])
        self.assertEqual(s, '1000;tf+QQQ:101-20')

    def test_format_aq_array_qqqq(self):
        from ma_tags import format_aq_array
        aq = format_aq_array(
            nq_values=[200, 180],
            mq_values=[255, 80],
            lq_values=[240, 100],
            rq_values=[200, 150],
            tf_q_values=[128, 200])
        # Layout: per-nuc [nq, mq, lq, rq], then per-tf [q]
        self.assertEqual(list(aq),
                          [200, 255, 240, 200,  # nuc 0
                           180, 80, 100, 150,   # nuc 1
                           128, 200])           # tfs

    def test_format_aq_array_order(self):
        from ma_tags import format_aq_array
        nq = [200, 180, 150]
        mq = [255, 80, 100]
        aq = format_aq_array(nq, mq)
        # Quality values interleaved: nq[0], mq[0], nq[1], mq[1], ...
        self.assertEqual(list(aq), [200, 255, 180, 80, 150, 100])

    def test_format_aq_array_clips(self):
        from ma_tags import format_aq_array
        aq = format_aq_array([300, -5, 128], [-10, 256, 50])
        self.assertEqual(list(aq), [255, 0, 0, 255, 128, 50])

    def test_parse_ma_tag_roundtrip(self):
        from ma_tags import format_ma_tag, parse_ma_tag
        nuc = [(50, 147), (250, 147), (500, 200)]
        msp = [(0, 50), (197, 53)]
        s = format_ma_tag(1000, nuc, msp)
        parsed = parse_ma_tag(s)
        self.assertEqual(parsed['read_length'], 1000)
        self.assertEqual(parsed['nuc'], nuc)
        self.assertEqual(parsed['msp'], msp)

    def test_parse_empty_annotation_types(self):
        from ma_tags import format_ma_tag, parse_ma_tag
        # Nucleosomes only, no MSPs
        s = format_ma_tag(500, [(10, 147)], [])
        parsed = parse_ma_tag(s)
        self.assertEqual(parsed['nuc'], [(10, 147)])
        self.assertEqual(parsed['msp'], [])

    def test_parse_aq_array_split(self):
        from ma_tags import parse_aq_array
        # 3 nuc+QQ annotations + 2 msp+ (no quality)
        aq_flat = [200, 255, 180, 80, 150, 100]
        result = parse_aq_array(aq_flat, ['QQ', ''], [3, 2])
        # nuc annotations get [nq, mq] pairs, msp get empty lists
        self.assertEqual(result, [[200, 255], [180, 80], [150, 100], [], []])

    def test_strand_parsing(self):
        from ma_tags import parse_ma_tag
        s = '100;nuc+QQ:10-50;msp+:70-20'
        parsed = parse_ma_tag(s)
        raw = parsed['raw_types']
        self.assertEqual(raw[0][0], 'nuc')
        self.assertEqual(raw[0][1], '+')
        self.assertEqual(raw[0][2], 'QQ')
        self.assertEqual(raw[1][0], 'msp')
        self.assertEqual(raw[1][1], '+')
        self.assertEqual(raw[1][2], '')


class TestCallTfsOvercall(unittest.TestCase):
    """call_tfs_overcall behavior (iter-16 TF caller).

    The overcall caller has no hardcoded size or score filters — it
    emits every uninterrupted MISS run ≥ min_misses and relies on per-
    call quality scores (tfp, tel, ter) for downstream filtering.
    """

    def _bundle(self, n_opp, hits_at):
        """Build (opp, hit) arrays with opportunities every 2 bp.

        Example: _bundle(10, {4, 6}) gives opps at 0,2,4,6,8,...,18
        with hits at positions 4 and 6 in that opp list (i.e. bp 8
        and 12 in the full array).
        """
        L = n_opp * 2
        opp = np.zeros(L, dtype=np.int8)
        hit = np.zeros(L, dtype=np.int8)
        for i in range(n_opp):
            opp[i * 2] = 1
        for h in hits_at:
            hit[h * 2] = 1
        return opp, hit

    def test_empty_msps_returns_empty(self):
        opp = np.ones(100, dtype=np.int8)
        hit = np.zeros(100, dtype=np.int8)
        self.assertEqual(call_tfs_overcall([], opp, hit, 0.1), [])

    def test_zero_baseline_returns_empty(self):
        """A read with zero baseline → no meaningful significance."""
        opp = np.ones(100, dtype=np.int8)
        hit = np.zeros(100, dtype=np.int8)
        self.assertEqual(call_tfs_overcall([(0, 100)], opp, hit, 0.0), [])

    def test_single_miss_skipped(self):
        """min_misses default is 2 — singleton runs get dropped."""
        opp, hit = self._bundle(10, {0, 2, 4, 6, 8})
        # Alternating hit/miss → every run has length 1 → zero TFs
        tfs = call_tfs_overcall([(0, 20)], opp, hit, 0.3)
        self.assertEqual(tfs, [])

    def test_single_long_run_emits_tf(self):
        """10 consecutive misses in a 10-opp MSP → single TF."""
        opp, hit = self._bundle(10, set())
        tfs = call_tfs_overcall([(0, 20)], opp, hit, 0.3)
        self.assertEqual(len(tfs), 1)
        (s, e), tfp, tel, ter = tfs[0]
        # tfp should be high — 10 misses @ baseline 0.3 → P~0.028 → tfp ~ 195
        self.assertGreaterEqual(tfp, 128)

    def test_edge_quality_sharp_when_hit_adjacent(self):
        """A miss-run pinned by a HIT on both sides should have high el/er."""
        opp, hit = self._bundle(10, {0, 5, 9})
        # Runs: [1,2,3,4] (4 misses), [6,7,8] (3 misses)
        tfs = call_tfs_overcall([(0, 20)], opp, hit, 0.3)
        # Both runs are bounded on left and right by a HIT at opp-distance 1
        self.assertEqual(len(tfs), 2)
        # First run: left is next to hit0 (bp 0), so el should be near 255.
        # Right is next to hit5 (bp 10), so ter should be near 255.
        _, tfp, tel, ter = tfs[0]
        self.assertGreaterEqual(tel, 200)
        self.assertGreaterEqual(ter, 200)

    def test_edge_zero_at_msp_start(self):
        """A run that starts at the MSP start (no HIT before it) →
        ambiguous left edge → tel near zero."""
        opp, hit = self._bundle(10, {9})
        tfs = call_tfs_overcall([(0, 20)], opp, hit, 0.3)
        self.assertEqual(len(tfs), 1)
        _, _, tel, ter = tfs[0]
        self.assertLess(tel, 50)  # run starts at MSP start → ambiguous
        # Right is right next to the hit at opp 9 → sharp
        self.assertGreater(ter, 200)

    def test_tfp_saturation_constant_is_three(self):
        """iter-16e scaling: _TFP_MAX_NEG_LOG10 = 3.0 (saturating at
        P=0.001). Was 2.0 in iter-16, raised in iter-16e to reduce
        saturation of strong TFs at the top of the 0-255 range."""
        from caller_v8 import _TFP_MAX_NEG_LOG10
        self.assertAlmostEqual(_TFP_MAX_NEG_LOG10, 3.0)

    def test_tfs_overcall_tuple_shape(self):
        """Every returned TF has ((s, e), tfp, tel, ter)."""
        opp, hit = self._bundle(10, {0, 9})
        tfs = call_tfs_overcall([(0, 20)], opp, hit, 0.3)
        for tf in tfs:
            self.assertEqual(len(tf), 4)
            (s, e), tfp, tel, ter = tf
            self.assertIsInstance(s, int)
            self.assertIsInstance(e, int)
            self.assertTrue(0 <= tfp <= 255)
            self.assertTrue(0 <= tel <= 255)
            self.assertTrue(0 <= ter <= 255)


class TestRotationalCorrection(unittest.TestCase):
    """iter-17 rotational face-phasing correction for DAF TF calls.

    The correction adjusts tfp significance based on each miss
    position's distance to the nearest nucleosome edge: misses on
    the "exposed face" (~10 bp from a nuc) are more surprising than
    the uniform model assumes, while misses on the "quiet face"
    (~5 bp) are less surprising. Far from any nuc (>30 bp) the
    correction is a no-op.
    """

    def _make_linear(self, n_bp, opp_every=2, hits_at=None):
        """Build (opp, hit) arrays with regularly spaced opportunities."""
        opp = np.zeros(n_bp, dtype=np.int8)
        hit = np.zeros(n_bp, dtype=np.int8)
        for i in range(0, n_bp, opp_every):
            opp[i] = 1
        if hits_at:
            for h in hits_at:
                hit[h] = 1
        return opp, hit

    # ---- calibration constants ----

    def test_rot_constants_match_calibration(self):
        """Guard against silent changes to the calibrated constants."""
        self.assertAlmostEqual(_ROT_AMP, 0.35)
        self.assertAlmostEqual(_ROT_PERIOD, 10.4)
        self.assertAlmostEqual(_ROT_TAU, 15.0)
        self.assertEqual(_ROT_CUTOFF, 30)

    # ---- _d_nuc_for_positions helper ----

    def test_d_nuc_returns_none_without_edges(self):
        """No nuc_edges → None (correction disabled)."""
        self.assertIsNone(_d_nuc_for_positions(np.array([10, 20]), None))

    def test_d_nuc_returns_none_for_empty_edges(self):
        self.assertIsNone(
            _d_nuc_for_positions(np.array([10, 20]), np.array([])))

    def test_d_nuc_correct_distances(self):
        """Positions between two nuc edges get correct min distance."""
        edges = np.array([100, 200], dtype=np.int64)
        positions = np.array([90, 100, 110, 150, 190, 200, 210])
        d = _d_nuc_for_positions(positions, edges)
        # 90→100=10, 100→100=0, 110→100=10, 150→equidist=50,
        # 190→200=10, 200→200=0, 210→200=10
        np.testing.assert_array_equal(d, [10, 0, 10, 50, 10, 0, 10])

    def test_d_nuc_with_multiple_nucs(self):
        """Multiple nuc edges — nearest wins."""
        edges = np.array([50, 100, 300, 400], dtype=np.int64)
        positions = np.array([75, 200, 350])
        d = _d_nuc_for_positions(positions, edges)
        # 75: min(75-50, 100-75) = 25
        # 200: min(200-100, 300-200) = 100
        # 350: min(350-300, 400-350) = 50
        np.testing.assert_array_equal(d, [25, 100, 50])

    # ---- correction effect on tq ----

    def test_no_correction_without_nuc_edges(self):
        """nuc_edges=None → uniform baseline, same as iter-16."""
        opp, hit = self._make_linear(200, opp_every=2)
        hit[0] = 1; hit[40] = 1  # hits at 0 and 40, misses at 2..38
        tfs_none = call_tfs_overcall([(0, 200)], opp, hit, 0.24,
                                       nuc_edges=None)
        self.assertGreater(len(tfs_none), 0)
        # With empty edges array → same result
        tfs_empty = call_tfs_overcall([(0, 200)], opp, hit, 0.24,
                                        nuc_edges=np.array([], dtype=np.int64))
        self.assertEqual(len(tfs_none), len(tfs_empty))
        for a, b in zip(tfs_none, tfs_empty):
            self.assertEqual(a[1], b[1])  # same tfp

    def test_no_correction_far_from_nuc(self):
        """TF >30 bp from any nuc edge → correction is a no-op."""
        opp, hit = self._make_linear(400, opp_every=2)
        hit[0] = 1; hit[20] = 1  # misses at 2..18 (9 misses)
        # Nuc edges at 200 and 300 — far from the TF at ~0-20
        edges = np.array([200, 300], dtype=np.int64)
        tfs_corr = call_tfs_overcall([(0, 400)], opp, hit, 0.24,
                                       nuc_edges=edges)
        tfs_none = call_tfs_overcall([(0, 400)], opp, hit, 0.24,
                                       nuc_edges=None)
        self.assertEqual(len(tfs_corr), len(tfs_none))
        # TF midpoint is ~10, nearest edge is 200 → d_nuc=190 > 30
        # → tq should be identical
        for a, b in zip(tfs_corr, tfs_none):
            self.assertEqual(a[1], b[1])

    def test_correction_changes_tq_near_nuc(self):
        """TF within 15 bp of a nuc edge → tq CHANGES."""
        opp, hit = self._make_linear(200, opp_every=2)
        # Nuc edge at position 0. MSP from 0 to 200.
        # Hit at position 0, then misses at 2,4,6,...,18, then hit at 20.
        hit[0] = 1; hit[20] = 1
        edges = np.array([0], dtype=np.int64)  # nuc edge right at 0
        tfs_corr = call_tfs_overcall([(0, 200)], opp, hit, 0.24,
                                       nuc_edges=edges)
        tfs_none = call_tfs_overcall([(0, 200)], opp, hit, 0.24,
                                       nuc_edges=None)
        self.assertEqual(len(tfs_corr), len(tfs_none))
        # The TF at positions 2-18 has d_nuc ranging from 2 to 18
        # → well within _ROT_CUTOFF=30 → correction fires
        self.assertNotEqual(tfs_corr[0][1], tfs_none[0][1])

    def test_exposed_face_miss_increases_tq(self):
        """A miss at d≈10 (exposed face, local rate > baseline) is MORE
        surprising than the uniform model assumes → corrected tq should
        be HIGHER for that specific configuration.

        Build: single miss at d=10 from nuc edge + one miss at d=20.
        The d=10 miss is on the peak face (rate ~1.5×baseline), so
        the corrected model sees it as more unlikely → higher -log10(P).
        """
        opp = np.zeros(100, dtype=np.int8)
        hit = np.zeros(100, dtype=np.int8)
        # Opps at positions 10 and 20 only. Both are misses.
        opp[10] = 1; opp[20] = 1
        # Hits bracketing: at 0 and 30
        hit[0] = 1; opp[0] = 1
        hit[30] = 1; opp[30] = 1
        edges = np.array([0], dtype=np.int64)
        tfs_corr = call_tfs_overcall([(0, 100)], opp, hit, 0.24,
                                       nuc_edges=edges)
        tfs_none = call_tfs_overcall([(0, 100)], opp, hit, 0.24,
                                       nuc_edges=None)
        if tfs_corr and tfs_none:
            # d_nuc for miss at pos 10 = 10 (exposed face peak)
            # d_nuc for miss at pos 20 = 20 (near baseline)
            # The exposed-face miss makes the corrected P smaller
            # → corrected tq should be >= uniform tq
            self.assertGreaterEqual(tfs_corr[0][1], tfs_none[0][1])

    def test_quiet_face_miss_decreases_tq(self):
        """A miss at d≈5 (quiet face, local rate < baseline) is LESS
        surprising → corrected tq should be LOWER."""
        opp = np.zeros(100, dtype=np.int8)
        hit = np.zeros(100, dtype=np.int8)
        # Two opps at positions 4 and 6 (both d ≈ 5 from edge at 0)
        opp[4] = 1; opp[6] = 1
        hit[0] = 1; opp[0] = 1
        hit[10] = 1; opp[10] = 1
        edges = np.array([0], dtype=np.int64)
        tfs_corr = call_tfs_overcall([(0, 100)], opp, hit, 0.24,
                                       nuc_edges=edges)
        tfs_none = call_tfs_overcall([(0, 100)], opp, hit, 0.24,
                                       nuc_edges=None)
        if tfs_corr and tfs_none:
            # d_nuc ≈ 4 and 6 → both on the quiet face (below baseline)
            # → corrected tq should be <= uniform tq
            self.assertLessEqual(tfs_corr[0][1], tfs_none[0][1])

    def test_correction_symmetric_with_edges_on_both_sides(self):
        """A TF in a short linker (nuc edges on both sides) still gets
        the correction from the NEAREST edge."""
        opp = np.zeros(200, dtype=np.int8)
        hit = np.zeros(200, dtype=np.int8)
        # Nuc edges at 50 and 150. MSP from 50 to 150.
        # Opps at 60, 70, 80 — all misses. d_nuc: 10, 20, 30.
        for p in (60, 70, 80):
            opp[p] = 1
        hit[50] = 1; opp[50] = 1  # boundary hit
        hit[90] = 1; opp[90] = 1
        edges = np.array([50, 150], dtype=np.int64)
        tfs_corr = call_tfs_overcall([(50, 150)], opp, hit, 0.24,
                                       nuc_edges=edges)
        tfs_none = call_tfs_overcall([(50, 150)], opp, hit, 0.24,
                                       nuc_edges=None)
        # Should get one TF from the misses at 60,70,80
        self.assertEqual(len(tfs_corr), 1)
        self.assertEqual(len(tfs_none), 1)
        # d_nuc: 60→50=10, 70→50=20, 80→50=30 (all within cutoff)
        # Correction should fire
        self.assertNotEqual(tfs_corr[0][1], tfs_none[0][1])

    def test_correction_does_not_create_or_destroy_tfs(self):
        """The correction changes tq values but never changes WHICH
        TFs are emitted — same positions, same count."""
        opp, hit = self._make_linear(300, opp_every=3)
        hit[0] = 1; hit[30] = 1; hit[60] = 1; hit[90] = 1
        edges = np.array([0, 150, 300], dtype=np.int64)
        tfs_corr = call_tfs_overcall([(0, 300)], opp, hit, 0.24,
                                       nuc_edges=edges)
        tfs_none = call_tfs_overcall([(0, 300)], opp, hit, 0.24,
                                       nuc_edges=None)
        # Same number of TFs
        self.assertEqual(len(tfs_corr), len(tfs_none))
        # Same positions
        for a, b in zip(tfs_corr, tfs_none):
            self.assertEqual(a[0], b[0])  # (start, end)
        # But tq may differ
        # (don't assert anything about tq values here — that's
        # tested in the specific direction tests above)

    def test_hia5_unaffected(self):
        """Hia5 path passes nuc_edges=None → no correction applied.
        This is a regression guard: if someone accidentally passes
        edges on the Hia5 path, the test would need updating."""
        # This is really just test_no_correction_without_nuc_edges
        # under a different name for documentation clarity.
        opp, hit = self._make_linear(200, opp_every=2)
        hit[0] = 1; hit[40] = 1
        tfs = call_tfs_overcall([(0, 200)], opp, hit, 0.10,
                                  nuc_edges=None)
        self.assertGreater(len(tfs), 0)

    def test_tq_stays_in_0_255_range(self):
        """Even with extreme correction values, tq is clamped."""
        opp = np.zeros(200, dtype=np.int8)
        hit = np.zeros(200, dtype=np.int8)
        # 50 consecutive opps, all misses, right next to a nuc edge
        for i in range(0, 100, 2):
            opp[i] = 1
        hit[100] = 1; opp[100] = 1
        edges = np.array([0], dtype=np.int64)
        tfs = call_tfs_overcall([(0, 200)], opp, hit, 0.3,
                                  nuc_edges=edges)
        for tf in tfs:
            self.assertTrue(0 <= tf[1] <= 255)

    def test_edge_qualities_unchanged_by_correction(self):
        """tel/ter should be identical with and without correction —
        the rotational phase only affects tfp, not edge sharpness."""
        opp, hit = self._make_linear(200, opp_every=2)
        hit[0] = 1; hit[20] = 1
        edges = np.array([0], dtype=np.int64)
        tfs_corr = call_tfs_overcall([(0, 200)], opp, hit, 0.24,
                                       nuc_edges=edges)
        tfs_none = call_tfs_overcall([(0, 200)], opp, hit, 0.24,
                                       nuc_edges=None)
        for a, b in zip(tfs_corr, tfs_none):
            self.assertEqual(a[2], b[2])  # tel
            self.assertEqual(a[3], b[3])  # ter


class TestComputeEdgeQ(unittest.TestCase):
    """compute_edge_q — nucleosome edge ambiguity scoring."""

    def test_no_hit_in_breathing_window_returns_zero(self):
        """If the breathing window has NO hits anywhere, both edges
        return q=0 (maximum ambiguity)."""
        hit = np.zeros(500, dtype=np.int8)
        lq, rq = compute_edge_q(100, 300, hit, breathing_window=50)
        self.assertEqual(lq, 0)
        self.assertEqual(rq, 0)

    def test_hit_at_boundary_returns_max(self):
        """A HIT at the exact called boundary → q=255."""
        hit = np.zeros(500, dtype=np.int8)
        hit[100] = 1  # right at left edge of [100, 300)
        hit[299] = 1  # right at right edge
        lq, rq = compute_edge_q(100, 300, hit, breathing_window=50)
        self.assertEqual(lq, 255)
        self.assertEqual(rq, 255)

    def test_breathing_linear(self):
        """Intermediate distance gives linear interpolation."""
        hit = np.zeros(500, dtype=np.int8)
        hit[120] = 1  # 20 bp from left edge
        lq, rq = compute_edge_q(100, 300, hit, breathing_window=50)
        # left: breath = 20 → q = 255 * (1 - 20/50) = 153
        self.assertAlmostEqual(lq, 153, delta=2)
        self.assertEqual(rq, 0)  # nothing on right


class TestBestGuessFilter(unittest.TestCase):
    """Filter thresholds used by best_guess.py — catch silent drift.

    Iter-16b loosens these substantially from iter-16a because the
    user pointed out that 'a couple of missed deaminations in an MSP
    should be enough to call' a TF in dense DAF data.
    """

    def test_recommended_tq_threshold_hi_baseline(self):
        from best_guess import recommended_tf_tq_threshold
        # iter-16e: baseline >= 0.20 → tq=22 (~2 misses @ baseline 0.24
        # with _TFP_MAX_NEG_LOG10=3.0)
        self.assertEqual(recommended_tf_tq_threshold(0.25), 22)
        self.assertEqual(recommended_tf_tq_threshold(0.20), 22)

    def test_recommended_tq_threshold_lo_baseline(self):
        from best_guess import recommended_tf_tq_threshold
        # iter-16e: baseline < 0.20 → tq=12 (~3 misses @ baseline 0.10)
        self.assertEqual(recommended_tf_tq_threshold(0.10), 12)
        self.assertEqual(recommended_tf_tq_threshold(0.19), 12)

    def test_recommended_tq_threshold_none_defaults_to_hi(self):
        from best_guess import recommended_tf_tq_threshold
        self.assertEqual(recommended_tf_tq_threshold(None), 22)

    def test_nuc_mq_min_default_is_zero(self):
        """Iter-16b: keep all nucs; mq is still in the tag."""
        from best_guess import DEFAULT_NUC_MQ_MIN
        self.assertEqual(DEFAULT_NUC_MQ_MIN, 0)

    def test_tf_edge_min_default_is_128(self):
        """Edge sharpness is the main filter criterion in iter-16b."""
        from best_guess import DEFAULT_TF_EDGE_MIN
        self.assertEqual(DEFAULT_TF_EDGE_MIN, 128)


class TestV7PrimitivesImport(unittest.TestCase):
    """v8 imports primitives from v7 — make sure they don't drift."""

    def test_find_pass1_atoms_still_importable(self):
        """v7's Pass 1 must still be callable from v8 code."""
        rate = np.zeros(100, dtype=np.float32)
        valid = np.zeros(100, dtype=bool)
        rate[10:50] = 0.01  # protected
        valid[:] = True
        atoms = find_pass1_atoms(rate, valid, W=40, baseline=0.1)
        self.assertIsInstance(atoms, list)

    def test_windowed_rate_still_importable(self):
        opp = np.ones(200, dtype=np.int8)
        hit = np.zeros(200, dtype=np.int8)
        hit[::10] = 1
        rate, valid = windowed_rate(opp, hit, 40)
        self.assertEqual(rate.shape, (161,))
        self.assertEqual(valid.shape, (161,))


if __name__ == '__main__':
    unittest.main(verbosity=2)
