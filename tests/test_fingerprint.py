"""
Tests for the footprint fingerprinting algorithm.
"""
import numpy as np
import pytest

from fiberhmm.inference.fingerprint import (
    fingerprint_footprints,
    _find_bumps,
    _evaluate_fragments,
)


def _make_posteriors(length, bumps=None, background=0.001):
    """
    Create a (T, 2) posterior matrix with optional accessible bumps.

    bumps: list of (start, end, peak_value) tuples
    """
    p = np.full((length, 2), 0.0)
    p[:, 0] = 1.0 - background  # P(footprint) = high
    p[:, 1] = background         # P(accessible) = low

    if bumps:
        for start, end, peak in bumps:
            mid = (start + end) // 2
            width = end - start
            # Gaussian-ish bump centered at mid
            for i in range(start, min(end, length)):
                dist = abs(i - mid) / max(width / 2, 1)
                p[i, 1] = peak * max(0, 1.0 - dist)
                p[i, 0] = 1.0 - p[i, 1]

    return p


class TestFindBumps:
    """Test the bump discovery phase."""

    def test_no_bumps_flat(self):
        """Flat low P(accessible) should find no bumps."""
        p = np.full(300, 0.002)
        bumps = _find_bumps(p, peak_threshold=0.10)
        assert len(bumps) == 0

    def test_single_clear_bump(self):
        """Clear peak should be found."""
        p = np.full(300, 0.001)
        p[140:160] = 0.3  # 20bp bump with peak 0.3
        bumps = _find_bumps(p, peak_threshold=0.10, local_auc_threshold=1.5)
        assert len(bumps) == 1
        l, r = bumps[0]
        assert 135 <= l <= 145
        assert 155 <= r <= 165

    def test_broad_mesa_no_trigger(self):
        """Flat P=0.04 across 100bp should NOT trigger (global AUC trap)."""
        p = np.full(400, 0.001)
        p[150:250] = 0.04  # broad mesa, global AUC = 4.0 but peak only 0.04
        bumps = _find_bumps(p, peak_threshold=0.10)
        assert len(bumps) == 0

    def test_two_bumps(self):
        """Two separated bumps should both be found."""
        p = np.full(600, 0.001)
        p[140:160] = 0.4
        p[400:420] = 0.35
        bumps = _find_bumps(p, peak_threshold=0.10, local_auc_threshold=1.5)
        assert len(bumps) == 2

    def test_auc_below_threshold(self):
        """Narrow peak with insufficient AUC should not be found."""
        p = np.full(300, 0.001)
        p[150:153] = 0.15  # only 3bp wide, AUC ~0.45
        bumps = _find_bumps(p, peak_threshold=0.10, local_auc_threshold=1.5)
        assert len(bumps) == 0

    def test_edge_bleed_left(self):
        """Bump at left edge should have correct bounds."""
        p = np.full(300, 0.001)
        p[0:20] = 0.3  # bump at very start
        bumps = _find_bumps(p, peak_threshold=0.10, local_auc_threshold=1.5)
        assert len(bumps) == 1
        l, r = bumps[0]
        assert l == 0

    def test_edge_bleed_right(self):
        """Bump at right edge should have correct bounds."""
        p = np.full(300, 0.001)
        p[280:300] = 0.3  # bump at very end
        bumps = _find_bumps(p, peak_threshold=0.10, local_auc_threshold=1.5)
        assert len(bumps) == 1
        l, r = bumps[0]
        assert r == 300


class TestEvaluateFragments:
    """Test the fragment evaluation phase."""

    def test_valid_asymmetric_split(self):
        """Asymmetric split (80 + 350) should pass."""
        assert _evaluate_fragments(450, [(80, 100)], min_sub_fp_size=50) is True

    def test_splinter_guard(self):
        """Fragment < min_sub_fp_size should abort."""
        assert _evaluate_fragments(300, [(20, 40)], min_sub_fp_size=50) is False

    def test_dinucleosome_guard(self):
        """Two ~147bp blocks should abort (poly-nucleosome)."""
        # 320bp footprint split at 150-170: produces 150bp + 150bp
        assert _evaluate_fragments(320, [(150, 170)],
                                   min_sub_fp_size=50, nuc_min=120, nuc_max=180) is False

    def test_poly_nucleosome_three_blocks(self):
        """Three ~150bp blocks should abort."""
        # 480bp: bumps at 150-165 and 315-330 → 150 + 150 + 150
        assert _evaluate_fragments(480, [(150, 165), (315, 330)],
                                   min_sub_fp_size=50, nuc_min=120, nuc_max=180) is False

    def test_internal_splinter(self):
        """Two bumps 30bp apart → internal splinter → abort."""
        # 400bp: bumps at 200-210 and 230-240 → 200 + 20(splinter) + 160
        assert _evaluate_fragments(400, [(200, 210), (230, 240)],
                                   min_sub_fp_size=50) is False

    def test_mixed_sizes_pass(self):
        """One nuc-sized + one large fragment should pass."""
        # 500bp: bump at 150-170 → 150bp + 330bp (one is nuc, one isn't)
        assert _evaluate_fragments(500, [(150, 170)],
                                   min_sub_fp_size=50, nuc_min=120, nuc_max=180) is True


class TestFingerprintFootprints:
    """Test the full fingerprint pipeline."""

    def test_small_footprint_no_split(self):
        """Footprints < 200bp should never be split."""
        posteriors = _make_posteriors(150, bumps=[(70, 90, 0.5)])
        fp_starts = np.array([0], dtype=np.int32)
        fp_ends = np.array([150], dtype=np.int32)

        new_starts, new_ends, n = fingerprint_footprints(
            fp_starts, fp_ends, posteriors, min_footprint_size=200
        )
        assert n == 0
        assert len(new_starts) == 1

    def test_no_bump_no_split(self):
        """Large footprint with no posterior bump should not split."""
        posteriors = _make_posteriors(400)  # flat background
        fp_starts = np.array([0], dtype=np.int32)
        fp_ends = np.array([400], dtype=np.int32)

        new_starts, new_ends, n = fingerprint_footprints(fp_starts, fp_ends, posteriors)
        assert n == 0
        assert len(new_starts) == 1

    def test_clear_bump_splits(self):
        """Large footprint with clear bump should split."""
        posteriors = _make_posteriors(500, bumps=[(200, 240, 0.5)])
        fp_starts = np.array([0], dtype=np.int32)
        fp_ends = np.array([500], dtype=np.int32)

        new_starts, new_ends, n = fingerprint_footprints(fp_starts, fp_ends, posteriors)
        assert n == 1
        assert len(new_starts) >= 2
        # All sub-footprints should be non-empty
        sizes = new_ends - new_starts
        assert all(s > 0 for s in sizes)

    def test_dinucleosome_no_split(self):
        """~300bp footprint splitting into two ~147bp blocks should abort."""
        posteriors = _make_posteriors(320, bumps=[(145, 175, 0.5)])
        fp_starts = np.array([0], dtype=np.int32)
        fp_ends = np.array([320], dtype=np.int32)

        new_starts, new_ends, n = fingerprint_footprints(
            fp_starts, fp_ends, posteriors, min_footprint_size=200
        )
        assert n == 0

    def test_asymmetric_split(self):
        """Asymmetric split (80bp + 350bp) should succeed."""
        posteriors = _make_posteriors(500, bumps=[(70, 110, 0.5)])
        fp_starts = np.array([0], dtype=np.int32)
        fp_ends = np.array([500], dtype=np.int32)

        new_starts, new_ends, n = fingerprint_footprints(fp_starts, fp_ends, posteriors)
        assert n == 1
        assert len(new_starts) >= 2

    def test_multiple_footprints_mixed(self):
        """Multiple footprints: only large ones with bumps should split."""
        posteriors = _make_posteriors(1000,
                                     bumps=[(350, 390, 0.5)])  # bump in 2nd footprint
        fp_starts = np.array([0, 200, 600], dtype=np.int32)
        fp_ends = np.array([150, 500, 800], dtype=np.int32)

        new_starts, new_ends, n = fingerprint_footprints(fp_starts, fp_ends, posteriors)
        # First footprint (150bp) untouched, second (300bp) split, third (200bp) maybe
        assert n >= 1

    def test_empty_input(self):
        """Empty footprint arrays should return empty."""
        posteriors = _make_posteriors(100)
        fp_starts = np.array([], dtype=np.int32)
        fp_ends = np.array([], dtype=np.int32)

        new_starts, new_ends, n = fingerprint_footprints(fp_starts, fp_ends, posteriors)
        assert n == 0
        assert len(new_starts) == 0

    def test_poly_nucleosome_abort(self):
        """450bp footprint with bumps at 150 and 300 → all ~150bp → no split."""
        posteriors = _make_posteriors(450,
                                     bumps=[(140, 160, 0.5), (290, 310, 0.5)])
        fp_starts = np.array([0], dtype=np.int32)
        fp_ends = np.array([450], dtype=np.int32)

        new_starts, new_ends, n = fingerprint_footprints(fp_starts, fp_ends, posteriors)
        assert n == 0
        assert len(new_starts) == 1

    def test_splinter_edge_abort(self):
        """Valid bump 15bp from edge → splinter guard aborts."""
        posteriors = _make_posteriors(300, bumps=[(5, 25, 0.5)])
        fp_starts = np.array([0], dtype=np.int32)
        fp_ends = np.array([300], dtype=np.int32)

        new_starts, new_ends, n = fingerprint_footprints(
            fp_starts, fp_ends, posteriors, min_sub_fp_size=50
        )
        assert n == 0
