"""Tests for DAF-seq strand-swap chimera detection."""
from __future__ import annotations

from fiberhmm.daf.encoder import is_daf_chimera
from fiberhmm.inference.chimeras import DEFAULT_MIN_SEG, DEFAULT_PURITY


def test_clean_ct_read_not_chimera():
    ct = list(range(10, 110, 10))      # 10 CT events
    ga = []                             # no opposite-strand signal
    assert not is_daf_chimera(ct, ga)


def test_clean_read_with_scattered_errors_not_chimera():
    # 8 CT events with a few scattered GA "errors" interspersed -> no pure
    # opposite-strand segment, so not a swap.
    ct = [10, 20, 30, 40, 50, 60, 70, 80]
    ga = [15, 35, 55, 75, 95]
    assert not is_daf_chimera(ct, ga)


def test_ct_then_ga_swap_is_chimera():
    ct = list(range(10, 10 + 10 * DEFAULT_MIN_SEG, 10))      # pure CT segment (early)
    ga = list(range(2000, 2000 + 10 * DEFAULT_MIN_SEG, 10))  # pure GA segment (late)
    assert is_daf_chimera(ct, ga)
    assert not is_daf_chimera(ct[1:], ga)     # one event short on the left


def test_ga_then_ct_swap_is_chimera():
    ga = list(range(10, 10 + 10 * DEFAULT_MIN_SEG, 10))
    ct = list(range(2000, 2000 + 10 * DEFAULT_MIN_SEG, 10))
    assert is_daf_chimera(ct, ga)


def test_default_purity():
    """One stray event per segment is tolerated only down to the purity."""
    n = DEFAULT_MIN_SEG
    ct = list(range(10, 10 + 10 * n, 10))
    ga = list(range(2000, 2000 + 10 * n, 10))
    stray = int(n * (1 - DEFAULT_PURITY) / DEFAULT_PURITY)   # largest allowed impurity
    assert is_daf_chimera(ct + [2001 + 10 * i for i in range(stray)], ga)
    assert not is_daf_chimera(ct + [2001 + 10 * i for i in range(stray + 1)], ga)


def test_legacy_setting_still_available():
    ct = [10, 20, 30, 40, 50]
    ga = [200, 210, 220, 230, 240]
    assert not is_daf_chimera(ct, ga)
    assert is_daf_chimera(ct, ga, min_seg_events=5, purity=0.8)


def test_too_few_events_not_chimera():
    # below min_seg_events on the minority side -> cannot confidently call
    assert not is_daf_chimera([10, 20, 30], [200, 210, 220])


def test_min_seg_threshold_tunable():
    ct = [10, 20, 30, 40]
    ga = [200, 210, 220, 230]
    assert not is_daf_chimera(ct, ga, min_seg_events=5)   # 4 < 5
    assert is_daf_chimera(ct, ga, min_seg_events=4)       # exactly 4
