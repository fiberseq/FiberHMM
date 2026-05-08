"""Tests for shared inference tagging helpers."""

from __future__ import annotations

import numpy as np

from fiberhmm.inference.tagging import scores_to_u8


def test_scores_to_u8_clips_and_returns_python_ints():
    values = scores_to_u8(np.asarray([-1.0, 0.0, 0.5, 1.0, 2.0]))

    assert values == [0, 0, 127, 255, 255]
    assert all(type(value) is int for value in values)
