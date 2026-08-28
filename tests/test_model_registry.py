"""Bundled model registry mode inference."""

import pytest

from fiberhmm.models import get_observation_mode


@pytest.mark.parametrize(
    ("enzyme", "seq", "expected"),
    [
        ("hia5", "pacbio", "pacbio-fiber"),
        ("hia5", "nanopore", "nanopore-fiber"),
        ("dddb", None, "daf"),
        ("dddb", "nanopore", "daf"),
        ("ddda", None, "daf"),
        ("ddda", "pacbio", "daf"),
    ],
)
def test_bundled_observation_modes(enzyme, seq, expected):
    assert get_observation_mode(enzyme, seq) == expected
