"""Bundled model registry mode inference."""

import json
from pathlib import Path

import pytest

from fiberhmm.models import (
    get_metadata_mode_aliases,
    get_model_path,
    get_observation_mode,
)


@pytest.mark.parametrize(
    ("enzyme", "seq", "expected"),
    [
        ("hia5", "pacbio", "pacbio-fiber"),
        ("hia5", "nanopore", "nanopore-fiber"),
        ("ecogii", "pacbio", "pacbio-fiber"),
        ("ecogii", "nanopore", "nanopore-fiber"),
        ("dddb", None, "daf"),
        ("dddb", "nanopore", "daf"),
        ("ddda", None, "daf"),
        ("ddda", "pacbio", "daf"),
    ],
)
def test_bundled_observation_modes(enzyme, seq, expected):
    assert get_observation_mode(enzyme, seq) == expected


def test_ecogii_platforms_share_emissions_but_not_observation_frame():
    pacbio = get_model_path("ecogii", tool="apply", seq="pacbio")
    nanopore = get_model_path("ecogii", tool="apply", seq="nanopore")
    assert Path(pacbio) == Path(nanopore)
    assert get_observation_mode("ecogii", "pacbio") == "pacbio-fiber"
    assert get_observation_mode("ecogii", "nanopore") == "nanopore-fiber"
    assert get_metadata_mode_aliases("ecogii", "pacbio") == ()
    assert get_metadata_mode_aliases("ecogii", "nanopore") == ("pacbio-fiber",)

    embedded_mode = json.loads(Path(nanopore).read_text())["mode"]
    assert embedded_mode in get_metadata_mode_aliases("ecogii", "nanopore")
