"""Bundled model registry mode inference."""

import hashlib
import json
from pathlib import Path

import pytest

from fiberhmm.models import (
    DEVELOPMENT_ENZYMES,
    SUPPORTED_ENZYMES,
    get_metadata_mode_aliases,
    get_model_path,
    get_observation_mode,
)


def test_public_supported_modes_manifest_matches_cli_surface():
    model_root = Path(__file__).resolve().parents[1] / "fiberhmm" / "models"
    manifest = json.loads((model_root / "SUPPORTED_MODES.json").read_text())
    enzymes = {row["enzyme"] for row in manifest["public_enzyme_presets"]}
    assert enzymes == set(SUPPORTED_ENZYMES) == {"ddda", "dddb", "hia5"}
    assert {"ecogii", "sssi"}.issubset(DEVELOPMENT_ENZYMES)
    assert not ({"ecogii", "sssi"} & enzymes)


def test_source_tree_current_model_mirror_matches_authoritative_package_root():
    source_root = Path(__file__).resolve().parents[1]
    package_root = source_root / "fiberhmm" / "models"
    mirror_root = source_root / "models"
    for filename in (
        "ddda_TF.json",
        "ddda_nuc.json",
        "ddda_nuc_profile.json",
        "dddb_nanopore.json",
        "hia5_nanopore.json",
        "hia5_pacbio.json",
    ):
        assert (mirror_root / filename).read_bytes() == (package_root / filename).read_bytes()


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


def test_development_ecogii_registry_entry_is_not_a_public_preset():
    assert "ecogii" not in SUPPORTED_ENZYMES
    pacbio = get_model_path("ecogii", tool="apply", seq="pacbio")
    nanopore = get_model_path("ecogii", tool="apply", seq="nanopore")
    assert Path(pacbio) == Path(nanopore)
    assert get_observation_mode("ecogii", "pacbio") == "pacbio-fiber"
    assert get_observation_mode("ecogii", "nanopore") == "nanopore-fiber"
    assert get_metadata_mode_aliases("ecogii", "pacbio") == ()
    assert get_metadata_mode_aliases("ecogii", "nanopore") == ("pacbio-fiber",)

    embedded_mode = json.loads(Path(nanopore).read_text())["mode"]
    assert embedded_mode in get_metadata_mode_aliases("ecogii", "nanopore")


def test_ddda_nuc_refinement_model_is_separately_frozen():
    tf_model = Path(get_model_path("ddda", tool="recall"))
    nuc_model = Path(get_model_path("ddda", tool="nuc_refine"))

    assert tf_model.name == "ddda_TF.json"
    assert nuc_model.name == "ddda_nuc_refine.json"
    # The frozen nucleosome-refinement table is an independent snapshot.
    # TF calibration may change ddda_TF.json without changing this contract.
    assert json.loads(nuc_model.read_text())["mode"] == "daf"
    assert hashlib.sha256(nuc_model.read_bytes()).hexdigest() == (
        "5e1f29ba6abbf7c1909f2566efbd4c3f82cf4ecf4aa5de2c96f0a51b48e85bfb"
    )


def test_ddda_tf_model_is_physical_duplex_calibrated_release_artifact():
    tf_model = Path(get_model_path("ddda", tool="recall"))
    expected = (
        "2b23b0905189d0638b682845ea5adae32144cf10e7a5e90b8f0bac199dbbbffe"
    )
    assert hashlib.sha256(tf_model.read_bytes()).hexdigest() == expected
    source_tree_mirror = tf_model.parents[2] / "models" / "ddda_TF.json"
    assert hashlib.sha256(source_tree_mirror.read_bytes()).hexdigest() == expected


def test_ddda_first_pass_hmm_is_unchanged_by_tf_calibration():
    apply_model = Path(get_model_path("ddda", tool="apply"))
    assert hashlib.sha256(apply_model.read_bytes()).hexdigest() == (
        "c9da3116b4148ba67a85fa5cf86edd31ea7e434fcea52794e65c953be0b86c4b"
    )
