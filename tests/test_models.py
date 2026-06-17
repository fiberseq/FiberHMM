"""Tests for bundled model path resolution."""

import pytest

import fiberhmm.models as models


def test_get_model_path_defaults_hia5_seq_with_warning():
    with pytest.warns(UserWarning, match="defaulting to 'pacbio'"):
        path = models.get_model_path("hia5", tool="apply")

    assert path.endswith("hia5_pacbio.json")


def test_get_model_path_uses_hia5_seq_and_ignores_seq_for_dddb():
    assert models.get_model_path("hia5", tool="recall", seq="nanopore").endswith(
        "hia5_nanopore.json"
    )
    assert models.get_model_path("dddb", tool="apply", seq="pacbio").endswith(
        "dddb_nanopore.json"
    )


def test_get_model_path_reports_invalid_combo_and_tool():
    with pytest.raises(KeyError, match="Valid enzyme/seq combos"):
        models.get_model_path("hia5", tool="apply", seq="unknown")

    with pytest.raises(KeyError, match="use 'apply' or 'recall'"):
        models.get_model_path("dddb", tool="unknown")


def test_get_model_path_reports_missing_bundled_file(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "_MODELS_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="installation may be incomplete"):
        models.get_model_path("dddb", tool="apply")
