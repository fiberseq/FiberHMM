"""Tests for bundled model path resolution."""

import pytest

import fiberhmm.models as models


def test_bundled_model_key_normalizes_enzyme_and_seq():
    with pytest.warns(UserWarning, match="defaulting to 'pacbio'"):
        assert models._bundled_model_key("HIA5", None) == ("hia5", "pacbio")

    assert models._bundled_model_key(" HIA5 ", " NanoPore ") == (
        "hia5",
        "nanopore",
    )
    assert models._bundled_model_key("DddB", "nanopore") == ("dddb", None)


def test_bundled_model_path_uses_models_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "_MODELS_DIR", str(tmp_path))

    assert models._bundled_model_path("model.json") == str(tmp_path / "model.json")


def test_unknown_bundled_model_message_lists_choices_and_override():
    message = models._unknown_bundled_model_message("hia5", "unknown", "apply")

    assert "enzyme='hia5'" in message
    assert "seq='unknown'" in message
    assert "Valid enzyme/seq combos" in message
    assert "Use --model" in message


def test_unknown_tool_message_lists_valid_tools():
    message = models._unknown_tool_message("score")

    assert "Tool 'score' not recognised" in message
    assert "use 'apply' or 'recall'" in message


def test_missing_bundled_model_message_names_path_and_installation():
    message = models._missing_bundled_model_message("/tmp/missing.json")

    assert "Bundled model file missing: /tmp/missing.json" in message
    assert "installation may be incomplete" in message


def test_get_model_path_defaults_hia5_seq_with_warning():
    with pytest.warns(UserWarning, match="defaulting to 'pacbio'"):
        path = models.get_model_path("hia5", tool="apply")

    assert path.endswith("hia5_pacbio.json")


def test_get_model_path_uses_hia5_seq_and_ignores_seq_for_dddb():
    assert models.get_model_path(" hia5 ", tool=" recall ", seq=" nanopore ").endswith(
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
