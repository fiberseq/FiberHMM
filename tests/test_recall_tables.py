"""Tests for shared TF-recall LLR table helpers."""

import pytest

from fiberhmm.inference import recall_tables


def test_build_recall_llr_tables_skips_identity_uplift(monkeypatch):
    calls = []

    monkeypatch.setattr(
        recall_tables,
        "build_llr_tables",
        lambda model: ("hit", "miss"),
    )
    monkeypatch.setattr(
        recall_tables,
        "apply_emission_uplift",
        lambda *args: calls.append(args),
    )

    assert recall_tables.build_recall_llr_tables("model", 1.0) == ("hit", "miss")
    assert calls == []


def test_build_recall_llr_tables_applies_nonidentity_uplift(monkeypatch):
    monkeypatch.setattr(
        recall_tables,
        "build_llr_tables",
        lambda model: ("hit", "miss"),
    )
    monkeypatch.setattr(
        recall_tables,
        "apply_emission_uplift",
        lambda hit, miss, model, uplift: (
            f"{hit}:{model}:{uplift}",
            f"{miss}:{model}:{uplift}",
        ),
    )

    assert recall_tables.build_recall_llr_tables("model", 1.5) == (
        "hit:model:1.5",
        "miss:model:1.5",
    )


def test_load_recall_llr_tables_uses_recall_path_or_fallback(monkeypatch):
    loaded_paths = []

    def fake_load(path):
        loaded_paths.append(path)
        return f"model:{path}", None, None

    monkeypatch.setattr(recall_tables, "load_model_with_metadata", fake_load)
    monkeypatch.setattr(
        recall_tables,
        "build_recall_llr_tables",
        lambda model, uplift: (model, uplift),
    )

    assert recall_tables.load_recall_llr_tables(
        "recall.json",
        "apply.json",
        1.2,
    ) == ("model:recall.json", 1.2)
    assert recall_tables.load_recall_llr_tables(
        None,
        "apply.json",
        1.0,
    ) == ("model:apply.json", 1.0)
    assert loaded_paths == ["recall.json", "apply.json"]


def test_load_recall_llr_tables_requires_a_model_path():
    with pytest.raises(ValueError, match="one of recall_model_path"):
        recall_tables.load_recall_llr_tables(None, None)
