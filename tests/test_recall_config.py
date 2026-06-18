"""Tests for shared recall CLI configuration helpers."""

from types import SimpleNamespace

from fiberhmm.cli.recall_config import (
    ENZYME_PRESETS,
    _arg_or_preset_value,
    _legacy_tags_enabled,
    _recall_preset_for_args,
    _recall_presets_or_default,
    _RecallDefaults,
    _resolve_recall_default_values,
    resolve_recall_defaults,
    should_write_legacy_tags,
)


def test_arg_or_preset_value_uses_override_preset_then_default():
    preset = {"min_llr": 2.5}

    assert _arg_or_preset_value(8.0, preset, "min_llr", 5.0) == 8.0
    assert _arg_or_preset_value(None, preset, "min_llr", 5.0) == 2.5
    assert _arg_or_preset_value(None, preset, "emission_uplift", 1.0) == 1.0


def test_recall_presets_or_default_preserves_explicit_mapping():
    presets = {"enzyme": {"min_llr": 2.5}}

    assert _recall_presets_or_default(presets) is presets
    assert _recall_presets_or_default(None) is ENZYME_PRESETS


def test_recall_config_resolves_preset_defaults_and_overrides():
    presets = {"enzyme": {"min_llr": 2.5, "emission_uplift": 1.2}}

    assert _recall_preset_for_args(
        SimpleNamespace(enzyme="enzyme"), presets,
    ) == {"min_llr": 2.5, "emission_uplift": 1.2}
    assert _recall_preset_for_args(SimpleNamespace(enzyme=None), presets) == {}

    args = SimpleNamespace(enzyme="enzyme", min_llr=None, emission_uplift=None)
    assert _resolve_recall_default_values(args, presets) == _RecallDefaults(
        min_llr=2.5,
        emission_uplift=1.2,
    )
    assert resolve_recall_defaults(args, presets) == (2.5, 1.2)

    args = SimpleNamespace(enzyme="enzyme", min_llr=8.0, emission_uplift=1.5)
    assert resolve_recall_defaults(args, presets) == (8.0, 1.5)

    args = SimpleNamespace(enzyme=None, min_llr=None, emission_uplift=None)
    assert resolve_recall_defaults(args, presets) == (5.0, 1.0)


def test_recall_config_resolves_legacy_tag_output():
    assert _legacy_tags_enabled(False, False) is True
    assert _legacy_tags_enabled(False, True) is False
    assert _legacy_tags_enabled(True, True) is True

    args = SimpleNamespace(downstream_compat=False, no_legacy_tags=False)
    assert should_write_legacy_tags(args) is True

    args = SimpleNamespace(downstream_compat=False, no_legacy_tags=True)
    assert should_write_legacy_tags(args) is False

    args = SimpleNamespace(downstream_compat=True, no_legacy_tags=True)
    assert should_write_legacy_tags(args) is True
