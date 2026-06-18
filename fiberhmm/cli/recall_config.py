"""Shared CLI decisions for TF/nucleosome recall commands."""

from dataclasses import dataclass

from fiberhmm.inference.tf_recaller import ENZYME_PRESETS


@dataclass(frozen=True)
class _RecallDefaults:
    min_llr: float
    emission_uplift: float

    def as_tuple(self) -> tuple[float, float]:
        return self.min_llr, self.emission_uplift


def _recall_preset_for_args(args, presets: dict) -> dict:
    return presets.get(args.enzyme, {}) if args.enzyme else {}


def _arg_or_preset_value(arg_value, preset: dict, key: str, default):
    return arg_value if arg_value is not None else preset.get(key, default)


def _recall_presets_or_default(presets):
    return ENZYME_PRESETS if presets is None else presets


def _resolve_recall_default_values(args, presets=None) -> _RecallDefaults:
    """Resolve min-LLR and emission-uplift from CLI overrides or enzyme presets."""
    presets = _recall_presets_or_default(presets)
    preset = _recall_preset_for_args(args, presets)
    min_llr = _arg_or_preset_value(args.min_llr, preset, 'min_llr', 5.0)
    uplift = _arg_or_preset_value(
        args.emission_uplift, preset, 'emission_uplift', 1.0,
    )
    return _RecallDefaults(min_llr=min_llr, emission_uplift=uplift)


def resolve_recall_defaults(args, presets=None):
    """Resolve min-LLR and emission-uplift from CLI overrides or enzyme presets."""
    return _resolve_recall_default_values(args, presets).as_tuple()


def _legacy_tags_enabled(downstream_compat: bool, no_legacy_tags: bool) -> bool:
    return True if downstream_compat else (not no_legacy_tags)


def should_write_legacy_tags(args):
    """Return whether legacy ns/nl/as/al tags should be emitted."""
    return _legacy_tags_enabled(args.downstream_compat, args.no_legacy_tags)
