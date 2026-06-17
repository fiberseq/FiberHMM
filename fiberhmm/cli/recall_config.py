"""Shared CLI decisions for TF/nucleosome recall commands."""

from fiberhmm.inference.tf_recaller import ENZYME_PRESETS


def _recall_preset_for_args(args, presets: dict) -> dict:
    return presets.get(args.enzyme, {}) if args.enzyme else {}


def _arg_or_preset_value(arg_value, preset: dict, key: str, default):
    return arg_value if arg_value is not None else preset.get(key, default)


def resolve_recall_defaults(args, presets=None):
    """Resolve min-LLR and emission-uplift from CLI overrides or enzyme presets."""
    presets = ENZYME_PRESETS if presets is None else presets
    preset = _recall_preset_for_args(args, presets)
    min_llr = _arg_or_preset_value(args.min_llr, preset, 'min_llr', 5.0)
    uplift = _arg_or_preset_value(
        args.emission_uplift, preset, 'emission_uplift', 1.0,
    )
    return min_llr, uplift


def should_write_legacy_tags(args):
    """Return whether legacy ns/nl/as/al tags should be emitted."""
    return True if args.downstream_compat else (not args.no_legacy_tags)
