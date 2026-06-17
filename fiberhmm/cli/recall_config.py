"""Shared CLI decisions for TF/nucleosome recall commands."""

from fiberhmm.inference.tf_recaller import ENZYME_PRESETS


def resolve_recall_defaults(args, presets=None):
    """Resolve min-LLR and emission-uplift from CLI overrides or enzyme presets."""
    presets = ENZYME_PRESETS if presets is None else presets
    preset = presets.get(args.enzyme, {}) if args.enzyme else {}
    min_llr = args.min_llr if args.min_llr is not None else preset.get('min_llr', 5.0)
    uplift = (
        args.emission_uplift if args.emission_uplift is not None
        else preset.get('emission_uplift', 1.0)
    )
    return min_llr, uplift


def should_write_legacy_tags(args):
    """Return whether legacy ns/nl/as/al tags should be emitted."""
    return True if args.downstream_compat else (not args.no_legacy_tags)
