"""Shared skip-reason bookkeeping for inference pipelines."""

from __future__ import annotations

BASE_SKIP_REASON_KEYS = (
    'unmapped',
    'secondary_supplementary',
    'low_mapq',
    'too_short',
    'training_excluded',
    'no_modifications',
    'extraction_failed',
)

NO_FOOTPRINTS_SKIP_REASON = 'no_footprints'
CHIMERA_SKIP_REASON = 'chimera'


def new_skip_reasons(*extra_reasons: str) -> dict:
    return {reason: 0 for reason in (*BASE_SKIP_REASON_KEYS, *extra_reasons)}
