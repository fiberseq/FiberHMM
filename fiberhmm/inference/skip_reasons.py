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


def _skip_reason_keys(extra_reasons) -> tuple:
    return (*BASE_SKIP_REASON_KEYS, *extra_reasons)


def new_skip_reasons(*extra_reasons: str) -> dict:
    return {reason: 0 for reason in _skip_reason_keys(extra_reasons)}


def record_skip_reason(skip_reasons: dict, reason: str) -> None:
    """Increment an existing skip-reason counter."""
    skip_reasons[reason] += 1


def iter_nonzero_skip_reasons(skip_reasons: dict):
    """Yield skip-reason counts sorted by descending count."""
    for reason, count in sorted(skip_reasons.items(), key=lambda item: -item[1]):
        if count > 0:
            yield reason, count
