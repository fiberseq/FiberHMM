"""Shared read skip/filter policy for inference pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AbstractSet, Optional


@dataclass(frozen=True)
class ReadFilterConfig:
    """Filtering options shared by streaming inference paths."""

    min_mapq: int = 0
    min_read_length: int = 0
    primary_only: bool = False
    process_unmapped: bool = False
    train_rids: AbstractSet[str] = field(default_factory=frozenset)


def _is_secondary_or_supplementary(read) -> bool:
    return read.is_secondary or read.is_supplementary


def is_primary_alignment(read) -> bool:
    """Return True for primary alignments, including unmapped primary reads."""
    return not _is_secondary_or_supplementary(read)


def is_primary_mapped_alignment(read) -> bool:
    """Return True for mapped primary alignments."""
    return not read.is_unmapped and is_primary_alignment(read)


def _filter_read_length(read):
    if read.is_unmapped:
        return read.query_length or 0
    return read.query_alignment_length


def _unmapped_skip_reason(read, config: ReadFilterConfig) -> Optional[str]:
    if read.is_unmapped:
        if not config.process_unmapped or read.query_sequence is None:
            return "unmapped"
    return None


def _alignment_skip_reason(read, config: ReadFilterConfig) -> Optional[str]:
    if config.primary_only and _is_secondary_or_supplementary(read):
        return "secondary_supplementary"

    if not read.is_unmapped and read.mapping_quality < config.min_mapq:
        return "low_mapq"

    return None


def streaming_skip_reason(read, config: ReadFilterConfig) -> Optional[str]:
    """Return the skip reason for a streaming read, or None if processable."""
    reason = _unmapped_skip_reason(read, config)
    if reason is not None:
        return reason

    reason = _alignment_skip_reason(read, config)
    if reason is not None:
        return reason

    read_len = _filter_read_length(read)
    if read_len is None or read_len < config.min_read_length:
        return "too_short"

    if config.train_rids and read.query_name in config.train_rids:
        return "training_excluded"

    return None
