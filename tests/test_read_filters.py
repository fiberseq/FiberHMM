"""Tests for shared inference read filtering."""

from __future__ import annotations

from dataclasses import dataclass

from fiberhmm.inference.read_filters import (
    ReadFilterConfig,
    is_primary_alignment,
    is_primary_mapped_alignment,
    streaming_skip_reason,
)
from fiberhmm.inference.skip_reasons import (
    BASE_SKIP_REASON_KEYS,
    CHIMERA_SKIP_REASON,
    NO_FOOTPRINTS_SKIP_REASON,
    new_skip_reasons,
)


@dataclass
class _Read:
    query_name: str = "read"
    is_unmapped: bool = False
    query_sequence: str | None = "ACGT"
    is_secondary: bool = False
    is_supplementary: bool = False
    mapping_quality: int = 60
    query_length: int | None = 4
    query_alignment_length: int | None = 4


def test_skip_reason_factory_zeroes_base_and_extra_reasons():
    reasons = new_skip_reasons(NO_FOOTPRINTS_SKIP_REASON, CHIMERA_SKIP_REASON)

    assert tuple(reasons) == (
        *BASE_SKIP_REASON_KEYS,
        NO_FOOTPRINTS_SKIP_REASON,
        CHIMERA_SKIP_REASON,
    )
    assert set(reasons.values()) == {0}

    reasons["low_mapq"] = 3
    assert new_skip_reasons()["low_mapq"] == 0


def test_streaming_filter_allows_processable_mapped_read():
    assert streaming_skip_reason(_Read(), ReadFilterConfig()) is None


def test_primary_alignment_predicates():
    assert is_primary_alignment(_Read())
    assert is_primary_alignment(_Read(is_unmapped=True))
    assert not is_primary_alignment(_Read(is_secondary=True))
    assert not is_primary_alignment(_Read(is_supplementary=True))

    assert is_primary_mapped_alignment(_Read())
    assert not is_primary_mapped_alignment(_Read(is_unmapped=True))
    assert not is_primary_mapped_alignment(_Read(is_secondary=True))
    assert not is_primary_mapped_alignment(_Read(is_supplementary=True))


def test_streaming_filter_skips_unmapped_without_process_unmapped():
    read = _Read(is_unmapped=True, query_alignment_length=None)

    assert streaming_skip_reason(read, ReadFilterConfig()) == "unmapped"


def test_streaming_filter_allows_unmapped_with_sequence_when_enabled():
    read = _Read(is_unmapped=True, query_alignment_length=None)
    config = ReadFilterConfig(process_unmapped=True, min_read_length=4)

    assert streaming_skip_reason(read, config) is None


def test_streaming_filter_skips_unmapped_without_sequence_even_when_enabled():
    read = _Read(is_unmapped=True, query_sequence=None, query_alignment_length=None)
    config = ReadFilterConfig(process_unmapped=True)

    assert streaming_skip_reason(read, config) == "unmapped"


def test_streaming_filter_priority_matches_existing_order():
    read = _Read(
        is_secondary=True,
        mapping_quality=0,
        query_alignment_length=1,
    )
    config = ReadFilterConfig(primary_only=True, min_mapq=30, min_read_length=100)

    assert streaming_skip_reason(read, config) == "secondary_supplementary"


def test_streaming_filter_low_mapq_and_too_short():
    assert (
        streaming_skip_reason(_Read(mapping_quality=10), ReadFilterConfig(min_mapq=30))
        == "low_mapq"
    )
    assert (
        streaming_skip_reason(
            _Read(query_alignment_length=10),
            ReadFilterConfig(min_read_length=100),
        )
        == "too_short"
    )


def test_streaming_filter_training_exclusion():
    read = _Read(query_name="holdout")
    config = ReadFilterConfig(train_rids={"holdout"})

    assert streaming_skip_reason(read, config) == "training_excluded"
