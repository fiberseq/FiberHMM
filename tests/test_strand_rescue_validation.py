"""Tests for held-out mask-and-recover validation primitives."""

import numpy as np
import pytest

from fiberhmm.inference.strand_rescue import IntervalCall, ReadEvidence, SiteTemplate
from fiberhmm.inference.strand_rescue_validation import (
    mask_tf_calls_as_accessible,
    negative_accessible_events,
    positive_mask_events,
    site_prediction,
    stable_molecule_fold,
    thin_interval_opportunities,
)


def _read(
    *,
    name="read",
    positions=(6, 11, 12, 13, 19, 24),
    tfs=(),
    nucs=(),
    msps=(IntervalCall(5, 10), IntervalCall(15, 20)),
    alignment_blocks=None,
):
    positions = np.asarray(positions, dtype=np.int64)
    return ReadEvidence(
        name=name,
        strand="CT",
        ref_start=0,
        ref_end=30,
        positions=positions,
        steps=np.arange(positions.size, dtype=float) + 1.0,
        hits=np.asarray([index % 2 for index in range(positions.size)], dtype=np.int8),
        contexts=np.arange(positions.size, dtype=np.int16) + 100,
        tfs=list(tfs),
        nucs=list(nucs),
        msps=list(msps),
        library_id="library",
        nuc_steps=np.arange(positions.size, dtype=float) + 1000.0,
        alignment_blocks=alignment_blocks,
    )


def _site(start=10, end=15):
    return SiteTemplate(
        site_id="site1",
        start=start,
        end=end,
        center=(start + end) // 2,
        support={"CT": 20, "GA": 20},
        start_mad=1.0,
        end_mad=1.0,
        local_enrichment=3.0,
    )


def test_mask_tf_call_merges_both_flanking_msps_without_touching_other_tf():
    masked = IntervalCall(10, 15, ordinal=0)
    retained = IntervalCall(22, 25, ordinal=1)
    read = _read(tfs=(masked, retained))

    variant = mask_tf_calls_as_accessible(read, [0], variant_name="read|masked")

    assert variant.name == "read|masked"
    assert variant.tfs == [retained]
    assert [(call.start, call.end) for call in variant.msps] == [(5, 20)]
    assert variant.msps[0].ordinal == 0
    assert variant.molecular_tfs is None
    assert read.tfs == [masked, retained]
    assert [(call.start, call.end) for call in read.msps] == [(5, 10), (15, 20)]


def test_mask_tf_calls_rejects_empty_or_invalid_selection():
    read = _read(tfs=(IntervalCall(10, 15),))
    with pytest.raises(ValueError, match="at least one"):
        mask_tf_calls_as_accessible(read, [], variant_name="empty")
    with pytest.raises(IndexError, match="outside"):
        mask_tf_calls_as_accessible(read, [1], variant_name="invalid")


def test_opportunity_thinning_is_deterministic_nested_and_array_aligned():
    read = _read(positions=(1, 2, 11, 12, 13, 14, 21), msps=())
    half, half_stats = thin_interval_opportunities(
        read, [(10, 20)], 0.5, stable_key="fixed"
    )
    quarter, quarter_stats = thin_interval_opportunities(
        read, [(10, 20)], 0.25, stable_key="fixed"
    )
    repeat, _ = thin_interval_opportunities(
        read, [(10, 20)], 0.5, stable_key="fixed"
    )

    assert half_stats["opportunities_original"] == 4
    assert half_stats["opportunities_retained"] == 2
    assert quarter_stats["opportunities_retained"] == 1
    assert set(quarter.positions).issubset(set(half.positions))
    assert np.array_equal(half.positions, repeat.positions)
    assert {1, 2, 21}.issubset(set(quarter.positions))
    assert len(half.positions) == len(half.steps) == len(half.hits) == len(half.contexts)
    assert len(half.positions) == len(half.nuc_steps)
    for position, step, context, nuc_step in zip(
        half.positions, half.steps, half.contexts, half.nuc_steps
    ):
        source_index = int(np.flatnonzero(read.positions == position)[0])
        assert step == read.steps[source_index]
        assert context == read.contexts[source_index]
        assert nuc_step == read.nuc_steps[source_index]
    assert read.positions.tolist() == [1, 2, 11, 12, 13, 14, 21]


@pytest.mark.parametrize("retention", (-0.1, 1.1, float("nan")))
def test_opportunity_thinning_rejects_invalid_retention(retention):
    with pytest.raises(ValueError, match="retention"):
        thin_interval_opportunities(
            _read(), [(10, 20)], retention, stable_key="invalid"
        )


def test_positive_mask_events_require_supported_fully_mapped_family():
    site = _site()
    read = _read(tfs=(IntervalCall(10, 15),))
    events = positive_mask_events(
        read, [site], {"CT": {0}}, center_radius=10
    )
    assert len(events) == 1
    assert events[0]["tf_index"] == 0
    assert events[0]["site_id"] == "site1"
    assert events[0]["opportunities"] == 3
    assert positive_mask_events(
        read, [site], {"CT": set()}, center_radius=10
    ) == []

    gapped = _read(
        tfs=(IntervalCall(10, 15),),
        alignment_blocks=((0, 12), (14, 30)),
    )
    assert positive_mask_events(
        gapped, [site], {"CT": {0}}, center_radius=10
    ) == []


def test_negative_accessible_events_mirror_pre_likelihood_blocking():
    site = _site()
    read = _read(msps=(IntervalCall(5, 20),), tfs=(), nucs=())
    events = negative_accessible_events(
        read,
        [site],
        {"CT": {0}},
        center_radius=10,
        accessible_site_gap=30,
    )
    assert len(events) == 1
    assert events[0]["site_id"] == "site1"
    assert events[0]["msp_length"] == 15
    assert events[0]["opportunities"] == 3

    for blocked in (
        _read(msps=(IntervalCall(5, 20),), tfs=(IntervalCall(10, 15),)),
        _read(msps=(IntervalCall(5, 20),), nucs=(IntervalCall(9, 16),)),
    ):
        assert negative_accessible_events(
            blocked,
            [site],
            {"CT": {0}},
            center_radius=10,
            accessible_site_gap=30,
        ) == []


def test_site_prediction_returns_strongest_exact_component():
    site = _site()
    decisions = [
        {
            "decision_id": "low",
            "proposal_tier": "review",
            "sr_hypothesis_probability": 0.6,
            "proposed_site_intervals": [[10, 15]],
        },
        {
            "decision_id": "high",
            "proposal_tier": "strong",
            "sr_hypothesis_probability": 0.95,
            "proposed_site_intervals": [[10, 15], [20, 25]],
        },
    ]
    prediction = site_prediction(decisions, site)
    assert prediction["q0"] == 0.95
    assert prediction["decision_id"] == "high"
    assert site_prediction(decisions, _site(16, 19))["q0"] == 0.0


def test_stable_fold_keeps_same_molecule_together():
    molecule = ("library", "read", "CT")
    first = stable_molecule_fold("NAPA", molecule, 5, "seed")
    assert first == stable_molecule_fold("NAPA", molecule, 5, "seed")
    assert 0 <= first < 5
    with pytest.raises(ValueError, match="at least two"):
        stable_molecule_fold("NAPA", molecule, 1, "seed")
