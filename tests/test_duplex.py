"""Tests for the sequence-identity-free fiberhmm-duplex scorer."""
from __future__ import annotations

import numpy as np
import pysam

from fiberhmm.crossstrand.duplex import (
    DuplexModel,
    DuplexParams,
    ProtectionProfile,
    STATUS_PAIRED,
    STATUS_UNRESOLVED,
    assign_duplex_pairs,
    build_pattern_feature,
    component_residual_profiles,
    infer_call_layer,
    load_duplex_model,
)
from fiberhmm.crossstrand.pairing import (
    FLAVOR_CT,
    FLAVOR_GA,
    ReadFeat,
    _gaussian_kernel,
)


def _feature(index, name, flavor, dyads, params, start=900, end=4900):
    grid = params.grid_bp
    kernel = _gaussian_kernel(params.dyad_sigma_bp, grid)
    radius = len(kernel) // 2
    grid0 = start // grid
    signal = np.zeros(end // grid - grid0 + 1, dtype=np.float32)
    for center in dyads:
        center_bin = center // grid - grid0
        lo = max(0, center_bin - radius)
        hi = min(len(signal), center_bin + radius + 1)
        kernel_lo = lo - (center_bin - radius)
        signal[lo:hi] += kernel[kernel_lo:kernel_lo + hi - lo]
    return ReadFeat(
        index=index,
        name=name,
        flavor=flavor,
        ref_start=start,
        ref_end=end,
        dyads=np.asarray(dyads, dtype=np.int64),
        grid0=grid0,
        signal=signal,
        sequence_pos=None,
        sequence_base=None,
    )


def _profile(index, name, flavor, values):
    values = np.asarray(values, dtype=np.float32)
    return ProtectionProfile(
        name=name,
        flavor=flavor,
        start_bin=45,
        end_bin=45 + len(values) - 1,
        rate=values,
        weight=np.full(len(values), 3.0, dtype=np.float32),
        opportunities=3 * len(values),
        hits=int(values.sum() * 3),
    )


def _baseline_model():
    return DuplexModel(
        model_id="test-baseline",
        feature_names=("baseline_score",),
        mean=np.asarray([0.0]),
        scale=np.asarray([1.0]),
        coefficients=np.asarray([10.0]),
        intercept=0.0,
        metadata={},
    )


def test_bundled_model_declares_no_sequence_identity_features():
    model = load_duplex_model()
    assert model.model_id == "ddda-duplex-v1"
    assert model.metadata["call_layer"] == "FiberHMM 2.16.3 recalled input MA nucleosomes"
    assert not any("sequence" in name.lower() or "haplotype" in name.lower()
                   or "tf" in name.lower() for name in model.feature_names)


def test_rotational_header_selects_matching_frozen_model():
    header = pysam.AlignmentHeader.from_dict({
        "SQ": [{"SN": "chr1", "LN": 1000}],
        "CO": ["FIBERHMM-CHEMISTRY:v1:assay=daf;nuc_model=ddda_phase_posterior_v1"],
    })
    assert infer_call_layer(header) == "rotational-recall"
    assert load_duplex_model(call_layer=infer_call_layer(header)).model_id == \
        "ddda-duplex-rotational-v1"


def test_model_decision_function_uses_frozen_standardization():
    model = load_duplex_model()
    at_mean = dict(zip(model.feature_names, model.mean))
    assert np.isclose(model.decision_function(at_mean), model.intercept)
    shifted = dict(at_mean)
    shifted[model.feature_names[0]] += model.scale[0]
    assert np.isclose(
        model.decision_function(shifted),
        model.intercept + model.coefficients[0],
    )


def test_pattern_feature_does_not_retain_sequence_signature():
    header = pysam.AlignmentHeader.from_dict({
        "SQ": [{"SN": "chr1", "LN": 5000}],
    })
    read = pysam.AlignedSegment(header)
    read.query_name = "ct"
    read.reference_id = 0
    read.reference_start = 100
    read.cigartuples = [(0, 500)]
    read.query_sequence = "Y" + "A" * 499
    read.set_tag("MA", "500;nuc.:1-147,181-147,361-120")
    feature = build_pattern_feature(
        read, 0, DuplexParams(min_overlap_bp=100, min_nucs=1),
    )
    assert feature is not None
    assert feature.flavor == FLAVOR_CT
    assert feature.sequence_pos is None
    assert feature.sequence_base is None


def test_component_residual_is_opportunity_weighted():
    a = _profile(0, "a", FLAVOR_CT, [0.0, 1.0, 0.0])
    b = ProtectionProfile(
        name="b", flavor=FLAVOR_GA, start_bin=45, end_bin=47,
        rate=np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
        weight=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        opportunities=3, hits=2,
    )
    result = component_residual_profiles({0: a, 1: b})
    expected_mean = (a.rate * 3.0 + b.rate) / 4.0
    assert np.allclose(result[0].rate, a.rate - expected_mean)
    assert np.allclose(result[1].rate, b.rate - expected_mean)


def test_reciprocal_model_recovers_two_distinct_lattices():
    params = DuplexParams(min_overlap_bp=1000, min_nucs=3, min_margin=1.0)
    dyads_a = [1000 + 190 * i for i in range(20)]
    dyads_b = [1095 + 205 * i for i in range(19)]
    features = [
        _feature(0, "ctA", FLAVOR_CT, dyads_a, params),
        _feature(1, "ctB", FLAVOR_CT, dyads_b, params),
        _feature(2, "gaA", FLAVOR_GA, dyads_a, params),
        _feature(3, "gaB", FLAVOR_GA, dyads_b, params),
    ]
    x = np.linspace(0, 8 * np.pi, 220)
    profile_a = 0.35 + 0.15 * np.sin(x)
    profile_b = 0.35 + 0.15 * np.cos(0.7 * x)
    profiles = {
        0: _profile(0, "ctA", FLAVOR_CT, profile_a),
        1: _profile(1, "ctB", FLAVOR_CT, profile_b),
        2: _profile(2, "gaA", FLAVOR_GA, profile_a + 0.005 * np.cos(x)),
        3: _profile(3, "gaB", FLAVOR_GA, profile_b + 0.005 * np.sin(x)),
    }
    result = assign_duplex_pairs(features, profiles, _baseline_model(), params)
    assert result.partner == {0: 2, 2: 0, 1: 3, 3: 1}
    assert all(result.status[index] == STATUS_PAIRED for index in range(4))


def test_equal_candidates_abstain_at_margin_gate():
    params = DuplexParams(min_overlap_bp=1000, min_nucs=3, min_margin=1.0)
    dyads = [1000 + 190 * i for i in range(20)]
    features = [
        _feature(0, "ct", FLAVOR_CT, dyads, params),
        _feature(1, "ga1", FLAVOR_GA, dyads, params),
        _feature(2, "ga2", FLAVOR_GA, dyads, params),
    ]
    x = np.linspace(0, 6 * np.pi, 220)
    values = 0.3 + 0.1 * np.sin(x)
    profiles = {
        0: _profile(0, "ct", FLAVOR_CT, values),
        1: _profile(1, "ga1", FLAVOR_GA, values + 0.01 * np.cos(x)),
        2: _profile(2, "ga2", FLAVOR_GA, values - 0.01 * np.cos(x)),
    }
    result = assign_duplex_pairs(features, profiles, _baseline_model(), params)
    assert result.partner == {}
    assert result.status[0] == STATUS_UNRESOLVED
