import numpy as np
import pandas as pd
import pytest

from fiberhmm.cli.utils import (
    AccessibilityCounter,
    _accessibility_priors_for_base,
    _estimate_emission_probs,
    _passes_transfer_base_filters,
    _scale_emission_probabilities,
    _target_bases_for_transfer_mode,
    _write_transfer_probability_tables,
)


def test_target_bases_for_transfer_mode():
    assert _target_bases_for_transfer_mode('pacbio-fiber') == ['A']
    assert _target_bases_for_transfer_mode('nanopore-fiber') == ['A']
    assert _target_bases_for_transfer_mode('daf') == ['C', 'G']


def test_accessibility_counter_records_accessible_and_protected_contexts():
    counter = AccessibilityCounter(max_context=1, center_base='A')
    footprint_mask = np.array([False, False, False, True, False])

    counter.process_read_with_footprints("CACAC", footprint_mask, edge_trim=0)
    counter.process_read_with_footprints("CANAC", footprint_mask, edge_trim=0)

    assert counter.counts["CAC"] == [1, 2]
    assert counter.total_accessible == 1
    assert counter.total_positions == 2


def test_estimate_emission_probs_falls_back_with_too_few_contexts(capsys):
    target_rates = pd.DataFrame([
        {"context": "AAA", "ratio": 0.2, "total": 100},
    ])
    accessibility = pd.DataFrame([
        {"context": "AAA", "p_accessible": 0.5, "total_bp": 100},
    ])

    p_acc, p_inacc, diagnostics = _estimate_emission_probs(
        target_rates,
        accessibility,
        min_observations=100,
    )

    assert (p_acc, p_inacc) == (0.5, 0.1)
    assert diagnostics["n_contexts"] == 1
    assert "Only 1 contexts" in capsys.readouterr().out


def test_estimate_emission_probs_fits_linear_accessibility_model():
    contexts = [f"ctx{i}" for i in range(12)]
    x = np.linspace(0.0, 1.0, len(contexts))
    y = 0.1 + 0.4 * x
    target_rates = pd.DataFrame({
        "context": contexts,
        "ratio": y,
        "total": np.full(len(contexts), 200),
    })
    accessibility = pd.DataFrame({
        "context": contexts,
        "p_accessible": x,
        "total_bp": np.full(len(contexts), 200),
    })

    p_acc, p_inacc, diagnostics = _estimate_emission_probs(
        target_rates,
        accessibility,
        min_observations=100,
    )

    assert p_acc == pytest.approx(0.5)
    assert p_inacc == pytest.approx(0.1)
    assert diagnostics["n_contexts"] == 12
    assert diagnostics["r_squared"] == pytest.approx(1.0)


def test_passes_transfer_base_filters_checks_primary_mapq_and_sequence():
    class Read:
        is_unmapped = False
        is_secondary = False
        is_supplementary = False
        mapping_quality = 20
        query_sequence = "ACGT"

    assert _passes_transfer_base_filters(Read(), min_mapq=20)

    read = Read()
    read.mapping_quality = 19
    assert not _passes_transfer_base_filters(read, min_mapq=20)

    read = Read()
    read.query_sequence = None
    assert not _passes_transfer_base_filters(read, min_mapq=20)

    read = Read()
    read.is_secondary = True
    assert not _passes_transfer_base_filters(read, min_mapq=20)


def test_scale_emission_probabilities_scales_one_state_and_clips():
    emission = np.array([
        [0.2, 0.4],
        [0.6, 0.8],
    ])

    adjusted, before, after = _scale_emission_probabilities(emission, 1, 2.0)

    np.testing.assert_allclose(emission, [[0.2, 0.4], [0.6, 0.8]])
    np.testing.assert_allclose(adjusted, [[0.2, 0.4], [1.0, 1.0]])
    np.testing.assert_allclose(before, (0.6, 0.8, 0.7))
    np.testing.assert_allclose(after, (1.0, 1.0, 1.0))


def test_scale_emission_probabilities_scales_all_states():
    emission = np.array([
        [0.2, 0.4],
        [0.6, 0.8],
    ])

    adjusted, before, after = _scale_emission_probabilities(emission, None, 0.5)

    np.testing.assert_allclose(adjusted, [[0.1, 0.2], [0.3, 0.4]])
    np.testing.assert_allclose(before, (0.2, 0.8, 0.5))
    np.testing.assert_allclose(after, (0.1, 0.4, 0.25))


def test_accessibility_priors_for_base_filters_tsv_and_counter():
    priors = pd.DataFrame({
        "context": ["AAA", "ACA", "CG"],
        "p_accessible": [0.1, 0.2, 0.3],
    })

    selected = _accessibility_priors_for_base(
        "C", 1, priors, accessibility_counters=None,
    )
    assert selected["context"].tolist() == ["ACA"]

    class FakeCounter:
        def get_accessibility_priors(self, context_size):
            return pd.DataFrame({
                "context": [f"k{context_size}"],
                "p_accessible": [0.5],
            })

    counter_selected = _accessibility_priors_for_base(
        "A", 3, None, {"A": FakeCounter()},
    )
    assert counter_selected.to_dict("list") == {
        "context": ["k3"],
        "p_accessible": [0.5],
    }

    assert _accessibility_priors_for_base("G", 3, None, {"A": FakeCounter()}) is None


def test_write_transfer_probability_tables_uses_shared_layout(tmp_path):
    target_rates = pd.DataFrame({
        "context": ["CCC", "AAA"],
        "ratio": [0.2, 0.3],
    })

    combined_file = _write_transfer_probability_tables(
        str(tmp_path),
        "run",
        "C",
        3,
        target_rates,
        p_acc=0.8,
        p_inacc=0.1,
    )

    acc = pd.read_csv(tmp_path / "run_accessible_C_k3.tsv", sep="\t")
    inacc = pd.read_csv(tmp_path / "run_inaccessible_C_k3.tsv", sep="\t")
    combined = pd.read_csv(combined_file, sep="\t")

    assert acc.to_dict("list") == {
        "encode": [0, 1],
        "context": ["AAA", "CCC"],
        "ratio": [0.8, 0.8],
    }
    assert inacc["ratio"].tolist() == [0.1, 0.1]
    assert combined.to_dict("list") == {
        "encode": [0, 1],
        "context": ["AAA", "CCC"],
        "accessible_prob": [0.8, 0.8],
        "inaccessible_prob": [0.1, 0.1],
    }
