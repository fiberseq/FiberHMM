import numpy as np
import pandas as pd
import pytest

from fiberhmm.core import bam_reader
from fiberhmm.cli.utils import (
    AccessibilityCounter,
    _accessibility_prior_row,
    _accessibility_priors_for_base,
    _accessibility_priors_dataframe,
    _aggregate_accessibility_counts,
    _converted_model_json_payload,
    _estimate_emission_probs,
    _load_raw_model_by_suffix,
    _passes_transfer_base_filters,
    _passes_transfer_target_filters,
    _raw_model_parameter_arrays,
    _save_accessibility_priors,
    _scale_emission_probabilities,
    _target_bases_for_transfer_mode,
    _target_mod_positions_from_bam_read,
    _trim_accessibility_context,
    _transfer_context_index,
    _transfer_probability_frame,
    _transfer_progress_postfix,
    _transfer_read_limit_reached,
    _write_transfer_probability_tables,
)


def test_target_bases_for_transfer_mode():
    assert _target_bases_for_transfer_mode('pacbio-fiber') == ['A']
    assert _target_bases_for_transfer_mode('nanopore-fiber') == ['A']
    assert _target_bases_for_transfer_mode('daf') == ['C', 'G']


def test_load_raw_model_by_suffix_uses_expected_loader(monkeypatch, tmp_path):
    import fiberhmm.cli.utils as cli_utils

    calls = []

    def fake_pickle(path):
        calls.append(("pickle", path.name))
        return {"loader": "pickle"}

    def fake_npz(path):
        calls.append(("npz", path.name))
        return {"loader": "npz"}

    monkeypatch.setattr(cli_utils, "_load_pickle_model_raw", fake_pickle)
    monkeypatch.setattr(cli_utils, "_load_npz_model_raw", fake_npz)

    assert _load_raw_model_by_suffix(tmp_path / "model.pkl") == {
        "loader": "pickle",
    }
    assert _load_raw_model_by_suffix(tmp_path / "model.npz") == {"loader": "npz"}
    assert calls == [("pickle", "model.pkl"), ("npz", "model.npz")]


def test_load_raw_model_by_suffix_falls_back_to_npz(monkeypatch, tmp_path):
    import fiberhmm.cli.utils as cli_utils

    calls = []

    def fail_pickle(path):
        calls.append(("pickle", path.name))
        raise ValueError("not pickle")

    def fake_npz(path):
        calls.append(("npz", path.name))
        return {"loader": "npz"}

    monkeypatch.setattr(cli_utils, "_load_pickle_model_raw", fail_pickle)
    monkeypatch.setattr(cli_utils, "_load_npz_model_raw", fake_npz)

    assert _load_raw_model_by_suffix(tmp_path / "model.legacy") == {
        "loader": "npz",
    }
    assert calls == [("pickle", "model.legacy"), ("npz", "model.legacy")]


def test_converted_model_json_payload_coerces_arrays_and_metadata():
    data = {
        "startprob": [0.25, 0.75],
        "transmat": [[0.9, 0.1], [0.2, 0.8]],
        "emissionprob": [[0.4, 0.6], [0.7, 0.3]],
        "n_states": "2",
        "context_size": "3",
        "mode": "daf",
    }
    startprob, transmat, emissionprob = _raw_model_parameter_arrays(data)

    assert _converted_model_json_payload(
        data, startprob, transmat, emissionprob,
    ) == {
        "model_type": "FiberHMM",
        "version": "2.0",
        "n_states": 2,
        "startprob": [0.25, 0.75],
        "transmat": [[0.9, 0.1], [0.2, 0.8]],
        "emissionprob": [[0.4, 0.6], [0.7, 0.3]],
        "context_size": 3,
        "mode": "daf",
    }


def test_accessibility_counter_records_accessible_and_protected_contexts():
    counter = AccessibilityCounter(max_context=1, center_base='A')
    footprint_mask = np.array([False, False, False, True, False])

    counter.process_read_with_footprints("CACAC", footprint_mask, edge_trim=0)
    counter.process_read_with_footprints("CANAC", footprint_mask, edge_trim=0)

    assert counter.counts["CAC"] == [1, 2]
    assert counter.total_accessible == 1
    assert counter.total_positions == 2


def test_accessibility_aggregation_trims_and_merges_context_counts():
    assert _trim_accessibility_context("AACAA", 1) == "ACA"
    assert _trim_accessibility_context("AACAA", 0) == "AACAA"

    assert _aggregate_accessibility_counts(
        {
            "AACAA": [1, 3],
            "TACAT": [2, 5],
            "GAGAG": [7, 11],
        },
        max_context=2,
        context_size=1,
    ) == {
        "ACA": [3, 8],
        "AGA": [7, 11],
    }


def test_accessibility_prior_rows_and_dataframe_are_sorted():
    assert _accessibility_prior_row("AAA", [0, 0]) == {
        "context": "AAA",
        "accessible_bp": 0,
        "total_bp": 0,
        "p_accessible": 0.5,
    }

    df = _accessibility_priors_dataframe({
        "CCC": [1, 4],
        "AAA": [3, 6],
    })

    assert df.to_dict("records") == [
        {
            "context": "AAA",
            "accessible_bp": 3,
            "total_bp": 6,
            "p_accessible": 0.5,
        },
        {
            "context": "CCC",
            "accessible_bp": 1,
            "total_bp": 4,
            "p_accessible": 0.25,
        },
    ]


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


def test_passes_transfer_target_filters_checks_reference_span():
    class Read:
        is_unmapped = False
        is_secondary = False
        is_supplementary = False
        mapping_quality = 20
        query_sequence = "ACGT"
        reference_start = 10
        reference_end = 14

    assert _passes_transfer_target_filters(Read(), min_mapq=20, min_read_length=4)

    read = Read()
    read.reference_end = 13
    assert not _passes_transfer_target_filters(read, min_mapq=20, min_read_length=4)

    read = Read()
    read.reference_start = None
    assert not _passes_transfer_target_filters(read, min_mapq=20, min_read_length=4)


def test_target_mod_positions_from_bam_read_uses_mm_ml_tags(monkeypatch):
    class Read:
        query_sequence = "AAAA"
        is_reverse = True
        tags = {"Mm": "A+a,0;", "Ml": b"\xc8"}

        def has_tag(self, tag):
            return tag in self.tags

        def get_tag(self, tag):
            return self.tags[tag]

    captured = {}

    def fake_parse(mm_tag, ml_tag, sequence, is_reverse, prob_threshold, mode):
        captured.update({
            "mm_tag": mm_tag,
            "ml_tag": ml_tag,
            "sequence": sequence,
            "is_reverse": is_reverse,
            "prob_threshold": prob_threshold,
            "mode": mode,
        })
        return {2}

    monkeypatch.setattr(bam_reader, "parse_mm_tag_query_positions", fake_parse)

    assert _target_mod_positions_from_bam_read(Read(), 125, "pacbio-fiber") == {2}
    assert captured == {
        "mm_tag": "A+a,0;",
        "ml_tag": b"\xc8",
        "sequence": "AAAA",
        "is_reverse": True,
        "prob_threshold": 125,
        "mode": "pacbio-fiber",
    }

    read = Read()
    read.tags = {}
    assert _target_mod_positions_from_bam_read(read, 125, "pacbio-fiber") is None


def test_transfer_progress_postfix_formats_optional_footprint_counts():
    assert _transfer_progress_postfix(12345) == {"reads": "12,345"}
    assert _transfer_progress_postfix(12345, reads_with_footprints=67) == {
        "reads": "12,345",
        "w/footprints": "67",
    }


def test_transfer_read_limit_reached_respects_zero_as_unbounded():
    assert not _transfer_read_limit_reached(100, 0)
    assert not _transfer_read_limit_reached(99, 100)
    assert _transfer_read_limit_reached(100, 100)
    assert _transfer_read_limit_reached(101, 100)


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


def test_transfer_context_index_sorts_contexts_and_assigns_encodes():
    target_rates = pd.DataFrame({
        "context": ["CCC", "AAA", "CAC"],
        "ratio": [0.2, 0.3, 0.4],
    })

    context_index = _transfer_context_index(target_rates)

    assert context_index.to_dict("list") == {
        "encode": [0, 1, 2],
        "context": ["AAA", "CAC", "CCC"],
    }


def test_transfer_probability_frame_adds_constant_ratio_column():
    context_index = pd.DataFrame({
        "encode": [0, 1],
        "context": ["AAA", "CCC"],
    })

    frame = _transfer_probability_frame(context_index, ratio=0.8)

    assert frame.to_dict("list") == {
        "encode": [0, 1],
        "context": ["AAA", "CCC"],
        "ratio": [0.8, 0.8],
    }
    assert "ratio" not in context_index.columns


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


def test_save_accessibility_priors_writes_known_bases(tmp_path, capsys):
    class FakeCounter:
        def __init__(self, base):
            self.base = base

        def get_accessibility_priors(self, context_size):
            return pd.DataFrame({
                "context": [f"{self.base}{context_size}"],
                "p_accessible": [0.5],
            })

    written = _save_accessibility_priors(
        str(tmp_path),
        "run",
        3,
        {"C": FakeCounter("C"), "N": FakeCounter("N")},
    )

    expected = str(tmp_path / "run_accessibility_priors_C_k3.tsv")
    assert written == [expected]
    saved = pd.read_csv(expected, sep="\t")
    assert saved.to_dict("list") == {
        "context": ["C3"],
        "p_accessible": [0.5],
    }
    assert expected in capsys.readouterr().out
