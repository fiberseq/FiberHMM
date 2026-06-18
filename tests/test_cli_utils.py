from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fiberhmm.cli.utils import (
    AccessibilityCounter,
    _accessibility_prior_row,
    _accessibility_priors_dataframe,
    _accessibility_priors_for_base,
    _aggregate_accessibility_counts,
    _build_utils_parser,
    _converted_model_json_payload,
    _dispatch_utils_command,
    _estimate_emission_probs,
    _estimate_transfer_context_size,
    _exit_if_missing_or_non_file,
    _load_raw_model_by_suffix,
    _load_transfer_accessibility_inputs,
    _maybe_generate_transfer_stats,
    _passes_transfer_base_filters,
    _passes_transfer_target_filters,
    _raw_model_parameter_arrays,
    _regression_diagnostic_plot_path,
    _regression_stats_summary_path,
    _save_accessibility_priors,
    _save_regression_diagnostic_plot,
    _scale_emission_probabilities,
    _target_bases_for_transfer_mode,
    _target_mod_positions_from_bam_read,
    _transfer_context_index,
    _transfer_probability_frame,
    _transfer_progress_postfix,
    _transfer_read_limit_reached,
    _TransferAccessibilityInputs,
    _trim_accessibility_context,
    _write_regression_stats_summary,
    _write_transfer_probability_tables,
)
from fiberhmm.core import bam_reader


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
    assert _load_raw_model_by_suffix(str(tmp_path / "model.pickle")) == {
        "loader": "pickle",
    }
    assert calls == [
        ("pickle", "model.pkl"),
        ("npz", "model.npz"),
        ("pickle", "model.pickle"),
    ]


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


def test_load_npz_model_raw_closes_archive(monkeypatch):
    import fiberhmm.cli.utils as cli_utils

    class FakeNpz:
        def __init__(self):
            self.closed = False
            self.values = {
                "startprob": np.array([0.4, 0.6]),
                "transmat": np.array([[0.9, 0.1], [0.2, 0.8]]),
                "emissionprob": np.array([[0.3, 0.7], [0.8, 0.2]]),
                "n_states": 2,
                "context_size": 3,
                "mode": "daf",
            }

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            self.closed = True

        def __getitem__(self, key):
            return self.values[key]

        def get(self, key, default=None):
            return self.values.get(key, default)

    fake_npz = FakeNpz()
    monkeypatch.setattr(
        cli_utils.np,
        "load",
        lambda path, allow_pickle: fake_npz,
    )

    data = cli_utils._load_npz_model_raw("model.npz")

    assert fake_npz.closed
    assert data["context_size"] == 3
    assert data["mode"] == "daf"


def test_converted_model_json_payload_coerces_arrays_and_metadata():
    data = {
        "startprob": [0.25, 0.75],
        "transmat": [[0.9, 0.1], [0.2, 0.8]],
        "emissionprob": [[0.4, 0.6], [0.7, 0.3]],
        "n_states": "2",
        "context_size": None,
        "mode": " daf ",
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


def test_exit_if_missing_or_non_file_reports_clear_errors(tmp_path, capsys):
    missing = tmp_path / "missing.json"
    directory = tmp_path / "model.json"
    model_file = tmp_path / "real.json"

    directory.mkdir()
    model_file.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit):
        _exit_if_missing_or_non_file(
            missing,
            missing_prefix="Input file not found",
            non_file_prefix="Input path is not a file",
        )
    assert "Input file not found" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        _exit_if_missing_or_non_file(
            directory,
            missing_prefix="File not found",
            non_file_prefix="Path is not a file",
        )
    assert "Path is not a file" in capsys.readouterr().err

    _exit_if_missing_or_non_file(
        model_file,
        missing_prefix="File not found",
        non_file_prefix="Path is not a file",
    )
    assert capsys.readouterr().err == ""


def test_build_utils_parser_parses_subcommands():
    parser = _build_utils_parser()

    convert = parser.parse_args(["convert", "in.pkl", "out.json"])
    assert (convert.command, convert.input, convert.output) == (
        "convert",
        "in.pkl",
        "out.json",
    )

    inspect = parser.parse_args(["inspect", "model.json", "--full"])
    assert (inspect.command, inspect.model, inspect.full) == (
        "inspect",
        "model.json",
        True,
    )

    transfer = parser.parse_args([
        "transfer",
        "--target",
        "daf.bam",
        "-o",
        "out",
        "--mode",
        "daf",
        "-k",
        "2",
        "3",
        "--stats",
    ])
    assert transfer.command == "transfer"
    assert transfer.target == "daf.bam"
    assert transfer.output == "out"
    assert transfer.context_sizes == [2, 3]
    assert transfer.stats is True

    adjust = parser.parse_args([
        "adjust",
        "model.json",
        "--state",
        "both",
        "--scale",
        "0.5",
        "-o",
        "adjusted.json",
    ])
    assert adjust.command == "adjust"
    assert adjust.state == "both"
    assert adjust.scale == 0.5
    assert adjust.output == "adjusted.json"


def test_dispatch_utils_command_routes_to_selected_subcommand(monkeypatch):
    import fiberhmm.cli.utils as cli_utils

    calls = []
    monkeypatch.setattr(cli_utils, "cmd_convert", lambda args: calls.append("convert"))
    monkeypatch.setattr(cli_utils, "cmd_inspect", lambda args: calls.append("inspect"))
    monkeypatch.setattr(cli_utils, "cmd_transfer", lambda args: calls.append("transfer"))
    monkeypatch.setattr(cli_utils, "cmd_adjust", lambda args: calls.append("adjust"))

    for command in ["convert", "inspect", "transfer", "adjust"]:
        _dispatch_utils_command(SimpleNamespace(command=command), parser=None)

    assert calls == ["convert", "inspect", "transfer", "adjust"]


def test_dispatch_utils_command_prints_help_without_subcommand():
    class Parser:
        printed = False

        def print_help(self):
            self.printed = True

    parser = Parser()

    with pytest.raises(SystemExit):
        _dispatch_utils_command(SimpleNamespace(command=None), parser)

    assert parser.printed


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

    estimate = _estimate_emission_probs(
        target_rates,
        accessibility,
        min_observations=100,
    )

    assert (estimate.p_acc, estimate.p_inacc) == (0.5, 0.1)
    assert estimate.diagnostics["n_contexts"] == 1
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

    estimate = _estimate_emission_probs(
        target_rates,
        accessibility,
        min_observations=100,
    )

    assert estimate.p_acc == pytest.approx(0.5)
    assert estimate.p_inacc == pytest.approx(0.1)
    assert estimate.diagnostics["n_contexts"] == 12
    assert estimate.diagnostics["r_squared"] == pytest.approx(1.0)


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

    read.tags = {"MM": "A+a,0;", "ML": np.asarray([], dtype=np.uint8)}
    monkeypatch.setattr(
        bam_reader,
        "parse_mm_tag_query_positions",
        lambda *args, **kwargs: pytest.fail("empty ML should not be parsed"),
    )
    assert _target_mod_positions_from_bam_read(read, 125, "pacbio-fiber") == set()


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

    result = _scale_emission_probabilities(emission, 1, 2.0)

    np.testing.assert_allclose(emission, [[0.2, 0.4], [0.6, 0.8]])
    np.testing.assert_allclose(result.adjusted, [[0.2, 0.4], [1.0, 1.0]])
    assert result.before.minimum == pytest.approx(0.6)
    assert result.before.maximum == pytest.approx(0.8)
    assert result.before.mean == pytest.approx(0.7)
    assert result.after.minimum == pytest.approx(1.0)
    assert result.after.maximum == pytest.approx(1.0)
    assert result.after.mean == pytest.approx(1.0)


def test_scale_emission_probabilities_scales_all_states():
    emission = np.array([
        [0.2, 0.4],
        [0.6, 0.8],
    ])

    result = _scale_emission_probabilities(emission, None, 0.5)

    np.testing.assert_allclose(result.adjusted, [[0.1, 0.2], [0.3, 0.4]])
    assert result.before.minimum == pytest.approx(0.2)
    assert result.before.maximum == pytest.approx(0.8)
    assert result.before.mean == pytest.approx(0.5)
    assert result.after.minimum == pytest.approx(0.1)
    assert result.after.maximum == pytest.approx(0.4)
    assert result.after.mean == pytest.approx(0.25)


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


def test_load_transfer_accessibility_inputs_reads_priors_tsv(tmp_path, capsys):
    priors_path = tmp_path / "priors.tsv"
    pd.DataFrame({
        "context": ["ACA"],
        "p_accessible": [0.75],
    }).to_csv(priors_path, sep="\t", index=False)
    args = SimpleNamespace(
        accessibility_priors=str(priors_path),
        reference_bam=None,
    )

    inputs = _load_transfer_accessibility_inputs(args, 3, ["C"])

    assert isinstance(inputs, _TransferAccessibilityInputs)
    assert inputs.counters is None
    assert inputs.priors.to_dict("list") == {
        "context": ["ACA"],
        "p_accessible": [0.75],
    }
    out = capsys.readouterr().out
    assert str(priors_path) in out
    assert "Loaded 1 contexts" in out


def test_load_transfer_accessibility_inputs_computes_reference_summary(
    monkeypatch,
    capsys,
):
    import fiberhmm.cli.utils as cli_utils

    class Counter:
        total_accessible = 25
        total_positions = 100

    calls = []

    def fake_process_reference_bam(bam_path, max_context, args):
        calls.append((bam_path, max_context, args))
        return {"C": Counter()}

    monkeypatch.setattr(
        cli_utils,
        "_process_reference_bam",
        fake_process_reference_bam,
    )
    args = SimpleNamespace(
        accessibility_priors=None,
        reference_bam="reference.bam",
    )

    inputs = _load_transfer_accessibility_inputs(args, 5, ["C", "G"])

    assert sorted(inputs.counters) == ["C"]
    assert isinstance(inputs.counters["C"], Counter)
    assert inputs.priors is None
    assert calls == [("reference.bam", 5, args)]
    out = capsys.readouterr().out
    assert "Computing accessibility priors from: reference.bam" in out
    assert "C: 100 positions, 25.0% accessible" in out


def test_load_transfer_accessibility_inputs_requires_source(capsys):
    args = SimpleNamespace(
        accessibility_priors=None,
        reference_bam=None,
    )

    with pytest.raises(SystemExit):
        _load_transfer_accessibility_inputs(args, 3, ["C"])

    assert "Must provide one of" in capsys.readouterr().out


def test_estimate_transfer_context_size_writes_available_bases(monkeypatch, capsys):
    import fiberhmm.cli.utils as cli_utils

    class TargetCounter:
        def __init__(self, base):
            self.base = base

        def get_probabilities(self, context_size):
            return pd.DataFrame({
                "context": [f"A{self.base}A"],
                "ratio": [0.2],
                "total": [200],
            })

    class PriorsCounter:
        def get_accessibility_priors(self, context_size):
            return pd.DataFrame({
                "context": ["ACA"],
                "p_accessible": [0.6],
                "total_bp": [200],
            })

    writes = []
    monkeypatch.setattr(
        cli_utils,
        "_estimate_emission_probs",
        lambda rates, priors, min_observations: cli_utils._EmissionEstimate(
            p_acc=0.7,
            p_inacc=0.2,
            diagnostics={
                "n_contexts": len(priors),
                "r_squared": 0.9,
                "x": np.array([0.6]),
                "y": np.array([0.2]),
                "w": np.array([10.0]),
            },
        ),
    )
    monkeypatch.setattr(
        cli_utils,
        "_write_transfer_probability_tables",
        lambda *args: writes.append(args) or "tables/run_C_k1.tsv",
    )
    args = SimpleNamespace(min_observations=100)

    regression_data = _estimate_transfer_context_size(
        1,
        args,
        "tables",
        "run",
        ["C", "G"],
        {"C": TargetCounter("C"), "G": TargetCounter("G")},
        None,
        {"C": PriorsCounter()},
    )

    assert list(regression_data) == ["C"]
    assert regression_data["C"]["p_acc"] == 0.7
    assert regression_data["C"]["p_inacc"] == 0.2
    assert len(writes) == 1
    assert writes[0][0:4] == ("tables", "run", "C", 1)
    out = capsys.readouterr().out
    assert "Regression R-squared: 0.900" in out
    assert "Warning: No accessibility priors for G" in out


def test_maybe_generate_transfer_stats_respects_stats_flag(monkeypatch, capsys):
    import fiberhmm.cli.utils as cli_utils

    calls = []
    monkeypatch.setattr(
        cli_utils,
        "_generate_regression_stats",
        lambda data, plots_dir, base_name, k: calls.append(
            (data, plots_dir, base_name, k)
        ),
    )
    regression_data = {1: {"C": {"diagnostics": {}}}, 2: {}}

    _maybe_generate_transfer_stats(
        SimpleNamespace(stats=False, context_sizes=[1, 2]),
        regression_data,
        "plots",
        "run",
    )
    assert calls == []

    _maybe_generate_transfer_stats(
        SimpleNamespace(stats=True, context_sizes=[1, 2]),
        regression_data,
        "plots",
        "run",
    )

    assert calls == [({"C": {"diagnostics": {}}}, "plots", "run", 1)]
    assert "Generating statistics and plots" in capsys.readouterr().out


def test_write_regression_stats_summary_formats_diagnostics(tmp_path):
    summary_file = _regression_stats_summary_path(tmp_path, "run", 3)
    summary_path = tmp_path / "run_k3_regression_stats.txt"

    assert _regression_diagnostic_plot_path(
        tmp_path, "run", "C", 3,
    ) == str(tmp_path / "run_C_k3_regression.png")

    _write_regression_stats_summary(
        {
            "C": {
                "diagnostics": {
                    "n_contexts": 12,
                    "total_target_obs": 1234,
                    "total_ref_obs": 5678,
                    "r_squared": 0.91234,
                    "intercept": 0.2,
                    "slope": 0.5,
                    "swapped": True,
                },
                "p_acc": 0.7,
                "p_inacc": 0.2,
            }
        },
        summary_file,
        context_size=3,
    )

    assert summary_file == str(summary_path)
    text = summary_path.read_text()
    assert "FiberHMM Transfer Learning Regression Statistics (k=3)" in text
    assert "C-centered Contexts" in text
    assert "Target observations:           1,234" in text
    assert "Reference observations:        5,678" in text
    assert "R-squared: 0.9123" in text
    assert "P(m|accessible):   0.7000" in text
    assert "P(m|inaccessible): 0.2000" in text
    assert "Enrichment ratio:  3.5x" in text
    assert "Warning: Values were swapped" in text


class _FakeRegressionAxis:
    def __init__(self):
        self.scatter_calls = []
        self.plot_calls = []
        self.xlabel = None
        self.ylabel = None
        self.title = None
        self.xlim = None
        self.ylim = None
        self.legend_kwargs = None
        self.grid_calls = []

    def scatter(self, *args, **kwargs):
        self.scatter_calls.append((args, kwargs))

    def plot(self, *args, **kwargs):
        self.plot_calls.append((args, kwargs))

    def set_xlabel(self, label, **kwargs):
        self.xlabel = (label, kwargs)

    def set_ylabel(self, label, **kwargs):
        self.ylabel = (label, kwargs)

    def set_title(self, label, **kwargs):
        self.title = (label, kwargs)

    def set_xlim(self, *args):
        self.xlim = args

    def set_ylim(self, *args):
        self.ylim = args

    def legend(self, **kwargs):
        self.legend_kwargs = kwargs

    def grid(self, *args, **kwargs):
        self.grid_calls.append((args, kwargs))


class _FakeRegressionPlt:
    def __init__(self):
        self.fig = object()
        self.ax = _FakeRegressionAxis()
        self.subplots_calls = []
        self.tight_layout_calls = 0
        self.savefig_calls = []
        self.closed = []

    def subplots(self, *args, **kwargs):
        self.subplots_calls.append((args, kwargs))
        return self.fig, self.ax

    def tight_layout(self):
        self.tight_layout_calls += 1

    def savefig(self, *args, **kwargs):
        self.savefig_calls.append((args, kwargs))

    def close(self, fig):
        self.closed.append(fig)


class _FailingRegressionPlt(_FakeRegressionPlt):
    def savefig(self, *args, **kwargs):
        raise RuntimeError("plot save failed")


def _regression_plot_data():
    return {
        "x": np.array([0.0, 0.5, 1.0]),
        "y": np.array([0.1, 0.4, 0.8]),
        "w": np.array([0.2, 25.0, 99.0]),
        "diagnostics": {
            "intercept": 0.1,
            "slope": 0.7,
            "r_squared": 0.95,
        },
    }


def test_save_regression_diagnostic_plot_writes_expected_png():
    plt = _FakeRegressionPlt()

    png_path = _save_regression_diagnostic_plot(
        plt, "plots", "run", "C", 3, _regression_plot_data(),
    )

    assert png_path == "plots/run_C_k3_regression.png"
    assert plt.subplots_calls == [((), {"figsize": (8, 6)})]
    assert plt.tight_layout_calls == 1
    assert plt.savefig_calls == [((png_path,), {"dpi": 150})]
    assert plt.closed == [plt.fig]
    np.testing.assert_allclose(plt.ax.scatter_calls[0][1]["s"], [1.0, 25.0, 50.0])
    assert plt.ax.scatter_calls[0][1]["c"] == "steelblue"
    assert len(plt.ax.scatter_calls) == 2
    assert len(plt.ax.plot_calls) == 1
    assert plt.ax.legend_kwargs == {"fontsize": 10}
    assert plt.ax.grid_calls == [((True,), {"alpha": 0.3})]


def test_save_regression_diagnostic_plot_closes_on_save_failure():
    plt = _FailingRegressionPlt()

    with pytest.raises(RuntimeError, match="plot save failed"):
        _save_regression_diagnostic_plot(
            plt, "plots", "run", "C", 3, _regression_plot_data(),
        )

    assert plt.closed == [plt.fig]
