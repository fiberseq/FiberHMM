"""Read-only regression coverage for the targeted SR production summarizer."""

import importlib.util
import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "summarize_targeted_strand_rescue.py"
# The completed NAPA production matrix is validation output, not a tracked
# fixture. Point FIBERHMM_TARGETED_SR_MATRIX at targeted_sr_run_matrix.json to
# run the end-to-end receipt check.
MATRIX_ENV = "FIBERHMM_TARGETED_SR_MATRIX"
MATRIX = Path(os.environ.get(MATRIX_ENV, "")).expanduser() if os.environ.get(MATRIX_ENV) else None


def _load_module():
    spec = importlib.util.spec_from_file_location("summarize_targeted_strand_rescue", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gnu_elapsed_parser_supports_both_time_shapes():
    module = _load_module()
    assert module._elapsed_seconds("3:38.73") == pytest.approx(218.73)
    assert module._elapsed_seconds("1:02:03.5") == pytest.approx(3723.5)


def test_declared_validation_path_relocates_only_repository_owned_tree(tmp_path):
    module = _load_module()
    matrix = (
        tmp_path / "repo" / "consensus_validation_outputs" / "run"
        / "manifests" / "matrix.json"
    )
    target = (
        tmp_path / "repo" / "consensus_validation_outputs" / "run"
        / "manifest.json"
    )
    matrix.parent.mkdir(parents=True)
    target.write_text("{}")

    relocated = module._resolve_declared_path(
        "/nonexistent/old-host/repo/consensus_validation_outputs/run/manifest.json",
        matrix,
    )
    assert relocated == target.resolve()

    unrelated = module._resolve_declared_path("/nonexistent/old-host/repo/input.bam", matrix)
    assert unrelated == Path("/nonexistent/old-host/repo/input.bam")


def test_completed_napa_receipt_report_timing_and_audit_agree():
    if MATRIX is None or not MATRIX.is_file():
        pytest.skip(f"set {MATRIX_ENV} to a targeted production run matrix")
    module = _load_module()
    summary = module.build_summary(MATRIX, {"ddda:napa"})
    assert summary["schema"] == "fiberhmm.validation.targeted_sr.production_summary.v1"
    assert summary["aggregate"]["target_count"] == 1
    assert summary["aggregate"]["fully_complete"] == 1
    assert summary["aggregate"]["targets_with_warnings"] == 0

    target = summary["targets"][0]
    assert target["inference"]["state"] == "complete"
    assert target["materialization"]["state"] == "complete"
    report = target["inference"]["report"]
    actions = report["actions"]
    audit = target["materialization"]["audit"]
    assert report["n_raw_reads"] > 20_000
    assert report["n_analyzed_molecules"] < report["n_raw_reads"]
    assert actions["rescue_decisions"] == audit["totals"]["msp_to_tf_rescues"]
    assert actions["tf_edge_updates"] == audit["totals"]["edge_refinements_tf_sr"]
    assert actions["nuc_edge_updates"] == audit["totals"]["edge_refinements_nuc_sr"]
    assert target["inference"]["gnu_time"]["wall_seconds"] > 200
    progress = target["inference"]["progress"]
    assert progress["by_name"]["bam_evidence_loading"]["wall_seconds"] > 50
    assert progress["event_count"] >= 16

    tsv = module.render_tsv(summary)
    markdown = module.render_markdown(summary)
    assert "stage_seconds__bam_evidence_loading" in tsv.splitlines()[0]
    assert "ddda:napa" in tsv
    assert "Inference stage totals" in markdown
    assert "ddda:napa" in markdown
