"""Failure cleanup tests for region-parallel orchestration."""

from __future__ import annotations

import pytest

from fiberhmm.inference import parallel, region_pipeline


class _FailingFuture:
    def result(self):
        raise RuntimeError("worker failed")


class _FailingExecutor:
    def __init__(self, *args, **kwargs):
        self.future = _FailingFuture()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, *args, **kwargs):
        return self.future


def _install_failing_region_pool(monkeypatch, tmp_path):
    temp_dirs = []

    def fake_mkdtemp(prefix, dir):
        temp_dir = tmp_path / f"{prefix}cleanup_{len(temp_dirs)}"
        temp_dir.mkdir()
        (temp_dir / "sentinel.txt").write_text("temp")
        temp_dirs.append(temp_dir)
        return str(temp_dir)

    monkeypatch.setattr(
        region_pipeline,
        "_get_genome_regions",
        lambda *args, **kwargs: [("chr1", 0, 100)],
    )
    monkeypatch.setattr(region_pipeline, "ProcessPoolExecutor", _FailingExecutor)
    monkeypatch.setattr(region_pipeline, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(region_pipeline.tempfile, "mkdtemp", fake_mkdtemp)

    return temp_dirs


def _indexed_input_bam(tmp_path):
    input_bam = tmp_path / "input.bam"
    input_bam.write_bytes(b"")
    (tmp_path / "input.bam.bai").write_bytes(b"")
    return str(input_bam)


def test_region_parallel_bam_cleans_temp_dir_on_worker_failure(monkeypatch, tmp_path):
    temp_dirs = _install_failing_region_pool(monkeypatch, tmp_path)
    input_bam = _indexed_input_bam(tmp_path)

    with pytest.raises(RuntimeError, match="worker failed"):
        parallel._process_bam_region_parallel(
            input_bam=input_bam,
            output_bam=str(tmp_path / "out.bam"),
            model_path="model.json",
            train_rids=set(),
            edge_trim=0,
            circular=False,
            mode="m6a",
            context_size=6,
            msp_min_size=0,
            n_cores=1,
        )

    assert temp_dirs
    assert all(not temp_dir.exists() for temp_dir in temp_dirs)


def test_region_parallel_bed_cleans_temp_dir_on_worker_failure(monkeypatch, tmp_path):
    temp_dirs = _install_failing_region_pool(monkeypatch, tmp_path)
    input_bam = _indexed_input_bam(tmp_path)

    with pytest.raises(RuntimeError, match="worker failed"):
        parallel._process_bed_region_parallel(
            input_bam=input_bam,
            output_bed=str(tmp_path / "out.bed"),
            model_path="model.json",
            train_rids=set(),
            edge_trim=0,
            circular=False,
            mode="m6a",
            context_size=6,
            msp_min_size=0,
            n_cores=1,
        )

    assert temp_dirs
    assert all(not temp_dir.exists() for temp_dir in temp_dirs)


def test_fused_region_parallel_bam_cleans_temp_dir_on_worker_failure(monkeypatch, tmp_path):
    temp_dirs = _install_failing_region_pool(monkeypatch, tmp_path)
    input_bam = _indexed_input_bam(tmp_path)

    with pytest.raises(RuntimeError, match="worker failed"):
        parallel._process_bam_region_parallel_fused(
            input_bam=input_bam,
            output_bam=str(tmp_path / "out.bam"),
            apply_model_path="model.json",
            recall_model_path=None,
            train_rids=set(),
            edge_trim=0,
            circular=False,
            mode="m6a",
            context_size=6,
            msp_min_size=0,
            nuc_min_size=0,
            min_mapq=0,
            prob_threshold=0,
            min_read_length=0,
            with_scores=False,
            min_llr=0.0,
            min_opps=0,
            unify_threshold=0,
            emission_uplift=1.0,
            also_write_legacy=True,
            downstream_compat=True,
            n_cores=1,
            region_size=100,
            skip_scaffolds=False,
            chroms=None,
            io_threads=1,
        )

    assert temp_dirs
    assert all(not temp_dir.exists() for temp_dir in temp_dirs)
