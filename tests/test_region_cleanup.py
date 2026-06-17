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


def test_region_skip_summary_formats_counts(capsys):
    aggregation = region_pipeline.RegionBamAggregation()
    aggregation.add_result(
        0,
        region_pipeline.RegionBamResult(
            "region.bam",
            total_reads=10,
            reads_with_footprints=3,
            written=10,
            skip_reasons={"low_mapq": 2, "empty": 0},
        ),
    )

    region_pipeline._print_skip_reasons_summary(
        aggregation,
        footprint_label="With FP",
    )

    out = capsys.readouterr().out
    assert "Processed: 10 | Skipped: 2 | With FP: 3" in out
    assert "low_mapq: 2 (16.7%)" in out
    assert "empty" not in out


def test_region_worker_ready_message_prints_once(monkeypatch, capsys):
    monkeypatch.setattr(region_pipeline.time, "time", lambda: 12.0)

    first = region_pipeline._report_workers_ready_once(
        None,
        pool_start=10.0,
        message="Processing regions...",
    )

    assert first == 12.0
    assert "Workers ready (2.0s). Processing regions..." in capsys.readouterr().out

    second = region_pipeline._report_workers_ready_once(
        first,
        pool_start=10.0,
        message="Ignored",
    )

    assert second == first
    assert capsys.readouterr().out == ""


def test_region_progress_formats_counts_and_rate(monkeypatch, capsys):
    monkeypatch.setattr(region_pipeline.time, "time", lambda: 12.0)
    aggregation = region_pipeline.RegionBamAggregation(
        total_reads=10,
        reads_with_footprints=4,
        temp_bams=[(0, "region_0.bam"), (1, "region_1.bam")],
    )

    region_pipeline._print_region_progress(
        aggregation,
        total_regions=3,
        start_time=10.0,
        footprint_label="With FP",
        rate_unit="r/s",
        rate_precision=0,
    )

    out = capsys.readouterr().out
    assert "Regions: 2/3 | Reads: 10 | With FP: 4 | 5 r/s" in out


def test_region_work_item_builders_use_stable_temp_names(tmp_path):
    regions = [("chr1", 0, 100), ("chr2", 5, 25)]
    temp_dir = str(tmp_path)

    bam_items = region_pipeline._region_bam_work_items(
        regions,
        "input.bam",
        temp_dir,
        include_tsv=True,
    )
    bed_items = region_pipeline._region_bed_work_items(regions, "input.bam", temp_dir)

    assert [item.region for item in bam_items] == regions
    assert [item.temp_bam_path for item in bam_items] == [
        str(tmp_path / "region_000000.bam"),
        str(tmp_path / "region_000001.bam"),
    ]
    assert [item.temp_tsv_path for item in bam_items] == [
        str(tmp_path / "region_000000.tsv"),
        str(tmp_path / "region_000001.tsv"),
    ]
    assert [item.temp_bed_path for item in bed_items] == [
        str(tmp_path / "region_000000.bed"),
        str(tmp_path / "region_000001.bed"),
    ]


def test_ordered_existing_temp_paths_sorts_and_filters_empty(tmp_path):
    nonempty_late = tmp_path / "region_2.bam"
    nonempty_early = tmp_path / "region_0.bam"
    empty = tmp_path / "region_1.bam"
    missing = tmp_path / "missing.bam"
    nonempty_late.write_bytes(b"late")
    nonempty_early.write_bytes(b"early")
    empty.write_bytes(b"")

    assert region_pipeline._ordered_existing_temp_paths([
        (2, str(nonempty_late)),
        (1, str(empty)),
        (3, str(missing)),
        (0, str(nonempty_early)),
    ]) == [str(nonempty_early), str(nonempty_late)]


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
