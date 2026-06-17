"""Failure cleanup tests for region-parallel orchestration."""

from __future__ import annotations

import os

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


def test_read_rate_handles_zero_elapsed():
    assert region_pipeline._read_rate(10, 2.0) == 5.0
    assert region_pipeline._read_rate(10, 0.0) == 0


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


def test_region_completion_summary_formats_counts_and_rate():
    assert region_pipeline._region_completion_summary(
        total_reads=1234,
        reads_with_footprints=56,
        elapsed=2.0,
    ) == "Completed: 1,234 reads | 56 with footprints | 617.0 reads/s | 2.0s"


def test_region_work_item_builders_use_stable_temp_names(tmp_path):
    regions = [("chr1", 0, 100), ("chr2", 5, 25)]
    temp_dir = str(tmp_path)

    assert region_pipeline._region_temp_path(temp_dir, 12, "bam") == (
        str(tmp_path / "region_000012.bam")
    )

    bam_item = region_pipeline._region_bam_work_item(
        regions[0],
        "input.bam",
        temp_dir,
        12,
        include_tsv=True,
    )
    bam_items = region_pipeline._region_bam_work_items(
        regions,
        "input.bam",
        temp_dir,
        include_tsv=True,
    )
    bed_items = region_pipeline._region_bed_work_items(regions, "input.bam", temp_dir)

    assert bam_item.region == regions[0]
    assert bam_item.temp_bam_path == str(tmp_path / "region_000012.bam")
    assert bam_item.temp_tsv_path == str(tmp_path / "region_000012.tsv")
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


def test_region_result_has_existing_tsv_requires_path_and_file(tmp_path):
    missing = tmp_path / "missing.tsv"
    existing = tmp_path / "region.tsv"
    existing.write_text("read\tposterior\n")

    assert not region_pipeline._region_result_has_existing_tsv(
        region_pipeline.RegionBamResult("region.bam", 1, 1, 1)
    )
    assert not region_pipeline._region_result_has_existing_tsv(
        region_pipeline.RegionBamResult("region.bam", 1, 1, 1, str(missing))
    )
    assert region_pipeline._region_result_has_existing_tsv(
        region_pipeline.RegionBamResult("region.bam", 1, 1, 1, str(existing))
    )


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


def test_submit_region_futures_maps_futures_to_region_indices():
    class Executor:
        def __init__(self):
            self.submitted = []

        def submit(self, worker, item):
            future = object()
            self.submitted.append((future, worker, item))
            return future

    def worker(item):
        return item

    executor = Executor()
    work_items = ["a", "b"]

    futures = region_pipeline._submit_region_futures(executor, worker, work_items)

    assert futures == {
        executor.submitted[0][0]: 0,
        executor.submitted[1][0]: 1,
    }
    assert [(worker_fn, item) for _, worker_fn, item in executor.submitted] == [
        (worker, "a"),
        (worker, "b"),
    ]


def test_make_output_temp_dir_places_directory_next_to_output(tmp_path):
    output_path = tmp_path / "nested" / "out.bam"
    output_path.parent.mkdir()

    temp_dir = region_pipeline._make_output_temp_dir(
        str(output_path),
        ".fiberhmm_test_",
    )

    assert os.path.dirname(temp_dir) == str(output_path.parent)
    assert os.path.isdir(temp_dir)
    assert os.path.basename(temp_dir).startswith(".fiberhmm_test_")


def test_merge_region_posterior_outputs_reports_tsv_and_conversion(
    monkeypatch,
    tmp_path,
    capsys,
):
    temp_tsvs = [(0, "region_0.tsv")]
    output_posteriors = str(tmp_path / "out.h5")
    tsv_path = tmp_path / "out.tsv.gz"
    tsv_path.write_bytes(b"x" * 1024)
    times = iter([10.0, 12.5])
    merge_calls = []

    def fake_merge(*args):
        merge_calls.append(args)
        return 7

    monkeypatch.setattr(region_pipeline.time, "time", lambda: next(times))
    monkeypatch.setattr(region_pipeline, "_merge_region_posteriors_tsv", fake_merge)
    monkeypatch.setattr(
        region_pipeline,
        "region_posteriors_tsv_output_path",
        lambda output: str(tsv_path),
    )

    region_pipeline._merge_region_posterior_outputs(
        temp_tsvs,
        output_posteriors,
        mode="pacbio-fiber",
        context_size=3,
        edge_trim=10,
        input_bam="input.bam",
    )

    assert merge_calls == [(
        temp_tsvs,
        output_posteriors,
        "pacbio-fiber",
        3,
        10,
        "input.bam",
    )]
    out = capsys.readouterr().out
    assert "Merging 1 posterior files..." in out
    assert f"Posteriors: 7 fibers -> {tsv_path} (0.0 MB, 2.5s)" in out
    assert f"tsv2h5 {tsv_path} {output_posteriors}" in out


def test_finalize_region_bam_output_concatenates_reports_and_indexes(
    monkeypatch,
    capsys,
):
    calls = []

    def fake_concatenate(input_bam, output_bam, non_empty_bams, temp_dir):
        calls.append(("concat", input_bam, output_bam, non_empty_bams, temp_dir))

    def fake_sort(output_bam, threads):
        calls.append(("sort", output_bam, threads))

    monkeypatch.setattr(
        region_pipeline, "_concatenate_region_bams", fake_concatenate
    )
    monkeypatch.setattr(region_pipeline, "_sort_and_index_bam", fake_sort)
    monkeypatch.setattr(region_pipeline.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        region_pipeline.os.path,
        "getsize",
        lambda path: 2 * 1024**3,
    )

    region_pipeline._finalize_region_bam_output(
        "input.bam",
        "output.bam",
        ["region_0.bam"],
        "tmp",
        n_cores=3,
    )

    assert calls == [
        ("concat", "input.bam", "output.bam", ["region_0.bam"], "tmp"),
        ("sort", "output.bam", 3),
    ]
    out = capsys.readouterr().out
    assert "Output BAM: 2.00GB" in out
    assert "Step: Index/Sort..." in out


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
