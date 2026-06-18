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


def test_fused_region_total_summary_formats_rate(monkeypatch):
    aggregation = region_pipeline.RegionBamAggregation(
        total_reads=10,
        reads_with_footprints=4,
    )
    monkeypatch.setattr(region_pipeline.time, "time", lambda: 14.0)

    assert region_pipeline._fused_region_total_summary(
        aggregation,
        start_time=10.0,
    ) == "  Total: 10 reads, 4 with footprints, 2.5 r/s"


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


def test_region_worker_param_builders_extend_base_contract():
    train_rids = {"read1"}

    bam_params = region_pipeline._region_bam_worker_params(
        edge_trim=11,
        circular=True,
        mode="daf",
        context_size=5,
        msp_min_size=61,
        nuc_min_size=87,
        min_mapq=20,
        prob_threshold=128,
        min_read_length=100,
        with_scores=True,
        train_rids=train_rids,
        primary_only=True,
        return_posteriors=True,
        write_msps=False,
        io_threads=3,
    )

    assert bam_params == {
        "edge_trim": 11,
        "circular": True,
        "mode": "daf",
        "context_size": 5,
        "msp_min_size": 61,
        "nuc_min_size": 87,
        "min_mapq": 20,
        "prob_threshold": 128,
        "min_read_length": 100,
        "with_scores": True,
        "train_rids": train_rids,
        "primary_only": True,
        "return_posteriors": True,
        "write_msps": False,
        "io_threads": 3,
    }

    fused_params = region_pipeline._fused_region_worker_params(
        edge_trim=11,
        circular=False,
        mode="pacbio-fiber",
        context_size=6,
        msp_min_size=62,
        nuc_min_size=88,
        min_mapq=21,
        prob_threshold=129,
        min_read_length=101,
        with_scores=False,
        train_rids=train_rids,
        primary_only=False,
        io_threads=4,
        min_llr=4.5,
        min_opps=3,
        unify_threshold=25,
        also_write_legacy=True,
        downstream_compat=False,
        recall_nucs=True,
        split_min_llr=5.5,
        split_min_opps=4,
        filter_chimeras=True,
        chimera_min_seg=6,
        chimera_purity=0.9,
        phase_nrl=147,
        pg_record={"ID": "fiberhmm"},
        ref_fasta_path="ref.fa",
    )

    assert fused_params["mode"] == "pacbio-fiber"
    assert fused_params["context_size"] == 6
    assert fused_params["train_rids"] is train_rids
    assert fused_params["io_threads"] == 4
    assert fused_params["min_llr"] == 4.5
    assert fused_params["min_opps"] == 3
    assert fused_params["unify_threshold"] == 25
    assert fused_params["also_write_legacy"] is True
    assert fused_params["downstream_compat"] is False
    assert fused_params["recall_nucs"] is True
    assert fused_params["split_min_llr"] == 5.5
    assert fused_params["split_min_opps"] == 4
    assert fused_params["filter_chimeras"] is True
    assert fused_params["chimera_min_seg"] == 6
    assert fused_params["chimera_purity"] == 0.9
    assert fused_params["phase_nrl"] == 147
    assert fused_params["pg_record"] == {"ID": "fiberhmm"}
    assert fused_params["ref_fasta_path"] == "ref.fa"


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
    bed_item = region_pipeline._region_bed_work_item(
        regions[1],
        "input.bam",
        temp_dir,
        13,
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
    assert bed_item.region == regions[1]
    assert bed_item.temp_bed_path == str(tmp_path / "region_000013.bed")
    assert [item.temp_bed_path for item in bed_items] == [
        str(tmp_path / "region_000000.bed"),
        str(tmp_path / "region_000001.bed"),
    ]


def test_region_result_has_existing_tsv_requires_path_and_file(tmp_path):
    missing = tmp_path / "missing.tsv"
    empty = tmp_path / "empty.tsv"
    existing = tmp_path / "region.tsv"
    empty.write_text("")
    existing.write_text("read\tposterior\n")

    assert not region_pipeline._region_result_has_existing_tsv(
        region_pipeline.RegionBamResult("region.bam", 1, 1, 1)
    )
    assert not region_pipeline._region_result_has_existing_tsv(
        region_pipeline.RegionBamResult("region.bam", 1, 1, 1, str(missing))
    )
    assert not region_pipeline._region_result_has_existing_tsv(
        region_pipeline.RegionBamResult("region.bam", 1, 1, 1, str(empty))
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


def test_add_region_result_to_aggregation_handles_optional_tsv(tmp_path):
    class Aggregation:
        def __init__(self):
            self.calls = []

        def add_result(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    tsv_path = tmp_path / "region.tsv"
    tsv_path.write_text("read\tposterior\n")
    result = region_pipeline.RegionBamResult(
        "region.bam",
        total_reads=10,
        reads_with_footprints=4,
        written=10,
        temp_tsv_path=str(tsv_path),
    )
    aggregation = Aggregation()

    region_pipeline._add_region_result_to_aggregation(
        aggregation,
        2,
        result,
        include_tsv=True,
    )
    region_pipeline._add_region_result_to_aggregation(
        aggregation,
        3,
        result,
        include_tsv=False,
    )

    assert aggregation.calls == [
        ((2, result), {"include_tsv": True}),
        ((3, result), {}),
    ]


def test_collect_region_results_aggregates_and_reports_progress(monkeypatch, tmp_path):
    class Future:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class Executor:
        def __init__(self, values):
            self.values = list(values)
            self.submitted = []

        def submit(self, worker, item):
            self.submitted.append((worker, item))
            return Future(self.values.pop(0))

    tsv_path = tmp_path / "region_0.tsv"
    tsv_path.write_text("read\tposterior\n")
    executor = Executor([
        region_pipeline.RegionBamResult(
            "region_0.bam",
            total_reads=10,
            reads_with_footprints=4,
            written=10,
            temp_tsv_path=str(tsv_path),
            skip_reasons={"low_mapq": 2},
        ),
        ("region_1.bam", 5, 1, 5, None, {}),
    ])
    ready_calls = []
    progress_calls = []

    monkeypatch.setattr(region_pipeline, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(
        region_pipeline,
        "_report_workers_ready_once",
        lambda first, pool_start, message: ready_calls.append(
            (first, pool_start, message)
        ) or "ready",
    )
    monkeypatch.setattr(
        region_pipeline,
        "_print_region_progress",
        lambda *args, **kwargs: progress_calls.append((args, kwargs)),
    )
    aggregation = region_pipeline.RegionBamAggregation()
    worker = object()

    region_pipeline._collect_region_results(
        executor,
        worker,
        ["item0", "item1"],
        aggregation,
        region_pipeline.RegionBamResult,
        total_regions=2,
        start_time=10.0,
        pool_start=9.0,
        ready_message="Processing regions...",
        include_tsv=True,
        progress_kwargs={"footprint_label": "With FP"},
    )

    assert executor.submitted == [(worker, "item0"), (worker, "item1")]
    assert aggregation.total_reads == 15
    assert aggregation.reads_with_footprints == 5
    assert aggregation.total_skipped == 2
    assert aggregation.skip_reasons == {"low_mapq": 2}
    assert aggregation.temp_bams == [(0, "region_0.bam"), (1, "region_1.bam")]
    assert aggregation.temp_tsvs == [(0, str(tsv_path))]
    assert ready_calls == [
        (None, 9.0, "Processing regions..."),
        ("ready", 9.0, "Processing regions..."),
    ]
    assert progress_calls == [
        ((aggregation, 2, 10.0), {"footprint_label": "With FP"}),
        ((aggregation, 2, 10.0), {"footprint_label": "With FP"}),
    ]


def test_collect_region_results_reports_error_prefix(monkeypatch, capsys):
    class Executor:
        def submit(self, worker, item):
            return _FailingFuture()

    monkeypatch.setattr(region_pipeline, "as_completed", lambda futures: list(futures))

    with pytest.raises(RuntimeError, match="worker failed"):
        region_pipeline._collect_region_results(
            Executor(),
            object(),
            ["item0"],
            region_pipeline.RegionBamAggregation(),
            region_pipeline.RegionBamResult,
            total_regions=1,
            start_time=10.0,
            pool_start=9.0,
            ready_message="Processing regions...",
            error_prefix="Error processing region",
        )

    assert "Error processing region: worker failed" in capsys.readouterr().out


def test_run_region_worker_pool_wires_executor_and_collector(monkeypatch, capsys):
    class Executor:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            Executor.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    collect_calls = []
    aggregation = region_pipeline.RegionBamAggregation()
    worker = object()
    initializer = object()
    work_items = ["item0"]
    progress_kwargs = {"footprint_label": "With FP"}

    monkeypatch.setattr(region_pipeline, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(region_pipeline.time, "time", lambda: 12.5)
    monkeypatch.setattr(
        region_pipeline,
        "_collect_region_results",
        lambda *args, **kwargs: collect_calls.append((args, kwargs)),
    )

    region_pipeline._run_region_worker_pool(
        n_cores=2,
        initializer=initializer,
        initargs=("model.json", {"param": True}),
        worker=worker,
        work_items=work_items,
        aggregation=aggregation,
        result_type=region_pipeline.RegionBamResult,
        total_regions=1,
        start_time=10.0,
        init_message="init workers",
        ready_message="Processing...",
        include_tsv=True,
        progress_kwargs=progress_kwargs,
        error_prefix="Error processing region",
    )

    assert "init workers" in capsys.readouterr().out
    assert Executor.instances[0].kwargs == {
        "max_workers": 2,
        "mp_context": region_pipeline._MP_CONTEXT,
        "initializer": initializer,
        "initargs": ("model.json", {"param": True}),
    }
    assert collect_calls == [(
        (
            Executor.instances[0],
            worker,
            work_items,
            aggregation,
            region_pipeline.RegionBamResult,
            1,
            10.0,
            12.5,
            "Processing...",
        ),
        {
            "include_tsv": True,
            "progress_kwargs": progress_kwargs,
            "error_prefix": "Error processing region",
        },
    )]


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


def test_prepare_region_parallel_run_indexes_reports_and_makes_temp(
    monkeypatch,
    tmp_path,
    capsys,
):
    calls = []
    regions = [("chr1", 0, 100), ("chr2", 0, 50)]

    monkeypatch.setattr(
        region_pipeline,
        "ensure_bam_index",
        lambda *args: calls.append(("index", args)),
    )
    monkeypatch.setattr(
        region_pipeline,
        "_get_genome_regions",
        lambda *args: calls.append(("regions", args)) or regions,
    )
    monkeypatch.setattr(
        region_pipeline,
        "_make_output_temp_dir",
        lambda *args: calls.append(("temp", args)) or str(tmp_path / "tmp"),
    )

    got_regions, temp_dir = region_pipeline._prepare_region_parallel_run(
        "input.bam",
        "out.bam",
        region_size=1000,
        skip_scaffolds=True,
        chroms={"chr1"},
        n_cores=3,
        temp_prefix=".fiberhmm_tmp_",
        output_posteriors="post.h5",
    )

    assert got_regions is regions
    assert temp_dir == str(tmp_path / "tmp")
    assert calls == [
        (
            "index",
            ("input.bam", "Indexing input BAM for region-parallel processing..."),
        ),
        ("regions", ("input.bam", 1000, True, {"chr1"})),
        ("temp", ("out.bam", ".fiberhmm_tmp_")),
    ]
    out = capsys.readouterr().out
    assert "Processing 2 regions with 3 cores..." in out
    assert "Posteriors will be written to: post.h5" in out


def test_prepare_region_parallel_run_can_skip_index(monkeypatch):
    monkeypatch.setattr(
        region_pipeline,
        "ensure_bam_index",
        lambda *args: (_ for _ in ()).throw(AssertionError("indexed")),
    )
    monkeypatch.setattr(
        region_pipeline,
        "_get_genome_regions",
        lambda *args: [("chr1", 0, 100)],
    )
    monkeypatch.setattr(region_pipeline, "_make_output_temp_dir", lambda *args: "tmp")

    regions, temp_dir = region_pipeline._prepare_region_parallel_run(
        "input.bam",
        "out.bam",
        region_size=1000,
        skip_scaffolds=False,
        chroms=None,
        n_cores=1,
        temp_prefix=".fiberhmm_call_tmp_",
        output_label=" (fused apply+recall)",
        ensure_index=False,
    )

    assert regions == [("chr1", 0, 100)]
    assert temp_dir == "tmp"


def test_region_completion_result_prints_summary(monkeypatch, capsys):
    aggregation = region_pipeline.RegionBamAggregation(
        total_reads=10,
        reads_with_footprints=4,
    )
    monkeypatch.setattr(region_pipeline.time, "time", lambda: 14.0)

    assert region_pipeline._region_completion_result(aggregation, 10.0) == (10, 4)
    assert (
        "Completed: 10 reads | 4 with footprints | 2.5 reads/s | 4.0s"
        in capsys.readouterr().out
    )


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


def test_finalize_region_bam_parallel_run_finishes_outputs_and_posteriors(
    monkeypatch,
    capsys,
):
    aggregation = region_pipeline.RegionBamAggregation(
        total_reads=10,
        reads_with_footprints=4,
        temp_bams=[(1, "region_1.bam")],
        temp_tsvs=[(1, "region_1.tsv.gz")],
    )
    calls = []

    monkeypatch.setattr(
        region_pipeline,
        "_print_skip_reasons_summary",
        lambda got_aggregation: calls.append(("skip", got_aggregation)),
    )
    monkeypatch.setattr(
        region_pipeline,
        "_ordered_existing_temp_paths",
        lambda paths: calls.append(("ordered", paths)) or ["region_1.bam"],
    )
    monkeypatch.setattr(
        region_pipeline,
        "_finalize_region_bam_output",
        lambda *args: calls.append(("finalize", args)),
    )
    monkeypatch.setattr(
        region_pipeline,
        "_merge_region_posterior_outputs",
        lambda *args: calls.append(("merge", args)),
    )
    monkeypatch.setattr(
        region_pipeline,
        "_region_completion_result",
        lambda got_aggregation, start_time: (
            calls.append(("complete", got_aggregation, start_time)) or (10, 4)
        ),
    )

    assert region_pipeline._finalize_region_bam_parallel_run(
        input_bam="in.bam",
        output_bam="out.bam",
        temp_dir="tmp",
        aggregation=aggregation,
        start_time=5.0,
        n_cores=3,
        return_posteriors=True,
        output_posteriors="post.h5",
        mode="daf",
        context_size=5,
        edge_trim=11,
    ) == (10, 4)

    assert calls == [
        ("skip", aggregation),
        ("ordered", aggregation.temp_bams),
        ("finalize", ("in.bam", "out.bam", ["region_1.bam"], "tmp", 3)),
        (
            "merge",
            (
                aggregation.temp_tsvs,
                "post.h5",
                "daf",
                5,
                11,
                "in.bam",
            ),
        ),
        ("complete", aggregation, 5.0),
    ]
    assert capsys.readouterr().out == "\n"


def test_finalize_region_bed_parallel_run_concatenates_and_completes(
    monkeypatch,
    capsys,
):
    aggregation = region_pipeline.RegionBedAggregation(
        total_reads=8,
        reads_with_footprints=3,
        temp_beds=[(0, "region_0.bed")],
    )
    calls = []

    monkeypatch.setattr(
        region_pipeline,
        "_ordered_existing_temp_paths",
        lambda paths: calls.append(("ordered", paths)) or ["region_0.bed"],
    )
    monkeypatch.setattr(
        region_pipeline,
        "_concatenate_region_beds",
        lambda output_bed, beds: calls.append(("concat", output_bed, beds)),
    )
    monkeypatch.setattr(
        region_pipeline,
        "_region_completion_result",
        lambda got_aggregation, start_time: (
            calls.append(("complete", got_aggregation, start_time)) or (8, 3)
        ),
    )

    assert region_pipeline._finalize_region_bed_parallel_run(
        "out.bed",
        aggregation,
        7.0,
    ) == (8, 3)

    assert calls == [
        ("ordered", aggregation.temp_beds),
        ("concat", "out.bed", ["region_0.bed"]),
        ("complete", aggregation, 7.0),
    ]
    assert capsys.readouterr().out == "\n"


def test_run_region_bam_worker_pool_wires_legacy_contract(monkeypatch):
    calls = []

    def fake_run_region_worker_pool(**kwargs):
        calls.append(kwargs)
        kwargs["aggregation"].total_reads = 12
        kwargs["aggregation"].reads_with_footprints = 5

    monkeypatch.setattr(
        region_pipeline,
        "_run_region_worker_pool",
        fake_run_region_worker_pool,
    )

    aggregation = region_pipeline._run_region_bam_worker_pool(
        n_cores=3,
        model_path="model.json",
        params={"mode": "daf"},
        work_items=["region"],
        total_regions=1,
        start_time=10.0,
        include_tsv=True,
    )

    assert aggregation.total_reads == 12
    assert aggregation.reads_with_footprints == 5
    kwargs = calls[0]
    assert kwargs["n_cores"] == 3
    assert kwargs["initializer"] is region_pipeline._init_region_worker
    assert kwargs["initargs"] == ("model.json", {"mode": "daf"})
    assert kwargs["worker"] is region_pipeline._process_region_to_bam
    assert kwargs["work_items"] == ["region"]
    assert kwargs["result_type"] is region_pipeline.RegionBamResult
    assert kwargs["total_regions"] == 1
    assert kwargs["start_time"] == 10.0
    assert kwargs["ready_message"] == "Processing regions..."
    assert kwargs["include_tsv"] is True
    assert kwargs["error_prefix"] == "Error processing region"


def test_run_region_bed_worker_pool_wires_bed_contract(monkeypatch):
    calls = []

    def fake_run_region_worker_pool(**kwargs):
        calls.append(kwargs)
        kwargs["aggregation"].total_reads = 8
        kwargs["aggregation"].reads_with_footprints = 3

    monkeypatch.setattr(
        region_pipeline,
        "_run_region_worker_pool",
        fake_run_region_worker_pool,
    )

    aggregation = region_pipeline._run_region_bed_worker_pool(
        n_cores=2,
        model_path="model.json",
        params={"mode": "m6a"},
        work_items=["region"],
        total_regions=1,
        start_time=10.0,
    )

    assert aggregation.total_reads == 8
    assert aggregation.reads_with_footprints == 3
    kwargs = calls[0]
    assert kwargs["n_cores"] == 2
    assert kwargs["initializer"] is region_pipeline._init_region_worker
    assert kwargs["initargs"] == ("model.json", {"mode": "m6a"})
    assert kwargs["worker"] is region_pipeline._process_region_to_bed
    assert kwargs["work_items"] == ["region"]
    assert kwargs["result_type"] is region_pipeline.RegionBedResult
    assert kwargs["total_regions"] == 1
    assert kwargs["start_time"] == 10.0
    assert kwargs["ready_message"] == "Processing regions..."
    assert kwargs["error_prefix"] == "Error processing region"


def test_run_fused_region_bam_worker_pool_wires_fused_contract(monkeypatch):
    calls = []

    def fake_run_region_worker_pool(**kwargs):
        calls.append(kwargs)
        kwargs["aggregation"].total_reads = 12
        kwargs["aggregation"].reads_with_footprints = 5

    monkeypatch.setattr(
        region_pipeline,
        "_run_region_worker_pool",
        fake_run_region_worker_pool,
    )

    aggregation = region_pipeline._run_fused_region_bam_worker_pool(
        n_cores=3,
        apply_model_path="apply.json",
        recall_model_path="recall.json",
        emission_uplift=1.5,
        params={"mode": "daf"},
        work_items=["region"],
        total_regions=1,
        start_time=10.0,
    )

    assert aggregation.total_reads == 12
    assert aggregation.reads_with_footprints == 5
    kwargs = calls[0]
    assert kwargs["n_cores"] == 3
    assert kwargs["initializer"] is region_pipeline._init_fused_region_worker
    assert kwargs["initargs"] == (
        "apply.json",
        "recall.json",
        1.5,
        {"mode": "daf"},
    )
    assert kwargs["worker"] is region_pipeline._process_region_to_bam_fused
    assert kwargs["work_items"] == ["region"]
    assert kwargs["result_type"] is region_pipeline.RegionBamResult
    assert kwargs["total_regions"] == 1
    assert kwargs["start_time"] == 10.0
    assert kwargs["ready_message"] == "Processing..."
    assert kwargs["progress_kwargs"] == {
        "footprint_label": "With FP",
        "rate_unit": "r/s",
        "rate_precision": 0,
    }


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


def test_finalize_fused_region_bam_output_concats_indexes_and_reports(
    monkeypatch,
    capsys,
):
    calls = []
    aggregation = region_pipeline.RegionBamAggregation(
        total_reads=10,
        reads_with_footprints=4,
        temp_bams=[(1, "late.bam"), (0, "early.bam")],
    )
    aggregation.total_skipped = 2
    aggregation.skip_reasons = {"low_mapq": 2}

    monkeypatch.setattr(
        region_pipeline,
        "_ordered_existing_temp_paths",
        lambda paths: calls.append(("ordered", paths)) or ["early.bam", "late.bam"],
    )
    monkeypatch.setattr(
        region_pipeline,
        "_concatenate_region_bams",
        lambda *args: calls.append(("concat", args)),
    )
    monkeypatch.setattr(
        region_pipeline.pysam,
        "index",
        lambda output_bam: calls.append(("index", output_bam)),
    )
    monkeypatch.setattr(region_pipeline.time, "time", lambda: 14.0)

    region_pipeline._finalize_fused_region_bam_output(
        "input.bam",
        "out.bam",
        "tmp",
        aggregation,
        start_time=10.0,
    )

    assert calls == [
        ("ordered", [(1, "late.bam"), (0, "early.bam")]),
        ("concat", ("input.bam", "out.bam", ["early.bam", "late.bam"], "tmp")),
        ("index", "out.bam"),
    ]
    out = capsys.readouterr().out
    assert "Skipped: 2" in out
    assert "low_mapq: 2 (16.7%)" in out
    assert "Total: 10 reads, 4 with footprints, 2.5 r/s" in out


def test_concatenate_region_beds_writes_inputs_in_order(tmp_path, capsys):
    bed_a = tmp_path / "a.bed"
    bed_b = tmp_path / "b.bed"
    output = tmp_path / "out.bed"
    bed_a.write_bytes(b"chr1\t0\t10\n")
    bed_b.write_bytes(b"chr1\t10\t20\n")

    region_pipeline._concatenate_region_beds(
        str(output), [str(bed_a), str(bed_b)],
    )

    assert output.read_bytes() == b"chr1\t0\t10\nchr1\t10\t20\n"
    assert "Concatenating 2 region BED files..." in capsys.readouterr().out


def test_concatenate_region_beds_preserves_output_on_copy_failure(tmp_path):
    bed_a = tmp_path / "a.bed"
    missing = tmp_path / "missing.bed"
    output = tmp_path / "out.bed"
    temp_output = tmp_path / "out.bed.tmp"
    bed_a.write_bytes(b"chr1\t0\t10\n")
    output.write_bytes(b"old\n")

    with pytest.raises(FileNotFoundError):
        region_pipeline._concatenate_region_beds(
            str(output), [str(bed_a), str(missing)],
        )

    assert output.read_bytes() == b"old\n"
    assert not temp_output.exists()


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


def test_fused_region_parallel_bam_cleans_temp_dir_on_setup_failure(
    monkeypatch,
    tmp_path,
):
    temp_dirs = _install_failing_region_pool(monkeypatch, tmp_path)
    input_bam = _indexed_input_bam(tmp_path)

    def fail_params(**kwargs):
        raise RuntimeError("setup failed")

    monkeypatch.setattr(region_pipeline, "_fused_region_worker_params", fail_params)

    with pytest.raises(RuntimeError, match="setup failed"):
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
