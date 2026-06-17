"""Failure-path coverage for posterior export CLI helpers."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from fiberhmm.cli import export_posteriors
from fiberhmm.posteriors import hdf5_backend, tsv_backend


def test_export_posteriors_detects_output_format():
    assert export_posteriors._detect_format("out.h5", "auto") == "hdf5"
    assert export_posteriors._detect_format("out.hdf5", "auto") == "hdf5"
    assert export_posteriors._detect_format("out.tsv.gz", "auto") == "tsv"
    assert export_posteriors._detect_format("out.any", "hdf5") == "hdf5"


def test_export_posteriors_model_resolution_uses_custom_path():
    args = SimpleNamespace(model="/tmp/custom.json", enzyme=None, seq=None)

    assert export_posteriors._resolve_model_path(args) == "/tmp/custom.json"


def test_export_posteriors_model_resolution_requires_model_or_enzyme(capsys):
    args = SimpleNamespace(model=None, enzyme=None, seq=None)

    with pytest.raises(SystemExit) as exc:
        export_posteriors._resolve_model_path(args)

    assert exc.value.code == 1
    assert "one of --model or --enzyme must be provided" in capsys.readouterr().err


def test_export_posteriors_chroms_set():
    assert export_posteriors._chroms_set(None) is None
    assert export_posteriors._chroms_set([]) is None
    assert export_posteriors._chroms_set(["chr2", "chr1", "chr2"]) == {"chr1", "chr2"}


def test_export_posteriors_regions_by_chrom_preserves_order():
    regions = [
        ("chr2", 0, 10),
        ("chr1", 5, 15),
        ("chr2", 10, 20),
    ]

    assert export_posteriors._regions_by_chrom(regions) == {
        "chr2": [(0, 10), (10, 20)],
        "chr1": [(5, 15)],
    }


def test_footprint_reference_intervals_clamps_and_skips_invalid_positions():
    starts, sizes = export_posteriors._footprint_reference_intervals(
        fp_start_idx=np.array([0, 2, 3]),
        fp_end_idx=np.array([2, 4, 10]),
        ref_positions=np.array([100, 101, -1, 103, 104], dtype=np.int32),
    )

    np.testing.assert_array_equal(starts, np.array([100, 103], dtype=np.int32))
    np.testing.assert_array_equal(sizes, np.array([1, 1], dtype=np.int32))


def test_modified_base_positions_forward_filters_quality_and_parser_errors():
    read = SimpleNamespace(
        modified_bases_forward={
            ("A", 0, "a"): [(1, 127), (2, 128)],
            ("C", 0, "m"): [(3, 255)],
        }
    )

    assert export_posteriors._modified_base_positions_forward(read) == {2, 3}
    assert export_posteriors._modified_base_positions_forward(read, min_qual=255) == {3}

    class BadRead:
        @property
        def modified_bases_forward(self):
            raise ValueError("bad MM/ML")

    assert export_posteriors._modified_base_positions_forward(BadRead()) == set()


def test_posterior_read_strand_uses_mode_policy(monkeypatch):
    monkeypatch.setattr(
        export_posteriors,
        "detect_daf_strand",
        lambda sequence, mod_positions: "ct",
    )

    assert export_posteriors._posterior_read_strand(
        "daf", "ACGT", {1}, is_reverse=False,
    ) == "ct"
    assert export_posteriors._posterior_read_strand(
        "pacbio-fiber", "ACGT", {1}, is_reverse=True,
    ) == "-"
    assert export_posteriors._posterior_read_strand(
        "pacbio-fiber", "ACGT", {1}, is_reverse=False,
    ) == "+"


def test_posterior_result_record_preserves_metadata_and_arrays():
    read = SimpleNamespace(query_name="read-a", reference_start=10, reference_end=30)
    posteriors = np.array([0.1, 0.2], dtype=np.float16)
    ref_positions = np.array([10, 11], dtype=np.int32)
    starts = np.array([10], dtype=np.int32)
    sizes = np.array([2], dtype=np.int32)

    record = export_posteriors._posterior_result_record(
        read, "+", posteriors, ref_positions, starts, sizes,
    )

    assert record["read_name"] == "read-a"
    assert record["ref_start"] == 10
    assert record["ref_end"] == 30
    assert record["strand"] == "+"
    assert record["posteriors"] is posteriors
    assert record["ref_positions"] is ref_positions
    assert record["footprint_starts"] is starts
    assert record["footprint_sizes"] is sizes


def test_posterior_sequence_or_none_applies_min_length_filter():
    assert export_posteriors._posterior_sequence_or_none(
        SimpleNamespace(query_sequence=None),
    ) is None
    assert export_posteriors._posterior_sequence_or_none(
        SimpleNamespace(query_sequence="A" * 99),
    ) is None
    assert export_posteriors._posterior_sequence_or_none(
        SimpleNamespace(query_sequence="A" * 100),
    ) == "A" * 100


def test_h5_batch_metadata_helpers_append_and_concatenate():
    ids, starts, ends, strands = export_posteriors._h5_batch_metadata([
        {"read_name": "read-a", "ref_start": 10, "ref_end": 15, "strand": "+"},
        {"read_name": "read-b", "ref_start": 20, "ref_end": 25, "strand": "-"},
    ])

    assert ids == ["read-a", "read-b"]
    assert strands == ["+", "-"]
    np.testing.assert_array_equal(starts, np.array([10, 20], dtype=np.int32))
    np.testing.assert_array_equal(ends, np.array([15, 25], dtype=np.int32))

    meta = {"ids": [], "starts": [], "ends": [], "strands": []}

    export_posteriors._append_h5_batch_metadata(
        meta,
        ids=["read-a", "read-b"],
        starts=np.array([10, 20], dtype=np.int32),
        ends=np.array([15, 25], dtype=np.int32),
        strands=["+", "-"],
    )

    assert meta["ids"] == ["read-a", "read-b"]
    assert meta["strands"] == ["+", "-"]
    np.testing.assert_array_equal(
        export_posteriors._concat_h5_metadata_arrays(meta["starts"]),
        np.array([10, 20], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        export_posteriors._concat_h5_metadata_arrays([]),
        np.array([], dtype=np.int32),
    )


def test_decode_h5_text_accepts_bytes_and_strings():
    assert export_posteriors._decode_h5_text(b"read-a") == "read-a"
    assert export_posteriors._decode_h5_text("read-b") == "read-b"


def test_h5_indexed_array_reads_present_group_or_returns_default(tmp_path):
    default = object()

    with h5py.File(tmp_path / "arrays.h5", "w") as h5:
        group = h5.create_group("values")
        group.create_dataset("2", data=np.array([1, 2], dtype=np.int32))

        np.testing.assert_array_equal(
            export_posteriors._h5_indexed_array(group, 2, default),
            np.array([1, 2], dtype=np.int32),
        )
        assert export_posteriors._h5_indexed_array(None, 2, default) is default


def test_fiber_region_index_helpers_select_expected_records():
    starts = np.array([0, 10, 20, 30], dtype=np.int32)
    ends = np.array([9, 25, 40, 50], dtype=np.int32)

    np.testing.assert_array_equal(
        export_posteriors._fiber_overlap_indices(
            starts,
            ends,
            start=15,
            end=35,
            min_overlap=1,
        ),
        np.array([1, 2, 3]),
    )
    np.testing.assert_array_equal(
        export_posteriors._fiber_spanning_indices(
            starts,
            ends,
            start=15,
            end=35,
        ),
        np.array([], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        export_posteriors._fiber_spanning_indices(
            starts,
            ends,
            start=22,
            end=24,
        ),
        np.array([1, 2], dtype=np.int64),
    )


def test_write_h5_fiber_arrays_uses_backend_dataset_specs(tmp_path):
    with h5py.File(tmp_path / "posteriors.h5", "w") as h5:
        grp = hdf5_backend.create_posterior_chrom_group(h5, "chr1")

        export_posteriors._write_h5_fiber_arrays(
            grp,
            4,
            {
                "posteriors": np.array([0.1, 0.9], dtype=np.float32),
                "ref_positions": [10, -1],
                "footprint_starts": [3],
                "footprint_sizes": [20],
            },
        )

        assert grp["posteriors"]["4"].dtype == np.float16
        np.testing.assert_allclose(grp["posteriors"]["4"][:], [0.1, 0.9], atol=1e-3)
        np.testing.assert_array_equal(grp["ref_positions"]["4"][:], [10, -1])
        np.testing.assert_array_equal(grp["footprint_starts"]["4"][:], [3])
        np.testing.assert_array_equal(grp["footprint_sizes"]["4"][:], [20])


def test_flush_h5_chrom_buffer_writes_and_clears(monkeypatch):
    h5_file = {"chr1": object()}
    buffers = {"chr1": [{"read_name": "a"}, {"read_name": "b"}]}
    counts = {"chr1": 5}
    metadata = {"chr1": {"ids": [], "starts": [], "ends": [], "strands": []}}
    calls = []

    def fake_write_batch(grp, fibers, start_idx):
        calls.append((grp, list(fibers), start_idx))
        return (
            ["read-a", "read-b"],
            np.array([10, 20], dtype=np.int32),
            np.array([15, 25], dtype=np.int32),
            ["+", "-"],
        )

    monkeypatch.setattr(export_posteriors, "_write_batch_to_h5", fake_write_batch)

    assert export_posteriors._flush_h5_chrom_buffer(
        h5_file, "chr1", buffers, counts, metadata,
    ) == 2

    assert calls == [(h5_file["chr1"], [{"read_name": "a"}, {"read_name": "b"}], 5)]
    assert buffers["chr1"] == []
    assert counts["chr1"] == 7
    assert metadata["chr1"]["ids"] == ["read-a", "read-b"]
    assert metadata["chr1"]["strands"] == ["+", "-"]
    np.testing.assert_array_equal(
        export_posteriors._concat_h5_metadata_arrays(metadata["chr1"]["starts"]),
        np.array([10, 20], dtype=np.int32),
    )


def test_flush_h5_chrom_buffer_skips_empty_buffer(monkeypatch):
    monkeypatch.setattr(
        export_posteriors,
        "_write_batch_to_h5",
        lambda *_: pytest.fail("unexpected write"),
    )

    assert export_posteriors._flush_h5_chrom_buffer(
        {"chr1": object()},
        "chr1",
        {"chr1": []},
        {"chr1": 3},
        {"chr1": {"ids": [], "starts": [], "ends": [], "strands": []}},
    ) == 0


def test_new_h5_export_chrom_state_initializes_parallel_trackers():
    counts, metadata, buffers = export_posteriors._new_h5_export_chrom_state({
        "chr2": [(0, 10), (10, 20)],
        "chr1": [(5, 15)],
    })

    assert counts == {"chr2": 0, "chr1": 0}
    assert metadata == {
        "chr2": {"ids": [], "starts": [], "ends": [], "strands": []},
        "chr1": {"ids": [], "starts": [], "ends": [], "strands": []},
    }
    assert buffers == {"chr2": [], "chr1": []}


def test_h5_region_buffer_flushes_at_batch_size():
    buffers = {"chr1": [{"read_name": "existing"}]}
    flushed = []

    export_posteriors._buffer_h5_region_results(
        "chr1",
        [{"read_name": "new"}],
        buffers,
        write_batch_size=3,
        flush_buffer=flushed.append,
    )

    assert buffers["chr1"] == [{"read_name": "existing"}, {"read_name": "new"}]
    assert flushed == []

    export_posteriors._buffer_h5_region_results(
        "chr1",
        [{"read_name": "third"}],
        buffers,
        write_batch_size=3,
        flush_buffer=flushed.append,
    )

    assert flushed == ["chr1"]


def test_write_h5_chrom_metadata_concatenates_sidecars():
    calls = []
    h5_file = {"chr1": object()}
    chrom_metadata = {
        "chr1": {
            "ids": ["read-a", "read-b"],
            "starts": [
                np.array([10], dtype=np.int32),
                np.array([20], dtype=np.int32),
            ],
            "ends": [
                np.array([15], dtype=np.int32),
                np.array([25], dtype=np.int32),
            ],
            "strands": ["+", "-"],
        }
    }
    chrom_fiber_counts = {"chr1": 2}

    def fake_write_metadata(*args, **kwargs):
        calls.append((args, kwargs))

    export_posteriors._write_h5_chrom_metadata(
        h5_file,
        "chr1",
        chrom_metadata,
        chrom_fiber_counts,
        fake_write_metadata,
    )

    args, kwargs = calls[0]
    assert args[0] is h5_file["chr1"]
    assert args[1] == ["read-a", "read-b"]
    np.testing.assert_array_equal(args[2], np.array([10, 20], dtype=np.int32))
    np.testing.assert_array_equal(args[3], np.array([15, 25], dtype=np.int32))
    assert args[4] == ["+", "-"]
    assert kwargs == {"n_fibers": 2}


def test_submit_next_region_records_pending_future():
    class FakeExecutor:
        def __init__(self):
            self.submitted = []

        def submit(self, fn, args):
            future = object()
            self.submitted.append((future, fn, args))
            return future

    executor = FakeExecutor()
    region_iter = iter([("chr1", 10, 20)])
    pending = {}

    assert export_posteriors._submit_next_region(
        executor,
        region_iter,
        "input.bam",
        pending,
    )
    future, fn, args = executor.submitted[0]
    assert fn is export_posteriors._process_region_worker
    assert args == ("chr1", 10, 20, "input.bam")
    assert pending == {future: ("chr1", 10, 20)}

    assert not export_posteriors._submit_next_region(
        executor,
        region_iter,
        "input.bam",
        pending,
    )


def test_submit_initial_regions_respects_pending_cap():
    class FakeExecutor:
        def __init__(self):
            self.submitted = []

        def submit(self, fn, args):
            future = object()
            self.submitted.append((future, fn, args))
            return future

    executor = FakeExecutor()
    pending = {}

    export_posteriors._submit_initial_regions(
        executor,
        iter([("chr1", 0, 10), ("chr1", 10, 20), ("chr2", 0, 10)]),
        "input.bam",
        pending,
        max_pending=2,
        total_regions=3,
    )

    assert [submitted[2] for submitted in executor.submitted] == [
        ("chr1", 0, 10, "input.bam"),
        ("chr1", 10, 20, "input.bam"),
    ]
    assert list(pending.values()) == [("chr1", 0, 10), ("chr1", 10, 20)]


def test_completed_region_future_helpers_callback_and_update_progress(capsys):
    class Future:
        def __init__(self, done, result=None, error=None):
            self._done = done
            self._result = result
            self._error = error

        def done(self):
            return self._done

        def result(self):
            if self._error:
                raise self._error
            return self._result

    class Progress:
        def __init__(self):
            self.n = 0

        def update(self, amount):
            self.n += amount

    complete = Future(True, ("chr1", 0, 10, [{"read_name": "a"}]))
    waiting = Future(False)
    pending = {complete: ("chr1", 0, 10), waiting: ("chr2", 0, 10)}
    pbar = Progress()
    callbacks = []

    assert export_posteriors._completed_region_futures(pending) == [complete]
    export_posteriors._handle_completed_region_future(
        complete,
        pending,
        lambda chrom, results: callbacks.append((chrom, results)),
        pbar,
    )

    assert pending == {waiting: ("chr2", 0, 10)}
    assert callbacks == [("chr1", [{"read_name": "a"}])]
    assert pbar.n == 1
    assert capsys.readouterr().out == ""

    failing = Future(True, error=RuntimeError("worker failed"))
    pending = {failing: ("chr3", 5, 15)}
    export_posteriors._handle_completed_region_future(
        failing,
        pending,
        lambda *_: pytest.fail("unexpected callback"),
        pbar,
    )

    assert pending == {}
    assert pbar.n == 2
    assert "Error processing ('chr3', 5, 15): worker failed" in capsys.readouterr().out


def test_export_posteriors_tsv_closes_writer_when_region_processing_fails(
    monkeypatch, tmp_path
):
    instances = []

    class FakeTSVWriter:
        def __init__(self, output_path, **kwargs):
            self.output_path = output_path
            self.closed = False
            self.close_count = 0
            instances.append(self)

        def write_fiber(self, **kwargs):
            raise AssertionError("region processing should fail before writes")

        def close(self):
            self.closed = True
            self.close_count += 1
            return 0

    def fail_process_regions(*args, **kwargs):
        raise RuntimeError("region processing failed")

    monkeypatch.setattr(tsv_backend, "PosteriorsTSVWriter", FakeTSVWriter)
    monkeypatch.setattr(
        export_posteriors,
        "load_model_with_metadata",
        lambda *args, **kwargs: (object(), 3, "pacbio-fiber"),
    )
    monkeypatch.setattr(
        export_posteriors,
        "_get_genome_regions",
        lambda *args, **kwargs: [("chr1", 0, 100)],
    )
    monkeypatch.setattr(export_posteriors, "_process_regions", fail_process_regions)

    with pytest.raises(RuntimeError, match="region processing failed"):
        export_posteriors.export_posteriors_tsv(
            input_bam="input.bam",
            model_path="model.json",
            output_path=str(tmp_path / "posteriors.tsv.gz"),
            n_cores=1,
            verbose=False,
        )

    assert instances
    assert instances[0].closed
    assert instances[0].close_count == 1
