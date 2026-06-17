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
