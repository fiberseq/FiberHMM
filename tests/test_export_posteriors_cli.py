"""Failure-path coverage for posterior export CLI helpers."""

from types import SimpleNamespace

import numpy as np
import pytest

from fiberhmm.cli import export_posteriors
from fiberhmm.posteriors import tsv_backend


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
