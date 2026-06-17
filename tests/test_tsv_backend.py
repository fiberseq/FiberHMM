import binascii
import gzip

import h5py
import numpy as np
import pytest

from fiberhmm.posteriors.region_tsv import (
    REGION_POSTERIORS_HEADER,
    format_region_posterior_line,
)
from fiberhmm.posteriors import tsv_backend


def _h5_text(value):
    if isinstance(value, bytes):
        return value.decode()
    return value


def test_posterior_tsv_output_path_respects_compression_flag():
    assert tsv_backend._posterior_tsv_output_path(
        "out.tsv", compress=False,
    ) == "out.tsv"
    assert tsv_backend._posterior_tsv_output_path(
        "out.tsv", compress=True,
    ) == "out.tsv.gz"
    assert tsv_backend._posterior_tsv_output_path(
        "out.tsv.gz", compress=False,
    ) == "out.tsv.gz"


class _TrackingHandle:
    def __init__(self, inner=None, *, fail_during_iteration: bool = False):
        self._inner = inner
        self._fail_during_iteration = fail_during_iteration
        self.closed = False
        self.writes = []

    def __enter__(self):
        if self._inner is not None:
            self._inner.__enter__()
        return self

    def __exit__(self, *args):
        self.close()
        return False

    def __iter__(self):
        if self._fail_during_iteration:
            yield "#metadata:{}\n"
            raise RuntimeError("read failed")
        yield from self._inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def write(self, text):
        self.writes.append(text)

    def close(self):
        if not self.closed and self._inner is not None:
            self._inner.close()
        self.closed = True


def test_tsv_to_h5_closes_gzip_inputs_when_conversion_fails(monkeypatch, tmp_path):
    tsv_path = tmp_path / "bad.tsv.gz"
    with gzip.open(tsv_path, "wt") as handle:
        handle.write("#metadata:{}\n")
        handle.write("#read_id\tchrom\tstart\tend\tstrand\tposteriors_b64\tfp_starts\tfp_sizes\n")
        handle.write("read1\tchr1\t0\t3\t+\tabc\t\t\n")

    real_gzip_open = gzip.open
    handles = []

    def tracking_gzip_open(*args, **kwargs):
        handle = _TrackingHandle(real_gzip_open(*args, **kwargs))
        handles.append(handle)
        return handle

    monkeypatch.setattr(tsv_backend.gzip, "open", tracking_gzip_open)

    with pytest.raises(binascii.Error):
        tsv_backend.tsv_to_h5(str(tsv_path), str(tmp_path / "bad.h5"), verbose=False)

    assert len(handles) == 2
    assert all(handle.closed for handle in handles)


def test_tsv_writer_reuses_region_posterior_row_format(tmp_path):
    output_path = tmp_path / "posteriors.tsv"
    writer = tsv_backend.PosteriorsTSVWriter(
        str(output_path),
        mode="pacbio-fiber",
        context_size=3,
        edge_trim=10,
        source_bam="input.bam",
        compress=False,
    )
    writer.write_fiber(
        read_id="read1",
        chrom="chr1",
        start=10,
        end=20,
        strand="+",
        posteriors=np.array([0.0, 0.5, 1.0], dtype=np.float32),
        fp_starts=np.array([12], dtype=np.int32),
        fp_sizes=np.array([4], dtype=np.int32),
    )
    assert writer.close() == 1

    lines = output_path.read_text().splitlines(keepends=True)
    assert lines[1] == REGION_POSTERIORS_HEADER
    assert lines[2] == format_region_posterior_line(
        read_name="read1",
        chrom="chr1",
        ref_start=10,
        ref_end=20,
        strand="+",
        posteriors=np.array([0.0, 0.5, 1.0], dtype=np.float32),
        footprint_starts=np.array([12], dtype=np.int32),
        footprint_sizes=np.array([4], dtype=np.int32),
    )


def test_scan_tsv_for_h5_collects_metadata_and_chrom_counts(tmp_path):
    tsv_path = tmp_path / "posteriors.tsv"
    tsv_path.write_text(
        '#metadata:{"mode":"daf","context_size":5,"edge_trim":12}\n'
        + REGION_POSTERIORS_HEADER
        + "read1\tchr2L\t0\t3\t+\tabc\t\t\n"
        + "read2\tchr2L\t4\t8\t-\tdef\t\t\n"
        + "read3\tchr3R\t10\t14\t+\tghi\t\t\n",
        encoding="utf-8",
    )

    metadata, chrom_counts, n_total = tsv_backend._scan_tsv_for_h5(
        str(tsv_path),
        verbose=False,
    )

    assert metadata == {"mode": "daf", "context_size": 5, "edge_trim": 12}
    assert chrom_counts == {"chr2L": 2, "chr3R": 1}
    assert n_total == 3


def test_posterior_record_from_fields_decodes_with_requested_dtype():
    line = format_region_posterior_line(
        read_name="read1",
        chrom="chr1",
        ref_start=10,
        ref_end=13,
        strand="-",
        posteriors=np.array([0.0, 0.5, 1.0], dtype=np.float32),
        footprint_starts=np.array([10, 12], dtype=np.int32),
        footprint_sizes=np.array([1, 2], dtype=np.int32),
    )

    fields = tsv_backend._split_posteriors_line(line)
    record = tsv_backend._posterior_record_from_fields(fields, np.float16)

    assert record["read_id"] == "read1"
    assert record["chrom"] == "chr1"
    assert record["start"] == 10
    assert record["end"] == 13
    assert record["strand"] == "-"
    assert record["posteriors"].dtype == np.float16
    np.testing.assert_allclose(
        record["posteriors"],
        [0.0, 127 / 255, 1.0],
        atol=1e-3,
    )
    np.testing.assert_array_equal(record["fp_starts"], [10, 12])
    np.testing.assert_array_equal(record["fp_sizes"], [1, 2])


def test_posterior_tsv_metadata_uses_source_bam_basename():
    assert tsv_backend._posterior_tsv_metadata(
        mode="daf",
        context_size=5,
        edge_trim=12,
        source_bam="/path/to/input.bam",
    ) == {
        "mode": "daf",
        "context_size": 5,
        "edge_trim": 12,
        "source_bam": "input.bam",
        "format_version": 1,
    }


def test_tsv_to_h5_writes_metadata_and_arrays(tmp_path):
    tsv_path = tmp_path / "posteriors.tsv"
    h5_path = tmp_path / "posteriors.h5"

    with tsv_backend.PosteriorsTSVWriter(
        str(tsv_path),
        mode="daf",
        context_size=5,
        edge_trim=12,
        source_bam="/path/input.bam",
        compress=False,
    ) as writer:
        writer.write_fiber(
            read_id="read1",
            chrom="chr1",
            start=10,
            end=13,
            strand="+",
            posteriors=np.array([0.0, 0.5, 1.0], dtype=np.float32),
            fp_starts=np.array([10], dtype=np.int32),
            fp_sizes=np.array([3], dtype=np.int32),
        )

    assert tsv_backend.tsv_to_h5(str(tsv_path), str(h5_path), verbose=False) == 1

    with h5py.File(h5_path, "r") as h5:
        assert h5.attrs["mode"] == "daf"
        assert h5.attrs["context_size"] == 5
        assert h5.attrs["edge_trim"] == 12
        assert h5.attrs["source_bam"] == "input.bam"
        assert h5["chr1"].attrs["n_fibers"] == 1
        assert _h5_text(h5["chr1"]["fiber_ids"][0]) == "read1"
        assert h5["chr1"]["fiber_starts"][0] == 10
        assert h5["chr1"]["fiber_ends"][0] == 13
        assert _h5_text(h5["chr1"]["strands"][0]) == "+"
        np.testing.assert_allclose(
            h5["chr1"]["posteriors"]["0"][:],
            np.array([0.0, 0.5, 1.0], dtype=np.float16),
            atol=1 / 255,
        )
        np.testing.assert_array_equal(
            h5["chr1"]["footprint_starts"]["0"][:],
            np.array([10], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            h5["chr1"]["footprint_sizes"]["0"][:],
            np.array([3], dtype=np.int32),
        )


def test_concatenate_tsvs_closes_input_and_output_when_read_fails(monkeypatch, tmp_path):
    input_path = tmp_path / "input.tsv"
    input_path.write_text("", encoding="utf-8")
    output_path = tmp_path / "output.tsv"

    input_handle = _TrackingHandle(fail_during_iteration=True)
    output_handle = _TrackingHandle()

    def fake_open_text_file(path, mode):
        if path == str(output_path) and mode == "wt":
            return output_handle
        if path == str(input_path) and mode == "rt":
            return input_handle
        raise AssertionError(f"unexpected open: {path} {mode}")

    monkeypatch.setattr(tsv_backend, "_open_text_file", fake_open_text_file)

    with pytest.raises(RuntimeError, match="read failed"):
        tsv_backend.concatenate_tsvs([str(input_path)], str(output_path))

    assert input_handle.closed
    assert output_handle.closed
