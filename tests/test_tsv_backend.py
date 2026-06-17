import binascii
import gzip

import numpy as np
import pytest

from fiberhmm.posteriors.region_tsv import (
    REGION_POSTERIORS_HEADER,
    format_region_posterior_line,
)
from fiberhmm.posteriors import tsv_backend


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
