"""Tests for region-parallel posterior TSV helpers."""

import gzip
import io
import json

import numpy as np

from fiberhmm.posteriors import region_tsv
from fiberhmm.posteriors.region_tsv import (
    format_region_posterior_line,
    format_region_posterior_line_from_record,
    merge_region_posteriors_tsv,
    region_posteriors_needs_h5_conversion,
    region_posteriors_tsv_output_path,
)
from fiberhmm.posteriors.tsv_backend import parse_posteriors_line


def test_region_posterior_encoding_helpers_clip_and_join_values():
    assert region_tsv._posterior_probabilities_b64(
        np.array([-1.0, 0.5, 2.0], dtype=np.float32),
    ) == "AH//"
    assert region_tsv._comma_join_ints(np.array([2, 8], dtype=np.int32)) == "2,8"
    assert region_tsv._comma_join_ints([]) == ""


def test_region_posterior_fields_preserve_tsv_column_order():
    record = region_tsv._RegionPosteriorLineRecord(
        read_name="read1",
        chrom="chr2",
        ref_start=10,
        ref_end=20,
        strand="+",
        posteriors=np.array([0.0, 1.0], dtype=np.float32),
        footprint_starts=np.array([2], dtype=np.int32),
        footprint_sizes=np.array([3], dtype=np.int32),
    )

    fields = ["read1", "chr2", "10", "20", "+", "AP8=", "2", "3"]
    assert region_tsv._region_posterior_fields_from_record(record) == fields
    assert region_tsv._region_posterior_fields(
        read_name="read1",
        chrom="chr2",
        ref_start=10,
        ref_end=20,
        strand="+",
        posteriors=np.array([0.0, 1.0], dtype=np.float32),
        footprint_starts=np.array([2], dtype=np.int32),
        footprint_sizes=np.array([3], dtype=np.int32),
    ) == fields


def test_format_region_posterior_line_matches_tsv_parser():
    record = region_tsv._RegionPosteriorLineRecord(
        read_name="read1",
        chrom="chr2",
        ref_start=10,
        ref_end=20,
        strand="+",
        posteriors=np.array([0.0, 0.5, 1.0], dtype=np.float32),
        footprint_starts=np.array([2, 8], dtype=np.int32),
        footprint_sizes=np.array([3, 4], dtype=np.int32),
    )
    line = format_region_posterior_line_from_record(record)

    parsed = parse_posteriors_line(line)

    assert parsed["read_id"] == "read1"
    assert parsed["chrom"] == "chr2"
    assert parsed["start"] == 10
    assert parsed["end"] == 20
    assert parsed["strand"] == "+"
    np.testing.assert_allclose(parsed["posteriors"], [0.0, 127 / 255, 1.0])
    np.testing.assert_array_equal(parsed["fp_starts"], [2, 8])
    np.testing.assert_array_equal(parsed["fp_sizes"], [3, 4])
    assert format_region_posterior_line(
        read_name="read1",
        chrom="chr2",
        ref_start=10,
        ref_end=20,
        strand="+",
        posteriors=np.array([0.0, 0.5, 1.0], dtype=np.float32),
        footprint_starts=np.array([2, 8], dtype=np.int32),
        footprint_sizes=np.array([3, 4], dtype=np.int32),
    ) == line


def test_region_posterior_record_from_fiber_names_worker_payload():
    fiber = {
        "read_name": "read1",
        "chrom": "chr2",
        "ref_start": 10,
        "ref_end": 20,
        "strand": "+",
        "posteriors": np.array([0.0, 1.0], dtype=np.float32),
        "footprint_starts": np.array([2], dtype=np.int32),
        "footprint_sizes": np.array([3], dtype=np.int32),
    }

    record = region_tsv._region_posterior_record_from_fiber(fiber)

    assert record.read_name == "read1"
    assert record.chrom == "chr2"
    assert record.ref_start == 10
    assert record.ref_end == 20
    assert record.strand == "+"
    assert record.posteriors is fiber["posteriors"]
    assert record.footprint_starts is fiber["footprint_starts"]
    assert record.footprint_sizes is fiber["footprint_sizes"]


def test_write_region_posteriors_tsv_request_writes_records(monkeypatch, tmp_path):
    tsv_path = tmp_path / "region.tsv"
    posteriors_data = [{
        "read_name": "read1",
        "chrom": "chr2",
        "ref_start": 10,
        "ref_end": 20,
        "strand": "+",
        "posteriors": np.array([0.0, 1.0], dtype=np.float32),
        "footprint_starts": np.array([2], dtype=np.int32),
        "footprint_sizes": np.array([3], dtype=np.int32),
    }]

    region_tsv.write_region_posteriors_tsv_from_request(
        region_tsv._RegionTsvWriteRequest(
            tsv_path=str(tsv_path),
            posteriors_data=posteriors_data,
        )
    )

    parsed = parse_posteriors_line(tsv_path.read_text(encoding="utf-8"))
    assert parsed["read_id"] == "read1"
    assert parsed["chrom"] == "chr2"
    np.testing.assert_array_equal(parsed["fp_starts"], [2])
    np.testing.assert_array_equal(parsed["fp_sizes"], [3])

    calls = []
    monkeypatch.setattr(
        region_tsv,
        "write_region_posteriors_tsv_from_request",
        lambda request: calls.append(request),
    )
    adapter_data = [{"read_name": "read2"}]
    region_tsv.write_region_posteriors_tsv("adapter.tsv", adapter_data)

    assert calls == [
        region_tsv._RegionTsvWriteRequest(
            tsv_path="adapter.tsv",
            posteriors_data=adapter_data,
        ),
    ]


def test_region_posteriors_tsv_output_path(tmp_path):
    assert region_posteriors_tsv_output_path("out.h5") == "out.tsv.gz"
    assert region_posteriors_tsv_output_path("out.hdf5") == "out.tsv.gz"
    assert region_posteriors_tsv_output_path("out.H5") == "out.tsv.gz"
    assert region_posteriors_tsv_output_path("out.tsv") == "out.tsv.gz"
    assert region_posteriors_tsv_output_path("out.TSV") == "out.TSV.gz"
    assert region_posteriors_tsv_output_path("out.tsv.gz") == "out.tsv.gz"
    assert region_posteriors_tsv_output_path("out.TSV.GZ") == "out.TSV.GZ"
    assert region_posteriors_tsv_output_path("out") == "out.tsv.gz"
    assert region_posteriors_tsv_output_path(tmp_path / "out.h5") == str(
        tmp_path / "out.tsv.gz",
    )
    assert region_posteriors_needs_h5_conversion("out.h5")
    assert region_posteriors_needs_h5_conversion("out.hdf5")
    assert region_posteriors_needs_h5_conversion("out.HDF5")
    assert not region_posteriors_needs_h5_conversion("out.tsv.gz")


def test_posterior_tsv_metadata_request_uses_source_bam_basename(tmp_path):
    assert region_tsv._posterior_tsv_metadata_from_request(
        region_tsv._PosteriorTsvMetadataRequest(
            mode="daf",
            context_size=5,
            edge_trim=12,
            source_bam=tmp_path / "input.bam",
        )
    ) == {
        "mode": "daf",
        "context_size": 5,
        "edge_trim": 12,
        "source_bam": "input.bam",
        "format_version": 1,
    }


def test_region_tsv_header_and_record_copy_helpers(tmp_path):
    output = io.StringIO()

    header_request = region_tsv._RegionTsvHeaderRequest(
        outfile=output,
        mode="daf",
        context_size=5,
        edge_trim=12,
        source_bam=tmp_path / "input.bam",
    )
    region_tsv._write_region_tsv_header_from_request(header_request)

    adapter_output = io.StringIO()
    region_tsv._write_region_tsv_header(
        adapter_output,
        mode="daf",
        context_size=5,
        edge_trim=12,
        source_bam=tmp_path / "input.bam",
    )
    assert adapter_output.getvalue() == output.getvalue()

    header_lines = output.getvalue().splitlines()
    metadata = json.loads(header_lines[0].removeprefix("#metadata:"))
    assert metadata == {
        "mode": "daf",
        "context_size": 5,
        "edge_trim": 12,
        "source_bam": "input.bam",
        "format_version": 1,
    }
    assert header_lines[1].startswith("#read_id\tchrom")

    records = tmp_path / "records.tsv"
    records.write_text(
        "#metadata:{}\n"
        "#read_id\tchrom\n"
        "\n"
        "read1\tchr1\t1\t2\t+\tAA==\t\t\n",
        encoding="utf-8",
    )

    assert region_tsv._copy_region_tsv_records_from_request(
        region_tsv._RegionTsvCopyRequest(
            outfile=output,
            tsv_path=str(records),
        )
    ) == 1
    assert output.getvalue().endswith("read1\tchr1\t1\t2\t+\tAA==\t\t\n")
    assert "#metadata:{}" not in output.getvalue()

    adapter_output = io.StringIO()
    assert region_tsv._copy_region_tsv_records(adapter_output, str(records)) == 1
    assert adapter_output.getvalue() == "read1\tchr1\t1\t2\t+\tAA==\t\t\n"


def test_merge_region_posteriors_tsv_orders_regions_and_preserves_input_list(tmp_path):
    first = tmp_path / "region_0.tsv"
    second = tmp_path / "region_1.tsv"
    empty = tmp_path / "empty.tsv"
    directory = tmp_path / "region_dir.tsv"
    output = tmp_path / "posteriors.h5"

    first.write_text("read_a\tchr1\t1\t5\t+\tAAE=\t1\t4\n")
    second.write_text("read_b\tchr1\t5\t9\t-\tAgM=\t2\t4\n")
    empty.write_text("")
    directory.mkdir()

    temp_files = [
        (1, str(second)),
        (0, str(first)),
        (2, str(empty)),
        (3, str(tmp_path / "missing.tsv")),
        (4, str(directory)),
    ]
    original_order = list(temp_files)

    assert region_tsv._valid_region_tsv_files(temp_files) == [
        region_tsv._IndexedRegionTsvFile(0, str(first)),
        region_tsv._IndexedRegionTsvFile(1, str(second)),
    ]

    n_fibers = region_tsv.merge_region_posteriors_tsv_from_request(
        region_tsv._RegionTsvMergeRequest(
            temp_tsv_files=temp_files,
            output_path=str(output),
            mode="pacbio-fiber",
            context_size=3,
            edge_trim=10,
            source_bam="/data/source.bam",
        )
    )
    assert merge_region_posteriors_tsv(
        [],
        str(tmp_path / "empty.h5"),
        mode="pacbio-fiber",
        context_size=3,
        edge_trim=10,
        source_bam="/data/source.bam",
    ) == 0

    assert n_fibers == 2
    assert temp_files == original_order

    with gzip.open(tmp_path / "posteriors.tsv.gz", "rt") as handle:
        lines = handle.readlines()

    assert lines[0].startswith("#metadata:")
    metadata = json.loads(lines[0].removeprefix("#metadata:"))
    assert metadata == {
        "mode": "pacbio-fiber",
        "context_size": 3,
        "edge_trim": 10,
        "source_bam": "source.bam",
        "format_version": 1,
    }
    assert lines[1].startswith("#read_id\tchrom")
    assert lines[2].startswith("read_a\t")
    assert lines[3].startswith("read_b\t")
