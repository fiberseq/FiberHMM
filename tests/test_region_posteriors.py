"""Tests for region-parallel posterior TSV helpers."""

import gzip
import json

import numpy as np

from fiberhmm.posteriors.region_tsv import (
    format_region_posterior_line,
    merge_region_posteriors_tsv,
    region_posteriors_tsv_output_path,
)
from fiberhmm.posteriors.tsv_backend import parse_posteriors_line


def test_format_region_posterior_line_matches_tsv_parser():
    line = format_region_posterior_line(
        read_name="read1",
        chrom="chr2",
        ref_start=10,
        ref_end=20,
        strand="+",
        posteriors=np.array([0.0, 0.5, 1.0], dtype=np.float32),
        footprint_starts=np.array([2, 8], dtype=np.int32),
        footprint_sizes=np.array([3, 4], dtype=np.int32),
    )

    parsed = parse_posteriors_line(line)

    assert parsed["read_id"] == "read1"
    assert parsed["chrom"] == "chr2"
    assert parsed["start"] == 10
    assert parsed["end"] == 20
    assert parsed["strand"] == "+"
    np.testing.assert_allclose(parsed["posteriors"], [0.0, 127 / 255, 1.0])
    np.testing.assert_array_equal(parsed["fp_starts"], [2, 8])
    np.testing.assert_array_equal(parsed["fp_sizes"], [3, 4])


def test_region_posteriors_tsv_output_path():
    assert region_posteriors_tsv_output_path("out.h5") == "out.tsv.gz"
    assert region_posteriors_tsv_output_path("out.tsv") == "out.tsv.gz"
    assert region_posteriors_tsv_output_path("out.tsv.gz") == "out.tsv.gz"
    assert region_posteriors_tsv_output_path("out") == "out.tsv.gz"


def test_merge_region_posteriors_tsv_orders_regions_and_preserves_input_list(tmp_path):
    first = tmp_path / "region_0.tsv"
    second = tmp_path / "region_1.tsv"
    empty = tmp_path / "empty.tsv"
    output = tmp_path / "posteriors.h5"

    first.write_text("read_a\tchr1\t1\t5\t+\tAAE=\t1\t4\n")
    second.write_text("read_b\tchr1\t5\t9\t-\tAgM=\t2\t4\n")
    empty.write_text("")

    temp_files = [
        (1, str(second)),
        (0, str(first)),
        (2, str(empty)),
        (3, str(tmp_path / "missing.tsv")),
    ]
    original_order = list(temp_files)

    n_fibers = merge_region_posteriors_tsv(
        temp_files,
        str(output),
        mode="pacbio-fiber",
        context_size=3,
        edge_trim=10,
        source_bam="/data/source.bam",
    )

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
