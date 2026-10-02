"""fiberhmm-extract and BAM indexes (release audit 2026-09-29, CLI M18).

Extract checked only for ``x.bam.bai``/``x.bai``, so a CSI-only index was
ignored and it ran ``samtools index`` next to the input, which fails in a
read-only directory (``SamtoolsError: failed to create or write index``).
It now uses whatever index htslib finds and, without one, indexes into a
temporary file.
"""
from __future__ import annotations

import os
import stat

import pysam
import pytest

from fiberhmm.cli.extract_tags import extract_tags_parallel


def _called_bam(directory, name="in.bam"):
    from conftest import make_synthetic_bam

    path = str(directory / name)
    make_synthetic_bam(path, n_reads=6, read_length=600, n_chroms=1,
                       chrom_length=20_000, seed=23)
    os.remove(path + ".bai")
    with pysam.AlignmentFile(path, "rb") as bam:
        reads = list(bam.fetch(until_eof=True))
        header = bam.header
    with pysam.AlignmentFile(path, "wb", header=header) as out:
        for read in reads:
            read.set_tag("ns", [50])
            read.set_tag("nl", [147])
            read.set_tag("as", [197])
            read.set_tag("al", [100])
            out.write(read)
    return path


def _extract(path, outdir):
    beds = {"nucleosome": str(outdir / "nuc.bed"), "msp": str(outdir / "msp.bed")}
    n_reads, features = extract_tags_parallel(
        path, beds, ["nucleosome", "msp"], n_cores=1)
    return n_reads, features


@pytest.fixture
def read_only(tmp_path):
    directory = tmp_path / "ro"
    directory.mkdir()
    yield directory
    directory.chmod(stat.S_IRWXU)


def test_csi_only_index_in_a_read_only_directory(read_only, tmp_path):
    path = _called_bam(read_only)
    pysam.index("-c", path)
    assert os.path.exists(path + ".csi") and not os.path.exists(path + ".bai")
    read_only.chmod(stat.S_IRUSR | stat.S_IXUSR)
    n_reads, features = _extract(path, tmp_path)
    assert n_reads == 6
    assert features["nucleosome"] == 6 and features["msp"] == 6
    assert sorted(os.listdir(read_only)) == ["in.bam", "in.bam.csi"]


def test_unindexed_input_is_indexed_into_a_temporary_file(read_only, tmp_path):
    path = _called_bam(read_only)
    read_only.chmod(stat.S_IRUSR | stat.S_IXUSR)
    n_reads, features = _extract(path, tmp_path)
    assert n_reads == 6 and features["nucleosome"] == 6
    assert os.listdir(read_only) == ["in.bam"]


def test_unsorted_input_is_a_clear_error(tmp_path):
    path = _called_bam(tmp_path)
    with pysam.AlignmentFile(path, "rb") as bam:
        reads = list(bam.fetch(until_eof=True))
        header = bam.header.to_dict()
    header["HD"]["SO"] = "unsorted"
    unsorted = str(tmp_path / "unsorted.bam")
    with pysam.AlignmentFile(unsorted, "wb", header=header) as out:
        for read in reversed(reads):
            out.write(pysam.AlignedSegment.fromstring(read.to_string(), out.header))
    with pytest.raises(RuntimeError, match="coordinate-sorted"):
        _extract(unsorted, tmp_path)
