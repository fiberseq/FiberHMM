from __future__ import annotations

import pytest

from fiberhmm.daf import encoder


class _FakeHandle:
    def __init__(self):
        self.header = {}
        self.closed = False

    def close(self):
        self.closed = True


class _FakeProgress:
    def __init__(self):
        self.n = 0

    def update(self, amount):
        self.n += amount


def test_write_skipped_daf_read_writes_updates_and_returns_increment():
    handle = _FakeHandle()
    handle.written = []
    handle.write = handle.written.append
    progress = _FakeProgress()
    read = object()

    assert encoder._write_skipped_daf_read(handle, progress, read) == 1
    assert handle.written == [read]
    assert progress.n == 1


def test_daf_encode_closes_input_and_reference_when_md_check_fails(monkeypatch):
    handles = {
        "fasta": _FakeHandle(),
        "bam": _FakeHandle(),
    }

    monkeypatch.setattr(encoder.pysam, "FastaFile", lambda *args, **kwargs: handles["fasta"])
    monkeypatch.setattr(
        encoder.pysam,
        "AlignmentFile",
        lambda *args, **kwargs: handles["bam"],
    )

    def fail_md_check(*args, **kwargs):
        raise RuntimeError("md check failed")

    monkeypatch.setattr(encoder, "_check_md_tag", fail_md_check)

    with pytest.raises(RuntimeError, match="md check failed"):
        encoder.process_bam_daf_encode(
            "input.bam",
            "output.bam",
            reference="reference.fa",
        )

    assert handles["bam"].closed
    assert handles["fasta"].closed


def test_aligned_pairs_from_fasta_fetches_reference_span_once():
    class FakeRead:
        reference_name = "chr1"
        reference_start = 100
        reference_end = 105

        def get_aligned_pairs(self):
            return [(0, 100), (1, 101), (2, None), (3, 104)]

    class FakeFasta:
        def __init__(self):
            self.fetches = []

        def fetch(self, chrom, start, end):
            self.fetches.append((chrom, start, end))
            assert (chrom, start, end) == ("chr1", 100, 105)
            return "ACGTA"

    fasta = FakeFasta()

    assert encoder._aligned_pairs_from_fasta(FakeRead(), fasta) == [
        (0, 100, "A"),
        (1, 101, "C"),
        (2, None, None),
        (3, 104, "A"),
    ]
    assert fasta.fetches == [("chr1", 100, 105)]
