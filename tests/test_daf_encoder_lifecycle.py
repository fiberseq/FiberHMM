from __future__ import annotations

import io
from types import SimpleNamespace

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


def _daf_read(**overrides):
    attrs = {
        "is_unmapped": False,
        "is_secondary": False,
        "is_supplementary": False,
        "mapping_quality": 60,
        "query_alignment_length": 1500,
    }
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


class _WritableDafRead:
    is_unmapped = False
    is_secondary = False
    is_supplementary = False
    mapping_quality = 60
    query_alignment_length = 1500
    query_length = 4

    def __init__(self):
        self.query_sequence = "ACGT"
        self.query_qualities = [30, 31, 32, 33]
        self.tags = {}

    def set_tag(self, tag, value, value_type=None):
        self.tags[tag] = (value, value_type)


def test_daf_encode_skip_reason_matches_filter_order():
    assert encoder._daf_encode_skip_reason(_daf_read(), 20, 1000) is None
    assert encoder._daf_encode_skip_reason(
        _daf_read(is_unmapped=True), 20, 1000
    ) == "not_primary_mapped"
    assert encoder._daf_encode_skip_reason(
        _daf_read(mapping_quality=10), 20, 1000
    ) == "low_mapq"
    assert encoder._daf_encode_skip_reason(
        _daf_read(query_alignment_length=999), 20, 1000
    ) == "too_short"
    assert encoder._daf_encode_skip_reason(
        _daf_read(query_alignment_length=None), 20, 1000
    ) is None


def test_select_daf_strand_handles_forced_dominant_and_ambiguous_cases():
    assert encoder._select_daf_strand(0, 0) is None
    assert encoder._select_daf_strand(5, 2) == "CT"
    assert encoder._select_daf_strand(2, 5) == "GA"
    assert encoder._select_daf_strand(3, 3) is None
    assert encoder._select_daf_strand(0, 0, force_strand="ga") == "GA"


def test_daf_mismatch_positions_from_pairs_collects_deamination_directions():
    pairs = [
        (0, 100, "C"),     # C->T
        (1, 101, "G"),     # G->A
        (2, 102, "A"),     # not a deamination-direction mismatch
        (None, 103, "C"),
        (3, None, "G"),
        (4, 104, None),
    ]

    assert encoder._daf_mismatch_positions_from_pairs(pairs, "TAGTA") == (
        [0],
        [1],
    )


def test_needs_fasta_pair_fallback_detects_missing_ref_bases():
    assert encoder._needs_fasta_pair_fallback(None)
    assert encoder._needs_fasta_pair_fallback([
        (0, 100, None),
        (None, 101, "A"),
        (1, None, "C"),
    ])
    assert not encoder._needs_fasta_pair_fallback([(0, 100, "C")])


def test_mark_iupac_positions_replaces_selected_bases():
    assert encoder._mark_iupac_positions("ACGT", [1, 3], "Y") == "AYGY"


def test_encoded_daf_sequence_marks_selected_strand_and_counts_events():
    assert encoder._encoded_daf_sequence("ACGT", [1], [2], "CT") == ("AYGT", 1)
    assert encoder._encoded_daf_sequence("ACGT", [1], [2], "GA") == ("ACRT", 1)


def test_daf_encode_throughput_handles_zero_elapsed():
    assert encoder._daf_encode_throughput({"total": 100, "elapsed": 2.0}) == 50.0
    assert encoder._daf_encode_throughput({"total": 100, "elapsed": 0.0}) is None


def test_daf_progress_rate_handles_zero_elapsed():
    assert encoder._daf_progress_rate(10_000, 2.0) == 5000.0
    assert encoder._daf_progress_rate(10_000, 0.0) is None


def test_print_daf_progress_formats_rate_counts_and_skips_zero_elapsed():
    log = io.StringIO()

    encoder._print_daf_progress(10_000, 10_000, 2.0, 12, 34, 56, log)
    assert log.getvalue() == (
        "  [10,000 reads] 5,000 reads/sec (CT=12 GA=34 skip=56)\n"
    )

    log = io.StringIO()
    encoder._print_daf_progress(10_000, 10_000, 0.0, 12, 34, 56, log)
    assert log.getvalue() == ""


def test_daf_encode_summary_and_report_format():
    summary = encoder._daf_encode_summary(
        total=100,
        encoded=80,
        ct_count=30,
        ga_count=50,
        skipped=20,
        total_deam=25,
        total_bases=1000,
        elapsed=2.0,
    )

    assert summary == {
        "total": 100,
        "encoded": 80,
        "ct": 30,
        "ga": 50,
        "skipped": 20,
        "mean_deam_rate": 0.025,
        "elapsed": 2.0,
    }

    log = io.StringIO()
    encoder._print_daf_encode_summary(summary, log)
    text = log.getvalue()

    assert "fiberhmm-daf-encode summary" in text
    assert "Total reads:" in text
    assert "Mean deam. rate:" in text
    assert "0.0250" in text
    assert "50 reads/sec" in text


def test_apply_daf_encoding_to_read_preserves_qualities_and_sets_st_tag():
    class FakeRead:
        def __init__(self):
            self.query_sequence = "ACT"
            self.query_qualities = [30, 31, 32]
            self.tags = {}

        def set_tag(self, tag, value, value_type=None):
            self.tags[tag] = (value, value_type)

    read = FakeRead()
    qualities = read.query_qualities

    encoder._apply_daf_encoding_to_read(read, "AYT", "CT")

    assert read.query_sequence == "AYT"
    assert read.query_qualities is qualities
    assert read.tags["st"] == ("CT", "Z")


def test_daf_encoded_read_stats_counts_selected_strand_and_bases():
    assert encoder._daf_encoded_read_stats(_daf_read(query_length=4), "CT", 2) == {
        "encoded": 1,
        "skipped": 0,
        "ct": 1,
        "ga": 0,
        "total_deam": 2,
        "total_bases": 4,
    }
    assert encoder._daf_encoded_read_stats(_daf_read(query_length=None), "GA", 3) == {
        "encoded": 1,
        "skipped": 0,
        "ct": 0,
        "ga": 1,
        "total_deam": 3,
        "total_bases": 0,
    }


def test_process_daf_encode_read_writes_encoded_read_and_returns_stats(monkeypatch):
    read = _WritableDafRead()
    qualities = read.query_qualities
    handle = _FakeHandle()
    handle.written = []
    handle.write = handle.written.append
    progress = _FakeProgress()

    monkeypatch.setattr(
        encoder,
        "encode_read_daf",
        lambda *args, **kwargs: ("AYGT", "CT", 1),
    )

    assert encoder._process_daf_encode_read(
        handle, progress, read, 20, 1000,
    ) == {
        "encoded": 1,
        "skipped": 0,
        "ct": 1,
        "ga": 0,
        "total_deam": 1,
        "total_bases": 4,
    }
    assert handle.written == [read]
    assert progress.n == 1
    assert read.query_sequence == "AYGT"
    assert read.query_qualities is qualities
    assert read.tags["st"] == ("CT", "Z")


def test_process_daf_encode_read_writes_skipped_read(monkeypatch):
    read = _WritableDafRead()
    read.mapping_quality = 10
    handle = _FakeHandle()
    handle.written = []
    handle.write = handle.written.append
    progress = _FakeProgress()

    def fail_encode(*args, **kwargs):
        raise AssertionError("filtered reads should not be encoded")

    monkeypatch.setattr(encoder, "encode_read_daf", fail_encode)

    assert encoder._process_daf_encode_read(
        handle, progress, read, 20, 1000,
    ) == encoder._daf_skipped_read_stats()
    assert handle.written == [read]
    assert progress.n == 1


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
