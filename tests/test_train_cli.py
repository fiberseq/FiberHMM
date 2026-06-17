import pytest

from fiberhmm.cli import train
from fiberhmm.core import bam_reader

pysam = pytest.importorskip("pysam")


def test_state_runs_groups_contiguous_states():
    assert train._state_runs([0, 0, 1, 1, 1, 0]) == [
        (0, 2, 0),
        (2, 5, 1),
        (5, 6, 0),
    ]
    assert train._state_runs([]) == []


def test_state_run_lengths_splits_footprint_and_msp_lengths():
    footprint_sizes, msp_sizes = train._state_run_lengths([0, 0, 1, 1, 1, 0])

    assert footprint_sizes == [2, 1]
    assert msp_sizes == [3]


def test_sample_reads_indexed_preserves_reverse_flag(monkeypatch):
    class FakeRead:
        is_unmapped = False
        is_secondary = False
        is_supplementary = False
        mapping_quality = 60
        query_sequence = "ACGT" * 300
        reference_start = 100
        reference_end = reference_start + len(query_sequence)
        query_name = "reverse-read"
        reference_name = "chr1"
        is_reverse = True

        def has_tag(self, tag):
            return False

        def get_tag(self, tag):
            raise KeyError(tag)

        def get_aligned_pairs(self):
            return [
                (query_pos, self.reference_start + query_pos)
                for query_pos in range(len(self.query_sequence))
            ]

    class FakeBam:
        references = ("chr1",)
        lengths = (2000,)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def fetch(self, chrom, start, end):
            return iter([FakeRead()])

    monkeypatch.setattr(pysam, "AlignmentFile", lambda *args, **kwargs: FakeBam())
    monkeypatch.setattr(bam_reader, "get_modified_positions_pysam", lambda *args, **kwargs: {5})

    reads = train.sample_reads_indexed(
        "fake.bam",
        n_samples=1,
        seed=1,
        min_mapq=0,
        min_read_length=0,
    )

    assert len(reads) == 1
    assert reads[0].strand == "-"
    assert reads[0].is_reverse is True
