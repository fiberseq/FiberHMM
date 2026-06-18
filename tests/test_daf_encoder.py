"""Tests for DAF raw-alignment encoder helpers."""

from fiberhmm.daf import encoder as daf_encoder


class _FakeRead:
    def __init__(self, *, md=None, cigartuples=None, pairs=None, aligned_error=None):
        self._md = md
        self.cigartuples = cigartuples
        self._pairs = pairs
        self._aligned_error = aligned_error
        self.aligned_pair_calls = 0

    def has_tag(self, tag):
        return tag == "MD" and self._md is not None

    def get_tag(self, tag):
        if tag != "MD" or self._md is None:
            raise KeyError(tag)
        return self._md

    def get_aligned_pairs(self, *, with_seq):
        assert with_seq is True
        self.aligned_pair_calls += 1
        if self._aligned_error is not None:
            raise self._aligned_error
        return self._pairs


def test_aligned_pairs_with_reference_bases_uses_valid_md_pairs():
    pairs = [(0, 0, "C"), (1, 1, "G")]
    read = _FakeRead(md="2", cigartuples=[(0, 2)], pairs=pairs)

    assert daf_encoder._aligned_pairs_with_reference_bases(read) is pairs
    assert read.aligned_pair_calls == 1


def test_aligned_pairs_with_reference_bases_rejects_bad_md_without_fasta():
    read = _FakeRead(
        md="1",
        cigartuples=[(0, 2)],
        pairs=[(0, 0, "C"), (1, 1, "G")],
    )

    assert daf_encoder._aligned_pairs_with_reference_bases(read) is None
    assert read.aligned_pair_calls == 0


def test_aligned_pairs_with_reference_bases_falls_back_after_pair_error(monkeypatch):
    read = _FakeRead(
        cigartuples=[(0, 2)],
        aligned_error=ValueError("missing MD"),
    )
    ref_fasta = object()
    fallback = [(0, 0, "C"), (1, 1, "G")]

    monkeypatch.setattr(
        daf_encoder,
        "_aligned_pairs_from_fasta",
        lambda got_read, got_ref: fallback,
    )

    assert daf_encoder._aligned_pairs_with_reference_bases(read, ref_fasta) is fallback


def test_aligned_pairs_with_reference_bases_falls_back_for_missing_ref_bases(
    monkeypatch,
):
    read = _FakeRead(
        md="2",
        cigartuples=[(0, 2)],
        pairs=[(0, 0, None), (1, 1, None)],
    )
    ref_fasta = object()
    fallback = [(0, 0, "C"), (1, 1, "G")]

    monkeypatch.setattr(
        daf_encoder,
        "_aligned_pairs_from_fasta",
        lambda got_read, got_ref: fallback,
    )

    assert daf_encoder._aligned_pairs_with_reference_bases(read, ref_fasta) is fallback
