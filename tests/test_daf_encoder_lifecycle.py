from __future__ import annotations

import pytest

from fiberhmm.daf import encoder


class _FakeHandle:
    def __init__(self):
        self.header = {}
        self.closed = False

    def close(self):
        self.closed = True


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
