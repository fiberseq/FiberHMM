"""The DAF SNP screen must not read reference bases from a malformed MD.

For an MD tag shorter than the CIGAR's M/=/X/D span, pysam's
``get_aligned_pairs(with_seq=True)`` copies whatever follows the parsed MD in
memory into the "reference", so the screen (and QC) saw different reference
bases, dominant directions and calls on every run. Such reads now use the
FASTA, or are unusable without one, as in ``get_daf_positions`` and dedup.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pysam
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_daf_mismatch_fastpath as fastpath  # noqa: E402

from fiberhmm.daf import snps  # noqa: E402
from fiberhmm.daf.aligned_arrays import md_disagrees_with_cigar  # noqa: E402
from fiberhmm.qc.core import _aligned_reference_pairs  # noqa: E402


# Seed 313 has 1/17 reads with a short MD, 1/17 with a long MD, and one C
# site whose MD reads G on every 7th amplicon read (both hypotheses profiled).
_SEED = 313


@pytest.fixture(scope="module")
def hard_dataset(tmp_path_factory):
    return fastpath._write_dataset(
        tmp_path_factory.mktemp("snpdet"), _SEED, short_md=True)


def _reads(bam_path):
    with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
        return list(bam.fetch(until_eof=True))


def _fasta_truth(read, fasta):
    """(rpos, ref_codes, query_codes) of the M/=/X bases, reference from FASTA."""
    reference = fasta.fetch(read.reference_name).upper()
    sequence = read.query_sequence.upper()
    pairs = read.get_aligned_pairs(matches_only=True)
    return ([r for _q, r in pairs], [ord(reference[r]) for _q, r in pairs],
            [ord(sequence[q]) for q, _r in pairs])


def _bad_md_reads(bam_path):
    """Mapped reads whose MD is shorter or longer than the CIGAR's span."""
    return [read for read in _reads(bam_path)
            if not read.is_unmapped and md_disagrees_with_cigar(read)]


class _SpyRead:
    """Wraps a pysam read and records ``get_aligned_pairs`` calls."""

    def __init__(self, read):
        self._read = read
        self.with_seq_calls = 0

    def get_aligned_pairs(self, *args, **kwargs):
        if kwargs.get("with_seq"):
            self.with_seq_calls += 1
        return self._read.get_aligned_pairs(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._read, name)


def test_md_reconstruction_is_skipped_when_md_does_not_describe_the_cigar(hard_dataset):
    bam_path, _fasta = hard_dataset
    bad = _bad_md_reads(bam_path)
    assert len(bad) >= 10
    for read in bad:
        spy = _SpyRead(read)
        assert snps._aligned_pairs(spy) is None
        assert _aligned_reference_pairs(spy) is None
        assert spy.with_seq_calls == 0
    # A valid MD still goes through pysam's reconstruction.
    good = next(read for read in _reads(bam_path)
                if not read.is_unmapped and read.has_tag("MD")
                and not md_disagrees_with_cigar(read))
    spy = _SpyRead(good)
    assert snps._aligned_pairs(spy) is not None
    assert spy.with_seq_calls == 1


def test_md_disagreement_matches_the_encoder_pre_validation(hard_dataset):
    from fiberhmm.daf.encoder import md_matches_cigar

    bam_path, _fasta = hard_dataset
    kinds = Counter()
    for read in _reads(bam_path):
        if read.is_unmapped:
            continue
        assert md_disagrees_with_cigar(read) == (not md_matches_cigar(read))
        kinds[md_disagrees_with_cigar(read)] += 1
    assert kinds[True] >= 10 and kinds[False] >= 300


def test_bad_md_reads_use_the_fasta_or_are_unusable(hard_dataset):
    bam_path, fasta_path = hard_dataset
    bad = _bad_md_reads(bam_path)
    assert bad
    with pysam.FastaFile(str(fasta_path)) as fasta:
        for read in bad:
            assert snps._read_arrays(read) is None
            rpos, ref_codes, query_codes = snps._read_arrays(read, fasta)
            assert (rpos.tolist(), ref_codes.tolist(), query_codes.tolist()) \
                == _fasta_truth(read, fasta)
            pairs = _aligned_reference_pairs(read, fasta)
            reference = fasta.fetch(read.reference_name).upper()
            assert pairs and all(base == reference[r] for _q, r, base in pairs)
