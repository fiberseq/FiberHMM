"""End-to-end contract tests for the unified ``fiberhmm-pair`` workflow."""
from __future__ import annotations

import json

import pysam
import pytest

from fiberhmm.cli.duplex import run_pairing
from fiberhmm.cli.merge import run_merge
from fiberhmm.crossstrand.duplex import DuplexParams
from fiberhmm.crossstrand.pairing import PairParams


def test_default_pairing_requires_reference_before_opening_input():
    with pytest.raises(ValueError, match="default pairing requires --reference"):
        run_pairing("missing.bam", "unused.bam", None)


def _write_sequence_resolved_fixture(tmp_path):
    length = 3000
    reference = tmp_path / "reference.fa"
    reference.write_text(">chr1\n" + "A" * length + "\n")
    pysam.faidx(str(reference))
    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": length}],
    })

    def read(name, flavor, family_b):
        sequence = list("A" * 2000)
        sequence[0 if flavor == "CT" else 1] = "Y" if flavor == "CT" else "R"
        if family_b:
            sequence[100:110] = "C" * 10
        record = pysam.AlignedSegment(header)
        record.query_name = name
        record.reference_id = 0
        record.reference_start = 100
        record.mapping_quality = 60
        record.cigartuples = [(0, len(sequence))]
        record.query_sequence = "".join(sequence)
        record.query_qualities = pysam.qualitystring_to_array("I" * len(sequence))
        record.set_tag("MA", f"{len(sequence)}")
        return record

    bam = tmp_path / "source.bam"
    with pysam.AlignmentFile(bam, "wb", header=header) as out:
        out.write(read("ct-a", "CT", False))
        out.write(read("ct-b", "CT", True))
        out.write(read("ga-a", "GA", False))
        out.write(read("ga-b", "GA", True))
    pysam.index(str(bam))
    return bam, reference


def test_sequence_only_accepts_only_sequence_supported_pairs(tmp_path):
    source, reference = _write_sequence_resolved_fixture(tmp_path)
    output = tmp_path / "paired.bam"
    receipt_path = tmp_path / "receipt.json"
    receipt = run_pairing(
        str(source), str(output), str(reference),
        params=DuplexParams(min_overlap_bp=100, min_nucs=1),
        sequence_params=PairParams(
            min_overlap_bp=100, min_nucs=1, min_sequence_bases=500,
            min_sequence_margin=0.002, max_sequence_pair_rate=0.01,
        ),
        pairing_mode="sequence-only", receipt_json=str(receipt_path),
        io_threads=1,
    )
    assert receipt["pairing_mode"] == "sequence-only"
    assert receipt["counts"]["sequence_pairs"] == 2
    assert receipt["counts"]["sequence_free_pairs"] == 0
    assert json.loads(receipt_path.read_text())["AT_mismatch_used"] is True
    with pysam.AlignmentFile(output, "rb") as bam:
        reads = list(bam.fetch(until_eof=True))
    assert len(reads) == 4
    assert all(read.get_tag("mt") == "P" for read in reads)
    assert all(read.get_tag("pm") == "S" for read in reads)
    assert all(not read.has_tag("dm") and not read.has_tag("mv") for read in reads)

    consensus = tmp_path / "consensus.bam"
    merged = run_merge(str(output), str(consensus), pairs_only=True, io_threads=1)
    assert merged["n_consensus"] == 2
    with pysam.AlignmentFile(consensus, "rb") as bam:
        assert sum(1 for _ in bam.fetch(until_eof=True)) == 2


def test_default_hybrid_retains_independent_sequence_pairs(tmp_path):
    source, reference = _write_sequence_resolved_fixture(tmp_path)
    output = tmp_path / "hybrid.bam"
    receipt = run_pairing(
        str(source), str(output), str(reference),
        params=DuplexParams(min_overlap_bp=100, min_nucs=1),
        sequence_params=PairParams(
            min_overlap_bp=100, min_nucs=1, min_sequence_bases=500,
            min_sequence_margin=0.002, max_sequence_pair_rate=0.01,
        ),
        pairing_mode="hybrid", io_threads=1,
    )
    assert receipt["counts"]["sequence_pairs"] == 2
    assert receipt["counts"]["pairs"] == 2
    with pysam.AlignmentFile(output, "rb") as bam:
        assert {read.get_tag("pm") for read in bam.fetch(until_eof=True)} == {"S"}


def _run_pair_cli(monkeypatch, *argv):
    import sys
    from fiberhmm.cli import pair
    monkeypatch.setattr(sys, "argv", ["fiberhmm-pair", *argv])
    pair.main()


def _records(path):
    with pysam.AlignmentFile(path, "rb", check_sq=False) as bam:
        return list(bam.fetch(until_eof=True))


def test_one_command_stages_and_from_paired(tmp_path, monkeypatch, capsys):
    source, reference = _write_sequence_resolved_fixture(tmp_path)
    common = ["-r", str(reference), "--sequence-only", "--min-overlap", "100", "--min-nucs", "1", "--io-threads", "1"]
    paired = tmp_path / "paired.bam"
    _run_pair_cli(monkeypatch, "-i", str(source), "-o", str(paired), *common, "--stop-after", "pair")
    assert "stages pair" in capsys.readouterr().err
    assert [r.get_tag("mt") for r in _records(paired)] == ["P"] * 4

    merged = tmp_path / "merged.bam"
    _run_pair_cli(monkeypatch, "-i", str(source), "-o", str(merged), *common,
                  "--stop-after", "merge", "--pairs-only")
    assert "stages pair -> merge" in capsys.readouterr().err
    assert len(_records(merged)) == 2

    # formerly fiberhmm-merge: start from the tagged pairs
    again = tmp_path / "again.bam"
    _run_pair_cli(monkeypatch, "-i", str(paired), "-o", str(again), "--from-paired",
                  "--stop-after", "merge", "--pairs-only", "--io-threads", "1")
    assert "stages merge" in capsys.readouterr().err
    assert len(_records(again)) == 2

    # older command line: --merge alone still means merge without re-calling
    legacy = tmp_path / "legacy.bam"
    _run_pair_cli(monkeypatch, "-i", str(source), "-o", str(legacy), *common, "--merge", "--pairs-only")
    assert "stages pair -> merge" in capsys.readouterr().err and "recall" not in capsys.readouterr().err
    assert len(_records(legacy)) == 2


def test_from_paired_cannot_stop_after_pair(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _run_pair_cli(monkeypatch, "-i", str(tmp_path / "x.bam"), "-o", str(tmp_path / "y.bam"),
                      "--from-paired", "--stop-after", "pair")
