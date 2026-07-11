"""Tests for BAM-header metadata helpers."""

import sys

import pysam
import pytest

from fiberhmm.cli import utils
from fiberhmm.io.bam_header import (
    append_ma_types,
    declared_ma_types,
    ma_types_from_tag,
)


def _header(*comments):
    return pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6"},
        "SQ": [{"SN": "chr1", "LN": 1000}],
        "CO": list(comments),
    })


def test_declared_ma_types_unions_valid_comments_in_first_seen_order():
    header = _header(
        "unrelated comment",
        "MA-TYPES:v1:nuc,msp",
        "MA-TYPES:v2:future",
        "MA-TYPES:v1:tf,nuc",
        "MA-TYPES:v1:nuc,tf.QQQ",
        "MA-TYPES:v1:",
        "MA-TYPES:v1:ddda_mcg,TF",
    )

    assert declared_ma_types(header) == ["nuc", "msp", "tf", "ddda_mcg", "TF"]


def test_append_ma_types_preserves_comments_and_appends_only_missing_names():
    header = _header("keep me", "MA-TYPES:v1:nuc,msp")

    updated = append_ma_types(header, ["msp", "tf", "ddda_mcg", "tf"])
    comments = updated.to_dict()["CO"]
    assert comments == [
        "keep me",
        "MA-TYPES:v1:nuc,msp",
        "MA-TYPES:v1:tf,ddda_mcg",
    ]
    assert declared_ma_types(updated) == ["nuc", "msp", "tf", "ddda_mcg"]
    assert "@CO\tMA-TYPES:v1:tf,ddda_mcg" in str(updated)

    repeated = append_ma_types(updated, ["nuc", "msp", "tf", "ddda_mcg"])
    assert repeated.to_dict()["CO"] == comments


@pytest.mark.parametrize("name", ["tf.QQQ", "tf+", "with space", "", "a,b"])
def test_append_ma_types_rejects_non_logical_names(name):
    with pytest.raises(ValueError, match="invalid MA annotation name"):
        append_ma_types(_header(), [name])


def test_ma_types_from_tag_discards_suffixes_and_ignores_empty_invalid_sections():
    ma = (
        "1000;nuc.Q:1-10;msp.:21-5;tf-QQQ:31-6;"
        "ddda_mcg+:41-7;nuc.P:51-8;empty.:;bad-name.:61-9;broken"
    )
    assert ma_types_from_tag(ma) == ["nuc", "msp", "tf", "ddda_mcg"]


def _write_ma_bam(path):
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": 1000}],
        "CO": ["keep me", "MA-TYPES:v1:nuc"],
    }
    ma_values = [
        "100;nuc.Q:1-10;tf.QQQ:21-5",
        "100;ddda_mcg-:31-7;ddda_mcg_hemi+:51-6",
        "100;empty.:",
    ]
    with pysam.AlignmentFile(path, "wb", header=header) as bam:
        for index, ma in enumerate(ma_values):
            read = pysam.AlignedSegment(bam.header)
            read.query_name = f"read{index}"
            read.query_sequence = "A" * 100
            read.flag = 0
            read.reference_id = 0
            read.reference_start = index * 200
            read.mapping_quality = 60
            read.cigar = ((0, 100),)
            read.set_tag("MA", ma, value_type="Z")
            bam.write(read)
    pysam.index(str(path))


def _bam_records(path):
    with pysam.AlignmentFile(path, "rb", check_sq=False) as bam:
        return [
            (read.query_name, read.reference_start, read.get_tag("MA"))
            for read in bam.fetch(until_eof=True)
        ]


def test_exhaustive_scan_finds_rare_types_in_record_order(tmp_path):
    bam_path = tmp_path / "scan.bam"
    _write_ma_bam(bam_path)

    names, records, with_ma = utils._scan_bam_ma_types(str(bam_path), io_threads=1)

    assert names == ["nuc", "tf", "ddda_mcg", "ddda_mcg_hemi"]
    assert (records, with_ma) == (3, 3)


def test_repair_ma_types_updates_bam_and_existing_index_in_place(tmp_path):
    bam_path = tmp_path / "repair.bam"
    _write_ma_bam(bam_path)
    before_records = _bam_records(bam_path)

    names, _records, _with_ma = utils._scan_bam_ma_types(str(bam_path), io_threads=1)
    missing, copied, rebuilt = utils._rewrite_bam_ma_types_in_place(
        str(bam_path), names, io_threads=1,
    )

    assert missing == ["tf", "ddda_mcg", "ddda_mcg_hemi"]
    assert copied == 3
    assert rebuilt == [str(bam_path) + ".bai"]
    assert _bam_records(bam_path) == before_records
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        assert bam.has_index()
        assert len(list(bam.fetch("chr1", 0, 1000))) == 3
        assert bam.header.to_dict()["CO"] == [
            "keep me",
            "MA-TYPES:v1:nuc",
            "MA-TYPES:v1:tf,ddda_mcg,ddda_mcg_hemi",
        ]
        assert declared_ma_types(bam.header) == [
            "nuc", "tf", "ddda_mcg", "ddda_mcg_hemi",
        ]

    before_bam = bam_path.read_bytes()
    before_index = (tmp_path / "repair.bam.bai").read_bytes()
    assert utils._rewrite_bam_ma_types_in_place(
        str(bam_path), names, io_threads=1,
    ) == ([], 0, [])
    assert bam_path.read_bytes() == before_bam
    assert (tmp_path / "repair.bam.bai").read_bytes() == before_index


def test_ma_types_cli_accepts_explicit_case_sensitive_custom_names(
    tmp_path, monkeypatch, capsys,
):
    bam_path = tmp_path / "explicit.bam"
    _write_ma_bam(bam_path)
    monkeypatch.setattr(sys, "argv", [
        "fiberhmm-utils", "ma-types", str(bam_path),
        "--types", "tf,Rare_Type", "--io-threads", "1",
    ])

    assert utils.main() is None

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        assert declared_ma_types(bam.header) == ["nuc", "tf", "Rare_Type"]
    assert "added: tf,Rare_Type" in capsys.readouterr().out


def test_repair_rebuilds_an_existing_csi_index(tmp_path):
    bam_path = tmp_path / "csi.bam"
    _write_ma_bam(bam_path)
    (tmp_path / "csi.bam.bai").unlink()
    pysam.index("-c", str(bam_path))

    missing, copied, rebuilt = utils._rewrite_bam_ma_types_in_place(
        str(bam_path), ["tf"], io_threads=1,
    )

    assert (missing, copied) == (["tf"], 3)
    assert rebuilt == [str(bam_path) + ".csi"]
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        assert bam.has_index()
        assert len(list(bam.fetch("chr1"))) == 3


def test_repair_rejects_suffixes_before_mutating_bam(tmp_path):
    bam_path = tmp_path / "invalid.bam"
    _write_ma_bam(bam_path)
    before = bam_path.read_bytes()

    with pytest.raises(ValueError, match="logical names only"):
        utils._rewrite_bam_ma_types_in_place(
            str(bam_path), ["tf.QQQ"], io_threads=1,
        )
    assert bam_path.read_bytes() == before
