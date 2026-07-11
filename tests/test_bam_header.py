"""Tests for BAM-header metadata helpers."""

import pysam
import pytest

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

