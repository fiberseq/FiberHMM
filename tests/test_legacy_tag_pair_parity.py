"""Legacy ns/nl/as/al readers follow the shared array-pair rule.

``fiberhmm.io.legacy_tags`` defines when a legacy tag is a footprint call: an
integer array with a partner of the same length. A scalar such as Dorado's
``ns:i`` (a signal sample count) is not a call. Every reader of these tags must
agree with that rule, as fiberhmm-call, extract, scan and QC do.
"""
from __future__ import annotations

from array import array

import pysam
import pytest


def _read(length=100, reverse=False):
    r = pysam.AlignedSegment()
    r.query_name = "legacy"
    r.query_sequence = "A" * length
    r.flag = 16 if reverse else 0
    r.reference_id = 0
    r.reference_start = 0
    r.cigarstring = f"{length}M"
    return r


# --------------------------------------------------------------------------
# inference.legacy_annotations (consensus / scan state loader)
# --------------------------------------------------------------------------

def test_dorado_scalar_ns_is_not_a_legacy_call():
    from fiberhmm.inference.legacy_annotations import has_legacy_tags, legacy_annotations

    r = _read()
    r.set_tag("ns", 5000, value_type="i")
    assert not has_legacy_tags(r)
    # No frame is needed for a read without legacy calls.
    assert legacy_annotations(r, "disabled") is None
    assert legacy_annotations(r, "seq") is None


def test_dorado_scalar_ns_does_not_hide_a_valid_msp_pair():
    from fiberhmm.inference.legacy_annotations import has_legacy_tags, legacy_annotations

    r = _read()
    r.set_tag("ns", 5000, value_type="i")
    r.set_tag("as", array("i", [10]))
    r.set_tag("al", array("i", [20]))
    assert has_legacy_tags(r)
    parsed = legacy_annotations(r, "seq")
    assert [(a["start"], a["length"]) for a in parsed["msp"]] == [(10, 20)]
    assert "nuc" not in parsed


def test_corrupt_integer_pairs_still_fail_loudly():
    from fiberhmm.inference.legacy_annotations import legacy_annotations

    r = _read()
    r.set_tag("ns", array("i", [10]))
    with pytest.raises(ValueError, match="Incomplete"):
        legacy_annotations(r, "seq")
    r.set_tag("nl", array("i", [5, 6]))
    with pytest.raises(ValueError, match="Mismatched"):
        legacy_annotations(r, "seq")


def test_hia5_replay_scaffold_ignores_a_scalar_ns():
    from fiberhmm.inference.consensus.upstream_recall import legacy_tag_intervals

    r = _read()
    r.set_tag("ns", 5000, value_type="i")
    r.set_tag("as", array("i", [10]))
    r.set_tag("al", array("i", [20]))
    assert legacy_tag_intervals(r, "ns", "nl", "seq") == []
    assert legacy_tag_intervals(r, "as", "al", "seq") == [(10, 30)]
    r.set_tag("nl", array("i", [5]))
    r.set_tag("ns", None)
    with pytest.raises(ValueError, match="Missing original"):
        legacy_tag_intervals(r, "ns", "nl", "seq")


# --------------------------------------------------------------------------
# DddA m5C (fiberhmm-tag-m5c / -call-m5c) and fiberhmm-utils transfer
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["scalar", "string", "unequal"])
def test_m5c_legacy_fallback_follows_the_pair_rule(bad):
    from fiberhmm.daf.m5c import _initial_msp_intervals_seq, _nucleosome_intervals_seq

    r = _read()
    if bad == "scalar":
        r.set_tag("ns", 40, value_type="i"); r.set_tag("nl", 20, value_type="i")
        r.set_tag("as", 40, value_type="i"); r.set_tag("al", 20, value_type="i")
    elif bad == "string":
        r.set_tag("ns", "40,60", value_type="Z"); r.set_tag("nl", "5,5", value_type="Z")
        r.set_tag("as", "40,60", value_type="Z"); r.set_tag("al", "5,5", value_type="Z")
    else:
        r.set_tag("ns", array("i", [10, 40])); r.set_tag("nl", array("i", [20]))
        r.set_tag("as", array("i", [10, 40])); r.set_tag("al", array("i", [20]))
    assert _nucleosome_intervals_seq(r, False) == []
    assert _initial_msp_intervals_seq(r, False) == []


def test_m5c_legacy_fallback_reads_a_valid_pair():
    from fiberhmm.daf.m5c import _initial_msp_intervals_seq, _nucleosome_intervals_seq

    r = _read()
    r.set_tag("ns", array("i", [10])); r.set_tag("nl", array("i", [20]))
    r.set_tag("as", array("i", [40])); r.set_tag("al", array("i", [15]))
    assert _nucleosome_intervals_seq(r, False) == [(10, 30)]
    assert _initial_msp_intervals_seq(r, False) == [(40, 55)]


@pytest.mark.parametrize("bad", ["scalar", "string", "unequal"])
def test_utils_transfer_reference_reader_follows_the_pair_rule(bad):
    from fiberhmm.cli.utils import _parse_footprints_from_bam_read

    r = _read()
    if bad == "scalar":
        r.set_tag("ns", 40, value_type="i"); r.set_tag("nl", 20, value_type="i")
    elif bad == "string":
        r.set_tag("ns", "40", value_type="Z"); r.set_tag("nl", "20", value_type="Z")
    else:
        r.set_tag("ns", array("i", [10, 40])); r.set_tag("nl", array("i", [20]))
    assert _parse_footprints_from_bam_read(r, molecular=False) is None
    r.set_tag("ns", array("i", [10])); r.set_tag("nl", array("i", [20]))
    mask = _parse_footprints_from_bam_read(r, molecular=False)
    assert mask.sum() == 20 and mask[10] and not mask[30]


# --------------------------------------------------------------------------
# fiberhmm-recall-tfs / -recall-nucs: MA-only footprints next to Dorado ns:i
# --------------------------------------------------------------------------

def test_recall_reads_ma_footprints_despite_a_scalar_ns():
    from fiberhmm.cli.recall_tfs import _annotation_footprints_as_legacy_tags

    r = _read()
    r.set_tag("MA", "100;nuc.:11-20;msp.:41-15")
    expected = {"ns": [10], "nl": [20], "as": [40], "al": [15]}
    assert _annotation_footprints_as_legacy_tags(r, True) == expected
    r.set_tag("ns", 5000, value_type="i")
    assert _annotation_footprints_as_legacy_tags(r, True) == expected
    # A real legacy pair still wins over MA.
    r.set_tag("ns", array("i", [10])); r.set_tag("nl", array("i", [20]))
    assert _annotation_footprints_as_legacy_tags(r, True) == {}
