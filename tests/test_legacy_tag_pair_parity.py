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
