"""``fiberhmm-call --hmm-only --keep-msps`` keeps the MA-first MSP selection.

* The MSPs kept are the ones MA/Ma calls (its ``msp`` group, possibly none)
  whenever MA/Ma holds the read's nuc/msp calls. Stale legacy ``as``/``al``
  beside a nuc-only MA must not come back once the ``nuc`` group is stripped
  (before the fix they did: ``msp.:301-100`` was resurrected).
* With ``--legacy-tags`` the rewritten ``as``/``al`` keep a legacy ``aq``
  (fibertools <= 0.12 FIRE) only when that ``aq`` scored exactly the written
  MSPs, re-keyed by interval; otherwise it is removed (before the fix it was
  attached to whatever MSPs were written).
"""
import array

import pysam
import pytest

from fiberhmm.inference.streaming_drain import strip_for_fused_call
from fiberhmm.inference.tagging import kept_seq_msps, write_fused_recall_tags
from fiberhmm.inference.tf_recaller import write_ma_tags

LENGTH = 1000
HEADER = {
    "HD": {"VN": "1.6", "SO": "coordinate"},
    "SQ": [{"SN": "chr1", "LN": 10_000}],
    "PG": [{"ID": "fiberhmm-call", "PN": "fiberhmm-call", "VN": "3.3.0",
            "DS": "FiberHMM fused apply+recall; coord=molecular; mode=pacbio-fiber"}],
    "CO": ["fiberhmm:coord=molecular"],
}


def _read(ma=None, **legacy):
    r = pysam.AlignedSegment(pysam.AlignmentHeader.from_dict(HEADER))
    r.query_name, r.flag, r.reference_id, r.reference_start = "r", 0, 0, 100
    r.mapping_quality = 60
    r.query_sequence, r.cigarstring = "A" * LENGTH, f"{LENGTH}M"
    if ma is not None:
        r.set_tag("MA", ma, value_type="Z")
        if "nuc.Q" in ma:
            r.set_tag("AQ", array.array("B", [7] * ma.count("-")))
    for tag, values in legacy.items():
        r.set_tag(tag, array.array("B" if tag == "aq" else "I", values))
    return r


def _kept_after_strip(read):
    strip_for_fused_call(read, also_write_legacy=False, keep_msps=True)
    return kept_seq_msps(read)


# --- finding: stale legacy MSPs must not be resurrected ------------------------

def test_nuc_only_ma_keeps_no_msps_beside_stale_legacy_msps():
    read = _read("1000;nuc.:101-150", **{"as": [300], "al": [100]})
    assert _kept_after_strip(read) == []
    assert not any(read.has_tag(t) for t in ("as", "al", "aq"))


def test_nuc_only_fibertools_ma_keeps_no_msps_beside_stale_legacy_msps():
    read = _read(None, **{"as": [300], "al": [100], "aq": [201]})
    read.set_tag("Ma", "1000;nuc.:101-150", value_type="Z")
    assert _kept_after_strip(read) == []


def test_ma_msps_win_over_disagreeing_legacy_msps():
    read = _read("1000;nuc.:101-150;msp.:501-200", **{"as": [300], "al": [100], "aq": [201]})
    assert _kept_after_strip(read) == [(500, 200)]
    assert not any(read.has_tag(t) for t in ("as", "al", "aq"))


def test_agreeing_legacy_msps_and_their_fire_scores_stay():
    read = _read("1000;nuc.:101-150;msp.:301-100,501-200",
                 **{"as": [500, 300], "al": [200, 100], "aq": [9, 8]})
    assert _kept_after_strip(read) == [(300, 100), (500, 200)]
    assert list(read.get_tag("aq")) == [9, 8]


def test_legacy_only_records_keep_their_msps():
    read = _read(None, **{"as": [300], "al": [100], "aq": [201]})
    assert _kept_after_strip(read) == [(300, 100)]
    assert list(read.get_tag("aq")) == [201]


def test_tf_only_hmm_only_input_keeps_no_stale_legacy_msps():
    """A --hmm-only record whose footprints are all TF-sized (lone tf group)
    is called by MA: stale legacy MSPs beside it are not kept."""
    global HEADER
    saved = HEADER
    HEADER = dict(saved, PG=[{"ID": "fiberhmm-call", "PN": "fiberhmm-call", "VN": "3.3.0",
                              "DS": "FiberHMM HMM-only call; coord=molecular hmm_only=on"}])
    try:
        read = _read("1000;tf.:101-40", **{"as": [300], "al": [100]})
    finally:
        HEADER = saved
    assert _kept_after_strip(read) == []


# --- finding: legacy aq follows its MSPs or is dropped -------------------------

def _write_kept(read, msps, legacy=True):
    write_ma_tags(read, LENGTH, [], [(0, 147)], msps, also_write_legacy=legacy,
                  msps_replaced=False, remove_legacy=False)


def test_legacy_aq_of_other_msps_is_dropped():
    read = _read(None, **{"as": [300], "al": [100], "aq": [201]})
    _write_kept(read, [(500, 200)])
    assert list(read.get_tag("as")) == [500] and list(read.get_tag("al")) == [200]
    assert not read.has_tag("aq")


def test_legacy_aq_is_reordered_onto_the_same_msps():
    read = _read(None, **{"as": [500, 300], "al": [200, 100], "aq": [9, 8]})
    _write_kept(read, [(300, 100), (500, 200)])
    assert list(read.get_tag("as")) == [300, 500]
    assert list(read.get_tag("aq")) == [8, 9]


def test_legacy_aq_with_a_bad_length_is_dropped():
    read = _read(None, **{"as": [300], "al": [100], "aq": [1, 2]})
    _write_kept(read, [(300, 100)])
    assert not read.has_tag("aq")


def test_ma_only_write_leaves_legacy_msps_and_their_aq_untouched():
    read = _read(None, **{"as": [300], "al": [100], "aq": [201]})
    _write_kept(read, [(300, 100)], legacy=False)
    assert list(read.get_tag("aq")) == [201]


def test_keep_msps_legacy_tags_end_to_end_attaches_no_foreign_fire_score():
    """The audit scenario: MA msp.:501-200, stale scored as/al elsewhere."""
    read = _read("1000;nuc.:101-150;msp.:501-200", **{"as": [300], "al": [100], "aq": [201]})
    strip_for_fused_call(read, also_write_legacy=True, keep_msps=True)
    result = {"ns": [0], "nl": [147], "as": [], "al": [], "tf_calls": []}
    write_fused_recall_tags(read, LENGTH, result, also_write_legacy=True,
                            downstream_compat=False, keep_msps=True)
    assert list(read.get_tag("as")) == [500] and list(read.get_tag("al")) == [200]
    assert not read.has_tag("aq")
    assert "msp.:501-200" in str(read.get_tag("MA"))


@pytest.mark.parametrize("aq", [[201], None])
def test_keep_msps_legacy_tags_end_to_end_keeps_matching_fire_scores(aq):
    legacy = {"as": [500], "al": [200], **({"aq": aq} if aq else {})}
    read = _read("1000;nuc.:101-150;msp.:501-200", **legacy)
    strip_for_fused_call(read, also_write_legacy=True, keep_msps=True)
    result = {"ns": [0], "nl": [147], "as": [], "al": [], "tf_calls": []}
    write_fused_recall_tags(read, LENGTH, result, also_write_legacy=True,
                            downstream_compat=False, keep_msps=True)
    assert list(read.get_tag("as")) == [500]
    assert (list(read.get_tag("aq")) if read.has_tag("aq") else None) == aq
