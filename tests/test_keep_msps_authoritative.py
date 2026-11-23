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

from fiberhmm.inference.streaming_drain import strip_for_fused_call
from fiberhmm.inference.tagging import kept_seq_msps

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
