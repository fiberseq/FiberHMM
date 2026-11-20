"""The array-pair rule for legacy ns/nl and as/al tags (H1, M8) and footprint-only MA (M7).

ns/nl and as/al are calls only when both tags are present, both are integer
arrays and they have the same length; anything else is no calls. An MA/Ma
holding only non-footprint groups (``ddda_mcg.``, ``fire.Q``) is not
authoritative footprint calls.
"""
from __future__ import annotations

import array
import io
from types import SimpleNamespace

import pysam
import pytest

from fiberhmm.io import legacy_tags as lt

LEN = 1000

HEADER = {"HD": {"VN": "1.6", "SO": "coordinate"},
          "SQ": [{"SN": "chr1", "LN": 10_000}],
          "PG": [{"ID": "minimap2", "PN": "minimap2", "VN": "2.28"}]}

# (tag name, value, value_type) lists; value_type None lets pysam infer it.
SCALAR = [("ns", 10, "i")]
SCALAR_PAIR = [("ns", 10, "i"), ("nl", 147, "i")]
STRING = [("ns", "10", "Z"), ("nl", "30", "Z")]
STRING_LIST = [("ns", "100,500", "Z"), ("nl", "147,147", "Z")]
UNEQUAL = [("ns", array.array("I", [100, 500]), None), ("nl", array.array("I", [147]), None)]
NS_ONLY = [("ns", array.array("I", [100, 500]), None)]
EMPTY = [("ns", array.array("I", []), None), ("nl", array.array("I", []), None)]
FLOAT = [("ns", array.array("f", [100.0]), None), ("nl", array.array("f", [147.0]), None)]
INVALID = {"scalar": SCALAR, "scalar_pair": SCALAR_PAIR, "string": STRING, "string_list": STRING_LIST,
           "unequal": UNEQUAL, "ns_only": NS_ONLY, "float": FLOAT}

VALID_NUC = [("ns", array.array("I", [100, 500]), None), ("nl", array.array("I", [147, 147]), None)]
VALID_MSP = [("as", array.array("I", [247, 647]), None), ("al", array.array("I", [253, 200]), None)]


def _read(tags=(), *, reverse=False, header=None):
    hdr = pysam.AlignmentHeader.from_dict(header or HEADER)
    read = pysam.AlignedSegment(hdr)
    read.query_name = "r1"
    read.flag = 16 if reverse else 0
    read.reference_id = 0
    read.reference_start = 100
    read.mapping_quality = 60
    read.cigarstring = f"{LEN}M"
    read.query_sequence = "ACGT" * (LEN // 4)
    read.query_qualities = array.array("B", [30] * LEN)
    for tag, value, vtype in tags:
        read.set_tag(tag, value, value_type=vtype)
    return read


# ------------------------------------------------------------------ the helper
@pytest.mark.parametrize("name", sorted(INVALID))
def test_invalid_pairs_are_absent(name):
    read = _read(INVALID[name])
    assert lt.array_tag_pair(read, "ns", "nl") is None
    assert not lt.has_legacy_pair(read)
    assert not lt.has_legacy_calls(read)


def test_valid_and_empty_pairs():
    read = _read(VALID_NUC)
    assert lt.array_tag_pair(read, "ns", "nl") == ([100, 500], [147, 147])
    assert lt.has_legacy_calls(read)
    empty = _read(EMPTY)
    assert lt.array_tag_pair(empty, "ns", "nl") == ([], [])
    assert lt.has_legacy_pair(empty)
    assert not lt.has_legacy_calls(empty)      # empty arrays are not call evidence
    assert lt.has_legacy_calls(_read(VALID_MSP))


def test_int_array_accepts_lists_not_strings():
    assert lt.int_array([1, 2]) == [1, 2]
    assert lt.int_array((3,)) == [3]
    for bad in (10, "10", b"10", 1.5, None, [1.5], [True], array.array("d", [1.0])):
        assert lt.int_array(bad) is None


def test_ma_footprint_groups():
    assert lt.ma_has_footprint_groups(f"{LEN};nuc+Q:101-147")
    assert lt.ma_has_footprint_groups(f"{LEN};tf+QQQ:101-20")
    assert not lt.ma_has_footprint_groups(f"{LEN};ddda_mcg.:11-2,31-2")
    assert not lt.ma_has_footprint_groups(f"{LEN};fire.Q:101-50")
    assert not lt.ma_has_footprint_groups("not an MA")


def test_qc_states_still_exports_the_rule():
    from fiberhmm.qc.states import _array_tag_pair
    assert _array_tag_pair(_read(VALID_NUC), "ns", "nl")
    assert not _array_tag_pair(_read(SCALAR_PAIR), "ns", "nl")


# ------------------------------------------------------------------ fiberhmm-lift
@pytest.mark.parametrize("name", sorted(INVALID))
def test_lift_passes_invalid_pairs_through(name):
    from fiberhmm.io import graph_lift as gl
    read = _read(INVALID[name], reverse=True)
    before = {t: read.get_tag(t) for t in ("ns", "nl") if read.has_tag(t)}
    assert gl._to_molecular(read) is True               # no TypeError/FrameError
    after = {t: read.get_tag(t) for t in ("ns", "nl") if read.has_tag(t)}
    assert {k: list(v) if not isinstance(v, (int, str)) else v for k, v in after.items()} == \
        {k: list(v) if not isinstance(v, (int, str)) else v for k, v in before.items()}
    assert gl.annotations_valid(read, LEN)
    assert gl.annotation_scope(read, LEN, LEN + 100) == "none"
    gl._require_legacy_frame(read)


def test_lift_string_tags_do_not_become_arrays():
    from fiberhmm.io import graph_lift as gl
    read = _read(STRING, reverse=True)
    gl._to_molecular(read)
    assert read.get_tag("ns") == "10" and read.get_tag("nl") == "30"


def test_lift_shift_skips_invalid_pairs():
    from fiberhmm.io import graph_lift as gl
    read = _read(SCALAR_PAIR + VALID_MSP)
    gl._shift_footprints(read, 50, LEN + 50)
    assert read.get_tag("ns") == 10
    assert list(read.get_tag("as")) == [297, 697]


def test_lift_valid_msp_only_read_is_scoped():
    from fiberhmm.io import graph_lift as gl
    read = _read(VALID_MSP)
    assert gl.annotations_valid(read, LEN)
    assert gl.annotation_scope(read, LEN, LEN + 100) == "partial"


# ------------------------------------------------------------------ fiberhmm-extract
def _extract(read, which):
    from fiberhmm.cli import extract_tags as et
    buf = io.StringIO()
    fn = et._extract_footprints if which == "nuc" else et._extract_msps
    n = fn(read, buf, with_scores=True, block_scores=True)
    return n, buf.getvalue()


@pytest.mark.parametrize("name", sorted(INVALID) + ["empty"])
def test_extract_invalid_pair_gives_no_rows(name):
    tags = EMPTY if name == "empty" else INVALID[name]
    assert _extract(_read(tags), "nuc") == (0, "")
    msp_tags = [("a" + t[1:], v, vt) for t, v, vt in tags]
    assert _extract(_read(msp_tags), "msp") == (0, "")


def test_extract_valid_pairs():
    n, text = _extract(_read(VALID_NUC + [("nq", 7, "i")]), "nuc")   # scalar nq: no scores, no crash
    assert n == 2 and text.count("\n") == 1
    n, _text = _extract(_read(VALID_MSP), "msp")
    assert n == 2


def test_extract_m5c_only_ma_falls_back_to_legacy():
    read = _read(VALID_NUC + VALID_MSP + [("MA", f"{LEN};ddda_mcg.:11-2,31-2", "Z")])
    assert _extract(read, "nuc")[0] == 2
    assert _extract(read, "msp")[0] == 2


def test_extract_footprint_ma_stays_authoritative():
    read = _read(VALID_NUC + [("MA", f"{LEN};nuc+:301-147;ddda_mcg.:11-2", "Z")])
    assert _extract(read, "nuc")[0] == 1


def test_extract_diagnostic_counts_follow_the_rule(tmp_path):
    from fiberhmm.cli import extract_tags as et
    path = str(tmp_path / "d.bam")
    with pysam.AlignmentFile(path, "wb", header=HEADER) as out:
        for tags in (SCALAR_PAIR, STRING, VALID_NUC, VALID_MSP):
            out.write(_read(tags))
    diag = et.diagnose_bam_tags(path, n_reads=10)
    assert diag["has_ns_nl"] == 1
    assert diag["has_as_al"] == 1


# ------------------------------------------------------------------ scan MA / scan reads
@pytest.mark.parametrize("name", sorted(INVALID) + ["empty"])
def test_scan_ma_legacy_invalid_pair_is_not_groups(name):
    from fiberhmm.io.scan_ma import _legacy_as_ma
    tags = EMPTY if name == "empty" else INVALID[name]
    assert _legacy_as_ma(_read(tags)) is None


def test_scan_ma_legacy_msp_only():
    from fiberhmm.io.scan_ma import _legacy_as_ma
    ma, aq = _legacy_as_ma(_read(SCALAR_PAIR + VALID_MSP))
    assert ma == f"{LEN};msp.:248-253,648-200" and aq == []


@pytest.mark.parametrize("name", sorted(INVALID) + ["empty"])
def test_scan_reads_invalid_pair_is_not_callable(name):
    from fiberhmm.scan.reads import has_annotations
    tags = EMPTY if name == "empty" else INVALID[name]
    assert not has_annotations(_read(tags))


def test_scan_reads_callable_evidence():
    from fiberhmm.scan.reads import has_annotations
    assert has_annotations(_read(VALID_MSP))              # as/al-only reads are callable
    assert has_annotations(_read(VALID_NUC))
    assert has_annotations(_read([("MA", f"{LEN};nuc+:101-147", "Z")]))
    assert not has_annotations(_read([("MA", f"{LEN};ddda_mcg.:11-2", "Z")]))
    assert has_annotations(_read([("MA", f"{LEN};ddda_mcg.:11-2", "Z"),
                                  ("Ma", f"{LEN};msp+:101-147", "Z")]))


def test_nfr_nucleosome_runs_follow_the_rules():
    from fiberhmm.scan.modules.nfr import nucleosome_runs
    assert nucleosome_runs(_read(SCALAR_PAIR)) == []
    assert nucleosome_runs(_read(UNEQUAL)) == []
    assert nucleosome_runs(_read(VALID_NUC)) == [(100, 147), (500, 147)]
    assert nucleosome_runs(_read(VALID_NUC + [("MA", f"{LEN};ddda_mcg.:11-2", "Z")])) == \
        [(100, 147), (500, 147)]


@pytest.mark.parametrize("name", sorted(INVALID))
def test_scan_carry_record_passes_invalid_pairs(name):
    from fiberhmm.scan.runner import _carry_record
    read = _read(INVALID[name], reverse=True)
    assert _carry_record(read) is True


# ------------------------------------------------------------------ nuc records (M8)
def _ctx(frame="molecular"):
    return SimpleNamespace(legacy_frame=SimpleNamespace(for_read=lambda _r: frame))


@pytest.mark.parametrize("name", sorted(INVALID) + ["empty"])
def test_nuc_records_invalid_pair_is_no_calls(name):
    from fiberhmm.nuc.records import _footprints_from_legacy
    tags = EMPTY if name == "empty" else INVALID[name]
    assert _footprints_from_legacy(_read(tags), _ctx(), LEN) is None


def test_nuc_records_msp_only_and_mixed():
    from fiberhmm.nuc.records import _footprints_from_legacy
    out = _footprints_from_legacy(_read(SCALAR_PAIR + VALID_MSP), _ctx(), LEN)
    assert out["nuc"] == [] and out["msp"] == [(247, 500), (647, 847)]


# ------------------------------------------------------------------ qc / advisories / pipeline
@pytest.mark.parametrize("name", sorted(INVALID))
def test_qc_footprint_lengths_invalid_pair(name):
    from fiberhmm.qc.core import _footprint_lengths
    assert _footprint_lengths(_read(INVALID[name])) == ([], [], False)


def test_qc_footprint_lengths_rules():
    from fiberhmm.qc.core import _footprint_lengths
    assert _footprint_lengths(_read(VALID_NUC)) == ([147, 147], [], True)
    assert _footprint_lengths(_read(VALID_MSP)) == ([], [], True)
    assert _footprint_lengths(_read([("ns", "1,2,3", "Z"), ("nl", "123", "Z")])) == ([], [], False)
    m5c = _read(VALID_NUC + [("MA", f"{LEN};ddda_mcg.:11-2", "Z")])
    assert _footprint_lengths(m5c) == ([147, 147], [], True)
    assert _footprint_lengths(_read([("MA", f"{LEN};fire.Q:11-20", "Z")])) == ([], [], False)


def _bam(tmp_path, reads_tags):
    path = str(tmp_path / "c.bam")
    with pysam.AlignmentFile(path, "wb", header=HEADER) as out:
        for tags in reads_tags:
            out.write(_read(tags))
    return path


def test_advisory_scan_ignores_invalid_pairs(tmp_path):
    from fiberhmm.advisories import scan_reads
    path = _bam(tmp_path, [SCALAR, SCALAR_PAIR, STRING, UNEQUAL, NS_ONLY,
                           [("MA", f"{LEN};ddda_mcg.:11-2", "Z")], VALID_MSP])
    with pysam.AlignmentFile(path, check_sq=False) as bam:
        scan = scan_reads(bam)
    assert scan.records == 7 and scan.call_tags == 1


def test_pipeline_count_called_follows_the_rules(tmp_path):
    from fiberhmm.pipeline.runner import _count_called
    path = _bam(tmp_path, [SCALAR, STRING, [("MA", f"{LEN};ddda_mcg.:11-2", "Z")],
                           VALID_MSP, [("MA", f"{LEN};nuc+:101-147", "Z")]])
    assert _count_called(path) == {"records": 5, "called": 2, "duplicate_flagged": 0}
