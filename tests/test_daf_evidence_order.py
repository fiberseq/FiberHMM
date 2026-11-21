"""DAF evidence order: every tool reads R/Y, then MD, then MM/ML, as fiberhmm-call.

A DAF read aligned with MD that kept basecaller MM/ML tags (Dorado 5mC/6mA)
must be read from MD. fiberhmm-recall-tfs/-nucs, the consensus loader and
strand rescue (all through ``tf_recaller.extract_modification_calls``) used to
read MM/ML first and took the basecaller's marks as deaminations.
"""
from __future__ import annotations

import numpy as np
import pysam
import pytest

from fiberhmm.inference import engine
from fiberhmm.inference.tf_recaller import extract_modification_calls

REF = "ACGTACGTCCAGGTACCATGCAGTCAGTCCATGACGTACGTCCAGGTACCATG"


@pytest.fixture(autouse=True)
def _no_fresh_chimera_detection():
    engine.configure_daf_chimera_filter(False)
    yield
    engine.configure_daf_chimera_filter(True)


def _read(mm=None, ry=False):
    """C->T at reference C positions 1 and 8, visible through MD (or R/Y)."""
    q = list(REF)
    q[1] = q[8] = "Y" if ry else "T"
    seq = "".join(q)
    header = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "c", "LN": 1000}]})
    r = pysam.AlignedSegment(header)
    r.query_name = "r"
    r.query_sequence = seq
    r.flag = 0
    r.reference_id = 0
    r.reference_start = 0
    r.mapping_quality = 60
    r.cigarstring = f"{len(seq)}M"
    r.query_qualities = pysam.qualitystring_to_array("I" * len(seq))
    if ry:
        r.set_tag("st", "CT")
    else:
        r.set_tag("MD", "1C6C%d" % (len(seq) - 9))
    if mm is not None:
        code, ml = mm
        r.set_tag("MM", code)
        r.set_tag("ML", ml)
    return r


DORADO_TAGS = [
    ("C+m,0;", [250]),          # 5mC on the first remaining C
    ("A+a,0;", [250]),          # 6mA on the first A
    ("C+m?,0,1;", [250, 240]),  # 5mC, '?' skip mode
]


def _engine_mods(read):
    fiber = engine._extract_fiber_read_from_pysam(read, "daf", 128)
    return set(fiber["m6a_query_positions"])


def _recall_mods(read):
    return set(extract_modification_calls(read, "daf", 3, prob_threshold=128)[0])


@pytest.mark.parametrize("tags", DORADO_TAGS)
def test_md_wins_over_basecaller_mm_in_call_and_recall(tags):
    read = _read(mm=tags)
    assert _engine_mods(read) == {1, 8}
    assert _recall_mods(read) == {1, 8}


@pytest.mark.parametrize("tags", DORADO_TAGS)
def test_recall_worker_payload_reads_md_first(tags):
    from fiberhmm.cli.recall_tfs import _make_payload, _PayloadRead

    payload = _make_payload(_read(mm=tags), mode="daf")
    stub = _PayloadRead(payload["seq"], payload["is_reverse"], payload["tags"],
                        payload.get("_daf_md_result"),
                        payload.get("_daf_unaligned_query_positions"),
                        payload.get("_daf_insert_mods"))
    assert _recall_mods(stub) == {1, 8}


@pytest.mark.parametrize("tags", DORADO_TAGS)
def test_strand_rescue_and_consensus_observations_ignore_basecaller_mm(tags):
    from fiberhmm.inference.strand_rescue import hard_observations

    with_mm, _ = hard_observations(_read(mm=tags), "daf", "daf", 3, None)
    md_only, _ = hard_observations(_read(), "daf", "daf", 3, None)
    assert np.array_equal(np.asarray(with_mm), np.asarray(md_only))


def test_md_only_and_ry_only_reads_are_unchanged():
    assert _engine_mods(_read()) == _recall_mods(_read()) == {1, 8}
    assert _engine_mods(_read(ry=True)) == _recall_mods(_read(ry=True)) == {1, 8}


def test_mm_native_deaminations_without_md_still_count():
    """No R/Y and no MD: MM/ML-native dU calls remain the evidence."""
    read = _read(mm=("T+u?,0;", [255]))
    read.set_tag("MD", None)
    assert _engine_mods(read) == _recall_mods(read)
    assert _recall_mods(read)
