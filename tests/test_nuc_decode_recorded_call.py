"""fiberhmm.nuc decodes marks with the threshold and edge trim the call used.

``DecodeContext`` (fiberhmm-scan nuc module, nuc API) and
``CallerSettings.from_header`` (nuc recall replay) read the same header; both
must take ``--prob-threshold`` and ``--edge-trim`` from the last fiberhmm-call
record, so the marks and informative sites match the ones the caller saw.
"""
from __future__ import annotations

import pysam

from fiberhmm.nuc.params import NucParams
from fiberhmm.nuc.recall import CallerSettings
from fiberhmm.nuc.records import DecodeContext


def _header(cl):
    return pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6"},
        "SQ": [{"SN": "chr1", "LN": 10_000}],
        "PG": [{"ID": "fiberhmm-call", "PN": "fiberhmm-call", "CL": cl}],
    })


def test_recorded_threshold_and_edge_trim_are_used():
    h = _header("fiberhmm-call -i in.bam -o out.bam --enzyme hia5 --seq pacbio "
                "--prob-threshold 200 --edge-trim 25")
    ctx = DecodeContext(h)
    settings = CallerSettings.from_header(h)
    assert (settings.prob_threshold, settings.edge_trim) == (200, 25)
    assert ctx.mark_threshold == settings.prob_threshold
    assert ctx.edge_trim == settings.edge_trim


def test_defaults_without_recorded_options():
    h = _header("fiberhmm-call -i in.bam -o out.bam --enzyme hia5 --seq nanopore")
    ctx = DecodeContext(h)
    assert ctx.mark_threshold == CallerSettings.from_header(h).prob_threshold == 248
    assert ctx.edge_trim == 10


def test_explicit_params_win_over_the_record():
    h = _header("fiberhmm-call -i in.bam -o out.bam --enzyme hia5 --seq pacbio "
                "--prob-threshold 200 --edge-trim 25")
    ctx = DecodeContext(h, NucParams(mark_threshold=150, edge_trim=5))
    assert (ctx.mark_threshold, ctx.edge_trim) == (150, 5)


def _header_pgs(*pgs):
    return pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6"},
        "SQ": [{"SN": "chr1", "LN": 10_000}],
        "PG": list(pgs),
    })


_CALL_CL = ("fiberhmm-call -i in.bam -o out.bam --enzyme hia5 --seq pacbio "
            "--prob-threshold 200 --edge-trim 25")


def test_a_later_call_m5c_record_does_not_hide_the_call():
    h = _header_pgs(
        {"ID": "fiberhmm-call", "PN": "fiberhmm-call", "CL": _CALL_CL},
        {"ID": "fiberhmm-call-m5c", "PN": "fiberhmm-call-m5c", "PP": "fiberhmm-call",
         "CL": "fiberhmm-call-m5c -i out.bam --tag-bam tagged.bam"},
    )
    ctx = DecodeContext(h)
    settings = CallerSettings.from_header(h)
    assert (ctx.mark_threshold, ctx.edge_trim) == (200, 25)
    assert (settings.prob_threshold, settings.edge_trim) == (200, 25)


def test_call_record_selector_is_exact():
    from fiberhmm.nuc.records import is_call_record
    assert is_call_record({"ID": "fiberhmm-call", "PN": "fiberhmm-call"})
    assert is_call_record({"ID": "fiberhmm-call.2", "PN": "fiberhmm-call"})
    assert is_call_record({"ID": "fiberhmm-call.1"})                  # no PN
    assert not is_call_record({"ID": "fiberhmm-call-m5c", "PN": "fiberhmm-call-m5c"})
    assert not is_call_record({"ID": "fiberhmm-call-m5c.1"})
    assert not is_call_record({"ID": "fiberhmm-call", "PN": "fiberhmm-call-m5c"})


def test_generated_id_suffix_without_pn_is_read():
    h = _header_pgs({"ID": "fiberhmm-call.1", "CL": _CALL_CL})
    assert DecodeContext(h).edge_trim == 25
    assert CallerSettings.from_header(h).edge_trim == 25


def test_an_explicit_edge_trim_of_10_wins_over_the_record():
    """10 is the shared default, but given explicitly it is still honoured."""
    h = _header(_CALL_CL)
    assert DecodeContext(h, NucParams(edge_trim=10)).edge_trim == 10
    assert DecodeContext(h, NucParams.from_dict({'edge_trim': 10})).edge_trim == 10
    assert DecodeContext(h, NucParams(edge_trim=0)).edge_trim == 0


def test_omitted_edge_trim_survives_a_round_trip():
    """Unset stays unset through to_dict/from_dict (scan settings) and
    pickling, so the recorded trim still applies."""
    import json
    import pickle
    h = _header(_CALL_CL)
    params = NucParams.from_dict(json.loads(json.dumps(NucParams().to_dict())))
    assert DecodeContext(h, params).edge_trim == 25
    assert DecodeContext(h, pickle.loads(pickle.dumps(NucParams()))).edge_trim == 25


def test_evidence_replay_reads_the_call_record_past_call_m5c():
    from fiberhmm.daf.replay import EvidenceReplay, evidence_token
    h = _header_pgs(
        {"ID": "fiberhmm-call", "PN": "fiberhmm-call", "CL": _CALL_CL,
         "DS": "mode=daf " + evidence_token()},
        {"ID": "fiberhmm-call-m5c", "PN": "fiberhmm-call-m5c", "PP": "fiberhmm-call",
         "CL": "fiberhmm-call-m5c -i out.bam --tag-bam tagged.bam"},
    )
    assert EvidenceReplay(h, "daf").portable
