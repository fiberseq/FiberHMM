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
