"""Tests for BAM header provenance helpers."""

import pysam

from fiberhmm.io.bam_header import (
    COORD_MOLECULAR_MARKER,
    _new_pg_record,
    append_coord_marker,
    append_pg_record,
    maybe_append_pg,
)


def _header(pg=None, comments=None):
    data = {
        "HD": {"VN": "1.6"},
        "SQ": [{"SN": "chr1", "LN": 1000}],
    }
    if pg is not None:
        data["PG"] = pg
    if comments is not None:
        data["CO"] = comments
    return pysam.AlignmentHeader.from_dict(data)


def test_append_pg_record_assigns_unique_id_and_chains_previous_pg():
    header = _header(pg=[{"ID": "fiberhmm", "PN": "fiberhmm"}])

    updated = append_pg_record(
        header,
        {"PN": "fiberhmm", "VN": "1.0", "CL": "fiberhmm call", "DS": "desc"},
    )

    pgs = updated.to_dict()["PG"]
    assert [pg["ID"] for pg in pgs] == ["fiberhmm", "fiberhmm.2"]
    assert pgs[1]["PP"] == "fiberhmm"
    assert pgs[1]["PN"] == "fiberhmm"
    assert pgs[1]["VN"] == "1.0"
    assert pgs[1]["CL"] == "fiberhmm call"
    assert pgs[1]["DS"] == "desc"


def test_new_pg_record_builds_unique_id_and_default_program_name():
    record = _new_pg_record(
        [{"ID": "fiberhmm", "PN": "fiberhmm"}],
        {"PN": " ", "VN": " 1.0 ", "CL": " ", "DS": " desc "},
    )

    assert record == {
        "ID": "fiberhmm.2",
        "VN": "1.0",
        "DS": "desc",
        "PP": "fiberhmm",
    }


def test_maybe_append_pg_returns_original_header_without_record():
    header = _header()

    assert maybe_append_pg(header, None) is header


def test_append_coord_marker_is_idempotent():
    header = _header(comments=[COORD_MOLECULAR_MARKER])

    updated = append_coord_marker(header)

    assert updated.to_dict()["CO"] == [COORD_MOLECULAR_MARKER]
