"""fiberhmm-check on outputs it cannot vouch for (audit H8, L11-L13).

- Calls with no FiberHMM provenance (the ind Hia5 BAMs carry calls from an old
  FiberHMM run that wrote no @PG) were reported "no re-run needed", exit 0.
  They cannot be verified: the report says so, recommends a re-run if they
  came from FiberHMM < 3.0, and exits 4 (a suggestion, not an error).
- A BAM with nothing FiberHMM made was reported "clean".
- A fiberhmm-pipeline OUTDIR was an error ("no consensus manifest").
- A QC report graded under a misdetected assay (mode=flag from fiberhmm-dedup's
  @PG) was "clean".
"""
from __future__ import annotations

import json
import shutil

import pysam
import pytest

from fiberhmm import advisories as adv
from fiberhmm.advisories import check_path, report
from fiberhmm.cli.check import EXIT_UNVERIFIABLE, main

# The header shape of the stale ind Hia5 BAM: PacBio tools, then the samtools
# cat FiberHMM 2.x ran over its own region files, and no FiberHMM @PG.
STALE_PG = [
    {"PN": "pbmm2", "ID": "pbmm2", "VN": "1.17.0", "CL": "pbmm2 align --preset CCS dm6.mmi x.bam y.bam"},
    {"PN": "samtools", "ID": "samtools", "VN": "1.21", "PP": "pbmm2",
     "CL": "samtools cat -b /tmp/fiberhmm_2-4hr_4/.fiberhmm_tmp/bam_list.txt "
           "-o /tmp/fiberhmm_2-4hr_4/2-4hr_4.aligned_footprints.bam"},
]
CURRENT_CALL = {
    "PN": "fiberhmm-call", "ID": "fiberhmm-call", "VN": "3.0.0",
    "CL": "fiberhmm-call --enzyme hia5 --seq pacbio --prob-threshold 125 --primary",
    "DS": "mode=pacbio-fiber enzyme=hia5",
}
OLD_ONT_CALL = {
    "PN": "fiberhmm-call", "ID": "fiberhmm-call", "VN": "2.16.7",
    "CL": "fiberhmm-call --enzyme hia5 --seq nanopore --prob-threshold 248 --primary",
    "DS": "mode=nanopore-fiber enzyme=hia5",
}


def _bam(path, programs=(), comments=(), call_tags=True, records=2):
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 1000}]}
    if programs:
        header["PG"] = [dict(p) for p in programs]
    if comments:
        header["CO"] = list(comments)
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for i in range(records):
            read = pysam.AlignedSegment(out.header)
            read.query_name = f"r{i}"
            read.reference_id = 0
            read.reference_start = 10
            read.mapping_quality = 60
            read.cigarstring = "50M"
            read.query_sequence = "A" * 50
            if call_tags:
                read.set_tag("ns", [5])
                read.set_tag("nl", [20])
            out.write(read)
    return path


def test_calls_without_provenance_are_unverifiable_not_clean(tmp_path, capsys):
    stale = _bam(tmp_path / "stale.bam", STALE_PG)
    payload = report(stale)
    assert payload["status"] == "unverifiable"
    assert payload["unverifiable"] is True and payload["needs_rerun"] is False
    (advisory,) = payload["advisories"]
    assert advisory["id"] == "untracked-calls" and advisory["severity"] == "unverifiable"
    # The 2.x work path in the samtools @PG names an earlier FiberHMM run.
    assert advisory["confidence"] == "medium"
    assert any(".fiberhmm_tmp/bam_list.txt" in line for line in advisory["evidence"])

    assert main([str(stale)]) == EXIT_UNVERIFIABLE == 4
    out = capsys.readouterr().out
    assert "no re-run needed" not in out
    assert "CANNOT VERIFY" in out and "re-run recommended if they came from FiberHMM < 3.0" in out


def test_untracked_calls_without_a_fiberhmm_trace_keep_low_confidence(tmp_path):
    bare = _bam(tmp_path / "bare.bam", [STALE_PG[0]])
    (advisory,) = adv.check_bam(bare)
    assert (advisory.severity, advisory.confidence) == ("unverifiable", "low")
    assert not advisory.needs_rerun


def test_exit_status_precedence(tmp_path):
    stale = _bam(tmp_path / "stale.bam", STALE_PG)
    current = _bam(tmp_path / "current.bam", [CURRENT_CALL])
    old = _bam(tmp_path / "old.bam", [OLD_ONT_CALL])
    assert main([str(current)]) == 0
    assert main([str(current), str(stale)]) == 4
    assert main([str(stale), str(old)]) == 3  # a known re-run outranks "cannot verify"
    assert main([str(stale), str(tmp_path / "missing.bam")]) == 2


def test_raw_and_header_only_bams_are_not_fiberhmm_outputs(tmp_path, capsys):
    raw = _bam(tmp_path / "raw.bam", [STALE_PG[0]], call_tags=False)
    empty = _bam(tmp_path / "empty.bam", records=0)
    for path in (raw, empty):
        payload = report(path)
        assert payload["status"] == "not-fiberhmm" and payload["advisories"] == []
    assert main([str(raw), str(empty)]) == 0
    assert "not a FiberHMM output" in capsys.readouterr().out
    # Without a record scan, reads may hold calls the header does not show.
    assert report(raw, scan_records=0)["status"] == "clean"
    # Anything FiberHMM wrote counts (a fiberhmm-call header with no records).
    assert report(_bam(tmp_path / "called.bam", [CURRENT_CALL], records=0))["status"] == "clean"


def _qc(path, mode, version="3.0.0"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 1, "fiberhmm_version": version, "input": "/x/ddda.fiberhmm.bam",
        "assay": {"mode": mode, "enzyme": "ddda", "reference_profile": None},
        "sampling": {}}))
    return path


def test_qc_report_with_a_misdetected_assay(tmp_path):
    flagged = _qc(tmp_path / "flag.qc.json", "flag")
    (advisory,) = check_path(flagged)
    assert advisory.id == "qc-assay-misdetected"
    assert (advisory.status, advisory.severity) == ("affected", "rerun-recommended")
    assert "'flag'" in advisory.evidence[0] and "ddda.fiberhmm.bam" in advisory.evidence[0]
    assert check_path(_qc(tmp_path / "daf.qc.json", "daf")) == []
    combined = tmp_path / "combined.qc.json"
    combined.write_text(json.dumps({"report_type": "fiberhmm_multi_sample_qc", "samples": [
        json.loads((tmp_path / "daf.qc.json").read_text()), json.loads(flagged.read_text())]}))
    (advisory,) = check_path(combined)
    assert advisory.id == "qc-assay-misdetected"


def _pipeline(outdir, *, qc_mode="flag"):
    outdir.mkdir(parents=True)
    called = _bam(outdir / "ddda.fiberhmm.bam", [CURRENT_CALL])
    qc = _qc(outdir / "qc" / "ddda.fiberhmm.qc.json", qc_mode)
    (outdir / "outputs.json").write_text(json.dumps({
        "schema": "fiberhmm.pipeline.outputs.v1", "version": "3.0.0",
        "called_bam": str(called), "qc_report": str(qc), "qc": {"json": str(qc)}}))
    return outdir


def test_pipeline_output_directory_checks_its_bam_and_qc(tmp_path, capsys):
    outdir = _pipeline(tmp_path / "run")
    assert adv.output_kind(outdir) == "pipeline"
    assert adv.output_kind(outdir / "outputs.json") == "pipeline"
    payload = report(outdir)
    assert payload["kind"] == "pipeline" and payload["status"] == "rerun-recommended"
    (advisory,) = payload["advisories"]
    assert advisory["id"] == "qc-assay-misdetected"
    assert advisory["path"].endswith("qc/ddda.fiberhmm.qc.json")
    assert main([str(outdir)]) == 3
    capsys.readouterr()

    # A moved OUTDIR: the recorded absolute paths are gone, the files are local.
    moved = tmp_path / "moved"
    shutil.move(str(outdir), str(moved))
    assert report(moved)["status"] == "rerun-recommended"
    clean = _pipeline(tmp_path / "clean", qc_mode="daf")
    assert report(clean)["status"] == "clean"
    (clean / "ddda.fiberhmm.bam").unlink()
    broken = report(clean)
    assert broken["status"] == "error" and "no longer exists" in broken["error"]


def test_index_declares_every_severity():
    index = adv.load_index()
    assert list(index["severities"]) == list(adv.SEVERITIES)
    assert adv.RERUN_SEVERITIES == ("rerun-required", "rerun-recommended")
    assert {r["id"] for r in index["advisories"]} >= {"untracked-calls", "qc-assay-misdetected"}
    assert index["revision"] >= 2
