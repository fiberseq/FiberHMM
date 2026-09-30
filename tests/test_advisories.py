"""fiberhmm.advisories / fiberhmm-check: which outputs need re-running.

The three ``*.header.txt`` fixtures (SAM header text) are real headers (paths anonymised):

- ``ont_hia5_2168_frozen_1ca7d0a``: ONT Hia5 called by a development tree at
  1ca7d0a (reports 2.16.8, context-swapped table) -> must be flagged;
- ``ont_hia5_2168_frozen_dc7abba``: the same call by a tree at dc7abba (also
  reports 2.16.8, fixed table) -> must not be flagged;
- ``geo_dddb_2167``: DddB DAF-seq GEO BAM called by 2.16.7 -> DddB table.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pysam
import pytest

from fiberhmm import advisories as adv
from fiberhmm.advisories import ReadScan, check_header, check_path, report

FIXTURES = Path(__file__).parent / "fixtures" / "advisories"
MODELS = Path(adv.__file__).parent / "models"
REPO_ROOT = Path(__file__).resolve().parents[1]
HIA5_FIX = "dc7abba13d8a295b4fdb1f0485680a6d349f9695"
PRE_FIX = "1ca7d0a"  # a recaller commit between v2.16.8 and dc7abba


def _fixture(name):
    return pysam.AlignmentFile(str(FIXTURES / f"{name}.header.txt"), "r",
                               check_sq=False).header.to_dict()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _by_id(advisories):
    return {a.id: a for a in advisories}


def _header(programs, comments=()):
    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 1000}], "PG": []}
    for i, program in enumerate(programs):
        record = dict(program)
        record.setdefault("ID", record["PN"] + (f".{i}" if i else ""))
        header["PG"].append(record)
    if comments:
        header["CO"] = list(comments)
    return header


def _hia5_call(version, cl_extra="", ds_extra="", pid=None):
    record = {
        "PN": "fiberhmm-call", "VN": version,
        "CL": f"fiberhmm-call -i in.bam -o out.bam --enzyme hia5 --seq nanopore {cl_extra}".strip(),
        "DS": f"FiberHMM fused apply+recall; mode=nanopore-fiber enzyme=hia5 {ds_extra}".strip(),
    }
    if pid:
        record["ID"] = pid
    return record


def _declaration(pg, **fields):
    core = {"assay": "fiber-seq", "enzyme": "hia5", "platform": "nanopore",
            "mode": "nanopore-fiber", "pg": pg}
    core.update(fields)
    return "FIBERHMM-CHEMISTRY:v1:" + ";".join(f"{k}={v}" for k, v in core.items())


# --- the index ---------------------------------------------------------------

def test_index_lists_the_legacy_and_bundled_tables():
    index = adv.load_index()
    status = {t["sha256"]: (t["table"], t["status"]) for t in index["tables"]}
    assert status[_sha(MODELS / "legacy" / "hia5_nanopore_gt_swapped_legacy.json")] == (
        "hia5_nanopore", "gt_swapped")
    assert status[_sha(MODELS / "legacy" / "dddb_nanopore_gt_swapped_legacy.json")] == (
        "dddb_nanopore", "gt_swapped")
    assert status[_sha(MODELS / "hia5_nanopore.json")] == ("hia5_nanopore", "fixed")
    # The shipped DddB table may be replaced (refit); it must never be a swapped one.
    assert status.get(_sha(MODELS / "dddb_nanopore.json"), ("", ""))[1] != "gt_swapped"
    swapped_hia5 = [t for t in index["tables"]
                    if t["table"] == "hia5_nanopore" and t["status"] == "gt_swapped"]
    shipped = {r for t in swapped_hia5 for r in t["releases"]}
    assert {"2.0.0", "2.9.1", "2.16.7", "2.16.8"} <= shipped


def test_index_rules_are_well_formed():
    index = adv.load_index()
    assert index["schema"] == adv.INDEX_SCHEMA
    ids = [rule["id"] for rule in index["advisories"]]
    assert len(ids) == len(set(ids))
    for rule in index["advisories"]:
        assert rule["severity"] in adv.SEVERITIES
        assert rule["title"] and rule["reason"] and rule["fix"] and rule["artifact"]
        fix = rule["fixed_in"].get("commit")
        if fix:
            assert any(fix in entry["contains"] for entry in index["commits"].values())
    # Development builds reported 2.16.8 on both sides of the 3.0 fixes.
    assert index["commits"][HIA5_FIX]["version"] == "2.16.8"
    assert HIA5_FIX in index["commits"][HIA5_FIX]["contains"]


def _full_history_available():
    if not (REPO_ROOT / ".git").exists() or shutil.which("git") is None:
        return False
    through = adv.load_index()["commits_through"]

    def git(*args):
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                              text=True)

    shallow = git("rev-parse", "--is-shallow-repository").stdout.strip() == "true"
    return not shallow and git("cat-file", "-e", f"{through}^{{commit}}").returncode == 0


@pytest.mark.skipif(not _full_history_available(),
                    reason="needs a full git checkout containing the indexed commits")
def test_index_tables_match_git_history():
    result = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "build_advisory_index.py"),
                             "--check"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# --- real headers ------------------------------------------------------------

def test_ont_hia5_from_pre_fix_tree_is_flagged():
    found = _by_id(check_header(_fixture("ont_hia5_2168_frozen_1ca7d0a")))
    advisory = found.pop("hia5-nanopore-gt-table")
    assert advisory.severity == "rerun-required"
    assert advisory.status == "affected"
    assert advisory.program == "fiberhmm-call.3"
    assert "1ca7d0a" in advisory.evidence[0] and "predates" in advisory.evidence[0]
    assert not found  # --prob-threshold 248 and --primary were explicit


def test_ont_hia5_from_fixed_dev_tree_reporting_2168_is_clean():
    assert check_header(_fixture("ont_hia5_2168_frozen_dc7abba")) == []


def test_geo_dddb_2167_is_flagged_for_the_dddb_table():
    found = _by_id(check_header(_fixture("geo_dddb_2167")))
    table = found["dddb-gt-table"]
    assert (table.status, table.confidence) == ("affected", "high")
    assert "2.16.7" in table.evidence[0]
    assert found["primary-only-default"].severity == "info"
    assert "hia5-nanopore-gt-table" not in found
    assert "daf-dedup-orientation" not in found  # header only: no read evidence
    scanned = _by_id(check_header(_fixture("geo_dddb_2167"),
                                  scan=ReadScan(records=5000, dedup_tags=61)))
    dedup = scanned["daf-dedup-orientation"]
    assert dedup.status == "possibly_affected" and "di/ds" in dedup.evidence[0]


# --- evidence ----------------------------------------------------------------

@pytest.mark.parametrize("version, status, confidence", [
    ("2.16.7", "affected", "high"),
    ("2.9.1", "affected", "high"),
    ("2.16.8", "possibly_affected", "low"),  # release and fixed dev builds share it
    ("3.0.0", None, None),
])
def test_version_only_evidence(version, status, confidence):
    header = _header([_hia5_call(version, "--prob-threshold 248 --primary")])
    found = _by_id(check_header(header)).get("hia5-nanopore-gt-table")
    if status is None:
        assert found is None
    else:
        assert (found.status, found.confidence) == (status, confidence)


def test_table_digest_decides_regardless_of_version():
    swapped = _sha(MODELS / "legacy" / "hia5_nanopore_gt_swapped_legacy.json")
    fixed = _sha(MODELS / "hia5_nanopore.json")
    call = _hia5_call("3.0.0", "-m old.json --prob-threshold 248 --primary",
                      pid="fiberhmm-call")
    bad = _header([call], [_declaration("fiberhmm-call", apply_sha256=swapped,
                                        recall_sha256=swapped, fiberhmm_version="3.0.0")])
    advisory = _by_id(check_header(bad))["hia5-nanopore-gt-table"]
    assert (advisory.status, advisory.confidence) == ("affected", "high")
    assert swapped[:12] in advisory.evidence[0]
    old = dict(call, VN="2.16.8")
    good = _header([old], [_declaration("fiberhmm-call", apply_sha256=fixed,
                                        recall_sha256=fixed)])
    assert "hia5-nanopore-gt-table" not in _by_id(check_header(good))


def test_declared_commit_beats_the_shared_version_string():
    index = adv.load_index()
    pre = next(sha for sha in index["commits"] if sha.startswith(PRE_FIX))
    call = _hia5_call("2.16.8", "--prob-threshold 248 --primary", pid="fiberhmm-call")
    fixed = _header([call], [_declaration("fiberhmm-call", fiberhmm_commit=HIA5_FIX)])
    assert check_header(fixed) == []
    broken = _header([call], [_declaration("fiberhmm-call", fiberhmm_commit=pre)])
    advisory = _by_id(check_header(broken))["hia5-nanopore-gt-table"]
    assert (advisory.status, advisory.confidence) == ("affected", "high")
    dirty = _header([call], [_declaration("fiberhmm-call", fiberhmm_commit=pre + "+dirty")])
    assert _by_id(check_header(dirty))["hia5-nanopore-gt-table"].confidence == "medium"
    # A commit the index does not know (made after it) falls back to the version.
    later = _header([dict(call, VN="3.0.1")],
                    [_declaration("fiberhmm-call", fiberhmm_commit="f" * 40)])
    assert check_header(later) == []


def test_only_the_current_calls_count():
    old = _hia5_call("2.16.3", pid="fiberhmm-call")
    new = _hia5_call("3.0.0", pid="fiberhmm-call.2")
    assert check_header(_header([old, new])) == []
    # A later recall with an old table still makes the file affected.
    recall = {"PN": "fiberhmm-recall-tfs", "ID": "fiberhmm-recall-tfs", "VN": "2.16.7",
              "CL": "fiberhmm-recall-tfs -i a.bam -o b.bam --enzyme hia5 --seq nanopore",
              "DS": "FiberHMM second-pass footprint refinement; mode=nanopore-fiber enzyme=hia5"}
    found = _by_id(check_header(_header([old, new, recall])))
    assert found["hia5-nanopore-gt-table"].program == "fiberhmm-recall-tfs"


def test_custom_table_on_daf_input_without_enzyme():
    first = {"PN": "fiberhmm-call", "VN": "2.16.7", "CL": "fiberhmm-call --enzyme ddda -i a -o b",
             "DS": "mode=daf enzyme=ddda primary_only=on cpg_mask=off"}
    custom = {"PN": "fiberhmm-recall-tfs", "VN": "2.16.7",
              "CL": "fiberhmm-recall-tfs -i b -o c -m /data/my refit.json",
              "DS": "second pass; mode=daf enzyme=custom"}
    found = _by_id(check_header(_header([first, custom])))
    advisory = found["custom-table-without-enzyme-defaults"]
    assert advisory.status == "affected" and advisory.program.startswith("fiberhmm-recall-tfs")
    explicit = dict(custom, CL=custom["CL"] + " --enzyme ddda")
    assert "custom-table-without-enzyme-defaults" not in _by_id(
        check_header(_header([first, explicit])))


def test_pair_and_tag_m5c_programs():
    pair = {"PN": "fiberhmm-pair", "VN": "2.16.8", "CL": "hybrid pairing", "DS": "mode=hybrid"}
    found = _by_id(check_header(_header([pair])))
    assert found["pair-paired-duplicates"].status == "possibly_affected"
    scan = ReadScan(records=100, pair_tags=40, paired_duplicates=3)
    old_pair = dict(pair, VN="2.16.7")
    found = _by_id(check_header(_header([old_pair]), scan=scan))
    assert found["pair-paired-duplicates"].status == "affected"
    assert check_header(_header([dict(pair, VN="3.0.0")]), scan=scan) == []
    tag = {"PN": "fiberhmm-tag-m5c", "VN": "2.16.8", "CL": "fiberhmm-tag-m5c", "DS": "mCG"}
    assert _by_id(check_header(_header([tag])))["tag-m5c-missing-ucg"].status == "possibly_affected"


def test_info_rules_respect_explicit_choices():
    implicit = _header([_hia5_call("2.16.7")])
    found = _by_id(check_header(implicit))
    assert found["hia5-nanopore-ml-threshold"].severity == "info"
    assert found["primary-only-default"].severity == "info"
    explicit = _header([_hia5_call("2.16.7", "--prob-threshold 200 --no-primary")])
    found = _by_id(check_header(explicit))
    assert "hia5-nanopore-ml-threshold" not in found and "primary-only-default" not in found


def test_untracked_calls_and_empty_headers():
    assert check_header({"HD": {"VN": "1.6"}}) == []
    header = {"HD": {"VN": "1.6"}, "CO": ["MA-TYPES:v1:nuc,msp,tf"]}
    (advisory,) = check_header(header)
    assert advisory.id == "untracked-calls" and advisory.severity == "info"


# --- files ---------------------------------------------------------------------

def _write_bam(path, header, reads=()):
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for name, tags, flag in reads:
            read = pysam.AlignedSegment(out.header)
            read.query_name = name
            read.flag = flag
            read.reference_id = 0
            read.reference_start = 10
            read.mapping_quality = 60
            read.cigarstring = "10M"
            read.query_sequence = "A" * 10
            for tag, value in tags:
                read.set_tag(tag, value)
            out.write(read)


def test_check_bam_reads_dedup_tags_and_qc_sidecar(tmp_path):
    header = _header([_hia5_call("3.0.0", "--prob-threshold 248 --primary")])
    header["PG"][0]["DS"] = "mode=daf enzyme=ddda primary_only=on cpg_mask=off dedup=off"
    header["PG"][0]["CL"] = "fiberhmm-call --enzyme ddda"
    bam = tmp_path / "calls.bam"
    _write_bam(bam, header, [("r1", [("di", 0), ("ds", 2)], 0), ("r2", [("di", 0), ("ds", 2)], 1024)])
    found = _by_id(adv.check_bam(bam))
    assert found["daf-dedup-orientation"].status == "possibly_affected"
    assert "daf-dedup-orientation" not in _by_id(adv.check_bam(bam, scan_records=0))

    (tmp_path / "qc").mkdir()
    sidecar = tmp_path / "qc" / "calls.qc.json"
    sidecar.write_text(json.dumps({"schema_version": 1, "assay": {"mode": "nanopore-fiber"},
                                   "sampling": {}}))
    found = _by_id(adv.check_bam(bam))
    assert found["qc-nanopore-opportunities"].path == str(sidecar)
    assert "qc-nanopore-opportunities" not in _by_id(adv.check_bam(bam, sidecars=False))


def test_fresh_dedup_output_is_not_flagged(tmp_path):
    from fiberhmm.cli.dedup import run_dedup
    from test_dedup import A_SITES, _make_bam

    source = tmp_path / "in.bam"
    _make_bam(source, [(f"A{i}", A_SITES, False, 60) for i in range(3)])
    out = tmp_path / "dedup.bam"
    run_dedup(str(source), str(out), collapse=False)
    assert "daf-dedup-orientation" not in _by_id(adv.check_bam(out))


def test_qc_reports(tmp_path):
    old = tmp_path / "old.qc.json"
    old.write_text(json.dumps({"schema_version": 1, "assay": {"mode": "nanopore-fiber"},
                               "sampling": {}}))
    (advisory,) = check_path(old)
    assert advisory.id == "qc-nanopore-opportunities" and advisory.status == "possibly_affected"
    new = tmp_path / "new.qc.json"
    new.write_text(json.dumps({"schema_version": 1, "fiberhmm_version": "3.0.0",
                               "assay": {"mode": "nanopore-fiber"}, "sampling": {}}))
    assert check_path(new) == []
    pacbio = tmp_path / "pacbio.qc.json"
    pacbio.write_text(json.dumps({"schema_version": 1, "assay": {"mode": "pacbio-fiber"},
                                  "sampling": {}}))
    assert check_path(pacbio) == []


def test_posteriors_files(tmp_path):
    old = tmp_path / "old.tsv.gz"
    with gzip.open(old, "wt") as handle:
        handle.write('#metadata:{"mode": "daf", "format_version": 1}\n')
    (advisory,) = check_path(old)
    assert advisory.id == "posteriors-reverse-frame" and advisory.artifact == "posteriors"
    new = tmp_path / "new.tsv"
    new.write_text('#metadata:{"mode": "daf", "format_version": 1, "fiberhmm_version": "3.0.0"}\n')
    assert check_path(new) == []


def test_consensus_results(tmp_path):
    def result(name, recaller, columns):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "manifest.json").write_text(json.dumps(
            {"schema": "fiberhmm.consensus.v1", "cr_mode": "lattice_recaller",
             "recaller": recaller}))
        (folder / "classes.tsv").write_text("\t".join(columns) + "\n")
        return folder

    before = result("before", {"classes": 3}, ["class_id", "prevalence", "prevalence_edge"])
    (advisory,) = check_path(before)
    assert advisory.id == "recaller-tier-double-count" and advisory.status == "affected"
    fixed = result("fixed", {"classes": 3, "unscored_classes": []},
                   ["class_id", "prevalence", "prevalence_edge"])
    assert check_path(fixed) == []
    no_tiers = result("no_tiers", {"classes": 3}, ["class_id", "prevalence"])
    assert check_path(no_tiers) == []


# --- report / CLI --------------------------------------------------------------

def test_report_shape(tmp_path):
    bam = tmp_path / "x.bam"
    _write_bam(bam, _header([_hia5_call("2.16.7")]))
    payload = report(bam)
    assert payload["schema"] == adv.REPORT_SCHEMA
    assert payload["kind"] == "bam"
    assert payload["status"] == "rerun-required"
    assert payload["needs_rerun"] is True and payload["confirmed"] is True
    first = payload["advisories"][0]
    assert set(first) == {"id", "severity", "status", "confidence", "title", "reason",
                          "artifact", "fix", "fixed_in", "evidence", "program", "path",
                          "needs_rerun"}
    json.dumps(payload)
    missing = report(tmp_path / "missing.bam")
    assert missing["status"] == "error" and missing["error"]


def test_cli_exit_codes_and_json(tmp_path, capsys):
    from fiberhmm.cli.check import main

    clean = tmp_path / "clean.bam"
    _write_bam(clean, _header([_hia5_call("3.0.0", "--prob-threshold 248 --primary")]))
    stale = tmp_path / "stale.bam"
    _write_bam(stale, _header([_hia5_call("2.16.7", "--prob-threshold 248 --primary")]))
    info_only = tmp_path / "info.bam"
    _write_bam(info_only, _header([{
        "PN": "fiberhmm-call", "VN": "3.0.0", "CL": "fiberhmm-call --enzyme hia5 --seq pacbio",
        "DS": "mode=pacbio-fiber enzyme=hia5"}]))
    assert main([str(clean)]) == 0
    assert main([str(info_only)]) == 0
    assert main([str(clean), str(stale)]) == 3
    capsys.readouterr()
    assert main([str(stale), "--json"]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "fiberhmm.advisory_check.v1"
    assert payload["reports"][0]["advisories"][0]["id"] == "hia5-nanopore-gt-table"
    assert main([str(tmp_path / "missing.bam"), str(stale)]) == 2
    assert main(["--list"]) == 0


def test_nanopore_reads_called_as_pacbio_without_seq():
    aligner = {"PN": "minimap2", "ID": "minimap2", "VN": "2.28",
               "CL": "minimap2 -y -a -x map-ont ref.mmi reads.fq"}
    call = {"PN": "fiberhmm-call", "ID": "fiberhmm-call", "VN": "2.16.7",
            "CL": "fiberhmm-call -i a.bam -o b.bam --enzyme hia5 --prob-threshold 248 --primary",
            "DS": "FiberHMM fused apply+recall; mode=pacbio-fiber enzyme=hia5"}
    found = _by_id(check_header(_header([aligner, call])))
    advisory = found["hia5-nanopore-called-as-pacbio"]
    assert (advisory.status, advisory.severity) == ("affected", "rerun-required")
    assert "map-ont" in advisory.evidence[0]
    assert "hia5-nanopore-gt-table" not in found  # the PacBio table was used
    pacbio = dict(aligner, PN="pbmm2", ID="pbmm2", CL="pbmm2 align --preset CCS")
    assert "hia5-nanopore-called-as-pacbio" not in _by_id(check_header(_header([pacbio, call])))
    fixed = dict(call, VN="3.0.0")
    assert "hia5-nanopore-called-as-pacbio" not in _by_id(check_header(_header([aligner, fixed])))
    assert adv.sequencing_platform(_fixture("geo_dddb_2167"))[0] == "nanopore"


def test_samtools_merged_histories():
    pacbio = {"PN": "fiberhmm-call", "ID": "fiberhmm-call", "VN": "2.13.0",
              "CL": "fiberhmm-call --enzyme hia5 --seq pacbio --primary",
              "DS": "mode=pacbio-fiber"}
    ont = {"PN": "fiberhmm-call", "ID": "fiberhmm-call-75E22241", "VN": "2.16.3",
           "CL": "fiberhmm-call --enzyme hia5 --seq nanopore --prob-threshold 248 --primary",
           "DS": "mode=nanopore-fiber enzyme=hia5"}
    # Merged-in histories of another chemistry: cannot say which reads.
    advisory = _by_id(check_header(_header([pacbio, ont])))["hia5-nanopore-gt-table"]
    assert (advisory.status, advisory.confidence) == ("possibly_affected", "low")
    assert "merges" in advisory.evidence[-1]
    # Merged replicates of one chemistry: every history counts.
    ont_first = dict(ont, ID="fiberhmm-call")
    advisory = _by_id(check_header(_header([ont_first, ont])))["hia5-nanopore-gt-table"]
    assert advisory.status == "affected"
    # A FiberHMM run after the merge re-calls every read.
    recall = _hia5_call("3.0.0", "--prob-threshold 248 --primary", pid="fiberhmm-call.2")
    assert check_header(_header([pacbio, ont, recall])) == []


@pytest.mark.parametrize("version, cl, expected", [
    # An input directory named after a fixed commit cannot clear an old call ...
    ("2.16.7", "fiberhmm-call -i /data/frozen_dc7abba/input.bam -o out.bam",
     ("affected", "high")),
    ("2.16.7", "/opt/env/bin/fiberhmm-call -i /data/frozen_dc7abba/input.bam -o o.bam",
     ("affected", "high")),
    # ... and one named after a pre-fix commit cannot condemn a fixed release.
    ("3.0.0", "fiberhmm-call -i /data/frozen_1ca7d0a/input.bam -o out.bam", None),
    ("3.0.0", "/usr/bin/fiberhmm-call -i a.bam -o /out/frozen_1ca7d0a/b.bam", None),
    # The script path itself (python -m / a source tree, spaces allowed) still counts.
    ("3.0.0", "/work/Fiber NET/frozen_1ca7d0a_p29/fiberhmm/cli/call.py -i a.bam -o b.bam",
     ("affected", "medium")),
])
def test_code_commit_comes_from_the_program_path_only(version, cl, expected):
    record = {"PN": "fiberhmm-call", "ID": "fiberhmm-call", "VN": version,
              "CL": cl + " --enzyme hia5 --seq nanopore --primary --prob-threshold 248",
              "DS": "mode=nanopore-fiber enzyme=hia5"}
    found = _by_id(check_header(_header([record]))).get("hia5-nanopore-gt-table")
    if expected is None:
        assert found is None
    else:
        assert (found.status, found.confidence) == expected
        if expected[1] == "medium":
            assert "1ca7d0a" in found.evidence[0]
