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


def _header(programs, comments=(), linked=True):
    """Header dict; ``linked`` chains each @PG to the one before (PP), as
    FiberHMM's append_pg_record and htslib do. A record's own PP (or
    ``"PP": None`` for a root) is kept."""
    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 1000}], "PG": []}
    for i, program in enumerate(programs):
        record = dict(program)
        record.setdefault("ID", record["PN"] + (f".{i}" if i else ""))
        if linked and i and "PP" not in record:
            record["PP"] = header["PG"][-1]["ID"]
        if record.get("PP") is None:
            record.pop("PP", None)
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
    assert advisory.id == "untracked-calls" and advisory.severity == "unverifiable"


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
    """samtools merge keeps each input's PP chain (clashing IDs get -XXXXXXXX)
    and appends one merge record per chain end; every branch's calls count."""
    pacbio = {"PN": "fiberhmm-call", "ID": "fiberhmm-call", "VN": "2.13.0",
              "CL": "fiberhmm-call --enzyme hia5 --seq pacbio --primary",
              "DS": "mode=pacbio-fiber"}
    ont = {"PN": "fiberhmm-call", "ID": "fiberhmm-call-75E22241", "VN": "2.16.3",
           "CL": "fiberhmm-call --enzyme hia5 --seq nanopore --prob-threshold 248 --primary",
           "DS": "mode=nanopore-fiber enzyme=hia5", "PP": None}
    merge = {"PN": "samtools", "VN": "1.21", "CL": "samtools merge -o m.bam a.bam b.bam"}
    merged = [pacbio, ont, dict(merge, ID="samtools", PP="fiberhmm-call"),
              dict(merge, ID="samtools.1", PP="fiberhmm-call-75E22241")]
    # A merged-in history of another chemistry: its reads are affected.
    advisory = _by_id(check_header(_header(merged)))["hia5-nanopore-gt-table"]
    assert (advisory.status, advisory.confidence) == ("affected", "high")
    assert advisory.program == "fiberhmm-call-75E22241"
    assert "merges" in advisory.evidence[-1]
    # Merged replicates of one chemistry: every history counts.
    ont_first = dict(ont, ID="fiberhmm-call", PP=None)
    replicates = [ont_first, dict(ont, PP=None),
                  dict(merge, ID="samtools", PP="fiberhmm-call"),
                  dict(merge, ID="samtools.1", PP="fiberhmm-call-75E22241")]
    assert _by_id(check_header(_header(replicates)))["hia5-nanopore-gt-table"].status == "affected"
    # A FiberHMM run after the merge (linked to the last merge record only)
    # re-calls every read: the merge event joins both chains.
    recall = _hia5_call("3.0.0", "--prob-threshold 248 --primary", pid="fiberhmm-call.2")
    assert check_header(_header([*merged, recall])) == []
    # The same run without the merge event would leave the other branch current.
    split = [pacbio, ont, dict(merge, ID="samtools", PP="fiberhmm-call-75E22241"),
             dict(recall, PP="samtools")]
    split_found = _by_id(check_header(_header(split, linked=False)))
    assert "hia5-nanopore-gt-table" not in split_found  # the PacBio branch is not ONT Hia5
    ont_split = [dict(ont, ID="fiberhmm-call", PP=None), dict(ont, PP=None),
                 dict(merge, ID="samtools", PP="fiberhmm-call-75E22241"),
                 dict(recall, PP="samtools")]
    advisory = _by_id(check_header(_header(ont_split, linked=False)))["hia5-nanopore-gt-table"]
    assert advisory.status == "affected" and advisory.program == "fiberhmm-call"


def test_unlinked_history_is_never_cleared():
    """Records without PP could be a history written without links or a merge:
    an old call either superseded or merged in is possibly affected, not clean."""
    old = _hia5_call("2.16.3", "--prob-threshold 248 --primary", pid="fiberhmm-call")
    new = _hia5_call("3.0.0", "--prob-threshold 248 --primary", pid="fiberhmm-call.2")
    advisory = _by_id(check_header(_header([old, new], linked=False)))["hia5-nanopore-gt-table"]
    assert (advisory.status, advisory.confidence) == ("possibly_affected", "low")
    assert "no PP link" in advisory.evidence[-1]
    # Linked, the later call supersedes the earlier one.
    assert check_header(_header([old, new])) == []


def _merged_bam(tmp_path, name, inputs):
    paths = []
    for i, (header, reads) in enumerate(inputs):
        path = tmp_path / f"{name}_in{i}.bam"
        _write_bam(path, header, reads)
        paths.append(str(path))
    out = tmp_path / f"{name}.bam"
    pysam.merge("-f", str(out), *paths)
    return out


def _one_read(name):
    return [(name, [], 0)]


def test_real_samtools_merges_keep_every_calling_branch(tmp_path):
    swapped = _sha(MODELS / "legacy" / "hia5_nanopore_gt_swapped_legacy.json")
    fixed = _sha(MODELS / "hia5_nanopore.json")

    def called(version, digest=None, pid="fiberhmm-call", aligner=True):
        programs = []
        if aligner:
            programs.append({"PN": "minimap2", "ID": "minimap2", "VN": "2.28",
                             "CL": "minimap2 -a -x map-ont ref.mmi reads.fq"})
        programs.append(_hia5_call(version, "--prob-threshold 248 --primary", pid=pid))
        comments = ([_declaration(pid, apply_sha256=digest, recall_sha256=digest)]
                    if digest else [])
        return _header(programs, comments)

    # Same @PG ID in both inputs: samtools renames one, not the declarations.
    for order in ((swapped, fixed), (fixed, swapped)):
        for aligner in (True, False):
            merged = _merged_bam(tmp_path, f"clash_{aligner}_{order[0][:4]}", [
                (called("3.0.0", order[0], aligner=aligner), _one_read("a")),
                (called("3.0.0", order[1], aligner=aligner), _one_read("b"))])
            payload = report(merged, sidecars=False)
            assert payload["status"] == "rerun-required", payload
            advisory = payload["advisories"][0]
            assert advisory["id"] == "hia5-nanopore-gt-table"
            assert advisory["status"] in ("affected", "possibly_affected")
    # Two fixed inputs merge clean.
    merged = _merged_bam(tmp_path, "both_fixed", [
        (called("3.0.0", fixed), _one_read("a")), (called("3.0.0", fixed), _one_read("b"))])
    assert report(merged, sidecars=False)["status"] == "clean"
    # Distinct IDs, no rename: an old 2.16.7 branch stays in the file.
    merged = _merged_bam(tmp_path, "distinct", [
        (called("2.16.7"), _one_read("a")),
        (called("3.0.0", pid="fiberhmm-call.2"), _one_read("b"))])
    payload = report(merged, sidecars=False)
    assert payload["status"] == "rerun-required" and payload["confirmed"] is True
    merged = _merged_bam(tmp_path, "distinct_bare", [
        (called("2.16.7", aligner=False), _one_read("a")),
        (called("3.0.0", pid="fiberhmm-call.2", aligner=False), _one_read("b"))])
    assert report(merged, sidecars=False)["status"] == "rerun-required"


def test_duplicate_declarations_for_one_renamed_id_are_all_considered():
    """Both inputs declared pg=fiberhmm-call; after the rename either run could
    own either declaration. Both runs hold reads, so the swapped table is in
    the file whichever way they pair."""
    swapped = _sha(MODELS / "legacy" / "hia5_nanopore_gt_swapped_legacy.json")
    fixed = _sha(MODELS / "hia5_nanopore.json")
    a = _hia5_call("3.0.0", "--prob-threshold 248 --primary", pid="fiberhmm-call")
    b = dict(a, ID="fiberhmm-call-6FFFAB4F")
    merge = {"PN": "samtools", "VN": "1.21", "CL": "samtools merge -o m.bam a.bam b.bam"}
    records = [a, dict(b, PP=None), dict(merge, ID="samtools", PP="fiberhmm-call"),
               dict(merge, ID="samtools.1", PP="fiberhmm-call-6FFFAB4F")]
    comments = [_declaration("fiberhmm-call", apply_sha256=swapped, recall_sha256=swapped),
                _declaration("fiberhmm-call", apply_sha256=fixed, recall_sha256=fixed)]
    advisory = _by_id(check_header(_header(records, comments)))["hia5-nanopore-gt-table"]
    assert advisory.status == "affected"
    # Identical declarations from two merged inputs are not collapsed into one.
    same = [_declaration("fiberhmm-call", apply_sha256=fixed, recall_sha256=fixed)] * 2
    assert check_header(_header(records, same)) == []
    # A later run supersedes one of them only: ambiguous which -> possibly.
    recall = _hia5_call("3.0.0", "--prob-threshold 248 --primary", pid="fiberhmm-call.2")
    chain = [a, dict(recall, PP="fiberhmm-call"), dict(b, PP=None)]
    advisory = _by_id(check_header(_header(chain, comments, linked=False))).get(
        "hia5-nanopore-gt-table")
    assert advisory is not None and advisory.status == "possibly_affected"
    assert any("cannot be matched" in line for line in advisory.evidence)

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


def test_report_never_raises_for_malformed_inputs(tmp_path):
    """Every malformed artifact gets an error report instead of an exception."""
    cases = {
        "broken.tsv": b"#metadata:{broken\n",
        "shape.tsv": b"#metadata:[]\n",
        "bad_unicode.tsv": b"\xff\xfe#metadata:{}\n",
        "bad.tsv.gz": b"not gzip",
        "bad.h5": b"not hdf5",
        "bad.bam": b"broken",
        "invalid.qc.json": json.dumps({"samples": [None]}).encode(),
        "assay.qc.json": json.dumps({"assay": "nanopore"}).encode(),
        "list.qc.json": b"[1, 2]",
        "bad_unicode.qc.json": b"\xff{}",
        "bad_sq.sam": b"@SQ\tSN:p\tLN:x\n",
        "missing_id.sam": b"@SQ\tSN:p\tLN:100\n@PG\tPN:fiberhmm-call\tVN:3.0.0\n",
    }
    for name, content in cases.items():
        (tmp_path / name).write_bytes(content)
    consensus = tmp_path / "consensus"
    consensus.mkdir()
    (consensus / "manifest.json").write_text(json.dumps(
        {"cr_mode": "lattice_recaller", "recaller": []}))
    (consensus / "consensus_run.json").write_text(json.dumps({"attempts": [None]}))
    other = tmp_path / "consensus2"
    other.mkdir()
    (other / "manifest.json").write_text(json.dumps({"cr_mode": "lattice_recaller"}))
    (other / "classes.tsv").write_bytes(b"\xff\xfe")
    for path in [*(tmp_path / name for name in cases), consensus, other,
                 tmp_path / "missing.bam"]:
        payload = report(path)
        assert payload["status"] == "error", (path, payload)
        assert payload["error"] and payload["advisories"] == []
        json.dumps(payload)


def test_cli_keeps_going_after_a_malformed_input(tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("#metadata:{broken\n")
    good = tmp_path / "good.bam"
    _write_bam(good, _header([_hia5_call("3.0.0", "--prob-threshold 248 --primary")]))
    result = subprocess.run(
        [sys.executable, "-m", "fiberhmm.cli.check", "--json", str(bad), str(good)],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 2, result.stderr
    assert "Traceback" not in result.stderr
    reports = json.loads(result.stdout)["reports"]
    assert [r["status"] for r in reports] == ["error", "clean"]


# --- malformed or incomplete histories never clear a file ----------------------

def _call(pid, version="3.0.0", pp=None):
    record = _hia5_call(version, "--prob-threshold 248 --primary", pid=pid)
    record["PP"] = pp
    return record


def test_pp_cycles_and_duplicate_ids_never_prove_replacement():
    cycle = _header([_call("old", "2.16.7", pp="new"), _call("new", pp="old")], linked=False)
    advisory = _by_id(check_header(cycle))["hia5-nanopore-gt-table"]
    assert advisory.status == "possibly_affected"
    assert any("forward" in line for line in advisory.evidence)
    # A cycle of fixed calls is still clean.
    fixed_cycle = _header([_call("a", pp="b"), _call("b", pp="a")], linked=False)
    assert check_header(fixed_cycle) == []
    # Two records share an ID; the child's parent is ambiguous.
    duplicate = _header([_call("same", "2.16.7"), _call("same"), _call("new", pp="same")],
                        linked=False)
    advisory = _by_id(check_header(duplicate))["hia5-nanopore-gt-table"]
    assert advisory.status == "possibly_affected"
    assert any("several @PG records carry the ID same" in line for line in advisory.evidence)
    # Unambiguous backward links still supersede.
    linear = _header([_call("old", "2.16.7"), _call("new", pp="old")], linked=False)
    assert check_header(linear) == []


def test_many_ambiguous_declarations_fall_back_to_an_upper_bound():
    """Too many pairings to enumerate: every live run is checked with every
    declaration it could carry, and the file is at least possibly affected."""
    swapped = _sha(MODELS / "legacy" / "hia5_nanopore_gt_swapped_legacy.json")
    fixed = _sha(MODELS / "hia5_nanopore.json")
    ids = ["fiberhmm-call"] + [f"fiberhmm-call-{i:08X}" for i in range(1, 6)]
    records = [_call(pid) for pid in ids] + [_call("fresh", pp=ids[0])]
    for comments in (
            [_declaration("fiberhmm-call", apply_sha256=swapped, recall_sha256=swapped)]
            + [_declaration("fiberhmm-call", apply_sha256=fixed, recall_sha256=fixed,
                            model=f"good{i}") for i in range(5)],
            [_declaration("fiberhmm-call", apply_sha256=fixed, recall_sha256=fixed)] * 1100
            + [_declaration("fiberhmm-call", apply_sha256=swapped, recall_sha256=swapped)]):
        header = _header(records, comments + [
            _declaration("fresh", apply_sha256=fixed, recall_sha256=fixed)], linked=False)
        advisory = _by_id(check_header(header))["hia5-nanopore-gt-table"]
        assert advisory.status == "possibly_affected"
        assert any("too many pairings" in line for line in advisory.evidence)
    # All declarations fixed: the upper bound is clean too.
    clean = [_declaration("fiberhmm-call", apply_sha256=fixed, recall_sha256=fixed,
                          model=f"good{i}") for i in range(6)]
    assert check_header(_header(records, clean, linked=False)) == []


def test_samtools_cat_hides_the_other_inputs_history(tmp_path):
    """samtools cat keeps the first input's header only: never clean unless a
    later full call re-called every read."""
    swapped = _sha(MODELS / "legacy" / "hia5_nanopore_gt_swapped_legacy.json")
    fixed = _sha(MODELS / "hia5_nanopore.json")

    def called(name, digest):
        path = tmp_path / f"{name}.bam"
        header = _header([_call("fiberhmm-call")], [
            _declaration("fiberhmm-call", apply_sha256=digest, recall_sha256=digest)])
        _write_bam(path, header, [(name, [], 0)])
        return str(path)
    good, bad = called("good", fixed), called("bad", swapped)
    joined = tmp_path / "good_bad.bam"
    pysam.cat("-o", str(joined), good, bad)
    payload = report(joined, sidecars=False)
    assert payload["status"] == "rerun-required" and payload["confirmed"] is False
    advisory = payload["advisories"][0]
    assert advisory["status"] == "possibly_affected"
    assert "kept one header" in advisory["evidence"][0]
    reverse = tmp_path / "bad_good.bam"
    pysam.cat("-o", str(reverse), bad, good)
    assert report(reverse, sidecars=False)["confirmed"] is True
    # A full call after the cat re-calls every read.
    with pysam.AlignmentFile(str(joined)) as handle:
        header = handle.header.to_dict()
    recalled = _header(header["PG"] + [_call("fiberhmm-call.2", pp=header["PG"][-1]["ID"])],
                       header["CO"] + [_declaration("fiberhmm-call.2", apply_sha256=fixed,
                                                    recall_sha256=fixed)], linked=False)
    assert check_header(recalled) == []
    # A cat of one file, and fiberhmm-call joining its own region files, hide nothing.
    for cl in ("samtools cat -o out.bam one.bam",
               "samtools cat -h /d/.fiberhmm_call_tmp_x1/region_000000.bam "
               "-b /d/.fiberhmm_call_tmp_x1/bam_list.txt -o /d/out.bam",
               "samtools cat -h /d/.out.bam.fiberhmm-work/region_000000.bam "
               "-b /d/.out.bam.fiberhmm-work/bam_list.txt -o /d/out.bam"):
        records = [_call("fiberhmm-call"), {"PN": "samtools", "ID": "samtools", "VN": "1.21",
                                            "CL": cl, "PP": "fiberhmm-call"}]
        header = _header(records, [_declaration("fiberhmm-call", apply_sha256=fixed,
                                                recall_sha256=fixed)], linked=False)
        assert check_header(header) == [], cl
    gather = [_call("fiberhmm-call"), {"PN": "GatherBamFiles", "ID": "GatherBamFiles",
                                       "CL": "picard GatherBamFiles I=a.bam I=b.bam O=c.bam",
                                       "PP": "fiberhmm-call"}]
    header = _header(gather, [_declaration("fiberhmm-call", apply_sha256=fixed,
                                           recall_sha256=fixed)], linked=False)
    assert _by_id(check_header(header))["hia5-nanopore-gt-table"].status == "possibly_affected"


_SAMTOOLS = shutil.which("samtools")


@pytest.mark.skipif(_SAMTOOLS is None, reason="samtools not installed")
@pytest.mark.parametrize("spelling", [
    ["-b", "{list}"], ["-b{list}"], ["-fb{list}"], ["-q", "-b{list}"], ["-f", "-b", "{list}"],
    ["{good}", "{bad}"], ["-f", "--", "{good}", "{bad}"],
])
def test_real_samtools_cat_spellings_are_all_concatenation(tmp_path, spelling):
    """Every spelling samtools cat accepts for several inputs (attached -bFILE,
    clustered flags, file lists, positional inputs) hides the later inputs'
    headers: the file is never clean."""
    swapped = _sha(MODELS / "legacy" / "hia5_nanopore_gt_swapped_legacy.json")
    fixed = _sha(MODELS / "hia5_nanopore.json")
    paths = {}
    for name, digest in (("good", fixed), ("bad", swapped)):
        paths[name] = str(tmp_path / f"{name}.bam")
        header = _header([_call("fiberhmm-call")], [
            _declaration("fiberhmm-call", apply_sha256=digest, recall_sha256=digest)])
        _write_bam(paths[name], header, [(name, [], 0)])
    paths["list"] = str(tmp_path / "inputs.list")
    Path(paths["list"]).write_text(f"{paths['good']}\n{paths['bad']}\n")
    out = tmp_path / "joined.bam"
    args = [a.format(**paths) for a in spelling]
    subprocess.run([_SAMTOOLS, "cat", "-o", str(out), *args], check=True,
                   capture_output=True)
    with pysam.AlignmentFile(str(out)) as handle:
        assert sorted(r.query_name for r in handle) == ["bad", "good"]
    payload = report(out, sidecars=False)
    assert payload["status"] == "rerun-required", (args, payload)
    assert "kept one header" in payload["advisories"][0]["evidence"][0]


@pytest.mark.parametrize("arguments, joins", [
    ("-b x.list -o o.bam", True),
    ("-bx.list -o o.bam", True),
    ("-o o.bam -fqbx.list", True),
    ("--output-fmt=BAM -o o.bam a.bam b.bam", True),
    ("--output-fmt BAM -o o.bam a.bam", False),
    ("--verb 3 -o o.bam a.bam", False),
    ("--no-PG -o o.bam -- -o.bam", False),
    ("-o o.bam a.bam - ", True),
    ("-hheader.sam -oo.bam a.bam", False),
    ("-h header.sam -o o.bam a.bam", False),
    ("-o o.bam a.bam", False),
    ("--bogus -o o.bam a.bam", True),          # unparsable: never cleared
    ("-Z -o o.bam a.bam", True),
])
def test_samtools_cat_command_lines_are_parsed_like_getopt(arguments, joins):
    record = {"PN": "samtools", "ID": "samtools", "CL": "samtools cat " + arguments}
    assert adv._drops_input_headers(record) is joins


def test_fiberhmm_region_concatenation_is_not_a_join_even_with_spaces():
    for directory in ("/data/run 1/.out.bam.fiberhmm-work",
                      "/Users/x/FiberHMM v1.0/.fiberhmm_call_tmp_ab12"):
        cl = (f"samtools cat -h {directory}/region_000000.bam "
              f"-b {directory}/bam_list.txt -o /Users/x/FiberHMM v1.0/out.bam")
        assert adv._drops_input_headers({"PN": "samtools", "CL": cl}) is False, cl
    # Same shape but files outside a FiberHMM work directory: a user's join.
    cl = "samtools cat -h /d/region_000000.bam -b /d/bam_list.txt -o /d/out.bam"
    assert adv._drops_input_headers({"PN": "samtools", "CL": cl}) is True
