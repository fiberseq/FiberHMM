"""Run identity in FIBERHMM-CHEMISTRY declarations (tables, version, commit, @PG link)."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pysam
import pytest

import fiberhmm
import fiberhmm.cli.recall_tfs as recall_tfs
from fiberhmm import identity
from fiberhmm.cli.provenance import (
    chemistry_declaration,
    output_header_with_provenance,
)
from fiberhmm.io.bam_header import declared_chemistries
from fiberhmm.models import _bundled_model_path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_declaration_records_tables_version_and_commit(monkeypatch):
    monkeypatch.setattr(identity, "fiberhmm_commit", lambda: "a" * 40)
    apply_path = _bundled_model_path("ddda_nuc.json")
    recall_path = _bundled_model_path("ddda_TF.json")
    declaration = chemistry_declaration(
        SimpleNamespace(enzyme="ddda", seq=None), "daf", apply_path, recall_path,
        nuc_model_path=_bundled_model_path("ddda_nuc_refine.json"),
    )
    assert declaration["apply_sha256"] == _sha(apply_path)
    assert declaration["recall_sha256"] == _sha(recall_path)
    assert declaration["nuc_model_sha256"] == _sha(
        _bundled_model_path("ddda_nuc_refine.json"))
    assert declaration["fiberhmm_version"] == fiberhmm.__version__
    assert declaration["fiberhmm_commit"] == "a" * 40


def test_declaration_omits_unknown_commit_and_missing_pass(monkeypatch):
    monkeypatch.setattr(identity, "fiberhmm_commit", lambda: None)
    path = _bundled_model_path("hia5_pacbio.json")
    declaration = chemistry_declaration(
        SimpleNamespace(enzyme="hia5", seq="pacbio"), "pacbio-fiber", path, None)
    assert "fiberhmm_commit" not in declaration
    assert "recall_sha256" not in declaration
    assert declaration["apply_sha256"] == _sha(path)


def test_declaration_is_backward_compatible_for_header_readers(monkeypatch):
    """Readers (FiberBrowser uses declared_chemistries) keep the core contract."""
    monkeypatch.setattr(identity, "fiberhmm_commit", lambda: "b" * 40 + "+dirty")
    path = _bundled_model_path("hia5_nanopore.json")
    record = {
        "PN": "fiberhmm-call", "VN": fiberhmm.__version__, "CL": "fiberhmm-call",
        "chemistry": chemistry_declaration(
            SimpleNamespace(enzyme="hia5", seq="nanopore"), "nanopore-fiber",
            path, path),
    }
    header = output_header_with_provenance({"HD": {"VN": "1.6"}}, record)
    (comment,) = [c for c in header.to_dict()["CO"]]
    assert comment.startswith(
        "FIBERHMM-CHEMISTRY:v1:assay=fiber-seq;enzyme=hia5;platform=nanopore;"
        "mode=nanopore-fiber;")
    (declared,) = declared_chemistries(header)
    assert declared["model"] == "hia5_nanopore"
    assert declared["pg"] == "fiberhmm-call"
    assert declared["fiberhmm_commit"] == "b" * 40 + "+dirty"
    assert declared["apply_sha256"] == declared["recall_sha256"] == _sha(path)

    # A second run links its own @PG ID; both declarations stay readable.
    again = output_header_with_provenance(header, record)
    declarations = declared_chemistries(again)
    assert [d["pg"] for d in declarations] == ["fiberhmm-call", "fiberhmm-call.2"]
    assert {d["enzyme"] for d in declarations} == {"hia5"}


def test_recall_tfs_declares_its_table_as_the_recall_table():
    model_path = _bundled_model_path("ddda_TF.json")
    nuc_path = _bundled_model_path("ddda_nuc_refine.json")
    record = recall_tfs._build_recall_pg_record(
        SimpleNamespace(enzyme="ddda", seq=None), "daf", model_path, None,
        nuc_model_path=nuc_path,
    )
    chemistry = record["chemistry"]
    assert "apply_sha256" not in chemistry
    assert chemistry["recall_sha256"] == _sha(model_path)
    assert chemistry["nuc_model_sha256"] == _sha(nuc_path)


def test_git_commit_of_checkout(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    assert identity.git_commit_of_checkout(tmp_path) is None  # no .git: omitted

    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True,
                       capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.org")
    git("config", "user.name", "t")
    (tmp_path / "fiberhmm").mkdir()
    (tmp_path / "fiberhmm" / "x.py").write_text("x = 1\n")
    git("add", ".")
    git("commit", "-qm", "init")
    head = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert identity.git_commit_of_checkout(tmp_path) == head
    (tmp_path / "fiberhmm" / "x.py").write_text("x = 2\n")
    assert identity.git_commit_of_checkout(tmp_path) == head + "+dirty"
    (tmp_path / "fiberhmm" / "x.py").write_text("x = 1\n")
    assert identity.git_commit_of_checkout(tmp_path) == head
    # A new module under fiberhmm/ can change results too.
    (tmp_path / "fiberhmm" / "new.py").write_text("")
    assert identity.git_commit_of_checkout(tmp_path) == head + "+dirty"
    (tmp_path / ".gitignore").write_text("fiberhmm/new.py\n")
    assert identity.git_commit_of_checkout(tmp_path) == head  # ignored files do not count


def test_commit_falls_back_to_build_info(monkeypatch):
    import sys
    import types

    monkeypatch.setattr(identity, "git_commit_of_checkout", lambda root: None)
    module = types.ModuleType("fiberhmm._build_info")
    module.COMMIT = "C" * 40
    monkeypatch.setitem(sys.modules, "fiberhmm._build_info", module)
    identity.fiberhmm_commit.cache_clear()
    try:
        assert identity.fiberhmm_commit() == "c" * 40
        monkeypatch.delitem(sys.modules, "fiberhmm._build_info")
        monkeypatch.setattr(identity, "_PACKAGE_DIR", Path("/nonexistent/fiberhmm"))
        identity.fiberhmm_commit.cache_clear()
        # No checkout and no build info: omitted, never guessed.
        import builtins
        real_import = builtins.__import__

        def no_build_info(name, *args, **kwargs):
            if name == "fiberhmm._build_info":
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_build_info)
        assert identity.fiberhmm_commit() is None
    finally:
        identity.fiberhmm_commit.cache_clear()


def test_file_sha256_tracks_content_changes(tmp_path):
    path = tmp_path / "t.json"
    path.write_text("{}")
    first = identity.file_sha256(path)
    assert first == hashlib.sha256(b"{}").hexdigest()
    path.write_text('{"a": 1}')
    assert identity.file_sha256(path) == hashlib.sha256(b'{"a": 1}').hexdigest()
    assert identity.file_sha256(tmp_path / "missing.json") is None
    assert identity.file_sha256(None) is None


def test_region_worker_header_carries_identity(tmp_path, monkeypatch):
    """The header written by the producers' shared path is a valid BAM header."""
    monkeypatch.setattr(identity, "fiberhmm_commit", lambda: None)
    path = _bundled_model_path("hia5_pacbio.json")
    record = {
        "PN": "fiberhmm-apply", "VN": fiberhmm.__version__,
        "chemistry": chemistry_declaration(
            SimpleNamespace(enzyme="hia5", seq="pacbio"), "pacbio-fiber", path, None),
    }
    header = output_header_with_provenance(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 100}]}, record)
    out = tmp_path / "x.bam"
    with pysam.AlignmentFile(str(out), "wb", header=header):
        pass
    with pysam.AlignmentFile(str(out), "rb", check_sq=False) as bam:
        (declared,) = declared_chemistries(bam.header)
    assert declared["pg"] == "fiberhmm-apply"
    assert declared["apply_sha256"] == _sha(path)
