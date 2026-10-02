"""Published outputs honour the user's umask (audit M12).

``tempfile.mkstemp`` creates 0600 files and a rename keeps that mode, so
results published that way could not be opened by other members of a lab
sharing a results directory. Every writer below must give a file the
permissions ``open()`` would (0666 reduced by the umask).
"""
import os
import stat
import tempfile

import pytest

from fiberhmm.io.output_files import mkstemp_shared


@pytest.fixture
def umask_022():
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_mkstemp_shared_applies_the_umask_like_open(tmp_path):
    previous = os.umask(0o027)
    try:
        descriptor, path = mkstemp_shared(prefix=".x.", suffix=".tmp", dir=tmp_path)
        os.close(descriptor)
        reference = tmp_path / "opened"
        reference.write_text("")
        raw_descriptor, raw_path = tempfile.mkstemp(dir=tmp_path)
        os.close(raw_descriptor)
    finally:
        os.umask(previous)
    assert os.path.basename(path).startswith(".x.") and path.endswith(".tmp")
    assert _mode(path) == _mode(reference) == 0o640
    assert _mode(raw_path) == 0o600  # what the writers used to publish


def test_mkstemp_shared_never_reuses_an_existing_name(tmp_path, monkeypatch):
    import fiberhmm.io.output_files as output_files

    names = iter(["aaaaaaaaaaaa", "aaaaaaaaaaaa", "bbbbbbbbbbbb"])
    monkeypatch.setattr(output_files.secrets, "token_hex", lambda _n: next(names))
    first, first_path = mkstemp_shared(prefix="p", dir=tmp_path)
    second, second_path = mkstemp_shared(prefix="p", dir=tmp_path)
    os.close(first)
    os.close(second)
    assert first_path != second_path
    assert second_path.endswith("pbbbbbbbbbbbb")


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
@pytest.mark.parametrize("name", ["manifest.json", "result.json.gz"])
def test_consensus_json_artifacts_are_group_readable(tmp_path, umask_022, name):
    from fiberhmm.inference.consensus.artifacts import read_json, write_json

    path = tmp_path / name
    write_json(path, {"a": 1})
    assert read_json(path) == {"a": 1}
    assert _mode(path) == 0o644


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_native_fit_cache_entries_are_group_readable(tmp_path, umask_022):
    from fiberhmm.inference.consensus.fit_cache import NativeFitCache

    cache = NativeFitCache(tmp_path / "cache")
    cache.put("k" * 64, [0.5], 1.0, True, 3, "ok", 2)
    (entry,) = [p for p in (tmp_path / "cache").rglob("*") if p.is_file()]
    assert _mode(entry) == 0o644


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_strand_rescue_reports_are_group_readable(tmp_path, umask_022):
    from fiberhmm.cli.strand_rescue import _atomic_write
    from fiberhmm.cli.strand_rescue_audit import _atomic_json

    report = tmp_path / "sr.json"
    _atomic_write(report, "{}\n")
    audit = tmp_path / "audit.json"
    _atomic_json(audit, {"ok": True})
    assert _mode(report) == _mode(audit) == 0o644


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_tag_consensus_bam_and_index_are_group_readable(tmp_path, umask_022):
    from fiberhmm.cli.tag_families import tag_tf_families
    from test_tf_family_tags import _make_input, _write_assignments

    source = tmp_path / "source.bam"
    output = tmp_path / "family.bam"
    assignments = tmp_path / "assignments.tsv"
    _make_input(source)
    _write_assignments(assignments)
    tag_tf_families(source, output, assignments)
    assert _mode(output) == 0o644
    assert _mode(str(output) + ".bai") == 0o644
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_consensus_family_bam_export_is_group_readable(tmp_path, umask_022):
    from fiberhmm.inference.consensus.bam_export import export_bams
    from test_consensus_bam_export import fixture

    _source, payload, result, _original = fixture(tmp_path)
    rows = export_bams([(result, payload)], tmp_path / "output", scope="full")
    assert _mode(rows[0]["bam"]) == 0o644
    assert _mode(rows[0]["index"]) == 0o644
