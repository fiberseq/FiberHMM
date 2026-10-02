"""Input/output path aliasing: a mistyped path never destroys data.

Regression tests for the 3.0 CLI audit, package P3: ``fiberhmm-dedup -i X -o X``
emptied the input (H3); ``fiberhmm-call --work-dir W -o W/calls.bam`` deleted
its own output with rc 0 (H4); ``fiberhmm-strand-rescue-audit -i X -o X``
replaced the BAM with JSON (H5); ``fiberhmm-pair --pairs-tsv/--receipt-json``
overwrote the input/output BAM (H6); ``fiberhmm-pair`` staged through a fixed
``<output>.paired.tmp.bam`` (M6); call/recall/daf-encode/merge silently
replaced the input on ``-i X -o X`` (M7). All go through one guard,
:func:`fiberhmm.cli.common.find_path_aliases`.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pysam
import pytest

from fiberhmm.cli.common import PathAliasError, check_path_aliases, find_path_aliases

sys.path.insert(0, str(Path(__file__).parent))


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _case_insensitive(tmp_path):
    probe = tmp_path / "CaseProbe"
    probe.write_text("x")
    try:
        return (tmp_path / "caseprobe").exists()
    finally:
        probe.unlink()


def _run_main(monkeypatch, module_main, argv):
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as caught:
        module_main()
    return caught.value.code


# ---------------------------------------------------------------------------
# The guard itself
# ---------------------------------------------------------------------------

def test_distinct_paths_and_streams_are_accepted(tmp_path):
    (tmp_path / "in.bam").write_text("x")
    assert find_path_aliases(
        inputs={"--input": str(tmp_path / "in.bam"), "--model": None},
        outputs={"--output": str(tmp_path / "out.bam"), "--tsv": "-",
                 "--json": None},
        deleted_dirs={"--work-dir": str(tmp_path / "work")},
    ) == []
    # stdin/stdout on both sides are streams, not files.
    assert find_path_aliases(inputs={"-i": "-"}, outputs={"-o": "-"}) == []


def test_same_path_is_refused(tmp_path):
    bam = tmp_path / "in.bam"
    bam.write_text("x")
    with pytest.raises(PathAliasError, match=r"--output .* same file as --input"):
        check_path_aliases(inputs={"--input": str(bam)},
                           outputs={"--output": str(tmp_path / "." / "in.bam")})


def test_symlink_and_hard_link_aliases_are_refused(tmp_path):
    bam = tmp_path / "in.bam"
    bam.write_text("x")
    link = tmp_path / "link.bam"
    link.symlink_to(bam)
    hard = tmp_path / "hard.bam"
    os.link(bam, hard)
    for alias in (link, hard):
        problems = find_path_aliases(inputs={"--input": str(bam)},
                                     outputs={"--output": str(alias)})
        assert len(problems) == 1 and "same file as --input" in problems[0]
    # a symlinked directory reaching the input
    linked_dir = tmp_path / "linked"
    linked_dir.symlink_to(tmp_path)
    assert find_path_aliases(inputs={"--input": str(bam)},
                             outputs={"--output": str(linked_dir / "in.bam")})


def test_case_variants_are_refused_on_case_insensitive_filesystems(tmp_path):
    if not _case_insensitive(tmp_path):
        pytest.skip("filesystem is case-sensitive")
    bam = tmp_path / "Sample.bam"
    bam.write_text("x")
    assert find_path_aliases(inputs={"--input": str(bam)},
                             outputs={"--output": str(tmp_path / "sample.BAM")})
    # two outputs that do not exist yet
    problems = find_path_aliases(outputs={"--output": str(tmp_path / "New" / "a.tsv"),
                                          "--pairs-tsv": str(tmp_path / "new" / "A.TSV")})
    assert problems and "name the same file" in problems[0]


def test_input_index_is_protected(tmp_path):
    bam = tmp_path / "in.bam"
    bam.write_text("x")
    (tmp_path / "in.bam.bai").write_text("i")
    problems = find_path_aliases(inputs={"--input": str(bam)},
                                 outputs={"--receipt-json": str(tmp_path / "in.bam.bai")})
    assert len(problems) == 1 and "is the index" in problems[0]


def test_two_outputs_naming_one_file_are_refused(tmp_path):
    problems = find_path_aliases(outputs={"--output": str(tmp_path / "out.bam"),
                                          "--receipt-json": str(tmp_path / "out.bam")})
    assert problems == [
        f"--output and --receipt-json name the same file ({tmp_path / 'out.bam'}, "
        f"{tmp_path / 'out.bam'}); each output needs its own path"]


@pytest.mark.parametrize("exists", [False, True])
def test_paths_inside_a_deleted_directory_are_refused(tmp_path, exists):
    work = tmp_path / "work"
    if exists:
        work.mkdir()
    problems = find_path_aliases(
        inputs={"--input": str(tmp_path / "in.bam")},
        outputs={"--output": str(work / "sub" / "calls.bam")},
        deleted_dirs={"--work-dir": str(work)})
    assert len(problems) == 1
    assert "--output" in problems[0] and "inside --work-dir" in problems[0]
    # the directory itself, and an input inside it
    assert find_path_aliases(outputs={"--output": str(work)},
                             deleted_dirs={"--work-dir": str(work)})
    assert find_path_aliases(inputs={"--input": str(work / "in.bam")},
                             deleted_dirs={"--work-dir": str(work)})
    # a sibling with a common name prefix is outside
    assert find_path_aliases(outputs={"--output": str(tmp_path / "work2" / "x.bam")},
                             deleted_dirs={"--work-dir": str(work)}) == []


def test_symlink_entry_inside_a_deleted_directory_is_refused(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    target = tmp_path / "kept.bam"
    target.write_text("x")
    (work / "out.bam").symlink_to(target)
    assert find_path_aliases(outputs={"--output": str(work / "out.bam")},
                             deleted_dirs={"--work-dir": str(work)})


# ---------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------

def _daf_bam(tmp_path):
    from test_dedup import A_SITES, _make_bam

    bam = tmp_path / "d.bam"
    _make_bam(bam, [(f"r{i}", A_SITES, False, 60) for i in range(4)])
    pysam.index(str(bam))
    return bam


def test_dedup_refuses_output_equal_to_input(tmp_path, monkeypatch, capsys):  # H3
    from fiberhmm.cli import dedup

    bam = _daf_bam(tmp_path)
    before = _digest(bam)
    code = _run_main(monkeypatch, dedup.main,
                     ["fiberhmm-dedup", "-i", str(bam), "-o", str(bam)])
    assert code == 2
    assert "--output" in capsys.readouterr().err
    assert _digest(bam) == before


def test_dedup_refuses_stats_tsv_equal_to_input(tmp_path, monkeypatch, capsys):
    from fiberhmm.cli import dedup

    bam = _daf_bam(tmp_path)
    before = _digest(bam)
    code = _run_main(monkeypatch, dedup.main,
                     ["fiberhmm-dedup", "-i", str(bam), "-o", str(tmp_path / "o.bam"),
                      "--stats-tsv", str(bam)])
    assert code == 2 and "--stats-tsv" in capsys.readouterr().err
    assert _digest(bam) == before
    assert not (tmp_path / "o.bam").exists()


def test_dedup_failure_leaves_no_partial_output(tmp_path, monkeypatch):
    from fiberhmm.cli import dedup

    bam = _daf_bam(tmp_path)
    out = tmp_path / "out.bam"
    out.write_bytes(b"previous")

    real_open = pysam.AlignmentFile

    class FailingWriter:
        """The real output file, whose first record write fails."""

        def __init__(self, handle):
            self._handle = handle

        def write(self, read):
            raise RuntimeError("write failure")

        def close(self):
            self._handle.close()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()

    def open_alignment_file(path, mode="r", *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        return FailingWriter(handle) if "w" in mode else handle

    monkeypatch.setattr(dedup.pysam, "AlignmentFile", open_alignment_file)
    with pytest.raises(RuntimeError, match="write failure"):
        dedup.run_dedup(str(bam), str(out), min_jaccard=0.95, min_deam=1,
                        ignore_strand=False, k=32, bands=8, seed=7,
                        collapse=True, io_threads=1)
    assert out.read_bytes() == b"previous"
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".")) == []


@pytest.fixture
def region_bam(tmp_path):
    from test_call_entrypoint_regressions import make_region_test_bam

    return make_region_test_bam(tmp_path / "in.bam")


def _call(monkeypatch, *argv):
    from fiberhmm.cli import call as cli

    monkeypatch.setattr(sys, "argv", ["fiberhmm-call", *map(str, argv)])
    try:
        cli.main()
    except SystemExit as exit:
        return exit.code or 0
    return 0


def test_call_refuses_output_inside_work_dir(tmp_path, monkeypatch, capsys,
                                             region_bam, benchmark_model_path):  # H4
    work = tmp_path / "work"
    code = _call(monkeypatch, "-i", region_bam, "-m", benchmark_model_path,
                 "--region-parallel", "--work-dir", work, "-o", work / "calls.bam",
                 "--no-qc", "-c", "1", "--io-threads", "1", "--min-read-length", "0")
    assert code == 2
    assert "inside --work-dir" in capsys.readouterr().err
    assert not work.exists()


def test_call_refuses_output_inside_default_work_dir_of_itself(
        tmp_path, monkeypatch, capsys, region_bam, benchmark_model_path):
    from fiberhmm.inference.region_resume import default_work_dir

    output = tmp_path / "calls.bam"
    progress = default_work_dir(output) / "progress.jsonl"
    code = _call(monkeypatch, "-i", region_bam, "-m", benchmark_model_path,
                 "--region-parallel", "-o", output, "--progress-json", progress,
                 "--no-qc", "-c", "1")
    assert code == 2 and "--progress-json" in capsys.readouterr().err


def test_call_refuses_in_place(tmp_path, monkeypatch, capsys, region_bam,
                               benchmark_model_path):  # M7
    before = _digest(region_bam)
    code = _call(monkeypatch, "-i", region_bam, "-m", benchmark_model_path,
                 "-o", region_bam, "--no-qc", "-c", "1")
    assert code == 2 and "same file as --input" in capsys.readouterr().err
    assert _digest(region_bam) == before


def test_strand_rescue_audit_refuses_output_equal_to_input(tmp_path, capsys):  # H5
    from fiberhmm.cli.strand_rescue_audit import main

    bam = _daf_bam(tmp_path)
    before = _digest(bam)
    with pytest.raises(SystemExit) as caught:
        main(["-i", str(bam), "-o", str(bam)])
    assert caught.value.code == 2
    assert "same file as --bam" in capsys.readouterr().err
    assert _digest(bam) == before


def _pair_fixture(tmp_path):
    from test_pair_unified import _write_sequence_resolved_fixture

    source, reference = _write_sequence_resolved_fixture(tmp_path)
    common = ["-r", str(reference), "--sequence-only", "--min-overlap", "100",
              "--min-nucs", "1", "--io-threads", "1"]
    return source, common


@pytest.mark.parametrize("flag", ["--pairs-tsv", "--receipt-json"])
def test_pair_sidecar_cannot_overwrite_input(tmp_path, monkeypatch, capsys, flag):  # H6
    from fiberhmm.cli import pair

    source, common = _pair_fixture(tmp_path)
    before = _digest(source)
    code = _run_main(monkeypatch, pair.main,
                     ["fiberhmm-pair", "-i", str(source), "-o", str(tmp_path / "p.bam"),
                      *common, "--stop-after", "pair", flag, str(source)])
    assert code == 2
    assert f"{flag} {source} is the same file as --input" in capsys.readouterr().err
    assert _digest(source) == before


@pytest.mark.parametrize("flag", ["--pairs-tsv", "--receipt-json"])
def test_pair_sidecar_cannot_overwrite_output(tmp_path, monkeypatch, capsys, flag):  # H6
    from fiberhmm.cli import pair

    source, common = _pair_fixture(tmp_path)
    output = tmp_path / "p.bam"
    code = _run_main(monkeypatch, pair.main,
                     ["fiberhmm-pair", "-i", str(source), "-o", str(output),
                      *common, "--stop-after", "pair", flag, str(output)])
    assert code == 2 and "name the same file" in capsys.readouterr().err
    assert not output.exists()


def test_pair_staging_never_touches_an_existing_file(tmp_path, monkeypatch, capsys):  # M6
    from fiberhmm.cli import pair

    source, common = _pair_fixture(tmp_path)
    output = tmp_path / "merged.bam"
    bystander = tmp_path / "merged.bam.paired.tmp.bam"
    bystander.write_bytes(b"someone else's file")
    monkeypatch.setattr(sys, "argv", ["fiberhmm-pair", "-i", str(source), "-o", str(output),
                                      *common, "--stop-after", "merge", "--pairs-only"])
    pair.main()
    assert bystander.read_bytes() == b"someone else's file"
    assert output.exists()
    leftovers = [p.name for p in tmp_path.iterdir()
                 if p.name.startswith(".") or "paired" in p.name]
    assert leftovers == [bystander.name]


def test_duplex_sidecar_cannot_overwrite_input(tmp_path, monkeypatch, capsys):  # H6
    from fiberhmm.cli import duplex

    source, _common = _pair_fixture(tmp_path)
    reference = tmp_path / "reference.fa"
    before = _digest(source)
    code = _run_main(monkeypatch, duplex.main,
                     ["duplex", "-i", str(source), "-o", str(tmp_path / "d.bam"),
                      "-r", str(reference), "--pairs-tsv", str(source)])
    assert code == 2 and "--pairs-tsv" in capsys.readouterr().err
    assert _digest(source) == before


def test_merge_refuses_in_place(tmp_path, monkeypatch, capsys):  # M7
    from fiberhmm.cli import merge

    bam = _daf_bam(tmp_path)
    before = _digest(bam)
    code = _run_main(monkeypatch, merge.main,
                     ["fiberhmm-merge", "-i", str(bam), "-o", str(bam)])
    assert code == 2 and "same file as --input" in capsys.readouterr().err
    assert _digest(bam) == before


def test_daf_encode_refuses_in_place(tmp_path, capsys):  # M7
    from fiberhmm.cli import daf_encode

    bam = _daf_bam(tmp_path)
    before = _digest(bam)
    with pytest.raises(SystemExit) as caught:
        daf_encode.main(["-i", str(bam), "-o", str(bam)])
    assert caught.value.code == 2
    assert "same file as --input" in capsys.readouterr().err
    assert _digest(bam) == before


@pytest.mark.parametrize("entry", ["main", "main_recall_nucs"])
def test_recall_refuses_in_place(tmp_path, monkeypatch, capsys, entry):  # M7
    from fiberhmm.cli import recall_tfs

    bam = _daf_bam(tmp_path)
    before = _digest(bam)
    code = _run_main(monkeypatch, getattr(recall_tfs, entry),
                     ["fiberhmm-recall-tfs", "-i", str(bam), "-o", str(bam),
                      "--enzyme", "ddda"])
    assert code == 2 and "same file as --in-bam" in capsys.readouterr().err
    assert _digest(bam) == before


# ---------------------------------------------------------------------------
# Review follow-ups: implied index files, generated files, kept work dirs
# ---------------------------------------------------------------------------

def test_sidecar_cannot_be_the_index_a_bam_output_publishes(tmp_path):
    out = tmp_path / "out.bam"
    for index in ("out.bam.bai", "out.bai", "out.bam.csi"):
        problems = find_path_aliases(outputs={"--output": str(out),
                                              "--stats-tsv": str(tmp_path / index)})
        assert problems and "replaces or deletes" in problems[0]
    # a TSV output has no index
    assert find_path_aliases(outputs={"--output": str(tmp_path / "p.tsv.gz"),
                                      "--stats": str(tmp_path / "p.tsv.gz.bai")}) == []


def test_dedup_refuses_stats_tsv_at_the_output_index(tmp_path, monkeypatch, capsys):
    from fiberhmm.cli import dedup

    bam = _daf_bam(tmp_path)
    code = _run_main(monkeypatch, dedup.main,
                     ["fiberhmm-dedup", "-i", str(bam), "-o", str(tmp_path / "o.bam"),
                      "--stats-tsv", str(tmp_path / "o.bam.bai")])
    assert code == 2 and "--stats-tsv" in capsys.readouterr().err


def test_input_index_inside_a_deleted_directory_is_refused(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    bam = tmp_path / "in.bam"
    bam.write_text("x")
    (work / "in.bam.bai").write_text("i")
    (tmp_path / "in.bam.bai").symlink_to(work / "in.bam.bai")
    problems = find_path_aliases(inputs={"--input": str(bam)},
                                 deleted_dirs={"--work-dir": str(work)})
    assert problems and "inside --work-dir" in problems[0]


def test_path_through_a_symlink_inside_a_deleted_directory_is_refused(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    (data / "in.bam").write_text("x")
    (work / "link").symlink_to(data)
    assert find_path_aliases(inputs={"--input": str(work / "link" / "in.bam")},
                             deleted_dirs={"--work-dir": str(work)})
    assert find_path_aliases(inputs={"--input": str(data / "in.bam")},
                             deleted_dirs={"--work-dir": str(work)}) == []


def _call_args(monkeypatch, *argv):
    from fiberhmm.cli import call as cli

    monkeypatch.setattr(sys, "argv", ["fiberhmm-call", *map(str, argv)])
    return cli, cli.parse_args()


def test_call_keep_work_dir_allows_output_inside_it(tmp_path, monkeypatch):
    work = tmp_path / "work"
    cli, args = _call_args(monkeypatch, "-i", tmp_path / "in.bam", "--enzyme", "hia5",
                           "--region-parallel", "--work-dir", work,
                           "--keep-work-dir", "-o", work / "calls.bam")
    cli._refuse_call_path_aliases(args, "model.json", "model.json")  # no exit


def test_call_generated_snp_and_qc_files_cannot_overwrite_inputs(
        tmp_path, monkeypatch, capsys):
    model = tmp_path / "run.json"
    model.write_text("{}")
    cli, args = _call_args(monkeypatch, "-i", tmp_path / "in.bam", "-m", model,
                           "-o", tmp_path / "calls.bam",
                           "--daf-snp-output-prefix", tmp_path / "run")
    with pytest.raises(SystemExit) as caught:
        cli._refuse_call_path_aliases(args, str(model), str(model))
    assert caught.value.code == 2
    assert "--daf-snp-output-prefix" in capsys.readouterr().err

    cli, args = _call_args(monkeypatch, "-i", tmp_path / "in.bam", "-m", model,
                           "-o", tmp_path / "calls.bam",
                           "--qc-output-prefix", tmp_path / "run.qc")
    model2 = tmp_path / "run.qc.qc.json"
    model2.write_text("{}")
    with pytest.raises(SystemExit):
        cli._refuse_call_path_aliases(args, str(model2), str(model2))


def test_call_bundled_model_is_protected(tmp_path, monkeypatch, capsys):
    cli, args = _call_args(monkeypatch, "-i", tmp_path / "in.bam", "--enzyme", "hia5",
                           "-o", tmp_path / "x.json", "--no-qc")
    bundled = tmp_path / "x.json"
    bundled.write_text("{}")
    with pytest.raises(SystemExit):
        cli._refuse_call_path_aliases(args, str(bundled), str(bundled))
    assert "same file as --model" in capsys.readouterr().err
