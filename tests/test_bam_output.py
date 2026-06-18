"""Tests for BAM/BED output helper behavior."""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fiberhmm.inference import bam_output


def test_samtools_index_error_requires_sort_detection():
    assert bam_output._samtools_index_error_requires_sort("records not sorted")
    assert bam_output._samtools_index_error_requires_sort("not in coordinate order")
    assert not bam_output._samtools_index_error_requires_sort("permission denied")
    assert not bam_output._samtools_index_error_requires_sort("")


def test_samtools_command_builders_preserve_thread_args():
    assert bam_output._samtools_index_cmd("out.bam", 8) == [
        "samtools", "index", "-@", "8", "out.bam",
    ]
    assert bam_output._samtools_sort_cmd("out.bam", "out.sorted.bam", 4) == [
        "samtools", "sort", "-@", "4", "-o", "out.sorted.bam", "out.bam",
    ]
    assert bam_output._samtools_cat_cmd(
        ["a.bam", "b.bam"], "out.bam", "list.txt",
    ) == [
        "samtools", "cat", "-h", "a.bam", "-b", "list.txt", "-o", "out.bam",
    ]
    assert bam_output._samtools_merge_cmd("out.bam", "list.txt") == [
        "samtools", "merge", "-f", "-b", "list.txt", "out.bam",
    ]


def test_sorted_bam_temp_path_preserves_bam_suffix_convention():
    assert bam_output._sorted_bam_temp_path("out.bam") == "out.sorted.bam"
    assert bam_output._sorted_bam_temp_path("out.BAM") == "out.sorted.bam"
    assert bam_output._sorted_bam_temp_path(Path("out.bam")) == "out.sorted.bam"
    assert bam_output._sorted_bam_temp_path("out.cram") == "out.cram.sorted.bam"


def test_file_size_gb_reports_binary_gigabytes(tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"0" * 1024)

    assert bam_output._file_size_gb(str(payload)) == pytest.approx(1024 / (1024 ** 3))


def test_total_file_size_gb_sums_binary_gigabytes(tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"0" * 512)
    second.write_bytes(b"1" * 256)

    assert bam_output._total_file_size_gb([str(first), str(second)]) == pytest.approx(
        768 / (1024 ** 3)
    )
    assert bam_output._total_file_size_gb([]) == 0.0


def test_throughput_gbs_handles_positive_and_non_positive_elapsed():
    assert bam_output._throughput_gbs(8.0, 4.0) == pytest.approx(2.0)
    assert bam_output._throughput_gbs(8.0, 0.0) == 0.0
    assert bam_output._throughput_gbs(8.0, -1.0) == 0.0


def test_try_pysam_index_reports_success(monkeypatch, capsys):
    indexed = []
    monkeypatch.setattr(bam_output.pysam, "index", lambda path: indexed.append(path))
    monkeypatch.setattr(bam_output.time, "time", lambda: 12.5)

    assert bam_output._try_pysam_index("out.bam", verbose=True, idx_start=10.0)

    assert indexed == ["out.bam"]
    assert "Index created (pysam) in 2.5s" in capsys.readouterr().out


def test_try_pysam_index_returns_false_on_samtools_error(monkeypatch, capsys):
    class FakeSamtoolsError(Exception):
        pass

    def fail_index(path):
        raise FakeSamtoolsError(path)

    monkeypatch.setattr(bam_output.pysam.utils, "SamtoolsError", FakeSamtoolsError)
    monkeypatch.setattr(bam_output.pysam, "index", fail_index)

    assert not bam_output._try_pysam_index("out.bam", verbose=True, idx_start=10.0)
    assert capsys.readouterr().out == ""


def test_sort_bam_with_fallback_uses_pysam_after_samtools_failure(monkeypatch, capsys):
    pysam_sorts = []

    def fail_samtools_sort(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "samtools sort")

    monkeypatch.setattr(bam_output, "_run_samtools_sort", fail_samtools_sort)
    monkeypatch.setattr(bam_output.pysam, "sort", lambda *args: pysam_sorts.append(args))
    monkeypatch.setattr(bam_output.time, "time", lambda: 12.0)

    bam_output._sort_bam_with_fallback(
        "out.bam",
        "out.sorted.bam",
        threads=4,
        verbose=True,
        bam_size_gb=2.0,
    )

    assert pysam_sorts == [("-o", "out.sorted.bam", "out.bam")]
    out = capsys.readouterr().out
    assert "Using pysam sort" in out
    assert "Sorted (pysam) in 0.0s" in out


def test_sort_bam_to_temp_and_replace_cleans_temp_after_replace(monkeypatch, tmp_path):
    output_bam = str(tmp_path / "out.bam")
    sorted_bam = tmp_path / "out.sorted.bam"
    calls = []

    def fake_sort(output, sorted_output, threads, verbose, bam_size_gb):
        calls.append(("sort", output, sorted_output, threads, verbose, bam_size_gb))
        sorted_bam.write_bytes(b"sorted")

    def fake_replace(source, destination):
        calls.append(("replace", source, destination))

    monkeypatch.setattr(bam_output, "_sort_bam_with_fallback", fake_sort)
    monkeypatch.setattr(bam_output.os, "replace", fake_replace)

    bam_output._sort_bam_to_temp_and_replace(
        output_bam,
        str(sorted_bam),
        threads=3,
        verbose=False,
        bam_size_gb=1.5,
    )

    assert calls == [
        ("sort", output_bam, str(sorted_bam), 3, False, 1.5),
        ("replace", str(sorted_bam), output_bam),
    ]
    assert not sorted_bam.exists()


def test_sort_bam_to_temp_and_replace_cleans_temp_after_sort_failure(monkeypatch, tmp_path):
    output_bam = str(tmp_path / "out.bam")
    sorted_bam = tmp_path / "out.sorted.bam"

    def fail_sort(*args, **kwargs):
        sorted_bam.write_bytes(b"partial")
        raise RuntimeError("sort failed")

    monkeypatch.setattr(bam_output, "_sort_bam_with_fallback", fail_sort)
    monkeypatch.setattr(
        bam_output.os,
        "replace",
        lambda *args: pytest.fail("unexpected replace"),
    )

    with pytest.raises(RuntimeError, match="sort failed"):
        bam_output._sort_bam_to_temp_and_replace(
            output_bam,
            str(sorted_bam),
            threads=3,
            verbose=False,
            bam_size_gb=1.5,
        )

    assert not sorted_bam.exists()


def test_index_sorted_bam_falls_back_to_pysam_when_samtools_missing(monkeypatch, capsys):
    indexed = []

    def fail_samtools_index(*args, **kwargs):
        raise FileNotFoundError("samtools")

    monkeypatch.setattr(bam_output, "_run_samtools_index", fail_samtools_index)
    monkeypatch.setattr(bam_output.pysam, "index", lambda path: indexed.append(path))
    monkeypatch.setattr(bam_output.time, "time", lambda: 22.0)

    bam_output._index_sorted_bam(
        "out.bam",
        threads=2,
        verbose=True,
        bam_size_gb=4.0,
    )

    assert indexed == ["out.bam"]
    out = capsys.readouterr().out
    assert "Indexing sorted BAM" in out
    assert "Index created in 0.0s" in out


def test_index_bam_if_already_sorted_uses_pysam_for_non_sort_errors(
    monkeypatch,
    capsys,
):
    pysam_attempts = []

    def fail_samtools_index(output_bam, threads):
        return subprocess.CompletedProcess(
            ["samtools", "index", output_bam],
            1,
            stdout="",
            stderr="permission denied",
        )

    def fake_try_pysam_index(output_bam, verbose, idx_start):
        pysam_attempts.append((output_bam, verbose, idx_start))
        return True

    monkeypatch.setattr(bam_output, "_run_samtools_index", fail_samtools_index)
    monkeypatch.setattr(bam_output, "_try_pysam_index", fake_try_pysam_index)
    monkeypatch.setattr(bam_output.time, "time", lambda: 5.0)

    indexed = bam_output._index_bam_if_already_sorted(
        "out.bam",
        threads=2,
        verbose=True,
        bam_size_gb=1.0,
    )

    assert indexed
    assert pysam_attempts == [("out.bam", True, 5.0)]
    out = capsys.readouterr().out
    assert "trying direct index" in out
    assert "permission denied, trying pysam" in out


def test_sorted_bed_temp_path_appends_sorted_suffix():
    assert bam_output._sorted_bed_temp_path("calls.bed") == "calls.bed.sorted"


def test_remove_file_if_exists_handles_missing_and_present_files(tmp_path):
    missing = tmp_path / "missing.tmp"
    bam_output._remove_file_if_exists(str(missing))

    present = tmp_path / "present.tmp"
    present.write_text("temporary")
    bam_output._remove_file_if_exists(str(present))

    assert not present.exists()


def test_remove_partial_output_bam_reports_success_and_filesystem_errors(
    monkeypatch,
    tmp_path,
    capsys,
):
    partial = tmp_path / "partial.bam"
    partial.write_text("partial")

    bam_output._remove_partial_output_bam(str(partial))

    assert not partial.exists()
    assert "Removed partial output file" in capsys.readouterr().out

    partial.write_text("partial")
    monkeypatch.setattr(
        bam_output.os,
        "remove",
        lambda path: (_ for _ in ()).throw(PermissionError("locked")),
    )

    bam_output._remove_partial_output_bam(str(partial))

    assert "Could not remove partial file: locked" in capsys.readouterr().out


def test_remove_partial_output_bam_propagates_unexpected_remove_errors(
    monkeypatch,
    tmp_path,
):
    partial = tmp_path / "partial.bam"
    partial.write_text("partial")
    monkeypatch.setattr(
        bam_output.os,
        "remove",
        lambda path: (_ for _ in ()).throw(RuntimeError("unexpected")),
    )

    with pytest.raises(RuntimeError, match="unexpected"):
        bam_output._remove_partial_output_bam(str(partial))


def test_raise_if_command_failed_preserves_process_output():
    ok = subprocess.CompletedProcess(["samtools"], 0, stdout="ok", stderr="")
    bam_output._raise_if_command_failed(ok, "samtools ok")

    failed = subprocess.CompletedProcess(
        ["samtools"],
        2,
        stdout="partial",
        stderr="failed",
    )
    with pytest.raises(subprocess.CalledProcessError) as exc:
        bam_output._raise_if_command_failed(failed, "samtools cat")

    assert exc.value.returncode == 2
    assert exc.value.cmd == "samtools cat"
    assert exc.value.output == "partial"
    assert exc.value.stderr == "failed"


def test_run_samtools_list_command_cleans_list_when_write_fails(monkeypatch, tmp_path):
    list_file = tmp_path / "inputs.txt"
    run_calls = []

    def fail_write(bam_files, got_list_file):
        assert got_list_file == str(list_file)
        list_file.write_text("partial\n")
        raise RuntimeError("list write failed")

    monkeypatch.setattr(bam_output, "_write_bam_list_file", fail_write)
    monkeypatch.setattr(
        bam_output.subprocess,
        "run",
        lambda *args, **kwargs: run_calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="list write failed"):
        bam_output._run_samtools_list_command(
            ["a.bam"],
            str(list_file),
            ["samtools", "cat"],
            "samtools cat",
        )

    assert run_calls == []
    assert not list_file.exists()


def test_concatenate_trivial_region_bams_handles_empty_single_and_many(monkeypatch):
    calls = []

    monkeypatch.setattr(
        bam_output,
        "_write_empty_bam_from_input_header",
        lambda input_bam, output_bam: calls.append(("empty", input_bam, output_bam)),
    )
    monkeypatch.setattr(
        bam_output.shutil,
        "copy",
        lambda source_bam, output_bam: calls.append(("copy", source_bam, output_bam)),
    )

    assert bam_output._concatenate_trivial_region_bams("input.bam", "out.bam", [])
    assert bam_output._concatenate_trivial_region_bams(
        "input.bam", "out.bam", ["region.bam"]
    )
    assert not bam_output._concatenate_trivial_region_bams(
        "input.bam", "out.bam", ["a.bam", "b.bam"]
    )
    assert calls == [
        ("empty", "input.bam", "out.bam"),
        ("copy", "region.bam", "out.bam"),
    ]


def test_prepare_pysam_concat_fallback_logs_stderr_and_prepares(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        bam_output,
        "_remove_partial_output_bam",
        lambda output_bam, verbose=True: calls.append(("remove", output_bam, verbose)),
    )
    monkeypatch.setattr(
        bam_output,
        "_ensure_output_dir_writable",
        lambda output_bam, verbose=True: calls.append(("probe", output_bam, verbose)),
    )
    error = subprocess.CalledProcessError(1, "samtools cat", stderr="bad sort\n")

    bam_output._prepare_pysam_concat_fallback("out.bam", error, verbose=True)

    captured = capsys.readouterr()
    assert "samtools cat failed: bad sort" in captured.out
    assert "Falling back to pysam" in captured.out
    assert calls == [
        ("remove", "out.bam", True),
        ("probe", "out.bam", True),
    ]


def test_prepare_pysam_concat_fallback_respects_quiet_mode(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        bam_output,
        "_remove_partial_output_bam",
        lambda output_bam, verbose=True: calls.append(("remove", output_bam, verbose)),
    )
    monkeypatch.setattr(
        bam_output,
        "_ensure_output_dir_writable",
        lambda output_bam, verbose=True: calls.append(("probe", output_bam, verbose)),
    )

    bam_output._prepare_pysam_concat_fallback(
        "out.bam", FileNotFoundError("samtools"), verbose=False,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert calls == [
        ("remove", "out.bam", False),
        ("probe", "out.bam", False),
    ]


def test_log_pysam_concat_failure_reports_output_context(tmp_path, capsys):
    output_bam = tmp_path / "out.bam"

    bam_output._log_pysam_concat_failure(str(output_bam), RuntimeError("read failed"))

    captured = capsys.readouterr()
    assert "pysam fallback also failed: read failed" in captured.out
    assert f"Output path: {output_bam}" in captured.out
    assert "Output dir exists: True" in captured.out
    assert "samtools merge" in captured.out


def test_bigbed_conversion_helpers_report_errors_and_score_types(monkeypatch, capsys):
    calls = []

    def fake_run(sorted_bed, chrom_sizes, output_bb, bed_type="bed12", autosql=None):
        calls.append((sorted_bed, chrom_sizes, output_bb, bed_type, autosql))
        return subprocess.CompletedProcess(
            ["bedToBigBed"], 1, stdout="", stderr="schema failed"
        )

    monkeypatch.setattr(bam_output, "_run_bed_to_bigbed", fake_run)

    assert bam_output._bed_type_for_scores(with_scores=False) == "bed12"
    assert bam_output._bed_type_for_scores(with_scores=True) == "bed12+1"
    assert not bam_output._convert_sorted_bed_to_bigbed(
        "calls.sorted",
        "chrom.sizes",
        "calls.bb",
        bed_type="bed12+1",
        autosql="schema.as",
    )

    assert calls == [
        ("calls.sorted", "chrom.sizes", "calls.bb", "bed12+1", "schema.as")
    ]
    assert "bedToBigBed error: schema failed" in capsys.readouterr().out


def test_convert_bigbed_score_fallback_reports_success(monkeypatch, capsys):
    calls = []

    def fake_run(sorted_bed, chrom_sizes, output_bb, bed_type="bed12", autosql=None):
        calls.append((sorted_bed, chrom_sizes, output_bb, bed_type, autosql))
        return subprocess.CompletedProcess(["bedToBigBed"], 0, stdout="", stderr="")

    monkeypatch.setattr(bam_output, "_run_bed_to_bigbed", fake_run)

    assert bam_output._convert_bigbed_score_fallback(
        "calls.sorted", "chrom.sizes", "calls.bb"
    )
    assert calls == [
        ("calls.sorted", "chrom.sizes", "calls.bb", "bed12", None)
    ]
    out = capsys.readouterr().out
    assert "Trying fallback to standard BED12" in out
    assert "Fallback succeeded" in out


def test_convert_to_bigbed_sorts_without_shell(monkeypatch, tmp_path):
    bed = tmp_path / "calls.bed"
    chrom_sizes = tmp_path / "chrom.sizes"
    output_bb = tmp_path / "calls.bb"
    bed.write_text("chr2\t20\t30\nchr1\t10\t20\n")
    chrom_sizes.write_text("chr1\t100\nchr2\t100\n")
    sorted_bed = Path(str(bed) + ".sorted")
    calls = []

    monkeypatch.setattr(bam_output.shutil, "which", lambda name: f"/bin/{name}")

    def fake_run(cmd, *args, **kwargs):
        calls.append((cmd, kwargs))
        assert kwargs.get("shell") is not True
        if cmd[0] == "sort":
            assert cmd == [
                "sort", "-k1,1", "-k2,2n", "-o", str(sorted_bed), str(bed)
            ]
            sorted_bed.write_text(bed.read_text())
            return subprocess.CompletedProcess(cmd, 0)
        if cmd[0] == "bedToBigBed":
            assert cmd == [
                "bedToBigBed", "-type=bed12", str(sorted_bed),
                str(chrom_sizes), str(output_bb)
            ]
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(bam_output.subprocess, "run", fake_run)

    assert bam_output.convert_to_bigbed(str(bed), str(chrom_sizes), str(output_bb))
    assert [cmd[0] for cmd, _ in calls] == ["sort", "bedToBigBed"]
    assert not sorted_bed.exists()


def test_convert_to_bigbed_removes_failed_output(monkeypatch, tmp_path):
    bed = tmp_path / "calls.bed"
    chrom_sizes = tmp_path / "chrom.sizes"
    output_bb = tmp_path / "calls.bb"
    sorted_bed = Path(str(bed) + ".sorted")
    bed.write_text("chr1\t10\t20\n")
    chrom_sizes.write_text("chr1\t100\n")
    output_bb.write_text("stale")

    monkeypatch.setattr(bam_output.shutil, "which", lambda name: f"/bin/{name}")

    def fake_run(cmd, *args, **kwargs):
        if cmd[0] == "sort":
            sorted_bed.write_text(bed.read_text())
            return subprocess.CompletedProcess(cmd, 0)
        if cmd[0] == "bedToBigBed":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="bad input")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(bam_output.subprocess, "run", fake_run)

    assert not bam_output.convert_to_bigbed(
        str(bed), str(chrom_sizes), str(output_bb),
    )
    assert not output_bb.exists()
    assert not sorted_bed.exists()


def test_convert_to_bigbed_with_schema_falls_back_without_shell(monkeypatch, tmp_path):
    bed = tmp_path / "calls.bed"
    chrom_sizes = tmp_path / "chrom.sizes"
    autosql = tmp_path / "schema.as"
    output_bb = tmp_path / "calls.bb"
    bed.write_text("chr1\t10\t20\n")
    chrom_sizes.write_text("chr1\t100\n")
    autosql.write_text("table fiberFootprints\n")
    sorted_bed = Path(str(bed) + ".sorted")
    bed_to_bigbed_types = []

    monkeypatch.setattr(bam_output.shutil, "which", lambda name: f"/bin/{name}")

    def fake_run(cmd, *args, **kwargs):
        assert kwargs.get("shell") is not True
        if cmd[0] == "sort":
            sorted_bed.write_text(bed.read_text())
            return subprocess.CompletedProcess(cmd, 0)
        if cmd[0] == "bedToBigBed":
            type_arg = next(arg for arg in cmd if arg.startswith("-type="))
            bed_to_bigbed_types.append(type_arg)
            if type_arg == "-type=bed12+1":
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="schema failed")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(bam_output.subprocess, "run", fake_run)

    assert bam_output.convert_to_bigbed_with_schema(
        str(bed), str(chrom_sizes), str(autosql), str(output_bb), with_scores=True
    )
    assert bed_to_bigbed_types == ["-type=bed12+1", "-type=bed12"]
    assert not sorted_bed.exists()


def test_convert_bed_with_sorted_temp_cleans_after_converter_error(monkeypatch, tmp_path):
    bed = tmp_path / "calls.bed"
    sorted_bed = Path(str(bed) + ".sorted")
    bed.write_text("chr1\t10\t20\n")

    def fake_sort(source, destination):
        assert source == str(bed)
        assert destination == str(sorted_bed)
        sorted_bed.write_text("sorted")

    def fail_convert(got_sorted_bed):
        assert got_sorted_bed == str(sorted_bed)
        raise RuntimeError("converter failed")

    monkeypatch.setattr(bam_output, "_sort_bed_for_bigbed", fake_sort)

    with pytest.raises(RuntimeError, match="converter failed"):
        bam_output._convert_bed_with_sorted_temp(str(bed), fail_convert)

    assert not sorted_bed.exists()


def test_bed12_columns_and_autosql_schema_include_scores_only_when_requested(tmp_path):
    assert bam_output._bed12_record_columns(with_scores=False)[-1] == "blockStarts"
    assert bam_output._bed12_record_columns(with_scores=True)[-1] == "blockScores"

    plain_schema = bam_output._autosql_schema(with_scores=False)
    scored_schema = bam_output._autosql_schema(with_scores=True)
    assert "blockScores" not in plain_schema
    assert "blockScores" in scored_schema

    schema_path = tmp_path / "schema.as"
    bam_output.write_autosql_schema(str(schema_path), with_scores=True)
    assert schema_path.read_text() == scored_schema


def test_extract_bed_from_tagged_bam_closes_bam_when_output_open_fails(
    monkeypatch, tmp_path
):
    opened = []

    class FakeAlignmentFile:
        def __init__(self, *args, **kwargs):
            self.closed = False
            opened.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()
            return False

        def __iter__(self):
            return iter(())

        def close(self):
            self.closed = True

    def fail_open(*args, **kwargs):
        raise OSError("bed open failed")

    monkeypatch.setattr(bam_output.pysam, "AlignmentFile", FakeAlignmentFile)
    monkeypatch.setattr(bam_output, "open", fail_open, raising=False)

    with pytest.raises(OSError, match="bed open failed"):
        bam_output.extract_bed_from_tagged_bam("input.bam", str(tmp_path / "out.bed"))

    assert len(opened) == 1
    assert opened[0].closed


def test_format_footprint_bed12_row_preserves_scores_and_item_rgb():
    row = bam_output._format_footprint_bed12_row(
        chrom="chr1",
        chrom_start=100,
        chrom_end=150,
        name="read1",
        strand="+",
        block_starts=[0, 40],
        block_sizes=[10, 10],
        valid_scores=[255, 128],
        with_scores=True,
    )

    assert row.split("\t") == [
        "chr1",
        "100",
        "150",
        "read1",
        "2",
        "+",
        "100",
        "150",
        "0,0,0",
        "2",
        "10,10",
        "0,40",
        "1000,501",
    ]


def test_format_bed_score_column_scales_quality_bytes_to_bed_range():
    assert bam_output._format_bed_score_column([0, 128, 255]) == "0,501,1000"


def test_format_footprint_bed12_row_omits_scores_when_disabled():
    row = bam_output._format_footprint_bed12_row(
        chrom="chr1",
        chrom_start=100,
        chrom_end=150,
        name="read1",
        strand="-",
        block_starts=[0],
        block_sizes=[50],
        valid_scores=[255],
        with_scores=False,
    )

    assert len(row.split("\t")) == 12
    assert row.split("\t")[4] == "1"


def test_normalize_bed12_blocks_sorts_merges_scores_and_pads():
    starts, sizes, scores = bam_output._normalize_bed12_blocks(
        block_starts=[20, 0, 5],
        block_sizes=[5, 10, 20],
        valid_scores=[100, 50, 80],
        read_length=30,
    )

    assert starts == [0, 29]
    assert sizes == [25, 1]
    assert scores == [82, 0]


def test_bed12_components_from_ref_blocks_builds_offsets_and_scores():
    assert bam_output._bed12_components_from_ref_blocks([], with_scores=True) is None
    assert bam_output._bed12_components_from_ref_blocks(
        [(120, 130, 255), (100, 105, 128)],
        with_scores=True,
    ) == (100, 130, [20, 0], [10, 5], [255, 128])
    assert bam_output._bed12_components_from_ref_blocks(
        [(120, 130, 255)],
        with_scores=False,
    ) == (120, 130, [0], [10], [])


def test_footprint_bed12_line_from_read_projects_and_formats(monkeypatch):
    read = SimpleNamespace(
        reference_name="chr1",
        is_reverse=True,
        query_name="read1",
    )
    ns = [10, 40]
    nl = [5, 10]
    nq = [255, 128]

    monkeypatch.setattr(bam_output, "build_query_to_ref", lambda got_read: {
        "read": got_read,
    })

    def fake_scored_interval_spans(got_ns, got_nl, got_scores, query_to_ref):
        assert got_ns is ns
        assert got_nl is nl
        assert got_scores is nq
        assert query_to_ref == {"read": read}
        return [(120, 130, 255), (100, 105, 128)]

    monkeypatch.setattr(bam_output, "scored_interval_spans", fake_scored_interval_spans)

    row = bam_output._footprint_bed12_line_from_read(read, ns, nl, nq, with_scores=True)

    assert row.split("\t") == [
        "chr1",
        "100",
        "130",
        "read1",
        "2",
        "-",
        "100",
        "130",
        "0,0,0",
        "2",
        "5,10",
        "0,20",
        "501,1000",
    ]


def test_footprint_bed12_line_from_read_returns_none_without_ref_blocks(monkeypatch):
    read = SimpleNamespace(reference_name="chr1", is_reverse=False, query_name="read1")

    monkeypatch.setattr(bam_output, "build_query_to_ref", lambda got_read: {})
    monkeypatch.setattr(bam_output, "scored_interval_spans", lambda *_: [])

    assert bam_output._footprint_bed12_line_from_read(
        read, [10], [5], [255], with_scores=True,
    ) is None


def test_read_footprint_tags_for_bed_handles_missing_empty_and_scores(monkeypatch):
    class Read:
        def __init__(self, tags):
            self.tags = tags

        def get_tag(self, key):
            if key not in self.tags:
                raise KeyError(key)
            return self.tags[key]

    assert bam_output._read_footprint_tags_for_bed(Read({}), with_scores=True) is None

    monkeypatch.setattr(bam_output, "flip_intervals_to_seq", lambda *_: ([], []))
    assert bam_output._read_footprint_tags_for_bed(
        Read({"ns": [1], "nl": [2]}),
        with_scores=True,
    ) is None

    monkeypatch.setattr(bam_output, "flip_intervals_to_seq", lambda *_: ([10], [5]))
    assert bam_output._read_footprint_tags_for_bed(
        Read({"ns": [1], "nl": [2]}),
        with_scores=True,
    ) == ([10], [5], None)
    assert bam_output._read_footprint_tags_for_bed(
        Read({"ns": [1], "nl": [2], "nq": [255]}),
        with_scores=True,
    ) == ([10], [5], [255])


def test_bed12_line_from_tagged_read_filters_and_formats(monkeypatch):
    read = SimpleNamespace(
        is_unmapped=False,
        is_secondary=False,
        is_supplementary=False,
    )
    calls = []

    monkeypatch.setattr(
        bam_output,
        "_read_footprint_tags_for_bed",
        lambda got_read, with_scores: ([10], [5], [255]),
    )

    def fake_line(got_read, ns, nl, nq, with_scores):
        calls.append((got_read, ns, nl, nq, with_scores))
        return "chr1\t0\t5"

    monkeypatch.setattr(bam_output, "_footprint_bed12_line_from_read", fake_line)

    assert bam_output._bed12_line_from_tagged_read(
        read, with_scores=True,
    ) == "chr1\t0\t5"
    assert calls == [(read, [10], [5], [255], True)]

    read.is_secondary = True
    assert bam_output._bed12_line_from_tagged_read(read, with_scores=True) is None


def test_bed12_line_from_tagged_read_returns_none_without_tags(monkeypatch):
    read = SimpleNamespace(
        is_unmapped=False,
        is_secondary=False,
        is_supplementary=False,
    )

    monkeypatch.setattr(
        bam_output,
        "_read_footprint_tags_for_bed",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        bam_output,
        "_footprint_bed12_line_from_read",
        lambda *args, **kwargs: pytest.fail("unexpected formatter call"),
    )

    assert bam_output._bed12_line_from_tagged_read(read, with_scores=False) is None


@pytest.mark.parametrize(
    "write_scores",
    [bam_output.create_scores_database, bam_output.append_to_scores_database],
)
def test_scores_database_closes_connection_when_record_parsing_fails(
    monkeypatch, write_scores
):
    class FakeCursor:
        def execute(self, *args, **kwargs):
            return None

    class FakeConnection:
        def __init__(self):
            self.closed = False
            self.committed = False

        def cursor(self):
            return FakeCursor()

        def commit(self):
            self.committed = True

        def close(self):
            self.closed = True

    connection = FakeConnection()
    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: connection)

    with pytest.raises(KeyError):
        write_scores([{}], "scores.db")

    assert connection.closed
    assert not connection.committed


def test_parse_int_csv_skips_empty_fields():
    assert bam_output._parse_int_csv("10,20,") == [10, 20]
    assert bam_output._parse_int_csv("10, , 20,") == [10, 20]
    assert bam_output._parse_int_csv("") == []
    assert bam_output._parse_int_csv("  ") == []


def test_mean_block_score_uses_zero_for_empty_scores():
    assert bam_output._mean_block_score([100, 200]) == 150
    assert bam_output._mean_block_score(np.asarray([100, 200])) == 150
    assert bam_output._mean_block_score([]) == 0


def test_score_database_row_helpers_shape_read_and_footprint_values():
    record = {
        "name": "read-a",
        "chrom": "chr1",
        "chromStart": 100,
        "chromEnd": 150,
        "strand": "+",
        "blockCount": 2,
    }

    assert bam_output._score_read_row(record, 150.0) == (
        "read-a", "chr1", 100, 150, "+", 2, 150.0,
    )
    assert bam_output._footprint_score_rows(
        "read-a", [0, 30], [10, 20], [100, 200],
    ) == [
        ("read-a", 0, 0, 10, 100),
        ("read-a", 1, 30, 20, 200),
    ]


def test_scores_database_create_and_append_write_expected_rows(tmp_path):
    db_path = str(tmp_path / "scores.db")
    first = {
        "name": "read-a",
        "chrom": "chr1",
        "chromStart": 100,
        "chromEnd": 150,
        "strand": "+",
        "blockCount": 2,
        "blockSizes": "10,20",
        "blockStarts": "0,30",
        "blockScores": "100,200",
    }
    second = {
        "name": "read-b",
        "chrom": "chr2",
        "chromStart": 20,
        "chromEnd": 30,
        "strand": "-",
        "blockCount": 1,
        "blockSizes": "5",
        "blockStarts": "2",
    }

    bam_output.create_scores_database([first], db_path)
    bam_output.append_to_scores_database([second], db_path)

    with sqlite3.connect(db_path) as conn:
        reads = conn.execute(
            "SELECT read_id, chrom, start, end, strand, n_footprints, mean_score "
            "FROM reads ORDER BY read_id"
        ).fetchall()
        footprints = conn.execute(
            "SELECT read_id, footprint_idx, rel_start, size, score "
            "FROM footprints ORDER BY read_id, footprint_idx"
        ).fetchall()

    assert reads == [
        ("read-a", "chr1", 100, 150, "+", 2, 150.0),
        ("read-b", "chr2", 20, 30, "-", 1, 0.0),
    ]
    assert footprints == [
        ("read-a", 0, 0, 10, 100),
        ("read-a", 1, 30, 20, 200),
        ("read-b", 0, 2, 5, 0),
    ]


def test_samtools_cat_bams_writes_list_and_cleans_on_success(monkeypatch, tmp_path):
    inputs = [str(tmp_path / f"region_{i}.bam") for i in range(3)]
    output_bam = str(tmp_path / "out.bam")
    list_file = str(tmp_path / "bam_list.txt")

    def fake_run(cmd, *args, **kwargs):
        # -h <first> forces one header (avoids @PG duplication across regions)
        assert cmd == ["samtools", "cat", "-h", inputs[0], "-b", list_file,
                       "-o", output_bam]
        assert kwargs == {"capture_output": True, "text": True}
        assert Path(list_file).read_text().splitlines() == inputs
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(bam_output.subprocess, "run", fake_run)

    bam_output._samtools_cat_bams(inputs, output_bam, list_file)

    assert not Path(list_file).exists()


def test_samtools_cat_bams_cleans_list_on_command_failure(monkeypatch, tmp_path):
    inputs = [str(tmp_path / "a.bam"), str(tmp_path / "b.bam")]
    output_bam = str(tmp_path / "out.bam")
    list_file = str(tmp_path / "bam_list.txt")

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="partial", stderr="cat failed")

    monkeypatch.setattr(bam_output.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError) as exc:
        bam_output._samtools_cat_bams(inputs, output_bam, list_file)

    assert exc.value.stderr == "cat failed"
    assert not Path(list_file).exists()


def test_samtools_cat_bams_cleans_list_when_samtools_missing(monkeypatch, tmp_path):
    inputs = [str(tmp_path / "a.bam"), str(tmp_path / "b.bam")]
    output_bam = str(tmp_path / "out.bam")
    list_file = str(tmp_path / "bam_list.txt")

    def fake_run(cmd, *args, **kwargs):
        raise FileNotFoundError("samtools")

    monkeypatch.setattr(bam_output.subprocess, "run", fake_run)

    with pytest.raises(FileNotFoundError):
        bam_output._samtools_cat_bams(inputs, output_bam, list_file)

    assert not Path(list_file).exists()


def test_samtools_merge_bams_writes_list_and_cleans_on_success(monkeypatch, tmp_path):
    inputs = [str(tmp_path / f"region_{i}.bam") for i in range(3)]
    output_bam = str(tmp_path / "out.bam")
    list_file = str(tmp_path / "bam_list.txt")

    def fake_run(cmd, *args, **kwargs):
        assert cmd == ["samtools", "merge", "-f", "-b", list_file, output_bam]
        assert kwargs == {"capture_output": True, "text": True}
        assert Path(list_file).read_text().splitlines() == inputs
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(bam_output.subprocess, "run", fake_run)

    bam_output._samtools_merge_bams(inputs, output_bam, list_file)

    assert not Path(list_file).exists()


def test_samtools_merge_bams_cleans_list_on_command_failure(monkeypatch, tmp_path):
    inputs = [str(tmp_path / "a.bam"), str(tmp_path / "b.bam")]
    output_bam = str(tmp_path / "out.bam")
    list_file = str(tmp_path / "bam_list.txt")

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="partial", stderr="merge failed")

    monkeypatch.setattr(bam_output.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError) as exc:
        bam_output._samtools_merge_bams(inputs, output_bam, list_file)

    assert exc.value.stderr == "merge failed"
    assert not Path(list_file).exists()


def test_samtools_merge_bams_cleans_list_when_samtools_missing(monkeypatch, tmp_path):
    inputs = [str(tmp_path / "a.bam"), str(tmp_path / "b.bam")]
    output_bam = str(tmp_path / "out.bam")
    list_file = str(tmp_path / "bam_list.txt")

    def fake_run(cmd, *args, **kwargs):
        raise FileNotFoundError("samtools")

    monkeypatch.setattr(bam_output.subprocess, "run", fake_run)

    with pytest.raises(FileNotFoundError):
        bam_output._samtools_merge_bams(inputs, output_bam, list_file)

    assert not Path(list_file).exists()


def test_concat_success_logging_reports_method_size_and_speed(monkeypatch, capsys):
    monkeypatch.setattr(bam_output, "_file_size_gb", lambda path: 2.0)

    bam_output._log_samtools_cat_success("out.bam", 4.0)
    bam_output._log_concat_method_success("pysam", 1.25)

    assert capsys.readouterr().out.splitlines() == [
        "  Concatenated with samtools cat in 4.0s (2.0GB, 0.50 GB/s)",
        "  Concatenated with pysam in 1.2s",
    ]


def test_concatenate_region_bams_writes_empty_bam_from_input_header(monkeypatch, tmp_path):
    input_bam = str(tmp_path / "input.bam")
    output_bam = str(tmp_path / "out.bam")
    opened = []
    header = {"HD": {"VN": "1.6"}}

    class FakeAlignmentFile:
        def __init__(self, path, mode, *args, **kwargs):
            opened.append((path, mode, kwargs))
            self.header = header

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(bam_output.pysam, "AlignmentFile", FakeAlignmentFile)

    bam_output._concatenate_region_bams(input_bam, output_bam, [], str(tmp_path), verbose=False)

    assert opened == [
        (input_bam, "rb", {"check_sq": False}),
        (output_bam, "wb", {"header": header}),
    ]


def test_concatenate_region_bams_copies_single_bam(tmp_path):
    input_bam = str(tmp_path / "input.bam")
    source_bam = tmp_path / "region_0.bam"
    output_bam = tmp_path / "out.bam"
    source_bam.write_bytes(b"region")

    bam_output._concatenate_region_bams(
        input_bam, str(output_bam), [str(source_bam)], str(tmp_path), verbose=False
    )

    assert output_bam.read_bytes() == b"region"


def test_concatenate_region_bams_uses_samtools_cat_first(monkeypatch, tmp_path):
    input_bam = str(tmp_path / "input.bam")
    output_bam = str(tmp_path / "out.bam")
    bam_files = []
    cat_calls = []

    for i in range(2):
        bam_path = tmp_path / f"region_{i}.bam"
        bam_path.write_bytes(b"region")
        bam_files.append(str(bam_path))

    def fake_cat(inputs, output, list_file):
        cat_calls.append((inputs, output, list_file))
        Path(output).write_bytes(b"cat")

    monkeypatch.setattr(bam_output, "_samtools_cat_bams", fake_cat)
    monkeypatch.setattr(
        bam_output, "_concatenate_bams_with_pysam",
        lambda *args, **kwargs: pytest.fail("unexpected pysam fallback")
    )
    monkeypatch.setattr(
        bam_output, "_samtools_merge_bams",
        lambda *args, **kwargs: pytest.fail("unexpected merge fallback")
    )

    bam_output._concatenate_region_bams(
        input_bam, output_bam, bam_files, str(tmp_path), verbose=False,
    )

    assert output_bam and Path(output_bam).read_bytes() == b"cat"
    assert cat_calls == [(bam_files, output_bam, str(tmp_path / "bam_list.txt"))]


def test_concatenate_region_bams_removes_partial_before_pysam_fallback(monkeypatch, tmp_path):
    input_bam = str(tmp_path / "input.bam")
    output_bam = str(tmp_path / "out.bam")
    bam_files = []

    for i in range(2):
        bam_path = tmp_path / f"region_{i}.bam"
        bam_path.write_bytes(b"region")
        bam_files.append(str(bam_path))

    def fake_cat(inputs, output, list_file):
        Path(output).write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, "samtools cat", stderr="cat failed")

    def fake_pysam(inputs, output, *args, **kwargs):
        assert inputs == bam_files
        assert not Path(output).exists()
        Path(output).write_bytes(b"pysam")

    monkeypatch.setattr(bam_output, "_samtools_cat_bams", fake_cat)
    monkeypatch.setattr(bam_output, "_concatenate_bams_with_pysam", fake_pysam)
    monkeypatch.setattr(
        bam_output, "_samtools_merge_bams",
        lambda *args, **kwargs: pytest.fail("unexpected merge fallback")
    )

    bam_output._concatenate_region_bams(
        input_bam, output_bam, bam_files, str(tmp_path), verbose=False,
    )

    assert Path(output_bam).read_bytes() == b"pysam"


def test_concatenate_region_bams_creates_output_dir_for_pysam_fallback(monkeypatch, tmp_path):
    input_bam = str(tmp_path / "input.bam")
    output_bam = str(tmp_path / "missing" / "out.bam")
    bam_files = []

    for i in range(2):
        bam_path = tmp_path / f"region_{i}.bam"
        bam_path.write_bytes(b"region")
        bam_files.append(str(bam_path))

    def fake_cat(inputs, output, list_file):
        raise FileNotFoundError("samtools")

    def fake_pysam(inputs, output, *args, **kwargs):
        assert Path(output).parent.exists()
        Path(output).write_bytes(b"pysam")

    monkeypatch.setattr(bam_output, "_samtools_cat_bams", fake_cat)
    monkeypatch.setattr(bam_output, "_concatenate_bams_with_pysam", fake_pysam)

    bam_output._concatenate_region_bams(
        input_bam, output_bam, bam_files, str(tmp_path), verbose=False,
    )

    assert Path(output_bam).read_bytes() == b"pysam"


def test_concatenate_region_bams_uses_merge_after_pysam_failure(monkeypatch, tmp_path):
    input_bam = str(tmp_path / "input.bam")
    output_bam = str(tmp_path / "out.bam")
    bam_files = []
    merge_calls = []

    for i in range(2):
        bam_path = tmp_path / f"region_{i}.bam"
        bam_path.write_bytes(b"region")
        bam_files.append(str(bam_path))

    def fake_cat(inputs, output, list_file):
        raise subprocess.CalledProcessError(1, "samtools cat", stderr="cat failed")

    def fake_pysam(inputs, output, *args, **kwargs):
        raise RuntimeError("pysam failed")

    def fake_merge(inputs, output, list_file):
        merge_calls.append((inputs, output, list_file))
        Path(output).write_bytes(b"merge")

    monkeypatch.setattr(bam_output, "_samtools_cat_bams", fake_cat)
    monkeypatch.setattr(bam_output, "_concatenate_bams_with_pysam", fake_pysam)
    monkeypatch.setattr(bam_output, "_samtools_merge_bams", fake_merge)

    bam_output._concatenate_region_bams(
        input_bam, output_bam, bam_files, str(tmp_path), verbose=False,
    )

    assert Path(output_bam).read_bytes() == b"merge"
    assert merge_calls == [(bam_files, output_bam, str(tmp_path / "bam_list.txt"))]


def test_sort_and_index_bam_direct_samtools_index_success(monkeypatch, tmp_path):
    output_bam = tmp_path / "out.bam"
    output_bam.write_bytes(b"bam")
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(bam_output.subprocess, "run", fake_run)
    monkeypatch.setattr(
        bam_output.pysam,
        "index",
        lambda *args, **kwargs: pytest.fail("unexpected pysam.index"),
    )
    monkeypatch.setattr(
        bam_output.pysam,
        "sort",
        lambda *args, **kwargs: pytest.fail("unexpected pysam.sort"),
    )

    bam_output._sort_and_index_bam(str(output_bam), verbose=False, threads=3)

    assert calls == [
        (["samtools", "index", "-@", "3", str(output_bam)],
         {"check": False, "capture_output": True, "text": True})
    ]


def test_sort_and_index_bam_sorts_when_direct_index_reports_unsorted(monkeypatch, tmp_path):
    output_bam = tmp_path / "out.bam"
    sorted_bam = tmp_path / "out.sorted.bam"
    output_bam.write_bytes(b"bam")
    calls = []
    replacements = []

    def fake_run(cmd, *args, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[1] == "index" and len(calls) == 1:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="not sorted")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(bam_output.subprocess, "run", fake_run)
    monkeypatch.setattr(
        bam_output.os,
        "replace",
        lambda src, dst: replacements.append((src, dst)),
    )
    monkeypatch.setattr(
        bam_output.pysam,
        "index",
        lambda *args, **kwargs: pytest.fail("unexpected pysam.index"),
    )
    monkeypatch.setattr(
        bam_output.pysam,
        "sort",
        lambda *args, **kwargs: pytest.fail("unexpected pysam.sort"),
    )

    bam_output._sort_and_index_bam(str(output_bam), verbose=False, threads=2)

    assert calls == [
        (["samtools", "index", "-@", "2", str(output_bam)],
         {"check": False, "capture_output": True, "text": True}),
        (["samtools", "sort", "-@", "2", "-o", str(sorted_bam), str(output_bam)],
         {"capture_output": True, "text": True}),
        (["samtools", "index", "-@", "2", str(output_bam)],
         {"check": True, "capture_output": True, "text": True}),
    ]
    assert replacements == [(str(sorted_bam), str(output_bam))]


def test_sort_and_index_bam_falls_back_to_pysam_when_samtools_missing(monkeypatch, tmp_path):
    output_bam = tmp_path / "out.bam"
    output_bam.write_bytes(b"bam")
    indexed = []

    def fake_run(cmd, *args, **kwargs):
        raise FileNotFoundError("samtools")

    monkeypatch.setattr(bam_output.subprocess, "run", fake_run)
    monkeypatch.setattr(bam_output.pysam, "index", lambda path: indexed.append(path))
    monkeypatch.setattr(
        bam_output.pysam,
        "sort",
        lambda *args, **kwargs: pytest.fail("unexpected pysam.sort"),
    )

    bam_output._sort_and_index_bam(str(output_bam), verbose=False, threads=4)

    assert indexed == [str(output_bam)]


def test_sort_and_index_bam_uses_pysam_sort_when_samtools_sort_fails(monkeypatch, tmp_path):
    output_bam = tmp_path / "out.bam"
    sorted_bam = tmp_path / "out.sorted.bam"
    output_bam.write_bytes(b"bam")
    calls = []
    pysam_sorts = []
    replacements = []

    def fake_run(cmd, *args, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[1] == "index" and len(calls) == 1:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="coordinate not sorted")
        if cmd[1] == "sort":
            return subprocess.CompletedProcess(cmd, 1, stdout="partial", stderr="sort failed")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(bam_output.subprocess, "run", fake_run)
    monkeypatch.setattr(bam_output.pysam, "sort", lambda *args: pysam_sorts.append(args))
    monkeypatch.setattr(
        bam_output.pysam,
        "index",
        lambda *args, **kwargs: pytest.fail("unexpected pysam.index"),
    )
    monkeypatch.setattr(
        bam_output.os,
        "replace",
        lambda src, dst: replacements.append((src, dst)),
    )

    bam_output._sort_and_index_bam(str(output_bam), verbose=False, threads=2)

    assert pysam_sorts == [("-o", str(sorted_bam), str(output_bam))]
    assert replacements == [(str(sorted_bam), str(output_bam))]
