"""Tests for BAM/BED output helper behavior."""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

import pytest

from fiberhmm.inference import bam_output


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

    bam_output._concatenate_region_bams(input_bam, output_bam, bam_files, str(tmp_path), verbose=False)

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

    bam_output._concatenate_region_bams(input_bam, output_bam, bam_files, str(tmp_path), verbose=False)

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

    bam_output._concatenate_region_bams(input_bam, output_bam, bam_files, str(tmp_path), verbose=False)

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

    bam_output._concatenate_region_bams(input_bam, output_bam, bam_files, str(tmp_path), verbose=False)

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
    monkeypatch.setattr(bam_output.pysam, "index", lambda *args, **kwargs: pytest.fail("unexpected pysam.index"))
    monkeypatch.setattr(bam_output.pysam, "sort", lambda *args, **kwargs: pytest.fail("unexpected pysam.sort"))

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
    monkeypatch.setattr(bam_output.os, "replace", lambda src, dst: replacements.append((src, dst)))
    monkeypatch.setattr(bam_output.pysam, "index", lambda *args, **kwargs: pytest.fail("unexpected pysam.index"))
    monkeypatch.setattr(bam_output.pysam, "sort", lambda *args, **kwargs: pytest.fail("unexpected pysam.sort"))

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
    monkeypatch.setattr(bam_output.pysam, "sort", lambda *args, **kwargs: pytest.fail("unexpected pysam.sort"))

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
    monkeypatch.setattr(bam_output.pysam, "index", lambda *args, **kwargs: pytest.fail("unexpected pysam.index"))
    monkeypatch.setattr(bam_output.os, "replace", lambda src, dst: replacements.append((src, dst)))

    bam_output._sort_and_index_bam(str(output_bam), verbose=False, threads=2)

    assert pysam_sorts == [("-o", str(sorted_bam), str(output_bam))]
    assert replacements == [(str(sorted_bam), str(output_bam))]
