"""Tests for BAM/BED output helper behavior."""

from __future__ import annotations

import subprocess
from pathlib import Path

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
