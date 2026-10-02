"""An explicit --seq is checked against the reads; m6A enzymes need m6A calls.

Release audit 2026-09-29 (CLI P6):
- M15: an explicit --seq was never checked against the reads' MM specs, so
  ``--seq pacbio`` on Nanopore reads ran silently and gave ~100x more TF
  calls. It is now refused unless --force-seq is given.
- M16: calling Hia5 on reads without m6A MM calls (a DAF-seq BAM, stripped
  tags) exited 0 with no footprints and no warning. It is now refused.
- L10: with no platform evidence and no --seq the same warning printed three
  times (once here, twice from the bundled-model lookup).
"""
from __future__ import annotations

from types import SimpleNamespace

import pysam
import pytest

from fiberhmm.cli.common import resolve_platform_argument


def _bam(tmp_path, name, mm_style="nanopore", n_reads=8, seed=31):
    from conftest import make_synthetic_bam

    path = str(tmp_path / f"{name}.bam")
    source = make_synthetic_bam(str(tmp_path / f"{name}.src.bam"),
                                n_reads=n_reads, read_length=600, n_chroms=1,
                                chrom_length=20_000, seed=seed)
    with pysam.AlignmentFile(source, "rb") as src, \
            pysam.AlignmentFile(path, "wb", header=src.header) as out:
        for read in src.fetch(until_eof=True):
            if mm_style == "pacbio":
                read.set_tag("MM", read.get_tag("MM") + "T-a;")
            elif mm_style == "none":
                read.set_tag("MM", None)
                read.set_tag("ML", None)
            out.write(read)
    pysam.index(path)
    return path


def _args(enzyme="hia5", seq=None, force_seq=False):
    return SimpleNamespace(enzyme=enzyme, seq=seq, force_seq=force_seq)


@pytest.mark.parametrize("given,style", [("pacbio", "nanopore"),
                                         ("nanopore", "pacbio")])
def test_explicit_seq_contradicted_by_mm_specs_is_refused(
        tmp_path, capsys, given, style):
    bam = _bam(tmp_path, style, mm_style=style)
    with pytest.raises(SystemExit) as exc:
        resolve_platform_argument(_args(seq=given), bam, tool="fiberhmm-call")
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert f"--seq {given} was given, but the input looks like {style}" in err
    assert "--force-seq" in err


def test_force_seq_keeps_the_given_platform_with_a_warning(tmp_path, capsys):
    bam = _bam(tmp_path, "ont", mm_style="nanopore")
    args = _args(seq="pacbio", force_seq=True)
    resolve_platform_argument(args, bam, tool="fiberhmm-call")
    assert args.seq == "pacbio"
    assert "because of --force-seq" in capsys.readouterr().err


def test_matching_explicit_seq_is_silent(tmp_path, capsys):
    bam = _bam(tmp_path, "pb", mm_style="pacbio")
    args = _args(seq="pacbio")
    resolve_platform_argument(args, bam, tool="fiberhmm-call")
    assert args.seq == "pacbio"
    assert capsys.readouterr().err == ""


def test_explicit_seq_is_checked_even_when_the_header_declares_it(tmp_path):
    """A declaration that agrees with --seq does not hide contradicting reads."""
    from fiberhmm.io.bam_header import append_chemistry

    source = _bam(tmp_path, "ont", mm_style="nanopore")
    declared = str(tmp_path / "declared.bam")
    with pysam.AlignmentFile(source, "rb") as src:
        header = append_chemistry(src.header, {
            "assay": "fiber-seq", "enzyme": "hia5", "platform": "pacbio",
            "mode": "pacbio-fiber"})
        with pysam.AlignmentFile(declared, "wb", header=header) as out:
            for read in src.fetch(until_eof=True):
                out.write(read)
    with pytest.raises(SystemExit) as exc:
        resolve_platform_argument(_args(seq="pacbio"), declared, tool="t")
    assert exc.value.code == 2
    # Without --seq the declaration still settles the platform.
    args = _args()
    resolve_platform_argument(args, declared, tool="t")
    assert args.seq == "pacbio"


def test_hia5_on_reads_without_m6a_calls_is_refused(tmp_path, capsys):
    bam = _bam(tmp_path, "nomm", mm_style="none")
    for seq in ("pacbio", None):
        with pytest.raises(SystemExit) as exc:
            resolve_platform_argument(_args(seq=seq), bam, tool="fiberhmm-call")
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "none of the first 8 primary reads carries an m6A" in err
        assert "--enzyme dddb or --enzyme ddda" in err
    # DAF enzymes need no m6A calls.
    resolve_platform_argument(_args(enzyme="dddb", seq="pacbio"), bam, tool="t")


def test_no_evidence_warns_once_and_records_the_assumed_platform(
        tmp_path, capsys):
    from conftest import make_synthetic_bam

    empty = make_synthetic_bam(str(tmp_path / "empty.bam"), n_reads=0)
    args = _args()
    resolve_platform_argument(args, empty, tool="fiberhmm-call")
    assert args.seq == "pacbio"
    assert capsys.readouterr().err.count("assuming PacBio") == 1


def test_call_cli_refuses_pacbio_seq_on_nanopore_reads(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    bam = _bam(tmp_path, "ont", mm_style="nanopore")
    env = dict(os.environ, FIBERHMM_NO_UPDATE_CHECK="1",
               PYTHONPATH=str(repo) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    base = [sys.executable, "-m", "fiberhmm.cli.call", "-i", bam,
            "--enzyme", "hia5", "--seq", "pacbio", "--no-qc",
            "--min-read-length", "0", "-c", "1"]
    refused = subprocess.run(base + ["-o", str(tmp_path / "a.bam")],
                             capture_output=True, text=True, env=env, cwd=repo)
    assert refused.returncode == 2
    assert "--force-seq" in refused.stderr
    assert not (tmp_path / "a.bam").exists()
    forced = subprocess.run(base + ["-o", str(tmp_path / "b.bam"), "--force-seq"],
                            capture_output=True, text=True, env=env, cwd=repo)
    assert forced.returncode == 0, forced.stderr
