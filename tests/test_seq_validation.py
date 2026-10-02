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


def _declare(tmp_path, source, platform):
    from fiberhmm.io.bam_header import append_chemistry

    declared = str(tmp_path / f"declared_{platform}.bam")
    with pysam.AlignmentFile(source, "rb") as src:
        header = append_chemistry(src.header, {
            "assay": "fiber-seq", "enzyme": "hia5", "platform": platform,
            "mode": f"{platform}-fiber"})
        with pysam.AlignmentFile(declared, "wb", header=header) as out:
            for read in src.fetch(until_eof=True):
                out.write(read)
    return declared


def test_a_chemistry_declaration_decides_like_the_auto_path(tmp_path):
    """A sample without T-a calls does not prove Nanopore origin: a header
    declaring PacBio accepts --seq pacbio (as omitting --seq does) and
    refuses --seq nanopore."""
    source = _bam(tmp_path, "ont", mm_style="nanopore")
    declared = _declare(tmp_path, source, "pacbio")
    args = _args(seq="pacbio")
    resolve_platform_argument(args, declared, tool="t")
    assert args.seq == "pacbio"
    with pytest.raises(SystemExit) as exc:
        resolve_platform_argument(_args(seq="nanopore"), declared, tool="t")
    assert exc.value.code == 2
    args = _args()
    resolve_platform_argument(args, declared, tool="t")
    assert args.seq == "pacbio"


@pytest.mark.parametrize("mm", ["A+21839.,0;", "A+ab.,0;", "T-a?,0;", "A+a.;C+m.,1;"])
def test_m6a_spec_spellings_are_recognised(mm):
    from fiberhmm.cli.common import _mm_has_m6a

    assert _mm_has_m6a(mm)


@pytest.mark.parametrize("mm", ["C+m.,0;", "C+h.,0;", "", "A+m.,0;"])
def test_non_m6a_specs_are_not_m6a(mm):
    from fiberhmm.cli.common import _mm_has_m6a

    assert not _mm_has_m6a(mm)


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
    # Only the first reads are read: --force-seq runs anyway, and an explicit
    # legacy --mode decides the observation mode itself.
    resolve_platform_argument(_args(seq="pacbio", force_seq=True), bam, tool="t")
    assert "running anyway because of --force-seq" in capsys.readouterr().err
    args = _args(seq="pacbio")
    args.mode = "daf"
    resolve_platform_argument(args, bam, tool="t")


def test_unmapped_records_do_not_trigger_the_m6a_refusal(tmp_path):
    """In an aligned BAM, untagged unmapped records are passed through; only
    the mapped reads count for the m6A check."""
    source = _bam(tmp_path, "pb", mm_style="pacbio", n_reads=4)
    mixed = str(tmp_path / "mixed.bam")
    with pysam.AlignmentFile(source, "rb") as src, \
            pysam.AlignmentFile(mixed, "wb", header=src.header) as out:
        reads = list(src.fetch(until_eof=True))
        for read in reads:
            out.write(read)
        for index in range(300):
            unmapped = pysam.AlignedSegment(out.header)
            unmapped.query_name = f"u{index}"
            unmapped.query_sequence = "ACGT" * 50
            unmapped.flag = 4
            out.write(unmapped)
    # Unmapped records sort last; put them first to fill the sample window.
    reordered = str(tmp_path / "unmapped_first.bam")
    with pysam.AlignmentFile(mixed, "rb", check_sq=False) as src, \
            pysam.AlignmentFile(reordered, "wb", header=src.header) as out:
        records = list(src.fetch(until_eof=True))
        for read in sorted(records, key=lambda r: not r.is_unmapped):
            out.write(read)
    args = _args(seq="pacbio")
    resolve_platform_argument(args, reordered, tool="t", )
    assert args.seq == "pacbio"


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
