"""Regressions for command-line bugs found while verifying the 3.0 docs.

Each test runs the documented command (or its entry point) the way a user
would: banners and @PG records, output directories, provenance of
``fiberhmm-apply`` output, chemistry errors in consensus/transfer, the
posteriors ML threshold and ``--version`` on every console script.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pysam
import pytest

from conftest import make_synthetic_bam

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(module: str, *args, timeout=240):
    env = dict(os.environ)
    env.setdefault("FIBERHMM_NO_UPDATE_CHECK", "1")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", module, *map(str, args)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout, env=env,
    )


def _programs(path, program):
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
        return [
            record for record in bam.header.to_dict().get("PG", [])
            if str(record.get("PN", "")) == program
        ]


@pytest.fixture(scope="module")
def pacbio_bam(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("docbugs")
    source = make_synthetic_bam(str(tmp / "src.bam"), n_reads=4,
                                read_length=1500, n_chroms=1,
                                chrom_length=20_000, seed=5)
    # PacBio-style MM (a T-a spec next to A+a): an explicit --seq pacbio is
    # checked against the reads' MM specs, and A+a-only reads are Nanopore.
    path = tmp / "pacbio.bam"
    with pysam.AlignmentFile(source, "rb") as src, \
            pysam.AlignmentFile(str(path), "wb", header=src.header) as out:
        for read in src.fetch(until_eof=True):
            read.set_tag("MM", read.get_tag("MM") + "T-a;")
            out.write(read)
    pysam.index(str(path))
    return path


# --------------------------------------------------------------------------
# Banners and @PG: no stale [BETA] label, no retired ddda_mcg token
# --------------------------------------------------------------------------

def test_call_and_recall_banners_and_pg_are_current(tmp_path, pacbio_bam):
    calls = tmp_path / "calls.bam"
    result = _run_cli(
        "fiberhmm.cli.call", "-i", pacbio_bam, "-o", calls, "--enzyme", "hia5",
        "--seq", "pacbio", "--no-qc", "--no-recall-nucs", "-c", "1",
        "--io-threads", "1", "--min-read-length", "0",
    )
    assert result.returncode == 0, result.stderr
    assert "[BETA]" not in result.stderr
    assert "ddda_mcg=" not in result.stderr
    (program,) = _programs(calls, "fiberhmm-call")
    assert "coord=molecular" in program["DS"]
    assert "ddda_mcg=" not in program["DS"]

    recalled = tmp_path / "recalled.bam"
    result = _run_cli(
        "fiberhmm.cli.recall_tfs", "-i", calls, "-o", recalled, "--enzyme", "hia5",
        "--seq", "pacbio", "-c", "1",
    )
    assert result.returncode == 0, result.stderr
    assert "BETA" not in result.stderr and "(beta)" not in result.stderr


# --------------------------------------------------------------------------
# fiberhmm-posteriors resolves the ML threshold like fiberhmm-call
# --------------------------------------------------------------------------

def _posteriors_run(monkeypatch, tmp_path, *argv):
    from fiberhmm.cli import export_posteriors

    captured = {}
    monkeypatch.setattr(export_posteriors, "export_posteriors",
                        lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(sys, "argv", [
        "fiberhmm-posteriors", "-o", str(tmp_path / "p.tsv.gz"), *map(str, argv)])
    export_posteriors.main()
    return captured


@pytest.mark.parametrize("argv, expected", [
    (("--enzyme", "hia5", "--seq", "nanopore"), 248),
    (("--enzyme", "hia5", "--seq", "pacbio"), 128),
    (("--enzyme", "hia5", "--seq", "nanopore", "--prob-threshold", "100"), 100),
    (("--enzyme", "dddb"), 128),
])
def test_posteriors_threshold_follows_the_chemistry(monkeypatch, tmp_path,
                                                    pacbio_bam, argv, expected):
    run = _posteriors_run(monkeypatch, tmp_path, "-i", pacbio_bam, *argv)
    assert run["prob_threshold"] == expected


def test_posteriors_detects_nanopore_and_its_threshold(monkeypatch, tmp_path,
                                                        pacbio_bam, capsys):
    # The synthetic reads carry A+a only (Nanopore-style MM), so a missing
    # --seq resolves to nanopore exactly as fiberhmm-call does.
    run = _posteriors_run(monkeypatch, tmp_path, "-i", pacbio_bam,
                          "--enzyme", "hia5")
    assert run["prob_threshold"] == 248
    assert Path(run["model_path"]).name == "hia5_nanopore.json"
    assert "using --seq nanopore" in capsys.readouterr().err


def test_posteriors_custom_model_takes_the_declared_threshold(monkeypatch, tmp_path):
    from fiberhmm.io.bam_header import append_chemistry
    from fiberhmm.models import get_model_path

    bam = tmp_path / "declared.bam"
    header = append_chemistry(
        pysam.AlignmentHeader.from_dict(
            {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 1000}]}),
        {"assay": "fiber-seq", "enzyme": "hia5", "platform": "nanopore",
         "mode": "nanopore-fiber"})
    with pysam.AlignmentFile(str(bam), "wb", header=header):
        pass
    model = get_model_path("hia5", tool="apply", seq="nanopore")
    run = _posteriors_run(monkeypatch, tmp_path, "-i", bam, "-m", model)
    assert run["prob_threshold"] == 248


# --------------------------------------------------------------------------
# consensus / transfer: missing or unsupported chemistry is a one-line error
# --------------------------------------------------------------------------

def _declared_bam(path, enzyme, platform="pacbio", mode="pacbio-fiber"):
    from fiberhmm.io.bam_header import append_chemistry

    header = append_chemistry(
        pysam.AlignmentHeader.from_dict(
            {"HD": {"VN": "1.6", "SO": "coordinate"},
             "SQ": [{"SN": "chr1", "LN": 5000}]}),
        {"assay": "fiber-seq", "enzyme": enzyme, "platform": platform,
         "mode": mode})
    with pysam.AlignmentFile(str(path), "wb", header=header):
        pass
    return path


_CHEMISTRY_CASES = [
    ("custom", (), "Every BAM needs chemistry metadata"),
    ("ecogii", (), "consensus has no ecogii profile"),
    ("hia5", ("--chemistry", "ddda"), "conflicts with BAM chemistry hia5-pacbio"),
]


def _assert_one_line_error(capsys, excinfo, prog, message):
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err and "usage:" not in err
    lines = [line for line in err.splitlines() if line.strip()]
    assert len(lines) == 1 and lines[0].startswith(f"{prog}: error: ")
    assert message in lines[0]


@pytest.mark.parametrize("enzyme, extra, message", _CHEMISTRY_CASES)
def test_consensus_chemistry_errors_are_one_line(tmp_path, capsys, enzyme,
                                                 extra, message):
    from fiberhmm.inference.consensus.cli import main

    bam = _declared_bam(tmp_path / f"{enzyme}.bam", enzyme)
    out = tmp_path / "classes"
    with pytest.raises(SystemExit) as excinfo:
        main(["--bam", str(bam), "--region", "chr1:100-400", *extra,
              "--output", str(out)])
    _assert_one_line_error(capsys, excinfo, "fiberhmm-consensus", message)
    assert not out.exists()


@pytest.fixture(scope="module")
def frozen_catalog(tmp_path_factory):
    from fiberhmm.inference.consensus.transfer import export_run
    from fiberhmm.inference.consensus.workflow import run_workflow
    from test_consensus_lattice_recaller import planted_payload

    tmp = tmp_path_factory.mktemp("frozen")
    run_workflow(planted_payload(n=200), {"cr": {"engine": "lattice_recaller"},
                                          "compute": {"cores": 1}}, tmp / "run")
    export_run(tmp / "run", tmp / "frozen_classes.json.gz")
    return tmp / "frozen_classes.json.gz"


@pytest.mark.parametrize("enzyme, extra, message", _CHEMISTRY_CASES)
def test_transfer_chemistry_errors_are_one_line(tmp_path, capsys, frozen_catalog,
                                                enzyme, extra, message):
    from fiberhmm.inference.consensus.transfer_cli import main

    bam = _declared_bam(tmp_path / f"{enzyme}.bam", enzyme)
    bed = tmp_path / "targets.bed"
    bed.write_text("chr1\t0\t300\tsite\t0\t+\n")
    with pytest.raises(SystemExit) as excinfo:
        main(["--models", str(frozen_catalog), "--bam", str(bam), "--bed", str(bed),
              *extra, "--output", str(tmp_path / "out")])
    _assert_one_line_error(capsys, excinfo, "fiberhmm-transfer", message)


def test_load_bam_payload_raises_a_chemistry_error(tmp_path):
    """FiberBrowser calls load_bam_payload directly; the error stays a ValueError."""
    from fiberhmm.inference.consensus.bam import load_bam_payload
    from fiberhmm.io.bam_header import ChemistryResolutionError

    bam = _declared_bam(tmp_path / "custom.bam", "custom")
    with pytest.raises(ChemistryResolutionError) as excinfo:
        load_bam_payload([dict(dataset_id="d1", paths=[str(bam)])],
                         dict(chrom="chr1", start=0, end=300))
    assert isinstance(excinfo.value, ValueError)


# --------------------------------------------------------------------------
# Missing output directories are created by every writer
# --------------------------------------------------------------------------

def test_atomic_output_creates_missing_directories(tmp_path):
    from fiberhmm.inference.bam_output import atomic_output

    final = tmp_path / "a" / "b" / "out.txt"
    with atomic_output(str(final)) as temporary:
        Path(temporary).write_text("done")
    assert final.read_text() == "done"


@pytest.mark.parametrize("parallel", [("--region-parallel", "-c", "2"), ("-c", "1")])
def test_call_creates_a_missing_output_directory(tmp_path, pacbio_bam, parallel):
    out = tmp_path / "new" / "nested" / "calls.bam"
    result = _run_cli(
        "fiberhmm.cli.call", "-i", pacbio_bam, "-o", out, "--enzyme", "hia5",
        "--seq", "pacbio", "--no-qc", "--no-recall-nucs", "--io-threads", "1",
        "--min-read-length", "0", *parallel,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists() and Path(str(out) + ".bai").exists()
    assert sorted(p.name for p in out.parent.iterdir()) == ["calls.bam", "calls.bam.bai"]


def test_recall_creates_a_missing_output_directory(tmp_path, pacbio_bam):
    calls = tmp_path / "calls.bam"
    assert _run_cli(
        "fiberhmm.cli.call", "-i", pacbio_bam, "-o", calls, "--enzyme", "hia5",
        "--seq", "pacbio", "--no-qc", "--no-recall-nucs", "-c", "1",
        "--io-threads", "1", "--min-read-length", "0").returncode == 0
    out = tmp_path / "new" / "recalled.bam"
    result = _run_cli("fiberhmm.cli.recall_tfs", "-i", calls, "-o", out,
                      "--enzyme", "hia5", "--seq", "pacbio", "-c", "1")
    assert result.returncode == 0, result.stderr
    assert out.exists()


def test_dedup_creates_missing_output_directories(tmp_path):
    from conftest import make_synthetic_iupac_bam

    bam = tmp_path / "daf.bam"
    make_synthetic_iupac_bam(str(bam), n_reads=6, read_length=1500)
    out = tmp_path / "new" / "dedup.bam"
    stats = tmp_path / "other" / "clusters.tsv"
    result = _run_cli("fiberhmm.cli.dedup", "-i", bam, "-o", out,
                      "--flag-only", "--stats-tsv", stats)
    assert result.returncode == 0, result.stderr
    assert out.exists() and stats.exists()


# --------------------------------------------------------------------------
# fiberhmm-apply writes the same provenance as fiberhmm-call
# --------------------------------------------------------------------------

def test_apply_output_declares_its_chemistry_and_pg(tmp_path, pacbio_bam):
    from fiberhmm.io.bam_header import declared_chemistries, header_has_coord_marker
    from fiberhmm.models import declared_prob_threshold_chemistry, get_model_path

    outdir = tmp_path / "apply"
    result = _run_cli("fiberhmm.cli.apply", "-i", pacbio_bam, "--enzyme", "hia5",
                      "--seq", "nanopore", "-o", outdir, "-c", "1",
                      "--min-read-length", "0")
    assert result.returncode == 0, result.stderr
    applied = outdir / "pacbio_footprints.bam"
    (program,) = _programs(applied, "fiberhmm-apply")
    assert program["VN"] and "fiberhmm" in program["CL"]
    assert "coord=molecular" in program["DS"]
    assert "enzyme=hia5" in program["DS"] and "prob_threshold=248" in program["DS"]
    with pysam.AlignmentFile(str(applied), "rb", check_sq=False) as bam:
        header = bam.header
    (chemistry,) = declared_chemistries(header)
    assert (chemistry["enzyme"], chemistry["platform"], chemistry["mode"]) == (
        "hia5", "nanopore", "nanopore-fiber")
    assert declared_prob_threshold_chemistry(header) == ("hia5", "nanopore")
    assert header_has_coord_marker(header)

    # A refit table on apply output now inherits the declared chemistry
    # (and its 248 threshold), exactly as on fiberhmm-call output.
    table = get_model_path("hia5", tool="recall", seq="nanopore")
    recalled = tmp_path / "recalled.bam"
    result = _run_cli("fiberhmm.cli.recall_tfs", "-i", applied, "-o", recalled,
                      "-m", table, "-c", "1")
    assert result.returncode == 0, result.stderr
    assert "using the defaults of --enzyme hia5 --seq nanopore" in result.stderr
    assert "ML threshold for MM/ML calls: 248" in result.stderr


def test_apply_on_stdin_with_a_custom_model_refuses_late_inheritance(tmp_path, pacbio_bam):
    """With a declaration now written, a custom -m on declared stdin input is
    refused like fiberhmm-call (its defaults were chosen before the header was
    readable), with a fix apply can actually follow."""
    from fiberhmm.models import get_model_path

    outdir = tmp_path / "apply"
    assert _run_cli("fiberhmm.cli.apply", "-i", pacbio_bam, "--enzyme", "hia5",
                    "--seq", "pacbio", "-o", outdir, "-c", "1",
                    "--min-read-length", "0").returncode == 0
    env = dict(os.environ, FIBERHMM_NO_UPDATE_CHECK="1")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "fiberhmm.cli.apply", "-i", "-", "-o", "-",
         "-m", get_model_path("hia5", tool="apply", seq="pacbio")],
        input=(outdir / "pacbio_footprints.bam").read_bytes(),
        capture_output=True, cwd=REPO_ROOT, env=env, timeout=240)
    err = result.stderr.decode()
    assert result.returncode == 2, err
    assert "Pass --enzyme hia5" in err and "Traceback" not in err
    assert "--replace-chemistry to declare" not in err


def test_posteriors_creates_a_missing_output_directory(tmp_path, pacbio_bam):
    out = tmp_path / "new" / "posteriors.tsv.gz"
    result = _run_cli("fiberhmm.cli.export_posteriors", "-i", pacbio_bam,
                      "--enzyme", "hia5", "--seq", "pacbio", "-o", out, "-c", "1")
    assert result.returncode == 0, result.stderr
    assert out.exists()
