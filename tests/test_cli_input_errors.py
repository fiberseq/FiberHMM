"""Ordinary user errors end in one ``error:`` line and exit 2, not a traceback.

Release audit 2026-09-29 (CLI P6):
- M13: a missing or garbage input BAM crashed ~15 console scripts with a
  Python traceback, and ``-m`` with a JSON that is not a model gave
  ``KeyError: 'n_states'`` (``fiberhmm-transfer --models``: ``ValueError``).
- M14: ``fiberhmm-consensus --region`` errors (unknown contig, reversed span,
  span over the analysis limit, malformed) were tracebacks; the size error
  did not name the limit or how to raise it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pysam
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _entry(function, *args):
    env = dict(os.environ, FIBERHMM_NO_UPDATE_CHECK="1",
               PYTHONPATH=str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env.pop("FIBERHMM_DEBUG", None)
    code = ("import sys; from fiberhmm.cli._entry import {0} as m; "
            "sys.argv = ['{0}'] + sys.argv[1:]; sys.exit(m())").format(function)
    return subprocess.run([sys.executable, "-c", code, *map(str, args)],
                          capture_output=True, text=True, env=env,
                          cwd=REPO_ROOT, timeout=180)


@pytest.fixture(scope="module")
def bad_inputs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bad_inputs")
    garbage = tmp / "garbage.bam"
    garbage.write_text("this is not a BAM file\n")
    notmodel = tmp / "notmodel.json"
    notmodel.write_text(json.dumps({"foo": 1}))
    return {"missing": tmp / "nope.bam", "garbage": garbage,
            "notmodel": notmodel, "tmp": tmp}


@pytest.mark.parametrize("kind", ["missing", "garbage"])
@pytest.mark.parametrize("function,args", [
    ("call_main", ["--enzyme", "hia5", "--seq", "pacbio", "-o", "{tmp}/c.bam"]),
    ("apply_main", ["--enzyme", "hia5", "--seq", "pacbio", "-o", "{tmp}/apply"]),
    ("recall_tfs_main", ["--enzyme", "hia5", "--seq", "pacbio", "-o", "{tmp}/r.bam"]),
    ("daf_encode_main", ["-o", "{tmp}/e.bam"]),
    ("dedup_main", ["-o", "{tmp}/d.bam"]),
    ("merge_main", ["-o", "{tmp}/m.bam"]),
])
def test_bad_input_bam_is_a_one_line_error(bad_inputs, kind, function, args):
    args = [a.format(tmp=bad_inputs["tmp"]) for a in args]
    proc = _entry(function, "-i", bad_inputs[kind], *args)
    assert proc.returncode in (1, 2), proc.stderr
    assert "Traceback" not in proc.stderr
    assert "rror" in proc.stderr
    if kind == "garbage":
        assert proc.returncode == 2


def test_notmodel_json_is_reported_by_name(bad_inputs, tmp_path, capsys):
    from fiberhmm.cli.common import require_model_files

    with pytest.raises(SystemExit) as exc:
        require_model_files("fiberhmm-call", ("-m/--model", str(bad_inputs["notmodel"])))
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "is not a FiberHMM model (missing n_states" in err
    with pytest.raises(SystemExit):
        require_model_files("t", ("--recall-model", str(bad_inputs["garbage"]) + ".json"))
    assert "does not exist" in capsys.readouterr().err
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json")
    with pytest.raises(SystemExit):
        require_model_files("t", ("-m", str(bad_json)))
    assert "is not valid JSON" in capsys.readouterr().err
    # A real bundled model and absent paths pass.
    from fiberhmm.models import get_model_path
    require_model_files("t", ("-m", get_model_path("hia5", "apply", seq="pacbio")),
                        ("--recall-model", None))


def test_call_with_notmodel_json_exits_2(bad_inputs):
    from conftest import make_synthetic_bam

    bam = make_synthetic_bam(str(bad_inputs["tmp"] / "in.bam"), n_reads=2,
                             read_length=600, n_chroms=1, chrom_length=20_000)
    proc = _entry("call_main", "-i", bam, "-o", bad_inputs["tmp"] / "nm.bam",
                  "-m", bad_inputs["notmodel"], "-c", "1", "--no-qc")
    assert proc.returncode == 2, proc.stderr
    assert "Traceback" not in proc.stderr
    assert "is not a FiberHMM model" in proc.stderr


def test_run_reporting_input_errors_only_catches_input_errors(monkeypatch, capsys):
    from fiberhmm.cli.common import is_input_error, run_reporting_input_errors

    def missing():
        raise FileNotFoundError(2, "No such file or directory", "x.bam")

    with pytest.raises(SystemExit) as exc:
        run_reporting_input_errors("fiberhmm-x", missing)
    assert exc.value.code == 2
    assert capsys.readouterr().err.startswith("error: fiberhmm-x: No such file")

    def bug():
        raise ValueError("some internal invariant failed")

    with pytest.raises(ValueError, match="internal invariant"):
        run_reporting_input_errors("fiberhmm-x", bug)

    monkeypatch.setenv("FIBERHMM_DEBUG", "1")
    with pytest.raises(FileNotFoundError):
        run_reporting_input_errors("fiberhmm-x", missing)

    assert is_input_error(ValueError("file does not contain alignment data"))
    assert is_input_error(OSError("no BGZF EOF marker; file may be truncated"))
    assert not is_input_error(BrokenPipeError(32, "Broken pipe"))


# ---------------------------------------------------------------------------
# fiberhmm-consensus --region / fiberhmm-transfer --models
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def declared_bam(tmp_path_factory):
    from conftest import make_synthetic_bam

    from fiberhmm.io.bam_header import append_chemistry

    tmp = tmp_path_factory.mktemp("consensus_inputs")
    source = make_synthetic_bam(str(tmp / "src.bam"), n_reads=4, read_length=600,
                                n_chroms=1, chrom_length=100_000)
    path = str(tmp / "declared.bam")
    with pysam.AlignmentFile(source, "rb") as src:
        header = append_chemistry(src.header, {
            "assay": "fiber-seq", "enzyme": "hia5", "platform": "pacbio",
            "mode": "pacbio-fiber"})
        with pysam.AlignmentFile(path, "wb", header=header) as out:
            for read in src.fetch(until_eof=True):
                out.write(read)
    pysam.index(path)
    return path


@pytest.mark.parametrize("region,expected", [
    ("chrNope:1-100", "chromosome 'chrNope' is absent"),
    ("chr1:10250-9900", "END greater than START"),
    ("chr1:abc", "expected CHROM:START-END"),
    ("chr1:200000-200100", "beyond the end of 'chr1'"),
    ("chr1:1000-70000", "more than the 50,000 bp analysis limit (compute.maximum_region_bp)"),
])
def test_consensus_region_errors_are_one_line(declared_bam, tmp_path, capsys,
                                              region, expected):
    from fiberhmm.inference.consensus.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--bam", declared_bam, "--region", region, "--cores", "1",
              "--output", str(tmp_path / "out")])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert expected in err
    if "maximum_region_bp" in expected:
        assert "--parameters" in err


def test_consensus_region_check_accepts_a_string_dataset_path(declared_bam, tmp_path, capsys):
    """--datasets entries may give "paths" as one string (the loader accepts it)."""
    from fiberhmm.inference.consensus.cli import main

    datasets = tmp_path / "datasets.json"
    datasets.write_text(json.dumps([{"dataset_id": "d", "paths": declared_bam,
                                     "chemistry": "hia5-pacbio"}]))
    with pytest.raises(SystemExit) as exc:
        main(["--datasets", str(datasets), "--region", "chrNope:1-100",
              "--output", str(tmp_path / "out")])
    assert exc.value.code == 2
    assert f"{declared_bam}: chromosome 'chrNope' is absent" in capsys.readouterr().err


def test_consensus_garbage_bam_is_one_line(bad_inputs, tmp_path, capsys):
    from fiberhmm.inference.consensus.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--bam", str(bad_inputs["garbage"]), "--region", "chr1:1-100",
              "--output", str(tmp_path / "out")])
    assert exc.value.code == 2
    assert "file does not contain alignment data" in capsys.readouterr().err


def test_transfer_with_a_json_that_is_not_a_catalog(bad_inputs, declared_bam,
                                                    tmp_path, capsys):
    from fiberhmm.inference.consensus.transfer_cli import main

    for models, expected in ((bad_inputs["notmodel"], "is not a frozen class catalog"),
                             (bad_inputs["garbage"], "is not a JSON frozen class catalog")):
        with pytest.raises(SystemExit) as exc:
            main(["--models", str(models), "--bam", declared_bam,
                  "--output", str(tmp_path / f"out_{Path(models).stem}")])
        assert exc.value.code == 2
        assert expected in capsys.readouterr().err


def test_merge_of_a_header_only_bam_writes_an_empty_bam(tmp_path):
    """M17: fiberhmm-merge on an empty BAM raised ``TypeError: '<' not
    supported between 'NoneType' and 'int'``."""
    from conftest import make_synthetic_bam

    empty = make_synthetic_bam(str(tmp_path / "empty.bam"), n_reads=0)
    output = tmp_path / "merged.bam"
    proc = _entry("merge_main", "-i", empty, "-o", output)
    assert proc.returncode == 0, proc.stderr
    with pysam.AlignmentFile(str(output), check_sq=False) as bam:
        assert sum(1 for _ in bam.fetch(until_eof=True)) == 0


# ---------------------------------------------------------------------------
# L5: out-of-range thresholds; L7: -m and --enzyme of different assays
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("function,flag,value", [
    ("call_main", "--prob-threshold", "300"),
    ("call_main", "--min-mapq", "-5"),
    ("call_main", "--min-read-length", "-1"),
    ("apply_main", "--prob-threshold", "256"),
    ("apply_main", "--min-mapq", "-1"),
])
def test_out_of_range_filters_are_usage_errors(bad_inputs, function, flag, value):
    """--prob-threshold 300 turned every modification call off and the run
    still reported footprints; --min-mapq -5 was accepted."""
    proc = _entry(function, "-i", bad_inputs["garbage"], "-o",
                  bad_inputs["tmp"] / "x", "--enzyme", "hia5", flag, value)
    assert proc.returncode == 2
    assert f"argument {flag}" in proc.stderr
    assert "Traceback" not in proc.stderr


@pytest.mark.parametrize("function", ["call_main", "apply_main"])
def test_daf_model_with_an_m6a_enzyme_names_the_conflict(bad_inputs, function):
    from conftest import make_synthetic_bam

    from fiberhmm.models import get_model_path

    bam = make_synthetic_bam(str(bad_inputs["tmp"] / f"l7_{function}.bam"),
                             n_reads=2, read_length=600, n_chroms=1,
                             chrom_length=20_000)
    proc = _entry(function, "-i", bam, "-o", bad_inputs["tmp"] / f"l7_{function}",
                  "--enzyme", "hia5", "--seq", "nanopore", "-c", "1",
                  "-m", get_model_path("dddb", "apply", seq="nanopore"))
    assert proc.returncode == 2, proc.stderr
    assert "is a DAF-seq (deamination) model" in proc.stderr
    assert "--enzyme hia5 is Fiber-seq (m6A)" in proc.stderr


# ---------------------------------------------------------------------------
# L6: region options on streaming call; L8: dedup with nothing to dedup;
# L9: .sam output names
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pacbio_style_bam(tmp_path_factory):
    from conftest import make_synthetic_bam

    tmp = tmp_path_factory.mktemp("l6_l9")
    source = make_synthetic_bam(str(tmp / "src.bam"), n_reads=3, read_length=600,
                                n_chroms=1, chrom_length=20_000, seed=41)
    path = str(tmp / "pacbio.bam")
    with pysam.AlignmentFile(source, "rb") as src, \
            pysam.AlignmentFile(path, "wb", header=src.header) as out:
        for read in src.fetch(until_eof=True):
            read.set_tag("MM", read.get_tag("MM") + "T-a;")
            out.write(read)
    pysam.index(path)
    return path


@pytest.mark.parametrize("flags", [["--chroms", "chrNope"], ["--skip-scaffolds"]])
def test_streaming_call_refuses_region_selection_options(pacbio_style_bam, tmp_path, flags):
    """--chroms chrNope without --region-parallel used to call every read."""
    output = tmp_path / "out.bam"
    proc = _entry("call_main", "-i", pacbio_style_bam, "-o", output, "--enzyme",
                  "hia5", "--seq", "pacbio", "--no-qc", "-c", "1", *flags)
    assert proc.returncode == 2, proc.stderr
    assert "--region-parallel" in proc.stderr
    assert not output.exists()


@pytest.mark.parametrize("function", ["call_main", "recall_tfs_main"])
def test_sam_or_cram_output_name_is_refused(pacbio_style_bam, tmp_path, function):
    """-o x.sam wrote BGZF BAM bytes under a .sam name (plus x.sam.bai)."""
    output = tmp_path / "out.sam"
    proc = _entry(function, "-i", pacbio_style_bam, "-o", output, "--enzyme",
                  "hia5", "--seq", "pacbio", "-c", "1")
    assert proc.returncode == 2, proc.stderr
    assert "Name it .bam" in proc.stderr
    assert not output.exists()


def test_dedup_without_fingerprintable_reads_exits_nonzero(pacbio_style_bam, tmp_path):
    """dedup on a Fiber-seq BAM printed 'Nothing to do', wrote nothing and
    exited 0, so an older output looked current."""
    output = tmp_path / "dedup.bam"
    proc = _entry("dedup_main", "-i", pacbio_style_bam, "-o", output)
    assert proc.returncode == 1
    assert "no output written" in proc.stderr
    assert not output.exists()
