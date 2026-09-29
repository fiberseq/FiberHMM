"""Custom-model chemistry inheritance drives enzyme-dependent defaults (3.0 verify HIGH).

Regressions for the Codex verification pass on b006409.
"""
from __future__ import annotations

import array
import os
import subprocess
import sys
from pathlib import Path

import pysam
import pytest
from conftest import make_synthetic_iupac_bam

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(module, *args, stdin=None, timeout=300):
    env = dict(os.environ)
    env.setdefault("FIBERHMM_NO_UPDATE_CHECK", "1")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", module, *map(str, args)],
        cwd=REPO_ROOT, capture_output=True, timeout=timeout, env=env,
        stdin=stdin,
    )


def _bundled(name):
    from fiberhmm.models import _bundled_model_path

    return _bundled_model_path(name)


def _pg_ds(path, program):
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
        records = [
            record for record in bam.header.to_dict().get("PG", [])
            if str(record.get("PN", "")).startswith(program)
        ]
    assert records, f"no {program} @PG record"
    return records[-1].get("DS", "")


def _ds_tokens(ds):
    return dict(
        token.split("=", 1) for token in ds.split() if "=" in token
    )


def _ma_tags(path):
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
        return [
            (read.query_name, read.get_tag("MA") if read.has_tag("MA") else None)
            for read in bam.fetch(until_eof=True)
        ]


def _declare(source, target, declaration):
    from fiberhmm.io.bam_header import append_chemistry

    with pysam.AlignmentFile(str(source), "rb", check_sq=False) as bam:
        header = append_chemistry(bam.header, declaration)
        with pysam.AlignmentFile(str(target), "wb", header=header) as out:
            for read in bam.fetch(until_eof=True):
                out.write(read)
    pysam.index(str(target))
    return target


# ---------------------------------------------------------------------------
# 1. custom-model chemistry inheritance drives enzyme-dependent defaults
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ddda_declared_bam(tmp_path_factory):
    root = tmp_path_factory.mktemp("ddda_declared")
    raw = make_synthetic_iupac_bam(
        str(root / "raw.bam"), n_reads=8, read_length=2000, n_chroms=1,
        chrom_length=30_000, seed=42,
    )
    return _declare(raw, root / "declared.bam", {
        "assay": "daf", "enzyme": "ddda", "platform": "pacbio",
        "mode": "daf", "model": "ddda_TF",
    })


def test_call_custom_model_inherits_ddda_defaults(tmp_path, ddda_declared_bam):
    base = ["-m", _bundled("ddda_TF.json"), "--no-dedup",
            "--no-daf-call-snps", "--no-qc", "--phase-nrl", "off",
            "--min-read-length", "0", "-c", "1", "--io-threads", "1"]
    inherited = tmp_path / "inherited.bam"
    explicit = tmp_path / "explicit.bam"
    result = _run_cli("fiberhmm.cli.call", "-i", ddda_declared_bam,
                      "-o", inherited, *base)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    result = _run_cli("fiberhmm.cli.call", "-i", ddda_declared_bam,
                      "-o", explicit, *base, "--enzyme", "ddda")
    assert result.returncode == 0, result.stderr.decode(errors="replace")

    ds = _ds_tokens(_pg_ds(inherited, "fiberhmm-call"))
    assert ds["enzyme"] == "ddda"
    assert ds["cpg_mask"] == "unmethylated-only"
    assert ds["daf_run_mask"] == ">=2/keep-one"
    assert ds["nuc_profile"] != "off"            # DddA radial nucleosome policy
    # Same table, same chemistry: the run is the explicit --enzyme ddda run.
    assert ds == _ds_tokens(_pg_ds(explicit, "fiberhmm-call"))
    assert _ma_tags(inherited) == _ma_tags(explicit)


def test_call_stdin_custom_model_refuses_uninheritable_defaults(
        tmp_path, ddda_declared_bam):
    with open(ddda_declared_bam, "rb") as handle:
        result = _run_cli(
            "fiberhmm.cli.call", "-i", "-", "-o", tmp_path / "out.bam",
            "-m", _bundled("ddda_TF.json"), "--no-qc", "--phase-nrl", "off",
            "--min-read-length", "0", "-c", "1", "--io-threads", "1",
            stdin=handle,
        )
    assert result.returncode == 2
    assert b"--enzyme ddda" in result.stderr
    assert not (tmp_path / "out.bam").exists()


def _cpg_only_ddda_input(path):
    from fiberhmm.io.bam_header import append_chemistry

    header = append_chemistry(
        pysam.AlignmentHeader.from_dict({
            "HD": {"SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 1000}]}),
        {"assay": "daf", "enzyme": "ddda", "platform": "pacbio", "mode": "daf"},
    )
    read = pysam.AlignedSegment(header)
    read.query_name = "cpg_only"
    sequence = list("ATCGA" * 80)
    sequence[22] = sequence[372] = "Y"
    read.query_sequence = "".join(sequence)
    read.flag = 0
    read.reference_id = 0
    read.reference_start = 0
    read.cigartuples = [(0, 400)]
    read.set_tag("st", "CT")
    read.set_tag("as", array.array("I", [0]))
    read.set_tag("al", array.array("I", [400]))
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        out.write(read)
    return path


@pytest.mark.parametrize("recall_nucs", [False, True])
def test_recall_custom_model_inherits_ddda_defaults(tmp_path, recall_nucs):
    source = _cpg_only_ddda_input(tmp_path / "in.bam")
    extra = ["--recall-nucs"] if recall_nucs else []
    base = ["-m", _bundled("ddda_TF.json"), "-c", "1", "--io-threads", "1",
            *extra]
    outputs = {}
    for name, flags in (("inherited", []), ("explicit", ["--enzyme", "ddda"])):
        out = tmp_path / f"{name}.bam"
        result = _run_cli("fiberhmm.cli.recall_tfs", "-i", source, "-o", out,
                          *base, *flags)
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        outputs[name] = out
    program = "fiberhmm-recall-nucs" if recall_nucs else "fiberhmm-recall-tfs"
    inherited_ds = _ds_tokens(_pg_ds(outputs["inherited"], program))
    assert inherited_ds["enzyme"] == "ddda"
    assert inherited_ds["daf_run_mask"] == ">=2/keep-one"
    if recall_nucs:
        assert inherited_ds["nuc_profile"] != "off"
    assert inherited_ds == _ds_tokens(_pg_ds(outputs["explicit"], program))
    # CpG-only deaminations are masked for DddA: no spurious TF footprint.
    assert _ma_tags(outputs["inherited"]) == _ma_tags(outputs["explicit"])
    assert "tf." not in (_ma_tags(outputs["inherited"])[0][1] or "")


def test_recall_custom_model_keeps_explicit_table(tmp_path):
    """Inheriting DddA defaults must not swap the refit table for a preset."""
    source = _cpg_only_ddda_input(tmp_path / "in.bam")
    result = _run_cli("fiberhmm.cli.recall_tfs", "-i", source,
                      "-o", tmp_path / "out.bam", "-m",
                      _bundled("dddb_nanopore.json"), "-c", "1",
                      "--io-threads", "1", "--replace-chemistry")
    # --replace-chemistry: the refit is declared as run, no inheritance.
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert b"enzyme=custom" in result.stderr
    result = _run_cli("fiberhmm.cli.recall_tfs", "-i", source,
                      "-o", tmp_path / "out2.bam", "-m",
                      _bundled("dddb_nanopore.json"), "-c", "1",
                      "--io-threads", "1")
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    with pysam.AlignmentFile(str(tmp_path / "out2.bam"), "rb",
                             check_sq=False) as bam:
        chemistry = [c for c in bam.header.to_dict().get("CO", [])
                     if c.startswith("FIBERHMM-CHEMISTRY")]
    assert chemistry[-1].endswith("model=dddb_nanopore")
    assert "enzyme=ddda" in chemistry[-1]


def test_effective_chemistry_helper_adopts_supported_enzymes_only():
    from types import SimpleNamespace

    from fiberhmm.cli.provenance import resolve_effective_chemistry
    from fiberhmm.io.bam_header import append_chemistry

    def header(enzyme, platform, mode):
        return append_chemistry(
            pysam.AlignmentHeader.from_dict({
                "HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 1000}]}),
            {"assay": "fiber-seq", "enzyme": enzyme, "platform": platform,
             "mode": mode},
        )

    args = SimpleNamespace(enzyme=None, seq=None)
    declaration = resolve_effective_chemistry(
        args, "nanopore-fiber", header("hia5", "nanopore", "nanopore-fiber"),
        "refit.json", None, tool="t")
    assert (args.enzyme, args.seq) == ("hia5", "nanopore")
    assert declaration["enzyme"] == "hia5"

    args = SimpleNamespace(enzyme=None, seq=None)
    resolve_effective_chemistry(
        args, "pacbio-fiber", header("ecogii", "pacbio", "pacbio-fiber"),
        "refit.json", None, tool="t")
    assert args.enzyme is None           # development chemistry: no preset

    args = SimpleNamespace(enzyme="hia5", seq="pacbio")
    resolve_effective_chemistry(args, "pacbio-fiber", None, "m.json", None,
                                tool="t")
    assert (args.enzyme, args.seq) == ("hia5", "pacbio")


