"""The Snakemake template (workflows/snakemake): config and sample sheet are
readable, and ``snakemake -n`` plans one fiberhmm-pipeline job per sample.

The dry run needs Snakemake (>= 8): ``snakemake`` on PATH, or the program
named by ``FIBERHMM_TEST_SNAKEMAKE``; it is skipped otherwise.
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOW = Path(__file__).resolve().parents[1] / "workflows" / "snakemake"
SNAKEMAKE = os.environ.get("FIBERHMM_TEST_SNAKEMAKE") or shutil.which("snakemake")


def test_template_files_are_consistent():
    config = yaml.safe_load((WORKFLOW / "config.yaml").read_text())
    for key in ("samples", "reference", "outdir", "threads", "dorado", "resources"):
        assert key in config
    assert set(config["resources"]) == {"default", "basecall"}
    lines = [line for line in (WORKFLOW / "samples.tsv").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    rows = list(csv.DictReader(lines, delimiter="\t"))
    assert rows and {"sample", "reads", "enzyme"} <= set(rows[0])
    assert all(row["enzyme"] in ("hia5", "ddda", "dddb") for row in rows)
    for profile in ("local", "slurm"):
        assert yaml.safe_load((WORKFLOW / "profiles" / profile / "config.yaml").read_text())
    text = (WORKFLOW / "Snakefile").read_text()
    assert "fiberhmm-pipeline" in text and "rule fiberhmm_pipeline" in text


@pytest.mark.skipif(not SNAKEMAKE, reason="snakemake not installed")
def test_dry_run_plans_one_pipeline_per_sample(tmp_path):
    (tmp_path / "run1" / "pod5").mkdir(parents=True)
    (tmp_path / "run1" / "pod5" / "a.pod5").write_bytes(b"x")
    for name in ("calls.bam", "reads.fastq.gz", "ref.fa", "map.dna"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "samples.tsv").write_text(
        "sample\treads\tenzyme\treference\tseq\textra\n"
        "ont\trun1/pod5\thia5\t\t\t\n"
        "ubam\tcalls.bam\thia5\t\tnanopore\t--tracks\n"
        "daf\treads.fastq.gz,calls.bam\tdddb\tmap.dna\t\t\n")
    config = yaml.safe_load((WORKFLOW / "config.yaml").read_text())
    config["reference"] = "ref.fa"
    config["dorado"]["device"] = "metal"
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    result = subprocess.run(
        [SNAKEMAKE, "-n", "-p", "-s", str(WORKFLOW / "Snakefile"), "-d", str(tmp_path),
         "--cores", "4"], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    out = result.stdout + result.stderr
    commands = [line for line in out.splitlines() if line.lstrip().startswith(
        ("fiberhmm-pipeline", "Shell command: fiberhmm-pipeline"))]
    assert len(commands) == 3
    ont = next(c for c in commands if "--sample ont" in c)
    assert "run1/pod5" in ont and "--dorado-device metal" in ont
    ubam = next(c for c in commands if "--sample ubam" in c)
    assert "--seq nanopore" in ubam and "--tracks" in ubam and "--dorado" not in ubam
    daf = next(c for c in commands if "--sample daf" in c)
    assert "reads.fastq.gz calls.bam" in daf and "--reference map.dna" in daf
    assert "gpu=1" in out and "--gres=gpu:1" in out
