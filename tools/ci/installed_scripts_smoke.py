#!/usr/bin/env python3
"""Run installed fiberhmm-call and fiberhmm-pipeline executables on tiny synthetic data.

The wheel CI job runs this with the clean venv's python, from outside the checkout, so the
console-script shims (fiberhmm/cli/_entry.py) of the installed wheel are what runs. `--help`
alone does not catch a broken shim that still prints help (coverage audit item 30).

Usage: installed_scripts_smoke.py --bin <venv>/bin --workdir <scratch dir>

Synthetic data only: random reference, simulated Hia5 m6A (PacBio MM/ML) and simulated DddB
deamination. Needs minimap2 on PATH for the pipeline run.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pysam

COMP = str.maketrans("ACGT", "TGCA")


def revcomp(seq: str) -> str:
    return seq.translate(COMP)[::-1]


def _mm_entry(sequence, positions, base, strand):
    index = {pos: i for i, pos in enumerate(p for p, b in enumerate(sequence) if b == base)}
    skips, previous = [], -1
    for pos in sorted(positions):
        skips.append(index[pos] - previous - 1)
        previous = index[pos]
    return f"{base}{strand}a." + "".join(f",{s}" for s in skips) + ";"


def make_hia5_bam(path: Path, n_reads=12, length=3000, seed=5) -> Path:
    """Hia5 PacBio reads over a phased nucleosome array, both strands (as tests/test_recall_io.py)."""
    rng = np.random.default_rng(seed)
    contig = 20_000
    reference = "".join(rng.choice(list("ACGT"), size=contig))
    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": contig}]})
    reads = []
    for i in range(n_reads):
        start = int(rng.integers(0, contig - length))
        protected = np.zeros(length, dtype=bool)
        pos = int(rng.integers(0, 60))
        while pos < length:
            if rng.random() < 0.9:
                protected[pos:pos + 147] = True
            pos += 192 + int(rng.integers(-10, 11))
        reverse = bool(i % 2)
        rate = np.where(protected, 0.01, 0.7)
        ref_seq = reference[start:start + length]
        original = revcomp(ref_seq) if reverse else ref_seq
        orig_rate = rate[::-1] if reverse else rate
        entries, ml = [], []
        for base, strand in (("A", "+"), ("T", "-")):
            hits = [k for k, b in enumerate(original) if b == base and rng.random() < orig_rate[k]]
            entries.append(_mm_entry(original, hits, base, strand))
            ml.extend([250] * len(hits))
        read = pysam.AlignedSegment(header)
        read.query_name = f"r{i:03d}"
        read.query_sequence = revcomp(original) if reverse else original
        read.flag = 16 if reverse else 0
        read.reference_id = 0
        read.reference_start = start
        read.mapping_quality = 60
        read.cigartuples = [(0, length)]
        read.query_qualities = pysam.qualitystring_to_array("I" * length)
        read.set_tag("MM", "".join(entries))
        read.set_tag("ML", ml)
        reads.append(read)
    unsorted = str(path) + ".unsorted.bam"
    with pysam.AlignmentFile(unsorted, "wb", header=header) as out:
        for read in reads:
            out.write(read)
    pysam.sort("-o", str(path), unsorted)
    os.remove(unsorted)
    pysam.index(str(path))
    return path


def write_plasmid_inputs(workdir: Path) -> tuple[Path, Path]:
    """A circular GenBank map and DddB-like FASTQ reads (as tests/test_pipeline.py)."""
    rng = random.Random(11)
    ref = "".join(rng.choice("ACGT") for _ in range(6000))
    gb = workdir / "pSmoke.gbk"
    lines = [f"LOCUS       pSmoke  {len(ref)} bp    DNA     circular SYN 01-JAN-2026",
             "FEATURES             Location/Qualifiers", "ORIGIN"]
    for i in range(0, len(ref), 60):
        chunk = ref[i:i + 60].lower()
        lines.append(f"{i + 1:>9} " + " ".join(chunk[j:j + 10] for j in range(0, 60, 10)))
    lines.append("//")
    gb.write_text("\n".join(lines) + "\n")
    rng = random.Random(12)
    fastq = workdir / "reads.fastq"
    with fastq.open("w") as handle:
        for i in range(80):
            size = rng.randint(1500, 2600)
            start = rng.randrange(0, len(ref))
            mol = (ref + ref)[start:start + size]
            seq = list(mol if rng.random() < 0.5 else revcomp(mol))
            for k, base in enumerate(seq):
                protected = (k // 180) % 2 == 0
                if base == "C" and rng.random() < (0.005 if protected else 0.08):
                    seq[k] = "T"
            read = "".join(seq)
            handle.write(f"@read{i}\n{read}\n+\n{'I' * len(read)}\n")
    return gb, fastq


def footprint_reads(path: Path) -> int:
    """Mapped reads carrying paired non-empty ns/nl and an MA tag."""
    n = 0
    with pysam.AlignmentFile(str(path), check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped or not (read.has_tag("ns") and read.has_tag("nl")):
                continue
            starts, lengths = list(read.get_tag("ns")), list(read.get_tag("nl"))
            if starts and len(starts) == len(lengths) and read.has_tag("MA"):
                n += 1
    return n


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-4000:] + result.stderr[-4000:])
        raise SystemExit(f"{Path(cmd[0]).name} exited {result.returncode}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bin", required=True, help="Directory holding the installed fiberhmm-* scripts")
    parser.add_argument("--workdir", required=True, help="Scratch directory (created)")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Run only fiberhmm-call (for a machine without minimap2)")
    args = parser.parse_args(argv)
    bindir = Path(args.bin)
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("FIBERHMM_NO_UPDATE_CHECK", "1")

    call = bindir / "fiberhmm-call"
    bam = make_hia5_bam(work / "hia5.bam")
    called = work / "hia5.call.bam"
    run([str(call), "-i", str(bam), "-o", str(called), "--enzyme", "hia5", "--seq", "pacbio",
         "-c", "1", "--no-qc", "--min-read-length", "0"])
    n = footprint_reads(called)
    if n == 0:
        raise SystemExit(f"fiberhmm-call wrote no footprint-tagged read to {called}")
    print(f"ok   fiberhmm-call: {n} footprint-tagged reads", flush=True)

    if args.skip_pipeline:
        return 0
    if shutil.which("minimap2") is None:
        raise SystemExit("minimap2 is not on PATH; install it or pass --skip-pipeline")
    gb, fastq = write_plasmid_inputs(work)
    out = work / "pipeline"
    progress = work / "progress.jsonl"
    env_mmi = work / "mmi"
    os.environ["FIBERHMM_MINIMAP2_INDEX_DIR"] = str(env_mmi)
    run([str(bindir / "fiberhmm-pipeline"), str(fastq), "--reference", str(gb), "--enzyme", "dddb",
         "-o", str(out), "--sample", "smoke", "-c", "2", "--min-read-length", "500",
         "--progress-json", str(progress)])
    outputs = json.loads((out / "outputs.json").read_text())
    assert outputs["schema"] == "fiberhmm.pipeline.outputs.v1", outputs.get("schema")
    last = json.loads(progress.read_text().splitlines()[-1])
    assert last["event"] == "done" and last["status"] == "ok", last
    n = footprint_reads(Path(outputs["called_bam"]))
    if n == 0:
        raise SystemExit(f"fiberhmm-pipeline wrote no footprint-tagged read to {outputs['called_bam']}")
    print(f"ok   fiberhmm-pipeline: {n} footprint-tagged reads; QC "
          f"{outputs['qc']['verdicts']['overall']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
