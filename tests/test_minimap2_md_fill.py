"""MD tags when the aligner writes none (minimap2 2.27 ignores --MD).

minimap2 2.27 (the Ubuntu 24.04 package) writes a ``cs`` tag in place of
``MD`` when given ``--MD``; fiberhmm-pipeline then stopped at fiberhmm-call
("no deamination-site source"). The pipeline now computes MD from the
reference. The end-to-end tests run fiberhmm-pipeline through a ``minimap2``
wrapper that reports version 2.27 and swaps ``--MD`` for ``--cs`` (exactly
2.27's output), and compare with a run on the real minimap2. Synthetic data only.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pysam
import pytest

from fiberhmm.cli.call import _check_daf_inputs
from fiberhmm.pipeline.circular import compute_md_nm
from fiberhmm.pipeline.md_fill import (
    MissingMdFiller,
    md_nm_linear,
    minimap2_md_broken,
    minimap2_md_bug_note,
)

REPO = Path(__file__).resolve().parents[1]
REAL_MINIMAP2 = shutil.which("minimap2")
needs_minimap2 = pytest.mark.skipif(REAL_MINIMAP2 is None, reason="minimap2 not installed")
COMP = str.maketrans("ACGT", "TGCA")


def revcomp(seq: str) -> str:
    return seq.translate(COMP)[::-1]


def random_seq(n: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


def deaminate(seq: str, rng: random.Random) -> str:
    out = list(seq)
    for i, base in enumerate(out):
        if base == "C" and rng.random() < (0.005 if (i // 180) % 2 == 0 else 0.08):
            out[i] = "T"
    return "".join(out)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_minimap2_md_broken_is_exactly_2_27():
    assert minimap2_md_broken("2.27-r1193")
    assert minimap2_md_broken("2.27")
    for version in ("2.26-r1175", "2.24-r1122", "2.28-r1209", "2.30-r1287", "", "unknown"):
        assert not minimap2_md_broken(version), version


def _random_alignment(rng: random.Random, ref: str):
    """(query, cigar, ref_start) with mismatches, indels, N skips and clips."""
    pos = rng.randrange(0, 200)
    r = pos
    query, cigar = [], []
    lead = rng.choice([0, 0, 15])
    if lead:
        cigar.append((5 if rng.random() < 0.5 else 4, lead))
        if cigar[-1][0] == 4:
            query.append(random_seq(lead, rng.randrange(10 ** 6)))
    for _ in range(rng.randint(1, 8)):
        op = rng.choice([0, 0, 0, 1, 2, 3, 7, 8])
        n = rng.randint(1, 40)
        if op in (0, 7, 8):
            block = list(ref[r:r + n])
            for i in range(len(block)):
                if rng.random() < 0.1:
                    block[i] = rng.choice("ACGTN")
            query.append("".join(block))
            r += n
        elif op == 1:
            query.append(random_seq(n, rng.randrange(10 ** 6)))
        else:
            r += n
        cigar.append((op, n))
    cigar.append((0, 5))
    query.append(ref[r:r + 5])
    r += 5
    if rng.random() < 0.5:
        cigar.append((4, 7))
        query.append("ACGTACG")
    return "".join(query), cigar, pos, r


def test_md_nm_linear_matches_compute_md_nm():
    rng = random.Random(3)
    ref = "".join(rng.choice("ACGTN" if i % 97 == 0 else "ACGT") for i in range(2000))
    for _ in range(300):
        query, cigar, pos, end = _random_alignment(rng, ref)
        assert md_nm_linear(query, cigar, ref[pos:end]) == compute_md_nm(query, cigar, ref, pos)


def _header(contigs, pg_version=None):
    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": n, "LN": len(s)} for n, s in contigs]}
    if pg_version:
        header["PG"] = [{"ID": "minimap2", "PN": "minimap2", "VN": pg_version,
                         "CL": "minimap2 -a -x map-ont --MD -Y ref.mmi -"}]
    return pysam.AlignmentHeader.from_dict(header)


def _record(header, contig_id, start, query, cigar, flag=0, tags=()):
    read = pysam.AlignedSegment(header)
    read.query_name = "r"
    read.flag = flag
    read.reference_id = contig_id
    read.reference_start = start
    read.mapping_quality = 60
    read.query_sequence = query
    read.cigartuples = cigar
    for tag, value, kind in tags:
        read.set_tag(tag, value, kind)
    return read


def test_filler_adds_md_to_primary_and_supplementary_records(tmp_path):
    chr_a, chr_b = random_seq(3000, 1), random_seq(2000, 2)
    fasta = tmp_path / "ref.fa"
    fasta.write_text(f">chrA\n{chr_a}\n>chrB\n{chr_b}\n")
    pysam.faidx(str(fasta))
    header = _header([("chrA", chr_a), ("chrB", chr_b)])
    q1 = list(chr_a[100:700])
    q1[50] = "A" if q1[50] != "A" else "C"
    q1 = "".join(q1)
    primary = _record(header, 0, 100, "GGGG" + q1, [(4, 4), (0, 600)],
                      tags=[("NM", 1, "i"), ("cs", ":50*xx:549", "Z")])
    supp = _record(header, 1, 500, chr_b[500:600] + chr_b[610:800], [(5, 30), (0, 100), (2, 10), (0, 190)],
                   flag=2048)
    has_md = _record(header, 0, 0, chr_a[:50], [(0, 50)], tags=[("MD", "keep", "Z")])
    unmapped = pysam.AlignedSegment(header)
    unmapped.query_name, unmapped.flag, unmapped.query_sequence = "u", 4, "ACGT"
    filler = MissingMdFiller(str(fasta))
    assert filler(primary) and filler(supp)
    assert not filler(has_md) and not filler(unmapped)
    filler.close()
    assert filler.filled == 2
    assert primary.get_tag("MD") == compute_md_nm(primary.query_sequence, primary.cigartuples,
                                                  chr_a, 100)[0]
    assert primary.get_tag("MD") == f"50{chr_a[150]}549"
    assert primary.get_tag("NM") == 1 and not primary.has_tag("cs")   # minimap2's NM kept
    assert supp.get_tag("MD") == "100^" + chr_b[600:610] + "190"
    assert supp.get_tag("NM") == 10
    assert has_md.get_tag("MD") == "keep"


def test_filler_reads_a_circular_contig_past_its_end(tmp_path):
    plasmid = random_seq(1000, 4)
    fasta = tmp_path / "p.fa"
    fasta.write_text(f">p\n{plasmid}\n")
    pysam.faidx(str(fasta))
    header = _header([("p", plasmid)])
    query = (plasmid + plasmid)[900:1200]
    read = _record(header, 0, 900, query, [(0, 300)])
    filler = MissingMdFiller(str(fasta), {})
    assert filler(read)
    filler.close()
    assert read.get_tag("MD") == "300" and read.get_tag("NM") == 0


def _raw_bam(path: Path, pg_version, with_md_every=0):
    ref = random_seq(5000, 7)
    header = _header([("chr1", ref)], pg_version)
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for i in range(6):
            tags = [("MD", "200", "Z")] if with_md_every and i % with_md_every == 0 else []
            read = _record(header, 0, 100 * i, ref[100 * i:100 * i + 200], [(0, 200)], tags=tags)
            read.query_name = f"r{i}"
            out.write(read)
    return path


def test_call_names_minimap2_2_27_when_md_is_missing(tmp_path, capsys):
    bam = _raw_bam(tmp_path / "mm227.bam", "2.27-r1193")
    with pytest.raises(SystemExit) as exc:
        _check_daf_inputs(str(bam), n_sniff=4)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "DAF-seq calling needs deamination calls" in err
    assert "minimap2 2.27-r1193" in err and "2.28" in err

    other = _raw_bam(tmp_path / "mm230.bam", "2.30-r1287")
    with pytest.raises(SystemExit):
        _check_daf_inputs(str(other), n_sniff=4)
    assert "ignores" not in capsys.readouterr().err
    assert minimap2_md_bug_note(_header([("c", "A")], "2.28-r1209")) == ""


def test_call_warns_when_only_some_2_27_reads_have_md(tmp_path, capsys):
    bam = _raw_bam(tmp_path / "partial.bam", "2.27-r1193", with_md_every=3)
    _check_daf_inputs(str(bam), n_sniff=6)
    err = capsys.readouterr().err
    assert "4/6 of the sniffed reads have no MD tag" in err and "2.28" in err


# ---------------------------------------------------------------------------
# fiberhmm-pipeline end to end, through an emulated minimap2 2.27
# ---------------------------------------------------------------------------

def _fake_minimap2_227(bindir: Path) -> Path:
    """A ``minimap2`` that reports 2.27 and, like 2.27, answers --MD with cs only."""
    bindir.mkdir(parents=True, exist_ok=True)
    script = bindir / "minimap2"
    script.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        f"REAL = {REAL_MINIMAP2!r}\n"
        "args = sys.argv[1:]\n"
        "if args == ['--version']:\n"
        "    print('2.27-r1193')\n"
        "    sys.exit(0)\n"
        "args = ['--cs' if a == '--MD' else a for a in args]\n"
        "os.execv(REAL, [REAL] + args)\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bindir


def _run_pipeline(args, path_prefix=None, timeout=300):
    env = dict(os.environ, FIBERHMM_NO_UPDATE_CHECK="1",
               PYTHONPATH=str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    if path_prefix:
        env["PATH"] = str(path_prefix) + os.pathsep + env.get("PATH", "")
    return subprocess.run([sys.executable, "-m", "fiberhmm.cli.pipeline", *args],
                          capture_output=True, text=True, timeout=timeout, env=env)


def _records(path: str) -> dict:
    out = {}
    with pysam.AlignmentFile(path, check_sq=False) as bam:
        for r in bam.fetch(until_eof=True):
            out[(r.query_name, r.flag, r.reference_name, r.reference_start)] = (
                r.cigarstring, r.query_sequence, sorted(r.get_tags()))
    return out


def _both_runs(tmp_path, monkeypatch, reads, reference, extra=()):
    monkeypatch.setenv("FIBERHMM_MINIMAP2_INDEX_DIR", str(tmp_path / "mmi"))
    fake = _fake_minimap2_227(tmp_path / "fakebin")
    outputs = {}
    for label, prefix in (("real", None), ("mm227", fake)):
        out = tmp_path / label
        result = _run_pipeline([str(reads), "--reference", str(reference), "--enzyme", "dddb",
                                "-o", str(out), "-c", "2", "--min-read-length", "500",
                                "--no-qc", *extra], prefix)
        assert result.returncode == 0, f"{label}: " + result.stderr[-3000:]
        outputs[label] = json.loads((out / "outputs.json").read_text())
    log = (tmp_path / "mm227" / "logs" / "pipeline.log").read_text()
    assert "2.27-r1193 ignores --MD" in log
    assert "without an MD tag; MD was computed from the reference" in log
    real, emulated = (_records(outputs[k]["aligned_bam"]) for k in ("real", "mm227"))
    assert real and emulated == real            # MD, NM and every other tag; no cs
    for record in emulated.values():
        tags = dict(record[2])
        assert "MD" in tags and "cs" not in tags
    assert _records(outputs["mm227"]["called_bam"]) == _records(outputs["real"]["called_bam"])
    return emulated


@needs_minimap2
def test_pipeline_under_minimap2_2_27_linear_reference_with_supplementary(tmp_path, monkeypatch):
    genome = random_seq(30000, 31)
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chrA\n" + genome[:20000] + "\n>chrB\n" + genome[20000:] + "\n")
    rng = random.Random(32)
    reads = tmp_path / "reads.fastq"
    with open(reads, "w") as handle:
        for i in range(40):
            start = rng.randrange(1000, 15000)
            mol = genome[start:start + 2000]
            if i % 4 == 0:   # chimeric molecule: a supplementary arm on chrB
                arm = rng.randrange(21000, 27000)
                mol += genome[arm:arm + 1200]
            seq = deaminate(mol if i % 2 else revcomp(mol), rng)
            handle.write(f"@m{i}\n{seq}\n+\n{'I' * len(seq)}\n")
    records = _both_runs(tmp_path, monkeypatch, reads, fasta)
    assert any(flag & 2048 for (_, flag, _, _) in records)


@needs_minimap2
def test_pipeline_under_minimap2_2_27_circular_plasmid(tmp_path, monkeypatch):
    ref = random_seq(6000, 41)
    gb = tmp_path / "pMd.gbk"
    lines = [f"LOCUS       pMd  {len(ref)} bp    DNA     circular SYN 01-JAN-2026",
             "FEATURES             Location/Qualifiers", "ORIGIN"]
    for i in range(0, len(ref), 60):
        chunk = ref[i:i + 60].lower()
        lines.append(f"{i + 1:>9} " + " ".join(chunk[j:j + 10] for j in range(0, 60, 10)))
    gb.write_text("\n".join(lines + ["//"]) + "\n")
    rng = random.Random(42)
    reads = tmp_path / "reads.fastq"
    with open(reads, "w") as handle:
        for i in range(50):
            size = rng.randint(1500, 2600)
            start = rng.randrange(0, len(ref))
            mol = (ref + ref)[start:start + size]
            seq = deaminate(mol if rng.random() < 0.5 else revcomp(mol), rng)
            handle.write(f"@read{i}\n{seq}\n+\n{'I' * len(seq)}\n")
    records = _both_runs(tmp_path, monkeypatch, reads, gb)
    assert len(records) == 50
    # Some molecules cross the origin: merged records run past the contig end.
    with pysam.AlignmentFile(str(next((tmp_path / "mm227").glob("*.aligned.bam")))) as bam:
        assert any(r.reference_end > len(ref) for r in bam.fetch(until_eof=True))
