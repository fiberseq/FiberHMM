"""fiberhmm-lift end to end through a real graph (needs vg and fibertools-rs ft).

Skipped unless ``vg`` and ``ft`` are found: ``FIBERHMM_TEST_VG`` /
``FIBERHMM_TEST_FT`` (a binary or a wrapper script, e.g. one running vg in a
container) or PATH.

Fixture ``fixtures/graph_lift/hg002_chr1_window.gbz`` (124 kB) is a
cactus-pangenome graph of three public sequences: HG002 Q100 v1.1
chr1_MATERNAL:21,000,001-21,300,000 (path ``HG002_MAT#0#chr1_MATERNAL``),
chr1_PATERNAL:21,010,001-21,340,000 (``HG002_PAT#0#chr1_PATERNAL``) and the
reverse complement of GRCh38 chr1:20,850,001-21,180,000 (``GRCh38RC#0#chr1``),
each with window coordinates. Lifting MAT -> PAT keeps every read's strand;
MAT -> GRCh38RC flips every read. Reads are synthetic, cut from the graph's
own MAT path.
"""
from __future__ import annotations

import array
import collections
import os
import shutil
import subprocess
from pathlib import Path

import pysam
import pytest

from fiberhmm.io import graph_lift as gl

GRAPH = Path(__file__).parent / "fixtures" / "graph_lift" / "hg002_chr1_window.gbz"


def _tool(env, name):
    path = os.environ.get(env) or shutil.which(name)
    return path if path and os.access(path, os.X_OK) else None


VG = _tool("FIBERHMM_TEST_VG", "vg")
FT = _tool("FIBERHMM_TEST_FT", "ft")
pytestmark = pytest.mark.skipif(not (VG and FT), reason="needs vg and fibertools-rs ft")


def _path_fasta(path_name, contig, out):
    """One graph path as FASTA, renamed to its contig (as pg-pansn strips it)."""
    text = subprocess.run([VG, "paths", "-x", str(GRAPH), "-F", "-Q", path_name],
                          capture_output=True, text=True, check=True).stdout
    seq = "".join(line.strip() for line in text.splitlines() if not line.startswith(">"))
    out.write_text(f">{contig}\n{seq}\n")
    pysam.faidx(str(out))
    return seq


def _source_bam(tmp_path, mat):
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": "chr1_MATERNAL", "LN": len(mat)}],
              "PG": [{"ID": "fiberhmm-call", "PN": "fiberhmm-call", "VN": "3.0.2",
                      "CL": "fiberhmm-call -i reads.bam", "DS": "coord=molecular"}],
              "CO": ["MA-TYPES:v1:nuc"]}
    reads = []

    def add(name, flag, pos, cigar, seq, tags=()):
        reads.append((pos, name, flag, cigar, seq, tags))

    add("fwd", 0, 10_000, "3000M", mat[10_000:13_000],
        [("MA", "3000;nuc.:101-147,1001-147")])
    add("rev", 16, 20_000, "3000M", gl.revcomp(mat[20_000:23_000]),
        [("MA", "3000;nuc.:201-147")])
    daf = list(mat[30_000:33_000])
    for i in range(0, len(daf), 7):
        if daf[i] == "C":
            daf[i] = "Y"
    add("daf", 0, 30_000, "3000M", "".join(daf), [("st", "CT")])
    molecule = mat[50_000:53_000] + mat[120_000:121_500]
    add("chim", 0, 50_000, "3000M1500S", molecule,
        [("SA", "chr1_MATERNAL,120001,+,3000H1500M,60,0;")])
    add("chim", 0x800, 120_000, "3000H1500M", molecule[3000:],
        [("SA", "chr1_MATERNAL,50001,+,3000M1500S,60,0;")])
    add("lonely", 0x800, 200_000, "500H1000M", mat[200_000:201_000])   # primary absent
    path = tmp_path / "src.bam"
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for pos, name, flag, cigar, seq, tags in sorted(reads):
            read = pysam.AlignedSegment(out.header)
            read.query_name, read.flag = name, flag
            read.reference_id, read.reference_start = 0, pos
            read.mapping_quality, read.cigarstring = 60, cigar
            read.query_sequence = seq
            read.query_qualities = array.array("B", [30] * len(seq))
            for tag, value in tags:
                read.set_tag(tag, value)
            out.write(read)
    pysam.index(str(path))
    return path


@pytest.fixture(scope="module")
def lifted(tmp_path_factory):
    from fiberhmm.cli import lift

    tmp = tmp_path_factory.mktemp("graph_lift")
    mat = _path_fasta("HG002_MAT", "chr1_MATERNAL", tmp / "mat.fa")
    _path_fasta("GRCh38RC", "chr1", tmp / "rc.fa")
    src = _source_bam(tmp, mat)
    out = {}
    for target, fasta in (("HG002_PAT", None), ("GRCh38RC", tmp / "rc.fa")):
        path = tmp / f"{target}.bam"
        argv = ["-i", str(src), "-o", str(path), "-g", str(GRAPH), "--prefix", "HG002_MAT#0#",
                "--target", target, "--vg", VG, "--ft", FT, "-t", "2"]
        if fasta:
            argv += ["--target-fasta", str(fasta)]
        assert lift.main(argv) == 0
        with pysam.AlignmentFile(str(path)) as bam:
            out[target] = (list(bam.fetch(until_eof=True)), bam.header.to_dict())
    return out


def _by_name(records):
    return {(r.query_name, r.is_supplementary): r for r in records}


@pytest.mark.parametrize("target, flipped", [("HG002_PAT", False), ("GRCh38RC", True)])
def test_lift_repairs_flags_strand_and_history(lifted, target, flipped):
    records, header = lifted[target]
    by = _by_name(records)
    assert ("lonely", True) not in by                       # excluded: no primary
    primaries = collections.Counter(r.query_name for r in records
                                    if not r.is_supplementary and not r.is_secondary)
    assert set(primaries) == {"fwd", "rev", "daf", "chim"} and set(primaries.values()) == {1}
    supp = by[("chim", True)]
    assert supp.infer_read_length() == 4500 and "S" in supp.cigarstring
    assert by[("fwd", False)].is_reverse == flipped
    assert by[("rev", False)].is_reverse != flipped
    daf = by[("daf", False)]
    assert daf.get_tag("st") == ("GA" if flipped else "CT")
    assert ("R" in daf.query_sequence) == flipped and ("Y" in daf.query_sequence) != flipped
    assert by[("fwd", False)].get_tag("MA") == "3000;nuc.:101-147,1001-147"
    assert by[("fwd", False)].get_tag("Yp") == "chr1_MATERNAL:10001:+"
    assert all(not r.has_tag("Yf") and not r.has_tag("Yh") for r in records)
    pgs = header["PG"]
    assert pgs[-1]["ID"] == "fiberhmm-lift" and pgs[-1]["PP"] == pgs[-2]["ID"]
    assert "orphan_supplementary_no_primary_in_input=1" in pgs[-1]["DS"]
    assert f"target={target}" in pgs[-1]["DS"]
    assert header["HD"]["SO"] == "coordinate"
    if flipped:
        assert daf.has_tag("MD") and daf.get_tag("NM") > 0           # Y vs C counts
        assert by[("chim", False)].get_tag("SA").startswith("chr1,")
    else:
        assert not any(r.has_tag("MD") or r.has_tag("SA") for r in records)
