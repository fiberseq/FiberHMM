"""fiberhmm-lift --bed --engine vg through a real graph (needs vg only, no fibertools-rs).

Skipped unless ``vg`` is found (``FIBERHMM_TEST_VG``, a binary or a wrapper script, or
PATH). The graph is ``fixtures/graph_lift/hg002_chr1_window.gbz`` (see
``test_graph_lift_vg.py``). When graphlift's ``gbz2chain`` and ``liftbed`` are found too
(``FIBERHMM_TEST_GRAPHLIFT`` or PATH), the vg lift is compared with the native one.
"""
from __future__ import annotations

import gzip
import os
import shutil
import subprocess
from pathlib import Path

import pysam
import pytest

GRAPH = Path(__file__).parent / "fixtures" / "graph_lift" / "hg002_chr1_window.gbz"


def _tool(env, name):
    base = os.environ.get(env)
    if base and os.path.isdir(base):
        base = os.path.join(base, name)
    path = base or shutil.which(name)
    return path if path and os.path.isfile(path) and os.access(path, os.X_OK) else None


VG = _tool("FIBERHMM_TEST_VG", "vg")
GBZ2CHAIN = _tool("FIBERHMM_TEST_GRAPHLIFT", "gbz2chain")
LIFTBED = _tool("FIBERHMM_TEST_GRAPHLIFT", "liftbed")
pytestmark = pytest.mark.skipif(not VG, reason="needs vg")

# Source rows on HG002_MAT window coordinates: a two-block BED12 row, a narrowPeak with a
# summit and one without, and a BED9 whose thick interval lies inside the row.
BED12 = "chr1_MATERNAL\t10100\t11147\tm\t0\t+\t10100\t11147\t0\t2\t147,147\t0,900\ta,b\n"
NARROWPEAK = ("chr1_MATERNAL\t20000\t20400\tp1\t0\t.\t5.0\t3.0\t2.0\t150\n"
              "chr1_MATERNAL\t40000\t40300\tp2\t0\t.\t5.0\t3.0\t2.0\t-1\n")
BED9 = "chr1_MATERNAL\t60000\t61000\tt1\t0\t+\t60200\t60800\t0,0,0\n"


@pytest.fixture(scope="module")
def mat_fasta(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("vg_bed")
    text = subprocess.run([VG, "paths", "-x", str(GRAPH), "-F", "-Q", "HG002_MAT"],
                          capture_output=True, text=True, check=True).stdout
    seq = "".join(line.strip() for line in text.splitlines() if not line.startswith(">"))
    fasta = tmp / "mat.fa"
    fasta.write_text(f">chr1_MATERNAL\n{seq}\n")
    pysam.faidx(str(fasta))
    return fasta


def _lift(tmp_path, name, text, target, fasta, *extra, engine="vg", suffix=""):
    from fiberhmm.cli import lift

    src = tmp_path / f"{name}.src{suffix}"
    if suffix.endswith(".gz"):
        with gzip.open(src, "wt") as handle:
            handle.write(text)
    else:
        src.write_text(text)
    out = tmp_path / f"{name}.{engine}.{target}{suffix}"
    if engine == "vg":
        argv = ["--bed", "-i", str(src), "-o", str(out), "-g", str(GRAPH), "--prefix",
                "HG002_MAT#0#", "--target", target, "--vg", VG, "--source-fasta", str(fasta),
                "-t", "2", *extra]
    else:
        chain = tmp_path / f"{target}.chain"
        subprocess.run([GBZ2CHAIN, "-g", str(GRAPH), "-t", f"{target}#0", "-s", "HG002_MAT",
                        "-o", str(chain)], check=True, capture_output=True)
        argv = ["--bed", "-i", str(src), "-o", str(out), "--map", f"{chain}.graphlift.json",
                "--liftbed", LIFTBED, *extra]
    assert lift.main(argv) == 0
    opener = gzip.open if suffix.endswith(".gz") else open
    with opener(out, "rt") as handle:
        return [line.rstrip("\n").split("\t") for line in handle if line.strip()]


@pytest.mark.parametrize("target, flipped", [("HG002_PAT", False), ("GRCh38RC", True)])
def test_vg_bed_lifts_blocks_summits_and_thick_without_fibertools(tmp_path, mat_fasta, target,
                                                                  flipped):
    strand = "-" if flipped else "+"
    (row,) = _lift(tmp_path, "b12", BED12, target, mat_fasta, suffix=".bed.gz")
    assert row[3] == "m" and row[5] == strand and row[9] == "2"
    assert row[12] == ("b,a" if flipped else "a,b")
    sizes = [int(v) for v in row[10].split(",")]
    assert all(140 <= s <= 154 for s in sizes)
    peaks = {f[3]: f for f in _lift(tmp_path, "np", NARROWPEAK, target, mat_fasta,
                                    "--bed-format", "narrowPeak")}
    assert 0 <= int(peaks["p1"][9]) < int(peaks["p1"][2]) - int(peaks["p1"][1])
    assert peaks["p2"][9] == "-1"
    (thick,) = _lift(tmp_path, "b9", BED9, target, mat_fasta)
    assert thick[5] == strand
    assert int(thick[1]) < int(thick[6]) < int(thick[7]) < int(thick[2])
    if GBZ2CHAIN and LIFTBED:
        # Same graph, native: within 10 bp everywhere (indel placement inside bubbles).
        def close(a, b, cols):
            return all(abs(int(a[c]) - int(b[c])) <= 10 for c in cols)

        (n12,) = _lift(tmp_path, "b12", BED12, target, mat_fasta, engine="native", suffix=".bed.gz")
        assert n12[5] == row[5] and close(n12, row, (1, 2))
        npk = {f[3]: f for f in _lift(tmp_path, "np", NARROWPEAK, target, mat_fasta,
                                     "--bed-format", "narrowPeak", engine="native")}
        summit = lambda f: int(f[1]) + int(f[9])
        assert abs(summit(npk["p1"]) - summit(peaks["p1"])) <= 10
        (n9,) = _lift(tmp_path, "b9", BED9, target, mat_fasta, engine="native")
        assert close(n9, thick, (1, 2, 6, 7))


def test_vg_bed_reports_unlifted_rows(tmp_path, mat_fasta):
    rows = ("chr1_MATERNAL\t10100\t10300\tok\t0\t+\n"
            "chrNope\t1\t100\tmissing\t0\t+\n")
    unl = tmp_path / "unl.bed"
    out = _lift(tmp_path, "u", rows, "HG002_PAT", mat_fasta, "--unlifted", str(unl))
    assert [f[3] for f in out] == ["ok"]
    assert unl.read_text() == "#contig_not_in_fasta\nchrNope\t1\t100\tmissing\t0\t+\n"
