"""fiberhmm-lift --engine native end to end through a real graph (needs graphlift).

Skipped unless graphlift's ``liftbam``, ``gbz2chain`` and ``pathseq`` are found:
``FIBERHMM_TEST_GRAPHLIFT`` (the directory holding them, e.g.
``graphlift/target/release``) or PATH. No vg or container.

The graph is the vg test's fixture (``fixtures/graph_lift/hg002_chr1_window.gbz``,
see ``test_graph_lift_vg.py``); the chain maps are built from it here with
``gbz2chain``. Lifting MAT -> PAT keeps every read's strand; MAT -> GRCh38RC
flips every read.
"""
from __future__ import annotations

import array
import collections
import json
import os
import shutil
import subprocess
from pathlib import Path

import pysam
import pytest

from fiberhmm.io import graph_lift as gl

GRAPH = Path(__file__).parent / "fixtures" / "graph_lift" / "hg002_chr1_window.gbz"


def _tool(name):
    base = os.environ.get("FIBERHMM_TEST_GRAPHLIFT")
    path = os.path.join(base, name) if base else shutil.which(name)
    return path if path and os.path.isfile(path) and os.access(path, os.X_OK) else None


LIFTBAM, GBZ2CHAIN, PATHSEQ = (_tool(n) for n in ("liftbam", "gbz2chain", "pathseq"))
pytestmark = pytest.mark.skipif(not (LIFTBAM and GBZ2CHAIN and PATHSEQ),
                                reason="needs graphlift liftbam, gbz2chain and pathseq")


def _map(tmp, target, name):
    chain = tmp / f"{name}.chain"
    subprocess.run([GBZ2CHAIN, "-g", str(GRAPH), "-t", target, "-s", "HG002_MAT", "-o", str(chain)],
                   check=True, capture_output=True)
    manifest = Path(str(chain) + ".graphlift.json")
    return manifest, json.loads(manifest.read_text())


def _fasta(path, contig, size, out):
    text = subprocess.run([PATHSEQ, str(GRAPH), f"{path}:0-{size}"], capture_output=True,
                          text=True, check=True).stdout
    seq = "".join(line.strip() for line in text.splitlines() if not line.startswith(">"))
    out.write_text(f">{contig}\n{seq.replace('-', 'N')}\n")
    pysam.faidx(str(out))
    return seq


def _source_bam(tmp, mat):
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": "chr1_MATERNAL", "LN": len(mat)}],
              "PG": [{"ID": "fiberhmm-call", "PN": "fiberhmm-call", "VN": "3.0.2",
                      "CL": "fiberhmm-call -i reads.bam", "DS": "coord=molecular"}],
              "CO": ["MA-TYPES:v1:nuc"]}
    reads = []

    def add(name, flag, pos, cigar, seq, tags=()):
        reads.append((pos, name, flag, cigar, seq, tags))

    add("fwd", 0, 10_000, "3000M", mat[10_000:13_000],
        [("MA", "3000;nuc.:101-147,1001-147"),
         ("MM", "A+a.,0,5,9;"), ("ML", array.array("B", [250, 128, 7]))])
    # SEQ is reference-forward; the molecule (and MM) is its reverse complement.
    add("rev", 16, 20_000, "3000M", mat[20_000:23_000],
        [("MA", "3000;nuc.:201-147"), ("MM", "A+a.,2,3;"), ("ML", array.array("B", [200, 100]))])
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
    path = tmp / "src.bam"
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


def _molecular_mods(read):
    """(molecular position, code, ML) of every modification, as htslib decodes them."""
    n = read.query_length
    out = []
    for (base, strand, code), calls in read.modified_bases.items():
        for pos, q in calls:
            out.append((n - 1 - pos if read.is_reverse else pos, code, q))
    return sorted(out)


@pytest.fixture(scope="module")
def lifted(tmp_path_factory):
    from fiberhmm.cli import lift

    tmp = tmp_path_factory.mktemp("graph_lift_native")
    pat_map, pat = _map(tmp, "HG002_PAT#0", "pat")
    rc_map, rc = _map(tmp, "GRCh38RC#0", "rc")
    mat_size = next(s["size"] for s in rc["sources"] if s["name"] == "chr1_MATERNAL")
    mat = _fasta("HG002_MAT#0#chr1_MATERNAL", "chr1_MATERNAL", mat_size, tmp / "mat.fa")
    rc_target = rc["targets"][0]
    _fasta(rc_target["path"], rc_target["name"], rc_target["size"], tmp / "rc.fa")
    src = _source_bam(tmp, mat)
    with pysam.AlignmentFile(str(src)) as bam:
        source = {(r.query_name, r.is_supplementary): r for r in bam.fetch(until_eof=True)}
    out = {}
    for target, manifest, fasta in (("HG002_PAT", pat_map, None), ("GRCh38RC", rc_map, tmp / "rc.fa")):
        path = tmp / f"{target}.bam"
        argv = ["-i", str(src), "-o", str(path), "--map", str(manifest), "--liftbam", LIFTBAM,
                "-t", "2", "--unlifted", str(tmp / f"{target}.unlifted.bam")]
        if fasta:
            argv += ["--target-fasta", str(fasta)]
        assert lift.main(argv) == 0
        with pysam.AlignmentFile(str(path)) as bam:
            out[target] = (list(bam.fetch(until_eof=True)), bam.header.to_dict())
    return out, source


@pytest.mark.parametrize("target, flipped", [("HG002_PAT", False), ("GRCh38RC", True)])
def test_native_lift_repairs_flags_strand_and_history(lifted, target, flipped):
    out, source = lifted
    records, header = out[target]
    by = {(r.query_name, r.is_supplementary): r for r in records}
    assert ("lonely", True) not in by                       # excluded: no primary
    primaries = collections.Counter(r.query_name for r in records
                                    if not r.is_supplementary and not r.is_secondary)
    assert set(primaries) == {"fwd", "rev", "daf", "chim"} and set(primaries.values()) == {1}
    supp = by[("chim", True)]
    assert supp.infer_read_length() == 4500
    assert by[("fwd", False)].is_reverse == flipped
    assert by[("rev", False)].is_reverse != flipped
    daf = by[("daf", False)]
    assert daf.get_tag("st") == ("GA" if flipped else "CT")
    assert ("R" in daf.query_sequence) == flipped and ("Y" in daf.query_sequence) != flipped
    assert by[("fwd", False)].get_tag("MA") == "3000;nuc.:101-147,1001-147"
    assert by[("fwd", False)].get_tag("Yp") == "chr1_MATERNAL:10001:+"
    assert all(not r.has_tag("Yf") and not r.has_tag("Yh") for r in records)
    # MM/ML are copied, and decode to the same molecular calls whatever the strand.
    for name in ("fwd", "rev"):
        lifted_read, src_read = by[(name, False)], source[(name, False)]
        assert lifted_read.get_tag("MM") == src_read.get_tag("MM")
        assert _molecular_mods(lifted_read) == _molecular_mods(src_read)
        assert len(_molecular_mods(lifted_read)) in (2, 3)
    pgs = header["PG"]
    assert pgs[-1]["ID"] == "fiberhmm-lift" and pgs[-1]["PP"] == pgs[-2]["ID"]
    assert pgs[-2]["PN"] == "liftbam" and pgs[-2]["PP"] == "fiberhmm-call"
    assert "engine=native" in pgs[-1]["DS"]
    assert "orphan_supplementary_no_primary_in_input=1" in pgs[-1]["DS"]
    assert header["HD"]["SO"] == "coordinate"
    assert gl.header_lift_state(pysam.AlignmentHeader.from_dict(header))["repaired"]
    if flipped:
        assert daf.has_tag("MD") and daf.get_tag("NM") > 0           # Y vs C counts
        assert by[("chim", False)].get_tag("SA").startswith("chr1,")
    else:
        assert not any(r.has_tag("MD") or r.has_tag("SA") for r in records)


def test_native_lift_places_reads_where_the_target_agrees(lifted):
    """Reads cut from the source path land where the target FASTA has the same bases
    (MD/NM from calmd): HG002 and GRCh38 differ at well under 1% of aligned bases."""
    out, _source = lifted
    records, _header = out["GRCh38RC"]
    for r in records:
        if r.query_name in ("fwd", "rev", "chim"):
            assert r.get_tag("NM") <= 0.01 * r.query_alignment_length, (r.query_name, r.cigarstring)
