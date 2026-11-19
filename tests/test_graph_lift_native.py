"""fiberhmm-lift --engine native end to end through a real graph (needs graphlift).

Skipped unless graphlift's ``liftbam``, ``gbz2chain`` and ``pathseq`` are found
(the BED tests also need ``liftbed``): ``FIBERHMM_TEST_GRAPHLIFT`` (the directory
holding them, e.g. ``graphlift/target/release``, built with ``--features bam``)
or PATH. No vg or container.

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


LIFTBAM, GBZ2CHAIN, PATHSEQ, LIFTBED = (_tool(n) for n in ("liftbam", "gbz2chain", "pathseq",
                                                            "liftbed"))
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
    out["maps"] = {"HG002_PAT": pat_map, "GRCh38RC": rc_map}
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
    # Every record, not just one per (name, supplementary): nothing else was split off.
    assert sorted((r.query_name, r.is_supplementary, r.is_secondary) for r in records) == [
        ("chim", False, False), ("chim", True, False), ("daf", False, False),
        ("fwd", False, False), ("rev", False, False)]
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


def _footprint_span(read, start, length):
    """Target span ``[first, last + 1)`` of molecular interval ``[start, start + length)``."""
    n = read.infer_read_length()
    if read.is_reverse:
        start = n - start - length
    q2r = dict(read.get_aligned_pairs(matches_only=True))
    refs = [q2r[q] for q in range(start, start + length) if q in q2r]
    return min(refs), max(refs) + 1


@pytest.mark.skipif(not LIFTBED, reason="needs graphlift liftbed")
@pytest.mark.parametrize("target, flipped", [("HG002_PAT", False), ("GRCh38RC", True)])
def test_native_bed_lift_lands_where_liftbam_puts_the_footprints(lifted, tmp_path, target, flipped):
    """The fwd read's nucleosomes (molecular 100+147, 1000+147) and the rev read's (200+147),
    as BED12 and BED6 rows in source coordinates, lift to the spans the same footprints take
    in the liftbam-lifted BAM."""
    from fiberhmm.cli import lift

    out, _source = lifted
    records, _header = out[target]
    by = {(r.query_name, r.is_supplementary): r for r in records}
    manifest = out["maps"][target]
    expected = {"fwd.0": _footprint_span(by[("fwd", False)], 100, 147),
                "fwd.1": _footprint_span(by[("fwd", False)], 1000, 147),
                "rev.0": _footprint_span(by[("rev", False)], 200, 147)}
    # Source spans: fwd at 10,000 forward; rev at 20,000 reversed (molecule = revcomp SEQ).
    source = {"fwd.0": (10_100, 10_247), "fwd.1": (11_000, 11_147), "rev.0": (22_653, 22_800)}
    bed6 = tmp_path / "nuc.bed"
    bed6.write_text("".join(f"chr1_MATERNAL\t{s}\t{e}\t{k}\t0\t+\tx\n" for k, (s, e) in source.items()))
    lifted_bed = tmp_path / "nuc.lifted.bed"
    argv = ["--bed", "-i", str(bed6), "-o", str(lifted_bed), "--map", str(manifest),
            "--liftbed", LIFTBED, "--unlifted", str(tmp_path / "u.bed")]
    assert lift.main(argv) == 0
    got = {f[3]: ((int(f[1]), int(f[2])), f[5], f[6])
           for f in (line.split("\t") for line in lifted_bed.read_text().splitlines())}
    strand = "-" if flipped else "+"
    assert got == {k: (v, strand, "x") for k, v in expected.items()}
    # The fwd read's two nucleosomes as one BED12 row.
    (s0, e0), (s1, e1) = source["fwd.0"], source["fwd.1"]
    bed12 = tmp_path / "nuc12.bed"
    bed12.write_text(f"chr1_MATERNAL\t{s0}\t{e1}\tfwd\t0\t+\t{s0}\t{e1}\t0\t2\t"
                     f"{e0 - s0},{e1 - s1}\t0,{s1 - s0}\ta,b\n")
    assert lift.main(["--bed", "-i", str(bed12), "-o", str(lifted_bed), "--map", str(manifest),
                      "--liftbed", LIFTBED]) == 0
    f = lifted_bed.read_text().split("\t")
    spans = sorted([expected["fwd.0"], expected["fwd.1"]])
    assert (int(f[1]), int(f[2]), f[5]) == (spans[0][0], spans[1][1], strand)
    starts = [int(x) + int(f[1]) for x in f[11].split(",")]
    sizes = [int(x) for x in f[10].split(",")]
    assert [(a, a + b) for a, b in zip(starts, sizes)] == spans
    assert f[12].strip() == ("b,a" if flipped else "a,b")
