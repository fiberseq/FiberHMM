"""nuc MA groups survive fiberhmm-lift (native engine, needs graphlift).

Annotate reads with nucsz/nucseg/nucmol, lift them MAT -> PAT (same strand)
and MAT -> GRCh38RC (every read flips), and read the groups back: they must
be present, not stale, unchanged in molecular coordinates on whole records,
rebased with the other groups on a filled hard-clipped supplementary, and
reproducible by re-running molecule_outputs on the lifted record.

Skipped unless graphlift's liftbam, gbz2chain and pathseq are found
(``FIBERHMM_TEST_GRAPHLIFT`` or PATH).
"""
from __future__ import annotations

import array
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pysam
import pytest

from nuc_synth import complement_msps, flat_calibration, mm_for_marks, revcomp

from fiberhmm.io.bam_header import SCAN_MA_CONTRACT_PREFIX, declared_ma_types
from fiberhmm.nuc.molecule import molecule_outputs
from fiberhmm.nuc.records import DecodeContext, decode_record
from fiberhmm.nuc.tags import NUC_GROUPS, NUCMOL, NUCSEG, NUCSZ, annotate_header, annotate_record, read_nuc_groups

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


def _calls(L, rng, start=150, gap=190, core=147):
    nucs, pos = [], start
    while pos + core < L - 150:
        nucs.append((pos, pos + core))
        pos += gap + int(rng.integers(-6, 7))
    return nucs, complement_msps(L, nucs)


def _marks(mol_seq, nucs, rng, rate=0.6):
    occ = np.zeros(len(mol_seq), bool)
    for s, e in nucs:
        occ[s:e] = True
    a = np.array([b == "A" for b in mol_seq])
    m = np.flatnonzero(a & ~occ & (rng.random(len(mol_seq)) < rate))
    return m[(m >= 12) & (m < len(mol_seq) - 12)]


def _ma(L, nucs, msps):
    return (f"{L};nuc.QQQ:" + ",".join(f"{s + 1}-{e - s}" for s, e in nucs)
            + ";msp.:" + ",".join(f"{s + 1}-{e - s}" for s, e in msps)), [200, 255, 255] * len(nucs)


@pytest.fixture(scope="module")
def lifted(tmp_path_factory):
    from fiberhmm.cli import lift

    rng = np.random.default_rng(17)
    tmp = tmp_path_factory.mktemp("nuc_lift")
    pat_map, _pat = _map(tmp, "HG002_PAT#0", "pat")
    rc_map, rc = _map(tmp, "GRCh38RC#0", "rc")
    mat_size = next(s["size"] for s in rc["sources"] if s["name"] == "chr1_MATERNAL")
    mat = _fasta("HG002_MAT#0#chr1_MATERNAL", "chr1_MATERNAL", mat_size, tmp / "mat.fa")
    rc_target = rc["targets"][0]
    _fasta(rc_target["path"], rc_target["name"], rc_target["size"], tmp / "rc.fa")
    hdr = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1_MATERNAL", "LN": len(mat)}],
           "PG": [{"ID": "fiberhmm-call", "PN": "fiberhmm-call", "VN": "3.0.2",
                   "CL": "fiberhmm-call -i reads.bam --enzyme hia5 --seq pacbio",
                   "DS": "coord=molecular mode=pacbio-fiber"}],
           "CO": ["MA-TYPES:v1:nuc,msp"]}
    header = annotate_header(pysam.AlignmentHeader.from_dict(hdr), calls_program="fiberhmm-call")
    cal = flat_calibration()
    src = tmp / "src.bam"
    records = []

    def build(out, name, flag, pos, cigar, stored_seq, mol_seq, nucs, msps, marks, extra=()):
        r = pysam.AlignedSegment(out.header)
        r.query_name, r.flag = name, flag
        r.reference_id, r.reference_start = 0, pos
        r.mapping_quality, r.cigarstring = 60, cigar
        r.query_sequence = stored_seq
        r.query_qualities = array.array("B", [30] * len(stored_seq))
        ma, aq = _ma(len(mol_seq), nucs, msps)
        r.set_tag("MA", ma)
        r.set_tag("AQ", array.array("B", aq))
        mm, ml = mm_for_marks(mol_seq, marks)
        r.set_tag("MM", mm)
        r.set_tag("ML", ml)
        for tag, value in extra:
            r.set_tag(tag, value)
        mol, why = decode_record(r, DecodeContext(out.header))
        assert mol is not None, why
        annotate_record(r, molecule_outputs(mol, cal=cal))
        return r

    with pysam.AlignmentFile(str(src), "wb", header=header) as out:
        fwd = mat[10_000:16_000]
        n1, m1 = _calls(len(fwd), rng)
        records.append(build(out, "fwd", 0, 10_000, f"{len(fwd)}M", fwd, fwd, n1, m1, _marks(fwd, n1, rng)))
        rev_mol = revcomp(mat[20_000:26_000])         # molecule is the reverse strand
        n2, m2 = _calls(len(rev_mol), rng)
        records.append(build(out, "rev", 16, 20_000, f"{len(rev_mol)}M", mat[20_000:26_000], rev_mol,
                             n2, m2, _marks(rev_mol, n2, rng)))
        # chimera: primary holds 4000 M + 2500 S; the supplementary stores only its 2500 aligned
        # bases (hard-clipped) with partial-scope MA of its own
        mol = mat[50_000:54_000] + mat[120_000:122_500]
        n3, m3 = _calls(len(mol), rng)
        records.append(build(out, "chim", 0, 50_000, "4000M2500S", mol, mol, n3, m3, _marks(mol, n3, rng),
                             [("SA", "chr1_MATERNAL,120001,+,4000H2500M,60,0;")]))
        part = mol[4000:]
        n4, m4 = _calls(len(part), rng, start=120)
        records.append(build(out, "chim", 0x800, 120_000, "4000H2500M", part, part, n4, m4,
                             _marks(part, n4, rng), [("SA", "chr1_MATERNAL,50001,+,4000M2500S,60,0;")]))
        for r in sorted(records, key=lambda x: x.reference_start):
            out.write(r)
    pysam.index(str(src))
    with pysam.AlignmentFile(str(src)) as bam:
        source = {(r.query_name, r.is_supplementary): r for r in bam.fetch(until_eof=True)}
    out = {}
    for target, manifest, fasta in (("HG002_PAT", pat_map, None), ("GRCh38RC", rc_map, tmp / "rc.fa")):
        path = tmp / f"{target}.bam"
        argv = ["-i", str(src), "-o", str(path), "--map", str(manifest), "--liftbam", LIFTBAM, "-t", "2"]
        if fasta:
            argv += ["--target-fasta", str(fasta)]
        assert lift.main(argv) == 0
        with pysam.AlignmentFile(str(path)) as bam:
            out[target] = (list(bam.fetch(until_eof=True)), bam.header)
    return out, source, cal


def _groups(read):
    g = read_nuc_groups(read)
    return {k: (v["intervals"], v["q"], v["digest"], v["stale"]) for k, v in g.items()}


@pytest.mark.parametrize("target, flipped", [("HG002_PAT", False), ("GRCh38RC", True)])
def test_nuc_groups_survive_lift(lifted, target, flipped):
    out, source, cal = lifted
    records, header = out[target]
    by = {(r.query_name, r.is_supplementary): r for r in records}
    assert set(by) == {("fwd", False), ("rev", False), ("chim", False), ("chim", True)}
    co = header.to_dict().get("CO", [])
    contracts = {json.loads(c[len(SCAN_MA_CONTRACT_PREFIX):])["group"] for c in co
                 if c.startswith(SCAN_MA_CONTRACT_PREFIX)}
    assert contracts == set(NUC_GROUPS)
    assert set(NUC_GROUPS) <= set(declared_ma_types(header))
    for key in (("fwd", False), ("rev", False), ("chim", False)):
        lifted_read, src_read = by[key], source[key]
        assert by[("fwd", False)].is_reverse == flipped
        g_l, g_s = _groups(lifted_read), _groups(src_read)
        assert set(g_l) == set(NUC_GROUPS)
        assert g_l == g_s, key                              # identical, molecular frame
        assert not any(v[3] for v in g_l.values())
        # re-running the module on the lifted record reproduces the bytes
        mol, why = decode_record(lifted_read, DecodeContext(header))
        assert mol is not None, why
        res = molecule_outputs(mol, cal=cal)
        from fiberhmm.nuc.tags import group_rows
        rows = group_rows(res)
        assert [(s, n, list(q)) for s, n, q, _a in rows[NUCSZ][1]] == \
            [(s, n, list(q)) for (s, n), q in zip(*g_l[NUCSZ][:2])]
    # the filled supplementary: its partial groups moved with nuc/msp by the clip offset
    supp_l, supp_s = by[("chim", True)], source[("chim", True)]
    g_l, g_s = _groups(supp_l), _groups(supp_s)
    assert set(g_l) == set(NUC_GROUPS) and not any(v[3] for v in g_l.values())
    assert g_l[NUCSZ][2] == g_s[NUCSZ][2]                  # digest unchanged
    offset = 4000
    for grp in (NUCSZ, NUCSEG):
        assert [(s + offset, n) for s, n in g_s[grp][0]] == g_l[grp][0]
        assert g_l[grp][1] == g_s[grp][1]
    assert int(supp_l.get_tag("MA").split(";")[0]) == 6500
