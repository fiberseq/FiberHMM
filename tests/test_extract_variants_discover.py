"""fiberhmm-extract --variants discover: callers, cache, regions, outputs.

Synthetic reads on a random reference with planted SNVs, in a Fiber-seq
setting (pileup caller) and a DAF setting (deamination on one strand per
fiber, R/Y encoded; the DAF SNP screen).
"""
from __future__ import annotations

import gzip
from array import array
import json
import os
import random
import shutil
import subprocess
import sys

import pysam
import pytest

from fiberhmm.cli import extract_tags
from fiberhmm.io import variant_discovery as vd
from fiberhmm.io.variant_discovery import (
    DiscoveryOptions,
    PileupParams,
    VariantDiscoveryError,
    discover_variants,
)

CHROM = "chr1"
LENGTH = 24_000


def _reference(seed=11):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(LENGTH))


def _set(seq, pos, base):
    return seq[:pos] + base + seq[pos + 1:]


def _header(platform="pacbio", daf=False, rgs=None):
    hd = {"HD": {"VN": "1.6", "SO": "coordinate"},
          "SQ": [{"SN": CHROM, "LN": LENGTH}]}
    rg_list = []
    for rg_id, sm in (rgs or [("rg1", "s1")]):
        rg = {"ID": rg_id, "SM": sm}
        if platform == "pacbio":
            rg["PL"] = "PACBIO"
        elif platform == "nanopore":
            rg["PL"] = "ONT"
        rg_list.append(rg)
    hd["RG"] = rg_list
    if daf:
        hd["PG"] = [{"ID": "fiberhmm-daf-encode", "PN": "fiberhmm-daf-encode",
                     "VN": "test"}]
    return pysam.AlignmentHeader.from_dict(hd)


def _md(read_seq, ref_seq):
    out, run = [], 0
    for q, r in zip(read_seq, ref_seq):
        if q == r:
            run += 1
        else:
            out.append(f"{run}{r}")
            run = 0
    out.append(str(run))
    return "".join(out)


def _eqx(read_seq, ref_seq):
    ops = []
    for q, r in zip(read_seq, ref_seq):
        op = 7 if q == r else 8
        if ops and ops[-1][0] == op:
            ops[-1][1] += 1
        else:
            ops.append([op, 1])
    return [tuple(o) for o in ops]


def simulate(ref, snvs, *, n_reads=500, read_len=2000, seed=3, daf=False,
             conv_rate=0.6, error_rate=0.002, forward_only_alt=(), m6a_sites=(),
             span=None, rg_ids=("rg1",)):
    """Read records: (name, start, seq, reverse, rg, mm_positions).

    ``snvs``: {pos: (alt, allele_fraction)}. ``forward_only_alt``: SNV
    positions whose ALT is only on forward reads (a strand artefact).
    ``m6a_sites``: reference positions whose REF base carries an m6A call on
    forward reads.
    """
    rng = random.Random(seed)
    lo, hi = span or (0, LENGTH)
    out = []
    for i in range(n_reads):
        start = rng.randrange(lo, max(lo + 1, hi - read_len))
        reverse = rng.random() < 0.5
        seq = list(ref[start:start + read_len])
        for pos, (alt, af) in snvs.items():
            if start <= pos < start + read_len and rng.random() < af:
                if pos in forward_only_alt and reverse:
                    continue
                seq[pos - start] = alt
        for k in range(len(seq)):
            if rng.random() < error_rate:
                seq[k] = rng.choice([b for b in "ACGT" if b != seq[k]])
        if daf:
            direction = "CT" if rng.random() < 0.5 else "GA"
            for k, b in enumerate(seq):
                if direction == "CT" and b == "C" and rng.random() < conv_rate:
                    seq[k] = "Y"
                elif direction == "GA" and b == "G" and rng.random() < conv_rate:
                    seq[k] = "R"
        mods = []
        if not reverse:
            for p in m6a_sites:
                if start <= p < start + read_len and seq[p - start] == "A":
                    mods.append(p - start)
        out.append((f"read{seed}_{i:05d}", start, "".join(seq), reverse,
                    rg_ids[i % len(rg_ids)], mods))
    return out


def write_bam(path, ref, records, *, header=None, cigar="M", md=True, qual=30):
    header = header or _header()
    records = sorted(records, key=lambda r: r[1])
    unsorted = str(path) + ".unsorted.bam"
    with pysam.AlignmentFile(unsorted, "wb", header=header) as out:
        for name, start, seq, reverse, rg, mods in records:
            a = pysam.AlignedSegment(header)
            a.query_name = name
            a.query_sequence = seq
            a.flag = 16 if reverse else 0
            a.reference_id = 0
            a.reference_start = start
            a.mapping_quality = 60
            refseg = ref[start:start + len(seq)]
            a.cigartuples = _eqx(seq, refseg) if cigar == "eqx" else [(0, len(seq))]
            a.query_qualities = pysam.qualitystring_to_array(chr(qual + 33) * len(seq))
            if md:
                a.set_tag("MD", _md(seq, refseg))
            a.set_tag("RG", rg)
            if mods:
                a_positions = [k for k, b in enumerate(seq) if b == "A"]
                rank = {q: n for n, q in enumerate(a_positions)}
                ranks = sorted(rank[q] for q in mods)
                skips, prev = [], -1
                for r in ranks:
                    skips.append(r - prev - 1)
                    prev = r
                a.set_tag("MM", "A+a." + "".join(f",{s}" for s in skips) + ";")
                a.set_tag("ML", array("B", [255] * len(ranks)))
            out.write(a)
    pysam.sort("-o", str(path), unsorted)
    os.remove(unsorted)
    pysam.index(str(path))
    return str(path)


def write_fasta(path, ref):
    with open(path, "w") as handle:
        handle.write(f">{CHROM}\n")
        for i in range(0, len(ref), 60):
            handle.write(ref[i:i + 60] + "\n")
    pysam.faidx(str(path))
    return str(path)


def vcf_records(path, pass_only=True):
    out = {}
    with pysam.VariantFile(path) as vcf:
        for rec in vcf:
            filters = list(rec.filter.keys())
            if pass_only and filters not in ([], ["PASS"]):
                continue
            out[rec.start] = (rec.ref, rec.alts[0], filters, dict(rec.info))
    return out


def _quiet(_msg):
    pass


def _discover(bam, tmp_path, **kwargs):
    windows = kwargs.pop("windows", None)
    opts = DiscoveryOptions(cache_dir=str(tmp_path / "cache"), **kwargs)
    return discover_variants(bam, opts, windows=windows, log=_quiet)


# ---------------------------------------------------------------------------
# planted SNVs
# ---------------------------------------------------------------------------

def _planted(ref):
    """Het and hom SNVs of every substitution class, away from the ends."""
    rng = random.Random(5)
    sites = {}
    for k, pos in enumerate(range(3000, 21000, 900)):
        r = ref[pos]
        alt = rng.choice([b for b in "ACGT" if b != r])
        sites[pos] = (alt, 1.0 if k % 4 == 0 else 0.5)
    return sites


@pytest.fixture(scope="module")
def fiberseq(tmp_path_factory):
    d = tmp_path_factory.mktemp("fs")
    ref = _reference()
    snvs = _planted(ref)
    bam = write_bam(d / "fs.bam", ref, simulate(ref, snvs))
    return ref, snvs, bam, d


def test_pileup_finds_planted_snvs(fiberseq, tmp_path):
    ref, snvs, bam, _ = fiberseq
    result = _discover(bam, tmp_path)
    assert result.caller == "pileup" and not result.reused
    calls = vcf_records(result.vcf)
    planted = {p: (ref[p], a) for p, (a, _af) in snvs.items()}
    found = {p: (r, a) for p, (r, a, _f, _i) in calls.items()}
    assert found == planted  # recall 1, precision 1
    with pysam.VariantFile(result.vcf) as vcf:
        sample = list(vcf.header.samples)[0]
        gts = {rec.start: rec.samples[sample]["GT"] for rec in vcf}
    for p, (_a, af) in snvs.items():
        assert gts[p] == ((1, 1) if af == 1.0 else (0, 1))


def test_pileup_reference_from_eqx_cigar_or_fasta(tmp_path):
    ref = _reference()
    snvs = _planted(ref)
    reads = simulate(ref, snvs, seed=4)
    eqx = write_bam(tmp_path / "eqx.bam", ref, reads, cigar="eqx", md=False)
    plain = write_bam(tmp_path / "plain.bam", ref, reads, md=False)
    fasta = write_fasta(tmp_path / "ref.fa", ref)
    want = {p for p, (_a, af) in snvs.items() if af < 1.0}
    # =/X only: a homozygous site has no '=' read, so its REF is unknown.
    got = set(vcf_records(_discover(eqx, tmp_path / "a").vcf))
    assert want <= got <= set(snvs)
    got = set(vcf_records(_discover(plain, tmp_path / "b", reference=fasta).vcf))
    assert got == set(snvs)
    with pytest.raises(VariantDiscoveryError, match="--reference"):
        _discover(plain, tmp_path / "c")


def test_strand_bias_homopolymer_and_indel_filters(tmp_path):
    ref = _reference(seed=21)
    # A run of 6 A's ending right before an SNV site.
    ref = ref[:8000] + "AAAAAA" + ref[8006:]
    site_hp = 8006
    ref = _set(ref, site_hp, "C")
    snvs = {5000: ("A" if ref[5000] != "A" else "C", 0.9),   # strand artefact
            site_hp: ("G", 0.5),
            12000: ("A" if ref[12000] != "A" else "C", 0.5)}  # clean
    reads = simulate(ref, snvs, seed=8, forward_only_alt=(5000,))
    pac = write_bam(tmp_path / "pac.bam", ref, reads, header=_header("pacbio"))
    ont = write_bam(tmp_path / "ont.bam", ref, reads, header=_header("nanopore"))
    allrec = vcf_records(_discover(pac, tmp_path / "p").vcf, pass_only=False)
    assert "StrandBias" in allrec[5000][2]
    assert allrec[site_hp][2] in ([], ["PASS"])          # PacBio: no homopolymer filter
    assert allrec[12000][2] in ([], ["PASS"])
    allrec = vcf_records(_discover(ont, tmp_path / "o").vcf, pass_only=False)
    assert "Homopolymer" in allrec[site_hp][2]          # Nanopore: filtered
    assert allrec[12000][2] in ([], ["PASS"])
    off = _discover(ont, tmp_path / "off", pileup=PileupParams(homopolymer="off"))
    assert vcf_records(off.vcf, pass_only=False)[site_hp][2] in ([], ["PASS"])


def test_mod_artifact_filter(tmp_path):
    ref = _reference(seed=31)
    site = 9000
    ref = _set(ref, site, "A")
    snvs = {site: ("G", 0.3)}
    reads = simulate(ref, snvs, seed=9, m6a_sites=(site,), n_reads=700)
    bam = write_bam(tmp_path / "m.bam", ref, reads)
    rec = vcf_records(_discover(bam, tmp_path / "on").vcf, pass_only=False)[site]
    assert "ModArtifact" in rec[2] and rec[3]["MR"] >= 0.3
    rec = vcf_records(_discover(bam, tmp_path / "off",
                                pileup=PileupParams(mod_filter=False)).vcf,
                      pass_only=False)[site]
    assert rec[2] in ([], ["PASS"])


# ---------------------------------------------------------------------------
# DAF
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def daf(tmp_path_factory):
    d = tmp_path_factory.mktemp("daf")
    ref = _reference(seed=41)
    # Non-conversion SNVs plus a C>T and a G>A SNP (both mimic deamination).
    snvs = {}
    for pos, alt in ((4000, None), (7000, None), (10000, None), (13000, "T"), (16000, "A")):
        if alt is None:
            r = ref[pos]
            alt = {"A": "G", "C": "A", "G": "T", "T": "C"}[r]
        else:
            ref = _set(ref, pos, "C" if alt == "T" else "G")
        snvs[pos] = (alt, 0.5)
    # An A>G (T>C) SNV's ALT G (C) deaminates too on G->A (C->T) fibers, so
    # its ALT fraction there is (1 - conversion rate) x AF; 0.3 keeps it
    # above the screen's 0.2 at AF 0.5.
    reads = simulate(ref, snvs, seed=12, daf=True, conv_rate=0.3, n_reads=700)
    bam = write_bam(d / "daf.bam", ref, reads, header=_header("pacbio", daf=True))
    return ref, snvs, bam, d


def test_daf_screen_calls_snvs_not_conversions(daf, tmp_path):
    ref, snvs, bam, _ = daf
    result = _discover(bam, tmp_path / "auto")
    assert result.caller == "daf"
    found = {p: (r, a) for p, (r, a, _f, _i) in vcf_records(result.vcf).items()}
    assert found == {p: (ref[p], a) for p, (a, _af) in snvs.items()}
    # The pileup screen on the same reads: R/Y bases are not counted, so it
    # sees no conversions, but it cannot use the deaminated bases either.
    forced = _discover(bam, tmp_path / "pileup", caller="pileup")
    assert forced.caller == "pileup"


def test_daf_conversions_as_plain_bases_fool_pileup_not_daf(tmp_path):
    """Deamination written as plain T/A (MD-encoded DAF): the pileup screen
    calls conversion sites, the DAF screen does not."""
    ref = _reference(seed=51)
    snvs = {6000: ({"A": "G", "C": "A", "G": "T", "T": "C"}[ref[6000]], 0.5)}
    reads = simulate(ref, snvs, seed=13, daf=True, conv_rate=0.7, n_reads=600)
    plain = [(n, s, q.replace("Y", "T").replace("R", "A"), rv, rg, m)
             for n, s, q, rv, rg, m in reads]
    bam = write_bam(tmp_path / "md.bam", ref, plain, header=_header("pacbio", daf=True))
    daf_calls = vcf_records(_discover(bam, tmp_path / "d").vcf)
    assert set(daf_calls) == {6000}
    pile = vcf_records(_discover(bam, tmp_path / "p", caller="pileup").vcf)
    conv = [p for p, (r, a, _f, _i) in pile.items() if (r, a) in (("C", "T"), ("G", "A"))]
    assert conv, "the pileup screen should be fooled by deamination"


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

def test_cache_hit_miss_and_invalidation(fiberseq, tmp_path):
    ref, snvs, bam0, _ = fiberseq
    bam = str(tmp_path / "c.bam")
    shutil.copy(bam0, bam)
    shutil.copy(bam0 + ".bai", bam + ".bai")
    first = _discover(bam, tmp_path)
    again = _discover(bam, tmp_path)
    assert again.reused and again.key == first.key and again.vcf == first.vcf
    # parameters, regions and caller change the key
    other = _discover(bam, tmp_path, pileup=PileupParams(min_depth=9))
    assert not other.reused and other.key != first.key
    region = _discover(bam, tmp_path, windows=[(CHROM, 0, 12000)])
    assert region.key != first.key
    assert set(vcf_records(region.vcf)) == {p for p in snvs if p < 12000}
    # a changed input (mtime) invalidates
    st = os.stat(bam)
    os.utime(bam, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))
    touched = _discover(bam, tmp_path)
    assert not touched.reused and touched.key != first.key
    # a broken cache entry (index gone) is rebuilt
    os.remove(touched.vcf + ".tbi")
    rebuilt = _discover(bam, tmp_path)
    assert not rebuilt.reused and rebuilt.key == touched.key
    meta = json.load(open(rebuilt.sidecar))
    assert meta["cache_key"] == rebuilt.key and meta["caller"] == "pileup"


def test_cores_and_chunks_do_not_change_the_vcf(fiberseq, tmp_path):
    _ref, _snvs, bam, _ = fiberseq
    one = _discover(bam, tmp_path / "one", chunk_bp=5000, cores=1)
    many = _discover(bam, tmp_path / "many", chunk_bp=5000, cores=3)
    assert one.key == many.key

    def body(path):
        with gzip.open(path, "rt") as handle:
            return [line for line in handle if not line.startswith("##")]
    assert body(one.vcf) == body(many.vcf)
    whole = _discover(bam, tmp_path / "whole", chunk_bp=100_000)
    assert body(whole.vcf) == body(one.vcf)


def test_interrupted_run_resumes(fiberseq, tmp_path):
    _ref, _snvs, bam, _ = fiberseq
    opts = DiscoveryOptions(cache_dir=str(tmp_path), chunk_bp=5000)
    bams = vd.pooled_inputs(bam, opts)
    windows = vd.merge_windows([(CHROM, 0, LENGTH)], [CHROM])
    material = vd.cache_material(bams, windows, "pileup", opts, None, vd.detect_platform(bam))
    material["sample"] = "fs"      # as discover_variants completes the key
    material["circular"] = []
    key = vd.cache_key(material)
    _vcf, _side, parts = vd.cache_paths(str(tmp_path), "fs", key)
    os.makedirs(parts)
    chunk = vd.discovery_chunks(windows, 5000)[0]
    vd._run_chunk((chunk, "pileup", tuple(bams), opts, None, False, parts, None))
    lines = []
    result = discover_variants(bam, opts, log=lines.append)
    assert result.key == key
    assert any("resuming: 1/" in line for line in lines)
    assert not os.path.exists(parts)


# ---------------------------------------------------------------------------
# pooling, read groups
# ---------------------------------------------------------------------------

def test_pool_equals_combined_bam_and_is_order_free(tmp_path):
    ref = _reference(seed=61)
    snvs = _planted(ref)
    reads = simulate(ref, snvs, seed=14, n_reads=600)
    combined = write_bam(tmp_path / "all.bam", ref, reads)
    a = write_bam(tmp_path / "a.bam", ref, reads[::2])
    b = write_bam(tmp_path / "b.bam", ref, reads[1::2])
    pooled = _discover(a, tmp_path / "p1", pool=(b,), sample_name="S")
    whole = _discover(combined, tmp_path / "w", sample_name="S")
    assert vcf_records(pooled.vcf) == vcf_records(whole.vcf)
    flipped = _discover(b, tmp_path / "p1", pool=(a,), sample_name="S")
    assert flipped.reused and flipped.key == pooled.key
    alone = _discover(a, tmp_path / "p2", pool=(b,), per_bam=True)
    assert alone.key != pooled.key


def test_read_group_restriction(tmp_path):
    ref = _reference(seed=71)
    s1 = {5000: ({"A": "G", "C": "A", "G": "T", "T": "C"}[ref[5000]], 0.5)}
    s2 = {9000: ({"A": "G", "C": "A", "G": "T", "T": "C"}[ref[9000]], 0.5)}
    r1 = simulate(ref, s1, seed=15, rg_ids=("rgA",))
    r2 = simulate(ref, s2, seed=16, rg_ids=("rgB",))
    hdr = _header(rgs=[("rgA", "alpha"), ("rgB", "beta")])
    bam = write_bam(tmp_path / "rg.bam", ref, r1 + r2, header=hdr)
    assert set(vcf_records(_discover(bam, tmp_path / "a", rg_sample="alpha").vcf)) == {5000}
    assert set(vcf_records(_discover(bam, tmp_path / "b", read_groups=("rgB",)).vcf)) == {9000}
    with pytest.raises(VariantDiscoveryError, match="SM:gamma"):
        _discover(bam, tmp_path / "c", rg_sample="gamma")


def test_pool_must_share_contigs(fiberseq, tmp_path):
    _ref, _snvs, bam, _ = fiberseq
    hdr = pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6", "SO": "coordinate"},
                                           "SQ": [{"SN": "other", "LN": 100}]})
    odd = str(tmp_path / "odd.bam")
    with pysam.AlignmentFile(odd, "wb", header=hdr):
        pass
    pysam.index(odd)
    with pytest.raises(VariantDiscoveryError, match="share a reference"):
        _discover(bam, tmp_path, pool=(odd,))


# ---------------------------------------------------------------------------
# fiberhmm-extract CLI
# ---------------------------------------------------------------------------

def _cli(args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fiberhmm-extract", *args])
    extract_tags.main()


def _bed_rows(path):
    with open(path) as handle:
        return [line.rstrip("\n").split("\t") for line in handle]


def test_cli_discover_then_vcf_path_gives_identical_beds(fiberseq, tmp_path, monkeypatch, capsys):
    _ref, snvs, bam, _ = fiberseq
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    _cli(["-i", bam, "-o", str(out1), "--variants", "discover", "--sv", "--bed-only",
          "--sv-min-len", "1000"], monkeypatch)
    text = capsys.readouterr().out
    assert "caller=pileup" in text and "rebuilt" in text
    [vcf] = [p for p in os.listdir(out1) if p.endswith(vd.VCF_SUFFIX)]
    assert os.path.exists(out1 / (vcf + ".tbi"))
    side = json.load(open(out1 / "fs_var.discovery.json"))
    assert side["caller"] == "pileup" and side["vcf"] == vcf and not side["reused"]
    assert side["cache_key"].startswith(vcf.split(".")[1])
    rows1 = _bed_rows(out1 / "fs_var.bed")
    assert rows1
    # rerun: reused
    _cli(["-i", bam, "-o", str(out1), "--variants", "discover", "--bed-only"], monkeypatch)
    assert "reused" in capsys.readouterr().out
    # the VCF path with the discovered VCF
    os.makedirs(out2)
    _cli(["-i", bam, "-o", str(out2), "--variants", str(out1 / vcf), "--bed-only"], monkeypatch)
    assert _bed_rows(out2 / "fs_var.bed") == rows1
    sites = {int(r[1]) + int(s) for r in rows1 for s in r[11].rstrip(",").split(",")}
    assert sites == set(snvs)


@pytest.mark.skipif(shutil.which("bedToBigBed") is None, reason="needs bedToBigBed")
def test_cli_bigbeds_from_discover_equal_vcf_path(fiberseq, tmp_path, monkeypatch):
    _ref, _snvs, bam, _ = fiberseq
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    _cli(["-i", bam, "-o", str(out1), "--variants", "discover", "--sv"], monkeypatch)
    [vcf] = [p for p in os.listdir(out1) if p.endswith(vd.VCF_SUFFIX)]
    _cli(["-i", bam, "-o", str(out2), "--variants", str(out1 / vcf)], monkeypatch)
    assert (out1 / "fs_var.bb").read_bytes() == (out2 / "fs_var.bb").read_bytes()


def test_cli_variant_cache_elsewhere_is_copied_beside_outputs(fiberseq, tmp_path, monkeypatch):
    _ref, _snvs, bam, _ = fiberseq
    cache, out = tmp_path / "cache", tmp_path / "out"
    _cli(["-i", bam, "-o", str(out), "--variants", "discover", "--variant-cache", str(cache),
          "--bed-only"], monkeypatch)
    [vcf] = [p for p in os.listdir(cache) if p.endswith(vd.VCF_SUFFIX)]
    assert os.path.exists(out / vcf) and os.path.exists(out / (vcf + ".tbi"))


def test_cli_region_limits_extraction_and_discovery(fiberseq, tmp_path, monkeypatch):
    _ref, snvs, bam, _ = fiberseq
    out = tmp_path / "r"
    _cli(["-i", bam, "-o", str(out), "--variants", "discover", "--bed-only",
          "--region", f"{CHROM}:8001-12000"], monkeypatch)
    [vcf] = [p for p in os.listdir(out) if p.endswith(vd.VCF_SUFFIX)]
    assert set(vcf_records(str(out / vcf))) == {p for p in snvs if 8000 <= p < 12000}
    names = [r[3] for r in _bed_rows(out / "fs_var.bed")]
    with pysam.AlignmentFile(bam) as handle:
        overlapping = {r.query_name for r in handle.fetch(CHROM, 8000, 12000)}
    assert set(names) <= overlapping
    assert len(names) == len(set(names))  # each read once


def test_cli_discover_error_exits_2(tmp_path, monkeypatch):
    ref = _reference()
    bam = write_bam(tmp_path / "x.bam", ref, simulate(ref, {}, n_reads=50), md=False)
    with pytest.raises(SystemExit) as exc:
        _cli(["-i", bam, "-o", str(tmp_path / "o"), "--variants", "discover",
              "--variant-caller", "pileup"], monkeypatch)
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# region planning (extract)
# ---------------------------------------------------------------------------

def test_region_parsing_and_bed(tmp_path):
    lengths = {"chr1": 1000, "chr2": 500}
    assert extract_tags.parse_region_arg("chr1", lengths) == ("chr1", 0, 1000)
    assert extract_tags.parse_region_arg("chr1:11-1,000", lengths) == ("chr1", 10, 1000)
    with pytest.raises(ValueError):
        extract_tags.parse_region_arg("chr9:1-5", lengths)
    bed = tmp_path / "r.bed"
    bed.write_text("#h\nchr2\t10\t20\nchr1\t0\t5000\n")
    assert extract_tags.read_regions_bed(str(bed), lengths) == [("chr2", 10, 20), ("chr1", 0, 1000)]


def test_region_extraction_writes_overlapping_reads_once(fiberseq, tmp_path):
    _ref, _snvs, bam, _ = fiberseq
    beds = {"sv": str(tmp_path / "w.bed")}
    # Every read has a 0-length event? Use the var layer through discover.
    windows = [(CHROM, 5000, 9000), (CHROM, 8500, 11000)]
    extract_tags.extract_tags_parallel(
        bam, {"var": str(tmp_path / "v.bed")}, ["var"], n_cores=2, region_size=1500,
        variants="discover", regions=windows,
        discovery=DiscoveryOptions(cache_dir=str(tmp_path / "c")))
    names = [r[3] for r in _bed_rows(tmp_path / "v.bed")]
    assert len(names) == len(set(names))
    with pysam.AlignmentFile(bam) as handle:
        want = set()
        for read in handle.fetch(CHROM, 5000, 11000):
            want.add(read.query_name)
    assert set(names) <= want
    del beds


# ---------------------------------------------------------------------------
# FiberBrowser contract
# ---------------------------------------------------------------------------

# FiberBrowser (rc/fb-311, browser/services/fiberhmm_variant_layers.py) reads
# the var layer by autoSQL table name and field names, and finds the source
# VCF from the description's ``VCF: name=...`` marker in the bigBed's folder.
FB_VCF_NOTE_RE = r"VCF:\s*name=([^;\s]+)"
FB_VCF_SAMPLE_RE = r"VCF:[^\"]*?sample=([^;\s]+?)\.?(?:\s|$)"


def test_var_schema_contract_for_fiberbrowser(fiberseq, tmp_path):
    import re

    from fiberhmm.io.autosql import get_schema, vcf_source_note
    _ref, _snvs, bam, _ = fiberseq
    result = _discover(bam, tmp_path)
    name = os.path.basename(result.vcf)
    note = vcf_source_note(result.vcf, "0" * 32, None)
    schema = get_schema("var", sample_name="fs", source_note=note)
    assert schema.startswith("table fiberhmm_var")
    fields = re.findall(r"^\s+\S+\s+(\w+);", schema, re.M)
    assert fields[12:15] == ["blockAllele", "blockBq", "isDuplicate"]
    assert re.search(FB_VCF_NOTE_RE, schema).group(1) == name
    assert re.search(FB_VCF_SAMPLE_RE, schema).group(1) == "*"
    assert name == os.path.basename(name) and "/" not in name


FB_REPO = os.environ.get("FIBERBROWSER_REPO")


@pytest.mark.skipif(not FB_REPO or shutil.which("bedToBigBed") is None,
                    reason="set FIBERBROWSER_REPO to a FiberBrowser checkout (needs bedToBigBed)")
def test_fiberbrowser_reader_accepts_discovered_var_bigbed(fiberseq, tmp_path, monkeypatch):
    """Run FiberBrowser's own reader on the bigBed in a subprocess (read-only,
    no bytecode written into the FiberBrowser checkout)."""
    _ref, snvs, bam, _ = fiberseq
    out = tmp_path / "o"
    _cli(["-i", bam, "-o", str(out), "--variants", "discover", "--sv",
          "--sv-min-len", "1000"], monkeypatch)
    script = r'''
import json, sys
from types import SimpleNamespace as NS
from services import fiberhmm_variant_layers as fvl
from services.bigbed_detection import detect_bigbed_autosql_type
var, chrom, end = sys.argv[1], sys.argv[2], int(sys.argv[3])
ds = NS(id="d", data_type="bigbed", companion_layers={"var": [var]}, layer_type="footprint",
        paths=[], lift=None)
rows = fvl.fetch_rows(var, fvl.VAR_LAYER, chrom, 0, end)
alleles = fvl.vcf_alleles(ds, chrom, 0, end)
print(json.dumps({"type": detect_bigbed_autosql_type(var), "n_rows": len(rows),
                  "sites": sorted({s[0] for r in rows for s in r.sites}),
                  "vcf": fvl.source_vcf_name(var), "alleles": sorted(alleles)}))
'''
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.path.join(FB_REPO, "browser"))
    proc = subprocess.run([sys.executable, "-c", script, str(out / "fs_var.bb"), CHROM,
                           str(LENGTH)], capture_output=True, text=True, env=env,
                          cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got["type"] == "var"
    assert got["n_rows"] > 0
    assert set(got["sites"]) == set(snvs)
    assert got["vcf"].endswith(vd.VCF_SUFFIX)
    assert set(got["alleles"]) == set(snvs)


# ---------------------------------------------------------------------------
# audit follow-ups
# ---------------------------------------------------------------------------

def test_read_group_ids_are_per_bam(tmp_path):
    """The same RG ID in two pooled BAMs can belong to different samples."""
    ref = _reference(seed=81)
    site = 7000
    alt = {"A": "G", "C": "A", "G": "T", "T": "C"}[ref[site]]
    a = write_bam(tmp_path / "a.bam", ref, simulate(ref, {}, seed=17, rg_ids=("rg1",)),
                  header=_header(rgs=[("rg1", "alpha")]))
    b = write_bam(tmp_path / "b.bam", ref,
                  simulate(ref, {site: (alt, 1.0)}, seed=18, rg_ids=("rg1",)),
                  header=_header(rgs=[("rg1", "beta")]))
    alpha = _discover(a, tmp_path / "x", pool=(b,), rg_sample="alpha")
    beta = _discover(a, tmp_path / "x", pool=(b,), rg_sample="beta")
    assert set(vcf_records(alpha.vcf)) == set()
    assert set(vcf_records(beta.vcf)) == {site}
    assert alpha.key != beta.key


def test_read_spanning_two_windows_is_written_once(fiberseq, tmp_path):
    _ref, _snvs, bam, _ = fiberseq
    out = tmp_path / "sv.bed"
    extract_tags.extract_tags_parallel(
        bam, {"sv": str(out)}, ["sv"], n_cores=2, region_size=700,
        sv_min_clip=0, sv_min_len=10**9,
        regions=[(CHROM, 5000, 5100), (CHROM, 5600, 5700), (CHROM, 9000, 9100)])
    names = [r[3] for r in _bed_rows(out)]
    assert len(names) == len(set(names))
    with pysam.AlignmentFile(bam) as handle:
        want = {r.query_name for w in ((5000, 5100), (5600, 5700), (9000, 9100))
                for r in handle.fetch(CHROM, *w)}
    # sv_min_clip=0 writes a clip event at each end of every read.
    assert set(names) == want


def test_region_size_must_be_positive(fiberseq):
    _ref, _snvs, bam, _ = fiberseq
    with pytest.raises(extract_tags.RegionPlanError):
        extract_tags.plan_extract_regions(bam, 0, False, None, [(CHROM, 0, 100)])


def test_circular_contig_reads_through_the_origin(tmp_path):
    rng = random.Random(91)
    length = 3000
    ref = "".join(rng.choice("ACGT") for _ in range(length))
    site = 20
    alt = {"A": "G", "C": "A", "G": "T", "T": "C"}[ref[site]]
    doubled = ref + ref
    hdr = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": CHROM, "LN": length, "TP": "circular"}],
        "RG": [{"ID": "rg1", "SM": "s1", "PL": "PACBIO"}]})
    records = []
    for i in range(40):
        start = rng.randrange(length - 800, length - 100)   # all run through the origin
        seq = list(doubled[start:start + 1000])
        if i % 2 == 0:
            seq[site + length - start] = alt
        records.append((f"c{i}", start, "".join(seq), i % 4 < 2, "rg1", []))
    unsorted = str(tmp_path / "c.unsorted.bam")
    with pysam.AlignmentFile(unsorted, "wb", header=hdr) as out:
        for name, start, seq, reverse, rg, _m in sorted(records, key=lambda r: r[1]):
            a = pysam.AlignedSegment(hdr)
            a.query_name, a.query_sequence, a.flag = name, seq, 16 if reverse else 0
            a.reference_id, a.reference_start, a.mapping_quality = 0, start, 60
            a.cigartuples = [(0, len(seq))]
            a.query_qualities = pysam.qualitystring_to_array("?" * len(seq))
            a.set_tag("MD", _md(seq, doubled[start:start + len(seq)]))
            a.set_tag("RG", rg)
            out.write(a)
    bam = str(tmp_path / "c.bam")
    pysam.sort("-o", bam, unsorted)
    pysam.index(bam)
    got = vcf_records(_discover(bam, tmp_path / "p", chunk_bp=1000).vcf)
    assert set(got) == {site} and got[site][1] == alt


def test_chunk_part_names_do_not_collide():
    assert vd._chunk_name(("chr.a", 0, 1000)) != vd._chunk_name(("chr_a", 0, 1000))


def test_reads_without_qualities_pass_no_quality_floor(tmp_path):
    ref = _reference(seed=95)
    snvs = {6000: ({"A": "G", "C": "A", "G": "T", "T": "C"}[ref[6000]], 0.5)}
    reads = simulate(ref, snvs, seed=19)
    bam = write_bam(tmp_path / "q.bam", ref, reads)
    noq = str(tmp_path / "noq.bam")
    with pysam.AlignmentFile(bam) as src, pysam.AlignmentFile(noq, "wb", template=src) as dst:
        for read in src:
            read.query_qualities = None
            dst.write(read)
    pysam.index(noq)
    assert vcf_records(_discover(noq, tmp_path / "a").vcf) == {}
    floor0 = _discover(noq, tmp_path / "b", pileup=PileupParams(min_baseq=0))
    assert set(vcf_records(floor0.vcf)) == {6000}


def test_pool_name_and_sample_name(tmp_path):
    ref = _reference(seed=97)
    reads = simulate(ref, {}, seed=20, n_reads=200)
    a = write_bam(tmp_path / "a.bam", ref, reads[::2])
    b = write_bam(tmp_path / "b.bam", ref, reads[1::2])
    first = _discover(b, tmp_path / "c", pool=(a,))
    assert os.path.basename(first.vcf).startswith("a_pool.")
    assert _discover(a, tmp_path / "c", pool=(b,)).reused
    named = _discover(a, tmp_path / "d", sample_name="HG002.1")
    with pysam.VariantFile(named.vcf) as vcf:
        assert list(vcf.header.samples) == ["HG002.1"]


def test_homopolymer_run_wraps_on_a_whole_circular_contig():
    import numpy as np
    ref = np.array([0, 0, 0, 1, 2, 3, 1, 0, 0, 0], dtype=np.int8)   # AAA ... AAA
    assert vd._run_length_at(ref, 0) == 3
    assert vd._run_length_at(ref, 0, wrap=True) == 6


def test_pool_must_agree_on_circular_contigs(tmp_path):
    def empty(path, circular):
        sq = {"SN": CHROM, "LN": LENGTH}
        if circular:
            sq["TP"] = "circular"
        hdr = pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6", "SO": "coordinate"},
                                               "SQ": [sq]})
        with pysam.AlignmentFile(str(path), "wb", header=hdr):
            pass
        pysam.index(str(path))
        return str(path)
    a, b = empty(tmp_path / "a.bam", False), empty(tmp_path / "b.bam", True)
    with pytest.raises(VariantDiscoveryError, match="circular"):
        vd.check_pool_headers([a, b])


def test_sample_name_is_part_of_the_key(fiberseq, tmp_path):
    _ref, _snvs, bam, _ = fiberseq
    one = _discover(bam, tmp_path, sample_name="HG002.1")
    two = _discover(bam, tmp_path, sample_name="HG002_1")
    assert one.key != two.key and not two.reused


def test_truncated_part_is_recomputed(fiberseq, tmp_path):
    _ref, _snvs, bam, _ = fiberseq
    opts = DiscoveryOptions(cache_dir=str(tmp_path), chunk_bp=5000)
    parts = str(tmp_path / "parts")
    os.makedirs(parts)
    chunk = (CHROM, 5000, 10000)
    vd._run_chunk((chunk, "pileup", (bam,), opts, None, False, parts, None))
    assert vd._part_done(parts, chunk)
    path = vd._part_path(parts, chunk)
    lines = open(path).read().splitlines()
    assert len(lines) > 1
    open(path, "w").write("\n".join(lines[:-1]) + "\n")   # lose a record
    assert not vd._part_done(parts, chunk)


def test_key_lock_serialises_runs(tmp_path):
    import threading
    import time
    order = []

    def hold(tag):
        with vd._key_lock(str(tmp_path / "k.lock"), _quiet):
            order.append(f"{tag}+")
            time.sleep(0.2)
            order.append(f"{tag}-")
    threads = [threading.Thread(target=hold, args=(t,)) for t in "ab"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert order in (["a+", "a-", "b+", "b-"], ["b+", "b-", "a+", "a-"])


def test_daf_fasta_fallback_reads_through_a_circular_origin(tmp_path):
    from fiberhmm.daf.snps import _aligned_pairs
    rng = random.Random(99)
    length = 500
    ref = "".join(rng.choice("ACGT") for _ in range(length))
    fasta = tmp_path / "c.fa"
    fasta.write_text(f">{CHROM}\n{ref}\n")
    pysam.faidx(str(fasta))
    hdr = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6"}, "SQ": [{"SN": CHROM, "LN": length, "TP": "circular"}]})
    read = pysam.AlignedSegment(hdr)
    read.query_name, read.reference_id, read.reference_start = "r", 0, 450
    seq = (ref + ref)[450:550]
    read.query_sequence, read.cigartuples = seq, [(0, 100)]
    with pysam.FastaFile(str(fasta)) as handle:
        pairs = _aligned_pairs(read, handle)
    assert pairs is not None and len(pairs) == 100
    assert all(base == seq[q] for q, _r, base in pairs)


def _holds_inode(fd, inode):
    try:
        return os.fstat(fd).st_ino == inode
    except OSError:
        return False


def test_discovery_workers_do_not_inherit_the_key_lock(tmp_path):
    path = str(tmp_path / "k.lock")
    with vd._key_lock(path, _quiet) as fd:
        if fd is None:
            pytest.skip("no fcntl")
        with vd._worker_pool(1) as pool:
            assert pool.submit(_holds_inode, fd, os.stat(path).st_ino).result() is False
