"""DAF insert consensus: evidence inside insertions carried by many reads.

On C->T reads a column that is sometimes C and sometimes T is a C (deaminated
in some molecules), one that is always T is a T; G/A likewise on G->A reads,
and each strand reads the other strand's bases unconverted. Carriers are
re-encoded against the consensus, so the insert gets real evidence instead of
the unaligned-base mask.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys

import numpy as np
import pysam
import pytest

from fiberhmm.daf import insert_consensus as ic
from fiberhmm.inference import engine
from fiberhmm.inference.engine import (
    _extract_fiber_read_from_pysam,
    configure_daf_insert_evidence,
    extract_fiber_read_from_payload,
    make_apply_payload,
)
from fiberhmm.inference.tf_recaller import extract_modification_calls

REF_START = 1_000


def _rand(n, seed):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


REF = _rand(3_000, 31)
INSERT = _rand(400, 32)
OPEN = range(0, 200)          # insert offsets that are accessible


def _header():
    return pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"LN": 100_000, "SN": "chr1"}]})


def _md(cigar, seq, ref_start):
    parts, run, q, r = [], 0, 0, ref_start
    for op, n in cigar:
        if op == 0:
            for j in range(n):
                if seq[q + j] == REF[r + j]:
                    run += 1
                else:
                    parts += [str(run), REF[r + j]]
                    run = 0
            q += n
            r += n
        elif op in (1, 4):
            q += n
    parts.append(str(run))
    return "".join(parts)


def _carriers(n, p_open=0.8, seed=3, strands=("CT", "GA")):
    """``n`` reads REF[0:1000] + INSERT + REF[1000:2000] (CIGAR 1000M400I1000M);
    flanks: linker every 197 bp; insert: OPEN accessible, rest protected."""
    rng = random.Random(seed)
    mol = REF[0:1000] + INSERT + REF[1000:2000]
    access = np.zeros(len(mol), dtype=bool)
    for i in range(len(mol)):
        if i < 1000 or i >= 1400:
            access[i] = (i % 197) >= 147
    access[1000 + OPEN.start:1000 + OPEN.stop] = True
    reads = []
    for k in range(n):
        strand = strands[k % len(strands)]
        tgt, prod = ("C", "T") if strand == "CT" else ("G", "A")
        seq = "".join(prod if b == tgt and rng.random() < (p_open if access[i] else 0.01)
                      else b for i, b in enumerate(mol))
        cigar = [(0, 1000), (1, 400), (0, 1000)]
        a = pysam.AlignedSegment(_header())
        a.query_name = f"c{k}"
        a.query_sequence = seq
        a.flag = 16 if k % 4 >= 2 else 0
        a.reference_id = 0
        a.reference_start = REF_START
        a.mapping_quality = 60
        a.cigartuples = cigar
        a.set_tag("MD", _md(cigar, seq, 0))
        reads.append(a)
    return reads


@pytest.fixture(autouse=True)
def _reset():
    configure_daf_insert_evidence(None)
    yield
    configure_daf_insert_evidence(None)


def test_column_call_uses_variability_not_majority():
    # C deaminated in 85% of CT molecules: majority T, still a C
    ct = np.array([0, 15, 0, 85, 0])
    assert ic.call_column(ct, np.zeros(5))[0] == ic.C
    # always T on CT reads: a T
    assert ic.call_column(np.array([0, 0, 0, 100, 0]), np.zeros(5))[0] == ic.T
    # the other strand reads C unconverted
    assert ic.call_column(np.array([0, 2, 0, 48, 0]), np.array([0, 50, 0, 0, 0]))[0] == ic.C
    # a mixed column on the unconvertible strand (two alleles) is not confident
    base, q = ic.call_column(np.array([0, 10, 0, 40, 0]), np.array([0, 25, 0, 25, 0]))
    assert q < ic.MIN_QUALITY


@pytest.mark.parametrize("strands", [("CT", "GA"), ("CT",)])
def test_consensus_recovers_insert_at_high_deamination(strands):
    reads = _carriers(40, p_open=0.85, strands=strands)
    events = ic.collect_carriers(reads)
    clusters = ic.cluster_events(events)
    assert len(clusters) == 1 and len(clusters[0]["members"]) == 40
    cons = ic.build_consensus(clusters[0]["members"])
    confident = cons.quality >= ic.MIN_QUALITY
    assert len(cons.sequence) == len(INSERT)
    targets = np.array([b in "CG" for b in cons.sequence])
    # one strand cannot tell an always-T column from a C deaminated in ~95% of
    # molecules, so T columns are unconfident there; that costs nothing (a T
    # is a non-target either way) and C columns stay confident
    assert confident[targets].mean() > 0.85
    # Confident columns are right, including open C/G that most molecules
    # deaminated (a majority vote calls them T/A when one strand is present);
    # an open C that too few CT molecules left unconverted is not confident.
    wrong = [i for i, (a, b) in enumerate(zip(cons.sequence, INSERT)) if a != b]
    assert not any(confident[i] for i in wrong)
    if len(strands) == 2:
        assert cons.sequence == INSERT
    else:
        assert len(wrong) <= 3


def test_carriers_get_insert_evidence_in_every_path():
    reads = _carriers(30)
    evidence, report = ic.build_insert_evidence(reads, min_carriers=20)
    assert report[0]["used"] and len(evidence) == 30
    configure_daf_insert_evidence(evidence)
    read = reads[0]                                   # CT strand
    live = _extract_fiber_read_from_pysam(read, "daf", 128)
    slim = extract_fiber_read_from_payload(make_apply_payload(read, mode="daf"), "daf", 128)
    for fr in (live, slim):
        inside = {p for p in fr["m6a_query_positions"] if 1000 <= p < 1400}
        assert inside and all(read.query_sequence[p] == "T" for p in inside)
        assert all(INSERT[p - 1000] == "C" for p in inside)
        # deaminations in the open half only (protected half: 1% noise)
        assert sum(p < 1200 for p in inside) > 5 * max(1, sum(p >= 1200 for p in inside))
        assert "no_call_blocks" not in fr
        unknown = fr.get("unknown_query_positions", set())
        assert len(unknown & set(range(1000, 1400))) < 40
    mods, _strand, _seq, unknown = extract_modification_calls(read, "daf")
    assert {p for p in mods if 1000 <= p < 1400} == {
        p for p in live["m6a_query_positions"] if 1000 <= p < 1400}


def test_too_few_carriers_keep_the_mask():
    reads = _carriers(10)
    evidence, report = ic.build_insert_evidence(reads, min_carriers=20)
    assert not evidence and not report[0]["used"]
    configure_daf_insert_evidence(evidence)
    fr = _extract_fiber_read_from_pysam(reads[0], "daf", 128)
    assert set(range(1000, 1400)) <= fr["unknown_query_positions"]
    assert fr["no_call_blocks"] == [(1000, 1400)]


def test_mismatching_carrier_gets_no_evidence():
    reads = _carriers(30)
    other = _rand(400, 99)
    odd = reads[1]
    seq = odd.query_sequence
    odd.query_sequence = seq[:1000] + other + seq[1400:]
    odd.set_tag("MD", _md(odd.cigartuples, odd.query_sequence, 0))
    evidence, _report = ic.build_insert_evidence(reads, min_carriers=20)
    ev = evidence.get(ic.record_key(odd))
    assert ev is None or not ev.known


def test_fiberhmm_call_calls_inside_the_insert(tmp_path):
    reads = _carriers(30)
    bam = tmp_path / "in.bam"
    unsorted = tmp_path / "u.bam"
    with pysam.AlignmentFile(str(unsorted), "wb", header=_header()) as out:
        for r in reads:
            out.write(r)
    pysam.sort("-o", str(bam), str(unsorted))
    pysam.index(str(bam))

    def call(name, *extra):
        out = tmp_path / f"{name}.bam"
        subprocess.run([sys.executable, "-m", "fiberhmm.cli.call", "-i", str(bam),
                        "-o", str(out), "--enzyme", "dddb", "--seq", "nanopore",
                        "--no-dedup", "--no-daf-call-snps", "--no-qc",
                        "--min-read-length", "0", "-c", "1", *extra],
                       check=True, capture_output=True)
        return out

    from fiberhmm.io.ma_tags import flip_interval_frame, parse_ma_tag

    def inside_cover(path):
        open_access, packed_nuc = [], []
        for r in pysam.AlignmentFile(str(path)):
            ma = parse_ma_tag(r.get_tag("MA"))
            L = ma["read_length"]
            cov = {}
            for kind in ("nuc", "msp", "tf"):
                m = np.zeros(L, dtype=bool)
                for s, length in ma[kind]:
                    if r.is_reverse:
                        s, length = flip_interval_frame(s, length, L)
                    m[s:s + length] = True
                cov[kind] = m
            open_access.append((cov["msp"] | cov["tf"])[1000:1200].mean())
            packed_nuc.append(cov["nuc"][1200:1400].mean())
        return float(np.mean(open_access)), float(np.mean(packed_nuc))

    mask_only = inside_cover(call("mask", "--daf-insert-consensus", "off"))
    consensus_out = call("cons")
    with_consensus = inside_cover(consensus_out)
    assert mask_only == (0.0, 0.0)
    assert with_consensus[0] > 0.8 and with_consensus[1] > 0.8
    header = pysam.AlignmentFile(str(consensus_out)).header.to_dict()
    ds = [pg["DS"] for pg in header["PG"] if pg.get("PN") == "fiberhmm-call"][0]
    assert "daf_insert_consensus=on/1of1/min20" in ds
    report = json.loads((tmp_path / "qc" / "cons.insert_consensus.json").read_text())
    assert report["insertions"][0]["consensus"] == INSERT


def test_pcr_duplicates_do_not_count_as_carriers():
    reads = _carriers(25)
    for r in reads[1:]:
        r.flag |= 0x400
    evidence, report = ic.build_insert_evidence(reads, min_carriers=20)
    assert report[0]["carriers"] == 1 and report[0]["duplicate_records"] == 24
    assert not evidence
    reads = _carriers(30)
    for r in reads[25:]:
        r.flag |= 0x400
    evidence, report = ic.build_insert_evidence(reads, min_carriers=20)
    assert report[0]["used"] and len(evidence) == 30   # duplicates re-encoded


def test_consensus_replaces_marks_inside_the_insert():
    """An R/Y mark a producer put on an inserted base the consensus calls T
    does not survive; the consensus deaminations replace the insert's marks."""
    from fiberhmm.daf.encoder import encode_read_daf
    reads = _carriers(30)
    evidence, _ = ic.build_insert_evidence(reads, min_carriers=20)
    configure_daf_insert_evidence(evidence)
    read = reads[0]
    new_seq, st, _n = encode_read_daf(read)
    t_col = next(i for i in range(200, 400) if INSERT[i] == "T")
    seq = list(new_seq)
    seq[1000 + t_col] = "Y"
    read.query_sequence = "".join(seq)
    read.set_tag("st", st)
    fr = _extract_fiber_read_from_pysam(read, "daf", 128)
    assert 1000 + t_col not in fr["m6a_query_positions"]
    assert {p for p in fr["m6a_query_positions"] if 1000 <= p < 1400} == \
        evidence[ic.record_key(read)].mods


def test_insert_evidence_rescues_a_read_without_flank_deaminations():
    reads = _carriers(30)
    evidence, _ = ic.build_insert_evidence(reads, min_carriers=20)
    configure_daf_insert_evidence(evidence)
    read = reads[0]                                    # CT
    seq = list(read.query_sequence)
    for i in list(range(0, 1000)) + list(range(1400, 2400)):
        seq[i] = (REF[0:1000] + REF[1000:2000])[i if i < 1000 else i - 400]
    read.query_sequence = "".join(seq)
    read.set_tag("MD", _md(read.cigartuples, read.query_sequence, 0))
    for fr in (_extract_fiber_read_from_pysam(read, "daf", 128),
               extract_fiber_read_from_payload(make_apply_payload(read, mode="daf"),
                                               "daf", 128)):
        assert fr is not None and fr["_daf_strand"] == "+"
        assert fr["m6a_query_positions"] == evidence[ic.record_key(read)].mods
    mods, strand, _seq, _unknown = extract_modification_calls(read, "daf")
    assert strand == "+" and mods == evidence[ic.record_key(read)].mods


def _masked_ct_ga_carriers(tmp_path, n=40):
    """GA carriers whose flanks have five genuine G->A conversions and twenty
    C->T mismatches at SNP-masked sites: unmasked, C->T wins the vote."""
    reads = _carriers(n, strands=("GA",))
    flank = REF[0:1000] + REF[1000:2000]
    c_sites = [i for i in range(50, 950) if REF[i] == "C"][:20]
    g_sites = [i for i in range(1050, 1950) if REF[i] == "G"][:5]
    for read in reads:
        seq = list(read.query_sequence)
        for i in range(2400):
            if not 1000 <= i < 1400:
                seq[i] = flank[i if i < 1000 else i - 400]
        for r in c_sites:
            seq[r] = "T"
        for r in g_sites:
            seq[r + 400] = "A"
        read.query_sequence = "".join(seq)
        read.set_tag("MD", _md(read.cigartuples, read.query_sequence, 0))
    bed = tmp_path / "snps.bed"
    bed.write_text("".join(f"chr1\t{REF_START + r}\t{REF_START + r + 1}\n" for r in c_sites))
    return reads, str(bed)


def test_carrier_strand_follows_the_callers_snp_mask(tmp_path):
    """The pre-pass gives each carrier the strand the caller will call it on
    (Codex review): with the caller's SNP mask, masked C->T mismatches do not
    make a G->A carrier C->T, so its consensus evidence is used, not dropped
    while the inserted bases count as unmodified (protected) targets."""
    from fiberhmm.daf.snps import load_snp_mask
    reads, bed = _masked_ct_ga_carriers(tmp_path)
    unmasked, _ = ic.build_insert_evidence(reads, min_carriers=20)
    assert {ev.strand for ev in unmasked.values()} == {ic.STRAND_CT}
    evidence, _ = ic.build_insert_evidence(reads, min_carriers=20,
                                           snp_mask=load_snp_mask(bed))
    assert {ev.strand for ev in evidence.values()} == {ic.STRAND_GA}
    engine.configure_daf_snp_mask(bed)
    try:
        configure_daf_insert_evidence(evidence)
        read = reads[0]
        ev = evidence[ic.record_key(read)]
        for fr in (_extract_fiber_read_from_pysam(read, "daf", 128),
                   extract_fiber_read_from_payload(make_apply_payload(read, mode="daf"),
                                                   "daf", 128)):
            assert fr["_daf_strand"] == "-"
            inside = {p for p in fr["m6a_query_positions"] if 1000 <= p < 1400}
            assert inside and inside == ev.mods
    finally:
        engine.configure_daf_snp_mask(None)


def test_prepass_reads_the_snp_mask_path(tmp_path):
    reads, bed = _masked_ct_ga_carriers(tmp_path)
    path = tmp_path / "in.bam"
    with pysam.AlignmentFile(str(path), "wb", header=_header()) as out:
        for read in reads:
            out.write(read)
    summary = ic.run_insert_consensus_prepass(str(path), str(tmp_path / "ev"),
                                              snp_mask_path=bed)
    assert summary["clusters_used"] == 1
    assert summary["insertions"][0]["carriers_ga"] == len(reads)


def test_ry_strand_ignores_masked_and_inserted_marks():
    """Without an st tag, R/Y marks the caller drops (inserted bases here) do
    not decide the strand; the marks on aligned bases do."""
    mol = REF[0:1000] + INSERT + REF[1000:2000]
    seq = list(mol)
    g_aligned = [i for i in range(1000) if mol[i] == "G"][:10]
    c_inserted = [i for i in range(1000, 1400) if mol[i] == "C"][:100]
    for i in g_aligned:
        seq[i] = "R"
    for i in c_inserted:
        seq[i] = "Y"
    a = pysam.AlignedSegment(_header())
    a.query_name = "ry"
    a.query_sequence = "".join(seq)
    a.flag = 0
    a.reference_id = 0
    a.reference_start = REF_START
    a.mapping_quality = 60
    a.cigartuples = [(0, 1000), (1, 400), (0, 1000)]
    fr = _extract_fiber_read_from_pysam(a, "daf", 128)
    assert fr["_daf_strand"] == "-"
    _mods, strand, _seq, _unknown = extract_modification_calls(a, "daf")
    assert strand == "-"
    assert ic._read_strand(a) == ic.STRAND_GA


def test_prepass_reads_st_on_ry_input_only():
    """The callers honour st on R/Y input only; a raw read's strand comes from
    its reference comparison, so the pre-pass does the same."""
    read = _carriers(1, strands=("CT",))[0]
    read.set_tag("st", "GA")
    assert ic._read_strand(read) == ic.STRAND_CT
    fr = _extract_fiber_read_from_pysam(read, "daf", 128)
    assert fr["_daf_strand"] == "+"


def test_prepass_uses_the_reference_when_mm_has_no_ml():
    read = _carriers(1, strands=("CT",))[0]
    read.set_tag("MM", "C+u?;")
    assert ic._read_strand(read, md_first=False) == ic.STRAND_CT
    assert ic._read_strand(read, md_first=True) == ic.STRAND_CT
    _mods, strand, _seq, _unknown = extract_modification_calls(read, "daf")
    assert strand == "+"
