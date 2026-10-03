"""DAF bases with no reference counterpart carry no evidence (fh-sv-daf).

Deamination marks are read-versus-reference mismatches on matched pairs, so a
CIGAR insertion or soft clip can never carry one. Before 3.0 their unconverted
C/G were encoded as unmodified (protected) targets and the HMM called
insertions and clips nucleosome-packed. They are now masked like SNP sites
(dropped from the calls and encoded as non-target), SNP-masked sites are
encoded as non-target too, and calls are removed from unaligned blocks of at
least 50 bp. Fiber-seq m6A is read-intrinsic and unchanged.
"""
from __future__ import annotations

import array
import os
import random
import subprocess
import sys

import numpy as np
import pysam
import pytest

from fiberhmm.core.bam_reader import ContextEncoder, encode_from_query_sequence
from fiberhmm.daf.aligned_arrays import unaligned_query_positions
from fiberhmm.daf.encoder import encode_read_daf
from fiberhmm.inference import engine
from fiberhmm.inference.engine import (
    _extract_fiber_read_from_pysam,
    configure_daf_snp_mask,
    configure_daf_unaligned_mask,
    daf_snp_mask_scope,
    extract_fiber_read_from_payload,
    make_apply_payload,
)
from fiberhmm.inference.no_evidence import suppress_calls_in_blocks, unaligned_blocks
from fiberhmm.inference.nuc_recaller import NucCall
from fiberhmm.inference.tf_recaller import TFCall, extract_modification_calls

REF_START = 1_000
NON_TARGET = ContextEncoder.get_n_codes(3) * 2 + 1   # non_target + unmethylated offset


def _rand(n, seed):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


REF = _rand(4_000, 11)
INSERT = _rand(300, 12)


def _header():
    return pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"LN": 100_000, "SN": "chr1"}]})


def _md(cigar, seq, ref_start=0):
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
        elif op == 2:
            parts += [str(run), "^" + REF[r:r + n]]
            run = 0
            r += n
    parts.append(str(run))
    return "".join(parts)


def _deaminate(seq, rate, seed, target="C", product="T"):
    rng = random.Random(seed)
    return "".join(product if b == target and rng.random() < rate else b for b in seq)


def _read(kind="ins", strand="CT", flag=0, md=True):
    """A DAF read whose flanks (matched) and insert/clip carry deaminations."""
    target, product = ("C", "T") if strand == "CT" else ("G", "A")
    left = REF[0:1000]
    right = REF[1000:2000]
    if kind == "ins":
        mol = left + INSERT + right
        cigar = [(0, 1000), (1, len(INSERT)), (0, 1000)]
        unaligned = range(1000, 1000 + len(INSERT))
    elif kind == "clip":
        mol = INSERT + right
        cigar = [(4, len(INSERT)), (0, 1000)]
        unaligned = range(0, len(INSERT))
    elif kind == "small":
        mol = left + "CCG" + right
        cigar = [(0, 1000), (1, 3), (0, 1000)]
        unaligned = range(1000, 1003)
    else:
        mol = left + right
        cigar = [(0, 2000)]
        unaligned = range(0)
    seq = _deaminate(mol, 0.3, 5, target, product)
    ref_start = 1000 if kind == "clip" else 0
    a = pysam.AlignedSegment(_header())
    a.query_name = f"{kind}_{strand}"
    a.query_sequence = seq
    a.flag = flag
    a.reference_id = 0
    a.reference_start = REF_START + ref_start
    a.mapping_quality = 60
    a.cigartuples = cigar
    if md:
        a.set_tag("MD", _md(cigar, seq, ref_start))
    return a, set(unaligned), mol


@pytest.fixture(autouse=True)
def _reset():
    configure_daf_snp_mask(None)
    yield
    configure_daf_snp_mask(None)


def test_unaligned_query_positions_and_blocks():
    cig = [(5, 7), (4, 3), (0, 5), (1, 2), (2, 4), (0, 5), (4, 60), (5, 9)]
    assert unaligned_query_positions(cig) == {0, 1, 2, 8, 9} | set(range(15, 75))
    assert unaligned_query_positions(None) == set()
    assert unaligned_blocks(cig) == [(15, 75)]
    assert unaligned_blocks(cig, min_length=2) == [(0, 3), (8, 10), (15, 75)]
    # adjacent I and S merge into one block
    assert unaligned_blocks([(0, 5), (1, 30), (4, 30)]) == [(5, 65)]


@pytest.mark.parametrize("strand", ["CT", "GA"])
@pytest.mark.parametrize("flag", [0, 16])
@pytest.mark.parametrize("kind", ["ins", "clip"])
@pytest.mark.parametrize("source", ["md", "iupac", "payload"])
def test_unaligned_bases_are_no_evidence(kind, source, flag, strand):
    read, unaligned, _mol = _read(kind, strand, flag)
    if source == "iupac":
        new_seq, st, _n = encode_read_daf(read)
        # The encoder marks matched mismatches only.
        assert not any(new_seq[p] in "RY" for p in unaligned)
        read.query_sequence = new_seq
        read.set_tag("st", st)
        fr = _extract_fiber_read_from_pysam(read, "daf", 128)
    elif source == "payload":
        fr = extract_fiber_read_from_payload(make_apply_payload(read, mode="daf"), "daf", 128)
    else:
        fr = _extract_fiber_read_from_pysam(read, "daf", 128)
    assert fr["_daf_strand"] == ("+" if strand == "CT" else "-")
    assert unaligned <= fr["unknown_query_positions"]
    assert not fr["m6a_query_positions"] & unaligned
    assert fr["no_call_blocks"] == [(min(unaligned), max(unaligned) + 1)]
    obs = encode_from_query_sequence(
        fr["query_sequence"], fr["m6a_query_positions"], 10, mode="daf",
        strand=fr["_daf_strand"], context_size=3,
        unknown_positions=fr["unknown_query_positions"])
    assert np.all(obs[sorted(unaligned)] == NON_TARGET)


def test_iupac_marks_inside_an_insertion_are_dropped():
    """Even an R/Y producer that marked an inserted base gets no evidence there:
    an insertion carries no mark either way, so one-sided marks would bias."""
    read, unaligned, _ = _read("ins")
    new_seq, st, _n = encode_read_daf(read)
    seq = list(new_seq)
    p = next(i for i in sorted(unaligned) if seq[i] == "T")
    seq[p] = "Y"
    read.query_sequence = "".join(seq)
    read.set_tag("st", st)
    fr = _extract_fiber_read_from_pysam(read, "daf", 128)
    assert p not in fr["m6a_query_positions"]
    assert p in fr["unknown_query_positions"]


def test_aligned_read_is_unchanged():
    read, unaligned, _ = _read("none")
    assert not unaligned
    fr = _extract_fiber_read_from_pysam(read, "daf", 128)
    assert "unknown_query_positions" not in fr and "no_call_blocks" not in fr


def test_small_insertion_masked_without_a_no_call_block():
    read, unaligned, _ = _read("small")
    fr = _extract_fiber_read_from_pysam(read, "daf", 128)
    assert fr["unknown_query_positions"] == unaligned
    assert "no_call_blocks" not in fr


def test_mask_can_be_disabled():
    read, _, _ = _read("ins")
    configure_daf_unaligned_mask(False)
    assert os.environ[engine._DAF_UNALIGNED_MASK_ENV] == "0"
    fr = _extract_fiber_read_from_pysam(read, "daf", 128)
    assert "unknown_query_positions" not in fr
    payload = make_apply_payload(read, mode="daf")
    assert "_daf_excluded_query_positions" not in payload
    assert "_no_call_blocks" not in payload


def test_snp_masked_target_is_no_evidence_not_unmodified():
    """Before 3.0 the SNP mask only removed marks: a masked reference C read as
    C stayed an unmodified (protected) target."""
    read, _, _ = _read("none")
    seq = read.query_sequence
    site = next(i for i in range(200, 1800) if seq[i] == "C" and REF[i] == "C")
    with daf_snp_mask_scope({"chr1": {REF_START + site}}):
        fr = _extract_fiber_read_from_pysam(read, "daf", 128)
        payload_fr = extract_fiber_read_from_payload(
            make_apply_payload(read, mode="daf"), "daf", 128)
    for got in (fr, payload_fr):
        assert got["unknown_query_positions"] == {site}
        obs = encode_from_query_sequence(
            got["query_sequence"], got["m6a_query_positions"], 10, mode="daf",
            strand="+", context_size=3,
            unknown_positions=got["unknown_query_positions"])
        assert obs[site] == NON_TARGET
        assert "no_call_blocks" not in got


def test_m6a_reads_with_insertions_are_unaffected():
    read, _, mol = _read("ins")
    read.query_sequence = mol
    a_pos = [i for i, b in enumerate(mol) if b == "A"]
    mods = a_pos[::5]
    idx = {p: k for k, p in enumerate(a_pos)}
    skips, prev = [], -1
    for p in mods:
        skips.append(idx[p] - prev - 1)
        prev = idx[p]
    read.set_tag("MM", "A+a," + ",".join(map(str, skips)) + ";")
    read.set_tag("ML", array.array("B", [250] * len(mods)))
    fr = _extract_fiber_read_from_pysam(read, "nanopore-fiber", 128)
    assert fr["m6a_query_positions"] == set(mods)
    assert "unknown_query_positions" not in fr and "no_call_blocks" not in fr


def test_recall_extraction_and_payload_carry_unaligned_bases():
    from fiberhmm.cli.recall_tfs import _make_payload, _PayloadRead
    read, unaligned, _ = _read("ins")
    mod, strand, _seq, unknown = extract_modification_calls(read, "daf")
    assert unknown == unaligned and not mod & unaligned and strand == "+"
    payload = _make_payload(read, "daf")
    stub = _PayloadRead(payload["seq"], payload["is_reverse"], payload["tags"],
                        payload.get("_daf_md_result"),
                        payload.get("_daf_unaligned_query_positions"))
    assert extract_modification_calls(stub, "daf")[3] == unaligned
    new_seq, st, _n = encode_read_daf(read)
    read.query_sequence = new_seq
    read.set_tag("st", st)
    assert extract_modification_calls(read, "daf")[3] == unaligned


def test_suppress_calls_in_blocks_keeps_tags_aligned():
    blocks = [(100, 200)]
    tf_in = TFCall(150, 20, 9.0, 4, 0, 0)
    tf_edge = TFCall(190, 20, 9.0, 4, 0, 0)
    tf_out = TFCall(300, 20, 9.0, 4, 0, 0)
    result = {
        "ns": np.array([0, 120, 180, 250], dtype=np.int32),
        "nl": np.array([50, 30, 300, 40], dtype=np.int32),   # 180-480 spans past
        "nq_for_kept_nucs": [10, 20, 30, 40],
        "nuc_el_for_kept": [255, 255, 255, 255],
        "nuc_er_for_kept": [255, 255, 255, 255],
        "ns_scores": None,
        "as": np.array([50, 90], dtype=np.int32),
        "al": np.array([40, 150], dtype=np.int32),          # 90-240 spans the block
        "as_scores": np.array([0.5, 0.7], dtype=np.float32),
        "tf_calls": [tf_in, tf_edge, tf_out],
    }
    out = suppress_calls_in_blocks(result, blocks)
    assert list(out["ns"]) == [0, 200, 250]
    assert list(out["nl"]) == [50, 280, 40]
    assert out["nq_for_kept_nucs"] == [10, 30, 40]
    assert out["nuc_el_for_kept"] == [255, 0, 255]
    assert out["nuc_er_for_kept"] == [255, 255, 255]
    assert list(out["as"]) == [50, 90, 200] and list(out["al"]) == [40, 10, 40]
    assert list(out["as_scores"]) == pytest.approx([0.5, 0.7, 0.7])
    assert out["tf_calls"] == [tf_out]
    assert out["ns"].dtype == np.int32
    # circular results are left alone
    circ = {"circular": True, "ns": [120], "nl": [30]}
    assert suppress_calls_in_blocks(circ, blocks) == circ


def _write_bam(path, reads):
    header = _header()
    unsorted = str(path) + ".u.bam"
    with pysam.AlignmentFile(unsorted, "wb", header=header) as out:
        for r in reads:
            out.write(r)
    pysam.sort("-o", str(path), unsorted)
    pysam.index(str(path))


def test_fiberhmm_call_leaves_long_insertions_uncalled(tmp_path):
    """End to end: no MA interval inside a 300 bp insertion or clip; reads
    without one are called exactly as with the mask off."""
    from fiberhmm.io.ma_tags import flip_interval_frame, parse_ma_tag
    reads = []
    for i, (kind, flag) in enumerate([("ins", 0), ("ins", 16), ("clip", 0), ("none", 0)]):
        r, _u, _m = _read(kind, "CT", flag)
        r.query_name = f"{kind}_{i}"
        reads.append(r)
    bam = tmp_path / "in.bam"
    _write_bam(bam, reads)

    def call(out, *extra):
        cmd = [sys.executable, "-m", "fiberhmm.cli.call", "-i", str(bam), "-o", str(out),
               "--enzyme", "ddda", "--seq", "pacbio", "--no-dedup", "--no-daf-call-snps",
               "--no-qc", "--min-read-length", "0", "-c", "1", *extra]
        subprocess.run(cmd, check=True, capture_output=True)
        with pysam.AlignmentFile(str(out)) as f:
            return {r.query_name: r for r in f}

    on = call(tmp_path / "on.bam")
    off = call(tmp_path / "off.bam", "--no-daf-mask-unaligned")
    for name in ("ins_0", "ins_1", "clip_2"):
        r = on[name]
        ma = parse_ma_tag(r.get_tag("MA"))
        lo, hi = (1000, 1300) if name.startswith("ins") else (0, 300)
        for kind in ("nuc", "msp", "tf"):
            for s, length in ma[kind]:
                if r.is_reverse:
                    s, length = flip_interval_frame(s, length, ma["read_length"])
                assert s + length <= lo or s >= hi, (name, kind, s, length)
        assert on[name].get_tag("MA") != off[name].get_tag("MA")
    assert on["none_3"].get_tag("MA") == off["none_3"].get_tag("MA")
    header = pysam.AlignmentFile(str(tmp_path / "on.bam")).header.to_dict()
    ds = [pg.get("DS", "") for pg in header["PG"] if pg.get("PN") == "fiberhmm-call"][0]
    assert "daf_unaligned_mask=on" in ds
    header = pysam.AlignmentFile(str(tmp_path / "off.bam")).header.to_dict()
    ds = [pg.get("DS", "") for pg in header["PG"] if pg.get("PN") == "fiberhmm-call"][0]
    assert "daf_unaligned_mask=off" in ds


def test_qc_fallback_opportunities_exclude_unaligned_bases():
    """QC's no-reference fallback (R/Y without MD) counts aligned targets only."""
    from fiberhmm.qc.core import _daf_signal_profile
    read, unaligned, _ = _read("ins")
    new_seq, st, _n = encode_read_daf(read)
    read.query_sequence = new_seq
    read.set_tag("st", st)
    read.set_tag("MD", None)
    events, opportunities, _s, _e = _daf_signal_profile(read)
    aligned = [i for i in range(len(new_seq)) if i not in unaligned]
    expected = sum(1 for i in aligned if new_seq[i] in "CGRY")
    assert opportunities == expected
    assert not set(events.tolist()) & unaligned


def test_recall_tf_only_path_applies_no_call_blocks():
    from fiberhmm.cli import recall_tfs
    from fiberhmm.inference.tf_recaller import TFCall

    payload = {"seq": "A" * 2000, "is_reverse": False, "tags": {},
               "_no_call_blocks": [(1000, 1300)]}

    def fake_recall_read(*_args, **_kwargs):
        return ([TFCall(1100, 20, 9.0, 4, 0, 0), TFCall(1500, 20, 9.0, 4, 0, 0)],
                [(950, 400), (1600, 150)], [(0, 950), (1350, 250)])

    old_worker = dict(recall_tfs._WORKER)
    old = recall_tfs.recall_read
    try:
        recall_tfs._WORKER.update(llr_hit=None, llr_miss=None, mode="daf", k=3,
                                  min_llr=5.0, min_opps=3, unify_threshold=90)
        recall_tfs._WORKER.pop("nuc_cfg", None)
        recall_tfs.recall_read = fake_recall_read
        (tf, nucs, msps, _nq), stats = recall_tfs._process_payload_record(payload)
    finally:
        recall_tfs.recall_read = old
        recall_tfs._WORKER.clear()
        recall_tfs._WORKER.update(old_worker)
    assert [c.start for c in tf] == [1500] and stats["tf"] == 1
    assert nucs == [(950, 50), (1300, 50), (1600, 150)]
    assert msps == [(0, 950), (1350, 250)]
