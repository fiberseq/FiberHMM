"""MM/ML parser regression tests: ensure our vectorized parser stays
byte-for-byte equivalent to pysam.AlignedSegment.modified_bases.

Why this test exists: for v2.0.0 through v2.8.2, parse_mm_tag_query_positions
walked skip-counts in the stored SEQ direction instead of the ORIGINAL
sequencing direction.  This produced wrong positions on reverse-aligned
reads (matching pysam on forward reads but diverging entirely on reverse
reads).  The bug was invisible from inside the pipeline because every
internal stage used the same parser and agreed with every other stage.

This test guarantees the regression can't come back: any time someone
touches MM/ML parsing, CI re-validates against pysam on a large mixed
forward/reverse hia5 BAM.  If you ever see one of these fail, DO NOT
disable it — the parser is wrong again.
"""
from __future__ import annotations

import os
import random

import numpy as np
import pytest

pysam = pytest.importorskip("pysam")

from fiberhmm.core.bam_reader import (
    parse_mm_tag_query_positions,
    parse_mm_ml_per_mod_type,
)


# Path to a real hia5 PacBio BAM with MM/ML tags and a mix of forward + reverse
# reads.  If missing, the integration-style tests skip gracefully.
REAL_BAM_CANDIDATES = [
    "/Users/tt7739/Dropbox/Fiber-NET-seq/FiberHMM v1.0/v3-caller/phase0/data/"
    "test_hia5_2-4hr_sna_eve_ftz.bam",
    "/tmp/bench/test_hia5_2-4hr_sna_eve_ftz.bam",
]


def _find_real_bam():
    for p in REAL_BAM_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Synthetic hand-built cases — don't depend on a real BAM
# ---------------------------------------------------------------------------

def _make_synthetic_bam(reads, tmp_path):
    """Build a tiny BAM with the given list of (name, seq, is_reverse, mm, ml) reads."""
    header = {
        "HD": {"VN": "1.6", "SO": "unsorted"},
        "SQ": [{"LN": 1_000_000, "SN": "chr1"}],
    }
    bam_path = str(tmp_path / "synth.bam")
    with pysam.AlignmentFile(bam_path, "wb", header=header) as f:
        for i, (name, seq, is_reverse, mm, ml) in enumerate(reads):
            a = pysam.AlignedSegment(f.header)
            a.query_name = name
            a.query_sequence = seq
            a.flag = 16 if is_reverse else 0
            a.reference_id = 0
            a.reference_start = 1000 + i * 50_000
            a.mapping_quality = 60
            a.cigartuples = [(0, len(seq))]  # all match (for simplicity)
            a.query_qualities = pysam.qualitystring_to_array("I" * len(seq))
            tags = [("MM", mm, "Z")]
            if isinstance(ml, (bytes, bytearray)):
                import array as pyarray
                tags.append(("ML", pyarray.array("B", list(ml))))
            else:
                tags.append(("ML", list(ml)))
            a.set_tags(tags)
            f.write(a)
    pysam.index(bam_path)
    return bam_path


def _pysam_positions(read, mod_codes={"a"}, prob_threshold=125):
    """Union of pysam modified_bases positions for given mod codes."""
    try:
        mb = read.modified_bases
    except Exception:
        return set()
    if not mb:
        return set()
    positions = set()
    for (base, strand_code, mod_code), pqs in mb.items():
        if mod_code not in mod_codes:
            continue
        for pos, qual in pqs:
            if qual == -1 or qual >= prob_threshold:
                positions.add(pos)
    return positions


# Tiny synthetic read: forward strand, palindromic-safe sequence
# Seq is 40 bp with A's at positions [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
# MM "A+a.,0,1,2" → skip 0 A's (mod at A_idx 0), skip 1 A (mod at A_idx 2),
# skip 2 A's (mod at A_idx 5).  So modified A's at positions [0, 8, 20].
SYN_FWD = {
    "name": "synfwd",
    "seq": "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT",
    "mm": "A+a.,0,1,2;",
    "ml": [200, 200, 200],
    "expected_forward_positions": {0, 8, 20},
}


def test_synthetic_forward_read(tmp_path):
    """Forward-strand synthetic read: parser should find exactly [0, 8, 20]."""
    bam = _make_synthetic_bam(
        [(SYN_FWD["name"], SYN_FWD["seq"], False, SYN_FWD["mm"], SYN_FWD["ml"])],
        tmp_path,
    )
    with pysam.AlignmentFile(bam, "rb") as f:
        r = next(iter(f))
        ours = parse_mm_tag_query_positions(
            SYN_FWD["mm"], bytes(SYN_FWD["ml"]),
            r.query_sequence, r.is_reverse, 125, "pacbio-fiber")
        ref = _pysam_positions(r)
        assert ours == SYN_FWD["expected_forward_positions"]
        assert ours == ref


def test_synthetic_reverse_read(tmp_path):
    """Reverse-strand synthetic read: parser must match pysam.

    With is_reverse=True, SEQ as stored is RC of the original sequenced read.
    MM walks positions in the original direction; pysam flips those to SEQ frame.
    """
    bam = _make_synthetic_bam(
        [(SYN_FWD["name"], SYN_FWD["seq"], True, SYN_FWD["mm"], SYN_FWD["ml"])],
        tmp_path,
    )
    with pysam.AlignmentFile(bam, "rb") as f:
        r = next(iter(f))
        ours = parse_mm_tag_query_positions(
            SYN_FWD["mm"], bytes(SYN_FWD["ml"]),
            r.query_sequence, r.is_reverse, 125, "pacbio-fiber")
        ref = _pysam_positions(r)
        assert ours == ref, (
            f"reverse-read parser drift: ours={sorted(ours)} pysam={sorted(ref)}")


def test_empty_mm_returns_empty_set():
    """Empty MM → empty positions."""
    assert parse_mm_tag_query_positions("", b"\x80", "ACGT", False) == set()


def test_empty_ml_returns_empty_set():
    """Empty ML → empty positions (also the pysam segfault surface)."""
    assert parse_mm_tag_query_positions("A+a.,0;", b"", "AAAA", False) == set()


def test_mm_skip_beyond_sequence(tmp_path):
    """Skip count overshooting the A count should silently drop excess."""
    # Sequence has 4 A's, MM tries to reference the 10th A → should return empty
    bam = _make_synthetic_bam(
        [("overshoot", "AAAAGGGGGGGGGG", False, "A+a.,10;", [200])],
        tmp_path,
    )
    with pysam.AlignmentFile(bam, "rb") as f:
        r = next(iter(f))
        ours = parse_mm_tag_query_positions(
            "A+a.,10;", bytes([200]),
            r.query_sequence, r.is_reverse, 125, "pacbio-fiber")
        ref = _pysam_positions(r)
        assert ours == ref == set()


def test_daf_mm_ml_reverse_strand(tmp_path):
    """DAF with MM/ML encoding (not IUPAC): reverse-strand parsing must match pysam.

    Legacy DAF BAMs sometimes encode deamination as MM/ML with C+m or similar
    entries rather than IUPAC R/Y in the sequence.  The same reverse-strand
    flip must apply as for hia5.
    """
    # 20 bp sequence with 4 C's at positions [1, 5, 9, 13]
    # MM "C+m.,0,1;" marks the 0th C (pos 1) and 2nd C (pos 9 — skip 1 more after)
    seq = "ACGTCTGACAGTCAGCAGTC"
    mm = "C+m.,0,1;"
    ml = [200, 200]

    for is_reverse in (False, True):
        bam = _make_synthetic_bam(
            [(f"daf_{is_reverse}", seq, is_reverse, mm, ml)], tmp_path)
        with pysam.AlignmentFile(bam, "rb") as f:
            r = next(iter(f))
            ours = parse_mm_tag_query_positions(
                mm, bytes(ml), r.query_sequence, r.is_reverse,
                prob_threshold=125, mode="daf")
            # pysam gives us positions per (base, strand, mod_code) — filter for 'm'
            ref = _pysam_positions(r, mod_codes={"m"}, prob_threshold=125)
            assert ours == ref, (
                f"DAF MM/ML reverse={is_reverse}: "
                f"ours={sorted(ours)} pysam={sorted(ref)}")


def test_threshold_filtering():
    """ML values below threshold must be excluded."""
    # 4 A's, MM marks all 4, ML values [200, 100, 200, 50], threshold=125
    seq = "AAAAG"
    mm = "A+a.,0,0,0,0;"
    ml = [200, 100, 200, 50]
    ours = parse_mm_tag_query_positions(mm, bytes(ml), seq, False, 125, "pacbio-fiber")
    # Positions at A indices 0, 1, 2, 3 = seq positions 0, 1, 2, 3
    # Kept: ML>=125 → indices 0 and 2 → seq positions 0 and 2
    assert ours == {0, 2}


# ---------------------------------------------------------------------------
# Bulk tests against real hia5 BAM (integration)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_reads", [100, 500])
def test_real_bam_bulk_match_pysam(n_reads):
    """On a real hia5 BAM, our parser's positions must match pysam's on every
    read, both forward and reverse."""
    bam_path = _find_real_bam()
    if bam_path is None:
        pytest.skip("no real hia5 BAM available in this environment")

    fwd_ok = fwd_total = rev_ok = rev_total = 0
    drift_examples = []

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as f:
        for i, r in enumerate(f):
            if i >= n_reads:
                break
            if r.is_unmapped:
                continue
            if not r.has_tag("MM") or not r.has_tag("ML"):
                continue
            if len(r.get_tag("ML")) == 0:
                continue

            ref = _pysam_positions(r)
            ours = parse_mm_tag_query_positions(
                r.get_tag("MM"), bytes(r.get_tag("ML")),
                r.query_sequence, r.is_reverse, 125, "pacbio-fiber")

            if r.is_reverse:
                rev_total += 1
                if ours == ref:
                    rev_ok += 1
                elif len(drift_examples) < 3:
                    drift_examples.append(
                        (r.query_name, "reverse",
                         sorted(ref)[:5], sorted(ours)[:5]))
            else:
                fwd_total += 1
                if ours == ref:
                    fwd_ok += 1
                elif len(drift_examples) < 3:
                    drift_examples.append(
                        (r.query_name, "forward",
                         sorted(ref)[:5], sorted(ours)[:5]))

    drift_msg = "\n".join(
        f"  {name} ({strand}): pysam={ref_p}, ours={ours_p}"
        for name, strand, ref_p, ours_p in drift_examples
    )
    assert fwd_ok == fwd_total, f"forward-read drift: {fwd_ok}/{fwd_total}\n{drift_msg}"
    assert rev_ok == rev_total, f"reverse-read drift: {rev_ok}/{rev_total}\n{drift_msg}"


def test_real_bam_per_mod_type_matches_pysam():
    """parse_mm_ml_per_mod_type must produce the same (base, mod_code) groups
    as pysam on real BAM."""
    bam_path = _find_real_bam()
    if bam_path is None:
        pytest.skip("no real hia5 BAM available in this environment")

    tested = matched = 0
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as f:
        for i, r in enumerate(f):
            if i >= 50:
                break
            if r.is_unmapped or not r.has_tag("MM") or not r.has_tag("ML"):
                continue
            if len(r.get_tag("ML")) == 0:
                continue

            try:
                ref_mods = r.modified_bases or {}
            except Exception:
                continue
            # Flatten pysam result to {(base, mod_code): set of (pos, qual)}
            ref_flat = {}
            for (base, strand_code, mod_code), pqs in ref_mods.items():
                key = (base, mod_code)
                ref_flat.setdefault(key, set()).update(pqs)

            ours = parse_mm_ml_per_mod_type(
                r.get_tag("MM"), bytes(r.get_tag("ML")),
                r.query_sequence, r.is_reverse)
            # Convert ours to same shape
            ours_flat = {}
            for key, (pos_arr, qual_arr) in ours.items():
                ours_flat[key] = set(zip(pos_arr.tolist(), qual_arr.tolist()))

            tested += 1
            if ref_flat == ours_flat:
                matched += 1
            else:
                # On failure: fail with a diff of the first mismatching key
                assert ref_flat.keys() == ours_flat.keys(), (
                    f"key mismatch on {r.query_name}: "
                    f"pysam={list(ref_flat)}, ours={list(ours_flat)}")
                for key in ref_flat:
                    if ref_flat[key] != ours_flat[key]:
                        only_p = ref_flat[key] - ours_flat[key]
                        only_o = ours_flat[key] - ref_flat[key]
                        raise AssertionError(
                            f"{r.query_name} {key}: only_pysam={len(only_p)} "
                            f"only_ours={len(only_o)} "
                            f"sample_pysam={sorted(only_p)[:3]} "
                            f"sample_ours={sorted(only_o)[:3]}")
    assert matched == tested
