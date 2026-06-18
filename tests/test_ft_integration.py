"""Integration test: fibertools (ft) must parse FiberHMM output without error.

Skipped unless the ``ft`` binary (fibertools-rs) is on PATH. Guards the
coordinate-frame + tiling regression: ft's liftover panics on unsorted /
overlapping / wrong-frame ns/nl, so this asserts ``ft extract`` succeeds on a
FiberHMM-tagged BAM containing both forward and reverse reads.
"""
from __future__ import annotations

import shutil
import subprocess

import pysam
import pytest

ft = pytest.mark.skipif(shutil.which("ft") is None,
                        reason="fibertools (ft) binary not on PATH")

READ_LEN = 3000
# alternating ~165 bp protected (nucleosome) / ~50 bp accessible (linker) blocks
PROTECTED = []
pos = 60
while pos + 165 < READ_LEN - 60:
    PROTECTED.append((pos, pos + 165))
    pos += 165 + 50


def _m6a_positions_forward():
    """A positions that are NOT inside a protected block (accessible -> methylated)."""
    prot = set()
    for a, b in PROTECTED:
        prot.update(range(a, b))
    return [p for p in range(40, READ_LEN - 40, 2) if p not in prot]


def _make_mm(seq, mod_positions, base="A", code="a"):
    base_idx = {p: i for i, p in enumerate(j for j, c in enumerate(seq) if c == base)}
    skips, prev = [], -1
    for p in mod_positions:
        skips.append(base_idx[p] - prev - 1)
        prev = base_idx[p]
    return f"{base}+{code}," + ",".join(map(str, skips)) + ";"


def _write_input(path):
    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": 100_000}],
    })
    fwd_pos = _m6a_positions_forward()
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for i, is_rev in enumerate((False, True)):
            r = pysam.AlignedSegment()
            r.query_name = f"read_{'rev' if is_rev else 'fwd'}"
            # SEQ is always forward-reference; A on forward, T on reverse (so the
            # original-read A's land where MM expects them after RC).
            r.query_sequence = ("T" if is_rev else "A") * READ_LEN
            r.query_qualities = pysam.qualitystring_to_array("I" * READ_LEN)
            r.flag = 16 if is_rev else 0
            r.reference_id = 0
            r.reference_start = 1000 + i * 10_000
            r.mapping_quality = 60
            r.cigartuples = [(0, READ_LEN)]
            mm_seq = "A" * READ_LEN
            mods = (sorted(READ_LEN - 1 - p for p in fwd_pos) if is_rev else fwd_pos)
            r.set_tag("MM", _make_mm(mm_seq, mods))
            r.set_tag("ML", [240] * len(mods))
            out.write(r)
    pysam.index(str(path))


@ft
def test_ft_extract_parses_fiberhmm_output(tmp_path):
    from fiberhmm.cli import call as call_cli

    in_bam = tmp_path / "in.bam"
    out_bam = tmp_path / "out.bam"
    _write_input(in_bam)

    # Run fiberhmm-call (recaller on by default) via its argv.
    import sys
    argv = ["fiberhmm-call", "-i", str(in_bam), "-o", str(out_bam),
            "--enzyme", "hia5", "--seq", "pacbio", "-c", "1"]
    old = sys.argv
    try:
        sys.argv = argv
        call_cli.main()
    finally:
        sys.argv = old
    pysam.index(str(out_bam))

    # ns/nl + as/al must be sorted, non-overlapping, in-bounds (what ft needs).
    bam = pysam.AlignmentFile(str(out_bam), "rb", check_sq=False)
    saw_reverse = False
    for r in bam:
        saw_reverse = saw_reverse or r.is_reverse
        L = r.query_length
        for stag, ltag in (("ns", "nl"), ("as", "al")):
            if not r.has_tag(stag):
                continue
            s = list(r.get_tag(stag))
            ln = list(r.get_tag(ltag))
            assert s == sorted(s)
            assert all(a + b <= L for a, b in zip(s, ln))
            assert all(s[i] + ln[i] <= s[i + 1] for i in range(len(s) - 1))
    bam.close()
    assert saw_reverse

    # ft extract must not panic and must produce nucleosome output.
    nuc_bed = tmp_path / "nuc.bed"
    res = subprocess.run(
        ["ft", "extract", "--nuc", str(nuc_bed), str(out_bam)],
        capture_output=True, text=True)
    assert res.returncode == 0, f"ft extract failed: {res.stderr}"
    assert "panic" not in res.stderr.lower()
    assert nuc_bed.read_text().strip(), "ft extract produced no nucleosomes"
