"""Supplementary alignments are called by default (3.0, fh-sv-daf).

A supplementary record is another part of the same read (a split alignment:
the far side of a structural variant, an insertion's TE copy elsewhere), so
3.0 calls primary and supplementary records and leaves secondary records
(alternative placements of the same bases) uncalled. A supplementary record
is called on its aligned bases only: its soft clips belong to the primary.
"""
from __future__ import annotations

import array
import subprocess
import sys

import pysam
import pytest

from fiberhmm.inference.engine import (
    _extract_fiber_read_from_pysam,
    extract_fiber_read_from_payload,
    make_apply_payload,
)
from fiberhmm.inference.read_filters import (
    ReadFilterConfig,
    alignment_skipped,
    resolve_alignments,
    streaming_skip_reason,
)
from fiberhmm.io.ma_tags import flip_interval_frame, parse_ma_tag
from test_daf_unaligned_mask import INSERT, REF, REF_START, _deaminate, _header, _md


def _split_records(m6a=False, flag=0):
    """One molecule REF[0:1000] + INSERT + REF[1000:2000], split minimap2 -Y
    style: primary 1000M1300S, supplementary 1300S1000M (and a secondary)."""
    mol = REF[0:1000] + INSERT + REF[1000:2000]
    seq = mol if m6a else _deaminate(mol, 0.3, 5)
    out = []
    for kind, start, cigar, extra in (
            ("prim", 0, [(0, 1000), (4, 1300)], 0),
            ("supp", 1000, [(4, 1300), (0, 1000)], 2048),
            ("sec", 1000, [(4, 1300), (0, 1000)], 256)):
        a = pysam.AlignedSegment(_header())
        a.query_name = "mol"
        a.query_sequence = seq
        a.flag = flag | extra
        a.reference_id = 0
        a.reference_start = REF_START + start
        a.mapping_quality = 60
        a.cigartuples = cigar
        a.set_tag("MD", _md(cigar, seq, start))
        if m6a:
            a_pos = [i for i, b in enumerate(seq) if b == "A"][::4]
            a.set_tag("MM", "A+a" + ",3" * len(a_pos) + ";")
            a.set_tag("ML", array.array("B", [250] * len(a_pos)))
        out.append(a)
    return out


def test_alignment_sets():
    prim, supp, sec = _split_records()
    assert resolve_alignments(None) == "primary-supplementary"
    assert resolve_alignments(True) == "primary"
    assert resolve_alignments(False) == "all"
    with pytest.raises(ValueError):
        resolve_alignments("supplementary")
    assert [alignment_skipped(r, "primary-supplementary") for r in (prim, supp, sec)] == [
        False, False, True]
    assert [alignment_skipped(r, "primary") for r in (prim, supp, sec)] == [False, True, True]
    assert [alignment_skipped(r, "all") for r in (prim, supp, sec)] == [False, False, False]
    config = ReadFilterConfig(primary_only="primary-supplementary", mode="daf")
    assert streaming_skip_reason(supp, config) is None
    assert streaming_skip_reason(sec, config) == "secondary_supplementary"


@pytest.mark.parametrize("m6a", [False, True])
def test_supplementary_record_is_called_on_its_aligned_bases(m6a):
    _prim, supp, _sec = _split_records(m6a)
    mode = "nanopore-fiber" if m6a else "daf"
    live = _extract_fiber_read_from_pysam(supp, mode, 128)
    slim = extract_fiber_read_from_payload(make_apply_payload(supp, mode=mode), mode, 128)
    for fr in (live, slim):
        assert fr is not None
        assert fr["no_call_blocks"] == [(0, 1300)]
        if not m6a:
            # DAF: its MD gives real deamination evidence on the aligned part
            assert fr["m6a_query_positions"]
            assert min(fr["m6a_query_positions"]) >= 1300


def test_cli_calls_supplementary_by_default(tmp_path):
    path = tmp_path / "split.bam"
    unsorted = tmp_path / "u.bam"
    with pysam.AlignmentFile(str(unsorted), "wb", header=_header()) as out:
        for r in _split_records() + [_split_records(flag=16)[1]]:
            out.write(r)
    pysam.sort("-o", str(path), str(unsorted))
    pysam.index(str(path))

    def call(*extra):
        out = tmp_path / f"out{len(extra)}{'_'.join(extra)}.bam"
        subprocess.run(
            [sys.executable, "-m", "fiberhmm.cli.call", "-i", str(path), "-o", str(out),
             "--enzyme", "ddda", "--seq", "pacbio", "--no-dedup", "--no-daf-call-snps",
             "--no-qc", "--min-read-length", "0", "-c", "1", *extra],
            check=True, capture_output=True)
        return list(pysam.AlignmentFile(str(out)))

    records = call()
    called = {("supp" if r.is_supplementary else "sec" if r.is_secondary else "prim",
               r.is_reverse): r.has_tag("MA") for r in records}
    assert called[("prim", False)] and called[("supp", False)] and called[("supp", True)]
    assert not called[("sec", False)]
    for r in records:
        if r.is_supplementary and r.has_tag("MA"):
            ma = parse_ma_tag(r.get_tag("MA"))
            for kind in ("nuc", "msp", "tf"):
                for s, length in ma[kind]:
                    if r.is_reverse:
                        s, length = flip_interval_frame(s, length, ma["read_length"])
                    assert s >= 1300
    header = pysam.AlignmentFile(str(tmp_path / "out0.bam")).header.to_dict()
    ds = [pg["DS"] for pg in header["PG"] if pg.get("PN") == "fiberhmm-call"][0]
    assert "alignments=primary-supplementary" in ds and "primary_only=off" in ds
    primary_only = call("--primary")
    assert not any(r.has_tag("MA") for r in primary_only if r.is_supplementary)
    everything = call("--no-primary")
    assert all(r.has_tag("MA") for r in everything)
