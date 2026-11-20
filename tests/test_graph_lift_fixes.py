"""fiberhmm-lift 3.2.0 fixes: DAF evidence on a strand change (H2), unmapped vg
surject records (H4), BED12 trailing-comma per-block columns (H11), edge-score
columns on a reverse lift (M4) and rows_flipped for unstranded rows."""
from __future__ import annotations

import pysam

from fiberhmm.daf import replay
from fiberhmm.daf.insert_consensus import inserted_query_positions
from fiberhmm.io import graph_lift as gl

from test_graph_lift import _fake_lift, _read, _records, _source_header, _write


def _call_pg(snps):
    return {"ID": "fiberhmm-call", "PN": "fiberhmm-call", "PP": "minimap2", "VN": "3.2.0",
            "CL": "fiberhmm-call -i in.bam", "DS": replay.evidence_token(snps)}


def _ei(read):
    return replay._unpack(read.get_tag(replay.INSERT_TAG))


# --- H2: DAF insertion evidence and the SNP mask on a lift --------------------

def _daf_lift(tmp_path, *, flip):
    header = _source_header(extra_pg=[_call_pg({"srcA": [102, 104]})])
    seq = "ACGTCCCAGT"
    ei = replay._pack([1, 10, 0, [2, 3, 4], [3]])          # the 3I of 2M3I5M, CT strand
    reads = [lambda h: _read(h, "daf", cigar="2M3I5M", seq=seq,
                             tags=[("st", "CT", "Z"), (replay.INSERT_TAG, ei, "Z")])]
    src = _write(tmp_path / "daf.src.bam", header, reads)
    stashed = tmp_path / "daf.stash.bam"
    stash = gl.stash_bam(src, str(stashed))
    lifted = _fake_lift(stashed, tmp_path / "daf.lifted.bam", flip=flip)
    out = tmp_path / "daf.out.bam"
    stats = gl.repair_bam(lifted, str(out), work_dir=str(tmp_path / "work"),
                          pg_args={"mode": "lift", "stash": stash.as_dict(),
                                   "command": "fiberhmm-lift test"})
    (rec,), hdr = _records(out)
    return stats, rec, hdr, lifted


def test_strand_changing_lift_mirrors_insertion_evidence(tmp_path):
    stats, rec, hdr, _lifted = _daf_lift(tmp_path, flip=True)
    assert rec.cigarstring == "5M3I2M" and rec.get_tag("st") == "GA"
    version, length, strand, known, mods = _ei(rec)
    assert (version, length, strand) == (1, 10, 1)            # GA after the flip
    assert known == [5, 6, 7] and mods == [6]
    assert set(known) == inserted_query_positions(rec.cigartuples)
    assert stats.insert_evidence_flipped == 1 and stats.insert_evidence_dropped == 0
    assert "insert_evidence_flipped=1" in hdr["PG"][-1]["DS"]

    with pysam.AlignmentFile(str(tmp_path / "daf.out.bam"), "rb", check_sq=False) as bam:
        evidence = replay.EvidenceReplay(bam.header, "daf")
    assert evidence.portable and evidence.snps == {}         # source coordinates: not applied
    assert any("SNP mask" in w for w in evidence.warnings)
    from fiberhmm.inference.engine import daf_insert_evidence
    with evidence.scope(rec):
        ev = daf_insert_evidence(rec)
    assert ev.known == {5, 6, 7} and ev.mods == {6} and ev.strand == 1


def test_forward_lift_keeps_insertion_evidence_but_not_the_snp_mask(tmp_path):
    stats, rec, hdr, lifted = _daf_lift(tmp_path, flip=False)
    assert _ei(rec)[1:] == [10, 0, [2, 3, 4], [3]]
    assert stats.insert_evidence_flipped == 0 and stats.insert_evidence_dropped == 0
    with pysam.AlignmentFile(str(tmp_path / "daf.out.bam"), "rb", check_sq=False) as bam:
        assert replay.EvidenceReplay(bam.header, "daf").snps == {}
    # An unrepaired lift: neither the mask nor ei (st and ei were not moved).
    with pysam.AlignmentFile(lifted, "rb", check_sq=False) as bam:
        evidence = replay.EvidenceReplay(bam.header, "daf")
    assert evidence.snps == {} and not evidence.portable


def test_unlifted_bam_keeps_its_snp_mask():
    evidence = replay.EvidenceReplay(_source_header(extra_pg=[_call_pg({"srcA": [5]})]), "daf")
    assert evidence.portable and evidence.snps == {"srcA": {5}} and not evidence.warnings
    # A re-call after the lift is in target coordinates again.
    pgs = [_call_pg({"srcA": [5]}),
           {"ID": "fiberhmm-lift", "PN": "fiberhmm-lift", "CL": "fiberhmm-lift"},
           dict(_call_pg({"tgt": [9]}), ID="fiberhmm-call.1", PP="fiberhmm-lift")]
    evidence = replay.EvidenceReplay(_source_header(extra_pg=pgs), "daf")
    assert evidence.snps == {"tgt": {9}} and evidence.portable


def test_lift_insert_evidence_shifts_and_drops_stale_tags():
    header = pysam.AlignmentHeader.from_dict(_source_header())
    rec = _read(header, "r", cigar="4S6M", seq="AAAACCCCGG",
                tags=[(replay.INSERT_TAG, replay._pack([1, 6, 0, [0, 1], [1]]), "Z")])
    assert replay.lift_insert_evidence(rec, 6, offset=4) is True    # 4 filled bases
    assert _ei(rec) == [1, 10, 0, [4, 5], [5]]
    assert replay.lift_insert_evidence(rec, 7) is False             # written for another SEQ
    assert not rec.has_tag(replay.INSERT_TAG)
    assert replay.lift_insert_evidence(rec, 10) is None


# --- H4: unmapped vg surject records ------------------------------------------

def _unmap(path, names):
    with pysam.AlignmentFile(path, "rb", check_sq=False) as bam:
        hdr = bam.header.to_dict()
        reads = list(bam.fetch(until_eof=True))
    tmp = path + ".tmp.bam"
    with pysam.AlignmentFile(tmp, "wb", header=hdr) as out:
        for read in reads:
            rec = pysam.AlignedSegment.from_dict(read.to_dict(), out.header)
            if (rec.query_name, rec.get_tag("Yf") & 0x800) in names:
                rec.flag = (rec.flag | 0x4) & ~0x10
                rec.reference_id = rec.next_reference_id = -1
                rec.reference_start = rec.next_reference_start = -1
                rec.cigartuples = None
                rec.mapping_quality = 0
            out.write(rec)
    import os
    os.replace(tmp, path)


def test_unmapped_surject_records_are_skipped_not_written(tmp_path):
    from fiberhmm.cli.lift import _count_records

    header = _source_header()
    seq = "ACGTACGTAC" * 5
    reads = [lambda h: _read(h, "a", seq=seq),
             lambda h: _read(h, "a", flag=0x800, pos=600, cigar="20S30M", seq=seq),
             lambda h: _read(h, "b", pos=300, seq=seq),
             lambda h: _read(h, "c", pos=400, seq=seq)]
    src = _write(tmp_path / "u.src.bam", header, reads)
    stashed = tmp_path / "u.stash.bam"
    gl.stash_bam(src, str(stashed))
    lifted = _fake_lift(stashed, tmp_path / "u.lifted.bam")
    _unmap(lifted, {("a", 0x800), ("b", 0)})
    assert _count_records(lifted, 1) == 2
    out = tmp_path / "u.out.bam"
    stats = gl.repair_bam(lifted, str(out), work_dir=str(tmp_path / "work"),
                          pg_args={"mode": "lift", "command": "fiberhmm-lift test"})
    records, hdr = _records(out)
    assert sorted(r.query_name for r in records) == ["a", "c"]
    assert not any(r.is_unmapped or r.is_supplementary or r.has_tag("Yo") for r in records)
    assert stats.unmapped_skipped == 2 and stats.records_out == 2
    assert stats.orphan_supplementary_kept == 0 and stats.supplementary_restored == 0
    assert "unmapped_skipped=2" in hdr["PG"][-1]["DS"]


# --- BED12: trailing commas (H11), edge columns (M4) ---------------------------

_HEADER = ("#chrom\tchromStart\tchromEnd\tname\tscore\tstrand\tthickStart\tthickEnd\t"
           "reserved\tblockCount\tblockSizes\tchromStarts\tblockNq\tblockEl\tblockEr")
_ROW = ("srcA\t100\t400\tr\t0\t+\t100\t400\t0\t3\t50,50,50,\t0,100,250,\t"
        "10,20,30,\t1,2,3,\t4,5,6,")


def test_partial_forward_lift_subsets_trailing_comma_columns():
    bed6, rows, header = gl.explode_bed12([_HEADER, _ROW], unstranded_as="+")
    lifted = [line.replace("srcA", "tgt") for line in bed6 if "|1\t" not in line]
    out, stats = gl.reassemble_bed12(lifted, rows, explicit_strand=True, partial="keep",
                                     edge_columns=gl.bed_edge_columns(header))
    f = out[0].split("\t")
    assert f[9:12] == ["2", "50,50", "0,250"]
    assert f[12:] == ["10,30,", "1,3,", "4,6,"]
    assert stats.rows_partial == 1 and stats.edge_columns_swapped == 0


def test_reverse_partial_lift_reverses_and_swaps_edge_columns():
    bed6, rows, header = gl.explode_bed12([_HEADER, _ROW], unstranded_as="+")
    assert gl.bed_edge_columns(header) == (13, 14)
    lifted = ["tgt\t2900\t2950\t0|0\t0\t-", "tgt\t2600\t2650\t0|2\t0\t-"]
    out, stats = gl.reassemble_bed12(lifted, rows, explicit_strand=True, partial="keep",
                                     edge_columns=gl.bed_edge_columns(header))
    f = out[0].split("\t")
    assert f[1:3] == ["2600", "2950"] and f[5] == "-"
    assert f[9:12] == ["2", "50,50", "0,300"]
    # blocks reversed; the old right edges are now the left ones
    assert f[12:] == ["30,10,", "6,4,", "3,1,"]
    assert stats.rows_flipped == 1 and stats.edge_columns_swapped == 1
    # Unknown columns (no header): reversed, not swapped.
    out, stats = gl.reassemble_bed12(lifted, rows, explicit_strand=True, partial="keep")
    assert out[0].split("\t")[12:] == ["30,10,", "3,1,", "6,4,"]


def test_bed_edge_columns_explicit_and_absent():
    assert gl.bed_edge_columns([], "14,15") == (13, 14)
    assert gl.bed_edge_columns(["track name=x"]) is None
    for bad in ("14", "3,4", "14,14", "a,b"):
        try:
            gl.bed_edge_columns([], bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_cli_edge_columns_needs_bed(tmp_path, capsys):
    import pytest
    from fiberhmm.cli import lift

    with pytest.raises(SystemExit):
        lift.main(["-i", str(tmp_path / "x.bam"), "-o", str(tmp_path / "y.bam"),
                   "--map", "m.json", "--edge-columns", "14,15"])
    assert "--edge-columns is for --bed" in capsys.readouterr().err


# --- rows_flipped for BED3/BED4/'.' rows ---------------------------------------

def test_rows_flipped_counts_unstranded_rows():
    rows_in = ["srcA\t100\t200", "srcA\t300\t400\tn", "srcA\t500\t600\tm\t0\t.",
               "srcA\t700\t800\tp\t0\t+", "srcA\t900\t950"]
    bed6, rows, _ = gl.explode_bed_rows(rows_in, unstranded_as="+")
    lifted = ["tgt\t1000\t1100\t0|m\t0\t-", "tgt\t1200\t1300\t1|m\t0\t-",
              "tgt\t1400\t1500\t2|m\t0\t-", "tgt\t1600\t1700\t3|m\t0\t-",
              "tgt\t1800\t1850\t4|m\t0\t+"]
    out, stats = gl.reassemble_bed_rows(lifted, rows, unstranded_as="+")
    assert out == ["tgt\t1000\t1100", "tgt\t1200\t1300\tn", "tgt\t1400\t1500\tm\t0\t.",
                   "tgt\t1600\t1700\tp\t0\t-", "tgt\t1800\t1850"]
    assert stats.rows_flipped == 4
    # Without unstranded_as (ft copies the strand column) only stranded flips are known.
    assert gl.reassemble_bed_rows(lifted, rows)[1].rows_flipped == 1
