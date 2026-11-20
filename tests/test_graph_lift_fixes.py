"""fiberhmm-lift 3.2.0 fixes: DAF evidence on a strand change (H2) and unmapped
vg surject records (H4)."""
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
