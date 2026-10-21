"""fiberhmm-lift stash and repair on synthetic BAMs (no vg needed).

``_fake_lift`` imitates what ``ft pg-pansn | vg inject | vg surject | ft
pg-pansn --strip`` does to a record, as measured on real lifts: the record
moves to the target contig (optionally changing strand: SEQ reverse-
complemented, QUAL reversed, CIGAR reversed), 0x800 and hard clips are lost,
SEQ/QUAL/aux tags are copied verbatim, and the header gains unlinked vg and
pg-pansn @PG lines.
"""
from __future__ import annotations

import array
from pathlib import Path

import pysam
import pytest

from fiberhmm.io import graph_lift as gl
from fiberhmm.io.annotation_frame import legacy_tag_frame, ma_annotation_frame

SRC_LEN = 2000
TGT_LEN = 3000


def _source_header(extra_pg=(), co=()):
    return {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "srcA", "LN": SRC_LEN}],
        "PG": [{"ID": "minimap2", "PN": "minimap2", "VN": "2.28", "CL": "minimap2 -a ref.fa r.fq"},
               *extra_pg],
        "CO": list(co),
    }


def _read(header, name, *, flag=0, pos=100, cigar="50M", seq=None, qual=True, mapq=60,
          tags=()):
    read = pysam.AlignedSegment(header)
    read.query_name = name
    read.flag = flag
    read.reference_id = 0
    read.reference_start = pos
    read.mapping_quality = mapq
    read.cigarstring = cigar
    if seq is not None:
        read.query_sequence = seq
        if qual:
            read.query_qualities = array.array("B", [(i * 7) % 40 for i in range(len(seq))])
    for tag, value, *vtype in tags:
        read.set_tag(tag, value, value_type=vtype[0] if vtype else None)
    return read


def _write(path, header, reads):
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for make in reads:
            out.write(make(out.header))
    return str(path)


def _records(path):
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
        return list(bam.fetch(until_eof=True)), bam.header.to_dict()


def _fake_lift(stashed, lifted, *, flip=False, shift=500):
    """Lift every record to contig ``tgt`` (see module docstring)."""
    with pysam.AlignmentFile(str(stashed), "rb", check_sq=False) as bam:
        hdr = bam.header.to_dict()
        reads = list(bam.fetch(until_eof=True))
    hdr["SQ"] = [{"SN": "tgt", "LN": TGT_LEN}]
    hdr["HD"] = {"VN": "1.5", "SO": "unknown"}
    pgs = list(hdr.get("PG", []))
    hdr["PG"] = [{"ID": "0", "PN": "vg"}] + pgs + [
        {"ID": "ft.1", "PN": "fibertools-rs", "VN": "0.13.0",
         "CL": "ft pg-pansn in.bam --prefix SRC#0# --uncompressed --header-out h"},
        {"ID": "ft.2", "PN": "fibertools-rs", "PP": "ft.1", "VN": "0.13.0",
         "CL": "ft pg-pansn - out.bam --strip --delimiter # --copy-header h --uncompressed"},
    ]
    with pysam.AlignmentFile(str(lifted), "wb", header=hdr) as out:
        for src in reads:
            rec = pysam.AlignedSegment(out.header)
            rec.query_name = src.query_name
            cigar = [(op, n) for op, n in src.cigartuples if op != 5]
            seq, qual = src.query_sequence, src.query_qualities
            flag = src.flag & ~0x800
            if flip:
                flag ^= 0x10
                cigar = cigar[::-1]
                seq = gl.revcomp(seq)
                qual = array.array("B", list(qual)[::-1]) if qual is not None else None
                ref_len = sum(n for op, n in cigar if op in (0, 2, 3, 7, 8))
                pos = TGT_LEN - (src.reference_start + shift) - ref_len
            else:
                pos = src.reference_start + shift
            rec.flag = flag
            rec.reference_id = 0
            rec.reference_start = pos
            rec.mapping_quality = 60 if src.mapping_quality == 0 else src.mapping_quality
            rec.cigartuples = cigar
            rec.query_sequence = seq
            if qual is not None:
                rec.query_qualities = qual
            rec.set_tags(src.get_tags())
            out.write(rec)
    return str(lifted)


def _seq(n, seed=1):
    bases = "ACGT"
    return "".join(bases[(i * 7 + seed * 3 + (i // 5)) % 4] for i in range(n))


def _lift_and_repair(tmp_path, header, reads, *, flip=False, fasta=None, name="x"):
    src = _write(tmp_path / f"{name}.src.bam", header, reads)
    stashed = tmp_path / f"{name}.stash.bam"
    stash = gl.stash_bam(src, str(stashed))
    lifted = _fake_lift(stashed, tmp_path / f"{name}.lifted.bam", flip=flip)
    out = tmp_path / f"{name}.out.bam"
    stats = gl.repair_bam(lifted, str(out), work_dir=str(tmp_path / f"{name}.work"),
                          target_fasta=fasta,
                          pg_args={"mode": "lift", "source": "SRC", "target": "TGT",
                                   "graph": "/some/dir/g.gbz", "stash": stash.as_dict(),
                                   "command": "fiberhmm-lift test"})
    records, hdr = _records(out)
    return stash, stats, records, hdr


# --- helpers -----------------------------------------------------------------

def test_revcomp_handles_iupac_daf_codes():
    assert gl.revcomp("ACGTRYKMBVDHSWN") == "NWSDHBVKMRYACGT"
    assert gl.revcomp(gl.revcomp("AYRGTN")) == "AYRGTN"


def test_stash_tag_names_do_not_collide_with_known_tags():
    known = {"ns", "nl", "as", "al", "nq", "aq", "MA", "AQ", "AN", "Ma", "Aq", "An", "st",
             "fp", "fs", "ec", "rq", "np", "MM", "ML", "MN", "HP", "PS", "AS", "NM", "MD",
             "SA", "cs", "tp", "cm", "s1", "s2", "de", "rl", "RG", "di", "mt", "mp"}
    tags = set(gl.STASH_TAGS + gl.PROVENANCE_TAGS)
    assert not tags & known
    assert all(tag[0] == "Y" for tag in tags)   # SAM user space


# --- forward and strand-flipped lifts ---------------------------------------------

def test_forward_lift_keeps_flags_mapq_and_drops_stale_alignment_tags(tmp_path):
    header = _source_header()
    seq = _seq(50)
    reads = [lambda h: _read(h, "fwd", seq=seq, mapq=7,
                             tags=[("MD", "50"), ("NM", 0), ("st", "CT", "Z"),
                                   ("cs", ":50", "Z"),
                                   ("MA", "50;nuc+Q:5-10", "Z")])]
    stash, stats, records, hdr = _lift_and_repair(tmp_path, header, reads)
    (rec,) = records
    assert rec.reference_name == "tgt" and rec.reference_start == 600
    assert rec.flag == 0 and rec.mapping_quality == 7
    assert rec.get_tag("Yq") == 7 and rec.get_tag("Yp") == "srcA:101:+"
    for tag in ("Yf", "Yh", "MD", "NM", "SA", "cs"):
        assert not rec.has_tag(tag)
    assert rec.get_tag("st") == "CT" and stats.st_swapped == 0
    assert hdr["HD"]["SO"] == "coordinate"
    assert (tmp_path / "x.out.bam.bai").exists()


def test_strand_flip_swaps_st_and_keeps_molecular_annotations(tmp_path):
    header = _source_header(co=["fiberhmm:coord=molecular"])
    seq = "ACYTTGCYAA" * 5
    reads = [lambda h: _read(h, "daf", seq=seq, tags=[("st", "CT", "Z"),
                                                     ("cs", "a.ct;a.ga", "Z"),
                                                     ("MA", "50;nuc.Q:5-10", "Z"),
                                                     ("AQ", array.array("B", [200]))])]
    _stash, stats, records, _hdr = _lift_and_repair(tmp_path, header, reads, flip=True)
    (rec,) = records
    assert rec.is_reverse and rec.query_sequence == gl.revcomp(seq)
    assert "R" in rec.query_sequence and "Y" not in rec.query_sequence
    assert rec.get_tag("st") == "GA"
    assert rec.get_tag("cs") == "a.ct;a.ga"       # duplex source names: kept
    assert rec.get_tag("MA") == "50;nuc.Q:5-10"   # molecular: untouched
    assert rec.get_tag("Yp") == "srcA:101:+"
    assert stats.strand_flipped == 1 and stats.st_swapped == 1


def test_seq_frame_annotations_are_carried_to_molecular_before_the_lift(tmp_path):
    # No coord=molecular anywhere: unmarked FiberHMM <= 2.12, SEQ frame.
    header = _source_header()
    seq = _seq(50)
    reads = [lambda h: _read(h, "rev", flag=0x10, seq=seq,
                             tags=[("ns", array.array("I", [5])), ("nl", array.array("I", [10]))])]
    stash, _stats, records, hdr = _lift_and_repair(tmp_path, header, reads, flip=True)
    (rec,) = records
    assert stash.frame_converted == 1
    assert list(rec.get_tag("ns")) == [35] and list(rec.get_tag("nl")) == [10]
    assert not rec.is_reverse   # flipped onto the forward strand
    assert legacy_tag_frame(hdr)[0] == "molecular"
    assert ma_annotation_frame(hdr) == "molecular"


def test_undecidable_legacy_frame_is_refused(tmp_path):
    # A merge of a FiberHMM 2.10 history (SEQ) and a fibertools one
    # (molecular), with nothing declaring coord=molecular.
    header = _source_header(extra_pg=[
        {"ID": "fiberhmm-apply", "PN": "fiberhmm-apply", "VN": "2.10.0", "PP": "minimap2",
         "CL": "fiberhmm-apply -i a.bam"},
        {"ID": "ft", "PN": "fibertools-rs", "VN": "0.6", "CL": "ft add-nucleosomes c.bam"},
    ])
    assert legacy_tag_frame(header)[0] is None
    # 5000 unannotated records first: the refusal must not depend on a sample.
    reads = [lambda h, k=k: _read(h, f"plain{k}", pos=10 + k % 50, seq=_seq(50))
             for k in range(5001)]
    reads.append(lambda h: _read(h, "late", flag=0x10, pos=200, seq=_seq(50),
                                 tags=[("ns", array.array("I", [1])),
                                       ("nl", array.array("I", [5]))]))
    src = _write(tmp_path / "amb.bam", header, reads)
    with pytest.raises(gl.FrameError, match="late"):
        gl.stash_bam(src, str(tmp_path / "amb.stash.bam"))
    # repair-only converts with the source orientation: same refusal.
    fwd = _write(tmp_path / "amb_src.bam", header, reads[-1:])
    lifted = _fake_lift(fwd, tmp_path / "amb.lifted.bam", flip=True)
    with pytest.raises(gl.FrameError):
        gl.stash_from_source(lifted, fwd, str(tmp_path / "amb.restash.bam"))


# --- supplementaries -----------------------------------------------------------

def _chimera(primary_reverse=True, with_primary=True, mods=True):
    """A 60 bp DAF molecule: primary over all of it, supplementary aligned
    over molecule bases 40..59 with the first 40 hard-clipped."""
    molecule = "ACGTTGCAAGCTTGCAYCGTAGCTAGRTCGATCGTAGCTAGCTAACGTYGCATGCAAGCT"
    assert len(molecule) == 60
    prim_seq = gl.revcomp(molecule) if primary_reverse else molecule
    supp_seq = molecule[40:]          # supplementary on the forward strand
    mm = [("MM", "C+m?,0;", "Z"), ("ML", array.array("B", [200]))] if mods else []
    out = []
    if with_primary:
        out.append(lambda h: _read(h, "chim", flag=0x10 if primary_reverse else 0, pos=100,
                                   cigar="60M", seq=prim_seq, mapq=50,
                                   tags=[("st", "GA" if primary_reverse else "CT", "Z"),
                                         ("SA", "srcA,1001,+,40H20M,9,0;", "Z"),
                                         ("MN", 60), *mm]))
    out.append(lambda h: _read(h, "chim", flag=0x800, pos=1000, cigar="40H20M", seq=supp_seq,
                               mapq=9, tags=[("st", "CT", "Z"), ("MN", 20),
                                             ("SA", "srcA,101,-,60M,50,0;", "Z"),
                                             ("MM", "C+m?;", "Z"), ("ML", array.array("B", []))]))
    return molecule, out


def test_hard_clipped_supplementary_is_filled_from_the_primary(tmp_path):
    molecule, reads = _chimera(primary_reverse=True)
    header = _source_header()
    src = _write(tmp_path / "c.bam", header, reads)
    stats = gl.stash_bam(src, str(tmp_path / "c.stash.bam"))
    assert stats.filled_supplementary == 1 and stats.excluded_supplementary == 0
    records, _ = _records(tmp_path / "c.stash.bam")
    supp = next(r for r in records if r.is_supplementary)
    assert supp.cigarstring == "40S20M"
    assert supp.query_sequence == molecule          # forward strand: molecule order
    assert supp.query_length == supp.infer_read_length() == 60
    assert supp.get_tag("Yh") == "S:40,0" and supp.get_tag("Yf") == 0x800
    prim = next(r for r in records if not r.is_supplementary)
    assert list(supp.query_qualities) == list(prim.query_qualities)[::-1][:40] + \
        [(i * 7) % 40 for i in range(20)]
    # Its own (empty) MM, valid on its 20 stored bases, now covers 60.
    assert supp.get_tag("MM") == "C+m?;" and supp.get_tag("MN") == 60


def test_supplementary_restored_one_primary_per_qname_and_sa_rewritten(tmp_path):
    _molecule, reads = _chimera(primary_reverse=True)
    fasta = tmp_path / "tgt.fa"
    fasta.write_text(">tgt\n" + _seq(TGT_LEN, seed=3) + "\n")
    pysam.faidx(str(fasta))
    _stash, stats, records, hdr = _lift_and_repair(
        tmp_path, _source_header(), reads, flip=True, fasta=str(fasta))
    assert len(records) == 2
    primaries = [r for r in records if not r.is_supplementary and not r.is_secondary]
    supps = [r for r in records if r.is_supplementary]
    assert len(primaries) == 1 and len(supps) == 1
    prim, supp = primaries[0], supps[0]
    assert supp.mapping_quality == 9 and prim.mapping_quality == 50
    assert stats.supplementary_restored == 1 and stats.st_swapped == 2
    assert prim.get_tag("st") == "CT" and supp.get_tag("st") == "GA"
    # MD/NM recomputed against the target, SA from the lifted sibling.
    assert prim.has_tag("MD") and prim.has_tag("NM") and supp.has_tag("NM")
    sa = prim.get_tag("SA").rstrip(";").split(",")
    assert sa[0] == "tgt" and int(sa[1]) == supp.reference_start + 1
    assert sa[2] == ("-" if supp.is_reverse else "+") and sa[3] == supp.cigarstring
    assert int(sa[4]) == 9 and int(sa[5]) == supp.get_tag("NM")
    assert supp.get_tag("SA").startswith(f"tgt,{prim.reference_start + 1},")
    ds = next(pg for pg in hdr["PG"] if pg["ID"] == "fiberhmm-lift")["DS"]
    assert "calmd=yes" in ds and "excluded_supplementary=0" in ds


def test_hard_clipped_supplementary_without_primary_is_excluded_and_recorded(tmp_path):
    _molecule, reads = _chimera(with_primary=False)
    reads.append(lambda h: _read(h, "other", seq=_seq(50)))
    stash, stats, records, hdr = _lift_and_repair(tmp_path, _source_header(), reads)
    assert stash.orphan_supplementary_no_primary_in_input == 1
    assert stash.excluded_supplementary == 0
    assert [r.query_name for r in records] == ["other"]
    pg = next(pg for pg in hdr["PG"] if pg["ID"] == "fiberhmm-lift")
    assert "orphan_supplementary_no_primary_in_input=1" in pg["DS"]
    assert "calmd=no" in pg["DS"] and "MD/NM/SA dropped" in pg["DS"]


def test_orphan_supplementary_is_dropped_when_its_primary_does_not_lift(tmp_path):
    header = _source_header()
    src = _write(tmp_path / "o.bam", header, [
        lambda h: _read(h, "q", seq=_seq(50)),
        lambda h: _read(h, "q", flag=0x800, pos=900, cigar="20S30M", seq=_seq(50)),
    ])
    gl.stash_bam(src, str(tmp_path / "o.stash.bam"))
    # vg drops reads it cannot place: remove the primary from the lifted BAM.
    lifted = _fake_lift(tmp_path / "o.stash.bam", tmp_path / "o.lifted.bam")
    records, hdr = _records(lifted)
    with pysam.AlignmentFile(str(tmp_path / "o.lifted2.bam"), "wb",
                             header=pysam.AlignmentHeader.from_dict(hdr)) as out:
        for rec in records:
            if rec.get_tag("Yf") & 0x800:
                out.write(rec)
    stats = gl.repair_bam(str(tmp_path / "o.lifted2.bam"), str(tmp_path / "o.out.bam"),
                          work_dir=str(tmp_path / "o.work"), pg_args={"mode": "lift"})
    assert stats.orphan_supplementary_dropped == 1 and stats.records_out == 0
    # 0x800 restored on a record that was then dropped is not reported.
    assert stats.supplementary_restored == 0


def test_supplementaries_without_a_primary_in_the_input_are_counted_at_stash(tmp_path):
    src = _write(tmp_path / "np.bam", _source_header(), [
        lambda h: _read(h, "soft", flag=0x800, pos=300, cigar="20S30M", seq=_seq(50)),
        lambda h: _read(h, "hard", flag=0x800, pos=400, cigar="20H30M", seq=_seq(30)),
        lambda h: _read(h, "late", flag=0x800, pos=500, cigar="20S30M", seq=_seq(50)),
        lambda h: _read(h, "late", pos=600, seq=_seq(50)),       # primary after its supp
    ])
    stats = gl.stash_bam(src, str(tmp_path / "np.stash.bam"))
    assert stats.orphan_supplementary_no_primary_in_input == 2
    assert stats.excluded_supplementary == 0 and stats.stashed == 2
    records, _ = _records(tmp_path / "np.stash.bam")
    assert sorted(r.query_name for r in records) == ["late", "late"]
    lifted = _fake_lift(tmp_path / "np.stash.bam", tmp_path / "np.lifted.bam")
    repair = gl.repair_bam(lifted, str(tmp_path / "np.out.bam"),
                           work_dir=str(tmp_path / "np.w"), pg_args={"mode": "lift"})
    assert repair.supplementary_restored == 1 and repair.orphan_supplementary_dropped == 0


# --- repair-only (lifted without the stash) -------------------------------------

def test_repair_only_rebuilds_the_stash_from_the_source(tmp_path):
    molecule, reads = _chimera(primary_reverse=False)
    header = _source_header()
    src = _write(tmp_path / "r.src.bam", header, reads + [
        lambda h: _read(h, "solo", pos=300, seq=_seq(50), mapq=3, tags=[("st", "GA", "Z")])])
    # ft pg-lift run directly: no stash tags, flipped target.
    lifted = _fake_lift(src, tmp_path / "r.lifted.bam", flip=True)
    restashed = tmp_path / "r.restash.bam"
    match = gl.stash_from_source(lifted, src, str(restashed))
    assert match.matched == 3 and match.unmatched == 0
    stats = gl.repair_bam(str(restashed), str(tmp_path / "r.out.bam"),
                          work_dir=str(tmp_path / "r.work"),
                          pg_args={"mode": "repair-only", "source": "SRC", "target": "TGT"})
    records, hdr = _records(tmp_path / "r.out.bam")
    by = {(r.query_name, r.is_supplementary): r for r in records}
    supp = by[("chim", True)]
    assert supp.cigarstring == "20M40H"            # hard clip re-added, flipped side
    assert supp.infer_read_length() == 60
    assert not by[("chim", False)].is_supplementary
    assert by[("solo", False)].get_tag("st") == "CT" and by[("solo", False)].mapping_quality == 3
    assert stats.hard_clips_restored == 1
    assert "mode=repair-only" in hdr["PG"][-1]["DS"]


def test_repair_refuses_unstashed_records_and_repaired_bams(tmp_path):
    src = _write(tmp_path / "u.bam", _source_header(), [lambda h: _read(h, "r", seq=_seq(50))])
    lifted = _fake_lift(src, tmp_path / "u.lifted.bam")
    with pytest.raises(gl.RepairError, match="--source"):
        gl.repair_bam(lifted, str(tmp_path / "u.out.bam"), work_dir=str(tmp_path / "w"))
    _stash, _stats, _records_, _hdr = _lift_and_repair(
        tmp_path, _source_header(), [lambda h: _read(h, "r", seq=_seq(50))], name="done")
    with pytest.raises(gl.RepairError, match="already"):
        gl.repair_bam(str(tmp_path / "done.out.bam"), str(tmp_path / "again.bam"),
                      work_dir=str(tmp_path / "w2"))


# --- @PG history ---------------------------------------------------------------

def test_pg_chain_is_linear_through_the_lift(tmp_path):
    header = _source_header(extra_pg=[
        {"ID": "fiberhmm-call", "PN": "fiberhmm-call", "VN": "3.0.2", "PP": "minimap2",
         "CL": "fiberhmm-call -i a.bam", "DS": "coord=molecular"}])
    _stash, _stats, _records_, hdr = _lift_and_repair(
        tmp_path, header, [lambda h: _read(h, "r", seq=_seq(50))])
    pgs = hdr["PG"]
    ids = [pg["ID"] for pg in pgs]
    assert ids == ["minimap2", "fiberhmm-call", "ft.1", "0", "ft.2", "fiberhmm-lift"]
    parents = {pg["ID"]: pg.get("PP") for pg in pgs}
    assert parents == {"minimap2": None, "fiberhmm-call": "minimap2", "ft.1": "fiberhmm-call",
                       "0": "ft.1", "ft.2": "0", "fiberhmm-lift": "ft.2"}
    lift = pgs[-1]
    assert lift["PN"] == "fiberhmm-lift" and lift["CL"] == "fiberhmm-lift test"
    for token in ("coord=molecular", "source=SRC", "target=TGT", "graph=g.gbz", "calmd=no"):
        assert token in lift["DS"]
    assert gl.header_lift_state(hdr) == {"lift": [], "repaired": True}


def test_a_repaired_bam_can_be_lifted_again(tmp_path):
    _stash, _stats, _records_, _hdr = _lift_and_repair(
        tmp_path, _source_header(), [lambda h: _read(h, "r", seq=_seq(50), mapq=5)])
    first = tmp_path / "x.out.bam"
    stashed = tmp_path / "again.stash.bam"
    gl.stash_bam(str(first), str(stashed))
    lifted = _fake_lift(stashed, tmp_path / "again.lifted.bam", flip=True)
    _recs, lifted_hdr = _records(lifted)
    assert gl.header_lift_state(lifted_hdr) == {"lift": ["0", "ft.1", "ft.2"], "repaired": False}
    gl.repair_bam(lifted, str(tmp_path / "again.out.bam"), work_dir=str(tmp_path / "w"),
                  pg_args={"mode": "lift"})
    (rec,), hdr = _records(tmp_path / "again.out.bam")
    ids = [pg["ID"] for pg in hdr["PG"]]
    assert len(ids) == len(set(ids))
    chain, parent = [], hdr["PG"][-1]["ID"]
    by_id = {pg["ID"]: pg for pg in hdr["PG"]}
    while parent:
        chain.append(parent)
        parent = by_id[parent].get("PP")
    assert chain == ["fiberhmm-lift.2", "ft.2.2", "0.2", "ft.1.2", "fiberhmm-lift",
                     "ft.2", "0", "ft.1", "minimap2"]
    assert gl.header_lift_state(hdr)["repaired"]
    assert rec.get_tag("Yp").startswith("tgt:") and rec.mapping_quality == 5


def test_lift_pg_detection():
    assert not gl.is_lift_pg({"ID": "0", "PN": "vg"})      # vg giraffe writes it too
    assert gl.is_lift_pg({"ID": "vg", "PN": "vg", "CL": "vg surject -x g.gbz -"})
    assert gl.is_lift_pg({"ID": "ft.3", "PN": "fibertools-rs", "CL": "ft pg-lift --bam -i x"})
    assert not gl.is_lift_pg({"ID": "ft", "PN": "fibertools-rs", "CL": "ft add-nucleosomes x"})
    assert not gl.is_lift_pg({"ID": "samtools", "PN": "samtools", "CL": "samtools view vg.bam"})


# --- BED12 -------------------------------------------------------------------------

def test_bed12_explode_and_reassemble_handles_forward_flip_and_lost_blocks():
    rows = [
        "srcA\t100\t300\tread1\t10\t+\t100\t300\t0\t2\t50,100\t0,100\t1,2\t5",
        "srcA\t500\t560\tread2\t0\t-\t500\t560\t0\t1\t60\t0\t7\t5",
        "srcA\t700\t900\tread3\t0\t+\t700\t900\t0\t2\t50,50\t0,150\t1,2\t5",
    ]
    bed6, parsed, header = gl.explode_bed12(["track name=x"] + rows)
    assert header == ["track name=x"] and len(bed6) == 5
    lifted = []
    for line in bed6:
        chrom, start, end, name, score, strand = line.split("\t")
        start, end = int(start), int(end)
        row = int(name.split("|")[0])
        if row == 0:      # forward, +1000
            lifted.append(f"tgt\t{start + 1000}\t{end + 1000}\t{name}\t{score}\t{strand}")
        elif row == 1:    # strand flip (single block): strand stays as ft wrote it
            lifted.append(f"tgt\t{5000 - end}\t{5000 - start}\t{name}\t{score}\t{strand}")
        elif name.endswith("|0"):   # row 2 loses its second block
            lifted.append(f"tgt\t{start}\t{end}\t{name}\t{score}\t{strand}")
    out, stats = gl.reassemble_bed12(lifted, parsed)
    assert stats.rows_out == 2 and stats.rows_lost_blocks == 1
    assert out[0].split("\t")[:12] == ["tgt", "1100", "1300", "read1", "10", "+", "1100", "1300",
                                       "0", "2", "50,100", "0,100"]
    # A flipped multi-block row reverses its blocks and per-block columns.
    flipped = [f"tgt\t{5000 - int(l.split(chr(9))[2])}\t{5000 - int(l.split(chr(9))[1])}\t"
               f"{l.split(chr(9))[3]}\t0\t+" for l in bed6 if l.split("\t")[3].startswith("0|")]
    out, stats = gl.reassemble_bed12(flipped, parsed[:1])
    fields = out[0].split("\t")
    assert stats.rows_flipped == 1 and fields[5] == "-"
    assert fields[1:3] == ["4700", "4900"] and fields[10] == "100,50" and fields[11] == "0,150"
    assert fields[12] == "2,1" and fields[13] == "5"


# --- fiberhmm-check -------------------------------------------------------------

def test_check_flags_an_unrepaired_lift_and_clears_a_repaired_one(tmp_path):
    from fiberhmm.advisories import check_header

    src = _write(tmp_path / "k.bam", _source_header(), [lambda h: _read(h, "r", seq=_seq(50))])
    _records_, lifted_hdr = _records(_fake_lift(src, tmp_path / "k.lifted.bam"))
    (advisory,) = [a for a in check_header(lifted_hdr) if a.id == "graph-lift-unrepaired"]
    assert advisory.severity == "rerun-recommended" and advisory.needs_rerun
    assert "fiberhmm-lift --repair-only" in advisory.fix
    assert "0, ft.1, ft.2" in advisory.evidence[0]
    _stash, _stats, _recs, repaired_hdr = _lift_and_repair(
        tmp_path, _source_header(), [lambda h: _read(h, "r", seq=_seq(50))], name="k2")
    assert not [a for a in check_header(repaired_hdr) if a.id == "graph-lift-unrepaired"]
    assert not [a for a in check_header(_source_header()) if a.id == "graph-lift-unrepaired"]


# --- CLI -----------------------------------------------------------------------------

def test_cli_reports_missing_tools_and_a_missing_source(tmp_path, capsys, monkeypatch):
    from fiberhmm.cli import lift

    src = _write(tmp_path / "s.bam", _source_header(), [lambda h: _read(h, "r", seq=_seq(50))])
    graph = tmp_path / "g.gbz"
    graph.write_bytes(b"")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    code = lift.main(["-i", src, "-o", str(tmp_path / "o.bam"), "-g", str(graph),
                      "--prefix", "SRC#0#", "--target", "TGT"])
    assert code == 2 and "`vg` not found" in capsys.readouterr().err
    lifted = _fake_lift(src, tmp_path / "l.bam")
    code = lift.main(["--repair-only", "-i", lifted, "-o", str(tmp_path / "o.bam")])
    assert code == 2 and "--source" in capsys.readouterr().err
    code = lift.main(["--repair-only", "-i", lifted, "--source", src,
                      "-o", str(tmp_path / "o.bam")])
    assert code == 0 and (tmp_path / "o.bam.bai").exists()
    assert not list(tmp_path.glob(".fiberhmm-lift-*"))   # work dir cleaned up


def test_target_fasta_must_hold_the_contigs_reads_landed_on(tmp_path):
    header = {"SQ": [{"SN": "tgt", "LN": 10}, {"SN": "alt", "LN": 5}]}
    fasta = tmp_path / "t.fa"
    fasta.write_text(">tgt\nACGTACGTAC\n")
    pysam.faidx(str(fasta))
    gl.validate_target_fasta(header, str(fasta), {"tgt"})
    with pytest.raises(gl.RepairError, match="alt"):
        gl.validate_target_fasta(header, str(fasta), {"tgt", "alt"})
    with pytest.raises(gl.RepairError, match="12"):
        gl.validate_target_fasta({"SQ": [{"SN": "tgt", "LN": 12}]}, str(fasta), set())


# --- audit regressions -----------------------------------------------------------

def _decoded(read):
    """Modified positions in SEQ order (pysam's own MM/ML decoder)."""
    return sorted(pos for calls in read.modified_bases.values() for pos, _q in calls)


def _fill_one(tmp_path, primary, supp, name="f"):
    src = _write(tmp_path / f"{name}.bam", _source_header(), [primary, supp])
    stats = gl.stash_bam(src, str(tmp_path / f"{name}.stash.bam"))
    assert stats.filled_supplementary == 1
    records, _ = _records(tmp_path / f"{name}.stash.bam")
    return next(r for r in records if r.is_supplementary)


def test_filled_supplementary_keeps_modifications_on_their_bases(tmp_path):
    # Primary C+m call on molecule base 3 (the third C). The supplementary
    # carries a deamination (Y) at base 1, so the filled SEQ has one C fewer
    # before base 3: MM must be re-encoded, not copied.
    primary = lambda h: _read(h, "m", seq="CCACGC", cigar="6M",
                              tags=[("MM", "C+m?,2;", "Z"), ("ML", array.array("B", [250]))])
    supp = lambda h: _read(h, "m", flag=0x800, pos=500, cigar="1H5M", seq="YACGC")
    rec = _fill_one(tmp_path, primary, supp)
    assert rec.query_sequence == "CYACGC" and rec.get_tag("MN") == 6
    assert _decoded(rec) == [3]
    # A call on a base the supplementary deaminated cannot stay: dropped.
    primary2 = lambda h: _read(h, "n", seq="CCACGC", cigar="6M",
                               tags=[("MM", "C+m?,1;", "Z"), ("ML", array.array("B", [250]))])
    supp2 = lambda h: _read(h, "n", flag=0x800, pos=500, cigar="1H5M", seq="YACGC")
    assert _decoded(_fill_one(tmp_path, primary2, supp2, name="g")) == []


def test_repair_only_matches_on_sequence_not_length(tmp_path):
    # Two 40H20M supplementaries of one 60 bp molecule: A forward over
    # molecule 40..59, B reverse over molecule 0..19. Same lengths and spans.
    import random

    rng = random.Random(5)
    molecule = "".join(rng.choice("ACGT") for _ in range(60))
    src = _write(tmp_path / "m.src.bam", _source_header(), [
        lambda h: _read(h, "z", seq=molecule, cigar="60M"),
        lambda h: _read(h, "z", flag=0x800, pos=600, cigar="40H20M", seq=molecule[40:],
                        tags=[("st", "CT", "Z")]),
        lambda h: _read(h, "z", flag=0x810, pos=900, cigar="40H20M",
                        seq=gl.revcomp(molecule)[40:], tags=[("st", "GA", "Z")]),
    ])
    lifted = _fake_lift(src, tmp_path / "m.lifted.bam", flip=True)
    match = gl.stash_from_source(lifted, src, str(tmp_path / "m.restash.bam"))
    assert match.matched == 3 and match.ambiguous == 0
    gl.repair_bam(str(tmp_path / "m.restash.bam"), str(tmp_path / "m.out.bam"),
                  work_dir=str(tmp_path / "mw"), pg_args={"mode": "repair-only"})
    records, _ = _records(tmp_path / "m.out.bam")
    by_locus = {r.get_tag("Yp"): r for r in records if r.is_supplementary}
    a, b = by_locus["srcA:601:+"], by_locus["srcA:901:-"]
    assert b.cigarstring == "20M40H" and b.get_tag("st") == "CT" and not b.is_reverse
    assert a.cigarstring == "20M40H" and a.get_tag("st") == "GA" and a.is_reverse


def test_repair_only_refuses_ambiguous_matches(tmp_path):
    seq = _seq(30, seed=2)
    src = _write(tmp_path / "amb2.bam", _source_header(), [
        lambda h: _read(h, "d", seq=seq, cigar="30M"),
        lambda h: _read(h, "d", flag=0x800, pos=700, cigar="30M", seq=seq),
    ])
    lifted = _fake_lift(src, tmp_path / "amb2.lifted.bam")
    match = gl.stash_from_source(lifted, src, str(tmp_path / "amb2.restash.bam"))
    assert match.ambiguous == 2 and match.matched == 0


def _walk(pgs, start):
    by_id = {pg["ID"]: pg for pg in pgs}
    chain, node = [], start
    while node:
        assert node not in chain, f"cycle through {node}"
        chain.append(node)
        node = by_id.get(node, {}).get("PP")
    return chain


def test_rechain_has_no_cycle_when_tools_ran_after_the_lift():
    pgs = [{"ID": "0", "PN": "vg"},
           {"ID": "minimap2", "PN": "minimap2"},
           {"ID": "fiberhmm-call", "PN": "fiberhmm-call", "PP": "minimap2"},
           {"ID": "ft.1", "PN": "fibertools-rs", "CL": "ft pg-pansn in.bam --prefix S#0#"},
           {"ID": "ft.2", "PN": "fibertools-rs", "PP": "ft.1",
            "CL": "ft pg-pansn - out.bam --strip"},
           {"ID": "samtools", "PN": "samtools", "PP": "ft.2", "CL": "samtools sort out.bam"},
           {"ID": "samtools.1", "PN": "samtools", "PP": "samtools", "CL": "samtools index"}]
    out = gl.rechain_lift_history({"PG": pgs})["PG"]
    for pg in out:
        _walk(out, pg["ID"])
    assert _walk(out, out[-1]["ID"]) == ["samtools.1", "samtools", "ft.2", "0", "ft.1",
                                         "fiberhmm-call", "minimap2"]


def test_native_vg_alignment_is_not_a_lift():
    from fiberhmm.advisories import check_header

    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 100}],
              "PG": [{"ID": "0", "PN": "vg"},
                     {"ID": "fiberhmm-call", "PN": "fiberhmm-call", "PP": "0", "VN": "3.0.2",
                      "CL": "fiberhmm-call -i giraffe.bam", "DS": "coord=molecular"}]}
    assert gl.header_lift_state(header) == {"lift": [], "repaired": False}
    assert not [a for a in check_header(header) if a.id == "graph-lift-unrepaired"]


def test_output_on_another_file_system_and_failures_keep_results(tmp_path, monkeypatch, capsys):
    import errno
    import os as _os

    from fiberhmm.cli import lift

    src = _write(tmp_path / "fs.bam", _source_header(), [lambda h: _read(h, "r", seq=_seq(50))])
    gl.stash_bam(src, str(tmp_path / "fs.stash.bam"))
    lifted = _fake_lift(tmp_path / "fs.stash.bam", tmp_path / "fs.lifted.bam")
    real_rename = _os.rename

    def cross_device(a, b, *args, **kwargs):
        if _os.path.dirname(_os.path.abspath(a)) != _os.path.dirname(_os.path.abspath(b)):
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real_rename(a, b, *args, **kwargs)

    monkeypatch.setattr(_os, "rename", cross_device)
    (tmp_path / "out").mkdir()
    gl.repair_bam(lifted, str(tmp_path / "out" / "fs.out.bam"), work_dir=str(tmp_path / "scratch"),
                  pg_args={"mode": "lift"})
    assert (tmp_path / "out" / "fs.out.bam").exists()
    assert (tmp_path / "out" / "fs.out.bam.bai").exists()
    monkeypatch.setattr(_os, "rename", real_rename)

    def broken(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(gl, "_finalize", broken)
    raw = _fake_lift(src, tmp_path / "fs.raw.bam")
    with pytest.raises(OSError):
        lift.main(["--repair-only", "-i", raw, "--source", src, "-o", str(tmp_path / "x.bam"),
                   "--tmp-dir", str(tmp_path / "out")])
    kept = list((tmp_path / "out").glob(".fiberhmm-lift-*"))
    assert kept and any(p.suffix == ".bam" for p in kept[0].iterdir())
    assert "kept in" in capsys.readouterr().err


def test_bed12_single_block_row_takes_the_lifted_strand():
    rows = ["srcA\t100\t160\tr\t0\t+\t100\t160\t0\t1\t60\t0"]
    _bed6, parsed, _ = gl.explode_bed12(rows)
    out, stats = gl.reassemble_bed12(["tgt\t900\t960\t0|0\t0\t-"], parsed)
    assert stats.rows_flipped == 1 and out[0].split("\t")[5] == "-"
    out, stats = gl.reassemble_bed12(["tgt\t900\t960\t0|0\t0\t+"], parsed)
    assert stats.rows_flipped == 0 and out[0].split("\t")[5] == "+"


# --- audit round 2 -------------------------------------------------------------------

def test_assignment_rechecks_ties_after_a_source_is_taken():
    lifted = [(0, 300, "k", (0, 100)), (1, 300, "k", (20, 150))]
    a, b, c = (300, "k", (0, 100), "A"), (300, "k", (100, 200), "B"), (300, "k", (100, 200), "C")
    for sources in ([a, b, c], [c, b, a]):
        result = gl._assign(lifted, sources)
        assert sources[result[0]][3] == "A" and result[1] is None


def _hc_supp(tmp_path, name, molecule, supp_tags, primary_tags=(), cigar="4H6M"):
    src = _write(tmp_path / f"{name}.bam", _source_header(), [
        lambda h: _read(h, name, seq=molecule, cigar=f"{len(molecule)}M", tags=list(primary_tags)),
        lambda h: _read(h, name, flag=0x800, pos=500, cigar=cigar, seq=molecule[4:],
                        tags=list(supp_tags))])
    stats = gl.stash_bam(src, str(tmp_path / f"{name}.stash.bam"))
    records, _ = _records(tmp_path / f"{name}.stash.bam")
    return stats, next((r for r in records if r.is_supplementary), None)


def test_native_vg_root_survives_lift_repair_and_relift(tmp_path):
    from fiberhmm.advisories import check_header

    native = {"ID": "0", "PN": "vg"}
    header = _source_header()
    header["PG"] = [native, {"ID": "fiberhmm-call", "PN": "fiberhmm-call", "PP": "0",
                             "VN": "3.0.2", "CL": "fiberhmm-call -i g.bam",
                             "DS": "coord=molecular"}]
    src = _write(tmp_path / "v.bam", header, [lambda h: _read(h, "r", seq=_seq(50))])
    source_pgs = _records(src)[1]["PG"]
    assert gl.header_lift_state(source_pgs and {"PG": source_pgs}) == {"lift": [],
                                                                         "repaired": False}
    current = src
    for round_ in (1, 2):
        stashed = tmp_path / f"v{round_}.stash.bam"
        gl.stash_bam(str(current), str(stashed))
        pgs_before = _records(current)[1]["PG"]
        lifted = _fake_lift(stashed, tmp_path / f"v{round_}.lifted.bam")
        lifted_pgs = _records(lifted)[1]["PG"]
        for known in (pgs_before, None):
            state = gl.header_lift_state({"PG": lifted_pgs}, known)
            assert state["lift"] == ["0", "ft.1", "ft.2"] and not state["repaired"]
        out = tmp_path / f"v{round_}.out.bam"
        gl.repair_bam(lifted, str(out), work_dir=str(tmp_path / f"vw{round_}"),
                      pg_args={"mode": "lift"}, source_pgs=pgs_before)
        _recs, hdr = _records(out)
        ids = [pg["ID"] for pg in hdr["PG"]]
        assert len(ids) == len(set(ids))
        assert gl.header_lift_state(hdr) == {"lift": [], "repaired": True}
        assert not [a for a in check_header(hdr) if a.id == "graph-lift-unrepaired"]
        chain = _walk(hdr["PG"], hdr["PG"][-1]["ID"])
        assert chain[-2:] == ["fiberhmm-call", "0"] and len(chain) == 2 + 4 * round_
        current = out


def test_finalize_publishes_bam_and_index_together(tmp_path, monkeypatch):
    from fiberhmm.inference import bam_output

    def bam(name, n):
        return _write(tmp_path / name, _source_header(),
                      [lambda h, k=k: _read(h, f"r{k}", pos=10 + k, seq=_seq(50)) for k in range(n)])

    final = tmp_path / "out" / "final.bam"
    final.parent.mkdir()
    old = bam("old.bam", 2)
    import shutil
    shutil.copy(old, final)
    pysam.index(str(final))
    (tmp_path / "out" / "final.bam.csi").write_bytes(b"stale")
    old_bai = (tmp_path / "out" / "final.bam.bai").read_bytes()
    old_bam = final.read_bytes()

    # Failure at the BAM's own publication step: the previous pair is restored.
    real_replace = bam_output.os.replace

    calls = []

    def failing(a, b, *args, **kwargs):
        if str(b) == str(final) and not calls:      # the publication step only
            calls.append(b)
            raise OSError("injected")
        return real_replace(a, b, *args, **kwargs)

    work = tmp_path / "work"
    work.mkdir()
    new = bam("work/new.bam", 3)
    monkeypatch.setattr(bam_output.os, "replace", failing)
    with pytest.raises(OSError, match="injected"):
        gl._finalize(new, str(final), index=True)
    monkeypatch.setattr(bam_output.os, "replace", real_replace)
    assert final.read_bytes() == old_bam
    assert (tmp_path / "out" / "final.bam.bai").read_bytes() == old_bai
    assert Path(new).exists()                         # the finished BAM is kept
    assert not list((tmp_path / "out").glob(".final.bam.*tmp*"))

    gl._finalize(new, str(final), index=True)
    assert len(_records(final)[0]) == 3
    assert not (tmp_path / "out" / "final.bam.csi").exists()
    with pysam.AlignmentFile(str(final)) as handle:
        assert handle.count() == 3                    # the index matches the new BAM


# --- hard-clipped supplementaries keep their own calls -----------------------------

def _stash_pair(tmp_path, name, primary_kw, supp_kw, header=None):
    src = _write(tmp_path / f"{name}.bam", header or _source_header(), [
        lambda h: _read(h, name, **primary_kw),
        lambda h: _read(h, name, flag=supp_kw.pop("flag", 0x800), pos=500, **supp_kw)])
    stats = gl.stash_bam(src, str(tmp_path / f"{name}.stash.bam"))
    records, _ = _records(tmp_path / f"{name}.stash.bam")
    supp = next((r for r in records if r.is_supplementary), None)
    prim = next(r for r in records if not r.is_supplementary)
    return stats, prim, supp


def test_modification_grammar_is_re_encoded_from_the_primary(tmp_path):
    cases = [
        ("ACCC", "C+C?,0;", [200], {("C", 0, "C"): [(1, 200)]}),
        ("ACCC", "C+17596?,1;", [9], {("C", 0, 17596): [(2, 9)]}),
        ("ACCC", "C+mh?,0;", [1, 2], {("C", 0, "m"): [(1, 1)], ("C", 0, "h"): [(1, 2)]}),
        ("ATTT", "U+U?,0;", [200], {("T", 0, "U"): [(1, 200)]}),     # finding 5
    ]
    for k, (seq, mm, ml, expected) in enumerate(cases):
        stats, _prim, supp = _stash_pair(
            tmp_path, f"g{k}", dict(seq=seq, cigar="4M", tags=[("MM", mm, "Z"),
                                                              ("ML", array.array("B", ml))]),
            dict(cigar="1H3M", seq=seq[1:]))
        assert supp.modified_bases_forward == expected, mm
        assert stats.mods_dropped == 0
    assert supp.get_tag("MM") == "U+U?,0;"
    # MM without ML stays without ML.
    _stats, _prim, supp = _stash_pair(
        tmp_path, "l", dict(seq="ACCC", cigar="4M", tags=[("MM", "C+m?,0;", "Z")]),
        dict(cigar="1H3M", seq="CCC"))
    assert supp.get_tag("MM") == "C+m?,0;" and not supp.has_tag("ML")


# --- audit round 4 -------------------------------------------------------------------

def test_only_groups_gaining_unassessed_bases_lose_their_dot_mode(tmp_path):
    # The record turns molecule C 2 into Y: the C group loses a candidate
    # (its '.' claim about the remaining Cs stands), the A group is untouched.
    stats, _prim, supp = _stash_pair(
        tmp_path, "d", dict(seq="AACCCAAA", cigar="8M",
                            tags=[("MM", "A+a.,0;C+m.,2;", "Z"),
                                  ("ML", array.array("B", [200, 210]))]),
        dict(cigar="1H7M", seq="AYCCAAA"))
    assert supp.get_tag("MM") == "A+a.,0;C+m.,1;" and list(supp.get_tag("ML")) == [200, 210]
    assert stats.mods_dropped == 0
    # The primary has Y where the record has C: the C group gains a base the
    # primary never assessed, so only it becomes '?'.
    stats, _prim, supp = _stash_pair(
        tmp_path, "e", dict(seq="AACYCAAA", cigar="8M",
                            tags=[("MM", "A+a.,0;C+m.,0;", "Z"),
                                  ("ML", array.array("B", [200, 210]))]),
        dict(cigar="1H7M", seq="ACCCAAA"))
    assert supp.get_tag("MM") == "A+a.,0;C+m?,0;" and stats.mods_dropped == 0
    stats, _prim, supp = _stash_pair(
        tmp_path, "f", dict(seq="AACCCAAA", cigar="8M",
                            tags=[("MM", "C+m.,0;", "Z"), ("ML", array.array("B", [99]))]),
        dict(cigar="1H7M", seq="AYCCAAA"))
    assert supp.get_tag("MM") == "C+m.;" and stats.mods_dropped == 1


def test_failed_frame_conversion_refuses_the_run(tmp_path):
    # Reverse record, SEQ-frame header, MA needing AQ bytes it lacks.
    src = _write(tmp_path / "cf.bam", _source_header(), [
        lambda h: _read(h, "bad", flag=0x10, seq=_seq(10), cigar="10M",
                        tags=[("MA", "10;nuc+Q:2-2", "Z")])])
    with pytest.raises(gl.AnnotationError, match="bad"):
        gl.stash_bam(src, str(tmp_path / "cf.stash.bam"))


def test_mm_and_ml_spellings_resolve_independently(tmp_path):
    _stats, _prim, supp = _stash_pair(
        tmp_path, "mx", dict(seq="ACCC", cigar="4M", tags=[("Mm", "C+m?,0;", "Z"),
                                                           ("ML", array.array("B", [200]))]),
        dict(cigar="1H3M", seq="CCC"))
    assert supp.modified_bases_forward == {("C", 0, "m"): [(1, 200)]}


def test_malformed_primary_annotations_refuse_the_run(tmp_path):
    cases = [
        [("MA", "4;nuc.QQQ:2-2", "Z"), ("AQ", array.array("B", [10]))],     # 1 of 3 AQ bytes
        [("MA", "4;nuc.:1-1,3-1", "Z"), ("AN", "x", "Z")],                   # 2 calls, 1 name
    ]
    for k, tags in enumerate(cases):
        src = _write(tmp_path / f"mf{k}.bam", _source_header(co=["fiberhmm:coord=molecular"]), [
            lambda h, tags=tags: _read(h, "p", flag=0x10, seq="ACCC", cigar="4M", tags=tags),
            lambda h: _read(h, "p", flag=0x800, pos=500, cigar="1H3M", seq="CCC")])
        with pytest.raises(gl.AnnotationError, match="'p'"):
            gl.stash_bam(src, str(tmp_path / f"mf{k}.stash.bam"))


def test_repeated_identical_mm_heads_are_merged(tmp_path):
    stats, _prim, supp = _stash_pair(
        tmp_path, "rp", dict(seq="ACCC", cigar="4M", tags=[("MM", "C+m?,0;C+m?,1;", "Z"),
                                                           ("ML", array.array("B", [10, 20]))]),
        dict(cigar="1H3M", seq="CCC"))
    assert supp is not None and stats.excluded_unverifiable == 0
    calls = supp.modified_bases_forward[("C", 0, "m")]
    assert sorted(calls) == [(1, 10), (2, 20)]


# --- audit round 5: supplementaries are called on their own bases ---------------------

def _split_daf(reverse):
    """A 40 bp DAF molecule split like minimap2 does: primary 20S20M over
    molecule 20..39, supplementary 21M19H over molecule 0..20 (forward), each
    called separately (FiberHMM alignments=primary-supplementary)."""
    import random

    rng = random.Random(11)
    molecule = "".join(rng.choice("ACGT") for _ in range(40))
    stored = gl.revcomp(molecule) if reverse else molecule
    supp_seq = stored[:21] if not reverse else stored[:21]
    primary = dict(seq=molecule, cigar="20S20M",
                   tags=[("MA", "40;nuc.:25-10;msp.:36-4", "Z"), ("ns", array.array("I", [24])),
                         ("nl", array.array("I", [10])), ("st", "CT", "Z")])
    supp = dict(flag=0x810 if reverse else 0x800, cigar="21M19H", seq=supp_seq,
                tags=[("MA", "21;nuc.:2-5,10-6;msp.:7-3", "Z"),
                      ("ns", array.array("I", [1, 9])), ("nl", array.array("I", [5, 6])),
                      ("as", array.array("I", [6])), ("al", array.array("I", [3]))])
    return molecule, primary, supp


@pytest.mark.parametrize("reverse", [False, True])
def test_split_daf_supplementary_keeps_its_own_calls(tmp_path, reverse):
    _molecule, primary, supp_kw = _split_daf(reverse)
    header = _source_header(co=["fiberhmm:coord=molecular"])
    stats, prim, supp = _stash_pair(tmp_path, f"s{int(reverse)}", primary, supp_kw, header)
    assert stats.filled_supplementary == 1 and stats.excluded_supplementary == 0
    offset = 19 if reverse else 0          # molecule offset of the stored bases
    assert supp.get_tag("MA") == (f"40;nuc.:{2 + offset}-5,{10 + offset}-6;msp.:{7 + offset}-3")
    assert list(supp.get_tag("ns")) == [1 + offset, 9 + offset]
    assert list(supp.get_tag("as")) == [6 + offset] and list(supp.get_tag("al")) == [3]
    assert supp.get_tag("MA") != prim.get_tag("MA")
    assert supp.infer_read_length() == 40 and supp.query_length == 40


def test_legacy_tags_follow_ma_scope(tmp_path):
    # Round-3 finding 1: legacy values equal to the primary's follow the
    # supplementary's partial MA, not the primary.
    molecule = "ACGTACGTAC"
    legacy = [("ns", array.array("I", [1])), ("nl", array.array("I", [2]))]
    _stats, _prim, supp = _stash_pair(
        tmp_path, "a", dict(seq=molecule, cigar="10M", tags=list(legacy)),
        dict(cigar="4H6M", seq=molecule[4:], tags=[("MA", "6;nuc.:2-2", "Z"), *legacy]))
    assert supp.get_tag("MA") == "10;nuc.:6-2" and list(supp.get_tag("ns")) == [5]


def test_whole_molecule_reverse_tags_are_converted_with_the_molecule_length(tmp_path):
    # Round-3 finding 2: SEQ-frame header, reverse 4H6M with whole-molecule
    # MA and legacy tags: both mirrored over 10 bases.
    molecule = "ACGTACGTAC"
    stored = gl.revcomp(molecule)
    stats, _prim, supp = _stash_pair(
        tmp_path, "r", dict(flag=0x10, seq=stored, cigar="10M"),
        dict(flag=0x810, cigar="4H6M", seq=stored[4:],
             tags=[("MA", "10;nuc+:2-2", "Z"), ("ns", array.array("I", [1])),
                   ("nl", array.array("I", [2]))]))
    assert supp.get_tag("MA") == "10;nuc+:8-2" and list(supp.get_tag("ns")) == [7]


def test_supplementary_without_calls_stays_without(tmp_path):
    molecule = "ACGTACGTAC"
    _stats, prim, supp = _stash_pair(
        tmp_path, "n", dict(seq=molecule, cigar="10M", tags=[("MA", "10;nuc.:6-2", "Z")]),
        dict(cigar="4H6M", seq=molecule[4:]))
    assert prim.has_tag("MA") and not supp.has_tag("MA")


def test_undecidable_scope_is_excluded(tmp_path):
    molecule = "ACGTACGTAC"
    stats, _prim, supp = _stash_pair(
        tmp_path, "x", dict(seq=molecule, cigar="10M"),
        dict(cigar="4H6M", seq=molecule[4:], tags=[("MA", "8;nuc.:2-2", "Z")]))
    assert supp is None and stats.excluded_unverifiable == 1


def test_supplementary_modifications_its_own_first(tmp_path):
    bad = [("MM", "C+m?,9;", "Z"), ("ML", array.array("B", [1]))]
    # Own MM valid on the stored bases (MN = 3): moved; the primary's unused.
    stats, _prim, supp = _stash_pair(
        tmp_path, "o", dict(seq="CCCC", cigar="4M", tags=bad),
        dict(cigar="1H3M", seq="CCC", tags=[("MM", "C+m?,0;", "Z"),
                                           ("ML", array.array("B", [7])), ("MN", 3)]))
    assert supp.modified_bases_forward == {("C", 0, "m"): [(1, 7)]} and supp.get_tag("MN") == 4
    # Own MM for the whole molecule (MN = 4): kept as it is.
    _stats, _prim, supp = _stash_pair(
        tmp_path, "w", dict(seq="ACCC", cigar="4M", tags=[("MM", "A+a?,0;", "Z"),
                                                          ("ML", array.array("B", [180]))]),
        dict(cigar="1H3M", seq="CCC", tags=[("MM", "C+m?,1;", "Z"),
                                           ("ML", array.array("B", [200])), ("MN", 4)]))
    assert supp.modified_bases_forward == {("C", 0, "m"): [(2, 200)]}
    # No own MM: the primary's, which must then be decodable.
    stats, _prim, supp = _stash_pair(
        tmp_path, "u", dict(seq="CCCC", cigar="4M", tags=bad), dict(cigar="1H3M", seq="CCC"))
    assert supp is None and stats.excluded_unverifiable == 1


# --- MAPQ mode -----------------------------------------------------------------------

def test_mapq_mode_option_and_pg_record(tmp_path, capsys):
    from fiberhmm.cli import lift, qc

    assert gl.parse_mapq_mode("source") == "source"
    assert gl.parse_mapq_mode("FIXED:060") == "fixed:60"
    for bad in ("fixed:256", "fixed:", "max", "fixed:-1"):
        with pytest.raises(ValueError):
            gl.parse_mapq_mode(bad)
    with pytest.raises(SystemExit):
        lift.build_parser().parse_args(["-i", "x", "-o", "y", "--mapq", "fixed:999"])

    src = _write(tmp_path / "mq.bam", _source_header(),
                 [lambda h: _read(h, "r", seq=_seq(50), mapq=3)])
    outputs = {}
    for mode in ("source", "fixed:60"):
        raw = _fake_lift(src, tmp_path / f"mq.{mode}.lifted.bam")
        out = tmp_path / f"mq.{mode.replace(':', '')}.bam"
        assert lift.main(["--repair-only", "-i", raw, "--source", src, "-o", str(out),
                          "--mapq", mode]) == 0
        (rec,), hdr = _records(out)
        outputs[mode] = out
        assert rec.get_tag("Yq") == 3
        assert rec.mapping_quality == (3 if mode == "source" else 60)
        assert f"mapq={mode} " in hdr["PG"][-1]["DS"]
        assert gl.lift_mapq_mode(hdr) == mode
    assert gl.lift_mapq_mode(_source_header()) is None

    capsys.readouterr()
    qc._note_lifted_mapq([str(outputs["source"]), str(outputs["fixed:60"]), src], 20)
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "keeps the MAPQ of its source assembly" in err[0]
    assert str(outputs["source"]) in err[0] and "--min-mapq 20" in err[0]


# --- native engine (graphlift liftbam) -----------------------------------------------

def _liftbam_like(stashed, lifted, *, split_at=None, second_flag=0x800):
    """What liftbam does to a stashed record (without the binary): move it to
    ``tgt`` +500, keep hard clips, and with ``split_at`` cut the alignment
    into a primary and a 0x800 record (whole SEQ, soft clips); a liftbam
    @PG chained on the source history."""
    with pysam.AlignmentFile(str(stashed), "rb", check_sq=False) as bam:
        hdr = bam.header.to_dict()
        reads = list(bam.fetch(until_eof=True))
    hdr["SQ"] = [{"SN": "tgt", "LN": TGT_LEN}]
    hdr["HD"] = {"VN": "1.6", "SO": "unsorted"}
    hdr["PG"] = list(hdr.get("PG", [])) + [
        {"ID": "liftbam", "PN": "liftbam", "PP": hdr["PG"][-1]["ID"], "VN": "0.4.0",
         "CL": "liftbam -i stashed.bam -m map.chain.graphlift.json -o lifted.bam"}]
    with pysam.AlignmentFile(str(lifted), "wb", header=hdr) as out:
        for src in reads:
            pieces = [list(src.cigartuples)]
            if split_at is not None:
                ops = list(src.cigartuples)
                lead = [ops.pop(0)] if ops[0][0] == 5 else []
                tail = [ops.pop()] if ops[-1][0] == 5 else []
                (op, n), = ops
                pieces = [lead + [(0, split_at), (4, n - split_at)] + tail,
                          lead + [(4, split_at), (0, n - split_at)] + tail]
            for k, cigar in enumerate(pieces):
                rec = pysam.AlignedSegment(out.header)
                rec.query_name = src.query_name
                rec.flag = src.flag | (second_flag if k else 0)
                rec.reference_id = 0
                rec.reference_start = src.reference_start + 500 + (2000 if k else 0)
                rec.mapping_quality = src.mapping_quality
                rec.cigartuples = cigar
                rec.query_sequence = src.query_sequence
                rec.query_qualities = src.query_qualities
                rec.set_tags(src.get_tags())
                out.write(rec)
    return str(lifted)


def test_repair_keeps_liftbam_splits_and_hard_clips(tmp_path):
    src = _write(tmp_path / "n.src.bam", _source_header(), [
        lambda h: _read(h, "p", cigar="10H40M", seq=_seq(40), mapq=33)])
    stashed = tmp_path / "n.stash.bam"
    gl.stash_bam(src, str(stashed))
    lifted = _liftbam_like(stashed, tmp_path / "n.lifted.bam", split_at=15)
    assert gl.header_lift_state(pysam.AlignmentFile(lifted, check_sq=False).header)["lift"] \
        == ["liftbam"]
    out = tmp_path / "n.out.bam"
    stats = gl.repair_bam(lifted, str(out), work_dir=str(tmp_path / "n.work"),
                          pg_args={"mode": "lift", "engine": "native", "graph": "m.json",
                                   "command": "fiberhmm-lift test"})
    records, hdr = _records(out)
    assert [(r.is_supplementary, r.cigarstring) for r in records] == \
        [(False, "10H15M25S"), (True, "10H15S25M")]
    assert stats.hard_clips_restored == 0 and stats.orphan_supplementary_dropped == 0
    assert all(r.mapping_quality == 33 for r in records)
    assert hdr["PG"][-1]["PN"] == "fiberhmm-lift" and hdr["PG"][-1]["PP"] == "liftbam"
    assert "engine=native" in hdr["PG"][-1]["DS"]
    assert gl.is_lift_pg({"ID": "x", "CL": "/opt/bin/liftbam -i a.bam"})
    assert not gl.is_lift_pg({"ID": "x", "PN": "minimap2", "CL": "minimap2 -a"})


def _stub_liftbam(tmp_path, *, exit_code=0):
    """A stand-in ``liftbam``: runs _liftbam_like and writes a report."""
    script = tmp_path / "liftbam"
    here = Path(__file__).resolve().parent
    script.write_text(f"""#!{__import__('sys').executable}
import json, sys
sys.path[:0] = [{str(here)!r}, {str(here.parent)!r}]
args = sys.argv[1:]
get = lambda flag: args[args.index(flag) + 1]
if {exit_code}:
    sys.stderr.write("liftbam: boom\\n"); sys.exit({exit_code})
import test_graph_lift as t
t._liftbam_like(get("-i"), get("-o"))
assert get("--footprint-frame") == "molecular"
json.dump({{"lifted": 1, "unlifted": 2, "unlifted_by_reason": {{"no_projection": 2}},
           "split": 0, "split_pieces": 0, "aligned_bases": 50, "projected_bases": 48}},
          open(get("--report"), "w"))
""")
    script.chmod(0o755)
    return str(script)


def _lift_map(tmp_path):
    path = tmp_path / "m.chain.graphlift.json"
    path.write_text('{"format": "graphlift.sidecar", "version": 2, '
                    '"sources": [{"name": "srcA", "path": "SRC#1#srcA"}], '
                    '"targets": [{"name": "tgt", "path": "TGT#0#tgt"}]}')
    return str(path)


def test_cli_native_engine_is_the_default_with_a_map(tmp_path, capsys):
    from fiberhmm.cli import lift

    src = _write(tmp_path / "s.bam", _source_header(), [lambda h: _read(h, "r", seq=_seq(50))])
    out = tmp_path / "o.bam"
    code = lift.main(["-i", src, "-o", str(out), "--map", _lift_map(tmp_path),
                      "--liftbam", _stub_liftbam(tmp_path)])
    err = capsys.readouterr().err
    assert code == 0, err
    assert "2 could not be lifted (no_projection=2)" in err
    records, hdr = _records(out)
    assert len(records) == 1 and records[0].reference_start == 600
    ds = hdr["PG"][-1]["DS"]
    assert "engine=native" in ds and "source=SRC" in ds and "target=TGT" in ds
    assert "graph=m.chain.graphlift.json" in ds and "unlifted=2" in ds


def test_cli_engine_choice_and_errors(tmp_path, capsys):
    from fiberhmm.cli import lift

    src = _write(tmp_path / "s.bam", _source_header(), [lambda h: _read(h, "r", seq=_seq(50))])
    lmap = _lift_map(tmp_path)
    base = ["-i", src, "-o", str(tmp_path / "o.bam")]
    with pytest.raises(SystemExit):
        lift.main(base + ["--engine", "native"])                        # no --map
    assert "--map" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        lift.main(base + ["--map", lmap, "--engine", "vg"])             # vg needs the graph
    assert "--graph" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        lift.main(base + ["--map", lmap, "--min-match", "0.5"])          # BAMs have no min-match
    assert "--min-match is for --bed" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        lift.main(base + ["--map", lmap, "--bed", "--min-match", "2"])
    assert "--min-match must be between 0 and 1" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        lift.main(base + ["-g", lmap, "--prefix", "S#0#", "--target", "T", "--unlifted", "u.bam"])
    assert "--unlifted is for the native engine" in capsys.readouterr().err
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert lift.main(base + ["--map", str(bad), "--liftbam", _stub_liftbam(tmp_path)]) == 2
    assert "not a graphlift map" in capsys.readouterr().err
    failing = tmp_path / "f"
    failing.mkdir()
    assert lift.main(base + ["--map", lmap, "--liftbam", _stub_liftbam(failing, exit_code=3)]) == 2
    assert "liftbam failed (exit 3)" in capsys.readouterr().err
    empty = tmp_path / "empty"
    empty.mkdir()
    import os
    old = os.environ["PATH"]
    os.environ["PATH"] = str(empty)
    try:
        assert lift.main(base + ["--map", lmap]) == 2
    finally:
        os.environ["PATH"] = old
    assert "`liftbam` not found" in capsys.readouterr().err


def test_repair_keeps_liftbam_secondary_placements(tmp_path):
    src = _write(tmp_path / "d.src.bam", _source_header(), [
        lambda h: _read(h, "p", cigar="40M", seq=_seq(40))])
    stashed = tmp_path / "d.stash.bam"
    gl.stash_bam(src, str(stashed))
    lifted = _liftbam_like(stashed, tmp_path / "d.lifted.bam", split_at=15, second_flag=0x100)
    out = tmp_path / "d.out.bam"
    gl.repair_bam(lifted, str(out), work_dir=str(tmp_path / "d.work"),
                  pg_args={"mode": "lift", "command": "t"})
    records, _ = _records(out)
    assert sorted(r.flag for r in records) == [0, 0x100]


def test_repair_only_refuses_a_bam_lifted_by_liftbam_directly(tmp_path, capsys):
    from fiberhmm.cli import lift

    src = _write(tmp_path / "s.bam", _source_header(), [lambda h: _read(h, "r", seq=_seq(50))])
    lifted = _liftbam_like(src, tmp_path / "l.bam")          # no stash: liftbam run directly
    code = lift.main(["--repair-only", "-i", lifted, "--source", src, "-o", str(tmp_path / "o.bam")])
    assert code == 2 and "lifted by graphlift's liftbam directly" in capsys.readouterr().err


# --- --bed with the native engine (liftbed) ---------------------------------------

def _stub_liftbed(tmp_path, *, exit_code=0):
    """A stand-in ``liftbed``: srcA x -> tgt 5000 - x (a - chain: strand swapped), except
    rows named ``*|1`` of row 1 and rows on srcB (unlifted); writes a report."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "liftbed"
    script.write_text(f"""#!{__import__('sys').executable}
import json, sys
args = sys.argv[1:]
get = lambda flag: args[args.index(flag) + 1] if flag in args else None
if {exit_code}:
    sys.stderr.write("liftbed: boom\\n"); sys.exit({exit_code})
assert get("--split") == "best"
out, unl = open(get("-o"), "w"), open(get("--unlifted"), "w") if get("--unlifted") else None
n = lifted = 0
for line in open(get("-i")):
    f = line.rstrip("\\n").split("\\t")
    if line.startswith(("#", "track")):
        out.write(line); continue
    n += 1
    if f[0] != "srcA" or (len(f) > 3 and f[3] == "1|1"):
        unl and unl.write("#no_projection\\n" + line); continue
    lifted += 1
    f[0], f[1], f[2] = "tgt", str(5000 - int(f[2])), str(5000 - int(f[1]))
    if len(f) > 5:
        f[5] = {{"+": "-", "-": "+"}}.get(f[5], f[5])
    out.write("\\t".join(f) + "\\n")
json.dump({{"rows_in": n, "lifted": lifted, "unlifted": n - lifted, "split": 0, "partial": 0,
           "strand_flipped": lifted, "unlifted_by_reason": {{"no_projection": n - lifted}} if n > lifted else {{}},
           "min_match": get("--min-match")}}, open(get("--report"), "w"))
""")
    script.chmod(0o755)
    return str(script)


def test_cli_bed_native_lifts_bed12_blocks_and_writes_dropped_rows(tmp_path, capsys):
    from fiberhmm.cli import lift

    bed = tmp_path / "in.bed"
    bed.write_text("track name=n\n"
                   "srcA\t100\t300\tr0\t10\t+\t100\t300\t0\t2\t50,100\t0,100\t1,2\n"
                   "srcA\t500\t700\tr1\t0\t+\t500\t700\t0\t2\t50,50\t0,150\t1,2\n")
    out, unl = tmp_path / "out.bed", tmp_path / "unl.bed"
    code = lift.main(["--bed", "-i", str(bed), "-o", str(out), "--map", _lift_map(tmp_path),
                      "--liftbed", _stub_liftbed(tmp_path / "bin"), "--unlifted", str(unl)])
    err = capsys.readouterr().err
    assert code == 0, err
    assert "1 of 2 BED12 rows lifted (1 lost a block" in err and "3 of 4 blocks lifted" in err
    lines = out.read_text().splitlines()
    assert lines[0] == "track name=n"
    # Blocks [100, 150) and [200, 300) -> [4850, 4900) and [4700, 4800): reversed, strand -.
    assert lines[1].split("\t") == ["tgt", "4700", "4900", "r0", "10", "-", "4700", "4900", "0",
                                     "2", "100,50", "0,150", "2,1"]
    assert unl.read_text() == ("#block_no_projection\n"
                               "srcA\t500\t700\tr1\t0\t+\t500\t700\t0\t2\t50,50\t0,150\t1,2\n")


def test_cli_bed_native_plain_bed_rows_and_liftbed_lookup(tmp_path, capsys):
    from fiberhmm.cli import lift

    bed = tmp_path / "peaks.bed"
    bed.write_text("srcA\t100\t200\tp1\t5\t+\textra\nsrcB\t1\t2\tp2\t0\t.\textra\n")
    out = tmp_path / "out.bed"
    bindir = tmp_path / "bin"
    _stub_liftbed(bindir)
    # liftbed is found next to an explicit --liftbam.
    code = lift.main(["--bed", "-i", str(bed), "-o", str(out), "--map", _lift_map(tmp_path),
                      "--liftbam", str(bindir / "liftbam"), "--min-match", "0.5"])
    err = capsys.readouterr().err
    assert code == 0, err
    assert "liftbed lifted 1 of 2 BED rows" in err and "(no_projection=1)" in err
    assert out.read_text() == "tgt\t4800\t4900\tp1\t5\t-\textra\n"
    mixed = tmp_path / "mixed.bed"
    mixed.write_text("srcA\t1\t2\nsrcA\t100\t160\tr\t0\t+\t100\t160\t0\t1\t60\t0\n")
    assert lift.main(["--bed", "-i", str(mixed), "-o", str(out), "--map", _lift_map(tmp_path),
                      "--liftbed", str(bindir / "liftbed")]) == 2
    assert "some rows are BED12" in capsys.readouterr().err
    failing = _stub_liftbed(tmp_path / "f", exit_code=3)
    assert lift.main(["--bed", "-i", str(bed), "-o", str(out), "--map", _lift_map(tmp_path),
                      "--liftbed", failing]) == 2
    assert "liftbed failed (exit 3)" in capsys.readouterr().err


def test_cli_bed_vg_engine_needs_the_source_fasta(tmp_path, capsys):
    from fiberhmm.cli import lift

    bed = tmp_path / "in.bed"
    bed.write_text("srcA\t100\t200\n")
    code = lift.main(["--bed", "-i", str(bed), "-o", str(tmp_path / "o.bed"), "-g", str(bed),
                      "--prefix", "S#0#", "--target", "T"])
    assert code == 2 and "--source-fasta" in capsys.readouterr().err


def test_bed12_reassembly_reports_dropped_rows():
    rows = ["srcA\t100\t300\tr\t0\t+\t100\t300\t0\t2\t50,100\t0,100",
            "srcA\t500\t560\tq\t0\t+\t500\t560\t0\t1\t60\t0"]
    _bed6, parsed, _ = gl.explode_bed12(rows)
    dropped = []
    out, stats = gl.reassemble_bed12(["tgt\t1100\t1150\t0|0\t0\t+", "chrX\t1200\t1300\t0|1\t0\t+"],
                                     parsed, dropped=dropped)
    assert out == [] and dropped == [(0, "split_contigs"), (1, "lost_block")]
