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
                                   ("MA", "50;nuc+Q:5-10", "Z")])]
    stash, stats, records, hdr = _lift_and_repair(tmp_path, header, reads)
    (rec,) = records
    assert rec.reference_name == "tgt" and rec.reference_start == 600
    assert rec.flag == 0 and rec.mapping_quality == 7
    assert rec.get_tag("Yq") == 7 and rec.get_tag("Yp") == "srcA:101:+"
    for tag in ("Yf", "Yh", "MD", "NM", "SA"):
        assert not rec.has_tag(tag)
    assert rec.get_tag("st") == "CT" and stats.st_swapped == 0
    assert hdr["HD"]["SO"] == "coordinate"
    assert (tmp_path / "x.out.bam.bai").exists()


def test_strand_flip_swaps_st_and_keeps_molecular_annotations(tmp_path):
    header = _source_header(co=["fiberhmm:coord=molecular"])
    seq = "ACYTTGCYAA" * 5
    reads = [lambda h: _read(h, "daf", seq=seq, tags=[("st", "CT", "Z"),
                                                     ("MA", "50;nuc.Q:5-10", "Z"),
                                                     ("AQ", array.array("B", [200]))])]
    _stash, stats, records, _hdr = _lift_and_repair(tmp_path, header, reads, flip=True)
    (rec,) = records
    assert rec.is_reverse and rec.query_sequence == gl.revcomp(seq)
    assert "R" in rec.query_sequence and "Y" not in rec.query_sequence
    assert rec.get_tag("st") == "GA"
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
    src = _write(tmp_path / "amb.bam", header, [
        lambda h: _read(h, "r", seq=_seq(50), tags=[("ns", array.array("I", [1])),
                                                   ("nl", array.array("I", [5]))])])
    with pytest.raises(gl.FrameError):
        gl.stash_bam(src, str(tmp_path / "amb.stash.bam"))


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
    # MM/ML/MN now describe the whole molecule: the primary's.
    assert supp.get_tag("MM") == "C+m?,0;" and supp.get_tag("MN") == 60


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
    assert stash.excluded_supplementary == 1
    assert [r.query_name for r in records] == ["other"]
    pg = next(pg for pg in hdr["PG"] if pg["ID"] == "fiberhmm-lift")
    assert "excluded_supplementary=1" in pg["DS"]
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
    assert gl.header_lift_state(hdr) == {"lift": ["ft.1", "0", "ft.2"], "repaired": True}


def test_lift_pg_detection():
    assert gl.is_lift_pg({"ID": "0", "PN": "vg"})
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
