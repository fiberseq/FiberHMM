"""fiberhmm-pipeline: reference naming, circular origins, markers, contract.

Synthetic fixtures only. The end-to-end tests need the ``minimap2`` program
(or the ``mappy`` module) and are skipped without it.
"""
from __future__ import annotations

import array
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pysam
import pytest

from fiberhmm.io.circular_bed import fold_bed12_row, fold_circular_bed
from fiberhmm.pipeline import aligner as mm2
from fiberhmm.pipeline import runner
from fiberhmm.pipeline.circular import compute_md_nm, hard_clip, merge_origin_pieces
from fiberhmm.pipeline.progress import ProgressReporter
from fiberhmm.pipeline.reference import (
    REFERENCE_COMMENT_PREFIX,
    declared_references,
    decorate_header,
    fiberbrowser_contig_name,
    header_matches_reference,
    parse_reference_comment,
    prepare_reference,
    read_plasmid_map,
    sequence_md5,
)
from fiberhmm.pipeline.runner import (
    Pipeline,
    PipelineConfig,
    PipelineError,
    densest_region,
    expand_read_inputs,
    parse_call_progress,
    parse_region,
)

REPO = Path(__file__).resolve().parents[1]
HAS_MINIMAP2 = shutil.which("minimap2") is not None
needs_minimap2 = pytest.mark.skipif(not HAS_MINIMAP2, reason="minimap2 not installed")

COMP = str.maketrans("ACGT", "TGCA")


def revcomp(seq: str) -> str:
    return seq.translate(COMP)[::-1]


def random_seq(n: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


# ---------------------------------------------------------------------------
# Plasmid maps and naming
# ---------------------------------------------------------------------------

def write_snapgene(path: Path, sequence: str, circular: bool = True) -> None:
    cookie = b"SnapGene" + bytes([0, 1, 0, 15, 0, 19])
    dna = bytes([1 if circular else 0]) + sequence.encode()
    payload = (bytes([9]) + len(cookie).to_bytes(4, "big") + cookie
               + bytes([0]) + len(dna).to_bytes(4, "big") + dna
               + bytes([6]) + (5).to_bytes(4, "big") + b"notes")
    path.write_bytes(payload)


def write_genbank(path: Path, name: str, sequence: str, topology: str = "circular") -> None:
    lines = [f"LOCUS       {name}  {len(sequence)} bp    DNA     {topology} SYN 01-JAN-2026",
             "FEATURES             Location/Qualifiers",
             "ORIGIN"]
    for i in range(0, len(sequence), 60):
        chunk = sequence[i:i + 60].lower()
        lines.append(f"{i + 1:>9} " + " ".join(chunk[j:j + 10] for j in range(0, 60, 10)))
    lines.append("//")
    path.write_text("\n".join(lines) + "\n")


def test_contig_name_mirrors_fiberbrowser_rules(tmp_path):
    # SnapGene: the file name stem, cut at whitespace, sanitised, stripped.
    assert fiberbrowser_contig_name("/x/L-HH (v2).dna") == "L-HH"
    assert fiberbrowser_contig_name("/x/my+plasmid#1.dna") == "my_plasmid_1"
    assert fiberbrowser_contig_name("/x/__.dna") == "plasmid"
    # GenBank / EMBL: the LOCUS / ID name (sanitised, not stripped).
    assert fiberbrowser_contig_name("/x/file.gbk", "pUC19") == "pUC19"
    assert fiberbrowser_contig_name("/x/file.gb", "a/b") == "a_b"
    assert fiberbrowser_contig_name("/x/My map.gb", None) == "My"
    assert fiberbrowser_contig_name("/x/file.embl", "X1", "embl") == "X1"


def test_snapgene_and_genbank_maps(tmp_path):
    seq = random_seq(600, 1)
    dna = tmp_path / "Construct A.dna"
    write_snapgene(dna, seq, circular=True)
    parsed = read_plasmid_map(str(dna))
    assert (parsed.name, parsed.sequence, parsed.circular) == ("Construct", seq, True)
    assert not parsed.warnings
    write_snapgene(dna, seq, circular=False)
    assert read_plasmid_map(str(dna)).circular is False

    gb = tmp_path / "whatever.gbk"
    write_genbank(gb, "pTest", seq, "linear")
    parsed = read_plasmid_map(str(gb))
    assert (parsed.name, parsed.sequence, parsed.circular) == ("pTest", seq, False)


def test_prepare_reference_from_map_writes_named_fasta(tmp_path):
    seq = random_seq(800, 2)
    gb = tmp_path / "in" / "map.gb"
    gb.parent.mkdir()
    write_genbank(gb, "pRef", seq)
    info = prepare_reference(str(gb), str(tmp_path / "out"))
    assert Path(info.fasta).name == "pRef.fa"
    assert pysam.FastaFile(info.fasta).fetch("pRef") == seq
    assert info.contigs[0].circular and info.contigs[0].md5 == sequence_md5(seq)
    assert Path(info.plasmid_map).read_bytes() == gb.read_bytes()
    assert info.source_format == "genbank"


def test_reference_comment_roundtrip_and_header(tmp_path):
    seq = random_seq(500, 3)
    gb = tmp_path / "odd name;x=y.gb"
    write_genbank(gb, "pRef", seq)
    info = prepare_reference(str(gb), str(tmp_path / "out"))
    header = decorate_header({"SQ": [{"SN": "pRef", "LN": len(seq)}], "CO": ["keep me"]}, info)
    assert header["SQ"][0]["M5"] == sequence_md5(seq)
    assert header["SQ"][0]["TP"] == "circular"
    line = [c for c in header["CO"] if c.startswith(REFERENCE_COMMENT_PREFIX)]
    assert len(line) == 1 and "keep me" in header["CO"]
    parsed = parse_reference_comment(line[0])
    assert parsed["contig"] == "pRef" and parsed["length"] == len(seq)
    assert parsed["topology"] == "circular" and parsed["source"] == gb.name
    assert parsed["source_format"] == "genbank" and parsed["md5"] == sequence_md5(seq)
    assert declared_references(header) == [parsed]
    assert header_matches_reference(header, info) == (True, "")
    bad = {"SQ": [{"SN": "pRef", "LN": len(seq) + 1}]}
    assert header_matches_reference(bad, info)[0] is False


# ---------------------------------------------------------------------------
# Origin-spanning reads
# ---------------------------------------------------------------------------

def _record(header, name, seq, pos, cigar, reverse=False, supplementary=False, mapq=60):
    read = pysam.AlignedSegment(header)
    read.query_name = name
    read.flag = (0x10 if reverse else 0) | (0x800 if supplementary else 0)
    read.reference_id = 0
    read.reference_start = pos
    read.mapping_quality = mapq
    read.query_sequence = seq
    read.query_qualities = pysam.qualitystring_to_array("I" * len(seq))
    read.cigartuples = cigar
    read.set_tag("RG", "s", "Z")
    return read


def _pieces(ref, start, length, reverse=False, deaminate=()):
    """A molecule of ``length`` bp from ``start`` around a circular ``ref``,
    split as minimap2 -Y reports it: (primary at the end, supplementary at 0)."""
    L = len(ref)
    mol = list((ref + ref)[start:start + length])
    for i in deaminate:
        if mol[i] == "C":
            mol[i] = "T"
    record_seq = "".join(mol)  # reference-forward orientation (SEQ as stored)
    head = L - start
    tail = length - head
    header = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "p", "LN": L}]})
    primary = _record(header, "r", record_seq, start, [(0, head), (4, tail)], reverse)
    sup = _record(header, "r", record_seq, 0, [(4, head), (0, tail)], reverse, True, 7)
    for rec in (primary, sup):
        md, nm = compute_md_nm(record_seq, rec.cigartuples, ref, rec.reference_start)
        rec.set_tag("MD", md)
        rec.set_tag("NM", nm)
    return primary, sup, record_seq


@pytest.mark.parametrize("reverse", [False, True])
def test_merge_origin_pieces_joins_one_record(reverse):
    ref = random_seq(2000, 4)
    deam = [i for i in range(0, 700, 7)]
    primary, sup, seq = _pieces(ref, 1700, 700, reverse, deam)
    merged = merge_origin_pieces(primary, sup, ref)
    assert merged is not None
    assert merged.reference_start == 1700
    assert merged.cigartuples == [(0, 700)]
    assert merged.reference_end == 2400  # past LN: SAM 1.4 circular form
    assert merged.query_sequence == seq
    assert merged.is_reverse == reverse and not merged.is_supplementary
    md, nm = compute_md_nm(seq, [(0, 700)], ref, 1700)
    assert merged.get_tag("MD") == md and merged.get_tag("NM") == nm
    assert nm == sum(1 for a, b in zip(seq, (ref + ref)[1700:2400]) if a != b)
    assert merged.mapping_quality == primary.mapping_quality
    # Also when the supplementary is the end piece.
    swapped = merge_origin_pieces(sup, primary, ref)
    assert swapped is not None and swapped.reference_start == 1700


def test_merge_handles_junction_overlap_gap_and_one_circle_cap():
    ref = random_seq(1000, 5)
    header = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "p", "LN": 1000}]})
    seq = (ref + ref)[900:1300]
    # The pieces claim read bases 95-99 twice: the primary's placement wins
    # and the supplementary resumes after it (ref 0-4 become a deletion).
    primary = _record(header, "r", seq, 900, [(0, 100), (4, 300)])
    sup = _record(header, "r", seq, 0, [(4, 95), (0, 305)], supplementary=True)
    merged = merge_origin_pieces(primary, sup, ref)
    assert merged.reference_start == 900
    assert merged.cigartuples == [(0, 100), (2, 5), (0, 300)]
    assert merged.get_tag("NM") == compute_md_nm(seq, merged.cigartuples, ref, 900)[1]
    # Read bases between the pieces (an insertion at the junction).
    ins_seq = seq[:100] + "GATCA" + seq[100:]
    primary = _record(header, "r", ins_seq, 900, [(0, 100), (4, 305)])
    sup = _record(header, "r", ins_seq, 0, [(4, 105), (0, 300)], supplementary=True)
    merged = merge_origin_pieces(primary, sup, ref)
    assert merged.cigartuples == [(0, 100), (1, 5), (0, 300)]
    assert merged.get_tag("NM") == 5
    # A read longer than one circle is capped at LN; the rest is soft-clipped.
    long_seq = (ref * 3)[900:2300]  # 1400 bp
    primary = _record(header, "r", long_seq, 900, [(0, 100), (4, 1300)])
    sup = _record(header, "r", long_seq, 0, [(4, 100), (0, 1000), (4, 300)],
                  supplementary=True)
    merged = merge_origin_pieces(primary, sup, ref)
    assert merged.reference_end - merged.reference_start == 1000
    assert merged.cigartuples[-1] == (4, 400)
    removed = hard_clip(merged)
    assert removed == 400 and len(merged.query_sequence) == 1000
    assert merged.cigartuples == [(0, 1000), (5, 400)]


def test_merge_refuses_pieces_that_are_not_one_crossing():
    ref = random_seq(1000, 6)
    primary, sup, _ = _pieces(ref, 800, 400)
    far = _record(sup.header, "r", sup.query_sequence, 300, sup.cigartuples,
                  supplementary=True)
    assert merge_origin_pieces(primary, far, ref) is None
    flipped = _record(sup.header, "r", sup.query_sequence, 0, sup.cigartuples,
                      reverse=True, supplementary=True)
    assert merge_origin_pieces(primary, flipped, ref) is None


def test_fold_bed12_rows_at_origin(tmp_path):
    # Second block (980-1020) crosses the origin of a 1000 bp contig.
    row = "p\t950\t1020\tr1\t0\t+\t950\t1020\t0\t2\t20,40\t0,30\t5,6\t0\n"
    rows = fold_bed12_row(row, 1000, n_block_columns=1)
    assert rows == [
        "p\t950\t1000\tr1\t0\t+\t950\t1000\t0\t2\t20,20\t0,30\t5,6\t0",
        "p\t0\t20\tr1\t0\t+\t0\t20\t0\t1\t20\t0\t6\t0",
    ]
    bed = tmp_path / "t.bed"
    bed.write_text("p\t5\t50\ta\t0\t+\t5\t50\t0\t1\t45\t0\t0\n" + row
                   .replace("\t5,6", ""))
    assert fold_circular_bed(str(bed), {"p": 1000}) == 1
    starts = [int(line.split("\t")[1]) for line in bed.read_text().splitlines()]
    assert starts == sorted(starts) == [0, 5, 950]


# ---------------------------------------------------------------------------
# Small units
# ---------------------------------------------------------------------------

def test_region_and_inputs(tmp_path):
    assert parse_region("chr2L:1,001-2,000") == ("chr2L", 1000, 2000)
    with pytest.raises(PipelineError):
        parse_region("chr2L:5-1")
    (tmp_path / "a.fastq.gz").write_bytes(b"")
    (tmp_path / "b.fq").write_text("")
    (tmp_path / "notes.txt").write_text("")
    assert [Path(p).name for p in expand_read_inputs([str(tmp_path)])] == \
        ["a.fastq.gz", "b.fq"]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(PipelineError):
        expand_read_inputs([str(empty)])


def test_call_progress_translation(tmp_path):
    assert parse_call_progress("  Fused: 1,140 | Skipped: 362 | Inflight: 3 | 346 r/s "
                               "(avg 355)") == (1502, 346.0)
    assert parse_call_progress("Pass 1: 4,976 records") is None
    events = []
    progress = ProgressReporter(callback=events.append)
    relay = runner._ProgressRelay(str(tmp_path / "p.jsonl"), progress)
    relay.poll()  # no file yet
    (tmp_path / "p.jsonl").write_text(
        json.dumps({"event": "start", "regions_total": 10, "regions_done": 4,
                    "regions_reused": 4}) + "\n"
        + json.dumps({"event": "region", "regions_done": 5, "regions_total": 10,
                      "reads": 900, "reads_per_s": 450.0, "eta_s": 12.5}) + "\n"
        + '{"event": "region", "regions_do')  # a partial line is not read yet
    relay.poll()
    assert [(e["done"], e["total"], e["unit"]) for e in events] == \
        [(4, 10, "regions"), (5, 10, "regions")]
    assert events[1]["eta_s"] == 12.5 and "resuming" in events[0]["message"]


def test_missing_minimap2_message(monkeypatch, tmp_path):
    monkeypatch.setattr(mm2.shutil, "which", lambda name: None)
    monkeypatch.setitem(sys.modules, "mappy", None)  # import fails
    with pytest.raises(mm2.AlignerNotFound) as excinfo:
        mm2.find_aligner()
    text = str(excinfo.value)
    assert "brew install minimap2" in text and "pip install mappy" in text
    assert "conda install -c bioconda minimap2" in text

    ref = tmp_path / "ref.fa"
    ref.write_text(">c\n" + random_seq(300, 7) + "\n")
    reads = tmp_path / "r.fastq"
    reads.write_text("@a\nACGT\n+\nIIII\n")
    events = []
    config = PipelineConfig(reads=[str(reads)], reference=str(ref), enzyme="dddb",
                            outdir=str(tmp_path / "out"), quiet=True)
    with pytest.raises(mm2.AlignerNotFound):
        Pipeline(config, ProgressReporter(callback=events.append)).run()
    assert events[-1]["event"] == "done" and events[-1]["status"] == "error"
    assert "brew install minimap2" in events[-1]["hint"]


def test_completed_steps_are_skipped_and_changes_refused(tmp_path):
    from fiberhmm.pipeline.progress import write_marker
    out = tmp_path / "out"
    (out / "x.bam").parent.mkdir(parents=True)
    (out / "x.bam").write_bytes(b"BAM\x01 records")
    config = PipelineConfig(reads=["r.fastq"], reference="ref.fa", enzyme="dddb",
                            outdir=str(out), quiet=True)
    pipe = Pipeline(config, ProgressReporter())
    pipe._log_handle = None
    write_marker(str(out), "align", {"a": 1}, {"bam": str(out / "x.bam")}, {"kept": 3})
    assert pipe._is_complete("align", {"a": 1}) is True
    assert pipe.stats["align"] == {"kept": 3}
    with pytest.raises(PipelineError, match="different settings or inputs") as refused:
        pipe._is_complete("align", {"a": 2})
    assert "--redo align" in refused.value.hint
    # A kept output must still be what the step wrote: truncated, or changed at
    # the same size with its mtime restored, it is made again.
    stat = (out / "x.bam").stat()
    (out / "x.bam").write_bytes(b"BAM\x01 RECORDS")
    os.utime(out / "x.bam", ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert pipe._is_complete("align", {"a": 1}) is False
    (out / "x.bam").write_bytes(b"")
    assert pipe._is_complete("align", {"a": 1}) is False
    (out / "x.bam").unlink()  # outputs gone: run again
    assert pipe._is_complete("align", {"a": 1}) is False
    forced = Pipeline(PipelineConfig(reads=["r.fastq"], reference="ref.fa", enzyme="dddb",
                                     outdir=str(out), redo="align", quiet=True))
    assert forced._is_complete("align", {"a": 2}) is False


# ---------------------------------------------------------------------------
# End to end (minimap2)
# ---------------------------------------------------------------------------

def _deaminate(seq: str, rng: random.Random, rate: float = 0.08) -> str:
    """DddB-like single-strand deamination (C->T on one strand) with nucleosome-
    sized protected stretches, so calling finds footprints."""
    out = list(seq)
    for i, base in enumerate(out):
        protected = (i // 180) % 2 == 0
        if base == "C" and rng.random() < (0.005 if protected else rate):
            out[i] = "T"
    return "".join(out)


def _write_reads(path: Path, ref: str, n: int, seed: int, circular: bool,
                 length=(1500, 2600)) -> int:
    rng = random.Random(seed)
    crossing = 0
    with open(path, "w") as handle:
        for i in range(n):
            size = rng.randint(*length)
            start = rng.randrange(0, len(ref) - (0 if circular else size))
            mol = (ref + ref)[start:start + size]
            crossing += start + size > len(ref)
            strand_seq = mol if rng.random() < 0.5 else revcomp(mol)
            read = _deaminate(strand_seq, rng)
            handle.write(f"@read{i}\n{read}\n+\n{'I' * len(read)}\n")
    return crossing


@pytest.fixture
def plasmid_run(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBERHMM_MINIMAP2_INDEX_DIR", str(tmp_path / "mmi"))
    monkeypatch.setenv("FIBERHMM_NO_UPDATE_CHECK", "1")
    ref = random_seq(6000, 11)
    gb = tmp_path / "pTest map.gbk"
    write_genbank(gb, "pTest", ref)
    reads = tmp_path / "run1.fastq"
    crossing = _write_reads(reads, ref, 80, 12, circular=True)
    assert crossing >= 10
    return {"ref": ref, "map": gb, "reads": reads, "out": tmp_path / "out",
            "progress": tmp_path / "progress.jsonl"}


def _run_cli(args, timeout=240):
    env = dict(os.environ, PYTHONPATH=str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return subprocess.run([sys.executable, "-m", "fiberhmm.cli.pipeline", *args],
                          capture_output=True, text=True, timeout=timeout, env=env)


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _assert_contract(events: list[dict], outdir: Path) -> dict:
    """The progress stream and outputs.json follow PIPELINE_CONTRACT.md (v1)."""
    assert events[0]["event"] == "start"
    for key in ("version", "sample", "steps", "outdir", "settings"):
        assert key in events[0]
    steps = events[0]["steps"]
    assert steps[:5] == ["prepare_reference", "index", "align", "call", "qc"]
    for event in events:
        assert event["event"] in ("start", "step", "log", "done")
        assert isinstance(event["time"], float)
        if event["event"] == "step":
            assert event["step"] in steps
            assert event["status"] in ("running", "done", "skipped")
            for key, kind in (("done", int), ("total", int), ("unit", str),
                              ("rate", float), ("eta_s", float), ("message", str)):
                if key in event:
                    assert isinstance(event[key], kind), (key, event)
            if "unit" in event:
                assert event["unit"] in ("reads", "regions")
        if event["event"] == "log":
            assert event["level"] in ("info", "warning")
            assert isinstance(event["message"], str)
    finished = {e["step"] for e in events
                if e["event"] == "step" and e["status"] in ("done", "skipped")}
    assert finished == set(steps)
    last = events[-1]
    assert last["event"] == "done" and last["status"] == "ok"
    outputs = json.loads((outdir / "outputs.json").read_text())
    assert last["outputs"] == outputs
    assert outputs["schema"] == "fiberhmm.pipeline.outputs.v1"
    for key in ("sample", "enzyme", "aligned_bam", "called_bam", "qc_report", "tracks",
                "reference_fasta", "plasmid_map", "contigs", "open", "qc"):
        assert key in outputs
    assert set(outputs["open"]) >= {"fasta", "datasets", "plasmid_maps", "region"}
    assert outputs["open"]["datasets"] == [outputs["called_bam"]]
    for contig in outputs["contigs"]:
        assert set(contig) == {"name", "length", "circular", "md5"}
    for path in (outputs["aligned_bam"], outputs["called_bam"], outputs["reference_fasta"]):
        assert Path(path).exists()
    return outputs


@needs_minimap2
def test_plasmid_end_to_end_contract_and_rerun(plasmid_run):
    run = plasmid_run
    result = _run_cli([str(run["reads"]), "--reference", str(run["map"]), "--enzyme",
                       "dddb", "-o", str(run["out"]), "--sample", "s1", "-c", "2",
                       "--min-read-length", "500", "--tracks",
                       "--progress-json", str(run["progress"])])
    assert result.returncode == 0, result.stderr[-3000:]
    outputs = _assert_contract(_events(run["progress"]), run["out"])
    assert outputs["contigs"] == [{"name": "pTest", "length": 6000, "circular": True,
                                   "md5": sequence_md5(run["ref"])}]
    assert Path(outputs["plasmid_map"]).name == "pTest map.gbk"
    assert outputs["open"]["plasmid_maps"] == [outputs["plasmid_map"]]
    assert outputs["open"]["region"] is None
    assert outputs["tracks"] and all(Path(p).exists() for p in outputs["tracks"])
    qc = outputs["qc"]
    assert qc["verdicts"]["overall"] in ("PASS", "WARN", "FAIL", "INSUFFICIENT")
    # The called BAM is deduplicated (fiberhmm-dedup @PG "mode=flag"); QC grades
    # it as the DAF assay it is, not as "flag" (audit H1).
    assay = json.loads(Path(qc["json"]).read_text())["assay"]
    assert (assay["mode"], assay["enzyme"]) == ("daf", "dddb")
    curves = json.loads(Path(qc["curves"]).read_text())
    assert curves["schema"] == "fiberhmm.qc.curves.v1"
    assert {"signal_rate", "phasogram", "footprint_sizes", "duplicates"} <= set(curves)

    # Reads through the origin are one record past LN on a TP:circular @SQ.
    with pysam.AlignmentFile(outputs["called_bam"]) as bam:
        header = bam.header.to_dict()
        assert header["SQ"][0]["TP"] == "circular"
        assert header["SQ"][0]["M5"] == sequence_md5(run["ref"])
        assert declared_references(header)[0]["contig"] == "pTest"
        records = list(bam.fetch(until_eof=True))
    wrapped = [r for r in records if r.reference_end > 6000]
    assert wrapped and not any(r.is_supplementary for r in records)
    assert outputs["stats"]["align"]["origin_merged"] == len(wrapped)
    assert any(r.has_tag("MA") for r in wrapped)

    # Same command again: every step is skipped.
    again = _run_cli([str(run["reads"]), "--reference", str(run["map"]), "--enzyme",
                      "dddb", "-o", str(run["out"]), "--sample", "s1", "-c", "2",
                      "--min-read-length", "500", "--tracks",
                      "--progress-json", str(run["progress"]) + ".2"])
    assert again.returncode == 0, again.stderr[-3000:]
    events = _events(Path(str(run["progress"]) + ".2"))
    _assert_contract(events, run["out"])
    status = {e["step"]: e["status"] for e in events if e["event"] == "step"}
    assert status["align"] == status["call"] == status["qc"] == status["tracks"] == "skipped"

    # A changed calling setting in the same OUTDIR is refused.
    changed = _run_cli([str(run["reads"]), "--reference", str(run["map"]), "--enzyme",
                        "dddb", "-o", str(run["out"]), "--sample", "s1",
                        "--min-read-length", "800", "--progress-json",
                        str(run["progress"]) + ".3"])
    assert changed.returncode != 0
    last = _events(Path(str(run["progress"]) + ".3"))[-1]
    assert last["event"] == "done" and last["status"] == "error"
    assert "min_read_length" in last["error"] and "--redo call" in last["hint"]


@needs_minimap2
def test_pipeline_qc_carries_state_aware_rates(plasmid_run):
    """outputs.json's QC verdicts carry the in-MSP/outside-MSP fields
    (fiberhmm-qc schema 1.1) that FiberBrowser reads."""
    run = plasmid_run
    result = _run_cli([str(run["reads"]), "--reference", str(run["map"]), "--enzyme",
                       "dddb", "-o", str(run["out"]), "--sample", "s1", "-c", "2",
                       "--min-read-length", "500",
                       "--progress-json", str(run["progress"])])
    assert result.returncode == 0, result.stderr[-3000:]
    outputs = _assert_contract(_events(run["progress"]), run["out"])
    report = json.loads(Path(outputs["qc"]["json"]).read_text())
    assert report["schema_version"] == 1 and report["schema_minor_version"] == 1
    states = report["state_rates"]
    assert states["source"] == "tags" and states["available"]
    assert states["msp"]["aggregate_rate"] > states["outside_msp"]["aggregate_rate"]
    verdicts = outputs["qc"]["verdicts"]
    for key in ("efficiency", "background", "verdict_basis", "state_rates"):
        assert key in verdicts
    assert verdicts["state_rates"]["msp_rate"] == states["msp"]["median_per_read_rate"]
    curves = json.loads(Path(outputs["qc"]["curves"]).read_text())
    assert curves["verdicts"]["efficiency"] == report["efficiency"]["status"]
    assert curves["state_rates"]["source"] == "tags"


@needs_minimap2
def test_linear_reference_and_aligned_input(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBERHMM_MINIMAP2_INDEX_DIR", str(tmp_path / "mmi"))
    genome = random_seq(30000, 21)
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chrA\n" + genome[:20000] + "\n>chrB\n" + genome[20000:] + "\n")
    amplicon = genome[5000:9000]
    reads = tmp_path / "amp.fastq"
    rng = random.Random(22)
    with open(reads, "w") as handle:
        for i in range(60):
            mol = amplicon if i % 2 else revcomp(amplicon)
            seq = _deaminate(mol, rng)
            handle.write(f"@m{i}\n{seq}\n+\n{'I' * len(seq)}\n")
    out = tmp_path / "out"
    progress = tmp_path / "p.jsonl"
    result = _run_cli([str(reads), "--reference", str(fasta), "--enzyme", "dddb",
                       "-o", str(out), "-c", "2", "--min-read-length", "500",
                       "--progress-json", str(progress)])
    assert result.returncode == 0, result.stderr[-3000:]
    outputs = _assert_contract(_events(progress), out)
    assert outputs["plasmid_map"] is None
    assert [c["circular"] for c in outputs["contigs"]] == [False, False]
    assert outputs["open"]["region"] == "chrA:5001-9000"
    with pysam.AlignmentFile(outputs["aligned_bam"]) as bam:
        placed = [(r.reference_name, r.reference_start, r.reference_end)
                  for r in bam.fetch(until_eof=True)]
        assert bam.header.to_dict()["PG"][-1]["ID"] == "fiberhmm-pipeline"
    assert len(placed) == 60
    assert all(c == "chrA" and abs(s - 5000) <= 5 and abs(e - 9000) <= 5
               for c, s, e in placed)

    # The aligned BAM given back as input is called as it is (no realignment).
    out2 = tmp_path / "out2"
    progress2 = tmp_path / "p2.jsonl"
    result = _run_cli([outputs["aligned_bam"], "--reference", str(fasta), "--enzyme",
                       "dddb", "-o", str(out2), "-c", "2", "--min-read-length", "500",
                       "--no-qc", "--progress-json", str(progress2)])
    assert result.returncode == 0, result.stderr[-3000:]
    events = _events(progress2)
    status = {e["step"]: (e["status"], e.get("message")) for e in events
              if e["event"] == "step" and e["status"] != "running"}
    assert status["index"] == ("skipped", "input already aligned")
    assert status["align"] == ("skipped", "input already aligned")
    assert status["qc"][0] == "skipped"
    final = json.loads((out2 / "outputs.json").read_text())
    assert final["aligned_bam"] == outputs["aligned_bam"]
    assert densest_region(final["called_bam"]) == "chrA:5001-9000"


@needs_minimap2
def test_sigterm_cancels_and_leaves_a_resumable_outdir(plasmid_run):
    run = plasmid_run
    env = dict(os.environ, PYTHONPATH=str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    args = [sys.executable, "-m", "fiberhmm.cli.pipeline", str(run["reads"]),
            "--reference", str(run["map"]), "--enzyme", "dddb", "-o", str(run["out"]),
            "-c", "2", "-q", "--progress-json", str(run["progress"])]
    proc = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.time() + 120
    while time.time() < deadline:
        if run["progress"].exists() and any(
                e.get("step") == "call" and e.get("status") == "running"
                for e in _events(run["progress"])):
            break
        time.sleep(0.2)
    else:
        proc.kill()
        pytest.fail("the call step never started")
    proc.send_signal(signal.SIGTERM)
    proc.communicate(timeout=60)
    assert proc.returncode == 128 + signal.SIGTERM
    last = _events(run["progress"])[-1]
    assert (last["event"], last["status"], last["error"]) == ("done", "error", "cancelled")
    assert (run["out"] / ".fiberhmm-pipeline" / "align.done").exists()
    assert not (run["out"] / ".fiberhmm-pipeline" / "call.done").exists()
    # The next run continues: alignment is kept.
    result = _run_cli([str(run["reads"]), "--reference", str(run["map"]), "--enzyme",
                       "dddb", "-o", str(run["out"]), "-c", "2", "--no-qc",
                       "--progress-json", str(run["progress"]) + ".2"])
    # --no-qc is a QC setting only, so the earlier steps still match.
    assert result.returncode == 0, result.stderr[-3000:]
    status = {e["step"]: e["status"] for e in _events(Path(str(run["progress"]) + ".2"))
              if e["event"] == "step" and e["status"] != "running"}
    assert status["align"] == "skipped" and status["call"] == "done"


def test_read_inputs_fastq_tags_and_bam_to_fastq(tmp_path):
    tagged = tmp_path / "mods.fastq"
    tagged.write_text("@r1\tMM:Z:A+a?,0;\tML:B:C,200\nACGT\n+\nIIII\n")
    plain = tmp_path / "plain.fastq"
    plain.write_text("@r1 runid=abc ch=5\nACGT\n+\nIIII\n")
    assert mm2.fastq_has_sam_tags(str(tagged)) is True
    assert mm2.fastq_has_sam_tags(str(plain)) is False

    header = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "c", "LN": 100}]})
    bam = tmp_path / "in.bam"
    with pysam.AlignmentFile(str(bam), "wb", header=header) as out:
        forward = _record(header, "f", "AACCG", 10, [(0, 5)])
        forward.set_tag("MM", "C+m?,0;")
        forward.set_tag("ML", array.array("B", [9]))
        out.write(forward)
        reverse = _record(header, "r", "AACCG", 20, [(0, 5)], reverse=True)
        reverse.query_qualities = pysam.qualitystring_to_array("ABCDE")
        out.write(reverse)
        out.write(_record(header, "f", "AACCG", 40, [(4, 2), (0, 3)],
                          supplementary=True))
    assert mm2.classify_read_file(str(bam)).kind == "aligned"
    records = [r.decode() for r in mm2.bam_records_as_fastq(str(bam), with_tags=True)]
    assert records[0] == "@f\tMM:Z:C+m?,0;\tML:B:C,9\nAACCG\n+\nIIIII\n"
    # Reverse-strand records go back to sequencing orientation.
    assert records[1] == "@r\nCGGTT\n+\nEDCBA\n"
    assert len(records) == 2
    fed = b"".join(mm2.ReadFeeder([mm2.classify_read_file(str(plain)),
                                   mm2.classify_read_file(str(bam))], False).chunks())
    names = [e[0] for e in mm2.iter_fastq_entries([fed])]
    # Every input record gets its own internal name (serial~name); the output
    # restores the original name.
    assert names == ["0~r1", "1~f", "2~r"]
    assert [mm2.original_name(n) for n in names] == ["r1", "f", "r"]


# ---------------------------------------------------------------------------
# Split reads (3.0 SV default): supplementary records and soft clips
# ---------------------------------------------------------------------------

def _split_group(circular_contig=False):
    header = pysam.AlignmentHeader.from_dict(
        {"SQ": [{"SN": "c1", "LN": 50_000}, {"SN": "te", "LN": 5_000}]})
    seq = random_seq(3_000, 21)
    prim = _record(header, "r", seq, 1_000, [(0, 2_000), (4, 1_000)])
    supp = _record(header, "r", seq, 100, [(4, 2_000), (0, 1_000)], supplementary=True)
    supp.reference_id = 1
    sec = _record(header, "r", seq, 9_000, [(0, 2_000), (4, 1_000)])
    sec.flag |= 0x100
    for rec in (prim, supp):
        rec.set_tag("SA", "x,1,+,10M,60,0;")
    circular = {"c1", "te"} if circular_contig else set()
    return [prim, supp, sec], circular


def _process(group, circular, **config):
    from types import SimpleNamespace
    cfg = runner.PipelineConfig(reads=[], reference="r.fa", enzyme="dddb", outdir="o",
                                **config)
    stub = SimpleNamespace(config=cfg, _overlaps_regions=lambda read: True)
    stats = {k: 0 for k in ("reads", "unmapped", "low_mapq", "outside_regions", "kept",
                            "supplementary_kept", "origin_merged", "hard_clipped_reads",
                            "hard_clipped_bases")}
    sequences = {name: "A" * 50_000 for name in circular}
    kept = runner.Pipeline._process_group(stub, group, circular, sequences, stats, "r")
    return kept, stats


def test_aligner_step_keeps_supplementary_and_soft_clips_on_linear_contigs():
    group, circular = _split_group()
    kept, stats = _process(group, circular)
    assert [r.is_supplementary for r in kept] == [False, True]
    assert [r.cigartuples[0][0] for r in kept] == [0, 4]      # soft clips kept
    assert all(r.has_tag("SA") for r in kept)                 # all pieces kept
    assert stats["kept"] == 1 and stats["supplementary_kept"] == 1
    assert stats["hard_clipped_reads"] == 0


def test_aligner_step_primary_only_and_hard_clip_options():
    group, circular = _split_group()
    kept, _ = _process(group, circular, alignments="primary")
    assert len(kept) == 1 and not kept[0].has_tag("SA")
    group, circular = _split_group()
    kept, stats = _process(group, circular, hard_clip=True)
    assert stats["hard_clipped_reads"] == 2
    assert all(op != 4 for r in kept for op, _n in r.cigartuples)


def test_daf_hard_clip_and_supplementary_drop_stay_on_circular_contigs():
    """Concatemer arms and copies of the same plasmid would annotate one
    molecule twice: DAF reads on circular contigs keep the 3.0 handling."""
    group, circular = _split_group(circular_contig=True)
    kept, stats = _process(group, circular)
    assert len(kept) == 1 and not kept[0].is_supplementary
    assert stats["hard_clipped_reads"] == 1
    assert runner.PipelineConfig(reads=[], reference="r", enzyme="hia5",
                                 outdir="o").resolved_hard_clip() == "off"
    assert runner.PipelineConfig(reads=[], reference="r", enzyme="ddda",
                                 outdir="o").resolved_hard_clip() == "circular"


def test_aligner_step_keeps_unique_supplementary_of_ambiguous_primary():
    group, circular = _split_group()
    group[0].mapping_quality = 0
    kept, stats = _process(group, circular)
    assert [r.is_supplementary for r in kept] == [True]
    assert stats["low_mapq"] == 1 and stats["supplementary_kept"] == 1
    assert not kept[0].has_tag("SA")
    group, circular = _split_group()
    group[0].mapping_quality = 0
    assert _process(group, circular, alignments="primary")[0] == []


@pytest.mark.parametrize("call_args, mask, insert", [
    ([], True, "auto"),
    (["--no-daf-mask-unaligned"], False, None),
    (["--no-daf-mask-unaligned", "--daf-mask-unaligned"], True, "auto"),
    (["--daf-insert-consensus", "off"], True, "off"),
    (["--daf-insert-consensus", "off", "--daf-insert-consensus", "auto"], True, "auto"),
    (["--daf-insert-consensus=off"], True, "off"),
])
def test_calling_settings_follow_call_args_as_fiberhmm_call_resolves_them(
        tmp_path, call_args, mask, insert):
    """outputs.json reports the DAF settings fiberhmm-call actually used:
    --call-args come last on its command line, so their last occurrence wins."""
    cfg = PipelineConfig(reads=["r.fq"], reference="ref.fa", enzyme="dddb",
                         outdir=str(tmp_path), call_args=call_args)
    settings = cfg.calling_settings()
    assert settings["daf_unaligned_mask"] is mask
    assert settings["daf_insert_consensus"] == insert


def test_call_fingerprint_binds_fiberhmm_defaults_and_bundled_tables():
    """A completed calling step must not be reused after a FiberHMM default or a
    bundled table changes under the same version string (3.0 changed both)."""
    import hashlib
    from fiberhmm.models import get_model_path
    from fiberhmm.pipeline.runner import bundled_model_identity, call_parser_defaults
    defaults = call_parser_defaults()
    assert str(defaults["phase_nrl"]).lower() == "off"
    assert defaults["nuc_recall_policy"] == "auto"
    ident = bundled_model_identity("hia5", "nanopore")
    path = get_model_path("hia5", tool="apply", seq="nanopore")
    with open(path, "rb") as handle:
        assert ident["apply"]["sha256"] == hashlib.sha256(handle.read()).hexdigest()
    assert bundled_model_identity(None, None) == {}


def test_crossstrand_recall_helper_has_no_periodicity_prior_by_default():
    import inspect
    from fiberhmm.crossstrand import recall
    func = next(f for _n, f in inspect.getmembers(recall, inspect.isfunction)
                if "phase_nrl" in inspect.signature(f).parameters
                and f.__module__ == recall.__name__)
    assert inspect.signature(func).parameters["phase_nrl"].default == 0


def test_call_fingerprint_records_the_alignment_default():
    """--primary/--no-primary share --alignments' destination; the recorded
    default is the one fiberhmm-call resolves."""
    from fiberhmm.pipeline.runner import call_parser_defaults
    assert call_parser_defaults()["alignments"] == "primary-supplementary"
    assert call_parser_defaults()["daf_mask_unaligned"] is True
    assert call_parser_defaults()["daf_insert_consensus"] == "auto"
