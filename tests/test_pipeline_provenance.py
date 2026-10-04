"""Basecaller provenance through fiberhmm-pipeline, and its dorado step.

Detection (fiberhmm.io.provenance), header carry-through (inputs' @RG/@PG/@CO,
RG tag rewrite, PP chain), FASTQ without provenance, platform detection on
pipeline output, and the optional dorado basecalling step with a fake dorado
(tests/fixtures/fake_dorado.py). End-to-end tests need minimap2.
"""
from __future__ import annotations

import array
import json
import os
import random
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pysam
import pytest

from fiberhmm.cli.common import sniff_sequencing_platform
from fiberhmm.io import provenance as bp
from fiberhmm.pipeline import aligner as mm2
from fiberhmm.pipeline import basecall as bc
from fiberhmm.pipeline.headers import carry_input_headers, missing_pp_links
from fiberhmm.pipeline.runner import (
    PipelineConfig,
    default_sample_name,
    expand_read_inputs,
    inputs_provenance,
)

REPO = Path(__file__).resolve().parents[1]
FAKE_DORADO = REPO / "tests" / "fixtures" / "fake_dorado.py"
needs_minimap2 = pytest.mark.skipif(shutil.which("minimap2") is None,
                                    reason="minimap2 not installed")

MODEL = "dna_r10.4.1_e8.2_400bps_sup@v5.2.0"
MODBASE = f"{MODEL}_6mA@v1"
RUN = "4e3f2a1b9c8d7e6f"


def dorado_header(run=RUN, modbase=True, version="2.0.1", sample="yw"):
    ds = f"runid={run} basecall_model={MODEL}" + (f" modbase_models={MODBASE}" if modbase else "")
    cl = (f"dorado basecaller /models/{MODEL} /data/pod5 --device metal"
          + (f" --modified-bases-models /models/{MODBASE}" if modbase else ""))
    return {"HD": {"VN": "1.6", "SO": "unknown"},
            "RG": [{"ID": f"{run}_{MODEL}", "PL": "ONT", "DS": ds, "SM": sample,
                    "LB": sample}],
            "PG": [{"ID": "basecaller", "PN": "dorado", "VN": version, "CL": cl}],
            "CO": ["run notes: yw 2-4 h"]}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_detection_prefers_rg_ds_then_pg_cl_then_read_rg():
    full = bp.basecaller_provenance(dorado_header(), reads=[f"{RUN}_{MODEL}"])
    assert (full["program"], full["version"], full["platform"]) == ("dorado", "2.0.1",
                                                                    "nanopore")
    assert full["basecall_model"] == MODEL and full["modbase_models"] == [MODBASE]
    assert full["sources"] == {"program": "pg", "version": "pg",
                               "basecall_model": "rg_ds", "modbase_models": "rg_ds"}
    assert not full["notes"] and full["available"]

    # Without @RG DS: the dorado @PG CL (model path and --modified-bases-models).
    header = dorado_header()
    del header["RG"][0]["DS"]
    cl = bp.basecaller_provenance(header)
    assert cl["basecall_model"] == MODEL and cl["modbase_models"] == [MODBASE]
    assert cl["sources"]["basecall_model"] == cl["sources"]["modbase_models"] == "pg_cl"

    # Only per-read read groups: the model; the modbase model stays unknown
    # (dorado 2.x per-read groups omit it).
    reads = bp.basecaller_provenance({}, reads=[f"{RUN}_{MODEL}"])
    assert reads["basecall_model"] == MODEL and reads["modbase_models"] is None
    assert reads["sources"] == {"basecall_model": "read_rg"}
    assert reads["platform"] == "nanopore"
    assert any("modbase" in note for note in reads["notes"])
    suffixed = bp.parse_read_group_id(f"{RUN}_dna_r10.4.1_e8.2_400bps_sup@v4.3.0_5mCG_5hmCG@v1")
    assert suffixed == {"basecall_model": "dna_r10.4.1_e8.2_400bps_sup@v4.3.0",
                        "modbase_models": ["dna_r10.4.1_e8.2_400bps_sup@v4.3.0_5mCG_5hmCG@v1"]}
    assert bp.parse_read_group_id("not-a-dorado-group") == {}


def test_known_none_unknown_and_command_forms():
    # @RG DS with a model and no modbase_models: dorado used none (DAF).
    daf = bp.basecaller_provenance(dorado_header(modbase=False))
    assert daf["modbase_models"] == [] and daf["sources"]["modbase_models"] == "rg_ds"
    assert "modbase_models=none" in bp.ds_tokens(daf)
    assert bp.parse_dorado_command("dorado basecaller sup,6mA pod5/ -x metal") == {
        "basecall_model": "sup", "modbase_models": ["6mA"]}
    assert bp.parse_dorado_command(
        "dorado basecaller hac /d --modified-bases 5mCG_5hmCG 6mA --emit-moves") == {
        "basecall_model": "hac", "modbase_models": ["5mCG_5hmCG", "6mA"]}
    assert bp.parse_dorado_command("dorado basecaller -x cuda:0 sup /d")["modbase_models"] == []
    # dorado aligner is not the basecaller.
    aligner_only = bp.basecaller_provenance(
        {"PG": [{"ID": "aligner", "PN": "dorado", "VN": "2.0.1",
                 "CL": "dorado aligner ref.fa calls.bam"}]})
    assert aligner_only["program"] is None
    nothing = bp.basecaller_provenance(None)
    assert nothing["available"] is False and nothing["modbase_models"] is None
    assert "not recorded" in bp.missing_note(nothing)


def test_pacbio_provenance():
    header = {"RG": [{"ID": "a1b2", "PL": "PACBIO",
                      "DS": "READTYPE=CCS;BINDINGKIT=102-739-100;BASECALLERVERSION=5.0.0.6236"}],
              "PG": [{"ID": "ccs", "PN": "ccs", "VN": "6.4.0"},
                     {"ID": "jasmine", "PN": "jasmine", "VN": "2.0.0", "PP": "ccs"}]}
    prov = bp.basecaller_provenance(header)
    assert (prov["platform"], prov["program"], prov["version"]) == ("pacbio", "ccs", "6.4.0")
    assert (prov["modbase_caller"], prov["modbase_caller_version"]) == ("jasmine", "2.0.0")
    assert prov["instrument_basecaller_version"] == "5.0.0.6236"
    assert prov["basecall_model"] is None


def test_override_beats_headers_and_round_trips_through_ds():
    override = bp.parse_override("program=dorado version=0.9.6 model=m_override",
                                 ["dna_x_6mA@v2"])
    prov = bp.basecaller_provenance(dorado_header(), override=override)
    assert prov["basecall_model"] == "m_override" and prov["modbase_models"] == ["dna_x_6mA@v2"]
    assert prov["version"] == "0.9.6"
    assert set(prov["sources"].values()) == {"override"}
    with pytest.raises(bp.OverrideError):
        bp.parse_override("just-a-model")
    with pytest.raises(bp.OverrideError):
        bp.parse_override("colour=blue")
    assert bp.parse_override(None, ["none"]) == {"modbase_models": []}
    assert bp.parse_override(None, None) is None

    # A FiberHMM @PG that recorded an override keeps it above the header;
    # other recorded values are only a fallback.
    tokens = bp.ds_tokens(prov)
    recorded = dict(dorado_header())
    recorded["PG"] = recorded["PG"] + [{"ID": "fiberhmm-pipeline", "PN": "fiberhmm-pipeline",
                                        "PP": "basecaller", "DS": "x=1; " + tokens}]
    again = bp.basecaller_provenance(recorded)
    assert again["basecall_model"] == "m_override"
    assert again["sources"]["basecall_model"] == "recorded-override"
    detected = bp.ds_tokens(bp.basecaller_provenance(dorado_header()))
    stale = {"PG": [{"ID": "fiberhmm-call", "PN": "fiberhmm-call",
                     "DS": detected.replace(MODEL, "old_model")}],
             "RG": dorado_header()["RG"]}
    assert bp.basecaller_provenance(stale)["basecall_model"] == MODEL
    lost = {"PG": stale["PG"]}
    fallback = bp.basecaller_provenance(lost)
    assert fallback["basecall_model"] == "old_model"
    assert fallback["sources"]["basecall_model"] == "recorded"
    assert fallback["platform"] == "nanopore"


# ---------------------------------------------------------------------------
# Header carry-through
# ---------------------------------------------------------------------------

def test_header_merge_dedupes_suffixes_and_relinks():
    a = dorado_header()
    a["PG"].append({"ID": "samtools", "PN": "samtools", "PP": "basecaller", "VN": "1.20"})
    b = dorado_header(run=RUN, modbase=False)  # same RG ID, different DS
    b["PG"][0]["CL"] += " /other/pod5"
    same = dorado_header()  # identical to a's RG and dorado PG
    called = {"RG": [{"ID": "fh", "SM": "x"}],
              "PG": [{"ID": "basecaller", "PN": "dorado", "VN": "1.0"},
                     {"ID": "fiberhmm-call", "PN": "fiberhmm-call", "PP": "basecaller"},
                     {"ID": "samtools", "PN": "samtools", "PP": "fiberhmm-call"}],
             "CO": ["FIBERHMM-CHEMISTRY:v1:assay=daf", "MA-TYPES:v1:nuc",
                    "run notes: yw 2-4 h", "other note"]}
    carried = carry_input_headers([a, None, b, same, called], reserved_rg=["s1"],
                                  reserved_pg=["minimap2", "fiberhmm-pipeline"])
    rid = f"{RUN}_{MODEL}"
    assert [g["ID"] for g in carried.rg] == [rid, f"{rid}-2", "fh"]
    assert carried.rg_maps == [{rid: rid}, {}, {rid: f"{rid}-2"}, {rid: rid}, {"fh": "fh"}]
    assert carried.rg[1]["DS"].endswith(MODEL)  # b's own DS (no modbase)
    ids = [p["ID"] for p in carried.pg]
    assert ids == ["basecaller", "samtools", "basecaller-2", "basecaller-3", "samtools-2"]
    by_id = {p["ID"]: p for p in carried.pg}
    assert by_id["samtools"]["PP"] == "basecaller"
    # FiberHMM records of a realigned input are dropped; the child is relinked.
    assert by_id["samtools-2"]["PP"] == "basecaller-3"
    assert not any(p["PN"].startswith("fiberhmm") for p in carried.pg)
    assert carried.co == ["run notes: yw 2-4 h", "other note"]
    assert carried.leaf == "samtools-2"
    assert missing_pp_links({"PG": carried.pg}) == []
    # A carried ID that equals the pipeline's own read group is renamed.
    clash = carry_input_headers([{"RG": [{"ID": "s1", "PL": "ONT"}]}], reserved_rg=["s1"])
    assert clash.rg[0]["ID"] == "s1-2" and clash.rg_maps == [{"s1": "s1-2"}]


def _ubam(path, header, reads, rg=None):
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for name, seq, tags in reads:
            record = pysam.AlignedSegment(out.header)
            record.query_name = name
            record.flag = 4
            record.query_sequence = seq
            record.query_qualities = pysam.qualitystring_to_array("5" * len(seq))
            record.set_tags(list(tags) + ([("RG", rg, "Z")] if rg else []))
            out.write(record)


def test_feeder_rewrites_read_groups_and_sanitises_fastq(tmp_path):
    rid = f"{RUN}_{MODEL}"
    one, two = tmp_path / "a.bam", tmp_path / "b.bam"
    _ubam(one, dorado_header(), [("r1", "ACGT", [("MM", "A+a?,0;", "Z"),
                                                 ("ML", array.array("B", [250]))])], rid)
    _ubam(two, dorado_header(modbase=False),
          [("r2", "ACGT", []), ("r3", "ACGT", [("RG", "undeclared", "Z")])], rid)
    fastq = tmp_path / "c.fastq"
    fastq.write_text(f"@r4\tRG:Z:{rid}\tqs:f:12.5\tNM:i:3\tMM:Z:A+a?,0;\tML:B:C,200\n"
                     "ACGT\n+\nIIII\n")
    files = [mm2.classify_read_file(str(p)) for p in (one, two, fastq)]
    carried = carry_input_headers([dorado_header(), dorado_header(modbase=False), None])
    feeder = mm2.ReadFeeder(files, carry_tags=True, rg_maps=carried.rg_maps)
    entries = list(mm2.iter_fastq_entries(feeder.chunks()))
    restored = [feeder.restore(name) for name, *_ in entries]
    assert restored == [("r1", rid), ("r2", f"{rid}-2"), ("r3", None), ("r4", None)]
    assert feeder.undeclared_groups == 1
    # No RG (minimap2 -R writes it) and no minimap2 tag in what -y copies;
    # other SAM tags pass through.
    comments = [comment for _, comment, *_ in entries]
    assert comments[0] == "MM:Z:A+a?,0;\tML:B:C,250"
    assert comments[3] == "qs:f:12.5\tMM:Z:A+a?,0;\tML:B:C,200"
    assert feeder._groups == {}  # released
    assert mm2.sanitize_fastq_header(b"@x runid=abc ch=5\n") == b"@x\n"


def test_inputs_provenance_for_fastq_and_mixed(tmp_path):
    fastq = tmp_path / "plain.fastq"
    fastq.write_text("@r1\nACGT\n+\nIIII\n")
    prov = inputs_provenance([str(fastq)])
    assert prov["available"] is False
    assert prov["inputs"] == [{"path": str(fastq), "kind": "fastq", "available": False}]
    assert "not recorded" in bp.missing_note(prov)
    overridden = inputs_provenance([str(fastq)], bp.parse_override(None, [MODBASE]))
    assert overridden["modbase_models"] == [MODBASE]
    assert overridden["sources"] == {"modbase_models": "override"}
    dorado_fastq = tmp_path / "dorado.fastq"
    dorado_fastq.write_text(f"@r1\tqs:f:9\tRG:Z:{RUN}_{MODEL}\nACGT\n+\nIIII\n")
    ubam = tmp_path / "u.bam"
    _ubam(ubam, dorado_header(), [("r1", "ACGT", [])], f"{RUN}_{MODEL}")
    mixed = inputs_provenance([str(dorado_fastq), str(fastq), str(ubam)])
    assert mixed["program"] == "dorado" and mixed["modbase_models"] == [MODBASE]
    assert [i["available"] for i in mixed["inputs"]] == [True, False, True]
    assert any("1 of 3 input files" in note for note in mixed["notes"])


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

COMP = str.maketrans("ACGT", "TGCA")


def _revcomp(seq):
    return seq.translate(COMP)[::-1]


def _genome(n=20000, seed=31):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


def _m6a_reads(genome, n, seed, prefix="read"):
    """Hia5-like Nanopore reads: A+a calls, high ML in linkers, low in
    nucleosome-sized protected stretches."""
    rng = random.Random(seed)
    reads = []
    for i in range(n):
        size = rng.randint(1500, 3000)
        start = rng.randrange(0, len(genome) - size)
        mol = genome[start:start + size]
        seq = mol if rng.random() < 0.5 else _revcomp(mol)
        ml = []
        for j, base in enumerate(seq):
            if base == "A":
                protected = ((start + j) // 180) % 2 == 0
                ml.append(rng.randint(0, 40) if protected or rng.random() > 0.5
                          else rng.randint(240, 255))
        mm = "A+a?," + ",".join("0" * 1 for _ in ml) + ";" if ml else "A+a?;"
        reads.append({"name": f"{prefix}{i}", "seq": seq, "mm": mm, "ml": ml})
    return reads


def _deaminated_reads(genome, n, seed, prefix="d"):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        size = rng.randint(1500, 3000)
        start = rng.randrange(0, len(genome) - size)
        mol = genome[start:start + size]
        mol = mol if rng.random() < 0.5 else _revcomp(mol)
        seq = "".join("T" if b == "C" and rng.random() < (0.005 if (k // 180) % 2 == 0
                                                          else 0.08) else b
                      for k, b in enumerate(mol))
        out.append((f"{prefix}{i}", seq))
    return out


def _run_cli(args, env_extra=None, timeout=300):
    env = dict(os.environ, PYTHONPATH=str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               FIBERHMM_NO_UPDATE_CHECK="1", **(env_extra or {}))
    return subprocess.run([sys.executable, "-m", "fiberhmm.cli.pipeline", *args],
                          capture_output=True, text=True, timeout=timeout, env=env)


def _check_read_groups(path):
    """Every record's RG is declared, and every @PG PP link resolves."""
    with pysam.AlignmentFile(str(path)) as bam:
        header = bam.header.to_dict()
        declared = {g["ID"] for g in header.get("RG", [])}
        groups = [r.get_tag("RG") for r in bam.fetch(until_eof=True)]
    assert groups and all(g in declared for g in groups)
    assert missing_pp_links(header) == []
    return header, groups


@needs_minimap2
def test_unaligned_bams_and_fastq_carry_provenance_end_to_end(tmp_path):
    genome = _genome()
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chrA\n" + genome + "\n")
    reads = _deaminated_reads(genome, 60, 41)
    rid = f"{RUN}_{MODEL}"
    one, two = tmp_path / "flowcell1.bam", tmp_path / "flowcell2.bam"
    _ubam(one, dorado_header(modbase=False, sample="yw"),
          [(n, s, [("qs", 12, "i")]) for n, s in reads[:25]], rid)
    second = dorado_header(modbase=False, version="2.0.0", sample="yw_rep2")
    second["PG"][0]["CL"] += " /flowcell2/pod5"
    _ubam(two, second, [(n, s, []) for n, s in reads[25:50]], rid)
    fastq = tmp_path / "extra.fastq"
    fastq.write_text("".join(f"@{n} runid=x\n{s}\n+\n{'I' * len(s)}\n" for n, s in reads[50:]))
    out = tmp_path / "out"
    result = _run_cli([str(one), str(two), str(fastq), "--reference", str(fasta),
                       "--enzyme", "dddb", "-o", str(out), "--sample", "s1", "-c", "2",
                       "--min-read-length", "500", "--no-qc"])
    assert result.returncode == 0, result.stderr[-3000:]
    outputs = json.loads((out / "outputs.json").read_text())
    header, groups = _check_read_groups(outputs["aligned_bam"])
    ids = [g["ID"] for g in header["RG"]]
    assert ids == ["s1", rid, f"{rid}-2"]
    with pysam.AlignmentFile(outputs["aligned_bam"]) as bam:
        by_name = {r.query_name: r.get_tag("RG") for r in bam.fetch(until_eof=True)}
    expected = {n: rid for n, _ in reads[:25]}
    expected.update({n: f"{rid}-2" for n, _ in reads[25:50]})
    expected.update({n: "s1" for n, _ in reads[50:]})
    assert by_name and all(expected[n] == g for n, g in by_name.items())
    assert set(by_name.values()) == {rid, f"{rid}-2", "s1"}
    pgs = header["PG"]
    assert [p["ID"] for p in pgs][:2] == ["basecaller", "basecaller-2"]
    minimap2 = next(p for p in pgs if p["ID"] == "minimap2")
    assert minimap2["PP"] == "basecaller-2"
    assert pgs[-1]["ID"] == "fiberhmm-pipeline" and pgs[-1]["PP"] == "minimap2"
    assert "basecaller=dorado" in pgs[-1]["DS"] and "modbase_models=none" in pgs[-1]["DS"]
    assert "run notes: yw 2-4 h" in header["CO"]
    prov = outputs["settings"]["basecaller"]
    assert prov["program"] == "dorado" and prov["version"] == "2.0.1,2.0.0"
    assert prov["mixed"] == ["version"] and prov["modbase_models"] == []
    assert [i["available"] for i in prov["inputs"]] == [True, True, False]
    # The called BAM keeps the chain, and fiberhmm-call records the provenance.
    called, _ = _check_read_groups(outputs["called_bam"])
    call_pg = called["PG"][-1]
    assert call_pg["PN"] == "fiberhmm-call" and "basecaller=dorado" in call_pg["DS"]
    assert sniff_sequencing_platform(outputs["aligned_bam"], inspect_reads=False).platform \
        == "nanopore"


@needs_minimap2
def test_fastq_without_provenance_takes_an_override(tmp_path):
    genome = _genome(seed=51)
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chrA\n" + genome + "\n")
    fastq = tmp_path / "reads.fastq"
    fastq.write_text("".join(f"@{n}\n{s}\n+\n{'I' * len(s)}\n"
                             for n, s in _deaminated_reads(genome, 40, 52)))
    out = tmp_path / "out"
    progress = tmp_path / "p.jsonl"
    result = _run_cli([str(fastq), "--reference", str(fasta), "--enzyme", "dddb",
                       "-o", str(out), "-c", "2", "--min-read-length", "500", "--no-qc",
                       "--progress-json", str(progress)])
    assert result.returncode == 0, result.stderr[-3000:]
    logs = [json.loads(line).get("message", "") for line in progress.read_text().splitlines()]
    assert any("basecaller provenance is not recorded" in m for m in logs)
    outputs = json.loads((out / "outputs.json").read_text())
    assert outputs["settings"]["basecaller"]["available"] is False
    with pysam.AlignmentFile(outputs["aligned_bam"]) as bam:
        assert "basecaller=unknown" in bam.header.to_dict()["PG"][-1]["DS"]

    out2 = tmp_path / "out2"
    result = _run_cli([str(fastq), "--reference", str(fasta), "--enzyme", "dddb",
                       "-o", str(out2), "-c", "2", "--min-read-length", "500", "--no-qc",
                       "--basecaller-info", f"program=dorado version=0.9.6 model={MODEL}",
                       "--modbase-model", "none"])
    assert result.returncode == 0, result.stderr[-3000:]
    outputs = json.loads((out2 / "outputs.json").read_text())
    prov = outputs["settings"]["basecaller"]
    assert (prov["program"], prov["version"], prov["basecall_model"],
            prov["modbase_models"]) == ("dorado", "0.9.6", MODEL, [])
    assert outputs["settings"]["basecaller_override"]["version"] == "0.9.6"
    with pysam.AlignmentFile(outputs["called_bam"]) as bam:
        call_ds = bam.header.to_dict()["PG"][-1]["DS"]
    assert "basecaller_version=0.9.6" in call_ds and "version:override" in call_ds
    # An aligned BAM made by the pipeline from FASTQ: the override recorded in
    # its @PG makes it detectable as Nanopore.
    assert sniff_sequencing_platform(outputs["aligned_bam"], inspect_reads=False).platform \
        == "nanopore"
    # A changed override on the same OUTDIR is refused like any changed setting.
    again = _run_cli([str(fastq), "--reference", str(fasta), "--enzyme", "dddb",
                      "-o", str(out2), "-c", "2", "--min-read-length", "500", "--no-qc",
                      "--modbase-model", "other"])
    assert again.returncode != 0 and "basecaller_override" in again.stderr


# ---------------------------------------------------------------------------
# dorado step
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_dorado(tmp_path):
    exe = tmp_path / "bin" / "dorado"
    exe.parent.mkdir()
    exe.write_text(f"#!{sys.executable}\n" + FAKE_DORADO.read_text())
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return exe


def test_dorado_resolver_and_command(tmp_path, fake_dorado, monkeypatch):
    monkeypatch.delenv(bc.DORADO_ENV, raising=False)
    found = bc.find_dorado(str(fake_dorado))
    assert found.version == "2.0.1+fake"
    assert bc.find_dorado(str(fake_dorado.parent.parent)).path == str(fake_dorado)
    monkeypatch.setenv(bc.DORADO_ENV, str(fake_dorado))
    assert bc.find_dorado().path == str(fake_dorado)
    monkeypatch.delenv(bc.DORADO_ENV)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(bc, "candidate_paths", lambda: [])
    with pytest.raises(bc.DoradoNotFound) as missing:
        bc.find_dorado()
    assert "github.com/nanoporetech/dorado" in str(missing.value)
    with pytest.raises(bc.DoradoNotFound):
        bc.find_dorado(str(tmp_path / "nope"))

    hia5 = bc.resolve_settings("hia5", models_directory=str(tmp_path / "models"))
    cmd = bc.basecall_command(found, hia5, "/data/pod5", recursive=True)
    assert cmd[:4] == [found.path, "basecaller", "sup", "/data/pod5"]
    assert cmd[cmd.index("--models-directory") + 1] == str(tmp_path / "models")
    assert cmd[-2:] == ["--modified-bases", "6mA"] and "--recursive" in cmd
    assert "--device" not in cmd
    daf = bc.resolve_settings("dddb", device="metal", batchsize=64, extra_args=["--min-qscore", "8"])
    cmd = bc.basecall_command(found, daf, "/x.pod5", recursive=False, resume_from="/r.bam")
    assert "--modified-bases" not in cmd and "--modified-bases-models" not in cmd
    assert cmd[cmd.index("--device") + 1] == "metal" and cmd[cmd.index("--batchsize") + 1] == "64"
    assert cmd[cmd.index("--resume-from") + 1] == "/r.bam" and cmd[-2:] == ["--min-qscore", "8"]
    models = bc.resolve_settings("hia5", modbase_models="/m/a_6mA@v1")
    assert models.modified_bases is None and models.modbase_models == ["/m/a_6mA@v1"]
    assert bc.resolve_settings("hia5", modified_bases="none").modified_bases is None
    assert bc.resolve_settings("hia5", model="sup,6mA").modified_bases is None
    with pytest.raises(ValueError):
        bc.resolve_settings("hia5", modified_bases="6mA", modbase_models="x")
    assert bc.accepts_fast5("0.8.3") and not bc.accepts_fast5("2.0.1")


def test_raw_inputs_are_recognised(tmp_path):
    run = tmp_path / "yw_run" / "pod5"
    (run / "sub").mkdir(parents=True)
    (run / "sub" / "a.pod5").write_bytes(b"x")
    assert bc.is_raw_path(str(run)) and expand_read_inputs([str(run)]) == [str(run)]
    assert default_sample_name(str(run)) == "yw_run"
    config = PipelineConfig(reads=[str(run)], reference="r.fa", enzyme="hia5", outdir="o")
    assert config.steps()[:2] == ["basecall", "prepare_reference"]


@needs_minimap2
def test_pod5_to_called_bam_with_fake_dorado_resumes(tmp_path, fake_dorado):
    genome = _genome(seed=61)
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chrA\n" + genome + "\n")
    reads = _m6a_reads(genome, 40, 62)
    reads_json = tmp_path / "reads.json"
    reads_json.write_text(json.dumps(reads))
    pod5 = tmp_path / "yw_2_4h" / "pod5"
    pod5.mkdir(parents=True)
    (pod5 / "batch0.pod5").write_bytes(b"POD5 stand-in")
    log = tmp_path / "dorado_calls.jsonl"
    env = {"FAKE_DORADO_READS": str(reads_json), "FAKE_DORADO_LOG": str(log),
           "FIBERHMM_DORADO_MODELS_DIR": str(tmp_path / "models")}
    out = tmp_path / "out"
    args = [str(pod5), "--reference", str(fasta), "--enzyme", "hia5", "-o", str(out),
            "-c", "2", "--min-read-length", "500", "--no-qc", "--dorado", str(fake_dorado)]

    # Missing dorado: a clear message with download instructions, exit 2.
    missing = _run_cli(args[:-1] + [str(tmp_path / "no-dorado")], env)
    assert missing.returncode == 2 and "nanoporetech/dorado" in missing.stderr

    # dorado is killed after 25 reads (a BAM cut short): the run fails, the
    # partial BAM stays.
    crashed = _run_cli(args, {**env, "FAKE_DORADO_FAIL_AFTER": "25"})
    assert crashed.returncode == 1 and "dorado failed" in crashed.stderr
    partial = out / ".fiberhmm-pipeline" / "yw_2_4h.basecalled.partial.bam"
    assert partial.exists()

    # The next run resumes: the complete reads are copied, not redone.
    result = _run_cli(args, env)
    assert result.returncode == 0, result.stderr[-3000:]
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    basecalls = [c for c in calls if c[:1] == ["basecaller"]]
    assert len(basecalls) == 2 and "--resume-from" in basecalls[1]
    assert basecalls[1][:3] == ["basecaller", "sup", str(pod5)]
    assert basecalls[1][basecalls[1].index("--modified-bases") + 1] == "6mA"
    outputs = json.loads((out / "outputs.json").read_text())
    with pysam.AlignmentFile(outputs["basecalled_bam"], check_sq=False) as bam:
        names = [r.query_name for r in bam.fetch(until_eof=True)]
    assert sorted(names) == sorted(r["name"] for r in reads)  # each read once
    assert not partial.exists()
    assert outputs["basecall"]["resumed"] is True
    assert outputs["basecall"]["dorado_version"] == "2.0.1+fake"
    assert "--modified-bases 6mA" in outputs["basecall"]["command"]

    # Provenance: dorado's @PG/@RG reach the aligned and called BAMs.
    header, groups = _check_read_groups(outputs["aligned_bam"])
    assert header["PG"][0]["PN"] == "dorado"
    assert set(groups) == {f"4e3f2a1b9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f_{MODEL}"}
    prov = outputs["settings"]["basecaller"]
    assert (prov["program"], prov["version"], prov["basecall_model"]) == ("dorado", "2.0.1",
                                                                          MODEL)
    assert prov["modbase_models"] == [MODBASE] and prov["sources"]["modbase_models"] == "rg_ds"
    called, _ = _check_read_groups(outputs["called_bam"])
    assert f"modbase_models={MODBASE}" in called["PG"][-1]["DS"]
    assert sniff_sequencing_platform(outputs["called_bam"], inspect_reads=False).platform \
        == "nanopore"

    # Same command again: basecalling is not repeated.
    before = len(log.read_text().splitlines())
    again = _run_cli(args, env)
    assert again.returncode == 0, again.stderr[-3000:]
    later = [json.loads(line) for line in log.read_text().splitlines()[before:]]
    assert not any(c[:1] == ["basecaller"] for c in later)
    assert "resuming" in result.stderr

    # --redo basecall basecalls again from nothing, even over a partial BAM.
    crashed = _run_cli(args + ["--redo", "basecall"], {**env, "FAKE_DORADO_FAIL_AFTER": "25"})
    assert crashed.returncode == 1 and partial.exists()
    before = len(log.read_text().splitlines())
    redone = _run_cli(args + ["--redo", "basecall"], env)
    assert redone.returncode == 0, redone.stderr[-3000:]
    later = [json.loads(line) for line in log.read_text().splitlines()[before:]]
    rerun = [c for c in later if c[:1] == ["basecaller"]]
    assert len(rerun) == 1 and "--resume-from" not in rerun[0]


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------

def test_recorded_override_survives_repeated_runs_and_pipeline_detection(tmp_path):
    override = bp.parse_override("model=corrected", None)
    first = bp.basecaller_provenance(dorado_header(), override=override)
    header = dorado_header()
    header["PG"].append({"ID": "fiberhmm-pipeline", "PN": "fiberhmm-pipeline",
                         "PP": "basecaller", "DS": bp.ds_tokens(first)})
    second = bp.basecaller_provenance(header)  # e.g. fiberhmm-call on that BAM
    assert second["sources"]["basecall_model"] == "recorded-override"
    header["PG"].append({"ID": "fiberhmm-call", "PN": "fiberhmm-call",
                         "PP": "fiberhmm-pipeline", "DS": bp.ds_tokens(second)})
    third = bp.basecaller_provenance(header)
    assert third["basecall_model"] == "corrected"
    assert third["sources"]["basecall_model"] == "recorded-override"
    # The pipeline's own detection over input files keeps FiberHMM records.
    bam = tmp_path / "recalled.bam"
    _ubam(bam, header, [("r1", "ACGT", [])])
    assert inputs_provenance([str(bam)])["basecall_model"] == "corrected"


def test_read_group_program_links_follow_renames():
    a = {"RG": [{"ID": "g", "PG": "minimap2"}], "PG": [{"ID": "minimap2", "PN": "minimap2"}]}
    b = {"RG": [{"ID": "h", "PG": "fiberhmm-call"}],
         "PG": [{"ID": "dorado", "PN": "dorado"},
                {"ID": "fiberhmm-call", "PN": "fiberhmm-call", "PP": "dorado"}]}
    c = {"RG": [{"ID": "k", "PG": "missing"}]}
    carried = carry_input_headers([a, b, c], reserved_pg=["minimap2"])
    groups = {g["ID"]: g for g in carried.rg}
    assert groups["g"]["PG"] == "minimap2-2"
    assert groups["h"]["PG"] == "dorado"  # dropped FiberHMM program: its parent
    assert "PG" not in groups["k"]


def test_salvage_reads_a_bam_cut_mid_file(tmp_path):
    full = tmp_path / "full.bam"
    _ubam(full, dorado_header(), [(f"r{i}", "ACGT" * 200, []) for i in range(3000)])
    data = full.read_bytes()
    cut = tmp_path / "cut.bam"
    cut.write_bytes(data[: len(data) // 2])  # no EOF block, a block cut short
    kept = bc.salvage_bam(str(cut), str(tmp_path / "resume.bam"))
    assert 0 < kept < 3000
    assert bc.count_records(str(tmp_path / "resume.bam")) == kept
    assert bc.salvage_bam(str(tmp_path / "missing.bam"), str(tmp_path / "x.bam")) == 0
    assert not (tmp_path / "x.bam").exists()


def test_mixed_fastq_comments_still_carry_modification_tags(tmp_path):
    fastq = tmp_path / "mixed.fastq"
    fastq.write_text("@r1 runid=abc MM:Z:A+a?,0; ML:B:C,250\nACGT\n+\nIIII\n")
    assert mm2.fastq_has_sam_tags(str(fastq)) is False
    assert mm2.fastq_has_mod_tags(str(fastq)) is True
    feeder = mm2.ReadFeeder([mm2.classify_read_file(str(fastq))], carry_tags=True)
    (_, comment, _, _), = mm2.iter_fastq_entries(feeder.chunks())
    assert comment == "MM:Z:A+a?,0;\tML:B:C,250"


def test_basecall_fingerprint_covers_files_named_in_arguments(tmp_path):
    ids = tmp_path / "read_ids.txt"
    ids.write_text("a\n")
    settings = bc.resolve_settings("hia5", extra_args=["--read-ids", str(ids)])
    before = settings.fingerprint()
    ids.write_text("a\nb\n")
    assert settings.fingerprint() != before
    model = tmp_path / "dna_model"
    model.mkdir()
    (model / "config.toml").write_text("x")
    assert bc.resolve_settings("hia5", model=str(model)).fingerprint()["files"][0][0] \
        == str(model / "config.toml")
