"""FiberHMM 3.0 default changes (user decisions for the release).

1. Hia5 + Nanopore reads ML at 248 by default wherever the chemistry preset
   resolves a threshold (call, apply, recall-tfs/-nucs, extract, qc); every
   other chemistry keeps its tool's historical default; an explicit
   --prob-threshold always wins.
2. fiberhmm-call and fiberhmm-apply call primary alignments only by default
   (--no-primary restores the old behaviour); secondary/supplementary records
   pass through uncalled.
3. DddA CpG-aware recall (the recall-tfs policy) is on by default in
   fiberhmm-call and in the joint recall of fiberhmm-pair/-merge.
4. DAF tools (dedup/pair/merge) read MM/ML-native dU at 128, like call.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pysam
import pytest

from conftest import make_synthetic_bam, make_synthetic_iupac_bam

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(module: str, *args, timeout=240):
    env = dict(os.environ)
    env.setdefault("FIBERHMM_NO_UPDATE_CHECK", "1")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", module, *map(str, args)],
        cwd=REPO_ROOT, capture_output=True, timeout=timeout, env=env,
    )


def _pg_ds(path, program):
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
        records = [
            record for record in bam.header.to_dict().get("PG", [])
            if str(record.get("PN", "")).startswith(program)
        ]
    assert records, f"no {program} @PG record"
    return records[-1].get("DS", "")


def _chemistry_header(enzyme, platform, mode):
    from fiberhmm.io.bam_header import append_chemistry

    return append_chemistry(
        pysam.AlignmentHeader.from_dict({
            "HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 10_000}]}),
        {"assay": "fiber-seq", "enzyme": enzyme, "platform": platform,
         "mode": mode},
    )


# --------------------------------------------------------------------------
# 1. per-chemistry ML threshold
# --------------------------------------------------------------------------

def test_threshold_resolver_hia5_nanopore_is_248():
    from fiberhmm.models import (
        DEFAULT_PROB_THRESHOLD,
        default_prob_threshold,
        resolve_prob_threshold,
    )

    assert DEFAULT_PROB_THRESHOLD == 128
    assert default_prob_threshold("hia5", "nanopore") == 248
    assert default_prob_threshold("HIA5", "Nanopore") == 248
    assert default_prob_threshold("hia5", "pacbio") == 128
    assert default_prob_threshold("dddb", "nanopore") == 128
    assert default_prob_threshold("ecogii", "nanopore") == 128
    assert default_prob_threshold(None, None) == 128
    # Each tool keeps its own fallback for every other chemistry.
    assert default_prob_threshold("hia5", "pacbio", 125) == 125
    assert default_prob_threshold("hia5", "nanopore", 125) == 248
    # An explicit value always wins.
    assert resolve_prob_threshold(90, "hia5", "nanopore") == 90
    assert resolve_prob_threshold(None, "hia5", "nanopore") == 248


def test_threshold_matches_calibrated_hia5_nanopore_qc_reference():
    """The QC reference caps the rate score when the run threshold differs
    by >5 from its calibration threshold; the preset must match it."""
    from fiberhmm.models import default_prob_threshold
    from fiberhmm.qc.core import default_qc_prob_threshold, load_references

    profile = load_references()["profiles"]["hia5_nanopore"]
    calibrated = int(profile["rate"]["probability_threshold"])
    assert default_prob_threshold("hia5", "nanopore") == calibrated
    assert default_qc_prob_threshold("nanopore-fiber", None) == calibrated
    assert default_qc_prob_threshold("nanopore-fiber", "hia5") == calibrated
    assert default_qc_prob_threshold("nanopore-fiber", "ecogii") == 125
    assert default_qc_prob_threshold("pacbio-fiber", "hia5") == 125
    assert default_qc_prob_threshold("daf", "dddb") == 125


def test_strand_rescue_hia5_nanopore_preset_agrees():
    from fiberhmm.inference.strand_rescue import PRESETS
    from fiberhmm.models import default_prob_threshold

    assert PRESETS["hia5-nanopore"]["prob_threshold"] == default_prob_threshold(
        "hia5", "nanopore")


def test_call_threshold_follows_resolved_chemistry(monkeypatch):
    from fiberhmm.cli import call

    monkeypatch.setattr(sys, "argv", ["fiberhmm-call", "-i", "x.bam", "-o", "y.bam",
                                      "--enzyme", "hia5"])
    args = call.parse_args()
    assert args.prob_threshold is None
    assert args.dedup_prob_threshold is None

    def resolve(**kw):
        base = dict(prob_threshold=None, enzyme=None, seq=None)
        base.update(kw)
        return call._resolve_call_prob_threshold(SimpleNamespace(**base))

    assert resolve(enzyme="hia5", seq="nanopore") == 248
    assert resolve(enzyme="hia5", seq="pacbio") == 128
    assert resolve(enzyme="dddb", seq="nanopore") == 128
    assert resolve(enzyme="hia5", seq="nanopore", prob_threshold=200) == 200
    # A custom -m inherits Hia5 Nanopore from the input's declaration.
    inherited = call._resolve_call_prob_threshold(
        SimpleNamespace(prob_threshold=None, enzyme=None, seq=None),
        {"enzyme": "hia5", "platform": "nanopore"},
    )
    assert inherited == 248


def test_apply_threshold_default_is_chemistry_preset(monkeypatch, tmp_path):
    from fiberhmm.cli import apply

    monkeypatch.setattr(sys, "argv", ["fiberhmm-apply", "-i", "x.bam", "-o", "out",
                                      "--enzyme", "hia5"])
    args = apply.parse_args()
    assert args.prob_threshold is None

    def resolve(**kw):
        base = dict(prob_threshold=None, enzyme=None, seq=None, input="-")
        base.update(kw)
        return apply._resolve_apply_prob_threshold(SimpleNamespace(**base))

    assert resolve(enzyme="hia5", seq="nanopore") == 248
    assert resolve(enzyme="hia5", seq="pacbio") == 128
    assert resolve(enzyme="hia5", seq="nanopore", prob_threshold=10) == 10

    # A custom -m (no --enzyme) on a BAM that declares Hia5 Nanopore.
    bam = tmp_path / "declared.bam"
    with pysam.AlignmentFile(str(bam), "wb",
                             header=_chemistry_header("hia5", "nanopore",
                                                      "nanopore-fiber")):
        pass
    assert resolve(input=str(bam)) == 248


def test_recall_threshold_default_uses_declared_chemistry():
    from fiberhmm.cli import recall_tfs

    def resolve(header=None, **kw):
        base = dict(prob_threshold=None, enzyme=None, seq=None)
        base.update(kw)
        return recall_tfs._resolve_recall_prob_threshold(
            SimpleNamespace(**base), header)

    ont = _chemistry_header("hia5", "nanopore", "nanopore-fiber")
    pacbio = _chemistry_header("hia5", "pacbio", "pacbio-fiber")
    assert resolve() == 125
    assert resolve(ont) == 248
    assert resolve(pacbio) == 125
    assert resolve(enzyme="hia5", seq="nanopore") == 248
    assert resolve(ont, enzyme="hia5") == 248
    assert resolve(ont, prob_threshold=128) == 128
    with pytest.raises(SystemExit):
        resolve(prob_threshold=300)


def test_recall_extract_modifications_honours_threshold():
    """recall-tfs used to hard-code 125 when re-reading MM/ML."""
    from fiberhmm.inference.tf_recaller import extract_modifications

    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 10_000}]})
    read = pysam.AlignedSegment(header)
    read.query_name = "r"
    read.query_sequence = "CCCACCCACCCACCC"
    read.flag = 0
    read.reference_id = 0
    read.reference_start = 0
    read.cigartuples = [(0, 15)]
    read.set_tag("MM", "A+a,0,0,0;")
    read.set_tag("ML", [130, 250, 200])
    low, _, _ = extract_modifications(read, "nanopore-fiber", 3)
    strict, _, _ = extract_modifications(read, "nanopore-fiber", 3,
                                         prob_threshold=248)
    assert low == {3, 7, 11}
    assert strict == {7}


def test_extract_threshold_default_uses_declared_chemistry(tmp_path):
    from fiberhmm.cli.extract_tags import _default_extract_prob_threshold

    ont = tmp_path / "ont.bam"
    plain = tmp_path / "plain.bam"
    with pysam.AlignmentFile(str(ont), "wb",
                             header=_chemistry_header("hia5", "nanopore",
                                                      "nanopore-fiber")):
        pass
    with pysam.AlignmentFile(str(plain), "wb", header={
            "HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 100}]}):
        pass
    assert _default_extract_prob_threshold(str(ont)) == 248
    assert _default_extract_prob_threshold(str(plain)) == 125


def test_call_hia5_nanopore_end_to_end_records_248(tmp_path):
    bam = str(tmp_path / "ont.bam")
    make_synthetic_bam(bam, n_reads=4, read_length=1500, n_chroms=1,
                       chrom_length=20_000, seed=3)
    out = tmp_path / "calls.bam"
    result = _run_cli(
        "fiberhmm.cli.call", "-i", bam, "-o", out, "--enzyme", "hia5",
        "--seq", "nanopore", "--no-qc", "--no-recall-nucs", "-c", "1",
        "--io-threads", "1", "--min-read-length", "0",
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert "prob_threshold=248" in _pg_ds(out, "fiberhmm-call")

    explicit = tmp_path / "explicit.bam"
    result = _run_cli(
        "fiberhmm.cli.call", "-i", bam, "-o", explicit, "--enzyme", "hia5",
        "--seq", "nanopore", "--prob-threshold", "128", "--no-qc",
        "--no-recall-nucs", "-c", "1", "--io-threads", "1",
        "--min-read-length", "0",
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert "prob_threshold=128" in _pg_ds(explicit, "fiberhmm-call")


# --------------------------------------------------------------------------
# 2. primary-only by default
# --------------------------------------------------------------------------

def test_primary_is_default_for_call_and_apply(monkeypatch):
    from fiberhmm.cli import apply, call

    monkeypatch.setattr(sys, "argv", ["fiberhmm-call", "-i", "x", "-o", "y"])
    assert call.parse_args().primary is True
    monkeypatch.setattr(sys, "argv", ["fiberhmm-call", "-i", "x", "-o", "y",
                                      "--no-primary"])
    assert call.parse_args().primary is False
    monkeypatch.setattr(sys, "argv", ["fiberhmm-apply", "-i", "x", "-o", "y"])
    assert apply.parse_args().primary is True
    monkeypatch.setattr(sys, "argv", ["fiberhmm-apply", "-i", "x", "-o", "y",
                                      "--no-primary"])
    assert apply.parse_args().primary is False


def _bam_with_supplementary(tmp_path):
    source = str(tmp_path / "src.bam")
    make_synthetic_bam(source, n_reads=3, read_length=1500, n_chroms=1,
                       chrom_length=20_000, seed=9)
    bam = tmp_path / "with_supp.bam"
    with pysam.AlignmentFile(source, "rb") as src, \
            pysam.AlignmentFile(str(bam), "wb", header=src.header) as out:
        for index, read in enumerate(src.fetch(until_eof=True)):
            if index == 1:
                read.flag |= 2048      # supplementary, full SEQ + MM/ML
            out.write(read)
    pysam.index(str(bam))
    return bam


@pytest.mark.parametrize("extra, supplementary_called", [
    ((), False),
    (("--no-primary",), True),
])
def test_call_passes_supplementary_through_uncalled(tmp_path, benchmark_model_path,
                                                    extra, supplementary_called):
    bam = _bam_with_supplementary(tmp_path)
    out = tmp_path / "out.bam"
    result = _run_cli(
        "fiberhmm.cli.call", "-i", bam, "-o", out, "-m", benchmark_model_path,
        "--min-read-length", "0", "--prob-threshold", "0", "--no-qc",
        "--no-recall-nucs", "-c", "1", "--io-threads", "1", *extra,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    with pysam.AlignmentFile(str(out), "rb") as handle:
        reads = list(handle.fetch(until_eof=True))
    assert len(reads) == 3                       # nothing is dropped
    supplementary = [read for read in reads if read.is_supplementary]
    assert len(supplementary) == 1
    assert supplementary[0].has_tag("ns") is supplementary_called
    assert all(read.has_tag("ns") for read in reads if not read.is_supplementary)


@pytest.mark.parametrize("mode_args", [(), ("--streaming",)])
@pytest.mark.parametrize("extra, supplementary_called", [
    ((), False),
    (("--no-primary",), True),
])
def test_apply_passes_supplementary_through_uncalled(tmp_path, benchmark_model_path,
                                                     mode_args, extra,
                                                     supplementary_called):
    """Both apply paths (-c 1 chunk pipeline and --streaming)."""
    bam = _bam_with_supplementary(tmp_path)
    outdir = tmp_path / "out"
    result = _run_cli(
        "fiberhmm.cli.apply", "-i", bam, "-o", outdir, "-m", benchmark_model_path,
        "--min-read-length", "0", "--prob-threshold", "0", "-c", "1",
        *mode_args, *extra,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    output = outdir / "with_supp_footprints.bam"
    with pysam.AlignmentFile(str(output), "rb", check_sq=False) as handle:
        reads = list(handle.fetch(until_eof=True))
    assert len(reads) == 3
    supplementary = [read for read in reads if read.is_supplementary]
    assert len(supplementary) == 1
    assert supplementary[0].has_tag("ns") is supplementary_called
    assert all(read.has_tag("ns") for read in reads if not read.is_supplementary)


# --------------------------------------------------------------------------
# 3. DddA CpG-aware recall everywhere
# --------------------------------------------------------------------------

def test_cpg_masking_policy_is_shared():
    from fiberhmm.inference.tf_recaller import resolve_cpg_masking

    assert resolve_cpg_masking(None, "ddda", "daf") is True
    assert resolve_cpg_masking(None, "dddb", "daf") is False
    assert resolve_cpg_masking(None, "hia5", "pacbio-fiber") is False
    assert resolve_cpg_masking(False, "ddda", "daf") is False
    assert resolve_cpg_masking(True, None, "daf") is True
    with pytest.raises(ValueError):
        resolve_cpg_masking(True, "dddb", "daf")
    with pytest.raises(ValueError):
        resolve_cpg_masking(True, None, "pacbio-fiber")


def test_cpg_mask_from_intervals_matches_build_cpg_mask():
    from fiberhmm.inference.tf_recaller import build_cpg_mask, cpg_mask_from_intervals

    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 1000}]})
    read = pysam.AlignedSegment(header)
    read.query_name = "r"
    read.query_sequence = "ACGT" * 25
    read.flag = 0
    read.reference_id = 0
    read.reference_start = 0
    read.cigartuples = [(0, 100)]
    read.set_tag("MA", "100;ddda_ucg.:11-20;ddda_mcg.:51-10")
    expected = build_cpg_mask(read, 100)
    assert not expected[10:30].any() and expected[:10].all() and expected[30:].all()
    np.testing.assert_array_equal(
        cpg_mask_from_intervals(100, "unmethylated-only", [(10, 30)], []), expected)
    methylated = build_cpg_mask(read, 100, "methylated-only")
    assert methylated[50:60].all() and methylated.sum() == 10


def test_apply_payload_carries_ddda_island_calls():
    from fiberhmm.inference.engine import make_apply_payload
    from fiberhmm.inference.fused_stages import payload_cpg_mask

    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 1000}]})
    read = pysam.AlignedSegment(header)
    read.query_name = "r"
    read.query_sequence = "ACYT" * 25
    read.flag = 0
    read.reference_id = 0
    read.reference_start = 0
    read.cigartuples = [(0, 100)]
    read.set_tag("st", "CT")
    read.set_tag("MA", "100;ddda_ucg.:11-20")
    payload = make_apply_payload(read, mode="daf")
    assert payload["_cpg_ma_intervals"]["ucg"] == [(10, 30)]
    mask = payload_cpg_mask(payload, 100, "unmethylated-only")
    assert not mask[10:30].any() and mask[:10].all() and mask[30:].all()
    # No MA: every CpG is masked.
    read.set_tag("MA", None)
    bare = make_apply_payload(read, mode="daf")
    assert "_cpg_ma_intervals" not in bare
    assert payload_cpg_mask(bare, 100, "unmethylated-only").all()


def test_fused_worker_applies_cpg_mask(monkeypatch):
    """The fused call worker hands recall a CpG mask and m5c tables."""
    from fiberhmm.inference import streaming_workers
    from fiberhmm.models import get_model_path

    captured = {}
    streaming_workers._init_fused_worker(
        get_model_path("ddda", tool="apply"),
        get_model_path("ddda", tool="recall"),
        recall_nucs=False, cpg_mask_policy="unmethylated-only",
    )
    state = streaming_workers._worker_recall_state
    assert state["cpg_mask_policy"] == "unmethylated-only"
    assert state["m5c_llr_hit"] is not None and state["m5c_llr_miss"] is not None

    def fake_recall(*args, **kwargs):
        captured.update(kwargs)
        return {"ns": [], "nl": [], "as": [], "al": [], "tf_calls": []}

    monkeypatch.setattr(streaming_workers, "build_fused_recall_result", fake_recall)
    monkeypatch.setattr(streaming_workers, "run_hmm_apply_stage",
                        lambda *a, **k: {"ns": [0], "nl": [150], "as": [], "al": []})
    monkeypatch.setattr(streaming_workers, "apply_result_has_footprints",
                        lambda result: True)
    seq = "ACGT" * 100
    monkeypatch.setattr(streaming_workers, "extract_fiber_read_from_payload",
                        lambda payload, mode, threshold: {"query_sequence": seq})
    payload = {"query_name": "r", "query_sequence": seq, "is_reverse": False,
               "tags": {}, "_cpg_ma_intervals": {"ucg": [(100, 200)], "mcg": []}}
    streaming_workers._process_fused_payload_chunk_worker(
        [payload], 10, False, "daf", 3, 0)
    mask = captured["m5c_mask"]
    assert mask is not None and len(mask) == len(seq)
    assert mask[:100].all() and not mask[100:200].any() and mask[200:].all()
    assert captured["m5c_llr_hit"] is state["m5c_llr_hit"]

    # Off (non-DddA): no mask, no tables.
    streaming_workers._init_fused_worker(
        get_model_path("ddda", tool="apply"), None, recall_nucs=False,
        cpg_mask_policy=None,
    )
    captured.clear()
    streaming_workers._process_fused_payload_chunk_worker(
        [payload], 10, False, "daf", 3, 0)
    assert captured["m5c_mask"] is None


def test_call_ddda_declares_cpg_masking(tmp_path):
    bam = str(tmp_path / "daf.bam")
    make_synthetic_iupac_bam(bam, n_reads=4, read_length=2000, n_chroms=1,
                             chrom_length=20_000, seed=5)
    common = ("--enzyme", "ddda", "--no-qc", "--no-dedup", "--no-daf-call-snps",
              "--phase-nrl", "off", "-c", "1", "--io-threads", "1",
              "--min-read-length", "0")
    out = tmp_path / "on.bam"
    result = _run_cli("fiberhmm.cli.call", "-i", bam, "-o", out, *common)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert "cpg_mask=unmethylated-only" in _pg_ds(out, "fiberhmm-call")
    off = tmp_path / "off.bam"
    result = _run_cli("fiberhmm.cli.call", "-i", bam, "-o", off, *common,
                      "--no-use-m5c")
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert "cpg_mask=off" in _pg_ds(off, "fiberhmm-call")


def test_call_refuses_cpg_masking_for_dddb(tmp_path):
    bam = str(tmp_path / "daf.bam")
    make_synthetic_iupac_bam(bam, n_reads=2, read_length=1500, n_chroms=1,
                             chrom_length=20_000, seed=6)
    result = _run_cli("fiberhmm.cli.call", "-i", bam, "-o", tmp_path / "o.bam",
                      "--enzyme", "dddb", "--use-m5c", "--no-qc", "--no-dedup")
    assert result.returncode == 2
    assert b"--use-m5c" in result.stderr


def test_pair_joint_recall_masks_cpgs_by_default(monkeypatch):
    from fiberhmm.crossstrand import recall

    context = recall.RecallContext("ddda")
    assert context.cpg_mask_policy == "unmethylated-only"
    assert context.m5c_llr_hit is not None and context.nuc_m5c_llr_hit is not None
    assert recall.RecallContext("ddda", use_m5c=False).cpg_mask_policy is None

    captured = {}

    def fake_recall(*args, **kwargs):
        captured.update(kwargs)
        return {"ns": [], "nl": [], "as": [], "al": [], "tf_calls": []}

    import fiberhmm.inference.fused_stages as fused_stages
    monkeypatch.setattr(fused_stages, "build_fused_recall_result", fake_recall)
    monkeypatch.setattr(recall, "write_ma_tags", lambda *a, **k: None, raising=False)
    import fiberhmm.inference.tf_recaller as tf_recaller
    monkeypatch.setattr(tf_recaller, "write_ma_tags", lambda *a, **k: None)

    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 10_000}]})
    seg = pysam.AlignedSegment(header)
    seg.query_name = "c.cs"
    seq = ("ACGTTGCA" * 100)[:600]
    seg.query_sequence = seq[:100] + "Y" + seq[101:300] + "R" + seq[301:]
    seg.flag = 0
    seg.reference_id = 0
    seg.reference_start = 0
    seg.cigartuples = [(0, 600)]
    seg.set_tag("MA", "600;deam+:1-600;deam-:1-600")
    assert recall.recall_consensus_full(
        seg, context, cpg_intervals={"ucg": [(200, 400)], "mcg": []})
    mask = captured["m5c_mask"]
    assert mask[:200].all() and not mask[200:400].any() and mask[400:].all()
    assert captured["m5c_llr_hit"] is context.m5c_llr_hit
    assert captured["nuc_m5c_llr_hit"] is context.nuc_m5c_llr_hit


def test_consensus_cpg_intervals_project_source_islands():
    from fiberhmm.crossstrand.recall import consensus_cpg_intervals

    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 10_000}]})

    def read(start, cigar, ma=None):
        seg = pysam.AlignedSegment(header)
        seg.query_name = f"r{start}"
        length = sum(n for op, n in cigar if op in (0, 1, 4))
        seg.query_sequence = "A" * length
        seg.flag = 0
        seg.reference_id = 0
        seg.reference_start = start
        seg.cigartuples = cigar
        if ma:
            seg.set_tag("MA", ma)
        return seg

    ct = read(1000, [(0, 500)], "500;ddda_ucg.:101-50")     # ref 1100-1150
    ga = read(1200, [(0, 100), (2, 10), (0, 300)], "400;ddda_ucg.:91-20")
    # ga query 90-110 -> ref 1290-1300 (query 90..99) and 1310-1320 (100..109)
    out = consensus_cpg_intervals(ct, ga, ref_start=1000, length=510)
    assert out["mcg"] == []
    assert out["ucg"] == [(100, 150), (290, 300), (310, 320)]


# --------------------------------------------------------------------------
# 4. DAF tools read MM/ML dU at 128
# --------------------------------------------------------------------------

@pytest.mark.parametrize("module, runner, argv", [
    ("fiberhmm.cli.dedup", "run_dedup", ["-i", "IN", "-o", "OUT"]),
    ("fiberhmm.cli.merge", "run_merge", ["-i", "IN", "-o", "OUT"]),
])
def test_daf_tools_default_ml_threshold_is_128(monkeypatch, tmp_path, module,
                                               runner, argv):
    import importlib

    mod = importlib.import_module(module)
    source = tmp_path / "in.bam"
    source.write_bytes(b"")
    argv = [str(source) if a == "IN" else str(tmp_path / "out.bam") if a == "OUT"
            else a for a in argv]
    captured = {}

    def fake(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(mod, runner, fake)
    monkeypatch.setattr(sys, "argv", [module, *argv])
    mod.main()
    assert captured["prob_threshold"] == 128


def test_pair_default_ml_threshold_is_128(monkeypatch, tmp_path):
    from fiberhmm.cli import duplex, merge, pair

    source = tmp_path / "in.bam"
    source.write_bytes(b"")
    captured = {}

    def fake_pairing(*args, **kwargs):
        captured["pair"] = kwargs["prob_threshold"]
        return {"counts": {}, "seconds": 0.0}

    def fake_merge(*args, **kwargs):
        captured["merge"] = kwargs["prob_threshold"]
        captured["use_m5c"] = kwargs["use_m5c"]

    monkeypatch.setattr(duplex, "run_pairing", fake_pairing)
    monkeypatch.setattr(merge, "run_merge", fake_merge)
    monkeypatch.setattr(sys, "argv", ["fiberhmm-pair", "-i", str(source), "-o",
                                      str(tmp_path / "out.bam"), "--sequence-only"])
    pair.main()
    assert captured == {"pair": 128, "merge": 128, "use_m5c": None}
