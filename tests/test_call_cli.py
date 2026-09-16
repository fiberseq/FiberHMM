"""CLI characterization tests for `fiberhmm-call`."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pysam
import pytest
from conftest import make_synthetic_bam, make_synthetic_iupac_bam

from fiberhmm.cli.call import (
    _check_daf_inputs,
    _chemistry_declaration,
    _configure_ddda_mcg,
    _daf_snp_depth_preflight,
    _nuc_profile_identity,
    _nuc_profile_sha256,
    _resolve_apply_model,
    _resolve_dedup,
    _resolve_derived_tf_edge_gap,
    _resolve_recall_model,
)
from fiberhmm.cli.call import parse_args as parse_call_args
from fiberhmm.daf.snps import (
    DEFAULT_SNP_MIN_ALT_FIBERS,
    DEFAULT_SNP_MIN_DEPTH,
    DEFAULT_SNP_MIN_FRACTION,
)
from fiberhmm.io.bam_header import declared_chemistries


def test_ddda_derived_tf_edge_gap_is_ddda_recall_only():
    args = SimpleNamespace(
        ddda_derived_tf_max_edge_gap=12,
        enzyme="ddda",
    )
    assert _resolve_derived_tf_edge_gap(args, recall_nucs=True) == 12
    assert _resolve_derived_tf_edge_gap(args, recall_nucs=False) is None
    args.enzyme = "hia5"
    assert _resolve_derived_tf_edge_gap(args, recall_nucs=True) is None
    args.enzyme = "ddda"
    args.ddda_derived_tf_max_edge_gap = -1
    assert _resolve_derived_tf_edge_gap(args, recall_nucs=True) is None


@pytest.mark.parametrize(
    ("enzyme", "mode", "seq", "expected"),
    [
        ("ddda", "daf", None, ("daf", "ddda", "pacbio", "daf")),
        ("dddb", "daf", None, ("daf", "dddb", "nanopore", "daf")),
        ("hia5", "pacbio-fiber", None, ("fiber-seq", "hia5", "pacbio", "pacbio-fiber")),
        ("hia5", "nanopore-fiber", None, ("fiber-seq", "hia5", "nanopore", "nanopore-fiber")),
        ("ecogii", "nanopore-fiber", None, ("fiber-seq", "ecogii", "nanopore", "nanopore-fiber")),
    ],
)
def test_chemistry_declaration_records_platform(enzyme, mode, seq, expected):
    declaration = _chemistry_declaration(
        SimpleNamespace(enzyme=enzyme, seq=seq),
        mode,
        "/models/apply model.json",
        None,
    )
    assert tuple(declaration[key] for key in ("assay", "enzyme", "platform", "mode")) == expected
    assert declaration["model"] == "apply_model"


def test_ddda_chemistry_declaration_records_locked_nuc_profile():
    from fiberhmm.models import _bundled_model_path

    profile_path = _bundled_model_path("ddda_nuc_profile.json")
    identity = _nuc_profile_identity(profile_path)
    digest = _nuc_profile_sha256(profile_path)
    declaration = _chemistry_declaration(
        SimpleNamespace(enzyme="ddda", seq=None),
        "daf",
        "/models/ddda_nuc.json",
        "/models/ddda_TF.json",
        identity,
        digest,
    )

    assert identity == "ddda_phase_posterior_v1"
    assert declaration["nuc_model"] == "ddda_phase_posterior_v1"
    assert digest == "c86b05dc07e45392880e3460cf7f8880593ecad174e0a338d36ac53b7d0172d6"
    assert declaration["nuc_sha256"] == digest


def test_fiberhmm_call_stdout_is_clean_bam_stream(benchmark_model_path, tmp_path):
    input_bam = str(tmp_path / "input.bam")
    make_synthetic_bam(
        input_bam,
        n_reads=4,
        read_length=1500,
        n_chroms=1,
        chrom_length=20_000,
        seed=321,
    )

    cmd = [
        sys.executable, "-m", "fiberhmm.cli.call",
        "-i", input_bam,
        "-o", "-",
        "-m", benchmark_model_path,
        "--mode", "pacbio-fiber",
        "--min-read-length", "0",
        "--prob-threshold", "0",
        "--min-llr", "1000",
        "--chunk-size", "2",
        "--io-threads", "1",
        "-c", "1",
        "--max-reads", "4",
    ]

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert b"fiberhmm-call" in result.stderr
    assert b"fiberhmm-call" not in result.stdout

    stdout_bam = tmp_path / "stdout.bam"
    stdout_bam.write_bytes(result.stdout)
    with pysam.AlignmentFile(stdout_bam, "rb", check_sq=False) as bam:
        reads = list(bam.fetch(until_eof=True))
        chemistry = declared_chemistries(bam.header)

    assert len(reads) == 4
    assert chemistry == [{
        "assay": "fiber-seq",
        "enzyme": "custom",
        "platform": "pacbio",
        "mode": "pacbio-fiber",
        "model": Path(benchmark_model_path).stem,
    }]


def test_fiberhmm_call_rejects_development_ecogii_as_public_preset(tmp_path):
    input_bam = str(tmp_path / "ecogii_ont_input.bam")
    output_bam = str(tmp_path / "ecogii_ont_output.bam")
    make_synthetic_bam(
        input_bam,
        n_reads=3,
        read_length=600,
        n_chroms=1,
        chrom_length=10_000,
        mod_rate=0.30,
        seed=322,
    )
    command = [
        sys.executable, "-m", "fiberhmm.cli.call",
        "-i", input_bam,
        "-o", output_bam,
        "--enzyme", "ecogii",
        "--seq", "nanopore",
        "--min-read-length", "0",
        "--prob-threshold", "0",
        "--min-llr", "1000",
        "--chunk-size", "3",
        "--io-threads", "1",
        "--no-qc",
        "-c", "1",
    ]
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 2
    assert "invalid choice: 'ecogii'" in result.stderr


def test_daf_input_sniff_accepts_iupac_encoding(tmp_path):
    input_bam = str(tmp_path / "iupac.bam")
    make_synthetic_iupac_bam(
        input_bam,
        n_reads=4,
        read_length=200,
        n_chroms=1,
        chrom_length=5_000,
        seed=11,
    )

    _check_daf_inputs(input_bam, n_sniff=4)


def test_daf_input_sniff_accepts_reference_fallback(tmp_path):
    input_bam = str(tmp_path / "raw.bam")
    make_synthetic_bam(
        input_bam,
        n_reads=4,
        read_length=200,
        n_chroms=1,
        chrom_length=5_000,
        seed=12,
    )

    _check_daf_inputs(input_bam, reference="ref.fa", n_sniff=4)


def test_daf_input_sniff_rejects_missing_deamination_source(tmp_path, capsys):
    input_bam = str(tmp_path / "raw.bam")
    make_synthetic_bam(
        input_bam,
        n_reads=4,
        read_length=200,
        n_chroms=1,
        chrom_length=5_000,
        seed=13,
    )

    with pytest.raises(SystemExit) as exc:
        _check_daf_inputs(input_bam, n_sniff=4)

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "DAF-seq calling needs deamination calls" in err
    assert "--reference ref.fa" in err


def test_call_model_resolution_uses_custom_paths():
    args = SimpleNamespace(
        model="/tmp/custom_apply.json",
        recall_model="/tmp/custom_recall.json",
        enzyme=None,
        seq=None,
    )

    assert _resolve_apply_model(args) == "/tmp/custom_apply.json"
    assert _resolve_recall_model(args) == "/tmp/custom_recall.json"


def test_call_qc_is_default_on_and_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["fiberhmm-call", "-i", "in.bam", "-o", "out.bam", "--enzyme", "dddb"],
    )
    args = parse_call_args()
    assert args.qc is True
    assert args.qc_min_mapq == 20

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fiberhmm-call", "-i", "in.bam", "-o", "out.bam",
            "--enzyme", "dddb", "--no-qc",
        ],
    )
    assert parse_call_args().qc is False


def test_daf_snp_and_nondestructive_dedup_defaults(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["fiberhmm-call", "-i", "in.bam", "-o", "out.bam", "--enzyme", "dddb"],
    )
    args = parse_call_args()
    assert args.daf_call_snps is None
    assert args.daf_snp_min_fraction == DEFAULT_SNP_MIN_FRACTION
    assert args.daf_snp_min_depth == DEFAULT_SNP_MIN_DEPTH
    assert args.daf_snp_min_alt_fibers == DEFAULT_SNP_MIN_ALT_FIBERS
    assert args.dedup is None
    assert _resolve_dedup(args, "daf") is True
    assert args.dedup_collapse is False
    assert args.dedup_max_end_diff == 50

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fiberhmm-call", "-i", "in.bam", "-o", "out.bam",
            "--enzyme", "dddb", "--no-dedup",
        ],
    )
    no_dedup = parse_call_args()
    assert no_dedup.dedup is False
    assert _resolve_dedup(no_dedup, "daf") is False

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fiberhmm-call", "-i", "in.bam", "-o", "out.bam",
            "--enzyme", "hia5",
        ],
    )
    fiber = parse_call_args()
    assert _resolve_dedup(fiber, "pacbio-fiber") is False

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fiberhmm-call",
            "-i",
            "in.bam",
            "-o",
            "out.bam",
            "--enzyme",
            "dddb",
            "--daf-snp-min-fraction",
            "0.3",
            "--daf-snp-min-depth",
            "10",
            "--daf-snp-min-alt-fibers",
            "7",
        ],
    )
    custom = parse_call_args()
    assert (
        custom.daf_snp_min_fraction,
        custom.daf_snp_min_depth,
        custom.daf_snp_min_alt_fibers,
    ) == (0.3, 10, 7)


def _write_depth_preflight_bam(path: Path, n_reads: int) -> None:
    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 10_000}]}
    )
    with pysam.AlignmentFile(path, "wb", header=header) as bam:
        for index in range(n_reads):
            read = pysam.AlignedSegment(header)
            read.query_name = f"read_{index}"
            read.query_sequence = "C" * 200
            read.reference_id = 0
            read.reference_start = 100 + index
            read.mapping_quality = 60
            read.cigar = [(0, 200)]
            bam.write(read)


def test_daf_snp_preflight_skips_low_depth_and_triggers_supported_locus(tmp_path):
    low = tmp_path / "low.bam"
    high = tmp_path / "high.bam"
    _write_depth_preflight_bam(low, 8)
    _write_depth_preflight_bam(high, 25)

    low_result = _daf_snp_depth_preflight(str(low), min_local_depth=20)
    high_result = _daf_snp_depth_preflight(str(high), min_local_depth=20)

    assert low_result["run"] is False
    assert low_result["max_local_depth"] == 8
    assert high_result["run"] is True
    assert high_result["max_local_depth"] == 25


def test_call_model_resolution_uses_separate_ddda_models():
    args = SimpleNamespace(model=None, recall_model=None, enzyme="ddda", seq=None)

    apply_model = _resolve_apply_model(args)
    recall_model = _resolve_recall_model(args)

    assert apply_model.endswith("ddda_nuc.json")
    assert recall_model.endswith("ddda_TF.json")
    assert apply_model != recall_model


def test_call_model_resolution_requires_model_or_enzyme(capsys):
    args = SimpleNamespace(model=None, recall_model=None, enzyme=None, seq=None)

    with pytest.raises(SystemExit) as exc:
        _resolve_apply_model(args)

    assert exc.value.code == 1
    assert "one of --model or --enzyme required" in capsys.readouterr().err


def test_ddda_mode_surfaces_whole_genome_mcg_hint(capsys):
    args = SimpleNamespace(ddda_mcg=False, enzyme="ddda")
    assert _configure_ddda_mcg(args, "daf") is False
    message = capsys.readouterr().err
    assert "whole-genome DddA DAF-seq" in message
    assert "--ddda-mcg --reference ref.fa" in message
    assert "unnecessary for targeted/amplicon" in message


def test_integrated_ddda_mcg_guardrails(capsys, tmp_path):
    reference = tmp_path / "ref.fa"
    reference.write_text(">chr1\nACGTACGT\n")
    pysam.faidx(str(reference))
    valid = SimpleNamespace(
        ddda_mcg=True,
        enzyme="ddda",
        reference=str(reference),
        circular=False,
        downstream_compat=False,
        input="-",
    )
    assert _configure_ddda_mcg(valid, "daf") is True
    assert "integrated DddA mCG calling enabled" in capsys.readouterr().err

    invalid = SimpleNamespace(
        ddda_mcg=True,
        enzyme="dddb",
        reference=None,
        circular=True,
        downstream_compat=True,
        input="-",
    )
    with pytest.raises(SystemExit) as exc:
        _configure_ddda_mcg(invalid, "daf")
    assert exc.value.code == 2
    message = capsys.readouterr().err
    assert "requires --enzyme ddda" in message
    assert "requires --reference ref.fa" in message
    assert "does not support --circular" in message
    assert "cannot be combined with --downstream-compat" in message
