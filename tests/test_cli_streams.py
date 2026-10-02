"""stdout carries data only; logs and progress go to stderr (audit L14, L22)."""
import pysam
import pytest

from test_call_entrypoint_regressions import _run_cli, make_region_test_bam


def test_region_parallel_call_logs_to_stderr_only(tmp_path, benchmark_model_path):
    bam = make_region_test_bam(tmp_path / "in.bam", seed=5)
    result = _run_cli(
        "fiberhmm.cli.call", "-i", bam, "-o", tmp_path / "out.bam", "-m", benchmark_model_path,
        "--min-read-length", "0", "--prob-threshold", "0", "--no-qc", "--region-parallel",
        "--region-size", "5000", "-c", "2", "--io-threads", "1")
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout == b""
    assert b"Regions:" in result.stderr
    # L19: no samtools cat @PG listing the temporary work directory.
    with pysam.AlignmentFile(str(tmp_path / "out.bam"), "rb") as out:
        programs = out.header.to_dict().get("PG", [])
    assert not any(" cat " in p.get("CL", "") for p in programs)
    assert not any("fiberhmm-work" in p.get("CL", "") or "region_0" in p.get("CL", "")
                   for p in programs)
    assert programs[-1]["ID"].startswith("fiberhmm-call")


def test_strand_rescue_unknown_contig_is_a_one_line_error(tmp_path, capsys):
    from fiberhmm.cli.strand_rescue import main

    bam = tmp_path / "calls.bam"
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chrDemo", "LN": 30000}]}
    with pysam.AlignmentFile(str(bam), "wb", header=header):
        pass
    pysam.index(str(bam))
    with pytest.raises(SystemExit) as raised:
        main(["-i", str(bam), "--preset", "dddb", "--region", "chr3L:100-200",
              "-o", str(tmp_path / "sr.json")])
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert "chr3L" in err and "chrDemo" in err and "Traceback" not in err
    assert not (tmp_path / "sr.json").exists()


def test_tolerated_region_worker_failures_are_reported_on_stderr(capsys):
    from types import SimpleNamespace

    from fiberhmm.inference.region_pipeline import _enforce_region_failures

    aggregation = SimpleNamespace(metrics={"worker_failures": 1}, total_reads=100_000,
                                  failure_messages=("Traceback: boom",))
    _enforce_region_failures(aggregation)
    captured = capsys.readouterr()
    assert captured.out == "" and "boom" in captured.err
