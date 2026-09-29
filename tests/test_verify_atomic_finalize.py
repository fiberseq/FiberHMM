"""Finalize (sort/index) on the temporary artifact, then publish (3.0 verify MEDIUM).

Regressions for the Codex verification pass on b006409.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pysam
import pytest
from conftest import make_synthetic_bam

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(module, *args, stdin=None, timeout=300):
    env = dict(os.environ)
    env.setdefault("FIBERHMM_NO_UPDATE_CHECK", "1")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", module, *map(str, args)],
        cwd=REPO_ROOT, capture_output=True, timeout=timeout, env=env,
        stdin=stdin,
    )


def _bundled(name):
    from fiberhmm.models import _bundled_model_path

    return _bundled_model_path(name)


def _pg_ds(path, program):
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
        records = [
            record for record in bam.header.to_dict().get("PG", [])
            if str(record.get("PN", "")).startswith(program)
        ]
    assert records, f"no {program} @PG record"
    return records[-1].get("DS", "")


def _ds_tokens(ds):
    return dict(
        token.split("=", 1) for token in ds.split() if "=" in token
    )


def _ma_tags(path):
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
        return [
            (read.query_name, read.get_tag("MA") if read.has_tag("MA") else None)
            for read in bam.fetch(until_eof=True)
        ]


def _declare(source, target, declaration):
    from fiberhmm.io.bam_header import append_chemistry

    with pysam.AlignmentFile(str(source), "rb", check_sq=False) as bam:
        header = append_chemistry(bam.header, declaration)
        with pysam.AlignmentFile(str(target), "wb", header=header) as out:
            for read in bam.fetch(until_eof=True):
                out.write(read)
    pysam.index(str(target))
    return target


# ---------------------------------------------------------------------------
# 3. finalize on the temporary artifact, then publish BAM + index together
# ---------------------------------------------------------------------------

def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _existing_output(tmp_path, source):
    out = tmp_path / "existing.bam"
    out.write_bytes(Path(source).read_bytes())
    pysam.index(str(out))
    return out, _digest(out), _digest(str(out) + ".bai")


def _assert_untouched(out, bam_digest, bai_digest):
    assert _digest(out) == bam_digest
    assert _digest(str(out) + ".bai") == bai_digest
    leftovers = [p.name for p in out.parent.iterdir()
                 if p.name.startswith(".") or ".sorting" in p.name]
    assert leftovers == []


def test_legacy_pipeline_finalization_failure_keeps_previous_output(tmp_path):
    from fiberhmm.core.model_io import load_model_with_metadata
    from fiberhmm.inference.legacy_pipeline import _process_bam_legacy_pipeline

    source = make_synthetic_bam(str(tmp_path / "src.bam"), n_reads=6,
                                read_length=1500, n_chroms=1,
                                chrom_length=20_000, seed=3)
    out, bam_digest, bai_digest = _existing_output(tmp_path, source)
    path = _bundled("hia5_nanopore.json")
    model, _, _ = load_model_with_metadata(path)
    with patch("fiberhmm.inference.legacy_pipeline._sort_and_index_bam",
               side_effect=OSError("injected finalization failure")):
        with pytest.raises(OSError, match="injected"):
            _process_bam_legacy_pipeline(
                source, str(out), model, path, set(), 10, False,
                "nanopore-fiber", 3, 0, n_cores=1, io_threads=1)
    os.remove(source)
    os.remove(source + ".bai")
    _assert_untouched(out, bam_digest, bai_digest)


def test_legacy_pipeline_publishes_bam_with_fresh_index(tmp_path):
    from fiberhmm.core.model_io import load_model_with_metadata
    from fiberhmm.inference.legacy_pipeline import _process_bam_legacy_pipeline

    source = make_synthetic_bam(str(tmp_path / "src.bam"), n_reads=6,
                                read_length=1500, n_chroms=1,
                                chrom_length=20_000, seed=3)
    out, bam_digest, _ = _existing_output(tmp_path, source)
    path = _bundled("hia5_nanopore.json")
    model, _, _ = load_model_with_metadata(path)
    _process_bam_legacy_pipeline(
        source, str(out), model, path, set(), 10, False,
        "nanopore-fiber", 3, 0, n_cores=1, io_threads=1)
    assert _digest(out) != bam_digest
    with pysam.AlignmentFile(str(out), "rb") as bam:
        assert bam.has_index()
        assert sum(1 for _ in bam.fetch("chr1")) == 6
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_atomic_output_finalize_failure_removes_temporaries(tmp_path):
    from fiberhmm.inference.bam_output import atomic_output

    out = tmp_path / "x.bam"
    out.write_bytes(b"old")
    (tmp_path / "x.bam.bai").write_bytes(b"old-index")

    def finalize(temporary):
        Path(temporary + ".bai").write_bytes(b"new-index")
        Path(temporary + ".sorting.bam").write_bytes(b"partial")
        raise RuntimeError("finalize failed")

    with pytest.raises(RuntimeError):
        with atomic_output(str(out), finalize=finalize) as temporary:
            Path(temporary).write_bytes(b"new")
    assert out.read_bytes() == b"old"
    assert (tmp_path / "x.bam.bai").read_bytes() == b"old-index"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.bam", "x.bam.bai"]

    def good(temporary):
        Path(temporary + ".bai").write_bytes(b"new-index")

    with atomic_output(str(out), finalize=good) as temporary:
        Path(temporary).write_bytes(b"new")
    assert out.read_bytes() == b"new"
    assert (tmp_path / "x.bam.bai").read_bytes() == b"new-index"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.bam", "x.bam.bai"]


@pytest.mark.parametrize("target", [
    "fiberhmm.inference.streaming_pipeline._sort_and_index_bam",
])
def test_streaming_pipeline_finalization_failure_keeps_previous_output(
        tmp_path, target):
    from fiberhmm.inference.streaming_pipeline import (
        _process_bam_streaming_pipeline,
    )

    source = make_synthetic_bam(str(tmp_path / "src.bam"), n_reads=6,
                                read_length=1500, n_chroms=1,
                                chrom_length=20_000, seed=3)
    out, bam_digest, bai_digest = _existing_output(tmp_path, source)
    path = _bundled("hia5_nanopore.json")
    with patch(target, side_effect=OSError("injected finalization failure")):
        with pytest.raises(OSError, match="injected"):
            _process_bam_streaming_pipeline(
                source, str(out), path, set(), 10, False,
                "nanopore-fiber", 3, 0, n_cores=1, io_threads=1)
    os.remove(source)
    os.remove(source + ".bai")
    _assert_untouched(out, bam_digest, bai_digest)


def _failing_index(*_args, **_kwargs):
    raise OSError("injected index failure")


def test_merge_index_failure_keeps_previous_output(tmp_path, monkeypatch):
    from fiberhmm.cli import merge

    bam = make_synthetic_bam(str(tmp_path / "in.bam"), n_reads=3,
                             read_length=500, n_chroms=1, chrom_length=20_000,
                             seed=31)
    outdir = tmp_path / "out"
    outdir.mkdir()
    output = outdir / "merged.bam"
    merge.run_merge(bam, str(output), io_threads=1)
    bam_digest, bai_digest = _digest(output), _digest(str(output) + ".bai")

    monkeypatch.setattr(merge.pysam, "index", _failing_index)
    with pytest.raises(OSError, match="injected"):
        merge.run_merge(bam, str(output), io_threads=1)
    _assert_untouched(output, bam_digest, bai_digest)


def test_pair_index_failure_keeps_previous_output(tmp_path, monkeypatch):
    from test_pair_unified import _write_sequence_resolved_fixture

    from fiberhmm.cli import duplex
    from fiberhmm.crossstrand.duplex import DuplexParams
    from fiberhmm.crossstrand.pairing import PairParams

    source, reference = _write_sequence_resolved_fixture(tmp_path)
    outdir = tmp_path / "out"
    outdir.mkdir()
    output = outdir / "paired.bam"
    kwargs = dict(
        params=DuplexParams(min_overlap_bp=100, min_nucs=1),
        sequence_params=PairParams(
            min_overlap_bp=100, min_nucs=1, min_sequence_bases=500,
            min_sequence_margin=0.002, max_sequence_pair_rate=0.01,
        ),
        pairing_mode="sequence-only", io_threads=1,
    )
    duplex.run_pairing(str(source), str(output), str(reference), **kwargs)
    bam_digest, bai_digest = _digest(output), _digest(str(output) + ".bai")

    monkeypatch.setattr(duplex.pysam, "index", _failing_index)
    with pytest.raises(OSError, match="injected"):
        duplex.run_pairing(str(source), str(output), str(reference), **kwargs)
    _assert_untouched(output, bam_digest, bai_digest)
