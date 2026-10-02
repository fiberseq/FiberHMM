"""Regression tests for fiberhmm-call/apply/recall entry-point plumbing.

Each test reproduces a release-audit finding (2026-09-29) against the public
CLI or the pipeline function the CLI dispatches to.
"""
from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path

import pysam
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(module: str, *args, stdin=None, timeout=180):
    env = dict(os.environ)
    env.setdefault("FIBERHMM_NO_UPDATE_CHECK", "1")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", module, *map(str, args)],
        cwd=REPO_ROOT,
        input=stdin,
        capture_output=True,
        timeout=timeout,
        env=env,
    )


def make_daf_chimera_bam(path, length=3000, seed=7):
    """Two MD-tagged DddB-like reads: one clean C->T read and one strand-swap
    chimera (C->T in the first half, G->A in the second)."""
    rng = random.Random(seed)
    ref = "".join(rng.choice("ACGT") for _ in range(length))

    def make(ct_range, ga_range, n=40):
        query = list(ref)
        cs = [i for i in range(*ct_range) if ref[i] == "C"]
        gs = [i for i in range(*ga_range) if ref[i] == "G"]
        mismatches = {}
        for i in rng.sample(cs, n):
            query[i] = "T"
            mismatches[i] = "C"
        if ga_range[1] > ga_range[0]:
            for i in rng.sample(gs, n // 3):
                query[i] = "A"
                mismatches[i] = "G"
        md, run = [], 0
        for i in range(length):
            if i in mismatches:
                md.append(str(run))
                md.append(mismatches[i])
                run = 0
            else:
                run += 1
        md.append(str(run))
        return "".join(query), "".join(md)

    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": "chrT", "LN": length + 10}]}
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for name, ct, ga in (("normal_ct", (0, length), (0, 0)),
                             ("chimera", (0, length // 2), (length // 2, length))):
            seq, md = make(ct, ga)
            read = pysam.AlignedSegment(out.header)
            read.query_name = name
            read.query_sequence = seq
            read.flag = 0
            read.reference_id = 0
            read.reference_start = 0
            read.mapping_quality = 60
            read.cigartuples = [(0, length)]
            read.query_qualities = pysam.qualitystring_to_array("I" * length)
            read.set_tag("MD", md)
            out.write(read)
    pysam.index(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# Item 1: fiberhmm-apply -c 1 crashed on the first DAF strand-swap chimera.
# ---------------------------------------------------------------------------

def test_apply_single_core_passes_daf_chimera_through(tmp_path):
    bam = make_daf_chimera_bam(tmp_path / "chim.bam")
    outdir = tmp_path / "out"
    result = _run_cli(
        "fiberhmm.cli.apply", "-i", bam, "-o", outdir, "--enzyme", "dddb",
        "-c", "1", "--io-threads", "1",
    )
    stderr = result.stderr.decode(errors="replace")
    stdout = result.stdout.decode(errors="replace")
    assert result.returncode == 0, stderr + stdout
    assert "chimera: 1" in stdout
    with pysam.AlignmentFile(str(outdir / "chim_footprints.bam"), "rb",
                             check_sq=False) as handle:
        reads = {read.query_name: read for read in handle.fetch(until_eof=True)}
    assert set(reads) == {"normal_ct", "chimera"}
    assert not reads["chimera"].has_tag("ns")


# ---------------------------------------------------------------------------
# Item 2: fiberhmm-apply -o - at -c 1 crashed after writing (getsize('-')).
# ---------------------------------------------------------------------------

def test_apply_single_core_stdout_is_complete_bam(tmp_path, benchmark_model_path):
    from conftest import make_synthetic_bam

    bam = str(tmp_path / "input.bam")
    make_synthetic_bam(bam, n_reads=4, read_length=1500, n_chroms=1,
                       chrom_length=20_000, seed=5)
    result = _run_cli(
        "fiberhmm.cli.apply", "-i", bam, "-o", "-", "-m", benchmark_model_path,
        "--min-read-length", "0", "-c", "1", "--io-threads", "1",
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    streamed = tmp_path / "stdout.bam"
    streamed.write_bytes(result.stdout)
    with pysam.AlignmentFile(str(streamed), "rb", check_sq=False) as handle:
        assert len(list(handle.fetch(until_eof=True))) == 4


# ---------------------------------------------------------------------------
# Shared helpers for pipeline-level tests.
# ---------------------------------------------------------------------------

def _fused_kwargs(model_path, **overrides):
    kwargs = dict(
        model_path=model_path, recall_model_path=None, train_rids=set(),
        edge_trim=10, circular=False, mode="pacbio-fiber", context_size=3,
        msp_min_size=0, nuc_min_size=85, min_mapq=0, prob_threshold=0,
        min_read_length=0, with_scores=False, min_llr=5.0, min_opps=3,
        unify_threshold=90, emission_uplift=1.0, also_write_legacy=True,
        downstream_compat=False, max_reads=0, n_cores=1, chunk_size=16,
        io_threads=1,
    )
    kwargs.update(overrides)
    return kwargs


def _names(path):
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as handle:
        return [read.query_name for read in handle.fetch(until_eof=True)]


def _leftovers(directory):
    return sorted(p.name for p in Path(directory).iterdir()
                  if p.name.startswith(".") and p.name.endswith(".tmp"))


# ---------------------------------------------------------------------------
# Item 8: per-read worker exceptions were swallowed (exit 0, untagged BAM).
# Item 9: failed runs must not leave valid-looking outputs.
# ---------------------------------------------------------------------------

def test_fused_streaming_fails_when_every_read_fails(tmp_path, monkeypatch,
                                                     benchmark_model_path, capsys):
    from conftest import make_synthetic_bam

    from fiberhmm.inference import streaming_workers
    from fiberhmm.inference.streaming_pipeline import (
        _process_bam_streaming_pipeline_fused,
    )
    from fiberhmm.inference.worker_results import WorkerFailureError

    bam = str(tmp_path / "in.bam")
    make_synthetic_bam(bam, n_reads=6, read_length=1500, n_chroms=1,
                       chrom_length=20_000, seed=3)

    def boom(*_args, **_kwargs):
        raise RuntimeError("injected recall failure")

    monkeypatch.setattr(streaming_workers, "build_fused_recall_result", boom)
    output = tmp_path / "out.bam"
    with pytest.raises(WorkerFailureError):
        _process_bam_streaming_pipeline_fused(
            input_bam=bam, output_bam=str(output),
            **_fused_kwargs(benchmark_model_path),
        )
    captured = capsys.readouterr()
    assert "injected recall failure" in captured.out + captured.err
    assert not output.exists()
    assert _leftovers(tmp_path) == []


def test_fused_streaming_tolerates_rare_failures(tmp_path, monkeypatch,
                                                 benchmark_model_path, capsys):
    from conftest import make_synthetic_bam

    from fiberhmm.inference import streaming_workers
    from fiberhmm.inference.streaming_pipeline import (
        _process_bam_streaming_pipeline_fused,
    )

    bam = str(tmp_path / "in.bam")
    make_synthetic_bam(bam, n_reads=150, read_length=400, n_chroms=1,
                       chrom_length=200_000, seed=4)
    victim = _names(bam)[7]
    real = streaming_workers.build_fused_recall_result

    def flaky(fiber_read, *args, **kwargs):
        if fiber_read.get("read_id") == victim:
            raise RuntimeError("injected single failure")
        return real(fiber_read, *args, **kwargs)

    monkeypatch.setattr(streaming_workers, "build_fused_recall_result", flaky)
    output = tmp_path / "out.bam"
    _process_bam_streaming_pipeline_fused(
        input_bam=bam, output_bam=str(output),
        **_fused_kwargs(benchmark_model_path, chunk_size=50),
    )
    captured = capsys.readouterr()
    assert "injected single failure" in captured.out + captured.err
    assert len(_names(output)) == 150


def test_atomic_output_publishes_only_on_success(tmp_path):
    from fiberhmm.inference.bam_output import atomic_output

    final = tmp_path / "result.bam"
    stale_index = tmp_path / "result.bam.bai"
    final.write_text("old")
    stale_index.write_text("old index")

    with pytest.raises(RuntimeError):
        with atomic_output(str(final)) as temporary:
            Path(temporary).write_text("partial")
            raise RuntimeError("crash mid-write")
    assert final.read_text() == "old"
    assert _leftovers(tmp_path) == []

    with atomic_output(str(final)) as temporary:
        assert Path(temporary).parent == tmp_path
        Path(temporary).write_text("new")
    assert final.read_text() == "new"
    assert not stale_index.exists()
    assert _leftovers(tmp_path) == []

    with atomic_output("-") as target:
        assert target == "-"


# ---------------------------------------------------------------------------
# Item 6: hard-clipped supplementary/secondary records were called with the
# parent read's MM/ML.
# ---------------------------------------------------------------------------

def _segment(header, seq, cigar, flag=0, tags=()):
    read = pysam.AlignedSegment(header)
    read.query_name = "r"
    read.query_sequence = seq
    read.flag = flag
    read.reference_id = 0
    read.reference_start = 100
    read.mapping_quality = 60
    read.cigartuples = cigar
    for tag, value in tags:
        read.set_tag(tag, value)
    return read


def test_hard_clip_guard_policy():
    from fiberhmm.inference.read_filters import (
        HARD_CLIPPED_MM,
        ReadFilterConfig,
        streaming_skip_reason,
    )

    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 10_000}]})
    seq = "ACGTAACGTA" * 20
    mm = (("MM", "A+a,0,1;"), ("ML", [200, 200]))
    clipped = [(5, 50), (0, len(seq))]
    config = ReadFilterConfig(mode="pacbio-fiber")

    primary = _segment(header, seq, [(0, len(seq))], tags=mm)
    assert streaming_skip_reason(primary, config) is None
    supplementary = _segment(header, seq, clipped, flag=2048, tags=mm)
    assert streaming_skip_reason(supplementary, config) == HARD_CLIPPED_MM
    rewritten = _segment(header, seq, clipped, flag=2048,
                         tags=mm + (("MN", len(seq)),))
    assert streaming_skip_reason(rewritten, config) is None
    stale = _segment(header, seq, clipped, flag=2048,
                     tags=mm + (("MN", len(seq) + 50),))
    assert streaming_skip_reason(stale, config) == HARD_CLIPPED_MM
    soft = _segment(header, seq, [(4, 50), (0, len(seq) - 50)], flag=2048, tags=mm)
    assert streaming_skip_reason(soft, config) is None
    daf_md = _segment(header, seq, clipped, flag=2048,
                      tags=mm + (("MD", str(len(seq))),))
    assert streaming_skip_reason(daf_md, ReadFilterConfig(mode="daf")) is None
    assert streaming_skip_reason(daf_md, config) == HARD_CLIPPED_MM


def test_call_skips_hard_clipped_supplementary(tmp_path, benchmark_model_path):
    from conftest import make_synthetic_bam

    source = str(tmp_path / "src.bam")
    make_synthetic_bam(source, n_reads=3, read_length=1500, n_chroms=1,
                       chrom_length=20_000, seed=9)
    bam = tmp_path / "clipped.bam"
    with pysam.AlignmentFile(source, "rb") as src, \
            pysam.AlignmentFile(str(bam), "wb", header=src.header) as out:
        for index, read in enumerate(src.fetch(until_eof=True)):
            if index == 1:
                # minimap2 without -Y: SEQ trimmed to the aligned piece, MM/ML
                # still describing the whole molecule.
                seq, qual = read.query_sequence, read.query_qualities
                read.query_sequence = seq[200:]
                read.query_qualities = qual[200:]
                read.cigartuples = [(5, 200), (0, len(seq) - 200)]
                read.flag |= 2048
            out.write(read)
    pysam.index(str(bam))
    output = tmp_path / "out.bam"
    # --primary is the default since 3.0 (supplementary records pass through
    # before the hard-clip guard is reached); --no-primary exercises the guard.
    result = _run_cli(
        "fiberhmm.cli.call", "-i", bam, "-o", output, "-m", benchmark_model_path,
        "--min-read-length", "0", "--prob-threshold", "0", "--no-qc",
        "--no-recall-nucs", "-c", "1", "--io-threads", "1", "--no-primary",
    )
    stderr = result.stderr.decode(errors="replace")
    stdout = result.stdout.decode(errors="replace")
    assert result.returncode == 0, stderr
    assert "hard_clipped_mm: 1" in stdout + stderr
    with pysam.AlignmentFile(str(output), "rb") as handle:
        reads = list(handle.fetch(until_eof=True))
    assert len(reads) == 3
    clipped = [read for read in reads if read.is_supplementary]
    assert len(clipped) == 1
    assert not clipped[0].has_tag("MA") and not clipped[0].has_tag("ns")


# ---------------------------------------------------------------------------
# Item 10: re-calling kept stale MA/AQ/ns on reads the new run skipped.
# ---------------------------------------------------------------------------

def test_strip_stale_call_tags_keeps_m5c_groups():
    from fiberhmm.inference.read_filters import strip_stale_call_tags

    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 10_000}]})
    seq = "ACGT" * 50
    read = _segment(header, seq, [(0, len(seq))], tags=(
        ("ns", [10]), ("nl", [100]), ("as", [0]), ("al", [10]),
        ("MA", "200;nuc+Q:11-100;ddda_ucg.:5-20"),
        ("AQ", [30]), ("AN", "fh_nuc_0,isl_0"),
    ))
    strip_stale_call_tags(read)
    assert not read.has_tag("ns") and not read.has_tag("as")
    assert read.get_tag("MA") == "200;ddda_ucg.:5-20"
    assert read.get_tag("AN") == "isl_0"
    assert not read.has_tag("AQ")

    wholesale = _segment(header, seq, [(0, len(seq))], tags=(
        ("MA", "200;nuc+Q:11-100;ddda_ucg.:5-20"), ("AQ", [30]),
        ("as", [0]), ("al", [10]),
    ))
    strip_stale_call_tags(wholesale, legacy_tags=("ns", "nl", "nq"),
                          keep_m5c_groups=False)
    assert not wholesale.has_tag("MA") and not wholesale.has_tag("AQ")
    assert wholesale.has_tag("as"), "apply --no-msps keeps existing MSP tags"


def test_recall_with_stricter_filter_drops_stale_calls(tmp_path,
                                                       benchmark_model_path):
    from conftest import make_synthetic_bam

    source = str(tmp_path / "src.bam")
    make_synthetic_bam(source, n_reads=3, read_length=1500, n_chroms=1,
                       chrom_length=20_000, seed=10)
    bam = tmp_path / "called_before.bam"
    with pysam.AlignmentFile(source, "rb") as src, \
            pysam.AlignmentFile(str(bam), "wb", header=src.header) as out:
        for read in src.fetch(until_eof=True):
            read.set_tag("ns", [5, 400])
            read.set_tag("nl", [150, 147])
            read.set_tag("MA", f"{read.query_length};nuc+Q:6-150,401-147")
            read.set_tag("AQ", [200, 180])
            out.write(read)
    pysam.index(str(bam))
    output = tmp_path / "recalled.bam"
    result = _run_cli(
        "fiberhmm.cli.call", "-i", bam, "-o", output, "-m", benchmark_model_path,
        "--min-mapq", "61", "--min-read-length", "0", "--no-qc",
        "-c", "1", "--io-threads", "1",
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    with pysam.AlignmentFile(str(output), "rb") as handle:
        reads = list(handle.fetch(until_eof=True))
    assert len(reads) == 3
    for read in reads:
        for tag in ("ns", "nl", "MA", "AQ"):
            assert not read.has_tag(tag), tag


# ---------------------------------------------------------------------------
# Item 7: region-parallel duplicated secondary/supplementary records with
# --primary, exited 0 with an empty BAM on unindexed input, dropped unmapped
# and skipped-contig reads, and misclassified main chromosomes.
# ---------------------------------------------------------------------------

def make_region_test_bam(path, seed=21, n_reads=24, sort=True):
    """Aligned reads on chr2L (with secondary/supplementary records crossing
    2 kb region boundaries), one read on a scaffold, one unplaced unmapped."""
    from conftest import make_synthetic_bam

    source = str(Path(path).with_suffix(".src.bam"))
    make_synthetic_bam(source, n_reads=n_reads, read_length=1500, n_chroms=1,
                       chrom_length=60_000, seed=seed)
    with pysam.AlignmentFile(source, "rb") as src:
        reads = list(src.fetch(until_eof=True))
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": "chr2L", "LN": 60_000},
                     {"SN": "chrUn_scaffold", "LN": 10_000}]}
    unsorted = str(Path(path).with_suffix(".unsorted.bam"))
    with pysam.AlignmentFile(unsorted, "wb", header=header) as out:
        for index, read in enumerate(reads):
            record = pysam.AlignedSegment(out.header)
            record.query_name = read.query_name
            record.query_sequence = read.query_sequence
            record.query_qualities = read.query_qualities
            record.flag = read.flag
            record.reference_id = 0
            record.reference_start = 900 + index * 1700
            record.mapping_quality = 60
            record.cigartuples = [(0, len(read.query_sequence))]
            for tag, value in read.get_tags():
                record.set_tag(tag, value)
            if index % 5 == 1:
                record.flag |= 2048
            elif index % 5 == 2:
                record.flag |= 256
            out.write(record)
        scaffold = pysam.AlignedSegment(out.header)
        scaffold.query_name = "scaffold_read"
        scaffold.query_sequence = reads[0].query_sequence
        scaffold.query_qualities = reads[0].query_qualities
        scaffold.reference_id = 1
        scaffold.reference_start = 100
        scaffold.mapping_quality = 60
        scaffold.cigartuples = [(0, len(reads[0].query_sequence))]
        for tag, value in reads[0].get_tags():
            scaffold.set_tag(tag, value)
        out.write(scaffold)
        unmapped = pysam.AlignedSegment(out.header)
        unmapped.query_name = "unmapped_read"
        unmapped.query_sequence = reads[1].query_sequence
        unmapped.query_qualities = reads[1].query_qualities
        unmapped.flag = 4
        out.write(unmapped)
    if sort:
        pysam.sort("-o", str(path), unsorted)
        pysam.index(str(path))
    else:
        os.replace(unsorted, str(path))
    return str(path)


def _call_region(bam, output, model, *extra):
    return _run_cli(
        "fiberhmm.cli.call", "-i", bam, "-o", output, "-m", model,
        "--min-read-length", "0", "--prob-threshold", "0", "--no-qc",
        "--no-recall-nucs", "--region-parallel", "--region-size", "2000",
        "-c", "2", "--io-threads", "1", *extra,
    )


def test_region_parallel_primary_keeps_every_record_once(tmp_path,
                                                         benchmark_model_path):
    from collections import Counter

    bam = make_region_test_bam(tmp_path / "in.bam")
    output = tmp_path / "out.bam"
    result = _call_region(bam, output, benchmark_model_path, "--primary",
                          "--skip-scaffolds")
    assert result.returncode == 0, result.stderr.decode(errors="replace")

    def keys(path):
        with pysam.AlignmentFile(str(path), "rb", check_sq=False) as handle:
            return Counter((r.query_name, r.flag, r.reference_id, r.reference_start)
                           for r in handle.fetch(until_eof=True))

    assert keys(output) == keys(bam)
    with pysam.AlignmentFile(str(output), "rb") as handle:
        by_name = {r.query_name: r for r in handle.fetch(until_eof=True)
                   if not (r.is_secondary or r.is_supplementary)}
    assert "unmapped_read" in by_name and "scaffold_read" in by_name
    assert not by_name["scaffold_read"].has_tag("MA")
    # Owned primary reads on the processed contig were actually called.
    assert any(read.has_tag("MA") for name, read in by_name.items()
               if name.startswith("read_"))


def test_region_parallel_refuses_unindexed_input(tmp_path, benchmark_model_path):
    bam = make_region_test_bam(tmp_path / "in.bam", sort=False)
    output = tmp_path / "out.bam"
    result = _call_region(bam, output, benchmark_model_path)
    assert result.returncode != 0
    assert b"index" in result.stderr
    assert not output.exists()


def test_region_parallel_rejects_unknown_chroms(tmp_path, benchmark_model_path):
    bam = make_region_test_bam(tmp_path / "in.bam")
    output = tmp_path / "out.bam"
    result = _call_region(bam, output, benchmark_model_path, "--chroms", "chrNope")
    assert result.returncode != 0
    assert b"chrNope" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("name, expected", [
    ("chrVII", True), ("chrXVI", True), ("chrI", True), ("NC_000001.11", True),
    ("chrEBV", True), ("chr2L", True), ("chr1", True), ("chrX", True),
    ("chrUn_KI270302v1", False), ("chr1_KI270706v1_random", False),
    ("NW_003315947.1", False), ("scaffold_12", False), ("chrC", False),
])
def test_skip_scaffolds_main_chromosome_names(name, expected):
    from fiberhmm.inference.region_planning import _is_main_chromosome

    assert _is_main_chromosome(name) is expected


def test_region_plan_errors_when_nothing_selected(tmp_path):
    from fiberhmm.inference.region_planning import RegionPlanError, plan_region_work

    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": "chrUn_a", "LN": 1000}]}
    path = tmp_path / "scaffold_only.bam"
    with pysam.AlignmentFile(str(path), "wb", header=header):
        pass
    with pytest.raises(RegionPlanError, match="no genomic regions"):
        plan_region_work(str(path), 500, skip_scaffolds=True)


def test_region_parallel_fused_fails_on_worker_errors(tmp_path, monkeypatch,
                                                      benchmark_model_path):
    from fiberhmm.inference import region_workers
    from fiberhmm.inference.region_pipeline import _process_bam_region_parallel_fused
    from fiberhmm.inference.worker_results import WorkerFailureError

    bam = make_region_test_bam(tmp_path / "in.bam")

    def boom(*_args, **_kwargs):
        raise RuntimeError("injected region failure")

    monkeypatch.setattr(region_workers, "build_fused_recall_result", boom)
    kwargs = _fused_kwargs(benchmark_model_path)
    for key in ("model_path", "max_reads", "chunk_size"):
        kwargs.pop(key)
    output = tmp_path / "out.bam"
    with pytest.raises(WorkerFailureError):
        _process_bam_region_parallel_fused(
            input_bam=bam, output_bam=str(output),
            apply_model_path=benchmark_model_path,
            region_size=20_000, skip_scaffolds=False, chroms=None,
            **{k: v for k, v in kwargs.items() if k != "recall_model_path"},
            recall_model_path=None,
        )
    assert not output.exists()


# ---------------------------------------------------------------------------
# Item 3: the README uBAM example skipped 100% of reads as unmapped, exit 0.
# ---------------------------------------------------------------------------

def _make_headerless_ubam(path, seed=12, n_reads=5):
    """A real uBAM: unmapped reads, no @SQ lines, no index."""
    from conftest import make_synthetic_bam

    source = make_synthetic_bam(str(path) + ".src.bam", n_reads=n_reads,
                                read_length=1500, n_chroms=1,
                                chrom_length=20_000, seed=seed, aligned=False)
    with pysam.AlignmentFile(source, "rb", check_sq=False) as src:
        reads = list(src.fetch(until_eof=True))
    header = pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6", "SO": "unknown"}})
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for read in reads:
            record = pysam.AlignedSegment(header)
            record.query_name = read.query_name
            record.query_sequence = read.query_sequence
            record.query_qualities = read.query_qualities
            record.flag = 4
            for tag, value in read.get_tags():
                record.set_tag(tag, value)
            out.write(record)
    return str(path)


def _called(path):
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as handle:
        return sum(1 for read in handle.fetch(until_eof=True) if read.has_tag("MA"))


@pytest.mark.parametrize("via_stdin", [False, True])
def test_call_calls_unaligned_bam(tmp_path, benchmark_model_path, via_stdin):
    ubam = _make_headerless_ubam(tmp_path / "reads.ubam")
    output = tmp_path / "calls.bam"
    common = ["-o", output, "-m", benchmark_model_path, "--min-read-length", "0",
              "--prob-threshold", "0", "--no-qc", "-c", "1", "--io-threads", "1"]
    if via_stdin:
        result = _run_cli("fiberhmm.cli.call", "-i", "-", *common,
                          stdin=Path(ubam).read_bytes())
    else:
        result = _run_cli("fiberhmm.cli.call", "-i", ubam, *common)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert _called(output) == 5


def test_call_fails_when_nearly_everything_is_skipped_as_unmapped(
        tmp_path, benchmark_model_path):
    from conftest import make_synthetic_bam

    bam = make_synthetic_bam(str(tmp_path / "unmapped.bam"), n_reads=5,
                             read_length=1500, n_chroms=1, chrom_length=20_000,
                             seed=13, aligned=False)
    pysam.index(bam)  # indexed + @SQ: auto mode leaves unmapped reads alone
    output = tmp_path / "calls.bam"
    base = ["-i", bam, "-o", output, "-m", benchmark_model_path,
            "--min-read-length", "0", "--no-qc", "-c", "1", "--io-threads", "1"]
    result = _run_cli("fiberhmm.cli.call", *base)
    assert result.returncode == 1
    assert b"skipped as unmapped" in result.stderr
    assert not output.exists()

    result = _run_cli("fiberhmm.cli.call", *base, "--no-process-unmapped")
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert output.exists() and _called(output) == 0


def test_phase_nrl_estimator_samples_processed_unmapped_reads(tmp_path,
                                                             benchmark_model_path):
    from conftest import make_synthetic_bam

    from fiberhmm.inference.nrl_estimate import estimate_phase_nrl

    bam = make_synthetic_bam(str(tmp_path / "u.bam"), n_reads=6, read_length=3000,
                             n_chroms=1, chrom_length=20_000, seed=14,
                             aligned=False, mod_rate=0.3)
    kwargs = dict(mode="pacbio-fiber", context_size=3, prob_threshold=0)
    without = estimate_phase_nrl(bam, benchmark_model_path, **kwargs)
    with_unmapped = estimate_phase_nrl(bam, benchmark_model_path,
                                       include_unmapped=True, **kwargs)
    assert without["n_reads"] == 0
    assert with_unmapped["n_reads"] > 0


def test_apply_calls_unaligned_bam_at_single_core(tmp_path, benchmark_model_path):
    ubam = _make_headerless_ubam(tmp_path / "reads.ubam", seed=15)
    outdir = tmp_path / "out"
    result = _run_cli("fiberhmm.cli.apply", "-i", ubam, "-o", outdir,
                      "-m", benchmark_model_path, "--min-read-length", "0",
                      "-c", "1", "--io-threads", "1")
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    with pysam.AlignmentFile(str(outdir / "reads.ubam_footprints.bam"), "rb",
                             check_sq=False) as handle:
        assert sum(read.has_tag("ns") for read in handle.fetch(until_eof=True)) > 0


# ---------------------------------------------------------------------------
# Item 4: custom --model recall/call crashed on FiberHMM-called BAMs
# (enzyme=custom vs the declared chemistry).
# ---------------------------------------------------------------------------

def _chemistry(path):
    from fiberhmm.io.bam_header import declared_chemistries

    # Core fields of each declaration, de-duplicated: a re-call with another
    # model file appends a second declaration that differs only in `model`.
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as handle:
        cores = []
        for item in declared_chemistries(handle.header):
            core = {key: item[key] for key in ("assay", "enzyme", "platform", "mode")}
            if core not in cores:
                cores.append(core)
        return cores


def _copy_model(name, destination):
    import shutil

    from fiberhmm.models import _bundled_model_path

    shutil.copy(_bundled_model_path(name), destination)
    return str(destination)


def test_recall_tfs_custom_model_on_dddb_called_bam(tmp_path):
    bam = make_daf_chimera_bam(tmp_path / "raw.bam")
    called = tmp_path / "called.bam"
    result = _run_cli(
        "fiberhmm.cli.call", "-i", bam, "-o", called, "--enzyme", "dddb",
        "--no-dedup", "--no-daf-call-snps", "--no-qc", "--no-recall-nucs",
        "--min-read-length", "0", "-c", "1", "--io-threads", "1",
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    declared = {"assay": "daf", "enzyme": "dddb", "platform": "nanopore",
                "mode": "daf"}
    assert _chemistry(called) == [declared]

    refit = _copy_model("dddb_nanopore.json", tmp_path / "refit.json")
    recalled = tmp_path / "recalled.bam"
    result = _run_cli("fiberhmm.cli.recall_tfs", "-i", called, "-o", recalled,
                      "--model", refit)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert _chemistry(recalled) == [declared]

    recalled_call = tmp_path / "recalled_call.bam"
    result = _run_cli(
        "fiberhmm.cli.call", "-i", called, "-o", recalled_call, "-m", refit,
        "--no-dedup", "--no-daf-call-snps", "--no-qc", "--no-recall-nucs",
        "--min-read-length", "0", "-c", "1", "--io-threads", "1",
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert _chemistry(recalled_call) == [declared]


def test_recall_tfs_chemistry_conflict_is_one_line_and_replaceable(tmp_path):
    from conftest import make_synthetic_bam

    bam = make_synthetic_bam(str(tmp_path / "in.bam"), n_reads=3,
                             read_length=1500, n_chroms=1, chrom_length=20_000,
                             seed=16)
    called = tmp_path / "called.bam"
    result = _run_cli(
        "fiberhmm.cli.call", "-i", bam, "-o", called, "--enzyme", "hia5",
        "--seq", "pacbio", "--force-seq", "--no-qc", "--min-read-length", "0",
        "--prob-threshold", "0", "-c", "1", "--io-threads", "1",
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")

    conflicted = tmp_path / "conflict.bam"
    result = _run_cli("fiberhmm.cli.recall_tfs", "-i", called, "-o", conflicted,
                      "--enzyme", "hia5", "--seq", "nanopore")
    stderr = result.stderr.decode(errors="replace")
    assert result.returncode == 2
    assert "Traceback" not in stderr
    assert "--replace-chemistry" in stderr and "--enzyme/--seq" in stderr
    assert not conflicted.exists()

    replaced = tmp_path / "replaced.bam"
    result = _run_cli("fiberhmm.cli.recall_tfs", "-i", called, "-o", replaced,
                      "--enzyme", "hia5", "--seq", "nanopore",
                      "--replace-chemistry")
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert _chemistry(replaced) == [{"assay": "fiber-seq", "enzyme": "hia5",
                                     "platform": "nanopore",
                                     "mode": "nanopore-fiber"}]


def test_custom_model_reuses_itself_for_tf_recall():
    from types import SimpleNamespace

    from fiberhmm.cli.call import _resolve_recall_model

    custom = SimpleNamespace(recall_model=None, model="refit.json",
                             enzyme="hia5", seq=None)
    assert _resolve_recall_model(custom) is None  # reuse -m, not hia5_pacbio
    custom.seq = "nanopore"
    assert _resolve_recall_model(custom) is None
    ddda = SimpleNamespace(recall_model=None, model="nuc.json",
                           enzyme="ddda", seq=None)
    assert _resolve_recall_model(ddda).endswith("ddda_TF.json")
    explicit = SimpleNamespace(recall_model="tf.json", model="x.json",
                               enzyme="hia5", seq=None)
    assert _resolve_recall_model(explicit) == "tf.json"


# ---------------------------------------------------------------------------
# Item 5: a missing --seq silently meant PacBio (and declared it) on ONT data.
# ---------------------------------------------------------------------------

def _with_mm_style(source, path, pacbio=(), ont=()):
    """Copy ``source`` giving the reads at the listed indices PacBio-style
    (A+a plus T-a) MM specs; all other reads keep ONT-style A+a only."""
    with pysam.AlignmentFile(source, "rb") as src, \
            pysam.AlignmentFile(str(path), "wb", header=src.header) as out:
        for index, read in enumerate(src.fetch(until_eof=True)):
            if index in pacbio:
                read.set_tag("MM", read.get_tag("MM") + "T-a;")
            out.write(read)
    pysam.index(str(path))
    return str(path)


def test_platform_sniff_from_mm_specs(tmp_path):
    from conftest import make_synthetic_bam

    from fiberhmm.cli.common import sniff_sequencing_platform

    source = make_synthetic_bam(str(tmp_path / "src.bam"), n_reads=10,
                                read_length=600, n_chroms=1,
                                chrom_length=20_000, seed=17)
    ont = sniff_sequencing_platform(source)
    assert ont.platform == "nanopore" and ont.conflict is None
    pacbio = _with_mm_style(source, tmp_path / "pb.bam", pacbio=range(10))
    assert sniff_sequencing_platform(pacbio).platform == "pacbio"
    mixed = _with_mm_style(source, tmp_path / "mixed.bam", pacbio=range(5))
    evidence = sniff_sequencing_platform(mixed)
    assert evidence.platform is None and "mixed" in evidence.conflict
    assert sniff_sequencing_platform("-").platform is None


def test_call_without_seq_declares_detected_platform(tmp_path,
                                                     benchmark_model_path):
    from conftest import make_synthetic_bam

    bam = make_synthetic_bam(str(tmp_path / "ont.bam"), n_reads=3,
                             read_length=1500, n_chroms=1, chrom_length=20_000,
                             seed=18)
    output = tmp_path / "calls.bam"
    base = ["-o", output, "--enzyme", "hia5", "--no-qc", "--min-read-length", "0",
            "-c", "1", "--io-threads", "1"]
    result = _run_cli("fiberhmm.cli.call", "-i", bam, *base)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert b"--seq nanopore" in result.stderr
    assert _chemistry(output)[0]["platform"] == "nanopore"
    assert _chemistry(output)[0]["mode"] == "nanopore-fiber"

    mixed = _with_mm_style(bam, tmp_path / "mixed.bam", pacbio=(0,))
    result = _run_cli("fiberhmm.cli.call", "-i", mixed, *base)
    assert result.returncode == 2
    assert b"--seq pacbio or --seq nanopore" in result.stderr

    # An explicit --seq that the reads' MM specs contradict is refused (it
    # used to run silently), unless --force-seq keeps it.
    result = _run_cli("fiberhmm.cli.call", "-i", bam, *base, "--seq", "pacbio")
    assert result.returncode == 2
    assert b"--force-seq" in result.stderr
    result = _run_cli("fiberhmm.cli.call", "-i", bam, *base, "--seq", "pacbio",
                      "--force-seq")
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert _chemistry(output)[0]["platform"] == "pacbio"


# ---------------------------------------------------------------------------
# Item 10: -k/--context-size was not validated against the table width.
# Item 11: CLI consistency.
# ---------------------------------------------------------------------------

def test_context_size_mismatch_is_rejected(tmp_path, benchmark_model_path):
    from conftest import make_synthetic_bam

    bam = make_synthetic_bam(str(tmp_path / "in.bam"), n_reads=2,
                             read_length=600, n_chroms=1, chrom_length=20_000,
                             seed=19)
    for module, out_flag in (("fiberhmm.cli.call", tmp_path / "o.bam"),
                             ("fiberhmm.cli.apply", tmp_path / "apply_out")):
        result = _run_cli(module, "-i", bam, "-o", out_flag, "-m",
                          benchmark_model_path, "-k", "4", "-c", "1")
        assert result.returncode == 2, module
        assert b"emission columns" in result.stderr, module
    result = _run_cli("fiberhmm.cli.recall_tfs", "-i", bam, "-o",
                      tmp_path / "r.bam", "-m", benchmark_model_path,
                      "--context-size", "4")
    assert result.returncode == 2
    assert b"emission columns" in result.stderr


def test_call_accepts_scores_alias_and_auto_cores(monkeypatch):
    from fiberhmm.cli.call import parse_args

    monkeypatch.setattr(sys, "argv", ["fiberhmm-call", "-i", "x", "-o", "y",
                                      "--scores", "-c", "0"])
    args = parse_args()
    assert args.with_scores is True and args.cores == 0


@pytest.mark.parametrize("flag", [["--chroms", "chr1"], ["--skip-scaffolds"],
                                  ["--region-size", "5000"], ["--scores-db"],
                                  ["-l", "3"]])
def test_apply_rejects_options_it_never_implemented(tmp_path, flag,
                                                    benchmark_model_path):
    result = _run_cli("fiberhmm.cli.apply", "-i", tmp_path / "missing.bam",
                      "-o", tmp_path / "out", "-m", benchmark_model_path, *flag)
    assert result.returncode == 2
    assert b"error:" in result.stderr


def test_apply_streaming_flag_selects_streaming_pipeline(tmp_path,
                                                         benchmark_model_path):
    from conftest import make_synthetic_bam

    bam = make_synthetic_bam(str(tmp_path / "in.bam"), n_reads=3,
                             read_length=1500, n_chroms=1, chrom_length=20_000,
                             seed=20)
    result = _run_cli("fiberhmm.cli.apply", "-i", bam, "-o", tmp_path / "out",
                      "-m", benchmark_model_path, "--min-read-length", "0",
                      "--streaming", "-c", "1", "--io-threads", "1")
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert b"streaming pipeline" in result.stdout


# ---------------------------------------------------------------------------
# Item 9 (remaining writers): merge, pair, extract and the call dedup temp.
# ---------------------------------------------------------------------------

def _all_files(directory):
    return sorted(p.name for p in Path(directory).iterdir())


def test_merge_failure_leaves_no_output_or_unsorted_temp(tmp_path, monkeypatch):
    from conftest import make_synthetic_bam

    from fiberhmm.cli import merge

    bam = make_synthetic_bam(str(tmp_path / "in.bam"), n_reads=3, read_length=500,
                             n_chroms=1, chrom_length=20_000, seed=31)
    outdir = tmp_path / "out"
    outdir.mkdir()
    output = outdir / "merged.bam"

    def failing_sort(*_args):
        raise RuntimeError("sort failed")

    monkeypatch.setattr(merge.pysam, "sort", failing_sort)
    with pytest.raises(RuntimeError, match="sort failed"):
        merge.run_merge(bam, str(output), io_threads=1)
    assert _all_files(outdir) == []

    monkeypatch.undo()
    merge.run_merge(bam, str(output), io_threads=1)
    assert _all_files(outdir) == ["merged.bam", "merged.bam.bai"]


def test_pair_failure_leaves_no_output(tmp_path, monkeypatch):
    from test_pair_unified import _write_sequence_resolved_fixture

    from fiberhmm.cli import duplex
    from fiberhmm.crossstrand.duplex import DuplexParams
    from fiberhmm.crossstrand.pairing import PairParams

    source, reference = _write_sequence_resolved_fixture(tmp_path)
    outdir = tmp_path / "out"
    outdir.mkdir()
    kwargs = dict(
        params=DuplexParams(min_overlap_bp=100, min_nucs=1),
        sequence_params=PairParams(
            min_overlap_bp=100, min_nucs=1, min_sequence_bases=500,
            min_sequence_margin=0.002, max_sequence_pair_rate=0.01,
        ),
        pairing_mode="sequence-only", io_threads=1,
    )
    real = duplex.read_flavor
    calls = []

    def counting(*args, **kw):
        calls.append(1)
        return real(*args, **kw)

    monkeypatch.setattr(duplex, "read_flavor", counting)
    duplex.run_pairing(str(source), str(outdir / "dry.bam"), str(reference),
                       **kwargs)
    total = len(calls)
    for name in os.listdir(outdir):
        os.remove(outdir / name)

    calls.clear()

    def failing_in_pass_two(*args, **kw):
        calls.append(1)
        if len(calls) == total:  # the last read of the writing pass
            raise RuntimeError("pass 2 failed")
        return real(*args, **kw)

    monkeypatch.setattr(duplex, "read_flavor", failing_in_pass_two)
    with pytest.raises(RuntimeError, match="pass 2 failed"):
        duplex.run_pairing(str(source), str(outdir / "paired.bam"),
                           str(reference), **kwargs)
    assert _all_files(outdir) == []


def test_extract_worker_failure_fails_run_without_bed(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from conftest import make_synthetic_bam

    from fiberhmm.cli import extract_tags

    bam = make_synthetic_bam(str(tmp_path / "in.bam"), n_reads=4, read_length=500,
                             n_chroms=2, chrom_length=20_000, seed=32)
    outdir = tmp_path / "out"
    outdir.mkdir()
    real_worker = extract_tags._extract_region_worker

    def failing_worker(item):
        (chrom, _start, _end), _bam, _paths = item
        if chrom == "chr2":
            raise RuntimeError("region worker crashed")
        return real_worker(item)

    monkeypatch.setattr(extract_tags, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(extract_tags, "_extract_region_worker", failing_worker)
    with pytest.raises(RuntimeError, match="region worker"):
        extract_tags.extract_tags_parallel(
            bam, {"msp": str(outdir / "x_msp.bed")}, ["msp"], n_cores=1,
        )
    assert _all_files(outdir) == []


def test_call_removes_dedup_temp_when_calling_fails(tmp_path, monkeypatch):
    from fiberhmm.cli import call

    bam = make_daf_chimera_bam(tmp_path / "raw.bam")
    outdir = tmp_path / "out"
    outdir.mkdir()

    def failing_pipeline(**_kwargs):
        raise RuntimeError("pipeline crashed")

    monkeypatch.setattr(call, "_process_bam_streaming_pipeline_fused",
                        failing_pipeline)
    monkeypatch.setattr(sys, "argv", [
        "fiberhmm-call", "-i", bam, "-o", str(outdir / "calls.bam"),
        "--enzyme", "dddb", "--dedup", "--dedup-min-deam", "1",
        "--no-daf-call-snps", "--no-qc", "--no-recall-nucs",
        "--min-read-length", "0", "-c", "1", "--io-threads", "1",
    ])
    with pytest.raises(RuntimeError, match="pipeline crashed"):
        call.main()
    assert _all_files(outdir) == []


def test_call_region_parallel_matches_streaming_tags(tmp_path, benchmark_model_path):
    """The rewritten fused region worker annotates reads exactly like streaming."""
    bam = make_region_test_bam(tmp_path / "in.bam", seed=33)
    common = ["-m", benchmark_model_path, "--min-read-length", "0",
              "--prob-threshold", "0", "--no-qc", "--phase-nrl", "off",
              "--io-threads", "1"]
    streamed = tmp_path / "streamed.bam"
    region = tmp_path / "region.bam"
    result = _run_cli("fiberhmm.cli.call", "-i", bam, "-o", streamed, "-c", "1",
                      "--no-process-unmapped", *common)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    result = _run_cli("fiberhmm.cli.call", "-i", bam, "-o", region, "-c", "2",
                      "--region-parallel", "--region-size", "5000", *common)
    assert result.returncode == 0, result.stderr.decode(errors="replace")

    def tags(path):
        with pysam.AlignmentFile(str(path), "rb", check_sq=False) as handle:
            return {
                (r.query_name, r.flag, r.reference_start): tuple(
                    str(r.get_tag(t)) if r.has_tag(t) else None
                    for t in ("MA", "AQ", "ns", "nl", "as", "al"))
                for r in handle.fetch(until_eof=True)
            }

    streamed_tags = tags(streamed)
    assert streamed_tags == tags(region)
    assert sum(value[0] is not None for value in streamed_tags.values()) > 0
