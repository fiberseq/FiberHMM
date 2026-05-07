"""Characterization tests for the fused `fiberhmm-call` pipelines."""

import os

import pysam

from conftest import make_synthetic_bam
from fiberhmm.inference.parallel import (
    _process_bam_region_parallel_fused,
    _process_bam_streaming_pipeline_fused,
)


def _run_fused_streaming(input_bam, output_bam, model_path, *, with_scores=False):
    return _process_bam_streaming_pipeline_fused(
        input_bam=input_bam,
        output_bam=output_bam,
        model_path=model_path,
        recall_model_path=None,
        train_rids=set(),
        edge_trim=10,
        circular=False,
        mode="pacbio-fiber",
        context_size=3,
        msp_min_size=0,
        nuc_min_size=85,
        min_mapq=0,
        prob_threshold=0,
        min_read_length=0,
        with_scores=with_scores,
        min_llr=1000.0,
        min_opps=3,
        unify_threshold=90,
        emission_uplift=1.0,
        also_write_legacy=True,
        downstream_compat=False,
        max_reads=0,
        n_cores=1,
        chunk_size=10,
        io_threads=1,
    )


def _run_fused_region(input_bam, output_bam, model_path, *, with_scores=False):
    return _process_bam_region_parallel_fused(
        input_bam=input_bam,
        output_bam=output_bam,
        apply_model_path=model_path,
        recall_model_path=None,
        train_rids=set(),
        edge_trim=10,
        circular=False,
        mode="pacbio-fiber",
        context_size=3,
        msp_min_size=0,
        nuc_min_size=85,
        min_mapq=0,
        prob_threshold=0,
        min_read_length=0,
        with_scores=with_scores,
        min_llr=1000.0,
        min_opps=3,
        unify_threshold=90,
        emission_uplift=1.0,
        also_write_legacy=True,
        downstream_compat=False,
        n_cores=1,
        region_size=10_000_000,
        skip_scaffolds=False,
        chroms=None,
        io_threads=1,
    )


def _read_tags_by_name(bam_path):
    tags_by_name = {}
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            tags = {}
            for tag in ("ns", "nl", "as", "al", "nq", "aq", "MA", "AQ"):
                if not read.has_tag(tag):
                    continue
                value = read.get_tag(tag)
                tags[tag] = value if isinstance(value, str) else list(value)
            tags_by_name[read.query_name] = tags
    return tags_by_name


def test_fused_streaming_emits_spec_and_legacy_tags(
    synthetic_bam_small, benchmark_model_path, tmp_path
):
    output = str(tmp_path / "fused_stream.bam")

    total, with_fp = _run_fused_streaming(synthetic_bam_small, output, benchmark_model_path)

    assert total > 0
    assert with_fp > 0
    tags = _read_tags_by_name(output)
    assert any("MA" in t for t in tags.values())
    assert any("AQ" in t for t in tags.values())
    assert any("ns" in t and "nl" in t for t in tags.values())
    assert any("as" in t and "al" in t for t in tags.values())


def test_fused_with_scores_writes_nq_for_kept_nucs(
    synthetic_bam_small, benchmark_model_path, tmp_path
):
    output = str(tmp_path / "fused_scores.bam")

    _run_fused_streaming(
        synthetic_bam_small, output, benchmark_model_path, with_scores=True
    )

    tags = _read_tags_by_name(output)
    scored = [t for t in tags.values() if "ns" in t]
    assert scored
    assert any("nq" in t for t in scored)
    for tag_set in scored:
        if "nq" in tag_set:
            assert len(tag_set["nq"]) == len(tag_set["ns"])
            assert all(0 <= q <= 255 for q in tag_set["nq"])


def test_fused_region_matches_streaming_on_single_region(benchmark_model_path, tmp_path):
    input_bam = str(tmp_path / "one_chrom.bam")
    make_synthetic_bam(
        input_bam,
        n_reads=30,
        read_length=1500,
        n_chroms=1,
        chrom_length=100_000,
        seed=123,
    )
    stream_out = str(tmp_path / "stream.bam")
    region_out = str(tmp_path / "region.bam")

    stream_counts = _run_fused_streaming(input_bam, stream_out, benchmark_model_path)
    region_counts = _run_fused_region(input_bam, region_out, benchmark_model_path)

    assert stream_counts == region_counts
    assert _read_tags_by_name(stream_out) == _read_tags_by_name(region_out)
    assert not any(
        p.name.startswith(".fiberhmm_call_tmp_")
        for p in tmp_path.iterdir()
        if os.path.isdir(p)
    )
