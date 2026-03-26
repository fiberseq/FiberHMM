"""
Correctness tests for the streaming producer-consumer pipeline.
"""
import os
import pytest
import pysam
import numpy as np
import tempfile

from fiberhmm.inference.parallel import process_bam_for_footprints


def _count_bam_reads(bam_path):
    """Count total reads in a BAM file."""
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        return sum(1 for _ in bam.fetch(until_eof=True))


def _read_tags_by_name(bam_path):
    """Read BAM and return dict of {read_name: {tag: value, ...}}."""
    result = {}
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(until_eof=True):
            tags = {}
            for tag in ['ns', 'nl', 'as', 'al', 'nq', 'aq']:
                if read.has_tag(tag):
                    tags[tag] = list(read.get_tag(tag))
            result[read.query_name] = tags
    return result


def _read_names_in_order(bam_path):
    """Return list of read names in BAM order."""
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        return [read.query_name for read in bam.fetch(until_eof=True)]


class TestStreamingBasic:
    """Basic correctness tests for the streaming pipeline."""

    def test_all_reads_preserved(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """Every read in input appears in output."""
        output = str(tmp_path / "out.bam")
        process_bam_for_footprints(
            input_bam=synthetic_bam_small, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=20,
            min_mapq=0, min_read_length=0, prob_threshold=0,
        )
        assert _count_bam_reads(output) == _count_bam_reads(synthetic_bam_small)

    def test_footprint_tags_present(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """Processed reads get ns/nl tags."""
        output = str(tmp_path / "out.bam")
        total, with_fp = process_bam_for_footprints(
            input_bam=synthetic_bam_small, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=20,
            min_mapq=0, min_read_length=0, prob_threshold=0,
        )
        tags = _read_tags_by_name(output)
        tagged = sum(1 for t in tags.values() if 'ns' in t)
        assert tagged == with_fp
        assert with_fp > 0

    def test_msp_tags_present(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """as/al tags present by default."""
        output = str(tmp_path / "out.bam")
        process_bam_for_footprints(
            input_bam=synthetic_bam_small, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=20,
            min_mapq=0, min_read_length=0, prob_threshold=0,
        )
        tags = _read_tags_by_name(output)
        has_msp = sum(1 for t in tags.values() if 'as' in t)
        assert has_msp > 0

    def test_no_msp_tags_when_disabled(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """as/al tags absent when write_msps=False."""
        output = str(tmp_path / "out.bam")
        process_bam_for_footprints(
            input_bam=synthetic_bam_small, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=20,
            min_mapq=0, min_read_length=0, prob_threshold=0,
            write_msps=False,
        )
        tags = _read_tags_by_name(output)
        has_msp = sum(1 for t in tags.values() if 'as' in t)
        assert has_msp == 0

    def test_tag_values_reasonable(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """ns values within read bounds, nl > 0."""
        output = str(tmp_path / "out.bam")
        process_bam_for_footprints(
            input_bam=synthetic_bam_small, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=20,
            min_mapq=0, min_read_length=0, prob_threshold=0,
        )
        with pysam.AlignmentFile(output, "rb") as bam:
            for read in bam:
                if read.has_tag('ns'):
                    starts = list(read.get_tag('ns'))
                    lengths = list(read.get_tag('nl'))
                    assert len(starts) == len(lengths)
                    for s, l in zip(starts, lengths):
                        assert s >= 0
                        assert l > 0
                        assert s + l <= read.query_length

    def test_scores_when_requested(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """nq tags present when with_scores=True."""
        output = str(tmp_path / "out.bam")
        process_bam_for_footprints(
            input_bam=synthetic_bam_small, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=20,
            min_mapq=0, min_read_length=0, prob_threshold=0,
            with_scores=True,
        )
        tags = _read_tags_by_name(output)
        has_scores = sum(1 for t in tags.values() if 'nq' in t)
        assert has_scores > 0


class TestStreamingOrderAndDeterminism:
    """Order preservation and deterministic output."""

    def test_order_preserved(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """Output read order matches input read order."""
        output = str(tmp_path / "out.bam")
        process_bam_for_footprints(
            input_bam=synthetic_bam_small, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=20,
            min_mapq=0, min_read_length=0, prob_threshold=0,
        )
        input_names = _read_names_in_order(synthetic_bam_small)
        output_names = _read_names_in_order(output)
        assert input_names == output_names

    def test_deterministic(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """Two runs produce identical tag values."""
        out1 = str(tmp_path / "out1.bam")
        out2 = str(tmp_path / "out2.bam")

        for out in [out1, out2]:
            process_bam_for_footprints(
                input_bam=synthetic_bam_small, output_bam=out,
                model_or_path=benchmark_model_path,
                train_rids=set(), edge_trim=10, circular=False,
                mode='pacbio-fiber', context_size=3, msp_min_size=60,
                n_cores=2, streaming_pipeline=True, chunk_size=20,
                min_mapq=0, min_read_length=0, prob_threshold=0,
            )

        tags1 = _read_tags_by_name(out1)
        tags2 = _read_tags_by_name(out2)
        assert tags1 == tags2

    def test_single_core_matches_multi_core(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """n_cores=1 and n_cores=2 produce same tags."""
        out1 = str(tmp_path / "out_1core.bam")
        out2 = str(tmp_path / "out_2core.bam")

        for out, cores in [(out1, 1), (out2, 2)]:
            process_bam_for_footprints(
                input_bam=synthetic_bam_small, output_bam=out,
                model_or_path=benchmark_model_path,
                train_rids=set(), edge_trim=10, circular=False,
                mode='pacbio-fiber', context_size=3, msp_min_size=60,
                n_cores=cores, streaming_pipeline=True, chunk_size=20,
                min_mapq=0, min_read_length=0, prob_threshold=0,
            )

        tags1 = _read_tags_by_name(out1)
        tags2 = _read_tags_by_name(out2)
        assert tags1 == tags2


class TestStreamingEdgeCases:
    """Edge cases and special inputs."""

    def test_empty_bam(self, empty_bam, benchmark_model_path, tmp_path):
        """Empty BAM produces empty output, returns (0, 0)."""
        output = str(tmp_path / "out.bam")
        total, fp = process_bam_for_footprints(
            input_bam=empty_bam, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=20,
            min_mapq=0, min_read_length=0, prob_threshold=0,
        )
        assert total == 0
        assert fp == 0
        assert _count_bam_reads(output) == 0

    def test_works_without_index(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """Streaming pipeline works with unindexed BAM."""
        # Copy BAM without its index
        import shutil
        unindexed = str(tmp_path / "noindex.bam")
        shutil.copy2(synthetic_bam_small, unindexed)
        # Don't copy .bai

        output = str(tmp_path / "out.bam")
        total, fp = process_bam_for_footprints(
            input_bam=unindexed, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=20,
            min_mapq=0, min_read_length=0, prob_threshold=0,
        )
        assert total > 0
        assert _count_bam_reads(output) == _count_bam_reads(synthetic_bam_small)

    def test_unmapped_passthrough(self, unaligned_bam, benchmark_model_path, tmp_path):
        """Unmapped reads written unchanged when process_unmapped=False."""
        output = str(tmp_path / "out.bam")
        total, fp = process_bam_for_footprints(
            input_bam=unaligned_bam, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=20,
            min_mapq=0, min_read_length=0, prob_threshold=0,
            process_unmapped=False,
        )
        assert total == 0  # All skipped as unmapped
        assert fp == 0
        # All reads should be written (passthrough)
        assert _count_bam_reads(output) == _count_bam_reads(unaligned_bam)

    def test_unmapped_processing(self, unaligned_bam, benchmark_model_path, tmp_path):
        """Unmapped reads with sequences are processed when process_unmapped=True."""
        output = str(tmp_path / "out.bam")
        total, fp = process_bam_for_footprints(
            input_bam=unaligned_bam, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=20,
            min_mapq=0, min_read_length=0, prob_threshold=0,
            process_unmapped=True,
        )
        assert total > 0  # Some reads processed
        tags = _read_tags_by_name(output)
        tagged = sum(1 for t in tags.values() if 'ns' in t)
        assert tagged > 0

    def test_max_reads_limit(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """max_reads parameter limits processing."""
        output = str(tmp_path / "out.bam")
        total, fp = process_bam_for_footprints(
            input_bam=synthetic_bam_small, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=5,
            min_mapq=0, min_read_length=0, prob_threshold=0,
            max_reads=10,
        )
        assert total == 10

    def test_chunk_size_1(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """chunk_size=1 still works correctly (degenerate case)."""
        output = str(tmp_path / "out.bam")
        total, fp = process_bam_for_footprints(
            input_bam=synthetic_bam_small, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=1,
            min_mapq=0, min_read_length=0, prob_threshold=0,
        )
        assert total == _count_bam_reads(synthetic_bam_small)
        assert _count_bam_reads(output) == _count_bam_reads(synthetic_bam_small)

    def test_large_chunk_size(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """chunk_size larger than total reads still works."""
        output = str(tmp_path / "out.bam")
        total, fp = process_bam_for_footprints(
            input_bam=synthetic_bam_small, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=10000,
            min_mapq=0, min_read_length=0, prob_threshold=0,
        )
        assert total == _count_bam_reads(synthetic_bam_small)
