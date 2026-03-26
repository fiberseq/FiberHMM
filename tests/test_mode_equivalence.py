"""
Cross-mode equivalence tests: verify all processing modes produce identical results.
"""
import pytest
import pysam

from fiberhmm.inference.parallel import process_bam_for_footprints


def _read_tags_by_name(bam_path):
    """Read BAM and return dict of {read_name: {tag: value, ...}}."""
    result = {}
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(until_eof=True):
            tags = {}
            for tag in ['ns', 'nl', 'as', 'al']:
                if read.has_tag(tag):
                    tags[tag] = list(read.get_tag(tag))
            result[read.query_name] = tags
    return result


def _process_with_mode(bam_path, model_path, output_path,
                       region_parallel=False, streaming_pipeline=False,
                       n_cores=1, chunk_size=20, with_scores=False):
    """Run process_bam_for_footprints with given mode settings."""
    return process_bam_for_footprints(
        input_bam=bam_path, output_bam=output_path,
        model_or_path=model_path,
        train_rids=set(), edge_trim=10, circular=False,
        mode='pacbio-fiber', context_size=3, msp_min_size=60,
        n_cores=n_cores,
        region_parallel=region_parallel,
        streaming_pipeline=streaming_pipeline,
        chunk_size=chunk_size,
        min_mapq=0, min_read_length=0, prob_threshold=0,
        with_scores=with_scores,
    )


class TestModeEquivalence:
    """All processing modes must produce identical HMM results."""

    def test_streaming_vs_region_parallel(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """Streaming pipeline and region-parallel produce identical tags per read."""
        out_stream = str(tmp_path / "streaming.bam")
        out_region = str(tmp_path / "region.bam")

        _process_with_mode(
            synthetic_bam_small, benchmark_model_path, out_stream,
            streaming_pipeline=True, n_cores=2, chunk_size=20,
        )
        _process_with_mode(
            synthetic_bam_small, benchmark_model_path, out_region,
            region_parallel=True, n_cores=2,
        )

        tags_stream = _read_tags_by_name(out_stream)
        tags_region = _read_tags_by_name(out_region)

        # Same set of reads tagged
        assert set(tags_stream.keys()) == set(tags_region.keys())

        # Same tag values per read
        mismatches = []
        for name in tags_stream:
            if tags_stream[name] != tags_region[name]:
                mismatches.append(name)

        assert len(mismatches) == 0, (
            f"{len(mismatches)} reads have different tags between streaming and "
            f"region-parallel modes. First: {mismatches[0]}"
        )

    def test_streaming_vs_legacy_chunk(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """Streaming pipeline matches legacy single-pass chunk mode."""
        out_stream = str(tmp_path / "streaming.bam")
        out_legacy = str(tmp_path / "legacy.bam")

        _process_with_mode(
            synthetic_bam_small, benchmark_model_path, out_stream,
            streaming_pipeline=True, n_cores=2, chunk_size=20,
        )
        # Legacy mode: region_parallel=False, streaming_pipeline=False, single core
        _process_with_mode(
            synthetic_bam_small, benchmark_model_path, out_legacy,
            region_parallel=False, streaming_pipeline=False, n_cores=1,
        )

        tags_stream = _read_tags_by_name(out_stream)
        tags_legacy = _read_tags_by_name(out_legacy)

        assert set(tags_stream.keys()) == set(tags_legacy.keys())

        mismatches = []
        for name in tags_stream:
            if tags_stream[name] != tags_legacy[name]:
                mismatches.append(name)

        assert len(mismatches) == 0, (
            f"{len(mismatches)} reads have different tags between streaming and "
            f"legacy chunk modes. First: {mismatches[0]}"
        )

    def test_with_scores_same_footprints(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """Enabling scores doesn't change footprint calls (ns/nl/as/al)."""
        out_no_scores = str(tmp_path / "no_scores.bam")
        out_scores = str(tmp_path / "scores.bam")

        _process_with_mode(
            synthetic_bam_small, benchmark_model_path, out_no_scores,
            streaming_pipeline=True, n_cores=2, chunk_size=20,
            with_scores=False,
        )
        _process_with_mode(
            synthetic_bam_small, benchmark_model_path, out_scores,
            streaming_pipeline=True, n_cores=2, chunk_size=20,
            with_scores=True,
        )

        tags_no = _read_tags_by_name(out_no_scores)
        tags_yes = _read_tags_by_name(out_scores)

        assert set(tags_no.keys()) == set(tags_yes.keys())

        # ns/nl/as/al should be identical
        mismatches = []
        for name in tags_no:
            for tag in ['ns', 'nl', 'as', 'al']:
                v1 = tags_no[name].get(tag)
                v2 = tags_yes[name].get(tag)
                if v1 != v2:
                    mismatches.append((name, tag))

        assert len(mismatches) == 0, (
            f"{len(mismatches)} tag differences when scores enabled. "
            f"First: read={mismatches[0][0]}, tag={mismatches[0][1]}"
        )

    def test_all_three_modes_agree(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """All three modes (streaming, region-parallel, legacy) produce identical results."""
        outputs = {}
        for name, kwargs in [
            ("streaming", dict(streaming_pipeline=True, n_cores=2, chunk_size=20)),
            ("region", dict(region_parallel=True, n_cores=2)),
            ("legacy", dict(region_parallel=False, streaming_pipeline=False, n_cores=1)),
        ]:
            out = str(tmp_path / f"{name}.bam")
            _process_with_mode(synthetic_bam_small, benchmark_model_path, out, **kwargs)
            outputs[name] = _read_tags_by_name(out)

        # All three should have identical tags
        for name_a, name_b in [("streaming", "region"), ("streaming", "legacy")]:
            tags_a = outputs[name_a]
            tags_b = outputs[name_b]
            assert set(tags_a.keys()) == set(tags_b.keys()), (
                f"Different read sets between {name_a} and {name_b}"
            )
            for read_name in tags_a:
                assert tags_a[read_name] == tags_b[read_name], (
                    f"Tags differ for {read_name} between {name_a} and {name_b}: "
                    f"{tags_a[read_name]} vs {tags_b[read_name]}"
                )
