"""Failure-path coverage for inline posterior writer lifetimes."""

import pytest

from fiberhmm.inference import parallel


def _install_fake_posterior_writer(monkeypatch):
    instances = []

    class FakePosteriorWriter:
        def __init__(self, *args, **kwargs):
            self.closed = False
            self.close_count = 0
            instances.append(self)

        def close(self):
            self.closed = True
            self.close_count += 1
            return 0, 0.0

    monkeypatch.setattr(parallel, "HAS_POSTERIOR_WRITER", True)
    monkeypatch.setattr(parallel, "PosteriorWriter", FakePosteriorWriter, raising=False)
    return instances


def test_streaming_posterior_writer_closes_when_drain_fails(
    monkeypatch, synthetic_bam_small, benchmark_model_path, tmp_path
):
    instances = _install_fake_posterior_writer(monkeypatch)

    def fail_drain(*args, **kwargs):
        raise RuntimeError("drain failed")

    monkeypatch.setattr(parallel, "_drain_oldest_chunk", fail_drain)
    monkeypatch.setattr(
        parallel,
        "streaming_skip_reason",
        lambda read, filter_config: "low_mapq",
    )

    with pytest.raises(RuntimeError, match="drain failed"):
        parallel._process_bam_streaming_pipeline(
            input_bam=synthetic_bam_small,
            output_bam=str(tmp_path / "streaming_failure.bam"),
            model_path=benchmark_model_path,
            train_rids=set(),
            edge_trim=10,
            circular=False,
            mode="pacbio-fiber",
            context_size=3,
            msp_min_size=60,
            output_posteriors=str(tmp_path / "streaming_failure.h5"),
            n_cores=1,
            chunk_size=1000,
            io_threads=1,
        )

    assert instances
    assert instances[0].closed
    assert instances[0].close_count == 1


def test_legacy_posterior_writer_closes_when_chunk_processing_fails(
    monkeypatch, synthetic_bam_small, benchmark_model_path, tmp_path
):
    instances = _install_fake_posterior_writer(monkeypatch)

    def fail_process_chunk(*args, **kwargs):
        raise RuntimeError("chunk failed")

    monkeypatch.setattr(parallel, "_process_and_write_chunk", fail_process_chunk)

    with pytest.raises(RuntimeError, match="chunk failed"):
        parallel.process_bam_for_footprints(
            input_bam=synthetic_bam_small,
            output_bam=str(tmp_path / "legacy_failure.bam"),
            model_or_path=benchmark_model_path,
            train_rids=set(),
            edge_trim=10,
            circular=False,
            mode="pacbio-fiber",
            context_size=3,
            msp_min_size=60,
            output_posteriors=str(tmp_path / "legacy_failure.h5"),
            n_cores=1,
            max_reads=1,
        )

    assert instances
    assert instances[0].closed
    assert instances[0].close_count == 1
