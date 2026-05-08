"""Failure-path coverage for apply-pipeline resource lifetimes."""

import pytest

from fiberhmm.inference import parallel
from fiberhmm.posteriors import hdf5_backend


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


def test_fused_streaming_reference_fasta_closes_when_drain_fails(
    monkeypatch, synthetic_bam_small, benchmark_model_path, tmp_path
):
    closed_paths = []

    class FakeFastaFile:
        def __init__(self, path):
            self.path = path

        def close(self):
            closed_paths.append(self.path)

    def fail_drain(*args, **kwargs):
        raise RuntimeError("fused drain failed")

    monkeypatch.setattr(parallel.pysam, "FastaFile", FakeFastaFile)
    monkeypatch.setattr(parallel, "_drain_oldest_fused_chunk", fail_drain)
    monkeypatch.setattr(
        parallel,
        "streaming_skip_reason",
        lambda read, filter_config: "low_mapq",
    )

    ref_fasta = str(tmp_path / "fake_reference.fa")
    with pytest.raises(RuntimeError, match="fused drain failed"):
        parallel._process_bam_streaming_pipeline_fused(
            input_bam=synthetic_bam_small,
            output_bam=str(tmp_path / "fused_failure.bam"),
            model_path=benchmark_model_path,
            recall_model_path=None,
            train_rids=set(),
            edge_trim=10,
            circular=False,
            mode="daf",
            context_size=3,
            msp_min_size=60,
            nuc_min_size=85,
            min_mapq=0,
            prob_threshold=0,
            min_read_length=0,
            with_scores=False,
            min_llr=1000.0,
            min_opps=3,
            unify_threshold=90,
            emission_uplift=1.0,
            also_write_legacy=True,
            downstream_compat=False,
            max_reads=0,
            n_cores=1,
            chunk_size=1000,
            io_threads=1,
            ref_fasta_path=ref_fasta,
        )

    assert closed_paths == [ref_fasta]


def test_hdf5_posterior_writer_closes_file_when_finalize_fails(monkeypatch, tmp_path):
    class FakeH5:
        def __init__(self):
            self.attrs = {}
            self.closed = False
            self.close_count = 0

        def close(self):
            self.closed = True
            self.close_count += 1

    fake_h5 = FakeH5()
    monkeypatch.setattr(hdf5_backend.h5py, "File", lambda *args, **kwargs: fake_h5)

    writer = hdf5_backend.PosteriorWriter(
        str(tmp_path / "posteriors.h5"),
        mode="pacbio-fiber",
        context_size=3,
        edge_trim=10,
        source_bam="input.bam",
    )

    def fail_finalize():
        raise RuntimeError("finalize failed")

    monkeypatch.setattr(writer, "finalize", fail_finalize)

    with pytest.raises(RuntimeError, match="finalize failed"):
        writer.close()

    assert fake_h5.closed
    assert fake_h5.close_count == 1
    assert writer._closed

    assert writer.close() == (0, 0)
    assert fake_h5.close_count == 1
