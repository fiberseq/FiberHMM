"""FiberHMM BAM processing compatibility orchestration."""

from dataclasses import dataclass
from typing import Optional, Set, Tuple

import pysam

from fiberhmm.core.model_io import freeze_model_for_inference, load_model
from fiberhmm.inference.legacy_pipeline import (
    _process_and_write_chunk,
    _process_bam_legacy_pipeline,
)
from fiberhmm.inference.mp_context import (
    _MP_CONTEXT,
    _select_mp_context,
)
from fiberhmm.inference.read_filters import (
    ReadFilterConfig,
    streaming_skip_reason,
)
from fiberhmm.inference.region_pipeline import (
    _process_bam_region_parallel,
    _process_bam_region_parallel_fused,
    _process_bed_region_parallel,
)
from fiberhmm.inference.region_planning import (
    _get_genome_regions,
)
from fiberhmm.inference.region_planning import (
    _is_main_chromosome as _region_is_main_chromosome,
)
from fiberhmm.inference.region_workers import (
    _init_fused_region_worker,
    _init_region_worker,
    _process_region_to_bam,
    _process_region_to_bam_fused,
    _process_region_to_bed,
)
from fiberhmm.inference.streaming_drain import (
    _drain_oldest_chunk,
    _drain_oldest_fused_chunk,
)
from fiberhmm.inference.streaming_pipeline import (
    _process_bam_streaming_pipeline,
    _process_bam_streaming_pipeline_fused,
)
from fiberhmm.inference.streaming_workers import (
    _init_bam_worker,
    _init_fused_worker,
    _process_chunk_worker,
    _process_fused_payload_chunk_worker,
    _process_payload_chunk_worker,
)

_is_main_chromosome = _region_is_main_chromosome


@dataclass(frozen=True)
class _ProcessingModelSource:
    model: object
    path: Optional[str]


@dataclass(frozen=True)
class _FootprintPipelineOptions:
    input_bam: str
    output_bam: str
    train_rids: Set[str]
    edge_trim: int
    circular: bool
    mode: str
    context_size: int
    msp_min_size: int
    nuc_min_size: int
    min_mapq: int
    prob_threshold: int
    min_read_length: int
    with_scores: bool
    n_cores: int
    primary_only: bool
    output_posteriors: Optional[str]
    write_msps: bool
    io_threads: int

    def as_kwargs(self) -> dict:
        return {
            'input_bam': self.input_bam,
            'output_bam': self.output_bam,
            'train_rids': self.train_rids,
            'edge_trim': self.edge_trim,
            'circular': self.circular,
            'mode': self.mode,
            'context_size': self.context_size,
            'msp_min_size': self.msp_min_size,
            'nuc_min_size': self.nuc_min_size,
            'min_mapq': self.min_mapq,
            'prob_threshold': self.prob_threshold,
            'min_read_length': self.min_read_length,
            'with_scores': self.with_scores,
            'n_cores': self.n_cores,
            'primary_only': self.primary_only,
            'output_posteriors': self.output_posteriors,
            'write_msps': self.write_msps,
            'io_threads': self.io_threads,
        }


__all__ = (
    "ReadFilterConfig",
    "_MP_CONTEXT",
    "_drain_oldest_chunk",
    "_drain_oldest_fused_chunk",
    "_get_genome_regions",
    "_init_bam_worker",
    "_init_fused_region_worker",
    "_init_fused_worker",
    "_init_region_worker",
    "_is_main_chromosome",
    "_process_and_write_chunk",
    "_process_bam_region_parallel",
    "_process_bam_region_parallel_fused",
    "_process_bam_streaming_pipeline",
    "_process_bam_streaming_pipeline_fused",
    "_process_bed_region_parallel",
    "_process_chunk_worker",
    "_process_fused_payload_chunk_worker",
    "_process_payload_chunk_worker",
    "_process_region_to_bam",
    "_process_region_to_bam_fused",
    "_process_region_to_bed",
    "_select_mp_context",
    "process_bam_for_footprints",
    "streaming_skip_reason",
)


def _model_and_path_for_processing(model_or_path):
    if isinstance(model_or_path, str):
        return _ProcessingModelSource(model=None, path=model_or_path)
    return _ProcessingModelSource(model=model_or_path, path=None)


def _legacy_model_for_processing(model, model_path: Optional[str], n_cores: int):
    if model is None and model_path is not None and n_cores <= 1:
        return freeze_model_for_inference(load_model(model_path))
    return model


def _dispatch_region_parallel_if_requested(
    *,
    region_parallel: bool,
    model_path: Optional[str],
    input_bam: str,
    output_bam: str,
    train_rids: Set[str],
    edge_trim: int,
    circular: bool,
    mode: str,
    context_size: int,
    msp_min_size: int,
    nuc_min_size: int,
    min_mapq: int,
    prob_threshold: int,
    min_read_length: int,
    with_scores: bool,
    n_cores: int,
    region_size: int,
    skip_scaffolds: bool,
    chroms: Optional[Set[str]],
    primary_only: bool,
    output_posteriors: Optional[str],
    write_msps: bool,
    io_threads: int,
) -> Optional[Tuple[int, int]]:
    if not region_parallel:
        return None
    if model_path is None:
        print("Warning: region_parallel requires model path, falling back to standard parallel")
        return None

    return _process_bam_region_parallel(
        input_bam=input_bam,
        output_bam=output_bam,
        model_path=model_path,
        train_rids=train_rids,
        edge_trim=edge_trim,
        circular=circular,
        mode=mode,
        context_size=context_size,
        msp_min_size=msp_min_size,
        nuc_min_size=nuc_min_size,
        min_mapq=min_mapq,
        prob_threshold=prob_threshold,
        min_read_length=min_read_length,
        with_scores=with_scores,
        n_cores=n_cores,
        region_size=region_size,
        skip_scaffolds=skip_scaffolds,
        chroms=chroms,
        primary_only=primary_only,
        output_posteriors=output_posteriors,
        write_msps=write_msps,
        io_threads=io_threads,
    )


def _dispatch_streaming_pipeline_if_requested(
    *,
    streaming_pipeline: bool,
    model_path: Optional[str],
    input_bam: str,
    output_bam: str,
    train_rids: Set[str],
    edge_trim: int,
    circular: bool,
    mode: str,
    context_size: int,
    msp_min_size: int,
    nuc_min_size: int,
    min_mapq: int,
    prob_threshold: int,
    min_read_length: int,
    with_scores: bool,
    n_cores: int,
    chunk_size: int,
    io_threads: int,
    primary_only: bool,
    output_posteriors: Optional[str],
    write_msps: bool,
    max_reads: Optional[int],
    debug_timing: bool,
    process_unmapped: bool,
) -> Optional[Tuple[int, int]]:
    if not streaming_pipeline:
        return None
    if model_path is None:
        print("Warning: streaming_pipeline requires model path, falling back to standard parallel")
        return None

    return _process_bam_streaming_pipeline(
        input_bam=input_bam,
        output_bam=output_bam,
        model_path=model_path,
        train_rids=train_rids,
        edge_trim=edge_trim,
        circular=circular,
        mode=mode,
        context_size=context_size,
        msp_min_size=msp_min_size,
        nuc_min_size=nuc_min_size,
        min_mapq=min_mapq,
        prob_threshold=prob_threshold,
        min_read_length=min_read_length,
        with_scores=with_scores,
        n_cores=n_cores,
        chunk_size=chunk_size,
        io_threads=io_threads,
        primary_only=primary_only,
        output_posteriors=output_posteriors,
        write_msps=write_msps,
        max_reads=max_reads,
        debug_timing=debug_timing,
        process_unmapped=process_unmapped,
    )


def _footprint_pipeline_options(
    *,
    input_bam: str,
    output_bam: str,
    train_rids: Set[str],
    edge_trim: int,
    circular: bool,
    mode: str,
    context_size: int,
    msp_min_size: int,
    nuc_min_size: int,
    min_mapq: int,
    prob_threshold: int,
    min_read_length: int,
    with_scores: bool,
    n_cores: int,
    primary_only: bool,
    output_posteriors: Optional[str],
    write_msps: bool,
    io_threads: int,
) -> _FootprintPipelineOptions:
    return _FootprintPipelineOptions(
        input_bam=input_bam,
        output_bam=output_bam,
        train_rids=train_rids,
        edge_trim=edge_trim,
        circular=circular,
        mode=mode,
        context_size=context_size,
        msp_min_size=msp_min_size,
        nuc_min_size=nuc_min_size,
        min_mapq=min_mapq,
        prob_threshold=prob_threshold,
        min_read_length=min_read_length,
        with_scores=with_scores,
        n_cores=n_cores,
        primary_only=primary_only,
        output_posteriors=output_posteriors,
        write_msps=write_msps,
        io_threads=io_threads,
    )


def _footprint_pipeline_kwargs(
    *,
    input_bam: str,
    output_bam: str,
    train_rids: Set[str],
    edge_trim: int,
    circular: bool,
    mode: str,
    context_size: int,
    msp_min_size: int,
    nuc_min_size: int,
    min_mapq: int,
    prob_threshold: int,
    min_read_length: int,
    with_scores: bool,
    n_cores: int,
    primary_only: bool,
    output_posteriors: Optional[str],
    write_msps: bool,
    io_threads: int,
) -> dict:
    return _footprint_pipeline_options(
        input_bam=input_bam,
        output_bam=output_bam,
        train_rids=train_rids,
        edge_trim=edge_trim,
        circular=circular,
        mode=mode,
        context_size=context_size,
        msp_min_size=msp_min_size,
        nuc_min_size=nuc_min_size,
        min_mapq=min_mapq,
        prob_threshold=prob_threshold,
        min_read_length=min_read_length,
        with_scores=with_scores,
        n_cores=n_cores,
        primary_only=primary_only,
        output_posteriors=output_posteriors,
        write_msps=write_msps,
        io_threads=io_threads,
    ).as_kwargs()


def _dispatch_requested_parallel_pipeline(
    *,
    model_path: Optional[str],
    region_parallel: bool,
    region_size: int,
    skip_scaffolds: bool,
    chroms: Optional[Set[str]],
    streaming_pipeline: bool,
    chunk_size: int,
    max_reads: Optional[int],
    debug_timing: bool,
    process_unmapped: bool,
    pipeline_options: _FootprintPipelineOptions,
) -> Optional[Tuple[int, int]]:
    result = _dispatch_region_parallel_if_requested(
        region_parallel=region_parallel,
        model_path=model_path,
        region_size=region_size,
        skip_scaffolds=skip_scaffolds,
        chroms=chroms,
        input_bam=pipeline_options.input_bam,
        output_bam=pipeline_options.output_bam,
        train_rids=pipeline_options.train_rids,
        edge_trim=pipeline_options.edge_trim,
        circular=pipeline_options.circular,
        mode=pipeline_options.mode,
        context_size=pipeline_options.context_size,
        msp_min_size=pipeline_options.msp_min_size,
        nuc_min_size=pipeline_options.nuc_min_size,
        min_mapq=pipeline_options.min_mapq,
        prob_threshold=pipeline_options.prob_threshold,
        min_read_length=pipeline_options.min_read_length,
        with_scores=pipeline_options.with_scores,
        n_cores=pipeline_options.n_cores,
        primary_only=pipeline_options.primary_only,
        output_posteriors=pipeline_options.output_posteriors,
        write_msps=pipeline_options.write_msps,
        io_threads=pipeline_options.io_threads,
    )
    if result is not None:
        return result

    return _dispatch_streaming_pipeline_if_requested(
        streaming_pipeline=streaming_pipeline,
        model_path=model_path,
        chunk_size=chunk_size,
        max_reads=max_reads,
        debug_timing=debug_timing,
        process_unmapped=process_unmapped,
        input_bam=pipeline_options.input_bam,
        output_bam=pipeline_options.output_bam,
        train_rids=pipeline_options.train_rids,
        edge_trim=pipeline_options.edge_trim,
        circular=pipeline_options.circular,
        mode=pipeline_options.mode,
        context_size=pipeline_options.context_size,
        msp_min_size=pipeline_options.msp_min_size,
        nuc_min_size=pipeline_options.nuc_min_size,
        min_mapq=pipeline_options.min_mapq,
        prob_threshold=pipeline_options.prob_threshold,
        min_read_length=pipeline_options.min_read_length,
        with_scores=pipeline_options.with_scores,
        n_cores=pipeline_options.n_cores,
        io_threads=pipeline_options.io_threads,
        primary_only=pipeline_options.primary_only,
        output_posteriors=pipeline_options.output_posteriors,
        write_msps=pipeline_options.write_msps,
    )


def _dispatch_legacy_footprint_pipeline(
    *,
    model,
    model_path: Optional[str],
    n_cores: int,
    max_reads: Optional[int],
    debug_timing: bool,
    pipeline_kwargs: dict,
) -> Tuple[int, int]:
    model = _legacy_model_for_processing(model, model_path, n_cores)
    return _process_bam_legacy_pipeline(
        model=model,
        model_path=model_path,
        max_reads=max_reads,
        debug_timing=debug_timing,
        **pipeline_kwargs,
    )


def process_bam_for_footprints(input_bam: str, output_bam: str,
                                model_or_path, train_rids: Set[str],
                                edge_trim: int, circular: bool,
                                mode: str, context_size: int,
                                msp_min_size: int,
                                nuc_min_size: int = 85,
                                min_mapq: int = 0,
                                prob_threshold: int = 0,
                                min_read_length: int = 0,
                                with_scores: bool = False,
                                n_cores: int = 1,
                                max_reads: Optional[int] = None,
                                debug_timing: bool = False,
                                region_parallel: bool = False,
                                region_size: int = 10_000_000,
                                skip_scaffolds: bool = False,
                                chroms: Optional[Set[str]] = None,
                                primary_only: bool = False,
                                output_posteriors: Optional[str] = None,
                                write_msps: bool = True,
                                io_threads: int = 4,
                                streaming_pipeline: bool = False,
                                chunk_size: int = 500,
                                process_unmapped: bool = False) -> Tuple[int, int]:
    """
    Process BAM file and add footprint tags - SINGLE PASS STREAMING.

    Reads chunks from BAM, processes in parallel, writes immediately.
    Memory usage is bounded to one chunk at a time.

    Args:
        region_parallel: Use region-based parallelism (each worker reads BAM)
        primary_only: Only process primary alignments (skip secondary/supplementary)
        skip_scaffolds: If True, skip scaffold/contig chromosomes
        chroms: If provided, only process these chromosomes
        output_posteriors: If provided, write HMM posteriors to this H5 file (inline)

    Returns:
        (total_reads_processed, reads_with_footprints)
    """
    # Suppress htslib warnings
    pysam.set_verbosity(0)

    # Get model path for workers
    model_source = _model_and_path_for_processing(model_or_path)
    model = model_source.model
    model_path = model_source.path
    pipeline_options = _footprint_pipeline_options(
        input_bam=input_bam,
        output_bam=output_bam,
        train_rids=train_rids,
        edge_trim=edge_trim,
        circular=circular,
        mode=mode,
        context_size=context_size,
        msp_min_size=msp_min_size,
        nuc_min_size=nuc_min_size,
        min_mapq=min_mapq,
        prob_threshold=prob_threshold,
        min_read_length=min_read_length,
        with_scores=with_scores,
        n_cores=n_cores,
        primary_only=primary_only,
        output_posteriors=output_posteriors,
        write_msps=write_msps,
        io_threads=io_threads,
    )

    result = _dispatch_requested_parallel_pipeline(
        model_path=model_path,
        region_parallel=region_parallel,
        region_size=region_size,
        skip_scaffolds=skip_scaffolds,
        chroms=chroms,
        streaming_pipeline=streaming_pipeline,
        chunk_size=chunk_size,
        max_reads=max_reads,
        debug_timing=debug_timing,
        process_unmapped=process_unmapped,
        pipeline_options=pipeline_options,
    )
    if result is not None:
        return result

    return _dispatch_legacy_footprint_pipeline(
        model=model,
        model_path=model_path,
        n_cores=n_cores,
        max_reads=max_reads,
        debug_timing=debug_timing,
        pipeline_kwargs=pipeline_options.as_kwargs(),
    )
