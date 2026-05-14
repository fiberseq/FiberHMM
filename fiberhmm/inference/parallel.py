"""FiberHMM BAM processing compatibility orchestration."""

from typing import Optional, Set, Tuple

import pysam

from fiberhmm.core.model_io import freeze_model_for_inference, load_model
from fiberhmm.inference.legacy_pipeline import (
    _process_and_write_chunk,  # noqa: F401
    _process_bam_legacy_pipeline,
)
from fiberhmm.inference.mp_context import _MP_CONTEXT, _select_mp_context  # noqa: F401
from fiberhmm.inference.read_filters import (  # noqa: F401
    ReadFilterConfig,
    streaming_skip_reason,
)
from fiberhmm.inference.region_pipeline import (
    _process_bam_region_parallel,
    _process_bam_region_parallel_fused,  # noqa: F401
    _process_bed_region_parallel,  # noqa: F401
)
from fiberhmm.inference.region_planning import (
    _get_genome_regions,  # noqa: F401
)
from fiberhmm.inference.region_planning import (
    _is_main_chromosome as _region_is_main_chromosome,
)
from fiberhmm.inference.region_workers import (
    _init_fused_region_worker,  # noqa: F401
    _init_region_worker,  # noqa: F401
    _process_region_to_bam,  # noqa: F401
    _process_region_to_bam_fused,  # noqa: F401
    _process_region_to_bed,  # noqa: F401
)
from fiberhmm.inference.streaming_drain import (
    _drain_oldest_chunk,  # noqa: F401
    _drain_oldest_fused_chunk,  # noqa: F401
)
from fiberhmm.inference.streaming_pipeline import (
    _process_bam_streaming_pipeline,
    _process_bam_streaming_pipeline_fused,  # noqa: F401
)
from fiberhmm.inference.streaming_workers import (
    _init_bam_worker,  # noqa: F401
    _init_fused_worker,  # noqa: F401
    _process_chunk_worker,  # noqa: F401
    _process_fused_payload_chunk_worker,  # noqa: F401
    _process_payload_chunk_worker,  # noqa: F401
)

_is_main_chromosome = _region_is_main_chromosome


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
    if isinstance(model_or_path, str):
        model_path = model_or_path
        model = freeze_model_for_inference(load_model(model_path))
    else:
        model = model_or_path
        model_path = None

    # Dispatch to region-parallel if requested (requires model_path for workers)
    if region_parallel:
        if model_path is None:
            print("Warning: region_parallel requires model path, falling back to standard parallel")
        else:
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
                io_threads=io_threads
            )

    # Dispatch to streaming pipeline if requested (requires model_path for workers)
    if streaming_pipeline:
        if model_path is None:
            print("Warning: streaming_pipeline requires model path, falling back to standard parallel")
        else:
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

    return _process_bam_legacy_pipeline(
        input_bam=input_bam,
        output_bam=output_bam,
        model=model,
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
        max_reads=max_reads,
        debug_timing=debug_timing,
        primary_only=primary_only,
        output_posteriors=output_posteriors,
        write_msps=write_msps,
        io_threads=io_threads,
    )
