"""FiberHMM region-parallel processing and worker management."""

import os
import shutil
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from typing import Optional, Set, Tuple

import numpy as np
import pysam

from fiberhmm.core.model_io import freeze_model_for_inference, load_model
from fiberhmm.inference.bam_output import (
    _concatenate_region_bams,
    _sort_and_index_bam,
)
from fiberhmm.inference.engine import (
    _extract_fiber_read_from_pysam,
    _process_single_read,
    make_apply_payload,
)
from fiberhmm.inference.mp_context import _MP_CONTEXT, _select_mp_context  # noqa: F401
from fiberhmm.inference.read_filters import ReadFilterConfig, streaming_skip_reason
from fiberhmm.inference.region_planning import (
    _get_genome_regions,
)
from fiberhmm.inference.region_planning import (
    _is_main_chromosome as _region_is_main_chromosome,
)
from fiberhmm.inference.region_types import (
    RegionBamAggregation,
    RegionBamResult,
    RegionBamWorkItem,
    RegionBedAggregation,
    RegionBedResult,
    RegionBedWorkItem,
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
from fiberhmm.inference.streaming_workers import (
    _init_bam_worker,
    _init_fused_worker,
    _process_chunk_worker,
    _process_fused_payload_chunk_worker,
    _process_payload_chunk_worker,
)
from fiberhmm.inference.tagging import (
    set_legacy_apply_tags,
)
from fiberhmm.inference.worker_results import coerce_worker_chunk_result
from fiberhmm.posteriors.region_tsv import (
    merge_region_posteriors_tsv as _merge_region_posteriors_tsv,
)
from fiberhmm.posteriors.region_tsv import (
    region_posteriors_tsv_output_path,
)

_is_main_chromosome = _region_is_main_chromosome

# Optional: inline posteriors export
try:
    from fiberhmm.posteriors.hdf5_backend import PosteriorWriter, get_ref_positions_from_read
    HAS_POSTERIOR_WRITER = True
except ImportError:
    HAS_POSTERIOR_WRITER = False


def _process_bam_region_parallel(input_bam: str, output_bam: str,
                                   model_path: str, train_rids: Set[str],
                                   edge_trim: int, circular: bool,
                                   mode: str, context_size: int,
                                   msp_min_size: int,
                                   nuc_min_size: int = 85,
                                   min_mapq: int = 0,
                                   prob_threshold: int = 0,
                                   min_read_length: int = 0,
                                   with_scores: bool = False,
                                   n_cores: int = 1,
                                   region_size: int = 10_000_000,
                                   skip_scaffolds: bool = False,
                                   chroms: Optional[Set[str]] = None,
                                   primary_only: bool = False,
                                   output_posteriors: Optional[str] = None,
                                   write_msps: bool = True,
                                   io_threads: int = 4) -> Tuple[int, int]:
    """
    Process BAM using region-based parallelism with indexed access.

    Each worker independently reads from the BAM using the index,
    enabling true parallel I/O. Results are concatenated at the end.

    Args:
        region_size: Size of each region in bp (default 10MB)
        skip_scaffolds: If True, skip scaffold/contig chromosomes
        chroms: If provided, only process these chromosomes
        primary_only: If True, skip secondary/supplementary alignments
        output_posteriors: If provided, write HMM posteriors to this H5 file
        write_msps: If True, write as/al/aq MSP tags to output BAM

    Returns:
        (total_reads_processed, reads_with_footprints)
    """
    import shutil
    import tempfile
    import time

    start_time = time.time()
    return_posteriors = output_posteriors is not None

    # Check that BAM is indexed
    if not os.path.exists(input_bam + '.bai') and not os.path.exists(input_bam.replace('.bam', '.bai')):
        print("Indexing input BAM for region-parallel processing...")
        pysam.index(input_bam)

    # Get regions
    regions = _get_genome_regions(input_bam, region_size, skip_scaffolds, chroms)
    print(f"Processing {len(regions)} regions with {n_cores} cores...")
    if return_posteriors:
        print(f"Posteriors will be written to: {output_posteriors}")
    sys.stdout.flush()

    # Create temp directory in output folder for easier cleanup
    output_dir = os.path.dirname(os.path.abspath(output_bam))
    temp_dir = tempfile.mkdtemp(prefix='.fiberhmm_tmp_', dir=output_dir)

    try:
        # Prepare parameters (will be passed to initializer)
        params = {
            'edge_trim': edge_trim,
            'circular': circular,
            'mode': mode,
            'context_size': context_size,
            'msp_min_size': msp_min_size,
            'nuc_min_size': nuc_min_size,
            'min_mapq': min_mapq,
            'prob_threshold': prob_threshold,
            'min_read_length': min_read_length,
            'with_scores': with_scores,
            'train_rids': train_rids,
            'primary_only': primary_only,
            'return_posteriors': return_posteriors,
            'write_msps': write_msps,
            'io_threads': io_threads,
        }

        # Work items - include temp H5 path if posteriors requested
        work_items = []
        for i, region in enumerate(regions):
            temp_bam = os.path.join(temp_dir, f'region_{i:06d}.bam')
            temp_h5 = os.path.join(temp_dir, f'region_{i:06d}.tsv') if return_posteriors else None
            work_items.append(RegionBamWorkItem(region, input_bam, temp_bam, temp_h5))

        # Process regions in parallel
        aggregation = RegionBamAggregation()
        first_result_time = None

        # Use initializer to load model once per worker
        print(f"  Initializing {n_cores} worker processes (loading HMM model in each)...")
        sys.stdout.flush()
        pool_start = time.time()

        with ProcessPoolExecutor(
            max_workers=n_cores,
            mp_context=_MP_CONTEXT,
            initializer=_init_region_worker,
            initargs=(model_path, params)
        ) as executor:
            futures = {executor.submit(_process_region_to_bam, item): i
                      for i, item in enumerate(work_items)}

            for future in as_completed(futures):
                try:
                    result = RegionBamResult.from_value(future.result())
                    include_tsv = bool(
                        result.temp_tsv_path and os.path.exists(result.temp_tsv_path)
                    )
                    aggregation.add_result(futures[future], result, include_tsv=include_tsv)

                    # Track first result
                    if first_result_time is None:
                        first_result_time = time.time()
                        init_time = first_result_time - pool_start
                        print(f"  Workers ready ({init_time:.1f}s). Processing regions...")
                        sys.stdout.flush()

                    elapsed = time.time() - start_time
                    rate = aggregation.total_reads / elapsed if elapsed > 0 else 0
                    print(f"\r  Regions: {aggregation.completed}/{len(regions)} | "
                          f"Reads: {aggregation.total_reads:,} | "
                          f"With footprints: {aggregation.reads_with_footprints:,} | "
                          f"{rate:.1f} reads/s", end='')
                    sys.stdout.flush()

                except Exception as e:
                    print(f"\nError processing region: {e}")
                    raise

        print()  # Newline after progress

        # Print skip reasons summary
        if aggregation.total_skipped > 0:
            total_encountered = aggregation.total_reads + aggregation.total_skipped
            print(
                f"  Processed: {aggregation.total_reads:,} | "
                f"Skipped: {aggregation.total_skipped:,} | "
                f"With footprints: {aggregation.reads_with_footprints:,}"
            )
            print("  Skip reasons:")
            for reason, count in sorted(
                aggregation.skip_reasons.items(), key=lambda x: -x[1]
            ):
                if count > 0:
                    pct = 100 * count / total_encountered
                    print(f"    {reason}: {count:,} ({pct:.1f}%)")

        # Sort temp BAMs by region order and filter to non-empty
        aggregation.temp_bams.sort(key=lambda x: x[0])
        non_empty_bams = [bam for _, bam in aggregation.temp_bams
                         if os.path.exists(bam) and os.path.getsize(bam) > 0]

        _concatenate_region_bams(input_bam, output_bam, non_empty_bams, temp_dir)

        sys.stdout.flush()

        # Verify output was created
        if os.path.exists(output_bam):
            output_size_gb = os.path.getsize(output_bam) / (1024**3)
            print(f"Output BAM: {output_size_gb:.2f}GB")

        # Index the output BAM (sort first if needed)
        print("Step: Index/Sort...")
        _sort_and_index_bam(output_bam, threads=n_cores)

        # Merge temp TSV files if posteriors were requested
        if return_posteriors and aggregation.temp_tsvs:
            print(f"Merging {len(aggregation.temp_tsvs)} posterior files...")
            merge_start = time.time()
            n_fibers = _merge_region_posteriors_tsv(
                aggregation.temp_tsvs, output_posteriors,
                mode, context_size, edge_trim, input_bam,
            )
            merge_time = time.time() - merge_start

            # Figure out actual output path (always .tsv.gz)
            tsv_path = region_posteriors_tsv_output_path(output_posteriors)

            if os.path.exists(tsv_path):
                file_size = os.path.getsize(tsv_path) / (1024 * 1024)
                print(
                    f"Posteriors: {n_fibers:,} fibers -> {tsv_path} "
                    f"({file_size:.1f} MB, {merge_time:.1f}s)"
                )
                if output_posteriors.endswith('.h5'):
                    print(f"  To convert to H5: python posteriors_io.py tsv2h5 {tsv_path} {output_posteriors}")

        elapsed = time.time() - start_time
        rate = aggregation.total_reads / elapsed if elapsed > 0 else 0
        print(
            f"Completed: {aggregation.total_reads:,} reads | "
            f"{aggregation.reads_with_footprints:,} with footprints | "
            f"{rate:.1f} reads/s | {elapsed:.1f}s"
        )

        return aggregation.total_reads, aggregation.reads_with_footprints

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def _process_bed_region_parallel(input_bam: str, output_bed: str,
                                  model_path: str, train_rids: Set[str],
                                  edge_trim: int, circular: bool,
                                  mode: str, context_size: int,
                                  msp_min_size: int,
                                  nuc_min_size: int = 85,
                                  min_mapq: int = 0,
                                  prob_threshold: int = 0,
                                  min_read_length: int = 0,
                                  with_scores: bool = False,
                                  n_cores: int = 1,
                                  region_size: int = 10_000_000,
                                  skip_scaffolds: bool = False,
                                  chroms: Optional[Set[str]] = None,
                                  primary_only: bool = False) -> Tuple[int, int]:
    """
    Process BAM using region-based parallelism, writing BED output directly.

    This is more space-efficient than processing to BAM first when only
    BED/bigBed output is needed - no large temp BAMs are created.

    Returns:
        (total_reads_processed, reads_with_footprints)
    """
    import time

    start_time = time.time()

    # Check that BAM is indexed
    if not os.path.exists(input_bam + '.bai') and not os.path.exists(input_bam.replace('.bam', '.bai')):
        print("Indexing input BAM for region-parallel processing...")
        pysam.index(input_bam)

    # Get regions
    regions = _get_genome_regions(input_bam, region_size, skip_scaffolds, chroms)
    print(f"Processing {len(regions)} regions with {n_cores} cores (BED output)...")
    sys.stdout.flush()

    # Create temp directory for BED files (small compared to BAMs)
    output_dir = os.path.dirname(os.path.abspath(output_bed))
    temp_dir = tempfile.mkdtemp(prefix='.fiberhmm_bed_tmp_', dir=output_dir)

    try:
        params = {
            'edge_trim': edge_trim,
            'circular': circular,
            'mode': mode,
            'context_size': context_size,
            'msp_min_size': msp_min_size,
            'nuc_min_size': nuc_min_size,
            'min_mapq': min_mapq,
            'prob_threshold': prob_threshold,
            'min_read_length': min_read_length,
            'with_scores': with_scores,
            'train_rids': train_rids,
            'primary_only': primary_only
        }

        # Work items - write temp BED files
        work_items = []
        for i, region in enumerate(regions):
            temp_bed = os.path.join(temp_dir, f'region_{i:06d}.bed')
            work_items.append(RegionBedWorkItem(region, input_bam, temp_bed))

        aggregation = RegionBedAggregation()
        first_result_time = None

        print(f"  Initializing {n_cores} worker processes (loading HMM model in each)...")
        sys.stdout.flush()
        pool_start = time.time()

        with ProcessPoolExecutor(
            max_workers=n_cores,
            mp_context=_MP_CONTEXT,
            initializer=_init_region_worker,
            initargs=(model_path, params)
        ) as executor:
            futures = {executor.submit(_process_region_to_bed, item): i
                      for i, item in enumerate(work_items)}

            for future in as_completed(futures):
                try:
                    result = RegionBedResult.from_value(future.result())
                    aggregation.add_result(futures[future], result)

                    # Track first result
                    if first_result_time is None:
                        first_result_time = time.time()
                        init_time = first_result_time - pool_start
                        print(f"  Workers ready ({init_time:.1f}s). Processing regions...")
                        sys.stdout.flush()

                    elapsed = time.time() - start_time
                    rate = aggregation.total_reads / elapsed if elapsed > 0 else 0
                    print(f"\r  Regions: {aggregation.completed}/{len(regions)} | "
                          f"Reads: {aggregation.total_reads:,} | "
                          f"With footprints: {aggregation.reads_with_footprints:,} | "
                          f"{rate:.1f} reads/s", end='')
                    sys.stdout.flush()

                except Exception as e:
                    print(f"\nError processing region: {e}")
                    raise

        print()  # Newline after progress

        # Sort temp BEDs by region order and concatenate
        aggregation.temp_beds.sort(key=lambda x: x[0])
        non_empty_beds = [bed for _, bed in aggregation.temp_beds
                         if os.path.exists(bed) and os.path.getsize(bed) > 0]

        print(f"Concatenating {len(non_empty_beds)} region BED files...")
        sys.stdout.flush()

        with open(output_bed, 'wb') as fout:
            for bed_path in non_empty_beds:
                with open(bed_path, 'rb') as fin:
                    shutil.copyfileobj(fin, fout)

        elapsed = time.time() - start_time
        rate = aggregation.total_reads / elapsed if elapsed > 0 else 0
        print(
            f"Completed: {aggregation.total_reads:,} reads | "
            f"{aggregation.reads_with_footprints:,} with footprints | "
            f"{rate:.1f} reads/s | {elapsed:.1f}s"
        )

        return aggregation.total_reads, aggregation.reads_with_footprints

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _process_and_write_chunk(chunk_reads: list, chunk_read_objs: list,
                              outbam, model, executor,
                              edge_trim: int, circular: bool,
                              mode: str, context_size: int,
                              msp_min_size: int, nuc_min_size: int = 85,
                              with_scores: bool = False,
                              return_posteriors: bool = False,
                              write_msps: bool = True) -> Tuple[int, int, int, Optional[list]]:
    """
    Process a chunk of reads and write to BAM.

    Returns:
        (reads_with_footprints, no_footprints, worker_failures,
         results_with_posteriors or None)

        If return_posteriors=True, returns results list for caller to write posteriors.
    """

    if executor is not None:
        # Parallel: submit chunk to worker
        future = executor.submit(
            _process_chunk_worker,
            chunk_reads, edge_trim, circular, mode, context_size, msp_min_size,
            nuc_min_size, with_scores, return_posteriors
        )
        results, worker_failures = coerce_worker_chunk_result(future.result())
    else:
        # Single-threaded: process directly
        results = []
        worker_failures = 0
        for fiber_read in chunk_reads:
            result = _process_single_read(
                fiber_read, model, edge_trim, circular,
                mode, context_size, msp_min_size, nuc_min_size=nuc_min_size,
                with_scores=with_scores,
                return_posteriors=return_posteriors
            )
            results.append(result)

    # Write annotated reads
    reads_with_footprints = 0
    no_footprints = 0
    for read_obj, result in zip(chunk_read_objs, results):
        if result is not None:
            set_legacy_apply_tags(read_obj, result, with_scores, write_msps)
            reads_with_footprints += 1
        else:
            no_footprints += 1

        outbam.write(read_obj)

    # Return results for posteriors if requested
    if return_posteriors:
        return (
            reads_with_footprints,
            no_footprints,
            worker_failures,
            list(zip(chunk_read_objs, chunk_reads, results)),
        )
    return reads_with_footprints, no_footprints, worker_failures, None


def _process_bam_region_parallel_fused(
    input_bam: str, output_bam: str,
    apply_model_path: str, recall_model_path: Optional[str],
    train_rids: Set[str],
    edge_trim: int, circular: bool,
    mode: str, context_size: int,
    msp_min_size: int, nuc_min_size: int,
    min_mapq: int, prob_threshold: int, min_read_length: int,
    with_scores: bool,
    min_llr: float, min_opps: int, unify_threshold: int,
    emission_uplift: float,
    also_write_legacy: bool, downstream_compat: bool,
    n_cores: int, region_size: int, skip_scaffolds: bool,
    chroms: Optional[Set[str]], io_threads: int,
    primary_only: bool = False,
    ref_fasta_path: Optional[str] = None,
):
    """Region-parallel fused apply+recall.

    Splits the BAM into genomic regions, runs fused apply+recall in each
    region as an independent worker, then concatenates sorted temp BAMs
    in region order.  Input BAM must be coordinate-sorted + indexed.

    Output is coordinate-sorted with no sort pass needed.
    """
    start_time = time.time()

    regions = _get_genome_regions(input_bam, region_size, skip_scaffolds, chroms)
    print(f"Processing {len(regions)} regions with {n_cores} cores (fused apply+recall)...")
    sys.stdout.flush()

    output_dir = os.path.dirname(os.path.abspath(output_bam))
    temp_dir = tempfile.mkdtemp(prefix='.fiberhmm_call_tmp_', dir=output_dir)

    params = {
        'edge_trim': edge_trim, 'circular': circular,
        'mode': mode, 'context_size': context_size,
        'msp_min_size': msp_min_size, 'nuc_min_size': nuc_min_size,
        'min_mapq': min_mapq, 'prob_threshold': prob_threshold,
        'min_read_length': min_read_length, 'with_scores': with_scores,
        'train_rids': train_rids, 'primary_only': primary_only,
        'io_threads': io_threads,
        'min_llr': min_llr, 'min_opps': min_opps,
        'unify_threshold': unify_threshold,
        'also_write_legacy': also_write_legacy,
        'downstream_compat': downstream_compat,
        # Path string, NOT an open handle: pysam.FastaFile is not fork-safe,
        # so each worker opens it lazily in _init_fused_region_worker.
        'ref_fasta_path': ref_fasta_path,
    }

    work_items = [
        RegionBamWorkItem(
            (r[0], r[1], r[2]), input_bam,
            os.path.join(temp_dir, f'region_{i:06d}.bam'),
        )
        for i, r in enumerate(regions)
    ]

    try:
        aggregation = RegionBamAggregation()

        print(f"  Initializing {n_cores} workers (loading apply model + LLR tables)...")
        sys.stdout.flush()
        pool_start = time.time()
        first_result = None

        with ProcessPoolExecutor(
            max_workers=n_cores,
            mp_context=_MP_CONTEXT,
            initializer=_init_fused_region_worker,
            initargs=(apply_model_path, recall_model_path, emission_uplift, params),
        ) as executor:
            futures = {executor.submit(_process_region_to_bam_fused, item): i
                       for i, item in enumerate(work_items)}

            for future in as_completed(futures):
                result = RegionBamResult.from_value(future.result())
                aggregation.add_result(futures[future], result)
                if first_result is None:
                    first_result = time.time()
                    print(f"  Workers ready ({first_result - pool_start:.1f}s). Processing...")
                    sys.stdout.flush()
                elapsed = time.time() - start_time
                rate = aggregation.total_reads / elapsed if elapsed > 0 else 0
                print(f"\r  Regions: {aggregation.completed}/{len(regions)} | "
                      f"Reads: {aggregation.total_reads:,} | "
                      f"With FP: {aggregation.reads_with_footprints:,} | "
                      f"{rate:.0f} r/s", end='')
                sys.stdout.flush()
        print()

        if aggregation.total_skipped > 0:
            total_enc = aggregation.total_reads + aggregation.total_skipped
            print(
                f"  Processed: {aggregation.total_reads:,} | "
                f"Skipped: {aggregation.total_skipped:,} | "
                f"With FP: {aggregation.reads_with_footprints:,}"
            )
            print("  Skip reasons:")
            for reason, count in sorted(
                aggregation.skip_reasons.items(), key=lambda x: -x[1]
            ):
                if count > 0:
                    print(f"    {reason}: {count:,} ({100*count/total_enc:.1f}%)")

        # Concat region BAMs in region-index order — preserves coord sort.
        aggregation.temp_bams.sort(key=lambda x: x[0])
        non_empty = [bam for _, bam in aggregation.temp_bams
                     if os.path.exists(bam) and os.path.getsize(bam) > 0]
        _concatenate_region_bams(input_bam, output_bam, non_empty, temp_dir)

        # Index directly (input sorted → each region sorted → concat sorted).
        try:
            pysam.index(output_bam)
        except pysam.SamtoolsError:
            pass

        elapsed = time.time() - start_time
        rate = aggregation.total_reads / elapsed if elapsed > 0 else 0
        print(
            f"  Total: {aggregation.total_reads:,} reads, "
            f"{aggregation.reads_with_footprints:,} with footprints, "
            f"{rate:.1f} r/s"
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return aggregation.total_reads, aggregation.reads_with_footprints


def _process_bam_streaming_pipeline_fused(
    input_bam: str, output_bam: str,
    model_path: str, recall_model_path: str,
    train_rids,
    edge_trim: int, circular: bool,
    mode: str, context_size: int,
    msp_min_size: int, nuc_min_size: int,
    min_mapq: int, prob_threshold: int, min_read_length: int,
    with_scores: bool,
    min_llr: float, min_opps: int, unify_threshold: int,
    emission_uplift: float,
    also_write_legacy: bool, downstream_compat: bool,
    max_reads: int, n_cores: int, chunk_size: int,
    io_threads: int,
    process_unmapped: bool = False,
    primary_only: bool = False,
    ref_fasta_path: Optional[str] = None,
):
    """Fused apply+recall streaming pipeline: one worker pool does both
    the HMM nucleosome/MSP calls and the TF Kadane scan per read.

    Eliminates the apply→recall pipe serialization (which was
    ~2-7× slower than the sum of the two stages run alone on this Mac
    benchmark) and the duplicate MM/ML parse + encode done by the
    streaming recall stage.
    """
    # Streaming mode reads the BAM sequentially in the main process, so it can
    # hand a live FastaFile to make_apply_payload for on-the-fly MD fallback on
    # raw DAF BAMs. The handle is opened after BAM output setup and closed from
    # the processing finally block so failures do not leak it.
    ref_fasta = None
    pysam.set_verbosity(0)
    max_inflight = n_cores + 2
    start_time = time.time()
    counters = {
        'reads_with_footprints': 0,
        'no_footprints': 0,
        'worker_failures': 0,
        'written': 0,
    }

    total_reads = 0
    skipped = 0
    skip_reasons = {
        'unmapped': 0, 'secondary_supplementary': 0, 'low_mapq': 0,
        'too_short': 0, 'training_excluded': 0, 'no_modifications': 0,
        'extraction_failed': 0,
    }
    filter_config = ReadFilterConfig(
        min_mapq=min_mapq,
        min_read_length=min_read_length,
        primary_only=primary_only,
        process_unmapped=process_unmapped,
        train_rids=train_rids,
    )

    _log = sys.stderr if output_bam == '-' else sys.stdout

    _output_target = output_bam
    if output_bam == '-':
        _output_target = os.fdopen(1, 'wb', closefd=False)

    with pysam.AlignmentFile(input_bam, "rb", threads=io_threads, check_sq=False) as inbam:
        with pysam.AlignmentFile(_output_target, "wb", header=inbam.header, threads=io_threads) as outbam:
            if ref_fasta_path:
                ref_fasta = pysam.FastaFile(ref_fasta_path)

            executor = ProcessPoolExecutor(
                max_workers=n_cores,
                mp_context=_MP_CONTEXT,
                initializer=_init_fused_worker,
                initargs=(model_path, recall_model_path, emission_uplift, False),
            )

            inflight = deque()
            chunk_payloads = []
            chunk_read_objs = []
            chunk_skip_flags = []
            last_progress_reads = 0
            last_progress_time = time.time()

            def _buffer_skip(rd, reason):
                nonlocal skipped
                chunk_read_objs.append(rd)
                chunk_skip_flags.append(True)
                skipped += 1
                skip_reasons[reason] += 1

            try:
                for read in inbam.fetch(until_eof=True):
                    skip_reason = streaming_skip_reason(read, filter_config)
                    if skip_reason:
                        _buffer_skip(read, skip_reason)
                        continue

                    payload = make_apply_payload(read, mode=mode, ref_fasta=ref_fasta)
                    if payload is None:
                        _buffer_skip(read, 'no_modifications')
                        continue

                    chunk_payloads.append(payload)
                    chunk_read_objs.append(read)
                    chunk_skip_flags.append(False)
                    total_reads += 1

                    if max_reads and total_reads >= max_reads:
                        break

                    if len(chunk_read_objs) >= chunk_size:
                        if len(inflight) >= max_inflight:
                            _drain_oldest_fused_chunk(
                                inflight, outbam, with_scores,
                                also_write_legacy, downstream_compat, counters,
                            )
                        future = executor.submit(
                            _process_fused_payload_chunk_worker,
                            chunk_payloads, edge_trim, circular, mode,
                            context_size, msp_min_size, nuc_min_size,
                            with_scores, prob_threshold,
                            mode, context_size,
                            min_llr, min_opps, unify_threshold,
                        )
                        inflight.append((future, chunk_read_objs, chunk_payloads, chunk_skip_flags))
                        chunk_payloads = []
                        chunk_read_objs = []
                        chunk_skip_flags = []

                        now = time.time()
                        elapsed = now - start_time
                        avg = total_reads / elapsed if elapsed > 0 else 0
                        dt = now - last_progress_time
                        inst = (total_reads - last_progress_reads) / dt if dt > 0 else 0
                        last_progress_reads = total_reads
                        last_progress_time = now
                        print(f"\r  Fused: {total_reads:,} | Skipped: {skipped:,} | "
                              f"Inflight: {len(inflight)} | {inst:.0f} r/s (avg {avg:.0f})",
                              end='', file=_log)
                        _log.flush()

                # Final partial chunk
                if chunk_read_objs:
                    if len(inflight) >= max_inflight:
                        _drain_oldest_fused_chunk(
                            inflight, outbam, with_scores,
                            also_write_legacy, downstream_compat, counters,
                        )
                    if chunk_payloads:
                        future = executor.submit(
                            _process_fused_payload_chunk_worker,
                            chunk_payloads, edge_trim, circular, mode,
                            context_size, msp_min_size, nuc_min_size,
                            with_scores, prob_threshold,
                            mode, context_size,
                            min_llr, min_opps, unify_threshold,
                        )
                    else:
                        future = Future()
                        future.set_result([])
                    inflight.append((future, chunk_read_objs, chunk_payloads, chunk_skip_flags))

                while inflight:
                    _drain_oldest_fused_chunk(
                        inflight, outbam, with_scores,
                        also_write_legacy, downstream_compat, counters,
                    )
            finally:
                try:
                    executor.shutdown(wait=True)
                finally:
                    if ref_fasta is not None:
                        ref_fasta.close()
                        ref_fasta = None

    elapsed = time.time() - start_time
    rate = total_reads / elapsed if elapsed > 0 else 0
    reads_with_fp = counters['reads_with_footprints']
    print(f"\r  Fused: {total_reads:,} | Skipped: {skipped:,} | "
          f"With footprints: {reads_with_fp:,} | {rate:.1f} r/s", file=_log)
    if counters['worker_failures']:
        print(
            f"  Worker read failures: {counters['worker_failures']:,} "
            f"(passed through unchanged)",
            file=_log,
        )
    if skipped > 0:
        print("  Skip reasons:", file=_log)
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = 100 * count / (total_reads + skipped)
                print(f"    {reason}: {count:,} ({pct:.1f}%)", file=_log)

    return total_reads, reads_with_fp


def _process_bam_streaming_pipeline(
    input_bam: str, output_bam: str,
    model_path: str, train_rids: Set[str],
    edge_trim: int, circular: bool,
    mode: str, context_size: int,
    msp_min_size: int,
    nuc_min_size: int = 85,
    min_mapq: int = 0,
    prob_threshold: int = 0,
    min_read_length: int = 0,
    with_scores: bool = False,
    n_cores: int = 4,
    chunk_size: int = 500,
    max_inflight: Optional[int] = None,
    io_threads: int = 4,
    primary_only: bool = False,
    output_posteriors: Optional[str] = None,
    write_msps: bool = True,
    max_reads: Optional[int] = None,
    debug_timing: bool = False,
    process_unmapped: bool = False,
) -> Tuple[int, int]:
    """
    Streaming producer-consumer pipeline for BAM processing.

    Note: ``ref_fasta`` for the DAF MD-only fallback is set to None here;
    --reference CLI plumbing is Step 4 of the daf-md fallback work.
    MD-only DAF BAMs work without it.

    Uses a sliding window of in-flight futures to overlap I/O and compute.
    Works with unaligned/unindexed BAMs and stdin. Order-preserving.

    The main process reads BAM sequentially, extracts fiber data, and submits
    chunks to a process pool. Multiple chunks can be in flight simultaneously
    (bounded by max_inflight), keeping workers saturated while the main
    process continues reading and writing.

    Args:
        max_inflight: Max chunks in flight (default: 2 * n_cores)
        chunk_size: Reads per compute chunk (default: 500)
        io_threads: htslib decompression threads (default: 4)
        process_unmapped: If True, process unmapped reads that have sequences

    Returns:
        (total_reads_processed, reads_with_footprints)
    """
    if max_inflight is None:
        max_inflight = 2 * n_cores

    pysam.set_verbosity(0)

    ref_fasta = None    # placeholder; --reference plumbing is Step 4 of the
                        # daf-md fallback work. MD-only DAF BAMs work without it.

    total_reads = 0
    skip_reasons = {
        'unmapped': 0,
        'secondary_supplementary': 0,
        'low_mapq': 0,
        'too_short': 0,
        'training_excluded': 0,
        'no_modifications': 0,
        'extraction_failed': 0,
        'no_footprints': 0,
    }
    filter_config = ReadFilterConfig(
        min_mapq=min_mapq,
        min_read_length=min_read_length,
        primary_only=primary_only,
        process_unmapped=process_unmapped,
        train_rids=train_rids,
    )

    counters = {
        'reads_with_footprints': 0,
        'no_footprints': 0,
        'worker_failures': 0,
        'written': 0,
    }

    # Initialized after BAM handles open so failure cleanup is centralized in
    # the processing finally block.
    posterior_writer = None
    posterior_stats = None
    return_posteriors = False

    # When writing to stdout, send progress to stderr to avoid corrupting BAM
    _log = sys.stderr if output_bam == '-' else sys.stdout

    print(f"Processing BAM (streaming pipeline, {n_cores} workers, "
          f"chunk_size={chunk_size}, max_inflight={max_inflight})...", file=_log)
    _log.flush()

    start_time = time.time()

    # For stdout output, ensure pysam writes to the real stdout fd
    # (sys.stdout may have been redirected to stderr for logging)
    _output_target = output_bam
    if output_bam == '-':
        _output_target = os.fdopen(1, 'wb', closefd=False)

    with pysam.AlignmentFile(input_bam, "rb", threads=io_threads, check_sq=False) as inbam:
        with pysam.AlignmentFile(_output_target, "wb", header=inbam.header, threads=io_threads) as outbam:

            if output_posteriors:
                if HAS_POSTERIOR_WRITER:
                    posterior_writer = PosteriorWriter(
                        output_posteriors, mode, context_size,
                        edge_trim, input_bam, batch_size=1000
                    )
                    return_posteriors = True
                    print(f"Posteriors will be written to: {output_posteriors}", file=_log)
                else:
                    print(
                        "WARNING: posterior_writer.py not found, skipping posteriors export",
                        file=_log,
                    )

            # Create worker pool
            executor = ProcessPoolExecutor(
                max_workers=n_cores,
                mp_context=_MP_CONTEXT,
                initializer=_init_bam_worker,
                initargs=(model_path, debug_timing)
            )

            inflight = deque()
            chunk_reads = []           # only fiber_reads to be processed
            chunk_read_objs = []       # ALL reads in stream order (skipped + processed)
            chunk_skip_flags = []      # bool per chunk_read_objs entry: True = pass-through
            skipped = 0
            last_progress_reads = 0
            last_progress_time = time.time()

            def _buffer_skip(rd, reason):
                """Buffer a skipped read at its stream position to preserve sort order."""
                nonlocal skipped
                chunk_read_objs.append(rd)
                chunk_skip_flags.append(True)
                skipped += 1
                skip_reasons[reason] += 1

            try:
                for read in inbam.fetch(until_eof=True):
                    # --- Filter phase ---
                    # Skipped reads are buffered into the chunk (with skip_flag=True)
                    # rather than written immediately — the drain step writes them
                    # at their original stream position so a coordinate-sorted
                    # input BAM produces a coordinate-sorted output BAM.
                    skip_reason = streaming_skip_reason(read, filter_config)
                    if skip_reason:
                        _buffer_skip(read, skip_reason)
                        continue

                    # --- Build slim payload (fast — no MM/ML parsing) ---
                    # The MM/ML parse is now done inside the worker (see
                    # _process_payload_chunk_worker) which lifts the apply
                    # throughput ceiling from ~150-300 r/s (serial main parse
                    # bottleneck) to whatever the worker pool can sustain.
                    # Reads with no usable modification data or per-read worker
                    # failures come back as result=None and are written through
                    # without footprint tags (handled in _drain_oldest_chunk).
                    payload = make_apply_payload(read, mode=mode, ref_fasta=ref_fasta)
                    if payload is None:
                        _buffer_skip(read, 'no_modifications')
                        continue

                    # --- Buffer into chunk ---
                    chunk_reads.append(payload)
                    chunk_read_objs.append(read)
                    chunk_skip_flags.append(False)
                    total_reads += 1

                    # Check max reads limit
                    if max_reads and total_reads >= max_reads:
                        break

                    # --- Submit when chunk is full ---
                    # Use chunk_read_objs length so skipped reads also count toward
                    # the chunk size (otherwise pure-skip stretches would buffer
                    # unboundedly waiting for processable reads).
                    if len(chunk_read_objs) >= chunk_size:
                        # Drain oldest if at capacity
                        if len(inflight) >= max_inflight:
                            _drain_oldest_chunk(
                                inflight, outbam, with_scores, write_msps,
                                posterior_writer, counters
                            )

                        # Submit chunk of slim payloads to slim-IPC workers
                        future = executor.submit(
                            _process_payload_chunk_worker,
                            chunk_reads, edge_trim, circular, mode, context_size,
                            msp_min_size, nuc_min_size, with_scores, return_posteriors,
                            prob_threshold,
                        )
                        inflight.append((future, chunk_read_objs, chunk_reads, chunk_skip_flags))
                        chunk_reads = []
                        chunk_read_objs = []
                        chunk_skip_flags = []

                        # Progress update
                        now = time.time()
                        elapsed = now - start_time
                        avg_rate = total_reads / elapsed if elapsed > 0 else 0
                        dt = now - last_progress_time
                        inst_rate = (total_reads - last_progress_reads) / dt if dt > 0 else 0
                        last_progress_reads = total_reads
                        last_progress_time = now
                        print(f"\r  Processed: {total_reads:,} | "
                              f"Skipped: {skipped:,} | "
                              f"Inflight: {len(inflight)} | "
                              f"{inst_rate:.0f} reads/s (avg {avg_rate:.0f})", end='', file=_log)
                        _log.flush()

                # Submit final partial chunk (if it has any reads — skipped or not).
                # Pass-through-only tail (e.g. unmapped reads at end) still needs
                # to be flushed in order through the drain mechanism.
                if chunk_read_objs:
                    if len(inflight) >= max_inflight:
                        _drain_oldest_chunk(
                            inflight, outbam, with_scores, write_msps,
                            posterior_writer, counters
                        )

                    if chunk_reads:
                        future = executor.submit(
                            _process_payload_chunk_worker,
                            chunk_reads, edge_trim, circular, mode, context_size,
                            msp_min_size, nuc_min_size, with_scores, return_posteriors,
                            prob_threshold,
                        )
                    else:
                        # All-skipped tail: no worker call needed, fake a
                        # completed future returning [].
                        future = Future()
                        future.set_result([])
                    inflight.append((future, chunk_read_objs, chunk_reads, chunk_skip_flags))

                # Drain all remaining in-flight chunks (in order)
                while inflight:
                    _drain_oldest_chunk(
                        inflight, outbam, with_scores, write_msps,
                        posterior_writer, counters
                    )

            finally:
                try:
                    executor.shutdown(wait=True)
                finally:
                    if posterior_writer:
                        posterior_stats = posterior_writer.close()
                        posterior_writer = None

    elapsed = time.time() - start_time
    rate = total_reads / elapsed if elapsed > 0 else 0
    reads_with_footprints = counters['reads_with_footprints']
    print(f"\r  Processed: {total_reads:,} | Skipped: {skipped:,} | "
          f"With footprints: {reads_with_footprints:,} | {rate:.1f} reads/s", file=_log)
    if counters['worker_failures']:
        print(
            f"  Worker read failures: {counters['worker_failures']:,} "
            f"(passed through unchanged)",
            file=_log,
        )

    # Print skip reasons
    if skipped > 0:
        print("  Skip reasons:", file=_log)
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = 100 * count / (total_reads + skipped)
                print(f"    {reason}: {count:,} ({pct:.1f}%)", file=_log)

    # Sort and index output BAM (skip for stdout and unaligned mode)
    if output_bam != '-' and not process_unmapped:
        _sort_and_index_bam(output_bam, threads=n_cores)

    # Report posteriors after the writer has been closed by the processing finally block.
    if posterior_stats:
        n_fibers, file_size = posterior_stats
        print(f"Posteriors: {n_fibers:,} fibers -> {output_posteriors} ({file_size:.1f} MB)", file=_log)

    return total_reads, reads_with_footprints


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
    import time

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

    total_reads = 0
    reads_with_footprints = 0
    written = 0
    skipped = 0
    worker_failures = 0

    # Track skip reasons
    skip_reasons = {
        'unmapped': 0,
        'secondary_supplementary': 0,
        'low_mapq': 0,
        'too_short': 0,
        'training_excluded': 0,
        'no_modifications': 0,
        'extraction_failed': 0,
        'no_footprints': 0,
    }

    chunk_size = 2000  # Reads per chunk
    start_time = time.time()

    # Initialized after BAM handles open so failure cleanup is centralized in
    # the processing finally block.
    posterior_writer = None
    posterior_stats = None
    return_posteriors = False

    print("Processing and writing BAM (streaming)...")
    sys.stdout.flush()

    # Open input and output BAMs
    with pysam.AlignmentFile(input_bam, "rb", threads=io_threads, check_sq=False) as inbam:
        with pysam.AlignmentFile(output_bam, "wb", header=inbam.header, threads=io_threads) as outbam:

            if output_posteriors:
                if HAS_POSTERIOR_WRITER:
                    posterior_writer = PosteriorWriter(
                        output_posteriors, mode, context_size,
                        edge_trim, input_bam, batch_size=1000
                    )
                    return_posteriors = True
                    print(f"Posteriors will be written to: {output_posteriors}")
                else:
                    print("WARNING: posterior_writer.py not found, skipping posteriors export")

            chunk_reads = []  # Buffer for current chunk
            chunk_read_objs = []  # Corresponding pysam read objects

            if n_cores > 1 and model_path:
                # Parallel processing with ProcessPoolExecutor
                executor = ProcessPoolExecutor(
                    max_workers=n_cores,
                    mp_context=_MP_CONTEXT,
                    initializer=_init_bam_worker,
                    initargs=(model_path, debug_timing)
                )
            else:
                executor = None

            try:
                for read in inbam:
                    # Pass through unmapped reads (no sequence to process)
                    if read.is_unmapped:
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['unmapped'] += 1
                        continue

                    # Skip secondary/supplementary if primary_only mode
                    if primary_only and (read.is_secondary or read.is_supplementary):
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['secondary_supplementary'] += 1
                        continue

                    # Check filters (apply to all alignments)
                    if read.mapping_quality < min_mapq:
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['low_mapq'] += 1
                        continue

                    if read.query_alignment_length < min_read_length:
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['too_short'] += 1
                        continue

                    read_id = read.query_name
                    if train_rids and read_id in train_rids:
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['training_excluded'] += 1
                        continue

                    # Extract data needed for processing
                    try:
                        fiber_read = _extract_fiber_read_from_pysam(read, mode, prob_threshold)
                        if fiber_read is None:
                            outbam.write(read)
                            written += 1
                            skipped += 1
                            skip_reasons['no_modifications'] += 1
                            continue
                    except Exception:
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['extraction_failed'] += 1
                        continue

                    chunk_reads.append(fiber_read)
                    chunk_read_objs.append(read)
                    total_reads += 1

                    # Check max reads limit
                    if max_reads and total_reads >= max_reads:
                        # Process final chunk and exit
                        break

                    # Process chunk when full
                    if len(chunk_reads) >= chunk_size:
                        n_fp, n_nofp, n_failed, chunk_results = _process_and_write_chunk(
                            chunk_reads, chunk_read_objs, outbam,
                            model, executor, edge_trim, circular,
                            mode, context_size, msp_min_size, nuc_min_size=nuc_min_size,
                            with_scores=with_scores,
                            return_posteriors=return_posteriors,
                            write_msps=write_msps
                        )
                        reads_with_footprints += n_fp
                        skip_reasons['no_footprints'] += n_nofp
                        worker_failures += n_failed
                        written += len(chunk_read_objs)

                        # Write posteriors if requested
                        if posterior_writer and chunk_results:
                            for read_obj, fiber_read_data, result in chunk_results:
                                if result and result.get('posteriors') is not None:
                                    chrom = read_obj.reference_name
                                    if chrom:
                                        # Get ref_positions from the read object
                                        ref_positions = get_ref_positions_from_read(read_obj) if HAS_POSTERIOR_WRITER else np.array([], dtype=np.int32)
                                        posterior_writer.add_fiber(chrom, {
                                            'read_name': read_obj.query_name,
                                            'ref_start': read_obj.reference_start,
                                            'ref_end': read_obj.reference_end,
                                            'strand': result.get('strand', '.'),
                                            'posteriors': result['posteriors'],
                                            'ref_positions': ref_positions,
                                            'footprint_starts': result['ns'],
                                            'footprint_sizes': result['nl'],
                                        })

                        # Progress update
                        elapsed = time.time() - start_time
                        rate = total_reads / elapsed if elapsed > 0 else 0
                        print(f"\r  Processed: {total_reads:,} | Skipped: {skipped:,} | {rate:.1f} reads/s", end='')
                        sys.stdout.flush()

                        chunk_reads = []
                        chunk_read_objs = []

                # Process final partial chunk
                if chunk_reads:
                    n_fp, n_nofp, n_failed, chunk_results = _process_and_write_chunk(
                        chunk_reads, chunk_read_objs, outbam,
                        model, executor, edge_trim, circular,
                        mode, context_size, msp_min_size, nuc_min_size=nuc_min_size,
                        with_scores=with_scores,
                        return_posteriors=return_posteriors,
                        write_msps=write_msps,
                    )
                    reads_with_footprints += n_fp
                    skip_reasons['no_footprints'] += n_nofp
                    worker_failures += n_failed
                    written += len(chunk_read_objs)

                    # Write posteriors if requested
                    if posterior_writer and chunk_results:
                        for read_obj, fiber_read_data, result in chunk_results:
                            if result and result.get('posteriors') is not None:
                                chrom = read_obj.reference_name
                                if chrom:
                                    ref_positions = get_ref_positions_from_read(read_obj) if HAS_POSTERIOR_WRITER else np.array([], dtype=np.int32)
                                    posterior_writer.add_fiber(chrom, {
                                        'read_name': read_obj.query_name,
                                        'ref_start': read_obj.reference_start,
                                        'ref_end': read_obj.reference_end,
                                        'strand': result.get('strand', '.'),
                                        'posteriors': result['posteriors'],
                                        'ref_positions': ref_positions,
                                        'footprint_starts': result['ns'],
                                        'footprint_sizes': result['nl'],
                                    })

            finally:
                try:
                    if executor:
                        executor.shutdown(wait=True)
                finally:
                    if posterior_writer:
                        posterior_stats = posterior_writer.close()
                        posterior_writer = None

    elapsed = time.time() - start_time
    rate = total_reads / elapsed if elapsed > 0 else 0
    print(f"\r  Processed: {total_reads:,} | Skipped: {skipped:,} | With footprints: {reads_with_footprints:,} | {rate:.1f} reads/s")
    if worker_failures:
        print(f"  Worker read failures: {worker_failures:,} (passed through unchanged)")

    # Print skip reasons summary
    if skipped > 0:
        print("  Skip reasons:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = 100 * count / (total_reads + skipped)
                print(f"    {reason}: {count:,} ({pct:.1f}%)")

    # Index the output BAM (sort first if needed)
    _sort_and_index_bam(output_bam, threads=n_cores)

    # Report posteriors after the writer has been closed by the processing finally block.
    if posterior_stats:
        n_fibers, file_size = posterior_stats
        print(f"Posteriors: {n_fibers:,} fibers -> {output_posteriors} ({file_size:.1f} MB)")

    return total_reads, reads_with_footprints
