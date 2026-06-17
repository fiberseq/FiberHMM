"""Region-parallel BAM and BED pipeline coordinators."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Set, Tuple

import pysam

from fiberhmm.io.bam_index import ensure_bam_index
from fiberhmm.inference.bam_output import (
    _concatenate_region_bams,
    _sort_and_index_bam,
)
from fiberhmm.inference.mp_context import _MP_CONTEXT
from fiberhmm.inference.region_planning import _get_genome_regions
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
from fiberhmm.posteriors.region_tsv import (
    merge_region_posteriors_tsv as _merge_region_posteriors_tsv,
    region_posteriors_needs_h5_conversion,
    region_posteriors_tsv_output_path,
)


def _base_region_worker_params(
    *,
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
    train_rids,
    primary_only: bool,
) -> dict:
    return {
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
    }


def _print_skip_reasons_summary(
    aggregation: RegionBamAggregation,
    footprint_label: str = "With footprints",
) -> None:
    if aggregation.total_skipped <= 0:
        return

    total_encountered = aggregation.total_reads + aggregation.total_skipped
    print(
        f"  Processed: {aggregation.total_reads:,} | "
        f"Skipped: {aggregation.total_skipped:,} | "
        f"{footprint_label}: {aggregation.reads_with_footprints:,}"
    )
    print("  Skip reasons:")
    for reason, count in sorted(
        aggregation.skip_reasons.items(), key=lambda x: -x[1]
    ):
        if count > 0:
            pct = 100 * count / total_encountered
            print(f"    {reason}: {count:,} ({pct:.1f}%)")


def _report_workers_ready_once(
    first_result_time: Optional[float],
    pool_start: float,
    message: str,
) -> float:
    if first_result_time is not None:
        return first_result_time

    first_result_time = time.time()
    init_time = first_result_time - pool_start
    print(f"  Workers ready ({init_time:.1f}s). {message}")
    sys.stdout.flush()
    return first_result_time


def _read_rate(total_reads: int, elapsed_seconds: float) -> float:
    return total_reads / elapsed_seconds if elapsed_seconds > 0 else 0


def _print_region_progress(
    aggregation,
    total_regions: int,
    start_time: float,
    *,
    footprint_label: str = "With footprints",
    rate_unit: str = "reads/s",
    rate_precision: int = 1,
) -> None:
    elapsed = time.time() - start_time
    rate = _read_rate(aggregation.total_reads, elapsed)
    rate_text = f"{rate:.{rate_precision}f}"
    print(
        f"\r  Regions: {aggregation.completed}/{total_regions} | "
        f"Reads: {aggregation.total_reads:,} | "
        f"{footprint_label}: {aggregation.reads_with_footprints:,} | "
        f"{rate_text} {rate_unit}",
        end='',
    )
    sys.stdout.flush()


def _region_temp_path(temp_dir: str, index: int, suffix: str) -> str:
    return os.path.join(temp_dir, f'region_{index:06d}.{suffix}')


def _region_bam_work_items(
    regions,
    input_bam: str,
    temp_dir: str,
    include_tsv: bool = False,
) -> list[RegionBamWorkItem]:
    work_items = []
    for i, region in enumerate(regions):
        temp_bam = _region_temp_path(temp_dir, i, 'bam')
        temp_tsv = _region_temp_path(temp_dir, i, 'tsv') if include_tsv else None
        work_items.append(
            RegionBamWorkItem((region[0], region[1], region[2]), input_bam, temp_bam, temp_tsv)
        )
    return work_items


def _region_bed_work_items(regions, input_bam: str, temp_dir: str) -> list[RegionBedWorkItem]:
    return [
        RegionBedWorkItem(region, input_bam, _region_temp_path(temp_dir, i, 'bed'))
        for i, region in enumerate(regions)
    ]


def _ordered_existing_temp_paths(indexed_paths) -> list:
    return [
        path
        for _, path in sorted(indexed_paths, key=lambda x: x[0])
        if os.path.exists(path) and os.path.getsize(path) > 0
    ]


def _submit_region_futures(executor, worker, work_items) -> dict:
    return {
        executor.submit(worker, item): i
        for i, item in enumerate(work_items)
    }


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
    start_time = time.time()
    return_posteriors = output_posteriors is not None

    ensure_bam_index(input_bam, "Indexing input BAM for region-parallel processing...")

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
        params = _base_region_worker_params(
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
            train_rids=train_rids,
            primary_only=primary_only,
        )
        params.update({
            'return_posteriors': return_posteriors,
            'write_msps': write_msps,
            'io_threads': io_threads,
        })

        # Work items - include temp TSV path if posteriors requested
        work_items = _region_bam_work_items(
            regions, input_bam, temp_dir, include_tsv=return_posteriors,
        )

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
            futures = _submit_region_futures(
                executor,
                _process_region_to_bam,
                work_items,
            )

            for future in as_completed(futures):
                try:
                    result = RegionBamResult.from_value(future.result())
                    include_tsv = bool(
                        result.temp_tsv_path and os.path.exists(result.temp_tsv_path)
                    )
                    aggregation.add_result(futures[future], result, include_tsv=include_tsv)

                    # Track first result
                    first_result_time = _report_workers_ready_once(
                        first_result_time,
                        pool_start,
                        "Processing regions...",
                    )
                    _print_region_progress(aggregation, len(regions), start_time)

                except Exception as e:
                    print(f"\nError processing region: {e}")
                    raise

        print()  # Newline after progress

        # Print skip reasons summary
        _print_skip_reasons_summary(aggregation)

        # Sort temp BAMs by region order and filter to non-empty
        non_empty_bams = _ordered_existing_temp_paths(aggregation.temp_bams)

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
                if region_posteriors_needs_h5_conversion(output_posteriors):
                    print(
                        "  To convert to H5: python posteriors_io.py "
                        f"tsv2h5 {tsv_path} {output_posteriors}"
                    )

        elapsed = time.time() - start_time
        rate = _read_rate(aggregation.total_reads, elapsed)
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
    start_time = time.time()

    ensure_bam_index(input_bam, "Indexing input BAM for region-parallel processing...")

    # Get regions
    regions = _get_genome_regions(input_bam, region_size, skip_scaffolds, chroms)
    print(f"Processing {len(regions)} regions with {n_cores} cores (BED output)...")
    sys.stdout.flush()

    # Create temp directory for BED files (small compared to BAMs)
    output_dir = os.path.dirname(os.path.abspath(output_bed))
    temp_dir = tempfile.mkdtemp(prefix='.fiberhmm_bed_tmp_', dir=output_dir)

    try:
        params = _base_region_worker_params(
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
            train_rids=train_rids,
            primary_only=primary_only,
        )

        # Work items - write temp BED files
        work_items = _region_bed_work_items(regions, input_bam, temp_dir)

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
            futures = _submit_region_futures(
                executor,
                _process_region_to_bed,
                work_items,
            )

            for future in as_completed(futures):
                try:
                    result = RegionBedResult.from_value(future.result())
                    aggregation.add_result(futures[future], result)

                    # Track first result
                    first_result_time = _report_workers_ready_once(
                        first_result_time,
                        pool_start,
                        "Processing regions...",
                    )
                    _print_region_progress(aggregation, len(regions), start_time)

                except Exception as e:
                    print(f"\nError processing region: {e}")
                    raise

        print()  # Newline after progress

        # Sort temp BEDs by region order and concatenate
        non_empty_beds = _ordered_existing_temp_paths(aggregation.temp_beds)

        print(f"Concatenating {len(non_empty_beds)} region BED files...")
        sys.stdout.flush()

        with open(output_bed, 'wb') as fout:
            for bed_path in non_empty_beds:
                with open(bed_path, 'rb') as fin:
                    shutil.copyfileobj(fin, fout)

        elapsed = time.time() - start_time
        rate = _read_rate(aggregation.total_reads, elapsed)
        print(
            f"Completed: {aggregation.total_reads:,} reads | "
            f"{aggregation.reads_with_footprints:,} with footprints | "
            f"{rate:.1f} reads/s | {elapsed:.1f}s"
        )

        return aggregation.total_reads, aggregation.reads_with_footprints

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


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
    recall_nucs: bool = False,
    split_min_llr: float = 4.0,
    split_min_opps: int = 3,
    filter_chimeras: bool = True,
    chimera_min_seg: int = 5,
    chimera_purity: float = 0.8,
    phase_nrl: int = 0,
    pg_record: dict = None,
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

    params = _base_region_worker_params(
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
        train_rids=train_rids,
        primary_only=primary_only,
    )
    params.update({
        'io_threads': io_threads,
        'min_llr': min_llr, 'min_opps': min_opps,
        'unify_threshold': unify_threshold,
        'also_write_legacy': also_write_legacy,
        'downstream_compat': downstream_compat,
        'recall_nucs': recall_nucs,
        'split_min_llr': split_min_llr,
        'split_min_opps': split_min_opps,
        'filter_chimeras': filter_chimeras,
        'chimera_min_seg': chimera_min_seg,
        'chimera_purity': chimera_purity,
        'phase_nrl': phase_nrl,
        'pg_record': pg_record,
        # Path string, NOT an open handle: pysam.FastaFile is not fork-safe,
        # so each worker opens it lazily in _init_fused_region_worker.
        'ref_fasta_path': ref_fasta_path,
    })

    work_items = _region_bam_work_items(regions, input_bam, temp_dir)

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
            futures = _submit_region_futures(
                executor,
                _process_region_to_bam_fused,
                work_items,
            )

            for future in as_completed(futures):
                result = RegionBamResult.from_value(future.result())
                aggregation.add_result(futures[future], result)
                first_result = _report_workers_ready_once(
                    first_result,
                    pool_start,
                    "Processing...",
                )
                _print_region_progress(
                    aggregation,
                    len(regions),
                    start_time,
                    footprint_label="With FP",
                    rate_unit="r/s",
                    rate_precision=0,
                )
        print()

        _print_skip_reasons_summary(aggregation, footprint_label="With FP")

        # Concat region BAMs in region-index order - preserves coord sort.
        non_empty = _ordered_existing_temp_paths(aggregation.temp_bams)
        _concatenate_region_bams(input_bam, output_bam, non_empty, temp_dir)

        # Index directly (input sorted -> each region sorted -> concat sorted).
        try:
            pysam.index(output_bam)
        except pysam.SamtoolsError:
            pass

        elapsed = time.time() - start_time
        rate = _read_rate(aggregation.total_reads, elapsed)
        print(
            f"  Total: {aggregation.total_reads:,} reads, "
            f"{aggregation.reads_with_footprints:,} with footprints, "
            f"{rate:.1f} r/s"
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return aggregation.total_reads, aggregation.reads_with_footprints
