"""Region-parallel BAM and BED pipeline coordinators."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Set, Tuple

import pysam

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
from fiberhmm.inference.skip_reasons import iter_nonzero_skip_reasons
from fiberhmm.io.bam_index import ensure_bam_index
from fiberhmm.io.path_status import (
    path_is_nonempty_file,
    path_is_regular_file,
    path_size_gb,
    path_size_mb,
)
from fiberhmm.posteriors.region_tsv import (
    merge_region_posteriors_tsv as _merge_region_posteriors_tsv,
)
from fiberhmm.posteriors.region_tsv import (
    region_posteriors_needs_h5_conversion,
    region_posteriors_tsv_output_path,
)


@dataclass(frozen=True)
class _RegionParallelRun:
    regions: list
    temp_dir: str


@dataclass(frozen=True)
class _RegionCompletionResult:
    total_reads: int
    reads_with_footprints: int

    def as_tuple(self) -> Tuple[int, int]:
        return self.total_reads, self.reads_with_footprints


@dataclass(frozen=True)
class _RegionProgressConfig:
    footprint_label: str = "With footprints"
    rate_unit: str = "reads/s"
    rate_precision: int = 1


@dataclass(frozen=True)
class _RegionResultCollectionRequest:
    executor: object
    worker: object
    work_items: object
    aggregation: object
    result_type: object
    total_regions: int
    start_time: float
    pool_start: float
    ready_message: str
    include_tsv: bool
    progress_config: Optional[_RegionProgressConfig]
    error_prefix: Optional[str]


@dataclass(frozen=True)
class _RegionWorkerPoolRequest:
    n_cores: int
    initializer: object
    initargs: tuple
    worker: object
    work_items: object
    aggregation: object
    result_type: object
    total_regions: int
    start_time: float
    init_message: str
    ready_message: str
    include_tsv: bool
    progress_config: Optional[_RegionProgressConfig]
    error_prefix: Optional[str]


@dataclass(frozen=True)
class _RegionBamWorkerPoolRequest:
    n_cores: int
    model_path: str
    params: dict
    work_items: object
    total_regions: int
    start_time: float
    include_tsv: bool


@dataclass(frozen=True)
class _RegionBedWorkerPoolRequest:
    n_cores: int
    model_path: str
    params: dict
    work_items: object
    total_regions: int
    start_time: float


@dataclass(frozen=True)
class _FusedRegionBamWorkerPoolRequest:
    n_cores: int
    apply_model_path: str
    recall_model_path: Optional[str]
    emission_uplift: float
    params: dict
    work_items: object
    total_regions: int
    start_time: float


@dataclass(frozen=True)
class _RegionBedPipelineRequest:
    input_bam: str
    output_bed: str
    model_path: str
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
    region_size: int
    skip_scaffolds: bool
    chroms: Optional[Set[str]]
    primary_only: bool


@dataclass(frozen=True)
class _RegionBamPipelineRequest:
    input_bam: str
    output_bam: str
    model_path: str
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
    region_size: int
    skip_scaffolds: bool
    chroms: Optional[Set[str]]
    primary_only: bool
    output_posteriors: Optional[str]
    write_msps: bool
    io_threads: int


@dataclass(frozen=True)
class _FusedRegionBamPipelineRequest:
    input_bam: str
    output_bam: str
    apply_model_path: str
    recall_model_path: Optional[str]
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
    min_llr: float
    min_opps: int
    unify_threshold: int
    emission_uplift: float
    also_write_legacy: bool
    downstream_compat: bool
    n_cores: int
    region_size: int
    skip_scaffolds: bool
    chroms: Optional[Set[str]]
    io_threads: int
    primary_only: bool
    ref_fasta_path: Optional[str]
    recall_nucs: bool
    split_min_llr: float
    split_min_opps: int
    filter_chimeras: bool
    chimera_min_seg: int
    chimera_purity: float
    phase_nrl: int
    nuc_profile_path: Optional[str]
    pg_record: Optional[dict]


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


def _region_bam_worker_params(
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
    return_posteriors: bool,
    write_msps: bool,
    io_threads: int,
) -> dict:
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
    return params


def _fused_region_worker_params(
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
    io_threads: int,
    min_llr: float,
    min_opps: int,
    unify_threshold: int,
    also_write_legacy: bool,
    downstream_compat: bool,
    recall_nucs: bool,
    split_min_llr: float,
    split_min_opps: int,
    filter_chimeras: bool,
    chimera_min_seg: int,
    chimera_purity: float,
    phase_nrl: int,
    nuc_profile_path: Optional[str],
    pg_record: dict,
    ref_fasta_path: Optional[str],
) -> dict:
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
        'min_llr': min_llr,
        'min_opps': min_opps,
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
        'nuc_profile_path': nuc_profile_path,
        'pg_record': pg_record,
        # Path string, NOT an open handle: pysam.FastaFile is not fork-safe,
        # so each worker opens it lazily in _init_fused_region_worker.
        'ref_fasta_path': ref_fasta_path,
    })
    return params


def _region_bam_worker_params_from_request(
    request: _RegionBamPipelineRequest,
    return_posteriors: bool,
) -> dict:
    return _region_bam_worker_params(
        edge_trim=request.edge_trim,
        circular=request.circular,
        mode=request.mode,
        context_size=request.context_size,
        msp_min_size=request.msp_min_size,
        nuc_min_size=request.nuc_min_size,
        min_mapq=request.min_mapq,
        prob_threshold=request.prob_threshold,
        min_read_length=request.min_read_length,
        with_scores=request.with_scores,
        train_rids=request.train_rids,
        primary_only=request.primary_only,
        return_posteriors=return_posteriors,
        write_msps=request.write_msps,
        io_threads=request.io_threads,
    )


def _fused_region_worker_params_from_request(
    request: _FusedRegionBamPipelineRequest,
) -> dict:
    return _fused_region_worker_params(
        edge_trim=request.edge_trim,
        circular=request.circular,
        mode=request.mode,
        context_size=request.context_size,
        msp_min_size=request.msp_min_size,
        nuc_min_size=request.nuc_min_size,
        min_mapq=request.min_mapq,
        prob_threshold=request.prob_threshold,
        min_read_length=request.min_read_length,
        with_scores=request.with_scores,
        train_rids=request.train_rids,
        primary_only=request.primary_only,
        io_threads=request.io_threads,
        min_llr=request.min_llr,
        min_opps=request.min_opps,
        unify_threshold=request.unify_threshold,
        also_write_legacy=request.also_write_legacy,
        downstream_compat=request.downstream_compat,
        recall_nucs=request.recall_nucs,
        split_min_llr=request.split_min_llr,
        split_min_opps=request.split_min_opps,
        filter_chimeras=request.filter_chimeras,
        chimera_min_seg=request.chimera_min_seg,
        chimera_purity=request.chimera_purity,
        phase_nrl=request.phase_nrl,
        nuc_profile_path=request.nuc_profile_path,
        pg_record=request.pg_record,
        ref_fasta_path=request.ref_fasta_path,
    )


def _prepare_region_bam_parallel_run_from_request(
    request: _RegionBamPipelineRequest,
    return_posteriors: bool,
) -> _RegionParallelRun:
    return _prepare_region_parallel_run(
        request.input_bam,
        request.output_bam,
        request.region_size,
        request.skip_scaffolds,
        request.chroms,
        request.n_cores,
        temp_prefix='.fiberhmm_tmp_',
        output_posteriors=(
            request.output_posteriors if return_posteriors else None
        ),
    )


def _prepare_fused_region_bam_parallel_run_from_request(
    request: _FusedRegionBamPipelineRequest,
) -> _RegionParallelRun:
    return _prepare_region_parallel_run(
        request.input_bam,
        request.output_bam,
        request.region_size,
        request.skip_scaffolds,
        request.chroms,
        request.n_cores,
        temp_prefix='.fiberhmm_call_tmp_',
        output_label=" (fused apply+recall)",
        ensure_index=False,
    )


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
    for reason, count in iter_nonzero_skip_reasons(aggregation.skip_reasons):
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


def _region_completion_summary(total_reads: int, reads_with_footprints: int, elapsed: float) -> str:
    rate = _read_rate(total_reads, elapsed)
    return (
        f"Completed: {total_reads:,} reads | "
        f"{reads_with_footprints:,} with footprints | "
        f"{rate:.1f} reads/s | {elapsed:.1f}s"
    )


def _region_temp_path(temp_dir: str, index: int, suffix: str) -> str:
    return os.path.join(temp_dir, f'region_{index:06d}.{suffix}')


def _region_bam_work_item(
    region,
    input_bam: str,
    temp_dir: str,
    index: int,
    include_tsv: bool = False,
) -> RegionBamWorkItem:
    temp_bam = _region_temp_path(temp_dir, index, 'bam')
    temp_tsv = _region_temp_path(temp_dir, index, 'tsv') if include_tsv else None
    return RegionBamWorkItem(
        (region[0], region[1], region[2]),
        input_bam,
        temp_bam,
        temp_tsv,
    )


def _region_bam_work_items(
    regions,
    input_bam: str,
    temp_dir: str,
    include_tsv: bool = False,
) -> list[RegionBamWorkItem]:
    return [
        _region_bam_work_item(region, input_bam, temp_dir, i, include_tsv)
        for i, region in enumerate(regions)
    ]


def _region_bed_work_item(
    region,
    input_bam: str,
    temp_dir: str,
    index: int,
) -> RegionBedWorkItem:
    return RegionBedWorkItem(
        (region[0], region[1], region[2]),
        input_bam,
        _region_temp_path(temp_dir, index, 'bed'),
    )


def _region_bed_work_items(regions, input_bam: str, temp_dir: str) -> list[RegionBedWorkItem]:
    return [
        _region_bed_work_item(region, input_bam, temp_dir, i)
        for i, region in enumerate(regions)
    ]


def _ordered_existing_temp_paths(indexed_paths) -> list:
    return [
        path
        for _, path in sorted(indexed_paths, key=lambda x: x[0])
        if path_is_nonempty_file(path)
    ]


def _region_result_has_existing_tsv(result: RegionBamResult) -> bool:
    return bool(result.temp_tsv_path and path_is_nonempty_file(result.temp_tsv_path))


def _submit_region_futures(executor, worker, work_items) -> dict:
    return {
        executor.submit(worker, item): i
        for i, item in enumerate(work_items)
    }


def _add_region_result_to_aggregation(
    aggregation,
    region_index: int,
    result,
    *,
    include_tsv: bool = False,
) -> None:
    if include_tsv:
        aggregation.add_result(
            region_index,
            result,
            include_tsv=_region_result_has_existing_tsv(result),
        )
        return

    aggregation.add_result(region_index, result)


def _collect_region_results(
    executor,
    worker,
    work_items,
    aggregation,
    result_type,
    total_regions: int,
    start_time: float,
    pool_start: float,
    ready_message: str,
    *,
    include_tsv: bool = False,
    progress_config: Optional[_RegionProgressConfig] = None,
    error_prefix: Optional[str] = None,
) -> None:
    _collect_region_results_from_request(
        _RegionResultCollectionRequest(
            executor=executor,
            worker=worker,
            work_items=work_items,
            aggregation=aggregation,
            result_type=result_type,
            total_regions=total_regions,
            start_time=start_time,
            pool_start=pool_start,
            ready_message=ready_message,
            include_tsv=include_tsv,
            progress_config=progress_config,
            error_prefix=error_prefix,
        )
    )


def _collect_region_results_from_request(
    request: _RegionResultCollectionRequest,
) -> None:
    futures = _submit_region_futures(
        request.executor,
        request.worker,
        request.work_items,
    )
    first_result_time = None
    progress_config = request.progress_config or _RegionProgressConfig()

    for future in as_completed(futures):
        try:
            result = request.result_type.from_value(future.result())
            _add_region_result_to_aggregation(
                request.aggregation,
                futures[future],
                result,
                include_tsv=request.include_tsv,
            )

            first_result_time = _report_workers_ready_once(
                first_result_time,
                request.pool_start,
                request.ready_message,
            )
            _print_region_progress(
                request.aggregation,
                request.total_regions,
                request.start_time,
                footprint_label=progress_config.footprint_label,
                rate_unit=progress_config.rate_unit,
                rate_precision=progress_config.rate_precision,
            )
        except Exception as e:
            if request.error_prefix:
                print(f"\n{request.error_prefix}: {e}")
            raise


def _run_region_worker_pool(
    *,
    n_cores: int,
    initializer,
    initargs: tuple,
    worker,
    work_items,
    aggregation,
    result_type,
    total_regions: int,
    start_time: float,
    init_message: str,
    ready_message: str,
    include_tsv: bool = False,
    progress_config: Optional[_RegionProgressConfig] = None,
    error_prefix: Optional[str] = None,
) -> None:
    _run_region_worker_pool_from_request(
        _RegionWorkerPoolRequest(
            n_cores=n_cores,
            initializer=initializer,
            initargs=initargs,
            worker=worker,
            work_items=work_items,
            aggregation=aggregation,
            result_type=result_type,
            total_regions=total_regions,
            start_time=start_time,
            init_message=init_message,
            ready_message=ready_message,
            include_tsv=include_tsv,
            progress_config=progress_config,
            error_prefix=error_prefix,
        )
    )


def _run_region_worker_pool_from_request(
    request: _RegionWorkerPoolRequest,
) -> None:
    print(request.init_message)
    sys.stdout.flush()
    pool_start = time.time()

    with ProcessPoolExecutor(
        max_workers=request.n_cores,
        mp_context=_MP_CONTEXT,
        initializer=request.initializer,
        initargs=request.initargs,
    ) as executor:
        _collect_region_results(
            executor,
            request.worker,
            request.work_items,
            request.aggregation,
            request.result_type,
            request.total_regions,
            request.start_time,
            pool_start,
            request.ready_message,
            include_tsv=request.include_tsv,
            progress_config=request.progress_config,
            error_prefix=request.error_prefix,
        )


def _make_output_temp_dir(output_path: str, prefix: str) -> str:
    output_dir = os.path.dirname(os.path.abspath(output_path))
    return tempfile.mkdtemp(prefix=prefix, dir=output_dir)


def _prepare_region_parallel_run(
    input_bam: str,
    output_path: str,
    region_size: int,
    skip_scaffolds: bool,
    chroms: Optional[Set[str]],
    n_cores: int,
    *,
    temp_prefix: str,
    output_label: str = "",
    ensure_index: bool = True,
    output_posteriors: Optional[str] = None,
):
    if ensure_index:
        ensure_bam_index(
            input_bam,
            "Indexing input BAM for region-parallel processing...",
        )

    regions = _get_genome_regions(input_bam, region_size, skip_scaffolds, chroms)
    print(f"Processing {len(regions)} regions with {n_cores} cores{output_label}...")
    if output_posteriors is not None:
        print(f"Posteriors will be written to: {output_posteriors}")
    sys.stdout.flush()

    return _RegionParallelRun(
        regions=regions,
        temp_dir=_make_output_temp_dir(output_path, temp_prefix),
    )


def _region_completion_result(aggregation, start_time: float) -> _RegionCompletionResult:
    elapsed = time.time() - start_time
    print(_region_completion_summary(
        aggregation.total_reads,
        aggregation.reads_with_footprints,
        elapsed,
    ))
    return _RegionCompletionResult(
        total_reads=aggregation.total_reads,
        reads_with_footprints=aggregation.reads_with_footprints,
    )


def _merge_region_posterior_outputs(
    temp_tsvs,
    output_posteriors: str,
    mode: str,
    context_size: int,
    edge_trim: int,
    input_bam: str,
) -> None:
    if not temp_tsvs:
        return

    print(f"Merging {len(temp_tsvs)} posterior files...")
    merge_start = time.time()
    n_fibers = _merge_region_posteriors_tsv(
        temp_tsvs, output_posteriors,
        mode, context_size, edge_trim, input_bam,
    )
    merge_time = time.time() - merge_start

    # Region posterior export always materializes a compressed TSV first.
    tsv_path = region_posteriors_tsv_output_path(output_posteriors)
    if not path_is_regular_file(tsv_path):
        return

    file_size = path_size_mb(tsv_path)
    print(
        f"Posteriors: {n_fibers:,} fibers -> {tsv_path} "
        f"({file_size:.1f} MB, {merge_time:.1f}s)"
    )
    if region_posteriors_needs_h5_conversion(output_posteriors):
        print(
            "  To convert to H5: python posteriors_io.py "
            f"tsv2h5 {tsv_path} {output_posteriors}"
        )


def _finalize_region_bam_output(
    input_bam: str,
    output_bam: str,
    non_empty_bams: list[str],
    temp_dir: str,
    n_cores: int,
) -> None:
    _concatenate_region_bams(input_bam, output_bam, non_empty_bams, temp_dir)
    sys.stdout.flush()

    if path_is_regular_file(output_bam):
        output_size_gb = path_size_gb(output_bam)
        print(f"Output BAM: {output_size_gb:.2f}GB")

    print("Step: Index/Sort...")
    _sort_and_index_bam(output_bam, threads=n_cores)


def _fused_region_total_summary(
    aggregation: RegionBamAggregation,
    start_time: float,
) -> str:
    elapsed = time.time() - start_time
    rate = _read_rate(aggregation.total_reads, elapsed)
    return (
        f"  Total: {aggregation.total_reads:,} reads, "
        f"{aggregation.reads_with_footprints:,} with footprints, "
        f"{rate:.1f} r/s"
    )


def _finalize_fused_region_bam_output(
    input_bam: str,
    output_bam: str,
    temp_dir: str,
    aggregation: RegionBamAggregation,
    start_time: float,
) -> None:
    _print_skip_reasons_summary(aggregation, footprint_label="With FP")

    # Concat region BAMs in region-index order - preserves coord sort.
    non_empty = _ordered_existing_temp_paths(aggregation.temp_bams)
    _concatenate_region_bams(input_bam, output_bam, non_empty, temp_dir)

    # Index directly (input sorted -> each region sorted -> concat sorted).
    try:
        pysam.index(output_bam)
    except pysam.SamtoolsError:
        pass

    print(_fused_region_total_summary(aggregation, start_time))


def _concatenate_region_beds(output_bed: str, non_empty_beds: list[str]) -> None:
    print(f"Concatenating {len(non_empty_beds)} region BED files...")
    sys.stdout.flush()

    temp_output = output_bed + ".tmp"
    try:
        with open(temp_output, 'wb') as fout:
            for bed_path in non_empty_beds:
                with open(bed_path, 'rb') as fin:
                    shutil.copyfileobj(fin, fout)
        os.replace(temp_output, output_bed)
    except BaseException:
        if os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except OSError as e:
                print(f"Warning: Could not remove temporary BED {temp_output}: {e}")
        raise


def _finalize_region_bam_parallel_run(
    input_bam: str,
    output_bam: str,
    temp_dir: str,
    aggregation: RegionBamAggregation,
    start_time: float,
    n_cores: int,
    return_posteriors: bool,
    output_posteriors: Optional[str],
    mode: str,
    context_size: int,
    edge_trim: int,
) -> Tuple[int, int]:
    print()
    _print_skip_reasons_summary(aggregation)

    non_empty_bams = _ordered_existing_temp_paths(aggregation.temp_bams)
    _finalize_region_bam_output(
        input_bam, output_bam, non_empty_bams, temp_dir, n_cores,
    )

    if return_posteriors and aggregation.temp_tsvs:
        _merge_region_posterior_outputs(
            aggregation.temp_tsvs,
            output_posteriors,
            mode,
            context_size,
            edge_trim,
            input_bam,
        )

    return _region_completion_result(aggregation, start_time).as_tuple()


def _finalize_region_bed_parallel_run(
    output_bed: str,
    aggregation: RegionBedAggregation,
    start_time: float,
) -> Tuple[int, int]:
    print()
    non_empty_beds = _ordered_existing_temp_paths(aggregation.temp_beds)
    _concatenate_region_beds(output_bed, non_empty_beds)
    return _region_completion_result(aggregation, start_time).as_tuple()


def _run_region_bam_worker_pool(
    *,
    n_cores: int,
    model_path: str,
    params: dict,
    work_items,
    total_regions: int,
    start_time: float,
    include_tsv: bool = True,
) -> RegionBamAggregation:
    return _run_region_bam_worker_pool_from_request(
        _RegionBamWorkerPoolRequest(
            n_cores=n_cores,
            model_path=model_path,
            params=params,
            work_items=work_items,
            total_regions=total_regions,
            start_time=start_time,
            include_tsv=include_tsv,
        )
    )


def _run_region_bam_worker_pool_from_request(
    request: _RegionBamWorkerPoolRequest,
) -> RegionBamAggregation:
    aggregation = RegionBamAggregation()
    _run_region_worker_pool(
        n_cores=request.n_cores,
        initializer=_init_region_worker,
        initargs=(request.model_path, request.params),
        worker=_process_region_to_bam,
        work_items=request.work_items,
        aggregation=aggregation,
        result_type=RegionBamResult,
        total_regions=request.total_regions,
        start_time=request.start_time,
        init_message=(
            f"  Initializing {request.n_cores} worker processes "
            "(loading HMM model in each)..."
        ),
        ready_message="Processing regions...",
        include_tsv=request.include_tsv,
        error_prefix="Error processing region",
    )
    return aggregation


def _run_region_bed_worker_pool(
    *,
    n_cores: int,
    model_path: str,
    params: dict,
    work_items,
    total_regions: int,
    start_time: float,
) -> RegionBedAggregation:
    return _run_region_bed_worker_pool_from_request(
        _RegionBedWorkerPoolRequest(
            n_cores=n_cores,
            model_path=model_path,
            params=params,
            work_items=work_items,
            total_regions=total_regions,
            start_time=start_time,
        )
    )


def _run_region_bed_worker_pool_from_request(
    request: _RegionBedWorkerPoolRequest,
) -> RegionBedAggregation:
    aggregation = RegionBedAggregation()
    _run_region_worker_pool(
        n_cores=request.n_cores,
        initializer=_init_region_worker,
        initargs=(request.model_path, request.params),
        worker=_process_region_to_bed,
        work_items=request.work_items,
        aggregation=aggregation,
        result_type=RegionBedResult,
        total_regions=request.total_regions,
        start_time=request.start_time,
        init_message=(
            f"  Initializing {request.n_cores} worker processes "
            "(loading HMM model in each)..."
        ),
        ready_message="Processing regions...",
        error_prefix="Error processing region",
    )
    return aggregation


def _run_fused_region_bam_worker_pool(
    *,
    n_cores: int,
    apply_model_path: str,
    recall_model_path: Optional[str],
    emission_uplift: float,
    params: dict,
    work_items,
    total_regions: int,
    start_time: float,
) -> RegionBamAggregation:
    return _run_fused_region_bam_worker_pool_from_request(
        _FusedRegionBamWorkerPoolRequest(
            n_cores=n_cores,
            apply_model_path=apply_model_path,
            recall_model_path=recall_model_path,
            emission_uplift=emission_uplift,
            params=params,
            work_items=work_items,
            total_regions=total_regions,
            start_time=start_time,
        )
    )


def _run_fused_region_bam_worker_pool_from_request(
    request: _FusedRegionBamWorkerPoolRequest,
) -> RegionBamAggregation:
    aggregation = RegionBamAggregation()
    _run_region_worker_pool(
        n_cores=request.n_cores,
        initializer=_init_fused_region_worker,
        initargs=(
            request.apply_model_path,
            request.recall_model_path,
            request.emission_uplift,
            request.params,
        ),
        worker=_process_region_to_bam_fused,
        work_items=request.work_items,
        aggregation=aggregation,
        result_type=RegionBamResult,
        total_regions=request.total_regions,
        start_time=request.start_time,
        init_message=(
            f"  Initializing {request.n_cores} workers "
            "(loading apply model + LLR tables)..."
        ),
        ready_message="Processing...",
        progress_config=_RegionProgressConfig(
            footprint_label="With FP",
            rate_unit="r/s",
            rate_precision=0,
        ),
    )
    return aggregation


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
    return _process_bam_region_parallel_from_request(
        _RegionBamPipelineRequest(
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
    )


def _process_bam_region_parallel_from_request(
    request: _RegionBamPipelineRequest,
) -> Tuple[int, int]:
    start_time = time.time()
    return_posteriors = request.output_posteriors is not None

    run = _prepare_region_bam_parallel_run_from_request(
        request,
        return_posteriors=return_posteriors,
    )

    try:
        params = _region_bam_worker_params_from_request(
            request,
            return_posteriors=return_posteriors,
        )

        work_items = _region_bam_work_items(
            run.regions,
            request.input_bam,
            run.temp_dir,
            include_tsv=return_posteriors,
        )

        aggregation = _run_region_bam_worker_pool(
            n_cores=request.n_cores,
            model_path=request.model_path,
            params=params,
            work_items=work_items,
            total_regions=len(run.regions),
            start_time=start_time,
            include_tsv=True,
        )

        return _finalize_region_bam_parallel_run(
            request.input_bam,
            request.output_bam,
            run.temp_dir,
            aggregation,
            start_time,
            request.n_cores,
            return_posteriors,
            request.output_posteriors,
            request.mode,
            request.context_size,
            request.edge_trim,
        )

    finally:
        # Clean up temp directory
        shutil.rmtree(run.temp_dir, ignore_errors=True)


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
    return _process_bed_region_parallel_from_request(
        _RegionBedPipelineRequest(
            input_bam=input_bam,
            output_bed=output_bed,
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
        )
    )


def _process_bed_region_parallel_from_request(
    request: _RegionBedPipelineRequest,
) -> Tuple[int, int]:
    start_time = time.time()

    run = _prepare_region_parallel_run(
        request.input_bam,
        request.output_bed,
        request.region_size,
        request.skip_scaffolds,
        request.chroms,
        request.n_cores,
        temp_prefix='.fiberhmm_bed_tmp_',
        output_label=" (BED output)",
    )

    try:
        params = _base_region_worker_params(
            edge_trim=request.edge_trim,
            circular=request.circular,
            mode=request.mode,
            context_size=request.context_size,
            msp_min_size=request.msp_min_size,
            nuc_min_size=request.nuc_min_size,
            min_mapq=request.min_mapq,
            prob_threshold=request.prob_threshold,
            min_read_length=request.min_read_length,
            with_scores=request.with_scores,
            train_rids=request.train_rids,
            primary_only=request.primary_only,
        )

        work_items = _region_bed_work_items(
            run.regions,
            request.input_bam,
            run.temp_dir,
        )

        aggregation = _run_region_bed_worker_pool(
            n_cores=request.n_cores,
            model_path=request.model_path,
            params=params,
            work_items=work_items,
            total_regions=len(run.regions),
            start_time=start_time,
        )

        return _finalize_region_bed_parallel_run(
            request.output_bed, aggregation, start_time,
        )

    finally:
        shutil.rmtree(run.temp_dir, ignore_errors=True)


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
    nuc_profile_path: str = None,
    pg_record: dict = None,
):
    """Region-parallel fused apply+recall.

    Splits the BAM into genomic regions, runs fused apply+recall in each
    region as an independent worker, then concatenates sorted temp BAMs
    in region order.  Input BAM must be coordinate-sorted + indexed.

    Output is coordinate-sorted with no sort pass needed.
    """
    return _process_bam_region_parallel_fused_from_request(
        _FusedRegionBamPipelineRequest(
            input_bam=input_bam,
            output_bam=output_bam,
            apply_model_path=apply_model_path,
            recall_model_path=recall_model_path,
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
            min_llr=min_llr,
            min_opps=min_opps,
            unify_threshold=unify_threshold,
            emission_uplift=emission_uplift,
            also_write_legacy=also_write_legacy,
            downstream_compat=downstream_compat,
            n_cores=n_cores,
            region_size=region_size,
            skip_scaffolds=skip_scaffolds,
            chroms=chroms,
            io_threads=io_threads,
            primary_only=primary_only,
            ref_fasta_path=ref_fasta_path,
            recall_nucs=recall_nucs,
            split_min_llr=split_min_llr,
            split_min_opps=split_min_opps,
            filter_chimeras=filter_chimeras,
            chimera_min_seg=chimera_min_seg,
            chimera_purity=chimera_purity,
            phase_nrl=phase_nrl,
            nuc_profile_path=nuc_profile_path,
            pg_record=pg_record,
        )
    )


def _process_bam_region_parallel_fused_from_request(
    request: _FusedRegionBamPipelineRequest,
) -> Tuple[int, int]:
    start_time = time.time()

    run = _prepare_fused_region_bam_parallel_run_from_request(request)

    try:
        params = _fused_region_worker_params_from_request(request)

        work_items = _region_bam_work_items(
            run.regions,
            request.input_bam,
            run.temp_dir,
        )

        aggregation = _run_fused_region_bam_worker_pool(
            n_cores=request.n_cores,
            apply_model_path=request.apply_model_path,
            recall_model_path=request.recall_model_path,
            emission_uplift=request.emission_uplift,
            params=params,
            work_items=work_items,
            total_regions=len(run.regions),
            start_time=start_time,
        )
        print()

        _finalize_fused_region_bam_output(
            request.input_bam,
            request.output_bam,
            run.temp_dir,
            aggregation,
            start_time,
        )

    finally:
        shutil.rmtree(run.temp_dir, ignore_errors=True)

    return aggregation.total_reads, aggregation.reads_with_footprints
