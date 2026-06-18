"""Multiprocessing worker entry points for region-parallel inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TextIO

import pysam

from fiberhmm.core.model_io import load_model_for_inference
from fiberhmm.inference.engine import (
    CHIMERA_SKIP,
    _extract_fiber_read_from_pysam,
    _process_single_read,
    configure_daf_chimera_filter,
    extract_fiber_read_from_payload,
    make_apply_payload,
)
from fiberhmm.inference.fused_stages import (
    apply_result_has_footprints,
    build_fused_recall_result,
    run_hmm_apply_stage,
)
from fiberhmm.inference.read_filters import ReadFilterConfig, streaming_skip_reason
from fiberhmm.inference.recall_tables import load_recall_llr_tables
from fiberhmm.inference.region_types import (
    RegionBamResult,
    RegionBamWorkItem,
    RegionBedResult,
    RegionBedWorkItem,
)
from fiberhmm.inference.skip_reasons import (
    CHIMERA_SKIP_REASON,
    NO_FOOTPRINTS_SKIP_REASON,
    new_skip_reasons,
    record_skip_reason,
)
from fiberhmm.inference.tagging import (
    set_legacy_apply_tags,
    write_fused_recall_tags,
)
from fiberhmm.inference.worker_warmup import (
    disable_numba_cache_locking,
    warm_up_model_predict,
    warm_up_tf_recaller,
)
from fiberhmm.io.bam_header import append_coord_marker, maybe_append_pg
from fiberhmm.io.bed import bed12_row
from fiberhmm.posteriors.region_tsv import format_region_posterior_line

_worker_model = None
_worker_region_params = None
_worker_recall_state = {}

_PRE_OWNERSHIP_SKIP_REASONS = {"unmapped", "secondary_supplementary"}
_REGION_ROUTE_PROCESS = "process"
_REGION_ROUTE_SKIP = "skip"
_REGION_ROUTE_OUTSIDE = "outside_region"


@dataclass(frozen=True)
class _RegionApplyConfig:
    edge_trim: int
    circular: bool
    mode: str
    context_size: int
    msp_min_size: int
    nuc_min_size: int
    prob_threshold: int
    with_scores: bool
    io_threads: int


@dataclass(frozen=True)
class _RegionBamOutputConfig:
    return_posteriors: bool
    write_msps: bool


@dataclass(frozen=True)
class _RegionPosteriorTsv:
    file: Optional[TextIO]
    enabled: bool


@dataclass(frozen=True)
class _RegionPosteriorRecordRequest:
    tsv_file: object
    read: object
    result: dict


@dataclass(frozen=True)
class _RegionReadRoute:
    route: str
    skip_reason: Optional[str]


@dataclass(frozen=True)
class _RegionReadRouteRequest:
    read: object
    start: int
    end: int
    filter_config: ReadFilterConfig


@dataclass(frozen=True)
class _RegionReadFilterConfigRequest:
    params: dict
    require_train_rids: bool


@dataclass(frozen=True)
class _RegionBed12Blocks:
    block_starts: list
    block_sizes: list
    score_list: Optional[list]


@dataclass(frozen=True)
class _RegionBed12BlocksRequest:
    ref_start: int
    ref_end: int
    starts: object
    lengths: object
    scores: object = None


@dataclass(frozen=True)
class _RegionBed12RowRequest:
    ref_name: str
    ref_start: int
    ref_end: int
    read_id: str
    strand: str
    starts: object
    lengths: object
    scores: object = None


@dataclass(frozen=True)
class _RegionFiberReadResult:
    fiber_read: object | None
    skip_reason: Optional[str]


@dataclass(frozen=True)
class _RegionBamWorkerRuntime:
    apply_config: _RegionApplyConfig
    output_config: _RegionBamOutputConfig
    filter_config: ReadFilterConfig


@dataclass(frozen=True)
class _FusedRegionRecallConfig:
    min_llr: float
    min_opps: int
    unify_threshold: int
    also_write_legacy: bool
    downstream_compat: bool


@dataclass(frozen=True)
class _FusedRegionWorkerRuntime:
    apply_config: _RegionApplyConfig
    recall_config: _FusedRegionRecallConfig
    recall_options: dict
    filter_config: ReadFilterConfig
    ref_fasta: Optional[object]


@dataclass(frozen=True)
class _FusedRegionApplyReadRequest:
    fiber_read: object
    model: object
    apply_config: _RegionApplyConfig


@dataclass(frozen=True)
class _FusedRegionRecallResultRequest:
    fiber_read: object
    apply_result: object
    llr_hit: object
    llr_miss: object
    apply_config: _RegionApplyConfig
    recall_config: _FusedRegionRecallConfig
    recall_options: dict


@dataclass(frozen=True)
class _RegionBamReadDelta:
    total_reads: int = 0
    reads_with_footprints: int = 0
    written: int = 0
    skipped: int = 0
    posteriors_written: int = 0


@dataclass(frozen=True)
class _SkippedRegionReadCounts:
    written: int
    skipped: int


@dataclass(frozen=True)
class _FootprintedRegionWrite:
    written: int
    posterior_written: bool


@dataclass(frozen=True)
class _FootprintedRegionReadWriteRequest:
    outbam: object
    read: object
    result: dict
    with_scores: bool
    write_msps: bool
    tsv_file: object


@dataclass(frozen=True)
class _RegionBamReadProcessRequest:
    read: object
    outbam: object
    model: object
    apply_config: _RegionApplyConfig
    output_config: _RegionBamOutputConfig
    filter_config: ReadFilterConfig
    start: int
    end: int
    skip_reasons: dict
    return_posteriors: bool
    tsv_file: object


@dataclass(frozen=True)
class _RegionBedReadProcessRequest:
    read: object
    bed_out: object
    model: object
    apply_config: _RegionApplyConfig
    filter_config: ReadFilterConfig
    start: int
    end: int


@dataclass(frozen=True)
class _FusedRegionBamReadProcessRequest:
    read: object
    outbam: object
    model: object
    llr_hit: object
    llr_miss: object
    apply_config: _RegionApplyConfig
    recall_config: _FusedRegionRecallConfig
    recall_options: dict
    ref_fasta: object
    filter_config: ReadFilterConfig
    start: int
    end: int
    skip_reasons: dict


@dataclass
class _RegionBamWorkerCounts:
    total_reads: int = 0
    reads_with_footprints: int = 0
    written: int = 0
    skipped: int = 0
    posteriors_written: int = 0

    def add(self, delta: _RegionBamReadDelta) -> None:
        self.total_reads += delta.total_reads
        self.reads_with_footprints += delta.reads_with_footprints
        self.written += delta.written
        self.skipped += delta.skipped
        self.posteriors_written += delta.posteriors_written


@dataclass(frozen=True)
class _RegionBamResultRequest:
    temp_bam_path: str
    counts: _RegionBamWorkerCounts
    skip_reasons: dict
    temp_tsv_path: Optional[str]
    return_posteriors: bool


@dataclass(frozen=True)
class _RegionBedReadDelta:
    total_reads: int = 0
    reads_with_footprints: int = 0


@dataclass
class _RegionBedWorkerCounts:
    total_reads: int = 0
    reads_with_footprints: int = 0

    def add(self, delta: _RegionBedReadDelta) -> None:
        self.total_reads += delta.total_reads
        self.reads_with_footprints += delta.reads_with_footprints


def _region_apply_config(
    params: dict,
    *,
    with_scores_default: Optional[bool] = None,
) -> _RegionApplyConfig:
    with_scores = (
        params['with_scores'] if with_scores_default is None
        else params.get('with_scores', with_scores_default)
    )
    return _RegionApplyConfig(
        edge_trim=int(params['edge_trim']),
        circular=params['circular'],
        mode=params['mode'],
        context_size=int(params['context_size']),
        msp_min_size=int(params['msp_min_size']),
        nuc_min_size=int(params.get('nuc_min_size', 85)),
        prob_threshold=int(params['prob_threshold']),
        with_scores=with_scores,
        io_threads=int(params.get('io_threads', 4)),
    )


def _region_bam_output_config(
    params: dict,
    temp_tsv_path: Optional[str],
) -> _RegionBamOutputConfig:
    return _RegionBamOutputConfig(
        return_posteriors=bool(
            params.get('return_posteriors', False) and temp_tsv_path is not None
        ),
        write_msps=params.get('write_msps', True),
    )


def _open_region_posterior_tsv(
    output_config: _RegionBamOutputConfig,
    temp_tsv_path: Optional[str],
) -> _RegionPosteriorTsv:
    if not output_config.return_posteriors or not temp_tsv_path:
        return _RegionPosteriorTsv(file=None, enabled=False)
    try:
        return _RegionPosteriorTsv(file=open(temp_tsv_path, 'w'), enabled=True)
    except OSError:
        return _RegionPosteriorTsv(file=None, enabled=False)


def _region_bam_result_from_request(
    request: _RegionBamResultRequest,
) -> RegionBamResult:
    tsv_path = None
    if (
        request.return_posteriors
        and request.counts.posteriors_written > 0
        and request.temp_tsv_path
    ):
        tsv_path = request.temp_tsv_path
    return RegionBamResult(
        request.temp_bam_path,
        request.counts.total_reads,
        request.counts.reads_with_footprints,
        request.counts.written,
        tsv_path,
        request.skip_reasons,
    )


def _region_bam_result_from_counts(
    temp_bam_path: str,
    counts: _RegionBamWorkerCounts,
    skip_reasons: dict,
    temp_tsv_path: Optional[str],
    return_posteriors: bool,
) -> RegionBamResult:
    return _region_bam_result_from_request(
        _RegionBamResultRequest(
            temp_bam_path=temp_bam_path,
            counts=counts,
            skip_reasons=skip_reasons,
            temp_tsv_path=temp_tsv_path,
            return_posteriors=return_posteriors,
        )
    )


def _fused_region_recall_config(params: dict) -> _FusedRegionRecallConfig:
    return _FusedRegionRecallConfig(
        min_llr=float(params['min_llr']),
        min_opps=int(params['min_opps']),
        unify_threshold=int(params['unify_threshold']),
        also_write_legacy=params['also_write_legacy'],
        downstream_compat=params['downstream_compat'],
    )


def _new_region_skip_reasons() -> dict:
    return new_skip_reasons(NO_FOOTPRINTS_SKIP_REASON, CHIMERA_SKIP_REASON)


def _write_skipped_region_read(outbam, read, skip_reasons: dict, reason: str) -> int:
    """Pass through a skipped BAM read and count its reason."""
    outbam.write(read)
    record_skip_reason(skip_reasons, reason)
    return 1


def _record_skipped_region_read(
    outbam,
    read,
    skip_reasons: dict,
    reason: str,
    written: int,
    skipped: int,
) -> _SkippedRegionReadCounts:
    return _SkippedRegionReadCounts(
        written=written + _write_skipped_region_read(
            outbam, read, skip_reasons, reason,
        ),
        skipped=skipped + 1,
    )


def _write_unfootprinted_region_read(outbam, read, skip_reasons: dict) -> int:
    outbam.write(read)
    record_skip_reason(skip_reasons, NO_FOOTPRINTS_SKIP_REASON)
    return 1


def _write_footprinted_region_read_from_request(
    request: _FootprintedRegionReadWriteRequest,
) -> _FootprintedRegionWrite:
    set_legacy_apply_tags(
        request.read,
        request.result,
        request.with_scores,
        request.write_msps,
    )
    posterior_written = (
        request.tsv_file is not None
        and request.result.get('posteriors') is not None
        and _write_region_posterior_record_from_request(
            _RegionPosteriorRecordRequest(
                tsv_file=request.tsv_file,
                read=request.read,
                result=request.result,
            )
        )
    )
    request.outbam.write(request.read)
    return _FootprintedRegionWrite(1, bool(posterior_written))


def _write_footprinted_region_read(
    outbam,
    read,
    result: dict,
    with_scores: bool,
    write_msps: bool,
    tsv_file,
) -> _FootprintedRegionWrite:
    return _write_footprinted_region_read_from_request(
        _FootprintedRegionReadWriteRequest(
            outbam=outbam,
            read=read,
            result=result,
            with_scores=with_scores,
            write_msps=write_msps,
            tsv_file=tsv_file,
        )
    )


def _fiber_read_skip_reason(fiber_read) -> Optional[str]:
    if fiber_read is CHIMERA_SKIP:
        return CHIMERA_SKIP_REASON
    if fiber_read is None:
        return 'no_modifications'
    return None


def _region_fiber_read_result(fiber_read) -> _RegionFiberReadResult:
    skip_reason = _fiber_read_skip_reason(fiber_read)
    if skip_reason:
        return _RegionFiberReadResult(fiber_read=None, skip_reason=skip_reason)
    return _RegionFiberReadResult(fiber_read=fiber_read, skip_reason=None)


def _extract_region_fiber_read(
    read,
    mode: str,
    prob_threshold: int,
) -> _RegionFiberReadResult:
    try:
        fiber_read = _extract_fiber_read_from_pysam(read, mode, prob_threshold)
    except Exception:
        return _RegionFiberReadResult(
            fiber_read=None,
            skip_reason='extraction_failed',
        )
    return _region_fiber_read_result(fiber_read)


def _extract_region_payload_fiber_read(
    payload,
    mode: str,
    prob_threshold: int,
) -> _RegionFiberReadResult:
    if payload is None:
        return _RegionFiberReadResult(
            fiber_read=None,
            skip_reason='no_modifications',
        )
    try:
        fiber_read = extract_fiber_read_from_payload(payload, mode, prob_threshold)
    except Exception:
        return _RegionFiberReadResult(
            fiber_read=None,
            skip_reason='extraction_failed',
        )
    return _region_fiber_read_result(fiber_read)


def _run_region_apply_read(
    fiber_read,
    model,
    apply_config: _RegionApplyConfig,
    *,
    return_posteriors: bool = False,
):
    return _process_single_read(
        fiber_read,
        model,
        apply_config.edge_trim,
        apply_config.circular,
        apply_config.mode,
        apply_config.context_size,
        apply_config.msp_min_size,
        nuc_min_size=apply_config.nuc_min_size,
        with_scores=apply_config.with_scores,
        return_posteriors=return_posteriors,
    )


def _run_fused_region_apply_read_from_request(
    request: _FusedRegionApplyReadRequest,
):
    return run_hmm_apply_stage(
        request.fiber_read,
        request.model,
        request.apply_config.edge_trim,
        request.apply_config.circular,
        request.apply_config.mode,
        request.apply_config.context_size,
        request.apply_config.msp_min_size,
        request.apply_config.nuc_min_size,
        request.apply_config.with_scores,
    )


def _run_fused_region_apply_read(
    fiber_read,
    model,
    apply_config: _RegionApplyConfig,
):
    return _run_fused_region_apply_read_from_request(
        _FusedRegionApplyReadRequest(
            fiber_read=fiber_read,
            model=model,
            apply_config=apply_config,
        )
    )


def _build_fused_region_recall_result_from_request(
    request: _FusedRegionRecallResultRequest,
):
    return build_fused_recall_result(
        request.fiber_read,
        request.apply_result,
        request.llr_hit,
        request.llr_miss,
        request.recall_config.min_llr,
        request.recall_config.min_opps,
        request.recall_config.unify_threshold,
        request.apply_config.with_scores,
        **request.recall_options,
    )


def _build_fused_region_recall_result(
    fiber_read,
    apply_result,
    llr_hit,
    llr_miss,
    apply_config: _RegionApplyConfig,
    recall_config: _FusedRegionRecallConfig,
    recall_options: dict,
):
    return _build_fused_region_recall_result_from_request(
        _FusedRegionRecallResultRequest(
            fiber_read=fiber_read,
            apply_result=apply_result,
            llr_hit=llr_hit,
            llr_miss=llr_miss,
            apply_config=apply_config,
            recall_config=recall_config,
            recall_options=recall_options,
        )
    )


def _read_starts_in_region(read, start: int, end: int) -> bool:
    return int(start) <= int(read.reference_start) < int(end)


def _region_read_route_from_request(
    request: _RegionReadRouteRequest,
) -> _RegionReadRoute:
    """Classify a fetched read before region-worker processing.

    Pre-ownership skip reasons are passed through before checking
    reference_start. Other reads are processed only by the region containing
    their start to avoid duplicate output from overlapping fetches.
    """
    skip_reason = streaming_skip_reason(request.read, request.filter_config)
    if skip_reason in _PRE_OWNERSHIP_SKIP_REASONS:
        return _RegionReadRoute(route=_REGION_ROUTE_SKIP, skip_reason=skip_reason)
    if not _read_starts_in_region(request.read, request.start, request.end):
        return _RegionReadRoute(route=_REGION_ROUTE_OUTSIDE, skip_reason=None)
    if skip_reason:
        return _RegionReadRoute(route=_REGION_ROUTE_SKIP, skip_reason=skip_reason)
    return _RegionReadRoute(route=_REGION_ROUTE_PROCESS, skip_reason=None)


def _region_read_route(read, start: int, end: int, filter_config: ReadFilterConfig):
    return _region_read_route_from_request(
        _RegionReadRouteRequest(
            read=read,
            start=start,
            end=end,
            filter_config=filter_config,
        )
    )


def _region_bed12_blocks_from_request(
    request: _RegionBed12BlocksRequest,
) -> _RegionBed12Blocks:
    read_length = request.ref_end - request.ref_start
    block_starts, block_sizes = _region_bed_block_components(
        request.ref_start,
        request.starts,
        request.lengths,
    )
    score_list = _region_bed_score_list(request.scores)

    # BED12 requires blocks to span chromStart to chromEnd.
    _pad_region_bed12_to_read_span(
        block_starts,
        block_sizes,
        score_list,
        read_length,
    )

    return _RegionBed12Blocks(
        block_starts=block_starts,
        block_sizes=block_sizes,
        score_list=score_list,
    )


def _region_bed12_blocks(
    ref_start,
    ref_end,
    starts,
    lengths,
    scores=None,
) -> _RegionBed12Blocks:
    return _region_bed12_blocks_from_request(
        _RegionBed12BlocksRequest(
            ref_start=ref_start,
            ref_end=ref_end,
            starts=starts,
            lengths=lengths,
            scores=scores,
        )
    )


def _pad_region_bed12_to_read_span(
    block_starts: list,
    block_sizes: list,
    score_list,
    read_length: int,
) -> None:
    if block_starts[0] != 0:
        block_starts.insert(0, 0)
        block_sizes.insert(0, 1)
        if score_list is not None:
            score_list.insert(0, 0)

    last_end = block_starts[-1] + block_sizes[-1]
    if last_end < read_length:
        block_starts.append(read_length - 1)
        block_sizes.append(1)
        if score_list is not None:
            score_list.append(0)


def _region_bed_block_components(ref_start, starts, lengths):
    return (
        [int(start - ref_start) for start in starts],
        [int(length) for length in lengths],
    )


def _region_bed_score_list(scores):
    if scores is None:
        return None
    return [int(score * 1000) for score in scores]


def _format_region_bed12_row_from_request(
    request: _RegionBed12RowRequest,
) -> str:
    """Format one region-worker BED12 row from reference-frame intervals."""
    bed12_blocks = _region_bed12_blocks_from_request(
        _RegionBed12BlocksRequest(
            ref_start=request.ref_start,
            ref_end=request.ref_end,
            starts=request.starts,
            lengths=request.lengths,
            scores=request.scores,
        )
    )
    blocks = [
        (request.ref_start + start, request.ref_start + start + size)
        for start, size in zip(
            bed12_blocks.block_starts,
            bed12_blocks.block_sizes,
        )
    ]
    extra = ()
    if bed12_blocks.score_list is not None:
        extra = (','.join(str(score) for score in bed12_blocks.score_list),)
    return bed12_row(
        request.ref_name,
        request.ref_start,
        request.ref_end,
        request.read_id,
        0,
        request.strand,
        blocks,
        extra,
        item_rgb='0,0,0',
    )


def _format_region_bed12_row(ref_name, ref_start, ref_end, read_id, strand,
                             starts, lengths, scores=None):
    return _format_region_bed12_row_from_request(
        _RegionBed12RowRequest(
            ref_name=ref_name,
            ref_start=ref_start,
            ref_end=ref_end,
            read_id=read_id,
            strand=strand,
            starts=starts,
            lengths=lengths,
            scores=scores,
        )
    )


def _region_read_filter_config_from_request(
    request: _RegionReadFilterConfigRequest,
) -> ReadFilterConfig:
    train_rids = (
        request.params['train_rids'] if request.require_train_rids
        else request.params.get('train_rids') or set()
    )
    return ReadFilterConfig(
        min_mapq=int(request.params['min_mapq']),
        min_read_length=int(request.params['min_read_length']),
        primary_only=request.params.get('primary_only', False),
        process_unmapped=False,
        train_rids=train_rids,
    )


def _region_read_filter_config(
    params: dict,
    *,
    require_train_rids: bool,
) -> ReadFilterConfig:
    return _region_read_filter_config_from_request(
        _RegionReadFilterConfigRequest(
            params=params,
            require_train_rids=require_train_rids,
        )
    )


def _region_bed_read_filter_config(params: dict) -> ReadFilterConfig:
    return ReadFilterConfig(
        min_mapq=int(params['min_mapq']),
        min_read_length=int(params['min_read_length']),
        primary_only=True,
        process_unmapped=False,
        train_rids=params['train_rids'],
    )


def _region_bam_worker_runtime(
    params: dict,
    temp_tsv_path: Optional[str],
) -> _RegionBamWorkerRuntime:
    return _RegionBamWorkerRuntime(
        apply_config=_region_apply_config(params),
        output_config=_region_bam_output_config(params, temp_tsv_path),
        filter_config=_region_read_filter_config_from_request(
            _RegionReadFilterConfigRequest(
                params=params,
                require_train_rids=True,
            )
        ),
    )


def _region_fused_recall_options(
    params: dict,
    nuc_min_size: int,
    msp_min_size: int,
) -> dict:
    return {
        'recall_nucs': bool(params.get('recall_nucs', False)),
        'split_min_llr': float(params.get('split_min_llr', 4.0)),
        'split_min_opps': int(params.get('split_min_opps', 3)),
        'nuc_min_size': nuc_min_size,
        'msp_min_size': msp_min_size,
        'phase_nrl': int(params.get('phase_nrl', 0)),
    }


def _fused_region_worker_runtime(params: dict) -> _FusedRegionWorkerRuntime:
    apply_config = _region_apply_config(params, with_scores_default=False)
    return _FusedRegionWorkerRuntime(
        apply_config=apply_config,
        recall_config=_fused_region_recall_config(params),
        recall_options=_region_fused_recall_options(
            params,
            apply_config.nuc_min_size,
            apply_config.msp_min_size,
        ),
        filter_config=_region_read_filter_config_from_request(
            _RegionReadFilterConfigRequest(
                params=params,
                require_train_rids=False,
            )
        ),
        ref_fasta=params.get('ref_fasta'),
    )


def _write_region_posterior_record_from_request(
    request: _RegionPosteriorRecordRequest,
) -> bool:
    try:
        request.tsv_file.write(
            format_region_posterior_line(
                read_name=request.read.query_name,
                chrom=request.read.reference_name,
                ref_start=request.read.reference_start,
                ref_end=request.read.reference_end,
                strand=request.result.get('strand', '.'),
                posteriors=request.result['posteriors'],
                footprint_starts=request.result['ns'],
                footprint_sizes=request.result['nl'],
            )
        )
        return True
    except Exception:
        return False


def _write_region_posterior_record(tsv_file, read, result: dict) -> bool:
    return _write_region_posterior_record_from_request(
        _RegionPosteriorRecordRequest(
            tsv_file=tsv_file,
            read=read,
            result=result,
        )
    )


def _region_result_ns_scores(result: dict, with_scores: bool):
    if not with_scores:
        return None
    scores = result['ns_scores']
    return scores if scores is not None else None


def _region_bed12_row_from_read_result(read, result: dict, with_scores: bool) -> str:
    strand = '-' if read.is_reverse else '+'
    return _format_region_bed12_row_from_request(
        _RegionBed12RowRequest(
            ref_name=read.reference_name,
            ref_start=read.reference_start,
            ref_end=read.reference_end,
            read_id=read.query_name,
            strand=strand,
            starts=result['ns'],
            lengths=result['nl'],
            scores=_region_result_ns_scores(result, with_scores),
        )
    )


def _skipped_region_bam_delta(outbam, read, skip_reasons: dict, reason: str):
    counts = _record_skipped_region_read(
        outbam, read, skip_reasons, reason, written=0, skipped=0,
    )
    return _RegionBamReadDelta(written=counts.written, skipped=counts.skipped)


def _process_region_bam_read(
    read,
    outbam,
    model,
    apply_config: _RegionApplyConfig,
    output_config: _RegionBamOutputConfig,
    filter_config: ReadFilterConfig,
    start: int,
    end: int,
    skip_reasons: dict,
    *,
    return_posteriors: bool,
    tsv_file,
) -> _RegionBamReadDelta:
    return _process_region_bam_read_from_request(
        _RegionBamReadProcessRequest(
            read=read,
            outbam=outbam,
            model=model,
            apply_config=apply_config,
            output_config=output_config,
            filter_config=filter_config,
            start=start,
            end=end,
            skip_reasons=skip_reasons,
            return_posteriors=return_posteriors,
            tsv_file=tsv_file,
        )
    )


def _process_region_bam_read_from_request(
    request: _RegionBamReadProcessRequest,
) -> _RegionBamReadDelta:
    read_route = _region_read_route(
        request.read,
        request.start,
        request.end,
        request.filter_config,
    )
    if read_route.route == _REGION_ROUTE_SKIP:
        return _skipped_region_bam_delta(
            request.outbam,
            request.read,
            request.skip_reasons,
            read_route.skip_reason,
        )
    if read_route.route == _REGION_ROUTE_OUTSIDE:
        return _RegionBamReadDelta()

    fiber_result = _extract_region_fiber_read(
        request.read,
        request.apply_config.mode,
        request.apply_config.prob_threshold,
    )
    if fiber_result.skip_reason:
        return _skipped_region_bam_delta(
            request.outbam,
            request.read,
            request.skip_reasons,
            fiber_result.skip_reason,
        )

    result = _run_region_apply_read(
        fiber_result.fiber_read,
        request.model,
        request.apply_config,
        return_posteriors=request.return_posteriors,
    )

    if result is not None:
        write_result = _write_footprinted_region_read(
            request.outbam,
            request.read,
            result,
            request.apply_config.with_scores,
            request.output_config.write_msps,
            request.tsv_file,
        )
        return _RegionBamReadDelta(
            total_reads=1,
            reads_with_footprints=1,
            written=write_result.written,
            posteriors_written=int(write_result.posterior_written),
        )

    written = _write_unfootprinted_region_read(
        request.outbam,
        request.read,
        request.skip_reasons,
    )
    return _RegionBamReadDelta(total_reads=1, written=written)


def _process_region_bed_read(
    read,
    bed_out,
    model,
    apply_config: _RegionApplyConfig,
    filter_config: ReadFilterConfig,
    start: int,
    end: int,
) -> _RegionBedReadDelta:
    return _process_region_bed_read_from_request(
        _RegionBedReadProcessRequest(
            read=read,
            bed_out=bed_out,
            model=model,
            apply_config=apply_config,
            filter_config=filter_config,
            start=start,
            end=end,
        )
    )


def _process_region_bed_read_from_request(
    request: _RegionBedReadProcessRequest,
) -> _RegionBedReadDelta:
    if streaming_skip_reason(request.read, request.filter_config):
        return _RegionBedReadDelta()

    if not _read_starts_in_region(request.read, request.start, request.end):
        return _RegionBedReadDelta()

    fiber_result = _extract_region_fiber_read(
        request.read,
        request.apply_config.mode,
        request.apply_config.prob_threshold,
    )
    if fiber_result.skip_reason:
        return _RegionBedReadDelta()

    result = _run_region_apply_read(
        fiber_result.fiber_read,
        request.model,
        request.apply_config,
    )

    if result is not None and len(result['ns']) > 0:
        request.bed_out.write(
            _region_bed12_row_from_read_result(
                request.read,
                result,
                request.apply_config.with_scores,
            ) + "\n"
        )
        return _RegionBedReadDelta(total_reads=1, reads_with_footprints=1)

    return _RegionBedReadDelta(total_reads=1)


def _process_fused_region_bam_read(
    read,
    outbam,
    model,
    llr_hit,
    llr_miss,
    apply_config: _RegionApplyConfig,
    recall_config: _FusedRegionRecallConfig,
    recall_options: dict,
    ref_fasta,
    filter_config: ReadFilterConfig,
    start: int,
    end: int,
    skip_reasons: dict,
) -> _RegionBamReadDelta:
    return _process_fused_region_bam_read_from_request(
        _FusedRegionBamReadProcessRequest(
            read=read,
            outbam=outbam,
            model=model,
            llr_hit=llr_hit,
            llr_miss=llr_miss,
            apply_config=apply_config,
            recall_config=recall_config,
            recall_options=recall_options,
            ref_fasta=ref_fasta,
            filter_config=filter_config,
            start=start,
            end=end,
            skip_reasons=skip_reasons,
        )
    )


def _process_fused_region_bam_read_from_request(
    request: _FusedRegionBamReadProcessRequest,
) -> _RegionBamReadDelta:
    read_route = _region_read_route(
        request.read,
        request.start,
        request.end,
        request.filter_config,
    )
    if read_route.route == _REGION_ROUTE_SKIP:
        return _skipped_region_bam_delta(
            request.outbam,
            request.read,
            request.skip_reasons,
            read_route.skip_reason,
        )
    if read_route.route == _REGION_ROUTE_OUTSIDE:
        return _RegionBamReadDelta()

    payload = make_apply_payload(
        request.read,
        mode=request.apply_config.mode,
        ref_fasta=request.ref_fasta,
    )
    fiber_result = _extract_region_payload_fiber_read(
        payload,
        request.apply_config.mode,
        request.apply_config.prob_threshold,
    )
    if fiber_result.skip_reason:
        return _skipped_region_bam_delta(
            request.outbam,
            request.read,
            request.skip_reasons,
            fiber_result.skip_reason,
        )

    try:
        apply_result = _run_fused_region_apply_read(
            fiber_result.fiber_read,
            request.model,
            request.apply_config,
        )
    except Exception:
        return _skipped_region_bam_delta(
            request.outbam,
            request.read,
            request.skip_reasons,
            'extraction_failed',
        )

    if not apply_result_has_footprints(apply_result):
        written = _write_unfootprinted_region_read(
            request.outbam,
            request.read,
            request.skip_reasons,
        )
        return _RegionBamReadDelta(total_reads=1, written=written)

    fused_result = _build_fused_region_recall_result(
        fiber_result.fiber_read,
        apply_result,
        request.llr_hit,
        request.llr_miss,
        request.apply_config,
        request.recall_config,
        request.recall_options,
    )
    write_fused_recall_tags(
        request.read,
        read_length=len(fiber_result.fiber_read['query_sequence']),
        result=fused_result,
        also_write_legacy=request.recall_config.also_write_legacy,
        downstream_compat=request.recall_config.downstream_compat,
    )
    request.outbam.write(request.read)
    return _RegionBamReadDelta(
        total_reads=1,
        reads_with_footprints=1,
        written=1,
    )


def _init_region_worker(model_path: str, params: dict):
    """Initialize worker for region-parallel processing."""
    global _worker_model, _worker_region_params

    try:
        # Disable numba caching to avoid file lock contention.
        disable_numba_cache_locking()

        # Load model once per worker.
        _worker_model = load_model_for_inference(model_path)
        _worker_region_params = params

        configure_daf_chimera_filter(
            params.get('filter_chimeras', True),
            params.get('chimera_min_seg', 5),
            params.get('chimera_purity', 0.8),
        )

        # Warm up numba JIT.
        warm_up_model_predict(_worker_model)

    except Exception as e:
        import traceback

        print(f"Region worker init error: {e}")
        traceback.print_exc()
        raise


def _process_region_to_bam(args: RegionBamWorkItem) -> RegionBamResult:
    """
    Worker function: process one genomic region and write to temp BAM.

    Each worker opens its own BAM file handle and uses the index to fetch
    reads from its assigned region. This enables true parallel I/O.

    Uses global _worker_model and _worker_region_params (set by _init_region_worker).

    Args:
        args: RegionBamWorkItem, or the legacy tuple shape.

    Returns:
        RegionBamResult with temp BAM, counts, optional TSV path, and skip reasons.
    """
    import traceback

    global _worker_model, _worker_region_params

    chrom = start = end = '?'
    try:
        work_item = RegionBamWorkItem.from_value(args)
        chrom, start, end = work_item.region
        input_bam = work_item.input_bam
        temp_bam_path = work_item.temp_bam_path
        temp_tsv_path = work_item.temp_tsv_path

        # Ensure start/end are Python ints (not numpy).
        start = int(start)
        end = int(end)

        # Use global model and params (loaded once per worker).
        model = _worker_model
        params = _worker_region_params

        runtime = _region_bam_worker_runtime(params, temp_tsv_path)
        return_posteriors = runtime.output_config.return_posteriors

        counts = _RegionBamWorkerCounts()
        skip_reasons = _new_region_skip_reasons()

        pysam.set_verbosity(0)

        # Open posteriors TSV file for streaming writes (if requested).
        posterior_tsv = _open_region_posterior_tsv(
            runtime.output_config,
            temp_tsv_path,
        )
        tsv_file = posterior_tsv.file
        return_posteriors = posterior_tsv.enabled

        try:
            with pysam.AlignmentFile(
                input_bam, "rb",
                threads=runtime.apply_config.io_threads,
                check_sq=False,
            ) as inbam:
                with pysam.AlignmentFile(temp_bam_path, "wb",
                                         header=append_coord_marker(inbam.header),
                                         threads=runtime.apply_config.io_threads) as outbam:

                    # Fetch reads from this region using the index.
                    try:
                        read_iter = inbam.fetch(chrom, start, end)
                    except ValueError:
                        # Region not in BAM (e.g., unplaced contigs).
                        return RegionBamResult(temp_bam_path, 0, 0, 0)

                    for read in read_iter:
                        counts.add(_process_region_bam_read(
                            read,
                            outbam,
                            model,
                            runtime.apply_config,
                            runtime.output_config,
                            runtime.filter_config,
                            start,
                            end,
                            skip_reasons,
                            return_posteriors=return_posteriors,
                            tsv_file=tsv_file,
                        ))

        finally:
            if tsv_file is not None:
                tsv_file.close()

        return _region_bam_result_from_request(
            _RegionBamResultRequest(
                temp_bam_path=temp_bam_path,
                counts=counts,
                skip_reasons=skip_reasons,
                temp_tsv_path=temp_tsv_path,
                return_posteriors=return_posteriors,
            )
        )

    except Exception as e:
        print(f"\nWorker error in region {chrom}:{start}-{end}: {e}")
        traceback.print_exc()
        raise


def _process_region_to_bed(args: RegionBedWorkItem) -> RegionBedResult:
    """
    Process a genomic region and write BED output directly (no temp BAM).

    This is more space-efficient than _process_region_to_bam when only
    BED/bigBed output is needed.

    Args:
        args: RegionBedWorkItem, or the legacy tuple shape.

    Returns:
        RegionBedResult with temp BED path and counts.
    """
    work_item = RegionBedWorkItem.from_value(args)
    region = work_item.region
    input_bam = work_item.input_bam
    temp_bed_path = work_item.temp_bed_path
    chrom, start, end = region

    try:
        start = int(start)
        end = int(end)

        model = _worker_model
        params = _worker_region_params

        apply_config = _region_apply_config(params)
        filter_config = _region_bed_read_filter_config(params)

        counts = _RegionBedWorkerCounts()

        pysam.set_verbosity(0)

        with pysam.AlignmentFile(
            input_bam, "rb", threads=apply_config.io_threads, check_sq=False
        ) as inbam:
            with open(temp_bed_path, 'w') as bed_out:
                try:
                    read_iter = inbam.fetch(chrom, start, end)
                except ValueError:
                    return RegionBedResult(temp_bed_path, 0, 0)

                for read in read_iter:
                    counts.add(_process_region_bed_read(
                        read,
                        bed_out,
                        model,
                        apply_config,
                        filter_config,
                        start,
                        end,
                    ))

        return RegionBedResult(
            temp_bed_path, counts.total_reads, counts.reads_with_footprints,
        )

    except Exception as e:
        import traceback

        print(f"\nWorker error in region {chrom}:{start}-{end}: {e}")
        traceback.print_exc()
        raise


def _init_fused_region_worker(
    apply_model_path: str,
    recall_model_path: Optional[str],
    emission_uplift: float,
    params: dict,
):
    """Per-worker init for region-parallel fused apply+recall.

    Loads the apply HMM model, builds the TF LLR tables (from the recall
    model or by reusing the apply model), warms up numba JIT, and stashes
    params for the region worker to pick up.
    """
    global _worker_model, _worker_region_params, _worker_recall_state

    disable_numba_cache_locking()

    _worker_model = load_model_for_inference(apply_model_path)
    # Open the reference FASTA after fork: pysam.FastaFile is not fork-safe.
    ref_path = params.get('ref_fasta_path')
    if ref_path:
        import pysam as _pysam

        params = dict(params)   # don't mutate the shared-across-workers dict
        params['ref_fasta'] = _pysam.FastaFile(ref_path)
    _worker_region_params = params

    llr_hit, llr_miss = load_recall_llr_tables(
        recall_model_path,
        apply_model_path,
        emission_uplift,
    )
    _worker_recall_state['llr_hit'] = llr_hit
    _worker_recall_state['llr_miss'] = llr_miss

    configure_daf_chimera_filter(
        params.get('filter_chimeras', True),
        params.get('chimera_min_seg', 5),
        params.get('chimera_purity', 0.8),
    )

    warm_up_model_predict(_worker_model)
    warm_up_tf_recaller(llr_hit, llr_miss)


def _process_region_to_bam_fused(args: RegionBamWorkItem) -> RegionBamResult:
    """Region worker: fetch reads in one genomic region, run fused
    apply+recall per read, write in-order to a coordinate-sorted temp BAM.

    Because pysam.fetch(chrom,start,end) yields reads in coordinate order
    AND we only process reads that START in this region (the reference_start
    filter), each temp BAM is coordinate-sorted within itself. Concatenating
    temp BAMs in region order gives a coordinate-sorted final BAM without
    any sort pass.

    Returns a RegionBamResult with temp BAM, counts, and skip reasons.
    """
    import traceback

    global _worker_model, _worker_region_params, _worker_recall_state

    try:
        work_item = RegionBamWorkItem.from_value(args)
        chrom, start, end = work_item.region
        input_bam = work_item.input_bam
        temp_bam_path = work_item.temp_bam_path
        start = int(start)
        end = int(end)

        params = _worker_region_params
        model = _worker_model
        llr_hit = _worker_recall_state['llr_hit']
        llr_miss = _worker_recall_state['llr_miss']

        runtime = _fused_region_worker_runtime(params)

        pysam.set_verbosity(0)

        counts = _RegionBamWorkerCounts()
        skip_reasons = _new_region_skip_reasons()

        with pysam.AlignmentFile(
            input_bam, "rb", threads=runtime.apply_config.io_threads, check_sq=False
        ) as inbam:
            with pysam.AlignmentFile(
                    temp_bam_path, "wb",
                    header=maybe_append_pg(inbam.header, params.get('pg_record')),
                    threads=runtime.apply_config.io_threads) as outbam:
                try:
                    read_iter = inbam.fetch(chrom, start, end)
                except ValueError:
                    return RegionBamResult(temp_bam_path, 0, 0, 0)

                for read in read_iter:
                    counts.add(_process_fused_region_bam_read(
                        read,
                        outbam,
                        model,
                        llr_hit,
                        llr_miss,
                        runtime.apply_config,
                        runtime.recall_config,
                        runtime.recall_options,
                        runtime.ref_fasta,
                        runtime.filter_config,
                        start,
                        end,
                        skip_reasons,
                    ))

        return RegionBamResult(
            temp_bam_path, counts.total_reads, counts.reads_with_footprints,
            counts.written, None, skip_reasons,
        )

    except Exception:
        traceback.print_exc()
        raise
