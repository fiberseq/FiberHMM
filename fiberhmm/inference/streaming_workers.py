"""Multiprocessing worker entry points for streaming inference pipelines."""

from __future__ import annotations

from dataclasses import dataclass

from fiberhmm.core.model_io import load_model_for_inference
from fiberhmm.inference.engine import (
    CHIMERA_SKIP,
    _process_single_read,
    configure_daf_chimera_filter,
    extract_fiber_read_from_payload,
)
from fiberhmm.inference.fused_stages import (
    apply_result_has_footprints,
    build_fused_recall_result,
    run_hmm_apply_stage,
)
from fiberhmm.inference.recall_tables import load_recall_llr_tables
from fiberhmm.inference.worker_results import WorkerChunkResult
from fiberhmm.inference.worker_warmup import (
    disable_numba_cache_locking,
    warm_up_model_predict,
    warm_up_tf_recaller,
)

# Picklable per-result marker for a DAF chimera-filtered read (CHIMERA_SKIP is an
# identity sentinel that does not survive worker IPC, so the worker emits this
# string instead and the drain tallies it).
CHIMERA_RESULT = "__fiberhmm_chimera_skip__"

_worker_model = None
_worker_debug_timing = False

# Per-worker recall state: LLR tables for the TF Kadane scan. Lives alongside
# _worker_model and is populated by _init_fused_worker.
_worker_recall_state = {}


@dataclass(frozen=True)
class _PayloadWorkerConfig:
    edge_trim: int
    circular: bool
    mode: str
    context_size: int
    msp_min_size: int
    nuc_min_size: int
    with_scores: bool
    return_posteriors: bool
    prob_threshold: int


@dataclass(frozen=True)
class _FusedPayloadWorkerConfig:
    edge_trim: int
    circular: bool
    mode: str
    context_size: int
    msp_min_size: int
    nuc_min_size: int
    with_scores: bool
    prob_threshold: int
    min_llr: float
    min_opps: int
    unify_threshold: int


@dataclass(frozen=True)
class _WorkerRecallTables:
    llr_hit: object
    llr_miss: object


@dataclass(frozen=True)
class _WorkerFusedApplyStageRequest:
    fiber_read: object
    edge_trim: int
    circular: bool
    mode: str
    context_size: int
    msp_min_size: int
    nuc_min_size: int
    with_scores: bool


@dataclass(frozen=True)
class _FusedConfiguredApplyStageRequest:
    fiber_read: object
    config: _FusedPayloadWorkerConfig


@dataclass(frozen=True)
class _WorkerFusedRecallResultRequest:
    fiber_read: object
    apply_result: object
    llr_hit: object
    llr_miss: object
    min_llr: float
    min_opps: int
    unify_threshold: int
    with_scores: bool
    nuc_min_size: int
    msp_min_size: int


@dataclass(frozen=True)
class _FusedConfiguredRecallResultRequest:
    fiber_read: object
    apply_result: object
    llr_hit: object
    llr_miss: object
    config: _FusedPayloadWorkerConfig


@dataclass(frozen=True)
class _FusedFiberReadProcessRequest:
    fiber_read: object
    config: _FusedPayloadWorkerConfig
    llr_hit: object
    llr_miss: object


@dataclass(frozen=True)
class _FusedPayloadItemProcessRequest:
    payload: object
    config: _FusedPayloadWorkerConfig
    llr_hit: object
    llr_miss: object


def _payload_worker_config(
    edge_trim: int,
    circular: bool,
    mode: str,
    context_size: int,
    msp_min_size: int,
    nuc_min_size: int,
    with_scores: bool,
    return_posteriors: bool,
    prob_threshold: int,
) -> _PayloadWorkerConfig:
    return _PayloadWorkerConfig(
        edge_trim=edge_trim,
        circular=circular,
        mode=mode,
        context_size=context_size,
        msp_min_size=msp_min_size,
        nuc_min_size=nuc_min_size,
        with_scores=with_scores,
        return_posteriors=return_posteriors,
        prob_threshold=prob_threshold,
    )


def _fused_payload_worker_config(
    edge_trim: int,
    circular: bool,
    mode: str,
    context_size: int,
    msp_min_size: int,
    nuc_min_size: int,
    with_scores: bool,
    prob_threshold: int,
    min_llr: float,
    min_opps: int,
    unify_threshold: int,
) -> _FusedPayloadWorkerConfig:
    return _FusedPayloadWorkerConfig(
        edge_trim=edge_trim,
        circular=circular,
        mode=mode,
        context_size=context_size,
        msp_min_size=msp_min_size,
        nuc_min_size=nuc_min_size,
        with_scores=with_scores,
        prob_threshold=prob_threshold,
        min_llr=min_llr,
        min_opps=min_opps,
        unify_threshold=unify_threshold,
    )


def _process_worker_items(items, process_item):
    results = []
    read_failures = 0
    for item in items:
        try:
            result = process_item(item)
        except Exception:
            result = None
            read_failures += 1
        results.append(result)
    return WorkerChunkResult(results, read_failures)


def _payload_fiber_read_result(
    payload, mode: str, prob_threshold: int, *, chimera_result=None
):
    fiber_read = extract_fiber_read_from_payload(payload, mode, prob_threshold)
    if fiber_read is CHIMERA_SKIP:
        return chimera_result
    return fiber_read


def _fused_payload_fiber_read_result(payload, mode: str, prob_threshold: int):
    return _payload_fiber_read_result(
        payload,
        mode,
        prob_threshold,
        chimera_result=CHIMERA_RESULT,
    )


def _run_worker_single_read(
    fiber_read,
    edge_trim: int,
    circular: bool,
    mode: str,
    context_size: int,
    msp_min_size: int,
    nuc_min_size: int,
    with_scores: bool,
    return_posteriors: bool,
):
    return _process_single_read(
        fiber_read,
        _worker_model,
        edge_trim,
        circular,
        mode,
        context_size,
        msp_min_size,
        nuc_min_size=nuc_min_size,
        with_scores=with_scores,
        return_posteriors=return_posteriors,
    )


def _run_payload_configured_read(fiber_read, config: _PayloadWorkerConfig):
    return _run_worker_single_read(
        fiber_read,
        config.edge_trim,
        config.circular,
        config.mode,
        config.context_size,
        config.msp_min_size,
        config.nuc_min_size,
        config.with_scores,
        config.return_posteriors,
    )


def _process_payload_item(payload, config: _PayloadWorkerConfig):
    fiber_read = _payload_fiber_read_result(
        payload,
        config.mode,
        config.prob_threshold,
    )
    if fiber_read is None:
        return None
    return _run_payload_configured_read(fiber_read, config)


def _fused_recall_state(llr_hit, llr_miss, recall_nucs: bool,
                        split_min_llr: float, split_min_opps: int,
                        phase_nrl: int) -> dict:
    return {
        'llr_hit': llr_hit,
        'llr_miss': llr_miss,
        'recall_nucs': recall_nucs,
        'split_min_llr': split_min_llr,
        'split_min_opps': split_min_opps,
        'phase_nrl': phase_nrl,
    }


def _worker_recall_options(nuc_min_size: int, msp_min_size: int) -> dict:
    return {
        'recall_nucs': _worker_recall_state.get('recall_nucs', False),
        'split_min_llr': _worker_recall_state.get('split_min_llr', 4.0),
        'split_min_opps': _worker_recall_state.get('split_min_opps', 3),
        'nuc_min_size': nuc_min_size,
        'msp_min_size': msp_min_size,
        'phase_nrl': _worker_recall_state.get('phase_nrl', 0),
    }


def _worker_recall_tables():
    return _WorkerRecallTables(
        llr_hit=_worker_recall_state['llr_hit'],
        llr_miss=_worker_recall_state['llr_miss'],
    )


def _run_worker_fused_apply_stage_from_request(
    request: _WorkerFusedApplyStageRequest,
):
    return run_hmm_apply_stage(
        request.fiber_read,
        _worker_model,
        request.edge_trim,
        request.circular,
        request.mode,
        request.context_size,
        request.msp_min_size,
        request.nuc_min_size,
        request.with_scores,
    )


def _run_worker_fused_apply_stage(
    fiber_read,
    edge_trim: int,
    circular: bool,
    mode: str,
    context_size: int,
    msp_min_size: int,
    nuc_min_size: int,
    with_scores: bool,
):
    return _run_worker_fused_apply_stage_from_request(
        _WorkerFusedApplyStageRequest(
            fiber_read=fiber_read,
            edge_trim=edge_trim,
            circular=circular,
            mode=mode,
            context_size=context_size,
            msp_min_size=msp_min_size,
            nuc_min_size=nuc_min_size,
            with_scores=with_scores,
        )
    )


def _run_fused_configured_apply_stage_from_request(
    request: _FusedConfiguredApplyStageRequest,
):
    config = request.config
    return _run_worker_fused_apply_stage_from_request(
        _WorkerFusedApplyStageRequest(
            fiber_read=request.fiber_read,
            edge_trim=config.edge_trim,
            circular=config.circular,
            mode=config.mode,
            context_size=config.context_size,
            msp_min_size=config.msp_min_size,
            nuc_min_size=config.nuc_min_size,
            with_scores=config.with_scores,
        )
    )


def _run_fused_configured_apply_stage(
    fiber_read,
    config: _FusedPayloadWorkerConfig,
):
    return _run_fused_configured_apply_stage_from_request(
        _FusedConfiguredApplyStageRequest(
            fiber_read=fiber_read,
            config=config,
        )
    )


def _build_worker_fused_recall_result_from_request(
    request: _WorkerFusedRecallResultRequest,
):
    return build_fused_recall_result(
        request.fiber_read,
        request.apply_result,
        request.llr_hit,
        request.llr_miss,
        request.min_llr,
        request.min_opps,
        request.unify_threshold,
        request.with_scores,
        **_worker_recall_options(request.nuc_min_size, request.msp_min_size),
    )


def _build_worker_fused_recall_result(
    fiber_read,
    apply_result,
    llr_hit,
    llr_miss,
    min_llr: float,
    min_opps: int,
    unify_threshold: int,
    with_scores: bool,
    nuc_min_size: int,
    msp_min_size: int,
):
    return _build_worker_fused_recall_result_from_request(
        _WorkerFusedRecallResultRequest(
            fiber_read=fiber_read,
            apply_result=apply_result,
            llr_hit=llr_hit,
            llr_miss=llr_miss,
            min_llr=min_llr,
            min_opps=min_opps,
            unify_threshold=unify_threshold,
            with_scores=with_scores,
            nuc_min_size=nuc_min_size,
            msp_min_size=msp_min_size,
        )
    )


def _build_fused_configured_recall_result_from_request(
    request: _FusedConfiguredRecallResultRequest,
):
    config = request.config
    return _build_worker_fused_recall_result_from_request(
        _WorkerFusedRecallResultRequest(
            fiber_read=request.fiber_read,
            apply_result=request.apply_result,
            llr_hit=request.llr_hit,
            llr_miss=request.llr_miss,
            min_llr=config.min_llr,
            min_opps=config.min_opps,
            unify_threshold=config.unify_threshold,
            with_scores=config.with_scores,
            nuc_min_size=config.nuc_min_size,
            msp_min_size=config.msp_min_size,
        )
    )


def _build_fused_configured_recall_result(
    fiber_read,
    apply_result,
    llr_hit,
    llr_miss,
    config: _FusedPayloadWorkerConfig,
):
    return _build_fused_configured_recall_result_from_request(
        _FusedConfiguredRecallResultRequest(
            fiber_read=fiber_read,
            apply_result=apply_result,
            llr_hit=llr_hit,
            llr_miss=llr_miss,
            config=config,
        )
    )


def _process_fused_fiber_read_from_request(
    request: _FusedFiberReadProcessRequest,
):
    apply_result = _run_fused_configured_apply_stage_from_request(
        _FusedConfiguredApplyStageRequest(
            fiber_read=request.fiber_read,
            config=request.config,
        )
    )

    # Match streaming semantics: if apply produced no footprints and no MSPs,
    # pass the read through unchanged and preserve any pre-existing tags.
    if not apply_result_has_footprints(apply_result):
        return None

    return _build_fused_configured_recall_result_from_request(
        _FusedConfiguredRecallResultRequest(
            fiber_read=request.fiber_read,
            apply_result=apply_result,
            llr_hit=request.llr_hit,
            llr_miss=request.llr_miss,
            config=request.config,
        )
    )


def _process_fused_fiber_read(
    fiber_read,
    config: _FusedPayloadWorkerConfig,
    llr_hit,
    llr_miss,
):
    return _process_fused_fiber_read_from_request(
        _FusedFiberReadProcessRequest(
            fiber_read=fiber_read,
            config=config,
            llr_hit=llr_hit,
            llr_miss=llr_miss,
        )
    )


def _process_fused_payload_item_from_request(
    request: _FusedPayloadItemProcessRequest,
):
    fiber_read = _fused_payload_fiber_read_result(
        request.payload,
        request.config.mode,
        request.config.prob_threshold,
    )
    if fiber_read == CHIMERA_RESULT:
        return CHIMERA_RESULT
    if fiber_read is None:
        return None

    return _process_fused_fiber_read_from_request(
        _FusedFiberReadProcessRequest(
            fiber_read=fiber_read,
            config=request.config,
            llr_hit=request.llr_hit,
            llr_miss=request.llr_miss,
        )
    )


def _process_fused_payload_item(
    payload,
    config: _FusedPayloadWorkerConfig,
    llr_hit,
    llr_miss,
):
    return _process_fused_payload_item_from_request(
        _FusedPayloadItemProcessRequest(
            payload=payload,
            config=config,
            llr_hit=llr_hit,
            llr_miss=llr_miss,
        )
    )


def _init_bam_worker(model_path, debug_timing=False):
    """Initialize worker process with model."""
    global _worker_model, _worker_debug_timing
    try:
        # Disable numba caching to avoid file lock contention between workers.
        disable_numba_cache_locking()

        _worker_model = load_model_for_inference(model_path)
        _worker_debug_timing = debug_timing

        # Warm up numba JIT compilation in this worker.
        warm_up_model_predict(_worker_model)
    except Exception as e:
        import traceback

        print(f"Worker init error: {e}")
        traceback.print_exc()
        raise


def _init_fused_worker(
    apply_model_path,
    recall_model_path=None,
    emission_uplift=1.0,
    debug_timing=False,
    recall_nucs=False,
    split_min_llr=4.0,
    split_min_opps=3,
    filter_chimeras=True,
    chimera_min_seg=5,
    chimera_purity=0.8,
    phase_nrl=0,
):
    """Initialize worker process for the fused apply+recall pipeline.

    Loads the apply HMM model plus the LLR tables used for the TF Kadane
    scan. recall_model_path=None means reuse the apply model's emissions
    (the common case -- same model file drives both passes).
    """
    global _worker_model, _worker_debug_timing, _worker_recall_state

    disable_numba_cache_locking()

    _worker_model = load_model_for_inference(apply_model_path)
    _worker_debug_timing = debug_timing

    llr_hit, llr_miss = load_recall_llr_tables(
        recall_model_path,
        apply_model_path,
        emission_uplift,
    )
    _worker_recall_state = _fused_recall_state(
        llr_hit, llr_miss, recall_nucs, split_min_llr, split_min_opps, phase_nrl,
    )
    configure_daf_chimera_filter(filter_chimeras, chimera_min_seg, chimera_purity)

    # Warmup: apply Viterbi + TF Kadane scan.
    warm_up_model_predict(_worker_model)
    warm_up_tf_recaller(llr_hit, llr_miss)


def _process_chunk_worker(
    chunk_reads: list,
    edge_trim: int,
    circular: bool,
    mode: str,
    context_size: int,
    msp_min_size: int,
    nuc_min_size: int = 85,
    with_scores: bool = False,
    return_posteriors: bool = False,
) -> list:
    """Worker function to process a chunk of reads.

    Per-read errors are caught and converted to None results so a single bad
    read can never bring down the entire worker process (and with it the whole
    chunk of ~500 reads). Reads that fail are written through to the output
    unchanged without footprint tags, with the failure count reported
    separately in WorkerChunkResult.
    """
    global _worker_model

    def process_item(fiber_read):
        return _run_worker_single_read(
            fiber_read,
            edge_trim,
            circular,
            mode,
            context_size,
            msp_min_size,
            nuc_min_size,
            with_scores,
            return_posteriors,
        )

    return _process_worker_items(chunk_reads, process_item)


def _process_fused_payload_chunk_worker(
    chunk_payloads: list,
    edge_trim: int,
    circular: bool,
    mode: str,
    context_size: int,
    msp_min_size: int,
    nuc_min_size: int = 85,
    with_scores: bool = False,
    prob_threshold: int = 125,
    # Recall params
    recall_mode: str = None,            # mode for the TF LLR tables (usually same as apply mode)
    recall_context_size: int = None,    # k for TF LLR tables (usually same)
    min_llr: float = 4.0,
    min_opps: int = 3,
    unify_threshold: int = 90,
) -> list:
    """Slim-IPC worker: apply HMM + TF recall in a single call per read.

    Returns WorkerChunkResult with one result entry per payload. Each entry
    is either None (no usable modification data or per-read worker failure)
    or a dict with:
        'ns', 'nl':  numpy int arrays of unified nucleosome footprints
                     (post-unification: short nucs overlapping TF calls are
                     demoted into the tf+ annotation track)
        'as', 'al':  numpy int arrays of MSPs (unchanged from apply)
        'tf_calls':  list of TFCall objects
        'ns_scores', 'as_scores': optional nq/aq scores if with_scores
    """
    global _worker_model, _worker_recall_state

    # Model/params set once per worker; TF LLR tables attached to the
    # worker globals via _init_fused_worker.
    recall_tables = _worker_recall_tables()
    config = _fused_payload_worker_config(
        edge_trim,
        circular,
        mode,
        context_size,
        msp_min_size,
        nuc_min_size,
        with_scores,
        prob_threshold,
        min_llr,
        min_opps,
        unify_threshold,
    )

    def process_item(payload):
        return _process_fused_payload_item(
            payload,
            config,
            recall_tables.llr_hit,
            recall_tables.llr_miss,
        )

    return _process_worker_items(chunk_payloads, process_item)


def _process_payload_chunk_worker(
    chunk_payloads: list,
    edge_trim: int,
    circular: bool,
    mode: str,
    context_size: int,
    msp_min_size: int,
    nuc_min_size: int = 85,
    with_scores: bool = False,
    return_posteriors: bool = False,
    prob_threshold: int = 125,
) -> list:
    """Slim-IPC worker: parses MM/ML payloads then runs HMM.

    The streaming pipeline ships slim payloads (built by make_apply_payload
    in main) instead of pre-parsed fiber_read dicts. Each worker does the
    MM/ML parse + HMM in parallel rather than serializing the parse on the
    main process. Returns WorkerChunkResult with one entry per payload, with
    None for reads that have no usable modification data or hit a per-read
    worker failure.
    """
    global _worker_model

    config = _payload_worker_config(
        edge_trim,
        circular,
        mode,
        context_size,
        msp_min_size,
        nuc_min_size,
        with_scores,
        return_posteriors,
        prob_threshold,
    )

    def process_item(payload):
        return _process_payload_item(payload, config)

    return _process_worker_items(chunk_payloads, process_item)
