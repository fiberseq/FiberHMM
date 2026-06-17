"""Multiprocessing worker entry points for streaming inference pipelines."""

from __future__ import annotations

from fiberhmm.core.model_io import load_model_for_inference
from fiberhmm.inference.engine import (
    CHIMERA_SKIP,
    _process_single_read,
    configure_daf_chimera_filter,
    extract_fiber_read_from_payload,
)

# Picklable per-result marker for a DAF chimera-filtered read (CHIMERA_SKIP is an
# identity sentinel that does not survive worker IPC, so the worker emits this
# string instead and the drain tallies it).
CHIMERA_RESULT = "__fiberhmm_chimera_skip__"
from fiberhmm.inference.fused_stages import (
    apply_result_has_footprints,
    build_fused_recall_result,
    run_hmm_apply_stage,
)
from fiberhmm.inference.worker_warmup import (
    disable_numba_cache_locking,
    warm_up_model_predict,
    warm_up_tf_recaller,
)
from fiberhmm.inference.worker_results import WorkerChunkResult

_worker_model = None
_worker_debug_timing = False

# Per-worker recall state: LLR tables for the TF Kadane scan. Lives alongside
# _worker_model and is populated by _init_fused_worker.
_worker_recall_state = {}


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

    # Build TF-recall LLR tables from the recall model (or reuse apply model).
    from fiberhmm.core.model_io import load_model_with_metadata
    from fiberhmm.inference.tf_recaller import (
        apply_emission_uplift,
        build_llr_tables,
    )

    r_path = recall_model_path or apply_model_path
    r_model, _, _ = load_model_with_metadata(r_path)
    llr_hit, llr_miss = build_llr_tables(r_model)
    if abs(emission_uplift - 1.0) > 1e-9:
        llr_hit, llr_miss = apply_emission_uplift(
            llr_hit,
            llr_miss,
            r_model,
            emission_uplift,
        )
    _worker_recall_state['llr_hit'] = llr_hit
    _worker_recall_state['llr_miss'] = llr_miss
    _worker_recall_state['recall_nucs'] = recall_nucs
    _worker_recall_state['split_min_llr'] = split_min_llr
    _worker_recall_state['split_min_opps'] = split_min_opps
    _worker_recall_state['phase_nrl'] = phase_nrl
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

    results = []
    read_failures = 0
    for fiber_read in chunk_reads:
        try:
            result = _process_single_read(
                fiber_read, _worker_model, edge_trim, circular,
                mode, context_size, msp_min_size, nuc_min_size=nuc_min_size,
                with_scores=with_scores,
                return_posteriors=return_posteriors,
            )
        except Exception:
            # Per-read failure: skip this read but keep the worker alive.
            result = None
            read_failures += 1
        results.append(result)

    return WorkerChunkResult(results, read_failures)


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
    llr_hit = _worker_recall_state['llr_hit']
    llr_miss = _worker_recall_state['llr_miss']

    results = []
    read_failures = 0
    for payload in chunk_payloads:
        try:
            fiber_read = extract_fiber_read_from_payload(payload, mode, prob_threshold)
            if fiber_read is CHIMERA_SKIP:
                # DAF strand-swap chimera filtered out -- distinct marker so the
                # drain can tally it (vs folding into "no footprints"). The
                # marker is a picklable string (CHIMERA_SKIP is an identity
                # sentinel that does not survive the worker IPC boundary).
                results.append(CHIMERA_RESULT)
                continue
            if fiber_read is None:
                results.append(None)
                continue

            apply_result = run_hmm_apply_stage(
                fiber_read,
                _worker_model,
                edge_trim,
                circular,
                mode,
                context_size,
                msp_min_size,
                nuc_min_size,
                with_scores,
            )

            # Match streaming semantics: if apply produced no footprints and
            # no MSPs, treat as "nothing to annotate" so the drain pass-throughs
            # the read unchanged (preserving any pre-existing tags on input).
            if not apply_result_has_footprints(apply_result):
                results.append(None)
                continue

            results.append(build_fused_recall_result(
                fiber_read,
                apply_result,
                llr_hit,
                llr_miss,
                min_llr,
                min_opps,
                unify_threshold,
                with_scores,
                recall_nucs=_worker_recall_state.get('recall_nucs', False),
                split_min_llr=_worker_recall_state.get('split_min_llr', 4.0),
                split_min_opps=_worker_recall_state.get('split_min_opps', 3),
                nuc_min_size=nuc_min_size,
                msp_min_size=msp_min_size,
                phase_nrl=_worker_recall_state.get('phase_nrl', 0),
            ))
        except Exception:
            # Per-read failure must not kill the worker or the whole chunk.
            read_failures += 1
            results.append(None)

    return WorkerChunkResult(results, read_failures)


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

    results = []
    read_failures = 0
    for payload in chunk_payloads:
        try:
            fiber_read = extract_fiber_read_from_payload(payload, mode, prob_threshold)
            if fiber_read is None or fiber_read is CHIMERA_SKIP:
                # CHIMERA_SKIP: DAF strand-swap chimera filtered out (streaming
                # passes it through unannotated; region-parallel tallies a count).
                results.append(None)
                continue
            result = _process_single_read(
                fiber_read, _worker_model, edge_trim, circular,
                mode, context_size, msp_min_size, nuc_min_size=nuc_min_size,
                with_scores=with_scores,
                return_posteriors=return_posteriors,
            )
        except Exception:
            result = None
            read_failures += 1
        results.append(result)

    return WorkerChunkResult(results, read_failures)
