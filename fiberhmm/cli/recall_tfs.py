#!/usr/bin/env python3
"""fiberhmm-recall-tfs CLI  --  [BETA]  LLR-based TF footprint recaller.

*** BETA FEATURE ***
This tool ships as beta: the algorithm, tag schema, and per-enzyme
defaults are stable enough to use, but downstream integrations (fibertools,
FiberBrowser) may still be catching up and calibration outside the
validated enzymes (Hia5 PacBio, DddB DAF, DddA amplicons) has not been
exhaustively tested. Please report surprises on the FiberHMM issue tracker.

Runs as a 2nd pass on a BAM already tagged by ``fiberhmm-apply``.
Writes spec-compliant ``MA``/``AQ`` Molecular-annotation tags
(https://github.com/fiberseq/Molecular-annotation-spec) plus refreshed
legacy ``ns``/``nl``/``as``/``al`` tags reflecting the unified call set.

Supports stdin/stdout piping (``-i -`` / ``-o -``) for composition with
``fiberhmm-apply`` and downstream fibertools stages:

    fiberhmm-apply -o - ... | fiberhmm-recall-tfs -i - -o - --enzyme hia5 | ft fire

By default, v2 short nuc calls (``nl < --unify-threshold``) that overlap
a recaller TF call are demoted out of the ``nuc+`` annotation -- the
recaller version (with proper LLR + edge-ambiguity scoring) replaces
them in the ``tf+`` annotation.

Examples:
  # DddA amplicon BAM (two-pass workflow) -- bundled models, no -m needed
  fiberhmm-apply -i input.bam --enzyme ddda -o tmp/
  fiberhmm-recall-tfs -i tmp/input_footprints.bam -o recalled.bam \\
                       --enzyme ddda -c 8

  # Hia5 streaming composition
  fiberhmm-apply -i input.bam --enzyme hia5 -o - | \\
      fiberhmm-recall-tfs -i - -o recalled.bam --enzyme hia5 -c 8

  # Override with a custom model
  fiberhmm-recall-tfs -i input.bam -o recalled.bam \\
                       -m /path/to/custom.json --min-llr 4.0
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from collections import deque
from dataclasses import dataclass

import pysam

from fiberhmm.cli.common import resolve_core_count
from fiberhmm.cli.model_selection import resolve_model_path as _resolve_cli_model_path
from fiberhmm.cli.recall_config import (
    resolve_recall_defaults as _shared_resolve_recall_defaults,
)
from fiberhmm.cli.recall_config import (
    should_write_legacy_tags,
)
from fiberhmm.core.bam_reader import _has_mm_ml_inputs
from fiberhmm.core.model_io import load_model_with_metadata
from fiberhmm.core.tag_access import compact_ml_value
from fiberhmm.inference.payload_read import PayloadRead
from fiberhmm.inference.recall_tables import (
    build_recall_llr_tables as _shared_build_recall_llr_tables,
)
from fiberhmm.inference.tf_recaller import (
    ENZYME_PRESETS,
    HAS_NUMBA,
    recall_read,
    write_ma_tags,
)
from fiberhmm.io.bam_header import append_coord_marker
from fiberhmm.io.ma_tags import flip_intervals_to_seq

# ---------------------------------------------------------------------------
# Per-worker global state (set by the initializer; avoids repickling arrays)
# ---------------------------------------------------------------------------

_WORKER = {}
_STATS_KEYS = ('v2', 'tf', 'demoted', 'failed')
_PAYLOAD_TAG_NAMES = ('MM', 'Mm', 'ML', 'Ml', 'ns', 'nl', 'as', 'al', 'nq', 'st')


@dataclass(frozen=True)
class _RecallResult:
    tf_calls: object
    kept_nucs: object
    msps: object
    nq_for_kept: object


@dataclass(frozen=True)
class _RecallProcessingSummary:
    n_reads: int
    n_v2: int
    n_tf: int
    n_demoted: int
    n_failed: int


@dataclass(frozen=True)
class _RecallModelConfig:
    model: object
    mode: str
    context_size: int


@dataclass(frozen=True)
class _RecallRuntime:
    model_path: object
    n_cores: int
    min_llr: float
    uplift: float
    model_config: _RecallModelConfig
    llr_hit: object
    llr_miss: object


@dataclass(frozen=True)
class _RecallRuntimeRequest:
    args: object
    model_path: object


@dataclass(frozen=True)
class _RecallStatusSettings:
    enzyme: object
    mode: str
    context_size: int
    min_llr: float
    uplift: float
    unify_threshold: int
    n_cores: int
    has_numba: bool


@dataclass(frozen=True)
class _RecallWorkerConfig:
    llr_hit: object
    llr_miss: object
    mode: str
    k: int
    min_llr: float
    min_opps: int
    unify_threshold: int


@dataclass(frozen=True)
class _RecallSingleThreadRequest:
    bam_in: object
    bam_out: object
    header_text: object
    worker_config: _RecallWorkerConfig
    also_write_legacy: bool
    downstream_compat: bool
    max_reads: int


@dataclass(frozen=True)
class _RecallParallelLoopRequest:
    bam_in: object
    bam_out: object
    header_text: object
    worker_config: _RecallWorkerConfig
    also_write_legacy: bool
    downstream_compat: bool
    max_reads: int
    n_cores: int
    chunk_size: int


@dataclass(frozen=True)
class _RecallProcessingRequest:
    args: object
    n_cores: int
    bam_in: object
    bam_out: object
    header_text: str
    worker_config: _RecallWorkerConfig
    also_write_legacy: bool


@dataclass(frozen=True)
class _RecallApplyResultRequest:
    read: object
    result: object
    also_write_legacy: bool
    downstream_compat: bool


@dataclass(frozen=True)
class _RecallWriteResultRequest:
    read: object
    result: object
    bam_out: object
    also_write_legacy: bool
    downstream_compat: bool


@dataclass(frozen=True)
class _RecallDrainChunkRequest:
    pending: object
    bam_out: object
    also_write_legacy: bool
    downstream_compat: bool


def _new_stats():
    return {key: 0 for key in _STATS_KEYS}


def _failed_stats():
    stats = _new_stats()
    stats['failed'] = 1
    return stats


def _add_stats(total, stats):
    for key in _STATS_KEYS:
        total[key] += stats.get(key, 0)


def _stats_summary(n_reads, stats):
    return _RecallProcessingSummary(
        n_reads=n_reads,
        n_v2=stats['v2'],
        n_tf=stats['tf'],
        n_demoted=stats['demoted'],
        n_failed=stats['failed'],
    )


def _process_payload_safely(payload):
    try:
        return _process_payload_record(payload)
    except Exception:
        return None, _failed_stats()


def _kept_nuc_nq_from_legacy(tags: dict, read, kept_nucs):
    v2_nq = tags.get('nq', None)
    if v2_nq is None or 'ns' not in tags or 'nl' not in tags:
        return None
    try:
        # kept_nucs come back from recall_read() in SEQ (query) frame, so the
        # nq lookup must key on SEQ-frame intervals too. The stored ns/nl are
        # molecular; flip them before building the lookup.
        ns_seq, nl_seq = flip_intervals_to_seq(tags['ns'], tags['nl'], read)
        old_to_nq = {
            (int(s), int(length)): int(v2_nq[i])
            for i, (s, length) in enumerate(zip(ns_seq, nl_seq))
            if i < len(v2_nq)
        }
        return [old_to_nq.get((s, length), 0) for s, length in kept_nucs]
    except Exception:
        return None


def _worker_init_from_config(config: _RecallWorkerConfig) -> None:
    """Set per-process globals once per worker.

    Slim version: workers receive compact payloads and return compact results —
    no pysam header, no SAM serialization inside the worker.
    """
    _WORKER['llr_hit'] = config.llr_hit
    _WORKER['llr_miss'] = config.llr_miss
    _WORKER['mode'] = config.mode
    _WORKER['k'] = config.k
    _WORKER['min_llr'] = config.min_llr
    _WORKER['min_opps'] = config.min_opps
    _WORKER['unify_threshold'] = config.unify_threshold


def _worker_init(llr_hit, llr_miss, mode, k, min_llr, min_opps, unify_threshold):
    _worker_init_from_config(
        _RecallWorkerConfig(
            llr_hit=llr_hit,
            llr_miss=llr_miss,
            mode=mode,
            k=k,
            min_llr=min_llr,
            min_opps=min_opps,
            unify_threshold=unify_threshold,
        ),
    )


# ---------------------------------------------------------------------------
# Slim IPC: compact payload helpers (no SAM text serialization in the hot path)
# ---------------------------------------------------------------------------


def _payload_tag_value(tag, value):
    if tag in ('ML', 'Ml'):
        # array.array('B', ...) -> bytes via buffer protocol: fast memcpy
        return compact_ml_value(value)
    return value


def _payload_tags(read) -> dict:
    return {
        tag: _payload_tag_value(tag, read.get_tag(tag))
        for tag in _PAYLOAD_TAG_NAMES
        if read.has_tag(tag)
    }


def _has_mm_ml_tags(tags: dict) -> bool:
    mm_tag = tags.get('MM', tags.get('Mm', ''))
    ml_tag = tags.get('ML', tags.get('Ml', None))
    return _has_mm_ml_inputs(mm_tag, ml_tag)


def _needs_daf_md_result(seq, tags: dict, mode) -> bool:
    if mode != 'daf' or not seq:
        return False
    from fiberhmm.core.bam_reader import has_iupac_encoding
    return not has_iupac_encoding(seq) and not _has_mm_ml_tags(tags)


def _short_v2_nuc_count(tags: dict, unify_threshold: int) -> int:
    if 'ns' not in tags or 'nl' not in tags:
        return 0
    return sum(1 for length in tags['nl'] if 0 < int(length) < unify_threshold)


def _record_recall_stats(
    stats: dict,
    tf_calls,
    kept_nucs,
    v2_short_count: int,
    unify_threshold: int,
) -> None:
    stats['tf'] = len(tf_calls)
    survived_short = sum(1 for _, length in kept_nucs if length < unify_threshold)
    stats['demoted'] = max(0, v2_short_count - survived_short)


def _record_v2_input_stats(stats: dict, tags: dict, unify_threshold: int) -> int:
    if 'ns' not in tags or 'nl' not in tags:
        return 0
    stats['v2'] = 1
    return _short_v2_nuc_count(tags, unify_threshold)


def _make_payload(read, mode=None) -> dict:
    """Extract only the tag data workers need from a pysam read.

    Runs in the main process.  The resulting dict is ~5–30 KB (sequence
    string + small tag arrays) versus ~50–100 KB for a full SAM string with
    MM/ML for a 20 kb PacBio read.

    ML is stored as raw bytes (not a Python list) — for a PacBio read with
    ~5000 modification probabilities this avoids creating 5000 Python int
    objects in the serial main process (~1–2 ms/read saved), and pickle of
    bytes is a straight memcpy vs. pickling a Python list.
    parse_mm_tag_query_positions accepts bytes directly via np.frombuffer.
    """
    tags = _payload_tags(read)
    payload = {
        'seq': read.query_sequence,
        'is_reverse': read.is_reverse,
        'tags': tags,
    }
    if _needs_daf_md_result(read.query_sequence, tags, mode):
        from fiberhmm.daf.encoder import get_daf_positions
        md_result = get_daf_positions(read)
        if md_result is not None:
            payload['_daf_md_result'] = md_result
    return payload


def _process_payload_record(payload) -> tuple:
    """Worker: compute TF calls from a compact payload.

    Returns (_RecallResult, stats).
    No pysam SAM serialization — only recall_read() + Python arithmetic.
    write_ma_tags() is intentionally left to the main process so the
    serialized return value stays small (<1 KB for typical call counts).
    """
    read = PayloadRead(
        payload['seq'],
        payload['is_reverse'],
        payload['tags'],
        daf_md_result=payload.get('_daf_md_result'),
    )
    tags = payload['tags']
    stats = _new_stats()
    unify_threshold = _WORKER['unify_threshold']

    v2_short_count = _record_v2_input_stats(stats, tags, unify_threshold)

    tf_calls, kept_nucs, msps = recall_read(
        read,
        _WORKER['llr_hit'], _WORKER['llr_miss'],
        _WORKER['mode'], _WORKER['k'],
        min_llr=_WORKER['min_llr'],
        min_opps=_WORKER['min_opps'],
        unify_threshold=unify_threshold,
    )
    _record_recall_stats(stats, tf_calls, kept_nucs, v2_short_count, unify_threshold)

    nq_for_kept = _kept_nuc_nq_from_legacy(tags, read, kept_nucs)

    return _RecallResult(tf_calls, kept_nucs, msps, nq_for_kept), stats


def _process_payload_chunk(payloads):
    """Worker: process a list of compact payloads."""
    out = []
    total = _new_stats()
    for payload in payloads:
        result, stats = _process_payload_safely(payload)
        out.append(result)
        _add_stats(total, stats)
    return out, total


def _read_sequence_length(read) -> int:
    return len(read.query_sequence) if read.query_sequence else 0


def _apply_result_from_request(
    request: _RecallApplyResultRequest,
):
    """Apply compact worker result to a pysam read in place (main process)."""
    write_ma_tags(
        request.read,
        read_length=_read_sequence_length(request.read),
        tf_calls=request.result.tf_calls,
        kept_nucs=request.result.kept_nucs,
        msps=request.result.msps,
        nq_for_kept_nucs=request.result.nq_for_kept,
        also_write_legacy=request.also_write_legacy,
        downstream_compat=request.downstream_compat,
    )


def _apply_result(read, result, also_write_legacy, downstream_compat):
    _apply_result_from_request(
        _RecallApplyResultRequest(
            read=read,
            result=result,
            also_write_legacy=also_write_legacy,
            downstream_compat=downstream_compat,
        )
    )


def _write_recall_result_from_request(
    request: _RecallWriteResultRequest,
):
    if request.result is not None:
        _apply_result_from_request(
            _RecallApplyResultRequest(
                read=request.read,
                result=request.result,
                also_write_legacy=request.also_write_legacy,
                downstream_compat=request.downstream_compat,
            )
        )
    request.bam_out.write(request.read)


def _write_recall_result(
    read,
    result,
    bam_out,
    also_write_legacy,
    downstream_compat,
):
    _write_recall_result_from_request(
        _RecallWriteResultRequest(
            read=read,
            result=result,
            bam_out=bam_out,
            also_write_legacy=also_write_legacy,
            downstream_compat=downstream_compat,
        )
    )


def _single_thread_loop_from_request(
    request: _RecallSingleThreadRequest,
):
    """Single-threaded path.  No IPC — process reads directly."""
    _worker_init_from_config(request.worker_config)
    n_reads = 0
    total = _new_stats()
    for read in request.bam_in:
        if request.max_reads and n_reads >= request.max_reads:
            break
        result, stats = _process_payload_safely(
            _make_payload(read, request.worker_config.mode),
        )
        _write_recall_result_from_request(
            _RecallWriteResultRequest(
                read=read,
                result=result,
                bam_out=request.bam_out,
                also_write_legacy=request.also_write_legacy,
                downstream_compat=request.downstream_compat,
            )
        )
        n_reads += 1
        _add_stats(total, stats)
    return _stats_summary(n_reads, total)


def _single_thread_loop(bam_in, bam_out, header_text,
                        llr_hit, llr_miss, mode, k,
                        min_llr, min_opps, unify_threshold,
                        also_write_legacy, downstream_compat, max_reads):
    return _single_thread_loop_from_request(
        _RecallSingleThreadRequest(
            bam_in=bam_in,
            bam_out=bam_out,
            header_text=header_text,
            worker_config=_RecallWorkerConfig(
                llr_hit=llr_hit,
                llr_miss=llr_miss,
                mode=mode,
                k=k,
                min_llr=min_llr,
                min_opps=min_opps,
                unify_threshold=unify_threshold,
            ),
            also_write_legacy=also_write_legacy,
            downstream_compat=downstream_compat,
            max_reads=max_reads,
        ),
    )


def _submit_recall_chunk(pool, pending, reads_chunk, payloads_chunk):
    future = pool.apply_async(_process_payload_chunk, (payloads_chunk,))
    pending.append((reads_chunk, future))


def _drain_recall_chunk_from_request(
    request: _RecallDrainChunkRequest,
):
    reads_chunk, fut = request.pending.popleft()
    out_results, stats = fut.get()   # blocks until result is ready
    for read, result in zip(reads_chunk, out_results):
        _write_recall_result_from_request(
            _RecallWriteResultRequest(
                read=read,
                result=result,
                bam_out=request.bam_out,
                also_write_legacy=request.also_write_legacy,
                downstream_compat=request.downstream_compat,
            )
        )
    return len(reads_chunk), stats


def _drain_recall_chunk(pending, bam_out, also_write_legacy, downstream_compat):
    return _drain_recall_chunk_from_request(
        _RecallDrainChunkRequest(
            pending=pending,
            bam_out=bam_out,
            also_write_legacy=also_write_legacy,
            downstream_compat=downstream_compat,
        )
    )


def _parallel_loop_from_request(
    request: _RecallParallelLoopRequest,
):
    """Multi-core path with slim IPC and bounded in-flight queue.

    Uses apply_async + a bounded deque instead of imap to cap how many chunks
    are held in memory simultaneously.  When downstream (disk write, sort
    backpressure) is slow, the submission loop blocks on the oldest pending
    result before submitting another chunk, so RSS stays bounded regardless
    of downstream speed.

    In-flight cap: n_cores + 2 chunks.  At chunk_size=1024 and n_cores=4,
    the worst-case working set is ~6 * 1024 pysam reads ≈ 180 MB, not 52 GB.
    """
    config = request.worker_config
    MAX_INFLIGHT = request.n_cores + 2   # workers + small head-start headroom
    pending: deque = deque()     # deque of (reads_chunk, AsyncResult)

    n_reads = 0
    total = _new_stats()

    def _drain_one():
        nonlocal n_reads
        chunk_reads, stats = _drain_recall_chunk_from_request(
            _RecallDrainChunkRequest(
                pending=pending,
                bam_out=request.bam_out,
                also_write_legacy=request.also_write_legacy,
                downstream_compat=request.downstream_compat,
            )
        )
        n_reads += chunk_reads
        _add_stats(total, stats)

    with mp.Pool(
        processes=request.n_cores,
        initializer=_worker_init_from_config,
        initargs=(config,),
    ) as pool:
        buf_reads: list = []
        buf_payloads: list = []
        n = 0

        for read in request.bam_in:
            if request.max_reads and n >= request.max_reads:
                break
            buf_reads.append(read)
            buf_payloads.append(_make_payload(read, config.mode))
            n += 1

            if len(buf_reads) >= request.chunk_size:
                _submit_recall_chunk(pool, pending, buf_reads, buf_payloads)
                buf_reads, buf_payloads = [], []

                # Backpressure: drain the oldest result before submitting more
                if len(pending) >= MAX_INFLIGHT:
                    _drain_one()

        # Submit last partial chunk
        if buf_reads:
            _submit_recall_chunk(pool, pending, buf_reads, buf_payloads)

        # Drain all remaining results
        while pending:
            _drain_one()

    return _stats_summary(n_reads, total)


def _parallel_loop(bam_in, bam_out, header_text,
                   llr_hit, llr_miss, mode, k,
                   min_llr, min_opps, unify_threshold,
                   also_write_legacy, downstream_compat,
                   max_reads, n_cores, chunk_size):
    return _parallel_loop_from_request(
        _RecallParallelLoopRequest(
            bam_in=bam_in,
            bam_out=bam_out,
            header_text=header_text,
            worker_config=_RecallWorkerConfig(
                llr_hit=llr_hit,
                llr_miss=llr_miss,
                mode=mode,
                k=k,
                min_llr=min_llr,
                min_opps=min_opps,
                unify_threshold=unify_threshold,
            ),
            also_write_legacy=also_write_legacy,
            downstream_compat=downstream_compat,
            max_reads=max_reads,
            n_cores=n_cores,
            chunk_size=chunk_size,
        ),
    )


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('-i', '--in-bam', required=True,
                   help='Input BAM tagged by fiberhmm-apply (has ns/nl/as/al). '
                        'Use "-" for stdin.')
    p.add_argument('-o', '--out-bam', required=True,
                   help='Output BAM with MA/AQ + refreshed legacy tags. '
                        'Use "-" for stdout (for piping to ft fire, samtools, etc).')
    p.add_argument('-m', '--model', default=None,
                   help='FiberHMM model JSON. If omitted, the bundled model '
                        'for --enzyme/--seq is used automatically.')
    p.add_argument('--enzyme', choices=sorted(ENZYME_PRESETS.keys()),
                   default=None,
                   help='Enzyme preset: auto-selects the bundled model and '
                        f'min-llr/emission-uplift defaults '
                        f'({", ".join(sorted(ENZYME_PRESETS))}).')
    p.add_argument('--seq', choices=['pacbio', 'nanopore'], default=None,
                   help='Sequencing platform. Required for hia5 '
                        '(pacbio or nanopore); ignored for dddb/ddda.')
    p.add_argument('--min-llr', type=float, default=None,
                   help='Override min LLR (nats). Default: enzyme preset.')
    p.add_argument('--min-opps', type=int, default=3,
                   help='Min informative target positions per call (default 3)')
    p.add_argument('--emission-uplift', type=float, default=None,
                   help='Power transform on emission probabilities. Default 1.0 '
                        '(identity). Use a pre-uplifted model file (e.g. '
                        'ddda_TF.json) for DddA rather than setting this.')
    p.add_argument('--unify-threshold', type=int, default=90,
                   help='v2 nucs with nl < this are scanned + may be demoted '
                        'to tf+ if overlapped by a recaller call (default 90)')
    p.add_argument('--no-legacy-tags', action='store_true',
                   help='Skip refreshed ns/nl/as/al -- emit only MA/AQ.')
    p.add_argument('--downstream-compat', action='store_true',
                   help='Downstream-compatibility mode: skip MA/AQ entirely '
                        'and write TF calls INTO the legacy ns/nl tag '
                        'alongside nucleosomes (sorted by start). Use for '
                        'older tools that do not understand the '
                        'Molecular-annotation spec. Loses per-TF quality '
                        'scoring (tq/el/er) -- positions and lengths only.')
    p.add_argument('-c', '--cores', type=int, default=1,
                   help='Worker processes. 0 = auto-detect (default 1).')
    p.add_argument('--chunk-size', type=int, default=1024,
                   help='Reads per worker chunk (default 1024). '
                        'Larger values reduce IPC overhead; decrease if RAM '
                        'is constrained (each chunk holds reads in memory).')
    p.add_argument('--io-threads', type=int, default=4,
                   help='htslib BAM compression threads (default 4).')
    p.add_argument('--mode', default=None,
                   help='Override observation mode. Default: read from model.')
    p.add_argument('--context-size', type=int, default=None,
                   help='Override context size. Default: read from model.')
    p.add_argument('--max-reads', type=int, default=0,
                   help='0 = no limit (default)')
    return p.parse_args()


def _resolve_model_metadata(model_path):
    mode = 'pacbio-fiber'
    k = 3
    model_path = os.fspath(model_path)
    if model_path.lower().endswith('.json'):
        try:
            with open(model_path) as f:
                d = json.load(f)
            mode_value = d.get('mode')
            if mode_value is not None:
                mode = str(mode_value).strip() or mode
            context_size = d.get('context_size')
            if context_size is not None:
                k = int(context_size)
        except (OSError, TypeError, ValueError):
            pass
    return mode, k


def _resolve_model_path(args):
    """Resolve the recall model path from --model or bundled --enzyme/--seq."""
    return _resolve_cli_model_path(
        args,
        tool='recall',
        bundled_message="[recall_tfs] using bundled model: {model_path}",
        bundled_message_file=sys.stderr,
    )


def _print_beta_banner(downstream_compat):
    # Prominent beta banner: this feature is new, expect rough edges.
    if downstream_compat:
        mode_banner = (
            "  MODE: DOWNSTREAM-COMPAT -- TF calls written into legacy ns/nl.\n"
            "  MA/AQ spec tags are NOT emitted. Per-TF quality scoring is lost.\n"
            "  Use this only for older tools that cannot read the MA/AQ spec.\n"
        )
    else:
        mode_banner = (
            "  MODE: SPEC -- MA/AQ tags emitted per the fiberseq Molecular-\n"
            "  annotation spec (tf+QQQ carries LLR + edge-ambiguity scores).\n"
            "  -> Update FiberBrowser to the MA/AQ-aware release to visualize.\n"
            "  -> Read the spec: https://github.com/fiberseq/Molecular-annotation-spec\n"
            "  -> For tools that do not yet speak MA/AQ, re-run with\n"
            "     --downstream-compat to put TF calls into legacy ns/nl instead.\n"
        )
    print(
        "\n"
        "========================================================================\n"
        "  fiberhmm-recall-tfs  [BETA]\n"
        "  LLR TF footprint recaller -- beta feature shipped in fiberhmm 2.6.0.\n"
        "  Defaults validated on Hia5 PacBio, DddB DAF, and DddA amplicons.\n"
        "\n"
        + mode_banner +
        "\n"
        "  File issues at https://github.com/fiberseq/FiberHMM/issues\n"
        "========================================================================",
        file=sys.stderr,
    )


def _resolve_cores(requested_cores):
    return resolve_core_count(requested_cores, mp.cpu_count)


def _resolve_recall_defaults(args):
    return _shared_resolve_recall_defaults(args, ENZYME_PRESETS)


def _load_recall_model_config(model_path, args):
    model, model_k, model_mode = load_model_with_metadata(model_path)
    if not model_mode or model_k is None:
        fb_mode, fb_k = _resolve_model_metadata(model_path)
        model_mode = model_mode or fb_mode
        model_k = fb_k if model_k is None else model_k
    arg_mode = str(args.mode).strip() if args.mode is not None else None
    model_mode = str(model_mode).strip() if model_mode is not None else None
    mode = arg_mode or model_mode or 'pacbio-fiber'
    k = (
        int(args.context_size)
        if args.context_size is not None
        else int(model_k if model_k is not None else 3)
    )
    return _RecallModelConfig(model, mode, k)


def _build_recall_llr_tables(model, uplift):
    return _shared_build_recall_llr_tables(model, uplift)


def _resolve_recall_runtime_from_request(
    request: _RecallRuntimeRequest,
) -> _RecallRuntime:
    args = request.args
    n_cores = _resolve_cores(args.cores)
    min_llr, uplift = _resolve_recall_defaults(args)
    model_config = _load_recall_model_config(request.model_path, args)
    llr_hit, llr_miss = _build_recall_llr_tables(model_config.model, uplift)
    return _RecallRuntime(
        model_path=request.model_path,
        n_cores=n_cores,
        min_llr=min_llr,
        uplift=uplift,
        model_config=model_config,
        llr_hit=llr_hit,
        llr_miss=llr_miss,
    )


def _resolve_recall_runtime(args, model_path) -> _RecallRuntime:
    return _resolve_recall_runtime_from_request(
        _RecallRuntimeRequest(args=args, model_path=model_path),
    )


def _recall_status_settings(
    args,
    runtime: _RecallRuntime,
) -> _RecallStatusSettings:
    return _RecallStatusSettings(
        enzyme=args.enzyme,
        mode=runtime.model_config.mode,
        context_size=runtime.model_config.context_size,
        min_llr=runtime.min_llr,
        uplift=runtime.uplift,
        unify_threshold=args.unify_threshold,
        n_cores=runtime.n_cores,
        has_numba=HAS_NUMBA,
    )


def _recall_status_message(settings: _RecallStatusSettings) -> str:
    return (
        f"[recall_tfs] enzyme={settings.enzyme or 'custom'} "
        f"mode={settings.mode} k={settings.context_size} "
        f"min_llr={settings.min_llr:.2f} uplift={settings.uplift:.2f} "
        f"unify_threshold={settings.unify_threshold} cores={settings.n_cores} "
        f"numba={'on' if settings.has_numba else 'off'}"
    )


def _also_write_legacy(args):
    return should_write_legacy_tags(args)


def _run_recall_processing_from_request(
    request: _RecallProcessingRequest,
):
    args = request.args
    config = request.worker_config
    if request.n_cores == 1:
        return _single_thread_loop_from_request(
            _RecallSingleThreadRequest(
                bam_in=request.bam_in,
                bam_out=request.bam_out,
                header_text=request.header_text,
                worker_config=config,
                also_write_legacy=request.also_write_legacy,
                downstream_compat=args.downstream_compat,
                max_reads=args.max_reads,
            ),
        )
    return _parallel_loop_from_request(
        _RecallParallelLoopRequest(
            bam_in=request.bam_in,
            bam_out=request.bam_out,
            header_text=request.header_text,
            worker_config=config,
            also_write_legacy=request.also_write_legacy,
            downstream_compat=args.downstream_compat,
            max_reads=args.max_reads,
            n_cores=request.n_cores,
            chunk_size=args.chunk_size,
        ),
    )


def _run_recall_processing(
    args,
    n_cores: int,
    bam_in,
    bam_out,
    header_text: str,
    llr_hit,
    llr_miss,
    mode: str,
    k: int,
    min_llr: float,
    also_write_legacy: bool,
):
    return _run_recall_processing_from_request(
        _RecallProcessingRequest(
            args=args,
            n_cores=n_cores,
            bam_in=bam_in,
            bam_out=bam_out,
            header_text=header_text,
            worker_config=_RecallWorkerConfig(
                llr_hit=llr_hit,
                llr_miss=llr_miss,
                mode=mode,
                k=k,
                min_llr=min_llr,
                min_opps=args.min_opps,
                unify_threshold=args.unify_threshold,
            ),
            also_write_legacy=also_write_legacy,
        ),
    )


def _print_recall_summary(summary: _RecallProcessingSummary) -> None:
    print(
        f"[recall_tfs] processed {summary.n_reads} reads; "
        f"{summary.n_v2} carried v2 tags; "
        f"{summary.n_tf} TF calls emitted; "
        f"{summary.n_demoted} v2 short nucs demoted to tf+",
        file=sys.stderr,
    )
    if summary.n_failed:
        print(
            f"[recall_tfs] warning: {summary.n_failed} reads passed through unchanged "
            "after recall errors",
            file=sys.stderr,
        )


def main():
    args = parse_args()

    stdout_mode = (args.out_bam == '-')
    if stdout_mode:
        # Redirect informational prints to stderr so BAM stream on stdout stays clean
        sys.stdout = sys.stderr

    model_path = _resolve_model_path(args)

    _print_beta_banner(args.downstream_compat)

    runtime = _resolve_recall_runtime(args, model_path)

    print(_recall_status_message(_recall_status_settings(args, runtime)),
          file=sys.stderr)

    # Open BAMs with io-threads. pysam accepts "-" as stdin/stdout natively.
    bam_in = pysam.AlignmentFile(args.in_bam, 'rb',
                                  check_sq=False,
                                  threads=args.io_threads)
    bam_out = None
    try:
        bam_out = pysam.AlignmentFile(args.out_bam, 'wb',
                                       header=append_coord_marker(bam_in.header),
                                       threads=args.io_threads)
        header_text = str(bam_in.header)
        also_write_legacy = _also_write_legacy(args)

        summary = _run_recall_processing(
            args,
            runtime.n_cores,
            bam_in,
            bam_out,
            header_text,
            runtime.llr_hit,
            runtime.llr_miss,
            runtime.model_config.mode,
            runtime.model_config.context_size,
            runtime.min_llr,
            also_write_legacy,
        )
    finally:
        bam_in.close()
        if bam_out is not None:
            bam_out.close()

    _print_recall_summary(summary)


if __name__ == '__main__':
    main()
