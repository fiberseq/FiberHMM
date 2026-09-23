#!/usr/bin/env python3
"""fiberhmm-recall-tfs CLI  --  [BETA]  LLR-based TF footprint recaller.

*** BETA FEATURE ***
This tool ships as beta: the algorithm, tag schema, and per-enzyme
defaults are stable enough to use, but downstream integrations (fibertools,
FiberBrowser) may still be catching up and calibration outside the
validated enzymes (Hia5 PacBio, DddB DAF, DddA DAF-seq) has not been
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

``--recall-nucs`` (also the ``fiberhmm-recall-nucs`` entry point) adds the
per-read nucleosome recaller BEFORE TF recall: it splits over-merged HMM
footprints on accessible evidence and refines each nucleosome's conservative
edges + quality (nuc+QQQ), re-derives MSPs, then runs TF recall over the
cleaner accessible space. It reuses the apply-tagged ``ns``/``nl``/``as``/``al``
-- the HMM is NOT re-run -- so its footprint tags are equivalent to
``fiberhmm-call --recall-nucs`` for a given ``--phase-nrl`` (linear reads only;
``--phase-nrl auto`` is estimated from the existing nuc tags). The BAM headers
retain distinct command provenance.

Examples:
  # Add nucleosome recall to an apply-tagged BAM (no HMM re-run)
  fiberhmm-recall-nucs -i apply_footprints.bam -o recalled.bam \\
                        --enzyme hia5 --seq pacbio -c 8
  # equivalent: fiberhmm-recall-tfs --recall-nucs ...

  # DddA DAF-seq BAM (two-pass workflow) -- bundled models, no -m needed
  fiberhmm-apply -i input.bam --enzyme ddda -o tmp/
  fiberhmm-recall-nucs -i tmp/input_footprints.bam -o recalled.bam \\
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
import sys
from collections import deque, namedtuple

import pysam

from fiberhmm.cli.common import (
    add_legacy_mode_override,
    resolve_observation_mode,
)
from fiberhmm.cli.provenance import (
    chemistry_declaration,
    nuc_profile_identity,
    nuc_profile_sha256,
)
from fiberhmm.core.bam_reader import encode_from_query_sequence
from fiberhmm.core.model_io import load_model_with_metadata
from fiberhmm.io.bam_header import (
    append_coord_marker,
    header_has_coord_marker,
    maybe_append_pg,
)
from fiberhmm.io.ma_tags import flip_intervals_to_seq
from fiberhmm.inference.fused_stages import build_fused_recall_result
from fiberhmm.inference.tagging import write_fused_recall_tags
from fiberhmm.inference.tf_recaller import (
    ENZYME_PRESETS,
    TF_DECODER_VERSION,
    HAS_NUMBA,
    apply_emission_uplift,
    build_conditional_hit_tables,
    build_cpg_mask,
    build_llr_tables,
    build_m5c_llr_tables,
    extract_modifications,
    recall_read,
    write_ma_tags,
)
# ---------------------------------------------------------------------------
# Per-worker global state (set by the initializer; avoids repickling arrays)
# ---------------------------------------------------------------------------

_WORKER = {}
_STATS_KEYS = ('v2', 'tf', 'demoted', 'failed')

# Nucleosome-recall config (None = TF-only, the default behavior). A picklable
# namedtuple so it survives the spawn-start multiprocessing pool initializer.
_NucCfg = namedtuple(
    '_NucCfg',
    ('recall_nucs', 'split_min_llr', 'split_min_opps',
     'nuc_min_size', 'msp_min_size', 'phase_nrl', 'nuc_recall_policy',
     'nuc_profile_path', 'derived_tf_max_edge_ambiguity'),
)
_NucCfg.__new__.__defaults__ = ('conservative', None, None)


def _build_recall_pg_record(args, mode, model_path, nuc_cfg):
    """Build command and exact nucleosome-profile provenance for output BAM."""
    profile_path = (
        getattr(nuc_cfg, 'nuc_profile_path', None)
        if nuc_cfg is not None else None
    )
    profile_identity = nuc_profile_identity(profile_path)
    profile_sha256 = nuc_profile_sha256(profile_path)
    recall_nucs = bool(nuc_cfg is not None and nuc_cfg.recall_nucs)
    program_name = (
        'fiberhmm-recall-nucs' if recall_nucs else 'fiberhmm-recall-tfs'
    )
    policy = (
        getattr(nuc_cfg, 'nuc_recall_policy', 'off')
        if recall_nucs else 'off'
    )
    phase_nrl = getattr(nuc_cfg, 'phase_nrl', 'off') if recall_nucs else 'off'

    import fiberhmm as _fh

    return {
        'PN': program_name,
        'VN': getattr(_fh, '__version__', 'unknown'),
        'CL': ' '.join(sys.argv),
        'chemistry': chemistry_declaration(
            args,
            mode,
            model_path,
            None,
            profile_identity,
            profile_sha256,
        ),
        'DS': (
            'FiberHMM second-pass footprint refinement; coord=molecular '
            '(ns/nl/as/al/MA in molecular original-fiber coordinates); '
            f'mode={mode} enzyme={args.enzyme or "custom"} '
            f'tf_decoder={TF_DECODER_VERSION} '
            f'recall_nucs={recall_nucs} nuc_recall_policy={policy} '
            f'nuc_profile={profile_identity or "off"} '
            f'nuc_sha256={profile_sha256 or "off"} phase_nrl={phase_nrl}'
        ),
    }


def _worker_init(llr_hit, llr_miss, mode, k, min_llr, min_opps, unify_threshold,
                 nuc_cfg=None, input_molecular_frame=True,
                 m5c_llr_hit=None, m5c_llr_miss=None,
                 cpg_mask_policy="unmethylated-only",
                 nuc_protected_hit=None, nuc_accessible_hit=None,
                 nuc_llr_hit=None, nuc_llr_miss=None,
                 nuc_m5c_llr_hit=None, nuc_m5c_llr_miss=None):
    """Set per-process globals once per worker.

    Slim version: workers receive compact payloads and return compact results —
    no pysam header, no SAM serialization inside the worker.
    """
    _WORKER['llr_hit'] = llr_hit
    _WORKER['llr_miss'] = llr_miss
    _WORKER['mode'] = mode
    _WORKER['k'] = k
    _WORKER['min_llr'] = min_llr
    _WORKER['min_opps'] = min_opps
    _WORKER['unify_threshold'] = unify_threshold
    _WORKER['nuc_cfg'] = nuc_cfg
    _WORKER['nuc_profile'] = None
    if nuc_cfg is not None and nuc_cfg.nuc_profile_path:
        from fiberhmm.inference.nuc_recaller import (
            attach_nuc_profile_emissions,
            load_nuc_profile,
        )
        profile = load_nuc_profile(nuc_cfg.nuc_profile_path)
        if nuc_protected_hit is not None and nuc_accessible_hit is not None:
            profile = attach_nuc_profile_emissions(
                profile, nuc_protected_hit, nuc_accessible_hit,
            )
        _WORKER['nuc_profile'] = profile
    # Frame of the input ns/nl/as/al: molecular (current FiberHMM, flip reverse
    # tags to seq) vs legacy seq/query (v1.0, use as-is). See recall_read().
    _WORKER['input_molecular_frame'] = input_molecular_frame
    _WORKER['m5c_llr_hit'] = m5c_llr_hit
    _WORKER['m5c_llr_miss'] = m5c_llr_miss
    _WORKER['cpg_mask_policy'] = cpg_mask_policy
    _WORKER['nuc_llr_hit'] = nuc_llr_hit
    _WORKER['nuc_llr_miss'] = nuc_llr_miss
    _WORKER['nuc_m5c_llr_hit'] = nuc_m5c_llr_hit
    _WORKER['nuc_m5c_llr_miss'] = nuc_m5c_llr_miss


# ---------------------------------------------------------------------------
# Slim IPC: compact payload helpers (no SAM text serialization in the hot path)
# ---------------------------------------------------------------------------

class _PayloadRead:
    """Minimal duck-type for pysam.AlignedSegment used inside worker processes.

    Workers receive a compact dict extracted by _make_payload() in the main
    process instead of a full SAM string.  This avoids four
    to_string()/fromstring() calls per read (two in the main process, two in
    the worker) and the associated MM/ML base64 encoding overhead.
    """
    __slots__ = ('query_sequence', 'is_reverse', '_tags', '_daf_md_result')

    def __init__(self, seq, is_reverse, tags, daf_md_result=None):
        self.query_sequence = seq
        self.is_reverse = is_reverse
        self._tags = tags
        self._daf_md_result = daf_md_result

    def has_tag(self, t):
        return t in self._tags

    def get_tag(self, t):
        return self._tags[t]

    @property
    def query_length(self):
        return len(self.query_sequence) if self.query_sequence else 0


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
    tags = {}
    for t in ('MM', 'Mm', 'ML', 'Ml', 'ns', 'nl', 'as', 'al', 'nq', 'st', 'MA'):
        if read.has_tag(t):
            val = read.get_tag(t)
            if t in ('ML', 'Ml'):
                # array.array('B', ...) → bytes via buffer protocol: fast memcpy
                try:
                    val = bytes(val)
                except TypeError:
                    pass  # scalar or already bytes
            tags[t] = val

    payload = {
        'seq': read.query_sequence,
        'is_reverse': read.is_reverse,
        'tags': tags,
    }
    if mode == 'daf' and read.query_sequence:
        from fiberhmm.core.bam_reader import has_iupac_encoding
        if (
            not has_iupac_encoding(read.query_sequence)
            and not (('MM' in tags or 'Mm' in tags) and ('ML' in tags or 'Ml' in tags))
        ):
            from fiberhmm.daf.encoder import get_daf_positions
            md_result = get_daf_positions(read)
            if md_result is not None:
                payload['_daf_md_result'] = md_result
    return payload


def _process_payload_record(payload) -> tuple:
    """Worker: compute TF calls from a compact payload.

    Returns ((tf_calls, kept_nucs, msps, nq_for_kept), stats).
    No pysam SAM serialization — only recall_read() + Python arithmetic.
    write_ma_tags() is intentionally left to the main process so the
    serialized return value stays small (<1 KB for typical call counts).
    """
    read = _PayloadRead(
        payload['seq'],
        payload['is_reverse'],
        payload['tags'],
        payload.get('_daf_md_result'),
    )
    nuc_cfg = _WORKER.get('nuc_cfg')
    if nuc_cfg is not None and nuc_cfg.recall_nucs:
        return _process_nuc_payload_record(read, payload, nuc_cfg)
    tags = payload['tags']
    stats = {key: 0 for key in _STATS_KEYS}
    unify_threshold = _WORKER['unify_threshold']

    has_ns = 'ns' in tags and 'nl' in tags
    if has_ns:
        stats['v2'] = 1
        nl_list = tags['nl']
        v2_short_count = sum(1 for length in nl_list if 0 < int(length) < unify_threshold)
        v2_nq = tags.get('nq', None)
    else:
        v2_short_count = 0
        v2_nq = None

    tf_calls, kept_nucs, msps = recall_read(
        read,
        _WORKER['llr_hit'], _WORKER['llr_miss'],
        _WORKER['mode'], _WORKER['k'],
        min_llr=_WORKER['min_llr'],
        min_opps=_WORKER['min_opps'],
        unify_threshold=unify_threshold,
        input_molecular_frame=_WORKER.get('input_molecular_frame', True),
        m5c_llr_hit=_WORKER.get('m5c_llr_hit'),
        m5c_llr_miss=_WORKER.get('m5c_llr_miss'),
        cpg_mask_policy=_WORKER.get('cpg_mask_policy', 'unmethylated-only'),
    )
    stats['tf'] = len(tf_calls)
    survived_short = sum(1 for _, length in kept_nucs if length < unify_threshold)
    stats['demoted'] = max(0, v2_short_count - survived_short)

    nq_for_kept = None
    if v2_nq is not None and has_ns:
        try:
            # kept_nucs come back from recall_read() in SEQ (query) frame, so the
            # nq lookup must key on SEQ-frame intervals too. The stored ns/nl are
            # molecular; flip them (order-preserving, so v2_nq[i] stays aligned)
            # before building the lookup -- otherwise reverse-read nucs miss their
            # original score and get 0.
            from fiberhmm.io.ma_tags import flip_intervals_to_seq
            if _WORKER.get('input_molecular_frame', True):
                ns_seq, nl_seq = flip_intervals_to_seq(tags['ns'], tags['nl'], read)
            else:
                # Legacy seq/query-frame input: kept_nucs come back unflipped,
                # so key the nq lookup on the original (seq-frame) intervals.
                ns_seq, nl_seq = tags['ns'], tags['nl']
            old_to_nq = {(int(s), int(length)): int(v2_nq[i])
                         for i, (s, length) in enumerate(zip(ns_seq, nl_seq))
                         if i < len(v2_nq)}
            nq_for_kept = [old_to_nq.get((s, length), 0) for s, length in kept_nucs]
        except Exception:
            nq_for_kept = None

    return (tf_calls, kept_nucs, msps, nq_for_kept), stats


def _process_nuc_payload_record(read, payload, nuc_cfg) -> tuple:
    """Worker (--recall-nucs): full nucleosome recall from an apply-tagged read.

    Reconstructs the per-base observation array from the read's own MM/ML+seq
    (exactly as the TF recaller does -- no HMM re-run) and reuses the fused
    stage: nuc recall -> MSP re-derive -> TF recall -> promotion. Returns a
    5-tuple (the 5th element is the fused result dict) so the writer routes it
    through write_fused_recall_tags and nuc QQQ edge bytes (el/er) survive.

    Linear reads only: a tags-reconstructed apply_result has no per-read
    circular tiling, so reads are treated as linear (the common case).
    """
    tags = payload['tags']
    stats = {key: 0 for key in _STATS_KEYS}
    unify_threshold = _WORKER['unify_threshold']

    has_ns = 'ns' in tags and 'nl' in tags
    if has_ns:
        stats['v2'] = 1
        v2_short_count = sum(
            1 for length in tags['nl'] if 0 < int(length) < unify_threshold
        )
    else:
        v2_short_count = 0

    try:
        ns_raw = read.get_tag('ns')
        nl_raw = read.get_tag('nl')
    except KeyError:
        ns_raw, nl_raw = (), ()
    try:
        as_raw = read.get_tag('as')
        al_raw = read.get_tag('al')
    except KeyError:
        as_raw, al_raw = (), ()

    if len(ns_raw) == 0 and len(as_raw) == 0:
        return (None, None, None, None, None), stats

    # Current FiberHMM stores tags molecular frame; recall works in SEQ (query)
    # frame. Legacy/v1.0 BAMs already store them seq-frame -- flipping those
    # again mis-places reverse-strand calls, so use as-is when not molecular.
    if _WORKER.get('input_molecular_frame', True):
        ns_seq, nl_seq = flip_intervals_to_seq(ns_raw, nl_raw, read)
        as_seq, al_seq = flip_intervals_to_seq(as_raw, al_raw, read)
    else:
        ns_seq, nl_seq = ns_raw, nl_raw
        as_seq, al_seq = as_raw, al_raw

    extracted = extract_modifications(read, _WORKER['mode'], _WORKER['k'])
    if extracted is None:
        # No modification data: pass v2 calls through unchanged (TF-only shape).
        nucs = [(int(s), int(L)) for s, L in zip(ns_seq, nl_seq) if int(L) > 0]
        msps = [(int(s), int(L)) for s, L in zip(as_seq, al_seq) if int(L) > 0]
        return (([], nucs, msps, None)), stats

    mod_pos, strand, seq = extracted
    obs = encode_from_query_sequence(
        seq, mod_pos, edge_trim=10, mode=_WORKER['mode'], strand=strand,
        context_size=_WORKER['k'], is_reverse=bool(read.is_reverse),
    )
    apply_result = {
        'encoded': obs,
        'ns': ns_seq, 'nl': nl_seq,
        'as': as_seq, 'al': al_seq,
        'ns_scores': None, 'as_scores': None,
    }
    m5c_mask = None
    if _WORKER.get('m5c_llr_hit') is not None:
        m5c_mask = build_cpg_mask(
            read, len(seq),
            _WORKER.get('cpg_mask_policy', 'unmethylated-only'),
        )
    fiber_read = {'query_sequence': payload['seq']}
    result = build_fused_recall_result(
        fiber_read, apply_result,
        _WORKER['llr_hit'], _WORKER['llr_miss'],
        _WORKER['min_llr'], _WORKER['min_opps'], unify_threshold,
        False,  # with_scores
        recall_nucs=True,
        split_min_llr=nuc_cfg.split_min_llr,
        split_min_opps=nuc_cfg.split_min_opps,
        nuc_min_size=nuc_cfg.nuc_min_size,
        msp_min_size=nuc_cfg.msp_min_size,
        phase_nrl=nuc_cfg.phase_nrl,
        nuc_recall_policy=nuc_cfg.nuc_recall_policy,
        nuc_profile=_WORKER.get('nuc_profile'),
        derived_tf_max_edge_ambiguity=(
            nuc_cfg.derived_tf_max_edge_ambiguity),
        m5c_mask=m5c_mask,
        m5c_llr_hit=_WORKER.get('m5c_llr_hit'),
        m5c_llr_miss=_WORKER.get('m5c_llr_miss'),
        nuc_llr_hit=_WORKER.get('nuc_llr_hit'),
        nuc_llr_miss=_WORKER.get('nuc_llr_miss'),
        nuc_m5c_llr_hit=_WORKER.get('nuc_m5c_llr_hit'),
        nuc_m5c_llr_miss=_WORKER.get('nuc_m5c_llr_miss'),
    )

    stats['tf'] = len(result['tf_calls'])
    survived_short = sum(
        1 for length in result['nl'] if 0 < int(length) < unify_threshold
    )
    stats['demoted'] = max(0, v2_short_count - survived_short)
    return (None, None, None, None, result), stats


def _process_payload_chunk(payloads):
    """Worker: process a list of compact payloads."""
    out = []
    total = {key: 0 for key in _STATS_KEYS}
    for payload in payloads:
        try:
            result, stats = _process_payload_record(payload)
        except Exception:
            result = None
            stats = {key: 0 for key in _STATS_KEYS}
            stats['failed'] = 1
        out.append(result)
        for key in _STATS_KEYS:
            total[key] += stats[key]
    return out, total


def _apply_result(read, result, also_write_legacy, downstream_compat):
    """Apply compact worker result to a pysam read in place (main process)."""
    read_length = len(read.query_sequence) if read.query_sequence else 0
    if len(result) == 5:
        # --recall-nucs path: result[4] is the fused result dict (or None when
        # the read had no usable tags). Route through write_fused_recall_tags so
        # nuc QQQ edge bytes (el/er) are emitted alongside tf/msp.
        fused = result[4]
        if fused is not None:
            write_fused_recall_tags(
                read,
                read_length=read_length,
                result=fused,
                also_write_legacy=also_write_legacy,
                downstream_compat=downstream_compat,
            )
        return
    tf_calls, kept_nucs, msps, nq_for_kept = result
    write_ma_tags(
        read,
        read_length=read_length,
        tf_calls=tf_calls,
        kept_nucs=kept_nucs,
        msps=msps,
        nq_for_kept_nucs=nq_for_kept,
        also_write_legacy=also_write_legacy,
        downstream_compat=downstream_compat,
    )


def _single_thread_loop(bam_in, bam_out, _header_text,
                        llr_hit, llr_miss, mode, k,
                        min_llr, min_opps, unify_threshold,
                        also_write_legacy, downstream_compat, max_reads,
                        nuc_cfg=None, input_molecular_frame=True,
                        m5c_llr_hit=None, m5c_llr_miss=None,
                        cpg_mask_policy="unmethylated-only",
                        nuc_protected_hit=None, nuc_accessible_hit=None,
                        nuc_llr_hit=None, nuc_llr_miss=None,
                        nuc_m5c_llr_hit=None, nuc_m5c_llr_miss=None):
    """Single-threaded path.  No IPC — process reads directly."""
    _worker_init(llr_hit, llr_miss, mode, k, min_llr, min_opps, unify_threshold,
                 nuc_cfg, input_molecular_frame, m5c_llr_hit, m5c_llr_miss,
                 cpg_mask_policy,
                 nuc_protected_hit, nuc_accessible_hit,
                 nuc_llr_hit, nuc_llr_miss,
                 nuc_m5c_llr_hit, nuc_m5c_llr_miss)
    n_reads = n_v2 = n_tf = n_demoted = n_failed = 0
    for read in bam_in:
        if max_reads and n_reads >= max_reads:
            break
        try:
            result, stats = _process_payload_record(_make_payload(read, mode))
        except Exception:
            result = None
            stats = {key: 0 for key in _STATS_KEYS}
            stats['failed'] = 1
        if result is not None:
            _apply_result(read, result, also_write_legacy, downstream_compat)
        bam_out.write(read)
        n_reads += 1
        n_v2 += stats['v2']
        n_tf += stats['tf']
        n_demoted += stats['demoted']
        n_failed += stats['failed']
    return n_reads, n_v2, n_tf, n_demoted, n_failed


def _parallel_loop(bam_in, bam_out, _header_text,
                   llr_hit, llr_miss, mode, k,
                   min_llr, min_opps, unify_threshold,
                   also_write_legacy, downstream_compat,
                   max_reads, n_cores, chunk_size, nuc_cfg=None,
                   input_molecular_frame=True, m5c_llr_hit=None, m5c_llr_miss=None,
                   cpg_mask_policy="unmethylated-only",
                   nuc_protected_hit=None, nuc_accessible_hit=None,
                   nuc_llr_hit=None, nuc_llr_miss=None,
                   nuc_m5c_llr_hit=None, nuc_m5c_llr_miss=None):
    """Multi-core path with slim IPC and bounded in-flight queue.

    Uses apply_async + a bounded deque instead of imap to cap how many chunks
    are held in memory simultaneously.  When downstream (disk write, sort
    backpressure) is slow, the submission loop blocks on the oldest pending
    result before submitting another chunk, so RSS stays bounded regardless
    of downstream speed.

    In-flight cap: n_cores + 2 chunks.  At chunk_size=1024 and n_cores=4,
    the worst-case working set is ~6 * 1024 pysam reads ≈ 180 MB, not 52 GB.
    """
    MAX_INFLIGHT = n_cores + 2   # workers + small head-start headroom
    pending: deque = deque()     # deque of (reads_chunk, AsyncResult)

    n_reads = n_v2 = n_tf = n_demoted = 0
    n_failed = 0

    def _drain_one():
        nonlocal n_reads, n_v2, n_tf, n_demoted, n_failed
        reads_chunk, fut = pending.popleft()
        out_results, stats = fut.get()   # blocks until result is ready
        for read, result in zip(reads_chunk, out_results):
            if result is not None:
                _apply_result(read, result, also_write_legacy, downstream_compat)
            bam_out.write(read)
        n_reads += len(reads_chunk)
        n_v2 += stats['v2']
        n_tf += stats['tf']
        n_demoted += stats['demoted']
        n_failed += stats['failed']

    with mp.Pool(
        processes=n_cores,
        initializer=_worker_init,
        initargs=(llr_hit, llr_miss, mode, k, min_llr, min_opps, unify_threshold,
                  nuc_cfg, input_molecular_frame, m5c_llr_hit, m5c_llr_miss,
                  cpg_mask_policy,
                  nuc_protected_hit, nuc_accessible_hit,
                  nuc_llr_hit, nuc_llr_miss,
                  nuc_m5c_llr_hit, nuc_m5c_llr_miss),
    ) as pool:
        buf_reads: list = []
        buf_payloads: list = []
        n = 0

        for read in bam_in:
            if max_reads and n >= max_reads:
                break
            buf_reads.append(read)
            buf_payloads.append(_make_payload(read, mode))
            n += 1

            if len(buf_reads) >= chunk_size:
                future = pool.apply_async(_process_payload_chunk, (buf_payloads,))
                pending.append((buf_reads, future))
                buf_reads, buf_payloads = [], []

                # Backpressure: drain the oldest result before submitting more
                if len(pending) >= MAX_INFLIGHT:
                    _drain_one()

        # Submit last partial chunk
        if buf_reads:
            future = pool.apply_async(_process_payload_chunk, (buf_payloads,))
            pending.append((buf_reads, future))

        # Drain all remaining results
        while pending:
            _drain_one()

    return n_reads, n_v2, n_tf, n_demoted, n_failed


def parse_args(default_recall_nucs: bool = False):
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
                   help='Hia5 sequencing platform; omission warns and defaults '
                        'to pacbio. Ignored for dddb/ddda.')
    p.add_argument('--min-llr', type=float, default=None,
                   help='Override native LLR cost per TF interval in joint decoding '
                        '(nats; default: enzyme preset; not an FDR threshold).')
    p.add_argument('--min-opps', type=int, default=3,
                   help='Min informative target positions per call (default 3)')
    p.add_argument('--emission-uplift', type=float, default=None,
                   help='Power transform on emission probabilities. Default 1.0 '
                        '(identity). Use a pre-uplifted model file (e.g. '
                        'ddda_TF.json) for DddA rather than setting this.')
    p.add_argument('--use-m5c', action=argparse.BooleanOptionalAction, default=None,
                   help='Enable CpG-aware DddA recall. By default, retain CpGs only inside confident ddda_ucg MA spans. Enabled '
                        'automatically for DddA, disabled for other enzymes; '
                        'use --use-m5c explicitly with a custom DddA model.')
    p.add_argument('--cpg-mask-policy',
                   choices=('unmethylated-only', 'methylated-only'),
                   default='unmethylated-only',
                   help='CpG-aware policy: retain CpGs only inside confident '
                        'unmethylated islands (default), or reproduce the '
                        'former behavior that masks only ddda_mcg spans.')
    p.add_argument('--unify-threshold', type=int, default=90,
                   help='v2 nucs with nl < this are scanned + may be demoted '
                        'to tf+ if overlapped by a recaller call (default 90)')
    p.add_argument('--input-frame', choices=['auto', 'molecular', 'query'],
                   default='auto',
                   help='Coordinate frame of the input ns/nl/as/al tags. '
                        '"auto" (default) detects the @CO fiberhmm:coord=molecular '
                        'marker: present -> molecular (current FiberHMM), absent '
                        '-> query/seq (legacy v1.0). Force with molecular/query. '
                        'Wrong frame mis-places reverse-strand calls.')
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
    add_legacy_mode_override(p)
    p.add_argument('--context-size', type=int, default=None,
                   help='Override context size. Default: read from model.')
    p.add_argument('--max-reads', type=int, default=0,
                   help='0 = no limit (default)')

    nuc = p.add_argument_group(
        'nucleosome recall (--recall-nucs)',
        'Run the per-read nucleosome recaller (split over-merged HMM footprints '
        'on accessible evidence + resolve platform-aware edges) BEFORE TF recall, '
        'reusing the existing apply-tagged ns/nl/as/al -- no HMM re-run. Linear '
        'reads only.',
    )
    nuc.add_argument('--recall-nucs', action=argparse.BooleanOptionalAction,
                     default=default_recall_nucs,
                     help='Enable nucleosome recall before TF recall. '
                          '(Default on for fiberhmm-recall-nucs.)')
    nuc.add_argument('--split-min-llr', type=float, default=4.0,
                     help='Min accessible-cut LLR to split a footprint; for '
                          'DddA, the molecule-local linker-residue '
                          'configuration LLR (default 4.0)')
    nuc.add_argument('--split-min-opps', type=int, default=3,
                     help='Min informative positions for a split cut or DddA '
                          'linker residue (default 3)')
    nuc.add_argument(
        '--ddda-derived-tf-max-edge-gap', type=int, default=12, metavar='BP',
        help='DddA phase-aware radial recall only: require TF scan space opened solely by '
             'nucleosome refinement to have a deamination hit within BP on '
             'both sides (default 12; -1 disables).',
    )
    nuc.add_argument(
        '--nuc-recall-policy',
        choices=['auto', 'conservative', 'topology'],
        default='auto',
        help='"auto" uses topology-constrained, ambiguity-preserving recall '
             'for Nanopore and conservative edges otherwise.',
    )
    nuc.add_argument('--nuc-min-size', type=int, default=85,
                     help='Min refined nucleosome size; smaller footprints are '
                          'demoted to accessible/MSP (default 85)')
    nuc.add_argument('--msp-min-size', type=int, default=0,
                     help='Min re-derived MSP size to keep (default 0)')
    nuc.add_argument('--phase-nrl', default='auto',
                     help='Pass-2 periodicity prior: off / auto / fixed bp. '
                          '"auto" (default) estimates the nucleosome repeat '
                          'length from the input BAM\'s existing nuc tags (no '
                          'HMM re-run). Lowers the split bar near phase-predicted '
                          'linkers in long footprints.')
    return p.parse_args()


def _resolve_model_metadata(model_path):
    mode = 'unknown'
    k = 3
    if model_path.endswith('.json'):
        try:
            with open(model_path) as f:
                d = json.load(f)
            mode = d.get('mode', mode)
            k = int(d.get('context_size', k))
        except (OSError, ValueError):
            pass
    return mode, k


def _resolve_nuc_recall_policy(args, mode: str) -> str:
    policy = str(getattr(args, 'nuc_recall_policy', 'auto')).lower()
    if policy == 'auto':
        return 'topology' if mode == 'nanopore-fiber' else 'conservative'
    return policy


def _parse_phase_nrl_option(raw):
    """Parse --phase-nrl (off / auto / fixed bp). Returns (kind, fixed_int)."""
    text = str(raw).strip().lower()
    if text in ('off', 'none', '', '0'):
        return ('off', 0)
    if text == 'auto':
        return ('auto', 0)
    try:
        return ('fixed', max(0, int(text)))
    except ValueError:
        return ('auto', 0)


def _estimate_phase_nrl_from_tags(path, nuc_min_size, sample_target=20000):
    """Estimate the nucleosome repeat length from existing ns/nl tags.

    Measures center-to-center spacing of nucleosome-sized (>= nuc_min_size)
    footprints already in the BAM -- no HMM re-run. Robust central estimate:
    histogram mode (10 bp bins) averaged with the in-peak median, clamped to
    [150, 215], anchored at 185 bp when the sample is too sparse. Mirrors
    fiberhmm-call's estimate_phase_nrl peak logic."""
    peak_lo, peak_hi = 120, 260
    anchor, clamp_lo, clamp_hi, min_pairs = 185, 150, 215, 300
    spacings = []
    reads_used = 0
    bam = pysam.AlignmentFile(path, 'rb', check_sq=False)
    try:
        for read in bam:
            if not read.has_tag('ns') or not read.has_tag('nl'):
                continue
            ns = list(read.get_tag('ns'))
            nl = list(read.get_tag('nl'))
            centers = sorted(
                s + length / 2.0
                for s, length in zip(ns, nl)
                if int(length) >= nuc_min_size
            )
            reads_used += 1
            for i in range(len(centers) - 1):
                spacings.append(centers[i + 1] - centers[i])
            if len(spacings) >= sample_target:
                break
    finally:
        bam.close()

    peak = [s for s in spacings if peak_lo <= s <= peak_hi]
    if len(peak) < min_pairs:
        return {'nrl': anchor, 'source': 'anchor',
                'n_pairs': len(peak), 'n_reads': reads_used}

    bins = {}
    for s in peak:
        b = int((s - peak_lo) // 10)
        bins[b] = bins.get(b, 0) + 1
    mode = peak_lo + max(bins, key=bins.get) * 10 + 5
    srt = sorted(peak)
    n = len(srt)
    median = srt[n // 2] if n % 2 else 0.5 * (srt[n // 2 - 1] + srt[n // 2])
    est = 0.5 * (mode + median)
    nrl = int(round(min(max(est, clamp_lo), clamp_hi)))
    return {'nrl': nrl, 'source': 'estimated',
            'n_pairs': len(peak), 'n_reads': reads_used}


def _resolve_recall_nucs_phase_nrl(args) -> int:
    """Resolve --phase-nrl to an int (0 = off). Only meaningful with --recall-nucs."""
    if not getattr(args, 'recall_nucs', False):
        return 0
    kind, fixed = _parse_phase_nrl_option(getattr(args, 'phase_nrl', 'auto'))
    if kind == 'off':
        return 0
    if kind == 'fixed':
        return fixed
    if args.in_bam == '-':
        print("  NOTE: --phase-nrl auto needs a file input to sample; "
              "using anchor 185 bp.", file=sys.stderr)
        return 185
    res = _estimate_phase_nrl_from_tags(
        args.in_bam, getattr(args, 'nuc_min_size', 85),
    )
    print(f"  [recall_nucs] phase-nrl auto -> {res['nrl']} bp "
          f"({res['source']}, {res['n_pairs']} pairs / {res['n_reads']} reads)",
          file=sys.stderr)
    return int(res['nrl'])


def _resolve_input_molecular_frame(args, header) -> bool:
    """Decide whether the input ns/nl/as/al are molecular-frame.

    --input-frame: auto (default) detects the @CO coord marker; molecular/query
    force it. Returns True for molecular (flip reverse tags to seq), False for
    legacy seq/query frame (use as-is)."""
    choice = str(getattr(args, 'input_frame', 'auto')).lower()
    if choice == 'molecular':
        return True
    if choice == 'query':
        print("  [recall] input-frame=query: treating ns/nl/as/al as legacy "
              "SEQ-frame tags (no molecular flip).", file=sys.stderr)
        return False
    is_mol = header_has_coord_marker(header)
    if not is_mol:
        print("  [recall] NOTE: input BAM has no @CO fiberhmm:coord=molecular "
              "marker -> treating ns/nl/as/al as legacy SEQ-frame (v1.0). "
              "Reverse-strand calls are kept in place (no double flip).",
              file=sys.stderr)
    return is_mol


def main(default_recall_nucs: bool = False):
    args = parse_args(default_recall_nucs=default_recall_nucs)
    using_bundled_model = args.model is None

    stdout_mode = (args.out_bam == '-')
    if stdout_mode:
        # Redirect informational prints to stderr so BAM stream on stdout stays clean
        sys.stdout = sys.stderr

    # Resolve model path: explicit -m wins; else use bundled model for --enzyme
    model_path = args.model
    if model_path is None:
        if args.enzyme is None:
            print(
                "error: one of --model or --enzyme must be provided.\n"
                "  Use --enzyme hia5/dddb/ddda to pick a bundled model, or\n"
                "  use --model /path/to/model.json for a custom model.",
                file=sys.stderr,
            )
            sys.exit(1)
        from fiberhmm.models import get_model_path as _get_bundled
        try:
            model_path = _get_bundled(args.enzyme, tool='recall', seq=args.seq)
        except (KeyError, FileNotFoundError) as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"[recall_tfs] using bundled model: {model_path}", file=sys.stderr)

    # Prominent beta banner: this feature is new, expect rough edges.
    # Mode line varies based on --downstream-compat.
    if args.downstream_compat:
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
        "  Defaults validated on Hia5 PacBio, DddB DAF, and DddA DAF-seq.\n"
        "\n"
        + mode_banner +
        "\n"
        "  File issues at https://github.com/fiberseq/FiberHMM/issues\n"
        "========================================================================",
        file=sys.stderr,
    )

    # Resolve cores
    if args.cores == 0:
        n_cores = mp.cpu_count()
    else:
        n_cores = max(1, args.cores)

    # Presets + overrides
    preset = ENZYME_PRESETS.get(args.enzyme, {}) if args.enzyme else {}
    min_llr = args.min_llr if args.min_llr is not None else preset.get('min_llr', 5.0)
    uplift = args.emission_uplift if args.emission_uplift is not None \
        else preset.get('emission_uplift', 1.0)

    model, model_k, model_mode = load_model_with_metadata(model_path)
    if not model_mode or not model_k:
        fb_mode, fb_k = _resolve_model_metadata(model_path)
        model_mode = model_mode or fb_mode
        model_k = model_k or fb_k
    from fiberhmm.models import get_metadata_mode_aliases, get_observation_mode
    inferred_mode = (
        get_observation_mode(
            args.enzyme, args.seq, warn_missing_seq=False
        )
        if using_bundled_model else None
    )
    metadata_mode_aliases = (
        get_metadata_mode_aliases(
            args.enzyme, args.seq, warn_missing_seq=False
        )
        if using_bundled_model else ()
    )
    try:
        mode = resolve_observation_mode(
            model_mode,
            inferred_mode=inferred_mode,
            explicit_mode=args.mode,
            source_label=(
                f"bundled {args.enzyme} model"
                if using_bundled_model else "custom model"
            ),
            metadata_mode_aliases=metadata_mode_aliases,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    k = args.context_size or int(model_k)
    nuc_recall_policy = _resolve_nuc_recall_policy(args, mode)

    llr_hit, llr_miss = build_llr_tables(model)
    m5c_llr_hit = m5c_llr_miss = None
    use_m5c_arg = getattr(args, 'use_m5c', None)
    use_m5c = args.enzyme == 'ddda' if use_m5c_arg is None else use_m5c_arg
    cpg_mask_policy = getattr(args, 'cpg_mask_policy', 'unmethylated-only')
    if use_m5c and (mode != 'daf' or args.enzyme not in (None, 'ddda')):
        raise SystemExit(
            '--use-m5c is calibrated only for DddA or a custom DAF/DddA model'
        )
    if use_m5c:
        m5c_llr_hit, m5c_llr_miss = build_m5c_llr_tables(
            model, emission_uplift=uplift,
        )
    if abs(uplift - 1.0) > 1e-9:
        llr_hit, llr_miss = apply_emission_uplift(llr_hit, llr_miss, model, uplift)

    print(
        f"[recall_tfs] enzyme={args.enzyme or 'custom'} mode={mode} k={k} "
        f"min_llr={min_llr:.2f} uplift={uplift:.2f} "
        f"tf_decoder={TF_DECODER_VERSION} "
        f"cpg_mask={cpg_mask_policy if m5c_llr_hit is not None else 'off'} "
        f"unify_threshold={args.unify_threshold} cores={n_cores} "
        f"numba={'on' if HAS_NUMBA else 'off'}",
        file=sys.stderr,
    )

    # Resolve nucleosome-recall config (None = TF-only, the default).
    nuc_cfg = None
    if getattr(args, 'recall_nucs', False):
        print("  +RECALL-NUCS: nucleosome recaller runs before TF recall "
              f"(policy={nuc_recall_policy}; reuses apply-tagged ns/nl/as/al "
              "-- no HMM re-run; linear reads).",
              file=sys.stderr)
        nuc_profile_path = None
        derived_tf_max_edge_ambiguity = None
        if args.enzyme == 'ddda':
            from fiberhmm.models import _bundled_model_path
            nuc_profile_path = _bundled_model_path('ddda_nuc_profile.json')
            if args.ddda_derived_tf_max_edge_gap < -1:
                raise SystemExit(
                    '--ddda-derived-tf-max-edge-gap must be -1 or >= 0')
            if args.ddda_derived_tf_max_edge_gap >= 0:
                derived_tf_max_edge_ambiguity = (
                    args.ddda_derived_tf_max_edge_gap)
        nuc_cfg = _NucCfg(
            recall_nucs=True,
            split_min_llr=args.split_min_llr,
            split_min_opps=args.split_min_opps,
            nuc_min_size=args.nuc_min_size,
            msp_min_size=args.msp_min_size,
            phase_nrl=_resolve_recall_nucs_phase_nrl(args),
            nuc_recall_policy=nuc_recall_policy,
            nuc_profile_path=nuc_profile_path,
            derived_tf_max_edge_ambiguity=derived_tf_max_edge_ambiguity,
        )

    nuc_protected_hit = nuc_accessible_hit = None
    nuc_llr_hit = nuc_llr_miss = None
    nuc_m5c_llr_hit = nuc_m5c_llr_miss = None
    if nuc_cfg is not None:
        nuc_model = model
        nuc_model_path = model_path
        nuc_uplift = uplift
        if args.enzyme == 'ddda':
            from fiberhmm.models import get_model_path as _get_bundled
            nuc_model_path = _get_bundled(
                'ddda', tool='nuc_refine', seq=args.seq,
            )
            nuc_model, _, _ = load_model_with_metadata(nuc_model_path)
            # A TF-emission sensitivity override must not retune the frozen
            # DddA radial-nucleosome likelihoods.
            nuc_uplift = 1.0
        nuc_llr_hit, nuc_llr_miss = build_llr_tables(nuc_model)
        if abs(nuc_uplift - 1.0) > 1e-9:
            nuc_llr_hit, nuc_llr_miss = apply_emission_uplift(
                nuc_llr_hit, nuc_llr_miss, nuc_model, nuc_uplift,
            )
        if use_m5c:
            nuc_m5c_llr_hit, nuc_m5c_llr_miss = build_m5c_llr_tables(
                nuc_model, emission_uplift=nuc_uplift,
            )
        print(
            f"  nuc likelihood model: {nuc_model_path}",
            file=sys.stderr,
        )
    if nuc_cfg is not None and nuc_cfg.nuc_profile_path:
        nuc_protected_hit, nuc_accessible_hit = build_conditional_hit_tables(
            nuc_model, emission_uplift=nuc_uplift,
        )

    # Open BAMs with io-threads. pysam accepts "-" as stdin/stdout natively.
    bam_in = pysam.AlignmentFile(args.in_bam, 'rb',
                                  check_sq=False,
                                  threads=args.io_threads)
    # Resolve the coordinate frame of the input ns/nl/as/al tags. Current
    # FiberHMM stamps the @CO molecular marker; legacy/v1.0 BAMs lack it and
    # store the tags in SEQ (query) frame -- flipping those again mis-places
    # every reverse-strand call. Auto-detect from the header, overridable.
    input_molecular_frame = _resolve_input_molecular_frame(args, bam_in.header)

    bam_out = None
    try:
        from fiberhmm.io.bam_header import append_ma_types
        output_header = append_coord_marker(bam_in.header)
        if not args.downstream_compat:
            output_header = append_ma_types(output_header, ("nuc", "msp", "tf"))
        output_header = maybe_append_pg(
            output_header,
            _build_recall_pg_record(args, mode, model_path, nuc_cfg),
        )
        bam_out = pysam.AlignmentFile(args.out_bam, 'wb',
                                       header=output_header,
                                       threads=args.io_threads)
        header_text = str(bam_in.header)

        # Compat mode always writes legacy tags (TFs live in ns/nl there).
        also_write_legacy = True if args.downstream_compat else (not args.no_legacy_tags)

        if n_cores == 1:
            n_reads, n_v2, n_tf, n_demoted, n_failed = _single_thread_loop(
                bam_in, bam_out, header_text,
                llr_hit, llr_miss, mode, k,
                min_llr, args.min_opps, args.unify_threshold,
                also_write_legacy, args.downstream_compat, args.max_reads,
                nuc_cfg, input_molecular_frame,
                m5c_llr_hit, m5c_llr_miss, cpg_mask_policy,
                nuc_protected_hit, nuc_accessible_hit,
                nuc_llr_hit, nuc_llr_miss,
                nuc_m5c_llr_hit, nuc_m5c_llr_miss,
            )
        else:
            n_reads, n_v2, n_tf, n_demoted, n_failed = _parallel_loop(
                bam_in, bam_out, header_text,
                llr_hit, llr_miss, mode, k,
                min_llr, args.min_opps, args.unify_threshold,
                also_write_legacy, args.downstream_compat, args.max_reads,
                n_cores, args.chunk_size, nuc_cfg, input_molecular_frame,
                m5c_llr_hit, m5c_llr_miss, cpg_mask_policy,
                nuc_protected_hit, nuc_accessible_hit,
                nuc_llr_hit, nuc_llr_miss,
                nuc_m5c_llr_hit, nuc_m5c_llr_miss,
            )
    finally:
        bam_in.close()
        if bam_out is not None:
            bam_out.close()

    print(
        f"[recall_tfs] processed {n_reads} reads; {n_v2} carried v2 tags; "
        f"{n_tf} TF calls emitted; {n_demoted} v2 short nucs demoted to tf+",
        file=sys.stderr,
    )
    if n_failed:
        print(
            f"[recall_tfs] warning: {n_failed} reads passed through unchanged "
            "after recall errors",
            file=sys.stderr,
        )


def main_recall_nucs():
    """Entry point for ``fiberhmm-recall-nucs`` -- same tool, nuc recall on."""
    main(default_recall_nucs=True)


if __name__ == '__main__':
    main()
