#!/usr/bin/env python3
"""fiberhmm-recall-tfs CLI  --  LLR-based TF footprint recaller.

Per-enzyme defaults are calibrated for Hia5 (PacBio, Nanopore), DddB and
DddA DAF-seq.

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
    add_force_seq_arg,
    add_legacy_mode_override,
    refuse_non_bam_output,
    require_model_files,
    resolve_observation_mode,
    resolve_platform_argument,
)
from fiberhmm.cli.provenance import (
    DEFAULTS_RESOLVED_KEY,
    REPLACE_CHEMISTRY_KEY,
    ChemistryConflictError,
    chemistry_declaration,
    nuc_profile_identity,
    nuc_profile_sha256,
    output_header_with_provenance,
    resolve_effective_chemistry,
)
from fiberhmm.core.bam_reader import encode_from_query_sequence
from fiberhmm.core.model_io import (
    ModelContextError,
    load_model_with_metadata,
    validate_context_size,
)
from fiberhmm.inference.bam_output import atomic_output
from fiberhmm.inference.worker_results import (
    WorkerFailureError,
    enforce_worker_failure_policy,
    extend_failure_messages,
    record_failure_message,
)
from fiberhmm.io.annotation_frame import (
    MOLECULAR,
    ambiguous_frame_message,
    legacy_tag_frame,
    ma_annotation_frame,
)
from fiberhmm.io.bam_header import append_coord_marker
from fiberhmm.io.ma_tags import (
    fibertools_ma_intervals,
    flip_intervals_to_seq,
    parse_aq_array,
    parse_ma_tag,
)
from fiberhmm.inference.fused_stages import build_fused_recall_result
from fiberhmm.inference.tagging import write_fused_recall_tags
from fiberhmm.inference.tf_recaller import (
    ENZYME_PRESETS,
    RECALL_PROB_THRESHOLD,
    TF_DECODER_VERSION,
    HAS_NUMBA,
    apply_emission_uplift,
    build_conditional_hit_tables,
    build_cpg_mask,
    build_llr_tables,
    build_m5c_llr_tables,
    extract_modification_calls,
    recall_read,
    resolve_cpg_masking,
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


def _build_recall_pg_record(args, mode, model_path, nuc_cfg, nuc_model_path=None):
    """Build command and exact table/nucleosome-profile provenance for output BAM.

    ``model_path`` is the table of this run's recall pass (it has no apply
    pass), so it is declared as ``recall_sha256``; ``nuc_model_path`` is a
    separate nucleosome-likelihood table when one is used (DddA).
    """
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
        REPLACE_CHEMISTRY_KEY: bool(getattr(args, 'replace_chemistry', False)),
        'PN': program_name,
        'VN': getattr(_fh, '__version__', 'unknown'),
        'CL': ' '.join(sys.argv),
        DEFAULTS_RESOLVED_KEY: True,
        'chemistry': chemistry_declaration(
            args,
            mode,
            None,
            model_path,
            profile_identity,
            profile_sha256,
            nuc_model_path=nuc_model_path,
        ),
        'DS': (
            'FiberHMM second-pass footprint refinement; coord=molecular '
            '(ns/nl/as/al/MA in molecular original-fiber coordinates); '
            f'mode={mode} enzyme={args.enzyme or "custom"} '
            f'prob_threshold={getattr(args, "prob_threshold", None)} '
            f'tf_decoder={TF_DECODER_VERSION} '
            f'recall_nucs={recall_nucs} nuc_recall_policy={policy} '
            f'nuc_profile={profile_identity or "off"} '
            f'nuc_sha256={profile_sha256 or "off"} phase_nrl={phase_nrl} '
            f'daf_run_mask={(">=" + str(args.daf_mask_runs) + "/" + args.daf_run_policy) if getattr(args, "daf_mask_runs", 0) else "off"}'
        ),
    }


def _worker_init(llr_hit, llr_miss, mode, k, min_llr, min_opps, unify_threshold,
                 nuc_cfg=None, input_molecular_frame=True,
                 m5c_llr_hit=None, m5c_llr_miss=None,
                 cpg_mask_policy="unmethylated-only",
                 nuc_protected_hit=None, nuc_accessible_hit=None,
                 nuc_llr_hit=None, nuc_llr_miss=None,
                 nuc_m5c_llr_hit=None, nuc_m5c_llr_miss=None,
                 prob_threshold=RECALL_PROB_THRESHOLD):
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
    _WORKER['prob_threshold'] = int(prob_threshold)


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
    __slots__ = ('query_sequence', 'is_reverse', '_tags', '_daf_md_result',
                 '_daf_unaligned_query_positions')

    def __init__(self, seq, is_reverse, tags, daf_md_result=None,
                 daf_unaligned_query_positions=None):
        self.query_sequence = seq
        self.is_reverse = is_reverse
        self._tags = tags
        self._daf_md_result = daf_md_result
        # CIGAR I/S bases, computed by the producer (the stub has no CIGAR).
        self._daf_unaligned_query_positions = set(
            daf_unaligned_query_positions or ())

    def has_tag(self, t):
        return t in self._tags

    def get_tag(self, t):
        return self._tags[t]

    @property
    def query_length(self):
        return len(self.query_sequence) if self.query_sequence else 0


def _fiberhmm_ma_footprints(read) -> dict | None:
    """Nucleosomes, MSPs and nucleosome ``nq`` bytes from a FiberHMM ``MA``.

    ``{'nuc': [(start, length), ...], 'msp': [...], 'nq': [...] or None}``,
    0-based, in the frame the MA was written in; ``None`` without a parsable
    ``MA``. ``nq`` is the first quality byte of each ``nuc`` annotation (the
    ``Q`` of ``nuc.Q``/``nuc.QQQ``), or None when AQ carries no nuc quality.
    """
    if not read.has_tag('MA'):
        return None
    try:
        parsed = parse_ma_tag(str(read.get_tag('MA')))
    except ValueError:
        return None
    aq = read.get_tag('AQ') if read.has_tag('AQ') else None
    raw_types = parsed['raw_types']
    quals = parse_aq_array(aq, [rt[2] for rt in raw_types],
                           [len(rt[3]) for rt in raw_types])
    nucs, msps, nq = [], [], []
    has_nq = aq is not None
    index = 0
    for name, _strand, qual_spec, intervals in raw_types:
        for start, length in intervals:
            q = quals[index]
            index += 1
            if name == 'nuc':
                nucs.append((int(start), int(length)))
                if q and qual_spec.startswith('Q'):
                    nq.append(int(q[0]))
                else:
                    has_nq = False
            elif name == 'msp':
                msps.append((int(start), int(length)))
    return {'nuc': nucs, 'msp': msps, 'nq': nq if has_nq and nucs else None}


# Frame of FiberHMM MA intervals in the input (molecular when the header
# declares coord=molecular, else SEQ: fiberhmm.io.annotation_frame), set by
# _recall() from the input header before any payload is built.
_MA_FRAME = {'molecular': True}


def _annotation_footprints_as_legacy_tags(read, input_molecular_frame=True) -> dict:
    """``ns/nl/as/al`` (and ``nq``) for a read whose footprints are only in ``MA``/``Ma``.

    fibertools-rs >= 0.13 writes nucleosomes and MSPs only to ``Ma`` (always
    molecular frame), and ``fiberhmm-call``/``-recall-tfs --no-legacy-tags``
    only to FiberHMM's ``MA`` (molecular when the header declares
    coord=molecular, else SEQ). Without this the recaller saw no footprints
    and stripped every annotation. The recaller reads ns/nl/as/al in the
    run's input frame, so the intervals are expressed in that frame: molecular
    for molecular runs, SEQ otherwise (``--input-frame query`` or an
    undecided legacy-tag frame). Reads that carry legacy tags keep them.
    Circular reads' MA pieces are used as linear pieces, like their legacy
    tags.
    """
    if read.has_tag('ns') or read.has_tag('as'):
        return {}
    ma = fibertools_ma_intervals(read)
    ma_molecular = True
    if ma is None:
        ma = _fiberhmm_ma_footprints(read)
        ma_molecular = bool(_MA_FRAME['molecular'])
    if not ma or not (ma['nuc'] or ma['msp']):
        return {}
    want_molecular = bool(input_molecular_frame)
    tags = {}
    for (start_tag, length_tag), feature in ((('ns', 'nl'), 'nuc'), (('as', 'al'), 'msp')):
        starts = [int(s) for s, _ in ma[feature]]
        lengths = [int(n) for _, n in ma[feature]]
        if ma_molecular != want_molecular:
            # flip_intervals_to_seq mirrors reverse reads' intervals; it is its
            # own inverse, so it also takes SEQ intervals to molecular.
            starts, lengths = flip_intervals_to_seq(starts, lengths, read)
        tags[start_tag], tags[length_tag] = starts, lengths
    if ma.get('nq') is not None:
        tags['nq'] = list(ma['nq'])
    return tags


# Name kept for callers of the fibertools-only helper it generalises.
_fibertools_ma_as_legacy_tags = _annotation_footprints_as_legacy_tags


def _make_payload(read, mode=None, input_molecular_frame=True) -> dict:
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
    if input_molecular_frame is None and any(
            read.has_tag(t) for t in ('ns', 'nl', 'as', 'al')):
        raise SystemExit("error: " + ambiguous_frame_message(
            _UNDECIDED_FRAME['reason'] or 'the header does not decide it'))
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
    tags.update(_annotation_footprints_as_legacy_tags(read, input_molecular_frame))

    payload = {
        'name': getattr(read, 'query_name', None),
        'seq': read.query_sequence,
        'is_reverse': read.is_reverse,
        'tags': tags,
    }
    from fiberhmm.inference.engine import read_no_call_blocks
    blocks = read_no_call_blocks(read, mode)
    if blocks:
        payload['_no_call_blocks'] = blocks
    if mode == 'daf' and read.query_sequence:
        from fiberhmm.core.bam_reader import has_iupac_encoding
        from fiberhmm.inference.engine import daf_unaligned_query_positions
        unaligned = daf_unaligned_query_positions(read)
        if unaligned:
            payload['_daf_unaligned_query_positions'] = unaligned
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
        payload.get('_daf_unaligned_query_positions'),
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
        prob_threshold=_WORKER.get('prob_threshold', RECALL_PROB_THRESHOLD),
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

    extracted = extract_modification_calls(
        read, _WORKER['mode'], _WORKER['k'],
        prob_threshold=_WORKER.get('prob_threshold', RECALL_PROB_THRESHOLD),
    )
    if extracted is None:
        # No modification data: pass v2 calls through unchanged (TF-only shape).
        nucs = [(int(s), int(L)) for s, L in zip(ns_seq, nl_seq) if int(L) > 0]
        msps = [(int(s), int(L)) for s, L in zip(as_seq, al_seq) if int(L) > 0]
        return (([], nucs, msps, None)), stats

    # Bases an MM '?' entry leaves unlisted carry no call: non-target, as in
    # the fused call path, not misses.
    mod_pos, strand, seq, unknown_pos = extracted
    obs = encode_from_query_sequence(
        seq, mod_pos, edge_trim=10, mode=_WORKER['mode'], strand=strand,
        context_size=_WORKER['k'], is_reverse=bool(read.is_reverse),
        unknown_positions=unknown_pos,
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
    if payload.get('_no_call_blocks'):
        fiber_read['no_call_blocks'] = payload['_no_call_blocks']
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


class _ChunkStats(dict):
    """Per-chunk counters plus the first per-read failure tracebacks."""

    failure_messages = ()


def _process_payload_chunk(payloads):
    """Worker: process a list of compact payloads."""
    out = []
    total = _ChunkStats({key: 0 for key in _STATS_KEYS})
    messages = []
    for payload in payloads:
        try:
            result, stats = _process_payload_record(payload)
        except Exception:
            result = None
            stats = {key: 0 for key in _STATS_KEYS}
            stats['failed'] = 1
            record_failure_message(
                messages,
                payload.get('name') if isinstance(payload, dict) else None,
            )
        out.append(result)
        for key in _STATS_KEYS:
            total[key] += stats[key]
    total.failure_messages = tuple(messages)
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
                        nuc_m5c_llr_hit=None, nuc_m5c_llr_miss=None,
                        failure_messages=None,
                        prob_threshold=RECALL_PROB_THRESHOLD):
    """Single-threaded path.  No IPC — process reads directly."""
    if failure_messages is None:
        failure_messages = []
    _worker_init(llr_hit, llr_miss, mode, k, min_llr, min_opps, unify_threshold,
                 nuc_cfg, input_molecular_frame, m5c_llr_hit, m5c_llr_miss,
                 cpg_mask_policy,
                 nuc_protected_hit, nuc_accessible_hit,
                 nuc_llr_hit, nuc_llr_miss,
                 nuc_m5c_llr_hit, nuc_m5c_llr_miss,
                 prob_threshold)
    n_reads = n_v2 = n_tf = n_demoted = n_failed = 0
    for read in bam_in:
        if max_reads and n_reads >= max_reads:
            break
        try:
            result, stats = _process_payload_record(_make_payload(read, mode, input_molecular_frame))
        except Exception:
            result = None
            stats = {key: 0 for key in _STATS_KEYS}
            stats['failed'] = 1
            record_failure_message(failure_messages,
                                   getattr(read, 'query_name', None))
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
                   nuc_m5c_llr_hit=None, nuc_m5c_llr_miss=None,
                   failure_messages=None,
                   prob_threshold=RECALL_PROB_THRESHOLD):
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
        if failure_messages is not None:
            extend_failure_messages(failure_messages,
                                    getattr(stats, 'failure_messages', ()))
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
                  nuc_m5c_llr_hit, nuc_m5c_llr_miss,
                  prob_threshold),
    ) as pool:
        buf_reads: list = []
        buf_payloads: list = []
        n = 0

        for read in bam_in:
            if max_reads and n >= max_reads:
                break
            buf_reads.append(read)
            buf_payloads.append(_make_payload(read, mode, input_molecular_frame))
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
                   help='Hia5 sequencing platform. When omitted it is taken '
                        'from the input\'s FIBERHMM-CHEMISTRY declaration or '
                        'detected from its MM specs (PacBio T-a vs Nanopore '
                        'A+a only); conflicting evidence stops the run, and '
                        'a given --seq that the reads contradict is refused '
                        '(see --force-seq). Ignored for dddb/ddda.')
    add_force_seq_arg(p)
    p.add_argument('--replace-chemistry', action='store_true',
                   help='Replace, instead of reconcile with, the input BAM\'s '
                        'FIBERHMM-CHEMISTRY declaration. By default a custom '
                        '--model inherits the input\'s enzyme/platform when its '
                        'observation mode matches, and a conflicting '
                        '--enzyme/--seq is refused.')
    p.add_argument('--daf-mask-runs', type=int, default=None, metavar='N',
                   help='DAF only: thin targets lying in same-strand runs of >= N original C (CT) or G (GA) bases (CC/GG and longer at N=2; see --daf-run-policy). Adjacent conversions are coupled and do not follow the per-site emission model. Default: 2 with keep-one for --enzyme ddda (duplex-validated), off otherwise; 0 disables.')
    p.add_argument('--daf-run-policy', choices=['keep-one', 'drop'], default='keep-one',
                   help="With --daf-mask-runs: keep each run's 5'-most target (default) or drop the run.")
    p.add_argument('--daf-mask-unaligned', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='DAF only: treat CIGAR insertion and soft-clip bases as no '
                        'evidence and leave unaligned stretches of >= 50 bp uncalled '
                        '(default on, as in fiberhmm-call).')
    p.add_argument('--min-llr', type=float, default=None,
                   help='Override native LLR cost per TF interval in joint decoding '
                        '(nats; default: enzyme preset; not an FDR threshold).')
    p.add_argument('--prob-threshold', type=int, default=None,
                   help='Min MM/ML probability 0-255 for re-reading modification '
                        'calls. Default: chemistry preset -- 248 for Hia5 '
                        'Nanopore (--seq nanopore, or the input\'s declared '
                        f'chemistry), {RECALL_PROB_THRESHOLD} otherwise. R/Y- and '
                        'MD-encoded DAF input is binary and ignores it.')
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
                        '"auto" (default) follows the @PG PP chain: the last '
                        'footprint writer decides (fibertools predict-m6a/'
                        'add-nucleosomes/fire -> molecular; FiberHMM '
                        'call/apply/recall -> molecular when declared '
                        'coord=molecular, else query); no writer -> '
                        'molecular if coord=molecular is declared, else '
                        'query (FiberHMM <= 2.12). Merged histories whose '
                        'writers disagree, with no declaration, stop the run '
                        'at the first read with these tags. A wrong frame '
                        'mirrors reverse-strand calls. fibertools Ma tags are '
                        'always molecular.')
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
                   help='Worker processes (0 = all CPUs; default 1).')
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
    from fiberhmm.cli.common import add_version_args
    add_version_args(p)
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
            if read.has_tag('ns') and read.has_tag('nl'):
                ns = list(read.get_tag('ns'))
                nl = list(read.get_tag('nl'))
            else:
                # fibertools >= 0.13 keeps nucleosomes only in Ma, and
                # --no-legacy-tags output only in FiberHMM's MA. Spacing is
                # the same in either frame, so no flip is needed here.
                ma = fibertools_ma_intervals(read) or _fiberhmm_ma_footprints(read)
                if not ma or not ma['nuc']:
                    continue
                ns = [s for s, _ in ma['nuc']]
                nl = [n for _, n in ma['nuc']]
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

    --input-frame molecular/query force it. auto (default) applies the shared
    footprint-tag frame rule (fiberhmm.io.annotation_frame.legacy_tag_frame,
    docs/reference/footprint-tag-frame.md). When merged @PG histories
    disagree and nothing declares coord=molecular, the run stops at the first
    read with those tags rather than risk mirroring reverse-strand footprints.
    Returns True for molecular (flip reverse tags to seq), False for SEQ, and
    None when undecided (then any read with ns/nl/as/al stops the run; reads
    whose only footprints are fibertools ``Ma``, always molecular, proceed).
    """
    choice = str(getattr(args, 'input_frame', 'auto')).lower()
    if choice == 'molecular':
        return True
    if choice == 'query':
        print("  [recall] input-frame=query: treating ns/nl/as/al as legacy "
              "SEQ-frame tags (no molecular flip).", file=sys.stderr)
        return False
    frame, reason = legacy_tag_frame(header)
    if frame is None:
        # Only reads with ns/nl/as/al need the frame (fibertools Ma is always
        # molecular), so the run stops at the first such read (_make_payload).
        _UNDECIDED_FRAME['reason'] = reason
        print(f"  [recall] input frame: undecided ({reason}); the run stops "
              "at the first read with ns/nl/as/al.", file=sys.stderr)
        return None
    print(f"  [recall] input frame: {frame} ({reason}).", file=sys.stderr)
    return frame == MOLECULAR


_UNDECIDED_FRAME = {'reason': ''}


def _header_declared_mode(header):
    """Observation mode declared by an input BAM header's chemistry, if unique."""
    from fiberhmm.io.bam_header import declared_chemistries
    modes = {item.get('mode') for item in declared_chemistries(header)}
    modes.discard(None)
    return modes.pop() if len(modes) == 1 else None


def _resolve_recall_prob_threshold(args, header=None):
    """Explicit --prob-threshold, else the chemistry preset.

    The chemistry is --enzyme/--seq when given, else the input BAM's own
    FIBERHMM-CHEMISTRY declaration in ``header`` (a custom --model recall of
    a Hia5 Nanopore call therefore also reads ML at 248).
    """
    from fiberhmm.models import (
        declared_prob_threshold_chemistry,
        resolve_prob_threshold,
    )

    explicit = getattr(args, 'prob_threshold', None)
    if explicit is not None:
        if not 0 <= int(explicit) <= 255:
            raise SystemExit('--prob-threshold must be in [0, 255]')
        return int(explicit)
    enzyme, seq = args.enzyme, args.seq
    if (not enzyme or not seq) and header is not None:
        declared_enzyme, declared_seq = declared_prob_threshold_chemistry(header)
        if not enzyme:
            enzyme = declared_enzyme
            seq = seq or declared_seq
        elif declared_enzyme == enzyme:
            seq = seq or declared_seq
    return resolve_prob_threshold(None, enzyme, seq, RECALL_PROB_THRESHOLD)


def _ddda_radial_call_in_history(header) -> bool:
    """Whether a ``fiberhmm-call`` @PG ran DddA radial nucleosome recall.

    Such a call picks its TF scan space from the HMM baseline footprints and
    filters TFs that only nucleosome refinement exposed; its output keeps only
    the refined footprints, so a recall cannot rebuild that scan space.
    """
    try:
        header_dict = header.to_dict() if hasattr(header, 'to_dict') else dict(header)
    except Exception:
        return False
    for record in header_dict.get('PG', []) or []:
        program = str(record.get('PN') or record.get('ID') or '')
        if not program.startswith('fiberhmm-call'):
            continue
        tokens = str(record.get('DS', '')).split()
        if 'recall_nucs=True' not in tokens:
            continue
        if any(token.startswith('nuc_profile=') and token != 'nuc_profile=off'
               for token in tokens):
            return True
    return False


def warn_ddda_call_recall(header, stream=None) -> bool:
    """Warn that recalling a DddA ``fiberhmm-call`` BAM does not reproduce the call."""
    if not _ddda_radial_call_in_history(header):
        return False
    print(
        "WARNING: this BAM was called by fiberhmm-call with DddA radial "
        "nucleosome recall. That call chooses its TF scan space from the HMM "
        "baseline footprints, which its output does not keep, so recalling it "
        "does not reproduce the call: TF calls (and with --recall-nucs, "
        "nucleosomes) change on most reads even with identical settings. To "
        "re-call DddA after fiberhmm-tag-m5c, or with a refit table or other "
        "settings, re-run fiberhmm-call on this BAM instead (it re-runs the "
        "HMM and keeps the ddda_ucg/ddda_mcg island calls).",
        file=stream if stream is not None else sys.stderr,
    )
    return True


def main(default_recall_nucs: bool = False):
    args = parse_args(default_recall_nucs=default_recall_nucs)
    try:
        _main(args)
    except ChemistryConflictError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    except WorkerFailureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


def _main(args):
    using_bundled_model = args.model is None

    stdout_mode = (args.out_bam == '-')
    if stdout_mode:
        # Redirect informational prints to stderr so BAM stream on stdout stays clean
        sys.stdout = sys.stderr

    # A missing --seq comes from the input's declaration or MM specs (refused
    # on conflicting evidence) before the bundled model is chosen.
    resolve_platform_argument(args, args.in_bam, tool='fiberhmm-recall-tfs')
    require_model_files('fiberhmm-recall-tfs', ('-m/--model', args.model))
    refuse_non_bam_output(args.out_bam, 'fiberhmm-recall-tfs')

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

    # -i X -o X (or an output naming the model) is refused, not done in place.
    import os

    from fiberhmm.cli.common import refuse_path_aliases
    refuse_path_aliases(
        os.path.basename(sys.argv[0]) or 'fiberhmm-recall-tfs',
        inputs={'--in-bam': args.in_bam, '--model': model_path},
        outputs={'--out-bam': args.out_bam})

    # Banner; the mode line varies with --downstream-compat.
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
        "  fiberhmm-recall-tfs  --  LLR TF footprint recaller\n"
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

    # Open the input first: its header (read without consuming records, so
    # stdin works too) settles the effective chemistry before any
    # enzyme-dependent default is chosen.
    bam_in = pysam.AlignmentFile(args.in_bam, 'rb',
                                 check_sq=False,
                                 threads=args.io_threads)
    try:
        _recall(args, bam_in, model_path, using_bundled_model, n_cores)
    finally:
        bam_in.close()


def _recall(args, bam_in, model_path, using_bundled_model, n_cores):
    """Resolve chemistry and defaults from ``bam_in``'s header, then recall."""
    model, model_k, model_mode = load_model_with_metadata(model_path)
    if not model_mode or not model_k:
        fb_mode, fb_k = _resolve_model_metadata(model_path)
        model_mode = model_mode or fb_mode
        model_k = model_k or fb_k
    if (
        not using_bundled_model
        and args.mode is None
        and model_mode in (None, '', 'unknown')
    ):
        # A custom table without mode metadata recalls a FiberHMM-called BAM
        # in the observation mode that BAM declares.
        declared_mode = _header_declared_mode(bam_in.header)
        if declared_mode:
            print(f"[recall_tfs] custom model has no mode metadata; using the "
                  f"input BAM's declared mode {declared_mode!r}.",
                  file=sys.stderr)
            model_mode = declared_mode
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
    try:
        validate_context_size(model, k, label=f"model {model_path}")
    except ModelContextError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    # Effective chemistry: a custom --model without --enzyme inherits the
    # input's declared enzyme/platform (same observation mode), and every
    # default below -- TF presets, DAF run mask, CpG masking, ML threshold,
    # nucleosome profile/likelihood model -- is then the one --enzyme
    # <inherited> selects, with the given model file. Conflicts fail here,
    # before any output is written.
    resolve_effective_chemistry(
        args, mode, bam_in.header, model_path, None,
        replace=bool(getattr(args, 'replace_chemistry', False)),
        tool=('fiberhmm-recall-nucs' if getattr(args, 'recall_nucs', False)
              else 'fiberhmm-recall-tfs'),
    )

    # Presets + overrides
    preset = ENZYME_PRESETS.get(args.enzyme, {}) if args.enzyme else {}
    min_llr = args.min_llr if args.min_llr is not None else preset.get('min_llr', 5.0)
    uplift = args.emission_uplift if args.emission_uplift is not None \
        else preset.get('emission_uplift', 1.0)

    nuc_recall_policy = _resolve_nuc_recall_policy(args, mode)
    # Same observation lattice as the first pass: unset means the chemistry
    # default (DddA keep-one on runs >= 2). Configured before any worker starts.
    from fiberhmm.core.bam_reader import configure_daf_run_mask, resolve_daf_run_mask
    requested = getattr(args, 'daf_mask_runs', None)
    if requested and mode != 'daf':
        print("error: --daf-mask-runs requires DAF mode", file=sys.stderr)
        sys.exit(2)
    try:
        args.daf_mask_runs, args.daf_run_policy = resolve_daf_run_mask(
            requested if mode == 'daf' else 0, getattr(args, 'daf_run_policy', 'keep-one'),
            getattr(args, 'enzyme', None))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    configure_daf_run_mask(args.daf_mask_runs, args.daf_run_policy)
    from fiberhmm.inference.engine import configure_daf_unaligned_mask
    configure_daf_unaligned_mask(
        bool(getattr(args, 'daf_mask_unaligned', True)) and mode == 'daf')

    llr_hit, llr_miss = build_llr_tables(model)
    m5c_llr_hit = m5c_llr_miss = None
    cpg_mask_policy = getattr(args, 'cpg_mask_policy', 'unmethylated-only')
    try:
        use_m5c = resolve_cpg_masking(
            getattr(args, 'use_m5c', None), args.enzyme, mode)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
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
    separate_nuc_model_path = None
    if nuc_cfg is not None:
        nuc_model = model
        nuc_model_path = model_path
        nuc_uplift = uplift
        if args.enzyme == 'ddda':
            from fiberhmm.models import get_model_path as _get_bundled
            nuc_model_path = _get_bundled(
                'ddda', tool='nuc_refine', seq=args.seq,
            )
            separate_nuc_model_path = nuc_model_path
            nuc_model, _, _ = load_model_with_metadata(nuc_model_path)
            try:
                validate_context_size(
                    nuc_model, k, label=f"nuc likelihood model {nuc_model_path}")
            except ModelContextError as exc:
                print(f"error: {exc}", file=sys.stderr)
                sys.exit(2)
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
    # A file output is written to a hidden sibling and published only when
    # the run (including the per-read failure policy) succeeds.
    failure_messages = []
    with atomic_output(args.out_bam) as out_path:
        # Resolve the coordinate frame of the input ns/nl/as/al tags with the
        # shared footprint-tag frame rule (last writer on each @PG PP chain);
        # merged histories that disagree stop at the first read that needs
        # it. A wrong frame mirrors every reverse-strand call.
        input_molecular_frame = _resolve_input_molecular_frame(args, bam_in.header)
        # Reads whose footprints are only in FiberHMM's MA (--no-legacy-tags
        # output) are read in the MA frame rule's frame.
        _MA_FRAME['molecular'] = ma_annotation_frame(bam_in.header) == MOLECULAR
        # ML threshold for re-reading MM/ML: explicit, else the chemistry
        # preset (Hia5 Nanopore 248, otherwise 125), taken from --enzyme/--seq
        # or the input's own chemistry declaration.
        prob_threshold = _resolve_recall_prob_threshold(args, bam_in.header)
        args.prob_threshold = prob_threshold
        print(f"[recall_tfs] ML threshold for MM/ML calls: {prob_threshold}",
              file=sys.stderr)
        from fiberhmm.inference.tf_recaller import warn_unapplied_call_daf_inputs
        warn_unapplied_call_daf_inputs(bam_in.header, mode)
        warn_ddda_call_recall(bam_in.header)

        bam_out = None
        try:
            from fiberhmm.io.bam_header import append_ma_types
            output_header = append_coord_marker(bam_in.header)
            if not args.downstream_compat:
                output_header = append_ma_types(output_header, ("nuc", "msp", "tf"))
            # Reconciles the run's chemistry with the input's declaration: a
            # custom --model inherits the input's enzyme/platform when the
            # mode matches; a real conflict raises ChemistryConflictError
            # before any output is written.
            output_header = output_header_with_provenance(
                output_header,
                _build_recall_pg_record(
                    args, mode, model_path, nuc_cfg,
                    nuc_model_path=separate_nuc_model_path),
            )
            bam_out = pysam.AlignmentFile(out_path, 'wb',
                                          header=output_header,
                                          threads=args.io_threads)
            header_text = str(bam_in.header)

            # Compat mode always writes legacy tags (TFs live in ns/nl there).
            also_write_legacy = (True if args.downstream_compat
                                 else (not args.no_legacy_tags))

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
                    failure_messages=failure_messages,
                    prob_threshold=prob_threshold,
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
                    failure_messages=failure_messages,
                    prob_threshold=prob_threshold,
                )
        finally:
            if bam_out is not None:
                bam_out.close()

        print(
            f"[recall_tfs] processed {n_reads} reads; {n_v2} carried v2 tags; "
            f"{n_tf} TF calls emitted; {n_demoted} v2 short nucs demoted to tf+",
            file=sys.stderr,
        )
        # Inside the atomic context: raising discards the temporary BAM.
        enforce_worker_failure_policy(
            n_failed, n_reads, failure_messages, log=sys.stderr, label='recall',
        )


def main_recall_nucs():
    """Entry point for ``fiberhmm-recall-nucs`` -- same tool, nuc recall on."""
    main(default_recall_nucs=True)


if __name__ == '__main__':
    main()
