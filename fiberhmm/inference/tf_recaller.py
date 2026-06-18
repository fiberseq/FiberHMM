"""LLR-based TF footprint recaller -- second pass on FiberHMM-tagged BAMs.

The first pass (``fiberhmm-apply``) calls nucleosomes and MSPs with the
trained 2-state HMM. This second pass scans the MSPs and short
sub-nucleosomal calls for sequence-context-aware TF footprints, using
the same emission table as the null model. The result: ``MA``/``AQ``
spec-compliant tags with a ``tf+QQQ`` annotation type.

Algorithm summary
-----------------

For each read:

1. Build per-position observation array using the same encoder as
   ``fiberhmm-apply`` (``encode_from_query_sequence``).
2. From v2 tags, derive the scan space:
     - all MSPs (``as``/``al``)
     - all short v2 nucs (``ns``/``nl`` with ``nl < unify_threshold``)
3. Inside each scan interval, run a Kadane local-maximum scan with
   per-context LLR steps:
     - miss step: ``log P(miss | ctx, protected) - log P(miss | ctx, accessible)``
     - hit  step: ``log P(hit  | ctx, protected) - log P(hit  | ctx, accessible)``
   When the running sum drops to 0 or below, flush the peak (if it
   crossed ``min_llr`` and contained at least ``min_opps`` informative
   target positions).
4. For each emitted call, compute edge ambiguity = bp distance from the
   conservative boundary (last informative miss + 1 on the right; first
   informative miss on the left) to the bracketing hit.
5. Emit MA + AQ tags. By default (``unify=True``), v2's ``ns``/``nl``
   short nucs that overlap a recaller call are *dropped* from the
   ``nuc+`` annotation (they live solely in ``tf+`` now).

Per-enzyme defaults are baked into ``ENZYME_PRESETS``. Hia5 uses the
trained pacbio model directly. DddB uses ``min_llr=4.0`` to compensate
for sparser per-position evidence. DddA reuses the DddB model with an
``emission_uplift=2.0`` power transform, since DddA's ~3x higher
deamination efficiency makes the DddB-trained P(hit | accessible)
underestimate the true signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from numba import jit as _numba_jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def _numba_jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from fiberhmm.core.bam_reader import (
    _has_mm_ml_inputs,
    daf_strand_from_tag,
    detect_daf_strand,
    encode_from_query_sequence,
    extract_daf_iupac_positions,
    has_iupac_encoding,
    parse_mm_tag_query_positions,
)
from fiberhmm.core.tag_access import get_preferred_tag
from fiberhmm.inference.tag_utils import clear_tags
from fiberhmm.io.ma_tags import (
    ambiguity_to_edge,
    flip_interval_frame,
    flip_intervals_to_seq,
    format_an_tag,
    format_aq_array,
    format_ma_tag,
    llr_to_tq,
    split_circular_interval,
)

# Observation-code constants (must match encode_from_query_sequence with k=3)
N_CTX = 4096           # 4^(2*3) hexamer contexts
NON_TARGET = N_CTX     # code 4096
UNMETH_OFFSET = 4097   # miss codes live at [4097, 4097 + 4096)


# Per-enzyme defaults: (min_llr, emission_uplift). All presets use uplift=1.0;
# enzymes that need it (DddA) get a pre-uplifted model file (ddda_TF.json)
# instead of a runtime power transform, to keep the preset semantics simple.
ENZYME_PRESETS = {
    'hia5':  dict(min_llr=5.0, emission_uplift=1.0),
    'dddb':  dict(min_llr=4.0, emission_uplift=1.0),
    'ddda':  dict(min_llr=5.0, emission_uplift=1.0),
}


@dataclass
class TFCall:
    """One TF call before it's converted to MA/AQ output."""
    start: int            # query coord, 0-based, inclusive
    length: int           # bp
    llr: float            # cumulative LLR (nats), positive
    n_opps: int           # number of informative target positions inside
    left_ambiguity: int   # bp gap to bracketing hit on the left (>=0)
    right_ambiguity: int  # bp gap to bracketing hit on the right (>=0)


@dataclass
class _TagOutputFrame:
    nuc_intervals: List[Tuple[int, int]]
    msp_intervals: List[Tuple[int, int]]
    tf_intervals: List[Tuple[int, int]]
    nq_values: List[int]
    tq_values: List[int]
    tf_el_values: List[int]
    tf_er_values: List[int]
    nuc_el_values: Optional[List[int]]
    nuc_er_values: Optional[List[int]]


@dataclass
class _TagOutputFrameRequest:
    read: object
    read_length: int
    kept_nucs: Sequence[Tuple[int, int]]
    msps: Sequence[Tuple[int, int]]
    tf_intervals: Sequence[Tuple[int, int]]
    nq_values: Sequence[int]
    tq_values: Sequence[int]
    tf_el_values: Sequence[int]
    tf_er_values: Sequence[int]
    nuc_el_values: Optional[Sequence[int]] = None
    nuc_er_values: Optional[Sequence[int]] = None


@dataclass
class _RecallTagPayload:
    kept_nucs: List[Tuple[int, int]]
    msps: List[Tuple[int, int]]
    tf_intervals: List[Tuple[int, int]]
    nq_values: List[int]
    nuc_qqq: bool
    ma_nucs: List[Tuple[int, int]]
    ma_msps: List[Tuple[int, int]]
    ma_tfs: List[Tuple[int, int]]
    nuc_names: List[str]
    msp_names: List[str]
    tf_names: List[str]
    nuc_q_split: Optional[List[Sequence[int]]]
    tf_q_split: Optional[List[Sequence[int]]]
    needs_an: bool


@dataclass
class _SplitNamedIntervals:
    intervals: List[Tuple[int, int]]
    names: List[str]
    qual_rows: Optional[List[Sequence[int]]]
    any_split: bool


@dataclass
class _RecallNucQualityInputs:
    nq_values: List[int]
    nuc_qqq: bool
    nuc_el_values: List[int]
    nuc_er_values: List[int]


@dataclass
class _RecallTfQualityInputs:
    intervals: List[Tuple[int, int]]
    tq_values: List[int]
    el_values: List[int]
    er_values: List[int]


@dataclass
class _RawLegacyRecallTags:
    nuc_starts: Sequence[int]
    nuc_lengths: Sequence[int]
    msp_starts: Sequence[int]
    msp_lengths: Sequence[int]


@dataclass
class _PreferredMmMlTags:
    mm_tag: object
    ml_tag: object


@dataclass
class _LegacyIntervalRow:
    start: int
    length: int
    score: Optional[int]


def _split_named_intervals(
    intervals: Sequence[Tuple[int, int]],
    prefix: str,
    read_length: int,
    qual_rows: Optional[Sequence[Sequence[int]]] = None,
) -> _SplitNamedIntervals:
    split_intervals: List[Tuple[int, int]] = []
    split_names: List[str] = []
    split_quals: Optional[List[Sequence[int]]] = [] if qual_rows is not None else None
    any_split = False
    for idx, (start, length) in enumerate(intervals):
        pieces = split_circular_interval(start, length, read_length)
        if len(pieces) > 1:
            any_split = True
        name = f"fhw_{prefix}_{idx}" if len(pieces) > 1 else f"fh_{prefix}_{idx}"
        for piece in pieces:
            split_intervals.append(piece)
            split_names.append(name)
            if split_quals is not None:
                split_quals.append(qual_rows[idx])
    return _SplitNamedIntervals(split_intervals, split_names, split_quals, any_split)


def _prepare_tag_output_frame(
    read,
    read_length: int,
    kept_nucs: Sequence[Tuple[int, int]],
    msps: Sequence[Tuple[int, int]],
    tf_intervals: Sequence[Tuple[int, int]],
    nq_values: Sequence[int],
    tq_values: Sequence[int],
    tf_el_values: Sequence[int],
    tf_er_values: Sequence[int],
    nuc_el_values: Optional[Sequence[int]] = None,
    nuc_er_values: Optional[Sequence[int]] = None,
) -> _TagOutputFrame:
    return _prepare_tag_output_frame_from_request(
        _TagOutputFrameRequest(
            read=read,
            read_length=read_length,
            kept_nucs=kept_nucs,
            msps=msps,
            tf_intervals=tf_intervals,
            nq_values=nq_values,
            tq_values=tq_values,
            tf_el_values=tf_el_values,
            tf_er_values=tf_er_values,
            nuc_el_values=nuc_el_values,
            nuc_er_values=nuc_er_values,
        ),
    )


def _prepare_tag_output_frame_from_request(
    request: _TagOutputFrameRequest,
) -> _TagOutputFrame:
    """Convert internal SEQ-frame calls to sorted molecular-frame tag rows."""
    nuc_qqq = (
        request.nuc_el_values is not None
        and request.nuc_er_values is not None
    )
    if not request.read_length:
        return _TagOutputFrame(
            nuc_intervals=list(request.kept_nucs),
            msp_intervals=list(request.msps),
            tf_intervals=list(request.tf_intervals),
            nq_values=list(request.nq_values),
            tq_values=list(request.tq_values),
            tf_el_values=list(request.tf_el_values),
            tf_er_values=list(request.tf_er_values),
            nuc_el_values=list(request.nuc_el_values)
            if request.nuc_el_values is not None else None,
            nuc_er_values=list(request.nuc_er_values)
            if request.nuc_er_values is not None else None,
        )

    rev = bool(getattr(request.read, 'is_reverse', False))

    def _mol(start, length):
        if rev:
            return flip_interval_frame(start, length, request.read_length)
        return int(start), int(length)

    nuc_recs = sorted(
        (
            _mol(start, length),
            request.nq_values[i],
            (request.nuc_er_values[i] if rev else request.nuc_el_values[i])
            if nuc_qqq else None,
            (request.nuc_el_values[i] if rev else request.nuc_er_values[i])
            if nuc_qqq else None,
        )
        for i, (start, length) in enumerate(request.kept_nucs)
    )
    tf_recs = sorted(
        (
            _mol(start, length),
            request.tq_values[i],
            request.tf_er_values[i] if rev else request.tf_el_values[i],
            request.tf_el_values[i] if rev else request.tf_er_values[i],
        )
        for i, (start, length) in enumerate(request.tf_intervals)
    )
    return _TagOutputFrame(
        nuc_intervals=[r[0] for r in nuc_recs],
        msp_intervals=sorted(
            _mol(start, length) for start, length in request.msps
        ),
        tf_intervals=[r[0] for r in tf_recs],
        nq_values=[r[1] for r in nuc_recs],
        tq_values=[r[1] for r in tf_recs],
        tf_el_values=[r[2] for r in tf_recs],
        tf_er_values=[r[3] for r in tf_recs],
        nuc_el_values=[r[2] for r in nuc_recs] if nuc_qqq else None,
        nuc_er_values=[r[3] for r in nuc_recs] if nuc_qqq else None,
    )


def build_llr_tables(model) -> Tuple[np.ndarray, np.ndarray]:
    """Return (llr_hit, llr_miss) lookup arrays, length N_CTX each.

    Assumes model.normalize_states() has been applied (state 0 = protected,
    state 1 = accessible). load_model_with_metadata enforces this.
    """
    EP = np.asarray(model.emissionprob_, dtype=np.float64)
    if EP.shape[0] != 2:
        raise ValueError(f"Expected 2-state model, got {EP.shape[0]}")
    if EP.shape[1] < UNMETH_OFFSET + N_CTX:
        raise ValueError(
            f"Emission table too small: {EP.shape[1]} columns, "
            f"need {UNMETH_OFFSET + N_CTX}")
    eps = 1e-30
    hit_prot = np.clip(EP[0, :N_CTX], eps, 1.0)
    hit_acc = np.clip(EP[1, :N_CTX], eps, 1.0)
    miss_prot = np.clip(EP[0, UNMETH_OFFSET:UNMETH_OFFSET + N_CTX], eps, 1.0)
    miss_acc = np.clip(EP[1, UNMETH_OFFSET:UNMETH_OFFSET + N_CTX], eps, 1.0)
    return (np.log(hit_prot) - np.log(hit_acc),
            np.log(miss_prot) - np.log(miss_acc))


def apply_emission_uplift(llr_hit: np.ndarray, llr_miss: np.ndarray,
                          model, uplift: float) -> Tuple[np.ndarray, np.ndarray]:
    """Sharpen the per-context emission table and rebuild LLR tables.

    For each context c:
        p_hit_acc_new(c)  = 1 - (1 - p_hit_acc(c)) ** uplift
        p_hit_prot_new(c) =      p_hit_prot(c)     ** uplift

    uplift > 1 moves the accessible state toward p(hit) = 1 and the
    protected state toward p(hit) = 0, which is appropriate when the
    underlying enzyme is more efficient than the trained model assumes
    (e.g. DddA on a DddB-trained model).
    """
    if abs(uplift - 1.0) < 1e-9:
        return llr_hit, llr_miss
    EP = np.asarray(model.emissionprob_, dtype=np.float64)
    eps = 1e-30
    hit_prot = np.clip(EP[0, :N_CTX], eps, 1.0)
    hit_acc = np.clip(EP[1, :N_CTX], eps, 1.0)
    miss_prot = np.clip(EP[0, UNMETH_OFFSET:UNMETH_OFFSET + N_CTX], eps, 1.0)
    miss_acc = np.clip(EP[1, UNMETH_OFFSET:UNMETH_OFFSET + N_CTX], eps, 1.0)
    p_hit_acc = hit_acc / (hit_acc + miss_acc)
    p_hit_prot = hit_prot / (hit_prot + miss_prot)
    p_hit_acc_new = 1.0 - np.power(np.clip(1.0 - p_hit_acc, eps, 1.0), uplift)
    p_hit_prot_new = np.power(np.clip(p_hit_prot, eps, 1.0), uplift)
    p_hit_acc_new = np.clip(p_hit_acc_new, eps, 1.0 - eps)
    p_hit_prot_new = np.clip(p_hit_prot_new, eps, 1.0 - eps)
    new_llr_miss = np.log(1.0 - p_hit_prot_new) - np.log(1.0 - p_hit_acc_new)
    new_llr_hit = np.log(p_hit_prot_new) - np.log(p_hit_acc_new)
    return new_llr_hit, new_llr_miss


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sort + merge a list of [start, end) intervals."""
    if not intervals:
        return []
    intervals = sorted((a, b) for a, b in intervals if b > a)
    merged = [list(intervals[0])]
    for a, b in intervals[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def _bounded_scan_interval(start, length, read_len: int) -> Optional[Tuple[int, int]]:
    start = int(start)
    length = int(length)
    if length <= 0:
        return None
    bounded = (max(0, start), min(int(read_len), start + length))
    return bounded if bounded[1] > bounded[0] else None


def _is_short_nuc_length(length, unify_threshold: int) -> bool:
    length = int(length)
    return 0 < length < int(unify_threshold)


def build_scan_intervals(ns: Sequence[int], nl: Sequence[int],
                         as_: Sequence[int], al: Sequence[int],
                         read_len: int, unify_threshold: int = 90
                         ) -> List[Tuple[int, int]]:
    """Construct the merged scan space.

    Sources:
      - all v2 MSPs (``as``/``al``)
      - all v2 nucs with ``nl < unify_threshold``
    """
    iv: List[Tuple[int, int]] = []
    for s, length in zip(as_, al):
        interval = _bounded_scan_interval(s, length, read_len)
        if interval is not None:
            iv.append(interval)
    for s, length in zip(ns, nl):
        if _is_short_nuc_length(length, unify_threshold):
            interval = _bounded_scan_interval(s, length, read_len)
            if interval is not None:
                iv.append(interval)
    return merge_intervals(iv)


def _is_target_code(code: int) -> bool:
    return (0 <= code < N_CTX) or (UNMETH_OFFSET <= code < UNMETH_OFFSET + N_CTX)


def _is_hit_code(code: int) -> bool:
    return 0 <= code < N_CTX


def _is_miss_code(code: int) -> bool:
    return UNMETH_OFFSET <= code < UNMETH_OFFSET + N_CTX


def _tf_call_from_scan_arrays(
    starts,
    ends,
    llrs,
    opps_arr,
    l_amb,
    r_amb,
    index: int,
) -> TFCall:
    return TFCall(
        start=int(starts[index]),
        length=int(ends[index] - starts[index]),
        llr=float(llrs[index]),
        n_opps=int(opps_arr[index]),
        left_ambiguity=int(l_amb[index]),
        right_ambiguity=int(r_amb[index]),
    )


@_numba_jit(nopython=True, cache=True)
def _call_tfs_numba(obs, lo, hi, llr_hit, llr_miss,
                    min_llr, min_opps):
    """Numba-JIT Kadane local-maximum scan with edge-ambiguity scoring.

    Returns five equal-length 1D numpy arrays:
      (starts, ends, llrs, opps, left_amb, right_amb)
    where ends is exclusive (call spans [start, end)).

    Observation code layout (must match encode_from_query_sequence, k=3):
      0 <= code < 4096         -> hit with hexamer context code
      code == 4096              -> non-target methylated (unused in practice)
      4097 <= code < 4097+4096  -> miss with context (code - 4097)
      other                     -> non-target (neutral)

    Constants inlined because numba dislikes reading module globals.
    """
    N_CTX = 4096
    UNMETH_OFFSET = 4097

    # Preallocate result buffers; worst-case per position is unlikely
    # to produce more than (hi-lo)//3 calls.
    max_calls = max(4, (hi - lo) // 2 + 1)
    starts = np.empty(max_calls, dtype=np.int64)
    ends = np.empty(max_calls, dtype=np.int64)
    llrs = np.empty(max_calls, dtype=np.float64)
    opps_out = np.empty(max_calls, dtype=np.int64)
    left_amb = np.empty(max_calls, dtype=np.int64)
    right_amb = np.empty(max_calls, dtype=np.int64)
    n_calls = 0

    cur_start = -1  # sentinel for "not in a run"
    running = 0.0
    opps = 0
    peak_llr = 0.0
    peak_end = lo
    peak_opps = 0

    for i in range(lo, hi):
        code = obs[i]
        is_opp = False
        step = 0.0
        if 0 <= code < N_CTX:
            step = llr_hit[code]
            is_opp = True
        elif UNMETH_OFFSET <= code < UNMETH_OFFSET + N_CTX:
            step = llr_miss[code - UNMETH_OFFSET]
            is_opp = True

        if cur_start < 0:
            if step > 0.0:
                cur_start = i
                running = step
                opps = 1 if is_opp else 0
                peak_llr = running
                peak_end = i + 1
                peak_opps = opps
            continue

        running += step
        if is_opp:
            opps += 1
        if running > peak_llr:
            peak_llr = running
            peak_end = i + 1
            peak_opps = opps

        if running <= 0.0:
            if peak_llr >= min_llr and peak_opps >= min_opps and n_calls < max_calls:
                starts[n_calls] = cur_start
                ends[n_calls] = peak_end
                llrs[n_calls] = peak_llr
                opps_out[n_calls] = peak_opps
                n_calls += 1
            cur_start = -1
            running = 0.0
            opps = 0
            peak_llr = 0.0
            peak_end = i + 1
            peak_opps = 0

    # End-of-interval flush
    if cur_start >= 0 and peak_llr >= min_llr and peak_opps >= min_opps \
            and n_calls < max_calls:
        starts[n_calls] = cur_start
        ends[n_calls] = peak_end
        llrs[n_calls] = peak_llr
        opps_out[n_calls] = peak_opps
        n_calls += 1

    # Edge ambiguity: walk left from each start to find nearest hit (or lo),
    # walk right from each end to find nearest hit (or hi).
    for ci in range(n_calls):
        s = starts[ci]
        e = ends[ci]
        # Left
        amb = 0
        j = s - 1
        while j >= lo:
            c = obs[j]
            if 0 <= c < N_CTX:
                break
            amb += 1
            j -= 1
        left_amb[ci] = amb
        # Right
        amb = 0
        j = e
        while j < hi:
            c = obs[j]
            if 0 <= c < N_CTX:
                break
            amb += 1
            j += 1
        right_amb[ci] = amb

    return (starts[:n_calls], ends[:n_calls], llrs[:n_calls],
            opps_out[:n_calls], left_amb[:n_calls], right_amb[:n_calls])


def call_tfs_in_interval(obs: np.ndarray, lo: int, hi: int,
                         llr_hit: np.ndarray, llr_miss: np.ndarray,
                         min_llr: float, min_opps: int) -> List[TFCall]:
    """Kadane local-maximum scan on obs[lo:hi].

    Thin Python wrapper around ``_call_tfs_numba``; builds TFCall objects
    from the numpy result arrays.
    """
    if hi <= lo:
        return []
    # Ensure dtypes numba can bind to cleanly
    obs_arr = np.ascontiguousarray(obs, dtype=np.int32)
    hit_arr = np.ascontiguousarray(llr_hit, dtype=np.float64)
    miss_arr = np.ascontiguousarray(llr_miss, dtype=np.float64)
    starts, ends, llrs, opps_arr, l_amb, r_amb = _call_tfs_numba(
        obs_arr, int(lo), int(hi), hit_arr, miss_arr,
        float(min_llr), int(min_opps),
    )
    calls: List[TFCall] = []
    for i in range(len(starts)):
        calls.append(
            _tf_call_from_scan_arrays(
                starts, ends, llrs, opps_arr, l_amb, r_amb, i,
            )
        )
    return calls


def _extract_daf_iupac_modifications(read, seq: str):
    try:
        st_tag = read.get_tag('st')
    except KeyError:
        st_tag = None
    mod_pos, strand, seq = extract_daf_iupac_positions(seq, st_tag)
    return mod_pos, strand, seq


def _extract_daf_md_modifications(read, seq: str):
    md_result = getattr(read, '_daf_md_result', None)
    if md_result is None and hasattr(read, 'get_aligned_pairs'):
        from fiberhmm.daf.encoder import get_daf_positions
        md_result = get_daf_positions(read)
    if md_result is None:
        return None
    ct_pos, ga_pos, strand_tag = md_result
    strand = daf_strand_from_tag(strand_tag)
    if strand == '+':
        return set(ct_pos), strand, seq.upper()
    return set(ga_pos), strand if strand != '.' else '-', seq.upper()


def _extract_mm_ml_modifications(read, seq: str, mode: str, mm_tag, ml_tag):
    mod_pos = parse_mm_tag_query_positions(
        mm_tag, ml_tag, seq, read.is_reverse,
        prob_threshold=125, mode=mode,
    )
    if mode == 'daf':
        strand = detect_daf_strand(seq, mod_pos)
    else:
        strand = '.'
    return mod_pos, strand, seq


def _preferred_mm_ml_tags(read):
    mm_tag = get_preferred_tag(read, 'MM', 'Mm', '')
    ml_tag = get_preferred_tag(read, 'ML', 'Ml', None)
    if not _has_mm_ml_inputs(mm_tag, ml_tag):
        return None
    return _PreferredMmMlTags(mm_tag, ml_tag)


def _extract_mm_ml_or_daf_md_modifications(read, seq: str, mode: str):
    tags = _preferred_mm_ml_tags(read)
    if tags is None:
        if mode == 'daf':
            return _extract_daf_md_modifications(read, seq)
        return None
    return _extract_mm_ml_modifications(read, seq, mode, tags.mm_tag, tags.ml_tag)


def extract_modifications(read, mode: str, context_size: int = 3
                          ) -> Optional[Tuple[set, str, str]]:
    """Pull (mod_positions, strand, sequence) for a read.

    Returns None if the read can't be processed (no MM tag, no sequence).
    Uses the manual MM/ML parser instead of pysam.modified_bases (the
    latter segfaults on some long Hia5 reads; SIGSEGV is uncatchable).
    """
    seq = read.query_sequence
    if seq is None or len(seq) < 2 * context_size + 1:
        return None
    if mode == 'daf' and has_iupac_encoding(seq):
        return _extract_daf_iupac_modifications(read, seq)
    return _extract_mm_ml_or_daf_md_modifications(read, seq, mode)


def _raw_legacy_recall_tags(read):
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
    return _RawLegacyRecallTags(ns_raw, nl_raw, as_raw, al_raw)


def _positive_length_intervals(starts, lengths) -> List[Tuple[int, int]]:
    return [
        (int(start), int(length))
        for start, length in zip(starts, lengths)
        if int(length) > 0
    ]


def _kept_legacy_nuc_interval(
    start,
    length,
    tf_intervals: Sequence[Tuple[int, int]],
    unify_threshold: int,
) -> Optional[Tuple[int, int]]:
    start = int(start)
    length = int(length)
    if length <= 0:
        return None
    if length >= unify_threshold:
        return start, length
    nuc_end = start + length
    if any(ts < nuc_end and te > start for ts, te in tf_intervals):
        return None
    return start, length


def recall_read(
    read,
    llr_hit: np.ndarray,
    llr_miss: np.ndarray,
    mode: str,
    context_size: int,
    min_llr: float,
    min_opps: int,
    unify_threshold: int,
) -> Tuple[List[TFCall], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Process one read.

    Returns:
        (tf_calls, kept_nuc_intervals, msp_intervals)
        - tf_calls: emitted TF calls
        - kept_nuc_intervals: v2 nucs that survive --unify
                              (>= unify_threshold OR no overlapping TF call)
        - msp_intervals: v2 MSPs unchanged
    """
    raw_tags = _raw_legacy_recall_tags(read)

    if len(raw_tags.nuc_starts) == 0 and len(raw_tags.msp_starts) == 0:
        return [], [], []

    # Tags are stored molecular frame; recall works in SEQ (query) frame, and
    # write_ma_tags flips back to molecular on output. Flip on read here.
    ns_raw, nl_raw = flip_intervals_to_seq(
        raw_tags.nuc_starts,
        raw_tags.nuc_lengths,
        read,
    )
    as_raw, al_raw = flip_intervals_to_seq(
        raw_tags.msp_starts,
        raw_tags.msp_lengths,
        read,
    )

    extracted = extract_modifications(read, mode, context_size)
    if extracted is None:
        # Pass through v2 calls unchanged
        nucs = _positive_length_intervals(ns_raw, nl_raw)
        msps = _positive_length_intervals(as_raw, al_raw)
        return [], nucs, msps

    mod_pos, strand, seq = extracted
    obs = encode_from_query_sequence(
        seq, mod_pos, edge_trim=10, mode=mode, strand=strand,
        context_size=context_size,
        is_reverse=bool(read.is_reverse),
    )
    read_len = len(seq)

    intervals = build_scan_intervals(ns_raw, nl_raw, as_raw, al_raw,
                                      read_len, unify_threshold=unify_threshold)

    tf_calls: List[TFCall] = []
    for lo, hi in intervals:
        tf_calls.extend(call_tfs_in_interval(
            obs, lo, hi, llr_hit, llr_miss, min_llr, min_opps,
        ))

    # Unify: drop v2 short-nucs (nl < threshold) that overlap any TF call.
    msps = _positive_length_intervals(as_raw, al_raw)
    kept_nucs: List[Tuple[int, int]] = []
    tf_intervals = [(c.start, c.start + c.length) for c in tf_calls]
    for s, length in zip(ns_raw, nl_raw):
        kept = _kept_legacy_nuc_interval(s, length, tf_intervals, unify_threshold)
        if kept is not None:
            kept_nucs.append(kept)

    return tf_calls, kept_nucs, msps


def _clamp_u8(value) -> int:
    return max(0, min(255, int(value)))


def _split_legacy_interval_row(interval: Tuple[int, int],
                               read_length: int,
                               score=None):
    encoded_score = None if score is None else _clamp_u8(score)
    return [
        _LegacyIntervalRow(piece[0], piece[1], encoded_score)
        for piece in split_circular_interval(interval[0], interval[1], read_length)
    ]


def _split_legacy_interval_rows(intervals: Sequence[Tuple[int, int]],
                                read_length: int,
                                scores: Optional[Sequence[int]] = None):
    if scores is None:
        rows = [
            row
            for interval in intervals
            for row in _split_legacy_interval_row(interval, read_length)
        ]
    else:
        rows = [
            row
            for interval, score in zip(intervals, scores)
            for row in _split_legacy_interval_row(interval, read_length, score)
        ]
    rows.sort(key=lambda row: (int(row.start), int(row.length)))
    return rows


def _legacy_starts_lengths(rows):
    return (
        [int(row.start) for row in rows],
        [int(row.length) for row in rows],
    )


def _nuc_quality_rows(nq_values: Sequence[int], nuc_qqq: bool,
                      nuc_el_values: Optional[Sequence[int]] = None,
                      nuc_er_values: Optional[Sequence[int]] = None):
    if nuc_qqq:
        return [
            [q, el, er]
            for q, el, er in zip(
                nq_values,
                nuc_el_values or [],
                nuc_er_values or [],
            )
        ]
    return [[q] for q in nq_values]


def _tf_quality_rows(tq_vals: Sequence[int], el_vals: Sequence[int],
                     er_vals: Sequence[int]):
    return [[tq, el, er] for tq, el, er in zip(tq_vals, el_vals, er_vals)]


def _format_split_aq(nuc_q_split, tf_q_split, nuc_qqq: bool):
    split_nq_values = [row[0] for row in (nuc_q_split or [])]
    split_tq_vals = [row[0] for row in (tf_q_split or [])]
    split_el_vals = [row[1] for row in (tf_q_split or [])]
    split_er_vals = [row[2] for row in (tf_q_split or [])]
    if nuc_qqq:
        split_nuc_el = [row[1] for row in (nuc_q_split or [])]
        split_nuc_er = [row[2] for row in (nuc_q_split or [])]
    else:
        split_nuc_el = ()
        split_nuc_er = ()
    return format_aq_array(
        nq_values=split_nq_values,
        tf_q_values=split_tq_vals,
        tf_lq_values=split_el_vals,
        tf_rq_values=split_er_vals,
        nuc_lq_values=split_nuc_el,
        nuc_rq_values=split_nuc_er,
    )


def _recall_nuc_quality_inputs(kept_nucs: Sequence[Tuple[int, int]],
                               nq_for_kept_nucs: Optional[Sequence[int]],
                               nuc_el_for_kept: Optional[Sequence[int]],
                               nuc_er_for_kept: Optional[Sequence[int]]
                               ) -> _RecallNucQualityInputs:
    nq_values = list(nq_for_kept_nucs) if nq_for_kept_nucs is not None \
        else [0] * len(kept_nucs)
    if len(nq_values) != len(kept_nucs):
        raise ValueError("nq_values length must match kept_nucs length")

    nuc_qqq = nuc_el_for_kept is not None and nuc_er_for_kept is not None
    nuc_el_values = []
    nuc_er_values = []
    if nuc_qqq:
        nuc_el_values = list(nuc_el_for_kept)
        nuc_er_values = list(nuc_er_for_kept)
        if not (len(nuc_el_values) == len(nuc_er_values) == len(kept_nucs)):
            raise ValueError("nuc edge arrays must match kept_nucs length")

    return _RecallNucQualityInputs(
        nq_values,
        nuc_qqq,
        nuc_el_values,
        nuc_er_values,
    )


def _recall_tf_quality_inputs(tf_calls: Sequence[TFCall]) -> _RecallTfQualityInputs:
    return _RecallTfQualityInputs(
        intervals=[(c.start, c.length) for c in tf_calls],
        tq_values=[llr_to_tq(c.llr) for c in tf_calls],
        el_values=[ambiguity_to_edge(c.left_ambiguity) for c in tf_calls],
        er_values=[ambiguity_to_edge(c.right_ambiguity) for c in tf_calls],
    )


def _recall_tag_payload_from_output_frame(
    output_frame: _TagOutputFrame,
    read_length: int,
    nuc_qqq: bool,
) -> _RecallTagPayload:
    nuc_q_rows = _nuc_quality_rows(
        output_frame.nq_values,
        nuc_qqq,
        output_frame.nuc_el_values if nuc_qqq else None,
        output_frame.nuc_er_values if nuc_qqq else None,
    )
    tf_q_rows = _tf_quality_rows(
        output_frame.tq_values,
        output_frame.tf_el_values,
        output_frame.tf_er_values,
    )
    split_nucs = _split_named_intervals(
        output_frame.nuc_intervals, "nuc", read_length, nuc_q_rows,
    )
    split_msps = _split_named_intervals(
        output_frame.msp_intervals, "msp", read_length, None,
    )
    split_tfs = _split_named_intervals(
        output_frame.tf_intervals, "tf", read_length, tf_q_rows,
    )

    return _RecallTagPayload(
        kept_nucs=output_frame.nuc_intervals,
        msps=output_frame.msp_intervals,
        tf_intervals=output_frame.tf_intervals,
        nq_values=output_frame.nq_values,
        nuc_qqq=nuc_qqq,
        ma_nucs=split_nucs.intervals,
        ma_msps=split_msps.intervals,
        ma_tfs=split_tfs.intervals,
        nuc_names=split_nucs.names,
        msp_names=split_msps.names,
        tf_names=split_tfs.names,
        nuc_q_split=split_nucs.qual_rows,
        tf_q_split=split_tfs.qual_rows,
        needs_an=split_nucs.any_split or split_msps.any_split or split_tfs.any_split,
    )


def _build_recall_tag_payload(
    read,
    read_length: int,
    tf_calls: Sequence[TFCall],
    kept_nucs: Sequence[Tuple[int, int]],
    msps: Sequence[Tuple[int, int]],
    nq_for_kept_nucs: Optional[Sequence[int]],
    nuc_el_for_kept: Optional[Sequence[int]],
    nuc_er_for_kept: Optional[Sequence[int]],
) -> _RecallTagPayload:
    nuc_quality = _recall_nuc_quality_inputs(
        kept_nucs, nq_for_kept_nucs, nuc_el_for_kept, nuc_er_for_kept,
    )
    tf_quality = _recall_tf_quality_inputs(tf_calls)

    # FiberHMM works in SEQ coords internally. Tags are written in molecular
    # frame and sorted within each annotation type for fibertools/spec readers.
    output_frame = _prepare_tag_output_frame(
        read,
        read_length,
        kept_nucs,
        msps,
        tf_quality.intervals,
        nuc_quality.nq_values,
        tf_quality.tq_values,
        tf_quality.el_values,
        tf_quality.er_values,
        nuc_quality.nuc_el_values if nuc_quality.nuc_qqq else None,
        nuc_quality.nuc_er_values if nuc_quality.nuc_qqq else None,
    )

    return _recall_tag_payload_from_output_frame(
        output_frame,
        read_length,
        nuc_quality.nuc_qqq,
    )


def _write_spec_recall_tags(
    read,
    read_length: int,
    payload: _RecallTagPayload,
) -> None:
    # Spec mode: write MA + AQ. The fiberseq Molecular-annotation spec
    # requires MA to contain at least one annotation, and AQ only when some
    # annotation type specifies quality values.
    has_any_annotation = bool(payload.ma_nucs or payload.ma_msps or payload.ma_tfs)
    if not has_any_annotation:
        clear_tags(read, ('MA', 'AQ', 'AN'))
        return

    ma = format_ma_tag(
        read_length=read_length,
        nuc_intervals=payload.ma_nucs,
        msp_intervals=payload.ma_msps,
        tf_intervals=payload.ma_tfs,
        nuc_qual_spec='QQQ' if payload.nuc_qqq else 'Q',
    )
    read.set_tag('MA', ma, value_type='Z')
    if payload.needs_an:
        names = payload.nuc_names + payload.msp_names + payload.tf_names
        read.set_tag('AN', format_an_tag(names), value_type='Z')
    else:
        clear_tags(read, ('AN',))

    has_quality = bool(payload.ma_nucs or payload.ma_tfs)
    if has_quality:
        read.set_tag(
            'AQ',
            _format_split_aq(
                payload.nuc_q_split,
                payload.tf_q_split,
                payload.nuc_qqq,
            ),
        )
    else:
        clear_tags(read, ('AQ',))


def _write_legacy_recall_tags(read, read_length: int,
                              kept_nucs: Sequence[Tuple[int, int]],
                              msps: Sequence[Tuple[int, int]],
                              tf_intervals: Sequence[Tuple[int, int]],
                              nq_for_kept_nucs: Optional[Sequence[int]],
                              nq_values: Sequence[int],
                              downstream_compat: bool) -> None:
    import array as pyarray

    # Build the ns/nl track. In default mode it's nucleosomes only.
    # In downstream_compat mode, TF calls are merged in, sorted by start.
    if downstream_compat and tf_intervals:
        combined = list(kept_nucs) + list(tf_intervals)
    else:
        combined = list(kept_nucs)
    legacy_nuc_rows = _split_legacy_interval_rows(combined, read_length)
    ns, nl = _legacy_starts_lengths(legacy_nuc_rows)

    legacy_msp_rows = _split_legacy_interval_rows(msps, read_length)
    a_s, a_l = _legacy_starts_lengths(legacy_msp_rows)

    if ns:
        read.set_tag('ns', pyarray.array('I', ns))
        read.set_tag('nl', pyarray.array('I', nl))
    else:
        clear_tags(read, ('ns', 'nl', 'nq'))
    if a_s:
        read.set_tag('as', pyarray.array('I', a_s))
        read.set_tag('al', pyarray.array('I', a_l))
    else:
        clear_tags(read, ('as', 'al', 'aq'))
    # nq must have len == len(ns) per fibertools invariant. If we wrote
    # new ns/nl without fresh scores, drop any stale nq from the input BAM
    # to avoid len(nq) != len(ns) failing ft validate (see fibertools-rs
    # bamannotations.rs set_qual assert).
    if ns and nq_for_kept_nucs is not None:
        legacy_nq_rows = _split_legacy_interval_rows(kept_nucs, read_length, nq_values)
        legacy_nq = [row.score for row in legacy_nq_rows]
        if len(legacy_nq) != len(ns):
            legacy_nq = [0] * len(ns)
        read.set_tag('nq', pyarray.array('B', legacy_nq))
    elif ns:
        clear_tags(read, ('nq',))
    # Same for aq: stale per-msp qualities from input would mismatch
    # the refreshed as/al length.
    if a_s:
        clear_tags(read, ('aq',))


def write_ma_tags(read, read_length: int,
                  tf_calls: Sequence[TFCall],
                  kept_nucs: Sequence[Tuple[int, int]],
                  msps: Sequence[Tuple[int, int]],
                  nq_for_kept_nucs: Optional[Sequence[int]] = None,
                  also_write_legacy: bool = True,
                  downstream_compat: bool = False,
                  nuc_el_for_kept: Optional[Sequence[int]] = None,
                  nuc_er_for_kept: Optional[Sequence[int]] = None) -> None:
    """Set MA/AQ (and optionally legacy ns/nl/as/al) tags on the read in place.

    Three output modes:

    - **Default** (``also_write_legacy=True, downstream_compat=False``):
      Write MA/AQ per the Molecular-annotation spec. Also refresh legacy
      ns/nl/as/al to reflect the unified call set (v2 short-nucs demoted
      to tf+ in MA are removed from ns/nl). TF calls live ONLY in
      MA/AQ. This is the preferred output for tools that understand the
      spec (FiberBrowser, future fibertools-rs releases).

    - ``downstream_compat=True``: **Skip MA/AQ entirely**. Write TF calls
      INTO the legacy ns/nl tag alongside nucleosomes, with entries
      sorted by start position. Any tool that reads ns/nl (legacy
      fibertools-rs, custom scripts) will see the full call set as
      "footprints" with a mix of sizes. No TF-specific scoring is
      preserved -- only positions and lengths. Use this only when you
      need to feed older downstream tools.

    - ``also_write_legacy=False``: Write MA/AQ only; leave existing
      ns/nl/as/al in place (they will be stale vs. the unified set but
      preserved unchanged for reference).

    ``downstream_compat=True`` and ``also_write_legacy=False`` are mutually
    exclusive; compat mode always writes the legacy track.
    """
    if downstream_compat and not also_write_legacy:
        raise ValueError(
            "downstream_compat=True requires also_write_legacy=True "
            "(compat mode writes TF calls into the legacy ns/nl track)."
        )

    payload = _build_recall_tag_payload(
        read,
        read_length,
        tf_calls,
        kept_nucs,
        msps,
        nq_for_kept_nucs,
        nuc_el_for_kept,
        nuc_er_for_kept,
    )

    if not downstream_compat:
        _write_spec_recall_tags(read, read_length, payload)
    else:
        # Compat mode: strip any stale MA/AQ so consumers that see both
        # tags don't get out-of-sync views.
        clear_tags(read, ('MA', 'AQ', 'AN'))

    if also_write_legacy:
        _write_legacy_recall_tags(
            read,
            read_length,
            payload.kept_nucs,
            payload.msps,
            payload.tf_intervals,
            nq_for_kept_nucs,
            payload.nq_values,
            downstream_compat,
        )
