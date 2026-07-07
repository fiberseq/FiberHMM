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
    detect_daf_strand,
    encode_from_query_sequence,
    extract_daf_iupac_positions,
    has_iupac_encoding,
    parse_mm_tag_query_positions,
)
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
    'hia5':   dict(min_llr=5.0, emission_uplift=1.0),
    # EcoGII deposits the same m6A mark as Hia5; the ecogii model carries EcoGII-calibrated
    # emissions, so the LLR needs no uplift. Same min_llr as Hia5 (same chemistry).
    'ecogii': dict(min_llr=5.0, emission_uplift=1.0),
    # M.SssI CpG 5mC footprinting (nanopore); emissions calibrated from naked controls.
    'sssi':   dict(min_llr=5.0, emission_uplift=1.0),
    'dddb':   dict(min_llr=4.0, emission_uplift=1.0),
    'ddda':   dict(min_llr=5.0, emission_uplift=1.0),
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
        s = int(s)
        length = int(length)
        if length > 0:
            iv.append((s, s + length))
    for s, length in zip(ns, nl):
        s = int(s)
        length = int(length)
        if 0 < length < unify_threshold:
            iv.append((s, s + length))
    iv = [(max(0, a), min(read_len, b)) for a, b in iv]
    iv = [(a, b) for a, b in iv if b > a]
    return merge_intervals(iv)


def _is_target_code(code: int) -> bool:
    return (0 <= code < N_CTX) or (UNMETH_OFFSET <= code < UNMETH_OFFSET + N_CTX)


def _is_hit_code(code: int) -> bool:
    return 0 <= code < N_CTX


def _is_miss_code(code: int) -> bool:
    return UNMETH_OFFSET <= code < UNMETH_OFFSET + N_CTX


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
        calls.append(TFCall(
            start=int(starts[i]),
            length=int(ends[i] - starts[i]),
            llr=float(llrs[i]),
            n_opps=int(opps_arr[i]),
            left_ambiguity=int(l_amb[i]),
            right_ambiguity=int(r_amb[i]),
        ))
    return calls


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
        try:
            st_tag = read.get_tag('st')
        except KeyError:
            st_tag = None
        mod_pos, strand, seq = extract_daf_iupac_positions(seq, st_tag)
        return mod_pos, strand, seq
    try:
        mm_tag = read.get_tag('MM') if read.has_tag('MM') else read.get_tag('Mm')
    except KeyError:
        mm_tag = ''
    try:
        ml_tag = read.get_tag('ML') if read.has_tag('ML') else read.get_tag('Ml')
    except KeyError:
        ml_tag = []
    if not mm_tag or not ml_tag:
        if mode == 'daf':
            md_result = getattr(read, '_daf_md_result', None)
            if md_result is None and hasattr(read, 'get_aligned_pairs'):
                from fiberhmm.daf.encoder import get_daf_positions
                md_result = get_daf_positions(read)
            if md_result is not None:
                ct_pos, ga_pos, strand_tag = md_result
                if strand_tag == 'CT':
                    return set(ct_pos), '+', seq.upper()
                return set(ga_pos), '-', seq.upper()
        return None
    mod_pos = parse_mm_tag_query_positions(
        mm_tag, ml_tag, seq, read.is_reverse,
        prob_threshold=125, mode=mode,
    )
    if mode == 'daf':
        strand = detect_daf_strand(seq, mod_pos)
    else:
        strand = '.'
    return mod_pos, strand, seq


def recall_read(read, llr_hit: np.ndarray, llr_miss: np.ndarray,
                mode: str, context_size: int,
                min_llr: float, min_opps: int,
                unify_threshold: int,
                input_molecular_frame: bool = True) -> Tuple[List[TFCall], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Process one read.

    Returns:
        (tf_calls, kept_nuc_intervals, msp_intervals)
        - tf_calls: emitted TF calls
        - kept_nuc_intervals: v2 nucs that survive --unify
                              (>= unify_threshold OR no overlapping TF call)
        - msp_intervals: v2 MSPs unchanged

    ``input_molecular_frame`` controls how the read's existing ns/nl/as/al are
    interpreted: True (default) = molecular frame (current FiberHMM output),
    flipped to seq for recall; False = legacy/v1.0 SEQ-frame tags, used as-is.
    Pass the wrong value and reverse-strand calls get mis-placed (they land on
    accessible m6A-rich DNA -- e.g. open promoters fill with spurious nuc).
    """
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
        return [], [], []

    # Current FiberHMM stores ns/nl/as/al in MOLECULAR frame; recall works in
    # SEQ (query) frame, and write_ma_tags flips back to molecular on output --
    # so flip on read here. But legacy/v1.0 BAMs already store these tags in SEQ
    # frame (no @CO coord marker); flipping those a second time mis-places every
    # reverse-strand call. When input_molecular_frame is False, use them as-is.
    # Forward reads are unaffected either way (the frames coincide).
    if input_molecular_frame:
        ns_raw, nl_raw = flip_intervals_to_seq(ns_raw, nl_raw, read)
        as_raw, al_raw = flip_intervals_to_seq(as_raw, al_raw, read)

    extracted = extract_modifications(read, mode, context_size)
    if extracted is None:
        # Pass through v2 calls unchanged
        nucs = [
            (int(s), int(length))
            for s, length in zip(ns_raw, nl_raw)
            if int(length) > 0
        ]
        msps = [
            (int(s), int(length))
            for s, length in zip(as_raw, al_raw)
            if int(length) > 0
        ]
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
    msps = [
        (int(s), int(length))
        for s, length in zip(as_raw, al_raw)
        if int(length) > 0
    ]
    kept_nucs: List[Tuple[int, int]] = []
    tf_intervals = [(c.start, c.start + c.length) for c in tf_calls]
    for s, length in zip(ns_raw, nl_raw):
        s = int(s)
        length = int(length)
        if length <= 0:
            continue
        if length >= unify_threshold:
            kept_nucs.append((s, length))
            continue
        # Short v2 nuc -- drop if overlapped by any TF call
        nuc_end = s + length
        if any(ts < nuc_end and te > s for ts, te in tf_intervals):
            continue
        kept_nucs.append((s, length))

    return tf_calls, kept_nucs, msps


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
    import array as pyarray

    if downstream_compat and not also_write_legacy:
        raise ValueError(
            "downstream_compat=True requires also_write_legacy=True "
            "(compat mode writes TF calls into the legacy ns/nl track)."
        )

    # Default nq for kept nucs to 0 (sentinel for "unverified") if not provided.
    nq_values = list(nq_for_kept_nucs) if nq_for_kept_nucs is not None \
        else [0] * len(kept_nucs)
    if len(nq_values) != len(kept_nucs):
        raise ValueError("nq_values length must match kept_nucs length")

    # nuc+QQQ mode: the nuc recaller supplies per-nuc edge-sharpness bytes.
    # When present, nucleosomes carry (nq, el, er) like tf+QQQ; otherwise the
    # legacy nuc+Q (single nq byte) layout is used.
    nuc_qqq = nuc_el_for_kept is not None and nuc_er_for_kept is not None
    if nuc_qqq:
        nuc_el_values = list(nuc_el_for_kept)
        nuc_er_values = list(nuc_er_for_kept)
        if not (len(nuc_el_values) == len(nuc_er_values) == len(kept_nucs)):
            raise ValueError("nuc edge arrays must match kept_nucs length")

    tf_intervals = [(c.start, c.length) for c in tf_calls]
    tq_vals = [llr_to_tq(c.llr) for c in tf_calls]
    el_vals = [ambiguity_to_edge(c.left_ambiguity) for c in tf_calls]
    er_vals = [ambiguity_to_edge(c.right_ambiguity) for c in tf_calls]

    # Coordinate frame + ordering for fibertools / Molecular-annotation spec:
    #  - FiberHMM works in SEQ (query_sequence, forward-reference) coords, but
    #    ns/nl/as/al and MA must be MOLECULAR (original-fiber) frame. For a
    #    reverse-mapped read the frames are reverse complements, so flip each
    #    interval [s,s+l) -> [L-(s+l), L-s) and swap its left/right edge bytes.
    #  - fibertools requires per-feature positions sorted ascending, so ALWAYS
    #    re-sort by (molecular) start -- the recaller can append promoted nucs
    #    out of order even on forward reads.
    if read_length:
        rev = bool(getattr(read, 'is_reverse', False))

        def _mol(s, length):
            return flip_interval_frame(s, length, read_length) if rev else (int(s), int(length))

        nuc_recs = sorted(
            (_mol(s, length), nq_values[i],
             (nuc_er_values[i] if rev else nuc_el_values[i]) if nuc_qqq else None,
             (nuc_el_values[i] if rev else nuc_er_values[i]) if nuc_qqq else None)
            for i, (s, length) in enumerate(kept_nucs)
        )
        kept_nucs = [r[0] for r in nuc_recs]
        nq_values = [r[1] for r in nuc_recs]
        if nuc_qqq:
            nuc_el_values = [r[2] for r in nuc_recs]
            nuc_er_values = [r[3] for r in nuc_recs]
        msps = sorted(_mol(s, length) for s, length in msps)
        tf_recs = sorted(
            (_mol(s, length), tq_vals[i],
             er_vals[i] if rev else el_vals[i],
             el_vals[i] if rev else er_vals[i])
            for i, (s, length) in enumerate(tf_intervals)
        )
        tf_intervals = [r[0] for r in tf_recs]
        tq_vals = [r[1] for r in tf_recs]
        el_vals = [r[2] for r in tf_recs]
        er_vals = [r[3] for r in tf_recs]

    def split_named_intervals(intervals, prefix, qual_rows=None):
        split_intervals = []
        split_names = []
        split_quals = [] if qual_rows is not None else None
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
        return split_intervals, split_names, split_quals, any_split

    if nuc_qqq:
        nuc_q_rows = [[q, el, er]
                      for q, el, er in zip(nq_values, nuc_el_values, nuc_er_values)]
    else:
        nuc_q_rows = [[q] for q in nq_values]
    tf_q_rows = [[tq, el, er] for tq, el, er in zip(tq_vals, el_vals, er_vals)]
    ma_nucs, nuc_names, nuc_q_split, nuc_split = split_named_intervals(
        kept_nucs, "nuc", nuc_q_rows,
    )
    ma_msps, msp_names, _msp_q_split, msp_split = split_named_intervals(
        msps, "msp", None,
    )
    ma_tfs, tf_names, tf_q_split, tf_split = split_named_intervals(
        tf_intervals, "tf", tf_q_rows,
    )
    needs_an = nuc_split or msp_split or tf_split

    if not downstream_compat:
        # Spec mode: write MA + AQ. The fiberseq Molecular-annotation spec
        # (https://github.com/fiberseq/Molecular-annotation-spec) requires:
        #   - the MA string to contain >= 1 annotation (regex
        #     ^\d+;(...annotation...;?)+$), so don't emit MA for reads with
        #     no nucs/msps/tfs at all
        #   - AQ to only be present if SOME annotation type specifies P or Q
        #     (we use Q on nuc+ and tf+; if neither has any annotations and
        #     only msp+ is emitted, AQ stays unwritten)
        has_any_annotation = bool(ma_nucs or ma_msps or ma_tfs)
        if not has_any_annotation:
            # Strip any stale tags, leave the read with no MA/AQ
            for tag in ('MA', 'AQ', 'AN'):
                if read.has_tag(tag):
                    try:
                        read.set_tag(tag, None)
                    except Exception:
                        pass
        else:
            ma = format_ma_tag(
                read_length=read_length,
                nuc_intervals=ma_nucs,
                msp_intervals=ma_msps,
                tf_intervals=ma_tfs,
                nuc_qual_spec='QQQ' if nuc_qqq else 'Q',
            )
            read.set_tag('MA', ma, value_type='Z')
            if needs_an:
                read.set_tag('AN', format_an_tag(nuc_names + msp_names + tf_names),
                             value_type='Z')
            elif read.has_tag('AN'):
                try:
                    read.set_tag('AN', None)
                except Exception:
                    pass
            # AQ only carries values for nuc+Q and tf+QQQ. If neither is
            # present in this read, no quality type is in MA -> spec says
            # AQ must not be written.
            has_quality = bool(ma_nucs or ma_tfs)
            if has_quality:
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
                aq = format_aq_array(
                    nq_values=split_nq_values,
                    tf_q_values=split_tq_vals,
                    tf_lq_values=split_el_vals,
                    tf_rq_values=split_er_vals,
                    nuc_lq_values=split_nuc_el,
                    nuc_rq_values=split_nuc_er,
                )
                read.set_tag('AQ', aq)
            elif read.has_tag('AQ'):
                try:
                    read.set_tag('AQ', None)
                except Exception:
                    pass
    else:
        # Compat mode: strip any stale MA/AQ so consumers that see both
        # tags don't get out-of-sync views.
        for tag in ('MA', 'AQ', 'AN'):
            if read.has_tag(tag):
                try:
                    read.set_tag(tag, None)
                except Exception:
                    pass

    if also_write_legacy:
        # Build the ns/nl track. In default mode it's nucleosomes only.
        # In downstream_compat mode, TF calls are merged in, sorted by start.
        if downstream_compat and tf_intervals:
            combined = list(kept_nucs) + list(tf_intervals)
        else:
            combined = list(kept_nucs)
        legacy_nuc_rows = [
            (piece[0], piece[1], None)
            for interval in combined
            for piece in split_circular_interval(interval[0], interval[1], read_length)
        ]
        legacy_nuc_rows.sort(key=lambda t: (int(t[0]), int(t[1])))
        ns = [int(s) for s, _, _ in legacy_nuc_rows]
        nl = [int(length) for _, length, _ in legacy_nuc_rows]

        legacy_msps = [
            piece
            for interval in msps
            for piece in split_circular_interval(interval[0], interval[1], read_length)
        ]
        legacy_msps.sort(key=lambda t: (int(t[0]), int(t[1])))
        a_s = [int(s) for s, _ in legacy_msps]
        a_l = [int(length) for _, length in legacy_msps]

        if ns:
            read.set_tag('ns', pyarray.array('I', ns))
            read.set_tag('nl', pyarray.array('I', nl))
        else:
            for tag in ('ns', 'nl', 'nq'):
                if read.has_tag(tag):
                    try:
                        read.set_tag(tag, None)
                    except Exception:
                        pass
        if a_s:
            read.set_tag('as', pyarray.array('I', a_s))
            read.set_tag('al', pyarray.array('I', a_l))
        else:
            for tag in ('as', 'al', 'aq'):
                if read.has_tag(tag):
                    try:
                        read.set_tag(tag, None)
                    except Exception:
                        pass
        # nq must have len == len(ns) per fibertools invariant. If we wrote
        # new ns/nl without fresh scores, drop any stale nq from the input BAM
        # to avoid len(nq) != len(ns) failing ft validate (see fibertools-rs
        # bamannotations.rs set_qual assert).
        if ns and nq_for_kept_nucs is not None:
            legacy_nq_rows = []
            for interval, q in zip(kept_nucs, nq_values):
                for piece in split_circular_interval(interval[0], interval[1], read_length):
                    legacy_nq_rows.append((piece[0], piece[1], max(0, min(255, int(q)))))
            legacy_nq_rows.sort(key=lambda t: (int(t[0]), int(t[1])))
            legacy_nq = [q for _, _, q in legacy_nq_rows]
            if len(legacy_nq) != len(ns):
                legacy_nq = [0] * len(ns)
            read.set_tag('nq', pyarray.array('B',
                          legacy_nq))
        elif ns and read.has_tag('nq'):
            try:
                read.set_tag('nq', None)
            except Exception:
                pass
        # Same for aq: stale per-msp qualities from input would mismatch
        # the refreshed as/al length.
        if a_s and read.has_tag('aq'):
            try:
                read.set_tag('aq', None)
            except Exception:
                pass
