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

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from fiberhmm.core.bam_reader import (
    detect_daf_strand,
    encode_from_query_sequence,
    extract_daf_iupac_positions,
    has_iupac_encoding,
    parse_mm_tag_query_positions,
)
from fiberhmm.io.ma_tags import (
    ambiguity_to_edge,
    format_aq_array,
    format_ma_tag,
    llr_to_tq,
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
    for s, l in zip(as_, al):
        s = int(s); l = int(l)
        if l > 0:
            iv.append((s, s + l))
    for s, l in zip(ns, nl):
        s = int(s); l = int(l)
        if 0 < l < unify_threshold:
            iv.append((s, s + l))
    iv = [(max(0, a), min(read_len, b)) for a, b in iv]
    iv = [(a, b) for a, b in iv if b > a]
    return merge_intervals(iv)


def _is_target_code(code: int) -> bool:
    return (0 <= code < N_CTX) or (UNMETH_OFFSET <= code < UNMETH_OFFSET + N_CTX)


def _is_hit_code(code: int) -> bool:
    return 0 <= code < N_CTX


def _is_miss_code(code: int) -> bool:
    return UNMETH_OFFSET <= code < UNMETH_OFFSET + N_CTX


def _scan_left_ambiguity(obs: np.ndarray, lo: int, conservative_left: int) -> int:
    """Bp from the conservative left boundary to the previous hit before it.

    Walks left from ``conservative_left - 1`` until a hit is found; ambiguity
    is the count of intervening positions. If no hit is found within the
    interval (or the read), ambiguity is unbounded and we return a large
    sentinel (>= EDGE_AMBIGUITY_SAT, so el = 0).
    """
    i = conservative_left - 1
    amb = 0
    while i >= lo:
        if _is_hit_code(int(obs[i])):
            return amb
        amb += 1
        i -= 1
    return amb  # ran off the start of the scan interval -> ambiguous


def _scan_right_ambiguity(obs: np.ndarray, hi: int, conservative_right: int) -> int:
    """Bp from the conservative right boundary to the terminating hit.

    The conservative right boundary is exclusive (= last_miss_position + 1).
    We walk right starting at ``conservative_right``; ambiguity is the count
    of non-target positions before the first hit.
    """
    i = conservative_right
    amb = 0
    while i < hi:
        if _is_hit_code(int(obs[i])):
            return amb
        amb += 1
        i += 1
    return amb  # ran off the end of the scan interval -> ambiguous


def call_tfs_in_interval(obs: np.ndarray, lo: int, hi: int,
                         llr_hit: np.ndarray, llr_miss: np.ndarray,
                         min_llr: float, min_opps: int) -> List[TFCall]:
    """Kadane local-maximum scan on obs[lo:hi]."""
    calls: List[TFCall] = []
    cur_start: Optional[int] = None
    running = 0.0
    opps = 0
    peak_llr = 0.0
    peak_end = lo  # exclusive; advances only on misses
    peak_opps = 0

    def _flush():
        # peak_end is exclusive and points to one past the last miss
        # that contributed positively to the peak.
        if cur_start is None:
            return
        if peak_llr < min_llr or peak_opps < min_opps:
            return
        l_amb = _scan_left_ambiguity(obs, lo, cur_start)
        r_amb = _scan_right_ambiguity(obs, hi, peak_end)
        calls.append(TFCall(
            start=cur_start,
            length=peak_end - cur_start,
            llr=peak_llr,
            n_opps=peak_opps,
            left_ambiguity=l_amb,
            right_ambiguity=r_amb,
        ))

    for i in range(lo, hi):
        code = int(obs[i])
        if _is_hit_code(code):
            step = llr_hit[code]
            is_opp = True
        elif _is_miss_code(code):
            step = llr_miss[code - UNMETH_OFFSET]
            is_opp = True
        else:
            step = 0.0
            is_opp = False

        if cur_start is None:
            if step > 0:
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

        if running <= 0:
            _flush()
            cur_start = None
            running = 0.0
            opps = 0
            peak_llr = 0.0
            peak_end = i + 1
            peak_opps = 0

    if cur_start is not None:
        _flush()
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
        ml_tag = list(read.get_tag('ML')) if read.has_tag('ML') else list(read.get_tag('Ml'))
    except KeyError:
        ml_tag = []
    if not mm_tag or not ml_tag:
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
                unify_threshold: int) -> Tuple[List[TFCall], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Process one read.

    Returns:
        (tf_calls, kept_nuc_intervals, msp_intervals)
        - tf_calls: emitted TF calls
        - kept_nuc_intervals: v2 nucs that survive --unify
                              (>= unify_threshold OR no overlapping TF call)
        - msp_intervals: v2 MSPs unchanged
    """
    try:
        ns_raw = list(read.get_tag('ns'))
        nl_raw = list(read.get_tag('nl'))
    except KeyError:
        ns_raw, nl_raw = [], []
    try:
        as_raw = list(read.get_tag('as'))
        al_raw = list(read.get_tag('al'))
    except KeyError:
        as_raw, al_raw = [], []

    if not (ns_raw or as_raw):
        return [], [], []

    extracted = extract_modifications(read, mode, context_size)
    if extracted is None:
        # Pass through v2 calls unchanged
        nucs = [(int(s), int(l)) for s, l in zip(ns_raw, nl_raw) if int(l) > 0]
        msps = [(int(s), int(l)) for s, l in zip(as_raw, al_raw) if int(l) > 0]
        return [], nucs, msps

    mod_pos, strand, seq = extracted
    obs = encode_from_query_sequence(
        seq, mod_pos, edge_trim=10, mode=mode, strand=strand,
        context_size=context_size,
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
    msps = [(int(s), int(l)) for s, l in zip(as_raw, al_raw) if int(l) > 0]
    kept_nucs: List[Tuple[int, int]] = []
    tf_intervals = [(c.start, c.start + c.length) for c in tf_calls]
    for s, l in zip(ns_raw, nl_raw):
        s = int(s); l = int(l)
        if l <= 0:
            continue
        if l >= unify_threshold:
            kept_nucs.append((s, l))
            continue
        # Short v2 nuc -- drop if overlapped by any TF call
        nuc_end = s + l
        if any(ts < nuc_end and te > s for ts, te in tf_intervals):
            continue
        kept_nucs.append((s, l))

    return tf_calls, kept_nucs, msps


def write_ma_tags(read, read_length: int,
                  tf_calls: Sequence[TFCall],
                  kept_nucs: Sequence[Tuple[int, int]],
                  msps: Sequence[Tuple[int, int]],
                  nq_for_kept_nucs: Optional[Sequence[int]] = None,
                  also_write_legacy: bool = True) -> None:
    """Set MA + AQ tags on the read in place.

    If ``also_write_legacy=True``, also rewrite the legacy ns/nl/as/al
    tags so that older consumers see the unified call set.
    """
    import array as pyarray

    # Default nq for kept nucs to 0 (sentinel for "unverified") if not provided.
    nq_values = list(nq_for_kept_nucs) if nq_for_kept_nucs is not None \
        else [0] * len(kept_nucs)
    if len(nq_values) != len(kept_nucs):
        raise ValueError("nq_values length must match kept_nucs length")

    tf_intervals = [(c.start, c.length) for c in tf_calls]
    tq_vals = [llr_to_tq(c.llr) for c in tf_calls]
    el_vals = [ambiguity_to_edge(c.left_ambiguity) for c in tf_calls]
    er_vals = [ambiguity_to_edge(c.right_ambiguity) for c in tf_calls]

    ma = format_ma_tag(
        read_length=read_length,
        nuc_intervals=kept_nucs,
        msp_intervals=msps,
        tf_intervals=tf_intervals,
    )
    aq = format_aq_array(
        nq_values=nq_values,
        tf_q_values=tq_vals,
        tf_lq_values=el_vals,
        tf_rq_values=er_vals,
    )
    read.set_tag('MA', ma, value_type='Z')
    read.set_tag('AQ', aq)

    if also_write_legacy:
        ns = [s for s, l in kept_nucs]
        nl = [l for s, l in kept_nucs]
        a_s = [s for s, l in msps]
        a_l = [l for s, l in msps]
        if ns:
            read.set_tag('ns', pyarray.array('I', ns))
            read.set_tag('nl', pyarray.array('I', nl))
        else:
            for tag in ('ns', 'nl', 'nq'):
                if read.has_tag(tag):
                    try: read.set_tag(tag, None)
                    except Exception: pass
        if a_s:
            read.set_tag('as', pyarray.array('I', a_s))
            read.set_tag('al', pyarray.array('I', a_l))
        else:
            for tag in ('as', 'al', 'aq'):
                if read.has_tag(tag):
                    try: read.set_tag(tag, None)
                    except Exception: pass
        if nq_for_kept_nucs is not None and ns:
            read.set_tag('nq', pyarray.array('B',
                          [max(0, min(255, int(v))) for v in nq_values]))
