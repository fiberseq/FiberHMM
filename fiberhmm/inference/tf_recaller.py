"""LLR-based TF footprint recaller -- second pass on FiberHMM-tagged BAMs.

The first pass (``fiberhmm-apply``) calls nucleosomes and MSPs with a trained
2-state HMM. This second pass scans MSPs and short sub-nucleosomal calls for
sequence-context-aware TF footprints. The result is ``MA``/``AQ``
spec-compliant output with a ``tf.QQQ`` annotation type.

The DddA TF table is calibrated from physical scDAF duplexes: one strand
nominates a strict state and its untouched mate estimates accessible- and
protected-state event probabilities. It is deliberately independent of both
the first-pass DddA HMM and the frozen likelihood table used by the radial
nucleosome refiner. DddA therefore does *not* reuse or transform the DddB
model.

Algorithm summary
-----------------

For each read:

1. Build per-position observation array using the same encoder as
   ``fiberhmm-apply`` (``encode_from_query_sequence``).
2. From v2 tags, derive the scan space:
     - all MSPs (``as``/``al``)
     - all short v2 nucs (``ns``/``nl`` with ``nl < unify_threshold``)
3. Inside each scan interval, find the best non-overlapping configuration
   of protected intervals using per-context LLR steps:
     - miss step: ``log P(miss | ctx, protected) - log P(miss | ctx, accessible)``
     - hit  step: ``log P(hit  | ctx, protected) - log P(hit  | ctx, accessible)``
   Maximize ``sum(interval LLR) - min_llr * number_of_intervals``, with
   at least ``min_opps`` informative target positions per interval. This is
   an exact interval dynamic program, not one maximum per positive-score
   excursion. A modified gap can therefore separate two footprints even
   when it does not exhaust the evidence accumulated by the first one.
4. For each emitted call, compute edge ambiguity = bp distance from the
   conservative boundary (last informative miss + 1 on the right; first
   informative miss on the left) to the bracketing hit.
5. Emit MA + AQ tags. By default (``unify=True``), v2's ``ns``/``nl``
   short nucs that overlap a recaller call are *dropped* from the
   ``nuc+`` annotation (they live solely in ``tf+`` now).

Per-enzyme defaults are baked into ``ENZYME_PRESETS``. All supported
chemistries use the conservative shared setting ``min_llr=5.0``. In the
native-context synthetic benchmark this is the first integer threshold at
which center-covering TF-recaller calls are below 2% in accessible controls
for every chemistry. For DddA, the same operating point remains on
the high-balanced-accuracy plateau in an independent held-out physical-mate
calibration across 12 libraries.

The emission tables and reported per-call LLRs are unchanged. The configuration
penalty is an explicit regularizer, not an FDR threshold or calibrated posterior.
Its numerical default retains the native minimum-score setting, but changing the
search/selection procedure requires a fresh detector-specific calibration. The
single-excursion implementation remains available for comparison and is kept by
nucleosome refinement; this TF change does not change its scanning algorithm.
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
    parse_mm_tag_query_calls,
)
from fiberhmm.io.ma_tags import (
    DDDA_MCG_HEMI_FEATURE,
    DDDA_MCG_FEATURE,
    DDDA_UCG_FEATURE,
    ambiguity_to_edge,
    flip_interval_frame,
    flip_intervals_to_seq,
    format_an_tag,
    format_aq_array,
    format_ma_tag,
    llr_to_tq,
    parse_an_tag,
    split_circular_interval,
)

# Observation-code constants (must match encode_from_query_sequence with k=3)
N_CTX = 4096           # 4^(2*3) hexamer contexts
NON_TARGET = N_CTX     # code 4096
UNMETH_OFFSET = 4097   # miss codes live at [4097, 4097 + 4096)
TF_DECODER_VERSION = "multi_interval_v1"
# The recall kernels index tables by the k=3 code layout above: hit contexts,
# one non-target code, miss contexts and (in model files) a trailing
# non-target code. The trailing column is never read.
RECALL_TABLE_COLUMNS = UNMETH_OFFSET + N_CTX + 1  # 8194
_RECALL_TABLE_WIDTHS = (UNMETH_OFFSET + N_CTX, RECALL_TABLE_COLUMNS)
CPG_MASK_POLICIES = ("unmethylated-only", "methylated-only")


def resolve_cpg_masking(use_m5c: Optional[bool], enzyme: Optional[str],
                        mode: Optional[str]) -> bool:
    """The DddA CpG-aware recall policy shared by every recalling command.

    ``fiberhmm-call``, ``fiberhmm-recall-tfs``/``-recall-nucs`` and the joint
    both-strand recall of ``fiberhmm-pair``/``-merge`` all resolve it here:
    unset (``None``) means on for DddA and off for every other enzyme; an
    explicit request is honoured, but enabling it outside DAF mode or for a
    non-DddA preset is refused (the CpG treatment is calibrated for DddA only).
    Raises :class:`ValueError` on such a request.
    """
    enabled = (enzyme == 'ddda') if use_m5c is None else bool(use_m5c)
    if enabled and (mode != 'daf' or enzyme not in (None, 'ddda')):
        raise ValueError(
            '--use-m5c is calibrated only for DddA or a custom DAF/DddA model'
        )
    return enabled


def cpg_mask_from_intervals(read_len: int, policy: str = "unmethylated-only",
                            ucg_intervals=(), mcg_intervals=()) -> np.ndarray:
    """CpG mask from SEQ-frame ``ddda_ucg``/``ddda_mcg`` intervals.

    ``True`` marks positions whose CpG observations are excluded from recall.
    ``unmethylated-only`` (the production policy) excludes every CpG except
    those inside confidently unmethylated whole-island calls (``ddda_ucg``);
    ``methylated-only`` reproduces the former policy that excluded CpGs only
    inside ``ddda_mcg`` intervals.
    """
    if policy not in CPG_MASK_POLICIES:
        raise ValueError(
            f"unknown CpG mask policy {policy!r}; expected one of "
            f"{', '.join(CPG_MASK_POLICIES)}"
        )
    read_len = int(read_len)
    if policy == "unmethylated-only":
        mask = np.ones(read_len, dtype=bool)
        for start, end in ucg_intervals or ():
            mask[max(0, int(start)):min(read_len, int(end))] = False
        return mask
    mask = np.zeros(read_len, dtype=bool)
    for start, end in mcg_intervals or ():
        mask[max(0, int(start)):min(read_len, int(end))] = True
    return mask


def read_cpg_intervals(read) -> dict:
    """SEQ-frame ``ddda_ucg``/``ddda_mcg`` intervals carried by ``read``'s MA.

    Returns ``{'ucg': [...], 'mcg': [...]}`` (empty lists without MA). Small
    and picklable, so pipelines can compute it in the main process and ship
    it to workers that only see a slim payload.
    """
    if not read.has_tag('MA'):
        return {'ucg': [], 'mcg': []}
    from fiberhmm.daf.m5c import ma_intervals

    return {
        'ucg': [tuple(map(int, iv)) for iv in ma_intervals(read, DDDA_UCG_FEATURE)],
        'mcg': [tuple(map(int, iv)) for iv in ma_intervals(read, DDDA_MCG_FEATURE)],
    }


def build_cpg_mask(read, read_len: int,
                   policy: str = "unmethylated-only") -> np.ndarray:
    """Return positions at which CpG observations are excluded from recall.

    The production policy is conservative: only CpGs inside confidently
    unmethylated whole-island calls (``ddda_ucg``) remain available.  The
    former policy, which masked only ``ddda_mcg`` intervals, is retained for
    explicit compatibility and comparison runs.
    """
    if policy not in CPG_MASK_POLICIES:
        raise ValueError(
            f"unknown CpG mask policy {policy!r}; expected one of "
            f"{', '.join(CPG_MASK_POLICIES)}"
        )
    intervals = read_cpg_intervals(read)
    return cpg_mask_from_intervals(
        read_len, policy, intervals['ucg'], intervals['mcg'],
    )


# Per-enzyme defaults: (min_llr, emission_uplift). All bundled models are used
# directly; ``emission_uplift`` remains available only as an explicit custom
# sensitivity override.
ENZYME_PRESETS = {
    'hia5':   dict(min_llr=5.0, emission_uplift=1.0),
    'dddb':   dict(min_llr=5.0, emission_uplift=1.0),
    # Confirmed on untouched physical mates from 23,388 scDAF duplexes across
    # 12 libraries. The interval penalty is not an FDR cutoff.
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


def require_recall_table(emissionprob) -> np.ndarray:
    """The two-state emission table as a float array, or ``ValueError``.

    TF and nucleosome recall read observation codes in the k=3 layout
    (``RECALL_TABLE_COLUMNS`` = 8194 columns); a table built for another
    context size would be sliced at the wrong offsets and silently give wrong
    calls, so it is refused. (HMM calling itself works for any k.)
    """
    EP = np.asarray(emissionprob, dtype=np.float64)
    if EP.ndim != 2 or EP.shape[0] != 2:
        raise ValueError(f"Expected a two-state emission table, got shape {EP.shape}")
    if EP.shape[1] not in _RECALL_TABLE_WIDTHS:
        width = EP.shape[1]
        k = None
        for candidate in range(1, 8):
            if width == 2 * 4 ** (2 * candidate) + 2:
                k = candidate
        size = f"context size k={k}" if k else "an unknown context size"
        raise ValueError(
            f"TF/nucleosome recall needs a k=3 emission table "
            f"({RECALL_TABLE_COLUMNS} columns); this model has {width} columns "
            f"({size}); TF recall in fiberhmm-call and the recall tools needs a k=3 model")
    return EP


def require_recall_models(*paths) -> None:
    """Check, in the calling process, that every model file recall will read
    has a k=3 table (``require_recall_table``); ``None`` entries are skipped.

    Worker pools build their tables in their initializers, where an error
    would surface late or not at all; pipelines call this before starting one.
    """
    import os

    from fiberhmm.core.model_io import load_model_with_metadata

    for path in dict.fromkeys(p for p in paths if p):
        if not os.path.isfile(path):
            continue  # a missing file is reported by whatever loads it
        model, _context, _mode = load_model_with_metadata(path)
        try:
            require_recall_table(model.emissionprob_)
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from None


def build_llr_tables(model) -> Tuple[np.ndarray, np.ndarray]:
    """Return (llr_hit, llr_miss) lookup arrays, length N_CTX each.

    Assumes model.normalize_states() has been applied (state 0 = protected,
    state 1 = accessible). load_model_with_metadata enforces this. Only
    ``emissionprob_`` enters these likelihood-ratio tables: ``startprob_`` and
    ``transmat_`` are required by the shared model container and participate in
    state-order normalization, but they do not contribute to TF recall scores
    or impose a duration/transition prior on the local scan.
    """
    EP = require_recall_table(model.emissionprob_)
    eps = 1e-30
    hit_prot = np.clip(EP[0, :N_CTX], eps, 1.0)
    hit_acc = np.clip(EP[1, :N_CTX], eps, 1.0)
    miss_prot = np.clip(EP[0, UNMETH_OFFSET:UNMETH_OFFSET + N_CTX], eps, 1.0)
    miss_acc = np.clip(EP[1, UNMETH_OFFSET:UNMETH_OFFSET + N_CTX], eps, 1.0)
    return (np.log(hit_prot) - np.log(hit_acc),
            np.log(miss_prot) - np.log(miss_acc))


def build_conditional_hit_tables(
    model,
    emission_uplift: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return P(hit | context, protected/accessibile) for generative recall.

    Unlike an LLR table, these conditional probabilities can be mixed into a
    partially exposed rotational state without linearly interpolating log odds.
    """
    EP = require_recall_table(model.emissionprob_)
    eps = 1e-12
    hit = np.clip(EP[:, :N_CTX], 0.0, None)
    miss = np.clip(EP[:, UNMETH_OFFSET:UNMETH_OFFSET + N_CTX], 0.0, None)
    conditional = hit / np.maximum(hit + miss, eps)
    protected = np.clip(conditional[0], eps, 1.0 - eps)
    accessible = np.clip(conditional[1], eps, 1.0 - eps)
    if emission_uplift <= 0.0:
        raise ValueError("emission_uplift must be positive")
    if abs(float(emission_uplift) - 1.0) > 1e-9:
        accessible = 1.0 - np.power(1.0 - accessible, emission_uplift)
        protected = np.power(protected, emission_uplift)
    return (
        np.clip(protected, eps, 1.0 - eps),
        np.clip(accessible, eps, 1.0 - eps),
    )


def build_m5c_llr_tables(model, rate_ratio: Optional[float] = None,
                         emission_uplift: float = 1.0
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """LLRs inside a confidently methylated DddA CpG island.

    CpG observations are confounded by endogenous 5mC and are therefore
    neutral.  The decoding kernels also remove these observations from the
    opportunity lattice so they cannot satisfy ``min_opps`` or define a
    boundary.  Non-CpG entries are identical to the ordinary recall tables.

    ``rate_ratio`` is retained for API compatibility with the older
    rate-adjustment implementation; it is validated but no longer changes the
    conservative, zero-information CpG treatment.
    """
    if rate_ratio is not None and not 0.0 < rate_ratio < 1.0:
        raise ValueError("rate_ratio must be between zero and one")
    EP = require_recall_table(model.emissionprob_)
    hit_prot = EP[0, :N_CTX]
    hit_acc = EP[1, :N_CTX]
    miss_prot = EP[0, UNMETH_OFFSET:UNMETH_OFFSET + N_CTX]
    miss_acc = EP[1, UNMETH_OFFSET:UNMETH_OFFSET + N_CTX]
    eps = 1e-30
    context_prot = np.clip(hit_prot + miss_prot, eps, None)
    context_acc = np.clip(hit_acc + miss_acc, eps, None)
    p_prot = hit_prot / context_prot
    p_acc = hit_acc / context_acc
    if emission_uplift <= 0:
        raise ValueError("emission_uplift must be positive")
    if abs(emission_uplift - 1.0) > 1e-9:
        p_acc = 1.0 - np.power(1.0 - p_acc, emission_uplift)
        p_prot = np.power(p_prot, emission_uplift)
    # Context codes are base-4 with A,C,T,G = 0,1,2,3. The first base of
    # the right flank is the 3' neighbor; CpG therefore has digit G (=3).
    codes = np.arange(N_CTX)
    is_cpg = ((codes % 64) // 16) == 3
    adjusted = np.clip(p_acc, eps, 1.0 - eps)
    p_prot = np.clip(p_prot, eps, 1.0 - eps)
    context_llr = np.log(context_prot) - np.log(context_acc)
    hit = context_llr + np.log(p_prot) - np.log(adjusted)
    miss = context_llr + np.log1p(-p_prot) - np.log1p(-adjusted)
    calibrated = getattr(model, 'cpg_methylated_probabilities_', None)
    if calibrated is None:
        hit[is_cpg] = 0.0
        miss[is_cpg] = 0.0
    else:
        pa = np.asarray(calibrated['accessible'], dtype=np.float64)
        pp = np.asarray(calibrated['protected'], dtype=np.float64)
        if pa.shape != (N_CTX,) or pp.shape != (N_CTX,) or not (
                np.all(np.isfinite(pa)) and np.all(np.isfinite(pp)) and
                np.all((pa > 0) & (pa < 1)) and np.all((pp > 0) & (pp < 1))):
            raise ValueError('Calibrated methylated CpG probabilities must be finite length-4096 arrays strictly between 0 and 1')
        if emission_uplift != 1.0:
            raise ValueError('Calibrated CpG probabilities require emission_uplift=1')
        hit[is_cpg] = np.log(pp[is_cpg] / pa[is_cpg])
        miss[is_cpg] = np.log1p(-pp[is_cpg]) - np.log1p(-pa[is_cpg])
    return hit, miss


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
    EP = require_recall_table(model.emissionprob_)
    eps = 1e-30
    hit_prot = np.clip(EP[0, :N_CTX], eps, 1.0)
    hit_acc = np.clip(EP[1, :N_CTX], eps, 1.0)
    miss_prot = np.clip(EP[0, UNMETH_OFFSET:UNMETH_OFFSET + N_CTX], eps, 1.0)
    miss_acc = np.clip(EP[1, UNMETH_OFFSET:UNMETH_OFFSET + N_CTX], eps, 1.0)
    context_prot = hit_prot + miss_prot
    context_acc = hit_acc + miss_acc
    p_hit_acc = hit_acc / context_acc
    p_hit_prot = hit_prot / context_prot
    p_hit_acc_new = 1.0 - np.power(np.clip(1.0 - p_hit_acc, eps, 1.0), uplift)
    p_hit_prot_new = np.power(np.clip(p_hit_prot, eps, 1.0), uplift)
    p_hit_acc_new = np.clip(p_hit_acc_new, eps, 1.0 - eps)
    p_hit_prot_new = np.clip(p_hit_prot_new, eps, 1.0 - eps)
    context_llr = np.log(context_prot) - np.log(context_acc)
    new_llr_miss = (context_llr + np.log(1.0 - p_hit_prot_new) -
                    np.log(1.0 - p_hit_acc_new))
    new_llr_hit = context_llr + np.log(p_hit_prot_new) - np.log(p_hit_acc_new)
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
                    min_llr, min_opps, use_m5c, m5c_mask,
                    m5c_llr_hit, m5c_llr_miss):
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
            is_masked_cpg = use_m5c and m5c_mask[i] and ((code % 64) // 16 == 3)
            if not is_masked_cpg or m5c_llr_hit[code] != 0.0 or m5c_llr_miss[code] != 0.0:
                step = m5c_llr_hit[code] if is_masked_cpg else llr_hit[code]
                is_opp = True
        elif UNMETH_OFFSET <= code < UNMETH_OFFSET + N_CTX:
            context = code - UNMETH_OFFSET
            is_masked_cpg = use_m5c and m5c_mask[i] and ((context % 64) // 16 == 3)
            if not is_masked_cpg or m5c_llr_hit[context] != 0.0 or m5c_llr_miss[context] != 0.0:
                step = m5c_llr_miss[context] if is_masked_cpg else llr_miss[context]
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
            masked_cpg = (
                use_m5c and m5c_mask[j] and 0 <= c < N_CTX
                and ((c % 64) // 16 == 3)
                and m5c_llr_hit[c] == 0.0 and m5c_llr_miss[c] == 0.0
            )
            if 0 <= c < N_CTX and not masked_cpg:
                break
            amb += 1
            j -= 1
        left_amb[ci] = amb
        # Right
        amb = 0
        j = e
        while j < hi:
            c = obs[j]
            masked_cpg = (
                use_m5c and m5c_mask[j] and 0 <= c < N_CTX
                and ((c % 64) // 16 == 3)
                and m5c_llr_hit[c] == 0.0 and m5c_llr_miss[c] == 0.0
            )
            if 0 <= c < N_CTX and not masked_cpg:
                break
            amb += 1
            j += 1
        right_amb[ci] = amb

    return (starts[:n_calls], ends[:n_calls], llrs[:n_calls],
            opps_out[:n_calls], left_amb[:n_calls], right_amb[:n_calls])


@_numba_jit(nopython=True, cache=True)
def _call_tf_configurations_numba(obs, lo, hi, llr_hit, llr_miss,
                                   interval_penalty, min_opps, use_m5c, m5c_mask,
                                   m5c_llr_hit, m5c_llr_miss, adjacent_run_mode=0):
    """Exact penalized multi-interval decoding on the native opportunity lattice.

    For disjoint intervals C, maximize sum(LR(I) - interval_penalty for I in C).
    Every interval contains >= min_opps target observations and begins/ends on
    a positive LLR step. Neutral bases do not create opportunities or evidence.
    The all-accessible configuration has score zero. Ties prefer fewer intervals
    and then less protected span; an interval exactly at the penalty need not be
    selected over the empty configuration. Reported call scores exclude the
    penalty and remain sums of the unchanged native emission LLRs.

    With prefix LLR S and optimal prefix objective F, a call ending at t has
    objective S[t] - penalty + max_s(F[s] - S[s]), where s <= t-min_opps.
    Admitting starts incrementally makes time/memory linear in the scan domain.
    No fixed cap on the number of footprints, family proposals, or unique hits
    is imposed. A constant per-interval cost is an unnormalized configuration
    prior; this MAP-style decoder does not produce posterior probabilities.
    """
    capacity = hi - lo
    positions = np.empty(capacity, dtype=np.int64)
    steps = np.empty(capacity, dtype=np.float64)
    n = 0
    for i in range(lo, hi):
        code = obs[i]
        if 0 <= code < 4096:
            calibrated = use_m5c and m5c_mask[i] and ((code % 64) // 16 == 3)
            if calibrated and m5c_llr_hit[code] == 0.0 and m5c_llr_miss[code] == 0.0:
                continue
            positions[n] = i
            steps[n] = m5c_llr_hit[code] if calibrated else llr_hit[code]
            n += 1
        elif 4097 <= code < 8193:
            context = code - 4097
            calibrated = use_m5c and m5c_mask[i] and ((context % 64) // 16 == 3)
            if calibrated and m5c_llr_hit[context] == 0.0 and m5c_llr_miss[context] == 0.0:
                continue
            positions[n] = i
            steps[n] = m5c_llr_miss[context] if calibrated else llr_miss[context]
            n += 1

    # Modes:0=native,1=mean evidence distributed across sites,2=one run unit.
    end_positions = positions[:n].copy() + 1
    if adjacent_run_mode:
        source_n = n
        out_n = 0
        i = 0
        while i < source_n:
            j = i + 1
            total = steps[i]
            while j < source_n and positions[j] == positions[j-1] + 1:
                total += steps[j]
                j += 1
            size = j - i
            if adjacent_run_mode == 2:
                end_positions[out_n] = positions[j-1] + 1
                positions[out_n] = positions[i]
                steps[out_n] = total / size
                out_n += 1
            else:
                for z in range(i,j):
                    steps[z] /= size
            i = j
        if adjacent_run_mode == 2:
            n = out_n

    prefix = np.zeros(n + 1, dtype=np.float64)
    objective = np.zeros(n + 1, dtype=np.float64)
    counts = np.zeros(n + 1, dtype=np.int64)
    protected_bp = np.zeros(n + 1, dtype=np.int64)
    chosen_start = np.full(n + 1, -1, dtype=np.int64)
    best_start = -1
    best_start_score = -np.inf
    best_start_count = 0
    best_start_span_key = 0
    eps = 1e-10
    for t in range(1, n + 1):
        prefix[t] = prefix[t - 1] + steps[t - 1]
        objective[t] = objective[t - 1]
        counts[t] = counts[t - 1]
        protected_bp[t] = protected_bp[t - 1]
        s = t - min_opps
        if s >= 0 and steps[s] > 0.0:
            start_score = objective[s] - prefix[s]
            span_key = protected_bp[s] - positions[s]
            better = start_score > best_start_score + eps
            if abs(start_score - best_start_score) <= eps:
                better = (counts[s] < best_start_count or
                          (counts[s] == best_start_count and span_key < best_start_span_key))
            if best_start < 0 or better:
                best_start = s
                best_start_score = start_score
                best_start_count = counts[s]
                best_start_span_key = span_key
        if best_start < 0 or steps[t - 1] <= 0.0:
            continue
        candidate = prefix[t] - interval_penalty + best_start_score
        candidate_count = best_start_count + 1
        candidate_bp = best_start_span_key + end_positions[t - 1]
        better = candidate > objective[t] + eps
        if abs(candidate - objective[t]) <= eps:
            better = (candidate_count < counts[t] or
                      (candidate_count == counts[t] and candidate_bp < protected_bp[t]))
        if better:
            objective[t] = candidate
            counts[t] = candidate_count
            protected_bp[t] = candidate_bp
            chosen_start[t] = best_start

    n_calls = counts[n]
    starts = np.empty(n_calls, dtype=np.int64)
    ends = np.empty(n_calls, dtype=np.int64)
    llrs = np.empty(n_calls, dtype=np.float64)
    opps_out = np.empty(n_calls, dtype=np.int64)
    left_amb = np.empty(n_calls, dtype=np.int64)
    right_amb = np.empty(n_calls, dtype=np.int64)
    t, ci = n, n_calls - 1
    while t > 0:
        s = chosen_start[t]
        if s < 0:
            t -= 1
            continue
        starts[ci] = positions[s]
        ends[ci] = end_positions[t - 1]
        llrs[ci] = prefix[t] - prefix[s]
        opps_out[ci] = t - s
        ci -= 1
        t = s

    for ci in range(n_calls):
        s, e = starts[ci], ends[ci]
        amb, j = 0, s - 1
        while j >= lo:
            code = obs[j]
            masked_cpg = (
                use_m5c and m5c_mask[j] and 0 <= code < 4096
                and ((code % 64) // 16 == 3)
                and m5c_llr_hit[code] == 0.0 and m5c_llr_miss[code] == 0.0
            )
            if 0 <= code < 4096 and not masked_cpg:
                break
            amb += 1
            j -= 1
        left_amb[ci] = amb
        amb, j = 0, e
        while j < hi:
            code = obs[j]
            masked_cpg = (
                use_m5c and m5c_mask[j] and 0 <= code < 4096
                and ((code % 64) // 16 == 3)
                and m5c_llr_hit[code] == 0.0 and m5c_llr_miss[code] == 0.0
            )
            if 0 <= code < 4096 and not masked_cpg:
                break
            amb += 1
            j += 1
        right_amb[ci] = amb
    return starts, ends, llrs, opps_out, left_amb, right_amb


def call_tfs_in_interval(obs: np.ndarray, lo: int, hi: int,
                         llr_hit: np.ndarray, llr_miss: np.ndarray,
                         min_llr: float, min_opps: int,
                         m5c_mask: Optional[np.ndarray] = None,
                         m5c_llr_hit: Optional[np.ndarray] = None,
                         m5c_llr_miss: Optional[np.ndarray] = None,
                         *, decoder: str = "multi_interval",
                         adjacent_run_mode: str = "none") -> List[TFCall]:
    """Decode TF intervals from unchanged native emission evidence.

    ``multi_interval`` selects a complete non-overlapping configuration with a
    cost of ``min_llr`` per footprint. ``single_excursion`` retains the previous
    one-peak-per-positive-excursion scan for audits and nucleosome refinement.
    Adjacent-run modes are opt-in for single-strand DddA observations: average
    divides each site contribution by run length; collapse additionally makes
    each contiguous observed run one opportunity and preserves its full span.
    Missing/masked observations break runs. Do not apply to pooled C+G lattices.
    Output LLRs exclude the interval penalty.
    """
    run_modes = {"none": 0, "average": 1, "collapse": 2}
    if adjacent_run_mode not in run_modes:
        raise ValueError("adjacent_run_mode must be none, average, or collapse")
    if adjacent_run_mode != "none" and decoder != "multi_interval":
        raise ValueError("Adjacent-run scoring requires the multi-interval decoder")
    if decoder not in {"multi_interval", "single_excursion"}:
        raise ValueError(f"Unknown TF decoder: {decoder!r}")
    if decoder == "multi_interval" and (min_opps < 1 or min_llr < 0 or not np.isfinite(min_llr)):
        raise ValueError("Multi-interval TF decoding requires min_opps >= 1 and finite min_llr >= 0")
    if hi <= lo:
        return []
    if lo < 0 or hi > len(obs):
        raise ValueError("TF scan interval must lie inside the observation array")
    # Ensure dtypes numba can bind to cleanly
    obs_arr = np.ascontiguousarray(obs, dtype=np.int32)
    hit_arr = np.ascontiguousarray(llr_hit, dtype=np.float64)
    miss_arr = np.ascontiguousarray(llr_miss, dtype=np.float64)
    use_m5c = m5c_mask is not None
    if use_m5c and (m5c_llr_hit is None or m5c_llr_miss is None):
        raise ValueError(
            "m5c_mask requires both m5c_llr_hit and m5c_llr_miss tables"
        )
    mask_arr = (np.zeros(1, dtype=np.bool_) if m5c_mask is None else
                np.ascontiguousarray(m5c_mask, dtype=np.bool_))
    m5c_hit_arr = hit_arr if m5c_llr_hit is None else \
        np.ascontiguousarray(m5c_llr_hit, dtype=np.float64)
    m5c_miss_arr = miss_arr if m5c_llr_miss is None else \
        np.ascontiguousarray(m5c_llr_miss, dtype=np.float64)
    if use_m5c and len(mask_arr) != len(obs_arr):
        raise ValueError("m5c_mask length must match obs")
    if use_m5c and (m5c_hit_arr.shape != hit_arr.shape or
                    m5c_miss_arr.shape != miss_arr.shape):
        raise ValueError("m5c LLR tables must match the standard LLR table shapes")
    kernel = _call_tf_configurations_numba if decoder == "multi_interval" else _call_tfs_numba
    starts, ends, llrs, opps_arr, l_amb, r_amb = kernel(
        obs_arr, int(lo), int(hi), hit_arr, miss_arr,
        float(min_llr), int(min_opps), use_m5c, mask_arr,
        m5c_hit_arr, m5c_miss_arr,
        *([run_modes[adjacent_run_mode]] if decoder == "multi_interval" else []),
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


def call_single_excursion_intervals(obs: np.ndarray, lo: int, hi: int,
                                    llr_hit: np.ndarray, llr_miss: np.ndarray,
                                    min_llr: float, min_opps: int,
                                    m5c_mask: Optional[np.ndarray] = None,
                                    m5c_llr_hit: Optional[np.ndarray] = None,
                                    m5c_llr_miss: Optional[np.ndarray] = None) -> List[TFCall]:
    """Preserve the existing single-excursion kernel for non-TF consumers."""
    return call_tfs_in_interval(
        obs, lo, hi, llr_hit, llr_miss, min_llr, min_opps,
        m5c_mask=m5c_mask, m5c_llr_hit=m5c_llr_hit, m5c_llr_miss=m5c_llr_miss,
        decoder="single_excursion",
    )


#: ML threshold recall-tfs/recall-nucs have always used for MM/ML input.
#: Chemistry presets override it (Hia5 Nanopore: 248); see
#: :func:`fiberhmm.models.resolve_prob_threshold`.
RECALL_PROB_THRESHOLD = 125


def extract_modifications(read, mode: str, context_size: int = 3,
                          prob_threshold: int = RECALL_PROB_THRESHOLD,
                          ) -> Optional[Tuple[set, str, str]]:
    """Pull (mod_positions, strand, sequence) for a read.

    Returns None if the read can't be processed (no MM tag, no sequence).
    ``prob_threshold`` applies to MM/ML calls only; R/Y and MD deaminations
    are binary. Callers that encode observations should use
    :func:`extract_modification_calls`, which also returns the bases an MM
    ``?`` entry leaves without a call.
    """
    extracted = extract_modification_calls(
        read, mode, context_size, prob_threshold=prob_threshold)
    if extracted is None:
        return None
    return extracted[:3]


def extract_modification_calls(read, mode: str, context_size: int = 3,
                               prob_threshold: int = RECALL_PROB_THRESHOLD,
                               ) -> Optional[Tuple[set, str, str, set]]:
    """Pull (mod_positions, strand, sequence, unknown_positions) for a read.

    ``unknown_positions`` are SEQ-frame target bases an MM ``?`` entry left
    unlisted (no call, SAM spec); pass them to
    :func:`encode_from_query_sequence` as ``unknown_positions`` so they are
    non-target rather than misses. Empty for R/Y, MD and fully listed
    (``.``/unflagged) MM data, whose encoding is therefore unchanged.

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
        return mod_pos, strand, seq, set()
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
                    return set(ct_pos), '+', seq.upper(), set()
                return set(ga_pos), '-', seq.upper(), set()
        return None
    mod_pos, unknown_pos = parse_mm_tag_query_calls(
        mm_tag, ml_tag, seq, read.is_reverse,
        prob_threshold=prob_threshold, mode=mode,
    )
    if mode == 'daf':
        strand = detect_daf_strand(seq, mod_pos)
    else:
        strand = '.'
    return mod_pos, strand, seq, unknown_pos


def unapplied_call_daf_inputs(header) -> List[str]:
    """Describe DAF call inputs recorded in @PG that recall cannot re-apply.

    ``fiberhmm-call`` may exclude SNP-masked sites (``--daf-snp-mask`` or
    auto ``--daf-call-snps``) and derive deaminations against a reference
    FASTA (``--reference``).  recall-tfs/recall-nucs re-derive deaminations
    from the read alone, so masked sites come back as hits.  Returns one
    human-readable item per such input found in a ``fiberhmm-call`` @PG line.
    """
    try:
        header_dict = header.to_dict() if hasattr(header, 'to_dict') else dict(header)
    except Exception:
        return []
    found: List[str] = []
    for record in header_dict.get('PG', []) or []:
        program = str(record.get('PN') or record.get('ID') or '')
        if not program.startswith('fiberhmm-call'):
            continue
        description = str(record.get('DS', ''))
        command = str(record.get('CL', '')).split()
        for token in description.split():
            if token.startswith('daf_snp_mask=on'):
                found.append(f"SNP mask ({token})")
        for flag in ('--daf-snp-mask', '--reference'):
            for index, argument in enumerate(command):
                value = None
                if argument == flag and index + 1 < len(command):
                    value = command[index + 1]
                elif argument.startswith(flag + '='):
                    value = argument.split('=', 1)[1]
                if value is not None:
                    found.append(f"{flag} {value}")
    return list(dict.fromkeys(found))


def warn_unapplied_call_daf_inputs(header, mode: str, stream=None) -> List[str]:
    """Warn (stderr) when the input's call used DAF inputs recall ignores."""
    if mode != 'daf':
        return []
    found = unapplied_call_daf_inputs(header)
    if found:
        import sys
        print(
            "WARNING: the input was called by fiberhmm-call with "
            + "; ".join(found)
            + ". Recall re-derives deaminations from each read and does not "
            "re-apply the SNP mask or reference, so masked sites count as "
            "hits again. Re-run fiberhmm-call with the same options for "
            "SNP-masked calls.",
            file=stream if stream is not None else sys.stderr,
        )
    return found


def recall_read(read, llr_hit: np.ndarray, llr_miss: np.ndarray,
                mode: str, context_size: int,
                min_llr: float, min_opps: int,
                unify_threshold: int,
                input_molecular_frame: bool = True,
                m5c_llr_hit: Optional[np.ndarray] = None,
                m5c_llr_miss: Optional[np.ndarray] = None,
                cpg_mask_policy: str = "unmethylated-only",
                prob_threshold: int = RECALL_PROB_THRESHOLD,
                ) -> Tuple[List[TFCall], List[Tuple[int, int]], List[Tuple[int, int]]]:
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

    extracted = extract_modification_calls(read, mode, context_size,
                                           prob_threshold=prob_threshold)
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

    mod_pos, strand, seq, unknown_pos = extracted
    obs = encode_from_query_sequence(
        seq, mod_pos, edge_trim=10, mode=mode, strand=strand,
        context_size=context_size,
        is_reverse=bool(read.is_reverse),
        unknown_positions=unknown_pos,
    )
    read_len = len(seq)
    m5c_mask = None
    if m5c_llr_hit is not None and m5c_llr_miss is not None:
        m5c_mask = build_cpg_mask(read, read_len, cpg_mask_policy)

    intervals = build_scan_intervals(ns_raw, nl_raw, as_raw, al_raw,
                                      read_len, unify_threshold=unify_threshold)

    tf_calls: List[TFCall] = []
    for lo, hi in intervals:
        tf_calls.extend(call_tfs_in_interval(
            obs, lo, hi, llr_hit, llr_miss, min_llr, min_opps,
            m5c_mask=m5c_mask, m5c_llr_hit=m5c_llr_hit,
            m5c_llr_miss=m5c_llr_miss,
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


# MA features write_ma_tags regenerates from the recall; every other group is
# carried through unchanged.
_REGENERATED_MA_FEATURES = frozenset({'nuc', 'msp', 'tf'})


def _ma_group_head(group: str) -> Tuple[str, str]:
    """Return ``(feature, qual_spec)`` for one MA group string."""
    head = group.partition(':')[0]
    for index, character in enumerate(head):
        if character in '.+-':
            return head[:index], head[index + 1:]
    return head, ''


def _preserved_ma_groups(read) -> Tuple[List[str], List[str], List[int]]:
    """Collect the read's MA groups that ``write_ma_tags`` does not rewrite.

    Returns ``(groups, names, aq_bytes)``: the verbatim MA group strings in
    their original order, their AN names (``''`` when unnamed), and the AQ
    bytes that belong to quality-bearing preserved groups.  A preserved group
    whose AQ bytes cannot be located (AQ missing or inconsistent with the MA
    quality specs) is dropped with a warning rather than written with
    misaligned qualities.
    """
    if not read.has_tag('MA'):
        return [], [], []
    groups = [group for group in str(read.get_tag('MA')).split(';')[1:] if group]
    old_names = (parse_an_tag(str(read.get_tag('AN')))
                 if read.has_tag('AN') else [])
    try:
        old_aq = list(read.get_tag('AQ')) if read.has_tag('AQ') else []
    except (TypeError, ValueError):
        old_aq = []

    layout = []
    expected_aq = 0
    for group in groups:
        feature, qual_spec = _ma_group_head(group)
        count = sum(bool(item) for item in group.partition(':')[2].split(','))
        n_bytes = len(qual_spec) * count
        layout.append((group, feature, count, expected_aq, n_bytes))
        expected_aq += n_bytes
    aq_consistent = len(old_aq) == expected_aq

    kept_groups: List[str] = []
    kept_names: List[str] = []
    kept_aq: List[int] = []
    name_offset = 0
    dropped = []
    for group, feature, count, aq_offset, n_bytes in layout:
        group_names = old_names[name_offset:name_offset + count]
        group_names.extend([''] * (count - len(group_names)))
        name_offset += count
        if feature in _REGENERATED_MA_FEATURES:
            continue
        if n_bytes and not aq_consistent:
            dropped.append(feature)
            continue
        kept_groups.append(group)
        kept_names.extend(group_names)
        if n_bytes:
            kept_aq.extend(old_aq[aq_offset:aq_offset + n_bytes])
    if dropped:
        import warnings
        warnings.warn(
            "write_ma_tags: dropping MA group(s) "
            f"{sorted(set(dropped))} on read "
            f"{getattr(read, 'query_name', '?')}: their AQ qualities could "
            "not be aligned (AQ missing or inconsistent with MA)",
            RuntimeWarning, stacklevel=3,
        )
    return kept_groups, kept_names, kept_aq


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

    # Groups this writer does not regenerate (ddda_mcg/ddda_ucg/hemi from
    # tag-m5c, deam+/deam- from the duplex merge, or any other tool's layer)
    # are preserved verbatim after the refreshed nuc/msp/tf groups, together
    # with their AN names and, for quality-bearing groups, their AQ bytes.
    preserved_groups, preserved_names, preserved_aq = _preserved_ma_groups(read)
    n_preserved = sum(
        sum(bool(item) for item in group.partition(':')[2].split(','))
        for group in preserved_groups
    )

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
    needs_an = (nuc_split or msp_split or tf_split or
                any(preserved_names))

    if not downstream_compat:
        # Spec mode: write MA + AQ. The fiberseq Molecular-annotation spec
        # (https://github.com/fiberseq/Molecular-annotation-spec) requires:
        #   - the MA string to contain >= 1 annotation (regex
        #     ^\d+;(...annotation...;?)+$), so don't emit MA for reads with
        #     no nucs/msps/tfs/m5c at all
        #   - AQ to only be present if SOME annotation type specifies P or Q
        #     (we use Q on nuc+ and tf+; if neither has any annotations and
        #     only msp+ is emitted, AQ stays unwritten)
        has_any_annotation = bool(ma_nucs or ma_msps or ma_tfs or preserved_groups)
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
            if preserved_groups:
                ma = ';'.join([ma, *preserved_groups])
            read.set_tag('MA', ma, value_type='Z')
            if needs_an:
                kept_names = (
                    preserved_names if any(preserved_names)
                    else [f'fh_{DDDA_MCG_FEATURE}_{i}'
                          for i in range(n_preserved)]
                )
                read.set_tag('AN', format_an_tag(
                    nuc_names + msp_names + tf_names + kept_names),
                             value_type='Z')
            elif read.has_tag('AN'):
                try:
                    read.set_tag('AN', None)
                except Exception:
                    pass
            # AQ only carries values for nuc+Q and tf+QQQ. If neither is
            # present in this read, no quality type is in MA -> spec says
            # AQ must not be written.
            has_quality = bool(ma_nucs or ma_tfs or preserved_aq)
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
                if preserved_aq:
                    aq.extend(preserved_aq)
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
