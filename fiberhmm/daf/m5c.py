"""DAF-seq CpG methylation-domain and molecule calling.

The production molecule caller compares CpG deamination with non-CpG
deamination inside the same initial MSP and reports one state for a complete
sequence-defined CpG island.  This cancels local accessibility while retaining
the CpG-specific suppression caused by 5mC, without claiming a boundary inside
the island.  Aggregate and per-CpG routines remain available for validation of
the underlying emission model, but are not the default molecule annotation.

Calibration was fit on HG002 LCL chr1_MATERNAL:20-23 Mb and evaluated on the
disjoint 30-33 Mb interval.  Truth coordinates are not used by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from fiberhmm.io.ma_tags import (
    DDDA_MCG_FEATURE,
    DDDA_MCG_HEMI_FEATURE,
    DDDA_UCG_FEATURE,
)

U_UNMETH = 1.113
F_METH = 0.167
BETA_UNMETH = 0.05
BETA_METH = 0.85
EPS = 1e-12
BASE_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}
COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
DDDA_FIVE_PRIME_FACTORS = np.array([0.865, 1.045, 0.944, 1.146])
STRAND_C = "C"
STRAND_G = "G"
# State labels are ordered as (C/plus strand, G/minus strand).
PAIRED_STATE_NAMES = ("UU", "UM", "MU", "MM")
PAIRED_C_METHYLATED = np.array([False, False, True, True])
PAIRED_G_METHYLATED = np.array([False, True, False, True])
COMPLEMENT_TABLE = np.full(256, ord("N"), dtype=np.uint8)
BASE_CODE_TABLE = np.full(256, -1, dtype=np.int8)
for _base, _complement in COMPLEMENT.items():
    COMPLEMENT_TABLE[ord(_base)] = ord(_complement)
for _base, _code in BASE_CODE.items():
    BASE_CODE_TABLE[ord(_base)] = _code


@dataclass(frozen=True)
class M5CObservation:
    molecule: int
    reference_pos: int
    is_cpg: bool
    deaminated: bool
    five_prime_base: int
    query_pos: int = -1
    strand: str = ""


@dataclass(frozen=True)
class M5CReadCall:
    """One equal-odds methylated run on an individual read (SEQ frame)."""
    start: int
    end: int
    mean_posterior: float
    min_posterior: float
    n_cpg: int


@dataclass(frozen=True)
class M5CReadResult:
    """Per-CpG HMM result for one read, ordered by reference position."""
    reference_pos: np.ndarray
    query_pos: np.ndarray
    baseline: np.ndarray
    deaminated: np.ndarray
    log_likelihood_ratio: np.ndarray
    methylated_posterior: np.ndarray
    calls: tuple[M5CReadCall, ...]


@dataclass(frozen=True)
class M5CIslandCall:
    """One whole-CpG-island state call on an individual molecule."""
    reference_start: int
    reference_end: int
    query_start: int
    query_end: int
    state: str
    methylated_posterior: float
    n_cpg: int
    deaminated_cpg: int
    n_non_cpg: int
    deaminated_non_cpg: int


@dataclass(frozen=True)
class M5CIslandResult:
    """Whole-island calls; no boundary is inferred within an island."""
    calls: tuple[M5CIslandCall, ...]


@dataclass(frozen=True)
class M5CPairedCall:
    """One posterior-supported state run on a paired DAF duplex."""
    start: int
    end: int
    state: str
    mean_posterior: float
    min_posterior: float
    n_cpg: int


@dataclass(frozen=True)
class M5CPairedReadResult:
    """Four-state result over CpGs observed on one or both DAF strands."""
    reference_pos: np.ndarray
    c_query_pos: np.ndarray
    g_query_pos: np.ndarray
    c_baseline: np.ndarray
    g_baseline: np.ndarray
    c_deaminated: np.ndarray
    g_deaminated: np.ndarray
    c_observed: np.ndarray
    g_observed: np.ndarray
    log_emission: np.ndarray
    state_posterior: np.ndarray
    c_methylated_posterior: np.ndarray
    g_methylated_posterior: np.ndarray
    c_calls: tuple[M5CReadCall, ...]
    g_calls: tuple[M5CReadCall, ...]
    hemi_calls: tuple[M5CPairedCall, ...]


@dataclass(frozen=True)
class M5CWindow:
    start: int
    baseline: np.ndarray
    deaminated: np.ndarray

    @property
    def n_cpg(self) -> int:
        return int(len(self.baseline))


@dataclass(frozen=True)
class M5CDomain:
    chrom: str
    start: int
    end: int
    methylated: bool
    posterior: float

    @property
    def name(self) -> str:
        return "m5c_methylated" if self.methylated else "m5c_unmethylated"

    @property
    def bed_score(self) -> int:
        return max(0, min(1000, int(1000 * self.posterior)))


def deamination_probability(beta, baseline):
    """CpG P(deamination) at methylated fraction *beta*."""
    b = np.asarray(baseline, dtype=np.float64)
    p_u = 1.0 - np.power(1.0 - b, U_UNMETH)
    p_m = 1.0 - np.power(1.0 - b, F_METH)
    return (1.0 - beta) * p_u + beta * p_m


def estimate_five_prime_factors(observations: Sequence[M5CObservation]) -> np.ndarray:
    """Estimate DddA 5' context factors from non-CpG observations."""
    sums = np.zeros(4, dtype=float)
    counts = np.zeros(4, dtype=np.int64)
    for obs in observations:
        if not obs.is_cpg and 0 <= obs.five_prime_base < 4:
            counts[obs.five_prime_base] += 1
            sums[obs.five_prime_base] += int(obs.deaminated)
    if np.any(counts == 0) or np.any(sums == 0):
        raise ValueError(
            "insufficient deaminated non-CpG observations in every A,C,G,T context"
        )
    rates = sums / counts
    rates /= rates.mean()
    return rates


def estimate_bam_five_prime_factors(input_bam: str, reference: str,
                                    max_eligible_reads: int = 5000,
                                    threads: int = 4,
                                    input_molecular_frame: bool | None = None
                                    ) -> np.ndarray:
    """Estimate DddA 5' factors from a bounded sample of one BAM."""
    import pysam

    sums = np.zeros(4, dtype=float)
    counts = np.zeros(4, dtype=np.int64)
    eligible = 0
    with pysam.FastaFile(reference) as fasta:
        with pysam.AlignmentFile(input_bam, "rb", threads=threads) as bam:
            if input_molecular_frame is None:
                from fiberhmm.io.bam_header import header_has_coord_marker
                input_molecular_frame = header_has_coord_marker(bam.header)
            for read in bam:
                c_observations, g_observations, _is_cross_strand = (
                    collect_read_m5c_channels(
                        read, fasta,
                        input_molecular_frame=input_molecular_frame,
                    )
                )
                observations = [*c_observations, *g_observations]
                if not observations:
                    continue
                eligible += 1
                for obs in observations:
                    if not obs.is_cpg:
                        counts[obs.five_prime_base] += 1
                        sums[obs.five_prime_base] += int(obs.deaminated)
                if eligible >= max_eligible_reads:
                    break
    if np.any(counts == 0):
        raise ValueError("insufficient non-CpG observations to estimate 5' factors")
    rates = sums / counts
    if np.any(sums == 0):
        raise ValueError(
            "insufficient deaminated non-CpG observations in every A,C,G,T context"
        )
    rates /= rates.mean()
    return rates


def make_windows(observations: Sequence[M5CObservation], start: int, end: int,
                 window_size: int = 1000, min_other: int = 10,
                 five_prime_factors=None) -> list[M5CWindow]:
    """Build non-overlapping windows with per-molecule internal baselines."""
    if window_size <= 0 or end <= start:
        raise ValueError("window_size must be positive and end must exceed start")
    factors = (estimate_five_prime_factors(observations)
               if five_prime_factors is None else np.asarray(five_prime_factors, dtype=float))
    if (factors.shape != (4,) or not np.all(np.isfinite(factors)) or
            np.any(factors <= 0)):
        raise ValueError("five_prime_factors must contain four positive finite values")
    factors = factors / factors.mean()
    by_window: list[list[M5CObservation]] = [
        [] for _ in range((end - start + window_size - 1) // window_size)
    ]
    for obs in observations:
        if start <= obs.reference_pos < end:
            by_window[(obs.reference_pos - start) // window_size].append(obs)

    result = []
    for index, records in enumerate(by_window):
        other: dict[int, list[int]] = {}
        for obs in records:
            if not obs.is_cpg:
                pair = other.setdefault(obs.molecule, [0, 0])
                pair[0] += int(obs.deaminated)
                pair[1] += 1
        baselines = {m: hits / n for m, (hits, n) in other.items() if n >= min_other}
        base, deam = [], []
        for obs in records:
            if obs.is_cpg and obs.molecule in baselines:
                corrected = baselines[obs.molecule] * factors[obs.five_prime_base]
                base.append(np.clip(corrected, 0.01, 0.97))
                deam.append(obs.deaminated)
        result.append(M5CWindow(
            start=start + index * window_size,
            baseline=np.asarray(base, dtype=float),
            deaminated=np.asarray(deam, dtype=bool),
        ))
    return result


def window_log_likelihood(window: M5CWindow, beta: float) -> float:
    p = np.clip(deamination_probability(beta, window.baseline), EPS, 1.0 - EPS)
    return float(np.sum(np.where(window.deaminated, np.log(p), np.log1p(-p))))


def forward_backward(log_emission: np.ndarray, transition: np.ndarray,
                     initial: np.ndarray) -> np.ndarray:
    """Scaled two-state forward-backward posterior."""
    e = np.asarray(log_emission, dtype=float)
    if e.ndim != 2 or e.shape[0] == 0:
        raise ValueError("log_emission must be a non-empty [windows, states] array")
    emission = np.exp(e - e.max(axis=1, keepdims=True))
    alpha = np.zeros_like(emission)
    beta = np.zeros_like(emission)
    scale = np.zeros(len(emission))
    alpha[0] = initial * emission[0]
    scale[0] = alpha[0].sum() + EPS
    alpha[0] /= scale[0]
    for t in range(1, len(emission)):
        alpha[t] = (alpha[t - 1] @ transition) * emission[t]
        scale[t] = alpha[t].sum() + EPS
        alpha[t] /= scale[t]
    beta[-1] = 1.0
    for t in range(len(emission) - 2, -1, -1):
        beta[t] = transition @ (emission[t + 1] * beta[t + 1])
        beta[t] /= scale[t + 1]
    posterior = alpha * beta
    posterior /= posterior.sum(axis=1, keepdims=True) + EPS
    return posterior


def symmetric_distance_transition(distance: float, expected_run_bp: float,
                                  n_states: int) -> np.ndarray:
    """Symmetric continuous-distance transition for an equal-odds chain.

    Its stationary distribution is uniform, so this supplies persistence but
    no methylation or hemimethylation-frequency prior. The decay matches the
    established two-state model; for the four-state duplex model, either
    strand's methylated/unmethylated marginal therefore has the same distance
    transition as an ordinary single-strand call. Direct symmetric<->symmetric
    transitions remain possible, avoiding a forced hemi intermediate at every
    domain boundary.
    """
    if expected_run_bp <= 0:
        raise ValueError("expected_run_bp must be positive")
    if n_states < 2:
        raise ValueError("n_states must be at least two")
    distance = max(float(distance), 0.0)
    decay = np.exp(-2.0 * distance / expected_run_bp)
    same = 1.0 / n_states + (1.0 - 1.0 / n_states) * decay
    different = (1.0 - same) / (n_states - 1)
    transition = np.full((n_states, n_states), different, dtype=float)
    np.fill_diagonal(transition, same)
    return transition


def distance_transition(distance: float, expected_run_bp: float) -> np.ndarray:
    """Symmetric continuous-distance two-state transition matrix."""
    return symmetric_distance_transition(distance, expected_run_bp, 2)


def distance_forward_backward_states(
    log_emission: np.ndarray,
    positions: np.ndarray,
    expected_run_bp: float,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """Scaled equal-odds forward-backward for any number of states."""
    if expected_run_bp <= 0:
        raise ValueError("expected_run_bp must be positive")
    emission_log = np.asarray(log_emission, dtype=float)
    positions = np.asarray(positions, dtype=np.int64)
    if emission_log.ndim != 2 or emission_log.shape[1] < 2:
        raise ValueError("log_emission must have shape [CpGs, states>=2]")
    if len(emission_log) != len(positions) or not len(positions):
        raise ValueError("positions must match a non-empty emission array")
    if np.any(np.diff(positions) < 0):
        raise ValueError("positions must be sorted")
    n_states = emission_log.shape[1]
    initial = (np.full(n_states, 1.0 / n_states) if initial is None
               else np.asarray(initial, dtype=float))
    if (initial.shape != (n_states,) or not np.all(np.isfinite(initial)) or
            np.any(initial < 0) or initial.sum() <= 0):
        raise ValueError(
            f"initial must contain {n_states} non-negative finite state weights"
        )
    initial = initial / initial.sum()
    emission = np.exp(emission_log - emission_log.max(axis=1, keepdims=True))
    alpha = np.zeros_like(emission)
    backward = np.zeros_like(emission)
    scale = np.zeros(len(emission))
    alpha[0] = initial * emission[0]
    scale[0] = alpha[0].sum() + EPS
    alpha[0] /= scale[0]
    transitions = []
    for i, distance in enumerate(np.diff(positions), start=1):
        matrix = symmetric_distance_transition(
            distance, expected_run_bp, n_states,
        )
        transitions.append(matrix)
        alpha[i] = (alpha[i - 1] @ matrix) * emission[i]
        scale[i] = alpha[i].sum() + EPS
        alpha[i] /= scale[i]
    backward[-1] = 1.0
    for i in range(len(emission) - 2, -1, -1):
        backward[i] = transitions[i] @ (emission[i + 1] * backward[i + 1])
        backward[i] /= scale[i + 1]
    posterior = alpha * backward
    posterior /= posterior.sum(axis=1, keepdims=True) + EPS
    return posterior


def distance_forward_backward(log_emission: np.ndarray, positions: np.ndarray,
                              expected_run_bp: float,
                              initial: np.ndarray | None = None) -> np.ndarray:
    """Scaled forward-backward with a transition matrix per CpG gap."""
    emission_log = np.asarray(log_emission, dtype=float)
    if emission_log.ndim != 2 or emission_log.shape[1] != 2:
        raise ValueError("log_emission must have shape [CpGs, 2]")
    return distance_forward_backward_states(
        emission_log, positions, expected_run_bp, initial=initial,
    )


def score_read_observations(observations: Sequence[M5CObservation],
                            five_prime_factors: Sequence[float],
                            baseline_radius: int = 500,
                            min_other: int = 10
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray]:
    """Score one molecule's CpGs using a centered local non-CpG baseline.

    Returns reference positions, query positions, corrected baselines,
    deamination observations and a two-column [unmethylated, methylated]
    log-emission matrix. No state or locus prior is applied here.
    """
    if baseline_radius <= 0:
        raise ValueError("baseline_radius must be positive")
    factors = np.asarray(five_prime_factors, dtype=float)
    if (factors.shape != (4,) or not np.all(np.isfinite(factors)) or
            np.any(factors <= 0)):
        raise ValueError("five_prime_factors must contain four positive finite values")
    factors = factors / factors.mean()
    records = sorted(observations, key=lambda o: o.reference_pos)
    other = [o for o in records if not o.is_cpg]
    cpg = [o for o in records if o.is_cpg]
    if not other or not cpg:
        empty = np.array([], dtype=float)
        return (empty.astype(np.int64), empty.astype(np.int64), empty,
                empty.astype(bool), np.empty((0, 2)))
    other_pos = np.asarray([o.reference_pos for o in other], dtype=np.int64)
    other_deam = np.asarray([o.deaminated for o in other], dtype=float)
    prefix = np.concatenate([[0.0], np.cumsum(other_deam)])
    ref_pos, query_pos, baseline, deaminated = [], [], [], []
    for obs in cpg:
        lo = np.searchsorted(other_pos, obs.reference_pos - baseline_radius, side="left")
        hi = np.searchsorted(other_pos, obs.reference_pos + baseline_radius, side="right")
        count = int(hi - lo)
        if count < min_other:
            continue
        raw = float((prefix[hi] - prefix[lo]) / count)
        corrected = np.clip(raw * factors[obs.five_prime_base], 0.01, 0.97)
        ref_pos.append(obs.reference_pos)
        query_pos.append(obs.query_pos)
        baseline.append(corrected)
        deaminated.append(obs.deaminated)
    ref_pos = np.asarray(ref_pos, dtype=np.int64)
    query_pos = np.asarray(query_pos, dtype=np.int64)
    baseline = np.asarray(baseline, dtype=float)
    deaminated = np.asarray(deaminated, dtype=bool)
    if not len(ref_pos):
        return ref_pos, query_pos, baseline, deaminated, np.empty((0, 2))
    pu = np.clip(1.0 - np.power(1.0 - baseline, U_UNMETH), EPS, 1.0 - EPS)
    pm = np.clip(1.0 - np.power(1.0 - baseline, F_METH), EPS, 1.0 - EPS)
    emission = np.column_stack([
        np.where(deaminated, np.log(pu), np.log1p(-pu)),
        np.where(deaminated, np.log(pm), np.log1p(-pm)),
    ])
    return ref_pos, query_pos, baseline, deaminated, emission


def call_read_m5c(observations: Sequence[M5CObservation],
                  five_prime_factors: Sequence[float],
                  expected_run_bp: float = 5000.0,
                  posterior_threshold: float = 0.99,
                  baseline_radius: int = 250,
                  min_other: int = 10,
                  min_call_cpg: int = 2,
                  max_call_gap_bp: float | None = None) -> M5CReadResult:
    """Run the equal-odds per-read CpG HMM and emit methylated runs.

    The HMM has a symmetric persistence prior but no aggregate/locus-state
    prior.  Output runs are also split across evidence-free gaps longer than
    ``max_call_gap_bp`` (the expected state length by default), so an MA span
    cannot label a long unobserved stretch merely because both flanks happen
    to be methylated.
    """
    if not 0.5 < posterior_threshold < 1.0:
        raise ValueError("posterior_threshold must be between 0.5 and 1")
    if min_call_cpg < 1:
        raise ValueError("min_call_cpg must be positive")
    if max_call_gap_bp is None:
        max_call_gap_bp = expected_run_bp
    if max_call_gap_bp <= 0:
        raise ValueError("max_call_gap_bp must be positive")
    ref_pos, query_pos, baseline, deaminated, emission = score_read_observations(
        observations, five_prime_factors, baseline_radius, min_other,
    )
    if not len(ref_pos):
        return M5CReadResult(ref_pos, query_pos, baseline, deaminated,
                             np.array([], dtype=float), np.array([], dtype=float), ())
    posterior = distance_forward_backward(emission, ref_pos, expected_run_bp)[:, 1]
    llr = emission[:, 1] - emission[:, 0]
    selected = posterior >= posterior_threshold
    calls = []
    i = 0
    while i < len(selected):
        if not selected[i]:
            i += 1
            continue
        j = i
        while (j + 1 < len(selected) and selected[j + 1] and
               ref_pos[j + 1] - ref_pos[j] <= max_call_gap_bp):
            j += 1
        if j - i + 1 >= min_call_cpg:
            q = query_pos[i:j + 1]
            if np.all(q >= 0):
                calls.append(M5CReadCall(
                    start=int(q.min()), end=int(q.max()) + 1,
                    mean_posterior=float(posterior[i:j + 1].mean()),
                    min_posterior=float(posterior[i:j + 1].min()),
                    n_cpg=j - i + 1,
                ))
        i = j + 1
    calls.sort(key=lambda call: call.start)
    return M5CReadResult(ref_pos, query_pos, baseline, deaminated, llr,
                         posterior, tuple(calls))


def call_read_m5c_islands(
    observations: Sequence[M5CObservation],
    islands: Sequence[tuple[int, int]],
    five_prime_factors: Sequence[float],
    posterior_threshold: float = 0.99,
    min_other: int = 10,
    min_cpg: int = 15,
) -> M5CIslandResult:
    """Call one methylation state per supplied CpG island.

    ``observations`` must already be restricted to the molecule's initially
    called MSPs.  The accessible non-CpG rate is estimated once over the
    molecule/island overlap and used as the accessibility baseline for every
    CpG in that overlap.  The calibrated methylated and unmethylated emission
    likelihoods are combined across CpGs with equal prior odds.  Calls below
    either evidence floor, or between the two posterior thresholds, are
    reported as ``uninformative``.  No transition model or internal island
    boundary is used.
    """
    if not 0.5 < posterior_threshold < 1.0:
        raise ValueError("posterior_threshold must be between 0.5 and 1")
    if min_other < 1 or min_cpg < 1:
        raise ValueError("min_other and min_cpg must be positive")
    factors = np.asarray(five_prime_factors, dtype=float)
    if (factors.shape != (4,) or not np.all(np.isfinite(factors)) or
            np.any(factors <= 0)):
        raise ValueError("five_prime_factors must contain four positive finite values")
    factors = factors / factors.mean()
    records = sorted(observations, key=lambda obs: obs.reference_pos)
    calls = []
    for start, end in sorted((int(a), int(b)) for a, b in islands):
        if end <= start:
            raise ValueError("CpG island intervals must have positive length")
        selected = [obs for obs in records if start <= obs.reference_pos < end]
        cpg = [obs for obs in selected if obs.is_cpg]
        other = [obs for obs in selected if not obs.is_cpg]
        n_cpg, n_other = len(cpg), len(other)
        d_cpg = sum(int(obs.deaminated) for obs in cpg)
        d_other = sum(int(obs.deaminated) for obs in other)
        state = "uninformative"
        posterior = float("nan")
        if n_cpg >= min_cpg and n_other >= min_other:
            raw_baseline = d_other / n_other
            baseline = np.clip(
                raw_baseline * factors[np.asarray(
                    [obs.five_prime_base for obs in cpg], dtype=np.int8,
                )],
                0.01, 0.97,
            )
            deaminated = np.asarray([obs.deaminated for obs in cpg], dtype=bool)
            pu = np.clip(
                deamination_probability(BETA_UNMETH, baseline), EPS, 1.0 - EPS,
            )
            pm = np.clip(
                deamination_probability(BETA_METH, baseline), EPS, 1.0 - EPS,
            )
            log_u = float(np.where(
                deaminated, np.log(pu), np.log1p(-pu),
            ).sum())
            log_m = float(np.where(
                deaminated, np.log(pm), np.log1p(-pm),
            ).sum())
            llr = log_m - log_u
            if llr >= 0:
                posterior = 1.0 / (1.0 + np.exp(-llr))
            else:
                odds = np.exp(llr)
                posterior = odds / (1.0 + odds)
            if posterior >= posterior_threshold:
                state = "methylated"
            elif posterior <= 1.0 - posterior_threshold:
                state = "unmethylated"
        query = [obs.query_pos for obs in selected if obs.query_pos >= 0]
        calls.append(M5CIslandCall(
            reference_start=start,
            reference_end=end,
            query_start=min(query) if query else -1,
            query_end=max(query) + 1 if query else -1,
            state=state,
            methylated_posterior=float(posterior),
            n_cpg=n_cpg,
            deaminated_cpg=d_cpg,
            n_non_cpg=n_other,
            deaminated_non_cpg=d_other,
        ))
    return M5CIslandResult(tuple(calls))


def _selected_runs(selected: np.ndarray, positions: np.ndarray,
                   max_gap_bp: float, min_cpg: int) -> list[tuple[int, int]]:
    """Return inclusive index runs passing a posterior/coverage mask."""
    selected = np.asarray(selected, dtype=bool)
    positions = np.asarray(positions, dtype=np.int64)
    runs = []
    i = 0
    while i < len(selected):
        if not selected[i]:
            i += 1
            continue
        j = i
        while (j + 1 < len(selected) and selected[j + 1] and
               positions[j + 1] - positions[j] <= max_gap_bp):
            j += 1
        if j - i + 1 >= min_cpg:
            runs.append((i, j))
        i = j + 1
    return runs


def _strand_calls(selected: np.ndarray, positions: np.ndarray,
                  query_pos: np.ndarray, posterior: np.ndarray,
                  max_gap_bp: float, min_cpg: int) -> tuple[M5CReadCall, ...]:
    calls = []
    for i, j in _selected_runs(selected, positions, max_gap_bp, min_cpg):
        query = query_pos[i:j + 1]
        if np.any(query < 0):
            continue
        calls.append(M5CReadCall(
            start=int(query.min()), end=int(query.max()) + 1,
            mean_posterior=float(posterior[i:j + 1].mean()),
            min_posterior=float(posterior[i:j + 1].min()),
            n_cpg=j - i + 1,
        ))
    calls.sort(key=lambda call: call.start)
    return tuple(calls)


def _empty_paired_result() -> M5CPairedReadResult:
    integer = np.array([], dtype=np.int64)
    floating = np.array([], dtype=float)
    boolean = np.array([], dtype=bool)
    return M5CPairedReadResult(
        integer, integer.copy(), integer.copy(), floating, floating.copy(),
        boolean, boolean.copy(), boolean.copy(), boolean.copy(),
        np.empty((0, 4)), np.empty((0, 4)), floating.copy(), floating.copy(),
        (), (), (),
    )


def call_paired_read_m5c(
    c_observations: Sequence[M5CObservation],
    g_observations: Sequence[M5CObservation],
    five_prime_factors: Sequence[float],
    expected_run_bp: float = 5000.0,
    posterior_threshold: float = 0.99,
    baseline_radius: int = 250,
    min_other: int = 10,
    min_call_cpg: int = 2,
    max_call_gap_bp: float | None = None,
) -> M5CPairedReadResult:
    """Joint four-state HMM for a cross-strand DAF consensus molecule.

    States are ``UU``, ``UM``, ``MU`` and ``MM`` in (C/plus, G/minus)
    order. Each strand supplies its own local non-CpG accessibility baseline.
    A missing strand contributes a neutral log emission, so single-strand
    flanks can still receive strand-specific mCG calls, while hemimethylation
    is emitted only at CpGs with scored evidence from both strands.
    """
    if not 0.5 < posterior_threshold < 1.0:
        raise ValueError("posterior_threshold must be between 0.5 and 1")
    if min_call_cpg < 1:
        raise ValueError("min_call_cpg must be positive")
    if max_call_gap_bp is None:
        max_call_gap_bp = expected_run_bp
    if max_call_gap_bp <= 0:
        raise ValueError("max_call_gap_bp must be positive")

    c = score_read_observations(
        c_observations, five_prime_factors, baseline_radius, min_other,
    )
    g = score_read_observations(
        g_observations, five_prime_factors, baseline_radius, min_other,
    )
    c_pos, c_query, c_baseline, c_deam, c_emission = c
    g_pos, g_query, g_baseline, g_deam, g_emission = g
    if not len(c_pos) and not len(g_pos):
        return _empty_paired_result()
    if len(np.unique(c_pos)) != len(c_pos) or len(np.unique(g_pos)) != len(g_pos):
        raise ValueError("paired read contains duplicate CpG coordinates on one strand")

    positions = np.union1d(c_pos, g_pos).astype(np.int64, copy=False)
    n = len(positions)
    c_index = np.searchsorted(positions, c_pos)
    g_index = np.searchsorted(positions, g_pos)
    c_observed = np.zeros(n, dtype=bool)
    g_observed = np.zeros(n, dtype=bool)
    c_observed[c_index] = True
    g_observed[g_index] = True
    c_query_all = np.full(n, -1, dtype=np.int64)
    g_query_all = np.full(n, -1, dtype=np.int64)
    c_query_all[c_index] = c_query
    g_query_all[g_index] = g_query
    c_baseline_all = np.full(n, np.nan, dtype=float)
    g_baseline_all = np.full(n, np.nan, dtype=float)
    c_baseline_all[c_index] = c_baseline
    g_baseline_all[g_index] = g_baseline
    c_deam_all = np.zeros(n, dtype=bool)
    g_deam_all = np.zeros(n, dtype=bool)
    c_deam_all[c_index] = c_deam
    g_deam_all[g_index] = g_deam

    # Missing-strand contributions stay at log(1)=0. State order is
    # UU, UM, MU, MM for (C/plus, G/minus).
    c_u = np.zeros(n, dtype=float)
    c_m = np.zeros(n, dtype=float)
    g_u = np.zeros(n, dtype=float)
    g_m = np.zeros(n, dtype=float)
    c_u[c_index], c_m[c_index] = c_emission[:, 0], c_emission[:, 1]
    g_u[g_index], g_m[g_index] = g_emission[:, 0], g_emission[:, 1]
    log_emission = np.column_stack([
        c_u + g_u,
        c_u + g_m,
        c_m + g_u,
        c_m + g_m,
    ])
    posterior = distance_forward_backward_states(
        log_emission, positions, expected_run_bp,
    )
    c_methylated = posterior[:, PAIRED_C_METHYLATED].sum(axis=1)
    g_methylated = posterior[:, PAIRED_G_METHYLATED].sum(axis=1)
    c_calls = _strand_calls(
        c_observed & (c_methylated >= posterior_threshold),
        positions, c_query_all, c_methylated, max_call_gap_bp, min_call_cpg,
    )
    g_calls = _strand_calls(
        g_observed & (g_methylated >= posterior_threshold),
        positions, g_query_all, g_methylated, max_call_gap_bp, min_call_cpg,
    )

    both = c_observed & g_observed
    hemi_calls = []
    for state_index in (1, 2):  # UM (G methylated), MU (C methylated)
        state_posterior = posterior[:, state_index]
        selected = both & (state_posterior >= posterior_threshold)
        for i, j in _selected_runs(
            selected, positions, max_call_gap_bp, min_call_cpg,
        ):
            query = np.concatenate([
                c_query_all[i:j + 1], g_query_all[i:j + 1],
            ])
            query = query[query >= 0]
            hemi_calls.append(M5CPairedCall(
                start=int(query.min()), end=int(query.max()) + 1,
                state=PAIRED_STATE_NAMES[state_index],
                mean_posterior=float(state_posterior[i:j + 1].mean()),
                min_posterior=float(state_posterior[i:j + 1].min()),
                n_cpg=j - i + 1,
            ))
    hemi_calls.sort(key=lambda call: (call.start, call.end, call.state))
    return M5CPairedReadResult(
        positions, c_query_all, g_query_all, c_baseline_all, g_baseline_all,
        c_deam_all, g_deam_all, c_observed, g_observed, log_emission,
        posterior, c_methylated, g_methylated, c_calls, g_calls,
        tuple(hemi_calls),
    )


def call_domains(windows: Sequence[M5CWindow], chrom: str, window_size: int = 1000,
                 beta_unmeth: float = BETA_UNMETH, beta_meth: float = BETA_METH,
                 unmeth_kb: float = 1.5, meth_kb: float = 50.0,
                 posterior_threshold: float = 0.99, min_cpg: int = 10,
                 max_gap: int = 1000) -> tuple[list[M5CDomain], np.ndarray]:
    """Call confident domains; windows below *min_cpg* remain unassigned."""
    if not windows:
        return [], np.empty((0, 2))
    emission = np.array([
        [window_log_likelihood(w, beta_unmeth), window_log_likelihood(w, beta_meth)]
        for w in windows
    ])
    return call_domains_from_emissions(
        emission, np.asarray([w.start for w in windows]),
        np.asarray([w.n_cpg for w in windows]), chrom,
        window_size=window_size, unmeth_kb=unmeth_kb, meth_kb=meth_kb,
        posterior_threshold=posterior_threshold, min_cpg=min_cpg,
        max_gap=max_gap,
    )


def call_domains_from_emissions(log_emission: np.ndarray, starts: np.ndarray,
                                n_cpg: np.ndarray, chrom: str,
                                window_size: int = 1000,
                                unmeth_kb: float = 1.5,
                                meth_kb: float = 50.0,
                                posterior_threshold: float = 0.99,
                                min_cpg: int = 10,
                                max_gap: int = 1000
                                ) -> tuple[list[M5CDomain], np.ndarray]:
    """Segment precomputed aggregate-window log emissions."""
    log_emission = np.asarray(log_emission, dtype=float)
    starts = np.asarray(starts, dtype=np.int64)
    n_cpg = np.asarray(n_cpg, dtype=np.int64)
    if window_size <= 0 or unmeth_kb <= 0 or meth_kb <= 0:
        raise ValueError("window and state run lengths must be positive")
    if not 0.5 <= posterior_threshold < 1.0:
        raise ValueError("posterior_threshold must be at least 0.5 and below 1")
    if min_cpg < 1 or max_gap < 0:
        raise ValueError("min_cpg must be positive and max_gap non-negative")
    if not len(starts):
        return [], np.empty((0, 2))
    if log_emission.shape != (len(starts), 2) or len(n_cpg) != len(starts):
        raise ValueError("emissions, starts and n_cpg must describe the same windows")
    step_kb = window_size / 1000.0
    transition = np.array([
        [max(1.0 - step_kb / unmeth_kb, 0.02), 0.0],
        [0.0, max(1.0 - step_kb / meth_kb, 0.02)],
    ])
    transition[0, 1] = 1.0 - transition[0, 0]
    transition[1, 0] = 1.0 - transition[1, 1]
    posterior = forward_backward(log_emission, transition, np.array([0.05, 0.95]))

    domains: list[M5CDomain] = []
    for state, methylated in ((0, False), (1, True)):
        selected = np.array([
            n_cpg[i] >= min_cpg and posterior[i, state] > posterior_threshold
            for i in range(len(starts))
        ])
        runs = []
        i = 0
        while i < len(selected):
            if not selected[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(selected) and selected[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        merged = []
        for run in runs:
            gap = (starts[run[0]] -
                   (starts[merged[-1][1]] + window_size)) if merged else None
            gap_indices = range(merged[-1][1] + 1, run[0]) if merged else ()
            # Bridge only abstentions. Bridging across a window confidently
            # assigned to the opposite state creates contradictory overlapping
            # BED/MA domains.
            opposite_confident = any(
                n_cpg[k] >= min_cpg and
                posterior[k, 1 - state] > posterior_threshold
                for k in gap_indices
            )
            if merged and gap <= max_gap and not opposite_confident:
                merged[-1][1] = run[1]
            else:
                merged.append(run)
        for i, j in merged:
            mean_post = float(posterior[i:j + 1, state].mean())
            domains.append(M5CDomain(chrom, int(starts[i]),
                                     int(starts[j] + window_size),
                                     methylated, mean_post))
    domains.sort(key=lambda d: (d.chrom, d.start, d.end, d.methylated))
    return domains, posterior


def write_bed(domains: Iterable[M5CDomain], handle) -> None:
    for domain in domains:
        handle.write(f"{domain.chrom}\t{domain.start}\t{domain.end}\t{domain.name}\t"
                     f"{domain.bed_score}\t.\n")


def ma_group_feature(group: str) -> str:
    """Return an MA group's feature name for `.`, `+`, or `-` strand forms."""
    head = str(group).partition(":")[0]
    for index, character in enumerate(head):
        if character in ".+-":
            return head[:index]
    return head


def ma_group_strand(group: str) -> str:
    """Return an MA group's strand qualifier (``.``, ``+`` or ``-``)."""
    head = str(group).partition(":")[0]
    for character in head:
        if character in ".+-":
            return character
    return ""


def ma_strand_intervals(read, feature: str) -> dict[str, list[tuple[int, int]]]:
    """Parse one MA feature into SEQ-frame intervals grouped by qualifier."""
    from fiberhmm.io.ma_tags import flip_interval_frame

    if not read.has_tag("MA"):
        return {}
    read_length = read.query_length or len(read.query_sequence or "")
    intervals: dict[str, list[tuple[int, int]]] = {}
    for group in str(read.get_tag("MA")).split(";")[1:]:
        header, sep, body = group.partition(":")
        if not sep or ma_group_feature(header) != feature:
            continue
        strand = ma_group_strand(header)
        if not strand:
            continue
        strand_intervals = intervals.setdefault(strand, [])
        for item in body.split(","):
            if "-" not in item:
                continue
            one_based, size = (int(v) for v in item.split("-", 1))
            interval = (one_based - 1, size)
            if read.is_reverse:
                interval = flip_interval_frame(*interval, read_length)
            strand_intervals.append((interval[0], interval[0] + interval[1]))
    for values in intervals.values():
        values.sort()
    return intervals


def ma_intervals(read, feature: str) -> list[tuple[int, int]]:
    """Parse all qualifiers of an MA feature as SEQ-frame intervals."""
    intervals = [
        interval
        for values in ma_strand_intervals(read, feature).values()
        for interval in values
    ]
    intervals.sort()
    return intervals


def collect_bam_observations(bam, fasta, chrom: str, start: int, end: int,
                             molecule_offset: int = 0, min_deaminations: int = 20,
                             max_strand_impurity: float = 0.05,
                             input_molecular_frame: bool | None = None
                             ) -> tuple[list[M5CObservation], int]:
    """Collect CpG/non-CpG observations outside recalled nucleosomes.

    MA coordinates are molecular-frame, while BAM SEQ and aligned-pair query
    coordinates are SEQ-frame; reverse-read intervals are flipped before use.
    """
    observations: list[M5CObservation] = []
    next_molecule = int(molecule_offset)
    if input_molecular_frame is None:
        from fiberhmm.io.bam_header import header_has_coord_marker
        input_molecular_frame = header_has_coord_marker(bam.header)
    for read in bam.fetch(chrom, start, end):
        c_observations, g_observations, _is_cross_strand = collect_read_m5c_channels(
            read, fasta, chrom=chrom, start=start, end=end,
            min_deaminations=min_deaminations,
            max_strand_impurity=max_strand_impurity,
            input_molecular_frame=input_molecular_frame,
        )
        for channel in (c_observations, g_observations):
            if not channel:
                continue
            observations.extend(M5CObservation(
                next_molecule, obs.reference_pos, obs.is_cpg,
                obs.deaminated, obs.five_prime_base, obs.query_pos, obs.strand,
            ) for obs in channel)
            next_molecule += 1
    return observations, next_molecule


def collect_read_observations(read, fasta, chrom: str | None = None,
                              start: int | None = None, end: int | None = None,
                              molecule: int = 0, min_deaminations: int = 20,
                              max_strand_impurity: float = 0.05,
                              input_molecular_frame: bool = True
                              ) -> list[M5CObservation]:
    """Collect one read's non-nucleosomal cytosines in SEQ coordinates.

    Recalled BAMs supply nucleosomes through ``MA``. Apply-only BAMs supply
    ``ns/nl`` instead; current files store those arrays in molecular frame,
    while legacy v1 files stored them in query/SEQ frame. The explicit frame
    argument keeps reverse reads correct in both workflows.
    """
    if read.is_unmapped or read.query_sequence is None:
        return []
    has_ma_structure = False
    if read.has_tag("MA"):
        has_ma_structure = any(
            ma_group_feature(group) in {"nuc", "msp", "tf"}
            for group in str(read.get_tag("MA")).split(";")[1:]
        )
    has_structure = (has_ma_structure or
                     (read.has_tag("ns") and read.has_tag("nl")) or
                     (read.has_tag("as") and read.has_tag("al")))
    if not has_structure:
        return []
    chrom = chrom or read.reference_name
    start = read.reference_start if start is None else int(start)
    end = read.reference_end if end is None else int(end)
    sequence = read.query_sequence.upper()
    sequence_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    n_y = int(np.count_nonzero(sequence_bytes == ord("Y")))
    n_r = int(np.count_nonzero(sequence_bytes == ord("R")))
    if n_y + n_r < min_deaminations:
        return []
    if min(n_y, n_r) / max(n_y, n_r, 1) > max_strand_impurity:
        return []
    top = n_y >= n_r
    original_base = ord("C") if top else ord("G")
    deamination_mark = ord("Y") if top else ord("R")
    excluded = np.zeros(len(sequence), dtype=bool)
    nuc_intervals = ma_intervals(read, "nuc") if read.has_tag("MA") else []
    if not nuc_intervals and read.has_tag("ns") and read.has_tag("nl"):
        starts = read.get_tag("ns")
        lengths = read.get_tag("nl")
        if input_molecular_frame:
            from fiberhmm.io.ma_tags import flip_intervals_to_seq
            starts, lengths = flip_intervals_to_seq(starts, lengths, read)
        nuc_intervals = [
            (int(lo), int(lo) + int(size))
            for lo, size in zip(starts, lengths) if int(size) > 0
        ]
    for lo, hi in nuc_intervals:
        excluded[max(0, lo):min(len(sequence), hi)] = True
    from fiberhmm.core.bam_reader import cigar_to_query_ref
    q_to_r = cigar_to_query_ref(read)
    if not len(q_to_r):
        return []
    fetch_start = max(0, read.reference_start - 2)
    reference = np.frombuffer(
        fasta.fetch(chrom, fetch_start, read.reference_end + 2).upper().encode("ascii"),
        dtype=np.uint8,
    )
    target = ((sequence_bytes == original_base) |
              (sequence_bytes == deamination_mark)) & ~excluded
    query_pos = np.flatnonzero(target)
    ref_pos = q_to_r[query_pos]
    valid = (ref_pos >= start) & (ref_pos < end)
    query_pos, ref_pos = query_pos[valid], ref_pos[valid]
    ref_index = ref_pos - fetch_start
    valid = (ref_index > 0) & (ref_index + 1 < len(reference))
    query_pos, ref_pos, ref_index = query_pos[valid], ref_pos[valid], ref_index[valid]
    if not len(query_pos):
        return []
    if top:
        five_prime = reference[ref_index - 1]
        three_prime = reference[ref_index + 1]
    else:
        five_prime = COMPLEMENT_TABLE[reference[ref_index + 1]]
        three_prime = COMPLEMENT_TABLE[reference[ref_index - 1]]
    five_code = BASE_CODE_TABLE[five_prime]
    valid = (five_code >= 0) & (three_prime != ord("N"))
    query_pos, ref_pos = query_pos[valid], ref_pos[valid]
    five_code, three_prime = five_code[valid], three_prime[valid]
    deaminated = sequence_bytes[query_pos] == deamination_mark
    is_cpg = three_prime == ord("G")
    # Store both strands at the canonical reference coordinate of the C in
    # the CpG dyad.  A bottom-strand observation is centred on the reference
    # G, one base to the right.  Canonicalising here makes top and bottom
    # evidence meet at the same site for truth validation, aggregate calling,
    # and haplotype/variance analyses; query_pos still points to the actually
    # observed base on the molecule.
    if not top:
        ref_pos = ref_pos - is_cpg.astype(np.int64)
    strand = STRAND_C if top else STRAND_G
    return [M5CObservation(
        molecule, int(r), bool(cpg), bool(deam), int(b5), int(q), strand,
    ) for q, r, cpg, deam, b5 in zip(
        query_pos, ref_pos, is_cpg, deaminated, five_code,
    )]


def _interval_mask(length: int, intervals: Sequence[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(int(length), dtype=bool)
    for lo, hi in intervals:
        mask[max(0, int(lo)):min(int(length), int(hi))] = True
    return mask


def _nucleosome_intervals_seq(read, input_molecular_frame: bool) -> list[tuple[int, int]]:
    intervals = ma_intervals(read, "nuc") if read.has_tag("MA") else []
    if not intervals and read.has_tag("ns") and read.has_tag("nl"):
        starts = read.get_tag("ns")
        lengths = read.get_tag("nl")
        if input_molecular_frame:
            from fiberhmm.io.ma_tags import flip_intervals_to_seq
            starts, lengths = flip_intervals_to_seq(starts, lengths, read)
        intervals = [
            (int(lo), int(lo) + int(size))
            for lo, size in zip(starts, lengths) if int(size) > 0
        ]
    return intervals


def collect_paired_read_observations(
    read,
    fasta,
    chrom: str | None = None,
    start: int | None = None,
    end: int | None = None,
    molecule: int = 0,
    min_deaminations: int = 20,
    input_molecular_frame: bool = True,
) -> tuple[list[M5CObservation], list[M5CObservation]]:
    """Collect separate C/plus and G/minus channels from a merged DAF read.

    Cross-strand consensus reads declare source-strand coverage with MA
    ``deam+`` and ``deam-`` intervals. The two channels are collected
    independently even where their coverage overlaps; bottom-strand CpGs are
    canonicalized onto the reference-C coordinate while retaining their actual
    query position at the paired reference G.
    """
    if read.is_unmapped or read.query_sequence is None:
        return [], []
    coverage = ma_strand_intervals(read, "deam")
    if "+" not in coverage and "-" not in coverage:
        return [], []
    has_structure = (
        (read.has_tag("MA") and any(
            ma_group_feature(group) in {"nuc", "msp", "tf"}
            for group in str(read.get_tag("MA")).split(";")[1:]
        ))
        or (read.has_tag("ns") and read.has_tag("nl"))
        or (read.has_tag("as") and read.has_tag("al"))
    )
    if not has_structure:
        return [], []

    sequence = read.query_sequence.upper()
    sequence_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    read_length = len(sequence_bytes)
    c_coverage = _interval_mask(read_length, coverage.get("+", ()))
    g_coverage = _interval_mask(read_length, coverage.get("-", ()))
    c_eligible = int(np.count_nonzero(
        c_coverage & (sequence_bytes == ord("Y"))
    )) >= min_deaminations
    g_eligible = int(np.count_nonzero(
        g_coverage & (sequence_bytes == ord("R"))
    )) >= min_deaminations
    if not c_eligible and not g_eligible:
        return [], []

    excluded = _interval_mask(
        read_length, _nucleosome_intervals_seq(read, input_molecular_frame),
    )
    from fiberhmm.core.bam_reader import cigar_to_query_ref
    q_to_r = cigar_to_query_ref(read)
    if not len(q_to_r):
        return [], []
    chrom = chrom or read.reference_name
    start = read.reference_start if start is None else int(start)
    end = read.reference_end if end is None else int(end)
    fetch_start = max(0, read.reference_start - 2)
    reference = np.frombuffer(
        fasta.fetch(chrom, fetch_start, read.reference_end + 2).upper().encode("ascii"),
        dtype=np.uint8,
    )

    def collect_channel(strand: str, channel_coverage: np.ndarray
                        ) -> list[M5CObservation]:
        top = strand == STRAND_C
        original_base = ord("C") if top else ord("G")
        deamination_mark = ord("Y") if top else ord("R")
        allowed = ((sequence_bytes == original_base) |
                   (sequence_bytes == deamination_mark))
        query_pos = np.flatnonzero(channel_coverage & ~excluded & allowed)
        ref_pos = q_to_r[query_pos]
        valid = (ref_pos >= start) & (ref_pos < end)
        query_pos, ref_pos = query_pos[valid], ref_pos[valid]
        ref_index = ref_pos - fetch_start
        valid = (ref_index > 0) & (ref_index + 1 < len(reference))
        query_pos, ref_pos, ref_index = (
            query_pos[valid], ref_pos[valid], ref_index[valid]
        )
        valid = reference[ref_index] == original_base
        query_pos, ref_pos, ref_index = (
            query_pos[valid], ref_pos[valid], ref_index[valid]
        )
        if not len(query_pos):
            return []
        if top:
            five_prime = reference[ref_index - 1]
            three_prime = reference[ref_index + 1]
        else:
            five_prime = COMPLEMENT_TABLE[reference[ref_index + 1]]
            three_prime = COMPLEMENT_TABLE[reference[ref_index - 1]]
        five_code = BASE_CODE_TABLE[five_prime]
        valid = (five_code >= 0) & (three_prime != ord("N"))
        query_pos, ref_pos = query_pos[valid], ref_pos[valid]
        five_code, three_prime = five_code[valid], three_prime[valid]
        deaminated = sequence_bytes[query_pos] == deamination_mark
        is_cpg = three_prime == ord("G")
        if not top:
            ref_pos = ref_pos - is_cpg.astype(np.int64)
        return [
            M5CObservation(
                molecule, int(ref), bool(cpg), bool(deam), int(five),
                int(query), strand,
            )
            for query, ref, cpg, deam, five in zip(
                query_pos, ref_pos, is_cpg, deaminated, five_code,
            )
        ]

    return (
        collect_channel(STRAND_C, c_coverage) if c_eligible else [],
        collect_channel(STRAND_G, g_coverage) if g_eligible else [],
    )


def collect_read_m5c_channels(
    read,
    fasta,
    chrom: str | None = None,
    start: int | None = None,
    end: int | None = None,
    min_deaminations: int = 20,
    max_strand_impurity: float = 0.05,
    input_molecular_frame: bool = True,
) -> tuple[list[M5CObservation], list[M5CObservation], bool]:
    """Return C observations, G observations, and cross-strand status."""
    coverage = ma_strand_intervals(read, "deam") if read.has_tag("MA") else {}
    is_cross_strand = "+" in coverage or "-" in coverage
    if is_cross_strand:
        c_observations, g_observations = collect_paired_read_observations(
            read, fasta, chrom=chrom, start=start, end=end,
            min_deaminations=min_deaminations,
            input_molecular_frame=input_molecular_frame,
        )
        return c_observations, g_observations, True
    observations = collect_read_observations(
        read, fasta, chrom=chrom, start=start, end=end,
        min_deaminations=min_deaminations,
        max_strand_impurity=max_strand_impurity,
        input_molecular_frame=input_molecular_frame,
    )
    if not observations:
        return [], [], False
    if observations[0].strand == STRAND_G:
        return [], observations, False
    return observations, [], False


def build_ddda_mcg_observation_payload(
    read,
    fasta,
    daf_result=None,
    min_deaminations: int = 20,
    max_strand_impurity: float = 0.05,
) -> dict | None:
    """Build reference-context observations for fused ``fiberhmm-call``.

    Unlike :func:`collect_read_observations`, this runs before the apply-stage
    nucleosome calls exist. It therefore records every eligible cytosine in a
    compact array payload; the fused worker removes preliminary nucleosome
    intervals immediately before running the per-read mCG HMM.

    ``daf_result`` is the optional ``(ct_positions, ga_positions, strand)``
    already computed for raw-MD/reference DAF input. R/Y-encoded reads derive
    the same information directly from their stored sequence. A reference
    FASTA is mandatory because CpG identity and DddA 5' context cannot be
    reconstructed reliably from an amplified/deaminated query alone.
    """
    if (fasta is None or read.is_unmapped or read.query_sequence is None or
            read.reference_name is None):
        return None
    if min_deaminations < 1:
        raise ValueError("min_deaminations must be positive")
    if not 0.0 <= max_strand_impurity < 1.0:
        raise ValueError("max_strand_impurity must be in [0, 1)")

    sequence = read.query_sequence.upper()
    sequence_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    y_pos = np.flatnonzero(sequence_bytes == ord("Y"))
    r_pos = np.flatnonzero(sequence_bytes == ord("R"))
    is_iupac = bool(len(y_pos) or len(r_pos))
    if is_iupac:
        n_top, n_bottom = len(y_pos), len(r_pos)
        top = n_top >= n_bottom
        deamination_pos = y_pos if top else r_pos
    else:
        if daf_result is None:
            from fiberhmm.daf.encoder import get_daf_positions
            daf_result = get_daf_positions(read, ref_fasta=fasta)
        if daf_result is None:
            return None
        ct_pos, ga_pos, strand = daf_result
        n_top, n_bottom = len(ct_pos), len(ga_pos)
        top = str(strand).upper() == "CT"
        deamination_pos = np.asarray(ct_pos if top else ga_pos, dtype=np.int64)

    if n_top + n_bottom < min_deaminations:
        return None
    if min(n_top, n_bottom) / max(n_top, n_bottom, 1) > max_strand_impurity:
        return None

    from fiberhmm.core.bam_reader import cigar_to_query_ref
    q_to_r = cigar_to_query_ref(read)
    if not len(q_to_r):
        return None
    fetch_start = max(0, int(read.reference_start) - 2)
    fetch_end = int(read.reference_end) + 2
    reference = np.frombuffer(
        fasta.fetch(read.reference_name, fetch_start, fetch_end)
        .upper().encode("ascii"),
        dtype=np.uint8,
    )

    mapped = q_to_r >= 0
    query_pos = np.flatnonzero(mapped)
    ref_pos = q_to_r[query_pos]
    ref_index = ref_pos - fetch_start
    valid = (ref_index > 0) & (ref_index + 1 < len(reference))
    query_pos, ref_pos, ref_index = (
        query_pos[valid], ref_pos[valid], ref_index[valid]
    )
    if not len(query_pos):
        return None

    original_base = ord("C") if top else ord("G")
    reference_base = reference[ref_index]
    if is_iupac:
        allowed = ((sequence_bytes[query_pos] == original_base) |
                   (sequence_bytes[query_pos] == (ord("Y") if top else ord("R"))))
    else:
        converted_base = ord("T") if top else ord("A")
        allowed = ((sequence_bytes[query_pos] == original_base) |
                   (sequence_bytes[query_pos] == converted_base))
    valid = (reference_base == original_base) & allowed
    query_pos, ref_pos, ref_index = (
        query_pos[valid], ref_pos[valid], ref_index[valid]
    )
    if not len(query_pos):
        return None

    if top:
        five_prime = reference[ref_index - 1]
        three_prime = reference[ref_index + 1]
    else:
        five_prime = COMPLEMENT_TABLE[reference[ref_index + 1]]
        three_prime = COMPLEMENT_TABLE[reference[ref_index - 1]]
    five_code = BASE_CODE_TABLE[five_prime]
    valid = (five_code >= 0) & (three_prime != ord("N"))
    query_pos, ref_pos = query_pos[valid], ref_pos[valid]
    five_code, three_prime = five_code[valid], three_prime[valid]
    if not len(query_pos):
        return None

    deamination_mask = np.zeros(len(sequence_bytes), dtype=bool)
    deamination_pos = np.asarray(deamination_pos, dtype=np.int64)
    deamination_pos = deamination_pos[
        (deamination_pos >= 0) & (deamination_pos < len(deamination_mask))
    ]
    deamination_mask[deamination_pos] = True
    deaminated = deamination_mask[query_pos]
    is_cpg = three_prime == ord("G")
    if not top:
        ref_pos = ref_pos - is_cpg.astype(np.int64)

    return {
        "query_pos": np.asarray(query_pos, dtype=np.int32),
        "reference_pos": np.asarray(ref_pos, dtype=np.int64),
        "is_cpg": np.asarray(is_cpg, dtype=bool),
        "deaminated": np.asarray(deaminated, dtype=bool),
        "five_prime_base": np.asarray(five_code, dtype=np.int8),
        "strand": STRAND_C if top else STRAND_G,
    }


def observations_from_ddda_mcg_payload(
    payload: dict | None,
    excluded_intervals: Sequence[tuple[int, int]] = (),
) -> list[M5CObservation]:
    """Materialize HMM observations after excluding SEQ-frame intervals."""
    if not payload:
        return []
    required = (
        "query_pos", "reference_pos", "is_cpg", "deaminated",
        "five_prime_base",
    )
    arrays = [np.asarray(payload.get(key, ())) for key in required]
    lengths = {len(values) for values in arrays}
    if len(lengths) != 1:
        raise ValueError("DddA mCG observation payload arrays must align")
    if not arrays or not len(arrays[0]):
        return []

    keep = np.ones(len(arrays[0]), dtype=bool)
    query_pos = arrays[0].astype(np.int64, copy=False)
    for lo, hi in excluded_intervals:
        keep &= ~((query_pos >= int(lo)) & (query_pos < int(hi)))
    query_pos, reference_pos, is_cpg, deaminated, five_prime_base = (
        values[keep] for values in arrays
    )
    return [
        M5CObservation(
            0, int(ref), bool(cpg), bool(deam), int(five), int(query),
            str(payload.get("strand", "")),
        )
        for query, ref, cpg, deam, five in zip(
            query_pos, reference_pos, is_cpg, deaminated, five_prime_base,
        )
    ]


def project_domains_to_query(read, domains: Sequence[M5CDomain]) -> list[tuple[int, int]]:
    """Project methylated reference domains to SEQ-frame query spans."""
    if read.is_unmapped or not read.query_sequence:
        return []
    from fiberhmm.core.bam_reader import cigar_to_query_ref

    q_to_ref = cigar_to_query_ref(read)
    spans = []
    for domain in domains:
        if not domain.methylated or domain.chrom != read.reference_name:
            continue
        positions = np.flatnonzero((q_to_ref >= domain.start) & (q_to_ref < domain.end))
        if len(positions):
            # Include query insertions between the first and last aligned base;
            # they belong to the same locus annotation on the molecule.
            spans.append((int(positions[0]), int(positions[-1]) + 1))
    spans.sort()
    merged = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def _replace_m5c_ma_groups(
    read,
    groups: Sequence[tuple[str, str, Sequence[tuple[int, int]]]],
) -> None:
    """Replace mCG/hemi MA groups while preserving all other MA/AN entries."""
    from fiberhmm.io.ma_tags import flip_interval_frame, parse_an_tag

    read_length = read.query_length or len(read.query_sequence or "")
    encoded_groups = []
    for feature, strand, query_spans in groups:
        if strand not in {".", "+", "-"}:
            raise ValueError("MA strand qualifier must be '.', '+' or '-'")
        molecular = []
        for lo, hi in query_spans:
            start, size = int(lo), int(hi - lo)
            if size <= 0:
                continue
            if read.is_reverse:
                start, size = flip_interval_frame(start, size, read_length)
            molecular.append((start, size))
        molecular.sort()
        if molecular:
            encoded_groups.append((feature, strand, molecular))

    old_groups = []
    preserved_names = []
    existing_names = (
        parse_an_tag(str(read.get_tag("AN"))) if read.has_tag("AN") else []
    )
    name_offset = 0
    if read.has_tag("MA"):
        tokens = str(read.get_tag("MA")).split(";")
        for group in tokens[1:]:
            body = group.partition(":")[2]
            annotation_count = sum(bool(item) for item in body.split(","))
            group_names = existing_names[
                name_offset:name_offset + annotation_count
            ]
            group_names.extend([""] * (annotation_count - len(group_names)))
            name_offset += annotation_count
            if ma_group_feature(group) in {
                DDDA_MCG_FEATURE, DDDA_UCG_FEATURE, DDDA_MCG_HEMI_FEATURE,
            }:
                continue
            old_groups.append(group)
            preserved_names.extend(group_names)
    generated_names = []
    for feature, strand, molecular in encoded_groups:
        body = ",".join(f"{start + 1}-{size}" for start, size in molecular)
        old_groups.append(f"{feature}{strand}:{body}")
        strand_name = {".": "", "+": "_plus", "-": "_minus"}[strand]
        generated_names.extend(
            f"fh_{feature}{strand_name}_{i}" for i in range(len(molecular))
        )
    if old_groups:
        read.set_tag("MA", ";".join([str(read_length), *old_groups]), value_type="Z")
    elif read.has_tag("MA"):
        read.set_tag("MA", None)

    # AN, when present, has one name per MA annotation. Preserve existing
    # names and append stable DddA-mCG names for the newly added group.
    if read.has_tag("AN"):
        names = preserved_names
        names.extend(generated_names)
        if names:
            read.set_tag("AN", ",".join(name or "." for name in names), value_type="Z")
        else:
            read.set_tag("AN", None)


def add_m5c_ma_tag(read, query_spans: Sequence[tuple[int, int]]) -> None:
    """Add/replace the unqualified ``ddda_mcg.`` MA group on one read."""
    _replace_m5c_ma_groups(read, [
        (DDDA_MCG_FEATURE, ".", query_spans),
    ])


def add_island_m5c_ma_tags(
    read,
    methylated_spans: Sequence[tuple[int, int]],
    unmethylated_spans: Sequence[tuple[int, int]],
) -> None:
    """Write confident whole-island mCpG and uCpG states on one molecule."""
    _replace_m5c_ma_groups(read, [
        (DDDA_MCG_FEATURE, ".", methylated_spans),
        (DDDA_UCG_FEATURE, ".", unmethylated_spans),
    ])


def add_paired_m5c_ma_tags(
    read,
    c_spans: Sequence[tuple[int, int]],
    g_spans: Sequence[tuple[int, int]],
    hemi_c_spans: Sequence[tuple[int, int]],
    hemi_g_spans: Sequence[tuple[int, int]],
) -> None:
    """Write strand-resolved mCG and explicit hemi groups on a merged read."""
    _replace_m5c_ma_groups(read, [
        (DDDA_MCG_FEATURE, "+", c_spans),
        (DDDA_MCG_FEATURE, "-", g_spans),
        (DDDA_MCG_HEMI_FEATURE, "+", hemi_c_spans),
        (DDDA_MCG_HEMI_FEATURE, "-", hemi_g_spans),
    ])


def annotate_bam_from_domains(input_bam: str, output_bam: str,
                              domains: Sequence[M5CDomain],
                              threads: int = 4,
                              header_record: dict | None = None
                              ) -> tuple[int, int]:
    """Write locus-level methylated domains as per-read ``ddda_mcg.`` spans."""
    import pysam

    by_chrom: dict[str, list[M5CDomain]] = {}
    for domain in domains:
        if domain.methylated:
            by_chrom.setdefault(domain.chrom, []).append(domain)
    indexed = {}
    for chrom, values in by_chrom.items():
        values.sort(key=lambda value: value.start)
        indexed[chrom] = (
            values, np.asarray([value.start for value in values], dtype=np.int64),
        )
    total = tagged = 0
    with pysam.AlignmentFile(input_bam, "rb", threads=threads) as source:
        from fiberhmm.io.bam_header import append_ma_types, maybe_append_pg
        output_header = append_ma_types(
            maybe_append_pg(source.header, header_record),
            (DDDA_MCG_FEATURE,),
        )
        with pysam.AlignmentFile(output_bam, "wb", header=output_header,
                                 threads=threads) as sink:
            for read in source:
                spans = []
                if not read.is_unmapped and read.reference_name in indexed:
                    values, starts = indexed[read.reference_name]
                    left = max(0, int(np.searchsorted(
                        starts, read.reference_start, side="left",
                    )) - 1)
                    right = int(np.searchsorted(
                        starts, read.reference_end, side="left",
                    ))
                    spans = project_domains_to_query(read, values[left:right])
                add_m5c_ma_tag(read, spans)
                tagged += int(bool(spans))
                total += 1
                sink.write(read)
    return total, tagged


def annotate_bam_per_read(input_bam: str, output_bam: str, reference: str,
                          five_prime_factors: Sequence[float],
                          expected_run_bp: float = 5000.0,
                          posterior_threshold: float = 0.99,
                          baseline_radius: int = 250,
                          min_other: int = 10,
                          min_call_cpg: int = 2,
                          max_call_gap_bp: float | None = None,
                          input_molecular_frame: bool | None = None,
                          threads: int = 4,
                          header_record: dict | None = None) -> dict[str, int]:
    """Call molecule mCG, including paired-strand hemi states when available."""
    import pysam

    stats = {"reads": 0, "eligible_reads": 0, "scored_reads": 0,
             "c_strand_reads": 0, "g_strand_reads": 0,
             "cross_strand_reads": 0, "tagged_reads": 0,
             "calls": 0, "called_cpgs": 0, "both_scored_cpgs": 0,
             "hemi_calls": 0, "hemi_c_calls": 0, "hemi_g_calls": 0,
             "hemi_cpgs": 0}
    with pysam.FastaFile(reference) as fasta:
        with pysam.AlignmentFile(input_bam, "rb", threads=threads) as source:
            if input_molecular_frame is None:
                from fiberhmm.io.bam_header import header_has_coord_marker
                input_molecular_frame = header_has_coord_marker(source.header)
            from fiberhmm.io.bam_header import append_ma_types, maybe_append_pg
            output_header = append_ma_types(
                maybe_append_pg(source.header, header_record),
                (DDDA_MCG_FEATURE, DDDA_MCG_HEMI_FEATURE),
            )
            with pysam.AlignmentFile(output_bam, "wb", header=output_header,
                                     threads=threads) as sink:
                for read in source:
                    stats["reads"] += 1
                    c_observations, g_observations, is_cross_strand = (
                        collect_read_m5c_channels(
                            read, fasta,
                            input_molecular_frame=input_molecular_frame,
                        )
                    )
                    if c_observations or g_observations:
                        stats["eligible_reads"] += 1
                    stats["c_strand_reads"] += int(bool(c_observations))
                    stats["g_strand_reads"] += int(bool(g_observations))
                    stats["cross_strand_reads"] += int(is_cross_strand)
                    if is_cross_strand:
                        result = call_paired_read_m5c(
                            c_observations, g_observations, five_prime_factors,
                            expected_run_bp=expected_run_bp,
                            posterior_threshold=posterior_threshold,
                            baseline_radius=baseline_radius,
                            min_other=min_other,
                            min_call_cpg=min_call_cpg,
                            max_call_gap_bp=max_call_gap_bp,
                        )
                        stats["scored_reads"] += int(bool(len(result.reference_pos)))
                        c_spans = [(call.start, call.end) for call in result.c_calls]
                        g_spans = [(call.start, call.end) for call in result.g_calls]
                        hemi_c_spans = [
                            (call.start, call.end) for call in result.hemi_calls
                            if call.state == "MU"
                        ]
                        hemi_g_spans = [
                            (call.start, call.end) for call in result.hemi_calls
                            if call.state == "UM"
                        ]
                        stats["calls"] += len(result.c_calls) + len(result.g_calls)
                        stats["called_cpgs"] += (
                            sum(call.n_cpg for call in result.c_calls) +
                            sum(call.n_cpg for call in result.g_calls)
                        )
                        stats["both_scored_cpgs"] += int(np.count_nonzero(
                            result.c_observed & result.g_observed
                        ))
                        stats["hemi_calls"] += len(result.hemi_calls)
                        stats["hemi_c_calls"] += len(hemi_c_spans)
                        stats["hemi_g_calls"] += len(hemi_g_spans)
                        stats["hemi_cpgs"] += sum(
                            call.n_cpg for call in result.hemi_calls
                        )
                        add_paired_m5c_ma_tags(
                            read, c_spans, g_spans, hemi_c_spans, hemi_g_spans,
                        )
                        spans = [*c_spans, *g_spans]
                    elif c_observations or g_observations:
                        observations = c_observations or g_observations
                        result = call_read_m5c(
                            observations, five_prime_factors,
                            expected_run_bp=expected_run_bp,
                            posterior_threshold=posterior_threshold,
                            baseline_radius=baseline_radius,
                            min_other=min_other,
                            min_call_cpg=min_call_cpg,
                            max_call_gap_bp=max_call_gap_bp,
                        )
                        stats["scored_reads"] += int(bool(len(result.reference_pos)))
                        spans = [(call.start, call.end) for call in result.calls]
                        stats["calls"] += len(result.calls)
                        stats["called_cpgs"] += sum(call.n_cpg for call in result.calls)
                        add_m5c_ma_tag(read, spans)
                    else:
                        spans = []
                        add_m5c_ma_tag(read, spans)
                    stats["tagged_reads"] += int(bool(spans))
                    sink.write(read)
    return stats


def load_cpg_island_bed(path: str) -> dict[str, list[tuple[int, int]]]:
    """Load a BED3 CpG-island file as sorted, non-overlapping intervals."""
    import gzip

    opener = gzip.open if str(path).endswith(".gz") else open
    by_chrom: dict[str, list[tuple[int, int]]] = {}
    with opener(path, "rt") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 3:
                raise ValueError(f"CpG-island BED line {line_number} has fewer than 3 columns")
            try:
                start, end = int(fields[1]), int(fields[2])
            except ValueError as error:
                raise ValueError(
                    f"CpG-island BED line {line_number} has invalid coordinates"
                ) from error
            if start < 0 or end <= start:
                raise ValueError(
                    f"CpG-island BED line {line_number} must be a positive BED interval"
                )
            by_chrom.setdefault(fields[0], []).append((start, end))
    for chrom, intervals in by_chrom.items():
        intervals.sort()
        previous_end = -1
        for start, end in intervals:
            if start < previous_end:
                raise ValueError(
                    f"CpG-island BED contains overlapping intervals on {chrom}; "
                    "merge them before calling"
                )
            previous_end = end
    return by_chrom


def infer_cpg_islands(
    reference: str,
    contigs: Sequence[str] | None = None,
    window: int = 200,
    step: int = 10,
    min_gc: float = 0.50,
    min_observed_expected: float = 0.60,
    chunk_bp: int = 5_000_000,
) -> dict[str, list[tuple[int, int]]]:
    """Infer merged CpG islands directly from an indexed reference FASTA.

    Windows use the conventional sequence criteria: GC fraction at least
    ``min_gc`` and CpG observed/expected at least
    ``min_observed_expected``, where O/E is ``n_CpG * window / (n_C * n_G)``.
    Qualifying overlapping windows are merged.  Chunking bounds memory without
    changing the globally aligned sliding-window grid.
    """
    import pysam

    if window < 2 or step < 1 or chunk_bp < step:
        raise ValueError("CpG-island window, step and chunk size must be positive")
    if not 0.0 <= min_gc <= 1.0 or min_observed_expected < 0.0:
        raise ValueError("invalid CpG-island GC or observed/expected threshold")

    result: dict[str, list[tuple[int, int]]] = {}
    with pysam.FastaFile(reference) as fasta:
        available = set(fasta.references)
        selected = list(fasta.references) if contigs is None else [
            str(chrom) for chrom in contigs if str(chrom) in available
        ]
        if not selected:
            raise ValueError("no BAM contigs are present in the reference FASTA")
        for chrom in selected:
            length = int(fasta.get_reference_length(chrom))
            max_start = length - int(window)
            if max_start < 0:
                continue
            merged: list[list[int]] = []
            # Partition possible window starts.  Each fetch includes the full
            # final window, so no window or dinucleotide crosses a chunk unseen.
            for core_start in range(0, max_start + 1, int(chunk_bp)):
                core_end = min(max_start + 1, core_start + int(chunk_bp))
                first = core_start + ((-core_start) % int(step))
                if first >= core_end:
                    continue
                starts = np.arange(first, core_end, int(step), dtype=np.int64)
                fetch_start = int(starts[0])
                fetch_end = int(starts[-1]) + int(window)
                sequence = np.frombuffer(
                    fasta.fetch(chrom, fetch_start, fetch_end).upper().encode("ascii"),
                    dtype=np.uint8,
                )
                is_c = sequence == ord("C")
                is_g = sequence == ord("G")
                is_cpg = np.zeros(len(sequence), dtype=np.int8)
                if len(sequence) > 1:
                    is_cpg[:-1] = is_c[:-1] & is_g[1:]

                def _prefix(values):
                    return np.concatenate((
                        np.zeros(1, dtype=np.int64),
                        np.cumsum(values, dtype=np.int64),
                    ))

                c_prefix = _prefix(is_c)
                g_prefix = _prefix(is_g)
                cpg_prefix = _prefix(is_cpg)
                offsets = starts - fetch_start
                ends = offsets + int(window)
                n_c = c_prefix[ends] - c_prefix[offsets]
                n_g = g_prefix[ends] - g_prefix[offsets]
                n_cpg = cpg_prefix[ends] - cpg_prefix[offsets]
                gc_fraction = (n_c + n_g) / float(window)
                denominator = n_c * n_g
                observed_expected = np.divide(
                    n_cpg * float(window), denominator,
                    out=np.zeros(len(starts), dtype=np.float64),
                    where=denominator > 0,
                )
                qualifying = starts[
                    (gc_fraction >= float(min_gc)) &
                    (observed_expected >= float(min_observed_expected))
                ]
                for start in qualifying:
                    lo, hi = int(start), int(start) + int(window)
                    if merged and lo <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], hi)
                    else:
                        merged.append([lo, hi])
            if merged:
                result[chrom] = [(lo, hi) for lo, hi in merged]
    return result


def write_cpg_island_bed(
    islands: dict[str, Sequence[tuple[int, int]]], path: str,
) -> None:
    """Write the exact normalized CpG-island catalog used for calling."""
    import gzip

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt") as handle:
        for chrom, intervals in islands.items():
            for start, end in intervals:
                handle.write(f"{chrom}\t{int(start)}\t{int(end)}\n")


def _initial_msp_intervals_seq(read, input_molecular_frame: bool) -> list[tuple[int, int]]:
    """Return initial MSPs in SEQ coordinates from MA or legacy as/al tags."""
    intervals = ma_intervals(read, "msp") if read.has_tag("MA") else []
    if intervals or not (read.has_tag("as") and read.has_tag("al")):
        return intervals
    starts, lengths = list(read.get_tag("as")), list(read.get_tag("al"))
    if input_molecular_frame:
        from fiberhmm.io.ma_tags import flip_intervals_to_seq
        starts, lengths = flip_intervals_to_seq(starts, lengths, read)
    return [
        (int(start), int(start) + int(length))
        for start, length in zip(starts, lengths) if int(length) > 0
    ]


def annotate_bam_per_read_islands(
    input_bam: str,
    output_bam: str,
    reference: str,
    island_bed: str | None,
    five_prime_factors: Sequence[float],
    posterior_threshold: float = 0.99,
    min_other: int = 10,
    min_cpg: int = 15,
    input_molecular_frame: bool | None = None,
    threads: int = 4,
    header_record: dict | None = None,
    calls_tsv: str | None = None,
    used_islands_bed: str | None = None,
    island_window: int = 200,
    island_step: int = 10,
    island_min_gc: float = 0.50,
    island_min_observed_expected: float = 0.60,
) -> dict[str, int]:
    """Call whole CpG islands from observations inside initial MSPs.

    Confident methylated and unmethylated islands are written as
    ``MA:ddda_mcg.`` and ``MA:ddda_ucg.`` spans, respectively.  CpG-aware TF
    recall uses the unmethylated layer as a whitelist: CpGs outside those
    spans are excluded from the opportunity lattice.  The optional table also
    records uninformative overlaps.  If ``island_bed`` is absent, islands are
    inferred from the reference sequence using the conventional windowed GC
    and CpG observed/expected definition.
    """
    import csv
    import pysam

    if island_bed:
        islands = load_cpg_island_bed(island_bed)
    else:
        with pysam.AlignmentFile(input_bam, "rb", check_sq=False) as source:
            contigs = source.references
        with pysam.FastaFile(reference) as fasta:
            available = set(fasta.references)
        missing = [chrom for chrom in contigs if chrom not in available]
        if missing:
            preview = ", ".join(missing[:5])
            suffix = " ..." if len(missing) > 5 else ""
            raise ValueError(
                "reference FASTA is missing BAM contigs required for automatic "
                f"CpG-island inference: {preview}{suffix}"
            )
        islands = infer_cpg_islands(
            reference,
            contigs=contigs,
            window=island_window,
            step=island_step,
            min_gc=island_min_gc,
            min_observed_expected=island_min_observed_expected,
        )
    if used_islands_bed:
        write_cpg_island_bed(islands, used_islands_bed)
    starts = {
        chrom: np.asarray([start for start, _ in values], dtype=np.int64)
        for chrom, values in islands.items()
    }
    stats = {
        "defined_islands": sum(len(values) for values in islands.values()),
        "reads": 0,
        "eligible_reads": 0,
        "overlapped_islands": 0,
        "methylated_islands": 0,
        "unmethylated_islands": 0,
        "uninformative_islands": 0,
        "tagged_reads": 0,
    }
    table_handle = open(calls_tsv, "w", newline="") if calls_tsv else None
    writer = None
    if table_handle:
        writer = csv.writer(table_handle, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "read_name", "chrom", "island_start", "island_end", "state",
            "methylated_posterior", "n_cpg", "deaminated_cpg", "n_non_cpg",
            "deaminated_non_cpg",
        ])
    try:
        with pysam.FastaFile(reference) as fasta:
            with pysam.AlignmentFile(input_bam, "rb", threads=threads) as source:
                if input_molecular_frame is None:
                    from fiberhmm.io.bam_header import header_has_coord_marker
                    input_molecular_frame = header_has_coord_marker(source.header)
                from fiberhmm.io.bam_header import append_ma_types, maybe_append_pg
                output_header = append_ma_types(
                    maybe_append_pg(source.header, header_record),
                    (DDDA_MCG_FEATURE, DDDA_UCG_FEATURE),
                )
                with pysam.AlignmentFile(
                    output_bam, "wb", header=output_header, threads=threads,
                ) as sink:
                    for read in source:
                        stats["reads"] += 1
                        candidate_islands = []
                        chrom = read.reference_name
                        if (not read.is_unmapped and chrom in islands and
                                read.reference_start is not None and
                                read.reference_end is not None):
                            values = islands[chrom]
                            chrom_starts = starts[chrom]
                            left = max(0, int(np.searchsorted(
                                chrom_starts, read.reference_start, side="right",
                            )) - 1)
                            right = int(np.searchsorted(
                                chrom_starts, read.reference_end, side="left",
                            ))
                            candidate_islands = [
                                interval for interval in values[left:right]
                                if interval[0] < read.reference_end and
                                interval[1] > read.reference_start
                            ]
                        calls = ()
                        if candidate_islands:
                            payload = build_ddda_mcg_observation_payload(read, fasta)
                            msp = _initial_msp_intervals_seq(
                                read, bool(input_molecular_frame),
                            )
                            if payload and msp:
                                observations = observations_from_ddda_mcg_payload(payload)
                                observations = [
                                    obs for obs in observations
                                    if any(lo <= obs.query_pos < hi for lo, hi in msp)
                                ]
                                if observations:
                                    stats["eligible_reads"] += 1
                                    calls = call_read_m5c_islands(
                                        observations,
                                        candidate_islands,
                                        five_prime_factors,
                                        posterior_threshold=posterior_threshold,
                                        min_other=min_other,
                                        min_cpg=min_cpg,
                                    ).calls
                        stats["overlapped_islands"] += len(calls)
                        for call in calls:
                            stats[f"{call.state}_islands"] += 1
                            if writer:
                                writer.writerow([
                                    read.query_name, chrom,
                                    call.reference_start, call.reference_end,
                                    call.state,
                                    call.methylated_posterior,
                                    call.n_cpg, call.deaminated_cpg,
                                    call.n_non_cpg, call.deaminated_non_cpg,
                                ])
                        methylated = [
                            M5CDomain(
                                chrom, call.reference_start, call.reference_end,
                                True, call.methylated_posterior,
                            )
                            for call in calls if call.state == "methylated"
                        ]
                        unmethylated = [
                            M5CDomain(
                                chrom, call.reference_start, call.reference_end,
                                False, 1.0 - call.methylated_posterior,
                            )
                            for call in calls if call.state == "unmethylated"
                        ]
                        methylated_spans = project_domains_to_query(read, methylated)
                        unmethylated_spans = project_domains_to_query(read, unmethylated)
                        add_island_m5c_ma_tags(
                            read, methylated_spans, unmethylated_spans,
                        )
                        stats["tagged_reads"] += int(bool(
                            methylated_spans or unmethylated_spans
                        ))
                        sink.write(read)
    finally:
        if table_handle:
            table_handle.close()
    return stats
