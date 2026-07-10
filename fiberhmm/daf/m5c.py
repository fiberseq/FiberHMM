"""DAF-seq CpG methylation-domain and molecule calling.

The aggregate caller compares each molecule's CpG deamination with its own
non-CpG deamination in the same non-nucleosomal 1 kb window.  The per-read HMM
uses the same contrast in a centered local neighborhood.  This cancels local
accessibility while retaining the CpG-specific suppression caused by 5mC.

Calibration was fit on HG002 LCL chr1_MATERNAL:20-23 Mb and evaluated on the
disjoint 30-33 Mb interval.  Truth coordinates are not used by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from fiberhmm.io.ma_tags import DDDA_MCG_FEATURE

U_UNMETH = 1.113
F_METH = 0.167
BETA_UNMETH = 0.05
BETA_METH = 0.85
EPS = 1e-12
BASE_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}
COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
DDDA_FIVE_PRIME_FACTORS = np.array([0.865, 1.045, 0.944, 1.146])
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
                observations = collect_read_observations(
                    read, fasta,
                    input_molecular_frame=input_molecular_frame,
                )
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


def distance_transition(distance: float, expected_run_bp: float) -> np.ndarray:
    """Symmetric continuous-distance two-state transition matrix.

    This is a persistence prior only: both states have stationary probability
    0.5 and the chain forgets its previous state across sufficiently long gaps.
    """
    if expected_run_bp <= 0:
        raise ValueError("expected_run_bp must be positive")
    distance = max(float(distance), 0.0)
    same = 0.5 + 0.5 * np.exp(-2.0 * distance / expected_run_bp)
    return np.array([[same, 1.0 - same], [1.0 - same, same]])


def distance_forward_backward(log_emission: np.ndarray, positions: np.ndarray,
                              expected_run_bp: float,
                              initial: np.ndarray | None = None) -> np.ndarray:
    """Scaled forward-backward with a transition matrix per CpG gap."""
    if expected_run_bp <= 0:
        raise ValueError("expected_run_bp must be positive")
    emission_log = np.asarray(log_emission, dtype=float)
    positions = np.asarray(positions, dtype=np.int64)
    if emission_log.ndim != 2 or emission_log.shape[1] != 2:
        raise ValueError("log_emission must have shape [CpGs, 2]")
    if len(emission_log) != len(positions) or not len(positions):
        raise ValueError("positions must match a non-empty emission array")
    if np.any(np.diff(positions) < 0):
        raise ValueError("positions must be sorted")
    initial = np.array([0.5, 0.5]) if initial is None else np.asarray(initial, dtype=float)
    if (initial.shape != (2,) or not np.all(np.isfinite(initial)) or
            np.any(initial < 0) or initial.sum() <= 0):
        raise ValueError("initial must contain two non-negative finite state weights")
    emission = np.exp(emission_log - emission_log.max(axis=1, keepdims=True))
    alpha = np.zeros_like(emission)
    backward = np.zeros_like(emission)
    scale = np.zeros(len(emission))
    alpha[0] = initial * emission[0]
    scale[0] = alpha[0].sum() + EPS
    alpha[0] /= scale[0]
    transitions = []
    for i, distance in enumerate(np.diff(positions), start=1):
        matrix = distance_transition(distance, expected_run_bp)
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


def ma_intervals(read, feature: str) -> list[tuple[int, int]]:
    """Parse one MA group into zero-based SEQ-frame half-open intervals."""
    from fiberhmm.io.ma_tags import flip_interval_frame

    if not read.has_tag("MA"):
        return []
    read_length = read.query_length or len(read.query_sequence or "")
    intervals = []
    for group in str(read.get_tag("MA")).split(";")[1:]:
        header, sep, body = group.partition(":")
        if not sep or ma_group_feature(header) != feature:
            continue
        for item in body.split(","):
            if "-" not in item:
                continue
            one_based, size = (int(v) for v in item.split("-", 1))
            interval = (one_based - 1, size)
            if read.is_reverse:
                interval = flip_interval_frame(*interval, read_length)
            intervals.append((interval[0], interval[0] + interval[1]))
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
        read_observations = collect_read_observations(
            read, fasta, chrom, start, end, molecule=next_molecule,
            min_deaminations=min_deaminations,
            max_strand_impurity=max_strand_impurity,
            input_molecular_frame=input_molecular_frame,
        )
        if read_observations:
            observations.extend(read_observations)
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
    return [M5CObservation(
        molecule, int(r), bool(cpg), bool(deam), int(b5), int(q),
    ) for q, r, cpg, deam, b5 in zip(
        query_pos, ref_pos, is_cpg, deaminated, five_code,
    )]


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


def add_m5c_ma_tag(read, query_spans: Sequence[tuple[int, int]]) -> None:
    """Add/replace the unqualified ``ddda_mcg.`` MA group on one read."""
    from fiberhmm.io.ma_tags import flip_interval_frame, parse_an_tag

    read_length = read.query_length or len(read.query_sequence or "")
    molecular = []
    for lo, hi in query_spans:
        start, size = int(lo), int(hi - lo)
        if size <= 0:
            continue
        if read.is_reverse:
            start, size = flip_interval_frame(start, size, read_length)
        molecular.append((start, size))
    molecular.sort()

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
            if ma_group_feature(group) == DDDA_MCG_FEATURE:
                continue
            old_groups.append(group)
            preserved_names.extend(group_names)
    if molecular:
        body = ",".join(f"{start + 1}-{size}" for start, size in molecular)
        old_groups.append(f"{DDDA_MCG_FEATURE}.:{body}")
    if old_groups:
        read.set_tag("MA", ";".join([str(read_length), *old_groups]), value_type="Z")
    elif read.has_tag("MA"):
        read.set_tag("MA", None)

    # AN, when present, has one name per MA annotation. Preserve existing
    # names and append stable DddA-mCG names for the newly added group.
    if read.has_tag("AN"):
        names = preserved_names
        names.extend(f"fh_{DDDA_MCG_FEATURE}_{i}" for i in range(len(molecular)))
        if names:
            read.set_tag("AN", ",".join(name or "." for name in names), value_type="Z")
        else:
            read.set_tag("AN", None)


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
        from fiberhmm.io.bam_header import maybe_append_pg
        output_header = maybe_append_pg(source.header, header_record)
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
    """Call and write molecule-specific ``ddda_mcg.`` spans for every read."""
    import pysam

    stats = {"reads": 0, "eligible_reads": 0, "scored_reads": 0,
             "tagged_reads": 0, "calls": 0, "called_cpgs": 0}
    with pysam.FastaFile(reference) as fasta:
        with pysam.AlignmentFile(input_bam, "rb", threads=threads) as source:
            if input_molecular_frame is None:
                from fiberhmm.io.bam_header import header_has_coord_marker
                input_molecular_frame = header_has_coord_marker(source.header)
            from fiberhmm.io.bam_header import maybe_append_pg
            output_header = maybe_append_pg(source.header, header_record)
            with pysam.AlignmentFile(output_bam, "wb", header=output_header,
                                     threads=threads) as sink:
                for read in source:
                    stats["reads"] += 1
                    observations = collect_read_observations(
                        read, fasta,
                        input_molecular_frame=input_molecular_frame,
                    )
                    if observations:
                        stats["eligible_reads"] += 1
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
                    else:
                        spans = []
                    add_m5c_ma_tag(read, spans)
                    stats["tagged_reads"] += int(bool(spans))
                    sink.write(read)
    return stats
