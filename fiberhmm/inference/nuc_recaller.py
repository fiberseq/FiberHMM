"""Per-read nucleosome recaller.

Splits over-merged HMM footprints on accessible (m6a/deam) evidence, then resolves
each resulting fragment's edges and quality. Reuses the TF recaller's Kadane kernel
with *inverted* emission tables for splitting and *non-inverted* tables for the
nucleosome edge + quality pass -- no new scoring code. The topology policy adds
an HMM-occupancy constraint for sparse single-strand evidence: cuts must separate
nucleosome-sized pieces, and unresolved edge ambiguity stays protected.

  SPLIT:  call_tfs_in_interval(obs, ..., -llr_hit, -llr_miss)  over a footprint
          interior -> accessible runs == cuts. Footprint is split at the cuts.
  EDGES:  call_tfs_in_interval(obs, ..., +llr_hit, +llr_miss)  over each resulting
          fragment -> protected call whose conservative start/length trims the
          Viterbi overshoot, whose cumulative LLR -> nq, whose left/right
          ambiguity -> el/er (conservative+loose edge convention, same as tf+QQQ).

Design notes: nuc_recaller_collab/DESIGN.md (esp. §7b). The split is evidence-only
(no size prior); DddB recovers ~20-30% of buried linkers, which is the accepted
floor for an under-deaminating enzyme. Fiber-seq / DddA give the kernel much more
signal per read.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import erf, sqrt
from typing import List, Sequence, Tuple

import numpy as np

from fiberhmm.inference.tf_recaller import (
    N_CTX,
    UNMETH_OFFSET,
    call_tfs_in_interval,
    merge_intervals,
)
from fiberhmm.io.ma_tags import ambiguity_to_edge, llr_to_tq

Interval = Tuple[int, int]


@dataclass
class NucCall:
    """A refined nucleosome before conversion to MA/AQ (nuc+QQQ) output."""
    start: int       # query coord, 0-based, inclusive (conservative edge)
    length: int      # bp (conservative span)
    nq: int          # quality byte from cumulative protected LLR (0-255)
    el: int          # left-edge sharpness byte (0-255; 255 = sharp)
    er: int          # right-edge sharpness byte
    # Internal-only radial-model anchor.  It is deliberately not serialized in
    # MA/AQ: the public call remains a start/length interval, while validation
    # can distinguish a true dyad-centred edge proposal from an arbitrary
    # protected block.  Non-DddA callers leave this unset.
    dyad: int | None = None
    # Raw density-transition proposal retained across iterative validation.
    # These are internal provenance only (never written to BAM tags) and let a
    # later TF-recall iteration test an asymmetric but empirically nominated
    # flank instead of reconstructing an artificial symmetric dyad envelope.
    radial_start: int | None = None
    radial_end: int | None = None
    # Baseline-only diagnostic: the molecule-local phase posterior's central
    # interval was narrower than the 30-bp Q saturation width. These flags are
    # never serialized and must never gate a coordinate or tiling decision;
    # posterior uncertainty belongs only in el/er.
    phase_resolved_left: bool = False
    phase_resolved_right: bool = False


def _refine_fragment(obs, a, b, llr_hit, llr_miss,
                     nuc_min_size, edge_min_llr, edge_min_opps,
                     preserve_fragment=False):
    """Edge-refine one protected fragment into a NucCall (or demote it).

    Returns ``(nuc_or_None, access_intervals)``. A fragment shorter than
    ``nuc_min_size`` is demoted. Under the historical conservative policy, a
    protected core that trims below the floor is also demoted; signal deserts
    retain a quality-0 NucCall. ``preserve_fragment`` instead keeps a qualifying
    HMM fragment and records unresolved edges.
    """
    access: List[Interval] = []
    if b - a < nuc_min_size:
        access.append((a, b - a))
        return None, access
    prot = call_tfs_in_interval(obs, a, b, llr_hit, llr_miss,
                                edge_min_llr, edge_min_opps)
    if not prot:
        # signal-desert fragment: keep raw extent, unknown quality/edges
        return NucCall(a, b - a, nq=0, el=0, er=0), access
    prot = sorted(prot, key=lambda p: p.start)
    first, last = prot[0], prot[-1]
    total_llr = sum(p.llr for p in prot)
    if preserve_fragment:
        # Sparse single-strand evidence does not identify conservative edges.
        # The HMM extent remains the occupancy prior; the protected scan still
        # supplies a quality score, while zero edge bytes honestly record that
        # the exact boundaries were not resolved by bracketing evidence.
        return NucCall(
            start=a,
            length=b - a,
            nq=llr_to_tq(total_llr),
            el=0,
            er=0,
        ), access
    cstart = first.start
    cend = last.start + last.length
    if cend - cstart < nuc_min_size:
        # Edge refinement trimmed the protected core below the floor (a sparse
        # protected island) -> not a nucleosome, demote the whole fragment.
        access.append((a, b - a))
        return None, access
    nuc = NucCall(
        start=cstart,
        length=cend - cstart,
        nq=llr_to_tq(total_llr),
        el=ambiguity_to_edge(first.left_ambiguity),
        er=ambiguity_to_edge(last.right_ambiguity),
    )
    if cstart > a:
        access.append((a, cstart - a))
    if b > cend:
        access.append((cend, b - cend))
    return nuc, access


def _select_nucleosome_separating_cuts(
    cuts,
    start: int,
    end: int,
    nuc_min_size: int,
):
    """Select the maximum-evidence cut chain with nuc-sized pieces throughout.

    The accessible Kadane scan can find isolated ONT events inside a single
    HMM-protected footprint. Treating every such run as a split can shatter one
    nucleosome into sub-floor fragments, after which the old recaller labels the
    entire footprint accessible. A true *nucleosome split* must instead leave a
    possible nucleosome on both outer sides and between consecutive cuts.

    Dynamic programming maximizes retained cut LLR subject to that topology.
    All current call LLRs are positive, so an eligible non-conflicting cut is
    retained unless a stronger incompatible chain exists.
    """
    floor = max(1, int(nuc_min_size))
    eligible = [
        cut
        for cut in sorted(cuts, key=lambda call: call.start)
        if (
            int(cut.start) - int(start) >= floor
            and int(end) - int(cut.start + cut.length) >= floor
        )
    ]
    if not eligible:
        return []

    best_score: List[float] = []
    predecessor: List[int | None] = []
    for i, cut in enumerate(eligible):
        score = float(cut.llr)
        pred = None
        for j in range(i):
            gap = int(cut.start) - int(
                eligible[j].start + eligible[j].length
            )
            candidate = best_score[j] + float(cut.llr)
            if gap >= floor and candidate > score:
                score = candidate
                pred = j
        best_score.append(score)
        predecessor.append(pred)

    cursor: int | None = int(np.argmax(np.asarray(best_score)))
    selected = []
    while cursor is not None:
        selected.append(eligible[cursor])
        cursor = predecessor[cursor]
    return list(reversed(selected))


def _phase_subfragments(obs, a, b, nhit, nmiss, nrl,
                        phase_min_llr, phase_min_opps, phase_window,
                        min_fragment_size=0):
    """Evidence-gated periodicity split of a long protected fragment.

    A fragment of length L >= 1.5*nrl is assumed to hold ``n = round(L/nrl)``
    nucleosomes. At each predicted internal linker (evenly spaced to fit L), scan
    a +-``phase_window`` bp window for an accessible run with the LOWERED
    ``phase_min_llr`` threshold; the strongest qualifying run becomes a cut.
    Returns ``(subfragments, cut_intervals)``; with no qualifying cut the
    fragment is returned whole (never split into a signal-desert).
    """
    L = b - a
    if L < int(1.5 * nrl):
        return [(a, b)], []
    n = int(round(L / float(nrl)))
    if n < 2:
        return [(a, b)], []
    spacing = L / float(n)
    cut_calls = []
    for i in range(1, n):
        pred = a + int(round(i * spacing))
        lo = max(a, pred - phase_window)
        hi = min(b, pred + phase_window)
        if hi - lo < 2:
            continue
        found = call_tfs_in_interval(obs, lo, hi, nhit, nmiss,
                                     phase_min_llr, phase_min_opps)
        if found:
            best = max(found, key=lambda c: c.llr)
            cut_calls.append(best)
    if min_fragment_size > 0:
        cut_calls = _select_nucleosome_separating_cuts(
            cut_calls, a, b, min_fragment_size)
    if not cut_calls:
        return [(a, b)], []
    cut_pairs = [
        (int(call.start), int(call.start + call.length))
        for call in cut_calls
    ]
    cut_pairs.sort()
    subs: List[Interval] = []
    cur = a
    for cs, ce in cut_pairs:
        if cs > cur:
            subs.append((cur, cs))
        cur = max(cur, ce)
    if cur < b:
        subs.append((cur, b))
    cut_intervals = [(cs, ce - cs) for cs, ce in cut_pairs]
    return subs, cut_intervals


# ===================================================================== #
#  DddA phase-aware radial split (nuc_profile mode)                      #
#                                                                        #
#  DddA deaminates *inside* nucleosomes, so the accessible-cut split     #
#  above shatters them. Instead, match-filter a single-nucleosome RADIAL #
#  template (deam rate vs distance from dyad) to place dyads, then infer #
#  each edge with one context-aware posterior marginalized over uncertain #
#  helical register and local pitch. The posterior median is always the  #
#  coordinate; posterior width is retained in edge quality.              #
# ===================================================================== #

@dataclass(frozen=True)
class NucProfile:
    """Empirical within-nucleosome deamination radial template (DddA mode)."""
    radial: np.ndarray       # deam rate vs |offset from dyad|, index 0..half
    linker: float            # flat linker deam rate
    half: int                # nucleosome footprint half-extent (bp)
    min_sep: int             # min dyad-dyad separation (bp)
    edge_frac: float         # threshold (x linker) for the edge crossing
    # Weak physical extent prior used in the same edge posterior for every
    # molecule.  Broad/censored posteriors emit a saturated Q0 edge so their
    # point estimate is never mistaken for a molecule-resolved boundary.
    edge_prior_center: float = 0.0
    edge_prior_sd: float = 0.0
    edge_likelihood_temperature: float = 1.0
    # Optional dropout-tolerant rotational-access model. A zero period keeps
    # historical/custom profiles on the radial-only behavior.
    rotation_period: float = 0.0
    rotation_period_sd: float = 0.0
    rotation_phase: float = 0.0
    rotation_phase_sd: float = 0.0
    rotation_phase_bins: int = 0
    rotation_band_sd: float = 1.0
    rotation_fraction: float = 0.0
    rotation_edge_break_min_surprisal: float = 0.0
    rotation_min_phase_information: float = 0.5
    protected_hit: np.ndarray | None = None
    accessible_hit: np.ndarray | None = None


def load_nuc_profile(path: str) -> NucProfile:
    import json
    with open(path) as handle:
        d = json.load(handle)
    rotation = d.get('rotation', {})
    edge_prior = d.get('edge_prior', {})
    return NucProfile(
        radial=np.asarray(d['dyad_rate'], dtype=np.float64),
        linker=float(d['linker']),
        half=int(d.get('half', 73)),
        min_sep=int(d.get('min_sep', 150)),
        edge_frac=float(d.get('edge_frac', 0.82)),
        edge_prior_center=float(edge_prior.get('center', 0.0)),
        edge_prior_sd=float(edge_prior.get('sd', 0.0)),
        edge_likelihood_temperature=float(
            edge_prior.get('likelihood_temperature', 1.0)
        ),
        rotation_period=float(rotation.get('period', 0.0)),
        rotation_period_sd=float(rotation.get('period_sd', 0.0)),
        rotation_phase=float(rotation.get('phase', 0.0)),
        rotation_phase_sd=float(rotation.get('phase_sd', 0.0)),
        rotation_phase_bins=int(rotation.get('phase_bins', 0)),
        rotation_band_sd=float(rotation.get('band_sd', 1.0)),
        rotation_fraction=float(
            rotation.get('fraction', rotation.get('accessibility', 0.0))
        ),
        rotation_edge_break_min_surprisal=float(
            rotation.get('edge_break_min_surprisal', 0.0)
        ),
        rotation_min_phase_information=float(
            rotation.get('min_phase_information', 0.5)
        ),
    )


def attach_nuc_profile_emissions(
    profile: NucProfile | None,
    protected_hit,
    accessible_hit,
) -> NucProfile | None:
    """Attach chemistry/context hit probabilities to an immutable profile."""
    if profile is None:
        return None
    return replace(
        profile,
        protected_hit=np.asarray(protected_hit, dtype=np.float64),
        accessible_hit=np.asarray(accessible_hit, dtype=np.float64),
    )


def _obs_opp_deam(obs: np.ndarray):
    """Opportunity (target) and deaminated (accessible 'hit') masks from obs."""
    obs = np.asarray(obs)
    hit = (obs >= 0) & (obs < N_CTX)
    miss = (obs >= UNMETH_OFFSET) & (obs < UNMETH_OFFSET + N_CTX)
    return (hit | miss), hit


def _logmeanexp(values, log_weights, axis=0):
    """Stable weighted log-mean-exp for a small latent-model grid."""
    array = np.asarray(values, dtype=np.float64)
    weights = np.asarray(log_weights, dtype=np.float64)
    shape = [1] * array.ndim
    shape[0] = len(weights)
    weighted = array + weights.reshape(shape)
    maximum = np.max(weighted, axis=axis, keepdims=True)
    total = np.sum(np.exp(weighted - maximum), axis=axis)
    normalizer = np.sum(np.exp(weights - np.max(weights)))
    return (
        np.squeeze(maximum, axis=axis)
        + np.log(total)
        - np.log(normalizer)
        - np.max(weights)
    )


def _rotation_grid(profile: NucProfile):
    """Return period/phase hypotheses and fixed prior log weights.

    Period and phase are global particle properties, not independently fitted
    to each observed hit. Marginalizing this small calibrated grid permits
    9--12-bp local spacing and uncertain dyad placement without allowing an
    arbitrary phase to explain every molecule.
    """
    period = float(profile.rotation_period)
    if period <= 0.0 or float(profile.rotation_fraction) <= 0.0:
        return None
    z = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    periods = (
        period + z * float(profile.rotation_period_sd)
        if float(profile.rotation_period_sd) > 0.0
        else np.asarray([period])
    )
    rows = []
    for period_index, candidate_period in enumerate(periods):
        if candidate_period <= 0.0:
            continue
        period_score = (
            -0.5 * float(z[period_index] ** 2)
            if len(periods) == len(z) else 0.0
        )
        phase_bins = max(0, int(profile.rotation_phase_bins))
        if phase_bins > 0:
            # The radial dyad does not localize the DNA helical register on an
            # individual molecule. Infer that register from its own internal
            # observations under a circularly uniform prior.
            phases = np.linspace(
                0.0, float(candidate_period), phase_bins, endpoint=False,
            )
            for phase in phases:
                rows.append((
                    float(candidate_period), float(phase), period_score,
                ))
        else:
            phase_z = (
                z if float(profile.rotation_phase_sd) > 0.0
                else np.asarray([0.0])
            )
            for phase_offset in phase_z:
                rows.append((
                    float(candidate_period),
                    float(profile.rotation_phase)
                    + float(phase_offset) * float(profile.rotation_phase_sd),
                    period_score - 0.5 * float(phase_offset ** 2),
                ))
    return rows


def _rotational_wrapped_llr_matrix(
    observations,
    positions,
    distances,
    profile,
    llr_hit,
    llr_miss,
    *,
    m5c_mask=None,
    m5c_llr_hit=None,
    m5c_llr_miss=None,
):
    """Context-aware wrapped-vs-linker LLR for every latent phase model.

    The protected/accessibility LLR from the chemistry model is attenuated at
    predicted outward-facing bands. Thus an on-phase hit is compatible with a
    wrapped particle, an off-phase hit favors linker, and a missed band merely
    contributes weaker protected evidence. Non-opportunity sequence is neutral.
    """
    grid = _rotation_grid(profile)
    protected_table = getattr(profile, "protected_hit", None)
    accessible_table = getattr(profile, "accessible_hit", None)
    if grid is None or protected_table is None or accessible_table is None:
        return None
    obs = np.asarray(observations)
    pos = np.asarray(positions, dtype=np.int64)
    dist = np.asarray(distances, dtype=np.float64)
    codes = obs[pos]
    hits = (codes >= 0) & (codes < N_CTX)
    misses = (
        (codes >= UNMETH_OFFSET)
        & (codes < UNMETH_OFFSET + N_CTX)
    )
    contexts = np.zeros(len(pos), dtype=np.int64)
    contexts[hits] = codes[hits]
    contexts[misses] = codes[misses] - UNMETH_OFFSET
    protected_probability = np.clip(
        np.asarray(protected_table, dtype=np.float64)[contexts],
        1e-12,
        1.0 - 1e-12,
    )
    accessible_probability = np.clip(
        np.asarray(accessible_table, dtype=np.float64)[contexts],
        1e-12,
        1.0 - 1e-12,
    )

    radial = np.clip(
        np.nan_to_num(
            np.asarray(profile.radial, dtype=np.float64), nan=0.05,
        ),
        0.0,
        float(profile.linker),
    )
    radial_indices = np.minimum(
        dist.astype(np.int64), min(int(profile.half), radial.size - 1),
    )
    base_accessibility = np.clip(
        radial[radial_indices]
        / max(float(profile.linker), 1e-6),
        0.0,
        0.95,
    )
    band_sd = max(0.25, float(profile.rotation_band_sd))
    rotation_fraction = float(np.clip(
        profile.rotation_fraction, 0.0, 1.0,
    ))
    matrix = []
    log_weights = []
    for period, phase, log_weight in grid:
        phase_distance = np.abs(
            np.mod(dist - phase + 0.5 * period, period) - 0.5 * period
        )
        band = np.exp(-0.5 * np.square(phase_distance / band_sd))
        # ``radial`` is already averaged over helical register. Modulate it by
        # a mean-one wrapped Gaussian so the rotation model redistributes, but
        # never inflates, the calibrated per-distance accessibility.
        mean_band = (
            sqrt(2.0 * np.pi) * band_sd / period
            * erf(period / (2.0 * sqrt(2.0) * band_sd))
        )
        modulation = (
            (1.0 - rotation_fraction)
            + rotation_fraction * band / max(mean_band, 1e-6)
        )
        accessibility = np.clip(base_accessibility * modulation, 0.0, 0.999)
        wrapped_probability = (
            (1.0 - accessibility) * protected_probability
            + accessibility * accessible_probability
        )
        row = np.zeros(len(pos), dtype=np.float64)
        row[hits] = np.log(
            wrapped_probability[hits] / accessible_probability[hits]
        )
        row[misses] = np.log(
            (1.0 - wrapped_probability[misses])
            / (1.0 - accessible_probability[misses])
        )
        matrix.append(row)
        log_weights.append(log_weight)
    return np.asarray(matrix), np.asarray(log_weights)


def _direct_edge_break_surprisal_scores(
    profile,
    distances,
    latent_log_weights,
    prefix,
    candidate_indices,
    outward_hits,
    max_hits=1,
):
    """Vectorized phase surprisal for candidate edges.

    Observations inward of each candidate update the period/phase grid. For the
    first outward hit, calculate posterior expected rotational exposure.
    The returned ``-log(E[band])`` is low on phase and high near a half-turn.
    The surrounding change-point still uses context-conditioned emissions; this
    is a direct-event gate, not a replacement likelihood.
    """
    grid = _rotation_grid(profile)
    distance = np.asarray(distances, dtype=np.float64)
    weights = np.asarray(latent_log_weights, dtype=np.float64)
    cumulative = np.asarray(prefix, dtype=np.float64)
    hit_mask = np.asarray(outward_hits, dtype=bool)
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    empty = np.zeros(candidates.size, dtype=np.float64)
    if (
        grid is None
        or cumulative.ndim != 2
        or cumulative.shape[1] != distance.size
        or cumulative.shape[0] != len(grid)
        or candidates.size == 0
    ):
        return empty
    if np.any(candidates < 0) or np.any(candidates >= distance.size):
        raise ValueError("candidate edge index outside distance track")
    hit_indices = np.flatnonzero(hit_mask)
    if hit_indices.size == 0:
        return empty
    # The first hit outside a proposed edge is the direct edge event. Looking
    # ahead to a later hit would let an off-phase event two turns away validate
    # a prematurely truncated boundary despite intervening on-phase hits.
    width = max(1, int(max_hits))
    first_outward = np.searchsorted(hit_indices, candidates + 1)
    ranks = first_outward[:, None] + np.arange(width)[None, :]
    selected_valid = ranks < hit_indices.size
    selected = hit_indices[np.minimum(ranks, hit_indices.size - 1)]

    posterior_log_weights = cumulative[:, candidates].T + weights[None, :]
    posterior_log_weights -= np.max(
        posterior_log_weights, axis=1, keepdims=True,
    )
    posterior = np.exp(posterior_log_weights)
    posterior /= np.sum(posterior, axis=1, keepdims=True)
    band_sd = max(0.25, float(profile.rotation_band_sd))
    bands = []
    for period, phase, _log_weight in grid:
        phase_distance = np.abs(
            np.mod(distance - phase + 0.5 * period, period)
            - 0.5 * period
        )
        bands.append(np.exp(-0.5 * np.square(phase_distance / band_sd)))
    band_matrix = np.asarray(bands, dtype=np.float64)
    selected_bands = band_matrix[:, selected]
    expected_band = np.einsum(
        'km,mkh->kh', posterior, selected_bands, optimize=True,
    )
    surprisal = -np.log(np.clip(expected_band, 1e-6, 1.0))
    surprisal[~selected_valid] = 0.0
    return np.max(surprisal, axis=1)


def _rotational_edge_posterior(
    profile,
    observations,
    opportunity,
    deaminated,
    c,
    direction,
    bound,
    llr_hit,
    llr_miss,
    phase_condition_sum=None,
    phase_core_radius=0,
):
    """Infer one particle edge with a single phase-marginal posterior.

    Candidate offset ``d`` says that positions 1..d from the dyad are wrapped
    and all positions beyond d are linker.  Because linker is the likelihood
    baseline, the candidate score is simply the marginal wrapped LLR prefix.
    The opposite side of the dyad may condition the latent helical register,
    but no observation is used twice.

    A weak physical extent prior, when configured, is applied to this same
    posterior for every molecule.  The posterior median is always the reported
    coordinate; a broad or boundary-censored central 90% interval changes only
    the edge-quality byte.  In particular, there is no confidence threshold at
    which the estimator switches to the HMM/adjacent-dyad topology: that switch
    produced an artificial cliff and structural-size spikes in population
    distributions.

    Returns ``None`` when the rotational/context model is unavailable so the
    historical radial-density path can handle custom profiles.  Otherwise
    returns ``(edge, ambiguity_bp)`` for both resolved and unresolved cases.
    """
    if (
        observations is None
        or opportunity is None
        or deaminated is None
        or llr_hit is None
        or llr_miss is None
    ):
        return None

    obs = np.asarray(observations)
    opp = np.asarray(opportunity, dtype=bool)
    max_distance = min(
        abs(int(bound) - int(c)),
        int(c) if int(direction) < 0 else len(opp) - 1 - int(c),
    )
    if max_distance <= 0:
        return int(c), 30

    distances = np.arange(1, max_distance + 1, dtype=np.int64)
    positions = int(c) + int(direction) * distances
    rotational = _rotational_wrapped_llr_matrix(
        obs,
        positions,
        distances,
        profile,
        llr_hit,
        llr_miss,
    )
    if rotational is None:
        return None
    wrapped_llr, latent_log_weights = rotational
    prefix = np.cumsum(wrapped_llr, axis=1)

    core_radius = min(max_distance, max(0, int(phase_core_radius)))
    if phase_condition_sum is None:
        other_condition = np.zeros(
            wrapped_llr.shape[0], dtype=np.float64,
        )
    else:
        current_core = (
            np.sum(wrapped_llr[:, :core_radius], axis=1)
            if core_radius > 0
            else np.zeros(wrapped_llr.shape[0], dtype=np.float64)
        )
        other_condition = (
            np.asarray(phase_condition_sum, dtype=np.float64)
            - current_core
        )

    baseline = float(_logmeanexp(
        other_condition[:, None], latent_log_weights, axis=0,
    )[0])
    log_likelihood = _logmeanexp(
        other_condition[:, None] + prefix,
        latent_log_weights,
        axis=0,
    ) - baseline
    temperature = max(
        1.0,
        float(getattr(profile, "edge_likelihood_temperature", 1.0)),
    )
    log_posterior = log_likelihood / temperature
    prior_center = float(getattr(profile, "edge_prior_center", 0.0))
    prior_sd = float(getattr(profile, "edge_prior_sd", 0.0))
    if prior_center > 0.0 and prior_sd > 0.0:
        # A weak physical particle-extent prior is part of the same posterior
        # for every molecule.  It must never be substituted as a point estimate
        # only for an ambiguity-selected subset: that estimator switch created
        # the historical 146-bp population cliff.
        log_posterior -= 0.5 * np.square(
            (distances.astype(np.float64) - prior_center) / prior_sd
        )
    log_posterior -= float(np.max(log_posterior))
    posterior = np.exp(log_posterior)
    posterior /= float(np.sum(posterior))
    cumulative = np.cumsum(posterior)

    def quantile(probability):
        return int(distances[min(
            len(distances) - 1,
            int(np.searchsorted(cumulative, probability)),
        )])

    low = quantile(0.05)
    median = quantile(0.5)
    high = quantile(0.95)
    ambiguity = int(high - low)

    # Censoring and broad support reduce only the confidence channel.  Never
    # swap the posterior estimator for an HMM/topology coordinate at a width
    # threshold: doing so piles selected molecules onto the few structural
    # constants used as search bounds (150/185/220-bp span artifacts).
    boundary_censored = low == int(distances[0]) or high == int(distances[-1])
    if boundary_censored:
        ambiguity = max(30, ambiguity)
    return int(c) + int(direction) * median, max(0, ambiguity)


def _conditional_phase_break_surprisal(
    profile,
    offsets,
    hit_mask,
    latent_log_weights,
    condition_sum,
):
    """Return flank-hit phase surprisal when the core learned a register.

    ``None`` means the core posterior did not carry enough information to
    support an exploratory phase-continuation rescue. A numeric value is the
    strongest ``-log(E[band])`` among flank hits under that posterior.
    """
    hits = np.asarray(hit_mask, dtype=bool)
    if not np.any(hits):
        return None
    grid = _rotation_grid(profile)
    if grid is None:
        return None
    prior_log = np.asarray(latent_log_weights, dtype=np.float64)
    prior_log -= float(np.max(prior_log))
    prior = np.exp(prior_log)
    prior /= float(np.sum(prior))
    posterior_log = prior_log + np.asarray(condition_sum, dtype=np.float64)
    posterior_log -= float(np.max(posterior_log))
    posterior = np.exp(posterior_log)
    posterior /= float(np.sum(posterior))
    information = float(np.sum(
        posterior * (
            np.log(np.clip(posterior, 1e-300, 1.0))
            - np.log(np.clip(prior, 1e-300, 1.0))
        )
    ))
    if information < max(0.0, float(profile.rotation_min_phase_information)):
        return None

    distances = np.asarray(offsets, dtype=np.float64)[hits]
    band_sd = max(0.25, float(profile.rotation_band_sd))
    bands = []
    for period, phase, _log_weight in grid:
        phase_distance = np.abs(
            np.mod(distances - phase + 0.5 * period, period)
            - 0.5 * period
        )
        bands.append(np.exp(-0.5 * np.square(phase_distance / band_sd)))
    expected_band = posterior @ np.asarray(bands, dtype=np.float64)
    return float(np.max(-np.log(np.clip(expected_band, 1e-6, 1.0))))


def _smoothed_deam_rate(
    opp,
    deam,
    win=21,
    min_opps=3,
    sparse_win=41,
    prior_mean=0.05,
    prior_weight=1.0,
):
    """Opportunity-supported local DddA deamination rate.

    A 21-bp window averages roughly two DNA turns, but the old estimator also
    accepted a denominator of one.  One rotationally exposed C/G then produced
    a 21-bp plateau at rate 1.0 and a false linker edge, creating the observed
    ~10-bp comb of short nucleosomes and TF-like residue.  Require three actual
    opportunities and use one protected-state pseudo-observation (mean 0.05).
    Thus 2/3 hits remain below the shipped 0.603 edge threshold while 3/4 hits
    cross it.  Where sequence is sparse, a 41-bp fallback collects the same
    minimum evidence at honestly lower spatial resolution; positions
    unsupported even there remain NaN.
    """
    opportunity = np.asarray(opp, dtype=bool)
    observed = np.asarray(deam, dtype=bool)
    required = max(1, int(min_opps))
    pseudo_weight = max(0.0, float(prior_weight))
    pseudo_hits = pseudo_weight * float(np.clip(prior_mean, 0.0, 1.0))

    def estimate(width):
        width = max(1, int(width))
        kernel = np.ones(width, dtype=np.float64)
        numerator = np.convolve(observed.astype(np.float64), kernel, 'same')
        denominator = np.convolve(opportunity.astype(np.float64), kernel, 'same')
        rate = np.full(denominator.shape, np.nan, dtype=np.float64)
        supported = denominator >= required
        rate[supported] = (
            numerator[supported] + pseudo_hits
        ) / (denominator[supported] + pseudo_weight)
        return rate, supported

    rate, supported = estimate(win)
    if sparse_win is not None and int(sparse_win) > int(win):
        sparse_rate, sparse_supported = estimate(sparse_win)
        fallback = ~supported & sparse_supported
        rate[fallback] = sparse_rate[fallback]
    return rate


def _profile_weights(profile: NucProfile):
    """Signed per-offset log-weights (index = offset + half) for the dyad LLR."""
    half = profile.half
    off = np.arange(-half, half + 1)
    r = np.array([profile.radial[min(abs(d), len(profile.radial) - 1)] for d in off])
    t = np.clip(np.nan_to_num(r, nan=0.05), 0.01, 0.6)
    w1 = np.log(t / profile.linker)
    w0 = np.log((1 - t) / (1 - profile.linker))
    return w1, w0


def _dyad_llr_full(opp, deam, w1, w0):
    """Per-position LLR that a nucleosome is centered there, for the whole read.

    LLR(c) = sum_d a[c+d]*w1[d+half] + b[c+d]*w0[d+half] over the +-half window,
    where a = deaminated opportunities, b = protected opportunities. That is a
    cross-correlation of (a, b) with the (w1, w0) template -- vectorized with
    np.correlate (zero-padded edges == the clipped window), ~100x faster than the
    per-candidate Python loop and numerically identical."""
    a = (opp & deam).astype(np.float64)
    b = (opp & ~deam).astype(np.float64)
    return np.correlate(a, w1, 'same') + np.correlate(b, w0, 'same')


def _dyad_llr_track(opp, deam, lo, hi, w1, w0, half):
    """Convenience wrapper: LLR track restricted to ``[lo, hi)``."""
    llr = _dyad_llr_full(opp, deam, w1, w0)
    return np.arange(lo, hi), llr[lo:hi]


def _place_dyads(cs, llr, min_sep):
    """Greedy peak picking: highest positive LLR first, min separation."""
    chosen: List[int] = []
    for idx in np.argsort(llr)[::-1]:
        if llr[idx] <= 0:
            break
        c = int(cs[idx])
        if all(abs(c - cc) >= min_sep for cc in chosen):
            chosen.append(c)
    return sorted(chosen)


def _find_density_edge(
    sr,
    c,
    direction,
    bound,
    profile: NucProfile,
    fallback=None,
    opportunity=None,
    deaminated=None,
    observations=None,
    llr_hit=None,
    llr_miss=None,
    phase_condition_sum=None,
    phase_core_radius=0,
    transition_window=31,
    transition_min_opps=5,
    transition_min_rate=0.45,
):
    """Resolve one DddA particle edge from molecule-local phase evidence.

    Production DddA profiles use one context-aware, phase-marginal posterior
    for every molecule. A rotationally exposed hit can therefore remain inside
    the same particle, missed turns are tolerated, and 9--12-bp local pitch
    variation is integrated rather than converted into a hard boundary. The
    posterior median is always returned; only edge quality changes when its
    support is broad or censored. There is no confidence-selected switch to an
    HMM, adjacent-dyad, or canonical coordinate.

    Historical/custom profiles that lack the rotational model or attached
    chemistry emissions retain the radial change-point/topology fallback.
    """
    L = profile.linker
    mid, lo_t, hi_t = profile.edge_frac * L, 0.30 * L, 0.60 * L
    x = edge = c
    found = False

    posterior_edge = _rotational_edge_posterior(
        profile,
        observations,
        opportunity,
        deaminated,
        c,
        direction,
        bound,
        llr_hit,
        llr_miss,
        phase_condition_sum=phase_condition_sum,
        phase_core_radius=phase_core_radius,
    )
    if posterior_edge is not None:
        return posterior_edge

    phase_evaluated = False
    if (
        opportunity is not None
        and deaminated is not None
        and observations is not None
    ):
        opp = np.asarray(opportunity, dtype=bool)
        hit = np.asarray(deaminated, dtype=bool)
        max_distance = min(
            abs(int(bound) - int(c)),
            int(c) if int(direction) < 0 else len(opp) - 1 - int(c),
        )
        if max_distance > 0:
            distances = np.arange(1, max_distance + 1, dtype=np.int64)
            positions = int(c) + int(direction) * distances
            informative = opp[positions]
            observed = hit[positions] & informative
            rotational = _rotational_wrapped_llr_matrix(
                observations,
                positions,
                distances,
                profile,
                llr_hit,
                llr_miss,
            )
            if rotational is not None:
                phase_evaluated = True
                wrapped_llr, latent_log_weights = rotational
                prefix = np.cumsum(wrapped_llr, axis=1)
                prior_log = np.asarray(latent_log_weights, dtype=np.float64)
                prior_log -= float(np.max(prior_log))
                prior = np.exp(prior_log)
                prior /= float(np.sum(prior))
                grid = _rotation_grid(profile)
                band_sd = max(0.25, float(profile.rotation_band_sd))
                band_matrix = np.asarray([
                    np.exp(-0.5 * np.square(
                        np.abs(
                            np.mod(
                                distances - phase + 0.5 * period,
                                period,
                            ) - 0.5 * period
                        ) / band_sd
                    ))
                    for period, phase, _log_weight in grid
                ], dtype=np.float64)
                break_threshold = max(
                    0.0,
                    float(profile.rotation_edge_break_min_surprisal),
                )
                information_threshold = max(
                    0.0,
                    float(profile.rotation_min_phase_information),
                )
                opportunity_threshold = max(1, int(transition_min_opps))
                local_width = max(1, int(transition_window))
                hit_indices = np.flatnonzero(observed)
                miss_indices = np.flatnonzero(informative & ~observed)

                for hit_index in hit_indices:
                    hit_index = int(hit_index)
                    # The proposed breaking event must not teach the phase that
                    # it is being tested against.
                    if hit_index <= 0:
                        continue
                    if phase_condition_sum is None:
                        condition_sum = prefix[:, hit_index - 1]
                    else:
                        condition_sum = np.asarray(
                            phase_condition_sum, dtype=np.float64,
                        ).copy()
                        # The shared dyad core already includes observations on
                        # both sides. Add only this flank's evidence beyond it.
                        extra_start = max(0, int(phase_core_radius))
                        if hit_index > extra_start:
                            condition_sum += np.sum(
                                wrapped_llr[:, extra_start:hit_index], axis=1,
                            )
                    posterior_log = prior_log + condition_sum
                    posterior_log -= float(np.max(posterior_log))
                    posterior = np.exp(posterior_log)
                    posterior /= float(np.sum(posterior))
                    information = float(np.sum(
                        posterior * (
                            np.log(np.clip(posterior, 1e-300, 1.0))
                            - np.log(np.clip(prior, 1e-300, 1.0))
                        )
                    ))
                    if information < information_threshold:
                        continue
                    expected_band = float(
                        posterior @ band_matrix[:, hit_index]
                    )
                    surprisal = -np.log(np.clip(expected_band, 1e-6, 1.0))
                    if surprisal < break_threshold:
                        continue

                    # Test only the local outward tract.  A remote protected
                    # block or neighboring particle must not move this edge.
                    local_stop = min(max_distance, hit_index + local_width)
                    local_opportunities = np.flatnonzero(
                        informative[hit_index:min(
                            max_distance, hit_index + 2 * local_width,
                        )]
                    )
                    if local_opportunities.size >= opportunity_threshold:
                        local_stop = max(
                            local_stop,
                            hit_index
                            + int(local_opportunities[opportunity_threshold - 1])
                            + 1,
                        )
                    local_indices = np.arange(
                        hit_index, local_stop, dtype=np.int64,
                    )
                    local_informative = informative[local_indices]
                    if int(np.count_nonzero(local_informative)) < opportunity_threshold:
                        continue
                    # The tract must also look accessible under the calibrated
                    # chemistry emissions.  Do not demand that every later hit
                    # refute a rotational model: once DNA is linker, hits at the
                    # old on-phase coordinates remain perfectly possible.
                    local_codes = np.asarray(observations)[
                        positions[local_indices]
                    ]
                    local_accessible_bf = 0.0
                    local_hits = (
                        (local_codes >= 0) & (local_codes < N_CTX)
                    )
                    local_misses = (
                        (local_codes >= UNMETH_OFFSET)
                        & (local_codes < UNMETH_OFFSET + N_CTX)
                    )
                    if np.any(local_hits):
                        local_accessible_bf -= float(np.sum(
                            np.asarray(llr_hit)[local_codes[local_hits]]
                        ))
                    if np.any(local_misses):
                        local_accessible_bf -= float(np.sum(
                            np.asarray(llr_miss)[
                                local_codes[local_misses] - UNMETH_OFFSET
                            ]
                        ))
                    if local_accessible_bf < 2.0:
                        continue

                    inward_misses = miss_indices[miss_indices < hit_index]
                    if inward_misses.size == 0:
                        continue
                    inner_index = int(inward_misses[-1])
                    inner_distance = int(distances[inner_index])
                    outer_distance = int(distances[hit_index])
                    best_offset = int(round(
                        0.5 * (inner_distance + outer_distance)
                    ))
                    edge = int(c) + int(direction) * best_offset
                    return edge, max(0, outer_distance - inner_distance)

                # A direct off-phase breaking event is the cleanest edge, but
                # it is often dropped out on one DddA strand.  In that case,
                # test every change point outside the shared protected core by
                # marginalizing the complete outward tract over the learned
                # period/phase grid.  The score is the Bayes factor for linker
                # rather than continued wrapping.  This recovers boundaries
                # supported by several individually weak events without
                # treating a single on-phase rotational hit as a linker.
                core_radius = min(
                    max_distance,
                    max(1, int(phase_core_radius)),
                )
                if phase_condition_sum is None:
                    condition_sum = np.sum(
                        wrapped_llr[:, :core_radius], axis=1,
                    )
                else:
                    condition_sum = np.asarray(
                        phase_condition_sum, dtype=np.float64,
                    )
                suffix_wrapped = np.cumsum(
                    wrapped_llr[:, ::-1], axis=1,
                )[:, ::-1]
                outside_wrapped = np.zeros_like(wrapped_llr)
                if max_distance > 1:
                    outside_wrapped[:, :-1] = suffix_wrapped[:, 1:]
                outside_opportunities = np.zeros(
                    max_distance, dtype=np.int64,
                )
                opportunity_suffix = np.cumsum(
                    informative[::-1], dtype=np.int64,
                )[::-1]
                if max_distance > 1:
                    outside_opportunities[:-1] = opportunity_suffix[1:]

                candidate_indices = np.arange(
                    core_radius - 1, max_distance, dtype=np.int64,
                )
                valid = candidate_indices[
                    outside_opportunities[candidate_indices]
                    >= opportunity_threshold
                ]
                if valid.size and break_threshold > 0.0:
                    # Aggregate evidence may sharpen a sparse change point,
                    # but it may not manufacture one from a train of purely
                    # on-phase rotational hits. At least the first observed
                    # hit outside the candidate must break the learned phase.
                    candidate_surprisal = _direct_edge_break_surprisal_scores(
                        profile,
                        distances,
                        latent_log_weights,
                        prefix,
                        valid,
                        observed,
                        max_hits=1,
                    )
                    valid = valid[candidate_surprisal >= break_threshold]
                if valid.size:
                    conditioned = float(_logmeanexp(
                        condition_sum[:, None],
                        latent_log_weights,
                        axis=0,
                    )[0])
                    continued = _logmeanexp(
                        condition_sum[:, None]
                        + outside_wrapped[:, valid],
                        latent_log_weights,
                        axis=0,
                    )
                    linker_bf = conditioned - continued
                    best_bf = float(np.max(linker_bf))
                    if best_bf >= 2.0:
                        map_indices = valid[np.isclose(
                            linker_bf,
                            best_bf,
                            rtol=0.0,
                            atol=1e-12,
                        )]
                        near = valid[linker_bf >= best_bf - 1.92]
                        best_offset = int(round(float(np.mean(
                            distances[map_indices]
                        ))))
                        ambiguity = int(
                            distances[near[-1]] - distances[near[0]]
                        )
                        edge = int(c) + int(direction) * best_offset
                        return edge, max(0, ambiguity)

    if opportunity is not None and deaminated is not None and not phase_evaluated:
        opp = np.asarray(opportunity, dtype=bool)
        hit = np.asarray(deaminated, dtype=bool)
        max_distance = min(
            abs(int(bound) - int(c)),
            int(c) if int(direction) < 0 else len(opp) - 1 - int(c),
        )
        if max_distance > 0:
            distances = np.arange(1, max_distance + 1, dtype=np.int64)
            positions = int(c) + int(direction) * distances
            informative = opp[positions]
            observed = hit[positions] & informative
            radial = np.clip(
                np.nan_to_num(
                    np.asarray(profile.radial, dtype=np.float64), nan=0.05,
                ),
                0.01,
                0.60,
            )
            wrapped_rate = radial[np.minimum(distances, radial.size - 1)]
            linker_rate = float(np.clip(
                profile.linker, 1e-6, 1.0 - 1e-6,
            ))
            delta = np.zeros(max_distance, dtype=np.float64)
            hits = informative & observed
            misses = informative & ~observed
            delta[hits] = np.log(linker_rate / wrapped_rate[hits])
            delta[misses] = np.log(
                (1.0 - linker_rate) / (1.0 - wrapped_rate[misses])
            )

            # For an edge after outward offset d, sites 1..d are wrapped and
            # d+1..bound are linker. Relative to an all-wrapped tract, the
            # suffix sum is the evidence for each candidate boundary.
            suffix_linker_llr = np.cumsum(delta[::-1])[::-1]
            candidate_offsets = np.arange(1, max_distance + 1)
            outward_opps = np.cumsum(informative[::-1])[::-1]
            candidate_llr = np.zeros(max_distance, dtype=np.float64)
            candidate_outward_opps = np.zeros(max_distance, dtype=np.int64)
            if max_distance > 1:
                candidate_llr[:-1] = suffix_linker_llr[1:]
                candidate_outward_opps[:-1] = outward_opps[1:]
            valid = candidate_outward_opps >= max(
                1, int(transition_min_opps),
            )
            if np.any(valid):
                valid_indices = np.flatnonzero(valid)
                best_llr = float(np.max(candidate_llr[valid_indices]))
                map_indices = valid_indices[np.isclose(
                    candidate_llr[valid_indices],
                    best_llr,
                    rtol=0.0,
                    atol=1e-12,
                )]
                # Require explicit transition evidence. Otherwise the fallback
                # remains unresolved: the radial profile is a soft prior, not
                # a hard instruction to emit a canonical-sized particle.
                if best_llr >= 2.0:
                    near = valid_indices[
                        candidate_llr[valid_indices] >= best_llr - 1.92
                    ]
                    # There is no information between successive opportunity
                    # sites. Use the midpoint of an exactly tied MAP plateau
                    # rather than snapping every edge to its inward endpoint.
                    best_offset = int(round(float(np.mean(
                        candidate_offsets[map_indices]
                    ))))
                    edge = int(c) + int(direction) * best_offset
                    ambiguity = int(
                        candidate_offsets[near[-1]]
                        - candidate_offsets[near[0]]
                    )
                    return edge, max(0, ambiguity)
    elif opportunity is None or deaminated is None:
        while 0 <= x + direction < len(sr) and direction * (bound - x) > 0:
            x += direction
            if np.isfinite(sr[x]) and sr[x] >= mid:
                edge = x
                found = True
                break
        else:
            edge = x
    if not found:
        # No molecule-supported transition.  A shipped chemistry profile may
        # provide a calibrated radial point estimate for display/tiling.  Its
        # saturated ambiguity is essential: this is a prior-supported edge,
        # not a claim that the molecule identified an exact boundary. Custom
        # profiles without that prior retain the coarse HMM topology.
        topology = int(bound) if fallback is None else int(fallback)
        if direction > 0:
            topology = min(max(topology, int(c)), int(bound), len(sr))
        else:
            topology = max(min(topology, int(c)), int(bound), 0)
        prior_center = float(getattr(profile, "edge_prior_center", 0.0))
        if phase_evaluated and prior_center > 0.0:
            max_offset = abs(int(topology) - int(c))
            if max_offset > 0:
                candidate_offsets = distances[:max_offset].astype(
                    np.float64,
                )
                prior_sd = max(
                    1.0, float(getattr(profile, "edge_prior_sd", 0.0)),
                )
                log_prior = -0.5 * np.square(
                    (candidate_offsets - prior_center) / prior_sd
                )
                # The HMM edge remains a coarse, molecule-specific structural
                # observation rather than a hard boundary. Combining its broad
                # error model with the chemistry prior preserves real span
                # variation and avoids a pile-up at one canonical length.
                topology_sd = max(
                    1.0,
                    float(getattr(profile, "edge_topology_sd", 0.0)),
                )
                if float(getattr(profile, "edge_topology_sd", 0.0)) > 0.0:
                    log_prior -= 0.5 * np.square(
                        (candidate_offsets - float(max_offset)) / topology_sd
                    )

                # Relative to an all-linker flank, the prefix likelihood says
                # how well each candidate edge explains its inward sequence as
                # wrapped. Condition the latent phase on the opposite core,
                # while subtracting this flank's shared core to avoid counting
                # those observations twice. A calibrated temperature absorbs
                # residual within-turn correlation not represented by the
                # independent context emissions.
                core_radius = min(
                    max_offset,
                    max(1, int(phase_core_radius)),
                )
                if phase_condition_sum is None:
                    other_condition = np.zeros(
                        wrapped_llr.shape[0], dtype=np.float64,
                    )
                else:
                    current_core = np.sum(
                        wrapped_llr[:, :core_radius], axis=1,
                    )
                    other_condition = (
                        np.asarray(phase_condition_sum, dtype=np.float64)
                        - current_core
                    )
                baseline = float(_logmeanexp(
                    other_condition[:, None],
                    latent_log_weights,
                    axis=0,
                )[0])
                edge_likelihood = _logmeanexp(
                    other_condition[:, None] + prefix[:, :max_offset],
                    latent_log_weights,
                    axis=0,
                ) - baseline
                temperature = max(
                    1.0,
                    float(getattr(
                        profile, "edge_likelihood_temperature", 1.0,
                    )),
                )
                log_posterior = log_prior + edge_likelihood / temperature
                log_posterior -= float(np.max(log_posterior))
                posterior = np.exp(log_posterior)
                posterior /= float(np.sum(posterior))
                posterior_modes = candidate_offsets[np.isclose(
                    log_posterior,
                    float(np.max(log_posterior)),
                    rtol=0.0,
                    atol=1e-12,
                )]
                prior_offset = int(round(float(np.mean(posterior_modes))))
                cumulative = np.cumsum(posterior)
                low = int(candidate_offsets[min(
                    len(candidate_offsets) - 1,
                    int(np.searchsorted(cumulative, 0.025)),
                )])
                high = int(candidate_offsets[min(
                    len(candidate_offsets) - 1,
                    int(np.searchsorted(cumulative, 0.975)),
                )])
                topology = int(c) + int(direction) * prior_offset
                return topology, max(30, high - low)
        return topology, max(30, abs(topology - c))
    lo = hi = edge
    while lo - direction >= 0 and direction * (lo - c) > 0 and \
            np.isfinite(sr[lo]) and sr[lo] > lo_t:
        lo -= direction
    while 0 <= hi + direction < len(sr) and np.isfinite(sr[hi]) and sr[hi] < hi_t:
        hi += direction
    return edge, abs(hi - lo)


def _radial_split_footprint(
    sr, s, e, profile, nuc_min_size, llr_full, opportunity=None,
    deaminated=None, observations=None, llr_hit=None, llr_miss=None,
    search_start=None, search_end=None,
):
    """Place dyads in protected footprint [s,e) and return (NucCalls, access)."""
    half = profile.half
    lo, hi = max(s + 20, 0), e - 20
    cs = np.arange(lo, hi)
    llr = llr_full[lo:hi]
    if len(cs) == 0:
        return [], [(s, e - s)]
    dyads = _place_dyads(cs, llr, profile.min_sep)
    if not dyads:
        return [], [(s, e - s)]
    nucs: List[NucCall] = []
    covered: List[Interval] = []
    for i, c in enumerate(dyads):
        lb = (
            (dyads[i - 1] + c) // 2
            if i > 0 else max(
                int(search_start) if search_start is not None else 0,
                c - (half + 37),
            )
        )
        rb = (
            (dyads[i + 1] + c) // 2
            if i + 1 < len(dyads) else min(
                int(search_end) if search_end is not None else len(sr) - 1,
                c + (half + 37),
            )
        )
        phase_condition_sum = None
        phase_core_radius = min(
            max(1, int(nuc_min_size) // 2),
            max(1, int(profile.half)),
            max(1, min(int(c), int(c) - max(0, int(lb)))),
            max(1, min(
                len(sr) - 1 - int(c),
                min(len(sr) - 1, int(rb)) - int(c),
            )),
        )
        if observations is not None and phase_core_radius > 0:
            core_distances = np.concatenate((
                np.arange(
                    phase_core_radius, 0, -1, dtype=np.int64,
                ),
                np.arange(
                    1, phase_core_radius + 1, dtype=np.int64,
                ),
            ))
            core_positions = np.concatenate((
                int(c) - np.arange(
                    phase_core_radius, 0, -1, dtype=np.int64,
                ),
                int(c) + np.arange(
                    1, phase_core_radius + 1, dtype=np.int64,
                ),
            ))
            core_rotational = _rotational_wrapped_llr_matrix(
                observations,
                core_positions,
                core_distances,
                profile,
                llr_hit,
                llr_miss,
            )
            if core_rotational is not None:
                core_llr, _core_weights = core_rotational
                phase_condition_sum = np.sum(core_llr, axis=1)
        eL, ambL = _find_density_edge(
            sr,
            c,
            -1,
            max(0, lb),
            profile,
            fallback=(dyads[i - 1] + c) // 2 if i > 0 else s,
            opportunity=opportunity,
            deaminated=deaminated,
            observations=observations,
            llr_hit=llr_hit,
            llr_miss=llr_miss,
            phase_condition_sum=phase_condition_sum,
            phase_core_radius=phase_core_radius,
        )
        eR, ambR = _find_density_edge(
            sr,
            c,
            +1,
            min(len(sr), rb),
            profile,
            fallback=(dyads[i + 1] + c) // 2 if i + 1 < len(dyads) else e,
            opportunity=opportunity,
            deaminated=deaminated,
            observations=observations,
            llr_hit=llr_hit,
            llr_miss=llr_miss,
            phase_condition_sum=phase_condition_sum,
            phase_core_radius=phase_core_radius,
        )
        if eR - eL < nuc_min_size:
            continue                       # sub-floor -> stays accessible
        peak = float(llr[int(np.argmin(np.abs(cs - c)))])
        nucs.append(NucCall(
            eL,
            eR - eL,
            llr_to_tq(max(0.0, peak)),
            ambiguity_to_edge(ambL),
            ambiguity_to_edge(ambR),
            dyad=int(c),
            radial_start=int(eL),
            radial_end=int(eR),
            phase_resolved_left=(
                float(profile.rotation_period) > 0.0 and int(ambL) < 30
            ),
            phase_resolved_right=(
                float(profile.rotation_period) > 0.0 and int(ambR) < 30
            ),
        ))
        covered.append((eL, eR))
    # accessible = footprint minus the emitted nucleosomes
    access: List[Interval] = []
    cur = s
    for a, b in sorted(covered):
        if a > cur:
            access.append((cur, a - cur))
        cur = max(cur, b)
    if e > cur:
        access.append((cur, e - cur))
    return nucs, access


def radial_split_in_read(
    obs, ns, nl, read_length, profile, nuc_min_size,
    llr_hit=None, llr_miss=None,
):
    """DddA nucleosome recall: match-filter each HMM footprint into nucleosomes
    + accessible residue. Same return contract as ``recall_nucs_in_read``."""
    opp, deam = _obs_opp_deam(obs)
    sr = _smoothed_deam_rate(opp, deam)
    w1, w0 = _profile_weights(profile)
    llr_full = _dyad_llr_full(opp, deam, w1, w0)   # once per read (vectorized)
    nucs: List[NucCall] = []
    access: List[Interval] = []
    footprints = []
    for s_raw, length_raw in zip(ns, nl):
        s = max(0, int(s_raw))
        length = int(length_raw)
        e = min(s + length, read_length)
        if e > s:
            footprints.append((s, e))
    footprints.sort()
    for footprint_index, (s, e) in enumerate(footprints):
        length = e - s
        if length <= 0:
            continue
        if e - s < nuc_min_size:
            access.append((s, e - s))      # too short to be a nucleosome
            continue
        search_start = (
            footprints[footprint_index - 1][1]
            if footprint_index > 0 else 0
        )
        search_end = (
            footprints[footprint_index + 1][0]
            if footprint_index + 1 < len(footprints) else read_length - 1
        )
        fn, fa = _radial_split_footprint(
            sr, s, e, profile, nuc_min_size, llr_full,
            opportunity=opp, deaminated=deam, observations=obs,
            llr_hit=llr_hit, llr_miss=llr_miss,
            search_start=search_start, search_end=search_end,
        )
        nucs.extend(fn)
        access.extend(fa)
    return nucs, access


def _subtract_calls_from_interval(start, end, calls):
    """Return portions of ``[start,end)`` not occupied by provisional calls."""
    covered = sorted(
        (
            max(int(start), int(call.start)),
            min(int(end), int(call.start) + int(call.length)),
        )
        for call in calls
        if int(call.start) < int(end)
        and int(call.start) + int(call.length) > int(start)
    )
    residue: List[Tuple[int, int]] = []
    cursor = int(start)
    for left, right in covered:
        if left > cursor:
            residue.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < int(end):
        residue.append((cursor, int(end)))
    return residue


def _accessible_configuration_evidence(
    obs,
    intervals,
    llr_hit,
    llr_miss,
    m5c_mask=None,
    m5c_llr_hit=None,
    m5c_llr_miss=None,
):
    """Return ``(opportunities, log BF)`` for accessible vs protected.

    The TF tables encode log P(protected) / P(accessible), so their negative is
    the desired evidence. Intervals supplied here are the residue left after
    excluding any explicitly supplied protected configuration. The production
    baseline supplies no provisional TF calls at this stage.
    """
    observations = np.asarray(obs)
    hit_table = np.asarray(llr_hit)
    miss_table = np.asarray(llr_miss)
    use_m5c = (
        m5c_mask is not None
        and m5c_llr_hit is not None
        and m5c_llr_miss is not None
    )
    methylated = np.asarray(m5c_mask, dtype=bool) if use_m5c else None
    methylated_hit = np.asarray(m5c_llr_hit) if use_m5c else None
    methylated_miss = np.asarray(m5c_llr_miss) if use_m5c else None
    opportunities = 0
    log_bf = 0.0
    for start, end in intervals:
        lo = max(0, int(start))
        hi = min(len(observations), int(end))
        for position in range(lo, hi):
            code = int(observations[position])
            if 0 <= code < N_CTX:
                table = methylated_hit if use_m5c and methylated[position] else hit_table
                log_bf -= float(table[code])
                opportunities += 1
            elif UNMETH_OFFSET <= code < UNMETH_OFFSET + N_CTX:
                context = code - UNMETH_OFFSET
                table = methylated_miss if use_m5c and methylated[position] else miss_table
                log_bf -= float(table[context])
                opportunities += 1
    return opportunities, log_bf


def _radial_extension_evidence(
    obs,
    start,
    end,
    dyad,
    profile,
    tf_calls,
    llr_hit,
    llr_miss,
    m5c_mask=None,
    m5c_llr_hit=None,
    m5c_llr_miss=None,
    condition_start=None,
    condition_end=None,
):
    """Score a proposed DddA nucleosome flank against its current tiling.

    Both scores are likelihood ratios against accessible/linker sequence.  The
    radial score uses the empirical offset-from-dyad deamination probability;
    when a conditioning core is supplied, the latent rotational register is
    inferred from that core before the flank is scored. Thus a skipped/jittered
    continuation is plausible, while an off-phase hit directly disfavors
    absorbing the flank into the same particle.
    the competing score uses the context-aware TF-protection LLR only where a
    provisional TF currently occupies the flank (unoccupied residue is the
    accessible baseline and contributes zero).  Comparing the two therefore
    asks the molecular question directly: would this exact pattern be better
    generated by the same nucleosome, or by the current linker/TF hypothesis?
    """
    observations = np.asarray(obs)
    lo = max(0, int(start))
    hi = min(len(observations), int(end))
    if hi <= lo:
        return 0, 0.0, 0.0, 0, None

    positions = np.arange(lo, hi, dtype=np.int64)
    codes = observations[lo:hi]
    hit = (codes >= 0) & (codes < N_CTX)
    miss = (
        (codes >= UNMETH_OFFSET)
        & (codes < UNMETH_OFFSET + N_CTX)
    )
    informative = hit | miss
    opportunities = int(np.count_nonzero(informative))
    if opportunities == 0:
        return 0, 0.0, 0.0, 0, None

    offsets = np.abs(positions - int(dyad))
    rotational = _rotational_wrapped_llr_matrix(
        observations,
        positions,
        offsets,
        profile,
        llr_hit,
        llr_miss,
        m5c_mask=m5c_mask,
        m5c_llr_hit=m5c_llr_hit,
        m5c_llr_miss=m5c_llr_miss,
    )
    if rotational is not None:
        wrapped_llr, latent_log_weights = rotational
        extension_sum = np.sum(wrapped_llr, axis=1)
        condition_lo = max(
            0, int(condition_start) if condition_start is not None else 0,
        )
        condition_hi = min(
            len(observations),
            int(condition_end) if condition_end is not None else 0,
        )
        if condition_hi > condition_lo:
            condition_positions = np.arange(
                condition_lo, condition_hi, dtype=np.int64,
            )
            # The extension and conditioning core are normally disjoint. Keep
            # that invariant explicit for callers that pass a broad core.
            condition_positions = condition_positions[
                (condition_positions < lo) | (condition_positions >= hi)
            ]
        else:
            condition_positions = np.asarray([], dtype=np.int64)
        if condition_positions.size:
            condition_rotational = _rotational_wrapped_llr_matrix(
                observations,
                condition_positions,
                np.abs(condition_positions - int(dyad)),
                profile,
                llr_hit,
                llr_miss,
                m5c_mask=m5c_mask,
                m5c_llr_hit=m5c_llr_hit,
                m5c_llr_miss=m5c_llr_miss,
            )
        else:
            condition_rotational = None
        phase_break_surprisal = None
        if condition_rotational is not None:
            condition_llr, condition_weights = condition_rotational
            if not np.allclose(condition_weights, latent_log_weights):
                raise ValueError("rotation grids differ between core and flank")
            condition_sum = np.sum(condition_llr, axis=1)
            joint = float(_logmeanexp(
                (condition_sum + extension_sum)[:, None],
                latent_log_weights,
                axis=0,
            )[0])
            core = float(_logmeanexp(
                condition_sum[:, None], latent_log_weights, axis=0,
            )[0])
            radial_llr = joint - core
            phase_break_surprisal = _conditional_phase_break_surprisal(
                profile,
                offsets,
                hit,
                latent_log_weights,
                condition_sum,
            )
        else:
            radial_llr = float(_logmeanexp(
                extension_sum[:, None], latent_log_weights, axis=0,
            )[0])
    else:
        radial = np.clip(
            np.nan_to_num(
                np.asarray(profile.radial, dtype=np.float64), nan=0.05,
            ),
            0.01,
            0.60,
        )
        rates = radial[np.minimum(offsets, radial.size - 1)]
        linker = float(np.clip(profile.linker, 1e-6, 1.0 - 1e-6))
        radial_llr = 0.0
        if np.any(hit):
            radial_llr += float(np.sum(np.log(rates[hit] / linker)))
        if np.any(miss):
            radial_llr += float(
                np.sum(np.log((1.0 - rates[miss]) / (1.0 - linker)))
            )

    occupied = np.zeros(hi - lo, dtype=bool)
    for call in tf_calls:
        call_start = max(lo, int(call.start))
        call_end = min(hi, int(call.start) + int(call.length))
        if call_end > call_start:
            occupied[call_start - lo:call_end - lo] = True

    tf_llr = 0.0
    use_m5c = (
        m5c_mask is not None
        and m5c_llr_hit is not None
        and m5c_llr_miss is not None
    )
    methylated = np.asarray(m5c_mask, dtype=bool) if use_m5c else None
    for index in np.flatnonzero(informative & occupied):
        code = int(codes[index])
        absolute = lo + int(index)
        use_methylated = bool(use_m5c and methylated[absolute])
        if code < N_CTX:
            table = m5c_llr_hit if use_methylated else llr_hit
            tf_llr += float(table[code])
        else:
            table = m5c_llr_miss if use_methylated else llr_miss
            tf_llr += float(table[code - UNMETH_OFFSET])

    return (
        opportunities,
        radial_llr,
        tf_llr,
        int(np.count_nonzero(hit)),
        phase_break_surprisal if rotational is not None else None,
    )


def _has_accessible_separator(
    obs,
    extension_start,
    extension_end,
    hmm_edge,
    side,
    tf_calls,
    llr_hit,
    llr_miss,
    m5c_mask=None,
    m5c_llr_hit=None,
    m5c_llr_miss=None,
):
    """Whether direct linker evidence separates an HMM edge from a flank TF."""
    overlapping = [
        call for call in tf_calls
        if int(call.start) < int(extension_end)
        and int(call.start) + int(call.length) > int(extension_start)
    ]
    for call in overlapping:
        if side == "left":
            gap_start = max(
                int(extension_start), int(call.start) + int(call.length)
            )
            gap_end = int(hmm_edge)
        else:
            gap_start = int(hmm_edge)
            gap_end = min(int(extension_end), int(call.start))
        if gap_end <= gap_start:
            continue
        # Intermediate TF calls are protected alternatives, not linker
        # evidence. Subtract them before evaluating the residue; otherwise
        # their misses can cancel a real accessible separator and allow a
        # distant TF to be swallowed by the nucleosome proposal.
        residue = _subtract_calls_from_interval(
            gap_start, gap_end, overlapping,
        )
        opportunities, log_bf = _accessible_configuration_evidence(
            obs,
            residue,
            llr_hit,
            llr_miss,
            m5c_mask=m5c_mask,
            m5c_llr_hit=m5c_llr_hit,
            m5c_llr_miss=m5c_llr_miss,
        )
        if opportunities >= 2 and log_bf >= 2.0:
            return True
    return False


def _complete_radial_nuc_from_adjacent_tf(
    call,
    obs,
    tf_calls,
    profile,
    left_limit,
    right_limit,
    nuc_min_size,
    llr_hit,
    llr_miss,
    m5c_mask=None,
    m5c_llr_hit=None,
    m5c_llr_miss=None,
):
    """Test whether an adjacent TF is an ambiguous tail of one radial nuc.

    This runs after HMM-edge restoration, when a zero edge byte identifies the
    unresolved side that may have been cut by a rotational DddA event.  A
    a one-nucleosome completion is preferred only when it is within two nats of
    the explicit nuc+linker/TF configuration and no supported linker separates
    the components. Candidate boundaries are supplied by every TF edge in the
    radial proposal, so a train of rotational fragments is tested jointly; no
    minimum or canonical nucleosome span is required.
    """
    dyad = getattr(call, "dyad", None)
    if dyad is None or int(call.length) >= 2 * int(profile.half):
        return call

    start = int(call.start)
    end = start + int(call.length)
    left_edge = int(call.el)
    right_edge = int(call.er)
    def acceptable(extension_start, extension_end, hmm_edge, side):
        opportunities, radial_llr, current_llr, _hits, phase_break = (
            _radial_extension_evidence(
                obs,
                extension_start,
                extension_end,
                int(dyad),
                profile,
                tf_calls,
                llr_hit,
                llr_miss,
                m5c_mask=m5c_mask,
                m5c_llr_hit=m5c_llr_hit,
                m5c_llr_miss=m5c_llr_miss,
                condition_start=start,
                condition_end=end,
            )
        )
        separated = _has_accessible_separator(
            obs,
            extension_start,
            extension_end,
            hmm_edge,
            side,
            tf_calls,
            llr_hit,
            llr_miss,
            m5c_mask=m5c_mask,
            m5c_llr_hit=m5c_llr_hit,
            m5c_llr_miss=m5c_llr_miss,
        )
        margin = radial_llr - current_llr
        phase_continuation = (
            phase_break is not None
            and phase_break < max(
                0.0, float(profile.rotation_edge_break_min_surprisal),
            )
        )
        no_observed_phase_break = (
            phase_break is None or phase_continuation
        )
        return (
            not separated
            and opportunities >= 2
            and margin >= -2.0
            and no_observed_phase_break
            and (radial_llr >= 2.0 or phase_continuation)
        ), margin

    if left_edge == 0:
        candidates = []
        for tf_call in tf_calls:
            tf_start = int(tf_call.start)
            tf_end = tf_start + int(tf_call.length)
            candidate = max(int(left_limit), tf_start)
            if (
                tf_start < start
                and tf_end > int(left_limit)
                and candidate < start
            ):
                candidates.append(candidate)
        accepted = []
        for candidate in sorted(set(candidates)):
            keep, margin = acceptable(candidate, start, start, "left")
            if keep:
                accepted.append((margin, -(start - candidate), candidate))
        if accepted:
            _margin, _negative_extension, start = max(accepted)
            left_edge = 0

    if right_edge == 0:
        candidates = []
        for tf_call in tf_calls:
            tf_start = int(tf_call.start)
            tf_end = tf_start + int(tf_call.length)
            candidate = min(int(right_limit), tf_end)
            if (
                tf_end > end
                and tf_start < int(right_limit)
                and candidate > end
            ):
                candidates.append(candidate)
        accepted = []
        for candidate in sorted(set(candidates)):
            keep, margin = acceptable(end, candidate, end, "right")
            if keep:
                accepted.append((margin, -(candidate - end), candidate))
        if accepted:
            _margin, _negative_extension, end = max(accepted)
            right_edge = 0

    return NucCall(
        start,
        end - start,
        int(call.nq),
        left_edge,
        right_edge,
        dyad=int(dyad),
        radial_start=getattr(call, "radial_start", None),
        radial_end=getattr(call, "radial_end", None),
        phase_resolved_left=bool(getattr(call, "phase_resolved_left", False)),
        phase_resolved_right=bool(getattr(call, "phase_resolved_right", False)),
    )


def validate_radial_access_in_read(
    obs,
    original_ns,
    original_nl,
    radial_nucs,
    provisional_tf_calls,
    read_length,
    llr_hit,
    llr_miss,
    *,
    min_llr,
    min_opps,
    nuc_min_size,
    nuc_profile=None,
    m5c_mask=None,
    m5c_llr_hit=None,
    m5c_llr_miss=None,
):
    """Validate DddA radial gaps as complete nuc/linker/TF configurations.

    A radial dyad match establishes protected occupancy; absence of radial
    coverage does *not* establish accessibility.  For every part of an
    original HMM nucleosome demoted by the radial pass, exclude any explicitly
    supplied protected configuration and require the remaining observations to
    support accessible rather than protected sequence. The production baseline
    supplies no provisional TF calls. Unsupported outer flanks are withheld from TF
    scanning but are not annexed back onto the radial particle: its Q0 edge
    records that the intervening state is unresolved. When two supported radial
    dyads lack linker evidence, they stay
    as two particles at their phase-supported core edges, with the facing edges
    marked Q0. The uncertain residue is not admitted to baseline TF scanning,
    but two dyads are not misreported as one giant nucleosome or inflated to the
    nucleosome-repeat length. An HMM footprint with no radial dyad is preserved
    whole. Restored or unresolved edges receive unresolved quality where the
    molecule did not identify exact topology. When the
    radial profile is supplied, a raw posterior edge that crosses an HMM
    boundary is tested against the HMM-accessible configuration. This test is
    identical at every posterior width: edge Q never selects the coordinate.
    A crossing candidate stays within the dyad search envelope, cannot cross a
    neighboring dyad or HMM nucleosome, needs at least two informative sequence
    opportunities and two nats of radial evidence, and must beat the current
    context-aware linker/TF configuration. Direct accessible evidence between
    the HMM edge and an abutting TF vetoes rescue.

    Returns ``(validated_nucleosomes, supported_accessible_intervals)``.  The
    caller should rebuild scan space and run TF recall only over directly
    supported accessible intervals.  Unsupported residue remains explicitly
    edge-ambiguous rather than being assigned to either neighboring particle.
    """
    nucs = [
        NucCall(int(call.start), int(call.length), int(call.nq),
                int(call.el), int(call.er),
                dyad=(
                    int(call.dyad)
                    if getattr(call, "dyad", None) is not None else None
                ),
                radial_start=(
                    int(call.radial_start)
                    if getattr(call, "radial_start", None) is not None else None
                ),
                radial_end=(
                    int(call.radial_end)
                    if getattr(call, "radial_end", None) is not None else None
                ),
                phase_resolved_left=bool(getattr(
                    call, "phase_resolved_left", False,
                )),
                phase_resolved_right=bool(getattr(
                    call, "phase_resolved_right", False,
                )))
        for call in radial_nucs
        if int(call.length) > 0
    ]
    tf_calls = list(provisional_tf_calls)
    originals = []
    for start_raw, length_raw in zip(original_ns, original_nl):
        start = max(0, int(start_raw))
        end = min(int(read_length), int(start_raw) + int(length_raw))
        if end > start:
            originals.append((start, end))

    assigned = [[] for _ in originals]
    unassigned = []
    for index, call in enumerate(nucs):
        center = int(call.start) + int(call.length) // 2
        owner = next(
            (i for i, (start, end) in enumerate(originals)
             if start <= center < end),
            None,
        )
        if owner is None:
            unassigned.append(call)
        else:
            assigned[owner].append(call)

    validated: List[NucCall] = list(unassigned)
    supported_access: List[Interval] = []
    score_threshold = float(min_llr)
    opportunity_threshold = max(1, int(min_opps))
    radial_dyads = sorted({
        int(call.dyad)
        for call in nucs
        if getattr(call, "dyad", None) is not None
    })

    def supports_access(start, end):
        if end <= start:
            return False, 0
        residue = _subtract_calls_from_interval(start, end, tf_calls)
        opportunities, log_bf = _accessible_configuration_evidence(
            obs,
            residue,
            llr_hit,
            llr_miss,
            m5c_mask=m5c_mask,
            m5c_llr_hit=m5c_llr_hit,
            m5c_llr_miss=m5c_llr_miss,
        )
        supported = (
            opportunities >= opportunity_threshold
            and log_bf >= score_threshold
        )
        return supported, (llr_to_tq(max(0.0, log_bf)) if supported else 0)

    for original_index, ((original_start, original_end), local) in enumerate(
        zip(originals, assigned)
    ):
        local.sort(key=lambda call: (int(call.start), int(call.length)))
        if original_end - original_start < int(nuc_min_size):
            # A sub-nucleosomal HMM footprint was already part of the ordinary
            # TF scan contract.  It is not a failed radial nucleosome and must
            # not be closed merely because no dyad can fit inside it.
            supported_access.append((
                original_start,
                original_end - original_start,
            ))
            continue
        if not local:
            validated.append(NucCall(
                original_start,
                original_end - original_start,
                nq=0,
                el=0,
                er=0,
            ))
            continue

        raw_starts = [int(call.start) for call in local]
        raw_ends = [int(call.start) + int(call.length) for call in local]
        # The HMM-accessible scan space is the default competing hypothesis.
        # A posterior edge may cross it only after the same molecule-local
        # likelihood comparison at every posterior width. This repairs DddA
        # HMM boundaries that terminate one turn early without letting an
        # el/er threshold choose between coordinate estimators.
        segments = []
        for call, raw_start, raw_end in zip(local, raw_starts, raw_ends):
            # Start from the HMM-clipped tiling for every edge. The raw
            # posterior median is tested below whenever it crosses that tiling;
            # phase_resolved_* is diagnostic only and cannot bypass the test.
            start = max(original_start, raw_start)
            end = min(original_end, raw_end)
            if end - start < int(nuc_min_size):
                continue
            left_edge = 0 if start != raw_start else int(call.el)
            right_edge = 0 if end != raw_end else int(call.er)
            dyad = (
                int(call.dyad)
                if getattr(call, "dyad", None) is not None else None
            )

            if (
                nuc_profile is not None
                and dyad is not None
                and (raw_start < start or raw_end > end)
            ):
                # Preserve the density model's actual (possibly asymmetric)
                # proposal. Neighboring dyads and HMM blocks bound the
                # particle; acceptance itself is likelihood-based, not a hard
                # nucleosome-size rule.
                proposed_left = getattr(call, "radial_start", None)
                proposed_right = getattr(call, "radial_end", None)
                left_limit = max(
                    0,
                    int(proposed_left)
                    if proposed_left is not None else raw_start,
                )
                right_limit = min(
                    int(read_length),
                    int(proposed_right)
                    if proposed_right is not None else raw_end,
                )
                if original_index > 0:
                    left_limit = max(left_limit, originals[original_index - 1][1])
                if original_index + 1 < len(originals):
                    right_limit = min(
                        right_limit, originals[original_index + 1][0]
                    )
                previous_dyads = [value for value in radial_dyads if value < dyad]
                next_dyads = [value for value in radial_dyads if value > dyad]
                if previous_dyads:
                    left_limit = max(
                        left_limit, (previous_dyads[-1] + dyad + 1) // 2
                    )
                if next_dyads:
                    right_limit = min(
                        right_limit, (dyad + next_dyads[0]) // 2
                    )

                candidate_left = max(raw_start, left_limit)
                if candidate_left < start:
                    opportunities, radial_llr, current_llr, _hits, _phase_break = (
                        _radial_extension_evidence(
                            obs,
                            candidate_left,
                            start,
                            dyad,
                            nuc_profile,
                            tf_calls,
                            llr_hit,
                            llr_miss,
                            m5c_mask=m5c_mask,
                            m5c_llr_hit=m5c_llr_hit,
                            m5c_llr_miss=m5c_llr_miss,
                            condition_start=start,
                            condition_end=end,
                        )
                    )
                    separated = _has_accessible_separator(
                        obs,
                        candidate_left,
                        start,
                        original_start,
                        "left",
                        tf_calls,
                        llr_hit,
                        llr_miss,
                        m5c_mask=m5c_mask,
                        m5c_llr_hit=m5c_llr_hit,
                        m5c_llr_miss=m5c_llr_miss,
                    )
                    if (
                        not separated
                        and opportunities >= 2
                        and radial_llr >= 2.0
                        and radial_llr >= current_llr
                    ):
                        start = candidate_left
                        left_edge = (
                            int(call.el) if candidate_left == raw_start else 0
                        )

                candidate_right = min(raw_end, right_limit)
                if candidate_right > end:
                    opportunities, radial_llr, current_llr, _hits, _phase_break = (
                        _radial_extension_evidence(
                            obs,
                            end,
                            candidate_right,
                            dyad,
                            nuc_profile,
                            tf_calls,
                            llr_hit,
                            llr_miss,
                            m5c_mask=m5c_mask,
                            m5c_llr_hit=m5c_llr_hit,
                            m5c_llr_miss=m5c_llr_miss,
                            condition_start=start,
                            condition_end=end,
                        )
                    )
                    separated = _has_accessible_separator(
                        obs,
                        end,
                        candidate_right,
                        original_end,
                        "right",
                        tf_calls,
                        llr_hit,
                        llr_miss,
                        m5c_mask=m5c_mask,
                        m5c_llr_hit=m5c_llr_hit,
                        m5c_llr_miss=m5c_llr_miss,
                    )
                    if (
                        not separated
                        and opportunities >= 2
                        and radial_llr >= 2.0
                        and radial_llr >= current_llr
                    ):
                        end = candidate_right
                        right_edge = (
                            int(call.er) if candidate_right == raw_end else 0
                        )

            segments.append(NucCall(
                start,
                end - start,
                int(call.nq),
                left_edge,
                right_edge,
                dyad=dyad,
                radial_start=getattr(call, "radial_start", None),
                radial_end=getattr(call, "radial_end", None),
                phase_resolved_left=bool(getattr(
                    call, "phase_resolved_left", False,
                )),
                phase_resolved_right=bool(getattr(
                    call, "phase_resolved_right", False,
                )),
            ))
        if not segments:
            validated.append(NucCall(
                original_start,
                original_end - original_start,
                nq=0,
                el=0,
                er=0,
            ))
            continue

        left_gap_end = min(original_end, int(segments[0].start))
        if left_gap_end > original_start:
            left_supported, left_quality = supports_access(
                original_start, left_gap_end,
            )
            if left_supported:
                supported_access.append((original_start, left_gap_end - original_start))
                first = segments[0]
                segments[0] = NucCall(
                    int(first.start),
                    int(first.length),
                    min(int(first.nq), int(left_quality)),
                    int(first.el),
                    int(first.er),
                    dyad=first.dyad,
                    radial_start=first.radial_start,
                    radial_end=first.radial_end,
                    phase_resolved_left=first.phase_resolved_left,
                    phase_resolved_right=first.phase_resolved_right,
                )
            else:
                first = segments[0]
                segments[0] = NucCall(
                    int(first.start),
                    int(first.length),
                    int(first.nq),
                    0,
                    int(first.er),
                    dyad=first.dyad,
                    radial_start=first.radial_start,
                    radial_end=first.radial_end,
                    phase_resolved_left=first.phase_resolved_left,
                    phase_resolved_right=first.phase_resolved_right,
                )

        # A radial dyad match alone cannot establish an accessible linker.
        # When the complete intervening configuration has direct linker
        # evidence, retain the observed gap as baseline TF scan space. Otherwise
        # preserve the phase-supported particle cores but mark their facing
        # edges Q0 and do not add the uncertain residue to ``supported_access``.
        # It therefore cannot seed baseline TF recall. The final BAM tiling may
        # display the residue between two Q0 edges, and a later coverage-gated
        # TF consensus may explicitly test an alternative configuration there.
        merged_segments: List[NucCall] = []
        current = segments[0]
        for next_call in segments[1:]:
            current_end = int(current.start) + int(current.length)
            next_start = int(next_call.start)
            gap_start = max(original_start, current_end)
            gap_end = min(original_end, next_start)
            split_supported, split_quality = supports_access(gap_start, gap_end)
            if gap_end > gap_start and split_supported:
                supported_access.append((gap_start, gap_end - gap_start))
                merged_segments.append(NucCall(
                    int(current.start),
                    int(current.length),
                    min(int(current.nq), int(split_quality)),
                    int(current.el),
                    int(current.er),
                    dyad=current.dyad,
                    radial_start=current.radial_start,
                    radial_end=current.radial_end,
                    phase_resolved_left=current.phase_resolved_left,
                    phase_resolved_right=current.phase_resolved_right,
                ))
                current = NucCall(
                    int(next_call.start),
                    int(next_call.length),
                    min(int(next_call.nq), int(split_quality)),
                    int(next_call.el),
                    int(next_call.er),
                    dyad=next_call.dyad,
                    radial_start=next_call.radial_start,
                    radial_end=next_call.radial_end,
                    phase_resolved_left=next_call.phase_resolved_left,
                    phase_resolved_right=next_call.phase_resolved_right,
                )
            else:
                current_start = int(current.start)
                next_end = int(next_call.start) + int(next_call.length)
                merged_segments.append(NucCall(
                    current_start,
                    current_end - current_start,
                    nq=int(current.nq),
                    el=int(current.el),
                    er=0,
                    dyad=current.dyad,
                    radial_start=current.radial_start,
                    radial_end=current.radial_end,
                    phase_resolved_left=current.phase_resolved_left,
                    phase_resolved_right=False,
                ))
                current = NucCall(
                    next_start,
                    next_end - next_start,
                    nq=int(next_call.nq),
                    el=0,
                    er=int(next_call.er),
                    dyad=next_call.dyad,
                    radial_start=next_call.radial_start,
                    radial_end=next_call.radial_end,
                    phase_resolved_left=False,
                    phase_resolved_right=next_call.phase_resolved_right,
                )
        merged_segments.append(current)

        right_gap_start = max(
            original_start,
            int(merged_segments[-1].start) + int(merged_segments[-1].length),
        )
        if original_end > right_gap_start:
            right_supported, right_quality = supports_access(
                right_gap_start, original_end,
            )
            if right_supported:
                supported_access.append((right_gap_start, original_end - right_gap_start))
                last = merged_segments[-1]
                merged_segments[-1] = NucCall(
                    int(last.start),
                    int(last.length),
                    min(int(last.nq), int(right_quality)),
                    int(last.el),
                    int(last.er),
                    dyad=last.dyad,
                    radial_start=last.radial_start,
                    radial_end=last.radial_end,
                    phase_resolved_left=last.phase_resolved_left,
                    phase_resolved_right=last.phase_resolved_right,
                )
            else:
                last = merged_segments[-1]
                merged_segments[-1] = NucCall(
                    int(last.start),
                    int(last.length),
                    int(last.nq),
                    int(last.el),
                    0,
                    dyad=last.dyad,
                    radial_start=last.radial_start,
                    radial_end=last.radial_end,
                    phase_resolved_left=last.phase_resolved_left,
                    phase_resolved_right=last.phase_resolved_right,
                )

        if nuc_profile is not None:
            completed_segments = []
            for segment in merged_segments:
                dyad = getattr(segment, "dyad", None)
                if dyad is None:
                    completed_segments.append(segment)
                    continue
                dyad = int(dyad)
                # Search beyond the emitted radial edge. ``radial_start`` and
                # ``radial_end`` are the edge estimates themselves; using them
                # as the completion limits made every adjacent-TF candidate
                # impossible by construction.  The broader interval below is
                # only a hypothesis-search envelope.  It does not set the
                # emitted boundary: a candidate TF edge still has to pass the
                # molecule likelihood, separator veto, and phase-continuation
                # test in ``_complete_radial_nuc_from_adjacent_tf``.
                #
                # Reuse the same deliberately broad envelope used when the
                # radial edge was first scanned, then clip it at neighboring
                # HMM particles and radial dyads.  This permits asymmetric and
                # non-canonical particles without allowing one dyad to consume
                # the next particle.
                search_half_span = int(nuc_profile.half) + 37
                left_limit = max(
                    0,
                    dyad - search_half_span,
                )
                right_limit = min(
                    int(read_length),
                    dyad + search_half_span,
                )
                if original_index > 0:
                    left_limit = max(
                        left_limit, originals[original_index - 1][1]
                    )
                if original_index + 1 < len(originals):
                    right_limit = min(
                        right_limit, originals[original_index + 1][0]
                    )
                previous_dyads = [value for value in radial_dyads if value < dyad]
                next_dyads = [value for value in radial_dyads if value > dyad]
                if previous_dyads:
                    left_limit = max(
                        left_limit, (previous_dyads[-1] + dyad + 1) // 2
                    )
                if next_dyads:
                    right_limit = min(
                        right_limit, (dyad + next_dyads[0]) // 2
                    )
                completed_segments.append(
                    _complete_radial_nuc_from_adjacent_tf(
                        segment,
                        obs,
                        tf_calls,
                        nuc_profile,
                        left_limit,
                        right_limit,
                        nuc_min_size,
                        llr_hit,
                        llr_miss,
                        m5c_mask=m5c_mask,
                        m5c_llr_hit=m5c_llr_hit,
                        m5c_llr_miss=m5c_llr_miss,
                    )
                )
            merged_segments = completed_segments

        validated.extend(merged_segments)

    validated.sort(key=lambda call: (int(call.start), int(call.length)))
    merged_access = merge_intervals([
        (int(start), int(start) + int(length))
        for start, length in supported_access
        if int(length) > 0
    ])
    return validated, [(start, end - start) for start, end in merged_access]


def exclude_nucleosomes_from_msps(msps, nuc_calls, msp_min_size):
    """Subtract called nucleosomes from candidate MSP scan intervals."""
    nuc_intervals = merge_intervals([
        (int(call.start), int(call.start) + int(call.length))
        for call in nuc_calls
        if int(call.length) > 0
    ])
    result: List[Interval] = []
    floor = max(1, int(msp_min_size))
    for start_raw, length_raw in msps:
        start = int(start_raw)
        end = start + int(length_raw)
        cursor = start
        for left, right in nuc_intervals:
            if right <= cursor:
                continue
            if left >= end:
                break
            if left > cursor and left - cursor >= floor:
                result.append((cursor, left - cursor))
            cursor = max(cursor, right)
            if cursor >= end:
                break
        if end - cursor >= floor:
            result.append((cursor, end - cursor))
    return result


def recall_nucs_in_read(
    obs: np.ndarray,
    ns: Sequence[int],
    nl: Sequence[int],
    read_length: int,
    llr_hit: np.ndarray,
    llr_miss: np.ndarray,
    *,
    split_min_llr: float,
    split_min_opps: int,
    nuc_min_size: int,
    edge_min_llr: float = 2.0,
    edge_min_opps: int = 2,
    phase_nrl: int = 0,
    phase_min_llr: float = 1.0,
    phase_min_opps: int = 1,
    phase_window: int = 35,
    nuc_profile: NucProfile | None = None,
    recall_policy: str = "conservative",
) -> Tuple[List[NucCall], List[Interval]]:
    """Split + edge-refine the footprints (``ns``/``nl``) of one read.

    Returns ``(nuc_calls, accessible_intervals)``:
      - ``nuc_calls``: refined nucleosomes (>= ``nuc_min_size``) with nq/el/er.
      - ``accessible_intervals``: (start, length) patches freed up by splitting
        (the cuts) or trimming (overshoot residue + sub-min-size fragments).
        These feed the MSP re-derivation.

    Pass 1 (``split_min_llr``/``split_min_opps``) is the evidence-driven split:
    accessible runs inside a footprint are cuts. Pass 2 (enabled when
    ``phase_nrl > 0``) is the evidence-gated periodicity prior: a footprint
    longer than ~1.5x the nucleosome repeat length is examined for cuts at
    phase-predicted linker positions using a LOWERED threshold
    (``phase_min_llr`` < ``split_min_llr``). The prior only lowers the bar near
    a predicted linker -- a cut still requires real local evidence there, so a
    signal-desert is never split.

    When ``nuc_profile`` is supplied (DddA mode), the accessible-cut split is
    replaced by a radial template match-filter -- see ``radial_split_in_read``.

    ``recall_policy="conservative"`` preserves the historical behavior: every
    qualifying accessible run becomes a cut and protected evidence defines
    conservative nucleosome edges. ``recall_policy="topology"`` is intended for
    sparse single-strand evidence such as Nanopore m6A: a cut is accepted only
    when every resulting piece can contain a nucleosome, and post-cut fragments
    retain their HMM extent instead of converting unresolved edge ambiguity to
    accessibility.
    """
    if recall_policy not in {"conservative", "topology"}:
        raise ValueError(
            "recall_policy must be 'conservative' or 'topology', got "
            f"{recall_policy!r}"
        )
    if nuc_profile is not None:
        return radial_split_in_read(
            obs,
            ns,
            nl,
            read_length,
            nuc_profile,
            nuc_min_size,
            llr_hit,
            llr_miss,
        )
    topology_policy = recall_policy == "topology"
    nhit = -llr_hit
    nmiss = -llr_miss
    nucs: List[NucCall] = []
    access: List[Interval] = []

    for s_raw, length_raw in zip(ns, nl):
        s = int(s_raw)
        length = int(length_raw)
        if length <= 0:
            continue
        e = min(s + length, read_length)
        if e <= s:
            continue

        # --- SPLIT: accessible runs inside the footprint are cuts ---
        cuts = call_tfs_in_interval(obs, s, e, nhit, nmiss,
                                    split_min_llr, split_min_opps)
        cuts = sorted(cuts, key=lambda c: c.start)
        if topology_policy:
            cuts = _select_nucleosome_separating_cuts(
                cuts, s, e, nuc_min_size)
        for c in cuts:
            access.append((c.start, c.length))

        # --- fragments = footprint minus the cut spans ---
        frags: List[Interval] = []
        cur = s
        for c in cuts:
            cs = c.start
            ce = c.start + c.length
            if cs > cur:
                frags.append((cur, cs))
            cur = max(cur, ce)
        if cur < e:
            frags.append((cur, e))

        # --- Pass 2 (optional): phase-prior split of long fragments ---
        for a, b in frags:
            if phase_nrl > 0:
                subs, phase_cuts = _phase_subfragments(
                    obs, a, b, nhit, nmiss, phase_nrl,
                    phase_min_llr, phase_min_opps, phase_window,
                    min_fragment_size=(
                        nuc_min_size if topology_policy else 0
                    ))
                access.extend(phase_cuts)
            else:
                subs = [(a, b)]
            # --- EDGES + quality per (sub)fragment (protected Kadane, +llr) ---
            for sa, sb in subs:
                nuc, acc = _refine_fragment(
                    obs, sa, sb, llr_hit, llr_miss,
                    nuc_min_size, edge_min_llr, edge_min_opps,
                    preserve_fragment=topology_policy)
                if nuc is not None:
                    nucs.append(nuc)
                access.extend(acc)

    return nucs, access


def rederive_msps(
    original_msps: Sequence[Interval],
    accessible_from_splits: Sequence[Interval],
    read_length: int,
    msp_min_size: int,
) -> List[Interval]:
    """Re-derive MSPs from the new nucleosome boundaries.

    MSPs after nuc-recall = the original HMM MSPs unioned with the accessible
    patches freed by splitting/trimming, merged, and filtered to
    ``>= msp_min_size``. Returns (start, length) intervals.
    """
    iv: List[Interval] = []
    for s_raw, length_raw in list(original_msps) + list(accessible_from_splits):
        s = int(s_raw)
        length = int(length_raw)
        if length <= 0:
            continue
        a = max(0, s)
        b = min(read_length, s + length)
        if b > a:
            iv.append((a, b))
    merged = merge_intervals(iv)
    floor = max(1, int(msp_min_size))
    return [(a, b - a) for a, b in merged if (b - a) >= floor]


def unify_nuc_calls_with_tf_calls(
    nuc_calls: Sequence[NucCall],
    tf_calls: Sequence,
    unify_threshold: int,
) -> List[NucCall]:
    """Drop short refined nucleosomes overlapped by a TF call (carry nq/el/er).

    Mirrors ``tagging.unify_nucs_with_tf_calls`` but operates on NucCall objects
    so the per-nuc quality bytes survive unification.
    """
    tf_intervals = [(c.start, c.start + c.length) for c in tf_calls]
    kept: List[NucCall] = []
    for nc in nuc_calls:
        if nc.length <= 0:
            continue
        keep = nc.length >= unify_threshold
        if not keep:
            nuc_end = nc.start + nc.length
            keep = not any(ts < nuc_end and te > nc.start
                           for ts, te in tf_intervals)
        if keep:
            kept.append(nc)
    return kept


def assemble_nuc_msp_tiling(nuc_calls, span_lo, span_hi, msp_min_size,
                            nuc_min_size=85):
    """Produce non-overlapping nucleosomes + complementary MSPs that TILE
    ``[span_lo, span_hi)``.

    Splitting, the phase prior and TF->nuc promotion can leave overlapping
    nucleosomes and stale MSPs, but fibertools / FIRE require nucleosomes
    (ns/nl) and MSPs (as/al) to be sorted, non-overlapping, and tiling. This
    clips overlaps and derives MSPs as the gaps between the final nucleosomes.

    Ordering/clipping rules:
      - sort by (start, -end) so the LONGER call at a given start wins; this
        keeps a promoted full-length nucleosome over a short same-start call
        (which would otherwise be clipped, splitting the promoted one back into
        sub-nucleosome pieces).
      - clip the left of an overlapping call to the previous end (zeroing the
        now-meaningless left edge byte), and
      - drop any call that falls below ``nuc_min_size`` after clipping (its span
        reverts to MSP), so no sub-nucleosome nuc+ calls leak out.
    Returns ``(kept_nucs, msp_intervals)``.
    """
    floor = max(1, int(msp_min_size))
    nfloor = max(1, int(nuc_min_size))
    ordered = sorted((n for n in nuc_calls if n.length > 0),
                     key=lambda n: (n.start, -(n.start + n.length)))
    kept = []
    last_end = span_lo
    for n in ordered:
        s = n.start
        e = n.start + n.length
        el = n.el
        if s < last_end:          # overlaps the previous nucleosome
            s = last_end
            el = 0                # clipped left edge is no longer meaningful
        if e - s < nfloor:
            continue              # swallowed, or clipped below the nuc floor
        kept.append(NucCall(s, e - s, n.nq, el, n.er))
        last_end = e

    msps = []
    cur = span_lo
    for k in kept:
        if k.start - cur >= floor:
            msps.append((cur, k.start - cur))
        cur = max(cur, k.start + k.length)
    if span_hi - cur >= floor:
        msps.append((cur, span_hi - cur))
    return kept, msps


def assemble_circular_nuc_msp_tiling(nuc_calls, read_length, msp_min_size,
                                     nuc_min_size=85):
    """Circular-aware ``assemble_nuc_msp_tiling``.

    On a circular molecule a nucleosome can wrap the origin
    (``start + length > read_length``). Running the linear tiler at a fixed
    origin would derive MSP gaps that overlap a wrapped nucleosome's tail (e.g.
    a nuc covering ``[95,100)+[0,15)`` plus a spurious MSP ``[0,95)`` overlapping
    ``[0,15)``). Instead rotate the circle to an origin, split any call that
    still wraps that origin into two linear pieces, tile linearly, then rotate
    the kept nucs and MSPs back. Returns ``(kept_nucs, msp_intervals)`` in
    molecular coordinates.

    Edge cases:
      - no nucleosomes -> the whole molecule is one accessible MSP;
      - fully covered (no uncovered cut point, e.g. overlapping nucs that tile
        the circle): the origin can fall inside a wrapped call, so straddling
        calls are split at the origin before the linear clip -- otherwise the
        tiler would emit overlapping/wrapped pieces.
    """
    rl = int(read_length)
    floor = max(1, int(msp_min_size))
    nfloor = max(1, int(nuc_min_size))
    calls = [n for n in nuc_calls if n.length > 0]
    if rl <= 0:
        return list(calls), []
    if not calls:
        # no nucleosomes -> the entire molecule tiles as one accessible MSP
        return [], ([(0, rl)] if rl >= floor else [])
    whole = next((n for n in calls if int(n.length) >= rl), None)
    if whole is not None:
        if rl >= nfloor:
            return [NucCall(0, rl, whole.nq, whole.el, whole.er)], []
        return [], ([(0, rl)] if rl >= floor else [])

    # Prefer an uncovered cut point (no call straddles it); fall back to 0 when
    # the circle is fully covered. Either way, straddlers are split below, so a
    # fallback origin landing inside a wrapped call is handled correctly.
    covered = np.zeros(rl, dtype=bool)
    for n in calls:
        s = n.start % rl
        span = min(n.length, rl)
        idx = (s + np.arange(span)) % rl
        covered[idx] = True
    uncovered = np.flatnonzero(~covered)
    cut = int(uncovered[0]) if uncovered.size else 0

    rotated = []
    for n in calls:
        rs = (n.start - cut) % rl
        length = min(n.length, rl)
        end = rs + length
        if end <= rl:
            rotated.append(NucCall(rs, length, n.nq, n.el, n.er))
        else:
            # wraps the rotated origin -> split into [rs, rl) and [0, end-rl);
            # each piece keeps its real outer edge, the cut edge byte is zeroed
            # (same convention as split_intervals_for_legacy on a wrapped nuc).
            rotated.append(NucCall(rs, rl - rs, n.nq, n.el, 0))
            rotated.append(NucCall(0, end - rl, n.nq, 0, n.er))
    kept_rot, msp_rot = assemble_nuc_msp_tiling(
        rotated, 0, rl, msp_min_size, nuc_min_size)

    kept = sorted(
        (NucCall((k.start + cut) % rl, k.length, k.nq, k.el, k.er) for k in kept_rot),
        key=lambda n: n.start)
    msps = sorted(((s + cut) % rl, length) for s, length in msp_rot)
    return kept, msps


def drop_short_nucs_overlapping_promoted(nuc_calls, promoted, unify_threshold):
    """Drop short (< ``unify_threshold``) nucleosomes that overlap a promoted one.

    Promotion moves a nucleosome-sized TF call into the nuc set and removes it
    from ``tf_calls``, so ``unify_nuc_calls_with_tf_calls`` no longer drops a
    short nuc that overlapped it. Apply the same rule here against the promoted
    intervals: a short call overlapping a real (promoted) nucleosome is spurious.
    Without this, the start-order tiling can keep the short call and clip/drop
    the promoted one. Returns the filtered nuc list.
    """
    if not promoted:
        return list(nuc_calls)
    pints = [(p.start, p.start + p.length) for p in promoted]
    out = []
    for n in nuc_calls:
        if n.length >= unify_threshold:
            out.append(n)
            continue
        n_end = n.start + n.length
        if any(ps < n_end and n.start < pe for ps, pe in pints):
            continue  # short nuc overlapping a promoted nucleosome -> drop
        out.append(n)
    return out


def promote_large_tf_calls(tf_calls, obs, llr_hit, llr_miss, threshold,
                           nuc_min_size, edge_min_llr=2.0, edge_min_opps=2,
                           preserve_fragment=False):
    """Promote nucleosome-sized TF calls (length >= ``threshold``) to NucCalls.

    The TF recaller emits ANY protected run inside an MSP as ``tf+`` with no size
    cap, so a nucleosome the HMM mis-placed in an MSP leaks into the TF track. A
    protected run >= ``threshold`` (``unify_threshold``) is a nucleosome by
    default -- relabel it, computing proper conservative edges via the same
    protected-Kadane edge pass. Returns ``(remaining_tf_calls, promoted_nucs)``.
    """
    remaining = []
    promoted: List[NucCall] = []
    for c in tf_calls:
        if c.length >= threshold:
            nuc, _ = _refine_fragment(obs, c.start, c.start + c.length,
                                      llr_hit, llr_miss, nuc_min_size,
                                      edge_min_llr, edge_min_opps,
                                      preserve_fragment=preserve_fragment)
            if nuc is not None:
                promoted.append(nuc)
                continue
        remaining.append(c)
    return remaining, promoted


def unify_circular_nuc_calls_with_tf_calls(
    nuc_calls: Sequence[NucCall],
    tf_calls: Sequence,
    unify_threshold: int,
    read_length: int,
) -> List[NucCall]:
    """Circular counterpart of ``unify_nuc_calls_with_tf_calls``.

    Nuc calls and TF calls are in molecular (circular) coordinates; overlap is
    tested with circular-aware segment overlap.
    """
    from fiberhmm.inference.circular import circular_intervals_overlap

    tf_intervals = [(c.start, c.length) for c in tf_calls]
    kept: List[NucCall] = []
    for nc in nuc_calls:
        if nc.length <= 0:
            continue
        keep = nc.length >= unify_threshold
        if not keep:
            iv = (nc.start, nc.length)
            keep = not any(
                circular_intervals_overlap(iv, tfi, read_length)
                for tfi in tf_intervals
            )
        if keep:
            kept.append(nc)
    return kept
