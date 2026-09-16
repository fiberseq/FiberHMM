"""Native-lattice diagnostics for a broad family versus existing TF pieces.

No source call or native family is changed. A compound is an explicit set of
original calls, NOT a merged XCR interval. All its observations are unfrozen in
both hypotheses. A conditional predictive reference re-runs a scan over ALL
internal protected/accessible patterns, not only the HMM-selected separator.

This caller-conditioned diagnostic is not a new-call rescue, state-equivalence
probability, native-FDR estimate, or a calibrated scan over proposed compounds.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from numba import njit
from scipy.special import logsumexp

from .native_cross import boundary_grid, transfer_density, _recipient_observations


def _penalties(density, starts, ends, relax_left, relax_right):
    base = density-density.max(); adjusted = base.copy()
    if relax_left:
        v = np.full(int(ends.max())+1, -np.inf); np.maximum.at(v, ends, base)
        adjusted = np.maximum(adjusted, v[ends])
    if relax_right:
        v = np.full(int(ends.max())+1, -np.inf); np.maximum.at(v, starts, base)
        adjusted = np.maximum(adjusted, v[starts])
    if relax_left and relax_right: adjusted[:] = 0.
    return base, adjusted


@njit(cache=True)
def compound_profile_statistic(values, starts, ends, native_penalty, edge_penalty):
    """Max(shape loss, internal-pattern gain), on one observation domain.

    The alternate internal pattern can assign accessible/protected independently
    at each observed opportunity INSIDE a drawn-family shape. Maximizing it is
    exactly sum(max(native step,0)); this exhausts every collection of gaps, not
    a chosen gap or a capped list of sub-footprints. The generative reference
    repeats that same overfit on every replicate. No LR is interpreted directly
    as a posterior, and the two diagnostics are not added/double-counted.
    """
    prefix = np.zeros(len(values)+1); positive = np.zeros(len(values)+1)
    for j in range(len(values)):
        prefix[j+1] = prefix[j]+values[j]
        positive[j+1] = positive[j]+max(0., values[j])
    unrestricted = -np.inf; continuous = -np.inf; relaxed = -np.inf; punctuated = -np.inf
    for g in range(len(starts)):
        value = prefix[ends[g]]-prefix[starts[g]]
        unrestricted = max(unrestricted, value)
        continuous = max(continuous, value+native_penalty[g])
        relaxed = max(relaxed, value+edge_penalty[g])
        punctuated = max(punctuated, positive[ends[g]]-positive[starts[g]]+native_penalty[g])
    shape = max(0., unrestricted-relaxed)
    gaps = max(0., punctuated-continuous)
    return max(shape, gaps), shape, gaps


@njit(cache=True)
def _compound_reference(pa, pp, starts, ends, penalty, edge_penalty, cdf, threshold, replicates, seed):
    np.random.seed(seed); exceed = 0
    for _ in range(replicates):
        g = np.searchsorted(cdf, np.random.random())
        values = np.zeros(len(pa))
        for j in range(len(pa)):
            prob = pp[j] if starts[g] <= j < ends[g] else pa[j]
            hit = np.random.random() < prob
            values[j] = np.log(pp[j]/pa[j]) if hit else np.log1p(-pp[j])-np.log1p(-pa[j])
        value, _, _ = compound_profile_statistic(values, starts, ends, penalty, edge_penalty)
        exceed += value >= threshold-1e-10
    return exceed


def compound_predictive_reference(log_density, log_mass, values, starts, ends, *,
                                  allowed, observed, p_accessible, p_protected,
                                  replicates=4095, seed=1, relax_left=False, relax_right=False):
    """One JOINT predictive statistic; every alternative is rescanned per draw."""
    d, lm, values = np.asarray(log_density, float), np.asarray(log_mass, float), np.asarray(values, float)
    a, b, valid = np.asarray(starts), np.asarray(ends), np.asarray(allowed, bool)
    obs, pa, pp = np.asarray(observed, bool), np.asarray(p_accessible, float), np.asarray(p_protected, float)
    if (not isinstance(replicates, (int, np.integer)) or replicates < 1 or not obs.any()
            or d.shape != lm.shape or a.shape != d.shape or b.shape != d.shape or valid.shape != d.shape
            or not valid.any() or not np.isfinite(d[valid]).any()
            or pa.shape != obs.shape or pp.shape != obs.shape or values.shape != obs.shape
            or np.any(~np.isfinite(values)) or np.any(np.isnan(d) | np.isposinf(d))
            or np.any((pa[obs] <= 0) | (pa[obs] >= 1) | (pp[obs] <= 0) | (pp[obs] >= 1))):
        raise ValueError('Complete native probabilities, observations and admissible shapes required')
    pen, adj = _penalties(d, a, b, relax_left, relax_right)
    index = np.r_[0, np.cumsum(obs)]
    unique, inv = np.unique(np.c_[index[a[valid]], index[b[valid]]], axis=0, return_inverse=True)
    w = np.exp(lm[valid]-logsumexp(lm[valid]))
    w = np.bincount(inv, weights=w, minlength=len(unique))
    p1 = np.full(len(unique), -np.inf); p2 = p1.copy()
    np.maximum.at(p1, inv, pen[valid]); np.maximum.at(p2, inv, adj[valid])
    statistic, shape, gaps = compound_profile_statistic(values[obs], unique[:, 0], unique[:, 1], p1, p2)
    result = dict(compound_statistic=float(statistic), native_shape_loss=float(shape),
        internal_pattern_gain=float(gaps), statistic_definition='max(shape_loss, all_internal_patterns_gain)',
        internal_pattern_scan_repeated_in_reference=True, simulated_missing_opportunities=False,
        fixed_selected_gap_veto=False, projection_classes=len(unique))
    if statistic <= 1e-10:
        return result | dict(predictive_tail=1., predictive_tail_interval=[1., 1.], simulations=0,
                             tail_exceedances=0, simulation_resolution=0.)
    cdf = np.cumsum(w); cdf[-1] = 1.
    exceed = int(_compound_reference(pa[obs], pp[obs], unique[:, 0], unique[:, 1], p1, p2,
        cdf, statistic, replicates, seed))
    z = 1.959963984540054; freq = exceed/replicates; den = 1+z*z/replicates
    center = (freq+z*z/(2*replicates))/den
    half = z*np.sqrt(freq*(1-freq)/replicates+z*z/(4*replicates**2))/den
    return result | dict(predictive_tail=(exceed+1.)/(replicates+1.),
        predictive_tail_interval=[max(0., float(center-half)), min(1., float(center+half))],
        simulations=replicates, tail_exceedances=exceed, simulation_resolution=1./(replicates+1.))


def score_existing_compound(frozen, unit, calls, *, floor_bp=0, replicates=4095, minimum_retained_mass=.05):
    """Test one explicit contiguous collection of original TF calls.

    The source shape must overlap EACH piece, not cover its raw outer edges:
    raw bounds are noisy measurements, not a second set of ground-truth edges.
    All unselected TFs and every nucleosome remain frozen. No virtual merged
    span is emitted as a footprint. The source prior is not fitted to recipients.
    """
    calls = sorted(calls, key=lambda c:(c['start'], c['end'], c['ordinal']))
    if len(calls) < 2 or len({c['unit_id'] for c in calls}) != 1 or len({c['ordinal'] for c in calls}) != len(calls):
        raise ValueError('At least two distinct original calls on one evidence unit required')
    span = [min(c['start'] for c in calls), max(c['end'] for c in calls)]
    chosen = {(c['start'], c['end']) for c in calls}
    raw = [tuple(s) for s in unit['representative_raw_tf_intervals']]
    if not chosen <= set(raw): raise ValueError('Compound membership not present in original native calls')
    key = f"native-compound-v1|{frozen['model']['family']}|{calls[0]['unit_id']}|"+','.join(str(c['ordinal']) for c in calls)
    record = dict(compound_id='C_'+hashlib.sha256(key.encode()).hexdigest()[:16],
        donor_family=frozen['model']['family'], unit_id=calls[0]['unit_id'],
        source_ordinals=[c['ordinal'] for c in calls], source_intervals=[[c['start'], c['end']] for c in calls],
        original_spans_unchanged=True, merged_interval=None, new_calls=0,
        edge_floor_bp=floor_bp, caller_conditioned_proposal=True, calibrated_FDR=False)
    outside = [s for s in raw if s not in chosen]
    if any(x < span[1] and y > span[0] for x, y in outside):
        return record | dict(status='unselected_overlapping_call_blocks_compound')
    if any(x < span[1] and y > span[0] for x, y in unit.get('raw_nuc_intervals', [])):
        return record | dict(status='frozen_nucleosome_blocks_compound')
    # Preserve exact fractional source mass at every overlap/neighbor cut.
    cuts = [v for c in calls for v in (c['start'], c['end']-1)]
    # A right boundary cell is [p[b-1]+1,p[b]]. Split at a following
    # neighbor START x, not x-1, so right_lo<=x also implies right_hi<=x.
    # A left boundary cell is [p[a-1]+1,p[a]]; a previous END y instead
    # requires y-1. These asymmetric integer/half-open cuts are intentional.
    cuts += [v for x, y in outside+list(unit.get('raw_nuc_intervals', [])) for v in (x, y-1)]
    positions = np.unique(np.r_[frozen['grid']['positions'], unit['positions'], cuts])
    grid = boundary_grid(positions, frozen['grid']['domain'])
    d = transfer_density(frozen, grid); lm = d+np.log(grid['areas']); lm -= logsumexp(lm)
    virtual = dict(calls[0], start=span[0], end=span[1])
    working = dict(unit, representative_raw_tf_intervals=outside)
    evidence = _recipient_observations(working, virtual, grid, require_call_overlap=False)
    # Because the grid was refined at the exact cuts, these predicates keep
    # entire positive-coordinate cells; no surviving-area tail is inflated.
    overlap = np.ones(len(d), bool)
    for c in calls:
        overlap &= (grid['left_hi'] < c['end']) & (grid['right_lo'] > c['start'])
    valid = evidence['allowed'] & overlap
    kept = float(np.exp(logsumexp(lm[valid]))) if valid.any() else 0.
    overlap_mass = float(np.exp(logsumexp(lm[overlap]))) if overlap.any() else 0.
    neighbor_mass = float(np.exp(logsumexp(lm[evidence['allowed']]))) if evidence['allowed'].any() else 0.
    positions = np.asarray(unit['positions']); member = np.zeros(len(positions), bool)
    for c in calls: member |= (positions >= c['start']) & (positions < c['end'])
    in_domain = (positions >= grid['domain'][0]) & (positions < grid['domain'][1])
    gaps = []
    for left, right in zip(calls, calls[1:]):
        if left['end'] >= right['start']: continue
        selected = ((grid['positions'] >= left['end']) & (grid['positions'] < right['start'])
                    & evidence['observed'])
        raw_gap = (positions >= left['end']) & (positions < right['start'])
        gaps.append(dict(interval=[left['end'], right['start']], observed_opportunities=int(selected.sum()),
            observed_hits=int(np.asarray(unit['hits'])[raw_gap & in_domain].sum()),
            native_log_lr=float(evidence['values'][selected].sum()),
            observed_opportunities_outside_source_domain=int((raw_gap & ~in_domain).sum()),
            descriptive_only=True, fixed_gap_pvalue=None))
    record.update(geometry_mass_overlapping_all_pieces=kept,
        geometry_mass_overlapping_all_pieces_and_avoiding_neighbors=kept,
        geometry_mass_overlapping_all_pieces_before_neighbor_conditioning=overlap_mass,
        geometry_mass_allowed_by_frozen_neighbors=neighbor_mass,
        observed_opportunities=int(evidence['observed'].sum()), member_observations_unfrozen=True,
        member_observed_opportunities=int(member.sum()),
        member_observed_opportunities_outside_source_domain=int((member & ~in_domain).sum()),
        inter_piece_gaps=gaps,
        inter_piece_observed_opportunities=sum(g['observed_opportunities'] for g in gaps),
        inter_piece_observed_hits=sum(g['observed_hits'] for g in gaps))
    if not evidence['observed'].any(): return record | dict(status='no_recipient_information')
    if kept < minimum_retained_mass or not np.isfinite(d[valid]).any():
        return record | dict(status='insufficient_geometry_mass_overlapping_all_pieces')
    opportunity_prefix = np.r_[0, np.cumsum(evidence['observed'])]
    counts = opportunity_prefix[grid['ends']]-opportunity_prefix[grid['starts']]
    weights = np.exp(lm[valid]-math.log(kept))
    visible = float(weights @ (counts[valid] > 0))
    record.update(geometry_observed_mass_fraction=visible,
                  expected_core_opportunities=float(weights @ counts[valid]))
    if visible < .05:
        return record | dict(status='transferred_compound_shape_unobserved')
    ref = frozen['model']['reference_interval']
    score = compound_predictive_reference(d, lm, evidence['values'], grid['starts'], grid['ends'],
        allowed=valid, observed=evidence['observed'], p_accessible=evidence['p_accessible'], p_protected=evidence['p_protected'],
        replicates=replicates, seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little'),
        relax_left=floor_bp > 0 and abs(span[0]-ref[0]) <= floor_bp,
        relax_right=floor_bp > 0 and abs(span[1]-ref[1]) <= floor_bp)
    return record | score | dict(status='scored',
        interpretation='conditional broad-shape adequacy of existing pieces; not proof of one physical complex')
