"""Experimental, measurement-only normalization of existing cross-strand calls.

BoundaryNormalizer is deliberately NOT population-variance CR or new-call
recall. Every input call has an output, with unchanged raw fallback. Source calls nominate joint
boundary-equivalence rectangles on their OWN observed opportunity lattices.
No Gaussian edge jitter, fixed bp tolerance, enrichment gate, or family cap is
used. A source rectangle is not a biological family or a fitted credible region.

For a recipient, intersect source rectangles with its opportunity cells and
score the resulting joint intervals using exact native likelihood differences.
Source recurrence ranks candidates; it is not an independent Bayes factor.
Normalized ranking masses are conditional on borrowing a source proposal, NOT
calibrated assignment, footprint-existence, or correspondence probabilities.
Each edge has a separate native evidence-loss firewall. A good move at one edge
cannot buy an arbitrarily bad move at the other. Conflicting proposed moves are
rolled back together, rather than resolved by processing order.

FootprintRescuer adds a separate, explicitly nominated new-call pass. It uses
the same source lattice cells, an opposite-strand detection-activity prior,
unchanged recipient emissions, and exact nonoverlapping interval competition.
Its model probabilities are not empirically calibrated confidence or FDR.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from math import log

import numpy as np
from scipy.special import logsumexp


@dataclass
class BoundaryRead:
    unit_id: str
    strand: str
    positions: np.ndarray
    hits: np.ndarray
    p_accessible: np.ndarray
    p_protected: np.ndarray
    calls: list
    obstacles: list
    blocks: list

    def __post_init__(self):
        original_positions = np.asarray(self.positions)
        original_hits = np.asarray(self.hits)
        if original_positions.size and original_positions.dtype.kind not in 'iu':
            raise ValueError("Opportunity positions must be integers, not rounded coordinates")
        if not np.all((original_hits == 0) | (original_hits == 1)):
            raise ValueError("Only native hard 0/1 observations are supported by this adapter")
        self.positions = np.asarray(self.positions, dtype=np.int64)
        self.hits = np.asarray(self.hits, dtype=np.int8)
        self.p_accessible = np.asarray(self.p_accessible, dtype=float)
        self.p_protected = np.asarray(self.p_protected, dtype=float)
        if np.any(np.diff(self.positions) <= 0):
            raise ValueError("Opportunity positions must be strictly increasing")
        if any(x.shape != self.positions.shape for x in
               (self.hits, self.p_accessible, self.p_protected)):
            raise ValueError("Every opportunity needs a native observation and both emissions")
        if not np.all((self.hits == 0) | (self.hits == 1)):
            raise ValueError("Only native hard 0/1 observations are supported by this adapter")
        for probabilities in (self.p_accessible, self.p_protected):
            if not np.all(np.isfinite(probabilities) & (probabilities > 0) & (probabilities < 1)):
                raise ValueError("Native emission probabilities must be finite and strictly between 0 and 1")
        self.steps = np.where(self.hits, np.log(self.p_protected / self.p_accessible),
                              np.log1p(-self.p_protected) - np.log1p(-self.p_accessible))
        self.prefix = np.r_[0., np.cumsum(self.steps)]
        self.hit_prefix = np.r_[0, np.cumsum(self.hits)]
        self.calls = [tuple(map(int, x)) for x in self.calls]
        self.obstacles = [tuple(map(int, x)) for x in self.obstacles]
        self.blocks = [tuple(map(int, x)) for x in self.blocks]
        if any(a >= b for a, b in self.calls + self.obstacles + self.blocks):
            raise ValueError("Intervals must have positive half-open width")
        merged = []
        for a, b in sorted(self.blocks):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a, b))
        self.blocks = merged


def edge_cell(positions, boundary):
    """Inclusive integer cell with identical cut index; None if unbounded."""
    cut = int(np.searchsorted(positions, boundary))
    if cut == 0 or cut == len(positions):
        return None
    return int(positions[cut - 1]) + 1, int(positions[cut])


def projection_parts(positions, lo, hi):
    """Exact integer-coordinate intersections, including missing-opportunity cells."""
    first, last = np.searchsorted(positions, [lo, hi])
    answer = []
    for cut in range(int(first), int(last) + 1):
        a = max(lo, int(positions[cut - 1]) + 1) if cut else lo
        b = min(hi, int(positions[cut])) if cut < len(positions) else hi
        if a <= b:
            answer.append((cut, int(a), int(b)))
    return answer


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


@lru_cache(maxsize=256)
def _integration_rule(n):
    # The likelihood is a degree-n polynomial in the mixing fraction. This
    # Gauss-Legendre rule integrates it exactly in exact arithmetic. Log-space
    # accumulation avoids multiplying tiny probabilities; no sampling involved.
    x, w = np.polynomial.legendre.leggauss(max(8, (n + 2) // 2))
    return (x + 1.) / 2., np.log(w / 2.)


def endpoint_pattern_evidence(read, start, end):
    """Two endpoint models versus one NORMALIZED diffuse alternative.

    Some hit/miss-rich extensions have nearly zero endpoint LR because both
    models fit badly. Integrate p_j(t)=pp_j+t*(pa_j-pp_j), t~Uniform(0,1),
    rather than mistaking this cancellation for simple boundary ambiguity.
    This is a conservative model-adequacy flag, NOT evidence to split the call.
    """
    a, b = np.searchsorted(read.positions, sorted((start, end)))
    if a == b:
        return {'opportunities': 0, 'hits': 0, 'best_endpoint_vs_diffuse_log_bf': 0.}
    h, pa, pp = read.hits[a:b], read.p_accessible[a:b], read.p_protected[a:b]
    la = float(np.where(h, np.log(pa), np.log1p(-pa)).sum())
    lp = float(np.where(h, np.log(pp), np.log1p(-pp)).sum())
    t, logw = _integration_rule(len(h))
    probabilities = pp[None, :] + t[:, None] * (pa - pp)[None, :]
    likelihood = np.where(h[None, :], np.log(probabilities), np.log1p(-probabilities)).sum(axis=1)
    diffuse = float(logsumexp(logw + likelihood))
    return {'opportunities': int(b-a), 'hits': int(h.sum()),
            'accessible_log_likelihood': la, 'protected_log_likelihood': lp,
            'diffuse_log_likelihood': diffuse,
            'best_endpoint_vs_diffuse_log_bf': max(la, lp) - diffuse}


def build_source_catalog(reads):
    """Every observed call nominates; exact member-lattice cells alone collapse.

    Rectangles are identical for ALL member reads, so moving within a rectangle
    preserves each member's own observed opportunity projection. Rare sequence
    differences or missing opportunities in OTHER reads cannot make this cell
    spuriously precise. Source weights count a unit at most once per rectangle.
    Censored/empty calls remain in the omission ledger and in raw output.
    """
    groups, omitted, seen = {}, [], set()
    for read in sorted(reads, key=lambda r: r.unit_id):
        if read.unit_id in seen:
            raise ValueError("Duplicate evidence-unit ID; deduplicate before source nomination")
        seen.add(read.unit_id)
        for ordinal, (a, b) in enumerate(read.calls):
            ia, ib = np.searchsorted(read.positions, [a, b])
            left, right = edge_cell(read.positions, a), edge_cell(read.positions, b)
            if ia == ib or left is None or right is None:
                omitted.append({"unit_id": read.unit_id, "ordinal": ordinal,
                                "raw": [a, b], "reason": "empty_or_censored_source_projection"})
                continue
            # No extrapolation across an alignment gap, including the cell ends.
            if not any(x <= left[0] and right[1] <= y for x, y in read.blocks):
                omitted.append({"unit_id": read.unit_id, "ordinal": ordinal,
                                "raw": [a, b], "reason": "source_cell_crosses_alignment_gap"})
                continue
            key = (*left, *right)
            group = groups.setdefault(key, {"members": {}, "aliases": set()})
            group["members"][read.unit_id] = float(read.prefix[ib] - read.prefix[ia])
            group["aliases"].add((a, b))
    nodes = []
    for rect, group in sorted(groups.items()):
        ids = sorted(group["members"])
        folds = [int(sha256(uid.encode()).hexdigest()[:8], 16) % 2 for uid in ids]
        nodes.append({"node_id": "S_" + sha256(repr(rect).encode()).hexdigest()[:16],
                      "rectangle": list(rect), "source_units": ids,
                      "support": len(ids), "fold_support": [folds.count(0), folds.count(1)],
                      "aliases": [list(x) for x in sorted(group["aliases"])],
                      "median_native_core_log_lr": float(np.median(list(group["members"].values())))})
    return nodes, omitted


class BoundaryNormalizer:
    """Frozen source catalog; scoring and stringency-dependent decoding separate."""
    def __init__(self, source_nodes):
        self.nodes = source_nodes
        self.rectangles = np.asarray([n["rectangle"] for n in source_nodes], dtype=np.int64).reshape(-1, 4)
        self._overlap_cache = {}

    def score_call(self, read, ordinal):
        raw = read.calls[ordinal]
        ia, ib = map(int, np.searchsorted(read.positions, raw))
        if ia == ib:
            return {"raw": list(raw), "ordinal": ordinal, "candidates": [],
                    "reason": "no_recipient_core_opportunity"}
        if raw not in self._overlap_cache:
            r = self.rectangles
            self._overlap_cache[raw] = np.flatnonzero((r[:, 0] < raw[1]) & (r[:, 3] > raw[0]))
        obstacles = read.obstacles + [x for j, x in enumerate(read.calls) if j != ordinal]
        baseline_overlaps = [overlap(raw, x) for x in obstacles]
        block = next(((a, b) for a, b in read.blocks if a <= raw[0] and raw[1] <= b), None)
        if block is None:
            return {"raw": list(raw), "ordinal": ordinal, "candidates": [],
                    "reason": "raw_call_crosses_alignment_gap"}
        # Clip the full cells BEFORE choosing representatives. Forbid newly
        # occupying any part of another call outside this call's ORIGINAL span.
        # Two sides are tied to one raw-overlapping interval, not averaged edges.
        minimum_start, maximum_end = block
        for a, b in obstacles:
            if a < raw[0]: minimum_start = max(minimum_start, min(b, raw[0]))
            if b > raw[1]: maximum_end = min(maximum_end, max(a, raw[1]))
        candidates = []
        for node_index in self._overlap_cache[raw]:
            node = self.nodes[int(node_index)]
            sl, sh, el, eh = node["rectangle"]
            area = (sh - sl + 1) * (eh - el + 1)
            sl = max(sl, minimum_start)
            eh = min(eh, maximum_end)
            # Half-open positive overlap with raw; do not nominate a relocation.
            sh = min(sh, raw[1] - 1)
            el = max(el, raw[0] + 1)
            if sl > sh or el > eh: continue
            for a, al, ah in projection_parts(read.positions, sl, sh):
                for b, bl, bh in projection_parts(read.positions, el, eh):
                    if a >= b:
                        continue  # A source interval invisible to this recipient cannot move its call.
                    geometry = [(al + ah) // 2, (bl + bh) // 2]
                    if not overlap(raw, geometry):
                        continue  # SR is normalization of this existing call, not a relocated call.
                    dl = float(read.prefix[ia] - read.prefix[a])
                    dr = float(read.prefix[b] - read.prefix[ib])
                    fraction = (ah - al + 1) * (bh - bl + 1) / area
                    score = log(node["support"]) + log(fraction) + dl + dr
                    equivalent = (a, b) == (ia, ib)
                    block_ok = any(x <= geometry[0] and geometry[1] <= y for x, y in read.blocks)
                    topology_ok = not any(overlap(geometry, obs) > old
                                          for obs, old in zip(obstacles, baseline_overlaps))
                    # Compare each changed edge separately; common interior cancels.
                    changed = abs(a - ia) + abs(b - ib)
                    changed_hits = (abs(int(read.hit_prefix[a] - read.hit_prefix[ia])) +
                                    abs(int(read.hit_prefix[b] - read.hit_prefix[ib])))
                    candidates.append({"node": int(node_index), "interval": geometry,
                                       "joint_cell": [al, ah, bl, bh], "recipient_projection": [a, b],
                                       "left_native_log_lr": dl, "right_native_log_lr": dr,
                                       "joint_native_log_lr": dl + dr, "log_rank_weight": score,
                                       "source_coordinate_fraction": fraction,
                                       "projection_equivalent": equivalent, "changed_opportunities": changed,
                                       "changed_hits": changed_hits, "topology_ok": topology_ok,
                                       "alignment_ok": block_ok})
        valid = [c for c in candidates if c["topology_ok"] and c["alignment_ok"]]
        total = logsumexp([c["log_rank_weight"] for c in valid]) if valid else 0.
        projection_mass = {}
        for c in candidates:
            mass = float(np.exp(c["log_rank_weight"] - total)) if c["topology_ok"] and c["alignment_ok"] else 0.
            c["conditional_rank_mass"] = mass
            key = tuple(c["recipient_projection"])
            projection_mass[key] = projection_mass.get(key, 0.) + mass
        for c in candidates:
            c["recipient_projection_rank_mass"] = projection_mass[tuple(c["recipient_projection"])]
        if valid:
            top = min(valid, key=lambda c: (-c['log_rank_weight'], tuple(c['joint_cell']), c['node']))
            if not top['projection_equivalent']:
                # Only the top-ranked proposal can change observed membership
                # under decode_call; all fallbacks are exactly equivalent.
                # Score this additional guard once, before policy decoding.
                left = endpoint_pattern_evidence(read, raw[0], top['interval'][0])
                right = endpoint_pattern_evidence(read, raw[1], top['interval'][1])
                top['pattern_adequacy'] = {'left': left, 'right': right,
                    'minimum_best_endpoint_vs_diffuse_log_bf': min(left['best_endpoint_vs_diffuse_log_bf'],
                                                                  right['best_endpoint_vs_diffuse_log_bf'])}
        return {"raw": list(raw), "ordinal": ordinal, "candidates": candidates,
                "reason": "scored" if valid else "no_represented_topology_compatible_source"}

    def decode_call(self, scored, *, loss_budget=0., minimum_source_support=3,
                    minimum_projection_mass=0.8, projection_only=False,
                    maximum_diffuse_odds=100.):
        """Decode one immutable candidate ledger; thresholds never delete a raw call.

        loss_budget is nats per edge AND per interval, not a calibrated Q score.
        The mass gate applies to changing the recipient's observed projection.
        Exact-equivalent canonicalization needs no invented molecular confidence.
        An ambiguous attempted change can fall back to an equivalent source cell.
        """
        if loss_budget < 0 or minimum_source_support < 1 or not 0 <= minimum_projection_mass <= 1 or maximum_diffuse_odds < 1:
            raise ValueError("Invalid decoding parameters")
        candidates = [c for c in scored["candidates"] if c["topology_ok"] and c["alignment_ok"]]
        candidates.sort(key=lambda c: (-c["log_rank_weight"], tuple(c["joint_cell"]), c["node"]))
        result = {"raw": scored["raw"], "interval": scored["raw"], "ordinal": scored["ordinal"],
                  "action": "unchanged", "reason": scored["reason"], "selected": None,
                  "top_proposal": candidates[0] if candidates else None}
        if not candidates:
            return result
        top = candidates[0]
        failures = []
        if self.nodes[top["node"]]["support"] < minimum_source_support:
            failures.append("insufficient_source_recurrence")
        if not top["projection_equivalent"]:
            if projection_only:
                failures.append("projection_only_policy")
            if min(top["left_native_log_lr"], top["right_native_log_lr"], top["joint_native_log_lr"]) < -loss_budget - 1e-10:
                failures.append("recipient_evidence_veto")
            if top["recipient_projection_rank_mass"] < minimum_projection_mass:
                failures.append("ambiguous_recipient_projection")
            pattern = top.get('pattern_adequacy')
            if pattern is None:
                failures.append('edge_pattern_adequacy_unavailable')
            elif pattern['minimum_best_endpoint_vs_diffuse_log_bf'] < -log(maximum_diffuse_odds):
                failures.append('mixed_edge_pattern_needs_richer_model')
        selected = top if not failures else next((c for c in candidates if c["projection_equivalent"] and
                    self.nodes[c["node"]]["support"] >= minimum_source_support), None)
        result["reason"] = ";".join(failures) if failures else "accepted_top_source_proposal"
        if selected is None:
            return result
        result["selected"] = selected
        result["interval"] = selected["interval"]
        if result["interval"] == result["raw"]:
            result["action"] = "already_at_source_canonical_cell"
        elif selected["projection_equivalent"]:
            result["action"] = "projection_equivalent_normalization"
        else:
            result["action"] = "likelihood_compatible_edge_change"
        return result


def resolve_topology(calls):
    """Simultaneous rollback of every new overlap; preserve original call topology."""
    conflicts = set()
    for i, x in enumerate(calls):
        for j in range(i + 1, len(calls)):
            y = calls[j]
            if overlap(x["interval"], y["interval"]) > overlap(x["raw"], y["raw"]):
                conflicts.update((i, j))
    result = []
    for i, call in enumerate(calls):
        call = dict(call)
        if i in conflicts and call["interval"] != call["raw"]:
            call.update(interval=call["raw"], action="unchanged", reason="joint_topology_rollback", selected=None)
        result.append(call)
    return result


def source_detection_activities(nodes, reads, minimum_opportunities=1):
    """Attach support/eligible-unit activity to each immutable native source cell.

    Eligibility uses continuous coverage of the complete source envelope and
    opportunities in its common core. This estimates native *detection*, not
    biological occupancy. Activities define an explicit configuration prior;
    their normalization is computed by the same interval DP as the likelihood.
    A read is counted once. Recalled calls must never train these source nodes.
    """
    if len({r.unit_id for r in reads}) != len(reads):
        raise ValueError("Duplicate source evidence-unit ID")
    if not nodes:
        return []
    rect = np.asarray([n['rectangle'] for n in nodes], dtype=np.int64)
    eligible = np.zeros(len(nodes), dtype=np.int64)
    source_ids = {r.unit_id for r in reads}
    for read in reads:
        if not read.blocks:
            continue
        blocks = np.asarray(read.blocks)
        idx = np.searchsorted(blocks[:, 0], rect[:, 0], side='right') - 1
        valid = (idx >= 0) & (blocks[np.maximum(idx, 0), 1] >= rect[:, 3])
        n = np.searchsorted(read.positions, rect[:, 2]) - np.searchsorted(read.positions, rect[:, 1])
        eligible += valid & (n >= minimum_opportunities)
    output = []
    for n, count in zip(nodes, eligible):
        if len(set(n['source_units'])) != n['support'] or not set(n['source_units']) <= source_ids:
            raise ValueError("Source support provenance mismatch")
        if count < n['support']:
            raise ValueError("Source support exceeds eligible native cohort")
        output.append({**n, 'eligible_source_units': int(count),
                       'detection_activity': n['support']/int(count) if count else 0.})
    return output


def _free_domains(read, msps, additional_obstacles):
    """Continuous MSP pieces minus both raw/normalized TFs and fixed nucleosomes."""
    pieces = []
    for a, b in msps:
        for x, y in read.blocks:
            if max(a, x) < min(b, y):
                pieces.append((max(a, x), min(b, y)))
    merged = []
    for a, b in sorted(pieces):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    obstacles = read.calls + read.obstacles + [tuple(v) for v in additional_obstacles]
    for x, y in sorted(obstacles):
        fresh = []
        for a, b in merged:
            if y <= a or x >= b:
                fresh.append((a, b))
            else:
                if a < x:
                    fresh.append((a, x))
                if y < b:
                    fresh.append((y, b))
        merged = fresh
    return merged


def interval_partition(starts, ends, log_weights, minimum_gap_opportunities=1):
    """Exact Z, per-interval inclusion marginals, and MAP on one common lattice.

    An interval protects [start, end) opportunity indices. At least one observed
    opportunity separates two new calls by default. No candidate or call cap.
    The empty configuration has weight one. A coherent normalized prior is
    obtained by calling this function separately with the prior log-activities.
    """
    starts, ends = np.asarray(starts, dtype=int), np.asarray(ends, dtype=int)
    w = np.asarray(log_weights, dtype=float)
    if starts.shape != ends.shape or w.shape != starts.shape or np.any(starts >= ends):
        raise ValueError("Invalid interval arrays")
    if np.any(np.isnan(w)) or np.any(np.isposinf(w)) or minimum_gap_opportunities < 0:
        raise ValueError("Invalid configuration weights or gap")
    n = len(w)
    if not n:
        return {'log_partition': 0., 'inclusion': np.zeros(0), 'map_indices': []}
    eo = np.lexsort((starts, ends))
    so = np.lexsort((ends, starts))
    previous = np.searchsorted(ends[eo], starts-minimum_gap_opportunities, side='right')
    following = np.searchsorted(starts[so], ends+minimum_gap_opportunities, side='left')
    forward, backward = np.zeros(n+1), np.zeros(n+1)
    best, chosen = np.zeros(n+1), np.zeros(n, dtype=bool)
    for j, i in enumerate(eo):
        forward[j+1] = np.logaddexp(forward[j], w[i]+forward[previous[i]])
        value = w[i]+best[previous[i]]
        # A true tie retains the smaller configuration, including empty.
        if value > best[j]+1e-12:
            best[j+1], chosen[j] = value, True
        else:
            best[j+1] = best[j]
    for j in range(n-1, -1, -1):
        i = so[j]
        backward[j] = np.logaddexp(backward[j+1], w[i]+backward[following[i]])
    z = float(forward[-1])
    if not np.isclose(z, backward[0], atol=1e-9):
        raise AssertionError("Forward/backward partition mismatch")
    inclusion = np.exp(w+forward[previous]+backward[following]-z)
    selected, j = [], n
    while j:
        if chosen[j-1]:
            i = int(eo[j-1])
            selected.append(i)
            j = int(previous[i])
        else:
            j -= 1
    return {'log_partition': z, 'inclusion': np.clip(inclusion, 0., 1.),
            'map_indices': sorted(selected, key=lambda i:(starts[i], ends[i]))}


class FootprintRescuer:
    """Frozen opposite-strand hypotheses -> recipient evidence -> nested displays.

    Source rectangles are integrated on the recipient opportunity lattice before
    any observations are read. Identical recipient projections collapse by
    summing their source-prior mass, NOT by adding recipient evidence repeatedly.

    For configuration C, prior(C) is proportional to product(activity_g), with
    the exact nonoverlap normalizer. The likelihood relative to the common
    accessible base is exp(sum(native_LLR_g)). Source detection activity is a
    transparent provisional prior, not a claim of calibrated prevalence.
    """
    def __init__(self, source_nodes, *, minimum_source_support=3,
                 minimum_opportunities=3, prior_scale=1.):
        if minimum_source_support < 1 or minimum_opportunities < 1 or prior_scale <= 0 or not np.isfinite(prior_scale):
            raise ValueError("Invalid recall nomination parameters")
        self.nodes = source_nodes
        self.rectangles = np.asarray([n['rectangle'] for n in source_nodes], dtype=np.int64).reshape(-1, 4)
        self.minimum_source_support = minimum_source_support
        self.minimum_opportunities = minimum_opportunities
        self.prior_scale = prior_scale

    def nominate(self, read, msps, additional_obstacles=()):
        """Pure geometry/lattice operation; hits and native LLRs cannot select seeds."""
        domains = _free_domains(read, msps, additional_obstacles)
        groups, diagnostics = {}, {'source_cells_in_free_domains': 0,
                                  'under_supported_source_cells': 0,
                                  'insufficient_opportunity_projection_cells': 0}
        for lo, hi in domains:
            r = self.rectangles
            indices = np.flatnonzero((r[:, 0] < hi) & (r[:, 3] > lo))
            for ni in indices:
                node = self.nodes[int(ni)]
                if node['support'] < self.minimum_source_support:
                    diagnostics['under_supported_source_cells'] += 1
                    continue
                sl, sh, el, eh = node['rectangle']
                area = (sh-sl+1)*(eh-el+1)
                sl, sh, el, eh = max(sl, lo), min(sh, hi-1), max(el, lo+1), min(eh, hi)
                if sl > sh or el > eh:
                    continue
                diagnostics['source_cells_in_free_domains'] += 1
                for a, al, ah in projection_parts(read.positions, sl, sh):
                    for b, bl, bh in projection_parts(read.positions, el, eh):
                        if b-a < self.minimum_opportunities:
                            diagnostics['insufficient_opportunity_projection_cells'] += 1
                            continue
                        # Missing/obstacle-truncated prior mass is NOT reassigned
                        # to a favored tiny surviving geometry.
                        fraction = (ah-al+1)*(bh-bl+1)/area
                        mass = node['detection_activity']*fraction*self.prior_scale
                        if mass <= 0:
                            continue
                        key = (a, b)
                        g = groups.setdefault(key, {'projection': [a,b], 'activity': 0.,
                            'source_nodes': set(), 'best_cell_mass': -1., 'source_max_support': 0})
                        g['activity'] += mass
                        g['source_nodes'].add(int(ni))
                        g['source_max_support'] = max(g['source_max_support'], node['support'])
                        cell = [al,ah,bl,bh]
                        if mass > g['best_cell_mass'] or (mass == g['best_cell_mass'] and cell < g['joint_cell']):
                            g.update(best_cell_mass=mass, joint_cell=cell,
                                     interval=[(al+ah)//2, (bl+bh)//2])
        candidates = []
        for key, g in sorted(groups.items()):
            g['source_nodes'] = sorted(g['source_nodes'])
            g['candidate_id'] = 'R_'+sha256(repr((read.unit_id,key)).encode()).hexdigest()[:16]
            candidates.append(g)
        diagnostics['unique_recipient_projection_candidates'] = len(candidates)
        return candidates, diagnostics

    def score(self, read, candidates, *, maximum_diffuse_odds=100., minimum_native_log_lr=0.):
        """Score once, select one physical configuration, retain ambiguity.

        New-call confidence is the marginal event that a footprint protects the
        selected call's central opportunity, summed over all compatible boundary
        alternatives. It is NOT confidence in one exact edge pair. Thresholding
        this immutable configuration cannot reassign edges or create overlaps.
        """
        if maximum_diffuse_odds < 1 or minimum_native_log_lr < 0:
            raise ValueError("Invalid native recall firewall")
        if not candidates:
            return [], {'log_predictive_vs_accessible': 0., 'candidate_count': 0,
                         'selected_before_display_threshold': 0}
        starts, ends = np.asarray([c['projection'] for c in candidates], dtype=int).T
        llrs = read.prefix[ends]-read.prefix[starts]
        log_prior = np.log([c['activity'] for c in candidates])
        prior = interval_partition(starts, ends, log_prior)
        vetoed, adequacy = {}, {}
        data_weights = log_prior+llrs
        for i in np.flatnonzero(llrs > minimum_native_log_lr):
            adequacy[int(i)] = endpoint_pattern_evidence(read, *candidates[int(i)]['interval'])
            e = adequacy[int(i)]
            # Enter the veto ONCE, as zero include-likelihood. A rejected broad
            # hypothesis must not inflate the event mass of an overlapping call.
            if e['protected_log_likelihood']-e['diffuse_log_likelihood'] < -log(maximum_diffuse_odds):
                data_weights[i] = -np.inf
                vetoed[i] = 'protected_core_loses_to_diffuse_model'
        posterior = interval_partition(starts, ends, data_weights)
        action_weights = np.where(llrs > minimum_native_log_lr, data_weights, -np.inf)
        action = interval_partition(starts, ends, action_weights)
        selected = set(action['map_indices'])
        event_profiles = []
        for distribution in (posterior, prior):
            difference = np.zeros(len(read.positions)+1)
            np.add.at(difference, starts, distribution['inclusion'])
            np.add.at(difference, ends, -distribution['inclusion'])
            event_profiles.append(np.cumsum(difference)[:-1])
        output = []
        for i, c in enumerate(candidates):
            a, b = c['projection']
            anchor_index = (a+b-1)//2
            event = float(event_profiles[0][anchor_index])
            baseline = float(event_profiles[1][anchor_index])
            if event > 1+1e-8 or baseline > 1+1e-8:
                raise AssertionError("Mutually exclusive protection-event mass exceeds one")
            event, baseline = float(np.clip(event,0,1)), float(np.clip(baseline,0,1))
            status = ('native_evidence_nonpositive' if llrs[i] <= minimum_native_log_lr else
                      vetoed.get(i, 'selected' if i in selected else 'alternative_configuration'))
            output.append({**c, 'native_log_lr': float(llrs[i]), 'opportunities': int(b-a),
                'hits': int(read.hit_prefix[b]-read.hit_prefix[a]),
                'anchor': int(read.positions[anchor_index]),
                'geometry_inclusion_mass': float(posterior['inclusion'][i]),
                'protection_event_mass': event, 'prior_only_protection_event_mass': baseline,
                'model_event_Q': float(-10*np.log10(max(1-event, 1e-12))),
                'selected': i in selected, 'status': status,
                'core_adequacy': adequacy.get(i)})
        return output, {'candidate_count': len(output),
            'log_prior_partition': prior['log_partition'],
            'log_data_partition': posterior['log_partition'],
            'log_predictive_vs_accessible': posterior['log_partition']-prior['log_partition'],
            'selected_before_display_threshold': len(selected),
            'nonpositive_native_candidates': int((llrs <= minimum_native_log_lr).sum()),
            'diffuse_vetoed_candidates': len(vetoed)}

    @staticmethod
    def decode(scored, minimum_event_mass=.9):
        if not 0 <= minimum_event_mass <= 1:
            raise ValueError("Invalid display/commit probability threshold")
        return [c for c in scored if c['selected'] and c['protection_event_mass'] >= minimum_event_mass]
