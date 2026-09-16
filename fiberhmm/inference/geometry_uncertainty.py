"""Training-defined, explicitly provisional boundary-uncertainty groups.

This is NOT exact deduplication, biological identity, or an overlap merge.
All geometry nodes survive. Complete-link groups have a common protected core,
small native predictive distance, and no observed strong two-way separation.
The native joint geometry likelihood, not this grouping, scores assignments.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UncertaintyGeometry:
    geometry_id: str
    start: int
    end: int


def information_lattice(units):
    """Conservative training-cohort predictive-distance upper envelope.

    Each column uses the maximum Bernoulli Bhattacharyya information among
    training units which observe it. Summing this envelope is an upper bound
    on any one training unit's distinguishability, not population evidence.
    Outcome bits are deliberately not read here. Missingness is not agreement.
    """
    columns = {}
    for unit in units:
        p = np.asarray(unit["p_accessible"], dtype=float)
        q = np.asarray(unit["p_protected"], dtype=float)
        pos = np.asarray(unit["positions"], dtype=np.int64)
        if p.shape != q.shape or p.shape != pos.shape or np.any((p <= 0) | (p >= 1) | (q <= 0) | (q >= 1)):
            raise ValueError("Invalid native opportunity model")
        values = -np.log(np.sqrt(p*q) + np.sqrt((1-p)*(1-q)))
        for x, value in zip(pos, values):
            columns[int(x)] = max(columns.get(int(x), 0.), max(0., float(value)))
    positions = np.array(sorted(columns), dtype=np.int64)
    return positions, np.r_[0., np.cumsum([columns[int(x)] for x in positions])]


def group_uncertain_geometries(geometries, positions, information_prefix,
                               training_log_evidence, *, training_priority=None,
                               distance_limit=math.log(4.),
                               separation_log_odds=math.log(100.),
                               protection_log_odds=math.log(10.)):
    """Return a disjoint ledger of uncertainty groups, retaining every node.

    The information radius is a declared modeling choice with sensitivity
    runs, NOT a calibrated equivalence margin. Two geometries are never grouped
    if training observations strongly favor each alternative on different
    units. A saturated extension therefore cannot be hidden as edge noise.
    Complete-link and a shared core prevent transitive chaining of nearby sites.
    Training priorities only make the deterministic grouping order explicit;
    they are not likelihood multipliers or assignment quality.
    """
    gs = tuple(geometries)
    n = len(gs)
    ev = np.asarray(training_log_evidence, dtype=float)
    pos, pref = np.asarray(positions), np.asarray(information_prefix, dtype=float)
    if ev.ndim != 2 or ev.shape[1] != n or not np.all(np.isfinite(ev)):
        raise ValueError("Need finite training-units by geometries evidence")
    if pref.shape != (len(pos)+1,) or np.any(np.diff(pos) <= 0) or np.any(np.diff(pref) < -1e-12):
        raise ValueError("Invalid information lattice")
    if len({g.geometry_id for g in gs}) != n or any(g.end <= g.start for g in gs):
        raise ValueError("Invalid geometry IDs or intervals")
    if not all(math.isfinite(x) and x > 0 for x in (distance_limit, separation_log_odds, protection_log_odds)):
        raise ValueError("Positive finite grouping controls required")
    priority = np.ones(n) if training_priority is None else np.asarray(training_priority, dtype=float)
    if priority.shape != (n,) or np.any(~np.isfinite(priority)):
        raise ValueError("Invalid training priorities")
    left = np.searchsorted(pos, [g.start for g in gs])
    right = np.searchsorted(pos, [g.end for g in gs])
    il, ir = pref[left], pref[right]
    cache = {}
    def compatible(a, b):
        key = (min(a,b), max(a,b))
        if key not in cache:
            distance = abs(il[a]-il[b])+abs(ir[a]-ir[b])
            overlap = min(gs[a].end,gs[b].end) > max(gs[a].start,gs[b].start)
            # No native information is not a license for a group identity.
            informative = ir[a] > il[a] and ir[b] > il[b]
            if not overlap or not informative or distance > distance_limit:
                cache[key] = False
            else:
                delta = ev[:,a]-ev[:,b]
                favors_a = np.any((delta >= separation_log_odds) & (ev[:,a] >= protection_log_odds))
                favors_b = np.any((-delta >= separation_log_odds) & (ev[:,b] >= protection_log_odds))
                cache[key] = not (favors_a and favors_b)
        return cache[key]
    groups = []
    for j in sorted(range(n), key=lambda j: (-priority[j],gs[j].start,gs[j].end,gs[j].geometry_id)):
        candidates = []
        for k, members in enumerate(groups):
            if max([gs[j].start, *(gs[h].start for h in members)]) >= min([gs[j].end, *(gs[h].end for h in members)]):
                continue
            if all(compatible(j,h) for h in members):
                distance = max(abs(il[j]-il[h])+abs(ir[j]-ir[h]) for h in members)
                candidates.append((distance,k))
        if candidates:
            groups[min(candidates)[1]].append(j)
        else:
            groups.append([j])
    result = []
    for members in groups:
        members.sort(key=lambda j: gs[j].geometry_id)
        identity = "\0".join(gs[j].geometry_id for j in members)
        result.append({"family_id":"uncertainty_"+hashlib.sha256(identity.encode()).hexdigest()[:20],
                       "geometry_indices":members,
                       "core":[max(gs[j].start for j in members),min(gs[j].end for j in members)],
                       "envelope":[min(gs[j].start for j in members),max(gs[j].end for j in members)],
                       "identity_status":"training_defined_boundary_uncertainty_group_not_biological_identity",
                       "distance_limit":float(distance_limit),
                       "two_way_separation_log_odds":float(separation_log_odds)})
    return sorted(result,key=lambda g:(g["envelope"][0],g["envelope"][1],g["family_id"]))


def geometry_conflict_classes(geometries, groups):
    """Return homogeneous hard conflicts and mixed-normalization blocks."""
    n = len(groups)
    hard, mixed = [], []
    adjacency = [set() for _ in groups]
    starts = np.array([g.start for g in geometries])
    ends = np.array([g.end for g in geometries])
    for a in range(n):
        ga = groups[a]["geometry_indices"]
        for b in range(a):
            gb = groups[b]["geometry_indices"]
            overlap = (starts[ga,None] < ends[gb]) & (ends[ga,None] > starts[gb])
            if np.all(overlap):
                hard.append((b,a))
            elif np.any(overlap):
                mixed.append((b,a)); adjacency[a].add(b); adjacency[b].add(a)
    unseen, blocks = set(range(n)), []
    while unseen:
        stack, block = [min(unseen)], []
        while stack:
            j = stack.pop()
            if j not in unseen:
                continue
            unseen.remove(j); block.append(j); stack.extend(adjacency[j] & unseen)
        blocks.append(sorted(block))
    return {"always_conflicting":hard,"mixed_pairs":mixed,"mixed_blocks":blocks}
