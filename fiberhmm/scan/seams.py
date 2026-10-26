"""Seam reconciliation between neighbouring units.

Anchor modules (``fp``, ``nfr``): at each zone boundary, the root hits of one
type from both units that meet ``[b - window, b + window)`` form connected
components by interval overlap; each component is resolved by one unit, the
one whose centre is nearest the midpoint of the component's union (ties go to
the left unit). Hits away from boundaries are kept when their anchor lies in
their unit's zone; children follow their parent.

Stitch modules (``peaks``, ``nuc`` patches): unit tracks cover zones only and
are concatenated into territory runs of adjacent computed zones.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .types import Hit, ScanUnit

SEAM_WINDOW_BP = 2000


def _span(h: Hit) -> Tuple[int, int]:
    return (h.start, h.end) if h.end > h.start else (h.start, h.start + 1)


def _overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _components(items: List[Tuple[str, Hit]]) -> List[List[int]]:
    """Connected components (transitive overlap) of hits; returns lists of item indices."""
    n = len(items)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = sorted(range(n), key=lambda i: _span(items[i][1]))
    # sweep: union each hit with every earlier hit still open
    active: List[int] = []
    for i in order:
        s = _span(items[i][1])
        active = [j for j in active if _span(items[j][1])[1] > s[0]]
        for j in active:
            if _overlaps(_span(items[j][1]), s):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
        active.append(i)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g, key=lambda i: (_span(items[i][1]), items[i][0], items[i][1].key)) for g in
            sorted(groups.values(), key=lambda g: min(_span(items[i][1]) for i in g))]


def reconcile_anchor(results: Sequence[Tuple[ScanUnit, Sequence[Hit]]], *,
                     window: int = SEAM_WINDOW_BP,
                     missing: Iterable[str] = ()) -> Tuple[List[Tuple[ScanUnit, Hit]], int, int]:
    """Keep each feature once across unit seams.

    ``results``: every computed unit of the scan with its hits (any order).
    ``missing``: unit ids that are planned but have no result (failed or not
    run); a boundary with a missing neighbour is an open seam, and hits there
    fall back to the anchor-in-zone rule.
    Returns ``(kept, components, open_seams)``; ``kept`` is in plan order."""
    by_id = {u.unit_id: (u, list(hits)) for u, hits in results}
    missing = set(missing)
    decided: Dict[Tuple[str, str], bool] = {}
    components = 0
    open_seams = 0
    for uid, (u, hits) in sorted(by_id.items(), key=lambda kv: kv[1][0].index):
        if u.right is None:
            continue
        if u.right not in by_id:
            if u.right in missing:
                open_seams += 1
            continue
        v, vhits = by_id[u.right]
        b = u.zone_end
        win = (b - window, b + window)
        types = sorted({h.type for h in hits if h.parent is None} | {h.type for h in vhits if h.parent is None})
        for t in types:
            items = [(u.unit_id, h) for h in hits if h.parent is None and h.type == t and _overlaps(_span(h), win)]
            items += [(v.unit_id, h) for h in vhits if h.parent is None and h.type == t and _overlaps(_span(h), win)]
            if not items:
                continue
            for comp in _components(items):
                components += 1
                lo = min(_span(items[i][1])[0] for i in comp)
                hi = max(_span(items[i][1])[1] for i in comp)
                mid = (lo + hi) / 2.0
                resolver = u.unit_id if abs(mid - u.centre) <= abs(mid - v.centre) else v.unit_id
                for i in comp:
                    owner, h = items[i]
                    keep = owner == resolver
                    key = (owner, h.key)
                    decided[key] = decided.get(key, True) and keep
    kept: List[Tuple[ScanUnit, Hit]] = []
    for uid, (u, hits) in sorted(by_id.items(), key=lambda kv: kv[1][0].index):
        keep_root: Dict[str, bool] = {}
        for h in hits:
            if h.parent is not None:
                continue
            k = decided.get((uid, h.key))
            if k is None:
                k = u.zone_start <= h.anchor < u.zone_end if u.mode == 'grid' or u.left or u.right else True
            keep_root[h.key] = k
        for h in hits:
            root = h
            seen = set()
            while root.parent is not None and root.parent not in seen:
                seen.add(root.parent)
                root = next((x for x in hits if x.key == root.parent), None)
                if root is None:
                    break
            if root is not None and keep_root.get(root.key, False):
                kept.append((u, h))
    return kept, components, open_seams


def territory_runs(zones: Sequence[Tuple[ScanUnit, Optional[Mapping[str, object]], str]]):
    """Concatenate adjacent computed zones of one contig into runs.

    ``zones``: ``(unit, tracks, status)`` in zone order, ``status`` one of
    ``'ok'`` (tracks over the zone), ``'empty'`` (scanned, no molecules: a
    territory break that is not open, since zero signal there equals the zero
    padding used at run ends) and ``'missing'`` (planned but not computed).
    Returns ``(start, end, {name: array}, left_open, right_open)`` per run;
    ``*_open`` is True where the run ends next to a missing zone."""
    import numpy as np
    runs = []
    cur = None
    prev = None              # (unit, status) of the previous zone
    for unit, tracks, status in zones:
        contiguous = prev is not None and prev[0].zone_end == unit.zone_start
        if status != 'ok' or tracks is None:
            if cur is not None:
                cur['right_open'] = contiguous and status == 'missing'
                runs.append(cur)
                cur = None
            prev = (unit, status)
            continue
        if cur is not None and contiguous:
            cur['end'] = unit.zone_end
            for name, arr in tracks.items():
                cur['parts'].setdefault(name, []).append(arr)
        else:
            if cur is not None:
                runs.append(cur)
            cur = dict(start=unit.zone_start, end=unit.zone_end, parts={n: [a] for n, a in tracks.items()},
                       left_open=bool(contiguous and prev is not None and prev[1] == 'missing'), right_open=False)
        prev = (unit, status)
    if cur is not None:
        runs.append(cur)
    out = []
    for r in runs:
        arrays = {n: np.concatenate(parts) if len(parts) > 1 else parts[0] for n, parts in r['parts'].items()}
        out.append((r['start'], r['end'], arrays, r['left_open'], r['right_open']))
    return out
