"""Seam reconciliation between neighbouring units.

Anchor modules (``fp``, ``nfr``): at each zone boundary, the root hits of one
type from both units (inside the bases both computed) form connected
components by interval overlap; each component that meets
``[b - window, b + window)`` is resolved by one unit, the one whose centre is
nearest the midpoint of the component's union (ties go to the left unit). Hits away from boundaries are kept when their anchor lies in
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

    At each zone boundary, the root hits of one type from both neighbours that lie in the bases both
    computed form connected components by overlap; components that meet the boundary window are seam
    components. Seam components that share a hit (a feature reaching several seams when units overlap
    more than two deep) are merged, and each merged component is resolved once: among the neighbours of
    the seams involved, the unit whose centre is nearest the midpoint of the component's union keeps its
    hits (ties: the earlier unit in the plan), every other unit's hits in it are dropped. Root hits in no seam component are kept when
    their anchor lies in their unit's zone; children follow their parent.

    ``results``: every computed unit with its hits. ``missing``: planned units without a result; a boundary
    with a missing neighbour is an open seam, and hits there fall back to the anchor-in-zone rule.
    Returns ``(kept, components, open_seams)``; ``kept`` is in plan order."""
    by_id = {u.unit_id: (u, list(hits)) for u, hits in results}
    missing = set(missing)
    open_seams = 0
    parent: Dict[Tuple[str, str], Tuple[str, str]] = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    spans: Dict[Tuple[str, str], Tuple[int, int]] = {}
    candidates: Dict[Tuple[str, str], set] = {}      # both neighbours of every seam a hit takes part in
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
        shared = (max(u.start, v.start), min(u.end, v.end))
        types = sorted({h.type for h in hits if h.parent is None} | {h.type for h in vhits if h.parent is None})
        for t in types:
            items = [(u.unit_id, h) for h in hits if h.parent is None and h.type == t and _overlaps(_span(h), shared)]
            items += [(v.unit_id, h) for h in vhits if h.parent is None and h.type == t and _overlaps(_span(h), shared)]
            if not items:
                continue
            for comp in _components(items):
                if not any(_overlaps(_span(items[i][1]), win) for i in comp):
                    continue
                nodes = [(items[i][0], items[i][1].key) for i in comp]
                for n, i in zip(nodes, comp):
                    parent.setdefault(n, n)
                    spans[n] = _span(items[i][1])
                    candidates.setdefault(n, set()).update((u.unit_id, v.unit_id))
                for n in nodes[1:]:
                    union(nodes[0], n)
    groups: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for n in parent:
        groups.setdefault(find(n), []).append(n)
    decided: Dict[Tuple[str, str], bool] = {}
    for members in groups.values():
        lo = min(spans[n][0] for n in members)
        hi = max(spans[n][1] for n in members)
        mid = (lo + hi) / 2.0
        # Candidates: both neighbours of every seam involved, so a component that only the farther unit
        # called is dropped when the nearer unit computed the same bases and called nothing.
        units_in = set().union(*(candidates[n] for n in members))
        resolver = min(units_in, key=lambda uid: (abs(mid - by_id[uid][0].centre), by_id[uid][0].index))
        for n in members:
            decided[n] = n[0] == resolver
    components = len(groups)
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
        by_key = {x.key: x for x in hits}
        for h in hits:
            root = h
            seen = set()
            while root is not None and root.parent is not None and root.parent not in seen:
                seen.add(root.parent)
                root = by_key.get(root.parent)
            if root is not None and keep_root.get(root.key, False):
                kept.append((u, h))
    return kept, components, open_seams


def territory_runs(zones: Sequence[Tuple[ScanUnit, Optional[Mapping[str, object]], str]], pad_bp: Optional[int] = None):
    """Concatenate adjacent computed zones of one contig into runs.

    ``zones``: ``(unit, tracks, status)`` in zone order, ``status`` one of ``'ok'`` (tracks over the zone),
    ``'empty'`` (scanned, no molecules) and ``'missing'`` (planned but not computed). Each run of computed
    zones is extended into the scanned empty zones next to it with zero tracks, ``pad_bp`` on each side
    (``None``: one whole empty zone), so a feature reaching into them (a smoothing tail, a peak edge) is
    computed as if the contig were one array; runs whose extensions meet are joined, the empty bases between
    them filled with zeros. Returns ``(start, end, {name: array}, left_open, right_open)`` per run; ``*_open``
    is True where the run ends next to a missing zone, or at the edge of the plan where the last zone's unit
    has a grid neighbour that is not planned (a ``zones`` subset of the genome grid): the territory goes on
    there, unscanned."""
    import numpy as np
    zones = list(zones)
    n = len(zones)
    kind = ['ok' if (st == 'ok' and tr is not None) else ('missing' if st == 'missing' else 'empty')
            for _u, tr, st in zones]

    def contiguous(i, j):          # zone i directly followed by zone j
        return zones[i][0].zone_end == zones[j][0].zone_start

    # 1. data runs: maximal stretches of contiguous ok zones
    data = []
    i = 0
    while i < n:
        if kind[i] != 'ok':
            i += 1
            continue
        j = i
        while j + 1 < n and kind[j + 1] == 'ok' and contiguous(j, j + 1):
            j += 1
        data.append([i, j])
        i = j + 1
    if not data:
        return []

    def plan_edge_open(k, step):
        """Zone k is the last planned zone in direction step and its unit has an unplanned grid neighbour
        there (a --zones subset): the territory goes on, unscanned."""
        nxt = k + step
        if 0 <= nxt < n and (contiguous(k, nxt) if step > 0 else contiguous(nxt, k)):
            return False
        u = zones[k][0]
        return (u.right if step > 0 else u.left) is not None

    def extend(i, step):
        """Zero padding beyond zone i in direction step: (bases, reached_missing, first_zone_index_used)."""
        got, k, last = 0, i, i
        limit = pad_bp
        while True:
            nxt = k + step
            if nxt < 0 or nxt >= n or not (contiguous(k, nxt) if step > 0 else contiguous(nxt, k)):
                # the plan stops here; open when the unit has an unplanned grid neighbour on this side
                return got, plan_edge_open(k, step), last
            if kind[nxt] == 'missing':
                return got, True, last
            if kind[nxt] == 'ok':
                return got, False, last
            zl = zones[nxt][0].zone_end - zones[nxt][0].zone_start
            if limit is None:
                return zl, plan_edge_open(nxt, step), nxt
            take = min(zl, limit - got)
            got += take
            last = nxt
            if got >= limit:
                # the padding ends exactly at the plan's edge: still open there
                return got, take == zl and plan_edge_open(nxt, step), last
            k = nxt

    runs = []
    for a, b in data:
        lpad, lmiss, _ = extend(a, -1)
        rpad, rmiss, _ = extend(b, +1)
        start = zones[a][0].zone_start - lpad
        end = zones[b][0].zone_end + rpad
        runs.append(dict(a=a, b=b, start=start, end=end, left_open=lmiss, right_open=rmiss))
    # 2. join runs whose padded extents meet across contiguous empty zones
    joined = [runs[0]]
    for r in runs[1:]:
        prev = joined[-1]
        between = all(kind[k] == 'empty' for k in range(prev['b'] + 1, r['a'])) and all(
            contiguous(k, k + 1) for k in range(prev['b'], r['a']))
        if between and r['start'] <= prev['end']:
            prev['b'], prev['end'], prev['right_open'] = r['b'], r['end'], r['right_open']
        else:
            joined.append(r)
    # 3. materialise: data zones' arrays, zeros elsewhere
    out = []
    for r in joined:
        length = r['end'] - r['start']
        ok = [k for k in range(r['a'], r['b'] + 1) if kind[k] == 'ok']
        # each track's common dtype over the run's zones (as concatenation would promote), zero-filled
        dtypes = {}
        for k in ok:
            for name, arr in zones[k][1].items():
                a = np.asarray(arr)
                dtypes[name] = a.dtype if name not in dtypes else np.result_type(dtypes[name], a.dtype)
        arrays = {name: np.zeros(length, dtype=dt) for name, dt in dtypes.items()}
        for k in ok:
            u, tracks, _st = zones[k]
            lo = u.zone_start - r['start']
            for name, arr in tracks.items():
                arrays[name][lo:lo + (u.zone_end - u.zone_start)] = arr
        out.append((r['start'], r['end'], arrays, bool(r['left_open']), bool(r['right_open'])))
    return out
