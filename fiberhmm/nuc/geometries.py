"""Shared patch geometries across molecules (pool -> find -> compare).

Patches (:mod:`fiberhmm.nuc.patches`) of every molecule of one or more
datasets are pooled in reference coordinates and grouped by start AND end
(port of the explorer prototype ``ensurePatches``):

1. every observed (start, end) is a candidate centre; its group is every
   unassigned patch with both ends within ``tol_bp`` of it;
2. the candidate whose group holds the most distinct molecules wins (ties: the
   more exact repeats of the centre, then the smaller start and end);
3. the geometry's consensus span is the median start and median end of its
   member patches; its members are removed and the search repeats;
4. it stops when the best group would hold fewer than
   ``max(min_molecules, ceil(min_share x n_molecules))`` molecules (or after
   ``max_geometries``; the prototype stopped at 12, its colour count).

Why not the consensus engines? ``inference.consensus.call_clustering``
(``weighted_clusters`` / ``fold_clusters``) scores pairs by overlap over the
shorter interval, so a span nested inside a longer one has distance 0 and the
two merge, which is the nested near-duplicate case this grouping has to keep
apart and count; it is also tuned for TF calls (``max_call_bp`` 100, 10 bp edge
tolerance, lattice terms in the distance). ``lattice_recaller.discovery`` fits
k-means on edges with a prediction-strength k and merges candidates with an
exact-likelihood test on the m6A lattice (bp-resolution edge grids), which
cannot express "both ends within 500 bp, at least 3% of molecules" and does
not scale to kb-wide spans. Its k-means core is used here only as a cross-check
(:func:`kmeans_cross_check`).

Nested geometries and folding (:func:`nested_pairs`, :func:`fold_geometries`).
A geometry is *nested* in a longer one when its span lies inside the longer
span (``tol_bp`` slack on each side), one end is shared within ``tol_bp`` and
the other end has moved by more than ``tol_bp``. ``GeometryParams.fold``:

- ``none``: geometries stay separate;
- ``near``: a nested geometry folds into a parent only when its other end moved
  at most ``near_max_bp``;
- ``all``: every nested geometry folds, whatever the move.

A folded child joins its longest eligible parent, resolved transitively to the
top (root) geometry; its carriers carry the root *partly*. The child itself
must be eligible against the root by the same rule (nested in it and, for
``near``, its other end moved at most ``near_max_bp``): a chain A -> B -> C
whose A is not eligible against C leaves A unfolded (a separate geometry; the
nested pairs still list it). Grouping itself is
unchanged by folding: folding only sets ``Geometry.parent`` and the partial
state, and shares are reported full and partial separately.

Edge slivers. With a view (a BED unit, the browser's view) a patch running past
its edge is clipped there (``RefPatch.clip_left`` / ``clip_right``). Clipped
patches with less than ``GeometryParams.edge_min_bp`` inside the view are left
out of the grouping (they would pile up as short open-ended "geometries" at the
edge); they stay in the molecule's layout and in the layer.

Kinds. Patches carry their kind (high or low regularity, :mod:`fiberhmm.nuc.patches`).
Geometries are found per kind (``find_geometries(..., kind=)``): a high and a
low patch with the same span never group together, nesting and folding stay
within a kind, and low geometries are numbered L1, L2, ... (high: G1, G2, ...).

Per molecule and geometry (:func:`assign_states`): ``carries`` (a member
patch), ``partial`` (no member patch of this geometry but one of a geometry
folded into it), ``absent`` (its tiles cover the span with calls and no member patch),
``no_call`` (covers the span but more than half of the tiles there have no
score) and ``not_spanned`` (its tiles do not cover the span). Scan states:
carries = 1, absent = 0, no_call / not_spanned = '.'.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fiberhmm.nuc.patches import HIGH, IN_PATCH, LOW, NO_CALL, PATCH_KINDS, MoleculePatches, patch_ref_interval

CARRIES, ABSENT, NO_CALL_STATE, NOT_SPANNED = 'carries', 'absent', 'no_call', 'not_spanned'
PARTIAL = 'partial'
STATES = (CARRIES, PARTIAL, ABSENT, NO_CALL_STATE, NOT_SPANNED)
#: scan state values; a partial carrier does not carry the full span ('0', reason ``partial``)
SCAN_STATE = {CARRIES: '1', PARTIAL: '0', ABSENT: '0', NO_CALL_STATE: '.', NOT_SPANNED: '.'}
FOLD_MODES = ('none', 'near', 'all')


@dataclass(frozen=True)
class GeometryParams:
    #: both ends of a member within this distance of the candidate centre
    tol_bp: int = 500
    #: a geometry needs at least this share of the pooled molecules ...
    min_share: float = 0.03
    #: ... and at least this many molecules
    min_molecules: int = 3
    #: stop after this many geometries (None = no limit)
    max_geometries: Optional[int] = None
    #: group the snapped edges (False: the tile-grid edges)
    snapped: bool = True
    #: fold nested geometries into their parents: 'none', 'near' (other end moved <= near_max_bp) or 'all'
    fold: str = 'near'
    #: the largest move of the other end for fold 'near' (bp)
    near_max_bp: int = 1250
    #: a patch cut by the view / BED-unit edge joins the grouping only when at least this much of it (bp) lies
    #: inside; shorter slivers would pile up as edge "geometries" (0 = keep every patch). Still in the layer.
    edge_min_bp: int = 500

    def __post_init__(self):
        if self.tol_bp < 0 or not 0 <= self.min_share < 1 or self.min_molecules < 1:
            raise ValueError('nucgeo: invalid tolerance, share or molecule count')
        if self.fold not in FOLD_MODES:
            raise ValueError(f'nucgeo: fold must be one of {FOLD_MODES}')
        if self.near_max_bp < 0:
            raise ValueError('nucgeo: near_max_bp must not be negative')
        if self.edge_min_bp < 0:
            raise ValueError('nucgeo: edge_min_bp must not be negative')
        if self.max_geometries is not None and self.max_geometries < 1:
            raise ValueError('nucgeo: max_geometries must be positive (None = no limit)')

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "GeometryParams":
        data = dict(data or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f'unknown nucgeo parameter(s): {", ".join(map(repr, unknown))}')
        return cls(**data)


DEFAULT_GEOMETRIES = GeometryParams()


@dataclass
class RefPatch:
    """A patch in reference coordinates (``open_left``/``open_right`` in reference orientation)."""
    start: int
    end: int
    open_left: bool
    open_right: bool
    mean_score: float
    period: int
    kind: str = HIGH
    #: cut by the view / unit edge on that side (open there because of the clipping, not the read end)
    clip_left: bool = False
    clip_right: bool = False

    @property
    def clipped(self) -> bool:
        return self.clip_left or self.clip_right


def groupable(p: RefPatch, params: "GeometryParams") -> bool:
    """Whether a patch joins the grouping: a patch cut by the view edge needs ``edge_min_bp`` inside the view."""
    return not (p.clip_left or p.clip_right) or (p.end - p.start) >= params.edge_min_bp


@dataclass
class MoleculeLayout:
    """One molecule's patches and scored-tile layout in reference coordinates."""
    uid: str
    dataset: str
    patches: List[RefPatch]
    tile_lo: np.ndarray            # reference tile envelopes, sorted
    tile_hi: np.ndarray
    tile_state: np.ndarray         # IN_PATCH / IRREGULAR / NO_CALL per tile
    density_per_kb: float = float('nan')
    meta: dict = field(default_factory=dict)

    @property
    def span(self) -> Tuple[int, int]:
        if not len(self.tile_lo):
            return (0, 0)
        return int(self.tile_lo[0]), int(self.tile_hi[-1])


@dataclass
class Geometry:
    geometry_id: str
    start: int
    end: int
    open_left: bool
    open_right: bool
    n_molecules: int
    n_patches: int
    members: List[Tuple[int, int]]         # (layout index, patch index)
    start_mad: float
    end_mad: float
    mean_score: float
    period: int
    #: index of the root geometry this one is folded into (None: not folded)
    parent: Optional[int] = None
    #: the parent's geometry_id (None: not folded)
    parent_id: Optional[str] = None
    #: 'high' or 'low' regularity
    kind: str = HIGH

    @property
    def width(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class NestedPair:
    child: int                 # index of the shorter geometry
    parent: int                # index of the longer one
    shared: str                # 'start' or 'end' (reference orientation)
    moved: int                 # bp the other end moved


def molecule_layout(mol, track, mp: MoleculePatches, dataset: str = '', *, snapped: bool = True,
                    view: Optional[Tuple[int, int]] = None, uid: Optional[str] = None) -> MoleculeLayout:
    """Project one molecule's patches and tiles onto the reference.

    With ``view`` (reference ``(start, end)``) tiles and patches are clipped to it,
    and a patch cut by a view edge is open on that side (the prototype's
    windowed cohort)."""
    lo, hi, st = [], [], []
    for i in range(len(track.tile_starts)):
        env = mol.mol_interval_to_ref(int(track.tile_starts[i]), int(track.tile_ends[i]))
        if env is None:
            continue
        a, b = env
        if view is not None:
            a, b = max(a, view[0]), min(b, view[1])
            if b <= a:
                continue
        lo.append(a); hi.append(b); st.append(int(mp.tile_state[i]))
    order = np.argsort(np.asarray(lo, dtype=np.int64), kind='stable')
    lo = np.asarray(lo, dtype=np.int64)[order]
    hi = np.asarray(hi, dtype=np.int64)[order]
    st = np.asarray(st, dtype=np.int8)[order]
    pts = []
    for p in mp.patches:
        r = patch_ref_interval(mol, p, snapped)
        if r is None:
            continue
        a, b, ol, orr = r
        cl = cr = False
        if view is not None:
            if b <= view[0] or a >= view[1]:
                continue
            if a < view[0]:
                a, ol, cl = view[0], True, True
            if b > view[1]:
                b, orr, cr = view[1], True, True
        pts.append(RefPatch(a, b, bool(ol), bool(orr), p.mean_score, p.period, getattr(p, 'kind', HIGH), cl, cr))
    pts.sort(key=lambda x: (x.start, x.end))
    return MoleculeLayout(uid if uid is not None else getattr(mol, 'name', ''), dataset, pts, lo, hi, st,
                          float(getattr(track, 'density_per_kb', float('nan'))))


def _mad(x: np.ndarray) -> float:
    return float(np.median(np.abs(x - np.median(x)))) if len(x) else float('nan')


def _box_count(R, order, S, left, a, b, tol) -> int:
    """Distinct molecules among the unassigned patches with both ends within ``tol`` of (a, b)."""
    lo = np.searchsorted(S, a - tol, side='left')
    hi = np.searchsorted(S, a + tol, side='right')
    sl = order[lo:hi]
    sl = sl[left[sl] & (np.abs(R[sl, 3] - b) <= tol)]
    return int(len(np.unique(R[sl, 0]))) if len(sl) else 0


def find_geometries(layouts: Sequence[MoleculeLayout], params: GeometryParams = DEFAULT_GEOMETRIES,
                    n_molecules: Optional[int] = None, chrom: str = '', kind: str = HIGH) -> List[Geometry]:
    """Shared geometries of the pooled molecules' patches of one ``kind`` (``n_molecules`` defaults to
    ``len(layouts)``).

    Each candidate's distinct-molecule count is computed once and then updated only for the candidates near the
    patches a new geometry removes (the result equals recounting every candidate in every round)."""
    if kind not in PATCH_KINDS:
        raise ValueError(f'nucgeo: kind must be one of {PATCH_KINDS}')
    rows = [(i, k, p.start, p.end) for i, lay in enumerate(layouts) for k, p in enumerate(lay.patches)
            if getattr(p, 'kind', HIGH) == kind and groupable(p, params)]
    if not rows:
        return []
    R = np.asarray(rows, dtype=np.int64)
    n_mol = len(layouts) if n_molecules is None else int(n_molecules)
    need = max(int(params.min_molecules), int(math.ceil(params.min_share * n_mol - 1e-9)))
    tol = int(params.tol_bp)
    left = np.ones(len(R), dtype=bool)
    U, inv = np.unique(R[:, 2:4], axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    reps = np.bincount(inv, minlength=len(U)).astype(np.int64)
    order = np.argsort(R[:, 2], kind='stable')
    S = R[order, 2]
    cnt = np.array([_box_count(R, order, S, left, int(u[0]), int(u[1]), tol) for u in U], dtype=np.int64)
    out: List[Geometry] = []
    while left.any() and (params.max_geometries is None or len(out) < params.max_geometries):
        alive = np.flatnonzero(reps > 0)
        if not len(alive):
            break
        # the best candidate: most molecules, then most exact repeats, then the smaller start and end
        j = alive[np.lexsort((-U[alive, 1], -U[alive, 0], reps[alive], cnt[alive]))[-1]]
        if cnt[j] < need:
            break
        a, b = int(U[j, 0]), int(U[j, 1])
        idx = np.flatnonzero(left)
        inbox = (np.abs(R[idx, 2] - a) <= tol) & (np.abs(R[idx, 3] - b) <= tol)
        mem = idx[inbox]
        left[mem] = False
        reps -= np.bincount(inv[mem], minlength=len(U))
        starts, ends = R[mem, 2], R[mem, 3]
        near = np.flatnonzero((reps > 0) & (U[:, 0] >= starts.min() - tol) & (U[:, 0] <= starts.max() + tol)
                              & (U[:, 1] >= ends.min() - tol) & (U[:, 1] <= ends.max() + tol))
        for c in near:
            cnt[c] = _box_count(R, order, S, left, int(U[c, 0]), int(U[c, 1]), tol)
        ps = [layouts[int(i)].patches[int(k)] for i, k, _a, _b in R[mem]]
        cs, ce = int(round(float(np.median(starts)))), int(round(float(np.median(ends))))
        sc = np.array([p.mean_score for p in ps], dtype=float)
        pr = np.array([p.period for p in ps if p.period > 0])
        tag = f'{chrom}:{cs}-{ce}:{len(out)}' if kind == HIGH else f'low:{chrom}:{cs}-{ce}:{len(out)}'
        gid = hashlib.sha256(tag.encode()).hexdigest()[:12]
        out.append(Geometry(f'{"G" if kind == HIGH else "L"}{len(out) + 1}_{gid}', cs, ce,
                            sum(p.open_left for p in ps) * 2 > len(ps),
                            sum(p.open_right for p in ps) * 2 > len(ps),
                            int(len(np.unique(R[mem, 0]))), int(len(mem)),
                            [(int(i), int(k)) for i, k in R[mem, :2]],
                            _mad(starts.astype(float)), _mad(ends.astype(float)),
                            float(np.nanmedian(sc)) if np.isfinite(sc).any() else float('nan'),
                            int(np.median(pr)) if len(pr) else -1, kind=kind))
    fold_geometries(out, params)
    return out


def _nested(c: Geometry, q: Geometry, tol_bp: int) -> Optional[NestedPair]:
    """The nesting of ``c`` in the longer ``q`` (module docstring), or None (indices left 0)."""
    if q.width <= c.width or c.kind != q.kind:
        return None
    if c.start < q.start - tol_bp or c.end > q.end + tol_bp:
        return None
    ds, de = abs(c.start - q.start), abs(c.end - q.end)
    shared = 'start' if ds <= tol_bp else 'end' if de <= tol_bp else None
    if shared is None:
        return None
    moved = de if shared == 'start' else ds
    if moved <= tol_bp:
        return None
    return NestedPair(0, 0, shared, int(moved))


def nested_pairs(geometries: Sequence[Geometry], tol_bp: int = 500) -> List[NestedPair]:
    """Every (child, parent) pair of one kind where the child is nested in the longer parent (module docstring)."""
    out = []
    for ci, c in enumerate(geometries):
        for qi, q in enumerate(geometries):
            if ci == qi:
                continue
            z = _nested(c, q, tol_bp)
            if z is not None:
                out.append(NestedPair(ci, qi, z.shared, z.moved))
    return out


def fold_geometries(geometries: List[Geometry], params: GeometryParams = DEFAULT_GEOMETRIES) -> List[NestedPair]:
    """Set ``parent`` / ``parent_id`` on folded geometries in place (``params.fold``); returns the folded pairs."""
    for g in geometries:
        g.parent = g.parent_id = None
    if params.fold == 'none' or len(geometries) < 2:
        return []
    pairs = nested_pairs(geometries, params.tol_bp)
    if params.fold == 'near':
        pairs = [z for z in pairs if z.moved <= params.near_max_bp]
    direct: Dict[int, int] = {}
    for ci in range(len(geometries)):
        ps = [z.parent for z in pairs if z.child == ci]
        if ps:
            # the longest eligible parent (ties: the earlier, i.e. better supported, geometry)
            direct[ci] = max(ps, key=lambda qi: (geometries[qi].width, -qi))

    def eligible(ci, qi) -> bool:
        z = _nested(geometries[ci], geometries[qi], params.tol_bp)
        return z is not None and (params.fold == 'all' or z.moved <= params.near_max_bp)
    folded = {}
    for ci, pi in direct.items():
        seen = {ci}
        while pi in direct and pi not in seen:      # transitively to the root (parents are strictly longer)
            seen.add(pi)
            pi = direct[pi]
        if pi == direct[ci] or eligible(ci, pi):    # the child itself must fold into the root (near_max_bp)
            folded[ci] = pi
    for ci, pi in folded.items():
        geometries[ci].parent = pi
        geometries[ci].parent_id = geometries[pi].geometry_id
    return [z for z in pairs if z.child in folded and direct.get(z.child) == z.parent]


def children_of(geometries: Sequence[Geometry]) -> Dict[int, List[int]]:
    """Root index -> indices of the geometries folded into it."""
    out: Dict[int, List[int]] = {}
    for i, g in enumerate(geometries):
        if g.parent is not None:
            out.setdefault(g.parent, []).append(i)
    return out


def molecule_state(lay: MoleculeLayout, g: Geometry, carriers: set, index: int,
                   partial_carriers: frozenset = frozenset()) -> str:
    if index in carriers:
        return CARRIES
    if index in partial_carriers:
        return PARTIAL
    if not len(lay.tile_lo):
        return NOT_SPANNED
    s0, s1 = lay.span
    if s0 > g.start or s1 < g.end:
        return NOT_SPANNED
    inside = (lay.tile_hi > g.start) & (lay.tile_lo < g.end)
    if not inside.any():
        return NOT_SPANNED
    if (lay.tile_state[inside] == NO_CALL).sum() * 2 > inside.sum():
        return NO_CALL_STATE
    return ABSENT


def assign_states(layouts: Sequence[MoleculeLayout], geometries: Sequence[Geometry]) -> List[Dict[str, str]]:
    """Per molecule: ``{geometry_id: state}`` (see the module docstring)."""
    out: List[Dict[str, str]] = [dict() for _ in layouts]
    kids = children_of(geometries)
    for gi, g in enumerate(geometries):
        carriers = {i for i, _k in g.members}
        part = frozenset(i for c in kids.get(gi, ()) for i, _k in geometries[c].members) - carriers
        for i, lay in enumerate(layouts):
            out[i][g.geometry_id] = molecule_state(lay, g, carriers, i, part)
    return out


def support(layouts: Sequence[MoleculeLayout], states: Sequence[Dict[str, str]], geometry_id: str,
            dataset: Optional[str] = None) -> dict:
    """Carriers (full), partial carriers (a folded child only), callable molecules (carries + partial +
    absent) and the shares, optionally for one dataset."""
    sel = [i for i, lay in enumerate(layouts) if dataset is None or lay.dataset == dataset]
    n = len(sel)
    c = sum(states[i][geometry_id] == CARRIES for i in sel)
    pa = sum(states[i][geometry_id] == PARTIAL for i in sel)
    callable_ = sum(states[i][geometry_id] in (CARRIES, PARTIAL, ABSENT) for i in sel)
    return dict(molecules=n, carriers=c, partial=pa, callable=callable_,
                share=c / n if n else None, share_partial=pa / n if n else None,
                share_callable=c / callable_ if callable_ else None,
                share_partial_callable=pa / callable_ if callable_ else None)


def density_matched_shares(layouts: Sequence[MoleculeLayout], states: Sequence[Dict[str, str]],
                           geometry_id: str, datasets: Sequence[str], n_bins: int = 5) -> Dict[str, Optional[float]]:
    """Carrier share per dataset standardised to the pooled m6A/kb distribution.

    Molecules of all ``datasets`` are cut into ``n_bins`` pooled density quantile
    bins; each dataset's within-bin share among callable molecules is weighted by
    the pooled bin sizes. Bins where a dataset has fewer than 5 callable
    molecules are left out for every dataset."""
    idx = [i for i, lay in enumerate(layouts) if lay.dataset in datasets and np.isfinite(lay.density_per_kb)
           and states[i][geometry_id] in (CARRIES, PARTIAL, ABSENT)]
    if not idx:
        return {d: None for d in datasets}
    dens = np.array([layouts[i].density_per_kb for i in idx])
    edges = np.quantile(dens, np.linspace(0, 1, n_bins + 1)[1:-1])
    b = np.searchsorted(edges, dens, side='right')
    num = {d: 0.0 for d in datasets}
    w = 0.0
    for k in range(n_bins):
        inb = [idx[j] for j in np.flatnonzero(b == k)]
        per = {d: [i for i in inb if layouts[i].dataset == d] for d in datasets}
        if any(len(v) < 5 for v in per.values()):
            continue
        for d, v in per.items():
            num[d] += len(inb) * sum(states[i][geometry_id] == CARRIES for i in v) / len(v)
        w += len(inb)
    return {d: (num[d] / w if w else None) for d in datasets}


# ---------------------------------------------------------------------------- evaluation
def nested_near_duplicates(geometries: Sequence[Geometry], tol_bp: int = 500,
                           smear_bp: int = 1250) -> List[Tuple[str, str]]:
    """Pairs of geometries that look like one array found twice with one end moved by the window smear.

    One span lies inside the other (``tol_bp`` slack), they share one end within
    ``tol_bp`` and the other end differs by more than ``tol_bp`` (otherwise the
    grouping would have joined them) and at most ``smear_bp`` (half the 2 kb
    window plus one step: how far a tile-grid edge can drift from the array's
    edge)."""
    out = []
    for i, g in enumerate(geometries):
        for h in geometries[i + 1:]:
            inner, outer = (g, h) if g.width <= h.width else (h, g)
            nested = outer.start - tol_bp <= inner.start and inner.end <= outer.end + tol_bp
            ds, de = abs(g.start - h.start), abs(g.end - h.end)
            if nested and min(ds, de) <= tol_bp and tol_bp < max(ds, de) <= smear_bp:
                out.append((g.geometry_id, h.geometry_id))
    return out


def find_all_geometries(layouts: Sequence[MoleculeLayout], params: GeometryParams = DEFAULT_GEOMETRIES,
                        n_molecules: Optional[int] = None, chrom: str = '',
                        kinds: Sequence[str] = (HIGH, LOW)) -> List[Geometry]:
    """High geometries, then low ones (each kind found and folded on its own)."""
    out: List[Geometry] = []
    for kind in kinds:
        G = find_geometries(layouts, params, n_molecules, chrom, kind=kind)
        off = len(out)
        for g in G:
            if g.parent is not None:
                g.parent += off
        out += G
    return out


def geometry_metrics(layouts: Sequence[MoleculeLayout], geometries: Sequence[Geometry],
                     tol_bp: int = 500) -> dict:
    """Count, share of patches assigned, nested near-duplicate pairs and edge spread."""
    n_p = sum(len(l.patches) for l in layouts)
    n_a = sum(g.n_patches for g in geometries)
    w = np.array([g.n_patches for g in geometries], dtype=float)
    smad = np.array([g.start_mad for g in geometries])
    emad = np.array([g.end_mad for g in geometries])
    closed_s = np.array([not g.open_left for g in geometries], dtype=bool)
    closed_e = np.array([not g.open_right for g in geometries], dtype=bool)
    mads = np.r_[smad[closed_s], emad[closed_e]] if len(geometries) else np.zeros(0)
    wts = np.r_[w[closed_s], w[closed_e]] if len(geometries) else np.zeros(0)
    nd = nested_near_duplicates(geometries, tol_bp)
    return dict(patches=n_p, geometries=len(geometries), assigned=n_a,
                share_assigned=n_a / n_p if n_p else None,
                nested_pairs=len(nd), nested_geometries=len({x for p in nd for x in p}),
                edge_mad_bp=float(np.average(mads, weights=wts)) if wts.sum() > 0 else None)


def kmeans_cross_check(layouts: Sequence[MoleculeLayout], seed: int = 0, kmax: int = 20,
                       stringency: float = 0.8, splits: int = 3) -> dict:
    """The lattice recaller's k-means / prediction-strength core on bare (start, end) features.

    A cross-check only: returns the chosen k and the prediction strength per k."""
    from types import SimpleNamespace
    from fiberhmm.inference.consensus.lattice_recaller.discovery import _nominate
    X, meta, groups = [], [], []
    for i, lay in enumerate(layouts):
        for p in lay.patches:
            X.append((p.start, p.end)); meta.append((i, p.start, p.end, p.start, p.start, p.end, p.end))
            groups.append(lay.uid or str(i))
    if len(X) < 2:
        return dict(k=0, prediction_strength=[])
    opt = SimpleNamespace(kmax=kmax, seed=seed, prediction_splits=splits, stringency=stringency,
                          minimum_candidate_calls=1)
    cands, k, choice = _nominate(np.asarray(X, float), meta, groups, opt)
    return dict(k=int(k), prediction_strength=[[int(kk), round(float(ps), 3)] for kk, ps in choice],
                spans=sorted([[int(np.median(c['X'][:, 0])), int(np.median(c['X'][:, 1])), len(c['calls'])]
                              for c in cands]))
