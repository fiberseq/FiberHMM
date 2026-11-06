"""Per-molecule regularity patches (MA group ``nucpatch``).

Discrete calls on a molecule's regularity track (:mod:`fiberhmm.nuc.regularity`),
so regular stretches can be joined with other features (TF occupancy, NFR
state, shared geometries in :mod:`fiberhmm.nuc.geometries`).

Calling (port of the explorer prototype ``callPatches``). Scores are read as
excess over the dataset's random-packing reference median (``ref_median``; with
``ends`` 'truncated' and the reference's width deltas, ``ref_widths``, each tile
against the reference of its own window width, see
:func:`fiberhmm.nuc.regularity.tile_reference`):

- a tile with no score is *bridged* when both neighbours have a score: for the
  threshold tests it takes their mean (it never enters the patch's mean score);
  two or more consecutive no-call tiles end a patch and are "no call", not
  irregular;
- a patch is a maximal run of tiles whose (bridged) excess is at least
  ``extend`` that holds at least one tile at or above ``enter`` and at least
  ``min_tiles`` tiles (hysteresis: start above +2.0, extend while above +1.25);
- a patch touching the molecule's first or last tile is open-ended on that side;
- attributes: mean raw score over its scored tiles, median period over its
  tiles with a period.

Edges. The tile-grid edges are ``tile_starts[first]`` and ``tile_ends[last]``
(first tile centre - step/2, last tile centre + step/2). With ``snap`` (default)
each closed edge moves to a nucleosome boundary (see :func:`snap_edge`):

- footprints are the molecule's nucleosome calls of at least ``nuc_min_bp``;
- a linker between consecutive footprints *breaks* the spacing when their dyad
  distance differs from the patch period by more than ``tol_frac`` x period. A
  pair with a footprint of at least ``merged_bp`` (two nucleosomes in one call)
  is judged by its edge-to-edge linker instead, which breaks when it is longer
  than ``period - core_bp + tol``. Beyond the outermost footprint of the
  molecule counts as a break;
- the edge moves to the break linker nearest the grid edge within
  ``search_bp``, onto the boundary of the footprint on the patch side of that
  linker (the outermost nucleosome inside the patch);
- with no break within ``search_bp``, or an edge that would cross the other
  edge, the grid edge stays and the edge is flagged not snapped;
- a snapped edge never crosses a neighbouring patch of its kind: an edge
  snapped into or past a neighbour (its grid span or its snapped edge) goes
  back to its grid edge and is flagged not snapped. A low-regularity patch is
  also clipped at a high-regularity neighbour's edge (flagged not snapped;
  high patches stay exactly as high-only calling makes them). Patches of a
  molecule therefore never overlap.

Open-ended edges are never snapped. The grid edges are always kept on the
patch (``grid_start``/``grid_end``).

Coordinates are molecular (the track's); :func:`patch_ref_interval` projects a
patch onto the reference (reverse records flip start/end).

Two kinds of patch. *High-regularity* patches (above) are the default. With
``PatchParams.low`` (:class:`LowPatchParams`) the same molecule is also called
for *low-regularity* patches by the mirror hysteresis on the low side: a run of
tiles whose (bridged) excess is at most ``low.extend`` holding one tile at or
below ``low.enter`` and at least ``low.min_tiles`` tiles. Low regularity is
chromatin less regular than random packing of the same nucleosomes, a real
state of the fibre and not a failed call. The two kinds never overlap
(``low.extend`` must lie below the high ``extend``; snapped edges are clamped as
above). Low patches have their own
edge settings; with the 'open' snap rule they snap without a period (the rule
does not use one), with the 'period' rule only when the patch has a period.

The ``nucpatch`` group stores one interval per patch (snapped edges): Q bytes
``[score_q, period_q, flags]`` with the ``nucreg`` score and period codecs and
``flags`` bit 0/1 = open at the molecular start/end, bit 2/3 = start/end
snapped, bit 4 = a low-regularity patch. With low calling off no patch has bit
4 and the parameters (and so the digests) are those of high-only calling, so
the output is byte-identical to the high-only group.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import List, Optional, Sequence, Tuple

import numpy as np

from fiberhmm.nuc.regularity import (RegularityTrack, decode_period, decode_score, encode_period,
                                     encode_score)

#: per-tile states of :func:`call_patches`: no score, scored outside every patch, in a high-regularity patch,
#: in a low-regularity patch
NO_CALL, OUTSIDE, IN_PATCH, IN_LOW_PATCH = -1, 0, 1, 2
#: former name of OUTSIDE (a scored tile in no patch; not a low-regularity call)
IRREGULAR = OUTSIDE
#: flag bits of the third Q byte
OPEN_START, OPEN_END, SNAPPED_START, SNAPPED_END, LOW_KIND = 1, 2, 4, 8, 16
SNAP_RULES = ('open', 'period')
#: patch kinds
HIGH, LOW = 'high', 'low'
PATCH_KINDS = (HIGH, LOW)


@dataclass(frozen=True)
class PatchParams:
    #: a patch needs one tile at least this far above the reference median (log2)
    enter: float = 2.0
    #: and extends over tiles at least this far above it
    extend: float = 1.25
    min_tiles: int = 2
    #: snap closed edges to nucleosome boundaries
    snap: bool = True
    #: nucleosome-sized footprints (nucleosome calls of at least this length)
    nuc_min_bp: int = 90
    #: break linkers are searched within this distance of the grid edge
    search_bp: int = 1000
    #: a linker breaks when the dyad distance is off the period by more than tol_frac x period
    tol_frac: float = 0.25
    #: footprints at least this long hold two nucleosomes: judged by the linker length
    merged_bp: int = 230
    #: nucleosome core length used for the expected linker (period - core_bp)
    core_bp: int = 147
    #: which linkers count as breaks: 'open' (at least open_bp long) or 'period' (spacing off the period)
    snap_rule: str = 'open'
    #: an open linker for snap_rule 'open'
    open_bp: int = 300
    #: low-regularity patches (LowPatchParams); None = high-regularity patches only
    low: Optional["LowPatchParams"] = None

    def _check_thresholds(self):
        if not self.extend <= self.enter:
            raise ValueError('nucpatch: need extend <= enter')

    def __post_init__(self):
        self._check_thresholds()
        if self.min_tiles < 1:
            raise ValueError('nucpatch: min_tiles must be at least 1')
        if self.nuc_min_bp < 1 or self.search_bp < 0 or self.merged_bp <= self.nuc_min_bp or self.core_bp < 1:
            raise ValueError('nucpatch: invalid footprint sizes or search distance')
        if not 0 < self.tol_frac < 1:
            raise ValueError('nucpatch: tol_frac must be in (0, 1)')
        if self.snap_rule not in SNAP_RULES:
            raise ValueError(f'nucpatch: snap_rule must be one of {SNAP_RULES}')
        if self.open_bp < 1:
            raise ValueError('nucpatch: open_bp must be positive')
        if self.low is not None:
            if isinstance(self, LowPatchParams) or not isinstance(self.low, LowPatchParams):
                raise ValueError('nucpatch: low must be LowPatchParams (and only on the high parameters)')
            if not self.low.extend < self.extend:
                raise ValueError('nucpatch: the low extend threshold must lie below the high extend threshold')

    def to_dict(self) -> dict:
        """The parameters; ``low`` only when low calling is on (so high-only digests are unchanged)."""
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name != 'low'}
        if self.low is not None:
            d['low'] = self.low.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "PatchParams":
        if not data:
            return cls()
        data = dict(data)
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f'unknown nucpatch parameter(s): {", ".join(map(repr, unknown))}')
        low = data.pop('low', None)
        if low is not None:
            if cls is LowPatchParams:
                raise ValueError('nucpatch: low parameters cannot nest')
            data['low'] = low if isinstance(low, LowPatchParams) else LowPatchParams.from_dict(low)
        return cls(**data)

    @property
    def kinds(self) -> Tuple[str, ...]:
        return (HIGH, LOW) if self.low is not None else (HIGH,)

    def digest(self, ref_median: Optional[float] = None, ref_widths: Optional[dict] = None) -> str:
        """Full SHA-256 of the parameters and the reference median the excess is read against (and, only when
        given, the reference's per-width deltas, so digests of full-window references are unchanged)."""
        d = self.to_dict()
        d['ref_median'] = None if ref_median is None else round(float(ref_median), 4)
        if ref_widths and ref_widths.get('deltas'):
            d['ref_widths'] = [int(ref_widths.get('window_bp') or 0), int(ref_widths.get('step') or 0),
                               sorted([int(w), round(float(v), 4)] for w, v in ref_widths['deltas'].items())]
            if ref_widths.get('centres'):
                # where the deltas are placed (the classes' sampled widths) changes the tiles' references too
                d['ref_widths'].append(sorted([int(w), round(float(v), 1)]
                                              for w, v in ref_widths['centres'].items()))
        blob = json.dumps(d, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(blob.encode()).hexdigest()


@dataclass(frozen=True)
class LowPatchParams(PatchParams):
    """Low-regularity patches: the mirror hysteresis (start at or below ``enter``, extend while at or below
    ``extend``, both log2 excess over the random-packing reference median) with their own edge settings.

    Provisional defaults (embryo Hia5 PacBio): ``enter`` -1.5 is about the random-packing reference's own 7th
    percentile and ``extend`` -0.75 its 25th, the same 0.75 hysteresis gap as the high side (+2.0 / +1.25)."""
    enter: float = -1.5
    extend: float = -0.75

    def _check_thresholds(self):
        if not self.enter <= self.extend:
            raise ValueError('nucpatch low: need enter <= extend')


DEFAULT_PATCHES = PatchParams()
DEFAULT_LOW_PATCHES = LowPatchParams()


def patch_digest(regularity_digest: str, params: PatchParams, ref_median: Optional[float],
                 ref_widths: Optional[dict] = None) -> str:
    """8 hex naming the patch parameters, the regularity parameters (``RegularityTrack.params_digest``) and
    the reference (median, and per-width deltas when given): the ``nucpatch`` AN params digest."""
    return hashlib.sha256(f'{regularity_digest[:8]}|{params.digest(ref_median, ref_widths)}'.encode()).hexdigest()[:8]


@dataclass
class Patch:
    """One regular patch of a molecule (molecular coordinates, half-open)."""
    start: int
    end: int
    grid_start: int
    grid_end: int
    first_tile: int
    last_tile: int
    mean_score: float              # raw log2 score, mean over scored tiles
    period: int                    # median tile period (bp), -1 = none
    open_start: bool = False
    open_end: bool = False
    snapped_start: bool = False
    snapped_end: bool = False
    #: 'high' or 'low' regularity
    kind: str = HIGH

    @property
    def n_tiles(self) -> int:
        return self.last_tile - self.first_tile + 1

    @property
    def flags(self) -> int:
        return (OPEN_START * self.open_start | OPEN_END * self.open_end
                | SNAPPED_START * self.snapped_start | SNAPPED_END * self.snapped_end
                | LOW_KIND * (self.kind == LOW))

    def edges(self, snapped: bool = True) -> Tuple[int, int]:
        return (self.start, self.end) if snapped else (self.grid_start, self.grid_end)


@dataclass
class MoleculePatches:
    """Patches of one molecule (both kinds, sorted by start) and the per-tile states (``IN_PATCH``,
    ``IN_LOW_PATCH``, ``OUTSIDE``, ``NO_CALL``)."""
    patches: List[Patch]
    tile_state: np.ndarray
    ref_median: float
    params_digest: str = ''


def bridged_excess(track: RegularityTrack, ref_median, ref_widths: Optional[dict] = None) -> np.ndarray:
    """Excess over the reference per tile (``ref_median``, per window width with ``ref_widths``); a lone no-score
    tile between two scored ones takes their mean."""
    from fiberhmm.nuc.regularity import tile_reference
    v = np.asarray(track.score, dtype=float) - tile_reference(track.window_widths, ref_median, ref_widths)
    w = v.copy()
    n = len(v)
    if n >= 3:
        miss = ~np.isfinite(v)
        mid = np.flatnonzero(miss[1:-1]) + 1
        ok = np.isfinite(v[mid - 1]) & np.isfinite(v[mid + 1])
        j = mid[ok]
        w[j] = (v[j - 1] + v[j + 1]) / 2.0
    return w


def call_patches(track: RegularityTrack, nucs: Optional[np.ndarray] = None, *, ref_median: float,
                 params: PatchParams = DEFAULT_PATCHES, ref_widths: Optional[dict] = None) -> MoleculePatches:
    """Patches of one molecule from its regularity track and nucleosome calls (``(n, 2)``, molecular):
    high-regularity patches, and low-regularity ones when ``params.low`` is set. ``ref_widths``: the reference's
    per-width deltas (:func:`fiberhmm.nuc.regularity.reference_widths`); tiles of cut windows (ends 'truncated')
    are then read against the reference of their own width."""
    from fiberhmm.nuc.regularity import tile_reference
    ref_tile = tile_reference(track.window_widths, ref_median, ref_widths)
    v = np.asarray(track.score, dtype=float) - ref_tile
    w = bridged_excess(track, ref_median, ref_widths)
    called = np.isfinite(w)
    state = np.where(called, OUTSIDE, NO_CALL).astype(np.int8)
    out = _call_kind(track, v, w, called, state, nucs, params, HIGH)
    _clamp_neighbours(out)
    if params.low is not None:
        low = _call_kind(track, v, w, called, state, nucs, params.low, LOW)
        _clamp_neighbours(low)
        out += _clamp_low(low, out, state)
        out.sort(key=lambda p: (p.start, p.end))
    widths = ref_widths if track.window_widths is not None else None
    return MoleculePatches(out, state, float(ref_median),
                           patch_digest(track.params_digest, params, ref_median, widths))


def _call_kind(track, v, w, called, state, nucs, params: PatchParams, kind: str) -> List[Patch]:
    """One kind's patches (hysteresis above for 'high', mirrored below for 'low'); marks their tiles in ``state``."""
    n = len(w)
    if kind == HIGH:
        ext = called & (w >= params.extend)
        hit = lambda run: (w[run] >= params.enter).any()          # noqa: E731
        mark = IN_PATCH
    else:
        ext = called & (w <= params.extend)
        hit = lambda run: (w[run] <= params.enter).any()          # noqa: E731
        mark = IN_LOW_PATCH
    fp = _footprints(nucs, params.nuc_min_bp) if params.snap else None
    out: List[Patch] = []
    j = 0
    while j < n:
        if not ext[j]:
            j += 1
            continue
        e = j
        while e + 1 < n and ext[e + 1]:
            e += 1
        run = slice(j, e + 1)
        if hit(run) and e - j + 1 >= params.min_tiles:
            state[run] = mark
            raw = np.asarray(track.score[run], dtype=float)
            sc = raw[np.isfinite(v[run])]
            pr = np.asarray(track.period[run])
            pr = pr[pr > 0]
            period = int(np.median(pr)) if len(pr) else -1
            gs, ge = int(track.tile_starts[j]), int(track.tile_ends[e])
            p = Patch(gs, ge, gs, ge, j, e, float(sc.mean()) if len(sc) else float('nan'),
                      period, open_start=j == 0, open_end=e == n - 1, kind=kind)
            # high patches snap only with a period (as before); low ones also without one under the 'open' rule
            if fp is not None and (period > 0 or (kind == LOW and params.snap_rule == 'open')):
                _snap(p, fp, params)
            out.append(p)
        j = e + 1
    return out


def _clamp_neighbours(patches: List[Patch]) -> None:
    """Snapped edges never cross a neighbouring patch (module docstring), in place. Tile runs are disjoint, so
    grid edges never overlap: an edge snapped into or past a neighbour goes back to its grid edge (not snapped)."""
    order = sorted(range(len(patches)), key=lambda i: (patches[i].grid_start, patches[i].grid_end))
    for a, b in zip(order, order[1:]):
        p, q = patches[a], patches[b]
        if q.start < p.grid_end:
            q.start, q.snapped_start = q.grid_start, False
        if p.end > q.start:
            p.end, p.snapped_end = p.grid_end, False
            if p.start >= p.end:
                p.start, p.snapped_start = p.grid_start, False
    patches.sort(key=lambda x: (x.start, x.end))


def _clamp_low(low: List[Patch], high: List[Patch], state: np.ndarray) -> List[Patch]:
    """Low patches clipped at the high patches' (snapped) edges, so the kinds never overlap while the high patches
    stay exactly what high-only calling gives. A low patch clipped to nothing is dropped (its tiles: OUTSIDE)."""
    out = []
    for q in low:
        for h in high:
            if h.grid_end <= q.grid_start and q.start < h.end:
                q.start, q.snapped_start = max(q.grid_start, h.end), False
            elif h.grid_start >= q.grid_end and q.end > h.start:
                q.end, q.snapped_end = min(q.grid_end, h.start), False
        if q.end > q.start:
            out.append(q)
        else:
            state[q.first_tile:q.last_tile + 1] = OUTSIDE
    return out


# ---------------------------------------------------------------------------- snapping
def _footprints(nucs: Optional[np.ndarray], min_bp: int) -> np.ndarray:
    if nucs is None:
        return np.zeros((0, 2), dtype=np.int64)
    a = np.asarray(nucs, dtype=np.int64).reshape(-1, 2)
    a = a[(a[:, 1] - a[:, 0]) >= min_bp]
    return a[np.argsort(a[:, 0], kind='stable')]


def break_linkers(fp: np.ndarray, period: float, params: PatchParams = DEFAULT_PATCHES) -> np.ndarray:
    """Boolean per linker (``len(fp) - 1``): True where the spacing breaks.

    ``fp`` are nucleosome-sized footprints sorted by start. Rule 'open': the
    edge-to-edge linker is at least ``open_bp``. Rule 'period': the dyad distance
    is off ``period`` by more than ``tol_frac`` x period (pairs with a merged
    footprint: the linker is longer than ``period - core_bp + tol``)."""
    if len(fp) < 2:
        return np.zeros(0, dtype=bool)
    if params.snap_rule == 'open':
        return (fp[1:, 0] - fp[:-1, 1]) >= params.open_bp
    tol = params.tol_frac * period
    dy = (fp[:, 0] + fp[:, 1]) / 2.0
    d = np.diff(dy)
    gap = fp[1:, 0] - fp[:-1, 1]
    ln = fp[:, 1] - fp[:, 0]
    merged = (ln[:-1] >= params.merged_bp) | (ln[1:] >= params.merged_bp)
    off = np.abs(d - period) > tol
    long_linker = gap > (period - params.core_bp + tol)
    return np.where(merged, long_linker, off)


def snap_edge(edge: int, side: str, fp: np.ndarray, period: float,
              params: PatchParams = DEFAULT_PATCHES) -> Optional[int]:
    """The snapped position of one grid edge, or None (no break within ``search_bp``).

    ``side`` is 'start' (patch to the right) or 'end' (patch to the left). Each
    linker is placed at its midpoint; beyond the outermost footprint counts as a
    break placed at that footprint's outer boundary. The break nearest ``edge``
    wins (ties: the one further out); the edge moves to the boundary of the
    footprint on the patch side of it."""
    if len(fp) == 0:
        return None
    brk = break_linkers(fp, period, params)
    # candidate (position, snapped edge) pairs
    if side == 'start':
        pos = np.r_[fp[0, 0], (fp[:-1, 1] + fp[1:, 0]) / 2.0]
        snap_to = fp[:, 0]
        ok = np.r_[True, brk]
    elif side == 'end':
        pos = np.r_[(fp[:-1, 1] + fp[1:, 0]) / 2.0, fp[-1, 1]]
        snap_to = fp[:, 1]
        ok = np.r_[brk, True]
    else:
        raise ValueError("side must be 'start' or 'end'")
    dist = np.abs(pos - edge)
    cand = np.flatnonzero(ok & (dist <= params.search_bp))
    if not len(cand):
        return None
    outward = -pos[cand] if side == 'start' else pos[cand]
    best = cand[np.lexsort((-outward, dist[cand]))[0]]
    return int(snap_to[best])


def _snap(p: Patch, fp: np.ndarray, params: PatchParams) -> None:
    s, e = p.grid_start, p.grid_end
    snapped_s = snapped_e = False
    if not p.open_start:
        x = snap_edge(p.grid_start, 'start', fp, p.period, params)
        if x is not None:
            s, snapped_s = x, True
    if not p.open_end:
        x = snap_edge(p.grid_end, 'end', fp, p.period, params)
        if x is not None:
            e, snapped_e = x, True
    if e <= s:
        return
    p.start, p.end, p.snapped_start, p.snapped_end = s, e, snapped_s, snapped_e


# ---------------------------------------------------------------------------- projection and tags
def patch_ref_interval(mol, patch: Patch, snapped: bool = True):
    """(ref_start, ref_end, open_left, open_right) in reference orientation, or None if unaligned."""
    a, b = patch.edges(snapped)
    env = mol.mol_interval_to_ref(a, b)
    if env is None or env[1] <= env[0]:
        return None
    if getattr(mol, 'is_reverse', False):
        return int(env[0]), int(env[1]), patch.open_end, patch.open_start
    return int(env[0]), int(env[1]), patch.open_start, patch.open_end


def patch_rows(mp: MoleculePatches) -> List[Tuple[int, int, List[int]]]:
    """``nucpatch`` rows: (start, length, [score_q, period_q, flags]) per patch, snapped edges."""
    rows = []
    for p in mp.patches:
        if p.end <= p.start:
            continue
        rows.append((int(p.start), int(p.end - p.start),
                     [encode_score(p.mean_score), encode_period(p.period), int(p.flags)]))
    return rows


def decode_patch_rows(intervals: Sequence[Tuple[int, int]], q: Sequence[Sequence[int]]) -> List[dict]:
    """Decode ``nucpatch`` rows: molecular ``(start, length)`` intervals (as ``read_nuc_groups`` returns
    them) and their Q bytes."""
    out = []
    for (s, n), b in zip(intervals, q):
        e = int(s) + int(n)
        fl = int(b[2]) if len(b) > 2 else 0
        out.append(dict(start=int(s), end=int(e), mean_score=decode_score(b[0]), period=decode_period(b[1]),
                        open_start=bool(fl & OPEN_START), open_end=bool(fl & OPEN_END),
                        snapped_start=bool(fl & SNAPPED_START), snapped_end=bool(fl & SNAPPED_END),
                        kind=LOW if fl & LOW_KIND else HIGH))
    return out
