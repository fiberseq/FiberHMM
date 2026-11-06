"""Single-fibre nucleosome regularity along each molecule (MA group ``nucreg``).

An FFT periodicity measure on what FiberHMM already calls and on the raw
marks; no new model. Per molecule, windows of ``window_bp`` every ``step_bp``
(molecular coordinates, anchored at molecular position 0). The signal per bp
is the m6A calls (``signal='m6a'``, default) or footprint occupancy
(``'occupancy'``: 1 inside a nucleosome call). Each window is mean-removed,
Hann tapered and transformed with an ``nfft``-point rfft. The score is
log2(max power at periods in ``band_bp`` / median power at periods in
``ref_bp`` outside the band); the local repeat length is the period of that
maximum. An m6A window needs ``min_calls`` calls.

Options (both off by default for the layer):

- ``mask_open_bp``: open stretches (a gap of at least this between
  consecutive nucleosome calls, and the read ends beyond the outer calls) are
  set to the window mean before the FFT; a window more than half masked has no
  score.
- ``thin_per_kb``: each call is kept with probability target / density
  (calls per kb outside masked stretches) before scoring; a molecule below the
  target is left unthinned (``thinned`` False). The generator is seeded from a
  SHA-256 of the read name, so every caller thins a molecule the same way.

Read ends (``ends``). With ``blank`` (default) only full windows are scored,
so the first and last ~``window_bp/2 - step_bp/2`` bp of a read (plus up to
``step_bp - 1`` bp more at the far end) have no tile and reads shorter than
``window_bp`` have none. ``extend`` stretches the first and last tile to the
read ends, and leading / trailing unscored tiles take the score of the
outermost scored tile. ``truncated`` continues the window grid to the read
ends with windows cut at the read end, scored while at least ``end_min_bp``
wide (``short_min_bp`` on reads shorter than ``window_bp``, whose windows are
all cut); each cut window is tapered and transformed at its own width, and the
outermost tile is anchored at the read end when it falls within one step of
it. ``min_calls`` stays an absolute count for cut windows.

Every window centre owns one ``step_bp`` tile (``centre - step/2`` to
``centre + step/2``), so tiles abut and never overlap. The ``nucreg`` group
stores one mark per tile with a score: Q bytes ``[score_q, period_q]``
(:func:`encode_score`, :func:`encode_period`).

The random-packing reference (:func:`reference_scores`) places each sampled
window's own footprints at random non-overlapping positions and, for m6A,
simulates calls on that layout at the window's own inside/outside marking
rates. It is a dataset-level comparison, never stored in the tag.

A cut window (ends ``truncated``) is a shorter FFT with fewer band bins, so
its random-packing null sits below the full window's. With ``ends``
'truncated' the reference also samples the cut windows the molecules have
(from their own window grid), simulates each at its own width and keeps a
median per width class (``step_bp`` wide); a tile is then read against the
reference of its own window width (:func:`tile_reference`, interpolated
between classes). Full-width windows are sampled exactly as before, so the
full-width median does not depend on the read-end mode.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

from fiberhmm.nuc.records import NucMolecule

SIGNALS = ('m6a', 'occupancy')
ENDS = ('blank', 'extend', 'truncated')
#: score byte: 1 + round(clip((score - SCORE_LO) / SCORE_SPAN, 0, 1) * 254); 0 is reserved
SCORE_LO, SCORE_SPAN = -2.0, 10.0
#: period byte: period - PERIOD_OFFSET (0 = none)
PERIOD_OFFSET = 100
REFERENCE_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
_CHUNK = 256          # windows per FFT batch (bounds memory on long molecules)


@dataclass(frozen=True)
class RegularityParams:
    window_bp: int = 2000
    step_bp: int = 250
    #: 'm6a' (raw calls) or 'occupancy' (1 inside a nucleosome call)
    signal: str = 'm6a'
    band_bp: Tuple[int, int] = (140, 260)
    ref_bp: Tuple[int, int] = (50, 1000)
    #: m6A windows need at least this many calls (after thinning)
    min_calls: int = 15
    #: open stretches of at least this many bp are masked; None = off
    mask_open_bp: Optional[int] = None
    #: thinning target (calls per kb outside masked stretches); None = off
    thin_per_kb: Optional[float] = None
    nfft: int = 8192
    #: read ends: 'blank' (full windows only), 'extend' (paint the outer tiles to the ends) or 'truncated'
    ends: str = 'blank'
    #: ends 'truncated': the narrowest cut window scored at a read end
    end_min_bp: int = 1000
    #: ends 'truncated': the narrowest window scored on a read shorter than window_bp
    short_min_bp: int = 1000

    def __post_init__(self):
        if self.ends not in ENDS:
            raise ValueError(f'nucreg: ends must be one of {ENDS}, not {self.ends!r}')
        # the cut-window widths matter (and are stored) only for ends 'truncated'
        if self.ends == 'truncated' and (not (64 <= self.end_min_bp <= self.window_bp)
                                         or not (64 <= self.short_min_bp <= self.window_bp)):
            raise ValueError('nucreg: end_min_bp and short_min_bp must be between 64 and window_bp')
        if self.signal not in SIGNALS:
            raise ValueError(f'nucreg: signal must be one of {SIGNALS}, not {self.signal!r}')
        if self.window_bp < 64 or not 1 <= self.step_bp <= self.window_bp or self.nfft < self.window_bp:
            raise ValueError('nucreg: need window_bp >= 64, 1 <= step_bp <= window_bp and nfft >= window_bp')
        if not (0 < self.band_bp[0] < self.band_bp[1]) or not (0 < self.ref_bp[0] < self.ref_bp[1]):
            raise ValueError('nucreg: band_bp and ref_bp must be increasing positive (lo, hi)')
        if self.band_bp[1] > PERIOD_OFFSET + 255 or self.band_bp[0] <= PERIOD_OFFSET:
            raise ValueError(f'nucreg: band_bp must lie within {PERIOD_OFFSET + 1}..{PERIOD_OFFSET + 255} bp '
                             '(the period byte stores period - 100)')
        if self.mask_open_bp is not None and self.mask_open_bp <= 0:
            raise ValueError('nucreg: mask_open_bp must be positive (None = off)')
        if self.thin_per_kb is not None and self.thin_per_kb <= 0:
            raise ValueError('nucreg: thin_per_kb must be positive (None = off)')

    def to_dict(self) -> dict:
        """The parameters; the read-end fields only when they matter (``ends`` 'blank' leaves all three out,
        'extend' the two widths), so tags, contracts and digests made before they existed stay identical."""
        out = asdict(self)
        for key, value in list(out.items()):
            if isinstance(value, tuple):
                out[key] = list(value)
        if self.ends == 'blank':
            for key in ('ends', 'end_min_bp', 'short_min_bp'):
                out.pop(key)
        elif self.ends == 'extend':
            out.pop('end_min_bp')
            out.pop('short_min_bp')
        return out

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "RegularityParams":
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        kwargs = {}
        for key, value in data.items():
            if key not in known:
                raise ValueError(f'unknown nucreg parameter: {key!r}')
            if isinstance(value, list):
                value = tuple(value)
            kwargs[key] = value
        return cls(**kwargs)

    def digest(self) -> str:
        """Full SHA-256 of the parameters; AN names carry the first 8 hex."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(blob.encode()).hexdigest()


DEFAULT_REGULARITY = RegularityParams()


@dataclass
class RegularityTrack:
    """Sliding-window regularity of one molecule (molecular coordinates).

    Per window: ``centres``, its tile ``[tile_starts, tile_ends)``, ``score``
    (log2; NaN = no score), ``period`` (bp; -1 = none) and ``n_calls`` (calls
    in the window after thinning)."""
    length: int
    centres: np.ndarray
    tile_starts: np.ndarray
    tile_ends: np.ndarray
    score: np.ndarray
    period: np.ndarray
    n_calls: np.ndarray
    thinned: bool = False
    #: calls per kb outside masked stretches, before thinning
    density_per_kb: float = float('nan')
    params_digest: str = ''
    #: per tile, the width of its window (bp); None = every window is ``window_bp`` wide (ends blank / extend)
    window_widths: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.centres)

    @property
    def scored(self) -> np.ndarray:
        return np.isfinite(self.score)


# ------------------------------------------------------------------ spectrum
@lru_cache(maxsize=512)      # ends 'truncated' cuts windows to many widths
def _bands(window_bp: int, nfft: int, band: Tuple[int, int], ref: Tuple[int, int]):
    freq = np.fft.rfftfreq(nfft)                       # cycles per bp
    with np.errstate(divide='ignore'):
        period = np.where(freq > 0, 1.0 / np.where(freq > 0, freq, 1.0), np.inf)
    in_band = (period >= band[0]) & (period <= band[1])
    in_ref = (period >= ref[0]) & (period <= ref[1]) & ~in_band
    band_periods = np.round(period[in_band]).astype(np.int64)
    return np.hanning(window_bp), in_band, in_ref, band_periods


def spectral_scores(X: np.ndarray, params: RegularityParams = DEFAULT_REGULARITY,
                    keep: Optional[np.ndarray] = None, *, workers: int = 1,
                    cut: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """(log2 score, period) per row of ``X`` (windows x ``window_bp``).

    ``keep`` (same shape, bool) marks the unmasked positions: masked ones are
    set to the row's unmasked mean, and rows more than half masked get NaN.
    ``workers``: threads for the FFT over rows (``scipy.fft``; the result is
    identical for any value). ``cut``: rows may be narrower than ``window_bp``
    (cut windows at read ends, tapered at their own width)."""
    X = np.asarray(X, dtype=float)
    n, W = X.shape
    if W != params.window_bp and not (W <= params.window_bp and cut):
        raise ValueError('spectral_scores: rows must be window_bp long')
    taper, in_band, in_ref, band_periods = _bands(W, params.nfft, tuple(params.band_bp), tuple(params.ref_bp))
    score = np.full(n, np.nan)
    period = np.full(n, -1, dtype=np.int64)
    if n == 0 or not in_band.any() or not in_ref.any():
        return score, period
    for a in range(0, n, _CHUNK):
        x = X[a:a + _CHUNK]
        ok = np.ones(len(x), dtype=bool)
        if keep is not None:
            k = np.asarray(keep[a:a + _CHUNK], dtype=bool)
            nk = k.sum(axis=1)
            ok = nk * 2 >= W
            mu = (x * k).sum(axis=1) / np.maximum(nk, 1)
            x = np.where(k, x, mu[:, None])
        x = (x - x.mean(axis=1, keepdims=True)) * taper
        P = np.abs(rfft(x, params.nfft, axis=1, workers=workers)) ** 2
        ref = np.median(P[:, in_ref], axis=1)
        bandP = P[:, in_band]
        j = np.argmax(bandP, axis=1)
        peak = bandP[np.arange(len(P)), j]
        good = ok & (ref > 0) & (peak > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            s = np.log2(peak / np.where(ref > 0, ref, 1.0))
        score[a:a + len(x)] = np.where(good, s, np.nan)
        period[a:a + len(x)] = np.where(good, band_periods[j], -1)
    return score, period


# ------------------------------------------------------------------ molecule signals
def open_mask(mol: NucMolecule, mask_open_bp: Optional[int]) -> Optional[np.ndarray]:
    """Bool keep-mask over the molecule (False in open stretches), or None when
    masking is off. Open: gaps of at least ``mask_open_bp`` between consecutive
    nucleosome calls, and the read ends beyond the outer calls."""
    if mask_open_bp is None:
        return None
    L = mol.length
    keep = np.zeros(L, dtype=bool)
    nucs = mol.nucs
    if not len(nucs):
        return keep
    starts = np.clip(nucs[:, 0], 0, L)
    ends = np.clip(np.maximum.accumulate(nucs[:, 1]), 0, L)
    keep[starts[0]:ends[-1]] = True
    gap = starts[1:] - ends[:-1]
    for i in np.flatnonzero(gap >= mask_open_bp):
        keep[ends[i]:starts[i + 1]] = False
    return keep


def occupancy(mol: NucMolecule) -> np.ndarray:
    """1.0 inside a nucleosome call, else 0.0."""
    L = mol.length
    d = np.zeros(L + 1, dtype=np.int64)
    if len(mol.nucs):
        np.add.at(d, np.clip(mol.nucs[:, 0], 0, L), 1)
        np.add.at(d, np.clip(mol.nucs[:, 1], 0, L), -1)
    return (np.cumsum(d[:-1]) > 0).astype(float)


def name_seed(name: str) -> int:
    """Stable 64-bit seed from a read name (SHA-256; never Python's hash())."""
    return int.from_bytes(hashlib.sha256(str(name).encode()).digest()[:8], 'little')


def thinned_hits(mol: NucMolecule, thin_per_kb: Optional[float],
                 keep: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool, float]:
    """(calls, thinned, density per kb). Density counts calls outside masked
    stretches (``keep``); below the target the calls are returned unchanged."""
    hits = mol.hits
    if keep is None:
        n_bp, n_hit = mol.length, int(hits.sum())
    else:
        n_bp, n_hit = int(keep.sum()), int((hits & keep).sum())
    density = 1000.0 * n_hit / n_bp if n_bp else float('nan')
    if thin_per_kb is None or not np.isfinite(density) or density <= thin_per_kb:
        return hits, False, density
    rng = np.random.default_rng(name_seed(mol.name))
    u = rng.random(mol.length)
    return hits & (u < thin_per_kb / density), True, density


def _window_starts(length: int, params: RegularityParams) -> np.ndarray:
    if length < params.window_bp:
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, length - params.window_bp + 1, params.step_bp, dtype=np.int64)


def _windows(v: np.ndarray, starts: np.ndarray, W: int) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view
    if not len(starts):
        return np.zeros((0, W), dtype=v.dtype)
    step = int(starts[1] - starts[0]) if len(starts) > 1 else 1
    return sliding_window_view(v, W)[starts[0]::step][:len(starts)]


def tiles(centres: np.ndarray, step_bp: int) -> Tuple[np.ndarray, np.ndarray]:
    """Step-wide tiles centred on each window centre (abutting, non-overlapping)."""
    a = np.asarray(centres, dtype=np.int64) - step_bp // 2
    return a, a + step_bp


def window_grid(length: int, params: RegularityParams = DEFAULT_REGULARITY):
    """(centres, tile_starts, tile_ends, window_starts, window_ends) of a molecule of ``length`` under
    ``params.ends`` (module docstring). For 'blank' and 'extend' every window is ``window_bp`` wide."""
    W, step, L = params.window_bp, params.step_bp, int(length)
    if params.ends != 'truncated':
        starts = _window_starts(L, params)
        centres = starts + W // 2
        ts, te = tiles(centres, step)
        if params.ends == 'extend' and len(ts):
            ts = ts.copy(); te = te.copy()
            ts[0], te[-1] = 0, L
        return centres, ts, te, starts, starts + W
    minw = params.short_min_bp if L < W else params.end_min_bp
    c = W // 2
    while c - step - step // 2 + step > 0:          # the grid continued down to the first tile touching 0
        c -= step
    cs = []
    while c - step // 2 < L:
        a, b = max(0, c - W // 2), min(L, c - W // 2 + W)
        full = a == c - W // 2 and b == c - W // 2 + W
        if full or b - a >= minw:
            cs.append(c)
        c += step
    centres = np.asarray(cs, dtype=np.int64)
    if not len(centres):
        z = np.zeros(0, dtype=np.int64)
        return z, z, z, z, z
    ts = np.clip(centres - step // 2, 0, L)
    te = np.clip(centres - step // 2 + step, 0, L)
    if ts[0] < step:
        ts[0] = 0
    if L - te[-1] < step:
        te[-1] = L
    ws = np.maximum(0, centres - W // 2)
    we = np.minimum(L, centres - W // 2 + W)
    return centres, ts, te, ws, we


def window_reach(length: int, params: RegularityParams = DEFAULT_REGULARITY) -> Optional[Tuple[int, int]]:
    """Molecular (start, end) covered by the window grid (beyond it: out of window reach), or None."""
    _c, ts, te, _ws, _we = window_grid(length, params)
    return (int(ts[0]), int(te[-1])) if len(ts) else None


def _paint_ends(score: np.ndarray, period: np.ndarray) -> None:
    """ends 'extend': unscored tiles before the first / after the last scored tile take its score and period."""
    ok = np.flatnonzero(np.isfinite(score))
    if not len(ok):
        return
    i0, i1 = ok[0], ok[-1]
    score[:i0], period[:i0] = score[i0], period[i0]
    score[i1 + 1:], period[i1 + 1:] = score[i1], period[i1]


def sliding_regularity(mol: NucMolecule, params: RegularityParams = DEFAULT_REGULARITY, *,
                       workers: int = 1) -> RegularityTrack:
    """Regularity track of one molecule (see the module docstring); ``workers``
    threads for the FFT (identical result)."""
    if params.ends == 'truncated':
        return _sliding_truncated(mol, params, workers)
    W = params.window_bp
    starts = _window_starts(mol.length, params)
    centres = starts + W // 2
    ts, te = tiles(centres, params.step_bp)
    keep = open_mask(mol, params.mask_open_bp)
    thin = params.thin_per_kb if params.signal == 'm6a' else None
    hits, thinned, density = thinned_hits(mol, thin, keep)
    # min_calls counts the calls the spectrum sees: masked stretches' calls are replaced by the window mean
    cum = np.concatenate([[0], np.cumsum(hits if keep is None else hits & keep, dtype=np.int64)])
    n_calls = cum[starts + W] - cum[starts] if len(starts) else np.zeros(0, dtype=np.int64)
    if params.signal == 'm6a':
        sig = hits.astype(float)
    else:
        sig = occupancy(mol)
    X = _windows(sig, starts, W)
    K = _windows(keep, starts, W) if keep is not None else None
    score, period = spectral_scores(X, params, K, workers=workers)
    if params.signal == 'm6a':
        low = n_calls < params.min_calls
        score[low] = np.nan
        period[low] = -1
    if params.ends == 'extend' and len(ts):
        ts, te = ts.copy(), te.copy()
        ts[0], te[-1] = 0, mol.length
        _paint_ends(score, period)
    return RegularityTrack(mol.length, centres, ts, te, score, period, n_calls.astype(np.int64),
                           thinned, density, params.digest()[:8])


def _sliding_truncated(mol: NucMolecule, params: RegularityParams, workers: int) -> RegularityTrack:
    """sliding_regularity for ends 'truncated': full windows as for 'blank', plus cut windows at the ends."""
    W = params.window_bp
    centres, ts, te, ws, we = window_grid(mol.length, params)
    keep = open_mask(mol, params.mask_open_bp)
    thin = params.thin_per_kb if params.signal == 'm6a' else None
    hits, thinned, density = thinned_hits(mol, thin, keep)
    cum = np.concatenate([[0], np.cumsum(hits if keep is None else hits & keep, dtype=np.int64)])
    n_calls = cum[we] - cum[ws] if len(ws) else np.zeros(0, dtype=np.int64)
    sig = hits.astype(float) if params.signal == 'm6a' else occupancy(mol)
    score = np.full(len(centres), np.nan)
    period = np.full(len(centres), -1, dtype=np.int64)
    width = we - ws
    for w in np.unique(width):
        idx = np.flatnonzero(width == w)
        if w == W:
            st = ws[idx]
            if len(st) > 1 and np.all(np.diff(st) == params.step_bp):
                X = _windows(sig, st, W)
                K = _windows(keep, st, W) if keep is not None else None
            else:
                X = np.stack([sig[a:a + W] for a in st])
                K = np.stack([keep[a:a + W] for a in st]) if keep is not None else None
            sc, pr = spectral_scores(X, params, K, workers=workers)
        else:
            X = np.stack([sig[a:a + w] for a in ws[idx]])
            K = np.stack([keep[a:a + w] for a in ws[idx]]) if keep is not None else None
            sc, pr = spectral_scores(X, params, K, workers=workers, cut=True)
        score[idx], period[idx] = sc, pr
    if params.signal == 'm6a':
        low = n_calls < params.min_calls
        score[low] = np.nan
        period[low] = -1
    return RegularityTrack(mol.length, centres, ts, te, score, period, n_calls.astype(np.int64),
                           thinned, density, params.digest()[:8], window_widths=width.astype(np.int64))


def track_summary(track: RegularityTrack, ref_median: Optional[float] = None) -> Dict[str, Optional[float]]:
    """Per-molecule summaries from the tiles: median and max score, the
    fraction of scored tiles above ``ref_median`` and the median repeat length."""
    s = np.asarray(track.score, dtype=float)
    ok = np.isfinite(s)
    out: Dict[str, Optional[float]] = {'n_tiles': int(ok.sum()), 'median': None, 'max': None,
                                       'frac_above_ref': None, 'repeat_length': None}
    if not ok.any():
        return out
    out['median'] = float(np.median(s[ok]))
    out['max'] = float(s[ok].max())
    out['repeat_length'] = float(np.median(np.asarray(track.period)[ok]))
    if ref_median is not None and np.isfinite(ref_median):
        out['frac_above_ref'] = float(np.mean(s[ok] > ref_median))
    return out


# ------------------------------------------------------------------ reference (random packing)
#: ends 'truncated': cut windows simulated per width class, and the fewest finite scores a class needs to count
#: (classes with fewer take the reference interpolated from their neighbours)
CUT_WINDOWS_PER_CLASS, MIN_CUT_SCORES = 500, 100
#: the fewest finite reference scores a usable reference needs (``ReferenceScores.usable``)
MIN_REFERENCE_SCORES = 100


@dataclass
class ReferenceScores:
    scores: np.ndarray
    median: float
    quantiles: Dict[float, float] = field(default_factory=dict)
    seed: int = 0
    params_digest: str = ''
    #: ends 'truncated': width class (lower edge, bp) -> median of the class's reference scores minus ``median``
    #: (classes with at least ``MIN_CUT_SCORES`` scores); empty for full windows only
    width_deltas: Dict[int, float] = field(default_factory=dict)
    #: width class -> its number of finite reference scores (every class sampled, kept or not)
    width_n: Dict[int, int] = field(default_factory=dict)
    #: the width-class step (bp); 0 = no width classes
    width_step: int = 0
    #: the full window width (bp) the deltas are relative to
    window_bp: int = 0

    @property
    def n(self) -> int:
        return int(len(self.scores))

    def usable(self, min_scores: int = MIN_REFERENCE_SCORES) -> bool:
        """At least ``min_scores`` finite full-width scores (or, without full windows, width-class scores) and a
        finite median."""
        n = self.n if self.n else sum(self.width_n.get(w, 0) for w in self.width_deltas)
        return bool(n >= min_scores and np.isfinite(self.median))

    def to_dict(self) -> dict:
        """JSON-safe: non-finite numbers (no reference) are None."""
        fin = lambda v: float(v) if v is not None and np.isfinite(v) else None      # noqa: E731
        out = {'n': self.n, 'median': fin(self.median), 'seed': self.seed, 'params_digest': self.params_digest,
               'quantiles': {str(q): fin(v) for q, v in self.quantiles.items()}}
        if self.width_n:
            # only with ends 'truncated', so records of full-window references stay identical
            out['width_deltas'] = {str(w): round(float(v), 6) for w, v in sorted(self.width_deltas.items())}
            out['width_n'] = {str(w): int(v) for w, v in sorted(self.width_n.items())}
            out['width_step'] = int(self.width_step)
            out['window_bp'] = int(self.window_bp)
        return out


def reference_widths(ref: Optional[dict]) -> Optional[dict]:
    """The per-width part of a reference record (``ReferenceScores.to_dict``) as ``call_patches`` takes it:
    ``{'deltas': {width class: delta}, 'step': step, 'window_bp': W}``, or None (full windows only, or an older
    record)."""
    if not ref or not ref.get('width_deltas'):
        return None
    return {'deltas': {int(w): float(v) for w, v in ref['width_deltas'].items()},
            'step': int(ref.get('width_step') or 0), 'window_bp': int(ref.get('window_bp') or 0)}


def tile_reference(widths: Optional[np.ndarray], ref_median: float, ref_widths: Optional[dict]) -> object:
    """The reference median each tile is read against: ``ref_median`` for full-width windows, plus the delta of
    the tile's window width (interpolated between width-class centres; beyond the outer classes, the nearest one).
    Returns the scalar ``ref_median`` when there are no per-tile widths or no width deltas."""
    if widths is None or not ref_widths or not ref_widths.get('deltas'):
        return float(ref_median)
    W, step = int(ref_widths['window_bp']), int(ref_widths['step'])
    cls = sorted((int(w), float(d)) for w, d in ref_widths['deltas'].items() if np.isfinite(d))
    if not cls or W <= 0 or step <= 0:
        return float(ref_median)
    xs = [min(W, w + step / 2.0) for w, _d in cls] + [float(W)]
    ys = [d for _w, d in cls] + [0.0]
    order = np.argsort(xs, kind='stable')
    xs, ys = np.asarray(xs)[order], np.asarray(ys)[order]
    w = np.asarray(widths, dtype=float)
    delta = np.where(w >= W, 0.0, np.interp(w, xs, ys))
    return float(ref_median) + delta


def _random_layout(sizes: np.ndarray, W: int, rng) -> np.ndarray:
    """Occupancy of ``sizes`` placed at random non-overlapping positions in
    ``W`` bp: random cut points for the free length, sizes in permuted order."""
    y = np.zeros(W)
    free = W - int(sizes.sum())
    if not len(sizes):
        return y
    if free <= 0:
        return np.ones(W)               # footprints fill the window: fully occupied, whatever the order
    cuts = np.sort(rng.integers(0, free + 1, len(sizes)))
    order = rng.permutation(len(sizes))
    sz = sizes[order]
    begin = cuts + np.concatenate([[0], np.cumsum(sz)[:-1]])
    d = np.zeros(W + 1, dtype=np.int64)
    np.add.at(d, begin, 1)
    np.add.at(d, begin + sz, -1)
    return (np.cumsum(d[:-1]) > 0).astype(float)


def _simulate(mol: NucMolecule, hits: np.ndarray, s: int, W: int, params: RegularityParams, rng):
    """One random-packing window ``[s, s + W)`` of ``mol`` (its own footprints placed at random, m6A simulated at
    its own inside / outside rates), or None (an m6A window below ``min_calls``)."""
    e = s + W
    nucs = mol.nucs
    sel = (nucs[:, 1] > s) & (nucs[:, 0] < e) if len(nucs) else np.zeros(0, dtype=bool)
    iv = np.stack([np.maximum(nucs[sel, 0], s) - s, np.minimum(nucs[sel, 1], e) - s], axis=1) \
        if sel.any() else np.zeros((0, 2), dtype=np.int64)
    y = np.zeros(W)
    for a, b in iv:
        y[a:b] = 1.0
    yn = _random_layout((iv[:, 1] - iv[:, 0]).astype(np.int64), W, rng)
    if params.signal != 'm6a':
        return yn
    x = hits[s:e].astype(float)
    fin, fout = y.sum(), W - y.sum()
    p_in = x[y == 1].sum() / fin if fin else 0.0
    p_out = x[y == 0].sum() / fout if fout else 0.0
    xn = (rng.random(W) < np.where(yn == 1, p_in, p_out)).astype(float)
    return None if xn.sum() < params.min_calls else xn


def _scored(mol: NucMolecule, hits: np.ndarray, ws: np.ndarray, we: np.ndarray, params: RegularityParams):
    """Which windows ``[ws, we)`` would be scored on the observed molecule (m6A: ``min_calls`` calls; occupancy:
    any footprint)."""
    if params.signal == 'm6a':
        cum = np.concatenate([[0], np.cumsum(hits, dtype=np.int64)])
        return (cum[we] - cum[ws]) >= params.min_calls
    occ = np.concatenate([[0], np.cumsum(occupancy(mol), dtype=np.int64)])
    return (occ[we] - occ[ws]) > 0


def reference_scores(mols: Sequence[NucMolecule], params: RegularityParams = DEFAULT_REGULARITY,
                     n_windows: int = 2000, seed: int = 0, *, cut_windows: int = CUT_WINDOWS_PER_CLASS,
                     min_cut_scores: int = MIN_CUT_SCORES) -> ReferenceScores:
    """Scores of the random-packing reference on windows sampled from ``mols``.

    Each sampled window keeps its own footprints (clipped to the window) and,
    for m6A, its own marking rates inside and outside them; the footprints are
    placed at random. Windows are drawn uniformly (without replacement) from
    those that would be scored on the observed molecule (m6A: ``min_calls``
    calls; occupancy: any footprint); simulated m6A windows below
    ``min_calls`` are dropped. Masking does not apply (the layout is random);
    thinning does. Deterministic for a fixed ``seed`` and molecule order.

    With ``params.ends`` 'truncated' the cut windows of the molecules' own window grids are also sampled, per width
    class of ``step_bp`` (up to ``cut_windows`` per class, from a separate random stream so the full-width sample
    is unchanged), each simulated and scored at its own width; ``width_deltas`` keeps each class's median minus the
    full-width median for classes with at least ``min_cut_scores`` finite scores. Without any full-width window
    (every molecule shorter than ``window_bp``) the widest kept class is the anchor: ``median`` is its median and
    the deltas are relative to it."""
    W = params.window_bp
    cand = []                               # (molecule index, window start)
    prepared = []
    thin = params.thin_per_kb if params.signal == 'm6a' else None
    for i, mol in enumerate(mols):
        starts = _window_starts(mol.length, params)
        hits, _t, _d = thinned_hits(mol, thin, open_mask(mol, params.mask_open_bp))
        prepared.append(hits)
        if not len(starts):
            continue
        ok = _scored(mol, hits, starts, starts + W, params)
        cand.extend((i, int(s)) for s in starts[ok])
    rng = np.random.default_rng(seed)
    nan_q = {q: float('nan') for q in REFERENCE_QUANTILES}
    score = np.zeros(0)
    if cand:
        pick = rng.choice(len(cand), size=min(n_windows, len(cand)), replace=False)
        pick.sort()
        rows = []
        for k in pick:
            i, s = cand[k]
            x = _simulate(mols[i], prepared[i], s, W, params, rng)
            if x is not None:
                rows.append(x)
        if rows:
            score, _p = spectral_scores(np.asarray(rows), params)
            score = score[np.isfinite(score)]
    median = float(np.median(score)) if len(score) else float('nan')
    qs = {q: float(np.quantile(score, q)) for q in REFERENCE_QUANTILES} if len(score) else dict(nan_q)
    deltas: Dict[int, float] = {}
    width_n: Dict[int, int] = {}
    step = int(params.step_bp)
    if params.ends == 'truncated':
        medians = {}
        by_class: Dict[int, list] = {}            # class -> [(molecule index, start, width)]
        for i, mol in enumerate(mols):
            _c, _ts, _te, ws, we = window_grid(mol.length, params)
            width = we - ws
            cut = width < W
            if not cut.any():
                continue
            ok = cut & _scored(mol, prepared[i], ws, we, params)
            for a, w in zip(ws[ok].tolist(), width[ok].tolist()):
                by_class.setdefault(step * (int(w) // step), []).append((i, int(a), int(w)))
        crng = np.random.default_rng([int(seed), 1])
        for c in sorted(by_class):
            items = by_class[c]
            pick = np.sort(crng.choice(len(items), size=min(int(cut_windows), len(items)), replace=False))
            got = []
            for k in pick:
                i, a, w = items[k]
                x = _simulate(mols[i], prepared[i], a, w, params, crng)
                if x is None:
                    continue
                sc, _p = spectral_scores(x[None, :], params, cut=True)
                if np.isfinite(sc[0]):
                    got.append(float(sc[0]))
            width_n[c] = len(got)
            if len(got) >= int(min_cut_scores):
                medians[c] = float(np.median(got))
        if medians and not np.isfinite(median):
            median = medians[max(medians)]          # no full window anywhere: the widest kept class anchors
        if np.isfinite(median):
            deltas = {c: m - median for c, m in medians.items()}
    return ReferenceScores(score, median, qs, seed, params.digest()[:8], deltas, width_n,
                           step if width_n else 0, W if width_n else 0)


# ------------------------------------------------------------------ masked mark ACF
def masked_mark_acf(mol: NucMolecule, lags: Sequence[int], mask_open_bp: Optional[int] = 300,
                    thin_per_kb: Optional[float] = 25.0) -> Optional[np.ndarray]:
    """Normalised autocorrelation of the molecule's calls at ``lags``, over
    position pairs that are both unmasked (open stretches masked as in
    :func:`open_mask`; None = no mask), after thinning as the track thins.
    None when nothing is left to correlate."""
    lags = np.asarray(lags, dtype=np.int64)
    if not len(lags) or lags.min() < 0:
        raise ValueError('masked_mark_acf: lags must be non-negative')
    keep = open_mask(mol, mask_open_bp)
    w = np.ones(mol.length) if keep is None else keep.astype(float)
    if w.sum() == 0:
        return None
    hits, _t, _d = thinned_hits(mol, thin_per_kb, keep)
    x = hits.astype(float)
    max_lag = int(lags.max())
    m = next_fast_len(mol.length + max_lag + 1, real=True)
    mu = (x * w).sum() / w.sum()
    y = (x - mu) * w
    fy, fw = rfft(y, m), rfft(w, m)
    num = irfft(fy * np.conj(fy), m)[:max_lag + 1]
    den = irfft(fw * np.conj(fw), m)[:max_lag + 1]
    c = num / np.maximum(den, 1)
    if not c[0] > 0:
        return None
    return (c / c[0])[lags]


# ------------------------------------------------------------------ nucreg bytes
def encode_score(score: float) -> int:
    """Score byte (1..255; 0 is reserved and means no score)."""
    if score is None or not np.isfinite(score):
        return 0
    f = min(1.0, max(0.0, (float(score) - SCORE_LO) / SCORE_SPAN))
    return 1 + int(round(f * 254))


def decode_score(q) -> float:
    q = int(q)
    return float('nan') if q <= 0 else SCORE_LO + SCORE_SPAN * (q - 1) / 254.0


def encode_period(period: int) -> int:
    """Period byte: period - 100, clipped to 1..255; 0 = none."""
    if period is None or int(period) <= 0:
        return 0
    return int(min(255, max(1, int(period) - PERIOD_OFFSET)))


def decode_period(q) -> int:
    q = int(q)
    return -1 if q <= 0 else q + PERIOD_OFFSET


def track_rows(track: RegularityTrack) -> List[Tuple[int, int, List[int]]]:
    """``[(tile_start, tile_length, [score_q, period_q]), ...]`` for every tile with a score."""
    out = []
    for i in np.flatnonzero(np.isfinite(track.score)):
        a = int(track.tile_starts[i])
        out.append((a, int(track.tile_ends[i]) - a,
                    [encode_score(float(track.score[i])), encode_period(int(track.period[i]))]))
    return out


def track_from_rows(length: int, intervals: Sequence[Tuple[int, int]], q: Sequence[Sequence[int]],
                    params: "RegularityParams" = None, density_per_kb: float = float('nan')) -> RegularityTrack:
    """A molecule's full regularity track rebuilt from its ``nucreg`` rows (scored tiles only) and its length:
    the window grid of ``params``, the rows' byte-resolution scores and periods, NaN / -1 for tiles without a row.
    ``n_calls`` is unknown (zeros). Patches called on it equal patches called on the original track except for
    scores within half a score byte (about 0.02 log2) of a threshold."""
    params = params or DEFAULT_REGULARITY
    centres, ts, te, ws, we = window_grid(int(length), params)
    score = np.full(len(centres), np.nan)
    period = np.full(len(centres), -1, dtype=np.int64)
    if len(ts) and len(intervals):
        d = decode_rows(intervals, q)
        idx = np.searchsorted(ts, d['tile_starts'])
        ok = (idx < len(ts)) & (ts[np.minimum(idx, len(ts) - 1)] == d['tile_starts'])
        score[idx[ok]] = d['score'][ok]
        period[idx[ok]] = d['period'][ok]
    return RegularityTrack(int(length), centres, ts, te, score, period, np.zeros(len(centres), dtype=np.int64),
                           False, float(density_per_kb), params.digest()[:8],
                           window_widths=(we - ws).astype(np.int64) if params.ends == 'truncated' else None)


def decode_rows(intervals: Sequence[Tuple[int, int]], q: Sequence[Sequence[int]]) -> Dict[str, np.ndarray]:
    """Decoded ``nucreg`` marks: ``tile_starts``, ``tile_ends``, ``centres``
    (tile midpoints), ``score`` (log2; byte resolution 10/254) and ``period``."""
    iv = np.asarray(list(intervals), dtype=np.int64).reshape(-1, 2)
    qb = np.asarray([list(x)[:2] for x in q], dtype=np.int64).reshape(-1, 2)
    starts = iv[:, 0]
    ends = iv[:, 0] + iv[:, 1]
    score = np.where(qb[:, 0] > 0, SCORE_LO + SCORE_SPAN * (qb[:, 0] - 1) / 254.0, np.nan)
    period = np.where(qb[:, 1] > 0, qb[:, 1] + PERIOD_OFFSET, -1)
    return {'tile_starts': starts, 'tile_ends': ends, 'centres': (starts + ends) // 2,
            'score': score, 'period': period.astype(np.int64)}
