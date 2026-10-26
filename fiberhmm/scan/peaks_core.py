"""Accessibility peak scoring and tiering, ported from FiberBrowser (``services/peak_scanning.py``).

Per base: ``numerator`` and ``total`` read counts from the reads' compact
tilings, the signal ``numerator / total`` (float32), smoothed by a boxcar of
``smooth_width`` (``np.convolve(..., 'same')``, float32), then progressively
masked threshold tiers: Strong, Medium, Weak and Rare. Regions wider than
``max_width`` are the view's background, not peaks ("broad").

Two rare-tier rules:
- ``window`` (FiberBrowser's): the rare threshold is
  ``max(threshold_rare, min_reads / n_reads)`` with ``n_reads`` the reads in the
  window, so it changes with the view;
- ``local``: the rare threshold is ``threshold_rare``, and a base takes part in the
  rare tier only where at least ``min_reads`` molecules are measured there
  (``total >= min_reads``). It does not depend on the window, so grid scans use it.

Both keep FiberBrowser's per-peak check that a rare peak's mean read count is
at least ``min_reads``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_MAX_PEAK_WIDTH = 2000
DEFAULTS = dict(min_width=80, min_reads=5, smooth_width=100, threshold_strong=0.50, threshold_medium=0.30,
                threshold_weak=0.15, threshold_rare=0.08, max_peak_width=DEFAULT_MAX_PEAK_WIDTH)
SUBNUCLEOSOMAL_BP = 90
# Signal sources ported with per-position identity. FiberBrowser's FIRE/MSP and 'layer:<name>' sources also
# depend on its legacy layer inference (layers built from ns/nl/as/al when MA lacks them); they stay in the
# browser until that is ported.
TILING_SOURCES = ('accessibility', 'tf_density')


def range_counts(starts: Sequence[int], ends: Sequence[int], width: int) -> np.ndarray:
    width = max(0, int(width))
    if width <= 0:
        return np.zeros(0, dtype=np.int32)
    if not len(starts):
        return np.zeros(width, dtype=np.int32)
    s = np.clip(np.asarray(starts, dtype=np.int64), 0, width)
    e = np.clip(np.asarray(ends, dtype=np.int64), 0, width)
    valid = e > s
    if not np.any(valid):
        return np.zeros(width, dtype=np.int32)
    delta = np.bincount(s[valid], minlength=width + 1) - np.bincount(e[valid], minlength=width + 1)
    return np.cumsum(delta[:-1], dtype=np.int64).astype(np.int32, copy=False)


def compute_scan_counts(rows: Sequence[dict], signal_source: str, region_width: int,
                        subnucleosomal_threshold: int = SUBNUCLEOSOMAL_BP,
                        coordinate_offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """(numerator, total) per base over ``region_width``; rows' intervals are region-relative."""
    if signal_source not in TILING_SOURCES:
        raise ValueError(f'signal source {signal_source!r} is not ported (use {", ".join(TILING_SOURCES)})')
    ts, te, ns, ne = [], [], [], []
    for row in rows:
        intervals = row.get('intervals') or []
        if signal_source == 'tf_density':
            for a, b, size in intervals:
                ts.append(int(a) - coordinate_offset); te.append(int(b) - coordinate_offset)
                if 20 <= size < 65:
                    ns.append(int(a) - coordinate_offset); ne.append(int(b) - coordinate_offset)
            continue
        for a, b, size in intervals:
            ts.append(int(a) - coordinate_offset); te.append(int(b) - coordinate_offset)
            if size < subnucleosomal_threshold:
                ns.append(int(a) - coordinate_offset); ne.append(int(b) - coordinate_offset)
    return range_counts(ns, ne, region_width), range_counts(ts, te, region_width)


def scan_signal_from_counts(numerator: np.ndarray, total: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(total > 0, numerator / total, 0).astype(np.float32)


def smooth_scan_signal(signal: np.ndarray, smooth_width: int) -> np.ndarray:
    if len(signal) == 0:
        return signal.astype(np.float32, copy=True)
    smooth_width = min(max(1, int(smooth_width)), len(signal))
    kernel = np.ones(smooth_width, dtype=np.float32) / smooth_width
    return np.convolve(signal, kernel, mode='same')


def build_peak_thresholds(threshold_strong, threshold_medium, threshold_weak, threshold_rare, min_reads, n_reads,
                          rule: str = 'window'):
    if rule == 'window':
        rare = threshold_rare if n_reads <= 0 else max(threshold_rare, min_reads / n_reads)
    else:
        rare = threshold_rare
    return [
        (0, threshold_strong, f'Strong (>{int(threshold_strong * 100)}%)'),
        (1, threshold_medium, f'Medium (>{int(threshold_medium * 100)}%)'),
        (2, threshold_weak, f'Weak (>{int(threshold_weak * 100)}%)'),
        (3, rare, f'Rare (>{int(threshold_rare * 100)}%)'),
    ]


def _find_threshold_regions(signal, threshold, masked, eligible=None):
    working = signal.copy()
    working[masked] = 0
    above = working >= threshold
    if eligible is not None:
        # the local rule: a base without min_reads molecules, or already masked by a higher tier, never qualifies
        above &= eligible & ~masked
    padded = np.concatenate([[False], above, [False]])
    transitions = np.diff(padded.astype(np.int8))
    return np.where(transitions == 1)[0], np.where(transitions == -1)[0]


def detect_threshold_layers(combined_signal, total_counts, thresholds, min_width, min_reads, smooth_width,
                            max_width=None, broad_out=None, rule: str = 'window'):
    """FiberBrowser's progressively masked tiers; ``rule='local'`` restricts the rare tier to bases with
    ``total_counts >= min_reads``."""
    region_width = len(total_counts)
    all_peaks: List[Dict[str, Any]] = []
    masked = np.zeros(region_width, dtype=bool)
    layer_info = []
    max_width = int(max_width or 0)
    local_ok = (np.asarray(total_counts) >= min_reads) if rule == 'local' else None
    for layer, threshold, label in thresholds:
        layer_peaks = []
        n_broad = 0
        starts, ends = _find_threshold_regions(combined_signal, threshold, masked,
                                               local_ok if layer == 3 else None)
        for reg_start, reg_end in zip(starts, ends):
            width = reg_end - reg_start
            if width < min_width:
                continue
            if max_width > 0 and width > max_width:
                n_broad += 1
                if broad_out is not None:
                    broad_out.append(dict(start=int(reg_start), end=int(reg_end), layer=layer,
                                          score=float(combined_signal[reg_start:reg_end].max()), threshold=threshold))
                masked[max(0, reg_start - smooth_width // 4):min(region_width, reg_end + smooth_width // 4)] = True
                continue
            peak_score = float(combined_signal[reg_start:reg_end].max())
            if layer == 3:
                peak_counts = total_counts[reg_start:reg_end]
                reads_at_peak = int(peak_counts.mean()) if peak_counts.size > 0 else 0
                if reads_at_peak < min_reads:
                    continue
            layer_peaks.append(dict(start=int(reg_start), end=int(reg_end), layer=layer, score=peak_score,
                                    threshold=threshold))
            masked[max(0, reg_start - smooth_width // 4):min(region_width, reg_end + smooth_width // 4)] = True
        all_peaks.extend(layer_peaks)
        layer_info.append(dict(layer=layer, threshold=threshold, label=label, n_peaks=len(layer_peaks),
                               n_broad=n_broad))
    return all_peaks, layer_info


def scan_peaks(numerator: np.ndarray, total: np.ndarray, *, n_reads: int, rule: str = 'window',
               params: Optional[dict] = None) -> dict:
    """The browser's whole pipeline over one window: signal, smoothing, thresholds, tiers."""
    p = dict(DEFAULTS, **(params or {}))
    signal = scan_signal_from_counts(numerator, total)
    smoothed = smooth_scan_signal(signal, p['smooth_width'])
    thresholds = build_peak_thresholds(p['threshold_strong'], p['threshold_medium'], p['threshold_weak'],
                                       p['threshold_rare'], p['min_reads'], n_reads, rule)
    broad: List[dict] = []
    peaks, info = detect_threshold_layers(smoothed, total, thresholds, p['min_width'], p['min_reads'],
                                          p['smooth_width'], max_width=p['max_peak_width'], broad_out=broad, rule=rule)
    return dict(signal=signal, smoothed=smoothed, thresholds=thresholds, peaks=peaks, layer_info=info, broad=broad)
