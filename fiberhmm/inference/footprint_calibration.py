"""ChIP-supervised width calibration for ``fiberhmm-footprint`` (``fiberhmm-footprint calibrate``).

The method follows the benchmark diagnosis (``ft_vs_fh_tf_diagnosis``: "ChIP-learned
width window" and "window2d"):

* Rule: the footprint rule of the normal command (MSP eligibility, containment by
  outer or conservative bounds, LLR floor, optional ``--require-unmodified``). An
  eligible molecule is positive for a window ``[wmin, wmax]`` when one qualifying
  call's conservative width lies in it (both bounds inclusive).
* Score per motif: footprinted / eligible molecules, pooled over a cell's BAMs.
  Motifs with no eligible molecule are left out.
* AUROC of that score against ChIP labels, per cell. The objective for a set of
  cells is the mean of the per-cell AUROCs; observations are never pooled across
  cells, because ChIP labels are cell specific.
* Grid: wmin 0-80 bp and wmax 10-150 bp in 5 bp steps by default (wmin < wmax).
* Held out: leave-one-cell-type-out when several cells are given; otherwise odd
  versus even chromosomes (chrX and every non-numbered contig with the even set),
  fitted in both directions. A window is always chosen on training motifs/cells
  only and then evaluated on the held-out ones.
* Robust range: the bounds of the windows within ``near`` (0.005) AUROC of the
  optimum.

A learned window is specific to the factor (and, to a lesser degree, the cell
type); the held-out AUROC is the honest estimate of its accuracy.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SETTINGS_SCHEMA = 'fiberhmm-footprint-settings/v1'


def parse_range(text: str) -> List[int]:
    """``START:STOP:STEP`` (inclusive) -> list of ints."""
    parts = [int(x) for x in str(text).split(':')]
    if len(parts) != 3 or parts[2] <= 0 or parts[1] < parts[0] or parts[0] < 0:
        raise ValueError(f'expected START:STOP:STEP with STOP >= START >= 0 and STEP > 0, got {text!r}')
    return list(range(parts[0], parts[1] + 1, parts[2]))


def auroc(score: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC with ties averaged; NaN without both classes."""
    from scipy.stats import rankdata
    y = np.asarray(labels, bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def odd_chromosome(chrom: str) -> bool:
    """Odd-numbered autosome (chr1, 3, ...); chrX and unnumbered contigs count as even."""
    m = re.fullmatch(r'(?:chr)?(\d+)', str(chrom))
    return bool(m) and int(m.group(1)) % 2 == 1


@dataclass
class CellData:
    """One cell type: per motif, the qualifying-width sets of its eligible molecules."""
    name: str
    motif_chrom: List[str]
    labels: np.ndarray                       # per motif, bool
    molecules: List[List[Tuple[int, ...]]]   # per motif, widths tuple per eligible molecule

    def counts(self, wmins: Sequence[int], wmaxs: Sequence[int]):
        """``(positives[motif, i, j], eligible[motif])`` for window (wmins[i], wmaxs[j])."""
        n = len(self.molecules)
        pos = np.zeros((n, len(wmins), len(wmaxs)), np.int64)
        elig = np.array([len(m) for m in self.molecules], np.int64)
        a = np.asarray(wmins)
        b = np.asarray(wmaxs)
        for k, mols in enumerate(self.molecules):
            if not mols:
                continue
            # smallest qualifying width >= wmin, per molecule; a window counts when it is <= wmax
            nge = np.full((len(mols), len(a)), np.inf)
            for r, ws in enumerate(mols):
                if ws:
                    w = np.asarray(ws)
                    idx = np.searchsorted(w, a, side='left')
                    ok = idx < len(w)
                    nge[r, ok] = w[idx[ok]]
            pos[k] = (nge[:, :, None] <= b[None, None, :]).sum(0)
        return pos, elig


@dataclass
class Surface:
    wmins: List[int]
    wmaxs: List[int]
    per_cell: Dict[str, np.ndarray]          # cell -> AUROC[i, j] (NaN where wmin >= wmax)
    positive_fraction: np.ndarray            # pooled positives / eligible per window
    motif_fraction: np.ndarray               # pooled fraction of motifs with >= 1 positive

    @property
    def mean(self) -> np.ndarray:
        with np.errstate(all='ignore'):
            return np.nanmean(np.stack(list(self.per_cell.values())), axis=0)


def _cell_aurocs(cell: CellData, pos, elig, keep: np.ndarray, wmins, wmaxs) -> np.ndarray:
    use = keep & (elig > 0)
    out = np.full((len(wmins), len(wmaxs)), np.nan)
    y = cell.labels[use]
    for i, a in enumerate(wmins):
        for j, b in enumerate(wmaxs):
            if a < b:
                out[i, j] = auroc(pos[use, i, j] / elig[use], y)
    return out


def surface(cells: Sequence[CellData], wmins, wmaxs, motif_mask: Optional[Dict[str, np.ndarray]] = None,
            cache=None) -> Surface:
    """AUROC surface per cell on the motifs ``motif_mask`` keeps (all by default)."""
    per_cell, tot_pos, tot_elig, any_pos, n_motifs = {}, 0, 0, 0, 0
    for cell in cells:
        pos, elig = cache[cell.name] if cache is not None and cell.name in cache else cell.counts(wmins, wmaxs)
        if cache is not None:
            cache[cell.name] = (pos, elig)
        keep = motif_mask[cell.name] if motif_mask is not None else np.ones(len(elig), bool)
        per_cell[cell.name] = _cell_aurocs(cell, pos, elig, keep, wmins, wmaxs)
        use = keep & (elig > 0)
        tot_pos = tot_pos + pos[use].sum(0)
        tot_elig += int(elig[use].sum())
        any_pos = any_pos + (pos[use] > 0).sum(0)
        n_motifs += int(use.sum())
    with np.errstate(all='ignore'):
        return Surface(list(wmins), list(wmaxs), per_cell, tot_pos / max(tot_elig, 1), any_pos / max(n_motifs, 1))


def optimum(values: np.ndarray, wmins, wmaxs) -> Tuple[int, int, float]:
    """Best window. Ties go to the first in grid order (smallest wmin, then smallest wmax), as
    ``np.argmax`` over the benchmark's window list does."""
    best = None
    for i, a in enumerate(wmins):
        for j, b in enumerate(wmaxs):
            v = values[i, j]
            if np.isnan(v):
                continue
            if best is None or v > best[2] + 1e-12:
                best = (a, b, float(v))
    if best is None:
        raise ValueError('no window could be scored (each cell needs ChIP-positive and -negative motifs with '
                         'eligible molecules)')
    return best


def robust_range(values: np.ndarray, wmins, wmaxs, near: float) -> dict:
    top = np.nanmax(values)
    sel = [(a, b) for i, a in enumerate(wmins) for j, b in enumerate(wmaxs)
           if not np.isnan(values[i, j]) and values[i, j] >= top - near]
    return dict(within=near, windows=len(sel), wmin=[min(a for a, _ in sel), max(a for a, _ in sel)],
                wmax=[min(b for _, b in sel), max(b for _, b in sel)])


def held_out(cells: Sequence[CellData], wmins, wmaxs, design: str, cache) -> List[dict]:
    """Learn on training data only, evaluate on held-out data. ``design``: 'cell' or 'chromosome'."""
    rows = []
    if design == 'cell':
        if len(cells) < 2:
            raise ValueError('leave-one-cell-type-out needs at least two cells')
        for test in cells:
            train = [c for c in cells if c.name != test.name]
            a, b, fit = optimum(surface(train, wmins, wmaxs, cache=cache).mean, wmins, wmaxs)
            ev = surface([test], wmins, wmaxs, cache=cache)
            i, j = wmins.index(a), wmaxs.index(b)
            ins = optimum(ev.mean, wmins, wmaxs)
            rows.append(dict(design='leave_one_cell_out', held_out=test.name, train=','.join(c.name for c in train),
                             wmin=a, wmax=b, train_auroc=fit, heldout_auroc=float(ev.mean[i, j]),
                             insample_window=f'{ins[0]}-{ins[1]}', insample_auroc=ins[2],
                             optimism=ins[2] - float(ev.mean[i, j])))
        return rows
    if design != 'chromosome':
        raise ValueError(f'unknown held-out design {design!r}')
    for cell in cells:
        odd = np.array([odd_chromosome(c) for c in cell.motif_chrom])
        for name, tr, te in (('fit_odd_eval_even', odd, ~odd), ('fit_even_eval_odd', ~odd, odd)):
            a, b, fit = optimum(surface([cell], wmins, wmaxs, {cell.name: tr}, cache).mean, wmins, wmaxs)
            ev = surface([cell], wmins, wmaxs, {cell.name: te}, cache)
            i, j = wmins.index(a), wmaxs.index(b)
            ins = optimum(ev.mean, wmins, wmaxs)
            rows.append(dict(design='chromosome_split', held_out=f'{cell.name}:{name.split("_eval_")[1]}',
                             train=f'{cell.name}:{name.split("_")[1]}', wmin=a, wmax=b, train_auroc=fit,
                             heldout_auroc=float(ev.mean[i, j]), insample_window=f'{ins[0]}-{ins[1]}',
                             insample_auroc=ins[2], optimism=ins[2] - float(ev.mean[i, j])))
    return rows


def label_motifs(motifs, peaks_path=None, label_column: Optional[int] = None, bed_path=None) -> np.ndarray:
    """ChIP labels per motif: overlap with a peak BED, or a 0/1 column of the motif BED (1-based)."""
    if peaks_path:
        import gzip
        from bisect import bisect_left
        peaks: Dict[str, List[Tuple[int, int]]] = {}
        opener = gzip.open if str(peaks_path).endswith('.gz') else open
        with opener(peaks_path, 'rt') as fh:
            for line in fh:
                if not line.strip() or line.startswith(('#', 'track', 'browser')):
                    continue
                f = line.split()
                peaks.setdefault(f[0], []).append((int(f[1]), int(f[2])))
        index = {c: (sorted(v), max(e - s for s, e in v)) for c, v in peaks.items()}
        out = []
        for m in motifs:
            iv, longest = index.get(m.chrom, ([], 0))
            k = bisect_left(iv, (m.start - longest, -1))
            out.append(any(s < m.end and e > m.start for s, e in iv[k:k + 10000] if s < m.end))
        return np.array(out, bool)
    if label_column is None:
        raise ValueError('give ChIP peaks (--chip) or a label column of the motif BED (--label-column)')
    import gzip
    opener = gzip.open if str(bed_path).endswith('.gz') else open
    values = []
    with opener(bed_path, 'rt') as fh:
        for line in fh:
            if not line.strip() or line.startswith(('#', 'track ', 'browser ')):
                continue
            f = line.rstrip('\n').split('\t') if '\t' in line else line.split()
            values.append(f[label_column - 1].strip() in ('1', 'true', 'True', 'pos', '+'))
    if len(values) != len(motifs):
        raise ValueError('--label-column: the motif BED rows and the labelled rows differ (filters cannot be used '
                         'with a label column)')
    return np.array(values, bool)


def settings_payload(wmin: int, wmax: int, options, calibration: dict) -> dict:
    """The JSON that ``fiberhmm-footprint --settings`` loads."""
    return dict(schema=SETTINGS_SCHEMA, min_width=wmin, max_width=wmax, min_llr=options.min_llr, edges=options.edges,
                outer_bound=options.outer_bound, require_unmodified=options.require_unmodified, ml=options.min_ml,
                calibration=calibration)


SETTINGS_KEYS = ('min_width', 'max_width', 'min_llr', 'edges', 'outer_bound', 'require_unmodified', 'ml')


def load_settings(path) -> dict:
    with open(path) as fh:
        data = json.load(fh)
    if data.get('schema') != SETTINGS_SCHEMA:
        raise ValueError(f'{path}: not a fiberhmm-footprint settings file (schema {data.get("schema")!r})')
    return {k: data[k] for k in SETTINGS_KEYS if k in data}


def write_heatmap(path, surf: Surface, best: Tuple[int, int], heldout_rows, near: float) -> bool:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    z = surf.mean
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(z, origin='lower', aspect='auto', cmap='viridis',
                   extent=(surf.wmaxs[0] - 2.5, surf.wmaxs[-1] + 2.5, surf.wmins[0] - 2.5, surf.wmins[-1] + 2.5))
    fig.colorbar(im, ax=ax, label='mean per-cell AUROC')
    top = np.nanmax(z)
    X, Y = np.meshgrid(surf.wmaxs, surf.wmins)
    if np.isfinite(top):
        ax.contour(X, Y, np.nan_to_num(z, nan=-1), levels=[top - near], colors='white', linewidths=1)
    ax.plot([best[1]], [best[0]], marker='*', ms=14, color='red', label=f'optimum [{best[0]}, {best[1]}]')
    for r in heldout_rows:
        ax.plot([r['wmax']], [r['wmin']], marker='o' if r['design'] == 'leave_one_cell_out' else 'x', ms=7,
                mfc='none', color='orange')
    ax.plot([], [], 'o', mfc='none', color='orange', label='held-out optima')
    ax.set_xlabel('max width (bp)')
    ax.set_ylabel('min width (bp)')
    ax.set_title(f'ChIP-calibrated width window (white: within {near} of max)')
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def write_surface_tsv(path, surf: Surface) -> None:
    names = list(surf.per_cell)
    with open(path, 'w') as fh:
        fh.write('\t'.join(['wmin', 'wmax', 'mean_auroc', *[f'auroc_{n}' for n in names], 'positive_fraction',
                            'motif_fraction', 'masked']) + '\n')
        z = surf.mean
        for i, a in enumerate(surf.wmins):
            for j, b in enumerate(surf.wmaxs):
                if a >= b:
                    continue
                masked = surf.positive_fraction[i, j] < 0.01 or surf.motif_fraction[i, j] < 0.20
                vals = [f'{surf.per_cell[n][i, j]:.6f}' for n in names]
                fh.write('\t'.join([str(a), str(b), f'{z[i, j]:.6f}', *vals, f'{surf.positive_fraction[i, j]:.6f}',
                                    f'{surf.motif_fraction[i, j]:.6f}', str(int(masked))]) + '\n')
