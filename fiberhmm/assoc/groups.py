"""Grouping variables for association contrasts (HUB 7.7).

- **Groups** are per-molecule labels: dataset, haplotype, cluster label, replicate.
- **Haplotypes** (owner decision D14): unphased molecules are excluded from haplotype contrasts and reported as
  their own group, descriptively.
- **Replicates**: when datasets are declared biological replicates, a contrast is estimated per replicate and the
  estimates are combined with a random-effects model. With one library per group a result describes "these
  libraries", not a population.
- **Cluster labels**: a contrast between clusters is circular when the cluster run used features in the tested
  window. :func:`post_selection_check` refuses it unless the window and its flanks lie outside every clustering
  region by at least the cluster window's flank.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from fiberhmm.assoc.resample import random_effects

UNPHASED_GROUP = 'unphased'
_UNPHASED = {'', '.', '0', 'none', 'None', 'nan', 'NaN', 'unphased', 'UNPHASED', 'NA'}


def is_unphased(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, (int, np.integer)) and int(value) == 0:
        return True
    return str(value).strip() in _UNPHASED


def group_labels(values, kind: str = 'label') -> Tuple[np.ndarray, dict]:
    """Labels for a contrast. Returns (labels, info): ``labels`` is an object array of group names, with ``None``
    for molecules that take no part in the contrast. For ``kind='haplotype'`` unphased molecules get ``None`` and
    are counted under ``info['unphased']``; ``info['groups']`` lists the contrasted groups in sorted order."""
    vals = list(values)
    out = np.empty(len(vals), dtype=object)
    unphased = 0
    for i, v in enumerate(vals):
        if kind == 'haplotype' and is_unphased(v):
            out[i] = None
            unphased += 1
        elif v is None or (isinstance(v, float) and math.isnan(v)):
            out[i] = None
        else:
            out[i] = str(v)
    groups = sorted({g for g in out if g is not None})
    info = {'kind': kind, 'groups': groups}
    if kind == 'haplotype':
        info['unphased'] = int(unphased)
        info['rule'] = 'unphased molecules are excluded from haplotype contrasts and reported as their own group (D14)'
    return out, info


def unphased_mask(values) -> np.ndarray:
    return np.array([is_unphased(v) for v in values], dtype=bool)


def combine_replicates(per_replicate: Mapping[str, Tuple[float, float]]) -> dict:
    """Random-effects combination of per-replicate contrasts {replicate: (estimate, se)}. With a single replicate
    the result is labelled conditional on the libraries."""
    names = sorted(per_replicate)
    est = [per_replicate[n][0] for n in names]
    se = [per_replicate[n][1] for n in names]
    out = random_effects(est, se)
    out['replicates'] = names
    out['scope'] = 'population (random effects over replicates)' if out['k'] >= 2 else 'these libraries'
    return out


def post_selection_check(tested: Sequence[Tuple[str, int, int]], cluster_regions: Iterable[Tuple[str, int, int]],
                         flank: int, accept_post_selection: bool = False) -> dict:
    """Is a cluster contrast over ``tested`` windows circular? ``cluster_regions`` are the regions the cluster run
    used (``ClusterTreeNode.peakRegions``); ``flank`` is the cluster window's flank. Returns {status: 'ok' |
    'refused' | 'post-selection', overlaps: [...]}. A refused contrast gets no p-value; with
    ``accept_post_selection`` it is reported descriptively, labelled 'post-selection'."""
    overlaps = []
    regions = list(cluster_regions)
    for chrom, start, end in tested:
        for c, s, e in regions:
            if c == chrom and s - flank < end + flank and start - flank < e + flank:
                overlaps.append({'tested': [chrom, int(start), int(end)], 'cluster_region': [c, int(s), int(e)]})
    if not overlaps:
        return {'status': 'ok', 'overlaps': []}
    return {'status': 'post-selection' if accept_post_selection else 'refused', 'overlaps': overlaps,
            'note': 'the cluster run used features within the tested window or its flanks: descriptive only, '
                    'no p-value'}


def group_contrast(estimates: Dict[str, float], boot, names: Optional[Dict[str, str]] = None,
                   reference: Optional[str] = None) -> dict:
    """Interaction contrast between groups: differences against ``reference`` (the first group) and a Wald
    chi-square with the bootstrap covariance. ``boot`` is a :class:`fiberhmm.assoc.resample.BootDraws` (``names``
    maps group -> statistic name; default the group name) or a dict {group: draws}."""
    from scipy.stats import chi2, norm
    from fiberhmm.assoc.resample import BootDraws, boot_p_two, percentile_ci
    groups = [g for g in estimates]
    if len(groups) < 2:
        return {'groups': groups, 'status': 'needs two groups'}
    if not isinstance(boot, BootDraws):
        boot = BootDraws('groups', draws={g: np.asarray(v, float) for g, v in boot.items()})
        names = None
    names = names or {g: g for g in groups}
    ref = reference if reference is not None else groups[0]
    others = [g for g in groups if g != ref]
    allg = [ref] + others
    cov = boot.cov([names[g] for g in allg])
    A = np.zeros((len(others), len(allg)))
    A[:, 0] = -1
    A[np.arange(len(others)), np.arange(1, len(allg))] = 1
    dobs = np.array([estimates[g] - estimates[ref] for g in others], float)
    cd = A @ cov @ A.T
    rows = []
    for k, g in enumerate(others):
        se = float(np.sqrt(cd[k, k])) if np.isfinite(cd[k, k]) and cd[k, k] > 0 else float('nan')
        row = {'group': g, 'reference': ref, 'difference': float(dobs[k]), 'se': se,
               'p_wald': float(2 * norm.sf(abs(dobs[k] / se))) if np.isfinite(se) and se > 0 else float('nan')}
        if boot.mode != 'twoway':
            d = boot.draws[names[g]] - boot.draws[names[ref]]
            fin = d[np.isfinite(d)]
            row['lo'], row['hi'] = percentile_ci(fin)
            row['p_boot'] = boot_p_two(fin)
        else:
            row['lo'], row['hi'] = dobs[k] - 1.959964 * se, dobs[k] + 1.959964 * se
        rows.append(row)
    out = {'groups': groups, 'reference': ref, 'differences': rows, 'variance': boot.mode}
    if np.all(np.isfinite(cd)) and np.all(np.isfinite(dobs)):
        try:
            stat = float(dobs @ np.linalg.solve(cd, dobs))
            out.update(chi2=stat, df=len(others), p=float(chi2.sf(stat, len(others))))
        except np.linalg.LinAlgError:
            out['p'] = float('nan')
    else:
        out['p'] = float('nan')
    return out
