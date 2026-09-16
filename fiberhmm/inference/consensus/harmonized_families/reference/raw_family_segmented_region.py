# Extracted reference kernels; see SOURCE_MANIFEST.json.
from collections import Counter
import hashlib
import numpy as np

def merge_intervals(intervals):
    out = []
    for (a, b) in sorted(intervals):
        if type(a) is not int or type(b) is not int or a >= b:
            raise ValueError('Positive half-open integer intervals required')
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out

def intersect_intervals(left, right):
    a = merge_intervals(left)
    b = merge_intervals(right)
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo < hi:
            out.append([lo, hi])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out

def subtract_intervals(intervals, excluded):
    out = []
    excluded = merge_intervals(excluded)
    for (a, b) in merge_intervals(intervals):
        for (x, y) in excluded:
            if y <= a:
                continue
            if x >= b:
                break
            if a < x:
                out.append([a, x])
            a = max(a, y)
            if a >= b:
                break
        if a < b:
            out.append([a, b])
    return out

def eligible_intervals(unit, region):
    aligned = unit.get('aligned_blocks', [[unit['reference_start'], unit['reference_end']]])
    msp = intersect_intervals(unit['msp_intervals'], [list(region)])
    return subtract_intervals(intersect_intervals(msp, aligned), unit['raw_nuc_intervals'])

def prepare_stratum(source, region, seed=20260912):
    units = []
    cohorts = []
    discovery = []
    seen = set()
    stats = Counter()
    for original in source['units']:
        uid = source['dataset_id'] + '::' + original['unit_id']
        group = source['dataset_id'] + '::' + original.get('fold_group_id', original['unit_id'])
        if group in seen:
            raise ValueError('Repeated physical group; collapse upstream, not per MSP')
        seen.add(group)
        eligible = eligible_intervals(original, region)
        p = np.asarray(original['positions'])
        mask = np.zeros(len(p), bool)
        for (a, b) in eligible:
            mask |= (p >= a) & (p < b)
        keep = np.flatnonzero(mask).tolist()
        if any((len(original[k]) != len(p) for k in ['hits', 'p_accessible', 'p_protected'])):
            raise ValueError('Unaligned native arrays')
        u = dict(unit_id=uid, fold_group_id=group, strand=original['strand'], eligible_intervals=eligible, **{k: [original[k][i] for i in keep] for k in ['positions', 'hits', 'p_accessible', 'p_protected']})
        if int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:16], 16) % 2 == 0:
            discovery.append(len(units))
        units.append(u)
        cohorts.append(source['dataset_id'] + '__' + original['strand'])
        stats['source_reads'] += 1
        stats['reads_with_eligible_bases'] += bool(eligible)
        stats['reads_with_eligible_native_sites'] += bool(keep)
        stats['eligible_native_sites'] += len(keep)
        stats['source_native_sites_in_region'] += int(((p >= region[0]) & (p < region[1])).sum())
    return (units, cohorts, discovery, dict(stats))
