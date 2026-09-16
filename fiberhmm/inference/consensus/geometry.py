# Numerical kernel promoted from the validated September 2026 consensus experiments.
from __future__ import annotations

import numpy as np


def representative_geometries(kernel):
    """A REAL integer geometry per projection, nearest the class center.

    Never round independent edge means into an invalid/chimeric interval. This
    tie choice stays inside an exactly identical cohort opportunity projection;
    it cannot change any observed modification likelihood or the fitted q mass.
    """
    coordinates = []
    x = kernel.ambiguity_bp
    for center, family in zip(kernel.centers, kernel.families):
        choices = {}
        for a in range(int(center[0])-x, int(center[0])+x+1):
            for b in range(int(center[1])-x, int(center[1])+x+1):
                if a >= b:
                    continue
                ia, ib = np.searchsorted(kernel.positions, [a, b])
                if ia == ib:
                    continue
                rank = ((a-center[0])**2+(b-center[1])**2,
                        abs((b-a)-(center[1]-center[0])), a, b)
                key = (int(ia), int(ib))
                if key not in choices or rank < choices[key][0]:
                    choices[key] = (rank, (a, b))
        coordinates.extend(choices[(int(a), int(b))][1]
                           for a, b in zip(family['starts'], family['ends']))
    coordinates = np.asarray(coordinates, dtype=np.int64)
    projected = np.searchsorted(kernel.positions, coordinates)
    np.testing.assert_array_equal(projected, np.c_[kernel.ga, kernel.gb])
    return coordinates


def merged(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def physically_allowed(unit, coordinates, opportunities, minimum_opportunities):
    """Caller-context eligibility, independent of this TF's membership.

    Alignment/opportunities are observation context. MSPs and nucleosomes are
    upstream inferred annotations, NOT outcome-free sequence covariates. Counts
    using this mask are therefore conditional on those frozen upstream calls.
    """
    a, b = coordinates.T
    aligned = np.zeros(len(a), bool)
    in_msp = np.zeros(len(a), bool)
    blocked = np.zeros(len(a), bool)
    for lo, hi in merged(unit['aligned_blocks']):
        aligned |= (a >= lo) & (b <= hi)
    for lo, hi in unit['msp_intervals']:
        in_msp |= (a >= lo) & (b <= hi)
    for lo, hi in unit['raw_nuc_intervals']:
        blocked |= (a < hi) & (b > lo)
    return aligned & in_msp & ~blocked & (opportunities >= minimum_opportunities)


def overlap_mask(calls, coordinates):
    a, b = coordinates.T
    out = np.zeros(len(a), bool)
    for lo, hi in calls:
        out |= (a < hi) & (b > lo)
    return out
