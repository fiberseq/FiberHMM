# Numerical kernel promoted from the validated September 2026 consensus experiments.
from __future__ import annotations

import numpy as np


def bounded_projection(positions, center, ambiguity_bp=10):
    positions = np.asarray(positions, dtype=np.int64)
    center = np.asarray(center, dtype=float)
    if positions.ndim != 1 or not len(positions) or np.any(np.diff(positions) <= 0):
        raise ValueError("Strictly ordered nonempty opportunity union required")
    if center.shape != (2,) or not np.all(np.isfinite(center)) or center[1] <= center[0]:
        raise ValueError("Valid finite start/end center required")
    if not isinstance(ambiguity_bp, (int, np.integer)) or ambiguity_bp < 0:
        raise ValueError("ambiguity_bp must be a nonnegative integer")
    center = np.rint(center).astype(np.int64)
    if center[1] <= center[0]:
        raise ValueError("Rounded center must have positive width")
    shift = np.arange(-ambiguity_bp, ambiguity_bp + 1)
    a, b = np.meshgrid(center[0] + shift, center[1] + shift, indexing="ij")
    physical = a < b
    raw = np.c_[a[physical], b[physical]]
    projected = np.searchsorted(positions, raw)
    visible = projected[:, 0] < projected[:, 1]
    if not visible.any():
        raise ValueError("Family has no visible nonempty opportunity projection")
    unique, inverse, counts = np.unique(projected[visible], axis=0, return_inverse=True, return_counts=True)
    mean_edges = np.column_stack([np.bincount(inverse, weights=raw[visible, j], minlength=len(unique)) / counts
                                  for j in range(2)])
    return {"center": center.tolist(), "ambiguity_bp": int(ambiguity_bp),
            "starts": unique[:, 0], "ends": unique[:, 1],
            "q": counts.astype(float) / counts.sum(),
            "mean_integer_edges_given_projection": mean_edges,
            "valid_integer_geometries": len(raw),
            "invalid_width_fraction": float(1 - physical.mean()),
            "invisible_projection_fraction": float(1 - visible.mean()),
            "conditional_on_visible_projection": True,
            "semantics": "Uniform independent integer edge shifts, conditioned on positive width and visible nonempty projection"}
