"""Lattice consensus recaller (engine ``lattice_recaller``).

Class geometries come from k-means on confident native calls; every molecule's raw lattice is then scored against
those geometries (whatever the caller put there) and EM gives each class's prevalence per channel. No Monte Carlo.

Stages
  1. Discovery (discovery.py): k-means on lattice-censored (left, right) edges of LLR >= call_min_llr calls in tiles;
     k is the largest value whose split-half prediction strength reaches ``recaller.stringency``. Candidates are merged
     while an exact-likelihood held-out identity test cannot tell their geometries apart (< identity_nats).
  2. Geometry (discovery.py): shared class core; pooled edge boxes at ``edge_quantiles``; optional outward
     per-chemistry jitter; classes whose boxes leave no protected core (< minimum_core_bp) are dropped.
  3. Membership and prevalence (model.py): local lattice model per overlap group and channel (protected class
     interval, accessible linker beyond each edge, unknown state further out; configurations weighted by the edge
     positions they cover), EM over classes + broader protection + other shape + accessible, optional learned
     internal spots (held-out accepted), nested held-out support gain, expected-evidence resolution and a three-way
     per-molecule label.
"""
from .workflow import run_lattice_recaller

__all__ = ['run_lattice_recaller']
