"""NFR variants and element co-accessibility (EXPERIMENTAL preview).

This package is an experimental preview, not a released analysis. Its outputs, parameters and file
formats may change without notice (schema ``fiberhmm.accessibility.preview.v1``; ``analysis`` holds
the views of a finished run: read splits per element pair, size/edge/V-plot arrays, phasing, frozen-catalogue
prevalence).

What it does, per region:

* per-read nucleosome-free regions (NFRs): gaps between consecutive >= 90-bp nucleosome calls; factor-sized
  protections inside a gap do not split it (the Timer preprint's rule);
* NFR variant discovery with the lattice recaller's recipe (k-means on gap edges, prediction strength for k,
  held-out identity merges, held-out support), read-level EM prevalence over configurations, per-read membership;
* element co-accessibility between variants (and footprint classes from a lattice-recaller run): spanning reads
  only, Timer's "shared" exclusion, an exact test stratified by per-read openness x channel, Mantel-Haenszel
  odds ratio, Benjamini-Hochberg;
* combination patterns against independence and a curveball (fixed-margin) null.

Entry points: :func:`run_accessibility` (Python) and the ``fiberhmm-nfr`` CLI.
"""
from .workflow import SCHEMA, NFROptions, run_accessibility, load_recaller_classes

EXPERIMENTAL = True

__all__ = ['SCHEMA', 'NFROptions', 'run_accessibility', 'load_recaller_classes', 'EXPERIMENTAL']
