"""Rough pre-run peak-memory estimate for a lattice-recaller consensus run, and --max-memory-gb.

Calibrated on PacBio Hia5 promoter windows (600 bp, ~30 kb reads, 3.0.6 +
evidence pad and pooled caps), summed RSS of the process tree. It is a planning
figure (typically within ~30%), not a guarantee:

* main process: interpreter and libraries plus the loaded evidence, which after
  cropping scales with molecules x window width (several Python copies of each
  molecule's window lattice: payload, pooled view, analysis copy);
* each worker process: its own interpreter and libraries plus one task's
  matrices, which scale with the molecules a tile or scoring group uses
  (at most the discovery / quantification cap); measured peaks overlap in
  about two workers at a time.

Calibration (6 cores): 1,500 molecules estimated 2.6 GB, measured 2.5 GB;
30,910 molecules with 20,000-molecule caps estimated 16.9 GB, measured 16.3 GB.
"""
from __future__ import annotations

import math

MAIN_BASE_GB = 0.30           # interpreter, numpy/scipy/sklearn/numba, FiberHMM
WORKER_BASE_GB = 0.25         # each loky worker's own interpreter and libraries
EVIDENCE_BYTES_PER_BP = 600.  # per molecule and window bp: all copies and per-molecule results (pad <= 2 kb)
WHOLE_READ_BYTES_PER_KB = 13e3 # extra per molecule and kb of read when --evidence-pad full
TASK_BYTES_PER_MOLECULE = 1e5  # one discovery/quantification task, per molecule it uses
HEAVY_TASKS = 2               # worker tasks at their peak at the same time (measured)


def estimate(molecules, window_bp, cores, discovery_cap=0, quantification_cap=0, read_kb=None, pad=1000):
    """Estimated peak GB (dict with the parts) for ``molecules`` loaded molecules."""
    molecules = max(0, int(molecules)); cores = max(1, int(cores))
    per_molecule = EVIDENCE_BYTES_PER_BP * window_bp
    if pad is None and read_kb:
        per_molecule += WHOLE_READ_BYTES_PER_KB * read_kb
    evidence = molecules * per_molecule / 1e9
    caps = [c for c in (discovery_cap, quantification_cap) if c]
    used = min([molecules] + caps)
    task = used * TASK_BYTES_PER_MOLECULE / 1e9
    workers = 0 if cores <= 1 else cores
    main = MAIN_BASE_GB + evidence + (task if cores <= 1 else 0.)
    worker_total = workers * WORKER_BASE_GB + min(workers, HEAVY_TASKS) * task
    return dict(total_gb=round(main + worker_total, 2), main_gb=round(main, 2), workers=workers,
                per_worker_gb=round(WORKER_BASE_GB + task, 2) if workers else 0., evidence_gb=round(evidence, 2),
                molecules=molecules, task_molecules=used)


def fit_cores(limit_gb, molecules, window_bp, cores, **kw):
    """Largest core count <= cores whose estimate fits limit_gb, or 0 when even one core does not fit."""
    for c in range(max(1, int(cores)), 0, -1):
        if estimate(molecules, window_bp, c, **kw)['total_gb'] <= limit_gb:
            return c
    return 0


def describe(e):
    workers = f" + {e['workers']} workers x {e['per_worker_gb']:.2f} GB" if e['workers'] else ''
    return (f"estimated peak memory {e['total_gb']:.1f} GB (main {e['main_gb']:.2f} GB incl. evidence "
            f"{e['evidence_gb']:.2f} GB for {e['molecules']:,} molecules{workers}; tasks use up to "
            f"{e['task_molecules']:,} molecules)")
