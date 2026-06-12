"""Estimate the nucleosome repeat length (NRL) for the Pass-2 phase prior.

Runs HMM apply + Pass-1 nuc recall on a sample of reads, collects the
center-to-center spacing of adjacent confidently-separated nucleosomes, and
takes a robust central estimate of the primary repeat peak. The estimate is
CLAMPED to a sane band (anchored near 185 bp) so it can never run away to, e.g.,
50 or 300 bp on a noisy sample; it falls back to the anchor if the sample is
too small.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pysam

from fiberhmm.core.model_io import freeze_model_for_inference, load_model, load_model_with_metadata
from fiberhmm.inference.engine import CHIMERA_SKIP, _extract_fiber_read_from_pysam
from fiberhmm.inference.fused_stages import run_hmm_apply_stage
from fiberhmm.inference.nuc_recaller import recall_nucs_in_read
from fiberhmm.inference.tf_recaller import build_llr_tables

# Window (bp) for the primary nucleosome-repeat peak: excludes sub-nucleosomal
# gaps and 2x+ merge spacings so the estimate tracks the single-repeat mode.
_PEAK_LO, _PEAK_HI = 120, 260


def estimate_phase_nrl(
    input_bam: str,
    apply_model_path: str,
    recall_model_path: Optional[str] = None,
    *,
    mode: str = 'daf',
    context_size: int = 3,
    anchor: int = 185,
    clamp_lo: int = 150,
    clamp_hi: int = 215,
    sample_target: int = 3000,
    min_pairs: int = 300,
    split_min_llr: float = 4.0,
    split_min_opps: int = 3,
    nuc_min_size: int = 85,
    msp_min_size: int = 0,
    prob_threshold: int = 128,
    edge_trim: int = 10,
) -> dict:
    """Return ``{'nrl', 'ci', 'n_pairs', 'n_reads', 'source'}``.

    ``nrl`` is the clamped integer estimate (always within
    ``[clamp_lo, clamp_hi]``). ``source`` is ``'estimated'`` or ``'anchor'``
    (insufficient data).
    """
    model = freeze_model_for_inference(load_model(apply_model_path))
    r_model, _, _ = load_model_with_metadata(recall_model_path or apply_model_path)
    llr_hit, llr_miss = build_llr_tables(r_model)

    spacings = []
    n_reads = 0
    bam = pysam.AlignmentFile(input_bam, 'rb', check_sq=False)
    try:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            fr = _extract_fiber_read_from_pysam(read, mode, prob_threshold)
            if fr is None or fr is CHIMERA_SKIP:
                continue
            apply_result = run_hmm_apply_stage(
                fr, model, edge_trim, False, mode, context_size,
                msp_min_size, nuc_min_size, False)
            if apply_result is None or len(apply_result['ns']) == 0:
                continue
            nucs, _ = recall_nucs_in_read(
                apply_result['encoded'], apply_result['ns'], apply_result['nl'],
                len(apply_result['encoded']), llr_hit, llr_miss,
                split_min_llr=split_min_llr, split_min_opps=split_min_opps,
                nuc_min_size=nuc_min_size)
            nucs = sorted(nucs, key=lambda n: n.start)
            for i in range(len(nucs) - 1):
                c1 = nucs[i].start + nucs[i].length / 2.0
                c2 = nucs[i + 1].start + nucs[i + 1].length / 2.0
                spacings.append(c2 - c1)
            n_reads += 1
            if n_reads >= sample_target:
                break
    finally:
        bam.close()

    sp = np.asarray(spacings, dtype=np.float64)
    peak = sp[(sp >= _PEAK_LO) & (sp <= _PEAK_HI)]
    if peak.size < min_pairs:
        return {'nrl': int(anchor), 'ci': None, 'n_pairs': int(peak.size),
                'n_reads': n_reads, 'source': 'anchor'}

    # Robust central estimate: histogram mode (10bp bins) cross-checked against
    # the in-peak median, then clamped to the anchored band.
    bins = np.arange(_PEAK_LO, _PEAK_HI + 10, 10)
    hist, edges = np.histogram(peak, bins=bins)
    mode = float(edges[int(np.argmax(hist))] + 5)
    median = float(np.median(peak))
    est = 0.5 * (mode + median)
    nrl = int(round(np.clip(est, clamp_lo, clamp_hi)))
    ci = (float(np.percentile(peak, 25)), float(np.percentile(peak, 75)))
    return {'nrl': nrl, 'ci': ci, 'n_pairs': int(peak.size),
            'n_reads': n_reads, 'source': 'estimated'}
