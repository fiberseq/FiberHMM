"""Footprint re-calling on both-strand consensus reads.

A consensus read from :mod:`fiberhmm.crossstrand.consensus` mixes both strands'
deaminations (C->T as Y, G->A as R) and carries a ``deam+``/``deam-`` regime in
its MA tag. The single-strand ``--mode daf`` encoder can't use it (it picks one
dominant strand and the chimera filter would reject the mixed read). This module
builds a **both-strand** HMM observation instead:

  * a ``+`` (C-target) encoding, valid only inside the ``deam+`` interval;
  * a ``-`` (G-target) encoding, valid only inside the ``deam-`` interval;

merged position-wise. Because reference C's and G's are disjoint, the two
encodings never both claim a position: in the both-strand core *both* C's and
G's are informative (double density); in a single-strand flank only that
strand's base is, and the other strand's bases are correctly left non-target
(absence of data, not evidence of protection).

The merged observation runs through the standard HMM (``predict_footprints_and_msps``),
so all downstream footprint logic is unchanged.
"""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

import numpy as np

from fiberhmm.core.bam_reader import (
    ContextEncoder, _encode_daf_observations, _mod_positions_mask,
    _sequence_base_int_array,
)
from fiberhmm.io.ma_tags import parse_ma_tag

_BASE_C = 1   # _BASE_TO_INT: A=0, C=1, T=2, G=3
_BASE_G = 3


def decode_ry_consensus(seq: str) -> Tuple[str, Set[int], Set[int]]:
    """Split an R/Y consensus sequence into (conv_seq, ct_mods, ga_mods).

    Y (C->T) -> 'T' and recorded in ct_mods; R (G->A) -> 'A' and recorded in
    ga_mods; other bases unchanged. Matches the T/A representation the DAF
    encoder expects (it reconstructs T->C / A->G internally for context).
    """
    arr = np.frombuffer(seq.upper().encode('ascii'), dtype=np.uint8)
    ct = set(np.where(arr == ord('Y'))[0].tolist())
    ga = set(np.where(arr == ord('R'))[0].tolist())
    conv = seq.upper().replace('Y', 'T').replace('R', 'A')
    return conv, ct, ga


def deam_regime_masks(read) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Boolean (plus_mask, minus_mask) over query positions from the MA ``deam``
    track (deam+ = CT coverage, deam- = GA coverage), or None if absent."""
    try:
        raw = parse_ma_tag(read.get_tag('MA'))['raw_types']
    except (KeyError, ValueError):
        return None
    n = read.query_length
    plus = np.zeros(n, dtype=bool)
    minus = np.zeros(n, dtype=bool)
    found = False
    for name, strand, _qspec, intervals in raw:
        if name != 'deam':
            continue
        found = True
        target = plus if strand == '+' else minus if strand == '-' else None
        if target is None:
            continue
        for s, length in intervals:              # 0-based start, length
            lo = max(0, int(s)); hi = min(n, int(s) + int(length))
            if hi > lo:
                target[lo:hi] = True
    return (plus, minus) if found else None


def encode_daf_both_strand(conv_seq: str, ct_mods: Set[int], ga_mods: Set[int],
                           plus_mask: np.ndarray, minus_mask: np.ndarray,
                           edge_trim: int, context_size: int) -> np.ndarray:
    """Merged both-strand DAF observation array (length == len(conv_seq))."""
    n_codes = ContextEncoder.get_n_codes(context_size)
    non_target_code = n_codes
    unmethylated_offset = n_codes + 1
    fill = non_target_code + unmethylated_offset      # non-target sentinel (2n+1)
    L = len(conv_seq)

    plus = _encode_daf_observations(conv_seq, ct_mods, edge_trim, '+',
                                    context_size, non_target_code, unmethylated_offset)
    minus = _encode_daf_observations(conv_seq, ga_mods, edge_trim, '-',
                                     context_size, non_target_code, unmethylated_offset)
    seq_int = _sequence_base_int_array(conv_seq, uppercase=True)
    ct_mask = _mod_positions_mask(ct_mods, L)
    ga_mask = _mod_positions_mask(ga_mods, L)
    # A position contributes on the + strand iff it is a reference C (canonical
    # C, or a deaminated C now stored as T) inside the deam+ region; symmetric
    # for the - strand at reference G's inside deam-.
    plus_target = plus_mask & ((seq_int == _BASE_C) | ct_mask)
    minus_target = minus_mask & ((seq_int == _BASE_G) | ga_mask)

    out = np.full(L, fill, dtype=np.int32)
    out[minus_target] = minus[minus_target]
    out[plus_target] = plus[plus_target]
    return out


def recall_consensus_read(read, model, context_size: int, edge_trim: int = 10,
                          msp_min_size: int = 147, nuc_min_size: int = 85,
                          with_scores: bool = True):
    """Run the HMM footprint layer on one both-strand consensus read.

    Returns the ``predict_footprints_and_msps`` result dict, or None if the read
    has no ``deam`` regime (not a consensus read) or no sequence.
    """
    from fiberhmm.inference.engine import predict_footprints_and_msps
    seq = read.query_sequence
    if seq is None:
        return None
    masks = deam_regime_masks(read)
    if masks is None:
        return None
    plus_mask, minus_mask = masks
    conv, ct_mods, ga_mods = decode_ry_consensus(seq)
    encoded = encode_daf_both_strand(conv, ct_mods, ga_mods, plus_mask,
                                     minus_mask, edge_trim, context_size)
    if len(encoded) == 0:
        return None
    return predict_footprints_and_msps(
        model, encoded, msp_min_size=msp_min_size, with_scores=with_scores,
        nuc_min_size=nuc_min_size)


class RecallContext:
    """Loaded models/tables for both-strand recall (build once, reuse per read)."""

    def __init__(self, enzyme: str = 'ddda'):
        from fiberhmm.core.model_io import load_model_with_metadata
        from fiberhmm.inference.nuc_recaller import load_nuc_profile
        from fiberhmm.inference.tf_recaller import build_llr_tables
        from fiberhmm.models import _bundled_model_path, get_model_path
        self.apply_model, self.k, _ = load_model_with_metadata(get_model_path(enzyme, tool='apply'))
        recall_model, _, _ = load_model_with_metadata(get_model_path(enzyme, tool='recall'))
        self.llr_hit, self.llr_miss = build_llr_tables(recall_model)
        self.nuc_profile = (load_nuc_profile(_bundled_model_path('ddda_nuc_profile.json'))
                            if enzyme == 'ddda' else None)


def recall_consensus_full(seg, ctx: RecallContext, *, edge_trim: int = 10,
                          min_llr: float = 5.0, min_opps: int = 3,
                          unify_threshold: int = 90, split_min_llr: float = 4.0,
                          split_min_opps: int = 3, nuc_min_size: int = 85,
                          msp_min_size: int = 0, phase_nrl: int = 196) -> bool:
    """Full both-strand recall (HMM + nucleosome recaller + TF recaller) on one
    consensus read, writing MA/AQ + legacy tags in place. Returns False if the
    read is not a both-strand consensus.

    Reuses the canonical fused recall (``build_fused_recall_result``) on the
    both-strand observation, so the LLR accumulates over both strands' informative
    positions (C via the C-table, G reverse-complemented into it), with
    single-strand-flank bases of the absent strand left non-target.
    """
    from fiberhmm.inference.engine import predict_footprints_and_msps
    from fiberhmm.inference.fused_stages import build_fused_recall_result
    from fiberhmm.inference.tf_recaller import write_ma_tags

    seq = seg.query_sequence
    masks = deam_regime_masks(seg)
    if seq is None or masks is None:
        return False
    plus_mask, minus_mask = masks
    conv, ct_mods, ga_mods = decode_ry_consensus(seq)
    obs = encode_daf_both_strand(conv, ct_mods, ga_mods, plus_mask, minus_mask,
                                 edge_trim, ctx.k)
    if len(obs) == 0:
        return False
    # preserve the strand-regime (deam+/deam-) to re-append after write_ma_tags
    deam_parts = [c for c in seg.get_tag('MA').split(';')[1:] if c.startswith('deam')]

    fp = predict_footprints_and_msps(ctx.apply_model, obs, msp_min_size=msp_min_size,
                                     with_scores=False, nuc_min_size=nuc_min_size)
    apply_result = {'ns': fp['footprint_starts'], 'nl': fp['footprint_sizes'],
                    'as': fp['msp_starts'], 'al': fp['msp_sizes'], 'encoded': obs}
    res = build_fused_recall_result(
        {'query_sequence': conv}, apply_result, ctx.llr_hit, ctx.llr_miss,
        min_llr, min_opps, unify_threshold, with_scores=True, recall_nucs=True,
        split_min_llr=split_min_llr, split_min_opps=split_min_opps,
        nuc_min_size=nuc_min_size, msp_min_size=msp_min_size, phase_nrl=phase_nrl,
        nuc_profile=ctx.nuc_profile)

    kept_nucs = list(zip([int(x) for x in res['ns']], [int(x) for x in res['nl']]))
    msps = list(zip([int(x) for x in res['as']], [int(x) for x in res['al']]))
    write_ma_tags(seg, seg.query_length, res['tf_calls'], kept_nucs, msps,
                  nq_for_kept_nucs=res.get('nq_for_kept_nucs'),
                  nuc_el_for_kept=res.get('nuc_el_for_kept'),
                  nuc_er_for_kept=res.get('nuc_er_for_kept'))
    if deam_parts:  # write_ma_tags rebuilt MA; restore the strand-regime tracks
        seg.set_tag('MA', seg.get_tag('MA') + ';' + ';'.join(deam_parts), value_type='Z')
    return True


def attach_footprint_tags(seg, fp, deam_ma_suffix: str) -> None:
    """Write HMM footprint calls onto a consensus segment.

    Sets legacy ``ns``/``nl`` (footprints) and ``as``/``al`` (MSPs) in query
    coordinates, and a combined ``MA`` tag ``L;nuc.:...;msp.:...;<deam...>`` that
    preserves the strand-regime (``deam+``/``deam-``) suffix.
    """
    import array
    fs = [int(x) for x in fp['footprint_starts']]
    fl = [int(x) for x in fp['footprint_sizes']]
    ms = [int(x) for x in fp['msp_starts']]
    ml = [int(x) for x in fp['msp_sizes']]
    if fs:
        seg.set_tag('ns', array.array('I', fs))
        seg.set_tag('nl', array.array('I', fl))
    if ms:
        seg.set_tag('as', array.array('I', ms))
        seg.set_tag('al', array.array('I', ml))
    parts = [str(seg.query_length)]
    if fs:
        parts.append('nuc.:' + ','.join(f'{s + 1}-{l}' for s, l in zip(fs, fl)))
    if ms:
        parts.append('msp.:' + ','.join(f'{s + 1}-{l}' for s, l in zip(ms, ml)))
    if deam_ma_suffix:
        parts.append(deam_ma_suffix)
    seg.set_tag('MA', ';'.join(parts), value_type='Z')
