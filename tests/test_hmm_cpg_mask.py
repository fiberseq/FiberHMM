"""First-pass HMM CpG masking for DddA CpG-aware calling."""
import numpy as np

import fiberhmm.inference.engine as engine
from fiberhmm.core.bam_reader import (
    ContextEncoder,
    daf_cpg_target_mask,
    encode_from_query_sequence,
)


def test_cpg_targets_ct_strand():
    # C->T strand: target C/T followed by G. Positions 4 (CG) and 9 (TG, a
    # deaminated CpG) are CpGs; 1 (CA) and 6 (CC) are not.
    seq = 'ACAACGCCATGA'
    mods = {9}  # the T at 9 is a converted C
    mask = daf_cpg_target_mask(seq, mods, '+')
    assert mask.tolist() == [i in (4, 9) for i in range(len(seq))]


def test_cpg_targets_ga_strand():
    # G->A strand: target G/A preceded by C. 'CG' at 2-3 and 'CA' (converted
    # CpG) at 6-7 are CpGs; the G at 9 follows T and is not.
    seq = 'ATCGTTCAATGA'
    mods = {7}
    mask = daf_cpg_target_mask(seq, mods, '-')
    assert mask.tolist() == [i in (3, 7) for i in range(len(seq))]


def _fiber_read(seq, mods):
    return {'read_id': 'r', 'query_sequence': seq, 'm6a_query_positions': set(mods),
            'query_length': len(seq), 'is_reverse': False, '_daf_strand': '+'}


def _capture(monkeypatch):
    seen = {}

    def fake_predict(model, encoded, *args, **kwargs):
        seen['encoded'] = np.array(encoded, copy=True)
        return {'footprint_starts': np.array([0]), 'footprint_sizes': np.array([5]),
                'msp_starts': np.array([]), 'msp_sizes': np.array([])}

    monkeypatch.setattr(engine, 'predict_footprints_and_msps', fake_predict)
    return seen


def test_hmm_sees_masked_cpgs_recall_keeps_them(monkeypatch):
    seq = 'ATTACGATCAGTTACGTACCATCGATTACA' * 2
    mods = {i for i, b in enumerate(seq) if b == 'T'}
    seen = _capture(monkeypatch)
    plain = encode_from_query_sequence(seq, mods, 0, mode='daf', strand='+', context_size=3)
    fr = _fiber_read(seq, mods)
    fr['_hmm_cpg_mask'] = np.ones(len(seq), dtype=bool)
    res = engine._process_single_read(fr, None, 0, False, 'daf', 3, 1, False,
                                      include_encoded=True)
    cpg = daf_cpg_target_mask(seq, mods, '+')
    non_target = 2 * ContextEncoder.get_n_codes(3) + 1
    assert cpg.any()
    hmm = seen['encoded']
    assert (hmm[cpg] == non_target).all()
    assert (hmm[~cpg] == plain[~cpg]).all()
    # Recall receives the unmasked encoding.
    assert (np.asarray(res['encoded']) == plain).all()


def test_mask_respects_unmethylated_islands(monkeypatch):
    seq = 'ATTACGATCAGTTACGTACCATCGATTACA' * 2
    mods = set()
    seen = _capture(monkeypatch)
    plain = encode_from_query_sequence(seq, mods, 0, mode='daf', strand='+', context_size=3)
    keep = np.ones(len(seq), dtype=bool)
    keep[:30] = False  # an unmethylated island: CpGs here stay visible
    fr = _fiber_read(seq, mods)
    fr['_hmm_cpg_mask'] = keep
    engine._process_single_read(fr, None, 0, False, 'daf', 3, 1, False)
    cpg = daf_cpg_target_mask(seq, mods, '+')
    hmm = seen['encoded']
    assert (hmm[cpg & ~keep] == plain[cpg & ~keep]).all()
    assert (hmm[cpg & keep] != plain[cpg & keep]).all()


def test_no_mask_is_unchanged(monkeypatch):
    seq = 'ATTACGATCAGTTACGTACCATCGATTACA' * 2
    mods = {3, 12}
    seen = _capture(monkeypatch)
    plain = encode_from_query_sequence(seq, mods, 0, mode='daf', strand='+', context_size=3)
    engine._process_single_read(_fiber_read(seq, mods), None, 0, False, 'daf', 3, 1, False)
    assert (seen['encoded'] == plain).all()
