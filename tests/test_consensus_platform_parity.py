"""Consensus reads and replays each platform exactly as fiberhmm-call does.

The Hia5 replay resolves its nucleosome recall policy with fiberhmm-call's own
resolver (``auto`` is conservative on every platform) and records it.
"""
from types import SimpleNamespace

import numpy as np
import pysam
import pytest

from fiberhmm.inference import strand_rescue
from fiberhmm.inference.consensus.upstream_recall import recall_hia5_alignment
from fiberhmm.inference.fused_stages import _build_fused_recall_result_with_nucs, resolve_nuc_recall_policy
from fiberhmm.inference.tf_recaller import build_llr_tables


def toy_model():
    ep = np.zeros((2, 8193))
    ep[0, :4096] = .02; ep[0, 4097:] = .98
    ep[1, :4096] = .8; ep[1, 4097:] = .2
    return SimpleNamespace(emissionprob_=ep)


def nucleosome_with_sparse_hits(monkeypatch):
    """One protected 130-bp footprint with a few accessible hits inside it.

    Conservative recall (fiberhmm-call's default) splits it and releases two
    TF-sized pieces; the retired topology policy keeps the nucleosome whole."""
    obs = np.full(400, 4096, np.int32); obs[::2] = 0
    obs[130:260:2] = 4097
    obs[186:205:2] = 0
    monkeypatch.setattr(strand_rescue, 'hard_observations', lambda *_: (obs, 'FWD'))
    read = pysam.AlignedSegment()
    read.query_sequence = 'A' * 400
    read.reference_id = 0; read.reference_start = 100; read.cigarstring = '400M'
    read.set_tag('ns', [130]); read.set_tag('nl', [130])
    read.set_tag('as', [0, 260]); read.set_tag('al', [130, 140])
    unit = dict(unit_id='u', _region=[100, 500], raw_nuc_intervals=[[230, 360]],
                msp_intervals=[[100, 230], [360, 500]], raw_tf_intervals=[], aligned_blocks=[[100, 500]])
    return obs, read, unit


def production(obs, mode):
    """fiberhmm-call's nucleosome+TF recall of the same read (auto policy for ``mode``)."""
    hit, miss = build_llr_tables(toy_model())
    result = _build_fused_recall_result_with_nucs(
        dict(query_sequence='A' * len(obs)),
        dict(encoded=obs, ns=[130], nl=[130], **{'as': [0, 260], 'al': [130, 140]}),
        hit, miss, min_llr=5, min_opps=3, unify_threshold=90, split_min_llr=4, split_min_opps=3,
        nuc_min_size=85, msp_min_size=0, phase_nrl=0,
        nuc_recall_policy=resolve_nuc_recall_policy('auto', mode), nuc_llr_hit=hit, nuc_llr_miss=miss)
    nucs = [[100 + a, 100 + a + b] for a, b in zip(result['ns'].tolist(), result['nl'].tolist())]
    return nucs, [[100 + c.start, 100 + c.start + c.length] for c in result['tf_calls']]


@pytest.mark.parametrize('mode,policy', [('nanopore-fiber', 'conservative'), ('pacbio-fiber', 'conservative')])
def test_hia5_replay_uses_the_calling_nucleosome_policy(monkeypatch, mode, policy):
    obs, read, unit = nucleosome_with_sparse_hits(monkeypatch)
    result = recall_hia5_alignment(read, unit, toy_model(), 'alignment', mode, 3, 125, 5,
                                   legacy_annotation_frame='seq')
    assert result['settings']['nuc_recall_policy'] == policy
    nucs, tfs = production(obs, mode)
    assert [c['interval'] for c in result['calls']] == tfs
    assert result['nucleosomes'] == nucs


def test_nanopore_replay_policy_is_passed_through_and_recorded(monkeypatch):
    obs, read, unit = nucleosome_with_sparse_hits(monkeypatch)
    ont = recall_hia5_alignment(read, unit, toy_model(), 'alignment', 'nanopore-fiber', 3, 125, 5,
                                legacy_annotation_frame='seq')
    assert len(ont['calls']) == 2 and ont['settings']['nuc_recall_policy'] == 'conservative'
    # An explicitly named (retired) topology policy reaches the recaller and is recorded.
    forced = recall_hia5_alignment(read, unit, toy_model(), 'alignment', 'nanopore-fiber', 3, 125, 5,
                                   legacy_annotation_frame='seq', nuc_recall_policy='topology')
    assert forced['calls'] == [] and forced['nucleosomes'] == [[230, 360]]
    assert forced['settings']['nuc_recall_policy'] == 'topology'


def test_policy_resolution_is_shared_with_the_calling_commands():
    from fiberhmm.cli import call, recall_tfs
    for mode in ('nanopore-fiber', 'pacbio-fiber', 'daf'):
        assert resolve_nuc_recall_policy('auto', mode) == 'conservative'
    for mode in ('nanopore-fiber', 'pacbio-fiber', 'daf'):
        for policy in ('auto', 'conservative', 'topology'):
            args = SimpleNamespace(nuc_recall_policy=policy)
            expected = resolve_nuc_recall_policy(policy, mode)
            assert call._resolve_nuc_recall_policy(args, mode) == expected
            assert recall_tfs._resolve_nuc_recall_policy(args, mode) == expected
    with pytest.raises(ValueError):
        resolve_nuc_recall_policy('aggressive', 'pacbio-fiber')

