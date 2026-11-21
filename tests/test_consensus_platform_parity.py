"""Consensus reads and replays each platform exactly as fiberhmm-call does.

The Hia5 replay resolves its nucleosome recall policy with fiberhmm-call's own
resolver (``auto`` is conservative on every platform) and records it, and
legacy ``Mm``/``Ml`` modification tags are read as fiberhmm-call reads them.
"""
import array
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


def hia5_read(header, name, *, legacy_tags=False, reverse=False):
    rng = np.random.default_rng(7)
    seq = ''.join(rng.choice(list('ACGT'), size=300))
    read = pysam.AlignedSegment(header)
    read.query_name = name; read.query_sequence = seq
    read.flag = 16 if reverse else 0
    read.reference_id = 0; read.reference_start = 100; read.mapping_quality = 60
    read.cigartuples = [(0, len(seq))]
    adenines = [i for i, b in enumerate(seq) if b == 'A']
    called = adenines[::3]
    skips, last = [], -1
    for index in called:
        skips.append(sum(1 for a in adenines if last < a < index)); last = index
    mm = 'A+a?,' + ','.join(map(str, skips)) + ';'
    ml = [250 if k % 2 else 40 for k in range(len(called))]
    read.set_tag('Mm' if legacy_tags else 'MM', mm, 'Z')
    read.set_tag('Ml' if legacy_tags else 'ML', array.array('B', ml))
    return read


@pytest.mark.parametrize('reverse', [False, True])
def test_hard_observations_read_legacy_mm_ml_like_fiberhmm_call(reverse):
    header = pysam.AlignmentHeader.from_dict(dict(HD=dict(VN='1.6'), SQ=[dict(SN='chr1', LN=1000)]))
    modern = strand_rescue.hard_observations(hia5_read(header, 'r', reverse=reverse), 'alignment', 'pacbio-fiber', 3, 125)
    legacy = strand_rescue.hard_observations(hia5_read(header, 'r', legacy_tags=True, reverse=reverse),
                                             'alignment', 'pacbio-fiber', 3, 125)
    assert modern is not None and legacy is not None
    np.testing.assert_array_equal(modern[0], legacy[0])
    assert modern[1] == legacy[1]


def test_region_loader_keeps_reads_with_legacy_mm_ml(tmp_path):
    from fiberhmm.core.model_io import load_model_with_metadata
    header = pysam.AlignmentHeader.from_dict(dict(HD=dict(VN='1.6', SO='coordinate'), SQ=[dict(SN='chr1', LN=1000)]))
    model, context, mode = load_model_with_metadata(
        strand_rescue.resolve_resource_path('fiberhmm/models/hia5_pacbio.json'))
    hit, miss = build_llr_tables(model)
    counts = {}
    for legacy in (False, True):
        path = tmp_path / f'legacy{legacy}.bam'
        with pysam.AlignmentFile(str(path), 'wb', header=header) as out:
            for i in range(4):
                out.write(hia5_read(header, f'r{i}', legacy_tags=legacy, reverse=bool(i % 2)))
        pysam.index(str(path))
        reads = strand_rescue.load_region_evidence(
            str(path), 'chr1', 100, 400, strand_mode='alignment', mode=mode, context_size=context,
            prob_threshold=125, llr_hit=hit, llr_miss=miss, min_mapq=0, max_reads=0)
        counts[legacy] = [(r.name, len(r.positions), int(np.sum(r.hits))) for r in reads]
    assert counts[False] and counts[True] == counts[False]


def test_empty_ml_is_no_observation_as_in_fiberhmm_call():
    """fiberhmm-call skips a read whose ML is empty; it must not become an all-unmodified read."""
    header = pysam.AlignmentHeader.from_dict(dict(HD=dict(VN='1.6'), SQ=[dict(SN='chr1', LN=1000)]))
    for mm_name, ml_name in (('MM', 'ML'), ('Mm', 'Ml')):
        read = pysam.AlignedSegment(header)
        read.query_name = 'r'; read.query_sequence = 'ACGT' * 75; read.flag = 0
        read.reference_id = 0; read.reference_start = 100; read.cigartuples = [(0, 300)]
        read.set_tag(mm_name, 'A+a.;', 'Z'); read.set_tag(ml_name, array.array('B', []))
        assert strand_rescue.hard_observations(read, 'alignment', 'pacbio-fiber', 3, 125) is None


def saved_stratum(chemistry, **manifest):
    return dict(dataset_id='d', chemistry=chemistry, units=[],
                model_manifest=dict(preset=chemistry, emissions_sha256='0' * 64, **manifest))


def test_saved_nanopore_evidence_must_match_the_calling_replay_policy():
    from fiberhmm.inference.consensus.bam import check_saved_evidence
    from fiberhmm.inference.consensus.parameters import parse_options
    options = parse_options({})
    # Unrecorded (older evidence) was replayed conservatively: still fiberhmm-call's policy.
    old = dict(strata=[saved_stratum('hia5-nanopore', upstream_nuc_tf_recall=True,
                                     upstream_nuc_tf_settings=dict(minimum_llr=5.0))])
    check_saved_evidence(old, options)
    new = dict(strata=[saved_stratum('hia5-nanopore', upstream_nuc_tf_recall=True,
                                     upstream_nuc_tf_settings=dict(nuc_recall_policy='conservative'))])
    check_saved_evidence(new, options)
    topology = dict(strata=[saved_stratum('hia5-nanopore', upstream_nuc_tf_recall=True,
                                          upstream_nuc_tf_settings=dict(nuc_recall_policy='topology'))])
    with pytest.raises(ValueError, match='Reload the evidence'):
        check_saved_evidence(topology, options)
    pacbio = dict(strata=[saved_stratum('hia5-pacbio', upstream_nuc_tf_recall=True,
                                        upstream_nuc_tf_settings=dict(nuc_recall_policy='topology'))])
    check_saved_evidence(pacbio, options)  # only the Nanopore replay is checked
