"""Consensus scores DddA CpGs exactly as fiberhmm-call / -recall-tfs do.

Production recall (``tf_recaller.build_cpg_mask``, policy ``unmethylated-only``)
excludes every CpG observation except inside confidently unmethylated
``ddda_ucg`` islands. Consensus once used the pre-3.0 rule (exclude only inside
``ddda_mcg``), so methylated-CpG misses on accessible DNA read as protection
and could form classes that fiberhmm-call never supports.
"""
import numpy as np
import pysam
import pytest

from fiberhmm.core.model_io import load_model_with_metadata
from fiberhmm.inference.consensus import adapter
from fiberhmm.inference.strand_rescue import PRESETS, cigar_to_query_ref, hard_observations, resolve_resource_path
from fiberhmm.inference.tf_recaller import (N_CTX, UNMETH_OFFSET, build_cpg_mask, build_llr_tables,
                                            build_m5c_llr_tables, call_tfs_in_interval)

START = 100
# Minimal input: an accessible molecule over a methylated CpG run. The
# CpG cytosines never deaminate; every other C did (Y). Production calls no TF.
SEQ = 'YAA' + 'CG' * 40 + 'AAA'

MA_TAGS = {
    'none': None,
    'ucg_island': f'{len(SEQ)};ddda_ucg.:21-40',  # SEQ [20, 60) is a confidently unmethylated island
    'mcg_only': f'{len(SEQ)};ddda_mcg.:21-40',
}


def ddda_model():
    model, context, mode = load_model_with_metadata(resolve_resource_path(PRESETS['ddda']['model']))
    return model, context, mode


def make_read(ma=None):
    header = pysam.AlignmentHeader.from_dict(dict(HD=dict(VN='1.6'), SQ=[dict(SN='chrSim', LN=10000)]))
    read = pysam.AlignedSegment(header)
    read.query_name = 'r'
    read.query_sequence = SEQ
    read.flag = 0
    read.reference_id = 0
    read.reference_start = START
    read.mapping_quality = 60
    read.cigartuples = [(0, len(SEQ))]
    if ma:
        read.set_tag('MA', ma, value_type='Z')
    return read


def production_calls(read, model, context, mode, policy='unmethylated-only', minimum_llr=5.0):
    """What fiberhmm-recall-tfs decodes inside one MSP spanning the read."""
    obs, _ = hard_observations(read, 'daf', mode, context, None)
    obs = np.asarray(obs, dtype=np.int32)
    hit, miss = build_llr_tables(model)
    m5c_hit, m5c_miss = build_m5c_llr_tables(model)
    mask = build_cpg_mask(read, len(read.query_sequence), policy)
    calls = call_tfs_in_interval(obs, 0, len(obs), hit, miss, minimum_llr, 3, m5c_mask=mask,
                                 m5c_llr_hit=m5c_hit, m5c_llr_miss=m5c_miss, decoder='multi_interval')
    return [[START + c.start, START + c.start + c.length] for c in calls]


def replay(read, model, context, mode, **kw):
    unit = dict(msp_intervals=[[START, START + len(SEQ)]], raw_nuc_intervals=[],
                aligned_blocks=[[START, START + len(SEQ)]])
    calls = adapter.replay_alignment(read, unit, model, 'daf', mode, context, None, 5.0, use_m5c=True, **kw)
    return [c['interval'] for c in calls]


@pytest.mark.parametrize('annotation', sorted(MA_TAGS))
def test_native_replay_matches_production_recall_cpg_policy(annotation):
    model, context, mode = ddda_model()
    read = make_read(MA_TAGS[annotation])
    assert replay(read, model, context, mode) == production_calls(read, model, context, mode)


def test_methylated_cpg_run_gives_no_replayed_tf_without_tag_m5c():
    model, context, mode = ddda_model()
    read = make_read()
    assert production_calls(read, model, context, mode) == []
    assert replay(read, model, context, mode) == []
    # The former policy reads the CpG misses as protection (the bug this guards).
    assert replay(read, model, context, mode, cpg_mask_policy='methylated-only')


def test_unmethylated_island_keeps_its_cpg_evidence():
    model, context, mode = ddda_model()
    read = make_read(MA_TAGS['ucg_island'])
    calls = replay(read, model, context, mode)
    assert calls and all(START + 20 <= a and b <= START + 60 for a, b in calls)


def lattice_unit(read, context, mode):
    """Every informative observation of the read, as the evidence loader builds it."""
    obs, _ = hard_observations(read, 'daf', mode, context, None)
    obs = np.asarray(obs)
    refs = np.asarray(cigar_to_query_ref(read))
    informative = ((obs >= 0) & (obs < N_CTX)) | ((obs >= UNMETH_OFFSET) & (obs < UNMETH_OFFSET + N_CTX))
    index = np.flatnonzero(informative & (refs >= 0))
    hits = obs[index] < N_CTX
    contexts = np.where(hits, obs[index], obs[index] - UNMETH_OFFSET)
    return index, dict(positions=refs[index].tolist(), hits=hits.astype(int).tolist(), contexts=contexts.tolist(),
                       p_accessible=[.5] * len(index), p_protected=[.5] * len(index), provenance={})


@pytest.mark.parametrize('annotation', sorted(MA_TAGS))
def test_lattice_keeps_exactly_the_production_opportunities(annotation):
    model, context, mode = ddda_model()
    read = make_read(MA_TAGS[annotation])
    index, unit = lattice_unit(read, context, mode)
    contexts = np.asarray(unit['contexts'])
    # Production: a masked CpG is no opportunity (neutral m5c tables, no calibrated CpG emissions).
    mask = build_cpg_mask(read, len(SEQ))
    expected = [p for p, i, c in zip(unit['positions'], index, contexts) if not (mask[i] and (c % 64) // 16 == 3)]
    assert len(expected) < len(unit['positions'])  # the fixture has CpGs to exclude
    adapter.condition_unit_on_m5c(read, unit)
    assert unit['positions'] == expected
    assert unit['provenance'].get('native_m5c_excluded_opportunities', 0) == len(index) - len(expected)


def test_lattice_without_annotations_has_no_cpg_opportunity():
    model, context, mode = ddda_model()
    read = make_read()
    _, unit = lattice_unit(read, context, mode)
    adapter.condition_unit_on_m5c(read, unit)
    assert unit['provenance']['native_m5c_excluded_opportunities'] > 0
    assert not any((c % 64) // 16 == 3 for c in unit['contexts'])


def test_consensus_parameter_exposes_the_production_policy():
    from fiberhmm.inference.consensus.parameters import parse_options
    from fiberhmm.inference.tf_recaller import CPG_MASK_POLICIES
    options = parse_options({})
    assert options['input'].ddda_cpg_mask_policy == CPG_MASK_POLICIES[0] == 'unmethylated-only'
    assert parse_options({'input': {'ddda_cpg_mask_policy': 'methylated-only'}})['input'].ddda_cpg_mask_policy == 'methylated-only'
    with pytest.raises(ValueError):
        parse_options({'input': {'ddda_cpg_mask_policy': 'off'}})
    from fiberhmm.inference.consensus.parameters import InputOptions
    from dataclasses import fields
    choices = next(f for f in fields(InputOptions) if f.name == 'ddda_cpg_mask_policy').metadata['choices']
    assert tuple(choices) == CPG_MASK_POLICIES


def test_consensus_mask_is_the_fiberhmm_call_mask():
    """fiberhmm-call builds its recall mask with payload_cpg_mask from the read's
    MA intervals; consensus must produce the identical array from the BAM record."""
    from fiberhmm.inference.fused_stages import payload_cpg_mask
    from fiberhmm.inference.tf_recaller import CPG_MASK_POLICIES, read_cpg_intervals
    for annotation in sorted(MA_TAGS):
        read = make_read(MA_TAGS[annotation])
        for policy in CPG_MASK_POLICIES:
            call_mask = payload_cpg_mask({'_cpg_ma_intervals': read_cpg_intervals(read)}, len(SEQ), policy)
            assert np.array_equal(adapter.m5c_query_mask(read, len(SEQ), policy), call_mask)
        assert np.array_equal(adapter.m5c_query_mask(read, len(SEQ)),
                              payload_cpg_mask({'_cpg_ma_intervals': read_cpg_intervals(read)}, len(SEQ), CPG_MASK_POLICIES[0]))


def saved(**manifest):
    return dict(strata=[dict(dataset_id='d', chemistry='ddda', units=[],
                             model_manifest=dict(preset='ddda', emissions_sha256='0' * 64, **manifest))])


def test_saved_ddda_evidence_must_match_the_requested_cpg_policy():
    from fiberhmm.inference.consensus.bam import check_saved_evidence
    from fiberhmm.inference.consensus.parameters import parse_options
    default = parse_options({})
    with pytest.raises(ValueError, match='Reload the evidence'):
        check_saved_evidence(saved(), default)  # written before the policy was recorded
    with pytest.raises(ValueError, match='methylated-only'):
        check_saved_evidence(saved(cpg_mask_policy='methylated-only'), default)
    check_saved_evidence(saved(cpg_mask_policy='unmethylated-only'), default)
    check_saved_evidence(saved(cpg_mask_policy='off'), parse_options({'input': {'ddda_m5c_correction': False}}))
    # Hand-built payloads carry no loader receipt and are taken as given.
    check_saved_evidence(dict(strata=[dict(dataset_id='d', chemistry='ddda', units=[])]), default)


def test_consensus_cli_refuses_pre_fix_ddda_evidence(tmp_path):
    from fiberhmm.inference.consensus.artifacts import write_json
    from fiberhmm.inference.consensus.cli import main
    path = tmp_path / 'evidence.json.gz'
    payload = saved()
    payload['region'] = dict(chrom='chr1', start=0, end=100)
    write_json(path, payload)
    with pytest.raises(ValueError, match='CpG policy'):
        main(['--evidence', str(path), '--output', str(tmp_path / 'out'), '--no-bam'])


@pytest.mark.parametrize('annotation,islands', [('none', 0), ('mcg_only', 0), ('ucg_island', 1)])
def test_lattice_records_tag_m5c_islands(annotation, islands):
    model, context, mode = ddda_model()
    read = make_read(MA_TAGS[annotation])
    _, unit = lattice_unit(read, context, mode)
    adapter.condition_unit_on_m5c(read, unit)
    assert unit['provenance']['ddda_ucg_islands'] == islands


def test_run_warns_when_no_molecule_has_unmethylated_islands():
    from fiberhmm.inference.consensus.lattice_recaller.workflow import cpg_policy_warnings
    untagged = saved(cpg_mask_policy='unmethylated-only', m5c_conditioned_opportunities=1234, ddda_ucg_molecules=0)
    warning, = cpg_policy_warnings(untagged)
    assert 'fiberhmm-tag-m5c' in warning and '1,234' in warning
    assert not cpg_policy_warnings(saved(cpg_mask_policy='unmethylated-only', m5c_conditioned_opportunities=1234,
                                         ddda_ucg_molecules=7))
    assert not cpg_policy_warnings(saved(cpg_mask_policy='off', m5c_conditioned_opportunities=0))


def pooled(*windows):
    """A CL-CR payload: strata keep the first window's manifest, every window's is in pooling."""
    first = dict(preset='ddda', emissions_sha256='0' * 64, **windows[0])
    return dict(strata=[dict(dataset_id='d', chemistry='ddda', units=[], model_manifest=first)],
                pooling=dict(window_preparation=[dict(window=f'w{i}', dataset_id='d',
                    model=dict(preset='ddda', emissions_sha256='0' * 64, **w)) for i, w in enumerate(windows)]))


def test_pooled_evidence_is_checked_window_by_window():
    from fiberhmm.inference.consensus.bam import check_saved_evidence
    from fiberhmm.inference.consensus.parameters import parse_options
    fresh = dict(cpg_mask_policy='unmethylated-only')
    check_saved_evidence(pooled(fresh, fresh), parse_options({}))
    with pytest.raises(ValueError, match='methylated-only'):
        check_saved_evidence(pooled(fresh, dict(cpg_mask_policy='methylated-only')), parse_options({}))


def test_pooled_warning_counts_every_window():
    from fiberhmm.inference.consensus.lattice_recaller.workflow import cpg_policy_warnings
    untagged = dict(cpg_mask_policy='unmethylated-only', m5c_conditioned_opportunities=10, ddda_ucg_molecules=0)
    tagged = dict(cpg_mask_policy='unmethylated-only', m5c_conditioned_opportunities=5, ddda_ucg_molecules=1)
    assert not cpg_policy_warnings(pooled(untagged, tagged))
    warning, = cpg_policy_warnings(pooled(untagged, untagged))
    assert ' 20 CpG' in warning
