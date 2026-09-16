"""Joint duplex observations and physical identity across CR and transfer."""
from copy import deepcopy
import numpy as np
import pysam
import pytest
from fiberhmm.inference.strand_rescue import hard_observations
from fiberhmm.inference.consensus.bam import load_bam_payload
from fiberhmm.inference.consensus.parameters import parse_options
from fiberhmm.inference.consensus.transfer import molecule_keys
from fiberhmm.inference.consensus.regions import pool_payloads


def joint_bam(path, *, paired=False):
    from fiberhmm.io.bam_header import append_chemistry
    header=pysam.AlignmentHeader.from_dict(dict(HD={'VN':'1.6','SO':'coordinate'},SQ=[{'SN':'chr1','LN':1000}]))
    header=append_chemistry(header,dict(assay='daf',enzyme='ddda',platform='pacbio',mode='daf'))
    r=pysam.AlignedSegment(header);r.query_name='ct.cs';r.query_sequence='CYGR'*50
    r.reference_id=0;r.reference_start=100;r.mapping_quality=60;r.cigarstring='200M'
    r.set_tag('MA','200;deam+:1-200;deam-:1-200;msp.:1-200;tf.:61-20')
    r.set_tag('cs','ct;ga');r.set_tag('pm','D',value_type='A');r.set_tag('mv','ddda-duplex-v1')
    if paired:
        r.set_tag('cs',None);r.set_tag('mt','P',value_type='A');r.set_tag('mp','ga')
        r.query_sequence='CYGT'*50
    with pysam.AlignmentFile(path,'wb',header=header) as out:out.write(r)
    pysam.index(str(path))
    return r


def test_joint_encoder_uses_both_channels_and_coverage_masks(tmp_path):
    from fiberhmm.crossstrand.recall import decode_ry_consensus,encode_daf_both_strand,deam_regime_masks
    r=joint_bam(tmp_path/'joint.bam')
    r.set_tag('MA','200;deam+:1-80,91-110;deam-:41-160')
    obs,strand=hard_observations(r,'daf','daf',3,None)
    sequence,ct,ga=decode_ry_consensus(r.query_sequence)
    expected=encode_daf_both_strand(sequence,ct,ga,*deam_regime_masks(r),edge_trim=10,context_size=3)
    np.testing.assert_array_equal(obs,expected)
    assert strand=='BOTH'
    assert all(obs[i]==8193 for i in range(80,90) if r.query_sequence[i] in 'CY')
    assert all(obs[i]==8193 for i in range(10,40) if r.query_sequence[i] in 'GR')
    assert all(obs[i]!=8193 for i in range(100,150))
    r.set_tag('MA','200;deam+:1-200')
    with pytest.raises(ValueError,match='coverage'):hard_observations(r,'daf','daf',3,None)


def test_joint_bam_is_one_unit_with_both_lattices_and_provenance(tmp_path):
    p=tmp_path/'joint.bam';joint_bam(p)
    result=load_bam_payload([dict(dataset_id='d',paths=[str(p)])],dict(chrom='chr1',start=120,end=220),
        parse_options({'input':{'correct_native':False},'cr':{'engine':'staged_native_families'}}))
    source=result['strata'][0];u,=source['units']
    assert u['strand']=='BOTH' and len(u['positions'])==100
    assert u['physical_source_names']==['ct','ga']
    assert u['pairing_method']=='D' and u['pairing_model']=='ddda-duplex-v1'
    assert u['provenance']['complementary_strands_paired']
    assert source['evidence_units']['joint_duplex_units']==1
    assert source['evidence_units']['physical_duplex_independence_established']
    assert molecule_keys(u)&molecule_keys(dict(read_name='ga'))


def test_unmerged_pair_is_rejected_before_population_fit(tmp_path):
    p=tmp_path/'paired.bam';joint_bam(p,paired=True)
    with pytest.raises(ValueError,match='fiberhmm-merge'):
        load_bam_payload([dict(dataset_id='d',paths=[str(p)])],dict(chrom='chr1',start=120,end=220))


def test_pooled_source_and_joint_views_share_physical_identity():
    from test_staged_families import payload
    a=payload();a['strata'][0]['units']=a['strata'][0]['units'][:1]
    u=a['strata'][0]['units'][0];u['read_name']='ct.cs';u['physical_source_names']=['ct','ga']
    b=deepcopy(a);b['strata'][0]['units'][0].update(read_name='ga',physical_source_names=[])
    windows=[dict(chrom='chr1',start=0,end=81,strand='+',name=n) for n in ['one','two']]
    result=pool_payloads([a,b],windows)
    assert result['pooling']['retained_molecules']==1
    assert len(result['pooling']['excluded_repeated_views'])==1
    with pytest.raises(ValueError,match='Repeated physical molecule'):
        pool_payloads([a,b],[windows[0],windows[0]])
