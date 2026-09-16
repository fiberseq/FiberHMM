import copy
import json
import numpy as np
from fiberhmm.inference.consensus.measurement_catalog import classify_catalog


def fixture():
    calls=[dict(unit_id=f'u{i}',unit_index=i,ordinal=0,strand='CT',start=a,end=b)
           for i,(a,b) in enumerate([(10,50),(10,52),(10,25),(11,26),(100,120)])]
    catalog=[dict(family='d:broad',consensus_start=10,consensus_end=50),
             dict(family='d:small',consensus_start=10,consensus_end=25)]
    i,j=np.triu_indices(4,1)
    pairs=dict(first=i,second=j,native_loss=np.array([0.,2.,2.,2.,2.,0.]),
               informative_both=np.ones(6,bool),core_contradicted=np.zeros(6,bool),
               edge_floor_compatible=np.zeros(6,bool))
    return calls,catalog,pairs


def test_compatible_broad_calls_keep_full_span_and_broad_label():
    calls,cat,pairs=fixture(); frozen=copy.deepcopy(calls)
    r=classify_catalog(calls,cat,pairs)
    a=r['partitions']['100.0']['assignments']
    assert [v['family'] for v in a[:4]]==['d:broad','d:broad','d:small','d:small']
    assert a[1]['interval']==[10,52]
    assert a[1]['compatible_alternatives']==['d:small']
    assert a[4]['classification_status']=='provisional_unresolved'
    assert calls==frozen
    assert len(a)==len(calls)
    json.dumps(r,allow_nan=False)


def test_native_distinction_and_core_veto_do_not_disappear_with_median():
    calls,cat,pairs=fixture()
    pairs['native_loss'][0]=30
    pairs['core_contradicted'][5]=True
    r=classify_catalog(calls,cat,pairs)
    a=r['partitions']['100.0']['assignments']
    assert a[0]['family']=='d:small'
    assert a[2]['family']=='d:broad'
    by={v['family']:v for v in r['call_family_evidence'][2]}
    assert by['d:small']['effective_loss_quantile'] is None


def test_unobserved_is_not_compatibility_and_region_is_explicit():
    calls,cat,pairs=fixture();pairs['informative_both'][:]=False
    pairs['native_loss'][:]=np.nan
    r=classify_catalog(calls,cat,pairs,region=(0,60))
    assert len(r['partitions']['100.0']['assignments'])==4
    assert r['partitions']['100.0']['unresolved_calls']==4
    assert r['diagnostics']['out_of_region_calls']==1


def test_catalog_order_does_not_change_labels_or_evidence():
    calls,cat,pairs=fixture()
    a=classify_catalog(calls,cat,pairs)
    b=classify_catalog(calls,cat[::-1],pairs)
    for lev in a['partitions']:
        assert [v['family'] for v in a['partitions'][lev]['assignments']]==[v['family'] for v in b['partitions'][lev]['assignments']]


def test_low_count_does_not_hide_compatible_class_and_threshold_sets_are_nested():
    calls,cat,pairs=fixture()
    r=classify_catalog(calls,cat,pairs,loss_odds_levels=(1,10,100))
    assert r['partitions']['1.0']['assignments'][0]['family']=='d:broad'
    assert r['partitions']['1.0']['assignments'][0]['compatible_alternatives']==[]
    assert r['partitions']['10.0']['assignments'][0]['compatible_alternatives']==['d:small']


def test_duplicate_source_unit_contributes_only_once():
    calls,cat,pairs=fixture();calls[1]['evidence_group_id']='dupe';calls[2]['evidence_group_id']='dupe'
    # Both source spans belong to the broad proposal. The closest source
    # occurrence is used, rather than selecting the one with lowest loss.
    cat=cat[:1]
    r=classify_catalog(calls,cat,pairs)
    evidence=r['call_family_evidence'][3][0]
    assert evidence['eligible_source_units']==2
