from copy import deepcopy
from types import SimpleNamespace
import numpy as np
from fiberhmm.inference.consensus.cross_quantification import summarize_correspondences


def run(prefix,n=2):
    return dict(catalog=[dict(family=f'{prefix}{i}',family_index=i,
        consensus_start=100+i,consensus_end=120+i) for i in range(n)],
        eligible=np.ones((4,n),bool),proposal_membership=np.ones((4,n)),
        native_proposal_membership=np.ones((4,n)))


def edge(a,b):
    return dict(edge_id=a+b,left_dataset='D',right_dataset='H',left_family=a,right_family=b,comparability_mask=True)


def test_star_counts_union_once_on_common_eligibility_and_not_marginal_sum():
    runs={'D':run('D',1),'H':run('H')}
    runs['H']['proposal_membership']=np.array([[.9,0],[0,.9],[.6,.7],[.9,0]])
    runs['H']['eligible'][3,1]=False
    runs['H']['membership']=np.ones((4,2)) # Not the decoded endpoint.
    groups,ann=summarize_correspondences([edge('D0','H0'),edge('D0','H1')],runs)
    g=groups[0]
    assert g['kind']=='coarse_union' and g['status']=='descriptive_counts'
    assert g['right_counts']['eligible_units']==3
    assert g['right_counts']['assigned_units']==3
    assert g['right_counts']['multiple_member_units']==1
    assert not g['quantification_equivalence_established']
    assert not ann['D0H0']['individual_count_comparison_unambiguous']
    assert 'interval' not in g and len(g['native_intervals'])==3


def test_transitive_chain_never_becomes_a_count_group():
    runs={'D':run('D'),'H':run('H')}
    groups,_=summarize_correspondences([edge('D0','H0'),edge('D1','H0'),edge('D1','H1')],runs)
    assert len(groups)==1 and groups[0]['kind']=='unresolved_mapping'
    assert 'nonrectangular_correspondence' in groups[0]['reasons']
    assert 'left_counts' not in groups[0]


def test_disjoint_same_assay_classes_not_silently_combined():
    runs={'D':run('D'),'H':run('H',1)}
    runs['D']['catalog'][1].update(consensus_start=130,consensus_end=150)
    groups,_=summarize_correspondences([edge('D0','H0'),edge('D1','H0')],runs)
    assert groups[0]['reasons']==['no_common_native_core']
    assert 'left_counts' not in groups[0]


def test_group_identity_independent_of_rates_order_and_threshold():
    runs={'D':run('D',1),'H':run('H')}; links=[edge('D0','H0'),edge('D0','H1')]
    before=deepcopy(links)
    g,_=summarize_correspondences(links,runs)
    runs['H']['proposal_membership'][:]=0
    other,_=summarize_correspondences(links[::-1],runs,.9)
    assert g[0]['group_id']==other[0]['group_id']
    assert other[0]['right_counts']['assigned_units']==0
    assert links==before


def test_missing_common_eligibility_not_zero_fraction():
    runs={'D':run('D',1),'H':run('H',1)}
    runs['H']['eligible'][:]=False
    groups,ann=summarize_correspondences([edge('D0','H0')],runs)
    assert groups[0]['status']=='no_common_eligible_units'
    assert groups[0]['right_counts']['fraction'] is None
    assert groups[0]['right_counts']['assigned_units'] is None
    assert not ann['D0H0']['individual_count_comparison_unambiguous']


def test_target_role_ambiguity_preserves_shape_graph_and_coarse_count_endpoint():
    runs={'D':run('D',2),'H':run('H',1)}
    links=[dict(edge('D0','H0'),target_role_comparability_mask=True),
           dict(edge('D1','H0'),target_role_comparability_mask=False)]
    groups,ann=summarize_correspondences(links,runs)
    group=groups[0]
    assert group['left_families']==['D0','D1']
    assert group['left_counts']['assigned_units']==4
    assert group['right_counts']['assigned_units']==4
    assert not group['individual_target_roles_established']
    assert group['target_role_unresolved_edges']==['D1H0']
    assert not group['coarse_event_replacement_validated']
    assert not any(v['individual_count_comparison_unambiguous'] for v in ann.values())


def test_chemical_strata_preserve_exact_union_not_member_threshold_counts(monkeypatch):
    from fiberhmm.inference.consensus import cross_quantification as module
    rr=run('D',2);rr['data']={'strands':np.array(['CT','CT','GA','GA'])}
    rr['proposal_membership']=np.array([[.4,0],[0,.4],[.6,0],[0,0]])
    rr['native_proposal_membership']=np.array([[.4,0],[0,.4],[.6,0],[0,0]])
    rr['core_eligible']=np.array([[False,False],[True,True],[True,True],[False,False]])
    monkeypatch.setattr(module,'_event_mass',lambda run,ix,native=False:
        np.array([.6,.3,.7,.8]) if native else np.array([.8,.6,.7,.4]))
    counts=module._count(rr,['D0','D1'],.5)
    assert counts['assigned_units']==3 and counts['per_member_threshold_assigned_units']==1
    ct,ga=counts['by_strand']['CT'],counts['by_strand']['GA']
    assert (ct['assigned_units'],ct['eligible_units'],ct['native_assigned_units'])==(2,2,1)
    assert (ga['assigned_units'],ga['eligible_units'],ga['native_assigned_units'])==(1,2,1)
    assert ct['canonical_core_eligible_units']==1 and ct['assigned_without_testable_canonical_core']==1
    assert ga['canonical_core_eligible_units']==1 and ga['assigned_without_testable_canonical_core']==0
    for field in ['eligible_units','assigned_units','native_assigned_units','union_eligible_units',
                  'union_eligible_assigned_units','canonical_core_eligible_units','canonical_core_assigned_units',
                  'assigned_without_testable_canonical_core']:
        assert sum(c[field] or 0 for c in counts['by_strand'].values())==(counts[field] or 0)
    assert counts['native_current_endpoints_match']


def test_untestable_stratum_is_not_a_zero_rate_and_hia5_has_one_pooled_stratum():
    from fiberhmm.inference.consensus.cross_quantification import _count
    rr=run('D',1);rr['data']={'strands':np.array(['CT','CT','GA','GA'])};rr['eligible'][2:]=False
    counts=_count(rr,['D0'],.5)
    assert counts['by_strand']['GA']['eligible_units']==0
    assert counts['by_strand']['GA']['assigned_units'] is None
    assert counts['by_strand']['GA']['fraction'] is None
    rr=run('H',1);rr['data']={'strands':np.array(['pooled']*4)}
    assert list(_count(rr,['H0'],.5)['by_strand'])==['pooled']


def test_native_exact_event_fallback_is_explicit_not_silent(monkeypatch):
    from fiberhmm.inference.consensus import cross_quantification as module
    rr=run('D',2)
    monkeypatch.setattr(module,'_event_mass',lambda run,ix,native=False:None if native else np.ones(4))
    counts=module._count(rr,['D0','D1'],.5)
    assert counts['group_event_mass_computed'] and not counts['native_group_event_mass_computed']
    assert not counts['native_current_endpoints_match']
    assert counts['native_endpoint']=='legacy per-member threshold fallback'


def test_stratum_vector_must_match_evidence_units():
    import pytest
    from fiberhmm.inference.consensus.cross_quantification import _count
    rr=run('D',1);rr['data']={'strands':np.array(['CT','GA'])}
    with pytest.raises(ValueError,match='stratum vector'):_count(rr,['D0'],.5)
