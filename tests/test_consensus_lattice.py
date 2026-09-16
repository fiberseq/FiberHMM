from itertools import product
import numpy as np
from scipy.special import logsumexp

from fiberhmm.inference.consensus.lattice import RegionFamilyLattice


def enumerate_model(k, values, eta):
    configs=[]
    for choices in product(*[range(-1,len(f['q'])) for f in k.families]):
        spans=sorted((k.families[f]['starts'][j],k.families[f]['ends'][j])
                     for f,j in enumerate(choices) if j>=0)
        if any(b>=c for (a,b),(c,d) in zip(spans,spans[1:])):
            continue
        mask=np.zeros(k.k); incidence=np.zeros(k.f); logp=0.
        for f,j in enumerate(choices):
            if j>=0:
                q=k.families[f]; mask[q['starts'][j]:q['ends'][j]]=1
                incidence[f]=1; logp+=eta[f]+np.log(q['q'][j])
        configs.append((mask,incidence,logp))
    lp=np.array([x[2] for x in configs]); obs=values @ np.array([x[0] for x in configs]).T+lp
    z=logsumexp(obs,axis=1); post=np.exp(obs-z[:,None])
    return z,post @ np.array([x[1] for x in configs]),logsumexp(lp)


def test_whole_region_matches_exhaustive_and_never_repeats_tiny_family():
    k=RegionFamilyLattice(np.arange(8),[[1,3],[3,5],[0,7]],2)
    # +/-2 permits two disjoint realizations of family 0. They must NOT both
    # occur in one path, even though ordinary weighted interval DP allows it.
    values=np.random.default_rng(31).normal(size=(3,8)); eta=np.array([.4,-.8,.6])
    z,inc,z0=enumerate_model(k,values,eta)
    got=k.evaluate(values,eta,export_geometry=True)
    np.testing.assert_allclose(got['log_partition'],z,atol=1e-11)
    np.testing.assert_allclose(got['family_inclusion'],inc,atol=1e-11)
    np.testing.assert_allclose(k.evaluate(np.zeros((1,8)),eta)['log_partition'],z0,atol=1e-11)
    for f in range(k.f):
        np.testing.assert_allclose(got['geometry_mass'][:,k.gf==f].sum(1),inc[:,f],atol=1e-11)


def test_no_window_family_or_run_cap_and_independent_site_factorization():
    p=np.arange(120); c=np.array([[j,j+3] for j in range(0,120,6)])
    k=RegionFamilyLattice(p,c,0); v=np.full((1,len(p)),5.); eta=np.zeros(len(c))
    got=k.evaluate(v,eta)
    assert k.f==20 and got['family_inclusion'].sum()>19.99
    np.testing.assert_allclose(got['log_partition'],20*np.logaddexp(0,15),atol=1e-10)
    # Every independent family is present with probability 1/2 under eta=0.
    np.testing.assert_allclose(k.evaluate(np.zeros_like(v),eta)['family_inclusion'],.5,atol=1e-11)


def test_prior_normalizer_gradient_missing_and_translation():
    pos=np.arange(9); centers=np.array([[0,3],[2,6],[6,9]])
    k=RegionFamilyLattice(pos,centers,1); v=np.random.default_rng(72).normal(size=(4,9)); eta=np.array([-.3,.2,.7])
    val,grad=k.objective(v,eta)
    for f in range(k.f):
        step=np.eye(k.f)[f]*1e-5
        numerical=(k.objective(v,eta+step)[0]-k.objective(v,eta-step)[0])/2e-5
        np.testing.assert_allclose(grad[f],numerical,atol=1e-8)
    np.testing.assert_allclose(k.objective(np.zeros_like(v),eta)[0],0,atol=1e-12)
    np.testing.assert_allclose(k.objective(np.zeros_like(v),eta)[1],0,atol=1e-12)
    shifted=RegionFamilyLattice(pos+1000,centers+1000,1)
    np.testing.assert_allclose(shifted.evaluate(v,eta)['log_partition'],k.evaluate(v,eta)['log_partition'],atol=1e-11)




def test_saturated_bridge_retains_broad_alternative_but_favors_separate_patches():
    k=RegionFamilyLattice(np.arange(15),[[0,4],[6,9],[11,15],[0,15]],0)
    v=np.array([[2.]*4+[-8.]*2+[2.]*3+[-8.]*2+[2.]*4])
    out=k.evaluate(v,np.zeros(4))
    assert out['family_inclusion'][0,:3].min()>.99
    assert out['family_inclusion'][0,3]<1e-10
    protected=k.evaluate(np.full((1,15),2.),np.zeros(4))
    assert protected['family_inclusion'][0,3]>.99


def test_resource_budget_fails_explicitly_not_by_truncating_calls():
    import pytest
    with pytest.raises(MemoryError,match='no candidate silently removed'):
        RegionFamilyLattice(np.arange(20),[[0,5],[7,12]],2,maximum_nodes=2)


def test_unit_specific_admission_is_in_both_partitions_not_a_score_bonus():
    k=RegionFamilyLattice(np.arange(12),[[0,4],[6,10]],1)
    allowed=np.array([[True,False],[False,True],[False,False]])
    eta=np.array([.7,-.4]);v=np.random.default_rng(17).normal(size=(3,12))
    out=k.evaluate(v,eta,allowed=allowed)
    assert np.all(out['family_inclusion'][~allowed]==0)
    assert out['log_partition'][2]==0
    for m in range(2):
        sub=RegionFamilyLattice(np.arange(12),[k.centers[m]],1)
        expected=sub.evaluate(v[m:m+1],eta[m:m+1])
        np.testing.assert_allclose(out['log_partition'][m],expected['log_partition'][0],atol=1e-11)
    np.testing.assert_allclose(k.objective(np.zeros_like(v),eta,allowed=allowed)[0],0,atol=1e-12)
    np.testing.assert_allclose(k.objective(np.zeros_like(v),eta,allowed=allowed)[1],0,atol=1e-12)
    _,grad=k.objective(v,eta,allowed=allowed)
    for f in range(2):
        d=np.eye(2)[f]*1e-5
        numeric=(k.objective(v,eta+d,allowed=allowed)[0]-k.objective(v,eta-d,allowed=allowed)[0])/2e-5
        np.testing.assert_allclose(grad[f],numeric,atol=1e-8)


def test_joint_map_matches_exhaustive_not_marginal_or_greedy_assignment():
    k=RegionFamilyLattice(np.arange(8),[[1,3],[3,5],[0,7]],2)
    values=np.random.default_rng(31).normal(size=(3,8));eta=np.array([.4,-.8,.6])
    got=k.map_configuration(values,eta)
    for m in range(len(values)):
        weights=[]
        for choice in product(*[range(-1,len(f['q'])) for f in k.families]):
            spans=sorted((k.families[f]['starts'][j],k.families[f]['ends'][j])
                         for f,j in enumerate(choice) if j>=0)
            if any(b>=c for (a,b),(c,d) in zip(spans,spans[1:])):continue
            weights.append(sum(eta[f]+np.log(k.families[f]['q'][j])+
                values[m,k.families[f]['starts'][j]:k.families[f]['ends'][j]].sum()
                for f,j in enumerate(choice) if j>=0))
        np.testing.assert_allclose(got['log_weight'][m],max(weights),atol=1e-12)
        gs=got['geometry_by_family'][m];gs=gs[gs>=0]
        assert len(gs)==len(set(k.gf[gs]))
        np.testing.assert_allclose(sum(eta[k.gf[g]]+k.logq[g]+values[m,k.ga[g]:k.gb[g]].sum()
                                      for g in gs),got['log_weight'][m],atol=1e-12)


def test_map_action_mask_never_changes_posterior_model():
    k=RegionFamilyLattice(np.arange(12),[[0,4],[6,10]],1)
    values=np.full((2,12),4.);eta=np.zeros(2)
    before=k.evaluate(values,eta)['family_inclusion'].copy()
    mask=np.ones((2,len(k.ga)),bool);mask[0,k.gf==0]=False;mask[1]=False
    got=k.map_configuration(values,eta,geometry_allowed=mask)
    assert got['geometry_by_family'][0,0]==-1
    assert got['geometry_by_family'][0,1]>=0
    assert (got['geometry_by_family'][1]==-1).all()
    assert got['log_weight'][1]==0
    np.testing.assert_array_equal(k.evaluate(values,eta)['family_inclusion'],before)


def test_map_keeps_more_than_two_or_three_nonoverlapping_families():
    k=RegionFamilyLattice(np.arange(120),[[j,j+3] for j in range(0,120,6)],0)
    got=k.map_configuration(np.full((1,120),5.),np.zeros(k.f))
    assert (got['geometry_by_family']>=0).sum()==20


def test_restricted_fit_uses_same_geometry_base_measure_in_both_partitions():
    k=RegionFamilyLattice(np.arange(10),[[1,5],[4,8]],1)
    rng=np.random.default_rng(492);v=rng.normal(size=(4,10));eta=np.array([.3,-.7])
    physical=rng.random((4,len(k.ga)))>.35
    physical[-1]=False
    adjustment=np.log(rng.uniform(.05,1,size=physical.shape))
    kw=dict(geometry_allowed=physical,geometry_log_adjustment=adjustment)
    value,grad=k.objective(v,eta,**kw)
    prior=k.evaluate(np.zeros_like(v),eta,**kw);post=k.evaluate(v,eta,**kw)
    np.testing.assert_allclose(value,np.mean(prior['log_partition']-post['log_partition']))
    for f in range(k.f):
        d=np.eye(k.f)[f]*1e-5
        numerical=(k.objective(v,eta+d,**kw)[0]-k.objective(v,eta-d,**kw)[0])/2e-5
        np.testing.assert_allclose(grad[f],numerical,atol=1e-8)
    empty,empty_grad=k.objective(np.zeros_like(v),eta,**kw)
    assert empty==0
    np.testing.assert_array_equal(empty_grad,0)
    # Legacy callers omit restrictions and are numerically unchanged.
    old=k.objective(v,eta)
    full=k.objective(v,eta,geometry_allowed=np.ones_like(physical),geometry_log_adjustment=np.zeros_like(adjustment))
    np.testing.assert_allclose(old[0],full[0],atol=1e-12)
    np.testing.assert_allclose(old[1],full[1],atol=1e-12)


def test_any_event_mass_bounds_and_monotonicity_with_overlapping_and_disjoint_members():
    k=RegionFamilyLattice(np.arange(15),[[0,4],[2,7],[9,14]],1)
    v=np.random.default_rng(72).normal(size=(7,15));eta=np.array([-.3,.2,.7])
    marginal=k.evaluate(v,eta)['family_inclusion']
    previous=np.zeros(len(v))
    for group in ([0],[0,1],[0,1,2]):
        got=k.any_family_inclusion(v,eta,group)
        assert np.all(got+1e-12>=marginal[:,group].max(1))
        assert np.all(got<=np.minimum(1,marginal[:,group].sum(1))+1e-12)
        assert np.all(got+1e-12>=previous)
        previous=got


def test_any_event_sampler_includes_cooccurring_targets_not_only_exactly_one():
    from fiberhmm.inference.consensus.comparability import sample_conditional_family_event
    k=RegionFamilyLattice(np.arange(12),[[0,3],[5,8],[9,12]],0)
    targets=np.array([True,True,False]);eta=np.zeros(3)
    arguments=(k.offsets,k.dest,k.edge_geo,k.ga,k.gb,k.gf,eta[k.gf]+k.logq,
        np.ones(len(k.ga),bool),k.n_nodes,k.f,targets)
    pa=np.full(k.k,.8);pp=np.full(k.k,.1);obs=np.ones(k.k,bool)
    _,yes,possible=sample_conditional_family_event(*arguments,True,pa,pp,obs,12000,128)
    assert possible and ((yes[:,:2]>=0).any(1)).all()
    # At independent prior .5 each, P(both | ANY)=1/3, not zero.
    np.testing.assert_allclose((yes[:,:2]>=0).all(1).mean(),1/3,atol=.02)
    np.testing.assert_allclose((yes[:,2]>=0).mean(),.5,atol=.02)
    _,no,possible=sample_conditional_family_event(*arguments,False,pa,pp,obs,2000,129)
    assert possible and (no[:,:2]<0).all()
    np.testing.assert_allclose((no[:,2]>=0).mean(),.5,atol=.04)


def test_geometry_event_marginal_agrees_with_class_event_and_overlapping_alias_union():
    k=RegionFamilyLattice(np.arange(10),[[1,5],[3,7]],1)
    rng=np.random.default_rng(161);values=rng.normal(size=(5,10));eta=np.array([-.2,.7])
    kw=dict(geometry_allowed=rng.random((5,len(k.ga)))>.2,
        geometry_log_adjustment=np.log(rng.uniform(.1,1,(5,len(k.ga)))))
    for indices in ([0],[0,1]):
        np.testing.assert_allclose(k.geometry_event_inclusion(values,eta,np.isin(k.gf,indices),**kw),
            k.any_family_inclusion(values,eta,indices,**kw),atol=1e-12)
    # Two overlapping labels can represent the same observable central feature.
    all_geometries=np.ones(len(k.ga),bool)
    out=k.geometry_event_inclusion(values,eta,all_geometries,**kw)
    np.testing.assert_allclose(out,1-np.exp(-k.evaluate(values,eta,**kw)['log_partition']))
    np.testing.assert_array_equal(k.geometry_event_inclusion(values,eta,np.zeros(len(k.ga),bool),**kw),0)


def test_partly_identified_alias_event_is_not_resolved_by_a_display_coordinate():
    from fiberhmm.inference.consensus.comparability import sample_conditional_alias_event
    k=RegionFamilyLattice(np.arange(8),[[1,4],[5,8]],0)
    eta=np.zeros(2);values=np.zeros((1,8));event=np.array([.5,0.])
    np.testing.assert_allclose(k.geometry_event_inclusion(values,eta,event),.25)
    args=(k.offsets,k.dest,k.edge_geo,k.ga,k.gb,k.gf,k.logq,
        np.ones(len(k.ga),bool),k.n_nodes,k.f,event)
    pa=np.full(8,.8);pp=np.full(8,.1);observed=np.ones(8,bool)
    _,yes,ok=sample_conditional_alias_event(*args,True,pa,pp,observed,4000,729)
    assert ok and (yes[:,0]>=0).all()
    _,no,ok=sample_conditional_alias_event(*args,False,pa,pp,observed,12000,730)
    assert ok
    # A non-event alias may still contain the first footprint. Its mass is
    # .5*.5 / (1-.5*.5) = 1/3, not zero as a Boolean representative would imply.
    np.testing.assert_allclose((no[:,0]>=0).mean(),1/3,atol=.02)
    np.testing.assert_allclose((no[:,1]>=0).mean(),.5,atol=.02)
