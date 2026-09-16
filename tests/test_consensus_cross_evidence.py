import numpy as np
from scipy.special import logsumexp

from fiberhmm.inference.consensus.cross_evidence import (prefix,shape_predictive,direction,
    plausible_pairs,link_status)
from fiberhmm.inference.consensus.projection import bounded_projection


def fixture(values,observed=None):
    values=np.atleast_2d(values).astype(float);n,k=values.shape
    if observed is None:observed=np.ones_like(values,dtype=bool)
    return dict(start=0,end=k,positions=np.arange(k),llr=prefix(values),
                observed=prefix(observed),hits=prefix(values<0),invalid=prefix(np.zeros_like(values)))


def test_exact_integer_integration_matches_existing_cr_kernel():
    data=fixture(np.arange(30)[None]%4-1.5)
    for x in (0,5,10):
        center=[12,18]
        q=bounded_projection(data['positions'],center,x)
        expected=logsumexp(data['llr'][:,q['ends']]-data['llr'][:,q['starts']]+np.log(q['q']),axis=1)
        got=shape_predictive(data,center,x)
        np.testing.assert_allclose(got['log_bf'],expected,atol=1e-12)


def test_saturated_core_and_no_opportunities_cannot_supply_support():
    data=fixture(np.full((2,30),-3.),np.r_[np.ones((1,30)),np.zeros((1,30))])
    data['llr'][1]=0
    got=shape_predictive(data,[8,18],5)
    assert (got['positive_core_posterior_mass']==0).all()
    assert direction(got,got,np.ones(2))['native_effective_support']==0


def test_extra_observed_hits_distinguish_broad_and_short_shape():
    values=np.full((5,30),-3.);values[:,5:12]=3.
    data=fixture(values)
    short=shape_predictive(data,[5,12],0)
    broad=shape_predictive(data,[5,20],0)
    good=direction(short,short,np.ones(5))
    bad=direction(short,broad,np.ones(5))
    assert good['acceptable_native_mass_fraction']==1.
    assert bad['acceptable_native_mass_fraction']==0.
    assert link_status(good,good)=='reciprocal_shape_compatible'
    assert link_status(good,bad)=='shape_disagreement'


def test_partial_exposure_cannot_renormalize_sliver_into_cross_support():
    data=fixture(np.full((4,30),3.))
    invalid=np.ones((4,30));invalid[:,12:15]=0
    data['invalid']=prefix(invalid)
    got=shape_predictive(data,[10,20],5)
    assert (got['physical_prior_mass']<.5).all()
    assert direction(got,got,np.ones(4))['native_effective_support']==0


def test_every_overlap_and_nested_alternative_remain_in_graph():
    pairs=list(plausible_pairs([[10,20],[11,40]],[[12,21],[20,38],[80,90]],5))
    assert pairs==[(0,0),(0,1),(1,0),(1,1)]


def test_likelihood_tolerance_filters_one_fixed_score_without_changing_geometry():
    data=fixture(np.full((5,30),.4))
    own=shape_predictive(data,[5,12],0)
    alternative=shape_predictive(data,[5,10],0)
    strict=direction(own,alternative,np.ones(5),1.)
    loose=direction(own,alternative,np.ones(5),10.)
    assert strict['acceptable_native_mass_fraction']==0
    assert loose['acceptable_native_mass_fraction']==1


def test_foreign_preference_is_not_reported_as_same_shape():
    data=fixture(np.full((5,30),2.))
    own=shape_predictive(data,[5,9],0)
    alternative=shape_predictive(data,[5,13],0)
    result=direction(own,alternative,np.ones(5),10.)
    assert result['acceptable_native_mass_fraction']==0
    assert result['foreign_shape_preferred_mass_fraction']==1
    keys=['acceptable_native_mass_fraction','foreign_shape_preferred_mass_fraction',
          'native_shape_preferred_mass_fraction','weak_transfer_mass_fraction','untestable_mass_fraction']
    np.testing.assert_allclose(sum(result[k] for k in keys),1.)


def test_competitor_absorption_cannot_preserve_the_targets_own_role():
    from fiberhmm.inference.consensus.cross_joint import replacement_diagnostics
    got=replacement_diagnostics(np.zeros(5),np.full(5,.9),np.full(5,.02),10.)
    assert got['retained_native_mass_fraction']==1.
    assert got['retained_target_inclusion_fraction']<.03
    assert got['jointly_retained_target_mass_fraction']<.03
    # Even unchanged total target mass cannot compensate for relocating it to
    # entirely different molecules within the same recipient cohort.
    shifted=replacement_diagnostics(np.zeros(4),np.array([.9,.9,0,0]),np.array([0,0,.9,.9]),10.)
    assert shifted['native_family_mass']==shifted['replacement_family_mass']
    assert shifted['retained_target_inclusion_fraction']==0


def test_joint_retention_requires_fit_and_target_mass_on_the_same_units():
    from fiberhmm.inference.consensus.cross_joint import replacement_diagnostics
    got=replacement_diagnostics(np.array([0.,-10.]),np.array([1.,1.]),np.array([0.,1.]),10.)
    assert got['retained_native_mass_fraction']==.5
    assert got['retained_target_inclusion_fraction']==.5
    assert got['jointly_retained_target_mass_fraction']==0
