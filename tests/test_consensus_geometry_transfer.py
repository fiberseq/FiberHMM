import numpy as np
from fiberhmm.inference.consensus.lattice import RegionFamilyLattice
from fiberhmm.inference.consensus.geometry_transfer import source_integer_mixture,recipient_projection_mixture
from fiberhmm.inference.consensus.geometry_transfer import completed_source_cells,projected_geometry_table
from fiberhmm.inference.consensus.cross_evidence import prefix,shape_predictive,direction
from fiberhmm.inference.consensus.cross_joint import prediction,prediction_with_geometry_groups


def data(values):
    n,k=values.shape
    return dict(start=0,end=k,positions=np.arange(k),llr=prefix(values),observed=prefix(np.ones_like(values)),
        hits=prefix(values<0),invalid=prefix(np.zeros_like(values)))


def test_explicit_uniform_shape_weights_reproduce_original_on_visible_grid():
    v=np.random.default_rng(972).normal(size=(5,35));d=data(v);k=RegionFamilyLattice(np.arange(35),[[12,22]],5)
    a,b,w,_=source_integer_mixture(k,0,k.families[0]['q'],0.)
    np.testing.assert_allclose(w,1/len(w),atol=1e-12)
    base=shape_predictive(d,[12,22],5);weighted=shape_predictive(d,[12,22],5,integer_weights=w)
    for key in base:np.testing.assert_allclose(base[key],weighted[key],atol=1e-12)


def test_default_uniform_prior_does_not_donate_recipient_invisible_mass():
    from fiberhmm.inference.consensus.geometry_transfer import integer_aliases
    positions=np.array([0,20,40]);values=np.full((4,3),3.)
    d=dict(start=0,end=45,positions=positions,llr=prefix(values),
        observed=prefix(np.ones_like(values)),hits=prefix(np.zeros_like(values)),
        invalid=prefix(np.zeros((4,45))))
    a,b=integer_aliases([13,16],5)
    visible=np.searchsorted(positions,a)<np.searchsorted(positions,b)
    assert 0<visible.mean()<.5
    score=shape_predictive(d,[13,16],5,minimum_opportunities=1)
    weighted=shape_predictive(d,[13,16],5,minimum_opportunities=1,integer_weights=np.ones(len(a)))
    for key in score:np.testing.assert_allclose(score[key],weighted[key],atol=1e-12)
    np.testing.assert_allclose(score['recipient_visible_prior_mass'],visible.mean())
    np.testing.assert_allclose(score['physical_prior_mass'],visible.mean())
    np.testing.assert_allclose(score['minimum_opportunity_prior_mass'],visible.mean())
    np.testing.assert_allclose(score['log_bf'],3.)  # Conditional likelihood is unchanged.
    assert direction(score,score,np.ones(4))['native_effective_support']==0


def test_fully_invisible_uniform_geometry_reports_zero_visible_mass():
    d=dict(start=0,end=45,positions=np.array([0,20,40]),llr=prefix(np.ones((1,3))),
        observed=prefix(np.ones((1,3))),hits=prefix(np.zeros((1,3))),invalid=prefix(np.zeros((1,45))))
    score=shape_predictive(d,[10,12],1)
    assert score['recipient_visible_prior_mass'][0]==0
    assert score['physical_prior_mass'][0]==0
    assert np.isnan(score['log_bf'][0])


def test_source_projection_aliases_remain_uniform_and_invisible_aliases_keep_floor():
    k=RegionFamilyLattice(np.array([0,10,20]),[[7,13]],5)
    q=k.families[0]['q'].copy();q[:]=0;q[0]=1
    a,b,w,meta=source_integer_mixture(k,0,q,.05)
    projection=np.c_[np.searchsorted(k.positions,a),np.searchsorted(k.positions,b)]
    for proj in np.unique(projection,axis=0):
        ids=(projection==proj).all(1)
        np.testing.assert_allclose(w[ids],w[ids][0],atol=1e-12)
    invisible=projection[:,0]==projection[:,1]
    assert invisible.any() and (w[invisible]>0).all()
    np.testing.assert_allclose(w[invisible],.05/len(w))
    recipient=RegionFamilyLattice(np.arange(25),[[7,13]],5)
    projected,retained=recipient_projection_mixture(recipient,0,a,b,w)
    np.testing.assert_allclose(retained,1)
    np.testing.assert_allclose(projected.sum(),1)


def test_weighted_favorable_sliver_never_supplies_full_shape_support():
    d=data(np.full((3,35),3.));k=RegionFamilyLattice(np.arange(35),[[12,22]],5)
    a,b,w,_=source_integer_mixture(k,0,k.families[0]['q'],0.)
    bad=np.ones((3,35));bad[:,14:20]=0;d['invalid']=prefix(bad)
    physical=(a>=14)&(b<=20);assert physical.any() and (~physical).any()
    w=np.where(physical,.05/physical.sum(),.95/(~physical).sum())
    score=shape_predictive(d,[12,22],5,integer_weights=w)
    np.testing.assert_allclose(score['physical_prior_mass'],.05)
    assert direction(score,score,np.ones(3))['native_effective_support']==0


def test_fold_specific_full_prediction_has_correct_prior_and_uniform_parity():
    k=RegionFamilyLattice(np.arange(15),[[2,6],[7,12]],2)
    rng=np.random.default_rng(419);v=rng.normal(size=(6,15));eta=np.array([.2,-.7]);allowed=rng.random((6,2))>.2
    groups=np.array([0,1,1,0,0,1]);adj=np.zeros((2,len(k.ga)))
    base,inc=prediction(k,v,eta,allowed)
    new,other=prediction_with_geometry_groups(k,v,eta,allowed,adj,groups)
    np.testing.assert_allclose(new,base,atol=1e-12);np.testing.assert_allclose(other,inc,atol=1e-12)
    adj=rng.normal(size=adj.shape)
    new,_=prediction_with_geometry_groups(k,v,eta,allowed,adj,groups)
    post=k.evaluate(v,eta,allowed=allowed,geometry_log_adjustment=adj[groups])
    prior=k.evaluate(np.zeros_like(v),eta,allowed=allowed,geometry_log_adjustment=adj[groups])
    np.testing.assert_allclose(new,post['log_partition']-prior['log_partition'],atol=1e-12)


def test_completing_source_cell_removes_no_data_distinguishable_boundary():
    pos=np.array([0,10,20,35,50]);k=RegionFamilyLattice(pos,[[11,30]],2)
    aa,bb,ww,meta=completed_source_cells(k,0,k.families[0]['q'],[0,55],0.)
    assert (bb==21).any() and (bb==35).any()
    assert meta['mass_outside_nominal_box']>0
    table=projected_geometry_table(pos,[11,30],aa,bb,ww)
    np.testing.assert_array_equal(table['starts'],k.ga);np.testing.assert_array_equal(table['ends'],k.gb)
    np.testing.assert_allclose(table['q'],np.exp(k.logq),atol=1e-12)
    override=RegionFamilyLattice(pos,[[11,30]],2,family_geometry_overrides={0:table})
    v=np.array([[1,-2,3,-4,5.]]);eta=np.array([.7])
    np.testing.assert_allclose(override.evaluate(v,eta)['log_partition'],k.evaluate(v,eta)['log_partition'],atol=1e-12)


def test_explicit_intervals_and_counterfactual_share_same_weighted_likelihood():
    from scipy.special import logsumexp
    v=np.arange(50)[None,:]/50-.4;d=data(v)
    source=RegionFamilyLattice(np.array([0,10,20,35,49]),[[11,30]],2)
    aa,bb,ww,_=completed_source_cells(source,0,source.families[0]['q'],[0,50],.05)
    score=shape_predictive(d,[11,30],2,integer_weights=ww,integer_intervals=np.c_[aa,bb])
    expected=logsumexp((prefix(v)[:,bb]-prefix(v)[:,aa])+np.log(ww),axis=1)
    np.testing.assert_allclose(score['log_bf'],expected,atol=1e-12)
    table=projected_geometry_table(d['positions'],[11,30],aa,bb,ww)
    k=RegionFamilyLattice(d['positions'],[[11,30]],2,family_geometry_overrides={0:table})
    np.testing.assert_allclose(k.evaluate(v,np.array([0.]))['log_partition'],np.logaddexp(0,expected),atol=1e-12)
