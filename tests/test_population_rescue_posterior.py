import numpy as np
from scipy.special import expit,logsumexp

from fiberhmm.inference.consensus.lattice import RegionFamilyLattice
from fiberhmm.inference.consensus.population_posterior import PathCache,JointActivityPosterior,integrate_recipient


def test_cached_path_measure_matches_original_entire_configuration_model():
    k=RegionFamilyLattice(np.arange(20),[[2,8],[5,11],[12,17]],ambiguity_bp=1)
    rng=np.random.default_rng(31);v=rng.normal(size=(7,20));ref=np.array([-2.,-.5,1.])
    mask=rng.random((7,len(k.ga)))>.15;adj=np.log(rng.uniform(.1,1,(7,len(k.ga))))
    cached=PathCache.build(k,v,ref,mask,adj)
    for eta in [ref,np.zeros(3),np.array([2.,-4.,.1])]:
        expected=k.evaluate(v,eta,geometry_allowed=mask,geometry_log_adjustment=adj,export_geometry=True)
        got=cached.evaluate(eta,geometry=True)
        for key in ['log_partition','family_inclusion','geometry_mass']:
            np.testing.assert_allclose(got[key],expected[key],atol=2e-12,rtol=2e-12)


def test_joint_gradient_hessian_and_overlap_covariance():
    k=RegionFamilyLattice(np.arange(12),[[2,7],[4,9]],ambiguity_bp=0)
    v=np.array([[0,0,1,1,1,1,1,-2,-2,0,0,0],np.zeros(12)])
    p=PathCache.build(k,np.zeros_like(v),np.zeros(2));d=PathCache.build(k,v,np.zeros(2))
    model=JointActivityPosterior(d,p);eta=np.array([-.3,.7]);step=1e-4
    numerical=[]
    for j in range(2):
        dx=np.eye(2)[j]*step
        numerical.append((model.objective(eta+dx,False)-model.objective(eta-dx,False))/(2*step))
    np.testing.assert_allclose(model.objective(eta)[1],numerical,atol=1e-8)
    # The only configurations are empty, first, second. Exact covariance.
    h=np.eye(2)/4
    for row in v:
        lr=np.array([row[2:7].sum(),row[4:9].sum()])
        a=np.exp(eta-logsumexp(np.r_[0.,eta]));b=np.exp(eta+lr-logsumexp(np.r_[0.,eta+lr]))
        h+=np.diag(a)-np.outer(a,a)-np.diag(b)+np.outer(b,b)
    got=np.column_stack([(model.objective(eta+np.eye(2)[j]*step)[1]-model.objective(eta-np.eye(2)[j]*step)[1])/(2*step) for j in range(2)])
    np.testing.assert_allclose(got,h,atol=1e-8)
    assert abs(h[0,1])>.01
    for cache,expected in [(p,k.evaluate(np.zeros_like(v),eta)),(d,k.evaluate(v,eta))]:
        excluded=cache.evaluate(eta,exclude_family=0)
        direct=k.evaluate(np.zeros_like(v) if cache is p else v,eta,allowed=np.tile([False,True],(2,1)))
        np.testing.assert_allclose(excluded['log_partition'],direct['log_partition'],atol=1e-12)


def test_parameter_marginalization_uses_likelihood_ratio_not_naive_mean():
    k=RegionFamilyLattice(np.arange(3),[[1,2]],ambiguity_bp=0)
    d=PathCache.build(k,np.array([[0.,np.log(9.),0.]]),np.zeros(1))
    p=PathCache.build(k,np.zeros((1,3)),np.zeros(1))
    draws=np.array([[-3.],[2.]])
    prior=.5*(expit(-3)+expit(2))
    expected=9*prior/(1-prior+9*prior)
    got=integrate_recipient(p,d,draws,np.log([.5,.5]))
    np.testing.assert_allclose(got['family_inclusion'][0,0],expected,atol=1e-12)
    naive=.5*(expit(-3+np.log(9))+expit(2+np.log(9)))
    assert abs(expected-naive)>.1


def test_neutral_row_cancels_exactly_and_does_not_shrink_population_uncertainty():
    k=RegionFamilyLattice(np.arange(8),[[1,4],[3,6]],ambiguity_bp=1)
    zero=np.zeros((6,8));p=PathCache.build(k,zero,np.zeros(2))
    model=JointActivityPosterior(p,p,prior_mean=-2.,prior_sd=2.)
    eta=np.array([-.5,-3.]);value,grad=model.objective(eta)
    np.testing.assert_allclose(grad,(eta+2)/4,atol=1e-14)
    h,cov,_=model.hessian(np.full(2,-2.))
    np.testing.assert_allclose(cov,np.eye(2)*4,atol=1e-10)
    got=integrate_recipient(p,p,np.array([[0.,1.],[-1.,2.],[3.,-.5]]),np.log([.2,.5,.3]))
    np.testing.assert_allclose(got['prior_inclusion'],got['family_inclusion'],atol=1e-13)
    np.testing.assert_allclose(got['recipient_log_bf'],0,atol=1e-12)


def test_population_depth_tightens_frequency_without_universal_occupancy():
    k=RegionFamilyLattice(np.arange(3),[[1,2]],ambiguity_bp=0)
    estimates=[];variances=[]
    for repeats in [1,20]:
        values=np.tile(np.array([[0.,8.,0.]]*3+[[0.,-8.,0.]]*7),(repeats,1))
        p=PathCache.build(k,np.zeros_like(values),np.zeros(1));d=PathCache.build(k,values,np.zeros(1))
        m=JointActivityPosterior(d,p);fit,_=m.fit(max_iterations=80)
        _,cov,_=m.hessian(fit['eta']);estimates.append(expit(fit['eta'][0]));variances.append(cov[0,0])
    assert all(.2<x<.4 for x in estimates)
    assert variances[1]<variances[0]/10


def small_recipient(positions, hits, *, existing=False, eta=-.4):
    from fiberhmm.inference.consensus.population_rescue import score_batch
    from fiberhmm.inference.consensus.recall import projection_rectangles
    from fiberhmm.inference.consensus.parameters import RescueOptions
    k=RegionFamilyLattice(np.arange(14),[[4,9]],ambiguity_bp=1)
    unit=dict(unit_id='u',strand='GA',_region=[0,14],positions=positions,hits=hits,
        p_accessible=[.9]*len(positions),p_protected=[.05]*len(positions),
        representative_raw_tf_intervals=[[4,9]] if existing else [],raw_nuc_intervals=[],
        aligned_blocks=[[0,14]],msp_intervals=[[0,14]])
    observed=np.zeros((1,14),bool);observed[0,positions]=True
    h=np.full((1,14),-1);h[0,positions]=hits
    v=np.zeros((1,14));v[0,positions]=np.where(hits,np.log(.05/.9),np.log(.95/.1))
    data=dict(units=[unit],unit_ids=['u'],strands=np.array(['GA']),folds=np.array([0]),
        family_ids=['f'],log_lr=v,observed=observed,hits=h)
    row=dict(unit_id='u',source_calls=unit['representative_raw_tf_intervals'],
        proposals=[dict(family_index=0,interval=[4,9])] if existing else [])
    source=dict(source_nomination_units=[20],training_digest='frozen',
        source_posterior_expected_units_at_MAP=[20.],source_prior_inclusion_at_MAP=[.4],
        activity_standard_deviation=[.1],integration=dict(effective_sample_size=32),numerical_provisional=False)
    q=np.exp(k.logq)
    return score_batch(data,k,np.array([0]),q,projection_rectangles(k),np.full((32,1),eta),
        np.full(32,-np.log(32)),np.array([eta]),source,RescueOptions(),{'u':row})


def test_one_and_two_position_new_calls_survive_and_have_graded_confidence():
    one=small_recipient([1,6,11],np.array([1,0,1]))
    two=small_recipient([1,6,7,11],np.array([1,0,0,1]))
    a=one[0][0]['calls'];b=two[0][0]['calls']
    assert len(a)==len(b)==1
    assert a[0]['opportunities']==1 and b[0]['opportunities']==2
    assert .5<a[0]['model_membership']<b[0]['model_membership']<1
    assert a[0]['information_status']=='weak_population_assisted'
    # Thresholding only removes a call; it never changes intervals/classes.
    sets=[{(c['family'],tuple(c['interval'])) for c in a if c['model_membership']>=t}
          for t in [.5,.75,.9,.95,.99]]
    assert all(b<=a for a,b in zip(sets,sets[1:]))


def test_blind_saturated_or_already_assigned_family_never_gets_new_call():
    assert not small_recipient([1,11],np.array([1,1]))[0][0]['calls']
    assert not small_recipient([1,6,11],np.array([1,1,1]),eta=9.)[0][0]['calls']
    assert not small_recipient([1,6,11],np.array([1,0,1]),existing=True)[0][0]['calls']


def test_activity_training_excludes_entire_recipient_group_fold():
    from fiberhmm.inference.consensus.population_rescue import training_indices
    data=dict(folds=np.array([0,0,1,1]),strands=np.array(['CT','GA','CT','GA']),
              fold_group_ids=['duplex1','duplex1','duplex2','duplex2'])
    np.testing.assert_array_equal(training_indices(data,'GA',0,'opposite_strand'),[2])
    np.testing.assert_array_equal(training_indices(data,'CT',0,'opposite_strand'),[3])
    np.testing.assert_array_equal(training_indices(data,'GA',0,'pooled'),[2,3])


def test_hmc_population_integrator_against_direct_one_dimensional_quadrature():
    from scipy.integrate import quad
    k=RegionFamilyLattice(np.arange(3),[[1,2]],ambiguity_bp=0)
    values=np.array([[0.,2.,0.]]*3+[[0.,-2.,0.]]*7)
    p=PathCache.build(k,np.zeros_like(values),np.zeros(1));d=PathCache.build(k,values,np.zeros(1))
    model=JointActivityPosterior(d,p)
    fit,_=model.fit();_,cov,_=model.hessian(fit['eta'])
    draws,lw,diagnostic=model.corrected_chain_draws(fit['eta'],cov,number=1024,seed=281)
    density=lambda x:np.exp(-model.objective(np.array([x]),False)+fit['objective'])
    normalizer=quad(density,-20,15)[0]
    mean=quad(lambda x:expit(x)*density(x),-20,15)[0]/normalizer
    np.testing.assert_allclose(np.mean(expit(draws[:,0])),mean,atol=.025)
    assert diagnostic['maximum_split_Rhat']<1.1
    assert not sum(c['divergences'] for c in diagnostic['acceptance'])
    np.testing.assert_allclose(np.exp(lw).sum(),1.)


def test_sparse_cache_concatenation_and_extreme_reweighting():
    k=RegionFamilyLattice(np.arange(12),[[1,5],[7,10]],ambiguity_bp=1)
    rng=np.random.default_rng(25);v=rng.normal(size=(6,12));eta=np.array([-2.,-2.])
    whole=PathCache.build(k,v,eta)
    pieces=PathCache.concatenate([PathCache.build(k,v[:2],eta),PathCache.build(k,v[2:],eta)])
    for e in [eta,np.array([22.,-22.]),np.array([-22.,22.])]:
        a=whole.evaluate(e,geometry=True);b=pieces.evaluate(e,geometry=True)
        direct=k.evaluate(v,e,export_geometry=True)
        for key in ['log_partition','family_inclusion','geometry_mass']:
            np.testing.assert_allclose(a[key],b[key],atol=2e-12)
            np.testing.assert_allclose(a[key],direct[key],atol=2e-11)


def test_family_label_action_does_not_credit_a_rare_class_with_its_competitors_protection():
    from fiberhmm.inference.consensus.recall_action import decode_posterior_accuracy,decode_posterior_family_accuracy
    k=RegionFamilyLattice(np.arange(9),[[2,6],[2,6]],ambiguity_bp=0)
    masses=np.where(k.gf==0,.001,.998)[None,:]
    posterior=dict(geometry_mass=masses)
    values=np.ones((1,9));observed=np.ones_like(values,bool);eta=np.array([8.,0.])
    allowed=np.ones((1,2),bool);physical=np.ones_like(masses,bool);adjust=np.zeros_like(masses)
    # Binary protection utility ties across classes and the old MAP tiebreak
    # gives the rare class protection supplied almost entirely by the other one.
    old=decode_posterior_accuracy(k,values,observed,posterior,eta,allowed,physical,adjust)
    new=decode_posterior_family_accuracy(k,values,observed,posterior,eta,allowed,physical,adjust)
    assert old['geometry_by_family'][0,0]>=0 and old['geometry_by_family'][0,1]<0
    assert new['geometry_by_family'][0,0]<0 and new['geometry_by_family'][0,1]>=0
    np.testing.assert_allclose(new['expected_accuracy_improvement'],[4*(.998-.001)],atol=1e-12)


def test_rao_blackwellized_probabilities_are_not_quantized_by_parameter_draw_count():
    k=RegionFamilyLattice(np.arange(3),[[1,2]],ambiguity_bp=0)
    values=np.array([[0.,np.log(1e6),0.]])
    p=PathCache.build(k,np.zeros_like(values),np.zeros(1));d=PathCache.build(k,values,np.zeros(1))
    eta=-.4;draws=np.full((8,1),eta)
    got=integrate_recipient(p,d,draws,np.full(8,-np.log(8)))
    np.testing.assert_allclose(got['family_inclusion'][0,0],expit(eta+np.log(1e6)),atol=1e-12)
    assert .999<got['family_inclusion'][0,0]<1.


def test_sparse_integrated_geometry_matches_dense_ratio_estimator():
    from fiberhmm.inference.consensus.population_posterior import _integrate_recipient_dense
    rng=np.random.default_rng(351)
    k=RegionFamilyLattice(np.arange(24),[[2,8],[5,13],[15,20]],ambiguity_bp=2)
    v=rng.normal(size=(5,24));eta=np.full(3,-2.)
    physical=rng.random((5,len(k.ga)))>.4;adjust=np.log(rng.uniform(.1,1,physical.shape))
    prior=PathCache.build(k,np.zeros_like(v),eta,physical,adjust)
    observed=PathCache.build(k,v,eta,physical,adjust)
    draws=rng.normal(size=(32,3))*2-1;lw=rng.normal(size=32)
    a=integrate_recipient(prior,observed,draws,lw)
    b=_integrate_recipient_dense(prior,observed,draws,lw)
    for key in a:np.testing.assert_allclose(a[key],b[key],atol=2e-12,rtol=2e-12)
    for f in range(k.f):
        np.testing.assert_allclose(a['geometry_mass'][:,k.gf==f].sum(1),a['family_inclusion'][:,f],atol=2e-12)
