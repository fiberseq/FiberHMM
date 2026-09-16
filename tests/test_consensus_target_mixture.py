import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize._numdiff import approx_derivative
from fiberhmm.inference.consensus.lattice import RegionFamilyLattice
from fiberhmm.inference.consensus.target_mixture import target_terms,mixture_prediction,mixture_objective,fit_mixture


def test_zero_information_cancels_with_nonuniform_prior():
    values=np.array([[1.,-3.],[-np.inf,-np.inf],[2.,4.]])
    p,inc,prior=mixture_prediction(values,values,[.9,.1],1.2)
    assert_allclose(p,0,atol=1e-14);assert_allclose(inc,prior)
    x=np.array([1.2,1.,-1.])
    objective,gradient=mixture_objective(x,[dict(data=values,prior=values)],weight_pseudocount=0)
    assert_allclose(objective,0,atol=1e-14);assert_allclose(gradient,0,atol=1e-14)


def test_gradient_includes_configuration_prior_normalizer():
    rng=np.random.default_rng(61)
    data=[dict(data=rng.normal(size=(8,3)),prior=rng.normal(size=(8,3))) for _ in range(2)]
    for shared in (False,True):
        x=rng.normal(size=2+(3 if shared else 6))
        value,gradient=mixture_objective(x,data,shared,1.7)
        actual=approx_derivative(lambda z:mixture_objective(z,data,shared,1.7)[0],x).ravel()
        assert_allclose(gradient,actual,atol=1e-8,rtol=1e-6)


import pytest


@pytest.mark.parametrize('a',[0.,.31,1.])
def test_full_competitor_template_affinity_matches_direct_dp(a):
    k=RegionFamilyLattice(np.arange(12),[[1,6],[4,9],[9,11]],1)
    rng=np.random.default_rng(12);v=rng.normal(size=(5,12));eta=np.array([.2,-.7,.5])
    allowed=np.ones((5,3),bool);allowed[2,0]=False
    gs=np.flatnonzero(k.gf==0);q1=np.arange(1,len(gs)+1,dtype=float);q1/=q1.sum();q2=q1[::-1]
    packed=[]
    for q in (q1,q2):
        adjust=np.zeros((1,len(k.ga)));adjust[0,gs]=np.log(q)-k.logq[gs]
        packed.append(target_terms(k,v,eta,allowed,adjust,np.zeros(5,int),0))
    assert_allclose(packed[0]['exclude_predictive'],packed[1]['exclude_predictive'],atol=1e-10)
    activity=.8
    prediction,inc,prior=mixture_prediction(np.c_[packed[0]['data'],packed[1]['data']],
        np.c_[packed[0]['prior'],packed[1]['prior']],[a,1-a],activity)
    adjust=np.zeros((5,len(k.ga)));adjust[:,gs]=np.log(a*q1+(1-a)*q2)-k.logq[gs]
    et=eta.copy();et[0]=activity
    full=k.evaluate(v,et,allowed=allowed,geometry_log_adjustment=adjust)
    base=k.evaluate(np.zeros_like(v),et,allowed=allowed,geometry_log_adjustment=adjust)
    assert_allclose(prediction+packed[0]['exclude_predictive'],full['log_partition']-base['log_partition'],atol=1e-10)
    assert_allclose(inc,full['family_inclusion'][:,0],atol=1e-12)
    assert_allclose(prior,base['family_inclusion'][:,0],atol=1e-12)


def test_separate_models_contain_shared_model_and_fit_different_weights():
    # Two distinguishable source templates, in opposite populations. No caller
    # or biological assumption: these are exact cached include/exclude ratios.
    a=np.tile([6.,-6.],(60,1));b=np.tile([-6.,6.],(60,1));base=np.zeros_like(a)
    data=[dict(data=a,prior=base),dict(data=b,prior=base)]
    common,_=fit_mixture(data,True,0);separate,_=fit_mixture(data,False,0)
    assert common['success'] and separate['success']
    assert separate['objective']<common['objective']-.5
    assert separate['weights'][0][0]>.99 and separate['weights'][1][1]>.99


def test_templates_identical_do_not_create_shape_evidence():
    rng=np.random.default_rng(25);v=rng.normal(size=(30,1));p=rng.normal(size=(30,1))
    data=[dict(data=np.repeat(v,2,axis=1),prior=np.repeat(p,2,axis=1))]
    fit,_=fit_mixture(data,True,1.)
    assert_allclose(fit['weights'][0],[.5,.5],atol=2e-4)
