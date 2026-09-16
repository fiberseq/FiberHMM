"""Exact speedups and live progress must not change model evidence."""
import numpy as np
import pytest
from fiberhmm.inference.consensus.measurement_distribution import _unique_projection_pairs
from fiberhmm.inference.consensus.nomination import family_availability
from fiberhmm.inference.consensus.progress import report, report_work, stage_progress


@pytest.mark.parametrize('scale',[1,50,2**32,2**62])
def test_packed_pairs_match_structured_sort_and_inverse(scale):
    rng=np.random.default_rng(21)
    a=rng.integers(0,scale,size=500);b=rng.integers(0,scale,size=500)
    # Repeated and invisible geometries must keep exactly the same CDF order.
    a[::3]=a[0];b[::3]=b[0]
    expected=np.unique(np.c_[a,b],axis=0,return_inverse=True)
    actual=_unique_projection_pairs(a,b)
    for x,y in zip(actual,expected):np.testing.assert_array_equal(x,y)
    weights=rng.random(len(a))
    np.testing.assert_array_equal(np.bincount(actual[1],weights=weights),np.bincount(expected[1],weights=weights))


def test_empty_projection_pairs():
    actual=_unique_projection_pairs([],[])
    assert actual[0].shape==(0,2) and actual[1].shape==(0,)


@pytest.mark.parametrize('extra',[0,10,3.5])
def test_compiled_availability_keeps_strict_edges_and_all_calls(extra):
    rng=np.random.default_rng(84)
    units=[dict(representative_raw_tf_intervals=[[int(a),int(a+b)] for a,b in zip(
        rng.integers(-20,200,20),rng.integers(1,80,20))]) for _ in range(100)]
    units.append(dict(representative_raw_tf_intervals=[]))
    centers=np.array([[0,10],[10,20],[50,130],[-30,-10]])
    expected=np.zeros((len(units),len(centers)),bool)
    for m,u in enumerate(units):
        for a,b in u['representative_raw_tf_intervals']:
            expected[m]|=(centers[:,0]-extra<b)&(centers[:,1]+extra>a)
    np.testing.assert_array_equal(family_availability(units,centers,extra),expected)
    assert family_availability([],centers,extra).shape==(0,len(centers))
    assert family_availability(units,[],extra).shape==(len(units),0)


def test_progress_adapters_retain_legacy_signatures_and_forward_structured_counts():
    texts=[]
    legacy=lambda stage,message:texts.append((stage,message))
    report(legacy,'sr','hello',completed=0,total=2)
    child=stage_progress(legacy,'cr',prefix='D: ',dataset_id='D')
    report_work(child,'fit',completed=1,total=10)
    child(None)
    assert texts==[('sr','hello'),('cr','D: fit'),('cr',None)]
    events=[]
    legacy.report=lambda stage,message,**work:events.append((stage,message,work))
    report_work(child,'next',completed=2,total=10)
    assert events==[('cr','D: next',dict(dataset_id='D',completed=2,total=10))]


def test_native_classification_reports_folds_recipients_and_finishes_without_evidence_changes():
    from test_consensus_measurement_family import fixture
    from fiberhmm.inference.consensus.measurement_family import classify_family_profiles
    s,catalog=fixture();kw=dict(region=(0,81),family_model='latent_distribution',
        scoring_folds=2,predictive_replicates=31,max_fit_iterations=25)
    reference=classify_family_profiles(s,catalog,**kw)
    events=[]
    cb=lambda msg:None
    cb.report=lambda msg,**work:events.append((msg,work))
    actual=classify_family_profiles(s,catalog,progress=cb,**kw)
    assert reference==actual
    assert any('held-out model' in msg for msg,_ in events)
    assert any('recipients' in msg for msg,_ in events)
    assert events[-1][1]['completed']==events[-1][1]['total']==len(catalog)
    assert [v['completed'] for _,v in events]==sorted(v['completed'] for _,v in events)


@pytest.mark.parametrize('extreme',[False,True])
def test_prior_mask_reuse_is_bit_exact_even_in_underflow(extreme):
    from fiberhmm.inference.consensus import measurement_distribution as md
    rng=np.random.default_rng(443)
    xy=rng.normal(size=(213,2));area=np.log(rng.uniform(.1,5,len(xy)))
    ll=rng.normal(size=(73,len(xy)))*8
    masks=rng.random((7,len(xy)))>.2
    allowed=masks[rng.integers(0,7,len(ll))]
    theta=np.array([.4,-.1,4.8 if extreme else -.4,5.,4.9 if extreme else -.3])
    density,derivative=md._density(theta,xy)
    actual=md._stable_cached_objective(density,derivative,area,ll,md._allowed_classes(allowed))
    expected=md.native_distribution_objective(theta,xy,area,ll,allowed)
    assert actual[0]==expected[0]
    np.testing.assert_array_equal(actual[1],expected[1])


def test_progress_callback_does_not_change_optimizer_or_fit():
    from fiberhmm.inference.consensus.measurement_distribution import fit_native_distribution
    rng=np.random.default_rng(42);xy=rng.normal(size=(70,2))
    ll=rng.normal(size=(35,70));mask=np.ones(ll.shape,bool)
    args=(ll,mask,xy,np.ones(70));kw=dict(reference=[0,0],max_iterations=10)
    old=fit_native_distribution(*args,**kw);events=[]
    new=fit_native_distribution(*args,progress=events.append,**kw)
    assert events
    for key in old:
        if isinstance(old[key],np.ndarray):np.testing.assert_array_equal(old[key],new[key])
        else:assert old[key]==new[key]
