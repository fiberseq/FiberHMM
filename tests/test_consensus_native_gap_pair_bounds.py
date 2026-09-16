import itertools
import numpy as np
import pytest
from scipy.special import logsumexp

from fiberhmm.inference.consensus.native_gap_pair_bounds import (
    SeparatedGapUniverse, certified_log_partition, predictive_log_ratio_bounds)
from fiberhmm.inference.consensus.native_internal_refinement import internal_gap_scores


def test_positive_prefix_pair_sums_and_incident_sums_match_full_enumeration():
    gaps=np.array(list(itertools.combinations(range(8),2)))
    u=SeparatedGapUniverse(gaps); rng=np.random.default_rng(17)
    for _ in range(10):
        active=rng.random(len(gaps))>.3
        left=rng.normal(0,100,len(gaps)); right=rng.normal(0,100,len(gaps))
        triples=[(i,j,left[i]+right[j]) for i in range(len(gaps)) for j in range(len(gaps))
                 if active[i] and active[j] and gaps[i,1]+1<=gaps[j,0]]
        assert u.count_pairs(active)==len(triples)
        assert u.pair_log_sum(left,right,active)==pytest.approx(logsumexp([t[2] for t in triples]),abs=1e-12)
        actual=u.incident_log_sums(left,right,active)
        for i in range(len(gaps)):
            assert actual[i]==pytest.approx(logsumexp([v for a,b,v in triples if i in (a,b)]),abs=1e-12)


def test_all_pairs_touching_successive_anchors_are_counted_once_and_exhaust_universe():
    gaps=np.array(list(itertools.combinations(range(12),2))); u=SeparatedGapUniverse(gaps)
    active=np.ones(len(gaps),bool); total=u.count_pairs(active); seen=set()
    for anchor in reversed(range(len(gaps))):
        partners=u.compatible_indices(anchor,active)
        for j in partners:
            pair=tuple(sorted([int(anchor),int(j)]));assert pair not in seen;seen.add(pair)
        active[anchor]=False
        assert len(seen)+u.count_pairs(active)==total


def test_full_train_predictive_bracket_encloses_exact_small_two_gap_model():
    gaps=np.array(list(itertools.combinations(range(1,8),2))); u=SeparatedGapUniverse(gaps)
    outer=np.array([[0,9],[0,6],[3,9],[2,7]]); q=np.log([.4,.2,.3,.1])
    observations=[dict(positions=np.array([1,3,5,7]),hits=np.array(y),
        p_accessible=np.full(4,.8),p_protected=np.full(4,.04)) for y in ([1,0,1,0],[0,0,1,0],[1,1,0,1])]
    singles=[internal_gap_scores(o,outer,q,gaps,moment_orders=[2.]) for o in observations]
    s=np.array([r['log_refined_vs_continuous'] for r in singles])
    moment=np.array([r['gap_log_moments_vs_current_pattern'][0]*.5 for r in singles])
    upper=np.array([r['gap_maximum_log_gain'] for r in singles])
    all_pairs=[];scores=[]
    for i,(a,b) in enumerate(gaps):
        for j,(c,d) in enumerate(gaps):
            if b+1>c:continue
            all_pairs.append((i,j));scores.append([internal_gap_scores(o,outer,q,[gaps[j]],
                existing_gaps=[gaps[i]])['log_refined_vs_continuous'][0] for o in observations])
    scores=np.array(scores); all_pairs=np.array(all_pairs)
    active=np.ones(len(gaps),bool);active[3]=False
    evaluated=np.any(all_pairs==3,axis=1)
    bounds=[];exact=[]
    for rows in (np.arange(3),np.array([0,1])):
        m=moment[rows].sum(0); sf=s[rows].sum(0); r=upper[rows].sum(0)
        tail=min(u.pair_log_sum(m,m,active),u.pair_log_sum(sf,r,active),u.pair_log_sum(r,sf,active))
        total=scores[:,rows].sum(1)
        b=certified_log_partition(logsumexp(total[evaluated]),tail)
        z=logsumexp(total); assert b['log_lower']<=z+1e-12<=b['log_upper']+1e-12
        bounds.append(b);exact.append(z)
    pred=predictive_log_ratio_bounds(*bounds)
    assert pred[0]<=exact[0]-exact[1]<=pred[1]


def test_log_scale_extremes_empty_remaining_set_and_neutral_prediction():
    u=SeparatedGapUniverse([[1,2],[3,4],[5,6]])
    active=np.array([True,False,True]);w=np.array([-900.,4.,1000.])
    assert u.pair_log_sum(w,w,active)==pytest.approx(100.)
    bound=certified_log_partition(1000.,-np.inf)
    assert bound==dict(log_lower=1000.,log_upper=1000.,omitted_mass_upper=0.)
    assert predictive_log_ratio_bounds(bound,bound)==[0.,0.]


def test_duplicate_gap_alias_and_invalid_weights_are_rejected():
    with pytest.raises(ValueError,match='Unique'):SeparatedGapUniverse([[1,3],[1,3]])
    u=SeparatedGapUniverse([[1,2],[3,4]])
    with pytest.raises(ValueError,match='weight'):u.pair_log_sum([0,np.inf],[0,0],np.ones(2,bool))
