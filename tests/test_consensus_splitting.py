import numpy as np
from fiberhmm.inference.consensus.splitting import geometry,evaluate


def test_intact_and_split_share_outer_span_and_modification_gap_decides():
    pos=np.arange(100,160);span=[100,160]
    family=dict(family='D001',consensus_start=100,consensus_end=120,source_units=100)
    g=geometry(pos,span,[family],3,15)
    assert (g['coordinates'][:,0]>100).all() and (g['coordinates'][:,1]<160).all()
    np.testing.assert_allclose(g['q'].sum(),1)
    score=np.full(len(pos),-2.);score[22:30]=3.
    pref=np.r_[0.,np.cumsum(score)];gains=pref[g['ends']]-pref[g['starts']]
    result=evaluate(g,gains,.1)
    assert result['log_bf_any_split']>5 and result['posterior_any_gap']>.9
    intact=evaluate(g,-2*(g['ends']-g['starts']),.1)
    assert intact['log_bf_any_split']<0 and len(intact['selected'])==0


def test_no_information_cannot_create_a_split_likelihood_bonus():
    pos=np.arange(100,160);f=dict(consensus_start=100,consensus_end=120)
    g=geometry(pos,[100,160],[f],5,20)
    result=evaluate(g,np.zeros(len(g['q'])),1.)
    np.testing.assert_allclose(result['log_bf_any_split'],0.,atol=1e-12)
