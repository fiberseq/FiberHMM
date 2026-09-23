import numpy as np
from fiberhmm.inference.tf_recaller import call_tfs_in_interval

def test_run_collapse_preserves_extent_and_requires_run_count():
    hit=np.full(4096,-5.);miss=np.full(4096,4.);obs=np.array([4097,4097,4097])
    assert call_tfs_in_interval(obs,0,3,hit,miss,5,3)
    assert not call_tfs_in_interval(obs,0,3,hit,miss,5,3,adjacent_run_mode='collapse')
    c=call_tfs_in_interval(obs,0,3,hit,miss,3,1,adjacent_run_mode='collapse')
    assert len(c)==1 and c[0].start==0 and c[0].length==3 and c[0].n_opps==1 and c[0].llr==4

def test_separated_runs_and_singletons():
    hit=np.full(4096,-5.);miss=np.full(4096,4.);obs=np.array([4097,4097,4096,4097,4096,4097])
    c=call_tfs_in_interval(obs,0,6,hit,miss,5,3,adjacent_run_mode='collapse')
    assert len(c)==1 and c[0].n_opps==3 and c[0].length==6 and c[0].llr==12
    obs=np.array([4097,4096,4097,4096,4097])
    for mode in ['none','average','collapse']:
        c=call_tfs_in_interval(obs,0,5,hit,miss,5,3,adjacent_run_mode=mode)
        assert c[0].llr==12 and c[0].n_opps==3

def test_mixed_run_cannot_select_only_its_miss():
    hit=np.full(4096,-5.);miss=np.full(4096,4.);obs=np.array([0,4097])
    assert call_tfs_in_interval(obs,0,2,hit,miss,1,1)
    assert not call_tfs_in_interval(obs,0,2,hit,miss,1,1,adjacent_run_mode='collapse')
