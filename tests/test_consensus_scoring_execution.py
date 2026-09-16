"""Threaded recipient scheduling preserves each conditional random experiment."""
import time
import threading

import numpy as np
import pytest

from fiberhmm.inference.consensus.scoring_execution import (
    native_scoring_pool, score_native_recipients)


def test_nogil_rng_exact_for_repeated_and_distinct_seeds():
    from fiberhmm.inference.consensus.measurement_distribution import _predictive_exceedances
    pa=np.array([.9,.4,.6,.7,.3,.8]); pp=pa*.03
    a,b=np.triu_indices(7,1)
    penalty=-np.arange(len(a))*.15
    cdf=np.cumsum(np.ones(len(a))/len(a)); cdf[-1]=1.
    seeds=[113,113,42,9,99,113,42,8]*8
    def score(seed):
        return _predictive_exceedances(pa,pp,a,b,penalty,cdf,1.1,4095,seed)
    expected=[score(seed) for seed in seeds]
    for workers in (2,4):
        with native_scoring_pool(workers):
            for _ in range(2):
                actual=score_native_recipients(score,seeds,maximum_bytes=1024**3,bytes_per_item=1024)
                assert actual==expected


def test_ordered_recipients_and_unchanged_transferred_records():
    from test_consensus_native_cross import frozen,unit
    from fiberhmm.inference.consensus.native_cross import transferred_call
    p=np.arange(1,60,2); model=frozen(p)
    rng=np.random.default_rng(450)
    units=[unit(rng.random(len(p))<np.where((p>=12)&(p<40),.05,.6)) for _ in range(32)]
    def score(i):
        return transferred_call(model,model['grid'],units[i],
            dict(unit_id=f'u{i}',ordinal=i,start=12,end=40,strand='CT'),replicates=127)
    expected=[score(i) for i in range(32)]
    events=[]
    with native_scoring_pool(4):
        actual=score_native_recipients(score,range(32),maximum_bytes=1024**3,
                                      bytes_per_item=1024,progress=lambda n,t:events.append((n,t)))
    assert actual==expected
    assert events[-1]==(32,32)
    assert all(a[0]<=b[0] for a,b in zip(events,events[1:]))


def test_cancel_drains_threads_and_does_not_affect_next_batch():
    live=set(); lock=threading.Lock()
    def score(i):
        with lock:live.add(i)
        try:
            time.sleep(.025)
            return i
        finally:
            with lock:live.remove(i)
    def cancel(n,t):
        if n:
            raise InterruptedError('cancelled')
    with native_scoring_pool(2):
        with pytest.raises(InterruptedError):
            score_native_recipients(score,range(40),maximum_bytes=1024,bytes_per_item=1,progress=cancel)
        assert not live
        assert score_native_recipients(lambda x:x,range(32),maximum_bytes=1024,bytes_per_item=1)==list(range(32))


def test_scratch_limit_uses_serial_without_truncating():
    main=threading.get_ident()
    with native_scoring_pool(4) as pool:
        actual=score_native_recipients(lambda x:(x,threading.get_ident()),range(32),
                                      maximum_bytes=1,bytes_per_item=1)
        assert pool.executor is None
    assert actual==[(i,main) for i in range(32)]


def test_deferred_transfer_pipeline_prepares_on_parent_and_exports_no_work_carrier():
    import json
    from test_consensus_native_cross import frozen,unit
    from fiberhmm.inference.consensus.native_cross import transferred_call
    from fiberhmm.inference.consensus.measurement_distribution import complete_predictive_reference
    p=np.arange(1,60,2); model=frozen(p); parent=threading.get_ident()
    def prepare(i, defer=True):
        assert threading.get_ident()==parent
        h=((p<12)|(p>=40)).astype(int)
        if i%2: h[12:16]=1
        return transferred_call(model,model['grid'],unit(h),
            dict(unit_id=f'u{i}',ordinal=i,start=12,end=40,strand='CT'),
            replicates=127,_defer_simulation=defer)
    expected=[prepare(i,False) for i in range(40)]
    with native_scoring_pool(4):
        actual=score_native_recipients(prepare,range(40),finish=complete_predictive_reference,
                                      maximum_bytes=1024**3,bytes_per_item=1024)
    assert actual==expected
    assert '_native_predictive_request' not in json.dumps(actual,allow_nan=False)


def test_native_classification_deferred_batches_are_exact():
    import copy,json
    from test_consensus_measurement_family import fixture
    from fiberhmm.inference.consensus.measurement_family import classify_family_profiles
    s,catalog=fixture(); originals=s['units'];s['units']=[]
    for i in range(12):
        for original in originals:
            value=copy.deepcopy(original);value['unit_id']+=f'copy{i}'
            s['units'].append(value)
    kw=dict(region=(0,81),family_model='latent_distribution',
            scoring_folds=2,predictive_replicates=127,max_fit_iterations=25)
    expected=classify_family_profiles(s,catalog,**kw)
    with native_scoring_pool(4):
        actual=classify_family_profiles(s,catalog,**kw)
    assert actual==expected
    assert '_native_predictive_request' not in json.dumps(actual,allow_nan=False)


def test_production_pipeline_cancellation_drains_simulations():
    live=set();lock=threading.Lock()
    def prepare(i):return {'_native_predictive_request':i}
    def finish(record):
        i=record['_native_predictive_request']
        with lock:live.add(i)
        try:
            time.sleep(.02)
            return {'i':i}
        finally:
            with lock:live.remove(i)
    def cancel(n,total):
        if n:raise InterruptedError('cancel pipeline')
    with native_scoring_pool(4):
        with pytest.raises(InterruptedError,match='cancel pipeline'):
            score_native_recipients(prepare,range(32),finish=finish,
                                   maximum_bytes=1024,bytes_per_item=1,progress=cancel)
        assert not live
        actual=score_native_recipients(prepare,range(32),finish=finish,maximum_bytes=1024,bytes_per_item=1)
    assert actual==[{'i':i} for i in range(32)]
