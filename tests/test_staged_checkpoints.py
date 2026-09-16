import time

import pytest

from fiberhmm.inference.consensus.artifacts import read_json,write_json
from fiberhmm.inference.consensus.harmonized_families.checkpoints import Checkpoints
from fiberhmm.inference.consensus.harmonized_families.parallel import ordered_tasks


def test_cache_dependency_and_integrity_checks(tmp_path):
    cache=Checkpoints(tmp_path,{'kernel':'version1'})
    key=cache.key('parent',{'native':'source1','radius':5})
    assert cache.get('parent',key) is None
    cache.put('parent',key,{'scores':[1,2,3]})
    assert cache.get('parent',key)=={'scores':[1,2,3]}
    assert cache.key('parent',{'native':'source1','radius':10})!=key
    assert cache.key('parent',{'native':'source2','radius':5})!=key
    assert Checkpoints(tmp_path,{'kernel':'version2'}).key('parent',{'native':'source1','radius':5})!=key
    path=cache.folder('parent',key)/'checkpoint.json.gz'
    record=read_json(path);record['value']['scores'][0]=999;write_json(path,record)
    with pytest.raises(ValueError,match='integrity'):cache.get('parent',key)


def test_cancel_keeps_completed_tasks_and_kills_pending_workers(tmp_path):
    class Cancelled(Exception):pass
    cache=Checkpoints(tmp_path,{'test':'v1'})
    completed=[]
    def save(key,value):
        cache.put('test',key,{'value':value});completed.append(key)
    def progress(stage,message):
        if completed:raise Cancelled()
    started=time.monotonic()
    with pytest.raises(Cancelled):
        ordered_tasks(time.sleep,[('fast',(.001,),1),('slow',(30,),1)],cores=2,
            maximum_bytes=2,progress=progress,stage='parent_fit',on_result=save)
    assert time.monotonic()-started<15
    assert cache.get('test','fast')=={'value':None}
    assert cache.get('test','slow') is None


def test_fixed_worker_blas_checkpoints_ignore_parent_thread_count(tmp_path,monkeypatch):
    import threadpoolctl
    def pools(n):
        return [{'user_api':'blas','internal_api':'openblas','version':'test','architecture':'test','num_threads':n}]
    monkeypatch.setattr(threadpoolctl,'threadpool_info',lambda:pools(1))
    first=Checkpoints(tmp_path,{'kernel':'v1'})
    monkeypatch.setattr(threadpoolctl,'threadpool_info',lambda:pools(8))
    second=Checkpoints(tmp_path,{'kernel':'v1'})
    assert first.key('native',{'source':1})==second.key('native',{'source':1})
