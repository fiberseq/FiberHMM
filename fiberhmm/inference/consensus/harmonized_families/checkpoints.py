"""Exact, dependency-keyed checkpoints. Incomplete writes are never cache hits."""
from pathlib import Path
import hashlib
import shutil
import sys

import json
from ..artifacts import canonical_bytes, digest, read_json, write_json


def numerical_signature():
    import numpy, scipy, numba
    from threadpoolctl import threadpool_info
    return dict(python=sys.version, numpy=numpy.__version__, scipy=scipy.__version__,
        numba=numba.__version__, blas=[{k:p.get(k) for k in
        ('internal_api','version','architecture')} for p in threadpool_info()
        if p.get('user_api')=='blas'])


class Checkpoints:
    def __init__(self, root, implementation):
        self.root=Path(root)
        self.numerical=numerical_signature()
        self.version=digest(dict(schema='staged-checkpoints-v2-fixed-worker-blas', implementation=implementation,
                                 numerical=self.numerical))
        self.hits={}; self.misses={}

    def key(self, stage, dependencies):
        return digest(dict(version=self.version, stage=stage, dependencies=dependencies))

    def folder(self, stage, key):
        return self.root/stage/key

    def get(self, stage, key):
        path=self.folder(stage,key)/'checkpoint.json.gz'
        if not path.exists():
            self.misses[stage]=self.misses.get(stage,0)+1
            return None
        record=read_json(path)
        if record['key']!=key or record['digest']!=digest(record['value']):
            raise ValueError('Checkpoint integrity check failed: '+str(path))
        self.hits[stage]=self.hits.get(stage,0)+1
        return record['value']

    def put(self, stage, key, value, encoded=None):
        """Store ``value``; ``encoded`` may supply ``canonical_bytes(value)``.

        The value is serialized once: its canonical bytes give the digest and
        are embedded in the record, which is the same JSON object as before
        (keys digest, key, value) and passes the same integrity check."""
        folder=self.folder(stage,key); folder.mkdir(parents=True,exist_ok=True)
        encoded=canonical_bytes(value) if encoded is None else encoded
        record=(b'{"digest":'+json.dumps(hashlib.sha256(encoded).hexdigest()).encode()
                +b',"key":'+json.dumps(key).encode()+b',"value":'+encoded+b'}')
        write_json(folder/'checkpoint.json.gz',None,encoded=record)

    def statistics(self):
        return dict(directory=str(self.root.resolve()), hits=dict(self.hits), misses=dict(self.misses),
                    exact_reuse=True, dependency_version=self.version,numerical_signature=self.numerical)


SOURCE_FILES=('input.json.gz','catalog.json','native.json.gz','source.json.gz')


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as handle:
        for chunk in iter(lambda:handle.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def publish_source(cache, key, folder, timing):
    target=cache.folder('native',key); target.mkdir(parents=True,exist_ok=True)
    for name in SOURCE_FILES:shutil.copyfile(folder/name,target/name)
    cache.put('native',key,dict(timing=timing,files={name:file_hash(target/name) for name in SOURCE_FILES}))


def restore_source(cache, key, folder):
    record=cache.get('native',key)
    if record is None:return None
    source=cache.folder('native',key)
    for name,sha in record['files'].items():
        if file_hash(source/name)!=sha:raise ValueError('Native checkpoint artifact changed: '+name)
    folder.mkdir(parents=True,exist_ok=True)
    for name in SOURCE_FILES:shutil.copyfile(source/name,folder/name)
    case=read_json(folder/'source.json.gz')
    case['native_cell_provenance']['path']=str((folder/'native.json.gz').resolve())
    write_json(folder/'source.json.gz',case)
    return case,record['timing']
