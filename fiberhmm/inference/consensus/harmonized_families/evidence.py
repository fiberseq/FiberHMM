"""Lossless shared storage for repeated, immutable stage-evidence subtrees.

References change storage only. Call-level evidence is expanded on demand, with
explicit depth/work limits and content-hash checks on every referenced value.
"""
from copy import deepcopy
import hashlib
import json

REF='__fiberhmm_family_evidence_ref__'


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def intern(value,pool):
    if isinstance(value,dict):
        if REF in value:raise ValueError('Reserved evidence reference field in source data')
        result={k:intern(v,pool) for k,v in value.items()}
    elif isinstance(value,list):result=[intern(v,pool) for v in value]
    else:return value
    raw=encoded(result)
    if len(raw)<512:return result
    key=hashlib.sha256(raw).hexdigest()
    pool.setdefault(key,result)
    return {REF:key}


def expand(value,pool,*,maximum_nodes=1000000,maximum_depth=64):
    remaining=maximum_nodes
    def visit(node,path,depth):
        nonlocal remaining
        remaining-=1
        if remaining<0 or depth>maximum_depth:raise ValueError('Selected-call evidence expansion budget exceeded')
        if isinstance(node,dict):
            if REF in node:
                if set(node)!={REF} or not isinstance(node[REF],str):raise ValueError('Malformed evidence reference')
                key=node[REF]
                if key in path:raise ValueError('Cyclic evidence reference')
                if key not in pool:raise ValueError('Missing shared evidence')
                target=pool[key]
                if hashlib.sha256(encoded(target)).hexdigest()!=key:raise ValueError('Shared evidence content hash mismatch')
                return visit(target,path|{key},depth+1)
            return {k:visit(v,path,depth+1) for k,v in node.items()}
        if isinstance(node,list):return [visit(v,path,depth+1) for v in node]
        return deepcopy(node)
    return visit(value,set(),0)
