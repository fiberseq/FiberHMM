"""Lossless shared storage for repeated, immutable stage-evidence subtrees.

References change storage only. Call-level evidence is expanded on demand, with
explicit depth/work limits and content-hash checks on every referenced value.
"""
from copy import deepcopy
import hashlib
import json
from json.encoder import encode_basestring_ascii

REF='__fiberhmm_family_evidence_ref__'


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def intern(value,pool,memo=None):
    """Intern ``value`` into ``pool``. ``memo`` (optional, owned by the caller
    together with ``pool``) remembers containers already interned by identity,
    so a subtree shared by several records or stage snapshots is serialized
    once. Only valid while those objects are not mutated."""
    return _intern(value,pool,memo)[0]


_INFINITY=float('inf')


def _scalar_text(value):
    """json.dumps(value) for a JSON scalar, without building an encoder per call."""
    if type(value) is str:return encode_basestring_ascii(value)
    if value is None:return 'null'
    if value is True:return 'true'
    if value is False:return 'false'
    if type(value) is int:return int.__repr__(value)
    if type(value) is float:
        if value!=value or value in (_INFINITY,-_INFINITY):
            raise ValueError('Out of range float values are not JSON compliant: '+repr(value))
        return float.__repr__(value)
    return encoded(value).decode()


def _intern(value,pool,memo=None):
    """(interned value, its canonical JSON text), built bottom-up.

    Each container's text is assembled from its children's, so every node is
    serialized once instead of once per ancestor. The text equals ``encoded``
    of the interned value (ASCII, sorted keys, compact separators); a key the
    fast path cannot order exactly as json.dumps does falls back to it."""
    if memo is not None and isinstance(value,(dict,list)):
        hit=memo.get(id(value))
        if hit is not None and hit[0] is value:
            return hit[1],hit[2]
        interned,raw=_intern_container(value,pool,memo)
        memo[id(value)]=(value,interned,raw)  # holding value keeps its id unique
        return interned,raw
    return _intern_container(value,pool,memo)


def _intern_container(value,pool,memo):
    if isinstance(value,dict):
        if REF in value:raise ValueError('Reserved evidence reference field in source data')
        parts={k:_intern(v,pool,memo) for k,v in value.items()}
        result={k:v for k,(v,_) in parts.items()}
        if all(type(k) is str for k in parts):
            raw='{'+','.join(encode_basestring_ascii(k)+':'+parts[k][1] for k in sorted(parts))+'}'
        else:raw=encoded(result).decode()
    elif isinstance(value,list):
        parts=[_intern(v,pool,memo) for v in value]
        result=[v for v,_ in parts]
        raw='['+','.join(r for _,r in parts)+']'
    else:return value,_scalar_text(value)
    if len(raw)<512:return result,raw
    key=hashlib.sha256(raw.encode()).hexdigest()
    pool.setdefault(key,result)
    reference={REF:key}
    return reference,'{"'+REF+'":"'+key+'"}'


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
