"""Explicit legacy annotation transport; never infer a frame from observed hits."""
from numbers import Integral


def legacy_annotations(read, frame):
    if frame not in ('disabled','seq','molecular'):
        raise ValueError('Unknown legacy annotation frame')
    if read.has_tag('MA'):
        return None  # MA is authoritative, including an explicitly empty layer.
    out={};n=int(read.query_length or 0)
    for target,starts_tag,lengths_tag in [('msp','as','al'),('nuc','ns','nl')]:
        present=[read.has_tag(starts_tag),read.has_tag(lengths_tag)]
        if not any(present):continue
        if frame=='disabled':
            raise ValueError('Legacy Hia5 as/al or ns/nl tags require an explicit annotation frame in Native input (seq for the verified ind 2–4 h BAM). No MSPs were silently discarded.')
        if not all(present):raise ValueError(f'Incomplete legacy {target} tag pair')
        starts,lengths=read.get_tag(starts_tag),read.get_tag(lengths_tag)
        if len(starts)!=len(lengths):raise ValueError(f'Mismatched legacy {target} tag lengths')
        out[target]=[]
        for start,length in zip(starts,lengths):
            if not isinstance(start,Integral) or not isinstance(length,Integral):
                raise ValueError(f'Non-integer legacy {target} annotation')
            a,b=int(start),int(start)+int(length)
            if not 0<=a<b<=n:raise ValueError(f'Legacy {target} annotation outside the query')
            if frame=='molecular' and read.is_reverse:a=n-b
            out[target].append(dict(start=a,length=int(length),read_length=n,quals=[],name=''))
    return out or None
