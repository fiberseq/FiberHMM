"""Complete, score-independent nomination of contiguous existing-call sets.

An admissible small compound must not prevent scoring a larger interpretation.
No new interval, call, fitted family or correspondence is constructed here.
"""
from __future__ import annotations


def contiguous_compound_candidates(calls, anchor_ordinals):
    """Yield every distinct contiguous block of >=2 calls containing an anchor.

    Input calls must already refer to one evidence unit and the relevant source
    model's fixed domain. Geometry/predictive evidence may subsequently reject
    a block, but neither is a reason to omit its supersets from nomination.
    Enumeration is quadratic in the number of overlapping-domain calls, with
    no per-window family/read cap and no score-dependent early stopping.
    """
    ordered = sorted(calls, key=lambda c: (c['start'], c['end'], c['ordinal']))
    if len({c['unit_id'] for c in ordered}) > 1:
        raise ValueError('A compound cannot combine different evidence units')
    identities = [c['ordinal'] for c in ordered]
    if len(identities) != len(set(identities)):
        raise ValueError('Original call ordinals must be unique on one unit')
    anchors = set(anchor_ordinals)
    for length in range(2, len(ordered)+1):
        for start in range(len(ordered)-length+1):
            block = ordered[start:start+length]
            if any(c['ordinal'] in anchors for c in block):
                yield block
