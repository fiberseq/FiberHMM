"""Explicit, nonexclusive counting events on an immutable atomic XCR graph.

Maximal complete bipartite subgraphs retain alternatives without collapsing a
transitive chain into one site. Every event is named as ANY or ALL of its native
members and counts each evidence group once. These descriptive endpoints are
not equal-detectability or occupancy claims; compound fragmentation needs a
separate native observation test before ANY and ALL can be equated.
"""
from __future__ import annotations
from collections import defaultdict
from bisect import bisect_left
import math

from .artifacts import digest
from .native_cross import _reference_members


def maximal_rectangles(pairs, *, maximum_concepts=100000):
    """All maximal bicliques, not a one-to-one matching or transitive union.

    Enumerate distinct intersections of right-neighbor sets, then take their
    closure. The safety budget raises explicitly; it never drops later groups.
    """
    neighbors=defaultdict(set)
    for a,b in pairs:neighbors[a].add(b)
    concepts=set()
    for a in sorted(neighbors):
        right=frozenset(neighbors[a])
        concepts |= {right} | {c&right for c in concepts if c&right}
        if len(concepts)>maximum_concepts:
            raise MemoryError('Exact XCR correspondence-group budget exceeded; no groups silently dropped')
    result=[]
    for right in sorted(concepts,key=lambda x:tuple(sorted(x))):
        left=frozenset(a for a in neighbors if right<=neighbors[a])
        closure=set.intersection(*(neighbors[a] for a in left))
        if closure==set(right):result.append((tuple(sorted(left)),tuple(sorted(right))))
    return sorted(result)


def covers(blocks, span):
    """Actual union of aligned blocks; outer alignment bounds are insufficient."""
    position=span[0]
    for a,b in sorted(blocks):
        if a>position:return False
        position=max(position,b)
        if position>=span[1]:return True
    return False


def _wilson(k,n):
    if not n:return None
    z=1.959963984540054;fraction=k/n;den=1+z*z/n
    center=(fraction+z*z/(2*n))/den
    width=z*math.sqrt(fraction*(1-fraction)/n+z*z/(4*n*n))/den
    return [max(0.,center-width),min(1.,center+width)]


def count_native_event(data, families, span, *, reference_percent=99.9, memberships=None):
    """Fixed original calls, complete coverage, and biochemical strata.

    Aligned coverage is the predeclared primary denominator. A separately
    reported opportunity-bearing denominator is NOT called equal detection
    power. No observed hit or apparent agreement determines either denominator.
    Unknown cross-strand duplex identity is not inferred from fingerprints.
    """
    families=tuple(sorted(families));members=memberships if memberships is not None else _reference_members(data['result'],reference_percent)
    calls=data['result']['calls']; units=data['units']; by_group=defaultdict(list)
    selected=defaultdict(set); member_calls=defaultdict(int)
    for fid in families:
        for i in members.get(fid,[]):
            c=calls[i];g=str(c.get('evidence_group_id',c['unit_id']))
            selected[g].add(fid);member_calls[g]+=1
    for u in units:by_group[str(u.get('fold_group_id',u['unit_id']))].append(u)
    pooled_hia5=data['chemistry'].startswith('hia5')
    rows=[]
    for group,observations in by_group.items():
        # For unresolved physical repeats, retain one group and use only its
        # actually observed coverage. No multiplication of repeated evidence.
        covered=[u for u in observations if covers(u.get('aligned_blocks',[]),span)]
        if len(covered)==1:
            p=covered[0]['positions'];opportunities=bisect_left(p,span[1])-bisect_left(p,span[0])
        else:
            opportunities=len({int(p) for u in covered for p in
                u['positions'][bisect_left(u['positions'],span[0]):bisect_left(u['positions'],span[1])]})
        strata={'pooled'} if pooled_hia5 else {u['strand'] for u in observations}
        present=selected.get(group,set())
        rows.append(dict(group=group,strand=next(iter(strata)) if len(strata)==1 else 'unresolved_mixed',
            covered=bool(covered),opportunities=opportunities,present=present,
            any_member=bool(present),all_members=set(families)<=present,
            multiple_members=len(present)>1,source_calls=member_calls.get(group,0)))
    def summarize(subset):
        eligible=[r for r in subset if r['covered']];observed=[r for r in eligible if r['opportunities']>0]
        n=len(eligible);k=sum(r['any_member'] for r in eligible);both=sum(r['all_members'] for r in eligible)
        return dict(eligible_units=n,assigned_units=k,fraction=k/n if n else None,
            descriptive_unit_wilson_interval=_wilson(k,n),all_member_units=both,all_member_fraction=both/n if n else None,
            multiple_member_units=sum(r['multiple_members'] for r in eligible),
            original_member_calls=sum(r['source_calls'] for r in eligible),
            opportunity_bearing_units=len(observed),opportunity_bearing_assigned_units=sum(r['any_member'] for r in observed),
            assigned_without_span_coverage=sum(r['any_member'] for r in subset if not r['covered']),
            member_counts={f:sum(f in r['present'] for r in eligible) for f in families})
    out=summarize(rows)
    out.update(families=list(families),comparison_span=list(span),endpoint='ANY listed native primary class on an evidence group',
        alternate_endpoint='ALL listed native primary classes on the same evidence group',
        denominator='complete aligned-block coverage of the same cross-dataset comparison span',
        by_strand={s:summarize([r for r in rows if r['strand']==s]) for s in sorted({r['strand'] for r in rows})},
        classification_reference_percent=reference_percent,
        repeated_groups_collapsed=len(units)-len(rows),unresolved_mixed_groups=sum(r['strand']=='unresolved_mixed' for r in rows),
        calibrated_occupancy=False,equal_detection_power_established=False,
        interval_semantics='descriptive binomial evidence-unit interval; unestablished duplex independence is not assumed solved')
    return out


def summarize_native_correspondences(graph, datasets, *, maximum_concepts=100000):
    """Keep every maximal reciprocal rectangle and expose event ambiguity."""
    by_pair=defaultdict(list)
    for e in graph['links']:
        if e['comparable']:by_pair[tuple(e['datasets'])].append(e)
    groups=[];reference=graph['parameters']['classification_reference_percent']
    odds=float(graph['parameters'].get('membership_loss_odds',1.))
    memberships={ds:_reference_members(data['result'],reference,odds) for ds,data in datasets.items()}
    for (left,right),edges in sorted(by_pair.items()):
        lookup={(e['families'][left],e['families'][right]):e for e in edges}
        coords={f:tuple(e['native_intervals'][ds]) for e in edges for ds,f in e['families'].items()}
        rectangles=maximal_rectangles(lookup,maximum_concepts=maximum_concepts)
        usage=defaultdict(int)
        for aa,bb in rectangles:
            for f in aa+bb:usage[f]+=1
        for aa,bb in rectangles:
            selected=[lookup[a,b] for a in aa for b in bb]
            span=min(coords[f][0] for f in aa+bb),max(coords[f][1] for f in aa+bb)
            lc=count_native_event(datasets[left],aa,span,reference_percent=reference,memberships=memberships[left])
            rc=count_native_event(datasets[right],bb,span,reference_percent=reference,memberships=memberships[right])
            multi=len(aa)>1 or len(bb)>1
            coexist=lc['multiple_member_units']+rc['multiple_member_units']>0
            groups.append(dict(group_id='NXG_'+digest([left,right,aa,bb])[:16],
                left_dataset=left,right_dataset=right,left_families=list(aa),right_families=list(bb),
                edge_ids=[e['link_id'] for e in selected],native_intervals={f:list(coords[f]) for f in aa+bb},
                merged_interval=None,comparison_span=list(span),span_is_only_counting_denominator=True,
                kind='one_to_one' if not multi else 'cooccurring_fragments_or_states' if coexist else 'alternative_class_union',
                status='descriptive_native_call_events',left_counts=lc,right_counts=rc,
                complete_bipartite=True,native_models_changed=False,transitive_union=False,
                overlaps_other_count_groups=any(usage[f]>1 for f in aa+bb),
                groups_form_a_partition=False,counts_must_not_be_summed_across_groups=True,
                event_equivalence_established=False,fragmentation_resolved=False if multi else None,
                rate_agreement_used_for_selection=False,rescued_calls_used=False,
                caution='ANY and ALL are different observed events. Co-occurrence alone cannot equate several protected pieces with one broad state.'))
    return groups
