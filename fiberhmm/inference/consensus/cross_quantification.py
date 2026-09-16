"""Descriptive counting endpoints for immutable XCR relationship graphs.

Atomic links are not unique count matches. Count an ANY-member event only when
the entire shape-compatible component is complete bipartite and each assay's
native intervals share a core. Incomplete/transitive chains remain unresolved;
we never choose an edge using agreement of observed fractions. These are counts
of decoded assignments, not marginal sums, calibrated occupancy, or proof of
equal detection power. Native intervals and inference are not changed.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from .artifacts import digest


def _event_mass(run, ix, native=False):
    """Exact union event under the frozen grouping model, never capped sums."""
    if run.get('kernel') is None or 'data' not in run:return None
    key='native_allowed' if native else 'allowed'
    if key not in run:return None
    cache=run.setdefault('_count_event_cache',{})
    event_key=(key,tuple(sorted(ix)))
    if event_key not in cache:
        kernel=run['kernel'];values=run['data']['log_lr'];allowed=run[key]
        base_key=(key,'partition')
        if base_key not in cache:
            cache[base_key]=kernel.evaluate(values,run['eta'],allowed=allowed)['log_partition']
        cache[event_key]=kernel.any_family_inclusion(values,run['eta'],ix,allowed=allowed,log_partition=cache[base_key])
    return cache[event_key]


def _count(run, families, threshold):
    indices = {c['family']: c['family_index'] for c in run['catalog']}
    ix = [indices[f] for f in families]
    eligible = np.asarray(run['eligible'], dtype=bool)[:, ix].all(axis=1)
    selected = np.asarray(run['proposal_membership'])[:, ix] >= threshold
    decoded = np.asarray(run['proposal_membership'])[:,ix]>0
    mass=_event_mass(run,ix)
    event=selected.any(axis=1) if mass is None else decoded.any(axis=1)&(mass>=threshold)
    n = int(eligible.sum())
    count = int((event & eligible).sum())
    union_eligible=np.asarray(run['eligible'],dtype=bool)[:,ix].any(axis=1)
    un=int(union_eligible.sum());uk=int((union_eligible&event).sum())
    native = run.get('native_proposal_membership')
    native_mass=_event_mass(run,ix,native=True) if native is not None else None
    native_event=(np.asarray(native)[:,ix]>=threshold).any(axis=1) if native is not None else None
    if native_mass is not None:native_event=(np.asarray(native)[:,ix]>0).any(axis=1)&(native_mass>=threshold)
    before=int((native_event&eligible).sum()) if native_event is not None else None
    result=dict(families=families, eligible_units=n, assigned_units=count if n else None,
        fraction=count/n if n else None,
        union_eligible_units=un,union_eligible_assigned_units=uk if un else None,union_eligible_fraction=uk/un if un else None,
        primary_denominator='all listed members have at least one physically eligible variant',
        native_assigned_units=before if n else None,
        native_fraction=before/n if n and before is not None else None,
        native_group_event_mass_computed=native_mass is not None,
        native_endpoint=('same exact ANY-event rule as current counts' if native_mass is not None else
                         'legacy per-member threshold fallback' if native is not None else 'unavailable'),
        native_current_endpoints_match=(native is not None and (mass is None)==(native_mass is None)),
        per_member_threshold_assigned_units=int((selected.any(axis=1)&eligible).sum()) if n else None,
        group_event_mass_computed=mass is not None,
        group_mass_passing_without_selected_action=int(((mass>=threshold)&~decoded.any(axis=1)&eligible).sum()) if mass is not None else None,
        selected_action_below_group_threshold=int((decoded.any(axis=1)&(mass<threshold)&eligible).sum()) if mass is not None else None,
        multiple_member_units=int(((selected.sum(axis=1)>1) & eligible).sum()),
        member_counts=[dict(family=f,assigned_units=int((selected[:,j]&eligible).sum())) for j,f in enumerate(families)],
        denominator='intersection of ANY-variant native physical eligibility across all listed members; not canonical-core testability',
        endpoint=('caller-conditioned physical MAP selects any listed member AND exact group inclusion >= fixed threshold'
                  if mass is not None else 'legacy: any selected native CR member at or above the fixed per-member threshold'),
        probability_semantics='frozen native caller-conditioned grouping model; not calibrated occupancy',
        status='descriptive' if n else 'no_common_eligible_units')
    if 'core_eligible' in run:
        core=np.asarray(run['core_eligible'],bool)[:,ix].all(axis=1)&eligible
        cn=int(core.sum());ck=int((core&event).sum())
        result.update(canonical_core_eligible_units=cn,canonical_core_assigned_units=ck if cn else None,
            canonical_core_fraction=ck/cn if cn else None,
            assigned_without_testable_canonical_core=int((eligible&event&~core).sum()),
            core_denominator_semantics='every listed native canonical core passes the same physical and opportunity-count floor; not equal detection power')
    strands=run.get('data',{}).get('strands')
    if strands is not None:
        strands=np.asarray(strands)
        if strands.shape!=eligible.shape:raise ValueError('Chemical stratum vector does not match evidence units')
        by_strand={}
        for strand in sorted(set(strands)):
            use=strands==strand;mask=use&eligible;sn=int(mask.sum());sk=int((mask&event).sum())
            sb=int((mask&native_event).sum()) if native_event is not None else None
            um=use&union_eligible;sun=int(um.sum());suk=int((um&event).sum())
            entry=dict(eligible_units=sn,assigned_units=sk if sn else None,fraction=sk/sn if sn else None,
                native_assigned_units=sb if sn else None,native_fraction=sb/sn if sn and sb is not None else None,
                union_eligible_units=sun,union_eligible_assigned_units=suk if sun else None,
                union_eligible_fraction=suk/sun if sun else None)
            if 'core_eligible' in run:
                cm=use&core;scn=int(cm.sum());sck=int((cm&event).sum())
                entry.update(canonical_core_eligible_units=scn,canonical_core_assigned_units=sck if scn else None,
                    canonical_core_fraction=sck/scn if scn else None,
                    assigned_without_testable_canonical_core=int((mask&event&~core).sum()))
            by_strand[str(strand)]=entry
        fields=['eligible_units','assigned_units','union_eligible_units','union_eligible_assigned_units']
        if native is not None:fields.append('native_assigned_units')
        if 'core_eligible' in run:
            fields.extend(['canonical_core_eligible_units','canonical_core_assigned_units',
                           'assigned_without_testable_canonical_core'])
        for field in fields:
            if sum(c[field] or 0 for c in by_strand.values()) != (result[field] or 0):
                raise ValueError(f'Chemical strata do not partition pooled {field}')
        result.update(by_strand=by_strand,
            stratum_semantics='CT/GA are separate chemical readouts; Hia5 alignment orientations are pooled. Exact same event and fixed denominator rule in every stratum; no rate balancing.')
    return result


def summarize_correspondences(edges, runs, threshold=.5, minimum_shared_support=3):
    if not 0 < threshold <= 1:
        raise ValueError('A positive fixed decoded-membership threshold is required')
    pair_edges=defaultdict(list)
    for e in edges:
        if e['comparability_mask']:
            pair_edges[e['left_dataset'],e['right_dataset']].append(e)
    groups=[]; annotations={}
    for (left,right),links in sorted(pair_edges.items()):
        adjacency=defaultdict(set)
        by_pair={}
        for e in links:
            a,b=e['left_family'],e['right_family']
            adjacency[a].add(b);adjacency[b].add(a)
            if (a,b) in by_pair:raise ValueError('Duplicate atomic XCR pair')
            by_pair[a,b]=e
        pending=set(adjacency)
        centers={c['family']:(c['consensus_start'],c['consensus_end'])
            for ds in (left,right) for c in runs[ds]['catalog']}
        left_ids={c['family'] for c in runs[left]['catalog']}
        while pending:
            stack=[min(pending)];component=set()
            while stack:
                node=stack.pop()
                if node in component:continue
                component.add(node);stack.extend(adjacency[node]-component)
            pending-=component
            aa=sorted(component & left_ids);bb=sorted(component-left_ids)
            contained=[by_pair[a,b] for a in aa for b in bb if (a,b) in by_pair]
            complete=len(contained)==len(aa)*len(bb)
            common_core=all(max(centers[f][0] for f in ff)<min(centers[f][1] for f in ff) for ff in (aa,bb))
            reasons=[]
            if not complete:reasons.append('nonrectangular_correspondence')
            if not common_core:reasons.append('no_common_native_core')
            gid='XG_'+digest([left,right,aa,bb])[:16]
            kind='one_to_one' if len(aa)==len(bb)==1 else 'coarse_union'
            group=dict(group_id=gid,left_dataset=left,right_dataset=right,
                left_families=aa,right_families=bb,edge_ids=sorted(e['edge_id'] for e in contained),
                native_intervals={f:list(centers[f]) for f in aa+bb},
                kind=kind if not reasons else 'unresolved_mapping',
                status='descriptive_counts' if not reasons else 'unresolved_mapping',reasons=reasons,
                membership_threshold=float(threshold),complete_bipartite=complete,
                shared_native_core_within_each_assay=common_core,
                rate_agreement_used_for_selection=False,rescue_counts_used=False,
                quantification_equivalence_established=False,molecule_detection_power_established=False)
            if not reasons:
                group['left_counts']=_count(runs[left],aa,threshold)
                group['right_counts']=_count(runs[right],bb,threshold)
                if not group['left_counts']['eligible_units'] or not group['right_counts']['eligible_units']:
                    group['status']='no_common_eligible_units'
            omitted={};recurrent_omitted=False
            for ds,ff in ((left,aa),(right,bb)):
                core=(max(centers[f][0] for f in ff),min(centers[f][1] for f in ff))
                siblings=[]
                for c in runs[ds]['catalog']:
                    if c['family'] in ff or not(c['consensus_start']<core[1] and c['consensus_end']>core[0]):continue
                    i=c['family_index'];support=int((runs[ds]['proposal_membership'][:,i]>=threshold).sum())
                    siblings.append(dict(family=c['family'],interval=[c['consensus_start'],c['consensus_end']],
                        per_member_threshold_assignments=support,recurrent=support>=minimum_shared_support))
                omitted[ds]=siblings;recurrent_omitted |= any(c['recurrent'] for c in siblings)
            group['omitted_overlapping_native_families']=omitted
            group['unrepresented_recurrent_overlap']=recurrent_omitted
            group['overlap_recurrence_floor']=minimum_shared_support
            group['count_identity_status']='overlapping_native_alternatives_unrepresented' if recurrent_omitted else 'no_recurrent_overlapping_native_alternatives'
            group['individual_target_roles_established']=all(e.get('target_role_comparability_mask',False) for e in contained)
            group['target_role_unresolved_edges']=[e['edge_id'] for e in contained if not e.get('target_role_comparability_mask',False)]
            group['coarse_event_replacement_validated']=False
            for e in contained:
                annotations[e['edge_id']]=dict(count_group_id=gid,
                    left_comparable_degree=len(adjacency[e['left_family']]),
                    right_comparable_degree=len(adjacency[e['right_family']]),
                    count_comparison=group['kind'],
                    individual_count_comparison_unambiguous=kind=='one_to_one' and not recurrent_omitted
                        and group['status']=='descriptive_counts' and bool(e.get('target_role_comparability_mask',False)),
                    unrepresented_recurrent_overlap=recurrent_omitted)
            groups.append(group)
    return groups,annotations
