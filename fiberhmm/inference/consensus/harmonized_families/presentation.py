"""Lossless Browser aliases for frozen stage hypotheses and original calls."""
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import combinations
import math
import numpy as np

from . import MODE
from .reference.raw_family_segmented_region import eligible_intervals
from .evidence import intern


def presentation_context(sources,compact=False):
    """Build physical coverage once; vectorize and cache repeated span queries."""
    groups = defaultdict(list)
    for source in sources:
        for u in source['units']:
            intervals = eligible_intervals(u, [u['reference_start'],u['reference_end']])
            groups[(source['dataset_id'],u['strand'])].extend(
                (a,b,source['dataset_id']+'::'+u['unit_id']) for a,b in intervals)
    # Chemical-strand lattices: cumulative all-miss protection LLR per unit, so
    # a state's core can be tested for what a fully protected molecule shows.
    lattices = {}; floors = {}
    for source in sources:
        if source['chemistry'] not in ('ddda', 'dddb'):
            continue
        floor = (source.get('model_manifest') or {}).get('native_minimum_llr')
        floors[source['dataset_id']] = float(floor if floor is not None else DAF_NATIVE_FLOOR)
        for u in source['units']:
            if 'positions' not in u:
                continue
            pa = np.asarray(u['p_accessible'], float); pp = np.asarray(u['p_protected'], float)
            lattices[source['dataset_id']+'::'+u['unit_id']] = (np.asarray(u['positions']),
                np.r_[0., np.cumsum(np.log1p(-pp)-np.log1p(-pa))])
    return dict(coverage={key:(np.array([a for a,b,uid in rows]),np.array([b for a,b,uid in rows]),
        np.array([uid for a,b,uid in rows],dtype=object)) for key,rows in groups.items()}, eligible={},compact=compact,evidence_pool={},
        lattices=lattices, native_floors=floors)


# Native TF LLR preset used when a source did not record its replay floor.
DAF_NATIVE_FLOOR = 5.
STRAND_LIMITED_RATIO = .5


def core_informativeness(context, uids, left, right):
    """Median over eligible units of core opportunities and the all-miss ceiling.

    The ceiling is the protection LLR a molecule would give if every core site
    were unconverted: the most one strand can say about this footprint alone."""
    ceilings = []; sites = []
    for uid in uids:
        lattice = context.get('lattices', {}).get(uid)
        if lattice is None:
            continue
        positions, cumulative = lattice
        a, b = np.searchsorted(positions, [left, right])
        ceilings.append(cumulative[b]-cumulative[a]); sites.append(b-a)
    if not ceilings:
        return None
    return dict(core_protection_ceiling_llr=float(np.median(ceilings)), core_opportunities=float(np.median(sites)))


def flag_strand_limits(by_strand, floor):
    """Which chemical strand can report this state's core, from the lattice alone.

    Sequence composition decides which sites exist on each strand and the
    context emission model decides how much each one says, so a strongly
    strand-biased core (GA-poor, CT-poor, poorly reactive contexts) leaves one
    strand unable to reach the native floor. A strand is limited when a fully
    protected molecule stays below the floor in the core and reads it at less
    than half the other strand's ceiling; its rate can err in either direction
    (missed calls, or broad calls compatible with many states) and must not be
    compared across strands. Returns the state-level verdict: the strand(s) to
    trust, and whether even they resolve the core on its own."""
    ceilings = {k: v['core_protection_ceiling_llr'] for k, v in by_strand.items()
                if v.get('core_protection_ceiling_llr') is not None}
    for strand, counts in by_strand.items():
        if strand not in ceilings:
            continue
        mine = ceilings[strand]; other = max((v for k, v in ceilings.items() if k != strand), default=None)
        counts['core_below_native_floor'] = bool(mine < floor)
        counts['strand_limited'] = bool(mine < floor and other is not None and mine < STRAND_LIMITED_RATIO*other)
        counts['native_floor_llr'] = floor
    if not ceilings:
        return None
    trusted = sorted(k for k in ceilings if not by_strand[k]['strand_limited'])
    return dict(trusted_strands=trusted,
                trusted_strand='both' if len(trusted) > 1 else trusted[0],
                core_resolution=('resolved' if all(ceilings[k] >= floor for k in trusted)
                                 else 'below_native_floor'),
                core_protection_ceiling_llr={k: ceilings[k] for k in sorted(ceilings)},
                native_floor_llr=floor, limited_ratio=STRAND_LIMITED_RATIO)


def eligible_units(context, ds, strand, left, right):
    key = (ds,strand,math.floor(left),math.ceil(right))
    if key not in context['eligible']:
        arrays=context['coverage'].get((ds,strand))
        context['eligible'][key]=(set(arrays[2][(arrays[0]<=key[2]) & (arrays[1]>=key[3])]) if arrays else set())
    return context['eligible'][key]


def browser_unit(ds, unit):
    # Native arrays are preserved once in evidence/source artifacts, not copied
    # into every presentation stage. Keep every projection identity field.
    fields=('genomic_provenance','read_name','strand','alignment_orientation','source_members','provenance',
        'library_id','source_path','record_sha256','alignment_occurrence','reference_start',
        'reference_end','fold_group_id','physical_molecule_id','physical_source_names','pairing_method','pairing_model','original_bam_tf_intervals',
        'original_bam_nuc_intervals','original_bam_msp_intervals')
    result = dict({k:deepcopy(unit[k]) for k in fields if k in unit},
        unit_id=ds+'::'+unit['unit_id'],native_intervals=deepcopy(unit['raw_tf_intervals']))
    if 'upstream_nuc_tf_recall' in unit:
        result.update(recalled_nuc_intervals=deepcopy(unit['raw_nuc_intervals']),
            recalled_msp_intervals=deepcopy(unit['msp_intervals']),
            upstream_recall_settings=deepcopy(unit['upstream_nuc_tf_recall']['settings']))
    return result


def browser_snapshot(scopes, sources, mode, stage, context=None,
                     minimum_primary_units=0, minimum_primary_fraction=0.,
                     assignment_reference_percent=99.9):
    if not 50 <= assignment_reference_percent <= 99.9:
        raise ValueError('Assignment reference must be between 50 and 99.9 percent')
    context = context or presentation_context(sources)
    datasets = {}; family_datasets = defaultdict(set); shared = {}
    raw = {s['dataset_id']+'::'+u['unit_id']:(s['dataset_id'],u) for s in sources for u in s['units']}
    for s in sources:
        ds = s['dataset_id']
        datasets[ds] = dict(chemistry=s['chemistry'], units=[browser_unit(ds,u) for u in s['units']],
            sr=dict(status='disabled', records=[], changed=0),
            cr=dict(status='complete', cr_mode=MODE, catalog=[], records=[], stage=stage),
            rescue=dict(status='disabled', records=[], accepted_calls=0),
            split=dict(status='disabled', records=[], accepted_spans=0),
            comparability=dict(status='disabled', records=[]))
    rows = defaultdict(dict)
    for case, annotation in scopes:
        hypotheses = {h['id']:h for h in annotation['hypotheses'] if h['display']}
        counts = defaultdict(Counter); primary = defaultdict(Counter); members = defaultdict(lambda: defaultdict(set))
        primary_members = defaultdict(lambda: defaultdict(set))
        for record in annotation['records']:
            uid = record['unit_id']; ds,u = raw[uid]; strand = u['strand']; iv = record['interval']
            families = sorted(record['display_hypotheses'])
            if assignment_reference_percent < 99.9:
                scores = record.get('assignment_compatibility', {})
                cut = 1. - assignment_reference_percent / 100.
                if any(s.get('exact_for_tail_cuts_at_most', 1.) < cut - 1e-12 for s in scores.values()):
                    raise ValueError('Scores were stopped for a more permissive assignment reference; '
                                     'rerun with this families.assignment_reference_percent or compute.predictive_stopping=full')
                families = [fid for fid in families
                    if scores.get(fid, {}).get('status') == 'scored'
                    and scores[fid].get('predictive_tail_interval', [0., 0.])[1]
                    >= 1. - assignment_reference_percent / 100.]
            if not set(families) <= hypotheses.keys(): raise ValueError('Membership has no active hypothesis')
            source_id = uid+':'+str(iv[0])+':'+str(iv[1])
            original = record['original']
            # Preserve the native display primary when it survives. A replaced
            # primary has no uniquely ranked merged successor: that tie break
            # is explicitly display-only, never an exclusive assignment.
            native_primary = next(iter(original.get('compatible_families',[])),None)
            if native_primary in families:
                families.remove(native_primary);families.insert(0,native_primary)
            if context['compact']:
                # intern() rebuilds every container, so only the containers
                # edited below need their own copies; the frozen record is
                # never modified.
                evidence=dict(record);evidence['original']=dict(original)
            else:evidence=deepcopy(record)
            native=evidence['original']; channel=record.get('source_channel',original.get('source_channel'))
            if channel:
                for key in ('primary_display_family','primary_evidence'):
                    value=native.get(key)
                    fid=value.get('family') if isinstance(value,dict) else value
                    if fid and not fid.startswith(channel+'::'):
                        native['source_local_'+key]=deepcopy(value)
                        if isinstance(value,dict):value=native[key]=dict(value);value['family']=channel+'::'+fid
                        else:native[key]=channel+'::'+fid
            proposal = dict(source_call_id=source_id, source_interval=list(iv), source_intervals=[list(iv)],
                source_record_indices=original.get('source_record_indices', []), interval=list(iv),
                family=families[0] if families else None, compatible_alternatives=families[1:],
                compatible_families=families, classification_status='compatible_catalog_label' if families else 'provisional_unresolved',
                assessment_status=record['status'], inference_eligible=original['inference_eligible'],
                unclassified=not families, cr_mode=MODE, new_call=False, stage=stage,
                family_evidence=intern(evidence,context['evidence_pool']) if context['compact'] else evidence, raw_interval_unchanged=True,
                primary_label_semantics='surviving_native_primary_else_deterministic_display_only',
                exclusive_assignment=False)
            proposal['assignment_reference_percent'] = assignment_reference_percent
            if record['display_hypotheses'] and not families:
                proposal['classification_status'] = 'below_assignment_stringency'
            if not families: proposal['display_color']='#94a3b8'
            row = rows[ds].setdefault(uid, dict(unit_id=uid, strand=strand, source_calls=[], proposals=[]))
            row['source_calls'].append(list(iv)); row['proposals'].append(proposal)
            for f in families:
                counts[f][(ds,strand)] += 1; members[f][(ds,strand)].add(uid); family_datasets[f].add(ds)
            if families:
                primary[families[0]][(ds,strand)] += 1
                primary_members[families[0]][(ds,strand)].add(uid)
        for fid,h in hypotheses.items():
            geometry = h.get('geometry') or {}; mean = geometry.get('mean') or h['reference_interval']
            left,right = map(float,mean); radius = h.get('physical_radius'); anchor = h.get('reference_interval')
            # Keep continuous fitted means and distinguish uncertainty from the
            # allowed physical domain. Neither is an original call boundary.
            common = dict(family=fid, consensus_start=left, consensus_end=right,
                edge_uncertainty=deepcopy(geometry.get('credible_boxes',{}).get('0.95')),
                physical_edge_support=([[v-radius,v+radius] for v in anchor] if radius is not None and anchor else None),
                hypothesis=deepcopy(h), model_status=h['status'], stage=stage,
                fit_flags=['nonconverged'] if h.get('fit_warning') else [], shared_scope=mode,
                evidence_summary=dict(children=h.get('children',[]), reused_from=h.get('reused_from'),
                    physical_radius_bp=radius, additional_matching_floor_bp=0, memberships='direct_native_predictive_compatibility'))
            shared[fid] = common
            represented=({s['dataset_id'] for s in sources} if mode=='XCR' else {key[0] for key in counts[fid]})
            for ds in sorted(represented):
                by_strand = {}
                for strand in sorted({u['strand'] for u in next(s for s in sources if s['dataset_id']==ds)['units']}):
                    eligible = eligible_units(context,ds,strand,left,right)
                    by_strand[strand] = dict(original_calls=counts[fid][(ds,strand)],
                        compatible_calls=counts[fid][(ds,strand)], primary_calls=primary[fid][(ds,strand)],
                        eligible_units=len(eligible), primary_units=len(primary_members[fid][(ds,strand)] & eligible),
                        compatible_units=len(members[fid][(ds,strand)] & eligible),
                        eligible_unit_semantics='fitted_mean_span_in_aligned_MSP_without_nucleosome_overlap')
                    if ds in context.get('native_floors', {}):
                        # Stages repeat states with identical geometry; the
                        # eligible set is determined by (ds, strand, span).
                        cache = context.setdefault('informativeness', {})
                        key = (ds, strand, left, right)
                        if key not in cache:
                            cache[key] = core_informativeness(context, eligible, left, right)
                        by_strand[strand].update(cache[key] or {})
                resolution = (flag_strand_limits(by_strand, context['native_floors'][ds])
                              if ds in context.get('native_floors', {}) else None)
                datasets[ds]['cr']['catalog'].append(dict(common, classification_counts=by_strand,
                    strand_resolution=resolution,
                    source_units=len(set().union(*(members[fid][key] for key in members[fid] if key[0]==ds)))))
    for ds in datasets:
        datasets[ds]['cr']['records'] = list(rows[ds].values())
        datasets[ds]['cr']['catalog'].sort(key=lambda f:(f['consensus_start'],f['consensus_end'],f['family']))
    # The fitted ledger deliberately retains every tested alternative, but the
    # main population-state layer must represent recurrent primary structure,
    # not every permissively compatible shape.  Gate only the Browser-facing
    # catalog/projection; the complete hypotheses and evidence remain frozen in
    # each proposal's evidence record and the stage artifacts.
    support = defaultdict(lambda: [0, 0])
    for dataset in datasets.values():
        for state in dataset['cr']['catalog']:
            for counts_by_strand in state.get('classification_counts', {}).values():
                support[state['family']][0] += int(counts_by_strand.get('primary_units', 0))
                support[state['family']][1] += int(counts_by_strand.get('eligible_units', 0))
    minimum_units = max(0, int(minimum_primary_units))
    minimum_fraction = max(0., min(1., float(minimum_primary_fraction)))
    visible = set(support)
    # Retire one weak state at a time. Simultaneous filtering discards the
    # support which its calls could contribute to a compatible alternative.
    # Memberships are frozen native-evidence tests, never proximity guesses.
    # Primary calls/units are maintained incrementally: retiring a state moves
    # only its own proposals, so each step touches those alone rather than
    # recounting every proposal (identical counts to a full recount).
    catalog_keys = Counter((ds, strand, state['family']) for ds, dataset in datasets.items()
                           for state in dataset['cr']['catalog'] for strand in state['classification_counts'])
    primary_calls = Counter(); unit_calls = defaultdict(Counter); member_count = Counter()
    unit_support = Counter(); by_family = defaultdict(list)

    def place(ds, strand, uid, fid, sign):
        key = (ds, strand, fid)
        primary_calls[key] += sign
        mean = shared[fid]
        if uid not in eligible_units(context, ds, strand, mean['consensus_start'], mean['consensus_end']):
            return
        before = unit_calls[key][uid] > 0
        unit_calls[key][uid] += sign
        change = (unit_calls[key][uid] > 0) - before
        member_count[key] += change
        unit_support[fid] += change * catalog_keys[key]

    for ds, dataset in datasets.items():
        for row in dataset['cr']['records']:
            for proposal in row['proposals']:
                if proposal['family'] is not None:
                    place(ds, row['strand'], row['unit_id'], proposal['family'], 1)
                    by_family[proposal['family']].append((ds, row['strand'], row['unit_id'], proposal))

    def rank(fid):
        units, eligible = support[fid]
        return (units, units / eligible if eligible else 0.)

    retired_any = False
    while True:
        weak = [fid for fid in visible if support[fid][0] < minimum_units
                or rank(fid)[1] < minimum_fraction]
        if not weak:
            break
        retired = min(weak, key=lambda fid: (*rank(fid), fid))
        visible.remove(retired)
        moves = []
        for ds, strand, uid, proposal in by_family.pop(retired, []):
            alternatives = [fid for fid in proposal['compatible_families'] if fid in visible]
            replacement = min(alternatives, key=lambda fid: (-rank(fid)[0], -rank(fid)[1], fid)) if alternatives else None
            proposal.setdefault('support_reassignment', dict(
                original_family=retired, steps=[],
                semantics='highest_support_evidence_compatible_remaining_state'))['steps'].append(
                    dict(hidden_family=retired, replacement_family=replacement))
            proposal['family'] = replacement
            proposal['primary_label_semantics'] = 'highest_support_evidence_compatible_remaining_state'
            moves.append((ds, strand, uid, proposal, replacement))
        for ds, strand, uid, proposal, replacement in moves:
            place(ds, strand, uid, retired, -1)
            if replacement is not None:
                place(ds, strand, uid, replacement, 1)
                by_family[replacement].append((ds, strand, uid, proposal))
        for fid in support:
            support[fid][0] = unit_support[fid]
        retired_any = True
    if retired_any:
        for ds, dataset in datasets.items():
            for state in dataset['cr']['catalog']:
                for strand, counts in state['classification_counts'].items():
                    key = (ds, strand, state['family'])
                    counts['primary_calls'] = primary_calls[key]
                    counts['primary_units'] = member_count[key]
    for dataset in datasets.values():
        original_count = len(dataset['cr']['catalog'])
        dataset['cr']['catalog'] = [
            state for state in dataset['cr']['catalog'] if state['family'] in visible
        ]
        for row in dataset['cr']['records']:
            for proposal in row['proposals']:
                retained = [
                    family for family in proposal['compatible_families']
                    if family in visible
                ]
                if proposal['family'] in retained:
                    retained.remove(proposal['family'])
                    retained.insert(0, proposal['family'])
                proposal['compatible_families'] = retained
                proposal['family'] = retained[0] if retained else None
                proposal['compatible_alternatives'] = retained[1:]
                proposal['unclassified'] = not retained
                proposal['classification_status'] = (
                    'compatible_catalog_label' if retained
                    else 'below_recurrent_state_support' if proposal['compatible_families'] or proposal.get('support_reassignment')
                    else proposal['classification_status']
                )
                if not retained:
                    proposal['display_color'] = '#94a3b8'
        dataset['cr']['support_filter'] = dict(
            minimum_primary_units=minimum_units,
            minimum_primary_fraction=minimum_fraction,
            fitted_hypotheses=original_count,
            displayed_states=len(dataset['cr']['catalog']),
            semantics='primary independent units divided by fitted-span eligible units',
        )
    edges = []
    if mode == 'XCR' and stage != 'native':
        for fid, ids in sorted(family_datasets.items()):
            if fid not in visible: continue
            for left,right in combinations(sorted(ids),2):
                f=shared[fid]; interval=[f['consensus_start'],f['consensus_end']]
                edges.append(dict(edge_id=fid+'|'+left+'|'+right, left_dataset=left, right_dataset=right,
                    left_family=fid, right_family=fid, left_interval=interval, right_interval=interval,
                    status='shared_native_hypothesis', comparability_mask=False, shared_hypothesis=True,
                    semantics='Shared geometry with direct native-call memberships; not equal detection power or occupancy'))
    return dict(datasets=datasets, cross=dict(status='complete' if mode=='XCR' and stage!='native' else 'disabled',
        edges=edges, count_groups=[], comparable_edges=0, shared_family_edges=len(edges),
        count_semantics='Original-call multi-compatible counts; not an exclusive abundance partition'))
