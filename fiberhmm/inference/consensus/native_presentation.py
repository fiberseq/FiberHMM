"""Pure display projection of frozen native-classification evidence.

No fitting, new calls, changed source geometry, or invented posterior values.
The one per-call score added here, q0 (class support), is a declared relative
profile-likelihood share under a uniform prior, not a calibrated probability.
The compatible set grows monotonically; the best full-span representative may
change as additional alternatives become available. The source span never does.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import math

MODE = 'native_family_distribution'

Q0_SEMANTICS = ('assigned_class_share_of_scored_class_evidence_x255: w_k = exp(recipient_optimum_k - '
                'floor_adjusted_loss_k) over every scored candidate class, uniform prior; relative profile '
                'likelihood, not a calibrated probability; 0 = unresolved')


def class_shares(scores):
    """Each scored class's share of the evidence for one call (uniform prior).

    ``floor_adjusted_loss`` is the recipient's best unconstrained profile
    log-likelihood minus the best value its projections reach under the
    class's shape penalty (nats), so exp(-loss) is a relative profile
    likelihood. Candidates can be scored on slightly different grids, so each
    is referred to a common scale through its own ``recipient_optimum`` when
    every candidate carries one. Unscored or loss-free entries are ignored.
    """
    usable = {}
    for score in scores:
        if score.get('status') == 'scored' and score.get('floor_adjusted_loss') is not None:
            usable[score['family']] = score
    if not usable:
        return {}
    anchored = all(s.get('recipient_optimum') is not None for s in usable.values())
    log_weight = {family: (float(s['recipient_optimum']) if anchored else 0.) - float(s['floor_adjusted_loss'])
                  for family, s in usable.items()}
    top = max(log_weight.values())
    total = sum(math.exp(v-top) for v in log_weight.values())
    return {family: math.exp(v-top)/total for family, v in log_weight.items()}


def q0_byte(share):
    """Class-support byte: round(255 x share); 0 when the class was not scored."""
    return int(round(255*share)) if share else 0


def member_families(accepted, membership_loss_odds=1.):
    """Families a call belongs to: the primary, plus ties within the margin.

    ``accepted`` is the compatible evidence already sorted by the primary rule.
    With odds 1 this is exactly ``[primary]``, the historical winner-take-all
    membership. Above 1 a call also joins every compatible family whose
    floor-adjusted loss is within ln(odds) of the BEST compatible loss; the
    anchor is the best loss, not the primary's, because the primary is chosen
    by endpoint distance and need not be the best supported hypothesis. The
    primary label is never changed by this.
    """
    if not accepted:
        return []
    if not membership_loss_odds > 1.:
        return [accepted[0]['family']]
    best = min(s['floor_adjusted_loss'] for s in accepted)
    limit = best+math.log(membership_loss_odds)+1e-9
    return [accepted[0]['family']]+sorted(s['family'] for s in accepted[1:]
                                          if s['floor_adjusted_loss'] <= limit)


def classify_proposal(proposal, reference_percent, membership_loss_odds=1.):
    if (isinstance(reference_percent, bool) or not math.isfinite(reference_percent)
            or not 50 <= reference_percent <= 99.999):
        raise ValueError('Classification reference must be between 50 and 99.999 percent')
    if isinstance(membership_loss_odds, bool) or not membership_loss_odds >= 1.:
        raise ValueError('Tie-set membership odds must be at least 1')
    accepted = [s for s in proposal.get('candidate_evidence', []) if s['status'] == 'scored'
                and s.get('predictive_tail_interval', [0., 0.])[1] >= 1.-reference_percent/100.]
    accepted.sort(key=lambda s:(s['geometry_distance_sq'],s['floor_adjusted_loss'],s['family']))
    out = dict(proposal, family=accepted[0]['family'] if accepted else proposal['unresolved_family'],
        classification_status='compatible_catalog_label' if accepted else 'provisional_unresolved',
        primary_evidence=accepted[0] if accepted else None,
        compatible_alternatives=[s['family'] for s in accepted[1:]],
        classification_reference_percent=reference_percent)
    # Default membership adds no field, so frozen artifacts stay byte-identical.
    if membership_loss_odds > 1.:
        out['member_families'] = member_families(accepted, membership_loss_odds)
        out['membership_loss_odds'] = float(membership_loss_odds)
    # Class support over EVERY scored candidate, not only those accepted at
    # this reference, so it does not move with the stringency slider.
    shares = class_shares(proposal.get('candidate_evidence', []))
    members = out.get('member_families') or [s['family'] for s in accepted]
    out['q0'] = q0_byte(shares.get(out['family'], 0.)) if accepted else 0
    out['member_q0'] = {family: q0_byte(shares.get(family, 0.)) for family in members}
    return out


def reclassify_records(cr, reference_percent, membership_loss_odds=1.):
    """Return records at a new display reference; `cr` stays byte-for-byte intact."""
    if cr.get('cr_mode') != MODE:
        raise ValueError('Native presentation cannot reinterpret legacy CR evidence')
    return [dict(row,proposals=[classify_proposal(p,reference_percent,membership_loss_odds)
                                for p in row['proposals']]) for row in cr['records']]


def classification_counts(records, catalog):
    counts=defaultdict(lambda:defaultdict(lambda:dict(original_calls=0,primary_calls=0,compatible_calls=0,
                                                    eligible_units=set(),primary_units=set())))
    tie_sets=any('member_families' in call for row in records for call in row['proposals'])
    for row in records:
        uid,strand=row['unit_id'],row['strand']
        for call in row['proposals']:
            for score in call.get('candidate_evidence',[]):
                c=counts[score['family']][strand];c['original_calls']+=1
                if score['status'] in ('scored','core_contradicted'):c['eligible_units'].add(uid)
                if score['family'] in [call['family'],*call['compatible_alternatives']]:c['compatible_calls']+=1
            if call['primary_evidence'] is not None:
                c=counts[call['family']][strand];c['primary_calls']+=1;c['primary_units'].add(uid)
            # Tie-set membership is reported beside the primary counts, never
            # folded into them: a call counted under several families must stay
            # distinguishable from a call that is the primary of one.
            if tie_sets:
                for fid in call.get('member_families',[]):
                    c=counts[fid][strand]
                    c.setdefault('member_calls',0);c.setdefault('member_units',set())
                    c['member_calls']+=1;c['member_units'].add(uid)
    return [dict(f,classification_counts={strand:{k:len(v) if isinstance(v,set) else v for k,v in c.items()}
                  for strand,c in counts[f['family']].items()}) for f in catalog]


def browser_cr(stratum, catalog, result, reference_percent=99.9, *, region=None, model_versions=None):
    """Portable span-preserving records with a complete candidate-score ledger.

    Tie-set membership is read from the producer's own diagnostics, so the
    display cannot declare a membership the classification did not use.
    """
    membership_loss_odds = float(result.get('diagnostics', {}).get('membership_loss_odds', 1.))
    pooled_hia5=stratum.get('chemistry','').startswith('hia5')
    rows={u['unit_id']:dict(unit_id=u['unit_id'],strand='pooled' if pooled_hia5 else u['strand'],
         source_calls=u['representative_raw_tf_intervals'],proposals=[]) for u in stratum['units']}
    if pooled_hia5:
        for u in stratum['units']:
            rows[u['unit_id']]['alignment_orientation']=u.get('alignment_orientation',u['strand'])
    region=region or stratum.get('region') or (stratum['units'][0].get('_region') if stratum['units'] else None)
    if isinstance(region,dict):region=(region['start'],region['end'])
    if region is None:raise ValueError('Explicit analysis region required for native Browser source-call accounting')
    for call,scores in zip(result['calls'],result['call_family_evidence']):
        span=[call['start'],call['end']]
        if region and (span[0]>=region[1] or span[1]<=region[0]):continue
        fid=stratum['dataset_id']+':unresolved_'+hashlib.sha256(f'{span[0]}|{span[1]}'.encode()).hexdigest()[:12]
        proposal=dict(source_call_id=f"{stratum['dataset_id']}:{call['unit_id']}:{call['ordinal']}",
            source_ordinal=call['ordinal'],source_ordinals=[call['ordinal']],source_intervals=[span],interval=span,
            unresolved_family=fid,candidate_evidence=scores,cr_mode=MODE,boundary_changed=False)
        rows[call['unit_id']]['proposals'].append(
            classify_proposal(proposal,reference_percent,membership_loss_odds))
    records=list(rows.values())
    for r in records:r['proposals'].sort(key=lambda p:p['source_ordinal'])
    models={m['family']:m for m in result['family_models']}
    versions=result.get('model_versions',{}) if model_versions is None else model_versions
    nodes=[]
    for f in catalog:
        model=models.get(f['family'],{})
        node=dict(f,cr_mode=MODE,model_status=model.get('status'),
            native_fit_center=model.get('normalized_geometry',{}).get('mean'),
            native_fit_covariance=model.get('normalized_geometry',{}).get('covariance'),
            native_geometry_summary=model.get('normalized_geometry'),
            untruncated_gaussian_location=model.get('fitted_distribution_center'),
            untruncated_gaussian_covariance=model.get('fitted_distribution_covariance'),
            fit_diagnostics=model.get('fit_diagnostics',{}),source_units=model.get('source_units',0))
        if f['family'] in versions:
            version=versions[f['family']]
            node['model_provenance']={k:version[k] for k in
                ('origin','model_version_sha256','producer_snapshot_digest')}
        nodes.append(node)
    output=dict(status='complete',cr_mode=MODE,records=records,catalog=classification_counts(records,nodes),
                diagnostics=result['diagnostics'],reference_percent=reference_percent,
                original_spans_preserved=True,new_calls=0)
    if membership_loss_odds>1.:
        output['membership_loss_odds']=membership_loss_odds
        output['membership_rule']='primary_plus_ties_within_loss_margin'
    if 'catalog_update' in result:
        update=result['catalog_update']
        output['native_catalog_update']={k:update[k] for k in ('policy','original_family_count',
            'added_family_count','initial_snapshot_digest','augmented_snapshot_digest')}
    return output
