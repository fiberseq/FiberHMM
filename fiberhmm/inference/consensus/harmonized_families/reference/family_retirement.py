# Extracted reference kernels; see SOURCE_MANIFEST.json.
from collections import defaultdict

def direct_decisions(row):
    decisions = {}
    for field in ('parent_evaluations', 'refinement_evaluations', 'representative_resolution_evaluations'):
        for score in row.get(field, []):
            value = score.get('compatible')
            if value is None or score.get('fit_warning'):
                continue
            fid = score['hypothesis']
            if fid in decisions and decisions[fid] != value:
                raise ValueError('Contradictory direct decisions for the same model')
            decisions[fid] = value
    for fid in row['display_hypotheses']:
        if decisions.get(fid) is False:
            raise ValueError('Displayed membership contradicts its native score')
        decisions[fid] = True
    return decisions

def retention_checks(case, old, replacements, evaluations, minimum_groups=2):
    """Find recurring bidirectional distinctions within proposed replacements.

An old model needs positive witnesses rejecting EVERY proposed replacement, and
each replacement needs its own recurring reverse witnesses. No all-members veto:
the parent remains promoted. This is an annotation-retention rule, not a new
population-level hypothesis test or proof of separate biochemical states.
Zero explicitly reproduces the historical unguarded retirement policy.
"""
    if not isinstance(minimum_groups, int) or isinstance(minimum_groups, bool) or minimum_groups < 0:
        raise ValueError('Nonnegative integer retention support required')
    groups = {(c['unit_id'], c['start'], c['end']): c.get('evidence_group_id', c['unit_id']) for c in case['calls']}
    forward = defaultdict(set)
    reverse = defaultdict(lambda : defaultdict(set))
    if minimum_groups:
        for row in old['records']:
            key = (row['unit_id'], *row['interval'])
            previous = direct_decisions(row)
            current = {e['hypothesis']: e.get('compatible') for e in evaluations.get(key, []) if not e.get('fit_warning')}
            group = groups.get(key, row['unit_id'])
            for (fid, replacing) in replacements.items():
                if not replacing or fid not in previous:
                    continue
                if previous[fid] is True and all((current.get(f) is False for f in replacing)):
                    forward[fid].add(group)
                elif previous[fid] is False:
                    for f in replacing:
                        if current.get(f) is True:
                            reverse[fid][f].add(group)
    checks = {}
    for (fid, replacing) in sorted(replacements.items()):
        if not replacing:
            continue
        conflicting = forward[fid] & set().union(*reverse[fid].values()) if reverse[fid] else set()
        positive = forward[fid] - conflicting
        negatives = {f: reverse[fid][f] - conflicting for f in sorted(replacing)}
        retain = bool(minimum_groups and len(positive) >= minimum_groups and all((len(v) >= minimum_groups for v in negatives.values())))
        checks[fid] = dict(retain=retain, minimum_discriminating_groups=minimum_groups, alternative_only_groups=sorted(positive), replacement_only_groups={f: sorted(v) for (f, v) in negatives.items()}, conflicting_physical_groups=sorted(conflicting), proposed_replacements=sorted(replacing), reason='recurring_bidirectional_native_discrimination' if retain else 'legacy_retirement_requested' if not minimum_groups else 'no_recurring_bidirectional_distinction', new_fits=0, new_predictive_simulations=0, population_identity_established=False)
    return checks
