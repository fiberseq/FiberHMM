# Extracted reference kernels; see SOURCE_MANIFEST.json.
from .run_bounded_parent_panel import evaluate_cohort, call_key

def candidate_inputs(case, proposal):
    models = {m['family']: m for m in case['models']}
    r = proposal['radius']
    keys = {tuple(k) for f in proposal['children'] for k in models[f]['source_call_keys']}
    calls = [c for c in case['calls'] if call_key(c) in keys]
    b = proposal['center_bounds']
    lo = min([c['start'] for c in calls] + [models[f]['domain'][0] for f in proposal['children']] + [b[0][0] - r])
    hi = max([c['end'] for c in calls] + [models[f]['domain'][1] for f in proposal['children']] + [b[1][1] + r])
    region = [max(lo, case['source_extent'][0]), min(hi, case['source_extent'][1])]
    return (calls, region)
