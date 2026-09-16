# Extracted reference kernels; see SOURCE_MANIFEST.json.
from collections import Counter
import hashlib
from .raw_family_segmented_region import intersect_intervals, subtract_intervals, prepare_stratum

def scaffold_intervals(original, region):
    return subtract_intervals(intersect_intervals(original['msp_intervals'], [list(region)]), original['raw_nuc_intervals'])

def call_ledger(source, region, prepared_units):
    """Every in-scope original baseline record has a stable, unchanged interval.

    Exact duplicate intervals on the same molecule are one biological mark
    with all original record indices retained. No matching/family gate drops a
    call. Later native rescans remain a separately identified inventory.
    """
    (lo, hi) = region
    records = []
    rescans = []
    stats = Counter()
    for (original, u) in zip(source['units'], prepared_units):
        native = {}
        for (j, c) in enumerate(original.get('native_multi_interval_calls', [])):
            (a, b) = c['interval']
            if a < hi and b > lo:
                native.setdefault((a, b), []).append(c)
                rescans.append(dict(unit_id=u['unit_id'], source_record_index=j, **c))
        by_interval = {}
        for (j, interval) in enumerate(original['raw_tf_intervals']):
            (a, b) = interval
            if a < hi and b > lo:
                by_interval.setdefault((a, b), []).append(j)
                stats['in_scope_baseline_records'] += 1
            else:
                stats['out_of_scope_baseline_records'] += 1
        scaffold = scaffold_intervals(original, region)
        for ((a, b), indices) in sorted(by_interval.items()):
            eligible = intersect_intervals(scaffold, [[a, b]])
            full = any((x <= a and b <= y for (x, y) in scaffold))
            status = 'fully_in_msp_scaffold' if full else 'partially_in_msp_scaffold' if eligible else 'outside_msp_scaffold'
            matches = native.get((a, b), [])
            records.append(dict(call_id=hashlib.sha256(f"{u['unit_id']}:{a}:{b}".encode()).hexdigest()[:20], unit_id=u['unit_id'], interval=[a, b], plot_interval=[max(lo, a), min(hi, b)], baseline_source_field='raw_tf_intervals', source_record_indices=indices, original_interval_unchanged=True, outer_region_censored=a < lo or b > hi, scaffold_status=status, eligible_scaffold_portions=eligible, matching_native_rescan_llrs=[m['llr'] for m in matches if 'llr' in m], matching_native_rescan_exists=bool(matches), family_assignment_status='not_yet_evaluated'))
            stats['unique_molecule_baseline_intervals'] += 1
            stats[status] += 1
            stats['outer_region_censored'] += a < lo or b > hi
            stats['baseline_with_exact_native_rescan_match'] += bool(matches)
    stats['native_rescan_records'] = len(rescans)
    assert sum((len(r['source_record_indices']) for r in records)) == stats['in_scope_baseline_records']
    return (records, rescans, dict(stats))
