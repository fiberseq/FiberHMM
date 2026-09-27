"""Run the lattice recaller on one evidence payload and write results compatible with the staged engine's outputs.

Outputs (output_dir): evidence.json.gz, manifest.json, result.json.gz (browser snapshot: per-dataset catalog and
per-call records, plus a 'recaller' block per class), classes.tsv (one row per class x channel), molecules.tsv.gz
(one row per class x channel x scored molecule) and the standard report (families.tsv, calls.tsv, report.html).
"""
from __future__ import annotations

import csv
import gzip
import tempfile
import time
from pathlib import Path

import numpy as np

from ..artifacts import digest, write_json
from ..parameters import options_dict
from . import discovery as D, model as Mo, units as Un

MODE = 'lattice_recaller'


def _run_all(fn, tasks, cores, done=None):
    """fn(*task) for every task, in task order; across processes when cores > 1 (results are deterministic, so the
    output does not depend on the worker count)."""
    done = done or (lambda *_: None)
    if cores <= 1 or len(tasks) <= 1:
        out = []
        for t in tasks:
            out.append(fn(*t)); done(len(out) - 1)
        return out
    import warnings
    from ..execution import _process_pool
    pool = _process_pool(min(cores, len(tasks)))
    try:
        with warnings.catch_warnings():
            # loky recycles a worker whose memory grew and reruns its task; results are unaffected.
            warnings.filterwarnings('ignore', message='A worker stopped while some jobs were given to the executor')
            futures = [pool.submit(fn, *t) for t in tasks]; out = []
            for i, fut in enumerate(futures):
                out.append(fut.result()); done(i)
        return out
    finally:
        pool.shutdown(wait=True)
SCHEMA = 'fiberhmm.consensus.v1'


def _tiles(r0, r1, tile, step):
    if r1 - r0 <= tile:
        return [(r0, r1)]
    starts = list(range(r0, max(r0 + 1, r1 - tile + step), step))
    return [(a, min(a + tile, r1)) for a in starts]


def _jitter(g, chemistry, opt):
    J = opt.jitter_hia5_bp if chemistry.startswith('hia5') else opt.jitter_dddb_bp if chemistry == 'dddb' else opt.jitter_ddda_bp
    return dict(L=[g['L'][0] - J, g['L'][1]], R=[g['R'][0], g['R'][1] + J], span=g['span'])


def discover(sources, region, opt, progress, cores=1):
    tiles = _tiles(region['start'], region['end'], opt.tile_bp, opt.tile_step_bp)
    found, diagnostics, tasks, which = [], [], [], []
    for t, (w0, w1) in enumerate(tiles):
        units = Un.tile_units(sources, w0, w1, opt.call_min_llr, opt.call_max_bp)
        if len(units) >= opt.minimum_channel_units:
            tasks.append((units, opt)); which.append((t, w0, w1, len(units)))
    done = lambda i: progress('recaller_discovery', f'tile {which[i][0] + 1}/{len(tiles)} {which[i][1]}-{which[i][2]}: {which[i][3]} molecules')
    for (t, w0, w1, _), (classes, diag) in zip(which, _run_all(D.discover_tile, tasks, cores, done)):
        for g in classes:
            g['tile'] = t
        found += classes; diagnostics.append(dict(tile=[w0, w1], classes=len(classes), **diag))
    classes = D.dedupe(found)
    kept = [g for g in classes if D.core_width(g) >= opt.minimum_core_bp]
    dropped = [dict(span=[round(g['span'][0], 1), round(g['span'][1], 1)], core_bp=D.core_width(g)) for g in classes if g not in kept]
    for i, g in enumerate(kept):
        g['id'] = f'class_{i + 1:03d}'
    return kept, dropped, tiles, diagnostics


def quantify(sources, classes, tiles, opt, progress, cores=1):
    rows, mols, broad = [], [], []
    efficiency = Un.efficiency_factors(sources) if opt.efficiency_calibration else None
    chem = {s['dataset_id']: s['chemistry'] for s in sources}
    groups = D.overlap_groups(classes)
    tasks, meta = [], []
    jmax = max(opt.jitter_ddda_bp, opt.jitter_dddb_bp, opt.jitter_hia5_bp)
    for gi, grp in enumerate(groups):
        centre = np.mean([sum(classes[x]['span'])/2 for x in grp])
        t = next((i for i, (a, b) in enumerate(tiles) if a <= centre < b), min(range(len(tiles)), key=lambda i: abs(sum(tiles[i])/2 - centre)))
        # The tile's sites must cover the group's whole scoring window (edge boxes, jitter and flank), or no molecule
        # spans it and the group is silently unscored (e.g. a class near a tile edge). Keep the centre tile when it
        # does; otherwise an overlapping tile that does; otherwise a window built around the group.
        lo = min(classes[x]['L'][0] for x in grp) - jmax - opt.flank_bp; hi = max(classes[x]['R'][1] for x in grp) + jmax + opt.flank_bp
        covers = lambda w: w[0] - Un.SITE_PAD + 10 <= lo and hi + 10 <= w[1] + Un.SITE_PAD
        if not covers(tiles[t]):
            ok = [i for i, w in enumerate(tiles) if covers(w)]
            t = min(ok, key=lambda i: abs(sum(tiles[i])/2 - centre)) if ok else None
        window = tiles[t] if t is not None else (int(lo) - 20, int(hi) + 20)
        units = Un.tile_units(sources, *window, opt.call_min_llr, opt.call_max_bp, efficiency)
        for ch in sorted({u['ch'] for u in units}):
            us = [u for u in units if u['ch'] == ch]
            if len(us) < opt.minimum_channel_units:
                continue
            chemistry = chem[ch.split('::', 1)[0]]
            gs = [_jitter(classes[x], chemistry, opt) for x in grp]
            f = Un.unknown_accessible_fraction(sources, ch, {u['uid'] for u in us})
            tasks.append((us, gs, f, opt)); meta.append((gi, grp, ch, len(us)))
    done = lambda i: progress('recaller_quantify', f'group {meta[i][0] + 1}/{len(groups)} ({len(meta[i][1])} classes), {meta[i][2]}: {meta[i][3]} molecules')
    for (gi, grp, ch, _), (us, gs, f, _o), res in zip(meta, tasks, _run_all(Mo.fit_channel, tasks, cores, done)):
        if res is None:
            continue
        k = len(grp); w = res['w']
        for c, x in enumerate(grp):
            g = classes[x]; lb = Mo.wilson_lo(w[c]*res['n'], res['n'])
            gain = res['support_gain'][c]
            rows.append(dict(class_id=g['id'], group=gi + 1, channel=ch, dataset=ch.split('::', 1)[0], strand=ch.split('::', 1)[1],
                             start=round(g['span'][0], 1), end=round(g['span'][1], 1), L0=res['gs'][c]['L'][0], L1=res['gs'][c]['L'][1], R0=res['gs'][c]['R'][0], R1=res['gs'][c]['R'][1],
                             calls=g['calls'], stability=round(g['stability'], 3), molecules=res['n'], prevalence=round(float(w[c]), 4),
                             prevalence_edge=round(res['tiers'][c]['edge'], 4), prevalence_loose=round(res['tiers'][c]['loose'], 4),
                             prevalence_lower_bound=round(lb, 4), broader=round(float(w[k]), 4), other_shape=round(float(w[k + 1]), 4),
                             accessible=round(float(w[k + 2]), 4), support_gain_nats=None if gain != gain else round(gain, 2),
                             supported=bool(gain == gain and gain >= opt.support_gain_nats and lb >= opt.support_minimum_lower_bound),
                             resolution_nats=round(res['resolution'][c], 2), resolved=bool(res['resolution'][c] >= opt.resolution_nats),
                             spots=';'.join(f'{p}:{v:.3f}' for p, v in res['spots'][c].items()), edge_contraction=res['edges'][c],
                             unknown_accessible_fraction=round(f, 4),
                             efficiency=None if not efficiency else round(efficiency.get(ch, 1.), 4)))
            for u, p, call in zip(res['units'], res['P'][:, c], res['calls']):
                lab, lbf = Mo.label(p, w[c], opt.bf_threshold)
                # tier: 'core' for members; for the rest, 'edge' / 'loose' when their likelier wider or other-shape
                # protection matches the class (the looser prevalence tiers), with that call's edges.
                upper = call['tiers'][c] if lab != 'member' else None
                edges = call['classes'][c] if lab == 'member' else (upper[1] if upper else None)
                tier = 'core' if lab == 'member' else (upper[0] if upper else '')
                mols.append(dict(class_id=g['id'], channel=ch, unit_id=u['uid'], posterior=round(float(p), 4), log_bf=round(lbf, 3), label=lab,
                                 tier=tier, start=edges[0] if edges else None, end=edges[1] if edges else None,
                                 edge_range=[list(edges[2]), list(edges[3])] if edges else None))
        # Molecules best explained by protection wider than every class of the group (e.g. a nucleosome over it).
        for u, pb, call in zip(res['units'], res['P'][:, k], res['calls']):
            if pb >= .5 and call['broader'] is not None:
                broad.append(dict(group=gi + 1, classes=';'.join(classes[x]['id'] for x in grp), channel=ch, unit_id=u['uid'],
                                  posterior=round(float(pb), 4), start=round(call['broader'][0]), end=round(call['broader'][1]),
                                  edge_range=[list(call['broader'][2]), list(call['broader'][3])]))
    return rows, mols, broad


def _write_tsv(path, rows, fields, compress=False):
    opener = (lambda p: gzip.open(p, 'wt', newline='')) if compress else (lambda p: open(p, 'w', newline=''))
    with opener(path) as handle:
        w = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', extrasaction='ignore'); w.writeheader(); w.writerows(rows)


def _unit_recaller_calls(calls, proposals, unit):
    """A molecule's recaller calls as displayed. A class call takes the edges of the molecule's native call for that
    class when there is one (the caller's own boundaries; edge_source 'native'), else keeps its lattice edges. Only
    calls labelled with the class lend their edges: an unlabelled native call over a looser-tier call is the wider
    protection it sits in, not the class footprint.
    Wider-protection stretches run to the molecule's nearest marks on either side (not the scoring window) and
    overlapping stretches are merged."""
    out = []
    for c in calls:
        if c['kind'] != 'class':
            continue
        a, b = c['lattice_interval']
        best = max(((min(b, p['source_interval'][1]) - max(a, p['source_interval'][0]), p['source_interval']) for p in proposals
                    if p.get('family') == c['family']), default=(0, None))
        if best[1] is not None and best[0] >= .5*min(b - a, best[1][1] - best[1][0]):
            c = dict(c, interval=[int(best[1][0]), int(best[1][1])], edge_source='native')
        out.append(c)
    stretches = sorted((c for c in calls if c['kind'] == 'broader'), key=lambda r: r['interval'])
    if stretches:
        pos = np.asarray(unit['positions']); hit = np.asarray(unit['hits']) > 0; marks = pos[hit]
        merged = []
        for c in stretches:
            a, b = c['interval']
            left = marks[marks < a]; right = marks[marks >= b]
            a = int(left.max()) + 1 if len(left) else int(unit['reference_start'])
            b = int(right.min()) if len(right) else int(unit['reference_end'])
            if merged and a < merged[-1]['interval'][1]:
                m = merged[-1]; m['interval'] = [m['interval'][0], max(b, m['interval'][1])]
                m['classes'] = sorted(set(m['classes']) | set(c.get('classes', []))); m['posterior'] = max(m['posterior'], c['posterior'])
            else:
                merged.append(dict(c, interval=[a, b], consensus_interval=[a, b], edge_range=None))
        out += merged
    return sorted(out, key=lambda r: r['interval'])


def snapshot(sources, classes, rows, mols, opt, stage='resolved', region=None, broad=None):
    from ..harmonized_families.presentation import browser_unit
    datasets = {}
    by_class_ch = {(r['class_id'], r['channel']): r for r in rows}
    # Classes no channel supports are candidates the scoring rejected: not catalogued, not used as call labels.
    shown = {g['id'] for g in classes} if opt.report_unsupported_classes else {r['class_id'] for r in rows if r['supported']}
    members = {}
    for m in mols:
        if m['label'] == 'member' and m['class_id'] in shown:
            members.setdefault((m['channel'].split('::', 1)[0], m['unit_id']), []).append((m['posterior'], m['class_id']))
    span = {g['id']: g['span'] for g in classes}
    # The recaller's own calls per molecule: each class it is a member of (its own edges and the class's consensus
    # edges), and the wider protection that best explains a molecule assigned to broader protection.
    rcalls = {}
    for m in mols:
        if m.get('tier') and m['class_id'] in shown and m.get('start') is not None:
            c0, c1 = span[m['class_id']]
            rcalls.setdefault((m['channel'].split('::', 1)[0], m['unit_id']), []).append(dict(
                kind='class', family=m['class_id'], tier=m['tier'], interval=[int(m['start']), int(m['end'])],
                lattice_interval=[int(m['start']), int(m['end'])], consensus_interval=[int(round(c0)), int(round(c1))],
                edge_range=m.get('edge_range'), edge_source='lattice', posterior=m['posterior'], log_bf=m['log_bf']))
    for b in broad or []:
        rcalls.setdefault((b['channel'].split('::', 1)[0], b['unit_id']), []).append(dict(
            kind='broader', family=None, interval=[int(b['start']), int(b['end'])], consensus_interval=[int(b['start']), int(b['end'])],
            edge_range=b.get('edge_range'), posterior=b['posterior'], classes=b['classes'].split(';')))
    member_counts = {}
    for m in mols:
        if m['label'] == 'member' and m['class_id'] in shown:
            key = (m['class_id'], m['channel']); member_counts[key] = member_counts.get(key, 0) + 1
    for s in sources:
        ds = s['dataset_id']; catalog = []; records = []; call_counts = {}
        for u in s['units']:
            key = (ds, u['unit_id']); proposals = []
            for c in u.get('native_multi_interval_calls', []):
                a, b = c['interval']
                if c.get('llr', 99.) < opt.call_min_llr or (region and not (a < region['end'] and b > region['start'])):
                    continue                      # records cover native calls overlapping the analysed region
                fams = [cid for post, cid in sorted(members.get(key, []), reverse=True)
                        if min(b, span[cid][1]) > max(a, span[cid][0])]
                for i, fid in enumerate(fams):
                    k2 = (fid, u['strand']); cc = call_counts.setdefault(k2, [0, 0]); cc[0] += 1; cc[1] += i == 0
                proposals.append(dict(source_call_id=f"{ds}::{u['unit_id']}:{a}:{b}", source_interval=[a, b], source_intervals=[[a, b]],
                                      source_record_indices=[], interval=[a, b], family=fams[0] if fams else None,
                                      compatible_alternatives=fams[1:], compatible_families=fams,
                                      classification_status='compatible_catalog_label' if fams else 'provisional_unresolved',
                                      assessment_status='lattice_member' if fams else 'lattice_unassigned', inference_eligible=True,
                                      unclassified=not fams, cr_mode=MODE, new_call=False, stage=stage, raw_interval_unchanged=True,
                                      exclusive_assignment=False, llr=c.get('llr')))
            rc = _unit_recaller_calls(rcalls.get(key, []), proposals, u)
            if proposals or rc:
                records.append(dict(unit_id=f"{ds}::{u['unit_id']}", strand=u['strand'], source_calls=[p['source_interval'] for p in proposals],
                                    proposals=proposals, recaller_calls=rc))
        for g in classes:
            if g['id'] not in shown:
                continue
            counts = {}; block = {}
            for (cid, ch), r in by_class_ch.items():
                if cid != g['id'] or r['dataset'] != ds:
                    continue
                n_members = member_counts.get((cid, ch), 0); calls, primary = call_counts.get((cid, r['strand']), [0, 0])
                counts[r['strand']] = dict(eligible_units=r['molecules'], compatible_units=n_members, primary_units=n_members,
                                           original_calls=calls, compatible_calls=calls, primary_calls=primary,
                                           eligible_unit_semantics='molecules spanning the scoring window (lattice recaller)')
                block[r['strand']] = {k: r[k] for k in ('prevalence', 'prevalence_edge', 'prevalence_loose', 'prevalence_lower_bound', 'broader', 'other_shape', 'accessible',
                                                        'support_gain_nats', 'supported', 'resolution_nats', 'resolved', 'spots', 'edge_contraction',
                                                        'L0', 'L1', 'R0', 'R1', 'molecules', 'unknown_accessible_fraction', 'efficiency')}
            if not counts:
                continue
            catalog.append(dict(family=g['id'], consensus_start=g['span'][0], consensus_end=g['span'][1],
                                edge_uncertainty=dict(left=list(g['L']), right=list(g['R'])), physical_edge_support=None,
                                hypothesis=None, model_status='fitted', stage=stage, fit_flags=[], shared_scope=MODE,
                                evidence_summary=dict(calls=g['calls'], stability=g['stability'], memberships='lattice_em_three_way'),
                                classification_counts=counts, strand_resolution=None,
                                source_units=sum(c['compatible_units'] for c in counts.values()), recaller=block))
        catalog.sort(key=lambda f: (f['consensus_start'], f['consensus_end'], f['family']))
        datasets[ds] = dict(chemistry=s['chemistry'], units=[browser_unit(ds, u) for u in s['units']],
                            sr=dict(status='disabled', records=[], changed=0),
                            cr=dict(status='complete', cr_mode=MODE, catalog=catalog, records=records, stage=stage),
                            rescue=dict(status='disabled', records=[], accepted_calls=0), split=dict(status='disabled', records=[], accepted_spans=0),
                            comparability=dict(status='disabled', records=[]))
    return dict(datasets=datasets, cross=dict(status='disabled', edges=[], count_groups=[], comparable_edges=0, shared_family_edges=0,
                                             count_semantics='Lattice recaller: no cross-dataset edges; classes are shared by construction'))


CLASS_FIELDS = ['class_id', 'group', 'channel', 'dataset', 'strand', 'start', 'end', 'L0', 'L1', 'R0', 'R1', 'calls', 'stability', 'molecules',
                'prevalence', 'prevalence_edge', 'prevalence_loose', 'prevalence_lower_bound', 'broader', 'other_shape', 'accessible', 'support_gain_nats', 'supported',
                'resolution_nats', 'resolved', 'spots', 'edge_contraction', 'unknown_accessible_fraction', 'efficiency']
MOLECULE_FIELDS = ['class_id', 'channel', 'unit_id', 'posterior', 'log_bf', 'label', 'tier', 'start', 'end']
BROADER_FIELDS = ['group', 'classes', 'channel', 'unit_id', 'posterior', 'start', 'end']


def run_lattice_recaller(payload, options, output_dir=None, progress=None):
    from ..harmonized_families.workflow import prepare_sources
    started = time.monotonic(); progress = progress or (lambda *_: None); opt = options['recaller']
    out = Path(output_dir or tempfile.mkdtemp(prefix='fiberhmm-recaller-')); out.mkdir(parents=True, exist_ok=True)
    region = payload['region']
    if region['end'] <= region['start'] or region['end'] - region['start'] > options['compute'].maximum_region_bp:
        raise ValueError('Invalid region or maximum analysis span exceeded; no cropping')
    before = digest(payload); sources = prepare_sources(payload, options)
    write_json(out/'evidence.json.gz', payload)
    if options['families'].stop_after == 'native':
        classes, dropped, tiles, diagnostics, rows, mols, broad = [], [], [], [], [], [], []
    else:
        cores = options['compute'].cores
        classes, dropped, tiles, diagnostics = discover(sources, region, opt, progress, cores)
        rows, mols, broad = quantify(sources, classes, tiles, opt, progress, cores) if classes else ([], [], [])
    snap = snapshot(sources, classes, rows, mols, opt, region=region, broad=broad)
    stages = [dict(id='resolved', label='Lattice recaller', seconds=time.monotonic() - started, families=len(classes),
                   original_calls=sum(len(r['proposals']) for d in snap['datasets'].values() for r in d['cr']['records']),
                   assignments=sum(bool(p['family']) for d in snap['datasets'].values() for r in d['cr']['records'] for p in r['proposals']))]
    realized = {s['dataset_id']: sorted({u['strand'] for u in s['units']}) for s in sources}
    populated = sum(bool(v) for v in realized.values())
    observed_sr = any(s['chemistry'] in ('ddda', 'dddb') and len([c for c in realized[s['dataset_id']] if c != 'BOTH']) > 1 for s in sources)
    mode = 'XCR' if options['cross'].enabled else 'SR' if options['sr'].enabled else 'CR'
    realized_mode = ('SR/XCR' if observed_sr else 'XCR') if populated > 1 else 'SR' if observed_sr else 'CR' if populated else 'no_evidence'
    warnings = []
    if options['cross'].enabled and populated < 2:
        warnings.append('Fewer than two datasets have eligible evidence; cross-dataset support is not established.')
    if options['sr'].enabled and not observed_sr:
        warnings.append('No dataset has both chemical strands represented; strand-shared support is not established.')
    if any(s['chemistry'] in ('ddda', 'dddb') and not s.get('evidence_units', {}).get('physical_duplex_independence_established') for s in sources):
        warnings.append('Physical duplex independence is not established by this workflow; strand evidence must not be counted as proven independent duplex molecules.')
    receipt = dict(schema=SCHEMA, status='complete', cr_mode=MODE, region=region, parameters=options_dict(options), input_digest=before,
                   mode=mode, mode_realized=realized_mode, realized_channels=realized, data_warnings=warnings, seconds=time.monotonic() - started, all_units=True, read_sample_cap=None,
                   family_count_cap=None, native_source_modified=False, stages=stages, last_stage='resolved',
                   datasets=[dict(dataset_id=s['dataset_id'], chemistry=s['chemistry'], units=len(s['units']), model=s.get('model_manifest'))
                             for s in sources],
                   recaller=dict(classes=len(classes), unsupported_classes=sorted({g['id'] for g in classes} - {r['class_id'] for r in rows if r['supported']}),
                                 dropped_by_core_rule=dropped, tiles=[list(t) for t in tiles], discovery=diagnostics,
                                 efficiency_calibration=opt.efficiency_calibration),
                   browser_sources=payload.get('browser_sources'), pooling=payload.get('pooling'), input_files=payload.get('input_files'),
                   display_mode=mode, presentation_revision='lattice_recaller_v1')
    result = dict(schema=SCHEMA, cr_mode=MODE, manifest=receipt, stages=stages, stage_results={'resolved': snap}, final_stage='resolved',
                  recaller=dict(classes=[dict(id=g['id'], start=g['span'][0], end=g['span'][1], L=list(g['L']), R=list(g['R']), calls=g['calls'],
                                              stability=g['stability'], supported_channels=sum(1 for r in rows if r['class_id'] == g['id'] and r['supported']))
                                         for g in classes], rows=rows), **snap)
    if digest(payload) != before:
        raise AssertionError('Input payload mutated')
    write_json(out/'manifest.json', receipt); write_json(out/'result.json.gz', result)
    _write_tsv(out/'classes.tsv', rows, CLASS_FIELDS); _write_tsv(out/'molecules.tsv.gz', mols, MOLECULE_FIELDS, compress=True)
    _write_tsv(out/'broader.tsv.gz', broad, BROADER_FIELDS, compress=True)
    from ..report import write_report
    write_report(result, out)
    progress('complete', f'{len(classes)} classes; {len(rows)} class x channel estimates')
    return result
