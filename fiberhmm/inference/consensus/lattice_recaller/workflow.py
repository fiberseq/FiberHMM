"""Run the lattice recaller on one evidence payload and write results compatible with the staged engine's outputs.

Outputs (output_dir): evidence.json.gz, manifest.json, result.json.gz (browser snapshot: per-dataset catalog and
per-call records, plus a 'recaller' block per class), classes.tsv (one row per class x channel, including channels
that could not score the class), molecules.tsv.gz (one row per class x channel x scored molecule), broader.tsv.gz
(molecules best explained by protection wider than a group's classes) and the standard report (families.tsv,
calls.tsv, report.html). Columns are documented in docs/CONSENSUS_WORKFLOW.md.
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
    output does not depend on the worker count).

    Uses the run's shared worker pool (execution.shared_worker_pool) when one is open, else a private pool. If a task
    fails, the progress callback raises (a cancelled Browser job) or the run is interrupted, the queued tasks are
    dropped and the workers killed at once instead of every remaining task running to completion."""
    done = done or (lambda *_: None)
    if cores <= 1 or len(tasks) <= 1:
        out = []
        for t in tasks:
            out.append(fn(*t)); done(len(out) - 1)
        return out
    import warnings
    from ..execution import stage_executor
    executor, release = stage_executor(min(cores, len(tasks)))
    failed = True
    try:
        with warnings.catch_warnings():
            # loky recycles a worker whose memory grew and reruns its task; results are unaffected.
            warnings.filterwarnings('ignore', message='A worker stopped while some jobs were given to the executor')
            futures = [executor.submit(fn, *t) for t in tasks]; out = []
            for i, fut in enumerate(futures):
                out.append(fut.result()); done(i)
        failed = False
        return out
    finally:
        # On failure the workers are killed and every pending task fails with the executor's shutdown error.
        # (Cancelling loky futures first would break its manager thread and leave the workers alive.)
        release(failed)


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


def _class_geometry(g):
    return dict(class_id=g['id'], start=round(g['span'][0], 1), end=round(g['span'][1], 1), L0=g['L'][0], L1=g['L'][1],
                R0=g['R'][0], R1=g['R'][1], calls=g['calls'], stability=round(g['stability'], 3))


def _unscored_row(g, gi, ch, reason, molecules):
    return dict(_class_geometry(g), group=gi + 1, channel=ch, dataset=ch.split('::', 1)[0], strand=ch.split('::', 1)[1],
                molecules=molecules, status='unscored', unscored_reason=reason)


def quantify(sources, classes, tiles, opt, progress, cores=1, frozen=None):
    """Per class x channel estimates. frozen: a frozen.FrozenClasses (transfer); each channel is then scored with its
    fixed boxes and spots, no fitting. Returns (rows, molecules, broader, unscored): unscored lists the class x channel
    pairs no estimate exists for, with the reason (too few molecules in the window, or none spanning the scoring
    window), so a class near a data end is reported as unscored rather than as unsupported."""
    rows, mols, broad, unscored = [], [], [], []
    efficiency = Un.efficiency_factors(sources) if opt.efficiency_calibration else None
    chem = {s['dataset_id']: s['chemistry'] for s in sources}
    channels = sorted({Un.channel_of(s, u) for s in sources for u in s['units']})
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
        for ch in channels:
            us = [u for u in units if u['ch'] == ch]
            if len(us) < opt.minimum_channel_units:
                unscored += [_unscored_row(classes[x], gi, ch, f'too_few_molecules ({len(us)} < recaller.minimum_channel_units '
                                           f'{opt.minimum_channel_units} overlapping the window)', len(us)) for x in grp]
                continue
            chemistry = chem[ch.split('::', 1)[0]]
            gs = [_jitter(classes[x], chemistry, opt) for x in grp]
            f = Un.unknown_accessible_fraction(sources, ch, {u['uid'] for u in us})
            tasks.append((us, gs, f, opt) if frozen is None else (us, gs, f, opt, frozen.fixed(classes, grp, ch, chemistry)))
            meta.append((gi, grp, ch, len(us)))
    done = lambda i: progress('recaller_quantify', f'group {meta[i][0] + 1}/{len(groups)} ({len(meta[i][1])} classes), {meta[i][2]}: {meta[i][3]} molecules')
    fit = Mo.fit_channel if frozen is None else frozen.score
    for (gi, grp, ch, n_units), (us, gs, f, _o, *_), res in zip(meta, tasks, _run_all(fit, tasks, cores, done)):
        if res is None:
            unscored += [_unscored_row(classes[x], gi, ch, f'no_spanning_molecules (none of {n_units} molecules has sites beyond '
                                       f'both ends of the scoring window, edge boxes +/- recaller.flank_bp {opt.flank_bp})', 0) for x in grp]
            continue
        k = len(grp); w = res['w']
        for c, x in enumerate(grp):
            g = classes[x]; lb = Mo.wilson_lo(w[c]*res['n'], res['n'])
            gain = res['support_gain'][c]
            supported = bool(gain == gain and gain >= opt.support_gain_nats and lb >= opt.support_minimum_lower_bound)
            rows.append(dict(class_id=g['id'], group=gi + 1, channel=ch, dataset=ch.split('::', 1)[0], strand=ch.split('::', 1)[1],
                             start=round(g['span'][0], 1), end=round(g['span'][1], 1), L0=res['gs'][c]['L'][0], L1=res['gs'][c]['L'][1], R0=res['gs'][c]['R'][0], R1=res['gs'][c]['R'][1],
                             status='supported' if supported else 'unsupported', unscored_reason='',
                             calls=g['calls'], stability=round(g['stability'], 3), molecules=res['n'], prevalence=round(float(w[c]), 4),
                             prevalence_edge=round(res['tiers'][c]['edge'], 4), prevalence_loose=round(res['tiers'][c]['loose'], 4),
                             prevalence_lower_bound=round(lb, 4), broader=round(float(w[k]), 4), other_shape=round(float(w[k + 1]), 4),
                             accessible=round(float(w[k + 2]), 4), support_gain_nats=None if gain != gain else round(gain, 2),
                             supported=supported,
                             resolution_nats=round(res['resolution'][c], 2), resolved=bool(res['resolution'][c] >= opt.resolution_nats),
                             spots=';'.join(f'{p}:{v:.3f}' for p, v in res['spots'][c].items()), edge_contraction=res['edges'][c],
                             spot_rates={str(p): float(v) for p, v in res['spots'][c].items()},   # full precision (frozen transfer)
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
    return rows, mols, broad, unscored


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


def _call_fits_class(lattice, a, b, g, opt):
    """Whether native call [a, b) could be class g's footprint: no wider than recaller.call_max_bp (the discovery
    filter), and its lattice-censored edge ranges (discovery's own edge rule) reach the class's left and right edge
    boxes. A member molecule's other calls (e.g. a nucleosome-sized call over a small class) keep no class label."""
    if b - a > opt.call_max_bp:
        return False
    l0, l1, r0, r1 = D.censor(lattice, a, b, opt.censor_bp)
    return l0 <= g['L'][1] and l1 >= g['L'][0] and r0 <= g['R'][1] and r1 >= g['R'][0]


def _strand_resolution(block, chemistry, opt):
    """Recaller strand trust for a DAF class, the analogue of the staged engine's per-strand core-ceiling verdict: a
    chemical strand is trusted when its expected evidence per molecule over the class (resolution_nats) reaches
    recaller.resolution_nats. None for Hia5 (alignment orientations of one molecule, pooled)."""
    strands = {k: v for k, v in block.items() if k in ('CT', 'GA')}
    if chemistry not in ('ddda', 'dddb') or not strands:
        return None
    trusted = sorted(k for k, v in strands.items() if v['resolved'])
    return dict(trusted_strands=trusted, trusted_strand='both' if len(trusted) > 1 else trusted[0] if trusted else 'none',
                core_resolution='resolved' if trusted else 'below_resolution_threshold',
                supported_strands=sorted(k for k, v in strands.items() if v['supported']),
                resolution_nats={k: strands[k]['resolution_nats'] for k in sorted(strands)},
                resolution_threshold_nats=opt.resolution_nats,
                semantics='lattice_recaller: trusted = expected evidence per molecule over the class reaches '
                          'recaller.resolution_nats; supported = the held-out support test passed on that strand')


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
    span = {g['id']: g['span'] for g in classes}; by_id = {g['id']: g for g in classes}
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
        boxes = {cid: _jitter(g, s['chemistry'], opt) for cid, g in by_id.items()}
        for u in s['units']:
            key = (ds, u['unit_id']); proposals = []; lattice = None
            for c in u.get('native_multi_interval_calls', []):
                a, b = c['interval']
                if c.get('llr', 99.) < opt.call_min_llr or (region and not (a < region['end'] and b > region['start'])):
                    continue                      # records cover native calls overlapping the analysed region
                fams, share = [], {}
                if members.get(key):
                    if lattice is None:
                        lattice = dict(pos=np.asarray(u['positions']), hit=np.asarray(u['hits']) > 0)
                    for post, cid in sorted(members[key], reverse=True):
                        if min(b, span[cid][1]) > max(a, span[cid][0]) and _call_fits_class(lattice, a, b, boxes[cid], opt):
                            # q0: the molecule's EM class posterior on its channel, x255 (1..255; 0 = no class).
                            fams.append(cid); share[cid] = max(1, min(255, int(round(255*post))))
                for i, fid in enumerate(fams):
                    k2 = (fid, u['strand']); cc = call_counts.setdefault(k2, [0, 0]); cc[0] += 1; cc[1] += i == 0
                proposals.append(dict(source_call_id=f"{ds}::{u['unit_id']}:{a}:{b}", source_interval=[a, b], source_intervals=[[a, b]],
                                      source_record_indices=[], interval=[a, b], family=fams[0] if fams else None,
                                      compatible_alternatives=fams[1:], compatible_families=fams,
                                      classification_status='compatible_catalog_label' if fams else 'provisional_unresolved',
                                      assessment_status='lattice_member' if fams else 'lattice_unassigned', inference_eligible=True,
                                      unclassified=not fams, cr_mode=MODE, new_call=False, stage=stage, raw_interval_unchanged=True,
                                      exclusive_assignment=False, llr=c.get('llr'),
                                      q0=share[fams[0]] if fams else 0, member_q0=share))
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
                                classification_counts=counts, strand_resolution=_strand_resolution(block, s['chemistry'], opt),
                                source_units=sum(c['compatible_units'] for c in counts.values()), recaller=block))
        catalog.sort(key=lambda f: (f['consensus_start'], f['consensus_end'], f['family']))
        datasets[ds] = dict(chemistry=s['chemistry'], units=[browser_unit(ds, u) for u in s['units']],
                            sr=dict(status='disabled', records=[], changed=0),
                            cr=dict(status='complete', cr_mode=MODE, catalog=catalog, records=records, stage=stage),
                            rescue=dict(status='disabled', records=[], accepted_calls=0), split=dict(status='disabled', records=[], accepted_spans=0),
                            comparability=dict(status='disabled', records=[]))
    return dict(datasets=datasets, cross=dict(status='disabled', edges=[], count_groups=[], comparable_edges=0, shared_family_edges=0,
                                             count_semantics='Lattice recaller: no cross-dataset edges; classes are shared by construction'))


CLASS_FIELDS = ['class_id', 'group', 'channel', 'dataset', 'strand', 'start', 'end', 'L0', 'L1', 'R0', 'R1', 'status', 'unscored_reason',
                'calls', 'stability', 'molecules',
                'prevalence', 'prevalence_edge', 'prevalence_loose', 'prevalence_lower_bound', 'broader', 'other_shape', 'accessible', 'support_gain_nats', 'supported',
                'resolution_nats', 'resolved', 'spots', 'edge_contraction', 'unknown_accessible_fraction', 'efficiency']
MOLECULE_FIELDS = ['class_id', 'channel', 'unit_id', 'posterior', 'log_bf', 'label', 'tier', 'start', 'end', 'edge_range']
BROADER_FIELDS = ['group', 'classes', 'channel', 'unit_id', 'posterior', 'start', 'end', 'edge_range']


def _range_text(value):
    """edge_range [[l0, l1], [r0, r1]] as 'l0-l1,r0-r1' (the left and right edge ranges); empty when absent."""
    return '' if not value else f'{value[0][0]}-{value[0][1]},{value[1][0]}-{value[1][1]}'


def _tsv_rows(rows):
    return [dict(r, edge_range=_range_text(r.get('edge_range'))) for r in rows]


def _abutting_warning():
    return ('recaller.abutting is EXPERIMENTAL: its configuration weights are not a normalized prior (the class gains '
            'likelihood with the number of possible abutting extensions, even without chemical evidence), so '
            'prevalence and support for abutting classes are biased upward. Treat these results as exploratory.')


def run_lattice_recaller(payload, options, output_dir=None, progress=None, frozen=None):
    """frozen: a frozen.FrozenClasses; classes and tiles then come from the catalog (no discovery)."""
    from ..harmonized_families.workflow import prepare_sources
    started = time.monotonic(); progress = progress or (lambda *_: None); opt = options['recaller']
    out = Path(output_dir or tempfile.mkdtemp(prefix='fiberhmm-recaller-')); out.mkdir(parents=True, exist_ok=True)
    region = payload['region']
    if region['end'] <= region['start'] or region['end'] - region['start'] > options['compute'].maximum_region_bp:
        raise ValueError('Invalid region or maximum analysis span exceeded; no cropping')
    if options['families'].stop_after != 'resolved':
        raise ValueError('families.stop_after applies to staged_native_families only; the lattice recaller runs in one pass')
    before = digest(payload); sources = prepare_sources(payload, options)
    write_json(out/'evidence.json.gz', payload)
    warnings = []
    if opt.abutting:
        import warnings as _warnings
        warnings.append(_abutting_warning()); _warnings.warn(warnings[-1]); progress('recaller_discovery', 'WARNING: ' + warnings[-1])
    from ..execution import shared_worker_pool
    cores = options['compute'].cores
    with shared_worker_pool(cores):
        classes, dropped, tiles, diagnostics = discover(sources, region, opt, progress, cores) if frozen is None else frozen.discovery()
        rows, mols, broad, unscored = quantify(sources, classes, tiles, opt, progress, cores, frozen=frozen) if classes else ([], [], [], [])
    snap = snapshot(sources, classes, rows, mols, opt, region=region, broad=broad)
    stages = [dict(id='resolved', label='Lattice recaller', seconds=time.monotonic() - started, families=len(classes),
                   original_calls=sum(len(r['proposals']) for d in snap['datasets'].values() for r in d['cr']['records']),
                   assignments=sum(bool(p['family']) for d in snap['datasets'].values() for r in d['cr']['records'] for p in r['proposals']))]
    realized = {s['dataset_id']: sorted({u['strand'] for u in s['units']}) for s in sources}
    populated = sum(bool(v) for v in realized.values())
    # One class catalog is discovered from every channel and dataset together and scored on each channel: CR, shared
    # by construction. No SR boundary normalization or XCR relationship graph is computed, whatever sr/cross.enabled say.
    mode = 'CR'
    realized_mode = 'CR' if populated else 'no_evidence'
    if any(s['chemistry'] in ('ddda', 'dddb') and not s.get('evidence_units', {}).get('physical_duplex_independence_established') for s in sources):
        warnings.append('Physical duplex independence is not established by this workflow; strand evidence must not be counted as proven independent duplex molecules.')
    status = {g['id']: 'supported' if any(r['class_id'] == g['id'] and r['supported'] for r in rows) else
              'unsupported' if any(r['class_id'] == g['id'] for r in rows) else 'unscored' for g in classes}
    receipt = dict(schema=SCHEMA, status='complete', cr_mode=MODE, region=region, parameters=options_dict(options), input_digest=before,
                   mode=mode, mode_realized=realized_mode, realized_channels=realized, data_warnings=warnings,
                   mode_semantics='lattice_recaller: CR over one class catalog shared by every channel and dataset; no SR or XCR is computed', seconds=time.monotonic() - started, all_units=True, read_sample_cap=None,
                   family_count_cap=None, native_source_modified=False, stages=stages, last_stage='resolved',
                   datasets=[dict(dataset_id=s['dataset_id'], chemistry=s['chemistry'], units=len(s['units']), model=s.get('model_manifest'))
                             for s in sources],
                   recaller=dict(classes=len(classes),
                                 # Scored on at least one channel and supported on none (hidden from the catalog unless
                                 # recaller.report_unsupported_classes).
                                 unsupported_classes=sorted(c for c, v in status.items() if v == 'unsupported'),
                                 # Scored on no channel: no molecule spans its scoring window, or too few molecules
                                 # (e.g. at a contig or data end). Not a verdict on the class; reasons per channel.
                                 unscored_classes=sorted(c for c, v in status.items() if v == 'unscored'),
                                 unscored=[{k: r[k] for k in ('class_id', 'channel', 'unscored_reason', 'molecules')} for r in unscored],
                                 dropped_by_core_rule=dropped, tiles=[list(t) for t in tiles], discovery=diagnostics,
                                 efficiency_calibration=opt.efficiency_calibration),
                   browser_sources=payload.get('browser_sources'), pooling=payload.get('pooling'), input_files=payload.get('input_files'),
                   display_mode=mode, presentation_revision='lattice_recaller_v1')
    if frozen is not None:
        receipt['transfer'] = frozen.provenance()
    result = dict(schema=SCHEMA, cr_mode=MODE, manifest=receipt, stages=stages, stage_results={'resolved': snap}, final_stage='resolved',
                  recaller=dict(classes=[dict(id=g['id'], start=g['span'][0], end=g['span'][1], L=list(g['L']), R=list(g['R']), calls=g['calls'],
                                              stability=g['stability'], supported_channels=sum(1 for r in rows if r['class_id'] == g['id'] and r['supported']),
                                              status=status[g['id']])
                                         for g in classes], rows=rows, unscored=unscored), **snap)
    if frozen is not None:
        result['transfer'] = receipt['transfer']
    if digest(payload) != before:
        raise AssertionError('Input payload mutated')
    write_json(out/'manifest.json', receipt); write_json(out/'result.json.gz', result)
    order = {g['id']: i for i, g in enumerate(classes)}
    _write_tsv(out/'classes.tsv', sorted(rows + unscored, key=lambda r: (order[r['class_id']], r['channel'])), CLASS_FIELDS)
    _write_tsv(out/'molecules.tsv.gz', _tsv_rows(mols), MOLECULE_FIELDS, compress=True)
    _write_tsv(out/'broader.tsv.gz', _tsv_rows(broad), BROADER_FIELDS, compress=True)
    from ..report import write_report
    write_report(result, out)
    progress('complete', f'{len(classes)} classes; {len(rows)} class x channel estimates')
    return result
