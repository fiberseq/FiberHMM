"""Footprint classes conditioned on fiberhmm-nfr module state (``fiberhmm-consensus --read-table --nfr-states DIR``).

fiberhmm-nfr writes each read's NFR variant state (``molecules.tsv.gz``: one row per read x NFR, ``map`` = the read's
most probable configuration, e.g. a variant name or ``closed``). This joins those states to the consensus per-read
table by read name (DAF: read name and chemical strand, CT/GA) and, for every consensus window x NFR of an nfr run
whose region overlaps the window x state x class x channel, writes ``class_by_nfr_state.tsv``:

* ``reads``: reads in that state scored for the class's group on the channel; ``mean_posterior`` and ``members``
  (label ``member``) from the window's pooled fit;
* ``em_prevalence``: the class prevalence refitted by EM on those reads alone, with the classes and every read's
  log-likelihoods fixed (the read table's ``loglik``/``ll_*`` columns). A state's prevalence is therefore not pulled
  toward the pooled weights, and classes are the same in every state (the pooled catalog). State ``all`` is every
  joined read, for reference.

Only reads whose NFR record is ``callable`` carry a state; censored reads (no nucleosome on both sides) are left out.
A run in which no read joins (for example read names rewritten by a trimming script) is an error, not an empty table.
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np

STATE_FIELDS = ['window_index', 'window', 'chrom', 'window_start', 'window_end', 'window_strand', 'nfr_run', 'nfr',
                'nfr_start', 'nfr_end', 'state', 'dataset', 'strand', 'group', 'class_id', 'role', 'reads',
                'mean_posterior', 'members', 'em_prevalence']
OUTPUT = 'class_by_nfr_state.tsv'
MIN_READS = 1


def find_tables(root):
    """[(run label, run directory, region dict)] of an nfr output: one run, or the window_* runs of a --bed run."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError('not a directory')
    runs = [root] if (root/'molecules.tsv.gz').is_file() else sorted(p for p in root.glob('window_*') if (p/'molecules.tsv.gz').is_file())
    if not runs:
        raise ValueError('no fiberhmm-nfr molecules.tsv.gz here or in window_* subdirectories')
    out = []
    for run in runs:
        try:
            result = json.loads((run/'result.json').read_text())
        except (OSError, ValueError):
            raise ValueError(f'{run}: missing or unreadable result.json (needed for the NFR window)') from None
        region = result.get('region') or {}
        if not {'chrom', 'start', 'end'} <= set(region):
            raise ValueError(f'{run}/result.json has no region')
        out.append((run.name if run != root else root.name, run, dict(chrom=region['chrom'], start=int(region['start']), end=int(region['end']))))
    return out


def _key(name, strand):
    return (name, strand if strand in ('CT', 'GA') else '')


def load_states(run):
    """{nfr: (start, end)} and {(read_name, strand key): {nfr: state}} of one nfr run (callable reads only)."""
    bounds = {}
    variants = Path(run)/'variants.tsv'
    if variants.is_file():
        with open(variants, newline='') as handle:
            for row in csv.DictReader(handle, delimiter='\t'):
                if row.get('nfr') and row.get('nfr_start'):
                    bounds[row['nfr']] = (int(float(row['nfr_start'])), int(float(row['nfr_end'])))
    states = {}
    with gzip.open(Path(run)/'molecules.tsv.gz', 'rt', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            if row.get('status') != 'callable' or not row.get('map'):
                continue
            states.setdefault(_key(row['read_name'], row.get('strand', '')), {})[row['nfr']] = row['map']
            bounds.setdefault(row['nfr'], (None, None))
    return bounds, states


def em_weights(M, iters=1000):
    """Mixture weights for a reads x hypotheses log-likelihood matrix (the recaller's EM, same floor)."""
    from .lattice_recaller.model import em
    return em(np.asarray(M, float), iters)[0]


def class_by_state(out, specs, root, *, rules=None, anchor_offset=0, progress=None):
    """Write out/class_by_nfr_state.tsv from the units' read tables; returns a summary with join rates."""
    from .read_table import READ_TABLE, assign_roles, window_anchor, window_classes
    from .lattice_recaller import discovery as D
    runs = find_tables(root)
    cache = {}
    joined_total = reads_total = 0; rates = []
    with open(Path(out)/OUTPUT, 'w', newline='') as handle:
        w = csv.writer(handle, delimiter='\t', lineterminator='\n')
        w.writerow(STATE_FIELDS)
        for s in specs:
            folder = Path(s['folder']); table = folder/READ_TABLE; win = s['window']
            if not table.is_file():
                continue
            overlapping = [(label, run) for label, run, reg in runs
                           if reg['chrom'] == win['chrom'] and reg['start'] < win['end'] and reg['end'] > win['start']]
            # Load each overlapping run once and keep only the runs this window needs (memory: a few runs at a time).
            cache = {label: cache.get(label) or load_states(run) for label, run in overlapping}
            # Read the window: per (dataset, strand, group) the reads' class posteriors and log-likelihood rows.
            groups = {}; class_ids = {}; reads = set()
            with gzip.open(table, 'rt', newline='') as fh:
                for row in csv.DictReader(fh, delimiter='\t'):
                    reads.add(_key(row['read_name'], row['strand']))
                    if row['scored'] != 'True':
                        continue
                    key = (row['dataset'], row['strand'], int(row['group']))
                    ids = class_ids.setdefault(key, [])
                    if row['class_id'] not in ids:
                        ids.append(row['class_id'])
                    rec = groups.setdefault(key, {}).setdefault(_key(row['read_name'], row['strand']),
                                                                 dict(post={}, label={}, ll={}, alt=None))
                    rec['post'][row['class_id']] = float(row['posterior']); rec['label'][row['class_id']] = row['label']
                    rec['ll'][row['class_id']] = float(row['loglik'])
                    rec['alt'] = [float(row['ll_broader']), float(row['ll_other_shape']), float(row['ll_accessible'])]
            joined = {k for k in reads if any(k in st for _, st in cache.values())}
            joined_total += len(joined); reads_total += len(reads)
            rates.append(dict(window=win['name'], reads=len(reads), joined=len(joined), nfr_runs=[label for label, _ in overlapping]))
            if not joined:
                continue
            roles = {}
            if rules:
                classes, _ = window_classes(folder)
                geo = [dict(id=c['id'], span=(c['start'], c['end'])) for c in classes]
                group_of = {geo[x]['id']: gi + 1 for gi, grp in enumerate(D.overlap_groups(geo)) for x in grp}
                roles, _ = assign_roles(rules, [dict(id=c['id'], start=float(c['start']), end=float(c['end']), group=group_of[c['id']])
                                                for c in classes], window_anchor(win, anchor_offset), win.get('strand', '+'))
            base = [s['index'] + 1, win['name'], win['chrom'], win['start'], win['end'], win.get('strand', '+')]
            for label, _run in overlapping:
                bounds, states = cache[label]
                for nfr in sorted(bounds):
                    nb = bounds[nfr]
                    for (ds, strand, g), members in sorted(groups.items()):
                        ids = class_ids[(ds, strand, g)]
                        by_state = {}
                        for k, rec in members.items():
                            st = states.get(k, {}).get(nfr)
                            if st is None:
                                continue
                            by_state.setdefault(st, []).append(rec); by_state.setdefault('all', []).append(rec)
                        for st in sorted(by_state, key=lambda x: (x != 'all', x)):
                            recs = by_state[st]
                            if len(recs) < MIN_READS:
                                continue
                            M = np.array([[r['ll'][c] for c in ids] + r['alt'] for r in recs], float)
                            wts = em_weights(M)
                            for ci, cid in enumerate(ids):
                                w.writerow(base + [label, nfr, '' if nb[0] is None else nb[0], '' if nb[1] is None else nb[1],
                                                   st, ds, strand, g, cid, roles.get(cid, (None, None, ''))[2], len(recs),
                                                   round(float(np.mean([r['post'][cid] for r in recs])), 4),
                                                   sum(r['label'][cid] == 'member' for r in recs), round(float(wts[ci]), 4)])
            if progress is not None:
                progress.report('nfr_states', f"{win['name']}: {len(joined)}/{len(reads)} reads joined to NFR states",
                                completed=s['index'] + 1, total=len(specs))
    if reads_total and not joined_total:
        raise ValueError(f'--nfr-states {root}: no read of the consensus windows has an NFR state. The two runs must read the '
                         'same BAM records (read names are the join key; a trimming script that renames reads breaks it), and '
                         'the NFR runs must overlap the consensus windows')
    return dict(directory=str(Path(root).resolve()), runs=len(runs), reads=reads_total, joined=joined_total,
                join_rate=round(joined_total/reads_total, 4) if reads_total else None, windows=rates)
