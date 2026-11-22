"""Per-read class posteriors of a lattice-recaller run (``recaller.read_table``), role rules, and batch aggregation.

One window (``read_posteriors.tsv.gz`` in the run directory) has one row per evidence unit (read; DAF: read and
chemical strand) x discovered class, so every read of the window appears once for every class, scored or not:

* ``read_name``, ``fold`` (its hash fold, fiberhmm.io.folds; md5 mod 10 when no cross-fitting was asked for) and
  ``discovery`` (whether that fold discovered the classes; every read when there was no cross-fitting);
* ``dataset``, ``strand`` (the channel's strand: ``pooled`` for Hia5, ``CT``/``GA`` for DAF) and ``unit_id``;
* the class (``group``, ``class_id``, ``class_start``/``class_end`` in reference coordinates) and its status on the
  read's channel (``supported``, ``unsupported`` or ``unscored``);
* ``scored``: whether the read spans the class's scoring window on its channel. A scored read has ``posterior``,
  ``log_bf`` and ``label`` for the class and the group's three non-class posteriors ``p_broader`` (protection wider
  than every class of the group, e.g. a nucleosome over it), ``p_other_shape`` and ``p_accessible``. Within one
  group and read, the class posteriors and these three sum to 1. ``loglik`` is the read's log-likelihood under the
  class and ``ll_broader``/``ll_other_shape``/``ll_accessible`` under the three alternatives (nats; the EM's
  molecule x hypothesis matrix), so prevalences can be refitted on any subset of reads with the classes fixed.
  An unscored read has these fields empty.

A batch run (``fiberhmm-consensus --bed ... --read-table``) concatenates the windows into the top-level
``read_posteriors.tsv.gz`` with window columns, positions relative to each window's anchor (the TSS) and, with
``--role-rules``, a role per class and per group's broader component; it also writes ``class_roles.tsv`` (one row
per window x class) and ``read_roles.tsv.gz`` (one row per window x read with a posterior per role). Roles are
applied when the aggregate is built, so changing the rule file needs no recompute (``--continue`` rebuilds it).

Role rule file (JSON)::

    {"schema": "fiberhmm.consensus.roles.v1",
     "rules": [
       {"role": "upstream_small", "width": [null, 30], "end": {"lt": 0}},
       {"role": "core_wide", "start": {"le": -10}, "end": {"ge": 10}},
       {"role": "wide_over_core", "component": "broader", "group_roles": ["core_wide"]}
     ]}

Each rule names a ``role`` and conditions on ``start``, ``end``, ``center`` and ``width`` (bp; ``start``/``end``
relative to the anchor in transcription orientation, so ``start`` is the upstream edge). A condition is either an
inclusive ``[min, max]`` (``null`` = open) or an object with any of ``gt``, ``ge``, ``lt``, ``le``. Rules are tried in
file order and the first match wins. ``component`` is ``class`` (default: the rule labels discovered classes) or
``broader`` (it labels a group's broader-protection component; its geometry is the span of the group's classes, and
``group_roles`` optionally requires a class of the group to carry one of those roles). No roles are built in.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

READ_FIELDS = ['read_name', 'fold', 'discovery', 'dataset', 'strand', 'unit_id', 'group', 'class_id', 'class_start',
               'class_end', 'class_status', 'scored', 'posterior', 'log_bf', 'label', 'p_broader', 'p_other_shape',
               'p_accessible', 'loglik', 'll_broader', 'll_other_shape', 'll_accessible']
WINDOW_FIELDS = ['window_index', 'window', 'chrom', 'window_start', 'window_end', 'window_strand', 'anchor']
REL_FIELDS = ['class_rel_start', 'class_rel_end', 'class_width', 'role', 'broader_role']
CLASS_ROLE_FIELDS = WINDOW_FIELDS + ['class_id', 'group', 'class_start', 'class_end', 'class_rel_start', 'class_rel_end',
                                     'class_center', 'class_width', 'role', 'broader_role', 'status', 'supported_channels',
                                     'max_prevalence']
READ_TABLE = 'read_posteriors.tsv.gz'
CLASS_ROLES = 'class_roles.tsv'
READ_ROLES = 'read_roles.tsv.gz'
ROLE_SCHEMA = 'fiberhmm.consensus.roles.v1'
GEOMETRY = ('start', 'end', 'center', 'width')
OPS = {'gt': lambda v, t: v > t, 'ge': lambda v, t: v >= t, 'lt': lambda v, t: v < t, 'le': lambda v, t: v <= t}


def _r(value, digits=4):
    return '' if value is None else round(float(value), digits)


def read_fold(name, spec):
    """(fold, discovery) of one read: its fold under the run's spec (or the default md5 mod 10 when none)."""
    from fiberhmm.io.folds import fold_of
    if spec is None:
        return fold_of(name), True
    fold = spec.fold(name)
    return fold, fold in spec.discover


def window_rows(sources, classes, rows, unscored, mols, components, spec):
    """The per-read rows of one window (see the module docstring), in a deterministic order."""
    from .lattice_recaller import discovery as D
    from .lattice_recaller import units as Un
    group_of, column = {}, {}
    for gi, grp in enumerate(D.overlap_groups(classes)):
        for c, x in enumerate(grp):
            group_of[classes[x]['id']] = gi + 1; column[classes[x]['id']] = c
    status = {(r['class_id'], r['channel']): r['status'] for r in rows}
    status.update({(r['class_id'], r['channel']): 'unscored' for r in unscored})
    post = {(m['class_id'], m['channel'], m['unit_id']): m for m in mols}
    comp = {(g, ch, uid): (pb, po, pa, ll) for g, ch, uid, pb, po, pa, ll in components}
    units = []
    for s in sources:
        for u in s['units']:
            ch = Un.channel_of(s, u)
            name = u.get('read_name') or u['unit_id']
            fold, disc = read_fold(name, spec)
            units.append((name, s['dataset_id'], ch.split('::', 1)[1], u['unit_id'], ch, fold, disc))
    units.sort()
    order = sorted(classes, key=lambda g: (group_of[g['id']], g['span'][0], g['span'][1], g['id']))
    for name, ds, strand, uid, ch, fold, disc in units:
        for g in order:
            gi = group_of[g['id']]; m = post.get((g['id'], ch, uid)); c = comp.get((gi, ch, uid))
            scored = m is not None and c is not None
            yield dict(read_name=name, fold=fold, discovery=disc, dataset=ds, strand=strand, unit_id=uid, group=gi,
                       class_id=g['id'], class_start=_r(g['span'][0], 1), class_end=_r(g['span'][1], 1),
                       class_status=status.get((g['id'], ch), 'unscored'), scored=scored,
                       posterior=m['posterior'] if scored else '', log_bf=m['log_bf'] if scored else '',
                       label=m['label'] if scored else '',
                       p_broader=_r(c[0]) if scored else '', p_other_shape=_r(c[1]) if scored else '',
                       p_accessible=_r(c[2]) if scored else '',
                       loglik=_r(c[3][column[g['id']]], 3) if scored else '', ll_broader=_r(c[3][-3], 3) if scored else '',
                       ll_other_shape=_r(c[3][-2], 3) if scored else '', ll_accessible=_r(c[3][-1], 3) if scored else '')


def _cell(v):
    if isinstance(v, bool):
        return 'True' if v else 'False'
    return '' if v is None else str(v)


def write_window_table(path, rows):
    """Write one window's read table (gzip, mtime 0: identical runs give identical bytes)."""
    with open(path, 'wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as gz, \
            io.TextIOWrapper(gz, encoding='utf-8', newline='') as handle:
        w = csv.writer(handle, delimiter='\t', lineterminator='\n')
        w.writerow(READ_FIELDS)
        n = 0
        for row in rows:
            w.writerow([_cell(row.get(f)) for f in READ_FIELDS]); n += 1
    return n


# ----------------------------------------------------------------------------------------------- role rules
def _condition(name, value):
    if isinstance(value, (list, tuple)):
        if len(value) != 2 or any(v is not None and not isinstance(v, (int, float)) for v in value):
            raise ValueError(f'role rule {name}: expected [min, max] with numbers or null')
        lo, hi = value
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f'role rule {name}: min {lo} is greater than max {hi}')
        tests = ([('ge', lo)] if lo is not None else []) + ([('le', hi)] if hi is not None else [])
    elif isinstance(value, dict):
        bad = sorted(set(value) - set(OPS))
        if bad or not value:
            raise ValueError(f'role rule {name}: use keys gt, ge, lt, le (got {", ".join(bad) or "none"})')
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in value.values()):
            raise ValueError(f'role rule {name}: bounds must be numbers')
        tests = sorted(value.items())
    else:
        raise ValueError(f'role rule {name}: expected [min, max] or {{"ge": ..., "lt": ...}}')
    return tests


def parse_role_rules(value):
    """A rule document (dict, JSON text or file path) as a normalised list of rules; raises ValueError when invalid."""
    if isinstance(value, (str, Path)):
        text = str(value)
        doc = json.loads(text) if text.lstrip().startswith('{') else json.loads(Path(text).read_text())
    else:
        doc = value
    if not isinstance(doc, dict) or not isinstance(doc.get('rules'), list):
        raise ValueError('role rules: expected a JSON object with a "rules" list')
    if doc.get('schema', ROLE_SCHEMA) != ROLE_SCHEMA:
        raise ValueError(f'role rules: schema must be {ROLE_SCHEMA}')
    unknown = sorted(set(doc) - {'schema', 'rules', 'description'})
    if unknown:
        raise ValueError(f'role rules: unknown key(s) {", ".join(unknown)}')
    out = []
    for i, rule in enumerate(doc['rules'], 1):
        if not isinstance(rule, dict) or not isinstance(rule.get('role'), str) or not rule['role'].strip():
            raise ValueError(f'role rule {i}: needs a non-empty "role" name')
        bad = sorted(set(rule) - {'role', 'component', 'group_roles', 'description', *GEOMETRY})
        if bad:
            raise ValueError(f'role rule {i} ({rule["role"]}): unknown key(s) {", ".join(bad)}')
        component = rule.get('component', 'class')
        if component not in ('class', 'broader'):
            raise ValueError(f'role rule {i} ({rule["role"]}): component must be "class" or "broader"')
        group_roles = rule.get('group_roles')
        if group_roles is not None and (component != 'broader' or not isinstance(group_roles, list)
                                        or not all(isinstance(r, str) for r in group_roles)):
            raise ValueError(f'role rule {i} ({rule["role"]}): group_roles is a list of role names, for broader rules only')
        out.append(dict(role=rule['role'].strip(), component=component, group_roles=group_roles,
                        tests={k: _condition(f'{i} ({rule["role"]}).{k}', rule[k]) for k in GEOMETRY if k in rule}))
    return out


def rules_digest(rules):
    return hashlib.sha256(json.dumps(rules, sort_keys=True).encode()).hexdigest()


def role_names(rules):
    """Role names in rule order, each once."""
    seen = []
    for r in rules:
        if r['role'] not in seen:
            seen.append(r['role'])
    return seen


def relative(a, b, anchor, strand):
    """[a, b) in reference coordinates as (start, end) relative to the anchor in transcription orientation."""
    return (a - anchor, b - anchor) if strand != '-' else (anchor - b, anchor - a)


def _matches(rule, start, end):
    values = dict(start=start, end=end, center=(start + end)/2, width=end - start)
    return all(OPS[op](values[k], t) for k, tests in rule['tests'].items() for op, t in tests)


def assign_roles(rules, classes, anchor, strand):
    """Per class id: (relative start, end, role); per group: broader role. classes: [{id, start, end, group}]."""
    out = {}
    for c in classes:
        s, e = relative(c['start'], c['end'], anchor, strand)
        role = next((r['role'] for r in rules if r['component'] == 'class' and _matches(r, s, e)), '')
        out[c['id']] = (s, e, role)
    groups = {}
    for c in classes:
        groups.setdefault(c['group'], []).append(c)
    broader = {}
    for g, members in groups.items():
        s = min(out[c['id']][0] for c in members); e = max(out[c['id']][1] for c in members)
        roles = {out[c['id']][2] for c in members}
        broader[g] = next((r['role'] for r in rules if r['component'] == 'broader' and _matches(r, s, e)
                           and (r['group_roles'] is None or roles & set(r['group_roles']))), '')
    return out, broader


def window_anchor(window, offset):
    """The anchor (TSS) of a BED window: its 5' end in transcription orientation plus ``offset`` bp."""
    return window['start'] + offset if window.get('strand', '+') != '-' else window['end'] - offset


# ------------------------------------------------------------------------------------------------ aggregation
def window_classes(folder):
    """(classes, max prevalence per class) of one window from its classes.tsv (small; result.json.gz holds every
    molecule and is not read): [{id, start, end, status, supported_channels}] in catalog order."""
    classes, prevalence = {}, {}
    path = Path(folder)/'classes.tsv'
    if not path.is_file():
        return [], {}
    with open(path, newline='') as handle:
        for r in csv.DictReader(handle, delimiter='\t'):
            c = classes.setdefault(r['class_id'], dict(id=r['class_id'], start=float(r['start']), end=float(r['end']),
                                                        scored=False, supported_channels=0))
            if r['status'] != 'unscored':
                c['scored'] = True
                prevalence[r['class_id']] = max(prevalence.get(r['class_id'], 0.), float(r.get('prevalence') or 0.))
            c['supported_channels'] += r.get('supported') == 'True'
    out = []
    for c in classes.values():
        status = 'supported' if c['supported_channels'] else 'unsupported' if c.pop('scored') else 'unscored'
        out.append(dict(c, status=status))
    return out, prevalence


def aggregate(out, specs, *, rules=None, anchor_offset=0, progress=None):
    """Stream every unit's read table into top-level read_posteriors.tsv.gz, class_roles.tsv and read_roles.tsv.gz.

    Memory holds one window's rows at a time. Returns a summary dict (rows, windows, rules digest)."""
    rules = rules or []
    names = role_names(rules)
    out = Path(out)
    tmp = {n: out/(n + '.partial') for n in (READ_TABLE, CLASS_ROLES, READ_ROLES)}
    total = windows = 0
    with open(tmp[READ_TABLE], 'wb') as r1, gzip.GzipFile(fileobj=r1, mode='wb', mtime=0, filename='') as g1, \
            io.TextIOWrapper(g1, encoding='utf-8', newline='') as reads_handle, \
            open(tmp[CLASS_ROLES], 'w', newline='') as class_handle, \
            open(tmp[READ_ROLES], 'wb') as r3, gzip.GzipFile(fileobj=r3, mode='wb', mtime=0, filename='') as g3, \
            io.TextIOWrapper(g3, encoding='utf-8', newline='') as roles_handle:
        reads_w = csv.writer(reads_handle, delimiter='\t', lineterminator='\n')
        class_w = csv.writer(class_handle, delimiter='\t', lineterminator='\n')
        roles_w = csv.writer(roles_handle, delimiter='\t', lineterminator='\n')
        reads_w.writerow(WINDOW_FIELDS + READ_FIELDS + REL_FIELDS)
        class_w.writerow(CLASS_ROLE_FIELDS)
        role_cols = [f'p_{n}' for n in names]
        roles_w.writerow(WINDOW_FIELDS + ['read_name', 'fold', 'discovery', 'dataset', 'strand', 'groups_scored'] + role_cols)
        for s in specs:
            folder = Path(s['folder']); table = folder/READ_TABLE
            w = s['window']; anchor = window_anchor(w, anchor_offset); strand = w.get('strand', '+')
            win = [s['index'] + 1, w['name'], w['chrom'], w['start'], w['end'], strand, anchor]
            if not table.is_file():
                continue
            windows += 1
            classes, prevalence = window_classes(folder)
            from .lattice_recaller import discovery as D
            geo = [dict(id=c['id'], span=(c['start'], c['end'])) for c in classes]
            group_of = {geo[x]['id']: gi + 1 for gi, grp in enumerate(D.overlap_groups(geo)) for x in grp}
            cl = [dict(id=c['id'], start=float(c['start']), end=float(c['end']), group=group_of[c['id']]) for c in classes]
            roles, broader = assign_roles(rules, cl, anchor, strand)
            for c in sorted(cl, key=lambda c: (c['group'], c['start'], c['end'], c['id'])):
                rs, re_, role = roles[c['id']]
                full = next(x for x in classes if x['id'] == c['id'])
                class_w.writerow(win + [c['id'], c['group'], c['start'], c['end'], round(rs, 1), round(re_, 1),
                                        round((rs + re_)/2, 1), round(re_ - rs, 1), role, broader.get(c['group'], ''),
                                        full.get('status', ''), full.get('supported_channels', ''),
                                        round(prevalence[c['id']], 4) if c['id'] in prevalence else ''])
            current, acc = None, None

            def flush():
                if current is None:
                    return
                cells = []
                for n in names:
                    vals = [v for (g, role), v in acc['roles'].items() if role == n]
                    cells.append(round(max(vals), 4) if vals else '')
                roles_w.writerow(win + list(current) + [len(acc['groups'])] + cells)

            with gzip.open(table, 'rt', newline='') as handle:
                reader = csv.DictReader(handle, delimiter='\t')
                for row in reader:
                    cid = row['class_id']; g = int(row['group'])
                    rs, re_, role = roles.get(cid, (None, None, ''))
                    reads_w.writerow(win + [row[f] for f in READ_FIELDS] +
                                     ['' if rs is None else round(rs, 1), '' if re_ is None else round(re_, 1),
                                      '' if rs is None else round(re_ - rs, 1), role, broader.get(g, '')])
                    total += 1
                    key = (row['read_name'], row['fold'], row['discovery'], row['dataset'], row['strand'])
                    if key != current:
                        flush(); current = key; acc = dict(groups=set(), roles={}, broader_done=set())
                    if row['scored'] != 'True':
                        continue
                    acc['groups'].add(g)
                    # Classes of one group are mutually exclusive: their posteriors add up within the group;
                    # across groups a role's value is the largest group total (the best-supported instance).
                    if role:
                        acc['roles'][(g, role)] = acc['roles'].get((g, role), 0.) + float(row['posterior'])
                    b = broader.get(g, '')
                    if b and g not in acc['broader_done']:
                        acc['broader_done'].add(g)
                        acc['roles'][(g, b)] = acc['roles'].get((g, b), 0.) + float(row['p_broader'])
            flush()
            if progress is not None:
                progress.report('read_table', f"{w['name']}: read table merged", completed=windows, total=len(specs))
    for name, path in tmp.items():
        path.replace(out/name)
    return dict(rows=total, windows=windows, roles=names, rules_sha256=rules_digest(rules) if rules else None,
                anchor_offset_bp=anchor_offset)
