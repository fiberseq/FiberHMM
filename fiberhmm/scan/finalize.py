"""Finish a scan: reconcile seams, stitch, write tables, write annotated BAMs and the manifest.

Everything here is a pure function of the unit results in ``work/`` and the
input BAMs, so a rerun (or a continued run) writes byte-identical outputs.
"""
from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import identity as I
from .seams import reconcile_anchor, territory_runs
from .tables import TableWriter
from .types import ScanUnit, molecule_result_from_json, unit_result_from_json

STATE_CODES = {'1': 1, '0': 2, '.': 3}
COMMON = ['chrom', 'start', 'end', 'idx_start', 'idx_end']
HIT_COLUMNS = ['hit_id', 'type', 'anchor', 'strand', 'unit_id', 'score', 'label', 'flags', 'parent_id']
VALUE_COLUMNS = ['hit_id', 'group', 'group_kind', 'value_name', 'value', 'lo', 'hi', 'n_molecules', 'n_callable']
STATE_COLUMNS = ['hit_id', 'dataset_id', 'qname', 'record', 'molecule', 'state']
MARK_COLUMNS = ['group', 'dataset_id', 'qname', 'record', 'molecule', 'mol_start', 'mol_len', 'q', 'name']
MOLECULE_COLUMNS = ['dataset_id', 'qname', 'record', 'molecule']
EVIDENCE_COLUMNS = ['evidence_unit', 'dataset_id', 'qname', 'record', 'molecule']
UNIT_COLUMNS = ['unit_id', 'zone_start', 'zone_end', 'mode', 'status', 'records']
CALLABLE_COLUMNS = ['dataset_id', 'min_molecules']


def _read_json(path: Path):
    import gzip
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        return json.load(handle)


def _unit_dir(out: Path, module: str, unit: ScanUnit) -> Path:
    return out / 'work' / module / I.unit_dir_name(unit.unit_id)


class _Annotations:
    """Per contig, per input file: record (sha, occ) -> MA entries to write."""

    def __init__(self):
        self.by_file: Dict[Tuple[str, int], Dict[Tuple[str, int], list]] = defaultdict(lambda: defaultdict(list))

    def add(self, rec, group: str, spec: str, ref: Tuple[int, int], mol, q, name: str) -> None:
        self.by_file[(rec.dataset_id, rec.input_index)][(rec.record_sha256, rec.occurrence)].append(
            (group, spec, int(ref[0]), int(ref[1]), mol if (mol is None or mol == 'whole') else tuple(mol),
             tuple(int(x) for x in q), name))

    def clear(self):
        self.by_file.clear()


def finalize(out: Path, manifest: dict, datasets, reference: Mapping[str, int], units: Sequence[ScanUnit],
             mods, unit_keys, annotate: bool, progress, *, command=None, calibrations=None) -> dict:
    from .units import UnitIndex
    index = UnitIndex(units)
    order = {c: i for i, c in enumerate(reference)}
    contigs_in_plan = sorted({u.chrom for u in units}, key=lambda c: order[c])
    units_by_chrom = defaultdict(list)
    for u in units:
        units_by_chrom[u.chrom].append(u)
    for c in units_by_chrom:
        units_by_chrom[c].sort(key=lambda u: (u.zone_start, u.zone_end))

    # unit status and survey
    surveys, status = {}, {}
    for u in units:
        try:
            surveys[u.unit_id] = _read_json(out / 'work' / '_unit' / I.unit_dir_name(u.unit_id) / 'survey.json')
        except (OSError, ValueError):
            surveys[u.unit_id] = None
        if surveys[u.unit_id] is None:
            status[u.unit_id] = 'planned'
        elif surveys[u.unit_id]['records'] == 0:
            status[u.unit_id] = 'empty'
        else:
            ok = all((_unit_dir(out, n, u) / 'marker.json').is_file() for n, _m, _s in mods)
            status[u.unit_id] = 'done' if ok else 'failed'

    tables: Dict[str, dict] = {}
    unit_writer = TableWriter(out / 'units.tsv.bgz', COMMON + UNIT_COLUMNS, order)
    for u in units:
        s = surveys[u.unit_id]
        unit_writer.add(u.chrom, u.start, u.end, [u.unit_id, u.zone_start, u.zone_end, u.mode, status[u.unit_id],
                                                  '' if s is None or s['records'] < 0 else s['records']])
    tables['units.tsv.bgz'] = unit_writer.close()
    min_mol = int(manifest['read_policy']['callable_min_molecules'])
    cw = TableWriter(out / 'callable.tsv.bgz', COMMON + CALLABLE_COLUMNS, order)
    for chrom in contigs_in_plan:
        for ds in datasets:
            runs: List[List[int]] = []
            for u in units_by_chrom[chrom]:
                s = surveys[u.unit_id]
                if not s:
                    continue
                for a, b in (s.get('callable') or {}).get(ds.dataset_id, []):
                    if runs and runs[-1][1] == a:
                        runs[-1][1] = b
                    else:
                        runs.append([a, b])
            for a, b in runs:
                cw.add(chrom, a, b, [ds.dataset_id, min_mol])
    tables['callable.tsv.bgz'] = cw.close()

    tsv_mode = not annotate
    writers: Dict[str, Dict[str, TableWriter]] = {}
    decls: Dict[str, dict] = {}
    seam_components: Dict[str, int] = {}
    open_ends: Dict[str, int] = {}
    seams_open = 0
    for name, mod, settings in mods:
        hit_cols = mod.hit_columns()
        w = dict(hits=TableWriter(out / name / 'hits.tsv.bgz', COMMON + HIT_COLUMNS + hit_cols, order),
                 values=TableWriter(out / name / 'values.tsv.bgz', COMMON + VALUE_COLUMNS, order))
        for t in mod.native_tables:
            w['native:' + t] = None          # created lazily (columns from the first rows)
        if tsv_mode:
            w['states'] = TableWriter(out / name / 'states.tsv.bgz', COMMON + STATE_COLUMNS + mod.state_columns(), order)
            w['marks'] = TableWriter(out / name / 'marks.tsv.bgz', COMMON + MARK_COLUMNS, order)
            if mod.scalar_names():
                w['molecules'] = TableWriter(out / name / 'molecules.tsv.bgz',
                                             COMMON + MOLECULE_COLUMNS + mod.scalar_names(), order)
            w['units'] = TableWriter(out / name / 'units.tsv.bgz', COMMON + EVIDENCE_COLUMNS, order)
        writers[name] = w
        decls[name] = {g.group: g for g in mod.ma_groups()}
        seam_components[name] = 0
        if mod.reconcile == 'stitch':
            open_ends[name] = 0
    native_rows: Dict[str, Dict[str, list]] = {n: defaultdict(list) for n, _m, _s in mods}
    seen_evidence: Dict[str, set] = {n: set() for n, _m, _s in mods}
    native_cols: Dict[str, Dict[str, List[str]]] = {n: {} for n, _m, _s in mods}

    ann = _Annotations() if annotate else None
    annotated_outputs = []
    bam_writers = _open_annotated(out, manifest, datasets, mods, decls, command) if annotate else None
    full_copy = bool(annotate and manifest['units'].get('contigs'))
    loop = list(reference) if full_copy else contigs_in_plan
    try:
        for chrom in loop:
            cunits = units_by_chrom.get(chrom, [])
            for name, mod, settings in (mods if cunits else ()):
                sh = I.settings_hash(settings)
                w = writers[name]
                results = {}
                for u in cunits:
                    if status[u.unit_id] == 'done':
                        results[u.unit_id] = unit_result_from_json(_read_json(_unit_dir(out, name, u) / 'result.json.gz'))
                keymap: Dict[Tuple[str, str], str] = {}
                if mod.reconcile == 'anchor':
                    missing = [u.unit_id for u in cunits if status[u.unit_id] in ('failed', 'planned')]
                    kept, comps, open_s = reconcile_anchor(
                        [(u, results[u.unit_id].hits) for u in cunits if u.unit_id in results], missing=missing)
                    seam_components[name] += comps
                    seams_open += open_s
                    for u, h in kept:
                        keymap[(u.unit_id, h.key)] = I.hit_id(name, h.type, h.chrom, h.anchor, h.start, h.end,
                                                              h.key, u.unit_id, mod.version, sh)
                    for u, h in kept:
                        _write_hit(w['hits'], h, keymap[(u.unit_id, h.key)], u.unit_id,
                                   keymap.get((u.unit_id, h.parent)) if h.parent else '', mod.hit_columns())
                    for u in cunits:
                        r = results.get(u.unit_id)
                        if r is None:
                            continue
                        hspan = {h.key: (h.chrom, h.start, h.end) for h in r.hits}
                        for v in r.values:
                            hid = keymap.get((u.unit_id, v.hit_key))
                            if hid is not None:
                                _write_value(w['values'], hspan[v.hit_key], hid, v)
                        for table, rows in r.native.items():
                            for row in rows:
                                hid = keymap.get((u.unit_id, row.get('hit_key')))
                                if hid is not None:
                                    native_rows[name][table].append((hspan[row['hit_key']], hid, row))
                                    native_cols[name].setdefault(table, [k for k in row if k != 'hit_key'])
                        _emit_states(name, mod, decls[name], r, u, keymap, w, ann, tsv_mode)
                        if tsv_mode:
                            # overlapping units load the same records into the same evidence units: one row each
                            for e in r.evidence:
                                rec = e.record
                                ekey = (e.evidence_unit, rec.key)
                                if ekey in seen_evidence[name]:
                                    continue
                                seen_evidence[name].add(ekey)
                                w['units'].add(rec.chrom, rec.ref_start, rec.ref_end,
                                               [e.evidence_unit, rec.dataset_id, rec.qname, rec.record, rec.molecule])
                else:
                    zones = []
                    for u in cunits:
                        st = status[u.unit_id]
                        if st == 'done' and results[u.unit_id].status == 'ok':
                            tracks = _load_tracks(_unit_dir(out, name, u))
                            zones.append((u, tracks, 'ok' if tracks is not None else 'empty'))
                        elif st == 'empty' or (st == 'done'):
                            zones.append((u, None, 'empty'))
                        else:
                            zones.append((u, None, 'missing'))
                    runs = territory_runs(zones, mod.stitch_support_bp(settings))
                    territory = [(s, e, lo, ro) for s, e, _a, lo, ro in runs]
                    names = sorted({k for _s, _e, arrays, _l, _r in runs for k in arrays})
                    tracks = {k: [arrays.get(k) for _s, _e, arrays, _l, _r in runs] for k in names}
                    stitched = mod.stitch(chrom, territory, tracks, settings, (calibrations or {}).get(name))
                    skey = {}
                    for h in stitched.hits:
                        skey[h.key] = I.hit_id(name, h.type, h.chrom, h.anchor, h.start, h.end, h.key, 'stitch',
                                               mod.version, sh)
                        if 'open_end' in h.flags:
                            open_ends[name] += 1
                    spans = {h.key: (h.start, h.end) for h in stitched.hits}
                    for h in stitched.hits:
                        _write_hit(w['hits'], h, skey[h.key], 'stitch', skey.get(h.parent, '') if h.parent else '',
                                   mod.hit_columns())
                    for v in stitched.values:
                        if v.hit_key in skey:
                            _write_value(w['values'], (chrom,) + spans[v.hit_key], skey[v.hit_key], v)
                    for table, rows in stitched.native.items():
                        for row in rows:
                            if row.get('hit_key') in skey:
                                native_rows[name][table].append(((chrom,) + spans[row['hit_key']], skey[row['hit_key']], row))
                                native_cols[name].setdefault(table, [k for k in row if k != 'hit_key'])
                # molecule-wide outputs: each record's result from the unit that owns it under this plan
                for u in cunits:
                    if status[u.unit_id] != 'done':
                        continue
                    p = _unit_dir(out, name, u) / 'molecules.json.gz'
                    if not p.is_file():
                        continue
                    for item in _read_json(p):
                        mr = molecule_result_from_json(item)
                        rec = mr.record
                        owner = index.owner_of_record(rec.chrom, rec.ref_start, max(rec.ref_end, rec.ref_start + 1))
                        if owner is None or owner.unit_id != u.unit_id:
                            continue
                        _emit_molecule(name, mod, decls[name], mr, w, ann, tsv_mode)
            if annotate:
                _write_annotated_contig(bam_writers, chrom, ann, cunits, full_copy)
                ann.clear()
        if annotate:
            annotated_outputs = _close_annotated(out, bam_writers, full_copy)
            bam_writers = None
    finally:
        if bam_writers is not None:
            _abort_annotated(bam_writers)

    for name, mod, settings in mods:
        w = writers[name]
        for key, writer in w.items():
            if writer is None:
                continue
            rel = f'{name}/{writer.path.name}'
            tables[rel] = writer.close()
        for table in mod.native_tables:
            cols = native_cols[name].get(table) or []
            nw = TableWriter(out / name / f'{table}.tsv.bgz', COMMON + ['hit_id'] + cols, order)
            for (chrom, s, e), hid, row in native_rows[name][table]:
                nw.add(chrom, s, e, [hid] + [row.get(c, '') for c in cols])
            tables[f'{name}/{table}.tsv.bgz'] = nw.close()

    done = sum(1 for v in status.values() if v == 'done')
    empty = sum(1 for v in status.values() if v == 'empty')
    manifest = dict(manifest)
    manifest['outputs'] = dict(annotate=bool(annotate), annotated=annotated_outputs,
                               tables={k: tables[k] for k in sorted(tables)})
    manifest['status'] = dict(units_total=len(units), units_done=done, units_empty=empty,
                              units_failed=sorted(k for k, v in status.items() if v in ('failed', 'planned')),
                              seams_open=seams_open, seam_components=seam_components, open_ends=open_ends)
    text = json.dumps(manifest, sort_keys=True, indent=1, default=I._json_default) + '\n'
    tmp = out / '.manifest.json.tmp'
    tmp.write_text(text)
    os.replace(tmp, out / 'manifest.json')
    progress.report('scan', f"complete: {done} units computed, {empty} empty", completed=len(units), total=len(units))
    return dict(status='complete', output=str(out), scan_id=manifest['scan_id'], units=len(units), done=done,
                empty=empty, failed=manifest['status']['units_failed'],
                tables={k: v['rows'] for k, v in manifest['outputs']['tables'].items()},
                annotated=[a['path'] for a in annotated_outputs])


# ------------------------------------------------------------------ table rows
def _write_hit(writer, h, hid, unit_id, parent_id, columns):
    flags = list(h.flags)
    if h.zero_width and 'zero_width' not in flags:
        flags.append('zero_width')
    writer.add(h.chrom, h.start, h.end, [hid, h.type, h.anchor, h.strand, unit_id, h.score, h.label,
                                         ','.join(sorted(flags)), parent_id] + [h.columns.get(c) for c in columns])


def _write_value(writer, span, hid, v):
    chrom, s, e = span
    writer.add(chrom, s, e, [hid, v.group, v.group_kind, v.value_name, v.value, v.lo, v.hi, v.n_molecules,
                             v.n_callable])


def _load_tracks(folder: Path):
    import numpy as np
    p = folder / 'tracks.npz'
    if not p.is_file():
        return None
    with np.load(p) as data:
        return {k: data[k] for k in data.files}


def _emit_states(name, mod, groups, result, unit, keymap, w, ann, tsv_mode):
    spans = {h.key: (h.chrom, h.start, h.end, h.type) for h in result.hits}
    for s in result.states:
        hid = keymap.get((unit.unit_id, s.hit_key))
        if hid is None:
            continue
        chrom, hs, he, htype = spans[s.hit_key]
        a, b = s.interval if s.interval is not None else (hs, he)
        rec = s.record
        if tsv_mode:
            w['states'].add(chrom, a, b, [hid, rec.dataset_id, rec.qname, rec.record, rec.molecule, s.state]
                            + [s.columns.get(c) for c in mod.state_columns()])
        else:
            group = mod.state_groups.get(htype)
            if group is None:
                continue
            decl = groups[group]
            ann.add(rec, group, decl.spec, (a, b), None, [STATE_CODES[s.state], *s.q], hid)


def _emit_molecule(name, mod, groups, mr, w, ann, tsv_mode):
    from fiberhmm.io.scan_ma import encode_scalars
    rec = mr.record
    scalar_groups = [g for g in groups.values() if g.kind == 'scalar']
    if tsv_mode:
        if mr.scalars and 'molecules' in w:
            w['molecules'].add(rec.chrom, rec.ref_start, rec.ref_end,
                               [rec.dataset_id, rec.qname, rec.record, rec.molecule]
                               + [mr.scalars.get(n) for n in mod.scalar_names()])
        for m in mr.marks:
            w['marks'].add(m.chrom, m.start, m.end, [m.group, rec.dataset_id, rec.qname, rec.record, rec.molecule,
                                                     m.mol[0] if m.mol else '', m.mol[1] if m.mol else '',
                                                     ','.join(str(int(x)) for x in m.q), m.name])
        return
    for g in scalar_groups:
        if mr.scalars:
            ann.add(rec, g.group, g.spec, (rec.ref_start, rec.ref_end), 'whole',
                    encode_scalars(mr.scalars, g.codecs), g.group)
    for m in mr.marks:
        decl = groups[m.group]
        ann.add(rec, m.group, decl.spec, (m.start, m.end), m.mol, m.q, m.name or m.group)


# ------------------------------------------------------------------ annotated BAMs
def _annotated_name(ds, input_index: int, path: str) -> str:
    stem = Path(path).name
    if stem.endswith('.bam'):
        stem = stem[:-4]
    return f'{ds.ordinal:03d}_{input_index}_{I.safe_name(stem)}.scan.bam'


def _open_annotated(out: Path, manifest, datasets, mods, decls, command):
    import pysam
    from fiberhmm.io.bam_header import append_coord_marker, append_ma_types, append_pg_record
    from fiberhmm.io.scan_ma import scan_contract_line, with_scan_contracts
    target_dir = out / 'annotated'
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    groups = []
    lines = []
    for name, mod, settings in mods:
        for g in mod.ma_groups():
            groups.append(g.group)
            lines.append(scan_contract_line(g.to_json(), module=name, module_version=mod.version,
                                            scan_id=manifest['scan_id'], settings_hash=I.settings_hash(settings)))
    handles = {}
    for ds in datasets:
        calls = next((d['fiberhmm_calls'] for d in manifest['datasets'] if d['dataset_id'] == ds.dataset_id), {})
        for i, path in enumerate(ds.paths):
            src = pysam.AlignmentFile(path, 'rb')
            from fiberhmm.io.annotation_frame import ma_annotation_frame
            input_molecular = ma_annotation_frame(src.header) == 'molecular'
            header = src.header
            if groups:
                header = append_ma_types(header, groups)
            header = append_coord_marker(header)
            hd = header.to_dict()
            hd = with_scan_contracts(hd, [l[:-1] + ',"calls_program":' + json.dumps(calls.get('id', '')) + '}'
                                          for l in lines])
            header = pysam.AlignmentHeader.from_dict(hd)
            header = append_pg_record(header, dict(PN='fiberhmm-scan', VN=manifest['fiberhmm']['version'],
                                                   DS=f"scan_id={manifest['scan_id']} groups={','.join(groups)}",
                                                   CL=' '.join(command) if command else 'fiberhmm-scan'))
            name = _annotated_name(ds, i, path)
            tmp = target_dir / ('.' + name + '.tmp')
            dest = pysam.AlignmentFile(str(tmp), 'wb', header=header)
            handles[(ds.dataset_id, i)] = dict(src=src, dest=dest, tmp=tmp, final=target_dir / name, dataset=ds,
                                               owned={g.group: g.spec for _n, m, _s in mods for g in m.ma_groups()},
                                               input_index=i, source=path, written=0, marked=0, done_contigs=set(),
                                               input_molecular=input_molecular, written_keys=set())
    return handles


def _annotate_read(read, entries, input_molecular, owned):
    """Write this scan's groups on one record. ``owned``: group -> spec of every group the scan writes;
    each is replaced, so a record with no new entry loses an older scan's marks of the same group."""
    from fiberhmm.io.frame_carry import carry_to_molecular
    from fiberhmm.io.scan_ma import ma_read_length, project_to_molecule, replace_scan_groups
    if not input_molecular:
        carry_to_molecular(read)
    stale = False
    if owned and read.has_tag('MA'):
        ma = str(read.get_tag('MA'))
        stale = any(f';{g}.' in ma for g in owned)
    if not entries:
        if stale:
            replace_scan_groups(read, {g: (spec, []) for g, spec in owned.items()})
        return 0
    length = ma_read_length(read)
    groups: Dict[str, Tuple[str, list]] = {g: (spec, []) for g, spec in owned.items()}
    for group, spec, a, b, mol, q, name in entries:
        if mol == 'whole':
            iv = (0, length)
        elif mol is not None:
            iv = mol
        else:
            iv = project_to_molecule(read, a, b, length)
        if iv is None:
            continue
        groups.setdefault(group, (spec, []))[1].append((iv[0], iv[1], q, name))
    if not any(items for _spec, items in groups.values()) and not stale:
        return 0
    return replace_scan_groups(read, groups)


def _scope_spans(units_by_chrom, chrom):
    return sorted((u.start, u.end) for u in units_by_chrom.get(chrom, ()))


def _write_annotated_contig(handles, chrom, ann, cunits, full_copy):
    """Write one contig's records to every annotated BAM (all of them in a genome scan, else the
    records overlapping the units), adding the planned scan groups."""
    from .identity import record_sha256
    for key, h in handles.items():
        src, dest = h['src'], h['dest']
        if chrom not in src.references:
            continue
        planned = ann.by_file.get(key, {})
        seen: Dict[str, int] = {}
        it = src.fetch(chrom) if full_copy else _regions_iter(src, chrom, sorted((u.start, u.end) for u in cunits))
        for read in it:
            entries = None
            if planned:
                sha = record_sha256(read)
                occ = seen.get(sha, 0)
                seen[sha] = occ + 1
                entries = planned.get((sha, occ))
            if _annotate_read(read, entries, h['input_molecular'], h['owned']):
                h['marked'] += 1
            dest.write(read)
            h['written'] += 1


def _regions_iter(src, chrom, spans):
    """Records overlapping the merged spans once each (a long record belongs to its first span)."""
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    for i, (a, b) in enumerate(merged):
        for read in src.fetch(chrom, a, b):
            if any(read.reference_start < z and (read.reference_end or read.reference_start + 1) > y
                   for y, z in merged[:i]):
                continue
            yield read


def _close_annotated(out, handles, full_copy):
    import pysam
    outputs = []
    for key, h in handles.items():
        src, dest = h['src'], h['dest']
        if full_copy:
            for read in src.fetch('*'):          # unplaced unmapped records, last as in the input
                _annotate_read(read, None, h['input_molecular'], h['owned'])
                dest.write(read)
                h['written'] += 1
        dest.close()
        src.close()
        pysam.index('-c', str(h['tmp']))
        os.replace(h['tmp'], h['final'])
        os.replace(str(h['tmp']) + '.csi', str(h['final']) + '.csi')
        outputs.append(dict(dataset_id=h['dataset'].dataset_id, input_index=h['input_index'],
                            source=Path(h['source']).name, path=str(h['final'].relative_to(out)),
                            content_id=I.content_id(h['final']), records=h['written'], marked_records=h['marked'],
                            scope='full' if full_copy else 'units'))
    outputs.sort(key=lambda o: o['path'])
    return outputs


def _abort_annotated(handles):
    for h in handles.values():
        for k in ('dest', 'src'):
            try:
                h[k].close()
            except Exception:
                pass
        for p in (h['tmp'], Path(str(h['tmp']) + '.csi')):
            try:
                os.unlink(p)
            except OSError:
                pass
