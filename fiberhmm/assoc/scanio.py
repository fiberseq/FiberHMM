"""Reading association inputs from a ``fiberhmm-scan`` directory (scan contracts v1.9, sections 1, 4 and 7).

- **States** (M1, M2 anchors): ``<module>/states.tsv.bgz`` rows (``hit_id dataset_id input_index qname record
  molecule state``), folded to one state per (hit, physical molecule) by the HUB 3.4 rule.
- **Molecule covariates**: ``nfr/molecules.tsv.bgz`` (``openness``), ``nuc/molecules.tsv.bgz`` (``rho_read``,
  ``lambda_acc`` and the other nucleosome scalars), each folded per molecule (median over records).
- **Marks** (M3): ``nuc/marks.tsv.bgz`` (``nucseg`` segments and events, ``nucsz`` footprints with the TF bit).
- **Hits and territory** (M4): ``<module>/hits.tsv.bgz`` and ``callable.tsv.bgz`` within the computed zones.

Per-molecule tables exist only for scans written in table mode (``--no-annotate``; the default for BED-scoped
scans). Annotated scans keep per-molecule results in MA tags; M1-M3 refuse them with that message (reading the
annotated BAMs is not implemented). M4 reads hits and territory, which every scan writes.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

STATE_CODES = {'1': 1, '0': 0}


class ScanInputError(ValueError):
    pass


def open_scan(path):
    from fiberhmm.scan.reader import ScanDir
    return ScanDir(path)


def _rows(scan, rel: str, regions=None) -> Iterable[dict]:
    if not scan.has(rel):
        return []
    t = scan.table(rel)
    if regions is None:
        return t.rows()
    return (r for chrom, s, e in regions for r in t.fetch(chrom, s, e))


def require_tables(scan, rels: Sequence[str], mode: str):
    missing = [r for r in rels if not scan.has(r)]
    if missing:
        hint = ''
        if (scan.manifest.get('outputs') or {}).get('annotate'):
            hint = (' This scan wrote per-molecule results as MA tags in annotated BAMs; rerun fiberhmm-scan with '
                    '--no-annotate to write the states, marks and molecules tables that association reads.')
        raise ScanInputError(f'{mode} needs {", ".join(missing)}.{hint}')


def module_of(type_id: str) -> str:
    return type_id.split('.', 1)[0]


class MoleculeIndex:
    """Physical molecules seen in a scan, with their records' datasets and QNAMEs."""

    def __init__(self):
        self.index: Dict[str, int] = {}
        self.dataset: List[str] = []
        self.qnames: List[set] = []

    def add(self, molecule: str, dataset: str, qname: str) -> int:
        i = self.index.get(molecule)
        if i is None:
            i = self.index[molecule] = len(self.dataset)
            self.dataset.append(dataset)
            self.qnames.append(set())
        self.qnames[i].add(qname)
        return i

    @property
    def n(self) -> int:
        return len(self.dataset)

    @property
    def ids(self) -> np.ndarray:
        out = np.empty(self.n, dtype=object)
        for m, i in self.index.items():
            out[i] = m
        return out


def read_states(scan, type_id: str, mi: MoleculeIndex, regions=None, footprint_like: Optional[bool] = None):
    """FeatureStates of one hit type from its module's states table (records folded per physical molecule)."""
    from fiberhmm.assoc.cooccur import FeatureStates
    mod = module_of(type_id)
    require_tables(scan, [f'{mod}/states.tsv.bgz', f'{mod}/hits.tsv.bgz'], 'this mode')
    hits = [h for h in _rows(scan, f'{mod}/hits.tsv.bgz', regions) if h['type'] == type_id]
    if not hits:
        raise ScanInputError(f'no {type_id} hits in {mod}/hits.tsv.bgz')
    hidx = {h['hit_id']: i for i, h in enumerate(hits)}
    H, M, S = [], [], []
    channel: Dict[int, str] = {}
    for r in _rows(scan, f'{mod}/states.tsv.bgz', regions):
        i = hidx.get(r['hit_id'])
        if i is None or r['state'] not in STATE_CODES:
            continue
        m = mi.add(r['molecule'], r['dataset_id'], r['qname'])
        H.append(i)
        M.append(m)
        S.append(STATE_CODES[r['state']])
        if r.get('channel'):
            channel[m] = r['channel']
    if footprint_like is None:
        footprint_like = mod in ('fp', 'nuc')
    fs = FeatureStates(type_id, [h['hit_id'] for h in hits], [h['chrom'] for h in hits],
                       [int(h['start']) for h in hits], [int(h['end']) for h in hits],
                       [int(h['anchor']) for h in hits], H, M, S, footprint_like=footprint_like)
    fs.channel_of = channel
    fs.hits_rows = hits
    return fs


def molecule_scalars(scan, module: str, columns: Sequence[str], mi: MoleculeIndex, regions=None) -> Dict[str, np.ndarray]:
    """Per-molecule scalars (median over a molecule's records; NaN when missing) for molecules in ``mi``."""
    rel = f'{module}/molecules.tsv.bgz'
    out = {c: np.full(mi.n, np.nan) for c in columns}
    if not scan.has(rel):
        return out
    acc = defaultdict(list)
    for r in _rows(scan, rel, regions):
        i = mi.index.get(r['molecule'])
        if i is None:
            continue
        for c in columns:
            v = r.get(c, '')
            if v not in ('', None):
                acc[(c, i)].append(float(v))
    for (c, i), vals in acc.items():
        out[c][i] = float(np.median(vals))
    return out


def haplotypes_from_bams(scan, mi: MoleculeIndex, tag: str = 'HP', regions=None) -> np.ndarray:
    """Per-molecule haplotype from the input BAMs' ``tag`` (0 = unphased; molecules whose records disagree are
    unphased)."""
    import pysam
    want = defaultdict(set)
    for i, qs in enumerate(mi.qnames):
        for q in qs:
            want[q].add(i)
    val: Dict[int, set] = defaultdict(set)
    for ds in scan.manifest.get('datasets', []):
        for f in ds.get('files', []):
            with pysam.AlignmentFile(f['path']) as bam:
                it = (bam.fetch(c, s, e) for c, s, e in regions) if regions else [bam.fetch(until_eof=True)]
                for chunk in it:
                    for r in chunk:
                        for i in want.get(r.query_name, ()):
                            val[i].add(r.get_tag(tag) if r.has_tag(tag) else 0)
    out = np.zeros(mi.n, dtype=object)
    for i in range(mi.n):
        v = val.get(i, {0})
        out[i] = next(iter(v)) if len(v) == 1 else 0
    return out


def molecule_table(scan, mi: MoleculeIndex, *, density: str = 'rho_read', haplotype_tag: Optional[str] = None,
                   channel_of: Optional[Dict[int, str]] = None, regions=None, labels_tsv: Optional[str] = None):
    """MoleculeTable with openness (nfr), density (nuc ``rho_read`` or ``lambda_acc``), channel, dataset and
    optional haplotype labels. ``labels_tsv``: extra labels, a TSV with a ``molecule`` column."""
    from fiberhmm.assoc.cooccur import MoleculeTable
    op = molecule_scalars(scan, 'nfr', ['openness'], mi, regions)['openness']
    dens = molecule_scalars(scan, 'nuc', [density], mi, regions)[density]
    labels = {'dataset': np.array(mi.dataset, dtype=object)}
    if haplotype_tag:
        labels['haplotype'] = haplotypes_from_bams(scan, mi, haplotype_tag, regions)
    if labels_tsv:
        import csv
        with open(labels_tsv) as fh:
            rows = list(csv.DictReader(fh, delimiter='\t'))
        for col in (rows[0].keys() if rows else []):
            if col == 'molecule':
                continue
            v = np.empty(mi.n, dtype=object)
            for r in rows:
                i = mi.index.get(r['molecule'])
                if i is not None:
                    v[i] = r[col]
            labels[col] = v
    ch = None
    if channel_of:
        ch = np.array([channel_of.get(i, 'pooled') for i in range(mi.n)], dtype=object)
    return MoleculeTable(mi.ids, channel=ch,
                         openness=op if np.isfinite(op).any() else None,
                         density=dens if np.isfinite(dens).any() else None, labels=labels)


def territory(scan, datasets: Optional[Sequence[str]] = None) -> List[Tuple[str, int, int]]:
    """Callable runs (``callable.tsv.bgz``) of the given datasets (intersection across them; default every
    dataset), clipped to the zones of computed units."""
    from fiberhmm.assoc.overlap import intersect_territories
    zones = [(u['chrom'], int(u['zone_start']), int(u['zone_end'])) for u in _rows(scan, 'units.tsv.bgz')
             if u.get('status') == 'done']
    by_ds = defaultdict(list)
    for r in _rows(scan, 'callable.tsv.bgz'):
        by_ds[r['dataset_id']].append((r['chrom'], int(r['start']), int(r['end'])))
    names = list(datasets) if datasets else sorted(by_ds)
    terr = [by_ds[d] for d in names if d in by_ds]
    if not terr:
        return []
    return intersect_territories(zones, *terr)


def read_hits(scan, type_id: str, regions=None):
    from fiberhmm.assoc.overlap import HitSet
    rows = [h for h in _rows(scan, f'{module_of(type_id)}/hits.tsv.bgz', regions) if h['type'] == type_id]
    if not rows:
        raise ScanInputError(f'no {type_id} hits')
    return HitSet.from_rows(rows)


NUCSEG_KIND = {'nuc.seg.regular': 1, 'nuc.seg.disordered': 2, 'nuc.seg.low_information': 3,
               'nuc.seg.undetermined': 4, 'nuc.event.undersized': 16, 'nuc.event.missing': 17}


def rate_data_from_marks(scan, a_type: str, b_type: str, regions=None, min_tf_bp: int = 0, mi=None):
    """RateData for M3 from ``nuc/marks.tsv.bgz`` (reference coordinates): A = ``nuc.seg.*``/``nuc.event.*`` rows;
    B = TF footprint centres (``nucsz`` rows with the TF bit, ``b_type='tf'``) or member call centres of a state
    type (``fp.class``: states rows with state 1, ``call_start``/``call_end``). Admissible bases: the record's
    aligned span (its ``nucseg`` partition) minus low-information segments. Records are kept separately (a
    supplementary record is its own stretch); ``molecule_ids`` names their physical molecule."""
    from fiberhmm.assoc.rate import RateData, compress
    require_tables(scan, ['nuc/marks.tsv.bgz'], 'rate')
    code = NUCSEG_KIND.get(a_type)
    if code is None:
        raise ScanInputError(f'A must be one of {sorted(NUCSEG_KIND)}')
    rec = {}
    for r in _rows(scan, 'nuc/marks.tsv.bgz', regions):
        key = (r['dataset_id'], r.get('input_index', '0'), r['record'])
        d = rec.setdefault(key, {'chrom': r['chrom'], 'molecule': r['molecule'], 'dataset': r['dataset_id'],
                                 'qname': r['qname'], 'seg': [], 'low': [], 'a': [], 'b': []})
        q = [int(x) for x in r['q'].split(',')] if r['q'] else []
        s, e = int(r['start']), int(r['end'])
        if r['group'] == 'nucseg' and q:
            if q[0] < 16:
                d['seg'].append((s, e))
            if q[0] == 3:
                d['low'].append((s, e))
            if q[0] == code:
                d['a'].append((s, e))
        elif r['group'] == 'nucsz' and q and b_type == 'tf' and (q[0] & 16) and (e - s) >= min_tf_bp:
            d['b'].append((s + e) // 2)
    if b_type != 'tf':
        mod = module_of(b_type)
        require_tables(scan, [f'{mod}/states.tsv.bgz'], 'rate')
        hit_type = {h['hit_id']: h['type'] for h in _rows(scan, f'{mod}/hits.tsv.bgz', regions)}
        for r in _rows(scan, f'{mod}/states.tsv.bgz', regions):
            if r['state'] != '1' or hit_type.get(r['hit_id']) != b_type:
                continue
            key = (r['dataset_id'], r.get('input_index', '0'), r['record'])
            if key in rec:
                cs, ce = r.get('call_start') or r['start'], r.get('call_end') or r['end']
                rec[key]['b'].append(int((float(cs) + float(ce)) / 2))
    L, IV, PT, CH, MID, MOL, DS, JN, QN = [], [], [], [], [], [], [], [], []
    for key in sorted(rec):
        d = rec[key]
        if not d['seg']:
            continue
        seg = sorted(d['seg'])
        adm = _subtract(_union(seg), _union(d['low']))
        if not adm:
            continue
        n, iv, pts, junc = compress(adm, d['a'], d['b'])
        if n <= 0:
            continue
        L.append(n); IV.append(iv); PT.append(pts); JN.append(junc)
        CH.append(d['chrom']); MID.append((seg[0][0] + seg[-1][1]) // 2); MOL.append(d['molecule'])
        DS.append(d['dataset']); QN.append(d['qname'])
    return RateData(np.array(L, np.int64), IV, PT, np.array(CH, dtype=object), np.array(MID, np.int64),
                    np.array(MOL, dtype=object), labels={'dataset': np.array(DS, dtype=object),
                                                         'qname': np.array(QN, dtype=object)}, junctions=JN)


def _union(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]


def _subtract(iv, cut):
    out = []
    for a, b in iv:
        pieces = [(a, b)]
        for c, d in cut:
            nxt = []
            for x, y in pieces:
                if d <= x or c >= y:
                    nxt.append((x, y))
                    continue
                if c > x:
                    nxt.append((x, c))
                if d < y:
                    nxt.append((d, y))
            pieces = nxt
        out.extend(p for p in pieces if p[1] > p[0])
    return out
