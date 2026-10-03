"""Analysis views of a finished NFR run (EXPERIMENTAL preview).

Pure functions over the result ``run_accessibility`` returns (schema v1; most also accept v0 and say what they cannot
do). Nothing here re-runs discovery:

* ``pair_split`` / ``combo_split``: the reads of a pair test (or a combination table) split into their groups with the
  same eligibility code the test used (``coaccess.pair_eligibility`` / ``combo_eligibility``), so the four groups of a
  pair are its 2x2 table, plus every other read with the reason it is not informative; per-stratum tables
  (channel x openness quintile, per-read openness recomputed from the stored nucleosome calls), the spacing between the
  two openings on molecules carrying both, and per-group accessibility profiles;
* ``variant_profiles``: per-variant mini profiles (share of member reads open at each position);
* ``size_shape``: opening widths per variant, the left-edge x right-edge density of every opening, and the boundary
  (-1 / +1) nucleosomes of each variant's openings;
* ``vplots``: opening centre x size, and the coverage V-plot (each opening adds to every position it covers, in its
  size row), for all openings of the NFR and for one variant's;
* ``phasing``: nucleosome occupancy around the NFR on member vs non-member reads;
* ``group_prevalence`` / ``quantify_frozen`` / ``transfer``: prevalence of the frozen variant catalogue on any set of
  reads (read groups, other datasets) by the same configuration EM, with a read bootstrap.

A variant's openings are the gaps its reads' most likely configuration assigns to it (one label per gap); its member
reads are those with P(variant) >= the run's open threshold (the co-accessibility rule).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from . import coaccess as C
from . import gaps as G
from . import variants as V
from .workflow import COMPATIBLE_SCHEMAS, SCHEMA, decode_element_states

STEP = 10
MAX_BINS = 400          # per axis of any grid a view returns (wider extents get coarser bins)


def _coarse(step, extent):
    """The smallest multiple of ``step`` that keeps ``extent`` within MAX_BINS bins."""
    return int(step*max(1, int(np.ceil(max(extent, 1)/(step*MAX_BINS)))))


class NeedsRerun(ValueError):
    """The result predates the stored data this view needs (schema v0): run Find variants again."""


def check(result):
    if result.get('schema') not in COMPATIBLE_SCHEMAS:
        raise ValueError(f"Unknown accessibility result schema {result.get('schema')!r}")


def has_context(result):
    return result.get('schema') == SCHEMA and bool(result.get('element_states'))


def _need_context(result, what):
    if not has_context(result):
        raise NeedsRerun(f'{what} needs a run made with this version (it stores each read\'s nucleosome calls and test '
                         'states); run Find variants again.')


def _threshold(result):
    return float((result.get('parameters') or {}).get('open_threshold', 0.5))


def _channel(m):
    return f"{m['dataset']}::{m['strand']}"


def gaps_by_uid(result):
    """{unit_id: [(g0, g1), ...]}: the callable openings of every NFR, as the run's shared rule saw them."""
    out = {}
    for nfr in result['nfrs']:
        for uid, m in result['molecules'].items():
            rec = (m.get('nfr') or {}).get(nfr['id'])
            if rec and rec.get('status') == 'callable':
                out.setdefault(uid, []).extend((int(a), int(b)) for a, b in rec.get('gaps') or ())
    return out


def covariates(result):
    """The run's per-read openness inputs, rebuilt from the stored nucleosome calls (schema v1)."""
    out = {}
    for uid, m in result['molecules'].items():
        if m.get('cov') is not None:
            out[uid] = G.covariate_from_nucs(m.get('nucs') or (), m['cov'], _channel(m))
    return out


def elements(result):
    _need_context(result, 'Splitting reads by elements')
    els = decode_element_states(result)
    labels = {e['id']: e for e in result.get('elements') or ()}
    for eid, e in els.items():
        e.update({k: v for k, v in labels.get(eid, {}).items() if k not in ('start', 'end', 'state', 'channel', 'kind')})
    return els


def find_pair(result, a, b):
    return next((p for p in result.get('pairs') or () if {p['a'], p['b']} == {a, b}), None)


def _nfr(result, nfr_id):
    n = next((n for n in result['nfrs'] if n['id'] == nfr_id), None)
    if n is None:
        raise ValueError(f'No NFR {nfr_id!r} in this run')
    return n


def _variant(result, variant_id):
    nfr_id, _, name = str(variant_id).partition(':')
    nfr = _nfr(result, nfr_id)
    v = next((v for v in nfr.get('variants') or () if v['name'] == name), None)
    if v is None:
        raise ValueError(f'No variant {variant_id!r} in this run')
    return nfr, v


# ------------------------------------------------------------------ per-read openings of a variant
def assigned_gaps(result, nfr_id):
    """Every callable read of the NFR: (uid, molecule, record, [(g0, g1, label), ...]) with each gap's label from the
    read's most likely configuration ('V2', 'other'); closed reads have no gaps."""
    out = []
    for uid in sorted(result['molecules']):
        m = result['molecules'][uid]
        rec = (m.get('nfr') or {}).get(nfr_id)
        if not rec or rec.get('status') != 'callable':
            continue
        gaps = [tuple(int(x) for x in g) for g in rec.get('gaps') or ()]
        cfg = rec.get('map') or ''
        tokens = cfg.split('+') if gaps and cfg and cfg != 'closed' and not cfg.startswith('widest') else []
        if len(tokens) != len(gaps):
            tokens = ['other']*len(gaps)
        out.append((uid, m, rec, [(a, b, t) for (a, b), t in zip(gaps, tokens)]))
    return out


def _member(rec, name, thr):
    p = (rec.get('p') or {}).get(name)
    return p is not None and p >= thr


def member_uids(result, variant_id):
    """The variant's member reads (P >= the open threshold) exactly as the pair tests used them (the stored element
    states, from unrounded probabilities); runs without them (v0, whole-NFR elements) use the stored probabilities."""
    block = (result.get('element_states') or {}).get('elements') or {}
    row = block.get(variant_id)
    if row is not None and len(row['states']) == len(result['molecules']):
        return {u for u, c in zip(sorted(result['molecules']), row['states']) if c == '1'}
    nfr_id, _, name = str(variant_id).partition(':')
    thr = _threshold(result)
    return {u for u, m in result['molecules'].items()
            if (m.get('nfr') or {}).get(nfr_id, {}).get('status') == 'callable' and _member(m['nfr'][nfr_id], name, thr)}


# ------------------------------------------------------------------ profiles
def _profile(mols, lo, hi, step=STEP):
    """Accessibility (1 - nucleosome occupancy) at each position: among reads whose nucleosome-bounded coverage spans
    it, the share not inside a >= 90-bp nucleosome call; n = those reads."""
    x = np.arange(lo, hi, step) + step/2
    cov = np.zeros(len(x)); occ = np.zeros(len(x))
    for m in mols:
        nucs = m.get('nucs') or ()
        extent = m.get('nuc_span') or ((nucs[0][0], nucs[-1][1]) if nucs else None)
        if not extent:
            continue
        s, e = extent
        span = m.get('span') or (s, e)
        s, e = max(s, span[0] if span[0] is not None else s), min(e, span[1] if span[1] is not None else e)
        cov += (x >= s) & (x < e)
        for a, b in nucs:
            occ += (x >= a) & (x < b) & (x >= s) & (x < e)
    with np.errstate(invalid='ignore', divide='ignore'):
        acc = np.where(cov > 0, 1 - occ/np.maximum(cov, 1), np.nan)
    return dict(x=[float(v) for v in x], accessibility=[None if np.isnan(v) else round(float(v), 4) for v in acc],
                n=[int(v) for v in cov])


def variant_profiles(result, nfr_id, pad=200, step=STEP):
    """Mini profiles for the discovery overview: per variant, the share of its member reads with an opening (any gap
    in this NFR) at each position; 'all': the share of every callable read. Needs only the gaps (v0 and v1)."""
    check(result)
    nfr = _nfr(result, nfr_id)
    lo, hi = int(nfr['start']) - pad, int(nfr['end']) + pad
    x = np.arange(lo, hi, step) + step/2
    rows = assigned_gaps(result, nfr_id)

    def openness(sel):
        acc = np.zeros(len(x))
        for _uid, _m, _rec, gaps in sel:
            o = np.zeros(len(x), bool)
            for a, b, _t in gaps:
                o |= (x >= a) & (x < b)
            acc += o
        return [round(float(v), 4) for v in acc/max(len(sel), 1)], len(sel)
    allp, n_all = openness(rows)
    out = dict(nfr=nfr_id, x=[float(v) for v in x], step=step, all=dict(profile=allp, n=n_all), variants={})
    for v in nfr.get('variants') or ():
        members = member_uids(result, v['id'])
        prof, n = openness([r for r in rows if r[0] in members])
        out['variants'][v['name']] = dict(profile=prof, n=n)
    return out


# ------------------------------------------------------------------ pair and combination splits
CELLS = ('11', '10', '01', '00')


def _haldane(t):
    a, b, c, d = (x + .5 for x in t)
    return float(np.log2(a*d/(b*c)))


def pair_split(result, a, b, *, quintiles=5, profile_pad=300, step=STEP):
    """Every read of the run split for elements a x b: the four groups of the pair test (A+B+, A+B-, A-B+, A-B-; the
    test's 2x2 table by construction) and the not-informative reads with their reason. Oriented like the tested pair
    row when there is one."""
    check(result)
    els = elements(result)
    if a not in els or b not in els:
        raise ValueError(f'Unknown element {a if a not in els else b!r}')
    pair = find_pair(result, a, b)
    if pair is not None and pair['a'] == b:
        a, b = b, a
    A, B = els[a], els[b]
    if A['kind'] == B['kind'] and C.overlapping(A, B):
        raise ValueError(f'{a} and {b} overlap: one opening cannot be two elements, so the pair has no split.')
    mols = result['molecules']
    cov = covariates(result)
    el = C.pair_eligibility(A, B, gaps_by_uid(result), cov, int((result.get('parameters') or {}).get('openness_pad_bp', 0)),
                            universe=mols)
    cells = {k: [] for k in CELLS}
    for u in el['uids']:
        cells[f"{A['state'][u]}{B['state'][u]}"].append(u)
    table = [len(cells[k]) for k in CELLS]
    out = dict(a=a, b=b, kind_a=A['kind'], kind_b=B['kind'], cells=cells, table=table, excluded=el['excluded'],
               not_informative=sorted(u for us in el['excluded'].values() for u in us), tested=pair is not None,
               pair=pair, skipped=next((s['reason'] for s in (result.get('family') or {}).get('skipped') or ()
                                        if {s['a'], s['b']} == {a, b}), None))
    uids = el['uids']
    if uids:
        x = np.array([A['state'][u] for u in uids]); y = np.array([B['state'][u] for u in uids])
        op = np.array([el['op'][u] for u in uids]); ch = np.array([cov[u]['ch'] for u in uids])
        per_bin = int((result.get('parameters') or {}).get('per_bin', C.PER_BIN))
        strata = C.openness_strata(op, ch, per_bin)
        mh = C.mantel_haenszel(x, y, strata)
        out['recomputed'] = dict(mh=None if np.isnan(mh[0]) else float(mh[0]), strata=int(len(np.unique(strata))))
        out['strata'] = _coarse_strata(x, y, op, ch, quintiles)
    else:
        out['recomputed'] = None
        out['strata'] = []
    out['spacing'] = _spacing(result, A, B, cells['11'])
    lo = int(min(A['start'], B['start'])) - profile_pad
    hi = int(max(A['end'], B['end'])) + profile_pad
    out['profiles'] = {k: _profile([mols[u] for u in cells[k]], lo, hi, step) | dict(reads=len(cells[k])) for k in CELLS}
    out['profiles']['na'] = _profile([mols[u] for u in out['not_informative']], lo, hi, step) | dict(reads=len(out['not_informative']))
    out['window'] = [lo, hi]
    return out


def _coarse_strata(x, y, op, ch, quintiles=5):
    """The breakdown shown in the detail card: per channel (DAF strand) x openness quintile, the 2x2 table and the
    Haldane log2 OR. (The test itself uses finer strata of ~50 reads.)"""
    rows = []
    for c in sorted(set(ch.tolist())):
        m = ch == c
        cuts = np.quantile(op[m], np.linspace(0, 1, quintiles + 1)[1:-1]) if m.sum() else []
        b = np.searchsorted(cuts, op[m], side='right')
        for q in range(quintiles):
            k = b == q
            if not k.any():
                continue
            xs, ys, os_ = x[m][k], y[m][k], op[m][k]
            t = C.table(xs, ys)
            rows.append(dict(channel=c, quintile=q + 1, n=int(k.sum()), openness=[round(float(os_.min()), 3), round(float(os_.max()), 3)],
                             table=list(t), log2or=round(_haldane(t), 3)))
    return rows


def _gap_for(result, e, uid):
    """The opening an NFR element holds on this read: the gap its configuration assigns to the variant (or, for a
    whole-NFR / depth element, the widest gap in the NFR)."""
    nfr_id = e.get('nfr') or str(e['id']).split(':')[0]
    rec = (result['molecules'][uid].get('nfr') or {}).get(nfr_id)
    if not rec or not rec.get('gaps'):
        return None
    gaps = [tuple(g) for g in rec['gaps']]
    name = str(e['id']).split(':', 1)[1] if ':' in str(e['id']) else ''
    cfg = rec.get('map') or ''
    tokens = cfg.split('+')
    if e.get('subtype') == 'variant' or (name.startswith('V') and name[1:].isdigit()):
        # a variant's opening is the gap its read's configuration assigns it; none when that configuration does not
        # carry it (membership can pass the threshold without the single most likely configuration holding it)
        return gaps[tokens.index(name)] if len(tokens) == len(gaps) and name in tokens else None
    return max(gaps, key=lambda g: g[1] - g[0])          # whole-NFR / depth elements: the widest opening


def _spacing(result, A, B, both):
    """On molecules where both are open: the protected stretch between the two openings (NFR x NFR), or where the
    footprint class sits relative to the opening (NFR x class)."""
    if not both:
        return None
    if A['kind'] == 'nfr' and B['kind'] == 'nfr':
        d = []
        for u in both:
            ga, gb = _gap_for(result, A, u), _gap_for(result, B, u)
            if ga and gb and ga != gb:
                left, right = sorted([ga, gb])
                d.append(right[0] - left[1])
        if not d:
            return None
        d = np.array(d, float)
        med = float(np.median(d))
        kind = ('a factor-sized protection' if med < 90 else 'about one nucleosome' if med <= 220
                else f'about {max(2, int(round(med/190)))} nucleosomes')
        return dict(kind='between', reads=len(d), median=med, q1=float(np.percentile(d, 25)), q3=float(np.percentile(d, 75)),
                    note=f'On {len(d):,} molecules with both open, {kind} separates the two openings '
                         f'(median {med:.0f} bp, IQR {np.percentile(d, 25):.0f}–{np.percentile(d, 75):.0f}).')
    nf, tf = (A, B) if A['kind'] == 'nfr' else (B, A)
    if tf['kind'] != 'tf' or nf['kind'] != 'nfr':
        return None
    c = (tf['start'] + tf['end'])/2
    inside = edge = 0
    for u in both:
        g = _gap_for(result, nf, u)
        if g is None:
            continue
        if g[0] <= tf['start'] and tf['end'] <= g[1]:
            inside += 1
        elif g[0] < c < g[1]:
            edge += 1
    n = len(both)
    return dict(kind='class', reads=n, inside=inside, edge=edge,
                note=f'On {n:,} molecules with both, the footprint lies inside the opening on {inside:,} and straddles its edge on {edge:,}.')


def combo_split(result, ids):
    """Reads spanning every chosen element (minus shared openings), grouped by their open/closed pattern; the
    combination table's eligibility (``coaccess.combo_eligibility``)."""
    check(result)
    els = elements(result)
    missing = [i for i in ids if i not in els]
    if missing:
        raise ValueError(f'Unknown element(s): {", ".join(missing)}')
    if not 2 <= len(ids) <= 8:
        raise ValueError('Choose 2-8 elements')
    chosen = [els[i] for i in ids]
    el = C.combo_eligibility(chosen, gaps_by_uid(result), universe=result['molecules'])
    groups = {}
    for u in el['uids']:
        groups.setdefault(''.join(str(e['state'][u]) for e in chosen), []).append(u)
    order = sorted(groups, key=lambda p: (-len(groups[p]), p))
    return dict(elements=list(ids), patterns=[dict(pattern=p, reads=groups[p]) for p in order], excluded=el['excluded'],
                not_informative=sorted(u for us in el['excluded'].values() for u in us), n=len(el['uids']))


# ------------------------------------------------------------------ size and shape
def _hist(values, lo, hi, step):
    """Counts in [lo, hi) by step (hi is the last bin's right edge)."""
    edges = np.arange(lo, hi + step/2, step)
    c, _ = np.histogram(values, edges) if len(values) else (np.zeros(len(edges) - 1, int), edges)
    return [int(v) for v in c]


def _quantiles(values):
    if not len(values):
        return None
    q = np.percentile(values, [5, 25, 50, 75, 95])
    return dict(n=int(len(values)), p5=float(q[0]), q1=float(q[1]), median=float(q[2]), q3=float(q[3]), p95=float(q[4]),
                mean=float(np.mean(values)))


def size_shape(result, variant_id, step=STEP, max_points=4000):
    """Width distribution per variant of the NFR (with 'other'), every opening's (left edge, right edge) with its label
    (a shift moves along the diagonal, a wider or narrower opening along the anti-diagonal), a binned edge density,
    and the boundary nucleosomes (-1 ends at the left edge, +1 starts at the right edge) of each variant's openings."""
    check(result)
    nfr, v = _variant(result, variant_id)
    rows = assigned_gaps(result, nfr['id'])
    names = [x['name'] for x in nfr.get('variants') or ()] + ['other']
    widths = {n: [] for n in names}
    points = []
    for _uid, _m, _rec, gaps in rows:
        for a, b, t in gaps:
            widths.setdefault(t, []).append(b - a)
            points.append((a, b, t))
    allw = [w for ws in widths.values() for w in ws]
    step = _coarse(step, max(allw, default=0) - min(allw, default=0))
    wlo = int(min(allw, default=0)//step*step); whi = int(max(allw, default=step)//step*step + step)
    sizes = dict(step=step, lo=wlo, hi=whi, variants={n: dict(summary=_quantiles(widths[n]), hist=_hist(widths[n], wlo, whi, step))
                                                      for n in names if n in widths})
    # edge density (all openings) on a common grid
    if points:
        L = np.array([p[0] for p in points]); R = np.array([p[1] for p in points])
        step = _coarse(step, max(L.max() - L.min(), R.max() - R.min()))
        llo, lhi = int(L.min()//step*step), int(L.max()//step*step + step)
        rlo, rhi = int(R.min()//step*step), int(R.max()//step*step + step)
        H, _, _ = np.histogram2d(L, R, [np.arange(llo, lhi + step, step), np.arange(rlo, rhi + step, step)])
        density = dict(step=step, left=[llo, lhi], right=[rlo, rhi], counts=H.astype(int).tolist())
    else:
        density = None
    keep = points if len(points) <= max_points else [points[i] for i in np.linspace(0, len(points) - 1, max_points).astype(int)]
    edges = dict(points=[[a, b, t] for a, b, t in keep], total=len(points), shown=len(keep), density=density,
                 centres={x['name']: [x['L'], x['R']] for x in nfr.get('variants') or ()})
    boundary = None
    if has_context(result):
        boundary = {}
        for name in names[:-1]:
            minus, plus = [], []
            for _uid, m, _rec, gaps in rows:
                nucs = [tuple(n) for n in m.get('nucs') or ()]
                for a, b, t in gaps:
                    if t != name:
                        continue
                    left = next((n for n in nucs if n[1] == a), None)
                    right = next((n for n in nucs if n[0] == b), None)
                    if left:
                        minus.append(dict(dyad=(left[0] + left[1])/2, size=left[1] - left[0]))
                    if right:
                        plus.append(dict(dyad=(right[0] + right[1])/2, size=right[1] - right[0]))
            boundary[name] = dict(minus1=_quantiles([d['dyad'] for d in minus]), plus1=_quantiles([d['dyad'] for d in plus]),
                                  minus1_size=_quantiles([d['size'] for d in minus]), plus1_size=_quantiles([d['size'] for d in plus]),
                                  minus1_dyads=[d['dyad'] for d in minus][:max_points], plus1_dyads=[d['dyad'] for d in plus][:max_points])
    return dict(variant=variant_id, nfr=nfr['id'], name=v['name'], sizes=sizes, edges=edges, boundary=boundary,
                boundary_note=None if boundary is not None else 'Boundary nucleosomes need a run made with this version.')


# ------------------------------------------------------------------ V-plots
def vplot_matrices(gaps, lo, hi, size_lo, size_hi, step=STEP, size_step=STEP):
    """(centre, coverage) matrices, rows = size bins [size_lo, size_hi) of size_step, columns = position bins
    [lo, hi) of step. Centre: an opening adds 1 at (its centre's bin, its size's bin). Coverage: it adds, in its size
    row, the share of every position bin it covers (so a row's sum is its width / step inside the window)."""
    nx = max(1, int(np.ceil((hi - lo)/step))); ny = max(1, int(np.ceil((size_hi - size_lo)/size_step)))
    centre = np.zeros((ny, nx)); cover = np.zeros((ny, nx))
    edges = np.minimum(lo + np.arange(nx + 1)*step, hi)     # a last partial bin ends at hi
    for a, b in gaps:
        w = b - a
        if not size_lo <= w < size_hi:
            continue
        r = int((w - size_lo)//size_step)
        c = (a + b)/2
        if lo <= c < hi:
            centre[r, int((c - lo)//step)] += 1
        ov = np.clip(np.minimum(edges[1:], b) - np.maximum(edges[:-1], a), 0, None)/step
        cover[r] += ov
    return centre, cover


def vplots(result, variant_id, pad=300, step=STEP, size_step=STEP):
    """Centre x size and coverage x size V-plots of the NFR's openings: every callable opening ('all') and the
    variant's own ('own'). Counts (not normalised); sizes from the run's minimum gap up to the widest opening."""
    check(result)
    nfr, v = _variant(result, variant_id)
    rows = assigned_gaps(result, nfr['id'])
    allg = [(a, b) for *_x, gaps in rows for a, b, _t in gaps]
    mine = [(a, b) for *_x, gaps in rows for a, b, t in gaps if t == v['name']]
    lo = int(min([nfr['start']] + [a for a, _ in allg])) - pad
    hi = int(max([nfr['end']] + [b for _, b in allg])) + pad
    # bounded grids: coarser bins rather than more than MAX_BINS per axis
    step = _coarse(step, hi - lo); size_step = _coarse(size_step, max([0] + [b - a for a, b in allg]))
    lo, hi = lo//step*step, -(-hi//step)*step
    size_lo = int((result.get('parameters') or {}).get('min_gap_bp', 60))//size_step*size_step
    size_hi = int(max([size_lo + size_step] + [b - a for a, b in allg]))//size_step*size_step + size_step
    out = dict(variant=variant_id, nfr=nfr['id'], window=[lo, hi], step=step, size=[size_lo, size_hi], size_step=size_step,
               counts=dict(all=len(allg), own=len(mine)), span=[v['L'], v['R']])
    for key, g in (('all', allg), ('own', mine)):
        c, cv = vplot_matrices(g, lo, hi, size_lo, size_hi, step, size_step)
        out[key] = dict(centre=c.astype(int).tolist(), coverage=np.round(cv, 3).tolist())
    return out


# ------------------------------------------------------------------ phasing
def phasing(result, variant_id, flank=1000, step=STEP):
    """Nucleosome occupancy (share of reads covering a position that have a >= 90-bp nucleosome call there) from the
    NFR's start - flank to its end + flank, on the variant's member reads and on the other callable reads."""
    check(result)
    _need_context(result, 'Nucleosome phasing')
    nfr, v = _variant(result, variant_id)
    rows = assigned_gaps(result, nfr['id'])
    lo, hi = int(nfr['start']) - flank, int(nfr['end']) + flank
    mine = member_uids(result, v['id'])
    members = [m for u, m, _rec, _g in rows if u in mine]
    others = [m for u, m, _rec, _g in rows if u not in mine]

    def occupancy(mols):
        p = _profile(mols, lo, hi, step)
        return dict(x=p['x'], occupancy=[None if a is None else round(1 - a, 4) for a in p['accessibility']], n=p['n'], reads=len(mols))
    shape = size_shape(result, variant_id)
    return dict(variant=variant_id, nfr=nfr['id'], window=[lo, hi], span=[v['L'], v['R']], members=occupancy(members),
                others=occupancy(others), boundary=(shape['boundary'] or {}).get(v['name']))


# ------------------------------------------------------------------ prevalence of the frozen catalogue
def _catalogue_reads(result, nfr_id, uids=None):
    reads = []
    for uid in sorted(result['molecules']) if uids is None else sorted(uids):
        m = result['molecules'].get(uid)
        rec = (m or {}).get('nfr', {}).get(nfr_id)
        if not rec or rec.get('status') != 'callable':
            continue
        edges = rec.get('edges') or []
        if rec.get('gaps') and len(edges) != len(rec['gaps']):
            raise NeedsRerun('Prevalence per group needs a run made with this version (it stores the edge features); run Find variants again.')
        reads.append(dict(uid=uid, callable=True, gaps=[dict(L=e[0], R=e[1], wL=e[2], wR=e[3]) for e in edges]))
    return reads


def quantify_frozen(reads, cat, bootstrap=200, seed=None):
    """The configuration EM of ``variants.quantify`` with the catalogue's geometry fixed (no discovery): prevalence
    (EM), strict share and the read-bootstrap interval per variant; conditional on the catalogue."""
    # 'cand' first, as discovery's variants: quantify finds the reference with list.index, and dict equality must
    # stop at a differing plain key before it reaches the arrays
    vs = [dict(cand=x['name'], mu=np.asarray(x['mu'], float), cov=np.asarray(x['cov'], float)) for x in cat['variants']]
    if not reads:
        return dict(n=0, closed=None, variants={x['name']: dict(prevalence=None, strict=None, ci=None) for x in cat['variants']})
    opt = SimpleNamespace(seed=int(cat.get('seed', 1)) if seed is None else seed, bootstrap=int(bootstrap), shift_bp=25)
    q = V.quantify(reads, vs, SimpleNamespace(log_other=float(cat['log_other'])), opt)
    out = {}
    for j, x in enumerate(cat['variants']):
        v = q['variants'][j]
        out[x['name']] = dict(prevalence=round(float(v['prevalence']), 4), strict=round(float(v['strict']), 4),
                              ci=[round(float(v['ci'][0]), 4), round(float(v['ci'][1]), 4)] if v.get('ci') else None)
    return dict(n=len(q['reads']), closed=round(float(q['closed']), 4), variants=out)


def group_prevalence(result, variant_id, groups, bootstrap=None):
    """groups: {label: [unit_id, ...]}. Per group, the variant's prevalence among the group's callable reads, by the
    configuration EM with the run's catalogue frozen (geometry fixed, weights refitted on the group), with a read
    bootstrap; plus 'all': every callable read, the run's own numbers.

    The read x configuration likelihoods are computed once for the NFR; each group's EM and its bootstrap
    replicates (warm-started from the group's fit) use its rows. Bootstrap seed: the run's."""
    check(result)
    nfr, v = _variant(result, variant_id)
    cat = nfr.get('catalogue')
    if not cat:
        raise NeedsRerun('Prevalence per group needs a run made with this version (it stores the variant catalogue); run Find variants again.')
    if bootstrap is None:      # the run's replicate count
        bootstrap = int((result.get('parameters') or {}).get('bootstrap', 200))
    reads = _catalogue_reads(result, nfr['id'])
    vs = [dict(cand=x['name'], mu=np.asarray(x['mu'], float), cov=np.asarray(x['cov'], float)) for x in cat['variants']]
    callable_reads, configs, LL = V.config_loglik(reads, vs, SimpleNamespace(log_other=float(cat['log_other'])))
    j = [x['name'] for x in cat['variants']].index(v['name'])
    contains = np.array([j in c for c in configs], float)
    row_of = {r['uid']: i for i, r in enumerate(callable_reads)}
    out = []
    for label, uids in groups.items():
        idx = np.array(sorted({row_of[u] for u in uids if u in row_of}), int)
        row = dict(group=label, callable=int(len(idx)), prevalence=None, strict=None, ci=None, reads=len(uids))
        if len(idx):
            w, P = V._em(LL[idx])
            Pv = P @ contains
            rng = np.random.default_rng(int(cat.get('seed', 1)))
            boots = [contains @ V._em(LL[idx[rng.integers(0, len(idx), len(idx))]], 500, w0=w)[0] for _ in range(int(bootstrap))]
            row.update(prevalence=round(float(contains @ w), 4), strict=round(float(np.mean(Pv >= 0.9)), 4),
                       ci=[round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)] if boots else None)
        out.append(row)
    out.append(dict(group='all', callable=int(nfr['callable']), prevalence=v['prevalence'], strict=v['strict'], ci=v['ci'], reads=None))
    return dict(variant=variant_id, nfr=nfr['id'], groups=out,
                note='Prevalence among each group\'s callable reads; the variant set and geometry are the run\'s (frozen); '
                     'intervals: read bootstrap, conditional on that catalogue. "all" is the run itself.')


def transfer(payload, result, nfr_id, bootstrap=None):
    """Quantify an NFR's frozen catalogue on another payload (other datasets at the same locus): per dataset, the
    callable reads (both flanking nucleosomes seen) and each variant's prevalence."""
    check(result)
    nfr = _nfr(result, nfr_id)
    cat = nfr.get('catalogue')
    if not cat:
        raise NeedsRerun('Quantifying in other datasets needs a run made with this version; run Find variants again.')
    if bootstrap is None:
        bootstrap = int((result.get('parameters') or {}).get('bootstrap', 200))
    units = G.units_of(payload)
    reads = G.collect(units, tuple(cat['region']), cat['min_gap_bp'], cat['max_gaps'])
    out = {}
    for ds in sorted({r['dataset'] for r in reads}):
        sub = [r for r in reads if r['dataset'] == ds]
        q = quantify_frozen([r for r in sub if r['callable']], cat, bootstrap)
        out[ds] = dict(reads=len(sub), callable=q['n'], variants=q['variants'], closed=q.get('closed'))
    return dict(nfr=nfr_id, region=cat['region'], datasets=out)


def load_result(directory):
    """A run written by ``fiberhmm-nfr`` (result.json + context.json.gz) as the in-memory result these views take."""
    import gzip
    import json
    from pathlib import Path
    d = Path(directory)
    result = json.loads((d/'result.json').read_text())
    ctx = d/'context.json.gz'
    if not ctx.is_file():
        raise NeedsRerun(f'{d}: no context.json.gz (fiberhmm-nfr writes it from schema v1 on); run it again for these views.')
    with gzip.open(ctx, 'rt') as fh:
        result.update(json.load(fh))
    check(result)
    return result
