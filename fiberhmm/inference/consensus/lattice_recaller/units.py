"""Molecule preparation for the lattice recaller: per-channel site lattices and native calls from evidence units."""
from __future__ import annotations

import numpy as np

SITE_PAD = 60          # sites kept around a tile (the prototype's lattice span)
MIN_OVERLAP = 50       # a molecule must overlap the tile by more than this


def channel_of(source, unit):
    """dataset::strand; Hia5 strands are alignment orientations of the same molecule and are pooled."""
    strand = 'pooled' if source['chemistry'].startswith('hia5') else unit['strand']
    return f"{source['dataset_id']}::{strand}"


def chemistry_of_channel(sources, channel):
    ds = channel.split('::', 1)[0]
    return next(s['chemistry'] for s in sources if s['dataset_id'] == ds)


def full_lattice(source, unit):
    """Whole-unit arrays (positions, hits, p_accessible, p_protected, per-site log-likelihood ratio d)."""
    pos = np.asarray(unit['positions'], dtype=np.int64)
    hit = np.asarray(unit['hits'], dtype=bool)
    pa = np.clip(np.asarray(unit['p_accessible'], float), 1e-6, 1 - 1e-6)
    pp = np.clip(np.asarray(unit['p_protected'], float), 1e-6, 1 - 1e-6)
    return pos, hit, pa, pp


def tile_units(sources, w0, w1, call_min_llr=5.0, call_max_bp=100, efficiency=None):
    """Units overlapping [w0, w1) with their sites within SITE_PAD bp, as the prototype's kclass.load + relattice.

    efficiency: optional {channel: factor} scaling p_accessible (p_protected kept below the scaled rate)."""
    out = []
    for source in sources:
        for u in source['units']:
            if u['reference_start'] >= w1 - MIN_OVERLAP or u['reference_end'] <= w0 + MIN_OVERLAP:
                continue
            pos, hit, pa, pp = full_lattice(source, u)
            keep = (pos >= w0 - SITE_PAD) & (pos < w1 + SITE_PAD)
            if keep.sum() < 3:
                continue
            ch = channel_of(source, u)
            pos, hit, pa, pp = pos[keep], hit[keep], pa[keep], pp[keep]
            if efficiency and ch in efficiency:
                pa = np.clip(pa*efficiency[ch], 1e-6, 1 - 1e-6); pp = np.minimum(pp, pa*0.9)
            d = np.where(hit, np.log(pp) - np.log(pa), np.log1p(-pp) - np.log1p(-pa))
            calls = [(int(c['interval'][0]), int(c['interval'][1]), float(c.get('llr', 99.)))
                     for c in u.get('native_multi_interval_calls', [])
                     if c['interval'][0] >= w0 and c['interval'][1] <= w1
                     and c['interval'][1] - c['interval'][0] <= call_max_bp and c.get('llr', 99.) >= call_min_llr]
            out.append(dict(uid=u['unit_id'], ch=ch, dataset=source['dataset_id'], strand=u['strand'], pos=pos, hit=hit,
                            pa=pa, pp=pp, d=d, calls=calls, nuc=u.get('raw_nuc_intervals', []), msp=u.get('msp_intervals', [])))
    return out


def unknown_accessible_fraction(sources, channel, uids):
    """Channel's mark rate over its mean accessible probability, whole lattices of the given units, clipped to [.05, .95]."""
    k = spa = 0.
    for source in sources:
        for u in source['units']:
            if u['unit_id'] not in uids or channel_of(source, u) != channel:
                continue
            k += float(np.sum(u['hits'])); spa += float(np.sum(u['p_accessible']))
    return float(np.clip(k/spa, .05, .95)) if spa > 0 else .5


def efficiency_factors(sources, top_fraction=.1):
    """Per-channel deamination/methylation efficiency: observed/expected rate in the most-marked molecules (<= 1)."""
    rates = {}
    for source in sources:
        for u in source['units']:
            h = np.asarray(u['hits'], float); p = np.asarray(u['p_accessible'], float)
            if len(h):
                rates.setdefault(channel_of(source, u), []).append((h.mean(), p.mean()))
    out = {}
    for ch, r in rates.items():
        r = np.asarray(r); top = r[:, 0] >= np.quantile(r[:, 0], 1 - top_fraction)
        out[ch] = float(min(1., r[top, 0].mean()/max(r[top, 1].mean(), 1e-9)))
    return out
