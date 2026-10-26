"""Window mode (NUC 3.3, 3.7) and observation descriptors (NUC 5.2).

A window is a reference interval (peak, NFR or footprint hit, scan hit, BED
row, selection) with an optional anchor and orientation. Every molecule whose
alignment covers at least ``window_min_cover`` of it is measured inside the
window projected into its molecular coordinates:

- coarse (and named) class counts of footprints centred in the window, with
  can't-tell and edge-truncated counted, never dropped;
- NRL, S and CV_g from the eligible dyads centred in the window (or too short);
- with an anchor: the molecule's local phase theta, resultant r and
  positional offset f, from its dyads in the anchor's flank bins (x0 to
  x0 + 5 periods either side, so beyond a small window; only molecules that
  span those flanks enter the phasing);
- overlap fractions of its segments with the window and the q-hat of the
  dominant regular segment;
- the nucleosome state at the anchor (``1`` covered by a nucleosome call,
  ``0`` not, ``.`` anchor not on the molecule).

Molecules below the calibration's low-information floor are counted
(``excluded_low_information``) and left out. Aggregates per observation: class mix (``insufficient`` when more than
``window_insufficient_frac`` of the footprints are can't-tell or
edge-truncated), NRL quantiles, regular fraction, and phasing (both frames,
with nulls) when an anchor is given.

An **observation** is one column of the panel: a dataset, a haplotype, a
strand or an explicit record set (haplotype pop-outs, strand splits, linked
loci and graph paths each become their own observation in the browser's
planner). :class:`Observation` is the FiberHMM side of that descriptor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from fiberhmm.nuc.params import DEFAULT_PARAMS, NucParams
from fiberhmm.nuc.phasing import anchor_offsets, phasing
from fiberhmm.nuc.records import NucMolecule
from fiberhmm.nuc.segment import IRREGULAR, LOWINFO, REGULAR
from fiberhmm.nuc.sizes import CANT_TELL, COARSE_NAMES, EDGE_TRUNCATED, MONO, NAMED_CODES
from fiberhmm.nuc.spacing import molecule_spacing


@dataclass(frozen=True)
class Observation:
    """One observation group.

    ``dataset_id`` matches the molecule's ``meta['dataset_id']``;
    ``haplotype`` its HP tag; ``strand`` '+' / '-' / 'both' (alignment
    strand); ``names`` an explicit set of QNAMEs (read subsets, filtered reads,
    graph-path carriers). Unset fields match everything.
    """
    obs_id: str
    dataset_id: Optional[str] = None
    haplotype: Optional[int] = None
    strand: str = 'both'
    names: Optional[frozenset] = None
    label: str = ''
    frame: Dict = field(default_factory=lambda: {'kind': 'linear'}, compare=False, hash=False)

    def matches(self, mol: NucMolecule) -> bool:
        if self.dataset_id is not None and mol.meta.get('dataset_id') not in (None, self.dataset_id):
            return False
        if self.haplotype is not None and mol.hp != self.haplotype:
            return False
        if self.strand == '+' and mol.is_reverse:
            return False
        if self.strand == '-' and not mol.is_reverse:
            return False
        if self.names is not None and mol.name not in self.names:
            return False
        return True

    def to_dict(self) -> dict:
        return {'obs_id': self.obs_id, 'dataset_id': self.dataset_id, 'haplotype': self.haplotype,
                'strand': self.strand, 'names': sorted(self.names) if self.names is not None else None,
                'label': self.label, 'frame': dict(self.frame)}

    @classmethod
    def from_dict(cls, d: dict) -> "Observation":
        names = d.get('names')
        return cls(obs_id=str(d['obs_id']), dataset_id=d.get('dataset_id'),
                   haplotype=d.get('haplotype'), strand=d.get('strand', 'both'),
                   names=frozenset(names) if names is not None else None,
                   label=d.get('label', ''), frame=dict(d.get('frame') or {'kind': 'linear'}))


ALL = Observation('all')


def _seg_fracs(segments, ms: int, me: int) -> Dict[str, float]:
    span = max(me - ms, 1)
    out = {'regular': 0.0, 'irregular': 0.0, 'low_information': 0.0, 'sub_rich': 0.0}
    best = (0, None)
    for s in segments:
        ov = min(s.end, me) - max(s.start, ms)
        if ov <= 0:
            continue
        key = {REGULAR: 'regular', IRREGULAR: 'irregular', LOWINFO: 'low_information'}[s.state]
        out[key] += ov / span
        if s.sub_rich:
            out['sub_rich'] += ov / span
        if s.state == REGULAR and ov > best[0]:
            best = (ov, s.q_hat)
    out['q_hat_dominant'] = best[1]
    return out


def molecule_window_row(mol: NucMolecule, result, ms: int, me: int, params: NucParams,
                        cal, anchor: Optional[int] = None, orientation: int = 0) -> dict:
    """Per-molecule window metrics in molecular ``[ms, me)``."""
    row: Dict = {'name': mol.name, 'record': mol.record_key(), 'mol_start': int(ms), 'mol_end': int(me)}
    fp = result.footprints if result is not None else None
    coarse = None
    if fp is not None:
        c = (fp['start'] + fp['end']) / 2.0
        inside = (c >= ms) & (c < me)
        codes = fp['coarse'][inside]                       # nucleosome and TF calls
        counts = {COARSE_NAMES[k]: int((codes == k).sum()) for k in COARSE_NAMES}
        counts['tf'] = int((inside & fp['is_tf']).sum())
        if cal is not None and cal.named_available:
            named = fp['named'][inside & ~fp['is_tf']]
            for nm, code in NAMED_CODES.items():
                counts[f'named_{nm}'] = int((named == code).sum())
        row['class_counts'] = counts
        coarse = np.full(len(mol.nucs), CANT_TELL)
        sel = ~fp['is_tf']
        coarse[fp['src_index'][sel]] = fp['coarse'][sel]
    nrl = cal.nrl_pooled if cal is not None else None
    sp, dy = molecule_spacing(mol, nrl, params, coarse_codes=coarse, window=(ms, me))
    row.update({'nrl': sp.nrl, 'regularity': sp.regularity, 'gap_cv': sp.gap_cv,
                'n_dyads': sp.n_dyads, 'too_short_nrl': sp.too_short_nrl,
                'too_short_regularity': sp.too_short_regularity})
    if result is not None:
        row['segments'] = _seg_fracs(result.segments, ms, me)
    if anchor is not None:
        a = int(mol.ref_to_mol([anchor])[0])
        if a < 0:
            row['state_at_anchor'] = '.'
        else:
            nucs = mol.nucs
            row['state_at_anchor'] = '1' if ((nucs[:, 0] <= a) & (nucs[:, 1] > a)).any() else '0'
    return row


@dataclass
class WindowResult:
    window: Tuple[str, int, int]
    anchor: Optional[int]
    orientation: int
    groups: Dict[str, dict]
    molecules: Dict[str, List[dict]]

    def as_dict(self):
        return {'window': list(self.window), 'anchor': self.anchor, 'orientation': self.orientation,
                'groups': self.groups}


def window_metrics(molecules: Sequence[NucMolecule], window: Tuple[str, int, int],
                   anchor: Optional[int] = None, orientation: int = 0,
                   params: NucParams = DEFAULT_PARAMS, cal=None,
                   results: Optional[Dict[int, object]] = None,
                   observations: Sequence[Observation] = (ALL,),
                   phase_frames: Sequence[str] = ('anchor', 'molecule'),
                   seed: int = 0) -> WindowResult:
    """Window metrics per observation (NUC 3.7).

    ``results`` maps ``id(molecule)`` to its :class:`MoleculeResult`; missing
    results are computed with ``molecule_outputs`` when ``cal`` is given.
    """
    from fiberhmm.nuc.molecule import molecule_outputs
    chrom, ws, we = window
    need = params.window_min_cover * (we - ws)
    groups: Dict[str, dict] = {}
    per_mol: Dict[str, List[dict]] = {}
    for obs in observations:
        rows = []
        members = []
        excluded = 0
        for m in molecules:
            if m.chrom != chrom or not obs.matches(m):
                continue
            env = m.ref_interval_to_mol(ws, we)
            if env is None or env[2] < need:
                continue
            res = results.get(id(m)) if results is not None else None
            if res is None and cal is not None:
                res = molecule_outputs(m, params, cal)
            if res is not None and res.flags.get('low_information'):
                excluded += 1                      # below the D8 floor: counted, not used
                continue
            rows.append(molecule_window_row(m, res, env[0], env[1], params, cal, anchor, orientation))
            members.append(m)
        agg: Dict = {'n_molecules': len(rows), 'excluded_low_information': excluded}
        if rows and 'class_counts' in rows[0]:
            tot = {}
            for r in rows:
                for k, v in r['class_counts'].items():
                    tot[k] = tot.get(k, 0) + v
            n_fp = sum(tot.get(COARSE_NAMES[k], 0) for k in COARSE_NAMES)
            bad = tot.get('cant_tell', 0) + tot.get('edge_truncated', 0)
            agg['class_mix'] = tot
            agg['insufficient'] = bool(n_fp == 0 or bad / n_fp > params.window_insufficient_frac)
        nrls = np.array([r['nrl'] for r in rows if not r['too_short_nrl']], float)
        agg['nrl_quantiles'] = [float(x) for x in np.percentile(nrls, [10, 25, 50, 75, 90])] if len(nrls) else None
        agg['too_short_fraction'] = float(np.mean([r['too_short_nrl'] for r in rows])) if rows else None
        regs = [r['segments']['regular'] for r in rows if 'segments' in r]
        agg['regular_fraction'] = float(np.mean(regs)) if regs else None
        if anchor is not None and members:
            # whole-molecule eligible dyads from the calibrated coarse calls
            dyads = {}
            for m in members:
                res = results.get(id(m)) if results is not None else None
                if res is None and cal is not None:
                    res = molecule_outputs(m, params, cal)
                if res is not None and res.footprints is not None:
                    fp = res.footprints
                    sel = (fp['coarse'] == MONO) & ~fp['is_tf'] & ~fp['truncated']
                    dyads[id(m)] = np.sort((fp['start'][sel] + fp['end'][sel]) / 2.0)
            states = [r.get('state_at_anchor', '.') for r in rows]
            agg['state_counts'] = {s: states.count(s) for s in ('1', '0', '.')}
            agg['phasing'] = {}
            for frame in phase_frames:
                # anchor-relative flank phasing (NUC 3.5): bins run x0 .. x0 + 5P
                # from the anchor, beyond a small window; molecules must span them
                ph = phasing(members, [(chrom, anchor, orientation)], frame=frame,
                             nrl_pooled=cal.nrl_pooled if cal is not None else None,
                             params=params, dyads=dyads or None,
                             side='down' if orientation else 'both', seed=seed, require_span=True)
                agg['phasing'][frame] = ph.as_dict()
                # per-molecule theta / r / f
                for pm in ph.per_molecule:
                    r = rows[pm['molecule']]           # phasing indexes ``members`` == ``rows``
                    r[f'theta_{frame}'] = pm['theta'] if pm['reliable'] else None
                    r[f'r_{frame}'] = pm['r']
                    r[f'f_{frame}'] = pm['f_bp'] if pm['reliable'] else None
        groups[obs.obs_id] = agg
        per_mol[obs.obs_id] = rows
    return WindowResult((chrom, int(ws), int(we)), anchor, int(orientation), groups, per_mol)
