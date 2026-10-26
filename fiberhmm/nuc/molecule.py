"""Molecule-wide outputs (NUC 5.1 ``molecule_outputs``).

Everything here is a pure function of one :class:`NucMolecule`, the
parameters and a :class:`NucCalibration`; no window, tile or neighbour enters,
so a live call and a scan give the same rows.

Footprints are the molecule's nucleosome calls and TF calls (the small end
must read TF calls: recall moves TF-sized footprints to MA ``tf``). Each gets
lambda_i, z, a coarse call and, with an absolute calibration, a named call.
Eligible dyads, spacing scalars and segments use nucleosome calls only.
A molecule whose lambda_acc is below the calibration's low-information floor
(D8) is flagged, gets one low-information segment and is left out of window
and patch aggregates (counted there as excluded).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from fiberhmm.nuc.calibration import NucCalibration
from fiberhmm.nuc.density import (accessible_counts, edge_informativeness, edge_truncated,
                                  footprint_lambdas, rho_read)
from fiberhmm.nuc.params import DEFAULT_PARAMS, NucParams
from fiberhmm.nuc.records import NucMolecule
from fiberhmm.nuc.segment import REGULAR, Segment, lowinfo_segment, segment_molecule
from fiberhmm.nuc.sizes import (CANT_TELL, LONG, MONO, MULTI2, MULTI3, SHORT, call_coarse,
                                call_named)
from fiberhmm.nuc.spacing import molecule_spacing

#: per-molecule scalar names, in their storage order
SCALARS = ('nrl', 'regularity', 'spectral_snr', 'gap_cv', 'lambda_acc', 'rho_read',
           'n_dyads', 'n_short', 'n_mono', 'n_long', 'n_multi', 'n_cant_tell',
           'reg_fraction', 'sub_rich_fraction')


def calls_digest(mol) -> str:
    """Provenance digest of the calls the nuc outputs were computed from.

    Shift-invariant (lengths and gaps in molecular order, per group), so a
    lift that rebases a hard-clipped record's annotations keeps it. ``mol``
    needs ``nucs``, ``tfs`` and ``msps`` arrays in molecular coordinates."""
    parts = []
    for name, iv in (('nuc', mol.nucs), ('tf', mol.tfs), ('msp', mol.msps)):
        if len(iv):
            iv = iv[np.lexsort((iv[:, 1], iv[:, 0]))]
            lens = (iv[:, 1] - iv[:, 0]).tolist()
            gaps = np.diff(iv[:, 0]).tolist()
        else:
            lens, gaps = [], []
        parts.append(f"{name}:{','.join(map(str, lens))}|{','.join(map(str, gaps))}")
    # relative placement of the groups
    allstarts = [int(iv[0, 0]) for iv in (mol.nucs, mol.tfs, mol.msps) if len(iv)]
    base = min(allstarts) if allstarts else 0
    parts.append('rel:' + ','.join(str(int(iv[0, 0]) - base) if len(iv) else '-'
                                    for iv in (mol.nucs, mol.tfs, mol.msps)))
    return hashlib.sha256(';'.join(parts).encode()).hexdigest()[:8]


def footprint_calls(mol: NucMolecule, params: NucParams = DEFAULT_PARAMS,
                    cal: Optional[NucCalibration] = None) -> Optional[Dict[str, np.ndarray]]:
    """Per-footprint arrays for nucleosome + TF calls, sorted by start."""
    if cal is None or cal.curve is None:
        raise ValueError('footprint calls need a calibration with a reference curve')
    nuc = mol.nucs
    tf = mol.tfs
    if len(nuc) + len(tf) == 0:
        return None
    iv = np.concatenate([nuc, tf]) if len(tf) else nuc.copy()
    is_tf = np.concatenate([np.zeros(len(nuc), bool), np.ones(len(tf), bool)])
    order = np.lexsort((iv[:, 1], iv[:, 0]))
    iv, is_tf = iv[order], is_tf[order]
    src_index = np.concatenate([np.arange(len(nuc)), np.arange(len(tf))])[order]
    k, n = accessible_counts(mol, params)
    lam_acc = float(cal.prior.shrink(k, n))
    lam = footprint_lambdas(mol, iv, lam_acc, params, cal.prior)
    nl, nr = edge_informativeness(mol, iv, params)
    trunc = edge_truncated(mol, iv)
    L = iv[:, 1] - iv[:, 0]
    cc = call_coarse(L, lam, cal.curve, cal.nrl_pooled, cal.sigma_gap, cal.sigma_loc_at,
                     nl, nr, trunc, params, lambda_supported=cal.is_supported)
    named, named_post = call_named(L, lam, cal.absolute if cal.named_available else None, nl, nr, params)
    # TF calls coinciding with a short nucleosome call, or short calls that are TF calls
    tf_overlap = is_tf.copy()
    if len(tf):
        for i in np.flatnonzero(~is_tf & (cc.code == SHORT)):
            s, e = iv[i]
            tf_overlap[i] = bool(((tf[:, 0] < e) & (tf[:, 1] > s)).any())
    return {'start': iv[:, 0].astype(np.int64), 'end': iv[:, 1].astype(np.int64),
            'length': L.astype(np.int64), 'is_tf': is_tf, 'src_index': src_index,
            'lambda': lam, 'z': cc.z, 'coarse': cc.code, 'posterior': cc.posterior,
            'named': named, 'named_posterior': named_post, 'n_left': nl, 'n_right': nr,
            'truncated': trunc, 'tf_overlap': tf_overlap, 'lambda_acc': lam_acc}


@dataclass
class MoleculeResult:
    name: str
    length: int
    footprints: Optional[Dict[str, np.ndarray]]
    segments: List[Segment]
    scalars: Dict[str, float]
    flags: Dict[str, bool]
    digest: str
    calibration_digest: str = ''
    record: tuple = field(default_factory=tuple)

    def scalar_vector(self) -> List[float]:
        return [self.scalars.get(k, float('nan')) for k in SCALARS]

    def marks_rows(self) -> List[dict]:
        """Marks-table rows (molecular coordinates): one per footprint and
        one per segment."""
        rows = []
        fp = self.footprints
        if fp is not None:
            for i in range(len(fp['start'])):
                rows.append({'type': 'nuc.size.coarse', 'start': int(fp['start'][i]), 'end': int(fp['end'][i]),
                             'q1': int(fp['coarse'][i]), 'q2': int(round(255 * float(fp['posterior'][i]))),
                             'L_mol': int(fp['length'][i]), 'z': float(fp['z'][i]),
                             'n_edge': int(min(fp['n_left'][i], fp['n_right'][i])),
                             'lambda': float(fp['lambda'][i]), 'tf_overlap': int(fp['tf_overlap'][i]),
                             'source': 'tf' if fp['is_tf'][i] else 'nuc'})
                if int(fp['named'][i]):
                    rows.append({'type': 'nuc.size.named', 'start': int(fp['start'][i]), 'end': int(fp['end'][i]),
                                 'q1': int(fp['named'][i]),
                                 'q2': int(round(255 * float(fp['named_posterior'][i])))})
        for s in self.segments:
            rows.append({'type': 'nuc.seg', 'start': s.start, 'end': s.end, 'q1': s.code,
                         'q2': int(round(255 * s.posterior)), 'q_hat': s.q_hat, 'S': s.regularity,
                         'n_dyads': s.n_dyads, 'boundary_ci': list(s.boundary_ci),
                         'comp': 'sub_rich' if s.sub_rich else 'canonical'})
        return rows


def molecule_outputs(mol: NucMolecule, params: NucParams = DEFAULT_PARAMS,
                     cal: Optional[NucCalibration] = None, *, segments: bool = True) -> MoleculeResult:
    """Coarse (and named) calls, segments and scalars for one molecule."""
    if cal is None:
        raise ValueError('molecule_outputs needs a calibration record (calibrate first)')
    fp = footprint_calls(mol, params, cal)
    flags = {'too_short_nrl': True, 'too_short_regularity': True, 'too_short_snr': True,
             'too_short_segments': mol.length < params.seg_min_molecule_bp,
             'low_information': False}
    scal = {k: float('nan') for k in SCALARS}
    scal['rho_read'] = rho_read(mol)
    k, n = accessible_counts(mol, params)
    lam_acc = float(cal.prior.shrink(k, n))
    scal['lambda_acc'] = lam_acc
    flags['low_information'] = bool(cal.below_floor(lam_acc))
    segs: List[Segment] = []
    if fp is not None:
        nuc_codes = np.full(len(mol.nucs), CANT_TELL)
        sel = ~fp['is_tf']
        nuc_codes[fp['src_index'][sel]] = fp['coarse'][sel]
        sp, dy = molecule_spacing(mol, cal.nrl_pooled, params, coarse_codes=nuc_codes)
        scal.update({'nrl': sp.nrl, 'regularity': sp.regularity, 'spectral_snr': sp.spectral_snr,
                     'gap_cv': sp.gap_cv, 'n_dyads': float(sp.n_dyads)})
        flags.update({'too_short_nrl': sp.too_short_nrl, 'too_short_regularity': sp.too_short_regularity,
                      'too_short_snr': sp.too_short_snr})
        # class counts over every footprint, TF calls included (the small end
        # of the size readout lives in the TF calls); spacing uses nucleosomes only
        c = fp['coarse']
        scal['n_short'] = float((c == SHORT).sum())
        scal['n_mono'] = float((c == MONO).sum())
        scal['n_long'] = float((c == LONG).sum())
        scal['n_multi'] = float(((c == MULTI2) | (c == MULTI3)).sum())
        scal['n_cant_tell'] = float((c == CANT_TELL).sum())
        if segments and flags['low_information'] and not flags['too_short_segments']:
            # below the calibration's floor (D8): excluded from segmentation;
            # one low-information segment over its calls
            nuc_sel = ~fp['is_tf']
            if nuc_sel.any():
                segs = [lowinfo_segment(int(fp['start'][nuc_sel].min()), int(fp['end'][nuc_sel].max()))]
        elif segments:
            segs, too_short = segment_molecule(mol.length, mol.msps, fp, cal, params)
            flags['too_short_segments'] = too_short
            for s in segs:
                if s.state == REGULAR:
                    ssp, _ = molecule_spacing(mol, cal.nrl_pooled, params, coarse_codes=nuc_codes,
                                              window=(s.start, s.end + 1))
                    s.regularity = ssp.regularity
            if segs and not flags['too_short_segments']:
                cov = sum(s.end - s.start for s in segs)
                if cov > 0:
                    scal['reg_fraction'] = sum(s.end - s.start for s in segs if s.state == REGULAR) / cov
                    scal['sub_rich_fraction'] = sum(s.end - s.start for s in segs if s.sub_rich) / cov
    return MoleculeResult(mol.name, mol.length, fp, segs, scal, flags, calls_digest(mol),
                          cal.digest(), mol.record_key())
