"""Calibration records (NUC 4).

Every dataset-level quantity the per-molecule outputs depend on lives in one
:class:`NucCalibration` record, so ``molecule_outputs(molecule, params, cal)``
depends on nothing else:

- the calibration population (molecule-name hashes), its source
  (``self`` / ``thin`` / ``absolute``) and the caller settings;
- NRL_pooled, the lambda_acc beta prior, the reference curve
  (mu_mono, sigma_mono knots), sigma_gap, the empirical gap density and the
  dyad-gap model (regular jitter, irregular spread, missing-dyad weights);
- segmentation composition frequencies;
- thinning: sigma_loc knots, outcome rates, overshoot, supported lambda range,
  held-out validation and the low-information floor (D8);
- absolute controls: class locations T_k and b(lambda); naked-DNA nulls.

``self`` calibration fits only what the data identify without a control or
thinning (sigma_loc = 0, every lambda supported, no floor).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from fiberhmm.nuc.density import (FLAT_PRIOR, BetaPrior, accessible_counts, edge_truncated,
                                  fit_beta_prior, footprint_lambdas)
from fiberhmm.nuc.params import DEFAULT_PARAMS, NucParams
from fiberhmm.nuc.records import NucMolecule
from fiberhmm.nuc.sizes import AbsoluteClasses, ReferenceCurve, fit_reference_curve
from fiberhmm.nuc.spacing import dataset_period, eligible_dyads, mono_range

SCHEMA = 'fiberhmm.nuc.calibration.v1'


def molecule_hash(name: str) -> str:
    """Stable 16-hex hash of a molecule name (calibration population ids)."""
    return hashlib.sha256(str(name).encode()).hexdigest()[:16]


def hash_rank_sample(molecules: Sequence[NucMolecule], n: int) -> List[NucMolecule]:
    """Up to ``n`` molecules with the lowest name hashes (stable, viewport-free)."""
    keyed = sorted(molecules, key=lambda m: molecule_hash(m.name))
    return keyed[:n] if n and len(keyed) > n else keyed


@dataclass
class NucCalibration:
    source: str = 'self'
    label: str = ''
    chemistry: Dict[str, Optional[str]] = field(default_factory=dict)
    fiberhmm_version: str = ''
    params_digest: str = ''
    caller: Dict = field(default_factory=dict)
    population: List[str] = field(default_factory=list)
    nrl_bootstrap: float = float('nan')
    nrl_pooled: float = 189.0
    prior: BetaPrior = FLAT_PRIOR
    curve: Optional[ReferenceCurve] = None
    sigma_gap: float = 15.0
    gap_density: Dict[str, List[float]] = field(default_factory=dict)
    gap_model: Dict = field(default_factory=dict)
    composition: Dict[str, List[float]] = field(default_factory=dict)
    sigma_loc: Dict[str, List[float]] = field(default_factory=dict)   # log_lambda, sigma
    overshoot: Dict[str, List[float]] = field(default_factory=dict)   # log_lambda, shift
    outcomes: Dict[str, List] = field(default_factory=dict)
    supported: Optional[List[float]] = None                           # [lambda_lo, lambda_hi]
    lambda_floor: Optional[float] = None
    validation: Dict = field(default_factory=dict)
    absolute: Optional[AbsoluteClasses] = None
    naked: Dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    # --- evaluation -----------------------------------------------------------
    def sigma_loc_at(self, lam) -> np.ndarray:
        lam = np.asarray(lam, dtype=float)
        if not self.sigma_loc:
            return np.zeros(lam.shape)
        x = np.log(np.clip(lam, 1e-6, 1.0))
        return np.interp(x, self.sigma_loc['log_lambda'], self.sigma_loc['sigma'])

    def is_supported(self, lam) -> np.ndarray:
        """Within the supported lambda range (all True when the record sets
        none, e.g. self calibration; all False for an empty range)."""
        lam = np.asarray(lam, dtype=float)
        ok = np.ones(lam.shape, dtype=bool)
        if self.supported is not None:
            lo, hi = self.supported
            if hi <= lo:
                return np.zeros(lam.shape, dtype=bool)
            ok &= (lam >= lo) & (lam <= hi)
        return ok

    def below_floor(self, lam_acc: float) -> bool:
        return self.lambda_floor is not None and np.isfinite(lam_acc) and lam_acc < self.lambda_floor

    @property
    def named_available(self) -> bool:
        return self.source == 'absolute' and self.absolute is not None

    # --- serialisation ----------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            'schema': SCHEMA, 'source': self.source, 'label': self.label,
            'chemistry': dict(self.chemistry), 'fiberhmm_version': self.fiberhmm_version,
            'params_digest': self.params_digest, 'caller': dict(self.caller),
            'population': list(self.population),
            'nrl_bootstrap': _f(self.nrl_bootstrap), 'nrl_pooled': _f(self.nrl_pooled),
            'prior': self.prior.to_dict(),
            'curve': self.curve.to_dict() if self.curve else None,
            'sigma_gap': _f(self.sigma_gap), 'gap_density': _clean(self.gap_density),
            'gap_model': _clean(self.gap_model),
            'composition': _clean(self.composition), 'sigma_loc': _clean(self.sigma_loc),
            'overshoot': _clean(self.overshoot), 'outcomes': _clean(self.outcomes),
            'supported': None if self.supported is None else [float(x) for x in self.supported],
            'lambda_floor': None if self.lambda_floor is None else float(self.lambda_floor),
            'validation': _clean(self.validation),
            'absolute': self.absolute.to_dict() if self.absolute else None,
            'naked': _clean(self.naked), 'notes': list(self.notes),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NucCalibration":
        if d.get('schema') != SCHEMA:
            raise ValueError(f"not a nuc calibration record: schema {d.get('schema')!r}")
        return cls(
            source=d['source'], label=d.get('label', ''), chemistry=d.get('chemistry', {}),
            fiberhmm_version=d.get('fiberhmm_version', ''), params_digest=d.get('params_digest', ''),
            caller=d.get('caller', {}), population=list(d.get('population', [])),
            nrl_bootstrap=_nan(d.get('nrl_bootstrap')), nrl_pooled=float(d['nrl_pooled']),
            prior=BetaPrior.from_dict(d['prior']),
            curve=ReferenceCurve.from_dict(d['curve']) if d.get('curve') else None,
            sigma_gap=float(d.get('sigma_gap', 15.0)), gap_density=d.get('gap_density', {}),
            gap_model=d.get('gap_model', {}),
            composition=d.get('composition', {}), sigma_loc=d.get('sigma_loc', {}),
            overshoot=d.get('overshoot', {}), outcomes=d.get('outcomes', {}),
            supported=d.get('supported'), lambda_floor=d.get('lambda_floor'),
            validation=d.get('validation', {}),
            absolute=AbsoluteClasses.from_dict(d['absolute']) if d.get('absolute') else None,
            naked=d.get('naked', {}), notes=list(d.get('notes', [])),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'))

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def save(self, path: str) -> str:
        with open(path, 'w') as fh:
            fh.write(json.dumps(self.to_dict(), sort_keys=True, indent=1))
        return self.digest()

    @classmethod
    def load(cls, path: str) -> "NucCalibration":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


def _f(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def _nan(x):
    return float('nan') if x is None else float(x)


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _version() -> str:
    try:
        from importlib.metadata import version
        return version('fiberhmm')
    except Exception:
        return 'unknown'


def footprint_table(molecules: Iterable[NucMolecule], cal: NucCalibration,
                    params: NucParams = DEFAULT_PARAMS) -> Dict[str, np.ndarray]:
    """Per-footprint arrays (length, lambda_i, truncated, molecule lambda_acc)
    for every nucleosome call in ``molecules``."""
    Ls, lams, trunc, lacc = [], [], [], []
    for m in molecules:
        if len(m.nucs) == 0:
            continue
        k, n = accessible_counts(m, params)
        la = float(cal.prior.shrink(k, n))
        li = footprint_lambdas(m, m.nucs, la, params, cal.prior)
        Ls.append(m.nucs[:, 1] - m.nucs[:, 0])
        lams.append(li)
        trunc.append(edge_truncated(m, m.nucs))
        lacc.append(np.full(len(m.nucs), la))
    if not Ls:
        z = np.zeros(0)
        return {'length': z, 'lambda': z, 'truncated': z.astype(bool), 'lambda_acc': z}
    return {'length': np.concatenate(Ls), 'lambda': np.concatenate(lams),
            'truncated': np.concatenate(trunc), 'lambda_acc': np.concatenate(lacc)}


def calibrate_self(molecules: Sequence[NucMolecule], params: NucParams = DEFAULT_PARAMS, *,
                   label: str = '', chemistry: Optional[dict] = None,
                   max_molecules: int = 0, fit_composition: bool = True) -> NucCalibration:
    """Dataset calibration without thinning or controls.

    The population is every molecule given, or the ``max_molecules`` with the
    lowest name hashes. NRL_pooled is bootstrapped (90-230 bp, then 0.45-1.25 x
    P0).
    """
    # canonical (hash) order: float sums, and so the record, do not depend on input order
    pop = hash_rank_sample(list(molecules), max_molecules or 0)
    cal = NucCalibration(source='self', label=label, chemistry=dict(chemistry or {}),
                         fiberhmm_version=_version(), params_digest=params.digest(),
                         population=sorted(molecule_hash(m.name) for m in pop))
    ks, ns = zip(*[accessible_counts(m, params) for m in pop]) if pop else ((), ())
    cal.prior = fit_beta_prior(ks, ns)
    p0 = dataset_period([_nrl(m, None, params) for m in pop])
    p1 = dataset_period([_nrl(m, p0, params) for m in pop])
    cal.nrl_bootstrap = p0
    cal.nrl_pooled = p1 if np.isfinite(p1) else (p0 if np.isfinite(p0) else 189.0)
    ft = footprint_table(pop, cal, params)
    keep = ~ft['truncated']
    cal.curve = fit_reference_curve(ft['length'][keep], ft['lambda'][keep], cal.nrl_pooled, params)
    gaps = np.concatenate([eligible_dyads(m, cal.nrl_pooled, params).gaps for m in pop]) if pop else np.zeros(0)
    if len(gaps) >= 20:
        mono = gaps[(gaps > 0.7 * cal.nrl_pooled) & (gaps < 1.3 * cal.nrl_pooled)]
        cal.sigma_gap = float(1.4826 * np.median(np.abs(mono - np.median(mono)))) if len(mono) else 15.0
        g = gaps[gaps < params.seg_max_gap]
        edges = np.arange(0, params.seg_max_gap + 5, 5)
        h, _ = np.histogram(g, bins=edges)
        dens = (h + 0.5) / (h.sum() + 0.5 * len(h)) / 5.0
        cal.gap_density = {'edges': edges.tolist(), 'density': dens.tolist()}
    from fiberhmm.nuc.segment import fit_gap_model
    cal.gap_model = fit_gap_model(pop, cal.nrl_pooled, params)
    if fit_composition:
        from fiberhmm.nuc.segment import fit_composition as _fit
        cal.composition = _fit(pop, cal, params)
    return cal


def _nrl(m, nrl_pooled, params):
    g = eligible_dyads(m, nrl_pooled, params).gaps
    return float(np.median(g)) if len(g) >= params.min_gaps_nrl else float('nan')
