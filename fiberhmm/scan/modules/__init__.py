"""Scan modules: the ``ScanModule`` protocol, a base class and the registry.

A module wraps one existing FiberHMM engine (``fp``: the lattice recaller,
``nfr``: the accessibility engine) or a ported one (``peaks``). The
orchestrator (:mod:`fiberhmm.scan.runner`) plans units, calls ``run_unit`` per
unit, reconciles seams and writes the scan directory. Signatures are part of
the scan contract (``docs/reference/scan-format.md``).
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence

from ..types import (FeatureTypeDecl, FullMolecule, MAGroupDecl, MoleculeResult, ScanInputs, ScanUnit,
                     StitchResult, UnitResult)

# name -> "module.path:ClassName" (imported lazily: engines pull in numba/sklearn)
MODULES: Dict[str, str] = {
    'peaks': 'fiberhmm.scan.modules.peaks:PeaksModule',
    'fp': 'fiberhmm.scan.modules.fp:FpModule',
    'nfr': 'fiberhmm.scan.modules.nfr:NfrModule',
    'nuc': 'fiberhmm.scan.modules.nuc:NucModule',
}
# Order within a unit when several modules run (dependencies first).
ORDER = ('peaks', 'fp', 'nfr', 'nuc')


class ScanModuleBase:
    """Defaults for the ``ScanModule`` protocol; concrete modules override what they use."""
    name: ClassVar[str] = ''
    version: ClassVar[str] = '0'
    experimental: ClassVar[bool] = False
    reconcile: ClassVar[str] = 'anchor'          # 'anchor' | 'stitch'
    max_support_bp: ClassVar[int] = 12_500
    wants_molecules: ClassVar[bool] = False
    # 'molecular': molecule_outputs gets the record with its annotations carried into molecular frame (inputs
    # without a coord=molecular declaration are carried by the orchestrator); 'source': the record as stored,
    # for a module that resolves each annotation's frame from the record's header itself (nuc)
    molecule_frame: ClassVar[str] = 'molecular'
    native_tables: ClassVar[Sequence[str]] = ()
    state_groups: ClassVar[Mapping[str, str]] = {}    # hit type -> MA state group written for it

    def parameter_schema(self) -> dict:
        return {}

    def resolve(self, params: Mapping[str, Any]) -> dict:
        """Validated settings with every default filled in; this dict is hashed."""
        params = dict(params or {})
        schema = self.parameter_schema()
        unknown = sorted(set(params) - set(schema) - {'caps'})
        if unknown:
            raise ValueError(f'{self.name}: unknown parameter(s): {", ".join(unknown)}')
        out = {k: v.get('default') for k, v in schema.items()}
        out.update({k: v for k, v in params.items() if k != 'caps'})
        return out

    def feature_types(self) -> List[FeatureTypeDecl]:
        return []

    def ma_groups(self) -> List[MAGroupDecl]:
        return []

    def needs(self, settings: Mapping[str, Any]) -> set:
        return set()

    def load_policy(self, settings: Mapping[str, Any]) -> dict:
        return {}

    def calibrate(self, inputs: ScanInputs, units: Sequence[ScanUnit], settings: Mapping[str, Any],
                  work: Path, progress) -> Optional[dict]:
        return None

    def run_unit(self, unit: ScanUnit, inputs: ScanInputs, settings: Mapping[str, Any],
                 calibration: Optional[dict], upstream: Mapping[str, UnitResult],
                 work: Path, progress) -> UnitResult:
        raise NotImplementedError

    def molecule_outputs(self, molecule: FullMolecule, settings: Mapping[str, Any],
                         calibration: Optional[dict]) -> Optional[MoleculeResult]:
        return None

    def stitch_support_bp(self, settings: Mapping[str, Any]) -> Optional[int]:
        """Zero context a stitch run needs beyond its data (None: one adjacent empty zone)."""
        return None

    def stitch(self, chrom: str, territory: Sequence[tuple], tracks: Mapping[str, Any],
               settings: Mapping[str, Any], calibration: Optional[dict]) -> StitchResult:
        return StitchResult()

    def calibration_uses_plan(self, settings: Mapping[str, Any]) -> bool:
        """Whether ``calibrate`` depends on the plan (then its cached record is reused only for the same plan).
        A module whose record depends only on the inputs and settings returns False, so extending a scan with
        new units keeps the record (and every unit key that contains its digest)."""
        return True

    # ---- scan-wide statistics for stitch modules (two passes over the contigs)
    def scan_statistics_tracks(self, settings: Mapping[str, Any]) -> Optional[Sequence[str]]:
        """Track names the first pass needs; None: this module has no scan-wide statistics."""
        return None

    def stitch_statistics(self, chrom: str, territory: Sequence[tuple], tracks: Mapping[str, Any],
                          settings: Mapping[str, Any], calibration: Optional[dict]) -> Any:
        """First pass, per contig (only the tracks named by ``scan_statistics_tracks``)."""
        return None

    def combine_statistics(self, per_contig: Mapping[str, Any], settings: Mapping[str, Any],
                           calibration: Optional[dict]) -> Optional[dict]:
        """The scan-wide record from every contig's first-pass result (JSON-able: it goes into the manifest's
        module block as ``scan_statistics``, and ``stitch`` receives it as ``scan_statistics=``)."""
        return None

    # ---- helpers for the manifest
    def hit_columns(self) -> List[str]:
        """Union of the declared module columns over this module's feature types, in declaration order."""
        cols: List[str] = []
        for t in self.feature_types():
            for c in t.columns:
                if c not in cols:
                    cols.append(c)
        return cols

    def state_columns(self) -> List[str]:
        return []

    def scalar_names(self) -> List[str]:
        names: List[str] = []
        for g in self.ma_groups():
            if g.kind == 'scalar':
                names.extend(c.name for c in g.codecs)
        return names


def load_module(name: str, registry: Optional[Mapping[str, Any]] = None):
    """Instantiate a registered module (``registry`` may map names to classes or 'mod:Class' strings)."""
    table = dict(MODULES)
    if registry:
        table.update(registry)
    if name not in table:
        raise ValueError(f'unknown scan module {name!r} (available: {", ".join(sorted(table))})')
    target = table[name]
    if isinstance(target, str):
        mod_path, cls_name = target.split(':')
        target = getattr(importlib.import_module(mod_path), cls_name)
    return target()


def available_modules(registry: Optional[Mapping[str, Any]] = None) -> List[str]:
    table = dict(MODULES)
    if registry:
        table.update(registry)
    return sorted(table, key=lambda n: (ORDER.index(n) if n in ORDER else len(ORDER), n))
