"""Nucleosome module: sizes, spacing, phasing and array segments per molecule.

Pure functions on full molecule records (molecular coordinates). The public
API (NUC 5.1):

- :func:`iter_molecules` / :func:`decode_record`: BAM records -> :class:`NucMolecule`
- :func:`calibrate_self`, :func:`calibrate_thin`, :func:`calibrate_absolute` ->
  :class:`NucCalibration` (dataset level, once)
- :func:`molecule_outputs` -> :class:`MoleculeResult` (coarse/named size calls,
  segments, per-molecule scalars)
- :func:`window_metrics` -> :class:`WindowResult` (called within a peak, hit,
  BED row or selection, per :class:`Observation`)
- :func:`phasing` -> :class:`PhasingResult` (anchor and molecule frames,
  whole-period bins, molecule-level phase null)
- :func:`mark_acf`: the caller-free spacing check
- :func:`anchor_phase`, :func:`dephasing_test`: fuzzy / dephased loci at
  anchors against matched backgrounds
- :mod:`fiberhmm.nuc.disrupted`: the regular rule, undersized runs and missing
  nucleosomes; :mod:`fiberhmm.nuc.enrichment`: the recurrence test
- :mod:`fiberhmm.nuc.tags`: MA groups ``nucsz``/``nucseg``/``nucmol``

``condition_stats`` and ``association_stats`` belong to ``fiberhmm.assoc``
(feature hub M2/M3) and are not part of this release.
"""
from fiberhmm.nuc.params import DEFAULT_PARAMS, NucParams, parameter_schema
from fiberhmm.nuc.records import (DecodeContext, NucMolecule, decode_record, iter_molecules,
                                  make_molecule)

__all__ = [
    'NucParams', 'DEFAULT_PARAMS', 'parameter_schema', 'NucMolecule', 'DecodeContext',
    'decode_record', 'iter_molecules', 'make_molecule', 'calibrate_self', 'calibrate_thin',
    'calibrate_absolute', 'NucCalibration', 'molecule_outputs', 'MoleculeResult',
    'window_metrics', 'WindowResult', 'Observation', 'phasing', 'PhasingResult', 'mark_acf',
    'anchor_phase', 'dephasing_test', 'AnchorPhase',
]


def __getattr__(name):
    # lazy re-exports keep `import fiberhmm.nuc` light
    if name in ('calibrate_self', 'NucCalibration'):
        from fiberhmm.nuc import calibration as m
        return getattr(m, name)
    if name in ('calibrate_thin', 'calibrate_absolute'):
        from fiberhmm.nuc import thinning as m
        return getattr(m, name)
    if name in ('molecule_outputs', 'MoleculeResult'):
        from fiberhmm.nuc import molecule as m
        return getattr(m, name)
    if name in ('window_metrics', 'WindowResult', 'Observation'):
        from fiberhmm.nuc import window as m
        return getattr(m, name)
    if name in ('phasing', 'PhasingResult'):
        from fiberhmm.nuc import phasing as m
        return getattr(m, name)
    if name in ('anchor_phase', 'dephasing_test', 'AnchorPhase'):
        from fiberhmm.nuc import dephasing as m
        return getattr(m, name)
    if name == 'mark_acf':
        from fiberhmm.nuc.markacf import mark_acf
        return mark_acf
    raise AttributeError(name)
