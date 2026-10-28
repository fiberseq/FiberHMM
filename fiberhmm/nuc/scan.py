"""Shared per-record scan outputs and reference-coordinate recurrence tracks."""
from __future__ import annotations

import numpy as np

from .params import DEFAULT_PARAMS


def record_outputs(read, record, cal, *, params=DEFAULT_PARAMS, context=None):
    """Return (decoded molecule, nuc result, scan result) for a complete record.

    ``record`` is its scan RecordRef; ``cal`` is a NucCalibration. A cached
    DecodeContext may be supplied. Unusable records return (None, None, None).
    Annotation frames are resolved from the record header as in decode_record.
    """
    from .records import DecodeContext, decode_record
    from .molecule import molecule_outputs
    from .tags import NUCMOL, FLAG_BITS, digest_pieces, group_rows
    from fiberhmm.scan.types import Mark, MoleculeResult
    from fiberhmm.scan.modules.nuc import _ref_envelope
    if context is None:
        context = DecodeContext(read.header, params=params)
    mol, _why = decode_record(read, context)
    if mol is None:
        return None, None, None
    result = molecule_outputs(mol, params, cal)
    marks = []
    for group, (_spec, rows) in group_rows(result).items():
        if group == NUCMOL:
            continue
        for start, length, q, name in rows:
            a, b = _ref_envelope(mol, start, start + length)
            marks.append(Mark(record, group, record.chrom, a, b, tuple(int(x) for x in q),
                              name, (int(start), int(length))))
    scalars = {k: None if v is None or not np.isfinite(v) else float(v) for k, v in result.scalars.items()}
    scalars['flags'] = float(sum(1 << i for i, k in enumerate(FLAG_BITS) if result.flags.get(k)))
    scalars['calls_digest_hi'], scalars['calls_digest_lo'] = digest_pieces(result.digest)
    return mol, result, MoleculeResult(record, scalars, marks)


def measured_track(mol, result):
    """Return (measured intervals, kind -> intervals) in reference coordinates.

    This is the patch recurrence-test input before clipping to a scan unit.
    Unusable/low-information molecules have no measured intervals. Kind keys
    are reg, dis, und, low, uds and mis; intervals are half-open.
    """
    from .segment import DISORDERED, LOWINFO, REGULAR, UNDETERMINED
    from fiberhmm.scan.modules.nuc import _union_iv, _intersect_iv, _subtract_iv
    kinds = {k: [] for k in ('reg', 'dis', 'und', 'low', 'uds', 'mis')}
    if result.flags.get('low_information') or not result.segments:
        return [], kinds
    state_key = {REGULAR: 'reg', DISORDERED: 'dis', UNDETERMINED: 'und', LOWINFO: 'low'}
    for segment in result.segments:
        env = mol.mol_interval_to_ref(segment.start, segment.end)
        if env is not None and env[1] > env[0]:
            kinds[state_key[segment.state]].append((int(env[0]), int(env[1])))
    for event in result.events:
        env = mol.mol_interval_to_ref(event.start, event.end)
        if env is not None and env[1] > env[0]:
            kinds['uds' if event.kind == 16 else 'mis'].append((int(env[0]), int(env[1])))
    blocks = [(int(r), int(r + n)) for _q, r, n in mol.blocks] if mol.blocks is not None else []
    segments = _union_iv(kinds['reg'] + kinds['dis'] + kinds['und'])
    measured = _subtract_iv(_intersect_iv(_union_iv(blocks), segments), kinds['low'])
    return measured, kinds
