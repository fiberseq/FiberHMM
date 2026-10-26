"""MA groups of the nucleosome module (NUC 5.1; feature-hub CONTRACTS 6).

Three groups, molecular coordinates, each self-contained:

``nucsz.QQQQ``  one interval per footprint (nucleosome and TF calls).
    Q1 = coarse code (0 can't tell, 1 short, 2 mono, 3 long, 4 multi2,
    5 multi3+, 6 edge-truncated) in the low 4 bits, +16 when the footprint is
    a TF call or overlaps one (``tf_overlap``); Q2 = named code (0 none,
    1 small, 2 hexasome, 3 octasome, 4 chromatosome; absolute calibration
    only); Q3 = round(255 x posterior); Q4 = clip(round(20 z + 128), 0, 254),
    255 = missing.
``nucseg.QQQ``  one interval per segment. Q1 = state (1 regular,
    2 irregular, 3 low-information; +8 sub-rich); Q2 = round(255 x
    posterior); Q3 = q-hat - 100 for regular segments, else 0.
``nucmol.Q x 20``  one whole-molecule interval with the per-molecule scalars
    (:data:`NUCMOL_CODECS`, CONTRACTS 6.4 codec: 1 byte value = offset +
    scale x q with 255 missing; 2 bytes big-endian with 65535 missing).

**Provenance.** Every AN name carries the digest of the calls the group was
computed from: ``nucsz@<digest8>_<i>``, ``nucseg@<digest8>_<i>``,
``nucmol@<digest8>``. Names are unique per interval (readers use AN names as
circular-piece ids). The digest is over call lengths and the gaps between
consecutive call starts (:func:`fiberhmm.nuc.molecule.calls_digest`), so it
survives lift (hard-clip fill rebases every group by the same offset). A
reader ignores a group whose digest no longer matches the record's calls.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

NUCSZ, NUCSEG, NUCMOL = 'nucsz', 'nucseg', 'nucmol'
NUC_GROUPS = (NUCSZ, NUCSEG, NUCMOL)
GROUP_VERSION = 1
TF_FLAG = 16

#: (name, nbytes, scale, offset, unit), in byte order
NUCMOL_CODECS: Tuple[Tuple[str, int, float, float, str], ...] = (
    ('nrl', 1, 1.0, 100.0, 'bp'),
    ('regularity', 1, 0.008, -1.0, 'acf'),
    ('spectral_snr', 1, 0.5, 0.0, 'ratio'),
    ('gap_cv', 1, 0.008, 0.0, 'ratio'),
    ('lambda_acc', 2, 2e-5, 0.0, 'fraction'),
    ('rho_read', 2, 2e-5, 0.0, 'fraction'),
    ('n_dyads', 2, 1.0, 0.0, 'count'),
    ('n_short', 1, 1.0, 0.0, 'count'),
    ('n_mono', 2, 1.0, 0.0, 'count'),
    ('n_long', 1, 1.0, 0.0, 'count'),
    ('n_multi', 1, 1.0, 0.0, 'count'),
    ('n_cant_tell', 2, 1.0, 0.0, 'count'),
    ('reg_fraction', 1, 0.004, 0.0, 'fraction'),
    ('sub_rich_fraction', 1, 0.004, 0.0, 'fraction'),
    ('flags', 1, 1.0, 0.0, 'bits'),
)
NUCMOL_SPEC = 'Q' * sum(c[1] for c in NUCMOL_CODECS)
#: bits of the ``flags`` scalar
FLAG_BITS = ('too_short_nrl', 'too_short_regularity', 'too_short_snr',
             'too_short_segments', 'low_information')
SPECS = {NUCSZ: 'QQQQ', NUCSEG: 'QQQ', NUCMOL: NUCMOL_SPEC}


# --- scalar codec (CONTRACTS 6.4, shared with fiberhmm.io.scan_ma) ----------------

from fiberhmm.io.scan_ma import decode_scalar, encode_scalar  # noqa: E402,F401


def encode_nucmol(scalars: Dict[str, float], flags: Dict[str, bool]) -> List[int]:
    bits = sum(1 << i for i, k in enumerate(FLAG_BITS) if flags.get(k))
    vals = dict(scalars)
    vals['flags'] = float(bits)
    out: List[int] = []
    for name, nb, sc, off, _u in NUCMOL_CODECS:
        out.extend(encode_scalar(vals.get(name), nb, sc, off))
    return out


def decode_nucmol(qbytes: Sequence[int]) -> Tuple[Dict[str, Optional[float]], Dict[str, bool]]:
    vals: Dict[str, Optional[float]] = {}
    i = 0
    for name, nb, sc, off, _u in NUCMOL_CODECS:
        vals[name] = decode_scalar(qbytes[i:i + nb], nb, sc, off)
        i += nb
    bits = int(vals.pop('flags') or 0)
    return vals, {k: bool(bits >> i & 1) for i, k in enumerate(FLAG_BITS)}


# --- rows -----------------------------------------------------------------------

def _z_byte(z: float) -> int:
    if z is None or not np.isfinite(z):
        return 255
    return int(max(0, min(254, round(20.0 * z + 128.0))))


def group_rows(result) -> Dict[str, Tuple[str, List[Tuple[int, int, List[int], str]]]]:
    """``{group: (spec, [(mol_start, length, qbytes, an_name), ...])}`` for a
    :class:`~fiberhmm.nuc.molecule.MoleculeResult`."""
    dg = result.digest
    out: Dict[str, Tuple[str, List]] = {}
    fp = result.footprints
    rows = []
    if fp is not None:
        for i in range(len(fp['start'])):
            q1 = int(fp['coarse'][i]) | (TF_FLAG if bool(fp['tf_overlap'][i]) else 0)
            rows.append((int(fp['start'][i]), int(fp['end'][i] - fp['start'][i]),
                         [q1, int(fp['named'][i]), int(round(255 * float(fp['posterior'][i]))),
                          _z_byte(float(fp['z'][i]))], ''))
    rows.sort(key=lambda r: (r[0], r[1]))
    out[NUCSZ] = (SPECS[NUCSZ], [(s, n, q, f'{NUCSZ}@{dg}_{k}') for k, (s, n, q, _a) in enumerate(rows)])
    seg_rows = []
    for k, s in enumerate(sorted(result.segments, key=lambda x: (x.start, x.end))):
        q3 = int(max(0, min(255, round(s.q_hat - 100)))) if s.q_hat is not None else 0
        seg_rows.append((int(s.start), int(s.end - s.start),
                         [int(s.code), int(round(255 * float(s.posterior))), q3], f'{NUCSEG}@{dg}_{k}'))
    out[NUCSEG] = (SPECS[NUCSEG], seg_rows)
    out[NUCMOL] = (SPECS[NUCMOL], [(0, int(result.length), encode_nucmol(result.scalars, result.flags),
                                    f'{NUCMOL}@{dg}')])
    return out


# --- writing ----------------------------------------------------------------------

def replace_groups(read, groups: Dict[str, Tuple[str, Sequence[Tuple[int, int, Sequence[int], str]]]]) -> int:
    """Replace (or add) MA groups on ``read`` in place (the scan appender:
    other groups are kept verbatim with their AQ bytes and AN names)."""
    from fiberhmm.io.scan_ma import replace_scan_groups
    return replace_scan_groups(read, groups)


def annotate_record(read, result) -> int:
    """Write ``result``'s nucsz/nucseg/nucmol groups onto ``read``.

    The record's annotations must already be in molecular frame (FiberHMM
    3.x output, whose header declares coord=molecular)."""
    return replace_groups(read, group_rows(result))


def ma_group_declarations() -> List[dict]:
    """The three groups as CONTRACTS 3 ``MAGroupDecl``-shaped dicts."""
    from fiberhmm.nuc.sizes import COARSE_NAMES
    return [
        {'group': NUCSZ, 'group_version': GROUP_VERSION, 'kind': 'mark',
         'quality_names': ['coarse', 'named', 'posterior', 'z'],
         'state_codes': {str(k): v for k, v in COARSE_NAMES.items()}, 'codecs': [],
         'an': 'nucsz@<calls digest8>_<index>'},
        {'group': NUCSEG, 'group_version': GROUP_VERSION, 'kind': 'mark',
         'quality_names': ['state', 'posterior', 'q_hat_minus_100'],
         'state_codes': {'1': 'regular', '2': 'irregular', '3': 'low_information',
                         '9': 'regular+sub_rich', '10': 'irregular+sub_rich',
                         '11': 'low_information+sub_rich'}, 'codecs': [],
         'an': 'nucseg@<calls digest8>_<index>'},
        {'group': NUCMOL, 'group_version': GROUP_VERSION, 'kind': 'scalar',
         'quality_names': [f'{n}' if nb == 1 else f'{n}[{k}]'
                           for n, nb, *_ in NUCMOL_CODECS for k in range(nb)],
         'state_codes': {},
         'codecs': [{'name': n, 'nbytes': nb, 'scale': sc, 'offset': off, 'unit': u}
                    for n, nb, sc, off, u in NUCMOL_CODECS],
         'an': 'nucmol@<calls digest8>'},
    ]


def contract_lines(calibration_digest: str = '', params_digest: str = '',
                   calls_program: str = '', module_version: str = '1.0',
                   scan_id: str = '') -> List[str]:
    """``@CO FIBERHMM-SCAN-MA:v1:`` lines for the three groups."""
    from fiberhmm.io.scan_ma import scan_contract_line
    return [scan_contract_line(d, module='nuc', module_version=module_version, scan_id=scan_id,
                               settings_hash=params_digest[:16], calibration=calibration_digest[:16],
                               coordinates='molecular', calls_program=calls_program)
            for d in ma_group_declarations()]


def annotate_header(header, calibration_digest: str = '', params_digest: str = '',
                    calls_program: str = ''):
    """Header with the groups declared (MA-TYPES and contract lines) and the
    molecular-frame marker; earlier nuc contract lines are replaced."""
    import pysam
    from fiberhmm.io.bam_header import append_coord_marker, append_ma_types
    from fiberhmm.io.scan_ma import with_scan_contracts
    header = append_coord_marker(append_ma_types(header, NUC_GROUPS))
    d = with_scan_contracts(header.to_dict(), contract_lines(calibration_digest, params_digest,
                                                             calls_program))
    return pysam.AlignmentHeader.from_dict(d)


# --- reading --------------------------------------------------------------------------

def _digest(mol) -> str:
    from fiberhmm.nuc.molecule import calls_digest
    return calls_digest(mol)


class _Calls:
    def __init__(self, nucs, tfs, msps):
        from fiberhmm.nuc.records import _as_intervals
        self.nucs, self.tfs, self.msps = _as_intervals(nucs), _as_intervals(tfs), _as_intervals(msps)


def record_calls_digest(read) -> Optional[str]:
    """Digest of a record's current nuc/tf/msp calls, read from MA alone (no
    marks needed, so hard-clipped records work). None without calls or when
    the header leaves the frame undecided."""
    from fiberhmm.nuc.records import DecodeContext, _footprints_from_legacy, _footprints_from_ma
    from fiberhmm.io.ma_tags import annotation_tags
    hdr = getattr(read, 'header', None)
    if hdr is None:
        return None
    ctx = DecodeContext(hdr)
    tags = annotation_tags(read)
    fp = None
    if tags is not None:
        length = int(str(tags[0]).split(';', 1)[0])
        fp = _footprints_from_ma(read, ctx, length)
    if fp is None:          # no MA, or an MA without footprint groups
        fp = _footprints_from_legacy(read, ctx, int(read.query_length or 0))
    if fp is None or isinstance(fp, str):
        return None
    return _digest(_Calls(fp['nuc'], fp['tf'], fp['msp']))


def read_nuc_groups(read, mol=None, check_digest: bool = True) -> Dict[str, dict]:
    """Parse the nuc groups of one record.

    Returns ``{group: {'intervals': [(start, length)], 'q': [[...]],
    'digest': str, 'stale': bool}}`` plus decoded scalars under
    ``nucmol['scalars']``/``['flags']``. ``stale`` is True when the digest in
    the AN names differs from the record's current calls (pass the decoded
    ``mol`` to avoid decoding it again)."""
    from fiberhmm.io.ma_tags import annotation_tags, parse_an_tag, parse_aq_array, parse_ma_tag
    tags = annotation_tags(read)
    if tags is None:
        return {}
    ma, aq, an, _src = tags
    parsed = parse_ma_tag(str(ma))
    types = parsed['raw_types']
    specs = [t[2] for t in types]
    counts = [len(t[3]) for t in types]
    quals = parse_aq_array(aq, specs, counts) if aq is not None else [[] for _ in range(sum(counts))]
    names = parse_an_tag(str(an)) if an is not None else [''] * sum(counts)
    out: Dict[str, dict] = {}
    k = 0
    for name, _st, spec, iv in types:
        q = quals[k:k + len(iv)]
        nm = names[k:k + len(iv)]
        k += len(iv)
        if name not in NUC_GROUPS:
            continue
        digests = {n.split('@', 1)[1].split('_', 1)[0] for n in nm if '@' in n}
        out[name] = {'intervals': list(iv), 'q': q, 'digest': digests.pop() if len(digests) == 1 else None,
                     'stale': False}
    if not out:
        return out
    if check_digest:
        current = record_calls_digest(read) if mol is None else _digest(mol)
        for g in out.values():
            g['stale'] = current is None or g['digest'] != current
    if NUCMOL in out and out[NUCMOL]['q']:
        vals, flags = decode_nucmol(out[NUCMOL]['q'][0])
        out[NUCMOL]['scalars'] = vals
        out[NUCMOL]['flags'] = flags
    return out
