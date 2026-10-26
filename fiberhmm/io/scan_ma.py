"""MA groups written by ``fiberhmm-scan``: registry, header contract, scalar codec and the appender.

A scan writes its per-molecule outputs as extra groups of the read's
Molecular-annotation tags (``MA``/``AQ``/``AN``), in molecular coordinates:

- *state* groups: one interval per (hit, record) with a state byte first
  (``fpcls``, ``nfrvar``); AN holds the hit id;
- *mark* groups: molecule-wide interval marks (``nucsz``, ``nucseg``);
- *scalar* groups: one whole-molecule interval per record whose bytes encode
  per-molecule scalars by declared codecs (``nfrmol``, ``nucmol``).

Each written group is declared in ``MA-TYPES:v1`` and in one
``@CO FIBERHMM-SCAN-MA:v1:<json>`` line. This module is dependency-light
(pysam only where a header is rebuilt) so the call writers can import the
registry. Contract: ``docs/reference/scan-format.md``.
"""
from __future__ import annotations

import array
import json
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCAN_CONTRACT_PREFIX = 'FIBERHMM-SCAN-MA:v1:'

# Registered scan groups (name -> producing module). Every writer that
# regenerates nuc/msp/tf calls drops these groups (they describe old calls).
SCAN_MA_GROUPS = {
    'fpcls': 'fp',
    'nfrvar': 'nfr',
    'nfrmol': 'nfr',
    'nucsz': 'nuc',
    'nucseg': 'nuc',
    'nucmol': 'nuc',
}

_SCAN_NAME_RE = re.compile(r'^[a-z][a-z0-9]{1,15}$')
MISSING_1 = 255
MISSING_2 = 65535


def is_scan_group_name(name: str, module: Optional[str] = None) -> bool:
    """A valid scan group name (``<module><kind>``, lowercase alphanumeric)."""
    if not _SCAN_NAME_RE.fullmatch(str(name)):
        return False
    return module is None or str(name).startswith(module)


# ------------------------------------------------------------------ scalar codec
def encode_scalar(value, nbytes: int, scale: float, offset: float) -> List[int]:
    """Bytes of one scalar: ``q = round((value - offset) / scale)`` clipped; missing -> 255 / 65535."""
    top = 254 if nbytes == 1 else 65534
    if value is None or value != value:
        q = MISSING_1 if nbytes == 1 else MISSING_2
    else:
        q = int(round((float(value) - float(offset)) / float(scale)))
        q = min(top, max(0, q))
    if nbytes == 1:
        return [q]
    return [q >> 8, q & 0xFF]


def decode_scalar(qbytes: Sequence[int], nbytes: int, scale: float, offset: float):
    """Inverse of :func:`encode_scalar`; None for the missing code."""
    if nbytes == 1:
        q = int(qbytes[0])
        return None if q == MISSING_1 else offset + scale * q
    q = 256 * int(qbytes[0]) + int(qbytes[1])
    return None if q == MISSING_2 else offset + scale * q


def encode_scalars(values: Mapping[str, Optional[float]], codecs: Sequence) -> List[int]:
    """Bytes of a scalar group in codec order (codecs: ScalarCodec or dicts with name/nbytes/scale/offset)."""
    out: List[int] = []
    for c in codecs:
        c = c if isinstance(c, dict) else c.to_json()
        out.extend(encode_scalar(values.get(c['name']), int(c['nbytes']), float(c['scale']), float(c['offset'])))
    return out


def decode_scalars(qbytes: Sequence[int], codecs: Sequence) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    i = 0
    for c in codecs:
        c = c if isinstance(c, dict) else c.to_json()
        n = int(c['nbytes'])
        out[c['name']] = decode_scalar(qbytes[i:i + n], n, float(c['scale']), float(c['offset']))
        i += n
    return out


# ------------------------------------------------------------------ header contract
def scan_contract_line(decl: Mapping, **provenance) -> str:
    """The ``@CO`` contract line of one group (decl: MAGroupDecl.to_json() plus provenance)."""
    value = dict(decl)
    value.update(provenance)
    value.setdefault('coordinates', 'molecular')
    return SCAN_CONTRACT_PREFIX + json.dumps(value, sort_keys=True, separators=(',', ':'))


def scan_contracts(header) -> Dict[str, dict]:
    """group -> its contract (the last valid line wins) from a header (pysam header or dict)."""
    hd = header if isinstance(header, dict) else header.to_dict()
    out: Dict[str, dict] = {}
    for comment in hd.get('CO', []) or []:
        text = str(comment)
        if not text.startswith(SCAN_CONTRACT_PREFIX):
            continue
        try:
            value = json.loads(text[len(SCAN_CONTRACT_PREFIX):])
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get('group'), str):
            out[value['group']] = value
    return out


def scan_molecular_groups(header) -> frozenset:
    """Groups a scan contract declares molecular (every scan group is)."""
    return frozenset(g for g, v in scan_contracts(header).items() if v.get('coordinates', 'molecular') == 'molecular')


def strip_scan_contracts(header_dict: dict, groups: Iterable[str]) -> dict:
    """``header_dict`` without the contract lines of ``groups`` (a new dict)."""
    drop = set(groups)
    out = dict(header_dict)
    kept = []
    for comment in header_dict.get('CO', []) or []:
        text = str(comment)
        if text.startswith(SCAN_CONTRACT_PREFIX):
            try:
                if json.loads(text[len(SCAN_CONTRACT_PREFIX):]).get('group') in drop:
                    continue
            except ValueError:
                pass
        kept.append(comment)
    out['CO'] = kept
    return out


def with_scan_contracts(header_dict: dict, lines: Sequence[str]) -> dict:
    """Replace the contract lines of the groups ``lines`` declare, keeping every other comment."""
    groups = [json.loads(line[len(SCAN_CONTRACT_PREFIX):])['group'] for line in lines]
    out = strip_scan_contracts(header_dict, groups)
    out['CO'] = list(out.get('CO', [])) + list(lines)
    return out


# ------------------------------------------------------------------ MA editing
def _legacy_as_ma(read):
    """(ma, aq) carrying a record's legacy ns/nl(+nq) and as/al calls as MA nuc/msp groups, or None.

    A record without MA/Ma whose calls are only in legacy tags must not gain an MA that holds only scan
    groups: readers take MA as authoritative and would stop reading its ns/nl. The legacy tags must already
    be in molecular frame (the scan writer carries every record into molecular frame first)."""
    length = int(read.query_length or len(read.query_sequence or ''))
    parts, aq = [str(length)], []

    def pairs(start_tag, len_tag):
        if not (read.has_tag(start_tag) and read.has_tag(len_tag)):
            return None
        try:
            starts = [int(x) for x in (read.get_tag(start_tag).split(',') if isinstance(read.get_tag(start_tag), str)
                                       else read.get_tag(start_tag)) if str(x) != '']
            lens = [int(x) for x in (read.get_tag(len_tag).split(',') if isinstance(read.get_tag(len_tag), str)
                                     else read.get_tag(len_tag)) if str(x) != '']
        except (TypeError, ValueError):
            return None
        if len(starts) != len(lens):
            return None
        return starts, lens

    nuc = pairs('ns', 'nl')
    msp = pairs('as', 'al')
    if nuc is None and msp is None:
        return None
    if nuc is not None:
        q = list(read.get_tag('nq')) if read.has_tag('nq') else None
        rows = sorted((s0, l0, (int(q[i]) if q is not None and i < len(q) else None))
                      for i, (s0, l0) in enumerate(zip(*nuc)) if l0 > 0 and 0 <= s0 and s0 + l0 <= length)
        if rows:
            with_q = q is not None and len(q) == len(nuc[0])
            parts.append(('nuc.Q:' if with_q else 'nuc.:') + ','.join(f'{s0 + 1}-{l0}' for s0, l0, _q in rows))
            if with_q:
                aq.extend(max(0, min(255, qq)) for _s, _l, qq in rows)
    if msp is not None:
        rows = sorted((s0, l0) for s0, l0 in zip(*msp) if l0 > 0 and 0 <= s0 and s0 + l0 <= length)
        if rows:
            parts.append('msp.:' + ','.join(f'{s0 + 1}-{l0}' for s0, l0 in rows))
    if len(parts) == 1:
        return None
    return ';'.join(parts), aq


def _read_annotations(read):
    """(ma, aq list, an names, source) of a read: FiberHMM MA, else fibertools Ma, else its legacy
    ns/nl/as/al calls as MA groups, else a bare length."""
    from fiberhmm.io.ma_tags import annotation_tags, parse_an_tag
    tags = annotation_tags(read)
    if tags is None:
        legacy = _legacy_as_ma(read)
        if legacy is not None:
            ma, aq = legacy
            return ma, list(aq), None, 'legacy'
        length = int(read.query_length or len(read.query_sequence or ''))
        return str(length), [], [], None
    ma, aq, an, source = tags
    aq_list = [int(v) for v in aq] if aq is not None else []
    names = parse_an_tag(str(an)) if an is not None else None
    return str(ma), aq_list, names, source


def ma_read_length(read) -> int:
    """The MA read length the read's annotations use (MA, else Ma, else the stored SEQ length)."""
    ma, _aq, _an, _src = _read_annotations(read)
    return int(str(ma).split(';', 1)[0])


def replace_scan_groups(read, groups: Mapping[str, Tuple[str, Sequence[Tuple[int, int, Sequence[int], str]]]]) -> int:
    """Replace (or add) scan groups on ``read`` in place.

    ``groups``: name -> (quality spec, [(mol_start, length, qbytes, an_name), ...]),
    molecular 0-based starts. Existing sections of those names (with their AQ
    bytes and AN names) are removed; every other group is kept verbatim. The new
    groups are appended sorted by start. Every interval gets an AN name ('.' for
    previously unnamed ones). A group with no intervals is removed and not
    re-added. Returns the number of intervals written.

    The read's existing annotations must already be in molecular frame (call
    :func:`fiberhmm.io.frame_carry.carry_to_molecular` first)."""
    from fiberhmm.io.ma_tags import format_an_tag, parse_ma_tag, FIBERTOOLS_ANNOTATION_TAGS
    ma, aq, names, source = _read_annotations(read)
    parsed = parse_ma_tag(ma) if ';' in ma else {'read_length': int(ma), 'raw_types': []}
    length = int(parsed['read_length'])
    sections = [s for s in ma.split(';')[1:] if s]
    types = parsed['raw_types']
    if len(sections) != len(types):
        raise ValueError(f'read {read.query_name!r}: unparsable MA')
    expected = sum(len(spec) * len(iv) for _n, _s, spec, iv in types)
    if len(aq) != expected:
        raise ValueError(f'read {read.query_name!r}: MA/AQ length mismatch')
    count = sum(len(iv) for *_x, iv in types)
    if names is None:
        names = [''] * count
    if len(names) != count:
        raise ValueError(f'read {read.query_name!r}: MA/AN length mismatch')
    kept_sections, kept_aq, kept_names = [], [], []
    qi = ni = 0
    for section, (name, _strand, spec, intervals) in zip(sections, types):
        nq = len(spec) * len(intervals)
        n = len(intervals)
        if name not in groups:
            kept_sections.append(section)
            kept_aq.extend(aq[qi:qi + nq])
            kept_names.extend(names[ni:ni + n])
        qi += nq
        ni += n
    written = 0
    for name in sorted(groups):
        spec, items = groups[name]
        if not is_scan_group_name(name):
            raise ValueError(f'invalid scan MA group name: {name!r}')
        if any(ch != 'Q' for ch in spec):
            raise ValueError(f'scan MA group {name}: quality spec must be Q bytes only')
        rows = []
        for start, size, qbytes, an_name in items:
            start, size = int(start), int(size)
            if size <= 0 or start < 0 or start + size > length:
                continue
            if len(qbytes) != len(spec):
                raise ValueError(f'scan MA group {name}: {len(qbytes)} bytes for spec {spec!r}')
            if ',' in str(an_name) or ';' in str(an_name):
                raise ValueError(f'scan MA group {name}: AN names cannot contain commas or semicolons')
            rows.append((start, size, [max(0, min(255, int(v))) for v in qbytes], str(an_name or '.')))
        rows.sort(key=lambda r: (r[0], r[1], r[3], r[2]))
        if not rows:
            continue
        kept_sections.append(f'{name}.{spec}:' + ','.join(f'{s + 1}-{n}' for s, n, _q, _a in rows))
        for _s, _n, q, a in rows:
            kept_aq.extend(q)
            kept_names.append(a)
        written += len(rows)
    if kept_sections:
        read.set_tag('MA', ';'.join([str(length), *kept_sections]), value_type='Z')
        read.set_tag('AQ', array.array('B', kept_aq) if kept_aq else None)
        if any(kept_names) or written:
            read.set_tag('AN', format_an_tag(kept_names), value_type='Z')
        elif read.has_tag('AN'):
            read.set_tag('AN', None)
    else:
        for tag in ('MA', 'AQ', 'AN'):
            if read.has_tag(tag):
                read.set_tag(tag, None)
    if source == 'Ma':
        # The MA written here carries the fibertools groups; FiberHMM's MA wins over Ma.
        pass
    return written


def scan_groups_of(read, contracts: Mapping[str, dict]) -> Dict[str, List[dict]]:
    """Decode a read's scan groups: group -> [{start, length, q, name, scalars?}] (molecular)."""
    from fiberhmm.io.ma_tags import parse_an_tag, parse_aq_array, parse_ma_tag
    if not read.has_tag('MA'):
        return {}
    parsed = parse_ma_tag(str(read.get_tag('MA')))
    types = parsed['raw_types']
    aq = list(read.get_tag('AQ')) if read.has_tag('AQ') else []
    counts = [len(iv) for *_x, iv in types]
    rows = parse_aq_array(aq, [t[2] for t in types], counts)
    total = sum(counts)
    names = parse_an_tag(str(read.get_tag('AN'))) if read.has_tag('AN') else [''] * total
    out: Dict[str, List[dict]] = {}
    cursor = 0
    for (name, _strand, _spec, intervals) in types:
        for i, (s, n) in enumerate(intervals):
            if name in contracts:
                item = dict(start=s, length=n, q=list(rows[cursor + i]), name=names[cursor + i])
                c = contracts[name]
                if c.get('kind') == 'scalar' and c.get('codecs'):
                    item['scalars'] = decode_scalars(item['q'], c['codecs'])
                out.setdefault(name, []).append(item)
        cursor += len(intervals)
    return out


# ------------------------------------------------------------------ projection
def project_to_molecule(read, start: int, end: int, ma_length: Optional[int] = None) -> Optional[Tuple[int, int]]:
    """Molecular ``(start, length)`` of the reference span ``[start, end)`` on ``read``.

    The span is the record's aligned bases inside it (None when there are
    none). The molecular frame is the original read 5'->3'; it includes hard
    clips when the MA read length covers the whole molecule."""
    pairs = read.get_aligned_pairs(matches_only=True)
    qs = [q for q, r in pairs if start <= r < max(end, start + 1)]
    if not qs:
        return None
    lo, hi = min(qs), max(qs) + 1
    cigar = read.cigartuples or []
    lead = cigar[0][1] if cigar and cigar[0][0] == 5 else 0
    trail = cigar[-1][1] if len(cigar) > 1 and cigar[-1][0] == 5 else 0
    stored = int(read.query_length or len(read.query_sequence or ''))
    if ma_length is None:
        ma_length = ma_read_length(read)
    if ma_length == stored:
        offset, total = 0, stored
    elif ma_length == stored + lead + trail:
        offset, total = lead, ma_length
    else:
        raise ValueError(f'read {read.query_name!r}: MA length {ma_length} matches neither the stored '
                         f'sequence ({stored}) nor the whole molecule ({stored + lead + trail})')
    lo, hi = lo + offset, hi + offset
    if read.is_reverse:
        lo, hi = total - hi, total - lo
    return lo, hi - lo
