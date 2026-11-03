"""Portable caller evidence, independent of prepass files and worker globals.

The last call's PG DS holds ``daf_evidence=v1:<zlib/base64 JSON>`` with
reference SNP positions and the unaligned-mask policy. ``ei:Z`` on a record
holds [version, SEQ length, strand, known SEQ positions, converted positions]
for insertion consensus. An absent ei tag means no insertion evidence.
Coordinates stay in stored SEQ/reference frame, including on reverse reads.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import json
import re
import zlib

INSERT_TAG = 'ei'


def _pack(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return base64.b64encode(zlib.compress(raw)).decode('ascii')


def _unpack(value):
    return json.loads(zlib.decompress(base64.b64decode(value, validate=True)))


def evidence_token(snp_mask=None, *, mask_unaligned=True):
    """A self-contained DS token; no paths, including when the mask is empty."""
    return 'daf_evidence=v1:' + _pack({
        'snps': {c: sorted(map(int, sites)) for c, sites in (snp_mask or {}).items()},
        'mask_unaligned': bool(mask_unaligned),
    })


def write_insert_evidence(read):
    """Persist this call's prepass evidence; replace/remove any stale tag."""
    if not hasattr(read, 'has_tag'):
        return
    from fiberhmm.inference.engine import daf_insert_evidence
    ev = daf_insert_evidence(read)
    value = None if ev is None else _pack(
        [1, len(read.query_sequence or ''), int(ev.strand), sorted(ev.known), sorted(ev.mods)])
    if value is not None or read.has_tag(INSERT_TAG):
        read.set_tag(INSERT_TAG, value, value_type='Z')


class EvidenceReplay:
    """Header-local evidence and recorded warnings for legacy/malformed BAMs."""

    def __init__(self, header, mode):
        self.snps = {}
        self.mask_unaligned = True
        self.portable = False
        self.warnings = []
        if mode != 'daf':
            return
        hd = header.to_dict() if hasattr(header, 'to_dict') else dict(header or {})
        pg = next((p for p in reversed(hd.get('PG', []))
                   if 'fiberhmm-call' in str(p.get('PN') or p.get('ID') or '')), {})
        ds = str(pg.get('DS', ''))
        token = re.search(r'(?:^|\s)daf_evidence=(\S+)', ds)
        if token:
            try:
                version, data = token.group(1).split(':', 1)
                if version != 'v1':
                    raise ValueError('unsupported version')
                obj = _unpack(data)
                self.snps = {c: set(map(int, sites)) for c, sites in obj['snps'].items()}
                self.mask_unaligned = bool(obj['mask_unaligned'])
                self.portable = True
                return
            except (ValueError, TypeError, KeyError, AttributeError, zlib.error):
                self.snps = {}
                self.warn('invalid portable DAF evidence; using legacy evidence fallback')
        self.mask_unaligned = not bool(re.search(r'\bdaf_unaligned_mask=off\b', ds))
        # Older outputs identify active prepasses but contain no coordinates or
        # per-record consensus. Never reopen their CL/QC/temp paths, or inherit
        # another dataset's process-global masks.
        if (not re.search(r'\bdaf_snp_mask=(?:off|auto-skip/\S+)\b', ds)
                or not re.search(r'\bdaf_insert_consensus=(?:off|auto-skip/\S+)\b', ds)):
            self.warn('legacy BAM lacks portable DAF SNP/insertion evidence; '
                      'replay uses unmasked aligned sites and no insertion consensus')

    def warn(self, message):
        if message not in self.warnings:
            self.warnings.append(message)

    @contextmanager
    def scope(self, read):
        from fiberhmm.inference.engine import (daf_snp_mask_scope,
                                               daf_insert_evidence_scope,
                                               daf_unaligned_mask_scope)
        ev = None
        if self.portable and read.has_tag(INSERT_TAG):
            try:
                from fiberhmm.daf.insert_consensus import ReadInsertEvidence
                version, length, strand, known, mods = _unpack(read.get_tag(INSERT_TAG))
                known, mods = set(known), set(mods)
                if (version != 1 or length != len(read.query_sequence or '') or strand not in (0, 1)
                        or not mods <= known or any(not isinstance(p, int) or p < 0 or p >= length
                                                    for p in known)):
                    raise ValueError('invalid insertion evidence')
                ev = ReadInsertEvidence(mods=mods, known=known, strand=strand)
            except (ValueError, TypeError, KeyError, zlib.error):
                self.warn('invalid portable DAF insertion evidence; no insertion consensus used')
        # Explicit empty scopes are essential: browser datasets can coexist in
        # the same process as a caller or another dataset.
        key = (read.query_name, int(read.flag) & ~0x400, int(read.reference_id),
               int(read.reference_start))
        with daf_snp_mask_scope(self.snps), \
                daf_insert_evidence_scope({key: ev} if ev is not None else {}), \
                daf_unaligned_mask_scope(self.mask_unaligned):
            yield
