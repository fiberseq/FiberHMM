"""Portable caller evidence, independent of prepass files and worker globals.

The last call's PG DS holds ``daf_evidence=v1:<zlib/base64 JSON>`` with
reference SNP positions and the unaligned-mask policy. ``ei:Z`` on a record
holds [version, SEQ length, strand, known SEQ positions, converted positions]
for insertion consensus. An absent ei tag means no insertion evidence.
Coordinates stay in stored SEQ/reference frame, including on reverse reads.

A graph lift (fiberhmm-lift) moves both frames. ``ei`` is SEQ-frame and is
transformed exactly by :func:`lift_insert_evidence` (mirrored with its strand
swapped when the read changes strand, shifted when hard clips are filled, and
dropped when it no longer matches SEQ). The reference SNP mask is in source
assembly coordinates and cannot be lifted, so replay does not apply it to a
BAM lifted after its last fiberhmm-call (and uses no ``ei`` at all while a
lift is not yet repaired): such reads replay with unmasked aligned sites,
as for a legacy BAM, with a recorded warning.
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


def lift_insert_evidence(read, old_length: int, *, reverse: bool = False,
                         offset: int = 0):
    """Move ``ei`` onto the record's new SEQ after a lift changed it.

    ``old_length`` is the SEQ length the tag was written for. ``reverse``: SEQ
    was reverse-complemented (a strand change), so position ``p`` becomes
    ``old_length - 1 - p`` and the CT/GA strand swaps. ``offset``: bases were
    added before the old SEQ (filled hard clips); positions move by it. The
    tag is removed when it is malformed, was written for another length, or
    does not fit the new SEQ. Returns None without a tag, True when kept,
    False when removed."""
    if not read.has_tag(INSERT_TAG):
        return None
    try:
        version, length, strand, known, mods = _unpack(read.get_tag(INSERT_TAG))
        known, mods = [int(p) for p in known], [int(p) for p in mods]
        if (version != 1 or length != old_length or strand not in (0, 1)
                or not set(mods) <= set(known)
                or any(p < 0 or p >= length for p in known)):
            raise ValueError('stale insertion evidence')
    except (ValueError, TypeError, KeyError, zlib.error):
        read.set_tag(INSERT_TAG, None)
        return False
    if reverse:
        known = [length - 1 - p for p in known]
        mods = [length - 1 - p for p in mods]
        strand = 1 - strand
    known = [p + offset for p in known]
    mods = [p + offset for p in mods]
    new_length = len(read.query_sequence or '')
    if any(p >= new_length for p in known):
        read.set_tag(INSERT_TAG, None)
        return False
    read.set_tag(INSERT_TAG, _pack([1, new_length, strand, sorted(known), sorted(mods)]),
                 value_type='Z')
    return True


def _lifted_since_call(pgs, call_index):
    """``(lifted, unrepaired)``: a graph lift ran after the @PG at
    ``call_index`` (-1: no caller line), and a lift is not yet repaired."""
    from fiberhmm.io.graph_lift import is_lift_pg, is_repair_pg, pending_lift_indices
    later = pgs[call_index + 1:]
    unrepaired = any(i > call_index for i in pending_lift_indices(pgs))
    lifted = unrepaired or any(is_lift_pg(pg) or is_repair_pg(pg) for pg in later)
    return lifted, unrepaired


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
        pgs = list(hd.get('PG', []) or [])
        from fiberhmm.nuc.records import is_call_record
        call_index = next((i for i in range(len(pgs) - 1, -1, -1)
                           if is_call_record(pgs[i])), -1)
        pg = pgs[call_index] if call_index >= 0 else {}
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
                lifted, unrepaired = _lifted_since_call(pgs, call_index)
                if lifted:
                    # Source-assembly coordinates: never apply them to lifted reads.
                    self.snps = {}
                    self.warn('BAM was lifted after fiberhmm-call: the reference DAF SNP '
                              'mask is in source coordinates and is not applied')
                if unrepaired:
                    # ei is moved by fiberhmm-lift's repair only.
                    self.portable = False
                    self.warn('BAM was lifted without fiberhmm-lift repair: DAF insertion '
                              'evidence is not used')
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
        from fiberhmm.daf.insert_consensus import record_key
        key = record_key(read)
        with daf_snp_mask_scope(self.snps), \
                daf_insert_evidence_scope({key: ev} if ev is not None else {}), \
                daf_unaligned_mask_scope(self.mask_unaligned):
            yield
