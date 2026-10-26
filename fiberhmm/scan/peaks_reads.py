"""Per-read footprint/gap tilings as FiberBrowser builds them for its peak scan (a port).

FiberBrowser's accessibility peak scan counts, per base, the reads whose
compact tiling covers it (``total``) and those whose tile there is a gap or a
footprint shorter than 90 bp (``numerator``). The tiling comes from the
browser's BAM loader (``services/compact_read_loader.py`` with its default
working-set options: terminal gaps hidden beyond 500 bp, read-end footprints
under 90 bp dropped, duplicates and secondary records skipped, one row per
primary record, supplementaries only for molecules whose primary is not in
view). This module ports that path so ``fiberhmm-scan``'s ``peaks`` module
gives the browser's per-position scores. Names follow the browser's.
"""
from __future__ import annotations

import bisect
import re
from operator import add, itemgetter
from typing import Dict, Iterable, List, Optional, Tuple

MA_FOOTPRINT_TYPES = ('nuc', 'tf', 'fp_v2')
SHORT_TERMINAL_FOOTPRINT_BP = 90
SHORT_TERMINAL_FOOTPRINT_MAX_END_GAP_BP = 30
TERMINAL_GAP_THRESHOLD = 500

BROWSER_DEFAULTS = dict(hide_terminal_gaps=True, terminal_gap_threshold=TERMINAL_GAP_THRESHOLD,
                        short_terminal_footprint_bp=SHORT_TERMINAL_FOOTPRINT_BP, include_duplicates=False,
                        min_fp_quality=0, spanning_only=False)


class CigarCoordMapper:
    """Query-to-reference mapping over CIGAR ranges (FiberBrowser ``models.domain.CigarCoordMapper``)."""

    __slots__ = ('_query_starts', '_query_ends', '_ref_starts', '_ref_offsets', '_insertions', '_simple_offset',
                 '_simple_query_start', '_simple_query_end', '_aligned_starts', '_aligned_ends', '_aligned_offsets')

    def __init__(self, cigartuples, ref_start):
        query_ends, query_starts, ref_starts, ref_offsets, insertions = [], [], [], [], []
        ref_pos, query_pos = ref_start, 0
        mapped_range_count, mapped_range_idx = 0, -1
        for op, length in cigartuples or []:
            if op in (0, 7, 8):
                query_starts.append(query_pos)
                query_ends.append(query_pos + length)
                ref_starts.append(ref_pos)
                ref_offsets.append(ref_pos - query_pos)
                mapped_range_count += 1
                mapped_range_idx = len(query_ends) - 1
                ref_pos += length
                query_pos += length
            elif op == 1:
                query_starts.append(query_pos)
                query_ends.append(query_pos + length)
                ref_starts.append(ref_pos)
                ref_offsets.append(0)
                insertions.append(len(query_ends) - 1)
                query_pos += length
            elif op in (2, 3):
                ref_pos += length
            elif op == 4:
                query_starts.append(query_pos)
                query_ends.append(query_pos + length)
                ref_starts.append(-1)
                ref_offsets.append(0)
                query_pos += length
        if mapped_range_count == 1:
            self._simple_offset = ref_offsets[mapped_range_idx]
            self._simple_query_start = query_starts[mapped_range_idx]
            self._simple_query_end = query_ends[mapped_range_idx]
        else:
            self._simple_offset = None
            self._simple_query_start = 0
            self._simple_query_end = 0
        self._query_starts = query_starts
        self._query_ends = query_ends
        self._ref_starts = ref_starts
        self._ref_offsets = ref_offsets
        self._insertions = set(insertions)
        aligned = [i for i, start in enumerate(ref_starts)
                   if start >= 0 and i not in self._insertions and query_starts[i] < query_ends[i]]
        self._aligned_starts = [query_starts[i] for i in aligned]
        self._aligned_ends = [query_ends[i] for i in aligned]
        self._aligned_offsets = [ref_offsets[i] for i in aligned]

    def query_to_ref(self, query_pos):
        if not self._query_ends:
            return None
        if self._simple_offset is not None:
            if self._simple_query_start <= query_pos < self._simple_query_end:
                return query_pos + self._simple_offset
            return None
        idx = bisect.bisect_right(self._query_ends, query_pos)
        if idx >= len(self._query_ends):
            return None
        ref_start = self._ref_starts[idx]
        if ref_start == -1:
            return None
        if idx in self._insertions:
            return ref_start
        return query_pos + self._ref_offsets[idx]

    def aligned_query_interval_to_ref(self, query_start, query_end):
        if query_end < query_start:
            return None, None
        first = bisect.bisect_right(self._aligned_ends, query_start)
        last = bisect.bisect_right(self._aligned_starts, query_end) - 1
        if first > last:
            return None, None
        left = max(query_start, self._aligned_starts[first]) + self._aligned_offsets[first]
        right = min(query_end + 1, self._aligned_ends[last]) + self._aligned_offsets[last]
        return left, right

    def footprint_to_ref(self, query_start, query_end):
        ref_start, ref_end = self.query_interval_to_ref(query_start, query_end)
        if ref_start is None or ref_end is None:
            ref_start, ref_end = self.aligned_query_interval_to_ref(query_start, query_end)
        return ref_start, ref_end

    def query_interval_to_ref(self, query_start, query_end):
        if not self._query_ends:
            return None, None
        if self._simple_offset is not None:
            if not (self._simple_query_start <= query_start < self._simple_query_end):
                return None, None
            mapped_end_query = min(query_end, self._simple_query_end - 1)
            return query_start + self._simple_offset, mapped_end_query + self._simple_offset + 1
        query_ends, ref_starts, ref_offsets, insertions = (self._query_ends, self._ref_starts, self._ref_offsets,
                                                           self._insertions)
        start_idx = bisect.bisect_right(query_ends, query_start)
        if start_idx >= len(query_ends):
            return None, None
        ref_start = ref_starts[start_idx]
        if ref_start == -1:
            return None, None
        if start_idx not in insertions:
            ref_start = query_start + ref_offsets[start_idx]
        end_ref = None
        end_idx = bisect.bisect_right(query_ends, query_end)
        if end_idx < len(query_ends):
            end_ref_start = ref_starts[end_idx]
            if end_ref_start != -1:
                end_ref = end_ref_start + 1 if end_idx in insertions else query_end + ref_offsets[end_idx] + 1
        if end_ref is None:
            search_idx = min(end_idx, len(query_ends) - 1)
            query_starts = self._query_starts
            while search_idx >= start_idx:
                candidate_ref_start = ref_starts[search_idx]
                candidate_query_pos = min(query_end, query_ends[search_idx] - 1)
                if (candidate_ref_start != -1 and candidate_query_pos > query_start
                        and candidate_query_pos >= query_starts[search_idx]):
                    end_ref = (candidate_ref_start + 1 if search_idx in insertions
                               else candidate_query_pos + ref_offsets[search_idx] + 1)
                    break
                search_idx -= 1
        if end_ref is None or end_ref <= ref_start:
            end_ref = ref_start + 1
        return ref_start, end_ref


def map_tagged_footprints(fp_starts_query, fp_lengths, fp_qualities, mapper, min_fp_quality=0):
    """(ref_start, ref_end, size, quality, tag_index) per footprint (``footprint_to_ref`` placement)."""
    out = []
    for tag_index, (fp_start, fp_len) in enumerate(zip(fp_starts_query, fp_lengths)):
        if min_fp_quality > 0 and fp_qualities is not None:
            if tag_index < len(fp_qualities) and fp_qualities[tag_index] < min_fp_quality:
                continue
        a, b = mapper.footprint_to_ref(fp_start, fp_start + fp_len - 1)
        if a is None or b is None:
            continue
        q = fp_qualities[tag_index] if fp_qualities is not None and tag_index < len(fp_qualities) else None
        out.append((a, b, fp_len, q, tag_index))
    return out


def walk_read_intervals(blocks, read_start, read_end, region_start, region_end, hide_terminal_gaps,
                        call_extent=None):
    out = []

    def add_gap(lo, hi, size):
        a, b = max(lo, region_start), min(hi, region_end)
        if b > a:
            out.append((a, b, size, None))

    call_lo = call_hi = None
    if hide_terminal_gaps and call_extent is not None:
        call_lo, call_hi = int(call_extent[0]), int(call_extent[1])
    prev = None if hide_terminal_gaps else int(read_start)
    if call_lo is not None and blocks and call_lo < int(blocks[0][0]):
        prev = call_lo
    for i, (fp_start, fp_end, fp_size) in enumerate(blocks):
        fp_start, fp_end = int(fp_start), int(fp_end)
        if prev is not None and fp_start > prev:
            if i == 0 and not hide_terminal_gaps:
                add_gap(prev, fp_start, -(min(fp_start, region_end) - max(prev, region_start)))
            else:
                add_gap(prev, fp_start, -(fp_start - prev))
        if fp_end > region_start and fp_start < region_end:
            out.append((max(fp_start, region_start), min(fp_end, region_end), int(fp_size), i))
        prev = fp_end if prev is None else max(prev, fp_end)
    if not hide_terminal_gaps and prev is not None and int(read_end) > prev:
        add_gap(prev, int(read_end), -(min(int(read_end), region_end) - max(prev, region_start)))
    elif call_hi is not None and prev is not None and call_hi > prev:
        add_gap(prev, call_hi, -(call_hi - prev))
    return out


def terminal_gap_rejects(sorted_footprints, ref_start, ref_end, terminal_gap_threshold, call_extent=None):
    if terminal_gap_threshold <= 0 or not sorted_footprints:
        return False
    first_call = sorted_footprints[0][0]
    last_call = sorted_footprints[-1][1]
    if call_extent is not None:
        first_call = min(first_call, call_extent[0])
        last_call = max(last_call, call_extent[1])
    return first_call - ref_start > terminal_gap_threshold or ref_end - last_call > terminal_gap_threshold


def build_intervals_from_footprints(footprints, ref_start, ref_end, region_start, region_end, *,
                                    hide_terminal_gaps=True, terminal_gap_threshold=TERMINAL_GAP_THRESHOLD,
                                    call_extent=None):
    """(intervals, terminal_filtered): ``[start, end, size]`` relative to region_start."""
    if not footprints:
        return [], False
    fps = sorted(footprints, key=lambda item: item[0])
    if hide_terminal_gaps and terminal_gap_rejects(fps, ref_start, ref_end, terminal_gap_threshold, call_extent):
        return [], True
    out = []
    for a, b, size, _i in walk_read_intervals([(fp[0], fp[1], fp[2]) for fp in fps], ref_start, ref_end,
                                              region_start, region_end, hide_terminal_gaps, call_extent=call_extent):
        out.append([int(a - region_start), int(b - region_start), int(size)])
    return out, False


def short_terminal_spans(ns_starts, ns_lengths, ma_spans, min_bp, *, read_length=0,
                         max_end_gap=SHORT_TERMINAL_FOOTPRINT_MAX_END_GAP_BP):
    if min_bp <= 0:
        return set()
    candidates = []
    if ns_starts is not None and len(ns_starts):
        ends = list(map(add, ns_starts, ns_lengths))
        first = min(range(len(ends)), key=ns_starts.__getitem__)
        last = max(range(len(ends)), key=ends.__getitem__)
        candidates.append((int(ns_starts[first]), int(ends[first])))
        candidates.append((int(ns_starts[last]), int(ends[last])))
    if ma_spans:
        first = min(ma_spans, key=itemgetter(1))
        last = max(ma_spans, key=itemgetter(2))
        candidates.append((first[1], first[2]))
        candidates.append((last[1], last[2]))
    if not candidates:
        return set()
    lo = min(c[0] for c in candidates)
    hi = max(c[1] for c in candidates)
    at_start = lo < max_end_gap
    at_end = bool(read_length) and read_length - hi < max_end_gap
    return {c for c in candidates
            if c[1] - c[0] < min_bp and ((at_start and c[0] == lo) or (at_end and c[1] == hi))}


def _extent_union(*extents):
    present = [e for e in extents if e is not None]
    if not present:
        return None
    return min(e[0] for e in present), max(e[1] for e in present)


def ma_tags_of(read, group_frame):
    """(MA, AQ, AN, frame) of FiberHMM's MA, else fibertools' Ma (always molecular), else None."""
    if read.has_tag('MA'):
        return (read.get_tag('MA'), read.get_tag('AQ') if read.has_tag('AQ') else None,
                read.get_tag('AN') if read.has_tag('AN') else None, group_frame)
    if read.has_tag('Ma'):
        return (read.get_tag('Ma'), read.get_tag('Aq') if read.has_tag('Aq') else None,
                read.get_tag('An') if read.has_tag('An') else None, 'all')
    return None


def _group_is_molecular(frame, name, strand):
    if frame == 'all':
        return True
    return frame.is_molecular(name, strand)


def ma_footprint_query_spans(ma_tags, *, read_length, is_reverse, molecular_layers=frozenset()):
    """A read's MA footprint calls (nuc, tf, fp_v2) as SEQ-frame query (name, start, end)."""
    if ma_tags is None or not ma_tags[0] or not isinstance(ma_tags[0], str):
        return []
    spans = []
    spec_by_name: Dict[str, str] = {}
    for part in ma_tags[0].split(';')[1:]:
        type_sig, sep, body = part.partition(':')
        if not sep:
            continue
        strand_idx = next((i for i in (type_sig.find(c) for c in '+-.') if i >= 0), -1)
        if strand_idx <= 0:
            continue
        name, qualspec = type_sig[:strand_idx], type_sig[strand_idx + 1:]
        if name not in MA_FOOTPRINT_TYPES:
            continue
        if spec_by_name.setdefault(name, qualspec) != qualspec:
            continue
        molecular = _group_is_molecular(ma_tags[3], name, type_sig[strand_idx]) or name in molecular_layers
        flip = bool(molecular and is_reverse and read_length)
        for iv in body.split(','):
            iv = iv.strip()
            if not iv or '-' not in iv:
                continue
            try:
                s_str, l_str = iv.split('-', 1)
                q_start, q_end = int(s_str) - 1, int(s_str) - 1 + int(l_str)
            except ValueError:
                continue
            if flip:
                q_start, q_end = read_length - q_end, read_length - q_start
            spans.append((name, q_start, q_end))
    return spans


def ma_footprint_call_extent(spans, mapper, ref_start, ref_end, skip=None):
    if ref_end <= ref_start:
        return None
    spans = [(a, b) for _n, a, b in spans if not skip or (a, b) not in skip]
    if not spans:
        return None

    def first_mapped(ordered, pick):
        for q_start, q_end in ordered:
            r_start, r_end = mapper.aligned_query_interval_to_ref(q_start, q_end - 1)
            if r_start is not None and r_end is not None and min(r_end, ref_end) > max(r_start, ref_start):
                return pick(r_start, r_end)
        return None

    lo = first_mapped(sorted(spans), lambda r_start, _r_end: max(int(r_start), ref_start))
    hi = first_mapped(sorted(spans, key=lambda s: s[1], reverse=True), lambda _r, r_end: min(int(r_end), ref_end))
    if lo is None or hi is None or hi <= lo:
        return None
    return lo, hi


def msp_complement(starts, lengths, read_length):
    msps = sorted((int(s), int(s) + int(l)) for s, l in zip(starts or [], lengths or []) if int(l) > 0)
    out_s, out_l = [], []
    pos = 0
    for lo, hi in msps:
        if lo > pos:
            out_s.append(pos)
            out_l.append(lo - pos)
        pos = max(pos, hi)
    if read_length > pos:
        out_s.append(pos)
        out_l.append(read_length - pos)
    return out_s, out_l


# ------------------------------------------------------------------ deletion cuts (FiberBrowser sv_events)
SV_MIN_LEN = 50
SV_PIECE_MIN = 10
SV_MERGE_GAP = 10


def _cigar_pieces(cigartuples, ref_start):
    pieces, ref_pos, q_pos = [], ref_start, 0
    for op, length in cigartuples:
        pieces.append((op, ref_pos, length, q_pos))
        if op in (0, 7, 8):
            ref_pos += length
            q_pos += length
        elif op in (2, 3):
            ref_pos += length
        elif op in (1, 4):
            q_pos += length
    return pieces


def _cigar_replacements(pieces, piece_min=SV_PIECE_MIN, merge_gap=SV_MERGE_GAP):
    """Indel pieces joined across short aligned stretches (FiberBrowser ``sv_alleles.cigar_replacements``)."""
    groups, group, bridge = [], [], []
    for op, ref, length, q in pieces:
        if op in (1, 2):
            group.extend(bridge)
            bridge = []
            group.append((op, ref, length, q))
        elif op in (0, 7, 8) and group and sum(p[2] for p in bridge) + length <= merge_gap:
            bridge.append((op, ref, length, q))
        else:
            if group:
                groups.append(group)
            group, bridge = [], []
    if group:
        groups.append(group)
    out = []
    for group in groups:
        indels = [p for p in group if p[0] in (1, 2)]
        if not any(p[2] >= piece_min for p in indels):
            continue
        ins = sum(p[2] for p in indels if p[0] == 1)
        deleted = sum(p[2] for p in indels if p[0] == 2)
        event = dict(t='I' if ins >= deleted else 'D', ref=group[0][1], len=max(ins, deleted))
        if len(group) > 1:
            event['raw_pieces'] = [dict(t={0: 'M', 1: 'I', 2: 'D', 7: '=', 8: 'X'}[op], ref=r, len=n)
                                   for op, r, n, _q in group]
        out.append(event)
    return out


def read_deletion_cuts(read, min_len=SV_MIN_LEN):
    """Absolute deleted spans of a record's indel events of at least ``min_len`` (the browser splits
    every call of the row at them)."""
    try:
        events = [e for e in _cigar_replacements(_cigar_pieces(read.cigartuples or (), int(read.reference_start)))
                  if e['len'] >= min_len]
    except Exception:
        return []
    return sorted((int(p['ref']), int(p['ref']) + int(p['len'])) for ev in events
                  for p in ev.get('raw_pieces') or [ev] if p.get('t') == 'D')


def _subtract_spans(a, b, cuts):
    pieces, cur = [], a
    for cs, ce in cuts:
        if ce <= cur:
            continue
        if cs >= b:
            break
        if cs > cur:
            pieces.append((cur, min(cs, b)))
        cur = max(cur, ce)
        if cur >= b:
            break
    if cur < b:
        pieces.append((cur, b))
    return pieces


def split_at_deletions(intervals, cuts):
    """A row's intervals minus the deleted spans (region-relative); each piece keeps its call's size."""
    cuts = sorted((int(a), int(b)) for a, b in cuts if b > a)
    if not cuts or not intervals:
        return intervals
    lo, hi = cuts[0][0], max(b for _a, b in cuts)
    out = []
    for iv in intervals:
        a, b = iv[0], iv[1]
        if b <= lo or a >= hi or not any(cs < b and ce > a for cs, ce in cuts):
            out.append(iv)
            continue
        for pa, pb in _subtract_spans(a, b, cuts):
            out.append([pa, pb, *iv[2:]])
    return out


# ------------------------------------------------------------------ the loader path
def _anchors(bam, chrom, start, end, include_duplicates=False, row_rule='view', hidden=None):
    """Rows of the region.

    ``view`` (FiberBrowser ``sv_molecules.molecule_alignments``): one row per primary record of each molecule
    in view, a supplementary row only for a molecule whose primary is not in view. Whether a supplementary
    counts then depends on the view's extent.
    ``records``: every primary and supplementary record that passes the record filters on its own, which
    depends on no extent; grid scans use it so stitched zones do not depend on where units fall."""
    groups: Dict[str, dict] = {}
    for read in bam.fetch(chrom, start, end):
        if read.is_unmapped or read.is_secondary or (hidden is not None and hidden(read)):
            continue
        status = include_duplicates or not read.is_duplicate
        g = groups.setdefault(read.query_name, dict(primary=False, recs=[]))
        g['primary'] |= not read.is_supplementary
        if not status:
            continue
        g['recs'].append(read)
    for name, g in groups.items():
        recs = g['recs']
        if row_rule == 'records':
            yield from recs
            continue
        primaries = [r for r in recs if not r.is_supplementary]
        if g['primary'] and not primaries:
            continue
        if not recs:
            continue
        recs.sort(key=lambda r: (bool(r.is_supplementary), -int(r.mapping_quality or 0), int(r.reference_start),
                                 r.cigarstring or '', bool(r.is_reverse)))
        for anchor in (primaries if primaries else [recs[0]]):
            yield anchor


def parse_int_tag_values(tag):
    """BAM array, list or comma-separated string tag values as ints (FiberBrowser ``parse_int_tag_values``)."""
    if tag is None:
        return None
    if isinstance(tag, str):
        return [int(x) for x in tag.split(',') if x]
    return [int(x) for x in tag]


def _optional_ints(tag):
    try:
        return parse_int_tag_values(tag)
    except Exception:
        return None


# ------------------------------------------------------------------ legacy ns/nl frame (FiberBrowser bam_tags)
_FIBERTOOLS_PROGRAMS = frozenset({'ft', 'fibertools', 'fibertools-rs'})
_FIBERTOOLS_FOOTPRINT_SUBCOMMANDS = frozenset({'predict', 'predict-m6a', 'm6a', 'add', 'add-nucleosomes',
                                               'add-nucleosome', 'fire', 'fiber-hmm'})
_FIBERTOOLS_FOOTPRINT_PREFIXES = (('add-nucleosomes', 3), ('predict-m6a', 4), ('fire', 3), ('fiber-hmm', 5))
_FIBERHMM_FOOTPRINT_PROGRAMS = frozenset({'fiberhmm-apply', 'fiberhmm-call', 'fiberhmm-recall-tfs',
                                          'fiberhmm-recall-nucs'})
_FIBERHMM_NAME_RE = re.compile(r'^fiberhmm(?:-[a-z0-9-]+)?$')
_PG_ID_SUFFIX_RE = re.compile(r'(?:\.\d+|-[0-9a-f]{6,8})$')
_COORD_MOLECULAR = 'coord=molecular'
_MA_DOT_FOOTPRINT_RE = re.compile(r';(?:nuc|msp|tf)\.[A-Za-z]*:')


def _fibertools_footprint_subcommand(token):
    if token in _FIBERTOOLS_FOOTPRINT_SUBCOMMANDS:
        return True
    return any(len(token) >= n and full.startswith(token) for full, n in _FIBERTOOLS_FOOTPRINT_PREFIXES)


def _pg_id_base(ident):
    base = ident.lower()
    while True:
        stripped = _PG_ID_SUFFIX_RE.sub('', base)
        if stripped == base or not stripped:
            return base
        base = stripped


def _cl_tokens(cl):
    return [t for t in str(cl or '').lower().split() if t]


def _basename(token):
    return token.rstrip('/').rsplit('/', 1)[-1]


def _pg_family(pg):
    pn = str(pg.get('PN', '') or '').strip().lower()
    if pn:
        if 'fibertools' in pn:
            return 'fibertools'
        return 'fiberhmm' if _FIBERHMM_NAME_RE.match(pn) else None
    base = _pg_id_base(str(pg.get('ID', '') or '').strip())
    if base in _FIBERTOOLS_PROGRAMS or base.startswith('fibertools'):
        return 'fibertools'
    if _FIBERHMM_NAME_RE.match(base):
        return 'fiberhmm'
    tokens = _cl_tokens(pg.get('CL', ''))
    program = _basename(tokens[0]) if tokens else ''
    if program in _FIBERTOOLS_PROGRAMS:
        return 'fibertools'
    if _FIBERHMM_NAME_RE.match(program):
        return 'fiberhmm'
    return None


def _pg_text(pg):
    try:
        return ' '.join(str(v) for v in pg.values()).lower()
    except AttributeError:
        return str(pg).lower()


def _pg_writer_frame(pg, header_marker):
    family = _pg_family(pg)
    if family == 'fibertools':
        tokens = _cl_tokens(pg.get('CL', ''))
        start = next((i + 1 for i, t in enumerate(tokens) if _basename(t) in _FIBERTOOLS_PROGRAMS), 1)
        if any(_fibertools_footprint_subcommand(t) for t in tokens[start:]):
            return 'molecular'
        return None
    if family == 'fiberhmm':
        record_marker = _COORD_MOLECULAR in _pg_text(pg)
        tokens = _cl_tokens(pg.get('CL', ''))
        names = {str(pg.get('PN', '') or '').strip().lower(), _pg_id_base(str(pg.get('ID', '') or '').strip()),
                 _basename(tokens[0]) if tokens else ''}
        if record_marker or names & _FIBERHMM_FOOTPRINT_PROGRAMS:
            return 'molecular' if (record_marker or header_marker) else 'seq'
    return None


def footprint_writer_id(header) -> str:
    """ID of the last footprint-writing @PG (FiberHMM apply/call/recall, fibertools nucleosome commands),
    following the @PG chains as the legacy-frame rule does; '' when there is none."""
    hdr = header.to_dict() if hasattr(header, 'to_dict') else dict(header or {})
    pgs = [pg for pg in (hdr.get('PG', []) or []) if isinstance(pg, dict)]
    marker = any(_COORD_MOLECULAR in str(c).lower() for c in (hdr.get('CO', []) or []))
    writers = [i for i, pg in enumerate(pgs) if _pg_writer_frame(pg, marker) is not None]
    return str(pgs[writers[-1]].get('ID', '')) if writers else ''


def legacy_frame(header):
    """``(molecular, unknown)`` for a BAM's legacy ns/nl/as/al (FiberBrowser ``bam_legacy_frame``): the @PG
    chains vote; merged files with conflicting votes resolve to molecular when coord=molecular is declared
    anywhere, else to the majority (ties: SEQ)."""
    hdr = header.to_dict() if hasattr(header, 'to_dict') else dict(header or {})
    pgs = [pg for pg in (hdr.get('PG', []) or []) if isinstance(pg, dict)]
    comments = [str(c).lower() for c in (hdr.get('CO', []) or [])]
    header_marker = any(_COORD_MOLECULAR in c for c in comments)
    declared = header_marker or any(_COORD_MOLECULAR in _pg_text(pg) for pg in pgs)
    frames = [_pg_writer_frame(pg, header_marker) for pg in pgs]
    ids = [str(pg.get('ID', '') or '') for pg in pgs]
    parent = [None] * len(pgs)
    if any('PP' in pg for pg in pgs):
        by_id: Dict[str, List[int]] = {}
        for i, ident in enumerate(ids):
            by_id.setdefault(ident, []).append(i)
        for i, pg in enumerate(pgs):
            pp = str(pg.get('PP', '') or '')
            candidates = [j for j in by_id.get(pp, []) if j != i] if pp else []
            if candidates:
                earlier = [j for j in candidates if j < i]
                parent[i] = earlier[-1] if earlier else candidates[-1]
    else:
        parent = [i - 1 if i > 0 else None for i in range(len(pgs))]
    referenced = {j for j in parent if j is not None}
    leaves = [i for i in range(len(pgs)) if i not in referenced]
    if pgs and not leaves:
        leaves = [len(pgs) - 1]
    votes = {'molecular': 0, 'seq': 0}
    for leaf in leaves:
        node, seen = leaf, set()
        while node is not None and node not in seen:
            seen.add(node)
            if frames[node] is not None:
                votes[frames[node]] += 1
                break
            node = parent[node]
    mol, seq = votes['molecular'], votes['seq']
    if mol and seq:
        return (bool(declared or mol > seq), False)
    if mol or seq:
        return (bool(mol), False)
    if declared:
        return (True, False)
    return (False, True)


def read_legacy_is_molecular(read, frame):
    molecular, unknown = frame
    if molecular or not unknown:
        return molecular
    try:
        return read.has_tag('MA') and _MA_DOT_FOOTPRINT_RE.search(str(read.get_tag('MA'))) is not None
    except (AttributeError, KeyError):
        return False


def lift_hidden_predicate(header):
    """FiberBrowser's rule for an unrepaired graph lift (``ft pg-lift``): hide a primary whose stash tag
    ``Yf`` (the source FLAG) marks a source supplementary or secondary. None for any other BAM."""
    try:
        from fiberhmm.io.graph_lift import header_lift_state
        if not header_lift_state(header)['lift']:
            return None
    except Exception:
        return None

    def hidden(read):
        if read.is_unmapped or read.is_secondary or read.is_supplementary or not read.has_tag('Yf'):
            return False
        try:
            return bool(int(read.get_tag('Yf')) & 0x900)
        except (TypeError, ValueError):
            return False
    return hidden


def _record_tiles(read, lo, hi, *, legacy, group_frame, hide_terminal_gaps, terminal_gap_threshold,
                  short_terminal_footprint_bp, min_fp_quality, molecular_layers):
    """A record's tiles inside [lo, hi) as absolute ``[start, end, size]`` (deletion cuts applied), or None
    when FiberBrowser shows no row for it there."""
    ref_start, ref_end = read.reference_start, read.reference_end

    def neutral():
        a, b = max(ref_start, lo), min(ref_end, hi)
        return _cut_abs([[int(a), int(b), 0]], read) if b > a else None

    if read.is_supplementary and any(op == 5 for op, _n in read.cigartuples or []):
        return None if min_fp_quality > 0 else neutral()
    ns_tag = nl_tag = nq_tag = None
    if read.has_tag('ns'):
        ns_tag = read.get_tag('ns')
        nl_tag = read.get_tag('nl') if read.has_tag('nl') else None
        nq_tag = read.get_tag('nq') if read.has_tag('nq') and nl_tag is not None else None
    if ns_tag is None and read.has_tag('as') and read.has_tag('al'):
        try:
            ns_tag, nl_tag = msp_complement(read.get_tag('as'), read.get_tag('al'), read.query_length or 0)
        except (TypeError, ValueError):
            return None          # an MSP stream FiberBrowser cannot read either: no row
    if ns_tag is None:
        # No legacy footprints (MA-only or untagged): FiberBrowser shows the read as one neutral tile.
        return neutral()
    try:
        ns = parse_int_tag_values(ns_tag)
        nl = parse_int_tag_values(nl_tag)
        if ns is None or nl is None:
            raise ValueError('missing ns/nl')
    except Exception:
        return None
    nq = _optional_ints(nq_tag)
    if len(ns) != len(nl):
        return None
    read_length = read.query_length or 0
    if read.is_reverse and read_length > 0 and read_legacy_is_molecular(read, legacy):
        ns = [read_length - (s + l) for s, l in zip(ns, nl)]
    mapper = CigarCoordMapper(read.cigartuples, ref_start)
    ma_tags = ma_tags_of(read, group_frame)
    footprints = map_tagged_footprints(ns, nl, nq, mapper, min_fp_quality=min_fp_quality)
    ma_spans = None
    terminal_drop = set()
    if short_terminal_footprint_bp > 0:
        if ma_tags is not None:
            ma_spans = ma_footprint_query_spans(ma_tags, read_length=read_length, is_reverse=read.is_reverse,
                                                molecular_layers=molecular_layers)
        terminal_drop = short_terminal_spans(ns, nl, ma_spans, short_terminal_footprint_bp, read_length=read_length)
    all_footprints = footprints
    dropped_extent = None
    if terminal_drop:
        footprints = []
        for fp in all_footprints:
            if fp[2] < short_terminal_footprint_bp and (ns[fp[4]], ns[fp[4]] + nl[fp[4]]) in terminal_drop:
                dropped_extent = _extent_union(dropped_extent, (fp[0], fp[1]))
            else:
                footprints.append(fp)
    # Without requested layers, only a supplementary row counts as explicit evidence.
    has_explicit = bool(read.is_supplementary) and min_fp_quality <= 0
    stubs_only = bool(terminal_drop) and not footprints and bool(all_footprints)
    if not footprints and not has_explicit and not stubs_only:
        return None
    call_extent = None
    if hide_terminal_gaps and (footprints or stubs_only) and ma_tags is not None:
        if ma_spans is None:
            ma_spans = ma_footprint_query_spans(ma_tags, read_length=read_length, is_reverse=read.is_reverse,
                                                molecular_layers=molecular_layers)
        call_extent = ma_footprint_call_extent(ma_spans, mapper, ref_start, ref_end)
    if terminal_drop and hide_terminal_gaps:
        call_extent = _extent_union(call_extent, dropped_extent)
    if stubs_only:
        terminal_filtered = hide_terminal_gaps and terminal_gap_rejects(
            sorted(all_footprints, key=lambda item: item[0]), ref_start, ref_end, terminal_gap_threshold, call_extent)
        intervals = []
    else:
        intervals, terminal_filtered = build_intervals_from_footprints(
            footprints, ref_start, ref_end, lo, hi, hide_terminal_gaps=hide_terminal_gaps,
            terminal_gap_threshold=terminal_gap_threshold, call_extent=call_extent)
        intervals = [[a + lo, b + lo, size] for a, b, size in intervals]
    if terminal_filtered and not has_explicit:
        return None
    if not intervals:
        body_start, body_end = ref_start, ref_end
        if stubs_only and hide_terminal_gaps:
            call_body = _extent_union(
                (min(fp[0] for fp in all_footprints), max(fp[1] for fp in all_footprints)) if all_footprints else None,
                call_extent)
            if call_body is not None:
                body_start, body_end = max(call_body[0], ref_start), min(call_body[1], ref_end)
        a, b = max(body_start, lo), min(body_end, hi)
        if not (has_explicit or stubs_only) or b <= a:
            return None
        intervals = [[int(a), int(b), 0]]
    return _cut_abs(intervals, read)


def _cut_abs(intervals, read):
    cuts = read_deletion_cuts(read)
    return split_at_deletions(intervals, cuts) if cuts else intervals


def compact_rows(bam, chrom, region_start, region_end, *, legacy_frame_value=None, group_frame=None,
                 hide_terminal_gaps=True, terminal_gap_threshold=TERMINAL_GAP_THRESHOLD,
                 short_terminal_footprint_bp=SHORT_TERMINAL_FOOTPRINT_BP, include_duplicates=False,
                 min_fp_quality=0, molecular_layers=frozenset(), row_rule='view'):
    """Rows ``{'id', 'intervals': [[s, e, size], ...]}`` (region-relative) for the records overlapping the
    region, as FiberBrowser's loader builds them for its accessibility and TF-density peak scans (no layers).

    ``row_rule='view'`` is FiberBrowser's: rows and tiles are decided for this region (the view).
    ``row_rule='records'`` decides each record's tiles over its whole alignment, then clips them to the
    region, so the counts at a base never depend on the region's extent (grid scans)."""
    from fiberhmm.io.annotation_frame import ma_group_frame_of
    header = bam.header
    group_frame = group_frame or ma_group_frame_of(header)
    legacy = legacy_frame_value or legacy_frame(header)
    hidden = lift_hidden_predicate(header)
    opts = dict(legacy=legacy, group_frame=group_frame, hide_terminal_gaps=hide_terminal_gaps,
                terminal_gap_threshold=terminal_gap_threshold, short_terminal_footprint_bp=short_terminal_footprint_bp,
                min_fp_quality=min_fp_quality, molecular_layers=molecular_layers)
    rows = []
    for read in _anchors(bam, chrom, region_start, region_end, include_duplicates, row_rule, hidden):
        if row_rule == 'records':
            tiles = _record_tiles(read, read.reference_start, read.reference_end, **opts)
            if tiles:
                tiles = [[max(a, region_start), min(b, region_end), size] for a, b, size in tiles
                         if min(b, region_end) > max(a, region_start)]
        else:
            tiles = _record_tiles(read, region_start, region_end, **opts)
        if tiles:
            rows.append(dict(id=read.query_name,
                             intervals=[[int(a - region_start), int(b - region_start), int(size)] for a, b, size in tiles]))
    return rows
