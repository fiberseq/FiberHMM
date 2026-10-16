"""Carry annotations a writer passes through into the frame its output declares.

The invariant (docs/reference/footprint-tag-frame.md): a writer whose output
header declares a frame must express every annotation it carries through
unchanged -- MA groups (tf_sr, deam, consensus and other layers), legacy
ns/nl/as/al with nq/aq, fibertools Ma -- in that frame. Each annotation's
input frame is resolved from the INPUT header and the read's ORIGINAL tags
(before any stripping), by the shared rules (``annotation_frame``); an
annotation whose input frame cannot be decided is dropped.

fiberhmm-apply, -call and -recall-tfs/-nucs declare ``coord=molecular``, so
:func:`carry_to_molecular` is their carrier. Pass-through tools (dedup,
merge, pair, tag-m5c, ...) add no declaration that would change a group's
meaning (``annotation_frame.pass_through_frame``); tag-m5c/call-m5c given an
explicit legacy frame use :func:`convert_legacy_tags` to express the legacy
tags in the frame their output header resolves.
"""
from __future__ import annotations

import array
from typing import Callable, Optional

from fiberhmm.io.annotation_frame import MOLECULAR, legacy_frame_of, ma_group_frame_of
from fiberhmm.io.ma_tags import flip_interval_frame, parse_an_tag, parse_aq_array, parse_ma_tag

LEGACY_PAIRS = (("ns", "nl", "nq"), ("as", "al", "aq"))


def _read_length(read) -> int:
    return int(read.query_length or len(read.query_sequence or ""))


def mirror_ma_groups(read, flip: Callable[[str, str], bool]) -> None:
    """Mirror, on a reverse read, the MA groups ``flip(name, strand)`` selects.

    Each mirrored group is re-sorted by start with its AQ rows (edge bytes
    el/er, positions 1 and 2 of a 3+-byte spec, swapped) and AN names; other
    groups are kept verbatim."""
    if not read.is_reverse or not read.has_tag("MA"):
        return
    ma = str(read.get_tag("MA"))
    parsed = parse_ma_tag(ma)
    types = parsed["raw_types"]
    if not any(flip(name, strand) and intervals for name, strand, _q, intervals in types):
        return
    tokens = [token for token in ma.split(";")[1:] if token]
    if len(tokens) != len(types):
        raise ValueError(f"read {read.query_name!r}: unparsable MA groups")
    length = int(parsed["read_length"])
    aq = list(read.get_tag("AQ")) if read.has_tag("AQ") else []
    counts = [len(intervals) for _n, _s, _q, intervals in types]
    expected = sum(len(spec) * n for (_n, _s, spec, _i), n in zip(types, counts))
    if expected and len(aq) != expected:
        raise ValueError(f"read {read.query_name!r}: MA/AQ length mismatch")
    rows = parse_aq_array(aq, [t[2] for t in types], counts) if expected else [[] for _ in range(sum(counts))]
    has_an = read.has_tag("AN")
    names = parse_an_tag(str(read.get_tag("AN"))) if has_an else [""] * sum(counts)
    if len(names) != sum(counts):
        raise ValueError(f"read {read.query_name!r}: MA/AN length mismatch")
    new_tokens, new_aq, new_names, cursor = [], [], [], 0
    for token, (name, strand, spec, intervals) in zip(tokens, types):
        items = [(start, size, list(rows[cursor + i]), names[cursor + i])
                 for i, (start, size) in enumerate(intervals)]
        cursor += len(intervals)
        if flip(name, strand) and items:
            mirrored = []
            for start, size, row, label in items:
                start, size = flip_interval_frame(int(start), int(size), length)
                if len(row) >= 3:
                    row[1], row[2] = row[2], row[1]
                mirrored.append((start, size, row, label))
            items = sorted(mirrored, key=lambda item: item[0])
            token = f"{name}{strand}{spec}:" + ",".join(f"{s + 1}-{n}" for s, n, _r, _l in items)
        new_tokens.append(token)
        for _s, _n, row, label in items:
            new_aq.extend(int(v) for v in row)
            new_names.append(label)
    read.set_tag("MA", ";".join([str(length), *new_tokens]), value_type="Z")
    if read.has_tag("AQ"):
        read.set_tag("AQ", array.array("B", new_aq))
    if has_an:
        read.set_tag("AN", ",".join(n or "." for n in new_names), value_type="Z")


def convert_legacy_tags(read, source: Optional[str], target: Optional[str]) -> None:
    """Express ns/nl(+nq) and as/al(+aq) of ``source`` frame in ``target``
    frame (``'molecular'``/``'seq'``); either ``None`` (undecidable) drops
    them from a reverse read. Forward reads are unchanged."""
    if not read.is_reverse or source == target and source is not None:
        return
    length = _read_length(read)
    for start_tag, length_tag, score_tag in LEGACY_PAIRS:
        if not (read.has_tag(start_tag) and read.has_tag(length_tag)):
            continue
        if source is None or target is None:
            for tag in (start_tag, length_tag, score_tag):
                if read.has_tag(tag):
                    read.set_tag(tag, None)
            continue
        starts = [int(v) for v in read.get_tag(start_tag)]
        sizes = [int(v) for v in read.get_tag(length_tag)]
        scores = list(read.get_tag(score_tag)) if read.has_tag(score_tag) else None
        rows = []
        for index, (start, size) in enumerate(zip(starts, sizes)):
            start, size = flip_interval_frame(start, size, length)
            rows.append((start, size, scores[index] if scores and index < len(scores) else None))
        rows.sort(key=lambda row: row[0])
        read.set_tag(start_tag, array.array("I", [r[0] for r in rows]))
        read.set_tag(length_tag, array.array("I", [r[1] for r in rows]))
        if scores is not None and len(scores) == len(starts):
            read.set_tag(score_tag, array.array("B", [int(r[2]) for r in rows]))


def carry_to_molecular(read, legacy_source=None) -> None:
    """Express every annotation of ``read`` in molecular frame, for writers
    that declare ``coord=molecular`` (apply, call, recall). Call it on the
    ORIGINAL read, before stripping: the legacy frame may depend on its MA
    (no-evidence '.' rule). ``legacy_source`` overrides the legacy tags' input
    frame (an explicit ``--input-frame``: True/False or 'molecular'/'seq')."""
    header = getattr(read, "header", None)
    if header is None or not getattr(read, "is_reverse", False):
        return
    if legacy_source is None:
        source = legacy_frame_of(header).for_read(read)
    elif legacy_source in (True, MOLECULAR):
        source = MOLECULAR
    else:
        source = "seq"
    groups = ma_group_frame_of(header)
    if groups.header_frame != MOLECULAR:
        try:
            mirror_ma_groups(read, lambda name, strand: not groups.is_molecular(name, strand))
        except ValueError as error:   # malformed MA/AQ/AN: left as it was
            import warnings
            warnings.warn(f"cannot carry MA groups into molecular frame: {error}",
                          RuntimeWarning, stacklevel=2)
    if any(read.has_tag(tag) for tag in ("ns", "as")):
        convert_legacy_tags(read, source, MOLECULAR)


def carry_legacy_into(read, source_molecular: bool, output_frame) -> None:
    """For pass-through writers that read the legacy tags in a frame of
    their own (an explicit ``--input-frame``): express them in the frame the
    output header resolves (``output_frame``, an ``annotation_frame.LegacyFrame``)
    when the two differ; an output that cannot decide leaves them as they are
    (their meaning is unchanged: still undecided)."""
    if not getattr(read, "is_reverse", False) or not any(read.has_tag(t) for t in ("ns", "as")):
        return
    target = output_frame.for_read(read)
    source = MOLECULAR if source_molecular else "seq"
    if target is not None and target != source:
        convert_legacy_tags(read, source, target)
