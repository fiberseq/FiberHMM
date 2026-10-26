"""Lift FiberHMM BAMs through a pangenome graph and repair what the lift breaks.

The vg route (fiberhmm-lift's fallback engine) moves reads from one assembly
to another through a GBZ graph: ``ft pg-pansn`` adds the panSN prefix, ``vg inject``
places the alignments on the graph, ``vg surject`` projects them onto the
target paths, and ``ft pg-pansn --strip`` removes the prefix again. SEQ, QUAL
and every aux tag are copied verbatim, so molecular-frame footprints stay
right, but the lifted records are not consistent BAM records:

- supplementaries lose 0x800 and come out as extra primaries;
- hard clips are dropped (``649M1845H`` becomes ``649M``), so the record no
  longer agrees with the read length in ``MA``;
- MD, NM, SA (and minimap2's ``cs`` difference string) still describe the
  source alignment;
- ``st:Z`` (DAF deamination strand, read in SEQ orientation) is not swapped
  when the read changes strand, so DAF readers see no deamination;
- SEQ-frame footprint annotations are mirrored on reads whose strand changed;
- the vg and pg-pansn @PG lines are not linked to the source history, and
  the output is unsorted.

:func:`stash_bam` runs before the lift and records what the repair needs in
aux tags (which the lift copies verbatim); :func:`repair_bam` runs after it.

The native engine replaces the vg stages with graphlift's ``liftbam``, which
lifts the stashed records directly through a graphlift chain map: it keeps
hard clips (swapped on a strand change), sets 0x800 on the extra records of a
split and drops MD/NM/SA, and leaves records carrying the ``Yf`` stash to this
repair (``st`` swap, MAPQ mode, calmd, SA, @PG). The repair accepts both
outputs: 0x800 is kept from the lift, and hard clips are re-added only when
the lifted record has none.
For a BAM lifted without the stash (by a vg-based tool run directly),
:func:`stash_from_source` rebuilds the same tags by matching each lifted
record with its source record, so one repair serves both routes.

Aux tags (SAM user space ``Y?``; none is written by FiberHMM, fibertools-rs,
minimap2, pbmm2 or vg):

``Yf:i`` (stash; removed by the repair)
    source FLAG.
``Yo:i`` (stash; removed by the repair)
    1 on a soft-clipped supplementary whose primary is not in the input (a
    region slice, say): it carries its own SEQ and calls, is lifted, and the
    repair keeps it although no primary of its read lifted.
``Yh:Z`` (stash; removed by the repair)
    hard clips of the source record as ``<kind>:<left>,<right>`` in its CIGAR
    order. ``S`` means the stash filled the clipped bases from the primary
    and turned them into soft clips (nothing to do after the lift); ``H``
    means they are still missing and the repair re-adds the hard clips.
``Yq:i`` (provenance; kept)
    source MAPQ. The repaired record keeps this value as its MAPQ too.
``Yp:Z`` (provenance; kept)
    source locus ``<contig>:<1-based POS>:<strand>``; marks a lifted record.

A lifted BAM can be lifted again; ``Yq``/``Yp`` then describe the BAM of the
latest lift, and the @PG chain records every lift.

Decisions:

- MAPQ stays the source value by default (as CrossMap does):
  how uniquely the read placed on the source assembly. Against a diploid
  assembly that is low wherever the haplotypes are identical.
  ``mapq='fixed:N'`` sets every lifted record to N instead; ``Yq`` keeps the
  source value either way, and the @PG records the mode.
- Hard-clipped supplementaries get their clipped bases (and QUAL) from the
  primary of the same QNAME before the lift, as soft clips, so the record
  carries the whole molecule; their own footprint calls and modifications
  (supplementaries are called separately) are rebased onto the filled SEQ
  (:func:`fill_hard_clips`). A
  hard-clipped supplementary whose primary is not in the input is excluded
  and counted; a soft-clipped one carries its own SEQ, calls and
  modifications, and is lifted (``Yo``).
- Footprint annotations are converted to molecular frame before the lift
  (:func:`fiberhmm.io.frame_carry.carry_to_molecular`); only molecular
  coordinates survive a strand change. ``MA`` m5C groups and fibertools
  ``Ma`` are molecular already.
- Without a target FASTA the repair drops MD, NM and SA (SA needs NM) and
  says so in its @PG; with one it recomputes MD/NM (``samtools calmd``) and
  rebuilds SA from the lifted sibling records.
- Secondary and unmapped source records are not lifted; supplementaries
  whose primary did not lift are dropped (no promotion), so every QNAME
  keeps exactly one primary.
"""
from __future__ import annotations

import array
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pysam

TAG_FLAG = "Yf"
TAG_HARDCLIP = "Yh"
TAG_ORPHAN = "Yo"
TAG_SOURCE_MAPQ = "Yq"
TAG_SOURCE_LOCUS = "Yp"
STASH_TAGS = (TAG_FLAG, TAG_HARDCLIP, TAG_ORPHAN)
PROVENANCE_TAGS = (TAG_SOURCE_MAPQ, TAG_SOURCE_LOCUS)
# Tags that describe the source alignment against the source reference.
# ``cs`` only in minimap2's difference-string form: FiberHMM's joint duplex
# molecules use ``cs:Z`` for their source names.
REFERENCE_TAGS = ("MD", "NM")
_MINIMAP2_CS = re.compile(r"^[:*+~=-]")

LIFT_PROGRAM = "fiberhmm-lift"

_SUPP = 0x800
_SECONDARY = 0x100
_UNMAPPED = 0x4
_REVERSE = 0x10
_HARD_CLIP = 5
_SOFT_CLIP = 4

# IUPAC-aware complement: DAF reads carry R/Y in SEQ.
_COMPLEMENT = str.maketrans("ACGTRYKMBVDHSWNacgtrykmbvdhswn",
                            "TGCAYRMKVBHDSWNtgcayrmkvbhdswn")
# 4-bit base sets (htslib nt16): two bases agree when their sets intersect.
_NT16 = np.zeros(256, dtype=np.uint8)
for _base, _code in zip("=ACMGRSVTWYHKDBN", range(16)):
    _NT16[ord(_base)] = _code
    _NT16[ord(_base.lower())] = _code


def revcomp(seq: str) -> str:
    """Reverse complement, IUPAC codes included (R<->Y, K<->M, B<->V, D<->H)."""
    return seq.translate(_COMPLEMENT)[::-1]


def _bases_agree(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    if not a:
        return True
    x = _NT16[np.frombuffer(a.encode(), dtype=np.uint8)]
    y = _NT16[np.frombuffer(b.encode(), dtype=np.uint8)]
    return bool(np.all((x & y) != 0))


def _hard_clips(read) -> Tuple[int, int]:
    cigar = read.cigartuples or []
    left = cigar[0][1] if cigar and cigar[0][0] == _HARD_CLIP else 0
    right = cigar[-1][1] if len(cigar) > 1 and cigar[-1][0] == _HARD_CLIP else 0
    return left, right


def _source_locus(read) -> str:
    return f"{read.reference_name}:{read.reference_start + 1}:{'-' if read.is_reverse else '+'}"


# ---------------------------------------------------------------------------
# Pre-lift stash
# ---------------------------------------------------------------------------

@dataclass
class StashStats:
    records_in: int = 0
    stashed: int = 0
    dropped_secondary: int = 0
    dropped_unmapped: int = 0
    filled_supplementary: int = 0
    excluded_supplementary: int = 0
    orphan_supplementary_no_primary_in_input: int = 0
    orphan_supplementary_kept: int = 0
    excluded_unverifiable: int = 0
    mods_dropped: int = 0
    hard_clipped_kept: int = 0
    frame_converted: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


class FrameError(ValueError):
    """The footprint annotations' frame cannot be decided."""


class AnnotationError(FrameError):
    """A record's footprint tags cannot be carried to molecular frame."""


def _require_legacy_frame(read) -> None:
    """Refuse a reverse read whose legacy ns/nl/as/al frame is undecidable.

    :func:`~fiberhmm.io.frame_carry.carry_to_molecular` would silently drop
    those tags; lifting is not the place to lose them. Checked on every
    record that is converted, not on a sample."""
    if not read.is_reverse or not (read.has_tag("ns") or read.has_tag("as")):
        return
    from fiberhmm.io.annotation_frame import legacy_frame_of

    resolver = legacy_frame_of(read.header)
    if resolver.for_read(read) is None:
        raise FrameError(
            f"cannot lift record {read.query_name!r}: the frame of its ns/nl/as/al tags "
            f"is undecidable ({resolver.reason}), and a strand change would leave them "
            "mirrored. Split the BAM by input, or re-call it with FiberHMM >= 3.0, then lift.")


# --- base modifications (MM/ML) ----------------------------------------------------
#
# Decoding is pysam's (htslib's), which accepts the whole SAMtags grammar:
# multi-letter, upper-case and ChEBI codes, N+ and U+ groups, and MM without
# ML. The modes ('.', '?') and the group order come from the MM heads.

_MM_HEAD = re.compile(r"^([ACGTUN])([+-])([A-Za-z]+|\d+)([.?]?)$")


def _mod_candidates(seq: str, base: str) -> List[int]:
    if base == "N":
        return list(range(len(seq)))
    bases = {"T", "U"} if base in ("T", "U") else {base}
    return [i for i, b in enumerate(seq) if b.upper() in bases]


def _mod_tags(read) -> Optional[Tuple[str, str]]:
    """``(MM tag, ML tag)`` names in use, each resolved on its own (``MM``
    before ``Mm``, ``ML`` before ``Ml``), as htslib does."""
    mm_tag = next((t for t in ("MM", "Mm") if read.has_tag(t)), None)
    if mm_tag is None:
        return None
    return mm_tag, next((t for t in ("ML", "Ml") if read.has_tag(t)), "ML")


def decode_mods(read) -> Optional[dict]:
    """The record's base modifications in original read orientation, or None
    when it has none.

    ``{'names': (MM, ML), 'has_ml': bool, 'groups': [(base, strand, code,
    mode, [(pos, ml), ...]), ...]}`` (one group per code, in MM order), or
    ``{'names': ..., 'groups': None}`` when htslib cannot decode them against
    this record's SEQ. htslib files U groups under T."""
    names = _mod_tags(read)
    if names is None:
        return None
    info = {"names": names, "has_ml": read.has_tag(names[1]), "groups": None}
    heads = []
    for chunk in str(read.get_tag(names[0])).split(";"):
        if not chunk:
            continue
        match = _MM_HEAD.match(chunk.split(",", 1)[0])
        if not match:
            return info
        base, strand, codes, mode = match.groups()
        for code in ([int(codes)] if codes.isdigit() else list(codes)):
            # htslib pools the calls of repeated heads; one group per code.
            same = [h for h in heads if h[:3] == (base, strand, code)]
            if same and same[0][3] != (mode or "."):
                return info               # repeated heads that disagree on mode
            if not same:
                heads.append((base, strand, code, mode or "."))
    try:
        decoded = read.modified_bases_forward
    except (ValueError, KeyError):
        decoded = None
    if decoded is None:
        return info
    found = 0
    groups = []
    for base, strand, code, mode in heads:
        key_base = "T" if base == "U" else base
        calls = decoded.get((key_base, 0 if strand == "+" else 1, code), [])
        found += len(calls)
        groups.append((base, strand, code, mode, [(int(p), int(q)) for p, q in calls]))
    if found != sum(len(v) for v in decoded.values()):
        return info                       # a call htslib decoded under another key
    info["groups"] = groups
    return info


def encode_mods(groups: list, seq: str, has_ml: bool = True) -> Tuple[str, array.array, int]:
    """``(MM, ML, dropped)`` of decoded ``groups`` on ``seq`` (original
    orientation); calls whose base in ``seq`` is no longer the group's base
    are dropped."""
    mm_parts, ml, dropped = [], array.array("B"), 0
    for base, strand, code, mode, calls in groups:
        rank = {pos: k for k, pos in enumerate(_mod_candidates(seq, base))}
        deltas, previous = [], -1
        for pos, value in sorted(calls):
            k = rank.get(pos)
            if k is None:
                dropped += 1
                continue
            deltas.append(str(k - previous - 1))
            previous = k
            if has_ml:
                ml.append(max(0, min(255, int(value))))
        mm_parts.append(f"{base}{strand}{code}{mode}" + "".join("," + d for d in deltas) + ";")
    return "".join(mm_parts), ml, dropped


def _gains_candidates(base: str, old: str, new: str) -> bool:
    return bool(set(_mod_candidates(new, base)) - set(_mod_candidates(old, base)))


# --- hard-clipped supplementaries --------------------------------------------------
#
# FiberHMM (3.x, alignments=primary-supplementary) and fibertools call each
# supplementary on its own aligned bases, so its footprint annotations are
# its own and describe its segment, not the primary's: they are kept and
# rebased onto the filled sequence, never replaced by the primary's. Their
# scope is decided once per record, before any frame conversion, from the
# read length MA (else Ma) states; legacy tags follow it (the same writer
# wrote them). Base modifications are the record's own when it has valid
# ones, else the primary's re-encoded on the filled SEQ.

# Footprint call tags FiberHMM and fibertools write.
FOOTPRINT_TAGS = ("MA", "AQ", "AN", "Ma", "Aq", "An", "ns", "nl", "nq", "as", "al", "aq", "hl")
_MOD_TAGS = ("MM", "ML", "Mm", "Ml", "MN")


def annotations_valid(read, length: int) -> bool:
    """The record's footprint tags describe a molecule of ``length``
    consistently: each MA/Ma parses (:func:`fiberhmm.io.ma_tags.parse_ma_tag`)
    with that read length, intervals inside it, one AQ/Aq byte per quality
    letter per interval and one AN/An name per interval; legacy starts,
    lengths and scores agree in number and lie inside it."""
    from fiberhmm.io.ma_tags import parse_an_tag, parse_ma_tag

    for ma_tag, aq_tag, an_tag in (("MA", "AQ", "AN"), ("Ma", "Aq", "An")):
        if not read.has_tag(ma_tag):
            continue
        try:
            parsed = parse_ma_tag(str(read.get_tag(ma_tag)))
        except ValueError:
            return False
        if parsed["read_length"] != length:
            return False
        types = parsed["raw_types"]
        intervals = [iv for *_h, ivs in types for iv in ivs]
        if any(s < 0 or n < 0 or s + n > length for s, n in intervals):
            return False
        expected = sum(len(spec) * len(ivs) for _n, _s, spec, ivs in types)
        aq = list(read.get_tag(aq_tag)) if read.has_tag(aq_tag) else []
        if len(aq) != expected:
            return False
        if read.has_tag(an_tag) and len(parse_an_tag(str(read.get_tag(an_tag)))) != len(intervals):
            return False
    for start_tag, size_tag, score_tag in (("ns", "nl", "nq"), ("as", "al", "aq")):
        if not (read.has_tag(start_tag) or read.has_tag(size_tag)):
            continue
        if not (read.has_tag(start_tag) and read.has_tag(size_tag)):
            return False
        starts, sizes = list(read.get_tag(start_tag)), list(read.get_tag(size_tag))
        if len(starts) != len(sizes) or any(s + n > length for s, n in zip(starts, sizes)):
            return False
        if read.has_tag(score_tag) and len(read.get_tag(score_tag)) != len(starts):
            return False
    return True




def _primary_info(read) -> dict:
    """What a hard-clipped supplementary takes from its primary: the
    flanking bases and qualities, and its base modifications (used when the
    supplementary has none of its own). Call it on the primary after
    :func:`_to_molecular`. A primary whose footprint tags are inconsistent
    stops the run (it would be published under coord=molecular)."""
    seq = read.query_sequence or ""
    quals = read.query_qualities
    mol_seq = revcomp(seq) if read.is_reverse else seq
    mol_qual = None
    if quals is not None:
        mol_qual = list(quals)[::-1] if read.is_reverse else list(quals)
    if not annotations_valid(read, len(mol_seq)):
        raise AnnotationError(
            f"cannot lift record {read.query_name!r}: its footprint tags (MA/AQ/AN, Ma/Aq/An "
            f"or ns/nl/as/al) do not describe its {len(mol_seq)} bp molecule consistently. "
            "Fix or strip them, then lift.")
    return {"seq": mol_seq, "qual": mol_qual, "mods": decode_mods(read)}


UNVERIFIABLE = "annotations or modifications cannot be carried over unambiguously"


def _shift_ma(value: str, offset: int, total: int) -> str:
    pieces = str(value).split(";")
    out = [str(total)]
    for chunk in pieces[1:]:
        head, sep, body = chunk.partition(":")
        if sep and body:
            items = []
            for item in body.split(","):
                start, dash, size = item.partition("-")
                items.append(f"{int(start) + offset}{dash}{size}" if dash else item)
            body = ",".join(items)
        out.append(head + sep + body)
    return ";".join(out)


def annotation_scope(read, partial: int, total: int) -> Optional[str]:
    """``'partial'`` (the record's stored SEQ), ``'whole'`` (the molecule),
    ``'none'`` (no footprint tags) or None (undecidable) for a hard-clipped
    record's footprint tags, decided before any frame conversion.

    MA's (else Ma's) read length decides; legacy tags follow it. A record with
    legacy tags only: partial when every interval fits its stored SEQ, whole
    when one does not but all fit the molecule."""
    for tag in ("MA", "Ma"):
        if read.has_tag(tag):
            try:
                length = int(str(read.get_tag(tag)).split(";", 1)[0])
            except ValueError:
                return None
            return "partial" if length == partial else "whole" if length == total else None
    ends = []
    for start_tag, size_tag in (("ns", "nl"), ("as", "al")):
        if read.has_tag(start_tag) and read.has_tag(size_tag):
            ends += [s + n for s, n in zip(read.get_tag(start_tag), read.get_tag(size_tag))]
        elif read.has_tag(start_tag) or read.has_tag(size_tag):
            return None
    if not ends:
        return "none"
    if max(ends) <= partial:
        return "partial"
    return "whole" if max(ends) <= total else None


def _shift_footprints(read, offset: int, total: int) -> None:
    for tag in ("MA", "Ma"):
        if read.has_tag(tag):
            read.set_tag(tag, _shift_ma(str(read.get_tag(tag)), offset, total), value_type="Z")
    for tag in ("ns", "as"):
        if read.has_tag(tag):
            read.set_tag(tag, array.array("I", [int(v) + offset for v in read.get_tag(tag)]))


def _mark_unassessed(groups, old_mol: str, new_mol: str) -> list:
    """A '.' group says its unlisted candidate bases are unmodified; one that
    gains a candidate the source never assessed becomes '?'."""
    return [(b, st, c, "?" if m == "." and _gains_candidates(b, old_mol, new_mol) else m, calls)
            for b, st, c, m, calls in groups]


def fill_hard_clips(read, primary: dict, notes: Optional[Dict[str, int]] = None) -> Optional[str]:
    """Fill a hard-clipped record's clipped bases from its primary, as soft
    clips, and rebase its own calls onto the filled sequence.

    SEQ: the record's own aligned bases (DAF R/Y encoding is per record)
    with the clipped flanks from the primary; QUAL likewise. Footprint tags
    stay the record's own (:func:`annotation_scope`): partial ones are
    carried to molecular frame on the stored SEQ, then moved by the molecule
    offset with the read length set to the molecule's; whole-molecule ones
    are carried on the filled SEQ. MM/ML: the record's own (MN = stored
    length, or no MN and valid on the stored SEQ: moved; MN = molecule
    length: kept), else the primary's re-encoded on the filled SEQ (a call
    on a base the record changed, C -> Y, is dropped and counted in
    ``notes['mods_dropped']``). Returns None on success, else why the record
    must be excluded (:data:`UNVERIFIABLE` when its tags cannot be carried
    over unambiguously); an excluded record may have been changed."""
    left, right = _hard_clips(read)
    if not (left or right):
        return None
    seq = read.query_sequence or ""
    partial = len(seq)
    total = left + partial + right
    mol = primary["seq"]
    if len(mol) != total:
        return f"primary length {len(mol)} != {total}"
    full = revcomp(mol) if read.is_reverse else mol
    if not _bases_agree(full[left:left + partial], seq):
        return "primary SEQ does not match the supplementary"
    scope = annotation_scope(read, partial, total)
    if scope is None or (scope != "none" and not annotations_valid(
            read, partial if scope == "partial" else total)):
        return UNVERIFIABLE
    offset = right if read.is_reverse else left        # molecule offset of the stored bases
    own_mol = revcomp(seq) if read.is_reverse else seq

    names = _mod_tags(read)
    mn = int(read.get_tag("MN")) if read.has_tag("MN") else None
    own_mods, keep_own = None, False
    if names is not None:
        if mn == total:
            keep_own = True
        elif mn is None or mn == partial:
            own_mods = decode_mods(read)
            if own_mods["groups"] is None:
                own_mods = None
    source = None
    if not keep_own and own_mods is None:
        source = primary["mods"]
        if source is not None and source["groups"] is None:
            return UNVERIFIABLE

    try:
        if scope == "partial":
            _to_molecular(read)                         # on the stored SEQ it describes
    except AnnotationError:
        return UNVERIFIABLE

    quals = read.query_qualities
    new_qual = None
    if quals is not None and primary["qual"] is not None:
        mq = primary["qual"][::-1] if read.is_reverse else primary["qual"]
        new_qual = array.array("B", list(mq[:left]) + list(quals) + list(mq[left + partial:]))
    cigar = list(read.cigartuples)
    if left:
        cigar[0] = (_SOFT_CLIP, left)
    if right:
        cigar[-1] = (_SOFT_CLIP, right)
    new_seq = full[:left] + seq + full[left + partial:]
    # Setting SEQ clears QUAL; QUAL is set again only when the primary has one.
    read.query_sequence = new_seq
    read.cigartuples = cigar
    if new_qual is not None:
        read.query_qualities = new_qual
    new_mol = revcomp(new_seq) if read.is_reverse else new_seq

    if scope == "partial":
        _shift_footprints(read, offset, total)
    elif scope == "whole":
        try:
            _to_molecular(read)                         # on the filled SEQ it describes
        except AnnotationError:
            return UNVERIFIABLE                         # the record is excluded

    if keep_own:
        return None
    if own_mods is not None:
        shifted_old = " " * offset + own_mol            # the record's bases at molecule positions
        groups = [(b, st, c, m, [(p + offset, v) for p, v in calls])
                  for b, st, c, m, calls in own_mods["groups"]]
        groups = _mark_unassessed(groups, shifted_old, new_mol)
    elif source is not None:
        own_mods = source
        groups = _mark_unassessed(source["groups"], mol, new_mol)
    else:
        for tag in _MOD_TAGS:
            if read.has_tag(tag):
                read.set_tag(tag, None)
        return None
    for tag in _MOD_TAGS:
        if read.has_tag(tag):
            read.set_tag(tag, None)
    mm, ml, dropped = encode_mods(groups, new_mol, own_mods["has_ml"])
    read.set_tag(own_mods["names"][0], mm, value_type="Z")
    if own_mods["has_ml"]:
        read.set_tag(own_mods["names"][1], ml)
    read.set_tag("MN", len(new_seq), value_type="i")
    if notes is not None and dropped:
        notes["mods_dropped"] = notes.get("mods_dropped", 0) + dropped
    return None


def _stash_tags(read, hard: Optional[str]) -> None:
    read.set_tag(TAG_FLAG, int(read.flag), value_type="i")
    read.set_tag(TAG_SOURCE_MAPQ, int(read.mapping_quality), value_type="i")
    read.set_tag(TAG_SOURCE_LOCUS, _source_locus(read), value_type="Z")
    if hard:
        read.set_tag(TAG_HARDCLIP, hard, value_type="Z")
    elif read.has_tag(TAG_HARDCLIP):
        read.set_tag(TAG_HARDCLIP, None)


def _to_molecular(read, stats: Optional["StashStats"] = None) -> bool:
    """Carry the record's footprint tags into molecular frame. A record
    whose MA cannot be carried (malformed MA/AQ/AN) stops the run
    (:class:`AnnotationError`): it would be published under coord=molecular
    in SEQ frame."""
    import warnings

    from fiberhmm.io.frame_carry import carry_to_molecular

    if not read.is_reverse:
        return True
    _require_legacy_frame(read)
    before = (read.get_tag("MA") if read.has_tag("MA") else None,
              tuple(read.get_tag("ns")) if read.has_tag("ns") else None,
              tuple(read.get_tag("as")) if read.has_tag("as") else None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        carry_to_molecular(read)
    failed = [str(w.message) for w in caught if "cannot carry" in str(w.message)]
    if failed:
        raise AnnotationError(
            f"cannot lift record {read.query_name!r}: {failed[0]}. Its footprint tags would be "
            "published as molecular while still in SEQ frame; fix or strip them, then lift.")
    for warning in caught:
        warnings.warn(warning.message, warning.category, stacklevel=2)
    after = (read.get_tag("MA") if read.has_tag("MA") else None,
             tuple(read.get_tag("ns")) if read.has_tag("ns") else None,
             tuple(read.get_tag("as")) if read.has_tag("as") else None)
    if stats is not None and before != after:
        stats.frame_converted += 1
    return True


def stash_bam(in_path: str, out_path: str, *, threads: int = 1) -> StashStats:
    """Write ``in_path`` ready for lifting: stash tags, molecular frame,
    hard-clipped supplementaries filled from their primary.

    Secondary and unmapped records are dropped (counted). Two passes: the
    first finds the QNAMEs with hard-clipped supplementaries; the second
    writes, holding a hard-clipped supplementary back until its primary has
    been seen. A reverse record whose legacy footprint frame is undecidable
    stops the run (:class:`FrameError`)."""
    stats = StashStats()
    need, supp_names, with_primary = set(), set(), set()
    with pysam.AlignmentFile(in_path, "rb", check_sq=False, threads=threads) as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_supplementary and not read.is_unmapped:
                supp_names.add(read.query_name)
                if any(_hard_clips(read)):
                    need.add(read.query_name)
    if supp_names:
        # Which of those have a primary record at all (second pass: the
        # primary may come before or after its supplementaries).
        with pysam.AlignmentFile(in_path, "rb", check_sq=False, threads=threads) as bam:
            for read in bam.fetch(until_eof=True):
                if (read.query_name in supp_names and not read.is_supplementary
                        and not read.is_secondary):
                    with_primary.add(read.query_name)

    primaries: Dict[str, dict] = {}
    waiting: Dict[str, list] = defaultdict(list)
    with pysam.AlignmentFile(in_path, "rb", check_sq=False, threads=threads) as bam, \
            pysam.AlignmentFile(out_path, "wb", header=bam.header, threads=threads) as out:

        notes: Dict[str, int] = {}

        def emit_supplementary(read, clips, primary):
            reason = fill_hard_clips(read, primary, notes)
            stats.mods_dropped = notes.get("mods_dropped", 0)
            if reason is not None:
                stats.excluded_supplementary += 1
                if reason == UNVERIFIABLE:
                    stats.excluded_unverifiable += 1
                return
            stats.filled_supplementary += 1
            _stash_tags(read, "S:{},{}".format(*clips))
            out.write(read)
            stats.stashed += 1

        for read in bam.fetch(until_eof=True):
            stats.records_in += 1
            if read.is_unmapped:
                stats.dropped_unmapped += 1
                continue
            if read.is_secondary:
                stats.dropped_secondary += 1
                continue
            name = read.query_name
            clips = _hard_clips(read)
            orphan = read.is_supplementary and name not in with_primary
            if orphan and any(clips):
                # Its primary was never in the input: nothing to fill its clips from.
                stats.orphan_supplementary_no_primary_in_input += 1
                continue
            hard_supp = read.is_supplementary and any(clips)
            # A hard-clipped supplementary is converted in fill_hard_clips,
            # once the scope of its annotations is known.
            if not hard_supp:
                _to_molecular(read, stats)
            if hard_supp:
                if name in primaries:
                    emit_supplementary(read, clips, primaries[name])
                else:
                    waiting[name].append((read, clips))
                continue
            if name in need and not read.is_supplementary:
                primaries[name] = _primary_info(read)
                for held, held_clips in waiting.pop(name, []):
                    emit_supplementary(held, held_clips, primaries[name])
            hard = None
            if any(clips):
                hard = "H:{},{}".format(*clips)
                stats.hard_clipped_kept += 1
            _stash_tags(read, hard)
            if orphan:
                # Soft-clipped: its own SEQ, calls and MM; lifted and kept.
                read.set_tag(TAG_ORPHAN, 1, value_type="i")
                stats.orphan_supplementary_kept += 1
            out.write(read)
            stats.stashed += 1
        for held in waiting.values():
            stats.excluded_supplementary += len(held)
    return stats


# ---------------------------------------------------------------------------
# Rebuilding the stash for a BAM lifted without it
# ---------------------------------------------------------------------------

def _molecular_span(read, length: int) -> Tuple[int, int]:
    start, end = read.query_alignment_start, read.query_alignment_end
    return (length - end, length - start) if read.is_reverse else (start, end)


def _molecular_seq_key(read) -> int:
    seq = read.query_sequence or ""
    return hash(revcomp(seq) if read.is_reverse else seq)


@dataclass
class SourceMatchStats:
    matched: int = 0
    unmatched: int = 0
    ambiguous: int = 0
    already_stashed: int = 0


def _assign(lifted: list, sources: list) -> Dict[int, Optional[int]]:
    """One-to-one match of a QNAME's lifted records with its source records.

    ``lifted``: ``[(index, length, seq_key, span)]``; ``sources``:
    ``[(length, seq_key, span, identity)]``. A pair needs the same SEQ (in
    molecular orientation) and is scored by the overlap of the aligned part of
    that SEQ. Pairs are taken best first, and the ties are re-checked against
    the sources still free at every step: a lifted record whose best free
    score is shared by sources of different identity, or whose best source is
    equally the best of another unassigned lifted record, is ambiguous (None)
    rather than resolved by input order. Returns ``{index: source or None}``
    for the records that had a candidate at all."""
    options = {}
    for index, length, key, span in lifted:
        scored = [(min(span[1], s[2][1]) - max(span[0], s[2][0]), j)
                  for j, s in enumerate(sources) if s[0] == length and s[1] == key]
        if scored:
            options[index] = scored
    result: Dict[int, Optional[int]] = {}
    used = set()
    while True:
        best = {}
        for index, scored in options.items():
            if index in result:
                continue
            free = [(score, j) for score, j in scored if j not in used]
            if not free:
                continue
            top = max(score for score, _j in free)
            best[index] = (top, [j for score, j in free if score == top])
        if not best:
            break
        index = max(best, key=lambda i: (best[i][0], -i))
        top, js = best[index]
        if len({sources[j][3] for j in js}) > 1:
            result[index] = None
            continue
        rivals = [i for i, (score, others) in best.items()
                  if i != index and score == top and set(others) & set(js)]
        if rivals and len(js) == 1:
            for i in [index, *rivals]:
                result[i] = None
            continue
        result[index] = js[0]
        used.add(js[0])
    return result


def stash_from_source(lifted_path: str, source_path: str, out_path: str, *,
                      threads: int = 1) -> SourceMatchStats:
    """Add the stash tags to a BAM lifted without :func:`stash_bam`.

    Each lifted record is matched one-to-one with a source record of the
    same QNAME and the same SEQ in molecular orientation (the lift copies SEQ,
    reverse-complementing it on a strand change); among those, the aligned
    part of the SEQ decides. Unmatched and ambiguous records are dropped
    (counted). SEQ-frame footprint annotations are carried into molecular
    frame using the source orientation. Holds one summary per source and per
    lifted record in memory."""
    from fiberhmm.io.frame_carry import carry_to_molecular

    stats = SourceMatchStats()
    source: Dict[str, list] = defaultdict(list)
    with_primary = set()
    with pysam.AlignmentFile(source_path, "rb", check_sq=False, threads=threads) as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped or read.is_secondary:
                continue
            if not read.is_supplementary:
                with_primary.add(read.query_name)
            length = read.query_length
            identity = (int(read.flag), int(read.mapping_quality), _source_locus(read),
                        _hard_clips(read))
            source[read.query_name].append(
                (length, _molecular_seq_key(read), _molecular_span(read, length), identity))
    lifted: Dict[str, list] = defaultdict(list)
    with pysam.AlignmentFile(lifted_path, "rb", check_sq=False, threads=threads) as bam:
        for index, read in enumerate(bam.fetch(until_eof=True)):
            if read.has_tag(TAG_FLAG) or read.is_unmapped:
                continue
            length = read.query_length
            lifted[read.query_name].append(
                (index, length, _molecular_seq_key(read), _molecular_span(read, length)))
    assignment: Dict[int, Optional[int]] = {}
    for name, records in lifted.items():
        for index, j in _assign(records, source.get(name, [])).items():
            assignment[index] = None if j is None else source[name][j][3]
    del lifted
    with pysam.AlignmentFile(lifted_path, "rb", check_sq=False, threads=threads) as bam, \
            pysam.AlignmentFile(out_path, "wb", header=bam.header, threads=threads) as out:
        for index, read in enumerate(bam.fetch(until_eof=True)):
            if read.has_tag(TAG_FLAG):
                stats.already_stashed += 1
                out.write(read)
                continue
            if index not in assignment:
                stats.unmatched += 1
                continue
            identity = assignment[index]
            if identity is None:
                stats.ambiguous += 1
                continue
            flag, mapq, locus, clips = identity
            lifted_flag = read.flag
            # SEQ-frame annotations were written against the source orientation.
            read.flag = (lifted_flag & ~_REVERSE) | (flag & _REVERSE)
            _to_molecular(read)
            read.flag = lifted_flag
            read.set_tag(TAG_FLAG, flag, value_type="i")
            read.set_tag(TAG_SOURCE_MAPQ, mapq, value_type="i")
            read.set_tag(TAG_SOURCE_LOCUS, locus, value_type="Z")
            if any(clips):
                read.set_tag(TAG_HARDCLIP, "H:{},{}".format(*clips), value_type="Z")
            elif flag & _SUPP and read.query_name not in with_primary:
                # As stash_bam: a soft-clipped supplementary whose primary is not in
                # the source carries its own SEQ and calls; the repair keeps it.
                read.set_tag(TAG_ORPHAN, 1, value_type="i")
            out.write(read)
            stats.matched += 1
    return stats


# ---------------------------------------------------------------------------
# @PG history
# ---------------------------------------------------------------------------

def _is_vg(pg: dict) -> bool:
    return str(pg.get("PN", "") or "").strip().lower() == "vg"


def _is_ft_lift(pg: dict) -> bool:
    tokens = str(pg.get("CL", "") or "").split()
    return bool(tokens) and os.path.basename(tokens[0]) in ("ft", "fibertools", "fibertools-rs") \
        and any(t in ("pg-pansn", "pg-lift", "pg-inject") for t in tokens[1:3])


def _is_vg_lift_cl(pg: dict) -> bool:
    return re.match(r"^\S*\bvg\s+(surject|inject)\b", str(pg.get("CL", "") or "")) is not None


LIFTBAM_PROGRAM = "liftbam"


def _is_liftbam(pg: dict) -> bool:
    """graphlift's native BAM lift (``liftbam``)."""
    if str(pg.get("PN", "") or "").strip().lower() == LIFTBAM_PROGRAM:
        return True
    tokens = str(pg.get("CL", "") or "").split()
    return bool(tokens) and os.path.basename(tokens[0]) == LIFTBAM_PROGRAM


def is_lift_pg(pg: dict) -> bool:
    """A @PG line that by itself shows a graph lift: ft pg-pansn / pg-lift /
    pg-inject, vg inject/surject in CL, or graphlift's liftbam. vg's own
    ``@PG ID:0 PN:vg`` has no CL and is a lift line only beside those (a vg
    giraffe alignment writes the same line)."""
    return _is_ft_lift(pg) or _is_vg_lift_cl(pg) or _is_liftbam(pg)


def is_repair_pg(pg: dict) -> bool:
    pn = str(pg.get("PN", "") or "").strip().lower()
    ident = str(pg.get("ID", "") or "").strip().lower()
    return pn == LIFT_PROGRAM or re.match(rf"^{LIFT_PROGRAM}(?:\.\d+)?$", ident) is not None


def _pg_key(pg: dict) -> tuple:
    return tuple(sorted((str(k), str(v)) for k, v in pg.items()))


def pending_lift_indices(pgs: List[dict], source_pgs: Optional[List[dict]] = None) -> List[int]:
    """Indices of the @PG lines of a lift that no fiberhmm-lift has repaired.

    With ``source_pgs`` (the @PG lines of the BAM that was lifted), the lift's
    lines are the lift-like lines (:func:`is_lift_pg`, or ``PN:vg``) that are
    not copies of a source line, matched as a multiset of whole records
    (latest copies first: vg puts its own line first), so a native
    ``ID:0 PN:vg`` root of the source stays history even when the lift adds
    an identical line.

    Without it: an :func:`is_lift_pg` line is pending unless a fiberhmm-lift
    @PG descends from it through PP. A bare ``PN:vg`` line (no CL; vg
    giraffe writes the same) belongs to a pending lift only when one exists,
    has no PP (the repair always gives the lift's one), and is not the root
    of other history: of identical-ID bare lines the first is the lift's
    (vg writes its line at the top), and a single one some other line names
    as its PP is a native alignment's."""
    if source_pgs is not None:
        remaining = Counter(_pg_key(pg) for pg in source_pgs)
        new = set()
        for i in range(len(pgs) - 1, -1, -1):
            key = _pg_key(pgs[i])
            if remaining[key] > 0:
                remaining[key] -= 1
            else:
                new.add(i)
        return [i for i in sorted(new) if is_lift_pg(pgs[i]) or _is_vg(pgs[i])]
    bare = {i for i, pg in enumerate(pgs)
            if _is_vg(pg) and not is_lift_pg(pg) and not pg.get("PP")}
    by_id = {}
    for i, pg in enumerate(pgs):
        if i not in bare:
            by_id.setdefault(pg.get("ID"), i)
    repaired = set()
    for pg in pgs:
        if not is_repair_pg(pg):
            continue
        seen, parent = set(), pg.get("PP")
        while parent in by_id and by_id[parent] not in seen:
            j = by_id[parent]
            seen.add(j)
            parent = pgs[j].get("PP")
        repaired |= seen
    pending = {i for i, pg in enumerate(pgs) if is_lift_pg(pg) and i not in repaired}
    if not pending:
        return []
    referenced = {pg.get("PP") for i, pg in enumerate(pgs)
                  if i not in bare and not is_lift_pg(pg) and not is_repair_pg(pg)}
    by_bare_id = defaultdict(list)
    for i in sorted(bare):
        by_bare_id[pgs[i].get("ID")].append(i)
    for ident, indices in by_bare_id.items():
        if len(indices) > 1 or ident not in referenced:
            pending.add(indices[0])
    return sorted(pending)


def header_lift_state(header, source_pgs: Optional[List[dict]] = None) -> dict:
    """``{'lift': [IDs of unrepaired lift @PG lines], 'repaired': bool}``.

    ``repaired`` means a fiberhmm-lift @PG is present and no lift after it
    is left unrepaired."""
    hdr = header.to_dict() if hasattr(header, "to_dict") else dict(header or {})
    pgs = list(hdr.get("PG", []) or [])
    pending = pending_lift_indices(pgs, source_pgs)
    return {"lift": [str(pgs[i].get("ID", "?")) for i in pending],
            "repaired": not pending and any(is_repair_pg(pg) for pg in pgs)}


def _lift_order(pg: dict) -> int:
    """0 panSN prefix (and anything else), 1 vg or liftbam, 2 panSN strip."""
    tokens = str(pg.get("CL", "") or "").split()
    if "--strip" in tokens:
        return 2
    if _is_vg(pg) or _is_liftbam(pg) or (tokens and os.path.basename(tokens[0]) == "vg"):
        return 1
    return 0


def rechain_lift_history(header_dict: dict, source_pgs: Optional[List[dict]] = None) -> dict:
    """Link the unrepaired lift's @PG lines into one chain in the history.

    The lift's lines (panSN prefix, vg, panSN strip) become one chain whose
    parent is the tip of the history the lift ran on: the last @PG, in
    append order, that does not descend from the lift. Lines that ran on the
    lifted output (a ``samtools sort`` with ``PP:ft.2``, say) follow the
    chain, re-parented on its last step where they pointed into it. The
    result has no cycles. IDs that collide with other lines (vg's ``0`` on a
    second lift) get a suffix."""
    pgs = [dict(pg) for pg in header_dict.get("PG", []) or []]
    pending = set(pending_lift_indices(pgs, source_pgs))
    if not pending:
        return header_dict
    others = [i for i in range(len(pgs)) if i not in pending]
    other_ids = {pgs[i].get("ID") for i in others}
    # Original IDs that name only a lift line (vg's "0" may also name an old one).
    lift_ids = {pgs[i].get("ID") for i in pending} - other_ids
    by_id = {}
    for i in others:
        by_id.setdefault(pgs[i].get("ID"), i)

    def descends_from_lift(i):
        seen, parent = set(), pgs[i].get("PP")
        while parent is not None and parent not in seen:
            if parent in lift_ids:
                return True
            seen.add(parent)
            j = by_id.get(parent)
            if j is None:
                return False
            parent = pgs[j].get("PP")
        return False

    after = [i for i in others if descends_from_lift(i)]
    before = [i for i in others if i not in set(after)]
    used = set(other_ids)
    parent = pgs[before[-1]].get("ID") if before else None
    chained = []
    for i in sorted(pending, key=lambda k: (_lift_order(pgs[k]), k)):
        pg = pgs[i]
        base, ident, n = pg.get("ID") or "lift", pg.get("ID") or "lift", 1
        while ident in used:
            n += 1
            ident = f"{base}.{n}"
        used.add(ident)
        pg["ID"] = ident
        if parent:
            pg["PP"] = parent
        else:
            pg.pop("PP", None)
        chained.append(pg)
        parent = ident
    tail = []
    for i in after:
        pg = pgs[i]
        if pg.get("PP") in lift_ids:
            pg["PP"] = parent
        tail.append(pg)
    out = dict(header_dict)
    out["PG"] = [pgs[i] for i in before] + chained + tail
    return out


# ---------------------------------------------------------------------------
# Post-lift repair
# ---------------------------------------------------------------------------

@dataclass
class RepairStats:
    records_in: int = 0
    records_out: int = 0
    supplementary_restored: int = 0
    orphan_supplementary_dropped: int = 0
    strand_flipped: int = 0
    st_swapped: int = 0
    hard_clips_restored: int = 0
    unstashed: int = 0
    sa_rewritten: int = 0
    sa_dropped: int = 0
    calmd: bool = False
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


class RepairError(ValueError):
    """The lifted BAM cannot be repaired as given."""


_ST_SWAP = {"CT": "GA", "GA": "CT"}


def _readd_hard_clips(read, spec: str, flipped: bool) -> bool:
    """Re-add the stashed hard clips a lift dropped (vg does; liftbam keeps
    them, and a record that still has hard clips is left alone)."""
    kind, _, values = spec.partition(":")
    if kind != "H":
        return False
    if any(op == _HARD_CLIP for op, _n in (read.cigartuples or [])):
        return False
    try:
        left, right = (int(v) for v in values.split(","))
    except ValueError:
        return False
    if flipped:
        left, right = right, left
    cigar = list(read.cigartuples or [])
    if not cigar or not (left or right):
        return False
    if left:
        cigar.insert(0, (_HARD_CLIP, left))
    if right:
        cigar.append((_HARD_CLIP, right))
    read.cigartuples = cigar
    return True


MAPQ_SOURCE = "source"


def parse_mapq_mode(value: str) -> str:
    """``'source'`` or ``'fixed:N'`` (0 <= N <= 255), normalised."""
    text = str(value).strip().lower()
    if text == MAPQ_SOURCE:
        return MAPQ_SOURCE
    kind, _, number = text.partition(":")
    if kind == "fixed" and number.isdigit() and 0 <= int(number) <= 255:
        return f"fixed:{int(number)}"
    raise ValueError(f"MAPQ mode must be 'source' or 'fixed:N' (0-255), not {value!r}")


def lift_mapq_mode(header) -> Optional[str]:
    """The MAPQ mode the last fiberhmm-lift @PG records (``'source'`` or
    ``'fixed:N'``), or None when the header has no fiberhmm-lift @PG."""
    hdr = header.to_dict() if hasattr(header, "to_dict") else dict(header or {})
    mode = None
    for pg in hdr.get("PG", []) or []:
        if is_repair_pg(pg):
            match = re.search(r"(?:^|[\s;])mapq=(source|fixed:\d+)", str(pg.get("DS", "")))
            mode = match.group(1) if match else MAPQ_SOURCE
    return mode


def repair_record(read, stats: RepairStats, mapq: str = MAPQ_SOURCE) -> bool:
    """Repair one lifted record in place; False when it must be dropped.

    Orphan handling needs the whole file and is done by :func:`repair_bam`."""
    if not read.has_tag(TAG_FLAG):
        stats.unstashed += 1
        raise RepairError(
            f"record {read.query_name!r} has no {TAG_FLAG} stash tag: the BAM was lifted "
            "without fiberhmm-lift; give the source BAM (--source) to rebuild it")
    source_flag = int(read.get_tag(TAG_FLAG))
    lifted_flag = int(read.flag)
    # Strand and placement come from the lift; 0x800 from either (vg drops it,
    # liftbam sets it on the extra records of a split), and 0x100 from liftbam
    # on an alternative placement of the same read bases.
    flag = (source_flag & ~(_REVERSE | _UNMAPPED)) | (lifted_flag & (_REVERSE | _UNMAPPED))
    if not source_flag & _SECONDARY:
        if lifted_flag & _SECONDARY:
            flag = (flag & ~_SUPP) | _SECONDARY
        else:
            flag |= lifted_flag & _SUPP
    read.flag = flag
    flipped = (not read.is_unmapped) and bool(source_flag & _REVERSE) != read.is_reverse
    if flipped:
        stats.strand_flipped += 1
        if read.has_tag("st"):
            value = str(read.get_tag("st"))
            swapped = _ST_SWAP.get(value.upper())
            if swapped is not None:
                read.set_tag("st", swapped, value_type="Z")
                stats.st_swapped += 1
    if mapq != MAPQ_SOURCE:
        read.mapping_quality = int(mapq.split(":", 1)[1])
    elif read.has_tag(TAG_SOURCE_MAPQ):
        read.mapping_quality = int(read.get_tag(TAG_SOURCE_MAPQ))
    if read.has_tag(TAG_HARDCLIP):
        if _readd_hard_clips(read, str(read.get_tag(TAG_HARDCLIP)), flipped):
            stats.hard_clips_restored += 1
    for tag in STASH_TAGS + REFERENCE_TAGS + ("SA",):
        if read.has_tag(tag):
            read.set_tag(tag, None)
    if read.has_tag("cs") and _MINIMAP2_CS.match(str(read.get_tag("cs"))):
        read.set_tag("cs", None)
    return True


def validate_target_fasta(header, fasta_path: str, used: Optional[set] = None) -> None:
    """Every @SQ of the lifted header that carries records (``used`` names;
    all when None) must be in the FASTA, and every @SQ in the FASTA must
    have its length. (A surjected header lists every target path, alts and
    all, whether or not a read landed there.)"""
    with pysam.FastaFile(fasta_path) as fasta:
        lengths = dict(zip(fasta.references, fasta.lengths))
    hdr = header.to_dict() if hasattr(header, "to_dict") else dict(header)
    for sq in hdr.get("SQ", []):
        name, length = sq.get("SN"), int(sq.get("LN", 0))
        if name not in lengths and used is not None and name not in used:
            continue
        if name not in lengths:
            raise RepairError(f"--target-fasta has no contig {name!r} (lifted @SQ)")
        if lengths[name] != length:
            raise RepairError(f"--target-fasta contig {name!r} is {lengths[name]} bp; "
                              f"the lifted @SQ says {length}")


def _sa_entry(read) -> str:
    nm = read.get_tag("NM") if read.has_tag("NM") else 0
    return (f"{read.reference_name},{read.reference_start + 1},{'-' if read.is_reverse else '+'},"
            f"{read.cigarstring},{read.mapping_quality},{nm};")


def _rewrite_sa(in_path: str, out_path: str, multi: set, stats: RepairStats,
                threads: int) -> None:
    entries: Dict[str, list] = defaultdict(list)
    with pysam.AlignmentFile(in_path, "rb", check_sq=False, threads=threads) as bam:
        for read in bam.fetch(until_eof=True):
            if (read.query_name in multi and not read.is_unmapped
                    and not read.is_secondary):
                entries[read.query_name].append(
                    (read.is_supplementary, read.reference_id, read.reference_start,
                     read.is_reverse, _sa_entry(read)))
    for name in entries:
        entries[name].sort()
    with pysam.AlignmentFile(in_path, "rb", check_sq=False, threads=threads) as bam, \
            pysam.AlignmentFile(out_path, "wb", header=bam.header, threads=threads) as out:
        for read in bam.fetch(until_eof=True):
            group = entries.get(read.query_name)
            if group and not read.is_unmapped and not read.is_secondary:
                own = _sa_entry(read)
                others = [e[-1] for e in group if e[-1] != own]
                if others:
                    read.set_tag("SA", "".join(others), value_type="Z")
                    stats.sa_rewritten += 1
            out.write(read)


def lift_pg_record(*, repair: RepairStats, calmd: bool, mode: str = "lift",
                   source: Optional[str] = None, target: Optional[str] = None,
                   graph: Optional[str] = None, stash: Optional[dict] = None,
                   command: Optional[str] = None, mapq: str = MAPQ_SOURCE,
                   engine: Optional[str] = None) -> dict:
    """The ``@PG ID:fiberhmm-lift`` record (coord=molecular)."""
    import sys

    import fiberhmm
    from fiberhmm.io.annotation_frame import MOLECULAR, append_coord_to_ds

    def token(value):
        return re.sub(r"\s+", "_", str(value)) if value not in (None, "") else "unknown"

    parts = [
        f"graph lift repair; mode={mode}",
        f"source={token(source)} target={token(target)}",
        f"graph={token(os.path.basename(graph)) if graph else 'unknown'}",
        f"calmd={'yes' if calmd else 'no'}",
    ]
    if engine:
        parts.insert(1, f"engine={token(engine)}")
    if not calmd:
        parts.append("MD/NM/SA dropped (no --target-fasta)")
    stash = stash or {}
    parts.append(f"excluded_supplementary={stash.get('excluded_supplementary', 0)}")
    if "orphan_supplementary_no_primary_in_input" in stash:
        parts.append("orphan_supplementary_no_primary_in_input="
                     f"{stash['orphan_supplementary_no_primary_in_input']}")
    if stash.get("orphan_supplementary_kept"):
        parts.append(f"orphan_supplementary_kept={stash['orphan_supplementary_kept']}")
    parts.append(f"filled_supplementary={stash.get('filled_supplementary', 0)}")
    for key in ("excluded_unverifiable", "mods_dropped"):
        if stash.get(key):
            parts.append(f"{key}={stash[key]}")
    parts.append(f"dropped_secondary={stash.get('dropped_secondary', 0)}")
    parts.append(f"dropped_unmapped={stash.get('dropped_unmapped', 0)}")
    for key in ("unlifted", "unmatched", "ambiguous", "split_records"):
        if key in stash:
            parts.append(f"{key}={stash[key]}")
    parts.append(f"orphan_supplementary_dropped={repair.orphan_supplementary_dropped}")
    parts.append(f"st_swapped={repair.st_swapped}")
    parts.append(f"mapq={mapq} provenance_tags={TAG_SOURCE_MAPQ},{TAG_SOURCE_LOCUS}")
    return {
        "PN": LIFT_PROGRAM,
        "VN": getattr(fiberhmm, "__version__", "unknown"),
        "CL": command if command is not None else " ".join(sys.argv),
        "DS": append_coord_to_ds("; ".join(parts), MOLECULAR),
    }


def _finalize(path: str, out_path: str, *, index: bool) -> None:
    """Publish ``path`` (any file system) at ``out_path`` with its index.

    The BAM is moved next to ``out_path`` under a temporary name and indexed
    there, then published by :func:`fiberhmm.inference.bam_output.
    commit_output`: stale indexes (``.bai``/``.csi``) are removed, the BAM
    and its new index never pair with the other's previous generation, and a
    failure restores the previous output. A failure before publication
    leaves ``path`` or the staged copy, never a half-published output."""
    import shutil

    from fiberhmm.inference.bam_output import commit_output

    final_dir = os.path.dirname(os.path.abspath(out_path))
    staging = os.path.join(final_dir, f".{os.path.basename(out_path)}.{os.getpid()}.tmp.bam")
    shutil.move(path, staging)
    try:
        if index:
            pysam.index(staging, staging + ".bai")
        commit_output(staging, out_path)
    except BaseException:
        # Keep the finished BAM where the caller's work dir can be found.
        if os.path.exists(staging):
            shutil.move(staging, path)
        for suffix in (".bai", ".csi"):
            if os.path.exists(staging + suffix):
                os.remove(staging + suffix)
        raise


def repair_bam(lifted_path: str, out_path: str, *, work_dir: str,
               target_fasta: Optional[str] = None, threads: int = 1,
               pg_args: Optional[dict] = None, index: bool = True,
               source_pgs: Optional[List[dict]] = None,
               mapq: str = MAPQ_SOURCE) -> RepairStats:
    """Repair a lifted (stashed) BAM into a sorted, indexed ``out_path``.

    Passes over the lifted BAM: (1) count what the repair will do (for the
    @PG) and find the QNAMEs with a supplementary; (2) for those, which have
    a lifted primary; (3) repair, drop orphan supplementaries and write with
    the rechained header plus the fiberhmm-lift @PG. Then sort; with
    ``target_fasta`` recompute MD/NM (``samtools calmd``) and rebuild SA;
    index. ``pg_args`` are :func:`lift_pg_record` keywords; ``source_pgs``
    the @PG lines of the BAM that was lifted, which tell the lift's own lines
    apart exactly (else :func:`pending_lift_indices` infers them). The
    output is written next to ``out_path`` first and moved into place, so
    ``work_dir`` may be on another file system; a failure never removes a
    finished output."""
    planned = RepairStats()
    supp_counts: Dict[str, int] = defaultdict(int)
    used_contigs = set()
    with pysam.AlignmentFile(lifted_path, "rb", check_sq=False, threads=threads) as bam:
        header = bam.header
        if header_lift_state(header, source_pgs)["repaired"]:
            raise RepairError("the BAM was already repaired (its last lift has a "
                              "fiberhmm-lift @PG)")
        for read in bam.fetch(until_eof=True):
            planned.records_in += 1
            repair_record(read, planned, mapq)
            if not read.is_unmapped:
                used_contigs.add(read.reference_name)
            if read.is_supplementary:
                supp_counts[read.query_name] += 1
    primary_names = set()
    kept_orphans = set()
    if supp_counts:
        with pysam.AlignmentFile(lifted_path, "rb", check_sq=False, threads=threads) as bam:
            for read in bam.fetch(until_eof=True):
                if (read.query_name in supp_counts and read.has_tag(TAG_FLAG)
                        and not int(read.get_tag(TAG_FLAG)) & (_SUPP | _SECONDARY)
                        and not read.is_unmapped):
                    primary_names.add(read.query_name)
                if read.query_name in supp_counts and read.has_tag(TAG_ORPHAN):
                    kept_orphans.add(read.query_name)
    planned.orphan_supplementary_dropped = sum(
        n for name, n in supp_counts.items()
        if name not in primary_names and name not in kept_orphans)
    if target_fasta:
        validate_target_fasta(header, target_fasta, used_contigs)

    from fiberhmm.io.bam_header import append_pg_record

    out_header = append_pg_record(
        rechain_lift_history(header.to_dict(), source_pgs),
        lift_pg_record(repair=planned, calmd=bool(target_fasta), mapq=mapq, **(pg_args or {})))
    os.makedirs(work_dir, exist_ok=True)
    unsorted = os.path.join(work_dir, "repaired.unsorted.bam")
    stats = RepairStats()
    multi = set()
    with pysam.AlignmentFile(lifted_path, "rb", check_sq=False, threads=threads) as bam, \
            pysam.AlignmentFile(unsorted, "wb", header=out_header, threads=threads) as out:
        for read in bam.fetch(until_eof=True):
            stats.records_in += 1
            lifted_supplementary = read.is_supplementary
            input_orphan = read.has_tag(TAG_ORPHAN)
            repair_record(read, stats, mapq)
            if read.is_supplementary:
                if read.query_name not in primary_names and not input_orphan:
                    stats.orphan_supplementary_dropped += 1
                    continue
                multi.add(read.query_name)
                if not lifted_supplementary:
                    stats.supplementary_restored += 1    # counted only when written
            out.write(read)
            stats.records_out += 1
    stats.calmd = bool(target_fasta)

    sorted_path = os.path.join(work_dir, "repaired.sorted.bam")
    pysam.sort("--no-PG", "-@", str(max(1, threads)), "-o", sorted_path, unsorted)
    os.remove(unsorted)
    current = sorted_path
    if target_fasta:
        calmd_path = os.path.join(work_dir, "repaired.calmd.bam")
        pysam.calmd("-b", "--no-PG", "-@", str(max(1, threads)), current, target_fasta,
                    save_stdout=calmd_path)
        os.remove(current)
        current = calmd_path
        if multi:
            sa_path = os.path.join(work_dir, "repaired.sa.bam")
            _rewrite_sa(current, sa_path, multi, stats, threads)
            os.remove(current)
            current = sa_path
    else:
        stats.sa_dropped = len(multi)
    _finalize(current, out_path, index=index)
    return stats


# ---------------------------------------------------------------------------
# BED files: BED12 block by block, coordinate columns as sub-intervals
# ---------------------------------------------------------------------------
#
# The engines (liftbed, vg inject/surject of one read per row) lift each row's start/end
# only; BED12 blocks, thickStart/thickEnd and narrowPeak summits would stay
# in source offsets. explode_bed12 writes one BED6 row per block (name
# "<row>|<block>") plus a thick interval when it lies inside the row
# ("<row>|t"); explode_bed_rows writes each plain row ("<row>|m") with its
# thick interval and a 1 bp summit ("<row>|s"). The engine lifts those, and
# reassemble_bed12 / reassemble_bed_rows rebuild the rows.

_BED_HEADER_PREFIXES = ("#", "track", "browser")
_FLIP_STRAND = {"+": "-", "-": "+"}
BED12_PARTIAL = ("drop", "keep")


@dataclass
class BedStats:
    rows_in: int = 0
    rows_out: int = 0
    blocks_in: int = 0
    rows_lost_blocks: int = 0
    rows_split: int = 0
    rows_rearranged: int = 0
    rows_flipped: int = 0
    rows_partial: int = 0
    blocks_dropped: int = 0
    rows_unlifted: int = 0
    thick_lifted: int = 0
    thick_lost: int = 0
    summits_lifted: int = 0
    summits_lost: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def _int_field(fields: list, col: int) -> Optional[int]:
    try:
        return int(fields[col])
    except (IndexError, ValueError):
        return None


def thick_inside(fields: list) -> bool:
    """Whether a row's thickStart/thickEnd (columns 7-8) form a non-empty interval that
    lies inside the row without spanning it, i.e. one that has to be lifted on its own."""
    start, end = int(fields[1]), int(fields[2])
    ts, te = _int_field(fields, 6), _int_field(fields, 7)
    return ts is not None and te is not None and start <= ts < te <= end and (ts > start or te < end)


def _lifted_thick(fields: list, start: int, end: int, chrom: str, unit, stats: "BedStats"):
    """``(thickStart, thickEnd)`` of a row lifted to ``chrom:[start, end)``. An empty
    thick interval stays empty (at the new start), one spanning the row spans the lifted
    row, one inside it takes its own lifted interval ``unit`` clamped to the row (empty
    at the new start when it did not lift or landed elsewhere)."""
    ts, te = int(fields[6]), int(fields[7])
    s0, e0 = int(fields[1]), int(fields[2])
    if te <= ts:
        return start, start
    if ts <= s0 and te >= e0:
        return start, end
    if unit is not None and unit[0] == chrom:
        a, b = max(start, unit[1]), min(end, unit[2])
        if a < b:
            stats.thick_lifted += 1
            return a, b
    stats.thick_lost += 1
    return start, start


def explode_bed12(lines: Iterable[str], unstranded_as: Optional[str] = None
                  ) -> Tuple[List[str], List[list], List[str]]:
    """``(bed6_rows, rows, header_lines)`` of a BED12(+N) file.

    ``unstranded_as`` (``"+"``) gives the blocks of rows without a ``+``/``-``
    strand that strand, so an engine that swaps the strand column on a ``-``
    chain (liftbed) reports their flips too (see ``reassemble_bed12``). A
    thickStart/thickEnd inside the row (not spanning it, not empty) is written
    as one more interval, ``<row>|t``."""
    bed6, rows, header = [], [], []
    for line in lines:
        line = line.rstrip("\n")
        if not line:
            continue
        if line.startswith(_BED_HEADER_PREFIXES):
            header.append(line)
            continue
        fields = line.split("\t")
        if len(fields) < 12:
            raise ValueError(f"not a BED12 row ({len(fields)} columns): {line[:80]}")
        index = len(rows)
        rows.append(fields)
        try:
            start = int(fields[1])
            sizes = [int(v) for v in fields[10].rstrip(",").split(",") if v != ""]
            offsets = [int(v) for v in fields[11].rstrip(",").split(",") if v != ""]
            count = int(fields[9])
            int(fields[6]), int(fields[7])
        except ValueError:
            raise ValueError(f"BED12 row {index + 1}: a coordinate column is not an "
                             "integer") from None
        if len(sizes) != count or len(offsets) != len(sizes):
            raise ValueError(f"BED12 row {index + 1}: blockCount does not match its blocks")
        strand = fields[5]
        if unstranded_as and strand not in _FLIP_STRAND:
            strand = unstranded_as
        for block, (size, offset) in enumerate(zip(sizes, offsets)):
            bed6.append((fields[0], start + offset, start + offset + size,
                         f"{index}|{block}", fields[4], strand))
        if thick_inside(fields):
            bed6.append((fields[0], int(fields[6]), int(fields[7]), f"{index}|t", fields[4],
                         strand))
    bed6.sort(key=lambda r: (r[0], r[1], r[2]))
    return ["\t".join(str(v) for v in r) for r in bed6], rows, header


def _parse_lifted_units(lifted: Iterable[str]):
    """``{row: {unit: (chrom, start, end, strand)}}`` of lifted ``<row>|<unit>`` rows
    (a unit seen twice is kept as a list under ``(unit, 'dup')``)."""
    units: Dict[int, dict] = defaultdict(dict)
    for line in lifted:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 4 or "|" not in fields[3]:
            continue
        row, _, unit = fields[3].rpartition("|")
        try:
            value = (fields[0], int(fields[1]), int(fields[2]),
                     fields[5] if len(fields) > 5 else ".")
            row = int(row)
        except ValueError:
            continue
        if unit in units[row]:
            units[row].setdefault((unit, "dup"), True)
        units[row][unit] = value
    return units


def reassemble_bed12(lifted: Iterable[str], rows: List[list],
                     dropped: Optional[list] = None, *,
                     explicit_strand: bool = False, partial: str = "drop",
                     dropped_blocks: Optional[list] = None) -> Tuple[List[str], BedStats]:
    """BED12 rows rebuilt from lifted BED6 blocks (named ``<row>|<block>``).

    A row is dropped (counted) when a block did not lift, its blocks land on
    different contigs or overlap, or their order is neither kept nor
    reversed. With ``partial="keep"`` a row whose blocks did not all lift is
    rebuilt from the blocks that did (blockCount, sizes, starts and the row
    bounds recomputed, per-block columns subset) when at least one lifted;
    ``dropped_blocks`` receives ``(row index, block index)`` of the blocks it
    left out. The segment changed strand when the lifted blocks carry the
    opposite strand or come back in reversed order (ft's BED mode copies the
    strand column): the row takes the opposite strand and per-block
    comma-list columns (13+) are reversed with the blocks. Blocks with
    disagreeing strands, or a strand change whose blocks kept their order,
    are dropped as rearranged. thickStart/thickEnd span the lifted row when
    they spanned the source row, stay empty when empty, and otherwise take
    their own lifted interval (``<row>|t``) clamped to the row.
    ``dropped``, when given, receives ``(row index, reason)`` for every dropped
    row (``lost_block``, ``split_contigs``, ``rearranged``). ft's BED mode
    copies the strand column, so a flip is inferred from reversed block
    order. ``explicit_strand`` (liftbed, which swaps the strand column on a
    ``-`` chain; explode with ``unstranded_as="+"``) takes the flip from the
    lifted strand alone and drops a multi-block row whose block order does
    not agree with it as rearranged."""
    if partial not in BED12_PARTIAL:
        raise ValueError(f"partial must be one of {BED12_PARTIAL}")

    def drop(index, reason):
        if dropped is not None:
            dropped.append((index, reason))

    stats = BedStats(rows_in=len(rows))
    units = _parse_lifted_units(lifted)
    out = []
    for index, fields in enumerate(rows):
        count = int(fields[9])
        stats.blocks_in += count
        row_units = units.get(index, {})
        lifted_blocks, bad = [], False
        for key, value in row_units.items():
            if isinstance(key, tuple):
                bad |= key[0].isdigit()
            elif key.isdigit():
                block = int(key)
                bad |= block >= count
                lifted_blocks.append((block,) + value)
        if bad or not lifted_blocks or (len(lifted_blocks) != count and partial != "keep"):
            stats.rows_lost_blocks += 1
            drop(index, "lost_block")
            continue
        n = len(lifted_blocks)
        if len({b[1] for b in lifted_blocks}) != 1:
            stats.rows_split += 1
            drop(index, "split_contigs")
            continue
        by_position = sorted(lifted_blocks, key=lambda b: (b[2], b[3]))
        order = [b[0] for b in by_position]
        strands = {b[4] for b in by_position}
        original = fields[5]
        if len(strands) != 1:
            stats.rows_rearranged += 1
            drop(index, "rearranged")
            continue
        block_strand = original if original in _FLIP_STRAND else ("+" if explicit_strand else original)
        strand_flipped = block_strand in _FLIP_STRAND and strands == {_FLIP_STRAND[block_strand]}
        if explicit_strand and block_strand in _FLIP_STRAND and strands != {block_strand} and not strand_flipped:
            stats.rows_rearranged += 1
            drop(index, "rearranged")
            continue
        if order == sorted(order):
            reversed_order = False
        elif order == sorted(order, reverse=True):
            reversed_order = True
        else:
            stats.rows_rearranged += 1
            drop(index, "rearranged")
            continue
        if n > 1 and (reversed_order != strand_flipped if explicit_strand
                      else strand_flipped and not reversed_order):
            stats.rows_rearranged += 1
            drop(index, "rearranged")
            continue
        flipped = strand_flipped if explicit_strand else strand_flipped or reversed_order
        if any(by_position[i][3] > by_position[i + 1][2] for i in range(n - 1)):
            stats.rows_rearranged += 1
            drop(index, "rearranged")
            continue
        chrom = by_position[0][1]
        start, end = by_position[0][2], by_position[-1][3]
        new = list(fields)
        new[0], new[1], new[2] = chrom, str(start), str(end)
        ts, te = _lifted_thick(fields, start, end, chrom, row_units.get("t"), stats)
        new[6], new[7] = str(ts), str(te)
        new[9] = str(n)
        new[10] = ",".join(str(b[3] - b[2]) for b in by_position)
        new[11] = ",".join(str(b[2] - start) for b in by_position)
        kept = sorted(order)
        if n != count:
            stats.rows_partial += 1
            stats.blocks_dropped += count - n
            if dropped_blocks is not None:
                dropped_blocks.extend((index, k) for k in sorted(set(range(count)) - set(kept)))
        for col in range(12, len(new)):
            values = new[col].split(",")
            if count > 1 and len(values) == count:
                values = [values[k] for k in kept]
                if flipped and n > 1:
                    values.reverse()
                new[col] = ",".join(values)
        if flipped:
            stats.rows_flipped += 1
            new[5] = _FLIP_STRAND.get(original, original)
        out.append("\t".join(new))
        stats.rows_out += 1
    return out, stats


def explode_bed_rows(lines: Iterable[str], *, thick: bool = False, summit: bool = False,
                     unstranded_as: Optional[str] = None
                     ) -> Tuple[List[str], List[list], List[str]]:
    """``(bed6_rows, rows, header_lines)`` of a BED3+ file whose coordinate columns are
    lifted with the row: each row as ``<row>|m``; with ``thick``, a thickStart/thickEnd
    (columns 7-8) inside the row as ``<row>|t``; with ``summit`` (narrowPeak), the 1 bp
    summit (column 10, an offset from the start; -1 = none) as ``<row>|s``."""
    bed6, rows, header = [], [], []
    for line in lines:
        line = line.rstrip("\n")
        if not line:
            continue
        if line.startswith(_BED_HEADER_PREFIXES):
            header.append(line)
            continue
        fields = line.split("\t")
        index = len(rows) + 1
        if len(fields) < 3:
            raise ValueError(f"BED row {index}: fewer than 3 columns: {line[:80]}")
        try:
            start, end = int(fields[1]), int(fields[2])
        except ValueError:
            raise ValueError(f"BED row {index}: start/end are not integers: {line[:80]}") from None
        if thick and (len(fields) < 8 or _int_field(fields, 6) is None
                      or _int_field(fields, 7) is None):
            raise ValueError(f"BED row {index}: thickStart/thickEnd (columns 7-8) are not "
                             f"integers: {line[:80]}")
        if summit and (len(fields) < 10 or _int_field(fields, 9) is None):
            raise ValueError(f"narrowPeak row {index}: the summit (column 10) is not an "
                             f"integer: {line[:80]}")
        index = len(rows)
        rows.append(fields)
        strand = fields[5] if len(fields) > 5 else "."
        if unstranded_as and strand not in _FLIP_STRAND:
            strand = unstranded_as
        score = fields[4] if len(fields) > 4 else "0"
        bed6.append((fields[0], start, end, f"{index}|m", score, strand))
        if thick and thick_inside(fields):
            bed6.append((fields[0], int(fields[6]), int(fields[7]), f"{index}|t", score, strand))
        if summit:
            offset = int(fields[9])
            if 0 <= offset < end - start:
                bed6.append((fields[0], start + offset, start + offset + 1, f"{index}|s", score,
                             strand))
    bed6.sort(key=lambda r: (r[0], r[1], r[2]))
    return ["\t".join(str(v) for v in r) for r in bed6], rows, header


def reassemble_bed_rows(lifted: Iterable[str], rows: List[list], *, thick: bool = False,
                        summit: bool = False, dropped: Optional[list] = None
                        ) -> Tuple[List[str], BedStats]:
    """Rows of ``explode_bed_rows`` rebuilt from their lifted units: chrom, start and end
    from ``<row>|m`` (a ``+``/``-`` strand column from it too: liftbed swaps it on a
    ``-`` chain, ft copies it), thickStart/thickEnd as in ``reassemble_bed12``, the
    summit as its lifted base's offset from the new start (-1 when it did not lift or
    landed outside the row; -1 stays -1). Other columns are copied. ``dropped`` receives
    ``(row index, 'unlifted')`` for rows whose interval did not lift."""
    stats = BedStats(rows_in=len(rows))
    units = _parse_lifted_units(lifted)
    out = []
    for index, fields in enumerate(rows):
        row_units = units.get(index, {})
        main = row_units.get("m")
        if main is None:
            stats.rows_unlifted += 1
            if dropped is not None:
                dropped.append((index, "unlifted"))
            continue
        chrom, start, end, strand = main
        new = list(fields)
        new[0], new[1], new[2] = chrom, str(start), str(end)
        if len(new) > 5 and new[5] in _FLIP_STRAND and strand in _FLIP_STRAND and strand != new[5]:
            new[5] = strand
            stats.rows_flipped += 1
        if thick:
            ts, te = _lifted_thick(fields, start, end, chrom, row_units.get("t"), stats)
            new[6], new[7] = str(ts), str(te)
        if summit and int(fields[9]) >= 0:
            point = row_units.get("s")
            if point is not None and point[0] == chrom and start <= point[1] < end:
                new[9] = str(point[1] - start)
                stats.summits_lifted += 1
            else:
                new[9] = "-1"
                stats.summits_lost += 1
        out.append("\t".join(new))
        stats.rows_out += 1
    return out, stats
