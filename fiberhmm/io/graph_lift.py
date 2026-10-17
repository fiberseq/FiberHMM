"""Lift FiberHMM BAMs through a pangenome graph and repair what the lift breaks.

``ft pg-lift`` (fibertools-rs) moves reads from one assembly to another
through a GBZ graph: ``ft pg-pansn`` adds the panSN prefix, ``vg inject``
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
For a BAM lifted without the stash (``ft pg-lift`` run directly),
:func:`stash_from_source` rebuilds the same tags by matching each lifted
record with its source record, so one repair serves both routes.

Aux tags (SAM user space ``Y?``; none is written by FiberHMM, fibertools-rs,
minimap2, pbmm2 or vg):

``Yf:i`` (stash; removed by the repair)
    source FLAG.
``Yh:Z`` (stash; removed by the repair)
    hard clips of the source record as ``<kind>:<left>,<right>`` in its CIGAR
    order. ``S`` means the stash filled the clipped bases from the primary
    and turned them into soft clips (nothing to do after the lift); ``H``
    means they are still missing and the repair re-adds the hard clips.
``Yq:i`` (provenance; kept)
    source MAPQ. The repaired record keeps this value as its MAPQ too.
``Yp:Z`` (provenance; kept)
    source locus ``<contig>:<1-based POS>:<strand>``; marks a lifted record.

Decisions:

- MAPQ stays the source value: how uniquely the read placed on the source
  assembly is what the lift inherits. 255 would pass every ``min_mapq``
  filter, 0 would fail them all.
- Hard-clipped supplementaries get their clipped bases (and QUAL) from the
  primary of the same QNAME before the lift, as soft clips, so the record
  carries the whole molecule; MM/ML/MN are replaced by the primary's when
  they differ (they describe the original read, whatever the strand). A
  hard-clipped supplementary whose primary is not in the input is excluded
  and counted.
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
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pysam

TAG_FLAG = "Yf"
TAG_HARDCLIP = "Yh"
TAG_SOURCE_MAPQ = "Yq"
TAG_SOURCE_LOCUS = "Yp"
STASH_TAGS = (TAG_FLAG, TAG_HARDCLIP)
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
    hard_clipped_kept: int = 0
    frame_converted: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


class FrameError(ValueError):
    """The footprint annotations' frame cannot be decided."""


def check_input_frame(header, sample_reads: Iterable = ()) -> None:
    """Refuse a BAM whose legacy ns/nl/as/al frame is undecidable.

    :func:`~fiberhmm.io.frame_carry.carry_to_molecular` would silently drop
    those tags from reverse reads; lifting is not the place to lose them."""
    from fiberhmm.io.annotation_frame import legacy_tag_frame

    frame, reason = legacy_tag_frame(header)
    if frame is not None:
        return
    if any(read.has_tag("ns") or read.has_tag("as") for read in sample_reads):
        raise FrameError(
            f"cannot lift: the frame of the ns/nl/as/al tags is undecidable ({reason}), "
            "and a strand change would leave them mirrored. Split the BAM by input, or "
            "re-call it with FiberHMM >= 3.0, then lift.")


def _primary_info(read) -> dict:
    seq = read.query_sequence or ""
    quals = read.query_qualities
    mol_seq = revcomp(seq) if read.is_reverse else seq
    mol_qual = None
    if quals is not None:
        mol_qual = list(quals)[::-1] if read.is_reverse else list(quals)
    mods = {}
    for tag in ("MM", "ML", "Mm", "Ml", "MN"):
        if read.has_tag(tag):
            mods[tag] = read.get_tag(tag, with_value_type=True)
    return {"seq": mol_seq, "qual": mol_qual, "mods": mods}


def fill_hard_clips(read, primary: dict) -> Optional[str]:
    """Fill a hard-clipped record's clipped bases from its primary, as soft clips.

    ``primary`` is :func:`_primary_info` of the primary record (molecular
    orientation). The record keeps its own aligned SEQ (DAF encoding is
    per-record); only the clipped flanks come from the primary. Returns None
    on success, else why it could not be filled (the record is unchanged)."""
    left, right = _hard_clips(read)
    if not (left or right):
        return None
    seq = read.query_sequence or ""
    total = left + len(seq) + right
    mol = primary["seq"]
    if len(mol) != total:
        return f"primary length {len(mol)} != {total}"
    full = revcomp(mol) if read.is_reverse else mol
    if not _bases_agree(full[left:left + len(seq)], seq):
        return "primary SEQ does not match the supplementary"
    quals = read.query_qualities
    new_qual = None
    if quals is not None and primary["qual"] is not None:
        mq = primary["qual"][::-1] if read.is_reverse else primary["qual"]
        new_qual = array.array("B", list(mq[:left]) + list(quals) + list(mq[left + len(seq):]))
    cigar = list(read.cigartuples)
    if left:
        cigar[0] = (_SOFT_CLIP, left)
    if right:
        cigar[-1] = (_SOFT_CLIP, right)
    # Setting SEQ clears QUAL; QUAL is set again only when the primary has one.
    read.query_sequence = full[:left] + seq + full[left + len(seq):]
    read.cigartuples = cigar
    if new_qual is not None:
        read.query_qualities = new_qual
    mods = primary["mods"]
    own = {tag: read.get_tag(tag, with_value_type=True)
           for tag in ("MM", "ML", "Mm", "Ml", "MN") if read.has_tag(tag)}
    if not mods and "MN" in own and own["MN"][0] != len(read.query_sequence):
        # MM/ML were trimmed with the hard clip and nothing replaces them.
        for tag in own:
            read.set_tag(tag, None)
    elif mods and own != mods:
        for tag in own:
            read.set_tag(tag, None)
        for tag, (value, vtype) in mods.items():
            read.set_tag(tag, value, value_type=None if str(vtype).startswith("B") else vtype)
    return None


def _stash_tags(read, hard: Optional[str]) -> None:
    read.set_tag(TAG_FLAG, int(read.flag), value_type="i")
    read.set_tag(TAG_SOURCE_MAPQ, int(read.mapping_quality), value_type="i")
    read.set_tag(TAG_SOURCE_LOCUS, _source_locus(read), value_type="Z")
    if hard:
        read.set_tag(TAG_HARDCLIP, hard, value_type="Z")
    elif read.has_tag(TAG_HARDCLIP):
        read.set_tag(TAG_HARDCLIP, None)


def _to_molecular(read, stats: StashStats) -> None:
    from fiberhmm.io.frame_carry import carry_to_molecular

    if not read.is_reverse:
        return
    before = (read.get_tag("MA") if read.has_tag("MA") else None,
              tuple(read.get_tag("ns")) if read.has_tag("ns") else None,
              tuple(read.get_tag("as")) if read.has_tag("as") else None)
    carry_to_molecular(read)
    after = (read.get_tag("MA") if read.has_tag("MA") else None,
             tuple(read.get_tag("ns")) if read.has_tag("ns") else None,
             tuple(read.get_tag("as")) if read.has_tag("as") else None)
    if before != after:
        stats.frame_converted += 1


def stash_bam(in_path: str, out_path: str, *, threads: int = 1,
              frame_sample: int = 5000) -> StashStats:
    """Write ``in_path`` ready for lifting: stash tags, molecular frame,
    hard-clipped supplementaries filled from their primary.

    Secondary and unmapped records are dropped (counted). Two passes: the
    first finds the QNAMEs with hard-clipped supplementaries; the second
    writes, holding a hard-clipped supplementary back until its primary has
    been seen."""
    stats = StashStats()
    need = set()
    sample = []
    with pysam.AlignmentFile(in_path, "rb", check_sq=False, threads=threads) as bam:
        for read in bam.fetch(until_eof=True):
            if len(sample) < frame_sample:
                sample.append(read)
            if read.is_supplementary and not read.is_unmapped and any(_hard_clips(read)):
                need.add(read.query_name)
        check_input_frame(bam.header, sample)
    del sample

    primaries: Dict[str, dict] = {}
    waiting: Dict[str, list] = defaultdict(list)
    with pysam.AlignmentFile(in_path, "rb", check_sq=False, threads=threads) as bam, \
            pysam.AlignmentFile(out_path, "wb", header=bam.header, threads=threads) as out:

        def emit_supplementary(read, clips, primary):
            if fill_hard_clips(read, primary) is not None:
                stats.excluded_supplementary += 1
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
            _to_molecular(read, stats)
            name = read.query_name
            clips = _hard_clips(read)
            if read.is_supplementary and any(clips):
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


@dataclass
class SourceMatchStats:
    matched: int = 0
    unmatched: int = 0
    already_stashed: int = 0


def stash_from_source(lifted_path: str, source_path: str, out_path: str, *,
                      threads: int = 1) -> SourceMatchStats:
    """Add the stash tags to a BAM lifted without :func:`stash_bam`.

    Each lifted record is matched with the source record of the same QNAME
    and SEQ length (the lift copies SEQ verbatim) whose aligned part of the
    molecule overlaps it most. SEQ-frame footprint annotations are carried
    into molecular frame using the source orientation. Unmatched records are
    dropped (counted). Holds one summary per source record in memory."""
    from fiberhmm.io.frame_carry import carry_to_molecular

    stats = SourceMatchStats()
    source: Dict[str, list] = defaultdict(list)
    with pysam.AlignmentFile(source_path, "rb", check_sq=False, threads=threads) as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped or read.is_secondary:
                continue
            length = read.query_length
            source[read.query_name].append((
                length, _molecular_span(read, length), int(read.flag),
                int(read.mapping_quality), _source_locus(read), _hard_clips(read)))
    with pysam.AlignmentFile(lifted_path, "rb", check_sq=False, threads=threads) as bam, \
            pysam.AlignmentFile(out_path, "wb", header=bam.header, threads=threads) as out:
        for read in bam.fetch(until_eof=True):
            if read.has_tag(TAG_FLAG):
                stats.already_stashed += 1
                out.write(read)
                continue
            length = read.query_length
            span = _molecular_span(read, length) if not read.is_unmapped else (0, 0)
            best, best_overlap = None, -1
            for entry in source.get(read.query_name, ()):
                if entry[0] != length:
                    continue
                overlap = min(span[1], entry[1][1]) - max(span[0], entry[1][0])
                if overlap > best_overlap:
                    best, best_overlap = entry, overlap
            if best is None:
                stats.unmatched += 1
                continue
            _length, _span, flag, mapq, locus, clips = best
            lifted_flag = read.flag
            # SEQ-frame annotations were written against the source orientation.
            read.flag = (lifted_flag & ~_REVERSE) | (flag & _REVERSE)
            carry_to_molecular(read)
            read.flag = lifted_flag
            read.set_tag(TAG_FLAG, flag, value_type="i")
            read.set_tag(TAG_SOURCE_MAPQ, mapq, value_type="i")
            read.set_tag(TAG_SOURCE_LOCUS, locus, value_type="Z")
            if any(clips):
                read.set_tag(TAG_HARDCLIP, "H:{},{}".format(*clips), value_type="Z")
            out.write(read)
            stats.matched += 1
    return stats


# ---------------------------------------------------------------------------
# @PG history
# ---------------------------------------------------------------------------

def is_lift_pg(pg: dict) -> bool:
    """A @PG line written by a graph lift: vg, or ft pg-pansn / pg-lift."""
    pn = str(pg.get("PN", "") or "").strip().lower()
    ident = str(pg.get("ID", "") or "").strip().lower()
    cl = str(pg.get("CL", "") or "")
    if pn == "vg" or re.match(r"^vg(?:[ .-]|$)", ident) or re.match(r"^\S*\bvg\s+(surject|inject)\b", cl):
        return True
    tokens = cl.split()
    if tokens and os.path.basename(tokens[0]) in ("ft", "fibertools", "fibertools-rs"):
        return any(t in ("pg-pansn", "pg-lift", "pg-inject") for t in tokens[1:3])
    return False


def is_repair_pg(pg: dict) -> bool:
    pn = str(pg.get("PN", "") or "").strip().lower()
    ident = str(pg.get("ID", "") or "").strip().lower()
    return pn == LIFT_PROGRAM or re.match(rf"^{LIFT_PROGRAM}(?:\.\d+)?$", ident) is not None


def header_lift_state(header) -> dict:
    """``{'lift': [@PG IDs of the lift], 'repaired': bool}`` of a header."""
    hdr = header.to_dict() if hasattr(header, "to_dict") else dict(header or {})
    pgs = hdr.get("PG", []) or []
    return {"lift": [str(pg.get("ID", "?")) for pg in pgs if is_lift_pg(pg)],
            "repaired": any(is_repair_pg(pg) for pg in pgs)}


def _lift_order(pg: dict) -> int:
    """0 panSN prefix (and anything else), 1 vg, 2 panSN strip."""
    tokens = str(pg.get("CL", "") or "").split()
    if "--strip" in tokens:
        return 2
    if str(pg.get("PN", "")).lower() == "vg" or (tokens and os.path.basename(tokens[0]) == "vg"):
        return 1
    return 0


def rechain_lift_history(header_dict: dict) -> dict:
    """Link the lift's @PG lines into one chain after the source history.

    The source chain's tip (the last non-lift @PG, append order) becomes the
    parent of the panSN-prefix step, then vg, then the panSN strip, each
    line moved after the source lines so the strip step is last. Every lift
    line left dangling would count as its own @PG chain."""
    pgs = list(header_dict.get("PG", []) or [])
    lift = [pg for pg in pgs if is_lift_pg(pg)]
    if not lift:
        return header_dict
    rest = [pg for pg in pgs if not is_lift_pg(pg)]
    lift = sorted(lift, key=_lift_order)   # stable: keeps their own order within a step
    parent = rest[-1].get("ID") if rest else None
    chained = []
    for pg in lift:
        pg = dict(pg)
        if parent:
            pg["PP"] = parent
        else:
            pg.pop("PP", None)
        chained.append(pg)
        parent = pg.get("ID")
    out = dict(header_dict)
    out["PG"] = rest + chained
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
    kind, _, values = spec.partition(":")
    if kind != "H":
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


def repair_record(read, stats: RepairStats) -> bool:
    """Repair one lifted record in place; False when it must be dropped.

    Orphan handling needs the whole file and is done by :func:`repair_bam`."""
    if not read.has_tag(TAG_FLAG):
        stats.unstashed += 1
        raise RepairError(
            f"record {read.query_name!r} has no {TAG_FLAG} stash tag: the BAM was lifted "
            "without fiberhmm-lift; give the source BAM (--source) to rebuild it")
    source_flag = int(read.get_tag(TAG_FLAG))
    lifted_flag = int(read.flag)
    read.flag = (source_flag & ~(_REVERSE | _UNMAPPED)) | (lifted_flag & (_REVERSE | _UNMAPPED))
    if source_flag & _SUPP and not lifted_flag & _SUPP:
        stats.supplementary_restored += 1
    flipped = (not read.is_unmapped) and bool(source_flag & _REVERSE) != read.is_reverse
    if flipped:
        stats.strand_flipped += 1
        if read.has_tag("st"):
            value = str(read.get_tag("st"))
            swapped = _ST_SWAP.get(value.upper())
            if swapped is not None:
                read.set_tag("st", swapped, value_type="Z")
                stats.st_swapped += 1
    if read.has_tag(TAG_SOURCE_MAPQ):
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


def validate_target_fasta(header, fasta_path: str) -> None:
    """Every @SQ of the lifted header must be in the FASTA with its length."""
    with pysam.FastaFile(fasta_path) as fasta:
        lengths = dict(zip(fasta.references, fasta.lengths))
    hdr = header.to_dict() if hasattr(header, "to_dict") else dict(header)
    for sq in hdr.get("SQ", []):
        name, length = sq.get("SN"), int(sq.get("LN", 0))
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
                   command: Optional[str] = None) -> dict:
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
    if not calmd:
        parts.append("MD/NM/SA dropped (no --target-fasta)")
    stash = stash or {}
    parts.append(f"excluded_supplementary={stash.get('excluded_supplementary', 0)}")
    parts.append(f"filled_supplementary={stash.get('filled_supplementary', 0)}")
    parts.append(f"dropped_secondary={stash.get('dropped_secondary', 0)}")
    parts.append(f"dropped_unmapped={stash.get('dropped_unmapped', 0)}")
    for key in ("unlifted", "unmatched"):
        if key in stash:
            parts.append(f"{key}={stash[key]}")
    parts.append(f"orphan_supplementary_dropped={repair.orphan_supplementary_dropped}")
    parts.append(f"st_swapped={repair.st_swapped}")
    parts.append(f"mapq=source provenance_tags={TAG_SOURCE_MAPQ},{TAG_SOURCE_LOCUS}")
    return {
        "PN": LIFT_PROGRAM,
        "VN": getattr(fiberhmm, "__version__", "unknown"),
        "CL": command if command is not None else " ".join(sys.argv),
        "DS": append_coord_to_ds("; ".join(parts), MOLECULAR),
    }


def repair_bam(lifted_path: str, out_path: str, *, work_dir: str,
               target_fasta: Optional[str] = None, threads: int = 1,
               pg_args: Optional[dict] = None, index: bool = True) -> RepairStats:
    """Repair a lifted (stashed) BAM into a sorted, indexed ``out_path``.

    Passes over the lifted BAM: (1) count what the repair will do (for the
    @PG) and find the QNAMEs with a supplementary; (2) for those, which have
    a lifted primary; (3) repair, drop orphan supplementaries and write with
    the rechained header plus the fiberhmm-lift @PG. Then sort; with
    ``target_fasta`` recompute MD/NM (``samtools calmd``) and rebuild SA;
    index. ``pg_args`` are :func:`lift_pg_record` keywords."""
    planned = RepairStats()
    supp_counts: Dict[str, int] = defaultdict(int)
    with pysam.AlignmentFile(lifted_path, "rb", check_sq=False, threads=threads) as bam:
        header = bam.header
        if header_lift_state(header)["repaired"]:
            raise RepairError("the BAM already has a fiberhmm-lift @PG (it was repaired)")
        for read in bam.fetch(until_eof=True):
            planned.records_in += 1
            repair_record(read, planned)
            if read.is_supplementary:
                supp_counts[read.query_name] += 1
    primary_names = set()
    if supp_counts:
        with pysam.AlignmentFile(lifted_path, "rb", check_sq=False, threads=threads) as bam:
            for read in bam.fetch(until_eof=True):
                if (read.query_name in supp_counts and read.has_tag(TAG_FLAG)
                        and not int(read.get_tag(TAG_FLAG)) & (_SUPP | _SECONDARY)
                        and not read.is_unmapped):
                    primary_names.add(read.query_name)
    planned.orphan_supplementary_dropped = sum(
        n for name, n in supp_counts.items() if name not in primary_names)
    if target_fasta:
        validate_target_fasta(header, target_fasta)

    from fiberhmm.io.bam_header import append_pg_record

    out_header = append_pg_record(
        rechain_lift_history(header.to_dict()),
        lift_pg_record(repair=planned, calmd=bool(target_fasta), **(pg_args or {})))
    os.makedirs(work_dir, exist_ok=True)
    unsorted = os.path.join(work_dir, "repaired.unsorted.bam")
    stats = RepairStats()
    multi = set()
    with pysam.AlignmentFile(lifted_path, "rb", check_sq=False, threads=threads) as bam, \
            pysam.AlignmentFile(unsorted, "wb", header=out_header, threads=threads) as out:
        for read in bam.fetch(until_eof=True):
            stats.records_in += 1
            repair_record(read, stats)
            if read.is_supplementary:
                if read.query_name not in primary_names:
                    stats.orphan_supplementary_dropped += 1
                    continue
                multi.add(read.query_name)
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
    os.replace(current, out_path)
    if index:
        pysam.index(out_path)
    return stats


# ---------------------------------------------------------------------------
# BED12 through ft's BED mode
# ---------------------------------------------------------------------------
#
# ft pg-lift's BED mode lifts each row's start/end only; BED12 thickStart/
# thickEnd and blocks stay in source offsets. explode_bed12 writes one BED6
# row per block (name "<row>|<block>"), ft lifts those, and reassemble_bed12
# rebuilds each BED12 row from its lifted blocks.

_BED_HEADER_PREFIXES = ("#", "track", "browser")


@dataclass
class BedStats:
    rows_in: int = 0
    rows_out: int = 0
    blocks_in: int = 0
    rows_lost_blocks: int = 0
    rows_split: int = 0
    rows_rearranged: int = 0
    rows_flipped: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def explode_bed12(lines: Iterable[str]) -> Tuple[List[str], List[list], List[str]]:
    """``(bed6_rows, rows, header_lines)`` of a BED12(+N) file."""
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
        start = int(fields[1])
        sizes = [int(v) for v in fields[10].rstrip(",").split(",") if v != ""]
        offsets = [int(v) for v in fields[11].rstrip(",").split(",") if v != ""]
        if len(sizes) != int(fields[9]) or len(offsets) != len(sizes):
            raise ValueError(f"BED12 row {index + 1}: blockCount does not match its blocks")
        for block, (size, offset) in enumerate(zip(sizes, offsets)):
            bed6.append((fields[0], start + offset, start + offset + size,
                         f"{index}|{block}", fields[4], fields[5]))
    bed6.sort(key=lambda r: (r[0], r[1], r[2]))
    return ["\t".join(str(v) for v in r) for r in bed6], rows, header


_FLIP_STRAND = {"+": "-", "-": "+"}


def reassemble_bed12(lifted: Iterable[str], rows: List[list]) -> Tuple[List[str], BedStats]:
    """BED12 rows rebuilt from ft-lifted BED6 blocks (named ``<row>|<block>``).

    A row is dropped (counted) when a block did not lift, its blocks land on
    different contigs or overlap, or their order is neither kept nor
    reversed. Reversed order means the segment changed strand: the strand
    is flipped and per-block comma-list columns (13+) are reversed with the
    blocks. thickStart/thickEnd span the lifted row."""
    stats = BedStats(rows_in=len(rows))
    blocks: Dict[int, list] = defaultdict(list)
    for line in lifted:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 4 or "|" not in fields[3]:
            continue
        row, _, block = fields[3].rpartition("|")
        try:
            blocks[int(row)].append((int(block), fields[0], int(fields[1]), int(fields[2])))
        except ValueError:
            continue
    out = []
    for index, fields in enumerate(rows):
        count = int(fields[9])
        stats.blocks_in += count
        lifted_blocks = blocks.get(index, [])
        if len({b[0] for b in lifted_blocks}) != count or len(lifted_blocks) != count:
            stats.rows_lost_blocks += 1
            continue
        if len({b[1] for b in lifted_blocks}) != 1:
            stats.rows_split += 1
            continue
        by_position = sorted(lifted_blocks, key=lambda b: (b[2], b[3]))
        order = [b[0] for b in by_position]
        if order == sorted(order):
            flipped = False
        elif order == sorted(order, reverse=True):
            flipped = True
        else:
            stats.rows_rearranged += 1
            continue
        if any(by_position[i][3] > by_position[i + 1][2] for i in range(count - 1)):
            stats.rows_rearranged += 1
            continue
        chrom = by_position[0][1]
        start, end = by_position[0][2], by_position[-1][3]
        new = list(fields)
        new[0], new[1], new[2] = chrom, str(start), str(end)
        new[6], new[7] = str(start), str(end)
        new[10] = ",".join(str(b[3] - b[2]) for b in by_position)
        new[11] = ",".join(str(b[2] - start) for b in by_position)
        if flipped:
            stats.rows_flipped += 1
            new[5] = _FLIP_STRAND.get(new[5], new[5])
            if count > 1:
                for col in range(12, len(new)):
                    values = new[col].split(",")
                    if len(values) == count:
                        new[col] = ",".join(reversed(values))
        out.append("\t".join(new))
        stats.rows_out += 1
    return out, stats
