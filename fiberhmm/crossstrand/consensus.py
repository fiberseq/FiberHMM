"""Both-strand consensus construction from a resolved cross-strand pair.

Given a CT read (C->T strand) and its GA mate (G->A strand), build one synthetic
consensus read spanning the **union** of their reference spans, with each
strand's deaminations applied to a reference-frame (ungapped) sequence:

    reference C's deaminated by the CT read  ->  ``Y``  (C-or-T)
    reference G's deaminated by the GA read  ->  ``R``  (A-or-G)

Deaminations are emitted in the same IUPAC R/Y encoding these DAF BAMs already
use, which is FiberHMM's native ``--mode daf`` path -- so no MD tag is required
and the re-caller reads the consensus the same way it reads the inputs.

The strand-coverage regime is recorded spec-natively in the MA tag as a custom
``deam`` annotation type with two strand-tagged intervals -- ``deam+`` (CT-read
coverage) and ``deam-`` (GA-read coverage). A position in both is a both-strand
site (C's *and* G's informative there); a position in only one is single-strand.

Simplification (v1): the consensus follows the reference frame with an all-``M``
CIGAR, so insertions are dropped and source deletions become missing-strand
coverage; deaminations are preserved. Reference bases are taken from the reads'
own aligned bases (canonicalized at deaminated sites). A disagreement between
two canonical source bases becomes ``N`` rather than letting either strand
fabricate protected evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from fiberhmm.cli.extract_tags import _build_query_to_ref, _deam_positions_list
from fiberhmm.crossstrand.pairing import FLAVOR_CT, FLAVOR_GA


def _ref_base_map_and_deam(read, prob_threshold=0):
    """Return ``(ref_base_map, ct_deam, ga_deam)`` for one read.

    ``ref_base_map`` maps reference position -> canonical reference base
    (A/C/G/T), taken from the read's aligned base and overridden to C/G at
    deaminated sites (where the stored base may be an IUPAC Y/R or a T/A). The
    two sets are the read's C->T and G->A deaminated reference positions.
    """
    seq = read.query_sequence
    if seq is None:
        return None
    q2r = _build_query_to_ref(read)
    calls = _deam_positions_list(read, q2r, prob_threshold)
    ct = {p for p, f in calls if f == FLAVOR_CT}
    ga = {p for p, f in calls if f == FLAVOR_GA}
    ref: Dict[int, str] = {}
    n = len(q2r)
    for qpos in range(min(n, len(seq))):
        rp = int(q2r[qpos])
        if rp >= 0:
            base = seq[qpos].upper()
            ref[rp] = base if base in 'ACGT' else 'N'
    for p in ct:
        ref[p] = 'C'
    for p in ga:
        ref[p] = 'G'
    return ref, ct, ga


def format_deam_ma_intervals(
    read_length: int,
    ct_intervals: Sequence[Tuple[int, int]],
    ga_intervals: Sequence[Tuple[int, int]],
) -> str:
    """Format exact CT/GA aligned-reference coverage as MA intervals.

    A source deletion is absence of strand evidence, not a protected target.
    Keeping disjoint aligned runs prevents the other strand's reference base
    from turning a deletion into a false miss during the joint re-call.
    """
    def encoded(intervals: Sequence[Tuple[int, int]]) -> str:
        return ','.join(f"{int(start) + 1}-{int(length)}"
                        for start, length in intervals if int(length) > 0)

    return (f"{read_length};deam+:{encoded(ct_intervals)};"
            f"deam-:{encoded(ga_intervals)}")


def format_deam_ma_tag(read_length: int,
                       ct_start_q: int, ct_len: int,
                       ga_start_q: int, ga_len: int) -> str:
    """MA:Z carrying the strand-coverage regime as a ``deam`` type.

    ``deam+`` = CT-strand coverage, ``deam-`` = GA-strand coverage, in query
    coordinates (0-based in; 1-based out per spec). Both-strand region is the
    intersection of the two intervals; the caller derives regime from that.
    """
    return format_deam_ma_intervals(
        read_length,
        [(ct_start_q, ct_len)],
        [(ga_start_q, ga_len)],
    )


def _coverage_intervals(
    reference_positions: Iterable[int], origin: int,
) -> List[Tuple[int, int]]:
    """Compress aligned reference positions into query-frame MA runs."""
    positions = sorted({int(position) for position in reference_positions})
    if not positions:
        return []
    intervals: List[Tuple[int, int]] = []
    run_start = previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1:
            intervals.append((run_start - origin, previous - run_start + 1))
            run_start = position
        previous = position
    intervals.append((run_start - origin, previous - run_start + 1))
    return intervals


@dataclass
class Consensus:
    ref_start: int          # 0-based
    length: int             # all-M length (== union span)
    seq: str                # R/Y-encoded reference-frame sequence
    deam_count: int         # number of deaminated (R/Y) positions
    base_conflicts: int     # source-base disagreements replaced by N
    mapq: int               # conservative minimum source alignment MAPQ
    ma: str                 # deam+/deam- regime tag
    both_start: int         # both-strand region (0-based, ref coords)
    both_end: int           # exclusive
    ct_name: str
    ga_name: str


def build_consensus(ct_read, ga_read, prob_threshold=0) -> Optional[Consensus]:
    """Build a both-strand :class:`Consensus` from a CT/GA pair, or None if a
    read is unusable (no sequence)."""
    a = _ref_base_map_and_deam(ct_read, prob_threshold)
    b = _ref_base_map_and_deam(ga_read, prob_threshold)
    if a is None or b is None:
        return None
    ref_ct, ct_deam, _ = a
    ref_ga, _, ga_deam = b
    if ct_deam & ga_deam:
        # A reference position cannot simultaneously be canonical C and G.
        return None
    # Keep source coverage maps immutable: they define the strand-specific
    # evidence masks below.  Mutating ``ref_ct`` into the union would falsely
    # claim CT coverage wherever only the GA source aligned.
    ref: Dict[int, str] = {}
    base_conflicts = 0
    for pos in ref_ct.keys() | ref_ga.keys():
        ct_base = ref_ct.get(pos)
        ga_base = ref_ga.get(pos)
        if ct_base is None:
            ref[pos] = ga_base
        elif ga_base is None:
            ref[pos] = ct_base
        elif ct_base == ga_base:
            ref[pos] = ct_base
        elif ct_base == 'N':
            ref[pos] = ga_base
        elif ga_base == 'N':
            ref[pos] = ct_base
        else:
            ref[pos] = 'N'
            base_conflicts += 1

    rs = min(ct_read.reference_start, ga_read.reference_start)
    re = max(ct_read.reference_end, ga_read.reference_end)
    length = re - rs

    seq_chars: List[str] = []
    deam_count = 0
    for pos in range(rs, re):
        if pos in ct_deam:
            seq_chars.append('Y')
            deam_count += 1
        elif pos in ga_deam:
            seq_chars.append('R')
            deam_count += 1
        else:
            seq_chars.append(ref.get(pos, 'N'))
    seq = ''.join(seq_chars)

    ma = format_deam_ma_intervals(
        length,
        _coverage_intervals(ref_ct, rs),
        _coverage_intervals(ref_ga, rs),
    )
    return Consensus(
        ref_start=rs, length=length, seq=seq,
        deam_count=deam_count, base_conflicts=base_conflicts,
        mapq=min(int(ct_read.mapping_quality), int(ga_read.mapping_quality)), ma=ma,
        both_start=max(ct_read.reference_start, ga_read.reference_start),
        both_end=min(ct_read.reference_end, ga_read.reference_end),
        ct_name=ct_read.query_name, ga_name=ga_read.query_name,
    )
