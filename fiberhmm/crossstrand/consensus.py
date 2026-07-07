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
CIGAR, so small indels in the source reads are dropped; deaminations, which are
substitutions, are preserved. Reference bases are taken from the reads' own
aligned bases (canonicalized at deaminated sites), so no reference FASTA is
required and this is robust to the input's deamination encoding (R/Y, MM/ML, or
MD) because deamination positions come from the shared ``_deam_positions_list``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from fiberhmm.cli.extract_tags import _build_query_to_ref, _deam_positions_list
from fiberhmm.crossstrand.pairing import FLAVOR_CT, FLAVOR_GA


def _ref_base_map_and_deam(read):
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
    calls = _deam_positions_list(read, q2r, 0)
    ct = {p for p, f in calls if f == FLAVOR_CT}
    ga = {p for p, f in calls if f == FLAVOR_GA}
    ref: Dict[int, str] = {}
    n = len(q2r)
    for qpos in range(min(n, len(seq))):
        rp = int(q2r[qpos])
        if rp >= 0:
            ref[rp] = seq[qpos]
    for p in ct:
        ref[p] = 'C'
    for p in ga:
        ref[p] = 'G'
    return ref, ct, ga


def format_deam_ma_tag(read_length: int,
                       ct_start_q: int, ct_len: int,
                       ga_start_q: int, ga_len: int) -> str:
    """MA:Z carrying the strand-coverage regime as a ``deam`` type.

    ``deam+`` = CT-strand coverage, ``deam-`` = GA-strand coverage, in query
    coordinates (0-based in; 1-based out per spec). Both-strand region is the
    intersection of the two intervals; the caller derives regime from that.
    """
    return (f"{read_length};"
            f"deam+:{ct_start_q + 1}-{ct_len};"
            f"deam-:{ga_start_q + 1}-{ga_len}")


@dataclass
class Consensus:
    ref_start: int          # 0-based
    length: int             # all-M length (== union span)
    seq: str                # R/Y-encoded reference-frame sequence
    nm: int                 # number of deaminated (R/Y) positions
    ma: str                 # deam+/deam- regime tag
    both_start: int         # both-strand region (0-based, ref coords)
    both_end: int           # exclusive
    ct_name: str
    ga_name: str


def build_consensus(ct_read, ga_read) -> Optional[Consensus]:
    """Build a both-strand :class:`Consensus` from a CT/GA pair, or None if a
    read is unusable (no sequence)."""
    a = _ref_base_map_and_deam(ct_read)
    b = _ref_base_map_and_deam(ga_read)
    if a is None or b is None:
        return None
    ref_ct, ct_deam, _ = a
    ref_ga, _, ga_deam = b
    ref = ref_ct
    for pos, base in ref_ga.items():
        ref.setdefault(pos, base)

    rs = min(ct_read.reference_start, ga_read.reference_start)
    re = max(ct_read.reference_end, ga_read.reference_end)
    length = re - rs

    seq_chars: List[str] = []
    nm = 0
    for pos in range(rs, re):
        if pos in ct_deam:
            seq_chars.append('Y')
            nm += 1
        elif pos in ga_deam:
            seq_chars.append('R')
            nm += 1
        else:
            seq_chars.append(ref.get(pos, 'N'))
    seq = ''.join(seq_chars)

    ma = format_deam_ma_tag(
        length,
        ct_read.reference_start - rs, ct_read.reference_end - ct_read.reference_start,
        ga_read.reference_start - rs, ga_read.reference_end - ga_read.reference_start,
    )
    return Consensus(
        ref_start=rs, length=length, seq=seq, nm=nm, ma=ma,
        both_start=max(ct_read.reference_start, ga_read.reference_start),
        both_end=min(ct_read.reference_end, ga_read.reference_end),
        ct_name=ct_read.query_name, ga_name=ga_read.query_name,
    )
