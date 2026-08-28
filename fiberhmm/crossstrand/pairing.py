"""Sequence-first cross-strand read pairing with footprint fallback.

Core (I/O-free) library behind ``fiberhmm-pair``. Given footprint-called DAF
reads, it:

1. compares opposite-flavor reads at reference A/T positions, excluding the
   reference C/G sites that DddA can alter;
2. accepts strict reciprocal sequence preferences, plus complete local 2x2
   assignments only when a grossly discordant edge rules out one diagonal;
3. removes those pairs and applies reciprocal-best nucleosome-dyad correlation
   to the sequence-ambiguous remainder, while vetoing gross sequence conflicts.

Staggered overlap chains are never globally optimized: weak sequence choices
cannot propagate into forced chromosome-scale assignments.

Everything here operates on lightweight feature objects so it is unit-testable
without a BAM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from fiberhmm.cli.extract_tags import _build_query_to_ref, _deam_positions_list
from fiberhmm.io.ma_tags import flip_interval_frame, parse_ma_tag

# Deamination flavor codes (match _deam_positions_list: 1 = Y = C->U, 0 = R = G->U).
FLAVOR_CT = 1
FLAVOR_GA = 0
_FLAVOR_NAME = {FLAVOR_CT: 'CT', FLAVOR_GA: 'GA'}


@dataclass(frozen=True)
class PairParams:
    """Tunables for cross-strand pairing.

    grid_bp        : reference resolution (bp) of the dyad-density signal.
    sigma_bp       : Gaussian dyad width; absorbs cross-strand edge variance.
    max_lag_bp     : +/- register shift searched when correlating (a whole-array
                     phase offset should not be penalized).
    min_overlap_bp : minimum genomic overlap for a pair to be scorable.
    min_nucs       : minimum dyads *within the overlap* on each read.
    min_score      : minimum cross-correlation floor to accept any pair.
    min_margin     : minimum (best - competitor) on both reads to accept, where
                     the competitor is ``max(second_best, null_floor)``. At a
                     2x2 locus the real second-best dominates (relative gate); at
                     a 1+1 locus (no alternative) ``null_floor`` dominates, so a
                     lone pair must beat the wrong-pair null to merge.
    null_floor     : the wrong-pair (different-homolog) correlation baseline,
                     calibrated from data (~null p90). Acts as a virtual
                     competitor so incomplete loci are still held to the null.
    min_sequence_bases: minimum shared reference-A/T bases for sequence use.
    max_sequence_mismatch_rate: gross-discordance threshold; edges above it
                     cannot enter footprint fallback.
    min_component_discordance_rate: minimum rejected-edge difference rate
                     needed to constrain a complete local 2x2 assignment.
    min_sequence_margin: minimum difference-rate advantage over the competing
                     reciprocal edge or 2x2 diagonal.
    max_sequence_pair_rate: maximum difference rate for a selected sequence
                     edge. This guards against choosing the least-bad edge in
                     a uniformly poor local component.

    Defaults are calibrated from the SRR33130342 2x2-locus null (p90~0.24,
    p95~0.30): the margin/reciprocal comparison controls precision, so the
    absolute floor is deliberately low and the null_floor guards lone pairs.
    """
    grid_bp: int = 10
    sigma_bp: float = 30.0
    max_lag_bp: int = 60
    min_overlap_bp: int = 1500
    min_nucs: int = 4
    min_score: float = 0.25
    min_margin: float = 0.05
    null_floor: float = 0.24
    min_sequence_bases: int = 500
    max_sequence_mismatch_rate: float = 0.002
    min_component_discordance_rate: float = 0.02
    min_sequence_margin: float = 0.002
    max_sequence_pair_rate: float = 0.01


@dataclass
class ReadFeat:
    """Per-read features for pairing (one primary alignment)."""
    index: int              # position in the batch (stable id)
    name: str               # query_name
    flavor: int             # FLAVOR_CT / FLAVOR_GA
    ref_start: int          # 0-based, inclusive
    ref_end: int            # 0-based, exclusive
    dyads: np.ndarray       # sorted reference dyad-center positions (int64)
    grid0: int              # first grid bin index (ref_start // grid_bp)
    signal: np.ndarray      # float32 dyad-density over [grid0 .. ref_end//grid]
    sequence_pos: Optional[np.ndarray] = None  # deamination-safe ref A/T sites
    sequence_base: Optional[np.ndarray] = None # canonical A/C/G/T query bases

    @property
    def flavor_name(self) -> str:
        return _FLAVOR_NAME[self.flavor]


def _gaussian_kernel(sigma_bp: float, grid_bp: int) -> np.ndarray:
    radius = max(1, int(round(3 * sigma_bp / grid_bp)))
    x = np.arange(-radius, radius + 1) * grid_bp
    return np.exp(-(x * x) / (2 * sigma_bp * sigma_bp)).astype(np.float32)


def read_flavor(read, prob_threshold: int = 0) -> Optional[int]:
    """Dominant deamination flavor of a read, or None if it has no calls.

    DAF reads are ~100% one flavor; ties (empty) return None.
    """
    calls = _deam_positions_list(read, _build_query_to_ref(read), prob_threshold)
    if not calls:
        return None
    ct = sum(1 for _, f in calls if f == FLAVOR_CT)
    ga = len(calls) - ct
    if ct == ga:
        return None
    return FLAVOR_CT if ct > ga else FLAVOR_GA


def nuc_dyads_ref(read) -> np.ndarray:
    """Reference-coordinate nucleosome dyad centers from the MA ``nuc`` track.

    MA intervals are in the molecular frame; for reverse reads they are flipped
    to the SEQ (query) frame before the query->reference lookup, matching
    ``fiberhmm-extract``. Returns a sorted int64 array (possibly empty).
    """
    try:
        parsed = parse_ma_tag(read.get_tag('MA'))
    except (KeyError, ValueError):
        return np.empty(0, dtype=np.int64)
    nucs = parsed['nuc']
    if not nucs:
        return np.empty(0, dtype=np.int64)
    q2r = _build_query_to_ref(read)
    n = len(q2r)
    read_len = int(parsed['read_length'])
    is_rev = bool(read.is_reverse)
    out: List[int] = []
    for s, length in nucs:
        if is_rev:
            s, length = flip_interval_frame(int(s), int(length), read_len)
        qc = int(s) + int(length) // 2
        if 0 <= qc < n:
            r = int(q2r[qc])
            if r >= 0:
                out.append(r)
    out.sort()
    return np.asarray(out, dtype=np.int64)


def _sequence_signature(read, reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return deamination-safe reference-position/query-base arrays.

    Reference C/G positions are excluded because DddA changes those bases.
    At reference A/T positions, query Y/R are canonicalized to C/G so genuine
    alternate alleles remain informative rather than being discarded.
    """
    q2r = _build_query_to_ref(read)
    seq = np.frombuffer((read.query_sequence or '').upper().encode(), dtype=np.uint8)
    n = min(len(q2r), len(seq))
    q2r, seq = q2r[:n], seq[:n].copy()
    seq[seq == ord('Y')] = ord('C')
    seq[seq == ord('R')] = ord('G')
    valid = (q2r >= 0) & (q2r < len(reference))
    q2r, seq = q2r[valid], seq[valid]
    ref = reference[q2r]
    valid = (((ref == ord('A')) | (ref == ord('T'))) &
             np.isin(seq, np.asarray(list(map(ord, 'ACGT')), dtype=np.uint8)))
    return q2r[valid].astype(np.int64), seq[valid]


def _sequence_signature_from_md(read) -> Tuple[np.ndarray, np.ndarray]:
    """Return the same A/T-only signature using the read's MD+CIGAR record.

    Archived DAF BAMs commonly retain an MD tag but not the reference FASTA
    used for alignment.  ``get_aligned_pairs(with_seq=True)`` reconstructs the
    reference base at each aligned query position from MD+CIGAR, which is all
    pairing needs.  This path is evidence-equivalent to indexing a FASTA at
    covered positions and fails closed when MD is absent or malformed.
    """
    if not getattr(read, 'has_tag', lambda _tag: False)('MD'):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint8)
    sequence = (getattr(read, 'query_sequence', None) or '').upper()
    if not sequence:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint8)
    try:
        pairs = read.get_aligned_pairs(with_seq=True)
    except (AttributeError, KeyError, TypeError, ValueError):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint8)

    positions: List[int] = []
    bases: List[int] = []
    for qpos, rpos, ref_base in pairs:
        if qpos is None or rpos is None or ref_base is None:
            continue
        ref = str(ref_base).upper()
        if ref not in ('A', 'T') or not (0 <= int(qpos) < len(sequence)):
            continue
        base = sequence[int(qpos)]
        if base == 'Y':
            base = 'C'
        elif base == 'R':
            base = 'G'
        if base not in 'ACGT':
            continue
        positions.append(int(rpos))
        bases.append(ord(base))
    return (np.asarray(positions, dtype=np.int64),
            np.asarray(bases, dtype=np.uint8))


def build_feature(read, index: int, params: PairParams,
                  prob_threshold: int = 0,
                  reference: Optional[np.ndarray] = None) -> Optional[ReadFeat]:
    """Build a :class:`ReadFeat` for a primary mapped read, or None if unusable.

    None when: unmapped/secondary/supplementary, no sequence, no deamination
    flavor. Nucleosome dyads and sequence evidence are optional individually;
    a read may be sequence-pairable even when it lacks a footprint pattern.
    """
    if read.is_unmapped or read.is_secondary or read.is_supplementary:
        return None
    if read.query_sequence is None or not read.has_tag('MA'):
        return None
    flavor = read_flavor(read, prob_threshold)
    if flavor is None:
        return None
    dyads = nuc_dyads_ref(read)
    g = params.grid_bp
    grid0 = int(read.reference_start) // g
    grid1 = int(read.reference_end) // g
    kern = _gaussian_kernel(params.sigma_bp, g)
    krad = len(kern) // 2
    sig = np.zeros(grid1 - grid0 + 1, dtype=np.float32)
    for c in dyads:
        cb = int(c) // g - grid0
        lo = max(0, cb - krad)
        hi = min(len(sig), cb + krad + 1)
        if hi > lo:
            klo = lo - (cb - krad)
            sig[lo:hi] += kern[klo:klo + (hi - lo)]
    if reference is not None:
        seq_pos, seq_base = _sequence_signature(read, reference)
    else:
        seq_pos, seq_base = _sequence_signature_from_md(read)
    return ReadFeat(index=index, name=read.query_name, flavor=flavor,
                    ref_start=int(read.reference_start),
                    ref_end=int(read.reference_end),
                    dyads=dyads, grid0=grid0, signal=sig,
                    sequence_pos=seq_pos, sequence_base=seq_base)


def score_pair(a: ReadFeat, b: ReadFeat, params: PairParams) -> Optional[float]:
    """Lag-tolerant normalized cross-correlation of two dyad signals over their
    genomic overlap, or None if the overlap is too small / under-covered.

    Returns a value in [-1, 1] (the max over lags in +/- max_lag_bp).
    """
    g = params.grid_bp
    lo = max(a.grid0, b.grid0)
    hi = min(a.grid0 + len(a.signal), b.grid0 + len(b.signal))
    if (hi - lo) * g < params.min_overlap_bp:
        return None
    lo_bp, hi_bp = lo * g, hi * g
    if int(np.sum((a.dyads >= lo_bp) & (a.dyads < hi_bp))) < params.min_nucs:
        return None
    if int(np.sum((b.dyads >= lo_bp) & (b.dyads < hi_bp))) < params.min_nucs:
        return None
    sa = a.signal[lo - a.grid0:hi - a.grid0]
    sb = b.signal[lo - b.grid0:hi - b.grid0]
    m = min(len(sa), len(sb))
    maxlag = params.max_lag_bp // g
    if m < 2 * maxlag + 5:
        return None
    sa = sa[:m] - sa[:m].mean()
    sb = sb[:m] - sb[:m].mean()
    na = float(np.sqrt(np.dot(sa, sa)))
    nb = float(np.sqrt(np.dot(sb, sb)))
    if na == 0.0 or nb == 0.0:
        return None
    best = -1.0
    for lag in range(-maxlag, maxlag + 1):
        if lag >= 0:
            x, y = sa[lag:], sb[:len(sb) - lag]
        else:
            x, y = sa[:len(sa) + lag], sb[-lag:]
        k = min(len(x), len(y))
        if k == 0:
            continue
        r = float(np.dot(x[:k], y[:k]) / (na * nb))
        if r > best:
            best = r
    return best


@dataclass(frozen=True)
class SequenceScore:
    bases: int
    mismatches: int
    rate: float


def score_sequence(a: ReadFeat, b: ReadFeat) -> SequenceScore:
    """Compare two reads only at shared, deamination-safe reference A/T sites."""
    if a.sequence_pos is None or b.sequence_pos is None:
        return SequenceScore(0, 0, float('nan'))
    pos, ai, bi = np.intersect1d(
        a.sequence_pos, b.sequence_pos, assume_unique=True, return_indices=True,
    )
    bases = int(len(pos))
    mismatches = int(np.count_nonzero(a.sequence_base[ai] != b.sequence_base[bi]))
    return SequenceScore(
        bases, mismatches, mismatches / bases if bases else float('nan'),
    )


# Pairing status codes written to the ``mt:A`` tag.
STATUS_PAIRED = 'P'       # resolved reciprocal-best pair above score+margin
STATUS_UNRESOLVED = 'U'   # had candidate(s) but failed margin/score gate
STATUS_NONE = '.'         # no overlapping opposite-strand candidate


@dataclass
class PairResult:
    partner: Dict[int, int]        # read index -> mate index (resolved only)
    score: Dict[int, float]        # read index -> pair correlation
    margin: Dict[int, float]       # read index -> min(best-2nd over both reads)
    status: Dict[int, str]         # read index -> STATUS_*
    method: Dict[int, str]         # S = sequence assignment, F = footprint
    sequence: Dict[int, SequenceScore]
    sequence_margin: Dict[int, float]
    sequence_kind: Dict[int, str]  # R = reciprocal, C = constrained 2x2


def _overlapping_opposite_pairs(feats: Sequence[ReadFeat]):
    """Yield each genomically overlapping opposite-flavor pair once."""
    order = sorted(feats, key=lambda f: f.ref_start)
    for ai, a in enumerate(order):
        for b in order[ai + 1:]:
            if b.ref_start >= a.ref_end:
                break
            if b.flavor != a.flavor:
                yield a, b


def _sequence_assignment(
    feats: Sequence[ReadFeat], params: PairParams,
) -> Tuple[Dict[int, int], Dict[int, SequenceScore], Dict[int, float],
           Dict[Tuple[int, int], SequenceScore], Set[int], Dict[int, str]]:
    """Resolve stable edges of local bipartite sequence assignments.

    Strict reciprocal-best edges are accepted with a two-sided difference-rate
    margin. Complete 2x2 components receive one additional comparison of the
    two possible diagonals, but only when a grossly incompatible rejected edge
    supplies the constraint; larger overlap chains are never jointly solved.
    """
    by_idx = {f.index: f for f in feats}
    edge_score: Dict[Tuple[int, int], SequenceScore] = {}
    neighbors: Dict[int, Set[int]] = {f.index: set() for f in feats}
    candidate_nodes: Set[int] = set()
    for a, b in _overlapping_opposite_pairs(feats):
        seq = score_sequence(a, b)
        key = tuple(sorted((a.index, b.index)))
        edge_score[key] = seq
        if seq.bases < params.min_sequence_bases:
            continue
        neighbors[a.index].add(b.index)
        neighbors[b.index].add(a.index)
        candidate_nodes.update((a.index, b.index))

    components: List[Set[int]] = []
    unseen = set(candidate_nodes)
    while unseen:
        root = unseen.pop()
        component, stack = {root}, [root]
        while stack:
            node = stack.pop()
            for other in neighbors[node]:
                if other not in component:
                    component.add(other)
                    unseen.discard(other)
                    stack.append(other)
        components.append(component)

    partner: Dict[int, int] = {}
    selected_score: Dict[int, SequenceScore] = {}
    margin: Dict[int, float] = {}
    kind: Dict[int, str] = {}

    # First take only strict two-sided sequence preferences. Unlike a global
    # chromosome matching, this cannot propagate a weak choice down a chain.
    best: Dict[int, Tuple[float, int]] = {}
    second: Dict[int, float] = {}
    for key, seq in edge_score.items():
        if (seq.bases < params.min_sequence_bases or
                seq.rate > params.max_sequence_pair_rate):
            continue
        i, j = key
        for left, right in ((i, j), (j, i)):
            old = best.get(left)
            if old is None or seq.rate < old[0]:
                if old is not None:
                    second[left] = old[0]
                best[left] = (seq.rate, right)
            elif left not in second or seq.rate < second[left]:
                second[left] = seq.rate
    for i, (rate, j) in best.items():
        if i in partner or j in partner or best.get(j, (None, None))[1] != i:
            continue
        if i not in second or j not in second:
            continue
        seq_margin = min(second[i] - rate, second[j] - best[j][0])
        if seq_margin + 1e-12 < params.min_sequence_margin:
            continue
        seq = edge_score[tuple(sorted((i, j)))]
        partner[i] = j
        partner[j] = i
        selected_score[i] = selected_score[j] = seq
        margin[i] = margin[j] = seq_margin
        kind[i] = kind[j] = 'R'

    for component in components:
        # Larger connected components are staggered overlap chains, not local
        # molecule sets. Never let joint optimization force matches through
        # them; the strict reciprocal pass above and footprint fallback below
        # remain available.
        if len(component) != 4 or any(i in partner for i in component):
            continue
        ct = sorted(i for i in component if by_idx[i].flavor == FLAVOR_CT)
        ga = sorted(i for i in component if by_idx[i].flavor == FLAVOR_GA)
        if len(ct) != 2 or len(ga) != 2:
            continue
        # Require both possible diagonals to be sequence-comparable.
        real_edges = sum(j in neighbors[i] for i in ct for j in ga)
        if real_edges != 4:
            continue
        diagonals = (
            ((ct[0], ga[0]), (ct[1], ga[1])),
            ((ct[0], ga[1]), (ct[1], ga[0])),
        )
        diagonal_scores = []
        for diagonal in diagonals:
            seqs = [edge_score[tuple(sorted(edge))] for edge in diagonal]
            diagonal_scores.append(sum(seq.rate for seq in seqs))
        chosen_index = int(diagonal_scores[1] < diagonal_scores[0])
        assignment_margin = abs(diagonal_scores[0] - diagonal_scores[1])
        if assignment_margin + 1e-12 < params.min_sequence_margin:
            continue
        chosen = diagonals[chosen_index]
        rejected = diagonals[1 - chosen_index]
        chosen_seqs = [edge_score[tuple(sorted(edge))] for edge in chosen]
        rejected_seqs = [edge_score[tuple(sorted(edge))] for edge in rejected]
        if any(seq.rate > params.max_sequence_pair_rate for seq in chosen_seqs):
            continue
        # A 2x2 constraint is used only for the case it was designed to solve:
        # at least one opposite edge is grossly sequence-incompatible. Ordinary
        # SNP-scale differences are too easily confounded by residual consensus
        # errors and remain for the physical footprint fallback.
        if (max(seq.rate for seq in rejected_seqs) <=
                params.min_component_discordance_rate):
            continue
        for (i, j), seq in zip(chosen, chosen_seqs):
            partner[i] = j
            partner[j] = i
            selected_score[i] = selected_score[j] = seq
            margin[i] = margin[j] = assignment_margin
            kind[i] = kind[j] = 'C'
    return partner, selected_score, margin, edge_score, candidate_nodes, kind


def _footprint_assignment(
    feats: Sequence[ReadFeat], params: PairParams,
    sequence_edges: Dict[Tuple[int, int], SequenceScore],
) -> Tuple[Dict[int, int], Dict[int, float], Dict[int, float], Set[int]]:
    """Reciprocal-best footprint matching, vetoing gross sequence conflicts."""
    by_idx = {f.index: f for f in feats}
    best: Dict[int, Tuple[float, int]] = {f.index: (-2.0, -1) for f in feats}
    second: Dict[int, float] = {f.index: -2.0 for f in feats}
    candidate_nodes: Set[int] = set()

    def offer(i: int, val: float, j: int) -> None:
        bs, _ = best[i]
        if val > bs:
            second[i] = bs
            best[i] = (val, j)
        elif val > second[i]:
            second[i] = val

    for a, b in _overlapping_opposite_pairs(feats):
        seq = sequence_edges.get(tuple(sorted((a.index, b.index))))
        if (seq is not None and seq.bases >= params.min_sequence_bases and
                seq.rate > params.max_sequence_mismatch_rate):
            continue
        value = score_pair(a, b, params)
        if value is None:
            continue
        candidate_nodes.update((a.index, b.index))
        offer(a.index, value, b.index)
        offer(b.index, value, a.index)

    partner: Dict[int, int] = {}
    score: Dict[int, float] = {}
    margin: Dict[int, float] = {}
    for idx in by_idx:
        bs, bj = best[idx]
        if bj < 0:
            continue
        comp_i = max(second[idx], params.null_floor)
        comp_j = max(second[bj], params.null_floor)
        marg = min(bs - comp_i, best[bj][0] - comp_j)
        if (best[bj][1] == idx and bs >= params.min_score and
                marg >= params.min_margin):
            partner[idx] = bj
            score[idx] = bs
            margin[idx] = marg
    return partner, score, margin, candidate_nodes


def assign_pairs(feats: Sequence[ReadFeat], params: PairParams) -> PairResult:
    """Resolve sequence assignments first, then footprint-match leftovers."""
    by_idx = {f.index: f for f in feats}
    seq_partner, seq_score, seq_margin, seq_edges, seq_nodes, seq_kind = \
        _sequence_assignment(feats, params)
    remaining = [f for f in feats if f.index not in seq_partner]
    fp_partner, fp_score, fp_margin, fp_nodes = _footprint_assignment(
        remaining, params, seq_edges,
    )

    partner = dict(seq_partner)
    partner.update(fp_partner)
    score: Dict[int, float] = {}
    margin: Dict[int, float] = {}
    method: Dict[int, str] = {}
    sequence: Dict[int, SequenceScore] = dict(seq_score)
    sequence_margin: Dict[int, float] = dict(seq_margin)
    for idx, mate in seq_partner.items():
        method[idx] = 'S'
        footprint = score_pair(by_idx[idx], by_idx[mate], params)
        if footprint is not None:
            score[idx] = footprint
        margin[idx] = seq_margin[idx]
    for idx, mate in fp_partner.items():
        method[idx] = 'F'
        score[idx] = fp_score[idx]
        margin[idx] = fp_margin[idx]
        seq = seq_edges.get(tuple(sorted((idx, mate))))
        if seq is not None:
            sequence[idx] = seq
    status: Dict[int, str] = {}
    for idx in by_idx:
        if idx in partner:
            status[idx] = STATUS_PAIRED
        elif idx in seq_nodes or idx in fp_nodes:
            status[idx] = STATUS_UNRESOLVED
        else:
            status[idx] = STATUS_NONE
    return PairResult(
        partner=partner, score=score, margin=margin, status=status,
        method=method, sequence=sequence, sequence_margin=sequence_margin,
        sequence_kind=seq_kind,
    )
