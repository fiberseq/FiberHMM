"""
BAM reader module for FiberHMM v2
Extracts m6A methylation calls from fiber-seq BAM files and computes
hexamer context directly from the read sequence (no reference H5 needed).

Supports:
- pacbio-fiber (default): PacBio fiber-seq (A-centered with RC for T positions)
- nanopore-fiber: Nanopore fiber-seq (A-centered only, no RC)
- daf: DAF-seq deamination (strand-specific C/G-centered)
- Variable context sizes (default k=3 for 7-mer, up to k=10 for 21-mer)
"""

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pysam

from fiberhmm.core.tag_access import get_preferred_tag

try:
    from numba import njit as _numba_njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def _numba_njit(*args, **kwargs):  # type: ignore[misc]
        def _wrap(fn): return fn
        return _wrap


# =============================================================================
# Context encoding with variable sizes
# =============================================================================

class ContextEncoder:
    """
    Builds and caches hexamer/context lookup tables for variable sizes.
    """
    _cache: Dict[Tuple[str, int, bool], Dict[str, int]] = {}

    @classmethod
    def get_lookup(cls, center_base: str, context_size: int = 3,
                   include_rc: bool = False) -> Dict[str, int]:
        """
        Get or build a context lookup table.

        Args:
            center_base: Center base ('A', 'T', 'C', 'G')
            context_size: Bases on each side (3 = 7-mer, 5 = 11-mer, etc.)
            include_rc: Include reverse complement mappings

        Returns:
            Dict mapping context string to numeric code
        """
        key = (center_base.upper(), context_size, include_rc)

        if key not in cls._cache:
            cls._cache[key] = cls._build_lookup(center_base, context_size, include_rc)

        return cls._cache[key]

    @classmethod
    def _build_lookup(cls, center_base: str, context_size: int,
                      include_rc: bool) -> Dict[str, int]:
        """Build a new lookup table."""
        center = center_base.upper()
        bases = ['A', 'C', 'G', 'T']

        # Generate all k-mers for flanking regions
        def gen_kmers(k):
            if k == 0:
                return ['']
            return [s + b for s in gen_kmers(k - 1) for b in bases]

        flanks = gen_kmers(context_size)

        # Build all contexts
        contexts = [left + center + right for left in flanks for right in flanks]
        lookup = {ctx: i for i, ctx in enumerate(sorted(contexts))}

        # Add reverse complements if requested
        if include_rc:
            rc_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}

            def rc(seq):
                return ''.join(rc_map.get(b, 'N') for b in reversed(seq))

            for ctx in list(lookup.keys()):
                rc_ctx = rc(ctx)
                if rc_ctx not in lookup:
                    lookup[rc_ctx] = lookup[ctx]

        return lookup

    @classmethod
    def get_n_codes(cls, context_size: int) -> int:
        """Get number of unique codes for a context size."""
        return 4 ** (2 * context_size)

    @classmethod
    def get_non_target_code(cls, context_size: int) -> int:
        """Get the code for non-target positions."""
        return cls.get_n_codes(context_size)


# Pre-built lookup tables for default hexamer size (k=3)
# These are kept for backwards compatibility
def _build_hexamer_lookup(center_base: str = 'A') -> Dict[str, int]:
    return ContextEncoder.get_lookup(center_base, context_size=3, include_rc=False)

def _build_hexamer_lookup_with_rc(center_base: str = 'A') -> Dict[str, int]:
    return ContextEncoder.get_lookup(center_base, context_size=3, include_rc=True)

# Legacy lookup tables (k=3 hexamers)
HEXAMER_LOOKUP_A = _build_hexamer_lookup_with_rc('A')
HEXAMER_LOOKUP_T = _build_hexamer_lookup('T')
HEXAMER_LOOKUP_C = _build_hexamer_lookup('C')
HEXAMER_LOOKUP_G = _build_hexamer_lookup('G')
HEXAMER_LOOKUP_A_STRANDED = _build_hexamer_lookup('A')
HEXAMER_LOOKUP = HEXAMER_LOOKUP_A

# Default constants (for k=3)
NON_TARGET_CODE = 4096
UNMETHYLATED_OFFSET = 4097
NON_A_CODE = NON_TARGET_CODE  # Backwards compatibility

_CIGAR_QUERY_REF_OPS = {0, 7, 8}  # M, =, X
_CIGAR_QUERY_ONLY_OPS = {1, 4}    # I, S
_CIGAR_REF_ONLY_OPS = {2, 3}      # D, N


@dataclass
class FiberRead:
    """Represents a single fiber-seq read with m6A calls and encoding."""
    read_id: str
    chrom: str
    ref_start: int
    ref_end: int
    strand: str
    query_sequence: str
    m6a_query_positions: Set[int]  # Query positions with m6A
    query_to_ref: List[Optional[int]]  # Maps query pos -> ref pos
    is_reverse: bool  # Whether read is reverse-aligned

    @property
    def query_length(self) -> int:
        return len(self.query_sequence)

    @property
    def ref_length(self) -> int:
        return self.ref_end - self.ref_start


def get_reference_positions(read: pysam.AlignedSegment) -> List[Optional[int]]:
    """
    Build mapping from query position to reference position using CIGAR.

    Returns list where index is query position and value is reference position.
    None indicates an insertion (no corresponding reference position).
    """
    if read.cigartuples is None:
        return []

    ref_positions = []
    ref_pos = read.reference_start

    for op, length in read.cigartuples:
        if op in _CIGAR_QUERY_REF_OPS:
            for _ in range(length):
                ref_positions.append(ref_pos)
                ref_pos += 1
        elif op in _CIGAR_QUERY_ONLY_OPS:
            for _ in range(length):
                ref_positions.append(None)
        elif op in _CIGAR_REF_ONLY_OPS:
            ref_pos += length
        elif op == 5:  # H - hard clip, not in query sequence.
            pass

    return ref_positions


def get_reference_positions_array(read: pysam.AlignedSegment) -> np.ndarray:
    """Return query-to-reference positions as int32, using -1 for insertions."""
    return np.array(
        [p if p is not None else -1 for p in get_reference_positions(read)],
        dtype=np.int32,
    )


def _pysam_modified_base_allowed(base, mod_code, mode: str) -> bool:
    if mode == 'pacbio-fiber':
        # 21839 is the ChEBI code for m6A.
        return base in ('A', 'T') and mod_code in ('a', 21839)
    if mode == 'nanopore-fiber':
        return base == 'A' and mod_code in ('a', 21839)
    if mode == 'daf':
        # DAF-seq deamination can be represented against converted or
        # original bases depending on the caller.
        return base in ('T', 'A', 'C', 'G')
    return True


def get_modified_positions_pysam(read, prob_threshold: int = 125, mode: str = 'pacbio-fiber') -> Set[int]:
    """
    Get modification positions using pysam's built-in modified_bases property.

    This is more reliable than manual MM/ML parsing.

    Args:
        read: pysam AlignedSegment
        prob_threshold: Minimum probability (0-255) to include modification
        mode: 'pacbio-fiber' for adenine methylation, 'daf' for cytosine modifications

    Returns:
        Set of query positions with confident modification calls
    """
    mod_positions = set()

    try:
        # pysam.modified_bases returns:
        # Dict[(canonical_base, strand, modification_code)] -> [(pos, qual), ...]
        # qual is 256*probability, or -1 if unknown
        mod_bases = read.modified_bases

        if not mod_bases:
            return mod_positions

        for (base, strand, mod_code), positions in mod_bases.items():
            if not _pysam_modified_base_allowed(base, mod_code, mode):
                continue

            for pos, qual in positions:
                # qual is 256*prob, threshold is on 0-255 scale
                # So we compare qual/256*255 >= threshold, or qual >= threshold * 256/255
                if qual == -1:  # Unknown quality, include it
                    mod_positions.add(pos)
                elif qual >= prob_threshold:  # qual is already on ~0-255 scale (actually 0-256)
                    mod_positions.add(pos)

    except Exception:
        # Fall back to manual parsing if modified_bases fails
        pass

    return mod_positions


_COMPLEMENT_TABLE = str.maketrans('ACGTacgtNn', 'TGCAtgcaNn')


def _ml_tag_to_uint8_array(ml_tag) -> np.ndarray:
    if isinstance(ml_tag, (bytes, bytearray, memoryview)):
        return np.frombuffer(ml_tag, dtype=np.uint8)
    if isinstance(ml_tag, np.ndarray):
        return ml_tag
    return np.asarray(ml_tag, dtype=np.uint8)


def _mm_search_sequence(seq_upper: str, is_reverse: bool) -> str:
    if is_reverse:
        return seq_upper.translate(_COMPLEMENT_TABLE)[::-1]
    return seq_upper


def _mm_skip_counts(raw_counts) -> np.ndarray:
    try:
        return np.asarray(raw_counts, dtype=np.int64)
    except (ValueError, TypeError):
        skip_counts = []
        for value in raw_counts:
            if value.strip():
                try:
                    skip_counts.append(int(value))
                except ValueError:
                    continue
        return np.asarray(skip_counts, dtype=np.int64)


def _mm_mod_spec_parts(mod_spec: str):
    parts = mod_spec.split(',')
    if len(parts) < 2:
        return None
    return parts[0], _mm_skip_counts(parts[1:])


def _mm_target_base(base_mod: str) -> Optional[str]:
    if len(base_mod) == 0:
        return None
    return base_mod[0].upper()


def _mm_base_and_mod_code(base_mod: str) -> Optional[Tuple[str, str]]:
    if len(base_mod) < 3:
        return None
    return base_mod[0].upper(), base_mod[2]


def _cached_base_positions(cache: Dict[str, np.ndarray], target_base: str,
                           seq_bytes: np.ndarray) -> np.ndarray:
    if target_base not in cache:
        cache[target_base] = np.where(seq_bytes == ord(target_base))[0]
    return cache[target_base]


def _target_base_allowed_for_mode(target_base: str, mode: str) -> bool:
    if mode == 'pacbio-fiber':
        return target_base in ('A', 'T')
    if mode == 'nanopore-fiber':
        return target_base == 'A'
    if mode == 'daf':
        return target_base in ('T', 'A', 'C', 'G')
    return True


def _mm_valid_positions_and_qualities(skip_arr: np.ndarray,
                                      base_positions: np.ndarray,
                                      ml_slice: np.ndarray,
                                      q_len: int,
                                      is_reverse: bool) -> Tuple[np.ndarray, np.ndarray]:
    n_mods = len(skip_arr)
    if n_mods == 0 or len(base_positions) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.uint8)

    base_indices = np.cumsum(skip_arr) + np.arange(n_mods)
    valid = base_indices < len(base_positions)
    if len(ml_slice) < n_mods:
        valid[len(ml_slice):] = False

    if not np.any(valid):
        return np.array([], dtype=np.int64), np.array([], dtype=np.uint8)

    positions = base_positions[base_indices[valid]]
    qualities = (
        ml_slice[valid]
        if len(ml_slice) >= n_mods
        else ml_slice[valid[:len(ml_slice)]]
    )

    if is_reverse:
        positions = q_len - 1 - positions

    return positions, qualities


def _mm_positions_from_spec(skip_arr: np.ndarray,
                            base_positions: np.ndarray,
                            ml_slice: np.ndarray,
                            q_len: int,
                            is_reverse: bool,
                            prob_threshold: int) -> np.ndarray:
    positions, qualities = _mm_valid_positions_and_qualities(
        skip_arr, base_positions, ml_slice, q_len, is_reverse,
    )
    if len(qualities) == 0:
        return positions
    return positions[qualities >= prob_threshold]


def parse_mm_ml_per_mod_type(mm_tag: str, ml_tag,
                               sequence: str, is_reverse: bool) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]:
    """Parse MM/ML into per-mod-type position + quality arrays (SEQ frame).

    Produces positions in the same frame as ``pysam.modified_bases``: stored
    query_sequence (SEQ) frame, accounting for the SAM spec's requirement
    that MM walks in the ORIGINAL sequencing direction for reverse-aligned
    reads.  Validated against pysam on both strands (see
    ``tests/test_mm_parser_vs_pysam.py``).

    Returns dict mapping ``(target_base, mod_code)`` -> ``(positions, qualities)``
    with positions in SEQ frame, unfiltered (caller applies prob_threshold).
    """
    result: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}

    if not mm_tag or not ml_tag:
        return result
    try:
        if len(ml_tag) == 0:
            return result
    except TypeError:
        pass

    ml_arr_all = _ml_tag_to_uint8_array(ml_tag)

    seq_upper = sequence.upper()
    q_len = len(seq_upper)

    # For reverse-aligned reads, MM walks in the RC of SEQ (original
    # sequencing direction).  See SAM specs on MM/ML orientation.
    search_seq = _mm_search_sequence(seq_upper, is_reverse)
    seq_bytes = np.frombuffer(search_seq.encode('ascii'), dtype=np.uint8)
    base_positions_cache: Dict[str, np.ndarray] = {}

    ml_idx = 0
    for mod_spec in mm_tag.split(';'):
        if not mod_spec:
            continue
        parts = _mm_mod_spec_parts(mod_spec)
        if parts is None:
            continue
        base_mod, skip_arr = parts

        n_mods = len(skip_arr)

        # Parse "X+y" or "X+y." or "X-y?" into (target_base, mod_code)
        base_and_mod = _mm_base_and_mod_code(base_mod)
        if base_and_mod is None:
            ml_idx += n_mods
            continue
        target_base, mod_code = base_and_mod

        # Target positions in the sequence (cached per base)
        base_positions = _cached_base_positions(
            base_positions_cache, target_base, seq_bytes,
        )

        ml_end = ml_idx + n_mods
        ml_slice = ml_arr_all[ml_idx:ml_end]
        ml_idx = ml_end

        positions, qualities = _mm_valid_positions_and_qualities(
            skip_arr, base_positions, ml_slice, q_len, is_reverse,
        )
        if len(positions) == 0:
            continue

        key = (target_base, mod_code)
        if key in result:
            # Multiple mod specs for same (base, mod_code) — concatenate
            prev_pos, prev_qual = result[key]
            result[key] = (
                np.concatenate([prev_pos, positions]),
                np.concatenate([prev_qual, qualities]),
            )
        else:
            result[key] = (positions, qualities)

    return result


@_numba_njit(cache=True)
def _cigar_walk_numba(cigar_ops, cigar_lens, ref_start, q_len):
    """Numba-JIT CIGAR walker: build query_pos -> ref_pos mapping array.

    Returns int64 array of length q_len where result[q] = ref_pos for
    matched positions and -1 for insertions / soft-clips / query positions
    that aren't aligned to a reference base.

    Matches pysam's get_aligned_pairs(matches_only=False) query_pos→ref_pos
    layout but ~10× faster because it skips Python tuple allocation entirely.

    Op codes (BAM standard):
      0 = M (match), 7 = = (match-eq), 8 = X (match-diff): q and r both advance
      1 = I (insertion),  4 = S (soft-clip):               only q advances
      2 = D (deletion),   3 = N (skip):                    only r advances
      5 = H (hard-clip),  6 = P (pad):                     neither (skip)
    """
    result = np.full(q_len, -1, dtype=np.int64)
    q = 0
    r = ref_start
    for i in range(len(cigar_ops)):
        op = cigar_ops[i]
        length = cigar_lens[i]
        if op == 0 or op == 7 or op == 8:
            for j in range(length):
                if q + j < q_len:
                    result[q + j] = r + j
            q += length
            r += length
        elif op == 1 or op == 4:
            q += length
        elif op == 2 or op == 3:
            r += length
    return result


def cigar_to_query_ref(read) -> np.ndarray:
    """Build query_pos -> ref_pos numpy array for a pysam read.

    Faster replacement for:
        {q: r for q, r in read.get_aligned_pairs(matches_only=False)
         if q is not None and r is not None}

    pysam's get_aligned_pairs allocates ~N Python tuples for an N-base
    read (40 KB+ per 20 kb PacBio read).  This function walks CIGAR tuples
    in a numba-compiled kernel and returns a numpy int64 array indexed
    by query position — ``-1`` means the query position has no reference
    mapping (insertion, soft-clip, unmapped).

    Returns an empty array if cigartuples is None (unmapped read).
    """
    cigar = read.cigartuples
    if cigar is None or not cigar:
        return np.array([], dtype=np.int64)
    ref_start = read.reference_start
    q_len = read.query_length or 0
    if q_len == 0:
        return np.array([], dtype=np.int64)
    cigar_ops = np.asarray([c[0] for c in cigar], dtype=np.int64)
    cigar_lens = np.asarray([c[1] for c in cigar], dtype=np.int64)
    return _cigar_walk_numba(cigar_ops, cigar_lens, int(ref_start), int(q_len))


def parse_mm_tag_query_positions(mm_tag: str, ml_tag,
                                  sequence: str, is_reverse: bool,
                                  prob_threshold: int = 125,
                                  mode: str = 'pacbio-fiber',
                                  debug: bool = False) -> Set[int]:
    """Parse MM/ML tags to extract modification query positions (SEQ frame).

    Produces the same positions as ``pysam.AlignedSegment.modified_bases`` —
    i.e., positions in the stored query_sequence (SEQ) frame of reference,
    accounting for the SAM spec's requirement that MM skip-counts walk in
    the ORIGINAL sequencing direction (= reverse complement of SEQ for
    reverse-aligned reads).

    Validated against pysam ``modified_bases`` on both forward and reverse
    reads (see ``tests/test_mm_parser_vs_pysam.py``).

    MM tag format: ``"A+a.,0,5,3;C+m?,1,2,4;"`` — base+strand+mod_code
    followed by comma-separated skip counts.

    Args:
        mm_tag: MM tag string from BAM.
        ml_tag: ML tag values (bytes, array.array, numpy uint8, or int list).
        sequence: Query sequence in stored SEQ direction (pysam.query_sequence).
        is_reverse: Whether read is reverse-aligned.  Triggers the RC walk.
        prob_threshold: Minimum ML probability (0-255) to call modification.
        mode: 'pacbio-fiber' (A/T both accepted), 'nanopore-fiber' (A only),
              or 'daf' (C/G + deamination products T/A).

    Returns:
        Set of query positions (in SEQ frame) with modification calls at or
        above prob_threshold.
    """
    mod_positions: Set[int] = set()

    if not mm_tag or not ml_tag:
        return mod_positions

    seq_upper = sequence.upper()
    q_len = len(seq_upper)

    # CORRECTNESS: MM walks positions in the ORIGINAL sequencing direction,
    # which equals SEQ for forward-aligned reads and equals the reverse
    # complement of SEQ for reverse-aligned reads.  We do the walk on
    # ``search_seq`` (== ORIGINAL) and flip positions back to SEQ frame at
    # the end via ``q_len - 1 - pos``.
    search_seq = _mm_search_sequence(seq_upper, is_reverse)

    # Convert ml_tag to a numpy uint8 array once.  Accepts raw bytes (fastest
    # IPC format), array.array (what pysam returns), or Python list (legacy
    # callers).  All downstream slicing becomes O(1) numpy views.
    ml_arr_all = _ml_tag_to_uint8_array(ml_tag)
    ml_len_total = len(ml_arr_all)

    if debug and mode == 'daf':
        # Count bases in sequence
        base_counts = {b: seq_upper.count(b) for b in 'ACGT'}
        print(f"  [DAF DEBUG] Seq len={len(sequence)}, bases: A={base_counts['A']} C={base_counts['C']} G={base_counts['G']} T={base_counts['T']}")
        print(f"  [DAF DEBUG] MM tag: {mm_tag[:200]}...")
        print(f"  [DAF DEBUG] ML tag len: {ml_len_total}, first 10 values: {ml_arr_all[:10].tolist()}")
        print(f"  [DAF DEBUG] is_reverse: {is_reverse}, walking on {'RC(SEQ)' if is_reverse else 'SEQ'}")

    ml_idx = 0
    # Pre-compute base position arrays per target base (cached within one call)
    base_pos_cache: Dict[str, np.ndarray] = {}
    search_bytes = np.frombuffer(search_seq.encode('ascii'), dtype=np.uint8)

    for mod_spec in mm_tag.split(';'):
        if not mod_spec:
            continue

        parts = _mm_mod_spec_parts(mod_spec)
        if parts is None:
            continue
        base_mod, skip_arr = parts

        n_mods = len(skip_arr)

        target_base = _mm_target_base(base_mod)
        if target_base is None:
            ml_idx += n_mods
            continue

        if not _target_base_allowed_for_mode(target_base, mode):
            ml_idx += n_mods
            continue

        base_positions = _cached_base_positions(
            base_pos_cache, target_base, search_bytes,
        )

        ml_end = ml_idx + n_mods
        ml_slice_arr = ml_arr_all[ml_idx:ml_end]
        ml_idx = ml_end

        hit_positions = _mm_positions_from_spec(
            skip_arr,
            base_positions,
            ml_slice_arr,
            q_len,
            is_reverse,
            prob_threshold,
        )
        if len(hit_positions) > 0:
            mod_positions.update(hit_positions.tolist())

    return mod_positions


def detect_daf_strand(sequence: str, mod_positions: Set[int]) -> str:
    """
    Detect strand for DAF-seq based on whether deaminated positions are T or A.

    In DAF-seq, deamination converts:
    - + strand: C -> T (so deaminated positions show as T in read)
    - - strand: G -> A (so deaminated positions show as A in read)

    The BAM sequence already contains the converted bases (T or A),
    so we detect strand by counting whether mod positions are mostly T or A.

    Returns '+', '-', or '.' if unclear
    """
    if not mod_positions:
        return '.'

    seq_upper = sequence.upper()
    t_count = 0
    a_count = 0

    for pos in mod_positions:
        if pos < len(seq_upper):
            base = seq_upper[pos]
            if base == 'T':
                t_count += 1
            elif base == 'A':
                a_count += 1

    if t_count > a_count:
        return '+'  # C->T deamination, + strand
    elif a_count > t_count:
        return '-'  # G->A deamination, - strand
    else:
        return '.'


def daf_strand_from_tag(st_tag: Optional[str]) -> str:
    if st_tag is None:
        return '.'
    st = str(st_tag).upper()
    if st == 'CT':
        return '+'
    if st == 'GA':
        return '-'
    return '.'


def has_iupac_encoding(sequence: str) -> bool:
    """
    Check if a BAM sequence contains IUPAC R/Y ambiguity codes.

    DAF-seq collaborators may encode deamination events as R (purine
    ambiguity, marks deaminated G on - strand) or Y (pyrimidine ambiguity,
    marks deaminated C on + strand) directly in the BAM sequence instead of
    using MM/ML tags.

    Args:
        sequence: Query sequence from BAM read

    Returns:
        True if sequence contains R or Y codes
    """
    seq = sequence.upper()
    return 'R' in seq or 'Y' in seq


def extract_daf_iupac_positions(sequence: str, st_tag: Optional[str] = None) -> Tuple[Set[int], str, str]:
    """
    Extract deamination positions from IUPAC R/Y encoded DAF-seq sequence.

    Converts Y→T and R→A to produce the same representation as the legacy
    MM/ML path (where deaminated C shows as T and deaminated G shows as A).
    The existing encoder already reconstructs T→C / A→G internally.

    Strand is determined from the ``st`` tag (``CT`` → ``+``, ``GA`` → ``-``).
    If the tag is absent, strand is inferred by counting R vs Y occurrences.

    Args:
        sequence: Query sequence (may contain R/Y IUPAC codes)
        st_tag: Value of the ``st:Z`` BAM tag, e.g. ``"CT"`` or ``"GA"``

    Returns:
        (mod_positions, strand, converted_sequence) where *converted_sequence*
        has all R replaced with A and all Y replaced with T (pure ACGT).
    """
    seq_upper = sequence.upper()
    mod_positions: Set[int] = set()

    # Determine strand
    strand = daf_strand_from_tag(st_tag)
    if strand == '.' and st_tag is None:
        # Infer from R vs Y counts
        y_count = seq_upper.count('Y')
        r_count = seq_upper.count('R')
        if y_count > r_count:
            strand = '+'
        elif r_count > y_count:
            strand = '-'
        else:
            strand = '.'

    # Vectorized IUPAC conversion: Y (→T) + R (→A) + collect mod positions.
    # 10-30× faster than the per-base Python loop for long reads.
    seq_arr = np.frombuffer(seq_upper.encode('ascii'), dtype=np.uint8)
    y_mask = seq_arr == ord('Y')
    r_mask = seq_arr == ord('R')
    mod_mask = y_mask | r_mask

    if np.any(mod_mask):
        mod_positions = set(np.where(mod_mask)[0].tolist())
        out_arr = seq_arr.copy()
        out_arr[y_mask] = ord('T')
        out_arr[r_mask] = ord('A')
        converted = out_arr.tobytes().decode('ascii')
    else:
        converted = seq_upper

    return mod_positions, strand, converted


def encode_from_query_sequence(sequence: str, mod_positions: Set[int],
                                edge_trim: int = 10,
                                mode: str = 'pacbio-fiber',
                                strand: str = '.',
                                context_size: int = 3,
                                is_reverse: bool = False) -> np.ndarray:
    """
    Encode a read for HMM using context from the query sequence.

    Vectorized implementation for speed.

    Args:
        sequence: Query sequence from BAM read
        mod_positions: Set of query positions with modifications
        edge_trim: Bases to mask at edges (0% modification probability)
        mode: 'pacbio-fiber' for PacBio, 'nanopore-fiber' for Nanopore, 'daf' for DAF-seq
        strand: For daf mode, '+' or '-' (auto-detected if '.')
        context_size: Bases on each side of center (3 = 7-mer hexamer, 5 = 11-mer, etc.)
        is_reverse: Whether this read is reverse-aligned. Critical for
            nanopore-fiber mode: reverse-aligned reads store SAM SEQ as
            the reverse-complement of the basecalled-forward strand, so
            basecalled A's (the m6A target) appear as T's in SEQ. The
            encoder must look at T positions with RC context to recover
            strand-symmetric behavior.

    Returns:
        Encoded observation array for HMM (length = len(sequence))

    Encoding scheme (for context_size k):
        0 to 4^(2k)-1: Modified base with specific context
        4^(2k): Non-target position (0% probability)
        4^(2k)+1 to 2*4^(2k): Unmodified target base with context
        2*4^(2k)+1: Non-target position (unmodified version)
    """
    # Calculate code offsets based on context size
    n_codes = ContextEncoder.get_n_codes(context_size)
    non_target_code = n_codes  # 4^(2k)
    unmethylated_offset = n_codes + 1  # 4^(2k) + 1

    # Determine target base based on mode
    if mode == 'pacbio-fiber':
        return _encode_pacbio_m6a_observations(
            sequence, mod_positions, edge_trim, context_size,
            non_target_code, unmethylated_offset,
        )

    elif mode == 'nanopore-fiber':
        return _encode_nanopore_m6a_observations(
            sequence, mod_positions, edge_trim, context_size,
            is_reverse, non_target_code, unmethylated_offset,
        )

    elif mode == 'daf':
        return _encode_daf_observations(
            sequence, mod_positions, edge_trim, strand, context_size,
            non_target_code, unmethylated_offset,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")


def _sequence_base_int_array(sequence: str, *, uppercase: bool = False,
                             copy: bool = False) -> np.ndarray:
    seq = sequence.upper() if uppercase else sequence
    seq_bytes = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
    seq_int = _BASE_TO_INT[seq_bytes]
    if copy:
        return seq_int.copy()
    return seq_int


def _mod_positions_mask(mod_positions: Set[int], seq_len: int) -> np.ndarray:
    mod_mask = np.zeros(seq_len, dtype=bool)
    if mod_positions:
        mod_arr = np.fromiter(mod_positions, dtype=np.int64,
                              count=len(mod_positions))
        valid_mods = mod_arr[(mod_arr >= 0) & (mod_arr < seq_len)]
        mod_mask[valid_mods] = True
    return mod_mask


def _empty_observation_array(seq_len: int, non_target_code: int,
                             unmethylated_offset: int) -> np.ndarray:
    return np.full(seq_len, non_target_code + unmethylated_offset, dtype=np.int32)


def _apply_m6a_mod_status(context_codes: np.ndarray, mod_mask: np.ndarray,
                          non_target_code: int, unmethylated_offset: int) -> np.ndarray:
    result = _empty_observation_array(len(context_codes), non_target_code,
                                      unmethylated_offset)
    valid_mask = context_codes != non_target_code

    meth_mask = valid_mask & mod_mask
    result[meth_mask] = context_codes[meth_mask]
    unmeth_mask = valid_mask & ~mod_mask
    result[unmeth_mask] = context_codes[unmeth_mask] + unmethylated_offset
    return result


def _encode_pacbio_m6a_observations(sequence: str, mod_positions: Set[int],
                                    edge_trim: int, context_size: int,
                                    non_target_code: int,
                                    unmethylated_offset: int) -> np.ndarray:
    # Encode A positions with forward context and T positions with RC context.
    # The numba path emits final HMM symbols directly and avoids N x k
    # intermediate arrays from the numpy broadcasting fallback.
    seq_len = len(sequence)
    mod_mask = _mod_positions_mask(mod_positions, seq_len)
    if _HAS_NUMBA:
        return _m6a_context_codes_numba(
            _sequence_base_int_array(sequence), mod_mask,
            context_size, edge_trim, non_target_code, unmethylated_offset,
            0, 2, True,
        )

    context_codes = _encode_vectorized(
        sequence, 'A', context_size, edge_trim, non_target_code, True,
    )
    return _apply_m6a_mod_status(context_codes, mod_mask, non_target_code,
                                 unmethylated_offset)


def _nanopore_m6a_targets(is_reverse: bool) -> Tuple[int, int, bool, str]:
    if is_reverse:
        # Basecalled A's are stored as T's in SAM SEQ for reverse-aligned reads.
        return 99, 2, True, 'T'
    return 0, 0, False, 'A'


def _encode_nanopore_m6a_observations(sequence: str, mod_positions: Set[int],
                                      edge_trim: int, context_size: int,
                                      is_reverse: bool, non_target_code: int,
                                      unmethylated_offset: int) -> np.ndarray:
    # Forward reads use A-centered forward context. Reverse reads use T
    # positions with RC context to recover basecalled-forward emission space.
    seq_len = len(sequence)
    mod_mask = _mod_positions_mask(mod_positions, seq_len)
    target_int, rc_target_int, do_rc, target_base = _nanopore_m6a_targets(is_reverse)
    if _HAS_NUMBA:
        return _m6a_context_codes_numba(
            _sequence_base_int_array(sequence), mod_mask,
            context_size, edge_trim, non_target_code, unmethylated_offset,
            target_int, rc_target_int, do_rc,
        )

    context_codes = _encode_vectorized(
        sequence, target_base, context_size, edge_trim, non_target_code, do_rc,
    )
    return _apply_m6a_mod_status(context_codes, mod_mask, non_target_code,
                                 unmethylated_offset)


def _daf_strand_params(sequence: str, mod_positions: Set[int],
                       strand: str) -> Tuple[int, int, bool]:
    if strand == '.':
        strand = detect_daf_strand(sequence, mod_positions)
    if strand == '+':
        return 2, 1, False  # T->C on + strand; use C-centered codes directly.
    return 0, 3, True      # A->G on -/unknown strand; RC to C-centered codes.


def _encode_daf_observations(sequence: str, mod_positions: Set[int],
                             edge_trim: int, strand: str, context_size: int,
                             non_target_code: int,
                             unmethylated_offset: int) -> np.ndarray:
    # DAF modifies sequence bases: + strand C->T, - strand G->A. The model has
    # one C-centered emission table, so - strand G contexts are reverse
    # complemented into equivalent C-centered codes.
    seq_len = len(sequence)
    deam_int, orig_int, use_rc = _daf_strand_params(sequence, mod_positions, strand)
    seq_int = _sequence_base_int_array(sequence, uppercase=True, copy=True)
    mod_mask = _mod_positions_mask(mod_positions, seq_len)

    if _HAS_NUMBA:
        return _daf_context_codes_numba(
            seq_int, mod_mask, context_size, edge_trim,
            non_target_code, unmethylated_offset,
            deam_int, orig_int, use_rc,
        )

    return _encode_daf_vectorized_observations(
        seq_int, mod_mask, context_size, edge_trim,
        non_target_code, unmethylated_offset, deam_int, orig_int, use_rc,
    )


def _context_codes_from_flanks(left_contexts: np.ndarray, right_contexts: np.ndarray,
                               powers: np.ndarray, k: int, use_rc: bool) -> np.ndarray:
    if use_rc:
        # For RC target contexts L[X]R, use RC(R)[target]RC(L).
        left = right_contexts[:, ::-1] ^ 2
        right = left_contexts[:, ::-1] ^ 2
    else:
        left = left_contexts
        right = right_contexts
    return np.sum(left * powers, axis=1) * (4 ** k) + np.sum(right * powers, axis=1)


def _encode_daf_vectorized_observations(seq_int: np.ndarray, mod_mask: np.ndarray,
                                        context_size: int, edge_trim: int,
                                        non_target_code: int,
                                        unmethylated_offset: int,
                                        deam_int: int, orig_int: int,
                                        use_rc: bool) -> np.ndarray:
    seq_len = len(seq_int)

    # Identify target positions in original sequence
    is_deam_base = seq_int == deam_int  # T or A positions
    is_orig_base = seq_int == orig_int  # C or G positions

    # Deaminated: T/A with MM tag (was C/G, now accessible)
    is_deaminated = is_deam_base & mod_mask
    # Non-deaminated: C/G without MM tag (still C/G, footprint)
    is_non_deaminated = is_orig_base & ~mod_mask

    # Reconstruct sequence: replace deaminated positions with original base
    recon_int = seq_int.copy()
    recon_int[is_deaminated] = orig_int  # T->C or A->G

    result = _empty_observation_array(seq_len, non_target_code,
                                      unmethylated_offset)

    k = context_size
    if seq_len < 2 * k + 1:
        return result

    start = max(edge_trim, k)
    end = seq_len - max(edge_trim, k)

    if start >= end:
        return result

    # Target positions (need context codes)
    target_mask = is_deaminated | is_non_deaminated
    positions = np.arange(start, end)
    target_positions = positions[target_mask[start:end]]

    if len(target_positions) == 0:
        return result

    powers = 4 ** np.arange(k - 1, -1, -1, dtype=np.int64)
    left_offsets = np.arange(-k, 0)
    right_offsets = np.arange(1, k + 1)

    left_indices = target_positions[:, np.newaxis] + left_offsets
    right_indices = target_positions[:, np.newaxis] + right_offsets

    left_contexts = recon_int[left_indices]
    right_contexts = recon_int[right_indices]

    # Check for invalid bases (N, etc.)
    valid_context = ~(np.any(left_contexts > 3, axis=1) |
                      np.any(right_contexts > 3, axis=1))

    if not np.any(valid_context):
        return result

    valid_positions = target_positions[valid_context]
    valid_left = left_contexts[valid_context].astype(np.int64)
    valid_right = right_contexts[valid_context].astype(np.int64)

    codes = _context_codes_from_flanks(valid_left, valid_right, powers, k, use_rc)

    # Apply methylated codes to deaminated positions.
    deam_at_valid = is_deaminated[valid_positions]
    result[valid_positions[deam_at_valid]] = codes[deam_at_valid].astype(np.int32)

    # Apply unmethylated codes to non-deaminated positions.
    non_deam_at_valid = is_non_deaminated[valid_positions]
    result[valid_positions[non_deam_at_valid]] = (
        codes[non_deam_at_valid] + unmethylated_offset
    ).astype(np.int32)

    return result


# Pre-computed base-to-int lookup (A=0, C=1, T=2, G=3)
# CRITICAL: This order must match original FiberHMM's hexamer_context() which uses
# bases=['A','C','T','G'] - NOT alphabetical order!
_BASE_TO_INT = np.full(256, 4, dtype=np.int8)
_BASE_TO_INT[ord('A')] = 0
_BASE_TO_INT[ord('C')] = 1
_BASE_TO_INT[ord('T')] = 2  # T=2, not G=2!
_BASE_TO_INT[ord('G')] = 3  # G=3, not T=3!
_BASE_TO_INT[ord('a')] = 0
_BASE_TO_INT[ord('c')] = 1
_BASE_TO_INT[ord('t')] = 2
_BASE_TO_INT[ord('g')] = 3

# Must match _BASE_TO_INT encoding: A=0, C=1, T=2, G=3
_TARGET_BASE_INT = {'A': 0, 'C': 1, 'T': 2, 'G': 3}


@_numba_njit(cache=True)
def _m6a_context_codes_numba(seq_int, mod_mask, k, edge_trim, non_target_code,
                             unmethylated_offset, target_int, rc_target_int, do_rc):
    """Single-pass m6A context encoder with final methylated/unmethylated codes."""
    N = len(seq_int)
    result = np.full(N, non_target_code + unmethylated_offset, dtype=np.int32)

    boundary = k if k > edge_trim else edge_trim
    four_k = 1
    for _ in range(k):
        four_k *= 4

    for i in range(boundary, N - boundary):
        b = int(seq_int[i])
        is_fwd = b == target_int
        is_rc_ = do_rc and b == rc_target_int
        if not (is_fwd or is_rc_):
            continue

        ok = True
        left = 0
        right = 0
        if is_fwd:
            for j in range(i - k, i):
                base = int(seq_int[j])
                if base > 3:
                    ok = False
                    break
                left = left * 4 + base
            if ok:
                for j in range(i + 1, i + k + 1):
                    base = int(seq_int[j])
                    if base > 3:
                        ok = False
                        break
                    right = right * 4 + base
        else:
            for j in range(i + k, i, -1):
                base = int(seq_int[j])
                if base > 3:
                    ok = False
                    break
                left = left * 4 + (base ^ 2)
            if ok:
                for j in range(i - 1, i - k - 1, -1):
                    base = int(seq_int[j])
                    if base > 3:
                        ok = False
                        break
                    right = right * 4 + (base ^ 2)

        if not ok:
            continue

        code = left * four_k + right
        if mod_mask[i]:
            result[i] = code
        else:
            result[i] = code + unmethylated_offset

    return result


@_numba_njit(cache=True)
def _daf_context_codes_numba(seq_int, mod_mask, k, edge_trim, non_target_code,
                             unmethylated_offset, deam_int, orig_int, use_rc):
    """Single-pass DAF context encoder with on-the-fly sequence reconstruction."""
    N = len(seq_int)
    result = np.full(N, non_target_code + unmethylated_offset, dtype=np.int32)

    boundary = k if k > edge_trim else edge_trim
    four_k = 1
    for _ in range(k):
        four_k *= 4

    for i in range(boundary, N - boundary):
        center = int(seq_int[i])
        is_deaminated = center == deam_int and mod_mask[i]
        is_non_deaminated = center == orig_int and not mod_mask[i]
        if not (is_deaminated or is_non_deaminated):
            continue

        ok = True
        left = 0
        right = 0

        if use_rc:
            for j in range(i + k, i, -1):
                base = int(seq_int[j])
                if base == deam_int and mod_mask[j]:
                    base = orig_int
                if base > 3:
                    ok = False
                    break
                left = left * 4 + (base ^ 2)
            if ok:
                for j in range(i - 1, i - k - 1, -1):
                    base = int(seq_int[j])
                    if base == deam_int and mod_mask[j]:
                        base = orig_int
                    if base > 3:
                        ok = False
                        break
                    right = right * 4 + (base ^ 2)
        else:
            for j in range(i - k, i):
                base = int(seq_int[j])
                if base == deam_int and mod_mask[j]:
                    base = orig_int
                if base > 3:
                    ok = False
                    break
                left = left * 4 + base
            if ok:
                for j in range(i + 1, i + k + 1):
                    base = int(seq_int[j])
                    if base == deam_int and mod_mask[j]:
                        base = orig_int
                    if base > 3:
                        ok = False
                        break
                    right = right * 4 + base

        if not ok:
            continue

        code = left * four_k + right
        if is_deaminated:
            result[i] = code
        else:
            result[i] = code + unmethylated_offset

    return result


def _encode_vectorized(sequence: str, target_base: str, context_size: int,
                       edge_trim: int, non_target_code: int,
                       include_rc: bool = False) -> np.ndarray:
    """
    Vectorized context encoding - much faster than position-by-position loop.

    Computes context codes as: left_code * 4^k + right_code
    where left_code and right_code are base-4 numbers from flanking sequences.

    For m6a mode (include_rc=True):
    - A positions: get forward context code (or min with RC for canonical form)
    - T positions: get RC context code (maps T-centered to A-centered codes)
    This matches old FiberHMM behavior where both A and T are target bases.
    """
    seq_len = len(sequence)
    k = context_size

    # Initialize output
    me_encode = np.full(seq_len, non_target_code, dtype=np.int32)

    if seq_len < 2 * k + 1:
        return me_encode

    # Convert sequence to integer array
    seq_bytes = np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)
    seq_int = _BASE_TO_INT[seq_bytes]

    # Target base as integer
    target_int = _TARGET_BASE_INT[target_base]

    # Valid position range (accounting for edge trim AND context size)
    start = max(edge_trim, k)
    end = seq_len - max(edge_trim, k)

    if start >= end:
        return me_encode

    positions = np.arange(start, end)
    powers = 4 ** np.arange(k - 1, -1, -1, dtype=np.int64)

    # Build index arrays for context extraction
    left_offsets = np.arange(-k, 0)
    right_offsets = np.arange(1, k + 1)

    def compute_codes_for_positions(target_pos, use_rc=False):
        """Compute context codes for a set of positions."""
        if len(target_pos) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int64)

        left_indices = target_pos[:, np.newaxis] + left_offsets
        right_indices = target_pos[:, np.newaxis] + right_offsets

        left_contexts = seq_int[left_indices]
        right_contexts = seq_int[right_indices]

        # Check for N's or invalid bases
        valid_mask = ~(np.any(left_contexts > 3, axis=1) |
                       np.any(right_contexts > 3, axis=1))

        if not np.any(valid_mask):
            return np.array([], dtype=np.int32), np.array([], dtype=np.int64)

        valid_left = left_contexts[valid_mask].astype(np.int64)
        valid_right = right_contexts[valid_mask].astype(np.int64)

        codes = _context_codes_from_flanks(valid_left, valid_right, powers, k, use_rc)

        return target_pos[valid_mask], codes

    # Process primary target base (A in m6a mode)
    is_target = seq_int[positions] == target_int
    target_pos = positions[is_target]

    if len(target_pos) > 0:
        # For A positions: use forward context codes
        valid_positions, codes = compute_codes_for_positions(target_pos, use_rc=False)

        if len(valid_positions) > 0:
            me_encode[valid_positions] = codes.astype(np.int32)

    # For m6a mode: also process T positions with reverse complement context
    # T positions represent m6A on the opposite strand - Hia5 methylates both strands
    # The context should be RC to match the A-centered context on the opposite strand
    if include_rc:
        rc_target_int = 2  # T = 2 (using A=0, C=1, T=2, G=3 encoding)
        is_rc_target = seq_int[positions] == rc_target_int
        rc_target_pos = positions[is_rc_target]

        if len(rc_target_pos) > 0:
            # For T positions, use RC encoding to map to A-centered codes
            valid_rc_positions, rc_codes = compute_codes_for_positions(rc_target_pos, use_rc=True)

            if len(valid_rc_positions) > 0:
                me_encode[valid_rc_positions] = rc_codes.astype(np.int32)

    return me_encode


def read_bam(bam_path: str,
             region: Optional[str] = None,
             min_mapq: int = 20,
             prob_threshold: int = 125,
             min_read_length: int = 1000,
             mode: str = 'pacbio-fiber') -> Iterator[FiberRead]:
    """
    Read a PacBio BAM file and yield FiberRead objects.

    Args:
        bam_path: Path to indexed BAM file
        region: Optional region string (e.g., "chr2L:1000-5000")
        min_mapq: Minimum mapping quality
        prob_threshold: Minimum ML probability (0-255) to call modification
        min_read_length: Minimum aligned read length
        mode: 'pacbio-fiber' for fiber-seq, 'daf' for DAF-seq

    Yields:
        FiberRead objects with modification positions and query-to-ref mapping
    """
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        iterator = bam.fetch(region=region) if region else bam.fetch()

        for read in iterator:
            # Skip unmapped, secondary, supplementary
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue

            if read.mapping_quality < min_mapq:
                continue

            if read.query_sequence is None:
                continue

            aligned_length = read.reference_end - read.reference_start
            if aligned_length < min_read_length:
                continue

            # Get MM/ML tags
            # Get modification positions using pysam's built-in API
            mod_query_pos = get_modified_positions_pysam(read, prob_threshold, mode)

            # Fall back to manual parsing if pysam API returns nothing
            if not mod_query_pos:
                mm_tag = get_preferred_tag(read, 'MM', 'Mm')
                ml_tag = get_preferred_tag(read, 'ML', 'Ml')

                # Parse modification positions in query coordinates
                if mm_tag and ml_tag:
                    mod_query_pos = parse_mm_tag_query_positions(
                        mm_tag, ml_tag, read.query_sequence,
                        read.is_reverse, prob_threshold, mode=mode
                    )

            # Build query-to-reference position mapping
            query_to_ref = get_reference_positions(read)

            yield FiberRead(
                read_id=read.query_name,
                chrom=read.reference_name,
                ref_start=read.reference_start,
                ref_end=read.reference_end,
                strand='-' if read.is_reverse else '+',
                query_sequence=read.query_sequence,
                m6a_query_positions=mod_query_pos,
                query_to_ref=query_to_ref,
                is_reverse=read.is_reverse,
            )


def bam_to_chunks(bam_path: str, chunk_size: int = 50000,
                  **kwargs) -> Iterator[List[FiberRead]]:
    """
    Read BAM file and yield chunks of reads for batch processing.
    """
    chunk = []
    for read in read_bam(bam_path, **kwargs):
        chunk.append(read)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []

    if chunk:
        yield chunk


def get_bam_chromosomes(bam_path: str) -> List[str]:
    """Get list of chromosome names from BAM header."""
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        return list(bam.references)


def get_bam_chrom_sizes(bam_path: str) -> Dict[str, int]:
    """Get chromosome sizes from BAM header."""
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        sizes = {}
        for sq in bam.header.get('SQ', []):
            sizes[sq['SN']] = sq['LN']
        return sizes
