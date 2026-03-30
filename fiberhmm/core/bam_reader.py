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

import numpy as np
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple, Dict, Set
import pysam


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
        if op == 0:  # M - match/mismatch
            for _ in range(length):
                ref_positions.append(ref_pos)
                ref_pos += 1
        elif op == 1:  # I - insertion to reference
            for _ in range(length):
                ref_positions.append(None)
        elif op == 2:  # D - deletion from reference
            ref_pos += length
        elif op == 3:  # N - skip (intron)
            ref_pos += length
        elif op == 4:  # S - soft clip
            for _ in range(length):
                ref_positions.append(None)
        elif op == 5:  # H - hard clip
            pass  # Not in query sequence
        elif op == 7:  # = - sequence match
            for _ in range(length):
                ref_positions.append(ref_pos)
                ref_pos += 1
        elif op == 8:  # X - sequence mismatch
            for _ in range(length):
                ref_positions.append(ref_pos)
                ref_pos += 1
    
    return ref_positions


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
            # For m6A mode, look at BOTH A and T positions with m6A
            # A positions = m6A on sequenced strand
            # T positions = m6A on opposite strand (T in read is A on template)
            if mode == 'pacbio-fiber':
                if base not in ('A', 'T') or mod_code not in ('a', 21839):  # 21839 is ChEBI code for m6A
                    continue
            elif mode == 'daf':
                # DAF-seq: deamination converts C->T (+ strand) or G->A (- strand)
                # The BAM sequence has converted bases, but MM tag may use original bases
                # Accept T, A, C, or G with any modification code
                if base not in ('T', 'A', 'C', 'G'):
                    continue
            # For nanopore mode, accept only A with m6A (same as m6a mode)
            elif mode == 'nanopore-fiber':
                if base != 'A' or mod_code not in ('a', 21839):
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


def parse_mm_tag_query_positions(mm_tag: str, ml_tag: List[int], 
                                  sequence: str, is_reverse: bool,
                                  prob_threshold: int = 125,
                                  mode: str = 'pacbio-fiber',
                                  debug: bool = False) -> Set[int]:
    """
    Parse MM/ML tags to extract modification query positions passing probability threshold.
    
    MM tag format: "A+a,0,5,3;C+m,1,2,4"
    - Base type, modification code, then comma-separated skip counts
    
    ML tag: List of probabilities (0-255) for each modification call
    
    Args:
        mm_tag: MM tag string from BAM
        ml_tag: ML tag values
        sequence: Query sequence
        is_reverse: Whether read is reverse strand
        prob_threshold: Minimum probability to call modification
        mode: 'pacbio-fiber' for adenine methylation, 'daf' for deamination
              For 'daf', modifications at C/G positions are used for strand detection
        debug: If True, print diagnostic info
    
    Returns set of query positions with confident modification calls.
    """
    mod_positions = set()
    
    if not mm_tag or not ml_tag:
        return mod_positions
    
    ml_idx = 0
    seq_upper = sequence.upper()
    
    if debug and mode == 'daf':
        # Count bases in sequence
        base_counts = {b: seq_upper.count(b) for b in 'ACGT'}
        print(f"  [DAF DEBUG] Seq len={len(sequence)}, bases: A={base_counts['A']} C={base_counts['C']} G={base_counts['G']} T={base_counts['T']}")
        print(f"  [DAF DEBUG] MM tag: {mm_tag[:200]}...")
        print(f"  [DAF DEBUG] ML tag len: {len(ml_tag)}, first 10 values: {ml_tag[:10]}")
    
    for mod_spec in mm_tag.split(';'):
        if not mod_spec:
            continue
        
        parts = mod_spec.split(',')
        if len(parts) < 2:
            continue
        
        base_mod = parts[0]
        skip_counts = []
        for x in parts[1:]:
            if x.strip():
                try:
                    skip_counts.append(int(x))
                except ValueError:
                    continue
        
        # Determine target base from modification specification
        # Format is typically "X+y" where X is the base
        if len(base_mod) > 0:
            target_base = base_mod[0].upper()
        else:
            ml_idx += len(skip_counts)
            continue
        
        # For m6A mode, process BOTH A and T modifications
        # A = m6A on sequenced strand, T = m6A on opposite strand
        if mode == 'pacbio-fiber':
            if target_base not in ('A', 'T'):
                ml_idx += len(skip_counts)
                continue
        # For nanopore mode, only process A (m6A only at A positions)
        elif mode == 'nanopore-fiber':
            if target_base != 'A':
                ml_idx += len(skip_counts)
                continue
        # For DAF mode, deamination shows as T (from C) or A (from G) in read sequence
        # BUT the MM tag may encode using EITHER:
        # 1. The converted base (T or A) - what we're looking for
        # 2. The original base (C or G) - what some aligners use
        # We need to handle both cases
        elif mode == 'daf':
            # Accept C, G, T, or A - we'll figure out which positions to use below
            if target_base not in ('T', 'A', 'C', 'G'):
                ml_idx += len(skip_counts)
                continue
        
        # Find all target base positions in the query sequence (vectorized)
        seq_bytes = np.frombuffer(seq_upper.encode('ascii'), dtype=np.uint8)
        base_positions = np.where(seq_bytes == ord(target_base))[0]

        n_mods = len(skip_counts)
        if n_mods == 0 or len(base_positions) == 0:
            continue

        # Vectorized skip-count walk: base_indices[i] = sum(skips[0:i+1]) + i
        skip_arr = np.array(skip_counts, dtype=np.int64)
        base_indices = np.cumsum(skip_arr) + np.arange(n_mods)

        # Get ML values for this mod spec
        ml_end = ml_idx + n_mods
        ml_slice = ml_tag[ml_idx:ml_end]
        ml_idx = ml_end

        # Filter: valid index into base_positions AND above threshold
        valid = base_indices < len(base_positions)
        if len(ml_slice) < n_mods:
            valid[len(ml_slice):] = False

        ml_arr = np.array(ml_slice, dtype=np.int64)
        above_thresh = np.zeros(n_mods, dtype=bool)
        above_thresh[:len(ml_arr)] = ml_arr >= prob_threshold

        keep = valid & above_thresh
        if np.any(keep):
            mod_positions.update(base_positions[base_indices[keep]].tolist())
    
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


def encode_from_query_sequence(sequence: str, mod_positions: Set[int],
                                edge_trim: int = 10,
                                mode: str = 'pacbio-fiber',
                                strand: str = '.',
                                context_size: int = 3) -> np.ndarray:
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
        
    Returns:
        Encoded observation array for HMM (length = len(sequence))
        
    Encoding scheme (for context_size k):
        0 to 4^(2k)-1: Modified base with specific context
        4^(2k): Non-target position (0% probability)
        4^(2k)+1 to 2*4^(2k): Unmodified target base with context
        2*4^(2k)+1: Non-target position (unmodified version)
    """
    seq_len = len(sequence)
    
    # Calculate code offsets based on context size
    n_codes = ContextEncoder.get_n_codes(context_size)
    non_target_code = n_codes  # 4^(2k)
    unmethylated_offset = n_codes + 1  # 4^(2k) + 1
    
    # Determine target base based on mode
    if mode == 'pacbio-fiber':
        target_base = 'A'
        include_rc = True
        
        # Get context codes for A positions (and T via RC)
        at_encode = _encode_vectorized(
            sequence, target_base, context_size, edge_trim, 
            non_target_code, include_rc
        )
        
        # Build final encoding (VECTORIZED):
        # - A/T positions: methylated code if in mod_positions, else unmethylated code
        # - C/G positions: non-target code 8193 (neutral)
        
        # Initialize all to non-target unmethylated (8193)
        result = np.full(seq_len, non_target_code + unmethylated_offset, dtype=np.int32)
        
        # Find positions with valid context codes (A/T positions)
        valid_mask = at_encode != non_target_code
        
        # Create modification mask (vectorized)
        mod_mask = np.zeros(seq_len, dtype=bool)
        mod_arr = np.array(list(mod_positions), dtype=np.int64)
        valid_mods = mod_arr[(mod_arr >= 0) & (mod_arr < seq_len)]
        mod_mask[valid_mods] = True
        
        # Apply codes: methylated where modified, unmethylated where not
        # Methylated (accessible): positions with valid context AND modification
        meth_mask = valid_mask & mod_mask
        result[meth_mask] = at_encode[meth_mask]
        
        # Unmethylated (footprint): positions with valid context but NO modification
        unmeth_mask = valid_mask & ~mod_mask
        result[unmeth_mask] = at_encode[unmeth_mask] + unmethylated_offset
        
        return result
        
    elif mode == 'nanopore-fiber':
        # Nanopore fiber-seq: only A-centered (single strand sequencing)
        # No T positions since nanopore only reports A+a modifications
        target_base = 'A'
        include_rc = False  # No RC - nanopore only sequences one strand
        
        # Get context codes for A positions only
        a_encode = _encode_vectorized(
            sequence, target_base, context_size, edge_trim, 
            non_target_code, include_rc
        )
        
        # Build final encoding (VECTORIZED)
        result = np.full(seq_len, non_target_code + unmethylated_offset, dtype=np.int32)
        
        valid_mask = a_encode != non_target_code
        
        mod_mask = np.zeros(seq_len, dtype=bool)
        mod_arr = np.array(list(mod_positions), dtype=np.int64)
        valid_mods = mod_arr[(mod_arr >= 0) & (mod_arr < seq_len)]
        mod_mask[valid_mods] = True
        
        meth_mask = valid_mask & mod_mask
        result[meth_mask] = a_encode[meth_mask]
        
        unmeth_mask = valid_mask & ~mod_mask
        result[unmeth_mask] = a_encode[unmeth_mask] + unmethylated_offset
        
        return result
        
    elif mode == 'daf':
        # DAF-seq: deamination modifies the sequence
        # + strand: C→T deamination, emission probs trained on C-centered hexamers
        # - strand: G→A deamination, MUST apply RC to get C-centered equivalent codes
        #
        # The model has ONE set of emission probs (C-centered). For - strand reads,
        # we apply reverse complement to G-centered contexts to map them to equivalent
        # C-centered codes. This is analogous to how PacBio mode handles A/T.
        
        if strand == '.':
            strand = detect_daf_strand(sequence, mod_positions)
        
        # Determine bases and whether RC is needed
        if strand == '+':
            deam_base = 'T'   # Deaminated C shows as T in read
            orig_base = 'C'   # Target base for emission probs (C-centered)
            deam_int = 2      # T = 2
            orig_int = 1      # C = 1
            use_rc = False    # + strand: use codes directly
        else:
            deam_base = 'A'   # Deaminated G shows as A in read
            orig_base = 'G'   # Target base for context computation (G-centered)
            deam_int = 0      # A = 0
            orig_int = 3      # G = 3
            use_rc = True     # - strand: apply RC to get C-centered codes
        
        # Convert sequence to int array
        seq_bytes = np.frombuffer(sequence.upper().encode('ascii'), dtype=np.uint8)
        seq_int = _BASE_TO_INT[seq_bytes].copy()  # Make a copy for reconstruction
        
        # Create modification mask (vectorized)
        mod_mask = np.zeros(seq_len, dtype=bool)
        mod_arr = np.array(list(mod_positions), dtype=np.int64)
        valid_mods = mod_arr[(mod_arr >= 0) & (mod_arr < seq_len)]
        mod_mask[valid_mods] = True
        
        # Identify target positions in original sequence
        is_deam_base = seq_int == deam_int  # T or A positions
        is_orig_base = seq_int == orig_int  # C or G positions
        
        # Deaminated: T/A with MM tag (was C/G, now accessible)
        is_deaminated = is_deam_base & mod_mask
        # Non-deaminated: C/G without MM tag (still C/G, footprint)
        is_non_deaminated = is_orig_base & ~mod_mask
        
        # Reconstruct sequence: replace deaminated positions with original base
        recon_int = seq_int.copy()
        recon_int[is_deaminated] = orig_int  # T→C or A→G
        
        # Now compute context codes using reconstructed sequence
        # We need codes for positions that are EITHER deaminated OR non-deaminated
        result = np.full(seq_len, non_target_code + unmethylated_offset, dtype=np.int32)
        
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
        
        # Compute context codes vectorized
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
        
        # Compute codes - apply RC for - strand to map G-centered to C-centered
        if use_rc:
            # RC mapping: A(0)<->T(2), C(1)<->G(3) using XOR with 2
            # For G-centered context L[G]R, RC gives RC(R)[C]RC(L)
            rc_left = valid_right[:, ::-1] ^ 2   # RC of right becomes new left
            rc_right = valid_left[:, ::-1] ^ 2   # RC of left becomes new right
            codes = np.sum(rc_left * powers, axis=1) * (4 ** k) + np.sum(rc_right * powers, axis=1)
        else:
            # + strand: use forward codes directly
            codes = np.sum(valid_left * powers, axis=1) * (4 ** k) + np.sum(valid_right * powers, axis=1)
        
        # Apply methylated codes to deaminated positions
        deam_at_valid = is_deaminated[valid_positions]
        result[valid_positions[deam_at_valid]] = codes[deam_at_valid].astype(np.int32)
        
        # Apply unmethylated codes to non-deaminated positions
        non_deam_at_valid = is_non_deaminated[valid_positions]
        result[valid_positions[non_deam_at_valid]] = (codes[non_deam_at_valid] + unmethylated_offset).astype(np.int32)
        
        return result
        
    else:
        raise ValueError(f"Unknown mode: {mode}")


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
        
        if use_rc:
            # For T positions: compute RC to map to A-centered codes
            # RC mapping for A=0, C=1, T=2, G=3: use XOR with 2
            # This gives: A(0)<->T(2), C(1)<->G(3)
            rc_left = valid_right[:, ::-1] ^ 2
            rc_right = valid_left[:, ::-1] ^ 2
            codes = np.sum(rc_left * powers, axis=1) * (4 ** k) + np.sum(rc_right * powers, axis=1)
        else:
            # Forward codes
            codes = np.sum(valid_left * powers, axis=1) * (4 ** k) + np.sum(valid_right * powers, axis=1)
        
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
    with pysam.AlignmentFile(bam_path, "rb") as bam:
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
                try:
                    mm_tag = None
                    ml_tag = None
                    
                    if read.has_tag('MM'):
                        mm_tag = read.get_tag('MM')
                    elif read.has_tag('Mm'):
                        mm_tag = read.get_tag('Mm')
                    
                    if read.has_tag('ML'):
                        ml_tag = list(read.get_tag('ML'))
                    elif read.has_tag('Ml'):
                        ml_tag = list(read.get_tag('Ml'))
                        
                except KeyError:
                    mm_tag = None
                    ml_tag = None
                
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
                query_to_ref=query_to_ref
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
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        return list(bam.references)


def get_bam_chrom_sizes(bam_path: str) -> Dict[str, int]:
    """Get chromosome sizes from BAM header."""
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        sizes = {}
        for sq in bam.header.get('SQ', []):
            sizes[sq['SN']] = sq['LN']
        return sizes
