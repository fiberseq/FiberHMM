#!/usr/bin/env python3
"""Context-counting utilities for probability generation scripts."""

import pickle
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from fiberhmm.probabilities.utils import (
    detect_strand_and_base,
    get_base_name,
    reverse_complement,
    setup_output_dirs,
)

_PROBABILITY_TABLE_COLUMNS = ['context', 'hit', 'nohit', 'ratio', 'encode']


def _reconstruct_deaminated_sequence(
    seq_upper: str,
    mod_positions: Set[int],
    deam_base: str,
    orig_base: str,
) -> str:
    seq_list = list(seq_upper)
    seq_len = len(seq_upper)
    for pos in mod_positions:
        if 0 <= pos < seq_len and seq_list[pos] == deam_base:
            seq_list[pos] = orig_base
    return ''.join(seq_list)


def _position_weight(weights, position: int) -> float:
    return weights[position] if position < len(weights) else 0.0


def _count_ratio(hit, nohit) -> float:
    total = hit + nohit
    return hit / total if total > 0 else 0.0


def _probability_row(context: str, counts) -> dict:
    hit, nohit = counts
    return {
        'context': context,
        'hit': int(hit),
        'nohit': int(nohit),
        'ratio': _count_ratio(hit, nohit),
    }


def _trim_context(full_context: str, trim: int) -> str:
    if trim == 0:
        return full_context
    return full_context[trim:len(full_context) - trim]


def _daf_reconstruction_bases(strand: str) -> Tuple[str, str]:
    if strand == '-':
        return 'A', 'G'
    return 'T', 'C'


def _daf_c_context_from_strand_context(context: str, strand: str) -> str:
    if strand == '-':
        return reverse_complement(context)
    return context


def _context_to_code(contexts) -> Dict[str, int]:
    return {ctx: i for i, ctx in enumerate(sorted(contexts))}


def _missing_probability_rows(all_contexts, existing, context_to_code: dict) -> list[dict]:
    return [
        {
            'context': ctx,
            'hit': 0,
            'nohit': 0,
            'ratio': 0.0,
            'encode': context_to_code[ctx],
        }
        for ctx in all_contexts
        if ctx not in existing
    ]


def _aggregate_context_counts(counts, max_context: int, context_size: int) -> dict:
    if context_size > max_context:
        raise ValueError(f"Requested context size {context_size} > max {max_context}")

    aggregated = defaultdict(lambda: [0, 0])
    trim = max_context - context_size
    for full_context, context_counts in counts.items():
        small_context = _trim_context(full_context, trim)
        aggregated[small_context][0] += context_counts[0]
        aggregated[small_context][1] += context_counts[1]
    return dict(aggregated)


def _probability_dataframe_from_counts(aggregated: dict) -> pd.DataFrame:
    rows = [
        _probability_row(context, counts)
        for context, counts in sorted(aggregated.items())
    ]
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values('context').reset_index(drop=True)
        df['encode'] = range(len(df))
    return df


def _empty_probability_table() -> pd.DataFrame:
    return pd.DataFrame(columns=_PROBABILITY_TABLE_COLUMNS)


def _encode_probability_table(
    probs: pd.DataFrame,
    contexts,
) -> Tuple[Dict[str, int], pd.DataFrame]:
    context_to_code = _context_to_code(contexts)
    probs = probs.copy()
    probs['encode'] = probs['context'].map(context_to_code)
    probs = probs.sort_values('encode').reset_index(drop=True)
    return context_to_code, probs


def _probability_table_with_missing_contexts(
    probs: pd.DataFrame,
    all_contexts,
    context_to_code: dict,
) -> pd.DataFrame:
    existing = set(probs['context'])
    missing = _missing_probability_rows(
        all_contexts, existing, context_to_code,
    )
    if missing:
        probs = pd.concat([probs, pd.DataFrame(missing)], ignore_index=True)
    return probs.sort_values('encode').reset_index(drop=True)


class ContextCounter:
    """
    Counts modification hits/misses per sequence context.
    Stores counts at maximum resolution, can aggregate for smaller contexts.

    This is the primary class for counting methylation events by hexamer/k-mer context
    used in probability estimation for HMM emission parameters.
    """

    def __init__(self, max_context: int = 10, center_base: str = 'A'):
        """
        Args:
            max_context: Number of bases on each side of center (10 = 21-mer)
            center_base: The base at center position ('A', 'T', 'C', or 'G')
        """
        self.max_context = max_context
        self.center_base = center_base.upper()

        # Store counts as {context_string: [hit_count, nohit_count]}
        # Context string is the full (2*max_context + 1)-mer
        self.counts: Dict[str, list] = defaultdict(lambda: [0, 0])

        # Track total positions processed
        self.total_positions = 0
        self.total_modified = 0

    def _context_from_upper(
        self,
        seq_upper: str,
        position: int,
        center_base: Optional[str] = None,
    ) -> Optional[str]:
        seq_len = len(seq_upper)
        target_base = self.center_base if center_base is None else center_base

        if position < self.max_context or position >= seq_len - self.max_context:
            return None
        if seq_upper[position] != target_base:
            return None

        context = seq_upper[position - self.max_context : position + self.max_context + 1]
        if len(context) != 2 * self.max_context + 1:
            return None
        if any(b not in 'ACGT' for b in context):
            return None
        return context

    def _iter_contexts(
        self,
        seq_upper: str,
        region_start: int,
        region_end: int,
        edge_trim: int,
        center_base: Optional[str] = None,
    ) -> Iterator[Tuple[int, str]]:
        seq_len = len(seq_upper)
        for i in range(max(region_start, edge_trim), min(region_end, seq_len - edge_trim)):
            context = self._context_from_upper(seq_upper, i, center_base)
            if context is not None:
                yield i, context

    def _record_context(self, context: str, is_modified: bool, weight=1):
        if is_modified:
            self.counts[context][0] += weight
            self.total_modified += weight
        else:
            self.counts[context][1] += weight
        self.total_positions += weight

    def _record_contexts_in_region(
        self,
        seq_upper: str,
        mod_positions: Set[int],
        region_start: int,
        region_end: int,
        edge_trim: int,
        weight_at=None,
    ) -> None:
        for i, context in self._iter_contexts(
            seq_upper,
            region_start,
            region_end,
            edge_trim,
        ):
            weight = 1 if weight_at is None else weight_at(i)
            self._record_context(context, i in mod_positions, weight)

    def add_position(self, sequence: str, position: int, is_modified: bool):
        """
        Add a single position observation.

        Args:
            sequence: Full read sequence
            position: Position of the base in the sequence
            is_modified: Whether this position was called as modified
        """
        context = self._context_from_upper(sequence.upper(), position)
        if context is not None:
            self._record_context(context, is_modified)

    def process_read(self, sequence: str, mod_positions: Set[int], edge_trim: int = 10):
        """
        Process all target base positions in a read.

        Args:
            sequence: Read sequence
            mod_positions: Set of positions that are modified
            edge_trim: Bases to skip at edges
        """
        seq_upper = sequence.upper()
        self._record_contexts_in_region(
            seq_upper, mod_positions, 0, len(sequence), edge_trim,
        )

    def process_read_daf(self, sequence: str, mod_positions: Set[int],
                         strand: str, edge_trim: int = 10):
        """
        Process a DAF-seq read with proper sequence reconstruction.

        All contexts are stored as C-centered. G-strand reads are reverse complemented
        to produce equivalent C-centered contexts, effectively doubling the data.

        DAF-seq encodes C>T (+ strand) or G>A (- strand) deaminations using m6A-style MM tags.
        The MM tag marks the CONVERTED base (T or A), not the original (C or G).

        For + strand (C→T deamination):
            - T positions in mod_positions were originally C, got deaminated (accessible)
            - C positions not in mod_positions stayed as C (protected)
            - Contexts stored directly as C-centered

        For - strand (G→A deamination):
            - A positions in mod_positions were originally G, got deaminated (accessible)
            - G positions not in mod_positions stayed as G (protected)
            - Contexts are reverse complemented to C-centered equivalents

        Args:
            sequence: Read sequence (with converted bases)
            mod_positions: Set of positions marked by MM tag (T or A that were deaminated)
            strand: '+' for C→T, '-' for G→A, '.' defaults to + strand
            edge_trim: Bases to skip at edges
        """
        # This counter must be C-centered for DAF mode
        if self.center_base != 'C':
            return

        seq_upper = sequence.upper()
        seq_len = len(sequence)
        deam_base, orig_base = _daf_reconstruction_bases(strand)

        reconstructed = _reconstruct_deaminated_sequence(
            seq_upper, mod_positions, deam_base, orig_base,
        )
        for i, context in self._iter_contexts(
            reconstructed,
            0,
            seq_len,
            edge_trim,
            center_base=orig_base,
        ):
            c_context = _daf_c_context_from_strand_context(context, strand)
            self._record_context(c_context, i in mod_positions)

    def add_region(self, sequence: str, mod_positions: Set[int],
                   region_start: int, region_end: int, edge_trim: int = 10):
        """
        Add observations from a specific region of a read.

        Used by bootstrap_probs.py to count within specific footprint/MSP regions.

        Args:
            sequence: Full read sequence
            mod_positions: Set of methylated positions (query coords)
            region_start: Start of region to count (query coords)
            region_end: End of region to count (query coords)
            edge_trim: Skip positions within this many bases of read edges
        """
        seq_upper = sequence.upper()
        self._record_contexts_in_region(
            seq_upper, mod_positions, region_start, region_end, edge_trim,
        )

    def get_probabilities(self, context_size: int = 3) -> pd.DataFrame:
        """
        Get probability table for a specific context size.

        Args:
            context_size: Bases on each side (3 = 7-mer hexamer)

        Returns:
            DataFrame with columns: context, hit, nohit, ratio
        """
        aggregated = _aggregate_context_counts(
            self.counts,
            self.max_context,
            context_size,
        )
        return _probability_dataframe_from_counts(aggregated)

    def get_encoding_table(self, context_size: int = 3, fill_missing: bool = False) -> Tuple[Dict[str, int], pd.DataFrame]:
        """
        Get encoding lookup table and probabilities for a context size.

        Args:
            context_size: Bases on each side (3 = 7-mer)
            fill_missing: If True, include all possible contexts (SLOW for large k!)
                         If False, only include observed contexts (fast)

        Returns:
            (context_to_code dict, probability DataFrame with 'encode' column)
        """
        probs = self.get_probabilities(context_size)

        # Handle empty DataFrame
        if len(probs) == 0:
            return {}, _empty_probability_table()

        if fill_missing and context_size <= 5:  # Only allow fill_missing for small k
            # Build deterministic encoding (alphabetical order of ALL possible)
            all_contexts = self._generate_all_contexts(context_size)
            context_to_code, probs = _encode_probability_table(
                probs, all_contexts,
            )
            probs = _probability_table_with_missing_contexts(
                probs, all_contexts, context_to_code,
            )
        else:
            # Fast path: only observed contexts, encode alphabetically
            context_to_code, probs = _encode_probability_table(
                probs, probs['context'].unique(),
            )

        return context_to_code, probs

    def _generate_all_contexts(self, context_size: int) -> List[str]:
        """Generate all possible context strings for a given size.

        WARNING: This is O(4^(2k+1)) - only use for small k (<=5)!
        k=3: 4,096 contexts
        k=5: 262,144 contexts
        k=7: 16,777,216 contexts (slow)
        k=10: 1,099,511,627,776 contexts (impossible)
        """
        if context_size > 5:
            raise ValueError(f"Cannot generate all contexts for k={context_size} - too many! "
                           f"Use fill_missing=False or k<=5")

        from itertools import product
        bases = 'ACGT'
        length = 2 * context_size + 1
        center_pos = context_size

        contexts = []
        for combo in product(bases, repeat=length):
            if combo[center_pos] == self.center_base:
                contexts.append(''.join(combo))

        return contexts

    def save(self, filepath: str):
        """Save counter state to file."""
        data = {
            'max_context': self.max_context,
            'center_base': self.center_base,
            'counts': dict(self.counts),
            'total_positions': self.total_positions,
            'total_modified': self.total_modified
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: str) -> 'ContextCounter':
        """Load counter state from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        counter = cls(data['max_context'], data['center_base'])
        counter.counts = defaultdict(lambda: [0, 0], data['counts'])
        counter.total_positions = data['total_positions']
        counter.total_modified = data['total_modified']
        return counter

    def merge(self, other: 'ContextCounter'):
        """Merge counts from another counter."""
        if self.max_context != other.max_context:
            raise ValueError("Cannot merge counters with different max_context")
        if self.center_base != other.center_base:
            raise ValueError("Cannot merge counters with different center_base")

        for context, counts in other.counts.items():
            self.counts[context][0] += counts[0]
            self.counts[context][1] += counts[1]

        self.total_positions += other.total_positions
        self.total_modified += other.total_modified

    def add_weighted_position(self, sequence: str, position: int,
                               is_modified: bool, weight: float):
        """
        Add a weighted position observation (for posterior-weighted counting).

        Args:
            sequence: Full read sequence
            position: Position of the base in the sequence
            is_modified: Whether this position was called as modified
            weight: Weight to apply (e.g., posterior probability for this state)
        """
        context = self._context_from_upper(sequence.upper(), position)
        if context is not None:
            self._record_context(context, is_modified, weight)

    def add_weighted_region(self, sequence: str, mod_positions: Set[int],
                            region_start: int, region_end: int,
                            weights: np.ndarray, edge_trim: int = 10):
        """
        Add weighted observations from a specific region of a read.

        Used for posterior-weighted counting in bootstrap_probs.py.

        Args:
            sequence: Full read sequence
            mod_positions: Set of methylated positions (query coords)
            region_start: Start of region to count (query coords)
            region_end: End of region to count (query coords)
            weights: Array of weights (posterior probabilities) for each position
            edge_trim: Skip positions within this many bases of read edges
        """
        seq_upper = sequence.upper()
        self._record_contexts_in_region(
            seq_upper,
            mod_positions,
            region_start,
            region_end,
            edge_trim,
            weight_at=lambda i: _position_weight(weights, i),
        )

    def reset(self):
        """Reset all counts to zero."""
        self.counts = defaultdict(lambda: [0, 0])
        self.total_positions = 0
        self.total_modified = 0

    def copy(self) -> 'ContextCounter':
        """Create a deep copy of this counter."""
        new_counter = ContextCounter(self.max_context, self.center_base)
        for context, counts in self.counts.items():
            new_counter.counts[context] = [counts[0], counts[1]]
        new_counter.total_positions = self.total_positions
        new_counter.total_modified = self.total_modified
        return new_counter
