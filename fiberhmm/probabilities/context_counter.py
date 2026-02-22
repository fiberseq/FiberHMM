#!/usr/bin/env python3
"""
FiberHMM prob_utils.py
Shared utilities for probability generation scripts (generate_probs.py, bootstrap_probs.py, transfer_probs.py).

Contains:
- ContextCounter: Counts modification events by sequence context
- detect_strand_and_base: Detects strand and target base based on mode
- setup_output_dirs: Creates standard output directory structure
"""

import os
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple, List, Optional


# Reverse complement lookup
_RC_TABLE = str.maketrans('ACGT', 'TGCA')


def reverse_complement(seq: str) -> str:
    """Return reverse complement of a DNA sequence."""
    return seq.translate(_RC_TABLE)[::-1]


def detect_strand_and_base(sequence: str, mod_positions: Set[int], mode: str) -> Tuple[str, str]:
    """
    Detect strand and target base based on mode.

    Args:
        sequence: Read sequence
        mod_positions: Set of query positions with modifications
        mode: Analysis mode ('pacbio-fiber', 'nanopore-fiber', 'daf')

    Returns:
        (strand, target_base)

    For pacbio-fiber and nanopore-fiber modes, returns ('.', 'A') - A-centered, no strand.
    For daf mode, detects strand by whether modifications are at T or A positions:
        - + strand: C→T deamination, MM tag marks T positions → target_base 'C'
        - - strand: G→A deamination, MM tag marks A positions → target_base 'G'
    """
    if mode in ('pacbio-fiber', 'nanopore-fiber'):
        return '.', 'A'

    seq_upper = sequence.upper()

    if mode == 'daf':
        # DAF-seq encodes C>T or G>A deaminations using m6A-style MM tags
        # The MM tag marks the CONVERTED base position (T or A), not the original (C or G)
        # + strand: C→T deamination, MM marks T positions
        # - strand: G→A deamination, MM marks A positions
        t_count = sum(1 for p in mod_positions if p < len(seq_upper) and seq_upper[p] == 'T')
        a_count = sum(1 for p in mod_positions if p < len(seq_upper) and seq_upper[p] == 'A')

        if t_count > a_count:
            return '+', 'C'  # C→T deamination, C-centered contexts
        elif a_count > t_count:
            return '-', 'G'  # G→A deamination, G-centered contexts
        else:
            return '.', 'C'  # Default to C-centered

    return '.', 'A'


def setup_output_dirs(output_path: str) -> Tuple[Path, Path, Path]:
    """
    Create standard output directory structure (tables/, plots/).

    Args:
        output_path: Base output directory path

    Returns:
        (output_dir, tables_dir, plots_dir) as Path objects
    """
    output_dir = Path(output_path)
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    return output_dir, tables_dir, plots_dir


def get_base_name(output_path: str, default: str = "probs") -> str:
    """
    Extract base name from output path for file naming.

    Args:
        output_path: Output directory path
        default: Default name if path is empty

    Returns:
        Base name string for output files
    """
    base_name = os.path.basename(output_path.rstrip('/'))
    return base_name if base_name else default


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

    def add_position(self, sequence: str, position: int, is_modified: bool):
        """
        Add a single position observation.

        Args:
            sequence: Full read sequence
            position: Position of the base in the sequence
            is_modified: Whether this position was called as modified
        """
        seq_len = len(sequence)

        # Check bounds
        if position < self.max_context or position >= seq_len - self.max_context:
            return

        # Check center base matches
        if sequence[position].upper() != self.center_base:
            return

        # Extract context
        start = position - self.max_context
        end = position + self.max_context + 1
        context = sequence[start:end].upper()

        # Skip if contains N or invalid bases
        if len(context) != 2 * self.max_context + 1:
            return
        if any(b not in 'ACGT' for b in context):
            return

        # Update counts
        if is_modified:
            self.counts[context][0] += 1
            self.total_modified += 1
        else:
            self.counts[context][1] += 1

        self.total_positions += 1

    def process_read(self, sequence: str, mod_positions: Set[int], edge_trim: int = 10):
        """
        Process all target base positions in a read.

        Args:
            sequence: Read sequence
            mod_positions: Set of positions that are modified
            edge_trim: Bases to skip at edges
        """
        seq_upper = sequence.upper()
        seq_len = len(sequence)

        for i in range(edge_trim, seq_len - edge_trim):
            if seq_upper[i] == self.center_base:
                is_mod = i in mod_positions
                self.add_position(sequence, i, is_mod)

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

        if strand == '-':
            # G→A deamination: process G positions and RC to C-centered
            deam_base = 'A'
            orig_base = 'G'

            # Reconstruct original sequence (replace A back to G at deaminated positions)
            seq_list = list(seq_upper)
            for pos in mod_positions:
                if 0 <= pos < seq_len and seq_list[pos] == deam_base:
                    seq_list[pos] = orig_base
            reconstructed = ''.join(seq_list)

            # Process G positions and convert contexts to C-centered via RC
            for i in range(edge_trim, seq_len - edge_trim):
                if reconstructed[i] == orig_base:
                    # Check context bounds
                    if i < self.max_context or i >= seq_len - self.max_context:
                        continue

                    # Extract G-centered context
                    start = i - self.max_context
                    end = i + self.max_context + 1
                    g_context = reconstructed[start:end]

                    # Skip if contains N or invalid bases
                    if len(g_context) != 2 * self.max_context + 1:
                        continue
                    if any(b not in 'ACGT' for b in g_context):
                        continue

                    # Reverse complement to get C-centered equivalent
                    c_context = reverse_complement(g_context)

                    # Was this position deaminated?
                    is_deaminated = i in mod_positions

                    # Update counts directly (bypass add_position which would check center base)
                    if is_deaminated:
                        self.counts[c_context][0] += 1
                        self.total_modified += 1
                    else:
                        self.counts[c_context][1] += 1
                    self.total_positions += 1
        else:
            # + strand (default): C→T deamination, process directly as C-centered
            deam_base = 'T'
            orig_base = 'C'

            # Reconstruct original sequence (replace T back to C at deaminated positions)
            seq_list = list(seq_upper)
            for pos in mod_positions:
                if 0 <= pos < seq_len and seq_list[pos] == deam_base:
                    seq_list[pos] = orig_base
            reconstructed = ''.join(seq_list)

            # Process C positions directly
            for i in range(edge_trim, seq_len - edge_trim):
                if reconstructed[i] == orig_base:
                    is_deaminated = i in mod_positions
                    self.add_position(reconstructed, i, is_deaminated)

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
        seq_len = len(sequence)
        seq_upper = sequence.upper()

        for i in range(max(region_start, edge_trim),
                       min(region_end, seq_len - edge_trim)):
            if seq_upper[i] != self.center_base:
                continue

            # Check context bounds
            if i < self.max_context or i >= seq_len - self.max_context:
                continue

            # Extract context
            context = seq_upper[i - self.max_context : i + self.max_context + 1]

            if len(context) != 2 * self.max_context + 1:
                continue
            if any(b not in 'ACGT' for b in context):
                continue

            # Record observation
            is_mod = i in mod_positions
            if is_mod:
                self.counts[context][0] += 1
                self.total_modified += 1
            else:
                self.counts[context][1] += 1

            self.total_positions += 1

    def get_probabilities(self, context_size: int = 3) -> pd.DataFrame:
        """
        Get probability table for a specific context size.

        Args:
            context_size: Bases on each side (3 = 7-mer hexamer)

        Returns:
            DataFrame with columns: context, hit, nohit, ratio
        """
        if context_size > self.max_context:
            raise ValueError(f"Requested context size {context_size} > max {self.max_context}")

        # Aggregate counts for smaller context
        aggregated = defaultdict(lambda: [0, 0])

        trim = self.max_context - context_size

        for full_context, counts in self.counts.items():
            # Extract the smaller context from center
            small_context = full_context[trim:len(full_context) - trim]
            aggregated[small_context][0] += counts[0]
            aggregated[small_context][1] += counts[1]

        # Build DataFrame
        rows = []
        for context, counts in sorted(aggregated.items()):
            hit, nohit = counts
            total = hit + nohit
            ratio = hit / total if total > 0 else 0.0
            rows.append({
                'context': context,
                'hit': int(hit),
                'nohit': int(nohit),
                'ratio': ratio
            })

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values('context').reset_index(drop=True)
            df['encode'] = range(len(df))

        return df

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
            return {}, pd.DataFrame(columns=['context', 'hit', 'nohit', 'ratio', 'encode'])

        if fill_missing and context_size <= 5:  # Only allow fill_missing for small k
            # Build deterministic encoding (alphabetical order of ALL possible)
            all_contexts = self._generate_all_contexts(context_size)
            context_to_code = {ctx: i for i, ctx in enumerate(sorted(all_contexts))}

            # Add encoding to probability table
            probs['encode'] = probs['context'].map(context_to_code)

            # Fill missing contexts with zeros
            existing = set(probs['context'])
            missing = []
            for ctx in all_contexts:
                if ctx not in existing:
                    missing.append({
                        'context': ctx,
                        'hit': 0,
                        'nohit': 0,
                        'ratio': 0.0,
                        'encode': context_to_code[ctx]
                    })

            if missing:
                probs = pd.concat([probs, pd.DataFrame(missing)], ignore_index=True)

            probs = probs.sort_values('encode').reset_index(drop=True)
        else:
            # Fast path: only observed contexts, encode alphabetically
            observed_contexts = sorted(probs['context'].unique())
            context_to_code = {ctx: i for i, ctx in enumerate(observed_contexts)}
            probs['encode'] = probs['context'].map(context_to_code)
            probs = probs.sort_values('encode').reset_index(drop=True)

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
        seq_len = len(sequence)

        # Check bounds
        if position < self.max_context or position >= seq_len - self.max_context:
            return

        # Check center base matches
        if sequence[position].upper() != self.center_base:
            return

        # Extract context
        start = position - self.max_context
        end = position + self.max_context + 1
        context = sequence[start:end].upper()

        # Skip if contains N or invalid bases
        if len(context) != 2 * self.max_context + 1:
            return
        if any(b not in 'ACGT' for b in context):
            return

        # Update counts with weight (counts stored as floats)
        if is_modified:
            self.counts[context][0] += weight
            self.total_modified += weight
        else:
            self.counts[context][1] += weight

        self.total_positions += weight

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
        seq_len = len(sequence)
        seq_upper = sequence.upper()

        for i in range(max(region_start, edge_trim),
                       min(region_end, seq_len - edge_trim)):
            if seq_upper[i] != self.center_base:
                continue

            # Check context bounds
            if i < self.max_context or i >= seq_len - self.max_context:
                continue

            # Extract context
            context = seq_upper[i - self.max_context : i + self.max_context + 1]

            if len(context) != 2 * self.max_context + 1:
                continue
            if any(b not in 'ACGT' for b in context):
                continue

            # Get weight for this position
            weight = weights[i] if i < len(weights) else 0.0

            # Record weighted observation
            is_mod = i in mod_positions
            if is_mod:
                self.counts[context][0] += weight
                self.total_modified += weight
            else:
                self.counts[context][1] += weight

            self.total_positions += weight

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
