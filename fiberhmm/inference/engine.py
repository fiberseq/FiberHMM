"""FiberHMM core per-read HMM inference engine."""

import numpy as np
import pysam
from typing import Optional, Tuple, Set

from fiberhmm.core.hmm import FiberHMM
from fiberhmm.core.bam_reader import encode_from_query_sequence, detect_daf_strand


def predict_footprints(model: FiberHMM, encoded_read: np.ndarray,
                       with_scores: bool = False) -> Tuple[np.ndarray, np.ndarray, int, Optional[np.ndarray]]:
    """
    Run HMM Viterbi prediction to call footprints.

    Args:
        model: Trained FiberHMM model
        encoded_read: Encoded observation sequence
        with_scores: If True, compute posterior probability scores per footprint

    Returns:
        (starts, sizes, count, scores) - footprint positions in read coordinates
        scores is None if with_scores=False, otherwise array of mean posteriors per footprint
    """
    if len(encoded_read) == 0:
        return np.array([]), np.array([]), 0, None

    # Predict states (0 = footprint, 1 = accessible)
    if with_scores:
        states, confidence = model.predict_with_confidence(encoded_read)
    else:
        states = model.predict(encoded_read)
        confidence = None

    # Count footprint positions (state 0)
    if np.sum(states == 0) == 0:
        return np.array([]), np.array([]), 0, None

    # Find transitions (pad with 1s = accessible at edges)
    # State 0 = footprint, State 1 = accessible
    states_padded = np.concatenate([[1], states, [1]])
    diff = np.diff(states_padded)

    # Footprint starts where we transition from 1 (accessible) to 0 (footprint): diff == -1
    # Footprint ends where we transition from 0 (footprint) to 1 (accessible): diff == 1
    starts = np.where(diff == -1)[0]
    ends = np.where(diff == 1)[0]

    if len(starts) == 0:
        return np.array([]), np.array([]), 0, None

    sizes = ends - starts

    # Compute per-footprint scores
    scores = None
    if with_scores and confidence is not None:
        scores = np.zeros(len(starts), dtype=np.float32)
        for i, (s, e) in enumerate(zip(starts, ends)):
            # Mean posterior probability for footprint state within this footprint
            scores[i] = np.mean(confidence[s:e])

    return starts, sizes, len(starts), scores


def _extract_footprints_from_states(states: np.ndarray, confidence: Optional[np.ndarray],
                                     msp_min_size: int, with_scores: bool) -> dict:
    """
    Extract footprints and MSPs from HMM states (without running HMM again).

    States: 0 = footprint, 1 = accessible

    Used for timing breakdown to separate HMM time from post-processing.
    """
    result = {
        'footprint_starts': np.array([], dtype=np.int32),
        'footprint_sizes': np.array([], dtype=np.int32),
        'footprint_scores': None,
        'msp_starts': np.array([], dtype=np.int32),
        'msp_sizes': np.array([], dtype=np.int32),
        'msp_scores': None,
    }

    if len(states) == 0:
        return result

    # Find footprints (state 0 regions)
    # Pad with 1 (accessible) at edges
    states_padded = np.concatenate([[1], states, [1]])
    diff = np.diff(states_padded)

    # Footprint starts: transition from 1 to 0 (diff == -1)
    # Footprint ends: transition from 0 to 1 (diff == 1)
    fp_starts = np.where(diff == -1)[0]
    fp_ends = np.where(diff == 1)[0]

    if len(fp_starts) > 0:
        result['footprint_starts'] = fp_starts.astype(np.int32)
        result['footprint_sizes'] = (fp_ends - fp_starts).astype(np.int32)

        if with_scores and confidence is not None:
            fp_scores = np.zeros(len(fp_starts), dtype=np.float32)
            for i, (s, e) in enumerate(zip(fp_starts, fp_ends)):
                fp_scores[i] = np.mean(confidence[s:e])
            result['footprint_scores'] = fp_scores

    # Find MSPs (accessible regions, state 1)
    # Pad with 0 (footprint) at edges
    acc_padded = np.concatenate([[0], states, [0]])
    acc_diff = np.diff(acc_padded)

    # MSP starts: transition from 0 to 1 (diff == 1)
    # MSP ends: transition from 1 to 0 (diff == -1)
    msp_starts = np.where(acc_diff == 1)[0]
    msp_ends = np.where(acc_diff == -1)[0]

    if len(msp_starts) > 0:
        msp_sizes = msp_ends - msp_starts
        size_mask = msp_sizes >= msp_min_size
        msp_starts = msp_starts[size_mask]
        msp_sizes = msp_sizes[size_mask]

        if len(msp_starts) > 0:
            result['msp_starts'] = msp_starts.astype(np.int32)
            result['msp_sizes'] = msp_sizes.astype(np.int32)

            if with_scores and confidence is not None:
                msp_scores = np.zeros(len(msp_starts), dtype=np.float32)
                for i, (s, sz) in enumerate(zip(msp_starts, msp_sizes)):
                    msp_scores[i] = np.mean(1.0 - confidence[s:s+sz])
                result['msp_scores'] = msp_scores

    return result


def predict_footprints_and_msps(model: FiberHMM, encoded_read: np.ndarray,
                                 msp_min_size: int = 147,
                                 with_scores: bool = False,
                                 return_posteriors: bool = False) -> dict:
    """
    Run HMM prediction to call both footprints (ns/nl) and MSPs (as/al).

    States: 0 = footprint, 1 = accessible

    MSPs (Methylase-Sensitive Patches) are accessible regions that span across
    small interrupting footprints. This prevents segmentation of large accessible
    regions by small footprints.

    Args:
        model: Trained FiberHMM model
        encoded_read: Encoded observation sequence
        msp_min_size: Minimum size for an accessible region to be called as MSP
        with_scores: If True, compute confidence scores
        return_posteriors: If True, return full posterior array for CNN training

    Returns:
        dict with:
            'footprint_starts': query positions where footprints start
            'footprint_sizes': footprint lengths
            'footprint_scores': per-footprint confidence (if with_scores)
            'msp_starts': query positions where MSPs start
            'msp_sizes': MSP lengths
            'msp_scores': per-MSP confidence (if with_scores)
            'states': raw HMM state array
            'posteriors': P(footprint) per position (if return_posteriors)
    """
    result = {
        'footprint_starts': np.array([], dtype=np.int32),
        'footprint_sizes': np.array([], dtype=np.int32),
        'footprint_scores': None,
        'msp_starts': np.array([], dtype=np.int32),
        'msp_sizes': np.array([], dtype=np.int32),
        'msp_scores': None,
        'states': np.array([], dtype=np.int8),
        'posteriors': None,
    }

    if len(encoded_read) == 0:
        return result

    # Predict states (0 = footprint, 1 = accessible)
    # Use predict_with_posteriors if we need posteriors or scores (shares computation)
    if with_scores or return_posteriors:
        states, posteriors_full = model.predict_with_posteriors(encoded_read)
        confidence = posteriors_full[np.arange(len(states)), states]

        if return_posteriors:
            # P(footprint) = posteriors_full[:, 0]
            result['posteriors'] = posteriors_full[:, 0].astype(np.float16)
    else:
        states = model.predict(encoded_read)
        confidence = None

    result['states'] = states

    # Find footprints (state 0 regions)
    # Pad with 1 (accessible) at edges
    states_padded = np.concatenate([[1], states, [1]])
    diff = np.diff(states_padded)

    # Footprint starts: transition from 1 to 0 (diff == -1)
    # Footprint ends: transition from 0 to 1 (diff == 1)
    fp_starts = np.where(diff == -1)[0]
    fp_ends = np.where(diff == 1)[0]

    if len(fp_starts) > 0:
        result['footprint_starts'] = fp_starts.astype(np.int32)
        result['footprint_sizes'] = (fp_ends - fp_starts).astype(np.int32)

        if with_scores and confidence is not None:
            fp_scores = np.zeros(len(fp_starts), dtype=np.float32)
            for i, (s, e) in enumerate(zip(fp_starts, fp_ends)):
                fp_scores[i] = np.mean(confidence[s:e])
            result['footprint_scores'] = fp_scores

    # Find MSPs (accessible regions, state 1)
    # Pad with 0 (footprint) at edges
    acc_padded = np.concatenate([[0], states, [0]])
    acc_diff = np.diff(acc_padded)

    # MSP starts: transition from 0 to 1 (diff == 1)
    # MSP ends: transition from 1 to 0 (diff == -1)
    msp_starts = np.where(acc_diff == 1)[0]
    msp_ends = np.where(acc_diff == -1)[0]

    if len(msp_starts) > 0:
        msp_sizes = msp_ends - msp_starts

        # Filter by minimum size
        size_mask = msp_sizes >= msp_min_size
        msp_starts = msp_starts[size_mask]
        msp_sizes = msp_sizes[size_mask]

        if len(msp_starts) > 0:
            result['msp_starts'] = msp_starts.astype(np.int32)
            result['msp_sizes'] = msp_sizes.astype(np.int32)

            if with_scores and confidence is not None:
                # For MSPs, confidence is P(accessible | obs) = 1 - P(footprint | obs)
                msp_scores = np.zeros(len(msp_starts), dtype=np.float32)
                for i, (s, sz) in enumerate(zip(msp_starts, msp_sizes)):
                    # Mean of (1 - footprint_confidence) in this region
                    msp_scores[i] = np.mean(1.0 - confidence[s:s+sz])
                result['msp_scores'] = msp_scores

    return result


def detect_mode_from_bam(bam_path: str, n_sample: int = 100) -> str:
    """
    Auto-detect the appropriate mode from MM tags in the BAM file.

    Samples the first n_sample reads with MM tags and checks:
    - DAF-seq: Has T-a (C→T deamination) or A+a (G→A deamination) tags
    - PacBio fiber-seq: Has A+a only (m6A methylation)
    - Nanopore fiber-seq: Has A+a only but typically lower modification rates

    Returns: 'daf', 'pacbio-fiber', 'nanopore-fiber', or 'unknown'
    """
    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            t_minus_a_count = 0  # T-a tags (DAF + strand)
            a_plus_a_count = 0   # A+a tags (DAF - strand or m6A)
            c_plus_m_count = 0   # C+m tags (5mC methylation)
            other_count = 0
            reads_with_mm = 0

            for read in bam.fetch(until_eof=True):
                if reads_with_mm >= n_sample:
                    break

                if read.is_unmapped or read.query_sequence is None:
                    continue

                # Get MM tag
                try:
                    mm_tag = read.get_tag('MM') if read.has_tag('MM') else \
                             read.get_tag('Mm') if read.has_tag('Mm') else None
                except KeyError:
                    mm_tag = None

                if not mm_tag:
                    continue

                reads_with_mm += 1

                # Parse MM tag to identify modification types
                for mod_spec in mm_tag.split(';'):
                    if not mod_spec:
                        continue
                    parts = mod_spec.split(',')
                    if len(parts) < 2:
                        continue
                    base_mod = parts[0].strip()

                    if base_mod.startswith('T-a') or base_mod.startswith('T+a'):
                        t_minus_a_count += 1
                    elif base_mod.startswith('A+a') or base_mod.startswith('A-a'):
                        a_plus_a_count += 1
                    elif base_mod.startswith('C+m') or base_mod.startswith('C-m'):
                        c_plus_m_count += 1
                    else:
                        other_count += 1

            # Determine mode based on tag patterns
            if reads_with_mm == 0:
                return 'unknown'

            # DAF-seq uses T-a for + strand deamination (C→T)
            # and A+a for - strand deamination (G→A)
            if t_minus_a_count > 0:
                # T-a tags are DAF-specific (deaminated C shows as T)
                return 'daf'
            elif a_plus_a_count > 0 and c_plus_m_count == 0:
                # Only A+a without 5mC - could be m6A fiber-seq or DAF - strand only
                # Check if we also see patterns suggesting DAF
                # For now, assume pacbio-fiber unless we see T-a
                return 'pacbio-fiber'
            else:
                return 'unknown'

    except Exception as e:
        print(f"  Warning: Could not auto-detect mode from BAM: {e}")
        return 'unknown'


def _extract_fiber_read_from_pysam(read, mode: str, prob_threshold: int) -> Optional[dict]:
    """Extract minimal data needed for HMM processing from a pysam read."""
    query_sequence = read.query_sequence
    if not query_sequence:
        return None

    # Parse MM/ML tags for modification calls
    m6a_query_positions = []

    try:
        mod_bases = read.modified_bases
        if mod_bases:
            for (base, strand, mod_code), positions in mod_bases.items():
                if mode == 'daf':
                    # DAF-seq: deamination converts C->T (+ strand) or G->A (- strand)
                    # Accept T or A bases
                    if base not in ('T', 'A'):
                        continue
                    for pos, qual in positions:
                        if qual == -1 or qual >= prob_threshold:
                            m6a_query_positions.append(pos)
                else:
                    # m6a or nanopore mode:
                    # A positions = m6A on forward strand
                    # T positions = m6A on reverse strand (T in read is A on template)
                    # mod_code 'a' = m6A, 21839 = ChEBI code for m6A
                    if base not in ('A', 'T'):
                        continue
                    if mod_code not in ('a', 21839):
                        continue
                    for pos, qual in positions:
                        if qual == -1 or qual >= prob_threshold:
                            m6a_query_positions.append(pos)
    except Exception:
        return None

    return {
        'read_id': read.query_name,
        'query_sequence': query_sequence,
        'm6a_query_positions': set(m6a_query_positions),
        'query_length': len(query_sequence)
    }


def _process_single_read(fiber_read: dict, model, edge_trim: int, circular: bool,
                          mode: str, context_size: int, msp_min_size: int,
                          with_scores: bool, return_posteriors: bool = False) -> Optional[dict]:
    """Process a single read through HMM. Returns footprint data or None."""

    query_sequence = fiber_read['query_sequence']
    m6a_positions = fiber_read['m6a_query_positions']

    # Detect strand
    if mode == 'daf':
        strand = detect_daf_strand(query_sequence, m6a_positions)
    elif mode == 'nanopore-fiber':
        strand = '.'  # No strand detection for nanopore
    else:
        strand = '.'

    # Encode
    encoded = encode_from_query_sequence(
        query_sequence, m6a_positions, edge_trim,
        mode=mode, strand=strand, context_size=context_size
    )

    if len(encoded) == 0:
        return None

    # Predict
    fp_result = predict_footprints_and_msps(model, encoded, msp_min_size, with_scores,
                                             return_posteriors=return_posteriors)

    # If no footprints and we don't need posteriors, skip
    if len(fp_result['footprint_starts']) == 0 and len(fp_result['msp_starts']) == 0:
        if not return_posteriors:
            return None

    result = {
        'ns': fp_result['footprint_starts'],
        'nl': fp_result['footprint_sizes'],
        'ns_scores': fp_result.get('footprint_scores'),
        'as': fp_result['msp_starts'],
        'al': fp_result['msp_sizes'],
        'as_scores': fp_result.get('msp_scores')
    }

    # Include posteriors data if requested
    if return_posteriors and fp_result.get('posteriors') is not None:
        result['posteriors'] = fp_result['posteriors']
        result['strand'] = strand

    return result
