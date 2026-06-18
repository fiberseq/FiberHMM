"""FiberHMM core per-read HMM inference engine."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pysam

from fiberhmm.core.bam_reader import (
    _has_mm_ml_inputs,
    _mm_mod_spec_parts,
    daf_strand_from_tag,
    detect_daf_strand,
    encode_from_query_sequence,
    extract_daf_iupac_positions,
    has_iupac_encoding,
    parse_mm_tag_query_positions,
)
from fiberhmm.core.hmm import FiberHMM
from fiberhmm.core.tag_access import compact_ml_value, get_preferred_tag
from fiberhmm.inference.circular import (
    project_center_runs,
    project_center_scores,
    split_intervals_for_legacy,
    tile_sequence_and_mods,
)
from fiberhmm.inference.payload_read import PayloadRead

try:
    from numba import njit as _numba_njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def _numba_njit(*args, **kwargs):  # type: ignore[misc]
        def _wrap(fn):
            return fn
        return _wrap


@_numba_njit(cache=True)
def _footprint_runs_numba(states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(states)
    count = 0
    in_run = False
    for i in range(n):
        if int(states[i]) == 0:
            if not in_run:
                count += 1
                in_run = True
        else:
            in_run = False

    starts = np.empty(count, dtype=np.int64)
    ends = np.empty(count, dtype=np.int64)
    idx = 0
    in_run = False
    for i in range(n):
        if int(states[i]) == 0:
            if not in_run:
                starts[idx] = i
                in_run = True
        else:
            if in_run:
                ends[idx] = i
                idx += 1
                in_run = False

    if in_run:
        ends[idx] = n

    return starts, ends


def footprint_runs(states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if _HAS_NUMBA:
        return _footprint_runs_numba(states)

    states_padded = np.concatenate([[1], states, [1]])
    diff = np.diff(states_padded)
    return np.where(diff == -1)[0], np.where(diff == 1)[0]


_footprint_runs = footprint_runs


@dataclass(frozen=True)
class _NucBoundaries:
    starts: np.ndarray
    ends: np.ndarray


@dataclass(frozen=True)
class _MspIntervals:
    starts: np.ndarray
    sizes: np.ndarray


@dataclass(frozen=True)
class _FootprintRunBoundaries:
    starts: np.ndarray
    ends: np.ndarray


@dataclass(frozen=True)
class _FootprintPredictionResult:
    starts: np.ndarray
    sizes: np.ndarray
    count: int
    scores: Optional[np.ndarray]

    def as_tuple(self) -> Tuple[np.ndarray, np.ndarray, int, Optional[np.ndarray]]:
        return self.starts, self.sizes, self.count, self.scores


@dataclass(frozen=True)
class _PredictionOutputs:
    states: np.ndarray
    confidence: Optional[np.ndarray]
    posteriors: Optional[np.ndarray]


@dataclass(frozen=True)
class _EncodingInputs:
    sequence: str
    mod_positions: object
    circular_read_length: Optional[int]


@dataclass(frozen=True)
class _CircularResultTrack:
    tiled_starts: np.ndarray
    tiled_sizes: np.ndarray
    circular_intervals: list
    circular_scores: Optional[np.ndarray]
    legacy_starts: np.ndarray
    legacy_sizes: np.ndarray
    legacy_scores: Optional[np.ndarray]


@dataclass(frozen=True)
class _SingleReadResultRequest:
    fp_result: dict
    strand: str
    encoded: np.ndarray
    return_posteriors: bool
    include_encoded: bool


def _new_mode_detection_counts() -> dict:
    return {
        't_minus_a': 0,
        'a_plus_a': 0,
        'c_plus_m': 0,
        'other': 0,
        'reads_with_mm': 0,
        'iupac': 0,
        'st': 0,
    }


def _mode_detection_key_for_mm_base(base_mod: str) -> str:
    if base_mod.startswith('T-a') or base_mod.startswith('T+a'):
        return 't_minus_a'
    if base_mod.startswith('A+a') or base_mod.startswith('A-a'):
        return 'a_plus_a'
    if base_mod.startswith('C+m') or base_mod.startswith('C-m'):
        return 'c_plus_m'
    return 'other'


def _record_mm_mode_specs(counts: dict, mm_tag: str) -> None:
    for mod_spec in mm_tag.split(';'):
        if not mod_spec:
            continue
        parts = _mm_mod_spec_parts(mod_spec)
        if parts is None:
            continue
        key = _mode_detection_key_for_mm_base(parts.base_mod.strip())
        counts[key] += 1


def _record_iupac_mode_indicators(counts: dict, read) -> None:
    if has_iupac_encoding(read.query_sequence):
        counts['iupac'] += 1
    if read.has_tag('st'):
        counts['st'] += 1


def _record_mm_mode_indicators(counts: dict, read) -> None:
    mm_tag = get_preferred_tag(read, 'MM', 'Mm')
    if mm_tag:
        counts['reads_with_mm'] += 1
        _record_mm_mode_specs(counts, mm_tag)


def _mode_from_detection_counts(counts: dict) -> str:
    if counts['reads_with_mm'] == 0:
        # No MM tags found -- check for IUPAC R/Y encoding.
        if counts['iupac'] > 0 and counts['st'] > 0:
            return 'daf'
        return 'unknown'

    # DAF-seq uses T-a for + strand deamination (C->T)
    # and A+a for - strand deamination (G->A).
    if counts['t_minus_a'] > 0:
        # T-a tags are DAF-specific (deaminated C shows as T).
        return 'daf'
    if counts['a_plus_a'] > 0 and counts['c_plus_m'] == 0:
        # Only A+a without 5mC: could be m6A fiber-seq or DAF - strand only.
        # For now, assume pacbio-fiber unless we see T-a.
        return 'pacbio-fiber'
    return 'unknown'


def _mean_interval_scores(confidence: np.ndarray, starts, ends,
                          invert: bool = False) -> np.ndarray:
    scores = np.zeros(len(starts), dtype=np.float32)
    for i, (s, e) in enumerate(zip(starts, ends)):
        segment = confidence[int(s):int(e)]
        scores[i] = np.mean(1.0 - segment) if invert else np.mean(segment)
    return scores


def _msp_intervals_from_nuc_boundaries(nuc_starts, nuc_ends, read_length: int,
                                       msp_min_size: int) -> _MspIntervals:
    msp_start_list = []
    msp_size_list = []

    if len(nuc_starts) > 0:
        if nuc_starts[0] > 0:
            msp_start_list.append(0)
            msp_size_list.append(int(nuc_starts[0]))
        for i in range(len(nuc_starts) - 1):
            gap_start = int(nuc_ends[i])
            gap_size = int(nuc_starts[i + 1]) - gap_start
            if gap_size > 0:
                msp_start_list.append(gap_start)
                msp_size_list.append(gap_size)
        if nuc_ends[-1] < read_length:
            msp_start_list.append(int(nuc_ends[-1]))
            msp_size_list.append(read_length - int(nuc_ends[-1]))
    else:
        # No nucleosome-sized footprints: entire read is one MSP.
        msp_start_list.append(0)
        msp_size_list.append(read_length)

    if not msp_start_list:
        return _MspIntervals(
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )

    starts = np.array(msp_start_list, dtype=np.int32)
    sizes = np.array(msp_size_list, dtype=np.int32)
    size_mask = sizes >= msp_min_size
    return _MspIntervals(starts[size_mask], sizes[size_mask])


def _nuc_boundaries_from_footprint_runs(
    fp_starts,
    fp_ends,
    nuc_min_size: int,
) -> _NucBoundaries:
    if len(fp_starts) == 0:
        return _NucBoundaries(
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )

    fp_sizes_arr = fp_ends - fp_starts
    nuc_mask = fp_sizes_arr >= nuc_min_size
    return _NucBoundaries(fp_starts[nuc_mask], fp_ends[nuc_mask])


def _empty_interval_result(include_predictions: bool = False) -> dict:
    result = {
        'footprint_starts': np.array([], dtype=np.int32),
        'footprint_sizes': np.array([], dtype=np.int32),
        'footprint_scores': None,
        'msp_starts': np.array([], dtype=np.int32),
        'msp_sizes': np.array([], dtype=np.int32),
        'msp_scores': None,
    }
    if include_predictions:
        result.update({
            'states': np.array([], dtype=np.int8),
            'posteriors': None,
        })
    return result


def _add_footprint_track(result: dict, states: np.ndarray,
                         confidence: Optional[np.ndarray],
                         with_scores: bool) -> _FootprintRunBoundaries:
    fp_starts, fp_ends = _footprint_runs(states)

    if len(fp_starts) > 0:
        result['footprint_starts'] = fp_starts.astype(np.int32)
        result['footprint_sizes'] = (fp_ends - fp_starts).astype(np.int32)

        if with_scores and confidence is not None:
            result['footprint_scores'] = _mean_interval_scores(
                confidence, fp_starts, fp_ends,
            )

    return _FootprintRunBoundaries(fp_starts, fp_ends)


def _add_msp_track(
    result: dict,
    read_length: int,
    fp_starts: np.ndarray,
    fp_ends: np.ndarray,
    confidence: Optional[np.ndarray],
    msp_min_size: int,
    with_scores: bool,
    nuc_min_size: int,
) -> None:
    nuc_boundaries = _nuc_boundaries_from_footprint_runs(
        fp_starts,
        fp_ends,
        nuc_min_size,
    )

    msp_intervals = _msp_intervals_from_nuc_boundaries(
        nuc_boundaries.starts,
        nuc_boundaries.ends,
        read_length,
        msp_min_size,
    )

    if len(msp_intervals.starts) > 0:
        result['msp_starts'] = msp_intervals.starts
        result['msp_sizes'] = msp_intervals.sizes

        if with_scores and confidence is not None:
            result['msp_scores'] = _mean_interval_scores(
                confidence,
                msp_intervals.starts,
                msp_intervals.starts + msp_intervals.sizes,
                invert=True,
            )


def predict_footprints(
    model: FiberHMM,
    encoded_read: np.ndarray,
    with_scores: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int, Optional[np.ndarray]]:
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
    return _predict_footprint_result(model, encoded_read, with_scores).as_tuple()


def _empty_footprint_prediction_result() -> _FootprintPredictionResult:
    return _FootprintPredictionResult(np.array([]), np.array([]), 0, None)


def _predict_footprint_result(
    model: FiberHMM,
    encoded_read: np.ndarray,
    with_scores: bool = False,
) -> _FootprintPredictionResult:
    """Run HMM Viterbi prediction and return named footprint arrays."""
    if len(encoded_read) == 0:
        return _empty_footprint_prediction_result()

    # Predict states (0 = footprint, 1 = accessible)
    if with_scores:
        states, confidence = model.predict_with_confidence(encoded_read)
    else:
        states = model.predict(encoded_read)
        confidence = None

    starts, ends = _footprint_runs(states)

    if len(starts) == 0:
        return _empty_footprint_prediction_result()

    sizes = ends - starts

    # Compute per-footprint scores
    scores = None
    if with_scores and confidence is not None:
        scores = _mean_interval_scores(confidence, starts, ends)

    return _FootprintPredictionResult(starts, sizes, len(starts), scores)


def _extract_footprints_from_states(states: np.ndarray, confidence: Optional[np.ndarray],
                                     msp_min_size: int, with_scores: bool,
                                     nuc_min_size: int = 85) -> dict:
    """
    Extract footprints and MSPs from HMM states (without running HMM again).

    States: 0 = footprint, 1 = accessible

    MSPs are bounded by nucleosome-sized footprints (>= nuc_min_size) only.
    Small footprints do not break MSPs, matching the fibertools convention.

    Used for timing breakdown to separate HMM time from post-processing.
    """
    result = _empty_interval_result()

    if len(states) == 0:
        return result

    footprint_runs = _add_footprint_track(
        result, states, confidence, with_scores,
    )

    # Find MSPs (accessible regions between nucleosome-sized footprints)
    # Only footprints >= nuc_min_size act as MSP boundaries
    _add_msp_track(
        result,
        len(states),
        footprint_runs.starts,
        footprint_runs.ends,
        confidence,
        msp_min_size,
        with_scores,
        nuc_min_size,
    )

    return result


def _project_circular_result_track(
    starts,
    sizes,
    scores,
    read_length: int,
) -> _CircularResultTrack:
    tiled_starts = np.asarray(starts, dtype=np.int64)
    tiled_ends = tiled_starts + np.asarray(sizes, dtype=np.int64)
    circular_intervals = project_center_runs(tiled_starts, tiled_ends, read_length)
    circular_scores = project_center_scores(
        tiled_starts,
        tiled_ends,
        scores,
        read_length,
    )
    legacy_starts, legacy_sizes, legacy_scores = split_intervals_for_legacy(
        circular_intervals,
        read_length,
        circular_scores,
    )
    return _CircularResultTrack(
        tiled_starts=tiled_starts,
        tiled_sizes=tiled_ends - tiled_starts,
        circular_intervals=circular_intervals,
        circular_scores=circular_scores,
        legacy_starts=legacy_starts,
        legacy_sizes=legacy_sizes,
        legacy_scores=legacy_scores,
    )


def _center_copy_states(states: np.ndarray, read_length: int) -> np.ndarray:
    return states[read_length:2 * read_length].astype(np.int8, copy=False)


def _extract_footprints_from_states_circular(
    states: np.ndarray,
    confidence: Optional[np.ndarray],
    read_length: int,
    msp_min_size: int,
    with_scores: bool,
    nuc_min_size: int = 85,
) -> dict:
    """Extract circular intervals from 3x-tiled HMM states.

    The public legacy arrays are split into valid linear pieces. The unsplit
    circular intervals are retained for MA/AN emission and circular TF recall.
    """
    tiled_result = _extract_footprints_from_states(
        states,
        confidence,
        msp_min_size=msp_min_size,
        with_scores=with_scores,
        nuc_min_size=nuc_min_size,
    )

    nuc_track = _project_circular_result_track(
        tiled_result['footprint_starts'],
        tiled_result['footprint_sizes'],
        tiled_result.get('footprint_scores'),
        read_length,
    )
    msp_track = _project_circular_result_track(
        tiled_result['msp_starts'],
        tiled_result['msp_sizes'],
        tiled_result.get('msp_scores'),
        read_length,
    )

    return {
        'footprint_starts': nuc_track.legacy_starts,
        'footprint_sizes': nuc_track.legacy_sizes,
        'footprint_scores': nuc_track.legacy_scores,
        'msp_starts': msp_track.legacy_starts,
        'msp_sizes': msp_track.legacy_sizes,
        'msp_scores': msp_track.legacy_scores,
        'states': _center_copy_states(states, read_length),
        'posteriors': None,
        'circular': True,
        'circular_read_length': read_length,
        'circular_ns': nuc_track.circular_intervals,
        'circular_as': msp_track.circular_intervals,
        'circular_ns_scores': nuc_track.circular_scores,
        'circular_as_scores': msp_track.circular_scores,
        'tiled_ns': nuc_track.tiled_starts.astype(np.int32),
        'tiled_nl': nuc_track.tiled_sizes.astype(np.int32),
        'tiled_as': msp_track.tiled_starts.astype(np.int32),
        'tiled_al': msp_track.tiled_sizes.astype(np.int32),
    }


def _predict_state_outputs(model: FiberHMM, encoded_read: np.ndarray,
                           with_scores: bool,
                           return_posteriors: bool) -> _PredictionOutputs:
    if with_scores or return_posteriors:
        states, posteriors_full = model.predict_with_posteriors(encoded_read)
        confidence = posteriors_full[np.arange(len(states)), states]
        return _PredictionOutputs(states, confidence, posteriors_full)

    states = model.predict(encoded_read)
    return _PredictionOutputs(states, None, None)


def _footprint_posterior_track(posteriors_full: np.ndarray,
                               start: int = 0,
                               end: Optional[int] = None) -> np.ndarray:
    return posteriors_full[start:end, 0].astype(np.float16)


def _circular_prediction_result(
    prediction: _PredictionOutputs,
    circular_read_length: int,
    msp_min_size: int,
    with_scores: bool,
    return_posteriors: bool,
    nuc_min_size: int,
) -> dict:
    result = _extract_footprints_from_states_circular(
        prediction.states,
        prediction.confidence,
        circular_read_length,
        msp_min_size,
        with_scores,
        nuc_min_size=nuc_min_size,
    )
    if return_posteriors and prediction.posteriors is not None:
        n = int(circular_read_length)
        result['posteriors'] = _footprint_posterior_track(
            prediction.posteriors, n, 2 * n,
        )
    return result


def _linear_prediction_result(
    prediction: _PredictionOutputs,
    msp_min_size: int,
    with_scores: bool,
    nuc_min_size: int,
) -> dict:
    result = {'states': prediction.states}
    result.update(
        _extract_footprints_from_states(
            prediction.states,
            prediction.confidence,
            msp_min_size,
            with_scores,
            nuc_min_size=nuc_min_size,
        )
    )
    return result


def predict_footprints_and_msps(model: FiberHMM, encoded_read: np.ndarray,
                                 msp_min_size: int = 147,
                                 with_scores: bool = False,
                                 return_posteriors: bool = False,
                                 nuc_min_size: int = 85,
                                 circular_read_length: Optional[int] = None) -> dict:
    """
    Run HMM prediction to call both footprints (ns/nl) and MSPs (as/al).

    States: 0 = footprint, 1 = accessible

    MSPs (Methylase-Sensitive Patches) are accessible regions between
    nucleosome-sized footprints (>= nuc_min_size). Small footprints do not
    break MSPs, matching the fibertools convention.

    Args:
        model: Trained FiberHMM model
        encoded_read: Encoded observation sequence
        msp_min_size: Minimum size for an accessible region to be called as MSP
        with_scores: If True, compute confidence scores
        return_posteriors: If True, return full posterior array for CNN training
        nuc_min_size: Minimum footprint size (bp) to count as nucleosome-sized
            for MSP boundary detection (default: 85)

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
    result = _empty_interval_result(include_predictions=True)

    if len(encoded_read) == 0:
        return result

    # Predict states (0 = footprint, 1 = accessible)
    # Use predict_with_posteriors if we need posteriors or scores (shares computation)
    prediction = _predict_state_outputs(
        model, encoded_read, with_scores, return_posteriors,
    )

    if return_posteriors:
        # P(footprint) = posteriors_full[:, 0]
        result['posteriors'] = _footprint_posterior_track(prediction.posteriors)

    if circular_read_length is not None:
        result.update(
            _circular_prediction_result(
                prediction,
                circular_read_length,
                msp_min_size,
                with_scores,
                return_posteriors,
                nuc_min_size=nuc_min_size,
            )
        )
        return result

    result.update(
        _linear_prediction_result(
            prediction,
            msp_min_size,
            with_scores,
            nuc_min_size=nuc_min_size,
        )
    )

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
        with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
            counts = _new_mode_detection_counts()
            n_scanned = 0

            for read in bam.fetch(until_eof=True):
                if counts['reads_with_mm'] >= n_sample and n_scanned >= n_sample:
                    break

                if read.is_unmapped or read.query_sequence is None:
                    continue

                # Track IUPAC indicators (always, up to n_sample)
                if n_scanned < n_sample:
                    n_scanned += 1
                    _record_iupac_mode_indicators(counts, read)

                # Get MM tag
                if counts['reads_with_mm'] < n_sample:
                    _record_mm_mode_indicators(counts, read)

            return _mode_from_detection_counts(counts)

    except Exception as e:
        print(f"  Warning: Could not auto-detect mode from BAM: {e}")
        return 'unknown'


# ---------------------------------------------------------------------------
# Slim IPC support for fiberhmm-apply (mirrors the recall_tfs pattern).
#
# The serial main process used to call _extract_fiber_read_from_pysam per
# read — which decodes the query sequence (1-2 ms for a 20 kb PacBio read)
# AND parses MM/ML (1-2 ms).  At ~3-7 ms/read serial in main, the apply
# pipeline ceiling was ~150-300 r/s regardless of worker count, leaving
# -c 4 workers at ~38% CPU instead of 400%.
#
# The slim-IPC path moves the MM/ML parse into the workers: main does
# only the cheap pysam tag access + sequence decode, builds a slim
# payload, ships it.  Workers parse + encode + Viterbi + decode.  This
# shifts ~1-3 ms/read off the main-process critical path.
# ---------------------------------------------------------------------------

_APPLY_PAYLOAD_TAGS = ('MM', 'Mm', 'ML', 'Ml', 'st')


def _apply_payload_tags(read) -> dict:
    tags = {}
    for tag in _APPLY_PAYLOAD_TAGS:
        if read.has_tag(tag):
            tags[tag] = _apply_payload_tag_value(read, tag)
    return tags


def _apply_payload_tag_value(read, tag: str):
    val = read.get_tag(tag)
    if tag in ('ML', 'Ml'):
        # array.array('B', ...) -> bytes via buffer protocol: fast memcpy,
        # avoids ~5000 PyInt allocations per Hia5 PacBio read.
        return compact_ml_value(val)
    return val


def _apply_payload_daf_md_result(read, query_sequence: str, ref_fasta=None):
    if has_iupac_encoding(query_sequence):
        return None
    from fiberhmm.daf.encoder import get_daf_positions
    return get_daf_positions(read, ref_fasta=ref_fasta)


def make_apply_payload(read, mode: str = 'fiber', ref_fasta=None) -> Optional[dict]:
    """Extract slim payload from a pysam read for the apply slim-IPC path.

    Runs in the *main* process.  Does NOT parse MM/ML — that moves to the
    worker via extract_fiber_read_from_payload().

    When ``mode == 'daf'`` and the read carries no R/Y IUPAC encoding,
    additionally pre-computes the MD-derived deamination positions so the
    worker can run the HMM directly on the raw BAM without an upstream
    ``fiberhmm-daf-encode`` pass. The MD walk has to happen here because
    the slim ``PayloadRead`` stub in the worker has no access to the
    live pysam alignment API. Cost: ~1-3ms per read for raw DAF input;
    zero cost for pre-encoded BAMs (the IUPAC fast path triggers first).

    Returns None only if the read has no sequence (caller treats as skip).
    """
    seq = read.query_sequence
    if not seq:
        return None

    tags = _apply_payload_tags(read)

    payload = {
        'query_name': read.query_name,
        'query_sequence': seq,
        'is_reverse': read.is_reverse,
        'tags': tags,
    }

    # DAF MD-fallback precomputation (live-read side, before slim-IPC handoff).
    if mode == 'daf':
        md_res = _apply_payload_daf_md_result(read, seq, ref_fasta=ref_fasta)
        if md_res is not None:
            payload['_daf_md_result'] = md_res   # (ct_list, ga_list, strand_tag)

    return payload


def extract_fiber_read_from_payload(
    payload: dict,
    mode: str,
    prob_threshold: int,
) -> Optional[dict]:
    """Worker-side: turn a slim payload into a fiber_read dict.

    Equivalent to _extract_fiber_read_from_pysam(real_read, mode, prob_threshold)
    but takes the slim payload built by make_apply_payload().  Returns None
    on read with no usable modification data (no MM/ML, no IUPAC codes,
    extraction failure) — caller treats result=None as 'no footprints'.
    """
    return _extract_fiber_read_from_pysam(
        PayloadRead(
            payload['query_sequence'],
            payload['is_reverse'],
            payload['tags'],
            query_name=payload['query_name'],
            daf_md_result=payload.get('_daf_md_result'),
        ),
        mode, prob_threshold,
    )


# DAF strand-swap chimera filter. Run-constant config, set per worker via
# configure_daf_chimera_filter() (default: filter ON). CHIMERA_SKIP is a
# distinct sentinel (vs None) so workers can tally chimeras as their own skip
# reason rather than folding them into "no_modifications".
CHIMERA_SKIP = object()
_DAF_CHIMERA_CFG = {'filter': True, 'min_seg': 5, 'purity': 0.8}


def configure_daf_chimera_filter(filter_chimeras: bool = True,
                                 min_seg: int = 5, purity: float = 0.8) -> None:
    """Set the DAF chimera-filter policy for this process (worker init)."""
    _DAF_CHIMERA_CFG['filter'] = bool(filter_chimeras)
    _DAF_CHIMERA_CFG['min_seg'] = int(min_seg)
    _DAF_CHIMERA_CFG['purity'] = float(purity)


def _extract_daf_iupac_fiber_read(read, query_sequence: str) -> Optional[dict]:
    st_tag = read.get_tag('st') if read.has_tag('st') else None
    mod_positions, strand, conv_seq = extract_daf_iupac_positions(query_sequence, st_tag)
    if not mod_positions:
        return None
    return {
        'read_id': read.query_name,
        'query_sequence': conv_seq,       # Y->T, R->A (pure ACGT)
        'm6a_query_positions': mod_positions,
        'query_length': len(conv_seq),
        '_daf_strand': strand,            # pre-computed from st tag
    }


def _extract_daf_md_fiber_read(read, query_sequence: str, ref_fasta=None):
    md_result = getattr(read, '_daf_md_result', None)
    if md_result is None and hasattr(read, 'get_aligned_pairs'):
        from fiberhmm.daf.encoder import get_daf_positions
        md_result = get_daf_positions(read, ref_fasta=ref_fasta)
    if md_result is None:
        return None

    ct_pos, ga_pos, strand_tag = md_result
    # Strand-swap chimera filter (DAF only): a read deaminated CT in one
    # segment and GA in another corrupts the single-strand assignment.
    # Drop it (returns a distinct sentinel so callers can report counts).
    if _DAF_CHIMERA_CFG['filter']:
        from fiberhmm.daf.encoder import is_daf_chimera
        if is_daf_chimera(ct_pos, ga_pos,
                          min_seg_events=_DAF_CHIMERA_CFG['min_seg'],
                          purity=_DAF_CHIMERA_CFG['purity']):
            return CHIMERA_SKIP

    mod_positions = set(ct_pos) if strand_tag == 'CT' else set(ga_pos)
    if not mod_positions:
        return None

    # query_sequence is already raw ACGT (no R/Y to decode); uppercase to match
    # what extract_daf_iupac_positions emits.
    daf_strand = daf_strand_from_tag(strand_tag)
    return {
        'read_id': read.query_name,
        'query_sequence': query_sequence.upper(),
        'm6a_query_positions': mod_positions,
        'query_length': len(query_sequence),
        '_daf_strand': daf_strand if daf_strand != '.' else '-',
    }


def _extract_mm_ml_fiber_read(read, query_sequence: str, mode: str,
                              prob_threshold: int) -> Optional[dict]:
    mm_tag = get_preferred_tag(read, 'MM', 'Mm', '')
    ml_raw = get_preferred_tag(read, 'ML', 'Ml', None)

    if not _has_mm_ml_inputs(mm_tag, ml_raw):
        return None

    # Convert ML to bytes once (fast memcpy, no PyInt allocations).
    ml_bytes = compact_ml_value(ml_raw)

    try:
        mod_pos_set = parse_mm_tag_query_positions(
            mm_tag, ml_bytes, query_sequence, read.is_reverse,
            prob_threshold=prob_threshold, mode=mode,
        )
    except Exception:
        return None

    return {
        'read_id': read.query_name,
        'query_sequence': query_sequence,
        'm6a_query_positions': mod_pos_set,
        'query_length': len(query_sequence),
        'is_reverse': bool(read.is_reverse),
    }


def _extract_fiber_read_from_pysam(read, mode: str, prob_threshold: int,
                                    ref_fasta=None) -> Optional[dict]:
    """Extract minimal data needed for HMM processing from a pysam read."""
    query_sequence = read.query_sequence
    if not query_sequence:
        return None

    # IUPAC R/Y branch: DAF-seq reads with deamination encoded in the sequence
    if mode == 'daf' and has_iupac_encoding(query_sequence):
        return _extract_daf_iupac_fiber_read(read, query_sequence)

    # MD fallback for DAF mode: raw aligned BAM (no R/Y in sequence yet).
    # Parse MD on the fly into the same (mod_positions, strand) the R/Y
    # path would have produced, so the HMM emits byte-identical calls vs.
    # the two-pass `fiberhmm-daf-encode | fiberhmm-call` pipeline.
    #
    # Two sources for the MD result:
    #   (a) Live pysam AlignedSegment with get_aligned_pairs available
    #       (region-parallel workers fetch reads directly).
    #   (b) Pre-computed by make_apply_payload and stashed on the slim
    #       payload stub as ``_daf_md_result`` (slim-IPC path).
    if mode == 'daf':
        md_read = _extract_daf_md_fiber_read(read, query_sequence, ref_fasta)
        if md_read is not None:
            return md_read

    # Legacy MM/ML path: use the fast vectorized parser instead of
    # read.modified_bases.  pysam's modified_bases returns a dict of
    # (base, strand, mod_code) -> [(pos, qual), ...] which forces a Python
    # iteration over every modification (~5000 per Hia5 PacBio read =
    # ~5-10 ms/read).  parse_mm_tag_query_positions does the same parse in
    # vectorized numpy and accepts ML as bytes (no PyInt materialization).
    return _extract_mm_ml_fiber_read(read, query_sequence, mode, prob_threshold)


def _single_read_result_from_request(request: _SingleReadResultRequest) -> dict:
    result = _base_single_read_fields(request.fp_result)
    if request.fp_result.get('circular'):
        result.update(_circular_single_read_fields(request.fp_result))

    if (
        request.return_posteriors
        and request.fp_result.get('posteriors') is not None
    ):
        result['posteriors'] = request.fp_result['posteriors']
        result['strand'] = request.strand

    if request.include_encoded:
        result['encoded'] = request.encoded
        result['strand'] = request.strand

    return result


def _single_read_result_from_prediction(fp_result: dict, strand: str,
                                        encoded: np.ndarray,
                                        return_posteriors: bool,
                                        include_encoded: bool) -> dict:
    return _single_read_result_from_request(
        _SingleReadResultRequest(
            fp_result=fp_result,
            strand=strand,
            encoded=encoded,
            return_posteriors=return_posteriors,
            include_encoded=include_encoded,
        ),
    )


def _base_single_read_fields(fp_result: dict) -> dict:
    return {
        'ns': fp_result['footprint_starts'],
        'nl': fp_result['footprint_sizes'],
        'ns_scores': fp_result.get('footprint_scores'),
        'as': fp_result['msp_starts'],
        'al': fp_result['msp_sizes'],
        'as_scores': fp_result.get('msp_scores')
    }


def _circular_single_read_fields(fp_result: dict) -> dict:
    return {
        'circular': True,
        'circular_read_length': fp_result['circular_read_length'],
        'circular_ns': fp_result['circular_ns'],
        'circular_as': fp_result['circular_as'],
        'circular_ns_scores': fp_result.get('circular_ns_scores'),
        'circular_as_scores': fp_result.get('circular_as_scores'),
        'tiled_ns': fp_result['tiled_ns'],
        'tiled_nl': fp_result['tiled_nl'],
        'tiled_as': fp_result['tiled_as'],
        'tiled_al': fp_result['tiled_al'],
    }


def _encoding_inputs_for_read(
    query_sequence: str,
    m6a_positions,
    circular: bool,
) -> _EncodingInputs:
    if circular and len(query_sequence) > 0:
        encode_sequence, encode_mods = tile_sequence_and_mods(
            query_sequence, m6a_positions,
        )
        return _EncodingInputs(encode_sequence, encode_mods, len(query_sequence))
    return _EncodingInputs(query_sequence, m6a_positions, None)


def _processing_strand_for_read(
    fiber_read: dict,
    query_sequence: str,
    m6a_positions,
    mode: str,
) -> str:
    if mode == 'daf':
        return fiber_read.get('_daf_strand') or detect_daf_strand(
            query_sequence,
            m6a_positions,
        )
    return '.'


def _should_skip_empty_prediction(
    fp_result: dict,
    return_posteriors: bool,
    include_encoded: bool,
) -> bool:
    has_intervals = (
        len(fp_result['footprint_starts']) > 0
        or len(fp_result['msp_starts']) > 0
    )
    return not has_intervals and not return_posteriors and not include_encoded


def _process_single_read(fiber_read: dict, model, edge_trim: int, circular: bool,
                          mode: str, context_size: int, msp_min_size: int,
                          with_scores: bool, return_posteriors: bool = False,
                          nuc_min_size: int = 85,
                          include_encoded: bool = False) -> Optional[dict]:
    """Process a single read through HMM. Returns footprint data or None.

    When include_encoded=True the encoded observation array and strand are
    attached to the result as 'encoded' and 'strand'.  Used by the fused
    apply+recall worker to avoid re-encoding the sequence for the TF scan.
    """

    query_sequence = fiber_read['query_sequence']
    m6a_positions = fiber_read['m6a_query_positions']

    # Detect strand
    strand = _processing_strand_for_read(
        fiber_read, query_sequence, m6a_positions, mode,
    )

    # Encode — pass is_reverse so nanopore mode handles strand correctly.
    # Circular mode keeps the 3x tiling private to inference and projects the
    # middle copy back to molecule coordinates before anything is written.
    is_reverse = fiber_read.get('is_reverse', False)
    encoding_inputs = _encoding_inputs_for_read(
        query_sequence, m6a_positions, circular,
    )

    encoded = encode_from_query_sequence(
        encoding_inputs.sequence, encoding_inputs.mod_positions, edge_trim,
        mode=mode, strand=strand, context_size=context_size,
        is_reverse=is_reverse,
    )

    if len(encoded) == 0:
        return None

    # Predict
    fp_result = predict_footprints_and_msps(
        model,
        encoded,
        msp_min_size,
        with_scores,
        return_posteriors=return_posteriors,
        nuc_min_size=nuc_min_size,
        circular_read_length=encoding_inputs.circular_read_length,
    )

    # If no footprints and we don't need posteriors or encoded, skip
    if _should_skip_empty_prediction(fp_result, return_posteriors, include_encoded):
        return None

    return _single_read_result_from_prediction(
        fp_result, strand, encoded, return_posteriors, include_encoded,
    )
