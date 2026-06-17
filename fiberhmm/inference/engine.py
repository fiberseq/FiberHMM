"""FiberHMM core per-read HMM inference engine."""

from typing import Optional, Tuple

import numpy as np
import pysam

from fiberhmm.core.bam_reader import (
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


def _mean_interval_scores(confidence: np.ndarray, starts, ends,
                          invert: bool = False) -> np.ndarray:
    scores = np.zeros(len(starts), dtype=np.float32)
    for i, (s, e) in enumerate(zip(starts, ends)):
        segment = confidence[int(s):int(e)]
        scores[i] = np.mean(1.0 - segment) if invert else np.mean(segment)
    return scores


def _msp_intervals_from_nuc_boundaries(nuc_starts, nuc_ends, read_length: int,
                                       msp_min_size: int) -> Tuple[np.ndarray, np.ndarray]:
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
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    starts = np.array(msp_start_list, dtype=np.int32)
    sizes = np.array(msp_size_list, dtype=np.int32)
    size_mask = sizes >= msp_min_size
    return starts[size_mask], sizes[size_mask]


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

    starts, ends = _footprint_runs(states)

    if len(starts) == 0:
        return np.array([]), np.array([]), 0, None

    sizes = ends - starts

    # Compute per-footprint scores
    scores = None
    if with_scores and confidence is not None:
        scores = _mean_interval_scores(confidence, starts, ends)

    return starts, sizes, len(starts), scores


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

    fp_starts, fp_ends = _footprint_runs(states)

    if len(fp_starts) > 0:
        result['footprint_starts'] = fp_starts.astype(np.int32)
        result['footprint_sizes'] = (fp_ends - fp_starts).astype(np.int32)

        if with_scores and confidence is not None:
            result['footprint_scores'] = _mean_interval_scores(
                confidence, fp_starts, fp_ends,
            )

    # Find MSPs (accessible regions between nucleosome-sized footprints)
    # Only footprints >= nuc_min_size act as MSP boundaries
    if len(states) > 0:
        if len(fp_starts) > 0:
            fp_sizes_arr = fp_ends - fp_starts
            nuc_mask = fp_sizes_arr >= nuc_min_size
            nuc_starts = fp_starts[nuc_mask]
            nuc_ends = fp_ends[nuc_mask]
        else:
            nuc_starts = np.array([], dtype=np.int64)
            nuc_ends = np.array([], dtype=np.int64)

        msp_starts_arr, msp_sizes_arr = _msp_intervals_from_nuc_boundaries(
            nuc_starts, nuc_ends, len(states), msp_min_size,
        )

        if len(msp_starts_arr) > 0:
            result['msp_starts'] = msp_starts_arr
            result['msp_sizes'] = msp_sizes_arr

            if with_scores and confidence is not None:
                result['msp_scores'] = _mean_interval_scores(
                    confidence,
                    msp_starts_arr,
                    msp_starts_arr + msp_sizes_arr,
                    invert=True,
                )

    return result


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

    fp_starts = np.asarray(tiled_result['footprint_starts'], dtype=np.int64)
    fp_ends = fp_starts + np.asarray(tiled_result['footprint_sizes'], dtype=np.int64)
    circular_nucs = project_center_runs(fp_starts, fp_ends, read_length)
    circular_nuc_scores = project_center_scores(
        fp_starts,
        fp_ends,
        tiled_result.get('footprint_scores'),
        read_length,
    )

    msp_starts = np.asarray(tiled_result['msp_starts'], dtype=np.int64)
    msp_ends = msp_starts + np.asarray(tiled_result['msp_sizes'], dtype=np.int64)
    circular_msps = project_center_runs(msp_starts, msp_ends, read_length)
    circular_msp_scores = project_center_scores(
        msp_starts,
        msp_ends,
        tiled_result.get('msp_scores'),
        read_length,
    )

    ns, nl, ns_scores = split_intervals_for_legacy(
        circular_nucs,
        read_length,
        circular_nuc_scores,
    )
    msp_s, msp_l, msp_scores = split_intervals_for_legacy(
        circular_msps,
        read_length,
        circular_msp_scores,
    )

    return {
        'footprint_starts': ns,
        'footprint_sizes': nl,
        'footprint_scores': ns_scores,
        'msp_starts': msp_s,
        'msp_sizes': msp_l,
        'msp_scores': msp_scores,
        'states': states[read_length:2 * read_length].astype(np.int8, copy=False),
        'posteriors': None,
        'circular': True,
        'circular_read_length': read_length,
        'circular_ns': circular_nucs,
        'circular_as': circular_msps,
        'circular_ns_scores': circular_nuc_scores,
        'circular_as_scores': circular_msp_scores,
        'tiled_ns': fp_starts.astype(np.int32),
        'tiled_nl': (fp_ends - fp_starts).astype(np.int32),
        'tiled_as': msp_starts.astype(np.int32),
        'tiled_al': (msp_ends - msp_starts).astype(np.int32),
    }


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
    if with_scores or return_posteriors:
        states, posteriors_full = model.predict_with_posteriors(encoded_read)
        confidence = posteriors_full[np.arange(len(states)), states]

        if return_posteriors:
            # P(footprint) = posteriors_full[:, 0]
            result['posteriors'] = posteriors_full[:, 0].astype(np.float16)
    else:
        states = model.predict(encoded_read)
        confidence = None

    if circular_read_length is not None:
        result.update(
            _extract_footprints_from_states_circular(
                states,
                confidence,
                circular_read_length,
                msp_min_size,
                with_scores,
                nuc_min_size=nuc_min_size,
            )
        )
        if return_posteriors and posteriors_full is not None:
            n = int(circular_read_length)
            result['posteriors'] = posteriors_full[n:2 * n, 0].astype(np.float16)
        return result

    result['states'] = states

    result.update(
        _extract_footprints_from_states(
            states, confidence, msp_min_size, with_scores,
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
            t_minus_a_count = 0  # T-a tags (DAF + strand)
            a_plus_a_count = 0   # A+a tags (DAF - strand or m6A)
            c_plus_m_count = 0   # C+m tags (5mC methylation)
            other_count = 0
            reads_with_mm = 0
            # Also track IUPAC indicators in the same pass
            iupac_count = 0
            st_count = 0
            n_scanned = 0

            for read in bam.fetch(until_eof=True):
                if reads_with_mm >= n_sample and n_scanned >= n_sample:
                    break

                if read.is_unmapped or read.query_sequence is None:
                    continue

                # Track IUPAC indicators (always, up to n_sample)
                if n_scanned < n_sample:
                    n_scanned += 1
                    if has_iupac_encoding(read.query_sequence):
                        iupac_count += 1
                    if read.has_tag('st'):
                        st_count += 1

                # Get MM tag
                if reads_with_mm < n_sample:
                    mm_tag = get_preferred_tag(read, 'MM', 'Mm')

                    if mm_tag:
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
                # No MM tags found — check for IUPAC R/Y encoding
                if iupac_count > 0 and st_count > 0:
                    return 'daf'
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

    tags = {}
    for t in ('MM', 'Mm', 'ML', 'Ml', 'st'):
        if read.has_tag(t):
            val = read.get_tag(t)
            if t in ('ML', 'Ml'):
                # array.array('B', ...) → bytes via buffer protocol: fast memcpy,
                # avoids ~5000 PyInt allocations per Hia5 PacBio read.
                val = compact_ml_value(val)
            tags[t] = val

    payload = {
        'query_name': read.query_name,
        'query_sequence': seq,
        'is_reverse': read.is_reverse,
        'tags': tags,
    }

    # DAF MD-fallback precomputation (live-read side, before slim-IPC handoff).
    if mode == 'daf':
        from fiberhmm.core.bam_reader import has_iupac_encoding
        if not has_iupac_encoding(seq):
            from fiberhmm.daf.encoder import get_daf_positions
            md_res = get_daf_positions(read, ref_fasta=ref_fasta)
            if md_res is not None:
                payload['_daf_md_result'] = md_res   # (ct_list, ga_list, strand_tag)

    return payload


def extract_fiber_read_from_payload(payload: dict, mode: str, prob_threshold: int) -> Optional[dict]:
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


def _extract_fiber_read_from_pysam(read, mode: str, prob_threshold: int,
                                    ref_fasta=None) -> Optional[dict]:
    """Extract minimal data needed for HMM processing from a pysam read."""
    query_sequence = read.query_sequence
    if not query_sequence:
        return None

    # IUPAC R/Y branch: DAF-seq reads with deamination encoded in the sequence
    if mode == 'daf' and has_iupac_encoding(query_sequence):
        st_tag = read.get_tag('st') if read.has_tag('st') else None
        mod_positions, strand, conv_seq = extract_daf_iupac_positions(query_sequence, st_tag)
        if not mod_positions:
            return None
        return {
            'read_id': read.query_name,
            'query_sequence': conv_seq,       # Y→T, R→A (pure ACGT)
            'm6a_query_positions': mod_positions,
            'query_length': len(conv_seq),
            '_daf_strand': strand,            # pre-computed from st tag
        }

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
        md_result = getattr(read, '_daf_md_result', None)
        if md_result is None and hasattr(read, 'get_aligned_pairs'):
            from fiberhmm.daf.encoder import get_daf_positions
            md_result = get_daf_positions(read, ref_fasta=ref_fasta)
        if md_result is not None:
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
            # query_sequence is already raw ACGT (no R/Y to decode);
            # uppercase to match what extract_daf_iupac_positions emits.
            daf_strand = daf_strand_from_tag(strand_tag)
            return {
                'read_id': read.query_name,
                'query_sequence': query_sequence.upper(),
                'm6a_query_positions': mod_positions,
                'query_length': len(query_sequence),
                '_daf_strand': daf_strand if daf_strand != '.' else '-',
            }

    # Legacy MM/ML path: use the fast vectorized parser instead of
    # read.modified_bases.  pysam's modified_bases returns a dict of
    # (base, strand, mod_code) -> [(pos, qual), ...] which forces a Python
    # iteration over every modification (~5000 per Hia5 PacBio read =
    # ~5-10 ms/read).  parse_mm_tag_query_positions does the same parse in
    # vectorized numpy and accepts ML as bytes (no PyInt materialization).
    mm_tag = get_preferred_tag(read, 'MM', 'Mm', '')
    ml_raw = get_preferred_tag(read, 'ML', 'Ml', None)

    if not mm_tag or ml_raw is None:
        return None

    # Empty-ML guard (also avoids the parser doing real work for nothing).
    try:
        if len(ml_raw) == 0:
            return None
    except TypeError:
        pass

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
    if mode == 'daf':
        strand = fiber_read.get('_daf_strand') or detect_daf_strand(query_sequence, m6a_positions)
    elif mode == 'nanopore-fiber':
        strand = '.'  # No strand detection for nanopore
    else:
        strand = '.'

    # Encode — pass is_reverse so nanopore mode handles strand correctly.
    # Circular mode keeps the 3x tiling private to inference and projects the
    # middle copy back to molecule coordinates before anything is written.
    is_reverse = fiber_read.get('is_reverse', False)
    encode_sequence = query_sequence
    encode_mods = m6a_positions
    circular_read_length = None
    if circular and len(query_sequence) > 0:
        encode_sequence, encode_mods = tile_sequence_and_mods(query_sequence, m6a_positions)
        circular_read_length = len(query_sequence)

    encoded = encode_from_query_sequence(
        encode_sequence, encode_mods, edge_trim,
        mode=mode, strand=strand, context_size=context_size,
        is_reverse=is_reverse,
    )

    if len(encoded) == 0:
        return None

    # Predict
    fp_result = predict_footprints_and_msps(model, encoded, msp_min_size, with_scores,
                                             return_posteriors=return_posteriors,
                                             nuc_min_size=nuc_min_size,
                                             circular_read_length=circular_read_length)

    # If no footprints and we don't need posteriors or encoded, skip
    if len(fp_result['footprint_starts']) == 0 and len(fp_result['msp_starts']) == 0:
        if not return_posteriors and not include_encoded:
            return None

    result = {
        'ns': fp_result['footprint_starts'],
        'nl': fp_result['footprint_sizes'],
        'ns_scores': fp_result.get('footprint_scores'),
        'as': fp_result['msp_starts'],
        'al': fp_result['msp_sizes'],
        'as_scores': fp_result.get('msp_scores')
    }
    if fp_result.get('circular'):
        result.update({
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
        })

    # Include posteriors data if requested
    if return_posteriors and fp_result.get('posteriors') is not None:
        result['posteriors'] = fp_result['posteriors']
        result['strand'] = strand

    # Include encoded obs for the fused recall pass (no re-encoding cost)
    if include_encoded:
        result['encoded'] = encoded
        result['strand'] = strand

    return result
