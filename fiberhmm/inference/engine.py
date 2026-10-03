"""FiberHMM core per-read HMM inference engine."""

from typing import Optional, Tuple

import os

import numpy as np
import pysam

from fiberhmm.core.bam_reader import (
    detect_daf_strand,
    encode_from_query_sequence,
    extract_daf_iupac_positions,
    has_iupac_encoding,
    parse_mm_tag_query_calls,
)
from fiberhmm.core.hmm import FiberHMM
from fiberhmm.inference.circular import (
    project_center_runs,
    project_center_scores,
    split_intervals_for_legacy,
    tile_sequence_and_mods,
)

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


def _footprint_runs(states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if _HAS_NUMBA:
        return _footprint_runs_numba(states)

    states_padded = np.concatenate([[1], states, [1]])
    diff = np.diff(states_padded)
    return np.where(diff == -1)[0], np.where(diff == 1)[0]


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
        scores = np.zeros(len(starts), dtype=np.float32)
        for i, (s, e) in enumerate(zip(starts, ends)):
            # Mean posterior probability for footprint state within this footprint
            scores[i] = np.mean(confidence[s:e])

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

    fp_starts, fp_ends = _footprint_runs(states)

    if len(fp_starts) > 0:
        result['footprint_starts'] = fp_starts.astype(np.int32)
        result['footprint_sizes'] = (fp_ends - fp_starts).astype(np.int32)

        if with_scores and confidence is not None:
            fp_scores = np.zeros(len(fp_starts), dtype=np.float32)
            for i, (s, e) in enumerate(zip(fp_starts, fp_ends)):
                fp_scores[i] = np.mean(confidence[s:e])
            result['footprint_scores'] = fp_scores

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
            if nuc_ends[-1] < len(states):
                msp_start_list.append(int(nuc_ends[-1]))
                msp_size_list.append(len(states) - int(nuc_ends[-1]))
        else:
            # No nucleosome-sized footprints: entire read is one MSP
            msp_start_list.append(0)
            msp_size_list.append(len(states))

        if msp_start_list:
            msp_starts_arr = np.array(msp_start_list, dtype=np.int32)
            msp_sizes_arr = np.array(msp_size_list, dtype=np.int32)

            size_mask = msp_sizes_arr >= msp_min_size
            msp_starts_arr = msp_starts_arr[size_mask]
            msp_sizes_arr = msp_sizes_arr[size_mask]

            if len(msp_starts_arr) > 0:
                result['msp_starts'] = msp_starts_arr
                result['msp_sizes'] = msp_sizes_arr

                if with_scores and confidence is not None:
                    # aq = mean P(accessible) over the MSP. ``confidence`` is
                    # the posterior of the decoded state, so P(accessible) is
                    # ``confidence`` at accessible (state 1) positions and
                    # ``1 - confidence`` inside small footprints the MSP spans.
                    p_accessible = np.where(states == 1, confidence, 1.0 - confidence)
                    msp_scores = np.zeros(len(msp_starts_arr), dtype=np.float32)
                    for i, (s, sz) in enumerate(zip(msp_starts_arr, msp_sizes_arr)):
                        msp_scores[i] = np.mean(p_accessible[s:s+sz])
                    result['msp_scores'] = msp_scores

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
            'msp_scores': per-MSP mean P(accessible) (if with_scores)
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
                    try:
                        mm_tag = read.get_tag('MM') if read.has_tag('MM') else \
                                 read.get_tag('Mm') if read.has_tag('Mm') else None
                    except KeyError:
                        mm_tag = None

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


_DAF_SNP_MASK = {}


def configure_daf_snp_mask(mask_path=None) -> None:
    """Load an optional 0-based BED mask once in each inference process."""
    global _DAF_SNP_MASK
    if not mask_path:
        _DAF_SNP_MASK = {}
        return
    from fiberhmm.daf.snps import load_snp_mask

    _DAF_SNP_MASK = load_snp_mask(mask_path)


# Context-local override of the process-wide mask (``daf_snp_mask_scope``),
# for callers such as QC that run beside calling in one process.
from contextvars import ContextVar as _ContextVar
_DAF_SNP_MASK_SCOPE = _ContextVar('fiberhmm_daf_snp_mask_scope', default=None)


def daf_snp_mask_scope(mask):
    """Context manager: use ``mask`` ({chrom: set(ref positions)}, possibly
    empty) instead of the process-wide DAF SNP mask in the current context."""
    from contextlib import contextmanager

    @contextmanager
    def scope():
        token = _DAF_SNP_MASK_SCOPE.set(dict(mask or {}))
        try:
            yield
        finally:
            _DAF_SNP_MASK_SCOPE.reset(token)
    return scope()


def _daf_reference_mask(read):
    scoped = _DAF_SNP_MASK_SCOPE.get()
    active = _DAF_SNP_MASK if scoped is None else scoped
    sites = active.get(getattr(read, "reference_name", None), set())
    if not sites:
        return sites
    # A record running past its contig end (a circular origin) reaches the
    # masked sites again at p + LN.
    from fiberhmm.daf.snps import wrapped_reference_sites
    return wrapped_reference_sites(read, sites)


# DAF unaligned-base mask. Deamination evidence is a read-versus-reference
# comparison on matched pairs, so query bases with no reference counterpart
# (CIGAR I insertions, S soft clips) can never carry a mark; without the mask
# their unconverted C/G would count as unmodified (protected) targets and an
# insertion or clip would be called nucleosome-packed. Masked like SNP sites:
# no evidence either way. On by default; mirrored into the environment so
# spawned workers inherit an opt-out.
_DAF_UNALIGNED_MASK_ENV = 'FIBERHMM_DAF_MASK_UNALIGNED'
_DAF_UNALIGNED_MASK = None


def configure_daf_unaligned_mask(enabled: bool = True) -> None:
    """Mask (default) or keep DAF query bases with no reference counterpart."""
    global _DAF_UNALIGNED_MASK
    _DAF_UNALIGNED_MASK = bool(enabled)
    os.environ[_DAF_UNALIGNED_MASK_ENV] = '1' if enabled else '0'


def daf_unaligned_mask_enabled() -> bool:
    global _DAF_UNALIGNED_MASK
    if _DAF_UNALIGNED_MASK is None:
        _DAF_UNALIGNED_MASK = os.environ.get(_DAF_UNALIGNED_MASK_ENV, '1') != '0'
    return _DAF_UNALIGNED_MASK


def daf_unaligned_query_positions(read):
    """SEQ positions of ``read``'s CIGAR I/S bases (empty when the mask is off,
    for unmapped records and for slim stubs without a CIGAR)."""
    if not daf_unaligned_mask_enabled():
        return set()
    from fiberhmm.daf.aligned_arrays import unaligned_query_positions
    try:
        cigar = getattr(read, 'cigartuples', None)
    except (ValueError, TypeError):
        return set()
    if not cigar:
        return set()
    seq = getattr(read, 'query_sequence', None)
    return unaligned_query_positions(cigar, len(seq) if seq else None)


def _daf_snp_masked_query_positions(read):
    reference_mask = _daf_reference_mask(read)
    if not reference_mask or not hasattr(read, "get_aligned_pairs"):
        return set()
    try:
        return {
            int(query_position)
            for query_position, reference_position in read.get_aligned_pairs()
            if query_position is not None
            and reference_position is not None
            and reference_position in reference_mask
        }
    except (ValueError, TypeError, IndexError):
        return set()


# Insert evidence (fiberhmm.daf.insert_consensus): per-record deaminations of
# inserted bases re-encoded against a local consensus of the insertion's
# carriers, keyed by insert_consensus.record_key. Empty unless a consensus
# pre-pass configured it.
_DAF_INSERT_EVIDENCE: dict = {}


def configure_daf_insert_evidence(evidence=None) -> None:
    """Load per-record insert evidence (a dict, or a pickle path) once per process."""
    global _DAF_INSERT_EVIDENCE
    if evidence is None:
        _DAF_INSERT_EVIDENCE = {}
        return
    if isinstance(evidence, (str, bytes, os.PathLike)):
        import pickle
        with open(evidence, 'rb') as handle:
            evidence = pickle.load(handle)
    _DAF_INSERT_EVIDENCE = dict(evidence)


def daf_insert_evidence(read):
    """The record's insert evidence, or None."""
    if not _DAF_INSERT_EVIDENCE or not daf_unaligned_mask_enabled():
        return None
    try:
        key = (read.query_name, int(read.flag) & ~0x400, int(read.reference_id),
               int(read.reference_start))
    except (AttributeError, TypeError, ValueError):
        return None
    return _DAF_INSERT_EVIDENCE.get(key)


def _daf_excluded_query_positions(read):
    """DAF query positions that carry no evidence: SNP-masked matched bases
    plus (unless disabled) unaligned I/S bases. Callers drop them from the
    deamination calls AND encode them as unknown, so a masked unconverted
    C/G is not counted as an unmodified (protected) target. Inserted bases a
    local insert consensus gives evidence for are not excluded."""
    excluded = _daf_snp_masked_query_positions(read)
    unaligned = daf_unaligned_query_positions(read)
    if unaligned:
        evidence = daf_insert_evidence(read)
        if evidence is not None:
            unaligned = unaligned - evidence.known
        excluded = excluded | unaligned if excluded else unaligned
    return excluded


def daf_unaligned_without_evidence(read):
    """Unaligned (I/S) SEQ positions minus those an insert consensus covers."""
    unaligned = daf_unaligned_query_positions(read)
    if unaligned:
        evidence = daf_insert_evidence(read)
        if evidence is not None:
            unaligned = unaligned - evidence.known
    return unaligned


def _insert_evidence_parts(read):
    """``(mods, strand, inserted)`` of a read's insert evidence, or None.

    ``inserted``: every CIGAR-insertion SEQ position, whose own marks the
    consensus replaces. Slim stubs carry the tuple precomputed."""
    parts = getattr(read, '_daf_insert_mods', None)
    if parts is not None:
        return parts
    evidence = daf_insert_evidence(read)
    if evidence is None:
        return None
    from fiberhmm.daf.insert_consensus import inserted_query_positions
    return (set(evidence.mods), evidence.strand,
            inserted_query_positions(getattr(read, 'cigartuples', None)))


def _merge_insert_mods(read, mods, strand_tag):
    """Replace the marks of a read's inserted bases with the insert-consensus
    deaminations (when it has evidence on ``strand_tag``, 'CT'/'GA'/'+'/'-').
    Returns the updated set."""
    parts = _insert_evidence_parts(read)
    if parts is None:
        return mods
    insert_mods, strand, inserted = parts
    mods = set(mods) - set(inserted)
    wanted = 0 if strand_tag in ('CT', '+') else 1
    if strand == wanted:
        mods |= set(insert_mods)
    return mods


def _insert_evidence_strand(read):
    """'CT'/'GA' of a read's insert evidence (when its own aligned bases give
    no deamination to decide the strand), or None."""
    parts = _insert_evidence_parts(read)
    if parts is None or not parts[0]:
        return None
    return 'CT' if parts[1] == 0 else 'GA'


def _daf_insert_mods(read, strand_tag):
    """Insert-consensus deaminations for a read called on ``strand_tag``."""
    parts = _insert_evidence_parts(read)
    if parts is None:
        return set()
    wanted = 0 if strand_tag in ('CT', '+') else 1
    return set(parts[0]) if parts[1] == wanted else set()


def _has_mm_tag(read):
    """True when the read has usable MM/ML deamination calls (both tags,
    non-empty), the condition of the MM/ML branch."""
    try:
        mm = read.get_tag('MM') if read.has_tag('MM') else (
            read.get_tag('Mm') if read.has_tag('Mm') else '')
        ml = read.get_tag('ML') if read.has_tag('ML') else (
            read.get_tag('Ml') if read.has_tag('Ml') else None)
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    return bool(mm) and ml is not None and len(ml) > 0


def _evidence_marks(read, marks, excluded=()):
    """``marks`` on bases that carry the read's own evidence: not masked
    (``excluded``) and not inserted bases an insert consensus re-encodes.
    The strand is decided on these, by the same rule as the insert
    pre-pass (``insert_consensus._read_strand``), so the consensus evidence
    of a carrier is always on the strand it is called on."""
    out = set(marks)
    if excluded:
        out -= set(excluded)
    parts = _insert_evidence_parts(read)
    if parts is not None:
        out -= set(parts[2])
    return out


def daf_iupac_strand(read, sequence, st_tag, strand, excluded=()):
    """Strand of an R/Y read. With an ``st`` tag, ``strand`` (from it);
    otherwise the majority of the R/Y marks on bases that carry evidence
    (masked and re-encoded inserted marks do not vote), and on a tie the
    insert-consensus strand ('+'/'-'/'.')."""
    if st_tag is not None:
        return strand
    upper = sequence.upper()
    y = r = 0
    for position in _evidence_marks(
            read, (i for i, base in enumerate(upper) if base == 'Y' or base == 'R'),
            excluded):
        if upper[position] == 'Y':
            y += 1
        else:
            r += 1
    if y != r:
        return '+' if y > r else '-'
    fallback = _insert_evidence_strand(read)
    if fallback is None:
        return '.'
    return '+' if fallback == 'CT' else '-'


def daf_no_call_blocks(read):
    """Long unaligned (I/S) SEQ spans of a live DAF ``read`` from which calls
    are removed (:mod:`fiberhmm.inference.no_evidence`)."""
    if not daf_unaligned_mask_enabled():
        return []
    try:
        cigar = getattr(read, 'cigartuples', None)
    except (ValueError, TypeError):
        return []
    if not cigar:
        return []
    from fiberhmm.inference.no_evidence import unaligned_blocks
    seq = getattr(read, 'query_sequence', None)
    blocks = unaligned_blocks(cigar, query_length=len(seq) if seq else None)
    evidence = daf_insert_evidence(read)
    if blocks and evidence is not None and evidence.known:
        # Stretches the insert consensus covers are evidence, not blocks; what
        # remains uncovered (>= the block length) stays uncalled.
        from fiberhmm.inference.no_evidence import NO_CALL_MIN_BLOCK
        kept = []
        for start, end in blocks:
            run = None
            for pos in range(start, end + 1):
                free = pos < end and pos not in evidence.known
                if free and run is None:
                    run = pos
                elif not free and run is not None:
                    if pos - run >= NO_CALL_MIN_BLOCK:
                        kept.append((run, pos))
                    run = None
        blocks = kept
    return blocks


def read_no_call_blocks(read, mode):
    """SEQ spans of ``read`` that get no calls: a DAF read's long unaligned
    blocks, and the soft clips of any supplementary record (those bases are
    the primary record's; a supplementary record is called on its aligned
    part only). Slim stubs carry the spans precomputed by their producer."""
    blocks = getattr(read, '_no_call_blocks', None)
    if blocks is not None:
        return list(blocks)
    out = list(daf_no_call_blocks(read)) if mode == 'daf' else []
    if getattr(read, 'is_supplementary', False):
        from fiberhmm.inference.no_evidence import supplementary_clip_blocks
        out += supplementary_clip_blocks(read)
    if len(out) > 1:
        from fiberhmm.inference.no_evidence import merge_blocks
        out = merge_blocks(out)
    return out


def _stub_or_live_excluded(read):
    excluded = getattr(read, '_daf_excluded_query_positions', None)
    if excluded is None:
        excluded = _daf_excluded_query_positions(read)
    return excluded


class _ApplyPayloadRead:
    """Minimal duck-type for pysam.AlignedSegment used inside apply workers.

    _extract_fiber_read_from_pysam only reads .query_name, .query_sequence,
    .is_reverse, .has_tag(t), .get_tag(t) — the same surface this class
    exposes — so it works unchanged on either a real pysam segment (in main)
    or this slim payload wrapper (in worker).

    For the DAF MD-fallback path (--mode daf with raw input), the producer
    side (``make_apply_payload``) computes ``_daf_md_result`` from
    ``read.get_aligned_pairs(with_seq=True)`` once, and stashes it on the
    stub. The MD branch in _extract_fiber_read_from_pysam reads it back
    from there instead of trying to call get_aligned_pairs (which the stub
    does not implement).
    """
    __slots__ = ('query_name', 'query_sequence', 'is_reverse', '_tags',
                 '_daf_md_result', '_daf_excluded_query_positions',
                 '_no_call_blocks', '_daf_insert_mods')

    def __init__(self, query_name, query_sequence, is_reverse, tags,
                 daf_md_result=None, daf_excluded_query_positions=None,
                 no_call_blocks=None, daf_insert_mods=None):
        self.query_name = query_name
        self.query_sequence = query_sequence
        self.is_reverse = is_reverse
        self._tags = tags
        self._daf_md_result = daf_md_result
        self._daf_excluded_query_positions = set(
            daf_excluded_query_positions or ()
        )
        self._no_call_blocks = list(no_call_blocks or ())
        self._daf_insert_mods = daf_insert_mods

    def has_tag(self, t):
        return t in self._tags

    def get_tag(self, t):
        return self._tags[t]


def make_apply_payload(read, mode: str = 'fiber', ref_fasta=None,
                       include_ddda_mcg: bool = False) -> Optional[dict]:
    """Extract slim payload from a pysam read for the apply slim-IPC path.

    Runs in the *main* process.  Does NOT parse MM/ML — that moves to the
    worker via extract_fiber_read_from_payload().

    When ``mode == 'daf'`` and the read carries no R/Y IUPAC encoding,
    additionally pre-computes the MD-derived deamination positions so the
    worker can run the HMM directly on the raw BAM without an upstream
    ``fiberhmm-daf-encode`` pass. The MD walk has to happen here because
    the slim ``_ApplyPayloadRead`` stub in the worker has no access to the
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
                try:
                    val = bytes(val)
                except TypeError:
                    pass
            tags[t] = val

    payload = {
        'query_name': read.query_name,
        'query_sequence': seq,
        'is_reverse': read.is_reverse,
        'tags': tags,
    }

    if include_ddda_mcg and (mode != 'daf' or ref_fasta is None):
        raise ValueError(
            "include_ddda_mcg requires DAF mode and an open reference FASTA"
        )

    # DAF MD-fallback precomputation (live-read side, before slim-IPC handoff).
    md_res = None
    if mode == 'daf':
        from fiberhmm.core.bam_reader import has_iupac_encoding
        excluded_query_positions = _daf_excluded_query_positions(read)
        if excluded_query_positions:
            payload['_daf_excluded_query_positions'] = excluded_query_positions
        insert_parts = _insert_evidence_parts(read)
        if insert_parts is not None:
            payload['_daf_insert_mods'] = insert_parts
        if not has_iupac_encoding(seq):
            from fiberhmm.daf.encoder import get_daf_positions
            md_res = get_daf_positions(
                read,
                ref_fasta=ref_fasta,
                excluded_reference_positions=_daf_reference_mask(read),
            )
            if (md_res is None and not _has_mm_tag(read)
                    and _insert_evidence_strand(read) is not None):
                # No deamination on the aligned bases: the insert consensus
                # decides the strand.
                md_res = ([], [], _insert_evidence_strand(read))
            if md_res is not None:
                payload['_daf_md_result'] = md_res   # (ct_list, ga_list, strand_tag)
        elif _DAF_CHIMERA_CFG['filter']:
            # R/Y input carries only the dominant flavour as IUPAC; the other
            # flavour's C->T / G->A mismatches are still raw bases, visible
            # through MD (or the reference). The worker needs both lists for
            # the strand-swap chimera filter.
            rest = _daf_raw_mismatch_lists(read, ref_fasta)
            if rest is not None:
                payload['_daf_md_result'] = rest

    no_call_blocks = read_no_call_blocks(read, mode)
    if no_call_blocks:
        payload['_no_call_blocks'] = no_call_blocks

    if mode == 'daf' and read.has_tag('MA'):
        # DddA CpG-aware recall reads the molecule's own tag-m5c island calls
        # (ddda_ucg exempts CpGs from masking); the worker only sees this
        # payload, so the SEQ-frame intervals travel with it.
        from fiberhmm.inference.tf_recaller import read_cpg_intervals
        cpg_intervals = read_cpg_intervals(read)
        if cpg_intervals['ucg'] or cpg_intervals['mcg']:
            payload['_cpg_ma_intervals'] = cpg_intervals

    if include_ddda_mcg:
        from fiberhmm.daf.m5c import build_ddda_mcg_observation_payload
        observations = build_ddda_mcg_observation_payload(
            read, ref_fasta, daf_result=md_res,
        )
        if observations:
            payload['_ddda_mcg_observations'] = observations

    return payload


def extract_fiber_read_from_payload(payload: dict, mode: str, prob_threshold: int) -> Optional[dict]:
    """Worker-side: turn a slim payload into a fiber_read dict.

    Equivalent to _extract_fiber_read_from_pysam(real_read, mode, prob_threshold)
    but takes the slim payload built by make_apply_payload().  Returns None
    on read with no usable modification data (no MM/ML, no IUPAC codes,
    extraction failure) — caller treats result=None as 'no footprints'.
    """
    return _extract_fiber_read_from_pysam(
        _ApplyPayloadRead(
            payload['query_name'], payload['query_sequence'],
            payload['is_reverse'], payload['tags'],
            daf_md_result=payload.get('_daf_md_result'),
            daf_excluded_query_positions=payload.get(
                '_daf_excluded_query_positions'
            ),
            no_call_blocks=payload.get('_no_call_blocks'),
            daf_insert_mods=payload.get('_daf_insert_mods'),
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


def _daf_raw_mismatch_lists(read, ref_fasta=None):
    """Raw (not IUPAC-encoded) C->T / G->A query mismatches of a live read.

    Returns ``(ct_list, ga_list, 'CT')`` or None when the read has no
    reference evidence (no usable MD and no FASTA). SNP-masked reference
    positions are excluded, as on the MD path.
    """
    if not hasattr(read, 'get_aligned_pairs'):
        return None
    if ref_fasta is None and not read.has_tag('MD'):
        return None
    from fiberhmm.daf.encoder import get_daf_positions
    try:
        return get_daf_positions(
            read,
            force_strand='CT',   # always return both lists; strand unused
            ref_fasta=ref_fasta,
            excluded_reference_positions=_daf_reference_mask(read),
        )
    except Exception:
        return None


def _is_iupac_daf_chimera(read, query_sequence: str, excluded_query_positions,
                          ref_fasta=None) -> bool:
    """Strand-swap chimera test for an R/Y-encoded DAF read.

    CT events are Y bases plus any raw C->T mismatches; GA events are R bases
    plus raw G->A mismatches (from MD/reference, when available). SNP-masked
    positions are removed from both, matching the MD path.
    """
    from fiberhmm.daf.encoder import is_daf_chimera
    min_seg = _DAF_CHIMERA_CFG['min_seg']
    seq_arr = np.frombuffer(query_sequence.upper().encode('ascii'), dtype=np.uint8)
    ct = set(np.flatnonzero(seq_arr == ord('Y')).tolist())
    ga = set(np.flatnonzero(seq_arr == ord('R')).tolist())
    if max(len(ct), len(ga)) < min_seg:
        return False
    rest = getattr(read, '_daf_md_result', None)
    if rest is None:
        rest = _daf_raw_mismatch_lists(read, ref_fasta)
    if rest is not None:
        ct.update(rest[0])
        ga.update(rest[1])
    if excluded_query_positions:
        ct.difference_update(excluded_query_positions)
        ga.difference_update(excluded_query_positions)
    return is_daf_chimera(sorted(ct), sorted(ga),
                          min_seg_events=min_seg,
                          purity=_DAF_CHIMERA_CFG['purity'])


def _with_unknown(fiber_read: dict, unknown_positions, read=None) -> dict:
    """Attach no-evidence positions (encoded as non-target) and the long
    no-call blocks of a DAF read to a fiber_read."""
    if unknown_positions:
        fiber_read['unknown_query_positions'] = set(unknown_positions)
    if read is not None:
        blocks = read_no_call_blocks(read, 'daf')
        if blocks:
            fiber_read['no_call_blocks'] = blocks
    return fiber_read


def _extract_fiber_read_from_pysam(read, mode: str, prob_threshold: int,
                                    ref_fasta=None) -> Optional[dict]:
    """Extract minimal data needed for HMM processing from a pysam read.

    Returns a fiber_read dict, None (no usable evidence), or
    :data:`CHIMERA_SKIP` (DAF strand-swap chimera, when the filter is on).
    """
    query_sequence = read.query_sequence
    if not query_sequence:
        return None

    # IUPAC R/Y branch: DAF-seq reads with deamination encoded in the sequence
    if mode == 'daf' and has_iupac_encoding(query_sequence):
        st_tag = read.get_tag('st') if read.has_tag('st') else None
        mod_positions, strand, conv_seq = extract_daf_iupac_positions(query_sequence, st_tag)
        excluded_query_positions = _stub_or_live_excluded(read)
        strand = daf_iupac_strand(read, query_sequence, st_tag, strand,
                                  excluded_query_positions)
        # Strand-swap chimera filter, same policy as the MD path below.
        if _DAF_CHIMERA_CFG['filter'] and _is_iupac_daf_chimera(
                read, query_sequence, excluded_query_positions, ref_fasta):
            return CHIMERA_SKIP
        mod_positions.difference_update(excluded_query_positions)
        mod_positions = _merge_insert_mods(read, mod_positions, strand)
        if not mod_positions:
            return None
        return _with_unknown({
            'read_id': read.query_name,
            'query_sequence': conv_seq,       # Y→T, R→A (pure ACGT)
            'm6a_query_positions': mod_positions,
            'query_length': len(conv_seq),
            '_daf_strand': strand,            # pre-computed from st tag
        }, excluded_query_positions, read)

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
            md_result = get_daf_positions(
                read,
                ref_fasta=ref_fasta,
                excluded_reference_positions=_daf_reference_mask(read),
            )
            if (md_result is None and not _has_mm_tag(read)
                    and _insert_evidence_strand(read) is not None):
                md_result = ([], [], _insert_evidence_strand(read))
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
            # SNP-masked sites are already out of the mismatch lists (they
            # were excluded by reference position); the same set, plus the
            # unaligned bases, is encoded as no evidence below.
            excluded_query_positions = _stub_or_live_excluded(read)
            mod_positions.difference_update(excluded_query_positions)
            mod_positions = _merge_insert_mods(read, mod_positions, strand_tag)
            if not mod_positions:
                return None
            # query_sequence is already raw ACGT (no R/Y to decode);
            # uppercase to match what extract_daf_iupac_positions emits.
            return _with_unknown({
                'read_id': read.query_name,
                'query_sequence': query_sequence.upper(),
                'm6a_query_positions': mod_positions,
                'query_length': len(query_sequence),
                '_daf_strand': '+' if strand_tag == 'CT' else '-',
            }, excluded_query_positions, read)

    # Legacy MM/ML path: use the fast vectorized parser instead of
    # read.modified_bases.  pysam's modified_bases returns a dict of
    # (base, strand, mod_code) -> [(pos, qual), ...] which forces a Python
    # iteration over every modification (~5000 per Hia5 PacBio read =
    # ~5-10 ms/read).  parse_mm_tag_query_positions does the same parse in
    # vectorized numpy and accepts ML as bytes (no PyInt materialization).
    try:
        mm_tag = read.get_tag('MM') if read.has_tag('MM') else (
                  read.get_tag('Mm') if read.has_tag('Mm') else '')
    except KeyError:
        mm_tag = ''
    try:
        ml_raw = read.get_tag('ML') if read.has_tag('ML') else (
                  read.get_tag('Ml') if read.has_tag('Ml') else None)
    except KeyError:
        ml_raw = None

    if not mm_tag or ml_raw is None:
        return None

    # Empty-ML guard (also avoids the parser doing real work for nothing).
    try:
        if len(ml_raw) == 0:
            return None
    except TypeError:
        pass

    # Convert ML to bytes once (fast memcpy, no PyInt allocations).
    try:
        ml_bytes = bytes(ml_raw)
    except TypeError:
        ml_bytes = ml_raw

    try:
        mod_pos_set, unknown_pos_set = parse_mm_tag_query_calls(
            mm_tag, ml_bytes, query_sequence, read.is_reverse,
            prob_threshold=prob_threshold, mode=mode,
        )
    except Exception:
        return None

    if mode == 'daf':
        # The SNP and unaligned-base masks apply to MM/ML-native deamination
        # calls too (DAF evidence is a reference comparison whatever its
        # carrier): masked bases are dropped and encoded as no evidence.
        excluded_query_positions = _stub_or_live_excluded(read)
        if excluded_query_positions:
            mod_pos_set.difference_update(excluded_query_positions)
            unknown_pos_set = set(unknown_pos_set) | set(excluded_query_positions)
        if _insert_evidence_parts(read) is not None:
            strand = detect_daf_strand(query_sequence, _evidence_marks(read, mod_pos_set))
            if strand == '.':
                strand = _insert_evidence_strand(read) or '.'
            mod_pos_set = _merge_insert_mods(read, mod_pos_set, strand)
            # Consensus-covered inserted bases are evidence, not '?' unknowns.
            parts = _insert_evidence_parts(read)
            if unknown_pos_set and parts is not None:
                covered = set(parts[2]) - set(excluded_query_positions or ())
                unknown_pos_set = set(unknown_pos_set) - covered

    fiber_read = {
        'read_id': read.query_name,
        'query_sequence': query_sequence,
        'm6a_query_positions': mod_pos_set,
        'query_length': len(query_sequence),
        'is_reverse': bool(read.is_reverse),
    }
    if unknown_pos_set:
        # Bases left unlisted by a '?' MM entry (and, for DAF, masked bases):
        # no call, not "unmodified".
        fiber_read['unknown_query_positions'] = unknown_pos_set
    blocks = read_no_call_blocks(read, mode)
    if blocks:
        fiber_read['no_call_blocks'] = blocks
    return fiber_read


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
    unknown_positions = fiber_read.get('unknown_query_positions')
    circular_read_length = None
    if circular and len(query_sequence) > 0:
        encode_sequence, encode_mods = tile_sequence_and_mods(query_sequence, m6a_positions)
        if unknown_positions:
            unknown_positions = tile_sequence_and_mods(
                query_sequence, unknown_positions)[1]
        circular_read_length = len(query_sequence)

    encoded = encode_from_query_sequence(
        encode_sequence, encode_mods, edge_trim,
        mode=mode, strand=strand, context_size=context_size,
        is_reverse=is_reverse, unknown_positions=unknown_positions,
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

    # DAF: no calls inside long no-evidence (unaligned) blocks.
    no_call_blocks = fiber_read.get('no_call_blocks')
    if no_call_blocks and not result.get('circular'):
        from fiberhmm.inference.no_evidence import suppress_calls_in_blocks
        suppress_calls_in_blocks(result, no_call_blocks)

    # Include posteriors data if requested
    if return_posteriors and fp_result.get('posteriors') is not None:
        result['posteriors'] = fp_result['posteriors']
        result['strand'] = strand

    # Include encoded obs for the fused recall pass (no re-encoding cost)
    if include_encoded:
        result['encoded'] = encoded
        result['strand'] = strand

    return result
