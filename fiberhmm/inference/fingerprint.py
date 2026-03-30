"""
Footprint fingerprinting: posterior-based splitting of merged TF clusters.

Scans large footprints (state 0 blocks) for internal posterior probability
bumps indicating hidden accessible regions. When found, splits the footprint
and returns modified footprint arrays that feed into MSP computation.

Two-phase algorithm:
  Phase 1: Propose bumps via greedy peak search on P(accessible)
  Phase 2: Evaluate all resulting fragments with safety guards
"""

import numpy as np
from typing import List, Tuple, Optional


def _find_bumps(p_accessible: np.ndarray,
                peak_threshold: float = 0.10,
                local_auc_threshold: float = 1.5,
                baseline_threshold: float = 0.01,
                max_bumps: int = 10) -> List[Tuple[int, int]]:
    """
    Phase 1: Find posterior bumps in a P(accessible) trace.

    Greedy peak search: find highest peak, walk to boundaries,
    check local AUC, zero out, repeat.

    Args:
        p_accessible: P(accessible) array for one footprint region
        peak_threshold: minimum peak height to consider (avoids noise)
        local_auc_threshold: minimum sum of P(accessible) within bump
        baseline_threshold: P(accessible) below this = solid footprint
        max_bumps: maximum bumps to find per footprint

    Returns:
        List of (left, right) bump boundaries (relative to slice start).
        right is exclusive: the bump spans [left, right).
    """
    p = p_accessible.copy()
    bumps = []

    for _ in range(max_bumps):
        peak_idx = np.argmax(p)
        if p[peak_idx] < peak_threshold:
            break

        # Walk left from peak until below baseline
        l = peak_idx
        while l > 0 and p[l - 1] > baseline_threshold:
            l -= 1

        # Walk right from peak until below baseline
        r = peak_idx
        while r < len(p) - 1 and p[r + 1] > baseline_threshold:
            r += 1

        # Local AUC (between walked boundaries only)
        local_auc = np.sum(p[l:r + 1])
        if local_auc >= local_auc_threshold:
            bumps.append((l, r + 1))  # right is exclusive

        # Zero out bump so argmax finds next peak
        p[l:r + 1] = 0.0

    return bumps


def _evaluate_fragments(fp_size: int,
                        bumps: List[Tuple[int, int]],
                        min_sub_fp_size: int = 50,
                        nuc_min: int = 120,
                        nuc_max: int = 180) -> bool:
    """
    Phase 2: Evaluate whether proposed splits produce valid fragments.

    Guards:
      - Splinter guard: all fragments must be >= min_sub_fp_size
      - Poly-nucleosome guard: if ALL fragments are nucleosome-sized, abort

    Args:
        fp_size: original footprint size
        bumps: sorted list of (left, right) bump boundaries
        min_sub_fp_size: minimum allowed fragment size
        nuc_min/nuc_max: nucleosome size range

    Returns:
        True if splits should be committed, False to abort.
    """
    # Compute fragment sizes
    fragments = []
    pos = 0
    for l, r in bumps:
        fragments.append(l - pos)
        pos = r
    fragments.append(fp_size - pos)

    # Splinter guard: any fragment too small?
    if any(f < min_sub_fp_size for f in fragments):
        return False

    # Poly-nucleosome guard: all fragments nucleosome-sized?
    if all(nuc_min <= f <= nuc_max for f in fragments):
        return False

    return True


def fingerprint_footprints(
    fp_starts: np.ndarray,
    fp_ends: np.ndarray,
    posteriors_full: np.ndarray,
    min_footprint_size: int = 200,
    peak_threshold: float = 0.10,
    local_auc_threshold: float = 1.5,
    baseline_threshold: float = 0.01,
    min_sub_fp_size: int = 50,
    nuc_min: int = 120,
    nuc_max: int = 180,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Scan footprints for internal posterior bumps and split if found.

    Two-phase algorithm:
      Phase 1: Propose bumps via greedy peak search
      Phase 2: Evaluate resulting fragments with safety guards

    Args:
        fp_starts: footprint start positions (int32)
        fp_ends: footprint end positions (int32)
        posteriors_full: (T, 2) posterior matrix from Forward-Backward
        min_footprint_size: only scan footprints >= this size
        peak_threshold: minimum P(accessible) peak to consider
        local_auc_threshold: minimum local AUC to trigger split
        baseline_threshold: P(accessible) below this = solid footprint
        min_sub_fp_size: minimum resulting fragment size
        nuc_min/nuc_max: nucleosome size range for poly-nuc guard

    Returns:
        (new_fp_starts, new_fp_ends, split_count)
        Modified footprint arrays with splits applied.
    """
    if len(fp_starts) == 0:
        return fp_starts, fp_ends, 0

    new_starts = []
    new_ends = []
    split_count = 0

    for i in range(len(fp_starts)):
        start = int(fp_starts[i])
        end = int(fp_ends[i])
        fp_size = end - start

        # Skip small footprints
        if fp_size < min_footprint_size:
            new_starts.append(start)
            new_ends.append(end)
            continue

        # Phase 1: Propose bumps
        p_accessible = posteriors_full[start:end, 1]
        bumps = _find_bumps(
            p_accessible,
            peak_threshold=peak_threshold,
            local_auc_threshold=local_auc_threshold,
            baseline_threshold=baseline_threshold,
        )

        if not bumps:
            new_starts.append(start)
            new_ends.append(end)
            continue

        # Sort bumps left-to-right
        bumps.sort(key=lambda x: x[0])

        # Phase 2: Evaluate fragments
        if not _evaluate_fragments(fp_size, bumps,
                                   min_sub_fp_size=min_sub_fp_size,
                                   nuc_min=nuc_min, nuc_max=nuc_max):
            new_starts.append(start)
            new_ends.append(end)
            continue

        # Commit splits
        pos = start
        for l, r in bumps:
            new_starts.append(pos)
            new_ends.append(start + l)
            pos = start + r
        # Trailing fragment
        new_starts.append(pos)
        new_ends.append(end)
        split_count += 1

    return (
        np.array(new_starts, dtype=np.int32),
        np.array(new_ends, dtype=np.int32),
        split_count,
    )
