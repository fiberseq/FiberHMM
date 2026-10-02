"""Frozen per-pair reference for the DAF mismatch scans (test-only).

Verbatim copies of the pre-vectorisation implementations at release head
45b9a1a: ``fiberhmm.daf.snps.call_opposite_conversion_snps`` (with its
``_aligned_pairs``/``_profile``/``_consider_profile_site`` helpers),
``fiberhmm.daf.encoder.get_daf_positions`` and the MD branch of
``fiberhmm.cli.extract_tags._deam_positions_list``.  The optimised
production code must reproduce these exactly; see
``tests/test_daf_mismatch_fastpath.py``.

Deliberate departures from the verbatim copies (the old behaviour was
nondeterministic, so it cannot be frozen):

* ``_aligned_pairs`` skips pysam's MD reconstruction when MD does not
  describe the CIGAR (``md_matches_cigar``): for a short MD pysam copies
  undefined memory into the reference string. Such reads use the FASTA or
  are unusable, as in ``get_daf_positions``.
* ``md_matches_cigar`` (used by ``_aligned_pairs``, ``get_daf_positions`` and
  the dedup MD branch) also rejects an MD of the right length whose ``^`` run
  covers a CIGAR insertion; ``pysam_md_walk_ends_short`` replays pysam's
  walk to detect it.
* Where reads' MD tags disagree about a position's base (C vs G), a
  position profiled as both used to keep whichever hypothesis set iteration
  visited last (``PYTHONHASHSEED`` dependent). Every profiled site now keeps
  its base only if at least as many dominant-direction fibers report it as
  report the other C/G base (a tie keeps C), whether or not the other base
  was profiled; both hypotheses are counted where both are.
* Amplicon IDs are renumbered with one complete old->new mapping. The
  verbatim copy renamed them one at a time, so when two amplicons swapped
  places both calls ended up naming the same amplicon (a bug, fixed in
  ``fiberhmm.daf.snps`` too).
"""
from __future__ import annotations

import json
import hashlib
import heapq
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import pysam


_SITE_PROFILE_SEED = 20260824
_SITE_PROFILE_MAX_SITES = 5000
_SITE_PROFILE_MIN_DEPTH = 3
_AMPLICON_BIN_BP = 10_000
_AMPLICON_MIN_READS = 20
_AMPLICON_MIN_BIN_READS = 3

# Canonical production policy selected by the FiberHMM DAF SNP downsampling
# validation.  Keep these values centralized: the Python API and both CLI
# routes import them, while callers can still override every threshold.
VALIDATED_SNP_POLICY_NAME = "bidirectional_five_fiber_v1"
DEFAULT_SNP_MIN_FRACTION = 0.20
DEFAULT_SNP_MIN_DEPTH = 5
DEFAULT_SNP_MIN_ALT_FIBERS = 5


def describe_snp_threshold_policy(
    min_fraction: float,
    min_depth: int,
    min_alt_fibers: int,
) -> dict:
    """Describe whether thresholds match the validated production policy."""
    validated = {
        "min_fraction_each_direction": DEFAULT_SNP_MIN_FRACTION,
        "min_depth_each_direction": DEFAULT_SNP_MIN_DEPTH,
        "min_mismatch_fibers_each_direction": DEFAULT_SNP_MIN_ALT_FIBERS,
        "bidirectional_support_required": True,
    }
    uses_validated_defaults = (
        float(min_fraction) == DEFAULT_SNP_MIN_FRACTION
        and int(min_depth) == DEFAULT_SNP_MIN_DEPTH
        and int(min_alt_fibers) == DEFAULT_SNP_MIN_ALT_FIBERS
    )
    return {
        "name": VALIDATED_SNP_POLICY_NAME if uses_validated_defaults else "custom",
        "uses_validated_defaults": uses_validated_defaults,
        "validated_defaults": validated,
        "validation": "scripts/validation/validate_daf_snp_downsampling.py",
    }


def _site_rank(key: tuple[str, int, str], seed: int) -> int:
    """Stable rank for deterministic bottom-k sampling of genomic sites."""
    digest = hashlib.blake2b(
        f"{seed}|{key[0]}|{key[1]}|{key[2]}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def _consider_profile_site(
    key: tuple[str, int, str],
    heap: list[tuple[int, tuple[str, int, str]]],
    selected: set[tuple[str, int, str]],
    maximum: int,
    seed: int,
) -> None:
    """Maintain a bounded, order-independent sample of reference C/G sites."""
    if key in selected or maximum <= 0:
        return
    rank = _site_rank(key, seed)
    item = (-rank, key)
    if len(heap) < maximum:
        heapq.heappush(heap, item)
        selected.add(key)
    elif rank < -heap[0][0]:
        _old_rank, old_key = heapq.heapreplace(heap, item)
        selected.remove(old_key)
        selected.add(key)


def _discover_amplicon_groups(
    bin_counts: Counter,
    minimum_reads: int,
) -> list[dict]:
    """Discover high-support amplicons from adjacent alignment-start bins.

    A small per-bin support floor rejects the nearly continuous sprinkling of
    genomic starts in ordinary whole-genome data. Adjacent supported bins are
    merged, then one occupied flanking bin is absorbed to retain endpoint
    jitter. Consensus endpoints and exact read counts are measured in pass 2.
    """
    core_floor = max(
        _AMPLICON_MIN_BIN_READS,
        min(10, int(round(minimum_reads * 0.10))),
    )
    cores_by_chrom: dict[str, list[int]] = defaultdict(list)
    for (chrom, bin_index), count in bin_counts.items():
        if count >= core_floor:
            cores_by_chrom[chrom].append(int(bin_index))

    groups = []
    for chrom, raw_bins in cores_by_chrom.items():
        current: list[int] = []
        core_groups: list[list[int]] = []
        for bin_index in sorted(raw_bins):
            if current and bin_index > current[-1] + 1:
                core_groups.append(current)
                current = []
            current.append(bin_index)
        if current:
            core_groups.append(current)
        for core in core_groups:
            members = set(core)
            for flank in (core[0] - 1, core[-1] + 1):
                if bin_counts.get((chrom, flank), 0) > 0:
                    members.add(flank)
            support = sum(bin_counts[(chrom, bin_index)] for bin_index in members)
            if support >= minimum_reads:
                groups.append(
                    {
                        "chrom": chrom,
                        "bin_indices": sorted(members),
                        "discovery_reads": int(support),
                    }
                )
    groups.sort(
        key=lambda item: (
            -item["discovery_reads"],
            item["chrom"],
            item["bin_indices"][0],
        )
    )
    for index, group in enumerate(groups, 1):
        group["amplicon_id"] = f"amplicon_{index}"
    return groups


def _counter_quantile(counter: Counter, probability: float) -> Optional[int]:
    total = sum(counter.values())
    if total <= 0:
        return None
    threshold = probability * (total - 1)
    cumulative = 0
    for value, count in sorted(counter.items()):
        cumulative += count
        if cumulative > threshold:
            return int(value)
    return int(max(counter))


def _aligned_pairs(read, reference_handle=None):
    if md_matches_cigar(read):
        try:
            return read.get_aligned_pairs(with_seq=True)
        except (ValueError, TypeError, IndexError, AssertionError):
            pass
    if reference_handle is None or read.reference_name is None:
        return None
    try:
        start = int(read.reference_start)
        end = int(read.reference_end)
        reference = reference_handle.fetch(read.reference_name, start, end).upper()
        return [
            (query_position, reference_position, reference[reference_position - start])
            for query_position, reference_position in read.get_aligned_pairs(
                matches_only=True
            )
            if start <= reference_position < end
        ]
    except (ValueError, TypeError, IndexError, OSError):
        return None


def _profile(read, reference_handle=None):
    sequence = (read.query_sequence or "").upper()
    pairs = _aligned_pairs(read, reference_handle)
    if not sequence or pairs is None:
        return None
    ct_positions = []
    ga_positions = []
    usable_pairs = []
    for query_position, reference_position, reference_base in pairs:
        if query_position is None or reference_position is None or reference_base is None:
            continue
        reference_base = reference_base.upper()
        query_base = sequence[query_position]
        usable_pairs.append(
            (int(query_position), int(reference_position), reference_base, query_base)
        )
        if reference_base == "C" and query_base in ("T", "Y"):
            ct_positions.append(int(reference_position))
        elif reference_base == "G" and query_base in ("A", "R"):
            ga_positions.append(int(reference_position))
    return ct_positions, ga_positions, usable_pairs


def _dominant_direction(
    ct_count: int,
    ga_count: int,
    minimum_events: int,
    minimum_purity: float,
) -> Optional[str]:
    total = ct_count + ga_count
    if total == 0:
        return None
    if ct_count >= minimum_events and ct_count / total >= minimum_purity:
        return "CT"
    if ga_count >= minimum_events and ga_count / total >= minimum_purity:
        return "GA"
    return None


def call_opposite_conversion_snps(
    input_path: str,
    min_fraction: float = DEFAULT_SNP_MIN_FRACTION,
    min_depth: int = DEFAULT_SNP_MIN_DEPTH,
    min_alt_fibers: int = DEFAULT_SNP_MIN_ALT_FIBERS,
    min_dominant_events: int = 5,
    min_dominant_purity: float = 0.80,
    min_mapq: int = 20,
    reference_fasta: Optional[str] = None,
    max_profile_sites: int = _SITE_PROFILE_MAX_SITES,
    profile_min_depth: int = _SITE_PROFILE_MIN_DEPTH,
    profile_seed: int = _SITE_PROFILE_SEED,
    min_amplicon_reads: int = _AMPLICON_MIN_READS,
) -> dict:
    """Two-pass DAF SNP call using recurrent events on opposite-direction fibers.

    A C→T event is screened on otherwise G→A-dominant molecules, and a G→A
    event on otherwise C→T-dominant molecules. A call then requires adequate
    depth, recurrent mismatch support, and the mismatch-fraction threshold in
    both conversion-direction classes. This rejects direction-specific
    deamination/basecalling artifacts without imposing a brittle high-depth
    cliff. A deterministic bounded sample of ordinary C/G positions is
    retained so QC can display the background mismatch distribution rather
    than showing called variants alone. Original MD tags are never modified.
    """
    if not 0 < min_fraction <= 1:
        raise ValueError("min_fraction must be in (0, 1]")
    if min_depth < 1 or min_alt_fibers < 1 or min_dominant_events < 1:
        raise ValueError("depth/event thresholds must be positive")
    if not 0.5 <= min_dominant_purity <= 1:
        raise ValueError("min_dominant_purity must be in [0.5, 1]")
    if max_profile_sites < 0 or profile_min_depth < 1:
        raise ValueError("site-profile limits must be non-negative/positive")
    if min_amplicon_reads < 1:
        raise ValueError("minimum amplicon reads must be positive")

    reference_handle = pysam.FastaFile(reference_fasta) if reference_fasta else None
    candidate_alt_counts: Counter = Counter()
    accounting = Counter()
    profile_heap: list[tuple[int, tuple[str, int, str]]] = []
    sampled_profile_sites: set[tuple[str, int, str]] = set()
    amplicon_bins: Counter = Counter()
    try:
        with pysam.AlignmentFile(input_path, "rb", check_sq=False) as bam:
            for read in bam.fetch(until_eof=True):
                accounting["records_examined_pass1"] += 1
                if (
                    read.is_unmapped
                    or read.is_secondary
                    or read.is_supplementary
                    or read.is_duplicate
                    or read.mapping_quality < min_mapq
                ):
                    continue
                amplicon_bins[
                    (read.reference_name, int(read.reference_start) // _AMPLICON_BIN_BP)
                ] += 1
                profile = _profile(read, reference_handle)
                if profile is None:
                    accounting["unusable_alignment_records"] += 1
                    continue
                ct_positions, ga_positions, _pairs = profile
                direction = _dominant_direction(
                    len(ct_positions),
                    len(ga_positions),
                    min_dominant_events,
                    min_dominant_purity,
                )
                if direction is None:
                    accounting["ambiguous_direction_records"] += 1
                    continue
                accounting[f"{direction.lower()}_dominant_records"] += 1
                for _query_position, position, reference_base, _query_base in _pairs:
                    if reference_base in ("C", "G"):
                        _consider_profile_site(
                            (read.reference_name, position, reference_base),
                            profile_heap,
                            sampled_profile_sites,
                            max_profile_sites,
                            profile_seed,
                        )
                if direction == "GA":
                    for position in set(ct_positions):
                        candidate_alt_counts[(read.reference_name, position, "C", "T", "GA")] += 1
                else:
                    for position in set(ga_positions):
                        candidate_alt_counts[(read.reference_name, position, "G", "A", "CT")] += 1

        candidates = {
            key: count
            for key, count in candidate_alt_counts.items()
            if count >= min_alt_fibers
        }
        profiled_sites = set(sampled_profile_sites)
        for chrom, position, reference, _alternate, _direction in candidates:
            profiled_sites.add((chrom, position, reference))
        profiled_by_chrom = defaultdict(lambda: defaultdict(list))
        for chrom, position, reference in profiled_sites:
            alternate = "T" if reference == "C" else "A"
            expected_direction = "CT" if reference == "C" else "GA"
            profiled_by_chrom[chrom][position].append((
                reference,
                alternate,
                expected_direction,
            ))

        site_stats: dict[tuple[str, int, str], Counter] = defaultdict(Counter)
        base_votes: dict[tuple[str, int], Counter] = defaultdict(Counter)
        amplicon_groups = _discover_amplicon_groups(
            amplicon_bins, min_amplicon_reads
        )
        amplicon_lookup = {}
        amplicon_endpoint_stats = []
        for group_index, group in enumerate(amplicon_groups):
            for bin_index in group["bin_indices"]:
                amplicon_lookup[(group["chrom"], bin_index)] = group_index
            amplicon_endpoint_stats.append(
                {"starts": Counter(), "ends": Counter(), "total_reads": 0}
            )
        with pysam.AlignmentFile(input_path, "rb", check_sq=False) as bam:
            for read in bam.fetch(until_eof=True):
                accounting["records_examined_pass2"] += 1
                if (
                    read.is_unmapped
                    or read.is_secondary
                    or read.is_supplementary
                    or read.is_duplicate
                    or read.mapping_quality < min_mapq
                ):
                    continue
                bin_key = (
                    read.reference_name,
                    int(read.reference_start) // _AMPLICON_BIN_BP,
                )
                amplicon_index = amplicon_lookup.get(bin_key)
                if amplicon_index is not None:
                    endpoint_stats = amplicon_endpoint_stats[amplicon_index]
                    endpoint_stats["starts"][int(read.reference_start)] += 1
                    endpoint_stats["ends"][int(read.reference_end or read.reference_start)] += 1
                    endpoint_stats["total_reads"] += 1
                chrom_sites = profiled_by_chrom.get(read.reference_name)
                if not chrom_sites:
                    continue
                profile = _profile(read, reference_handle)
                if profile is None:
                    continue
                ct_positions, ga_positions, pairs = profile
                direction = _dominant_direction(
                    len(ct_positions),
                    len(ga_positions),
                    min_dominant_events,
                    min_dominant_purity,
                )
                if direction is None:
                    continue
                seen = set()
                voted = set()
                for _query_position, position, reference_base, query_base in pairs:
                    hypotheses = chrom_sites.get(position)
                    if hypotheses is None:
                        continue
                    if reference_base in ("C", "G") and (position, reference_base) not in voted:
                        voted.add((position, reference_base))
                        base_votes[(read.reference_name, position)][reference_base] += 1
                    if position in seen:
                        continue
                    site = next((h for h in hypotheses if h[0] == reference_base), None)
                    if site is None:
                        continue
                    reference, alternate, expected_direction = site
                    key = (read.reference_name, position, reference)
                    mismatch = query_base in (("T", "Y") if reference == "C" else ("A", "R"))
                    if direction == expected_direction:
                        site_stats[key]["expected_depth"] += 1
                        site_stats[key]["expected_mismatches"] += int(mismatch)
                    else:
                        site_stats[key]["opposite_depth"] += 1
                        site_stats[key]["opposite_mismatches"] += int(mismatch)
                    seen.add(position)
    finally:
        if reference_handle is not None:
            reference_handle.close()

    rejected = set()
    disputed = set()
    for chrom, position, reference in profiled_sites:
        votes = base_votes.get((chrom, position), Counter())
        own, other = votes[reference], votes["G" if reference == "C" else "C"]
        if own and other:
            disputed.add((chrom, position))
        if other > own or (other == own and other and reference == "G"):
            rejected.add((chrom, position, reference))
    if disputed:
        accounting["reference_base_conflict_sites"] += len(disputed)
    profiled_sites -= rejected

    calls = []
    for key, alt_fibers in sorted(candidates.items()):
        chrom, position, reference, alternate, direction = key
        if (chrom, position, reference) in rejected:
            continue
        stats = site_stats[(chrom, position, reference)]
        depth = stats["opposite_depth"]
        observed_alt = stats["opposite_mismatches"]
        # Pass-2 mismatch accounting is authoritative; pass-1 counts are kept
        # only as a candidate-screening statistic.
        fraction = observed_alt / depth if depth else 0.0
        expected_depth = stats["expected_depth"]
        expected_mismatches = stats["expected_mismatches"]
        expected_fraction = (
            expected_mismatches / expected_depth if expected_depth else 0.0
        )
        if (
            depth >= min_depth
            and expected_depth >= min_depth
            and observed_alt >= min_alt_fibers
            and expected_mismatches >= min_alt_fibers
            and fraction >= min_fraction
            and expected_fraction >= min_fraction
        ):
            calls.append(
                {
                    "chrom": chrom,
                    "position_0based": int(position),
                    "reference": reference,
                    "alternate": alternate,
                    "opposite_dominant_direction": direction,
                    "alternate_fibers": int(observed_alt),
                    "opposite_direction_depth": int(depth),
                    "alternate_fraction": float(fraction),
                    "expected_dominant_direction": (
                        "CT" if reference == "C" else "GA"
                    ),
                    "expected_direction_depth": int(expected_depth),
                    "expected_direction_mismatch_fibers": int(expected_mismatches),
                    "expected_direction_mismatch_fraction": (
                        float(expected_fraction)
                    ),
                    "candidate_alt_fibers_pass1": int(alt_fibers),
                }
            )
    called_keys = {
        (call["chrom"], call["position_0based"], call["reference"])
        for call in calls
    }
    site_distribution = []
    for chrom, position, reference in sorted(profiled_sites):
        stats = site_stats[(chrom, position, reference)]
        expected_depth = int(stats["expected_depth"])
        opposite_depth = int(stats["opposite_depth"])
        is_called = (chrom, position, reference) in called_keys
        if (
            not is_called
            and (expected_depth < profile_min_depth or opposite_depth < profile_min_depth)
        ):
            continue
        expected_mismatches = int(stats["expected_mismatches"])
        opposite_mismatches = int(stats["opposite_mismatches"])
        site_distribution.append(
            {
                "chrom": chrom,
                "position_0based": int(position),
                "reference": reference,
                "alternate": "T" if reference == "C" else "A",
                "expected_dominant_direction": "CT" if reference == "C" else "GA",
                "expected_direction_depth": expected_depth,
                "expected_direction_mismatch_fibers": expected_mismatches,
                "expected_direction_mismatch_fraction": (
                    float(expected_mismatches / expected_depth)
                    if expected_depth else None
                ),
                "opposite_direction_depth": opposite_depth,
                "opposite_direction_mismatch_fibers": opposite_mismatches,
                "opposite_direction_mismatch_fraction": (
                    float(opposite_mismatches / opposite_depth)
                    if opposite_depth else None
                ),
                "total_dominant_fiber_depth": expected_depth + opposite_depth,
                "called_as_snp": is_called,
            }
        )

    amplicons = []
    for group, endpoint_stats in zip(amplicon_groups, amplicon_endpoint_stats):
        consensus_start = _counter_quantile(endpoint_stats["starts"], 0.50)
        consensus_end = _counter_quantile(endpoint_stats["ends"], 0.50)
        if consensus_start is None or consensus_end is None or consensus_end <= consensus_start:
            continue
        start_q05 = _counter_quantile(endpoint_stats["starts"], 0.05)
        start_q95 = _counter_quantile(endpoint_stats["starts"], 0.95)
        end_q05 = _counter_quantile(endpoint_stats["ends"], 0.05)
        end_q95 = _counter_quantile(endpoint_stats["ends"], 0.95)
        amplicon_calls = []
        for call in calls:
            position = int(call["position_0based"])
            if (
                call["chrom"] == group["chrom"]
                and consensus_start <= position < consensus_end
            ):
                relative_bp = position - consensus_start
                amplicon_calls.append(
                    {
                        "position_0based": position,
                        "change": f"{call['reference']}>{call['alternate']}",
                        "relative_position_bp": int(relative_bp),
                        "relative_position_fraction": float(
                            relative_bp / (consensus_end - consensus_start)
                        ),
                        "opposite_mismatch_fraction": float(
                            call["alternate_fraction"]
                        ),
                    }
                )
                call.setdefault("amplicon_ids", []).append(group["amplicon_id"])
        amplicons.append(
            {
                "amplicon_id": group["amplicon_id"],
                "chrom": group["chrom"],
                "consensus_start_0based": int(consensus_start),
                "consensus_end_0based_exclusive": int(consensus_end),
                "consensus_length_bp": int(consensus_end - consensus_start),
                "total_aligned_reads": int(endpoint_stats["total_reads"]),
                "discovery_start_bin_reads": int(group["discovery_reads"]),
                "endpoint_quantiles_0based": {
                    "start_q05": start_q05,
                    "start_q95": start_q95,
                    "end_q05": end_q05,
                    "end_q95": end_q95,
                },
                "n_called_snps": len(amplicon_calls),
                "snp_positions": amplicon_calls,
            }
        )
    amplicons.sort(
        key=lambda item: (
            -item["total_aligned_reads"],
            item["chrom"],
            item["consensus_start_0based"],
        )
    )
    # One-shot remap (see the module docstring): replacing IDs one at a time
    # let a later rename overwrite an earlier one when two amplicons swap.
    renamed = {amplicon["amplicon_id"]: f"amplicon_{index}"
               for index, amplicon in enumerate(amplicons, 1)}
    for call in calls:
        call["amplicon_ids"] = [renamed.get(value, value)
                                for value in call.get("amplicon_ids", [])]
    for amplicon in amplicons:
        amplicon["amplicon_id"] = renamed[amplicon["amplicon_id"]]

    dominant_amplicon = None
    if amplicons:
        first = amplicons[0]
        dominant_amplicon = {
            "chrom": first["chrom"],
            "start_0based": first["consensus_start_0based"],
            "end_0based_exclusive": first["consensus_end_0based_exclusive"],
            "overlapping_dominant_fibers": first["total_aligned_reads"],
            "selection": (
                f"highest-coverage discovered amplicon with >= "
                f"{min_amplicon_reads:,} aligned reads"
            ),
        }
    return {
        "schema_version": 3,
        "method": "bidirectional_recurrent_daf_snp",
        "threshold_policy": describe_snp_threshold_policy(
            min_fraction,
            min_depth,
            min_alt_fibers,
        ),
        "input": str(Path(input_path).resolve()),
        "parameters": {
            "min_fraction": min_fraction,
            "min_depth": min_depth,
            "min_alt_fibers": min_alt_fibers,
            "bidirectional_support_required": True,
            "min_dominant_events": min_dominant_events,
            "min_dominant_purity": min_dominant_purity,
            "min_mapq": min_mapq,
            "site_profile_max_sites": int(max_profile_sites),
            "site_profile_min_depth_per_direction": int(profile_min_depth),
            "site_profile_seed": int(profile_seed),
            "min_amplicon_reads": int(min_amplicon_reads),
            "amplicon_start_bin_bp": int(_AMPLICON_BIN_BP),
        },
        "accounting": dict(accounting),
        "n_candidate_sites_after_min_alt": len(candidates),
        "n_called_snps": len(calls),
        "n_profiled_sites": len(site_distribution),
        "site_distribution": site_distribution,
        "n_discovered_amplicons": len(amplicons),
        "amplicons": amplicons,
        "dominant_amplicon": dominant_amplicon,
        "calls": calls,
    }




# --- fiberhmm.daf.encoder at 45b9a1a -------------------------------------

from fiberhmm.daf.encoder import _aligned_pairs_from_fasta  # unchanged helper

_CIGAR_REF_CONSUMING_FOR_MD = {0, 2, 7, 8}


def _md_tag_ref_length(md_string: str) -> int:
    """Return the reference length encoded by an MD tag.

    MD is a sequence of: runs of digits (ref positions matched), single-base
    mismatches (one ref base consumed each), and ``^<SEQ>`` deletions
    (len(<SEQ>) ref bases consumed). See SAM spec section 1.4.11.
    """
    n = 0
    i = 0
    L = len(md_string)
    while i < L:
        c = md_string[i]
        if c.isdigit():
            j = i
            while j < L and md_string[j].isdigit():
                j += 1
            n += int(md_string[i:j])
            i = j
        elif c == '^':
            # ^<seq> deletion; each letter is one ref base consumed.
            j = i + 1
            while j < L and md_string[j].isalpha():
                j += 1
            n += j - (i + 1)
            i = j
        elif c.isalpha():
            # Single-base mismatch, one ref base consumed.
            n += 1
            i += 1
        else:
            # Skip any stray punctuation (the spec doesn't allow it but
            # be permissive on read-side).
            i += 1
    return n


def md_matches_cigar(read) -> bool:
    """Cheap pre-validation: does the MD tag's encoded reference length
    match the CIGAR's reference-consuming operations?

    Returns True if the read has no MD tag (caller should handle that
    case separately), True if the lengths match, False if they disagree.

    Motivation: ``pysam.AlignedSegment.get_aligned_pairs(with_seq=True)``
    raises AssertionError on mismatch AND, in at least some pysam
    versions, corrupts internal malloc state before the exception
    propagates — which then manifests as ``malloc(): invalid size``
    somewhere later in the worker. Skipping the call on obviously-bad
    MD avoids triggering the crash path at all.
    """
    try:
        md = read.get_tag('MD') if read.has_tag('MD') else None
    except Exception:
        return True
    if md is None:
        return True
    try:
        md_len = _md_tag_ref_length(md)
    except Exception:
        return False
    cigar = read.cigartuples
    if cigar is None:
        return True
    cigar_ref_len = sum(length for op, length in cigar
                        if op in _CIGAR_REF_CONSUMING_FOR_MD)
    # Departure from the frozen copy: equal lengths are not enough when a
    # deletion run covers an insertion (pysam then reads undefined memory).
    return md_len == cigar_ref_len and not pysam_md_walk_ends_short(md, cigar)


def pysam_md_walk_ends_short(md, cigar) -> bool:
    """Replay pysam's ``build_alignment_sequence`` MD walk (per character).

    Returns True when the walk ends before the last M/=/X/D base of the
    layout, i.e. when ``build_reference_sequence`` would index past the
    string it built (undefined memory). Insertions (I/P) are skipped before
    every MD token but not inside a ``^`` run, exactly as in pysam 0.24.
    """
    layout = []
    for op, length in cigar:
        if op in (0, 2, 7, 8):
            layout.extend("R" * length)
        elif op in (1, 6):
            layout.extend("i" * length)
    size = len(layout)
    position = 0

    def skip_insertions(position):
        while position < size and layout[position] == "i":
            position += 1
        return position

    matches = 0
    index = 0
    while index < len(md):
        char = md[index]
        if char.isdigit():
            matches = matches * 10 + int(char)
            index += 1
            continue
        for _ in range(matches):
            position = skip_insertions(position) + 1
        position = skip_insertions(position)
        matches = 0
        if char == "^":
            index += 1
            while index < len(md) and "A" <= md[index] <= "Z":
                position += 1
                index += 1
        else:
            position += 1
            index += 1
    for _ in range(matches):
        position = skip_insertions(position) + 1
    position = skip_insertions(position)
    last_reference = max((i for i, kind in enumerate(layout) if kind == "R"), default=-1)
    return position <= last_reference


def get_daf_positions(
    read,
    force_strand=None,
    ref_fasta=None,
    excluded_reference_positions=None,
):
    """Collect C->T and G->A mismatch positions for a DAF-seq read.

    Pure position-collection helper used by both ``encode_read_daf``
    (which rewrites the query sequence with R/Y IUPAC codes) and the
    in-memory fallback inside ``fiberhmm-call --mode daf`` (which
    consumes positions directly without touching the stored sequence).

    Parameters
    ----------
    read : pysam.AlignedSegment
        Aligned read from a BAM file.
    force_strand : str or None
        ``"CT"``, ``"GA"``, or ``None`` (auto-detect per read).
    ref_fasta : pysam.FastaFile or None
        Opened reference FASTA, used as fallback when the MD tag is absent.

    Returns
    -------
    tuple or None
        ``(ct_positions, ga_positions, strand)`` where ``strand`` is
        ``"CT"`` or ``"GA"`` and the two lists are the full sets of
        C->T and G->A query-position mismatches (both populated even
        though only the selected-strand list is used downstream -- the
        other is returned for diagnostics and future use).

        ``None`` if the read should be skipped (unmapped, secondary,
        supplementary, no mismatches, or ambiguous strand with
        no ``force_strand``).
    """
    # Skip unmapped / secondary / supplementary
    if read.is_unmapped or read.is_secondary or read.is_supplementary:
        return None

    seq = read.query_sequence
    if seq is None:
        return None

    # Get aligned pairs with reference bases.
    # Pre-validate MD vs CIGAR to avoid pysam's AssertionError path on
    # malformed BAMs — that path can corrupt malloc state in some pysam
    # versions, crashing the worker later with "malloc(): invalid size".
    pairs = None
    if md_matches_cigar(read):
        try:
            pairs = read.get_aligned_pairs(with_seq=True)
        except Exception:
            # MD tag missing and no way to get ref bases without it.
            if ref_fasta is None:
                return None
            pairs = None
    elif ref_fasta is None:
        # MD disagrees with CIGAR and no reference available — can't
        # safely get ref bases. Skip this read.
        return None

    # Fallback: build pairs from reference FASTA when MD tag is absent
    if pairs is None or all(p[2] is None for p in pairs if p[0] is not None and p[1] is not None):
        if ref_fasta is None:
            return None
        try:
            pairs = _aligned_pairs_from_fasta(read, ref_fasta)
        except Exception:
            return None

    # Collect mismatch positions
    ct_positions = []  # C->T (+ strand deamination)
    ga_positions = []  # G->A (- strand deamination)

    excluded_reference_positions = excluded_reference_positions or set()
    for query_pos, ref_pos, ref_base in pairs:
        if query_pos is None or ref_pos is None or ref_base is None:
            continue
        if ref_pos in excluded_reference_positions:
            continue
        ref_base = ref_base.upper()
        query_base = seq[query_pos].upper()
        if ref_base == "C" and query_base == "T":
            ct_positions.append(query_pos)
        elif ref_base == "G" and query_base == "A":
            ga_positions.append(query_pos)

    n_ct = len(ct_positions)
    n_ga = len(ga_positions)

    # Determine conversion strand
    if force_strand is not None:
        strand = force_strand.upper()
    else:
        if n_ct == 0 and n_ga == 0:
            return None
        if n_ct > n_ga:
            strand = "CT"
        elif n_ga > n_ct:
            strand = "GA"
        else:
            # Equal and nonzero -- ambiguous, skip
            return None

    return (ct_positions, ga_positions, strand)



# --- MD branch (priority 3) of fiberhmm.cli.extract_tags._deam_positions_list at 45b9a1a

def deam_positions_md_branch(read):
    """Priority-3 (MD mismatch) branch of ``_deam_positions_list``.

    Only reached when the read has no MM/ML ``u`` calls and no R/Y codes.
    """
    positions_list = []
    if not positions_list and read.has_tag('MD'):
        seq = read.query_sequence
        if seq:
            pairs = None
            if md_matches_cigar(read):
                try:
                    pairs = read.get_aligned_pairs(with_seq=True)
                except Exception:
                    pairs = None
            if pairs:
                for qpos, rpos, ref_base in pairs:
                    if qpos is None or rpos is None or ref_base is None:
                        continue
                    ref_up = ref_base.upper()
                    q_up = seq[qpos].upper() if qpos < len(seq) else ''
                    if ref_up == 'C' and q_up == 'T':
                        positions_list.append((int(rpos), 1))
                    elif ref_up == 'G' and q_up == 'A':
                        positions_list.append((int(rpos), 0))
    return positions_list
