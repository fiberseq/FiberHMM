"""Call recurrent opposite-conversion SNPs in DAF-seq alignments."""
from __future__ import annotations

import json
import hashlib
import heapq
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pysam

from fiberhmm.daf.aligned_arrays import (
    BASE_A,
    BASE_C,
    BASE_G,
    BASE_R,
    BASE_T,
    BASE_Y,
    matched_base_arrays,
    md_disagrees_with_cigar,
)


_SITE_PROFILE_SEED = 20260824
_SITE_PROFILE_MAX_SITES = 5000
_SITE_PROFILE_MIN_DEPTH = 3
_AMPLICON_BIN_BP = 10_000
_AMPLICON_MIN_READS = 20
_AMPLICON_MIN_BIN_READS = 3
# Cap (64 KiB blocks) on the SNP screen's repeat-offer cache; see _OfferedSites.
_OFFERED_SITES_MAX_BLOCKS = 2048

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
    # An MD tag that does not describe the CIGAR gives no defined reference
    # bases (pysam reads past a short MD into undefined memory): such a read
    # uses the FASTA when one is given and is unusable otherwise.
    if not md_disagrees_with_cigar(read):
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


def _base_code(base: str) -> int:
    """ASCII code of a single upper-cased base (0 for anything else)."""
    return ord(base) if len(base) == 1 else 0


_TOPOLOGY: dict = {}


def circular_contig_lengths(header) -> dict:
    """``{name: LN}`` of the contigs a header declares circular.

    A contig is circular when its ``@SQ`` line has ``TP:circular`` or a
    ``FIBERHMM-REFERENCE`` ``@CO`` line (written by ``fiberhmm-pipeline``)
    gives it ``topology=circular``. Cached per header object (the cache holds
    the header, so its id cannot be reused while cached).
    """
    cached = _TOPOLOGY.get(id(header))
    if cached is not None and cached[0] is header:
        return cached[1]
    try:
        data = header.to_dict() if hasattr(header, "to_dict") else dict(header or {})
    except (TypeError, ValueError, AttributeError):
        data = {}
    lengths: dict = {}
    declared: dict = {}
    for sq in data.get("SQ", []) or []:
        try:
            length = int(sq.get("LN"))
        except (TypeError, ValueError):
            continue
        declared[str(sq.get("SN"))] = length
        if str(sq.get("TP", "")).lower() == "circular":
            lengths[str(sq.get("SN"))] = length
    comments = [str(c) for c in data.get("CO", []) or []]
    if any(c.startswith("FIBERHMM-REFERENCE:") for c in comments):
        from fiberhmm.pipeline.reference import parse_reference_comment

        for comment in comments:
            parsed = parse_reference_comment(comment)
            if (parsed and str(parsed.get("topology", "")).lower() == "circular"
                    and parsed["contig"] in declared):
                lengths[parsed["contig"]] = declared[parsed["contig"]]
    if len(_TOPOLOGY) > 64:
        _TOPOLOGY.clear()
    _TOPOLOGY[id(header)] = (header, lengths)
    return lengths


def _past_end(read) -> tuple:
    """``(LN, circular)`` when ``read`` runs past the end of its contig, else
    ``(None, False)``."""
    end = getattr(read, "reference_end", None)
    header = getattr(read, "header", None)
    name = getattr(read, "reference_name", None)
    if end is None or header is None or name is None:
        return None, False
    try:
        length = int(header.get_reference_length(name))
    except (AttributeError, KeyError, ValueError, TypeError):
        return None, False
    if length <= 0 or end <= length:
        return None, False
    return length, name in circular_contig_lengths(header)


def contig_length_past_end(read) -> Optional[int]:
    """The contig length when ``read`` runs past the end of a circular contig.

    A record on a circular contig may extend past ``LN`` (SAM section 1.4;
    ``fiberhmm-pipeline`` writes origin-spanning reads that way); its reference
    positions ``p >= LN`` mean ``p mod LN``, on every turn. On a linear
    contig (no ``TP:circular`` and no circular ``FIBERHMM-REFERENCE`` line)
    positions past ``LN`` are not on the reference and return None here.
    """
    length, circular = _past_end(read)
    return length if circular else None


_WRAPPED_SITES: dict = {}


def wrapped_reference_sites(read, sites):
    """``sites`` (reference positions on the read's contig) as the read sees them.

    For a record that runs past the end of a circular contig, each site ``p``
    is also present as ``p + k*LN`` for every further turn ``k`` the record
    reaches, so a mask applies to every part of the read after the origin.
    Other records (including linear overhangs, whose positions past ``LN``
    are no reference site) get ``sites`` unchanged.

    Results are memoised only for immutable ``frozenset`` masks (what
    :func:`load_snp_mask` returns), keyed on the object, which the cache
    holds; a mutable ``set`` may change in place, so it is expanded afresh
    on every call.
    """
    if not sites:
        return sites
    length = contig_length_past_end(read)
    if length is None:
        return sites
    turns = (int(read.reference_end) - 1) // length
    if not isinstance(sites, frozenset):
        return frozenset(int(p) + k * length
                         for k in range(turns + 1) for p in sites)
    key = (id(sites), length, turns)
    cached = _WRAPPED_SITES.get(key)
    if cached is None or cached[0] is not sites:
        if len(_WRAPPED_SITES) > 64:
            _WRAPPED_SITES.clear()
        cached = (sites, frozenset(int(p) + k * length
                                   for k in range(turns + 1) for p in sites))
        _WRAPPED_SITES[key] = cached
    return cached[1]


def _read_arrays(read, reference_handle=None):
    """Per-read ``(rpos, ref_codes, query_codes)`` arrays for the SNP screen.

    Equivalent to the usable pairs of :func:`_profile` (same pairs, same
    order, same upper-cased bases); ``None`` exactly when ``_profile`` is.
    The vectorised MD path is used whenever it is exact; anything else
    (no MD tag, FASTA fallback, malformed alignments) goes through
    ``_profile`` itself. Positions past the contig end are folded onto the
    contig (``p mod LN``, every turn) on a circular contig and dropped on a
    linear one (:func:`_on_contig`).
    """
    arrays = matched_base_arrays(read)
    if arrays is not None:
        _qpos, rpos, ref_codes, query_codes = arrays
        return _on_contig(read, rpos, ref_codes, query_codes)
    profile = _profile(read, reference_handle)
    if profile is None:
        return None
    pairs = profile[2]
    rpos = np.fromiter((pair[1] for pair in pairs), dtype=np.int64, count=len(pairs))
    ref_codes = np.fromiter(
        (_base_code(pair[2]) for pair in pairs), dtype=np.int64, count=len(pairs)
    )
    query_codes = np.fromiter(
        (_base_code(pair[3]) for pair in pairs), dtype=np.int64, count=len(pairs)
    )
    return _on_contig(read, rpos, ref_codes, query_codes)


def _on_contig(read, rpos, ref_codes, query_codes):
    """Aligned positions on the contig: past its end, ``p mod LN`` on a
    circular contig (every turn), dropped on a linear one."""
    length, circular = _past_end(read)
    if length is None:
        return rpos, ref_codes, query_codes
    if circular:
        return np.where(rpos >= length, rpos % length, rpos), ref_codes, query_codes
    keep = rpos < length
    return rpos[keep], ref_codes[keep], query_codes[keep]


def _covers_a_turn_twice(read) -> bool:
    """Whether a record on a circular contig spans more than one full turn
    (so folded positions repeat)."""
    length, circular = _past_end(read)
    return bool(circular and read.reference_end - read.reference_start > length)


def _first_occurrences(positions: np.ndarray) -> np.ndarray:
    """Indices of the first occurrence of each position, in read order (a
    record on a circular contig may cover a position on several turns)."""
    _, first = np.unique(positions, return_index=True)
    return np.sort(first)


def _conversion_masks(ref_codes, query_codes):
    """Boolean C->T/Y and G->A/R masks, as ``_profile`` classifies pairs."""
    ct = (ref_codes == BASE_C) & ((query_codes == BASE_T) | (query_codes == BASE_Y))
    ga = (ref_codes == BASE_G) & ((query_codes == BASE_A) | (query_codes == BASE_R))
    return ct, ga


class _OfferedSites:
    """Remembers which ``(chrom, position, C|G)`` keys were already offered.

    Offering a key to :func:`_consider_profile_site` a second time is always
    a no-op: a selected key returns immediately, and a rejected or evicted
    key ranks at or above the heap threshold, which never increases. Skipping
    repeat offers therefore leaves the bottom-k sample (and the order of the
    effective offers) unchanged while avoiding one hash per covered C/G per
    read. The memory is a cache only: it is cleared when it exceeds
    ``max_blocks`` blocks, which merely re-admits harmless repeat offers.
    """

    _BLOCK_BITS = 16
    _BLOCK_MASK = (1 << _BLOCK_BITS) - 1

    def __init__(self, max_blocks: Optional[int] = None):
        self._blocks: dict[tuple[str, int], np.ndarray] = {}
        self._max_blocks = (
            _OFFERED_SITES_MAX_BLOCKS if max_blocks is None else max_blocks
        )

    def first_offers(self, chrom: str, positions: np.ndarray, flags: np.ndarray) -> np.ndarray:
        """Mark keys as offered; True where a key had not been offered before.

        ``positions`` must be unique within the call (one read's aligned
        reference positions); ``flags`` is 1 for reference C, 2 for G.
        """
        fresh = np.empty(positions.size, dtype=bool)
        block_ids = positions >> self._BLOCK_BITS
        boundaries = np.flatnonzero(np.diff(block_ids)) + 1
        starts = [0] + boundaries.tolist()
        ends = boundaries.tolist() + [positions.size]
        for start, end in zip(starts, ends):
            key = (chrom, int(block_ids[start]))
            block = self._blocks.get(key)
            if block is None:
                if len(self._blocks) >= self._max_blocks:
                    self._blocks.clear()
                block = np.zeros(1 << self._BLOCK_BITS, dtype=np.uint8)
                self._blocks[key] = block
            offsets = positions[start:end] & self._BLOCK_MASK
            segment_flags = flags[start:end]
            fresh[start:end] = (block[offsets] & segment_flags) == 0
            block[offsets] |= segment_flags
        return fresh


class _SiteAccumulator:
    """Pass-2 per-site fiber counts for one contig's profiled sites."""

    def __init__(self, chrom_sites: dict):
        self.positions = np.fromiter(
            sorted(chrom_sites), dtype=np.int64, count=len(chrom_sites)
        )
        self.references = [
            chrom_sites[position][0] for position in self.positions.tolist()
        ]
        self.reference_codes = np.array(
            [_base_code(reference) for reference in self.references], dtype=np.int64
        )
        self.site_is_c = self.reference_codes == BASE_C
        size = self.positions.size
        self.expected_depth = np.zeros(size, dtype=np.int64)
        self.expected_mismatches = np.zeros(size, dtype=np.int64)
        self.opposite_depth = np.zeros(size, dtype=np.int64)
        self.opposite_mismatches = np.zeros(size, dtype=np.int64)
        # Amplicons/plasmids: a dense position -> site-index table over the
        # profiled span. Sparse whole-genome sites fall back to searchsorted.
        self.lookup = None
        self.lookup_start = 0
        if size:
            span = int(self.positions[-1] - self.positions[0]) + 1
            if span <= self._MAX_DENSE_SPAN:
                self.lookup_start = int(self.positions[0])
                self.lookup = np.full(span, -1, dtype=np.int32)
                self.lookup[self.positions - self.lookup_start] = np.arange(size)

    _MAX_DENSE_SPAN = 1 << 22

    def _site_index(self, rpos):
        """Site index of each position (any in-range index where absent)."""
        if self.lookup is None:
            index = np.searchsorted(self.positions, rpos)
            np.minimum(index, self.positions.size - 1, out=index)
            return index
        offsets = rpos - self.lookup_start
        np.clip(offsets, 0, self.lookup.size - 1, out=offsets)
        index = self.lookup[offsets]
        index[index < 0] = 0
        return index

    def add_read(self, direction: str, rpos, ref_codes, query_codes) -> None:
        index = self._site_index(rpos)
        hit = (self.positions[index] == rpos) & (self.reference_codes[index] == ref_codes)
        if not hit.any():
            return
        sites = index[hit]
        query = query_codes[hit]
        site_is_c = self.site_is_c[sites]
        mismatch = np.where(
            site_is_c,
            (query == BASE_T) | (query == BASE_Y),
            (query == BASE_A) | (query == BASE_R),
        )
        # A site's expected direction is CT for reference C and GA for G.
        expected = site_is_c if direction == "CT" else ~site_is_c
        # A read counts once per site: fancy-index increments do not
        # accumulate repeated indices, so a site a circular record covers on
        # several turns adds one depth, and one mismatch if any turn has it.
        self.expected_depth[sites[expected]] += 1
        self.expected_mismatches[sites[expected & mismatch]] += 1
        self.opposite_depth[sites[~expected]] += 1
        self.opposite_mismatches[sites[~expected & mismatch]] += 1

    def export(self, chrom: str, site_stats) -> None:
        touched = np.flatnonzero((self.expected_depth + self.opposite_depth) > 0)
        for site in touched.tolist():
            site_stats[(chrom, int(self.positions[site]), self.references[site])] = Counter(
                {
                    "expected_depth": int(self.expected_depth[site]),
                    "expected_mismatches": int(self.expected_mismatches[site]),
                    "opposite_depth": int(self.opposite_depth[site]),
                    "opposite_mismatches": int(self.opposite_mismatches[site]),
                }
            )


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


def _resolve_reference_conflicts(conflicted_by_chrom, site_stats) -> set:
    """Pick one reference base where reads' MD tags disagree (C vs G).

    For each position profiled under both hypotheses, the base reported by
    more of the screen's dominant-direction fibers (expected + opposite
    depth, pass 2) is kept; a tie keeps C, the earlier base in A<C<G<T
    order. Returns the rejected ``(chrom, position, reference)`` keys, which
    are then neither called nor profiled. The result depends only on the
    reads, not on set/dict iteration order or ``PYTHONHASHSEED``.
    """
    rejected = set()
    for chrom in sorted(conflicted_by_chrom):
        for position in sorted(conflicted_by_chrom[chrom]):
            depths = {}
            for reference in ("C", "G"):
                stats = site_stats.get((chrom, position, reference)) or {}
                depths[reference] = (
                    int(stats.get("expected_depth", 0))
                    + int(stats.get("opposite_depth", 0))
                )
            loser = "G" if depths["C"] >= depths["G"] else "C"
            rejected.add((chrom, position, loser))
    return rejected


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
    offered_sites = _OfferedSites()
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
                arrays = _read_arrays(read, reference_handle)
                if arrays is None:
                    accounting["unusable_alignment_records"] += 1
                    continue
                rpos, ref_codes, query_codes = arrays
                ct_mask, ga_mask = _conversion_masks(ref_codes, query_codes)
                ct_positions = rpos[ct_mask].tolist()
                ga_positions = rpos[ga_mask].tolist()
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
                if max_profile_sites > 0:
                    is_c = ref_codes == BASE_C
                    cg_index = np.flatnonzero(is_c | (ref_codes == BASE_G))
                    if cg_index.size:
                        cg_positions = rpos[cg_index]
                        cg_is_c = is_c[cg_index]
                        if _covers_a_turn_twice(read):
                            first = _first_occurrences(cg_positions)
                            cg_positions = cg_positions[first]
                            cg_is_c = cg_is_c[first]
                        fresh = offered_sites.first_offers(
                            read.reference_name,
                            cg_positions,
                            np.where(cg_is_c, 1, 2).astype(np.uint8),
                        )
                        chrom = read.reference_name
                        for position, site_is_c in zip(
                            cg_positions[fresh].tolist(), cg_is_c[fresh].tolist()
                        ):
                            _consider_profile_site(
                                (chrom, position, "C" if site_is_c else "G"),
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
        # Reads whose MD tags disagree about a position's reference base can
        # put it in the profile as both C and G. Each hypothesis is counted
        # separately (the second in its own accumulator) and resolved after
        # pass 2 by _resolve_reference_conflicts; sorted order puts the C
        # hypothesis in the primary map.
        profiled_by_chrom = defaultdict(dict)
        conflicted_by_chrom = defaultdict(dict)
        for chrom, position, reference in sorted(profiled_sites):
            alternate = "T" if reference == "C" else "A"
            expected_direction = "CT" if reference == "C" else "GA"
            target = profiled_by_chrom[chrom]
            if position in target:
                target = conflicted_by_chrom[chrom]
            target[position] = (
                reference,
                alternate,
                expected_direction,
            )

        site_stats: dict[tuple[str, int, str], Counter] = defaultdict(Counter)
        site_accumulators: dict[str, _SiteAccumulator] = {}
        conflict_accumulators: dict[str, _SiteAccumulator] = {}
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
                arrays = _read_arrays(read, reference_handle)
                if arrays is None:
                    continue
                rpos, ref_codes, query_codes = arrays
                ct_mask, ga_mask = _conversion_masks(ref_codes, query_codes)
                direction = _dominant_direction(
                    int(np.count_nonzero(ct_mask)),
                    int(np.count_nonzero(ga_mask)),
                    min_dominant_events,
                    min_dominant_purity,
                )
                if direction is None:
                    continue
                accumulator = site_accumulators.get(read.reference_name)
                if accumulator is None:
                    accumulator = _SiteAccumulator(chrom_sites)
                    site_accumulators[read.reference_name] = accumulator
                accumulator.add_read(direction, rpos, ref_codes, query_codes)
                conflicted_sites = conflicted_by_chrom.get(read.reference_name)
                if conflicted_sites:
                    accumulator = conflict_accumulators.get(read.reference_name)
                    if accumulator is None:
                        accumulator = _SiteAccumulator(conflicted_sites)
                        conflict_accumulators[read.reference_name] = accumulator
                    accumulator.add_read(direction, rpos, ref_codes, query_codes)
        for chrom, accumulator in site_accumulators.items():
            accumulator.export(chrom, site_stats)
        for chrom, accumulator in conflict_accumulators.items():
            accumulator.export(chrom, site_stats)
    finally:
        if reference_handle is not None:
            reference_handle.close()

    rejected_sites = _resolve_reference_conflicts(conflicted_by_chrom, site_stats)
    if rejected_sites:
        accounting["reference_base_conflict_sites"] += len(rejected_sites)
        profiled_sites -= rejected_sites

    calls = []
    for key, alt_fibers in sorted(candidates.items()):
        chrom, position, reference, alternate, direction = key
        if (chrom, position, reference) in rejected_sites:
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
    for index, amplicon in enumerate(amplicons, 1):
        old_id = amplicon["amplicon_id"]
        new_id = f"amplicon_{index}"
        if old_id != new_id:
            for call in calls:
                ids = call.get("amplicon_ids", [])
                call["amplicon_ids"] = [new_id if value == old_id else value for value in ids]
        amplicon["amplicon_id"] = new_id

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


def write_snp_outputs(payload: dict, output_prefix: str) -> dict:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    bed_path = Path(str(prefix) + ".bed")
    vcf_path = Path(str(prefix) + ".vcf")
    json_path = Path(str(prefix) + ".json")
    amplicon_path = Path(str(prefix) + ".amplicons.tsv")
    with bed_path.open("w") as handle:
        handle.write(
            "#chrom\tstart\tend\tchange\talt_fibers\topposite_depth\talt_fraction\topposite_direction\n"
        )
        for call in payload["calls"]:
            handle.write(
                f"{call['chrom']}\t{call['position_0based']}\t{call['position_0based'] + 1}\t"
                f"{call['reference']}>{call['alternate']}\t{call['alternate_fibers']}\t"
                f"{call['opposite_direction_depth']}\t{call['alternate_fraction']:.8f}\t"
                f"{call['opposite_dominant_direction']}\n"
            )
    with vcf_path.open("w") as handle:
        handle.write("##fileformat=VCFv4.3\n")
        handle.write("##source=FiberHMM-bidirectional-recurrent-DAF-SNP\n")
        handle.write('##INFO=<ID=AF,Number=1,Type=Float,Description="Opposite-direction fiber fraction">\n')
        handle.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="Opposite-direction fiber depth">\n')
        handle.write('##INFO=<ID=AC,Number=1,Type=Integer,Description="Alternate fibers">\n')
        handle.write('##INFO=<ID=ED,Number=1,Type=Integer,Description="Expected-direction fiber depth">\n')
        handle.write('##INFO=<ID=EA,Number=1,Type=Integer,Description="Expected-direction mismatch fibers">\n')
        handle.write('##INFO=<ID=EF,Number=1,Type=Float,Description="Expected-direction mismatch fraction">\n')
        handle.write('##INFO=<ID=OD,Number=1,Type=String,Description="Dominant direction on which event is unexpected">\n')
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for call in payload["calls"]:
            handle.write(
                f"{call['chrom']}\t{call['position_0based'] + 1}\t.\t"
                f"{call['reference']}\t{call['alternate']}\t.\tPASS\t"
                f"AF={call['alternate_fraction']:.8f};DP={call['opposite_direction_depth']};"
                f"AC={call['alternate_fibers']};ED={call['expected_direction_depth']};"
                f"EA={call['expected_direction_mismatch_fibers']};"
                f"EF={call['expected_direction_mismatch_fraction']:.8f};"
                f"OD={call['opposite_dominant_direction']}\n"
            )
    with amplicon_path.open("w") as handle:
        handle.write(
            "amplicon_id\tchrom\tconsensus_start_0based\t"
            "consensus_end_0based_exclusive\tconsensus_length_bp\t"
            "total_aligned_reads\tn_called_snps\tsnp_positions_relative_bp\n"
        )
        for amplicon in payload.get("amplicons", []):
            positions = ",".join(
                f"{site['change']}@{site['relative_position_bp']}"
                for site in amplicon.get("snp_positions", [])
            )
            handle.write(
                f"{amplicon['amplicon_id']}\t{amplicon['chrom']}\t"
                f"{amplicon['consensus_start_0based']}\t"
                f"{amplicon['consensus_end_0based_exclusive']}\t"
                f"{amplicon['consensus_length_bp']}\t"
                f"{amplicon['total_aligned_reads']}\t"
                f"{amplicon['n_called_snps']}\t{positions}\n"
            )
    payload = {
        **payload,
        "outputs": {
            "bed": str(bed_path.resolve()),
            "vcf": str(vcf_path.resolve()),
            "json": str(json_path.resolve()),
            "amplicons_tsv": str(amplicon_path.resolve()),
        },
    }
    temporary = json_path.with_name(json_path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, json_path)
    return payload


def load_snp_mask(path: Optional[str]) -> dict[str, frozenset[int]]:
    """``{contig: frozenset(0-based positions)}`` from a BED mask (immutable,
    so :func:`wrapped_reference_sites` can memoise per mask)."""
    mask: dict[str, set[int]] = defaultdict(set)
    if not path:
        return {}
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"invalid SNP mask line: {line.rstrip()!r}")
            mask[fields[0]].update(range(int(fields[1]), int(fields[2])))
    return {contig: frozenset(positions) for contig, positions in mask.items()}


def mask_summary(path: Optional[str]) -> dict:
    mask = load_snp_mask(path)
    return {
        "path": str(Path(path).resolve()) if path else None,
        "n_sites": sum(len(positions) for positions in mask.values()),
        "n_contigs": len(mask),
    }


__all__ = [
    "call_opposite_conversion_snps",
    "load_snp_mask",
    "mask_summary",
    "write_snp_outputs",
]
