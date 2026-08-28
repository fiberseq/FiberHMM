"""Compact, locally reusable identifiers for genomic TF-footprint families.

The identifiers produced here are display/analysis slots, not durable model
keys.  A slot is never reused while another family carrying that slot overlaps
the current canonical interval.  Slots therefore remain unambiguous when
combined with genomic position, while a single unsigned byte is sufficient for
the BAM annotation contract.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


MAX_TF_FAMILY_ID = 255
DEFAULT_FAMILY_SEPARATION_BP = 24


@dataclass(frozen=True)
class TFFamilyInterval:
    """One canonical half-open genomic family interval."""

    family_key: str
    contig: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.family_key:
            raise ValueError("family_key must be non-empty")
        if not self.contig:
            raise ValueError("contig must be non-empty")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("family interval must satisfy 0 <= start < end")


def allocate_repeating_family_ids(
    families: Iterable[TFFamilyInterval],
    *,
    maximum_id: int = MAX_TF_FAMILY_ID,
    separation_bp: int = DEFAULT_FAMILY_SEPARATION_BP,
) -> Dict[str, int]:
    """Assign sequential uint8 slots without collisions among overlaps.

    The counter advances from 1 through ``maximum_id`` and then wraps.  Slots
    still used by an overlapping canonical interval are skipped.  The counter
    restarts on each contig, making targeted and genome-wide results stable
    with respect to unrelated contigs.
    """

    if isinstance(maximum_id, bool) or not 1 <= int(maximum_id) <= 255:
        raise ValueError("maximum_id must be an integer in [1,255]")
    maximum_id = int(maximum_id)
    if isinstance(separation_bp, bool) or int(separation_bp) < 0:
        raise ValueError("separation_bp must be a nonnegative integer")
    separation_bp = int(separation_bp)
    ordered = sorted(
        tuple(families),
        key=lambda family: (
            family.contig,
            family.start,
            family.end,
            family.family_key,
        ),
    )
    if len({family.family_key for family in ordered}) != len(ordered):
        raise ValueError("family_key values must be unique")

    result: Dict[str, int] = {}
    active: list[Tuple[int, int]] = []
    active_slots: set[int] = set()
    current_contig = None
    next_slot = 1

    for family in ordered:
        if family.contig != current_contig:
            current_contig = family.contig
            active = []
            active_slots = set()
            next_slot = 1

        while active and active[0][0] <= family.start:
            _end, expired_slot = heapq.heappop(active)
            active_slots.remove(expired_slot)

        selected = None
        for _attempt in range(maximum_id):
            candidate = next_slot
            next_slot = 1 if next_slot == maximum_id else next_slot + 1
            if candidate not in active_slots:
                selected = candidate
                break
        if selected is None:
            raise ValueError(
                f"more than {maximum_id} TF families overlap at "
                f"{family.contig}:{family.start}-{family.end}"
            )
        result[family.family_key] = selected
        active_slots.add(selected)
        heapq.heappush(active, (family.end + separation_bp, selected))

    return result


__all__ = [
    "MAX_TF_FAMILY_ID",
    "DEFAULT_FAMILY_SEPARATION_BP",
    "TFFamilyInterval",
    "allocate_repeating_family_ids",
]
