"""Deterministic BED planning for targeted footprint-family batch scans."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence, Tuple


@dataclass(frozen=True)
class BedTarget:
    """One validated zero-based, half-open BED target."""

    ordinal: int
    target_id: str
    contig: str
    start: int
    end: int
    name: str
    score: str = "."
    strand: str = "."

    def expanded(self, padding: int, contig_length: int) -> Tuple[int, int]:
        return max(0, self.start - padding), min(contig_length, self.end + padding)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TargetedFamilyWorkUnit:
    """One non-overlapping indexed-fetch region containing one or more targets."""

    ordinal: int
    unit_id: str
    contig: str
    start: int
    end: int
    target_ordinals: Tuple[int, ...]
    oversized_connected_component: bool

    @property
    def region(self) -> str:
        return f"{self.contig}:{self.start}-{self.end}"

    def as_dict(self) -> dict:
        value = asdict(self)
        value["target_ordinals"] = list(self.target_ordinals)
        value["region"] = self.region
        return value


def load_bed_targets(
    path: str | Path,
    reference_lengths: Mapping[str, int],
) -> Tuple[BedTarget, ...]:
    """Parse BED3--BED6 targets without changing order.

    BED6 strand is retained so a motif-centered report can use the motif as
    coordinate zero and orient all intervals in the same biological
    direction.  Strand does not alter discovery or any parent family call.
    """

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"BED file does not exist: {path}")
    targets = []
    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith(("#", "track ", "browser ")):
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                raise ValueError(f"BED line {line_number} has fewer than three columns")
            contig = fields[0]
            if contig not in reference_lengths:
                raise ValueError(f"BED line {line_number} uses unknown contig {contig!r}")
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError as error:
                raise ValueError(
                    f"BED line {line_number} has non-integer coordinates"
                ) from error
            if start < 0 or end <= start:
                raise ValueError(f"BED line {line_number} is not a positive BED interval")
            if end > int(reference_lengths[contig]):
                raise ValueError(f"BED line {line_number} exceeds {contig}")
            ordinal = len(targets)
            name = fields[3].strip() if len(fields) >= 4 and fields[3].strip() else "."
            score = fields[4].strip() if len(fields) >= 5 and fields[4].strip() else "."
            strand = fields[5].strip() if len(fields) >= 6 and fields[5].strip() else "."
            if strand not in {"+", "-", "."}:
                raise ValueError(
                    f"BED line {line_number} has invalid strand {strand!r}; "
                    "expected '+', '-', or '.'"
                )
            targets.append(
                BedTarget(
                    ordinal=ordinal,
                    target_id=f"target_{ordinal + 1:06d}",
                    contig=contig,
                    start=start,
                    end=end,
                    name=name,
                    score=score,
                    strand=strand,
                )
            )
    if not targets:
        raise ValueError("BED contains no target intervals")
    return tuple(targets)


def plan_targeted_family_work_units(
    targets: Sequence[BedTarget],
    reference_lengths: Mapping[str, int],
    *,
    padding: int = 500,
    merge_gap: int = 500,
    maximum_work_unit_bp: int = 25000,
    grid_size: int = 1,
) -> Tuple[TargetedFamilyWorkUnit, ...]:
    """Merge nearby targets without ever splitting an overlapping component.

    Overlapping padded targets must stay together to prevent duplicate edge
    families. Non-overlapping targets merge only when the intervening gap is
    small and the resulting unit remains bounded. A connected component larger
    than the requested bound remains one explicitly marked oversized unit.
    """

    for name, value in (
        ("padding", padding),
        ("merge_gap", merge_gap),
        ("maximum_work_unit_bp", maximum_work_unit_bp),
        ("grid_size", grid_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if maximum_work_unit_bp < 1:
        raise ValueError("maximum_work_unit_bp must be positive")
    if grid_size < 1:
        raise ValueError("grid_size must be positive")

    expanded = []
    for target in targets:
        if target.contig not in reference_lengths:
            raise ValueError(f"target uses unknown contig {target.contig!r}")
        start, end = target.expanded(padding, int(reference_lengths[target.contig]))
        start = (start // grid_size) * grid_size
        end = min(
            int(reference_lengths[target.contig]),
            ((end + grid_size - 1) // grid_size) * grid_size,
        )
        expanded.append((target.contig, start, end, target.ordinal))
    expanded.sort(key=lambda value: (value[0], value[1], value[2], value[3]))

    provisional = []
    current_contig = None
    current_start = current_end = None
    current_targets = []
    current_oversized = False

    def finish() -> None:
        if not current_targets:
            return
        provisional.append(
            (
                current_contig,
                int(current_start),
                int(current_end),
                tuple(sorted(current_targets)),
                bool(current_oversized),
            )
        )

    for contig, start, end, ordinal in expanded:
        if current_contig != contig or not current_targets:
            finish()
            current_contig = contig
            current_start, current_end = start, end
            current_targets = [ordinal]
            current_oversized = end - start > maximum_work_unit_bp
            continue
        overlaps = start <= current_end
        near = start <= current_end + merge_gap
        proposed_end = max(current_end, end)
        bounded = proposed_end - current_start <= maximum_work_unit_bp
        if overlaps or (near and bounded):
            current_end = proposed_end
            current_targets.append(ordinal)
            current_oversized = current_oversized or (
                current_end - current_start > maximum_work_unit_bp
            )
        else:
            finish()
            current_contig = contig
            current_start, current_end = start, end
            current_targets = [ordinal]
            current_oversized = end - start > maximum_work_unit_bp
    finish()

    units = []
    for ordinal, (contig, start, end, members, oversized) in enumerate(provisional):
        digest = hashlib.sha256(
            f"{contig}:{start}-{end}|{','.join(map(str, members))}".encode("utf-8")
        ).hexdigest()[:10]
        units.append(
            TargetedFamilyWorkUnit(
                ordinal=ordinal,
                unit_id=f"unit_{ordinal + 1:06d}_{digest}",
                contig=contig,
                start=start,
                end=end,
                target_ordinals=members,
                oversized_connected_component=oversized,
            )
        )
    for left, right in zip(units, units[1:]):
        if left.contig == right.contig and left.end > right.start:
            raise AssertionError("planned work units must not overlap")
    return tuple(units)


def family_target_memberships(
    family: Mapping[str, object],
    unit: TargetedFamilyWorkUnit,
    targets: Sequence[BedTarget],
    reference_lengths: Mapping[str, int],
    *,
    padding: int,
) -> Tuple[BedTarget, ...]:
    """Return padded BED targets containing the frozen family center."""

    center_twice = int(family["start"]) + int(family["end"])
    matched = []
    for ordinal in unit.target_ordinals:
        target = targets[int(ordinal)]
        start, end = target.expanded(padding, int(reference_lengths[target.contig]))
        if 2 * start <= center_twice < 2 * end:
            matched.append(target)
    return tuple(matched)


__all__ = [
    "BedTarget",
    "TargetedFamilyWorkUnit",
    "family_target_memberships",
    "load_bed_targets",
    "plan_targeted_family_work_units",
]
