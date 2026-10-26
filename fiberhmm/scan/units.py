"""Unit planning for ``fiberhmm-scan``: half-overlapping grid units with owned zones, or BED units.

A grid unit computes ``unit_bp`` and owns its central ``stride_bp``; zones
partition the scanned territory, and every owned base has at least
``(unit_bp - stride_bp) / 2`` bp of context on each side inside its unit
(clipped at contig and region ends). BED intervals are merged when they
overlap or touch; a merged interval longer than ``unit_bp`` is planned like a
grid restricted to it. BED and grid units are never mixed in one plan.
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from .types import ScanUnit

LATTICE_BP = 250            # the lattice recaller tiles on 250 bp multiples
DEFAULT_UNIT_BP = 50_000
DEFAULT_STRIDE_BP = 25_000


class PlanError(ValueError):
    """A plan request that cannot be honoured (refused rather than silently changed)."""


def check_grid(unit_bp: int, stride_bp: int, max_unit_bp: int) -> None:
    for name, value in (('unit_bp', unit_bp), ('stride_bp', stride_bp)):
        if int(value) != value or value <= 0 or value % LATTICE_BP:
            raise PlanError(f'{name} must be a positive multiple of {LATTICE_BP} bp (got {value})')
    if stride_bp > unit_bp:
        raise PlanError(f'stride_bp ({stride_bp}) must not exceed unit_bp ({unit_bp})')
    if (unit_bp - stride_bp) % 2:
        raise PlanError('unit_bp - stride_bp must be even (equal context on both sides of a zone)')
    if unit_bp > max_unit_bp:
        raise PlanError(f'unit_bp {unit_bp:,} exceeds the engines\' {max_unit_bp:,} bp analysis budget '
                        '(compute.maximum_region_bp); it is never raised silently')


def _grid_zones(lo: int, hi: int, stride: int, offset: int) -> List[Tuple[int, int]]:
    """Zones partitioning [lo, hi): boundaries at lo-relative offset + k * stride."""
    bounds = [lo]
    first = lo + (offset % stride if offset else stride)
    b = first
    while b < hi:
        if b > lo:
            bounds.append(b)
        b += stride
    bounds.append(hi)
    return [(a, z) for a, z in zip(bounds, bounds[1:]) if z > a]


def _grid_units(chrom: str, lo: int, hi: int, unit_bp: int, stride_bp: int, offset: int,
                start_index: int) -> List[ScanUnit]:
    margin = (unit_bp - stride_bp) // 2
    zones = _grid_zones(lo, hi, stride_bp, offset)
    spans = [(max(lo, a - margin), min(hi, z + margin), a, z) for a, z in zones]
    ids = [f'{chrom}:{s}-{e}' for s, e, _, _ in spans]
    units = []
    for i, (s, e, a, z) in enumerate(spans):
        units.append(ScanUnit(unit_id=ids[i], chrom=chrom, start=s, end=e, zone_start=a, zone_end=z,
                              mode='grid', index=start_index + i,
                              left=ids[i - 1] if i > 0 else None,
                              right=ids[i + 1] if i + 1 < len(ids) else None))
    return units


def read_bed(path) -> List[Tuple[str, int, int, str]]:
    """BED3+ rows (chrom, start, end, name); plain or gzip/bgzip; '#', 'track', 'browser' lines skipped."""
    p = Path(path)
    opener = gzip.open if p.suffix in ('.gz', '.bgz') else open
    rows = []
    with opener(p, 'rt') as handle:
        for n, line in enumerate(handle, 1):
            if not line.strip() or line.startswith(('#', 'track', 'browser')):
                continue
            f = line.rstrip('\n').split('\t')
            if len(f) < 3:
                raise PlanError(f'{p}:{n}: a BED row needs chrom, start and end')
            try:
                a, b = int(f[1]), int(f[2])
            except ValueError:
                raise PlanError(f'{p}:{n}: start and end must be integers') from None
            if a < 0 or b <= a:
                raise PlanError(f'{p}:{n}: need 0 <= start < end (0-based half-open)')
            rows.append((f[0], a, b, f[3] if len(f) > 3 else ''))
    return rows


def merge_intervals(rows: Iterable[Tuple[str, int, int, str]]):
    """Merge overlapping or bookended intervals per contig; returns (merged, merge records)."""
    by_chrom = {}
    for chrom, a, b, name in rows:
        by_chrom.setdefault(chrom, []).append((a, b, name))
    merged, records = [], []
    for chrom in by_chrom:
        cur = None
        for a, b, name in sorted(by_chrom[chrom]):
            if cur is not None and a <= cur[1]:
                cur[1] = max(cur[1], b)
                cur[2].append(name)
                cur[3].append([a, b])
            else:
                if cur is not None:
                    merged.append((chrom, cur[0], cur[1], ';'.join(n for n in cur[2] if n)))
                    if len(cur[3]) > 1:
                        records.append(dict(chrom=chrom, start=cur[0], end=cur[1], parts=cur[3]))
                cur = [a, b, [name], [[a, b]]]
        if cur is not None:
            merged.append((chrom, cur[0], cur[1], ';'.join(n for n in cur[2] if n)))
            if len(cur[3]) > 1:
                records.append(dict(chrom=chrom, start=cur[0], end=cur[1], parts=cur[3]))
    return merged, records


def plan(reference: Mapping[str, int], *, genome: bool = False, contigs: Optional[Sequence[str]] = None,
         regions: Optional[Sequence[Tuple[str, int, int]]] = None, bed=None,
         unit_bp: int = DEFAULT_UNIT_BP, stride_bp: int = DEFAULT_STRIDE_BP, grid_offset: int = 0,
         max_unit_bp: int = 50_000):
    """Plan units. Returns ``(units, info)``; ``info`` is the manifest's ``units`` block (minus count).

    ``reference`` maps contig -> length in header order. Exactly one of
    ``genome``, ``regions`` and ``bed`` selects the territory."""
    check_grid(unit_bp, stride_bp, max_unit_bp)
    if sum(bool(x) for x in (genome, regions, bed)) != 1:
        raise PlanError('choose exactly one of genome, regions and bed')
    if grid_offset < 0 or grid_offset % LATTICE_BP:
        raise PlanError(f'grid_offset must be a non-negative multiple of {LATTICE_BP} bp')
    order = {c: i for i, c in enumerate(reference)}
    units: List[ScanUnit] = []
    info = dict(unit_bp=unit_bp, stride_bp=stride_bp, grid_offset=grid_offset, contigs=None,
                regions=None, bed=None, merged=[])
    if genome:
        chosen = list(contigs) if contigs else list(reference)
        missing = [c for c in chosen if c not in reference]
        if missing:
            raise PlanError(f'contig(s) not in the BAM header: {", ".join(missing)}')
        chosen.sort(key=lambda c: order[c])
        info.update(mode='grid', contigs=chosen)
        for chrom in chosen:
            units.extend(_grid_units(chrom, 0, int(reference[chrom]), unit_bp, stride_bp, grid_offset, len(units)))
        return tuple(units), info
    if regions:
        rows = []
        for chrom, a, b in regions:
            if chrom not in reference:
                raise PlanError(f'--region {chrom}:{a}-{b}: contig not in the BAM header')
            if a < 0 or b <= a or b > int(reference[chrom]):
                raise PlanError(f'--region {chrom}:{a}-{b}: need 0 <= start < end <= contig length')
            rows.append((chrom, int(a), int(b), f'region_{len(rows) + 1}'))
        info['regions'] = [[c, a, b] for c, a, b, _ in rows]
    else:
        rows = read_bed(bed)
        for chrom, a, b, _ in rows:
            if chrom not in reference:
                raise PlanError(f'BED interval {chrom}:{a}-{b}: contig not in the BAM header')
            if b > int(reference[chrom]):
                raise PlanError(f'BED interval {chrom}:{a}-{b}: beyond the end of {chrom}')
        info['bed'] = str(Path(bed).resolve())
    merged, records = merge_intervals(rows)
    merged.sort(key=lambda r: (order[r[0]], r[1], r[2]))
    info['merged'] = records
    gridded = False
    for chrom, a, b, name in merged:
        if b - a <= unit_bp:
            units.append(ScanUnit(unit_id=f'{chrom}:{a}-{b}', chrom=chrom, start=a, end=b, zone_start=a,
                                  zone_end=b, mode='bed', index=len(units), name=name))
        else:
            gridded = True
            sub = _grid_units(chrom, a, b, unit_bp, stride_bp, 0, len(units))
            units.extend(ScanUnit(**{**u.to_json(), 'name': name}) for u in sub)
    # A --region longer than one unit is a grid scan restricted to it; a BED file stays a BED scan.
    info['mode'] = 'grid' if (regions and gridded) else 'bed'
    return tuple(units), info


def default_annotate(info: Mapping) -> bool:
    """Annotated BAMs by default for grid scans (--genome, or a --region longer than one unit);
    the TSV fallback for BED-scoped scans."""
    return info.get('mode') == 'grid'


class UnitIndex:
    """Zone and span lookups over a plan (per contig, by bisection)."""

    def __init__(self, units: Sequence[ScanUnit]):
        import bisect
        self._bisect = bisect
        self.units = tuple(units)
        self.by_chrom = {}
        for u in self.units:
            self.by_chrom.setdefault(u.chrom, []).append(u)
        for chrom in self.by_chrom:
            self.by_chrom[chrom].sort(key=lambda u: (u.zone_start, u.zone_end))
        self.zone_starts = {c: [u.zone_start for u in us] for c, us in self.by_chrom.items()}

    def zone_unit(self, chrom: str, pos: int) -> Optional[ScanUnit]:
        """The unit whose zone holds ``pos``, else None."""
        us = self.by_chrom.get(chrom)
        if not us:
            return None
        i = self._bisect.bisect_right(self.zone_starts[chrom], pos) - 1
        if i >= 0 and us[i].zone_start <= pos < us[i].zone_end:
            return us[i]
        return None

    def first_overlapping(self, chrom: str, start: int, end: int) -> Optional[ScanUnit]:
        """The first unit in plan order whose span overlaps [start, end)."""
        best = None
        for u in self.by_chrom.get(chrom, ()):
            if u.start < max(end, start + 1) and start < u.end and (best is None or u.index < best.index):
                best = u
        return best

    def owner_of_record(self, chrom: str, start: int, end: int) -> Optional[ScanUnit]:
        """The unit that owns a record's molecule-wide outputs: the zone holding its start, else
        (BED plans) the first unit in plan order that the record overlaps."""
        return self.zone_unit(chrom, start) or self.first_overlapping(chrom, start, end)
