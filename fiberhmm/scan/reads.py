"""Source records for the orchestrator: read policy, record identity, unit emptiness and callable territory.

The fp and nfr modules keep their own loaders; this reader serves what the
orchestrator itself needs (empty units, callable territory, molecule-wide
outputs and annotation) and the full-alignment modules (peaks, nuc).
"""
from __future__ import annotations

import copy
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from .identity import molecule_id, record_sha256
from .types import Dataset, RecordRef, ScanUnit

DEFAULT_READ_POLICY = {
    'filters': {'min_mapq': 0, 'secondary': 'exclude', 'supplementary': 'include', 'duplicates': 'exclude',
                'unmapped': 'exclude', 'strand': 'both'},
    'caps': {'max_discovery_molecules': 0, 'max_quantification_molecules': 0, 'sample_seed': 1},
    'callable_min_molecules': 5,
}

_FILTER_CHOICES = {'secondary': ('exclude', 'include'), 'supplementary': ('include', 'exclude'),
                   'duplicates': ('exclude', 'include'), 'unmapped': ('exclude',), 'strand': ('both',)}


def resolve_read_policy(user: Optional[Mapping] = None) -> dict:
    """The default read policy updated with ``user`` (nested dicts merged); unknown keys are refused."""
    out = copy.deepcopy(DEFAULT_READ_POLICY)
    for key, value in (user or {}).items():
        if key in ('filters', 'caps'):
            unknown = set(value) - set(out[key])
            if unknown:
                raise ValueError(f'read_policy.{key}: unknown key(s) {", ".join(sorted(unknown))}')
            out[key].update(value)
        elif key == 'callable_min_molecules':
            out[key] = int(value)
        elif key == 'browser_compact_reads':
            out[key] = dict(value)
        else:
            raise ValueError(f'read_policy: unknown key {key!r}')
    for key, choices in _FILTER_CHOICES.items():
        if out['filters'][key] not in choices:
            raise ValueError(f'read_policy.filters.{key} must be one of {choices}')
    if int(out['filters']['min_mapq']) < 0 or out['callable_min_molecules'] < 1:
        raise ValueError('read_policy: min_mapq must be >= 0 and callable_min_molecules >= 1')
    for k in out['caps']:
        if int(out['caps'][k]) < 0:
            raise ValueError(f'read_policy.caps.{k} must be >= 0')
    return out


def passes(read, filters: Mapping) -> bool:
    if read.is_unmapped:
        return False
    if read.is_secondary and filters.get('secondary', 'exclude') == 'exclude':
        return False
    if read.is_supplementary and filters.get('supplementary', 'include') == 'exclude':
        return False
    if read.is_duplicate and filters.get('duplicates', 'exclude') == 'exclude':
        return False
    return read.mapping_quality >= int(filters.get('min_mapq', 0))


def has_annotations(read) -> bool:
    return any(read.has_tag(t) for t in ('MA', 'Ma', 'ns'))


def iter_records(bam, chrom: str, start: int, end: int, filters: Mapping
                 ) -> Iterator[Tuple[object, str, int]]:
    """Records overlapping [start, end) that pass ``filters``, with (record_sha256, occurrence).

    Identical records share a position, so one fetch sees every copy and the
    occurrence equals the file-wide occurrence. The SHA-256 is computed lazily
    only for records that pass."""
    seen: Dict[str, int] = {}
    for read in bam.fetch(chrom, max(0, start), end):
        if not passes(read, filters):
            continue
        sha = record_sha256(read)
        occ = seen.get(sha, 0)
        seen[sha] = occ + 1
        yield read, sha, occ


def record_ref(read, dataset: Dataset, input_index: int, sha: str, occ: int, chrom: str,
               molecule: Optional[str] = None) -> RecordRef:
    return RecordRef(dataset.dataset_id, int(input_index), str(read.query_name), sha, int(occ), chrom,
                     int(read.reference_start), int(read.reference_end or read.reference_start),
                     molecule if molecule is not None else molecule_id([read.query_name]))


def unit_survey(datasets: Sequence[Dataset], unit: ScanUnit, read_policy: Mapping) -> dict:
    """Mapped records overlapping the unit span (any flags), and callable runs over its zone per dataset.

    callable: bases covered (aligned blocks) by at least ``callable_min_molecules``
    annotated records of the dataset."""
    import numpy as np
    import pysam
    filters = read_policy['filters']
    min_mol = int(read_policy['callable_min_molecules'])
    zlen = unit.zone_end - unit.zone_start
    records = 0
    callable_runs = {}
    for ds in datasets:
        delta = np.zeros(zlen + 1, dtype=np.int64)
        for path in ds.paths:
            with pysam.AlignmentFile(path, 'rb') as bam:
                if unit.chrom not in bam.references:
                    continue
                for read in bam.fetch(unit.chrom, unit.start, unit.end):
                    if read.is_unmapped:
                        continue
                    # Emptiness counts every mapped record, whatever the filters: a module may admit records
                    # the read policy leaves out (peaks with include_duplicates, say).
                    records += 1
                    if not passes(read, filters) or not has_annotations(read):
                        continue
                    for a, b in read.get_blocks():
                        a = max(a, unit.zone_start) - unit.zone_start
                        b = min(b, unit.zone_end) - unit.zone_start
                        if b > a:
                            delta[a] += 1
                            delta[b] -= 1
        cov = np.cumsum(delta[:-1])
        ok = cov >= min_mol
        runs = []
        if ok.any():
            padded = np.concatenate([[False], ok, [False]]).astype(np.int8)
            d = np.diff(padded)
            for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
                runs.append([int(a) + unit.zone_start, int(b) + unit.zone_start])
        callable_runs[ds.dataset_id] = runs
    return dict(records=records, callable=callable_runs)


def overlapping_records(datasets: Sequence[Dataset], unit: ScanUnit, read_policy: Mapping):
    """Every source record overlapping the unit's span that passes the read policy:
    (dataset, input_index, read, sha, occ)."""
    import pysam
    filters = read_policy['filters']
    for ds in datasets:
        for i, path in enumerate(ds.paths):
            with pysam.AlignmentFile(path, 'rb') as bam:
                if unit.chrom not in bam.references:
                    continue
                for read, sha, occ in iter_records(bam, unit.chrom, unit.start, unit.end, filters):
                    yield ds, i, read, sha, occ


def owned_records(datasets: Sequence[Dataset], unit: ScanUnit, index, read_policy: Mapping):
    """Every source record whose molecule-wide outputs this unit owns: (dataset, input_index, read, sha, occ)."""
    import pysam
    filters = read_policy['filters']
    for ds in datasets:
        for i, path in enumerate(ds.paths):
            with pysam.AlignmentFile(path, 'rb') as bam:
                if unit.chrom not in bam.references:
                    continue
                for read, sha, occ in iter_records(bam, unit.chrom, unit.start, unit.end, filters):
                    end = int(read.reference_end or read.reference_start + 1)
                    owner = index.owner_of_record(unit.chrom, int(read.reference_start), end)
                    if owner is not None and owner.unit_id == unit.unit_id:
                        yield ds, i, read, sha, occ
