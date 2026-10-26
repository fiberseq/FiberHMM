"""Reading a scan directory: the manifest, region queries over its tables, validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .tables import Table, sha256_file
from .types import SCHEMA


class ScanDir:
    """A scan directory opened for reading (manifest + tabix readers)."""

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        manifest = self.path / 'manifest.json'
        if not manifest.is_file():
            raise FileNotFoundError(f'{self.path}: no manifest.json (not a finished fiberhmm-scan directory)')
        self.manifest = json.loads(manifest.read_text())
        if self.manifest.get('schema') != SCHEMA:
            raise ValueError(f'{manifest}: schema {self.manifest.get("schema")!r}, expected {SCHEMA!r}')
        self._tables: Dict[str, Table] = {}

    @property
    def modules(self) -> List[str]:
        return sorted(self.manifest.get('modules', {}))

    def table(self, rel: str) -> Table:
        if rel not in self._tables:
            self._tables[rel] = Table(self.path / rel)
        return self._tables[rel]

    def has(self, rel: str) -> bool:
        return (self.path / rel).is_file()

    def _fetch(self, rel, chrom, start, end) -> List[dict]:
        if not self.has(rel):
            return []
        return list(self.table(rel).fetch(chrom, start, end))

    def hits(self, module: str, chrom: str, start: Optional[int] = None, end: Optional[int] = None) -> List[dict]:
        return self._fetch(f'{module}/hits.tsv.bgz', chrom, start, end)

    def values(self, module: str, chrom: str, start=None, end=None) -> List[dict]:
        return self._fetch(f'{module}/values.tsv.bgz', chrom, start, end)

    def states(self, module: str, chrom: str, start=None, end=None) -> List[dict]:
        return self._fetch(f'{module}/states.tsv.bgz', chrom, start, end)

    def units(self, chrom: str, start=None, end=None) -> List[dict]:
        return self._fetch('units.tsv.bgz', chrom, start, end)

    def callable(self, chrom: str, start=None, end=None) -> List[dict]:
        return self._fetch('callable.tsv.bgz', chrom, start, end)

    def close(self) -> None:
        for t in self._tables.values():
            t.close()
        self._tables.clear()

    def validate(self) -> List[str]:
        """Problems found (empty when the directory is consistent): every listed table exists with its
        recorded SHA-256 and a tabix index; every annotated BAM exists with an index and its content id."""
        from .identity import content_id
        problems = []
        for rel, rec in sorted((self.manifest.get('outputs', {}).get('tables') or {}).items()):
            p = self.path / rel
            if not p.is_file():
                problems.append(f'missing table {rel}')
                continue
            if not Path(str(p) + '.tbi').is_file():
                problems.append(f'missing index {rel}.tbi')
            if sha256_file(p) != rec.get('sha256'):
                problems.append(f'{rel}: SHA-256 differs from the manifest')
            try:
                Table(p).close()
            except Exception as error:   # pragma: no cover - corrupt files
                problems.append(f'{rel}: unreadable ({error})')
        for out in self.manifest.get('outputs', {}).get('annotated') or []:
            p = self.path / out['path']
            if not p.is_file():
                problems.append(f'missing annotated BAM {out["path"]}')
                continue
            if not Path(str(p) + '.csi').is_file():
                problems.append(f'missing index {out["path"]}.csi')
            if content_id(p) != out.get('content_id'):
                problems.append(f'{out["path"]}: content id differs from the manifest')
        return problems

    def summary(self) -> dict:
        m = self.manifest
        return dict(
            scan_id=m.get('scan_id'), schema=m.get('schema'), fiberhmm=m.get('fiberhmm', {}).get('version'),
            frame=m.get('frame'), datasets=[dict(dataset_id=d['dataset_id'], ordinal=d['ordinal'],
                                                 chemistry=d.get('chemistry'),
                                                 files=[f['name'] for f in d.get('files', [])])
                                            for d in m.get('datasets', [])],
            units={k: m.get('units', {}).get(k) for k in ('mode', 'unit_bp', 'stride_bp', 'grid_offset', 'contigs',
                                                          'regions', 'bed', 'count')},
            modules={n: dict(version=v.get('version'), experimental=v.get('experimental'),
                             settings_hash=v.get('settings_hash'),
                             hits=(m.get('outputs', {}).get('tables', {}).get(f'{n}/hits.tsv.bgz') or {}).get('rows'))
                     for n, v in m.get('modules', {}).items()},
            annotate=m.get('outputs', {}).get('annotate'),
            annotated=[a['path'] for a in m.get('outputs', {}).get('annotated') or []],
            status=m.get('status'))
