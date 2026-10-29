"""bgzip + tabix TSV tables of a scan directory: deterministic writer and region reader.

Every table starts with ``chrom start end idx_start idx_end``. ``idx_start`` /
``idx_end`` are the columns tabix indexes (``tabix -0 -s1 -b4 -e5``); they equal
``start``/``end`` except for zero-width rows, which index ``[p, p + 1)`` so a
fetch at ``p`` finds them. Rows are sorted by contig order (the reference
header's), ``idx_start``, ``idx_end`` and then the remaining cells as text, so
the same rows always give the same bytes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

COMMON = ('chrom', 'start', 'end', 'idx_start', 'idx_end')
TBI_MAX = (1 << 29) - 1
_SORT_ROWS = 50_000
_MERGE_FANIN = 32


def index_path(path) -> Path:
    """The table's index: ``.tbi``, or ``.csi`` for tables with positions beyond 2^29."""
    p = Path(path)
    csi = Path(str(p) + '.csi')
    return csi if csi.is_file() else Path(str(p) + '.tbi')


def cell(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, bool):
        return 'True' if value else 'False'
    if isinstance(value, float):
        return '' if (math.isnan(value) or math.isinf(value)) else repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (list, tuple)):
        return ','.join(cell(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(',', ':'))
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return cell(value.item())
    except ImportError:  # pragma: no cover
        pass
    text = str(value)
    if '\t' in text or '\n' in text:
        raise ValueError(f'table cell contains a tab or newline: {text[:40]!r}')
    return text


def index_span(start: int, end: int):
    start, end = int(start), int(end)
    return (start, end) if end > start else (start, start + 1)


class TableWriter:
    """Publishes a sorted, BGZF-compressed, indexed table using bounded merge-sort runs."""

    def __init__(self, path, columns: Sequence[str], contig_order: Mapping[str, int]):
        self.path = Path(path)
        self.columns = list(columns)
        if tuple(self.columns[:5]) != COMMON:
            raise ValueError(f'{self.path.name}: a scan table must start with {COMMON}')
        self.order = dict(contig_order)
        self._rows: List[tuple] = []
        self._count = 0
        self._csi = False
        self._sort_dir = None
        self._runs = []
        self._run_number = 0

    def add(self, chrom: str, start: int, end: int, values: Sequence[Any]) -> None:
        if len(values) != len(self.columns) - 5:
            raise ValueError(f'{self.path.name}: {len(values)} cells for {len(self.columns) - 5} columns')
        if chrom not in self.order:
            raise ValueError(f'{self.path.name}: contig {chrom!r} is not in the reference')
        a, b = index_span(start, end)
        self._rows.append((self.order[chrom], a, b, chrom, int(start), int(end),
                           tuple(cell(v) for v in values)))
        self._count += 1
        self._csi |= b > TBI_MAX
        if len(self._rows) >= _SORT_ROWS:
            self._spill()

    def __len__(self) -> int:
        return self._count

    @staticmethod
    def _key(row):
        return row[0], row[1], row[2], row[6]

    def _new_run(self):
        if self._sort_dir is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._sort_dir = tempfile.TemporaryDirectory(prefix='.' + self.path.name + '.sort.',
                                                         dir=self.path.parent)
        self._run_number += 1
        return Path(self._sort_dir.name) / str(self._run_number)

    def _spill(self):
        import pickle
        if not self._rows:
            return
        self._rows.sort(key=self._key)
        path = self._new_run()
        with open(path, 'wb') as handle:
            for row in self._rows:
                pickle.dump(row, handle, protocol=4)
        self._runs.append(path)
        self._rows = []

    def _merge(self, paths):
        import heapq
        import pickle
        from contextlib import ExitStack
        def rows(handle):
            while True:
                try:
                    yield pickle.load(handle)
                except EOFError:
                    return
        with ExitStack() as stack:
            streams = [rows(stack.enter_context(open(path, 'rb'))) for path in paths]
            # heapq.merge is stable between streams, preserving insertion order
            # even when zero-width and one-base rows have identical sort keys.
            yield from heapq.merge(*streams, key=self._key)

    def _sorted_rows(self):
        import pickle
        if not self._runs:
            self._rows.sort(key=self._key)
            yield from self._rows
            return
        self._spill()
        while len(self._runs) > _MERGE_FANIN:
            merged = []
            for i in range(0, len(self._runs), _MERGE_FANIN):
                paths = self._runs[i:i + _MERGE_FANIN]
                dest = self._new_run()
                with open(dest, 'wb') as handle:
                    for row in self._merge(paths):
                        pickle.dump(row, handle, protocol=4)
                for path in paths:
                    path.unlink()
                merged.append(dest)
            self._runs = merged
        yield from self._merge(self._runs)

    def close(self) -> dict:
        """Write ``path`` and ``path.tbi``; returns ``{"rows", "sha256"}`` of the compressed table."""
        import pysam
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw = tempfile.mkstemp(prefix='.' + self.path.name + '.', suffix='.tsv', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w', newline='') as handle:
                handle.write('#' + '\t'.join(self.columns) + '\n')
                for _o, a, b, chrom, start, end, cells in self._sorted_rows():
                    handle.write('\t'.join((chrom, str(start), str(end), str(a), str(b), *cells)) + '\n')
            tmp_gz = raw + '.bgz'
            pysam.tabix_compress(raw, tmp_gz, force=True)
            # TBI addresses positions below 2^29; a table reaching further gets a CSI index instead.
            csi = self._csi
            pysam.tabix_index(tmp_gz, seq_col=0, start_col=3, end_col=4, zerobased=True,
                              meta_char='#', force=True, csi=csi)
            suffix = '.csi' if csi else '.tbi'
            for other in ('.tbi', '.csi'):
                if other != suffix and os.path.exists(str(self.path) + other):
                    os.unlink(str(self.path) + other)
            os.replace(tmp_gz, self.path)
            os.replace(tmp_gz + suffix, str(self.path) + suffix)
        finally:
            for leftover in (raw, raw + '.bgz', raw + '.bgz.tbi', raw + '.bgz.csi'):
                if os.path.exists(leftover):
                    os.unlink(leftover)
            if self._sort_dir is not None:
                self._sort_dir.cleanup()
                self._sort_dir = None
                self._runs = []
        n = self._count
        self._count = 0
        self._csi = False
        self._rows = []
        return dict(rows=n, sha256=sha256_file(self.path))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class Table:
    """Region reader over one scan table."""

    def __init__(self, path):
        import pysam
        self.path = Path(path)
        self._tabix = pysam.TabixFile(str(self.path), index=str(index_path(self.path)))
        header = list(self._tabix.header)
        if not header or not header[-1].startswith('#'):
            raise ValueError(f'{self.path}: missing header line')
        self.columns = header[-1][1:].split('\t')

    @property
    def contigs(self) -> List[str]:
        return list(self._tabix.contigs)

    def fetch(self, chrom: str, start: Optional[int] = None, end: Optional[int] = None) -> Iterator[Dict[str, str]]:
        if chrom not in self._tabix.contigs:
            return iter(())
        it = self._tabix.fetch(chrom, start, end) if start is not None else self._tabix.fetch(chrom)
        return (dict(zip(self.columns, line.split('\t'))) for line in it)

    def rows(self) -> Iterator[Dict[str, str]]:
        for chrom in self._tabix.contigs:
            yield from self.fetch(chrom)

    def close(self) -> None:
        self._tabix.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_all(path) -> List[Dict[str, str]]:
    """Every row of a table, in file order (header via the gzip stream, no index needed)."""
    import gzip
    with gzip.open(path, 'rt') as handle:
        lines = handle.read().splitlines()
    if not lines or not lines[0].startswith('#'):
        raise ValueError(f'{path}: missing header line')
    cols = lines[0][1:].split('\t')
    return [dict(zip(cols, line.split('\t'))) for line in lines[1:]]
