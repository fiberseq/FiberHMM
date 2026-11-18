"""Streaming motif input from bigBed files (e.g. JASPAR genome-wide TFBS tracks).

A bigBed is queried region by region, never expanded whole: JASPAR tracks hold
hundreds of millions of sites. Two readers are supported, neither a hard
dependency of FiberHMM:

* ``pyBigWig`` (``pip install 'fiberhmm[bigbed]'``), used when importable;
* the UCSC ``bigBedToBed`` binary on PATH (``-chrom/-start/-end`` per region),
  with ``bigBedInfo -as`` for field names when available.
"""
from __future__ import annotations

import math
import re
import shutil
import struct
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

BIGBED_MAGIC = 0x8789F2EB
INSTALL_HINT = ("reading bigBed needs pyBigWig (pip install 'fiberhmm[bigbed]' or pip install pyBigWig) "
                "or the UCSC bigBedToBed binary on PATH "
                "(https://hgdownload.soe.ucsc.edu/admin/exe/, or conda install -c bioconda ucsc-bigbedtobed)")


def is_bigbed(path) -> bool:
    """True when the file starts with the bigBed signature (either byte order)."""
    try:
        with open(path, 'rb') as handle:
            head = handle.read(4)
    except OSError:
        return False
    if len(head) < 4:
        return False
    return BIGBED_MAGIC in (struct.unpack('<I', head)[0], struct.unpack('>I', head)[0])


def _autosql_fields(text: str) -> List[Tuple[str, str]]:
    """``[(name, description)]`` of an autoSql table definition."""
    out = []
    for line in (text or '').splitlines():
        m = re.match(r'\s*[\w\[\]]+(?:\s+\w+)?\s+(\w+)\s*;\s*"?([^"]*)"?', line)
        if m and not line.strip().startswith(('table', '(', ')')):
            out.append((m.group(1), m.group(2)))
    return out


class BigBedReader:
    """Region queries on a bigBed: ``chroms()``, ``fields`` and ``entries(chrom, start, end)``
    yielding ``(chrom, start, end, [field4, field5, ...])``."""

    def __init__(self, path, backend: Optional[str] = None):
        self.path = str(path)
        self.backend = backend or self._choose()
        if self.backend == 'pybigwig':
            import pyBigWig
            self._bb = pyBigWig.open(self.path)
            if self._bb is None or not self._bb.isBigBed():
                raise ValueError(f'{self.path}: not a readable bigBed')
            sql = self._bb.SQL()
            sql = sql.decode() if isinstance(sql, bytes) else (sql or '')
            self._chroms = dict(self._bb.chroms())
        else:
            self._bin = shutil.which('bigBedToBed')
            info = shutil.which('bigBedInfo')
            sql = ''
            if info:
                res = subprocess.run([info, '-as', '-chroms', self.path], capture_output=True, text=True)
                if res.returncode == 0:
                    sql = res.stdout.split('as:', 1)[1] if 'as:' in res.stdout else ''
                    self._chroms = self._parse_info_chroms(res.stdout)
                else:
                    self._chroms = None
            else:
                self._chroms = None
        fields = _autosql_fields(sql)
        if not fields and self.backend != 'pybigwig':
            fields = [(name, '') for name in self._header_fields()]
        self.fields = [name for name, _ in fields] or ['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand']
        self.descriptions = dict(fields)

    def _header_fields(self) -> List[str]:
        """Field names from ``bigBedToBed -header`` (no bigBedInfo needed): one tiny query."""
        # The header line comes first even for an empty query, so ask for a
        # region no item can occupy (start = end is refused, so use 0-1 on an
        # unlikely contig) and stop after the first line.
        proc = subprocess.Popen([self._bin, '-header', '-chrom=__fiberhmm_none__', '-start=0', '-end=1',
                                 self.path, 'stdout'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        first = proc.stdout.readline()
        proc.stdout.close()
        proc.wait()
        return first.rstrip('\n').lstrip('#').split('\t') if first.startswith('#') else []

    @staticmethod
    def _choose() -> str:
        try:
            import pyBigWig  # noqa: F401
            return 'pybigwig'
        except ImportError:
            pass
        if shutil.which('bigBedToBed'):
            return 'bigbedtobed'
        raise ValueError(INSTALL_HINT)

    @staticmethod
    def _parse_info_chroms(text: str) -> Optional[Dict[str, int]]:
        chroms = {}
        block = text.split('chromCount:', 1)
        if len(block) < 2:
            return None
        for line in block[1].splitlines()[1:]:
            parts = line.split()
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                chroms[parts[0]] = int(parts[2])
            elif parts and not line.startswith('\t') and not line.startswith(' '):
                break
        return chroms or None

    def chroms(self) -> Optional[Dict[str, int]]:
        return self._chroms

    def entries(self, chrom: str, start: int, end: int) -> Iterator[Tuple[str, int, int, List[str]]]:
        if self.backend == 'pybigwig':
            if chrom not in self._chroms:
                return
            end = min(end, self._chroms[chrom])
            if end <= start:
                return
            for s, e, rest in self._bb.entries(chrom, start, end) or ():
                yield chrom, int(s), int(e), rest.split('\t') if rest else []
            return
        if self._chroms is not None:
            if chrom not in self._chroms:
                return
            end = min(end, self._chroms[chrom])
            if end <= start:
                return
        proc = subprocess.Popen([self._bin, f'-chrom={chrom}', f'-start={max(0, start)}', f'-end={end}',
                                 self.path, 'stdout'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in proc.stdout:
            f = line.rstrip('\n').split('\t')
            if len(f) >= 3:
                yield f[0], int(f[1]), int(f[2]), f[3:]
        proc.stdout.close()
        err = proc.stderr.read()
        if proc.wait() != 0:
            # Without bigBedInfo the chromosome sizes are unknown; bigBedToBed names them on overrun.
            size = re.search(r'chromSize=(\d+)', err)
            if size and int(size.group(1)) < end:
                yield from self.entries(chrom, start, int(size.group(1)))
                return
            if 'not in' in err or 'not found' in err.lower():
                return
            raise ValueError(f'bigBedToBed failed on {self.path} {chrom}:{start}-{end}: {err.strip()}')

    def close(self):
        if self.backend == 'pybigwig':
            self._bb.close()


@dataclass
class MotifFilter:
    """Row filters for bigBed (and BED) motifs.

    ``names`` match, case-insensitively, the name field, any TF-name field
    (e.g. JASPAR's ``TFName``) and matrix IDs with or without the version
    (``MA0139`` matches ``MA0139.2``). ``min_score`` applies to the score
    column. ``max_pvalue`` applies to a p-value field when the file has one
    (values read as -log10(p) when the field's name or description says log)."""
    names: Optional[Sequence[str]] = None
    min_score: Optional[float] = None
    max_pvalue: Optional[float] = None

    def __post_init__(self):
        self._names = {n.strip().lower() for n in self.names or () if n.strip()} or None


def _field_index(fields: Sequence[str], pattern: str) -> Optional[int]:
    """Index into the rest-of-line fields (after chrom/start/end) of the first field matching ``pattern``."""
    for i, name in enumerate(fields[3:]):
        if re.fullmatch(pattern, name, flags=re.I):
            return i
    return None


class RowInterpreter:
    """Turns bigBed rows into ``(name, label_source, strand, score, keep)``."""

    def __init__(self, fields: Sequence[str], descriptions: Dict[str, str], filt: MotifFilter):
        self.filt = filt
        self.i_name = _field_index(fields, 'name')
        self.i_score = _field_index(fields, 'score')
        self.i_strand = _field_index(fields, 'strand')
        self.i_tf = _field_index(fields, r'tf_?name|tf|gene_?name|gene|factor')
        self.i_p = _field_index(fields, r'p_?val(ue)?|log_?p_?val(ue)?|minus_?log10_?p(val(ue)?)?')
        if self.i_name is None and len(fields) <= 3:
            self.i_name = 0
        if self.i_score is None and self.i_name == 0:
            self.i_score = 1
        if self.i_strand is None and self.i_name == 0:
            self.i_strand = 2
        self.p_is_log = False
        if self.i_p is not None:
            pname = fields[3 + self.i_p]
            self.p_is_log = 'log' in pname.lower() or 'log' in descriptions.get(pname, '').lower()
        if filt.max_pvalue is not None and self.i_p is None:
            raise ValueError('--max-pvalue: this bigBed has no p-value field (JASPAR UCSC tracks encode the '
                             'p-value in the score column: score = -log10(p) x 100, so use --min-score)')

    @staticmethod
    def _get(rest, i):
        return rest[i] if i is not None and i < len(rest) else None

    def interpret(self, rest: List[str]):
        name = self._get(rest, self.i_name) or 'motif'
        tf = self._get(rest, self.i_tf)
        strand = self._get(rest, self.i_strand)
        strand = '-' if strand == '-' else '+'
        f = self.filt
        if f._names is not None:
            keys = {name.lower(), name.split('.')[0].lower()}
            if tf:
                keys.add(tf.lower())
            if not keys & f._names:
                return None
        if f.min_score is not None:
            try:
                if float(self._get(rest, self.i_score)) < f.min_score:
                    return None
            except (TypeError, ValueError):
                return None
        if f.max_pvalue is not None:
            try:
                v = float(self._get(rest, self.i_p))
            except (TypeError, ValueError):
                return None
            p = 10 ** (-v) if self.p_is_log else v
            if p > f.max_pvalue:
                return None
        return name, (tf or name), strand


def parse_region(value: str, chroms: Optional[Dict[str, int]] = None) -> Tuple[str, int, int]:
    chrom, sep, span = value.rpartition(':')
    if not sep:
        if chroms is not None and value not in chroms:
            raise ValueError(f'--region {value!r}: unknown contig')
        return value, 0, (chroms or {}).get(value, 2 ** 31 - 1)
    a, dash, b = span.replace(',', '').partition('-')
    try:
        start, end = int(a), int(b)
    except ValueError:
        raise ValueError(f'--region {value!r}: expected CHROM or CHROM:START-END') from None
    if not dash or start < 0 or end <= start:
        raise ValueError(f'--region {value!r}: expected CHROM:START-END with END > START >= 0')
    return chrom, start, end
