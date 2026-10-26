"""Data types shared by ``fiberhmm-scan``, its modules and its readers.

Pure dataclasses with no heavy imports, so the browser backend and other
modules can import them cheaply. Coordinates are 0-based half-open on the
scan's reference. The contract these types implement is the scan format
(``docs/reference/scan-format.md``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, Mapping, Optional, Protocol, Tuple

SCHEMA = 'fiberhmm.scan.v1'


@dataclass(frozen=True)
class Dataset:
    """One logical dataset of a scan (one or more BAMs pooled as one)."""
    dataset_id: str
    ordinal: int                        # canonical: rank by the files' content ids
    chemistry: Optional[str]
    paths: Tuple[str, ...]              # absolute; input_index = position here
    content_ids: Tuple[str, ...]

    def to_json(self) -> dict:
        return dict(dataset_id=self.dataset_id, ordinal=self.ordinal, chemistry=self.chemistry,
                    paths=list(self.paths), content_ids=list(self.content_ids))


@dataclass(frozen=True)
class ScanUnit:
    """A fixed unit of computation and the zone of the contig it owns."""
    unit_id: str                        # '<chrom>:<start>-<end>' of the computed span
    chrom: str
    start: int
    end: int
    zone_start: int
    zone_end: int
    mode: str                           # 'grid' | 'bed'
    index: int
    name: str = ''
    left: Optional[str] = None
    right: Optional[str] = None

    @property
    def centre(self) -> float:
        return (self.start + self.end) / 2.0

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> 'ScanUnit':
        return cls(**{f.name: value[f.name] for f in fields(cls) if f.name in value})


@dataclass(frozen=True)
class RecordRef:
    """A source SAM record: the identity scan rows and MA marks refer to."""
    dataset_id: str
    input_index: int
    qname: str
    record_sha256: str
    occurrence: int
    chrom: str
    ref_start: int
    ref_end: int
    molecule: str = ''

    @property
    def record(self) -> str:
        return f'{self.record_sha256}:{self.occurrence}'

    @property
    def key(self) -> tuple:
        """(dataset_id, input_index, record_sha256, occurrence): unique per source record."""
        return (self.dataset_id, self.input_index, self.record_sha256, self.occurrence)

    def to_json(self) -> list:
        return [self.dataset_id, self.input_index, self.qname, self.record_sha256, self.occurrence,
                self.chrom, self.ref_start, self.ref_end, self.molecule]

    @classmethod
    def from_json(cls, value) -> 'RecordRef':
        return cls(str(value[0]), int(value[1]), str(value[2]), str(value[3]), int(value[4]),
                   str(value[5]), int(value[6]), int(value[7]), str(value[8]))


@dataclass
class ScanInputs:
    datasets: Tuple[Dataset, ...]
    read_policy: dict
    reference: Mapping[str, int]
    plan: Tuple[ScanUnit, ...]
    cores: int = 1


class Progress(Protocol):
    def report(self, stage: str, message: str, **work: Any) -> None: ...


@dataclass
class Hit:
    type: str
    key: str
    chrom: str
    start: int
    end: int
    anchor: int
    strand: str = '.'
    score: Optional[float] = None
    label: str = ''
    flags: Tuple[str, ...] = ()
    parent: Optional[str] = None
    columns: dict = field(default_factory=dict)

    @property
    def zero_width(self) -> bool:
        return self.end == self.start

    @property
    def index_span(self) -> Tuple[int, int]:
        """The span tabix indexes: [p, p + 1) for a zero-width hit at p."""
        return (self.start, self.end) if self.end > self.start else (self.start, self.start + 1)


@dataclass
class Value:
    hit_key: str
    group: str
    group_kind: str                     # 'all' | 'dataset' | 'channel' | 'haplotype'
    value_name: str
    value: Optional[float]
    lo: Optional[float] = None
    hi: Optional[float] = None
    n_molecules: Optional[int] = None
    n_callable: Optional[int] = None


@dataclass
class State:
    hit_key: str
    record: RecordRef
    state: str                          # '1' | '0' | '.'
    q: Tuple[int, ...] = ()
    columns: dict = field(default_factory=dict)
    interval: Optional[Tuple[int, int]] = None


@dataclass
class Mark:
    record: RecordRef
    group: str
    chrom: str
    start: int
    end: int
    q: Tuple[int, ...] = ()
    name: str = ''
    mol: Optional[Tuple[int, int]] = None


@dataclass
class MoleculeResult:
    record: RecordRef
    scalars: dict = field(default_factory=dict)
    marks: list = field(default_factory=list)


@dataclass
class EvidenceMember:
    evidence_unit: str
    record: RecordRef


@dataclass
class UnitResult:
    status: str = 'ok'                  # 'ok' | 'empty'
    hits: list = field(default_factory=list)
    values: list = field(default_factory=list)
    states: list = field(default_factory=list)
    native: dict = field(default_factory=dict)
    evidence: list = field(default_factory=list)
    tracks: Optional[dict] = None
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FullMolecule:
    record: RecordRef
    read: Any
    dataset: Dataset


@dataclass
class StitchResult:
    hits: list = field(default_factory=list)
    values: list = field(default_factory=list)
    native: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreDecl:
    name: str
    label: str
    kind: str
    better: str
    per_group: bool

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FeatureTypeDecl:
    type_id: str
    module: str
    label: str
    plural: str
    provides: Tuple[str, ...]
    anchor: str
    zero_width: bool = False
    scores: Tuple[ScoreDecl, ...] = ()
    state_rule: str = ''
    transferable_state: bool = False
    columns: Tuple[str, ...] = ()

    def to_json(self) -> dict:
        out = asdict(self)
        out['provides'] = list(self.provides)
        out['columns'] = list(self.columns)
        out['scores'] = [s.to_json() for s in self.scores]
        return out


@dataclass(frozen=True)
class ScalarCodec:
    name: str
    nbytes: int                         # 1 or 2
    scale: float
    offset: float
    unit: str

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MAGroupDecl:
    group: str
    group_version: int
    kind: str                           # 'state' | 'mark' | 'scalar'
    quality_names: Tuple[str, ...]
    state_codes: Tuple[Tuple[str, str], ...] = ()
    codecs: Tuple[ScalarCodec, ...] = ()
    an: str = 'hit_id'

    @property
    def spec(self) -> str:
        return 'Q' * len(self.quality_names)

    def to_json(self) -> dict:
        return dict(group=self.group, group_version=self.group_version, kind=self.kind,
                    quality_names=list(self.quality_names), state_codes=dict(self.state_codes),
                    codecs=[c.to_json() for c in self.codecs], an=self.an)


# ------------------------------------------------------------------ serialisation of unit results
def _hit_to_json(h: Hit) -> dict:
    return dict(type=h.type, key=h.key, chrom=h.chrom, start=h.start, end=h.end, anchor=h.anchor,
                strand=h.strand, score=h.score, label=h.label, flags=list(h.flags), parent=h.parent,
                columns=h.columns)


def _hit_from_json(v: Mapping[str, Any]) -> Hit:
    return Hit(type=v['type'], key=v['key'], chrom=v['chrom'], start=int(v['start']), end=int(v['end']),
               anchor=int(v['anchor']), strand=v.get('strand', '.'), score=v.get('score'),
               label=v.get('label', ''), flags=tuple(v.get('flags') or ()), parent=v.get('parent'),
               columns=dict(v.get('columns') or {}))


def unit_result_to_json(r: UnitResult) -> dict:
    """A JSON-able unit result (tracks are stored separately by the runner)."""
    return dict(
        status=r.status,
        hits=[_hit_to_json(h) for h in r.hits],
        values=[[v.hit_key, v.group, v.group_kind, v.value_name, v.value, v.lo, v.hi, v.n_molecules, v.n_callable]
                for v in r.values],
        states=[[s.hit_key, s.record.to_json(), s.state, list(s.q), s.columns,
                 list(s.interval) if s.interval is not None else None] for s in r.states],
        native=r.native,
        evidence=[[e.evidence_unit, e.record.to_json()] for e in r.evidence],
        extra=r.extra,
    )


def unit_result_from_json(v: Mapping[str, Any]) -> UnitResult:
    return UnitResult(
        status=v.get('status', 'ok'),
        hits=[_hit_from_json(h) for h in v.get('hits') or ()],
        values=[Value(*row) for row in v.get('values') or ()],
        states=[State(row[0], RecordRef.from_json(row[1]), row[2], tuple(row[3]), dict(row[4] or {}),
                      tuple(row[5]) if row[5] is not None else None) for row in v.get('states') or ()],
        native={k: list(rows) for k, rows in (v.get('native') or {}).items()},
        evidence=[EvidenceMember(row[0], RecordRef.from_json(row[1])) for row in v.get('evidence') or ()],
        extra=dict(v.get('extra') or {}),
    )


def molecule_result_to_json(m: MoleculeResult) -> list:
    return [m.record.to_json(), m.scalars,
            [[k.group, k.chrom, k.start, k.end, list(k.q), k.name, list(k.mol) if k.mol else None] for k in m.marks]]


def molecule_result_from_json(v) -> MoleculeResult:
    rec = RecordRef.from_json(v[0])
    marks = [Mark(rec, g, c, int(s), int(e), tuple(q), n, tuple(mol) if mol else None)
             for g, c, s, e, q, n, mol in v[2]]
    return MoleculeResult(rec, dict(v[1]), marks)
