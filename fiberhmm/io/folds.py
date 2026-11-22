"""Deterministic read folds for cross-fitting (shared by fiberhmm-nfr and fiberhmm-consensus).

A read's fold is a hash of its read name modulo K::

    fold = int(hashlib.<hash>(read_name.encode('utf-8')).hexdigest(), 16) % mod

so the same read lands in the same fold in every tool, run, window and dataset, independent of BAM order or of
which other reads are present. With the defaults (``md5``, ``mod=10``) this is ``md5(read_name) % 10`` read as one
big-endian integer. A :class:`FoldSpec` names the folds a tool *discovers* on (defines variants or classes from);
the remaining folds are held out, and tools still assign membership to every read.

Spec strings (``--discover-fold-spec``)::

    hash=md5,mod=10,discover=0-4        # folds 0..4 discover, 5..9 are held out
    mod=5,discover=0+2+4                # several folds: '+' between items, ranges with '-'
    hash=sha1,mod=2,discover=1

Keys may come in any order; ``hash`` defaults to md5 and ``mod`` to 10. ``discover`` is required and must name at
least one fold in ``0..mod-1``. :meth:`FoldSpec.canonical` gives the normalised string that tools record.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

HASHES = ('md5', 'sha1', 'sha256')
DEFAULT_HASH = 'md5'
DEFAULT_MOD = 10


def fold_of(name, mod=DEFAULT_MOD, hash=DEFAULT_HASH):
    """The fold (0..mod-1) of one read name."""
    if hash not in HASHES:
        raise ValueError(f'fold hash must be one of {", ".join(HASHES)}')
    if int(mod) < 1:
        raise ValueError('fold mod must be >= 1')
    return int(hashlib.new(hash, str(name).encode('utf-8')).hexdigest(), 16) % int(mod)


def _folds(text, mod):
    out = set()
    for item in str(text).replace(' ', '').split('+'):
        if not item:
            continue
        a, dash, b = item.partition('-')
        try:
            lo, hi = (int(a), int(b)) if dash else (int(a), int(a))
        except ValueError:
            raise ValueError(f'discover={text!r}: expected folds like 0-4 or 0+2+4') from None
        if lo > hi or lo < 0 or hi >= mod:
            raise ValueError(f'discover={text!r}: folds must lie in 0..{mod - 1}')
        out.update(range(lo, hi + 1))
    if not out:
        raise ValueError('discover: name at least one fold')
    return tuple(sorted(out))


@dataclass(frozen=True)
class FoldSpec:
    """Which hash folds discover (define) and which are held out."""
    discover: tuple
    mod: int = DEFAULT_MOD
    hash: str = DEFAULT_HASH

    def __post_init__(self):
        if self.hash not in HASHES:
            raise ValueError(f'fold hash must be one of {", ".join(HASHES)}')
        if isinstance(self.mod, bool) or not isinstance(self.mod, int) or self.mod < 1:
            raise ValueError('fold mod must be an integer >= 1')
        folds = tuple(sorted({int(f) for f in self.discover}))
        if not folds or folds[0] < 0 or folds[-1] >= self.mod:
            raise ValueError(f'discover folds must be a non-empty subset of 0..{self.mod - 1}')
        object.__setattr__(self, 'discover', folds)

    @classmethod
    def parse(cls, value):
        """A spec string, a dict {hash, mod, discover}, a FoldSpec, or None/'' (no cross-fitting: returns None)."""
        if value is None or value == '':
            return None
        if isinstance(value, FoldSpec):
            return value
        if isinstance(value, dict):
            unknown = sorted(set(value) - {'hash', 'mod', 'discover'})
            if unknown:
                raise ValueError(f'fold spec: unknown key(s) {", ".join(unknown)}')
            mod = int(value.get('mod', DEFAULT_MOD))
            disc = value.get('discover')
            if isinstance(disc, (list, tuple)):
                disc = '+'.join(str(int(x)) for x in disc)
            if disc is None:
                raise ValueError('fold spec: discover is required')
            return cls(discover=_folds(disc, mod), mod=mod, hash=str(value.get('hash', DEFAULT_HASH)))
        fields = {}
        for part in str(value).split(','):
            part = part.strip()
            if not part:
                continue
            key, eq, val = part.partition('=')
            key = key.strip().lower()
            if not eq or key not in ('hash', 'mod', 'discover'):
                raise ValueError(f'fold spec {value!r}: expected hash=md5,mod=10,discover=0-4')
            if key in fields:
                raise ValueError(f'fold spec {value!r}: {key} given twice')
            fields[key] = val.strip()
        if 'discover' not in fields:
            raise ValueError(f'fold spec {value!r}: discover=FOLDS is required (e.g. discover=0-4)')
        try:
            mod = int(fields.get('mod', DEFAULT_MOD))
        except ValueError:
            raise ValueError(f'fold spec {value!r}: mod must be an integer') from None
        if mod < 1:
            raise ValueError('fold mod must be >= 1')
        return cls(discover=_folds(fields['discover'], mod), mod=mod, hash=fields.get('hash', DEFAULT_HASH).lower())

    def fold(self, name):
        return fold_of(name, self.mod, self.hash)

    def discovers(self, name):
        """True when the read is in a discovery fold."""
        return self.fold(name) in self.discover

    @property
    def all_folds(self):
        """True when every fold discovers (no read is held out)."""
        return len(self.discover) == self.mod

    def canonical(self):
        """Normalised spec string: hash, mod, then discover as sorted ranges."""
        runs, folds = [], list(self.discover)
        start = prev = folds[0]
        for f in folds[1:] + [None]:
            if f is not None and f == prev + 1:
                prev = f
                continue
            runs.append(f'{start}-{prev}' if prev > start else f'{start}')
            if f is not None:
                start = prev = f
        return f'hash={self.hash},mod={self.mod},discover={"+".join(runs)}'

    def as_dict(self):
        return dict(hash=self.hash, mod=self.mod, discover=list(self.discover))
