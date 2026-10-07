"""Deterministic molecule budgets, independently within each dataset and chemical channel."""
from collections import Counter
import hashlib
import heapq

NAMES = ('max_discovery_molecules', 'max_quantification_molecules', 'sample_seed')


def rank(identity, seed):
    """A tile-independent priority; never Python's process-randomized hash."""
    name, tie = identity if isinstance(identity, tuple) else (identity, identity)
    return hashlib.sha256(f'{seed}\0{name}'.encode()).digest(), str(tie)


def identity(unit):
    uid = unit.get('unit_id') or unit.get('uid')
    return unit.get('read_name') or uid, uid


def select(items, cap, seed, channel, identity):
    """Keep lowest-hash N per stratum, preserving input order (including the exact no-op at 0)."""
    if not cap:
        return items
    groups = {}
    for i, item in enumerate(items):
        groups.setdefault(channel(item), []).append(i)
    keep = set()
    for indices in groups.values():
        keep.update(heapq.nsmallest(cap, indices, key=lambda i: rank(identity(items[i]), seed)))
    return [item for i, item in enumerate(items) if i in keep]


def enabled(opt):
    return bool(opt.max_discovery_molecules or opt.max_quantification_molecules)


def record(available, discovery, quantification, opt, channel, **scope):
    counts = [None if us is None else Counter(channel(u) for u in us) for us in (available, discovery, quantification)]
    return dict(**scope, **{n: getattr(opt, n) for n in NAMES},
                denominator='molecules actually used for quantification after eligibility filters',
                strata=[dict(stratum=ch, available=counts[0][ch], discovery=None if counts[1] is None else counts[1][ch],
                             quantification=None if counts[2] is None else counts[2][ch],
                             short={phase: bool(getattr(opt, f'max_{phase}_molecules') and count is not None
                                                and count[ch] < getattr(opt, f'max_{phase}_molecules'))
                                    for phase, count in zip(('discovery', 'quantification'), counts[1:])})
                        for ch in sorted(counts[0])])


def notes(records):
    out = []
    for r in records:
        for s in r['strata']:
            for phase in ('discovery', 'quantification'):
                if (r.get('phase', phase) == phase and r[f'max_{phase}_molecules'] > 0
                        and s['available'] > r[f'max_{phase}_molecules']
                        and s[phase] is not None and s[phase] < s['available']):
                    out.append(f"{phase} used {s[phase]:,} of {s['available']:,} molecules of {s['stratum']} "
                               f"in {r.get('window', 'the region')}")
    return out


def add_arguments(parser):
    for name, default in zip(NAMES, (2000, 0, 1)):
        help = ('Stable molecule hash seed' if name == 'sample_seed' else
                'Molecules per dataset and chemical channel for ' + ('discovery' if 'discovery' in name else 'quantification') + '; 0 = all')
        parser.add_argument('--' + name.replace('_', '-'), type=int, help=f'{help} (default {default})')
