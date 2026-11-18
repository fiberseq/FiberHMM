"""FiberHMM call bigBeds (``fiberhmm-extract`` output) as an alternative to the BAM.

``fiberhmm-extract`` writes one BED12 row per read and call type
(``{dataset}_tf.bb``, ``_msp.bb``, ``_nucleosome.bb``, ``_m6a.bb``); each block
is a call in reference coordinates. With ``--block-scores`` the rows also carry
per-block qualities in genomic left/right order: ``blockTq``/``blockEl``/
``blockEr`` for TF calls and ``blockMl`` for m6A. Files are queried per region,
never loaded whole, through the same reader as motif bigBeds.

What the format cannot carry, and what follows from it:

* A row spans its first to last block, not the alignment, so a read's span is
  the union of its rows across the given tracks. Eligibility (an MSP spanning
  the motif) is exact; ``n_spanning_fibers`` can miss reads with no call
  reaching across the motif.
* Outer edges are applied on the reference (block edge +- ambiguity), not in
  read coordinates as from the BAM; they differ only when an insertion or
  deletion falls inside the ambiguity zone.
* Without ``--block-scores``: no per-call tq (the LLR floor is the one
  ``fiberhmm-extract --min-tq`` applied, default 50), no edge bytes (outer
  edges unavailable: conservative edges are used), no per-position ML
  (``--require-unmodified`` uses every extracted m6A position, i.e. those
  above ``fiberhmm-extract --prob-threshold``).
* Secondary/supplementary records are not extracted; MAPQ and flag filters
  were applied by ``fiberhmm-extract`` (``-q``), and duplicates are marked in
  ``isDuplicate``.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import numpy as np

from fiberhmm.io.bigbed_motifs import BigBedReader

TRACKS = ('tf', 'msp', 'nucleosome', 'm6a')


def resolve_call_bigbeds(prefix: Optional[str] = None, **explicit) -> Dict[str, str]:
    """Paths per track from ``PREFIX_{tf,msp,nucleosome,m6a}.bb`` and/or explicit paths."""
    paths = {}
    for track in TRACKS:
        if explicit.get(track):
            paths[track] = str(explicit[track])
        elif prefix:
            for name in (f'{prefix}_{track}.bb', f'{prefix}_footprint.bb' if track == 'nucleosome' else None):
                if name and os.path.exists(name):
                    paths[track] = name
                    break
    missing = [t for t in ('tf', 'msp') if t not in paths]
    if missing:
        raise ValueError('call bigBed input needs the tf and msp tracks; missing: ' + ', '.join(missing) +
                         (f' (looked for {prefix}_<type>.bb)' if prefix else ''))
    return paths


def _ints(value) -> List[int]:
    return [int(x) for x in str(value).split(',') if x != ''] if value not in (None, '') else []


class _Track:
    def __init__(self, path, backend=None):
        self.reader = BigBedReader(path, backend=backend)
        self.col = {name: i - 3 for i, name in enumerate(self.reader.fields)}

    def has(self, name):
        return name in self.col

    def rows(self, chrom, start, end):
        """(read, strand, [(block_start, block_end)], {column: value}) for rows overlapping the region."""
        c = self.col
        for _chrom, s, e, rest in self.reader.entries(chrom, start, end):
            name = rest[c['name']] if 'name' in c else rest[0]
            name = name.split('|', 1)[0]                     # circular pieces: read|type|id|part
            sizes = _ints(rest[c.get('blockSizes', 7)])
            starts = _ints(rest[c.get('chromStarts', 8)])
            blocks = [(s + a, s + a + n) for a, n in zip(starts, sizes)]
            yield name, rest[c.get('strand', 2)], s, e, blocks, rest


class CallBigBeds:
    """Molecules for a region, assembled from the per-type call bigBeds."""

    def __init__(self, tf, msp, nucleosome=None, m6a=None, backend=None, quiet=False):
        self.paths = dict(tf=tf, msp=msp, nucleosome=nucleosome, m6a=m6a)
        self.tracks = {k: _Track(v, backend) for k, v in self.paths.items() if v}
        tf_track = self.tracks['tf']
        self.has_tq = tf_track.has('blockTq')
        self.has_edges = tf_track.has('blockEl') and tf_track.has('blockEr')
        self.has_ml = 'm6a' in self.tracks and self.tracks['m6a'].has('blockMl')
        self.warnings = []
        if not self.has_tq:
            self.warnings.append('tf bigBed has no blockTq (extracted without --block-scores): the LLR floor is '
                                 "the one fiberhmm-extract applied (--min-tq, default 50); --min-llr cannot raise it")
        if not self.has_edges:
            self.warnings.append('tf bigBed has no blockEl/blockEr: outer edges are unavailable, conservative edges '
                                 'are used (re-extract with --block-scores)')
        if 'm6a' in self.tracks and not self.has_ml:
            self.warnings.append('m6a bigBed has no blockMl: --require-unmodified uses every extracted position '
                                 '(those above fiberhmm-extract --prob-threshold)')
        if not quiet:
            for w in self.warnings:
                print(f'fiberhmm-footprint: warning: {w}', file=sys.stderr)

    def spec(self) -> dict:
        return dict(self.paths, quiet=True)

    def molecules(self, chrom, start, end, options):
        """Molecules with any row overlapping ``[start, end)``; see the module notes for format limits."""
        from fiberhmm.inference.motif_footprint import Molecule, TFCall, decode_ambiguity
        reads: Dict[str, dict] = {}

        def entry(name, strand):
            r = reads.get(name)
            if r is None:
                r = reads[name] = dict(lo=None, hi=None, strand=strand, calls=[], msps=[], nucs=[], m6a=[], hp='UNK', dup=False)
            return r

        def extend(r, s, e):
            r['lo'] = s if r['lo'] is None else min(r['lo'], s)
            r['hi'] = e if r['hi'] is None else max(r['hi'], e)

        inc = 1 if options.outer_bound == 'inclusive' else 0
        outer = options.edges == 'outer' and self.has_edges
        for kind, track in self.tracks.items():
            c = track.col
            for name, strand, s, e, blocks, rest in track.rows(chrom, start, end):
                r = entry(name, strand)
                extend(r, s, e)
                if 'isDuplicate' in c and rest[c['isDuplicate']] == '1':
                    r['dup'] = True
                if 'hp' in c and r['hp'] == 'UNK':
                    hp = rest[c['hp']]
                    r['hp'] = 'UNK' if hp in ('-1', '') else hp
                if kind == 'tf':
                    tq = _ints(rest[c['blockTq']]) if self.has_tq else [-1] * len(blocks)   # -1 = unknown
                    el = _ints(rest[c['blockEl']]) if self.has_edges else [255] * len(blocks)
                    er = _ints(rest[c['blockEr']]) if self.has_edges else [255] * len(blocks)
                    for (x, y), q, a, b in zip(blocks, tq, el, er):
                        call = TFCall(-1, -1, decode_ambiguity(a), decode_ambiguity(b), q, x, y, x, y)
                        if outer:
                            call.ox, call.oy = x - call.amb_l - inc, y + call.amb_r + inc
                        r['calls'].append(call)
                elif kind == 'msp':
                    r['msps'].extend(blocks)
                elif kind == 'nucleosome':
                    r['nucs'].extend(blocks)
                elif kind == 'm6a':
                    ml = _ints(rest[c['blockMl']]) if self.has_ml else [255] * len(blocks)
                    r['m6a'].extend(bs for (bs, _be), q in zip(blocks, ml) if q >= options.min_ml)
        out = []
        for name, r in reads.items():
            out.append(Molecule(key=(name,), name=name, ref_start=r['lo'], ref_end=r['hi'],
                                is_reverse=r['strand'] == '-', read_length=0, hp=r['hp'],
                                calls=r['calls'], nucs=sorted(set(r['nucs'])), msps=sorted(set(r['msps'])),
                                refmap=None,
                                m6a=np.array(sorted(set(r['m6a'])), dtype=np.int64) if options.require_unmodified
                                else None, duplicate=r['dup']))
        out.sort(key=lambda m: (m.ref_start, m.name))
        return out
