"""Synthetic molecules, records and calibrations for the fiberhmm.nuc tests."""
from __future__ import annotations

import array

import numpy as np
import pysam

from fiberhmm.nuc.calibration import NucCalibration
from fiberhmm.nuc.density import BetaPrior
from fiberhmm.nuc.records import make_molecule
from fiberhmm.nuc.sizes import ReferenceCurve

COMP = str.maketrans('ACGT', 'TGCA')


def revcomp(s: str) -> str:
    return s.translate(COMP)[::-1]


def complement_msps(length, footprints):
    iv = sorted((int(s), int(e)) for s, e in footprints)
    out, p = [], 0
    for s, e in iv:
        if s > p:
            out.append((p, s))
        p = max(p, e)
    if p < length:
        out.append((p, length))
    return out


def array_molecule(blocks, rng, start=200, core=147, mark_rate=0.5, tail=300, **kw):
    """Molecule from blocks: ('reg', n, gap) regular, ('irr', n, _) gaps
    U(150, 290), ('sub', n, step) alternating 70 bp short calls and 30 bp TF
    calls. Marks fall in linkers at ``mark_rate``; every base informative."""
    nucs, tfs, pos = [], [], start
    for kind, n, arg in blocks:
        for k in range(n):
            if kind == 'reg':
                nucs.append((int(pos), int(pos + core)))
                pos += arg + rng.normal(0, 4)
            elif kind == 'irr':
                nucs.append((int(pos), int(pos + core)))
                pos += rng.uniform(150, 290)
            elif kind == 'sub':
                if k % 2 == 0:
                    nucs.append((int(pos), int(pos + 70)))
                else:
                    tfs.append((int(pos), int(pos + 30)))
                pos += arg
    L = int(pos + tail)
    occ = np.zeros(L, bool)
    for s, e in nucs + tfs:
        occ[s:e] = True
    hits = np.flatnonzero(~occ & (rng.random(L) < mark_rate))
    hits = hits[(hits > 5) & (hits < L - 5)]
    return make_molecule(L, nucs=nucs, tfs=tfs, msps=complement_msps(L, nucs + tfs), hits=hits, **kw)


def flat_calibration(nrl=190.0, mu=147.0, sigma=15.0, sigma_gap=12.0, **kw):
    return NucCalibration(source='self', nrl_pooled=nrl, prior=BetaPrior(5, 5),
                          curve=ReferenceCurve([np.log(0.5)], [mu], [sigma]), sigma_gap=sigma_gap, **kw)


def header(contig='chrS', length=200_000, coord_molecular=True, extra_co=()):
    pg = {'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call', 'VN': '3.0.2',
          'CL': 'fiberhmm-call -i in.bam -o out.bam --enzyme hia5 --seq pacbio'}
    if coord_molecular:
        pg['DS'] = 'FiberHMM fused apply+recall; coord=molecular; mode=pacbio-fiber'
    return {'HD': {'VN': '1.6', 'SO': 'coordinate'}, 'SQ': [{'SN': contig, 'LN': length}],
            'PG': [pg], 'CO': list(extra_co)}


def molecule_sequence(length, rng):
    return ''.join(rng.choice(list('ACGT'), size=length))


def mm_for_marks(mol_seq: str, marks) -> tuple:
    """MM/ML for m6A at molecular A positions ``marks`` (original orientation)."""
    a_pos = [i for i, b in enumerate(mol_seq) if b == 'A']
    marks = set(int(m) for m in marks)
    skips, mls, run = [], [], 0
    for p in a_pos:
        if p in marks:
            skips.append(run)
            mls.append(250)
            run = 0
        else:
            run += 1
    return 'A+a.' + ''.join(f',{s}' for s in skips) + ';', array.array('B', mls)


def make_record(out, name, mol_seq, ref_start, nucs=(), msps=(), tfs=(), marks=(),
                reverse=False, extra_groups=(), flag_extra=0, cigar=None):
    """pysam record of molecule ``mol_seq`` (original orientation) with MA calls
    in molecular coordinates and m6A marks at molecular A positions."""
    L = len(mol_seq)
    r = pysam.AlignedSegment(out.header)
    r.query_name = name
    r.flag = (16 if reverse else 0) | flag_extra
    r.reference_id = 0
    r.reference_start = ref_start
    r.mapping_quality = 60
    r.cigarstring = cigar or f'{L}M'
    r.query_sequence = revcomp(mol_seq) if reverse else mol_seq
    r.query_qualities = array.array('B', [30] * L)
    groups = []
    aq = []
    if nucs:
        groups.append('nuc.QQQ:' + ','.join(f'{s + 1}-{e - s}' for s, e in sorted(nucs)))
        aq += [200, 255, 255] * len(nucs)
    if msps:
        groups.append('msp.:' + ','.join(f'{s + 1}-{e - s}' for s, e in sorted(msps)))
    if tfs:
        groups.append('tf.QQQ:' + ','.join(f'{s + 1}-{e - s}' for s, e in sorted(tfs)))
        aq += [100, 255, 255] * len(tfs)
    for g, q in extra_groups:
        groups.append(g)
        aq += list(q)
    r.set_tag('MA', ';'.join([str(L)] + groups))
    if aq:
        r.set_tag('AQ', array.array('B', aq))
    mm, ml = mm_for_marks(mol_seq, marks)
    r.set_tag('MM', mm)
    r.set_tag('ML', ml)
    return r


def planted_record_molecule(rng, n_nuc=12, gap=190, core=147, start=300, mark_rate=0.6):
    """(mol_seq, nucs, msps, marks) for a regular array; marks on A bases in linkers."""
    pos = start
    nucs = []
    for _ in range(n_nuc):
        nucs.append((pos, pos + core))
        pos += gap
    L = pos + 400
    seq = molecule_sequence(L, rng)
    occ = np.zeros(L, bool)
    for s, e in nucs:
        occ[s:e] = True
    a = np.array([b == 'A' for b in seq])
    marks = np.flatnonzero(a & ~occ & (rng.random(L) < mark_rate))
    marks = marks[(marks >= 12) & (marks < L - 12)]
    return seq, nucs, complement_msps(L, nucs), marks
