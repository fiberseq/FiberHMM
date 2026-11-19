"""Synthetic inputs and planted-feature modules for testing the scan orchestrator.

Used by FiberHMM's tests and by FiberBrowser's browser = CLI parity tests. The
planted modules report fixed features, so seam reconciliation and stitching
can be checked against known truth without any engine.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .modules import ScanModuleBase
from .types import FeatureTypeDecl, Hit, MAGroupDecl, MoleculeResult, ScalarCodec, State, StitchResult, UnitResult, Value


def write_synthetic_bam(path, contig: str = 'chrS', length: int = 200_000, step: int = 2_000,
                        read_bp: int = 12_000, with_calls: bool = True, chemistry_co: str = 'hia5-pacbio',
                        reverse_every: int = 3, seed: int = 1) -> str:
    """A small coordinate-sorted, indexed BAM: reads every ``step`` bp, with MA nuc/msp calls.

    Read names are synthetic ``m<movie>/<zmw>/ccs``. Calls are molecular-frame
    (the header declares ``fiberhmm:coord=molecular``)."""
    import random
    import pysam
    rng = random.Random(seed)
    enzyme, platform = chemistry_co.split('-') if '-' in chemistry_co else (chemistry_co, 'pacbio')
    header = {'HD': {'VN': '1.6', 'SO': 'coordinate'}, 'SQ': [{'SN': contig, 'LN': length}],
              'CO': ['fiberhmm:coord=molecular',
                     f'FIBERHMM-CHEMISTRY:v1:assay=fiber-seq;enzyme={enzyme};platform={platform};mode=pacbio'],
              'PG': [{'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call', 'VN': '3.0.0', 'DS': 'coord=molecular'}]}
    path = str(path)
    unsorted = path + '.unsorted.bam'
    with pysam.AlignmentFile(unsorted, 'wb', header=header) as out:
        n = 0
        for start in range(0, max(1, length - read_bp), step):
            L = min(read_bp, length - start)
            a = pysam.AlignedSegment()
            a.query_name = f'm1/{1000 + n}/ccs'
            a.query_sequence = ''.join(rng.choice('ACGT') for _ in range(L))
            a.flag = 16 if (reverse_every and n % reverse_every == 2) else 0
            a.reference_id = 0
            a.reference_start = start
            a.mapping_quality = 60
            a.cigarstring = f'{L}M'
            if with_calls:
                nucs, msps = [], []
                pos = rng.randint(0, 80)
                while pos + 147 < L:
                    nucs.append((pos, 147))
                    gap = rng.choice((30, 40, 60, 200, 45, 35))
                    if pos + 147 + gap < L:
                        msps.append((pos + 147, gap))
                    pos += 147 + gap
                parts = [str(L)]
                if nucs:
                    parts.append('nuc.Q:' + ','.join(f'{s + 1}-{l}' for s, l in nucs))
                if msps:
                    parts.append('msp.:' + ','.join(f'{s + 1}-{l}' for s, l in msps))
                a.set_tag('MA', ';'.join(parts), value_type='Z')
                import array
                a.set_tag('AQ', array.array('B', [200] * len(nucs)))
            out.write(a)
            n += 1
    pysam.sort('-o', path, unsorted)
    Path(unsorted).unlink()
    pysam.index(path)
    return path


class PlantedAnchorModule(ScanModuleBase):
    """Reports planted features that lie inside the unit span.

    ``settings['features']``: ``[[key, start, end, mode], ...]`` where mode is
    ``'both'`` (every unit reports it as is), ``'merge_left'`` (the left unit of
    the nearest boundary reports this and the next feature as one merged hit,
    the right unit reports them separately), or ``'child:<parent key>'``."""
    name = 'plant'
    version = '1'
    reconcile = 'anchor'
    native_tables = ('planted',)
    state_groups = {'plant.feature': 'plantst'}

    def parameter_schema(self):
        return {'features': {'default': [], 'type': 'list', 'help': 'planted features'},
                'boundaries': {'default': [], 'type': 'list', 'help': 'grid boundaries used for merge cases'},
                'states': {'default': False, 'type': 'bool',
                           'help': 'state 1 for every record overlapping a planted feature'}}

    def feature_types(self):
        return [FeatureTypeDecl('plant.feature', 'plant', 'Planted feature', 'Planted features', ('hit', 'state'),
                                'centre', columns=('planted',)),
                FeatureTypeDecl('plant.child', 'plant', 'Planted child', 'Planted children', ('hit',), 'centre',
                                columns=('planted',))]

    def ma_groups(self):
        return [MAGroupDecl('plantst', 1, 'state', ('state',), (('1', 'present'),))]

    def run_unit(self, unit, inputs, settings, calibration, upstream, work, progress):
        feats = [tuple(f) for f in settings['features']]
        res = UnitResult()
        inside = [f for f in feats if unit.start <= f[1] and f[2] <= unit.end]
        merged_keys = set()
        for i, (key, s, e, mode) in enumerate(feats):
            if mode != 'merge_left' or (key, s, e, mode) not in inside:
                continue
            nxt = feats[i + 1]
            b = min(settings['boundaries'], key=lambda x: abs(x - (s + nxt[2]) / 2))
            if unit.zone_end == b and nxt in inside:          # the left unit of that boundary merges
                res.hits.append(Hit('plant.feature', f'{key}+{nxt[0]}', unit.chrom, s, nxt[2], (s + nxt[2]) // 2,
                                    columns={'planted': f'{key}+{nxt[0]}'}))
                merged_keys.update({key, nxt[0]})
        for key, s, e, mode in inside:
            if key in merged_keys:
                continue
            if mode.startswith('child:'):
                parent = mode.split(':', 1)[1]
                if any(h.key == parent for h in res.hits) or any(f[0] == parent for f in inside):
                    res.hits.append(Hit('plant.child', key, unit.chrom, s, e, (s + e) // 2, parent=parent,
                                        columns={'planted': key}))
                continue
            res.hits.append(Hit('plant.feature', key, unit.chrom, s, e, (s + e) // 2, columns={'planted': key}))
        # parents must precede children in the hit list only for readability
        res.hits.sort(key=lambda h: (h.parent is not None, h.start, h.key))
        if settings.get('states') and inputs.datasets:
            import pysam
            from .reads import iter_records, record_ref
            roots = [h for h in res.hits if h.type == 'plant.feature']
            for ds in inputs.datasets:
                for i, path in enumerate(ds.paths):
                    with pysam.AlignmentFile(path, 'rb') as bam:
                        for read, sha, occ in iter_records(bam, unit.chrom, unit.start, unit.end,
                                                           inputs.read_policy['filters']):
                            end = read.reference_end or read.reference_start + 1
                            for h in roots:
                                if read.reference_start < max(h.end, h.start + 1) and h.start < end:
                                    res.states.append(State(h.key, record_ref(read, ds, i, sha, occ, unit.chrom), '1'))
        for h in res.hits:
            res.values.append(Value(h.key, 'all', 'all', 'width', float(h.end - h.start)))
            res.native.setdefault('planted', []).append(dict(hit_key=h.key, planted=h.columns['planted'],
                                                             unit=unit.unit_id))
        return res


class PlantedStitchModule(ScanModuleBase):
    """Per-base tracks with planted runs; ``stitch`` thresholds the concatenated signal."""
    name = 'stitchy'
    version = '1'
    reconcile = 'stitch'

    def parameter_schema(self):
        return {'runs': {'default': [], 'type': 'list', 'help': '[start, end) planted runs'},
                'threshold': {'default': 0.5, 'type': 'float', 'help': 'signal threshold'}}

    def feature_types(self):
        return [FeatureTypeDecl('stitchy.run', 'stitchy', 'Run', 'Runs', ('hit',), 'centre', columns=('width',))]

    def run_unit(self, unit, inputs, settings, calibration, upstream, work, progress):
        import numpy as np
        n = unit.zone_end - unit.zone_start
        num = np.zeros(n, dtype=np.int32)
        den = np.ones(n, dtype=np.int32)
        for a, b in settings['runs']:
            lo, hi = max(a, unit.zone_start), min(b, unit.zone_end)
            if hi > lo:
                num[lo - unit.zone_start:hi - unit.zone_start] = 1
        return UnitResult(tracks={'num': num, 'den': den})

    def stitch(self, chrom, territory, tracks, settings, calibration):
        import numpy as np
        out = StitchResult()
        for (start, end, left_open, right_open), num, den in zip(territory, tracks['num'], tracks['den']):
            sig = np.where(den > 0, num / np.maximum(den, 1), 0.0)
            above = np.concatenate([[False], sig >= settings['threshold'], [False]]).astype(np.int8)
            d = np.diff(above)
            for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
                s, e = start + int(a), start + int(b)
                flags = ()
                if (a == 0 and left_open) or (b == len(sig) and right_open):
                    flags = ('open_end',)
                key = f'{chrom}:{s}-{e}'
                out.hits.append(Hit('stitchy.run', key, chrom, s, e, (s + e) // 2, flags=flags,
                                    columns={'width': e - s}))
                out.values.append(Value(key, 'all', 'all', 'width', float(e - s)))
        return out


class PlantedMoleculeModule(ScanModuleBase):
    """Molecule-wide scalars from the record itself (read length and nucleosome count)."""
    name = 'molly'
    version = '1'
    reconcile = 'anchor'
    wants_molecules = True

    def ma_groups(self):
        return [MAGroupDecl('mollymol', 1, 'scalar', ('length_hi', 'length_lo', 'n_nuc'),
                            codecs=(ScalarCodec('length', 2, 1.0, 0.0, 'bp'), ScalarCodec('n_nuc', 1, 1.0, 0.0, 'count')),
                            an='group')]

    def run_unit(self, unit, inputs, settings, calibration, upstream, work, progress):
        return UnitResult()

    def molecule_outputs(self, molecule, settings, calibration):
        from fiberhmm.io.ma_tags import parse_ma_tag
        read = molecule.read
        n = 0
        if read.has_tag('MA'):
            n = len(parse_ma_tag(str(read.get_tag('MA')))['nuc'])
        return MoleculeResult(molecule.record, scalars={'length': float(read.query_length), 'n_nuc': float(n)})


def write_daf_locus_bam(path, seed: int = 3, reads: int = 40, contig: str = 'chr1', length: int = 6000,
                        nfr=(2000, 2400), tf_at: int = 2150) -> str:
    """A DddA locus both fp and nfr find features in: nucleosome arrays, an NFR open in about 60% of
    molecules holding a TF footprint, C -> Y deamination outside protection (st CT), MA calls."""
    import random
    import pysam
    from fiberhmm.io.bam_header import append_chemistry
    rng = random.Random(seed)
    ref = ''.join(rng.choice('ACGT') for _ in range(length))
    header = pysam.AlignmentHeader.from_dict(dict(HD={'VN': '1.6', 'SO': 'coordinate'},
                                                  SQ=[{'SN': contig, 'LN': length}],
                                                  CO=['fiberhmm:coord=molecular']))
    header = append_chemistry(header, dict(assay='daf', enzyme='ddda', platform='pacbio', mode='daf'))
    recs = []
    for i in range(reads):
        start = 500 + rng.randint(0, 40)
        n = 4000
        prot = []
        p = 30 + rng.randint(0, 20)
        nfr_open = rng.random() < 0.6
        while p + 147 < n:
            a = start + p
            if nfr[0] <= a <= nfr[1] and nfr_open:
                p = nfr[1] - start + rng.randint(0, 10)
                continue
            prot.append((p, 147))
            p += 147 + rng.choice((25, 35, 45, 30))
        tf = (tf_at - start + rng.randint(-2, 2), 40) if (nfr_open and rng.random() < 0.7) else None
        seq = []
        for q in range(n):
            b = ref[start + q]
            covered = any(s0 <= q < s0 + l for s0, l in prot) or (tf and tf[0] <= q < tf[0] + tf[1])
            if b == 'C' and not covered and rng.random() < .85:
                b = 'Y'
            seq.append(b)
        r = pysam.AlignedSegment(header)
        r.query_name = f'm9/{2000 + i}/ccs'
        r.query_sequence = ''.join(seq)
        r.reference_id = 0
        r.reference_start = start
        r.mapping_quality = 60
        r.cigarstring = f'{n}M'
        r.set_tag('st', 'CT')
        msp, prev = [], 0
        for s0, l in prot:
            if s0 > prev:
                msp.append((prev, s0 - prev))
            prev = s0 + l
        parts = [str(n), 'nuc.:' + ','.join(f'{s0 + 1}-{l}' for s0, l in prot)]
        if msp:
            parts.append('msp.:' + ','.join(f'{s0 + 1}-{l}' for s0, l in msp))
        if tf:
            parts.append(f'tf.:{tf[0] + 1}-{tf[1]}')
        r.set_tag('MA', ';'.join(parts))
        recs.append(r)
    recs.sort(key=lambda r: r.reference_start)
    with pysam.AlignmentFile(str(path), 'wb', header=header) as h:
        for r in recs:
            h.write(r)
    pysam.index(str(path))
    return str(path)


def write_peaks_fixture_bam(path, chemistry: str = 'hia5-pacbio', seed: Optional[int] = None, contig: str = 'chrP',
                            length: int = 30_000, molecules: int = 140) -> str:
    """A synthetic BAM that exercises FiberBrowser's peak-scan loader paths.

    Nucleosome arrays with accessible sites at planted positions, TF-sized
    calls, reverse reads (molecular-frame ns/nl and MA), soft clips, small
    insertions and deletions, short read-end calls, long terminal gaps,
    hard- and soft-clipped supplementaries (primary in or out of view),
    duplicates, secondary records, reads with only as/al and reads with only
    MA. ``chemistry``: hia5-pacbio, hia5-nanopore or ddda."""
    import array
    import random
    import pysam
    if seed is None:
        seed = {'hia5-pacbio': 7, 'ddda': 8, 'hia5-nanopore': 9}[chemistry]
    rng = random.Random(seed)
    enzyme, platform = {'hia5-pacbio': ('hia5', 'pacbio'), 'hia5-nanopore': ('hia5', 'nanopore'),
                        'ddda': ('ddda', 'pacbio')}[chemistry]
    assay = 'daf' if enzyme == 'ddda' else 'fiber-seq'
    header = {'HD': {'VN': '1.6', 'SO': 'coordinate'}, 'SQ': [{'SN': contig, 'LN': length}],
              'PG': [{'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call', 'VN': '3.0.2', 'DS': 'coord=molecular'}],
              'CO': [f'FIBERHMM-CHEMISTRY:v1:assay={assay};enzyme={enzyme};platform={platform};mode={assay}']}
    sites = [(6000, 6350), (14000, 14200), (21500, 22000), (26000, 26150)]
    ref = ''.join(rng.choice('ACGT') for _ in range(length))
    records = []

    def calls_for(mol_len, ref_start, reverse):
        """Molecular-frame footprints (start, length) and msps over a molecule of mol_len."""
        fps, pos = [], rng.choice((5, 12, 18, 40, 400, 700))
        if rng.random() < 0.15:                                  # short read-end call
            fps.append((rng.randint(8, 20), rng.randint(30, 60)))
            pos = fps[-1][0] + fps[-1][1] + rng.randint(20, 60)
        while pos + 147 < mol_len - rng.choice((10, 20, 600)):
            ref_pos = ref_start + (mol_len - pos if reverse else pos)
            in_site = any(a - 40 <= ref_pos <= b for a, b in sites) and rng.random() < 0.7
            if in_site:
                if rng.random() < 0.3:
                    fps.append((pos, rng.randint(25, 60)))       # a TF-sized call in the site
                pos += rng.randint(200, 420)
                continue
            n = rng.choice((140, 146, 147, 150, 120, 95))
            fps.append((pos, n))
            pos += n + rng.choice((15, 25, 35, 45, 60))
        fps = [(s, n) for s, n in fps if s + n <= mol_len]
        msps, prev = [], 0
        for s, n in sorted(fps):
            if s > prev:
                msps.append((prev, s - prev))
            prev = max(prev, s + n)
        if prev < mol_len:
            msps.append((prev, mol_len - prev))
        return sorted(fps), msps

    def cigar_for(mol_len, softl, softr):
        ops, aligned = [], mol_len - softl - softr
        if softl:
            ops.append((4, softl))
        done = 0
        while done < aligned:
            step = min(aligned - done, rng.randint(300, 2000))
            ops.append((0, step))
            done += step
            if done < aligned and rng.random() < 0.5:
                r = rng.random()
                if r < 0.4:
                    k = min(rng.randint(1, 5), aligned - done)
                    if k > 0:
                        ops.append((1, k))
                        done += k
                elif r < 0.8:
                    ops.append((2, rng.randint(1, 5)))
                elif r < 0.9:
                    ops.append((2, rng.randint(50, 200)))       # a deletion the browser cuts calls at
                else:                                            # indel pieces joined across a short bridge
                    ops.append((2, rng.randint(12, 40)))
                    ops.append((0, 4))
                    done += 4
                    ops.append((2, rng.randint(12, 40)))
        if softr:
            ops.append((4, softr))
        return ops

    def ref_len(ops):
        return sum(n for op, n in ops if op in (0, 2, 3, 7, 8))

    n_mol = 0
    for i in range(molecules):
        mol_len = rng.randint(2500, 8000)
        start = rng.randint(0, length - mol_len - 50)
        reverse = rng.random() < 0.5
        softl = rng.choice((0, 0, 0, 60, 150))
        softr = rng.choice((0, 0, 0, 80))
        ops = cigar_for(mol_len, softl, softr)
        if start + ref_len(ops) >= length:
            continue
        fps, msps = calls_for(mol_len, start, reverse)
        name = f'p{seed}/{100 + i}/ccs'
        kind = rng.random()
        records.append(dict(name=name, flag=16 if reverse else 0, start=start, ops=ops, mol_len=mol_len, fps=fps,
                            msps=msps, kind=kind))
        n_mol += 1
        if kind < 0.05:                                          # a duplicate copy (flag 0x400)
            records.append(dict(records[-1], flag=records[-1]['flag'] | 0x400, kind=1.0))
        elif kind < 0.08:                                        # a secondary alignment
            records.append(dict(records[-1], flag=records[-1]['flag'] | 0x100, kind=1.0,
                                start=min(length - mol_len - 1, start + 777)))
    # supplementaries: hard-clipped and soft-clipped, with the primary in view or not
    for j in range(10):
        seg = rng.randint(800, 2000)
        clip = rng.randint(500, 3000)
        start = rng.randint(0, length - seg - 10)
        hard = j % 2 == 0
        fps, msps = calls_for(seg + (0 if hard else clip), start, False)
        ops = ([(5, clip)] if hard else [(4, clip)]) + [(0, seg)]
        name = records[j * 7]['name'] if j % 3 == 0 else f'p{seed}/{900 + j}/ccs'
        records.append(dict(name=name, flag=0x800, start=start, ops=ops, mol_len=seg + (0 if hard else clip),
                            fps=fps, msps=msps, kind=0.5))
    records.sort(key=lambda r: (r['start'], r['name'], r['flag']))
    path = str(path)
    with pysam.AlignmentFile(path, 'wb', header=header) as out:
        for r in records:
            a = pysam.AlignedSegment(out.header)
            a.query_name = r['name']
            a.flag = r['flag']
            a.reference_id = 0
            a.reference_start = r['start']
            a.mapping_quality = 60 if not r['flag'] & 0x100 else 0
            a.cigartuples = r['ops']
            q_len = sum(n for op, n in r['ops'] if op in (0, 1, 4, 7, 8))
            a.query_sequence = ''.join(rng.choice('ACGT') for _ in range(q_len))
            fps, msps, L = r['fps'], r['msps'], q_len
            fps = [(s, n) for s, n in fps if s + n <= L]
            msps = [(s, n) for s, n in msps if s + n <= L]
            k = r['kind']
            if enzyme == 'ddda':
                a.set_tag('st', 'GA' if a.is_reverse else 'CT')
            if k < 0.85 or k == 1.0 or k == 0.5:
                a.set_tag('ns', array.array('I', [s for s, _n in fps]))
                a.set_tag('nl', array.array('I', [n for _s, n in fps]))
                a.set_tag('nq', array.array('B', [rng.randint(10, 250) for _ in fps]))
            if k < 0.85 or k == 1.0 or 0.92 <= k:
                if msps:
                    a.set_tag('as', array.array('I', [s for s, _n in msps]))
                    a.set_tag('al', array.array('I', [n for _s, n in msps]))
            if k < 0.85 or k == 1.0 or 0.85 <= k < 0.92 or k == 0.5:
                nucs = [(s, n) for s, n in fps if n >= 90]
                tfs = [(s, n) for s, n in fps if n < 90]
                parts = [str(L)]
                if nucs:
                    parts.append('nuc.Q:' + ','.join(f'{s + 1}-{n}' for s, n in nucs))
                if msps:
                    parts.append('msp.:' + ','.join(f'{s + 1}-{n}' for s, n in msps))
                if tfs:
                    parts.append('tf.QQQ:' + ','.join(f'{s + 1}-{n}' for s, n in tfs))
                if len(parts) > 1:
                    a.set_tag('MA', ';'.join(parts))
                    a.set_tag('AQ', array.array('B', [200] * len(nucs) + [100, 200, 200] * len(tfs)))
            out.write(a)
    pysam.index(path)
    return path
