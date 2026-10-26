"""nuc as a ScanModule (NUC P-E): a scan unit and a direct API call give
identical outputs; stitched patches do not depend on the tiling; planted
regular, irregular and phased stretches are found.

Molecules are cut from one planted reference chromatin map, so arrays are
phased across molecules: regular 190 bp in [20, 60) kb, irregular in
[60, 95) kb, regular 172 bp in [95, 130) kb.
"""
from __future__ import annotations

import json

import numpy as np
import pysam
import pytest

from nuc_synth import header, make_record, molecule_sequence, revcomp

from fiberhmm.nuc.calibration import NucCalibration
from fiberhmm.nuc.molecule import molecule_outputs
from fiberhmm.nuc.records import DecodeContext, decode_record
from fiberhmm.nuc.tags import NUCMOL, NUCSEG, NUCSZ, group_rows, read_nuc_groups
from fiberhmm.scan import run_units
from fiberhmm.scan.tables import read_all
from fiberhmm.scan.units import plan

LENGTH = 160_000


def _map(rng):
    dyads, pos = [], 2_000.0
    while pos < LENGTH - 2_000:
        if 20_000 <= pos < 60_000:
            pos += 190 + rng.normal(0, 3)
        elif 95_000 <= pos < 130_000:
            pos += 172 + rng.normal(0, 3)
        else:
            pos += rng.uniform(150, 300)
        dyads.append(pos)
    return np.array(dyads)


@pytest.fixture(scope='module')
def scan_bam(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('nuc_scan')
    rng = np.random.default_rng(21)
    dyads = _map(rng)
    genome = molecule_sequence(LENGTH, rng)
    path = str(tmp / 'chromatin.bam')
    h = header(contig='chrS', length=LENGTH)
    recs = []
    k = 0
    for start in range(0, LENGTH - 9_000, 900):
        L = int(rng.integers(8_000, 12_000))
        L = min(L, LENGTH - start)
        ref_seq = genome[start:start + L]
        dd = dyads[(dyads > start + 80) & (dyads < start + L - 80)] - start
        nucs_ref = [(int(d - 73), int(d + 74)) for d in dd]
        rev = k % 3 == 1
        mol_seq = revcomp(ref_seq) if rev else ref_seq
        nucs = [(L - e, L - s) for s, e in nucs_ref] if rev else nucs_ref
        nucs.sort()
        occ = np.zeros(L, bool)
        for s, e in nucs:
            occ[s:e] = True
        a = np.array([b == 'A' for b in mol_seq])
        marks = np.flatnonzero(a & ~occ & (rng.random(L) < 0.55))
        marks = marks[(marks >= 12) & (marks < L - 12)]
        msps = []
        p = 0
        for s, e in nucs:
            if s > p:
                msps.append((p, s))
            p = e
        if p < L:
            msps.append((p, L))
        recs.append((start, f'm1/{5000 + k}/ccs', mol_seq, nucs, msps, marks, rev, 1 + k % 2))
        k += 1
    with pysam.AlignmentFile(path, 'wb', header=h) as out:
        for start, name, mol_seq, nucs, msps, marks, rev, hp in recs:
            r = make_record(out, name, mol_seq, start, nucs, msps, (), marks, reverse=rev)
            r.set_tag('HP', hp)
            out.write(r)
    pysam.index(path)
    return path


def _scan(out, bam, *, annotate, offset=0, cores=1):
    units = plan({'chrS': LENGTH}, genome=True, contigs=['chrS'], grid_offset=offset)
    return run_units([{'dataset_id': 'synth', 'paths': [bam]}], units, modules=['nuc'],
                     params={'nuc': {'calibration': 'self', 'calibration_population': 400}},
                     out=out, annotate=annotate, cores=cores)


@pytest.fixture(scope='module')
def tsv_scan(scan_bam, tmp_path_factory):
    out = tmp_path_factory.mktemp('nuc_tsv') / 'scan'
    _scan(out, scan_bam, annotate=False)
    return out


def _calibration(scan_dir):
    rec = json.loads((scan_dir / 'nuc' / 'extra' / 'calibration.json').read_text())
    assert rec['schema'] == 'fiberhmm.nuc.scan_calibration.v1'
    return NucCalibration.from_dict(rec['datasets']['synth'])


def _direct(bam, cal):
    out = {}
    with pysam.AlignmentFile(bam) as b:
        ctx = DecodeContext(b.header)
        for r in b.fetch(until_eof=True):
            mol, _ = decode_record(r, ctx)
            if mol is not None:
                out[r.query_name] = (mol, molecule_outputs(mol, cal=cal))
    return out


def test_scan_molecules_and_marks_equal_direct_api(tsv_scan, scan_bam):
    cal = _calibration(tsv_scan)
    direct = _direct(scan_bam, cal)
    mols = read_all(tsv_scan / 'nuc' / 'molecules.tsv.bgz')
    assert {r['qname'] for r in mols} == set(direct)
    for row in mols:
        _mol, res = direct[row['qname']]
        for name, v in res.scalars.items():
            got = row[name]
            if v is None or not np.isfinite(v):
                assert got == '', (name, got)
            else:
                assert float(got) == pytest.approx(v, rel=0, abs=1e-12), name
    marks = read_all(tsv_scan / 'nuc' / 'marks.tsv.bgz')
    by = {}
    for m in marks:
        by.setdefault((m['qname'], m['group']), []).append((int(m['mol_start']), int(m['mol_len']),
                                                            [int(x) for x in m['q'].split(',')], m['name']))
    for qname, (_mol, res) in direct.items():
        rows = group_rows(res)
        for g in (NUCSZ, NUCSEG):
            want = sorted((s, n, list(q), a) for s, n, q, a in rows[g][1])
            assert sorted(by.get((qname, g), [])) == want, (qname, g)


def test_annotated_scan_equals_direct_api(scan_bam, tmp_path):
    out = tmp_path / 'ann'
    _scan(out, scan_bam, annotate=True)
    cal = _calibration(out)
    direct = _direct(scan_bam, cal)
    m = json.loads((out / 'manifest.json').read_text())
    path = out / m['outputs']['annotated'][0]['path']
    seen = 0
    with pysam.AlignmentFile(str(path)) as bam:
        for r in bam.fetch(until_eof=True):
            g = read_nuc_groups(r)
            _mol, res = direct[r.query_name]
            rows = group_rows(res)
            for grp in (NUCSZ, NUCSEG, NUCMOL):
                want = sorted((s, n, list(q)) for s, n, q, _a in rows[grp][1])
                got = sorted((s, n, list(q)) for (s, n), q in zip(g[grp]['intervals'], g[grp]['q']))
                assert got == want, (r.query_name, grp)
            assert not any(v['stale'] for v in g.values())
            seen += 1
    assert seen == len(direct)
    assert m['modules']['nuc']['calibration']['digest']


def test_stitched_patches_do_not_depend_on_the_grid(scan_bam, tsv_scan, tmp_path):
    other = tmp_path / 'off'
    _scan(other, scan_bam, annotate=False, offset=10_000)
    for name in ('hits.tsv.bgz', 'values.tsv.bgz', 'molecules.tsv.bgz', 'marks.tsv.bgz'):
        assert (tsv_scan / 'nuc' / name).read_bytes() == (other / 'nuc' / name).read_bytes(), name


def test_planted_patches_are_found(tsv_scan):
    hits = read_all(tsv_scan / 'nuc' / 'hits.tsv.bgz')
    reg = [h for h in hits if h['type'] == 'nuc.patch.regular']
    phased = [h for h in hits if h['type'] == 'nuc.patch.phased']

    def covered(rows, a, b):
        return sum(max(0, min(int(h['end']), b) - max(int(h['start']), a)) for h in rows) / (b - a)
    assert covered(reg, 24_000, 56_000) > 0.6
    assert covered(reg, 99_000, 126_000) > 0.6
    assert covered(reg, 65_000, 90_000) < 0.1
    q = [float(h['q_hat']) for h in reg if int(h['start']) < 60_000 and h['q_hat']]
    assert q and abs(np.median(q) - 190) <= 6
    q2 = [float(h['q_hat']) for h in reg if int(h['start']) > 95_000 and h['q_hat']]
    assert q2 and abs(np.median(q2) - 172) <= 6
    assert covered(phased, 24_000, 56_000) > 0.4
    values = read_all(tsv_scan / 'nuc' / 'values.tsv.bgz')
    groups = {v['group'] for v in values}
    assert {'all', 'synth', 'synth:HP1', 'synth:HP2'} <= groups


def test_ref_envelope_of_unaligned_interval_is_the_nearest_aligned_base():
    from fiberhmm.nuc.records import make_molecule
    from fiberhmm.scan.modules.nuc import _ref_envelope
    m = make_molecule(12_000, blocks=np.array([[0, 1000, 5000]]), ref_start=1000, ref_end=6000)
    assert _ref_envelope(m, 10_000, 10_147) == (5999, 5999)
    assert _ref_envelope(m, 100, 247) == (1100, 1247)
    r = make_molecule(12_000, blocks=np.array([[7000, 1000, 5000]]), ref_start=1000, ref_end=6000,
                      is_reverse=True)
    assert _ref_envelope(r, 100, 247) == (5753, 5900)          # aligned: molecular start = SEQ end
    assert _ref_envelope(r, 9000, 9147) == (1000, 1000)        # in the clip before the block


def test_thinning_calibrated_scans_are_byte_identical_across_runs_and_cores(scan_bam, tmp_path):
    def run(out, cores):
        units = plan({'chrS': LENGTH}, genome=True, contigs=['chrS'])
        run_units([{'dataset_id': 'synth', 'paths': [scan_bam]}], units, modules=['nuc'],
                  params={'nuc': {'calibration': 'thin', 'calibration_population': 120,
                                  'calibration_seeds': 1, 'calibration_targets': 2}},
                  out=out, annotate=False, cores=cores)
    run(tmp_path / 'one', 1)
    run(tmp_path / 'two', 2)
    for name in ('manifest.json', 'nuc/extra/calibration.json', 'nuc/hits.tsv.bgz', 'nuc/values.tsv.bgz',
                 'nuc/molecules.tsv.bgz', 'nuc/marks.tsv.bgz'):
        assert (tmp_path / 'one' / name).read_bytes() == (tmp_path / 'two' / name).read_bytes(), name
    cal = json.loads((tmp_path / 'one' / 'nuc' / 'extra' / 'calibration.json').read_text())
    assert cal['datasets']['synth']['source'] == 'thin'


def test_molecules_below_the_floor_are_not_measured(tmp_path):
    from nuc_synth import array_molecule, flat_calibration as _flat, planted_record_molecule
    from fiberhmm.scan.modules.nuc import NucModule
    from fiberhmm.scan.reads import resolve_read_policy
    from fiberhmm.scan.types import Dataset, ScanInputs, ScanUnit
    rng = np.random.default_rng(2)
    seq, nucs, msps, marks = planted_record_molecule(rng, n_nuc=20)
    path = str(tmp_path / 'one.bam')
    with pysam.AlignmentFile(path, 'wb', header=header(contig='chrS', length=60_000)) as out:
        out.write(make_record(out, 'm1/1/ccs', seq, 1000, nucs, msps, (), marks))
    pysam.index(path)
    unit = ScanUnit('chrS:0-50000', 'chrS', 0, 50_000, 0, 50_000, 'bed', 0)
    ds = Dataset('d', 0, None, (path,), ('x',))
    inputs = ScanInputs(datasets=(ds,), read_policy=resolve_read_policy(None), reference={'chrS': 60_000},
                        plan=(unit,))
    mod = NucModule()
    settings = mod.resolve({})
    for floor, expect_measured in ((0.9, False), (None, True)):
        cal = {'datasets': {'d': _flat(lambda_floor=floor).to_dict()}}
        res = NucModule().run_unit(unit, inputs, settings, cal, {}, tmp_path, None)
        assert (res.tracks['measured:all'].sum() > 0) == expect_measured


def test_stitch_handles_territories_shorter_than_the_smoothing_window():
    from fiberhmm.scan.modules.nuc import NucModule, _smooth
    m = NucModule()
    for n in (1, 100, 199, 200, 201):
        tracks = {k + ':all': [np.zeros(n)] for k in ('measured', 'reg', 'irr', 'sub', 'qsum')}
        m.stitch('chrS', [(1000, 1000 + n, False, False)], tracks, m.resolve({}), None)
        assert len(_smooth(np.ones(n), 200)) == n
    x = np.zeros(1000)
    x[400:600] = 1.0
    np.testing.assert_allclose(_smooth(x, 200), np.convolve(x, np.ones(200) / 200, mode='same'))
