"""nuc as a ScanModule (NUC P-E): a scan unit and a direct API call give
identical outputs; stitched patches do not depend on the tiling; planted
recurrent arrays and phased stretches are found.

Molecules (8-12 kb) are cut from one planted reference chromatin map, so
arrays are phased across molecules and recur at the same place: regular
190 bp in ARRAYS_190, regular 172 bp in ARRAYS_172 (each shorter than a
read, as in real data: a read regular end to end carries no positional
evidence under the shift-within-read null), irregular gaps elsewhere.
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
ARRAYS_190 = ((20_000, 27_000), (38_000, 45_000), (54_000, 60_000))
ARRAYS_172 = ((95_000, 102_000), (116_000, 122_000))


def _map(rng):
    dyads, pos = [], 2_000.0
    while pos < LENGTH - 2_000:
        if any(a <= pos < b for a, b in ARRAYS_190):
            pos += 190 + rng.normal(0, 3)
        elif any(a <= pos < b for a, b in ARRAYS_172):
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
    for a, b in ARRAYS_190 + ARRAYS_172:
        assert covered(reg, a + 500, b - 500) > 0.6, (a, b)
    assert covered(reg, 65_000, 90_000) < 0.1
    q = [float(h['q_hat']) for h in reg if int(h['start']) < 61_000 and h['q_hat']]
    assert q and abs(np.median(q) - 190) <= 6
    q2 = [float(h['q_hat']) for h in reg if int(h['start']) > 94_000 and h['q_hat']]
    assert q2 and abs(np.median(q2) - 172) <= 6
    assert np.mean([covered(phased, a, b) for a, b in ARRAYS_190]) > 0.4
    # recurrence patches carry the test's numbers
    values = read_all(tsv_scan / 'nuc' / 'values.tsv.bgz')
    pmin = [float(v['value']) for v in values if v['value_name'] == 'p_min']
    assert pmin and max(pmin) <= 0.1             # every patch holds a BH rejection at FDR 0.1
    assert not [h for h in hits if h['type'] in ('nuc.patch.irregular', 'nuc.patch.sub_rich')]
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


def _stitch(m, chrom, territory, tracks, settings):
    """stitch() as the scan finalizer calls it: with the scan-wide statistics when the FDR scope is the scan."""
    extra = {}
    if m.scan_statistics_tracks(settings) is not None:
        extra['scan_statistics'] = m.combine_statistics(
            {chrom: m.stitch_statistics(chrom, territory, tracks, settings, None)}, settings, None)
    return m.stitch(chrom, territory, tracks, settings, None, **extra)


def test_stitch_handles_territories_shorter_than_a_window():
    from fiberhmm.scan.modules.nuc import ENRICH_KINDS, SEG_METRICS, NucModule, _smooth
    m = NucModule()
    for n in (1, 100, 249, 250, 499, 500, 501):
        tracks = {k + ':all': [np.ones(n)] for k in ('measured',) + SEG_METRICS + ('qsum',)}
        for kind in ENRICH_KINDS:
            p = np.full(n, np.nan)
            p[n - 1] = 1e-9                     # a significant window starting at the last base
            tracks[f'enr_p:{kind}'] = [p]
            for name, v in (('obs', 0.5), ('exp', 0.2), ('n', 5.0)):
                tracks[f'enr_{name}:{kind}'] = [np.where(np.isfinite(p), v, np.nan)]
        for scope in ('contig', 'scan'):
            res = _stitch(m, 'chrS', [(1000, 1000 + n, False, False)], tracks, m.resolve({'enrich_fdr_scope': scope}))
            pats = [h for h in res.hits if h.type.startswith('nuc.patch.')]
            assert pats and all(1000 <= h.start < h.end <= 1000 + n for h in pats)
        assert len(_smooth(np.ones(n), 200)) == n
    x = np.zeros(1000)
    x[400:600] = 1.0
    np.testing.assert_allclose(_smooth(x, 200), np.convolve(x, np.ones(200) / 200, mode='same'))


def test_calibration_sample_does_not_depend_on_the_grid():
    from fiberhmm.scan.modules.nuc import calibration_regions
    a, _ = plan({'chrL': 2_500_000}, genome=True, contigs=['chrL'], grid_offset=0)
    b, _ = plan({'chrL': 2_500_000}, genome=True, contigs=['chrL'], grid_offset=10_000)
    assert len(a) != len(b) or a[0].unit_id != b[0].unit_id
    assert calibration_regions(a, 40) == calibration_regions(b, 40)
    assert len(calibration_regions(a, 40)) == 40
    locus = (1_210_000, 1_390_000)

    def probe(chrom, s, e):
        return s < locus[1] and e > locus[0]
    got = calibration_regions(a, 40, has_records=probe)
    assert got == calibration_regions(b, 40, has_records=probe)
    assert got == [('chrL', 1_200_000, 1_250_000), ('chrL', 1_250_000, 1_300_000),
                   ('chrL', 1_300_000, 1_350_000), ('chrL', 1_350_000, 1_400_000)]


def test_genome_scan_of_targeted_data_calibrates_on_the_target(scan_bam, tmp_path):
    """A genome scan of a BAM covering one locus samples the windows that hold
    records, not the first hashed windows of the contig."""
    import hashlib
    contig_len = 10_000_000
    hashed = sorted((hashlib.sha256(f'chrT:{k}'.encode()).hexdigest(), k) for k in range(contig_len // 50_000))
    first = {k for _h, k in hashed[:40]}
    k0 = next(k for k in range(contig_len // 50_000 - 5) if not first & set(range(k, k + 5)))
    shift = k0 * 50_000
    path = str(tmp_path / 'targeted.bam')
    with pysam.AlignmentFile(scan_bam) as src:
        hd = src.header.to_dict()
        hd['SQ'] = [{'SN': 'chrT', 'LN': contig_len}]
        new = pysam.AlignmentHeader.from_dict(hd)
        with pysam.AlignmentFile(path, 'wb', header=new) as out:
            for r in src.fetch(until_eof=True):
                d = r.to_dict()
                d['ref_name'] = 'chrT'
                d['ref_pos'] = str(int(d['ref_pos']) + shift)
                out.write(pysam.AlignedSegment.from_dict(d, new))
    pysam.index(path)
    units = plan({'chrT': contig_len}, genome=True, contigs=['chrT'])
    out = tmp_path / 'scan'
    run_units([{'dataset_id': 'synth', 'paths': [path]}], units, modules=['nuc'],
              params={'nuc': {'calibration': 'self', 'calibration_population': 400}},
              out=out, annotate=False, cores=2)
    rec = json.loads((out / 'nuc' / 'extra' / 'calibration.json').read_text())
    regions = rec['regions']['synth']
    assert regions and all(shift <= int(r.split(':')[1].split('-')[0]) < shift + LENGTH for r in regions)
    assert rec['datasets']['synth']['population']
    assert read_all(out / 'nuc' / 'molecules.tsv.bgz')


def test_cli_scan_unit_equals_direct_api(scan_bam, tmp_path):
    """fiberhmm-scan --modules nuc (the CLI) against molecule_outputs (the API)."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    params = tmp_path / 'params.json'
    params.write_text(json.dumps({'nuc': {'calibration': 'self', 'calibration_population': 400}}))
    out = tmp_path / 'cli_scan'
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(repo) + os.pathsep + os.environ.get('PYTHONPATH', ''),
               FIBERHMM_NO_UPDATE_CHECK='1')
    res = subprocess.run([sys.executable, '-m', 'fiberhmm.scan.cli', '--bam', scan_bam, '--genome',
                          '--modules', 'nuc', '--parameters', str(params), '--no-annotate', '--cores', '1',
                          '-o', str(out)], capture_output=True, text=True, env=env, cwd=repo, timeout=900)
    assert res.returncode == 0, res.stderr[-2000:]
    rec = json.loads((out / 'nuc' / 'extra' / 'calibration.json').read_text())
    (cal_d,) = rec['datasets'].values()          # the CLI names the dataset by --bam position
    cal = NucCalibration.from_dict(cal_d)
    direct = _direct(scan_bam, cal)
    rows = read_all(out / 'nuc' / 'molecules.tsv.bgz')
    assert {r['qname'] for r in rows} == set(direct)
    for row in rows:
        _mol, r = direct[row['qname']]
        for name, v in r.scalars.items():
            if v is None or not np.isfinite(v):
                assert row[name] == ''
            else:
                assert float(row[name]) == pytest.approx(v, rel=0, abs=1e-12), name
    marks = read_all(out / 'nuc' / 'marks.tsv.bgz')
    got = sorted((m['qname'], m['group'], int(m['mol_start']), int(m['mol_len']), m['q']) for m in marks)
    want = sorted((q, g, s, n, ','.join(str(int(x)) for x in qb))
                  for q, (_m, r) in direct.items() for g, (_spec, items) in group_rows(r).items()
                  if g != NUCMOL for s, n, qb, _a in items)
    assert got == want


def test_record_probe_skips_filtered_and_uncalled_reads(scan_bam, tmp_path):
    """The calibration probe looks past any number of filtered records and
    needs an annotated (called) primary record."""
    from fiberhmm.scan.modules.nuc import _RecordProbe
    from fiberhmm.scan.reads import passes, resolve_read_policy
    filters = resolve_read_policy(None)['filters']
    path = str(tmp_path / 'probe.bam')
    with pysam.AlignmentFile(scan_bam) as src:
        hdr = src.header
        recs = list(src.fetch(until_eof=True))
    with pysam.AlignmentFile(path, 'wb', header=hdr) as out:
        first = recs[0]
        for k in range(300):                     # duplicates ahead of the called record
            d = pysam.AlignedSegment.fromstring(first.to_string(), hdr)
            d.query_name = f'dup{k}'
            d.is_duplicate = True
            out.write(d)
        out.write(first)
        bare = pysam.AlignedSegment.fromstring(recs[-1].to_string(), hdr)
        for t in ('MA', 'AQ', 'AN', 'Ma', 'Aq', 'An', 'ns', 'nl', 'as', 'al'):
            if bare.has_tag(t):
                bare.set_tag(t, None)
        out.write(bare)                          # an uncalled read elsewhere
    pysam.index(path)
    with _RecordProbe([path], lambda r: passes(r, filters)) as probe:
        assert probe('chrS', first.reference_start, first.reference_start + 50)
        assert not probe('chrS', bare.reference_end - 10, bare.reference_end)


def test_padded_empty_zones_are_never_tested_windows():
    """territory_runs zero-fills empty zones; zeros there are not p = 0 windows
    (review case)."""
    from fiberhmm.scan.modules.nuc import ENRICH_KINDS, NucModule
    from fiberhmm.scan.seams import territory_runs
    m = NucModule()
    s = m.resolve({})
    units, _ = plan({'c': 50_000}, genome=True)
    t = {k + ':all': np.zeros(25_000) for k in ('measured', 'qsum') + ENRICH_KINDS}
    for k in ENRICH_KINDS:
        for v in ('p', 'obs', 'exp', 'n'):
            t[f'enr_{v}:{k}'] = np.full(25_000, np.nan)
    for v in ('R', 'null', 'logp', 'n', 'P', 'C'):
        t[f'phase_{v}:all'] = np.full(25_000, np.nan)
    runs = territory_runs([(units[0], t, 'ok'), (units[1], None, 'empty')], m.stitch_support_bp(s))
    for scope in ('scan', 'contig'):
        out = _stitch(m, 'c', [(a, b, lo, hi) for a, b, _x, lo, hi in runs], {k: [x[2][k] for x in runs] for k in t},
                      m.resolve({'enrich_fdr_scope': scope}))
        assert not out.hits


def test_read_ends_outside_segments_are_not_measured(tmp_path):
    """Reads regular from their first to their last dyad, all at one place:
    their unsegmented ends must not count as measured-but-not-regular, or the
    shift null would call the shared middle recurrent (review case)."""
    from nuc_synth import planted_record_molecule
    rng = np.random.default_rng(1)
    path = str(tmp_path / 'ends.bam')
    with pysam.AlignmentFile(path, 'wb', header=header('c', 30_000)) as out:
        for i, st in enumerate(sorted(5000 + rng.integers(0, 300, 25))):
            seq, nucs, msps, marks = planted_record_molecule(rng, 28)
            out.write(make_record(out, f'r{i}', seq, int(st), nucs, msps, (), marks))
    pysam.index(path)
    units = plan({'c': 30_000}, regions=[('c', 5000, 11_000)])
    run_units([{'dataset_id': 'd', 'paths': [path]}], units, modules=['nuc'],
              params={'nuc': {'calibration': 'self'}}, out=tmp_path / 'o', annotate=False)
    hits = read_all(tmp_path / 'o' / 'nuc' / 'hits.tsv.bgz')
    assert not [h for h in hits if h['type'] == 'nuc.patch.regular']


# ------------------------------------------------------------------ calibration and FDR scope (contracts v1.9)
def test_view_scan_calibration_does_not_depend_on_the_plan(scan_bam, tsv_scan, tmp_path, monkeypatch):
    """calibration_scope 'data' (default): a view scan (a --zones subset of the genome grid) calibrates on the
    same windows as the genome scan, so its unit keys are the genome scan's; extending the view recomputes
    neither the calibration nor any computed nuc unit; and the view's molecule and mark rows equal the genome
    scan's for the records its units own. Patches are an FDR over the scanned windows, so they are not compared."""
    from fiberhmm.scan.modules.nuc import NucModule
    params = {'nuc': {'calibration': 'self', 'calibration_population': 400}}
    ds = [{'dataset_id': 'synth', 'paths': [scan_bam]}]
    view = plan({'chrS': LENGTH}, zones=[('chrS', 30_000, 70_000)])
    out = tmp_path / 'view'
    run_units(ds, view, modules=['nuc'], params=params, out=out, annotate=False)
    genome_cal = (tsv_scan / 'nuc' / 'extra' / 'calibration.json').read_bytes()
    assert (out / 'nuc' / 'extra' / 'calibration.json').read_bytes() == genome_cal
    calls = []
    real = NucModule.calibrate

    def counting(self, *a, **kw):
        calls.append(1)
        return real(self, *a, **kw)
    monkeypatch.setattr(NucModule, 'calibrate', counting)
    wider = plan({'chrS': LENGTH}, zones=[('chrS', 10_000, 90_000)])
    run_units(ds, wider, modules=['nuc'], params=params, out=out, annotate=False, continue_run=True, extend=True)
    att = json.loads((out / 'run.json').read_text())['attempts'][-1]
    assert calls == [] and att['units_reused'] == len(view[0])
    assert att['units_run'] == len(wider[0]) - len(view[0])
    owned = [(u.zone_start, u.zone_end) for u in wider[0]]

    def owned_rows(rows):
        return sorted(str(sorted(r.items())) for r in rows if any(a <= int(r['start']) < b for a, b in owned))
    g_mol = read_all(tsv_scan / 'nuc' / 'molecules.tsv.bgz')
    v_mol = read_all(out / 'nuc' / 'molecules.tsv.bgz')
    assert owned_rows(g_mol) and owned_rows(g_mol) == owned_rows(v_mol)
    assert 'input_index' in v_mol[0]
    records = {r['record'] for r in v_mol if any(a <= int(r['start']) < b for a, b in owned)}

    def marks_of(rows):
        return sorted(str(sorted(r.items())) for r in rows if r['record'] in records)
    g_marks, v_marks = read_all(tsv_scan / 'nuc' / 'marks.tsv.bgz'), read_all(out / 'nuc' / 'marks.tsv.bgz')
    assert marks_of(g_marks) and marks_of(g_marks) == marks_of(v_marks)


def test_data_calibration_regions_use_contigs_with_records(scan_bam):
    from fiberhmm.scan.modules.nuc import calibration_regions, data_calibration_regions
    ref = {'chrS': LENGTH, 'chrEmpty': 5_000_000}
    got = data_calibration_regions(ref, [scan_bam], 40, has_records=lambda c, s, e: True)
    assert got and {c for c, _s, _e in got} == {'chrS'}
    # a whole-genome territory gives the same windows
    units, _ = plan({'chrS': LENGTH}, genome=True)
    assert got == calibration_regions(units, 40, has_records=lambda c, s, e: True)


def test_scan_wide_bh_differs_from_per_contig_bh():
    """Benjamini-Hochberg over every tested window of the scan: a window significant within its own small
    contig is not significant among all the scan's windows (and a strong one still is)."""
    from fiberhmm.nuc.enrichment import bh, bh_threshold
    from fiberhmm.scan.modules.nuc import ENRICH_KINDS, NucModule
    m = NucModule()

    def tracks(pvals):
        n = len(pvals) * 250
        t = {k + ':all': [np.ones(n)] for k in ('measured', 'qsum', 'reg', 'dis', 'und', 'low', 'uds', 'mis')}
        for kind in ENRICH_KINDS:
            p = np.full(n, np.nan)
            for i, v in enumerate(pvals):
                p[i * 250] = v
            t[f'enr_p:{kind}'] = [p]
            for name, v in (('obs', 0.5), ('exp', 0.2), ('n', 5.0)):
                t[f'enr_{name}:{kind}'] = [np.where(np.isfinite(p), v, np.nan)]
        return [(0, n, False, False)], t
    rng = np.random.default_rng(3)
    small = [0.004] + [0.9] * 9
    big = list(rng.uniform(0.05, 1.0, 1000))
    for strong, expect_scan in ((0.004, False), (1e-7, True)):
        small[0] = strong
        ta, tra = tracks(small)
        tb, trb = tracks(big)
        contig = m.resolve({'enrich_fdr_scope': 'contig'})
        per = m.stitch('chrA', ta, tra, contig, None)
        assert any(h.type == 'nuc.patch.regular' for h in per.hits)          # within its contig: rejected
        scan = m.resolve({})
        stats = m.combine_statistics({'chrA': m.stitch_statistics('chrA', ta, tra, scan, None),
                                      'chrB': m.stitch_statistics('chrB', tb, trb, scan, None)}, scan, None)
        assert stats['kinds']['reg']['windows_tested'] == 1010
        hits = m.stitch('chrA', ta, tra, scan, None, scan_statistics=stats).hits
        assert any(h.type == 'nuc.patch.regular' for h in hits) == expect_scan
        allp = small + big
        thr = bh_threshold([p for p in allp if p <= 0.1], len(allp), 0.1)
        assert (np.asarray(allp) <= (thr if thr is not None else -1)).tolist() == bh(allp, 0.1).tolist()
    with pytest.raises(ValueError):
        m.stitch('chrA', ta, tra, m.resolve({}), None)              # scan scope without the scan statistics


def test_scan_records_the_fdr_threshold_in_the_manifest(tsv_scan):
    man = json.loads((tsv_scan / 'manifest.json').read_text())
    st = man['modules']['nuc']['scan_statistics']
    assert st['fdr_scope'] == 'scan' and st['enrich_fdr'] == 0.1
    for kind in ('reg', 'dis', 'uds', 'mis'):
        rec = st['kinds'][kind]
        assert rec['windows_tested'] > 0 and rec['windows_rejected'] >= 0
    vals = read_all(tsv_scan / 'nuc' / 'values.tsv.bgz')
    tested = {r['value'] for r in vals if r['value_name'] == 'windows_tested'}
    assert tested <= {repr(float(st['kinds'][k]['windows_tested'])) for k in st['kinds']}


def test_bh_threshold_keeps_p_values_bh_rounds_in():
    """q * k / m can round above q: a p-value one ulp above q is rejected by bh() and must not be pruned
    from the scan-wide candidates (review case)."""
    from fiberhmm.nuc.enrichment import bh, bh_candidate_bound, bh_threshold
    p = np.array([0.01, 0.02, np.nextafter(0.1, np.inf)])
    assert bh(p, 0.1).tolist() == [True, True, True]
    thr = bh_threshold(p[p <= bh_candidate_bound(0.1)], 3, 0.1)
    assert (p <= thr).tolist() == bh(p, 0.1).tolist()
    rng = np.random.default_rng(5)
    for _ in range(200):
        m = int(rng.integers(1, 60))
        p = np.concatenate([rng.uniform(0, 0.02, int(rng.integers(0, 5))), rng.uniform(0, 1, m)])
        p[rng.random(len(p)) < 0.1] = np.nan
        ok = np.isfinite(p)
        thr = bh_threshold(p[ok], int(ok.sum()), 0.1)
        assert ((p <= thr) if thr is not None else np.zeros(len(p), bool)).tolist() == bh(p, 0.1).tolist()


def test_contigs_with_records_do_not_trust_zero_index_counts(scan_bam, monkeypatch):
    """An index without count metadata reports 0 mapped records for contigs that hold some (review case):
    contigs are confirmed by a fetch."""
    import pysam as _pysam
    from fiberhmm.scan.modules import nuc as N

    class Zero:
        def __init__(self, contig):
            self.contig, self.mapped, self.unmapped, self.total = contig, 0, 0, 0
    real = _pysam.AlignmentFile

    class NoCounts:
        def __init__(self, *a, **kw):
            self._bam = real(*a, **kw)

        def get_index_statistics(self):
            return [Zero(c) for c in self._bam.references]

        def __getattr__(self, name):
            return getattr(self._bam, name)
    monkeypatch.setattr(_pysam, 'AlignmentFile', NoCounts)
    assert N._contigs_with_records([scan_bam], {'chrS': LENGTH, 'chrNone': 10_000}) == ['chrS']
