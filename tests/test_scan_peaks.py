"""fiberhmm-scan S2: the peaks module, ported from FiberBrowser's peak scan.

Gates:
- per-position scores (numerator, total, smoothed signal) and peaks identical to FiberBrowser's on three
  chemistry fixtures (golden values generated from FiberBrowser's own loader and scoring code; a live
  comparison runs when FIBERHMM_TEST_FIBERBROWSER points at a FiberBrowser 'browser' directory);
- the old (window) rare-tier rule stays reachable; grid scans use the local rule; tiers 0-2 never depend
  on the rule.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

pysam = pytest.importorskip('pysam')

from fiberhmm.scan import peaks_core as C
from fiberhmm.scan.peaks_reads import compact_rows, read_deletion_cuts, split_at_deletions
from fiberhmm.scan.tables import read_all
from fiberhmm.scan.testing import write_peaks_fixture_bam

GOLDEN = Path(__file__).parent / 'fixtures' / 'scan_peaks' / 'browser_golden.json'
CHEMISTRIES = ('hia5-pacbio', 'ddda', 'hia5-nanopore')


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


@pytest.fixture(scope='module')
def fixtures(tmp_path_factory):
    d = tmp_path_factory.mktemp('peaks')
    return {chem: write_peaks_fixture_bam(d / f'{chem}.bam', chem) for chem in CHEMISTRIES}


def _fh(path, chrom, start, end, source, rule='window', params=None, row_rule='view'):
    with pysam.AlignmentFile(path) as bam:
        rows = compact_rows(bam, chrom, start, end, row_rule=row_rule)
    num, tot = C.compute_scan_counts(rows, source, end - start)
    res = C.scan_peaks(num, tot, n_reads=len(rows), rule=rule, params=params)
    return rows, num, tot, res


@pytest.mark.parametrize('chem', CHEMISTRIES)
def test_per_position_scores_identical_to_browser_golden(fixtures, chem):
    golden = json.loads(GOLDEN.read_text())
    views = golden['fixtures'][chem]
    assert len(views) >= 8
    for key, want in views.items():
        chrom, span, source = key.split(':')
        start, end = map(int, span.split('-'))
        rows, num, tot, res = _fh(fixtures[chem], chrom, start, end, source, params=golden['params'])
        assert len(rows) == want['n_reads'], key
        assert _sha(num) == want['numerator'], key
        assert _sha(tot) == want['total'], key
        assert res['smoothed'].dtype == np.float32 and _sha(res['smoothed']) == want['score'], key
        assert res['peaks'] == want['peaks'], key
        assert res['broad'] == want['broad'], key


def _browser():
    base = os.environ.get('FIBERHMM_TEST_FIBERBROWSER')
    if not base or not Path(base, 'services', 'compact_read_loader.py').is_file():
        return None
    if base not in sys.path:
        sys.path.insert(0, base)
    from services.compact_read_loader import build_compact_reads_from_bam
    from services import peak_scanning
    return build_compact_reads_from_bam, peak_scanning


@pytest.mark.parametrize('chem', CHEMISTRIES)
def test_live_browser_parity(fixtures, chem):
    fb = _browser()
    if fb is None:
        pytest.skip('set FIBERHMM_TEST_FIBERBROWSER to a FiberBrowser browser/ directory')
    build, PS = fb
    for start, end in ((0, 30_000), (7_000, 9_000), (21_000, 23_000)):
        for source in C.TILING_SOURCES:
            rows = build(fixtures[chem], 'chrP', start, end, spanning_only=False, include_layers=False,
                         short_terminal_footprint_bp=90, hide_terminal_gaps=True, terminal_gap_threshold=500)
            n1, t1 = PS.compute_scan_counts(rows, source, end - start)
            s1 = PS.smooth_scan_signal(PS.scan_signal_from_counts(n1, t1), 100)
            _rows, n2, t2, res = _fh(fixtures[chem], 'chrP', start, end, source)
            assert len(rows) == len(_rows)
            assert np.array_equal(n1, n2) and np.array_equal(t1, t2) and np.array_equal(s1, res['smoothed'])


def test_deletion_cuts_split_every_call():
    class R:
        cigartuples = [(0, 100), (2, 60), (0, 50), (2, 15), (0, 4), (2, 20), (0, 30), (2, 5), (0, 10)]
        reference_start = 1000
    assert read_deletion_cuts(R()) == [(1100, 1160)]           # 15 + 20 joined over 4 bp = 35 < 50: not cut
    R.cigartuples = [(0, 10), (2, 30), (0, 6), (2, 25), (0, 10)]
    assert read_deletion_cuts(R()) == [(1010, 1040), (1046, 1071)]   # joined event of 55 bp: both pieces cut
    assert split_at_deletions([[0, 100, 147], [100, 120, -20]], [(50, 60)]) == [[0, 50, 147], [60, 100, 147],
                                                                               [100, 120, -20]]


def _scan(argv):
    from fiberhmm.scan.cli import main
    with redirect_stdout(io.StringIO()) as out:
        assert main(argv) == 0
    return json.loads(out.getvalue().strip().splitlines()[-1])


@pytest.mark.parametrize('rules', [{'tier_rule': 'window', 'row_rule': 'view'}, {'tier_rule': 'auto', 'row_rule': 'auto'}])
def test_bed_unit_scan_equals_browser_window(tmp_path, fixtures, rules):
    """The earlier browser rules stay reachable for BED/region units: explicitly, or through 'auto' (the
    pre-decision default, kept for one release)."""
    bam = fixtures['hia5-pacbio']
    p = tmp_path / 'p.json'
    p.write_text(json.dumps({'peaks': rules}))
    _scan(['--bam', bam, '--region', 'chrP:5000-15000', '--modules', 'peaks', '--parameters', str(p),
           '-o', str(tmp_path / 's')])
    hits = read_all(tmp_path / 's' / 'peaks' / 'hits.tsv.bgz')
    _rows, _n, _t, res = _fh(bam, 'chrP', 5000, 15000, 'accessibility')
    want = sorted((5000 + p['start'], 5000 + p['end'], p['layer'], p['score']) for p in res['peaks'])
    got = sorted((int(h['start']), int(h['end']), int(h['tier']), float(h['score'])) for h in hits
                 if h['type'] == 'peaks.peak')
    assert want and got == want
    assert {h['tier_rule'] for h in hits} == {'window'} and {h['row_rule'] for h in hits} == {'view'}
    cal = json.loads((tmp_path / 's' / 'peaks' / 'extra' / 'calibration.json').read_text())
    assert cal['tier_rule'] == 'window' and cal['row_rule'] == 'view'


def test_bed_unit_scan_defaults_to_the_local_rules(tmp_path, fixtures):
    """Owner decision (26 Oct, after the S2 comparison): BED/region units default to the local rare tier and
    every record, as grid scans do, so a CLI BED scan equals the browser's without parameters."""
    bam = fixtures['hia5-pacbio']
    bed = tmp_path / 'u.bed'
    bed.write_text('chrP\t5000\t15000\tu1\n')
    _scan(['--bam', bam, '--bed', str(bed), '--modules', 'peaks', '-o', str(tmp_path / 's')])
    hits = read_all(tmp_path / 's' / 'peaks' / 'hits.tsv.bgz')
    _rows, _n, _t, res = _fh(bam, 'chrP', 5000, 15000, 'accessibility', rule='local', row_rule='records')
    want = sorted((5000 + p['start'], 5000 + p['end'], p['layer'], p['score']) for p in res['peaks'])
    got = sorted((int(h['start']), int(h['end']), int(h['tier']), float(h['score'])) for h in hits
                 if h['type'] == 'peaks.peak')
    assert want and got == want
    assert {h['tier_rule'] for h in hits} == {'local'} and {h['row_rule'] for h in hits} == {'records'}
    m = json.loads((tmp_path / 's' / 'manifest.json').read_text())
    assert m['modules']['peaks']['settings']['tier_rule'] == 'local'
    assert m['modules']['peaks']['settings']['row_rule'] == 'records'


def test_grid_scan_local_rule_stitched_across_offsets(tmp_path, fixtures):
    bam = fixtures['ddda']
    common = ['--bam', bam, '--genome', '--modules', 'peaks', '--unit-bp', '5000', '--stride-bp', '2500',
              '--no-annotate']
    _scan(common + ['-o', str(tmp_path / 'g0')])
    _scan(common + ['--grid-offset', '1250', '-o', str(tmp_path / 'g1')])
    a = (tmp_path / 'g0' / 'peaks' / 'hits.tsv.bgz').read_bytes()
    b = (tmp_path / 'g1' / 'peaks' / 'hits.tsv.bgz').read_bytes()
    assert a == b
    hits = read_all(tmp_path / 'g0' / 'peaks' / 'hits.tsv.bgz')
    assert hits and {h['tier_rule'] for h in hits} == {'local'} and {h['row_rule'] for h in hits} == {'records'}
    # the stitched contig equals one whole-contig computation with the local rule
    _rows, _n, _t, res = _fh(bam, 'chrP', 0, 30_000, 'accessibility', rule='local', row_rule='records')
    assert sorted((p['start'], p['end'], p['layer']) for p in res['peaks']) == sorted(
        (int(h['start']), int(h['end']), int(h['tier'])) for h in hits if h['type'] == 'peaks.peak')


def test_window_rule_refused_for_grid_scans(tmp_path, fixtures):
    p = tmp_path / 'p.json'
    p.write_text(json.dumps({'peaks': {'tier_rule': 'window'}}))
    from fiberhmm.scan.cli import main
    with pytest.raises((SystemExit, ValueError)):
        with redirect_stdout(io.StringIO()):
            main(['--bam', fixtures['ddda'], '--genome', '--modules', 'peaks', '--unit-bp', '5000',
                  '--stride-bp', '2500', '--parameters', str(p), '-o', str(tmp_path / 'x')])


@pytest.mark.parametrize('chem', CHEMISTRIES)
def test_rules_differ_only_in_the_rare_tier(fixtures, chem):
    for start, end in ((0, 30_000), (5_000, 15_000), (20_000, 30_000)):
        _r, num, tot, w = _fh(fixtures[chem], 'chrP', start, end, 'accessibility', rule='window')
        loc = C.scan_peaks(num, tot, n_reads=len(_r), rule='local')
        assert [p for p in w['peaks'] if p['layer'] < 3] == [p for p in loc['peaks'] if p['layer'] < 3]
        assert all(int(tot[p['start']:p['end']].min()) >= 5 for p in loc['peaks'] if p['layer'] == 3)



# ------------------------------------------------------------------ record shapes from the review probes
def _bam(path, records, header_extra=None):
    import array
    header = {'HD': {'VN': '1.6', 'SO': 'coordinate'}, 'SQ': [{'SN': 'chrP', 'LN': 1500}],
              'PG': [{'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call', 'VN': '3.0.2', 'DS': 'coord=molecular'}]}
    header.update(header_extra or {})
    with pysam.AlignmentFile(str(path), 'wb', header=header) as out:
        for spec in records:
            a = pysam.AlignedSegment(out.header)
            a.query_name = spec.get('name', 'm/1/ccs')
            a.flag = spec.get('flag', 0)
            a.reference_id = 0
            a.reference_start = spec.get('start', 100)
            a.mapping_quality = 60
            a.cigarstring = spec.get('cigar', '1000M')
            a.query_sequence = 'A' * a.infer_query_length()
            for tag, value in spec.get('tags', {}).items():
                a.set_tag(tag, array.array('I', value) if isinstance(value, list) else value)
            out.write(a)
    pysam.index(str(path))
    return str(path)


def _rows(path, start=0, end=1500, **kw):
    with pysam.AlignmentFile(path) as bam:
        return compact_rows(bam, 'chrP', start, end, **kw)


def test_string_footprint_tags_parse_like_arrays(tmp_path):
    a = _bam(tmp_path / 'a.bam', [dict(tags={'ns': [100, 700], 'nl': [147, 147]})])
    b = _bam(tmp_path / 'b.bam', [dict(tags={'ns': '100,700', 'nl': '147,147'})])
    assert _rows(a) == _rows(b) and len(_rows(a)) == 1


def test_ns_without_nl_gives_no_row(tmp_path):
    assert _rows(_bam(tmp_path / 'a.bam', [dict(tags={'ns': [100, 700]})])) == []


def test_unrepaired_lift_hides_former_supplementaries(tmp_path):
    lift = {'PG': [{'ID': 'lift', 'PN': 'ft', 'CL': 'ft pg-lift x'}]}
    recs = [dict(tags={'ns': [100, 700], 'nl': [147, 147], 'Yf': 2048})]
    assert _rows(_bam(tmp_path / 'a.bam', recs, lift)) == []
    assert len(_rows(_bam(tmp_path / 'b.bam', recs))) == 1             # not a lift: shown


def test_merged_header_frame_majority(tmp_path):
    from fiberhmm.scan.peaks_reads import legacy_frame
    pgs = [{'ID': 'mm2', 'PN': 'minimap2'},
           {'ID': 'ft', 'PN': 'fibertools', 'CL': 'ft add-nucleosomes in.bam', 'PP': 'mm2'},
           {'ID': 'ft.1', 'PN': 'fibertools', 'CL': 'ft add-nucleosomes in.bam', 'PP': 'mm2'},
           {'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call', 'PP': 'mm2'}]
    assert legacy_frame({'PG': pgs}) == (True, False)
    rec = [dict(flag=16, tags={'ns': [100, 700], 'nl': [147, 147]})]
    rows = _rows(_bam(tmp_path / 'a.bam', rec, {'PG': pgs}))
    first = min(iv for iv in rows[0]['intervals'] if iv[2] > 0)
    assert first[:2] == [253, 400]                                    # flipped as molecular


def test_records_rule_counts_do_not_depend_on_the_zone(tmp_path):
    rec = [dict(flag=0x800, tags={'ns': [400, 700], 'nl': [147, 147]})]
    path = _bam(tmp_path / 'a.bam', rec)
    whole = C.compute_scan_counts(_rows(path, row_rule='records'), 'accessibility', 1500)
    for cut in (250, 500, 600, 1000):
        left = C.compute_scan_counts(_rows(path, 0, cut, row_rule='records'), 'accessibility', cut)
        right = C.compute_scan_counts(_rows(path, cut, 1500, row_rule='records'), 'accessibility', 1500 - cut)
        for k in (0, 1):
            assert np.array_equal(np.concatenate([left[k], right[k]]), whole[k]), cut


def test_review_records_match_browser(tmp_path):
    fb = _browser()
    if fb is None:
        pytest.skip('set FIBERHMM_TEST_FIBERBROWSER to a FiberBrowser browser/ directory')
    build, PS = fb
    pgs = [{'ID': 'mm2', 'PN': 'minimap2'},
           {'ID': 'ft', 'PN': 'fibertools', 'CL': 'ft add-nucleosomes in.bam', 'PP': 'mm2'},
           {'ID': 'ft.1', 'PN': 'fibertools', 'CL': 'ft add-nucleosomes in.bam', 'PP': 'mm2'},
           {'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call', 'PP': 'mm2'}]
    cases = [([dict(tags={'ns': '100,700', 'nl': '147,147'})], None),
             ([dict(tags={'ns': [100, 700]})], None),
             ([dict(tags={'ns': [100, 700], 'nl': [147, 147], 'Yf': 2048})],
              {'PG': [{'ID': 'lift', 'PN': 'ft', 'CL': 'ft pg-lift x'}]}),
             ([dict(flag=16, tags={'ns': [100, 700], 'nl': [147, 147]})], {'PG': pgs})]
    for i, (recs, extra) in enumerate(cases):
        path = _bam(tmp_path / f'c{i}.bam', recs, extra)
        for start, end in ((0, 1500), (300, 900)):
            rows = build(path, 'chrP', start, end, spanning_only=False, include_layers=False,
                         short_terminal_footprint_bp=90, hide_terminal_gaps=True, terminal_gap_threshold=500)
            a = PS.compute_scan_counts(rows, 'accessibility', end - start)
            b = C.compute_scan_counts(_rows(path, start, end), 'accessibility', end - start)
            assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), (i, start, end)



def test_empty_zones_do_not_truncate_stitched_peaks(tmp_path):
    from fiberhmm.scan import plan, run_units
    path = _bam(tmp_path / 'a.bam', [dict(start=421, cigar='79M', tags={'ns': [0], 'nl': [79]})])
    hits = []
    for offset in (0, 250):
        units = plan({'chrP': 1500}, genome=True, unit_bp=500, stride_bp=500, grid_offset=offset)
        out = tmp_path / f'o{offset}'
        run_units([{'dataset_id': 'x', 'paths': [path]}], units, modules=['peaks'], out=out, annotate=False)
        hits.append((out / 'peaks' / 'hits.tsv.bgz').read_bytes())
    assert hits[0] == hits[1]


def test_msp_only_string_tags_do_not_crash(tmp_path):
    path = _bam(tmp_path / 'a.bam', [dict(tags={'as': '100,700', 'al': '147,147'})])
    assert isinstance(_rows(path), list)


def test_wide_smoothing_stitched_peaks_do_not_depend_on_unit_placement(tmp_path):
    from fiberhmm.scan import plan, run_units
    path = _bam(tmp_path / 'a.bam', [dict(start=701, cigar='250M', tags={})], {'SQ': [{'SN': 'chrP', 'LN': 4000}]})
    params = {'peaks': {'smooth_width': 2000}}
    hits = []
    for offset in (0, 250):
        units = plan({'chrP': 4000}, genome=True, unit_bp=500, stride_bp=500, grid_offset=offset)
        out = tmp_path / f'o{offset}'
        run_units([{'dataset_id': 'x', 'paths': [path]}], units, modules=['peaks'], params=params, out=out,
                  annotate=False)
        hits.append(read_all(out / 'peaks' / 'hits.tsv.bgz'))
    assert hits[0] == hits[1]
    with pysam.AlignmentFile(path) as bam:
        rows = compact_rows(bam, 'chrP', 0, 4000, row_rule='records')
    num, tot = C.compute_scan_counts(rows, 'accessibility', 4000)
    whole = C.scan_peaks(num, tot, n_reads=len(rows), rule='local', params={'smooth_width': 2000})
    assert sorted((p['start'], p['end'], p['layer']) for p in whole['peaks']) == sorted(
        (int(h['start']), int(h['end']), int(h['tier'])) for h in hits[0] if h['type'] == 'peaks.peak')


def test_local_rule_with_zero_rare_threshold_needs_coverage():
    num = np.zeros(1500, dtype=np.int32)
    tot = np.zeros(1500, dtype=np.int32)
    tot[100:1100] = 10
    res = C.scan_peaks(num, tot, n_reads=10, rule='local', params={'threshold_rare': 0.0})
    for p in res['peaks']:
        if p['layer'] == 3:
            assert tot[p['start']:p['end']].min() >= 5
