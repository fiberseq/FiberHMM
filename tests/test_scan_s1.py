"""fiberhmm-scan S1: the fp and nfr modules and the CLI.

Gates:
- fp and nfr rows on one BED unit equal fiberhmm-consensus / fiberhmm-nfr outputs with the same caps;
- running a subset of modules gives the same rows per module;
- a byte-identical rerun (annotated mode, real modules);
- annotated MA groups read back as the TSV states and scalars.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

pysam = pytest.importorskip('pysam')

from fiberhmm.scan.tables import read_all

REGION = 'chr1:500-5000'
NFR_PARAMS = {'nfr': {'bootstrap': 0, 'kmax': 3, 'splits': 2}}


@pytest.fixture(scope='module')
def locus_bam(tmp_path_factory):
    from fiberhmm.scan.testing import write_daf_locus_bam
    return write_daf_locus_bam(tmp_path_factory.mktemp('locus') / 'locus.bam')


def _scan(argv):
    from fiberhmm.scan.cli import main
    with redirect_stdout(io.StringIO()) as out:
        assert main(argv) == 0
    return json.loads(out.getvalue().strip().splitlines()[-1])


def _caps(cap):
    return ['--max-discovery-molecules', str(cap), '--max-quantification-molecules', str(cap)] if cap else []


def _tsv(path, gz=False):
    opener = (lambda p: gzip.open(p, 'rt', newline='')) if gz else (lambda p: open(p, newline=''))
    with opener(path) as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def _scan_dir_state_index(d: Path, module: str):
    hits = {r['hit_id']: r for r in read_all(d / module / 'hits.tsv.bgz')}
    units = {}
    for r in read_all(d / module / 'units.tsv.bgz'):
        units[r['record']] = r['evidence_unit']
    return hits, units


@pytest.mark.parametrize('cap', [0, 25])
def test_fp_bed_unit_rows_equal_consensus(tmp_path, locus_bam, cap):
    from fiberhmm.inference.consensus.cli import main as consensus
    with redirect_stdout(io.StringIO()):
        consensus(['--bam', locus_bam, '--region', REGION, '--output', str(tmp_path / 'cons'), '--no-bam',
                   '--cores', '1', *_caps(cap)])
    _scan(['--bam', locus_bam, '--region', REGION, '--modules', 'fp', '-o', str(tmp_path / 'scan'), '--cores', '1',
           *_caps(cap)])
    cons = _tsv(tmp_path / 'cons' / 'classes.tsv')
    scan = read_all(tmp_path / 'scan' / 'fp' / 'classes.tsv.bgz')
    assert cons, 'the fixture must yield classes'
    cols = list(cons[0])
    key = lambda r: tuple(r[c] for c in cols)
    assert sorted(map(key, cons)) == sorted(map(key, scan))
    hits, units = _scan_dir_state_index(tmp_path / 'scan', 'fp')
    assert sorted(h['class_id'] for h in hits.values()) == sorted({r['class_id'] for r in cons})
    mols = _tsv(tmp_path / 'cons' / 'molecules.tsv.gz', gz=True)
    label = {'member': '1', 'non_member': '0', 'abstain': '.'}
    want = sorted((m['class_id'], m['channel'], m['unit_id'], label[m['label']], m['posterior'], m['log_bf'], m['tier'])
                  for m in mols)
    got = sorted((hits[s['hit_id']]['class_id'], s['channel'], units[s['record']], s['state'], s['posterior'],
                  s['log_bf'], s['tier']) for s in read_all(tmp_path / 'scan' / 'fp' / 'states.tsv.bgz'))
    norm = lambda rows: [(a, b, c, d, float(e), float(f), g) for a, b, c, d, e, f, g in rows]
    assert norm(want) == norm(got)


@pytest.mark.parametrize('cap', [0, 25])
def test_nfr_bed_unit_rows_equal_fiberhmm_nfr(tmp_path, locus_bam, cap):
    from fiberhmm.inference.accessibility.cli import main as nfr
    params = tmp_path / 'p.json'
    params.write_text(json.dumps(NFR_PARAMS))
    with redirect_stdout(io.StringIO()):
        nfr(['--bam', locus_bam, '--region', REGION, '--output', str(tmp_path / 'nfr'), '--parameters', str(params),
             '--cores', '1', *_caps(cap)])
    sp = tmp_path / 'sp.json'
    sp.write_text(json.dumps({'nfr': NFR_PARAMS}))
    _scan(['--bam', locus_bam, '--region', REGION, '--modules', 'nfr', '--parameters', str(sp),
           '-o', str(tmp_path / 'scan'), '--cores', '1', *_caps(cap)])
    want = _tsv(tmp_path / 'nfr' / 'variants.tsv')
    got = read_all(tmp_path / 'scan' / 'nfr' / 'variants.tsv.bgz')
    assert want, 'the fixture must yield NFR variants'
    cols = list(want[0])
    assert sorted(tuple(r[c] for c in cols) for r in want) == sorted(tuple(r[c] for c in cols) for r in got)
    hits, units = _scan_dir_state_index(tmp_path / 'scan', 'nfr')
    mols = [m for m in _tsv(tmp_path / 'nfr' / 'molecules.tsv.gz', gz=True) if m['status'] == 'callable']
    want_states = sorted((m['nfr'], m['unit_id'], '1' if m['gaps'] else '0', m['map'], m['map_posterior'])
                         for m in mols)
    states = read_all(tmp_path / 'scan' / 'nfr' / 'states.tsv.bgz')
    got_states = sorted((hits[s['hit_id']]['nfr'], units[s['record']], s['state'], s['map'], s['map_posterior'])
                        for s in states if hits[s['hit_id']]['type'] == 'nfr.region')
    assert want_states == got_states
    # variant memberships: P per variant as written by the engine
    want_p = sorted((m['nfr'], m['unit_id'], kv.split('=')[0], float(kv.split('=')[1]))
                    for m in mols for kv in (m['p'].split(';') if m['p'] else []))
    got_p = sorted((hits[s['hit_id']]['nfr'], units[s['record']], hits[s['hit_id']]['name'], float(s['p']))
                   for s in states if hits[s['hit_id']]['type'] == 'nfr.variant')
    assert want_p == got_p


def _module_files(d: Path, module: str):
    return {p.name: p.read_bytes() for p in sorted((d / module).glob('*.tsv.bgz'))}


def test_module_subsets_give_the_same_rows(tmp_path, locus_bam):
    sp = tmp_path / 'sp.json'
    sp.write_text(json.dumps({'nfr': NFR_PARAMS}))
    common = ['--bam', locus_bam, '--region', REGION, '--cores', '1', '--parameters', str(sp)]
    _scan(common + ['--modules', 'fp,nfr', '-o', str(tmp_path / 'both')])
    _scan(common + ['--modules', 'nfr', '-o', str(tmp_path / 'nfr')])
    sp_fp = tmp_path / 'fp.json'
    sp_fp.write_text('{}')
    _scan(['--bam', locus_bam, '--region', REGION, '--cores', '1', '--modules', 'fp', '-o', str(tmp_path / 'fp')])
    both_fp, both_nfr = _module_files(tmp_path / 'both', 'fp'), _module_files(tmp_path / 'both', 'nfr')
    assert both_fp and both_nfr
    assert both_fp == _module_files(tmp_path / 'fp', 'fp')
    assert both_nfr == _module_files(tmp_path / 'nfr', 'nfr')


def _files(d: Path):
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob('*'))
            if p.is_file() and 'work' not in p.relative_to(d).parts and p.name not in ('run.json', '.scan.lock')}


def test_byte_identical_rerun_annotated(tmp_path, locus_bam):
    sp = tmp_path / 'sp.json'
    sp.write_text(json.dumps({'nfr': NFR_PARAMS}))
    common = ['--bam', locus_bam, '--region', REGION, '--modules', 'fp,nfr', '--parameters', str(sp), '--annotate']
    _scan(common + ['-o', str(tmp_path / 'a'), '--cores', '1'])
    _scan(common + ['-o', str(tmp_path / 'b'), '--cores', '2'])
    a, b = _files(tmp_path / 'a'), _files(tmp_path / 'b')
    assert sorted(a) == sorted(b) and any(k.endswith('.scan.bam') for k in a)
    for k in a:
        assert a[k] == b[k], k


def test_annotated_groups_read_back_as_tsv(tmp_path, locus_bam):
    from fiberhmm.io.scan_ma import decode_scalar, project_to_molecule, scan_contracts, scan_groups_of
    from fiberhmm.scan.identity import record_sha256
    from fiberhmm.scan.modules.fp import TIERS
    sp = tmp_path / 'sp.json'
    sp.write_text(json.dumps({'nfr': NFR_PARAMS}))
    common = ['--bam', locus_bam, '--region', REGION, '--modules', 'fp,nfr', '--parameters', str(sp), '--cores', '1']
    _scan(common + ['--annotate', '-o', str(tmp_path / 'ann')])
    _scan(common + ['--no-annotate', '-o', str(tmp_path / 'tsv')])
    m = json.loads((tmp_path / 'ann' / 'manifest.json').read_text())
    assert m['outputs']['annotate'] and not (tmp_path / 'ann' / 'fp' / 'states.tsv.bgz').exists()
    ann_path = tmp_path / 'ann' / m['outputs']['annotated'][0]['path']
    states = {}
    for module in ('fp', 'nfr'):
        for s in read_all(tmp_path / 'tsv' / module / 'states.tsv.bgz'):
            states.setdefault(s['record'], []).append((module, s))
    scalars = {r['record']: r for r in read_all(tmp_path / 'tsv' / 'nfr' / 'molecules.tsv.bgz')}
    hits = {r['hit_id']: r for mod in ('fp', 'nfr') for r in read_all(tmp_path / 'tsv' / mod / 'hits.tsv.bgz')}
    code = {'1': 1, '0': 2, '.': 3}
    checked = 0
    with pysam.AlignmentFile(locus_bam) as src, pysam.AlignmentFile(str(ann_path)) as ann:
        contracts = scan_contracts(ann.header)
        assert set(contracts) == {'fpcls', 'nfrvar', 'nfrmol'}
        assert all(c['scan_id'] == m['scan_id'] for c in contracts.values())
        seen = {}
        for s_read, a_read in zip(src.fetch(until_eof=True), ann.fetch(until_eof=True)):
            assert s_read.query_name == a_read.query_name
            sha = record_sha256(s_read)
            occ = seen.get(sha, 0)
            seen[sha] = occ + 1
            rec = f'{sha}:{occ}'
            groups = scan_groups_of(a_read, contracts)
            want = set()
            for module, s in states.get(rec, []):
                h = hits[s['hit_id']]
                iv = project_to_molecule(s_read, int(s['start']), int(s['end']), s_read.query_length)
                if module == 'fp':
                    q = (code[s['state']], int(round(255 * float(s['posterior']))), TIERS[s['tier']])
                    want.add(('fpcls', s['hit_id'], iv, q))
                else:
                    p = 255 if (h['type'] == 'nfr.region' and s['state'] == '1') else 0 if h['type'] == 'nfr.region' \
                        else int(round(255 * float(s['p'])))
                    want.add(('nfrvar', s['hit_id'], iv, (code[s['state']], p)))
            got = {(g, item['name'], (item['start'], item['length']), tuple(item['q']))
                   for g in ('fpcls', 'nfrvar') for item in groups.get(g, [])}
            assert got == want
            mol = groups.get('nfrmol', [])
            exact = scalars.get(rec)
            if exact is not None:
                assert len(mol) == 1 and mol[0]['start'] == 0 and mol[0]['length'] == a_read.query_length
                for name, tol in (('openness', 0.002 + 1e-9), ('n_nfr', 0), ('longest_linker', 0)):
                    if exact[name] == '':
                        assert mol[0]['scalars'][name] is None
                    else:
                        assert abs(mol[0]['scalars'][name] - float(exact[name])) <= tol
            checked += 1
    assert checked == m['outputs']['annotated'][0]['records'] > 0


def test_cli_schema_inspect_and_refusals(tmp_path, locus_bam, capsys):
    from fiberhmm.scan.cli import main
    assert main(['--schema', '--modules', 'fp,nfr']) == 0
    schema = json.loads(capsys.readouterr().out)
    assert {'fp', 'nfr', 'read_policy'} <= set(schema) and schema['nfr']['experimental'] is True
    out = tmp_path / 'scan'
    _scan(['--bam', locus_bam, '--region', REGION, '--modules', 'fp', '-o', str(out), '--cores', '1'])
    capsys.readouterr()
    assert main(['--inspect', str(out)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['problems'] == [] and report['summary']['modules']['fp']['hits'] >= 1
    with pytest.raises(SystemExit) as error:
        main(['--bam', locus_bam, '--region', REGION, '--modules', 'fp', '-o', str(out)])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        main(['--bam', locus_bam, '--region', 'chr1:0-60000', '--modules', 'fp', '-o', str(tmp_path / 'x'),
              '--unit-bp', '60000'])
    assert error.value.code == 2
    (out / 'fp' / 'hits.tsv.bgz').write_bytes(b'')
    assert main(['--inspect', str(out)]) == 1


def test_continue_with_real_modules(tmp_path, locus_bam):
    sp = tmp_path / 'sp.json'
    sp.write_text(json.dumps({'nfr': NFR_PARAMS}))
    common = ['--bam', locus_bam, '--region', 'chr1:0-6000', '--unit-bp', '2500', '--stride-bp', '1250',
              '--modules', 'fp,nfr', '--parameters', str(sp), '--cores', '1']
    _scan(common + ['-o', str(tmp_path / 'full')])
    out = tmp_path / 'cont'
    _scan(common + ['-o', str(out)])
    for marker in sorted((out / 'work' / 'nfr').glob('*/marker.json'))[1:3]:
        marker.unlink()
    (out / 'manifest.json').unlink()
    _scan(common + ['-o', str(out), '--continue'])
    a, b = _files(tmp_path / 'full'), _files(out)
    assert sorted(a) == sorted(b)
    for k in a:
        assert a[k] == b[k], k
