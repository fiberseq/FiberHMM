"""fiberhmm-scan (3.2) on real tool output, not synthetic stubs.

The fixtures are BAMs written by real FiberHMM and fibertools runs
(tests/fixtures/fibertools_frame, tests/fixtures/fire_contract). The nuc
module's per-footprint rows must be exactly the BAM's nucleosome and TF calls
(forward and reverse reads, MA and fibertools Ma), its reserved named-class
byte must be 0 (named classes removed in 3.2.0), and the fp/nfr modules must
read every layout, including fibertools-only output.
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import pysam
import pytest

from fiberhmm.io.ma_tags import parse_ma_tag

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / 'tests' / 'fixtures'
FH300 = FIX / 'fibertools_frame' / 'fiberhmm3.0.0_call.bam'
FH321_FT13 = FIX / 'fire_contract' / 'fiberhmm3.2.1_call_ft0.13_fire.bam'
FT13_ONLY = FIX / 'fibertools_frame' / 'ft0.13_addnuc_fire.bam'


def _scan(bam, out, module, *extra):
    env = dict(os.environ, FIBERHMM_NO_UPDATE_CHECK='1')
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(REPO), env.get('PYTHONPATH')]))
    return subprocess.run([sys.executable, '-m', 'fiberhmm.scan.cli', '--bam', str(bam), '--genome', '--modules', module,
                           '--chemistry', 'hia5-pacbio', '--no-annotate', '--cores', '1', '-o', str(out), *extra],
                          capture_output=True, text=True, env=env, cwd=REPO, timeout=900)


def _table(path):
    with gzip.open(path, 'rt') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


@pytest.fixture(scope='module', params=[FH300, FH321_FT13], ids=['fiberhmm3.0.0', 'fiberhmm3.2.1+ft0.13'])
def nuc_scan(request, tmp_path_factory):
    out = tmp_path_factory.mktemp('nucscan') / 'scan'
    proc = _scan(request.param, out, 'nuc')
    assert proc.returncode == 0, proc.stderr[-3000:]
    return request.param, out


def test_nuc_footprint_rows_are_the_bams_calls(nuc_scan):
    bam, out = nuc_scan
    rows = [r for r in _table(out / 'nuc' / 'marks.tsv.bgz') if r['group'] == 'nucsz']
    with pysam.AlignmentFile(str(bam)) as handle:
        reads = list(handle)
    assert {r.is_reverse for r in reads} == {False, True}
    for read in reads:
        tags = dict(read.get_tags())
        calls = parse_ma_tag(str(tags.get('MA') or tags.get('Ma')))
        want = set(calls['nuc']) | set(calls.get('tf') or [])
        got = {(int(r['mol_start']), int(r['mol_len'])) for r in rows if r['qname'] == read.query_name}
        assert got == want, (read.query_name, sorted(want ^ got)[:6])


def test_nuc_writes_no_named_size_class(nuc_scan):
    _bam, out = nuc_scan
    rows = [r for r in _table(out / 'nuc' / 'marks.tsv.bgz') if r['group'] == 'nucsz']
    assert rows and {r['q'].split(',')[1] for r in rows} == {'0'}     # Q2 reserved, always 0
    manifest = (out / 'manifest.json').read_text()
    assert 'nuc.size.named' not in manifest
    molecules = _table(out / 'nuc' / 'molecules.tsv.bgz')
    assert len(molecules) == 2 and not any(k.startswith('named') for k in molecules[0])


def test_nuc_molecule_spacing_on_the_regular_read(nuc_scan):
    _bam, out = nuc_scan
    fwd = next(r for r in _table(out / 'nuc' / 'molecules.tsv.bgz') if r['qname'] == 'read_fwd')
    # the forward read's nucleosomes repeat every ~170-200 bp (tests/fixtures/fibertools_frame/expected.json)
    assert 150 <= float(fwd['nrl']) <= 230, fwd['nrl']
    assert int(float(fwd['n_dyads'])) >= 20


@pytest.mark.parametrize('module', ['fp', 'nfr'])
@pytest.mark.parametrize('bam', [FH300, FH321_FT13, FT13_ONLY], ids=['fiberhmm3.0.0', 'fiberhmm3.2.1+ft0.13', 'ft0.13-only'])
def test_fp_and_nfr_read_every_real_layout(module, bam, tmp_path):
    """Smoke: the module completes on each real layout and writes a unit row per read (two reads are too few for
    classes or hits; the NFR values are compared across tools below)."""
    proc = _scan(bam, tmp_path / 'scan', module)
    assert proc.returncode == 0, proc.stderr[-3000:]
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary['status'] == 'complete' and summary['done'] == 1 and not summary['failed']
    assert summary['tables'][f'{module}/units.tsv.bgz'] == 2


def test_nfr_values_agree_between_fiberhmm_and_fibertools_calls_of_the_same_fibers(tmp_path):
    """The FiberHMM 3.0.0 call and the fibertools 0.13 add-nucleosomes call of the same simulated fibers
    (tests/fixtures/fibertools_frame/expected.json) give the same per-molecule NFR summary: openness, NFR count and
    longest linker, on the forward and the reverse read."""
    rows = {}
    for name, bam in (('fiberhmm', FH300), ('fibertools', FT13_ONLY)):
        out = tmp_path / name
        proc = _scan(bam, out, 'nfr')
        assert proc.returncode == 0, proc.stderr[-3000:]
        rows[name] = {r['qname']: r for r in _table(out / 'nfr' / 'molecules.tsv.bgz')}
    assert set(rows['fiberhmm']) == set(rows['fibertools']) == {'read_fwd', 'read_rev'}
    for q in ('read_fwd', 'read_rev'):
        a, b = rows['fiberhmm'][q], rows['fibertools'][q]
        assert 0.3 < float(a['openness']) < 0.6, a['openness']
        assert abs(float(a['openness']) - float(b['openness'])) < 0.02, (q, a['openness'], b['openness'])
        assert abs(float(a['n_nfr']) - float(b['n_nfr'])) <= 1, (q, a['n_nfr'], b['n_nfr'])
        assert abs(float(a['longest_linker']) - float(b['longest_linker'])) <= 10, (q, a['longest_linker'], b['longest_linker'])


def test_nuc_takes_the_dataset_chemistry_on_fibertools_only_output(tmp_path):
    proc = _scan(FT13_ONLY, tmp_path / 'scan', 'nuc')
    assert proc.returncode == 0, proc.stderr[-3000:]


def test_nuc_module_record_outputs_keeps_its_four_argument_signature():
    """FiberBrowser's live nuc module overrides NucModule._record_outputs(read, record, settings,
    calibration) and calls super() with those four arguments; the dataset --chemistry must not be
    added as a fifth argument."""
    import inspect
    from fiberhmm.scan.modules.nuc import NucModule
    params = list(inspect.signature(NucModule._record_outputs).parameters)
    assert params == ['self', 'read', 'record', 'settings', 'calibration']
