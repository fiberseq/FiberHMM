"""Public scan API additions, using synthetic BAM snapshots only."""
import json
from pathlib import Path

import pysam
import pytest

from fiberhmm.scan import open_scan, plan, run_units
from fiberhmm.scan.identity import content_id
from fiberhmm.scan.testing import PlantedAnchorModule, PlantedMoleculeModule, write_synthetic_bam


@pytest.mark.parametrize('pg', [
    {'ID': 'ft', 'PN': 'ft', 'CL': 'ft add-nucleosomes calls.bam'},
    {'ID': 'ft.2', 'CL': '/opt/bin/ft fire calls.bam'},
    {'ID': 'caller', 'PN': 'fibertools-rs', 'CL': 'fibertools-rs predict-m6a calls.bam'},
    {'ID': 'fibertools.1', 'CL': 'fibertools add-nucleosomes calls.bam'},
    {'ID': 'caller', 'CL': '"/opt/fiber tools/ft" add-nucleosomes calls.bam'},
])
@pytest.mark.parametrize('calls', ['Ma', 'legacy'])
def test_fibertools_inputs_write_annotated_bams(tmp_path, pg, calls):
    original = write_synthetic_bam(tmp_path / 'original.bam', length=20000)
    path = tmp_path / 'fibertools.bam'
    with pysam.AlignmentFile(original) as src:
        hdr = src.header.to_dict()
        hdr['CO'] = [c for c in hdr['CO'] if 'coord=' not in c]
        hdr['PG'] = [pg, {'ID': 'sort', 'PN': 'samtools', 'PP': pg['ID'], 'CL': 'samtools sort calls.bam'}]
        with pysam.AlignmentFile(str(path), 'wb', header=hdr) as dst:
            for read in src:
                if calls == 'Ma':
                    read.set_tag('Ma', read.get_tag('MA'))
                    read.set_tag('Aq', read.get_tag('AQ'))
                else:
                    import array
                    from fiberhmm.io.ma_tags import parse_ma_tag
                    parsed = parse_ma_tag(read.get_tag('MA'))
                    for kind, starts, lengths in [('nuc', 'ns', 'nl'), ('msp', 'as', 'al')]:
                        ivs = parsed[kind]
                        read.set_tag(starts, array.array('I', [a for a, n in ivs]))
                        read.set_tag(lengths, array.array('I', [n for a, n in ivs]))
                read.set_tag('MA', None)
                read.set_tag('AQ', None)
                dst.write(read)
    pysam.index(str(path))
    from fiberhmm.nuc.tags import record_calls_digest
    with pysam.AlignmentFile(str(path)) as bam:
        digests = {r.query_name: record_calls_digest(r) for r in bam}
    out = tmp_path / 'scan'
    run_units([{'paths': [str(path)]}], plan({'chrS': 20000}, genome=True), modules=['molly'],
              out=out, registry={'molly': PlantedMoleculeModule})
    manifest = open_scan(out).manifest
    assert manifest['outputs']['annotate']
    assert manifest['outputs']['annotated'][0]['marked_records'] > 0
    with pysam.AlignmentFile(str(out / manifest['outputs']['annotated'][0]['path'])) as bam:
        from fiberhmm.io.annotation_frame import ma_annotation_frame
        assert ma_annotation_frame(bam.header) == 'molecular'
        reads = list(bam)
        assert any(r.has_tag('MA') and 'mollymol' in r.get_tag('MA') for r in reads)
        assert {r.query_name: record_calls_digest(r) for r in reads} == digests


def test_unrelated_program_does_not_authorize_molecular_annotation(tmp_path):
    from fiberhmm.scan.runner import _declares_molecular
    path = tmp_path / 'unknown.bam'
    hdr = {'SQ': [{'SN': 'chrS', 'LN': 20000}], 'PG': [
        {'ID': 'ft', 'PN': 'samtools', 'CL': 'samtools view fibertools.bam'}]}
    with pysam.AlignmentFile(str(path), 'wb', header=hdr):
        pass
    assert not _declares_molecular(path)


@pytest.mark.parametrize('callback', [False, True])
@pytest.mark.parametrize('annotate', [False, True])
def test_record_evidence_matches_bam_outputs(tmp_path, callback, annotate):
    path = write_synthetic_bam(tmp_path / 'input.bam', length=30000, step=3000)
    with pysam.AlignmentFile(path) as bam:
        header = bam.header.to_dict()
        records = list(bam)
    units = plan({'chrS': 30000}, zones=[('chrS', 5000, 20000)])
    kw = dict(modules=['plant', 'molly'], params={'plant': {'features': [['x', 10000, 10200, 'both']],
                                                          'states': True}}, annotate=annotate,
              registry={'plant': PlantedAnchorModule, 'molly': PlantedMoleculeModule})
    run_units([{'dataset_id': 'sample', 'paths': [path]}], units, out=tmp_path / 'bam', **kw)
    calls = []
    def source():
        calls.append(True)
        return iter(records)
    run_units([{'dataset_id': 'sample', 'header': header, 'content_ids': [content_id(path)]}], units,
              evidence=source if callback else iter(records), out=tmp_path / 'records', **kw)
    assert len(calls) == int(callback)
    a, b = open_scan(tmp_path / 'bam'), open_scan(tmp_path / 'records')
    assert a.manifest['scan_id'] == b.manifest['scan_id']
    assert a.manifest['units'] == b.manifest['units']
    assert a.manifest['outputs']['tables'] == b.manifest['outputs']['tables']
    for rel in a.manifest['outputs']['tables']:
        assert (tmp_path / 'bam' / rel).read_bytes() == (tmp_path / 'records' / rel).read_bytes()
    if annotate:
        # Headers have source-file metadata, but the annotated alignments match exactly.
        with pysam.AlignmentFile(str(tmp_path / 'bam' / a.manifest['outputs']['annotated'][0]['path'])) as ba, \
             pysam.AlignmentFile(str(tmp_path / 'records' / b.manifest['outputs']['annotated'][0]['path'])) as bb:
            assert [r.to_string() for r in ba] == [r.to_string() for r in bb]
    assert b.summary()['units']['zones'] == [['chrS', 5000, 20000]]


def test_evidence_multi_input_empty_and_frame_validation(tmp_path):
    from fiberhmm.scan.runner import ScanRefused
    path = write_synthetic_bam(tmp_path / 'input.bam', length=20000)
    with pysam.AlignmentFile(path) as bam:
        hdr, recs = bam.header.to_dict(), list(bam)
    units = plan({'chrS': 20000}, genome=True)
    kw = dict(modules=['plant'], registry={'plant': PlantedAnchorModule})
    specs = [{'dataset_id': 'sample', 'headers': [hdr, hdr]}]
    run_units(specs, units, evidence=[('sample', 1, r) for r in recs], out=tmp_path / 'multi', **kw)
    assert open_scan(tmp_path / 'multi').manifest['datasets'][0]['files'][0]['size'] > 0
    with pytest.raises(ScanRefused, match='multi-input'):
        run_units(specs, units, evidence=recs, out=tmp_path / 'ambiguous', **kw)
    wrong = dict(hdr, SQ=[{'SN': 'other', 'LN': 20000}])
    with pytest.raises(ScanRefused, match='reference differs'):
        run_units([{'header': wrong}], units, evidence=recs, out=tmp_path / 'wrong', **kw)


def test_real_modules_evidence_snapshot_matches_bam(tmp_path):
    from fiberhmm.scan.testing import write_daf_locus_bam
    path = write_daf_locus_bam(tmp_path / 'locus.bam')
    with pysam.AlignmentFile(path) as bam:
        header, records = bam.header.to_dict(), list(bam)
    units = plan({'chr1': 6000}, regions=[('chr1', 500, 5000)])
    settings = {'nfr': {'nfr': {'bootstrap': 0, 'kmax': 3, 'splits': 2}}, 'nuc': {'calibration': 'self'}}
    for source, evidence in [('bam', None), ('records', iter(records))]:
        spec = {'dataset_id': 'sample', 'paths': [path]} if evidence is None else {
            'dataset_id': 'sample', 'header': header, 'content_ids': [content_id(path)]}
        run_units([spec], units, modules=['fp', 'nfr', 'nuc', 'peaks'], params=settings, evidence=evidence,
                  annotate=False, out=tmp_path / source)
    a, b = open_scan(tmp_path / 'bam'), open_scan(tmp_path / 'records')
    assert a.manifest['scan_id'] == b.manifest['scan_id']
    assert a.manifest['modules'] == b.manifest['modules']
    assert a.manifest['outputs']['tables'] == b.manifest['outputs']['tables']


def test_inspect_cli_reports_zones(tmp_path, capsys):
    from fiberhmm.scan.cli import main
    path = write_synthetic_bam(tmp_path / 'input.bam', length=30000)
    run_units([{'paths': [path]}], plan({'chrS': 30000}, zones=[('chrS', 5000, 20000)]),
              modules=['plant'], registry={'plant': PlantedAnchorModule}, out=tmp_path / 'scan')
    assert main(['--inspect', str(tmp_path / 'scan')]) == 0
    assert json.loads(capsys.readouterr().out)['summary']['units']['zones'] == [['chrS', 5000, 20000]]


def test_evidence_continue_reuses_snapshot_and_refuses_changes_safely(tmp_path):
    from fiberhmm.scan.runner import ScanRefused
    path = write_synthetic_bam(tmp_path / 'input.bam', length=30000)
    with pysam.AlignmentFile(path) as bam:
        hdr, records = bam.header.to_dict(), list(bam)
    units = plan({'chrS': 30000}, genome=True)
    out = tmp_path / 'scan'
    kw = dict(modules=['plant'], registry={'plant': PlantedAnchorModule}, out=out)
    run_units([{'header': hdr}], units, evidence=records, **kw)
    before = open_scan(out).manifest
    snapshot = Path(before['datasets'][0]['files'][0]['path'])
    original_bytes = snapshot.read_bytes()
    run_units([{'header': hdr}], units, evidence=iter(records), continue_run=True, **kw)
    assert open_scan(out).manifest == before
    assert json.loads((out / 'run.json').read_text())['attempts'][-1]['units_run'] == 0
    changed = list(records[:-1])
    with pytest.raises(ScanRefused, match='continue refused'):
        run_units([{'header': hdr}], units, evidence=changed, continue_run=True, **kw)
    assert snapshot.read_bytes() == original_bytes
    assert open_scan(out).manifest == before


def test_evidence_cancel_removes_partial_snapshot(tmp_path):
    from fiberhmm.scan.runner import ScanRefused
    path = write_synthetic_bam(tmp_path / 'input.bam', length=20000)
    with pysam.AlignmentFile(path) as bam:
        hdr, records = bam.header.to_dict(), list(bam)
    out = tmp_path / 'scan'
    with pytest.raises(ScanRefused, match='cancelled'):
        run_units([{'header': hdr}], plan({'chrS': 20000}, genome=True), modules=['plant'],
                  registry={'plant': PlantedAnchorModule}, evidence=records, cancel=lambda: True, out=out)
    assert not list((out / 'work' / '_evidence').glob('snapshot_*'))
    assert not (out / 'manifest.json').exists()
