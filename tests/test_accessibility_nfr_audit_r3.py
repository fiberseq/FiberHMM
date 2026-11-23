"""fiberhmm-nfr BED runs and cross-fitting: audit round 3 regressions (scalar dataset paths, fold-aware loading and
held-out fitting, mask identity on --continue, relocated output directories)."""
import json

import pytest

from fiberhmm.inference.accessibility import batch


def test_run_record_accepts_a_scalar_dataset_path(tmp_path):
    bam = tmp_path/'sample.bam'
    bam.write_bytes(b'not read')
    record = batch.run_record([dict(dataset_id='d', paths=str(bam))], [dict(chrom='chr1', start=0, end=10, name='w')], {}, {})
    assert record['datasets'][0]['paths'] == [str(bam)]
    assert [f['path'] for f in record['files']] == [str(bam.resolve())]
    listed = batch.run_record([dict(dataset_id='d', paths=[str(bam)])], [dict(chrom='chr1', start=0, end=10, name='w')], {}, {})
    assert listed == record


def test_bed_run_with_a_scalar_dataset_path_matches_the_list_form(tmp_path, capsys):
    from fiberhmm.inference.accessibility.cli import main
    from test_accessibility_nfr import nfr_bam
    bam = nfr_bam(tmp_path/'nfr.bam')
    (tmp_path/'w.bed').write_text('chr1\t700\t2400\tlocus\n')
    for name, paths in (('scalar', str(bam)), ('listed', [str(bam)])):
        (tmp_path/f'{name}.json').write_text(json.dumps([dict(dataset_id='d', paths=paths)]))
        main(['--datasets', str(tmp_path/f'{name}.json'), '--bed', str(tmp_path/'w.bed'), '--bootstrap', '0', '--cores', '1',
              '--fast-landscape', '--output', str(tmp_path/name)])
    capsys.readouterr()
    rows = json.loads((tmp_path/'scalar'/'regions.json').read_text())
    assert [r['status'] for r in rows] == ['complete']
    for f in ('variants.tsv', 'configurations.tsv', 'molecules.tsv.gz'):
        assert (tmp_path/'scalar'/'window_000001'/f).read_bytes() == (tmp_path/'listed'/'window_000001'/f).read_bytes()



@pytest.mark.parametrize('bed', [False, True])
def test_capped_loading_under_cross_fitting_keeps_every_discovery_fold_read(tmp_path, capsys, bed):
    from fiberhmm.inference.accessibility.cli import main
    from test_accessibility_nfr import nfr_bam
    bam = nfr_bam(tmp_path/'nfr.bam')
    window = ['--region', 'chr1:700-2400']
    if bed:
        (tmp_path/'w.bed').write_text('chr1\t700\t2400\tlocus\n')
        window = ['--bed', str(tmp_path/'w.bed')]
    common = ['--bam', str(bam), *window, '--bootstrap', '0', '--cores', '1', '--fast-landscape',
              '--discover-fold-spec', 'mod=10,discover=0-4']
    main(common + ['--output', str(tmp_path/'full')])
    main(common + ['--max-discovery-molecules', '10', '--max-quantification-molecules', '10',
                   '--output', str(tmp_path/'capped')])
    capsys.readouterr()
    sub = 'window_000001' if bed else '.'
    full, capped = (json.loads((tmp_path/d/sub/'result.json').read_text())['cross_fit'] for d in ('full', 'capped'))
    # the loader no longer drops discovery-fold (or held-out) molecules before the in-memory, fold-aware caps
    assert full['discovery_molecules'] > 20
    assert (capped['discovery_molecules'], capped['heldout_molecules']) == (full['discovery_molecules'], full['heldout_molecules'])


def test_heldout_fit_uses_fold_membership_not_the_discovery_sample(tmp_path, monkeypatch):
    from fiberhmm.inference.accessibility import run_accessibility
    from fiberhmm.inference.accessibility import variants as V
    from fiberhmm.io.folds import FoldSpec
    from test_accessibility_nfr import FAST, payload
    from test_accessibility_nfr_landscape import REGIONS, _units
    units = _units(n=480)
    name = {u['unit_id']: u['read_name'] for u in units}
    spec = FoldSpec.parse('discover=0-4')
    fits = []
    real = V.quantify

    def spy(reads, *a, fit=None, **k):
        fits.append(fit)
        return real(reads, *a, fit=fit, **k)
    monkeypatch.setattr(V, 'quantify', spy)
    run_accessibility(payload(units), dict(FAST, bootstrap=0, nfr_regions=REGIONS, discover_fold_spec='discover=0-4',
                                           fold_weights='heldout', max_discovery_molecules=5))
    assert fits and all(fits)
    assert not any(spec.discovers(name[uid]) for fit in fits for _, uid in fit)   # no discovery-fold read in the fit


def test_no_heldout_read_is_an_unanalysable_fit_not_a_fit_on_every_read(tmp_path):
    from fiberhmm.inference.accessibility import run_accessibility
    from fiberhmm.io.folds import FoldSpec
    from test_accessibility_nfr import FAST, payload
    from test_accessibility_nfr_landscape import REGIONS, _units
    spec = FoldSpec.parse('mod=10,discover=0-8')
    units = [u for u in _units(n=480) if spec.discovers(u['read_name'])]
    r = run_accessibility(payload(units), dict(FAST, bootstrap=0, nfr_regions=REGIONS,
                                               discover_fold_spec='mod=10,discover=0-8', fold_weights='heldout'),
                         tmp_path/'out')
    assert r['nfrs'] and all(n['status'] == 'not_analysable' and 'held-out' in n['message'] for n in r['nfrs'])
    assert not any(n['variants'] for n in r['nfrs'])
    for name in ('variants.tsv', 'configurations.tsv', 'molecules.tsv.gz', 'coaccess.tsv', 'result.json'):
        assert (tmp_path/'out'/name).is_file()            # every writer (and the co-accessibility stage) copes


@pytest.fixture
def no_run_mask(monkeypatch):
    from fiberhmm.core import bam_reader
    for name in ('FIBERHMM_DAF_RUN_MASK', 'FIBERHMM_DAF_RUN_POLICY'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(bam_reader, '_DAF_RUN_MASK_MIN', None)
    monkeypatch.setattr(bam_reader, '_DAF_RUN_POLICY', None)
    monkeypatch.setattr(bam_reader, '_DAF_RUN_MASK_CONFIGURED', False)
    return bam_reader


def test_continue_refuses_a_changed_snp_mask_or_run_mask(tmp_path, monkeypatch, no_run_mask):
    import os
    bam = tmp_path/'sample.bam'; bam.write_bytes(b'not read')
    mask = tmp_path/'mask.bed'; mask.write_text('chr1\t10\t11\n')
    datasets, windows = [dict(dataset_id='d', paths=[str(bam)])], [dict(chrom='chr1', start=0, end=100, name='w')]
    values = {'input': {'daf_snp_mask': str(mask)}}
    out = tmp_path/'run'; out.mkdir()
    first = batch.run_record(datasets, windows, values, {})
    digest = batch.open_run(out, first, False)
    assert batch.open_run(out, batch.run_record(datasets, windows, values, {}), True) == digest
    # the mask edited in place, size and mtime preserved
    st = mask.stat()
    mask.write_text('chr1\t20\t21\n'); os.utime(mask, ns=(st.st_atime_ns, st.st_mtime_ns))
    with pytest.raises(batch.ContinueRefused, match='daf_snp_mask'):
        batch.open_run(out, batch.run_record(datasets, windows, values, {}), True)
    mask.write_text('chr1\t10\t11\n'); os.utime(mask, ns=(st.st_atime_ns, st.st_mtime_ns))
    # an explicit DAF run mask from the environment
    monkeypatch.setenv('FIBERHMM_DAF_RUN_MASK', '6')
    with pytest.raises(batch.ContinueRefused, match='daf_run_mask'):
        batch.open_run(out, batch.run_record(datasets, windows, values, {}), True)


def test_runs_without_masks_keep_their_record(tmp_path, no_run_mask):
    bam = tmp_path/'sample.bam'; bam.write_bytes(b'not read')
    record = batch.run_record([dict(dataset_id='d', paths=[str(bam)])], [dict(chrom='chr1', start=0, end=9, name='w')], {}, {})
    assert set(record) == {'datasets', 'files', 'windows', 'loader_parameters', 'nfr', 'fiberhmm'}


def test_continue_after_moving_the_output_directory_keeps_reused_windows(tmp_path, capsys):
    from fiberhmm.inference.accessibility.cli import main
    from test_accessibility_nfr import nfr_bam
    bam = nfr_bam(tmp_path/'nfr.bam')
    bed = tmp_path/'w.bed'
    bed.write_text('chr1\t700\t2400\tlocus\nchr1\t800\t2300\tinner\n')
    common = ['--bam', str(bam), '--bed', str(bed), '--bootstrap', '0', '--cores', '1', '--fast-landscape']
    main(common + ['--output', str(tmp_path/'run')])
    capsys.readouterr()
    before = (tmp_path/'run'/'variants.tsv').read_text()
    (tmp_path/'run'/'window_000002'/'window_complete.json').unlink()        # interrupted before the second window ended
    (tmp_path/'run').rename(tmp_path/'moved')
    main(common + ['--output', str(tmp_path/'moved'), '--continue'])
    capsys.readouterr()
    summary = json.loads((tmp_path/'moved'/'regions.json').read_text())
    assert [r['status'] for r in summary] == ['complete', 'complete']
    assert [r['folder'] for r in summary] == [str((tmp_path/'moved'/f'window_{i:06d}').resolve()) for i in (1, 2)]
    assert {ln.split('\t')[0] for ln in (tmp_path/'moved'/'variants.tsv').read_text().splitlines()[1:]} == {'locus', 'inner'}
    assert (tmp_path/'moved'/'variants.tsv').read_text() == before
