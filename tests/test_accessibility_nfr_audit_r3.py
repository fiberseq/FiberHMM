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


def test_no_heldout_read_is_an_unanalysable_fit_not_a_fit_on_every_read():
    from fiberhmm.inference.accessibility import run_accessibility
    from fiberhmm.io.folds import FoldSpec
    from test_accessibility_nfr import FAST, payload
    from test_accessibility_nfr_landscape import REGIONS, _units
    spec = FoldSpec.parse('mod=10,discover=0-8')
    units = [u for u in _units(n=480) if spec.discovers(u['read_name'])]
    r = run_accessibility(payload(units), dict(FAST, bootstrap=0, nfr_regions=REGIONS,
                                               discover_fold_spec='mod=10,discover=0-8', fold_weights='heldout'))
    assert r['nfrs'] and all(n['status'] == 'not_analysable' and 'held-out' in n['message'] for n in r['nfrs'])
    assert not any(n['variants'] for n in r['nfrs'])
