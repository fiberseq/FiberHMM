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
