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
