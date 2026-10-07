"""Independent NFR windows keep the single-region bytes across worker counts."""
import json

import pytest

from fiberhmm.inference.accessibility.cli import main


@pytest.mark.parametrize('cap', [0, 25])
def test_bed_matches_single_regions_and_worker_counts(tmp_path, capsys, cap):
    bam = tmp_path/'input.bam'
    import array
    import pysam
    header = pysam.AlignmentHeader.from_dict(dict(HD={'SO': 'coordinate'}, SQ=[{'SN': 'chr1', 'LN': 2000}]))
    with pysam.AlignmentFile(bam, 'wb', header=header) as handle:
        for i in range(50):
            read = pysam.AlignedSegment(header)
            read.query_name = f'molecule{i}'
            read.query_sequence = 'ACGT'*250
            read.reference_id = 0; read.reference_start = 0
            read.mapping_quality = 60; read.cigarstring = '1000M'
            read.set_tag('MM', 'A+a,0,1,2;')
            read.set_tag('ML', array.array('B', [255]*3))
            left, right = 200 + i % 7 - 3, 500 + i % 5 - 2
            read.set_tag('MA', f'1000;nuc.:1-{left},{right+1}-200;msp.:{left+1}-{right-left},{right+201}-{800-right}')
            handle.write(read)
    pysam.index(str(bam))
    params = tmp_path/'parameters.json'
    params.write_text(json.dumps(dict(families=dict(recall_hia5_nucleosomes=False), nfr=dict(kmax=2, splits=2))))
    bed = tmp_path/'sites.bed'
    bed.write_text('chr1\t100\t900\tfirst\nchr1\t120\t800\tsecond\nchr1\t100\t900\trepeated\n')
    common = ['--bam', str(bam), '--chemistry', 'hia5-pacbio', '--parameters', str(params), '--bootstrap', '0', '--cores', '2',
              '--max-discovery-molecules', str(cap), '--max-quantification-molecules', str(cap)]
    for jobs in (1, 2):
        main(common + ['--bed', str(bed), '--window-jobs', str(jobs), '--output', str(tmp_path/f'batch{jobs}')])
    for i, span in enumerate(('100-900', '120-800', '100-900')):
        single = tmp_path/f'single{i}'
        main(common + ['--region', 'chr1:'+span, '--output', str(single)])
        for original in single.iterdir():
            for jobs in (1, 2):
                actual = tmp_path/f'batch{jobs}'/f'window_{i+1:06d}'/original.name
                assert actual.read_bytes() == original.read_bytes(), (jobs, i, original.name)
    rows = json.loads((tmp_path/'batch2'/'regions.json').read_text())
    assert [r['name'] for r in rows] == ['first', 'second', 'repeated']
    assert all(r['nfrs'] > 0 for r in rows)


@pytest.mark.parametrize('extra', [['--window-jobs', '-1'], ['--nfr', '100-200'], ['--classes', 'unused']])
def test_bed_rejects_invalid_or_single_region_controls(tmp_path, extra):
    with pytest.raises(SystemExit) as error:
        main(['--bam', 'unused.bam', '--bed', 'unused.bed', '--output', str(tmp_path/'out'), *extra])
    assert error.value.code == 2
