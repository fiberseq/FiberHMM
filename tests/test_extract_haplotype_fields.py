"""Focused coverage for opt-in HP/PS BED and bigBed extraction."""

from __future__ import annotations

import io
import inspect
import shutil
from array import array

import numpy as np
import pysam
import pytest

from fiberhmm.cli.extract_tags import (
    MISSING_HAPLOTYPE_VALUE,
    _extract_tfs,
    _haplotype_columns,
    bed_to_bigbed,
    extract_tags_parallel,
)
from fiberhmm.io.autosql import (
    HAPLOTYPE_FIELD_COUNT,
    get_schema,
    write_autosql_for,
)


EXTRACT_TYPES = ('nucleosome', 'msp', 'tf', 'm6a', 'm5c', 'deam')


def test_parallel_extractor_preserves_legacy_positional_parameter_order():
    """The opt-in field must not shift arguments accepted before v2.16.4."""
    parameter_names = list(inspect.signature(extract_tags_parallel).parameters)
    assert parameter_names == [
        'input_bam',
        'output_beds',
        'extract_types',
        'n_cores',
        'region_size',
        'min_mapq',
        'prob_threshold',
        'with_scores',
        'min_tq',
        'block_scores',
        'circular_groups',
        'skip_scaffolds',
        'chroms',
        'sort_mem',
        'sort_parallel',
        'haplotype_fields',
    ]


class _FakeRead:
    def __init__(self):
        self._tags = {}
        self.query_name = 'phased_read'
        self.reference_name = 'chr1'
        self.is_reverse = False

    def has_tag(self, tag):
        return tag in self._tags

    def get_tag(self, tag):
        if tag not in self._tags:
            raise KeyError(tag)
        return self._tags[tag]

    def set_tag(self, tag, value):
        self._tags[tag] = value


def _identity_map(length=200, start=1000):
    return np.arange(start, start + length, dtype=np.int64)


def _tf_read(*, phased=True):
    read = _FakeRead()
    read.set_tag('MA', '200;tf+QQQ:11-20')
    read.set_tag('AQ', array('B', [180, 20, 30]))
    if phased:
        read.set_tag('HP', 2)
        read.set_tag('PS', 4242)
    return read


def _write_phased_and_unphased_bam(path):
    header = pysam.AlignmentHeader.from_dict({
        'HD': {'VN': '1.6', 'SO': 'coordinate'},
        'SQ': [{'SN': 'chr1', 'LN': 10_000}],
    })
    sequence = 'ACGT' * 50
    with pysam.AlignmentFile(path, 'wb', header=header) as out:
        for name, ref_start, phased in (
            ('phased', 1000, True),
            ('unphased', 2000, False),
        ):
            read = pysam.AlignedSegment(header)
            read.query_name = name
            read.query_sequence = sequence
            read.query_qualities = pysam.qualitystring_to_array('I' * len(sequence))
            read.flag = 0
            read.reference_id = 0
            read.reference_start = ref_start
            read.mapping_quality = 60
            read.cigar = [(0, len(sequence))]
            read.set_tag(
                'MA',
                '200;nuc+QQQ:11-50;msp+:61-30;tf+QQQ:101-20',
                value_type='Z',
            )
            read.set_tag('AQ', array('B', [210, 11, 12, 190, 21, 22]))
            # One m6A, one m5C, and one dU event, respectively.
            read.set_tag('MM', 'A+a,0;C+m,0;C+u,1;', value_type='Z')
            read.set_tag('ML', array('B', [200, 210, 220]))
            if phased:
                read.set_tag('HP', 1, value_type='i')
                read.set_tag('PS', 12345, value_type='i')
            out.write(read)
    pysam.index(str(path))


def _extract_all(bam_path, output_dir, *, block_scores, circular_groups,
                 haplotype_fields):
    output_beds = {
        kind: str(output_dir / f'{kind}.bed') for kind in EXTRACT_TYPES
    }
    total_reads, counts = extract_tags_parallel(
        input_bam=str(bam_path),
        output_beds=output_beds,
        extract_types=list(EXTRACT_TYPES),
        n_cores=1,
        region_size=10_000,
        min_mapq=0,
        prob_threshold=0,
        with_scores=True,
        min_tq=0,
        block_scores=block_scores,
        circular_groups=circular_groups,
        haplotype_fields=haplotype_fields,
    )
    return total_reads, counts, output_beds


def test_haplotype_columns_have_independent_unambiguous_missing_values():
    read = _FakeRead()
    assert _haplotype_columns(read, False) == ()
    assert _haplotype_columns(read, True) == (
        MISSING_HAPLOTYPE_VALUE,
        MISSING_HAPLOTYPE_VALUE,
    )

    read.set_tag('HP', 2)
    assert _haplotype_columns(read, True) == (2, -1)
    read.set_tag('PS', 0)
    assert _haplotype_columns(read, True) == (2, 0)

    read.set_tag('HP', 'not-an-integer')
    assert _haplotype_columns(read, True) == (-1, 0)

    # Numeric-looking strings and floats are not integer-typed BAM tags and
    # must not be silently parsed or truncated.
    read.set_tag('HP', '2')
    read.set_tag('PS', 4.5)
    assert _haplotype_columns(read, True) == (-1, -1)

    # Values that cannot be represented by the signed autoSQL int columns are
    # independently mapped to the missing sentinel.
    read.set_tag('HP', 2 ** 31)
    read.set_tag('PS', -(2 ** 31) - 1)
    assert _haplotype_columns(read, True) == (-1, -1)

    read.set_tag('HP', 2 ** 31 - 1)
    read.set_tag('PS', -(2 ** 31))
    assert _haplotype_columns(read, True) == (2 ** 31 - 1, -(2 ** 31))


def test_autosql_appends_haplotype_fields_after_other_optional_fields(tmp_path):
    schema = get_schema(
        'tf', block_scores=True, circular_groups=True,
        haplotype_fields=True,
    )
    assert HAPLOTYPE_FIELD_COUNT == 2
    assert schema.index('blockTq') < schema.index('circId')
    assert schema.index('circId') < schema.index('int hp;')
    assert schema.index('int hp;') < schema.index('int ps;')
    assert schema.count('-1 if absent or not an integer') == 2

    path = write_autosql_for(
        'tf', out_dir=str(tmp_path), block_scores=True,
        circular_groups=True, haplotype_fields=True,
    )
    assert path.endswith('fiberhmm_tf.bs.circ.hap.as')
    assert 'int hp;' in open(path).read()


@pytest.mark.parametrize('block_scores', [False, True])
@pytest.mark.parametrize('circular_groups', [False, True])
def test_hp_ps_are_last_with_score_and_circular_schema_combinations(
    block_scores, circular_groups
):
    read = _tf_read(phased=True)
    bed = io.StringIO()
    assert _extract_tfs(
        read,
        bed,
        with_scores=True,
        min_tq=0,
        query_to_ref=_identity_map(),
        block_scores=block_scores,
        circular_groups=circular_groups,
        haplotype_fields=True,
    ) == 1
    columns = bed.getvalue().rstrip().split('\t')
    expected_count = 12 + (3 if block_scores else 0) + (
        5 if circular_groups else 0
    ) + 2
    assert len(columns) == expected_count
    assert columns[-2:] == ['2', '4242']


def test_default_output_is_unchanged_even_when_source_read_is_phased():
    read = _tf_read(phased=True)
    implicit_default = io.StringIO()
    explicit_off = io.StringIO()
    kwargs = dict(
        read=read,
        with_scores=True,
        min_tq=0,
        query_to_ref=_identity_map(),
        block_scores=True,
        circular_groups=True,
    )
    _extract_tfs(bed_out=implicit_default, **kwargs)
    _extract_tfs(bed_out=explicit_off, haplotype_fields=False, **kwargs)
    assert implicit_default.getvalue() == explicit_off.getvalue()
    assert len(implicit_default.getvalue().rstrip().split('\t')) == 20


def test_all_extract_types_round_trip_phased_and_unphased_bam(tmp_path):
    bam_path = tmp_path / 'phasing.bam'
    _write_phased_and_unphased_bam(bam_path)

    total_reads, counts, output_beds = _extract_all(
        bam_path,
        tmp_path,
        block_scores=True,
        circular_groups=True,
        haplotype_fields=True,
    )
    assert total_reads == 2
    assert counts == {kind: 2 for kind in EXTRACT_TYPES}

    expected_columns = {
        'nucleosome': 22,  # BED12 + QQQ + circular + HP/PS
        'msp': 20,         # BED12 + AQ + circular + HP/PS
        'tf': 22,          # BED12 + QQQ + circular + HP/PS
        'm6a': 15,         # BED12 + ML + HP/PS
        'm5c': 15,         # BED12 + ML + HP/PS
        'deam': 15,        # BED12 + blockMod + HP/PS
    }
    for kind, bed_path in output_beds.items():
        rows = [line.rstrip().split('\t') for line in open(bed_path) if line.strip()]
        assert [row[3].split('|', 1)[0] for row in rows] == ['phased', 'unphased']
        assert all(len(row) == expected_columns[kind] for row in rows)
        assert rows[0][-2:] == ['1', '12345']
        assert rows[1][-2:] == ['-1', '-1']


@pytest.mark.skipif(
    shutil.which('bedToBigBed') is None,
    reason='UCSC bedToBigBed is not installed',
)
def test_bigbed_embeds_haplotype_schema_and_preserves_rows(tmp_path):
    pybigwig = pytest.importorskip('pyBigWig')
    bam_path = tmp_path / 'phasing.bam'
    _write_phased_and_unphased_bam(bam_path)
    _, _, output_beds = _extract_all(
        bam_path,
        tmp_path,
        block_scores=True,
        circular_groups=True,
        haplotype_fields=True,
    )

    bed_path = output_beds['tf']
    bb_path = str(tmp_path / 'tf.bb')
    assert bed_to_bigbed(
        bed_path,
        bb_path,
        {'chr1': 10_000},
        extract_type='tf',
        block_scores=True,
        sample_name='phasing_test',
        circular_groups=True,
        haplotype_fields=True,
    )

    source_rows = [line.rstrip().split('\t') for line in open(bed_path)]
    with pybigwig.open(bb_path) as bigbed:
        schema = bigbed.SQL()
        if isinstance(schema, bytes):
            schema = schema.decode('utf-8')
        assert 'int hp;' in schema
        assert 'int ps;' in schema
        assert schema.index('circId') < schema.index('int hp;')
        entries = bigbed.entries('chr1', 0, 10_000)

    round_trip_rows = [
        ['chr1', str(start), str(end), *payload.split('\t')]
        for start, end, payload in entries
    ]
    assert round_trip_rows == source_rows
    assert round_trip_rows[0][-2:] == ['1', '12345']
    assert round_trip_rows[1][-2:] == ['-1', '-1']
