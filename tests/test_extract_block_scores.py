"""Tests for the --block-scores (BED12+N) path of fiberhmm-extract.

Covers:
  - autoSQL schemas emit the right extra int[blockCount] columns per type
  - each extractor appends comma-separated arrays of length == blockCount
  - BED12-only (default) output is unchanged when the flag is off
  - EXTRA_FIELD_COUNTS matches the number of comma arrays written
"""
from __future__ import annotations

import io
import os
from array import array
from types import SimpleNamespace

import numpy as np
import pytest

from fiberhmm.cli import extract_tags
from fiberhmm.cli.extract_tags import (
    _extract_deam,
    _extract_footprints,
    _extract_msps,
    _extract_tfs,
    _parse_mod_positions_safe,
)
from fiberhmm.io.autosql import (
    AUTOSQL_SCHEMAS,
    CIRCULAR_FIELD_COUNT,
    EXTRA_FIELD_COUNTS,
    _autosql_file_name,
    _autosql_file_suffix,
    _autosql_variant_suffix,
    _canonical_autosql_type,
    _create_autosql_output_path,
    _schema_description,
    _schema_fields,
    get_schema,
    write_autosql_for,
)


class _FakeRead:
    """Minimal pysam-like read backed by an identity query->ref mapping.

    Only implements the tag/identity surface the extractors use. We avoid
    pysam fixtures to keep the test fast and hermetic.
    """
    def __init__(self, read_id='r', ref_name='chr1', is_reverse=False,
                 query_len=2000, ref_start=1_000_000):
        self._tags = {}
        self.query_name = read_id
        self.reference_name = ref_name
        self.is_reverse = is_reverse
        self._query_len = query_len
        self._ref_start = ref_start

    def has_tag(self, t):
        return t in self._tags

    def get_tag(self, t):
        if t not in self._tags:
            raise KeyError(t)
        return self._tags[t]

    def set_tag(self, t, v, value_type=None):
        if v is None:
            self._tags.pop(t, None)
        else:
            self._tags[t] = v


def _identity_map(read):
    """Identity query->ref map offset by ref_start. Matches what
    cigar_to_query_ref would produce for an ungapped alignment.
    """
    return np.arange(read._ref_start, read._ref_start + read._query_len,
                     dtype=np.int64)


class _FakeOutput:
    def __init__(self):
        self.closed = False
        self.writes = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False

    def write(self, text):
        self.writes.append(text)

    def close(self):
        self.closed = True


class _CountingSequence:
    def __init__(self, values):
        self.values = values
        self.accessed = []

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        self.accessed.append(index)
        return self.values[index]


class _NoToListSequence:
    def __init__(self, values):
        self.values = values

    def __iter__(self):
        return iter(self.values)

    def tolist(self):
        raise AssertionError("tolist should not be called")


def test_extract_region_worker_closes_temp_beds_when_partial_open_fails(
    monkeypatch, tmp_path
):
    params = {
        "extract_types": ["nucleosome", "msp"],
        "min_tq": 50,
        "min_mapq": 0,
        "prob_threshold": 0,
        "with_scores": False,
        "block_scores": False,
    }
    monkeypatch.setattr(extract_tags, "_worker_params", params)

    opened = []

    def fake_open(path, mode):
        if opened:
            raise OSError("open failed")
        handle = _FakeOutput()
        opened.append(handle)
        return handle

    monkeypatch.setattr(extract_tags, "open", fake_open, raising=False)

    temp_bed_paths = {
        "nucleosome": str(tmp_path / "nucleosome.bed"),
        "msp": str(tmp_path / "msp.bed"),
    }

    returned_paths, n_reads, n_features = extract_tags._extract_region_worker(
        (("chr1", 0, 100), "input.bam", temp_bed_paths)
    )

    assert returned_paths == temp_bed_paths
    assert n_reads == 0
    assert n_features == {"nucleosome": 0, "msp": 0}
    assert opened[0].closed


def test_extract_region_read_selected_applies_primary_region_and_mapq_filters():
    def read(**overrides):
        attrs = {
            "is_unmapped": False,
            "is_secondary": False,
            "is_supplementary": False,
            "reference_start": 100,
            "mapping_quality": 20,
        }
        attrs.update(overrides)
        return SimpleNamespace(**attrs)

    assert extract_tags._extract_region_read_selected(read(), 100, 200, 20)
    assert not extract_tags._extract_region_read_selected(
        read(reference_start=99), 100, 200, 20,
    )
    assert not extract_tags._extract_region_read_selected(
        read(reference_start=200), 100, 200, 20,
    )
    assert not extract_tags._extract_region_read_selected(
        read(mapping_quality=19), 100, 200, 20,
    )
    assert not extract_tags._extract_region_read_selected(
        read(is_unmapped=True), 100, 200, 20,
    )
    assert not extract_tags._extract_region_read_selected(
        read(is_secondary=True), 100, 200, 20,
    )
    assert not extract_tags._extract_region_read_selected(
        read(is_supplementary=True), 100, 200, 20,
    )


def test_extract_read_types_reuses_query_ref_map(monkeypatch):
    read = _FakeRead()
    read.set_tag('ns', array('I', [100]))
    read.set_tag('nl', array('I', [20]))
    read.set_tag('as', array('I', [200]))
    read.set_tag('al', array('I', [30]))

    built_for = []

    def fake_build_query_to_ref(read_arg):
        built_for.append(read_arg)
        return _identity_map(read_arg)

    monkeypatch.setattr(extract_tags, "_build_query_to_ref", fake_build_query_to_ref)

    bed_outs = {
        'nucleosome': io.StringIO(),
        'msp': io.StringIO(),
    }

    counts = extract_tags._extract_read_types(
        read,
        ['nucleosome', 'msp'],
        bed_outs,
        with_scores=False,
        block_scores=False,
        circular_groups=False,
        min_tq=50,
        prob_threshold=0,
    )

    assert counts == {'nucleosome': 1, 'msp': 1}
    assert built_for == [read]


def test_legacy_interval_blocks_from_tags_flips_and_scores_reverse_reads():
    assert extract_tags._legacy_interval_blocks_from_tags(
        _FakeRead(), 'ns', 'nl', 'nq', np.array([], dtype=np.int64), True,
    ) == []

    read = _FakeRead(is_reverse=True, query_len=200, ref_start=1_000)
    read.query_sequence = "A" * 200
    read.set_tag('ns', array('I', [10]))
    read.set_tag('nl', array('I', [5]))
    read.set_tag('nq', array('B', [180]))

    assert extract_tags._legacy_interval_blocks_from_tags(
        read, 'ns', 'nl', 'nq', _identity_map(read), True,
    ) == [(1185, 1190, 180)]


def test_ma_annotation_tag_inputs_handles_required_and_optional_tags():
    read = _FakeRead()
    assert extract_tags._ma_annotation_tag_inputs(read) is None

    read.set_tag('MA', '100;tf+Q:1-5')
    assert extract_tags._ma_annotation_tag_inputs(read) == (
        '100;tf+Q:1-5', [], [],
    )

    aq = array('B', [200])
    read.set_tag('AQ', aq)
    read.set_tag('AN', 'tf_a')
    assert extract_tags._ma_annotation_tag_inputs(read) == (
        '100;tf+Q:1-5', aq, ['tf_a'],
    )


def test_ma_interval_and_quals_helpers_preserve_genomic_order():
    assert extract_tags._ma_interval_to_seq(10, 5, 100, False) == (10, 5)
    assert extract_tags._ma_interval_to_seq(10, 5, 100, True) == (85, 5)

    assert extract_tags._ma_quals_in_genomic_order([1, 2, 3], False) == [1, 2, 3]
    assert extract_tags._ma_quals_in_genomic_order([1, 2, 3], True) == [1, 3, 2]
    assert extract_tags._ma_quals_in_genomic_order([1, 2], True) == [1, 2]


def test_ma_annotation_record_normalizes_reverse_read_annotation():
    assert extract_tags._ma_annotation_record(
        10,
        5,
        [1, 2, 3],
        "call-a",
        read_length=100,
        is_reverse=True,
    ) == {
        "start": 85,
        "length": 5,
        "quals": [1, 3, 2],
        "name": "call-a",
        "read_length": 100,
    }


def test_ma_annotation_quality_values_pad_triplets_and_ignore_msp_quals():
    assert extract_tags._quality_triplet([10, 20]) == (10, 20, 0)
    assert extract_tags._ma_annotation_quality_values('tf', [10, 20]) == (
        10, (10, 20, 0),
    )
    assert extract_tags._ma_annotation_quality_values('nuc', []) == (
        0, (0, 0, 0),
    )
    assert extract_tags._ma_annotation_quality_values('msp', [99]) == (
        0, (0,),
    )


def test_ma_annotation_min_tq_filter_only_applies_to_tf():
    assert extract_tags._ma_annotation_tq([]) == 0
    assert extract_tags._ma_annotation_tq([51, 1, 2]) == 51
    assert extract_tags._ma_annotation_passes_min_tq('tf', [50], 50)
    assert not extract_tags._ma_annotation_passes_min_tq('tf', [49], 50)
    assert not extract_tags._ma_annotation_passes_min_tq('tf', [], 1)
    assert extract_tags._ma_annotation_passes_min_tq('nuc', [], 255)
    assert extract_tags._ma_annotation_passes_min_tq('msp', [], 255)


def test_ma_block_score_columns_match_annotation_shape():
    triplet_blocks = [
        (100, 110, 5, (5, 6, 7), {}),
        (120, 130, 8, (8, 9, 10), {}),
    ]
    scalar_blocks = [
        (100, 110, 0, (0,), {}),
        (120, 130, 0, (0,), {}),
    ]

    assert extract_tags._ma_block_score_columns('tf', triplet_blocks) == [
        '5,8', '6,9', '7,10',
    ]
    assert extract_tags._ma_block_score_columns('nuc', triplet_blocks) == [
        '5,8', '6,9', '7,10',
    ]
    assert extract_tags._ma_block_score_columns('msp', scalar_blocks) == ['0,0']


def test_ma_annotation_block_builds_ref_block_or_returns_none():
    ann = {'start': 10, 'length': 5, 'quals': [60, 7, 8]}
    block = extract_tags._ma_annotation_block('tf', ann, _identity_map(_FakeRead()), 50)

    assert block == (1_000_010, 1_000_015, 60, (60, 7, 8), ann)
    assert extract_tags._ma_annotation_block(
        'tf', ann, _identity_map(_FakeRead()), 61,
    ) is None
    assert extract_tags._ma_annotation_block('nuc', ann, {}, 255) is None


def test_ma_circular_row_helpers_build_extra_columns_and_names():
    ann = {
        'circ_id': 'call-a',
        'circ_part': 2,
        'circ_parts': 3,
        'mol_start': 90,
        'mol_length': 30,
    }

    assert extract_tags._ma_circular_extra_columns((10, 20, 30), ann, True) == [
        '10', '20', '30', 'call-a', 2, 3, 90, 30,
    ]
    assert extract_tags._ma_circular_extra_columns((10, 20, 30), ann, False) == [
        'call-a', 2, 3, 90, 30,
    ]
    assert extract_tags._ma_circular_row_name('read1', 'tf', ann) == (
        'read1|tf|call-a|2/3'
    )

    ann = {**ann, 'circ_id': '.'}
    assert extract_tags._ma_circular_row_name('read1', 'tf', ann) == 'read1'


def test_circular_annotation_group_helpers_detect_wrapped_named_pieces():
    left = {'name': 'call-a', 'start': 0, 'length': 15}
    right = {'name': 'call-a', 'start': 85, 'length': 15}
    unnamed = {'name': '', 'start': 0, 'length': 15}

    assert extract_tags._annotation_group_key(2, unnamed) == '__single_2'
    assert extract_tags._circular_annotation_groups([right, left, unnamed]) == {
        'call-a': [right, left],
        '__single_2': [unnamed],
    }
    assert extract_tags._sorted_circular_pieces([right, left]) == [left, right]
    assert extract_tags._is_wrapped_circular_group(
        'call-a', [left, right], read_length=100,
    )
    assert extract_tags._wrapped_group_span([left, right], read_length=100) == (
        85,
        30,
    )
    assert not extract_tags._is_wrapped_circular_group(
        '', [left, right], read_length=100,
    )
    assert not extract_tags._is_wrapped_circular_group(
        'call-a', [left], read_length=100,
    )


def test_selected_extract_types_defaults_and_preserves_cli_order():
    def args(**overrides):
        values = {
            'all': False,
            'nucleosome': False,
            'msp': False,
            'tf': False,
            'm6a': False,
            'm5c': False,
            'deam': False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    assert extract_tags._selected_extract_types(args()) == list(
        extract_tags.ALL_EXTRACT_TYPES
    )
    assert extract_tags._selected_extract_types(args(m5c=True, msp=True)) == [
        'msp',
        'm5c',
    ]
    assert extract_tags._selected_extract_types(args(all=True, tf=True)) == list(
        extract_tags.ALL_EXTRACT_TYPES
    )


def test_extract_region_temp_beds_uses_stable_region_and_type_names(tmp_path):
    assert extract_tags._extract_region_temp_beds(
        str(tmp_path),
        7,
        ["nucleosome", "tf"],
    ) == {
        "nucleosome": str(tmp_path / "region_000007_nucleosome.bed"),
        "tf": str(tmp_path / "region_000007_tf.bed"),
    }


def test_extract_region_work_items_preserve_region_order_and_temp_paths(tmp_path):
    regions = [("chr1", 0, 100), ("chr2", 50, 75)]

    work_items = extract_tags._extract_region_work_items(
        regions,
        "input.bam",
        str(tmp_path),
        ["msp", "deam"],
    )

    assert [item[0] for item in work_items] == regions
    assert [item[1] for item in work_items] == ["input.bam", "input.bam"]
    assert work_items[0][2] == {
        "msp": str(tmp_path / "region_000000_msp.bed"),
        "deam": str(tmp_path / "region_000000_deam.bed"),
    }
    assert work_items[1][2] == {
        "msp": str(tmp_path / "region_000001_msp.bed"),
        "deam": str(tmp_path / "region_000001_deam.bed"),
    }


def test_submit_extract_region_futures_maps_region_indices():
    class Executor:
        def __init__(self):
            self.submitted = []

        def submit(self, worker, item):
            future = object()
            self.submitted.append((future, worker, item))
            return future

    executor = Executor()
    work_items = ["region-a", "region-b"]

    futures = extract_tags._submit_extract_region_futures(executor, work_items)

    assert futures == {
        executor.submitted[0][0]: 0,
        executor.submitted[1][0]: 1,
    }
    assert [(worker, item) for _, worker, item in executor.submitted] == [
        (extract_tags._extract_region_worker, "region-a"),
        (extract_tags._extract_region_worker, "region-b"),
    ]


def test_normalize_parallel_extract_args_handles_aliases_and_backcompat_paths():
    output_beds, extract_types = extract_tags._normalize_parallel_extract_args(
        "footprints.bed",
        "footprint",
    )

    assert output_beds == {"nucleosome": "footprints.bed"}
    assert extract_types == ["nucleosome"]

    output_beds, extract_types = extract_tags._normalize_parallel_extract_args(
        {"footprint": "nuc.bed", "tf": "tf.bed"},
        ["footprint", "tf"],
    )

    assert output_beds == {"nucleosome": "nuc.bed", "tf": "tf.bed"}
    assert extract_types == ["nucleosome", "tf"]

    with pytest.raises(ValueError, match="exactly one extract_type"):
        extract_tags._normalize_parallel_extract_args(
            "combined.bed",
            ["nucleosome", "tf"],
        )


def test_bed12_type_flag_formats_extra_column_count():
    assert extract_tags._bed12_type_flag(0) == "-type=bed12"
    assert extract_tags._bed12_type_flag(4) == "-type=bed12+4"


def test_bigbed_chrom_sizes_file_writes_sorted_sizes(tmp_path):
    sizes_file = tmp_path / "input.bed.sizes"

    extract_tags._write_bigbed_chrom_sizes_file(
        {"chr2": 200, "chr1": 100},
        str(sizes_file),
    )

    assert sizes_file.read_text() == "chr1\t100\nchr2\t200\n"


def test_bigbed_command_helpers_count_extra_fields_and_schema_files():
    assert extract_tags._bigbed_extra_field_count(
        "tf", block_scores=True, circular_groups=True,
    ) == EXTRA_FIELD_COUNTS["tf"] + CIRCULAR_FIELD_COUNT
    assert extract_tags._bigbed_extra_field_count(
        "tf", block_scores=False, circular_groups=False,
    ) == 0

    assert extract_tags._bigbed_autosql_file(
        None,
        block_scores=True,
        sample_name=None,
        circular_groups=True,
    ) is None

    as_file = extract_tags._bigbed_autosql_file(
        "tf",
        block_scores=True,
        sample_name="sample-a",
        circular_groups=False,
    )
    assert as_file is not None
    assert os.path.exists(as_file)

    try:
        assert extract_tags._bed_to_bigbed_cmd(
            "calls.bed",
            "calls.bed.sizes",
            "calls.bb",
            "bed12",
            as_file,
            n_extra=EXTRA_FIELD_COUNTS["tf"],
        ) == [
            "bedToBigBed",
            f"-as={as_file}",
            "-type=bed12+3",
            "calls.bed",
            "calls.bed.sizes",
            "calls.bb",
        ]
        assert extract_tags._bed_to_bigbed_cmd(
            "calls.bed",
            "calls.bed.sizes",
            "calls.bb",
            "bed6",
            None,
            n_extra=0,
        ) == ["bedToBigBed", "calls.bed", "calls.bed.sizes", "calls.bb"]
    finally:
        extract_tags._remove_bigbed_autosql_file(as_file)

    assert not os.path.exists(as_file)


# ------------------- autoSQL schemas --------------------------------

def test_autosql_default_is_bed12_only():
    """AUTOSQL_SCHEMAS has the classic BED12 shape (12 columns)."""
    for t, schema in AUTOSQL_SCHEMAS.items():
        # Classic BED12 fields; no per-block score arrays.
        assert 'chromStarts' in schema
        assert 'blockNq' not in schema
        assert 'blockAq' not in schema
        assert 'blockTq' not in schema
        assert 'blockMl' not in schema


def test_autosql_block_scores_adds_per_type_columns():
    foot = get_schema('nucleosome', block_scores=True)
    assert 'blockNq' in foot
    assert 'blockAq' not in foot

    msp = get_schema('msp', block_scores=True)
    assert 'blockAq' in msp
    assert 'blockNq' not in msp

    tf = get_schema('tf', block_scores=True)
    assert 'blockTq' in tf
    assert 'blockEl' in tf
    assert 'blockEr' in tf

    for t in ('m6a', 'm5c'):
        schema = get_schema(t, block_scores=True)
        assert 'blockMl' in schema


def test_autosql_circular_groups_adds_group_columns():
    schema = get_schema('tf', block_scores=True, circular_groups=True)

    assert CIRCULAR_FIELD_COUNT == 5
    assert 'circId' in schema
    assert 'circPart' in schema
    assert 'molLength' in schema
    assert schema.count('int[blockCount]') == 5  # BED12 arrays + tf QQQ arrays


def test_extra_field_counts_match_written_arrays():
    """EXTRA_FIELD_COUNTS must match the number of int[blockCount] fields
    in the block_scores schema -- this is the number passed as
    ``-type=bed12+N`` to bedToBigBed."""
    assert EXTRA_FIELD_COUNTS['nucleosome'] == 3  # blockNq, blockEl, blockEr
    assert EXTRA_FIELD_COUNTS['msp'] == 1
    assert EXTRA_FIELD_COUNTS['tf'] == 3
    assert EXTRA_FIELD_COUNTS['m6a'] == 1
    assert EXTRA_FIELD_COUNTS['m5c'] == 1


def test_footprint_is_deprecated_alias_for_nucleosome():
    # 'footprint' was the type name through 2.13.x; it must still resolve to the
    # nucleosome schema (table fiberhmm_nucleosome), now that the canonical name
    # is 'nucleosome'.
    assert _canonical_autosql_type('footprint') == 'nucleosome'
    assert _canonical_autosql_type('tf') == 'tf'
    assert get_schema('footprint', block_scores=True) == \
           get_schema('nucleosome', block_scores=True)
    assert 'fiberhmm_nucleosome' in get_schema('footprint')

    for t, n in EXTRA_FIELD_COUNTS.items():
        schema = get_schema(t, block_scores=True)
        assert schema.count('int[blockCount]') == 2 + n, (
            f'{t}: expected 2 base arrays + {n} extras, got '
            f'{schema.count("int[blockCount]")}')


def test_write_autosql_for_writes_both_variants(tmp_path):
    assert _autosql_variant_suffix(False, False) == ''
    assert _autosql_variant_suffix(True, False) == '.bs'
    assert _autosql_variant_suffix(False, True) == '.circ'
    assert _autosql_variant_suffix(True, True) == '.bs.circ'

    classic = write_autosql_for('tf', out_dir=str(tmp_path), block_scores=False)
    bs = write_autosql_for('tf', out_dir=str(tmp_path), block_scores=True)
    circ = write_autosql_for(
        'tf', out_dir=str(tmp_path), block_scores=True, circular_groups=True,
    )
    assert classic != bs
    assert circ != bs
    with open(classic) as f:
        assert 'blockTq' not in f.read()
    with open(bs) as f:
        content = f.read()
        assert 'blockTq' in content
        assert 'blockEl' in content
        assert 'blockEr' in content
    with open(circ) as f:
        assert 'circId' in f.read()


def test_autosql_file_name_and_suffix_helpers():
    assert _autosql_file_suffix('') == '.as'
    assert _autosql_file_suffix('.bs') == '.bs.as'
    assert _autosql_file_name('tf', '') == 'fiberhmm_tf.as'
    assert _autosql_file_name('tf', '.bs.circ') == 'fiberhmm_tf.bs.circ.as'


def test_create_autosql_output_path_handles_tempfile_and_output_dir(tmp_path):
    out_dir = tmp_path / 'schemas'
    fixed_path = _create_autosql_output_path(
        'tf', '.bs', '.bs.as', str(out_dir),
    )

    assert fixed_path == str(out_dir / 'fiberhmm_tf.bs.as')
    assert out_dir.is_dir()

    temp_path = _create_autosql_output_path('tf', '.bs', '.bs.as', None)
    try:
        assert os.path.exists(temp_path)
        assert os.path.basename(temp_path).startswith('fiberhmm_tf_')
        assert temp_path.endswith('.bs.as')
    finally:
        os.unlink(temp_path)


def test_schema_description_prepends_sample_marker_when_present():
    assert _schema_description("Track description") == "Track description"
    assert _schema_description("Track description", "sample-a") == (
        "Sample: sample-a. Track description"
    )


def test_schema_fields_add_block_score_and_circular_columns():
    default_fields = _schema_fields()
    assert 'blockTq' not in default_fields
    assert 'circId' not in default_fields

    tf_fields = _schema_fields('tf', block_scores=True)
    assert 'blockTq' in tf_fields
    assert 'blockEl' in tf_fields

    circular_fields = _schema_fields('msp', circular_groups=True)
    assert 'blockAq' not in circular_fields
    assert 'circId' in circular_fields


# ------------------- footprint --------------------------------------

def test_footprint_bed12_when_flag_off():
    read = _FakeRead()
    read.set_tag('ns', array('I', [100, 500, 900]))
    read.set_tag('nl', array('I', [120, 140, 100]))
    read.set_tag('nq', array('B', [180, 220, 160]))

    buf = io.StringIO()
    n = _extract_footprints(read, buf, with_scores=True,
                            query_to_ref=_identity_map(read),
                            block_scores=False)
    assert n == 3
    line = buf.getvalue().rstrip('\n')
    cols = line.split('\t')
    assert len(cols) == 12  # classic BED12
    assert int(cols[9]) == 3   # blockCount
    assert cols[10].count(',') == 2  # blockSizes (3 entries)


def test_footprint_bed12_plus_nq_when_flag_on():
    read = _FakeRead()
    read.set_tag('ns', array('I', [100, 500, 900]))
    read.set_tag('nl', array('I', [120, 140, 100]))
    read.set_tag('nq', array('B', [180, 220, 160]))

    buf = io.StringIO()
    n = _extract_footprints(read, buf, with_scores=False,
                            query_to_ref=_identity_map(read),
                            block_scores=True)
    assert n == 3
    cols = buf.getvalue().rstrip('\n').split('\t')
    assert len(cols) == 15  # BED12 + blockNq + blockEl + blockEr
    block_count = int(cols[9])
    nq_arr = cols[12].split(',')
    assert len(nq_arr) == block_count == 3
    # with_scores=False leaves the mean score column at 0, but per-block nq
    # should still carry real values from the nq tag.
    assert int(cols[4]) == 0
    assert [int(v) for v in nq_arr] == [180, 220, 160]
    # legacy ns/nl nucs have no edge refinement -> el/er all zero
    assert [int(v) for v in cols[13].split(',')] == [0, 0, 0]
    assert [int(v) for v in cols[14].split(',')] == [0, 0, 0]


def test_footprint_missing_nq_when_flag_on_writes_zeros():
    """If the BAM has no nq tag, blockNq array falls back to zeros."""
    read = _FakeRead()
    read.set_tag('ns', array('I', [100, 500]))
    read.set_tag('nl', array('I', [120, 140]))
    # NO nq tag on this read.

    buf = io.StringIO()
    n = _extract_footprints(read, buf, with_scores=False,
                            query_to_ref=_identity_map(read),
                            block_scores=True)
    assert n == 2
    cols = buf.getvalue().rstrip('\n').split('\t')
    assert len(cols) == 15  # BED12 + blockNq + blockEl + blockEr
    assert [int(v) for v in cols[12].split(',')] == [0, 0]
    assert [int(v) for v in cols[13].split(',')] == [0, 0]
    assert [int(v) for v in cols[14].split(',')] == [0, 0]


# ------------------- msp --------------------------------------------

def test_msp_bed12_plus_aq_when_flag_on():
    read = _FakeRead()
    read.set_tag('as', array('I', [200, 700]))
    read.set_tag('al', array('I', [150, 180]))
    read.set_tag('aq', array('B', [90, 255]))

    buf = io.StringIO()
    n = _extract_msps(read, buf, with_scores=True,
                      query_to_ref=_identity_map(read),
                      block_scores=True)
    assert n == 2
    cols = buf.getvalue().rstrip('\n').split('\t')
    assert len(cols) == 13
    assert [int(v) for v in cols[12].split(',')] == [90, 255]


# ------------------- tf (MA/AQ tf+QQQ) ------------------------------

def _build_ma_aq(nuc_intervals, tf_intervals, nq_values, tf_qqqs,
                 read_length=2000):
    """Build MA:Z + AQ:B:C tag values matching the Molecular-annotation spec.

    ``tf_qqqs`` is a list of (tq, el, er) triplets, one per tf interval.
    """
    from fiberhmm.io.ma_tags import format_aq_array, format_ma_tag
    ma = format_ma_tag(read_length=read_length,
                       nuc_intervals=nuc_intervals,
                       msp_intervals=(),
                       tf_intervals=tf_intervals)
    tq_vals = [q[0] for q in tf_qqqs]
    el_vals = [q[1] for q in tf_qqqs]
    er_vals = [q[2] for q in tf_qqqs]
    aq = format_aq_array(nq_values=nq_values,
                         tf_q_values=tq_vals,
                         tf_lq_values=el_vals,
                         tf_rq_values=er_vals)
    return ma, aq


def test_tf_bed12_plus_qqq_when_flag_on():
    read = _FakeRead()
    # 1 nuc + 2 tfs, spec-mode MA + AQ
    ma, aq = _build_ma_aq(
        nuc_intervals=[(50, 120)],
        tf_intervals=[(300, 30), (800, 25)],
        nq_values=[200],
        tf_qqqs=[(150, 30, 25), (220, 10, 5)],
    )
    read.set_tag('MA', ma)
    read.set_tag('AQ', aq)

    buf = io.StringIO()
    n = _extract_tfs(read, buf, with_scores=True, min_tq=0,
                     query_to_ref=_identity_map(read),
                     block_scores=True)
    assert n == 2
    cols = buf.getvalue().rstrip('\n').split('\t')
    assert len(cols) == 15   # 12 + 3 (tq, el, er)

    block_count = int(cols[9])
    tq_arr = [int(v) for v in cols[12].split(',')]
    el_arr = [int(v) for v in cols[13].split(',')]
    er_arr = [int(v) for v in cols[14].split(',')]
    assert len(tq_arr) == len(el_arr) == len(er_arr) == block_count == 2
    # Blocks are sorted by ref start; our two TFs were already in order.
    assert tq_arr == [150, 220]
    assert el_arr == [30, 10]
    assert er_arr == [25, 5]


def test_tf_min_tq_filter_drops_low_quality_and_shrinks_blocks():
    read = _FakeRead()
    ma, aq = _build_ma_aq(
        nuc_intervals=[],
        tf_intervals=[(100, 30), (500, 25)],
        nq_values=[],
        tf_qqqs=[(40, 10, 10), (200, 20, 20)],  # first TF is tq=40
    )
    read.set_tag('MA', ma)
    read.set_tag('AQ', aq)

    buf = io.StringIO()
    n = _extract_tfs(read, buf, with_scores=True, min_tq=50,
                     query_to_ref=_identity_map(read),
                     block_scores=True)
    assert n == 1  # only the tq=200 TF survives
    cols = buf.getvalue().rstrip('\n').split('\t')
    assert len(cols) == 15
    assert int(cols[9]) == 1
    assert [int(v) for v in cols[12].split(',')] == [200]


def test_tf_bed12_only_unchanged_when_flag_off():
    """Sanity: BED12-only output has exactly 12 tab-separated columns."""
    read = _FakeRead()
    ma, aq = _build_ma_aq(
        nuc_intervals=[],
        tf_intervals=[(100, 30), (500, 25)],
        nq_values=[],
        tf_qqqs=[(150, 30, 25), (220, 10, 5)],
    )
    read.set_tag('MA', ma)
    read.set_tag('AQ', aq)

    buf = io.StringIO()
    n = _extract_tfs(read, buf, with_scores=True, min_tq=0,
                     query_to_ref=_identity_map(read),
                     block_scores=False)
    assert n == 2
    cols = buf.getvalue().rstrip('\n').split('\t')
    assert len(cols) == 12


def test_tf_extractor_consumes_aq_without_upfront_list_copy():
    read = _FakeRead()
    ma, _aq = _build_ma_aq(
        nuc_intervals=[],
        tf_intervals=[(100, 30)],
        nq_values=[],
        tf_qqqs=[(150, 30, 25)],
    )
    aq = _CountingSequence([150, 30, 25, 99, 88])
    read.set_tag('MA', ma)
    read.set_tag('AQ', aq)

    buf = io.StringIO()
    n = _extract_tfs(
        read,
        buf,
        with_scores=True,
        min_tq=0,
        query_to_ref=_identity_map(read),
        block_scores=True,
    )

    assert n == 1
    assert aq.accessed == [0, 1, 2]
    assert buf.getvalue().rstrip('\n').split('\t')[12:15] == ["150", "30", "25"]


def test_tf_circular_groups_emit_one_row_per_clipped_piece():
    read = _FakeRead(query_len=1000, ref_start=0)
    read.set_tag('MA', '1000;tf+QQQ:1-45,971-30')
    read.set_tag('AQ', array('B', [180, 20, 30, 180, 20, 30]))
    read.set_tag('AN', 'fhw_tf_0,fhw_tf_0')

    buf = io.StringIO()
    n = _extract_tfs(
        read,
        buf,
        with_scores=True,
        min_tq=0,
        query_to_ref=_identity_map(read),
        block_scores=True,
        circular_groups=True,
    )

    assert n == 2
    rows = [line.split('\t') for line in buf.getvalue().rstrip('\n').splitlines()]
    assert len(rows) == 2
    assert [int(row[1]) for row in rows] == [0, 970]
    assert all(len(row) == 20 for row in rows)  # BED12 + tf QQQ + circular fields
    assert rows[0][15:] == ['fhw_tf_0', '1', '2', '970', '75']
    assert rows[1][15:] == ['fhw_tf_0', '2', '2', '970', '75']


def test_footprint_and_msp_extractors_prefer_ma_an_when_present():
    read = _FakeRead(query_len=1000, ref_start=0)
    read.set_tag('MA', '1000;nuc+Q:1-10,991-10;msp+:1-20,981-20')
    read.set_tag('AQ', array('B', [222, 222]))
    read.set_tag('AN', 'fhw_nuc_0,fhw_nuc_0,fhw_msp_0,fhw_msp_0')
    # Stale legacy tags should not win when MA is available.
    read.set_tag('ns', array('I', [100]))
    read.set_tag('nl', array('I', [10]))
    read.set_tag('as', array('I', [200]))
    read.set_tag('al', array('I', [10]))

    nuc_buf = io.StringIO()
    msp_buf = io.StringIO()

    assert _extract_footprints(
        read, nuc_buf, with_scores=True, query_to_ref=_identity_map(read),
        block_scores=True, circular_groups=True,
    ) == 2
    assert _extract_msps(
        read, msp_buf, with_scores=True, query_to_ref=_identity_map(read),
        block_scores=True, circular_groups=True,
    ) == 2

    nuc_rows = [line.split('\t') for line in nuc_buf.getvalue().rstrip('\n').splitlines()]
    msp_rows = [line.split('\t') for line in msp_buf.getvalue().rstrip('\n').splitlines()]
    assert [int(row[1]) for row in nuc_rows] == [0, 990]
    assert [int(row[1]) for row in msp_rows] == [0, 980]
    # nucleosome now carries blockNq, blockEl, blockEr (nuc.Q -> el/er = 0),
    # so the circular group columns start after those three. (Input MA uses the
    # legacy '+' strand to prove backward-compatible parsing.)
    assert nuc_rows[0][12] == '222'
    assert nuc_rows[0][13:15] == ['0', '0']
    assert nuc_rows[0][15:] == ['fhw_nuc_0', '1', '2', '990', '20']
    assert msp_rows[0][13:] == ['fhw_msp_0', '1', '2', '980', '40']


# ------------------- deam (DAF IUPAC R/Y) ---------------------------

class _FakeReadWithSeq(_FakeRead):
    """_FakeRead with a mutable query_sequence attribute so we can test
    the R/Y scan against known IUPAC-encoded strings."""
    def __init__(self, seq, **kw):
        super().__init__(**kw)
        self.query_sequence = seq
        self._query_len = len(seq)


def test_deam_iupac_positions_maps_flavors_and_skips_unmapped():
    q2r = np.array([100, -1, 102, 103, 104])

    assert extract_tags._sort_position_values_by_ref([(104, 0), (100, 1)]) == [
        (100, 1),
        (104, 0),
    ]
    assert extract_tags._deam_ref_pos_for_query(q2r, 0) == 100
    assert extract_tags._deam_ref_pos_for_query(q2r, 1) is None
    assert extract_tags._deam_ref_pos_for_query(q2r, -1) is None
    assert extract_tags._deam_ref_pos_for_query(q2r, len(q2r)) is None

    assert extract_tags._deam_iupac_positions("RYACR", q2r) == [
        (100, 0),
        (104, 0),
    ]
    assert extract_tags._deam_iupac_positions("AAYC", q2r) == [(102, 1)]
    assert extract_tags._deam_iupac_positions("", q2r) == []


def test_deam_extracts_both_codes_and_sorts_by_ref_position():
    """A read with interleaved R and Y should produce one BED row per
    read, with each R/Y as a 1 bp block sorted by reference position."""
    # query: ACGYRTAYR  -> R/Y at positions 3,4,7,8
    seq = 'ACGYRTAYR'
    read = _FakeReadWithSeq(seq, ref_start=1000)

    buf = io.StringIO()
    n = _extract_deam(read, buf, query_to_ref=_identity_map(read),
                    block_scores=False)
    assert n == 4
    cols = buf.getvalue().rstrip('\n').split('\t')
    assert len(cols) == 12
    # chromStart = first mod position = 1003 ; chromEnd = last + 1 = 1009
    assert int(cols[1]) == 1003
    assert int(cols[2]) == 1009
    # score is the 255 sentinel (not probability)
    assert int(cols[4]) == 255
    assert int(cols[9]) == 4  # blockCount
    assert cols[10] == '1,1,1,1'
    assert cols[11] == '0,1,4,5'  # offsets relative to chromStart=1003


def test_deam_block_scores_disambiguates_r_vs_y():
    """With block_scores=True, the blockMod column must encode
    0 for R (GA-strand) and 1 for Y (CT-strand)."""
    seq = 'ACGYYRRYN'   # R/Y at 3,4,5,6,7 -> Y,Y,R,R,Y
    read = _FakeReadWithSeq(seq, ref_start=500)

    buf = io.StringIO()
    n = _extract_deam(read, buf, query_to_ref=_identity_map(read),
                    block_scores=True)
    assert n == 5
    cols = buf.getvalue().rstrip('\n').split('\t')
    assert len(cols) == 13
    assert [int(v) for v in cols[12].split(',')] == [1, 1, 0, 0, 1]


def test_deam_empty_sequence_returns_zero():
    read = _FakeReadWithSeq('ACGT' * 50, ref_start=1_000_000)
    buf = io.StringIO()
    n = _extract_deam(read, buf, query_to_ref=_identity_map(read),
                    block_scores=True)
    assert n == 0
    assert buf.getvalue() == ''


def test_deam_skips_positions_with_no_ref_mapping():
    """Insertion bases (query position with no ref mapping) must not
    produce BED rows with negative or None coordinates."""
    seq = 'ACYGR'  # Y at 2, R at 4
    read = _FakeReadWithSeq(seq, ref_start=2000)
    # Force query position 2 (the Y) to be an insertion (-1).
    q2r = _identity_map(read).copy()
    q2r[2] = -1
    buf = io.StringIO()
    n = _extract_deam(read, buf, query_to_ref=q2r, block_scores=True)
    # Only the R at q=4 should survive.
    assert n == 1
    cols = buf.getvalue().rstrip('\n').split('\t')
    assert int(cols[9]) == 1
    assert cols[12] == '0'  # code 0 = R


def test_deam_schema_has_blockmod_with_block_scores():
    schema = get_schema('deam', block_scores=True)
    # Check for the field declaration specifically, not a substring in the
    # description (which references the field name in prose).
    assert 'int[blockCount] blockMod' in schema
    classic = get_schema('deam', block_scores=False)
    assert 'int[blockCount] blockMod' not in classic


def test_deam_extra_field_count_is_one():
    assert EXTRA_FIELD_COUNTS['deam'] == 1


# ------------------- deam MM/ML-native path (priority 1) -----------

class _FakeReadWithModBases(_FakeReadWithSeq):
    """FakeRead that exposes a canned modified_bases dict so the
    MM/ML-native code path in _extract_deam can be exercised without
    constructing a real pysam MM/ML string.

    Dict shape matches pysam.AlignedSegment.modified_bases:
      {(canonical_base, strand_code, mod_type): [(query_pos, prob), ...]}
    """
    def __init__(self, seq, modified_bases, **kw):
        super().__init__(seq, **kw)
        self.modified_bases = modified_bases


# v2.10.5 note: the deam priority-1 MM/ML path now uses our safe MM/ML
# parser (parse_mm_ml_per_mod_type) instead of pysam.modified_bases, which
# segfaults on unusual MM/ML layouts (surjected fiber-seq, etc.). Tests
# below construct real MM/ML tag strings instead of fake modified_bases
# dicts. The ChEBI 55797 numeric mod-code test was intentionally removed
# because the safe parser is single-char only; users needing numeric
# encodings should run through the R/Y or MD fallbacks.

def _read_with_mm_ml(seq, mm_tag, ml_bytes, ref_start=0, is_reverse=False):
    """Build a _FakeReadWithSeq that exposes real MM/ML tags."""
    read = _FakeReadWithSeq(seq=seq, ref_start=ref_start, is_reverse=is_reverse)
    read.set_tag('MM', mm_tag, value_type='Z')
    # ML is a bytes-like array of per-call probability bytes.
    import array as pyarr
    read.set_tag('ML', pyarr.array('B', ml_bytes))
    return read


def test_deam_mm_ml_positions_maps_flavor_and_threshold():
    seq = list("A" * 30)
    seq[10] = "C"
    seq[20] = "C"
    read = _read_with_mm_ml(
        seq="".join(seq),
        mm_tag="C+u,0,0;",
        ml_bytes=[100, 200],
        ref_start=500,
    )

    assert extract_tags._deam_mm_ml_positions(
        read, _identity_map(read), prob_threshold=125,
    ) == [(520, 1)]

    q2r = np.array([100, -1, 102])
    assert extract_tags._deam_mm_ml_ref_position(q2r, 0, 200, 1, 125) == (100, 1)
    assert extract_tags._deam_mm_ml_ref_position(q2r, 0, 100, 1, 125) is None
    assert extract_tags._deam_mm_ml_ref_position(q2r, 1, 200, 1, 125) is None


def test_deam_base_flavor_maps_supported_canonical_bases():
    assert extract_tags._deam_base_flavor('C') == 1
    assert extract_tags._deam_base_flavor('g') == 0
    assert extract_tags._deam_base_flavor(ord('C')) == 1
    assert extract_tags._deam_base_flavor(ord('G')) == 0
    assert extract_tags._deam_base_flavor('A') is None
    assert extract_tags._deam_md_mismatch_flavor('C', 'T') == 1
    assert extract_tags._deam_md_mismatch_flavor('G', 'A') == 0
    assert extract_tags._deam_md_mismatch_flavor('C', 'A') is None


def test_deam_priority1_mm_ml_u_code_wins_over_ry():
    """If MM/ML carries dU 'u' calls, they must win over R/Y in the
    sequence -- matches FiberBrowser's first-non-empty-source rule."""
    # Sequence has R/Y at q=10,20 (should be ignored), and C's at q=100,200
    # which MM/ML marks as dU.
    seq = list('A' * 300)
    seq[10] = 'R'
    seq[20] = 'Y'
    seq[100] = 'C'
    seq[200] = 'C'
    # MM: "C+u,0,0;" = 0 skip to first C (q=100), then 0 skip to next (q=200)
    # ML: two probability bytes (200, 150).
    read = _read_with_mm_ml(
        seq=''.join(seq),
        mm_tag='C+u,0,0;',
        ml_bytes=[200, 150],
        ref_start=5000,
    )

    buf = io.StringIO()
    n = _extract_deam(read, buf, query_to_ref=_identity_map(read),
                      block_scores=True)
    assert n == 2
    cols = buf.getvalue().rstrip('\n').split('\t')
    assert int(cols[9]) == 2
    assert [int(v) for v in cols[12].split(',')] == [1, 1]


def test_parse_mod_positions_safe_avoids_numpy_tolist(monkeypatch):
    def fake_parse_mm_ml_per_mod_type(mm_tag, ml_bytes, seq, is_reverse):
        return {
            ("A", "a"): (
                _NoToListSequence([2, 5]),
                _NoToListSequence([200, 180]),
            )
        }

    monkeypatch.setattr(
        "fiberhmm.core.bam_reader.parse_mm_ml_per_mod_type",
        fake_parse_mm_ml_per_mod_type,
    )

    read = _read_with_mm_ml(
        seq="A" * 20,
        mm_tag="A+a,0,0;",
        ml_bytes=[200, 180],
    )

    per_mod = extract_tags._safe_mm_ml_per_mod(read)
    pos_arr, qual_arr = per_mod[("A", "a")]
    assert list(pos_arr) == [2, 5]
    assert list(qual_arr) == [200, 180]
    assert _parse_mod_positions_safe(read, {"a"}) == [(2, 200), (5, 180)]


def test_deam_priority1_g_base_is_flavor_0():
    """Canonical base G -> flavor 0 (R / GA-dea) per FiberBrowser."""
    seq = list('A' * 100)
    seq[50] = 'G'
    read = _read_with_mm_ml(
        seq=''.join(seq),
        mm_tag='G+u,0;',
        ml_bytes=[200],
        ref_start=1000,
    )
    buf = io.StringIO()
    n = _extract_deam(read, buf, query_to_ref=_identity_map(read),
                      block_scores=True)
    assert n == 1
    cols = buf.getvalue().rstrip('\n').split('\t')
    assert [int(v) for v in cols[12].split(',')] == [0]


def test_deam_priority1_respects_prob_threshold():
    """prob_threshold filters MM/ML dU calls below the cutoff."""
    # Two C's in the sequence at q=10 and q=20, with ML = [100, 200].
    # At threshold 125, only the q=20 entry should survive.
    seq = list('A' * 100)
    seq[10] = 'C'
    seq[20] = 'C'
    read = _read_with_mm_ml(
        seq=''.join(seq),
        mm_tag='C+u,0,0;',
        ml_bytes=[100, 200],
        ref_start=100,
    )
    buf = io.StringIO()
    n = _extract_deam(read, buf, query_to_ref=_identity_map(read),
                      block_scores=True, prob_threshold=125)
    assert n == 1
    cols = buf.getvalue().rstrip('\n').split('\t')
    assert int(cols[1]) == 120   # chromStart = ref of q=20 = ref_start(100) + 20


class _FakeReadWithMD(_FakeReadWithSeq):
    """FakeRead with get_aligned_pairs(with_seq=True) returning a canned
    list, letting us exercise the path-3 MD mismatch branch of
    _extract_deam without building a real BAM."""
    def __init__(self, seq, pairs, has_md=True, **kw):
        super().__init__(seq, **kw)
        self._pairs = pairs
        self._has_md = has_md

    def has_tag(self, t):
        if t == 'MD' and self._has_md:
            return True
        return super().has_tag(t)

    def get_aligned_pairs(self, with_seq=False):
        return self._pairs if with_seq else [(p[0], p[1]) for p in self._pairs]


def test_deam_md_mismatch_positions_collects_deamination_flavors():
    read_seq = 'ATAGTACG'
    ref_for_pairs = 'ACGGTACG'
    pairs = [(i, 100 + i, ref_for_pairs[i]) for i in range(len(read_seq))]
    read = _FakeReadWithMD(seq=read_seq, pairs=pairs, ref_start=100)

    assert extract_tags._deam_md_mismatch_positions(read) == [
        (101, 1),
        (102, 0),
    ]


def test_deam_priority3_md_mismatch_extracts_c_to_t_and_g_to_a():
    """When MM/ML has no dU and the sequence has no R/Y, path-3 falls
    back to walking get_aligned_pairs(with_seq=True) and picks out
    deamination-direction mismatches."""
    # Ref sequence: A C G T A C G T ; Read: A T G A A T G A
    # Mismatches: pos 1 C->T (Y, flavor 1), pos 3 T->A (not dea),
    #             pos 6 G->G (match), pos 7 T->A (not dea).
    # Actually make it cleaner: pos 1 C->T, pos 2 G->A, pos 5 C->T.
    read_seq = 'ATAGTACG'
    ref_for_pairs = 'ACGGTACG'
    # ref_start=100; pairs: (qpos, rpos, ref_base)
    pairs = [(i, 100 + i, ref_for_pairs[i]) for i in range(len(read_seq))]
    read = _FakeReadWithMD(seq=read_seq, pairs=pairs, ref_start=100)

    buf = io.StringIO()
    n = _extract_deam(read, buf, query_to_ref=_identity_map(read),
                      block_scores=True)
    # Mismatches: q=1 C->T (flavor 1), q=2 G->A (flavor 0). Other
    # positions are either matches or non-dea mismatches.
    assert n == 2
    cols = buf.getvalue().rstrip('\n').split('\t')
    flavors = [int(v) for v in cols[12].split(',')]
    assert flavors == [1, 0]


def test_deam_positions_by_priority_stops_at_first_nonempty_source(monkeypatch):
    read = SimpleNamespace(query_sequence="RY")
    aligned_pairs = np.array([10, 11])
    calls = []

    monkeypatch.setattr(
        extract_tags,
        "_deam_mm_ml_positions",
        lambda *_: calls.append("mm") or [(1, 0)],
    )
    monkeypatch.setattr(
        extract_tags,
        "_deam_iupac_positions",
        lambda *_: calls.append("iupac") or [(2, 1)],
    )
    monkeypatch.setattr(
        extract_tags,
        "_deam_md_mismatch_positions",
        lambda *_: calls.append("md") or [(3, 0)],
    )

    assert extract_tags._deam_positions_by_priority(
        read, aligned_pairs, 125,
    ) == [(1, 0)]
    assert calls == ["mm"]

    calls.clear()
    monkeypatch.setattr(
        extract_tags,
        "_deam_mm_ml_positions",
        lambda *_: calls.append("mm") or [],
    )
    assert extract_tags._deam_positions_by_priority(
        read, aligned_pairs, 125,
    ) == [(2, 1)]
    assert calls == ["mm", "iupac"]

    calls.clear()
    monkeypatch.setattr(
        extract_tags,
        "_deam_iupac_positions",
        lambda *_: calls.append("iupac") or [],
    )
    assert extract_tags._deam_positions_by_priority(
        read, aligned_pairs, 125,
    ) == [(3, 0)]
    assert calls == ["mm", "iupac", "md"]


def test_looks_like_fiber_seq_detects_hia5_bams():
    """Regression for Christy LaFlamme's crash on surjected fiber-seq
    CRAMs: --deam should auto-skip on fiber-seq BAMs in --all mode to
    avoid triggering the path-3 MD walker (which can segfault workers
    on malformed MD from vg surject and take m6a/m5c writes with it).
    """
    from fiberhmm.cli.extract_tags import _looks_like_fiber_seq

    # Classic PacBio fiber-seq CRAM: A+a + C+m, no u code, no R/Y.
    fiber_diag = {
        'mm_subtypes': ['A+a', 'C+m', 'T-a'],
        'has_ry_in_seq': 0,
    }
    assert _looks_like_fiber_seq(fiber_diag) is True

    # SAM 1.7 suffixes on subtypes should still be recognized.
    fiber_diag_17 = {
        'mm_subtypes': ['A+a.', 'C+m?', 'T-a.'],
        'has_ry_in_seq': 0,
    }
    assert _looks_like_fiber_seq(fiber_diag_17) is True

    # DAF with IUPAC encoding -> NOT fiber-seq (run --deam normally).
    daf_iupac = {
        'mm_subtypes': [],
        'has_ry_in_seq': 50,
    }
    assert _looks_like_fiber_seq(daf_iupac) is False

    # DAF with modkit-style u code -> NOT fiber-seq.
    daf_u = {
        'mm_subtypes': ['C+u'],
        'has_ry_in_seq': 0,
    }
    assert _looks_like_fiber_seq(daf_u) is False

    # DAF with 55797 numeric code -> NOT fiber-seq.
    daf_chebi = {
        'mm_subtypes': ['C+55797'],
        'has_ry_in_seq': 0,
    }
    assert _looks_like_fiber_seq(daf_chebi) is False

    # Empty BAM (no tags detected at all) -> NOT fiber-seq (can't
    # safely assume). Let the caller decide whether to run --deam.
    assert _looks_like_fiber_seq({'mm_subtypes': [], 'has_ry_in_seq': 0}) is False

    # Mixed (fiber-seq + deaminase on same read somehow): has u code ->
    # NOT fiber-seq (run --deam because the deaminase signal is real).
    mixed = {
        'mm_subtypes': ['A+a', 'C+m', 'C+u'],
        'has_ry_in_seq': 0,
    }
    assert _looks_like_fiber_seq(mixed) is False


def test_md_tag_ref_length_parser():
    """Parser should correctly sum matches, mismatches, and deletions."""
    from fiberhmm.daf.encoder import _md_tag_ref_length
    # 10 matches
    assert _md_tag_ref_length("10") == 10
    # mismatch in the middle: 3 match + C + 2 match = 6 ref bases
    assert _md_tag_ref_length("3C2") == 6
    # deletion: 5 match + ^ACG (3-base deletion) + 2 match = 10 ref bases
    assert _md_tag_ref_length("5^ACG2") == 10
    # combined
    assert _md_tag_ref_length("5A3^T10G2") == 5 + 1 + 3 + 1 + 10 + 1 + 2


def test_md_matches_cigar_catches_mismatch():
    """Regression for Christy LaFlamme's malloc crash: pre-validation
    should return False for the BAM she reported, which had an MD tag
    whose ref length (37729) disagreed with the CIGAR's
    reference-consuming op total (43799 - 9742 insertions = 34057).
    """
    from fiberhmm.daf.encoder import md_matches_cigar

    class _FakeReadMD:
        def __init__(self, md, cigartuples):
            self._md = md
            self.cigartuples = cigartuples

        def has_tag(self, t):
            return t == 'MD'

        def get_tag(self, t):
            if t == 'MD':
                return self._md
            raise KeyError(t)

    # Matching: CIGAR ref_len = 10 (10M), MD = 10 matches.
    r_ok = _FakeReadMD("10", [(0, 10)])
    assert md_matches_cigar(r_ok) is True

    # Mismatching: CIGAR says 20 ref bases, MD only accounts for 10.
    r_bad = _FakeReadMD("10", [(0, 20)])
    assert md_matches_cigar(r_bad) is False

    # No MD tag at all -> treat as "nothing to validate", caller falls back.
    class _NoMD:
        cigartuples = [(0, 10)]
        def has_tag(self, t): return False
        def get_tag(self, t): raise KeyError(t)
    assert md_matches_cigar(_NoMD()) is True


def test_deam_priority3_skips_malformed_md_without_calling_pysam():
    """Regression for Christy's malloc crash: when MD length disagrees
    with CIGAR, _extract_deam must skip path-3 without calling
    get_aligned_pairs(with_seq=True) at all -- that call corrupts
    pysam's internal state and crashes the worker later.
    """
    class _ReadBadMD(_FakeReadWithMD):
        # MD says 5 ref bases; CIGAR says 100. Real mismatch.
        cigartuples = [(0, 100)]   # 100M
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._bad_md = "5"
            self._get_aligned_pairs_called = False
        def has_tag(self, t):
            return t == 'MD' or super().has_tag(t)
        def get_tag(self, t):
            if t == 'MD':
                return self._bad_md
            return super().get_tag(t)
        def get_aligned_pairs(self, with_seq=False):
            # If this fires, pre-validation failed us.
            self._get_aligned_pairs_called = True
            raise AssertionError("should not be reached on malformed MD")

    read = _ReadBadMD(seq='A' * 100, pairs=[], has_md=True, ref_start=0)
    buf = io.StringIO()
    n = _extract_deam(read, buf, query_to_ref=_identity_map(read),
                      block_scores=True)
    assert n == 0
    assert buf.getvalue() == ''
    assert read._get_aligned_pairs_called is False, (
        "pre-validation should have skipped the call; it fired -> bug"
    )


def test_deam_priority3_skips_read_on_pysam_assertion_error():
    """pysam raises AssertionError (not ValueError) when the MD tag length
    disagrees with the CIGAR — seen in the wild on malformed BAMs. The
    path-3 walker must swallow it and skip the read, not crash the worker.
    Regression for Christy LaFlamme's report:
      AssertionError: Invalid MD tag: MD length 37729 mismatch with CIGAR
      length 43799 and 9742 insertions
    """
    class _ReadWithBrokenMD(_FakeReadWithMD):
        def get_aligned_pairs(self, with_seq=False):
            if with_seq:
                raise AssertionError(
                    "Invalid MD tag: MD length 37729 mismatch with "
                    "CIGAR length 43799 and 9742 insertions"
                )
            return []

    read = _ReadWithBrokenMD(seq='ACGT' * 10, pairs=[], has_md=True,
                              ref_start=1_000)
    buf = io.StringIO()
    # Must not raise; just return 0 (read skipped).
    n = _extract_deam(read, buf, query_to_ref=_identity_map(read),
                      block_scores=True)
    assert n == 0
    assert buf.getvalue() == ''


def test_deam_priority3_skipped_when_no_md_tag():
    read = _FakeReadWithMD(seq='ATCG', pairs=[], has_md=False, ref_start=0)
    buf = io.StringIO()
    n = _extract_deam(read, buf, query_to_ref=_identity_map(read),
                      block_scores=True)
    assert n == 0  # no MD -> no path-3 fallback


def test_deam_priority3_ignored_when_priority2_populates():
    """If R/Y are present in the sequence, path-3 must never be consulted
    (priority ordering: MM/ML > R/Y > MD)."""
    read_seq = 'ACYGT'   # Y at q=2 -> path 2 finds it
    pairs = [(i, i, 'ACGGT'[i]) for i in range(5)]  # would also produce a C->Y mismatch
    read = _FakeReadWithMD(seq=read_seq, pairs=pairs, ref_start=0)
    buf = io.StringIO()
    n = _extract_deam(read, buf, query_to_ref=_identity_map(read),
                      block_scores=True)
    # Only the R/Y path should run (1 call, flavor 1 for Y), not the MD path.
    assert n == 1
    cols = buf.getvalue().rstrip('\n').split('\t')
    assert [int(v) for v in cols[12].split(',')] == [1]


def test_deam_priority2_ry_fallback_when_mm_ml_absent():
    """When MM/ML has no 'u'/55797 entries, we must fall back to the
    R/Y sequence scan -- the existing v2.9.3/4 behavior."""
    read = _FakeReadWithModBases(
        seq='ACRYT',
        modified_bases={
            ('A', 0, 'a'): [(0, 200)],  # m6A present but no dU; should be ignored
        },
        ref_start=0,
    )
    buf = io.StringIO()
    n = _extract_deam(read, buf, query_to_ref=_identity_map(read),
                      block_scores=True)
    # R at q=2, Y at q=3 -> 2 blocks, flavors [0, 1].
    assert n == 2
    cols = buf.getvalue().rstrip('\n').split('\t')
    assert [int(v) for v in cols[12].split(',')] == [0, 1]


# ------------------- sample_name in autoSQL --------------------------

def test_sample_name_prepended_to_description():
    schema = get_schema('nucleosome', block_scores=False,
                        sample_name='Dl_recalled')
    assert 'Sample: Dl_recalled. ' in schema
    # The schema still has the classic description afterwards.
    assert 'FiberHMM nucleosome calls' in schema


def test_sample_name_absent_by_default():
    schema = get_schema('footprint', block_scores=False)
    assert 'Sample:' not in schema


def test_sample_name_with_block_scores():
    schema = get_schema('deam', block_scores=True, sample_name='yw_2-4')
    assert 'Sample: yw_2-4. ' in schema
    # extra block score column still present
    assert 'int[blockCount] blockMod' in schema


def test_write_autosql_for_with_sample_name(tmp_path):
    p = write_autosql_for('tf', out_dir=str(tmp_path),
                          block_scores=True, sample_name='Dl_recalled')
    with open(p) as f:
        content = f.read()
    assert 'Sample: Dl_recalled. ' in content
    assert 'int[blockCount] blockTq' in content


def _write_circular_ma_an_bam(path, *, read_length=1000, ref_start=100_000):
    """Write a one-read indexed BAM whose MA/AN tags describe a wrapped TF call
    plus a normal nucleosome. Used as input to extract_tags_parallel."""
    import pysam

    header = pysam.AlignmentHeader.from_dict({
        'HD': {'VN': '1.6', 'SO': 'coordinate'},
        'SQ': [{'SN': 'chr1', 'LN': max(1_000_000, ref_start + read_length + 10)}],
    })
    unsorted = str(path) + '.unsorted'
    with pysam.AlignmentFile(unsorted, 'wb', header=header) as out:
        read = pysam.AlignedSegment()
        read.query_name = 'circ_read'
        read.query_sequence = 'A' * read_length
        read.query_qualities = pysam.qualitystring_to_array('I' * read_length)
        read.flag = 0
        read.reference_id = 0
        read.reference_start = ref_start
        read.mapping_quality = 60
        read.cigar = [(0, read_length)]
        # tf call wraps the origin: pieces at [0, 15) and [985, 1000). Same AN
        # name. Standalone nuc at [200, 250). MA prefix is the read length.
        read.set_tag('MA',
                     f'{read_length};nuc+Q:201-50;tf+QQQ:1-15,986-15',
                     value_type='Z')
        read.set_tag('AN', 'fh_nuc_0,fhw_tf_0,fhw_tf_0', value_type='Z')
        # AQ: nuc Q (1 value), then 3-tuple for each tf piece.
        read.set_tag('AQ', array('B', [200, 180, 20, 30, 180, 20, 30]))
        out.write(read)
    pysam.sort('-o', str(path), unsorted)
    import os as _os
    _os.remove(unsorted)
    pysam.index(str(path))


def test_extract_tags_parallel_circular_groups_end_to_end(tmp_path):
    """Drive extract_tags_parallel(circular_groups=True) end-to-end on a BAM
    with hand-crafted wrapped MA/AN tags. Assert BED12+5 rows are emitted
    with the right circular grouping metadata for both pieces."""
    bam_path = tmp_path / 'circular.bam'
    _write_circular_ma_an_bam(bam_path, read_length=1000, ref_start=100_000)

    nuc_bed = tmp_path / 'nuc.bed'
    tf_bed = tmp_path / 'tf.bed'

    # Pass the deprecated 'footprint' type on purpose: extract_tags_parallel must
    # normalize it (type + output_beds key) to the canonical 'nucleosome'.
    total_reads, n_features = extract_tags.extract_tags_parallel(
        input_bam=str(bam_path),
        output_beds={'footprint': str(nuc_bed), 'tf': str(tf_bed)},
        extract_types=['footprint', 'tf'],
        n_cores=1,
        region_size=10_000_000,
        min_mapq=0,
        prob_threshold=0,
        with_scores=True,
        min_tq=0,
        block_scores=True,
        circular_groups=True,
    )

    assert total_reads == 1
    assert n_features['nucleosome'] == 1   # 'footprint' alias normalized
    assert n_features['tf'] == 2

    nuc_rows = [line.split('\t') for line in nuc_bed.read_text().splitlines()]
    tf_rows = [line.split('\t') for line in tf_bed.read_text().splitlines()]

    # BED12 (12) + nucleosome block_scores (3: blockNq/El/Er) + circular (5) = 20.
    assert all(len(row) == 20 for row in nuc_rows)
    # BED12 (12) + tf block_scores (3 blockTq/El/Er) + circular (5) = 20 cols.
    assert all(len(row) == 20 for row in tf_rows)

    # Standalone (non-wrapped) nuc: blockNq=200, blockEl/Er=0 (nuc.Q, unrefined),
    # then circId=., circPart/Parts=1, molStart/molLength echo its coords.
    assert nuc_rows[0][12:15] == ['200', '0', '0']
    assert nuc_rows[0][15:] == ['.', '1', '1', '200', '50']

    # Wrapped TF: both clipped pieces share AN name, have circParts=2, and
    # their molStart/molLength describe the molecular feature (not the clipped
    # ref interval). Pieces are ordered by ref start.
    tf_rows.sort(key=lambda r: int(r[1]))
    assert [int(r[1]) for r in tf_rows] == [100_000, 100_985]
    assert [int(r[2]) for r in tf_rows] == [100_015, 101_000]
    for row in tf_rows:
        assert row[15] == 'fhw_tf_0'
        assert row[17] == '2'  # circParts
        # molStart is the start of the end-piece (the one furthest right on
        # the molecule); molLength is the fused circular feature length.
        assert int(row[18]) == 985
        assert int(row[19]) == 30
    # circPart numbers cover both pieces.
    assert sorted(int(r[16]) for r in tf_rows) == [1, 2]
    # Names encode the wrapped grouping.
    assert all('|' in row[3] and 'fhw_tf_0' in row[3] for row in tf_rows)
