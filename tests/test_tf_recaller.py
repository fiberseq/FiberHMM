"""Tests for the LLR TF recaller (fiberhmm.inference.tf_recaller +
fiberhmm.io.ma_tags).
"""
import array

import numpy as np
import pytest

from fiberhmm.inference import tf_recaller
from fiberhmm.inference.tf_recaller import (
    ENZYME_PRESETS,
    N_CTX,
    UNMETH_OFFSET,
    TFCall,
    build_scan_intervals,
    call_tfs_in_interval,
    merge_intervals,
    write_ma_tags,
)
from fiberhmm.io.ma_tags import (
    EDGE_AMBIGUITY_SAT,
    ambiguity_to_edge,
    format_aq_array,
    format_ma_tag,
    llr_to_tq,
    parse_aq_array,
    parse_ma_tag,
    tq_to_llr,
)

# ------------------- ma_tags ---------------------------------------

def test_llr_to_tq_roundtrip():
    for llr in [0, 1, 5, 10, 25.5, 100]:
        tq = llr_to_tq(llr)
        assert 0 <= tq <= 255
    assert llr_to_tq(0) == 0
    assert llr_to_tq(5) == 50
    assert llr_to_tq(100) == 255  # saturated
    assert tq_to_llr(255) == 25.5


def test_ambiguity_to_edge():
    assert ambiguity_to_edge(0) == 255
    assert ambiguity_to_edge(EDGE_AMBIGUITY_SAT) == 0
    assert ambiguity_to_edge(EDGE_AMBIGUITY_SAT + 5) == 0
    assert 0 < ambiguity_to_edge(15) < 255


def test_split_legacy_interval_rows_splits_sorts_and_clamps_scores():
    rows = tf_recaller._split_legacy_interval_rows(
        [(90, 20), (10, 5)],
        read_length=100,
        scores=[300, -5],
    )

    assert rows == [(0, 10, 255), (10, 5, 0), (90, 10, 255)]
    assert tf_recaller._legacy_starts_lengths(rows) == ([0, 10, 90], [10, 5, 10])


def test_quality_row_helpers_preserve_aq_layout():
    assert tf_recaller._recall_nuc_quality_inputs(
        [(0, 100), (200, 50)], None, None, None,
    ) == ([0, 0], False, [], [])
    assert tf_recaller._recall_nuc_quality_inputs(
        [(0, 100)], [7], [8], [9],
    ) == ([7], True, [8], [9])

    with pytest.raises(ValueError, match="nq_values"):
        tf_recaller._recall_nuc_quality_inputs([(0, 100)], [1, 2], None, None)

    with pytest.raises(ValueError, match="nuc edge arrays"):
        tf_recaller._recall_nuc_quality_inputs([(0, 100)], [1], [2, 3], [4])

    assert tf_recaller._nuc_quality_rows([7, 8], nuc_qqq=False) == [[7], [8]]
    assert tf_recaller._nuc_quality_rows(
        [7], nuc_qqq=True, nuc_el_values=[8], nuc_er_values=[9],
    ) == [[7, 8, 9]]
    assert tf_recaller._tf_quality_rows([10], [11], [12]) == [[10, 11, 12]]

    default_aq = tf_recaller._format_split_aq(
        [[7]],
        [[10, 11, 12]],
        nuc_qqq=False,
    )
    nuc_qqq_aq = tf_recaller._format_split_aq(
        [[7, 8, 9]],
        [[10, 11, 12]],
        nuc_qqq=True,
    )

    assert list(default_aq) == [7, 10, 11, 12]
    assert list(nuc_qqq_aq) == [7, 8, 9, 10, 11, 12]


def test_ma_tag_matches_spec_regex():
    """Every MA string we emit must match the official spec regex
    (https://github.com/fiberseq/Molecular-annotation-spec)."""
    import re
    SPEC_REGEX = re.compile(
        r'^\d+;(([a-zA-Z0-9_]+)[+-.][PQ]*:((\d+-\d+)(,\d+-\d+)*);?)+$'
    )
    cases = [
        # (description, kwargs)
        ('typical', dict(read_length=4521,
                         nuc_intervals=[(43, 147), (200, 100)],
                         msp_intervals=[(1, 42)],
                         tf_intervals=[(50, 20), (300, 15)])),
        ('only nucs', dict(read_length=1000,
                           nuc_intervals=[(0, 100)],
                           msp_intervals=[],
                           tf_intervals=[])),
        ('only msps', dict(read_length=1000,
                           nuc_intervals=[],
                           msp_intervals=[(50, 100)],
                           tf_intervals=[])),
        ('only tfs', dict(read_length=1000,
                          nuc_intervals=[],
                          msp_intervals=[],
                          tf_intervals=[(0, 30)])),
    ]
    for desc, kw in cases:
        ma = format_ma_tag(**kw)
        assert SPEC_REGEX.match(ma), f'{desc}: MA={ma!r} fails spec regex'


def test_emits_unknown_strand_and_reads_legacy_plus():
    # We now emit the '.' (unknown) strand for nuc/msp/tf, matching fibertools.
    ma = format_ma_tag(read_length=1000,
                       nuc_intervals=[(0, 100)],
                       msp_intervals=[(200, 50)],
                       tf_intervals=[(400, 20)])
    assert ma == '1000;nuc.Q:1-100;msp.:201-50;tf.QQQ:401-20'
    # Tags written by FiberHMM <= 2.13.1 used '+'; both must parse identically.
    legacy = '1000;nuc+Q:1-100;msp+:201-50;tf+QQQ:401-20'
    p_new, p_old = parse_ma_tag(ma), parse_ma_tag(legacy)
    for key in ('read_length', 'nuc', 'msp', 'tf'):
        assert p_new[key] == p_old[key]
    # strand field differs ('.' vs '+'); qual_spec + intervals are identical.
    assert [(n, q, iv) for n, _s, q, iv in p_new['raw_types']] == \
           [(n, q, iv) for n, _s, q, iv in p_old['raw_types']]
    assert [s for _n, s, _q, _iv in p_new['raw_types']] == ['.', '.', '.']


def test_format_and_parse_ma_tag():
    nucs = [(43, 147), (216, 155)]
    msps = [(1, 42)]
    tfs = [(50, 20)]
    ma = format_ma_tag(read_length=4521,
                       nuc_intervals=nucs,
                       msp_intervals=msps,
                       tf_intervals=tfs)
    assert ma.startswith('4521;')
    assert 'nuc.Q:44-147,217-155' in ma
    assert 'msp.:2-42' in ma
    assert 'tf.QQQ:51-20' in ma
    parsed = parse_ma_tag(ma)
    assert parsed['read_length'] == 4521
    assert parsed['nuc'] == nucs
    assert parsed['msp'] == msps
    assert parsed['tf'] == tfs


def test_aq_layout():
    nq = [200, 180]
    tq = [45]
    el = [180]
    er = [220]
    aq = format_aq_array(nq, tq, el, er)
    # 2 bytes (nucs) + 3 bytes (tf) = 5 bytes total
    assert list(aq) == [200, 180, 45, 180, 220]
    parsed = parse_aq_array(aq, ['Q', '', 'QQQ'], [2, 0, 1])
    assert parsed == [[200], [180], [45, 180, 220]]


class _CountingAq:
    def __init__(self, values):
        self.values = values
        self.accessed = []

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        self.accessed.append(index)
        return self.values[index]


def test_parse_aq_array_reads_only_consumed_quality_bytes():
    aq = _CountingAq([200, 180, 45, 180, 220, 99, 88, 77])

    parsed = parse_aq_array(aq, ['Q', '', 'QQQ'], [1, 2, 1])

    assert parsed == [[200], [], [], [180, 45, 180]]
    assert aq.accessed == [0, 1, 2, 3]


def test_parse_aq_array_preserves_short_quality_arrays():
    assert parse_aq_array([7], ['QQQ'], [1]) == [[7]]


# ------------------- merge_intervals -------------------------------

def test_merge_intervals():
    assert merge_intervals([]) == []
    assert merge_intervals([(0, 10), (5, 15)]) == [(0, 15)]
    assert merge_intervals([(0, 10), (10, 20)]) == [(0, 20)]  # touching = merge
    assert merge_intervals([(0, 10), (20, 30)]) == [(0, 10), (20, 30)]


def test_build_scan_intervals_includes_short_nucs():
    # MSPs at [0, 100), short nuc at [200, 240) (nl=40), big nuc at [400, 600)
    iv = build_scan_intervals(
        ns=[200, 400], nl=[40, 200],
        as_=[0], al=[100],
        read_len=1000, unify_threshold=90,
    )
    # Should include MSP + short nuc, NOT the big nuc
    assert (0, 100) in iv
    assert (200, 240) in iv
    assert all(start != 400 for start, _ in iv)


# ------------------- Kadane scan -----------------------------------

def _make_obs(seq):
    """Helper: convert a string of chars (h=hit@ctx0, m=miss@ctx0, .=non-target)
    to an obs np.ndarray."""
    arr = []
    for c in seq:
        if c == 'h':
            arr.append(0)
        elif c == 'm':
            arr.append(UNMETH_OFFSET)
        else:
            arr.append(N_CTX)  # non-target
    return np.array(arr, dtype=np.int32)


def test_kadane_clean_miss_run():
    llr_hit = np.full(N_CTX, -2.0)
    llr_miss = np.full(N_CTX, 0.5)
    obs = _make_obs('hhmmmmmmmmmmmhh')  # 5 hits, 11 misses... no wait
    obs = _make_obs('hhmmmmmmmmmmhh')   # 2 hits, 10 misses, 2 hits
    calls = call_tfs_in_interval(obs, 0, len(obs), llr_hit, llr_miss,
                                  min_llr=3.0, min_opps=3)
    assert len(calls) == 1
    c = calls[0]
    assert c.start == 2
    assert c.length == 10
    assert c.llr == pytest.approx(5.0)
    assert c.left_ambiguity == 0      # hit immediately at left
    assert c.right_ambiguity == 0     # hit immediately at right


def test_kadane_absorbs_one_rogue_hit():
    """A 20-bp miss run with 1 hit in the middle should remain a single call,
    because Kadane allows the running LLR to dip without resetting if there's
    enough surrounding positive evidence."""
    llr_hit = np.full(N_CTX, -2.0)
    llr_miss = np.full(N_CTX, 0.5)
    # 5 hits + 10 misses + 1 hit + 10 misses + 5 hits
    obs = _make_obs('h' * 5 + 'm' * 10 + 'h' + 'm' * 10 + 'h' * 5)
    calls = call_tfs_in_interval(obs, 0, len(obs), llr_hit, llr_miss,
                                  min_llr=3.0, min_opps=3)
    assert len(calls) == 1
    c = calls[0]
    # Call should span both miss runs + the rogue hit
    assert c.length >= 20
    # LLR = 20 * 0.5 - 1 * 2.0 = 8.0
    assert c.llr == pytest.approx(8.0)


def test_kadane_subthreshold_rejected():
    llr_hit = np.full(N_CTX, -2.0)
    llr_miss = np.full(N_CTX, 0.5)
    # Only 4 misses -> LLR = 2 < 3
    obs = _make_obs('hhmmmmhh')
    calls = call_tfs_in_interval(obs, 0, len(obs), llr_hit, llr_miss,
                                  min_llr=3.0, min_opps=3)
    assert calls == []


def test_kadane_ambiguous_edges():
    """Non-target padding between miss run and hit -> ambiguity > 0."""
    llr_hit = np.full(N_CTX, -2.0)
    llr_miss = np.full(N_CTX, 0.5)
    # hit + 5 non-target + 10 misses + 5 non-target + hit
    obs = _make_obs('h' + '.' * 5 + 'm' * 10 + '.' * 5 + 'h')
    calls = call_tfs_in_interval(obs, 0, len(obs), llr_hit, llr_miss,
                                  min_llr=3.0, min_opps=3)
    assert len(calls) == 1
    c = calls[0]
    assert c.left_ambiguity == 5
    assert c.right_ambiguity == 5


# ------------------- enzyme presets --------------------------------

# ------------------- write_ma_tags downstream-compat mode ---------

class _FakeRead:
    """Minimal pysam-like stand-in for unit-testing write_ma_tags."""
    def __init__(self):
        self._tags = {}
        self.query_sequence = 'A' * 200

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


def test_spec_mode_emits_ma_aq_and_clean_legacy():
    read = _FakeRead()
    nucs = [(50, 120)]
    msps = [(180, 20)]
    tfs = [TFCall(start=10, length=30, llr=7.5, n_opps=8,
                  left_ambiguity=2, right_ambiguity=0)]
    write_ma_tags(read, 200, tfs, nucs, msps, nq_for_kept_nucs=[200],
                  also_write_legacy=True, downstream_compat=False)
    # MA + AQ present
    assert read.has_tag('MA')
    assert read.has_tag('AQ')
    ma = read.get_tag('MA')
    assert 'nuc.' in ma and 'tf.' in ma
    # Legacy ns/nl has only nucs, NOT TFs (spec mode)
    ns = list(read.get_tag('ns'))
    nl = list(read.get_tag('nl'))
    assert ns == [50]
    assert nl == [120]


def test_downstream_compat_merges_tfs_into_ns_nl():
    read = _FakeRead()
    nucs = [(50, 120)]
    msps = [(180, 20)]
    tfs = [
        TFCall(start=10, length=30, llr=7.5, n_opps=8,
               left_ambiguity=2, right_ambiguity=0),
        TFCall(start=300, length=15, llr=6.0, n_opps=5,
               left_ambiguity=1, right_ambiguity=3),
    ]
    write_ma_tags(read, 500, tfs, nucs, msps, nq_for_kept_nucs=[200],
                  also_write_legacy=True, downstream_compat=True)
    # MA + AQ must NOT be present in compat mode
    assert not read.has_tag('MA')
    assert not read.has_tag('AQ')
    # Legacy ns/nl contains nuc + both TFs, sorted by start
    ns = list(read.get_tag('ns'))
    nl = list(read.get_tag('nl'))
    assert ns == [10, 50, 300]
    assert nl == [30, 120, 15]


def test_compat_mode_strips_stale_ma_aq():
    read = _FakeRead()
    # Simulate a BAM that already had MA/AQ from a prior spec-mode run
    read.set_tag('MA', 'stale content', value_type='Z')
    read.set_tag('AQ', [1, 2, 3])
    write_ma_tags(read, 200, tf_calls=[], kept_nucs=[(0, 100)], msps=[],
                  nq_for_kept_nucs=[128],
                  also_write_legacy=True, downstream_compat=True)
    assert not read.has_tag('MA')
    assert not read.has_tag('AQ')


def test_spec_skips_ma_when_no_annotations():
    """fiberseq MA spec requires >=1 annotation in the MA string;
    we must not write MA when nothing was called."""
    read = _FakeRead()
    write_ma_tags(read, 200, tf_calls=[], kept_nucs=[], msps=[],
                  also_write_legacy=True, downstream_compat=False)
    assert not read.has_tag('MA')
    assert not read.has_tag('AQ')


def test_spec_skips_aq_when_only_unqualified_types():
    """spec: AQ is only present if any annotation type specifies P or Q.
    msp. has no quality; if only MSPs are emitted, AQ must not appear."""
    read = _FakeRead()
    write_ma_tags(read, 200, tf_calls=[], kept_nucs=[],
                  msps=[(10, 80)],
                  also_write_legacy=True, downstream_compat=False)
    assert read.has_tag('MA')
    ma = read.get_tag('MA')
    assert 'msp.' in ma
    assert 'nuc.' not in ma and 'tf.' not in ma
    assert not read.has_tag('AQ')


def test_spec_strips_stale_ma_aq_when_no_annotations():
    """If a prior run left MA/AQ on a read that now has no calls,
    we should clear those stale tags."""
    read = _FakeRead()
    read.set_tag('MA', 'stale', value_type='Z')
    read.set_tag('AQ', [1, 2, 3])
    write_ma_tags(read, 200, tf_calls=[], kept_nucs=[], msps=[],
                  also_write_legacy=True, downstream_compat=False)
    assert not read.has_tag('MA')
    assert not read.has_tag('AQ')


def test_spec_mode_splits_wrapped_annotations_and_writes_an():
    read = _FakeRead()
    tfs = [TFCall(start=80, length=30, llr=5.0, n_opps=5,
                  left_ambiguity=1, right_ambiguity=2)]

    write_ma_tags(
        read,
        100,
        tf_calls=tfs,
        kept_nucs=[(90, 20)],
        msps=[(95, 10)],
        nq_for_kept_nucs=[201],
        also_write_legacy=True,
        downstream_compat=False,
    )

    assert read.get_tag('MA') == (
        '100;nuc.Q:1-10,91-10;msp.:1-5,96-5;tf.QQQ:1-10,81-20'
    )
    assert read.get_tag('AN') == (
        'fhw_nuc_0,fhw_nuc_0,fhw_msp_0,fhw_msp_0,fhw_tf_0,fhw_tf_0'
    )
    # AQ duplicates the nuc quality and TF QQQ triplet for the two clipped pieces.
    assert list(read.get_tag('AQ')) == [201, 201, 50, 246, 238, 50, 246, 238]
    assert list(read.get_tag('ns')) == [0, 90]
    assert list(read.get_tag('nl')) == [10, 10]
    assert list(read.get_tag('nq')) == [201, 201]


def test_write_ma_tags_reverse_read_flips_intervals_and_edges():
    read = _FakeRead()
    read.is_reverse = True
    tfs = [TFCall(start=60, length=10, llr=5.0, n_opps=5,
                  left_ambiguity=3, right_ambiguity=6)]

    write_ma_tags(
        read,
        100,
        tf_calls=tfs,
        kept_nucs=[(10, 20)],
        msps=[(35, 15)],
        nq_for_kept_nucs=[200],
        nuc_el_for_kept=[240],
        nuc_er_for_kept=[120],
        also_write_legacy=True,
        downstream_compat=False,
    )

    assert read.get_tag('MA') == '100;nuc.QQQ:71-20;msp.:51-15;tf.QQQ:31-10'
    assert list(read.get_tag('AQ')) == [
        200, 120, 240,
        50, ambiguity_to_edge(6), ambiguity_to_edge(3),
    ]
    assert list(read.get_tag('ns')) == [70]
    assert list(read.get_tag('nl')) == [20]
    assert list(read.get_tag('nq')) == [200]
    assert list(read.get_tag('as')) == [50]
    assert list(read.get_tag('al')) == [15]


def test_spec_mode_strips_stale_an_when_next_read_is_linear():
    read = _FakeRead()
    read.set_tag('AN', 'stale', value_type='Z')

    write_ma_tags(
        read,
        100,
        tf_calls=[],
        kept_nucs=[(10, 20)],
        msps=[],
        nq_for_kept_nucs=[99],
        also_write_legacy=True,
        downstream_compat=False,
    )

    assert not read.has_tag('AN')


def test_compat_requires_legacy():
    import pytest
    read = _FakeRead()
    with pytest.raises(ValueError):
        write_ma_tags(read, 200, tf_calls=[], kept_nucs=[(0, 100)], msps=[],
                      also_write_legacy=False, downstream_compat=True)


def test_extract_modifications_keeps_raw_ml_container(monkeypatch):
    import array as pyarray

    read = _FakeRead()
    read.query_sequence = 'A' * 20
    read.is_reverse = False
    raw_ml = pyarray.array('B', [255])
    read.set_tag('MM', 'A+a,0;')
    read.set_tag('ML', raw_ml)
    captured = {}

    def fake_parse(mm_tag, ml_tag, sequence, is_reverse, prob_threshold, mode):
        captured['ml_tag'] = ml_tag
        return {0}

    monkeypatch.setattr(tf_recaller, 'parse_mm_tag_query_positions', fake_parse)

    assert tf_recaller.extract_modifications(read, 'pacbio-fiber', 3) == (
        {0}, '.', read.query_sequence,
    )
    assert captured['ml_tag'] is raw_ml


def test_extract_modifications_handles_daf_iupac_branch():
    read = _FakeRead()
    read.query_sequence = 'ACYR'
    read.set_tag('st', 'CT')

    assert tf_recaller.extract_modifications(read, 'daf', context_size=1) == (
        {2, 3}, '+', 'ACTA',
    )


def test_stale_nq_aq_cleared_when_refreshing_legacy_tags():
    """Regression for ft-validate panic reported by Shane Neph:
    input BAM pre-tagged by ft/modkit with ns+nl+nq (and as+al+aq) at
    one length; fiberhmm-call overwrites ns+nl to a different length
    without fresh scores. The stale nq/aq would then mismatch the new
    ns/as length and trigger fibertools-rs
      assert_eq!(forward_qual.len(), self.annotations.len())
    at bamannotations.rs:28. We must drop stale nq/aq whenever we refresh
    legacy tags without providing new scores.
    """
    import array as pyarray
    read = _FakeRead()
    # Pre-existing ft/modkit tags: 84 nucs + 84 msps, all quality-scored.
    read.set_tag('ns', pyarray.array('I', list(range(0, 840, 10))))
    read.set_tag('nl', pyarray.array('I', [8] * 84))
    read.set_tag('nq', pyarray.array('B', [200] * 84))
    read.set_tag('as', pyarray.array('I', list(range(0, 840, 10))))
    read.set_tag('al', pyarray.array('I', [8] * 84))
    read.set_tag('aq', pyarray.array('B', [150] * 84))
    assert len(list(read.get_tag('nq'))) == 84
    assert len(list(read.get_tag('aq'))) == 84

    # Refresh with 65 nucs + 65 msps, no scores passed.
    new_nucs = [(i * 10, 90) for i in range(65)]
    new_msps = [(i * 10 + 5, 85) for i in range(65)]
    write_ma_tags(read, 2000, tf_calls=[], kept_nucs=new_nucs, msps=new_msps,
                  nq_for_kept_nucs=None,
                  also_write_legacy=True, downstream_compat=False)

    # New ns/nl/as/al at length 65.
    assert len(list(read.get_tag('ns'))) == 65
    assert len(list(read.get_tag('nl'))) == 65
    assert len(list(read.get_tag('as'))) == 65
    assert len(list(read.get_tag('al'))) == 65
    # Stale nq + aq MUST be cleared to satisfy fibertools invariant
    # len(nq) == len(ns), len(aq) == len(as).
    assert not read.has_tag('nq'), "stale nq must be deleted when refreshing ns without scores"
    assert not read.has_tag('aq'), "stale aq must be deleted when refreshing as without scores"


def test_stale_aq_cleared_when_nucs_empty_but_msps_refresh():
    """If the new call produces MSPs but no nucs, stale aq from the input
    must still be cleared."""
    import array as pyarray
    read = _FakeRead()
    read.set_tag('as', pyarray.array('I', list(range(0, 840, 10))))
    read.set_tag('al', pyarray.array('I', [8] * 84))
    read.set_tag('aq', pyarray.array('B', [150] * 84))

    write_ma_tags(read, 2000, tf_calls=[], kept_nucs=[],
                  msps=[(0, 100), (200, 150)],
                  nq_for_kept_nucs=None,
                  also_write_legacy=True, downstream_compat=False)

    assert len(list(read.get_tag('as'))) == 2
    assert len(list(read.get_tag('al'))) == 2
    assert not read.has_tag('aq')


def test_fresh_nq_preserved_when_scores_provided():
    """Sanity: when caller DOES provide fresh nq, it gets written at the
    new length — we must not accidentally delete it."""
    import array as pyarray
    read = _FakeRead()
    read.set_tag('ns', pyarray.array('I', [0, 100]))
    read.set_tag('nl', pyarray.array('I', [50, 50]))
    read.set_tag('nq', pyarray.array('B', [200, 200]))

    write_ma_tags(read, 500, tf_calls=[],
                  kept_nucs=[(10, 80), (200, 80), (400, 80)],
                  msps=[],
                  nq_for_kept_nucs=[128, 255, 0],
                  also_write_legacy=True, downstream_compat=False)

    assert list(read.get_tag('ns')) == [10, 200, 400]
    assert list(read.get_tag('nq')) == [128, 255, 0]


def test_enzyme_presets_present():
    for enz in ('hia5', 'dddb', 'ddda'):
        assert enz in ENZYME_PRESETS
        for k in ('min_llr', 'emission_uplift'):
            assert k in ENZYME_PRESETS[enz]
    # All presets use uplift=1.0; DddA gets a pre-uplifted model file
    # (ddda_TF.json) rather than a runtime power transform.
    for enz in ('hia5', 'dddb', 'ddda'):
        assert ENZYME_PRESETS[enz]['emission_uplift'] == 1.0
    # DddB uses lower min_llr than Hia5 (single-strand evidence)
    assert ENZYME_PRESETS['dddb']['min_llr'] < ENZYME_PRESETS['hia5']['min_llr']


def test_recall_read_accepts_compact_array_tag_sequences_without_modifications():
    class FakeRead:
        query_sequence = "A" * 300
        is_reverse = False
        _tags = {
            "ns": array.array("I", [10, 200]),
            "nl": array.array("I", [80, 120]),
            "as": array.array("I", [0]),
            "al": array.array("I", [50]),
        }

        def has_tag(self, tag):
            return tag in self._tags

        def get_tag(self, tag):
            if tag not in self._tags:
                raise KeyError(tag)
            return self._tags[tag]

    tf_calls, kept_nucs, msps = tf_recaller.recall_read(
        FakeRead(),
        np.zeros(N_CTX),
        np.zeros(N_CTX),
        mode="pacbio-fiber",
        context_size=3,
        min_llr=5.0,
        min_opps=3,
        unify_threshold=90,
    )

    assert tf_calls == []
    assert kept_nucs == [(10, 80), (200, 120)]
    assert msps == [(0, 50)]
