"""Tests for fiberhmm-dedup (DAF-seq deamination-fingerprint PCR dedup)."""
from __future__ import annotations

import pysam
import pytest

from fiberhmm.cli.dedup import cluster_reads, run_dedup

# Two molecules, 200 bp reads, deamination encoded as IUPAC 'Y' in the query
# (priority-2 source in _deam_positions_list). Molecule A = even ref positions
# 0..98 (50 sites); molecule B = even positions 100..198 (50 sites).
A_SITES = list(range(0, 100, 2))
B_SITES = list(range(100, 200, 2))


def _seq_with_y(sites):
    s = ['A'] * 200
    for p in sites:
        s[p] = 'Y'
    return ''.join(s)


def _make_bam(path, reads):
    """reads: list of (name, sites, is_reverse, mapq). Writes a 200M-CIGAR BAM."""
    header = pysam.AlignmentHeader.from_dict({
        'HD': {'VN': '1.6', 'SO': 'coordinate'},
        'SQ': [{'SN': 'chr1', 'LN': 1000}],
    })
    with pysam.AlignmentFile(str(path), 'wb', header=header) as out:
        for name, sites, is_reverse, mapq in reads:
            a = pysam.AlignedSegment(header)
            a.query_name = name
            a.flag = 16 if is_reverse else 0
            a.reference_id = 0
            a.reference_start = 0
            a.mapping_quality = mapq
            a.cigarstring = '200M'
            a.query_sequence = _seq_with_y(sites)
            a.query_qualities = pysam.qualitystring_to_array('I' * 200)
            out.write(a)


def _near(sites, drop_add):
    """Copy of `sites` with `drop_add` positions swapped (1 swap -> Jaccard 49/51=0.96)."""
    s = list(sites)
    for k in range(drop_add):
        s[k] = s[k] + 1  # shift one site by 1 bp -> symmetric-diff of 2
    return s


def test_cluster_separates_molecules_and_tolerates_near_copies():
    # 4 exact A + 1 near-A (J=0.96) -> one cluster; 3 B -> another.
    pos_sets = [frozenset(A_SITES)] * 4 + [frozenset(_near(A_SITES, 1))] \
        + [frozenset(B_SITES)] * 3
    keys = [('c', '+')] * len(pos_sets)
    labels = cluster_reads(pos_sets, keys, min_jaccard=0.95, k=32, bands=8, seed=7)
    # 5 A-reads share a label, 3 B-reads share another, 2 labels total.
    assert len(set(labels[:5])) == 1
    assert len(set(labels[5:])) == 1
    assert labels[0] != labels[5]
    assert len(set(labels.tolist())) == 2


def test_strand_grouping():
    pos_sets = [frozenset(A_SITES), frozenset(A_SITES)]
    # Same pattern, opposite strands: not duplicates by default...
    diff = cluster_reads(pos_sets, [('c', '+'), ('c', '-')], 0.95, 32, 8, 7)
    assert diff[0] != diff[1]
    # ...but merge when strand is ignored.
    same = cluster_reads(pos_sets, [('c', ''), ('c', '')], 0.95, 32, 8, 7)
    assert same[0] == same[1]


@pytest.fixture
def daf_bam(tmp_path):
    reads = (
        [(f'A{i}', A_SITES, False, 60) for i in range(4)]
        + [('Anear', _near(A_SITES, 1), False, 60)]
        + [(f'B{i}', B_SITES, False, 60) for i in range(3)]
        + [('lowdeam', [2, 4, 6], False, 60)]  # 3 sites < min-deam -> passthrough
    )
    p = tmp_path / 'in.bam'
    _make_bam(p, reads)
    return p


def test_flag_mode_marks_duplicates(daf_bam, tmp_path):
    out = tmp_path / 'flag.bam'
    stats = run_dedup(str(daf_bam), str(out), min_jaccard=0.95, min_deam=10,
                      collapse=False)
    assert stats['n_fingerprintable'] == 8   # 9 reads, 1 below min-deam
    assert stats['n_clusters'] == 2
    assert stats['n_duplicates'] == 6         # (5-1) + (3-1)
    # All 9 reads kept; 6 carry the 0x400 duplicate flag.
    flags = [r.is_duplicate for r in pysam.AlignmentFile(str(out), check_sq=False)]
    assert len(flags) == 9
    assert sum(flags) == 6


def test_collapse_is_default_keeps_one_per_molecule(daf_bam, tmp_path):
    out = tmp_path / 'collapse.bam'
    run_dedup(str(daf_bam), str(out), min_jaccard=0.95, min_deam=10)  # collapse default
    recs = list(pysam.AlignmentFile(str(out), check_sq=False))
    # 2 molecule representatives + 1 passed-through low-deam read.
    assert len(recs) == 3
    assert not any(r.is_duplicate for r in recs)
    assert 'lowdeam' in {r.query_name for r in recs}


def test_call_dedup_first_collapses_daf(daf_bam):
    from fiberhmm.cli.call import _dedup_input_first
    # Pre-footprinting dedup returns a deduped temp BAM to footprint instead of
    # the original: 9 reads -> 2 reps + 1 low-deam passthrough.
    tmp = _dedup_input_first(str(daf_bam), str(daf_bam.parent / 'out.bam'),
                             0.95, False, 1, region_parallel=False)
    assert tmp is not None
    recs = list(pysam.AlignmentFile(tmp, check_sq=False))
    assert len(recs) == 3
    # Original input is left untouched (dedup writes a temp, not in place).
    assert len(list(pysam.AlignmentFile(str(daf_bam), check_sq=False))) == 9


def test_call_dedup_forwards_practical_params(daf_bam, monkeypatch):
    # The call wrapper must forward min_deam/prob_threshold/ignore_strand/
    # stats_tsv through to run_dedup, not silently drop them at defaults.
    import fiberhmm.cli.call as callmod
    captured = {}

    def fake_run_dedup(in_bam, out_bam, **kw):
        captured.update(kw)
        return {'n_clusters': 1}

    monkeypatch.setattr('fiberhmm.cli.dedup.run_dedup', fake_run_dedup)
    callmod._dedup_input_first(
        str(daf_bam), str(daf_bam.parent / 'o.bam'), 0.90, True, 2,
        region_parallel=False, min_deam=25, prob_threshold=200,
        ignore_strand=True, stats_tsv='x.tsv')
    assert captured['min_jaccard'] == 0.90
    assert captured['collapse'] is False  # flag_only=True
    assert captured['min_deam'] == 25
    assert captured['prob_threshold'] == 200
    assert captured['ignore_strand'] is True
    assert captured['stats_tsv'] == 'x.tsv'


def test_cluster_tags_present(daf_bam, tmp_path):
    out = tmp_path / 'flag.bam'
    run_dedup(str(daf_bam), str(out), min_jaccard=0.95, min_deam=10, collapse=False)
    for r in pysam.AlignmentFile(str(out), check_sq=False):
        if r.is_duplicate:
            assert r.has_tag('di') and r.has_tag('ds')
            assert r.get_tag('ds') > 1
