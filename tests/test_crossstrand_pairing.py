"""Tests for cross-strand pairing (fiberhmm-pair core).

These exercise the I/O-free pairing library directly with synthetic
:class:`ReadFeat` objects, so no BAM fixtures are needed. The scenarios mirror
the diploid locus structure: <=2 CT + <=2 GA reads, where the same molecule's
two strands share nucleosome dyads and different homologs do not.
"""
from dataclasses import replace

import numpy as np
import pysam
import pytest

from fiberhmm.crossstrand.pairing import (
    FLAVOR_CT, FLAVOR_GA, PairParams, ReadFeat, STATUS_NONE, STATUS_PAIRED,
    STATUS_UNRESOLVED, SequenceScore, _gaussian_kernel, _sequence_signature,
    _sequence_signature_from_md, assign_pairs, read_flavor, score_pair,
)
from fiberhmm.cli.pair import _has_usable_sequence_score, run_pair


def _mk(index, name, flavor, start, end, dyads, params):
    """Build a ReadFeat with a rasterized dyad-density signal (test helper)."""
    g = params.grid_bp
    kern = _gaussian_kernel(params.sigma_bp, g)
    krad = len(kern) // 2
    grid0 = start // g
    sig = np.zeros(end // g - grid0 + 1, dtype=np.float32)
    for c in dyads:
        cb = c // g - grid0
        lo = max(0, cb - krad)
        hi = min(len(sig), cb + krad + 1)
        sig[lo:hi] += kern[(lo - (cb - krad)):(lo - (cb - krad)) + (hi - lo)]
    return ReadFeat(index, name, flavor, start, end,
                    np.array(sorted(dyads), dtype=np.int64), grid0, sig)


def _with_sequence(feat, base, n=1000):
    feat.sequence_pos = np.arange(1000, 1000 + n, dtype=np.int64)
    feat.sequence_base = np.full(n, ord(base), dtype=np.uint8)
    return feat


def test_sequence_signature_excludes_reference_cg_but_keeps_at_variants():
    header = pysam.AlignmentHeader.from_dict(
        {'SQ': [{'SN': 'chr1', 'LN': 6}]},
    )
    read = pysam.AlignedSegment(header)
    read.query_name = 'r'
    read.reference_id = 0
    read.reference_start = 0
    read.cigartuples = [(0, 6)]
    read.query_sequence = 'ATGTYR'
    pos, base = _sequence_signature(
        read, np.frombuffer(b'ACGTAT', dtype=np.uint8),
    )
    assert pos.tolist() == [0, 3, 4, 5]
    assert bytes(base).decode() == 'ATCG'


def test_read_flavor_rejects_mixed_tie():
    header = pysam.AlignmentHeader.from_dict(
        {'SQ': [{'SN': 'chr1', 'LN': 100}]},
    )
    read = pysam.AlignedSegment(header)
    read.query_name = 'mixed'
    read.reference_id = 0
    read.reference_start = 0
    read.cigartuples = [(0, 4)]
    read.query_sequence = 'YRAA'
    assert read_flavor(read, 0) is None


@pytest.fixture
def params():
    return PairParams(min_overlap_bp=1000, min_nucs=3, min_score=0.4,
                      min_margin=0.05, grid_bp=10, sigma_bp=30.0, max_lag_bp=40)


@pytest.fixture
def homolog_dyads():
    # Two homologs: same-ish spacing but distinct phase/period -> low cross-corr.
    a = [1000 + 190 * i for i in range(20)]
    b = [1000 + 95 + 205 * i for i in range(19)]
    return a, b


def test_same_molecule_scores_high_diff_homolog_low(params, homolog_dyads):
    a, b = homolog_dyads
    ctA = _mk(0, 'ctA', FLAVOR_CT, 900, 4800, a, params)
    gaA = _mk(1, 'gaA', FLAVOR_GA, 1000, 4900, a, params)
    gaB = _mk(2, 'gaB', FLAVOR_GA, 980, 4750, b, params)
    assert score_pair(ctA, gaA, params) > 0.9      # same molecule
    assert score_pair(ctA, gaB, params) < 0.4      # different homolog


def test_2x2_resolves_diagonal(params, homolog_dyads):
    a, b = homolog_dyads
    feats = [
        _mk(0, 'ctA', FLAVOR_CT, 900, 4800, a, params),
        _mk(1, 'ctB', FLAVOR_CT, 950, 4700, b, params),
        _mk(2, 'gaA', FLAVOR_GA, 1000, 4900, a, params),
        _mk(3, 'gaB', FLAVOR_GA, 980, 4750, b, params),
    ]
    res = assign_pairs(feats, params)
    assert res.partner == {0: 2, 2: 0, 1: 3, 3: 1}
    assert all(res.status[i] == STATUS_PAIRED for i in range(4))


def test_same_flavor_never_paired(params, homolog_dyads):
    a, _ = homolog_dyads
    # Two identical-pattern CT reads must not pair with each other.
    feats = [
        _mk(0, 'ct1', FLAVOR_CT, 900, 4800, a, params),
        _mk(1, 'ct2', FLAVOR_CT, 950, 4850, a, params),
    ]
    res = assign_pairs(feats, params)
    assert res.partner == {}
    assert all(res.status[i] == STATUS_NONE for i in range(2))


def test_ambiguous_locus_left_unresolved(params):
    # One CT read equally (mis)matching two GA reads -> margin gate declines.
    dy = [1000 + 190 * i for i in range(20)]
    ct = _mk(0, 'ct', FLAVOR_CT, 900, 4800, dy, params)
    # Two GA reads with the SAME dyads as ct: both score ~1.0 -> no margin.
    gaX = _mk(1, 'gaX', FLAVOR_GA, 1000, 4900, dy, params)
    gaY = _mk(2, 'gaY', FLAVOR_GA, 980, 4850, dy, params)
    res = assign_pairs([ct, gaX, gaY], params)
    assert 0 not in res.partner              # ct not confidently paired
    assert res.status[0] == STATUS_UNRESOLVED


def test_lone_pair_must_beat_null_floor(params, homolog_dyads):
    # A 1+1 locus (no alternative) still pairs when its score clears the null,
    # but is held back when the null floor is raised above it -- the virtual
    # competitor stands in for the missing second-best.
    a, _ = homolog_dyads
    ct = _mk(0, 'ct', FLAVOR_CT, 900, 4800, a, params)
    ga = _mk(1, 'ga', FLAVOR_GA, 1000, 4900, a, params)
    res = assign_pairs([ct, ga], params)          # score ~1.0, null_floor 0.25
    assert res.partner == {0: 1, 1: 0}
    strict = replace(params, null_floor=0.99, min_margin=0.05)
    res2 = assign_pairs([ct, ga], strict)
    assert 0 not in res2.partner
    assert res2.status[0] == STATUS_UNRESOLVED


def test_non_overlapping_not_scored(params, homolog_dyads):
    a, _ = homolog_dyads
    ct = _mk(0, 'ct', FLAVOR_CT, 900, 4800, a, params)
    far = _mk(1, 'ga', FLAVOR_GA, 50000, 54000,
              [50000 + 190 * i for i in range(20)], params)
    assert score_pair(ct, far, params) is None
    res = assign_pairs([ct, far], params)
    assert res.status[0] == STATUS_NONE and res.status[1] == STATUS_NONE


def test_sequence_component_resolves_diagonal_without_footprints(params):
    # The joint 2x2 assignment uses the discordant off-diagonal edges to infer
    # both opposite pairs. No dyads are present, so footprint matching cannot
    # be responsible for the result.
    empty = []
    feats = [
        _with_sequence(_mk(0, 'ctA', FLAVOR_CT, 900, 4800, empty, params), 'A'),
        _with_sequence(_mk(1, 'ctB', FLAVOR_CT, 900, 4800, empty, params), 'G'),
        _with_sequence(_mk(2, 'gaA', FLAVOR_GA, 900, 4800, empty, params), 'A'),
        _with_sequence(_mk(3, 'gaB', FLAVOR_GA, 900, 4800, empty, params), 'G'),
    ]
    res = assign_pairs(feats, params)
    assert res.partner == {0: 2, 2: 0, 1: 3, 3: 1}
    assert set(res.method.values()) == {'S'}
    assert all(res.sequence[i].mismatches == 0 for i in range(4))


def test_sequence_ambiguous_component_falls_back_to_footprints(
        params, homolog_dyads):
    a, b = homolog_dyads
    feats = [
        _with_sequence(_mk(0, 'ctA', FLAVOR_CT, 900, 4800, a, params), 'A'),
        _with_sequence(_mk(1, 'ctB', FLAVOR_CT, 900, 4800, b, params), 'A'),
        _with_sequence(_mk(2, 'gaA', FLAVOR_GA, 900, 4800, a, params), 'A'),
        _with_sequence(_mk(3, 'gaB', FLAVOR_GA, 900, 4800, b, params), 'A'),
    ]
    res = assign_pairs(feats, params)
    assert res.partner == {0: 2, 2: 0, 1: 3, 3: 1}
    assert set(res.method.values()) == {'F'}


def test_sequence_discordance_vetoes_high_footprint_match(params, homolog_dyads):
    a, _ = homolog_dyads
    ct = _with_sequence(_mk(0, 'ct', FLAVOR_CT, 900, 4800, a, params), 'A')
    ga = _with_sequence(_mk(1, 'ga', FLAVOR_GA, 900, 4800, a, params), 'G')
    res = assign_pairs([ct, ga], params)
    assert res.partner == {}
    assert res.status[0] == STATUS_UNRESOLVED


def test_sequence_does_not_force_lone_identical_overlap(params):
    empty = []
    ct = _with_sequence(_mk(0, 'ct', FLAVOR_CT, 900, 4800, empty, params), 'A')
    ga = _with_sequence(_mk(1, 'ga', FLAVOR_GA, 900, 4800, empty, params), 'A')
    res = assign_pairs([ct, ga], params)
    assert res.partner == {}
    assert res.status[0] == STATUS_UNRESOLVED


def test_single_cell_haplotype_pairs_staggered_one_plus_one_without_footprints(params):
    phased = replace(params, single_cell_haplotype=True)
    ct = _mk(0, 'ct', FLAVOR_CT, 900, 4800, [], phased)
    ga = _mk(1, 'ga', FLAVOR_GA, 1800, 5600, [], phased)
    res = assign_pairs([ct, ga], phased)
    assert res.partner == {0: 1, 1: 0}
    assert set(res.method.values()) == {'H'}
    assert all(res.status[i] == STATUS_PAIRED for i in range(2))


def test_single_cell_haplotype_fails_closed_on_ambiguous_overlap(params):
    phased = replace(params, single_cell_haplotype=True)
    ct = _mk(0, 'ct', FLAVOR_CT, 900, 4800, [], phased)
    ga1 = _mk(1, 'ga1', FLAVOR_GA, 1000, 4900, [], phased)
    ga2 = _mk(2, 'ga2', FLAVOR_GA, 1100, 5000, [], phased)
    res = assign_pairs([ct, ga1, ga2], phased)
    assert res.partner == {}
    assert all(res.status[i] == STATUS_UNRESOLVED for i in range(3))


def test_single_cell_haplotype_preserves_sequence_fallback_for_ambiguous_component(params):
    phased = replace(params, single_cell_haplotype=True)
    feats = [
        _with_sequence(_mk(0, 'ctA', FLAVOR_CT, 900, 4800, [], phased), 'A'),
        _with_sequence(_mk(1, 'ctB', FLAVOR_CT, 900, 4800, [], phased), 'G'),
        _with_sequence(_mk(2, 'gaA', FLAVOR_GA, 900, 4800, [], phased), 'A'),
        _with_sequence(_mk(3, 'gaB', FLAVOR_GA, 900, 4800, [], phased), 'G'),
    ]
    res = assign_pairs(feats, phased)
    assert res.partner == {0: 2, 2: 0, 1: 3, 3: 1}
    assert set(res.method.values()) == {'S'}


def test_snp_scale_2x2_difference_is_not_force_assigned(params):
    # A single SNP supports one diagonal, but without gross discordance this is
    # deliberately left for footprints. This guards against the overmatching
    # seen when residual consensus errors were treated as global constraints.
    strict = replace(params, min_sequence_margin=0.002)
    feats = [
        _with_sequence(_mk(0, 'ctA', FLAVOR_CT, 900, 4800, [], strict), 'A'),
        _with_sequence(_mk(1, 'ctB', FLAVOR_CT, 900, 4800, [], strict), 'A'),
        _with_sequence(_mk(2, 'gaA', FLAVOR_GA, 900, 4800, [], strict), 'A'),
        _with_sequence(_mk(3, 'gaB', FLAVOR_GA, 900, 4800, [], strict), 'A'),
    ]
    # Homolog B differs at one of 1,000 safe positions.
    feats[1].sequence_base[0] = ord('G')
    feats[3].sequence_base[0] = ord('G')
    res = assign_pairs(feats, strict)
    assert res.partner == {}
    assert all(res.status[i] == STATUS_UNRESOLVED for i in range(4))

    # Tightening the footprint fallback veto must not make the independent 2x2
    # constraint more permissive.
    tighter_fallback = replace(strict, max_sequence_mismatch_rate=0.0005)
    res = assign_pairs(feats, tighter_fallback)
    assert res.partner == {}
    assert all(res.status[i] == STATUS_UNRESOLVED for i in range(4))


def test_md_signature_recovers_safe_reference_bases_without_fasta():
    header = pysam.AlignmentHeader.from_dict({
        'HD': {'VN': '1.6'}, 'SQ': [{'SN': 'chr1', 'LN': 1000}],
    })
    read = pysam.AlignedSegment(header)
    read.query_name = 'md'
    read.reference_id = 0
    read.reference_start = 100
    read.cigarstring = '6M'
    # Reference ACGTAT; query differs G-for-A at the first safe position.
    read.query_sequence = 'GCGTAT'
    read.set_tag('MD', '0A5')
    positions, bases = _sequence_signature_from_md(read)
    assert positions.tolist() == [100, 103, 104, 105]
    assert bytes(bases).decode() == 'GTAT'


def test_empty_sequence_score_is_not_serialized():
    assert not _has_usable_sequence_score(None)
    assert not _has_usable_sequence_score(SequenceScore(0, 0, float('nan')))
    assert _has_usable_sequence_score(SequenceScore(1000, 0, 0.0))


def test_pair_cli_fails_closed_on_duplicate_primary_query_names(tmp_path):
    header = pysam.AlignmentHeader.from_dict({
        'HD': {'VN': '1.6', 'SO': 'coordinate'},
        'SQ': [{'SN': 'chr1', 'LN': 5000}],
    })

    def called(name, start, marker):
        read = pysam.AlignedSegment(header)
        read.query_name = name
        read.reference_id = 0
        read.reference_start = start
        read.mapping_quality = 60
        sequence = list('A' * 1000)
        sequence[100] = marker
        read.query_sequence = ''.join(sequence)
        read.query_qualities = pysam.qualitystring_to_array('I' * 1000)
        read.cigartuples = [(0, 1000)]
        read.set_tag('MA', '1000;nuc.:1-150,201-150,401-150,601-150')
        return read

    input_bam = tmp_path / 'duplicates.bam'
    with pysam.AlignmentFile(input_bam, 'wb', header=header) as out:
        out.write(called('duplicate', 100, 'Y'))
        out.write(called('duplicate', 100, 'Y'))
        out.write(called('ga', 100, 'R'))
    pysam.index(str(input_bam))

    with pytest.raises(ValueError, match='unique primary query names'):
        run_pair(
            str(input_bam), str(tmp_path / 'paired.bam'),
            PairParams(min_overlap_bp=100, min_nucs=1, min_sequence_bases=100),
            io_threads=1,
        )


def test_pair_then_merge_bam_io_contract(tmp_path):
    """Exercise real tagging, TSV serialization, consensus BAM, and index."""
    from fiberhmm.cli.merge import run_merge

    header = pysam.AlignmentHeader.from_dict({
        'HD': {'VN': '1.6', 'SO': 'coordinate'},
        'SQ': [{'SN': 'chr1', 'LN': 5000}],
    })

    def called(name, marker, opposite_base):
        read = pysam.AlignedSegment(header)
        read.query_name = name
        read.reference_id = 0
        read.reference_start = 100
        read.mapping_quality = 42
        sequence = list('A' * 500)
        sequence[100] = marker
        sequence[101] = opposite_base
        read.query_sequence = ''.join(sequence)
        read.query_qualities = pysam.qualitystring_to_array('I' * 500)
        read.cigartuples = [(0, 500)]
        read.set_tag('MA', '500;nuc.:51-147,251-147')
        return read

    # CT Y is canonical C on GA; GA R is canonical G on CT.
    ct = called('ct-io', 'Y', 'G')
    ga = called('ga-io', 'C', 'R')
    source = tmp_path / 'source.bam'
    with pysam.AlignmentFile(source, 'wb', header=header) as out:
        out.write(ct)
        supplementary = ct.__copy__()
        supplementary.flag |= 0x800
        out.write(supplementary)
        out.write(ga)
    pysam.index(str(source))

    paired = tmp_path / 'paired.bam'
    pairs_tsv = tmp_path / 'pairs.tsv'
    params = PairParams(
        min_overlap_bp=100, min_nucs=1, min_score=0.1,
        min_margin=0.01, null_floor=0.0, min_sequence_bases=100,
    )
    result = run_pair(
        str(source), str(paired), params, pairs_tsv=str(pairs_tsv), io_threads=1,
    )
    assert result['n_pairs'] == 1
    assert pairs_tsv.read_text().splitlines()[0].split('\t')[5] == 'footprint_margin'

    with pysam.AlignmentFile(paired, 'rb') as bam:
        rows = list(bam.fetch(until_eof=True))
    assert len(rows) == 3
    primary = [row for row in rows if not row.is_supplementary]
    assert len(primary) == 2
    assert all(row.get_tag('mt') == 'P' for row in primary)
    assert all(row.get_tag('pm') == 'F' for row in primary)
    assert all(row.has_tag('mg') and not row.has_tag('sg') for row in primary)
    assert not next(row for row in rows if row.is_supplementary).has_tag('mt')

    consensus = tmp_path / 'consensus.bam'
    merged = run_merge(
        str(paired), str(consensus), pairs_only=True, io_threads=1,
    )
    assert merged['n_consensus'] == 1
    assert consensus.with_suffix('.bam.bai').exists()
    with pysam.AlignmentFile(consensus, 'rb') as bam:
        joint = next(bam.fetch(until_eof=True))
    assert joint.get_tag('cs') == 'ct-io;ga-io'
    assert joint.get_tag('dc') == 2
    assert joint.get_tag('bc') == 0
    assert not joint.has_tag('NM')
    assert joint.query_qualities is None
    assert joint.mapping_quality == 42

    full = tmp_path / 'full-consensus.bam'
    run_merge(str(paired), str(full), pairs_only=False, io_threads=1)
    with pysam.AlignmentFile(full, 'rb') as bam:
        full_rows = list(bam.fetch(until_eof=True))
    assert [row.query_name for row in full_rows] == ['ct-io.cs']
