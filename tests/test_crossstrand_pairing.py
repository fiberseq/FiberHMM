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
    STATUS_UNRESOLVED, _gaussian_kernel, _sequence_signature, assign_pairs,
    score_pair,
)


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
