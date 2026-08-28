"""Tests for both-strand consensus footprint re-calling (fiberhmm-merge --recall)."""
import numpy as np
import pytest

from fiberhmm.cli.crossstrand import run_pipeline
from fiberhmm.core.bam_reader import ContextEncoder
from fiberhmm.crossstrand.pairing import PairParams
from fiberhmm.crossstrand.recall import decode_ry_consensus, encode_daf_both_strand


def test_decode_ry_consensus():
    conv, ct, ga = decode_ry_consensus('ACYGTRAC')
    assert conv == 'ACTGTAAC'      # Y->T, R->A
    assert ct == {2}               # Y position (C->T)
    assert ga == {5}               # R position (G->A)


def test_both_strand_encoder_regime_masking():
    # Layout (k=1, edge_trim=0): C-only region [0,15), G-only [15,30),
    # both [30,45). A reference base is informative only if its strand covers it.
    n = 45
    seq = ['A'] * n
    seq[7] = 'C'    # C in C-only  -> informative (+ strand covers)
    seq[10] = 'G'   # G in C-only  -> NOT informative (- strand absent here)
    seq[22] = 'G'   # G in G-only  -> informative
    seq[25] = 'C'   # C in G-only  -> NOT informative (+ strand absent here)
    seq[33] = 'C'   # C in both    -> informative
    seq[36] = 'G'   # G in both    -> informative (double density)
    conv = ''.join(seq)

    plus = np.array([True] * 15 + [False] * 15 + [True] * 15)
    minus = np.array([False] * 15 + [True] * 30)

    enc = encode_daf_both_strand(conv, set(), set(), plus, minus,
                                 edge_trim=0, context_size=1)
    n_codes = ContextEncoder.get_n_codes(1)
    fill = 2 * n_codes + 1

    assert enc[7] != fill        # C in C-only: informative
    assert enc[10] == fill       # G in C-only: no GA data -> non-target
    assert enc[22] != fill       # G in G-only: informative
    assert enc[25] == fill       # C in G-only: no CT data -> non-target
    assert enc[33] != fill       # C in both: informative
    assert enc[36] != fill       # G in both: informative


def test_both_strand_hit_miss_and_swap():
    # Deaminated targets -> hit codes; non-deaminated targets -> miss codes; and
    # the merged obs equals each strand's own production encoding in its region.
    from fiberhmm.core.bam_reader import _encode_daf_observations
    n_codes = ContextEncoder.get_n_codes(3)
    non_target, unmeth = n_codes, n_codes + 1
    # long enough that interior C/G have valid 7-mer context past edge_trim
    conv = ('ACGTACGT' * 8)             # regular C's and G's throughout
    L = len(conv)
    both = np.ones(L, dtype=bool)       # whole read is both-strand
    ct = {i for i, b in enumerate(conv) if b == 'C' and 12 <= i <= 20}  # some C's deaminated
    ga = {i for i, b in enumerate(conv) if b == 'G' and 30 <= i <= 40}  # some G's deaminated
    conv_d = list(conv)
    for i in ct:
        conv_d[i] = 'T'                 # deaminated C shows as T
    for i in ga:
        conv_d[i] = 'A'                 # deaminated G shows as A
    conv_d = ''.join(conv_d)

    enc = encode_daf_both_strand(conv_d, ct, ga, both, both, edge_trim=10, context_size=3)
    # The expected per-strand encodings retain that strand's T/A hit markers
    # but restore the opposite strand before deriving sequence context.
    plus_seq = list(conv_d)
    minus_seq = list(conv_d)
    for i in ga:
        plus_seq[i] = 'G'
    for i in ct:
        minus_seq[i] = 'C'
    plus_full = _encode_daf_observations(
        ''.join(plus_seq), ct, 10, '+', 3, non_target, unmeth,
    )
    minus_full = _encode_daf_observations(
        ''.join(minus_seq), ga, 10, '-', 3, non_target, unmeth,
    )

    def is_hit(c):
        return 0 <= c < n_codes

    def is_miss(c):
        return unmeth <= c < unmeth + n_codes

    # swap: merged == + encoding at C-origin, == - encoding at G-origin (interior)
    for i in range(15, L - 15):
        b = conv[i]
        if b == 'C':
            assert enc[i] == plus_full[i]
        elif b == 'G':
            assert enc[i] == minus_full[i]
    # every interior deaminated C/G that is scorable is a hit; non-deam target is a miss
    for i in ct:
        if 12 < i < L - 12 and is_hit(plus_full[i]):
            assert is_hit(enc[i])
    for i in range(15, L - 15):
        if conv[i] == 'C' and i not in ct and is_miss(plus_full[i]):
            assert is_miss(enc[i])


def test_opposite_strand_deamination_does_not_change_context_code():
    """The other channel may change hit density, never a target's 7-mer."""
    from fiberhmm.core.bam_reader import _encode_daf_observations

    reference = list('ACGT' * 20)
    target_c = 37
    assert reference[target_c] == 'C'
    nearby_g = 38
    assert reference[nearby_g] == 'G'
    ga = {nearby_g}
    mixed = reference.copy()
    mixed[nearby_g] = 'A'
    mixed = ''.join(mixed)
    mask = np.ones(len(mixed), dtype=bool)

    observed = encode_daf_both_strand(
        mixed, set(), ga, mask, mask, edge_trim=10, context_size=3,
    )
    n_codes = ContextEncoder.get_n_codes(3)
    expected_plus = _encode_daf_observations(
        ''.join(reference), set(), 10, '+', 3, n_codes, n_codes + 1,
    )
    stale_mixed_plus = _encode_daf_observations(
        mixed, set(), 10, '+', 3, n_codes, n_codes + 1,
    )

    assert observed[target_c] == expected_plus[target_c]
    assert stale_mixed_plus[target_c] != expected_plus[target_c]


def test_both_strand_doubles_informative_density_in_core():
    # A stretch of alternating C/G: in a both-strand region every C and G is
    # informative; in a C-only region only the C's are.
    conv = ('CG' * 30)
    n = len(conv)
    both = np.ones(n, dtype=bool)
    only_plus = np.ones(n, dtype=bool)
    no_minus = np.zeros(n, dtype=bool)
    fill = 2 * ContextEncoder.get_n_codes(1) + 1
    enc_both = encode_daf_both_strand(conv, set(), set(), both, both, 0, 1)
    enc_conly = encode_daf_both_strand(conv, set(), set(), only_plus, no_minus, 0, 1)
    info_both = np.mean(enc_both[1:-1] != fill)
    info_conly = np.mean(enc_conly[1:-1] != fill)
    assert info_both > 1.8 * info_conly   # ~2x denser


def test_crossstrand_pipeline_rejects_non_ddda_enzyme():
    with pytest.raises(ValueError, match='specific to double-strand DddA'):
        run_pipeline('unused.bam', 'unused.out.bam', PairParams(), enzyme='dddb')
