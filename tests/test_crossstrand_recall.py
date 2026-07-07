"""Tests for both-strand consensus footprint re-calling (fiberhmm-merge --recall)."""
import numpy as np

from fiberhmm.core.bam_reader import ContextEncoder
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
