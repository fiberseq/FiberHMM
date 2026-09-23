"""DAF emission tables must be indexed with the inference encoder's context codes.

The table builder's legacy path numbered contexts alphabetically (ACGT), while the
encoder uses A=0, C=1, T=2, G=3; the DddB table shipped G/T-swapped (fixed 2026-09-23).
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from fiberhmm.core.bam_reader import _encode_vectorized, encode_from_query_sequence

MODELS = Path(__file__).resolve().parents[1] / 'fiberhmm' / 'models'
K = 3; NC = 4 ** (2 * K); OFF = NC + 1


def builder_code(ctx):
    """Code the table builder assigns with encode_by_code=True (C-centred context)."""
    return int(_encode_vectorized(ctx, 'C', K, 0, NC, include_rc=False)[K])


def encoder_code(read, pos, strand):
    code = int(encode_from_query_sequence(read, set(), 0, mode='daf', strand=strand, context_size=K)[pos])
    return code - OFF if code >= OFF else code


def revcomp(s):
    return s.translate(str.maketrans('ACGT', 'TGCA'))[::-1]


@pytest.mark.parametrize('seed', range(3))
def test_builder_codes_match_daf_encoder_on_both_strands(seed):
    rng = np.random.default_rng(seed)
    for _ in range(50):
        flank = ''.join(rng.choice(list('ACGT'), 2 * K))
        ctx = flank[:K] + 'C' + flank[K:]                  # C-centred context as stored by the builder
        read = 'TTTTTTTT' + ctx + 'TTTTTTTT'
        assert encoder_code(read, 8 + K, '+') == builder_code(ctx)
        # the GA strand sees the reverse complement (G-centred); the encoder RCs back
        read_ga = 'AAAAAAAA' + revcomp(ctx) + 'AAAAAAAA'
        assert encoder_code(read_ga, 8 + K, '-') == builder_code(ctx)


def test_dddb_table_is_encoder_indexed():
    new = np.array(json.loads((MODELS / 'dddb_nanopore.json').read_text())['emissionprob'])
    old = np.array(json.loads((MODELS / 'legacy' / 'dddb_nanopore_gt_swapped_legacy.json').read_text())['emissionprob'])
    contexts = [''.join(p[:K]) + 'C' + ''.join(p[K:]) for p in itertools.product('ACGT', repeat=2 * K)]
    alphabetical = {c: i for i, c in enumerate(sorted(contexts))}
    for ctx in contexts[::97]:
        e, a = builder_code(ctx), alphabetical[ctx]
        assert np.allclose(new[:, e], old[:, a]) and np.allclose(new[:, OFF + e], old[:, OFF + a])
