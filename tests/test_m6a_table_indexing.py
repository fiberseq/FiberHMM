"""m6A emission tables must be indexed with the inference encoder's context codes.

The table builder used to number contexts alphabetically (ACGT) while the encoder
uses A=0, C=1, T=2, G=3, which G/T-swaps every context. The Hia5 Nanopore table
shipped swapped through 2.16.8 (fixed 2026-09-29; DddB was fixed 2026-09-23).
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from fiberhmm.core.bam_reader import encode_from_query_sequence
from fiberhmm.probabilities.context_counter import (
    ContextCounter, encoder_context_code, reverse_complement,
)

MODELS = Path(__file__).resolve().parents[1] / 'fiberhmm' / 'models'
K = 3; NC = 4 ** (2 * K); OFF = NC + 1
CONTEXTS = [''.join(p[:K]) + 'A' + ''.join(p[K:]) for p in itertools.product('ACGT', repeat=2 * K)]


def meth_rate(name):
    e = np.array(json.loads((MODELS / name).read_text())['emissionprob'], float)
    return e[:, :NC] / (e[:, :NC] + e[:, OFF:OFF + NC])


def digit_perm(mapping):
    """Code permutation that relabels every base digit of a context code."""
    c = np.arange(NC); out = np.zeros(NC, int)
    for pos in range(2 * K):
        out += np.array(mapping)[(c // 4 ** pos) % 4] * 4 ** pos
    return out


def encoder_code(read, pos, is_reverse):
    code = int(encode_from_query_sequence(read, set(), 0, mode='nanopore-fiber', strand='.',
                                          context_size=K, is_reverse=is_reverse)[pos])
    return code - OFF if code >= OFF else code


@pytest.mark.parametrize('seed', range(3))
def test_builder_codes_match_nanopore_encoder_on_both_strands(seed):
    rng = np.random.default_rng(seed)
    for _ in range(50):
        flank = ''.join(rng.choice(list('ACGT'), 2 * K))
        ctx = flank[:K] + 'A' + flank[K:]
        assert encoder_code('CCCCCCCC' + ctx + 'CCCCCCCC', 8 + K, False) == encoder_context_code(ctx, 'A')
        # a reverse-aligned read stores the basecall reverse-complemented (methylated A on a SEQ T)
        assert encoder_code('GGGGGGGG' + reverse_complement(ctx) + 'GGGGGGGG', 8 + K, True) == encoder_context_code(ctx, 'A')


def test_reindexed_v29_table_is_the_gt_swapped_table_in_encoder_order():
    new = np.array(json.loads((MODELS / 'legacy' / 'hia5_nanopore_v2.9_reindexed_legacy.json').read_text())['emissionprob'])
    old = np.array(json.loads((MODELS / 'legacy' / 'hia5_nanopore_gt_swapped_legacy.json').read_text())['emissionprob'])
    alphabetical = {c: i for i, c in enumerate(sorted(CONTEXTS))}
    for ctx in CONTEXTS[::37]:
        e, a = encoder_context_code(ctx, 'A'), alphabetical[ctx]
        assert np.allclose(new[:, e], old[:, a]) and np.allclose(new[:, OFF + e], old[:, OFF + a])


@pytest.mark.parametrize('name,reference,states', [
    # The control-built Nanopore table's protected state is basecaller background on untreated
    # DNA, which has no methylase context preference to share; compare its accessible state only.
    ('hia5_nanopore.json', 'hia5_pacbio.json', 'accessible'),
    ('legacy/hia5_nanopore_v2.9_reindexed_legacy.json', 'hia5_pacbio.json', 'both'),
    ('ecogii_pacbio.json', 'hia5_pacbio.json', 'both'),
    ('hia5_pacbio.json', 'ecogii_pacbio.json', 'both'),
])
def test_bundled_m6a_tables_agree_with_each_other_in_encoder_order(name, reference, states):
    """Adenine-methylase context preferences are shared across tables. A G/T digit-order
    bug makes the G/T relabelling fit far better than the identity (legacy Hia5 Nanopore:
    0.61/0.39 swapped vs 0.12/0.04 as used)."""
    table, ref = meth_rate(name), meth_rate(reference)
    rows = range(table.shape[0])
    if states == 'accessible':
        rows = [int(np.argmax(table.mean(axis=1)))]
        assert rows[0] == int(np.argmax(ref.mean(axis=1)))
    for s in rows:
        score = {m: np.corrcoef(table[s][digit_perm(m)], ref[s])[0, 1] for m in itertools.permutations(range(4))}
        top2 = sorted(score, key=score.get, reverse=True)[:2]
        assert (0, 1, 2, 3) in top2, (name, s, sorted(score.items(), key=lambda kv: -kv[1])[:3])
        assert score[(0, 1, 2, 3)] - score[(0, 1, 3, 2)] > 0.2, (name, s, score[(0, 1, 2, 3)], score[(0, 1, 3, 2)])


def test_counter_round_trip_nanopore_reverse_reads_count_like_forward_reads():
    """probs -> table on synthetic data: each context's rate lands at its encoder code,
    and reverse-aligned nanopore reads contribute the same counts as forward reads."""
    rng = np.random.default_rng(0)
    rate = {c: rng.uniform(0.05, 0.95) for c in CONTEXTS}
    fwd, rev = ContextCounter(K, 'A'), ContextCounter(K, 'A')
    for _ in range(1500):
        read = ''.join(rng.choice(list('ACGT'), 1000))
        mods = {i for i in range(K, len(read) - K) if read[i] == 'A' and rng.random() < rate[read[i - K:i + K + 1]]}
        fwd.process_read(read, mods, edge_trim=K)
        L = len(read)
        rev.process_read(reverse_complement(read), {L - 1 - p for p in mods}, edge_trim=K, is_reverse=True)
    _, tf = fwd.get_encoding_table(K)
    _, tr = rev.get_encoding_table(K)
    assert tf[['encode', 'hit', 'nohit']].equals(tr[['encode', 'hit', 'nohit']])
    for _, row in tf[tf['hit'] + tf['nohit'] >= 30].iterrows():
        assert row['encode'] == encoder_context_code(row['context'], 'A')
    observed = np.array([r['ratio'] for _, r in tf.iterrows() if r['hit'] + r['nohit'] >= 30])
    truth = np.array([rate[r['context']] for _, r in tf.iterrows() if r['hit'] + r['nohit'] >= 30])
    assert np.corrcoef(observed, truth)[0, 1] > 0.9


def test_alphabetical_numbering_is_refused():
    with pytest.raises(ValueError):
        ContextCounter(K, 'A').get_probabilities(K, encode_by_code=False)


def test_hia5_nanopore_table_provenance():
    """The bundled Nanopore Hia5 table is the 2026-10-02 control build: naked-DNA (accessible) and untreated
    (protected) yw 2-4 h embryo libraries, m6A at ML >= 248, with the Hia5 PacBio start/transition probabilities."""
    model = json.loads((MODELS / 'hia5_nanopore.json').read_text())
    pacbio = json.loads((MODELS / 'hia5_pacbio.json').read_text())
    assert model['mode'] == 'nanopore-fiber' and model['context_size'] == K
    assert model['startprob'] == pacbio['startprob'] and model['transmat'] == pacbio['transmat']
    for words in ('naked', 'untreated', '-p 248', 'encoder-order', 'hia5_pacbio.json'):
        assert words in model['note'], words
    acc, prot = sorted(meth_rate('hia5_nanopore.json'), key=lambda r: -r.mean())
    assert 0.15 < acc.mean() < 0.35 and prot.mean() < 0.002


def test_hia5_nanopore_both_states_equal_their_control_counts_by_encoder_code():
    """Each state of the bundled Nanopore table is exactly the m6A rate counted per
    context in its control (tests/fixtures: the fiberhmm-probs counts it was built
    from). Looked up by the encoder's code for each context, so a G/T swap of either
    row -- not only the accessible one the cross-table check above can see -- fails."""
    import gzip
    import pandas as pd
    path = Path(__file__).resolve().parent / 'fixtures' / 'hia5_nanopore_control_counts_k3.tsv.gz'
    with gzip.open(path, 'rt') as handle:
        counts = pd.read_csv(handle, sep='\t', comment='#')
    rates = meth_rate('hia5_nanopore.json')
    acc_row = int(np.argmax(rates.mean(axis=1)))
    for state, row in (('accessible', acc_row), ('protected', 1 - acc_row)):
        part = counts[counts.state == state]
        assert len(part) == NC
        codes = np.array([encoder_context_code(c, 'A') for c in part.context])
        assert np.array_equal(codes, part.encode.to_numpy())
        expected = (part.hit / (part.hit + part.nohit)).to_numpy()
        assert np.allclose(rates[row][codes], expected, rtol=0, atol=1e-12), state
        swapped = digit_perm((0, 1, 3, 2))
        assert not np.allclose(rates[row][swapped][codes], expected, rtol=0, atol=1e-6), state
