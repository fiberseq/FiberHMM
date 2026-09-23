import numpy as np
import pytest

from fiberhmm.core import bam_reader as br

NON_TARGET = 8193   # non_target_code (4096) + unmethylated_offset (4097) at k=3


def targets(codes):
    return [i for i, c in enumerate(codes) if c < 4096 or 4097 <= c < 8193]


def hits(codes):
    return [i for i in targets(codes) if codes[i] < 4096]


@pytest.fixture(autouse=True)
def reset_mask():
    br.configure_daf_run_mask(0)
    yield
    br.configure_daf_run_mask(0)


def test_disabled_by_default_in_a_clean_environment(monkeypatch):
    monkeypatch.setattr(br, '_DAF_RUN_MASK_MIN', None)
    monkeypatch.delenv(br._DAF_RUN_MASK_ENV, raising=False)
    assert br.daf_run_mask_min_length() == 0


def ct_read():
    # isolated C at 4, CC at 10-11, CCC at 18-20; C4 and C11 deaminated (read shows T)
    seq = list('AATACGATTACCATAGTACCCATAT'); seq[4] = 'T'; seq[11] = 'T'
    return ''.join(seq), {4, 11}


def ga_read():
    # isolated G at 4, GG at 10-11, GGG at 18-20; G4 and G10 deaminated (read shows A)
    seq = list('AATAGCATTAGGATACTAGGGATAT'); seq[4] = 'A'; seq[10] = 'A'
    return ''.join(seq), {4, 10}


def test_mask_off_keeps_every_target():
    read, mods = ct_read()
    codes = br.encode_from_query_sequence(read, mods, 0, mode='daf', strand='+', context_size=3)
    assert targets(codes) == [4, 10, 11, 18, 19, 20] and hits(codes) == [4, 11]


@pytest.mark.parametrize('make,strand', [(ct_read, '+'), (ga_read, '-')])
def test_runs_of_two_or_more_are_removed_on_both_strands(make, strand):
    read, mods = make()
    br.configure_daf_run_mask(2, 'drop')
    codes = br.encode_from_query_sequence(read, mods, 0, mode='daf', strand=strand, context_size=3)
    assert targets(codes) == [4] and hits(codes) == [4]
    assert all(codes[i] == NON_TARGET for i in (10, 11, 18, 19, 20))


def test_run_length_is_measured_on_the_original_molecule():
    # C11 converted to T still belongs to the CC run; a threshold of 3 keeps CC, removes CCC
    read, mods = ct_read()
    br.configure_daf_run_mask(3, 'drop')
    codes = br.encode_from_query_sequence(read, mods, 0, mode='daf', strand='+', context_size=3)
    assert targets(codes) == [4, 10, 11] and hits(codes) == [4, 11]


def test_keep_one_keeps_the_five_prime_target_of_each_run():
    # CT strand: 5' is leftmost in SEQ -> keep 10 of CC and 18 of CCC; C11's hit is dropped
    read, mods = ct_read()
    br.configure_daf_run_mask(2)          # default policy keep-one
    assert br.daf_run_mask_policy() == 'keep-one'
    codes = br.encode_from_query_sequence(read, mods, 0, mode='daf', strand='+', context_size=3)
    assert targets(codes) == [4, 10, 18] and hits(codes) == [4]
    # GA strand: 5' of the bottom strand is rightmost in SEQ -> keep 11 of GG and 20 of GGG
    read, mods = ga_read()
    codes = br.encode_from_query_sequence(read, mods, 0, mode='daf', strand='-', context_size=3)
    assert targets(codes) == [4, 11, 20] and hits(codes) == [4]


def test_keep_one_retains_the_kept_site_own_observation():
    read, mods = ct_read()
    off = br.encode_from_query_sequence(read, mods, 0, mode='daf', strand='+', context_size=3)
    br.configure_daf_run_mask(2, 'keep-one')
    on = br.encode_from_query_sequence(read, mods, 0, mode='daf', strand='+', context_size=3)
    assert on[10] == off[10] and on[18] == off[18]


def test_unmasked_context_codes_are_unchanged():
    read, mods = ct_read()
    off = br.encode_from_query_sequence(read, mods, 0, mode='daf', strand='+', context_size=3)
    br.configure_daf_run_mask(2)
    on = br.encode_from_query_sequence(read, mods, 0, mode='daf', strand='+', context_size=3)
    assert on[4] == off[4]


def test_environment_is_inherited_and_validated(monkeypatch):
    br.configure_daf_run_mask(2, 'drop')
    import os
    assert os.environ[br._DAF_RUN_MASK_ENV] == '2' and os.environ[br._DAF_RUN_POLICY_ENV] == 'drop'
    monkeypatch.setattr(br, '_DAF_RUN_MASK_MIN', None); monkeypatch.setattr(br, '_DAF_RUN_POLICY', None)
    monkeypatch.setenv(br._DAF_RUN_MASK_ENV, '3'); monkeypatch.setenv(br._DAF_RUN_POLICY_ENV, 'keep-one')
    assert br.daf_run_mask_min_length() == 3 and br.daf_run_mask_policy() == 'keep-one'
    with pytest.raises(ValueError):
        br.configure_daf_run_mask(2, 'first')
    with pytest.raises(ValueError):
        br.configure_daf_run_mask(1)
    with pytest.raises(ValueError):
        br.configure_daf_run_mask(-2)


def test_non_daf_modes_are_unaffected():
    br.configure_daf_run_mask(2)
    seq = 'AATTACCATTGGAATTAAAATTTT'
    a = br.encode_from_query_sequence(seq, {0, 1, 2}, 0, mode='pacbio-fiber', context_size=3)
    br.configure_daf_run_mask(0)
    b = br.encode_from_query_sequence(seq, {0, 1, 2}, 0, mode='pacbio-fiber', context_size=3)
    assert np.array_equal(a, b)
