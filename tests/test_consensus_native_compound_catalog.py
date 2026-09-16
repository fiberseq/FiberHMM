import itertools

import pytest

from fiberhmm.inference.consensus.native_compound_catalog import contiguous_compound_candidates


def calls(n, unit='u'):
    return [dict(unit_id=unit, ordinal=i, start=6*i, end=6*i+4) for i in range(n)]


def test_two_piece_feasibility_cannot_remove_broad_three_piece_candidate():
    actual = list(contiguous_compound_candidates(calls(3), {0}))
    assert [[c['ordinal'] for c in row] for row in actual] == [[0, 1], [0, 1, 2]]
    # A consumer's result for the first row has no access to the remaining
    # candidate iterator and cannot silently prune the complete catalog.


def test_exact_contiguous_catalog_for_all_small_anchor_subsets():
    for n in range(1, 7):
        source = calls(n)
        for flags in itertools.product((False, True), repeat=n):
            anchor = {i for i, enabled in enumerate(flags) if enabled}
            expected = {tuple(range(a, b)) for a in range(n) for b in range(a+2, n+1)
                        if set(range(a, b)) & anchor}
            actual = {tuple(c['ordinal'] for c in row)
                      for row in contiguous_compound_candidates(list(reversed(source)), anchor)}
            assert actual == expected


def test_no_count_cap_and_no_mutation():
    source = calls(120); snapshot = [dict(c) for c in source]
    actual = list(contiguous_compound_candidates(source, {0}))
    assert len(actual) == 119 and len(actual[-1]) == 120
    assert source == snapshot


def test_distinct_molecules_or_duplicate_source_ordinals_are_errors():
    with pytest.raises(ValueError, match='different evidence'):
        list(contiguous_compound_candidates(calls(1, 'a')+calls(1, 'b'), {0}))
    with pytest.raises(ValueError, match='unique'):
        list(contiguous_compound_candidates(calls(1)+calls(1), {0}))
