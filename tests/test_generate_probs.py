from collections import defaultdict

import pandas as pd

from fiberhmm.cli.generate_probs import (
    FILTER_STAT_KEYS,
    _combined_probability_frame,
    _new_filter_stats,
    _record_mm_tag_types,
    _target_bases_for_mode,
)


def test_new_filter_stats_returns_zeroed_independent_stats():
    stats = _new_filter_stats()

    assert tuple(stats) == FILTER_STAT_KEYS
    assert all(value == 0 for value in stats.values())

    stats['scanned'] = 10
    assert _new_filter_stats()['scanned'] == 0


def test_target_bases_for_mode():
    assert _target_bases_for_mode('pacbio-fiber') == ['A']
    assert _target_bases_for_mode('nanopore-fiber') == ['A']
    assert _target_bases_for_mode('daf') == ['C']


def test_record_mm_tag_types_counts_non_empty_specs():
    counts = defaultdict(int)

    _record_mm_tag_types('C+m,0,1;G-a?,2;A+a;', counts)
    _record_mm_tag_types('C+m,3;', counts)

    assert dict(counts) == {
        'C+m': 2,
        'G-a?': 1,
        'A+a': 1,
    }


def test_combined_probability_frame_outer_merges_contexts_and_fills_missing():
    accessible = pd.DataFrame({
        'context': ['TTT', 'AAA'],
        'ratio': [0.9, 0.8],
        'encode': [10, 11],
    })
    inaccessible = pd.DataFrame({
        'context': ['CCC', 'TTT'],
        'ratio': [0.1, 0.2],
        'encode': [20, 21],
    })

    combined = _combined_probability_frame(accessible, inaccessible)

    assert combined.to_dict('list') == {
        'encode': [0, 1, 2],
        'context': ['AAA', 'CCC', 'TTT'],
        'accessible_prob': [0.8, 0.0, 0.9],
        'inaccessible_prob': [0.0, 0.1, 0.2],
    }
