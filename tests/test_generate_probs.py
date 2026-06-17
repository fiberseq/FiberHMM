from collections import defaultdict

import pandas as pd

from fiberhmm.cli.generate_probs import (
    FILTER_STAT_KEYS,
    _accumulate_filter_stats,
    _combined_probability_frame,
    _combined_probability_table_path,
    _generate_probs_skip_reason,
    _max_reads_per_file,
    _new_filter_stats,
    _probability_counter_path,
    _probability_table_path,
    _read_reference_span,
    _record_mm_tag_types,
    _target_bases_for_mode,
)


class _Read:
    is_unmapped = False
    is_secondary = False
    is_supplementary = False
    mapping_quality = 60
    query_sequence = "A" * 200
    reference_start = 100
    reference_end = 300


def test_new_filter_stats_returns_zeroed_independent_stats():
    stats = _new_filter_stats()

    assert tuple(stats) == FILTER_STAT_KEYS
    assert all(value == 0 for value in stats.values())

    stats['scanned'] = 10
    assert _new_filter_stats()['scanned'] == 0


def test_generate_probs_skip_reason_filters_in_expected_order():
    read = _Read()
    read.is_unmapped = True
    read.is_secondary = True

    assert _generate_probs_skip_reason(read, 20, 100) == 'unmapped'

    read = _Read()
    read.mapping_quality = 10
    assert _generate_probs_skip_reason(read, 20, 100) == 'low_mapq'

    read = _Read()
    read.reference_end = 150
    assert _generate_probs_skip_reason(read, 20, 100) == 'short_read'

    read = _Read()
    read.reference_start = None
    assert _generate_probs_skip_reason(read, 20, 100) == 'no_sequence'

    read = _Read()
    assert _generate_probs_skip_reason(read, 20, 100) is None


def test_read_reference_span_handles_missing_coordinates():
    assert _read_reference_span(_Read()) == 200

    read = _Read()
    read.reference_end = None
    assert _read_reference_span(read) is None


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


def test_probability_output_path_helpers():
    assert _probability_counter_path("out", "run", "accessible", "A") == (
        "out/run_accessible_A.probs.pkl"
    )
    assert _probability_counter_path(
        "out",
        "run",
        "inaccessible",
        "C",
        temporary=True,
    ) == "out/run_inaccessible_C.probs.pkl.tmp"
    assert _probability_table_path("out/tables", "run", "accessible", "A", 3) == (
        "out/tables/run_accessible_A_k3.tsv"
    )
    assert _combined_probability_table_path("out/tables", "run", "C", 5) == (
        "out/tables/run_C_k5_probs.tsv"
    )


def test_sample_set_bookkeeping_helpers():
    assert _max_reads_per_file(0, 3) == 0
    assert _max_reads_per_file(10, 3) == 3

    combined = defaultdict(int, {"scanned": 5, "processed": 2})
    _accumulate_filter_stats(combined, {"scanned": 4, "low_mapq": 1})

    assert dict(combined) == {
        "scanned": 9,
        "processed": 2,
        "low_mapq": 1,
    }
