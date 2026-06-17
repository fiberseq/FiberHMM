from collections import defaultdict

import pandas as pd

from fiberhmm.cli.generate_probs import (
    FILTER_STAT_KEYS,
    _accumulate_filter_stats,
    _combined_probability_frame,
    _combined_probability_table_path,
    _context_size_label,
    _count_items_desc,
    _generate_probs_skip_reason,
    _max_reads_per_file,
    _new_filter_stats,
    _probability_counter_path,
    _probability_table_path,
    _print_daf_diagnostics,
    _progress_postfix,
    _process_probability_read,
    _read_reference_span,
    _read_mm_ml_tags_or_skip,
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
    is_reverse = False

    def __init__(self, tags=None):
        self.tags = tags or {}

    def has_tag(self, tag):
        return tag in self.tags

    def get_tag(self, tag):
        if tag not in self.tags:
            raise KeyError(tag)
        return self.tags[tag]


class _Counter:
    def __init__(self):
        self.processed = []
        self.processed_daf = []

    def process_read(self, sequence, mod_positions, edge_trim):
        self.processed.append((sequence, mod_positions, edge_trim))

    def process_read_daf(self, sequence, mod_positions, strand, edge_trim):
        self.processed_daf.append((sequence, mod_positions, strand, edge_trim))


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


def test_read_mm_ml_tags_or_skip_prefers_supported_tag_names():
    read = _Read({"MM": "A+a,0;", "ML": [200]})
    assert _read_mm_ml_tags_or_skip(read) == ("A+a,0;", [200], None)

    read = _Read({"Mm": "C+m,0;", "Ml": (150,)})
    assert _read_mm_ml_tags_or_skip(read) == ("C+m,0;", [150], None)

    assert _read_mm_ml_tags_or_skip(_Read({"ML": [200]})) == (
        None,
        None,
        "no_mm_tag",
    )
    assert _read_mm_ml_tags_or_skip(_Read({"MM": "A+a,0;"})) == (
        "A+a,0;",
        None,
        "no_ml_tag",
    )


def test_target_bases_for_mode():
    assert _target_bases_for_mode('pacbio-fiber') == ['A']
    assert _target_bases_for_mode('nanopore-fiber') == ['A']
    assert _target_bases_for_mode('daf') == ['C']


def test_context_size_label_formats_single_and_range_variants():
    assert _context_size_label([3]) == "k=3 (7-mer)"
    assert _context_size_label([3], include_mer_span=False) == "k=3"
    assert _context_size_label([6, 3, 4]) == "k=3 to k=6 (7-mer to 13-mer)"
    assert _context_size_label([6, 3, 4], include_mer_span=False) == "k=3 to k=6"


def test_record_mm_tag_types_counts_non_empty_specs():
    counts = defaultdict(int)

    _record_mm_tag_types('C+m,0,1;G-a?,2;A+a;', counts)
    _record_mm_tag_types('C+m,3;', counts)

    assert dict(counts) == {
        'C+m': 2,
        'G-a?': 1,
        'A+a': 1,
    }


def test_process_probability_read_updates_target_counter(monkeypatch):
    monkeypatch.setattr(
        "fiberhmm.cli.generate_probs.parse_mm_tag_query_positions",
        lambda *args, **kwargs: {2, 4},
    )
    monkeypatch.setattr(
        "fiberhmm.cli.generate_probs.detect_strand_and_base",
        lambda *args, **kwargs: ("+", "A"),
    )
    counter = _Counter()
    mm_tag_types = defaultdict(int)
    strand_assignments = defaultdict(int)

    _process_probability_read(
        _Read(), {"A": counter}, "pacbio-fiber", 125, 9,
        "A+a,0;", [200], mm_tag_types, strand_assignments,
    )

    assert dict(mm_tag_types) == {"A+a": 1}
    assert dict(strand_assignments) == {"+:A": 1}
    assert counter.processed == [("A" * 200, {2, 4}, 9)]
    assert counter.processed_daf == []


def test_process_probability_read_routes_daf_to_c_counter(monkeypatch):
    monkeypatch.setattr(
        "fiberhmm.cli.generate_probs.parse_mm_tag_query_positions",
        lambda *args, **kwargs: {5},
    )
    monkeypatch.setattr(
        "fiberhmm.cli.generate_probs.detect_strand_and_base",
        lambda *args, **kwargs: ("-", "G"),
    )
    counter = _Counter()
    strand_assignments = defaultdict(int)

    _process_probability_read(
        _Read(), {"C": counter}, "daf", 125, 11,
        "G-a,0;", [220], defaultdict(int), strand_assignments,
    )

    assert dict(strand_assignments) == {"-:G": 1}
    assert counter.processed == []
    assert counter.processed_daf == [("A" * 200, {5}, "-", 11)]


def test_progress_postfix_formats_counts_and_rate():
    assert _progress_postfix(1234, 5000) == {
        "processed": "1,234",
        "scanned": "5,000",
        "rate": "24.7%",
    }
    assert _progress_postfix(0, 0)["rate"] == "0.0%"


def test_count_items_desc_sorts_by_count_descending():
    assert _count_items_desc({"low": 1, "high": 3, "mid": 2}) == [
        ("high", 3),
        ("mid", 2),
        ("low", 1),
    ]


def test_print_daf_diagnostics_formats_counts(capsys):
    _print_daf_diagnostics(
        {"C+m": 1000, "G-a": 2},
        {"+:C": 3, "-:G": 4000},
    )

    out = capsys.readouterr().out
    assert "MM tag modification types found" in out
    assert "C+m: 1,000" in out
    assert "Strand assignments" in out
    assert "-:G: 4,000" in out


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
