from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from fiberhmm.cli.generate_probs import (
    FILTER_STAT_KEYS,
    PROBABILITY_TSV_COLUMNS,
    _accumulate_filter_stats,
    _combined_probability_frame,
    _combined_probability_table_path,
    _context_size_label,
    _count_items_desc,
    _generate_probs_skip_reason,
    _max_reads_per_file,
    _maybe_update_probability_progress,
    _new_filter_stats,
    _new_probability_counters,
    _probability_counter_path,
    _probability_table_path,
    _print_probability_generation_header,
    _print_daf_diagnostics,
    _print_probability_base_summary,
    _probability_read_tags_or_skip,
    _probability_counter_summary,
    _print_probability_results_summary,
    _progress_postfix,
    _process_probability_sample_group,
    _process_probability_read,
    _read_limit_reached,
    _read_reference_span,
    _read_mm_ml_tags_or_skip,
    _record_filter_skip,
    _record_mm_tag_types,
    _remove_temporary_probability_counters,
    _safe_percent,
    _save_temporary_probability_counters,
    _target_bases_for_mode,
    _write_probability_tables_for_base,
    _write_probability_table,
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
    def __init__(self, tables=None):
        self.processed = []
        self.processed_daf = []
        self.saved_paths = []
        self.tables = tables or {}
        self.total_positions = 0
        self.total_modified = 0
        self.counts = {}

    def process_read(self, sequence, mod_positions, edge_trim):
        self.processed.append((sequence, mod_positions, edge_trim))

    def process_read_daf(self, sequence, mod_positions, strand, edge_trim):
        self.processed_daf.append((sequence, mod_positions, strand, edge_trim))

    def get_encoding_table(self, context_size):
        return {}, self.tables[context_size].copy()

    def save(self, path):
        self.saved_paths.append(path)


class _Progress:
    def __init__(self):
        self.postfixes = []

    def set_postfix(self, value):
        self.postfixes.append(value)


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


def test_print_probability_generation_header_lists_settings_and_inputs(capsys):
    args = SimpleNamespace(
        mode="daf",
        context_sizes=[3, 4],
        max_reads=0,
        min_mapq=20,
        min_read_length=1000,
        edge_trim=10,
        prob_threshold=128,
        accessible=["acc1.bam", "acc2.bam"],
        inaccessible=["inacc.bam"],
    )

    _print_probability_generation_header(
        args, "out", "out/tables", "out/plots",
    )

    output = capsys.readouterr().out
    assert "FiberHMM Emission Probability Generator" in output
    assert "Output directory: out" in output
    assert "Tables: out/tables/" in output
    assert "Mode: daf" in output
    assert "Context sizes: k=3 to k=4 (7-mer to 9-mer)" in output
    assert "Max reads per sample: all" in output
    assert "ML prob threshold:  128/255 (50.2%)" in output
    assert "Accessible samples (naked/dechromatinized): 2 files" in output
    assert "  - acc1.bam" in output
    assert "Inaccessible samples (untreated/native): 1 files" in output


def test_new_probability_counters_builds_context_counters():
    counters = _new_probability_counters(["A", "C"], max_context=5)

    assert sorted(counters) == ["A", "C"]
    assert counters["A"].center_base == "A"
    assert counters["A"].max_context == 5
    assert counters["C"].center_base == "C"


def test_process_probability_sample_group_prints_and_delegates(monkeypatch, capsys):
    calls = []

    def fake_process_sample_set(
        bam_files, counters, mode, args, sample_name, output_dir, base_name,
    ):
        calls.append(
            (bam_files, counters, mode, args, sample_name, output_dir, base_name)
        )
        return 7, 11, {"processed": 7}

    monkeypatch.setattr(
        "fiberhmm.cli.generate_probs.process_sample_set",
        fake_process_sample_set,
    )
    args = SimpleNamespace()

    counters, reads, scanned, stats = _process_probability_sample_group(
        section_label="ACCESSIBLE",
        sample_description="naked DNA",
        estimate_description="P(methylation | accessible)",
        bam_files=["a.bam"],
        target_bases=["A"],
        max_context=4,
        mode="pacbio-fiber",
        args=args,
        sample_name="accessible",
        output_dir="out",
        base_name="run",
    )

    assert reads == 7
    assert scanned == 11
    assert stats == {"processed": 7}
    assert counters["A"].center_base == "A"
    assert calls[0][0] == ["a.bam"]
    assert calls[0][1] is counters
    assert calls[0][2:] == ("pacbio-fiber", args, "accessible", "out", "run")
    output = capsys.readouterr().out
    assert "Processing ACCESSIBLE samples (naked DNA)" in output
    assert "This estimates P(methylation | accessible)" in output


def test_print_probability_results_summary_formats_pass_rates(capsys):
    _print_probability_results_summary(
        accessible_reads=50,
        accessible_scanned=100,
        inaccessible_reads=5,
        inaccessible_scanned=0,
    )

    output = capsys.readouterr().out
    assert "Results Summary" in output
    assert "Accessible (naked DNA):" in output
    assert "Reads processed: 50 (scanned 100, 50.0% pass rate)" in output
    assert "Inaccessible (native):" in output
    assert "Reads processed: 5 (scanned 0, 500.0% pass rate)" in output


def test_print_probability_base_summary_formats_counter_totals(capsys):
    accessible = _Counter()
    accessible.total_positions = 1234
    accessible.total_modified = 432
    accessible.counts = {"AAA": [1, 2], "AAC": [3, 4]}
    inaccessible = _Counter()
    inaccessible.total_positions = 10
    inaccessible.total_modified = 1
    inaccessible.counts = {"AAA": [1, 9]}

    _print_probability_base_summary("A", accessible, inaccessible)

    output = capsys.readouterr().out
    assert "A-centered contexts:" in output
    assert "Accessible:" in output
    assert "Positions: 1,234" in output
    assert "Modified: 432" in output
    assert "Rate: 0.3501" in output
    assert "Unique contexts: 2" in output
    assert "Inaccessible:" in output
    assert "Positions: 10" in output
    assert "Rate: 0.1000" in output


def test_record_mm_tag_types_counts_non_empty_specs():
    counts = defaultdict(int)

    _record_mm_tag_types('C+m,0,1;G-a?,2;A+a;', counts)
    _record_mm_tag_types('C+m,3;', counts)

    assert dict(counts) == {
        'C+m': 2,
        'G-a?': 1,
        'A+a': 1,
    }


def test_safe_percent_uses_one_as_zero_denominator_floor():
    assert _safe_percent(25, 100) == 25.0
    assert _safe_percent(0, 0) == 0.0
    assert _safe_percent(1, 0) == 100.0


def test_record_filter_skip_increments_when_reason_is_present():
    stats = defaultdict(int)

    assert not _record_filter_skip(stats, None)
    assert stats == {}
    assert _record_filter_skip(stats, "low_mapq")
    assert stats["low_mapq"] == 1


def test_probability_read_tags_or_skip_returns_tags_and_records_filter_failures():
    stats = defaultdict(int)
    read = _Read({"MM": "A+a,0;", "ML": [200]})

    assert _probability_read_tags_or_skip(read, stats, 20, 100) == (
        "A+a,0;",
        [200],
    )
    assert stats == {}

    read = _Read({"MM": "A+a,0;", "ML": [200]})
    read.mapping_quality = 10
    assert _probability_read_tags_or_skip(read, stats, 20, 100) is None
    assert stats["low_mapq"] == 1

    assert _probability_read_tags_or_skip(
        _Read({"MM": "A+a,0;"}), stats, 20, 100,
    ) is None
    assert stats["no_ml_tag"] == 1


def test_remove_temporary_probability_counters_removes_expected_files(tmp_path):
    output_dir = str(tmp_path)
    base_name = "sample"
    paths = [
        Path(
            _probability_counter_path(
                output_dir,
                base_name,
                sample,
                "A",
                temporary=True,
            )
        )
        for sample in ("accessible", "inaccessible")
    ]
    for path in paths:
        path.write_text("counter")
    keep = tmp_path / "sample_accessible_A_counts.pkl"
    keep.write_text("counter")

    _remove_temporary_probability_counters(output_dir, base_name, ["A"])

    assert all(not path.exists() for path in paths)
    assert keep.exists()


def test_save_temporary_probability_counters_uses_probability_paths():
    counter_a = _Counter()
    counter_c = _Counter()

    _save_temporary_probability_counters(
        {"A": counter_a, "C": counter_c},
        "out",
        "sample",
        "accessible",
    )

    assert counter_a.saved_paths == ["out/sample_accessible_A.probs.pkl.tmp"]
    assert counter_c.saved_paths == ["out/sample_accessible_C.probs.pkl.tmp"]


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


def test_maybe_update_probability_progress_updates_every_5000_reads():
    pbar = _Progress()

    _maybe_update_probability_progress(pbar, reads_processed=0, reads_scanned=0)
    _maybe_update_probability_progress(pbar, reads_processed=10, reads_scanned=4999)
    assert pbar.postfixes == []

    _maybe_update_probability_progress(pbar, reads_processed=2500, reads_scanned=5000)
    assert pbar.postfixes == [
        {"processed": "2,500", "scanned": "5,000", "rate": "50.0%"},
    ]


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


def test_read_limit_reached_treats_zero_as_unlimited():
    assert not _read_limit_reached(0, 10_000)
    assert not _read_limit_reached(10, 9)
    assert _read_limit_reached(10, 10)


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


def test_write_probability_table_uses_stable_probability_columns(tmp_path):
    probs = pd.DataFrame({
        'context': ['AAA'],
        'ratio': [0.25],
        'encode': [7],
        'hit': [1],
        'nohit': [3],
        'ignored': ['x'],
    })
    output = tmp_path / 'probs.tsv'

    _write_probability_table(probs, str(output))

    assert PROBABILITY_TSV_COLUMNS == (
        'encode', 'context', 'hit', 'nohit', 'ratio',
    )
    assert output.read_text() == (
        "encode\tcontext\thit\tnohit\tratio\n"
        "7\tAAA\t1\t3\t0.25\n"
    )


def test_write_probability_tables_for_base_writes_each_context(monkeypatch, capsys):
    acc_k3 = pd.DataFrame({
        "encode": [0, 1],
        "context": ["AAA", "AAT"],
        "hit": [3, 4],
        "nohit": [7, 6],
        "ratio": [0.3, 0.4],
    })
    inacc_k3 = pd.DataFrame({
        "encode": [0],
        "context": ["AAA"],
        "hit": [1],
        "nohit": [9],
        "ratio": [0.1],
    })
    acc_k4 = acc_k3.head(1)
    inacc_k4 = inacc_k3.head(0)
    accessible = _Counter(tables={3: acc_k3, 4: acc_k4})
    inaccessible = _Counter(tables={3: inacc_k3, 4: inacc_k4})
    writes = []

    def fake_write_probability_table(probs, output_path):
        writes.append((probs, output_path))

    monkeypatch.setattr(
        "fiberhmm.cli.generate_probs._write_probability_table",
        fake_write_probability_table,
    )

    _write_probability_tables_for_base(
        "out/tables", "run", "A", [3, 4], accessible, inaccessible,
    )

    assert [path for _, path in writes] == [
        "out/tables/run_accessible_A_k3.tsv",
        "out/tables/run_inaccessible_A_k3.tsv",
        "out/tables/run_accessible_A_k4.tsv",
        "out/tables/run_inaccessible_A_k4.tsv",
    ]
    pd.testing.assert_frame_equal(writes[0][0], acc_k3)
    pd.testing.assert_frame_equal(writes[1][0], inacc_k3)
    output = capsys.readouterr().out
    assert "Generating TSV files for k=3 to k=4" in output
    assert "k=3 (7-mer): 2 accessible, 1 inaccessible contexts" in output
    assert "k=4 (9-mer): 1 accessible, 0 inaccessible contexts" in output


def test_probability_counter_summary_reports_totals_rate_and_contexts():
    counter = _Counter()
    counter.total_positions = 100
    counter.total_modified = 25
    counter.counts = {"AAA": [1, 2], "AAC": [3, 4]}

    assert _probability_counter_summary(counter) == {
        "positions": 100,
        "modified": 25,
        "rate": 0.25,
        "unique_contexts": 2,
    }

    counter.total_positions = 0
    counter.total_modified = 1
    counter.counts = {}
    assert _probability_counter_summary(counter)["rate"] == 1.0


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
