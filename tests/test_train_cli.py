from types import SimpleNamespace

import pytest
import numpy as np

from fiberhmm.cli import train
from fiberhmm.core import bam_reader

pysam = pytest.importorskip("pysam")


def test_state_runs_groups_contiguous_states():
    assert train._state_runs([0, 0, 1, 1, 1, 0]) == [
        (0, 2, 0),
        (2, 5, 1),
        (5, 6, 0),
    ]
    assert train._state_runs([]) == []


def test_state_run_lengths_splits_footprint_and_msp_lengths():
    footprint_sizes, msp_sizes = train._state_run_lengths([0, 0, 1, 1, 1, 0])

    assert footprint_sizes == [2, 1]
    assert msp_sizes == [3]


def test_state_block_specs_include_plot_widths_and_colors():
    assert train._state_block_specs([]) == []
    assert train._state_block_specs([0, 0, 1, 0]) == [
        (0, 2, "forestgreen"),
        (2, 1, "white"),
        (3, 1, "forestgreen"),
    ]


def test_expected_model_durations_from_self_transitions():
    durations = train._expected_model_durations(np.array([[0.8, 0.2], [0.1, 1.0]]))

    assert durations[0] == pytest.approx(5.0)
    assert durations[1] == float('inf')


def test_viterbi_state_size_stats_skips_empty_reads():
    class FakeModel:
        def predict(self, encoded):
            return encoded

    footprint_sizes, msp_sizes, states = train._viterbi_state_size_stats(
        FakeModel(),
        [
            np.array([0, 0, 1, 1]),
            np.array([], dtype=int),
            np.array([1, 0, 0]),
        ],
    )

    assert footprint_sizes == [2, 2]
    assert msp_sizes == [2, 1]
    assert len(states) == 2
    np.testing.assert_array_equal(states[0], [0, 0, 1, 1])
    np.testing.assert_array_equal(states[1], [1, 0, 0])


def test_nonzero_emissions_by_state_filters_zero_entries():
    fp, msp = train._nonzero_emissions_by_state(
        np.array([[0.0, 0.2, 0.0, 0.5], [0.1, 0.0, 0.3, 0.0]])
    )

    np.testing.assert_array_equal(fp, [0.2, 0.5])
    np.testing.assert_array_equal(msp, [0.1, 0.3])


def test_shuffled_training_arrays_are_deterministic_int_arrays():
    encoded_reads = [
        np.array([1, 1], dtype=np.int64),
        np.array([2], dtype=np.int64),
        np.array([3, 3], dtype=np.int64),
    ]

    first = train._shuffled_training_arrays(encoded_reads, n_iterations=3)
    second = train._shuffled_training_arrays(encoded_reads, n_iterations=3)

    assert list(first) == [0, 1, 2]
    for iteration, values in first.items():
        np.testing.assert_array_equal(values, second[iteration])
        assert np.issubdtype(values.dtype, np.integer)
        assert sorted(values.tolist()) == [1, 1, 2, 3, 3]


def test_training_strand_for_read_detects_only_daf_mode(monkeypatch):
    read = SimpleNamespace(query_sequence="ACGT", m6a_query_positions={1})

    monkeypatch.setattr(train, "detect_daf_strand", lambda seq, positions: "-")
    assert train._training_strand_for_read(read, "daf") == "-"
    assert train._training_strand_for_read(read, "pacbio-fiber") == "."
    assert train._training_strand_for_read(read, "nanopore-fiber") == "."


def test_encode_training_read_passes_mode_strand_context_and_reverse(monkeypatch):
    read = SimpleNamespace(
        query_sequence="ACGT",
        m6a_query_positions={1},
        is_reverse=True,
    )
    captured = {}

    monkeypatch.setattr(train, "_training_strand_for_read", lambda fiber_read, mode: "+")

    def fake_encode(sequence, mod_positions, edge_trim, **kwargs):
        captured.update(
            sequence=sequence,
            mod_positions=mod_positions,
            edge_trim=edge_trim,
            kwargs=kwargs,
        )
        return np.array([1, 2, 3], dtype=np.int32)

    monkeypatch.setattr(train, "encode_from_query_sequence", fake_encode)

    encoded = train._encode_training_read(
        read,
        edge_trim=12,
        mode="daf",
        context_size=5,
    )

    np.testing.assert_array_equal(encoded, [1, 2, 3])
    assert captured == {
        "sequence": "ACGT",
        "mod_positions": {1},
        "edge_trim": 12,
        "kwargs": {
            "mode": "daf",
            "strand": "+",
            "context_size": 5,
            "is_reverse": True,
        },
    }


def test_training_zoom_window_starts_are_deterministic_and_bounded():
    assert train._training_zoom_window_starts(
        seq_len=500, window_size=1000, n_windows=3, seed=0,
    ) == [0, 0, 0]

    starts = train._training_zoom_window_starts(
        seq_len=6000, window_size=1000, n_windows=3, seed=7,
    )
    assert starts == train._training_zoom_window_starts(
        seq_len=6000, window_size=1000, n_windows=3, seed=7,
    )
    assert len(starts) == 3
    assert all(0 <= start <= 5000 for start in starts)
    assert starts[0] < 2000
    assert 2000 <= starts[1] < 4000
    assert 4000 <= starts[2] <= 5000


def test_training_size_summary_formats_nonempty_and_empty_sizes():
    assert train._training_size_summary(
        "footprints", "Footprint sizes", [10, 30],
    ) == (
        "  Total footprints: 2\n"
        "  Footprint sizes: mean=20.0, median=20.0, range=10-30\n"
    )
    assert train._training_size_summary("MSPs", "MSP sizes", []) == ""


def test_training_example_plot_data_sorts_positions_and_uses_footprint_probs():
    class FakeModel:
        def predict_proba(self, encoded):
            np.testing.assert_array_equal(encoded, [3, 4])
            return np.array([[0.2, 0.8], [0.7, 0.3]])

    read = SimpleNamespace(query_sequence="AC", m6a_query_positions={1, 0})

    seq_len, m6a_positions, footprint_prob = train._training_example_plot_data(
        FakeModel(), read, np.array([3, 4]),
    )

    assert seq_len == 2
    assert m6a_positions == [0, 1]
    np.testing.assert_array_equal(footprint_prob, [0.2, 0.7])


def test_add_training_state_blocks_uses_state_specs_and_sets_limits():
    rectangles = []
    collections = []

    class FakeRectangle:
        def __init__(self, xy, width, height):
            self.xy = xy
            self.width = width
            self.height = height
            rectangles.append(self)

    class FakePatchCollection:
        def __init__(self, patches, **kwargs):
            self.patches = patches
            self.kwargs = kwargs
            collections.append(self)

    class FakeAxis:
        def __init__(self):
            self.added = []
            self.xlim = None
            self.ylim = None

        def add_collection(self, collection):
            self.added.append(collection)

        def set_xlim(self, start, end):
            self.xlim = (start, end)

        def set_ylim(self, start, end):
            self.ylim = (start, end)

    ax = FakeAxis()
    train._add_training_state_blocks(
        ax,
        [0, 0, 1],
        10,
        FakeRectangle,
        FakePatchCollection,
    )

    assert [(rect.xy, rect.width, rect.height) for rect in rectangles] == [
        ((0, 0), 2, 1),
        ((2, 0), 1, 1),
    ]
    assert collections[0].patches == rectangles
    assert collections[0].kwargs == {
        "facecolors": ["forestgreen", "white"],
        "edgecolors": "lightgray",
        "linewidths": 0.3,
    }
    assert ax.added == collections
    assert ax.xlim == (0, 10)
    assert ax.ylim == (0, 1)


def test_training_stats_paths_are_under_plots_dir():
    assert train._training_stats_paths("/tmp/out") == {
        "plots_dir": "/tmp/out/plots",
        "pdf": "/tmp/out/plots/training_stats.pdf",
        "summary": "/tmp/out/plots/training_stats.txt",
    }


def test_write_training_stats_summary(tmp_path):
    model = SimpleNamespace(
        transmat_=np.array([[0.8, 0.2], [0.25, 0.75]]),
    )
    summary_path = tmp_path / "training_stats.txt"

    train._write_training_stats_summary(
        str(summary_path),
        model,
        emission_probs=np.array([[0.0, 0.2, 0.4], [0.1, 0.0, 0.9]]),
        sampled_reads=[object(), object()],
        all_footprint_sizes=[10, 30],
        all_msp_sizes=[20],
    )

    text = summary_path.read_text()
    assert "FiberHMM Training Statistics" in text
    assert "Footprint → Footprint: 0.800000" in text
    assert "Expected footprint duration: 5.0 bp" in text
    assert "Footprint contexts: 2 (mean=0.3000, median=0.3000)" in text
    assert "Reads used: 2" in text
    assert "Footprint sizes: mean=20.0, median=20.0, range=10-30" in text
    assert "MSP sizes: mean=20.0, median=20.0, range=20-20" in text


def test_training_config_includes_base_model_only_when_used():
    args = SimpleNamespace(
        context_size=3,
        mode="daf",
        edge_trim=10,
        prob_adjust=1.25,
        base_model=None,
    )

    assert train._training_config(args) == {
        "context_size": 3,
        "mode": "daf",
        "edge_trim": 10,
        "prob_adjust": 1.25,
    }

    args.base_model = "base-model.json"
    assert train._training_config(args)["base_model"] == "base-model.json"


def test_training_output_paths_use_canonical_filenames():
    assert train._training_output_paths("/tmp/out") == {
        "best_json": "/tmp/out/best-model.json",
        "best_npz": "/tmp/out/best-model.npz",
        "all_models": "/tmp/out/all_models.json",
        "train_reads": "/tmp/out/training-reads.tsv",
        "config": "/tmp/out/model_config.json",
    }


def test_model_json_record_uses_plain_lists():
    model = SimpleNamespace(
        n_states=2,
        startprob_=np.array([0.4, 0.6]),
        transmat_=np.array([[0.9, 0.1], [0.2, 0.8]]),
        emissionprob_=np.array([[0.1, 0.2], [0.8, 0.9]]),
    )

    record = train._model_json_record(model)

    assert record == {
        "n_states": 2,
        "startprob": [0.4, 0.6],
        "transmat": [[0.9, 0.1], [0.2, 0.8]],
        "emissionprob": [[0.1, 0.2], [0.8, 0.9]],
    }


def test_training_chrom_is_sampleable_filters_short_and_scaffold_names():
    assert train._training_chrom_is_sampleable("chr2L", 1_000_000)
    assert not train._training_chrom_is_sampleable("tiny", 99_999)
    assert not train._training_chrom_is_sampleable("chrUn_random", 1_000_000)
    assert not train._training_chrom_is_sampleable("CHR_ALT_1", 1_000_000)


def test_training_sampling_chrom_lengths_filters_scaffolds_and_falls_back():
    ref_lengths = {
        "chr1": 1_000_000,
        "chrUn_random": 2_000_000,
        "tiny": 10_000,
    }

    assert train._training_sampling_chrom_lengths(ref_lengths) == {"chr1": 1_000_000}
    assert train._training_sampling_chrom_lengths({"tiny": 10_000}) == {
        "tiny": 10_000,
    }


def test_chrom_pos_from_genome_offset_maps_cumulative_lengths():
    chroms = ["chr1", "chr2", "chr3"]
    cum_lengths = np.array([100, 250, 300])

    assert train._chrom_pos_from_genome_offset(chroms, cum_lengths, 50) == (
        "chr1", 50,
    )
    assert train._chrom_pos_from_genome_offset(chroms, cum_lengths, 125) == (
        "chr2", 25,
    )


def test_training_sample_filter_helper_applies_alignment_filters():
    def read(**overrides):
        attrs = {
            "is_unmapped": False,
            "is_secondary": False,
            "is_supplementary": False,
            "mapping_quality": 30,
            "query_sequence": "ACGT",
            "reference_start": 10,
            "reference_end": 14,
        }
        attrs.update(overrides)
        return SimpleNamespace(**attrs)

    assert train._passes_training_sample_filters(read(), 20, 4) is True
    assert train._passes_training_sample_filters(
        read(mapping_quality=19), 20, 4,
    ) is False
    assert train._passes_training_sample_filters(
        read(query_sequence=None), 20, 4,
    ) is False
    assert train._passes_training_sample_filters(
        read(reference_end=13), 20, 4,
    ) is False
    assert train._training_reference_span(read(reference_start=None)) is None
    assert train._passes_training_sample_filters(
        read(reference_start=None), 20, 4,
    ) is False
    assert train._passes_training_sample_filters(
        read(is_supplementary=True), 20, 4,
    ) is False


def test_training_mod_query_positions_prefers_pysam_then_mm_fallback(monkeypatch):
    class TaggedRead:
        query_sequence = "AAAA"
        is_reverse = False

        def __init__(self, tags):
            self.tags = tags

        def has_tag(self, tag):
            return tag in self.tags

        def get_tag(self, tag):
            return self.tags[tag]

    read = TaggedRead({"MM": "A+a,0;", "ML": b"\xc8"})

    monkeypatch.setattr(bam_reader, "get_modified_positions_pysam", lambda *args: {3})
    monkeypatch.setattr(
        bam_reader,
        "parse_mm_tag_query_positions",
        lambda *args, **kwargs: pytest.fail("MM/ML fallback should not run"),
    )
    assert train._training_mod_query_positions(read, 125, "pacbio-fiber") == {3}

    captured = {}

    def fake_parse(mm_tag, ml_tag, sequence, is_reverse, prob_threshold, mode):
        captured["ml_tag"] = ml_tag
        return {1}

    monkeypatch.setattr(bam_reader, "get_modified_positions_pysam", lambda *args: set())
    monkeypatch.setattr(bam_reader, "parse_mm_tag_query_positions", fake_parse)

    assert train._training_mod_query_positions(read, 125, "pacbio-fiber") == {1}
    assert train._training_mm_ml_query_positions(read, 125, "pacbio-fiber") == {1}
    assert captured["ml_tag"] == b"\xc8"


def test_training_sample_candidate_filters_duplicates_and_requires_mods(monkeypatch):
    read = SimpleNamespace(query_name="read1")
    fiber = object()

    monkeypatch.setattr(
        train,
        "_passes_training_sample_filters",
        lambda got_read, min_mapq, min_read_length: got_read is read,
    )
    monkeypatch.setattr(
        train,
        "_training_mod_query_positions",
        lambda got_read, prob_threshold, mode: {2},
    )
    monkeypatch.setattr(
        train,
        "_training_fiber_read_from_segment",
        lambda got_read, mods: fiber,
    )

    assert train._training_sample_candidate(
        read, set(), 20, 1000, 125, "pacbio-fiber",
    ) is fiber
    assert train._training_sample_candidate(
        read, {"read1"}, 20, 1000, 125, "pacbio-fiber",
    ) is None
    assert train._training_sample_candidate(
        SimpleNamespace(query_name="read2"), set(), 20, 1000, 125, "pacbio-fiber",
    ) is None

    monkeypatch.setattr(
        train,
        "_training_mod_query_positions",
        lambda got_read, prob_threshold, mode: set(),
    )
    assert train._training_sample_candidate(
        read, set(), 20, 1000, 125, "pacbio-fiber",
    ) is None


def test_reads_per_training_file_has_one_read_minimum():
    assert train._reads_per_training_file(100, 4) == 25
    assert train._reads_per_training_file(2, 5) == 1


def test_top_up_training_reads_truncates_or_samples_with_replacement():
    assert train._top_up_training_reads([], 3) == []
    assert train._top_up_training_reads(["a", "b", "c"], 2) == ["a", "b"]

    np.random.seed(0)
    topped = train._top_up_training_reads(["a", "b"], 5)

    assert len(topped) == 5
    assert topped[:2] == ["a", "b"]
    assert set(topped) <= {"a", "b"}


def test_build_model_from_base_copies_transitions_and_replaces_emissions(monkeypatch):
    base_model = SimpleNamespace(
        startprob_=np.array([0.7, 0.3]),
        transmat_=np.array([[0.95, 0.05], [0.1, 0.9]]),
        emissionprob_=np.zeros((2, 4)),
    )
    emission_probs = np.full((2, 4), 0.25)

    monkeypatch.setattr(
        train,
        "load_model",
        lambda path, normalize=False: base_model,
    )

    model, all_models = train._build_model_from_base(
        "base-model.json",
        emission_probs,
        context_size=3,
    )

    assert all_models == [model]
    np.testing.assert_array_equal(model.startprob_, [0.7, 0.3])
    np.testing.assert_array_equal(model.transmat_, [[0.95, 0.05], [0.1, 0.9]])
    assert model.emissionprob_ is emission_probs

    base_model.startprob_[0] = 0.1
    base_model.transmat_[0, 0] = 0.1
    assert model.startprob_[0] == 0.7
    assert model.transmat_[0, 0] == 0.95


def test_sample_reads_indexed_preserves_reverse_flag(monkeypatch):
    class FakeRead:
        is_unmapped = False
        is_secondary = False
        is_supplementary = False
        mapping_quality = 60
        query_sequence = "ACGT" * 300
        reference_start = 100
        reference_end = reference_start + len(query_sequence)
        query_name = "reverse-read"
        reference_name = "chr1"
        is_reverse = True

        def has_tag(self, tag):
            return False

        def get_tag(self, tag):
            raise KeyError(tag)

        def get_aligned_pairs(self):
            return [
                (query_pos, self.reference_start + query_pos)
                for query_pos in range(len(self.query_sequence))
            ]

    class FakeBam:
        references = ("chr1",)
        lengths = (2000,)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def fetch(self, chrom, start, end):
            return iter([FakeRead()])

    monkeypatch.setattr(pysam, "AlignmentFile", lambda *args, **kwargs: FakeBam())
    monkeypatch.setattr(bam_reader, "get_modified_positions_pysam", lambda *args, **kwargs: {5})

    reads = train.sample_reads_indexed(
        "fake.bam",
        n_samples=1,
        seed=1,
        min_mapq=0,
        min_read_length=0,
    )

    assert len(reads) == 1
    assert reads[0].strand == "-"
    assert reads[0].is_reverse is True
