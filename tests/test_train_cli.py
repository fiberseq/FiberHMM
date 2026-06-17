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
    assert captured["ml_tag"] == b"\xc8"


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
