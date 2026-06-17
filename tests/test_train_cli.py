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
