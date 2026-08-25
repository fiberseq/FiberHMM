"""Tests for endpoint-constrained deamination-fingerprint deduplication."""

import numpy as np
import pysam
import pytest

from fiberhmm.cli.dedup import cluster_reads, run_dedup


A_SITES = list(range(0, 100, 2))
B_SITES = list(range(100, 200, 2))


def _seq_with_y(sites):
    sequence = ["A"] * 200
    for position in sites:
        sequence[position] = "Y"
    return "".join(sequence)


def _make_bam(path, reads):
    header = pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": "chr1", "LN": 1000}],
        }
    )
    with pysam.AlignmentFile(str(path), "wb", header=header) as output:
        for name, sites, is_reverse, mapq in reads:
            read = pysam.AlignedSegment(header)
            read.query_name = name
            read.flag = 16 if is_reverse else 0
            read.reference_id = 0
            read.reference_start = 0
            read.mapping_quality = mapq
            read.cigarstring = "200M"
            read.query_sequence = _seq_with_y(sites)
            read.query_qualities = pysam.qualitystring_to_array("I" * 200)
            output.write(read)


def _near(sites, swaps):
    result = list(sites)
    for index in range(swaps):
        result[index] += 1
    return result


def test_fingerprint_match_requires_similar_alignment_ends():
    fingerprint = frozenset(range(100, 200, 3))
    labels = cluster_reads(
        [fingerprint, fingerprint, fingerprint],
        [(0, "+"), (0, "+"), (0, "+")],
        min_jaccard=0.95,
        k=32,
        bands=8,
        seed=7,
        endpoints=[(1_000, 2_000), (1_030, 2_040), (1_500, 2_500)],
        max_end_diff=50,
    )

    assert labels[0] == labels[1]
    assert labels[2] != labels[0]
    assert np.all(labels >= 0)


def test_cluster_separates_molecules_and_tolerates_near_copies():
    pos_sets = (
        [frozenset(A_SITES)] * 4
        + [frozenset(_near(A_SITES, 1))]
        + [frozenset(B_SITES)] * 3
    )
    keys = [("c", "+")] * len(pos_sets)
    labels = cluster_reads(pos_sets, keys, 0.95, 32, 8, 7)
    assert len(set(labels[:5])) == 1
    assert len(set(labels[5:])) == 1
    assert labels[0] != labels[5]


def test_strand_grouping_is_enforced_by_group_key():
    pos_sets = [frozenset(A_SITES), frozenset(A_SITES)]
    separate = cluster_reads(
        pos_sets, [("c", "+"), ("c", "-")], 0.95, 32, 8, 7
    )
    together = cluster_reads(
        pos_sets, [("c", ""), ("c", "")], 0.95, 32, 8, 7
    )
    assert separate[0] != separate[1]
    assert together[0] == together[1]


@pytest.fixture
def daf_bam(tmp_path):
    reads = (
        [(f"A{index}", A_SITES, False, 60) for index in range(4)]
        + [("Anear", _near(A_SITES, 1), False, 60)]
        + [(f"B{index}", B_SITES, False, 60) for index in range(3)]
        + [("lowdeam", [2, 4, 6], False, 60)]
    )
    path = tmp_path / "input.bam"
    _make_bam(path, reads)
    return path


def test_flag_mode_retains_all_reads_and_marks_nonrepresentatives(
    daf_bam, tmp_path
):
    output = tmp_path / "flag.bam"
    stats = run_dedup(
        str(daf_bam), str(output), min_jaccard=0.95, min_deam=10, collapse=False
    )
    records = list(pysam.AlignmentFile(str(output), check_sq=False))
    assert stats["n_fingerprintable"] == 8
    assert stats["n_clusters"] == 2
    assert stats["n_duplicates"] == 6
    assert len(records) == 9
    assert sum(read.is_duplicate for read in records) == 6
    for read in records:
        if read.is_duplicate:
            assert read.has_tag("di")
            assert read.get_tag("ds") > 1


def test_explicit_collapse_keeps_one_representative_per_molecule(
    daf_bam, tmp_path
):
    output = tmp_path / "collapse.bam"
    run_dedup(str(daf_bam), str(output), min_jaccard=0.95, min_deam=10)
    records = list(pysam.AlignmentFile(str(output), check_sq=False))
    assert len(records) == 3
    assert not any(read.is_duplicate for read in records)
    assert "lowdeam" in {read.query_name for read in records}


def test_call_wrapper_mark_retain_mode_is_nondestructive(daf_bam):
    from fiberhmm.cli.call import _dedup_input_first

    temporary_bam, stats = _dedup_input_first(
        str(daf_bam),
        str(daf_bam.parent / "output.bam"),
        0.95,
        True,
        1,
        region_parallel=False,
    )
    records = list(pysam.AlignmentFile(temporary_bam, check_sq=False))
    assert stats["mode"] == "flagged"
    assert len(records) == 9
    assert sum(read.is_duplicate for read in records) == 6
    assert len(list(pysam.AlignmentFile(str(daf_bam), check_sq=False))) == 9


def test_call_wrapper_forwards_dedup_parameters(daf_bam, monkeypatch):
    import fiberhmm.cli.call as callmod

    captured = {}

    def fake_run_dedup(in_bam, out_bam, **kwargs):
        captured.update(kwargs)
        return {"n_clusters": 1}

    monkeypatch.setattr("fiberhmm.cli.dedup.run_dedup", fake_run_dedup)
    callmod._dedup_input_first(
        str(daf_bam),
        str(daf_bam.parent / "output.bam"),
        0.90,
        True,
        2,
        region_parallel=False,
        min_deam=25,
        prob_threshold=200,
        ignore_strand=True,
        stats_tsv="clusters.tsv",
        max_end_diff=75,
    )
    assert captured == {
        "min_jaccard": 0.90,
        "collapse": False,
        "io_threads": 2,
        "min_deam": 25,
        "prob_threshold": 200,
        "ignore_strand": True,
        "stats_tsv": "clusters.tsv",
        "max_end_diff": 75,
    }
