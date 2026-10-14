"""Tests for endpoint-constrained deamination-fingerprint deduplication."""

from itertools import permutations

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


@pytest.mark.parametrize('order', list(permutations(range(3))))
@pytest.mark.parametrize('threshold,swaps', [(0.95, 2), (0.90, 4)])
def test_bucket_bridges_use_nonrepresentatives_and_merge_all_families(
    monkeypatch, order, threshold, swaps
):
    # A--B and B--C qualify; A--C does not. Both A,B,C and A,C,B used to
    # lose the B--C edge: B was skipped, or matching stopped at one family.
    base = set(range(100))
    middle = (base - set(range(swaps))) | set(range(100, 100 + swaps))
    last = (base - set(range(2 * swaps))) | set(range(100, 100 + 2 * swaps))
    fingerprints = [frozenset(base), frozenset(middle), frozenset(last)]
    # Isolate candidate verification from the approximate hash retrieval.
    monkeypatch.setattr('fiberhmm.cli.dedup._minhash',
                        lambda pos_sets, k, seed: np.zeros((len(pos_sets), k), dtype=np.int64))
    labels = cluster_reads([fingerprints[i] for i in order], [('c', 'CT')] * 3,
                          threshold, 32, 8, 7, endpoints=[(0, 200)] * 3)
    assert len(set(labels)) == 1


def test_real_hash_bridge_regression():
    # Minimal deterministic reproduction of the Tohn 1x-brain failure:
    # B/C have Jaccard .9608 and share seven of eight candidate bands.
    p = np.random.default_rng(8).choice(12000, size=104, replace=False)
    fingerprints = [frozenset(p[:100]), frozenset(np.r_[p[:98], p[100:102]]),
                    frozenset(np.r_[p[:96], p[100:104]])]
    labels = cluster_reads(fingerprints, [('c', 'CT')] * 3, .95, 32, 8, 7,
                          endpoints=[(0, 12000)] * 3)
    assert len(set(labels)) == 1


def test_candidate_components_match_exact_graph_with_group_and_end_gates(monkeypatch):
    rng = np.random.default_rng(19)
    fingerprints = []
    for index in range(40):
        base = set(range(100 * (index % 2), 100 * (index % 2) + 100))
        remove = rng.choice(sorted(base), size=int(rng.integers(1, 6)), replace=False)
        fingerprints.append(frozenset((base - set(remove)) | {1000 + index}))
    keys = [('c', 'CT' if i % 3 else 'GA') for i in range(40)]
    endpoints = [(int(rng.integers(0, 4)) * 40, 300) for _ in fingerprints]
    monkeypatch.setattr('fiberhmm.cli.dedup._minhash',
                        lambda pos_sets, k, seed: np.zeros((len(pos_sets), k), dtype=np.int64))
    connected = np.eye(len(fingerprints), dtype=bool)
    for i, a in enumerate(fingerprints):
        for j, b in enumerate(fingerprints):
            if (keys[i] == keys[j] and abs(endpoints[i][0] - endpoints[j][0]) <= 50
                    and len(a & b) / len(a | b) >= .90):
                connected[i, j] = True
    # Independent transitive closure of the exact qualifying-pair graph.
    for k in range(len(fingerprints)):
        connected |= connected[:, k:k + 1] & connected[k:k + 1, :]
    labels = cluster_reads(fingerprints, keys, .90, 32, 8, 7, endpoints=endpoints)
    assert np.array_equal(labels[:, None] == labels[None, :], connected)


def test_identical_family_avoids_quadratic_fingerprint_comparisons(monkeypatch):
    from fiberhmm.cli import dedup

    comparisons = []
    original = dedup._jaccard

    def count_jaccard(a, b):
        comparisons.append(1)
        return original(a, b)

    monkeypatch.setattr(dedup, '_jaccard', count_jaccard)
    n = 1000
    labels = cluster_reads([frozenset(A_SITES)] * n, [('c', 'CT')] * n,
                          .95, 32, 8, 7, endpoints=[(0, 200)] * n)
    assert len(set(labels)) == 1
    assert len(comparisons) <= n


def test_optional_lower_threshold_marks_near_copies(tmp_path):
    source = tmp_path / 'near.bam'
    # 46/54 = .852 would be below both cutoffs; 48/52 = .923 distinguishes them.
    _make_bam(source, [('first', A_SITES, False, 60),
                       ('near', _near(A_SITES, 2), False, 50)])
    counts = []
    for threshold in [.95, .90]:
        output = tmp_path / f'j{threshold}.bam'
        stats = run_dedup(str(source), str(output), min_jaccard=threshold, collapse=False)
        with pysam.AlignmentFile(output) as bam:
            records = list(bam)
            assert len(records) == 2
            counts.append(sum(r.is_duplicate for r in records))
        assert stats['n_duplicates'] == counts[-1]
    assert counts == [0, 1]


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


# ---------------------------------------------------------------------------
# Grouping key: deamination flavour (CT/GA), never alignment orientation.
# PCR copies of one deaminated template strand keep its flavour but can align
# in either orientation; on NAPA DddA, orientation grouping missed ~40% of
# duplicates (130 clusters mixed orientations, 0 mixed flavours).
# ---------------------------------------------------------------------------


def _seq_with_code(sites, code):
    sequence = ["A"] * 200
    for position in sites:
        sequence[position] = code
    return "".join(sequence)


def _make_flavour_bam(path, reads):
    header = pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": "chr1", "LN": 1000}],
        }
    )
    with pysam.AlignmentFile(str(path), "wb", header=header) as output:
        for name, sites, code, is_reverse in reads:
            read = pysam.AlignedSegment(header)
            read.query_name = name
            read.flag = 16 if is_reverse else 0
            read.reference_id = 0
            read.reference_start = 0
            read.mapping_quality = 60
            read.cigarstring = "200M"
            read.query_sequence = _seq_with_code(sites, code)
            read.query_qualities = pysam.qualitystring_to_array("I" * 200)
            output.write(read)


def test_deamination_flavour_key():
    from fiberhmm.cli.dedup import deamination_flavour_key

    assert deamination_flavour_key([(1, 1), (2, 1), (3, 0)]) == "CT"
    assert deamination_flavour_key([(1, 0), (2, 0), (3, 1)]) == "GA"
    assert deamination_flavour_key([(1, 0), (2, 1)]) == "mixed"


def test_pcr_copies_in_both_orientations_are_one_molecule(tmp_path):
    path = tmp_path / "orient.bam"
    _make_flavour_bam(
        path,
        [("fwd0", A_SITES, "Y", False), ("fwd1", A_SITES, "Y", False),
         ("rev0", A_SITES, "Y", True), ("rev1", A_SITES, "Y", True)],
    )
    stats = run_dedup(str(path), str(tmp_path / "out.bam"), collapse=False)
    assert stats["n_fingerprintable"] == 4
    assert stats["n_clusters"] == 1
    assert stats["n_duplicates"] == 3


def test_opposite_flavours_never_merge_even_with_shared_positions(tmp_path):
    path = tmp_path / "flavour.bam"
    _make_flavour_bam(
        path,
        [("ct0", A_SITES, "Y", False), ("ct1", A_SITES, "Y", True),
         ("ga0", A_SITES, "R", False), ("ga1", A_SITES, "R", True)],
    )
    stats = run_dedup(str(path), str(tmp_path / "out.bam"), collapse=False)
    assert stats["n_clusters"] == 2
    records = list(pysam.AlignmentFile(str(tmp_path / "out.bam"), check_sq=False))
    clusters = {read.query_name: read.get_tag("di") for read in records}
    assert clusters["ct0"] == clusters["ct1"]
    assert clusters["ga0"] == clusters["ga1"]
    assert clusters["ct0"] != clusters["ga0"]
    # --ignore-strand still clusters across flavours (flag kept for compatibility).
    merged = run_dedup(str(path), str(tmp_path / "merged.bam"),
                       collapse=False, ignore_strand=True)
    assert merged["n_clusters"] == 1


def test_call_auto_dedup_uses_flavour_key(tmp_path):
    from fiberhmm.cli.call import _dedup_input_first

    path = tmp_path / "orient.bam"
    _make_flavour_bam(
        path,
        [("fwd0", A_SITES, "Y", False), ("rev0", A_SITES, "Y", True)],
    )
    temporary_bam, stats = _dedup_input_first(
        str(path), str(tmp_path / "output.bam"), 0.95, True, 1,
        region_parallel=False,
    )
    records = list(pysam.AlignmentFile(temporary_bam, check_sq=False))
    assert stats["n_clusters"] == 1
    assert sum(read.is_duplicate for read in records) == 1


@pytest.mark.parametrize("collapse", [False, True])
def test_supplementary_records_follow_their_duplicate_primary(tmp_path, collapse):
    """Supplementary records are called by default in 3.0; a PCR duplicate's
    supplementary record (another part of the same read) is flagged with it
    (and dropped with it under collapse); a representative's is kept."""
    path = tmp_path / "split.bam"
    _make_bam(path, [(f"A{index}", A_SITES, False, 60) for index in range(3)])
    header = pysam.AlignmentFile(str(path)).header
    records = list(pysam.AlignmentFile(str(path)))
    for index in range(3):
        supp = pysam.AlignedSegment(header)
        supp.query_name = f"A{index}"
        supp.flag = 2048
        supp.reference_id = 0
        supp.reference_start = 500
        supp.mapping_quality = 60
        supp.cigarstring = "200M"
        supp.query_sequence = "A" * 200
        records.append(supp)
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for read in records:
            out.write(read)
    output = tmp_path / "out.bam"
    stats = run_dedup(str(path), str(output), min_jaccard=0.95, min_deam=10,
                      collapse=collapse)
    out = list(pysam.AlignmentFile(str(output), check_sq=False))
    primaries = {r.query_name: r for r in out if not r.is_supplementary}
    supps = {r.query_name: r for r in out if r.is_supplementary}
    assert stats["n_duplicates"] == 2
    assert stats["n_duplicate_supplementary_records"] == 2
    if collapse:
        assert set(supps) == set(primaries) and len(primaries) == 1
    else:
        for name, supp in supps.items():
            assert supp.is_duplicate == primaries[name].is_duplicate
        assert sum(r.is_duplicate for r in supps.values()) == 2


@pytest.mark.parametrize("collapse", [False, True])
def test_prior_duplicate_marks_are_recomputed_not_kept(daf_bam, tmp_path, collapse):
    """Re-deduplicating a BAM that already carries 0x400/di/ds (an earlier
    dedup or call) yields exactly the newly computed duplicate set."""
    fresh = tmp_path / "fresh.bam"
    expected = run_dedup(str(daf_bam), str(fresh), min_jaccard=0.95,
                         min_deam=10, collapse=False)
    fresh_dups = {r.query_name for r in pysam.AlignmentFile(str(fresh), check_sq=False)
                  if r.is_duplicate}

    stale = tmp_path / "stale.bam"
    with pysam.AlignmentFile(str(daf_bam), check_sq=False) as source, \
            pysam.AlignmentFile(str(stale), "wb", header=source.header) as sink:
        for read in source:
            # Mark every record (representatives and the unfingerprintable one
            # included) as a stale duplicate of a nonexistent cluster.
            read.is_duplicate = True
            read.set_tag("di", 999999, value_type="i")
            read.set_tag("ds", 7, value_type="i")
            sink.write(read)

    output = tmp_path / "redone.bam"
    stats = run_dedup(str(stale), str(output), min_jaccard=0.95, min_deam=10,
                      collapse=collapse)
    records = list(pysam.AlignmentFile(str(output), check_sq=False))
    assert stats["n_duplicates"] == expected["n_duplicates"]
    assert stats["n_prior_marks_cleared"] == 9
    assert not any(r.has_tag("di") and r.get_tag("di") == 999999 for r in records)
    assert sum(r.is_duplicate for r in records) == (0 if collapse else stats["n_duplicates"])
    if not collapse:
        assert {r.query_name for r in records if r.is_duplicate} == fresh_dups
    lowdeam = next(r for r in records if r.query_name == "lowdeam")
    assert not lowdeam.is_duplicate and not lowdeam.has_tag("ds")


def test_call_dedup_on_a_marked_bam_reports_what_it_writes(daf_bam, tmp_path):
    from fiberhmm.cli.call import _dedup_input_first

    stale = tmp_path / "stale.bam"
    with pysam.AlignmentFile(str(daf_bam), check_sq=False) as source, \
            pysam.AlignmentFile(str(stale), "wb", header=source.header) as sink:
        for read in source:
            read.is_duplicate = True
            read.set_tag("di", 999999, value_type="i")
            sink.write(read)
    temporary_bam, stats = _dedup_input_first(
        str(stale), str(tmp_path / "output.bam"), 0.95, True, 1,
        region_parallel=False,
    )
    with pysam.AlignmentFile(temporary_bam, check_sq=False) as marked:
        records = list(marked)
        description = marked.header.to_dict()["PG"][-1]["DS"]
    assert sum(r.is_duplicate for r in records) == stats["n_duplicates"] == 6
    assert "prior_marks=cleared" in description
