import io
import itertools

import numpy as np
import pytest

from fiberhmm.daf.m5c import (
    F_METH,
    U_UNMETH,
    M5CDomain,
    M5CObservation,
    add_m5c_ma_tag,
    build_ddda_mcg_observation_payload,
    call_read_m5c,
    call_domains,
    deamination_probability,
    distance_forward_backward,
    distance_transition,
    collect_read_observations,
    estimate_five_prime_factors,
    make_windows,
    ma_intervals,
    observations_from_ddda_mcg_payload,
    project_domains_to_query,
    write_bed,
)
from fiberhmm.inference.tf_recaller import (
    N_CTX,
    UNMETH_OFFSET,
    build_llr_tables,
    build_m5c_llr_tables,
    call_tfs_in_interval,
    apply_emission_uplift,
)
from fiberhmm.cli.extract_tags import _extract_m5c
from fiberhmm.io.ma_tags import parse_ma_tag


def test_m5c_clis_require_explicit_ddda_chemistry():
    from fiberhmm.cli.call_m5c import parse_args as parse_call_args
    from fiberhmm.cli.tag_m5c import parse_args as parse_tag_args

    call_base = [
        "-i", "input.bam", "-r", "ref.fa", "-o", "out.bed",
        "--region", "chr1",
    ]
    tag_base = ["-i", "input.bam", "-r", "ref.fa", "-o", "out.bam"]
    with pytest.raises(SystemExit):
        parse_call_args(call_base)
    with pytest.raises(SystemExit):
        parse_tag_args(tag_base)
    assert parse_call_args([*call_base, "--enzyme", "ddda"]).enzyme == "ddda"
    assert parse_tag_args([*tag_base, "--enzyme", "ddda"]).enzyme == "ddda"
    with pytest.raises(SystemExit):
        parse_tag_args([*tag_base, "--enzyme", "dddb"])


def test_ddda_mcg_preflight_rejects_declared_dddb_header(tmp_path):
    import pysam

    from fiberhmm.cli.tag_m5c import _preflight_input

    path = tmp_path / "dddb.bam"
    header = {
        "HD": {"VN": "1.6"},
        "SQ": [{"SN": "chr1", "LN": 100}],
        "PG": [{
            "ID": "fiberhmm-call",
            "PN": "fiberhmm-call",
            "CL": "fiberhmm-call --enzyme dddb",
            "DS": "mode=daf enzyme=dddb",
        }],
    }
    with pysam.AlignmentFile(path, "wb", header=header):
        pass
    with pytest.raises(SystemExit, match="Only DddA DAF-seq is supported"):
        _preflight_input(str(path))


def test_rate_exponents_have_expected_order_and_effect_size():
    baseline = np.array([0.1, 0.5, 0.9])
    unmethylated = deamination_probability(0.0, baseline)
    methylated = deamination_probability(1.0, baseline)
    assert np.all(methylated < baseline)
    assert np.all(unmethylated > baseline)
    assert np.isclose(U_UNMETH / F_METH, 6.6646706587)


def test_fused_ddda_mcg_payload_uses_reference_context_and_can_exclude_nucs():
    reference_read = ("ACACG" * 80)[:300]
    sequence = list(reference_read)
    non_cpg_c = [
        index for index, base in enumerate(reference_read[:-1])
        if base == "C" and reference_read[index + 1] != "G"
    ]
    for index in non_cpg_c[:25]:
        sequence[index] = "Y"

    class Read:
        query_sequence = "".join(sequence)
        query_length = len(query_sequence)
        reference_name = "chr1"
        reference_start = 100
        reference_end = 100 + query_length
        cigartuples = [(0, query_length)]
        is_unmapped = False
        is_secondary = False
        is_supplementary = False
        is_reverse = False

    class Fasta:
        sequence = "N" * 100 + reference_read + "NN"

        def fetch(self, _chrom, start, end):
            return self.sequence[start:end]

    payload = build_ddda_mcg_observation_payload(Read(), Fasta())
    assert payload is not None
    assert payload["deaminated"].sum() == 25
    assert payload["is_cpg"].any()
    assert (~payload["is_cpg"]).any()

    excluded = [(0, 80)]
    observations = observations_from_ddda_mcg_payload(payload, excluded)
    assert observations
    assert all(obs.query_pos >= 80 for obs in observations)


def test_fused_ddda_mcg_payload_canonicalizes_bottom_strand_cpgs():
    reference_read = ("AGACG" * 80)[:300]
    sequence = list(reference_read)
    non_cpg_g = [
        index for index, base in enumerate(reference_read)
        if base == "G" and (index == 0 or reference_read[index - 1] != "C")
    ]
    for index in non_cpg_g[:25]:
        sequence[index] = "R"

    class Read:
        query_sequence = "".join(sequence)
        query_length = len(query_sequence)
        reference_name = "chr1"
        reference_start = 100
        reference_end = 100 + query_length
        cigartuples = [(0, query_length)]
        is_unmapped = False
        is_secondary = False
        is_supplementary = False
        is_reverse = True

    class Fasta:
        sequence = "N" * 100 + reference_read + "NN"

        def fetch(self, _chrom, start, end):
            return self.sequence[start:end]

    payload = build_ddda_mcg_observation_payload(Read(), Fasta())
    assert payload is not None
    assert payload["deaminated"].sum() == 25
    cpg_queries = payload["query_pos"][payload["is_cpg"]]
    cpg_refs = payload["reference_pos"][payload["is_cpg"]]
    assert len(cpg_queries)
    assert np.array_equal(cpg_refs, 100 + cpg_queries - 1)


def test_make_windows_uses_each_molecules_own_non_cpg_baseline():
    obs = []
    for molecule, hits in ((0, 8), (1, 2)):
        obs.extend(M5CObservation(molecule, pos, False, pos < hits, 0)
                   for pos in range(10))
        obs.append(M5CObservation(molecule, 100 + molecule, True, False, 0))
    windows = make_windows(obs, 0, 1000, min_other=10,
                           five_prime_factors=np.ones(4))
    assert np.allclose(windows[0].baseline, [0.8, 0.2])
    assert windows[0].deaminated.tolist() == [False, False]


def test_factor_estimation_rejects_missing_or_zero_deamination_contexts():
    observations = [
        M5CObservation(0, context, False, context < 3, context)
        for context in range(4)
    ]
    try:
        estimate_five_prime_factors(observations)
    except ValueError as error:
        assert "every A,C,G,T context" in str(error)
    else:
        raise AssertionError("zero-rate context should not silently become a factor")


def test_sparse_windows_are_never_emitted_even_if_neighbors_are_confident():
    obs = []
    for window in (0, 2):
        start = window * 1000
        for molecule in range(20):
            obs.extend(M5CObservation(molecule + window * 100, start + p,
                                      False, p < 8, 0) for p in range(10))
            obs.append(M5CObservation(molecule + window * 100, start + 100,
                                      True, False, 0))
    windows = make_windows(obs, 0, 3000, min_other=10,
                           five_prime_factors=np.ones(4))
    domains, _ = call_domains(windows, "chr1", posterior_threshold=0.5,
                              min_cpg=10, max_gap=0)
    assert all(not (d.start < 2000 and d.end > 1000) for d in domains)


def test_opposite_confident_window_is_not_bridged_into_overlapping_domain():
    from fiberhmm.daf import m5c

    windows = [m5c.M5CWindow(i * 1000, np.array([0.5] * 10),
                             np.zeros(10, dtype=bool)) for i in range(3)]
    posterior = np.array([[0.01, 0.99], [0.99, 0.01], [0.01, 0.99]])
    original = m5c.forward_backward
    m5c.forward_backward = lambda *_args: posterior
    try:
        domains, _ = call_domains(windows, "chr1", posterior_threshold=0.98,
                                  min_cpg=10, max_gap=1000)
    finally:
        m5c.forward_backward = original
    methylated = [d for d in domains if d.methylated]
    assert [(d.start, d.end) for d in methylated] == [(0, 1000), (2000, 3000)]


def test_bed_uses_state_name_and_mean_posterior_score():
    output = io.StringIO()
    write_bed([M5CDomain("chr1", 10, 20, True, 0.9919)], output)
    assert output.getvalue() == "chr1\t10\t20\tm5c_methylated\t991\t.\n"


def test_locus_projection_uses_only_overlapping_methylated_domains():
    class Read:
        query_sequence = "A" * 10
        query_length = 10
        reference_start = 100
        reference_name = "chr1"
        is_unmapped = False
        cigartuples = [(0, 10)]

    domains = [
        M5CDomain("chr1", 103, 107, True, 0.99),
        M5CDomain("chr1", 108, 110, False, 0.99),
        M5CDomain("chr2", 100, 110, True, 0.99),
    ]
    assert project_domains_to_query(Read(), domains) == [(3, 7)]


def test_m5c_tf_table_changes_only_cpg_accessible_probability():
    class Model:
        emissionprob_ = np.zeros((2, UNMETH_OFFSET + N_CTX))

    model = Model()
    model.emissionprob_[:, :N_CTX] = [[0.1], [0.8]]
    model.emissionprob_[:, UNMETH_OFFSET:] = [[0.9], [0.2]]
    hit, miss = build_llr_tables(model)
    m_hit, m_miss = build_m5c_llr_tables(model)
    non_cpg = 0
    cpg = 3 * 16  # first right-flank base is G in A,C,T,G base-4 coding
    assert np.isclose(m_hit[non_cpg], hit[non_cpg])
    assert np.isclose(m_miss[non_cpg], miss[non_cpg])
    assert m_miss[cpg] < miss[cpg]  # an undeaminated CpG is less footprint-like
    assert m_hit[cpg] > hit[cpg]    # a deaminated CpG is a less extreme veto

    uplift_hit, uplift_miss = apply_emission_uplift(hit, miss, model, 1.5)
    uplift_m5c_hit, uplift_m5c_miss = build_m5c_llr_tables(
        model, emission_uplift=1.5,
    )
    assert np.isclose(uplift_m5c_hit[non_cpg], uplift_hit[non_cpg])
    assert np.isclose(uplift_m5c_miss[non_cpg], uplift_miss[non_cpg])


def test_m5c_tf_table_preserves_non_cpg_with_unequal_context_marginals():
    class Model:
        emissionprob_ = np.zeros((2, UNMETH_OFFSET + N_CTX))

    model = Model()
    # Conditional hit probabilities and total context mass both differ.  The
    # latter must remain part of the original LLR for custom emission models.
    model.emissionprob_[0, :N_CTX] = 0.02
    model.emissionprob_[0, UNMETH_OFFSET:] = 0.08
    model.emissionprob_[1, :N_CTX] = 0.72
    model.emissionprob_[1, UNMETH_OFFSET:] = 0.18
    hit, miss = build_llr_tables(model)
    m_hit, m_miss = build_m5c_llr_tables(model)
    assert np.isclose(m_hit[0], hit[0])
    assert np.isclose(m_miss[0], miss[0])


def test_tf_scan_selects_m5c_llr_only_inside_mask():
    obs = np.array([UNMETH_OFFSET, UNMETH_OFFSET], dtype=np.int32)
    hit = np.zeros(N_CTX)
    miss = np.ones(N_CTX)
    m_hit = np.zeros(N_CTX)
    m_miss = -np.ones(N_CTX)
    unmasked = call_tfs_in_interval(obs, 0, 2, hit, miss, 1.5, 1)
    masked = call_tfs_in_interval(
        obs, 0, 2, hit, miss, 1.5, 1,
        m5c_mask=np.ones(2, dtype=bool),
        m5c_llr_hit=m_hit, m5c_llr_miss=m_miss,
    )
    assert len(unmasked) == 1
    assert masked == []


def test_tf_scan_rejects_mask_without_m5c_tables():
    obs = np.array([UNMETH_OFFSET], dtype=np.int32)
    hit = np.zeros(N_CTX)
    miss = np.ones(N_CTX)
    try:
        call_tfs_in_interval(
            obs, 0, 1, hit, miss, 0.0, 1,
            m5c_mask=np.ones(1, dtype=bool),
        )
    except ValueError as error:
        assert "requires both" in str(error)
    else:
        raise AssertionError("a mask without adjusted emissions must not be silent")


def test_add_m5c_tag_projects_seq_span_to_reverse_molecular_frame():
    class Read:
        query_sequence = "A" * 100
        query_length = 100
        is_reverse = True

        def __init__(self):
            self.tags = {"MA": "100;nuc.Q:1-10"}

        def has_tag(self, tag):
            return tag in self.tags

        def get_tag(self, tag):
            return self.tags[tag]

        def set_tag(self, tag, value, **_kwargs):
            if value is None:
                self.tags.pop(tag, None)
            else:
                self.tags[tag] = value

    read = Read()
    add_m5c_ma_tag(read, [(10, 30)])
    assert read.get_tag("MA") == "100;nuc.Q:1-10;ddda_mcg.:71-20"


def test_ma_interval_parser_accepts_legacy_plus_strand_group():
    class Read:
        query_sequence = "A" * 100
        query_length = 100
        is_reverse = False

        def has_tag(self, tag):
            return tag == "MA"

        def get_tag(self, tag):
            return "100;nuc+Q:11-20"

    assert ma_intervals(Read(), "nuc") == [(10, 30)]


def test_shared_ma_parser_exposes_ddda_mcg_intervals():
    parsed = parse_ma_tag("100;nuc.Q:1-10;ddda_mcg.:21-5,31-6")
    assert parsed["ddda_mcg"] == [(20, 5), (30, 6)]


def test_add_m5c_tag_preserves_an_alignment_when_replacing_middle_group():
    class Read:
        query_sequence = "A" * 100
        query_length = 100
        is_reverse = False

        def __init__(self):
            self.tags = {
                "MA": "100;nuc.Q:1-10;ddda_mcg.:21-5,31-5;tf.QQQ:41-6",
                "AN": "old_nuc,old_ddda_mcg_a,old_ddda_mcg_b,old_tf",
            }

        def has_tag(self, tag):
            return tag in self.tags

        def get_tag(self, tag):
            return self.tags[tag]

        def set_tag(self, tag, value, **_kwargs):
            if value is None:
                self.tags.pop(tag, None)
            else:
                self.tags[tag] = value

    read = Read()
    add_m5c_ma_tag(read, [(50, 60)])
    assert read.get_tag("MA") == (
        "100;nuc.Q:1-10;tf.QQQ:41-6;ddda_mcg.:51-10"
    )
    assert read.get_tag("AN") == "old_nuc,old_tf,fh_ddda_mcg_0"


def test_extract_m5c_reads_daf_ma_spans_without_native_mm_tag():
    class Read:
        reference_name = "chr1"
        query_name = "read1"
        is_reverse = False

        def __init__(self):
            self.tags = {"MA": "100;ddda_mcg.:11-20,51-10"}

        def has_tag(self, tag):
            return tag in self.tags

        def get_tag(self, tag):
            if tag not in self.tags:
                raise KeyError(tag)
            return self.tags[tag]

    output = io.StringIO()
    count = _extract_m5c(
        Read(), output, 125, np.arange(100, dtype=np.int64) + 1000,
    )
    assert count == 2
    assert output.getvalue() == (
        "chr1\t1010\t1060\tread1\t0\t+\t1010\t1060\t0\t2\t20,10\t0,40\n"
    )


def test_distance_transition_is_symmetric_and_forgets_across_long_gap():
    near = distance_transition(1, 1000)
    far = distance_transition(100_000, 1000)
    assert np.allclose(near.sum(axis=1), 1)
    assert np.allclose(near, near.T)
    assert near[0, 0] > 0.99
    assert np.allclose(far, 0.5)


def test_single_site_distance_hmm_still_validates_run_length():
    try:
        distance_forward_backward(np.zeros((1, 2)), np.array([10]), 0)
    except ValueError as error:
        assert "expected_run_bp" in str(error)
    else:
        raise AssertionError("non-positive state length must be rejected")


def test_distance_hmm_finds_a_sharp_synthetic_state_cliff():
    # Strong unmethylated emissions followed by strong methylated emissions.
    emission = np.vstack([
        np.tile([0.0, -3.0], (8, 1)),
        np.tile([-3.0, 0.0], (8, 1)),
    ])
    positions = np.arange(16) * 25
    posterior = distance_forward_backward(emission, positions, 500)[:, 1]
    assert np.all(posterior[:6] < 0.05)
    assert np.all(posterior[10:] > 0.95)
    assert np.argmax(posterior >= 0.5) in (7, 8)


def test_distance_forward_backward_matches_brute_force_state_enumeration():
    emission = np.array([
        [-0.2, -1.1], [-0.7, -0.3], [-1.5, -0.1], [-0.4, -0.9],
    ])
    positions = np.array([10, 13, 80, 450])
    initial = np.array([0.7, 0.3])
    observed = distance_forward_backward(
        emission, positions, expected_run_bp=120, initial=initial,
    )
    path_weights = []
    paths = list(itertools.product((0, 1), repeat=len(positions)))
    transitions = [
        distance_transition(distance, 120)
        for distance in np.diff(positions)
    ]
    for path in paths:
        weight = initial[path[0]] * np.exp(emission[0, path[0]])
        for index in range(1, len(path)):
            weight *= transitions[index - 1][path[index - 1], path[index]]
            weight *= np.exp(emission[index, path[index]])
        path_weights.append(weight)
    path_weights = np.asarray(path_weights)
    expected = np.zeros_like(observed)
    for path, weight in zip(paths, path_weights):
        for index, state in enumerate(path):
            expected[index, state] += weight
    expected /= path_weights.sum()
    assert np.allclose(observed, expected)


def _read_observations(cpg_deaminated, reverse_query=False):
    observations = []
    # Dense non-CpG internal standard at b=0.8 throughout the read.
    for i in range(80):
        observations.append(M5CObservation(
            0, i * 25, False, i % 5 != 0, 0,
            2000 - i * 25 if reverse_query else i * 25,
        ))
    for i, deaminated in enumerate(cpg_deaminated):
        observations.append(M5CObservation(
            0, 300 + i * 50, True, deaminated, 0,
            1700 - i * 50 if reverse_query else 300 + i * 50,
        ))
    return observations


def test_call_read_m5c_emits_only_the_methylated_side_of_a_cliff():
    # Deaminated CpGs support u; undeaminated CpGs support m.
    result = call_read_m5c(
        _read_observations([True] * 6 + [False] * 8), np.ones(4),
        expected_run_bp=500, posterior_threshold=0.9,
        baseline_radius=500, min_other=10, min_call_cpg=2,
    )
    assert result.methylated_posterior[:4].max() < 0.1
    assert result.methylated_posterior[-5:].min() > 0.9
    assert len(result.calls) == 1
    assert result.calls[0].n_cpg >= 5


def test_reverse_query_call_span_is_sorted_in_seq_frame():
    result = call_read_m5c(
        _read_observations([False] * 10, reverse_query=True), np.ones(4),
        expected_run_bp=1000, posterior_threshold=0.9,
        baseline_radius=500, min_other=10, min_call_cpg=2,
    )
    assert len(result.calls) == 1
    assert result.calls[0].start < result.calls[0].end


def test_single_cpg_cannot_create_a_methylated_span():
    result = call_read_m5c(
        _read_observations([False]), np.ones(4), expected_run_bp=1000,
        posterior_threshold=0.6, baseline_radius=500, min_call_cpg=2,
    )
    assert result.calls == ()


def test_long_gap_prevents_state_support_leaking_to_distant_neutral_cpg():
    emission = np.vstack([np.tile([-3.0, 0.0], (5, 1)), [[0.0, 0.0]]])
    positions = np.array([0, 20, 40, 60, 80, 100_000])
    posterior = distance_forward_backward(emission, positions, 1000)[:, 1]
    assert posterior[:4].min() > 0.95
    assert np.isclose(posterior[-1], 0.5, atol=1e-6)


def test_read_calls_do_not_bridge_long_unsupported_reference_gap():
    observations = _read_observations([False] * 8)
    shifted = []
    for obs in observations:
        if obs.is_cpg and obs.reference_pos >= 500:
            obs = M5CObservation(
                obs.molecule, obs.reference_pos + 100_000, obs.is_cpg,
                obs.deaminated, obs.five_prime_base, obs.query_pos,
            )
        shifted.append(obs)
    result = call_read_m5c(
        shifted, np.ones(4), expected_run_bp=1000,
        posterior_threshold=0.8, baseline_radius=200_000,
        min_other=10, min_call_cpg=2,
    )
    assert len(result.calls) == 2
    assert all(call.n_cpg == 4 for call in result.calls)


def test_vectorized_read_collection_keeps_query_positions_and_excludes_nuc():
    class Read:
        query_sequence = "ACGYACGCACGY"
        query_length = 12
        reference_start = 100
        reference_end = 112
        reference_name = "chr1"
        is_unmapped = False
        is_reverse = False
        cigartuples = [(0, 12)]

        def has_tag(self, tag):
            return tag == "MA"

        def get_tag(self, tag):
            return "12;nuc.:4-1"  # exclude query position 3 (Y)

    class Fasta:
        sequence = "N" * 100 + "ACGCACGCACGC" + "NN"

        def fetch(self, _chrom, start, end):
            return self.sequence[start:end]

    observations = collect_read_observations(
        Read(), Fasta(), min_deaminations=0,
    )
    assert all(obs.query_pos != 3 for obs in observations)
    assert {obs.query_pos for obs in observations} == {1, 5, 7, 9}
    assert [obs.is_cpg for obs in observations] == [True, True, False, True]


def test_read_collection_uses_apply_only_legacy_nuc_arrays():
    class Read:
        query_sequence = "ACGYACGCACGY"
        query_length = 12
        reference_start = 100
        reference_end = 112
        reference_name = "chr1"
        is_unmapped = False
        is_reverse = True
        cigartuples = [(0, 12)]
        # Molecular [8,9) flips to SEQ/query [3,4) on a 12-base reverse read.
        tags = {"ns": [8], "nl": [1], "as": [0], "al": [12]}

        def has_tag(self, tag):
            return tag in self.tags

        def get_tag(self, tag):
            return self.tags[tag]

    class Fasta:
        sequence = "N" * 100 + "ACGCACGCACGC" + "NN"

        def fetch(self, _chrom, start, end):
            return self.sequence[start:end]

    observations = collect_read_observations(
        Read(), Fasta(), min_deaminations=0,
        input_molecular_frame=True,
    )
    assert all(obs.query_pos != 3 for obs in observations)
    assert {obs.query_pos for obs in observations} == {1, 5, 7, 9}


def test_bottom_strand_cpg_uses_canonical_reference_c_coordinate():
    class Read:
        query_sequence = "AGRGA"
        query_length = 5
        reference_start = 100
        reference_end = 105
        reference_name = "chr1"
        is_unmapped = False
        is_reverse = False
        cigartuples = [(0, 5)]

        def has_tag(self, tag):
            return tag == "MA"

        def get_tag(self, tag):
            return "5;msp.:1-5"

    class Fasta:
        sequence = "N" * 100 + "ACGCA" + "NN"

        def fetch(self, _chrom, start, end):
            return self.sequence[start:end]

    observations = collect_read_observations(
        Read(), Fasta(), min_deaminations=0,
    )
    cpg = [obs for obs in observations if obs.is_cpg]
    assert len(cpg) == 1
    assert cpg[0].query_pos == 2  # observation is the reference G
    assert cpg[0].reference_pos == 101  # canonical CpG C coordinate
