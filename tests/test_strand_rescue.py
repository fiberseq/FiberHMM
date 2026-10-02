from __future__ import annotations

import copy
import errno
import hashlib
import io
import json
import math
import random
from array import array
from pathlib import Path

import numpy as np
import pysam
import pytest

from fiberhmm.cli import strand_rescue as strand_rescue_cli
from fiberhmm.inference import strand_rescue as strand_rescue_inference
from fiberhmm.inference.strand_rescue import (
    IntervalCall,
    PRESETS,
    ReadEvidence,
    SiteTemplate,
    _build_call_catalog,
    _build_site_template,
    _cluster_edge_records,
    _prepare_edge_population_statistics,
    analyze_strand_rescue,
    analyze_shared_geometry,
    build_edge_action_candidate,
    build_rescue_action_candidate,
    discover_edge_sites,
    discover_sites,
    edge_hypothesis_evidence,
    edge_log_bayes_factor,
    enumerate_configurations,
    fit_mixture_weights,
    fit_site_state_model,
    finalize_alignment_actions,
    hard_observations,
    load_region_evidence,
    latent_geometry_pair_evidence,
    oriented_quality_bytes,
    probability_to_uint8,
    project_reference_interval_to_molecular,
    resolve_rescue_edge_topology,
    wilson_lower_bound,
)
from fiberhmm.inference.tf_recaller import N_CTX


def _site(site_id="site1", start=100, end=120, support=20):
    return SiteTemplate(
        site_id=site_id,
        start=start,
        end=end,
        center=(start + end) // 2,
        support={"FWD": support, "REV": 0},
        start_mad=2.0,
        end_mad=2.0,
        local_enrichment=20.0,
        local_enrichment_by_strand={"FWD": 20.0, "REV": 0.0},
        strand_geometry={
            "FWD": {
                "start": start,
                "end": end,
                "start_mad": 2.0,
                "end_mad": 2.0,
                "calls": support,
                "molecules": support,
            }
        },
        geometry_reliability=0.9,
    )


def _read(
    name,
    strand,
    positions,
    steps,
    *,
    tfs=(),
    nucs=(),
    msps=(),
    ref_start=50,
    ref_end=180,
    alignment_flag=0,
    cigar=None,
    record_sha256=None,
    alignment_occurrence=0,
):
    steps = np.asarray(steps, dtype=float)
    return ReadEvidence(
        name=name,
        strand=strand,
        ref_start=ref_start,
        ref_end=ref_end,
        positions=np.asarray(positions, dtype=np.int64),
        steps=steps,
        hits=steps < 0,
        contexts=np.zeros(len(steps), dtype=np.int64),
        tfs=list(tfs),
        nucs=list(nucs),
        msps=list(msps),
        library_id="/cohort.bam",
        alignment_flag=alignment_flag,
        cigar=cigar,
        record_sha256=record_sha256,
        alignment_occurrence=alignment_occurrence,
    )


def _fully_maps_brute_reference(read, start, end):
    if not read.spans(start, end):
        return False
    if read.alignment_blocks is None:
        return True
    blocks = read.alignment_blocks
    left_mapped = any(left <= start < right for left, right in blocks)
    right_mapped = any(left <= end - 1 < right for left, right in blocks)
    mapped = sum(
        max(0, min(end, right) - max(start, left))
        for left, right in blocks
    )
    return left_mapped and right_mapped and mapped / (end - start) >= 0.95


@pytest.mark.parametrize(
    ("blocks", "interval", "expected"),
    [
        (((100, 147), (152, 200)), (100, 200), True),  # exactly 95%
        (((100, 147), (153, 200)), (100, 200), False),
        (((100, 140), (145, 200)), (140, 200), False),
        (((100, 140), (145, 200)), (100, 145), False),
        (((100, 140), (140, 200)), (100, 200), True),
        (((100, 140), (145, 200)), (145, 200), True),
        ((), (100, 200), False),
    ],
)
def test_fully_maps_index_preserves_boundaries_gaps_and_threshold(
    blocks, interval, expected
):
    read = _read("indexed", "FWD", [], [], ref_start=50, ref_end=250)
    read.alignment_blocks = blocks
    start, end = interval
    assert read.fully_maps(start, end) is expected
    assert read.fully_maps(start, end) == _fully_maps_brute_reference(
        read, start, end
    )


def test_fully_maps_index_falls_back_for_unsafe_blocks_and_invalidates():
    read = _read("fallback", "FWD", [], [], ref_start=50, ref_end=250)
    read.alignment_blocks = ((100, 147), (152, 200))
    assert read.fully_maps(100, 200)
    original_index = read._alignment_block_index
    assert original_index is not None
    assert read.fully_maps(101, 199) == _fully_maps_brute_reference(
        read, 101, 199
    )
    assert read._alignment_block_index is original_index
    assert "_alignment_block_index" not in strand_rescue_inference.asdict(read)

    # These overlaps deliberately make the former summed coverage exceed
    # the union coverage.  The indexed path must not silently normalize it.
    read.alignment_blocks = ((100, 180), (150, 190), (199, 200))
    assert read.fully_maps(100, 200) == _fully_maps_brute_reference(
        read, 100, 200
    )
    assert read._alignment_block_index is None

    read.alignment_blocks = ((199, 200), (100, 190))
    assert read.fully_maps(100, 200) == _fully_maps_brute_reference(
        read, 100, 200
    )
    assert read._alignment_block_index is None

    # A mutable, otherwise ordinary-looking synthetic collection also stays
    # on the brute path so later list edits cannot stale an index.
    read.alignment_blocks = [(100, 147), (152, 200)]
    assert read.fully_maps(100, 200) == _fully_maps_brute_reference(
        read, 100, 200
    )
    assert read._alignment_block_index is None


def test_fully_maps_rejects_nonpositive_intervals_consistently():
    read = _read("degenerate", "FWD", [], [], ref_start=50, ref_end=250)
    assert not read.fully_maps(150, 150)
    assert not read.fully_maps(151, 150)

    read.alignment_blocks = ()
    assert not read.fully_maps(150, 150)

    read.alignment_blocks = ((90, 160), (160, 210))
    assert not read.fully_maps(150, 150)
    assert not read.fully_maps(151, 150)


def test_fully_maps_randomized_differential_safe_and_unsafe_blocks():
    rng = random.Random(0xF11B10C)
    evaluated = 0
    for read_index in range(400):
        cursor = rng.randrange(0, 20)
        safe_blocks = []
        for _ in range(rng.randrange(2, 35)):
            cursor += rng.randrange(0, 8)
            if cursor >= 500:
                break
            right = min(500, cursor + rng.randrange(1, 20))
            safe_blocks.append((cursor, right))
            cursor = right
        if len(safe_blocks) < 2:
            continue

        variants = [tuple(safe_blocks), tuple(reversed(safe_blocks))]
        overlap = list(safe_blocks)
        overlap.insert(
            1,
            (
                overlap[0][0] + 1,
                min(500, overlap[0][1] + rng.randrange(1, 10)),
            ),
        )
        variants.append(tuple(overlap))

        for variant in variants:
            read = _read(
                f"differential-{read_index}",
                "FWD",
                [],
                [],
                ref_start=0,
                ref_end=500,
            )
            read.alignment_blocks = variant
            for _ in range(12):
                start = rng.randrange(-20, 520)
                end = start + rng.randrange(1, 180)
                assert read.fully_maps(
                    start, end
                ) == _fully_maps_brute_reference(read, start, end)
                evaluated += 1
    assert evaluated >= 10_000


def _strong_source_reads(count=20, *, interval=(100, 120), score=0):
    return [
        _read(
            f"source-{index}",
            "FWD",
            [90, interval[0] + 5, interval[1] - 5, 130],
            [-4.0, 4.0, 4.0, -4.0],
            tfs=[IntervalCall(*interval, score)],
        )
        for index in range(count)
    ]


def _v5_read(
    *,
    name="v5-read",
    ref_start=100,
    ref_end=200,
    query_length=100,
    alignment_flag=0,
    cigar="100M",
    cigar_tuples=((0, 100),),
    molecular_tfs=(),
    molecular_nucs=(),
    molecular_msps=(),
    input_index=0,
    input_record_ordinal=0,
):
    read = _read(
        name,
        "REV" if alignment_flag & 16 else "FWD",
        [],
        [],
        ref_start=ref_start,
        ref_end=ref_end,
        alignment_flag=alignment_flag,
        cigar=cigar,
    )
    read.query_length = query_length
    read.cigar_tuples = tuple(cigar_tuples)
    read.molecular_tfs = tuple(molecular_tfs)
    read.molecular_nucs = tuple(molecular_nucs)
    read.molecular_msps = tuple(molecular_msps)
    read.input_index = input_index
    read.input_record_ordinal = input_record_ordinal
    return read


@pytest.mark.parametrize(
    ("alignment_flag", "start", "end", "expected"),
    [
        (0, 100, 142, (5, 43)),
        (16, 100, 142, (0, 43)),
        (0, 105, 118, (10, 16)),
        (16, 105, 118, (22, 16)),
        (0, 105, 128, None),  # Two deleted bases put mapping below 95%.
        (0, 120, 130, None),  # The reference-left endpoint is deleted.
    ],
)
def test_v5_generator_projects_complex_cigar_exactly(
    alignment_flag, start, end, expected
):
    read = _v5_read(
        ref_end=142,
        query_length=48,
        alignment_flag=alignment_flag,
        cigar="5S10M3I10M2D20M",
        cigar_tuples=((4, 5), (0, 10), (1, 3), (0, 10), (2, 2), (0, 20)),
    )

    assert project_reference_interval_to_molecular(read, start, end) == expected


def test_v5_generator_quality_bytes_are_exact_and_molecularly_oriented():
    assert [
        probability_to_uint8(value)
        for value in (-1.0, 0.0, 0.1, 0.5, 0.9, 1.0, 2.0)
    ] == [0, 0, 26, 128, 230, 255, 255]
    assert oriented_quality_bytes(0.5, 0.1, 0.9, reverse=False) == (
        128,
        26,
        230,
    )
    assert oriented_quality_bytes(0.5, 0.1, 0.9, reverse=True) == (
        128,
        230,
        26,
    )


def test_v5_reverse_multi_component_rescue_retains_configuration_order_and_q():
    read = _v5_read(
        name="reverse-multi",
        alignment_flag=16,
        molecular_msps=((10, 80),),
        input_index=2,
        input_record_ordinal=7,
    )
    decision = {
        "decision_id": "decision-1",
        "library_id": "/cohort.bam",
        "current_interval": [110, 190],
        "current_annotation_ordinal": 0,
        "current_molecular_interval": [10, 80],
        "sr_hypothesis_probability": 0.6,
        "proposed_site_intervals": [[120, 130], [160, 175]],
        "proposed_site_edge_confidence": [[0.1, 0.9], [0.2, 0.8]],
    }
    token = hashlib.sha256(
        b"sr:/cohort.bam:decision-1"
    ).hexdigest()[:16]

    candidate, rejection = build_rescue_action_candidate(read, decision)

    assert rejection is None
    assert candidate == {
        "kind": "rescue_candidate",
        "input_index": 2,
        "ordinal": 7,
        "decision_id": "decision-1",
        "current_interval": [110, 190],
        "token": token,
        "source_ordinal": 0,
        "source_interval": [10, 80],
        "q0": 153,
        # Reference/configuration order is intentionally reverse molecular order.
        "components": [
            {"component_index": 0, "interval": [70, 10], "q1": 230, "q2": 26},
            {"component_index": 1, "interval": [25, 15], "q1": 204, "q2": 51},
        ],
    }

    rescues, edges, diagnostics = finalize_alignment_actions(read, [candidate], [])

    assert edges == []
    assert diagnostics == {
        "edge_duplicate_source": 0,
        "edge_joint_collision": 0,
        "edge_rescue_collision": 0,
        "rescue_collision": 0,
    }
    assert rescues == [
        {
            "token": token,
            "source_ordinal": 0,
            "source_interval": [10, 80],
            "q0": 153,
            "components": candidate["components"],
        }
    ]


def test_v5_reverse_edge_candidate_has_exact_identity_interval_and_q():
    read = _v5_read(
        name="reverse-edge",
        alignment_flag=16,
        molecular_nucs=((65, 20),),
        input_record_ordinal=11,
    )
    decision = {
        "status": "edge_update",
        "call_type": "nuc",
        "decision_id": "edge-1",
        "library_id": "/cohort.bam",
        "current_annotation_ordinal": 0,
        "current_molecular_interval": [65, 20],
        "canonical_interval": [110, 138],
        "edge_hypothesis": {
            "alternative_probability": 0.7,
            "left": {"alternative_probability": 0.2},
            "right": {"alternative_probability": 0.8},
        },
    }
    token = hashlib.sha256(
        b"sr-edge:nuc:/cohort.bam:edge-1"
    ).hexdigest()[:16]

    candidate, rejection = build_edge_action_candidate(read, decision)

    assert rejection is None
    assert candidate == {
        "kind": "edge_candidate",
        "input_index": 0,
        "ordinal": 11,
        "decision_id": "edge-1",
        "token": token,
        "call_type": "nuc",
        "source_ordinal": 0,
        "source_interval": [65, 20],
        "alternative_interval": [62, 28],
        "q": [178, 204, 51],
        "_aggregate": {
            "prior_only": True,
            "chemistry_opposed": False,
            "extreme_edge_shift": False,
        },
    }


def test_v5_edge_candidate_preserves_q0_but_marks_unobserved_changed_edges():
    read = _v5_read(
        name="prior-only-edge",
        molecular_nucs=((65, 20),),
        input_record_ordinal=12,
    )
    decision = {
        "status": "edge_update",
        "call_type": "nuc",
        "decision_id": "edge-prior-only",
        "library_id": "/cohort.bam",
        "current_annotation_ordinal": 0,
        "current_molecular_interval": [65, 20],
        "canonical_interval": [105, 135],
        "edge_hypothesis": {
            "alternative_probability": 0.95,
            "left": {"alternative_probability": 0.94},
            "right": {"alternative_probability": 0.93},
        },
        "materialized_edge_confidence": [0.0, 0.0],
        "target_edge_evidence": {"changed_opportunities": 0},
    }

    candidate, rejection = build_edge_action_candidate(read, decision)

    assert rejection is None
    assert candidate["q"] == [242, 0, 0]
    assert candidate["_aggregate"]["prior_only"] is True


def test_v5_finalizer_gives_rescue_priority_over_colliding_edge_expansion():
    read = _v5_read(
        molecular_nucs=((0, 20),),
        molecular_msps=((20, 50),),
    )
    rescue = {
        "kind": "rescue_candidate",
        "decision_id": "rescue",
        "current_interval": [120, 170],
        "token": "a" * 16,
        "source_ordinal": 0,
        "source_interval": [20, 50],
        "q0": 200,
        "components": [
            {"component_index": 0, "interval": [25, 10], "q1": 210, "q2": 220}
        ],
    }
    edge = {
        "kind": "edge_candidate",
        "decision_id": "edge",
        "token": "b" * 16,
        "call_type": "nuc",
        "source_ordinal": 0,
        "source_interval": [0, 20],
        "alternative_interval": [0, 30],
        "q": [190, 200, 210],
    }

    rescues, edges, diagnostics = finalize_alignment_actions(
        read, [rescue], [edge]
    )

    assert [value["token"] for value in rescues] == ["a" * 16]
    assert edges == []
    assert diagnostics["edge_rescue_collision"] == 1
    assert diagnostics["rescue_collision"] == 0


def test_v5_finalizer_gives_tf_consensus_precedence_over_nuc_edge():
    read = _v5_read(
        molecular_nucs=((0, 20),),
        molecular_tfs=((30, 10),),
    )
    edges = [
        {
            "kind": "edge_candidate",
            "decision_id": "nuc-edge",
            "token": "a" * 16,
            "call_type": "nuc",
            "source_ordinal": 0,
            "source_interval": [0, 20],
            "alternative_interval": [0, 25],
            "q": [200, 200, 200],
        },
        {
            "kind": "edge_candidate",
            "decision_id": "tf-edge",
            "token": "b" * 16,
            "call_type": "tf",
            "source_ordinal": 0,
            "source_interval": [30, 10],
            "alternative_interval": [22, 10],
            "q": [200, 200, 200],
        },
    ]

    rescues, accepted_edges, diagnostics = finalize_alignment_actions(
        read, [], edges
    )

    assert rescues == []
    assert [value["call_type"] for value in accepted_edges] == ["tf"]
    assert diagnostics["edge_joint_collision"] == 1
    assert diagnostics["nuc_edge_joint_collision"] == 1


def test_inline_topology_resolution_gives_tf_precedence_over_nuc():
    common = {
        "status": "edge_update",
        "library_id": "/cohort.bam",
        "read": "read-1",
        "alignment": {
            "reference_start": 0,
            "flag": 0,
            "cigar": "100M",
            "record_sha256": "a" * 64,
            "occurrence": 0,
        },
        "target_edge_evidence": {"changed_opportunities": 2},
        "molecule_probability": 0.9,
        "extreme_edge_shift": False,
    }
    tf_decision = {
        **common,
        "call_type": "tf",
        "current_interval": [30, 40],
        "canonical_interval": [22, 32],
    }
    nuc_decision = {
        **common,
        "call_type": "nuc",
        "current_interval": [0, 20],
        "canonical_interval": [0, 25],
    }
    tf_result = {"harmonizations": [tf_decision], "counts": {}}
    nuc_result = {"harmonizations": [nuc_decision], "counts": {}}

    strand_rescue_inference.resolve_joint_edge_topology(
        tf_result, nuc_result,
    )

    assert tf_decision["status"] == "edge_update"
    assert nuc_decision["status"] == "joint_topology_conflict_retained"
    assert tf_result["counts"]["edge_updates"] == 1
    assert nuc_result["counts"]["edge_updates"] == 0


def test_v5_joint_h_rejection_precedes_rescue_vs_h_rejection():
    read = _v5_read(
        molecular_nucs=((0, 10),),
        molecular_tfs=((30, 10),),
        molecular_msps=((10, 10),),
    )
    rescue = {
        "decision_id": "rescue",
        "token": "a" * 16,
        "source_ordinal": 0,
        "source_interval": [10, 10],
        "current_interval": [110, 120],
        "q0": 200,
        "components": [
            {"component_index": 0, "interval": [12, 5], "q1": 200, "q2": 200}
        ],
    }
    edges = [
        {
            "decision_id": "left",
            "token": "b" * 16,
            "call_type": "nuc",
            "source_ordinal": 0,
            "source_interval": [0, 10],
            "alternative_interval": [0, 25],
            "q": [200, 200, 200],
        },
        {
            "decision_id": "right",
            "token": "c" * 16,
            "call_type": "tf",
            "source_ordinal": 0,
            "source_interval": [30, 10],
            "alternative_interval": [20, 20],
            "q": [200, 200, 200],
        },
    ]

    rescues, accepted_edges, diagnostics = finalize_alignment_actions(
        read, [rescue], edges
    )

    assert len(rescues) == 1
    assert [value["call_type"] for value in accepted_edges] == ["tf"]
    assert diagnostics["edge_joint_collision"] == 1
    # The downstream nuc proposal is rejected first; TF consensus and the
    # non-overlapping rescue survive.
    assert diagnostics["edge_rescue_collision"] == 0


def test_v5_finalizer_rejects_order_collapsed_same_class_pair():
    read = _v5_read(molecular_tfs=((0, 20), (10, 20)))
    edges = [
        {
            "decision_id": "left",
            "token": "a" * 16,
            "call_type": "tf",
            "source_ordinal": 0,
            "source_interval": [0, 20],
            "alternative_interval": [5, 20],
            "q": [200, 200, 200],
        },
        {
            "decision_id": "right",
            "token": "b" * 16,
            "call_type": "tf",
            "source_ordinal": 1,
            "source_interval": [10, 20],
            "alternative_interval": [5, 25],
            "q": [200, 200, 200],
        },
    ]

    rescues, accepted_edges, diagnostics = finalize_alignment_actions(
        read, [], edges
    )

    assert rescues == []
    assert accepted_edges == []
    assert diagnostics["edge_joint_collision"] == 2


def test_v5_finalizer_rejects_complete_pairwise_order_component():
    read = _v5_read(molecular_tfs=((0, 5), (10, 5), (20, 5)))
    alternatives = ((30, 5), (0, 5), (21, 5))
    edges = [
        {
            "decision_id": f"edge-{ordinal}",
            "token": chr(ord("a") + ordinal) * 16,
            "call_type": "tf",
            "source_ordinal": ordinal,
            "source_interval": [ordinal * 10, 5],
            "alternative_interval": list(alternative),
            "q": [200, 200, 200],
        }
        for ordinal, alternative in enumerate(alternatives)
    ]

    rescues, accepted_edges, diagnostics = finalize_alignment_actions(
        read, [], edges
    )

    assert rescues == []
    assert accepted_edges == []
    assert diagnostics["edge_joint_collision"] == 3


def test_v5_finalizer_rejects_duplicate_output_names():
    read = _v5_read(molecular_msps=((0, 20), (20, 20)))
    rescues = [
        {
            "kind": "rescue_candidate",
            "decision_id": decision_id,
            "current_interval": [index * 20, index * 20 + 20],
            "token": "a" * 16,
            "source_ordinal": index,
            "source_interval": [index * 20, 20],
            "q0": 200,
            "components": [
                {
                    "component_index": 0,
                    "interval": [index * 20 + 2, 5],
                    "q1": 200,
                    "q2": 200,
                }
            ],
        }
        for index, decision_id in enumerate(("one", "two"))
    ]

    with pytest.raises(ValueError, match="duplicate v5 action name token"):
        finalize_alignment_actions(read, rescues, [])


def test_strand_rescue_parser_constructs_and_exposes_bounded_layouts():
    parser = strand_rescue_cli.build_parser()
    arguments = parser.parse_args(
        [
            "--bam",
            "input.bam",
            "--preset",
            "dddb",
            "--region",
            "chr1:100-200",
            "--output",
            "report.json",
        ]
    )

    assert arguments.report_layout == "auto"
    assert arguments.diagnostics == "aggregate"


def test_generation_paths_reject_duplicate_canonical_bams(tmp_path):
    bam = tmp_path / "input.bam"
    model = tmp_path / "model.json"

    with pytest.raises(ValueError, match="canonically unique"):
        strand_rescue_cli.validate_generation_paths(
            [str(bam), str(bam.parent / "." / bam.name)],
            [str(model)],
            tmp_path / "report.json",
            None,
        )


@pytest.mark.parametrize("collision", ("report-bam", "report-model", "tsv-model", "same-output"))
def test_generation_paths_reject_destructive_output_collisions(tmp_path, collision):
    bam = tmp_path / "input.bam"
    model = tmp_path / "model.json"
    report = tmp_path / "report.json"
    proposal = tmp_path / "proposal.tsv"
    if collision == "report-bam":
        report = bam
    elif collision == "report-model":
        report = model
    elif collision == "tsv-model":
        proposal = model
    elif collision == "same-output":
        proposal = report

    with pytest.raises(ValueError, match="overwrite|different paths"):
        strand_rescue_cli.validate_generation_paths(
            [str(bam)], [str(model)], report, proposal
        )


@pytest.mark.parametrize(
    "index_name", ("input.bam.bai", "input.bai", "input.bam.csi", "input.csi")
)
def test_generation_paths_protects_every_bam_index_naming_convention(
    tmp_path, index_name
):
    bam = tmp_path / "input.bam"
    model = tmp_path / "model.json"

    with pytest.raises(ValueError, match="BAM index"):
        strand_rescue_cli.validate_generation_paths(
            [str(bam)], [str(model)], tmp_path / index_name, None
        )


@pytest.mark.parametrize("operation", ("replace", "link"))
def test_generator_publication_retries_transient_permission_errors(
    tmp_path, monkeypatch, operation
):
    source = tmp_path / f"{operation}.stage"
    destination = tmp_path / f"{operation}.final"
    source.write_bytes(b"validated-output")
    real_operation = getattr(strand_rescue_cli.os, operation)
    calls = []
    sleeps = []

    def flaky(source_path, destination_path):
        calls.append((Path(source_path), Path(destination_path)))
        if len(calls) <= 2:
            raise PermissionError(errno.EACCES, "transient DrvFS denial")
        return real_operation(source_path, destination_path)

    monkeypatch.setattr(strand_rescue_cli.os, operation, flaky)
    monkeypatch.setattr(strand_rescue_cli.time, "sleep", sleeps.append)

    helper = getattr(strand_rescue_cli, f"_{operation}_with_permission_retry")
    helper(source, destination)

    assert destination.read_bytes() == b"validated-output"
    assert len(calls) == 3
    assert sleeps == [0.05, 0.1]
    assert source.exists() is (operation == "link")


def test_generator_stage_creation_retries_transient_permission_errors(
    tmp_path, monkeypatch
):
    # Stages are created by mkstemp_shared (umask-honouring, audit M12).
    real_mkstemp = strand_rescue_cli.mkstemp_shared
    calls = []
    sleeps = []

    def flaky_mkstemp(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) <= 2:
            raise PermissionError(errno.EACCES, "transient staging denial")
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(strand_rescue_cli, "mkstemp_shared", flaky_mkstemp)
    monkeypatch.setattr(strand_rescue_cli.time, "sleep", sleeps.append)
    registry = strand_rescue_cli._StagingRegistry()

    stage = registry.create(tmp_path, "report.actions", ".jsonl")

    assert stage.exists()
    assert stage in registry.paths
    assert len(calls) == 3
    assert sleeps == [0.05, 0.1]
    registry.cleanup()
    assert not stage.exists()


def test_generator_bgzf_stage_open_retries_transient_permission_errors(
    tmp_path, monkeypatch
):
    real_bgzfile = strand_rescue_cli.pysam.BGZFile
    calls = []
    sleeps = []

    def flaky_bgzfile(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) <= 2:
            raise PermissionError(errno.EPERM, "transient BGZF staging denial")
        return real_bgzfile(*args, **kwargs)

    monkeypatch.setattr(strand_rescue_cli.pysam, "BGZFile", flaky_bgzfile)
    monkeypatch.setattr(strand_rescue_cli.time, "sleep", sleeps.append)
    registry = strand_rescue_cli._StagingRegistry()

    writer = strand_rescue_cli._V5ActionWriter(
        tmp_path / "report.json", 0, ["chr1", 100, 200], registry
    )

    assert len(calls) == 3
    assert sleeps == [0.05, 0.1]
    writer.handle.close()
    stages = set(registry.paths)
    registry.cleanup()
    assert all(not path.exists() for path in stages)


def test_generator_text_stage_open_retries_transient_permission_errors(
    tmp_path, monkeypatch
):
    real_open = strand_rescue_cli.Path.open
    denied = []
    sleeps = []

    def flaky_open(path, *args, **kwargs):
        if args and args[0] == "w" and len(denied) < 2:
            denied.append(Path(path))
            raise PermissionError(errno.EACCES, "transient text staging denial")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(strand_rescue_cli.Path, "open", flaky_open)
    monkeypatch.setattr(strand_rescue_cli.time, "sleep", sleeps.append)
    registry = strand_rescue_cli._StagingRegistry()

    sink = strand_rescue_cli._GenerationSink(
        tmp_path / "report.json", "stream", registry
    )

    assert len(denied) == 2
    assert sleeps == [0.05, 0.1]
    sink.finish_callbacks()
    stages = set(registry.paths)
    registry.cleanup()
    assert all(not path.exists() for path in stages)


def test_generator_stage_retry_does_not_retry_unrelated_io_errors(
    monkeypatch,
):
    calls = []
    sleeps = []

    def fail_once():
        calls.append(True)
        raise OSError(errno.EIO, "not a sharing denial")

    monkeypatch.setattr(strand_rescue_cli.time, "sleep", sleeps.append)

    with pytest.raises(OSError, match="not a sharing denial"):
        strand_rescue_cli._retry_transient_permission(fail_once)

    assert len(calls) == 1
    assert sleeps == []


@pytest.mark.parametrize(
    "flag",
    (
        "--max-boundary-mad",
        "--nuc-max-boundary-mad",
        "--min-local-enrichment",
        "--strand-min-source-enrichment",
        "--efficiency-pseudo-count",
    ),
)
def test_cli_rejects_nonfinite_numeric_arguments_before_inference(flag, capsys):
    with pytest.raises(SystemExit):
        strand_rescue_cli.main(
            [
                "--bam",
                "missing.bam",
                "--preset",
                "dddb",
                "--region",
                "chr1:100-200",
                flag,
                "nan",
                "--output",
                "report.json",
            ]
        )

    assert flag in capsys.readouterr().err


def test_v5_writer_rejects_derived_sidecar_destination_collision(tmp_path):
    registry = strand_rescue_cli._StagingRegistry()
    output = tmp_path / "report.json"
    region = ["chr1", 100, 200]
    writer = strand_rescue_cli._V5ActionWriter(output, 0, region, registry)
    header = {
        "kind": "header",
        "schema": strand_rescue_cli.V5_ACTION_STREAM_SCHEMA,
        "input_index": 0,
        "input_id": "input0000",
        "loaded_region": region,
        "ordinal_base": 0,
    }
    trailer = {
        "kind": "trailer",
        "fetch_record_count": 0,
        "action_record_count": 0,
        "first_action_ordinal": None,
        "last_action_ordinal": None,
        "rescue_decision_count": 0,
        "rescue_component_count": 0,
        "tf_edge_update_count": 0,
        "nuc_edge_update_count": 0,
    }
    digest = hashlib.sha256(
        strand_rescue_cli._canonical_json_bytes(header)
        + strand_rescue_cli._canonical_json_bytes(trailer)
    ).hexdigest()
    collision = tmp_path / (
        f"report.input0000.sr-actions.{digest[:12]}.jsonl.bgz"
    )

    with pytest.raises(ValueError, match="sidecar would collide"):
        writer.finish(0, protected_paths=[collision])
    registry.cleanup()
    assert not collision.exists()


@pytest.mark.parametrize(
    ("details", "size", "expected"),
    [
        (99_999, 256 * 1024 * 1024 - 1, "inline"),
        (100_000, 1, "stream"),
        (1, 256 * 1024 * 1024, "stream"),
    ],
)
def test_v5_auto_layout_spill_limits_are_inclusive(details, size, expected):
    assert (
        strand_rescue_cli.select_report_layout("auto", details, size)
        == expected
    )


def test_v5_explicit_inline_fails_at_bounded_limit():
    with pytest.raises(ValueError, match="explicit inline report crossed"):
        strand_rescue_cli.select_report_layout("inline", 100_000, 1)


def test_cli_stream_writes_valid_manifest_bgzf_gzi_and_proposal_tsv(
    tmp_path, monkeypatch
):
    bam_path = tmp_path / "input.bam"
    model_path = tmp_path / "model.json"
    output = tmp_path / "strand-rescue.json"
    proposal = tmp_path / "proposals.tsv"
    bam_path.write_bytes(b"input-metadata-fixture")
    model_path.write_text("{}\n")

    read = _v5_read(
        name="stream-read",
        molecular_msps=((10, 80),),
        input_record_ordinal=2,
    )
    read.library_id = str(bam_path.resolve())
    read.record_sha256 = "a" * 64

    def fake_load(_path, _chrom, _start, _end, **kwargs):
        kwargs["load_diagnostics"].update(
            {
                "input_index": kwargs["input_index"],
                "fetch_record_count": 5,
                "eligible_read_count": 1,
            }
        )
        return [read]

    def fake_analyze(_reads, _sites, **kwargs):
        decision = {
            "decision_id": "stream-decision",
            "read": read.name,
            "library_id": read.library_id,
            "alignment": {
                "reference_start": read.ref_start,
                "flag": read.alignment_flag,
                "cigar": read.cigar,
                "record_sha256": read.record_sha256,
                "occurrence": 0,
            },
            "target_strand": "CT",
            "source_prior_strand": "GA",
            "proposal_tier": "retain_current",
            "current": "A",
            "current_annotation": "msp",
            "current_interval": [110, 190],
            "current_molecular_interval": [10, 80],
            "current_annotation_ordinal": 0,
            "proposed": "TF1",
            "proposed_site_intervals": [[120, 140]],
            "proposed_site_edge_confidence": [[0.7, 0.8]],
            "posterior": 0.4,
            "sr_hypothesis_probability": 0.4,
            "current_posterior": 0.6,
            "baseline_hypothesis_probability": 0.6,
            "supported_tf_probability_vs_accessible": 0.45,
            "best_configuration_posterior_given_tf": 0.9,
            "molecule_probability": 0.4,
            "population_probability": 0.8,
            "source_support_reliability": 0.7,
            "log_bf_vs_current": -0.2,
            "templates_truncated": False,
        }
        kwargs["decision_callback"](read, decision)
        return {
            "applicable": True,
            "strands": ["CT", "GA"],
            "sites": [],
            "target_sites": [],
            "site_models": {},
            "decisions": [],
            "edge_refinement": {
                "tf": {"call_type": "tf", "sites": [], "harmonizations": [], "counts": {}},
                "nuc": {"call_type": "nuc", "sites": [], "harmonizations": [], "counts": {}},
            },
            "counts": {
                "from_msp": 1,
                "strong": 0,
                "review": 0,
                "retain_current": 1,
                "templates_truncated": 0,
            },
        }

    monkeypatch.setattr(
        strand_rescue_cli,
        "load_model_with_metadata",
        lambda _path: (object(), 3, "daf"),
    )
    monkeypatch.setattr(
        strand_rescue_cli,
        "build_llr_tables",
        lambda _model: (np.zeros(N_CTX), np.zeros(N_CTX)),
    )
    monkeypatch.setattr(strand_rescue_cli, "load_region_evidence", fake_load)
    monkeypatch.setattr(
        strand_rescue_cli,
        "assign_global_efficiency_steps",
        lambda reads, _model, **_kwargs: {"enabled": False, "reads": len(reads)},
    )
    monkeypatch.setattr(strand_rescue_cli, "discover_sites", lambda *_a, **_k: [])
    monkeypatch.setattr(
        strand_rescue_cli, "merge_forced_sites", lambda *_a, **_k: []
    )
    monkeypatch.setattr(strand_rescue_cli, "analyze_strand_rescue", fake_analyze)

    result = strand_rescue_cli.main(
        [
            "--bam",
            str(bam_path),
            "--preset",
            "dddb",
            "--region",
            "chr1:100-200",
            "--model",
            str(model_path),
            "--nuc-model",
            str(model_path),
            "--global-efficiency",
            "--molecule-collapse",
            "off",
            "--skip-nuc-edge-refinement",
            "--report-layout",
            "stream",
            "--proposal-tsv",
            str(proposal),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    report = json.loads(output.read_text())
    assert report["schema"] == "fiberhmm.strand_rescue.v6"
    assert report["schema_version"] == 6
    assert report["parameters"]["max_reads"] == 0
    assert report["parameters"]["control_flank"] == 2000
    assert report["parameters"]["center_radius"] == 10
    assert report["parameters"]["max_auto_sites"] == 0
    assert "decisions" not in report["strand_rescue"]
    assert "harmonizations" not in report["strand_rescue"]["edge_refinement"]["tf"]
    storage = report["strand_rescue"]["action_storage"]
    assert storage["totals"] == {
        "fetch_records": 5,
        "action_records": 1,
        "rescue_decisions": 1,
        "rescue_components": 1,
        "tf_edge_updates": 0,
        "nuc_edge_updates": 0,
    }

    from fiberhmm.cli.strand_rescue_annotate import (
        V5ActionStreamReader,
        _validate_v5_action_storage,
        _validate_v5_sidecar_files,
    )

    streams = _validate_v5_action_storage(report, output)
    assert len(streams) == 1
    stream = streams[0]
    _validate_v5_sidecar_files(stream)
    with V5ActionStreamReader(stream.bgzf_path, stream.manifest) as reader:
        assert reader.pop_for_ordinal(0) is None
        action = reader.pop_for_ordinal(2)
        assert action is not None
        assert action.read_name == "stream-read"
        assert action.rescues[0].alternative_q == 102
        validation = reader.finish(5)
    assert validation["rescue_component_count"] == 1
    proposal_lines = proposal.read_text().splitlines()
    assert len(proposal_lines) == 2
    assert proposal_lines[1].startswith("stream-decision\t")


def test_nanopore_hia5_uses_requested_248_hard_threshold():
    assert PRESETS["hia5-nanopore"]["prob_threshold"] == 248


def test_ddda_preset_resolves_packaged_model_independent_of_cwd(
    monkeypatch, tmp_path
):
    stale = tmp_path / "models"
    stale.mkdir()
    (stale / "ddda_TF.json").write_text("stale")
    shadow = tmp_path / "fiberhmm" / "models"
    shadow.mkdir(parents=True)
    (shadow / "ddda_TF.json").write_text("shadow")
    monkeypatch.chdir(tmp_path)

    resolved = Path(strand_rescue_inference.resolve_resource_path(
        PRESETS["ddda"]["model"]
    ))

    assert resolved.name == "ddda_TF.json"
    assert resolved.parent.name == "models"
    assert resolved.parent.parent.name == "fiberhmm"
    assert resolved.read_text() != "stale"


def test_stage_profiler_records_wall_time_and_unambiguous_rss(monkeypatch):
    timestamps = iter([10.0, 12.5, 14.0])
    monkeypatch.setattr(
        strand_rescue_cli.time, "perf_counter", lambda: next(timestamps)
    )
    monkeypatch.setattr(strand_rescue_cli, "_current_rss_bytes", lambda: 1000)
    monkeypatch.setattr(
        strand_rescue_cli, "_process_peak_rss_bytes", lambda: 2000
    )

    profiler = strand_rescue_cli._StageProfiler()
    profiler.mark("load", {"reads": 3})
    snapshot = profiler.snapshot(scope="before_write")

    assert snapshot["schema"] == "fiberhmm.performance.v1"
    assert snapshot["scope"] == "before_write"
    assert snapshot["total_wall_seconds"] == pytest.approx(4.0)
    assert snapshot["process_peak_rss_bytes"] == 2000
    assert snapshot["stages"] == [
        {
            "name": "load",
            "wall_seconds": 2.5,
            "rss_bytes_after": 1000,
            "process_peak_rss_bytes_after": 2000,
            "details": {"reads": 3},
        }
    ]


def test_stage_profiler_streams_each_completed_stage_as_jsonl(monkeypatch):
    timestamps = iter([10.0, 12.5])
    monkeypatch.setattr(
        strand_rescue_cli.time, "perf_counter", lambda: next(timestamps)
    )
    monkeypatch.setattr(strand_rescue_cli, "_current_rss_bytes", lambda: 1000)
    monkeypatch.setattr(
        strand_rescue_cli, "_process_peak_rss_bytes", lambda: 2000
    )
    stream = io.StringIO()

    profiler = strand_rescue_cli._StageProfiler(progress_stream=stream)
    profiler.mark("load", {"reads": 3})

    progress = json.loads(stream.getvalue())
    assert progress == {
        "schema": "fiberhmm.performance.progress.v1",
        "event": "stage_complete",
        "elapsed_wall_seconds": 2.5,
        "stage": {
            "name": "load",
            "wall_seconds": 2.5,
            "rss_bytes_after": 1000,
            "process_peak_rss_bytes_after": 2000,
            "details": {"reads": 3},
        },
    }


def test_strand_rescue_has_no_pacbio_alignment_orientation_preset():
    assert "hia5-pacbio" not in PRESETS


def test_nanopore_hard_threshold_discards_subthreshold_m6a():
    read = pysam.AlignedSegment()
    read.query_name = "nanopore"
    read.query_sequence = "C" * 15 + "A" + "C" * 14 + "A" + "C" * 19
    read.flag = 0
    read.set_tag("MM", "A+a.,0,0;")
    read.set_tag("ML", array("B", [247, 248]))

    observations, strand = hard_observations(
        read, "alignment", "nanopore-fiber", 3, 248
    )

    assert strand == "FWD"
    assert observations[15] >= 4096
    assert 0 <= observations[30] < 4096


def test_site_discovery_uses_every_tf_and_balances_strand_geometry():
    reads = _strong_source_reads(20, interval=(100, 120), score=0)
    reads.extend(
        _read(
            f"reverse-{index}",
            "REV",
            [109, 119],
            [4.0, 4.0],
            tfs=[IntervalCall(104, 126, 0)],
        )
        for index in range(5)
    )

    sites = discover_sites(
        reads,
        80,
        150,
        min_support=3,
        minimum_geometry_support=3,
        min_local_enrichment=1.0,
    )

    assert len(sites) == 1
    assert (sites[0].start, sites[0].end) == (102, 123)
    assert sites[0].support == {"FWD": 20, "REV": 5}


def test_site_discovery_separates_co_centered_tf_size_families():
    reads = []
    for strand in ("FWD", "REV"):
        for family, interval in enumerate(((90, 110), (80, 120))):
            reads.extend(
                _read(
                    f"{strand}-{family}-{index}",
                    strand,
                    [100],
                    [4.0],
                    tfs=[IntervalCall(*interval)],
                )
                for index in range(4)
            )

    diagnostics = {}
    sites = discover_sites(
        reads,
        60,
        140,
        min_support=3,
        minimum_geometry_support=3,
        min_local_enrichment=0.0,
        edge_compatibility_bp=5,
        diagnostics=diagnostics,
    )

    assert [(site.start, site.end) for site in sites] == [
        (80, 120),
        (90, 110),
    ]
    assert [site.support for site in sites] == [
        {"FWD": 4, "REV": 4},
        {"FWD": 4, "REV": 4},
    ]
    assert diagnostics == {
        "center_modes": 1,
        "geometry_families": 2,
        "separate_strand_maps": False,
        "per_strand_map_families": {},
        "matched_cross_strand_families": 0,
        "one_sided_stratum_families": 0,
        "eligible_before_cap": 2,
        "retained_after_cap": 2,
        "max_auto_sites": 0,
        "cap_bound": False,
    }

    capped_diagnostics = {}
    capped = discover_sites(
        reads,
        60,
        140,
        min_support=3,
        minimum_geometry_support=3,
        min_local_enrichment=0.0,
        edge_compatibility_bp=5,
        max_auto_sites=1,
        diagnostics=capped_diagnostics,
    )
    assert len(capped) == 1
    assert capped_diagnostics["eligible_before_cap"] == 2
    assert capped_diagnostics["cap_bound"] is True


def test_daf_strand_maps_are_discovered_then_consolidated():
    positions = list(range(94, 129))
    shared_steps = [
        2.0 if 101 <= position < 121 else -4.0
        for position in positions
    ]
    reads = [
        _read(
            f"CT-{index}",
            "CT",
            positions,
            shared_steps,
            tfs=[IntervalCall(100, 120)],
        )
        for index in range(4)
    ]
    reads.extend(
        _read(
            f"GA-{index}",
            "GA",
            positions,
            shared_steps,
            tfs=[IntervalCall(102, 122)],
        )
        for index in range(4)
    )
    diagnostics = {}

    sites = discover_sites(
        reads,
        80,
        150,
        min_support=3,
        minimum_geometry_support=3,
        min_local_enrichment=0.0,
        edge_compatibility_bp=5,
        separate_strand_maps=True,
        diagnostics=diagnostics,
    )

    assert len(sites) == 1
    assert (sites[0].start, sites[0].end) == (101, 121)
    assert sites[0].support == {"CT": 4, "GA": 4}
    assert sites[0].discovery_strata == ("CT", "GA")
    assert sites[0].consolidation_status == "matched_cross_strand"
    assert diagnostics["per_strand_map_families"] == {"CT": 1, "GA": 1}
    assert diagnostics["matched_cross_strand_families"] == 1
    assert diagnostics["one_sided_stratum_families"] == 0


def test_cross_strand_support_is_pooled_after_geometry_initialization():
    positions = list(range(94, 129))
    steps = [
        2.0 if 101 <= position < 121 else -4.0
        for position in positions
    ]
    reads = [
        _read(
            f"CT-subthreshold-{index}",
            "CT",
            positions,
            steps,
            tfs=[IntervalCall(100, 120)],
        )
        for index in range(7)
    ]
    reads.extend(
        _read(
            f"GA-subthreshold-{index}",
            "GA",
            positions,
            steps,
            tfs=[IntervalCall(102, 122)],
        )
        for index in range(8)
    )

    sites = discover_sites(
        reads,
        80,
        150,
        min_support=10,
        minimum_geometry_support=3,
        min_local_enrichment=0.0,
        edge_compatibility_bp=5,
        separate_strand_maps=True,
    )

    assert len(sites) == 1
    assert sites[0].support == {"CT": 7, "GA": 8}
    assert sites[0].consolidation_status == "matched_cross_strand"


def _latent_test_family(strand, interval, positions, steps, count=12):
    site = SiteTemplate(
        site_id=f"{strand}-{interval[0]}-{interval[1]}",
        start=interval[0],
        end=interval[1],
        center=int(round(sum(interval) / 2.0)),
        support={strand: count},
        start_mad=0.0,
        end_mad=0.0,
        local_enrichment=5.0,
    )
    records = []
    for index in range(count):
        read = _read(
            f"{strand}-latent-{index}",
            strand,
            positions,
            steps,
            tfs=[IntervalCall(*interval)],
        )
        records.append((read.tfs[0], read.molecule_id, read))
    return {"site": site, "center": site.center, "records": records}


def test_latent_geometry_pair_records_exact_strand_nonidentifiability():
    ct = _latent_test_family(
        "CT",
        (100, 120),
        [96, 100, 105, 110, 115, 119, 123],
        [-4.0, 1.0, 1.0, 1.0, 1.0, 1.0, -4.0],
    )
    ga = _latent_test_family(
        "GA",
        (99, 122),
        [96, 99, 105, 110, 115, 121, 123],
        [-4.0, 1.0, 1.0, 1.0, 1.0, 1.0, -4.0],
    )

    evidence = latent_geometry_pair_evidence(
        ct, ga, boundary_search_radius=5
    )

    assert evidence["log_bf_shared_vs_separate"] >= 0.0
    assert evidence["same_geometry_probability_equal_prior"] >= 0.5
    assert evidence["zero_differential_opportunity_strands"] == ["CT"]
    assert evidence["differential_opportunities_by_strand"]["CT"] == {
        "distinct_positions": 0,
        "positions": [],
        "informative_molecules": 0,
        "molecule_opportunities": 0,
        "hits": 0,
    }
    assert evidence["differential_opportunities_by_strand"]["GA"][
        "positions"
    ] == [99, 121]
    assert evidence["matching_score_source"] == (
        "fixed_boundary_conditional_log_bf"
    )
    assert evidence["opportunity_lattice_projection_by_strand"]["CT"] == {
        "representative_molecules": 12,
        "envelope_mapped_molecules": 12,
        "equivalent_projection_molecules": 12,
        "distinguishing_molecules": 0,
        "equivalent_projection_fraction": 1.0,
    }
    assert evidence["opportunity_lattice_projection_by_strand"]["GA"][
        "equivalent_projection_fraction"
    ] == 0.0


def test_opportunity_lattice_projection_reports_identifiable_coordinate_set():
    read = _read(
        "projection",
        "CT",
        [95, 100, 105, 110, 115, 120, 125],
        [-4.0, -4.0, 1.0, 1.0, 1.0, -4.0, -4.0],
        ref_start=90,
        ref_end=130,
    )

    left = strand_rescue_inference.opportunity_lattice_projection(
        read, (101, 119)
    )
    right = strand_rescue_inference.opportunity_lattice_projection(
        read, (102, 118)
    )

    assert left["signature"] == right["signature"] == [2, 5]
    assert left["minimal_projected_interval"] == [105, 116]
    assert left["equivalent_start_range"] == [101, 105]
    assert left["equivalent_end_range"] == [116, 120]
    assert left["opportunities"] == 3
    assert left["hits"] == 0


def test_latent_geometry_pair_rejects_distinct_supported_boundaries():
    positions = list(range(94, 131, 2))
    left_steps = [
        1.0 if 100 <= position < 120 else -4.0 for position in positions
    ]
    right_steps = [
        1.0 if 104 <= position < 124 else -4.0 for position in positions
    ]
    left = _latent_test_family(
        "CT", (100, 120), positions, left_steps, count=20
    )
    right = _latent_test_family(
        "GA", (104, 124), positions, right_steps, count=20
    )

    evidence = latent_geometry_pair_evidence(
        left, right, boundary_search_radius=5
    )

    assert evidence["log_bf_shared_vs_separate"] < -10.0
    assert evidence["same_geometry_probability_equal_prior"] < 0.001


def test_latent_geometry_pair_filters_nonpositive_grid_intervals():
    positions = list(range(90, 125))
    left = _latent_test_family(
        "CT",
        (100, 106),
        positions,
        [1.0 if 100 <= value < 106 else -4.0 for value in positions],
    )
    right = _latent_test_family(
        "GA",
        (108, 114),
        positions,
        [1.0 if 108 <= value < 114 else -4.0 for value in positions],
    )

    evidence = latent_geometry_pair_evidence(
        left, right, boundary_search_radius=5
    )

    assert evidence["candidate_grid_size"] > 0
    assert evidence["molecule_boundary_random_effect"]["available"] is False
    assert evidence["matching_log_score"] < 0.0


def test_zero_information_strand_pair_remains_explicitly_ambiguous():
    reads = [
        _read(
            f"CT-silent-{index}",
            "CT",
            [],
            [],
            tfs=[IntervalCall(100, 120)],
        )
        for index in range(4)
    ]
    reads.extend(
        _read(
            f"GA-silent-{index}",
            "GA",
            [],
            [],
            tfs=[IntervalCall(102, 122)],
        )
        for index in range(4)
    )

    sites = discover_sites(
        reads,
        80,
        150,
        min_support=3,
        minimum_geometry_support=3,
        min_local_enrichment=0.0,
        edge_compatibility_bp=5,
        separate_strand_maps=True,
    )

    assert [(site.start, site.end) for site in sites] == [
        (100, 120),
        (102, 122),
    ]
    assert all(
        site.consolidation_status == "one_sided_stratum_map"
        for site in sites
    )


def test_daf_incompatible_co_centered_strand_families_remain_separate():
    reads = [
        _read(
            f"CT-{index}",
            "CT",
            [110],
            [4.0],
            tfs=[IntervalCall(100, 120)],
        )
        for index in range(4)
    ]
    reads.extend(
        _read(
            f"GA-{index}",
            "GA",
            [110],
            [4.0],
            tfs=[IntervalCall(80, 140)],
        )
        for index in range(4)
    )
    diagnostics = {}

    sites = discover_sites(
        reads,
        60,
        160,
        min_support=3,
        minimum_geometry_support=3,
        min_local_enrichment=0.0,
        edge_compatibility_bp=5,
        separate_strand_maps=True,
        diagnostics=diagnostics,
    )

    assert [(site.start, site.end) for site in sites] == [
        (80, 140),
        (100, 120),
    ]
    assert {site.discovery_strata for site in sites} == {("CT",), ("GA",)}
    assert {
        site.consolidation_status for site in sites
    } == {"one_sided_stratum_map"}
    assert diagnostics["matched_cross_strand_families"] == 0
    assert diagnostics["one_sided_stratum_families"] == 2


def test_site_discovery_edge_compatibility_cannot_chain_tf_families():
    intervals = ((90, 130), (95, 125), (100, 120))
    reads = [
        _read(
            f"{strand}-{family}-{index}",
            strand,
            [110],
            [4.0],
            tfs=[IntervalCall(*interval)],
        )
        for strand in ("FWD", "REV")
        for family, interval in enumerate(intervals)
        for index in range(3)
    ]

    sites = discover_sites(
        reads,
        70,
        150,
        min_support=2,
        minimum_geometry_support=2,
        min_local_enrichment=0.0,
        edge_compatibility_bp=5,
    )

    assert len(sites) == 2
    assert (95, 125) not in {
        (site.start, site.end) for site in sites
    }


def test_center_catalog_site_templates_match_brute_scan_at_boundaries():
    rng = random.Random(20260716)
    reads = []
    for index in range(80):
        calls = []
        for _call_index in range(rng.randrange(1, 6)):
            start = rng.randrange(80, 321)
            calls.append(
                IntervalCall(
                    start,
                    start + rng.randrange(8, 121),
                    geometry_eligible=rng.random() > 0.08,
                )
            )
        read = _read(
            f"molecule-{index // 2}",
            "FWD" if index % 2 == 0 else "REV",
            [],
            [],
            nucs=calls,
            ref_start=50,
            ref_end=500,
        )
        if index % 9 == 0:
            read.alignment_blocks = ((50, 185), (205, 500))
        reads.append(read)

    # Exact center-radius and background-radius boundaries, plus half bases.
    for index, (start, end) in enumerate(
        [
            (162, 188),  # center 175: nearby boundary
            (212, 238),  # center 225: nearby boundary
            (137, 163),  # center 150: excluded inner boundary
            (136, 163),  # center 149.5: included background
            (87, 113),  # center 100: included background outer boundary
            (86, 113),  # center 99.5: excluded outside background
            (287, 313),  # center 300: included background outer boundary
        ]
    ):
        reads.append(
            _read(
                f"boundary-{index}",
                "FWD" if index % 2 == 0 else "REV",
                [],
                [],
                nucs=[IntervalCall(start, end)],
                ref_start=50,
                ref_end=500,
            )
        )

    catalog = _build_call_catalog(reads, "nuc")
    for center, seed, edge_radius in (
        (200, None, None),
        (175, (150, 200), 48),
        (225, (200, 250), 24),
        (260, None, None),
    ):
        arguments = dict(
            center=center,
            center_radius=25,
            local_background_radius=100,
            minimum_geometry_support=2,
            site_id="test",
            call_type="nuc",
            boundary_reliability_scale=24.0,
            source_boundary_margin=5,
            seed_interval=seed,
            edge_assignment_radius=edge_radius,
        )
        brute = _build_site_template(reads, **arguments)
        indexed = _build_site_template(
            reads,
            **arguments,
            call_catalog=catalog,
        )
        assert indexed == brute


def test_call_catalog_caches_source_coverage_by_read_call_and_margin(
    monkeypatch,
):
    # Reusing one immutable call object across reads guards against an unsafe
    # call-only identity cache.
    shared_call = IntervalCall(100, 200)
    complete = _read(
        "complete",
        "FWD",
        [],
        [],
        nucs=[shared_call],
        ref_start=50,
        ref_end=250,
    )
    gapped = _read(
        "gapped",
        "REV",
        [],
        [],
        nucs=[shared_call],
        ref_start=50,
        ref_end=250,
    )
    complete.alignment_blocks = ((50, 250),)
    gapped.alignment_blocks = ((50, 140), (160, 250))
    catalog = _build_call_catalog([complete, gapped], "nuc")

    original = strand_rescue_inference._source_interval_fully_maps
    evaluations = []

    def counted(read, call, margin):
        evaluations.append((read.name, id(call), margin))
        return original(read, call, margin)

    monkeypatch.setattr(
        strand_rescue_inference,
        "_source_interval_fully_maps",
        counted,
    )

    for _repeat in range(3):
        assert catalog.source_interval_fully_maps(complete, shared_call, 0)
        assert not catalog.source_interval_fully_maps(gapped, shared_call, 0)
    assert evaluations == [
        ("complete", id(shared_call), 0),
        ("gapped", id(shared_call), 0),
    ]

    # Margins have independent semantics, and false values are cached too.
    for _repeat in range(2):
        assert catalog.source_interval_fully_maps(complete, shared_call, 10)
        assert not catalog.source_interval_fully_maps(gapped, shared_call, 10)
    assert evaluations[-2:] == [
        ("complete", id(shared_call), 10),
        ("gapped", id(shared_call), 10),
    ]
    assert len(evaluations) == 4

    arguments = dict(
        center=150,
        center_radius=25,
        local_background_radius=100,
        minimum_geometry_support=1,
        site_id="cached",
        call_type="nuc",
        source_boundary_margin=7,
        call_catalog=catalog,
    )
    first = _build_site_template([complete, gapped], **arguments)
    second = _build_site_template([complete, gapped], **arguments)
    assert first == second
    assert evaluations[-2:] == [
        ("complete", id(shared_call), 7),
        ("gapped", id(shared_call), 7),
    ]
    assert len(evaluations) == 6


def _cluster_signature(clusters):
    return [
        (
            tuple(
                (call.start, call.end, molecule, strand)
                for call, molecule, strand in cluster["records"]
            ),
            cluster["starts"].value,
            cluster["ends"].value,
        )
        for cluster in clusters
    ]


@pytest.mark.parametrize("seed", range(24))
def test_dynamic_cluster_edge_index_matches_all_cluster_scan(seed):
    rng = random.Random(seed)
    records = []
    modes = [
        (-240, -80),
        (100, 140),
        (170, 230),
        (260, 420),
        (500, 680),
    ]
    for index in range(300):
        mode_start, mode_end = rng.choice(modes)
        start = mode_start + rng.randrange(-55, 56)
        end = mode_end + rng.randrange(-55, 56)
        if end <= start:
            end = start + 1
        strand = "FWD" if rng.randrange(2) == 0 else "REV"
        records.append(
            (
                IntervalCall(start, end),
                ("cohort", f"read-{index}", strand),
                strand,
            )
        )
    records.extend(
        [
            (IntervalCall(100, 120), ("cohort", "exact-a", "FWD"), "FWD"),
            (IntervalCall(125, 145), ("cohort", "exact-b", "REV"), "REV"),
            (IntervalCall(150, 170), ("cohort", "outside", "FWD"), "FWD"),
        ]
    )
    records.sort(
        key=lambda value: (
            value[0].center,
            value[0].end - value[0].start,
            value[0].start,
            value[0].end,
            value[1],
        )
    )
    center_radius = (0, 1, 10, 25, 40, 48)[seed % 6]
    edge_radius = (0, 1, 12, 24, 48)[seed % 5]

    brute = _cluster_edge_records(
        records,
        center_radius=center_radius,
        edge_assignment_radius=edge_radius,
        use_spatial_index=False,
    )
    indexed = _cluster_edge_records(
        records,
        center_radius=center_radius,
        edge_assignment_radius=edge_radius,
        use_spatial_index=True,
    )

    assert _cluster_signature(indexed) == _cluster_signature(brute)


def test_dynamic_cluster_edge_index_same_center_many_widths_matches_brute():
    lengths = range(20, 510, 10)
    records = []
    for length in lengths:
        start = 1000 - length // 2
        end = start + length
        for replicate in range(4):
            strand = "FWD" if replicate % 2 == 0 else "REV"
            records.append(
                (
                    IntervalCall(start, end),
                    ("cohort", f"length-{length}-{replicate}", strand),
                    strand,
                )
            )
    records.sort(
        key=lambda value: (
            value[0].center,
            value[0].end - value[0].start,
            value[0].start,
            value[0].end,
            value[1],
        )
    )

    brute = _cluster_edge_records(
        records,
        center_radius=0,
        edge_assignment_radius=4,
        use_spatial_index=False,
    )
    indexed = _cluster_edge_records(
        records,
        center_radius=0,
        edge_assignment_radius=4,
        use_spatial_index=True,
    )

    assert _cluster_signature(indexed) == _cluster_signature(brute)
    assert len(indexed) == len(lengths)


@pytest.mark.parametrize(
    ("intervals", "center_radius", "edge_radius"),
    [
        ([(24, 74), (26, 76), (50, 100)], 25, 48),
        ([(47, 147), (49, 149), (96, 196)], 48, 48),
        ([(10, 143), (12, 145), (59, 192)], 48, 48),
    ],
)
def test_dynamic_cluster_edge_index_tracks_median_bucket_crossings(
    intervals, center_radius, edge_radius
):
    records = [
        (
            IntervalCall(start, end),
            ("cohort", f"crossing-{index}", "FWD"),
            "FWD",
        )
        for index, (start, end) in enumerate(intervals)
    ]

    brute = _cluster_edge_records(
        records,
        center_radius=center_radius,
        edge_assignment_radius=edge_radius,
        use_spatial_index=False,
    )
    indexed = _cluster_edge_records(
        records,
        center_radius=center_radius,
        edge_assignment_radius=edge_radius,
        use_spatial_index=True,
    )

    assert _cluster_signature(indexed) == _cluster_signature(brute)
    assert len(indexed) == 1


def test_dynamic_cluster_edge_index_preserves_equal_distance_index_tie():
    records = [
        (IntervalCall(40, 60), ("cohort", "first", "FWD"), "FWD"),
        (IntervalCall(20, 80), ("cohort", "second", "REV"), "REV"),
        (IntervalCall(30, 70), ("cohort", "bridge", "FWD"), "FWD"),
    ]

    brute = _cluster_edge_records(
        records,
        center_radius=0,
        edge_assignment_radius=10,
        use_spatial_index=False,
    )
    indexed = _cluster_edge_records(
        records,
        center_radius=0,
        edge_assignment_radius=10,
        use_spatial_index=True,
    )

    assert _cluster_signature(indexed) == _cluster_signature(brute)
    assert [
        molecule[1]
        for _call, molecule, _strand in indexed[0]["records"]
    ] == ["first", "bridge"]


@pytest.mark.parametrize(
    ("second_interval", "expected_clusters"),
    [((148, 248), 1), ((149, 249), 2)],
)
def test_dynamic_cluster_edge_index_inclusive_radius_boundary(
    second_interval, expected_clusters
):
    records = [
        (IntervalCall(100, 200), ("cohort", "first", "FWD"), "FWD"),
        (
            IntervalCall(*second_interval),
            ("cohort", "second", "REV"),
            "REV",
        ),
    ]
    brute = _cluster_edge_records(
        records,
        center_radius=48,
        edge_assignment_radius=48,
        use_spatial_index=False,
    )
    indexed = _cluster_edge_records(
        records,
        center_radius=48,
        edge_assignment_radius=48,
        use_spatial_index=True,
    )

    assert _cluster_signature(indexed) == _cluster_signature(brute)
    assert len(indexed) == expected_clusters


def test_dynamic_cluster_edge_index_zero_radius_and_negative_coordinates():
    records = [
        (IntervalCall(-10, 10), ("cohort", "first", "FWD"), "FWD"),
        (IntervalCall(-10, 10), ("cohort", "duplicate", "REV"), "REV"),
        (IntervalCall(-11, 11), ("cohort", "wider", "FWD"), "FWD"),
    ]
    brute = _cluster_edge_records(
        records,
        center_radius=0,
        edge_assignment_radius=0,
        use_spatial_index=False,
    )
    indexed = _cluster_edge_records(
        records,
        center_radius=0,
        edge_assignment_radius=0,
        use_spatial_index=True,
    )

    assert _cluster_signature(indexed) == _cluster_signature(brute)
    assert len(indexed) == 2


def test_discover_edge_sites_three_axis_index_matches_brute(monkeypatch):
    reads = []
    intervals = ((100, 300), (140, 260))
    for strand in ("FWD", "REV"):
        for interval_index, interval in enumerate(intervals):
            for replicate in range(5):
                reads.append(
                    _read(
                        f"{strand}-{interval_index}-{replicate}",
                        strand,
                        [90, 200, 310],
                        [2.0, -2.0, 2.0],
                        nucs=[IntervalCall(*interval)],
                        ref_start=0,
                        ref_end=400,
                    )
                )
    arguments = dict(
        chrom_start=50,
        chrom_end=350,
        call_type="nuc",
        min_support=4,
        minimum_geometry_support=3,
        center_radius=25,
        edge_assignment_radius=20,
        max_boundary_mad=24.0,
        min_local_enrichment=0.0,
        local_background_radius=250,
        max_auto_sites=0,
        boundary_reliability_scale=24.0,
        source_boundary_margin=0,
    )
    indexed = discover_edge_sites(reads, **arguments)
    cluster_edge_records = strand_rescue_inference._cluster_edge_records

    def brute_cluster_edge_records(records, **kwargs):
        kwargs["use_spatial_index"] = False
        return cluster_edge_records(records, **kwargs)

    monkeypatch.setattr(
        strand_rescue_inference,
        "_cluster_edge_records",
        brute_cluster_edge_records,
    )
    brute = discover_edge_sites(reads, **arguments)

    assert indexed == brute
    assert [(site.start, site.end) for site in indexed] == [
        (100, 300),
        (140, 260),
    ]


def test_configuration_enumeration_allows_arbitrary_nonoverlapping_subsets():
    sites = [
        _site("a", 10, 20),
        _site("b", 30, 40),
        _site("c", 50, 60),
    ]
    names = {configuration.name for configuration in enumerate_configurations(sites)}
    assert {"TF:a", "TF:a,c", "TF:a,b,c", "A", "N"} <= names


def test_configuration_enumeration_rejects_overlapping_sites():
    sites = [_site("a", 10, 30), _site("b", 20, 40)]
    names = {configuration.name for configuration in enumerate_configurations(sites)}
    assert "TF:a,b" not in names


def test_tf_class_model_learns_broad_vs_two_component_configurations():
    sites = [
        _site("broad", 90, 150),
        _site("left", 90, 110),
        _site("right", 130, 150),
    ]
    configurations = enumerate_configurations(
        sites, include_nucleosome=False
    )
    reads = [
        _read(
            f"broad-{index}",
            "FWD",
            [100, 140],
            [1.0, 1.0],
            tfs=[IntervalCall(90, 150)],
            ref_start=70,
            ref_end=170,
        )
        for index in range(10)
    ]
    reads.extend(
        _read(
            f"pair-{index}",
            "FWD",
            [100, 140],
            [1.0, 1.0],
            tfs=[IntervalCall(90, 110), IntervalCall(130, 150)],
            ref_start=70,
            ref_end=170,
        )
        for index in range(10)
    )

    model = strand_rescue_inference.fit_tf_configuration_class_model(
        reads,
        sites,
        configurations,
        center_radius=25,
        pseudocount=0.5,
        source_stratum="FWD",
    )

    classes = {value["configuration"]: value for value in model["classes"]}
    assert model["schema"] == "fiberhmm.tf_configuration_class_model.v1"
    assert model["anchored_molecules"] == 20
    assert classes["TF:broad"]["anchored_molecule_support"] == 10
    assert classes["TF:left,right"]["anchored_molecule_support"] == 10
    assert classes["TF:broad"]["probability"] > 0.4
    assert classes["TF:left,right"]["probability"] > 0.4
    assert classes["TF:left"]["probability"] < 0.05
    assert classes["TF:right"]["probability"] < 0.05


def test_tf_class_model_excludes_overlapping_unmodeled_tf_topology():
    sites = [_site("modeled", 90, 110)]
    configurations = enumerate_configurations(
        sites, include_nucleosome=False
    )
    read = _read(
        "overlapping-unmodeled",
        "FWD",
        [92, 102],
        [1.0, 1.0],
        tfs=[IntervalCall(70, 96)],
        ref_start=60,
        ref_end=130,
    )

    model = strand_rescue_inference.fit_tf_configuration_class_model(
        [read],
        sites,
        configurations,
        center_radius=10,
        pseudocount=0.5,
        source_stratum="FWD",
    )

    assert model["eligible_molecules"] == 1
    assert model["modeled_molecules"] == 0
    assert model["unmodeled_topology_molecules"] == 1
    assert model["chemistry_only_molecules"] == 0


def test_iterative_tf_geometry_recovers_pooled_boundary_from_raw_chemistry():
    site = _site("latent", 100, 120)
    positions = list(range(95, 129))
    occupied_steps = [
        3.0 if 102 <= position < 122 else -4.0
        for position in positions
    ]
    reads = [
        _read(
            f"occupied-{strand}-{index}",
            strand,
            positions,
            occupied_steps,
            tfs=[IntervalCall(100, 120) if strand == "CT" else IntervalCall(102, 122)],
            ref_start=80,
            ref_end=150,
        )
        for strand in ("CT", "GA")
        for index in range(20)
    ]
    reads.extend(
        _read(
            f"accessible-{strand}-{index}",
            strand,
            positions,
            [-4.0] * len(positions),
            ref_start=80,
            ref_end=150,
        )
        for strand in ("CT", "GA")
        for index in range(20)
    )

    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        [site],
        boundary_search_radius=2,
        pseudocount=0.5,
    )

    assert model["schema"] == "fiberhmm.iterative_tf_class_geometry_model.v2"
    assert model["converged"] is True
    assert model["assignment_unit"] == "collapsed_molecule_configuration"
    assert model["geometry"][0]["selected_interval"] == [102, 122]
    assert model["geometry"][0]["boundary_status"] == (
        "identified_on_candidate_grid"
    )
    assert model["geometry"][0]["edge_identifiability"]["start"][
        "status"
    ] == "resolved_both_strata"
    assert model["geometry"][0]["edge_identifiability"]["start"][
        "candidate_coordinate_range"
    ] == [98, 102]
    assert model["geometry"][0]["edge_identifiability"]["start"][
        "selected_at_search_boundary"
    ] is True
    assert model["legacy_field_aliases"][
        "minimum_edge_information_spread_nats"
    ] == "minimum_edge_q_margin_to_best_non_equivalent_nats"
    assert model["modeled_molecules_by_strand"] == {"CT": 40, "GA": 40}
    assert model["evidence_backend"] == (
        "numpy_float64_slice_candidate_prefix_spatial_v2"
    )


def test_batched_interval_evidence_matches_scalar_reference_without_mutation():
    read = _read(
        "batch-evidence",
        "CT",
        [90, 93, 100, 107, 121, 130],
        [-4.0, 1.5, 3.0, -0.25, 2.0, -3.5],
        tfs=[IntervalCall(99, 110)],
        ref_start=80,
        ref_end=140,
    )
    intervals = [(88, 94), (93, 108), (100, 122), (122, 131)]
    steps_before = read.steps.copy()
    calls_before = [(call.start, call.end) for call in read.tfs]
    starts = np.asarray([start for start, _end in intervals], dtype=np.int64)
    ends = np.asarray([end for _start, end in intervals], dtype=np.int64)

    scores, opportunities, hits, left, right = read._interval_evidence_batch(
        starts, ends
    )
    scalar = [read.interval_evidence(start, end) for start, end in intervals]

    np.testing.assert_allclose(
        scores,
        [value[0] for value in scalar],
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_array_equal(opportunities, [value[1] for value in scalar])
    np.testing.assert_array_equal(hits, [value[2] for value in scalar])
    np.testing.assert_array_equal(right - left, opportunities)
    np.testing.assert_array_equal(read.steps, steps_before)
    assert [(call.start, call.end) for call in read.tfs] == calls_before


def test_efficiency_calibration_excludes_candidate_and_scales_both_states():
    read = _read(
        "efficiency-exclusion",
        "CT",
        [100, 110, 115, 125],
        [-1.0, -1.0, -1.0, -1.0],
        msps=[IntervalCall(95, 130)],
        ref_start=90,
        ref_end=140,
    )
    read.hits = np.asarray([True, False, False, False])
    read.contexts = np.zeros(4, dtype=np.int64)
    protected_hit = np.asarray([0.1], dtype=np.float64)
    accessible_hit = np.asarray([0.5], dtype=np.float64)

    result = strand_rescue_inference.calibrate_read_efficiency(
        read,
        protected_hit,
        accessible_hit,
        pseudo_count=0.0,
        min_opportunities=1,
        excluded_intervals=[(109, 121)],
        scale_protected_hit=True,
    )

    assert result["accessible_opportunities"] == 2
    assert result["excluded_candidate_opportunities"] == 2
    assert result["protected_hit_scaled_with_efficiency"] is True
    assert result["factor"] == pytest.approx(1.0)
    assert read.efficiency_factor == pytest.approx(1.0)
    assert read.steps[0] == pytest.approx(math.log(0.1 / 0.5))

    lower_efficiency = _read(
        "efficiency-symmetric",
        "CT",
        [100, 110, 115, 125],
        [-1.0, -1.0, -1.0, -1.0],
        msps=[IntervalCall(95, 130)],
        ref_start=90,
        ref_end=140,
    )
    lower_efficiency.hits = np.asarray([True, False, False, False])
    lower_efficiency.contexts = np.zeros(4, dtype=np.int64)
    symmetric = strand_rescue_inference.calibrate_read_efficiency(
        lower_efficiency,
        protected_hit,
        accessible_hit,
        pseudo_count=0.0,
        min_opportunities=1,
        scale_protected_hit=True,
    )
    assert symmetric["factor"] == pytest.approx(0.5)
    assert lower_efficiency.efficiency_factor == pytest.approx(0.5)
    assert lower_efficiency.steps[0] == pytest.approx(math.log(0.1 / 0.5))


def test_vectorized_efficiency_interval_union_matches_brute_reference():
    positions = np.asarray(
        [3, 8, 12, 17, 23, 29, 34, 41, 47, 55, 63, 71, 82, 94],
        dtype=np.int64,
    )
    read = _read(
        "efficiency-union",
        "CT",
        positions,
        np.zeros(len(positions), dtype=np.float64),
        msps=[
            IntervalCall(5, 36),
            IntervalCall(20, 50),
            IntervalCall(60, 90),
        ],
        ref_start=0,
        ref_end=100,
    )
    exclusions = [(10, 18), (15, 25), (34, 42), (70, 80)]
    expected = np.zeros(len(positions), dtype=bool)
    for call in read.msps:
        expected[
            int(np.searchsorted(positions, call.start, side="left")) :
            int(np.searchsorted(positions, call.end, side="left"))
        ] = True
    initially_selected = int(np.sum(expected))
    for start, end in exclusions:
        expected[
            int(np.searchsorted(positions, start, side="left")) :
            int(np.searchsorted(positions, end, side="left"))
        ] = False

    prepared = strand_rescue_inference._prepare_efficiency_exclusion_union(
        exclusions
    )
    observed, excluded = strand_rescue_inference._efficiency_opportunity_mask(
        read, prepared
    )
    np.testing.assert_array_equal(observed, expected)
    assert excluded == initially_selected - int(np.sum(expected))


def test_iterative_tf_geometry_retains_exact_projection_equivalence_set():
    site = _site("sparse", 100, 120)
    positions = [90, 105, 115, 130]
    reads = [
        _read(
            f"sparse-occupied-{index}",
            "CT" if index % 2 == 0 else "GA",
            positions,
            [-4.0, 3.0, 3.0, -4.0],
            tfs=[IntervalCall(100, 120)],
            ref_start=80,
            ref_end=140,
        )
        for index in range(20)
    ]

    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        [site],
        boundary_search_radius=2,
    )

    geometry = model["geometry"][0]
    assert geometry["selected_interval"] == [100, 120]
    assert geometry["maximizing_interval_count"] == 25
    assert geometry["start_equivalence_range"] == [98, 102]
    assert geometry["end_equivalence_range"] == [118, 122]
    assert geometry["boundary_status"] == (
        "identified_up_to_opportunity_projection"
    )


def test_iterative_tf_geometry_names_the_informative_stratum():
    site = _site("strand-resolved", 100, 120)
    reads = []
    for index in range(20):
        reads.append(
            _read(
                f"CT-blind-{index}",
                "CT",
                [95, 105, 115, 125],
                [-4.0, 3.0, 3.0, -4.0],
                tfs=[IntervalCall(100, 120)],
                ref_start=80,
                ref_end=140,
            )
        )
        reads.append(
            _read(
                f"GA-informative-{index}",
                "GA",
                [95, 99, 105, 115, 125],
                [-4.0, 3.0, 3.0, 3.0, -4.0],
                tfs=[IntervalCall(99, 120)],
                ref_start=80,
                ref_end=140,
            )
        )

    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        [site],
        boundary_search_radius=1,
    )

    geometry = model["geometry"][0]
    start = geometry["edge_identifiability"]["start"]
    assert geometry["selected_interval"][0] == 99
    assert start["status"] == "resolved_informative_stratum"
    assert start["ambiguous_strands"] == ["CT"]
    assert start["uniquely_resolving_strands"] == ["GA"]


def test_iterative_tf_geometry_resolves_complementary_strand_ambiguity():
    site = _site("complementary", 100, 120)
    reads = []
    for index in range(20):
        reads.append(
            _read(
                f"CT-complement-{index}",
                "CT",
                [100, 105, 115, 125],
                [3.0, 3.0, 3.0, -4.0],
                tfs=[IntervalCall(100, 120)],
                ref_start=80,
                ref_end=140,
            )
        )
        reads.append(
            _read(
                f"GA-complement-{index}",
                "GA",
                [99, 100, 105, 115, 125],
                [-4.0, 3.0, 3.0, 3.0, -4.0],
                tfs=[IntervalCall(100, 120)],
                ref_start=80,
                ref_end=140,
            )
        )

    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        [site],
        boundary_search_radius=1,
    )

    start = model["geometry"][0]["edge_identifiability"]["start"]
    assert start["selected_coordinate"] == 100
    assert start["maximizing_coordinates_by_strand"] == {
        "CT": [99, 100],
        "GA": [100],
    }
    # Both strata clear the explicit information floor; their ambiguity sets
    # intersect at one coordinate even though CT is not unique by itself.
    assert start["status"] == (
        "resolved_by_single_stratum_consistent_with_others"
    )

    complementary_reads = []
    for index in range(20):
        complementary_reads.append(
            _read(
                f"CT-intersection-{index}",
                "CT",
                [100, 105, 115, 125],
                [3.0, 3.0, 3.0, -4.0],
                tfs=[IntervalCall(100, 120)],
                ref_start=80,
                ref_end=140,
            )
        )
        complementary_reads.append(
            _read(
                f"GA-intersection-{index}",
                "GA",
                [99, 101, 105, 115, 125],
                [-4.0, 3.0, 3.0, 3.0, -4.0],
                tfs=[IntervalCall(100, 120)],
                ref_start=80,
                ref_end=140,
            )
        )
    complementary = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        complementary_reads,
        [site],
        boundary_search_radius=1,
        stratum_semantics="physical_complementary",
    )
    complementary_start = complementary["geometry"][0][
        "edge_identifiability"
    ]["start"]
    assert complementary_start["maximizing_coordinates_by_strand"] == {
        "CT": [99, 100],
        "GA": [100, 101],
    }
    assert complementary_start["status"] == (
        "resolved_jointly_complementary_strata"
    )

    diagnostic = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        complementary_reads,
        [site],
        boundary_search_radius=1,
        stratum_semantics="diagnostic_partition",
    )
    assert diagnostic["geometry"][0]["edge_identifiability"]["start"][
        "status"
    ] == "resolved_jointly_across_diagnostic_strata"


def test_iterative_tf_geometry_assigns_broad_vs_two_small_as_configurations():
    sites = [
        _site("broad", 90, 150),
        _site("left", 90, 110),
        _site("right", 130, 150),
    ]
    reads = [
        _read(
            f"iterative-broad-{index}",
            "CT" if index % 2 == 0 else "GA",
            [100, 120, 140],
            [3.0, 3.0, 3.0],
            tfs=[IntervalCall(90, 150)],
            ref_start=70,
            ref_end=170,
        )
        for index in range(30)
    ]
    reads.extend(
        _read(
            f"iterative-pair-{index}",
            "CT" if index % 2 == 0 else "GA",
            [100, 120, 140],
            [3.0, -6.0, 3.0],
            tfs=[IntervalCall(90, 110), IntervalCall(130, 150)],
            ref_start=70,
            ref_end=170,
        )
        for index in range(30)
    )
    reads.extend(
        _read(
            f"iterative-accessible-{index}",
            "CT" if index % 2 == 0 else "GA",
            [100, 120, 140],
            [-6.0, -6.0, -6.0],
            ref_start=70,
            ref_end=170,
        )
        for index in range(30)
    )
    configurations = enumerate_configurations(
        sites, include_nucleosome=False
    )

    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        sites,
        configurations,
        boundary_search_radius=0,
    )

    probabilities = dict(
        zip(
            model["configuration_names"],
            model["configuration_probabilities"],
        )
    )
    assert probabilities["A"] > 0.2
    assert probabilities["TF:broad"] > 0.2
    assert probabilities["TF:left,right"] > 0.2
    assert probabilities["TF:left"] < 0.05
    assert probabilities["TF:right"] < 0.05
    assert all(
        later + 1e-8 >= earlier
        for earlier, later in zip(
            model["objective_trace"], model["objective_trace"][1:]
        )
    )


def test_iterative_tf_geometry_collapses_exact_configuration_projection():
    sites = [
        _site("broad", 90, 150),
        _site("left", 90, 110),
        _site("right", 130, 150),
    ]
    reads = [
        _read(
            f"projection-equivalent-{index}",
            "CT" if index % 2 == 0 else "GA",
            [100, 140],
            [3.0, 3.0],
            tfs=[IntervalCall(90, 150)],
            ref_start=70,
            ref_end=170,
        )
        for index in range(20)
    ]
    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        sites,
        enumerate_configurations(sites, include_nucleosome=False),
        boundary_search_radius=0,
        minimum_molecule_opportunities=1,
    )

    equivalent = next(
        value
        for value in model["configuration_equivalence_classes"]
        if set(value["configuration_names"])
        == {"TF:broad", "TF:left,right"}
    )
    assert equivalent["individual_configuration_weights_identifiable"] is False
    assert equivalent["probability_sum"] > 0.5
    indices = [
        model["configuration_names"].index("TF:broad"),
        model["configuration_names"].index("TF:left,right"),
    ]
    assert len(
        {
            model["configuration_equivalence_class_ids"][index]
            for index in indices
        }
    ) == 1


def test_iterative_tf_geometry_keeps_nucleosome_seeded_molecule_as_unmodeled():
    site = _site("tf", 100, 120)
    read = _read(
        "nucleosome-seeded",
        "CT",
        [95, 105, 115, 125],
        [3.0, 3.0, 3.0, 3.0],
        nucs=[IntervalCall(90, 140)],
        ref_start=80,
        ref_end=150,
    )

    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        [read],
        [site],
        boundary_search_radius=1,
    )

    assert model["eligible_molecules"] == 1
    assert model["modeled_molecules"] == 1
    assert model["unmodeled_topology_molecules"] == 1
    assert model["nucleosome_seed_topology_molecules"] == 1
    assert model["modeled_molecule_fraction"] == 1.0


def test_iterative_tf_geometry_spatial_null_pays_placement_penalty():
    site = _site("anchored", 100, 120)
    positions = list(range(94, 127))
    read = _read(
        "anchored-pattern",
        "CT",
        positions,
        [3.0 if 100 <= position < 120 else -5.0 for position in positions],
        ref_start=80,
        ref_end=140,
    )
    envelope = IntervalCall(94, 126)
    structured_candidates = [
        [
            (start, end)
            for start in range(94, 107)
            for end in range(114, 127)
            if end > start
        ]
    ]
    candidates = strand_rescue_inference._spatial_null_candidate_intervals(
        envelope, structured_candidates
    )
    anchored_llr = read.interval_evidence(100, 120)[0]
    marginal_llr = (
        strand_rescue_inference._spatial_null_configuration_log_likelihood(
            read, candidates
        )
    )

    assert (100, 120) not in candidates
    assert marginal_llr < anchored_llr
    assert marginal_llr > 0.0


def test_spatial_null_edge_localization_does_not_credit_distant_site():
    positions = list(range(80, 241))
    read = _read(
        "localized-p0",
        "GA",
        positions,
        [3.0 if 106 <= position < 120 else -4.0 for position in positions],
        ref_start=70,
        ref_end=250,
    )
    candidates = strand_rescue_inference._spatial_null_candidate_intervals(
        IntervalCall(80, 241),
        [[]],
        minimum_width=14,
        maximum_width=14,
    )

    likelihoods, localization = (
        strand_rescue_inference._spatial_null_configuration_evidence(
            [read],
            candidates,
            localization_edges=[(0, 94, 106), (0, 194, 206)],
        )
    )

    assert likelihoods.shape == (1,)
    assert localization[0, 0] > 0.99
    assert localization[0, 1] < 1e-20


def test_iterative_tf_geometry_off_register_footprint_stays_in_spatial_null():
    site = _site("anchored", 100, 120)
    positions = list(range(80, 141))
    reads = [
        _read(
            f"off-register-{index}",
            "CT" if index % 2 == 0 else "GA",
            positions,
            [3.0 if 128 <= position < 138 else -4.0 for position in positions],
            tfs=[IntervalCall(128, 138)],
            ref_start=70,
            ref_end=150,
        )
        for index in range(48)
    ]

    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        [site],
        boundary_search_radius=2,
    )
    probabilities = dict(
        zip(model["configuration_names"], model["configuration_probabilities"])
    )

    assert probabilities["P0:unanchored_single_interval"] > 0.8
    assert probabilities["TF:anchored"] < 0.05
    assert model["geometry"][0]["selected_interval"][1] <= 122


def test_iterative_tf_geometry_accessible_null_keeps_anchored_tf_at_floor():
    site = _site("null-anchor", 100, 120)
    positions = list(range(80, 141))
    reads = [
        _read(
            f"accessible-null-{index}",
            "CT" if index % 2 == 0 else "GA",
            positions,
            [-4.0] * len(positions),
            ref_start=70,
            ref_end=150,
        )
        for index in range(80)
    ]

    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        [site],
        boundary_search_radius=2,
    )
    probabilities = dict(
        zip(model["configuration_names"], model["configuration_probabilities"])
    )

    assert probabilities["A"] > 0.95
    assert probabilities["TF:null-anchor"] < 0.02


def test_iterative_tf_geometry_reports_null_width_misspecification():
    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        [],
        [_site("too-wide-for-null", 100, 130)],
        boundary_search_radius=2,
        spatial_null_minimum_width=10,
        spatial_null_maximum_width=20,
    )
    spatial_null = model["spatial_null_component"]

    assert spatial_null["can_represent_all_anchored_candidate_widths"] is False
    assert max(
        spatial_null["anchored_candidate_widths_outside_null_prior"]
    ) > 20


def test_iterative_tf_geometry_reports_realized_null_width_support():
    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        [],
        [_site("envelope-capped-null", 100, 120)],
        boundary_search_radius=2,
        analysis_envelope=(95, 135),
        spatial_null_minimum_width=1,
        spatial_null_maximum_width=80,
    )

    assert model["spatial_null_requested_width_range"] == [1, 80]
    assert model["spatial_null_effective_width_range"] == [1, 40]
    assert model["spatial_null_component"]["effective_width_range"] == [1, 40]


def test_diffuse_null_quadrature_scales_to_high_opportunity_count():
    opportunity_count = 300
    step = 0.01
    read = _read(
        "diffuse-quadrature",
        "CT",
        list(range(opportunity_count)),
        [step] * opportunity_count,
        ref_start=0,
        ref_end=opportunity_count,
    )
    observed = (
        strand_rescue_inference._diffuse_unmodeled_configuration_log_likelihood(
            read, IntervalCall(0, opportunity_count)
        )
    )
    expected = (
        math.log(math.expm1(step * (opportunity_count + 1)))
        - math.log(math.expm1(step))
        - math.log(opportunity_count + 1)
    )

    assert observed == pytest.approx(expected, abs=1e-10)


def test_iterative_tf_geometry_uniform_initialization_reaches_same_solution():
    site = _site("stable", 100, 120)
    positions = list(range(95, 128))
    reads = [
        _read(
            f"stable-occupied-{index}",
            "CT" if index % 2 == 0 else "GA",
            positions,
            [3.0 if 102 <= position < 122 else -4.0 for position in positions],
            tfs=[IntervalCall(100, 120)],
            ref_start=80,
            ref_end=140,
        )
        for index in range(50)
    ]
    reads.extend(
        _read(
            f"stable-accessible-{index}",
            "CT" if index % 2 == 0 else "GA",
            positions,
            [-4.0] * len(positions),
            ref_start=80,
            ref_end=140,
        )
        for index in range(50)
    )

    call_seeded = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        [site],
        boundary_search_radius=2,
        initialization_mode="ordinary_calls",
        tol=1e-10,
    )
    uniform = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        [site],
        boundary_search_radius=2,
        initialization_mode="uniform",
        tol=1e-10,
    )

    assert call_seeded["geometry"][0]["selected_interval"] == [102, 122]
    assert uniform["geometry"][0]["selected_interval"] == [102, 122]
    assert call_seeded["configuration_names"] == uniform["configuration_names"]
    assert call_seeded["configuration_probabilities"] == pytest.approx(
        uniform["configuration_probabilities"], abs=1e-7
    )


def test_iterative_tf_geometry_zero_pseudocount_empty_cohort_is_well_formed():
    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        [],
        [_site("empty", 100, 120)],
        boundary_search_radius=1,
        pseudocount=0.0,
    )

    assert model["eligible_molecules"] == 0
    assert model["configuration_names"] == [
        "A",
        "TF:empty",
        "P0:unanchored_single_interval",
        "U:diffuse_iid_opportunity_protection",
    ]
    assert model["configuration_probabilities"] == pytest.approx([0.25] * 4)


def test_iterative_tf_geometry_hierarchical_prior_is_catalog_size_invariant():
    sites = [_site("left", 90, 100), _site("right", 120, 130)]
    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        [],
        sites,
        boundary_search_radius=1,
        pseudocount=0.5,
    )
    probabilities = dict(
        zip(model["configuration_names"], model["configuration_probabilities"])
    )

    assert probabilities["A"] == pytest.approx(0.25)
    assert sum(
        probability
        for name, probability in probabilities.items()
        if name.startswith("TF:")
    ) == pytest.approx(0.25)
    assert probabilities["P0:unanchored_single_interval"] == pytest.approx(0.25)
    assert probabilities["U:diffuse_iid_opportunity_protection"] == pytest.approx(
        0.25
    )


def test_iterative_tf_geometry_fixed_analysis_envelope_is_catalog_comparable():
    positions = list(range(70, 151))
    reads = [
        _read(
            f"fixed-envelope-{index}",
            "CT" if index % 2 == 0 else "GA",
            positions,
            [3.0 if 100 <= position < 120 else -4.0 for position in positions],
            ref_start=60,
            ref_end=160,
        )
        for index in range(20)
    ]
    sites = [_site("left", 90, 98), _site("center", 100, 120)]
    common_null_exclusions = [
        (start, end)
        for site in sites
        for start in range(site.start - 2, site.start + 3)
        for end in range(site.end - 2, site.end + 3)
        if end > start
    ]
    compact = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        [sites[1]],
        boundary_search_radius=2,
        analysis_envelope=(75, 145),
        spatial_null_exclusion_intervals=common_null_exclusions,
    )
    expanded_catalog = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        sites,
        boundary_search_radius=2,
        analysis_envelope=(75, 145),
        spatial_null_exclusion_intervals=common_null_exclusions,
    )

    assert compact["envelope"] == [75, 145]
    assert expanded_catalog["envelope"] == [75, 145]
    assert compact["analysis_envelope_explicit"] is True
    assert compact["modeled_molecules"] == expanded_catalog["modeled_molecules"]
    assert compact["spatial_null_component"]["candidate_envelope"] == [75, 145]
    assert expanded_catalog["spatial_null_component"]["candidate_envelope"] == [
        75,
        145,
    ]
    assert compact["spatial_null_exclusion_sha256"] == expanded_catalog[
        "spatial_null_exclusion_sha256"
    ]
    assert compact["spatial_null_component"]["candidate_interval_count"] == (
        expanded_catalog["spatial_null_component"]["candidate_interval_count"]
    )


def test_iterative_tf_geometry_rejects_analysis_envelope_inside_candidate_grid():
    with pytest.raises(
        ValueError, match="analysis envelope must contain every geometry candidate"
    ):
        strand_rescue_inference.fit_iterative_tf_class_geometry_model(
            [],
            [_site("center", 100, 120)],
            boundary_search_radius=3,
            analysis_envelope=(98, 122),
        )


def test_iterative_tf_geometry_rejects_null_universe_missing_anchor():
    with pytest.raises(
        ValueError,
        match="spatial-null exclusion universe must include every anchored candidate",
    ):
        strand_rescue_inference.fit_iterative_tf_class_geometry_model(
            [],
            [_site("center", 100, 120)],
            boundary_search_radius=1,
            analysis_envelope=(75, 145),
            spatial_null_exclusion_intervals=[(99, 119)],
        )


def test_iterative_tf_geometry_frozen_scorer_does_not_refit_or_mutate_model():
    site = _site("heldout", 100, 120)
    positions = list(range(95, 126))
    training = [
        _read(
            f"train-{index}",
            "CT" if index % 2 == 0 else "GA",
            positions,
            [3.0 if 101 <= position < 121 else -4.0 for position in positions],
            tfs=[IntervalCall(100, 120)],
            ref_start=80,
            ref_end=140,
        )
        for index in range(40)
    ]
    heldout = [
        _read(
            f"holdout-{index}",
            "CT" if index % 2 == 0 else "GA",
            positions,
            [3.0 if 101 <= position < 121 else -4.0 for position in positions],
            ref_start=80,
            ref_end=140,
        )
        for index in range(12)
    ]
    configurations = enumerate_configurations(
        [site], include_nucleosome=False
    )
    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        training,
        [site],
        configurations,
        boundary_search_radius=2,
    )
    before = json.dumps(model, sort_keys=True)

    score = strand_rescue_inference.score_iterative_tf_class_geometry_model(
        heldout,
        [site],
        configurations,
        model,
        include_molecule_records=True,
    )

    assert json.dumps(model, sort_keys=True) == before
    assert score["frozen_parameters"] is True
    assert score["eligible_molecules"] == len(heldout)
    assert len(score["molecules"]) == len(heldout)
    assert score["raw_mixture_log_likelihood_ratio"] > 0.0
    assert score["component_names"] == model["configuration_names"]

    shifted_site = _site("heldout", 101, 121)
    with pytest.raises(
        ValueError, match="frozen model seed coordinates do not match inputs"
    ):
        strand_rescue_inference.score_iterative_tf_class_geometry_model(
            heldout,
            [shifted_site],
            enumerate_configurations([shifted_site], include_nucleosome=False),
            model,
        )


def test_iterative_tf_geometry_reports_mutually_exclusive_family_substates():
    narrow = _site("family_narrow", 100, 120)
    narrow.family_id = "tf_family_1"
    narrow.substate_id = "narrow"
    narrow.seed_provenance_stratum = "CT"
    broad = _site("family_broad", 98, 123)
    broad.family_id = "tf_family_1"
    broad.substate_id = "broad"
    broad.seed_provenance_stratum = "GA"
    sites = [narrow, broad]
    configurations = enumerate_configurations(
        sites, include_nucleosome=False
    )
    assert [configuration.site_indices for configuration in configurations] == [
        (),
        (0,),
        (1,),
    ]

    positions = list(range(94, 128))
    reads = [
        _read(
            f"family-{index}",
            "CT" if index % 2 == 0 else "GA",
            positions,
            [
                3.0 if 99 <= position < 122 else -4.0
                for position in positions
            ],
            tfs=[IntervalCall(100, 120)],
            ref_start=80,
            ref_end=140,
        )
        for index in range(40)
    ]
    raw_before = [
        (
            read.steps.copy(),
            [(call.start, call.end) for call in read.tfs],
        )
        for read in reads
    ]
    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        sites,
        configurations,
        boundary_search_radius=1,
    )

    assert len(model["tf_families"]) == 1
    family = model["tf_families"][0]
    assert family["family_id"] == "tf_family_1"
    assert {value["substate_id"] for value in family["substates"]} == {
        "narrow",
        "broad",
    }
    expected_family_probability = sum(
        model["configuration_probabilities"][index]
        for index in family["configuration_indices"]
    )
    assert family["marginal_probability"] == pytest.approx(
        expected_family_probability
    )

    score = strand_rescue_inference.score_iterative_tf_class_geometry_model(
        reads,
        sites,
        configurations,
        model,
        include_molecule_records=True,
    )
    molecule_family_sum = 0.0
    for molecule in score["molecules"]:
        expected_posterior = sum(
            molecule["component_posteriors"][
                model["configuration_names"][index]
            ]
            for index in family["configuration_indices"]
        )
        observed_posterior = molecule["family_posteriors"]["tf_family_1"]
        assert observed_posterior == pytest.approx(expected_posterior)
        molecule_family_sum += observed_posterior
    assert score["family_effective_molecule_support"][
        "tf_family_1"
    ] == pytest.approx(molecule_family_sum)
    for read, (steps, calls) in zip(reads, raw_before):
        np.testing.assert_array_equal(read.steps, steps)
        assert [(call.start, call.end) for call in read.tfs] == calls


def test_iterative_tf_geometry_rejects_two_substates_in_one_configuration():
    left = _site("left", 90, 100)
    left.family_id = "same_family"
    left.substate_id = "left"
    right = _site("right", 110, 120)
    right.family_id = "same_family"
    right.substate_id = "right"
    with pytest.raises(
        ValueError,
        match="multiple substates of one site-consensus state",
    ):
        strand_rescue_inference.fit_iterative_tf_class_geometry_model(
            [],
            [left, right],
            [
                strand_rescue_inference.Configuration("A", ()),
                strand_rescue_inference.Configuration("TF:both", (0, 1)),
            ],
            boundary_search_radius=0,
        )


def test_family_sum_remains_identifiable_when_substates_are_lattice_equivalent():
    narrow = _site("equivalent_narrow", 100, 120)
    narrow.family_id = "equivalent_family"
    narrow.substate_id = "narrow"
    broad = _site("equivalent_broad", 101, 121)
    broad.family_id = "equivalent_family"
    broad.substate_id = "broad"
    sites = [narrow, broad]
    configurations = enumerate_configurations(
        sites, include_nucleosome=False
    )
    reads = [
        _read(
            f"equivalent-{index}",
            "CT" if index % 2 == 0 else "GA",
            [90, 105, 115, 130],
            [-4.0, 3.0, 3.0, -4.0],
            ref_start=80,
            ref_end=150,
        )
        for index in range(40)
    ]

    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        sites,
        configurations,
        boundary_search_radius=0,
    )
    family = model["tf_families"][0]
    substate_by_id = {
        value["substate_id"]: value for value in family["substates"]
    }

    assert family["marginal_probability_identifiable"] is True
    assert substate_by_id["narrow"][
        "marginal_probability_identifiable"
    ] is False
    assert substate_by_id["broad"][
        "marginal_probability_identifiable"
    ] is False
    assert (
        substate_by_id["narrow"][
            "component_likelihood_equivalence_class_ids"
        ]
        == substate_by_id["broad"][
            "component_likelihood_equivalence_class_ids"
        ]
    )


def test_hierarchical_prior_is_neutral_to_family_substate_expansion():
    first_a = _site("first_a", 90, 100)
    first_a.family_id = "family_1"
    first_a.substate_id = "a"
    first_b = _site("first_b", 91, 101)
    first_b.family_id = "family_1"
    first_b.substate_id = "b"
    second = _site("second", 120, 130)
    second.family_id = "family_2"
    second.substate_id = "only"
    sites = [first_a, first_b, second]
    configurations = enumerate_configurations(
        sites, include_nucleosome=False
    )

    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        [],
        sites,
        configurations,
        boundary_search_radius=0,
        pseudocount=0.6,
    )
    prior = model["hierarchical_weight_prior"]
    pseudocounts = np.asarray(prior["component_pseudocounts"])
    configuration_by_name = {
        name: index for index, name in enumerate(model["configuration_names"])
    }
    family_1_prior = sum(
        pseudocounts[index]
        for name, index in configuration_by_name.items()
        if name.startswith("TF:")
        and any(token in name for token in ("first_a", "first_b"))
    )
    family_2_prior = sum(
        pseudocounts[index]
        for name, index in configuration_by_name.items()
        if name.startswith("TF:") and "second" in name
    )

    assert prior["anchored_tf_substate_expansion_neutral"] is True
    assert len(prior["anchored_tf_family_set_groups"]) == 3
    assert all(
        value["total_pseudocount"] == pytest.approx(0.2)
        for value in prior["anchored_tf_family_set_groups"]
    )
    assert family_1_prior == pytest.approx(family_2_prior)
    assert family_1_prior == pytest.approx(0.4)


def test_iterative_tf_geometry_multistart_selects_best_training_objective():
    site = _site("multistart", 100, 120)
    positions = list(range(94, 127))
    reads = [
        _read(
            f"multistart-{index}",
            "CT" if index % 2 == 0 else "GA",
            positions,
            [3.0 if 102 <= position < 122 else -4.0 for position in positions],
            tfs=[IntervalCall(100, 120)],
            ref_start=80,
            ref_end=140,
        )
        for index in range(40)
    ]

    model = (
        strand_rescue_inference.fit_multistart_iterative_tf_class_geometry_model(
            reads,
            [site],
            boundary_search_radius=2,
        )
    )

    assert model["initialization_strategy"] == (
        "deterministic_multistart_maximum_penalized_training_objective"
    )
    assert len(model["multistart"]) == 6
    selected = next(
        record
        for record in model["multistart"]
        if record["start"] == model["selected_multistart"]
    )
    assert model["objective"] == pytest.approx(
        max(record["objective"] for record in model["multistart"])
    )
    assert selected["objective_delta_from_selected"] == pytest.approx(0.0)
    assert [record["start"] for record in model["multistart"]] == [
        "calls.seed",
        "calls.shift_left",
        "calls.shift_right",
        "calls.expanded",
        "calls.contracted",
        "uniform.seed",
    ]
    assert model["multistart_attempted"] == 6
    assert model["multistart_fitted"] == 6
    assert model["multistart_skipped"] == []


def test_iterative_tf_geometry_multistart_skips_invalid_expanded_topology():
    sites = [_site("left", 100, 110), _site("right", 112, 122)]
    configurations = [
        strand_rescue_inference.Configuration("A", ()),
        strand_rescue_inference.Configuration("TF:left", (0,)),
        strand_rescue_inference.Configuration("TF:right", (1,)),
        strand_rescue_inference.Configuration("TF:left+right", (0, 1)),
    ]
    positions = list(range(75, 148))
    reads = [
        _read(
            f"close-sites-{index}",
            "CT" if index % 2 == 0 else "GA",
            positions,
            [
                3.0
                if 100 <= position < 110 or 112 <= position < 122
                else -4.0
                for position in positions
            ],
            tfs=[IntervalCall(100, 110), IntervalCall(112, 122)],
            ref_start=70,
            ref_end=150,
        )
        for index in range(24)
    ]

    model = strand_rescue_inference.fit_multistart_iterative_tf_class_geometry_model(
        reads,
        sites,
        configurations,
        boundary_search_radius=2,
    )

    assert model["multistart_attempted"] == 6
    assert model["multistart_fitted"] == 5
    assert [record["start"] for record in model["multistart_skipped"]] == [
        "calls.expanded"
    ]
    assert model["multistart_skipped"][0]["status"] == (
        "skipped_invalid_configuration_topology"
    )
    assert model["selected_multistart"] != "calls.expanded"


def test_iterative_tf_geometry_multistart_reports_duplicate_start():
    site = _site("narrow", 100, 104)
    positions = list(range(78, 127))
    reads = [
        _read(
            f"narrow-start-{index}",
            "CT",
            positions,
            [3.0 if 100 <= position < 104 else -4.0 for position in positions],
            tfs=[IntervalCall(100, 104)],
            ref_start=70,
            ref_end=135,
        )
        for index in range(16)
    ]

    model = strand_rescue_inference.fit_multistart_iterative_tf_class_geometry_model(
        reads, [site], boundary_search_radius=2
    )

    assert model["multistart_attempted"] == 6
    assert model["multistart_fitted"] == 5
    assert any(
        record["start"] == "calls.contracted"
        and record["status"] == "skipped_duplicate_initialization"
        for record in model["multistart_skipped"]
    )


def test_iterative_tf_geometry_multistart_all_invalid_has_clear_error():
    sites = [_site("left", 100, 115), _site("right", 110, 125)]
    configurations = [
        strand_rescue_inference.Configuration("A", ()),
        strand_rescue_inference.Configuration("TF:invalid", (0, 1)),
    ]

    with pytest.raises(
        ValueError,
        match="all multistart initial geometries overlap within a structured configuration",
    ):
        strand_rescue_inference.fit_multistart_iterative_tf_class_geometry_model(
            [], sites, configurations, boundary_search_radius=2
        )


def test_iterative_tf_geometry_profiles_other_boundary_for_edge_information():
    site = _site("edge-profile", 100, 120)
    positions = list(range(117, 124))
    reads = [
        _read(
            f"edge-profile-{index}",
            "CT" if index % 2 == 0 else "GA",
            positions,
            [3.0 if position < 120 else -4.0 for position in positions],
            tfs=[IntervalCall(100, 120)],
            ref_start=70,
            ref_end=150,
        )
        for index in range(32)
    ]

    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        [site],
        boundary_search_radius=2,
    )
    edges = model["geometry"][0]["edge_identifiability"]

    # No represented base lies near the start grid.  End evidence must not be
    # recycled into a false claim that the start itself is informative.
    assert edges["start"]["pooled_effective_edge_opportunities"] == 0.0
    assert edges["start"]["pooled_information_spread_nats"] == pytest.approx(0.0)
    assert edges["start"]["pooled_information_qualified"] is False
    assert edges["start"]["operationally_identified"] is False
    assert edges["end"]["pooled_effective_edge_opportunities"] > 0.0
    assert edges["end"]["pooled_information_spread_nats"] > 0.0


def test_iterative_tf_geometry_uses_runner_up_margin_not_profile_range():
    site = _site("runner-up", 100, 120)
    positions = [90, 98, 99, 100, 105, 115, 120, 125, 130]
    reads = [
        _read(
            f"runner-up-{index}",
            "CT" if index % 2 == 0 else "GA",
            positions,
            [-4.0, -10.0, -0.01, 3.0, 3.0, 3.0, -4.0, -4.0, -4.0],
            tfs=[IntervalCall(100, 120)],
            ref_start=70,
            ref_end=150,
        )
        for index in range(24)
    ]

    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        [site],
        boundary_search_radius=2,
    )
    start = model["geometry"][0]["edge_identifiability"]["start"]

    assert start["pooled_profile_range_nats"] > math.log(10.0)
    assert (
        start["pooled_information_margin_to_best_non_equivalent_nats"]
        < math.log(10.0)
    )
    assert start["pooled_information_qualified"] is False
    assert start["operationally_identified"] is False


def test_iterative_tf_geometry_denests_spatial_null_at_zero_radius():
    site = _site("solo", 100, 120)
    positions = list(range(80, 140))
    reads = [
        _read(
            f"zero-radius-{index}",
            "CT",
            positions,
            [3.0 if 100 <= position < 120 else -4.0 for position in positions],
            tfs=[IntervalCall(100, 120)],
            ref_start=70,
            ref_end=150,
        )
        for index in range(40)
    ]
    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        [site],
        boundary_search_radius=0,
    )

    tf_index = model["configuration_names"].index("TF:solo")
    p0_index = model["configuration_names"].index(
        "P0:unanchored_single_interval"
    )
    assert model["converged"] is True
    assert model["spatial_null_component"][
        "anchored_reference_intervals_excluded"
    ] is True
    assert model["component_likelihood_equivalence_class_ids"][tf_index] != (
        model["component_likelihood_equivalence_class_ids"][p0_index]
    )


def test_iterative_tf_geometry_exact_lattice_ties_survive_high_coverage():
    site = _site("high-depth-sparse", 100, 120)
    reads = [
        _read(
            f"high-depth-sparse-{index}",
            "CT" if index % 2 == 0 else "GA",
            [90, 105, 115, 130],
            [-4.0, 3.0, 3.0, -4.0],
            tfs=[IntervalCall(100, 120)],
            ref_start=70,
            ref_end=150,
        )
        for index in range(2000)
    ]
    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        [site],
        boundary_search_radius=2,
    )

    geometry = model["geometry"][0]
    assert geometry["maximizing_interval_count"] == 25
    assert len(
        geometry["selected_opportunity_projection_equivalent_intervals"]
    ) == 25
    assert geometry["boundary_status"] == (
        "identified_up_to_opportunity_projection"
    )


def test_iterative_tf_geometry_excludes_zero_opportunity_molecules_from_support():
    site = _site("opportunity-gated", 100, 120)
    informative = [
        _read(
            f"informative-{index}",
            "CT",
            [85, 105, 115, 135],
            [-4.0, 3.0, 3.0, -4.0],
            tfs=[IntervalCall(100, 120)],
            ref_start=70,
            ref_end=150,
        )
        for index in range(10)
    ]
    uninformative = [
        _read(
            f"uninformative-{index}",
            "GA",
            [60, 160],
            [-4.0, -4.0],
            ref_start=70,
            ref_end=150,
        )
        for index in range(90)
    ]
    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        [*informative, *uninformative],
        [site],
        boundary_search_radius=2,
    )

    assert model["eligible_molecules"] == 100
    assert model["modeled_molecules"] == 10
    assert model["uninformative_molecules"] == 90
    assert model["uninformative_molecules_by_strand"] == {"GA": 90}
    assert model["modeled_molecules_by_strand"] == {"CT": 10}
    assert model["geometry"][0]["effective_molecule_support"] <= 10.0


def test_iterative_tf_geometry_reports_site_opportunity_coverage_separately():
    site = _site("padding-only", 100, 120)
    reads = [
        _read(
            f"padding-only-{index}",
            "CT",
            [82, 84, 86, 88, 132, 134, 136, 138],
            [3.0] * 8,
            ref_start=70,
            ref_end=150,
        )
        for index in range(40)
    ]
    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        [site],
        boundary_search_radius=2,
    )

    assert model["modeled_molecules"] == 40
    assert model["modeled_molecule_fraction"] == 1.0
    assert model["informative_molecule_fraction"] == 0.0
    assert model["geometry"][0]["site_opportunity_bearing_molecules"] == 0
    assert model["geometry"][0]["site_total_opportunities"] == 0


def test_iterative_tf_geometry_edge_opportunity_window_excludes_max_coordinate():
    site = _site("half-open-edge", 100, 120)
    reads = [
        _read(
            f"half-open-edge-{index}",
            "CT",
            [90, 101, 110, 115, 125, 130],
            [-4.0, 3.0, 3.0, 3.0, -4.0, -4.0],
            tfs=[IntervalCall(100, 120)],
            ref_start=70,
            ref_end=150,
        )
        for index in range(24)
    ]
    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        [site],
        boundary_search_radius=1,
    )
    start = model["geometry"][0]["edge_identifiability"]["start"]

    assert start["raw_edge_opportunities_by_strand"]["CT"] == 0
    assert start["effective_edge_opportunities_by_strand"]["CT"] == 0.0
    assert start["pooled_candidate_opportunity_projection_class_count"] == 1
    assert start["operationally_identified"] is False


def test_iterative_tf_geometry_is_invariant_to_site_permutation():
    site_a = _site("A", 100, 120)
    site_b = _site("B", 200, 220)
    positions = [85, 105, 115, 135, 185, 205, 215, 235]
    reads = [
        _read(
            f"permutation-{index}",
            "CT" if index % 2 == 0 else "GA",
            positions,
            [-4.0, 3.0, 3.0, -4.0, -4.0, 3.0, 3.0, -4.0],
            tfs=[IntervalCall(100, 120), IntervalCall(200, 220)],
            ref_start=60,
            ref_end=260,
        )
        for index in range(30)
    ]
    ordered_sites = [site_a, site_b]
    reversed_sites = [site_b, site_a]
    ordered = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        ordered_sites,
        enumerate_configurations(ordered_sites, include_nucleosome=False),
        boundary_search_radius=1,
    )
    reversed_model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        reversed_sites,
        enumerate_configurations(reversed_sites, include_nucleosome=False),
        boundary_search_radius=1,
    )

    assert ordered["configuration_names"] == reversed_model["configuration_names"]
    assert ordered["configuration_probabilities"] == pytest.approx(
        reversed_model["configuration_probabilities"], abs=1e-12
    )
    assert ordered["geometry"] == reversed_model["geometry"]
    assert ordered["model_structure_id"] == reversed_model["model_structure_id"]
    assert ordered["model_id"] == reversed_model["model_id"]


def test_iterative_tf_geometry_does_not_call_sparse_stratum_conflicting():
    site = _site("unbalanced", 100, 120)
    ct_reads = [
        _read(
            f"unbalanced-ct-{index}",
            "CT",
            [95, 100, 105, 115, 125],
            [-4.0, 3.0, 3.0, 3.0, -4.0],
            tfs=[IntervalCall(100, 120)],
            ref_start=70,
            ref_end=150,
        )
        for index in range(200)
    ]
    ga_reads = [
        _read(
            f"unbalanced-ga-{index}",
            "GA",
            [95, 99, 105, 115, 125],
            [-4.0, 3.0, 3.0, 3.0, -4.0],
            tfs=[IntervalCall(99, 120)],
            ref_start=70,
            ref_end=150,
        )
        for index in range(3)
    ]
    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        [*ct_reads, *ga_reads],
        [site],
        boundary_search_radius=1,
    )
    start = model["geometry"][0]["edge_identifiability"]["start"]

    assert start["status"] != "conflicting_strata"
    assert "GA" in start["blind_strands"]
    assert "GA" not in start["informative_strands"]


def test_iterative_tf_geometry_conflicting_strata_are_not_operational_edges():
    site = _site("conflict", 100, 120)
    ct_reads = [
        _read(
            f"conflict-ct-{index}",
            "CT",
            [95, 99, 100, 105, 115, 125],
            [-4.0, -4.0, 3.0, 3.0, 3.0, -4.0],
            tfs=[IntervalCall(100, 120)],
            ref_start=70,
            ref_end=150,
        )
        for index in range(80)
    ]
    ga_reads = [
        _read(
            f"conflict-ga-{index}",
            "GA",
            [95, 99, 100, 105, 115, 125],
            [-4.0, 3.0, -4.0, 3.0, 3.0, -4.0],
            tfs=[IntervalCall(99, 120)],
            ref_start=70,
            ref_end=150,
        )
        for index in range(80)
    ]

    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        [*ct_reads, *ga_reads],
        [site],
        boundary_search_radius=1,
    )
    start = model["geometry"][0]["edge_identifiability"]["start"]

    assert start["status"] == "conflicting_strata"
    assert start["pooled_information_qualified"] is True
    assert start["operationally_identified"] is False


def test_iterative_tf_geometry_labels_valid_pooled_complementary_resolution():
    site = _site("pooled-complementary", 100, 120)
    positions = [95, 99, 100, 105, 115, 119, 120, 125]
    reads = []
    for strand in ("CT", "GA"):
        reads.extend(
            _read(
                f"pooled-{strand}-occupied-{index}",
                strand,
                positions,
                [
                    3.0 if 100 <= position < 120 else -4.0
                    for position in positions
                ],
                tfs=[IntervalCall(100, 120)],
                ref_start=70,
                ref_end=150,
            )
            for index in range(3)
        )
        reads.append(
            _read(
                f"pooled-{strand}-accessible",
                strand,
                positions,
                [-4.0] * len(positions),
                ref_start=70,
                ref_end=150,
            )
        )

    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        [site],
        boundary_search_radius=1,
        stratum_semantics="physical_complementary",
    )
    start = model["geometry"][0]["edge_identifiability"]["start"]

    assert start["informative_strands"] == []
    assert all(
        value < start["minimum_effective_edge_opportunities"]
        for value in start["effective_edge_opportunities_by_strand"].values()
    )
    assert start["pooled_information_qualified"] is True
    assert start["status"] == "resolved_by_pooled_complementary_evidence"
    assert start["operationally_identified"] is True


def test_iterative_tf_geometry_rejects_tiny_q_margin_effect_at_high_depth():
    site = _site("tiny-effect", 100, 120)
    positions = list(range(95, 126))
    reads = [
        _read(
            f"tiny-effect-occupied-{index}",
            "CT" if index % 2 == 0 else "GA",
            positions,
            [
                0.05 if 100 <= position < 120 else -4.0
                for position in positions
            ],
            tfs=[IntervalCall(100, 120)],
            ref_start=70,
            ref_end=150,
        )
        for index in range(400)
    ]
    reads.extend(
        _read(
            f"tiny-effect-accessible-{index}",
            "CT" if index % 2 == 0 else "GA",
            positions,
            [-4.0] * len(positions),
            ref_start=70,
            ref_end=150,
        )
        for index in range(100)
    )

    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        [site],
        boundary_search_radius=1,
        stratum_semantics="physical_complementary",
    )
    start = model["geometry"][0]["edge_identifiability"]["start"]

    assert start["pooled_effective_edge_opportunities"] > 100
    assert (
        start["pooled_information_margin_to_best_non_equivalent_nats"]
        > math.log(10.0)
    )
    assert start["pooled_information_margin_per_effective_molecule_nats"] < 0.1
    assert start["pooled_information_qualified"] is False
    assert start["status"] == "undercovered"
    assert start["operationally_identified"] is False


def test_amplification_collapse_marks_validation_fingerprintability():
    reads = [
        _read(
            f"family-{index}",
            "CT",
            list(range(20)),
            [1.0] * 20,
            ref_start=0,
            ref_end=30,
        )
        for index in range(3)
    ]
    reads[0].fingerprint_positions = np.arange(10, dtype=np.int64)
    reads[1].fingerprint_positions = np.arange(10, dtype=np.int64)
    reads[2].fingerprint_positions = np.arange(3, dtype=np.int64)

    retained, diagnostics = strand_rescue_inference.collapse_amplified_reads(
        reads
    )

    assert diagnostics["duplicate_reads_collapsed"] == 1
    assert diagnostics["unfingerprintable_reads"] == 1
    assert len(retained) == 2
    assert {
        read.amplification_fingerprint_status for read in retained
    } == {"fingerprintable_representative", "unfingerprintable"}


def test_iterative_tf_geometry_detects_stratum_absorbed_by_spatial_null():
    site = _site("absorbed-conflict", 100, 120)
    positions = list(range(90, 131))
    ct_reads = [
        _read(
            f"absorbed-ct-{index}",
            "CT",
            positions,
            [3.0 if 100 <= position < 120 else -4.0 for position in positions],
            tfs=[IntervalCall(100, 120)],
            ref_start=70,
            ref_end=150,
        )
        for index in range(80)
    ]
    ga_reads = [
        _read(
            f"absorbed-ga-{index}",
            "GA",
            positions,
            [3.0 if 106 <= position < 120 else -4.0 for position in positions],
            tfs=[IntervalCall(106, 120)],
            ref_start=70,
            ref_end=150,
        )
        for index in range(80)
    ]

    model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        [*ct_reads, *ga_reads],
        [site],
        boundary_search_radius=6,
    )
    start = model["geometry"][0]["edge_identifiability"]["start"]

    assert start["status"] == "stratum_absorbed_by_residual_component"
    assert start["residual_absorbed_strands"] == ["GA"]
    assert start["raw_edge_opportunities_by_strand"]["GA"] >= 960
    assert start["effective_edge_opportunities_by_strand"]["GA"] < 1.0
    assert start["spatial_null_effective_molecule_support_by_strand"]["GA"] > 70
    assert start[
        "spatial_null_fraction_of_residual_aware_support_by_strand"
    ]["GA"] > 0.8
    assert start["minimum_spatial_null_conflict_fraction"] == 0.1
    assert start["operationally_identified"] is False


def test_spatial_null_conflict_gate_is_fractional_not_absolute_depth():
    site = _site("fractional-conflict", 100, 120)
    positions = list(range(90, 131))
    reads = [
        _read(
            f"fractional-ct-{index}",
            "CT",
            positions,
            [3.0 if 100 <= position < 120 else -4.0 for position in positions],
            tfs=[IntervalCall(100, 120)],
            ref_start=70,
            ref_end=150,
        )
        for index in range(80)
    ]
    reads.extend(
        _read(
            f"fractional-ga-{index}",
            "GA",
            positions,
            [3.0 if 106 <= position < 120 else -4.0 for position in positions],
            tfs=[IntervalCall(106, 120)],
            ref_start=70,
            ref_end=150,
        )
        for index in range(80)
    )

    default_model = (
        strand_rescue_inference.fit_iterative_tf_class_geometry_model(
            reads,
            [site],
            boundary_search_radius=6,
        )
    )
    default_start = default_model["geometry"][0]["edge_identifiability"][
        "start"
    ]
    assert default_start["status"] == "stratum_absorbed_by_residual_component"
    fraction = default_start[
        "spatial_null_fraction_of_residual_aware_support_by_strand"
    ]["GA"]
    assert fraction == pytest.approx(
        default_start["spatial_null_effective_molecule_support_by_strand"][
            "GA"
        ]
        / default_start[
            "residual_aware_effective_molecule_support_by_strand"
        ]["GA"]
    )
    assert 0.0 <= fraction <= 1.0
    assert default_start["minimum_spatial_null_conflict_fraction"] == 0.1


def test_iterative_tf_geometry_initial_tie_break_samples_exact_equivalence():
    site = _site("tie-break", 100, 120)
    reads = [
        _read(
            f"tie-break-{index}",
            "CT" if index % 2 == 0 else "GA",
            [90, 105, 115, 130],
            [-4.0, 3.0, 3.0, -4.0],
            tfs=[IntervalCall(100, 120)],
            ref_start=70,
            ref_end=150,
        )
        for index in range(20)
    ]
    selected = []
    for initial in ((98, 118), (100, 120), (102, 122)):
        model = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
            reads,
            [site],
            boundary_search_radius=2,
            initial_geometry_intervals=[initial],
            tie_break_mode="initial",
        )
        selected.append(tuple(model["geometry"][0]["selected_interval"]))

    assert selected == [(98, 118), (100, 120), (102, 122)]


def test_iterative_tf_geometry_model_id_includes_fit_provenance():
    site = _site("provenance", 100, 120)
    reads = [
        _read(
            f"provenance-{index}",
            "CT",
            [85, 105, 115, 135],
            [-4.0, 3.0, 3.0, -4.0],
            tfs=[IntervalCall(100, 120)],
            ref_start=70,
            ref_end=150,
        )
        for index in range(20)
    ]
    weak_prior = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads, [site], boundary_search_radius=1, pseudocount=0.5
    )
    strong_prior = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads, [site], boundary_search_radius=1, pseudocount=17.0
    )

    assert weak_prior["model_structure_id"] == strong_prior["model_structure_id"]
    assert weak_prior["training_data_sha256"] == strong_prior["training_data_sha256"]
    assert weak_prior["model_id"] != strong_prior["model_id"]


def test_iterative_tf_geometry_model_id_includes_edge_reporting_thresholds():
    site = _site("threshold-provenance", 100, 120)
    reads = [
        _read(
            f"threshold-provenance-{index}",
            "CT",
            [85, 100, 105, 115, 120, 135],
            [-4.0, 3.0, 3.0, 3.0, -4.0, -4.0],
            tfs=[IntervalCall(100, 120)],
            ref_start=70,
            ref_end=150,
        )
        for index in range(24)
    ]
    permissive = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        [site],
        boundary_search_radius=1,
        minimum_edge_effective_opportunities=0.0,
        minimum_edge_information_spread=0.0,
    )
    strict = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        reads,
        [site],
        boundary_search_radius=1,
        minimum_edge_effective_opportunities=1e9,
        minimum_edge_information_spread=1e9,
    )
    strict_effect_size = (
        strand_rescue_inference.fit_iterative_tf_class_geometry_model(
            reads,
            [site],
            boundary_search_radius=1,
            minimum_edge_effective_opportunities=0.0,
            minimum_edge_information_spread=0.0,
            minimum_edge_q_margin_per_effective_molecule=1e9,
        )
    )

    assert permissive["model_structure_id"] == strict["model_structure_id"]
    assert permissive["training_data_sha256"] == strict["training_data_sha256"]
    assert permissive["configuration_probabilities"] == pytest.approx(
        strict["configuration_probabilities"]
    )
    assert permissive["model_id"] != strict["model_id"]
    assert permissive["model_id"] != strict_effect_size["model_id"]
    assert permissive["geometry"][0]["edge_identifiability"]["start"][
        "operationally_identified"
    ] is True
    assert strict["geometry"][0]["edge_identifiability"]["start"][
        "operationally_identified"
    ] is False
    assert strict_effect_size["geometry"][0]["edge_identifiability"]["start"][
        "operationally_identified"
    ] is False


def test_iterative_tf_geometry_rejects_overlapping_supplied_configuration():
    sites = [_site("left", 100, 125), _site("right", 120, 140)]
    configurations = [
        strand_rescue_inference.Configuration("A", ()),
        strand_rescue_inference.Configuration("TF:left,right", (0, 1)),
    ]
    with pytest.raises(
        ValueError, match="overlapping atomic site intervals"
    ):
        strand_rescue_inference.fit_iterative_tf_class_geometry_model(
            [], sites, configurations, boundary_search_radius=1
        )


def test_iterative_tf_geometry_duplicate_selection_is_read_order_invariant():
    site = _site("dedup", 100, 120)
    positive = _read(
        "same-molecule",
        "CT",
        [85, 105, 115, 135],
        [-4.0, 3.0, 3.0, -4.0],
        tfs=[IntervalCall(100, 120)],
        ref_start=70,
        ref_end=150,
    )
    negative = _read(
        "same-molecule",
        "CT",
        [85, 105, 115, 135],
        [-4.0, -4.0, -4.0, -4.0],
        ref_start=70,
        ref_end=150,
    )
    forward = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        [positive, negative], [site], boundary_search_radius=1
    )
    reverse = strand_rescue_inference.fit_iterative_tf_class_geometry_model(
        [negative, positive], [site], boundary_search_radius=1
    )

    assert forward["eligible_molecules"] == 1
    assert forward["model_id"] == reverse["model_id"]
    assert forward["configuration_probabilities"] == pytest.approx(
        reverse["configuration_probabilities"], abs=1e-12
    )


def test_strand_rescue_q0_compares_recurrent_composite_tf_classes():
    broad = _site("broad", 90, 150, support=10)
    left = _site("left", 90, 110, support=10)
    right = _site("right", 130, 150, support=10)
    sites = [broad, left, right]
    source = [
        _read(
            f"broad-{index}",
            "FWD",
            [100, 140],
            [4.0, 4.0],
            tfs=[IntervalCall(90, 150)],
            ref_start=70,
            ref_end=170,
        )
        for index in range(10)
    ]
    source.extend(
        _read(
            f"pair-{index}",
            "FWD",
            [100, 140],
            [4.0, 4.0],
            tfs=[IntervalCall(90, 110), IntervalCall(130, 150)],
            ref_start=70,
            ref_end=170,
        )
        for index in range(10)
    )
    target = _read(
        "ambiguous-target",
        "REV",
        [100, 140],
        [4.0, 4.0],
        msps=[IntervalCall(80, 160)],
        ref_start=70,
        ref_end=170,
    )

    result = analyze_strand_rescue(
        [*source, target],
        sites,
        min_source_support=10,
        center_radius=25,
    )

    assert len(result["decisions"]) == 1
    decision = result["decisions"][0]
    model = result["tf_class_models"][decision["source_tf_class_model_id"]]
    classes = {value["configuration"]: value for value in model["classes"]}
    assert classes["TF:broad"]["anchored_molecule_support"] == 10
    assert classes["TF:left,right"]["anchored_molecule_support"] == 10
    assert decision["other_supported_tf_configuration_probability"] > 0.4
    assert decision["sr_hypothesis_probability"] < 0.6
    assert decision["selected_tf_class_id"].startswith("tfclass_")


def test_em_recovers_dominant_latent_states():
    likelihoods = np.vstack(
        [
            np.tile([8.0, 0.0, 0.0], (80, 1)),
            np.tile([0.0, 8.0, 0.0], (20, 1)),
        ]
    )
    weights, _iterations = fit_mixture_weights(likelihoods)
    np.testing.assert_allclose(weights[:2], [0.8, 0.2], atol=0.01)
    assert weights[2] < 0.01


def test_short_source_read_only_needs_to_span_the_site():
    read = _read(
        "short",
        "FWD",
        [105, 115],
        [3.0, 3.0],
        ref_start=100,
        ref_end=120,
    )

    model = fit_site_state_model([read], _site(), flank=100)

    assert model["coverage"] == 1
    assert model["weights"]["TF"] > model["weights"]["A"]


def test_center_only_reads_do_not_dilute_full_site_coverage():
    spanning = _read(
        "spanning",
        "FWD",
        [105, 115],
        [3.0, 3.0],
        ref_start=100,
        ref_end=120,
    )
    center_only = _read(
        "center-only",
        "FWD",
        [110],
        [3.0],
        ref_start=109,
        ref_end=111,
    )

    model = fit_site_state_model([spanning, center_only], _site())

    assert model["site_coverage"] == 1
    assert model["center_coverage"] == 2


def test_rescue_q4_uses_full_site_coverage_not_center_only_depth():
    source = _read(
        "source",
        "FWD",
        [105, 115],
        [3.0, 3.0],
        tfs=[IntervalCall(100, 120)],
        ref_start=90,
        ref_end=130,
    )
    center_only = [
        _read(
            f"center-only-{index}",
            "FWD",
            [110],
            [3.0],
            ref_start=109,
            ref_end=111,
        )
        for index in range(9)
    ]
    target = _read(
        "target",
        "REV",
        [105, 115],
        [0.4, 0.4],
        msps=[IntervalCall(80, 140)],
        ref_start=70,
        ref_end=150,
    )

    result = analyze_strand_rescue(
        [source, *center_only, target],
        [_site(support=1)],
        min_source_support=1,
    )

    decision = result["decisions"][0]
    evidence = decision["source_prior_evidence"][0]
    assert evidence["spanning_coverage"] == 1
    assert evidence["center_coverage"] == 10
    assert decision["source_support_reliability"] == wilson_lower_bound(1, 1)


def test_accepted_tf_calls_anchor_the_source_state_model():
    reads = [
        _read(
            f"accepted-{index}",
            "FWD",
            [105, 115],
            [-8.0, -8.0],
            tfs=[IntervalCall(100, 120, 0)],
        )
        for index in range(5)
    ]

    model = fit_site_state_model(reads, _site())

    assert model["anchored_tf_calls"] == 5
    assert model["weights"]["TF"] > 0.99


def test_rescue_direction_uses_call_fraction_not_raw_strand_depth():
    site = _site()
    site.support = {"FWD": 20, "REV": 10}
    site.local_enrichment_by_strand = {"FWD": 20.0, "REV": 20.0}
    forward = [
        _read(
            f"forward-call-{index}",
            "FWD",
            [105, 115],
            [-2.0, -2.0],
            tfs=[IntervalCall(100, 120)],
        )
        for index in range(20)
    ]
    forward.extend(
        _read(f"forward-bg-{index}", "FWD", [105, 115], [-2.0, -2.0])
        for index in range(80)
    )
    reverse = [
        _read(
            f"reverse-call-{index}",
            "REV",
            [105, 115],
            [-2.0, -2.0],
            tfs=[IntervalCall(100, 120)],
        )
        for index in range(10)
    ]
    reverse.extend(
        _read(f"reverse-bg-{index}", "REV", [105, 115], [-2.0, -2.0])
        for index in range(10)
    )
    forward_target = _read(
        "forward-target",
        "FWD",
        [105, 115],
        [0.4, 0.4],
        msps=[IntervalCall(80, 140)],
    )
    reverse_target = _read(
        "reverse-target",
        "REV",
        [105, 115],
        [0.4, 0.4],
        msps=[IntervalCall(80, 140)],
    )

    result = analyze_strand_rescue(
        [*forward, *reverse, forward_target, reverse_target],
        [site],
        min_source_support=5,
    )

    assert {decision["read"] for decision in result["decisions"]} == {
        "forward-target"
    }


def test_borderline_positive_msp_is_rescued_by_opposite_strand_prior():
    target = _read(
        "target",
        "REV",
        [105, 115],
        [0.4, 0.4],
        msps=[IntervalCall(80, 140)],
    )

    reverse_background = [
        _read(f"reverse-background-{index}", "REV", [105, 115], [-2.0, -2.0])
        for index in range(9)
    ]
    result = analyze_strand_rescue(
        [*_strong_source_reads(), *reverse_background, target],
        [_site()],
        min_source_support=10,
    )

    assert len(result["decisions"]) == 1
    decision = result["decisions"][0]
    assert decision["current"] == "A"
    assert decision["current_annotation"] == "msp"
    assert decision["site_evidence"][0]["llr"] > 0.0
    assert decision["posterior"] > 0.5
    candidate = result["strand_family_candidates"]["families"][0]
    assert candidate["candidate_class"] == "one_sided_with_interior_opportunity"
    assert candidate["reporting_only"] is True
    assert candidate["inference_enabled"] is False


def test_call_fraction_denominators_separate_spanning_from_detectable_reads():
    site = _site(support=2)
    reads = [
        _read(
            "direct-no-raw-opportunity",
            "FWD",
            [],
            [],
            tfs=[IntervalCall(100, 120)],
        ),
        _read(
            "direct-with-opportunity",
            "FWD",
            [110],
            [1.0],
            tfs=[IntervalCall(100, 120)],
        ),
        _read("uncalled-with-opportunity", "FWD", [110], [0.1]),
        _read("uncalled-without-opportunity", "FWD", [], []),
    ]

    model = fit_site_state_model(reads, site)

    assert model["site_coverage"] == 4
    assert model["opportunity_eligible_molecules"] == 2
    assert model["tf_callable_molecules"] == 3
    assert model["canonical_explicit_tf_molecules"] == 2
    assert model["canonical_explicit_tf_with_opportunities"] == 1
    assert model["direct_tf_without_site_opportunities"] == 1
    assert strand_rescue_inference.explicit_call_fraction(
        site, "FWD", model
    ) == 0.5
    assert strand_rescue_inference.opportunity_conditioned_call_fraction(
        site, "FWD", model
    ) == 0.5


def test_canonical_call_fraction_does_not_reuse_discovery_support():
    site = _site(start=90, end=130, support=20)
    spanning = _read(
        "canonical-spanning",
        "FWD",
        [100, 120],
        [1.0, 1.0],
        tfs=[IntervalCall(100, 120)],
        ref_start=80,
        ref_end=140,
    )

    model = fit_site_state_model([spanning], site)

    assert model["site_coverage"] == 1
    assert model["canonical_explicit_tf_molecules"] == 1
    assert strand_rescue_inference.explicit_call_fraction(
        site, "FWD", model
    ) == 1.0


def test_site_model_reports_boundary_and_interior_opportunities():
    read = _read(
        "segmented",
        "FWD",
        [95, 105, 125],
        [1.0, 1.0, 1.0],
        ref_start=80,
        ref_end=150,
    )

    model = fit_site_state_model([read], _site(), boundary_band=10)

    segments = model["opportunity_by_segment"]
    assert segments["left_boundary"]["interval"] == [90, 100]
    assert segments["interior"]["interval"] == [100, 120]
    assert segments["right_boundary"]["interval"] == [120, 130]
    assert {
        name: value["total_opportunities"] for name, value in segments.items()
    } == {"left_boundary": 1, "interior": 1, "right_boundary": 1}


@pytest.mark.parametrize(
    ("reverse_positions", "expected_class"),
    [
        ([], "one_sided_zero_interior_opportunity"),
        ([110], "one_sided_with_interior_opportunity"),
    ],
)
def test_one_sided_family_candidates_report_other_strand_opportunity(
    reverse_positions, expected_class
):
    site = _site(support=20)
    forward = [
        _read(
            f"forward-{index}",
            "FWD",
            [110],
            [1.0],
            tfs=[IntervalCall(100, 120)],
        )
        for index in range(20)
    ]
    reverse = [
        _read(
            f"reverse-{index}",
            "REV",
            reverse_positions,
            [0.1] * len(reverse_positions),
        )
        for index in range(20)
    ]

    result = analyze_strand_rescue(forward + reverse, [site])

    candidate = result["strand_family_candidates"]["families"][0]
    assert candidate["candidate_class"] == expected_class
    assert candidate["reporting_only"] is True
    assert candidate["inference_enabled"] is False
    assert candidate["supported_strands"] == ["FWD"]
    assert (
        candidate["by_strand"]["REV"]["opportunity_eligible_molecules"]
        == (20 if reverse_positions else 0)
    )


def test_inference_performance_callback_marks_each_neutral_stage():
    target = _read(
        "target",
        "REV",
        [105, 115],
        [0.4, 0.4],
        msps=[IntervalCall(80, 140)],
    )
    events = []

    analyze_strand_rescue(
        [*_strong_source_reads(), target],
        [_site()],
        min_source_support=10,
        performance_callback=lambda name, details: events.append((name, details)),
        performance_stage_prefix="primary_",
    )

    assert [name for name, _details in events] == [
        "primary_source_state_model_fitting",
        "primary_tf_configuration_class_model_fitting",
        "primary_msp_tf_rescue_scoring",
        "primary_tf_edge_normalization",
        "primary_nuc_edge_normalization",
        "primary_topology_resolution",
    ]
    assert events[2][1]["decisions"] == 1


def test_held_out_target_population_is_audited_and_propagated():
    training = [
        *_strong_source_reads(),
        _read("training-reverse", "REV", [110], [0.1]),
    ]
    held_out = _read(
        "held-out",
        "REV",
        [105, 115],
        [4.0, 4.0],
        msps=[IntervalCall(80, 140)],
    )

    result = analyze_strand_rescue(
        training,
        [_site()],
        target_reads=[held_out],
        evaluation_partition={"fold": 2, "folds": 5},
    )

    provenance = result["evaluation_provenance"]
    assert provenance["mode"] == "held_out_population"
    assert provenance["population_prior_target_disjoint"] is True
    assert provenance["overlapping_molecules"] == 0
    assert provenance["declared_partition"] == {"fold": 2, "folds": 5}
    assert result["decisions"][0]["evaluation_provenance"] == provenance


def test_overlapping_target_population_is_labeled_same_cohort():
    population = [*_strong_source_reads()]

    result = analyze_strand_rescue(
        population,
        [_site()],
        target_reads=[population[0]],
        evaluation_partition={"scheme": "deliberate-overlap-test"},
    )

    provenance = result["evaluation_provenance"]
    assert provenance["mode"] == "same_cohort_or_overlapping_population"
    assert provenance["population_prior_target_disjoint"] is False
    assert provenance["overlapping_molecules"] == 1


def test_target_component_without_positive_local_evidence_is_not_rescued():
    target = _read(
        "target",
        "REV",
        [105, 115],
        [-0.1, 0.0],
        msps=[IntervalCall(80, 140)],
    )

    result = analyze_strand_rescue(
        [*_strong_source_reads(), target], [_site()], min_source_support=10
    )

    assert result["decisions"] == []
    assert result["counts"]["locally_unsupported"] == 1


def test_existing_shifted_tf_overlap_blocks_duplicate_rescue():
    target = _read(
        "target",
        "REV",
        [105, 115],
        [0.4, 0.4],
        tfs=[IntervalCall(90, 105)],
        msps=[IntervalCall(80, 140)],
    )

    result = analyze_strand_rescue(
        [*_strong_source_reads(), target], [_site()], min_source_support=10
    )

    assert result["decisions"] == []
    assert result["counts"]["source_sites_already_covered_by_tf"] == 1


def test_nucleosome_is_never_an_sr_occupancy_candidate():
    target = _read(
        "target",
        "REV",
        [105, 115],
        [2.0, 2.0],
        nucs=[IntervalCall(60, 160)],
        msps=[IntervalCall(80, 140)],
    )

    result = analyze_strand_rescue(
        [*_strong_source_reads(), target], [_site()], min_source_support=10
    )

    assert result["decisions"] == []
    assert result["counts"]["source_sites_inside_nucs_ignored"] == 1


def test_raw_targets_are_revisited_after_population_collapse():
    population_target = _read(
        "target-a",
        "REV",
        [105, 115],
        [0.4, 0.4],
        msps=[IntervalCall(80, 140)],
    )
    raw_target = _read(
        "target-b",
        "REV",
        [105, 115],
        [0.4, 0.4],
        msps=[IntervalCall(80, 140)],
    )
    population = [*_strong_source_reads(), population_target]

    result = analyze_strand_rescue(
        population,
        [_site()],
        target_reads=[*population, raw_target],
        min_source_support=10,
    )

    assert {decision["read"] for decision in result["decisions"]} == {
        "target-a",
        "target-b",
    }
    assert result["site_models"]["REV"]["site1"]["coverage"] == 1


def test_duplicate_query_names_have_unique_alignment_scoped_ids():
    targets = [
        _read(
            "duplicate",
            "REV",
            [105, 115],
            [0.4, 0.4],
            msps=[IntervalCall(80, 140)],
            alignment_flag=flag,
            cigar=cigar,
        )
        for flag, cigar in ((0, "100M"), (16, "90M10S"))
    ]

    result = analyze_strand_rescue(
        [*_strong_source_reads(), *targets],
        [_site()],
        min_source_support=10,
    )

    assert len(result["decisions"]) == 2
    assert len({decision["decision_id"] for decision in result["decisions"]}) == 2


def test_byte_identical_alignment_occurrences_have_unique_ids():
    targets = [
        _read(
            "duplicate",
            "REV",
            [105, 115],
            [0.4, 0.4],
            msps=[IntervalCall(80, 140, ordinal=0)],
            alignment_flag=0,
            cigar="100M",
            record_sha256="same-record",
            alignment_occurrence=occurrence,
        )
        for occurrence in (0, 1)
    ]

    result = analyze_strand_rescue(
        [*_strong_source_reads(), *targets],
        [_site()],
        min_source_support=10,
    )

    assert len(result["decisions"]) == 2
    assert len({decision["decision_id"] for decision in result["decisions"]}) == 2
    assert {
        decision["alignment"]["occurrence"] for decision in result["decisions"]
    } == {0, 1}


def test_one_msp_gets_one_atomic_multi_tf_decision():
    sites = [_site("left", 90, 110), _site("right", 130, 150)]
    source = [
        _read(
            f"source-{index}",
            "FWD",
            [75, 95, 105, 135, 145, 165],
            [-4.0, 4.0, 4.0, 4.0, 4.0, -4.0],
            tfs=[IntervalCall(90, 110), IntervalCall(130, 150)],
            ref_start=60,
            ref_end=180,
        )
        for index in range(20)
    ]
    target = _read(
        "target",
        "REV",
        [95, 105, 135, 145],
        [0.5, 0.5, 0.5, 0.5],
        msps=[IntervalCall(70, 170)],
        ref_start=60,
        ref_end=180,
    )

    result = analyze_strand_rescue(source + [target], sites, min_source_support=10)

    assert len(result["decisions"]) == 1
    assert result["decisions"][0]["proposed_site_intervals"] == [
        [90, 110],
        [130, 150],
    ]


def test_selected_configuration_probability_includes_tf_family_ambiguity():
    sites = [_site("left", 90, 110), _site("right", 130, 150)]
    target = _read(
        "ambiguous",
        "REV",
        [100, 140],
        [0.01, 0.01],
        msps=[IntervalCall(70, 170)],
        ref_start=60,
        ref_end=180,
    )
    source_models = {
        index: {
            "coverage": 20,
            "site_coverage": 20,
            "center_coverage": 20,
            "weights": {"A": 0.5, "TF": 0.5, "N": 0.0},
        }
        for index in range(2)
    }

    decision = strand_rescue_inference._score_decision(
        target,
        target.msps[0],
        [0, 1],
        sites=sites,
        target_sites=sites,
        source="FWD",
        target="REV",
        source_models=source_models,
        source_reads=[],
        class_model_cache={},
        center_radius=10,
        class_model_pseudocount=0.5,
        strong_posterior=0.95,
        review_posterior=0.5,
        maximum_sites=8,
    )

    assert decision is not None
    assert decision["pairwise_selected_configuration_probability_vs_accessible"] > 0.5
    assert decision["best_configuration_posterior_given_tf"] < 0.5
    assert (
        decision[
            "selected_configuration_probability_vs_accessible_and_supported_tf"
        ]
        < 0.3
    )
    assert (
        decision[
            "selected_configuration_probability_vs_accessible_and_supported_tf"
        ]
        + decision["accessible_probability_within_supported_action_set"]
        + decision["other_supported_tf_configuration_probability"]
    ) == pytest.approx(1.0)
    assert decision["unresolved_action_set_probability"] == 0.0
    assert decision["q0_action_set_complete"] is True


def test_truncated_action_set_emits_zero_q0_and_names_dropped_families():
    sites = [
        _site("left", 90, 100),
        _site("middle", 110, 120),
        _site("right", 130, 140),
    ]
    target = _read(
        "capped",
        "REV",
        [95, 115, 135],
        [0.01, 0.01, 0.01],
        msps=[IntervalCall(80, 150)],
        ref_start=70,
        ref_end=160,
    )
    source_models = {
        index: {
            "coverage": 20,
            "site_coverage": 20,
            "center_coverage": 20,
            "canonical_explicit_tf_molecules": 20,
            "canonical_explicit_tf_with_opportunities": 20,
            "opportunity_eligible_molecules": 20,
            "weights": {"A": 0.5, "TF": 0.5, "N": 0.0},
        }
        for index in range(3)
    }

    decision = strand_rescue_inference._score_decision(
        target,
        target.msps[0],
        [0, 1, 2],
        sites=sites,
        target_sites=sites,
        source="FWD",
        target="REV",
        source_models=source_models,
        source_reads=[],
        class_model_cache={},
        center_radius=10,
        class_model_pseudocount=0.5,
        strong_posterior=0.95,
        review_posterior=0.5,
        maximum_sites=2,
    )

    assert decision is not None
    assert decision["templates_truncated"] is True
    assert decision["q0_action_set_complete"] is False
    assert decision["dropped_template_site_ids"] == ["right"]
    assert decision["sr_hypothesis_probability"] == 0.0
    assert decision["posterior"] == 0.0
    assert decision["baseline_hypothesis_probability"] == 0.0
    assert decision["other_supported_tf_configuration_probability"] == 0.0
    assert decision["unresolved_action_set_probability"] == 1.0
    assert decision["proposal_tier"] == "retain_current"
    assert (
        decision[
            "conditional_selected_configuration_probability_within_considered_action_set"
        ]
        > 0.0
    )


def test_shared_existing_tf_edges_are_harmonized_without_touching_nucs():
    site = _site()
    site.support = {"FWD": 10, "REV": 10}
    reverse = [
        _read(
            f"reverse-{index}",
            "REV",
            [105, 115],
            [3.0, 3.0],
            tfs=[IntervalCall(102, 122)],
        )
        for index in range(10)
    ]

    result = analyze_strand_rescue(
        [*_strong_source_reads(10), *reverse],
        [site],
        min_source_support=5,
        minimum_geometry_support=3,
    )

    updates = result["edge_refinement"]["tf"]["harmonizations"]
    reverse_updates = [
        value for value in updates
        if value["target_strand"] == "REV" and value["status"] == "edge_update"
    ]
    assert len(reverse_updates) == 10
    assert reverse_updates[0]["canonical_interval"] == [100, 120]

    reverse[0].nucs = [IntervalCall(80, 102)]
    conflicted = analyze_strand_rescue(
        [*_strong_source_reads(10), *reverse],
        [site],
        min_source_support=5,
        minimum_geometry_support=3,
    )
    statuses = {
        value["status"]
        for value in conflicted["edge_refinement"]["tf"]["harmonizations"]
        if value["read"] == "reverse-0"
    }
    assert statuses == {"topology_conflict_retained"}


def test_preexisting_overlap_is_grandfathered_during_edge_refinement():
    site = _site()
    site.support = {"FWD": 3, "REV": 3}
    read = _read(
        "overlapping",
        "REV",
        [101, 110, 121],
        [1.0, 1.0, 1.0],
        tfs=[IntervalCall(102, 122)],
        nucs=[IntervalCall(80, 105)],
    )
    source = _read(
        "source",
        "FWD",
        [101, 110, 119],
        [1.0, 1.0, 1.0],
        tfs=[IntervalCall(100, 120)],
    )

    result = analyze_shared_geometry(
        [source, read],
        [site],
        minimum_geometry_support=1,
    )

    decision = next(
        value for value in result["harmonizations"] if value["read"] == "overlapping"
    )
    assert decision["status"] == "edge_update"


def test_edge_bayes_factor_scores_only_changed_left_and_right_bases():
    read = _read(
        "edge",
        "FWD",
        [95, 105, 115, 125],
        [2.0, -9.0, -9.0, 3.0],
    )

    evidence = edge_log_bayes_factor(
        read, IntervalCall(100, 120), IntervalCall(90, 130)
    )

    assert evidence["log_bf_canonical_vs_current"] == 5.0
    assert evidence["left_edge"]["log_bf_canonical_vs_current"] == 2.0
    assert evidence["right_edge"]["log_bf_canonical_vs_current"] == 3.0
    assert evidence["changed_opportunities"] == 2


def _zero_edge_chemistry():
    return {
        "log_bf_canonical_vs_current": 0.0,
        "left_edge": {"log_bf_canonical_vs_current": 0.0},
        "right_edge": {"log_bf_canonical_vs_current": 0.0},
    }


def _predictive_edge_population_terms(current, canonical, source_calls):
    edge_terms = {}
    for edge, current_value, canonical_value, sample_values in (
        (
            "left",
            current.start,
            canonical.start,
            [call.start for call in source_calls],
        ),
        (
            "right",
            current.end,
            canonical.end,
            [call.end for call in source_calls],
        ),
    ):
        sample_mad = (
            float(
                np.median(
                    np.abs(
                        np.asarray(sample_values, dtype=np.float64)
                        - np.median(sample_values)
                    )
                )
            )
            if sample_values
            else 0.0
        )
        scale = max(2.0, 1.4826 * sample_mad)
        predictive_scale = scale * (
            math.sqrt(1.0 + 1.0 / len(sample_values))
            if sample_values
            else 1.0
        )
        edge_terms[edge] = (
            0.5
            * (
                ((current_value - np.median(sample_values)) / predictive_scale)
                ** 2
                - (
                    (canonical_value - np.median(sample_values))
                    / predictive_scale
                )
                ** 2
            )
            if sample_values and current_value != canonical_value
            else 0.0
        )

    sample_matrix = np.asarray(
        [[call.start, call.end] for call in source_calls], dtype=np.float64
    ).reshape((-1, 2))
    start_scale = max(
        2.0,
        1.4826
        * (
            float(
                np.median(
                    np.abs(sample_matrix[:, 0] - np.median(sample_matrix[:, 0]))
                )
            )
            if source_calls
            else 0.0
        ),
    )
    end_scale = max(
        2.0,
        1.4826
        * (
            float(
                np.median(
                    np.abs(sample_matrix[:, 1] - np.median(sample_matrix[:, 1]))
                )
            )
            if source_calls
            else 0.0
        ),
    )
    correlation = 0.0
    if sample_matrix.shape[0] >= 3:
        centered_start = sample_matrix[:, 0] - np.mean(sample_matrix[:, 0])
        centered_end = sample_matrix[:, 1] - np.mean(sample_matrix[:, 1])
        denominator = math.sqrt(
            float(centered_start @ centered_start)
            * float(centered_end @ centered_end)
        )
        if denominator > 0.0:
            correlation = max(
                -0.9,
                min(0.9, float((centered_start @ centered_end) / denominator)),
            )
    covariance = np.asarray(
        [
            [start_scale**2, correlation * start_scale * end_scale],
            [correlation * start_scale * end_scale, end_scale**2],
        ]
    )
    multiplier = 1.0 + 1.0 / len(source_calls) if source_calls else 1.0
    predictive_covariance = covariance * multiplier
    inverse = np.linalg.inv(predictive_covariance)
    baseline_center = np.asarray([current.start, current.end], dtype=np.float64)
    canonical_center = np.asarray(
        [canonical.start, canonical.end], dtype=np.float64
    )
    joint = 0.0
    if source_calls:
        population_center = np.median(sample_matrix, axis=0)
        baseline_residual = baseline_center - population_center
        canonical_residual = canonical_center - population_center
        joint = 0.5 * float(
            baseline_residual @ inverse @ baseline_residual
            - canonical_residual @ inverse @ canonical_residual
        )
    return edge_terms, covariance, predictive_covariance, correlation, joint


@pytest.mark.parametrize(
    "current,canonical,source_calls",
    [
        (IntervalCall(100, 120), IntervalCall(102, 122), []),
        (
            IntervalCall(100, 120),
            IntervalCall(102, 122),
            [IntervalCall(101, 121)],
        ),
        (
            IntervalCall(100, 120),
            IntervalCall(100, 124),
            [IntervalCall(100, 122), IntervalCall(102, 124)],
        ),
        (
            IntervalCall(47_515_001, 47_515_020),
            IntervalCall(47_515_004, 47_515_024),
            [
                IntervalCall(47_515_000 + index % 17, 47_515_020 + index % 23)
                for index in range(1000)
            ],
        ),
    ],
)
def test_edge_population_prepared_statistics_match_predictive_reference(
    current, canonical, source_calls
):
    chemistry = {
        "log_bf_canonical_vs_current": 1.25,
        "left_edge": {"log_bf_canonical_vs_current": 0.50},
        "right_edge": {"log_bf_canonical_vs_current": 0.75},
    }
    statistics = _prepare_edge_population_statistics(source_calls)
    optimized = edge_hypothesis_evidence(
        current,
        canonical,
        (),
        chemistry,
        population_statistics=statistics,
    )
    wrapper = edge_hypothesis_evidence(
        current, canonical, source_calls, chemistry
    )
    (
        edge_terms,
        covariance,
        predictive_covariance,
        correlation,
        joint,
    ) = _predictive_edge_population_terms(current, canonical, source_calls)

    assert wrapper == optimized
    assert optimized["left"]["population_log_bf"] == pytest.approx(
        edge_terms["left"], abs=1e-9
    )
    assert optimized["right"]["population_log_bf"] == pytest.approx(
        edge_terms["right"], abs=1e-9
    )
    assert optimized["joint_population_log_bf"] == pytest.approx(
        joint, abs=1e-9
    )
    assert np.asarray(optimized["population_covariance"]) == pytest.approx(
        covariance
    )
    assert np.asarray(
        optimized["population_predictive_covariance"]
    ) == pytest.approx(predictive_covariance)
    assert optimized["population_correlation"] == pytest.approx(correlation)


def test_edge_population_prior_is_stable_to_source_depth_replication():
    current = IntervalCall(100, 120)
    canonical = IntervalCall(102, 122)

    evidence = {
        depth: edge_hypothesis_evidence(
            current,
            canonical,
            [IntervalCall(102, 122) for _index in range(depth)],
            _zero_edge_chemistry(),
        )
        for depth in (10, 100, 1000)
    }

    assert evidence[10]["joint_population_log_bf"] == pytest.approx(
        1.0 / 1.1
    )
    assert abs(
        evidence[100]["joint_population_log_bf"]
        - evidence[10]["joint_population_log_bf"]
    ) < 0.1
    assert abs(
        evidence[1000]["joint_population_log_bf"]
        - evidence[100]["joint_population_log_bf"]
    ) < 0.01
    assert evidence[1000]["joint_population_log_bf"] < 1.0


def test_edge_hypothesis_population_can_favor_either_baseline_or_sr():
    current = IntervalCall(100, 120)
    canonical = IntervalCall(102, 122)

    baseline_supported = edge_hypothesis_evidence(
        current,
        canonical,
        [IntervalCall(100, 120) for _index in range(3)],
        _zero_edge_chemistry(),
    )
    sr_supported = edge_hypothesis_evidence(
        current,
        canonical,
        [IntervalCall(102, 122) for _index in range(3)],
        _zero_edge_chemistry(),
    )

    assert baseline_supported["alternative_probability"] < 0.5
    assert baseline_supported["left"]["alternative_probability"] < 0.5
    assert baseline_supported["right"]["alternative_probability"] < 0.5
    assert sr_supported["alternative_probability"] > 0.5
    assert sr_supported["left"]["alternative_probability"] > 0.5
    assert sr_supported["right"]["alternative_probability"] > 0.5


def test_edge_hypothesis_joint_q_uses_regularized_bivariate_geometry():
    chemistry = {
        "log_bf_canonical_vs_current": 1.25,
        "left_edge": {"log_bf_canonical_vs_current": 0.50},
        "right_edge": {"log_bf_canonical_vs_current": 0.75},
    }
    evidence = edge_hypothesis_evidence(
        IntervalCall(98, 118),
        IntervalCall(102, 122),
        [IntervalCall(100, 120), IntervalCall(102, 122), IntervalCall(104, 124)],
        chemistry,
    )

    covariance = np.asarray(evidence["population_covariance"])
    assert evidence["population_correlation"] == 0.9
    assert np.linalg.det(covariance) > 0.0
    assert evidence["joint_chemistry_log_bf"] == 1.25
    probability = evidence["alternative_probability"]
    assert math.log(probability / (1.0 - probability)) == pytest.approx(
        evidence["joint_log_bf"]
    )
    assert evidence["joint_log_bf"] == pytest.approx(
        evidence["joint_population_log_bf"] + 1.25
    )


def test_unchanged_edge_has_unit_confidence_and_no_marginal_evidence():
    evidence = edge_hypothesis_evidence(
        IntervalCall(100, 118),
        IntervalCall(100, 122),
        [IntervalCall(100, 122) for _index in range(3)],
        _zero_edge_chemistry(),
    )

    assert evidence["left"]["changed"] is False
    assert evidence["left"]["alternative_probability"] == 1.0
    assert evidence["left"]["combined_log_bf"] == 0.0
    assert evidence["right"]["alternative_probability"] > 0.5


def test_empty_opposite_population_is_finite_and_chemistry_only():
    evidence = edge_hypothesis_evidence(
        IntervalCall(100, 120),
        IntervalCall(102, 122),
        [],
        _zero_edge_chemistry(),
    )

    assert evidence["opposite_strand_molecules"] == 0
    assert evidence["joint_population_log_bf"] == 0.0
    assert evidence["alternative_probability"] == 0.5
    assert evidence["left"]["opposite_strand_mad"] == 0.0


def test_geometry_null_retains_same_center_edge_outlier():
    site = _site(start=100, end=120)
    site.support = {"FWD": 5, "REV": 5}
    target = _read(
        "outlier",
        "REV",
        [90, 110, 130],
        [1.0, 1.0, 1.0],
        tfs=[IntervalCall(80, 140)],
    )
    source = _read(
        "source",
        "FWD",
        [110],
        [1.0],
        tfs=[IntervalCall(100, 120)],
    )

    result = analyze_shared_geometry(
        [source, target], [site], minimum_geometry_support=3
    )

    decision = next(
        value
        for value in result["harmonizations"]
        if value["read"] == "outlier"
    )
    assert decision["status"] == "unassigned_retained"
    assert decision["assignment_beats_null"] is False


def test_edge_q0_includes_geometry_family_assignment_probability():
    site = _site(start=100, end=120)
    site.support = {"FWD": 3, "REV": 3}
    source = [
        _read(
            f"source-{index}",
            "FWD",
            [105, 115],
            [1.0, 1.0],
            tfs=[IntervalCall(100, 120)],
        )
        for index in range(3)
    ]
    target = _read(
        "target",
        "REV",
        [98, 122],
        [1.0, 1.0],
        tfs=[IntervalCall(96, 124)],
    )

    result = analyze_shared_geometry(
        [*source, target], [site], minimum_geometry_support=3
    )

    decision = next(
        value for value in result["harmonizations"] if value["read"] == "target"
    )
    hypothesis = decision["edge_hypothesis"]
    conditional = hypothesis[
        "conditional_alternative_probability_given_geometry_family"
    ]
    assignment = decision["assignment_probability"]
    assert decision["status"] == "edge_update"
    assert 0.0 < assignment < 1.0
    assert hypothesis["alternative_probability"] == pytest.approx(
        assignment * conditional
    )
    assert hypothesis["alternative_probability"] < conditional
    assert (
        hypothesis["alternative_probability"]
        + hypothesis["baseline_probability"]
        + hypothesis["unresolved_geometry_probability"]
    ) == pytest.approx(1.0)


def test_edge_update_requires_actual_assigned_opposite_strand_samples():
    site = _site()
    site.support = {"FWD": 3, "REV": 3}
    reads = [
        _read(
            strand.lower(),
            strand,
            [105, 115],
            [1.0, 1.0],
            tfs=[IntervalCall(102, 122)],
        )
        for strand in ("FWD", "REV")
    ]

    result = analyze_shared_geometry(
        reads,
        [site],
        minimum_geometry_support=3,
    )

    assert result["counts"]["edge_updates"] == 0
    assert result["counts"]["insufficient_opposite_geometry_samples"] == 2
    assert {
        decision["status"] for decision in result["harmonizations"]
    } == {"insufficient_opposite_geometry_samples"}


def test_edge_population_diagnostics_count_preparation_not_depth_rescans():
    site = _site()
    site.support = {"FWD": 1, "REV": 2}
    reads = [
        _read(
            "source",
            "FWD",
            [105, 115],
            [1.0, 1.0],
            tfs=[IntervalCall(100, 120)],
        ),
        *[
            _read(
                f"target-{index}",
                "REV",
                [105, 115],
                [1.0, 1.0],
                tfs=[IntervalCall(102, 122)],
            )
            for index in range(2)
        ],
    ]

    result = analyze_shared_geometry(
        reads,
        [site],
        minimum_geometry_support=1,
    )

    counts = result["counts"]
    assert counts["edge_population_models_prepared"] == 2
    assert counts["edge_population_source_samples_prepared"] == 3
    assert counts["edge_population_sufficient_stat_evaluations"] == 3
    assert counts["edge_population_naive_sample_evaluations_avoided"] == 4


def test_nuc_edge_refinement_has_no_220_bp_length_ceiling():
    site = SiteTemplate(
        site_id="nuc_site1",
        start=100,
        end=400,
        center=250,
        support={"FWD": 5, "REV": 5},
        start_mad=2.0,
        end_mad=2.0,
        local_enrichment=4.0,
        local_enrichment_by_strand={"FWD": 4.0, "REV": 4.0},
        geometry_reliability=0.8,
        call_type="nuc",
    )
    reads = [
        _read(
            strand.lower(),
            strand,
            [101, 399],
            [2.0, 2.0],
            nucs=[IntervalCall(102, 398)],
            ref_start=50,
            ref_end=450,
        )
        for strand in ("FWD", "REV")
    ]

    result = analyze_shared_geometry(
        reads,
        [site],
        call_type="nuc",
        center_radius=25,
        minimum_geometry_support=1,
    )

    assert result["counts"]["edge_updates"] == 2
    assert all(
        value["canonical_interval"] == [100, 400]
        for value in result["harmonizations"]
    )


def test_topology_only_annotation_blocks_edge_expansion():
    site = SiteTemplate(
        site_id="nuc_site1",
        start=100,
        end=130,
        center=115,
        support={"FWD": 3, "REV": 3},
        start_mad=5.0,
        end_mad=5.0,
        local_enrichment=4.0,
        local_enrichment_by_strand={"FWD": 4.0, "REV": 4.0},
        geometry_reliability=0.8,
        call_type="nuc",
    )
    current = IntervalCall(105, 120)
    softclipped_neighbor = IntervalCall(
        125,
        150,
        mapped_fraction=0.1,
        geometry_eligible=False,
    )
    reads = [
        _read(
            strand.lower(),
            strand,
            [101, 119, 129],
            [2.0, 2.0, 2.0],
            nucs=[current, softclipped_neighbor],
            ref_start=50,
            ref_end=180,
        )
        for strand in ("FWD", "REV")
    ]

    result = analyze_shared_geometry(
        reads,
        [site],
        call_type="nuc",
        center_radius=25,
        minimum_geometry_support=3,
    )

    assert {
        decision["status"] for decision in result["harmonizations"]
    } == {"topology_conflict_retained"}
    assert result["counts"]["geometry_eligible_calls"] == 2
    assert result["counts"]["topology_only_obstacles"] == 2
    assert result["topology_only_obstacles_by_strand"] == {"FWD": 1, "REV": 1}


def test_softclipped_ma_is_loaded_as_topology_only_and_cannot_teach_geometry(
    tmp_path,
):
    bam_path = tmp_path / "softclipped.bam"
    header = pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": "chr1", "LN": 1000}],
        }
    )
    with pysam.AlignmentFile(bam_path, "wb", header=header) as bam:
        read = pysam.AlignedSegment(header)
        read.query_name = "softclipped"
        read.query_sequence = "C" * 50 + "Y" + "C" * 149
        read.flag = 0
        read.reference_id = 0
        read.reference_start = 100
        read.mapping_quality = 60
        read.cigarstring = "100S100M"
        read.query_qualities = pysam.qualitystring_to_array("I" * 200)
        read.set_tag("st", "CT", value_type="Z")
        read.set_tag("MA", "200;nuc.Q:1-110", value_type="Z")
        read.set_tag("AQ", array("B", [200]))
        bam.write(read)
    pysam.index(str(bam_path))

    reads = load_region_evidence(
        str(bam_path),
        "chr1",
        90,
        220,
        strand_mode="daf",
        mode="daf",
        context_size=3,
        prob_threshold=None,
        llr_hit=np.zeros(N_CTX),
        llr_miss=np.zeros(N_CTX),
        min_mapq=20,
    )

    assert len(reads) == 1
    assert len(reads[0].nucs) == 1
    call = reads[0].nucs[0]
    assert (call.start, call.end) == (100, 110)
    assert call.mapped_fraction == 10 / 110
    assert call.geometry_eligible is False
    sites = discover_edge_sites(
        reads,
        90,
        220,
        call_type="nuc",
        min_support=1,
        minimum_geometry_support=1,
        center_radius=25,
        edge_assignment_radius=48,
        max_boundary_mad=24.0,
        min_local_enrichment=0.0,
        local_background_radius=250,
        max_auto_sites=12,
        boundary_reliability_scale=24.0,
        source_boundary_margin=0,
    )
    assert sites == []


def test_load_region_evidence_can_select_post_inference_tf_and_nuc_layers(tmp_path):
    bam_path = tmp_path / "selected_tf_layer.bam"
    header = pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": "chr1", "LN": 1000}],
            "CO": ["MA-TYPES:v1:nuc,nuc_sr,tf,tf_sr"],
        }
    )
    with pysam.AlignmentFile(bam_path, "wb", header=header) as bam:
        read = pysam.AlignedSegment(header)
        read.query_name = "selected-layer"
        read.query_sequence = "C" * 50 + "Y" + "C" * 149
        read.flag = 0
        read.reference_id = 0
        read.reference_start = 100
        read.mapping_quality = 60
        read.cigarstring = "200M"
        read.query_qualities = pysam.qualitystring_to_array("I" * 200)
        read.set_tag("st", "CT", value_type="Z")
        read.set_tag(
            "MA",
            "200;nuc.Q:1-20;nuc_sr.QQQ:81-40;tf.Q:21-10;tf_sr.QQQ:41-12",
            value_type="Z",
        )
        read.set_tag(
            "AQ",
            array("B", [190, 191, 192, 193, 200, 210, 220, 230]),
        )
        bam.write(read)
    pysam.index(str(bam_path))

    common = {
        "strand_mode": "daf",
        "mode": "daf",
        "context_size": 3,
        "prob_threshold": None,
        "llr_hit": np.zeros(N_CTX),
        "llr_miss": np.zeros(N_CTX),
        "min_mapq": 20,
    }
    ordinary = load_region_evidence(
        str(bam_path), "chr1", 90, 320, **common
    )
    normalized = load_region_evidence(
        str(bam_path),
        "chr1",
        90,
        320,
        tf_layer="tf_sr",
        nuc_layer="nuc_sr",
        **common,
    )

    assert [(call.start, call.end) for call in ordinary[0].tfs] == [(120, 130)]
    assert [(call.start, call.end) for call in normalized[0].tfs] == [(140, 152)]
    assert [(call.start, call.end) for call in ordinary[0].nucs] == [(100, 120)]
    assert [(call.start, call.end) for call in normalized[0].nucs] == [(180, 220)]
    assert normalized[0].molecular_tfs == ((40, 12),)
    assert normalized[0].molecular_nucs == ((80, 40),)

    with pytest.raises(ValueError, match="not declared"):
        load_region_evidence(
            str(bam_path),
            "chr1",
            90,
            320,
            tf_layer="tf_missing",
            **common,
        )
    with pytest.raises(ValueError, match="not declared"):
        load_region_evidence(
            str(bam_path),
            "chr1",
            90,
            320,
            nuc_layer="nuc_missing",
            **common,
        )


def test_load_region_evidence_max_reads_stops_after_same_bam_order_prefix(
    tmp_path,
):
    bam_path = tmp_path / "deep_targeted.bam"
    header = pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": "chr1", "LN": 1000}],
        }
    )
    with pysam.AlignmentFile(bam_path, "wb", header=header) as bam:
        for index in range(6):
            read = pysam.AlignedSegment(header)
            read.query_name = f"read-{index}"
            read.query_sequence = "C" * 50 + "Y" + "C" * 149
            read.flag = 0
            read.reference_id = 0
            read.reference_start = 100 + index
            read.mapping_quality = 60
            read.cigarstring = "200M"
            read.query_qualities = pysam.qualitystring_to_array("I" * 200)
            read.set_tag("st", "CT", value_type="Z")
            bam.write(read)
    pysam.index(str(bam_path))
    common = {
        "strand_mode": "daf",
        "mode": "daf",
        "context_size": 3,
        "prob_threshold": None,
        "llr_hit": np.zeros(N_CTX),
        "llr_miss": np.zeros(N_CTX),
        "min_mapq": 20,
    }
    full = load_region_evidence(
        str(bam_path), "chr1", 90, 400, **common
    )
    diagnostics = {}
    capped = load_region_evidence(
        str(bam_path),
        "chr1",
        90,
        400,
        max_reads=2,
        load_diagnostics=diagnostics,
        **common,
    )
    skipped_diagnostics = {}
    skipped = load_region_evidence(
        str(bam_path),
        "chr1",
        90,
        400,
        load_reads=False,
        load_diagnostics=skipped_diagnostics,
        **common,
    )

    assert [read.name for read in capped] == [read.name for read in full[:2]]
    assert diagnostics["fetch_record_count"] == 2
    assert diagnostics["truncated_at_max_reads"] is True
    assert diagnostics["regional_fetch_skipped"] is False
    assert skipped == []
    assert skipped_diagnostics["fetch_record_count"] == 0
    assert skipped_diagnostics["regional_fetch_skipped"] is True


def test_load_region_evidence_can_prefilter_msp_overlap_without_changing_calls(
    tmp_path,
):
    bam_path = tmp_path / "msp_prefilter.bam"
    header = pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": "chr1", "LN": 1000}],
        }
    )
    with pysam.AlignmentFile(bam_path, "wb", header=header) as bam:
        for name, ma_tag in (
            ("inside", "200;msp.Q:41-20;tf.Q:46-10"),
            ("outside", "200;msp.Q:151-20;tf.Q:156-10"),
        ):
            read = pysam.AlignedSegment(header)
            read.query_name = name
            read.query_sequence = "C" * 50 + "Y" + "C" * 149
            read.flag = 0
            read.reference_id = 0
            read.reference_start = 100
            read.mapping_quality = 60
            read.cigarstring = "200M"
            read.query_qualities = pysam.qualitystring_to_array("I" * 200)
            read.set_tag("st", "CT", value_type="Z")
            read.set_tag("MA", ma_tag, value_type="Z")
            read.set_tag("AQ", array("B", [200, 200]))
            bam.write(read)
    pysam.index(str(bam_path))
    common = {
        "strand_mode": "daf",
        "mode": "daf",
        "context_size": 3,
        "prob_threshold": None,
        "llr_hit": np.zeros(N_CTX),
        "llr_miss": np.zeros(N_CTX),
        "min_mapq": 20,
    }
    full = load_region_evidence(
        str(bam_path), "chr1", 90, 320, **common
    )
    diagnostics = {}
    selected = load_region_evidence(
        str(bam_path),
        "chr1",
        90,
        320,
        required_annotation_overlap=("msp", 135, 165),
        load_diagnostics=diagnostics,
        **common,
    )

    assert [read.name for read in full] == ["inside", "outside"]
    assert [read.name for read in selected] == ["inside"]
    assert [(call.start, call.end) for call in selected[0].msps] == [(140, 160)]
    assert [(call.start, call.end) for call in selected[0].tfs] == [(145, 155)]
    assert diagnostics["fetch_record_count"] == 2
    assert diagnostics["annotation_overlap_excluded_count"] == 1
    assert diagnostics["required_annotation_overlap"] == ["msp", 135, 165]


def test_load_region_evidence_read_name_allowlist_skips_before_evidence_parsing(
    tmp_path,
):
    bam_path = tmp_path / "name_prefilter.bam"
    header = pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": "chr1", "LN": 1000}],
        }
    )
    with pysam.AlignmentFile(bam_path, "wb", header=header) as bam:
        for index in range(5):
            read = pysam.AlignedSegment(header)
            read.query_name = f"read-{index}"
            read.query_sequence = "C" * 50 + "Y" + "C" * 149
            read.flag = 0
            read.reference_id = 0
            read.reference_start = 100
            read.mapping_quality = 60
            read.cigarstring = "200M"
            read.query_qualities = pysam.qualitystring_to_array("I" * 200)
            read.set_tag("st", "CT", value_type="Z")
            bam.write(read)
    pysam.index(str(bam_path))
    diagnostics = {}
    selected = load_region_evidence(
        str(bam_path),
        "chr1",
        90,
        320,
        strand_mode="daf",
        mode="daf",
        context_size=3,
        prob_threshold=None,
        llr_hit=np.zeros(N_CTX),
        llr_miss=np.zeros(N_CTX),
        min_mapq=20,
        required_read_names=("read-1", "read-3"),
        load_diagnostics=diagnostics,
    )

    assert [read.name for read in selected] == ["read-1", "read-3"]
    assert diagnostics["required_read_name_count"] == 2
    assert diagnostics["read_name_excluded_count"] == 3
    assert diagnostics["eligible_read_count"] == 2

    with pytest.raises(ValueError, match="sequence"):
        load_region_evidence(
            str(bam_path),
            "chr1",
            90,
            320,
            strand_mode="daf",
            mode="daf",
            context_size=3,
            prob_threshold=None,
            llr_hit=np.zeros(N_CTX),
            llr_miss=np.zeros(N_CTX),
            min_mapq=20,
            required_read_names="read-1",
        )


def test_nuc_geometry_discovery_excludes_alignment_edge_truncations():
    reads = []
    for strand in ("FWD", "REV"):
        reads.extend(
            _read(
                f"complete-{strand}-{index}",
                strand,
                [125, 275],
                [2.0, 2.0],
                nucs=[IntervalCall(100, 300)],
                ref_start=50,
                ref_end=350,
            )
            for index in range(3)
        )
        reads.extend(
            _read(
                f"clipped-{strand}-{index}",
                strand,
                [125, 275],
                [2.0, 2.0],
                nucs=[IntervalCall(100, 300)],
                ref_start=95,
                ref_end=305,
            )
            for index in range(10)
        )

    sites = discover_edge_sites(
        reads,
        80,
        320,
        call_type="nuc",
        min_support=3,
        minimum_geometry_support=3,
        center_radius=25,
        edge_assignment_radius=48,
        max_boundary_mad=24.0,
        min_local_enrichment=0.0,
        local_background_radius=250,
        max_auto_sites=12,
        boundary_reliability_scale=24.0,
        source_boundary_margin=20,
    )

    assert len(sites) == 1
    assert sites[0].support == {"FWD": 3, "REV": 3}


def test_edge_expansion_yields_to_nonoverlapping_baseline_tf_rescue():
    rescue = {
        "library_id": "/cohort.bam",
        "read": "target",
        "alignment": {
            "reference_start": 50,
            "flag": 0,
            "cigar": "100M",
            "record_sha256": "abc",
        },
        "proposed_site_intervals": [[100, 120]],
    }
    edge = {
        "status": "edge_update",
        "library_id": "/cohort.bam",
        "read": "target",
        "alignment": dict(rescue["alignment"]),
        "current_interval": [120, 200],
        "canonical_interval": [110, 200],
        "target_edge_evidence": {"changed_opportunities": 4},
        "molecule_probability": 0.7,
        "extreme_edge_shift": False,
    }
    edge_result = {"harmonizations": [edge], "counts": {"edge_updates": 1}}

    resolve_rescue_edge_topology([rescue], edge_result)

    assert edge["status"] == "rescue_topology_conflict_retained"
    assert edge_result["counts"]["edge_updates"] == 0
    assert edge_result["counts"]["rescue_topology_conflicts_retained"] == 1


def test_nuc_edge_population_q_uses_reads_spanning_the_family():
    site = SiteTemplate(
        site_id="nuc_site1",
        start=100,
        end=120,
        center=110,
        support={"FWD": 1, "REV": 1},
        start_mad=2.0,
        end_mad=2.0,
        local_enrichment=4.0,
        local_enrichment_by_strand={"FWD": 4.0, "REV": 4.0},
        geometry_reliability=0.8,
        call_type="nuc",
    )
    source = _read(
        "source",
        "FWD",
        [105, 115],
        [2.0, 2.0],
        nucs=[IntervalCall(101, 119)],
        ref_start=90,
        ref_end=130,
    )
    center_only = [
        _read(
            f"short-{index}",
            "FWD",
            [110],
            [2.0],
            ref_start=109,
            ref_end=111,
        )
        for index in range(9)
    ]
    target = _read(
        "target",
        "REV",
        [105, 115],
        [2.0, 2.0],
        nucs=[IntervalCall(101, 119)],
        ref_start=90,
        ref_end=130,
    )

    result = analyze_shared_geometry(
        [source, *center_only, target],
        [site],
        call_type="nuc",
        center_radius=25,
        minimum_geometry_support=1,
    )
    target_decision = next(
        decision
        for decision in result["harmonizations"]
        if decision["read"] == "target"
    )

    assert target_decision["population_probability"] == 0.75
    assert target_decision["source_evidence"]["spanning_coverage"] == 1
    assert target_decision["source_evidence"]["center_coverage"] == 10


def test_boundary_variant_consolidation_is_complete_linkage():
    first = _site("first", 100, 115)
    middle = _site("middle", 103, 118)
    last = _site("last", 106, 121)

    groups = strand_rescue_inference.consolidate_tf_boundary_variant_sites(
        [last, first, middle],
        maximum_boundary_delta=4,
        maximum_center_delta=4.0,
        minimum_shorter_overlap_fraction=0.75,
    )

    assert [[site.site_id for site in group] for group in groups] == [
        ["first", "middle"],
        ["last"],
    ]


def test_boundary_variant_consolidation_keeps_broad_composite_candidate_separate():
    narrow = _site("narrow", 100, 115)
    modest_variant = _site("modest", 98, 116)
    broad = _site("broad", 96, 123)

    groups = strand_rescue_inference.consolidate_tf_boundary_variant_sites(
        [broad, modest_variant, narrow],
        maximum_boundary_delta=8,
        maximum_width_delta=8,
        maximum_center_delta=6.0,
        minimum_shorter_overlap_fraction=0.75,
    )

    assert [[site.site_id for site in group] for group in groups] == [
        ["broad"],
        ["modest", "narrow"],
    ]


def test_boundary_marginalized_family_keeps_nested_widths_in_one_family():
    positions = np.arange(90, 141, 5, dtype=np.int64)
    reads = []
    for index in range(24):
        steps = np.full(positions.size, -1.5, dtype=np.float64)
        steps[(positions >= 100) & (positions < 120)] = 2.0
        reads.append(_read(f"protected-{index}", "CT", positions, steps))
    for index in range(12):
        reads.append(
            _read(
                f"accessible-{index}",
                "GA",
                positions,
                np.full(positions.size, -1.5, dtype=np.float64),
            )
        )
    snapshots = [
        (
            read.positions.copy(),
            read.steps.copy(),
            read.hits.copy(),
            read.contexts.copy(),
            [(call.start, call.end) for call in read.tfs],
            [(call.start, call.end) for call in read.nucs],
            [(call.start, call.end) for call in read.msps],
        )
        for read in reads
    ]

    model = strand_rescue_inference.fit_boundary_marginalized_tf_family_model(
        reads,
        "shared_family",
        [(100, 120), (101, 118)],
        boundary_search_radius=2,
        pseudocount=0.5,
        analysis_envelope=(90, 140),
        minimum_molecule_opportunities=3,
    )

    assert model["family_id"] == "shared_family"
    assert model["seed_intervals"] == [[100, 120], [101, 118]]
    assert model["candidate_interval_count"] > model[
        "opportunity_projection_class_count"
    ]
    assert 0.0 < model["family_probability"] < 1.0
    assert model["converged"] is True
    assert math.isclose(
        sum(
            record["conditional_probability_given_family"]
            for record in model["geometry_classes"]
        ),
        1.0,
        abs_tol=1e-10,
    )
    assert model["family_interpretation"].startswith(
        "one_locus_family_with_marginalized_boundary_distribution"
    )
    training_score = (
        strand_rescue_inference.score_boundary_marginalized_tf_family_model(
            reads, model
        )
    )
    assert training_score["family_effective_molecule_support"] == pytest.approx(
        model["family_effective_molecule_support"], abs=1e-8
    )
    for read, (
        positions_before,
        steps_before,
        hits_before,
        contexts_before,
        calls_before,
        nucs_before,
        msps_before,
    ) in zip(
        reads, snapshots
    ):
        assert np.array_equal(read.positions, positions_before)
        assert np.array_equal(read.steps, steps_before)
        assert np.array_equal(read.hits, hits_before)
        assert np.array_equal(read.contexts, contexts_before)
        assert [(call.start, call.end) for call in read.tfs] == calls_before
        assert [(call.start, call.end) for call in read.nucs] == nucs_before
        assert [(call.start, call.end) for call in read.msps] == msps_before


def test_boundary_family_prefix_backend_matches_reference_discrete_outputs():
    positions = np.arange(80, 151, 3, dtype=np.int64)
    reads = []
    for index in range(30):
        steps = np.full(positions.size, -1.15, dtype=np.float64)
        left = 99 + index % 3
        right = 119 + (index // 3) % 3
        steps[(positions >= left) & (positions < right)] = 1.85
        reads.append(
            _read(
                f"protected-{index}",
                "CT" if index % 2 == 0 else "GA",
                positions,
                steps,
            )
        )
    for index in range(10):
        reads.append(
            _read(
                f"accessible-{index}",
                "CT" if index % 2 == 0 else "GA",
                positions,
                np.full(positions.size, -1.15, dtype=np.float64),
            )
        )
    kwargs = {
        "boundary_search_radius": 4,
        "pseudocount": 0.5,
        "analysis_envelope": (85, 145),
        "minimum_molecule_opportunities": 3,
    }
    reference = strand_rescue_inference.fit_boundary_marginalized_tf_family_model(
        reads,
        "shared_family",
        [(99, 119), (101, 121)],
        evidence_summation_mode="slice_sum",
        **kwargs,
    )
    optimized = strand_rescue_inference.fit_boundary_marginalized_tf_family_model(
        reads,
        "shared_family",
        [(99, 119), (101, 121)],
        evidence_summation_mode="prefix",
        **kwargs,
    )

    for key in (
        "candidate_intervals",
        "opportunity_projection_class_count",
        "canonical_interval",
        "canonical_projection_class_index",
        "boundary_credible_envelope_95",
        "converged",
        "convergence_reason",
        "iterations",
        "model_structure_id",
        "training_cohort_evidence_sha256",
    ):
        assert optimized[key] == reference[key]
    assert optimized["component_probabilities"] == pytest.approx(
        reference["component_probabilities"], abs=1e-11
    )
    assert optimized["family_probability"] == pytest.approx(
        reference["family_probability"], abs=1e-11
    )

    reference_score = (
        strand_rescue_inference.score_boundary_marginalized_tf_family_model(
            reads, reference, include_molecule_records=True
        )
    )
    optimized_score = (
        strand_rescue_inference.score_boundary_marginalized_tf_family_model(
            reads, optimized, include_molecule_records=True
        )
    )
    for key in (
        "eligible_molecules",
        "family_posterior_over_half_molecules",
        "family_posterior_over_nine_tenths_molecules",
        "standardized_family_posterior_over_half_molecules",
        "standardized_family_posterior_over_nine_tenths_molecules",
        "family_vs_null_log_bayes_factor_nonnegative_molecules",
        "family_vs_null_log_bayes_factor_at_least_log_10_molecules",
    ):
        assert optimized_score[key] == reference_score[key]
    reference_records = {
        tuple(record["molecule_id"]): record
        for record in reference_score["molecules"]
    }
    optimized_records = {
        tuple(record["molecule_id"]): record
        for record in optimized_score["molecules"]
    }
    assert optimized_records.keys() == reference_records.keys()
    for molecule_id, record in optimized_records.items():
        reference_record = reference_records[molecule_id]
        assert record["conditional_map_interval"] == reference_record[
            "conditional_map_interval"
        ]
        assert (
            record["standardized_family_posterior_equal_prior"] >= 0.5
        ) == (
            reference_record["standardized_family_posterior_equal_prior"]
            >= 0.5
        )
        assert (
            record["family_vs_null_log_bayes_factor"] >= 0.0
        ) == (
            reference_record["family_vs_null_log_bayes_factor"] >= 0.0
        )
        assert record[
            "standardized_family_posterior_equal_prior"
        ] == pytest.approx(
            reference_record["standardized_family_posterior_equal_prior"],
            abs=1e-11,
        )


def test_boundary_family_legacy_models_default_to_reference_backend():
    positions = np.arange(90, 141, 5, dtype=np.int64)
    reads = [
        _read(
            f"read-{index}",
            "CT" if index % 2 == 0 else "GA",
            positions,
            np.where((positions >= 100) & (positions < 120), 2.0, -1.0),
        )
        for index in range(12)
    ]
    model = strand_rescue_inference.fit_boundary_marginalized_tf_family_model(
        reads,
        "legacy_family",
        [(100, 120)],
        analysis_envelope=(90, 140),
        evidence_summation_mode="slice_sum",
    )
    legacy_model = copy.deepcopy(model)
    del legacy_model["interval_evidence_summation_mode"]
    del legacy_model["interval_evidence_backend_semantics"]

    implicit = strand_rescue_inference.score_boundary_marginalized_tf_family_model(
        reads, legacy_model, include_molecule_records=True
    )
    explicit = strand_rescue_inference.score_boundary_marginalized_tf_family_model(
        reads,
        legacy_model,
        include_molecule_records=True,
        evidence_summation_mode="slice_sum",
    )

    assert implicit == explicit
    assert implicit["interval_evidence_summation_mode"] == "slice_sum"


def test_boundary_family_record_filter_preserves_full_cohort_summaries():
    positions = np.arange(90, 141, 5, dtype=np.int64)
    reads = []
    for index in range(16):
        steps = np.full(positions.size, -1.2, dtype=np.float64)
        steps[(positions >= 100) & (positions < 120)] = 2.0
        reads.append(_read(f"protected-{index}", "CT", positions, steps))
    for index in range(8):
        reads.append(
            _read(
                f"accessible-{index}",
                "GA",
                positions,
                np.full(positions.size, -1.2, dtype=np.float64),
            )
        )
    model = strand_rescue_inference.fit_boundary_marginalized_tf_family_model(
        reads,
        "filtered_family",
        [(100, 120)],
        analysis_envelope=(90, 140),
    )
    full = strand_rescue_inference.score_boundary_marginalized_tf_family_model(
        reads, model, include_molecule_records=True
    )
    filtered = strand_rescue_inference.score_boundary_marginalized_tf_family_model(
        reads,
        model,
        include_molecule_records=True,
        molecule_record_minimum_standardized_family_posterior=0.5,
        molecule_record_minimum_family_vs_null_log_bayes_factor=0.0,
    )
    expected = [
        record
        for record in full["molecules"]
        if record["standardized_family_posterior_equal_prior"] >= 0.5
        and record["family_vs_null_log_bayes_factor"] >= 0.0
    ]

    assert filtered["molecules"] == expected
    assert filtered["molecule_record_selection"] == {
        "eligible_molecules": len(reads),
        "emitted_molecules": len(expected),
        "minimum_standardized_family_posterior": 0.5,
        "minimum_family_vs_null_log_bayes_factor": 0.0,
    }
    for key in (
        "eligible_molecules",
        "total_opportunities",
        "family_effective_molecule_support",
        "standardized_family_effective_molecule_support_equal_prior",
        "family_posterior_over_half_molecules",
        "standardized_family_posterior_over_half_molecules",
        "family_vs_null_log_bayes_factor_nonnegative_molecules",
        "raw_mixture_log_likelihood_ratio",
    ):
        assert filtered[key] == full[key]


def test_frozen_boundary_family_score_propagates_target_lattice_ambiguity():
    training_positions = np.asarray([90, 95, 100, 105, 110, 115, 120, 125, 130, 135])
    training_reads = []
    for index in range(18):
        steps = np.full(training_positions.size, -1.0)
        steps[(training_positions >= 100) & (training_positions < 120)] = 2.0
        training_reads.append(_read(f"train-{index}", "CT", training_positions, steps))
    model = strand_rescue_inference.fit_boundary_marginalized_tf_family_model(
        training_reads,
        "shared_family",
        [(100, 120), (101, 118)],
        boundary_search_radius=2,
        analysis_envelope=(90, 140),
    )
    target_positions = np.asarray([91, 97, 103, 109, 116, 122, 128, 134])
    target_steps = np.full(target_positions.size, -1.0)
    target_steps[(target_positions >= 100) & (target_positions < 120)] = 2.0
    target = _read("target", "GA", target_positions, target_steps)
    positions_before = target.positions.copy()
    steps_before = target.steps.copy()

    score = strand_rescue_inference.score_boundary_marginalized_tf_family_model(
        [target], model, include_molecule_records=True
    )

    assert score["eligible_molecules"] == 1
    record = score["molecules"][0]
    assert 0.0 <= record["family_posterior"] <= 1.0
    assert record["conditional_map_interval"] is not None
    assert record["conditional_boundary_credible_envelope_95"] is not None
    assert record["molecule_opportunity_projection_class_count"] < model[
        "candidate_interval_count"
    ]
    assert np.array_equal(target.positions, positions_before)
    assert np.array_equal(target.steps, steps_before)


def test_boundary_family_equal_prior_evidence_is_invariant_to_block_prior():
    positions = np.arange(90, 141, 5, dtype=np.int64)
    training = []
    for index in range(24):
        steps = np.full(positions.size, -1.2, dtype=np.float64)
        steps[(positions >= 100) & (positions < 120)] = 1.8
        training.append(_read(f"train-{index}", "CT", positions, steps))
    model = strand_rescue_inference.fit_boundary_marginalized_tf_family_model(
        training,
        "shared_family",
        [(100, 120), (101, 118)],
        boundary_search_radius=2,
        analysis_envelope=(90, 140),
    )
    target_steps = np.full(positions.size, -1.2, dtype=np.float64)
    target_steps[(positions >= 100) & (positions < 120)] = 1.8
    target = _read("target", "GA", positions, target_steps)

    def rescale_family_block(value: float) -> dict:
        copied = copy.deepcopy(model)
        weights = np.asarray(copied["component_probabilities"], dtype=float)
        geometry_count = len(copied["geometry_classes"])
        family = weights[1 : 1 + geometry_count]
        null = np.asarray([weights[0], weights[-2], weights[-1]])
        weights[1 : 1 + geometry_count] = value * family / np.sum(family)
        scaled_null = (1.0 - value) * null / np.sum(null)
        weights[0], weights[-2], weights[-1] = scaled_null
        copied["component_probabilities"] = weights.tolist()
        return copied

    low = strand_rescue_inference.score_boundary_marginalized_tf_family_model(
        [target], rescale_family_block(0.05), include_molecule_records=True
    )["molecules"][0]
    high = strand_rescue_inference.score_boundary_marginalized_tf_family_model(
        [target], rescale_family_block(0.95), include_molecule_records=True
    )["molecules"][0]

    assert not math.isclose(low["family_posterior"], high["family_posterior"])
    assert low["family_vs_null_log_bayes_factor"] == pytest.approx(
        high["family_vs_null_log_bayes_factor"], abs=1e-12
    )
    assert low["standardized_family_posterior_equal_prior"] == pytest.approx(
        high["standardized_family_posterior_equal_prior"], abs=1e-12
    )


def test_boundary_family_prior_is_stable_to_unobservable_coordinate_expansion():
    positions = np.arange(90, 141, 10, dtype=np.int64)
    reads = []
    for index in range(24):
        steps = np.full(positions.size, -1.0, dtype=np.float64)
        steps[(positions >= 100) & (positions < 120)] = 2.0
        reads.append(_read(f"protected-{index}", "CT", positions, steps))
    for index in range(12):
        reads.append(
            _read(
                f"accessible-{index}",
                "GA",
                positions,
                np.full(positions.size, -1.0, dtype=np.float64),
            )
        )

    narrow = strand_rescue_inference.fit_boundary_marginalized_tf_family_model(
        reads,
        "shared_family",
        [(100, 120)],
        boundary_search_radius=1,
        analysis_envelope=(90, 140),
    )
    expanded = strand_rescue_inference.fit_boundary_marginalized_tf_family_model(
        reads,
        "shared_family",
        [(100, 120)],
        boundary_search_radius=5,
        analysis_envelope=(90, 140),
    )

    assert expanded["candidate_interval_count"] > narrow["candidate_interval_count"]
    assert expanded["family_probability"] == pytest.approx(
        narrow["family_probability"], abs=2e-3
    )


def test_boundary_family_empty_frozen_score_has_complete_summary_schema():
    positions = np.arange(90, 141, 5, dtype=np.int64)
    training = [
        _read(
            f"train-{index}",
            "CT",
            positions,
            np.where(
                (positions >= 100) & (positions < 120),
                2.0,
                -1.0,
            ),
        )
        for index in range(8)
    ]
    model = strand_rescue_inference.fit_boundary_marginalized_tf_family_model(
        training,
        "shared_family",
        [(100, 120)],
        boundary_search_radius=2,
        analysis_envelope=(90, 140),
    )
    outside = _read(
        "outside", "GA", [200, 205, 210], [-1.0, -1.0, -1.0],
        ref_start=195, ref_end=215,
    )

    score = strand_rescue_inference.score_boundary_marginalized_tf_family_model(
        [outside], model, include_molecule_records=True
    )

    assert score["eligible_molecules"] == 0
    assert score["standardized_family_posterior_over_half_molecules"] == 0
    assert score["family_vs_null_log_bayes_factor_nonnegative_molecules"] == 0
    assert score["median_family_vs_null_log_bayes_factor"] is None


def test_boundary_family_model_ids_include_hyperparameters_and_fit():
    positions = np.arange(90, 141, 5, dtype=np.int64)
    reads = [
        _read(
            f"train-{index}",
            "CT",
            positions,
            np.where(
                (positions >= 100) & (positions < 120),
                2.0,
                -1.0,
            ),
        )
        for index in range(10)
    ]
    low_prior = strand_rescue_inference.fit_boundary_marginalized_tf_family_model(
        reads,
        "shared_family",
        [(100, 120)],
        pseudocount=0.5,
        analysis_envelope=(90, 140),
    )
    high_prior = strand_rescue_inference.fit_boundary_marginalized_tf_family_model(
        reads,
        "shared_family",
        [(100, 120)],
        pseudocount=5.0,
        analysis_envelope=(90, 140),
    )

    assert low_prior["model_structure_id"] != high_prior["model_structure_id"]
    assert low_prior["model_fit_id"] != high_prior["model_fit_id"]
    assert len(low_prior["training_cohort_evidence_sha256"]) == 64


def test_boundary_family_supports_a_shared_spatial_null_universe():
    positions = np.arange(90, 141, 5, dtype=np.int64)
    reads = [
        _read(
            f"train-{index}",
            "CT" if index % 2 == 0 else "GA",
            positions,
            np.where(
                (positions >= 100) & (positions < 122),
                2.0,
                -1.0,
            ),
        )
        for index in range(20)
    ]
    native_probe = strand_rescue_inference.fit_boundary_marginalized_tf_family_model(
        reads,
        "native_probe",
        [(100, 120)],
        boundary_search_radius=1,
        analysis_envelope=(90, 140),
    )
    transported_probe = (
        strand_rescue_inference.fit_boundary_marginalized_tf_family_model(
            reads,
            "transported_probe",
            [(102, 122)],
            boundary_search_radius=1,
            analysis_envelope=(90, 140),
        )
    )
    common_exclusions = sorted(
        {
            tuple(interval)
            for model in (native_probe, transported_probe)
            for interval in model["candidate_intervals"]
        }
    )

    native = strand_rescue_inference.fit_boundary_marginalized_tf_family_model(
        reads,
        "native",
        [(100, 120)],
        boundary_search_radius=1,
        analysis_envelope=(90, 140),
        spatial_null_exclusion_intervals=common_exclusions,
    )
    transported = strand_rescue_inference.fit_boundary_marginalized_tf_family_model(
        reads,
        "transported",
        [(102, 122)],
        boundary_search_radius=1,
        analysis_envelope=(90, 140),
        spatial_null_exclusion_intervals=common_exclusions,
    )

    assert native["candidate_intervals"] != transported["candidate_intervals"]
    assert native["spatial_null_exclusions_explicit"] is True
    assert native["spatial_null_exclusion_intervals"] == transported[
        "spatial_null_exclusion_intervals"
    ]
    assert native["spatial_null_exclusion_sha256"] == transported[
        "spatial_null_exclusion_sha256"
    ]
    assert native["spatial_null_candidate_interval_count"] == transported[
        "spatial_null_candidate_interval_count"
    ]

    with pytest.raises(
        ValueError, match="must contain every anchored family candidate"
    ):
        strand_rescue_inference.fit_boundary_marginalized_tf_family_model(
            reads,
            "invalid_common_null",
            [(100, 120)],
            boundary_search_radius=1,
            analysis_envelope=(90, 140),
            spatial_null_exclusion_intervals=[(99, 119)],
        )


def test_boundary_family_exact_seed_mode_does_not_invent_hybrid_intervals():
    positions = np.arange(90, 141, 5, dtype=np.int64)
    reads = [
        _read(
            f"train-{index}",
            "CT",
            positions,
            np.where(
                (positions >= 100) & (positions < 120),
                2.0,
                -1.0,
            ),
        )
        for index in range(12)
    ]
    seeds = [(100, 120), (105, 115)]

    exact = strand_rescue_inference.fit_boundary_marginalized_tf_family_model(
        reads,
        "exact_support",
        seeds,
        boundary_search_radius=0,
        candidate_interval_mode="exact_seed_intervals",
        analysis_envelope=(90, 140),
    )
    rectangular = strand_rescue_inference.fit_boundary_marginalized_tf_family_model(
        reads,
        "rectangular_support",
        seeds,
        boundary_search_radius=0,
        analysis_envelope=(90, 140),
    )
    seed_local = strand_rescue_inference.fit_boundary_marginalized_tf_family_model(
        reads,
        "seed_local_support",
        seeds,
        boundary_search_radius=1,
        candidate_interval_mode="seed_local_boundary_grid",
        analysis_envelope=(90, 140),
    )

    assert exact["candidate_intervals"] == [[100, 120], [105, 115]]
    assert [100, 115] not in exact["candidate_intervals"]
    assert [105, 120] not in exact["candidate_intervals"]
    assert [100, 115] in rectangular["candidate_intervals"]
    assert [105, 120] in rectangular["candidate_intervals"]
    assert [100, 115] not in seed_local["candidate_intervals"]
    assert [105, 120] not in seed_local["candidate_intervals"]
    assert [99, 120] in seed_local["candidate_intervals"]
    assert [104, 115] in seed_local["candidate_intervals"]
    with pytest.raises(ValueError, match="require boundary_search_radius=0"):
        strand_rescue_inference.fit_boundary_marginalized_tf_family_model(
            reads,
            "invalid_exact_support",
            seeds,
            boundary_search_radius=1,
            candidate_interval_mode="exact_seed_intervals",
            analysis_envelope=(90, 140),
        )


def test_boundary_family_molecule_records_expose_paired_predictive_score():
    positions = np.arange(90, 141, 5, dtype=np.int64)
    reads = [
        _read(
            f"train-{index}",
            "CT",
            positions,
            np.where(
                (positions >= 100) & (positions < 120),
                2.0,
                -1.0,
            ),
        )
        for index in range(12)
    ]
    model = strand_rescue_inference.fit_boundary_marginalized_tf_family_model(
        reads,
        "paired_predictive",
        [(100, 120)],
        boundary_search_radius=1,
        analysis_envelope=(90, 140),
    )

    score = strand_rescue_inference.score_boundary_marginalized_tf_family_model(
        reads, model, include_molecule_records=True
    )

    assert sum(
        molecule["mixture_log_likelihood_ratio_to_accessible"]
        for molecule in score["molecules"]
    ) == pytest.approx(score["raw_mixture_log_likelihood_ratio"], abs=1e-12)
    for molecule in score["molecules"]:
        assert molecule["family_vs_null_log_bayes_factor"] == pytest.approx(
            molecule["family_log_predictive_ratio_to_accessible"]
            - molecule["null_log_predictive_ratio_to_accessible"],
            abs=1e-12,
        )


def test_boundary_family_null_cohort_does_not_force_a_family():
    positions = np.arange(90, 141, 5, dtype=np.int64)
    reads = [
        _read(
            f"accessible-{index}",
            "CT" if index % 2 == 0 else "GA",
            positions,
            np.full(positions.size, -1.5, dtype=np.float64),
        )
        for index in range(40)
    ]

    model = strand_rescue_inference.fit_boundary_marginalized_tf_family_model(
        reads,
        "null_family",
        [(100, 120)],
        boundary_search_radius=3,
        analysis_envelope=(90, 140),
    )

    assert model["family_probability"] < 0.05
    assert model["component_probabilities"][0] > 0.9
