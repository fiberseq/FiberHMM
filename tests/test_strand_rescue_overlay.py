from __future__ import annotations

import errno
import hashlib
import json
from array import array
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pysam
import pytest

from fiberhmm.cli.strand_rescue_annotate import (
    HEADER_PREFIX,
    LEGACY_HEADER_PREFIX,
    GeometryDecision,
    ResolvedV5ActionStream,
    StrandDecision,
    V6_HEADER_PREFIX,
    V5_ACTION_STREAM_SCHEMA,
    V5ActionStreamReader,
    _publish_staged_outputs,
    _sha256_file,
    _validate_v5_action_storage,
    add_strand_rescue_groups,
    collect_decisions,
    collect_harmonizations,
    geometry_quality_row,
    main,
    reference_interval_to_molecular,
    rescue_quality_row,
    write_overlay_bam,
    write_streaming_overlay_bam,
)
from fiberhmm.cli.strand_rescue_audit import audit_bam, audit_bams
from fiberhmm.io.bam_header import declared_ma_types
from fiberhmm.io.ma_tags import parse_an_tag, parse_aq_array, parse_ma_tag


class StubRead:
    def __init__(self, *, reverse=False):
        self.query_name = "read1"
        self.query_length = 100
        self.query_sequence = "A" * 100
        self.is_reverse = reverse
        self._tags = {
            "MA": "100;nuc.Q:11-20;msp.:31-30;tf.QQQ:70-10",
            "AQ": array("B", [200, 150, 10, 20]),
        }

    def get_reference_positions(self, full_length=False):
        assert full_length
        return list(range(100, 200))

    def has_tag(self, name):
        return name in self._tags

    def get_tag(self, name):
        return self._tags[name]

    def set_tag(self, name, value, value_type=None):
        if value is None:
            self._tags.pop(name, None)
        else:
            self._tags[name] = value


def _decision():
    return StrandDecision(
        decision_id="rescue-decision",
        read_name="read1",
        library_id="input.bam",
        tier="review",
        current_state="A",
        current_interval=(130, 160),
        tf_intervals=((135, 145), (150, 160)),
        tf_posterior=0.70,
        current_posterior=0.30,
        configuration_probability=0.80,
        molecule_probability=0.60,
        population_probability=0.75,
        support_reliability=0.90,
        geometry_reliability=0.85,
        edge_confidences=((0.80, 0.60), (0.70, 0.90)),
    )


def _geometry_decision():
    return GeometryDecision(
        decision_id="geometry-decision",
        read_name="read1",
        library_id="input.bam",
        current_interval=(169, 179),
        canonical_interval=(168, 180),
        assignment_probability=0.95,
        molecule_probability=0.85,
        population_probability=0.75,
        geometry_reliability=0.90,
        alternative_probability=0.65,
        left_edge_probability=0.80,
        right_edge_probability=0.90,
    )


def _raw_groups(read):
    parsed = parse_ma_tag(read.get_tag("MA"))
    return {
        name: (specification, intervals)
        for name, _strand, specification, intervals in parsed["raw_types"]
    }


def _quality_rows(read):
    parsed = parse_ma_tag(read.get_tag("MA"))
    return parse_aq_array(
        read.get_tag("AQ"),
        [raw[2] for raw in parsed["raw_types"]],
        [len(raw[3]) for raw in parsed["raw_types"]],
    )


def _ordinary_records(read):
    parsed = parse_ma_tag(read.get_tag("MA"))
    rows = _quality_rows(read)
    names = (
        parse_an_tag(read.get_tag("AN"))
        if read.has_tag("AN")
        else [""] * sum(len(raw[3]) for raw in parsed["raw_types"])
    )
    result = []
    cursor = 0
    for name, strand, specification, intervals in parsed["raw_types"]:
        for interval in intervals:
            if name not in {"nuc_sr", "tf_sr"}:
                result.append(
                    (name, strand, specification, interval, rows[cursor], names[cursor])
                )
            cursor += 1
    return result


def test_quality_rows_share_q0_but_keep_component_edge_confidence():
    assert rescue_quality_row(_decision(), 0) == (178, 204, 153)
    assert rescue_quality_row(_decision(), 1) == (178, 178, 230)
    assert geometry_quality_row(_geometry_decision()) == (166, 204, 230)


def test_v4_report_parser_uses_exact_alternative_and_per_component_edges():
    report = {
        "schema": "fiberhmm.strand_rescue.v4",
        "strand_rescue": {
            "decisions": [
                {
                    "decision_id": "r1",
                    "read": "read1",
                    "library_id": "input.bam",
                    "current": "A",
                    "current_interval": [130, 160],
                    "proposed_site_intervals": [[135, 145], [150, 160]],
                    "proposed_site_edge_confidence": [
                        [0.8, 0.6],
                        [0.7, 0.9],
                    ],
                    "sr_hypothesis_probability": 0.7,
                    "baseline_hypothesis_probability": 0.3,
                }
            ]
        },
    }

    decision = collect_decisions(report)[0]

    assert decision.tf_posterior == 0.7
    assert decision.current_posterior == pytest.approx(0.3)
    assert decision.edge_confidences == ((0.8, 0.6), (0.7, 0.9))


def test_v6_report_parser_preserves_competing_configuration_mass():
    report = {
        "schema": "fiberhmm.strand_rescue.v6",
        "schema_version": 6,
        "strand_rescue": {
            "decisions": [
                {
                    "decision_id": "v6-r1",
                    "read": "read1",
                    "library_id": "input.bam",
                    "current": "A",
                    "current_interval": [130, 160],
                    "proposed_site_intervals": [[135, 145]],
                    "proposed_site_edge_confidence": [[0.8, 0.6]],
                    "sr_hypothesis_probability": 0.25,
                    "baseline_hypothesis_probability": 0.20,
                    "other_supported_tf_configuration_probability": 0.55,
                }
            ]
        },
    }

    decision = collect_decisions(report)[0]

    assert decision.tf_posterior == 0.25
    assert decision.current_posterior == 0.20
    assert rescue_quality_row(decision, 0)[0] == 64


def test_v6_report_parser_accepts_unresolved_incomplete_action_set():
    report = {
        "schema": "fiberhmm.strand_rescue.v6",
        "schema_version": 6,
        "strand_rescue": {
            "decisions": [
                {
                    "decision_id": "v6-capped",
                    "read": "read1",
                    "library_id": "input.bam",
                    "current": "A",
                    "current_interval": [130, 160],
                    "proposed_site_intervals": [[135, 145]],
                    "proposed_site_edge_confidence": [[0.8, 0.6]],
                    "sr_hypothesis_probability": 0.0,
                    "baseline_hypothesis_probability": 0.0,
                    "other_supported_tf_configuration_probability": 0.0,
                    "unresolved_action_set_probability": 1.0,
                    "q0_action_set_complete": False,
                }
            ]
        },
    }

    decision = collect_decisions(report)[0]

    assert decision.tf_posterior == 0.0
    assert decision.current_posterior == 0.0
    assert rescue_quality_row(decision, 0)[0] == 0


def test_v3_report_parser_derives_exact_configuration_probability():
    report = {
        "schema": "fiberhmm.strand_rescue.v3",
        "strand_rescue": {
            "decisions": [
                {
                    "decision_id": "legacy-r1",
                    "read": "read1",
                    "library_id": "input.bam",
                    "current": "A",
                    "current_interval": [130, 160],
                    "proposed_site_intervals": [[135, 145]],
                    "posterior": 0.8,
                    "current_posterior": 0.2,
                    "best_configuration_posterior_given_tf": 0.5,
                    "canonical_geometry_reliability": 0.75,
                }
            ]
        },
    }

    decision = collect_decisions(report)[0]

    assert decision.tf_posterior == pytest.approx(2.0 / 3.0)
    assert decision.edge_confidences == ((0.75, 0.75),)


def test_v4_harmonization_parser_uses_joint_and_marginal_edge_probabilities():
    raw = {
        "decision_id": "h1",
        "read": "read1",
        "library_id": "input.bam",
        "status": "edge_update",
        "call_type": "nuc",
        "current_interval": [110, 130],
        "canonical_interval": [108, 132],
        "edge_hypothesis": {
            "alternative_probability": 0.65,
            "left": {"alternative_probability": 0.8},
            "right": {"alternative_probability": 0.9},
        },
    }
    report = {
        "schema": "fiberhmm.strand_rescue.v4",
        "strand_rescue": {
            "edge_refinement": {
                "tf": {"harmonizations": []},
                "nuc": {"harmonizations": [raw]},
            }
        },
    }

    decision = collect_harmonizations(report)[0]

    assert decision.call_type == "nuc"
    assert geometry_quality_row(decision) == (166, 204, 230)


def test_v6_report_overlay_uses_materialized_edge_confidence(tmp_path):
    source = tmp_path / "source.bam"
    output = tmp_path / "v6.bam"
    _make_bam(source)
    raw = {
        "decision_id": "h1",
        "read": "read1",
        "library_id": str(source),
        "status": "edge_update",
        "call_type": "tf",
        "current_interval": [169, 179],
        "canonical_interval": [168, 180],
        "current_molecular_interval": [69, 10],
        "current_annotation_ordinal": 0,
        "edge_hypothesis": {
            "alternative_probability": 0.72,
            "left": {"alternative_probability": 0.45},
            "right": {"alternative_probability": 1.0},
        },
        "materialized_edge_confidence": [0.0, 1.0],
    }
    report = {
        "schema": "fiberhmm.strand_rescue.v6",
        "strand_rescue": {
            "edge_refinement": {
                "tf": {"harmonizations": [raw]},
                "nuc": {"harmonizations": []},
            }
        },
    }
    decision = collect_harmonizations(report)[0]

    assert geometry_quality_row(decision) == (184, 0, 255)
    assert geometry_quality_row(decision, reverse=True) == (184, 255, 0)

    write_overlay_bam(
        str(source),
        output,
        region=("chr1", 90, 210),
        decisions_by_read={},
        harmonizations_by_read={"read1": [decision]},
        report_sha256="a" * 64,
        minimum_posterior=0.0,
        command_line="fiberhmm-strand-rescue-annotate v6-test",
        require_all_matches=True,
        quality_contract_version=6,
    )
    with pysam.AlignmentFile(output, "rb") as bam:
        read = next(bam.fetch(until_eof=True))
    names = parse_an_tag(read.get_tag("AN"))
    quality_rows = _quality_rows(read)
    named_rows = {
        name.rsplit("_", 1)[-1]: tuple(row)
        for name, row in zip(names, quality_rows)
        if name not in {"", "."}
    }
    assert named_rows["H"] == (184, 0, 255)
    assert audit_bam(str(output))["valid"] is True


def test_msp_rescue_and_existing_tf_harmonization_write_complete_layers():
    read = StubRead()

    counts = add_strand_rescue_groups(
        read, [_decision()], [_geometry_decision()]
    )

    groups = _raw_groups(read)
    assert groups["nuc"] == ("Q", [(10, 20)])
    assert groups["nuc_sr"] == ("QQQ", [(10, 20)])
    assert groups["tf_sr"] == (
        "QQQ",
        [(35, 10), (50, 10), (68, 12)],
    )
    assert counts == {"nuc_sr": 1, "tf_sr": 3}
    names = [
        name for name in read.get_tag("AN").split(",") if name not in {"", "."}
    ]
    assert sorted(name.rsplit("_", 1)[1] for name in names) == ["H", "R0", "R1"]
    rescue_rows = [row for row in _quality_rows(read) if row and row[0] == 178]
    assert rescue_rows == [
        list(rescue_quality_row(_decision(), 0)),
        list(rescue_quality_row(_decision(), 1)),
    ]


def test_nucleosome_tags_are_byte_for_byte_unchanged():
    read = StubRead()
    before = _raw_groups(read)["nuc"]

    add_strand_rescue_groups(read, [_decision()], [_geometry_decision()])

    assert _raw_groups(read)["nuc"] == before
    assert _raw_groups(read)["nuc_sr"] == ("QQQ", [(10, 20)])


def test_every_ordinary_annotation_quality_and_name_is_preserved():
    read = StubRead()
    read._tags["AN"] = "nuc-id,.,tf-id"
    before = _ordinary_records(read)

    add_strand_rescue_groups(read, [_decision()], [_geometry_decision()])

    assert _ordinary_records(read) == before


def test_rescue_collision_never_replaces_baseline_or_writes_partial_group():
    read = StubRead()
    read._tags["MA"] = "100;nuc.Q:11-20;msp.:31-30;tf.QQQ:56-10"
    applied = set()

    add_strand_rescue_groups(
        read,
        [_decision()],
        applied_decision_ids=applied,
    )

    groups = _raw_groups(read)
    assert groups["tf_sr"] == ("QQQ", [(55, 10)])
    assert groups["nuc_sr"] == ("QQQ", [(10, 20)])
    assert applied == set()
    assert not read.has_tag("AN")


def test_multi_tf_rescue_projection_and_msp_containment_are_atomic():
    for intervals in (
        ((135, 145), (95, 105)),
        ((135, 145), (165, 175)),
    ):
        read = StubRead()
        applied = set()
        decision = replace(_decision(), tf_intervals=intervals)

        add_strand_rescue_groups(
            read,
            [decision],
            applied_decision_ids=applied,
        )

        assert _raw_groups(read)["tf_sr"] == ("QQQ", [(69, 10)])
        assert applied == set()
        assert not read.has_tag("AN")


def test_reverse_projection_uses_molecular_coordinate_frame():
    read = StubRead(reverse=True)
    assert reference_interval_to_molecular(read, 110, 130) == (70, 20)


def test_reverse_materialization_swaps_reference_edge_confidences():
    read = StubRead(reverse=True)
    read._tags["MA"] = "100;msp.:41-30"
    read._tags["AQ"] = array("B")
    decision = replace(
        _decision(),
        current_molecular_interval=(40, 30),
        current_annotation_ordinal=0,
    )

    add_strand_rescue_groups(read, [decision])

    names = parse_an_tag(read.get_tag("AN"))
    rows = _quality_rows(read)
    by_role = {
        name.rsplit("_", 1)[1]: tuple(row)
        for name, row in zip(names, rows)
        if name and "_R" in name
    }
    assert by_role == {
        "R0": (178, 153, 204),
        "R1": (178, 230, 178),
    }


def test_nuc_edge_harmonization_is_one_for_one_and_keeps_ordinary_nuc():
    read = StubRead()
    decision = replace(
        _geometry_decision(),
        decision_id="nuc-edge",
        call_type="nuc",
        current_interval=(110, 130),
        canonical_interval=(108, 132),
        current_molecular_interval=(10, 20),
        current_annotation_ordinal=0,
    )
    applied = set()

    counts = add_strand_rescue_groups(
        read,
        [],
        [decision],
        applied_harmonization_ids=applied,
    )

    groups = _raw_groups(read)
    assert groups["nuc"] == ("Q", [(10, 20)])
    assert groups["nuc_sr"] == ("QQQ", [(8, 24)])
    assert groups["tf_sr"] == ("QQQ", [(69, 10)])
    assert counts == {"nuc_sr": 1, "tf_sr": 1}
    assert applied == {"nuc-edge"}


def test_edge_materialization_rejects_partial_span_and_new_cross_layer_overlap():
    read = StubRead()
    partial = replace(
        _geometry_decision(),
        decision_id="partial",
        call_type="nuc",
        current_interval=(110, 130),
        canonical_interval=(95, 130),
        current_molecular_interval=(10, 20),
        current_annotation_ordinal=0,
    )
    overlapping = replace(
        partial,
        decision_id="overlap",
        canonical_interval=(110, 175),
    )
    applied = set()

    add_strand_rescue_groups(
        read,
        [],
        [partial, overlapping],
        applied_harmonization_ids=applied,
    )

    assert _raw_groups(read)["nuc_sr"] == ("QQQ", [(10, 20)])
    assert applied == set()


def test_topology_only_ordinary_nuc_remains_a_complete_baseline_shadow():
    read = StubRead()
    read._tags["MA"] = "100;nuc.Q:1-95"
    read._tags["AQ"] = array("B", [200])

    counts = add_strand_rescue_groups(read, [], [])

    assert _raw_groups(read)["nuc"] == ("Q", [(0, 95)])
    assert _raw_groups(read)["nuc_sr"] == ("QQQ", [(0, 95)])
    assert _quality_rows(read) == [[200], [255, 0, 0]]
    assert counts == {"nuc_sr": 1}


def test_duplicate_ordinary_intervals_each_receive_one_shadow_annotation():
    read = StubRead()
    read._tags["MA"] = "100;nuc.Q:11-20,11-20"
    read._tags["AQ"] = array("B", [200, 201])

    counts = add_strand_rescue_groups(read, [], [])

    assert _raw_groups(read)["nuc"] == ("Q", [(10, 20), (10, 20)])
    assert _raw_groups(read)["nuc_sr"] == (
        "QQQ",
        [(10, 20), (10, 20)],
    )
    assert counts == {"nuc_sr": 2}


def test_alignment_occurrence_routes_byte_identical_records_separately():
    first = replace(_decision(), alignment_occurrence=0)
    second = replace(
        _decision(),
        decision_id="second-occurrence",
        alignment_occurrence=1,
        tf_posterior=0.9,
    )
    read = StubRead()
    applied = set()

    add_strand_rescue_groups(
        read,
        [first, second],
        alignment_occurrence=0,
        applied_decision_ids=applied,
    )

    assert applied == {"rescue-decision"}
    rescue_rows = [row for row in _quality_rows(read) if row and row[0] == 178]
    assert len(rescue_rows) == 2


def _make_bam(
    path: Path,
    *,
    comments=(),
    ma="100;nuc.Q:11-20;msp.:31-30;tf.QQQ:70-10",
    aq=(200, 150, 10, 20),
    flag=0,
):
    header_dict = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": 1000}],
    }
    if comments:
        header_dict["CO"] = list(comments)
    header = pysam.AlignmentHeader.from_dict(header_dict)
    with pysam.AlignmentFile(path, "wb", header=header) as bam:
        read = pysam.AlignedSegment(header)
        read.query_name = "read1"
        read.query_sequence = "A" * 100
        read.flag = flag
        read.reference_id = 0
        read.reference_start = 100
        read.mapping_quality = 60
        read.cigarstring = "100M"
        read.query_qualities = pysam.qualitystring_to_array("I" * 100)
        if ma is not None:
            read.set_tag(
                "MA",
                ma,
                value_type="Z",
            )
            read.set_tag("AQ", array("B", aq))
        bam.write(read)
    pysam.index(str(path))


def _source_record(path: Path, region=("chr1", 90, 210), ordinal=0):
    with pysam.AlignmentFile(path, "rb") as bam:
        for index, read in enumerate(bam.fetch(*region)):
            if index == ordinal:
                return read
    raise AssertionError(f"no record at ordinal {ordinal}")


def _jsonl_line(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )


def _make_v5_action_stream(
    directory: Path,
    source: Path,
    action_rows,
    *,
    region=("chr1", 90, 210),
    fetch_record_count=1,
    input_index=0,
):
    input_id = f"input{input_index:04d}"
    rows = list(action_rows)
    rescue_count = sum(len(row["rescues"]) for row in rows)
    component_count = sum(
        len(rescue["components"])
        for row in rows
        for rescue in row["rescues"]
    )
    tf_edges = sum(
        edge["call_type"] == "tf"
        for row in rows
        for edge in row["edge_updates"]
    )
    nuc_edges = sum(
        edge["call_type"] == "nuc"
        for row in rows
        for edge in row["edge_updates"]
    )
    header = {
        "kind": "header",
        "schema": V5_ACTION_STREAM_SCHEMA,
        "input_index": input_index,
        "input_id": input_id,
        "loaded_region": list(region),
        "ordinal_base": 0,
    }
    trailer = {
        "kind": "trailer",
        "fetch_record_count": fetch_record_count,
        "action_record_count": len(rows),
        "first_action_ordinal": rows[0]["ordinal"] if rows else None,
        "last_action_ordinal": rows[-1]["ordinal"] if rows else None,
        "rescue_decision_count": rescue_count,
        "rescue_component_count": component_count,
        "tf_edge_update_count": tf_edges,
        "nuc_edge_update_count": nuc_edges,
    }
    uncompressed = b"".join(
        [_jsonl_line(header), *(_jsonl_line(row) for row in rows), _jsonl_line(trailer)]
    )
    jsonl_sha256 = hashlib.sha256(uncompressed).hexdigest()
    basename = (
        f"fixture.input{input_index:04d}.sr-actions."
        f"{jsonl_sha256[:12]}.jsonl.bgz"
    )
    bgzf_path = directory / basename
    gzi_path = Path(f"{bgzf_path}.gzi")
    with pysam.BGZFile(str(bgzf_path), "wb", index=str(gzi_path)) as handle:
        handle.write(uncompressed)
    manifest = {
        "input_index": input_index,
        "input_id": input_id,
        "path": basename,
        "gzi_path": f"{basename}.gzi",
        "loaded_region": list(region),
        "fetch_record_count": fetch_record_count,
        "action_record_count": len(rows),
        "first_action_ordinal": trailer["first_action_ordinal"],
        "last_action_ordinal": trailer["last_action_ordinal"],
        "rescue_decision_count": rescue_count,
        "rescue_component_count": component_count,
        "tf_edge_update_count": tf_edges,
        "nuc_edge_update_count": nuc_edges,
        "compressed_size_bytes": bgzf_path.stat().st_size,
        "uncompressed_size_bytes": len(uncompressed),
        "bgzf_sha256": _sha256_file(bgzf_path),
        "jsonl_sha256": jsonl_sha256,
        "gzi_size_bytes": gzi_path.stat().st_size,
        "gzi_sha256": _sha256_file(gzi_path),
    }
    return ResolvedV5ActionStream(
        input_index=input_index,
        input_path=str(source.resolve()),
        manifest=manifest,
        bgzf_path=bgzf_path,
        gzi_path=gzi_path,
    )


def _v5_action_row(read, *, ordinal=0, rescues=(), edge_updates=()):
    return {
        "kind": "actions",
        "ordinal": ordinal,
        "read": read.query_name,
        "record_sha256": hashlib.sha256(
            read.to_string().encode("utf-8")
        ).hexdigest(),
        "rescues": list(rescues),
        "edge_updates": list(edge_updates),
    }


def _v5_report(source: Path, stream: ResolvedV5ActionStream):
    stat = source.stat()
    manifest = dict(stream.manifest)
    totals = {
        "fetch_records": manifest["fetch_record_count"],
        "action_records": manifest["action_record_count"],
        "rescue_decisions": manifest["rescue_decision_count"],
        "rescue_components": manifest["rescue_component_count"],
        "tf_edge_updates": manifest["tf_edge_update_count"],
        "nuc_edge_updates": manifest["nuc_edge_update_count"],
    }
    return {
        "schema": "fiberhmm.strand_rescue.v5",
        "schema_version": 5,
        "input": {
            "bams": [str(source.resolve())],
            "files": [
                {
                    "path": str(source.resolve()),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            ],
            "loaded_region": manifest["loaded_region"],
        },
        "strand_rescue": {
            "edge_refinement": {
                "tf": {"call_type": "tf", "sites": [], "counts": {}},
                "nuc": {"call_type": "nuc", "sites": [], "counts": {}},
            },
            "action_storage": {
                "layout": "per_input_bgzf_jsonl_v1",
                "stream_schema": V5_ACTION_STREAM_SCHEMA,
                "quality_encoding": "uint8_round_255_times_unit_probability",
                "coordinate_frame": "molecular_zero_based_start_length",
                "streams": [manifest],
                "totals": totals,
            },
        },
    }


def test_regional_overlay_is_indexed_and_passes_v4_audit(tmp_path):
    source = tmp_path / "source.bam"
    output = tmp_path / "overlay.bam"
    _make_bam(source)

    summary = write_overlay_bam(
        str(source),
        output,
        region=("chr1", 90, 210),
        decisions_by_read={"read1": [_decision()]},
        harmonizations_by_read={"read1": [_geometry_decision()]},
        report_sha256="a" * 64,
        minimum_posterior=0.0,
        command_line="fiberhmm-strand-rescue-annotate test",
    )

    assert summary["decisions_matched"] == 1
    assert summary["harmonizations_matched"] == 1
    assert output.is_file()
    assert Path(str(output) + ".bai").is_file()
    with pysam.AlignmentFile(output, "rb") as bam:
        assert {"nuc_sr", "tf_sr"} <= set(declared_ma_types(bam.header))
        read = next(bam.fetch(until_eof=True))
        assert any("_O0_H" in name for name in parse_an_tag(read.get_tag("AN")))
    audit = audit_bam(str(output))
    assert audit["valid"] is True
    assert audit["counts"]["msp_to_tf_rescues"] == 1
    assert audit["counts"]["geometry_harmonizations"] == 1
    assert audit["contract_version"] == 4
    assert audit["threshold_states"]["128"] == {
        "all": {"SR": 2, "baseline": 0},
        "R": {"TF": 1, "A": 0},
        "H": {"SR_edges": 1, "baseline_edges": 0},
    }
    assert audit_bams([str(output)])["threshold_states"] == audit[
        "threshold_states"
    ]


def test_deferred_overlay_is_not_published_until_the_cohort_is_ready(tmp_path):
    source = tmp_path / "source.bam"
    output = tmp_path / "overlay.bam"
    _make_bam(source)

    summary = write_overlay_bam(
        str(source),
        output,
        region=("chr1", 90, 210),
        decisions_by_read={},
        harmonizations_by_read={},
        report_sha256="a" * 64,
        minimum_posterior=0.0,
        command_line="fiberhmm-strand-rescue-annotate test",
        defer_publish=True,
    )

    assert not output.exists()
    assert not Path(str(output) + ".bai").exists()
    assert Path(summary["_staged_output"]).is_file()
    assert Path(summary["_staged_index"]).is_file()
    Path(summary["_staged_output"]).unlink()
    Path(summary["_staged_index"]).unlink()


def test_cohort_publication_restores_existing_outputs_after_rename_failure(
    tmp_path, monkeypatch
):
    stage_bam = tmp_path / "stage.bam"
    stage_bai = tmp_path / "stage.bam.bai"
    output_bam = tmp_path / "output.bam"
    output_bai = tmp_path / "output.bam.bai"
    stage_bam.write_bytes(b"new-bam")
    stage_bai.write_bytes(b"new-index")
    output_bam.write_bytes(b"old-bam")
    output_bai.write_bytes(b"old-index")
    summary = {
        "_staged_output": str(stage_bam),
        "_staged_index": str(stage_bai),
        "output": str(output_bam),
        "index": str(output_bai),
    }
    real_replace = __import__("os").replace

    def fail_on_index(source, destination):
        if Path(source) == stage_bai:
            raise OSError("injected publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(
        "fiberhmm.cli.strand_rescue_annotate.os.replace", fail_on_index
    )

    with pytest.raises(OSError, match="injected publication failure"):
        _publish_staged_outputs([summary])

    assert output_bam.read_bytes() == b"old-bam"
    assert output_bai.read_bytes() == b"old-index"
    assert not stage_bam.exists()
    assert not stage_bai.exists()


def test_cohort_publication_retries_transient_permission_error(
    tmp_path, monkeypatch
):
    stage_bam = tmp_path / "stage.bam"
    stage_bai = tmp_path / "stage.bam.bai"
    output_bam = tmp_path / "output.bam"
    output_bai = tmp_path / "output.bam.bai"
    stage_bam.write_bytes(b"new-bam")
    stage_bai.write_bytes(b"new-index")
    output_bam.write_bytes(b"old-bam")
    output_bai.write_bytes(b"old-index")
    summary = {
        "_staged_output": str(stage_bam),
        "_staged_index": str(stage_bai),
        "output": str(output_bam),
        "index": str(output_bai),
    }
    real_replace = __import__("os").replace
    index_attempts = 0
    sleeps = []

    def transient_on_index(source, destination):
        nonlocal index_attempts
        if Path(source) == stage_bai:
            index_attempts += 1
            if index_attempts < 3:
                raise PermissionError(errno.EACCES, "transient sharing lock")
        return real_replace(source, destination)

    monkeypatch.setattr(
        "fiberhmm.cli.strand_rescue_annotate.os.replace", transient_on_index
    )
    monkeypatch.setattr(
        "fiberhmm.cli.strand_rescue_annotate.time.sleep", sleeps.append
    )

    _publish_staged_outputs([summary])

    assert index_attempts == 3
    assert sleeps == [0.05, 0.1]
    assert output_bam.read_bytes() == b"new-bam"
    assert output_bai.read_bytes() == b"new-index"
    assert not stage_bam.exists()
    assert not stage_bai.exists()
    assert not list(tmp_path.glob("*.strand-rescue-backup"))


def test_cohort_publication_permission_retry_is_bounded_and_rolls_back(
    tmp_path, monkeypatch
):
    stage_bam = tmp_path / "stage.bam"
    stage_bai = tmp_path / "stage.bam.bai"
    output_bam = tmp_path / "output.bam"
    output_bai = tmp_path / "output.bam.bai"
    stage_bam.write_bytes(b"new-bam")
    stage_bai.write_bytes(b"new-index")
    output_bam.write_bytes(b"old-bam")
    output_bai.write_bytes(b"old-index")
    summary = {
        "_staged_output": str(stage_bam),
        "_staged_index": str(stage_bai),
        "output": str(output_bam),
        "index": str(output_bai),
    }
    real_replace = __import__("os").replace
    index_attempts = 0
    sleeps = []

    def locked_index(source, destination):
        nonlocal index_attempts
        if Path(source) == stage_bai:
            index_attempts += 1
            raise PermissionError(errno.EACCES, "persistent sharing lock")
        return real_replace(source, destination)

    monkeypatch.setattr(
        "fiberhmm.cli.strand_rescue_annotate.os.replace", locked_index
    )
    monkeypatch.setattr(
        "fiberhmm.cli.strand_rescue_annotate.time.sleep", sleeps.append
    )

    with pytest.raises(PermissionError, match="persistent sharing lock"):
        _publish_staged_outputs([summary])

    assert index_attempts == 10
    assert len(sleeps) == 9
    assert sum(sleeps) == pytest.approx(4.5)
    assert output_bam.read_bytes() == b"old-bam"
    assert output_bai.read_bytes() == b"old-index"
    assert not stage_bam.exists()
    assert not stage_bai.exists()
    assert not list(tmp_path.glob("*.strand-rescue-backup"))


def test_cohort_publication_retries_transient_permission_during_rollback(
    tmp_path, monkeypatch
):
    stage_bam = tmp_path / "stage.bam"
    stage_bai = tmp_path / "stage.bam.bai"
    output_bam = tmp_path / "output.bam"
    output_bai = tmp_path / "output.bam.bai"
    stage_bam.write_bytes(b"new-bam")
    stage_bai.write_bytes(b"new-index")
    output_bam.write_bytes(b"old-bam")
    output_bai.write_bytes(b"old-index")
    summary = {
        "_staged_output": str(stage_bam),
        "_staged_index": str(stage_bai),
        "output": str(output_bam),
        "index": str(output_bai),
    }
    real_replace = __import__("os").replace
    rollback_attempts = 0
    sleeps = []

    def fail_publication_then_retry_rollback(source, destination):
        nonlocal rollback_attempts
        source = Path(source)
        destination = Path(destination)
        if source == stage_bai:
            raise OSError("injected publication failure")
        if (
            destination == output_bam
            and source.name.endswith(".strand-rescue-backup")
        ):
            rollback_attempts += 1
            if rollback_attempts == 1:
                raise PermissionError(errno.EPERM, "transient rollback lock")
        return real_replace(source, destination)

    monkeypatch.setattr(
        "fiberhmm.cli.strand_rescue_annotate.os.replace",
        fail_publication_then_retry_rollback,
    )
    monkeypatch.setattr(
        "fiberhmm.cli.strand_rescue_annotate.time.sleep", sleeps.append
    )

    with pytest.raises(OSError, match="injected publication failure"):
        _publish_staged_outputs([summary])

    assert rollback_attempts == 2
    assert sleeps == [0.05]
    assert output_bam.read_bytes() == b"old-bam"
    assert output_bai.read_bytes() == b"old-index"
    assert not stage_bam.exists()
    assert not stage_bai.exists()
    assert not list(tmp_path.glob("*.strand-rescue-backup"))


def test_duplicate_ordinary_intervals_pass_one_for_one_v4_audit(tmp_path):
    source = tmp_path / "source.bam"
    output = tmp_path / "overlay.bam"
    _make_bam(
        source,
        ma="100;nuc.Q:11-20,11-20",
        aq=(200, 201),
    )

    write_overlay_bam(
        str(source),
        output,
        region=("chr1", 90, 210),
        decisions_by_read={},
        harmonizations_by_read={},
        report_sha256="a" * 64,
        minimum_posterior=0.0,
        command_line="fiberhmm-strand-rescue-annotate test",
    )

    result = audit_bam(str(output))
    assert result["valid"] is True
    assert result["counts"]["annotations_nuc_sr"] == 2


def test_v4_audit_rejects_h_with_invalid_source_ordinal(tmp_path):
    source = tmp_path / "source.bam"
    valid = tmp_path / "valid.bam"
    invalid = tmp_path / "invalid.bam"
    _make_bam(source)
    write_overlay_bam(
        str(source),
        valid,
        region=("chr1", 90, 210),
        decisions_by_read={},
        harmonizations_by_read={"read1": [_geometry_decision()]},
        report_sha256="a" * 64,
        minimum_posterior=0.0,
        command_line="fiberhmm-strand-rescue-annotate test",
    )

    with pysam.AlignmentFile(valid, "rb") as input_bam:
        with pysam.AlignmentFile(invalid, "wb", header=input_bam.header) as output_bam:
            read = next(input_bam.fetch(until_eof=True))
            read.set_tag(
                "AN",
                read.get_tag("AN").replace("_O0_H", "_O9_H"),
                value_type="Z",
            )
            output_bam.write(read)
    pysam.index(str(invalid))

    result = audit_bam(str(invalid))
    assert result["valid"] is False
    assert any(
        error["message"] == "H source annotation ordinal is out of range"
        for error in result["errors"]
    )


def test_v4_audit_rejects_swapped_h_source_order(tmp_path):
    source = tmp_path / "source.bam"
    baseline = tmp_path / "baseline.bam"
    invalid = tmp_path / "invalid.bam"
    _make_bam(
        source,
        ma="100;tf.QQQ:11-10,41-10",
        aq=(150, 10, 20, 151, 11, 21),
    )
    write_overlay_bam(
        str(source),
        baseline,
        region=("chr1", 90, 210),
        decisions_by_read={},
        harmonizations_by_read={},
        report_sha256="a" * 64,
        minimum_posterior=0.0,
        command_line="fiberhmm-strand-rescue-annotate test",
    )

    with pysam.AlignmentFile(baseline, "rb") as input_bam:
        with pysam.AlignmentFile(invalid, "wb", header=input_bam.header) as output_bam:
            read = next(input_bam.fetch(until_eof=True))
            read.set_tag(
                "AN",
                ".,.,fhsr_1111111111111111_O1_H,fhsr_2222222222222222_O0_H",
                value_type="Z",
            )
            output_bam.write(read)
    pysam.index(str(invalid))

    result = audit_bam(str(invalid))
    assert result["valid"] is False
    assert any(
        error["message"] == "SR shadow call order differs from ordinary calls"
        for error in result["errors"]
    )


def test_legacy_v2_contract_is_replaced_when_writing_v4(tmp_path):
    source = tmp_path / "legacy.bam"
    output = tmp_path / "v4.bam"
    legacy = (
        LEGACY_HEADER_PREFIX
        + "groups=tf_sr;semantics=two_strand_tf_normalization;quality_spec=QQQQQ"
    )
    _make_bam(source, comments=(legacy, "unrelated-comment"))

    write_overlay_bam(
        str(source),
        output,
        region=("chr1", 90, 210),
        decisions_by_read={},
        harmonizations_by_read={},
        report_sha256="a" * 64,
        minimum_posterior=0.0,
        command_line="fiberhmm-strand-rescue-annotate test",
    )

    with pysam.AlignmentFile(output, "rb") as bam:
        comments = [str(value) for value in bam.header.to_dict().get("CO", [])]
    assert "unrelated-comment" in comments
    assert not any(value.startswith(LEGACY_HEADER_PREFIX) for value in comments)
    assert sum(value.startswith(HEADER_PREFIX) for value in comments) == 1
    contract = next(value for value in comments if value.startswith(HEADER_PREFIX))
    assert "changed_edge_without_target_opportunity_q=0" not in contract
    assert audit_bam(str(output))["valid"] is True


def test_v6_quality_contract_is_declared_and_audited(tmp_path):
    source = tmp_path / "source.bam"
    output = tmp_path / "v6.bam"
    _make_bam(source)

    write_overlay_bam(
        str(source),
        output,
        region=("chr1", 90, 210),
        decisions_by_read={},
        harmonizations_by_read={},
        report_sha256="a" * 64,
        minimum_posterior=0.0,
        command_line="fiberhmm-strand-rescue-annotate v6-test",
        quality_contract_version=6,
    )

    with pysam.AlignmentFile(output, "rb") as bam:
        comments = [str(value) for value in bam.header.to_dict().get("CO", [])]
    contract = next(
        value for value in comments if value.startswith(V6_HEADER_PREFIX)
    )
    assert "exact_selected_configuration_probability" in contract
    assert "h_q0=assignment_marginalized_canonical_geometry_probability" in contract
    assert "changed_edge_without_target_opportunity_q=0" in contract
    assert audit_bam(str(output))["contract_version"] == 6
    assert audit_bam(str(output))["valid"] is True


def test_v6_audit_accepts_declared_family_latent_configuration(tmp_path):
    source = tmp_path / "source.bam"
    baseline = tmp_path / "v6.bam"
    latent = tmp_path / "latent.bam"
    _make_bam(source)
    write_overlay_bam(
        str(source),
        baseline,
        region=("chr1", 90, 210),
        decisions_by_read={},
        harmonizations_by_read={},
        report_sha256="a" * 64,
        minimum_posterior=0.0,
        command_line="fiberhmm-strand-rescue-annotate v6-test",
        quality_contract_version=6,
    )
    family_contract = (
        "FIBERHMM-TF-FAMILY:v1:layer=tf_sr;quality_spec=QQQQQ;"
        "q3=fi_local_repeating_family_id_uint8;fi_zero=unassigned;"
        "fi_identity=reference_neighborhood_plus_fi;fi_reuse_separation_bp=24;"
        "q4=fq_conditional_family_assignment_confidence;"
        "fq_scale=round_255_times_confidence;"
        "fq_producer_calibration=declared_in_assignment_metadata;"
        f"assignments_sha256={'b' * 64};"
        "assignment_scope=complete_post_latent_tf_sr_fi_gt_0;"
        f"assignment_metadata_sha256={'c' * 64}"
    )
    latent_contract = (
        "FIBERHMM-LATENT-TILING:v1:layers=nuc_sr,tf_sr;"
        "semantics=complete_advisory_consensus_shadow;"
        "baseline_nuc_tf_unchanged=true;nuc_sr_structural_replacement=true;"
        "msp_baseline_not_refined=true;family_geometry_crossfit=false;"
        "v6_nuc_identity_cardinality_fixed_overridden_for_named_latent_actions=true;"
        "family_geometry_max_expansion_bp=6;"
        "nucleosome_length_prior_crossfit=false;"
        "molecule_efficiency_calibration_crossfit=false;"
        "validated_result_scope=CT_only_at_NAPA_family_021;GA_supported_actions=0;"
        "q0=map_configuration_posterior_within_family_group_uint8_endpoints_reserved;"
        "decision_confidence=weakest_split_identity_crossfit_component_in_actions_sidecar;"
        "q1=molecular_left_edge_confidence;q2=molecular_right_edge_confidence;"
        "edge_confidence=molecule_conditioned_within_family_configuration_marginal;"
        "map_geometry=maximum_posterior_configuration_within_family_group;"
        "map_geometry_share=recorded_in_actions_sidecar;"
        "named_actions=fhlt_token_optional_P_Rn_tf_and_fhlt_token_optional_P_Nn_H_nuc;"
        "propagated_name_marker=fhlt_token_P;"
        "amplification_family_policy=propagate_only_to_compatible_exact_baseline_state;"
        "incompatible_amplification_siblings=unchanged;"
        "ambiguous_molecules_unchanged=true"
    )
    with pysam.AlignmentFile(baseline, "rb") as input_bam:
        header = input_bam.header.to_dict()
        header["CO"] = list(header.get("CO", [])) + [family_contract, latent_contract]
        output_header = pysam.AlignmentHeader.from_dict(header)
        with pysam.AlignmentFile(latent, "wb", header=output_header) as output_bam:
            read = next(input_bam.fetch(until_eof=True))
            read.set_tag(
                "MA",
                (
                    "100;nuc.Q:11-20;msp.:31-30;tf.QQQ:70-10;"
                    "nuc_sr.QQQ:21-10;tf_sr.QQQQQ:11-8,70-10"
                ),
                value_type="Z",
            )
            read.set_tag(
                "AQ",
                array(
                    "B",
                    [
                        200, 150, 10, 20,
                        200, 170, 180,
                        200, 170, 180, 21, 230,
                        255, 0, 0, 0, 0,
                    ],
                ),
            )
            read.set_tag(
                "AN",
                (
                    ".,.,.,fhlt_1111111111111111_N0_H,"
                    "fhlt_1111111111111111_R0,."
                ),
                value_type="Z",
            )
            output_bam.write(read)
    pysam.index(str(latent))

    result = audit_bam(str(latent))

    assert result["valid"] is True
    assert result["latent_extension"] is True
    assert result["counts"]["latent_tiling_groups"] == 1


def test_v6_audit_accepts_declared_multifamily_latent_configuration(tmp_path):
    source = tmp_path / "source.bam"
    baseline = tmp_path / "v6.bam"
    multifamily = tmp_path / "multifamily.bam"
    _make_bam(source)
    write_overlay_bam(
        str(source),
        baseline,
        region=("chr1", 90, 210),
        decisions_by_read={},
        harmonizations_by_read={},
        report_sha256="a" * 64,
        minimum_posterior=0.0,
        command_line="fiberhmm-strand-rescue-annotate v6-test",
        quality_contract_version=6,
    )
    family_contract = (
        "FIBERHMM-TF-FAMILY:v1:layer=tf_sr;quality_spec=QQQQQ;"
        "q3=fi_local_repeating_family_id_uint8;fi_zero=unassigned;"
        "fi_identity=reference_neighborhood_plus_fi;fi_reuse_separation_bp=24;"
        "q4=fq_conditional_family_assignment_confidence;"
        "fq_scale=round_255_times_confidence;"
        "fq_producer_calibration=declared_in_assignment_metadata;"
        f"assignments_sha256={'b' * 64};"
        "assignment_scope=complete_post_multifamily_tf_sr_fi_gt_0;"
        f"assignment_metadata_sha256={'c' * 64}"
    )
    multifamily_contract = (
        "FIBERHMM-LATENT-TILING:v2:layers=nuc_sr,tf_sr;"
        "semantics=complete_advisory_multifamily_configuration;"
        "baseline_nuc_tf_unchanged=true;multiple_named_tf_segments=true;"
        "tf_roles=R0_through_Rn_zero_based_contiguous;"
        "nuc_roles=N0H_through_NnH_zero_based_contiguous;"
        "shared_decision_q0=false;q1q2=segment_specific_edge_marginals;"
        "named_actions=fhlt_token_optional_P_Rn_tf_and_fhlt_token_optional_P_Nn_H_nuc;"
        "family_slots=R0_21_R1_22;first_stage_geometry_and_q0_preserved=true;"
        "new_R1_and_replaced_N0H_q0=multifamily_map_configuration_posterior;"
        "equal_prior_hybrid_hypothesis_probability=actions_sidecar_only"
    )
    with pysam.AlignmentFile(baseline, "rb") as input_bam:
        header = input_bam.header.to_dict()
        header["CO"] = list(header.get("CO", [])) + [
            family_contract, multifamily_contract
        ]
        output_header = pysam.AlignmentHeader.from_dict(header)
        with pysam.AlignmentFile(multifamily, "wb", header=output_header) as output_bam:
            read = next(input_bam.fetch(until_eof=True))
            read.set_tag(
                "MA",
                (
                    "100;nuc.Q:11-20;msp.:31-30;tf.QQQ:70-10;"
                    "nuc_sr.QQQ:24-7;tf_sr.QQQQQ:11-8,19-5,70-10"
                ),
                value_type="Z",
            )
            read.set_tag(
                "AQ",
                array(
                    "B",
                    [
                        200, 150, 10, 20,
                        200, 170, 180,
                        150, 170, 180, 21, 230,
                        200, 165, 175, 22, 210,
                        255, 0, 0, 0, 0,
                    ],
                ),
            )
            read.set_tag(
                "AN",
                (
                    ".,.,.,fhlt_1111111111111111_N0_H,"
                    "fhlt_1111111111111111_R0,"
                    "fhlt_1111111111111111_R1,."
                ),
                value_type="Z",
            )
            output_bam.write(read)
    pysam.index(str(multifamily))

    result = audit_bam(str(multifamily))

    assert result["valid"] is True
    assert result["latent_extension"] is True
    assert result["counts"]["latent_tiling_groups"] == 1
    assert result["counts"]["family_id_21"] == 1
    assert result["counts"]["family_id_22"] == 1


def test_v6_audit_accepts_postfamily_nuc_sr_configuration(tmp_path):
    """v3 replaces a conservative nuc_sr, not a nearby ordinary nuc."""
    source = tmp_path / "source.bam"
    baseline = tmp_path / "v6.bam"
    postfamily = tmp_path / "postfamily.bam"
    _make_bam(source)
    write_overlay_bam(
        str(source),
        baseline,
        region=("chr1", 90, 210),
        decisions_by_read={},
        harmonizations_by_read={},
        report_sha256="a" * 64,
        minimum_posterior=0.0,
        command_line="fiberhmm-strand-rescue-annotate v6-test",
        quality_contract_version=6,
    )
    family_contract = (
        "FIBERHMM-TF-FAMILY:v1:layer=tf_sr;quality_spec=QQQQQ;"
        "q3=fi_local_repeating_family_id_uint8;fi_zero=unassigned;"
        "fi_identity=reference_neighborhood_plus_fi;fi_reuse_separation_bp=24;"
        "q4=fq_conditional_family_assignment_confidence;"
        "fq_scale=round_255_times_confidence;"
        "fq_producer_calibration=declared_in_assignment_metadata;"
        f"assignments_sha256={'b' * 64};"
        "assignment_scope=complete_postfamily_tf_sr_fi_gt_0;"
        f"assignment_metadata_sha256={'c' * 64}"
    )
    postfamily_contract = (
        "FIBERHMM-LATENT-TILING:v3:layers=nuc_sr,tf_sr;"
        "semantics=advisory_post_family_configuration;"
        "baseline_nuc_tf_unchanged=true;source_layer=nuc_sr;"
        "exactly_one_covering_nuc_sr_replaced=true;"
        "single_named_tf_segment=true;single_residual_nuc_segment=true;"
        "q0=equal_prior_weakest_grid_robust_hypothesis_probability_uint8_endpoints_reserved;"
        "v6_q0_semantics_overridden_for_named_postfamily_actions=true;"
        "q1q2=segment_specific_edge_marginals;"
        "named_actions=fhlt_token_optional_P_Rn_tf_and_fhlt_token_optional_P_Nn_H_nuc;"
        "selection=proxy_operating_point_plus_grid_stability;"
        "family_geometry_crossfit=opposite_assayed_strand;"
        "nucleosome_length_prior_crossfit=false;"
        "molecule_efficiency_calibration_crossfit=false;"
        "nomination=permissive_exact_family_assignment;"
        f"parent_bam_sha256={'d' * 64};actions_sha256={'e' * 64};"
        "materializer_grid=family_mass_0.95_linker_step_3_dyad_step_3;"
        "ambiguous_multifamily_blocks=unchanged"
    )
    with pysam.AlignmentFile(baseline, "rb") as input_bam:
        header = input_bam.header.to_dict()
        header["CO"] = list(header.get("CO", [])) + [
            family_contract, postfamily_contract
        ]
        output_header = pysam.AlignmentHeader.from_dict(header)
        with pysam.AlignmentFile(postfamily, "wb", header=output_header) as output_bam:
            read = next(input_bam.fetch(until_eof=True))
            # The v3 configuration is deliberately far from the ordinary nuc
            # to prove the audit does not apply the obsolete +/-6 bp rule.
            read.set_tag(
                "MA",
                (
                    "100;nuc.Q:11-20;msp.:31-30;tf.QQQ:70-10;"
                    "nuc_sr.QQQ:45-10;tf_sr.QQQQQ:35-10,70-10"
                ),
                value_type="Z",
            )
            read.set_tag(
                "AQ",
                array(
                    "B",
                    [
                        200, 150, 10, 20,
                        200, 170, 180,
                        200, 170, 180, 21, 230,
                        255, 0, 0, 0, 0,
                    ],
                ),
            )
            read.set_tag(
                "AN",
                (
                    ".,.,.,fhlt_1111111111111111_N0_H,"
                    "fhlt_1111111111111111_R0,."
                ),
                value_type="Z",
            )
            output_bam.write(read)
    pysam.index(str(postfamily))

    result = audit_bam(str(postfamily))

    assert result["valid"] is True
    assert result["latent_extension"] is True
    assert result["counts"]["latent_tiling_groups"] == 1


def test_v4_audit_rejects_rescue_overlapping_another_tf_group(tmp_path):
    source = tmp_path / "source.bam"
    valid = tmp_path / "valid.bam"
    invalid = tmp_path / "invalid.bam"
    _make_bam(source)
    write_overlay_bam(
        str(source),
        valid,
        region=("chr1", 90, 210),
        decisions_by_read={"read1": [_decision()]},
        harmonizations_by_read={},
        report_sha256="a" * 64,
        minimum_posterior=0.0,
        command_line="fiberhmm-strand-rescue-annotate test",
    )

    with pysam.AlignmentFile(valid, "rb") as input_bam:
        with pysam.AlignmentFile(invalid, "wb", header=input_bam.header) as output_bam:
            read = next(input_bam.fetch(until_eof=True))
            read.set_tag(
                "MA",
                read.get_tag("MA").replace(
                    "tf_sr.QQQ:36-10,51-10,70-10",
                    "tf_sr.QQQ:66-10,51-10,70-10",
                ),
                value_type="Z",
            )
            output_bam.write(read)
    pysam.index(str(invalid))

    audit = audit_bam(str(invalid))

    assert audit["valid"] is False
    assert any(
        error["message"] == "rescue overlaps another tf_sr group"
        for error in audit["errors"]
    )


def test_v5_streamed_overlay_is_ma_aq_an_equivalent_to_v4(tmp_path):
    source = tmp_path / "source.bam"
    legacy_output = tmp_path / "legacy.bam"
    streamed_output = tmp_path / "streamed.bam"
    _make_bam(source)
    nuc_geometry = replace(
        _geometry_decision(),
        decision_id="nuc-geometry",
        call_type="nuc",
        current_interval=(110, 130),
        canonical_interval=(108, 132),
        current_molecular_interval=(10, 20),
        current_annotation_ordinal=0,
    )
    write_overlay_bam(
        str(source),
        legacy_output,
        region=("chr1", 90, 210),
        decisions_by_read={"read1": [_decision()]},
        harmonizations_by_read={"read1": [_geometry_decision(), nuc_geometry]},
        report_sha256="a" * 64,
        minimum_posterior=0.0,
        command_line="fiberhmm-strand-rescue-annotate golden-v4",
    )
    source_read = _source_record(source)
    rescue_token = hashlib.sha256(
        b"sr:input.bam:rescue-decision"
    ).hexdigest()[:16]
    edge_token = hashlib.sha256(
        b"sr-edge:tf:input.bam:geometry-decision"
    ).hexdigest()[:16]
    nuc_edge_token = hashlib.sha256(
        b"sr-edge:nuc:input.bam:nuc-geometry"
    ).hexdigest()[:16]
    row = _v5_action_row(
        source_read,
        rescues=(
            {
                "token": rescue_token,
                "source_ordinal": 0,
                "source_interval": [30, 30],
                "q0": 178,
                "components": [
                    {"component_index": 0, "interval": [35, 10], "q1": 204, "q2": 153},
                    {"component_index": 1, "interval": [50, 10], "q1": 178, "q2": 230},
                ],
            },
        ),
        edge_updates=(
            {
                "token": nuc_edge_token,
                "call_type": "nuc",
                "source_ordinal": 0,
                "source_interval": [10, 20],
                "alternative_interval": [8, 24],
                "q": [166, 204, 230],
            },
            {
                "token": edge_token,
                "call_type": "tf",
                "source_ordinal": 0,
                "source_interval": [69, 10],
                "alternative_interval": [68, 12],
                "q": [166, 204, 230],
            },
        ),
    )
    stream = _make_v5_action_stream(tmp_path, source, [row])

    summary = write_streaming_overlay_bam(
        str(source),
        streamed_output,
        action_stream=stream,
        report_sha256="b" * 64,
        minimum_posterior=0.0,
        command_line="fiberhmm-strand-rescue-annotate golden-v5",
    )

    with pysam.AlignmentFile(legacy_output, "rb") as legacy_bam, pysam.AlignmentFile(
        streamed_output, "rb"
    ) as streamed_bam:
        legacy = next(legacy_bam.fetch(until_eof=True))
        streamed = next(streamed_bam.fetch(until_eof=True))
        assert streamed.to_string().split("\t")[:11] == legacy.to_string().split(
            "\t"
        )[:11]
        assert {
            key: value
            for key, value in streamed.get_tags()
            if key not in {"MA", "AQ", "AN"}
        } == {
            key: value
            for key, value in legacy.get_tags()
            if key not in {"MA", "AQ", "AN"}
        }
        assert streamed.get_tag("MA") == legacy.get_tag("MA")
        assert list(streamed.get_tag("AQ")) == list(legacy.get_tag("AQ"))
        assert streamed.get_tag("AN") == legacy.get_tag("AN")
        comments = streamed_bam.header.to_dict().get("CO", [])
        assert any(stream.manifest["bgzf_sha256"] in value for value in comments)
        assert any(stream.manifest["jsonl_sha256"] in value for value in comments)
    assert summary["reads_with_actions"] == 1
    assert summary["rescue_decisions_materialized"] == 1
    assert summary["tf_edge_updates_materialized"] == 1
    assert summary["nuc_edge_updates_materialized"] == 1
    assert summary["action_stream"]["validation"]["fetch_record_count"] == 1
    assert audit_bam(str(streamed_output))["valid"] is True


def test_v5_reverse_rescue_preserves_configuration_indices_and_final_q_orientation(
    tmp_path,
):
    source = tmp_path / "reverse.bam"
    legacy_output = tmp_path / "reverse.legacy.bam"
    streamed_output = tmp_path / "reverse.streamed.bam"
    _make_bam(source, ma="100;msp.:41-30", aq=(), flag=16)
    decision = replace(
        _decision(),
        current_molecular_interval=(40, 30),
        current_annotation_ordinal=0,
    )
    write_overlay_bam(
        str(source),
        legacy_output,
        region=("chr1", 90, 210),
        decisions_by_read={"read1": [decision]},
        harmonizations_by_read={},
        report_sha256="a" * 64,
        minimum_posterior=0.0,
        command_line="golden reverse v4",
    )
    token = hashlib.sha256(b"sr:input.bam:rescue-decision").hexdigest()[:16]
    row = _v5_action_row(
        _source_record(source),
        rescues=(
            {
                "token": token,
                "source_ordinal": 0,
                "source_interval": [40, 30],
                "q0": 178,
                "components": [
                    {"component_index": 0, "interval": [55, 10], "q1": 153, "q2": 204},
                    {"component_index": 1, "interval": [40, 10], "q1": 230, "q2": 178},
                ],
            },
        ),
    )
    stream = _make_v5_action_stream(tmp_path, source, [row])
    write_streaming_overlay_bam(
        str(source),
        streamed_output,
        action_stream=stream,
        report_sha256="b" * 64,
        minimum_posterior=0.0,
        command_line="golden reverse v5",
    )

    with pysam.AlignmentFile(legacy_output, "rb") as legacy_bam, pysam.AlignmentFile(
        streamed_output, "rb"
    ) as streamed_bam:
        legacy = next(legacy_bam.fetch(until_eof=True))
        streamed = next(streamed_bam.fetch(until_eof=True))
    assert streamed.get_tag("MA") == legacy.get_tag("MA")
    assert list(streamed.get_tag("AQ")) == list(legacy.get_tag("AQ"))
    assert streamed.get_tag("AN") == legacy.get_tag("AN")
    names = parse_an_tag(streamed.get_tag("AN"))
    rows = _quality_rows(streamed)
    named_rows = {name.rsplit("_", 1)[1]: tuple(q) for name, q in zip(names, rows) if name}
    assert named_rows == {
        "R1": (178, 230, 178),
        "R0": (178, 153, 204),
    }


def test_v5_zero_action_cli_stream_synthesizes_baseline_layers(tmp_path, capsys):
    source = tmp_path / "source.bam"
    output = tmp_path / "output.bam"
    report_path = tmp_path / "report.json"
    _make_bam(
        source,
        ma=(
            "100;nuc.Q:11-20;msp.:31-30;tf.QQQ:70-10;"
            "nuc_sr.QQQ:1-5;tf_sr.QQQ:2-5"
        ),
        aq=(200, 150, 10, 20, 1, 2, 3, 4, 5, 6),
    )
    stream = _make_v5_action_stream(tmp_path, source, [])
    report_path.write_text(json.dumps(_v5_report(source, stream), sort_keys=True))

    assert main(["--report", str(report_path), "--output", str(output)]) == 0

    emitted = json.loads(capsys.readouterr().out)
    assert emitted["streamed_actions"] is True
    assert emitted["decisions"] == 0
    with pysam.AlignmentFile(output, "rb") as bam:
        read = next(bam.fetch(until_eof=True))
    assert _raw_groups(read)["nuc_sr"] == ("QQQ", [(10, 20)])
    assert _raw_groups(read)["tf_sr"] == ("QQQ", [(69, 10)])


def _make_three_record_bam(path: Path):
    header = pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": "chr1", "LN": 1000}],
        }
    )
    with pysam.AlignmentFile(path, "wb", header=header) as bam:
        for index, flag in enumerate((256, 2048, 0)):
            read = pysam.AlignedSegment(header)
            read.query_name = "repeated"
            read.query_sequence = "A" * 100
            read.flag = flag
            read.reference_id = 0
            read.reference_start = 100 + index
            read.mapping_quality = 0 if index < 2 else 60
            read.cigarstring = "100M"
            read.query_qualities = pysam.qualitystring_to_array("I" * 100)
            read.set_tag("MA", "100;msp.:31-30", value_type="Z")
            read.set_tag("AQ", array("B"))
            bam.write(read)
    pysam.index(str(path))


def test_v5_ordinal_counts_secondary_and_supplementary_records(tmp_path):
    source = tmp_path / "three.bam"
    output = tmp_path / "three.overlay.bam"
    _make_three_record_bam(source)
    token = "1" * 16
    action = _v5_action_row(
        _source_record(source, ordinal=2),
        ordinal=2,
        rescues=(
            {
                "token": token,
                "source_ordinal": 0,
                "source_interval": [30, 30],
                "q0": 1,
                "components": [
                    {"component_index": 0, "interval": [35, 10], "q1": 2, "q2": 3}
                ],
            },
        ),
    )
    stream = _make_v5_action_stream(
        tmp_path, source, [action], fetch_record_count=3
    )

    summary = write_streaming_overlay_bam(
        str(source),
        output,
        action_stream=stream,
        report_sha256="c" * 64,
        minimum_posterior=0.0,
        command_line="ordinal test",
    )

    with pysam.AlignmentFile(output, "rb") as bam:
        reads = list(bam.fetch("chr1", 90, 210))
    assert len(reads) == 3
    assert not reads[0].has_tag("AN")
    assert not reads[1].has_tag("AN")
    assert "_R0" in reads[2].get_tag("AN")
    assert summary["reads_written"] == 3
    assert summary["reads_with_actions"] == 1


def test_v5_wrong_source_interval_fails_without_publishing_output(tmp_path):
    source = tmp_path / "source.bam"
    output = tmp_path / "must-not-exist.bam"
    _make_bam(source)
    action = _v5_action_row(
        _source_record(source),
        rescues=(
            {
                "token": "2" * 16,
                "source_ordinal": 0,
                "source_interval": [29, 31],
                "q0": 200,
                "components": [
                    {"component_index": 0, "interval": [35, 10], "q1": 200, "q2": 200}
                ],
            },
        ),
    )
    stream = _make_v5_action_stream(tmp_path, source, [action])

    with pytest.raises(ValueError, match="source MSP interval differs"):
        write_streaming_overlay_bam(
            str(source),
            output,
            action_stream=stream,
            report_sha256="d" * 64,
            minimum_posterior=0.0,
            command_line="bad source",
        )

    assert not output.exists()
    assert not Path(f"{output}.bai").exists()


def test_v5_manifest_rejects_inline_mix_and_unsafe_sidecar_path(tmp_path):
    source = tmp_path / "source.bam"
    report_path = tmp_path / "report.json"
    _make_bam(source)
    stream = _make_v5_action_stream(tmp_path, source, [])
    report = _v5_report(source, stream)
    assert _validate_v5_action_storage(report, report_path)[0].input_index == 0

    mixed = json.loads(json.dumps(report))
    mixed["strand_rescue"]["decisions"] = []
    with pytest.raises(ValueError, match="mixes streamed and inline"):
        _validate_v5_action_storage(mixed, report_path)

    unsafe = json.loads(json.dumps(report))
    unsafe["strand_rescue"]["action_storage"]["streams"][0]["path"] = (
        "../unsafe.jsonl.bgz"
    )
    with pytest.raises(ValueError, match="must not be absolute or contain"):
        _validate_v5_action_storage(unsafe, report_path)


def test_v5_sidecar_checksum_and_out_of_order_rows_are_hard_errors(tmp_path):
    source = tmp_path / "source.bam"
    output = tmp_path / "output.bam"
    _make_bam(source)
    read = _source_record(source)
    first = _v5_action_row(
        read,
        ordinal=1,
        rescues=(
            {
                "token": "3" * 16,
                "source_ordinal": 0,
                "source_interval": [30, 30],
                "q0": 1,
                "components": [
                    {"component_index": 0, "interval": [35, 5], "q1": 1, "q2": 1}
                ],
            },
        ),
    )
    second = _v5_action_row(
        read,
        ordinal=0,
        rescues=(
            {
                "token": "4" * 16,
                "source_ordinal": 0,
                "source_interval": [30, 30],
                "q0": 1,
                "components": [
                    {"component_index": 0, "interval": [45, 5], "q1": 1, "q2": 1}
                ],
            },
        ),
    )
    stream = _make_v5_action_stream(
        tmp_path, source, [first, second], fetch_record_count=2
    )
    with V5ActionStreamReader(stream.bgzf_path, stream.manifest) as reader:
        with pytest.raises(ValueError, match="strictly increasing"):
            reader.pop_for_ordinal(1)

    corrupt_manifest = dict(stream.manifest)
    corrupt_manifest["bgzf_sha256"] = "f" * 64
    corrupt = replace(stream, manifest=corrupt_manifest)
    with pytest.raises(ValueError, match="BGZF SHA-256 mismatch"):
        write_streaming_overlay_bam(
            str(source),
            output,
            action_stream=corrupt,
            report_sha256="e" * 64,
            minimum_posterior=0.0,
            command_line="checksum test",
        )
    assert not output.exists()


def test_v5_threshold_uses_q0_byte_and_allows_two_alternatives_from_one_msp(
    tmp_path,
):
    source = tmp_path / "source.bam"
    output = tmp_path / "threshold.bam"
    _make_bam(source)
    action = _v5_action_row(
        _source_record(source),
        rescues=(
            {
                "token": "5" * 16,
                "source_ordinal": 0,
                "source_interval": [30, 30],
                "q0": 127,
                "components": [
                    {"component_index": 0, "interval": [35, 5], "q1": 1, "q2": 2}
                ],
            },
            {
                "token": "6" * 16,
                "source_ordinal": 0,
                "source_interval": [30, 30],
                "q0": 128,
                "components": [
                    {"component_index": 0, "interval": [50, 5], "q1": 3, "q2": 4}
                ],
            },
        ),
    )
    stream = _make_v5_action_stream(tmp_path, source, [action])

    summary = write_streaming_overlay_bam(
        str(source),
        output,
        action_stream=stream,
        report_sha256="f" * 64,
        minimum_posterior=0.5,
        command_line="threshold test",
    )

    with pysam.AlignmentFile(output, "rb") as bam:
        read = next(bam.fetch(until_eof=True))
    names = [name for name in parse_an_tag(read.get_tag("AN")) if name]
    assert names == [f"fhsr_{'6' * 16}_R0"]
    assert summary["rescue_decisions_available"] == 2
    assert summary["rescue_decisions_materialized"] == 1
    assert summary["rescue_decisions_thresholded"] == 1


def test_v5_action_on_record_without_ma_is_a_hard_error(tmp_path):
    source = tmp_path / "no-ma.bam"
    output = tmp_path / "no-ma.overlay.bam"
    _make_bam(source, ma=None, aq=())
    action = _v5_action_row(
        _source_record(source),
        rescues=(
            {
                "token": "7" * 16,
                "source_ordinal": 0,
                "source_interval": [30, 30],
                "q0": 200,
                "components": [
                    {"component_index": 0, "interval": [35, 5], "q1": 1, "q2": 1}
                ],
            },
        ),
    )
    stream = _make_v5_action_stream(tmp_path, source, [action])

    with pytest.raises(ValueError, match="no MA tag"):
        write_streaming_overlay_bam(
            str(source),
            output,
            action_stream=stream,
            report_sha256="8" * 64,
            minimum_posterior=0.0,
            command_line="no MA test",
        )
    assert not output.exists()


def test_v5_matching_checksum_still_rejects_missing_bgzf_eof(tmp_path):
    source = tmp_path / "source.bam"
    output = tmp_path / "truncated.overlay.bam"
    _make_bam(source)
    stream = _make_v5_action_stream(tmp_path, source, [])
    stream.bgzf_path.write_bytes(stream.bgzf_path.read_bytes()[:-28])
    manifest = dict(stream.manifest)
    manifest["compressed_size_bytes"] = stream.bgzf_path.stat().st_size
    manifest["bgzf_sha256"] = _sha256_file(stream.bgzf_path)
    truncated = replace(stream, manifest=manifest)

    with pytest.raises(ValueError, match="no standard EOF block"):
        write_streaming_overlay_bam(
            str(source),
            output,
            action_stream=truncated,
            report_sha256="9" * 64,
            minimum_posterior=0.0,
            command_line="missing EOF test",
        )
    assert not output.exists()


def test_v5_multi_input_cli_selects_one_stream_and_resets_ordinal(tmp_path, capsys):
    first_source = tmp_path / "first.bam"
    second_source = tmp_path / "second.bam"
    report_path = tmp_path / "multi.report.json"
    output = tmp_path / "selected.bam"
    _make_bam(first_source)
    _make_bam(second_source)
    first_stream = _make_v5_action_stream(
        tmp_path, first_source, [], input_index=0
    )
    action = _v5_action_row(
        _source_record(second_source),
        ordinal=0,
        rescues=(
            {
                "token": "a" * 16,
                "source_ordinal": 0,
                "source_interval": [30, 30],
                "q0": 200,
                "components": [
                    {"component_index": 0, "interval": [35, 5], "q1": 2, "q2": 3}
                ],
            },
        ),
    )
    second_stream = _make_v5_action_stream(
        tmp_path, second_source, [action], input_index=1
    )
    report = _v5_report(first_source, first_stream)
    second_stat = second_source.stat()
    report["input"]["bams"].append(str(second_source.resolve()))
    report["input"]["files"].append(
        {
            "path": str(second_source.resolve()),
            "size_bytes": second_stat.st_size,
            "mtime_ns": second_stat.st_mtime_ns,
        }
    )
    storage = report["strand_rescue"]["action_storage"]
    storage["streams"].append(dict(second_stream.manifest))
    for total_key, manifest_key in (
        ("fetch_records", "fetch_record_count"),
        ("action_records", "action_record_count"),
        ("rescue_decisions", "rescue_decision_count"),
        ("rescue_components", "rescue_component_count"),
        ("tf_edge_updates", "tf_edge_update_count"),
        ("nuc_edge_updates", "nuc_edge_update_count"),
    ):
        storage["totals"][total_key] += second_stream.manifest[manifest_key]
    report_path.write_text(json.dumps(report, sort_keys=True))

    assert (
        main(
            [
                "--report",
                str(report_path),
                "--bam",
                str(second_source),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    emitted = json.loads(capsys.readouterr().out)
    assert len(emitted["outputs"]) == 1
    assert emitted["outputs"][0]["input_index"] == 1
    with pysam.AlignmentFile(output, "rb") as bam:
        read = next(bam.fetch(until_eof=True))
        comments = bam.header.to_dict().get("CO", [])
    assert f"fhsr_{'a' * 16}_R0" in parse_an_tag(read.get_tag("AN"))
    assert any("action_input_index=1" in value for value in comments)


def test_generator_v5_writer_to_real_bam_annotation_smoke(tmp_path):
    from fiberhmm.cli.strand_rescue import _StagingRegistry, _V5ActionWriter

    source = tmp_path / "source.bam"
    report_path = tmp_path / "generated.report.json"
    output = tmp_path / "generated.overlay.bam"
    _make_bam(source)
    source_read = _source_record(source)
    record_sha256 = hashlib.sha256(
        source_read.to_string().encode("utf-8")
    ).hexdigest()
    rescue = {
        "token": "b" * 16,
        "source_ordinal": 0,
        "source_interval": [30, 30],
        "q0": 173,
        "components": [
            {"component_index": 0, "interval": [35, 5], "q1": 201, "q2": 230}
        ],
    }
    registry = _StagingRegistry()
    writer = _V5ActionWriter(
        report_path,
        input_index=0,
        loaded_region=["chr1", 90, 210],
        registry=registry,
    )
    writer.write_actions(
        0,
        SimpleNamespace(name=source_read.query_name, record_sha256=record_sha256),
        [rescue],
        [],
    )
    manifest = writer.finish(fetch_record_count=1)
    registry.cleanup()
    stream = ResolvedV5ActionStream(
        input_index=0,
        input_path=str(source.resolve()),
        manifest=manifest,
        bgzf_path=tmp_path / manifest["path"],
        gzi_path=tmp_path / manifest["gzi_path"],
    )

    summary = write_streaming_overlay_bam(
        str(source),
        output,
        action_stream=stream,
        report_sha256="c" * 64,
        minimum_posterior=0.0,
        command_line="generator-to-annotator smoke",
    )

    with pysam.AlignmentFile(output, "rb") as bam:
        output_read = next(bam.fetch(until_eof=True))
    assert f"fhsr_{'b' * 16}_R0" in parse_an_tag(output_read.get_tag("AN"))
    assert summary["action_stream"]["validation"]["jsonl_sha256"] == manifest[
        "jsonl_sha256"
    ]
    assert audit_bam(str(output))["valid"] is True
