import csv
from array import array

import pysam
import pytest

from fiberhmm.cli.tag_families import (
    ASSIGNMENT_FIELDS,
    ASSIGNMENT_FIELDS_V2,
    FAMILY_HEADER_PREFIX,
    load_family_assignments,
    tag_tf_families,
)
from fiberhmm.inference.tf_family_ids import (
    TFFamilyInterval,
    allocate_repeating_family_ids,
)
from fiberhmm.io.ma_tags import parse_aq_array, parse_ma_tag


def test_repeating_family_ids_wrap_and_avoid_active_overlap():
    families = [
        TFFamilyInterval(f"f{i:03d}", "chr1", i * 10, i * 10 + 5)
        for i in range(260)
    ]
    slots = allocate_repeating_family_ids(families)
    assert [slots[f"f{i:03d}"] for i in range(260)] == list(range(1, 256)) + list(range(1, 6))

    separated = [
        TFFamilyInterval("a", "chr3", 0, 10),
        TFFamilyInterval("b", "chr3", 10, 20),
    ]
    assert allocate_repeating_family_ids(separated, maximum_id=1, separation_bp=0) == {
        "a": 1, "b": 1,
    }
    with pytest.raises(ValueError, match="more than 1"):
        allocate_repeating_family_ids(separated, maximum_id=1)

    overlapping = [
        TFFamilyInterval(f"o{i}", "chr2", i, 1000)
        for i in range(4)
    ]
    overlap_slots = allocate_repeating_family_ids(overlapping, maximum_id=4)
    assert len(set(overlap_slots.values())) == 4
    with pytest.raises(ValueError, match="more than 3"):
        allocate_repeating_family_ids(overlapping, maximum_id=3)


def _make_input(path, *, declare_tf_sr=True, reverse=False, tf_sr_spec="QQQ"):
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": 1000}],
        "CO": [
            "MA-TYPES:v1:nuc,tf,nuc_sr,tf_sr",
            "FIBERHMM-STRAND-RESCUE:v6:test-contract",
        ] if declare_tf_sr else ["FIBERHMM-STRAND-RESCUE:v6:test-contract"],
    }
    with pysam.AlignmentFile(path, "wb", header=header) as bam:
        read = pysam.AlignedSegment(bam.header)
        read.query_name = "read1"
        read.query_sequence = "A" * 100
        read.flag = 16 if reverse else 0
        read.reference_id = 0
        read.reference_start = 100
        read.mapping_quality = 60
        read.cigarstring = "100M"
        read.query_qualities = pysam.qualitystring_to_array("I" * 100)
        read.set_tag(
            "MA",
            f"100;nuc.Q:1-20;tf.QQQ:31-8;nuc_sr.QQQ:1-20;tf_sr.{tf_sr_spec}:31-8,51-10",
            value_type="Z",
        )
        tf_sr_rows = (
            [255, 0, 0, 200, 10, 20]
            if tf_sr_spec == "QQQ"
            else [255, 0, 0, 9, 9, 200, 10, 20, 9, 9]
        )
        read.set_tag("AQ", array("B", [200, 90, 80, 70, 255, 0, 0] + tf_sr_rows))
        bam.write(read)
    pysam.index(str(path))


def _write_assignments(path, *, call_start=150, call_end=160):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ASSIGNMENT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerow({
            "read_name": "read1",
            "alignment_occurrence": 0,
            "tf_sr_ordinal": 1,
            "family_id": 7,
            "assignment_probability": 0.8,
            "family_key": "family-seven",
            "contig": "chr1",
            "call_start": call_start,
            "call_end": call_end,
        })


def test_assignment_v2_records_calibration_scope(tmp_path):
    assignments = tmp_path / "assignments.v2.tsv"
    with assignments.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ASSIGNMENT_FIELDS_V2, delimiter="\t")
        writer.writeheader()
        writer.writerow({
            "read_name": "read1",
            "alignment_occurrence": 0,
            "tf_sr_ordinal": 1,
            "family_id": 7,
            "assignment_probability": 0.8,
            "family_key": "family-seven",
            "contig": "chr1",
            "call_start": 150,
            "call_end": 160,
            "calibration_scope": "equal_prior_named_family_identity_probability",
        })

    assignment = next(iter(load_family_assignments(assignments).values()))
    assert assignment.calibration_scope == "equal_prior_named_family_identity_probability"


def test_tag_tf_families_appends_fi_fq_and_preserves_intervals(tmp_path):
    source = tmp_path / "source.bam"
    output = tmp_path / "family.bam"
    assignments = tmp_path / "assignments.tsv"
    _make_input(source)
    _write_assignments(assignments)

    summary = tag_tf_families(source, output, assignments)
    assert summary["assigned_tf_sr"] == 1
    assert output.with_suffix(".bam.bai").is_file()

    with pysam.AlignmentFile(output, "rb") as bam:
        read = next(bam.fetch(until_eof=True))
        comments = [str(value) for value in bam.header.to_dict().get("CO", [])]
    parsed = parse_ma_tag(read.get_tag("MA"))
    tf_sr = next(group for group in parsed["raw_types"] if group[0] == "tf_sr")
    assert tf_sr[3] == [(30, 8), (50, 10)]
    assert tf_sr[2] == "QQQQQ"
    rows = parse_aq_array(
        read.get_tag("AQ"),
        [group[2] for group in parsed["raw_types"]],
        [len(group[3]) for group in parsed["raw_types"]],
    )
    assert rows[-2] == [255, 0, 0, 0, 0]
    assert rows[-1] == [200, 10, 20, 7, 204]
    assert any(value.startswith(FAMILY_HEADER_PREFIX) for value in comments)


def test_tag_tf_families_requires_declared_tf_sr_prerequisite(tmp_path):
    source = tmp_path / "source.bam"
    output = tmp_path / "family.bam"
    assignments = tmp_path / "assignments.tsv"
    _make_input(source, declare_tf_sr=False)
    _write_assignments(assignments)

    with pytest.raises(ValueError, match="does not declare the prerequisite tf_sr"):
        tag_tf_families(source, output, assignments)
    assert not output.exists()


def test_tag_tf_families_projects_reverse_molecular_interval(tmp_path):
    source = tmp_path / "reverse.bam"
    output = tmp_path / "reverse.family.bam"
    assignments = tmp_path / "reverse.assignments.tsv"
    _make_input(source, reverse=True)
    _write_assignments(assignments, call_start=140, call_end=150)

    tag_tf_families(source, output, assignments)
    with pysam.AlignmentFile(output, "rb") as bam:
        read = next(bam.fetch(until_eof=True))
    parsed = parse_ma_tag(read.get_tag("MA"))
    tf_sr = next(group for group in parsed["raw_types"] if group[0] == "tf_sr")
    assert tf_sr[3] == [(30, 8), (50, 10)]


def test_tag_tf_families_rejects_uncontracted_qqqqq_legacy_bytes(tmp_path):
    source = tmp_path / "legacy.bam"
    output = tmp_path / "legacy.family.bam"
    assignments = tmp_path / "legacy.assignments.tsv"
    _make_input(source, tf_sr_spec="QQQQQ")
    _write_assignments(assignments)

    with pytest.raises(ValueError, match="cannot be overwritten as fi/fq"):
        tag_tf_families(source, output, assignments)
