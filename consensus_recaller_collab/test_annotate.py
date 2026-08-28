from __future__ import annotations

from array import array
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pysam
import pytest

from consensus_recaller_collab.annotate import (
    OverlayProposal,
    PairedDecision,
    add_overlay_groups,
    add_paired_overlay_groups,
    collect_overlay_proposals,
    collect_paired_decisions,
    paired_quality_rows,
    posterior_to_q,
    reference_interval_to_molecular,
    write_overlay_bam,
)
from consensus_recaller_collab.audit_paired_bam import audit_bam
from fiberhmm.io.bam_header import declared_ma_types
from fiberhmm.io.ma_tags import parse_aq_array, parse_ma_tag


class StubRead:
    def __init__(self, *, reverse=False):
        self.query_name = "read1"
        self.query_length = 100
        self.query_sequence = "A" * 100
        self.is_reverse = reverse
        self._tags = {
            "MA": "100;nuc.Q:11-40;tf.QQQ:70-10",
            "AQ": array("B", [200, 150, 10, 20]),
        }

    def get_reference_positions(self, full_length=False):
        assert full_length
        return list(range(100, 200))

    def has_tag(self, name):
        return name in self._tags

    def get_tag(self, name):
        if name not in self._tags:
            raise KeyError(name)
        return self._tags[name]

    def set_tag(self, name, value, value_type=None):
        if value is None:
            self._tags.pop(name, None)
        else:
            self._tags[name] = value


def proposal(pass_name="cr", *, tier="strong"):
    return OverlayProposal(
        pass_name=pass_name,
        proposal_id=f"{pass_name}-proposal",
        read_name="read1",
        library_id="input.bam",
        tier=tier,
        current_interval=(110, 150),
        tf_intervals=((110, 125), (130, 150)),
        tf_posterior=0.96,
        nuc_posterior=0.04,
    )


def paired_decision(pass_name="cr", *, current_state="N"):
    return PairedDecision(
        pass_name=pass_name,
        proposal_id=f"{pass_name}-paired-decision",
        read_name="read1",
        library_id="input.bam",
        tier="retain_n" if pass_name == "cr" else "retain_current",
        current_state=current_state,
        current_interval=(110, 150),
        tf_intervals=((110, 125), (130, 150)),
        tf_posterior=0.70,
        current_posterior=0.30,
        configuration_posterior=0.80,
        molecule_probability=0.60,
        population_probability=0.75,
        specificity_probability=0.90,
    )


def quality_rows(parsed, aq):
    return parse_aq_array(
        aq,
        [raw[2] for raw in parsed["raw_types"]],
        [len(raw[3]) for raw in parsed["raw_types"]],
    )


def test_posterior_q_is_linear_probability_not_phred():
    assert posterior_to_q(0.0) == 0
    assert posterior_to_q(0.5) == 128
    assert posterior_to_q(0.95) == 242
    assert posterior_to_q(1.0) == 255


def test_paired_quality_rows_have_complementary_state_bytes():
    current, tf = paired_quality_rows(paired_decision())
    assert len(current) == len(tf) == 5
    assert current[0] + tf[0] == 255
    assert current[1:] == tf[1:] == (204, 153, 191, 230)


def test_paired_cr_groups_keep_n_and_tf_alternatives_linked():
    read = StubRead()
    counts = add_paired_overlay_groups(
        read, [paired_decision()], active_passes=["cr"]
    )
    parsed = parse_ma_tag(read.get_tag("MA"))
    raw = {
        name: (quality_spec, intervals)
        for name, _strand, quality_spec, intervals in parsed["raw_types"]
    }
    assert raw["nuc_cr"] == ("QQQQQ", [(10, 40)])
    assert raw["tf_cr"] == (
        "QQQQQ", [(10, 15), (30, 20), (69, 10)]
    )
    assert counts == {"nuc_cr": 1, "tf_cr": 3}

    rows = quality_rows(parsed, read.get_tag("AQ"))
    names = read.get_tag("AN").split(",")
    assert len(names) == len(rows) == 6
    nuc_name = names[2]
    tf_names = names[3:5]
    assert nuc_name.startswith("fhcr_") and nuc_name.endswith("_N")
    prefix = nuc_name[:-2]
    assert tf_names == [f"{prefix}_T0", f"{prefix}_T1"]
    assert rows[2][0] + rows[3][0] == 255
    assert rows[2][1:] == rows[3][1:] == rows[4][1:]
    # Unnamed original/baseline annotations use positional empty fields, not
    # the non-spec placeholder used by the legacy selected-state writer.
    assert names[:2] == ["", ""]
    assert names[-1] == ""
    assert "." not in names


def test_paired_sr_accessible_target_is_one_sided_and_keeps_msp():
    read = StubRead()
    read.set_tag("MA", "100;msp.:11-40")
    read.set_tag("AQ", array("B"))
    counts = add_paired_overlay_groups(
        read,
        [paired_decision("sr", current_state="A")],
        active_passes=["sr"],
    )
    parsed = parse_ma_tag(read.get_tag("MA"))
    raw = {
        name: (quality_spec, intervals)
        for name, _strand, quality_spec, intervals in parsed["raw_types"]
    }
    assert raw["msp"] == ("", [(10, 40)])
    assert "nuc_sr" not in raw
    assert raw["tf_sr"] == ("QQQQQ", [(10, 15), (30, 20)])
    assert counts == {"tf_sr": 2}
    names = read.get_tag("AN").split(",")[1:]
    assert all(name.startswith("fhsr_") for name in names)
    assert [name.rsplit("_", 1)[1] for name in names] == ["AT0", "AT1"]


def test_collect_paired_decisions_includes_low_posterior_cr_and_sr_states():
    cr_state = {
        "decision_id": "cr-low",
        "read": "cr-read",
        "library_id": "/x.bam",
        "proposal_tier": "retain_n",
        "current_interval": [10, 110],
        "replacement_intervals": [[10, 30], [60, 90]],
        "nuc_posterior": 0.8,
        "complex_posterior": 0.2,
        "best_decomposition_posterior_given_complex": 0.6,
        "best_tf_integrated_raw_log_bf_vs_n": -1.0,
        "local_complex_prior": {"selected_complex_prior": 0.4},
        "boundary_control_log_bf": 2.0,
    }
    sr_state = {
        "decision_id": "sr-low",
        "read": "sr-read",
        "library_id": "/x.bam",
        "proposal_tier": "retain_current",
        "current": "N",
        "current_interval": [20, 120],
        "proposed_site_intervals": [[30, 50]],
        "posterior": 0.25,
        "current_posterior": 0.75,
        "best_configuration_posterior_given_tf": 0.7,
        "log_bf_vs_current": -0.5,
        "tf_prior_probability_vs_current": 0.8,
        "source_support_fraction": 0.9,
    }
    report = {
        "parameters": {"production_nuc_prior_odds": 10.0},
        "composite_deconvolution": {
            "enabled": True,
            "production_scenario_key": "10.0",
            "scenarios": {"10.0": {"candidate_states": [cr_state]}},
        },
        "strand_rescue": {
            "enabled": True,
            "windows": [{
                "cross_strand": {
                    "REV": {
                        "scenarios": {"10.0": {"candidate_states": [sr_state]}}
                    }
                }
            }],
        },
    }
    decisions = collect_paired_decisions(report)
    assert [(item.pass_name, item.proposal_id) for item in decisions] == [
        ("cr", "cr-low"), ("sr", "sr-low")
    ]
    assert [item.tf_posterior for item in decisions] == [0.2, 0.25]


def test_collect_paired_rejects_two_decisions_for_one_current_nuc():
    first = {
        "decision_id": "one",
        "read": "same-read",
        "library_id": "/x.bam",
        "current": "N",
        "current_interval": [20, 120],
        "proposed_site_intervals": [[30, 50]],
        "posterior": 0.25,
        "current_posterior": 0.75,
    }
    second = {
        **first,
        "decision_id": "two",
        "proposed_site_intervals": [[80, 100]],
    }
    report = {
        "parameters": {"production_nuc_prior_odds": 1.0},
        "composite_deconvolution": {"enabled": False},
        "strand_rescue": {
            "enabled": True,
            "windows": [{
                "cross_strand": {
                    "REV": {
                        "scenarios": {
                            "1.0": {"candidate_states": [first, second]}
                        }
                    }
                }
            }],
        },
    }
    with pytest.raises(ValueError, match="multiple paired decisions"):
        collect_paired_decisions(report)


def test_add_cr_groups_preserves_original_ma_and_quality_rows():
    read = StubRead()
    counts = add_overlay_groups(read, [proposal()])
    parsed = parse_ma_tag(read.get_tag("MA"))
    raw = {name: intervals for name, _, _, intervals in parsed["raw_types"]}
    assert raw["nuc"] == [(10, 40)]
    assert raw["tf"] == [(69, 10)]
    assert "nuc_cr" not in raw
    assert raw["tf_cr"] == [(10, 15), (30, 20), (69, 10)]
    assert counts == {"tf_cr": 3}
    assert quality_rows(parsed, read.get_tag("AQ")) == [
        [200], [150, 10, 20], [245], [245], [255]
    ]


def test_retained_nuc_and_baseline_tf_form_complete_shadow_callset():
    read = StubRead()
    retained = OverlayProposal(
        **{
            **proposal().__dict__,
            "proposal_id": "retained",
            "replacement_selected": False,
            "tf_posterior": 0.1,
            "nuc_posterior": 0.9,
        }
    )
    counts = add_overlay_groups(read, [retained])
    parsed = parse_ma_tag(read.get_tag("MA"))
    raw = {name: intervals for name, _, _, intervals in parsed["raw_types"]}
    assert raw["nuc_cr"] == [(10, 40)]
    assert raw["tf_cr"] == [(69, 10)]
    assert counts == {"nuc_cr": 1, "tf_cr": 1}
    assert quality_rows(parsed, read.get_tag("AQ"))[-2:] == [[230], [255]]


def test_active_pass_copies_complete_baseline_even_without_focal_decision():
    read = StubRead()
    counts = add_overlay_groups(read, [], active_passes=["cr"])
    parsed = parse_ma_tag(read.get_tag("MA"))
    raw = {name: intervals for name, _, _, intervals in parsed["raw_types"]}
    assert raw["nuc_cr"] == [(10, 40)]
    assert raw["tf_cr"] == [(69, 10)]
    assert counts == {"nuc_cr": 1, "tf_cr": 1}
    assert quality_rows(parsed, read.get_tag("AQ"))[-2:] == [[255], [255]]


def test_baseline_tf_overwrites_conflicting_nuc_in_shadow_callset():
    read = StubRead()
    read.set_tag("MA", "100;nuc.Q:11-40;tf.QQQ:30-10")
    diagnostics = defaultdict(int)
    counts = add_overlay_groups(
        read, [], active_passes=["sr"], diagnostics=diagnostics
    )
    parsed = parse_ma_tag(read.get_tag("MA"))
    raw = {name: intervals for name, _, _, intervals in parsed["raw_types"]}
    assert "nuc_sr" not in raw
    assert raw["tf_sr"] == [(29, 10)]
    assert counts == {"tf_sr": 1}
    assert diagnostics["nuc_removed_by_tf_overlap"] == 1


def test_add_overlay_is_idempotent_and_replaces_stale_groups():
    read = StubRead()
    add_overlay_groups(read, [proposal()])
    first = (read.get_tag("MA"), list(read.get_tag("AQ")))
    add_overlay_groups(read, [proposal()])
    assert (read.get_tag("MA"), list(read.get_tag("AQ"))) == first


def test_existing_an_is_extended_and_remains_aligned():
    read = StubRead()
    read.set_tag("AN", "original_nuc,original_tf")
    add_overlay_groups(read, [proposal()])
    names = read.get_tag("AN").split(",")
    parsed = parse_ma_tag(read.get_tag("MA"))
    annotation_count = sum(len(raw[3]) for raw in parsed["raw_types"])
    assert names[:2] == ["original_nuc", "original_tf"]
    assert len(names) == annotation_count == 5
    first = read.get_tag("AN")
    add_overlay_groups(read, [proposal()])
    assert read.get_tag("AN") == first


def test_unannotated_sequence_less_secondary_is_left_untouched():
    class EmptyRead:
        query_length = 0
        query_sequence = None

        @staticmethod
        def has_tag(_name):
            return False

    assert add_overlay_groups(EmptyRead(), []) == {}


def test_reverse_projection_writes_molecular_frame():
    read = StubRead(reverse=True)
    assert reference_interval_to_molecular(read, 110, 130) == (70, 20)
    add_overlay_groups(
        read, [replace(proposal("sr"), current_interval=(150, 190))]
    )
    parsed = parse_ma_tag(read.get_tag("MA"))
    raw = {name: intervals for name, _, _, intervals in parsed["raw_types"]}
    assert "nuc_sr" not in raw
    assert raw["tf_sr"] == [(50, 20), (69, 10), (75, 15)]


def test_collect_uses_production_scenario_and_tier_filter():
    strong = {
        "proposal_id": "strong",
        "read": "r1",
        "library_id": "/x.bam",
        "proposal_tier": "strong",
        "current_interval": [10, 110],
        "replacement_intervals": [[10, 30]],
        "best_tf_posterior": 0.97,
        "nuc_posterior": 0.02,
    }
    review = {**strong, "proposal_id": "review", "proposal_tier": "review"}
    retain = {
        **strong,
        "proposal_id": None,
        "decision_id": "retain",
        "proposal_tier": "retain_n",
        "best_tf_posterior": 0.10,
        "nuc_posterior": 0.85,
    }
    strong_state = {**strong, "decision_id": "strong"}
    review_state = {**review, "decision_id": "review"}
    report = {
        "parameters": {"production_nuc_prior_odds": 10.0},
        "composite_deconvolution": {
            "enabled": True,
            "production_scenario_key": "10.0",
            "scenarios": {
                "1.0": {"proposals": [{**strong, "proposal_id": "wrong"}]},
                "10.0": {
                    "proposals": [strong, review],
                    "candidate_states": [strong_state, review_state, retain],
                },
            },
        },
        "strand_rescue": {"enabled": False},
    }
    assert [item.proposal_id for item in collect_overlay_proposals(report)] == [
        "review", "strong"
    ]
    assert [
        item.proposal_id
        for item in collect_overlay_proposals(report, include_review=False)
    ] == ["strong"]
    assert {
        item.proposal_id
        for item in collect_overlay_proposals(
            report, include_retain_n=True
        )
    } == {"strong", "review", "retain"}


def make_bam(path: Path):
    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": 1000}],
    })
    with pysam.AlignmentFile(path, "wb", header=header) as bam:
        read = pysam.AlignedSegment(header)
        read.query_name = "read1"
        read.query_sequence = "A" * 100
        read.flag = 0
        read.reference_id = 0
        read.reference_start = 100
        read.mapping_quality = 60
        read.cigarstring = "100M"
        read.query_qualities = pysam.qualitystring_to_array("I" * 100)
        read.set_tag("MA", "100;nuc.Q:11-40;tf.QQQ:70-10", value_type="Z")
        read.set_tag("AQ", array("B", [200, 150, 10, 20]))
        bam.write(read)
    pysam.index(str(path))


def test_write_overlay_bam_is_indexed_and_declares_layers(tmp_path):
    source = tmp_path / "source.bam"
    output = tmp_path / "overlay.bam"
    make_bam(source)
    summary = write_overlay_bam(
        str(source),
        output,
        region=("chr1", 90, 210),
        proposals_by_read={"read1": [proposal()]},
        report_sha256="a" * 64,
        include_review=True,
        command_line="fiberhmm-consensus-annotate test",
    )
    assert output.is_file()
    assert Path(str(output) + ".bai").is_file()
    assert summary["reads_written"] == 1
    assert summary["proposals_matched"] == 1
    with pysam.AlignmentFile(output, "rb") as bam:
        assert set(("nuc_cr", "tf_cr", "nuc_sr", "tf_sr")) <= set(
            declared_ma_types(bam.header)
        )
        assert any(
            str(comment).startswith("FIBERHMM-CONSENSUS:v2:")
            for comment in bam.header.to_dict().get("CO", [])
        )
        read = next(bam.fetch("chr1", 90, 210))
        parsed = parse_ma_tag(read.get_tag("MA"))
        assert parsed["nuc"] == [(10, 40)]
        raw = dict((name, intervals) for name, _, _, intervals in parsed["raw_types"])
        assert "nuc_cr" not in raw
        assert raw["tf_cr"] == [(10, 15), (30, 20), (69, 10)]


def test_write_paired_overlay_declares_threshold_contract(tmp_path):
    source = tmp_path / "source.bam"
    output = tmp_path / "paired-overlay.bam"
    make_bam(source)
    summary = write_overlay_bam(
        str(source),
        output,
        region=("chr1", 90, 210),
        proposals_by_read={"read1": [paired_decision()]},
        report_sha256="b" * 64,
        include_review=True,
        paired=True,
        active_passes=["cr"],
        command_line="fiberhmm-consensus-annotate --paired test",
    )
    assert summary["paired_hypotheses"] is True
    with pysam.AlignmentFile(output, "rb") as bam:
        comments = [str(item) for item in bam.header.to_dict().get("CO", [])]
        contract = next(
            item for item in comments
            if item.startswith("FIBERHMM-CONSENSUS:v3:")
        )
        assert "semantics=paired_hypotheses" in contract
        assert "quality_spec=QQQQQ" in contract
        assert "pair_sum=255" in contract
        assert "tf_if=q0_tf>=T;nuc_if=q0_nuc>=256-T" in contract
        read = next(bam.fetch("chr1", 90, 210))
        parsed = parse_ma_tag(read.get_tag("MA"))
        assert all(
            spec == "QQQQQ"
            for name, _strand, spec, _intervals in parsed["raw_types"]
            if name in {"nuc_cr", "tf_cr"}
        )
    audit = audit_bam(str(output))
    assert audit["valid"] is True
    assert audit["indexed"] is True
    assert audit["counts"]["decisions"] == 1
    assert audit["counts"]["paired_n_vs_tf_decisions"] == 1
    assert audit["tf_component_histogram"] == {"2": 1}
    assert audit["threshold_states"]["128"] == {
        "TF": 1, "N": 0, "A": 0
    }
