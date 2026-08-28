from __future__ import annotations

import json
from pathlib import Path

import pysam
import pytest

from fiberhmm.cli.footprint_model import main
from fiberhmm.inference import build_footprint_population_model
from fiberhmm.io import (
    BamFootprintInputError,
    load_footprint_molecules_from_bam,
)

HEADER = {
    "HD": {"VN": "1.6", "SO": "coordinate"},
    "SQ": [{"SN": "chr1", "LN": 2_000}],
}


def _read(
    name,
    *,
    start=100,
    length=100,
    reverse=False,
    ma=None,
    an=None,
    st=None,
    mapq=60,
    extra_flag=0,
    cigar=None,
):
    read = pysam.AlignedSegment()
    read.query_name = name
    read.query_sequence = "A" * length
    read.query_qualities = pysam.qualitystring_to_array("I" * length)
    read.flag = (16 if reverse else 0) | extra_flag
    read.reference_id = 0
    read.reference_start = start
    read.mapping_quality = mapq
    read.cigartuples = cigar or [(0, length)]
    if ma is not None:
        read.set_tag("MA", ma, value_type="Z")
    if an is not None:
        read.set_tag("AN", an, value_type="Z")
    if st is not None:
        read.set_tag("st", st, value_type="Z")
    return read


def _write_bam(path: Path, reads, *, index=True):
    with pysam.AlignmentFile(str(path), "wb", header=HEADER) as output:
        for read in sorted(reads, key=lambda value: value.reference_start):
            output.write(read)
    if index:
        pysam.index(str(path))
    return path


def test_bam_loader_projects_forward_and_reverse_ma_and_keeps_zero_tf_denominator(
    tmp_path,
):
    bam = _write_bam(
        tmp_path / "calls.bam",
        [
            _read(
                "forward",
                ma="100;msp.:11-50;tf.QQQ:21-20",
                st="CT",
            ),
            # Molecular [60,80) flips to stored-query [20,40), the same
            # reference interval as the forward call.
            _read(
                "reverse",
                reverse=True,
                ma="100;msp.:51-40;tf.QQQ:61-20",
                st="GA",
            ),
            _read("no-tf", ma="100;msp.:11-50", st="CT"),
            _read("not-analyzed", ma=None, st="CT"),
        ],
    )

    loaded = load_footprint_molecules_from_bam(bam)

    assert [molecule.molecule_id for molecule in loaded.molecules] == [
        "forward",
        "reverse",
        "no-tf",
    ]
    assert [molecule.stratum for molecule in loaded.molecules] == ["CT", "GA", "CT"]
    assert [(call.start, call.end) for call in loaded.molecules[0].tfs] == [(120, 140)]
    assert [(call.start, call.end) for call in loaded.molecules[1].tfs] == [(120, 140)]
    assert loaded.molecules[1].msps == ((110, 150),)
    assert loaded.diagnostics.primary_mapped_records == 4
    assert loaded.diagnostics.emitted_molecules == 3
    assert loaded.diagnostics.skipped_missing_ma == 1

    model = build_footprint_population_model(loaded.molecules)
    assert len(model.sites) == 1
    hypothesis = model.sites[0]
    assert (hypothesis.start, hypothesis.end) == (120, 140)
    assert hypothesis.n_fully_mapped == 3
    assert hypothesis.n_tf == 2
    assert hypothesis.occupancy_overall == pytest.approx(2 / 3)
    assert hypothesis.n_msp == 3
    assert hypothesis.occupancy_given_msp == pytest.approx(2 / 3)


def test_partially_projected_tf_is_retained_but_does_not_teach_geometry(tmp_path):
    bam = _write_bam(
        tmp_path / "softclip.bam",
        [
            _read(
                "partial",
                ma="100;msp.:6-20;tf.QQQ:6-20",
                cigar=[(4, 10), (0, 80), (4, 10)],
            )
        ],
    )

    loaded = load_footprint_molecules_from_bam(bam)

    assert len(loaded.molecules[0].tfs) == 1
    call = loaded.molecules[0].tfs[0]
    assert (call.start, call.end, call.geometry_eligible) == (100, 115, False)
    assert loaded.molecules[0].msps == ()
    assert loaded.diagnostics.projected_tf_annotations == 1
    assert loaded.diagnostics.geometry_eligible_tf_annotations == 0
    assert loaded.diagnostics.incomplete_msp_annotations == 1


def test_hard_clips_indels_and_reference_gaps_preserve_projection_semantics(tmp_path):
    bam = _write_bam(
        tmp_path / "cigar.bam",
        [
            _read(
                "hard-clipped",
                start=100,
                length=80,
                ma="100;tf.QQQ:11-20",
                cigar=[(5, 10), (0, 80), (5, 10)],
            ),
            _read(
                "insertion",
                start=300,
                ma="100;tf.QQQ:36-15",
                cigar=[(0, 40), (1, 5), (0, 55)],
            ),
            _read(
                "skipped-reference",
                start=500,
                ma="100;tf.QQQ:36-10",
                cigar=[(0, 40), (3, 20), (0, 60)],
            ),
            _read(
                "deletion",
                start=700,
                ma="100;tf.QQQ:36-10",
                cigar=[(0, 40), (2, 5), (0, 60)],
            ),
        ],
    )

    loaded = load_footprint_molecules_from_bam(bam)
    calls = {
        molecule.molecule_id: molecule.tfs[0] for molecule in loaded.molecules
    }

    assert (calls["hard-clipped"].start, calls["hard-clipped"].end) == (100, 120)
    assert calls["hard-clipped"].geometry_eligible
    assert (calls["insertion"].start, calls["insertion"].end) == (335, 345)
    assert not calls["insertion"].geometry_eligible
    assert (calls["skipped-reference"].start, calls["skipped-reference"].end) == (
        535,
        565,
    )
    assert (calls["deletion"].start, calls["deletion"].end) == (735, 750)
    assert loaded.molecules[2].mapped_blocks == ((500, 540), (560, 620))
    assert loaded.molecules[3].mapped_blocks == ((700, 740), (745, 805))

    model = build_footprint_population_model(loaded.molecules)
    assert model.diagnostics.geometry_tf_calls == 1


def test_bam_loader_filters_flags_mapq_and_missing_ma(tmp_path):
    annotated = "100;tf.QQQ:21-20"
    bam = _write_bam(
        tmp_path / "filters.bam",
        [
            _read("kept", ma=annotated),
            _read("low-mapq", ma=annotated, mapq=9),
            _read("duplicate", ma=annotated, extra_flag=0x400),
            _read("qcfail", ma=annotated, extra_flag=0x200),
            _read("secondary", ma=annotated, extra_flag=0x100),
            _read("supplementary", ma=annotated, extra_flag=0x800),
            _read("missing", ma=None),
        ],
    )

    loaded = load_footprint_molecules_from_bam(bam, min_mapq=10)

    assert [molecule.molecule_id for molecule in loaded.molecules] == ["kept"]
    diagnostics = loaded.diagnostics
    assert diagnostics.skipped_low_mapq == 1
    assert diagnostics.skipped_duplicate == 1
    assert diagnostics.skipped_qcfail == 1
    assert diagnostics.skipped_secondary == 1
    assert diagnostics.skipped_supplementary == 1
    assert diagnostics.skipped_missing_ma == 1

    with_duplicates = load_footprint_molecules_from_bam(
        bam,
        min_mapq=10,
        include_duplicates=True,
    )
    assert {molecule.molecule_id for molecule in with_duplicates.molecules} == {
        "kept",
        "duplicate",
    }


def test_region_is_zero_based_filters_tf_centers_and_requires_index(tmp_path):
    indexed = _write_bam(
        tmp_path / "indexed.bam",
        [
            _read(
                "spanning",
                length=400,
                ma="400;tf.QQQ:21-20,221-20",
            )
        ],
    )

    loaded = load_footprint_molecules_from_bam(indexed, regions=("chr1:110-150",))

    assert loaded.region_labels == ("chr1:110-150",)
    assert [(call.start, call.end) for call in loaded.molecules[0].tfs] == [(120, 140)]
    assert loaded.diagnostics.out_of_scope_tf_annotations == 1

    disjoint = load_footprint_molecules_from_bam(
        indexed,
        regions=("chr1:110-150", "chr1:310-350"),
    )
    assert len(disjoint.molecules) == 1
    assert [(call.start, call.end) for call in disjoint.molecules[0].tfs] == [
        (120, 140),
        (320, 340),
    ]
    assert disjoint.diagnostics.duplicate_region_records == 1

    unindexed = _write_bam(
        tmp_path / "unindexed.bam",
        [_read("read", ma="100;tf.QQQ:21-20")],
        index=False,
    )
    assert len(load_footprint_molecules_from_bam(unindexed).molecules) == 1
    with pytest.raises(BamFootprintInputError, match="indexed BAM"):
        load_footprint_molecules_from_bam(unindexed, regions=("chr1:0-200",))


def test_malformed_ma_fails_with_read_identity(tmp_path):
    bam = _write_bam(
        tmp_path / "malformed.bam",
        [_read("broken", ma="not-an-ma-tag")],
    )

    with pytest.raises(BamFootprintInputError, match="broken"):
        load_footprint_molecules_from_bam(bam)


def test_an_linked_ordinary_annotations_fail_instead_of_becoming_linear_sites(tmp_path):
    bam = _write_bam(
        tmp_path / "circular.bam",
        [
            _read(
                "wrapped",
                ma="100;msp.:1-10,91-10;tf.QQQ:1-5,96-5",
                an="msp-wrap,msp-wrap,tf-wrap,tf-wrap",
            )
        ],
    )

    with pytest.raises(BamFootprintInputError, match="AN-linked.*wrapped"):
        load_footprint_molecules_from_bam(bam)


def test_present_invalid_stratum_tag_is_not_silently_treated_as_missing(tmp_path):
    bam = _write_bam(
        tmp_path / "invalid-stratum.bam",
        [_read("broken", ma="100;tf.QQQ:21-20", st="unknown")],
    )

    with pytest.raises(BamFootprintInputError, match="invalid st tag.*broken"):
        load_footprint_molecules_from_bam(bam)

    compatible = load_footprint_molecules_from_bam(
        bam,
        invalid_stratum_tag_policy="alignment",
    )
    assert len(compatible.molecules) == 1
    assert compatible.molecules[0].stratum in {"FWD", "REV"}


def test_cli_writes_auditable_footprint_model_bundle(tmp_path, capsys):
    bam = _write_bam(
        tmp_path / "calls.bam",
        [_read(f"tf-{index}", ma="100;msp.:11-50;tf.QQQ:21-20", st="CT") for index in range(3)]
        + [_read("empty", ma="100;msp.:11-50", st="GA")],
    )
    prefix = tmp_path / "result" / "sample"

    assert (
        main(
            [
                "--input",
                str(bam),
                "--output-prefix",
                str(prefix),
                "--genome",
                "dm6",
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["model"] == {
        "assignments": 3,
        "binding_hypotheses": 1,
        "loci": 1,
    }
    assert summary["bam_diagnostics"]["emitted_molecules"] == 4
    assert summary["files"]["population_tsv"].endswith(".footprint-model.tsv")
    manifest_path = Path(summary["files"]["fiberlayers_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["classes"]) == {"footprint_model_assignments"}
    assert manifest["model"]["object"] == "footprint_population_model"
    assert manifest["model"]["row_object"] == "tf_binding_hypothesis"
    assert manifest["model"]["config"]["smoothing_sigma_bp"] == 3.0
    assert manifest["provenance"]["exporter_version"]
    adapter = manifest["source_dataset"]["adapter"]
    assert adapter["annotation_source"] == "MA:tf,msp"
    assert adapter["stratum_rule"] == {
        "accepted_tag_values": ["CT", "GA"],
        "fallback": "alignment_orientation:FWD/REV",
        "preferred_tag": "st",
    }
    assert adapter["diagnostics"]["emitted_molecules"] == 4

    assert main(["-i", str(bam), "-o", str(prefix)]) == 2
    assert "--force" in capsys.readouterr().err


def test_cli_rejects_whole_bam_without_analyzable_ma_but_allows_empty_region(
    tmp_path,
    capsys,
):
    bam = _write_bam(tmp_path / "raw.bam", [_read("raw", ma=None)])
    whole_prefix = tmp_path / "whole"
    stale_bigbed = Path(str(whole_prefix) + ".footprint-model.fiberlayers.bb")
    stale_bigbed.write_bytes(b"old")

    assert (
        main(
            [
                "-i",
                str(bam),
                "-o",
                str(whole_prefix),
                "--force",
            ]
        )
        == 2
    )
    assert "footprint-called/recalled BAM" in capsys.readouterr().err
    assert stale_bigbed.read_bytes() == b"old"

    assert (
        main(
            [
                "-i",
                str(bam),
                "-o",
                str(tmp_path / "targeted"),
                "--region",
                "chr1:1000-1100",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "writing an empty footprint population model" in captured.err
    assert json.loads(captured.out)["model"]["binding_hypotheses"] == 0


def test_cli_preflights_bigbed_converter_before_replacing_outputs(
    tmp_path,
    capsys,
    monkeypatch,
):
    bam = _write_bam(
        tmp_path / "calls.bam",
        [_read("tf", ma="100;tf.QQQ:21-20")],
    )
    prefix = tmp_path / "sample"
    stale_bigbed = Path(str(prefix) + ".footprint-model.fiberlayers.bb")
    stale_bigbed.write_bytes(b"old")
    monkeypatch.setattr("fiberhmm.cli.footprint_model.shutil.which", lambda _value: None)

    assert (
        main(
            [
                "-i",
                str(bam),
                "-o",
                str(prefix),
                "--bigbed",
                "--force",
            ]
        )
        == 2
    )
    assert "bedToBigBed executable was not found" in capsys.readouterr().err
    assert stale_bigbed.read_bytes() == b"old"
