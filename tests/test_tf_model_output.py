from __future__ import annotations

import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fiberhmm.inference.tf_sites import (
    BaselineMolecule,
    SiteDiscoveryConfig,
    TFObservation,
    build_tf_model_catalog,
)
from fiberhmm.io.tf_models import (
    convert_tf_model_bundle_to_bigbed,
    write_tf_model_bundle,
)

CREATED_AT = "2026-07-22T12:34:56+00:00"


def _molecule(
    name,
    *,
    tfs=(),
    msps=(),
    blocks=((0, 1_000),),
    contig="chr1",
    stratum="FWD",
):
    return BaselineMolecule(
        molecule_id=name,
        contig=contig,
        stratum=stratum,
        mapped_blocks=tuple(blocks),
        tfs=tuple(
            TFObservation(
                call_id=f"{name}-tf-{index}",
                start=start,
                end=end,
                geometry_eligible=geometry_eligible,
            )
            for index, (start, end, geometry_eligible) in enumerate(tfs)
        ),
        msps=tuple(msps),
    )


@pytest.fixture
def tf_model_catalog():
    molecules = [
        *[_molecule(f"wide-{index}", tfs=((90, 130, True),)) for index in range(3)],
        *[_molecule(f"short-{index}", tfs=((100, 120, True),)) for index in range(3)],
        *[_molecule(f"distal-{index}", tfs=((300, 320, True),)) for index in range(3)],
        # One molecule carries two nested families at the first locus and a
        # second, non-overlapping locus.  The nested calls force two physical
        # BED rows; the distal call can share a row with one of them.
        _molecule(
            "overlap-read",
            tfs=(
                (90, 130, False),
                (100, 120, False),
                (300, 320, False),
            ),
            msps=((80, 140),),
        ),
        _molecule("msp-only", msps=((80, 140),)),
        _molecule("unoccupied"),
    ]
    return build_tf_model_catalog(
        molecules,
        config=SiteDiscoveryConfig(edge_compatibility_bp=5),
    )


def _read_tsv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _bed_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [line.rstrip("\n").split("\t") for line in handle if line.strip()]


def _comma_values(value: str):
    return [item for item in value.split(",") if item]


def _fiberlayer_blocks(fields):
    assert len(fields) == 17
    block_count = int(fields[9])
    sizes = [int(value) for value in _comma_values(fields[10])]
    offsets = [int(value) for value in _comma_values(fields[11])]
    qualities = [int(value) for value in _comma_values(fields[15])]
    site_ids = _comma_values(fields[16])
    assert len(sizes) == len(offsets) == len(qualities) == len(site_ids) == block_count
    intervals = [
        (int(fields[1]) + offset, int(fields[1]) + offset + size)
        for offset, size in zip(offsets, sizes)
    ]
    return intervals, qualities, site_ids


def test_population_export_has_one_stable_row_per_geometry_family(tmp_path, tf_model_catalog):
    paths = write_tf_model_bundle(
        tf_model_catalog,
        tmp_path / "sample",
        created_at=CREATED_AT,
    )

    rows = _read_tsv(paths.population_tsv)
    assert [row["site_id"] for row in rows] == [site.site_id for site in tf_model_catalog.sites]
    assert len(rows) == 3

    for row, site in zip(rows, tf_model_catalog.sites):
        assert (row["contig"], int(row["start"]), int(row["end"])) == (
            site.contig,
            site.start,
            site.end,
        )
        assert row["locus_id"] == site.locus_id
        assert int(row["locus_summit"]) == site.locus_summit
        assert int(row["summit"]) == site.summit
        assert int(row["family_index"]) == site.family_index
        assert int(row["n_fully_mapped"]) == site.n_fully_mapped
        assert int(row["n_tf"]) == site.n_tf
        assert float(row["occupancy_overall"]) == pytest.approx(site.occupancy_overall)
        if site.occupancy_given_msp is None:
            assert row["occupancy_given_msp"] == "."
        else:
            assert float(row["occupancy_given_msp"]) == pytest.approx(site.occupancy_given_msp)

    nested = [site for site in tf_model_catalog.sites if site.summit == 110]
    assert len(nested) == 2
    assert len({site.locus_id for site in nested}) == 1
    assert [site.family_index for site in nested] == [1, 2]

    population_bed = _bed_rows(paths.population_bed)
    assert len(population_bed) == len(tf_model_catalog.sites)
    assert all(len(fields) == 26 for fields in population_bed)
    assert [fields[3] for fields in population_bed] == [
        site.site_id for site in tf_model_catalog.sites
    ]
    assert [int(fields[4]) for fields in population_bed] == [
        min(site.population_support, 1000) for site in tf_model_catalog.sites
    ]


def test_population_bed_score_ranks_support_not_occupancy(tmp_path):
    molecules = [
        _molecule(
            "singleton",
            blocks=((90, 130),),
            tfs=((100, 120, True),),
        ),
        *[
            _molecule(
                f"recurrent-{index}",
                blocks=((290, 330),),
                tfs=((300, 320, True),),
            )
            for index in range(3)
        ],
        *[
            _molecule(
                f"recurrent-denominator-{index}",
                blocks=((290, 330),),
            )
            for index in range(97)
        ],
    ]
    catalog = build_tf_model_catalog(molecules)
    paths = write_tf_model_bundle(catalog, tmp_path / "support-score")
    rows_by_start = {int(fields[1]): fields for fields in _bed_rows(paths.population_bed)}
    sites_by_start = {site.start: site for site in catalog.sites}

    assert sites_by_start[100].occupancy_overall == 1.0
    assert sites_by_start[300].occupancy_overall == pytest.approx(0.03)
    assert int(rows_by_start[100][4]) == 1
    assert int(rows_by_start[300][4]) == 3
    assert not sites_by_start[100].analysis_ready
    assert sites_by_start[300].analysis_ready


def test_fiberlayer_has_one_ui_class_and_block_site_ids_are_aligned(tmp_path, tf_model_catalog):
    paths = write_tf_model_bundle(
        tf_model_catalog,
        tmp_path / "sample",
        created_at=CREATED_AT,
    )
    rows = _bed_rows(paths.fiberlayers_bed)

    assert rows
    assert {fields[12] for fields in rows} == {"footprint_model_assignments"}
    assert {fields[13] for fields in rows} == {"tf"}
    assert {fields[14] for fields in rows} == {"fiberhmm_footprint_model_v1"}
    assert {fields[4] for fields in rows} == {"1000"}

    observed_assignments = []
    for fields in rows:
        intervals, qualities, site_ids = _fiberlayer_blocks(fields)
        assert qualities == [255] * len(intervals)
        observed_assignments.extend(
            (fields[3], start, end, site_id) for (start, end), site_id in zip(intervals, site_ids)
        )

    expected_assignments = sorted(
        (
            assignment.molecule_id,
            assignment.start,
            assignment.end,
            assignment.site_id,
        )
        for assignment in tf_model_catalog.assignments
        if assignment.site_id is not None
    )
    assert sorted(observed_assignments) == expected_assignments

    manifest = json.loads(paths.fiberlayers_manifest.read_text(encoding="utf-8"))
    assert set(manifest["classes"]) == {"footprint_model_assignments"}
    assert not set(manifest["classes"]) & {site.site_id for site in tf_model_catalog.sites}
    autosql = paths.fiberlayers_autosql.read_text(encoding="utf-8")
    assert "table fiberbrowser_fiberlayers" in autosql
    assert "uint reserved;" in autosql
    assert "lstring blockQuality;" in autosql
    assert "lstring blockSiteIds;" in autosql


def test_overlapping_calls_are_partitioned_into_non_overlapping_bed12_lanes(
    tmp_path, tf_model_catalog
):
    paths = write_tf_model_bundle(
        tf_model_catalog,
        tmp_path / "sample",
        created_at=CREATED_AT,
    )
    overlap_rows = [
        fields for fields in _bed_rows(paths.fiberlayers_bed) if fields[3] == "overlap-read"
    ]

    # Duplicate read/class rows are intentional physical lanes, not new UI
    # layers.  Three calls need two lanes because two are nested.
    assert len(overlap_rows) == 2
    assert {fields[12] for fields in overlap_rows} == {"footprint_model_assignments"}
    all_blocks = []
    for fields in overlap_rows:
        intervals, _qualities, _site_ids = _fiberlayer_blocks(fields)
        assert intervals == sorted(intervals)
        assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))
        all_blocks.extend(intervals)

    assert sorted(all_blocks) == [(90, 130), (100, 120), (300, 320)]
    assert sorted(int(fields[9]) for fields in overlap_rows) == [1, 2]


def test_manifest_declares_extensions_and_fixed_created_at_is_reproducible(
    tmp_path, tf_model_catalog
):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    kwargs = {
        "source_dataset": {"path": "/data/sample.bam", "id": "sample"},
        "genome": "dm6",
        "created_at": CREATED_AT,
        "genomewide": True,
    }
    first = write_tf_model_bundle(tf_model_catalog, first_dir / "models", **kwargs)
    second = write_tf_model_bundle(tf_model_catalog, second_dir / "models", **kwargs)

    manifest = json.loads(first.fiberlayers_manifest.read_text(encoding="utf-8"))
    assert manifest["schema"] == "fiberlayers.v1"
    assert set(manifest["classes"]) == {"footprint_model_assignments"}
    assert set(manifest["extensions"]) == {
        "block_site_ids",
        "duplicate_class_rows",
    }
    assert manifest["extensions"]["block_site_ids"]
    assert manifest["extensions"]["duplicate_class_rows"]
    assert manifest["source_dataset"]["path"] == "/data/sample.bam"
    assert manifest["source_dataset"]["id"] == "sample"
    assert manifest["source_dataset"]["genome"] == "dm6"
    assert manifest["scope"]["genomewide"] is True
    assert manifest["provenance"]["created_at"] == CREATED_AT
    assert manifest["provenance"]["exporter_version"]
    assert manifest["model"]["support"]["bed_score"]["formula"] == (
        "min(population_support, 1000)"
    )
    assert manifest["model"]["support"]["null_background"] is None
    assert manifest["classes"]["footprint_model_assignments"]["rules"][
        "recommended_site_filter"
    ] == {
        "all": [
            {"equals": 1, "field": "geometry_ready"},
            {"equals": 1, "field": "population_ready"},
        ]
    }
    assert manifest["statistics"]["analysis_ready_sites"] == sum(
        site.analysis_ready for site in tf_model_catalog.sites
    )

    for field in first.__dataclass_fields__:
        first_path = getattr(first, field)
        second_path = getattr(second, field)
        assert first_path.name == second_path.name
        assert first_path.read_bytes() == second_path.read_bytes()


def test_targeted_scope_is_explicit_and_conservative_by_default(tmp_path, tf_model_catalog):
    default_paths = write_tf_model_bundle(tf_model_catalog, tmp_path / "default")
    default_manifest = json.loads(default_paths.fiberlayers_manifest.read_text(encoding="utf-8"))
    assert default_manifest["scope"]["genomewide"] is False
    assert default_manifest["scope"]["regions"] == []

    targeted_paths = write_tf_model_bundle(
        tf_model_catalog,
        tmp_path / "targeted",
        regions=("chr1:80-340",),
    )
    targeted_manifest = json.loads(targeted_paths.fiberlayers_manifest.read_text(encoding="utf-8"))
    assert targeted_manifest["scope"]["genomewide"] is False
    assert targeted_manifest["scope"]["regions"] == ["chr1:80-340"]


@pytest.mark.parametrize("corruption", ["missing_site", "duplicate_site"])
def test_writer_rejects_catalog_identity_corruption_before_writing(
    tmp_path, tf_model_catalog, corruption
):
    if corruption == "missing_site":
        malformed = replace(tf_model_catalog, sites=(), loci=())
    else:
        malformed = replace(
            tf_model_catalog,
            sites=tf_model_catalog.sites + (tf_model_catalog.sites[0],),
        )
    prefix = tmp_path / corruption / "sample"

    with pytest.raises(ValueError, match="site_id"):
        write_tf_model_bundle(malformed, prefix)

    assert not prefix.parent.exists()


def test_invalid_manifest_metadata_fails_before_artifacts_are_written(tmp_path, tf_model_catalog):
    prefix = tmp_path / "invalid-metadata" / "sample"

    with pytest.raises(TypeError):
        write_tf_model_bundle(
            tf_model_catalog,
            prefix,
            source_dataset={"not_json": object()},
        )

    assert not prefix.parent.exists()


def test_empty_catalog_writes_headers_and_zero_count_manifest(tmp_path):
    catalog = build_tf_model_catalog([])
    paths = write_tf_model_bundle(catalog, tmp_path / "empty")

    assert _read_tsv(paths.population_tsv) == []
    assert _read_tsv(paths.strata_tsv) == []
    assert _read_tsv(paths.assignments_tsv) == []
    assert _bed_rows(paths.population_bed) == []
    assert _bed_rows(paths.fiberlayers_bed) == []
    manifest = json.loads(paths.fiberlayers_manifest.read_text(encoding="utf-8"))
    assert manifest["statistics"]["loci"] == 0
    assert manifest["statistics"]["sites"] == 0
    assert manifest["statistics"]["assigned_calls"] == 0


def test_optional_bigbed_conversion_updates_manifest_atomically(
    tmp_path, tf_model_catalog, monkeypatch
):
    paths = write_tf_model_bundle(tf_model_catalog, tmp_path / "sample")
    chrom_sizes = tmp_path / "chrom.sizes"
    chrom_sizes.write_text("chr1\t1000\n", encoding="utf-8")
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"synthetic-bigbed")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("fiberhmm.io.tf_models.subprocess.run", fake_run)
    bigbeds = convert_tf_model_bundle_to_bigbed(
        paths,
        chrom_sizes,
        bed_to_bigbed="/tools/bedToBigBed",
    )

    assert bigbeds.population_bigbed.read_bytes() == b"synthetic-bigbed"
    assert bigbeds.fiberlayers_bigbed.read_bytes() == b"synthetic-bigbed"
    assert {command[2] for command in commands} == {
        "-type=bed6+20",
        "-type=bed12+5",
    }
    manifest = json.loads(paths.fiberlayers_manifest.read_text(encoding="utf-8"))
    assert manifest["files"]["bigbed"] == bigbeds.fiberlayers_bigbed.name
    assert manifest["files"]["population_bigbed"] == bigbeds.population_bigbed.name


@pytest.mark.skipif(shutil.which("bedToBigBed") is None, reason="UCSC tools not installed")
def test_real_bigbed_accepts_long_block_aligned_site_id_payload(tmp_path):
    calls = tuple((20 + index * 50, 40 + index * 50, True) for index in range(16))
    molecules = [_molecule(f"read-{index}", tfs=calls) for index in range(3)]
    catalog = build_tf_model_catalog(molecules)
    paths = write_tf_model_bundle(catalog, tmp_path / "long-payload")
    rows = _bed_rows(paths.fiberlayers_bed)

    assert max(len(fields[16]) for fields in rows) > 255
    chrom_sizes = tmp_path / "chrom.sizes"
    chrom_sizes.write_text("chr1\t1000\n", encoding="utf-8")
    bigbeds = convert_tf_model_bundle_to_bigbed(paths, chrom_sizes)

    assert bigbeds.population_bigbed.is_file()
    assert bigbeds.fiberlayers_bigbed.is_file()
