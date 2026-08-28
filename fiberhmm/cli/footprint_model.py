#!/usr/bin/env python3
"""Build a footprint population model from an annotated FiberHMM BAM."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

from fiberhmm import __version__
from fiberhmm.cli.common import add_version_args
from fiberhmm.inference import SiteDiscoveryConfig, build_footprint_population_model
from fiberhmm.io import (
    BamFootprintInputError,
    convert_footprint_model_bundle_to_bigbed,
    footprint_model_bundle_paths,
    load_footprint_molecules_from_bam,
    write_footprint_model_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    defaults = SiteDiscoveryConfig()
    parser = argparse.ArgumentParser(
        prog="fiberhmm-footprint-model",
        description=(
            "Infer a footprint population model and TF-binding hypotheses from "
            "ordinary tf/msp Molecular Annotation groups in a recalled BAM. "
            "Every projected TF is retained; TQ is not read or filtered."
        ),
    )
    add_version_args(parser)
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Recalled BAM containing ordinary tf and msp MA groups",
    )
    parser.add_argument(
        "-o",
        "--output-prefix",
        required=True,
        help="Output filename prefix (not a directory)",
    )

    scope = parser.add_argument_group("scope and output")
    scope.add_argument(
        "--region",
        action="append",
        default=[],
        metavar="CONTIG:START-END",
        help=(
            "Analyze a zero-based, half-open region; repeatable and requires a BAM index. "
            "A bare CONTIG selects that complete contig."
        ),
    )
    scope.add_argument(
        "--genome",
        default=None,
        help="Genome/assembly label recorded in provenance (for example dm6)",
    )
    scope.add_argument(
        "--genomewide",
        action="store_true",
        help="Assert that scanning the complete BAM represents a genome-wide analysis",
    )
    scope.add_argument(
        "--bigbed",
        action="store_true",
        help="Also create indexed population and FiberBrowser BigBeds",
    )
    scope.add_argument(
        "--bed-to-bigbed",
        default=None,
        help="Path to UCSC bedToBigBed; implies --bigbed",
    )
    scope.add_argument(
        "--force",
        action="store_true",
        help="Replace existing artifacts for this output prefix",
    )

    filtering = parser.add_argument_group("BAM filtering")
    filtering.add_argument(
        "-q",
        "--min-mapq",
        type=int,
        default=0,
        help="Minimum alignment MAPQ (default: 0)",
    )
    filtering.add_argument(
        "--include-duplicates",
        action="store_true",
        help="Include records carrying the BAM duplicate flag (excluded by default)",
    )

    model = parser.add_argument_group("model parameters")
    model.add_argument(
        "--smoothing-sigma",
        type=float,
        default=defaults.smoothing_sigma_bp,
        help=f"Footprint-center Gaussian sigma in bp (default: {defaults.smoothing_sigma_bp:g})",
    )
    model.add_argument(
        "--peak-distance",
        type=int,
        default=defaults.peak_distance_bp,
        help=f"Minimum distance between center modes (default: {defaults.peak_distance_bp})",
    )
    model.add_argument(
        "--assignment-radius",
        type=int,
        default=defaults.assignment_radius_bp,
        help=f"Maximum center-to-mode assignment radius (default: {defaults.assignment_radius_bp})",
    )
    model.add_argument(
        "--edge-compatibility",
        type=int,
        default=defaults.edge_compatibility_bp,
        help=f"Maximum within-family diameter for each boundary (default: {defaults.edge_compatibility_bp})",
    )
    model.add_argument(
        "--minimum-geometry-support-per-stratum",
        type=int,
        default=defaults.minimum_geometry_support_per_stratum,
        help=(
            "Molecules per stratum needed for that stratum to vote on canonical geometry "
            f"(default: {defaults.minimum_geometry_support_per_stratum})"
        ),
    )
    model.add_argument(
        "--minimum-geometry-support",
        type=int,
        default=defaults.minimum_geometry_support,
        help=f"Descriptive geometry-ready threshold (default: {defaults.minimum_geometry_support})",
    )
    model.add_argument(
        "--minimum-population-support",
        type=int,
        default=defaults.minimum_population_support,
        help=f"Descriptive population-ready threshold (default: {defaults.minimum_population_support})",
    )
    model.add_argument(
        "--minimum-mapped-fraction",
        type=float,
        default=defaults.minimum_mapped_fraction,
        help=(
            "Required mapped fraction for geometry, MSP projection, and site denominators "
            f"(default: {defaults.minimum_mapped_fraction:g})"
        ),
    )
    return parser


def _model_config(args) -> SiteDiscoveryConfig:
    return SiteDiscoveryConfig(
        smoothing_sigma_bp=args.smoothing_sigma,
        peak_distance_bp=args.peak_distance,
        assignment_radius_bp=args.assignment_radius,
        edge_compatibility_bp=args.edge_compatibility,
        minimum_geometry_support_per_stratum=args.minimum_geometry_support_per_stratum,
        minimum_geometry_support=args.minimum_geometry_support,
        minimum_population_support=args.minimum_population_support,
        minimum_mapped_fraction=args.minimum_mapped_fraction,
    )


def _existing_outputs(prefix: str) -> Sequence[Path]:
    paths = footprint_model_bundle_paths(prefix)
    candidates = [Path(value) for value in paths.as_dict().values()]
    candidates.extend(
        [
            paths.population_bed.with_suffix(".bb"),
            paths.fiberlayers_bed.with_suffix(".bb"),
        ]
    )
    return tuple(path for path in candidates if path.exists())


def _convert_bigbeds(paths, references, executable: Optional[str]):
    with tempfile.TemporaryDirectory(prefix="fiberhmm_footprint_model_") as temporary:
        chrom_sizes = Path(temporary) / "bam.chrom.sizes"
        with chrom_sizes.open("w", encoding="utf-8", newline="\n") as handle:
            for contig, length in references:
                handle.write(f"{contig}\t{length}\n")
        return convert_footprint_model_bundle_to_bigbed(
            paths,
            chrom_sizes,
            bed_to_bigbed=executable,
        )


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.genomewide and args.region:
        parser.error("--genomewide cannot be combined with --region")

    make_bigbed = bool(args.bigbed or args.bed_to_bigbed)
    existing = _existing_outputs(args.output_prefix)
    if existing and not args.force:
        print(
            "Error: output artifacts already exist; use --force to replace them:\n  "
            + "\n  ".join(str(path) for path in existing),
            file=sys.stderr,
        )
        return 2
    bed_to_bigbed = None
    if make_bigbed:
        requested_converter = args.bed_to_bigbed or "bedToBigBed"
        bed_to_bigbed = shutil.which(requested_converter)
        if bed_to_bigbed is None:
            print(
                f"Error: bedToBigBed executable was not found: {requested_converter}",
                file=sys.stderr,
            )
            return 2
    try:
        config = _model_config(args)
        loaded = load_footprint_molecules_from_bam(
            args.input,
            regions=args.region,
            min_mapq=args.min_mapq,
            include_duplicates=args.include_duplicates,
            minimum_projection_fraction=config.minimum_mapped_fraction,
        )
        if loaded.diagnostics.emitted_molecules == 0 and not args.region:
            raise BamFootprintInputError(
                "no analyzable primary records with valid MA annotations were found; "
                "use a footprint-called/recalled BAM"
            )
        if loaded.diagnostics.projected_tf_annotations == 0 and not args.region:
            raise BamFootprintInputError(
                "no projectable ordinary tf annotations were found; "
                "use a footprint-called/recalled BAM"
            )
        if loaded.diagnostics.projected_tf_annotations == 0:
            scope = "the requested region(s)" if args.region else "the analyzed BAM"
            print(
                f"Warning: no projectable ordinary tf annotations were found in {scope}; "
                "writing an empty footprint population model.",
                file=sys.stderr,
            )
        catalog = build_footprint_population_model(loaded.molecules, config=config)
        stat = loaded.input_path.stat()
        source_dataset = {
            "id": loaded.input_path.stem,
            "path": str(loaded.input_path),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "adapter": {
                "schema": "fiberhmm.footprint_bam_adapter.v1",
                "annotation_source": "MA:tf,msp",
                "stratum_rule": {
                    "preferred_tag": "st",
                    "accepted_tag_values": ["CT", "GA"],
                    "fallback": "alignment_orientation:FWD/REV",
                },
                "filters": {
                    "primary_only": True,
                    "exclude_qcfail": True,
                    "exclude_duplicates": not args.include_duplicates,
                    "min_mapq": args.min_mapq,
                    "missing_ma": "exclude_as_not_analyzed",
                    "minimum_projection_fraction": config.minimum_mapped_fraction,
                },
                "diagnostics": loaded.diagnostics.as_dict(),
            },
        }
        if args.force:
            # Preserve an existing indexed generation until BAM validation and
            # model construction have succeeded. Removing it here also keeps
            # stale BigBeds from surviving a forced text-only replacement.
            for path in existing:
                if path.suffix == ".bb":
                    path.unlink()
        paths = write_footprint_model_bundle(
            catalog,
            args.output_prefix,
            source_dataset=source_dataset,
            genome=args.genome,
            genomewide=bool(args.genomewide),
            regions=loaded.region_labels,
        )
        bigbeds = None
        if make_bigbed:
            bigbeds = _convert_bigbeds(paths, loaded.references, bed_to_bigbed)
    except (BamFootprintInputError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    summary = {
        "schema": "fiberhmm.footprint_model_run.v1",
        "fiberhmm_version": __version__,
        "input": str(loaded.input_path),
        "scope": {
            "regions": list(loaded.region_labels),
            "genomewide": bool(args.genomewide),
        },
        "config": asdict(config),
        "bam_diagnostics": loaded.diagnostics.as_dict(),
        "model_diagnostics": asdict(catalog.diagnostics),
        "model": {
            "loci": len(catalog.loci),
            "binding_hypotheses": len(catalog.sites),
            "assignments": len(catalog.assignments),
        },
        "files": dict(paths.as_dict()),
        "bigbeds": None
        if bigbeds is None
        else {
            "population": str(bigbeds.population_bigbed),
            "fiberlayers": str(bigbeds.fiberlayers_bigbed),
        },
    }
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
