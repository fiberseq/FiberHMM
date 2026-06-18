"""TSV helpers for region-parallel posterior export."""

from __future__ import annotations

import base64
import gzip
import json
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from fiberhmm.io.path_status import path_is_nonempty_file
from fiberhmm.posteriors.writer import _resolve_writer_format

REGION_POSTERIORS_HEADER = (
    "#read_id\tchrom\tstart\tend\tstrand\tposteriors_b64\tfp_starts\tfp_sizes\n"
)


@dataclass(frozen=True)
class _IndexedRegionTsvFile:
    region_index: int
    path: str


@dataclass(frozen=True)
class _RegionPosteriorLineRecord:
    read_name: str
    chrom: str
    ref_start: int
    ref_end: int
    strand: str
    posteriors: np.ndarray
    footprint_starts: Sequence[int]
    footprint_sizes: Sequence[int]


def _posterior_probabilities_b64(posteriors: np.ndarray) -> str:
    post_u8 = np.clip(posteriors * 255, 0, 255).astype(np.uint8)
    return base64.b64encode(post_u8.tobytes()).decode("ascii")


def _comma_join_ints(values: Sequence[int]) -> str:
    return ",".join(map(str, values)) if len(values) > 0 else ""


def _region_posterior_fields_from_record(
    record: _RegionPosteriorLineRecord,
) -> list[str]:
    return [
        record.read_name,
        record.chrom,
        str(record.ref_start),
        str(record.ref_end),
        record.strand,
        _posterior_probabilities_b64(record.posteriors),
        _comma_join_ints(record.footprint_starts),
        _comma_join_ints(record.footprint_sizes),
    ]


def _region_posterior_fields(
    read_name: str,
    chrom: str,
    ref_start: int,
    ref_end: int,
    strand: str,
    posteriors: np.ndarray,
    footprint_starts: Sequence[int],
    footprint_sizes: Sequence[int],
) -> list[str]:
    return _region_posterior_fields_from_record(
        _RegionPosteriorLineRecord(
            read_name=read_name,
            chrom=chrom,
            ref_start=ref_start,
            ref_end=ref_end,
            strand=strand,
            posteriors=posteriors,
            footprint_starts=footprint_starts,
            footprint_sizes=footprint_sizes,
        )
    )


def format_region_posterior_line_from_record(
    record: _RegionPosteriorLineRecord,
) -> str:
    """Format one region-worker posterior record without a header."""
    fields = _region_posterior_fields_from_record(record)
    return "\t".join(fields) + "\n"


def format_region_posterior_line(
    read_name: str,
    chrom: str,
    ref_start: int,
    ref_end: int,
    strand: str,
    posteriors: np.ndarray,
    footprint_starts: Sequence[int],
    footprint_sizes: Sequence[int],
) -> str:
    """Format one region-worker posterior record without a header."""
    return format_region_posterior_line_from_record(
        _RegionPosteriorLineRecord(
            read_name=read_name,
            chrom=chrom,
            ref_start=ref_start,
            ref_end=ref_end,
            strand=strand,
            posteriors=posteriors,
            footprint_starts=footprint_starts,
            footprint_sizes=footprint_sizes,
        )
    )


def write_region_posteriors_tsv(tsv_path: str, posteriors_data: Iterable[dict]) -> None:
    """Write region-worker posterior records without metadata/header rows."""
    with open(tsv_path, "w") as handle:
        for fiber in posteriors_data:
            handle.write(
                format_region_posterior_line_from_record(
                    _RegionPosteriorLineRecord(
                        read_name=fiber["read_name"],
                        chrom=fiber["chrom"],
                        ref_start=fiber["ref_start"],
                        ref_end=fiber["ref_end"],
                        strand=fiber["strand"],
                        posteriors=fiber["posteriors"],
                        footprint_starts=fiber["footprint_starts"],
                        footprint_sizes=fiber["footprint_sizes"],
                    )
                )
            )


def region_posteriors_tsv_output_path(output_path: str) -> str:
    """Return the gzipped TSV path produced for a requested posterior path."""
    output_path = os.fspath(output_path)
    if region_posteriors_needs_h5_conversion(output_path):
        root, _ext = os.path.splitext(output_path)
        return root + ".tsv.gz"
    output_path_lower = output_path.lower()
    if output_path_lower.endswith(".tsv"):
        return output_path + ".gz"
    if output_path_lower.endswith(".tsv.gz"):
        return output_path
    return output_path + ".tsv.gz"


def region_posteriors_needs_h5_conversion(output_path: str) -> bool:
    return _resolve_writer_format(output_path, "auto") == "hdf5"


def _posterior_tsv_metadata(
    mode: str,
    context_size: int,
    edge_trim: int,
    source_bam: str,
) -> dict:
    return {
        "mode": mode,
        "context_size": context_size,
        "edge_trim": edge_trim,
        "source_bam": os.path.basename(os.fspath(source_bam)),
        "format_version": 1,
    }


def format_posterior_metadata_line(metadata: dict) -> str:
    return f"#metadata:{json.dumps(metadata)}\n"


def _valid_region_tsv_files(
    temp_tsv_files: Iterable[tuple[int, str]],
) -> list[_IndexedRegionTsvFile]:
    return [
        _IndexedRegionTsvFile(region_index=idx, path=path)
        for idx, path in sorted(temp_tsv_files, key=lambda item: item[0])
        if path_is_nonempty_file(path)
    ]


def _write_region_tsv_header(
    outfile,
    mode: str,
    context_size: int,
    edge_trim: int,
    source_bam: str,
) -> None:
    metadata = _posterior_tsv_metadata(
        mode,
        context_size,
        edge_trim,
        source_bam,
    )
    outfile.write(format_posterior_metadata_line(metadata))
    outfile.write(REGION_POSTERIORS_HEADER)


def _is_region_tsv_record_line(line: str) -> bool:
    return bool(line.strip()) and not line.startswith("#")


def _copy_region_tsv_records(outfile, tsv_path: str) -> int:
    n_fibers = 0
    with open(tsv_path, "r") as infile:
        for line in infile:
            if not _is_region_tsv_record_line(line):
                continue
            outfile.write(line)
            n_fibers += 1
    return n_fibers


def merge_region_posteriors_tsv(
    temp_tsv_files: Iterable[tuple[int, str]],
    output_path: str,
    mode: str,
    context_size: int,
    edge_trim: int,
    source_bam: str,
) -> int:
    """
    Merge region-worker TSV records into one gzipped TSV with metadata.

    H5 conversion is left as a separate step to avoid memory/parallel issues.
    """
    valid_files = _valid_region_tsv_files(temp_tsv_files)
    if not valid_files:
        return 0

    tsv_output = region_posteriors_tsv_output_path(output_path)
    with gzip.open(tsv_output, "wt", compresslevel=4) as outfile:
        _write_region_tsv_header(
            outfile, mode, context_size, edge_trim, source_bam,
        )

        n_fibers = 0
        for tsv_file in valid_files:
            n_fibers += _copy_region_tsv_records(outfile, tsv_file.path)

    return n_fibers
