"""TSV helpers for region-parallel posterior export."""

from __future__ import annotations

import base64
import gzip
import json
import os
from typing import Iterable, Sequence

import numpy as np

from fiberhmm.posteriors.writer import _resolve_writer_format

REGION_POSTERIORS_HEADER = (
    "#read_id\tchrom\tstart\tend\tstrand\tposteriors_b64\tfp_starts\tfp_sizes\n"
)


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
    post_u8 = np.clip(posteriors * 255, 0, 255).astype(np.uint8)
    post_b64 = base64.b64encode(post_u8.tobytes()).decode("ascii")
    fp_starts_str = ",".join(map(str, footprint_starts)) if len(footprint_starts) > 0 else ""
    fp_sizes_str = ",".join(map(str, footprint_sizes)) if len(footprint_sizes) > 0 else ""

    return (
        f"{read_name}\t{chrom}\t{ref_start}\t{ref_end}\t{strand}\t"
        f"{post_b64}\t{fp_starts_str}\t{fp_sizes_str}\n"
    )


def write_region_posteriors_tsv(tsv_path: str, posteriors_data: Iterable[dict]) -> None:
    """Write region-worker posterior records without metadata/header rows."""
    with open(tsv_path, "w") as handle:
        for fiber in posteriors_data:
            handle.write(
                format_region_posterior_line(
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


def region_posteriors_tsv_output_path(output_path: str) -> str:
    """Return the gzipped TSV path produced for a requested posterior path."""
    if _resolve_writer_format(output_path, "auto") == "hdf5":
        root, _ext = os.path.splitext(output_path)
        return root + ".tsv.gz"
    if output_path.endswith(".tsv"):
        return output_path + ".gz"
    if output_path.endswith(".tsv.gz"):
        return output_path
    return output_path + ".tsv.gz"


def region_posteriors_needs_h5_conversion(output_path: str) -> bool:
    return _resolve_writer_format(output_path, "auto") == "hdf5"


def _region_posteriors_metadata(
    mode: str,
    context_size: int,
    edge_trim: int,
    source_bam: str,
) -> dict:
    return {
        "mode": mode,
        "context_size": context_size,
        "edge_trim": edge_trim,
        "source_bam": os.path.basename(source_bam),
        "format_version": 1,
    }


def _valid_region_tsv_files(temp_tsv_files: Iterable[tuple[int, str]]) -> list[tuple[int, str]]:
    return [
        (idx, path)
        for idx, path in sorted(temp_tsv_files, key=lambda item: item[0])
        if os.path.exists(path) and os.path.getsize(path) > 0
    ]


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
        metadata = _region_posteriors_metadata(
            mode,
            context_size,
            edge_trim,
            source_bam,
        )
        outfile.write(f"#metadata:{json.dumps(metadata)}\n")
        outfile.write(REGION_POSTERIORS_HEADER)

        n_fibers = 0
        for _region_idx, tsv_path in valid_files:
            with open(tsv_path, "r") as infile:
                for line in infile:
                    outfile.write(line)
                    n_fibers += 1

    return n_fibers
