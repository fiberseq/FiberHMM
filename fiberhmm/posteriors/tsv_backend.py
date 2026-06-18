#!/usr/bin/env python3
"""
Simple posteriors I/O for FiberHMM.

Format: TSV with one line per read
  read_id, chrom, start, end, strand, posteriors_b64, fp_starts, fp_sizes

Posteriors are quantized to uint8 (0-255) and base64 encoded for compactness.
File can be gzipped for additional compression.

Usage:
    # Convert TSV to H5
    python -m fiberhmm.posteriors.tsv_backend tsv2h5 posteriors.tsv.gz posteriors.h5

    # Or as library
    from fiberhmm.posteriors.tsv_backend import PosteriorsTSVWriter, tsv_to_h5
"""

import argparse
import base64
import gzip
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional, TextIO

import numpy as np

from fiberhmm.io.path_status import path_is_regular_file, path_size_mb
from fiberhmm.posteriors.region_tsv import (
    REGION_POSTERIORS_HEADER,
    _posterior_tsv_metadata,
    format_posterior_metadata_line,
    format_region_posterior_line,
)


@dataclass(frozen=True)
class _TsvH5ScanResult:
    metadata: dict
    chrom_counts: dict
    total_fibers: int


@dataclass(frozen=True)
class _TsvCopyResult:
    copied_fibers: int
    header_written: bool


@dataclass(frozen=True)
class _PosteriorTsvFields:
    read_id: str
    chrom: str
    start: int
    end: int
    strand: str
    post_b64: str
    fp_starts: str
    fp_sizes: str

    def as_tuple(self) -> tuple:
        return (
            self.read_id,
            self.chrom,
            self.start,
            self.end,
            self.strand,
            self.post_b64,
            self.fp_starts,
            self.fp_sizes,
        )


@dataclass(frozen=True)
class _PosteriorTsvRecord:
    read_id: str
    chrom: str
    start: int
    end: int
    strand: str
    posteriors: np.ndarray
    fp_starts: np.ndarray
    fp_sizes: np.ndarray


@dataclass(frozen=True)
class _H5PosteriorRecordDatasetSpec:
    group_name: str
    data: np.ndarray
    compression_opts: Optional[int]


def _open_text_file(path: str, mode: str) -> TextIO:
    """Open plain or gzip-compressed text files with consistent settings."""
    path = os.fspath(path)
    if path.lower().endswith('.gz'):
        kwargs = {}
        if any(flag in mode for flag in ('w', 'a', 'x')):
            kwargs['compresslevel'] = 4
        return gzip.open(path, mode, **kwargs)
    return open(path, mode)


def _split_posteriors_line(line: str):
    if line.startswith('#'):
        return None

    parts = line.strip().split('\t')
    if len(parts) < 6:
        return None

    read_id, chrom, start, end, strand, post_b64 = parts[:6]
    fp_starts_str = parts[6] if len(parts) > 6 else ''
    fp_sizes_str = parts[7] if len(parts) > 7 else ''
    return _PosteriorTsvFields(
        read_id=read_id,
        chrom=chrom,
        start=int(start),
        end=int(end),
        strand=strand,
        post_b64=post_b64,
        fp_starts=fp_starts_str,
        fp_sizes=fp_sizes_str,
    )


def _decode_posteriors_b64(post_b64: str, dtype) -> np.ndarray:
    post_bytes = base64.b64decode(post_b64)
    return np.frombuffer(post_bytes, dtype=np.uint8).astype(dtype) / 255.0


def _parse_int_array(values: str) -> np.ndarray:
    return np.array(
        [int(item) for item in (x.strip() for x in values.split(',')) if item],
        dtype=np.int32,
    )


def _chrom_from_countable_tsv_line(line: str) -> Optional[str]:
    if line.startswith('#'):
        return None

    parts = line.strip().split('\t')
    if len(parts) < 2:
        return None
    return parts[1]


def _metadata_from_tsv_line(line: str) -> Optional[dict]:
    if not line.startswith('#metadata:'):
        return None
    return json.loads(line.split(':', 1)[1])


def _posterior_record_from_fields(fields, dtype) -> _PosteriorTsvRecord:
    return _PosteriorTsvRecord(
        read_id=fields.read_id,
        chrom=fields.chrom,
        start=fields.start,
        end=fields.end,
        strand=fields.strand,
        posteriors=_decode_posteriors_b64(fields.post_b64, dtype),
        fp_starts=_parse_int_array(fields.fp_starts),
        fp_sizes=_parse_int_array(fields.fp_sizes),
    )


def _posterior_record_dict(record: _PosteriorTsvRecord) -> dict:
    return {
        'read_id': record.read_id,
        'chrom': record.chrom,
        'start': record.start,
        'end': record.end,
        'strand': record.strand,
        'posteriors': record.posteriors,
        'fp_starts': record.fp_starts,
        'fp_sizes': record.fp_sizes,
    }


def _iter_tsv_posterior_fields(tsv_path: str):
    with _open_text_file(tsv_path, 'rt') as infile:
        for line in infile:
            fields = _split_posteriors_line(line)
            if fields is not None:
                yield fields


def _scan_tsv_for_h5(tsv_path: str, verbose: bool):
    if verbose:
        print(f"  Scanning {tsv_path}...")

    metadata = {}
    chrom_counts = {}
    n_total = 0

    with _open_text_file(tsv_path, 'rt') as infile:
        for line in infile:
            line_metadata = _metadata_from_tsv_line(line)
            if line_metadata is not None:
                metadata = line_metadata
                continue

            chrom = _chrom_from_countable_tsv_line(line)
            if chrom is None:
                continue

            chrom_counts[chrom] = chrom_counts.get(chrom, 0) + 1
            n_total += 1

            if verbose and n_total % 500000 == 0:
                print(f"\r    Counted {n_total:,} fibers...", end='')
                sys.stdout.flush()

    if verbose:
        print(f"\r    Found {n_total:,} fibers across {len(chrom_counts)} chromosomes")

    return _TsvH5ScanResult(
        metadata=metadata,
        chrom_counts=chrom_counts,
        total_fibers=n_total,
    )


def _create_h5_chrom_groups(h5_file, chrom_counts, string_dtype):
    from fiberhmm.posteriors.hdf5_backend import create_posterior_chrom_group

    chrom_indices = {}
    for chrom, count in chrom_counts.items():
        grp = create_posterior_chrom_group(
            h5_file,
            chrom,
            include_ref_positions=False,
        )
        grp.attrs['n_fibers'] = count
        chrom_indices[chrom] = 0

        # Pre-allocate metadata arrays
        grp.create_dataset('fiber_ids', shape=(count,), dtype=string_dtype)
        grp.create_dataset('fiber_starts', shape=(count,), dtype=np.int32)
        grp.create_dataset('fiber_ends', shape=(count,), dtype=np.int32)
        grp.create_dataset('strands', shape=(count,), dtype=string_dtype)
    return chrom_indices


def _write_h5_metadata_from_tsv_metadata(h5_file, metadata: dict) -> None:
    from fiberhmm.posteriors.hdf5_backend import write_hdf5_file_metadata

    write_hdf5_file_metadata(
        h5_file,
        mode=metadata.get('mode', 'pacbio-fiber'),
        context_size=metadata.get('context_size', 3),
        edge_trim=metadata.get('edge_trim', 10),
        source_bam=metadata.get('source_bam', ''),
    )


def _h5_posterior_record_dataset_specs(
    record: _PosteriorTsvRecord,
) -> list[_H5PosteriorRecordDatasetSpec]:
    return [
        _H5PosteriorRecordDatasetSpec(
            group_name='posteriors',
            data=record.posteriors,
            compression_opts=4,
        ),
        _H5PosteriorRecordDatasetSpec(
            group_name='footprint_starts',
            data=record.fp_starts,
            compression_opts=None,
        ),
        _H5PosteriorRecordDatasetSpec(
            group_name='footprint_sizes',
            data=record.fp_sizes,
            compression_opts=None,
        ),
    ]


def _write_h5_record_array_datasets(
    group,
    index: int,
    record: _PosteriorTsvRecord,
) -> None:
    idx = str(index)
    for spec in _h5_posterior_record_dataset_specs(record):
        kwargs = {'compression': 'gzip'}
        if spec.compression_opts is not None:
            kwargs['compression_opts'] = spec.compression_opts
        group[spec.group_name].create_dataset(idx, data=spec.data, **kwargs)


def _write_h5_record_metadata(
    group,
    index: int,
    record: _PosteriorTsvRecord,
) -> None:
    group['fiber_ids'][index] = record.read_id
    group['fiber_starts'][index] = record.start
    group['fiber_ends'][index] = record.end
    group['strands'][index] = record.strand


def _write_h5_posterior_record(h5_file, chrom_indices, fields) -> None:
    record = _posterior_record_from_fields(fields, np.float16)
    chrom = record.chrom

    # Get index for this chromosome
    idx = chrom_indices[chrom]
    chrom_indices[chrom] += 1

    grp = h5_file[chrom]

    _write_h5_record_array_datasets(grp, idx, record)
    _write_h5_record_metadata(grp, idx, record)


def _posterior_tsv_output_path(output_path: str, compress: bool) -> str:
    output_path = os.fspath(output_path)
    is_gzip_path = output_path.lower().endswith('.gz')
    if compress or is_gzip_path:
        return output_path if is_gzip_path else output_path + '.gz'
    return output_path


class PosteriorsTSVWriter:
    """
    Simple streaming writer for posteriors in TSV format.

    Thread-safe for append-only writes. Each line is independent.
    Can write to gzipped file for compression.
    """

    def __init__(self, output_path: str, mode: str = 'pacbio-fiber',
                 context_size: int = 3, edge_trim: int = 10,
                 source_bam: str = '', compress: bool = True):
        """
        Args:
            output_path: Path to output file (.tsv or .tsv.gz)
            mode: HMM mode for metadata
            context_size: HMM context size
            edge_trim: Edge trim setting
            source_bam: Source BAM filename
            compress: If True, gzip compress the output
        """
        output_path = os.fspath(output_path)
        self.output_path = output_path
        self.compress = compress or output_path.lower().endswith('.gz')
        self.output_path = _posterior_tsv_output_path(output_path, compress)
        self._file = _open_text_file(self.output_path, 'wt')
        self._closed = False

        # Write header with metadata
        metadata = _posterior_tsv_metadata(mode, context_size, edge_trim, source_bam)
        self._file.write(format_posterior_metadata_line(metadata))
        self._file.write(REGION_POSTERIORS_HEADER)

        self.n_written = 0

    def write_fiber(self, read_id: str, chrom: str, start: int, end: int,
                    strand: str, posteriors: np.ndarray,
                    fp_starts: np.ndarray, fp_sizes: np.ndarray):
        """
        Write one fiber's posteriors.

        Args:
            read_id: Read name
            chrom: Chromosome
            start: Reference start
            end: Reference end
            strand: Strand (+, -, or .)
            posteriors: P(footprint) array, float values 0-1
            fp_starts: Footprint start positions (query coords)
            fp_sizes: Footprint sizes
        """
        if self._closed:
            raise RuntimeError("PosteriorsTSVWriter is closed")

        self._file.write(
            format_region_posterior_line(
                read_name=read_id,
                chrom=chrom,
                ref_start=start,
                ref_end=end,
                strand=strand,
                posteriors=posteriors,
                footprint_starts=fp_starts,
                footprint_sizes=fp_sizes,
            )
        )
        self.n_written += 1

    def flush(self):
        """Flush buffered writes."""
        self._file.flush()

    def close(self):
        """Close the file."""
        if self._closed:
            return self.n_written
        try:
            self._file.close()
        finally:
            self._closed = True
        return self.n_written

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def parse_posteriors_line(line: str) -> Optional[dict]:
    """Parse a single line from posteriors TSV."""
    fields = _split_posteriors_line(line)
    if fields is None:
        return None

    return _posterior_record_dict(
        _posterior_record_from_fields(fields, np.float32)
    )


def tsv_to_h5(tsv_path: str, h5_path: str, verbose: bool = True) -> int:
    """
    Convert posteriors TSV to HDF5 format.

    Uses streaming approach to avoid loading all data into memory.

    Args:
        tsv_path: Input TSV file (can be .gz)
        h5_path: Output HDF5 file
        verbose: Print progress

    Returns:
        Number of fibers converted
    """
    import h5py

    # First pass: count fibers per chromosome
    scan_result = _scan_tsv_for_h5(tsv_path, verbose)

    # Create H5 file with pre-allocated structure
    if verbose:
        print(f"  Writing {h5_path}...")

    with h5py.File(h5_path, 'w') as f:
        _write_h5_metadata_from_tsv_metadata(f, scan_result.metadata)

        dt = h5py.special_dtype(vlen=str)

        chrom_indices = _create_h5_chrom_groups(
            f, scan_result.chrom_counts, dt,
        )

        n_written = 0

        for fields in _iter_tsv_posterior_fields(tsv_path):
            _write_h5_posterior_record(f, chrom_indices, fields)
            n_written += 1

            if verbose and n_written % 100000 == 0:
                print(
                    f"\r    Written {n_written:,} / "
                    f"{scan_result.total_fibers:,} fibers...",
                    end='',
                )
                sys.stdout.flush()

    file_size = path_size_mb(h5_path)
    if verbose:
        print(f"\r    Done: {n_written:,} fibers -> {h5_path} ({file_size:.1f} MB)")

    return n_written


def _copy_tsv_records(
    inpath: str,
    outfile,
    header_written: bool,
) -> _TsvCopyResult:
    n_fibers = 0
    with _open_text_file(inpath, 'rt') as infile:
        for line in infile:
            if not line.strip():
                continue
            if line.startswith('#'):
                # Only write header from first file
                if not header_written:
                    outfile.write(line)
                    header_written = True
            else:
                outfile.write(line)
                n_fibers += 1

    return _TsvCopyResult(
        copied_fibers=n_fibers,
        header_written=header_written,
    )


def _remove_concatenated_tsv_inputs(input_files: list[str]) -> None:
    for inpath in input_files:
        os.remove(inpath)


def concatenate_tsvs(input_files: list, output_path: str,
                     delete_inputs: bool = False) -> int:
    """
    Concatenate multiple posteriors TSV files.

    Args:
        input_files: List of input TSV paths (can be .gz)
        output_path: Output path (will be gzipped if ends in .gz)
        delete_inputs: If True, delete input files after concatenating

    Returns:
        Total number of fibers
    """
    n_fibers = 0
    header_written = False
    copied_inputs = []

    with _open_text_file(output_path, 'wt') as outfile:
        for inpath in input_files:
            if not path_is_regular_file(inpath):
                continue

            copy_result = _copy_tsv_records(
                inpath, outfile, header_written,
            )
            n_fibers += copy_result.copied_fibers
            header_written = copy_result.header_written
            copied_inputs.append(inpath)

    if delete_inputs:
        _remove_concatenated_tsv_inputs(copied_inputs)

    return n_fibers


def main():
    parser = argparse.ArgumentParser(description='Posteriors TSV/H5 utilities')
    subparsers = parser.add_subparsers(dest='command', help='Command')

    # tsv2h5 command
    p_convert = subparsers.add_parser('tsv2h5', help='Convert TSV to H5')
    p_convert.add_argument('input', help='Input TSV file (.tsv or .tsv.gz)')
    p_convert.add_argument('output', help='Output H5 file')
    p_convert.add_argument('-q', '--quiet', action='store_true', help='Quiet mode')

    # concat command
    p_concat = subparsers.add_parser('concat', help='Concatenate TSV files')
    p_concat.add_argument('output', help='Output TSV file')
    p_concat.add_argument('inputs', nargs='+', help='Input TSV files')
    p_concat.add_argument('--delete', action='store_true', help='Delete inputs after')

    args = parser.parse_args()

    if args.command == 'tsv2h5':
        n = tsv_to_h5(args.input, args.output, verbose=not args.quiet)
        print(f"Converted {n:,} fibers")

    elif args.command == 'concat':
        n = concatenate_tsvs(args.inputs, args.output, delete_inputs=args.delete)
        print(f"Concatenated {n:,} fibers -> {args.output}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
