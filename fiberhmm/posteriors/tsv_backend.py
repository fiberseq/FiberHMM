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

import os
import sys
import gzip
import base64
import argparse
import numpy as np
from typing import Optional, BinaryIO, TextIO, Union
import json


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
        self.output_path = output_path
        self.compress = compress or output_path.endswith('.gz')

        # Open file
        if self.compress:
            if not output_path.endswith('.gz'):
                self.output_path = output_path + '.gz'
            self._file = gzip.open(self.output_path, 'wt', compresslevel=4)
        else:
            self._file = open(output_path, 'w')

        # Write header with metadata
        metadata = {
            'mode': mode,
            'context_size': context_size,
            'edge_trim': edge_trim,
            'source_bam': os.path.basename(source_bam),
            'format_version': 1,
        }
        self._file.write(f"#metadata:{json.dumps(metadata)}\n")
        self._file.write("#read_id\tchrom\tstart\tend\tstrand\tposteriors_b64\tfp_starts\tfp_sizes\n")

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
        # Quantize posteriors to uint8
        post_u8 = np.clip(posteriors * 255, 0, 255).astype(np.uint8)
        post_b64 = base64.b64encode(post_u8.tobytes()).decode('ascii')

        # Encode footprint arrays as comma-separated
        fp_starts_str = ','.join(map(str, fp_starts)) if len(fp_starts) > 0 else ''
        fp_sizes_str = ','.join(map(str, fp_sizes)) if len(fp_sizes) > 0 else ''

        line = f"{read_id}\t{chrom}\t{start}\t{end}\t{strand}\t{post_b64}\t{fp_starts_str}\t{fp_sizes_str}\n"
        self._file.write(line)
        self.n_written += 1

    def flush(self):
        """Flush buffered writes."""
        self._file.flush()

    def close(self):
        """Close the file."""
        self._file.close()
        return self.n_written

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def parse_posteriors_line(line: str) -> Optional[dict]:
    """Parse a single line from posteriors TSV."""
    if line.startswith('#'):
        return None

    parts = line.strip().split('\t')
    if len(parts) < 6:
        return None

    read_id, chrom, start, end, strand, post_b64 = parts[:6]
    fp_starts_str = parts[6] if len(parts) > 6 else ''
    fp_sizes_str = parts[7] if len(parts) > 7 else ''

    # Decode posteriors
    post_bytes = base64.b64decode(post_b64)
    posteriors = np.frombuffer(post_bytes, dtype=np.uint8).astype(np.float32) / 255.0

    # Parse footprint arrays
    fp_starts = np.array([int(x) for x in fp_starts_str.split(',') if x], dtype=np.int32)
    fp_sizes = np.array([int(x) for x in fp_sizes_str.split(',') if x], dtype=np.int32)

    return {
        'read_id': read_id,
        'chrom': chrom,
        'start': int(start),
        'end': int(end),
        'strand': strand,
        'posteriors': posteriors,
        'fp_starts': fp_starts,
        'fp_sizes': fp_sizes,
    }


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
    if verbose:
        print(f"  Scanning {tsv_path}...")

    # Open input
    if tsv_path.endswith('.gz'):
        infile = gzip.open(tsv_path, 'rt')
    else:
        infile = open(tsv_path, 'r')

    # Parse metadata from header
    metadata = {}
    first_line = infile.readline()
    if first_line.startswith('#metadata:'):
        metadata = json.loads(first_line.split(':', 1)[1])

    # Skip column header
    for line in infile:
        if not line.startswith('#'):
            break

    # Count fibers per chromosome
    chrom_counts = {}
    n_total = 0

    # We already consumed one data line, process it
    if line and not line.startswith('#'):
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            chrom = parts[1]
            chrom_counts[chrom] = chrom_counts.get(chrom, 0) + 1
            n_total += 1

    for line in infile:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            chrom = parts[1]
            chrom_counts[chrom] = chrom_counts.get(chrom, 0) + 1
            n_total += 1

        if verbose and n_total % 500000 == 0:
            print(f"\r    Counted {n_total:,} fibers...", end='')
            sys.stdout.flush()

    infile.close()

    if verbose:
        print(f"\r    Found {n_total:,} fibers across {len(chrom_counts)} chromosomes")

    # Create H5 file with pre-allocated structure
    if verbose:
        print(f"  Writing {h5_path}...")

    with h5py.File(h5_path, 'w') as f:
        # Metadata
        f.attrs['mode'] = metadata.get('mode', 'pacbio-fiber')
        f.attrs['context_size'] = metadata.get('context_size', 3)
        f.attrs['edge_trim'] = metadata.get('edge_trim', 10)
        f.attrs['source_bam'] = metadata.get('source_bam', '')
        f.attrs['format_version'] = 2

        dt = h5py.special_dtype(vlen=str)

        # Pre-create chromosome groups
        chrom_indices = {}  # Track current index per chromosome
        for chrom, count in chrom_counts.items():
            grp = f.create_group(chrom)
            grp.create_group('posteriors')
            grp.create_group('footprint_starts')
            grp.create_group('footprint_sizes')
            grp.attrs['n_fibers'] = count
            chrom_indices[chrom] = 0

            # Pre-allocate metadata arrays
            grp.create_dataset('fiber_ids', shape=(count,), dtype=dt)
            grp.create_dataset('fiber_starts', shape=(count,), dtype=np.int32)
            grp.create_dataset('fiber_ends', shape=(count,), dtype=np.int32)
            grp.create_dataset('strands', shape=(count,), dtype=dt)

        # Second pass: stream data into H5
        if tsv_path.endswith('.gz'):
            infile = gzip.open(tsv_path, 'rt')
        else:
            infile = open(tsv_path, 'r')

        # Skip headers
        for line in infile:
            if not line.startswith('#'):
                break

        n_written = 0

        # Process first data line
        def process_line(line):
            nonlocal n_written
            parts = line.strip().split('\t')
            if len(parts) < 6:
                return

            read_id, chrom, start, end, strand, post_b64 = parts[:6]
            fp_starts_str = parts[6] if len(parts) > 6 else ''
            fp_sizes_str = parts[7] if len(parts) > 7 else ''

            # Decode posteriors
            post_bytes = base64.b64decode(post_b64)
            posteriors = np.frombuffer(post_bytes, dtype=np.uint8).astype(np.float16) / 255.0

            # Parse footprint arrays
            fp_starts = np.array([int(x) for x in fp_starts_str.split(',') if x], dtype=np.int32)
            fp_sizes = np.array([int(x) for x in fp_sizes_str.split(',') if x], dtype=np.int32)

            # Get index for this chromosome
            idx = chrom_indices[chrom]
            chrom_indices[chrom] += 1

            grp = f[chrom]

            # Write data
            grp['posteriors'].create_dataset(
                str(idx),
                data=posteriors,
                compression='gzip', compression_opts=4
            )
            grp['footprint_starts'].create_dataset(
                str(idx),
                data=fp_starts,
                compression='gzip'
            )
            grp['footprint_sizes'].create_dataset(
                str(idx),
                data=fp_sizes,
                compression='gzip'
            )

            grp['fiber_ids'][idx] = read_id
            grp['fiber_starts'][idx] = int(start)
            grp['fiber_ends'][idx] = int(end)
            grp['strands'][idx] = strand

            n_written += 1

        # Process the first line we already read
        if line and not line.startswith('#'):
            process_line(line)

        # Process rest
        for line in infile:
            process_line(line)

            if verbose and n_written % 100000 == 0:
                print(f"\r    Written {n_written:,} / {n_total:,} fibers...", end='')
                sys.stdout.flush()

        infile.close()

    file_size = os.path.getsize(h5_path) / (1024 * 1024)
    if verbose:
        print(f"\r    Done: {n_written:,} fibers -> {h5_path} ({file_size:.1f} MB)")

    return n_written


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
    compress = output_path.endswith('.gz')

    if compress:
        outfile = gzip.open(output_path, 'wt', compresslevel=4)
    else:
        outfile = open(output_path, 'w')

    n_fibers = 0
    header_written = False

    for i, inpath in enumerate(input_files):
        if not os.path.exists(inpath):
            continue

        if inpath.endswith('.gz'):
            infile = gzip.open(inpath, 'rt')
        else:
            infile = open(inpath, 'r')

        for line in infile:
            if line.startswith('#'):
                # Only write header from first file
                if not header_written:
                    outfile.write(line)
            else:
                outfile.write(line)
                n_fibers += 1

        infile.close()
        header_written = True

        if delete_inputs:
            os.remove(inpath)

    outfile.close()
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
