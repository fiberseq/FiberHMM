#!/usr/bin/env python3
"""
extract_tags.py - Extract tags from FiberHMM-tagged BAMs to BED12/bigBed.

Extracts various tag types from tagged BAM files using region-parallel processing:
  - footprint: Nucleosome footprints (ns/nl tags) -> BED12 (one line per read)
  - msp: Methylase-sensitive patches (as/al tags) -> BED12 (one line per read)
  - m6a: m6A modification positions (from MM/ML tags) -> BED12 (one line per read)
  - m5c: 5mC modification positions (for DAF-seq) -> BED12 (one line per read)

Output files are named: {dataset}_{type}.bb (bigBed by default)

Usage:
    # Default: extract all types to bigBed in same directory as input
    python extract_tags.py -i tagged.bam

    # Extract only footprints
    python extract_tags.py -i tagged.bam --footprint

    # Extract to specific directory with 8 cores
    python extract_tags.py -i tagged.bam -o output/ -c 8

    # Keep BED files (in addition to bigBed)
    python extract_tags.py -i tagged.bam --keep-bed
"""

import argparse
import os
import sys
import time
import tempfile
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple, Set
import pysam
import numpy as np

from fiberhmm.inference.parallel import _get_genome_regions


def get_chrom_sizes(bam_path: str) -> Dict[str, int]:
    """Extract chromosome sizes from BAM header."""
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        return {sq['SN']: sq['LN'] for sq in bam.header['SQ']}


# Global worker state
_worker_params = None

def _init_extract_worker(params: dict):
    """Initialize worker with parameters."""
    global _worker_params
    _worker_params = params


def _extract_region_worker(args) -> Tuple[str, int, int]:
    """
    Worker to extract tags from a BAM region to a temp BED file.

    Returns: (temp_bed_path, n_reads, n_features)
    """
    global _worker_params

    try:
        (chrom, start, end), input_bam, temp_bed_path = args

        start = int(start)
        end = int(end)

        params = _worker_params
        extract_type = params['extract_type']
        min_mapq = params['min_mapq']
        prob_threshold = params['prob_threshold']
        with_scores = params['with_scores']

        n_reads = 0
        n_features = 0

        pysam.set_verbosity(0)

        with pysam.AlignmentFile(input_bam, "rb") as inbam:
            with open(temp_bed_path, 'w') as bed_out:
                try:
                    read_iter = inbam.fetch(chrom, start, end)
                except ValueError:
                    return (temp_bed_path, 0, 0)

                for read in read_iter:
                    # Skip unmapped/secondary/supplementary
                    if read.is_unmapped or read.is_secondary or read.is_supplementary:
                        continue

                    # Only process reads that START in this region
                    if read.reference_start < start or read.reference_start >= end:
                        continue

                    if read.mapping_quality < min_mapq:
                        continue

                    n_reads += 1

                    if extract_type == 'footprint':
                        n_features += _extract_footprints(read, bed_out, with_scores)
                    elif extract_type == 'msp':
                        n_features += _extract_msps(read, bed_out, with_scores)
                    elif extract_type == 'm6a':
                        n_features += _extract_m6a(read, bed_out, prob_threshold)
                    elif extract_type == 'm5c':
                        n_features += _extract_m5c(read, bed_out, prob_threshold)

        return (temp_bed_path, n_reads, n_features)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (temp_bed_path, 0, 0)


def _extract_footprints(read, bed_out, with_scores: bool) -> int:
    """Extract footprint intervals from ns/nl tags as BED12 format (one line per read)."""
    try:
        ns = read.get_tag('ns')  # Footprint starts (query coords)
        nl = read.get_tag('nl')  # Footprint lengths
    except KeyError:
        return 0

    if len(ns) == 0:
        return 0

    # Get scores if available
    scores = None
    if with_scores:
        try:
            scores = read.get_tag('nq')  # Footprint quality scores
        except KeyError:
            pass

    ref_name = read.reference_name
    strand = '-' if read.is_reverse else '+'
    read_id = read.query_name

    # Convert query positions to reference positions
    query_to_ref = {}
    for query_pos, ref_pos in read.get_aligned_pairs(matches_only=False):
        if query_pos is not None and ref_pos is not None:
            query_to_ref[query_pos] = ref_pos

    # Collect all footprint blocks
    blocks = []  # list of (ref_start, ref_end, score)
    for i, (qstart, length) in enumerate(zip(ns, nl)):
        qend = qstart + length

        # Get reference positions
        ref_start = query_to_ref.get(qstart)
        ref_end = query_to_ref.get(qend - 1)

        if ref_start is None or ref_end is None:
            continue

        ref_start, ref_end = min(ref_start, ref_end), max(ref_start, ref_end) + 1

        score = 0
        if scores is not None and i < len(scores):
            score = int(scores[i])

        blocks.append((ref_start, ref_end, score))

    if not blocks:
        return 0

    # Sort blocks by position
    blocks.sort(key=lambda x: x[0])

    # BED12 format
    chrom_start = blocks[0][0]
    chrom_end = blocks[-1][1]
    block_count = len(blocks)
    block_sizes = ','.join(str(e - s) for s, e, _ in blocks)
    block_starts = ','.join(str(s - chrom_start) for s, _, _ in blocks)

    # Use mean score for the read
    mean_score = int(sum(sc for _, _, sc in blocks) / len(blocks)) if blocks else 0

    # BED12: chrom, chromStart, chromEnd, name, score, strand, thickStart, thickEnd, itemRgb, blockCount, blockSizes, blockStarts
    bed_out.write(f"{ref_name}\t{chrom_start}\t{chrom_end}\t{read_id}\t{mean_score}\t{strand}\t{chrom_start}\t{chrom_end}\t0\t{block_count}\t{block_sizes}\t{block_starts}\n")

    return len(blocks)


def _extract_msps(read, bed_out, with_scores: bool) -> int:
    """Extract MSP intervals from as/al tags as BED12 format (one line per read)."""
    try:
        as_starts = read.get_tag('as')  # MSP starts (query coords)
        al_lengths = read.get_tag('al')  # MSP lengths
    except KeyError:
        return 0

    if len(as_starts) == 0:
        return 0

    # Get scores if available
    scores = None
    if with_scores:
        try:
            scores = read.get_tag('aq')  # MSP quality scores
        except KeyError:
            pass

    ref_name = read.reference_name
    strand = '-' if read.is_reverse else '+'
    read_id = read.query_name

    # Convert query positions to reference positions
    query_to_ref = {}
    for query_pos, ref_pos in read.get_aligned_pairs(matches_only=False):
        if query_pos is not None and ref_pos is not None:
            query_to_ref[query_pos] = ref_pos

    # Collect all MSP blocks
    blocks = []  # list of (ref_start, ref_end, score)
    for i, (qstart, length) in enumerate(zip(as_starts, al_lengths)):
        qend = qstart + length

        # Get reference positions
        ref_start = query_to_ref.get(qstart)
        ref_end = query_to_ref.get(qend - 1)

        if ref_start is None or ref_end is None:
            continue

        ref_start, ref_end = min(ref_start, ref_end), max(ref_start, ref_end) + 1

        score = 0
        if scores is not None and i < len(scores):
            score = int(scores[i])

        blocks.append((ref_start, ref_end, score))

    if not blocks:
        return 0

    # Sort blocks by position
    blocks.sort(key=lambda x: x[0])

    # BED12 format
    chrom_start = blocks[0][0]
    chrom_end = blocks[-1][1]
    block_count = len(blocks)
    block_sizes = ','.join(str(e - s) for s, e, _ in blocks)
    block_starts = ','.join(str(s - chrom_start) for s, _, _ in blocks)

    # Use mean score for the read
    mean_score = int(sum(sc for _, _, sc in blocks) / len(blocks)) if blocks else 0

    # BED12: chrom, chromStart, chromEnd, name, score, strand, thickStart, thickEnd, itemRgb, blockCount, blockSizes, blockStarts
    bed_out.write(f"{ref_name}\t{chrom_start}\t{chrom_end}\t{read_id}\t{mean_score}\t{strand}\t{chrom_start}\t{chrom_end}\t0\t{block_count}\t{block_sizes}\t{block_starts}\n")

    return len(blocks)


def _extract_m6a(read, bed_out, prob_threshold: int) -> int:
    """Extract m6A positions from MM/ML tags as BED12 format (one line per read)."""
    try:
        mod_bases = read.modified_bases
        if not mod_bases:
            return 0
    except (KeyError, TypeError):
        return 0

    ref_name = read.reference_name
    strand = '-' if read.is_reverse else '+'
    read_id = read.query_name

    # Get aligned pairs for coordinate conversion
    aligned_pairs = dict((q, r) for q, r in read.get_aligned_pairs() if q is not None and r is not None)

    # Collect all m6A positions for this read
    positions_list = []  # list of (ref_pos, score)

    # Look for m6A modifications
    for (base, strand_code, mod_type), positions in mod_bases.items():
        if mod_type != 'a':  # m6A
            continue

        for query_pos, prob in positions:
            if prob < prob_threshold:
                continue

            ref_pos = aligned_pairs.get(query_pos)
            if ref_pos is None:
                continue

            score = int(prob)  # Keep as 0-255
            positions_list.append((ref_pos, score))

    if not positions_list:
        return 0

    # Sort by position
    positions_list.sort(key=lambda x: x[0])

    # BED12 format - each modification is a 1bp block
    chrom_start = positions_list[0][0]
    chrom_end = positions_list[-1][0] + 1
    block_count = len(positions_list)
    block_sizes = ','.join('1' for _ in positions_list)
    block_starts = ','.join(str(pos - chrom_start) for pos, _ in positions_list)

    # Use mean score
    mean_score = int(sum(sc for _, sc in positions_list) / len(positions_list))

    bed_out.write(f"{ref_name}\t{chrom_start}\t{chrom_end}\t{read_id}\t{mean_score}\t{strand}\t{chrom_start}\t{chrom_end}\t0\t{block_count}\t{block_sizes}\t{block_starts}\n")

    return len(positions_list)


def _extract_m5c(read, bed_out, prob_threshold: int) -> int:
    """Extract 5mC positions from MM/ML tags as BED12 format (one line per read)."""
    try:
        mod_bases = read.modified_bases
        if not mod_bases:
            return 0
    except (KeyError, TypeError):
        return 0

    ref_name = read.reference_name
    strand = '-' if read.is_reverse else '+'
    read_id = read.query_name

    # Get aligned pairs for coordinate conversion
    aligned_pairs = dict((q, r) for q, r in read.get_aligned_pairs() if q is not None and r is not None)

    # Collect all 5mC positions for this read
    positions_list = []  # list of (ref_pos, score)

    # Look for 5mC modifications
    for (base, strand_code, mod_type), positions in mod_bases.items():
        if mod_type != 'm':  # 5mC
            continue

        for query_pos, prob in positions:
            if prob < prob_threshold:
                continue

            ref_pos = aligned_pairs.get(query_pos)
            if ref_pos is None:
                continue

            score = int(prob)  # Keep as 0-255
            positions_list.append((ref_pos, score))

    if not positions_list:
        return 0

    # Sort by position
    positions_list.sort(key=lambda x: x[0])

    # BED12 format - each modification is a 1bp block
    chrom_start = positions_list[0][0]
    chrom_end = positions_list[-1][0] + 1
    block_count = len(positions_list)
    block_sizes = ','.join('1' for _ in positions_list)
    block_starts = ','.join(str(pos - chrom_start) for pos, _ in positions_list)

    # Use mean score
    mean_score = int(sum(sc for _, sc in positions_list) / len(positions_list))

    bed_out.write(f"{ref_name}\t{chrom_start}\t{chrom_end}\t{read_id}\t{mean_score}\t{strand}\t{chrom_start}\t{chrom_end}\t0\t{block_count}\t{block_sizes}\t{block_starts}\n")

    return len(positions_list)


def extract_tags_parallel(input_bam: str, output_bed: str, extract_type: str,
                          n_cores: int = 1, region_size: int = 10_000_000,
                          min_mapq: int = 0, prob_threshold: int = 125,
                          with_scores: bool = True,
                          skip_scaffolds: bool = False,
                          chroms: Optional[Set[str]] = None) -> Tuple[int, int]:
    """
    Extract tags from BAM using region-parallel processing.

    Returns: (n_reads_processed, n_features_extracted)
    """
    start_time = time.time()

    # Check BAM index
    if not os.path.exists(input_bam + '.bai') and not os.path.exists(input_bam.replace('.bam', '.bai')):
        print("Indexing input BAM...")
        pysam.index(input_bam)

    # Get regions
    regions = _get_genome_regions(input_bam, region_size, skip_scaffolds, chroms)
    print(f"Processing {len(regions)} regions with {n_cores} cores...")

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix='extract_tags_')

    try:
        params = {
            'extract_type': extract_type,
            'min_mapq': min_mapq,
            'prob_threshold': prob_threshold,
            'with_scores': with_scores,
        }

        # Work items
        work_items = []
        for i, region in enumerate(regions):
            temp_bed = os.path.join(temp_dir, f'region_{i:06d}.bed')
            work_items.append((region, input_bam, temp_bed))

        total_reads = 0
        total_features = 0
        temp_beds = []
        completed = 0

        with ProcessPoolExecutor(
            max_workers=n_cores,
            initializer=_init_extract_worker,
            initargs=(params,)
        ) as executor:
            futures = {executor.submit(_extract_region_worker, item): i
                      for i, item in enumerate(work_items)}

            for future in as_completed(futures):
                completed += 1

                try:
                    temp_bed, n_reads, n_features = future.result()
                    total_reads += n_reads
                    total_features += n_features

                    if os.path.exists(temp_bed) and os.path.getsize(temp_bed) > 0:
                        temp_beds.append((futures[future], temp_bed))

                except Exception as e:
                    print(f"Worker error: {e}")

                # Progress
                elapsed = time.time() - start_time
                rate = total_reads / elapsed if elapsed > 0 else 0
                print(f"\r  Regions: {completed}/{len(regions)} | Reads: {total_reads:,} | Features: {total_features:,} | {rate:.0f} reads/s", end='')

        print()

        # Sort temp_beds by region order and concatenate
        temp_beds.sort(key=lambda x: x[0])

        print("Concatenating results...")
        with open(output_bed, 'w') as outf:
            for _, temp_bed in temp_beds:
                with open(temp_bed, 'r') as inf:
                    shutil.copyfileobj(inf, outf)

        # Sort the output BED
        print("Sorting BED...")
        sorted_bed = output_bed + '.sorted'
        subprocess.run(['sort', '-k1,1', '-k2,2n', output_bed, '-o', sorted_bed], check=True)
        os.replace(sorted_bed, output_bed)

        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.1f}s: {total_reads:,} reads -> {total_features:,} features")

        return total_reads, total_features

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def bed_to_bigbed(bed_path: str, bigbed_path: str, chrom_sizes: Dict[str, int], bed_type: str = 'bed12') -> bool:
    """Convert BED to bigBed using bedToBigBed.

    Args:
        bed_path: Input BED file
        bigbed_path: Output bigBed file
        chrom_sizes: Dict of chromosome sizes
        bed_type: 'bed12' for all FiberHMM outputs
    """
    # Write chrom sizes file
    sizes_file = bed_path + '.sizes'
    with open(sizes_file, 'w') as f:
        for chrom, size in sorted(chrom_sizes.items()):
            f.write(f"{chrom}\t{size}\n")

    try:
        # Run bedToBigBed
        cmd = ['bedToBigBed']
        if bed_type == 'bed12':
            cmd.extend(['-type=bed12'])
        cmd.extend([bed_path, sizes_file, bigbed_path])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"bedToBigBed error: {result.stderr}")
            return False

        return True

    except FileNotFoundError:
        print("Warning: bedToBigBed not found in PATH. Skipping bigBed conversion.")
        print("Install with: conda install -c bioconda ucsc-bedtobigbed")
        return False

    finally:
        if os.path.exists(sizes_file):
            try:
                os.remove(sizes_file)
            except (PermissionError, OSError) as e:
                print(f"  Warning: Could not remove temp file {sizes_file}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract tags from FiberHMM-tagged BAMs to BED12/bigBed',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: extract all types to bigBed in same directory as input
    python extract_tags.py -i tagged.bam

    # Extract only footprints
    python extract_tags.py -i tagged.bam --footprint

    # Extract to specific directory with 8 cores
    python extract_tags.py -i tagged.bam -o output/ -c 8

    # Keep BED files (in addition to bigBed)
    python extract_tags.py -i tagged.bam --keep-bed
        """
    )

    parser.add_argument('-i', '--input', required=True, help='Input tagged BAM file')
    parser.add_argument('-o', '--outdir', default=None, help='Output directory (default: same as input)')
    parser.add_argument('-c', '--cores', type=int, default=1, help='Number of CPU cores')

    # Tag types (default: all)
    parser.add_argument('--footprint', action='store_true', help='Extract footprints (ns/nl tags)')
    parser.add_argument('--msp', action='store_true', help='Extract MSPs (as/al tags)')
    parser.add_argument('--m6a', action='store_true', help='Extract m6A positions')
    parser.add_argument('--m5c', action='store_true', help='Extract 5mC positions (DAF-seq)')
    parser.add_argument('--all', action='store_true', help='Extract all tag types (default if none specified)')

    # Output options (default: bigbed)
    parser.add_argument('--bed-only', action='store_true', help='Output BED only (no bigBed)')
    parser.add_argument('--keep-bed', action='store_true', help='Keep BED files when creating bigBed')
    # Legacy flag for compatibility
    parser.add_argument('--bigbed', action='store_true', help=argparse.SUPPRESS)

    # Filtering
    parser.add_argument('-q', '--min-mapq', type=int, default=0, help='Min mapping quality (default: 0, no filtering)')
    parser.add_argument('-p', '--prob-threshold', type=int, default=125,
                        help='Min probability for m6a/m5c (0-255)')
    parser.add_argument('--no-scores', action='store_true', help='Omit scores from output')

    # Region options
    parser.add_argument('--region-size', type=int, default=10_000_000, help='Region size for parallel')
    parser.add_argument('--skip-scaffolds', action='store_true', help='Skip scaffold chromosomes')
    parser.add_argument('--chroms', type=str, help='Comma-separated chromosomes to process')

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Default output directory to input file's directory
    if args.outdir is None:
        args.outdir = os.path.dirname(os.path.abspath(args.input))
        if not args.outdir:
            args.outdir = '.'

    # Determine what to extract (default: all)
    extract_types = []
    if args.all or not (args.footprint or args.msp or args.m6a or args.m5c):
        extract_types = ['footprint', 'msp', 'm6a', 'm5c']
    else:
        if args.footprint:
            extract_types.append('footprint')
        if args.msp:
            extract_types.append('msp')
        if args.m6a:
            extract_types.append('m6a')
        if args.m5c:
            extract_types.append('m5c')

    # Default to bigbed unless --bed-only specified
    make_bigbed = not args.bed_only

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Parse chromosome filter
    chroms = None
    if args.chroms:
        chroms = set(args.chroms.split(','))

    # Get dataset name
    dataset = os.path.basename(args.input).replace('.bam', '').replace('_footprints', '')

    # Get chrom sizes for bigBed
    chrom_sizes = get_chrom_sizes(args.input)

    print(f"Input: {args.input}")
    print(f"Output: {args.outdir}")
    print(f"Extract types: {', '.join(extract_types)}")
    print(f"Cores: {args.cores}")
    print()

    # Extract each type
    for extract_type in extract_types:
        print(f"=== Extracting {extract_type} ===")

        bed_path = os.path.join(args.outdir, f"{dataset}_{extract_type}.bed")
        bb_path = os.path.join(args.outdir, f"{dataset}_{extract_type}.bb")

        n_reads, n_features = extract_tags_parallel(
            input_bam=args.input,
            output_bed=bed_path,
            extract_type=extract_type,
            n_cores=args.cores,
            region_size=args.region_size,
            min_mapq=args.min_mapq,
            prob_threshold=args.prob_threshold,
            with_scores=not args.no_scores,
            skip_scaffolds=args.skip_scaffolds,
            chroms=chroms
        )

        if n_features == 0:
            print(f"  No {extract_type} features found, skipping.")
            if os.path.exists(bed_path):
                try:
                    os.remove(bed_path)
                except (PermissionError, OSError):
                    pass  # File locked, leave it
            continue

        print(f"  BED: {bed_path}")

        # Convert to bigBed (default behavior)
        if make_bigbed:
            print("  Converting to bigBed...")
            # All outputs are BED12 format
            if bed_to_bigbed(bed_path, bb_path, chrom_sizes, 'bed12'):
                print(f"  bigBed: {bb_path}")
                if not args.keep_bed:
                    try:
                        os.remove(bed_path)
                    except (PermissionError, OSError) as e:
                        print(f"  Warning: Could not remove BED file (keeping it): {e}")

        print()

    print("Done!")


if __name__ == '__main__':
    main()
