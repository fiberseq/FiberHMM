"""FiberHMM region-parallel processing and worker management."""

import os
import sys
import re
import time
import base64
import json
import gzip
import shutil
import subprocess
import tempfile
import numpy as np
import pysam
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple, Set

from fiberhmm.core.model_io import load_model
from fiberhmm.core.bam_reader import encode_from_query_sequence, detect_daf_strand
from fiberhmm.inference.engine import (
    predict_footprints_and_msps,
    _extract_fiber_read_from_pysam,
    _process_single_read,
)
from fiberhmm.inference.bam_output import _sort_and_index_bam

# Optional: inline posteriors export
try:
    from fiberhmm.posteriors.hdf5_backend import PosteriorWriter, get_ref_positions_from_read
    HAS_POSTERIOR_WRITER = True
except ImportError:
    HAS_POSTERIOR_WRITER = False


# Global for worker processes
_worker_model = None
_worker_debug_timing = False
_worker_region_params = None


def _init_bam_worker(model_path, debug_timing=False):
    """Initialize worker process with model."""
    global _worker_model, _worker_debug_timing
    try:
        # Disable numba caching to avoid file lock contention between workers
        import os
        os.environ['NUMBA_CACHE_DIR'] = ''

        _worker_model = load_model(model_path)
        _worker_debug_timing = debug_timing

        # Warmup numba JIT compilation in this worker
        from fiberhmm.core.hmm import HAS_NUMBA
        if HAS_NUMBA:
            dummy_obs = np.array([0, 1, 2, 3], dtype=np.int64)
            _ = _worker_model.predict(dummy_obs)
    except Exception as e:
        import traceback
        print(f"Worker init error: {e}")
        traceback.print_exc()
        raise


def _is_main_chromosome(chrom: str) -> bool:
    """
    Check if a chromosome name is a main chromosome (not a scaffold/contig).

    Returns True for:
    - chr1-chr22, chrX, chrY, chrM, chrMT (human with chr prefix)
    - 1-22, X, Y, M, MT (human without chr prefix)
    - 2L, 2R, 3L, 3R, 4, X, Y (Drosophila)

    Returns False for:
    - *_random, chrUn_*, scaffolds, contigs, etc.
    """
    import re

    # Normalize to uppercase for comparison
    c = chrom.upper()

    # Skip obvious scaffolds/contigs
    skip_patterns = [
        '_RANDOM', '_ALT', '_FIX', '_HAP',
        'CHRUN_', 'UN_', 'SCAFFOLD', 'CONTIG',
        '_GL', '_KI', '_JH', '_KB'  # Common GenBank accession prefixes
    ]
    for pattern in skip_patterns:
        if pattern in c:
            return False

    # Strip chr prefix if present
    if c.startswith('CHR'):
        c = c[3:]

    # Accept numbered chromosomes 1-22 (or more for other organisms)
    if c.isdigit():
        return True

    # Accept X, Y, M, MT, W, Z (sex chromosomes and mitochondrial)
    if c in ('X', 'Y', 'M', 'MT', 'W', 'Z'):
        return True

    # Accept Drosophila chromosomes: 2L, 2R, 3L, 3R, 4
    if re.match(r'^[234][LR]?$', c):
        return True

    # Accept C. elegans chromosomes: I, II, III, IV, V, X
    if c in ('I', 'II', 'III', 'IV', 'V', 'VI'):
        return True

    return False


def _get_genome_regions(bam_path: str, region_size: int = 10_000_000,
                        skip_scaffolds: bool = False,
                        chroms: Optional[Set[str]] = None) -> List[Tuple[str, int, int]]:
    """
    Split genome into regions for parallel processing.

    Args:
        bam_path: Path to indexed BAM file
        region_size: Target size of each region in bp (default 10MB)
        skip_scaffolds: If True, skip scaffold/contig chromosomes
        chroms: If provided, only include these chromosomes

    Returns:
        List of (chrom, start, end) tuples
    """
    regions = []
    region_size = int(region_size)  # Ensure Python int

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for chrom in bam.references:
            # Filter by explicit chromosome list
            if chroms is not None and chrom not in chroms:
                continue

            # Filter scaffolds if requested
            if skip_scaffolds and not _is_main_chromosome(chrom):
                continue

            chrom_len = int(bam.get_reference_length(chrom))

            # Split chromosome into regions
            for start in range(0, chrom_len, region_size):
                end = min(start + region_size, chrom_len)
                regions.append((chrom, int(start), int(end)))

    return regions


def _init_region_worker(model_path: str, params: dict):
    """Initialize worker for region-parallel processing."""
    global _worker_model, _worker_region_params
    import os
    import numpy as np

    try:
        # Disable numba caching to avoid file lock contention
        os.environ['NUMBA_CACHE_DIR'] = ''

        # Load model once per worker
        _worker_model = load_model(model_path)
        _worker_region_params = params

        # Warmup numba JIT (just the basic predict, posteriors will warmup on first use)
        from fiberhmm.core.hmm import HAS_NUMBA
        if HAS_NUMBA:
            dummy_obs = np.array([0, 1, 2, 3], dtype=np.int64)
            _ = _worker_model.predict(dummy_obs)

    except Exception as e:
        import traceback
        print(f"Region worker init error: {e}")
        traceback.print_exc()
        raise


def _process_region_to_bam(args: Tuple) -> Tuple[str, int, int, int, Optional[str]]:
    """
    Worker function: process one genomic region and write to temp BAM.

    Each worker opens its own BAM file handle and uses the index to fetch
    reads from its assigned region. This enables true parallel I/O.

    Uses global _worker_model and _worker_region_params (set by _init_region_worker).

    Args:
        args: Tuple of (region, input_bam, temp_bam_path, temp_tsv_path or None)

    Returns:
        (temp_bam_path, total_reads, reads_with_footprints, written, temp_tsv_path or None)
    """
    import numpy as np  # Ensure numpy is available in worker
    import traceback
    import base64
    global _worker_model, _worker_region_params

    try:
        # Handle both old (3-tuple) and new (4-tuple) args
        if len(args) == 3:
            (chrom, start, end), input_bam, temp_bam_path = args
            temp_tsv_path = None
        else:
            (chrom, start, end), input_bam, temp_bam_path, temp_tsv_path = args

        # Ensure start/end are Python ints (not numpy)
        start = int(start)
        end = int(end)

        # Use global model and params (loaded once per worker)
        model = _worker_model
        params = _worker_region_params

        # Unpack parameters
        edge_trim = int(params['edge_trim'])
        circular = params['circular']
        mode = params['mode']
        context_size = int(params['context_size'])
        msp_min_size = int(params['msp_min_size'])
        min_mapq = int(params['min_mapq'])
        prob_threshold = int(params['prob_threshold'])
        min_read_length = int(params['min_read_length'])
        with_scores = params['with_scores']
        train_rids = params['train_rids']
        primary_only = params.get('primary_only', False)
        return_posteriors = params.get('return_posteriors', False) and temp_tsv_path is not None

        total_reads = 0
        reads_with_footprints = 0
        written = 0
        posteriors_written = 0

        # Suppress htslib warnings
        pysam.set_verbosity(0)

        # Open posteriors TSV file for streaming writes (if requested)
        tsv_file = None
        if return_posteriors and temp_tsv_path:
            try:
                tsv_file = open(temp_tsv_path, 'w')
            except Exception:
                return_posteriors = False  # Can't write, disable

        try:
            with pysam.AlignmentFile(input_bam, "rb") as inbam:
                with pysam.AlignmentFile(temp_bam_path, "wb", header=inbam.header) as outbam:

                    # Fetch reads from this region using the index
                    try:
                        read_iter = inbam.fetch(chrom, start, end)
                    except ValueError:
                        # Region not in BAM (e.g., unplaced contigs)
                        if tsv_file:
                            tsv_file.close()
                        return (temp_bam_path, 0, 0, 0, None)

                    for read in read_iter:
                        # Pass through unmapped reads (no sequence to process)
                        if read.is_unmapped:
                            outbam.write(read)
                            written += 1
                            continue

                        # Skip secondary/supplementary if primary_only mode
                        if primary_only and (read.is_secondary or read.is_supplementary):
                            outbam.write(read)
                            written += 1
                            continue

                        # Only process reads that START in this region to avoid duplicates
                        # (fetch returns reads that overlap the region)
                        if read.reference_start < start or read.reference_start >= end:
                            continue

                        # Check filters
                        if read.mapping_quality < min_mapq:
                            outbam.write(read)
                            written += 1
                            continue

                        if read.query_alignment_length is None or read.query_alignment_length < min_read_length:
                            outbam.write(read)
                            written += 1
                            continue

                        read_id = read.query_name
                        if train_rids and read_id in train_rids:
                            outbam.write(read)
                            written += 1
                            continue

                        # Extract data needed for processing
                        try:
                            fiber_read = _extract_fiber_read_from_pysam(read, mode, prob_threshold)
                            if fiber_read is None:
                                outbam.write(read)
                                written += 1
                                continue
                        except Exception:
                            outbam.write(read)
                            written += 1
                            continue

                        total_reads += 1

                        # Process the read (with posteriors if requested)
                        result = _process_single_read(
                            fiber_read, model, edge_trim, circular,
                            mode, context_size, msp_min_size, with_scores,
                            return_posteriors=return_posteriors
                        )

                        # Add tags to read
                        if result is not None:
                            reads_with_footprints += 1

                            # Only set tags if arrays are non-empty
                            if len(result['ns']) > 0:
                                read.set_tag('ns', result['ns'].astype(np.uint32).tolist())
                                read.set_tag('nl', result['nl'].astype(np.uint32).tolist())
                                if result['ns_scores'] is not None:
                                    # nq:B:C - unsigned 8-bit per fiberseq spec (0-255)
                                    scores_u8 = np.clip(np.array(result['ns_scores']) * 255, 0, 255).astype(np.uint8)
                                    read.set_tag('nq', scores_u8.tolist())

                            if len(result['as']) > 0:
                                read.set_tag('as', result['as'].astype(np.uint32).tolist())
                                read.set_tag('al', result['al'].astype(np.uint32).tolist())
                                if result['as_scores'] is not None:
                                    # aq:B:C - unsigned 8-bit per fiberseq spec (0-255)
                                    scores_u8 = np.clip(np.array(result['as_scores']) * 255, 0, 255).astype(np.uint8)
                                    read.set_tag('aq', scores_u8.tolist())

                            # Stream posteriors to TSV immediately (no memory accumulation)
                            if tsv_file and result.get('posteriors') is not None:
                                try:
                                    post_u8 = np.clip(result['posteriors'] * 255, 0, 255).astype(np.uint8)
                                    post_b64 = base64.b64encode(post_u8.tobytes()).decode('ascii')
                                    fp_starts_str = ','.join(map(str, result['ns'])) if len(result['ns']) > 0 else ''
                                    fp_sizes_str = ','.join(map(str, result['nl'])) if len(result['nl']) > 0 else ''
                                    strand = result.get('strand', '.')

                                    tsv_file.write(f"{read.query_name}\t{read.reference_name}\t"
                                                  f"{read.reference_start}\t{read.reference_end}\t"
                                                  f"{strand}\t{post_b64}\t{fp_starts_str}\t{fp_sizes_str}\n")
                                    posteriors_written += 1
                                except Exception:
                                    pass  # Don't crash on posteriors write failure

                        outbam.write(read)
                        written += 1

        finally:
            if tsv_file:
                tsv_file.close()

        # Return TSV path if we wrote any posteriors
        if return_posteriors and posteriors_written > 0 and temp_tsv_path:
            return (temp_bam_path, total_reads, reads_with_footprints, written, temp_tsv_path)

        return (temp_bam_path, total_reads, reads_with_footprints, written, None)

    except Exception as e:
        import traceback
        print(f"\nWorker error in region {chrom}:{start}-{end}: {e}")
        traceback.print_exc()
        raise


def _write_region_posteriors_tsv(tsv_path: str, posteriors_data: list):
    """Write posteriors data from a single region to TSV (simple append-friendly format)."""
    import base64

    with open(tsv_path, 'w') as f:
        for fiber in posteriors_data:
            # Quantize posteriors to uint8 and base64 encode
            post_u8 = np.clip(fiber['posteriors'] * 255, 0, 255).astype(np.uint8)
            post_b64 = base64.b64encode(post_u8.tobytes()).decode('ascii')

            # Encode footprint arrays
            fp_starts = fiber['footprint_starts']
            fp_sizes = fiber['footprint_sizes']
            fp_starts_str = ','.join(map(str, fp_starts)) if len(fp_starts) > 0 else ''
            fp_sizes_str = ','.join(map(str, fp_sizes)) if len(fp_sizes) > 0 else ''

            f.write(f"{fiber['read_name']}\t{fiber['chrom']}\t{fiber['ref_start']}\t"
                   f"{fiber['ref_end']}\t{fiber['strand']}\t{post_b64}\t"
                   f"{fp_starts_str}\t{fp_sizes_str}\n")


def _merge_region_posteriors_tsv(temp_tsv_files: list, output_path: str,
                                  mode: str, context_size: int, edge_trim: int,
                                  source_bam: str) -> int:
    """
    Merge multiple region TSV files into a single gzipped TSV.

    H5 conversion is left as a separate step to avoid memory/parallel issues.

    Args:
        temp_tsv_files: List of (region_idx, tsv_path) tuples
        output_path: Output path (will produce .tsv.gz regardless of extension)
        mode: HMM mode
        context_size: HMM context size
        edge_trim: Edge trim setting
        source_bam: Source BAM filename for metadata

    Returns:
        Total number of fibers merged
    """
    import json
    import gzip

    # Sort by region index to maintain genomic order
    temp_tsv_files.sort(key=lambda x: x[0])

    # Filter to existing files
    valid_files = [(idx, path) for idx, path in temp_tsv_files
                   if os.path.exists(path) and os.path.getsize(path) > 0]

    if not valid_files:
        return 0

    # Always output as gzipped TSV
    if output_path.endswith('.h5'):
        tsv_output = output_path.replace('.h5', '.tsv.gz')
    elif output_path.endswith('.tsv'):
        tsv_output = output_path + '.gz'
    elif output_path.endswith('.tsv.gz'):
        tsv_output = output_path
    else:
        tsv_output = output_path + '.tsv.gz'

    # Simple concatenation with gzip compression
    with gzip.open(tsv_output, 'wt', compresslevel=4) as outfile:
        # Write header
        metadata = {
            'mode': mode,
            'context_size': context_size,
            'edge_trim': edge_trim,
            'source_bam': os.path.basename(source_bam),
            'format_version': 1,
        }
        outfile.write(f"#metadata:{json.dumps(metadata)}\n")
        outfile.write("#read_id\tchrom\tstart\tend\tstrand\tposteriors_b64\tfp_starts\tfp_sizes\n")

        n_fibers = 0
        for region_idx, tsv_path in valid_files:
            with open(tsv_path, 'r') as infile:
                for line in infile:
                    outfile.write(line)
                    n_fibers += 1

    return n_fibers


def _process_region_to_bed(args: Tuple) -> Tuple[str, int, int]:
    """
    Process a genomic region and write BED output directly (no temp BAM).

    This is more space-efficient than _process_region_to_bam when only
    BED/bigBed output is needed.

    Args:
        args: Tuple of (region, input_bam, temp_bed_path)

    Returns:
        (temp_bed_path, total_reads, reads_with_footprints)
    """
    region, input_bam, temp_bed_path = args
    chrom, start, end = region

    try:
        start = int(start)
        end = int(end)

        model = _worker_model
        params = _worker_region_params

        edge_trim = int(params['edge_trim'])
        circular = params['circular']
        mode = params['mode']
        context_size = int(params['context_size'])
        msp_min_size = int(params['msp_min_size'])
        min_mapq = int(params['min_mapq'])
        prob_threshold = int(params['prob_threshold'])
        min_read_length = int(params['min_read_length'])
        with_scores = params['with_scores']
        train_rids = params['train_rids']

        total_reads = 0
        reads_with_footprints = 0

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

                    if read.query_alignment_length is None or read.query_alignment_length < min_read_length:
                        continue

                    read_id = read.query_name
                    if train_rids and read_id in train_rids:
                        continue

                    try:
                        fiber_read = _extract_fiber_read_from_pysam(read, mode, prob_threshold)
                        if fiber_read is None:
                            continue
                    except Exception:
                        continue

                    total_reads += 1

                    result = _process_single_read(
                        fiber_read, model, edge_trim, circular,
                        mode, context_size, msp_min_size, with_scores
                    )

                    if result is not None and len(result['ns']) > 0:
                        reads_with_footprints += 1

                        # Write BED12 line
                        ref_name = read.reference_name
                        ref_start = read.reference_start
                        ref_end = read.reference_end
                        strand = '-' if read.is_reverse else '+'
                        read_length = ref_end - ref_start

                        # Footprint blocks (convert to relative positions)
                        ns = result['ns']
                        nl = result['nl']
                        block_starts_list = [int(s - ref_start) for s in ns]
                        block_sizes_list = [int(l) for l in nl]

                        # Get scores if available
                        score_list = None
                        if with_scores and result['ns_scores'] is not None:
                            score_list = [int(s * 1000) for s in result['ns_scores']]

                        # BED12 requires blocks to span chromStart to chromEnd
                        # Add 1bp padding blocks at start/end if needed
                        if block_starts_list[0] != 0:
                            block_starts_list.insert(0, 0)
                            block_sizes_list.insert(0, 1)
                            if score_list is not None:
                                score_list.insert(0, 0)

                        last_end = block_starts_list[-1] + block_sizes_list[-1]
                        if last_end < read_length:
                            block_starts_list.append(read_length - 1)
                            block_sizes_list.append(1)
                            if score_list is not None:
                                score_list.append(0)

                        block_count = len(block_starts_list)
                        block_sizes = ','.join(str(s) for s in block_sizes_list)
                        block_starts = ','.join(str(s) for s in block_starts_list)

                        if score_list is not None:
                            scores = ','.join(str(s) for s in score_list)
                            bed_out.write(f"{ref_name}\t{ref_start}\t{ref_end}\t{read_id}\t0\t{strand}\t"
                                        f"{ref_start}\t{ref_end}\t0,0,0\t{block_count}\t{block_sizes}\t{block_starts}\t{scores}\n")
                        else:
                            bed_out.write(f"{ref_name}\t{ref_start}\t{ref_end}\t{read_id}\t0\t{strand}\t"
                                        f"{ref_start}\t{ref_end}\t0,0,0\t{block_count}\t{block_sizes}\t{block_starts}\n")

        return (temp_bed_path, total_reads, reads_with_footprints)

    except Exception as e:
        import traceback
        print(f"\nWorker error in region {chrom}:{start}-{end}: {e}")
        traceback.print_exc()
        raise


def _hierarchical_merge(bam_files: List[str], output_bam: str, temp_dir: str,
                        batch_size: int = 200) -> None:
    """
    Concatenate BAM files hierarchically to avoid 'too many open files' error.

    Uses samtools cat for fast concatenation (copies raw BGZF blocks).
    Falls back to pysam if samtools not available.
    """
    import shutil

    def _concat_bams_fast(inputs: List[str], output: str) -> None:
        """Fast concatenation using samtools cat, fallback to pysam."""
        try:
            # Write file list for samtools cat -b
            list_file = output + '.list'
            with open(list_file, 'w') as f:
                for bam_path in inputs:
                    f.write(bam_path + '\n')

            result = subprocess.run(
                ['samtools', 'cat', '-b', list_file, '-o', output],
                capture_output=True, text=True
            )
            os.remove(list_file)

            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, 'samtools cat', result.stderr)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to pysam
            with pysam.AlignmentFile(inputs[0], "rb") as first_bam:
                header = first_bam.header
            with pysam.AlignmentFile(output, "wb", header=header) as outbam:
                for bam_path in inputs:
                    with pysam.AlignmentFile(bam_path, "rb") as inbam:
                        for read in inbam:
                            outbam.write(read)

    if len(bam_files) == 0:
        raise ValueError("No BAM files to concatenate")

    if len(bam_files) == 1:
        shutil.copy(bam_files[0], output_bam)
        return

    if len(bam_files) <= batch_size:
        # Can concatenate directly
        _concat_bams_fast(bam_files, output_bam)
        return

    # Need hierarchical concatenation
    current_level = bam_files
    level_num = 0

    while len(current_level) > batch_size:
        next_level = []
        batch_num = 0

        for i in range(0, len(current_level), batch_size):
            batch = current_level[i:i + batch_size]

            if len(batch) == 1:
                # Single file, just pass through
                next_level.append(batch[0])
            else:
                # Concatenate this batch
                intermediate = os.path.join(temp_dir, f'cat_L{level_num}_B{batch_num}.bam')
                _concat_bams_fast(batch, intermediate)
                next_level.append(intermediate)
                batch_num += 1

        print(f"    Level {level_num}: {len(current_level)} -> {len(next_level)} files")
        current_level = next_level
        level_num += 1

    # Final concatenation
    if len(current_level) == 1:
        shutil.copy(current_level[0], output_bam)
    else:
        _concat_bams_fast(current_level, output_bam)


def _merge_bams_simple(bam_files: List[str], output_bam: str) -> None:
    """
    Concatenate BAM files using samtools cat (fast - copies raw BGZF blocks).

    Since regions are in genomic order and each BAM only contains reads
    that START in that region, we can concatenate without sorting.
    Falls back to pysam if samtools not available.
    """
    import shutil

    if len(bam_files) == 0:
        raise ValueError("No BAM files to concatenate")

    if len(bam_files) == 1:
        shutil.copy(bam_files[0], output_bam)
        return

    # Filter to existing non-empty files
    valid_files = [f for f in bam_files if os.path.exists(f) and os.path.getsize(f) > 0]

    if len(valid_files) == 0:
        raise ValueError("No valid BAM files to concatenate")
    elif len(valid_files) == 1:
        shutil.copy(valid_files[0], output_bam)
    else:
        # Use samtools cat for fast concatenation
        try:
            # Write file list for samtools cat -b
            list_file = output_bam + '.list'
            with open(list_file, 'w') as f:
                for bam_path in valid_files:
                    f.write(bam_path + '\n')

            result = subprocess.run(
                ['samtools', 'cat', '-b', list_file, '-o', output_bam],
                capture_output=True, text=True
            )
            os.remove(list_file)

            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, 'samtools cat', result.stderr)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to pysam
            with pysam.AlignmentFile(valid_files[0], "rb") as first_bam:
                header = first_bam.header

            with pysam.AlignmentFile(output_bam, "wb", header=header) as outbam:
                for bam_path in valid_files:
                    with pysam.AlignmentFile(bam_path, "rb") as inbam:
                        for read in inbam:
                            outbam.write(read)


def _process_bam_region_parallel(input_bam: str, output_bam: str,
                                   model_path: str, train_rids: Set[str],
                                   edge_trim: int, circular: bool,
                                   mode: str, context_size: int,
                                   msp_min_size: int,
                                   min_mapq: int, prob_threshold: int,
                                   min_read_length: int,
                                   with_scores: bool,
                                   n_cores: int,
                                   region_size: int = 10_000_000,
                                   skip_scaffolds: bool = False,
                                   chroms: Optional[Set[str]] = None,
                                   primary_only: bool = False,
                                   output_posteriors: Optional[str] = None) -> Tuple[int, int]:
    """
    Process BAM using region-based parallelism with indexed access.

    Each worker independently reads from the BAM using the index,
    enabling true parallel I/O. Results are concatenated at the end.

    Args:
        region_size: Size of each region in bp (default 10MB)
        skip_scaffolds: If True, skip scaffold/contig chromosomes
        chroms: If provided, only process these chromosomes
        primary_only: If True, skip secondary/supplementary alignments
        output_posteriors: If provided, write HMM posteriors to this H5 file

    Returns:
        (total_reads_processed, reads_with_footprints)
    """
    import tempfile
    import time
    import shutil

    start_time = time.time()
    return_posteriors = output_posteriors is not None

    # Check that BAM is indexed
    if not os.path.exists(input_bam + '.bai') and not os.path.exists(input_bam.replace('.bam', '.bai')):
        print("Indexing input BAM for region-parallel processing...")
        pysam.index(input_bam)

    # Get regions
    regions = _get_genome_regions(input_bam, region_size, skip_scaffolds, chroms)
    print(f"Processing {len(regions)} regions with {n_cores} cores...")
    if return_posteriors:
        print(f"Posteriors will be written to: {output_posteriors}")
    sys.stdout.flush()

    # Create temp directory in output folder for easier cleanup
    output_dir = os.path.dirname(os.path.abspath(output_bam))
    temp_dir = os.path.join(output_dir, '.fiberhmm_tmp')
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Prepare parameters (will be passed to initializer)
        params = {
            'edge_trim': edge_trim,
            'circular': circular,
            'mode': mode,
            'context_size': context_size,
            'msp_min_size': msp_min_size,
            'min_mapq': min_mapq,
            'prob_threshold': prob_threshold,
            'min_read_length': min_read_length,
            'with_scores': with_scores,
            'train_rids': train_rids,
            'primary_only': primary_only,
            'return_posteriors': return_posteriors,
        }

        # Work items - include temp H5 path if posteriors requested
        work_items = []
        for i, region in enumerate(regions):
            temp_bam = os.path.join(temp_dir, f'region_{i:06d}.bam')
            temp_h5 = os.path.join(temp_dir, f'region_{i:06d}.tsv') if return_posteriors else None
            work_items.append((region, input_bam, temp_bam, temp_h5))

        # Process regions in parallel
        total_reads = 0
        reads_with_footprints = 0
        temp_bams = []
        temp_h5s = []  # Collect temp H5 files for posteriors
        completed = 0
        first_result_time = None

        # Use initializer to load model once per worker
        print(f"  Initializing {n_cores} worker processes (loading HMM model in each)...")
        sys.stdout.flush()
        pool_start = time.time()

        with ProcessPoolExecutor(
            max_workers=n_cores,
            initializer=_init_region_worker,
            initargs=(model_path, params)
        ) as executor:
            futures = {executor.submit(_process_region_to_bam, item): i
                      for i, item in enumerate(work_items)}

            for future in as_completed(futures):
                try:
                    result = future.result()
                    # Handle both old (4-tuple) and new (5-tuple) returns
                    if len(result) == 5:
                        temp_bam, n_reads, n_fp, n_written, temp_h5 = result
                    else:
                        temp_bam, n_reads, n_fp, n_written = result
                        temp_h5 = None

                    # Track first result
                    if first_result_time is None:
                        first_result_time = time.time()
                        init_time = first_result_time - pool_start
                        print(f"  Workers ready ({init_time:.1f}s). Processing regions...")
                        sys.stdout.flush()

                    total_reads += n_reads
                    reads_with_footprints += n_fp
                    temp_bams.append((futures[future], temp_bam))
                    if temp_h5 and os.path.exists(temp_h5):
                        temp_h5s.append((futures[future], temp_h5))
                    completed += 1

                    elapsed = time.time() - start_time
                    rate = total_reads / elapsed if elapsed > 0 else 0
                    print(f"\r  Regions: {completed}/{len(regions)} | "
                          f"Reads: {total_reads:,} | "
                          f"With footprints: {reads_with_footprints:,} | "
                          f"{rate:.1f} reads/s", end='')
                    sys.stdout.flush()

                except Exception as e:
                    print(f"\nError processing region: {e}")
                    raise

        print()  # Newline after progress

        # Sort temp BAMs by region order and filter to non-empty
        temp_bams.sort(key=lambda x: x[0])
        non_empty_bams = [bam for _, bam in temp_bams
                         if os.path.exists(bam) and os.path.getsize(bam) > 0]

        # Calculate total size of temp BAMs
        total_temp_size = sum(os.path.getsize(b) for b in non_empty_bams)
        total_temp_size_gb = total_temp_size / (1024**3)

        # Concatenate BAMs using samtools cat (fast - copies raw BGZF blocks)
        print(f"Concatenating {len(non_empty_bams)} region BAMs ({total_temp_size_gb:.1f}GB total)...")
        sys.stdout.flush()
        concat_start = time.time()

        if len(non_empty_bams) == 0:
            with pysam.AlignmentFile(input_bam, "rb") as inbam:
                with pysam.AlignmentFile(output_bam, "wb", header=inbam.header) as outbam:
                    pass
        elif len(non_empty_bams) == 1:
            shutil.copy(non_empty_bams[0], output_bam)
        else:
            # Use samtools cat for fast concatenation
            # Files are already sorted by region index (region_000000.bam, region_000001.bam, etc.)
            try:
                # Write BAM list to file for samtools cat -b
                bam_list_file = os.path.join(temp_dir, 'bam_list.txt')
                with open(bam_list_file, 'w') as f:
                    for bam_path in non_empty_bams:
                        f.write(bam_path + '\n')

                result = subprocess.run(
                    ['samtools', 'cat', '-b', bam_list_file, '-o', output_bam],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    raise subprocess.CalledProcessError(result.returncode, 'samtools cat', result.stderr)

                concat_time = time.time() - concat_start
                output_size_gb = os.path.getsize(output_bam) / (1024**3)
                speed_gbs = output_size_gb / concat_time if concat_time > 0 else 0
                print(f"  Concatenated with samtools cat in {concat_time:.1f}s ({output_size_gb:.1f}GB, {speed_gbs:.2f} GB/s)")

            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                # Fallback to pysam if samtools not available or failed
                # Print the actual error message including stderr
                if hasattr(e, 'stderr') and e.stderr:
                    print(f"  WARNING: samtools cat failed: {e.stderr.strip()}")
                else:
                    print(f"  WARNING: samtools cat failed: {e}")
                print(f"  Falling back to pysam (slower, reads each record)...")

                # Remove any partial output file from failed samtools
                if os.path.exists(output_bam):
                    try:
                        os.remove(output_bam)
                        print(f"    Removed partial output file")
                    except Exception as rm_err:
                        print(f"    Warning: Could not remove partial file: {rm_err}")

                # Ensure output directory exists and is writable
                output_dir_path = os.path.dirname(output_bam)
                if output_dir_path:
                    if not os.path.exists(output_dir_path):
                        print(f"    Creating output directory: {output_dir_path}")
                        os.makedirs(output_dir_path, exist_ok=True)

                    # Test that we can write to this directory
                    test_file = os.path.join(output_dir_path, '.write_test')
                    try:
                        with open(test_file, 'w') as f:
                            f.write('test')
                        os.remove(test_file)
                    except Exception as write_err:
                        print(f"    ERROR: Cannot write to output directory: {write_err}")
                        print(f"    Directory: {output_dir_path}")
                        print(f"    Directory exists: {os.path.exists(output_dir_path)}")
                        raise

                try:
                    print(f"    Reading header from: {non_empty_bams[0]}")
                    with pysam.AlignmentFile(non_empty_bams[0], "rb") as first_bam:
                        header = first_bam.header

                    print(f"    Opening output file: {output_bam}")
                    with pysam.AlignmentFile(output_bam, "wb", header=header) as outbam:
                        for i, bam_path in enumerate(non_empty_bams):
                            with pysam.AlignmentFile(bam_path, "rb") as inbam:
                                for read in inbam:
                                    outbam.write(read)
                            if (i + 1) % 10 == 0:
                                print(f"\r    Concatenated {i+1}/{len(non_empty_bams)} BAMs...", end='')
                                sys.stdout.flush()
                    print()

                    concat_time = time.time() - concat_start
                    print(f"  Concatenated with pysam in {concat_time:.1f}s")

                except Exception as pysam_err:
                    print(f"  ERROR: pysam fallback also failed: {pysam_err}")
                    print(f"    Output path: {output_bam}")
                    print(f"    Output dir exists: {os.path.exists(os.path.dirname(output_bam))}")
                    print(f"  Attempting manual BAM concatenation via samtools merge...")

                    # Last resort: use samtools merge instead of cat
                    try:
                        # samtools merge can be more robust than cat
                        result = subprocess.run(
                            ['samtools', 'merge', '-f', '-b', bam_list_file, output_bam],
                            capture_output=True, text=True
                        )
                        if result.returncode == 0:
                            concat_time = time.time() - concat_start
                            print(f"  Concatenated with samtools merge in {concat_time:.1f}s")
                        else:
                            raise Exception(f"samtools merge failed: {result.stderr}")
                    except Exception as merge_err:
                        print(f"  ERROR: All concatenation methods failed: {merge_err}")
                        raise

        sys.stdout.flush()

        # Verify output was created
        if os.path.exists(output_bam):
            output_size_gb = os.path.getsize(output_bam) / (1024**3)
            print(f"Output BAM: {output_size_gb:.2f}GB")

        # Index the output BAM (sort first if needed)
        print("Step: Index/Sort...")
        _sort_and_index_bam(output_bam, threads=n_cores)

        # Merge temp TSV files if posteriors were requested
        if return_posteriors and temp_h5s:
            print(f"Merging {len(temp_h5s)} posterior files...")
            merge_start = time.time()
            n_fibers = _merge_region_posteriors_tsv(
                temp_h5s, output_posteriors, mode, context_size, edge_trim, input_bam
            )
            merge_time = time.time() - merge_start

            # Figure out actual output path (always .tsv.gz)
            if output_posteriors.endswith('.h5'):
                tsv_path = output_posteriors.replace('.h5', '.tsv.gz')
            elif output_posteriors.endswith('.tsv'):
                tsv_path = output_posteriors + '.gz'
            elif output_posteriors.endswith('.tsv.gz'):
                tsv_path = output_posteriors
            else:
                tsv_path = output_posteriors + '.tsv.gz'

            if os.path.exists(tsv_path):
                file_size = os.path.getsize(tsv_path) / (1024 * 1024)
                print(f"Posteriors: {n_fibers:,} fibers -> {tsv_path} ({file_size:.1f} MB, {merge_time:.1f}s)")
                if output_posteriors.endswith('.h5'):
                    print(f"  To convert to H5: python posteriors_io.py tsv2h5 {tsv_path} {output_posteriors}")

        elapsed = time.time() - start_time
        rate = total_reads / elapsed if elapsed > 0 else 0
        print(f"Completed: {total_reads:,} reads | {reads_with_footprints:,} with footprints | {rate:.1f} reads/s | {elapsed:.1f}s")

        return total_reads, reads_with_footprints

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def _process_bed_region_parallel(input_bam: str, output_bed: str,
                                  model_path: str, train_rids: Set[str],
                                  edge_trim: int, circular: bool,
                                  mode: str, context_size: int,
                                  msp_min_size: int,
                                  min_mapq: int, prob_threshold: int,
                                  min_read_length: int,
                                  with_scores: bool,
                                  n_cores: int,
                                  region_size: int = 10_000_000,
                                  skip_scaffolds: bool = False,
                                  chroms: Optional[Set[str]] = None,
                                  primary_only: bool = False) -> Tuple[int, int]:
    """
    Process BAM using region-based parallelism, writing BED output directly.

    This is more space-efficient than processing to BAM first when only
    BED/bigBed output is needed - no large temp BAMs are created.

    Returns:
        (total_reads_processed, reads_with_footprints)
    """
    import time

    start_time = time.time()

    # Check that BAM is indexed
    if not os.path.exists(input_bam + '.bai') and not os.path.exists(input_bam.replace('.bam', '.bai')):
        print("Indexing input BAM for region-parallel processing...")
        pysam.index(input_bam)

    # Get regions
    regions = _get_genome_regions(input_bam, region_size, skip_scaffolds, chroms)
    print(f"Processing {len(regions)} regions with {n_cores} cores (BED output)...")
    sys.stdout.flush()

    # Create temp directory for BED files (small compared to BAMs)
    output_dir = os.path.dirname(os.path.abspath(output_bed))
    temp_dir = os.path.join(output_dir, '.fiberhmm_tmp')
    os.makedirs(temp_dir, exist_ok=True)

    try:
        params = {
            'edge_trim': edge_trim,
            'circular': circular,
            'mode': mode,
            'context_size': context_size,
            'msp_min_size': msp_min_size,
            'min_mapq': min_mapq,
            'prob_threshold': prob_threshold,
            'min_read_length': min_read_length,
            'with_scores': with_scores,
            'train_rids': train_rids,
            'primary_only': primary_only
        }

        # Work items - write temp BED files
        work_items = []
        for i, region in enumerate(regions):
            temp_bed = os.path.join(temp_dir, f'region_{i:06d}.bed')
            work_items.append((region, input_bam, temp_bed))

        total_reads = 0
        reads_with_footprints = 0
        temp_beds = []
        completed = 0
        first_result_time = None

        print(f"  Initializing {n_cores} worker processes (loading HMM model in each)...")
        sys.stdout.flush()
        pool_start = time.time()

        with ProcessPoolExecutor(
            max_workers=n_cores,
            initializer=_init_region_worker,
            initargs=(model_path, params)
        ) as executor:
            futures = {executor.submit(_process_region_to_bed, item): i
                      for i, item in enumerate(work_items)}

            for future in as_completed(futures):
                try:
                    temp_bed, n_reads, n_fp = future.result()

                    # Track first result
                    if first_result_time is None:
                        first_result_time = time.time()
                        init_time = first_result_time - pool_start
                        print(f"  Workers ready ({init_time:.1f}s). Processing regions...")
                        sys.stdout.flush()

                    total_reads += n_reads
                    reads_with_footprints += n_fp
                    temp_beds.append((futures[future], temp_bed))
                    completed += 1

                    elapsed = time.time() - start_time
                    rate = total_reads / elapsed if elapsed > 0 else 0
                    print(f"\r  Regions: {completed}/{len(regions)} | "
                          f"Reads: {total_reads:,} | "
                          f"With footprints: {reads_with_footprints:,} | "
                          f"{rate:.1f} reads/s", end='')
                    sys.stdout.flush()

                except Exception as e:
                    print(f"\nError processing region: {e}")
                    raise

        print()  # Newline after progress

        # Sort temp BEDs by region order and concatenate
        temp_beds.sort(key=lambda x: x[0])
        non_empty_beds = [bed for _, bed in temp_beds
                         if os.path.exists(bed) and os.path.getsize(bed) > 0]

        print(f"Concatenating {len(non_empty_beds)} region BED files...")
        sys.stdout.flush()

        with open(output_bed, 'wb') as fout:
            for bed_path in non_empty_beds:
                with open(bed_path, 'rb') as fin:
                    shutil.copyfileobj(fin, fout)

        elapsed = time.time() - start_time
        rate = total_reads / elapsed if elapsed > 0 else 0
        print(f"Completed: {total_reads:,} reads | {reads_with_footprints:,} with footprints | {rate:.1f} reads/s | {elapsed:.1f}s")

        return total_reads, reads_with_footprints

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _process_and_write_chunk(chunk_reads: list, chunk_read_objs: list,
                              outbam, model, executor,
                              edge_trim: int, circular: bool,
                              mode: str, context_size: int,
                              msp_min_size: int, with_scores: bool,
                              return_posteriors: bool = False) -> Tuple[int, Optional[list]]:
    """
    Process a chunk of reads and write to BAM.

    Returns:
        (reads_with_footprints, results_with_posteriors or None)

        If return_posteriors=True, returns results list for caller to write posteriors.
    """

    if executor is not None:
        # Parallel: submit chunk to worker
        future = executor.submit(
            _process_chunk_worker,
            chunk_reads, edge_trim, circular, mode, context_size, msp_min_size,
            with_scores, return_posteriors
        )
        results = future.result()
    else:
        # Single-threaded: process directly
        results = []
        for fiber_read in chunk_reads:
            result = _process_single_read(
                fiber_read, model, edge_trim, circular,
                mode, context_size, msp_min_size, with_scores,
                return_posteriors=return_posteriors
            )
            results.append(result)

    # Write annotated reads
    reads_with_footprints = 0
    for read_obj, result in zip(chunk_read_objs, results):
        if result is not None:
            # Add footprint tags
            if len(result['ns']) > 0:
                read_obj.set_tag('ns', result['ns'].astype(np.uint32).tolist())
                read_obj.set_tag('nl', result['nl'].astype(np.uint32).tolist())
                if with_scores and result.get('ns_scores') is not None:
                    nq_scores = np.clip(result['ns_scores'] * 255, 0, 255).astype(np.uint8)
                    read_obj.set_tag('nq', nq_scores.tolist())

            if len(result['as']) > 0:
                read_obj.set_tag('as', result['as'].astype(np.uint32).tolist())
                read_obj.set_tag('al', result['al'].astype(np.uint32).tolist())
                if with_scores and result.get('as_scores') is not None:
                    aq_scores = np.clip(result['as_scores'] * 255, 0, 255).astype(np.uint8)
                    read_obj.set_tag('aq', aq_scores.tolist())

            reads_with_footprints += 1

        outbam.write(read_obj)

    # Return results for posteriors if requested
    if return_posteriors:
        return reads_with_footprints, list(zip(chunk_read_objs, chunk_reads, results))
    return reads_with_footprints, None


def _process_chunk_worker(chunk_reads: list, edge_trim: int, circular: bool,
                           mode: str, context_size: int, msp_min_size: int,
                           with_scores: bool, return_posteriors: bool = False) -> list:
    """Worker function to process a chunk of reads."""
    global _worker_model

    results = []
    for fiber_read in chunk_reads:
        result = _process_single_read(
            fiber_read, _worker_model, edge_trim, circular,
            mode, context_size, msp_min_size, with_scores,
            return_posteriors=return_posteriors
        )
        results.append(result)

    return results


def process_bam_for_footprints(input_bam: str, output_bam: str,
                                model_or_path, train_rids: Set[str],
                                edge_trim: int, circular: bool,
                                mode: str, context_size: int,
                                msp_min_size: int,
                                min_mapq: int, prob_threshold: int,
                                min_read_length: int,
                                with_scores: bool = False,
                                n_cores: int = 1,
                                max_reads: Optional[int] = None,
                                debug_timing: bool = False,
                                region_parallel: bool = False,
                                region_size: int = 10_000_000,
                                skip_scaffolds: bool = False,
                                chroms: Optional[Set[str]] = None,
                                primary_only: bool = False,
                                output_posteriors: Optional[str] = None) -> Tuple[int, int]:
    """
    Process BAM file and add footprint tags - SINGLE PASS STREAMING.

    Reads chunks from BAM, processes in parallel, writes immediately.
    Memory usage is bounded to one chunk at a time.

    Args:
        region_parallel: Use region-based parallelism (each worker reads BAM)
        primary_only: Only process primary alignments (skip secondary/supplementary)
        skip_scaffolds: If True, skip scaffold/contig chromosomes
        chroms: If provided, only process these chromosomes
        output_posteriors: If provided, write HMM posteriors to this H5 file (inline)

    Returns:
        (total_reads_processed, reads_with_footprints)
    """
    import time

    # Suppress htslib warnings
    pysam.set_verbosity(0)

    # Get model path for workers
    if isinstance(model_or_path, str):
        model_path = model_or_path
        model = load_model(model_path)
    else:
        model = model_or_path
        model_path = None

    # Dispatch to region-parallel if requested (requires model_path for workers)
    if region_parallel:
        if model_path is None:
            print("Warning: region_parallel requires model path, falling back to standard parallel")
        else:
            return _process_bam_region_parallel(
                input_bam=input_bam,
                output_bam=output_bam,
                model_path=model_path,
                train_rids=train_rids,
                edge_trim=edge_trim,
                circular=circular,
                mode=mode,
                context_size=context_size,
                msp_min_size=msp_min_size,
                min_mapq=min_mapq,
                prob_threshold=prob_threshold,
                min_read_length=min_read_length,
                with_scores=with_scores,
                n_cores=n_cores,
                region_size=region_size,
                skip_scaffolds=skip_scaffolds,
                chroms=chroms,
                primary_only=primary_only,
                output_posteriors=output_posteriors
            )

    total_reads = 0
    reads_with_footprints = 0
    written = 0
    skipped = 0

    # Track skip reasons
    skip_reasons = {
        'unmapped': 0,
        'secondary_supplementary': 0,
        'low_mapq': 0,
        'too_short': 0,
        'training_excluded': 0,
        'no_sequence': 0,
        'extraction_failed': 0,
    }

    chunk_size = 2000  # Reads per chunk
    start_time = time.time()

    # Initialize posteriors writer if requested
    posterior_writer = None
    return_posteriors = False
    if output_posteriors:
        if HAS_POSTERIOR_WRITER:
            posterior_writer = PosteriorWriter(
                output_posteriors, mode, context_size,
                edge_trim, input_bam, batch_size=1000
            )
            return_posteriors = True
            print(f"Posteriors will be written to: {output_posteriors}")
        else:
            print("WARNING: posterior_writer.py not found, skipping posteriors export")

    print("Processing and writing BAM (streaming)...")
    sys.stdout.flush()

    # Open input and output BAMs
    with pysam.AlignmentFile(input_bam, "rb") as inbam:
        with pysam.AlignmentFile(output_bam, "wb", header=inbam.header) as outbam:

            chunk_reads = []  # Buffer for current chunk
            chunk_read_objs = []  # Corresponding pysam read objects

            if n_cores > 1 and model_path:
                # Parallel processing with ProcessPoolExecutor
                executor = ProcessPoolExecutor(
                    max_workers=n_cores,
                    initializer=_init_bam_worker,
                    initargs=(model_path, debug_timing)
                )
            else:
                executor = None

            try:
                for read in inbam:
                    # Pass through unmapped reads (no sequence to process)
                    if read.is_unmapped:
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['unmapped'] += 1
                        continue

                    # Skip secondary/supplementary if primary_only mode
                    if primary_only and (read.is_secondary or read.is_supplementary):
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['secondary_supplementary'] += 1
                        continue

                    # Check filters (apply to all alignments)
                    if read.mapping_quality < min_mapq:
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['low_mapq'] += 1
                        continue

                    if read.query_alignment_length < min_read_length:
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['too_short'] += 1
                        continue

                    read_id = read.query_name
                    if train_rids and read_id in train_rids:
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['training_excluded'] += 1
                        continue

                    # Extract data needed for processing
                    try:
                        fiber_read = _extract_fiber_read_from_pysam(read, mode, prob_threshold)
                        if fiber_read is None:
                            outbam.write(read)
                            written += 1
                            skipped += 1
                            skip_reasons['no_sequence'] += 1
                            continue
                    except Exception:
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['extraction_failed'] += 1
                        continue

                    chunk_reads.append(fiber_read)
                    chunk_read_objs.append(read)
                    total_reads += 1

                    # Check max reads limit
                    if max_reads and total_reads >= max_reads:
                        # Process final chunk and exit
                        break

                    # Process chunk when full
                    if len(chunk_reads) >= chunk_size:
                        n_fp, chunk_results = _process_and_write_chunk(
                            chunk_reads, chunk_read_objs, outbam,
                            model, executor, edge_trim, circular,
                            mode, context_size, msp_min_size, with_scores,
                            return_posteriors=return_posteriors
                        )
                        reads_with_footprints += n_fp
                        written += len(chunk_read_objs)

                        # Write posteriors if requested
                        if posterior_writer and chunk_results:
                            for read_obj, fiber_read_data, result in chunk_results:
                                if result and result.get('posteriors') is not None:
                                    chrom = read_obj.reference_name
                                    if chrom:
                                        # Get ref_positions from the read object
                                        ref_positions = get_ref_positions_from_read(read_obj) if HAS_POSTERIOR_WRITER else np.array([], dtype=np.int32)
                                        posterior_writer.add_fiber(chrom, {
                                            'read_name': read_obj.query_name,
                                            'ref_start': read_obj.reference_start,
                                            'ref_end': read_obj.reference_end,
                                            'strand': result.get('strand', '.'),
                                            'posteriors': result['posteriors'],
                                            'ref_positions': ref_positions,
                                            'footprint_starts': result['ns'],
                                            'footprint_sizes': result['nl'],
                                        })

                        # Progress update
                        elapsed = time.time() - start_time
                        rate = total_reads / elapsed if elapsed > 0 else 0
                        print(f"\r  Processed: {total_reads:,} | Skipped: {skipped:,} | {rate:.1f} reads/s", end='')
                        sys.stdout.flush()

                        chunk_reads = []
                        chunk_read_objs = []

                # Process final partial chunk
                if chunk_reads:
                    n_fp, chunk_results = _process_and_write_chunk(
                        chunk_reads, chunk_read_objs, outbam,
                        model, executor, edge_trim, circular,
                        mode, context_size, msp_min_size, with_scores,
                        return_posteriors=return_posteriors
                    )
                    reads_with_footprints += n_fp
                    written += len(chunk_read_objs)

                    # Write posteriors if requested
                    if posterior_writer and chunk_results:
                        for read_obj, fiber_read_data, result in chunk_results:
                            if result and result.get('posteriors') is not None:
                                chrom = read_obj.reference_name
                                if chrom:
                                    ref_positions = get_ref_positions_from_read(read_obj) if HAS_POSTERIOR_WRITER else np.array([], dtype=np.int32)
                                    posterior_writer.add_fiber(chrom, {
                                        'read_name': read_obj.query_name,
                                        'ref_start': read_obj.reference_start,
                                        'ref_end': read_obj.reference_end,
                                        'strand': result.get('strand', '.'),
                                        'posteriors': result['posteriors'],
                                        'ref_positions': ref_positions,
                                        'footprint_starts': result['ns'],
                                        'footprint_sizes': result['nl'],
                                    })

            finally:
                if executor:
                    executor.shutdown(wait=True)

    elapsed = time.time() - start_time
    rate = total_reads / elapsed if elapsed > 0 else 0
    print(f"\r  Processed: {total_reads:,} | Skipped: {skipped:,} | With footprints: {reads_with_footprints:,} | {rate:.1f} reads/s")

    # Print skip reasons summary
    if skipped > 0:
        print("  Skip reasons:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = 100 * count / (total_reads + skipped)
                print(f"    {reason}: {count:,} ({pct:.1f}%)")

    # Index the output BAM (sort first if needed)
    _sort_and_index_bam(output_bam, threads=n_cores)

    # Close posteriors writer and report
    if posterior_writer:
        n_fibers, file_size = posterior_writer.close()
        print(f"Posteriors: {n_fibers:,} fibers -> {output_posteriors} ({file_size:.1f} MB)")

    return total_reads, reads_with_footprints
