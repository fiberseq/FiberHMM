"""FiberHMM region-parallel processing and worker management."""

import base64
import gzip
import json
import multiprocessing
import os
import re
import shutil
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from typing import List, Optional, Set, Tuple

import numpy as np
import pysam

from fiberhmm.core.model_io import freeze_model_for_inference, load_model
from fiberhmm.inference.bam_output import (
    _concatenate_region_bams,
    _sort_and_index_bam,
)
from fiberhmm.inference.engine import (
    _extract_fiber_read_from_pysam,
    _process_single_read,
    extract_fiber_read_from_payload,
    make_apply_payload,
)
from fiberhmm.inference.fused_stages import (
    apply_result_has_footprints,
    build_fused_recall_result,
    run_hmm_apply_stage,
)
from fiberhmm.inference.read_filters import ReadFilterConfig, streaming_skip_reason
from fiberhmm.inference.region_types import (
    RegionBamAggregation,
    RegionBamResult,
    RegionBamWorkItem,
    RegionBedAggregation,
    RegionBedResult,
    RegionBedWorkItem,
)
from fiberhmm.inference.tagging import (
    set_legacy_apply_tags,
    write_fused_recall_tags,
)
from fiberhmm.inference.worker_results import WorkerChunkResult, coerce_worker_chunk_result

# Optional: inline posteriors export
try:
    from fiberhmm.posteriors.hdf5_backend import PosteriorWriter, get_ref_positions_from_read
    HAS_POSTERIOR_WRITER = True
except ImportError:
    HAS_POSTERIOR_WRITER = False


# Multiprocessing start method:
#
#   spawn (default):  workers re-import everything from scratch.  Safe — avoids
#                     pysam/htslib segfaults that fire reliably under Python
#                     ≥3.14 and occasionally on earlier versions, and avoids
#                     fiberhmm-run pipe-EOF deadlocks (forked workers inherit
#                     the stdout pipe FD and prevent EOF reaching downstream
#                     stages).
#   fork:             workers inherit the parent's already-imported modules,
#                     JIT-compiled numba functions, loaded model state, and
#                     htslib decompression buffers — significantly faster
#                     ramp-up and steady-state on long jobs (10× observed on
#                     360 GB Hia5 PacBio).  But triggers segfaults on some
#                     datasets/Python versions and is incompatible with
#                     streaming-pipeline EOF.
#
# Auto-default: spawn on Python ≥3.14 (fork unsafe), fork on Python <3.14
# Linux (fast, fork-safe enough for most data), fork on macOS (spawn cost
# is high, fork is the macOS default).  Override via env var
# FIBERHMM_MP_CONTEXT=spawn|fork.  Set =spawn explicitly if you hit a worker
# segfault during processing.
def _select_mp_context() -> 'multiprocessing.context.BaseContext':
    override = os.environ.get('FIBERHMM_MP_CONTEXT', '').strip().lower()
    if override in ('spawn', 'fork', 'forkserver'):
        return multiprocessing.get_context(override)
    if sys.version_info >= (3, 14):
        return multiprocessing.get_context('spawn')
    if sys.platform == 'darwin':
        return multiprocessing.get_context('fork')
    # Linux, Python <3.14: fork is much faster and segfaults are rare.
    return multiprocessing.get_context('fork')


_MP_CONTEXT = _select_mp_context()


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

        _worker_model = freeze_model_for_inference(load_model(model_path))
        _worker_debug_timing = debug_timing

        # Warmup numba JIT compilation in this worker
        from fiberhmm.core.hmm import HAS_NUMBA
        if HAS_NUMBA:
            dummy_obs = np.array([0, 1, 2, 3], dtype=np.int32)
            _ = _worker_model.predict(dummy_obs)
    except Exception as e:
        import traceback
        print(f"Worker init error: {e}")
        traceback.print_exc()
        raise


# Per-worker recall state: LLR tables for the TF Kadane scan.  Lives
# alongside _worker_model (set by _init_fused_worker).
_worker_recall_state = {}


def _init_fused_worker(apply_model_path, recall_model_path=None,
                       emission_uplift=1.0, debug_timing=False):
    """Initialize worker process for the fused apply+recall pipeline.

    Loads the apply HMM model plus the LLR tables used for the TF Kadane
    scan.  recall_model_path=None means reuse the apply model's emissions
    (the common case — same model file drives both passes).
    """
    global _worker_model, _worker_debug_timing, _worker_recall_state
    import os
    os.environ['NUMBA_CACHE_DIR'] = ''

    _worker_model = freeze_model_for_inference(load_model(apply_model_path))
    _worker_debug_timing = debug_timing

    # Build TF-recall LLR tables from the recall model (or reuse apply model).
    from fiberhmm.core.model_io import load_model_with_metadata
    from fiberhmm.inference.tf_recaller import (
        apply_emission_uplift,
        build_llr_tables,
    )
    r_path = recall_model_path or apply_model_path
    r_model, _, _ = load_model_with_metadata(r_path)
    llr_hit, llr_miss = build_llr_tables(r_model)
    if abs(emission_uplift - 1.0) > 1e-9:
        llr_hit, llr_miss = apply_emission_uplift(llr_hit, llr_miss, r_model, emission_uplift)
    _worker_recall_state['llr_hit'] = llr_hit
    _worker_recall_state['llr_miss'] = llr_miss

    # Warmup: apply Viterbi + TF Kadane scan
    from fiberhmm.core.hmm import HAS_NUMBA
    if HAS_NUMBA:
        dummy_obs = np.array([0, 1, 2, 3], dtype=np.int32)
        _ = _worker_model.predict(dummy_obs)
        from fiberhmm.inference.tf_recaller import call_tfs_in_interval
        _ = call_tfs_in_interval(
            np.zeros(16, dtype=np.int32), 0, 16,
            llr_hit, llr_miss, min_llr=4.0, min_opps=3,
        )


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

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
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
        _worker_model = freeze_model_for_inference(load_model(model_path))
        _worker_region_params = params

        # Warmup numba JIT (just the basic predict, posteriors will warmup on first use)
        from fiberhmm.core.hmm import HAS_NUMBA
        if HAS_NUMBA:
            dummy_obs = np.array([0, 1, 2, 3], dtype=np.int32)
            _ = _worker_model.predict(dummy_obs)

    except Exception as e:
        import traceback
        print(f"Region worker init error: {e}")
        traceback.print_exc()
        raise


def _process_region_to_bam(args: RegionBamWorkItem) -> RegionBamResult:
    """
    Worker function: process one genomic region and write to temp BAM.

    Each worker opens its own BAM file handle and uses the index to fetch
    reads from its assigned region. This enables true parallel I/O.

    Uses global _worker_model and _worker_region_params (set by _init_region_worker).

    Args:
        args: RegionBamWorkItem, or the legacy tuple shape.

    Returns:
        RegionBamResult with temp BAM, counts, optional TSV path, and skip reasons.
    """
    import traceback

    import numpy as np  # Ensure numpy is available in worker
    global _worker_model, _worker_region_params

    try:
        work_item = RegionBamWorkItem.from_value(args)
        chrom, start, end = work_item.region
        input_bam = work_item.input_bam
        temp_bam_path = work_item.temp_bam_path
        temp_tsv_path = work_item.temp_tsv_path

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
        nuc_min_size = int(params.get('nuc_min_size', 85))
        min_mapq = int(params['min_mapq'])
        prob_threshold = int(params['prob_threshold'])
        min_read_length = int(params['min_read_length'])
        with_scores = params['with_scores']
        train_rids = params['train_rids']
        primary_only = params.get('primary_only', False)
        return_posteriors = params.get('return_posteriors', False) and temp_tsv_path is not None
        write_msps = params.get('write_msps', True)
        io_threads = int(params.get('io_threads', 4))

        total_reads = 0
        reads_with_footprints = 0
        written = 0
        skipped = 0
        posteriors_written = 0

        # Track skip reasons
        skip_reasons = {
            'unmapped': 0,
            'secondary_supplementary': 0,
            'low_mapq': 0,
            'too_short': 0,
            'training_excluded': 0,
            'no_modifications': 0,
            'extraction_failed': 0,
            'no_footprints': 0,
        }

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
            with pysam.AlignmentFile(input_bam, "rb", threads=io_threads, check_sq=False) as inbam:
                with pysam.AlignmentFile(temp_bam_path, "wb", header=inbam.header, threads=io_threads) as outbam:

                    # Fetch reads from this region using the index
                    try:
                        read_iter = inbam.fetch(chrom, start, end)
                    except ValueError:
                        # Region not in BAM (e.g., unplaced contigs)
                        if tsv_file:
                            tsv_file.close()
                        return RegionBamResult(temp_bam_path, 0, 0, 0)

                    for read in read_iter:
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

                        # Only process reads that START in this region to avoid duplicates
                        # (fetch returns reads that overlap the region)
                        if read.reference_start < start or read.reference_start >= end:
                            continue

                        # Check filters
                        if read.mapping_quality < min_mapq:
                            outbam.write(read)
                            written += 1
                            skipped += 1
                            skip_reasons['low_mapq'] += 1
                            continue

                        if read.query_alignment_length is None or read.query_alignment_length < min_read_length:
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
                                skip_reasons['no_modifications'] += 1
                                continue
                        except Exception:
                            outbam.write(read)
                            written += 1
                            skipped += 1
                            skip_reasons['extraction_failed'] += 1
                            continue

                        total_reads += 1

                        # Process the read (with posteriors if requested)
                        result = _process_single_read(
                            fiber_read, model, edge_trim, circular,
                            mode, context_size, msp_min_size, nuc_min_size=nuc_min_size,
                            with_scores=with_scores,
                            return_posteriors=return_posteriors
                        )

                        # Add tags to read
                        if result is not None:
                            reads_with_footprints += 1

                            set_legacy_apply_tags(read, result, with_scores, write_msps)

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
                        else:
                            skip_reasons['no_footprints'] += 1

                        outbam.write(read)
                        written += 1

        finally:
            if tsv_file:
                tsv_file.close()

        # Return TSV path if we wrote any posteriors
        if return_posteriors and posteriors_written > 0 and temp_tsv_path:
            return RegionBamResult(
                temp_bam_path, total_reads, reads_with_footprints,
                written, temp_tsv_path, skip_reasons,
            )

        return RegionBamResult(
            temp_bam_path, total_reads, reads_with_footprints,
            written, None, skip_reasons,
        )

    except Exception as e:
        import traceback
        print(f"\nWorker error in region {chrom}:{start}-{end}: {e}")
        traceback.print_exc()
        raise


def _write_region_posteriors_tsv(tsv_path: str, posteriors_data: list):
    """Write posteriors data from a single region to TSV (simple append-friendly format)."""

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


def _process_region_to_bed(args: RegionBedWorkItem) -> RegionBedResult:
    """
    Process a genomic region and write BED output directly (no temp BAM).

    This is more space-efficient than _process_region_to_bam when only
    BED/bigBed output is needed.

    Args:
        args: RegionBedWorkItem, or the legacy tuple shape.

    Returns:
        RegionBedResult with temp BED path and counts.
    """
    work_item = RegionBedWorkItem.from_value(args)
    region = work_item.region
    input_bam = work_item.input_bam
    temp_bed_path = work_item.temp_bed_path
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
        nuc_min_size = int(params.get('nuc_min_size', 85))
        min_mapq = int(params['min_mapq'])
        prob_threshold = int(params['prob_threshold'])
        min_read_length = int(params['min_read_length'])
        with_scores = params['with_scores']
        train_rids = params['train_rids']
        io_threads = int(params.get('io_threads', 4))

        total_reads = 0
        reads_with_footprints = 0

        pysam.set_verbosity(0)

        with pysam.AlignmentFile(input_bam, "rb", threads=io_threads, check_sq=False) as inbam:
            with open(temp_bed_path, 'w') as bed_out:
                try:
                    read_iter = inbam.fetch(chrom, start, end)
                except ValueError:
                    return RegionBedResult(temp_bed_path, 0, 0)

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
                        mode, context_size, msp_min_size, nuc_min_size=nuc_min_size,
                        with_scores=with_scores
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
                        block_sizes_list = [int(length) for length in nl]

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

        return RegionBedResult(temp_bed_path, total_reads, reads_with_footprints)

    except Exception as e:
        import traceback
        print(f"\nWorker error in region {chrom}:{start}-{end}: {e}")
        traceback.print_exc()
        raise


def _process_bam_region_parallel(input_bam: str, output_bam: str,
                                   model_path: str, train_rids: Set[str],
                                   edge_trim: int, circular: bool,
                                   mode: str, context_size: int,
                                   msp_min_size: int,
                                   nuc_min_size: int = 85,
                                   min_mapq: int = 0,
                                   prob_threshold: int = 0,
                                   min_read_length: int = 0,
                                   with_scores: bool = False,
                                   n_cores: int = 1,
                                   region_size: int = 10_000_000,
                                   skip_scaffolds: bool = False,
                                   chroms: Optional[Set[str]] = None,
                                   primary_only: bool = False,
                                   output_posteriors: Optional[str] = None,
                                   write_msps: bool = True,
                                   io_threads: int = 4) -> Tuple[int, int]:
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
        write_msps: If True, write as/al/aq MSP tags to output BAM

    Returns:
        (total_reads_processed, reads_with_footprints)
    """
    import shutil
    import tempfile
    import time

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
    temp_dir = tempfile.mkdtemp(prefix='.fiberhmm_tmp_', dir=output_dir)

    try:
        # Prepare parameters (will be passed to initializer)
        params = {
            'edge_trim': edge_trim,
            'circular': circular,
            'mode': mode,
            'context_size': context_size,
            'msp_min_size': msp_min_size,
            'nuc_min_size': nuc_min_size,
            'min_mapq': min_mapq,
            'prob_threshold': prob_threshold,
            'min_read_length': min_read_length,
            'with_scores': with_scores,
            'train_rids': train_rids,
            'primary_only': primary_only,
            'return_posteriors': return_posteriors,
            'write_msps': write_msps,
            'io_threads': io_threads,
        }

        # Work items - include temp H5 path if posteriors requested
        work_items = []
        for i, region in enumerate(regions):
            temp_bam = os.path.join(temp_dir, f'region_{i:06d}.bam')
            temp_h5 = os.path.join(temp_dir, f'region_{i:06d}.tsv') if return_posteriors else None
            work_items.append(RegionBamWorkItem(region, input_bam, temp_bam, temp_h5))

        # Process regions in parallel
        aggregation = RegionBamAggregation()
        first_result_time = None

        # Use initializer to load model once per worker
        print(f"  Initializing {n_cores} worker processes (loading HMM model in each)...")
        sys.stdout.flush()
        pool_start = time.time()

        with ProcessPoolExecutor(
            max_workers=n_cores,
            mp_context=_MP_CONTEXT,
            initializer=_init_region_worker,
            initargs=(model_path, params)
        ) as executor:
            futures = {executor.submit(_process_region_to_bam, item): i
                      for i, item in enumerate(work_items)}

            for future in as_completed(futures):
                try:
                    result = RegionBamResult.from_value(future.result())
                    include_tsv = bool(
                        result.temp_tsv_path and os.path.exists(result.temp_tsv_path)
                    )
                    aggregation.add_result(futures[future], result, include_tsv=include_tsv)

                    # Track first result
                    if first_result_time is None:
                        first_result_time = time.time()
                        init_time = first_result_time - pool_start
                        print(f"  Workers ready ({init_time:.1f}s). Processing regions...")
                        sys.stdout.flush()

                    elapsed = time.time() - start_time
                    rate = aggregation.total_reads / elapsed if elapsed > 0 else 0
                    print(f"\r  Regions: {aggregation.completed}/{len(regions)} | "
                          f"Reads: {aggregation.total_reads:,} | "
                          f"With footprints: {aggregation.reads_with_footprints:,} | "
                          f"{rate:.1f} reads/s", end='')
                    sys.stdout.flush()

                except Exception as e:
                    print(f"\nError processing region: {e}")
                    raise

        print()  # Newline after progress

        # Print skip reasons summary
        if aggregation.total_skipped > 0:
            total_encountered = aggregation.total_reads + aggregation.total_skipped
            print(
                f"  Processed: {aggregation.total_reads:,} | "
                f"Skipped: {aggregation.total_skipped:,} | "
                f"With footprints: {aggregation.reads_with_footprints:,}"
            )
            print("  Skip reasons:")
            for reason, count in sorted(
                aggregation.skip_reasons.items(), key=lambda x: -x[1]
            ):
                if count > 0:
                    pct = 100 * count / total_encountered
                    print(f"    {reason}: {count:,} ({pct:.1f}%)")

        # Sort temp BAMs by region order and filter to non-empty
        aggregation.temp_bams.sort(key=lambda x: x[0])
        non_empty_bams = [bam for _, bam in aggregation.temp_bams
                         if os.path.exists(bam) and os.path.getsize(bam) > 0]

        _concatenate_region_bams(input_bam, output_bam, non_empty_bams, temp_dir)

        sys.stdout.flush()

        # Verify output was created
        if os.path.exists(output_bam):
            output_size_gb = os.path.getsize(output_bam) / (1024**3)
            print(f"Output BAM: {output_size_gb:.2f}GB")

        # Index the output BAM (sort first if needed)
        print("Step: Index/Sort...")
        _sort_and_index_bam(output_bam, threads=n_cores)

        # Merge temp TSV files if posteriors were requested
        if return_posteriors and aggregation.temp_tsvs:
            print(f"Merging {len(aggregation.temp_tsvs)} posterior files...")
            merge_start = time.time()
            n_fibers = _merge_region_posteriors_tsv(
                aggregation.temp_tsvs, output_posteriors,
                mode, context_size, edge_trim, input_bam,
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
                print(
                    f"Posteriors: {n_fibers:,} fibers -> {tsv_path} "
                    f"({file_size:.1f} MB, {merge_time:.1f}s)"
                )
                if output_posteriors.endswith('.h5'):
                    print(f"  To convert to H5: python posteriors_io.py tsv2h5 {tsv_path} {output_posteriors}")

        elapsed = time.time() - start_time
        rate = aggregation.total_reads / elapsed if elapsed > 0 else 0
        print(
            f"Completed: {aggregation.total_reads:,} reads | "
            f"{aggregation.reads_with_footprints:,} with footprints | "
            f"{rate:.1f} reads/s | {elapsed:.1f}s"
        )

        return aggregation.total_reads, aggregation.reads_with_footprints

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def _process_bed_region_parallel(input_bam: str, output_bed: str,
                                  model_path: str, train_rids: Set[str],
                                  edge_trim: int, circular: bool,
                                  mode: str, context_size: int,
                                  msp_min_size: int,
                                  nuc_min_size: int = 85,
                                  min_mapq: int = 0,
                                  prob_threshold: int = 0,
                                  min_read_length: int = 0,
                                  with_scores: bool = False,
                                  n_cores: int = 1,
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
    temp_dir = tempfile.mkdtemp(prefix='.fiberhmm_bed_tmp_', dir=output_dir)

    try:
        params = {
            'edge_trim': edge_trim,
            'circular': circular,
            'mode': mode,
            'context_size': context_size,
            'msp_min_size': msp_min_size,
            'nuc_min_size': nuc_min_size,
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
            work_items.append(RegionBedWorkItem(region, input_bam, temp_bed))

        aggregation = RegionBedAggregation()
        first_result_time = None

        print(f"  Initializing {n_cores} worker processes (loading HMM model in each)...")
        sys.stdout.flush()
        pool_start = time.time()

        with ProcessPoolExecutor(
            max_workers=n_cores,
            mp_context=_MP_CONTEXT,
            initializer=_init_region_worker,
            initargs=(model_path, params)
        ) as executor:
            futures = {executor.submit(_process_region_to_bed, item): i
                      for i, item in enumerate(work_items)}

            for future in as_completed(futures):
                try:
                    result = RegionBedResult.from_value(future.result())
                    aggregation.add_result(futures[future], result)

                    # Track first result
                    if first_result_time is None:
                        first_result_time = time.time()
                        init_time = first_result_time - pool_start
                        print(f"  Workers ready ({init_time:.1f}s). Processing regions...")
                        sys.stdout.flush()

                    elapsed = time.time() - start_time
                    rate = aggregation.total_reads / elapsed if elapsed > 0 else 0
                    print(f"\r  Regions: {aggregation.completed}/{len(regions)} | "
                          f"Reads: {aggregation.total_reads:,} | "
                          f"With footprints: {aggregation.reads_with_footprints:,} | "
                          f"{rate:.1f} reads/s", end='')
                    sys.stdout.flush()

                except Exception as e:
                    print(f"\nError processing region: {e}")
                    raise

        print()  # Newline after progress

        # Sort temp BEDs by region order and concatenate
        aggregation.temp_beds.sort(key=lambda x: x[0])
        non_empty_beds = [bed for _, bed in aggregation.temp_beds
                         if os.path.exists(bed) and os.path.getsize(bed) > 0]

        print(f"Concatenating {len(non_empty_beds)} region BED files...")
        sys.stdout.flush()

        with open(output_bed, 'wb') as fout:
            for bed_path in non_empty_beds:
                with open(bed_path, 'rb') as fin:
                    shutil.copyfileobj(fin, fout)

        elapsed = time.time() - start_time
        rate = aggregation.total_reads / elapsed if elapsed > 0 else 0
        print(
            f"Completed: {aggregation.total_reads:,} reads | "
            f"{aggregation.reads_with_footprints:,} with footprints | "
            f"{rate:.1f} reads/s | {elapsed:.1f}s"
        )

        return aggregation.total_reads, aggregation.reads_with_footprints

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _process_and_write_chunk(chunk_reads: list, chunk_read_objs: list,
                              outbam, model, executor,
                              edge_trim: int, circular: bool,
                              mode: str, context_size: int,
                              msp_min_size: int, nuc_min_size: int = 85,
                              with_scores: bool = False,
                              return_posteriors: bool = False,
                              write_msps: bool = True) -> Tuple[int, int, int, Optional[list]]:
    """
    Process a chunk of reads and write to BAM.

    Returns:
        (reads_with_footprints, no_footprints, worker_failures,
         results_with_posteriors or None)

        If return_posteriors=True, returns results list for caller to write posteriors.
    """

    if executor is not None:
        # Parallel: submit chunk to worker
        future = executor.submit(
            _process_chunk_worker,
            chunk_reads, edge_trim, circular, mode, context_size, msp_min_size,
            nuc_min_size, with_scores, return_posteriors
        )
        results, worker_failures = coerce_worker_chunk_result(future.result())
    else:
        # Single-threaded: process directly
        results = []
        worker_failures = 0
        for fiber_read in chunk_reads:
            result = _process_single_read(
                fiber_read, model, edge_trim, circular,
                mode, context_size, msp_min_size, nuc_min_size=nuc_min_size,
                with_scores=with_scores,
                return_posteriors=return_posteriors
            )
            results.append(result)

    # Write annotated reads
    reads_with_footprints = 0
    no_footprints = 0
    for read_obj, result in zip(chunk_read_objs, results):
        if result is not None:
            set_legacy_apply_tags(read_obj, result, with_scores, write_msps)
            reads_with_footprints += 1
        else:
            no_footprints += 1

        outbam.write(read_obj)

    # Return results for posteriors if requested
    if return_posteriors:
        return (
            reads_with_footprints,
            no_footprints,
            worker_failures,
            list(zip(chunk_read_objs, chunk_reads, results)),
        )
    return reads_with_footprints, no_footprints, worker_failures, None


def _process_chunk_worker(chunk_reads: list, edge_trim: int, circular: bool,
                           mode: str, context_size: int, msp_min_size: int,
                           nuc_min_size: int = 85,
                           with_scores: bool = False,
                           return_posteriors: bool = False) -> list:
    """Worker function to process a chunk of reads.

    Per-read errors are caught and converted to None results so a single bad
    read can never bring down the entire worker process (and with it the whole
    chunk of ~500 reads). Reads that fail are written through to the output
    unchanged without footprint tags, with the failure count reported
    separately in WorkerChunkResult.
    """
    global _worker_model

    results = []
    read_failures = 0
    for fiber_read in chunk_reads:
        try:
            result = _process_single_read(
                fiber_read, _worker_model, edge_trim, circular,
                mode, context_size, msp_min_size, nuc_min_size=nuc_min_size,
                with_scores=with_scores,
                return_posteriors=return_posteriors
            )
        except Exception:
            # Per-read failure: skip this read but keep the worker alive
            result = None
            read_failures += 1
        results.append(result)

    return WorkerChunkResult(results, read_failures)


# ---------------------------------------------------------------------------
# Fused apply+recall worker — eliminates the streaming pipe serialization
# that costs ~2-7× throughput on chained pipelines.  Main process sends one
# slim payload per read; worker runs extract → HMM → TF scan → unify in a
# single call, keeping the encoded obs array in cache between apply and
# recall passes.
# ---------------------------------------------------------------------------

def _process_fused_payload_chunk_worker(
    chunk_payloads: list,
    edge_trim: int, circular: bool,
    mode: str, context_size: int, msp_min_size: int,
    nuc_min_size: int = 85,
    with_scores: bool = False,
    prob_threshold: int = 125,
    # Recall params
    recall_mode: str = None,            # mode for the TF LLR tables (usually same as apply mode)
    recall_context_size: int = None,    # k for TF LLR tables (usually same)
    min_llr: float = 4.0,
    min_opps: int = 3,
    unify_threshold: int = 90,
) -> list:
    """Slim-IPC worker: apply HMM + TF recall in a single call per read.

    Returns WorkerChunkResult with one result entry per payload.  Each entry
    is either None (no usable modification data or per-read worker failure)
    or a dict with:
        'ns', 'nl':  numpy int arrays of unified nucleosome footprints
                     (post-unification: short nucs overlapping TF calls are
                     demoted into the tf+ annotation track)
        'as', 'al':  numpy int arrays of MSPs (unchanged from apply)
        'tf_calls':  list of TFCall objects
        'ns_scores', 'as_scores': optional nq/aq scores if with_scores
    """
    global _worker_model, _worker_recall_state

    # Model/params set once per worker; TF LLR tables attached to the
    # worker globals via _init_fused_worker.
    llr_hit = _worker_recall_state['llr_hit']
    llr_miss = _worker_recall_state['llr_miss']

    results = []
    read_failures = 0
    for payload in chunk_payloads:
        try:
            fiber_read = extract_fiber_read_from_payload(payload, mode, prob_threshold)
            if fiber_read is None:
                results.append(None)
                continue

            apply_result = run_hmm_apply_stage(
                fiber_read,
                _worker_model,
                edge_trim,
                circular,
                mode,
                context_size,
                msp_min_size,
                nuc_min_size,
                with_scores,
            )

            # Match streaming semantics: if apply produced no footprints and
            # no MSPs, treat as "nothing to annotate" — return None so the
            # drain pass-throughs the read unchanged (preserving any
            # pre-existing tags on the input).  With include_encoded=True
            # _process_single_read does NOT early-return on empty output.
            if not apply_result_has_footprints(apply_result):
                results.append(None)
                continue

            results.append(build_fused_recall_result(
                fiber_read,
                apply_result,
                llr_hit,
                llr_miss,
                min_llr,
                min_opps,
                unify_threshold,
                with_scores,
            ))
        except Exception:
            # Per-read failure must not kill the worker (or the whole chunk).
            read_failures += 1
            results.append(None)

    return WorkerChunkResult(results, read_failures)


def _process_payload_chunk_worker(chunk_payloads: list, edge_trim: int, circular: bool,
                                    mode: str, context_size: int, msp_min_size: int,
                                    nuc_min_size: int = 85,
                                    with_scores: bool = False,
                                    return_posteriors: bool = False,
                                    prob_threshold: int = 125) -> list:
    """Slim-IPC worker: parses MM/ML payloads then runs HMM.

    The streaming pipeline ships slim payloads (built by make_apply_payload
    in main) instead of pre-parsed fiber_read dicts.  Each worker does the
    MM/ML parse + HMM in parallel rather than serializing the parse on the
    main process.  Returns WorkerChunkResult with one entry per payload, with
    None for reads that have no usable modification data or hit a per-read
    worker failure.
    """
    global _worker_model

    results = []
    read_failures = 0
    for payload in chunk_payloads:
        try:
            fiber_read = extract_fiber_read_from_payload(payload, mode, prob_threshold)
            if fiber_read is None:
                results.append(None)
                continue
            result = _process_single_read(
                fiber_read, _worker_model, edge_trim, circular,
                mode, context_size, msp_min_size, nuc_min_size=nuc_min_size,
                with_scores=with_scores,
                return_posteriors=return_posteriors
            )
        except Exception:
            result = None
            read_failures += 1
        results.append(result)

    return WorkerChunkResult(results, read_failures)


# ---------------------------------------------------------------------------
# Streaming pipeline: sliding-window producer-consumer architecture
# ---------------------------------------------------------------------------


def _drain_oldest_chunk(inflight, outbam, with_scores, write_msps,
                        posterior_writer, counters):
    """
    Block on the oldest in-flight chunk, apply tags, write to BAM.

    Walks chunk_read_objs in original stream order so that skipped reads
    (which the worker doesn't see) are interleaved with processed reads at
    their correct positions — preserves coordinate-sortedness of an already
    coordinate-sorted input BAM.  results is shorter than chunk_read_objs
    by exactly the number of skipped reads in this chunk.

    Args:
        inflight: deque of (future, chunk_read_objs, chunk_reads, chunk_skip_flags)
        outbam: pysam.AlignmentFile for writing
        with_scores: whether to write nq/aq score tags
        write_msps: whether to write as/al/aq MSP tags
        posterior_writer: PosteriorWriter instance or None
        counters: mutable dict with 'reads_with_footprints', 'no_footprints', 'written'
    """
    future, chunk_read_objs, chunk_reads, chunk_skip_flags = inflight.popleft()
    results, worker_failures = coerce_worker_chunk_result(future.result())
    if worker_failures:
        counters['worker_failures'] = counters.get('worker_failures', 0) + worker_failures
    result_iter = iter(results)
    fiber_iter = iter(chunk_reads)

    for read_obj, is_skipped in zip(chunk_read_objs, chunk_skip_flags):
        if is_skipped:
            # Pass-through: write at original stream position to preserve
            # coordinate sort order.
            outbam.write(read_obj)
            counters['written'] += 1
            continue

        next(fiber_iter)
        result = next(result_iter)
        if result is not None:
            set_legacy_apply_tags(read_obj, result, with_scores, write_msps)

            counters['reads_with_footprints'] += 1

            # Posteriors export
            if posterior_writer and result.get('posteriors') is not None:
                chrom = read_obj.reference_name
                if chrom:
                    try:
                        ref_positions = get_ref_positions_from_read(read_obj) if HAS_POSTERIOR_WRITER else np.array([], dtype=np.int32)
                    except Exception:
                        ref_positions = np.array([], dtype=np.int32)
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
        else:
            counters['no_footprints'] += 1

        outbam.write(read_obj)
        counters['written'] += 1


def _init_fused_region_worker(apply_model_path: str, recall_model_path: Optional[str],
                              emission_uplift: float, params: dict):
    """Per-worker init for region-parallel fused apply+recall.

    Loads the apply HMM model, builds the TF LLR tables (from the recall
    model or by reusing the apply model), warms up numba JIT, and stashes
    params for the region worker to pick up.
    """
    global _worker_model, _worker_region_params, _worker_recall_state
    import os
    os.environ['NUMBA_CACHE_DIR'] = ''

    _worker_model = freeze_model_for_inference(load_model(apply_model_path))
    # Open the reference FASTA *after* fork: pysam.FastaFile is not
    # fork-safe. Stash the live handle on params so the region worker
    # can pass it through to get_daf_positions.
    ref_path = params.get('ref_fasta_path')
    if ref_path:
        import pysam as _pysam
        params = dict(params)   # don't mutate the shared-across-workers dict
        params['ref_fasta'] = _pysam.FastaFile(ref_path)
    _worker_region_params = params

    from fiberhmm.core.model_io import load_model_with_metadata
    from fiberhmm.inference.tf_recaller import apply_emission_uplift, build_llr_tables
    r_path = recall_model_path or apply_model_path
    r_model, _, _ = load_model_with_metadata(r_path)
    llr_hit, llr_miss = build_llr_tables(r_model)
    if abs(emission_uplift - 1.0) > 1e-9:
        llr_hit, llr_miss = apply_emission_uplift(llr_hit, llr_miss, r_model, emission_uplift)
    _worker_recall_state['llr_hit'] = llr_hit
    _worker_recall_state['llr_miss'] = llr_miss

    # Warmup
    from fiberhmm.core.hmm import HAS_NUMBA
    if HAS_NUMBA:
        dummy_obs = np.array([0, 1, 2, 3], dtype=np.int32)
        _ = _worker_model.predict(dummy_obs)
        from fiberhmm.inference.tf_recaller import call_tfs_in_interval
        _ = call_tfs_in_interval(
            np.zeros(16, dtype=np.int32), 0, 16,
            llr_hit, llr_miss, min_llr=4.0, min_opps=3,
        )


def _process_region_to_bam_fused(args: RegionBamWorkItem) -> RegionBamResult:
    """Region worker: fetch reads in one genomic region, run fused
    apply+recall per read, write in-order to a coordinate-sorted temp BAM.

    Because pysam.fetch(chrom,start,end) yields reads in coordinate order
    AND we only process reads that START in this region (the reference_start
    filter), each temp BAM is coordinate-sorted within itself.  Concatenating
    temp BAMs in region order gives a coordinate-sorted final BAM without
    any sort pass.

    Returns a RegionBamResult with temp BAM, counts, and skip reasons.
    """
    import traceback

    from fiberhmm.inference.engine import extract_fiber_read_from_payload, make_apply_payload
    global _worker_model, _worker_region_params, _worker_recall_state

    try:
        work_item = RegionBamWorkItem.from_value(args)
        chrom, start, end = work_item.region
        input_bam = work_item.input_bam
        temp_bam_path = work_item.temp_bam_path
        start = int(start)
        end = int(end)

        params = _worker_region_params
        model = _worker_model
        llr_hit = _worker_recall_state['llr_hit']
        llr_miss = _worker_recall_state['llr_miss']

        edge_trim = int(params['edge_trim'])
        circular = params['circular']
        mode = params['mode']
        ref_fasta = params.get('ref_fasta')   # opened FastaFile or None; --reference path
        context_size = int(params['context_size'])
        msp_min_size = int(params['msp_min_size'])
        nuc_min_size = int(params.get('nuc_min_size', 85))
        min_mapq = int(params['min_mapq'])
        prob_threshold = int(params['prob_threshold'])
        min_read_length = int(params['min_read_length'])
        with_scores = params.get('with_scores', False)
        train_rids = params.get('train_rids') or set()
        primary_only = params.get('primary_only', False)
        io_threads = int(params.get('io_threads', 4))
        min_llr = float(params['min_llr'])
        min_opps = int(params['min_opps'])
        unify_threshold = int(params['unify_threshold'])
        also_write_legacy = params['also_write_legacy']
        downstream_compat = params['downstream_compat']

        pysam.set_verbosity(0)

        total_reads = 0
        reads_with_fp = 0
        written = 0
        skipped = 0
        skip_reasons = {
            'unmapped': 0, 'secondary_supplementary': 0, 'low_mapq': 0,
            'too_short': 0, 'training_excluded': 0, 'no_modifications': 0,
            'extraction_failed': 0, 'no_footprints': 0,
        }

        with pysam.AlignmentFile(input_bam, "rb", threads=io_threads, check_sq=False) as inbam:
            with pysam.AlignmentFile(temp_bam_path, "wb", header=inbam.header, threads=io_threads) as outbam:
                try:
                    read_iter = inbam.fetch(chrom, start, end)
                except ValueError:
                    return RegionBamResult(temp_bam_path, 0, 0, 0)

                for read in read_iter:
                    if read.is_unmapped:
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['unmapped'] += 1
                        continue
                    if primary_only and (read.is_secondary or read.is_supplementary):
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['secondary_supplementary'] += 1
                        continue
                    # Only process reads starting in this region (fetch is overlap-based).
                    if read.reference_start < start or read.reference_start >= end:
                        continue
                    if read.mapping_quality < min_mapq:
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['low_mapq'] += 1
                        continue
                    if (read.query_alignment_length is None
                            or read.query_alignment_length < min_read_length):
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['too_short'] += 1
                        continue
                    if train_rids and read.query_name in train_rids:
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['training_excluded'] += 1
                        continue

                    payload = make_apply_payload(read, mode=mode, ref_fasta=ref_fasta)
                    if payload is None:
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['no_modifications'] += 1
                        continue

                    try:
                        fiber_read = extract_fiber_read_from_payload(payload, mode, prob_threshold)
                        if fiber_read is None:
                            outbam.write(read)
                            written += 1
                            skipped += 1
                            skip_reasons['no_modifications'] += 1
                            continue
                        apply_result = run_hmm_apply_stage(
                            fiber_read,
                            model,
                            edge_trim,
                            circular,
                            mode,
                            context_size,
                            msp_min_size,
                            nuc_min_size,
                            with_scores,
                        )
                    except Exception:
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['extraction_failed'] += 1
                        continue

                    total_reads += 1

                    if not apply_result_has_footprints(apply_result):
                        outbam.write(read)
                        written += 1
                        skip_reasons['no_footprints'] += 1
                        continue

                    fused_result = build_fused_recall_result(
                        fiber_read,
                        apply_result,
                        llr_hit,
                        llr_miss,
                        min_llr,
                        min_opps,
                        unify_threshold,
                        with_scores,
                    )
                    write_fused_recall_tags(
                        read,
                        read_length=len(fiber_read['query_sequence']),
                        result=fused_result,
                        also_write_legacy=also_write_legacy,
                        downstream_compat=downstream_compat,
                    )
                    outbam.write(read)
                    written += 1
                    reads_with_fp += 1

        return RegionBamResult(
            temp_bam_path, total_reads, reads_with_fp,
            written, None, skip_reasons,
        )

    except Exception:
        traceback.print_exc()
        raise


def _process_bam_region_parallel_fused(
    input_bam: str, output_bam: str,
    apply_model_path: str, recall_model_path: Optional[str],
    train_rids: Set[str],
    edge_trim: int, circular: bool,
    mode: str, context_size: int,
    msp_min_size: int, nuc_min_size: int,
    min_mapq: int, prob_threshold: int, min_read_length: int,
    with_scores: bool,
    min_llr: float, min_opps: int, unify_threshold: int,
    emission_uplift: float,
    also_write_legacy: bool, downstream_compat: bool,
    n_cores: int, region_size: int, skip_scaffolds: bool,
    chroms: Optional[Set[str]], io_threads: int,
    primary_only: bool = False,
    ref_fasta_path: Optional[str] = None,
):
    """Region-parallel fused apply+recall.

    Splits the BAM into genomic regions, runs fused apply+recall in each
    region as an independent worker, then concatenates sorted temp BAMs
    in region order.  Input BAM must be coordinate-sorted + indexed.

    Output is coordinate-sorted with no sort pass needed.
    """
    start_time = time.time()

    regions = _get_genome_regions(input_bam, region_size, skip_scaffolds, chroms)
    print(f"Processing {len(regions)} regions with {n_cores} cores (fused apply+recall)...")
    sys.stdout.flush()

    output_dir = os.path.dirname(os.path.abspath(output_bam))
    temp_dir = tempfile.mkdtemp(prefix='.fiberhmm_call_tmp_', dir=output_dir)

    params = {
        'edge_trim': edge_trim, 'circular': circular,
        'mode': mode, 'context_size': context_size,
        'msp_min_size': msp_min_size, 'nuc_min_size': nuc_min_size,
        'min_mapq': min_mapq, 'prob_threshold': prob_threshold,
        'min_read_length': min_read_length, 'with_scores': with_scores,
        'train_rids': train_rids, 'primary_only': primary_only,
        'io_threads': io_threads,
        'min_llr': min_llr, 'min_opps': min_opps,
        'unify_threshold': unify_threshold,
        'also_write_legacy': also_write_legacy,
        'downstream_compat': downstream_compat,
        # Path string, NOT an open handle: pysam.FastaFile is not fork-safe,
        # so each worker opens it lazily in _init_fused_region_worker.
        'ref_fasta_path': ref_fasta_path,
    }

    work_items = [
        RegionBamWorkItem(
            (r[0], r[1], r[2]), input_bam,
            os.path.join(temp_dir, f'region_{i:06d}.bam'),
        )
        for i, r in enumerate(regions)
    ]

    try:
        aggregation = RegionBamAggregation()

        print(f"  Initializing {n_cores} workers (loading apply model + LLR tables)...")
        sys.stdout.flush()
        pool_start = time.time()
        first_result = None

        with ProcessPoolExecutor(
            max_workers=n_cores,
            mp_context=_MP_CONTEXT,
            initializer=_init_fused_region_worker,
            initargs=(apply_model_path, recall_model_path, emission_uplift, params),
        ) as executor:
            futures = {executor.submit(_process_region_to_bam_fused, item): i
                       for i, item in enumerate(work_items)}

            for future in as_completed(futures):
                result = RegionBamResult.from_value(future.result())
                aggregation.add_result(futures[future], result)
                if first_result is None:
                    first_result = time.time()
                    print(f"  Workers ready ({first_result - pool_start:.1f}s). Processing...")
                    sys.stdout.flush()
                elapsed = time.time() - start_time
                rate = aggregation.total_reads / elapsed if elapsed > 0 else 0
                print(f"\r  Regions: {aggregation.completed}/{len(regions)} | "
                      f"Reads: {aggregation.total_reads:,} | "
                      f"With FP: {aggregation.reads_with_footprints:,} | "
                      f"{rate:.0f} r/s", end='')
                sys.stdout.flush()
        print()

        if aggregation.total_skipped > 0:
            total_enc = aggregation.total_reads + aggregation.total_skipped
            print(
                f"  Processed: {aggregation.total_reads:,} | "
                f"Skipped: {aggregation.total_skipped:,} | "
                f"With FP: {aggregation.reads_with_footprints:,}"
            )
            print("  Skip reasons:")
            for reason, count in sorted(
                aggregation.skip_reasons.items(), key=lambda x: -x[1]
            ):
                if count > 0:
                    print(f"    {reason}: {count:,} ({100*count/total_enc:.1f}%)")

        # Concat region BAMs in region-index order — preserves coord sort.
        aggregation.temp_bams.sort(key=lambda x: x[0])
        non_empty = [bam for _, bam in aggregation.temp_bams
                     if os.path.exists(bam) and os.path.getsize(bam) > 0]
        _concatenate_region_bams(input_bam, output_bam, non_empty, temp_dir)

        # Index directly (input sorted → each region sorted → concat sorted).
        try:
            pysam.index(output_bam)
        except pysam.SamtoolsError:
            pass

        elapsed = time.time() - start_time
        rate = aggregation.total_reads / elapsed if elapsed > 0 else 0
        print(
            f"  Total: {aggregation.total_reads:,} reads, "
            f"{aggregation.reads_with_footprints:,} with footprints, "
            f"{rate:.1f} r/s"
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return aggregation.total_reads, aggregation.reads_with_footprints


def _drain_oldest_fused_chunk(inflight, outbam, with_scores,
                              also_write_legacy, downstream_compat, counters):
    """Fused apply+recall drain: applies ns/nl/as/al AND MA/AQ tags via
    the shared fused recall tag writer.

    Skipped reads (is_skipped=True) are passed through unchanged at their
    original stream position so coordinate-sorted input stays coordinate
    sorted.  results is shorter than chunk_read_objs by the number of
    skipped reads.
    """
    future, chunk_read_objs, chunk_payloads, chunk_skip_flags = inflight.popleft()
    results, worker_failures = coerce_worker_chunk_result(future.result())
    if worker_failures:
        counters['worker_failures'] = counters.get('worker_failures', 0) + worker_failures
    result_iter = iter(results)
    payload_iter = iter(chunk_payloads)

    for read_obj, is_skipped in zip(chunk_read_objs, chunk_skip_flags):
        if is_skipped:
            outbam.write(read_obj)
            counters['written'] += 1
            continue

        _ = next(payload_iter)       # discard paired payload
        result = next(result_iter)
        if result is not None:
            write_fused_recall_tags(
                read_obj,
                read_length=len(read_obj.query_sequence) if read_obj.query_sequence else 0,
                result=result,
                also_write_legacy=also_write_legacy,
                downstream_compat=downstream_compat,
            )
            counters['reads_with_footprints'] += 1
        else:
            counters['no_footprints'] += 1

        outbam.write(read_obj)
        counters['written'] += 1


def _process_bam_streaming_pipeline_fused(
    input_bam: str, output_bam: str,
    model_path: str, recall_model_path: str,
    train_rids,
    edge_trim: int, circular: bool,
    mode: str, context_size: int,
    msp_min_size: int, nuc_min_size: int,
    min_mapq: int, prob_threshold: int, min_read_length: int,
    with_scores: bool,
    min_llr: float, min_opps: int, unify_threshold: int,
    emission_uplift: float,
    also_write_legacy: bool, downstream_compat: bool,
    max_reads: int, n_cores: int, chunk_size: int,
    io_threads: int,
    process_unmapped: bool = False,
    primary_only: bool = False,
    ref_fasta_path: Optional[str] = None,
):
    """Fused apply+recall streaming pipeline: one worker pool does both
    the HMM nucleosome/MSP calls and the TF Kadane scan per read.

    Eliminates the apply→recall pipe serialization (which was
    ~2-7× slower than the sum of the two stages run alone on this Mac
    benchmark) and the duplicate MM/ML parse + encode done by the
    streaming recall stage.
    """
    # Streaming mode reads the BAM sequentially in the main process, so we
    # can open the FastaFile here (no fork yet) and hand the live handle
    # to make_apply_payload for on-the-fly MD fallback on raw DAF BAMs.
    ref_fasta = None
    if ref_fasta_path:
        ref_fasta = pysam.FastaFile(ref_fasta_path)
    pysam.set_verbosity(0)
    max_inflight = n_cores + 2
    start_time = time.time()
    counters = {
        'reads_with_footprints': 0,
        'no_footprints': 0,
        'worker_failures': 0,
        'written': 0,
    }

    total_reads = 0
    skipped = 0
    skip_reasons = {
        'unmapped': 0, 'secondary_supplementary': 0, 'low_mapq': 0,
        'too_short': 0, 'training_excluded': 0, 'no_modifications': 0,
        'extraction_failed': 0,
    }
    filter_config = ReadFilterConfig(
        min_mapq=min_mapq,
        min_read_length=min_read_length,
        primary_only=primary_only,
        process_unmapped=process_unmapped,
        train_rids=train_rids,
    )

    _log = sys.stderr if output_bam == '-' else sys.stdout

    _output_target = output_bam
    if output_bam == '-':
        _output_target = os.fdopen(1, 'wb', closefd=False)

    with pysam.AlignmentFile(input_bam, "rb", threads=io_threads, check_sq=False) as inbam:
        with pysam.AlignmentFile(_output_target, "wb", header=inbam.header, threads=io_threads) as outbam:
            executor = ProcessPoolExecutor(
                max_workers=n_cores,
                mp_context=_MP_CONTEXT,
                initializer=_init_fused_worker,
                initargs=(model_path, recall_model_path, emission_uplift, False),
            )

            inflight = deque()
            chunk_payloads = []
            chunk_read_objs = []
            chunk_skip_flags = []
            last_progress_reads = 0
            last_progress_time = time.time()

            def _buffer_skip(rd, reason):
                nonlocal skipped
                chunk_read_objs.append(rd)
                chunk_skip_flags.append(True)
                skipped += 1
                skip_reasons[reason] += 1

            try:
                for read in inbam.fetch(until_eof=True):
                    skip_reason = streaming_skip_reason(read, filter_config)
                    if skip_reason:
                        _buffer_skip(read, skip_reason)
                        continue

                    payload = make_apply_payload(read, mode=mode, ref_fasta=ref_fasta)
                    if payload is None:
                        _buffer_skip(read, 'no_modifications')
                        continue

                    chunk_payloads.append(payload)
                    chunk_read_objs.append(read)
                    chunk_skip_flags.append(False)
                    total_reads += 1

                    if max_reads and total_reads >= max_reads:
                        break

                    if len(chunk_read_objs) >= chunk_size:
                        if len(inflight) >= max_inflight:
                            _drain_oldest_fused_chunk(
                                inflight, outbam, with_scores,
                                also_write_legacy, downstream_compat, counters,
                            )
                        future = executor.submit(
                            _process_fused_payload_chunk_worker,
                            chunk_payloads, edge_trim, circular, mode,
                            context_size, msp_min_size, nuc_min_size,
                            with_scores, prob_threshold,
                            mode, context_size,
                            min_llr, min_opps, unify_threshold,
                        )
                        inflight.append((future, chunk_read_objs, chunk_payloads, chunk_skip_flags))
                        chunk_payloads = []
                        chunk_read_objs = []
                        chunk_skip_flags = []

                        now = time.time()
                        elapsed = now - start_time
                        avg = total_reads / elapsed if elapsed > 0 else 0
                        dt = now - last_progress_time
                        inst = (total_reads - last_progress_reads) / dt if dt > 0 else 0
                        last_progress_reads = total_reads
                        last_progress_time = now
                        print(f"\r  Fused: {total_reads:,} | Skipped: {skipped:,} | "
                              f"Inflight: {len(inflight)} | {inst:.0f} r/s (avg {avg:.0f})",
                              end='', file=_log)
                        _log.flush()

                # Final partial chunk
                if chunk_read_objs:
                    if len(inflight) >= max_inflight:
                        _drain_oldest_fused_chunk(
                            inflight, outbam, with_scores,
                            also_write_legacy, downstream_compat, counters,
                        )
                    if chunk_payloads:
                        future = executor.submit(
                            _process_fused_payload_chunk_worker,
                            chunk_payloads, edge_trim, circular, mode,
                            context_size, msp_min_size, nuc_min_size,
                            with_scores, prob_threshold,
                            mode, context_size,
                            min_llr, min_opps, unify_threshold,
                        )
                    else:
                        future = Future()
                        future.set_result([])
                    inflight.append((future, chunk_read_objs, chunk_payloads, chunk_skip_flags))

                while inflight:
                    _drain_oldest_fused_chunk(
                        inflight, outbam, with_scores,
                        also_write_legacy, downstream_compat, counters,
                    )
            finally:
                executor.shutdown(wait=True)

    if ref_fasta is not None:
        ref_fasta.close()

    elapsed = time.time() - start_time
    rate = total_reads / elapsed if elapsed > 0 else 0
    reads_with_fp = counters['reads_with_footprints']
    print(f"\r  Fused: {total_reads:,} | Skipped: {skipped:,} | "
          f"With footprints: {reads_with_fp:,} | {rate:.1f} r/s", file=_log)
    if counters['worker_failures']:
        print(
            f"  Worker read failures: {counters['worker_failures']:,} "
            f"(passed through unchanged)",
            file=_log,
        )
    if skipped > 0:
        print("  Skip reasons:", file=_log)
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = 100 * count / (total_reads + skipped)
                print(f"    {reason}: {count:,} ({pct:.1f}%)", file=_log)

    return total_reads, reads_with_fp


def _process_bam_streaming_pipeline(
    input_bam: str, output_bam: str,
    model_path: str, train_rids: Set[str],
    edge_trim: int, circular: bool,
    mode: str, context_size: int,
    msp_min_size: int,
    nuc_min_size: int = 85,
    min_mapq: int = 0,
    prob_threshold: int = 0,
    min_read_length: int = 0,
    with_scores: bool = False,
    n_cores: int = 4,
    chunk_size: int = 500,
    max_inflight: Optional[int] = None,
    io_threads: int = 4,
    primary_only: bool = False,
    output_posteriors: Optional[str] = None,
    write_msps: bool = True,
    max_reads: Optional[int] = None,
    debug_timing: bool = False,
    process_unmapped: bool = False,
) -> Tuple[int, int]:
    """
    Streaming producer-consumer pipeline for BAM processing.

    Note: ``ref_fasta`` for the DAF MD-only fallback is set to None here;
    --reference CLI plumbing is Step 4 of the daf-md fallback work.
    MD-only DAF BAMs work without it.

    Uses a sliding window of in-flight futures to overlap I/O and compute.
    Works with unaligned/unindexed BAMs and stdin. Order-preserving.

    The main process reads BAM sequentially, extracts fiber data, and submits
    chunks to a process pool. Multiple chunks can be in flight simultaneously
    (bounded by max_inflight), keeping workers saturated while the main
    process continues reading and writing.

    Args:
        max_inflight: Max chunks in flight (default: 2 * n_cores)
        chunk_size: Reads per compute chunk (default: 500)
        io_threads: htslib decompression threads (default: 4)
        process_unmapped: If True, process unmapped reads that have sequences

    Returns:
        (total_reads_processed, reads_with_footprints)
    """
    if max_inflight is None:
        max_inflight = 2 * n_cores

    pysam.set_verbosity(0)

    ref_fasta = None    # placeholder; --reference plumbing is Step 4 of the
                        # daf-md fallback work. MD-only DAF BAMs work without it.

    total_reads = 0
    skip_reasons = {
        'unmapped': 0,
        'secondary_supplementary': 0,
        'low_mapq': 0,
        'too_short': 0,
        'training_excluded': 0,
        'no_modifications': 0,
        'extraction_failed': 0,
        'no_footprints': 0,
    }
    filter_config = ReadFilterConfig(
        min_mapq=min_mapq,
        min_read_length=min_read_length,
        primary_only=primary_only,
        process_unmapped=process_unmapped,
        train_rids=train_rids,
    )

    counters = {
        'reads_with_footprints': 0,
        'no_footprints': 0,
        'worker_failures': 0,
        'written': 0,
    }

    # Initialize posteriors writer
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

    # When writing to stdout, send progress to stderr to avoid corrupting BAM
    _log = sys.stderr if output_bam == '-' else sys.stdout

    print(f"Processing BAM (streaming pipeline, {n_cores} workers, "
          f"chunk_size={chunk_size}, max_inflight={max_inflight})...", file=_log)
    _log.flush()

    start_time = time.time()

    # For stdout output, ensure pysam writes to the real stdout fd
    # (sys.stdout may have been redirected to stderr for logging)
    _output_target = output_bam
    if output_bam == '-':
        _output_target = os.fdopen(1, 'wb', closefd=False)

    with pysam.AlignmentFile(input_bam, "rb", threads=io_threads, check_sq=False) as inbam:
        with pysam.AlignmentFile(_output_target, "wb", header=inbam.header, threads=io_threads) as outbam:

            # Create worker pool
            executor = ProcessPoolExecutor(
                max_workers=n_cores,
                mp_context=_MP_CONTEXT,
                initializer=_init_bam_worker,
                initargs=(model_path, debug_timing)
            )

            inflight = deque()
            chunk_reads = []           # only fiber_reads to be processed
            chunk_read_objs = []       # ALL reads in stream order (skipped + processed)
            chunk_skip_flags = []      # bool per chunk_read_objs entry: True = pass-through
            skipped = 0
            last_progress_reads = 0
            last_progress_time = time.time()

            def _buffer_skip(rd, reason):
                """Buffer a skipped read at its stream position to preserve sort order."""
                nonlocal skipped
                chunk_read_objs.append(rd)
                chunk_skip_flags.append(True)
                skipped += 1
                skip_reasons[reason] += 1

            try:
                for read in inbam.fetch(until_eof=True):
                    # --- Filter phase ---
                    # Skipped reads are buffered into the chunk (with skip_flag=True)
                    # rather than written immediately — the drain step writes them
                    # at their original stream position so a coordinate-sorted
                    # input BAM produces a coordinate-sorted output BAM.
                    skip_reason = streaming_skip_reason(read, filter_config)
                    if skip_reason:
                        _buffer_skip(read, skip_reason)
                        continue

                    # --- Build slim payload (fast — no MM/ML parsing) ---
                    # The MM/ML parse is now done inside the worker (see
                    # _process_payload_chunk_worker) which lifts the apply
                    # throughput ceiling from ~150-300 r/s (serial main parse
                    # bottleneck) to whatever the worker pool can sustain.
                    # Reads with no usable modification data or per-read worker
                    # failures come back as result=None and are written through
                    # without footprint tags (handled in _drain_oldest_chunk).
                    payload = make_apply_payload(read, mode=mode, ref_fasta=ref_fasta)
                    if payload is None:
                        _buffer_skip(read, 'no_modifications')
                        continue

                    # --- Buffer into chunk ---
                    chunk_reads.append(payload)
                    chunk_read_objs.append(read)
                    chunk_skip_flags.append(False)
                    total_reads += 1

                    # Check max reads limit
                    if max_reads and total_reads >= max_reads:
                        break

                    # --- Submit when chunk is full ---
                    # Use chunk_read_objs length so skipped reads also count toward
                    # the chunk size (otherwise pure-skip stretches would buffer
                    # unboundedly waiting for processable reads).
                    if len(chunk_read_objs) >= chunk_size:
                        # Drain oldest if at capacity
                        if len(inflight) >= max_inflight:
                            _drain_oldest_chunk(
                                inflight, outbam, with_scores, write_msps,
                                posterior_writer, counters
                            )

                        # Submit chunk of slim payloads to slim-IPC workers
                        future = executor.submit(
                            _process_payload_chunk_worker,
                            chunk_reads, edge_trim, circular, mode, context_size,
                            msp_min_size, nuc_min_size, with_scores, return_posteriors,
                            prob_threshold,
                        )
                        inflight.append((future, chunk_read_objs, chunk_reads, chunk_skip_flags))
                        chunk_reads = []
                        chunk_read_objs = []
                        chunk_skip_flags = []

                        # Progress update
                        now = time.time()
                        elapsed = now - start_time
                        avg_rate = total_reads / elapsed if elapsed > 0 else 0
                        dt = now - last_progress_time
                        inst_rate = (total_reads - last_progress_reads) / dt if dt > 0 else 0
                        last_progress_reads = total_reads
                        last_progress_time = now
                        print(f"\r  Processed: {total_reads:,} | "
                              f"Skipped: {skipped:,} | "
                              f"Inflight: {len(inflight)} | "
                              f"{inst_rate:.0f} reads/s (avg {avg_rate:.0f})", end='', file=_log)
                        _log.flush()

                # Submit final partial chunk (if it has any reads — skipped or not).
                # Pass-through-only tail (e.g. unmapped reads at end) still needs
                # to be flushed in order through the drain mechanism.
                if chunk_read_objs:
                    if len(inflight) >= max_inflight:
                        _drain_oldest_chunk(
                            inflight, outbam, with_scores, write_msps,
                            posterior_writer, counters
                        )

                    if chunk_reads:
                        future = executor.submit(
                            _process_payload_chunk_worker,
                            chunk_reads, edge_trim, circular, mode, context_size,
                            msp_min_size, nuc_min_size, with_scores, return_posteriors,
                            prob_threshold,
                        )
                    else:
                        # All-skipped tail: no worker call needed, fake a
                        # completed future returning [].
                        future = Future()
                        future.set_result([])
                    inflight.append((future, chunk_read_objs, chunk_reads, chunk_skip_flags))

                # Drain all remaining in-flight chunks (in order)
                while inflight:
                    _drain_oldest_chunk(
                        inflight, outbam, with_scores, write_msps,
                        posterior_writer, counters
                    )

            finally:
                executor.shutdown(wait=True)

    elapsed = time.time() - start_time
    rate = total_reads / elapsed if elapsed > 0 else 0
    reads_with_footprints = counters['reads_with_footprints']
    print(f"\r  Processed: {total_reads:,} | Skipped: {skipped:,} | "
          f"With footprints: {reads_with_footprints:,} | {rate:.1f} reads/s", file=_log)
    if counters['worker_failures']:
        print(
            f"  Worker read failures: {counters['worker_failures']:,} "
            f"(passed through unchanged)",
            file=_log,
        )

    # Print skip reasons
    if skipped > 0:
        print("  Skip reasons:", file=_log)
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = 100 * count / (total_reads + skipped)
                print(f"    {reason}: {count:,} ({pct:.1f}%)", file=_log)

    # Sort and index output BAM (skip for stdout and unaligned mode)
    if output_bam != '-' and not process_unmapped:
        _sort_and_index_bam(output_bam, threads=n_cores)

    # Close posteriors writer
    if posterior_writer:
        n_fibers, file_size = posterior_writer.close()
        print(f"Posteriors: {n_fibers:,} fibers -> {output_posteriors} ({file_size:.1f} MB)", file=_log)

    return total_reads, reads_with_footprints


def process_bam_for_footprints(input_bam: str, output_bam: str,
                                model_or_path, train_rids: Set[str],
                                edge_trim: int, circular: bool,
                                mode: str, context_size: int,
                                msp_min_size: int,
                                nuc_min_size: int = 85,
                                min_mapq: int = 0,
                                prob_threshold: int = 0,
                                min_read_length: int = 0,
                                with_scores: bool = False,
                                n_cores: int = 1,
                                max_reads: Optional[int] = None,
                                debug_timing: bool = False,
                                region_parallel: bool = False,
                                region_size: int = 10_000_000,
                                skip_scaffolds: bool = False,
                                chroms: Optional[Set[str]] = None,
                                primary_only: bool = False,
                                output_posteriors: Optional[str] = None,
                                write_msps: bool = True,
                                io_threads: int = 4,
                                streaming_pipeline: bool = False,
                                chunk_size: int = 500,
                                process_unmapped: bool = False) -> Tuple[int, int]:
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
        model = freeze_model_for_inference(load_model(model_path))
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
                nuc_min_size=nuc_min_size,
                min_mapq=min_mapq,
                prob_threshold=prob_threshold,
                min_read_length=min_read_length,
                with_scores=with_scores,
                n_cores=n_cores,
                region_size=region_size,
                skip_scaffolds=skip_scaffolds,
                chroms=chroms,
                primary_only=primary_only,
                output_posteriors=output_posteriors,
                write_msps=write_msps,
                io_threads=io_threads
            )

    # Dispatch to streaming pipeline if requested (requires model_path for workers)
    if streaming_pipeline:
        if model_path is None:
            print("Warning: streaming_pipeline requires model path, falling back to standard parallel")
        else:
            return _process_bam_streaming_pipeline(
                input_bam=input_bam,
                output_bam=output_bam,
                model_path=model_path,
                train_rids=train_rids,
                edge_trim=edge_trim,
                circular=circular,
                mode=mode,
                context_size=context_size,
                msp_min_size=msp_min_size,
                nuc_min_size=nuc_min_size,
                min_mapq=min_mapq,
                prob_threshold=prob_threshold,
                min_read_length=min_read_length,
                with_scores=with_scores,
                n_cores=n_cores,
                chunk_size=chunk_size,
                io_threads=io_threads,
                primary_only=primary_only,
                output_posteriors=output_posteriors,
                write_msps=write_msps,
                max_reads=max_reads,
                debug_timing=debug_timing,
                process_unmapped=process_unmapped,
            )

    total_reads = 0
    reads_with_footprints = 0
    written = 0
    skipped = 0
    worker_failures = 0

    # Track skip reasons
    skip_reasons = {
        'unmapped': 0,
        'secondary_supplementary': 0,
        'low_mapq': 0,
        'too_short': 0,
        'training_excluded': 0,
        'no_modifications': 0,
        'extraction_failed': 0,
        'no_footprints': 0,
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
    with pysam.AlignmentFile(input_bam, "rb", threads=io_threads, check_sq=False) as inbam:
        with pysam.AlignmentFile(output_bam, "wb", header=inbam.header, threads=io_threads) as outbam:

            chunk_reads = []  # Buffer for current chunk
            chunk_read_objs = []  # Corresponding pysam read objects

            if n_cores > 1 and model_path:
                # Parallel processing with ProcessPoolExecutor
                executor = ProcessPoolExecutor(
                    max_workers=n_cores,
                    mp_context=_MP_CONTEXT,
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
                            skip_reasons['no_modifications'] += 1
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
                        n_fp, n_nofp, n_failed, chunk_results = _process_and_write_chunk(
                            chunk_reads, chunk_read_objs, outbam,
                            model, executor, edge_trim, circular,
                            mode, context_size, msp_min_size, nuc_min_size=nuc_min_size,
                            with_scores=with_scores,
                            return_posteriors=return_posteriors,
                            write_msps=write_msps
                        )
                        reads_with_footprints += n_fp
                        skip_reasons['no_footprints'] += n_nofp
                        worker_failures += n_failed
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
                    n_fp, n_nofp, n_failed, chunk_results = _process_and_write_chunk(
                        chunk_reads, chunk_read_objs, outbam,
                        model, executor, edge_trim, circular,
                        mode, context_size, msp_min_size, nuc_min_size=nuc_min_size,
                        with_scores=with_scores,
                        return_posteriors=return_posteriors,
                        write_msps=write_msps,
                    )
                    reads_with_footprints += n_fp
                    skip_reasons['no_footprints'] += n_nofp
                    worker_failures += n_failed
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
    if worker_failures:
        print(f"  Worker read failures: {worker_failures:,} (passed through unchanged)")

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
