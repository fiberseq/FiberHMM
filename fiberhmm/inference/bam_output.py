"""FiberHMM BAM/BED output utilities."""

import os
import shutil
import subprocess
import sys
import time
from typing import List, Optional, Sequence

import numpy as np
import pysam

from fiberhmm.core.bam_reader import get_bam_chrom_sizes
from fiberhmm.inference.reference_mapping import (
    build_query_to_ref,
    scored_interval_spans,
)
from fiberhmm.inference.read_filters import is_primary_mapped_alignment
from fiberhmm.io.bed import bed12_row
from fiberhmm.io.ma_tags import flip_intervals_to_seq


_BED12_RECORD_COLUMNS = (
    'chrom',
    'chromStart',
    'chromEnd',
    'name',
    'score',
    'strand',
    'thickStart',
    'thickEnd',
    'itemRgb',
    'blockCount',
    'blockSizes',
    'blockStarts',
)


def _samtools_index_cmd(output_bam: str, threads: int) -> List[str]:
    return ['samtools', 'index', '-@', str(threads), output_bam]


def _samtools_sort_cmd(output_bam: str, sorted_bam: str, threads: int) -> List[str]:
    return ['samtools', 'sort', '-@', str(threads), '-o', sorted_bam, output_bam]


def _samtools_cat_cmd(
    bam_files: List[str],
    output_bam: str,
    list_file: str,
) -> List[str]:
    return ['samtools', 'cat', '-h', bam_files[0], '-b', list_file, '-o', output_bam]


def _samtools_merge_cmd(output_bam: str, list_file: str) -> List[str]:
    return ['samtools', 'merge', '-f', '-b', list_file, output_bam]


def _run_samtools_index(output_bam: str, threads: int, check: bool = False) -> subprocess.CompletedProcess:
    """Run `samtools index` with the shared command shape."""
    return subprocess.run(
        _samtools_index_cmd(output_bam, threads),
        check=check, capture_output=True, text=True
    )


def _run_samtools_sort(output_bam: str, sorted_bam: str, threads: int) -> None:
    """Run `samtools sort` and raise with stderr preserved on failure."""
    result = subprocess.run(
        _samtools_sort_cmd(output_bam, sorted_bam, threads),
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, 'samtools sort',
            output=result.stdout, stderr=result.stderr,
        )


def _samtools_index_error_requires_sort(stderr: str) -> bool:
    error_text = (stderr or '').lower()
    return 'not sorted' in error_text or 'coordinate' in error_text


def _sorted_bam_temp_path(output_bam: str) -> str:
    if output_bam.endswith('.bam'):
        return output_bam[:-4] + '.sorted.bam'
    return output_bam + '.sorted.bam'


def _file_size_gb(path: str) -> float:
    return os.path.getsize(path) / (1024 ** 3)


def _total_file_size_gb(paths: Sequence[str]) -> float:
    return sum(os.path.getsize(path) for path in paths) / (1024 ** 3)


def _throughput_gbs(size_gb: float, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0:
        return 0.0
    return size_gb / elapsed_seconds


def _try_pysam_index(output_bam: str, verbose: bool, idx_start: float) -> bool:
    try:
        pysam.index(output_bam)
    except pysam.utils.SamtoolsError:
        return False

    if verbose:
        idx_time = time.time() - idx_start
        print(f"  ✓ Index created (pysam) in {idx_time:.1f}s")
    return True


def _sort_bam_with_fallback(
    output_bam: str,
    sorted_bam: str,
    threads: int,
    verbose: bool,
    bam_size_gb: float,
) -> None:
    sort_start = time.time()
    try:
        _run_samtools_sort(output_bam, sorted_bam, threads)
        if verbose:
            sort_time = time.time() - sort_start
            speed = _throughput_gbs(bam_size_gb, sort_time)
            print(f"  Sorted in {sort_time:.1f}s ({speed:.2f} GB/s)")
    except (subprocess.CalledProcessError, FileNotFoundError):
        if verbose:
            print("  Using pysam sort (slower)...")
        pysam.sort("-o", sorted_bam, output_bam)
        if verbose:
            sort_time = time.time() - sort_start
            print(f"  Sorted (pysam) in {sort_time:.1f}s")


def _index_sorted_bam(
    output_bam: str,
    threads: int,
    verbose: bool,
    bam_size_gb: float,
) -> None:
    if verbose:
        print("  Indexing sorted BAM...")
        sys.stdout.flush()

    idx_start = time.time()
    try:
        _run_samtools_index(output_bam, threads, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pysam.index(output_bam)

    if verbose:
        idx_time = time.time() - idx_start
        speed = _throughput_gbs(bam_size_gb, idx_time)
        print(f"  ✓ Index created in {idx_time:.1f}s ({speed:.2f} GB/s)")


def _sort_and_index_bam(output_bam: str, verbose: bool = True, threads: int = 4):
    """
    Index a BAM file, sorting first only if needed.

    For region-parallel processing, the output should already be sorted
    since regions are processed and concatenated in order. Try indexing
    first to save time; only sort if indexing fails.
    """
    bam_size_gb = _file_size_gb(output_bam)

    # Try indexing directly first (using samtools with threads for speed)
    try:
        if verbose:
            print(f"  Indexing BAM ({bam_size_gb:.1f}GB) - trying direct index (no sort)...")
            sys.stdout.flush()

        idx_start = time.time()
        result = _run_samtools_index(output_bam, threads)
        if result.returncode == 0:
            if verbose:
                idx_time = time.time() - idx_start
                speed = _throughput_gbs(bam_size_gb, idx_time)
                print(f"  ✓ Index created in {idx_time:.1f}s ({speed:.2f} GB/s) - BAM was already sorted!")
            return

        # samtools failed - check if it's a sort issue
        if _samtools_index_error_requires_sort(result.stderr):
            if verbose:
                print("  ✗ Direct index failed - BAM is NOT sorted")
                print(f"    Error: {result.stderr.strip()}")
            # Fall through to sorting
        else:
            # Some other error, try pysam
            if verbose:
                print(f"  samtools index error: {result.stderr.strip()}, trying pysam...")
            if _try_pysam_index(output_bam, verbose, idx_start):
                return

    except FileNotFoundError:
        # samtools not found, try pysam
        if verbose:
            print("  samtools not found, using pysam...")
        idx_start = time.time()
        if _try_pysam_index(output_bam, verbose, idx_start):
            return
    except pysam.utils.SamtoolsError:
        pass

    # Indexing failed - BAM is not sorted, need to sort first
    if verbose:
        print(f"  Sorting BAM ({bam_size_gb:.1f}GB) with samtools sort -@ {threads}...")
        print("    (This means samtools cat did not preserve sort order)")
        sys.stdout.flush()

    # Sort using samtools (faster than pysam for large files)
    sorted_bam = _sorted_bam_temp_path(output_bam)
    _sort_bam_with_fallback(output_bam, sorted_bam, threads, verbose, bam_size_gb)
    os.replace(sorted_bam, output_bam)
    _index_sorted_bam(output_bam, threads, verbose, bam_size_gb)


def _write_bam_list_file(bam_files: List[str], list_file: str) -> None:
    """Write a samtools-compatible BAM list file."""
    with open(list_file, 'w') as f:
        for bam_path in bam_files:
            f.write(bam_path + '\n')


def _remove_file_if_exists(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def _raise_if_command_failed(result: subprocess.CompletedProcess, label: str) -> None:
    if result.returncode == 0:
        return
    raise subprocess.CalledProcessError(
        result.returncode,
        label,
        output=result.stdout,
        stderr=result.stderr,
    )


def _run_samtools_list_command(
    bam_files: List[str],
    list_file: str,
    cmd: List[str],
    label: str,
) -> None:
    _write_bam_list_file(bam_files, list_file)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        _raise_if_command_failed(result, label)
    finally:
        _remove_file_if_exists(list_file)


def _samtools_cat_bams(bam_files: List[str], output_bam: str, list_file: str) -> None:
    """Concatenate BAMs with `samtools cat -b`, cleaning the list file.

    ``-h bam_files[0]`` forces the output header to the first region BAM's header
    verbatim. Without it, ``samtools cat`` merges the @PG lines from every input,
    which would duplicate FiberHMM's @PG provenance line once per region.
    """
    _run_samtools_list_command(
        bam_files,
        list_file,
        _samtools_cat_cmd(bam_files, output_bam, list_file),
        'samtools cat',
    )


def _samtools_merge_bams(bam_files: List[str], output_bam: str, list_file: str) -> None:
    """Merge BAMs with `samtools merge -b`, cleaning the list file."""
    _run_samtools_list_command(
        bam_files,
        list_file,
        _samtools_merge_cmd(output_bam, list_file),
        'samtools merge',
    )


def _remove_partial_output_bam(output_bam: str, verbose: bool = True) -> None:
    """Remove a partial output BAM left by a failed external tool."""
    if not os.path.exists(output_bam):
        return
    try:
        os.remove(output_bam)
        if verbose:
            print("    Removed partial output file")
    except Exception as rm_err:
        if verbose:
            print(f"    Warning: Could not remove partial file: {rm_err}")


def _ensure_output_dir_writable(output_bam: str, verbose: bool = True) -> None:
    """Create and probe the output directory before slow fallback writes."""
    output_dir_path = os.path.dirname(output_bam)
    if not output_dir_path:
        return

    if not os.path.exists(output_dir_path):
        if verbose:
            print(f"    Creating output directory: {output_dir_path}")
        os.makedirs(output_dir_path, exist_ok=True)

    test_file = os.path.join(output_dir_path, '.write_test')
    try:
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
    except Exception as write_err:
        if verbose:
            print(f"    ERROR: Cannot write to output directory: {write_err}")
            print(f"    Directory: {output_dir_path}")
            print(f"    Directory exists: {os.path.exists(output_dir_path)}")
        raise


def _prepare_pysam_concat_fallback(
    output_bam: str,
    error: Exception,
    verbose: bool = True,
) -> None:
    if verbose:
        stderr = getattr(error, 'stderr', None)
        if stderr:
            print(f"  WARNING: samtools cat failed: {stderr.strip()}")
        else:
            print(f"  WARNING: samtools cat failed: {error}")
        print("  Falling back to pysam (slower, reads each record)...")

    _remove_partial_output_bam(output_bam, verbose=verbose)
    _ensure_output_dir_writable(output_bam, verbose=verbose)


def _log_pysam_concat_failure(output_bam: str, error: Exception) -> None:
    output_dir_path = os.path.dirname(output_bam)
    print(f"  ERROR: pysam fallback also failed: {error}")
    print(f"    Output path: {output_bam}")
    print(f"    Output dir exists: {os.path.exists(output_dir_path)}")
    print("  Attempting manual BAM concatenation via samtools merge...")


def _write_empty_bam_from_input_header(input_bam: str, output_bam: str) -> None:
    """Create an empty BAM using the input BAM header."""
    with pysam.AlignmentFile(input_bam, "rb", check_sq=False) as inbam:
        with pysam.AlignmentFile(output_bam, "wb", header=inbam.header):
            pass


def _concatenate_trivial_region_bams(
    input_bam: str,
    output_bam: str,
    bam_files: List[str],
) -> bool:
    if len(bam_files) == 0:
        _write_empty_bam_from_input_header(input_bam, output_bam)
        return True
    if len(bam_files) == 1:
        shutil.copy(bam_files[0], output_bam)
        return True
    return False


def _concatenate_bams_with_pysam(
    bam_files: List[str],
    output_bam: str,
    progress_every: int = 10,
    verbose: bool = True,
) -> None:
    """Concatenate BAMs by reading and writing records with pysam."""
    if verbose:
        print(f"    Reading header from: {bam_files[0]}")
    with pysam.AlignmentFile(bam_files[0], "rb", check_sq=False) as first_bam:
        header = first_bam.header

    if verbose:
        print(f"    Opening output file: {output_bam}")
    with pysam.AlignmentFile(output_bam, "wb", header=header) as outbam:
        for i, bam_path in enumerate(bam_files):
            with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as inbam:
                for read in inbam:
                    outbam.write(read)
            if verbose and progress_every > 0 and (i + 1) % progress_every == 0:
                print(f"\r    Concatenated {i+1}/{len(bam_files)} BAMs...", end='')
                sys.stdout.flush()
    if verbose:
        print()


def _concatenate_region_bams(
    input_bam: str,
    output_bam: str,
    bam_files: List[str],
    temp_dir: str,
    verbose: bool = True,
) -> None:
    """Concatenate sorted region BAMs with fast external tools and fallbacks."""
    import time

    total_temp_size_gb = _total_file_size_gb(bam_files)

    if verbose:
        print(f"Concatenating {len(bam_files)} region BAMs ({total_temp_size_gb:.1f}GB total)...")
        sys.stdout.flush()

    concat_start = time.time()

    if _concatenate_trivial_region_bams(input_bam, output_bam, bam_files):
        return

    bam_list_file = os.path.join(temp_dir, 'bam_list.txt')
    try:
        _samtools_cat_bams(bam_files, output_bam, bam_list_file)

        if verbose:
            concat_time = time.time() - concat_start
            output_size_gb = _file_size_gb(output_bam)
            speed_gbs = _throughput_gbs(output_size_gb, concat_time)
            print(
                f"  Concatenated with samtools cat in {concat_time:.1f}s "
                f"({output_size_gb:.1f}GB, {speed_gbs:.2f} GB/s)"
            )
        return

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        _prepare_pysam_concat_fallback(output_bam, e, verbose=verbose)

    try:
        _concatenate_bams_with_pysam(bam_files, output_bam, verbose=verbose)
        if verbose:
            concat_time = time.time() - concat_start
            print(f"  Concatenated with pysam in {concat_time:.1f}s")

    except Exception as pysam_err:
        if verbose:
            _log_pysam_concat_failure(output_bam, pysam_err)

        try:
            _samtools_merge_bams(bam_files, output_bam, bam_list_file)
            if verbose:
                concat_time = time.time() - concat_start
                print(f"  Concatenated with samtools merge in {concat_time:.1f}s")
        except Exception as merge_err:
            if verbose:
                print(f"  ERROR: All concatenation methods failed: {merge_err}")
            raise


def write_bed12_records_direct(records: List[dict], filepath: str, with_scores: bool = False):
    """
    Write BED12 records directly to file without DataFrame overhead.
    Much faster for small-to-medium chunk sizes.
    """
    if not records:
        return

    cols = _bed12_record_columns(with_scores)

    with open(filepath, 'w') as f:
        for rec in records:
            line = '\t'.join(str(rec[col]) for col in cols)
            f.write(line + '\n')


def _sort_bed_for_bigbed(bed_file: str, sorted_bed: str) -> None:
    """Sort BED records by chrom/start before UCSC bigBed conversion."""
    subprocess.run(
        ['sort', '-k1,1', '-k2,2n', '-o', sorted_bed, bed_file],
        check=True,
    )


def _sorted_bed_temp_path(bed_file: str) -> str:
    return bed_file + '.sorted'


def _bed_to_bigbed_command(
    sorted_bed: str,
    chrom_sizes: str,
    output_bb: str,
    bed_type: str = 'bed12',
    autosql: Optional[str] = None,
) -> List[str]:
    cmd = ['bedToBigBed', f'-type={bed_type}']
    if autosql is not None:
        cmd.append('-as=' + autosql)
    cmd.extend([sorted_bed, chrom_sizes, output_bb])
    return cmd


def _run_bed_to_bigbed(
    sorted_bed: str,
    chrom_sizes: str,
    output_bb: str,
    bed_type: str = 'bed12',
    autosql: Optional[str] = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        _bed_to_bigbed_command(sorted_bed, chrom_sizes, output_bb, bed_type, autosql),
        capture_output=True,
        text=True,
    )


def _bed_to_bigbed_available() -> bool:
    if not shutil.which('bedToBigBed'):
        print("Warning: bedToBigBed not found in PATH. Skipping bigBed conversion.")
        print("Download from: https://hgdownload.soe.ucsc.edu/admin/exe/")
        return False
    return True


def _convert_sorted_bed_to_bigbed(
    sorted_bed: str,
    chrom_sizes: str,
    output_bb: str,
    bed_type: str = 'bed12',
    autosql: Optional[str] = None,
) -> bool:
    result = _run_bed_to_bigbed(
        sorted_bed, chrom_sizes, output_bb, bed_type, autosql
    )
    if result.returncode == 0:
        return True

    print(f"bedToBigBed error: {result.stderr}")
    return False


def _bed_type_for_scores(with_scores: bool) -> str:
    return 'bed12+1' if with_scores else 'bed12'


def _convert_bigbed_score_fallback(
    sorted_bed: str,
    chrom_sizes: str,
    output_bb: str,
) -> bool:
    print("Trying fallback to standard BED12...")
    result = _run_bed_to_bigbed(sorted_bed, chrom_sizes, output_bb)
    if result.returncode != 0:
        return False

    print("Fallback succeeded (scores in BED only, not bigBed)")
    return True


def _convert_sorted_bed_to_bigbed_with_optional_fallback(
    sorted_bed: str,
    chrom_sizes: str,
    autosql: str,
    output_bb: str,
    with_scores: bool,
) -> bool:
    bed_type = _bed_type_for_scores(with_scores)
    converted = _convert_sorted_bed_to_bigbed(
        sorted_bed, chrom_sizes, output_bb, bed_type, autosql
    )
    if converted:
        return True

    if with_scores:
        return _convert_bigbed_score_fallback(sorted_bed, chrom_sizes, output_bb)
    return False


def convert_to_bigbed(bed_file: str, chrom_sizes: str, output_bb: str) -> bool:
    """Convert BED12 to bigBed format."""
    if not _bed_to_bigbed_available():
        return False

    sorted_bed = _sorted_bed_temp_path(bed_file)
    try:
        # Sort BED file
        _sort_bed_for_bigbed(bed_file, sorted_bed)

        # Convert
        return _convert_sorted_bed_to_bigbed(sorted_bed, chrom_sizes, output_bb)

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error during bigBed conversion: {e}")
        return False
    finally:
        _remove_file_if_exists(sorted_bed)


def write_chrom_sizes(bam_path: str, output_path: str) -> str:
    """Extract and write chromosome sizes from BAM header."""
    sizes = get_bam_chrom_sizes(bam_path)
    with open(output_path, 'w') as f:
        for chrom, size in sorted(sizes.items()):
            f.write(f"{chrom}\t{size}\n")
    return output_path


def _bed12_record_columns(with_scores: bool = False) -> List[str]:
    cols = list(_BED12_RECORD_COLUMNS)
    if with_scores:
        cols.append('blockScores')
    return cols


def _autosql_schema(with_scores: bool = False) -> str:
    schema = '''table fiberFootprints
"FiberHMM footprint calls with per-footprint confidence scores"
    (
    string chrom;        "Chromosome"
    uint chromStart;     "Start position"
    uint chromEnd;       "End position"
    string name;         "Read ID"
    uint score;          "Mean footprint confidence (0-1000)"
    char[1] strand;      "Strand"
    uint thickStart;     "Start position (same as chromStart)"
    uint thickEnd;       "End position (same as chromEnd)"
    string itemRgb;      "Color"
    int blockCount;      "Number of footprints"
    int[blockCount] blockSizes;   "Footprint sizes"
    int[blockCount] blockStarts;  "Footprint start positions"
'''
    if with_scores:
        schema += '''    int[blockCount] blockScores;  "Per-footprint confidence scores (0-1000)"
'''
    schema += '''    )
'''
    return schema


def write_autosql_schema(filepath: str, with_scores: bool = False):
    """
    Write autoSql schema file for bigBed conversion.

    This defines the extended BED12+ format with blockScores.
    """
    with open(filepath, 'w') as f:
        f.write(_autosql_schema(with_scores))


def convert_to_bigbed_with_schema(bed_file: str, chrom_sizes: str,
                                   autosql: str, output_bb: str,
                                   with_scores: bool = False) -> bool:
    """Convert BED12+ to bigBed format using custom autoSql schema."""
    if not _bed_to_bigbed_available():
        return False

    sorted_bed = _sorted_bed_temp_path(bed_file)
    try:
        # Sort BED file
        _sort_bed_for_bigbed(bed_file, sorted_bed)

        return _convert_sorted_bed_to_bigbed_with_optional_fallback(
            sorted_bed, chrom_sizes, autosql, output_bb, with_scores,
        )

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error during bigBed conversion: {e}")
        return False
    finally:
        _remove_file_if_exists(sorted_bed)


def _format_footprint_bed12_row(
    chrom: str,
    chrom_start: int,
    chrom_end: int,
    name: str,
    strand: str,
    block_starts,
    block_sizes,
    valid_scores,
    with_scores: bool,
) -> str:
    score = len(block_starts)
    blocks = _bed_blocks_from_relative(chrom_start, block_starts, block_sizes)
    extra = ()
    if with_scores and valid_scores:
        extra = (_format_bed_score_column(valid_scores),)
    return bed12_row(
        chrom,
        chrom_start,
        chrom_end,
        name,
        score,
        strand,
        blocks,
        extra,
        item_rgb="0,0,0",
    )


def _bed_blocks_from_relative(chrom_start: int, block_starts, block_sizes):
    return [
        (chrom_start + start, chrom_start + start + size)
        for start, size in zip(block_starts, block_sizes)
    ]


def _format_bed_score_column(scores: Sequence[int]) -> str:
    scaled_scores = [int(score * 1000 / 255) for score in scores]
    return ','.join(str(score) for score in scaled_scores)


def _sort_bed12_blocks(block_starts, block_sizes, valid_scores):
    if valid_scores:
        sorted_indices = sorted(range(len(block_starts)), key=lambda i: block_starts[i])
        return (
            [block_starts[i] for i in sorted_indices],
            [block_sizes[i] for i in sorted_indices],
            [valid_scores[i] for i in sorted_indices],
        )

    sorted_pairs = sorted(zip(block_starts, block_sizes))
    return [p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs], []


def _merge_bed12_blocks(block_starts, block_sizes, valid_scores):
    merged_starts = [block_starts[0]]
    merged_sizes = [block_sizes[0]]
    merged_scores = [valid_scores[0]] if valid_scores else []

    for i in range(1, len(block_starts)):
        prev_end = merged_starts[-1] + merged_sizes[-1]
        curr_start = block_starts[i]
        curr_size = block_sizes[i]

        if curr_start <= prev_end:  # Adjacent or overlapping
            new_end = max(prev_end, curr_start + curr_size)
            merged_sizes[-1] = new_end - merged_starts[-1]
            if merged_scores:
                merged_scores[-1] = (merged_scores[-1] + valid_scores[i]) // 2
        else:
            merged_starts.append(curr_start)
            merged_sizes.append(curr_size)
            if valid_scores:
                merged_scores.append(valid_scores[i])

    return merged_starts, merged_sizes, merged_scores


def _pad_bed12_blocks(block_starts, block_sizes, valid_scores, read_length: int):
    # BED12 requires blocks to span chromStart to chromEnd. Add 1bp padding
    # blocks at start/end if needed.
    if block_starts[0] != 0:
        block_starts.insert(0, 0)
        block_sizes.insert(0, 1)
        if valid_scores:
            valid_scores.insert(0, 0)

    last_end = block_starts[-1] + block_sizes[-1]
    if last_end < read_length:
        block_starts.append(read_length - 1)
        block_sizes.append(1)
        if valid_scores:
            valid_scores.append(0)

    return block_starts, block_sizes, valid_scores


def _normalize_bed12_blocks(block_starts, block_sizes, valid_scores, read_length: int):
    block_starts, block_sizes, valid_scores = _sort_bed12_blocks(
        block_starts, block_sizes, valid_scores,
    )
    block_starts, block_sizes, valid_scores = _merge_bed12_blocks(
        block_starts, block_sizes, valid_scores,
    )
    return _pad_bed12_blocks(block_starts, block_sizes, valid_scores, read_length)


def _bed12_components_from_ref_blocks(blocks, with_scores: bool):
    ref_starts = [start for start, _, _ in blocks]
    if not ref_starts:
        return None

    ref_ends = [end for _, end, _ in blocks]
    valid_scores = [score for _, _, score in blocks] if with_scores else []
    chrom_start = min(ref_starts)
    chrom_end = max(ref_ends)
    block_sizes = [end - start for start, end in zip(ref_starts, ref_ends)]
    block_starts = [start - chrom_start for start in ref_starts]
    return chrom_start, chrom_end, block_starts, block_sizes, valid_scores


def _footprint_bed12_line_from_read(read, ns, nl, nq, with_scores: bool) -> Optional[str]:
    chrom = read.reference_name
    strand = '-' if read.is_reverse else '+'

    query_to_ref = build_query_to_ref(read)
    blocks = scored_interval_spans(ns, nl, nq if with_scores else None, query_to_ref)
    components = _bed12_components_from_ref_blocks(blocks, with_scores)
    if components is None:
        return None

    chrom_start, chrom_end, block_starts, block_sizes, valid_scores = components

    read_length = chrom_end - chrom_start
    block_starts, block_sizes, valid_scores = _normalize_bed12_blocks(
        block_starts,
        block_sizes,
        valid_scores,
        read_length,
    )

    return _format_footprint_bed12_row(
        chrom,
        chrom_start,
        chrom_end,
        read.query_name,
        strand,
        block_starts,
        block_sizes,
        valid_scores,
        with_scores,
    )


def _read_footprint_tags_for_bed(read, with_scores: bool):
    try:
        ns, nl = flip_intervals_to_seq(
            read.get_tag('ns'), read.get_tag('nl'), read)
    except KeyError:
        return None

    if len(ns) == 0:
        return None

    nq = None
    if with_scores:
        try:
            nq = read.get_tag('nq')
        except KeyError:
            pass
    return ns, nl, nq


def _bed12_line_from_tagged_read(read, with_scores: bool) -> Optional[str]:
    if not is_primary_mapped_alignment(read):
        return None

    tags = _read_footprint_tags_for_bed(read, with_scores)
    if tags is None:
        return None
    ns, nl, nq = tags

    return _footprint_bed12_line_from_read(read, ns, nl, nq, with_scores)


def extract_bed_from_tagged_bam(input_bam: str, output_bed: str,
                                  with_scores: bool = False,
                                  n_cores: int = 1) -> int:
    """
    Extract BED12 footprints from a BAM file with ns/nl tags.

    This is MUCH faster than re-processing because we just read tags,
    no HMM or encoding needed.

    Args:
        input_bam: Path to BAM with ns/nl footprint tags
        output_bed: Output BED12 file path
        with_scores: Include nq scores as extra column
        n_cores: Number of cores (for parallel reading)

    Returns:
        Number of reads with footprints written
    """
    import pysam

    count = 0

    with pysam.AlignmentFile(input_bam, "rb", check_sq=False) as bam, open(output_bed, 'w') as out:
        for read in bam:
            line = _bed12_line_from_tagged_read(read, with_scores)
            if line is None:
                continue

            out.write(line + '\n')
            count += 1

    return count


def _parse_int_csv(value: str) -> List[int]:
    return [int(x) for x in value.split(',') if x]


def _parse_score_block_values(record: dict):
    block_sizes = _parse_int_csv(record['blockSizes'])
    block_starts = _parse_int_csv(record['blockStarts'])

    if 'blockScores' in record:
        block_scores = _parse_int_csv(record['blockScores'])
    else:
        block_scores = [0] * len(block_sizes)

    return block_starts, block_sizes, block_scores


def _mean_block_score(block_scores: Sequence[int]):
    return np.mean(block_scores) if block_scores else 0


def _score_read_row(record: dict, mean_score):
    return (
        record['name'],
        record['chrom'],
        record['chromStart'],
        record['chromEnd'],
        record['strand'],
        record['blockCount'],
        mean_score,
    )


def _footprint_score_rows(
    read_id: str,
    block_starts,
    block_sizes,
    block_scores,
) -> List[tuple]:
    return [
        (read_id, i, start, size, score)
        for i, (start, size, score) in enumerate(
            zip(block_starts, block_sizes, block_scores)
        )
    ]


def _insert_footprint_score_records(
    cursor,
    read_id: str,
    block_starts,
    block_sizes,
    block_scores,
) -> None:
    for row in _footprint_score_rows(
        read_id, block_starts, block_sizes, block_scores,
    ):
        cursor.execute('''
            INSERT INTO footprints
            (read_id, footprint_idx, rel_start, size, score)
            VALUES (?, ?, ?, ?, ?)
        ''', row)


def _insert_score_record(cursor, record: dict) -> None:
    block_starts, block_sizes, block_scores = _parse_score_block_values(record)
    mean_score = _mean_block_score(block_scores)

    cursor.execute('''
        INSERT OR REPLACE INTO reads
        (read_id, chrom, start, end, strand, n_footprints, mean_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', _score_read_row(record, mean_score))

    _insert_footprint_score_records(
        cursor,
        record['name'],
        block_starts,
        block_sizes,
        block_scores,
    )


def _insert_score_records(cursor, records: List[dict]) -> None:
    for record in records:
        _insert_score_record(cursor, record)


def _create_scores_schema(cursor) -> None:
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reads (
            read_id TEXT PRIMARY KEY,
            chrom TEXT,
            start INTEGER,
            end INTEGER,
            strand TEXT,
            n_footprints INTEGER,
            mean_score REAL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS footprints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            read_id TEXT,
            footprint_idx INTEGER,
            rel_start INTEGER,
            size INTEGER,
            score INTEGER,
            FOREIGN KEY (read_id) REFERENCES reads(read_id)
        )
    ''')

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_footprints_read_id ON footprints(read_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_reads_chrom_start ON reads(chrom, start)')


def _write_scores_database(
    records: List[dict],
    db_path: str,
    *,
    create_schema: bool,
) -> None:
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        if create_schema:
            _create_scores_schema(cursor)
        _insert_score_records(cursor, records)

        conn.commit()
    finally:
        conn.close()


def create_scores_database(records: List[dict], db_path: str):
    """
    Create SQLite database with detailed per-footprint scores.

    Schema:
        reads: read_id, chrom, start, end, strand, n_footprints, mean_score
        footprints: read_id, footprint_idx, start, size, score

    This allows fast lookup by read_id for downstream analysis.
    """
    _write_scores_database(records, db_path, create_schema=True)


def append_to_scores_database(records: List[dict], db_path: str):
    """Append records to existing scores database."""
    _write_scores_database(records, db_path, create_schema=False)
