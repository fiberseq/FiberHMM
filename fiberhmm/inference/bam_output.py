"""FiberHMM BAM/BED output utilities."""

import os
import sys
import subprocess
import shutil
import numpy as np
import pysam
from typing import List, Dict, Optional

from fiberhmm.core.bam_reader import get_bam_chrom_sizes


def _sort_and_index_bam(output_bam: str, verbose: bool = True, threads: int = 4):
    """
    Index a BAM file, sorting first only if needed.

    For region-parallel processing, the output should already be sorted
    since regions are processed and concatenated in order. Try indexing
    first to save time; only sort if indexing fails.
    """
    import time

    bam_size_mb = os.path.getsize(output_bam) / (1024 * 1024)
    bam_size_gb = bam_size_mb / 1024

    # Try indexing directly first (using samtools with threads for speed)
    try:
        if verbose:
            print(f"  Indexing BAM ({bam_size_gb:.1f}GB) - trying direct index (no sort)...")
            sys.stdout.flush()

        idx_start = time.time()
        result = subprocess.run(
            ['samtools', 'index', '-@', str(threads), output_bam],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            if verbose:
                idx_time = time.time() - idx_start
                speed = bam_size_gb / idx_time if idx_time > 0 else 0
                print(f"  ✓ Index created in {idx_time:.1f}s ({speed:.2f} GB/s) - BAM was already sorted!")
            return

        # samtools failed - check if it's a sort issue
        if 'not sorted' in result.stderr.lower() or 'coordinate' in result.stderr.lower():
            if verbose:
                print(f"  ✗ Direct index failed - BAM is NOT sorted")
                print(f"    Error: {result.stderr.strip()}")
            # Fall through to sorting
        else:
            # Some other error, try pysam
            if verbose:
                print(f"  samtools index error: {result.stderr.strip()}, trying pysam...")
            pysam.index(output_bam)
            if verbose:
                idx_time = time.time() - idx_start
                print(f"  ✓ Index created (pysam) in {idx_time:.1f}s")
            return

    except FileNotFoundError:
        # samtools not found, try pysam
        if verbose:
            print("  samtools not found, using pysam...")
        try:
            idx_start = time.time()
            pysam.index(output_bam)
            if verbose:
                idx_time = time.time() - idx_start
                print(f"  ✓ Index created (pysam) in {idx_time:.1f}s")
            return
        except pysam.utils.SamtoolsError:
            pass
    except pysam.utils.SamtoolsError:
        pass

    # Indexing failed - BAM is not sorted, need to sort first
    if verbose:
        print(f"  Sorting BAM ({bam_size_gb:.1f}GB) with samtools sort -@ {threads}...")
        print(f"    (This means samtools cat did not preserve sort order)")
        sys.stdout.flush()

    # Sort using samtools (faster than pysam for large files)
    sorted_bam = output_bam.replace('.bam', '.sorted.bam')
    sort_start = time.time()
    try:
        result = subprocess.run(
            ['samtools', 'sort', '-@', str(threads), '-o', sorted_bam, output_bam],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, 'samtools sort', result.stderr)
        if verbose:
            sort_time = time.time() - sort_start
            speed = bam_size_gb / sort_time if sort_time > 0 else 0
            print(f"  Sorted in {sort_time:.1f}s ({speed:.2f} GB/s)")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to pysam
        if verbose:
            print("  Using pysam sort (slower)...")
        pysam.sort("-o", sorted_bam, output_bam)
        if verbose:
            sort_time = time.time() - sort_start
            print(f"  Sorted (pysam) in {sort_time:.1f}s")

    os.replace(sorted_bam, output_bam)

    if verbose:
        print(f"  Indexing sorted BAM...")
        sys.stdout.flush()

    # Index the sorted BAM
    idx_start = time.time()
    try:
        subprocess.run(['samtools', 'index', '-@', str(threads), output_bam], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pysam.index(output_bam)

    if verbose:
        idx_time = time.time() - idx_start
        speed = bam_size_gb / idx_time if idx_time > 0 else 0
        print(f"  ✓ Index created in {idx_time:.1f}s ({speed:.2f} GB/s)")


def write_bed12_records_direct(records: List[dict], filepath: str, with_scores: bool = False):
    """
    Write BED12 records directly to file without DataFrame overhead.
    Much faster for small-to-medium chunk sizes.
    """
    if not records:
        return

    cols = ['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand',
            'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockStarts']
    if with_scores:
        cols.append('blockScores')

    with open(filepath, 'w') as f:
        for rec in records:
            line = '\t'.join(str(rec[col]) for col in cols)
            f.write(line + '\n')


def convert_to_bigbed(bed_file: str, chrom_sizes: str, output_bb: str) -> bool:
    """Convert BED12 to bigBed format."""
    if not shutil.which('bedToBigBed'):
        print("Warning: bedToBigBed not found in PATH. Skipping bigBed conversion.")
        print("Download from: https://hgdownload.soe.ucsc.edu/admin/exe/")
        return False

    sorted_bed = bed_file + '.sorted'
    try:
        # Sort BED file
        subprocess.run(
            f"sort -k1,1 -k2,2n {bed_file} > {sorted_bed}",
            shell=True, check=True
        )

        # Convert
        result = subprocess.run(
            ['bedToBigBed', '-type=bed12', sorted_bed, chrom_sizes, output_bb],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"bedToBigBed error: {result.stderr}")
            return False

        return True

    except subprocess.CalledProcessError as e:
        print(f"Error during bigBed conversion: {e}")
        return False
    finally:
        if os.path.exists(sorted_bed):
            os.remove(sorted_bed)


def write_chrom_sizes(bam_path: str, output_path: str) -> str:
    """Extract and write chromosome sizes from BAM header."""
    sizes = get_bam_chrom_sizes(bam_path)
    with open(output_path, 'w') as f:
        for chrom, size in sorted(sizes.items()):
            f.write(f"{chrom}\t{size}\n")
    return output_path


def write_autosql_schema(filepath: str, with_scores: bool = False):
    """
    Write autoSql schema file for bigBed conversion.

    This defines the extended BED12+ format with blockScores.
    """
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

    with open(filepath, 'w') as f:
        f.write(schema)


def convert_to_bigbed_with_schema(bed_file: str, chrom_sizes: str,
                                   autosql: str, output_bb: str,
                                   with_scores: bool = False) -> bool:
    """Convert BED12+ to bigBed format using custom autoSql schema."""
    if not shutil.which('bedToBigBed'):
        print("Warning: bedToBigBed not found in PATH. Skipping bigBed conversion.")
        print("Download from: https://hgdownload.soe.ucsc.edu/admin/exe/")
        return False

    sorted_bed = bed_file + '.sorted'
    try:
        # Sort BED file
        subprocess.run(
            f"sort -k1,1 -k2,2n {bed_file} > {sorted_bed}",
            shell=True, check=True
        )

        # Build command - use BED12+ type if we have scores
        bed_type = 'bed12+1' if with_scores else 'bed12'
        cmd = ['bedToBigBed', f'-type={bed_type}', '-as=' + autosql,
               sorted_bed, chrom_sizes, output_bb]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"bedToBigBed error: {result.stderr}")
            # Fall back to standard BED12 if extended format fails
            if with_scores:
                print("Trying fallback to standard BED12...")
                result = subprocess.run(
                    ['bedToBigBed', '-type=bed12', sorted_bed, chrom_sizes, output_bb],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    print("Fallback succeeded (scores in BED only, not bigBed)")
                    return True
            return False

        return True

    except subprocess.CalledProcessError as e:
        print(f"Error during bigBed conversion: {e}")
        return False
    finally:
        if os.path.exists(sorted_bed):
            os.remove(sorted_bed)


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

    bam = pysam.AlignmentFile(input_bam, "rb")
    count = 0

    with open(output_bed, 'w') as out:
        for read in bam:
            # Skip unmapped and secondary/supplementary (BED should have one row per molecule)
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue

            # Get footprint tags
            try:
                ns = read.get_tag('ns')  # starts (query coords)
                nl = read.get_tag('nl')  # lengths
            except KeyError:
                continue  # No footprint tags

            if len(ns) == 0:
                continue

            # Get scores if requested
            nq = None
            if with_scores:
                try:
                    nq = read.get_tag('nq')
                except KeyError:
                    pass

            # Convert query coords to reference coords
            chrom = read.reference_name
            strand = '-' if read.is_reverse else '+'

            # Get aligned pairs for coordinate mapping
            # This gives us (query_pos, ref_pos) pairs
            aligned_pairs = read.get_aligned_pairs()
            query_to_ref = {}
            for qpos, rpos in aligned_pairs:
                if qpos is not None and rpos is not None:
                    query_to_ref[qpos] = rpos

            # Map footprints to reference coordinates
            ref_starts = []
            ref_ends = []
            valid_scores = []

            for i, (start, length) in enumerate(zip(ns, nl)):
                end = start + length

                # Find reference positions for start and end
                # Handle cases where exact position isn't aligned (indels)
                ref_start = None
                ref_end = None

                # Find closest aligned position for start
                for offset in range(length):
                    if start + offset in query_to_ref:
                        ref_start = query_to_ref[start + offset]
                        break

                # Find closest aligned position for end (going backwards)
                for offset in range(length):
                    if end - 1 - offset in query_to_ref:
                        ref_end = query_to_ref[end - 1 - offset] + 1
                        break

                if ref_start is not None and ref_end is not None and ref_end > ref_start:
                    ref_starts.append(ref_start)
                    ref_ends.append(ref_end)
                    if nq is not None and i < len(nq):
                        valid_scores.append(nq[i])
                    elif with_scores:
                        valid_scores.append(0)

            if not ref_starts:
                continue

            # Build BED12 record
            chrom_start = min(ref_starts)
            chrom_end = max(ref_ends)
            name = read.query_name

            # Block coordinates (relative to chromStart)
            block_sizes = [ref_ends[i] - ref_starts[i] for i in range(len(ref_starts))]
            block_starts = [ref_starts[i] - chrom_start for i in range(len(ref_starts))]

            # Sort blocks by start position (required for BED12)
            if valid_scores:
                sorted_indices = sorted(range(len(block_starts)), key=lambda i: block_starts[i])
                block_starts = [block_starts[i] for i in sorted_indices]
                block_sizes = [block_sizes[i] for i in sorted_indices]
                valid_scores = [valid_scores[i] for i in sorted_indices]
            else:
                sorted_pairs = sorted(zip(block_starts, block_sizes))
                block_starts = [p[0] for p in sorted_pairs]
                block_sizes = [p[1] for p in sorted_pairs]

            # Merge adjacent/overlapping blocks (can happen due to indels in alignment)
            merged_starts = [block_starts[0]]
            merged_sizes = [block_sizes[0]]
            merged_scores = [valid_scores[0]] if valid_scores else []

            for i in range(1, len(block_starts)):
                prev_end = merged_starts[-1] + merged_sizes[-1]
                curr_start = block_starts[i]
                curr_size = block_sizes[i]

                if curr_start <= prev_end:  # Adjacent or overlapping
                    # Extend previous block
                    new_end = max(prev_end, curr_start + curr_size)
                    merged_sizes[-1] = new_end - merged_starts[-1]
                    # Average scores if merging
                    if merged_scores:
                        merged_scores[-1] = (merged_scores[-1] + valid_scores[i]) // 2
                else:
                    # New separate block
                    merged_starts.append(curr_start)
                    merged_sizes.append(curr_size)
                    if valid_scores:
                        merged_scores.append(valid_scores[i])

            block_starts = merged_starts
            block_sizes = merged_sizes
            valid_scores = merged_scores

            # BED12 requires blocks to span chromStart to chromEnd
            # Add 1bp padding blocks at start/end if needed
            read_length = chrom_end - chrom_start

            # Check if first block starts at 0
            if block_starts[0] != 0:
                block_starts.insert(0, 0)
                block_sizes.insert(0, 1)
                if valid_scores:
                    valid_scores.insert(0, 0)

            # Check if last block ends at read_length
            last_end = block_starts[-1] + block_sizes[-1]
            if last_end < read_length:
                block_starts.append(read_length - 1)
                block_sizes.append(1)
                if valid_scores:
                    valid_scores.append(0)

            # BED12 fields
            thick_start = chrom_start
            thick_end = chrom_end
            item_rgb = "0,0,0"
            block_count = len(block_starts)  # Use merged count
            score = block_count  # Number of footprints after merging
            block_sizes_str = ','.join(str(s) for s in block_sizes)
            block_starts_str = ','.join(str(s) for s in block_starts)

            line = f"{chrom}\t{chrom_start}\t{chrom_end}\t{name}\t{score}\t{strand}\t"
            line += f"{thick_start}\t{thick_end}\t{item_rgb}\t{block_count}\t"
            line += f"{block_sizes_str}\t{block_starts_str}"

            if with_scores and valid_scores:
                # Scale BAM scores (0-255) to BED scores (0-1000)
                scaled_scores = [int(s * 1000 / 255) for s in valid_scores]
                scores_str = ','.join(str(s) for s in scaled_scores)
                line += f"\t{scores_str}"

            out.write(line + '\n')
            count += 1

    bam.close()
    return count


def create_scores_database(records: List[dict], db_path: str):
    """
    Create SQLite database with detailed per-footprint scores.

    Schema:
        reads: read_id, chrom, start, end, strand, n_footprints, mean_score
        footprints: read_id, footprint_idx, start, size, score

    This allows fast lookup by read_id for downstream analysis.
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
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

    # Create indices for fast lookup
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_footprints_read_id ON footprints(read_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_reads_chrom_start ON reads(chrom, start)')

    # Insert data
    for record in records:
        # Parse block data
        block_sizes = [int(x) for x in record['blockSizes'].split(',') if x]
        block_starts = [int(x) for x in record['blockStarts'].split(',') if x]

        if 'blockScores' in record:
            block_scores = [int(x) for x in record['blockScores'].split(',') if x]
        else:
            block_scores = [0] * len(block_sizes)

        # Calculate mean score
        mean_score = np.mean(block_scores) if block_scores else 0

        # Insert read record
        cursor.execute('''
            INSERT OR REPLACE INTO reads
            (read_id, chrom, start, end, strand, n_footprints, mean_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            record['name'],
            record['chrom'],
            record['chromStart'],
            record['chromEnd'],
            record['strand'],
            record['blockCount'],
            mean_score
        ))

        # Insert footprint records
        for i, (start, size, score) in enumerate(zip(block_starts, block_sizes, block_scores)):
            cursor.execute('''
                INSERT INTO footprints
                (read_id, footprint_idx, rel_start, size, score)
                VALUES (?, ?, ?, ?, ?)
            ''', (record['name'], i, start, size, score))

    conn.commit()
    conn.close()


def append_to_scores_database(records: List[dict], db_path: str):
    """Append records to existing scores database."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for record in records:
        block_sizes = [int(x) for x in record['blockSizes'].split(',') if x]
        block_starts = [int(x) for x in record['blockStarts'].split(',') if x]

        if 'blockScores' in record:
            block_scores = [int(x) for x in record['blockScores'].split(',') if x]
        else:
            block_scores = [0] * len(block_sizes)

        mean_score = np.mean(block_scores) if block_scores else 0

        cursor.execute('''
            INSERT OR REPLACE INTO reads
            (read_id, chrom, start, end, strand, n_footprints, mean_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            record['name'],
            record['chrom'],
            record['chromStart'],
            record['chromEnd'],
            record['strand'],
            record['blockCount'],
            mean_score
        ))

        for i, (start, size, score) in enumerate(zip(block_starts, block_sizes, block_scores)):
            cursor.execute('''
                INSERT INTO footprints
                (read_id, footprint_idx, rel_start, size, score)
                VALUES (?, ?, ?, ?, ?)
            ''', (record['name'], i, start, size, score))

    conn.commit()
    conn.close()
