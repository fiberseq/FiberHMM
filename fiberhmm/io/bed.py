"""BED formatting helpers shared by CLI and inference code."""


def _bed12_block_fields(blocks, chrom_start):
    block_count = len(blocks)
    block_sizes = ','.join(str(end - start) for start, end, *_ in blocks)
    block_starts = ','.join(str(start - chrom_start) for start, *_ in blocks)
    return block_count, block_sizes, block_starts


def _bed12_core_columns(ref_name, chrom_start, chrom_end, read_id, score, strand,
                        blocks, item_rgb='0'):
    block_count, block_sizes, block_starts = _bed12_block_fields(
        blocks, chrom_start,
    )
    return [
        ref_name,
        chrom_start,
        chrom_end,
        read_id,
        score,
        strand,
        chrom_start,
        chrom_end,
        item_rgb,
        block_count,
        block_sizes,
        block_starts,
    ]


def bed12_row(ref_name, chrom_start, chrom_end, read_id, score, strand,
              blocks, extra_columns=(), item_rgb='0'):
    """Format a BED12 or BED12+ row from reference-frame block intervals."""
    columns = _bed12_core_columns(
        ref_name, chrom_start, chrom_end, read_id, score, strand,
        blocks, item_rgb,
    )
    if extra_columns is not None and len(extra_columns) > 0:
        columns.extend(extra_columns)
    return "\t".join(str(column) for column in columns)
