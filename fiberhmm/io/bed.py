"""BED formatting helpers shared by CLI and inference code."""


def bed12_row(ref_name, chrom_start, chrom_end, read_id, score, strand,
              blocks, extra_columns=(), item_rgb='0'):
    """Format a BED12 or BED12+ row from reference-frame block intervals."""
    block_count = len(blocks)
    block_sizes = ','.join(str(end - start) for start, end, *_ in blocks)
    block_starts = ','.join(str(start - chrom_start) for start, *_ in blocks)
    row = (
        f"{ref_name}\t{chrom_start}\t{chrom_end}\t{read_id}\t{score}\t{strand}\t"
        f"{chrom_start}\t{chrom_end}\t{item_rgb}\t{block_count}\t{block_sizes}\t"
        f"{block_starts}"
    )
    if extra_columns:
        row += "\t" + "\t".join(str(column) for column in extra_columns)
    return row
