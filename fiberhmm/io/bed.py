"""BED formatting helpers shared by CLI and inference code."""

from dataclasses import dataclass


@dataclass(frozen=True)
class _Bed12BlockFields:
    count: int
    sizes: str
    starts: str


def _bed12_block_fields(blocks, chrom_start):
    return _Bed12BlockFields(
        count=len(blocks),
        sizes=','.join(str(end - start) for start, end, *_ in blocks),
        starts=','.join(str(start - chrom_start) for start, *_ in blocks),
    )


def _bed12_core_columns(ref_name, chrom_start, chrom_end, read_id, score, strand,
                        blocks, item_rgb='0'):
    block_fields = _bed12_block_fields(
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
        block_fields.count,
        block_fields.sizes,
        block_fields.starts,
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
