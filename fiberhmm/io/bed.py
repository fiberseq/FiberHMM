"""BED formatting helpers shared by CLI and inference code."""

from dataclasses import dataclass


@dataclass(frozen=True)
class _Bed12BlockFields:
    count: int
    sizes: str
    starts: str


@dataclass(frozen=True)
class _Bed12Record:
    ref_name: str
    chrom_start: int
    chrom_end: int
    read_id: str
    score: int
    strand: str
    blocks: object
    item_rgb: str = '0'


def _bed12_block_fields(blocks, chrom_start):
    return _Bed12BlockFields(
        count=len(blocks),
        sizes=','.join(str(end - start) for start, end, *_ in blocks),
        starts=','.join(str(start - chrom_start) for start, *_ in blocks),
    )


def _bed12_core_record(
    ref_name,
    chrom_start,
    chrom_end,
    read_id,
    score,
    strand,
    blocks,
    item_rgb='0',
) -> _Bed12Record:
    return _Bed12Record(
        ref_name=ref_name,
        chrom_start=chrom_start,
        chrom_end=chrom_end,
        read_id=read_id,
        score=score,
        strand=strand,
        blocks=blocks,
        item_rgb=item_rgb,
    )


def _bed12_core_columns_from_record(record: _Bed12Record):
    block_fields = _bed12_block_fields(
        record.blocks, record.chrom_start,
    )
    return [
        record.ref_name,
        record.chrom_start,
        record.chrom_end,
        record.read_id,
        record.score,
        record.strand,
        record.chrom_start,
        record.chrom_end,
        record.item_rgb,
        block_fields.count,
        block_fields.sizes,
        block_fields.starts,
    ]


def _bed12_core_columns(ref_name, chrom_start, chrom_end, read_id, score, strand,
                        blocks, item_rgb='0'):
    return _bed12_core_columns_from_record(
        _bed12_core_record(
            ref_name,
            chrom_start,
            chrom_end,
            read_id,
            score,
            strand,
            blocks,
            item_rgb,
        )
    )


def _bed12_row_from_record(record: _Bed12Record, extra_columns=()):
    columns = _bed12_core_columns_from_record(record)
    if extra_columns is not None and len(extra_columns) > 0:
        columns.extend(extra_columns)
    return "\t".join(str(column) for column in columns)


def bed12_row(ref_name, chrom_start, chrom_end, read_id, score, strand,
              blocks, extra_columns=(), item_rgb='0'):
    """Format a BED12 or BED12+ row from reference-frame block intervals."""
    return _bed12_row_from_record(
        _bed12_core_record(
            ref_name,
            chrom_start,
            chrom_end,
            read_id,
            score,
            strand,
            blocks,
            item_rgb,
        ),
        extra_columns,
    )
