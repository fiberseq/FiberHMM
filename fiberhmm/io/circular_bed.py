"""Fold BED12 rows that run past the end of a circular contig.

On a contig declared ``@SQ TP:circular`` an alignment may extend past the
contig length (SAM specification 1.4, "Circular reference sequences"; this is
how ``fiberhmm-pipeline`` stores reads that run through a plasmid's origin).
Features extracted from such a read then have reference coordinates
``>= length``, which BED and bigBed cannot hold. :func:`fold_bed12_row` splits
a row at the origin: blocks before the end stay in the first row, blocks past
it move (shifted by ``-length``) to a second row with the same name, and a
block that crosses the origin is cut into one piece in each row. Per-block
columns (``int[blockCount]`` fields after ``chromStarts``) follow their
blocks; the other columns are copied.
"""
from __future__ import annotations

import heapq
import os
from typing import Iterable, Optional

import pysam


def circular_contig_sizes(bam_path: str) -> dict[str, int]:
    """``{name: length}`` of the ``@SQ TP:circular`` contigs of a BAM."""
    with pysam.AlignmentFile(bam_path, check_sq=False) as bam:
        return {sq["SN"]: int(sq["LN"]) for sq in bam.header.to_dict().get("SQ", [])
                if str(sq.get("TP", "")).lower() == "circular"}


def _row(fields: list[str], start: int, blocks: list[tuple[int, int, list[str]]],
         n_block_columns: int) -> str:
    end = max(e for _, e, _ in blocks)
    out = list(fields)
    out[1] = str(start)
    out[2] = str(end)
    out[6] = str(start)
    out[7] = str(end)
    out[9] = str(len(blocks))
    out[10] = ",".join(str(e - s) for s, e, _ in blocks)
    out[11] = ",".join(str(s - start) for s, _, _ in blocks)
    for i in range(n_block_columns):
        out[12 + i] = ",".join(extra[i] for _, _, extra in blocks)
    return "\t".join(out)


def fold_bed12_row(line: str, length: int, n_block_columns: int = 0) -> list[str]:
    """Return the row(s) for one BED12 line on a circular contig of ``length``."""
    fields = line.rstrip("\n").split("\t")
    start = int(fields[1])
    end = int(fields[2])
    if end <= length and start < length:
        return [line.rstrip("\n")]
    sizes = [int(v) for v in fields[10].rstrip(",").split(",") if v != ""]
    offsets = [int(v) for v in fields[11].rstrip(",").split(",") if v != ""]
    per_block = [
        [v for v in fields[12 + i].rstrip(",").split(",")] if 12 + i < len(fields) else []
        for i in range(n_block_columns)
    ]
    head: list[tuple[int, int, list[str]]] = []
    tail: list[tuple[int, int, list[str]]] = []
    for index, (size, offset) in enumerate(zip(sizes, offsets)):
        extra = [column[index] if index < len(column) else "0" for column in per_block]
        s = start + offset
        e = s + size
        # A read covers at most one circle; reduce anything beyond it too.
        cycle = (s // length) * length
        s -= cycle
        e -= cycle
        if e <= length:
            (tail if cycle else head).append((s, e, extra))
        else:
            (tail if cycle else head).append((s, length, extra))
            tail.append((0, min(e - length, length), extra))
    rows: list[str] = []
    if head:
        rows.append(_row(fields, min(s for s, _, _ in head), sorted(head), n_block_columns))
    if tail:
        tail = sorted(tail)
        rows.append(_row(fields, tail[0][0], tail, n_block_columns))
    return rows


def fold_circular_bed(path: str, circular_sizes: dict[str, int],
                      n_block_columns: int = 0) -> int:
    """Fold, in place, every row of a sorted BED12 file that runs past the end
    of a circular contig; return the number of rows folded.

    Rows that stay in place keep the file's order; the new rows (which start
    near the origin) are merged back in by (contig order of the file, start).
    """
    if not circular_sizes or not os.path.exists(path):
        return 0
    moved: list[tuple[int, int, str]] = []
    rank: dict[str, int] = {}
    folded = 0
    tmp = f"{path}.fold.tmp{os.getpid()}"
    with open(path, encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as out:
        for line in src:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                out.write(line)
                continue
            chrom, start_text, end_text = line.split("\t", 3)[:3]
            rank.setdefault(chrom, len(rank))
            length = circular_sizes.get(chrom)
            if length is None or (int(end_text) <= length and int(start_text) < length):
                out.write(line if line.endswith("\n") else line + "\n")
                continue
            folded += 1
            rows = fold_bed12_row(line, length, n_block_columns)
            for row in rows:
                row_start = int(row.split("\t", 2)[1])
                if row_start >= int(start_text):
                    out.write(row + "\n")  # keeps the file's order
                else:
                    moved.append((rank[chrom], row_start, row))
    if not folded:
        os.remove(tmp)
        return 0
    moved.sort(key=lambda item: (item[0], item[1]))

    def _keyed(lines: Iterable[str]):
        for line in lines:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                yield (-1, -1, line)
                continue
            chrom, start_text = line.split("\t", 2)[:2]
            yield (rank.setdefault(chrom, len(rank)), int(start_text), line)

    final = f"{path}.fold2.tmp{os.getpid()}"
    with open(tmp, encoding="utf-8") as sorted_rows, open(final, "w", encoding="utf-8") as out:
        merged = heapq.merge(_keyed(sorted_rows),
                             ((r, s, row + "\n") for r, s, row in moved),
                             key=lambda item: (item[0], item[1]))
        for _, _, line in merged:
            out.write(line)
    os.remove(tmp)
    os.replace(final, path)
    return folded


def block_column_count(extract_type: Optional[str], block_scores: bool) -> int:
    if not block_scores or not extract_type:
        return 0
    from fiberhmm.io.autosql import EXTRA_FIELD_COUNTS
    return EXTRA_FIELD_COUNTS.get(extract_type, 0)
