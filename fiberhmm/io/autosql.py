"""autoSQL schemas for FiberHMM bigBed outputs.

UCSC bigBed files can embed an autoSQL table definition
(https://genome.ucsc.edu/goldenPath/help/bigBed.html#Ex3) that carries:

  - a machine-readable table name (e.g. ``fiberhmm_tf``) -> the browser
    uses this as the track-type identifier
  - a free-form description string -> shown as the track subtitle
  - per-column documentation

We ship one autoSQL per extract type so any bigBed viewer (UCSC, IGV,
FiberBrowser) opens a FiberHMM file and immediately knows whether it's
a nucleosome / MSP / TF / m6A / m5C track and what each column means.

All schemas are BED12 (the standard 12 columns) -- no extra columns, so
downstream tools that expect plain BED12 still work. The schema metadata
is informational; browsers render BED12 identically with or without it.
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional


_BED12_FIELDS = """    string  chrom;       "Reference chromosome / contig"
    uint    chromStart;  "Start position on the reference (0-based)"
    uint    chromEnd;    "End position on the reference (exclusive)"
    string  name;        "Read ID of the source fiber-seq / DAF-seq read"
    uint    score;       "Mean per-read quality, 0-255 (see description)"
    char[1] strand;      "Read alignment strand, + or -"
    uint    thickStart;  "Same as chromStart (no thick region distinction)"
    uint    thickEnd;    "Same as chromEnd"
    uint    reserved;    "Item color (unused; 0)"
    int     blockCount;  "Number of feature blocks in this read"
    int[blockCount] blockSizes;  "Size of each feature block"
    int[blockCount] chromStarts; "Offset of each block from chromStart"
"""


def _make_schema(table_name: str, description: str) -> str:
    return (
        f'table {table_name}\n'
        f'"{description}"\n'
        f'(\n'
        f'{_BED12_FIELDS}'
        f')\n'
    )


# One schema per extract_type. Keys match fiberhmm-extract's --{type} flags.
AUTOSQL_SCHEMAS = {
    'footprint': _make_schema(
        'fiberhmm_footprint',
        'FiberHMM nucleosome footprint calls (ns/nl BAM tags). '
        'One BED12 row per read; each block is a called nucleosome. '
        'BED score = mean nq (HMM posterior confidence, 0-255).'
    ),
    'msp': _make_schema(
        'fiberhmm_msp',
        'FiberHMM methylase-sensitive patches (as/al BAM tags). '
        'One BED12 row per read; each block is an accessible patch '
        'between nucleosomes. BED score = mean aq (0-255 when present).'
    ),
    'tf': _make_schema(
        'fiberhmm_tf',
        'FiberHMM TF / Pol II footprint recaller calls (MA/AQ tf+QQQ '
        'annotations per the fiberseq Molecular-annotation spec). '
        'Second-pass LLR-based recaller output. One BED12 row per read; '
        'each block is a TF footprint. BED score = mean tq (LLR-derived '
        'confidence; every 23 points = 1 order of magnitude in '
        'likelihood ratio; tq=50 -> LR ~148:1, tq=100 -> LR ~22,000:1).'
    ),
    'm6a': _make_schema(
        'fiberhmm_m6a',
        'FiberHMM per-position m6A modification calls (from MM/ML tags). '
        'One BED12 row per read; each block is a 1 bp modified position '
        'passing --prob-threshold. BED score = mean per-position ML '
        'probability above the threshold.'
    ),
    'm5c': _make_schema(
        'fiberhmm_m5c',
        'FiberHMM per-position 5mC / DAF-seq deamination calls (from '
        'MM/ML tags). One BED12 row per read; each block is a 1 bp '
        'modified/deaminated position passing --prob-threshold.'
    ),
}


def write_autosql_for(extract_type: str, out_dir: Optional[str] = None) -> Optional[str]:
    """Write the autoSQL file for a given extract_type to disk.

    Returns the path to the written file, or None if no schema exists
    for that type (caller should skip ``-as=`` in that case).
    """
    schema = AUTOSQL_SCHEMAS.get(extract_type)
    if schema is None:
        return None
    if out_dir is None:
        fd, path = tempfile.mkstemp(prefix=f'fiberhmm_{extract_type}_',
                                     suffix='.as')
        os.close(fd)
    else:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f'fiberhmm_{extract_type}.as')
    with open(path, 'w') as f:
        f.write(schema)
    return path
