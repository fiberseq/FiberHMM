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

Two schema flavors:

  - **BED12** (default): the standard 12 columns. Mean quality flattened
    into column 5 (``score``).
  - **BED12 + N** (``block_scores=True``): adds one or more per-block
    ``int[blockCount]`` arrays after ``chromStarts``. Carries the
    per-feature quality that would otherwise be lost to the read-level
    mean. Comma-separated, same shape as ``blockSizes`` / ``chromStarts``
    so tools that already parse those two arrays only need to split one
    more string. Column count per type:

      - footprint : +1  (blockNq)
      - msp       : +1  (blockAq)
      - m6a / m5c : +1  (blockMl)
      - tf        : +3  (blockTq, blockEl, blockEr) -- matches MA tf+QQQ
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


# Per-type block-score field definitions. Each key appends N int[blockCount]
# columns after chromStarts when block_scores=True.
_BLOCK_SCORE_FIELDS = {
    'footprint': (
        '    int[blockCount] blockNq; '
        '"Per-block nucleosome quality (nq) score, 0-255"\n'
    ),
    'msp': (
        '    int[blockCount] blockAq; '
        '"Per-block MSP quality (aq) score, 0-255"\n'
    ),
    'tf': (
        '    int[blockCount] blockTq; '
        '"Per-block TF quality (tq = round(LLR*10)), 0-255"\n'
        '    int[blockCount] blockEl; '
        '"Per-block left-edge ambiguity quality (el), 0-255"\n'
        '    int[blockCount] blockEr; '
        '"Per-block right-edge ambiguity quality (er), 0-255"\n'
    ),
    'm6a': (
        '    int[blockCount] blockMl; '
        '"Per-position ML (modified-base probability), 0-255"\n'
    ),
    'm5c': (
        '    int[blockCount] blockMl; '
        '"Per-position ML (modified-base probability), 0-255"\n'
    ),
    'ry': (
        '    int[blockCount] blockMod; '
        '"0 = R (G->A, GA-strand deamination), 1 = Y (C->T, CT-strand deamination)"\n'
    ),
}


# Number of extra int[blockCount] columns per type when block_scores=True.
EXTRA_FIELD_COUNTS = {t: f.count('int[blockCount]')
                     for t, f in _BLOCK_SCORE_FIELDS.items()}


def _make_schema(table_name: str, description: str,
                 extract_type: Optional[str] = None,
                 block_scores: bool = False) -> str:
    fields = _BED12_FIELDS
    if block_scores and extract_type in _BLOCK_SCORE_FIELDS:
        fields = fields + _BLOCK_SCORE_FIELDS[extract_type]
    return (
        f'table {table_name}\n'
        f'"{description}"\n'
        f'(\n'
        f'{fields}'
        f')\n'
    )


_DESCRIPTIONS = {
    'footprint': (
        'FiberHMM nucleosome footprint calls (ns/nl BAM tags). '
        'One BED12 row per read; each block is a called nucleosome. '
        'BED score = mean nq (HMM posterior confidence, 0-255).'
    ),
    'msp': (
        'FiberHMM methylase-sensitive patches (as/al BAM tags). '
        'One BED12 row per read; each block is an accessible patch '
        'between nucleosomes. BED score = mean aq (0-255 when present).'
    ),
    'tf': (
        'FiberHMM TF / Pol II footprint recaller calls (MA/AQ tf+QQQ '
        'annotations per the fiberseq Molecular-annotation spec). '
        'Second-pass LLR-based recaller output. One BED12 row per read; '
        'each block is a TF footprint. BED score = mean tq (LLR-derived '
        'confidence; every 23 points = 1 order of magnitude in '
        'likelihood ratio; tq=50 -> LR ~148:1, tq=100 -> LR ~22,000:1).'
    ),
    'm6a': (
        'FiberHMM per-position m6A modification calls (from MM/ML tags). '
        'One BED12 row per read; each block is a 1 bp modified position '
        'passing --prob-threshold. BED score = mean per-position ML '
        'probability above the threshold.'
    ),
    'm5c': (
        'FiberHMM per-position 5mC / DAF-seq deamination calls (from '
        'MM/ML tags). One BED12 row per read; each block is a 1 bp '
        'modified/deaminated position passing --prob-threshold.'
    ),
    'ry': (
        'FiberHMM DAF-seq IUPAC deamination calls (R/Y codes written into '
        'the query sequence by fiberhmm-daf-encode). Each block is a 1 bp '
        'deaminated base (Y = C->T on the CT strand, R = G->A on the GA '
        'strand). BED score = 255 (deamination calls are deterministic, '
        'not probabilistic). Use --block-scores to carry the R/Y '
        'disambiguation in a blockMod per-block column.'
    ),
}


AUTOSQL_SCHEMAS = {
    t: _make_schema(f'fiberhmm_{t}', desc, extract_type=t, block_scores=False)
    for t, desc in _DESCRIPTIONS.items()
}


def get_schema(extract_type: str, block_scores: bool = False) -> Optional[str]:
    """Return the autoSQL schema string for ``extract_type``, optionally
    with the per-block score columns appended."""
    desc = _DESCRIPTIONS.get(extract_type)
    if desc is None:
        return None
    return _make_schema(f'fiberhmm_{extract_type}', desc,
                        extract_type=extract_type, block_scores=block_scores)


def write_autosql_for(extract_type: str, out_dir: Optional[str] = None,
                      block_scores: bool = False) -> Optional[str]:
    """Write the autoSQL file for a given extract_type to disk.

    Returns the path to the written file, or None if no schema exists
    for that type (caller should skip ``-as=`` in that case).
    """
    schema = get_schema(extract_type, block_scores=block_scores)
    if schema is None:
        return None
    suffix = '.bs.as' if block_scores else '.as'
    if out_dir is None:
        fd, path = tempfile.mkstemp(prefix=f'fiberhmm_{extract_type}_',
                                     suffix=suffix)
        os.close(fd)
    else:
        os.makedirs(out_dir, exist_ok=True)
        fname = f'fiberhmm_{extract_type}{".bs" if block_scores else ""}.as'
        path = os.path.join(out_dir, fname)
    with open(path, 'w') as f:
        f.write(schema)
    return path
