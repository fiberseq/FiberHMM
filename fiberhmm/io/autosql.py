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

      - nucleosome: +3  (blockNq, blockEl, blockEr) -- matches MA nuc.QQQ;
                        el/er are 0 for HMM-only/legacy nucs (edges not refined)
      - msp       : +1  (blockAq)
      - m6a / m5c : +1  (blockMl)
      - tf        : +3  (blockTq, blockEl, blockEr) -- matches MA tf.QQQ
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
    'nucleosome': (
        '    int[blockCount] blockNq; '
        '"Per-block nucleosome quality (nq) score, 0-255"\n'
        '    int[blockCount] blockEl; '
        '"Per-block left-edge sharpness (el), 0-255; 255=sharp, 0=ambiguous '
        'or not computed (HMM-only/legacy nucs)"\n'
        '    int[blockCount] blockEr; '
        '"Per-block right-edge sharpness (er), 0-255; 255=sharp, 0=ambiguous '
        'or not computed"\n'
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
    'deam': (
        '    int[blockCount] blockMod; '
        '"0 = R (G->A, GA-strand deamination), 1 = Y (C->T, CT-strand deamination)"\n'
    ),
}

_CIRCULAR_FIELDS = (
    '    string circId; "Shared circular feature ID for clipped pieces, or ."\n'
    '    uint   circPart; "1-based clipped-piece index for circId, or 1"\n'
    '    uint   circParts; "Number of clipped pieces in circId, or 1"\n'
    '    uint   molStart; "0-based molecular start of fused circular feature"\n'
    '    uint   molLength; "Molecular length of fused circular feature"\n'
)

CIRCULAR_FIELD_COUNT = 5


# Number of extra int[blockCount] columns per type when block_scores=True.
EXTRA_FIELD_COUNTS = {t: f.count('int[blockCount]')
                     for t, f in _BLOCK_SCORE_FIELDS.items()}


def _schema_description(description: str, sample_name: Optional[str] = None) -> str:
    sample_name = '' if sample_name is None else str(sample_name).strip()
    if not sample_name:
        return description
    # Prepend a machine-parseable "Sample: <name>." marker. The autoSQL
    # description is a free-form string so we stay format-compatible;
    # bigBedInfo -as surfaces this to downstream tools.
    return f'Sample: {sample_name}. {description}'


def _escape_autosql_description(description: str) -> str:
    return (
        description
        .replace('\\', '\\\\')
        .replace('"', '\\"')
        .replace('\r', ' ')
        .replace('\n', ' ')
    )


def _schema_fields(
    extract_type: Optional[str] = None,
    block_scores: bool = False,
    circular_groups: bool = False,
) -> str:
    fields = _BED12_FIELDS
    if block_scores and extract_type in _BLOCK_SCORE_FIELDS:
        fields = fields + _BLOCK_SCORE_FIELDS[extract_type]
    if circular_groups:
        fields = fields + _CIRCULAR_FIELDS
    return fields


def _make_schema(table_name: str, description: str,
                 extract_type: Optional[str] = None,
                 block_scores: bool = False,
                 sample_name: Optional[str] = None,
                 circular_groups: bool = False) -> str:
    fields = _schema_fields(
        extract_type,
        block_scores,
        circular_groups,
    )
    description = _schema_description(
        description,
        sample_name,
    )
    description = _escape_autosql_description(description)
    return (
        f'table {table_name}\n'
        f'"{description}"\n'
        f'(\n'
        f'{fields}'
        f')\n'
    )


_DESCRIPTIONS = {
    'nucleosome': (
        'FiberHMM nucleosome calls (MA nuc.QQQ, or legacy ns/nl '
        'BAM tags). One BED12 row per read; each block is a called nucleosome. '
        'BED score = mean nq. With --block-scores: per-block quality plus '
        'left/right conservative-edge sharpness (0-255; 0 = unrefined '
        'HMM-only nuc).'
    ),
    'msp': (
        'FiberHMM methylase-sensitive patches (as/al BAM tags). '
        'One BED12 row per read; each block is an accessible patch '
        'between nucleosomes. BED score = mean aq (0-255 when present).'
    ),
    'tf': (
        'FiberHMM TF / Pol II footprint recaller calls (MA/AQ tf.QQQ '
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
    'deam': (
        'FiberHMM DAF-seq deamination calls (R/Y IUPAC codes written into '
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

# Back-compat: the nucleosome type was named 'footprint' through 2.13.x.
_TYPE_ALIASES = {'footprint': 'nucleosome'}


def _canonical_autosql_type(extract_type: str) -> str:
    return _TYPE_ALIASES.get(extract_type, extract_type)


def get_schema(extract_type: str, block_scores: bool = False,
               sample_name: Optional[str] = None,
               circular_groups: bool = False) -> Optional[str]:
    """Return the autoSQL schema string for ``extract_type``, optionally
    with the per-block score columns appended and/or a ``Sample: <name>.``
    marker prepended to the description."""
    extract_type = _canonical_autosql_type(extract_type)
    desc = _DESCRIPTIONS.get(extract_type)
    if desc is None:
        return None
    return _make_schema(
        f'fiberhmm_{extract_type}',
        desc,
        extract_type=extract_type,
        block_scores=block_scores,
        sample_name=sample_name,
        circular_groups=circular_groups,
    )


def _autosql_variant_suffix(block_scores: bool, circular_groups: bool) -> str:
    variant = ''
    if block_scores:
        variant += '.bs'
    if circular_groups:
        variant += '.circ'
    return variant


def _autosql_file_suffix(variant: str) -> str:
    return f'{variant}.as' if variant else '.as'


def _autosql_file_name(extract_type: str, variant: str) -> str:
    return f'fiberhmm_{extract_type}{variant}.as'


def _create_autosql_output_path(extract_type: str, variant: str, suffix: str,
                                out_dir: Optional[str]) -> str:
    if out_dir is None:
        fd, path = tempfile.mkstemp(prefix=f'fiberhmm_{extract_type}_',
                                    suffix=suffix)
        os.close(fd)
        return path

    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(
        out_dir,
        _autosql_file_name(extract_type, variant),
    )


def write_autosql_for(extract_type: str, out_dir: Optional[str] = None,
                      block_scores: bool = False,
                      sample_name: Optional[str] = None,
                      circular_groups: bool = False) -> Optional[str]:
    """Write the autoSQL file for a given extract_type to disk.

    Returns the path to the written file, or None if no schema exists
    for that type (caller should skip ``-as=`` in that case).

    Pass ``sample_name`` to embed a machine-parseable ``Sample: <name>.``
    prefix in the autoSQL description (visible via ``bigBedInfo -as``).
    """
    extract_type = _canonical_autosql_type(extract_type)
    schema = get_schema(extract_type, block_scores=block_scores,
                        sample_name=sample_name,
                        circular_groups=circular_groups)
    if schema is None:
        return None
    variant = _autosql_variant_suffix(block_scores, circular_groups)
    suffix = _autosql_file_suffix(variant)
    path = _create_autosql_output_path(extract_type, variant, suffix, out_dir)
    with open(path, 'w') as f:
        f.write(schema)
    return path
