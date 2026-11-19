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

The default schema is BED12. Optional fields are appended in this order:
per-block scores, circular grouping, then haplotype fields. The opt-in
``haplotype_fields=True`` variant adds scalar ``hp`` and ``ps`` columns copied
from the source BAM record; ``-1`` means the corresponding tag was absent or
not an integer.

Two base schema flavors:

  - **BED12 + 1** (default): the standard 12 columns plus ``isDuplicate``.
    Mean quality is flattened into column 5 (``score``); the final flag
    preserves BAM 0x400 for browsers and other downstream consumers.
  - **BED12 + N** (``block_scores=True``): adds one or more per-block
    ``int[blockCount]`` arrays after ``chromStarts``. Carries the
    per-feature quality that would otherwise be lost to the read-level
    mean. Comma-separated, same shape as ``blockSizes`` / ``chromStarts``
    so tools that already parse those two arrays only need to split one
    more string. Column count per type:

      - nucleosome: +3  (blockNq, blockEl, blockEr) -- matches MA nuc.QQQ;
                        el/er are 0 for HMM-only/legacy nucs (edges not refined)
      - msp       : +1  (blockAq)
      - m6a / m5c : +1  (blockMl; DddA ddda_mcg interval spans carry 0)
      - tf        : +3  (blockTq, blockEl, blockEr) -- matches MA tf.QQQ

The opt-in read layers ``var`` (``--variants``) and ``sv`` (``--sv``) always
carry their own columns right after ``chromStarts`` (``_LAYER_FIELDS``):
``var`` blockAllele, blockBq; ``sv`` blockType, blockLen, mapq, readLen,
flag, sa. Haplotype fields and ``isDuplicate`` follow as for every layer.
"""
from __future__ import annotations

import os
import re
import tempfile
from typing import Optional


def sample_token(sample_name: str) -> str:
    """Collapse a sample name into a single dot-free, whitespace-free token.

    The ``Sample: <name>.`` autoSQL marker is parsed by downstream viewers
    (FiberBrowser ``bigbed_detection._SAMPLE_TAG_RE``) to group a sample's
    layers (footprint/msp/tf/...) into one track. Older viewers capture the
    tag only up to the *first* ``.`` or space, so a dotted name like
    ``BioSample70.decorated.reads.MA.filtered_T`` collapsed to
    ``BioSample70`` -- identical to its ``..._GA`` sibling pool -- and the
    two pools' layers overwrote each other. Replacing ``.`` and whitespace
    runs with ``_`` keeps the full distinguishing name intact under every
    viewer. Other identifier-safe characters (``-``, digits) are preserved.
    """
    token = re.sub(r'[\s.]+', '_', sample_name.strip()).strip('_')
    return token or sample_name

def _note_token(value: str) -> str:
    """A value safe inside the quoted autoSQL description and the
    ``;``-separated VCF marker."""
    return re.sub(r'[\s";]+', '_', str(value)).strip('_') or '.'


def vcf_source_note(vcf_path: str, md5: str, sample: Optional[str]) -> str:
    """``VCF: name=<basename>; md5=<hex>; sample=<name|*>.``: the ``var``
    layer's source marker (``*`` = every site, no sample restriction).
    Viewers parse ``name=``/``md5=``/``sample=`` up to ``;`` / the final
    ``.``."""
    return (f'VCF: name={_note_token(os.path.basename(vcf_path))}; '
            f'md5={md5}; sample={_note_token(sample) if sample else "*"}.')


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
        '"Native per-position ML probability, 0-255; 0 for DddA-inferred '
        'ddda_mcg interval spans"\n'
    ),
    'fp': (
        '    int[blockCount] blockTq; '
        '"Per-block tq (LLR x 10) of the FiberHMM call that footprints the motif, 0-255"\n'
        '    int[blockCount] blockCw; '
        '"Per-block conservative width of that call, bp (saturated 255)"\n'
        '    int[blockCount] blockEf; '
        '"Per-block edge flag: 1 = conservative call bounds contain the motif, 2 = only the outer bounds do"\n'
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

_HAPLOTYPE_FIELDS = (
    '    int hp; "Source BAM HP tag (haplotype); -1 if absent or not an integer"\n'
    '    int ps; "Source BAM PS tag (phase set); -1 if absent or not an integer"\n'
)

HAPLOTYPE_FIELD_COUNT = 2

_DUPLICATE_FIELD = (
    '    uint isDuplicate; "1 when the source BAM record carried flag 0x400; 0 otherwise"\n'
)
DUPLICATE_FIELD_COUNT = 1


# Always-on columns of the opt-in read layers (written whatever
# --block-scores says), between the BED12 fields and the circular/haplotype
# suffix. Per-block arrays come first so circular folding and graph lifts,
# which carry ``int[blockCount]`` columns 13.. with their blocks, keep them
# aligned.
_LAYER_FIELDS = {
    'var': (
        '    int[blockCount] blockAllele; '
        '"Allele the read carries at each VCF site: k >= 0 VCF allele index '
        '(0 REF, 1.. ALT); -1 other sequence; -2 site deleted in the read '
        '(spanning deletion); -3 ambiguous because of DAF conversion"\n'
        '    int[blockCount] blockBq; '
        '"Base quality at the site (minimum over the compared bases of an '
        'indel or MNV); -1 if none"\n'
    ),
    'sv': (
        '    int[blockCount] blockType; '
        '"Event type: 0 DEL (CIGAR D), 1 INS (CIGAR I, block = anchor base '
        'before it), 2 CLIP_LEFT, 3 CLIP_RIGHT (soft+hard clip at the '
        'reference-left / reference-right end, block = first / last aligned '
        'base)"\n'
        '    int[blockCount] blockLen; '
        '"Event length: deleted bp, inserted bases or clipped bases"\n'
        '    uint mapq; "Mapping quality of the alignment record"\n'
        '    uint readLen; "Full read length including soft and hard clips"\n'
        '    uint flag; "SAM flag of the alignment record (0x800 = supplementary)"\n'
        '    lstring sa; "SA tag verbatim (. if none; entries past 4000 '
        'characters are dropped and the value ends in ...)"\n'
    ),
}

# Total always-on columns and how many of them are int[blockCount] arrays.
LAYER_FIELD_COUNTS = {t: len(f.strip('\n').splitlines())
                      for t, f in _LAYER_FIELDS.items()}
LAYER_BLOCK_COLUMN_COUNTS = {t: f.count('int[blockCount]')
                             for t, f in _LAYER_FIELDS.items()}


# Number of extra int[blockCount] columns per type when block_scores=True.
EXTRA_FIELD_COUNTS = {t: f.count('int[blockCount]')
                     for t, f in _BLOCK_SCORE_FIELDS.items()}


def _make_schema(table_name: str, description: str,
                 extract_type: Optional[str] = None,
                 block_scores: bool = False,
                 sample_name: Optional[str] = None,
                 circular_groups: bool = False,
                 haplotype_fields: bool = False,
                 source_note: Optional[str] = None) -> str:
    fields = _BED12_FIELDS
    if extract_type in _LAYER_FIELDS:
        fields = fields + _LAYER_FIELDS[extract_type]
    if block_scores and extract_type in _BLOCK_SCORE_FIELDS:
        fields = fields + _BLOCK_SCORE_FIELDS[extract_type]
    if circular_groups:
        fields = fields + _CIRCULAR_FIELDS
    if haplotype_fields:
        fields = fields + _HAPLOTYPE_FIELDS
    fields = fields + _DUPLICATE_FIELD
    if source_note:
        description = f'{source_note} {description}'
    if sample_name:
        # Prepend a machine-parseable "Sample: <name>." marker. The autoSQL
        # description is a free-form string so we stay format-compatible;
        # bigBedInfo -as surfaces this to downstream tools. The name is
        # collapsed to a dot/space-free token (see sample_token) so viewers
        # that split the tag on the first '.' still recover the full,
        # pool-distinguishing identifier.
        description = f'Sample: {sample_token(sample_name)}. {description}'
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
        'FiberHMM 5mC annotations. For DddA data, each block is a conservative '
        'molecule-specific methylated-CpG interval from an MA ddda_mcg. group; '
        'absence means no confident call, not unmethylated. For native 5mC '
        'assays, each block is a 1 bp MM/ML modification call passing '
        '--prob-threshold. DAF spans have BED score 0 and no quality byte.'
    ),
    'deam': (
        'FiberHMM DAF-seq deamination calls (R/Y IUPAC codes written into '
        'the query sequence by fiberhmm-daf-encode). Each block is a 1 bp '
        'deaminated base (Y = C->T on the CT strand, R = G->A on the GA '
        'strand). BED score = 255 (deamination calls are deterministic, '
        'not probabilistic). Use --block-scores to carry the R/Y '
        'disambiguation in a blockMod per-block column.'
    ),
    'fp': (
        'FiberHMM motif footprints (fiberhmm-footprint --out-bb). One BED12 row per read and motif set; '
        'each block is a motif instance footprinted on that read: an MSP spans it and one FiberHMM TF '
        'call contains it. BED score = mean tq of those calls.'
    ),
    'bothstrand': (
        'FiberHMM paired-duplex DAF coverage. Each BED12 block marks a span '
        'where both physical strands contributed deamination evidence to a '
        'joint consensus molecule. Ordinary single-strand reads contribute '
        'no row.'
    ),
    'var': (
        'FiberHMM per-read alleles at VCF sites (fiberhmm-extract --variants). '
        'One BED12 row per read (more when sites overlap); each block is a '
        'site the read covers (size = REF length); blockAllele gives the '
        'allele the read carries. Pileup-style genotyping, no realignment. '
        'BED score = mean base quality of the sites.'
    ),
    'sv': (
        'FiberHMM per-read structural events (fiberhmm-extract --sv): CIGAR '
        'deletions/insertions and long clips, the signatures long-read SV '
        'callers start from, plus the SA tag. One BED12 row per alignment '
        'record (supplementary records included; more rows when events '
        'overlap); BED score = MAPQ.'
    ),
}


AUTOSQL_SCHEMAS = {
    t: _make_schema(f'fiberhmm_{t}', desc, extract_type=t, block_scores=False)
    for t, desc in _DESCRIPTIONS.items()
}

# Back-compat: the nucleosome type was named 'footprint' through 2.13.x.
_TYPE_ALIASES = {'footprint': 'nucleosome'}


def get_schema(extract_type: str, block_scores: bool = False,
               sample_name: Optional[str] = None,
               circular_groups: bool = False,
               haplotype_fields: bool = False,
               source_note: Optional[str] = None) -> Optional[str]:
    """Return the autoSQL schema string for ``extract_type``, optionally
    with the per-block score columns appended and/or a ``Sample: <name>.``
    marker prepended to the description. ``source_note`` (e.g. the ``var``
    layer's ``VCF: ...`` marker, :func:`vcf_source_note`) goes between the
    sample marker and the description."""
    extract_type = _TYPE_ALIASES.get(extract_type, extract_type)
    desc = _DESCRIPTIONS.get(extract_type)
    if desc is None:
        return None
    return _make_schema(f'fiberhmm_{extract_type}', desc,
                        extract_type=extract_type,
                        block_scores=block_scores,
                        sample_name=sample_name,
                        circular_groups=circular_groups,
                        haplotype_fields=haplotype_fields,
                        source_note=source_note)


def write_autosql_for(extract_type: str, out_dir: Optional[str] = None,
                      block_scores: bool = False,
                      sample_name: Optional[str] = None,
                      circular_groups: bool = False,
                      haplotype_fields: bool = False,
                      source_note: Optional[str] = None) -> Optional[str]:
    """Write the autoSQL file for a given extract_type to disk.

    Returns the path to the written file, or None if no schema exists
    for that type (caller should skip ``-as=`` in that case).

    Pass ``sample_name`` to embed a machine-parseable ``Sample: <name>.``
    prefix in the autoSQL description (visible via ``bigBedInfo -as``).
    """
    extract_type = _TYPE_ALIASES.get(extract_type, extract_type)
    schema = get_schema(extract_type, block_scores=block_scores,
                        sample_name=sample_name,
                        circular_groups=circular_groups,
                        haplotype_fields=haplotype_fields,
                        source_note=source_note)
    if schema is None:
        return None
    variant = ''
    if block_scores:
        variant += '.bs'
    if circular_groups:
        variant += '.circ'
    if haplotype_fields:
        variant += '.hap'
    suffix = f'{variant}.as' if variant else '.as'
    if out_dir is None:
        fd, path = tempfile.mkstemp(prefix=f'fiberhmm_{extract_type}_',
                                     suffix=suffix)
        os.close(fd)
    else:
        os.makedirs(out_dir, exist_ok=True)
        fname = f'fiberhmm_{extract_type}{variant}.as'
        path = os.path.join(out_dir, fname)
    with open(path, 'w') as f:
        f.write(schema)
    return path
