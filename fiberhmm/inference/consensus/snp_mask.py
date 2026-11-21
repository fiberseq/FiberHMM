"""Explicit SNP masks for consensus BAM preparation; never modify saved evidence."""
from pathlib import Path
import hashlib


def read_mask(path):
    """Read and fingerprint the same bytes; BED intervals are 0-based half-open."""
    if not path:
        return {}, None
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    mask = {}
    for number, line in enumerate(raw.decode('utf-8').splitlines(), 1):
        if not line.strip() or line.startswith(('#', 'track ', 'browser ')):
            continue
        fields = line.split()
        try:
            chrom, start, end = fields[:3]
            start, end = int(start), int(end)
            if start < 0 or end <= start:
                raise ValueError
        except ValueError:
            raise ValueError(f'{path}:{number}: invalid SNP mask BED interval (expected CHROM START END, 0 <= START < END)') from None
        mask.setdefault(chrom, set()).update(range(start, end))
    mask = {chrom: frozenset(sites) for chrom, sites in mask.items()}
    return mask, dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                      sites=sum(map(len, mask.values())), contigs=len(mask),
                      policy='exclude_observations')


def prepare_mask_option(values, path=None, *, replay=False):
    """Normalize and validate a caller's explicit preparation options."""
    if path is not None:
        values.setdefault('input', {})['daf_snp_mask'] = path
    path = values.get('input', {}).get('daf_snp_mask')
    if path is not None and not isinstance(path, str):
        raise ValueError('input.daf_snp_mask must be a BED path string')
    if not path:
        return
    if replay:
        raise ValueError('DAF SNP masking requires BAM input; --evidence/--resume already fixes the observations. Rebuild evidence from BAM with --daf-snp-mask')
    _, receipt = read_mask(path)
    values['input']['daf_snp_mask'] = receipt['path']


def masked_query_positions(read, sites):
    """Query positions aligned to reference ``sites``, walking the CIGAR.

    pysam's ``get_aligned_pairs()`` advances the query index on ``P``
    (padding) operations, shifting every later position by one; padding
    consumes neither query nor reference here.
    """
    if not sites:
        return set()
    cigar = getattr(read, 'cigartuples', None)
    if not cigar:
        return set()
    q, r, out = 0, int(read.reference_start), set()
    for op, length in cigar:
        if op in (0, 7, 8):        # M, =, X
            out.update(q + i for i in range(length) if r + i in sites)
            q += length; r += length
        elif op in (1, 4):         # I, S
            q += length
        elif op in (2, 3):         # D, N
            r += length
        # H (5) and P (6) consume neither.
    return out
