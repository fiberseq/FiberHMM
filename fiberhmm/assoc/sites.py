"""Reference site density for M2 matching and M4 relocation strata."""
from contextlib import contextmanager
from pathlib import Path
import numpy as np


@contextmanager
def reference_sites(scan, path, bases='AT'):
    import pysam
    from .scanio import ScanInputError
    bases = set(bases.upper())
    if not bases or bases - set('ACGT'):
        raise ScanInputError('site bases must be a nonempty subset of ACGT')
    if not Path(str(path) + '.fai').is_file():
        raise ScanInputError('reference FASTA requires an existing .fai index; create it before association')
    if str(path).endswith(('.gz', '.bgz')) and not Path(str(path) + '.gzi').is_file():
        raise ScanInputError('compressed FASTA requires an existing .gzi index')
    with pysam.FastaFile(str(path)) as fasta:
        lengths = dict(zip(fasta.references, fasta.lengths))
        def density(chrom, start, end):
            expected = dict(scan.manifest['reference']['contigs']).get(chrom)
            if lengths.get(chrom) != expected:
                raise ScanInputError(f'FASTA contig {chrom} is missing or has a different length from the scan')
            seq = fasta.fetch(chrom, max(0, int(start)), min(expected, int(end))).upper()
            measured = sum(seq.count(b) for b in 'ACGT')
            return sum(seq.count(b) for b in bases) / measured if measured else float('nan')
        yield density


def locus_density(scan, path, chrom, positions, *, bases='AT', window=1000):
    with reference_sites(scan, path, bases) as density:
        return np.array([density(c, int(p) - window // 2, int(p) + (window + 1) // 2)
                         for c, p in zip(chrom, positions)])


def density_strata(scan, path, territory, *, bases='AT', bin_bp=10000):
    from .strata import quantile_bins
    windows, values = [], []
    with reference_sites(scan, path, bases) as density:
        for c, a, b in territory:
            for s in range(a, b, bin_bp):
                e = min(b, s + bin_bp)
                windows.append((c, s, e))
                values.append(density(c, s, e))
    bins = quantile_bins(np.array(values), 3)
    return [(c, s, e, f'{c}:site{int(k)}') for (c, s, e), k in zip(windows, bins)]
