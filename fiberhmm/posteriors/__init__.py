"""HMM posterior probability I/O (HDF5 and TSV backends)."""

from fiberhmm.posteriors.tsv_backend import PosteriorsTSVWriter

try:
    from fiberhmm.posteriors.hdf5_backend import PosteriorWriter
except ImportError:
    PosteriorWriter = None
