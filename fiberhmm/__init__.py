"""
FiberHMM - Hidden Markov Model toolkit for chromatin footprint calling
from fiber-seq and DAF-seq single-molecule data.
"""

__version__ = "2.0.0"

from fiberhmm.core.hmm import FiberHMM
from fiberhmm.core.bam_reader import ContextEncoder, FiberRead, read_bam
from fiberhmm.core.model_io import load_model, save_model, load_model_with_metadata
