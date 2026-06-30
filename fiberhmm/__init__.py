"""
FiberHMM - Hidden Markov Model toolkit for chromatin footprint calling
from fiber-seq and DAF-seq single-molecule data.
"""

__version__ = "2.16.1"

from fiberhmm.core.bam_reader import ContextEncoder, FiberRead, read_bam
from fiberhmm.core.hmm import FiberHMM
from fiberhmm.core.model_io import load_model, load_model_with_metadata, save_model

__all__ = [
    "ContextEncoder",
    "FiberHMM",
    "FiberRead",
    "load_model",
    "load_model_with_metadata",
    "read_bam",
    "save_model",
]
