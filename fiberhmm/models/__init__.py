"""Bundled FiberHMM model files shipped with the package.

Most users can rely on enzyme-specific defaults:

    fiberhmm-apply -i data.bam --enzyme hia5 --seq pacbio -o out/
    fiberhmm-recall-tfs -i out/data_footprints.bam -o recalled.bam --enzyme hia5 --seq pacbio

Use ``--model /path/to/custom.json`` to override with a custom file.

Bundled models
--------------
Enzyme  Seq       Tool     Model
------  --------  -------  ----------------------
hia5    pacbio    apply    hia5_pacbio.json
hia5    pacbio    recall   hia5_pacbio.json
hia5    nanopore  apply    hia5_nanopore.json
hia5    nanopore  recall   hia5_nanopore.json
dddb    (any)     apply    dddb_nanopore.json
dddb    (any)     recall   dddb_nanopore.json
ddda    (any)     apply    ddda_nuc.json
ddda    (any)     recall   ddda_TF.json
"""
from __future__ import annotations

import os
import warnings

_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

# (enzyme, seq_or_None)  →  {tool: filename}
# 'seq_or_None' is None for enzymes where platform does not matter
_BUNDLED: dict[tuple[str, str | None], dict[str, str]] = {
    ('hia5', 'pacbio'):   {'apply': 'hia5_pacbio.json',   'recall': 'hia5_pacbio.json'},
    ('hia5', 'nanopore'): {'apply': 'hia5_nanopore.json', 'recall': 'hia5_nanopore.json'},
    ('dddb', None):       {'apply': 'dddb_nanopore.json', 'recall': 'dddb_nanopore.json'},
    ('ddda', None):       {'apply': 'ddda_nuc.json',      'recall': 'ddda_TF.json'},
}

SUPPORTED_ENZYMES = sorted({e for e, _ in _BUNDLED})
# Enzymes where --seq matters
_SEQ_REQUIRED = {'hia5'}
_SEQ_DEFAULT  = 'pacbio'   # default when --seq omitted for hia5


def _seq_key_for_enzyme(enz: str, seq: str | None) -> str | None:
    if enz in _SEQ_REQUIRED:
        if seq is None:
            warnings.warn(
                f"--seq not specified for {enz}; defaulting to '{_SEQ_DEFAULT}'. "
                f"Use --seq nanopore if your data is Nanopore.",
                stacklevel=3,
            )
            return _SEQ_DEFAULT
        return seq.lower()
    return None


def _valid_enzyme_seq_choices() -> list[str]:
    return [f"{e}/{s or 'any'}" for e, s in sorted(_BUNDLED)]


def _unknown_bundled_model_message(enzyme: str, seq: str | None, tool: str) -> str:
    choices = _valid_enzyme_seq_choices()
    return (
        f"No bundled model for enzyme={enzyme!r} seq={seq!r} tool={tool!r}. "
        f"Valid enzyme/seq combos: {choices}. "
        f"Use --model to provide a custom JSON file."
    )


def _unknown_tool_message(tool: str) -> str:
    return f"Tool {tool!r} not recognised; use 'apply' or 'recall'."


def _missing_bundled_model_message(path: str) -> str:
    return (
        f"Bundled model file missing: {path}. "
        f"The fiberhmm installation may be incomplete."
    )


def _bundled_model_key(enzyme: str, seq: str | None) -> tuple[str, str | None]:
    enz = enzyme.lower()
    return enz, _seq_key_for_enzyme(enz, seq)


def _bundled_model_path(filename: str) -> str:
    return os.path.join(_MODELS_DIR, filename)


def get_model_path(enzyme: str, tool: str = 'recall', seq: str | None = None) -> str:
    """Return the absolute path to a bundled model.

    Parameters
    ----------
    enzyme:
        One of ``'hia5'``, ``'dddb'``, ``'ddda'``.
    tool:
        ``'apply'`` (fiberhmm-apply nuc HMM) or ``'recall'``
        (fiberhmm-recall-tfs TF recaller).
    seq:
        Sequencing platform: ``'pacbio'`` or ``'nanopore'``.
        Required for Hia5; ignored for DddB / DddA.

    Raises
    ------
    KeyError
        If no bundled model exists for the given combination.
    FileNotFoundError
        If the bundled file is missing from the installation.
    """
    t = tool.lower()

    key = _bundled_model_key(enzyme, seq)
    entry = _BUNDLED.get(key)
    if entry is None:
        raise KeyError(_unknown_bundled_model_message(enzyme, seq, tool))

    fname = entry.get(t)
    if fname is None:
        raise KeyError(_unknown_tool_message(tool))

    path = _bundled_model_path(fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(_missing_bundled_model_message(path))
    return path
