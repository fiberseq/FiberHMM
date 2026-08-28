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
ecogii  pacbio    apply    ecogii_pacbio.json
ecogii  pacbio    recall   ecogii_pacbio.json
ecogii  nanopore  apply    ecogii_pacbio.json
ecogii  nanopore  recall   ecogii_pacbio.json
sssi    nanopore  apply    cpg_nanopore.json
sssi    nanopore  recall   cpg_nanopore.json
dddb    (any)     apply    dddb_nanopore.json
dddb    (any)     recall   dddb_nanopore.json
ddda    (any)     apply    ddda_nuc.json
ddda    (any)     recall   ddda_TF.json
"""
from __future__ import annotations

import os
import warnings

_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))


def _bundled_model_path(filename: str) -> str:
    """Absolute path to a bundled file under fiberhmm/models/ (no existence check)."""
    return os.path.join(_MODELS_DIR, filename)


# (enzyme, seq_or_None)  →  {tool: filename, mode: observation mode}
# 'seq_or_None' is None for enzymes where platform does not matter
_BUNDLED: dict[tuple[str, str | None], dict[str, object]] = {
    ('hia5', 'pacbio'): {
        'apply': 'hia5_pacbio.json',
        'recall': 'hia5_pacbio.json',
        'mode': 'pacbio-fiber',
    },
    ('hia5', 'nanopore'): {
        'apply': 'hia5_nanopore.json',
        'recall': 'hia5_nanopore.json',
        'mode': 'nanopore-fiber',
    },
    ('ecogii', 'pacbio'): {
        'apply': 'ecogii_pacbio.json',
        'recall': 'ecogii_pacbio.json',
        'mode': 'pacbio-fiber',
    },
    # The existing EcoGII parameter file can be evaluated on both platforms.
    # ONT changes the observation frame to strand-aware nanopore-fiber; the
    # parameter file itself was fitted on PacBio data and remains subject to
    # independent ONT calibration/benchmarking.
    ('ecogii', 'nanopore'): {
        'apply': 'ecogii_pacbio.json',
        'recall': 'ecogii_pacbio.json',
        'mode': 'nanopore-fiber',
        # The shared JSON predates ONT support and truthfully records the
        # PacBio frame in its standalone metadata. For this bundled alias the
        # registry supplies the ONT frame, so this mismatch is intentional.
        'metadata_mode_aliases': ('pacbio-fiber',),
    },
    ('sssi', 'nanopore'): {
        'apply': 'cpg_nanopore.json',
        'recall': 'cpg_nanopore.json',
        'mode': 'cpg',
    },
    ('dddb', None): {
        'apply': 'dddb_nanopore.json',
        'recall': 'dddb_nanopore.json',
        'mode': 'daf',
    },
    ('ddda', None): {
        'apply': 'ddda_nuc.json',
        'recall': 'ddda_TF.json',
        'mode': 'daf',
    },
}

SUPPORTED_ENZYMES = sorted({e for e, _ in _BUNDLED})
# Enzymes where --seq selects the observation frame. EcoGII reuses one
# chemistry-calibrated emission table; ONT selects strand-aware nanopore-fiber
# encoding while PacBio selects pacbio-fiber encoding.
_SEQ_REQUIRED = {'hia5', 'ecogii', 'sssi'}
_SEQ_DEFAULT  = 'pacbio'   # default when --seq is omitted for a platform model


def _get_bundled_entry(
    enzyme: str,
    seq: str | None,
    *,
    warn_missing_seq: bool,
) -> dict[str, object]:
    """Resolve a registry entry shared by model-path and mode lookup."""
    enz = enzyme.lower()

    if enz in _SEQ_REQUIRED:
        if seq is None:
            if warn_missing_seq:
                warnings.warn(
                    f"--seq not specified for {enz}; defaulting to "
                    f"'{_SEQ_DEFAULT}'. Use --seq nanopore if your data is "
                    "Nanopore.",
                    stacklevel=3,
                )
            seq_key: str | None = _SEQ_DEFAULT
        else:
            seq_key = seq.lower()
    else:
        seq_key = None

    entry = _BUNDLED.get((enz, seq_key))
    if entry is None:
        choices = [f"{e}/{s or 'any'}" for e, s in sorted(_BUNDLED)]
        raise KeyError(
            f"No bundled model for enzyme={enzyme!r} seq={seq!r}. "
            f"Valid enzyme/seq combos: {choices}. "
            f"Use --model to provide a custom JSON file."
        )
    return entry


def get_observation_mode(
    enzyme: str,
    seq: str | None = None,
    *,
    warn_missing_seq: bool = True,
) -> str:
    """Return the authoritative mode for a bundled enzyme/platform workflow."""
    entry = _get_bundled_entry(
        enzyme, seq, warn_missing_seq=warn_missing_seq
    )
    return str(entry['mode'])


def get_metadata_mode_aliases(
    enzyme: str,
    seq: str | None = None,
    *,
    warn_missing_seq: bool = True,
) -> tuple[str, ...]:
    """Return intentional bundled metadata modes accepted without warning.

    This is narrowly used when one calibrated chemistry file is registered for
    more than one platform observation frame. It does not weaken validation
    for custom models or unrelated bundled-model metadata mismatches.
    """
    entry = _get_bundled_entry(
        enzyme, seq, warn_missing_seq=warn_missing_seq
    )
    return tuple(entry.get('metadata_mode_aliases', ()))


def get_model_path(enzyme: str, tool: str = 'recall', seq: str | None = None) -> str:
    """Return the absolute path to a bundled model.

    Parameters
    ----------
    enzyme:
        One of the bundled enzyme presets, including ``'hia5'``, ``'ecogii'``,
        ``'dddb'``, ``'ddda'``, and ``'sssi'``.
    tool:
        ``'apply'`` (fiberhmm-apply nuc HMM) or ``'recall'``
        (fiberhmm-recall-tfs TF recaller).
    seq:
        Sequencing platform: ``'pacbio'`` or ``'nanopore'``.
        Required for Hia5, EcoGII, and SssI; ignored for DddB / DddA.

    Raises
    ------
    KeyError
        If no bundled model exists for the given combination.
    FileNotFoundError
        If the bundled file is missing from the installation.
    """
    t   = tool.lower()
    entry = _get_bundled_entry(enzyme, seq, warn_missing_seq=True)

    fname = entry.get(t)
    if not isinstance(fname, str):
        raise KeyError(
            f"Tool {tool!r} not recognised; use 'apply' or 'recall'."
        )

    path = os.path.join(_MODELS_DIR, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Bundled model file missing: {path}. "
            f"The fiberhmm installation may be incomplete."
        )
    return path
