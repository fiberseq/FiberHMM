"""Bundled FiberHMM model files shipped with the package.

Most users can rely on enzyme-specific defaults:

    fiberhmm-apply -i data.bam --enzyme hia5 --seq pacbio -o out/
    fiberhmm-recall-tfs -i out/data_footprints.bam -o recalled.bam --enzyme hia5 --seq pacbio

Use ``--model /path/to/custom.json`` to override with a custom file.

Supported bundled models
------------------------
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
ddda    (any)     nuc_refine  ddda_nuc_refine.json (internal, deliberately permissive frozen likelihoods)

Additional model-development artifacts may be present in this package, but
they are not public enzyme presets and carry no supported-workflow claim.
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
        # Deliberately NOT DddA emissions. The radial nucleosome refiner needs
        # permissive likelihoods for loose segmentation (it was calibrated with
        # the original TF-table likelihoods, which are the legacy G/T-swapped
        # DddB table). True SsDddA LLRs over-split nucleosomes on dense internal
        # deamination. Do not "correct" or re-index this file; keep it frozen
        # independently of DddA TF-recaller or DddB table changes.
        'nuc_refine': 'ddda_nuc_refine.json',
        'mode': 'daf',
    },
}

SUPPORTED_ENZYMES = ("ddda", "dddb", "hia5")
DEVELOPMENT_ENZYMES = tuple(
    sorted({enzyme for enzyme, _seq in _BUNDLED} - set(SUPPORTED_ENZYMES))
)
# Enzymes where --seq selects the observation frame. EcoGII reuses one
# chemistry-calibrated emission table; ONT selects strand-aware nanopore-fiber
# encoding while PacBio selects pacbio-fiber encoding.
_SEQ_REQUIRED = {'hia5', 'ecogii', 'sssi'}
_SEQ_DEFAULT  = 'pacbio'   # default when --seq is omitted for a platform model


def enzyme_requires_platform(enzyme: str | None) -> bool:
    """True when ``--seq`` selects the bundled model / observation frame."""
    return bool(enzyme) and str(enzyme).lower() in _SEQ_REQUIRED


def bundled_models_differ_by_tool(enzyme: str | None, seq: str | None = None) -> bool:
    """True when the preset ships different apply and recall tables (DddA)."""
    if not enzyme:
        return False
    try:
        entry = _get_bundled_entry(enzyme, seq, warn_missing_seq=False)
    except KeyError:
        return False
    return entry.get('apply') != entry.get('recall')


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
        One of the public enzyme presets: ``'hia5'``, ``'dddb'`` or ``'ddda'``.
        Registry entries for development artifacts are an internal API and do
        not constitute a supported workflow.
    tool:
        ``'apply'`` (fiberhmm-apply nuc HMM), ``'recall'``
        (fiberhmm-recall-tfs TF recaller), or the internal ``'nuc_refine'``
        likelihood model where bundled separately.
    seq:
        Sequencing platform: ``'pacbio'`` or ``'nanopore'``.
        Required for the public Hia5 preset; ignored for DddB / DddA.
        Internal development entries follow their registry metadata.

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
            f"Tool {tool!r} not recognised for {enzyme!r}."
        )

    path = os.path.join(_MODELS_DIR, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Bundled model file missing: {path}. "
            f"The fiberhmm installation may be incomplete."
        )
    return path


# ---------------------------------------------------------------------------
# Per-chemistry default ML threshold
# ---------------------------------------------------------------------------

#: Default minimum ML probability (0-255) for calling from MM/ML, used by
#: ``fiberhmm-call`` and ``fiberhmm-apply`` and by the DAF tools
#: (``fiberhmm-dedup``/``-pair``/``-merge``) for MM/ML-native dU calls.
DEFAULT_PROB_THRESHOLD = 128

#: Chemistry-specific overrides of that default, keyed by (enzyme, platform).
#: Nanopore Hia5 m6A is a strict hard-call assay: Dorado's m6A ML values are
#: only reliable near the top of the scale, and the bundled Hia5 Nanopore QC
#: reference (``fiberhmm/qc/references.json``) and the strand-rescue
#: ``hia5-nanopore`` preset are calibrated at 248.
PROB_THRESHOLD_OVERRIDES: dict[tuple[str, str], int] = {
    ('hia5', 'nanopore'): 248,
}


def default_prob_threshold(
    enzyme: str | None,
    seq: str | None,
    fallback: int = DEFAULT_PROB_THRESHOLD,
) -> int:
    """Default ML threshold for one chemistry.

    ``enzyme``/``seq`` are the *resolved* chemistry (after ``--seq`` detection
    and after a custom model inherits the input BAM's declared chemistry).
    Returns the chemistry override when one exists (Hia5 + Nanopore: 248) and
    ``fallback`` otherwise, so each tool keeps its own historical default for
    every other chemistry. An explicit ``--prob-threshold`` always wins; call
    this only when the user did not pass one.
    """
    key = (str(enzyme or '').lower(), str(seq or '').lower())
    return int(PROB_THRESHOLD_OVERRIDES.get(key, fallback))


def resolve_prob_threshold(
    explicit: int | None,
    enzyme: str | None,
    seq: str | None,
    fallback: int = DEFAULT_PROB_THRESHOLD,
) -> int:
    """``explicit`` when given, else :func:`default_prob_threshold`."""
    if explicit is not None:
        return int(explicit)
    return default_prob_threshold(enzyme, seq, fallback)


def declared_prob_threshold_chemistry(header) -> tuple[str | None, str | None]:
    """(enzyme, platform) from a BAM header's FIBERHMM-CHEMISTRY declaration.

    Falls back to the compatibility inference from a pre-declaration
    ``fiberhmm-call`` @PG record. Returns ``(None, None)`` when nothing is
    known, or when the header declares several different chemistries. Used by
    tools that read an existing FiberHMM BAM (extract, recall, qc) and have no
    ``--enzyme`` of their own.
    """
    try:
        from fiberhmm.io.bam_header import (
            declared_chemistries,
            infer_legacy_chemistry,
        )
        declared = declared_chemistries(header)
        if not declared:
            legacy = infer_legacy_chemistry(header)
            declared = [legacy] if legacy else []
    except Exception:
        return None, None
    pairs = {
        (str(item.get('enzyme', '')).lower() or None,
         str(item.get('platform', '')).lower() or None)
        for item in declared
    }
    if len(pairs) != 1:
        return None, None
    return next(iter(pairs))
