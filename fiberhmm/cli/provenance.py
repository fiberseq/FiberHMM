"""Shared scientific-provenance helpers for FiberHMM producer CLIs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


def chemistry_declaration(
    args,
    mode,
    apply_model_path,
    recall_model_path,
    nuc_profile_identity=None,
    nuc_profile_sha256=None,
):
    """Build the stable BAM-header chemistry contract for one call run."""
    if mode == "daf":
        assay = "daf"
    elif mode in {"pacbio-fiber", "nanopore-fiber"}:
        assay = "fiber-seq"
    else:
        assay = "custom"

    platform = args.seq
    if not platform and mode == "pacbio-fiber":
        platform = "pacbio"
    elif not platform and mode == "nanopore-fiber":
        platform = "nanopore"
    elif not platform and mode == "daf" and args.enzyme == "ddda":
        # Supported DddA models are calibrated to the PacBio DAF workflow.
        platform = "pacbio"
    elif not platform and mode == "daf" and args.enzyme == "dddb":
        # The bundled DddB model is the Nanopore DAF calibration.
        platform = "nanopore"

    model_stem = Path(recall_model_path or apply_model_path).stem
    model_name = re.sub(r"[^A-Za-z0-9_.+-]+", "_", model_stem).strip("_")
    declaration = {
        "assay": assay,
        "enzyme": args.enzyme or "custom",
        "platform": platform or "unknown",
        "mode": mode,
    }
    if model_name:
        declaration["model"] = model_name
    if nuc_profile_identity:
        declaration["nuc_model"] = str(nuc_profile_identity)
    if nuc_profile_sha256:
        declaration["nuc_sha256"] = str(nuc_profile_sha256)
    return declaration


def nuc_profile_identity(path):
    """Stable BAM-header identity for a nucleosome profile, or ``None``."""
    if not path:
        return None
    profile_path = Path(path)
    identity = profile_path.stem
    try:
        with profile_path.open() as handle:
            identity = str(json.load(handle).get("kind") or identity)
    except (OSError, ValueError, TypeError):
        pass
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", identity).strip("_") or None


def nuc_profile_sha256(path):
    """Exact BAM-header digest for a nucleosome profile, or ``None``."""
    if not path:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Reconciling a run's chemistry declaration with the input BAM's
# ---------------------------------------------------------------------------

# Declaration values that mean "not stated by this run" and may therefore be
# inherited from an input BAM that states them.
_PLACEHOLDER_VALUES = {"custom", "unknown"}
_CORE_FIELDS = ("assay", "enzyme", "platform", "mode")
# Private pg_record key: replace, rather than reconcile with, the input's
# chemistry declarations (``--replace-chemistry``).
REPLACE_CHEMISTRY_KEY = "_replace_chemistry"


class ChemistryConflictError(ValueError):
    """The run's chemistry contradicts the chemistry the input BAM declares."""


def _core(declaration):
    return {key: str(declaration.get(key, "")).lower() for key in _CORE_FIELDS}


def _describe(declaration):
    return " ".join(f"{key}={declaration.get(key, '?')}" for key in _CORE_FIELDS)


def reconcile_chemistry(input_header, requested, *, replace=False, tool=None):
    """Return the declaration this run should append to ``input_header``.

    A custom-model run (``enzyme=custom`` and/or ``platform=unknown``) whose
    observation mode matches the input's declaration inherits the input's
    assay, enzyme and platform, so recalling a DddB- or Hia5-called BAM with a
    refit table keeps its chemistry. Any remaining disagreement raises
    :class:`ChemistryConflictError` with a one-line fix, unless ``replace`` is
    set, in which case the requested declaration is used unchanged (the caller
    then drops the input's declarations; see :func:`output_header_with_provenance`).
    """
    if not requested:
        return requested
    from fiberhmm.io.bam_header import declared_chemistries

    existing = declared_chemistries(input_header)
    if not existing or replace:
        return dict(requested)

    resolved = dict(requested)
    same_mode = [
        declaration for declaration in existing
        if str(declaration.get("mode", "")).lower()
        == str(requested.get("mode", "")).lower()
    ]
    if same_mode:
        source = same_mode[0]
        for key in ("assay", "enzyme", "platform"):
            value = str(resolved.get(key, "")).lower()
            if value in _PLACEHOLDER_VALUES or not value:
                resolved[key] = source[key]

    for declaration in existing:
        if _core(declaration) != _core(resolved):
            prefix = f"{tool}: " if tool else ""
            raise ChemistryConflictError(
                f"{prefix}the input BAM declares chemistry "
                f"[{_describe(declaration)}] but this run would declare "
                f"[{_describe(resolved)}]. Pass --enzyme/--seq matching the "
                "input (with --model for a custom table), or "
                "--replace-chemistry to re-declare the output deliberately."
            )
    return resolved


def strip_chemistry_declarations(header):
    """Copy of ``header`` without ``FIBERHMM-CHEMISTRY`` @CO declarations."""
    import pysam

    from fiberhmm.io.bam_header import CHEMISTRY_PREFIX

    data = header.to_dict() if hasattr(header, "to_dict") else dict(header)
    comments = [
        comment for comment in data.get("CO", [])
        if not str(comment).startswith(CHEMISTRY_PREFIX)
    ]
    if comments:
        data["CO"] = comments
    else:
        data.pop("CO", None)
    return pysam.AlignmentHeader.from_dict(data)


def output_header_with_provenance(input_header, pg_record):
    """Input header plus this run's @PG record and reconciled chemistry.

    The single place producer pipelines turn a ``pg_record`` into an output
    header. ``pg_record`` may carry :data:`REPLACE_CHEMISTRY_KEY` to replace
    the input's chemistry declarations instead of reconciling with them.
    """
    from fiberhmm.io.bam_header import maybe_append_pg

    if not pg_record:
        return input_header
    record = dict(pg_record)
    replace = bool(record.pop(REPLACE_CHEMISTRY_KEY, False))
    header = input_header
    if record.get("chemistry"):
        record["chemistry"] = reconcile_chemistry(
            input_header, record["chemistry"], replace=replace,
            tool=record.get("PN"),
        )
        if replace:
            header = strip_chemistry_declarations(input_header)
    return maybe_append_pg(header, record)
