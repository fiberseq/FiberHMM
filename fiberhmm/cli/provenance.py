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
    *,
    nuc_model_path=None,
):
    """Build the stable BAM-header chemistry contract for one call run.

    ``apply_model_path`` is the emission table of the footprint (apply) pass
    and ``recall_model_path`` the table of the recall pass; either is None
    when the run has no such pass (``fiberhmm-recall-tfs`` has no apply pass,
    ``fiberhmm-apply`` no recall pass). ``nuc_model_path`` is a separate
    nucleosome-likelihood table (DddA nucleosome refinement), when used.

    Besides assay/enzyme/platform/mode, the declaration records the exact
    code and tables behind the calls (see :func:`run_identity_fields`), which
    :mod:`fiberhmm.advisories` uses to decide whether a BAM needs re-running.
    """
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
    declaration.update(run_identity_fields(
        apply_model_path, recall_model_path, nuc_model_path))
    return declaration


def run_identity_fields(apply_model_path=None, recall_model_path=None,
                        nuc_model_path=None):
    """Declaration fields naming the code and tables of one producer run.

    ``apply_sha256``/``recall_sha256``/``nuc_model_sha256`` are sha256 digests
    of the emission-table files the run read (omitted for a pass the run does
    not have), ``fiberhmm_version`` is the package version and
    ``fiberhmm_commit`` the git commit when known (``+dirty`` when the
    package's tracked files differ from it; omitted for source trees without
    git metadata). All values fit the ``FIBERHMM-CHEMISTRY`` value alphabet.
    """
    from fiberhmm.identity import fiberhmm_commit, fiberhmm_version, file_sha256

    fields = {}
    for key, path in (
        ("apply_sha256", apply_model_path),
        ("recall_sha256", recall_model_path),
        ("nuc_model_sha256", nuc_model_path),
    ):
        digest = file_sha256(path)
        if digest:
            fields[key] = digest
    version = re.sub(r"[^A-Za-z0-9_.+-]+", "_", fiberhmm_version()).strip("_")
    if version:
        fields["fiberhmm_version"] = version
    commit = fiberhmm_commit()
    if commit and re.fullmatch(r"[0-9a-f]{40}(\+dirty)?", commit):
        fields["fiberhmm_commit"] = commit
    return fields


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
# Private pg_record key: the run already chose its enzyme-dependent defaults
# (see :func:`resolve_effective_chemistry`). Inheriting a supported enzyme
# only when the output header is written (stdin input, whose header cannot be
# read in advance) would advertise defaults the run did not use, so it is
# refused.
DEFAULTS_RESOLVED_KEY = "_enzyme_defaults_resolved"


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


def resolve_effective_chemistry(args, mode, input_header, apply_model_path,
                                recall_model_path, *, replace=False, tool=None):
    """Reconcile this run with the input and adopt the inherited chemistry.

    Call after the model paths and observation mode are resolved and BEFORE
    any enzyme-dependent default (CpG masking, DAF run mask, ML threshold,
    nucleosome policy, TF presets, dedup/SNP screens) is chosen. A custom
    ``--model`` without ``--enzyme``/``--seq`` inherits the input BAM's
    declared enzyme/platform (see :func:`reconcile_chemistry`); the inherited
    values are written back to ``args.enzyme``/``args.seq`` so every later
    default is exactly the one ``--enzyme <inherited>`` would select, while
    the explicitly given model files stay in use. Only supported presets are
    adopted (development chemistries have no defaults to inherit). Returns the
    reconciled declaration; raises :class:`ChemistryConflictError` on a
    conflict.
    """
    from fiberhmm.models import SUPPORTED_ENZYMES

    requested = chemistry_declaration(
        args, mode, apply_model_path, recall_model_path)
    if input_header is None:
        return requested
    resolved = reconcile_chemistry(
        input_header, requested, replace=replace, tool=tool)
    adopted = []
    enzyme = str(resolved.get("enzyme", "")).lower()
    if not getattr(args, "enzyme", None) and enzyme in SUPPORTED_ENZYMES:
        args.enzyme = enzyme
        adopted.append(f"--enzyme {enzyme}")
    platform = str(resolved.get("platform", "")).lower()
    if (
        adopted
        and not getattr(args, "seq", None)
        and platform in ("pacbio", "nanopore")
    ):
        args.seq = platform
        adopted.append(f"--seq {platform}")
    if adopted:
        import sys

        prefix = f"{tool}: " if tool else ""
        print(
            f"NOTE: {prefix}custom model on an input declaring "
            f"[{_describe(resolved)}]; using the defaults of "
            f"{' '.join(adopted)} with the given model file(s).",
            file=sys.stderr,
        )
    return resolved


def _refuse_late_enzyme_inheritance(requested, resolved, tool):
    from fiberhmm.models import SUPPORTED_ENZYMES

    before = str(requested.get("enzyme", "")).lower()
    after = str(resolved.get("enzyme", "")).lower()
    if before in _PLACEHOLDER_VALUES and after in SUPPORTED_ENZYMES:
        prefix = f"{tool}: " if tool else ""
        raise ChemistryConflictError(
            f"{prefix}the input declares enzyme={after}, but its header was "
            "not available before this custom-model run chose its "
            "enzyme-dependent defaults (stdin input). Pass "
            f"--enzyme {after} (with --model for the custom table), or "
            "--replace-chemistry to declare the run as custom."
        )


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
    from fiberhmm.io.bam_header import append_chemistry, append_pg_record

    if not pg_record:
        return input_header
    record = dict(pg_record)
    replace = bool(record.pop(REPLACE_CHEMISTRY_KEY, False))
    defaults_resolved = bool(record.pop(DEFAULTS_RESOLVED_KEY, False))
    header = input_header
    if record.get("chemistry"):
        requested = record["chemistry"]
        record["chemistry"] = reconcile_chemistry(
            input_header, requested, replace=replace,
            tool=record.get("PN"),
        )
        if defaults_resolved:
            _refuse_late_enzyme_inheritance(
                requested, record["chemistry"], record.get("PN"))
        if replace:
            header = strip_chemistry_declarations(input_header)
    chemistry = record.pop("chemistry", None)
    output = append_pg_record(header, record)
    if not chemistry:
        return output
    # Link the declaration to the @PG line of this run (IDs are made unique
    # by append_pg_record), so each run's tables and code can be attributed.
    programs = output.to_dict().get("PG", [])
    program_id = str(programs[-1].get("ID", "")) if programs else ""
    chemistry = dict(chemistry)
    if program_id and re.fullmatch(r"[A-Za-z0-9_.+-]+", program_id):
        chemistry["pg"] = program_id
    return append_chemistry(output, chemistry)
