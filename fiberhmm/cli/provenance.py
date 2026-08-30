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
