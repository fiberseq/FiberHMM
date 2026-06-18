"""Helpers for constructing TF-recall LLR tables."""

from __future__ import annotations

from typing import Optional

from fiberhmm.core.model_io import load_model_with_metadata
from fiberhmm.inference.tf_recaller import apply_emission_uplift, build_llr_tables


def build_recall_llr_tables(model, emission_uplift: float = 1.0):
    """Build TF LLR tables and optionally apply the recall emission uplift."""
    llr_hit, llr_miss = build_llr_tables(model)
    if abs(emission_uplift - 1.0) > 1e-9:
        llr_hit, llr_miss = apply_emission_uplift(
            llr_hit,
            llr_miss,
            model,
            emission_uplift,
        )
    return llr_hit, llr_miss


def _resolve_recall_model_path(
    recall_model_path: Optional[str],
    fallback_model_path: Optional[str],
) -> str:
    model_path = recall_model_path or fallback_model_path
    if model_path is None:
        raise ValueError("one of recall_model_path or fallback_model_path is required")
    return model_path


def load_recall_llr_tables(
    recall_model_path: Optional[str],
    fallback_model_path: Optional[str],
    emission_uplift: float = 1.0,
):
    """Load the recall model path, falling back to the apply model, and build tables."""
    model_path = _resolve_recall_model_path(recall_model_path, fallback_model_path)
    model, _, _ = load_model_with_metadata(model_path)
    return build_recall_llr_tables(model, emission_uplift)
