"""Shared posterior record builders for inference output paths."""

from __future__ import annotations

from typing import Mapping


def posterior_fiber_data(read_obj, result: Mapping, ref_positions) -> dict:
    """Build the HDF5 PosteriorWriter payload for one annotated read."""
    return {
        'read_name': read_obj.query_name,
        'ref_start': read_obj.reference_start,
        'ref_end': read_obj.reference_end,
        'strand': result.get('strand', '.'),
        'posteriors': result['posteriors'],
        'ref_positions': ref_positions,
        'footprint_starts': result['ns'],
        'footprint_sizes': result['nl'],
    }
