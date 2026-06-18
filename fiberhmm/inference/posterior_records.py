"""Shared posterior record builders for inference output paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class _PosteriorFiberDataRequest:
    read_obj: object
    result: Mapping
    ref_positions: object


def posterior_fiber_data_from_request(
    request: _PosteriorFiberDataRequest,
) -> dict:
    """Build the HDF5 PosteriorWriter payload for one annotated read."""
    return {
        'read_name': request.read_obj.query_name,
        'ref_start': request.read_obj.reference_start,
        'ref_end': request.read_obj.reference_end,
        'strand': request.result.get('strand', '.'),
        'posteriors': request.result['posteriors'],
        'ref_positions': request.ref_positions,
        'footprint_starts': request.result['ns'],
        'footprint_sizes': request.result['nl'],
    }


def posterior_fiber_data(read_obj, result: Mapping, ref_positions) -> dict:
    return posterior_fiber_data_from_request(
        _PosteriorFiberDataRequest(
            read_obj=read_obj,
            result=result,
            ref_positions=ref_positions,
        )
    )
