"""Small result containers shared by multiprocessing worker drains."""

from typing import NamedTuple


class WorkerChunkResult(NamedTuple):
    """Per-chunk worker output plus failures hidden behind pass-through reads."""

    results: list
    read_failures: int = 0


def coerce_worker_chunk_result(value) -> WorkerChunkResult:
    """Accept current structured worker results and legacy bare result lists."""
    if isinstance(value, WorkerChunkResult):
        return value
    return WorkerChunkResult(value, 0)
