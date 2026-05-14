"""Small result containers shared by multiprocessing worker drains."""

from typing import NamedTuple, Tuple


class WorkerChunkResult(NamedTuple):
    """Per-chunk worker output plus failures hidden behind pass-through reads."""

    results: list
    read_failures: int = 0


def coerce_worker_chunk_result(value) -> Tuple[list, int]:
    """Accept current structured worker results and legacy bare result lists."""
    if isinstance(value, WorkerChunkResult):
        return value.results, int(value.read_failures)
    return value, 0
