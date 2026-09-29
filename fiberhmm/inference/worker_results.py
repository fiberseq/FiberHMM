"""Small result containers shared by multiprocessing worker drains.

Also holds the shared per-read failure policy: workers catch per-read
exceptions so one malformed read cannot take down a chunk, but the failures
must stay visible. Workers ship the first few tracebacks back with their
chunk results, and a run whose failure count exceeds a small fraction of the
reads it processed ends with :class:`WorkerFailureError` instead of exiting 0
with silently untagged reads.
"""

import sys
import traceback
from typing import Iterable, List, NamedTuple, Tuple

# Tracebacks retained per worker chunk and printed per run.
MAX_FAILURE_MESSAGES_PER_CHUNK = 3
MAX_FAILURE_MESSAGES = 5
# A run fails when more than this fraction of processed reads failed. Small
# runs (fewer than MIN_READS_FOR_FRACTION processed reads) fail on any failure.
MAX_FAILURE_FRACTION = 0.01
MIN_READS_FOR_FRACTION = 100


class WorkerChunkResult(NamedTuple):
    """Per-chunk worker output plus failures hidden behind pass-through reads."""

    results: list
    read_failures: int = 0
    failure_messages: tuple = ()


class WorkerFailureError(RuntimeError):
    """Too many per-read failures inside workers for the output to be trusted."""


def coerce_worker_chunk_result(value) -> Tuple[list, int]:
    """Accept current structured worker results and legacy bare result lists."""
    if isinstance(value, WorkerChunkResult):
        return value.results, int(value.read_failures)
    return value, 0


def worker_chunk_failure_messages(value) -> tuple:
    """Return the tracebacks a worker attached to one chunk result, if any."""
    if isinstance(value, WorkerChunkResult):
        return tuple(value.failure_messages or ())
    return ()


def record_failure_message(messages: List[str], read_id=None,
                           limit: int = MAX_FAILURE_MESSAGES_PER_CHUNK) -> None:
    """Append the active exception's traceback (with the read id) if room."""
    if len(messages) >= limit:
        return
    prefix = f"read {read_id}: " if read_id else ""
    messages.append(prefix + traceback.format_exc())


def extend_failure_messages(store: List[str], messages: Iterable[str],
                            limit: int = MAX_FAILURE_MESSAGES) -> None:
    """Keep at most ``limit`` tracebacks across a run."""
    for message in messages:
        if len(store) >= limit:
            return
        store.append(message)


def failure_limit_exceeded(failures: int, processed: int) -> bool:
    """True when per-read failures are too frequent to trust the output."""
    failures = int(failures)
    processed = int(processed)
    if failures <= 0:
        return False
    if processed < MIN_READS_FOR_FRACTION:
        return True
    return failures > MAX_FAILURE_FRACTION * processed


def enforce_worker_failure_policy(failures: int, processed: int,
                                  messages: Iterable[str] = (),
                                  log=None, label: str = "worker") -> None:
    """Report per-read failures and raise when they exceed the policy.

    ``processed`` is the number of reads sent for annotation (not the number
    of records written). The first tracebacks are always printed when any
    read failed, so a tolerated failure is still diagnosable.
    """
    failures = int(failures)
    if failures <= 0:
        return
    log = log or sys.stderr
    messages = list(messages)
    print(
        f"  {label} read failures: {failures:,} of {int(processed):,} processed "
        f"reads (passed through unannotated)",
        file=log,
    )
    for index, message in enumerate(messages[:MAX_FAILURE_MESSAGES], start=1):
        print(f"  --- {label} failure traceback {index} ---\n{message.rstrip()}",
              file=log)
    if failure_limit_exceeded(failures, processed):
        limit = (
            "any failure when fewer than "
            f"{MIN_READS_FOR_FRACTION} reads are processed"
            if int(processed) < MIN_READS_FOR_FRACTION
            else f"{MAX_FAILURE_FRACTION:.0%} of processed reads"
        )
        raise WorkerFailureError(
            f"{failures:,} of {int(processed):,} reads failed inside {label}s "
            f"(limit: {limit}); the output was not written. See the "
            "traceback(s) above."
        )
