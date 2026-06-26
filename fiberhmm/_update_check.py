"""Best-effort "a newer fiberhmm is on PyPI" reminder.

Design constraints (all of them matter):

* **stderr only.** Many tools stream a BAM to stdout (``-o -``); a single byte
  on stdout corrupts that stream. The reminder is written to ``sys.stderr``
  exclusively.
* **Silent on failure.** Offline, PyPI down, slow DNS, unwritable cache -- every
  path is wrapped so the reminder can never delay or break the actual tool.
* **At most one network call per day.** The latest-known version is cached under
  the user cache dir; between checks the cached value is reused, so the common
  case is a single tiny file read. The reminder itself still prints on *every*
  run while outdated -- only the PyPI lookup is throttled.
* **Opt-out.** Set ``FIBERHMM_NO_UPDATE_CHECK=1`` to disable entirely.
"""
from __future__ import annotations

import json
import os
import sys
import time

from fiberhmm import __version__

_PYPI_URL = "https://pypi.org/pypi/fiberhmm/json"
_CHECK_INTERVAL = 24 * 60 * 60  # throttle the network lookup to once per day
_TIMEOUT = 1.5                  # seconds; fail fast rather than hang the CLI


def _cache_path() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache")
    return os.path.join(base, "fiberhmm", "update_check.json")


def _version_tuple(v):
    """Parse a PEP 440-ish version into a comparable int tuple.

    Stops at the first non-numeric token so pre-release suffixes (``2.13.4rc1``)
    don't crash the comparison -- they simply compare as the numeric prefix.
    """
    parts = []
    for tok in str(v).split('.'):
        lead = ''
        for ch in tok:
            if ch.isdigit():
                lead += ch
            else:
                break
        if not lead:
            break
        parts.append(int(lead))
        if lead != tok:
            # token carried a non-numeric suffix (e.g. "4rc1") -> stop here
            break
    return tuple(parts)


def _read_cache() -> dict:
    try:
        with open(_cache_path()) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_cache(data: dict) -> None:
    try:
        path = _cache_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as fh:
            json.dump(data, fh)
    except Exception:
        pass


def _fetch_latest() -> str:
    import urllib.request
    with urllib.request.urlopen(_PYPI_URL, timeout=_TIMEOUT) as resp:
        data = json.load(resp)
    return str(data['info']['version'])


def _resolve_latest(now: int):
    """Latest version string, from cache when fresh else PyPI.

    Falls back to the cached value (possibly None) when the network lookup
    fails, so an offline user with a warm cache still gets reminded.
    """
    cache = _read_cache()
    cached = cache.get('latest')
    if cached and (now - int(cache.get('last_check', 0))) < _CHECK_INTERVAL:
        return cached
    try:
        latest = _fetch_latest()
    except Exception:
        return cached
    _write_cache({'last_check': now, 'latest': latest})
    return latest


def notify_if_outdated(stream=None) -> None:
    """Print a one-line update reminder to stderr if a newer fiberhmm exists.

    No-op on opt-out, on any error, when offline with a cold cache, or when
    already current. Never writes to stdout.
    """
    if os.environ.get("FIBERHMM_NO_UPDATE_CHECK"):
        return
    try:
        latest = _resolve_latest(int(time.time()))
        if not latest:
            return
        if _version_tuple(latest) > _version_tuple(__version__):
            out = stream if stream is not None else sys.stderr
            print(
                f"[fiberhmm] update available: {latest} "
                f"(installed {__version__}) -- pip install -U fiberhmm "
                f"  [silence: FIBERHMM_NO_UPDATE_CHECK=1]",
                file=out,
            )
    except Exception:
        return
