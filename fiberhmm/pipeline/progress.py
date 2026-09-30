"""Progress events and step completion markers for ``fiberhmm-pipeline``.

``--progress-json FILE`` appends one JSON object per line and flushes after
each (``-`` writes to stdout). The events follow the FiberBrowser contract
(``fiberhmm.pipeline.outputs.v1``)::

    {"event": "start", "version": "3.0.0", "sample": "...", "outdir": "...",
     "steps": ["prepare_reference", "index", "align", "call", "qc"]}
    {"event": "step", "step": "align", "status": "running", "done": 1200,
     "total": null, "unit": "reads", "rate": 850.0, "eta_s": 4.1}
    {"event": "step", "step": "align", "status": "done", "message": "..."}
    {"event": "step", "step": "index", "status": "skipped", "message": "..."}
    {"event": "log", "level": "info", "message": "..."}
    {"event": "done", "status": "ok", "outputs": {...outputs.json...}}
    {"event": "done", "status": "error", "error": "...", "hint": "..."}

``status`` is ``running``, ``done`` or ``skipped``; ``done``/``total``/
``unit``/``rate``/``eta_s``/``message`` appear when known. Every event also
carries ``time`` (Unix seconds) and ``elapsed_s`` (since the start). The last
line of a run is always a ``done`` event.

A finished step writes ``OUTDIR/.fiberhmm-pipeline/<step>.done`` holding its
fingerprint (the settings and input files it depends on) and its outputs. A
later run skips the step when the fingerprint matches and the outputs exist,
and refuses to run when the fingerprint differs (a different input or
setting in the same output directory) unless the step is redone on purpose.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Callable, Optional

STATE_DIR = ".fiberhmm-pipeline"


class ProgressReporter:
    def __init__(self, path: Optional[str] = None,
                 callback: Optional[Callable[[dict], None]] = None):
        self.start = time.time()
        self.callback = callback
        self._lock = threading.Lock()
        self._handle = None
        if path == "-":
            self._handle = sys.stdout
        elif path:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            self._handle = open(path, "a", encoding="utf-8")
        self._step_start: dict[str, float] = {}

    def emit(self, event: str, **fields) -> dict:
        now = time.time()
        payload = {"event": event}
        payload.update({k: v for k, v in fields.items() if v is not None})
        payload["time"] = round(now, 3)
        payload["elapsed_s"] = round(now - self.start, 3)
        with self._lock:
            if self._handle is not None:
                self._handle.write(json.dumps(payload, default=str) + "\n")
                self._handle.flush()
        if self.callback is not None:
            self.callback(payload)
        return payload

    def step(self, step: str, status: str, *, done: Optional[int] = None,
             total: Optional[int] = None, unit: Optional[str] = None,
             rate: Optional[float] = None, eta_s: Optional[float] = None,
             message: Optional[str] = None, **extra) -> dict:
        if status == "running" and step not in self._step_start:
            self._step_start[step] = time.time()
        if status == "running" and eta_s is None and rate and total and done is not None:
            eta_s = max(0.0, (total - done) / rate)
        return self.emit("step", step=step, status=status, done=done, total=total,
                         unit=unit, rate=None if rate is None else round(float(rate), 2),
                         eta_s=None if eta_s is None else round(float(eta_s), 1),
                         message=message, **extra)

    def step_running_for(self, step: str) -> float:
        return time.time() - self._step_start.get(step, time.time())

    def log(self, message: str, level: str = "info") -> dict:
        return self.emit("log", level=level, message=message)

    def close(self) -> None:
        if self._handle is not None and self._handle is not sys.stdout:
            self._handle.close()
        self._handle = None


def marker_path(outdir: str, step: str) -> str:
    return os.path.join(outdir, STATE_DIR, f"{step}.done")


def file_fingerprint(path: str) -> dict:
    stat = os.stat(path)
    return {"path": os.path.abspath(path), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def read_marker(outdir: str, step: str) -> Optional[dict]:
    try:
        with open(marker_path(outdir, step), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def outputs_exist(outputs) -> bool:
    return all(os.path.exists(path) for path in _output_paths(outputs or {}))


def _output_paths(outputs) -> list[str]:
    paths: list[str] = []
    if isinstance(outputs, str):
        paths.append(outputs)
    elif isinstance(outputs, dict):
        for value in outputs.values():
            paths.extend(_output_paths(value))
    elif isinstance(outputs, (list, tuple)):
        for value in outputs:
            paths.extend(_output_paths(value))
    return paths


def fingerprint_changes(old: dict, new: dict) -> list[str]:
    """Top-level keys whose values differ between two fingerprints."""
    keys = sorted(set(old or {}) | set(new or {}))
    return [k for k in keys if (old or {}).get(k) != (new or {}).get(k)]


def write_marker(outdir: str, step: str, fingerprint: dict, outputs: dict,
                 summary: Optional[dict] = None) -> None:
    path = marker_path(outdir, step)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump({"step": step, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "fingerprint": fingerprint, "outputs": outputs,
                   "summary": summary or {}}, handle, indent=2, default=str)
        handle.write("\n")
    os.replace(tmp, path)


def clear_marker(outdir: str, step: str) -> None:
    try:
        os.remove(marker_path(outdir, step))
    except FileNotFoundError:
        pass
