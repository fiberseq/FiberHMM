"""Tests for the best-effort PyPI update reminder.

None of these hit the network: _resolve_latest / _fetch_latest are monkeypatched
so the suite stays hermetic. The key invariants are (1) never write stdout,
(2) silent on opt-out / error / up-to-date, (3) the network lookup is throttled
and degrades to the cache when offline.
"""
from __future__ import annotations

import fiberhmm._update_check as uc
from fiberhmm import __version__


def test_version_tuple_parsing():
    assert uc._version_tuple("2.13.4") == (2, 13, 4)
    assert uc._version_tuple("2.14") == (2, 14)
    # pre-release suffix degrades to its numeric prefix, never crashes
    assert uc._version_tuple("2.13.4rc1") == (2, 13, 4)
    assert uc._version_tuple("2.13.4") > uc._version_tuple("2.13.3")
    assert uc._version_tuple("2.14.0") > uc._version_tuple("2.13.9")


def test_notify_opt_out_is_silent(monkeypatch, capsys):
    monkeypatch.setenv("FIBERHMM_NO_UPDATE_CHECK", "1")
    monkeypatch.setattr(uc, "_resolve_latest", lambda now: "99.0.0")
    uc.notify_if_outdated()
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_notify_outdated_writes_stderr_only(monkeypatch, capsys):
    monkeypatch.delenv("FIBERHMM_NO_UPDATE_CHECK", raising=False)
    monkeypatch.setattr(uc, "_resolve_latest", lambda now: "999.0.0")
    uc.notify_if_outdated()
    captured = capsys.readouterr()
    assert captured.out == ""                     # MUST never touch stdout
    assert "update available" in captured.err
    assert "999.0.0" in captured.err
    assert __version__ in captured.err


def test_notify_up_to_date_is_silent(monkeypatch, capsys):
    monkeypatch.delenv("FIBERHMM_NO_UPDATE_CHECK", raising=False)
    monkeypatch.setattr(uc, "_resolve_latest", lambda now: __version__)
    uc.notify_if_outdated()
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_notify_silent_when_resolve_raises(monkeypatch, capsys):
    monkeypatch.delenv("FIBERHMM_NO_UPDATE_CHECK", raising=False)

    def boom(now):
        raise RuntimeError("offline")

    monkeypatch.setattr(uc, "_resolve_latest", boom)
    uc.notify_if_outdated()   # must not raise
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_resolve_latest_uses_fresh_cache_without_network(monkeypatch):
    monkeypatch.setattr(uc, "_read_cache",
                        lambda: {"last_check": 1000, "latest": "5.0.0"})

    def no_net():
        raise AssertionError("network must not be called when cache is fresh")

    monkeypatch.setattr(uc, "_fetch_latest", no_net)
    assert uc._resolve_latest(1000 + 10) == "5.0.0"


def test_resolve_latest_falls_back_to_cache_when_fetch_fails(monkeypatch):
    # Stale cache -> attempts fetch -> fetch fails -> returns last-known latest.
    monkeypatch.setattr(uc, "_read_cache",
                        lambda: {"last_check": 0, "latest": "5.0.0"})

    def boom():
        raise RuntimeError("offline")

    monkeypatch.setattr(uc, "_fetch_latest", boom)
    assert uc._resolve_latest(10 ** 9) == "5.0.0"


def test_resolve_latest_refreshes_when_stale(monkeypatch):
    monkeypatch.setattr(uc, "_read_cache",
                        lambda: {"last_check": 0, "latest": "1.0.0"})
    monkeypatch.setattr(uc, "_fetch_latest", lambda: "7.0.0")
    written = {}
    monkeypatch.setattr(uc, "_write_cache", lambda d: written.update(d))
    assert uc._resolve_latest(10 ** 9) == "7.0.0"
    assert written.get("latest") == "7.0.0"


def test_entry_notify_is_isolated(monkeypatch):
    from fiberhmm.cli import _entry

    def boom(*a, **k):
        raise RuntimeError("reminder blew up")

    monkeypatch.setattr(uc, "notify_if_outdated", boom)
    # A broken reminder must never propagate out of the entry wrapper.
    _entry._notify()
