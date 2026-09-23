"""Signal handling around the shared worker pool."""
import signal

from fiberhmm.inference.consensus.execution import _terminate_as_exit


def test_ignored_hangup_stays_ignored_under_nohup():
    previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        with _terminate_as_exit():
            assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGHUP, previous)


def test_default_terminate_becomes_system_exit_and_is_restored():
    previous = signal.getsignal(signal.SIGTERM)
    with _terminate_as_exit():
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler) and handler not in (signal.SIG_DFL, signal.SIG_IGN)
        try:
            handler(signal.SIGTERM, None)
        except SystemExit as exit_:
            assert exit_.code == 128 + signal.SIGTERM
        else:
            raise AssertionError('SIGTERM handler must raise SystemExit')
    assert signal.getsignal(signal.SIGTERM) == previous
