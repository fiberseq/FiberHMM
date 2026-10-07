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


def test_explicit_pool_borrow_keeps_workers_until_outer_run_finishes(monkeypatch):
    from fiberhmm.inference.consensus import execution as E
    events = []
    class Pool:
        def shutdown(self, **kwargs):
            events.append(('closed', kwargs))
    def create(cores):
        events.append(('created', cores))
        return Pool()
    monkeypatch.setattr(E, '_process_pool', create)
    with E.shared_worker_pool(2) as outer:
        executor = outer.get()
        for _ in range(3):
            with E.shared_worker_pool(2, reuse=True) as inner:
                assert inner is outer and inner.get() is executor
            assert events == [('created', 2)]
    assert events == [('created', 2), ('closed', dict(wait=True, kill_workers=False))]
