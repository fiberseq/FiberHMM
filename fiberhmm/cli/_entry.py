"""Console-script entry points.

Each ``fiberhmm-*`` command resolves to a thin wrapper here that fires the
best-effort PyPI update reminder (stderr-only, cached, opt-out) and then
delegates to the tool's real ``main()``. Centralizing it means the reminder
reaches every CLI without sprinkling calls through ten modules, and the
"never touch stdout" reasoning lives in exactly one place. The reminder is
fully isolated -- a failure in it can never block the underlying tool.
"""
from __future__ import annotations


def _notify() -> None:
    try:
        from fiberhmm._update_check import notify_if_outdated
        notify_if_outdated()
    except Exception:
        pass


def apply_main():
    _notify()
    from fiberhmm.cli.apply import main
    return main()


def train_main():
    _notify()
    from fiberhmm.cli.train import main
    return main()


def extract_main():
    _notify()
    from fiberhmm.cli.extract_tags import main
    return main()


def probs_main():
    _notify()
    from fiberhmm.cli.generate_probs import main
    return main()


def utils_main():
    _notify()
    from fiberhmm.cli.utils import main
    return main()


def posteriors_main():
    _notify()
    from fiberhmm.cli.export_posteriors import main
    return main()


def daf_encode_main():
    _notify()
    from fiberhmm.cli.daf_encode import main
    return main()


def recall_tfs_main():
    _notify()
    from fiberhmm.cli.recall_tfs import main
    return main()


def recall_nucs_main():
    _notify()
    from fiberhmm.cli.recall_tfs import main_recall_nucs
    return main_recall_nucs()


def call_main():
    _notify()
    from fiberhmm.cli.call import main
    return main()


def run_main():
    _notify()
    from fiberhmm.cli.run import main
    return main()


def dedup_main():
    _notify()
    from fiberhmm.cli.dedup import main
    return main()
