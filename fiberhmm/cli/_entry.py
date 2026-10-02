"""Console-script entry points.

Each ``fiberhmm-*`` command resolves to a thin wrapper here that fires the
best-effort PyPI update reminder (stderr-only, cached, opt-out) and then
delegates to the tool's real ``main()``. Centralizing it means the reminder
reaches every CLI without sprinkling calls through ten modules, and the
"never touch stdout" reasoning lives in exactly one place. The reminder is
fully isolated -- a failure in it can never block the underlying tool.

Each wrapper also turns the ordinary user errors that would otherwise end in
a Python traceback -- a missing or unreadable input path, or a file that is
not a BAM/CRAM/SAM -- into a one-line ``error:`` message and exit status 2
(:func:`fiberhmm.cli.common.run_reporting_input_errors`).
"""
from __future__ import annotations

from fiberhmm.cli.common import run_reporting_input_errors


def _notify() -> None:
    try:
        from fiberhmm._update_check import notify_if_outdated
        notify_if_outdated()
    except Exception:
        pass


def apply_main():
    _notify()
    from fiberhmm.cli.apply import main
    return run_reporting_input_errors('fiberhmm-apply', main)


def train_main():
    _notify()
    from fiberhmm.cli.train import main
    return run_reporting_input_errors('fiberhmm-train', main)


def extract_main():
    _notify()
    from fiberhmm.cli.extract_tags import main
    return run_reporting_input_errors('fiberhmm-extract', main)


def probs_main():
    _notify()
    from fiberhmm.cli.generate_probs import main
    return run_reporting_input_errors('fiberhmm-probs', main)


def utils_main():
    _notify()
    from fiberhmm.cli.utils import main
    return run_reporting_input_errors('fiberhmm-utils', main)


def posteriors_main():
    _notify()
    from fiberhmm.cli.export_posteriors import main
    return run_reporting_input_errors('fiberhmm-posteriors', main)


def daf_encode_main():
    _notify()
    from fiberhmm.cli.daf_encode import main
    return run_reporting_input_errors('fiberhmm-daf-encode', main)


def daf_snps_main():
    _notify()
    from fiberhmm.cli.daf_snps import main
    return run_reporting_input_errors('fiberhmm-daf-snps', main)


def call_m5c_main():
    _notify()
    from fiberhmm.cli.call_m5c import main
    return run_reporting_input_errors('fiberhmm-call-m5c', main)


def tag_m5c_main():
    _notify()
    from fiberhmm.cli.tag_m5c import main
    return run_reporting_input_errors('fiberhmm-tag-m5c', main)


def recall_tfs_main():
    _notify()
    from fiberhmm.cli.recall_tfs import main
    return run_reporting_input_errors('fiberhmm-recall-tfs', main)


def recall_nucs_main():
    _notify()
    from fiberhmm.cli.recall_tfs import main_recall_nucs
    return run_reporting_input_errors('fiberhmm-recall-nucs', main_recall_nucs)


def call_main():
    _notify()
    from fiberhmm.cli.call import main
    return run_reporting_input_errors('fiberhmm-call', main)


def pipeline_main():
    _notify()
    from fiberhmm.cli.pipeline import main
    return run_reporting_input_errors('fiberhmm-pipeline', main)


def dedup_main():
    _notify()
    from fiberhmm.cli.dedup import main
    return run_reporting_input_errors('fiberhmm-dedup', main)


def pair_main():
    _notify()
    from fiberhmm.cli.pair import main
    return run_reporting_input_errors('fiberhmm-pair', main)


def merge_main():
    _notify()
    from fiberhmm.cli.merge import main
    return run_reporting_input_errors('fiberhmm-merge', main)


def strand_rescue_main():
    _notify()
    from fiberhmm.cli.strand_rescue import main
    return run_reporting_input_errors('fiberhmm-strand-rescue', main)


def strand_rescue_annotate_main():
    _notify()
    from fiberhmm.cli.strand_rescue_annotate import main
    return run_reporting_input_errors('fiberhmm-strand-rescue-annotate', main)


def strand_rescue_audit_main():
    _notify()
    from fiberhmm.cli.strand_rescue_audit import main
    return run_reporting_input_errors('fiberhmm-strand-rescue-audit', main)


def footprint_model_main():
    _notify()
    from fiberhmm.cli.footprint_model import main
    return run_reporting_input_errors('fiberhmm-footprint-model', main)


def tag_families_main():
    _notify()
    from fiberhmm.cli.tag_families import main
    return run_reporting_input_errors('fiberhmm-tag-consensus', main)


def qc_main():
    _notify()
    from fiberhmm.cli.qc import main
    return run_reporting_input_errors('fiberhmm-qc', main)


def check_main():
    _notify()
    from fiberhmm.cli.check import main
    return run_reporting_input_errors('fiberhmm-check', main)
