"""Progress lines in logs, and worker inits leaving NUMBA_CACHE_DIR alone."""
import io
import os

from fiberhmm.core.progress import progress, progress_end


class _Tty(io.StringIO):
    def isatty(self):
        return True


def test_progress_off_a_terminal_writes_whole_lines():
    stream = io.StringIO()
    progress("  Processed: 10", file=stream)
    progress("  Processed: 20", file=stream)
    progress("  Processed: 20 | done", file=stream, final=True)
    progress_end(stream)
    assert stream.getvalue() == (
        "  Processed: 10\n  Processed: 20\n  Processed: 20 | done\n")


def test_progress_on_a_terminal_rewrites_in_place():
    stream = _Tty()
    progress("a", file=stream)
    progress("b", file=stream)
    progress_end(stream)
    assert stream.getvalue() == "\ra\rb\n"


def test_worker_inits_keep_the_users_numba_cache_dir(monkeypatch, tmp_path):
    from fiberhmm.cli import export_posteriors
    from fiberhmm.inference import region_workers, streaming_workers
    from fiberhmm.models import get_model_path

    model = get_model_path('hia5', tool='apply', seq='pacbio')
    cache = str(tmp_path / 'numba')
    monkeypatch.setenv('NUMBA_CACHE_DIR', cache)
    # The inits set per-process worker globals; restore them afterwards.
    for module, names in ((streaming_workers, ('_worker_model', '_worker_debug_timing',
                                               '_worker_recall_state')),
                          (region_workers, ('_worker_model', '_worker_region_params')),
                          (export_posteriors, ('_worker_model', '_worker_params'))):
        for name in names:
            if hasattr(module, name):
                monkeypatch.setattr(module, name, getattr(module, name))
    streaming_workers._init_bam_worker(model)
    assert os.environ['NUMBA_CACHE_DIR'] == cache
    streaming_workers._init_fused_worker(model)
    assert os.environ['NUMBA_CACHE_DIR'] == cache
    region_workers._init_region_worker(model, {})
    assert os.environ['NUMBA_CACHE_DIR'] == cache
    export_posteriors._init_worker(model, {'mode': 'pacbio-fiber'})
    assert os.environ['NUMBA_CACHE_DIR'] == cache
