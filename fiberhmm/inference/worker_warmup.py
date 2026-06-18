"""Worker process warmup helpers shared by inference backends."""

from __future__ import annotations

import os

import numpy as np


def disable_numba_cache_locking() -> None:
    os.environ['NUMBA_CACHE_DIR'] = ''


def warm_up_model_predict(model) -> None:
    from fiberhmm.core.hmm import HAS_NUMBA

    if HAS_NUMBA:
        dummy_obs = np.array([0, 1, 2, 3], dtype=np.int32)
        try:
            _ = model.predict(dummy_obs)
        except Exception:
            pass


def _posterior_warmup_obs(length: int) -> np.ndarray:
    return np.zeros(max(1, int(length)), dtype=np.int32)


def _tf_warmup_obs() -> np.ndarray:
    return np.zeros(16, dtype=np.int32)


def warm_up_model_posteriors(model, length: int = 100) -> None:
    from fiberhmm.core.hmm import HAS_NUMBA

    if not HAS_NUMBA:
        return

    dummy_obs = _posterior_warmup_obs(length)
    try:
        _ = model.predict(dummy_obs)
        _ = model.predict_proba(dummy_obs)
    except Exception:
        pass


def warm_up_tf_recaller(llr_hit, llr_miss) -> None:
    from fiberhmm.core.hmm import HAS_NUMBA

    if HAS_NUMBA:
        from fiberhmm.inference.tf_recaller import call_tfs_in_interval

        try:
            _ = call_tfs_in_interval(
                _tf_warmup_obs(), 0, 16,
                llr_hit, llr_miss, min_llr=4.0, min_opps=3,
            )
        except Exception:
            pass
