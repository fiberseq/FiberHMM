"""Optional CUDA/MPS scoring of the unchanged native predictive experiment.

Random draws, log increments and ordered prefix sums stay on the reference
CPU/Numba path. Only interval maximization is accelerated. This deliberately
avoids substituting a different RNG, changing draw counts, or rounding native
emission probabilities to float32 on MPS. Borderline device results are
recomputed with the original float64 prefixes on CPU. No torch import occurs
on the default CPU workflow.
"""
from __future__ import annotations

from collections import Counter
import os
import time
import warnings

import numpy as np
from numba import njit


def resolve_device(requested):
    if requested not in ('cpu', 'cuda', 'mps', 'auto'):
        raise ValueError('Predictive backend must be cpu, cuda, mps or auto')
    if requested == 'cpu':
        return 'cpu', None
    try:
        import torch
    except ImportError:
        return 'cpu', 'PyTorch is not installed'
    if requested in ('cuda', 'auto') and torch.cuda.is_available():
        return 'cuda', None
    if requested in ('mps', 'auto') and torch.backends.mps.is_available():
        if os.environ.get('PYTORCH_MPS_FAST_MATH') == '1':
            return 'cpu', 'MPS fast math invalidates the numerical recheck contract'
        return 'mps', None
    return 'cpu', f'{requested} device is unavailable'


@njit(cache=True, nogil=True)
def _reference_prefixes(pa, pp, starts, ends, cdf, replicates, seed):
    """Same random-call order and scalar float64 arithmetic as CPU scoring."""
    np.random.seed(seed)
    hit_step = np.empty(len(pa)); miss_step = np.empty(len(pa))
    for j in range(len(pa)):
        hit_step[j] = np.log(pp[j]/pa[j])
        miss_step[j] = np.log1p(-pp[j])-np.log1p(-pa[j])
    out = np.zeros((replicates, len(pa)+1))
    for i in range(replicates):
        g = np.searchsorted(cdf, np.random.random())
        for j in range(len(pa)):
            p = pp[j] if starts[g] <= j < ends[g] else pa[j]
            hit = np.random.random() < p
            out[i, j+1] = out[i, j]+(hit_step[j] if hit else miss_step[j])
    return out


@njit(cache=True, nogil=True)
def _reference_events(prefixes, starts, ends, penalty, threshold):
    out = np.empty(len(prefixes), np.bool_)
    for i in range(len(prefixes)):
        best = -np.inf; explained = -np.inf
        for j in range(len(starts)):
            value = prefixes[i, ends[j]]-prefixes[i, starts[j]]
            best = max(best, value)
            explained = max(explained, value+penalty[j])
        out[i] = best-explained >= threshold-1e-10
    return out


def validate_request(request):
    pa, pp, a, b, penalty, cdf, threshold, draws, seed = request
    if (not isinstance(draws, (int, np.integer)) or isinstance(draws, bool) or draws < 1
            or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32):
        raise ValueError('Positive draws and a uint32 seed required')
    if (np.ndim(pa) != 1 or np.shape(pp) != np.shape(pa) or np.ndim(a) != 1
            or not len(a) or any(np.shape(v) != np.shape(a) for v in (b, penalty, cdf))
            or np.any(np.asarray(a) != np.asarray(a, dtype=np.int64))
            or np.any(np.asarray(b) != np.asarray(b, dtype=np.int64))
            or np.any(a < 0) or np.any(b < a) or np.any(b > len(pa))
            or np.any(~np.isfinite(pa)) or np.any(~np.isfinite(pp))
            or np.any((pa <= 0) | (pa >= 1) | (pp <= 0) | (pp >= 1))
            or np.any(np.isnan(penalty) | np.isposinf(penalty)) or np.isnan(threshold)
            or np.any(~np.isfinite(cdf)) or np.any(np.diff(cdf[:-1]) < 0)
            # The producer sets cdf[-1]=1 after cumsum; accumulated rounding
            # can put the penultimate entry a few ulps above one. Preserve
            # this exact legacy array/RNG behavior instead of renormalizing it.
            or np.any(cdf > 1.+8*np.finfo(float).eps*len(cdf))
            or cdf[0] < 0 or cdf[-1] != 1.):
        raise ValueError('Invalid native predictive experiment')
    with np.errstate(over='ignore', invalid='ignore'):
        hit = np.log(pp/pa)
        miss = np.log1p(-pp)-np.log1p(-pa)
    if np.any(~np.isfinite(hit)) or np.any(~np.isfinite(miss)):
        raise ValueError('Finite predictive log-likelihood increments required')


class PredictiveAccelerator:
    """One bounded, explicitly selected execution owner per workflow.

    device='cpu' with precision='float32' is a numerical emulation test, not
    an assertion that an Apple GPU was tested. MPS always uses float32 only
    for max/subtract operations, with a CPU float64 near-threshold fallback.
    """
    def __init__(self, device, maximum_bytes=256*1024**2, *, precision=None):
        import torch
        if device not in ('cpu', 'cuda', 'mps') or maximum_bytes < 1024**2:
            raise ValueError('Valid device and at least 1 MiB scratch required')
        self.torch = torch; self.device = device
        self.dtype = torch.float32 if (precision == 'float32' or device == 'mps') else torch.float64
        self.maximum_bytes = int(maximum_bytes)
        self.stats = Counter()
        self.fused = None
        if device == 'cuda' and self.dtype == torch.float64:
            try:
                from .cuda_predictive import profile_losses
                self.fused = profile_losses
            except ImportError:
                self.stats['triton_unavailable'] += 1

    def count(self, request):
        from .measurement_distribution import _predictive_exceedances
        validate_request(request)
        pa, pp, a, b, penalty, cdf, threshold, draws, seed = request
        self.stats['requests'] += 1
        # Keep bounded host-prefix and device scratch allocations. Larger
        # requests run on CPU in full; never truncate opportunities or draws.
        if (draws*(len(pa)+1)*8 > self.maximum_bytes//2
                or not np.isfinite(threshold) or not np.isfinite(penalty).any()):
            self.stats['cpu_budget_or_nonfinite_fallbacks'] += 1
            return int(_predictive_exceedances(*request))
        begun = time.perf_counter()
        prefixes = _reference_prefixes(pa, pp, a, b, cdf, draws, seed)
        self.stats['prefix_seconds'] += time.perf_counter()-begun
        if not np.isfinite(prefixes).all():
            self.stats['cpu_numeric_fallbacks'] += 1
            return int(_predictive_exceedances(*request))
        begun = time.perf_counter()
        try:
            count = self.score_prefixes(prefixes, a, b, penalty, threshold)
        except RuntimeError as exc:
            # Device absence/OOM/unsupported operations do not change a result.
            # The explicitly reported fallback replays this entire experiment.
            message = str(exc).lower()
            recoverable = (isinstance(exc, self.torch.OutOfMemoryError)
                or 'out of memory' in message
                or ('not implemented' in message and ('mps' in message or 'cuda' in message)))
            if not recoverable:
                # Illegal access and unknown kernel errors must fail visibly;
                # do not disguise a poisoned context as a successful CPU run.
                raise
            self.stats['cpu_device_fallbacks'] += 1
            if self.stats['cpu_device_fallbacks'] == 1:
                warnings.warn(f'{self.device} predictive scoring fell back to CPU: {exc}', RuntimeWarning)
            count = int(_predictive_exceedances(*request))
        self.stats['scoring_seconds'] += time.perf_counter()-begun
        return count

    def score_prefixes(self, prefixes, a, b, penalty, threshold):
        """No RNG here: useful for exact CPU/device numerical comparisons."""
        torch = self.torch
        if np.ndim(prefixes) != 2 or not np.isfinite(prefixes).all():
            raise ValueError('Finite two-dimensional CPU prefixes are required')
        eps = np.finfo(np.float32 if self.dtype == torch.float32 else np.float64).eps
        finite_penalty = np.asarray(penalty)[np.isfinite(penalty)]
        if not len(finite_penalty) or not np.isfinite(threshold):
            return int(_reference_events(prefixes, a, b, penalty, threshold).sum())
        magnitude = float(np.abs(finite_penalty).max())
        if self.dtype == torch.float32 and (np.abs(prefixes).max(initial=0.) > np.finfo(np.float32).max/8
                or magnitude > np.finfo(np.float32).max/8):
            self.stats['cpu_float32_range_fallbacks'] += 1
            return int(_reference_events(prefixes, a, b, penalty, threshold).sum())
        item = 4 if self.dtype == torch.float32 else 8
        # Inputs + both gathers + differences/penalized values + reductions.
        rows = max(1, min(2048, self.maximum_bytes//max(1, 8*item*(len(a)+prefixes.shape[1]))))
        columns = max(1, min(len(a), self.maximum_bytes//max(1, rows*item*8)))
        if self.fused is not None:
            rows = max(1, min(len(prefixes),
                (self.maximum_bytes-24*len(a))//max(1, (prefixes.shape[1]+8)*8)))
        count = 0
        with torch.inference_mode():
            starts = torch.as_tensor(np.array(a, dtype=np.int64, copy=True), device=self.device)
            ends = torch.as_tensor(np.array(b, dtype=np.int64, copy=True), device=self.device)
            penalties = torch.as_tensor(np.array(penalty, copy=True), device=self.device, dtype=self.dtype)
            for begin in range(0, len(prefixes), rows):
                original = prefixes[begin:begin+rows]
                x = torch.as_tensor(original, device=self.device, dtype=self.dtype)
                if self.fused is not None:
                    loss = self.fused(x, starts, ends, penalties).cpu().numpy()
                else:
                    best = torch.full((len(original),), -float('inf'), device=self.device, dtype=self.dtype)
                    explained = best.clone()
                    for j in range(0, len(a), columns):
                        values = x[:, ends[j:j+columns]]-x[:, starts[j:j+columns]]
                        best = torch.maximum(best, values.amax(dim=1))
                        explained = torch.maximum(explained, (values+penalties[j:j+columns]).amax(dim=1))
                    loss = (best-explained).cpu().numpy().astype(np.float64)
                target = threshold-1e-10
                # Prefix sums themselves were constructed on CPU. Only casts,
                # two subtractions and an addition can incur device error;
                # max is nonexpansive. This conservative absolute envelope
                # deliberately avoids claiming bitwise equality from CUDA/MPS.
                bound = 32*eps*(np.abs(original).max(axis=1)+magnitude+abs(target)+1.)
                uncertain = (~np.isfinite(loss)) | (np.abs(loss-target) <= bound)
                events = loss >= target
                if uncertain.any():
                    events[uncertain] = _reference_events(original[uncertain], a, b, penalty, threshold)
                self.stats['cpu_rechecked_draws'] += int(uncertain.sum())
                self.stats['device_draws'] += len(original)
                count += int(events.sum())
        return count

    def finish(self, record):
        from .measurement_distribution import predictive_count_record
        if '_native_predictive_request' not in record:
            return record
        result = dict(record); request = result.pop('_native_predictive_request')
        return predictive_count_record(result, self.count(request), request[-2])
