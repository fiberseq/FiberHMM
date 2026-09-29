import json
import os
from pathlib import Path

import numpy as np
import pytest

from fiberhmm.inference.consensus.accelerated_predictive import (
    PredictiveAccelerator, _reference_prefixes, _reference_events, resolve_device)
from fiberhmm.inference.consensus.measurement_distribution import _predictive_exceedances
from fiberhmm.inference.consensus.native_catalog_update import _canonical, _encoded
from fiberhmm.inference.consensus.artifacts import read_json, write_json


def request(k=9, draws=511, threshold=1.3, seed=77, holes=False):
    rng = np.random.default_rng(98)
    a, b = np.triu_indices(k+1, 0)
    if holes:
        keep = rng.random(len(a)) > .4; a, b = a[keep], b[keep]
    penalty = -rng.exponential(4., len(a)); penalty[0] = 0.; penalty[1::5] = -np.inf
    q = rng.random(len(a)); cdf = np.cumsum(q/q.sum()); cdf[-1] = 1.
    return (rng.uniform(.15,.99,k), rng.uniform(.001,.14,k), a, b,
            penalty, cdf, threshold, draws, seed)


@pytest.mark.parametrize('k', [1, 3, 9, 17, 35])
def test_prefix_rng_and_event_replay_are_exact(k):
    for holes in (False, True):
        args = request(k, holes=holes)
        pa, pp, a, b, penalty, cdf, threshold, draws, seed = args
        prefix = _reference_prefixes(pa, pp, a, b, cdf, draws, seed)
        assert _reference_events(prefix, a, b, penalty, threshold).sum() == _predictive_exceedances(*args)


@pytest.mark.parametrize('precision', ['float64', 'float32'])
def test_torch_cpu_emulates_precision_policy_not_mps_hardware(precision):
    pytest.importorskip('torch')
    accelerator = PredictiveAccelerator('cpu', 16*1024**2, precision=precision)
    for k in (1, 5, 17):
        for holes in (False, True):
            args = request(k, draws=127, holes=holes)
            assert accelerator.count(args) == _predictive_exceedances(*args)


def test_float32_threshold_ties_are_rechecked_in_float64():
    pytest.importorskip('torch')
    accelerator = PredictiveAccelerator('cpu', precision='float32')
    args = request(3, draws=63)
    pa, pp, a, b, penalty, cdf, _, draws, seed = args
    prefix = _reference_prefixes(pa, pp, a, b, cdf, draws, seed)
    for row in prefix[:10]:
        values = row[b]-row[a]
        loss = float(values.max()-(values+penalty).max())
        for delta in (-2e-10, 0., 2e-10):
            tested = (*args[:6], loss+delta, draws, seed)
            assert accelerator.count(tested) == _predictive_exceedances(*tested)
    assert accelerator.stats['cpu_rechecked_draws'] > 0


@pytest.mark.parametrize('backend', ['cuda', 'mps'])
def test_real_device_matches_cpu_when_available(backend):
    if os.environ.get('FIBERHMM_TEST_ACCELERATORS') != '1':
        pytest.skip('Real GPU tests are opt-in; do not take a shared GPU during ordinary CPU tests')
    pytest.importorskip('torch')
    if resolve_device(backend)[0] != backend:
        pytest.skip(f'{backend} hardware unavailable; not a performance validation')
    accelerator = PredictiveAccelerator(backend, 32*1024**2)
    for k in (3, 9, 25):
        for holes in (False, True):
            args = request(k, draws=1023, holes=holes)
            assert accelerator.count(args) == _predictive_exceedances(*args)
    assert accelerator.stats['device_draws'] > 0
    assert not accelerator.stats['cpu_device_fallbacks']


def test_budget_fallback_keeps_all_draws():
    pytest.importorskip('torch')
    accelerator = PredictiveAccelerator('cpu', 1024**2)
    args = request(35, draws=4095)
    assert accelerator.count(args) == _predictive_exceedances(*args)
    assert accelerator.stats['cpu_budget_or_nonfinite_fallbacks'] == 1


def test_cdf_final_roundoff_is_preserved_not_repaired():
    pytest.importorskip('torch')
    args = list(request(3))
    args[5][-2] = 1.+4*np.finfo(float).eps
    before = args[5].copy()
    assert PredictiveAccelerator('cpu').count(tuple(args)) == _predictive_exceedances(*args)
    np.testing.assert_array_equal(args[5], before)


def test_memo_is_request_local_across_seed_penalty_and_threshold():
    args = list(request(5, draws=4095))
    expected = []
    for i in range(3):
        args[6] = .2+i*3
        args[4] = args[4]-.7*i
        pa, pp, a, b, penalty, cdf, threshold, draws, seed = args
        prefix = _reference_prefixes(pa, pp, a, b, cdf, draws, seed)
        count = int(_reference_events(prefix, a, b, penalty, threshold).sum())
        assert _predictive_exceedances(*args) == count
        expected.append(count)
    assert len(set(expected)) > 1


def test_nonfinite_prefix_and_float32_overflow_do_not_hide_in_device_max():
    pytest.importorskip('torch')
    device = PredictiveAccelerator('cpu', precision='float32')
    a=np.array([0,0]); b=np.array([1,2]); penalty=np.array([0.,-np.inf])
    with pytest.raises(ValueError):
        device.score_prefixes(np.array([[0.,1.,np.nan]]),a,b,penalty,2.)
    prefix=np.array([[0.,1e100,1e100+1e90]])
    assert device.score_prefixes(prefix,a,b,penalty,2.) == _reference_events(prefix,a,b,penalty,2.).sum()
    assert device.stats['cpu_float32_range_fallbacks'] == 1


def test_unknown_device_error_is_fatal_but_oom_replays_cpu(monkeypatch):
    torch=pytest.importorskip('torch')
    device=PredictiveAccelerator('cpu'); args=request()
    def illegal(*args):raise RuntimeError('CUDA illegal memory access')
    monkeypatch.setattr(device,'score_prefixes',illegal)
    with pytest.raises(RuntimeError,match='illegal memory access'):device.count(args)
    def oom(*args):raise torch.OutOfMemoryError('injected out of memory')
    monkeypatch.setattr(device,'score_prefixes',oom)
    with pytest.warns(RuntimeWarning,match='fell back to CPU'):
        assert device.count(args)==_predictive_exceedances(*args)
    assert device.stats['cpu_device_fallbacks']==1


def test_all_negative_non_block_multiple_reductions():
    pytest.importorskip('torch')
    device=PredictiveAccelerator('cpu',precision='float32')
    prefixes=np.array([[0.,-1.,-3.,-6.],[0.,-5.,-9.,-12.]])
    a=np.array([0,1,2]);b=a+1;penalty=np.array([-3.,0.,-2.])
    for t in (0.,1.,2.,3.):
        assert device.score_prefixes(prefixes,a,b,penalty,t)==_reference_events(prefixes,a,b,penalty,t).sum()


def test_cuda_fit_validation_cannot_replace_cpu_models(monkeypatch):
    from fiberhmm.inference.consensus import accelerated_predictive as ap
    from fiberhmm.inference.consensus import measurement_distribution as md
    from fiberhmm.inference.consensus.fit_execution import NativeFitPool
    original=md.fit_native_distribution
    x=np.array([[0.,3.],[0.,4.],[1.,4.]])
    ll=np.array([[1.,2.,3.],[2.,1.,3.],[1.,3.,2.]])
    allowed=np.ones_like(ll,dtype=bool);area=np.ones(3)
    kwargs=dict(row_sets={'full':None},reference=np.array([0.,3.]),max_iterations=5)
    expected=NativeFitPool(1,32*1024**2).fit(ll,allowed,x,area,**kwargs)
    monkeypatch.setattr(ap,'resolve_device',lambda requested:('cpu' if requested=='cpu' else 'cuda',None))
    def divergent(*args,objective_backend='cpu',**options):
        value=original(*args,**options)
        if objective_backend=='cuda':
            value['log_mass']=np.array([0.,-np.inf,-np.inf])
            value['center']=np.array([999.,1000.])
        return value
    monkeypatch.setattr(md,'fit_native_distribution',divergent)
    pool=NativeFitPool(1,32*1024**2,backend='cuda')
    actual=pool.fit(ll,allowed,x,area,**kwargs)
    for key in expected['full']:
        np.testing.assert_equal(actual['full'][key],expected['full'][key])
    assert pool.execution['cuda_validation_only']
    assert pool.execution['cuda_distribution_disagreements']==1


def test_device_controls_are_explicit_and_cpu_default():
    from fiberhmm.inference.consensus.parameters import parse_options,parameter_schema
    # Device controls belong to the historical engines (the library default before 3.0); the recaller rejects them.
    defaults=parse_options({'cr':{'engine':'call_harmonization'}})['compute']
    assert defaults.predictive_backend==defaults.fit_backend=='cpu'
    controls={c['name']:c for c in parameter_schema()['compute']}
    assert controls['predictive_backend']['choices']==['cpu','vectorized','cuda','mps','auto']   # vectorized: declared versioned kernel (v11)
    assert 'validation-only' in controls['fit_backend']['help']
    for backend in ('cpu','cuda','mps','auto'):
        assert parse_options({'cr':{'engine':'call_harmonization'},'compute':{'predictive_backend':backend}})['compute'].predictive_backend==backend
    with pytest.raises(ValueError,match='not used by the lattice recaller'):parse_options({'compute':{'predictive_backend':'auto'}})


@pytest.mark.parametrize('bad', ['nan', 'cdf', 'zero_draws'])
def test_invalid_gpu_request_is_not_silently_accepted(bad):
    pytest.importorskip('torch')
    args = list(request())
    if bad == 'nan': args[0][0] = np.nan
    if bad == 'cdf': args[5][-1] = .9
    if bad == 'zero_draws': args[-2] = 0
    with pytest.raises(ValueError):
        PredictiveAccelerator('cpu').count(tuple(args))


def test_provenance_c_encoder_retains_canonical_bytes():
    samples = [dict(z=np.array([1., -0., 1e100]), a=[np.int32(3), np.bool_(True)]),
               [{'b':(None,True,2), 'a':np.array([[1,2],[3,4]])}], 'é', 5.]
    for sample in samples:
        expected = json.dumps(_canonical(sample), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        assert _encoded(sample) == expected
    for bad in ({1:'invalid'}, {'a':[{2:'invalid'}]}, np.array([float('nan')]), Path('/tmp')):
        with pytest.raises(ValueError):
            _encoded(bad)


@pytest.mark.parametrize('suffix', ['.json', '.json.gz'])
def test_fast_atomic_artifacts_preserve_values_and_failed_write(suffix, tmp_path):
    path = tmp_path/('test'+suffix)
    data = {'a':list(range(100)), 'float':[.1,-0.,1e100], 'unicode':'é'}
    write_json(path, data)
    assert read_json(path) == data
    before = path.read_bytes()
    with pytest.raises(ValueError):
        write_json(path, {'a':float('nan')})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
