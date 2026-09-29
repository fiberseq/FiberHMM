import numpy as np
import pytest
from fiberhmm.inference.consensus.nomination import choose_predictive_complexity, overlap_neighborhoods


def candidate(k, prediction):
    prediction=np.asarray(prediction,dtype=float)
    return dict(families=k,prediction=prediction,validation_mean=float(prediction.mean()))


def test_more_generous_tolerance_selects_weakly_worse_simpler_grouping():
    # The two-class advantage is noisy across units: mean .1, paired SE .0447.
    rows=[candidate(1,np.zeros(20)),candidate(2,[.1-.2]*10+[.1+.2]*10)]
    assert choose_predictive_complexity(rows,1)[1]['families']==2
    assert choose_predictive_complexity(rows,3)[1]['families']==1


def test_strong_reproducible_separation_survives_generous_tolerance():
    rows=[candidate(1,np.zeros(100)),candidate(7,[1-.02]*50+[1+.02]*50)]
    assert choose_predictive_complexity(rows,10)[1]['families']==7


def test_paired_loss_is_invariant_to_common_likelihood_baseline():
    rng=np.random.default_rng(22);baseline=rng.normal(0,50,100)
    delta=rng.normal(.1,.5,100)
    a=[candidate(1,np.zeros(100)),candidate(2,delta)]
    b=[candidate(1,baseline),candidate(2,baseline+delta)]
    assert choose_predictive_complexity(a,3)[1]['families']==choose_predictive_complexity(b,3)[1]['families']
    np.testing.assert_allclose(a[0]['paired_SE'],b[0]['paired_SE'],rtol=1e-12)


def test_granularity_is_monotone_on_one_frozen_candidate_set():
    rng=np.random.default_rng(19)
    rows=[candidate(k,rng.normal(k*.02,.4,200)) for k in range(1,13)]
    chosen=[choose_predictive_complexity(rows,se)[1]['families'] for se in [0,1,2,3,4,10]]
    assert chosen==sorted(chosen,reverse=True)


@pytest.mark.parametrize('bad',[-1,np.nan,np.inf])
def test_invalid_tolerance_rejected(bad):
    with pytest.raises(ValueError):choose_predictive_complexity([candidate(1,[0,0])],bad)


def test_broader_nomination_bandwidth_keeps_nonoverlapping_sites_separate():
    calls=[dict(start=10,end=20,observation_id='a'),dict(start=15,end=25,observation_id='b'),
           dict(start=35,end=45,observation_id='c')]
    groups=overlap_neighborhoods(calls,0,100,30)
    assert sorted(i for ids in groups for i in ids)==list(range(3))
    assert all(not (0 in ids and 2 in ids) for ids in groups)


def test_practical_allowance_does_not_vanish_with_depth():
    # Replication is an algebraic sensitivity test, not independent data.
    loss=np.array([-.15]*50+[.25]*50)
    for repeats in (1,10,100):
        delta=np.tile(loss,repeats)
        rows=[candidate(1,np.zeros(len(delta))),candidate(2,delta)]
        assert choose_predictive_complexity(rows,1,predictive_loss_tolerance=.06)[1]['families']==1
    assert choose_predictive_complexity(rows,1)[1]['families']==2


def test_empty_exposure_units_do_not_dilute_practical_loss():
    for padding in (0,100,1000):
        delta=np.r_[np.full(20,.3),np.zeros(padding)]
        rows=[candidate(1,np.zeros(len(delta))),candidate(2,delta)]
        assert choose_predictive_complexity(rows,0,predictive_loss_tolerance=.2,reference_units=20)[1]['families']==2
        assert choose_predictive_complexity(rows,0,predictive_loss_tolerance=.4,reference_units=20)[1]['families']==1


def test_meaningful_separation_survives_the_practical_floor():
    rows=[candidate(1,np.zeros(1000)),candidate(7,np.ones(1000))]
    assert choose_predictive_complexity(rows,3,predictive_loss_tolerance=.1)[1]['families']==7


def test_practical_allowance_is_method_independent_and_opt_in():
    from fiberhmm.inference.consensus.parameters import parse_options
    # Historical-engine control (call_harmonization was the library default before 3.0); request it explicitly.
    engine={'engine':'call_harmonization'}
    assert parse_options({'cr':engine})['cr'].predictive_loss_tolerance==0
    assert parse_options({'cr':{**engine,'predictive_loss_tolerance':.2}})['cr'].predictive_loss_tolerance==.2
    with pytest.raises(ValueError):parse_options({'cr':{**engine,'predictive_loss_tolerance':-1}})


def test_full_workflow_keeps_disconnected_sites_even_at_coarse_setting(tmp_path):
    from test_consensus_workflow import payload
    from fiberhmm.inference.consensus.workflow import run_workflow
    result=run_workflow(payload(),{'input':{'correct_native':False},'sr':{'enabled':False},
        'cr':{'engine':'legacy_lattice','ambiguity_bp':2,'neighborhood_bandwidth':30,'predictive_loss_tolerance':5.,
              'local_iterations':20,'global_iterations':100}},tmp_path)
    for ds in result['datasets'].values():
        assert len(ds['cr']['catalog'])==2
        assert sum(len(r['proposals']) for r in ds['cr']['records'])==24
