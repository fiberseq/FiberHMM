from fiberhmm.inference.consensus.numerics import summarize_fits,nomination_fit_status
from fiberhmm.inference.consensus.parameters import CROptions


def test_default_budget_covers_observed_full_locus_convergence():
    assert CROptions().global_iterations>=800


def test_summary_describes_the_selected_fit_not_any_successful_start():
    fits=[dict(objective=-2.,success=True),dict(objective=-3.,success=False,iterations=160,
        max_projected_gradient=.01,message='Iteration limit reached')]
    result=summarize_fits(fits)
    assert result['selected_start_index']==1
    assert result['selected_converged'] is False
    assert result['status']=='numerically_provisional'
    assert result['converged_starts']==1
    assert result['iterations']==160
    assert fits[1]['success'] is False


def test_legacy_unknown_not_silently_converged_and_converged_not_calibrated():
    assert summarize_fits([])['selected_converged'] is None
    result=summarize_fits([dict(objective=1.,success=True,iterations=300)])
    assert result['selected_converged'] is True
    assert 'not empirical calibration' in result['semantics']


def test_nomination_status_separates_unselected_candidates():
    ledger=[dict(selected_families=1,candidates=[dict(families=1,fit_success=True),dict(families=2,fit_success=False)],
        all_unit_fit=dict(success=True)),dict(status='exploratory_selection_unavailable')]
    result=nomination_fit_status(ledger)
    assert result['unconverged_candidate_fits']==1
    assert result['unconverged_selected_candidate_fits']==0
    assert result['unconverged_final_local_fits']==0
    assert result['selection_unavailable_neighborhoods']==1
