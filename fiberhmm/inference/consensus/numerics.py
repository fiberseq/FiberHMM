"""Numerical status is explicit and separate from scientific support or Q."""
from __future__ import annotations
import math


def summarize_fits(fits):
    """Describe the same minimum-objective start used by the native solver.

    Completing a job is not proof that an optimizer converged. Do not silently
    delete exploratory hypotheses at a low user-selected iteration budget, but
    do not present their numerical status as established either. Legacy results
    without a recorded fit remain unreported, not assumed converged.
    """
    usable=[(i,f) for i,f in enumerate(fits or [])
        if isinstance(f.get('objective'),(int,float)) and math.isfinite(f['objective'])]
    if not usable:return dict(status='unreported',selected_converged=None,starts=len(fits or []))
    i,selected=min(usable,key=lambda v:v[1]['objective'])
    converged=bool(selected.get('success',False))
    return dict(status='converged' if converged else 'numerically_provisional',
        selected_converged=converged,selected_start_index=i,starts=len(fits),
        converged_starts=sum(bool(f.get('success',False)) for f in fits),
        all_starts_converged=all(bool(f.get('success',False)) for f in fits),
        objective=selected['objective'],iterations=selected.get('iterations'),
        max_projected_gradient=selected.get('max_projected_gradient'),
        optimizer_message=selected.get('message'),
        semantics='Optimizer termination only; not empirical calibration or a class-support score')


def nomination_fit_status(ledger):
    candidates=[c for r in ledger for c in r.get('candidates',[])]
    selected=[c for r in ledger for c in r.get('candidates',[]) if c.get('families')==r.get('selected_families')]
    final=[r['all_unit_fit'] for r in ledger if r.get('all_unit_fit')]
    return dict(candidate_fits=len(candidates),unconverged_candidate_fits=sum(not c['fit_success'] for c in candidates),
        selected_candidate_fits=len(selected),unconverged_selected_candidate_fits=sum(not c['fit_success'] for c in selected),
        final_local_fits=len(final),unconverged_final_local_fits=sum(not f['success'] for f in final),
        selection_unavailable_neighborhoods=sum(r.get('status')=='exploratory_selection_unavailable' for r in ledger))
