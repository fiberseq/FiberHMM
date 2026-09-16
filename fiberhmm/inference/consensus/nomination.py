# Numerical kernel promoted from the validated September 2026 consensus experiments.
from __future__ import annotations

import time
import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from .projection import bounded_projection
from .lattice import RegionFamilyLattice
from .clustering import gmm_centers, group_hash


def overlap_neighborhoods(calls, start, end, bandwidth):
    """Every source call has one nomination home; its full interval survives.

    Basin assignment uses the raw center only for work organization. Splitting
    a basin at true non-overlap gaps prevents a GMM from combining disconnected
    sites. A long source call is retained in full and can compete with families
    nominated in neighboring basins during whole-region inference.
    """
    if not calls:
        return []
    centers=np.clip(np.rint([(c['start']+c['end'])/2 for c in calls]).astype(int)-start,0,end-start-1)
    density=gaussian_filter1d(np.bincount(centers,minlength=end-start).astype(float),max(1,bandwidth))
    peaks=find_peaks(density,distance=max(1,2*bandwidth))[0]
    valleys=[int(a+np.argmin(density[a:b+1])) for a,b in zip(peaks,peaks[1:])]
    labels=np.searchsorted(valleys,centers,side='right')
    out=[]
    for label in np.unique(labels):
        ordered=sorted(np.flatnonzero(labels==label),key=lambda i:(calls[i]['start'],calls[i]['end'],calls[i]['observation_id']))
        component=[]; right=-1
        for i in ordered:
            c=calls[i]
            if component and c['start']>=right:
                out.append(component); component=[];right=-1
            component.append(int(i)); right=max(right,c['end'])
        if component:
            out.append(component)
    out.sort(key=lambda ids:(min(calls[i]['start'] for i in ids),max(calls[i]['end'] for i in ids)))
    if sorted(i for ids in out for i in ids)!=list(range(len(calls))):
        raise AssertionError('Nomination lost or duplicated a source call')
    return out


@njit(cache=True)
def _family_availability(offsets, spans, centers, x):
    out=np.zeros((len(offsets)-1,len(centers)),dtype=np.bool_)
    for m in range(len(out)):
        for j in range(offsets[m],offsets[m+1]):
            a,b=spans[j]
            for f in range(len(centers)):
                if centers[f,0]-x < b and centers[f,1]+x > a:
                    out[m,f]=True
    return out


def family_availability(units,centers,x):
    """Group existing native TFs; do not turn uncalled nucleosomes into TFs.

    Raw calls only admit candidates, never add a score or bypass native evidence.
    A family's +/-X envelope must overlap a source native TF on that unit.
    Every geometry of an admitted family is still integrated with its original
    normalized q. Priors are normalized on each unit's admissible state set.
    This is caller-conditioned grouping, not a de novo TF/recall probability.
    """
    # Same strict-overlap tests, in native code instead of one temporary vector
    # per source call. This does not cap/merge calls or alter admissibility.
    c=np.asarray(centers).reshape(-1,2)
    intervals=[u['representative_raw_tf_intervals'] for u in units]
    offsets=np.r_[0,np.cumsum([len(v) for v in intervals])]
    spans=np.asarray([v for group in intervals for v in group],dtype=np.int64).reshape(-1,2)
    return _family_availability(offsets,spans,c,x)


def fit_local_geometry(values, positions, centers, x, max_iter, units, args=None):
    kernel=RegionFamilyLattice(positions,centers,x,
        maximum_nodes=getattr(args,'maximum_nodes',100000), maximum_edges=getattr(args,'maximum_edges',2000000))
    allowed=family_availability(units,centers,x)
    fit,_=kernel.fit(values,allowed=allowed,max_iter=max_iter,starts=(-2.,),check=getattr(args,'check',None))
    return kernel,fit


def choose_predictive_complexity(candidates, selection_se=1., *, predictive_loss_tolerance=0., reference_units=None):
    """Prefer the fewest classes within a user-set *paired* fit-loss tolerance.

    This changes catalog granularity, not the emission model or a molecule's
    display threshold. Strongly supported subpopulations can still require any
    number of classes; there is no class-count ceiling.
    """
    if not np.isfinite(selection_se) or selection_se < 0:
        raise ValueError('selection_se must be finite and nonnegative')
    if not candidates:
        raise ValueError('At least one fitted complexity candidate is required')
    if not np.isfinite(predictive_loss_tolerance) or predictive_loss_tolerance < 0:
        raise ValueError('predictive_loss_tolerance must be finite and nonnegative')
    n=len(candidates[0]['prediction'])
    if n<2 or any(np.shape(c['prediction'])!=(n,) for c in candidates):
        raise ValueError('At least two identically indexed validation predictions required')
    reference_units=n if reference_units is None else reference_units
    if not isinstance(reference_units,(int,np.integer)) or not 0<=reference_units<=n:
        raise ValueError('reference_units must be a count inside the validation cohort')
    # Scores still include EVERY validation unit. A fixed, K-independent count
    # of units bearing an upstream TF prevents unrelated amplicon depth from
    # diluting the practical loss. Never normalize by selected family support.
    practical_mean_loss=predictive_loss_tolerance*reference_units/n
    best=max(candidates,key=lambda r:r['validation_mean'])
    eligible=[]
    for row in candidates:
        delta=best['prediction']-row['prediction']
        se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
        row['loss_vs_best']=float(delta.mean());row['paired_SE']=se
        if delta.mean()<=max(selection_se*se,practical_mean_loss)+1e-12:
            eligible.append(row)
    return best,min(eligible,key=lambda r:(r['families'],-r['validation_mean']))


def discover_neighborhood(data,calls,args,number):
    edges=np.array([[c['start'],c['end']] for c in calls],dtype=int)
    lo=max(args.start,int(edges[:,0].min())-args.ambiguity_bp)
    hi=min(args.end,int(edges[:,1].max())+args.ambiguity_bp)
    keep=(data['grid_positions']>=lo)&(data['grid_positions']<hi)
    positions=data['grid_positions'][keep]; values=data['log_lr'][:,keep]
    unit_obs=data['observed'][:,keep].any(1)
    split=np.array([group_hash(g,'napa-region-complexity-v1')%3 for g in data['fold_group_ids']])
    train=np.flatnonzero((split!=0)&unit_obs); val=np.flatnonzero((split==0)&unit_obs)
    train_units={data['unit_ids'][i] for i in train}
    # Fixed domain and reference cohort throughout this K search. The source
    # calls were frozen by SR; selected assignments and display Q cannot change
    # the normalization. No extra pseudoevidence is added to the likelihood.
    reference_units=sum(any(a<hi and b>lo for a,b in data['units'][i]['representative_raw_tf_intervals']) for i in val)
    training_edges=np.array([[c['start'],c['end']] for c in calls if c['unit_id'] in train_units],dtype=int).reshape(-1,2)
    ledger={'neighborhood':number,'domain':[lo,hi],'source_call_count':len(calls),
            'source_unit_count':len({c['unit_id'] for c in calls}),
            'raw_span_range':[int(edges[:,0].min()),int(edges[:,1].max())],
            'training_observed_units':len(train),'validation_observed_units':len(val),
            'validation_call_bearing_units':int(reference_units),
            'practical_loss_status':'available' if reference_units else 'unavailable_no_call_bearing_validation_units',
            'all_source_observation_ids':[c['observation_id'] for c in calls],
            'fixed_family_ceiling':None,'candidates':[],'failures':[]}
    if not len(positions):
        return [],ledger|{'status':'no_observed_opportunities','all_calls_retained_in_ledger':True}
    # A singleton/selection-unavailable neighborhood remains a proposal. It is
    # not thrown away for failing a recurrence gate.
    if len(train)<2 or len(val)<2 or len(training_edges)<2:
        centers=np.rint(np.mean(edges,axis=0)).astype(int)[None]
        ledger.update(status='exploratory_selection_unavailable',selected_families=1)
        return centers.tolist(),ledger
    maximum=len(np.unique(training_edges,axis=0))
    candidates=[]; begun=time.monotonic(); k=1
    while k<=maximum:
        if getattr(args,'check',None): args.check()
        if time.monotonic()-begun>args.neighborhood_seconds:
            raise TimeoutError(f'Neighborhood {number} adaptive search exceeded {args.neighborhood_seconds}s; no silent family-count truncation')
        try:
            centers=gmm_centers(training_edges,k,getattr(args,'seed',20260905)+k,
                regularization=getattr(args,'gmm_regularization',1.),restarts=getattr(args,'gmm_restarts',4),
                max_iter=getattr(args,'gmm_iterations',400))
            kernel,fit=fit_local_geometry(values[train],positions,centers,args.ambiguity_bp,args.local_iterations,
                                         [data['units'][i] for i in train],args)
            available=family_availability([data['units'][i] for i in val],centers,args.ambiguity_bp)
            uq,_,inv=kernel.prior_recipe(available,len(val))
            z0=kernel.evaluate(np.zeros((len(uq),len(positions))),fit['eta'],allowed=uq)['log_partition'][inv]
            prediction=kernel.evaluate(values[val],fit['eta'],allowed=available)['log_partition']-z0
            row={'families':len(centers),'centers':centers.tolist(),'validation_mean':float(prediction.mean()),
                 'fit_success':fit['success'],'projected_gradient':fit['max_projected_gradient'],
                 'iterations':fit['iterations'],'prediction':prediction}
            candidates.append(row)
        except (ValueError,RuntimeError,FloatingPointError) as exc:
            ledger['failures'].append({'K':k,'error':str(exc)})
        if candidates:
            best,selected=choose_predictive_complexity(candidates,getattr(args,'selection_se',1.),
                predictive_loss_tolerance=getattr(args,'predictive_loss_tolerance',0.),reference_units=reference_units)
            # An adaptive stopping rule, NOT a maximum allowed K. Search at
            # least K=1..4 (when possible), and expand whenever the best model
            # sits at the search frontier. The full search history is saved.
            if k>=4 and best['families']<k and selected['families']<k:
                break
        k+=1
    if not candidates:
        raise RuntimeError(f'No valid local native model in neighborhood {number}')
    ledger['candidates']=[{key:v for key,v in c.items() if key!='prediction'} for c in candidates]
    ledger.update(status='exploratory_selected',selected_families=selected['families'],
                  adaptive_stopping='first K>=4 beyond best and tolerance-selected K, or all unique training geometries tested',
                  stopping_K=k,selection_se=float(getattr(args,'selection_se',1.)),
                  predictive_loss_tolerance=float(getattr(args,'predictive_loss_tolerance',0.)),
                  selection='fewest families within max(selection_se times paired SE, practical loss allowance) of best internal validation score')
    # ALL calls now nominate final centers. No unit is excluded from final
    # discovery/fitting by the internal complexity-selection split.
    full_centers=gmm_centers(edges,selected['families'],getattr(args,'seed',20260905)+selected['families'],
        regularization=getattr(args,'gmm_regularization',1.),restarts=getattr(args,'gmm_restarts',4),
        max_iter=getattr(args,'gmm_iterations',400))
    alternatives=[full_centers,np.asarray(selected['centers'])]
    fits=[]
    for c in alternatives:
        ker,fit=fit_local_geometry(values[unit_obs],positions,c,args.ambiguity_bp,args.local_iterations,
                                  [data['units'][i] for i in np.flatnonzero(unit_obs)],args)
        fits.append((fit['objective'],c,fit))
    _,centers,fit=min(fits,key=lambda item:item[0])
    ledger['full_data_centers']=centers.tolist();ledger['all_unit_fit']={k:v for k,v in fit.items() if k!='eta'}
    ledger['full_data_nomination_centers']=full_centers.tolist()
    return centers.tolist(),ledger


_NEIGHBORHOOD_WORKER = {}
from .execution import register_worker_state as _register_worker_state
_register_worker_state(_NEIGHBORHOOD_WORKER)


def _load_neighborhood_worker(path):
    import pickle
    from types import SimpleNamespace
    with open(path, 'rb') as handle:
        state = pickle.load(handle)
    data = state['data']
    data['log_lr'] = np.load(state['log_lr_path'], mmap_mode='r', allow_pickle=False)
    data['observed'] = np.load(state['observed_path'], mmap_mode='r', allow_pickle=False)
    state['args'] = SimpleNamespace(**state['args'], check=None)
    return state


def _neighborhood_task(path, threads, number, ids):
    from .execution import task_thread_budget, load_worker_state
    task_thread_budget(threads)
    state = load_worker_state(_NEIGHBORHOOD_WORKER, path, _load_neighborhood_worker)
    calls = [state['calls'][i] for i in ids]
    local, record = discover_neighborhood(state['data'], calls, state['args'], number)
    return number, local, record


def _thread_schedule(sizes, cores, threads_per_worker, limit):
    """Threads per neighborhood, largest first: the few dominant neighborhoods
    take up to ``limit`` threads while the rest keep the declared budget. The
    lattice evaluation is per-unit independent, so a thread count changes
    scheduling only, never values; the caller caps concurrency so the sum of
    running threads never exceeds the core budget."""
    order = sorted(range(len(sizes)), key=lambda n: -sizes[n])
    threads = {}
    big = max(1, int(cores)//max(1, int(limit)))
    for rank, n in enumerate(order):
        threads[n] = int(limit) if rank < big and sizes[n] > 1 else max(1, int(threads_per_worker))
    return order, threads


def discover_neighborhoods_in_processes(data, calls, neighborhoods, args, workers, note, threads_per_worker=1,
                                        cores=None, large_threads=None):
    """Independent neighborhoods in single-threaded spawned workers, in order.

    Every neighborhood runs the unchanged adaptive search on the same matrices
    (memory-mapped read-only), seeds and budgets. Only scheduling changes.
    """
    import pickle, tempfile
    from concurrent.futures import FIRST_COMPLETED, wait
    from pathlib import Path
    from .execution import stage_executor, WORKER_NUMBA_THREAD_LIMIT
    threads = max(1, min(int(threads_per_worker), WORKER_NUMBA_THREAD_LIMIT))
    budget = int(cores) if cores else int(workers)*threads
    limit = max(threads, min(int(large_threads or threads), WORKER_NUMBA_THREAD_LIMIT))
    order, per_task = _thread_schedule([len(n) for n in neighborhoods], budget, threads, limit)
    results = [None]*len(neighborhoods)
    with tempfile.TemporaryDirectory(prefix='fiberhmm-native-nomination-') as directory:
        root = Path(directory)
        np.save(root/'log_lr.npy', np.ascontiguousarray(data['log_lr']), allow_pickle=False)
        np.save(root/'observed.npy', np.ascontiguousarray(data['observed']), allow_pickle=False)
        slim = dict(grid_positions=np.asarray(data['grid_positions']), unit_ids=list(data['unit_ids']),
                    fold_group_ids=list(data['fold_group_ids']),
                    units=[dict(representative_raw_tf_intervals=u['representative_raw_tf_intervals'])
                           for u in data['units']])
        payload = dict(data=slim, calls=calls, log_lr_path=str(root/'log_lr.npy'),
                       observed_path=str(root/'observed.npy'),
                       args={k: v for k, v in vars(args).items() if k != 'check'})
        with open(root/'payload.pkl', 'wb') as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # The exact row-parallel lattice evaluation is per-unit independent, so
        # a worker's numba thread count changes scheduling only, never values.
        # Each task sets its own budget; in-flight tasks are capped at the
        # stage's worker count so threads x tasks never exceeds the core budget.
        path = str(root/'payload.pkl')
        executor, release = stage_executor(min(workers, len(neighborhoods)))
        # Heaviest neighborhoods first shorten the tail; output order is by number.
        # Running threads are capped at the core budget: a task is submitted only
        # when its thread count fits beside the tasks already in flight.
        pending = {}; queue = list(order); running = {}; completed = 0
        def submit():
            while queue:
                number = queue[0]; need = per_task[number]
                if running and sum(running.values())+need > budget:
                    return
                queue.pop(0); running[number] = need
                pending[executor.submit(_neighborhood_task, path, need, number, list(neighborhoods[number]))] = number
        try:
            submit()
            note(0)
            while pending:
                done, _ = wait(pending, timeout=.2, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    number, local, record = future.result()
                    running.pop(number, None)
                    results[number] = (local, record); completed += 1
                submit()
                note(completed)
        except BaseException:
            for future in pending:
                future.cancel()
            release(failed=True)
            raise
        release()
    if any(r is None for r in results):
        raise RuntimeError('Neighborhood nomination finished without every result')
    return results


def deduplicate(centers,aliases,positions,x):
    # Only EXACT full-cohort opportunity-kernel equality merges proposals.
    # Rounded coordinate coincidence alone is not sufficient when q differs.
    seen={};out=[];links=[];unavailable=[]
    for center,alias in zip(centers,aliases):
        try:
            projection=bounded_projection(positions,center,x)
        except ValueError as exc:
            unavailable.append({'center':center,'alias':alias,'reason':str(exc)});continue
        key=(projection['starts'].tobytes(),projection['ends'].tobytes(),projection['q'].tobytes())
        if key not in seen:
            seen[key]=len(out);out.append(center);links.append([])
        links[seen[key]].append(alias)
    order=sorted(range(len(out)),key=lambda i:tuple(out[i]))
    return np.asarray([out[i] for i in order]),[links[i] for i in order],unavailable


def export_posteriors(kernel,data,eta,rows,allowed):
    n=len(data['unit_ids']);ng=len(kernel.ga)
    inclusion=np.zeros((n,kernel.f));total=np.zeros(ng);selected=np.zeros((len(rows),ng))
    relative=np.zeros(n);row_lookup={int(r):j for j,r in enumerate(rows)}
    uq,_,inv=kernel.prior_recipe(allowed,n)
    prior=kernel.evaluate(np.zeros((len(uq),kernel.k)),eta,allowed=uq)
    prior_log_partition=prior['log_partition'][inv]
    # Explicit scalar evidence budget for batch geometry marginals; observations
    # and whole-locus prefix sums are reused, never subsampled.
    batch=max(1,min(64,int(128*1024**2/max(8*ng,1))))
    for begin in range(0,n,batch):
        end=min(n,begin+batch);out=kernel.evaluate(data['log_lr'][begin:end],eta,allowed=allowed[begin:end],export_geometry=True)
        inclusion[begin:end]=out['family_inclusion'];total+=out['geometry_mass'].sum(0)
        relative[begin:end]=out['log_partition']-prior_log_partition[begin:end]
        for r in range(begin,end):
            if r in row_lookup:selected[row_lookup[r]]=out['geometry_mass'][r-begin]
    return {'family_inclusion':inclusion,'total_geometry_mass':total,'display_geometry_mass':selected,
            'log_predictive_relative_accessible':relative,'prior_family_inclusion':prior['family_inclusion'][inv]}
