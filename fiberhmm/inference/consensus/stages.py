"""Boundary SR and full-region caller-conditioned population CR."""
from __future__ import annotations
from types import SimpleNamespace
import numpy as np
from .. import strand_boundary_normalization as sbn
from .adapter import read_adapter
from .artifacts import digest
from .geometry import representative_geometries, physically_allowed, overlap_mask
from .lattice import RegionFamilyLattice
from .nomination import overlap_neighborhoods, discover_neighborhood, deduplicate, family_availability
from .observations import prepare_population
from .numerics import summarize_fits, nomination_fit_status
from .progress import report


def boundary_sr(stratum, options, progress, cores=1):
    units = stratum['units']
    report(progress,'sr','Preparing strand-boundary source catalogs',dataset_id=stratum['dataset_id'])
    if stratum['chemistry'] not in ('ddda', 'dddb'):
        return dict(status='not_applicable', reason='Chemistry has no separate CT/GA evidence strata', records=[])
    cohorts = {s: [read_adapter(u) for u in units if u['strand'] == s] for s in ('CT', 'GA')}
    catalogs, omitted = {}, {}
    for s in cohorts:
        catalogs[s], omitted[s] = sbn.build_source_catalog(cohorts[s])
    records = []
    completed=0
    total=sum(len(reads) for reads in cohorts.values())
    decoding = dict(loss_odds=options.loss_odds, minimum_source_support=options.minimum_source_support,
                    minimum_projection_mass=options.minimum_projection_mass, projection_only=options.projection_only,
                    maximum_diffuse_odds=options.maximum_diffuse_odds)
    workers = min(int(cores), 32)
    if isinstance(cores, bool) or not isinstance(cores, (int, np.integer)) or cores < 1:
        raise ValueError('Positive integer SR worker budget required')
    for strand, reads in cohorts.items():
        source = 'GA' if strand == 'CT' else 'CT'
        if workers > 1 and len(reads) >= 2*workers:
            # Reads are independent given the frozen opposite-strand catalog.
            # Single-threaded spawned workers return identical records in the
            # original read order; no read, candidate or decision changes.
            def note(done):
                report(progress,'sr',f'{strand}: normalizing {done}/{len(reads)} units ({workers} workers)',
                    dataset_id=stratum['dataset_id'],completed=completed+done,total=total,unit='evidence units')
            records.extend(_normalize_reads_in_processes(reads, catalogs[source], strand, source,
                                                         decoding, workers, note))
        else:
            normalizer = sbn.BoundaryNormalizer(catalogs[source])
            for i, read in enumerate(reads):
                if i % 32 == 0:
                    report(progress,'sr',f'{strand}: normalizing {i}/{len(reads)} units',
                        dataset_id=stratum['dataset_id'],completed=completed+i,total=total,unit='evidence units')
                records.append(_normalize_read(normalizer, read, strand, source, decoding))
        completed+=len(reads)
    report(progress,'sr','Strand-boundary normalization complete',dataset_id=stratum['dataset_id'],
        completed=completed,total=total,unit='evidence units')
    by_id = {r['unit_id']: r for r in records}
    changed = 0
    for u in units:
        old = u['representative_raw_tf_intervals']
        row = by_id.get(u['unit_id'])
        normalized = [list(v) for v in old]
        if row:
            indices = [i for i, (a,b) in enumerate(old) if a < u['_region'][1] and b > u['_region'][0]]
            if len(indices) != len(row['calls']):
                raise AssertionError('Boundary SR changed call cardinality')
            for index, call in zip(indices, row['calls']):
                if list(old[index]) != call['raw']:
                    raise AssertionError('Boundary SR lost source ordinal lineage')
                normalized[index] = call['interval']
                changed += int(normalized[index] != list(old[index]))
        u['sr_boundary_only_tf_intervals'] = normalized
        u['sr_tf_intervals'] = normalized
        u['representative_raw_tf_intervals'] = normalized
    return dict(status='complete', records=records, changed=changed,
        source_cells={s:len(v) for s,v in catalogs.items()}, source_omissions=omitted,
        new_calls=0, missing_strands=[s for s,v in cohorts.items() if not v])


def _normalize_read(normalizer, read, strand, source, decoding):
    """Exactly the historical per-read decision sequence."""
    decoded = [normalizer.decode_call(normalizer.score_call(read, j),
        loss_budget=np.log(decoding['loss_odds']), minimum_source_support=decoding['minimum_source_support'],
        minimum_projection_mass=decoding['minimum_projection_mass'], projection_only=decoding['projection_only'],
        maximum_diffuse_odds=decoding['maximum_diffuse_odds']) for j in range(len(read.calls))]
    calls = sbn.resolve_topology(decoded)
    return dict(unit_id=read.unit_id, strand=strand, source_strand=source, calls=calls)


_SR_WORKER = {}
from .execution import register_worker_state as _register_worker_state
_register_worker_state(_SR_WORKER)


def _load_sr_worker(path):
    import pickle
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def _normalize_task(path, index, reads):
    from .execution import task_thread_budget, load_worker_state
    task_thread_budget(1)
    state = load_worker_state(_SR_WORKER, path, _load_sr_worker)
    if 'normalizer' not in state:
        state['normalizer'] = sbn.BoundaryNormalizer(state['source_nodes'])
    return index, [_normalize_read(state['normalizer'], read, state['strand'], state['source'], state['decoding'])
                   for read in reads]


def _normalize_reads_in_processes(reads, source_nodes, strand, source, decoding, workers, note, chunk=32):
    import pickle, tempfile
    from concurrent.futures import FIRST_COMPLETED, wait
    from pathlib import Path
    from .execution import stage_executor
    chunks = [reads[i:i+chunk] for i in range(0, len(reads), chunk)]
    results = [None]*len(chunks)
    with tempfile.TemporaryDirectory(prefix='fiberhmm-native-sr-') as directory:
        path = str(Path(directory)/'source.pkl')
        with open(path, 'wb') as handle:
            pickle.dump(dict(source_nodes=source_nodes, strand=strand, source=source, decoding=decoding),
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
        executor, release = stage_executor(min(workers, len(chunks)))
        pending = {}; remaining = iter(enumerate(chunks)); done_reads = 0
        def submit():
            item = next(remaining, None)
            if item is not None:
                index, block = item
                pending[executor.submit(_normalize_task, path, index, block)] = index
        try:
            for _ in range(2*workers):
                submit()
            note(0)
            while pending:
                done, _ = wait(pending, timeout=.2, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    index, block = future.result()
                    results[index] = block; done_reads += len(block)
                    submit()
                note(done_reads)
        except BaseException:
            for future in pending:
                future.cancel()
            release(failed=True)
            raise
        release()
    if any(block is None for block in results):
        raise RuntimeError('Strand normalization finished without every read')
    return [record for block in results for record in block]


def extract_calls(stratum, start, end):
    calls = []
    for u in stratum['units']:
        for i, (a,b) in enumerate(u['representative_raw_tf_intervals']):
            if a >= end or b <= start:
                continue
            if b <= a:
                raise ValueError('Nonpositive source footprint')
            calls.append(dict(start=a, end=b, unit_id=u['unit_id'],
                observation_id=digest([stratum['stratum_id'],u['unit_id'],i,a,b])))
    return sorted(calls, key=lambda v:(v['start'],v['end'],v['observation_id']))


def make_kernel(data, centers, options, compute):
    return RegionFamilyLattice(data['grid_positions'], centers, options.ambiguity_bp,
        maximum_nodes=compute.maximum_nodes, maximum_edges=compute.maximum_edges)


def discover_cr(stratum, start, end, options, compute, progress):
    data = prepare_population(stratum, start, end, grid_bp=1, max_intervals=0,
        max_matrix_bytes=compute.maximum_matrix_mb*1024**2)
    calls = extract_calls(stratum, start, end)
    neighborhoods = overlap_neighborhoods(calls, start, end,
        options.neighborhood_bandwidth or options.ambiguity_bp)
    centers, aliases, ledger = [], [], []
    args = SimpleNamespace(**vars(options), start=start, end=end,
        neighborhood_seconds=compute.neighborhood_seconds, maximum_nodes=compute.maximum_nodes,
        maximum_edges=compute.maximum_edges, check=lambda: progress('cr', None))
    for number, ids in enumerate(neighborhoods):
        progress('cr', f'{stratum["dataset_id"]}: neighborhood {number+1}/{len(neighborhoods)}, {len(centers)} proposed classes')
        local, record = discover_neighborhood(data, [calls[i] for i in ids], args, number)
        centers.extend(local)
        aliases.extend([dict(neighborhood=number, local_family=f) for f in range(len(local))])
        ledger.append(record)
    centers, aliases, unavailable = deduplicate(centers, aliases, data['grid_positions'], options.ambiguity_bp)
    if not len(centers):
        return dict(data=data, kernel=None, catalog=[], records=[], ledger=ledger, unavailable=unavailable,
                    status='empty', reason='No source calls with visible opportunity geometries')
    progress('cr', f'{stratum["dataset_id"]}: fitting {len(centers)} classes jointly on all {len(data["units"])} units')
    kernel = make_kernel(data, centers, options, compute)
    allowed = family_availability(data['units'], centers, options.ambiguity_bp)
    fit, fits = kernel.fit(data['log_lr'], allowed=allowed, max_iter=options.global_iterations,
        check=lambda: progress('cr_fit', None),
        report=lambda initial,iteration:progress('cr_fit',
            f'{stratum["dataset_id"]}: whole-region {kernel.f}-class fit, start {initial:g}, iteration {iteration}/{options.global_iterations}'))
    family_ids = [f'{stratum["dataset_id"]}:F{i+1:04d}' for i in range(kernel.f)]
    data['family_ids'] = family_ids
    # Preserve the historical split exactly (string hash, not JSON string hash).
    import hashlib
    data['folds'] = np.asarray([int(hashlib.sha256(('cr-source-fold-v1|'+g).encode()).hexdigest()[:12],16)%2 for g in data['fold_group_ids']])
    data['strands'] = np.asarray([u['strand'] for u in data['units']])
    catalog = [dict(family=fid, family_index=f, consensus_start=int(c[0]), consensus_end=int(c[1]),
        ambiguity_bp=options.ambiguity_bp, log_activity=float(fit['eta'][f]), aliases=aliases[f])
        for f,(fid,c) in enumerate(zip(family_ids,centers))]
    assigned = assign_cr(data, kernel, fit['eta'], options, compute, progress)
    return dict(data=data, kernel=kernel, eta=fit['eta'], catalog=catalog, ledger=ledger,
        unavailable=unavailable, fits=fits,
        fit_diagnostics=dict(global_fit=summarize_fits(fits),nomination=nomination_fit_status(ledger)),
        status='complete', **assigned)


def assign_cr(data, kernel, eta, options, compute, progress, calls=None):
    """Freeze marginals first, then choose physically admissible joint actions."""
    units = data['units']; n, f = len(units), kernel.f
    spans = calls if calls is not None else [u['representative_raw_tf_intervals'] for u in units]
    allowed = family_availability([{**u,'representative_raw_tf_intervals':cc} for u,cc in zip(units,spans)],
                                  kernel.centers, kernel.ambiguity_bp)
    coordinates = representative_geometries(kernel)
    membership=np.zeros((n,f)); selected_mass=np.zeros((n,f)); eligible=np.zeros((n,f),bool)
    core_eligible=np.zeros((n,f),bool);core_opportunities=np.zeros((n,f),dtype=np.int32)
    core_ix=np.searchsorted(kernel.positions,kernel.centers)
    records=[]
    batch = min(compute.batch_size, max(1, compute.maximum_matrix_mb*1024**2//max(8*len(kernel.ga)*8,1)))
    for begin in range(0,n,batch):
        progress('cr_assign', f'{data["dataset_id"]}: assigning {begin}/{n} units')
        end=min(n,begin+batch); values=data['log_lr'][begin:end]
        prefix=lambda v: np.c_[np.zeros(len(v)),np.cumsum(v,axis=1)]
        lp=prefix(values); op=prefix(data['observed'][begin:end]); hp=prefix(data['hits'][begin:end]==1)
        llrs=lp[:,kernel.gb]-lp[:,kernel.ga]; opps=op[:,kernel.gb]-op[:,kernel.ga]; hits=hp[:,kernel.gb]-hp[:,kernel.ga]
        core_opps=op[:,core_ix[:,1]]-op[:,core_ix[:,0]]
        core_hits=hp[:,core_ix[:,1]]-hp[:,core_ix[:,0]]
        core_llrs=lp[:,core_ix[:,1]]-lp[:,core_ix[:,0]]
        physical=np.asarray([physically_allowed(u,coordinates,opps[j],options.minimum_opportunities)
                             for j,u in enumerate(units[begin:end])])
        for j in range(end-begin):
            eligible[begin+j,np.unique(kernel.gf[physical[j]])]=True
            core_opportunities[begin+j]=core_opps[j]
            core_eligible[begin+j]=physically_allowed(units[begin+j],kernel.centers,core_opps[j],options.minimum_opportunities)
        inc=kernel.evaluate(values,eta,allowed=allowed[begin:end])['family_inclusion']
        membership[begin:end]=inc
        action=physical & (llrs>0)
        for j,cc in enumerate(spans[begin:end]):
            action[j] &= overlap_mask(cc,coordinates)
        decoded=kernel.map_configuration(values,eta,allowed=allowed[begin:end],geometry_allowed=action)['geometry_by_family']
        for j,u in enumerate(units[begin:end]):
            cc=spans[begin+j]; proposals=[]
            for family in np.flatnonzero(decoded[j]>=0):
                g=int(decoded[j,family]); a,b=map(int,coordinates[g])
                lineage=[k for k,(lo,hi) in enumerate(cc) if a<hi and b>lo]
                if not lineage: raise AssertionError('CR grouping invented a call without an upstream TF')
                selected_mass[begin+j,family]=inc[j,family]
                proposals.append(dict(family=data['family_ids'][family],family_index=int(family),
                    consensus=kernel.centers[family].tolist(),interval=[a,b],model_membership=float(inc[j,family]),
                    native_log_lr=float(llrs[j,g]),opportunities=int(opps[j,g]),hits=int(hits[j,g]),
                    canonical_core_opportunities=int(core_opps[j,family]),canonical_core_hits=int(core_hits[j,family]),
                    canonical_core_log_lr=float(core_llrs[j,family]),canonical_core_eligible=bool(core_eligible[begin+j,family]),
                    source_ordinals=lineage,source_intervals=[cc[k] for k in lineage],
                    boundary_changed=all([a,b]!=list(cc[k]) for k in lineage)))
            proposals.sort(key=lambda v:v['interval'])
            if any(a['interval'][1]>b['interval'][0] for a,b in zip(proposals,proposals[1:])):
                raise AssertionError('Joint CR assignment produced overlapping intervals')
            records.append(dict(unit_id=u['unit_id'],strand=u['strand'],source_calls=cc,proposals=proposals))
    return dict(records=records,membership=membership,proposal_membership=selected_mass,
        eligible=eligible,core_eligible=core_eligible,core_opportunities=core_opportunities,allowed=allowed)
