"""Independent reference tasks, deterministic ordering, bounded live matrices.

Only scheduling changes. Kernels, seeds, folds, native lattices and decisions
are the extracted reference implementation. Workers have single-threaded BLAS.
"""
from concurrent.futures import FIRST_COMPLETED, wait
import time

from ..artifacts import read_json
from ..execution import register_worker_state, load_worker_state, task_thread_budget, stage_executor
from ..measurement_distribution import set_predictive_decision_stop, reset_predictive_decision_stop
from ..progress import report
from .reference.run_bounded_parent_panel import evaluate_cohort
from .reference.overlapping_family_update import extend_parent
from .reference.run_native_cell_consolidation import candidate_inputs
from .reference.cross_source_family_consolidation import foreign_child_scores

_state=register_worker_state({})


def _context(path):
    task_thread_budget(1)
    load_worker_state(_state,path,read_json)
    return _state


def parent_task(path, proposal, maximum_bytes, stop=0):
    started=time.monotonic(); case=_context(path)['case']
    calls,region=candidate_inputs(case,proposal)
    token=set_predictive_decision_stop(stop)
    try:
        _,result,_=next(evaluate_cohort(case['units'],calls,region,[proposal['radius']],
                        maximum_bytes,center_bounds=proposal['center_bounds']))
        result=extend_parent(case,result,region,maximum_bytes=maximum_bytes)
        return dict(result=result,seconds=time.monotonic()-started)
    except ValueError as exc:
        expected=('Fewer than two unambiguous physical fitting groups','No native opportunities;',
                  'Event has no admissible geometry;')
        if not str(exc).startswith(expected):raise
        return dict(failure=dict(proposal=proposal['id'],error=str(exc),status='unassessed'),
                    seconds=time.monotonic()-started)
    finally:reset_predictive_decision_stop(token)


def foreign_task(path, channel, block=None, stop=0):
    started=time.monotonic(); context=_context(path)
    token=set_predictive_decision_stop(stop)
    try:records=foreign_child_scores(context['case'],{channel:context['parts'][channel]},block=block)
    finally:reset_predictive_decision_stop(token)
    return dict(records=records,seconds=time.monotonic()-started)


def parent_working_bytes(case,proposal):
    calls,region=candidate_inputs(case,proposal)
    size=region[1]-region[0]+1
    # Reference's full event-grid estimate plus headroom for masks/cell arrays.
    return 2*(len(calls)*size*size*9+size*size*8*30)+128*1024**2


def ordered_tasks(function,tasks,*,cores,maximum_bytes,progress,stage,on_result):
    """At most cores tasks; memory-limited jobs can use the whole budget alone.

    Results are delivered as completed for checkpointing, returned in input
    order. Cancellation kills the stage workers; already published checkpoints
    remain reusable. Polling is responsive even inside an uninstrumented kernel.
    """
    tasks=list(tasks); results={}; pending={}; remaining=list(tasks)
    executor,release=stage_executor(cores); failed=True
    try:
        while remaining or pending:
            resident=sum(t[2] for t in pending.values())
            while remaining and len(pending)<cores:
                index=next((i for i,t in enumerate(remaining)
                    if not pending or resident+t[2]<=maximum_bytes),None)
                if index is None:break
                key,args,cost=remaining.pop(index)
                pending[executor.submit(function,*args)]=(key,args,cost)
                resident+=cost
            done,_=wait(pending,timeout=.2,return_when=FIRST_COMPLETED)
            for future in done:
                key,_,_=pending.pop(future)
                value=future.result();on_result(key,value);results[key]=value
            report(progress,stage,f'{stage.replace("_"," ").capitalize()}: {len(results)}/{len(tasks)} completed',
                   completed=len(results),total=len(tasks),unit='tasks')
        failed=False
    finally:release(failed)
    return {key:results[key] for key,_,_ in tasks}
