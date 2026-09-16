"""Diagnostic only: exact wider observation-pattern memoization, paired timing.

Executes an inspected copy of the existing kernel. Does not modify runtime
source, reduce draws or publish an alternative model. Every count must agree.
"""
import argparse
import inspect
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fiberhmm.inference.consensus import measurement_distribution as md
from fiberhmm.inference.consensus.artifacts import read_json,write_json
from fiberhmm.inference.consensus.harmonized_families.parallel import parent_task


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('context',type=Path)
    p.add_argument('proposals',type=Path);p.add_argument('proposal_id');p.add_argument('output',type=Path)
    args=p.parse_args()
    original=md._predictive_exceedances
    source=inspect.getsource(original.py_func).replace('cache=True','cache=False').replace(
        'def _predictive_exceedances(', 'def _memo_probe(')
    source=source.replace('    for _ in range(replicates):',
        '    wide = 16 < len(pa) <= 63\n    lookup = {np.int64(-1): np.int8(-1)}\n    for _ in range(replicates):')
    source=source.replace('if len(memo) and hit:','if (len(memo) or wide) and hit:')
    source=source.replace('        best = -np.inf',
        '        if wide and pattern in lookup:\n            count += lookup[pattern]\n            continue\n        best = -np.inf')
    source=source.replace('        count += exceeds','        if wide:\n            lookup[pattern] = np.int8(exceeds)\n        count += exceeds')
    namespace=dict(vars(md));exec(compile(source,'<exact-memo-probe>','exec'),namespace)
    alternative=namespace['_memo_probe'];rows=[]
    def paired(*values):
        if not rows:original(*values);alternative(*values)
        t=time.monotonic();expected=original(*values);a=time.monotonic()-t
        t=time.monotonic();observed=alternative(*values);b=time.monotonic()-t
        if expected!=observed:raise AssertionError('Predictive count changed')
        rows.append(dict(opportunities=len(values[0]),reference_seconds=a,alternative_seconds=b,count=int(expected)))
        return expected
    md._predictive_exceedances=paired
    proposal=next(p for p in read_json(args.proposals) if p['id']==args.proposal_id)
    try:parent_task(str(args.context.resolve()),proposal,2*1024**3)
    finally:md._predictive_exceedances=original
    report=dict(calls=len(rows),identical_counts=True,reference_seconds=sum(r['reference_seconds'] for r in rows),
                alternative_seconds=sum(r['alternative_seconds'] for r in rows),records=rows)
    write_json(args.output,report);print({k:v for k,v in report.items() if k!='records'},flush=True)


if __name__=='__main__':main()
