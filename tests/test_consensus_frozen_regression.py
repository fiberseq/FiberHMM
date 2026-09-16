"""All-unit post-fit regression for an exported production consensus run.

Set FIBERHMM_CONSENSUS_FIXTURE to its directory. This tests exact geometry,
source lineage, margins and threshold nesting; it does not establish empirical
accuracy or independence of full-cohort family discovery.
"""
import os
from pathlib import Path
import numpy as np
import pytest
from fiberhmm.inference.consensus.artifacts import read_json,digest
from fiberhmm.inference.consensus.parameters import parse_options
from fiberhmm.inference.consensus.observations import prepare_population
from fiberhmm.inference.consensus.stages import make_kernel,assign_cr
from fiberhmm.inference.consensus.workflow import _pool


@pytest.mark.skipif(not os.environ.get('FIBERHMM_CONSENSUS_FIXTURE'),reason='External production consensus fixture not configured')
def test_all_exported_units_reproduce_geometry_margins_lineage_and_nested_q():
    from numba import config,set_num_threads
    path=Path(os.environ['FIBERHMM_CONSENSUS_FIXTURE'])
    result=read_json(path/'result.json.gz');payload=read_json(path/'evidence.json.gz')
    options=parse_options(result['manifest']['parameters'])
    options['compute'].cores=min(2,config.NUMBA_NUM_THREADS);set_num_threads(options['compute'].cores)
    for source in _pool(payload,options):
        ds=source['dataset_id'];stored=result['datasets'][ds]['cr']
        if stored.get('status')!='complete':continue
        expected={r['unit_id']:r for r in stored['records']}
        assert set(expected)=={u['unit_id'] for u in source['units']}
        for unit in source['units']:unit['representative_raw_tf_intervals']=expected[unit['unit_id']]['source_calls']
        with np.load(path/digest(ds)[:16]/'cr_model.npz',allow_pickle=False) as cache:
            region=payload['region']
            data=prepare_population(source,region['start'],region['end'],grid_bp=1,max_intervals=0,
                max_matrix_bytes=options['compute'].maximum_matrix_mb*1024**2)
            data['strands']=np.array([u['strand'] for u in data['units']])
            data['family_ids']=[f['family'] for f in stored['catalog']]
            np.testing.assert_array_equal(data['unit_ids'],cache['unit_ids'])
            kernel=make_kernel(data,cache['centers'],options['cr'],options['compute'])
            actual=assign_cr(data,kernel,cache['log_activities'],options['cr'],options['compute'],lambda *_:None)
            for key,column in [('membership','family_inclusion'),('proposal_membership','proposal_membership')]:
                np.testing.assert_allclose(actual[key],cache[column],atol=1e-11,rtol=1e-10)
            for key in ['eligible','core_eligible','core_opportunities','allowed']:
                if key in cache:np.testing.assert_array_equal(actual[key],cache[key])
        for row in actual['records']:
            old=expected[row['unit_id']]
            columns=['family','interval','source_ordinals']
            assert [[v[k] for k in columns] for v in row['proposals']]==[[v[k] for k in columns] for v in old['proposals']]
            preceding=None
            for q in [0,3,10,20,60]:
                visible={(c['family'],tuple(c['interval']),tuple(c['source_ordinals'])) for c in row['proposals']
                         if c['model_membership']>=1-10**(-q/10)}
                if preceding is not None:assert visible<=preceding
                preceding=visible
        print(ds,len(expected),len(stored['catalog']),'exact frozen parity, Q0/3/10/20/60 nesting')
