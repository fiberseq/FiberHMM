"""All-unit golden regression; set FIBERHMM_NAPA_CR_FIXTURE to the validated CR folder.

This test deliberately loads data fixtures, never benchmark inference code.
"""
import os
from pathlib import Path
import numpy as np
import pytest
from fiberhmm.inference.consensus.artifacts import read_json
from fiberhmm.inference.consensus.parameters import parse_options
from fiberhmm.inference.consensus.observations import prepare_population
from fiberhmm.inference.consensus.lattice import RegionFamilyLattice
from fiberhmm.inference.consensus.stages import assign_cr


@pytest.mark.skipif(not os.environ.get('FIBERHMM_NAPA_CR_FIXTURE'),reason='External all-unit NAPA fixture not configured')
def test_all_napa_assignments_match_validated_geometry_and_mass():
    path=Path(os.environ['FIBERHMM_NAPA_CR_FIXTURE']);manifest=read_json(path/'manifest.json')
    payload=read_json(manifest['input']);options=parse_options()
    from numba import set_num_threads
    set_num_threads(4)
    for s in payload['strata']:
        letter='D' if s['chemistry']=='ddda' else 'H';condition='boundary_sr' if letter=='D' else 'native'
        expected=read_json(path/f'{letter}_{condition}.proposals.json.gz')
        for u,r in zip(s['units'],expected):
            assert u['unit_id']==r['unit_id']
            u['representative_raw_tf_intervals']=r['source_calls']
        data=prepare_population(s,47514487,47518985,grid_bp=1,max_intervals=0)
        model_path=Path(manifest['reference_cr'])/s['stratum_id'].replace(':','_')/'assignments.npz'
        with np.load(model_path,allow_pickle=False) as z:
            centers=z['centers'].copy();eta=z['log_activities'].copy()
            np.testing.assert_array_equal(data['unit_ids'],z['unit_ids'])
        kernel=RegionFamilyLattice(data['grid_positions'],centers,10)
        data['family_ids']=[f'{letter}{f+1:03d}' for f in range(kernel.f)]
        actual=assign_cr(data,kernel,eta,options['cr'],options['compute'],lambda *_:None)
        with np.load(path/f'{letter}_memberships.npz',allow_pickle=False) as z:
            np.testing.assert_allclose(actual['membership'],z[f'{condition}_membership'],atol=1e-10,rtol=0)
            np.testing.assert_allclose(actual['proposal_membership'],z[f'{condition}_proposal_membership'],atol=1e-10,rtol=0)
            np.testing.assert_array_equal(actual['eligible'],z['eligible'])
        for a,b in zip(actual['records'],expected):
            assert [(c['family'],c['interval'],c['source_ordinals']) for c in a['proposals']]==[(c['family'],c['interval'],c['source_ordinals']) for c in b['proposals']]
        print(letter,len(data['units']),kernel.f,sum(len(r['proposals']) for r in actual['records']),'exact geometry parity')
