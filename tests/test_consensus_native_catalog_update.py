from copy import deepcopy

import numpy as np
import pytest

from fiberhmm.inference.consensus.native_catalog_update import (
    bind_native_result, compose_append_frozen, native_snapshot_digest,
)
from fiberhmm.inference.consensus.native_presentation import browser_cr, reclassify_records


REGION = dict(chrom='chr1', start=0, end=100)
OPTIONS = dict(family_model='latent_distribution', minimum_edge_tolerance_bp=10,
    core_contradiction_odds=100., max_fit_iterations=100, predictive_replicates=4095,
    scoring_folds=10, maximum_matrix_bytes=2*1024**3, loss_odds_levels=[10., 100., 1000.])
IMPLEMENTATION = {'measurement_family.py': 'a'*64, 'native_presentation.py': 'b'*64}


def fixture():
    units = [dict(unit_id=f'u{i}', fold_group_id=f'g{i}', read_name=f'r{i}', strand='CT',
        positions=[12, 22, 32], hits=[0, 0, i % 2], p_accessible=[.9]*3,
        p_protected=[.1]*3, representative_raw_tf_intervals=[[10+i//2, 40+i//2]],
        raw_nuc_intervals=[], quality_mask=[True]*3) for i in range(3)]
    stratum = dict(dataset_id='ds', chemistry='ddda', units=units)
    calls = [dict(unit_index=i, unit_id=f'u{i}', ordinal=0, start=10+i//2,
        end=40+i//2, strand='CT', evidence_group_id=f'g{i}') for i in range(3)]
    catalog = [dict(family='ds:F1', consensus_start=10, consensus_end=40, aliases=[]),
               dict(family='ds:F2', consensus_start=10, consensus_end=40, aliases=[])]

    def model(fid, indices, interval):
        groups = [f'g{i}' for i in indices]
        training = {'full': groups, **{str(i): [g for g in groups if g != f'g{i}'] for i in indices}}
        return dict(family=fid, status='fitted', source_units=len(indices),
            source_call_indices=indices, reference_interval=interval, domain=[0,100],
            fold_models={f: dict(parameters=[1., 2., 3.], projection_grid_sha256='c'*64,
                                parameter_reference=interval) for f in training},
            training_evidence_groups=training,
            fit_diagnostics={f: dict(source_units=len(g), converged=True) for f,g in training.items()})

    old = model('ds:F1', [0,1], [10,40])
    new = model('ds:RN_new', [0,1,2], [11,41])

    def score(fid, i, mod, *, upper, distance, status='scored'):
        fold = str(i) if str(i) in mod['fold_models'] else 'full'
        return dict(family=fid, status=status, predictive_tail_interval=[0.,upper],
            geometry_distance_sq=distance, floor_adjusted_loss=2., native_loss=3.,
            source_units=len(mod['training_evidence_groups'][fold]), scoring_fold=fold,
            fit_converged=True, simulations=4095, tail_exceedances=5,
            own_evidence_excluded=i in mod['source_call_indices'])

    diagnostics = dict(source_calls=3,out_of_region_calls=0,source_units=3,
        original_catalog_families=2, **{k: OPTIONS[k] for k in
        ('family_model','minimum_edge_tolerance_bp','core_contradiction_odds','predictive_replicates','scoring_folds')})
    initial = dict(status='complete',calls=calls,family_models=[old,
        dict(family='ds:F2',status='no_source_observations',source_units=0)],
        call_family_evidence=[[score('ds:F1',i,old,upper=.1,distance=10,
            status='core_contradicted' if i==2 else 'scored')] for i in range(3)],
        source_homes=['ds:F1','ds:F1',None],diagnostics=diagnostics,
        partitions={},predictive_partitions={})
    addition = dict(family='ds:RN_new',consensus_start=11,consensus_end=41,
        nomination_provenance='unexplained_existing_call_projection',
        source_aliases=[dict(unit_id='u2',ordinal=0,interval=[11,41])],
        nomination_source_units=1,nomination_source_calls=1)
    augmented_catalog=[*deepcopy(catalog),addition]
    augmented=dict(status='complete',calls=deepcopy(calls),
        family_models=[dict(family='ds:F1',status='no_source_observations',source_units=0),
                       deepcopy(initial['family_models'][1]),new],
        call_family_evidence=[[score('ds:RN_new',i,new,upper=.002,distance=0)] for i in range(3)],
        source_homes=['ds:RN_new']*3,diagnostics=dict(diagnostics,original_catalog_families=3),
        partitions={},predictive_partitions={},nomination_update=dict(initial_proposals=2,
            added_proposals=1,additions=[addition],nomination_reference_percent=99.9))
    return stratum,catalog,augmented_catalog,initial,augmented


def bindings(s,c,a,i,r,*,augmented_stratum=None,augmented_options=None,implementation=None):
    initial=bind_native_result(i,stratum=s,catalog=c,region=REGION,
        model_options=OPTIONS,implementation_contract=IMPLEMENTATION)
    augmented=bind_native_result(r,stratum=augmented_stratum or s,catalog=a,region=REGION,
        model_options=augmented_options or OPTIONS,implementation_contract=implementation or IMPLEMENTATION,
        nomination_parent_digest=initial['snapshot_digest'])
    return initial,augmented


def compose(s,c,a,i,r,**kwargs):
    ib,ab=bindings(s,c,a,i,r,**kwargs)
    return compose_append_frozen(i,r,initial_binding=ib,augmented_binding=ab,
                                 initial_catalog=c,augmented_catalog=a)


def test_exact_old_and_new_payloads_and_detached_outputs():
    args=fixture();s,c,a,i,r=args;before=native_snapshot_digest(args)
    out=compose(*args)
    assert out.result['family_models']==[*i['family_models'],r['family_models'][-1]]
    assert out.result['call_family_evidence']==[[*old,*new] for old,new in zip(i['call_family_evidence'],r['call_family_evidence'])]
    assert 'source_homes' not in out.result
    assert out.result['initial_source_homes_historical']==i['source_homes']
    assert out.model_versions['ds:RN_new']['source_call_ids']==['ds:u0:0','ds:u1:0','ds:u2:0']
    assert out.model_versions['ds:F1']['source_evidence_group_ids']==['g0','g1']
    assert out.model_versions['ds:RN_new']['source_selection'].startswith('actual model-specific')
    assert out.result['partitions']=={} # Obsolete loss tiers are not relabeled as predictive tiers.
    assert out.report['downstream_requires_recomputation']
    out.result['family_models'][0]['source_call_indices'].append(999)
    out.result['call_family_evidence'][0][0]['native_loss']=999
    out.catalog[0]['consensus_start']=999
    assert native_snapshot_digest(args)==before


def test_new_primary_preserves_old_alternative_and_can_classify_unresolved():
    out=compose(*fixture())
    strict=out.result['predictive_partitions']['95.0']['assignments']
    loose=out.result['predictive_partitions']['99.9']['assignments']
    assert [p['family'] for p in strict[:2]]==['ds:F1']*2
    assert strict[2]['primary_evidence'] is None
    assert all(p['family']=='ds:RN_new' for p in loose)
    assert loose[0]['compatible_alternatives']==['ds:F1']
    assert loose[2]['compatible_alternatives']==[] # Old core contradiction remains a veto.
    assert out.report['references']['99.9']['primary_switches_to_new']==2
    assert out.report['references']['99.9']['newly_classified_existing_calls']==1
    assert all(p['interval']==[p['start'],p['end']] for p in loose)


@pytest.mark.parametrize('status',['core_contradicted','no_independent_training_unit','no_recipient_information'])
def test_unavailable_or_vetoed_new_evidence_does_not_promote(status):
    args=fixture();r=args[-1]
    for row in r['call_family_evidence']:
        row[0]['status']=status
    out=compose(*args)
    loose=out.result['predictive_partitions']['99.9']['assignments']
    assert [p['family'] for p in loose[:2]]==['ds:F1']*2
    assert loose[2]['primary_evidence'] is None


def test_failed_fit_diagnostic_is_preserved_without_inventing_new_gate():
    args=fixture();r=args[-1];model=r['family_models'][-1]
    for fit in model['fit_diagnostics'].values():fit['converged']=False
    for row in r['call_family_evidence']:row[0]['fit_converged']=False
    out=compose(*args)
    assert out.result['predictive_partitions']['99.9']['assignments'][0]['family']=='ds:RN_new'
    assert out.result['family_models'][-1]==model


def test_no_expansion_returns_exact_initial_result_not_refitted_versions():
    s,c,a,i,r=fixture();r=deepcopy(i)
    r['family_models'][0]['fit_diagnostics']['full']['converged']=False
    r['call_family_evidence'][2][0]['fit_converged']=False
    r['nomination_update']=dict(initial_proposals=2,added_proposals=0,additions=[])
    out=compose(s,c,c,i,r)
    assert out.result==i and out.catalog==c and out.report['no_expansion']
    assert not out.report['downstream_requires_recomputation']
    assert out.result is not i


def test_list_and_array_axes_have_identical_bindings():
    s,c,a,i,r=fixture();changed=deepcopy(s)
    for unit in changed['units']:
        for key in ('positions','hits','p_accessible','p_protected','representative_raw_tf_intervals','quality_mask'):
            unit[key]=np.asarray(unit[key])
    left=bind_native_result(i,stratum=s,catalog=c,region=REGION,model_options=OPTIONS,implementation_contract=IMPLEMENTATION)
    right=bind_native_result(i,stratum=changed,catalog=c,region=REGION,model_options=OPTIONS,implementation_contract=IMPLEMENTATION)
    assert left==right


@pytest.mark.parametrize('field',['hits','p_accessible','quality_mask'])
def test_observation_emission_and_mask_drift_is_rejected(field):
    s,c,a,i,r=fixture();changed=deepcopy(s)
    changed['units'][0][field][0]={'hits':1,'p_accessible':.8,'quality_mask':False}[field]
    with pytest.raises(ValueError,match='observation_sha256'):
        compose(s,c,a,i,r,augmented_stratum=changed)


def test_option_and_implementation_drift_is_rejected():
    s,c,a,i,r=fixture();r['diagnostics']['minimum_edge_tolerance_bp']=11
    with pytest.raises(ValueError,match='model_options'):
        compose(s,c,a,i,r,augmented_options=dict(OPTIONS,minimum_edge_tolerance_bp=11))
    s,c,a,i,r=fixture()
    with pytest.raises(ValueError,match='implementation_contract'):
        compose(s,c,a,i,r,implementation=dict(IMPLEMENTATION,**{'measurement_family.py':'d'*64}))


@pytest.mark.parametrize('where',['model','score','grid'])
def test_changes_after_binding_are_stale_even_if_same_family_id(where):
    s,c,a,i,r=fixture();ib,ab=bindings(s,c,a,i,r)
    if where=='model':i['family_models'][0]['domain'][0]=1
    elif where=='grid':i['family_models'][0]['fold_models']['0']['projection_grid_sha256']='d'*64
    else:i['call_family_evidence'][0][0]['floor_adjusted_loss']=99.
    with pytest.raises(ValueError,match='changed after'):
        compose_append_frozen(i,r,initial_binding=ib,augmented_binding=ab,initial_catalog=c,augmented_catalog=a)


@pytest.mark.parametrize('where',['model','catalog','score'])
def test_duplicate_ids_are_rejected(where):
    s,c,a,i,r=fixture()
    if where=='model':r['family_models'].append(deepcopy(r['family_models'][0]))
    elif where=='catalog':a.append(deepcopy(a[0]))
    else:r['call_family_evidence'][0].append(deepcopy(r['call_family_evidence'][0][0]))
    with pytest.raises(ValueError,match='duplicate family ID'):
        compose(s,c,a,i,r)


def test_missing_old_candidate_and_call_order_are_rejected():
    s,c,a,i,r=fixture();i['call_family_evidence'][0]=[]
    with pytest.raises(ValueError,match='Missing or out-of-domain'):
        compose(s,c,a,i,r)
    s,c,a,i,r=fixture();r['calls']=r['calls'][::-1]
    with pytest.raises(ValueError,match='call order'):
        compose(s,c,a,i,r)


def test_source_group_and_fold_exclusion_are_enforced():
    s,c,a,i,r=fixture();r['family_models'][-1]['training_evidence_groups']['0'].append('g0')
    r['family_models'][-1]['fit_diagnostics']['0']['source_units']=3
    r['call_family_evidence'][0][0]['source_units']=3
    with pytest.raises(ValueError,match='own scoring model'):
        compose(s,c,a,i,r)
    s,c,a,i,r=fixture();r['family_models'][-1]['source_call_indices']=[0,0,2]
    with pytest.raises(ValueError,match='Duplicate model-specific'):
        compose(s,c,a,i,r)


def test_unidentified_nomination_and_stale_parent_are_rejected():
    s,c,a,i,r=fixture();r['nomination_update']['additions']=[]
    with pytest.raises(ValueError,match='nomination provenance'):
        compose(s,c,a,i,r)
    s,c,a,i,r=fixture();ib,ab=bindings(s,c,a,i,r)
    # A fully internally consistent binding that names the WRONG parent.
    ab=bind_native_result(r,stratum=s,catalog=a,region=REGION,model_options=OPTIONS,
        implementation_contract=IMPLEMENTATION,nomination_parent_digest='f'*64)
    with pytest.raises(ValueError,match='stale or unidentified initial'):
        compose_append_frozen(i,r,initial_binding=ib,augmented_binding=ab,initial_catalog=c,augmented_catalog=a)


@pytest.mark.parametrize('key',['cross','rescue','count_groups','curation'])
def test_stale_downstream_payload_is_never_carried_forward(key):
    args=fixture();args[3][key]={}
    with pytest.raises(ValueError,match='Stale downstream'):
        compose(*args)


def test_existing_proposal_cannot_be_overwritten_under_same_id():
    s,c,a,i,r=fixture();a[0]['consensus_start']=9
    with pytest.raises(ValueError,match='proposal identity/geometry'):
        compose(s,c,a,i,r)


def test_browser_projection_needs_no_global_source_homes_and_is_nested():
    s,c,a,i,r=fixture();out=compose(s,c,a,i,r)
    cr=browser_cr(s,out.catalog,out.result,region=REGION);before=native_snapshot_digest(cr)
    strict=reclassify_records(cr,95);loose=reclassify_records(cr,99.9)
    assert [p['interval'] for row in strict for p in row['proposals']]==[p['interval'] for row in loose for p in row['proposals']]
    assert all(p['source_call_id'].startswith('ds:u') for row in loose for p in row['proposals'])
    assert native_snapshot_digest(cr)==before
