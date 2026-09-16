"""XCR controls must reach the exact graph without altering CR/display evidence."""
from copy import deepcopy
import pytest
from fiberhmm.inference.consensus.parameters import parse_options, parameter_schema, options_dict


def test_native_cross_defaults_are_separate_and_in_shared_schema():
    options=parse_options({'cr':{'predictive_replicates':63}})
    assert options['cr'].predictive_replicates==63
    assert options['cross'].native_predictive_replicates==4095
    fields={f['name']:f for f in parameter_schema()['cross']}
    for name in ('native_minimum_call_attribution_mass','native_minimum_geometry_retention','native_minimum_visible_geometry_mass'):
        assert fields[name]['default']==.05
    assert fields['native_minimum_testable_fraction']['default']==.5
    assert options_dict(options)['cross']['native_predictive_replicates']==4095


@pytest.mark.parametrize('name',['native_minimum_call_attribution_mass','native_minimum_geometry_retention','native_minimum_visible_geometry_mass'])
@pytest.mark.parametrize('value',[0.,-1.,1.01,float('nan'),True])
def test_geometry_fractions_reject_zero_nonfinite_and_invalid_values(name,value):
    with pytest.raises(ValueError):parse_options({'cross':{name:value}})


def test_precision_gate_applies_only_to_enabled_native_cross_and_independent_draws():
    parse_options({'cr':{'engine':'native_family_distribution'},'cross':{'native_predictive_replicates':31}})  # disabled
    parse_options({'cr':{'engine':'legacy_lattice'},'cross':{'enabled':True,'native_predictive_replicates':31}})
    parse_options({'cr':{'engine':'native_family_distribution','predictive_replicates':31},'cross':{'enabled':True}})
    with pytest.raises(ValueError,match='Native XCR predictive gate unresolved'):
        parse_options({'cr':{'engine':'native_family_distribution','predictive_replicates':65535},'cross':{'enabled':True,'native_predictive_replicates':31}})
    parse_options({'cr':{'engine':'native_family_distribution'},'cross':{'enabled':True,'native_reference_percent':99.99,'native_predictive_replicates':65535}})
    with pytest.raises(ValueError,match='65535-draw budget'):
        parse_options({'cr':{'engine':'native_family_distribution'},'cross':{'enabled':True,'native_reference_percent':99.999,'native_predictive_replicates':65535}})
    with pytest.raises(ValueError):parse_options({'cr':{'engine':'native_family_distribution'},'cross':{'native_predictive_replicates':65536}})
    with pytest.raises(ValueError):parse_options({'cr':{'engine':'native_family_distribution'},'cross':{'native_minimum_testable_fraction':1.1}})


def test_native_workflow_passes_all_controls_without_changing_source_or_cr_budget(tmp_path,monkeypatch):
    from fiberhmm.inference.consensus import native_workflow as module,workflow
    strata=[dict(dataset_id=name,chemistry=chemistry,units=[]) for name,chemistry in [('d','ddda'),('h','hia5-pacbio')]]
    payload=dict(region=dict(chrom='chr1',start=0,end=100),strata=strata)
    before=deepcopy(payload);captures=[];cr_draws=[]
    monkeypatch.setattr(workflow,'_pool',lambda *_:strata)
    monkeypatch.setattr(module,'nominate_catalog',lambda *_:([dict(family='F1')],{}))
    def classify(*args,**kwargs):
        cr_draws.append(kwargs['predictive_replicates']);return dict(calls=[],family_models=[])
    monkeypatch.setattr(module,'classify_family_profiles',classify)
    monkeypatch.setattr(module,'browser_cr',lambda *_,**__:dict(records=[],catalog=[]))
    monkeypatch.setattr(module,'run_native_auxiliary',lambda *_,**__:{})
    def graph(*args,**kwargs):
        captures.append(kwargs);return dict(links=[],nodes=[],status='complete')
    monkeypatch.setattr(module,'reciprocal_native_graph',graph)
    monkeypatch.setattr(module,'summarize_native_correspondences',lambda *_:[])
    options=parse_options({'sr':{'enabled':False},'cr':{'predictive_replicates':63,'residual_nomination':False},
        'cross':{'enabled':True,'native_predictive_replicates':16383,
            'native_minimum_call_attribution_mass':.07,'native_minimum_geometry_retention':.08,
            'native_minimum_visible_geometry_mass':.09,'native_minimum_testable_fraction':.6},
        'compute':{'maximum_matrix_mb':64}})
    module.run_native_workflow(payload,options,tmp_path)
    assert payload==before and cr_draws==[63,63]
    assert len(captures)==1
    call=captures[0]
    assert call['replicates']==16383
    assert call['minimum_call_attribution_mass']==.07
    assert call['minimum_geometry_retention']==.08
    assert call['minimum_visible_geometry_mass']==.09
    assert call['minimum_testable_fraction']==.6
    assert call['maximum_matrix_bytes']==64*1024**2
