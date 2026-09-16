import numpy as np
from fiberhmm.inference.consensus.lattice import RegionFamilyLattice
from fiberhmm.inference.consensus.recall_action import decode_posterior_accuracy
from fiberhmm.inference.consensus.recall import projection_rectangles,exposure,source_recipe


def test_edge_mass_dilution_does_not_turn_supported_protection_into_empty_call():
    k=RegionFamilyLattice(np.arange(20),[[6,11]],3)
    value=np.full((1,20),.3);eta=np.zeros(1)
    post=k.evaluate(value,eta,export_geometry=True)
    assert post['family_inclusion'][0,0]>.5
    assert k.map_configuration(value,eta)['geometry_by_family'][0,0]==-1
    got=decode_posterior_accuracy(k,value,np.ones_like(value,dtype=bool),post,eta,
        np.ones((1,1),bool),np.ones((1,len(k.ga)),bool),np.zeros((1,len(k.ga))))
    assert got['geometry_by_family'][0,0]>=0
    assert got['expected_accuracy_improvement'][0]>0


def test_no_observation_is_not_a_reward_for_filling_missing_bases():
    k=RegionFamilyLattice(np.arange(20),[[6,11]],3)
    value=np.zeros((1,20));eta=np.zeros(1)
    post=k.evaluate(value,eta,export_geometry=True)
    got=decode_posterior_accuracy(k,value,np.zeros_like(value,dtype=bool),post,eta,
        np.ones((1,1),bool),np.zeros((1,len(k.ga)),bool),np.zeros((1,len(k.ga))))
    assert got['geometry_by_family'][0,0]==-1
    assert got['expected_accuracy_improvement'][0]==0


def test_projection_exposure_counts_actual_integer_aliases_without_donating_mass():
    k=RegionFamilyLattice(np.array([2,8,12,19,27]),[[5,22]],5)
    rects=projection_rectangles(k);fraction,coords=exposure(rects,k.centers[k.gf],[(6,24)])
    for g,(sl,sh,el,eh) in enumerate(rects):
        possible=[(a,b) for a in range(sl,sh+1) for b in range(el,eh+1)]
        valid=[(a,b) for a,b in possible if a>=6 and b<=24]
        assert fraction[g]==len(valid)/len(possible)
        if valid:assert tuple(coords[g]) in valid


def test_source_rates_exclude_held_fold_and_opposite_mode_uses_other_strand():
    mem=np.array([[1.],[1.],[1.],[1.],[1.],[0.]])
    eligible=np.ones_like(mem,bool);strands=np.array(['CT','GA','GA','GA','GA','CT'])
    folds=np.array([0,0,1,1,1,1])
    _,allowed,recipe=source_recipe(mem,eligible,strands,folds,'CT',0,'opposite_strand',1.)
    assert recipe['source_units']==3 and recipe['counts']==[3] and allowed[0]


def test_recall_null_uses_the_same_user_adequacy_threshold(monkeypatch):
    from fiberhmm.inference.consensus import recall
    from fiberhmm.inference.consensus.parameters import RescueOptions
    positions=np.arange(100,150,2);hits=(positions<120)|(positions>=134)
    unit=dict(unit_id='control-threshold',strand='CT',positions=positions.tolist(),hits=hits.astype(int).tolist(),
        p_accessible=[.8]*len(positions),p_protected=[.1]*len(positions),
        representative_raw_tf_intervals=[],native_multi_interval_tf_intervals=[],raw_nuc_intervals=[],
        aligned_blocks=[[100,150]],msp_intervals=[[100,150]],_region=[100,150])
    options=RescueOptions(maximum_diffuse_odds=7.)
    values=np.where(hits,np.log(.1/.8),np.log(.9/.2))[None,:]
    data=dict(units=[unit],log_lr=values,observed=np.ones_like(values,bool),hits=hits[None,:].astype(int),
        start=100,end=150,rescue_options=options,family_ids=['f'])
    kernel=RegionFamilyLattice(positions,[[120,134]],2)
    seen=[];original=recall.adequacy_mask
    def check(*args,**kwargs):
        seen.append(kwargs.get('maximum_diffuse_odds',args[5] if len(args)>5 else 100.))
        return original(*args,**kwargs)
    monkeypatch.setattr(recall,'adequacy_mask',check)
    recall.score_group(kernel,data,np.array([0]),np.array([0.]),np.array([True]),projection_rectangles(kernel),
        [.9],np.random.default_rng(76),2,'posterior_accuracy')
    assert seen==[7.,7.,7.]
