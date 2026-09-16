import itertools
import math
import numpy as np
import pytest
from fiberhmm.inference.consensus.call_clustering import (ClusterOptions, RescueOptions, cluster_calls, fold_clusters,
    geometry_distance, prepare, rescue_decision, rescue_families, weighted_clusters,
    accessible_lower_tail, physical_reason,window_physics)
from fiberhmm.inference.consensus.call_clustering import run_call_clustering


def payload(units):
    return dict(region=dict(chrom='chr1',start=0,end=200),strata=[dict(dataset_id='d',
        model_manifest=dict(native_minimum_llr=5.),units=units)])


def unit(uid, strand='CT', calls=(), positions=(20,23,26,29), hits=(0,0,0,0), pa=.75, pp=.05):
    return dict(unit_id=uid,fold_group_id=uid,strand=strand,positions=list(positions),hits=list(hits),
        p_accessible=[pa]*len(positions),p_protected=[pp]*len(positions),
        reference_start=0,reference_end=200,aligned_blocks=[[0,200]],raw_nuc_intervals=[],
        msp_intervals=[[0,200]],native_multi_interval_calls=[dict(interval=list(iv)) for iv in calls])


def decision(hits, pa=.75, pp=.05, floor=5., opt=None):
    h=np.array(hits,dtype=bool); hit=np.full(len(h),math.log(pp/pa)); miss=np.full(len(h),math.log((1-pp)/(1-pa)))
    return rescue_decision(np.where(h,hit,miss),hit,miss,h,floor,opt)


def test_one_extra_hit_rescues_actual_subthreshold_positive_evidence():
    reason,ev,ce,gain=decision([0,0,0,1])
    assert reason=='one_hit_short' and 0 < ev < 4 and ev+gain >= 5
    assert ce > 5
    assert decision([0,0,0,1],opt=RescueOptions(allow_one_hit=False))[0] is None


def test_sparse_positive_lattice_and_empty_or_contradictory_controls():
    assert decision([0,0],pa=.5,floor=7.)[0]=='sparse_lattice'
    assert decision([0,0],pa=.5,floor=7.,opt=RescueOptions(allow_sparse=False))[0] is None
    assert decision([])[0] is None
    assert decision([1,1,1])[0] is None
    assert decision([0],pa=.1)[0] is None


def test_same_and_opposite_strand_support_and_no_self_support():
    reads,floors,region=prepare([payload([unit('s1',calls=[(20,30)]),unit('s2',calls=[(20,30)]),
        unit('s3',calls=[(20,30)]),unit('r','GA',hits=(0,0,0,1))])])
    f=cluster_calls(reads,region)['results']['XCR']
    r=rescue_families(reads,floors,f)
    obs=next(x['observations'][0] for x in r['records'] if x['unit_id']=='r')
    assert obs['state']=='rescued' and obs['source_support']=={'opposite_strand':3}
    r=rescue_families(reads,floors,f,RescueOptions(source_mode='same_strand'),retain_all=True)
    assert next(x['observations'][0]['state'] for x in r['records'] if x['unit_id']=='r')=='unsupported'
    reads['r']['strand']='CT';reads['r']['stratum']='d:CT'
    r=rescue_families(reads,floors,f,RescueOptions(source_mode='same_strand'))
    assert next(x['observations'][0]['state'] for x in r['records'] if x['unit_id']=='r')=='rescued'
    reads['r']['group']=reads['s1']['group']
    r=rescue_families(reads,floors,f,retain_all=True)
    assert next(x['observations'][0]['state'] for x in r['records'] if x['unit_id']=='r')=='unsupported'


@pytest.mark.parametrize('field,value,reason',[
    ('raw_nuc_intervals',[[25,27]],'nucleosome_overlap'),
    ('msp_intervals',[[0,25],[27,200]],'outside_msp'),
    ('aligned_blocks',[[0,25],[26,200]],'alignment_gap')])
def test_physical_masks_are_not_stitched_across(field,value,reason):
    recipient=unit('r',hits=(0,0,0,1));recipient[field]=value
    reads,floors,region=prepare([payload([unit(str(i),calls=[(20,30)]) for i in range(3)]+[recipient])])
    fams=cluster_calls(reads,region)['results']['XCR']
    row=next(r for r in rescue_families(reads,floors,fams,retain_all=True)['records'] if r['unit_id']=='r')['observations'][0]
    assert row['state']=='unmeasurable' and row['reason']==reason


def test_overlapping_nonmember_call_does_not_become_member():
    units=[unit(str(i),calls=[(20,30)]) for i in range(3)]+[unit('r',calls=[(29,40)])]
    reads,floors,region=prepare([payload(units)])
    fams=cluster_calls(reads,region,ClusterOptions(cut=.1,max_similarity=.99))['results']['XCR']
    target=next(f for f in fams if f['interval']==[20,30])
    row=next(r for r in rescue_families(reads,floors,[target],retain_all=True)['records'] if r['unit_id']=='r')['observations'][0]
    assert row['state']=='competing_call'


def test_folding_does_not_chain_through_a_recipient():
    groups=[]
    for j,(iv,n) in enumerate([([0,10],3),([3,13],4),([6,16],5)]):
        groups.append([dict(unit=f'{j}_{i}',ordinal=0,interval=iv,stratum='d:CT') for i in range(n)])
    folded=fold_clusters(groups,ClusterOptions())
    assert sorted(map(len,folded))==[3,9]


def test_sparse_lattice_excuses_truncation_and_dense_lattice_separates():
    rows=[dict(unit='r')]
    sparse={'r':dict(positions=np.array([20,21,22]))}
    dense={'r':dict(positions=np.arange(0,100))}
    assert geometry_distance([20,23],[10,50],rows,rows,sparse,ClusterOptions())==0
    assert geometry_distance([20,23],[10,50],rows,rows,dense,ClusterOptions())==1


def test_no_union_lattice_rounding_or_1500_call_split_and_input_order_invariance():
    units=[unit(str(i),calls=[(19,31)]) for i in range(1601)]
    p=payload(units); reads,floors,region=prepare([p]); a=cluster_calls(reads,region)
    p['strata'][0]['units'].reverse(); reads,_,region=prepare([p]); b=cluster_calls(reads,region)
    assert a==b
    assert len(a['results']['XCR'])==1 and a['results']['XCR'][0]['interval']==[19,31]


def test_weighted_average_linkage_against_bruteforce_unique_geometry_reference():
    # Dense reference using the same mandated initial zero-distance compression.
    opt=ClusterOptions(cut=.45)
    rows=[];reads={}
    for k,(iv,n) in enumerate([([0,10],3),([3,14],2),([10,20],4),([40,45],2)]):
        for j in range(n):
            uid=f'{k}_{j}'; rows.append(dict(unit=uid,ordinal=0,interval=iv))
            reads[uid]=dict(positions=np.arange(50))
    got=sorted(sorted(c['unit'] for c in g) for g in weighted_clusters(rows,reads,opt))
    groups=[[c for c in rows if tuple(c['interval'])==iv] for iv in sorted({tuple(c['interval']) for c in rows})]
    while True:
        candidates=[]
        for i,j in itertools.combinations(range(len(groups)),2):
            d=np.mean([geometry_distance(a['interval'],b['interval'],[a],[b],reads,opt) for a in groups[i] for b in groups[j]])
            candidates.append((d,i,j))
        if not candidates or min(candidates)[0]>opt.cut:break
        _,i,j=min(candidates);groups[i]+=groups[j];groups.pop(j)
    assert got==sorted(sorted(c['unit'] for c in g) for g in groups)


def test_duplicate_units_and_wrong_chromosome_fail_closed():
    with pytest.raises(ValueError,match='Duplicate'):
        prepare([payload([unit('r'),unit('r')])])
    a=payload([unit('r')]); b=payload([unit('s')]);b['region']['chrom']='chr2'
    with pytest.raises(ValueError,match='chromosomes'):prepare([a,b])


def test_exact_conditional_tail_against_exhaustive_enumeration():
    pa=np.array([.1,.2,.7,.9,.6])
    for k in range(6):
        exact=sum(np.prod(np.where(h,pa,1-pa)) for h in itertools.product([0,1],repeat=5) if sum(h)<=k)
        assert accessible_lower_tail(pa,k)==pytest.approx(exact)
    assert accessible_lower_tail([.5,.5],0)==pytest.approx(.25)


def test_sparse_positive_rescue_retains_provisional_label():
    units=[unit(str(i),calls=[(20,30)]) for i in range(3)]+[unit('r',positions=(21,28),hits=(0,0),pa=.5)]
    reads,floors,region=prepare([payload(units)])
    f=cluster_calls(reads,region)['results']['XCR'];result=rescue_families(reads,floors,f)
    obs=next(r['observations'][0] for r in result['records'] if r['unit_id']=='r')
    assert obs['state']=='rescued' and obs['rescue_support']=='provisional'
    assert obs['accessible_null_tail']==pytest.approx(.25)
    fractions=result['families'][0]['detection_fraction']
    assert fractions['with_supported_rescue']<fractions['including_provisional']


def test_vectorized_physics_matches_interval_reference():
    r=dict(blocks=[[0,12],[15,45]],nucs=[[5,8],[35,38]],msps=[[0,20],[25,45]])
    windows=np.array([[a,b] for a in range(46) for b in range(a+1,47)])
    for allowance in (0,1,3):
        opt=RescueOptions(maximum_alignment_gap_bp=allowance)
        assert window_physics(r,windows,opt).tolist()==[physical_reason(r,w,opt) or '' for w in windows]


def test_xcr_can_rescue_across_datasets_and_default_calls_are_not_mutated():
    import copy
    p=payload([unit(str(i),calls=[(20,30)]) for i in range(3)])
    p['strata'].append(dict(dataset_id='foreign',model_manifest=dict(native_minimum_llr=5.),
                           units=[unit('r',hits=(0,0,0,1))]))
    before=copy.deepcopy(p)
    result=run_call_clustering([p],rescue_scopes=('XCR','SR foreign'))
    xcr=next(r for r in result['rescue']['XCR']['records'] if r['unit_id']=='r')['observations'][0]
    assert xcr['state']=='rescued' and xcr['source_support']=={'other_dataset':3}
    assert not result['rescue']['SR foreign']['records'][0]['observations']
    assert p==before


def test_compact_state_ledger_agrees_with_full_details():
    reads,floors,region=prepare([payload([unit(str(i),calls=[(20,30)]) for i in range(3)]+[unit('r',hits=(1,1,1,1))])])
    families=cluster_calls(reads,region)['results']['XCR']
    compact=rescue_families(reads,floors,families)
    full=rescue_families(reads,floors,families,retain_all=True)
    assert compact['counts']==full['counts']
    for left,right in zip(compact['records'],full['records']):
        assert [compact['state_codebook'][code] for code in left['state_codes']]==[row['state'] for row in right['observations']]


def test_harmonization_does_not_rescue_by_default():
    p = payload([unit(str(i), calls=[(20,30)]) for i in range(3)] +
                [unit('recipient', hits=(0,0,0,1))])
    result = run_call_clustering([p])
    assert result['rescue'] == {}
    assert sum(f['calls'] for f in result['results']['XCR']) == 3
