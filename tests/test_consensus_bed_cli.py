from pathlib import Path
from copy import deepcopy
import json
import pytest
from fiberhmm.inference.consensus.regions import load_bed, orient_unit, pool_payloads, automatic_parameters
from fiberhmm.inference.consensus.artifacts import write_json, read_json
from fiberhmm.inference.consensus.cli import main


def test_minus_base_interval_and_emission_transport():
    u=dict(unit_id='u',read_name='movie/1/ccs',strand='CT',positions=[101,104,109],hits=[1,0,1],
           contexts=[1,2,3],p_accessible=[.6,.7,.8],p_protected=[.1,.2,.3],m5c_observations=[True,False,False],
           raw_tf_intervals=[[102,109]],representative_raw_tf_intervals=[[102,109]],
           native_multi_interval_tf_intervals=[[102,109]],native_multi_interval_calls=[dict(interval=[102,109],query_interval=[2,9])],
           msp_intervals=[[100,110]],raw_nuc_intervals=[],aligned_blocks=[[100,110]],reference_start=100,reference_end=110)
    before=deepcopy(u);w=dict(chrom='chr1',start=100,end=110,strand='-',name='w')
    v=orient_unit(u,w)
    assert u==before and v['positions']==[0,5,8] and v['p_accessible']==[.8,.7,.6]
    assert v['contexts']==[3,2,1] and v['raw_tf_intervals']==[[1,8]] and v['strand']=='GA'
    assert v['native_multi_interval_calls'][0]==dict(interval=[1,8],query_interval=[2,9])
    assert v['m5c_observations']==[False,False,True]


def test_bed_orientation_and_width_are_explicit(tmp_path):
    p=tmp_path/'w.bed';p.write_text('chr1\t0\t100\n')
    assert load_bed(p)[0]['start']==0
    with pytest.raises(ValueError,match='BED6'):load_bed(p,pooled=True)
    p.write_text('chr1\t0\t100\ta\t0\t+\nchr2\t20\t121\tb\t0\t-\n')
    with pytest.raises(ValueError,match='equal widths'):load_bed(p,pooled=True)


def test_pool_deduplicates_global_molecule_and_is_order_independent():
    from test_staged_families import payload
    a=payload();b=deepcopy(a)
    for source in (a,b):
        for i,u in enumerate(source['strata'][0]['units']):u['read_name']=f'movie/{i}/ccs'
    windows=[dict(chrom='chr1',start=0,end=81,strand='+',name='a'),dict(chrom='chr2',start=0,end=81,strand='-',name='b')]
    first=pool_payloads([a,b],windows);second=pool_payloads([b,a],windows[::-1])
    assert first['strata'][0]['units']==second['strata'][0]['units']
    assert first['pooling']['retained_molecules']==6
    assert len(first['pooling']['excluded_repeated_views'])==6


def test_automatic_modes_follow_the_data_but_never_override_explicit_switches():
    # Release audit 2026-09-29: explicit sr/cross.enabled used to be overwritten (and the run mislabelled XCR).
    strata=[dict(dataset_id='a',chemistry='ddda'),dict(dataset_id='b',chemistry='hia5-pacbio')]
    params={'sr':{'enabled':False},'cross':{'enabled':False}}
    result=automatic_parameters(params,strata)
    assert not result['sr']['enabled'] and not result['cross']['enabled']
    auto=automatic_parameters({},strata)
    assert auto['sr']['enabled'] and auto['cross']['enabled']
    assert not automatic_parameters({},[dict(dataset_id='a',chemistry='hia5-pacbio')])['sr']['enabled']
    assert not params['sr']['enabled']


def test_cli_reports_and_strict_resume(tmp_path,monkeypatch):
    from test_staged_families import payload,parameters
    from fiberhmm.inference.consensus.harmonized_families import workflow
    data=payload();opts=parameters(stop_after='native');ev=tmp_path/'evidence.json.gz';params=tmp_path/'p.json'
    write_json(ev,data);write_json(params,opts)
    first=tmp_path/'first';second=tmp_path/'second'
    main(['--evidence',str(ev),'--parameters',str(params),'--output',str(first)])
    assert (first/'families.tsv').exists() and (first/'report.html').exists()
    def forbidden(*a,**k): raise AssertionError('native fitting rerun')
    monkeypatch.setattr(workflow,'fit_source',forbidden)
    main(['--resume',str(first),'--start-at','consolidation','--stop-after','resolved','--consolidation-bp','5','--output',str(second)])
    result=read_json(second/'manifest.json')
    assert result['checkpoints']['hits']['native']==1
    assert result['last_stage']=='resolved'
    assert result['parameters']['families']['physical_radius_bp']==5
    with pytest.raises(ValueError,match='Missing or incompatible'):
        main(['--evidence',str(ev),'--parameters',str(params),'--start-at','consolidation','--cache',str(tmp_path/'missing'),'--output',str(tmp_path/'third')])


def make_bam(path, *, declared=True):
    import pysam
    from fiberhmm.io.bam_header import append_chemistry
    header=pysam.AlignmentHeader.from_dict(dict(HD={'VN':'1.6','SO':'coordinate'},SQ=[{'SN':'chr1','LN':1000}]))
    if declared: header=append_chemistry(header,dict(assay='daf',enzyme='ddda',platform='pacbio',mode='daf'))
    with pysam.AlignmentFile(str(path),'wb',header=header) as handle:
        r=pysam.AlignedSegment(header);r.query_name='molecule1';r.query_sequence='AYGT'*50
        r.reference_id=0;r.reference_start=100;r.mapping_quality=60;r.cigarstring='200M'
        r.set_tag('st','CT');r.set_tag('MA','200;msp:1-200;tf:61-80')
        handle.write(r)
    pysam.index(str(path))


def test_real_bam_metadata_native_replay_and_no_source_mutation(tmp_path):
    from fiberhmm.inference.consensus.bam import load_bam_payload
    from fiberhmm.inference.consensus.parameters import parse_options
    from hashlib import sha256
    p=tmp_path/'input.bam';make_bam(p);before=sha256(p.read_bytes()).hexdigest()
    data=load_bam_payload([dict(dataset_id='one',paths=[str(p)])],dict(chrom='chr1',start=120,end=220),parse_options({'cr':{'engine':'staged_native_families'}}))
    assert data['strata'][0]['chemistry']=='ddda'
    assert data['strata'][0]['units']
    u=data['strata'][0]['units'][0]
    assert 'native_multi_interval_tf_intervals' in u
    assert len(u['positions'])==len(u['hits'])==len(u['p_accessible'])
    assert all(120<=pos<220 for pos in u['positions'])
    assert before==sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='conflicts'):
        load_bam_payload([dict(dataset_id='one',paths=[str(p)],chemistry='dddb')],data['region'])
    missing=tmp_path/'missing.bam';make_bam(missing,declared=False)
    with pytest.raises(ValueError,match='Every BAM'):
        load_bam_payload([dict(dataset_id='one',paths=[str(p),str(missing)])],data['region'])


def test_recall_nested_geometry_has_same_oriented_frame():
    from test_staged_families import payload
    u=payload()['strata'][0]['units'][0]
    u['upstream_nuc_tf_recall']=dict(calls=[dict(interval=[12,51],query_interval=[12,51])],
        excluded_calls=[dict(interval=[60,70])],nucleosomes=[[60,80]],msps=[[0,60]],
        original_tag_msps_absent_from_loaded_scaffold=[[10,20]])
    v=orient_unit(u,dict(chrom='chr1',start=0,end=81,strand='-',name='minus'))
    recall=v['upstream_nuc_tf_recall']
    assert recall['calls'][0]['interval']==[30,69]
    assert recall['calls'][0]['query_interval']==[12,51]
    assert recall['nucleosomes']==[[1,21]] and recall['excluded_calls'][0]['interval']==[11,21]


def test_pooled_two_loci_runs_real_staged_kernel_and_exports_memberships(tmp_path):
    from test_staged_families import payload,parameters
    from fiberhmm.inference.consensus.workflow import run_analysis
    from fiberhmm.inference.consensus.execution import single_threaded_blas
    first=payload();second=deepcopy(first)
    for k,data in enumerate([first,second]):
        for i,u in enumerate(data['strata'][0]['units']):u['read_name']=f'locus{k}_read{i}'
    windows=[dict(chrom='chr1',start=0,end=81,strand='+',name='plus'),dict(chrom='chr2',start=0,end=81,strand='-',name='minus')]
    # Construct the genomic minus locus with the same biological geometry.
    second['strata'][0]['units']=[orient_unit(u,windows[1]) for u in second['strata'][0]['units']]
    data=pool_payloads([first,second],windows)
    before=deepcopy(data)
    with single_threaded_blas(): result=run_analysis(data,parameters(),tmp_path)
    assert data==before and result['manifest']['pooling']['retained_molecules']==12
    assert all(stage['original_calls']==12 for stage in result['stages'])
    assert sum(v['actual_simulations'] for v in result['manifest']['native_timings'].values())>0
    import csv
    with (tmp_path/'calls.tsv').open() as f: rows=list(csv.DictReader(f,delimiter='\t'))
    final=[row for row in rows if row['stage']=='resolved']
    assert len(final)==12
    assert {json.loads(row['window'])['name'] for row in final}=={'plus','minus'}


def test_pool_tracks_amplification_aliases_when_representative_changes():
    from test_staged_families import payload
    first=payload();second=deepcopy(first)
    for data,name in [(first,'PCR_copy_A'),(second,'PCR_copy_B')]:
        u=data['strata'][0]['units'][0];u['read_name']=name
        u['source_members']=[dict(read_name='PCR_copy_A'),dict(read_name='PCR_copy_B')]
        data['strata'][0]['units']=[u]
    windows=[dict(chrom='chr1',start=0,end=81,strand='+',name='one'),dict(chrom='chr2',start=0,end=81,strand='+',name='two')]
    pooled=pool_payloads([first,second],windows)
    assert pooled['pooling']['retained_molecules']==1
    assert len(pooled['pooling']['excluded_repeated_views'])==1
    with pytest.raises(ValueError,match='one payload'):pool_payloads([first],windows)


def test_shared_loader_default_hia5_includes_upstream_recall(tmp_path):
    import pysam
    from array import array
    from fiberhmm.io.bam_header import append_chemistry
    from fiberhmm.inference.consensus.bam import load_bam_payload
    p=tmp_path/'hia5.bam'
    header=pysam.AlignmentHeader.from_dict(dict(HD={'VN':'1.6','SO':'coordinate'},SQ=[{'SN':'chr1','LN':1000}]))
    header=append_chemistry(header,dict(assay='fiber-seq',enzyme='hia5',platform='pacbio',mode='pacbio-fiber'))
    with pysam.AlignmentFile(str(p),'wb',header=header) as bam:
        r=pysam.AlignedSegment(header);r.query_name='movie/123/ccs';r.query_sequence='ACGT'*50
        r.reference_id=0;r.reference_start=100;r.mapping_quality=60;r.cigarstring='200M'
        r.set_tag('MM','A+a.,'+','.join(['0']*50)+';');r.set_tag('ML',array('B',[255]*50))
        r.set_tag('MA','200;msp.:1-200');bam.write(r)
    pysam.index(str(p))
    data=load_bam_payload([dict(dataset_id='h',paths=[str(p)])],dict(chrom='chr1',start=100,end=300))
    assert len(data['strata'][0]['units'])==1
    assert 'upstream_nuc_tf_recall' in data['strata'][0]['units'][0]


def test_xcr_zero_membership_keeps_eligible_coverage_in_report():
    from test_staged_families import payload
    from fiberhmm.inference.consensus.harmonized_families.presentation import browser_snapshot
    source=payload()['strata'][0];source['units']=source['units'][:1]
    absent=deepcopy(source);absent['dataset_id']='absent'
    record=dict(unit_id='test::u0',interval=[12,51],display_hypotheses=['family1'],
        original=dict(compatible_families=['family1'],inference_eligible=True),status='compatible')
    annotation=dict(records=[record],hypotheses=[dict(id='family1',reference_interval=[12,51],display=True,status='native')])
    result=browser_snapshot([(None,annotation)],[source,absent],'XCR','resolved')
    family=result['datasets']['absent']['cr']['catalog'][0]
    assert family['source_units']==0
    assert family['classification_counts']['CT']['eligible_units']==1
    assert family['classification_counts']['CT']['compatible_units']==0
    assert result['cross']['edges']==[]


@pytest.mark.parametrize('pooled',[False,True])
def test_real_bam_bed_cli_exports_indexed_bams_and_reports(tmp_path,pooled):
    from test_consensus_bam_export import fixture
    source,_,_,_=fixture(tmp_path)
    bed=tmp_path/'windows.bed'
    bed.write_text('chr1\t100\t300\tone\t0\t+\nchr1\t400\t600\ttwo\t0\t-\n')
    params=tmp_path/'parameters.json'
    # The fixture BAM has no native replay, so the original calls are classified (correct_native=false): that needs
    # the staged engine. Before 3.0 this ran the lattice recaller, which silently ignored correct_native=false.
    write_json(params,dict(cr=dict(engine='staged_native_families'),input=dict(correct_native=False),compute=dict(cores=1,maximum_matrix_mb=128)))
    out=tmp_path/'run'
    args=['--bam',str(source),'--bed',str(bed),'--parameters',str(params),'--output',str(out)]
    if pooled:args.append('--pool-loci')
    main(args)
    bam=list((out/'bams').glob('*.bam'))
    assert len(bam)==1 and Path(str(bam[0])+'.csi').exists()
    assert (out/'report.html').exists()
    manifest=read_json(out/'manifest.json' if pooled else out/'window_000001'/'manifest.json')
    assert manifest['parameters']['sr']['enabled']
    assert manifest['mode_realized']==('SR' if pooled else 'CR')
    assert manifest['data_warnings']
    assert manifest['datasets'][0]['model']['chemistry_resolution']['files'][0]['source']=='declared_v1'


def test_same_window_physical_duplicates_fail_instead_of_discarding_strands():
    from test_staged_families import payload
    p=payload()
    for i,u in enumerate(p['strata'][0]['units'][:2]):u['read_name']='movie/12/ccs/'+('fwd' if i==0 else 'rev')
    with pytest.raises(ValueError,match='within one window'):
        pool_payloads([p],[dict(chrom='chr1',start=0,end=81,name='one',strand='+')])


def test_reversed_call_metadata_remains_in_interval_order():
    from test_staged_families import payload
    u=payload()['strata'][0]['units'][0]
    u['native_multi_interval_tf_intervals']=[[10,20],[50,70]]
    u['native_multi_interval_calls']=[dict(interval=[10,20],llr=3),dict(interval=[50,70],llr=8)]
    v=orient_unit(u,dict(chrom='chr1',start=0,end=81,name='minus',strand='-'))
    assert [c['interval'] for c in v['native_multi_interval_calls']]==v['native_multi_interval_tf_intervals']
    assert [c['llr'] for c in v['native_multi_interval_calls']]==[8,3]


def test_unset_compute_follows_the_machine_and_explicit_values_win():
    from fiberhmm.inference.consensus.regions import machine_compute_defaults
    from fiberhmm.inference.consensus.parameters import parse_options
    strata=[dict(dataset_id='a',chemistry='ddda')]
    machine=machine_compute_defaults()
    assert 1<=machine['cores']<=16 and 2048<=machine['maximum_matrix_mb']<=32768 and machine['maximum_matrix_mb']%16==0
    auto=automatic_parameters({},strata)['compute']
    assert auto['cores']==machine['cores'] and auto['maximum_matrix_mb']==machine['maximum_matrix_mb']
    assert 'predictive_stopping' not in auto          # default engine: lattice recaller, no Monte Carlo draws
    staged=automatic_parameters({'cr':{'engine':'staged_native_families'}},strata)['compute']
    assert staged['predictive_stopping']=='decision'
    explicit=automatic_parameters({'compute':{'cores':3,'maximum_matrix_mb':4096,'predictive_stopping':'full'}},strata)['compute']
    assert (explicit['cores'],explicit['maximum_matrix_mb'],explicit['predictive_stopping'])==(3,4096,'full')
    parse_options(automatic_parameters({},strata))
