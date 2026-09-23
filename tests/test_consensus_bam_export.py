from copy import deepcopy
from pathlib import Path
import array
import hashlib

import pysam
import pytest

from fiberhmm.io.bam_header import append_chemistry
from fiberhmm.io.ma_tags import parse_ma_tag, parse_aq_array, parse_an_tag
from fiberhmm.inference.consensus.bam_export import export_bams, CONTRACT, FAMILY
from fiberhmm.inference.consensus.artifacts import digest


def fixture(tmp_path, name='a', reverse=False, chrom='chr1'):
    path=tmp_path/(name+'.bam')
    header=pysam.AlignmentHeader.from_dict({'HD':{'VN':'1.6','SO':'coordinate'},'SQ':[{'SN':chrom,'LN':1000}]})
    header=append_chemistry(header,dict(assay='daf',enzyme='ddda',platform='pacbio',mode='daf'))
    with pysam.AlignmentFile(str(path),'wb',header=header) as bam:
        for i in range(2):
            read=pysam.AlignedSegment(header);read.query_name=name+str(i);read.query_sequence='AYGT'*50
            read.reference_id=0;read.reference_start=100+300*i;read.mapping_quality=60;read.cigarstring='200M'
            read.flag=16 if reverse else 0
            read.set_tag('st','CT')
            read.set_tag('MA','200;msp.:1-200;tf.QQQ:61-20');read.set_tag('AQ',array.array('B',[80,20,30]))
            read.set_tag('AN','original_msp,original_tf');bam.write(read)
    pysam.index(str(path))
    with pysam.AlignmentFile(str(path),'rb') as bam:
        read=next(bam);sha=hashlib.sha256(read.to_string().encode()).hexdigest()
        original=read.to_string()
    span=[220,240] if reverse else [160,180]
    unit=dict(unit_id='u',read_name=name+'0',positions=list(range(100,300)),
        source_members=[dict(read_name=name+'0',library_id=str(path),record_sha256=sha,alignment_occurrence=0)],
        native_multi_interval_calls=[dict(interval=span,llr=8,opportunities=20)])
    payload=dict(region=dict(chrom=chrom,start=100,end=300),strata=[dict(dataset_id=name,chemistry='ddda',units=[unit])],
        input_files=[dict(dataset_id=name,path=str(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)])
    result=dict(final_stage='resolved',manifest=dict(input_digest=digest(payload),parameters={'cross':{'enabled':False}},display_mode='SR'),
        datasets={name:dict(cr=dict(records=[dict(unit_id=name+'::u',proposals=[dict(source_interval=span,compatible_families=['compact','alternative'])])]))})
    return path,payload,result,original


@pytest.mark.parametrize('reverse',[False,True])
def test_family_tags_roundtrip_preserves_native_annotations_and_all_memberships(tmp_path,reverse):
    source,payload,result,original=fixture(tmp_path,reverse=reverse);before=source.read_bytes()
    rows=export_bams([(result,payload)],tmp_path/'output',scope='full')
    assert len(rows)==1 and rows[0]['annotations']==2
    with pysam.AlignmentFile(rows[0]['bam'],'rb') as bam:
        records=list(bam.fetch('chr1',100,700));comments=bam.header.to_dict()['CO']
        read=records[0];parsed=parse_ma_tag(read.get_tag('MA'))
        assert read.get_tag('MA').startswith('200;msp.:1-200;tf.QQQ:61-20;tf_consensus.QQQQQ:')
        assert parsed['raw_types'][-1][3]==[(60,20),(60,20)]
        assert list(read.get_tag('AQ'))[:3]==[80,20,30]
        extra=list(read.get_tag('AQ'))[3:]
        assert sorted([extra[:5],extra[5:]])==[[80,1,0,20,0],[80,2,0,20,0]]
        names=parse_an_tag(read.get_tag('AN'))
        assert names[:2]==['original_msp','original_tf'] and len(set(names[2:]))==2
        assert records[1].get_tag('MA')=='200;msp.:1-200;tf.QQQ:61-20'
        assert all(r.has_tag('RG') for r in records)
        assert any(c.startswith(CONTRACT) for c in comments)
        assert len([c for c in comments if c.startswith(FAMILY)])==2
    assert source.read_bytes()==before and Path(rows[0]['index']).is_file()


def test_export_grouping_tracks_current_merged_view_and_file_option(tmp_path):
    a,pa,ra,_=fixture(tmp_path,'a');b,pb,rb,_=fixture(tmp_path,'b')
    analyses=[(ra,pa),(rb,pb)]
    groups=[dict(dataset_id='merged_view',paths=[str(a),str(b)])]
    merged=export_bams(analyses,tmp_path/'merged',dataset_groups=groups,scope='full')
    assert len(merged)==1 and merged[0]['annotations']==4
    with pysam.AlignmentFile(merged[0]['bam'],'rb') as bam:
        reads=list(bam.fetch());assert len(reads)==4
        assert len({r.get_tag('RG') for r in reads})==2
        assert [r.reference_start for r in reads]==[100,100,400,400]
    separate=export_bams(analyses,tmp_path/'separate',grouping='files',dataset_groups=groups,scope='full')
    assert len(separate)==2
    with pytest.raises(ValueError,match='new or empty'):export_bams(analyses,tmp_path/'merged')


def test_pooled_minus_call_returns_to_original_genomic_and_molecular_coordinates(tmp_path):
    source,payload,result,_=fixture(tmp_path)
    unit=payload['strata'][0]['units'][0]
    unit['genomic_provenance']=dict(window=dict(chrom='chr1',start=100,end=300,strand='-',name='minus'))
    unit['positions']=list(range(200));unit['native_multi_interval_calls'][0]['interval']=[120,140]
    result['datasets']['a']['cr']['records'][0]['proposals'][0]['source_interval']=[120,140]
    result['manifest']['parameters']['cross']['enabled']=True
    rows=export_bams([(result,payload)],tmp_path/'out')
    with pysam.AlignmentFile(rows[0]['bam'],'rb') as bam:
        parsed=parse_ma_tag(next(bam).get_tag('MA'))
        assert parsed['raw_types'][-1][0]=='tf_cross_consensus'
        assert parsed['raw_types'][-1][3]==[(60,20),(60,20)]


def test_changed_source_or_unmatched_record_fails_without_publishing_bam(tmp_path):
    _,payload,result,_=fixture(tmp_path)
    payload['input_files'][0]['size']+=1
    with pytest.raises(ValueError,match='changed'):export_bams([(result,payload)],tmp_path/'out')
    assert not list((tmp_path/'out').glob('*.bam'))
    payload['input_files'][0]['size']-=1
    payload['strata'][0]['units'][0]['source_members'][0]['record_sha256']='missing'
    with pytest.raises(ValueError,match='did not match'):export_bams([(result,payload)],tmp_path/'other')
    assert not list((tmp_path/'other').glob('*.bam'))


def test_subset_default_retains_unassigned_reads_and_deduplicates_windows(tmp_path):
    source,payload,result,_=fixture(tmp_path)
    # Two disjoint windows overlap the same long alignment; retain it once.
    payload['pooling']={'windows':[dict(chrom='chr1',start=100,end=180),dict(chrom='chr1',start=200,end=250)]}
    rows=export_bams([(result,payload)],tmp_path/'subset')
    with pysam.AlignmentFile(rows[0]['bam'],'rb') as bam:
        reads=list(bam);assert len(reads)==1
        assert reads[0].cigarstring=='200M'
    assert rows[0]['export_scope']=='regions' and rows[0]['written_alignments']==1
    # No family assignments is not grounds to remove an overlapping read.
    result['datasets']['a']['cr']['records']=[]
    rows=export_bams([(result,payload)],tmp_path/'unassigned')
    with pysam.AlignmentFile(rows[0]['bam'],'rb') as bam:
        reads=list(bam);assert len(reads)==1
        assert 'tf_consensus' not in reads[0].get_tag('MA')


def test_rerun_refreshes_owned_layers_and_generated_rg(tmp_path):
    _,payload,result,_=fixture(tmp_path)
    first=export_bams([(result,payload)],tmp_path/'first')[0]
    path=Path(first['bam'])
    with pysam.AlignmentFile(str(path),'rb') as bam:
        read=next(bam);sha=hashlib.sha256(read.to_string().encode()).hexdigest()
    payload['input_files']=[dict(dataset_id='a',path=str(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)]
    payload['strata'][0]['units'][0]['source_members'][0].update(library_id=str(path),record_sha256=sha)
    result['manifest']['input_digest']='new_run'
    second=export_bams([(result,payload)],tmp_path/'second')[0]
    with pysam.AlignmentFile(second['bam'],'rb') as bam:
        read=next(bam);header=bam.header.to_dict()
        assert len(header['RG'])==1
        assert read.get_tag('RG')==header['RG'][0]['ID']
        assert len(parse_an_tag(read.get_tag('AN')))==4
        assert read.get_tag('MA').count('tf_consensus.')==1
        assert sum(c.startswith(CONTRACT) for c in header['CO'])==1
        assert sum(c.startswith(FAMILY) for c in header['CO'])==2


def test_explicit_unknown_source_is_not_silently_reassigned(tmp_path):
    _,payload,result,_=fixture(tmp_path)
    payload['strata'][0]['units'][0]['source_members'][0]['library_id']=str(tmp_path/'wrong.bam')
    with pytest.raises(ValueError,match='source BAM'):
        export_bams([(result,payload)],tmp_path/'out')


def test_insertion_at_native_boundary_keeps_original_molecular_span():
    from fiberhmm.inference.consensus.bam_export import _project_to_molecule
    read=pysam.AlignedSegment();read.query_sequence='A'*200;read.reference_start=100
    read.cigarstring='60M2I138M';read.set_tag('MA','200;tf.QQQ:61-20')
    assert _project_to_molecule(read,[160,178],200)==(60,20)


def test_chr_aliases_share_family_slot_coordinate_space(tmp_path):
    _,payload,result,_=fixture(tmp_path,chrom='1')
    other=deepcopy(payload);other['region']['chrom']='chr1'
    r2=deepcopy(result);r2['manifest']['input_digest']='second'
    from fiberhmm.inference.consensus.bam_export import assignment_plan
    _,families,_=assignment_plan([(result,payload),(r2,other)])
    assert {f['chrom'] for f in families}=={'1'}
    assert len({f['fi'] for f in families})==4


def test_native_library_rg_survives_rerun_and_has_source_provenance(tmp_path):
    source,payload,result,_=fixture(tmp_path)
    with pysam.AlignmentFile(str(source),'rb') as bam:
        hd=bam.header.to_dict();reads=list(bam)
    hd['RG']=[dict(ID='library',SM='sample',LB='original_library',DS='original description')]
    with pysam.AlignmentFile(str(source),'wb',header=hd) as bam:
        for read in reads: read.set_tag('RG','library');bam.write(read)
    pysam.index(str(source))
    with pysam.AlignmentFile(str(source),'rb') as bam: sha=hashlib.sha256(next(bam).to_string().encode()).hexdigest()
    payload['input_files'][0].update(size=source.stat().st_size,mtime_ns=source.stat().st_mtime_ns)
    payload['strata'][0]['units'][0]['source_members'][0]['record_sha256']=sha
    rows=export_bams([(result,payload)],tmp_path/'out')
    with pysam.AlignmentFile(rows[0]['bam'],'rb') as bam:
        assert next(bam).get_tag('RG')=='library'
        rg=next(r for r in bam.header.to_dict()['RG'] if r['ID']=='library')
        assert rg['SM']=='sample' and rg['LB']=='original_library'
        assert rg['DS']=='original description; FiberHMM source BAM: '+str(source)


def test_same_global_family_catalog_can_merge_distinct_window_bounds():
    from fiberhmm.inference.consensus.bam_export import read_family_catalog,FAMILY
    import json
    a=dict(layer='tf_consensus',annotation_name='fhcr_global',chrom='chr1',
        family_key='F1',input_digest='model',stage='resolved',start=100,end=120)
    b=dict(a,start=1000,end=1040)
    header={'CO':[FAMILY+json.dumps(v) for v in [a,b]]}
    catalog=read_family_catalog(header)
    assert catalog['families']==[dict(a,end=1040)]
    b['family_key']='different'
    with pytest.raises(ValueError,match='Conflicting'):
        read_family_catalog({'CO':[FAMILY+json.dumps(v) for v in [a,b]]})


def test_strand_quality_byte_and_family_strand_resolution(tmp_path):
    import json, math
    source,payload,result,_=fixture(tmp_path)
    unit=payload['strata'][0]['units'][0]
    unit['p_accessible']=[.6]*200;unit['p_protected']=[.05]*200
    resolution=dict(trusted_strand='CT',core_resolution='resolved')
    result['datasets']['a']['cr']['catalog']=[
        dict(family='compact',consensus_start=160,consensus_end=180,strand_resolution=resolution),
        dict(family='alternative',consensus_start=150,consensus_end=150.5)]
    rows=export_bams([(result,payload)],tmp_path/'output',scope='full')
    with pysam.AlignmentFile(rows[0]['bam'],'rb') as bam:
        read=next(bam.fetch('chr1',100,300));comments=bam.header.to_dict()['CO']
    extra=list(read.get_tag('AQ'))[3:];names=parse_an_tag(read.get_tag('AN'))[2:]
    step=math.log1p(-.05)-math.log1p(-.6)
    catalog=[json.loads(c[len(FAMILY):]) for c in comments if c.startswith(FAMILY)]
    by_family={f['family_key']:f for f in catalog}
    sq={f['family_key']:extra[5*names.index(f['annotation_name'])+4] for f in catalog}
    assert sq['compact']==min(255,1+round(10*20*step)) and sq['alternative']==1+round(10*step)
    assert by_family['compact']['strand_resolution']=={'a':resolution}
    assert 'strand_resolution' not in by_family['alternative']
    contract=json.loads(next(c for c in comments if c.startswith(CONTRACT))[len(CONTRACT):])
    assert contract['quality_names']==['tq','fi','fq','op','sq']
