from array import array
import numpy as np
import pysam
import pytest

from fiberhmm.cli.extract_tags import _parse_all_ma_annotations
from fiberhmm.inference.strand_rescue import load_region_evidence, N_CTX
from fiberhmm.io.annotation_frame import ma_annotation_frame


@pytest.mark.parametrize('frame', ['seq', 'molecular'])
@pytest.mark.parametrize('reverse', [False, True])
def test_auto_frame_projects_annotations_and_preserves_molecular_replay(tmp_path, frame, reverse):
    header={'HD':{'SO':'coordinate'},'SQ':[{'SN':'chr1','LN':1000}]}
    if frame=='molecular':header['CO']=['coord=molecular']
    path=tmp_path/'frame.bam'
    with pysam.AlignmentFile(path,'wb',header=header) as bam:
        read=pysam.AlignedSegment(bam.header)
        read.query_name='r';read.query_sequence='C'*50+'Y'+'C'*49;read.flag=16 if reverse else 0
        read.reference_id=0;read.reference_start=100;read.mapping_quality=60
        read.cigarstring='5S35M3I57M'
        read.set_tag('st','CT');read.set_tag('MA','100;tf+QQQ:11-20;msp+:1-100')
        read.set_tag('AQ',array('B',[200,40,90]));bam.write(read)
        qstart=70 if reverse and frame=='molecular' else 10
        positions=read.get_reference_positions(full_length=True)[qstart:qstart+20]
        expected=[p for p in positions if p is not None]
        annotations=_parse_all_ma_annotations(read,annotation_frame=frame)
        assert annotations['tf'][0]['start']==qstart
        assert annotations['tf'][0]['quals']==([200,90,40] if reverse and frame=='molecular' else [200,40,90])
    pysam.index(str(path));diagnostics={}
    reads=load_region_evidence(str(path),'chr1',90,210,strand_mode='daf',mode='daf',
        context_size=3,prob_threshold=None,llr_hit=np.zeros(N_CTX),llr_miss=np.zeros(N_CTX),
        min_mapq=20,ma_annotation_frame='auto',load_diagnostics=diagnostics)
    assert diagnostics['ma_annotation_frame']==frame
    assert [(c.start,c.end) for c in reads[0].tfs]==[(min(expected),max(expected)+1)]
    assert reads[0].molecular_tfs==((100-qstart-20 if reverse else qstart,20),)


def test_header_frame_policy():
    assert ma_annotation_frame({})=='seq'
    assert ma_annotation_frame({'PG':[{'CL':'caller coord=molecular'}]})=='molecular'
