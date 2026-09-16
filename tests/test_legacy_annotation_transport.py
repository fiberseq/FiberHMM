from array import array
from copy import deepcopy
import numpy as np
import pysam
import pytest
from fiberhmm.inference.legacy_annotations import legacy_annotations
from fiberhmm.inference.strand_rescue import _mapped_annotations, _molecular_annotation_intervals


def read(reverse=False):
    r=pysam.AlignedSegment();r.query_name='legacy';r.query_sequence='A'*100
    r.flag=16 if reverse else 0;r.reference_id=0;r.reference_start=100;r.cigarstring='100M'
    r.set_tag('as',array('i',[10,60]));r.set_tag('al',array('i',[20,10]))
    r.set_tag('ns',array('i',[30]));r.set_tag('nl',array('i',[25]))
    return r


@pytest.mark.parametrize('reverse',[False,True])
def test_verified_seq_frame_is_not_flipped_on_reverse_reads(reverse):
    r=read(reverse);before=r.to_string() if r.header else str(r)
    parsed=legacy_annotations(r,'seq')
    assert [a['start'] for a in parsed['msp']]==[10,60]
    mapped=_mapped_annotations(r,'msp',np.arange(100)+100,parsed_annotations=parsed)
    assert [(v.start,v.end) for v in mapped]==[(110,130),(160,170)]
    assert (r.to_string() if r.header else str(r))==before


def test_explicit_molecular_frame_differs_and_ma_always_wins():
    r=read(True)
    assert [a['start'] for a in legacy_annotations(r,'molecular')['msp']]==[70,30]
    r.set_tag('MA','authoritative-even-if-malformed')
    assert legacy_annotations(r,'seq') is None


def test_missing_frame_and_corrupt_tag_pairs_fail_loudly():
    r=read()
    with pytest.raises(ValueError,match='explicit annotation frame'):legacy_annotations(r,'disabled')
    r.set_tag('al',array('i',[20]))
    with pytest.raises(ValueError,match='Mismatched'):legacy_annotations(r,'seq')
    r.set_tag('al',array('i',[20,1000]))
    with pytest.raises(ValueError,match='outside'):legacy_annotations(r,'seq')
