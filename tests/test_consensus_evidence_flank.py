"""compute.evidence_flank_bp crops only the analysis copy of whole-molecule interval lists."""
from fiberhmm.inference.consensus.evidence_flank import crop_unit
from fiberhmm.inference.consensus.parameters import parse_options


def _unit():
    return dict(unit_id='u', positions=[1000, 1010], msp_intervals=[[0, 100], [900, 1100], [1700, 1800], [5000, 6000]],
                raw_nuc_intervals=[[200, 347], [1200, 1347]], aligned_blocks=[[0, 10000]],
                native_multi_interval_calls=[dict(interval=[950, 990], llr=7.0), dict(interval=[4000, 4040], llr=6.0)],
                upstream_nuc_tf_recall=dict(nucleosomes=[[200, 347], [1200, 1347]], msps=[[5000, 6000]],
                                            calls=[dict(interval=[950, 990]), dict(interval=[4000, 4040])],
                                            settings=dict(a=1), status='ok'))


def test_crop_keeps_overlapping_intervals_whole():
    u = _unit()
    c = crop_unit(u, 1000, 1100, 500)
    assert c['msp_intervals'] == [[900, 1100]]       # 0-100, 1700-1800, 5000-6000 lie outside 500-1600
    assert c['raw_nuc_intervals'] == [[1200, 1347]]
    assert c['aligned_blocks'] == [[0, 10000]]                        # overlapping: kept whole, never clipped
    assert [x['interval'] for x in c['native_multi_interval_calls']] == [[950, 990]]
    assert c['upstream_nuc_tf_recall']['msps'] == [] and c['upstream_nuc_tf_recall']['settings'] == dict(a=1)
    assert c['positions'] is u['positions']
    # the original unit (payload, digest, evidence.json.gz) is untouched
    assert len(u['msp_intervals']) == 4 and len(u['upstream_nuc_tf_recall']['calls']) == 2


def test_negative_flank_keeps_everything_and_default_is_500():
    u = _unit()
    assert crop_unit(u, 1000, 1100, -1) is u
    assert parse_options({})['compute'].evidence_flank_bp == 500
