"""input.evidence_pad_bp / --evidence-pad: loaded reads keep only geometry near the window; crossing calls stay whole."""
from array import array

import numpy as np
import pysam
import pytest

from fiberhmm.inference.evidence_crop import crop_unit_intervals, effective_span, parse_pad
from fiberhmm.inference.strand_rescue import N_CTX, load_region_evidence
from fiberhmm.inference.consensus.parameters import parse_options
from fiberhmm.inference.consensus import memory_estimate as ME


def _bam(tmp_path):
    """One 3 kb DAF-style read on chr1:1000-4000: TF calls far left, crossing the pad boundary and inside the window."""
    header = {'HD': {'SO': 'coordinate'}, 'SQ': [{'SN': 'chr1', 'LN': 10000}]}
    path = tmp_path / 'pad.bam'
    seq = ''.join('Y' if i % 7 == 0 else 'C' for i in range(3000))
    with pysam.AlignmentFile(path, 'wb', header=header) as bam:
        read = pysam.AlignedSegment(bam.header)
        read.query_name = 'r'; read.query_sequence = seq; read.flag = 0
        read.reference_id = 0; read.reference_start = 1000; read.mapping_quality = 60
        read.cigarstring = '3000M'
        # 1-based MA starts: tf at ref 1100-1120 (far), 1880-1960 (crosses lo=1900), 2050-2070 (window);
        # nuc 2400-2547 (beyond hi=2300 + nothing), msp 1700-2400 (covers window, crosses both edges)
        read.set_tag('st', 'CT')
        read.set_tag('MA', '3000;tf+:101-20,881-80,1051-20;nuc+:1401-147;msp+:701-700')
        bam.write(read)
    pysam.index(str(path))
    return str(path)


def _load(path, pad):
    return load_region_evidence(path, 'chr1', 2000, 2200, strand_mode='daf', mode='daf', context_size=3,
                                prob_threshold=None, llr_hit=np.zeros(N_CTX), llr_miss=np.zeros(N_CTX), min_mapq=20,
                                ma_annotation_frame='seq', evidence_pad=pad)[0]


def test_pad_keeps_crossing_calls_whole_and_drops_far_ones(tmp_path):
    path = _bam(tmp_path)
    full, cropped = _load(path, None), _load(path, 100)
    assert [(c.start, c.end) for c in full.tfs] == [(1100, 1120), (1880, 1960), (2050, 2070)]
    # window 2000-2200 +/- 100 = 1900-2300: the call starting at 1880 crosses lo and is kept unclipped
    assert [(c.start, c.end) for c in cropped.tfs] == [(1880, 1960), (2050, 2070)]
    assert [(c.start, c.end) for c in cropped.msps] == [(1700, 2400)]
    assert cropped.nucs == []
    # lattice evidence (window only) is identical; whole-read-only fields are not retained
    assert np.array_equal(full.positions, cropped.positions) and np.array_equal(full.steps, cropped.steps)
    assert cropped.molecular_tfs is None and cropped.cigar_tuples is None and full.molecular_tfs is not None
    assert cropped.alignment_blocks == full.alignment_blocks == ((1000, 4000),)
    # DAF duplicate fingerprints stay whole-read
    assert np.array_equal(full.fingerprint_positions, cropped.fingerprint_positions)


def test_pad_zero_still_keeps_every_call_overlapping_the_window(tmp_path):
    cropped = _load(_bam(tmp_path), 0)
    assert [(c.start, c.end) for c in cropped.tfs] == [(2050, 2070)]
    assert [(c.start, c.end) for c in cropped.msps] == [(1700, 2400)]


def test_crop_unit_intervals_and_span():
    unit = dict(raw_nuc_intervals=[[0, 147], [900, 1047], [3000, 3147]], msp_intervals=[[147, 900], [1047, 3000]],
                native_multi_interval_calls=[dict(interval=[100, 120]), dict(interval=[1200, 1230])],
                upstream_nuc_tf_recall=dict(nucleosomes=[[0, 147], [3000, 3147]], calls=[dict(interval=[1200, 1230])],
                                            settings=dict(a=1)))
    crop_unit_intervals(unit, 1000, 1500)
    assert unit['raw_nuc_intervals'] == [[900, 1047]]
    assert unit['msp_intervals'] == [[1047, 3000]]
    assert [c['interval'] for c in unit['native_multi_interval_calls']] == [[1200, 1230]]
    assert unit['upstream_nuc_tf_recall']['nucleosomes'] == [] and unit['upstream_nuc_tf_recall']['settings'] == dict(a=1)
    assert effective_span(1000, 1500, [(900, 1047), (1047, 3000), (5, 10)]) == (900, 3000)


def test_api_default_is_uncropped():
    """FiberBrowser and other library callers keep whole-read evidence unless they ask for a pad."""
    from fiberhmm.inference.consensus.bam import evidence_pad
    from fiberhmm.inference.consensus.harmonized_families.workflow import prepare_sources
    from test_consensus_lattice_recaller import planted_payload
    options = parse_options({'cr': {'engine': 'lattice_recaller'}})
    assert options['input'].evidence_pad_bp == -1 and evidence_pad(options) is None
    payload = planted_payload(20)
    for u in payload['strata'][0]['units']:
        u['msp_intervals'] = [[0, 100], [880, 1520], [5000, 6000]]
    source = prepare_sources(payload, options)[0]
    assert all(u['msp_intervals'] == [[0, 100], [880, 1520], [5000, 6000]] for u in source['units'])
    # an explicit pad crops the analysis copy only; the payload keeps what was loaded
    padded = parse_options({'cr': {'engine': 'lattice_recaller'}, 'input': {'evidence_pad_bp': 250}})
    source = prepare_sources(payload, padded)[0]
    assert all(u['msp_intervals'] == [[880, 1520]] for u in source['units'])
    assert payload['strata'][0]['units'][0]['msp_intervals'] == [[0, 100], [880, 1520], [5000, 6000]]


def test_cli_default_pad_is_250(tmp_path):
    import json
    from fiberhmm.inference.consensus import cli
    from fiberhmm.inference.consensus.artifacts import write_json
    from test_consensus_lattice_recaller import planted_payload
    write_json(tmp_path/'evidence.json.gz', planted_payload(40))
    for extra, expected in (([], 250), (['--evidence-pad', 'full'], -1), (['--evidence-pad', '500'], 500)):
        out = tmp_path/f'out{expected}'
        cli.main(['--evidence', str(tmp_path/'evidence.json.gz'), '--cores', '1', '--no-bam', '--output', str(out), *extra])
        manifest = json.loads((out/'manifest.json').read_text())
        assert manifest['parameters']['input']['evidence_pad_bp'] == expected
    assert cli.CLI_EVIDENCE_PAD_BP == 250


def test_option_and_cli_values():
    assert parse_pad('full') is None and parse_pad(-1) is None and parse_pad('250') == 250
    with pytest.raises(ValueError):
        parse_pad('wide')


def test_memory_estimate_scales_and_fits():
    small = ME.estimate(1000, 600, 6, 20000, 20000)
    large = ME.estimate(100000, 600, 6, 20000, 20000)
    assert large['evidence_gb'] > small['evidence_gb'] and large['task_molecules'] == 20000
    assert ME.fit_cores(small['total_gb'], 1000, 600, 6, discovery_cap=20000, quantification_cap=20000) == 6
    assert ME.fit_cores(0.01, 1000, 600, 6) == 0
