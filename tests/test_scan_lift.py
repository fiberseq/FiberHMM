"""Scan MA groups (state marks and whole-molecule scalars) survive fiberhmm-lift --map.

Needs graphlift's liftbam, gbz2chain and pathseq (``FIBERHMM_TEST_GRAPHLIFT`` or
PATH), like ``test_graph_lift_native.py``, whose toy graph and source records
this test reuses. A source BAM is annotated by fiberhmm-scan (planted features
with per-record states, and per-record scalars), lifted MAT -> PAT (strand
kept) and MAT -> GRCh38RC (every read flips), and every lifted record's scan
groups are read back and compared with its source record's.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pysam
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_graph_lift_native import GBZ2CHAIN, LIFTBAM, PATHSEQ, _fasta, _map, _source_bam  # noqa: E402

pytestmark = pytest.mark.skipif(not (LIFTBAM and GBZ2CHAIN and PATHSEQ),
                                reason="needs graphlift liftbam, gbz2chain and pathseq")

REG = {'plant': 'fiberhmm.scan.testing:PlantedAnchorModule',
       'molly': 'fiberhmm.scan.testing:PlantedMoleculeModule'}


def _groups_by_record(path):
    from fiberhmm.io.scan_ma import scan_contracts, scan_groups_of
    out = {}
    with pysam.AlignmentFile(str(path)) as bam:
        contracts = scan_contracts(bam.header)
        for read in bam.fetch(until_eof=True):
            g = scan_groups_of(read, contracts)
            out[(read.query_name, read.is_supplementary)] = (read, g)
    return contracts, out


@pytest.fixture(scope='module')
def annotated_and_lifted(tmp_path_factory):
    from fiberhmm.cli import lift
    from fiberhmm.scan import plan, run_units
    tmp = tmp_path_factory.mktemp('scan_lift')
    pat_map, _pat = _map(tmp, 'HG002_PAT#0', 'pat')
    rc_map, rc = _map(tmp, 'GRCh38RC#0', 'rc')
    mat_size = next(s['size'] for s in rc['sources'] if s['name'] == 'chr1_MATERNAL')
    mat = _fasta('HG002_MAT#0#chr1_MATERNAL', 'chr1_MATERNAL', mat_size, tmp / 'mat.fa')
    src = _source_bam(tmp, mat)
    units = plan({'chr1_MATERNAL': mat_size}, genome=True)
    features = [['f1', 10_500, 10_700, 'both'], ['f2', 20_100, 20_300, 'both'], ['f3', 50_500, 51_000, 'both'],
                ['f4', 120_200, 120_400, 'both']]
    scan = tmp / 'scan'
    run_units([{'dataset_id': 'mat', 'paths': [str(src)]}], units, modules=['plant', 'molly'],
              params={'plant': {'features': features, 'states': True}}, out=scan, annotate=True, registry=REG)
    manifest = json.loads((scan / 'manifest.json').read_text())
    annotated = scan / manifest['outputs']['annotated'][0]['path']
    lifted = {}
    for target, m in (('HG002_PAT', pat_map), ('GRCh38RC', rc_map)):
        out = tmp / f'{target}.bam'
        assert lift.main(['-i', str(annotated), '-o', str(out), '--map', str(m), '--liftbam', LIFTBAM, '-t', '2',
                          '--unlifted', str(tmp / f'{target}.unlifted.bam')]) == 0
        lifted[target] = out
    return annotated, lifted


@pytest.mark.parametrize('target, flipped', [('HG002_PAT', False), ('GRCh38RC', True)])
def test_scan_groups_survive_lift(annotated_and_lifted, target, flipped):
    annotated, lifted = annotated_and_lifted
    src_contracts, source = _groups_by_record(annotated)
    dst_contracts, dest = _groups_by_record(lifted[target])
    assert set(src_contracts) == set(dst_contracts) == {'plantst', 'mollymol'}
    assert src_contracts == dst_contracts                     # header contract lines carried by the lift
    assert dest, 'nothing lifted'
    marked = 0
    for key, (read, groups) in dest.items():
        s_read, s_groups = source[key]
        assert read.is_reverse == (s_read.is_reverse != flipped)
        # whole-molecule scalars: identical values on every lifted record
        assert [g['scalars'] for g in groups['mollymol']] == [g['scalars'] for g in s_groups['mollymol']]
        # state marks: same hits and bytes; molecular intervals move only by a hard-clip fill
        offset = 0
        if s_read.cigartuples and s_read.cigartuples[0][0] == 5 and not s_read.is_reverse:
            offset = s_read.cigartuples[0][1]
        want = sorted((g['name'], g['start'] + offset, g['length'], tuple(g['q'])) for g in s_groups.get('plantst', []))
        got = sorted((g['name'], g['start'], g['length'], tuple(g['q'])) for g in groups.get('plantst', []))
        assert got == want, key
        marked += bool(got)
        whole = groups['mollymol'][0]
        assert whole['length'] == s_groups['mollymol'][0]['length']
    assert marked >= 3
