"""state_at: live states of transferable feature types (nfr.region) on one record, equal to a scan's."""
from __future__ import annotations

import json

import pytest

pysam = pytest.importorskip('pysam')

from fiberhmm.scan import run_units, state_at
from fiberhmm.scan.identity import record_sha256
from fiberhmm.scan.states import StateReader, nfr_region_state
from fiberhmm.scan.tables import read_all
from fiberhmm.scan.units import plan


@pytest.fixture(scope='module')
def locus_bam(tmp_path_factory):
    from fiberhmm.scan.testing import write_daf_locus_bam
    return write_daf_locus_bam(tmp_path_factory.mktemp('states') / 'locus.bam')


def test_rule_matches_read_gaps():
    nucs = [(100, 247), (277, 424), (600, 747), (777, 924)]       # gaps 30, 176, 30 bp
    assert nfr_region_state(nucs, 450, 550, 60) == '1'            # inside the 176 bp gap
    assert nfr_region_state(nucs, 300, 310, 60) == '0'            # on a nucleosome
    assert nfr_region_state(nucs, 250, 260, 60) == '0'            # a 30 bp linker is not an opening
    assert nfr_region_state(nucs, 420, 430, 60) == '1'            # overlaps the gap's edge
    assert nfr_region_state(nucs, 50, 60, 60) == '.'              # no nucleosome before the region
    assert nfr_region_state(nucs, 900, 950, 60) == '.'            # none after it
    assert nfr_region_state(nucs, 450, 550, 200) == '0'           # a gap below min_gap_bp
    assert nfr_region_state([(100, 247), (400, 480), (600, 747)], 420, 430, 60) == '1'   # an 80 bp call is no nucleosome


def test_state_at_equals_the_scan_states(tmp_path, locus_bam):
    """Every nfr.region state of a scan equals state_at on the same record, and records the scan gives no
    state are '.' (DddA: no nucleosome replay; uncapped)."""
    units = plan({'chr1': 6000}, regions=[('chr1', 0, 6000)])
    run_units([{'paths': [locus_bam]}], units, modules=['nfr'], out=tmp_path / 's', annotate=False)
    man = json.loads((tmp_path / 's' / 'manifest.json').read_text())
    hits = [h for h in read_all(tmp_path / 's' / 'nfr' / 'hits.tsv.bgz') if h['type'] == 'nfr.region']
    states = {(r['hit_id'], r['record']): r['state'] for r in read_all(tmp_path / 's' / 'nfr' / 'states.tsv.bgz')}
    assert hits and states
    n = 0
    with pysam.AlignmentFile(locus_bam) as bam:
        reader = StateReader(bam.header, settings=man['modules']['nfr']['settings'])
        for h in hits:
            seen = {}
            for read in bam.fetch('chr1', int(h['start']), int(h['end'])):
                sha = record_sha256(read)
                occ = seen.get(sha, 0)
                seen[sha] = occ + 1
                want = states.get((h['hit_id'], f'{sha}:{occ}'), '.')
                assert reader.state_at(read, 'nfr.region', 'chr1', int(h['start']), int(h['end'])) == want
                n += 1
        assert n > 20
        # any position: open inside the NFR in open molecules, closed on a nucleosome
        read = next(r for r in bam.fetch('chr1', 2200, 2201))
        got = {reader.state_at(read, 'nfr.region', 'chr1', p) for p in range(1000, 4000, 50)}
        assert got <= {'0', '1', '.'} and '0' in got
        assert state_at(read, 'nfr.region', 'chr1', 2200) == reader.state_at(read, 'nfr.region', 'chr1', 2200)
        assert reader.state_at(read, 'nfr.region', 'chrOther', 2200) == '.'
        with pytest.raises(ValueError):
            reader.state_at(read, 'nfr.variant', 'chr1', 2200)          # not transferable
    with pytest.raises(ValueError):
        StateReader(pysam.AlignmentFile(locus_bam).header, settings={'nfr': {'clip_gaps_to_region': True}})


def test_cache_size_zero_disables_the_cache(locus_bam):
    with pysam.AlignmentFile(locus_bam) as bam:
        read = next(bam.fetch('chr1', 2200, 2201))
        assert StateReader(bam.header, cache_size=0).state_at(read, 'nfr.region', 'chr1', 2200) == \
            StateReader(bam.header).state_at(read, 'nfr.region', 'chr1', 2200)
        with pytest.raises(ValueError):
            StateReader(bam.header, cache_size=-1)
