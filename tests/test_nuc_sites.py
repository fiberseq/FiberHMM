"""Planted arrays, honest discovery, state rules, and optional local embryo regression."""
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fiberhmm.nuc.records import NucMolecule
from fiberhmm.nuc.sites import SitesParams, analyse_sites, callable_positions, molecular_dyads, site_state
from fiberhmm.assoc.nucleosome import hash_half


def planted(n=160, seed=21, shift=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        L = 5000
        # DAF-like informative density, with marks outside footprints.
        informative = rng.random(L) < .12
        hits = informative & (rng.random(L) < .6)
        centres = np.r_[np.arange(350, 2200, 190), np.arange(2700, 4700, 190)]
        centres = centres + rng.normal(0, 7, len(centres)) + shift
        nucs = np.column_stack([centres-70, centres+70]).astype(int)
        # Closed molecules have a fuzzy central occluder and random flank phases.
        if i % 4 == 0:
            centres = np.arange(300, 4800, 190) + rng.uniform(-90, 90)
            nucs = np.column_stack([centres-70, centres+70]).astype(int)
            nucs = np.vstack([nucs, [2420, 2580]])
            msp = []
        else:
            msp = [[2250, 2630]]
        for a, b in nucs:
            hits[a:b] = False
        # Enough outer marks for the truncation test, even at low marking density.
        hits[[10, L-10]] = True
        informative |= hits
        out.append(NucMolecule(f'm{i}', L, False, nucs, [[2420, 2440]] if i%4==1 else [], msp,
                               hits, informative, chrom='chrS', ref_start=0, ref_end=L,
                               blocks=np.array([[0, 0, L]])))
    return out


QUICK = SitesParams(n_perm=9, n_boot=19, min_coverage=10, phasing_grid_bp=500)


def test_planted_sites_and_coverage_matched_baseline():
    result = analyse_sites({'a': planted()}, ('chrS', 0, 5000), params=QUICK, phasing=False)
    rows = [r for r in result['rows'] if r['kind']=='site']
    site = min(rows, key=lambda r: abs(r['pos']-2700))
    assert abs(site['pos']-2700) <= 15
    s = site['samples']['a']
    assert s['positioning']['value'] > .65
    assert s['held_out_positioning']['value'] > .6
    assert s['positioning_over_baseline'] > 3
    assert s['baseline_percentile'] > 95
    assert s['held_out_over_baseline'] > 3
    assert s['held_out_percentile'] > 95
    assert 5 < s['fuzziness_sd_bp'] < 25
    assert s['fuzziness_cap_bp'] == 60
    assert len([r for r in result['rows'] if r['kind']=='anchor' and abs(r['pos']-2440)<150]) == 1


def test_discovery_never_uses_evaluation_half():
    ms = planted()
    one = analyse_sites({'a': ms}, ('chrS', 0, 5000), params=QUICK, phasing=False)
    for m in ms:
        if hash_half(m) == 1:
            m.nucs = m.nucs + 65
    two = analyse_sites({'a': ms}, ('chrS', 0, 5000), params=QUICK, phasing=False)
    assert sorted(r['pos'] for r in one['rows']) == sorted(r['pos'] for r in two['rows'])
    chosen = next(r for r in one['rows'] if r['kind']=='site')
    s1 = chosen['samples']['a']
    s2 = next(r for r in two['rows'] if r['pos']==chosen['pos'])['samples']['a']
    assert s1['held_out_positioning']['value'] > s2['held_out_positioning']['value'] + .4


def test_three_states_and_fallback_are_explicit():
    ms = planted(80)
    result = analyse_sites({'a': ms}, ('chrS', 0, 5000), params=QUICK)
    anchor = min((r for r in result['rows'] if r['kind']=='anchor'), key=lambda r: abs(r['pos']-2440))
    s = anchor['samples']['a']
    assert s['states']['open']['value'] == .5
    assert s['states']['bound']['value'] == .25
    assert s['states']['occluded']['value'] == .25
    assert s['occluder_dyad']['n'] == 20
    assert s['minus1'] < anchor['pos'] < s['plus1']
    for f in ('molecule', 'anchor'):
        phase = s['phasing'][f]
        assert phase['background_method'] == 'locus-grid'
        assert phase['matched_backgrounds'] == 0
        assert phase['locus_grid_n'] > 0
        assert phase['ci'] is not None


def test_three_state_rule_and_callable_gaps():
    m = planted(2)[1]
    assert site_state(m, 2435) == 'bound'
    m.msps = np.array([[2470, 2530]])  # 60 bp linker is not an open barrier.
    assert site_state(m, 2500) == 'unclassified'
    m.msps = np.array([[2450, 2550]])
    m.tfs = np.empty((0, 2), int)
    assert site_state(m, 2500) == 'open'
    m.blocks = np.array([[0, 0, 2400], [2400, 2500, 2500]])
    assert not callable_positions(m, [2450])[0]
    m.no_call = np.array([[2400, 2410]])
    assert not callable_positions(m, [2505])[0]


def test_length_rule_excludes_tf_and_truncation_and_can_be_configured():
    m = planted(2)[1]
    m.nucs = np.array([[0, 140], [500, 640], [1000, 1110], [1300, 1550]])
    m.tfs = np.array([[700, 840]])
    assert np.array_equal(molecular_dyads(m), [570, 1055])
    assert np.array_equal(molecular_dyads(m, (130, 170)), [570])


def test_comparisons_chemistry_flag_ci_q_and_pool_dedup():
    ms = planted()
    res = analyse_sites({'a': ms, 'b': planted(seed=24, shift=12)}, ('chrS', 0, 5000),
                        chemistries={'a':'Hia5', 'b':'DddB'}, params=QUICK, phasing=False)
    c = next(r for r in res['rows'] if r['kind']=='site')['comparisons'][0]
    assert c['caveat'] == 'chemistry contrast, not biology'
    assert c['ci'][0] <= c['difference'] <= c['ci'][1]
    assert 0 <= c['q'] <= 1
    assert c['shift']['ci'][0] < 12 < c['shift']['ci'][1]
    doubled = analyse_sites({'a':ms, 'copy':ms}, ('chrS',0,5000), pool_keys={'a':'same','copy':'same'}, params=QUICK, phasing=False)
    assert doubled['discovery']['molecules'] == len(ms)
    assert all(c['p'] is None and c['ci'] is None for r in doubled['rows'] for c in r['comparisons'])


@pytest.mark.parametrize('frame',['molecule','anchor'])
def test_vectorized_association_path_agrees_with_reference(frame):
    from fiberhmm.assoc.condition import LocusTable
    from fiberhmm.assoc.nucleosome import anchor_appearances, vectorized_anchor_appearances, condition_stats
    ms = planted(24)
    # Include reverse molecules, deletions, uneven coverage and no state.
    ms[0].is_reverse = True
    ms[1].blocks = np.array([[0,0,2000],[2000,2100,2900]])
    anchors = [('chrS',q) for q in [500,2100,2435,3200,4700]]
    loci = LocusTable.from_sets(anchors,[],[(i,[]) for i in range(len(anchors))])
    a,bins,_ = anchor_appearances(ms,loci,frame=frame)
    b,fast_bins,_ = vectorized_anchor_appearances(ms,loci,frame=frame)
    # Row order is not a statistical input; match via locus and molecule key.
    # Molecular ids are local, so compare state/density/stats multisets per locus.
    for li in range(len(anchors)):
        assert sum(a.locus==li) == sum(b.locus==li)
        if not sum(a.locus==li):
            continue
        x = np.c_[a.state[a.locus==li],a.mask[a.locus==li],a.density[a.locus==li],a.stats[a.locus==li].reshape(sum(a.locus==li),-1)]
        y = np.c_[b.state[b.locus==li],b.mask[b.locus==li],b.density[b.locus==li],b.stats[b.locus==li].reshape(sum(b.locus==li),-1)]
        x = x[np.lexsort(x.T[::-1])]; y = y[np.lexsort(y.T[::-1])]
        np.testing.assert_allclose(x,y,atol=1e-12)
    assert bins==fast_bins
    summary,summary_bins,_ = vectorized_anchor_appearances(ms,loci,frame=frame,summary_only=True)
    np.testing.assert_array_equal(summary.locus,b.locus)
    np.testing.assert_array_equal(summary.stratum,b.stratum)
    np.testing.assert_array_equal(summary.mask,(b.mask>>(len(bins)-1))&1)
    np.testing.assert_allclose(summary.stats[:,0,:],b.stats[:,-1,:])
    assert summary_bins==bins[-1:]


def test_single_locus_molecule_bootstrap_is_finite_and_explicit():
    from fiberhmm.assoc.nucleosome import condition_stats
    result=condition_stats(planted(80),[('chrS',2435)],backgrounds=None,n_perm=9,n_boot=19,
                           bootstrap_unit='molecule',vectorized=True)
    for frame in result['frames'].values():
        metric=frame['metrics']['R_minus_null']
        assert metric['bootstrap']['mode']=='molecules'
        ci=metric['bins'][-1]['delta_anchor_ci']
        assert np.isfinite(ci['lo']) and np.isfinite(ci['hi'])


def test_callable_bases_and_hits_have_matching_smoothing_at_shared_deletions():
    from fiberhmm.nuc.sites import _bin_bases
    m=planted(2)[1]
    m.blocks=np.array([[0,0,2000],[2000,2100,2900]])
    m.no_call=np.array([[100,120]])
    bp=_bin_bases(m,0,5000,5)
    assert bp[400:420].sum()==0
    assert bp[20:24].sum()==0
    assert bp.sum()==4880
    # A coverage deletion cannot create a dyad-density peak without dyad hits.
    m.nucs=np.empty((0,2),int)
    r=analyse_sites({'a':[m]},('chrS',0,5000),params=replace(QUICK,min_coverage=1),phasing=False)
    assert max(r['track']['density'])==0


@pytest.mark.parametrize('reverse',[False,True])
def test_batched_projection_preserves_deletion_and_outside_rules(reverse):
    m=planted(2)[1]
    m.is_reverse=reverse
    m.blocks=np.array([[0,1000,100],[100,1150,200],[300,1400,300]])
    positions=np.arange(900,1800)
    expected=[]
    for pos in positions:
        q=-1
        for i,(qs,rs,length) in enumerate(m.blocks):
            if rs<=pos<rs+length:
                q=qs+pos-rs
                break
            if rs+length<=pos and i+1<len(m.blocks) and pos<m.blocks[i+1,1]:
                q=m.blocks[i+1,0]
                break
        expected.append(m.length-1-q if reverse and q>=0 else q)
    np.testing.assert_array_equal(m.ref_to_mol(positions),expected)


# Local embryo BAMs (ind_hia5.v3.bam, ind_dddb.v3.bam); the test is skipped unless this names their folder.
EMBRYO = Path(os.environ.get('FIBERHMM_TEST_EMBRYO_DIR', '/nonexistent'))


@pytest.mark.skipif(not all((EMBRYO/f'ind_{c}.v3.bam').exists() for c in ('hia5','dddb')),
                    reason='needs local embryo recall BAMs')
def test_local_embryo_top_barrier():
    import pysam
    from fiberhmm.nuc.records import DecodeContext, decode_record
    observations = {}
    for c in ('hia5','dddb'):
        with pysam.AlignmentFile(str(EMBRYO/f'ind_{c}.v3.bam'), 'rb') as bam:
            ctx = DecodeContext(bam.header)
            ms = []
            for record in bam.fetch('chr3L', 15038000, 15054000):
                if record.flag & (4|256|2048|512|1024):
                    continue
                m, _ = decode_record(record, ctx)
                if m is not None and len(m.nucs):
                    ms.append(m)
            observations[c] = ms
    result = analyse_sites(observations, ('chr3L',15038000,15054000), params=QUICK,
                           chemistries={'hia5':'Hia5', 'dddb':'DddB'})
    anchors = [r for r in result['rows'] if r['kind']=='anchor']
    barrier = min(anchors, key=lambda r: abs(r['pos']-15048368))
    assert abs(barrier['pos']-15048368) < 65
    for c in observations:
        ordered = sorted(anchors, key=lambda r: r['samples'][c]['phasing']['molecule']['delta']
                         if np.isfinite(r['samples'][c]['phasing']['molecule']['delta']) else -np.inf, reverse=True)
        assert barrier in ordered[:5]
        assert barrier['samples'][c]['phasing']['molecule']['delta'] > .08
