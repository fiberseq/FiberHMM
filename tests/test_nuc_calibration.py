"""Thinning calibration (NUC P-CAL): emission rescale, outcome matching, edge
noise, and calibrate_thin end to end on a synthetic called BAM."""
from __future__ import annotations

import numpy as np
import pysam
import pytest

from nuc_synth import flat_calibration, header, make_record, planted_record_molecule

from fiberhmm.nuc.calibration import NucCalibration
from fiberhmm.nuc.molecule import molecule_outputs
from fiberhmm.nuc.recall import CallerSettings, rescale_emissions, thin_fiber_read
from fiberhmm.nuc.thinning import (KEEP_GRID, calibrate_thin, edge_sigma, match_outcomes,
                                   quantise_keep, supported_range)


def test_rescale_emissions_scales_hits_and_keeps_context_mass(benchmark_model_path):
    from fiberhmm.core.model_io import load_model
    m = load_model(benchmark_model_path)
    r = rescale_emissions(m, 0.4)
    E0, E1 = np.asarray(m.emissionprob_), np.asarray(r.emissionprob_)
    np.testing.assert_allclose(E1[:, :4096], 0.4 * E0[:, :4096])
    np.testing.assert_allclose(E1[:, :4096] + E1[:, 4097:8193], E0[:, :4096] + E0[:, 4097:8193])
    np.testing.assert_allclose(E1.sum(axis=1), E0.sum(axis=1))
    np.testing.assert_allclose(rescale_emissions(m, 1.0).emissionprob_, m.emissionprob_)
    with pytest.raises(ValueError):
        rescale_emissions(m, 0.0)


def test_thin_fiber_read_keeps_the_expected_fraction():
    fr = {'m6a_query_positions': set(range(0, 20_000, 2)), 'query_sequence': 'A' * 20_000}
    t = thin_fiber_read(fr, 0.3, np.random.default_rng(0))
    assert abs(len(t['m6a_query_positions']) / 10_000 - 0.3) < 0.02
    assert t['m6a_query_positions'] <= fr['m6a_query_positions']
    assert len(fr['m6a_query_positions']) == 10_000          # input untouched


def test_quantise_keep_grid():
    assert quantise_keep(1.0) == 1.0
    q = quantise_keep(0.37)
    assert q in KEEP_GRID and abs(np.log(q) - np.log(0.37)) < 0.04


def test_match_outcomes_classes():
    dense = np.array([[100, 250], [300, 450], [500, 650], [700, 850], [1000, 1300]])
    thin = np.array([[102, 251],            # kept, shifted edges
                     [300, 650],            # merges dense 2 and 3
                     [1000, 1120], [1180, 1300]])   # splits dense 5; dense 4 dropped
    out = match_outcomes(dense, thin)
    assert [o[0] for o in out] == ['kept', 'merged', 'merged', 'dropped', 'split']
    # a split whose larger piece still covers half the footprint is a split, not kept
    assert match_outcomes(np.array([[100, 300]]), np.array([[100, 220], [230, 300]]))[0][0] == 'split'
    assert out[0][2:] == (2.0, 1.0, -1.0)
    assert out[1][1] == 2


def test_edge_sigma_is_rms_of_central_shifts():
    v = np.r_[np.zeros(80), np.full(10, 4.0), np.full(10, -4.0)]
    assert abs(edge_sigma(v) - np.sqrt(0.2 * 16)) < 1e-9
    assert edge_sigma(np.zeros(50)) == 0.0


def test_low_information_floor_and_support_reach_molecule_outputs():
    rng = np.random.default_rng(2)
    from nuc_synth import array_molecule
    m = array_molecule([('reg', 20, 190)], rng, mark_rate=0.5)
    cal = flat_calibration(lambda_floor=0.9)
    assert molecule_outputs(m, cal=cal).flags['low_information']
    cal = flat_calibration(supported=[0.95, 1.0])
    r = molecule_outputs(m, cal=cal)
    assert (r.footprints['coarse'][~r.footprints['truncated']] == 0).all()   # all can't tell


@pytest.fixture(scope='module')
def called_bam(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('nuc_thin')
    rng = np.random.default_rng(9)
    path = str(tmp / 'called.bam')
    with pysam.AlignmentFile(path, 'wb', header=header()) as out:
        for i in range(36):
            rate = 0.25 + 0.6 * (i % 6) / 5
            seq, nucs, msps, marks = planted_record_molecule(rng, n_nuc=12, mark_rate=rate)
            out.write(make_record(out, f'm{i:02d}', seq, 1000 + i * 3000, nucs, msps, (), marks,
                                  reverse=bool(i % 3 == 0)))
    pysam.index(path)
    return path


def test_calibrate_thin_end_to_end(called_bam):
    cal = calibrate_thin(called_bam, label='synthetic', n_population=36, seeds=1, n_targets=4, cores=1)
    assert cal.source == 'thin'
    assert cal.caller['enzyme'] == 'hia5' and cal.caller['mode'] == 'pacbio-fiber'
    assert cal.outcomes['n'] and abs(sum(cal.outcomes[k][0] for k in ('kept', 'merged', 'split', 'dropped')) - 1) < 1e-9
    assert cal.validation['caller_errors'] == 0
    assert cal.validation['caller_runs'] > 36
    if cal.supported is not None:
        assert cal.supported[0] < cal.supported[1]
    rt = NucCalibration.from_dict(cal.to_dict())
    assert rt.digest() == cal.digest()
    # deterministic: same inputs, same record
    again = calibrate_thin(called_bam, label='synthetic', n_population=36, seeds=1, n_targets=4, cores=1)
    strip = lambda c: {k: v for k, v in c.to_dict().items() if k not in ('notes', 'validation')}  # noqa: E731
    assert strip(again) == strip(cal)


def test_supported_range_ends_at_the_dense_references():
    rng = np.random.default_rng(3)
    nat = rng.uniform(0.1, 0.7, 5000)
    thin = rng.uniform(0.05, 0.45, 50_000)         # many thinned footprints
    dense = rng.uniform(0.45, 0.7, 2000)
    lo, hi = supported_range(nat, thin, dense)
    assert abs(lo - np.quantile(nat, 0.025)) < 1e-9
    assert hi > 0.68                                # not cut at the thinned pool's top
    assert supported_range(nat, [], dense) is None
    empty = supported_range([0.1, 0.2] * 50, [0.3, 0.4] * 50, [0.5, 0.6] * 50)
    assert empty[1] <= empty[0]
    from fiberhmm.nuc.calibration import NucCalibration as _C
    c = _C(supported=empty)
    assert not c.is_supported([0.0001, 0.15, 0.35, 0.9]).any()


def test_daf_thinning_turns_dropped_conversions_into_misses():
    from fiberhmm.core.bam_reader import encode_from_query_sequence
    seq = 'A' * 50 + 'T' + 'A' * 50                 # CT strand: T at 50 is a converted C
    fr = {'query_sequence': seq, 'm6a_query_positions': {50}, '_daf_strand': '+'}
    t = thin_fiber_read(fr, 0.0, np.random.default_rng(0))
    assert t['query_sequence'][50] == 'C' and not t['m6a_query_positions']
    enc0 = encode_from_query_sequence(seq, {50}, 0, mode='daf', strand='+')
    enc1 = encode_from_query_sequence(t['query_sequence'], set(), 0, mode='daf', strand='+')
    assert enc0[50] < 4096 and 4097 <= enc1[50] < 8193      # hit -> miss, not non-target
    ga = {'query_sequence': 'C' * 20 + 'A' + 'C' * 20, 'm6a_query_positions': {20}, '_daf_strand': '-'}
    assert thin_fiber_read(ga, 0.0, np.random.default_rng(0))['query_sequence'][20] == 'G'


def test_caller_settings_resolve_ddda_cpg_masking():
    hdr = {'PG': [{'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call',
                   'CL': 'fiberhmm-call -i a.bam -o b.bam --enzyme ddda'}]}
    s = CallerSettings.from_header(hdr)
    assert s.enzyme == 'ddda' and s.cpg_mask_policy == 'unmethylated-only'
    off = CallerSettings.from_header({'PG': [{'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call',
                                              'CL': 'fiberhmm-call --enzyme ddda --no-use-m5c'}]})
    assert off.cpg_mask_policy is None
    ds = CallerSettings.from_header({'PG': [{'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call',
                                             'CL': 'fiberhmm-call --enzyme ddda',
                                             'DS': 'cpg_mask=methylated-only hmm_cpg_mask=on'}]})
    assert ds.cpg_mask_policy == 'methylated-only'
    assert CallerSettings.from_header({'PG': []}).cpg_mask_policy is None     # Hia5


def test_caller_settings_take_the_declared_chemistry_and_daf_run_mask():
    daf = {'CO': ['FIBERHMM-CHEMISTRY:v1:assay=daf;enzyme=ddda;platform=nanopore;mode=daf'],
           'PG': [{'ID': 'fiberhmm-apply', 'PN': 'fiberhmm-apply', 'CL': 'fiberhmm-apply -i a.bam'}]}
    s = CallerSettings.from_header(daf)
    assert s.enzyme == 'ddda' and s.mode == 'daf' and s.daf_mask_runs == 2
    ont = {'CO': ['FIBERHMM-CHEMISTRY:v1:assay=fiber-seq;enzyme=hia5;platform=nanopore;mode=nanopore-fiber'],
           'PG': [{'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call', 'CL': 'fiberhmm-call -i a.bam --enzyme hia5'}]}
    o = CallerSettings.from_header(ont)
    assert o.mode == 'nanopore-fiber' and o.prob_threshold == 248
    with pytest.raises(ValueError):
        CallerSettings.from_header({'CO': ['FIBERHMM-CHEMISTRY:v1:assay=daf;enzyme=custom;platform=unknown;mode=daf']})


def test_calibrate_thin_refuses_an_empty_sample(tmp_path):
    path = str(tmp_path / 'empty.bam')
    with pysam.AlignmentFile(path, 'wb', header=header()) as out:
        pass
    pysam.index(path)
    with pytest.raises(ValueError):
        calibrate_thin(path, n_population=10, seeds=1, n_targets=2)


def test_named_calls_only_for_single_particle_footprints_and_weights_are_fitted():
    from fiberhmm.nuc.molecule import footprint_calls
    from fiberhmm.nuc.records import make_molecule
    from fiberhmm.nuc.sizes import AbsoluteClasses, fit_named_weights
    model = AbsoluteClasses(T={'hexasome': 110.0, 'octasome': 147.0, 'chromatosome': 167.0},
                            b_log_lambda=[np.log(0.5)], b=[0.0], sigma_b=[3.0],
                            sigma_struct={'hexasome': 5.0, 'octasome': 5.0, 'chromatosome': 5.0})
    nucs = [(300, 447), (700, 950), (1300, 1660), (2000, 2110)]
    m = make_molecule(4000, nucs=nucs, msps=[(0, 300), (447, 700), (950, 1300), (1660, 2000), (2110, 4000)],
                      hits=[299, 448, 699, 951, 1299, 1661, 1999, 2111])
    cal = flat_calibration()
    cal.source, cal.absolute = 'absolute', model
    fp = footprint_calls(m, cal=cal)
    named = dict(zip(fp['length'].tolist(), fp['named'].tolist()))
    assert named[147] == 3 and named[110] == 2           # octasome, hexasome
    assert named[250] == 0 and named[360] == 0           # long / multi: never named
    rng = np.random.default_rng(0)
    L = np.r_[rng.normal(147, 5, 800), rng.normal(110, 5, 200)]
    w = fit_named_weights(model, L, np.full(len(L), 0.5))
    assert abs(w['octasome'] - 0.8) < 0.05 and abs(w['hexasome'] - 0.2) < 0.05
