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
