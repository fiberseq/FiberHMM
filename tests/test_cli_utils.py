import numpy as np

from fiberhmm.cli.utils import (
    _scale_emission_probabilities,
    _target_bases_for_transfer_mode,
)


def test_target_bases_for_transfer_mode():
    assert _target_bases_for_transfer_mode('pacbio-fiber') == ['A']
    assert _target_bases_for_transfer_mode('nanopore-fiber') == ['A']
    assert _target_bases_for_transfer_mode('daf') == ['C', 'G']


def test_scale_emission_probabilities_scales_one_state_and_clips():
    emission = np.array([
        [0.2, 0.4],
        [0.6, 0.8],
    ])

    adjusted, before, after = _scale_emission_probabilities(emission, 1, 2.0)

    np.testing.assert_allclose(emission, [[0.2, 0.4], [0.6, 0.8]])
    np.testing.assert_allclose(adjusted, [[0.2, 0.4], [1.0, 1.0]])
    np.testing.assert_allclose(before, (0.6, 0.8, 0.7))
    np.testing.assert_allclose(after, (1.0, 1.0, 1.0))


def test_scale_emission_probabilities_scales_all_states():
    emission = np.array([
        [0.2, 0.4],
        [0.6, 0.8],
    ])

    adjusted, before, after = _scale_emission_probabilities(emission, None, 0.5)

    np.testing.assert_allclose(adjusted, [[0.1, 0.2], [0.3, 0.4]])
    np.testing.assert_allclose(before, (0.2, 0.8, 0.5))
    np.testing.assert_allclose(after, (0.1, 0.4, 0.25))
