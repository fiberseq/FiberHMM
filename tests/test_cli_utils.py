from fiberhmm.cli.utils import _target_bases_for_transfer_mode


def test_target_bases_for_transfer_mode():
    assert _target_bases_for_transfer_mode('pacbio-fiber') == ['A']
    assert _target_bases_for_transfer_mode('nanopore-fiber') == ['A']
    assert _target_bases_for_transfer_mode('daf') == ['C', 'G']
