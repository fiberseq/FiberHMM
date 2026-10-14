"""--cpg-mask-policy resolution shared by the DddA CpG-aware commands."""
import pytest

from fiberhmm.inference.tf_recaller import resolve_cpg_mask_policy

POLICY = 'unmethylated-only'


def test_cpg_mask_policy_resolution(capsys):
    assert resolve_cpg_mask_policy(None, None, 'ddda', 'daf') == POLICY
    assert resolve_cpg_mask_policy(None, 'methylated-only', 'ddda', 'daf') == 'methylated-only'
    assert resolve_cpg_mask_policy(False, None, 'ddda', 'daf') is None
    assert capsys.readouterr().err == ''
    # A policy that cannot apply is reported, not silently dropped.
    assert resolve_cpg_mask_policy(None, 'methylated-only', 'hia5', 'pacbio-fiber',
                                   tool='fiberhmm-call') is None
    assert 'has no effect' in capsys.readouterr().err
    assert resolve_cpg_mask_policy(False, POLICY, 'ddda', 'daf') is None
    assert '--no-use-m5c' in capsys.readouterr().err
    with pytest.raises(ValueError):
        resolve_cpg_mask_policy(True, None, 'hia5', 'pacbio-fiber')
