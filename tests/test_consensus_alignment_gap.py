"""Corrected native calls and alignment gaps: the frozen veto and the declared allowance."""
from fiberhmm.inference.consensus.adapter import reference_gap_inside
from fiberhmm.inference.consensus.parameters import parse_options, parameter_schema


def test_gap_measures_unaligned_reference_inside_a_call():
    domains = [(100, 160), (161, 170), (170, 200)]      # one-base deletion at 160; 170 is adjacent, not a gap
    assert reference_gap_inside(130, 165, domains) == 1
    assert reference_gap_inside(100, 160, domains) == 0
    assert reference_gap_inside(150, 190, domains) == 1
    assert reference_gap_inside(0, 100, domains) == 100  # nothing aligned there at all


def test_default_allowance_is_the_frozen_veto():
    options = parse_options()
    assert options['input'].native_maximum_alignment_gap_bp == 0
    controls = {c['name']: c for c in parameter_schema()['input']}
    assert controls['native_maximum_alignment_gap_bp']['default'] == 0
    assert parse_options({'input': {'native_maximum_alignment_gap_bp': 5}})['input'].native_maximum_alignment_gap_bp == 5
