"""DAF evidence order (3.3) in the insert-consensus pre-pass: a carrier with
MD (C->T) that kept basecaller 6mA marks (Dorado A+a) is a CT read, as the
callers read it; md_first=False reproduces the former recall-tfs order, which
took the basecaller's A marks as G->A deaminations. Complements
tests/test_daf_insert_consensus.py::test_prepass_uses_the_reference_when_mm_has_no_ml,
whose read gives the same strand under both orders.
"""
from __future__ import annotations

import inspect

import pytest

from fiberhmm.daf import insert_consensus as ic
from test_daf_evidence_order import _read  # noqa: E402  (tests/ is on sys.path under pytest)


@pytest.mark.parametrize("mm", [("A+a,0,0,0,0;", [250] * 4), ("A+a?,0,0,0;", [250] * 3)])
def test_prepass_reads_md_before_basecaller_mm(mm):
    read = _read(mm=mm)
    assert ic._read_strand(read) == ic.STRAND_CT                    # the default order: reference first
    assert ic._read_strand(read, md_first=True) == ic.STRAND_CT
    assert ic._read_strand(read, md_first=False) == ic.STRAND_GA    # the former order differs on this read


def test_every_prepass_entry_point_defaults_to_md_first():
    for name, fn in inspect.getmembers(ic, inspect.isfunction):
        p = inspect.signature(fn).parameters.get("md_first")
        if p is not None:
            assert p.default is True, name
