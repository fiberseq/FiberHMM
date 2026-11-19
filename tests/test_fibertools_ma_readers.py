"""Readers of fibertools (>= 0.13) ``Ma``/``Aq``/``An`` annotations.

fibertools-rs 0.13 writes its nucleosomes, MSPs and FIRE elements only as
``Ma``/``Aq`` (molecular frame, no ``ns/nl/as/al``). Every FiberHMM reader that
falls back from ``MA`` must read ``Ma`` too, and a FiberHMM writer must not
leave a stale ``Ma`` beside the ``MA`` it writes.

Fixtures: ``tests/fixtures/fibertools_frame`` (one forward and one reverse
read). The ``Ma`` variants of the FiberHMM fixture are made in-test by renaming
``MA/AQ/AN`` to ``Ma/Aq/An`` and dropping the header's MA-TYPES line.
"""
from __future__ import annotations

import os
import subprocess
import sys
from array import array
from pathlib import Path

import pysam
import pytest

from fiberhmm.io.ma_tags import parse_aq_array, parse_ma_tag

FX = Path(__file__).parent / 'fixtures' / 'fibertools_frame'
FT_FIRE = FX / 'ft0.13_addnuc_fire.bam'
FH_CALL = FX / 'fiberhmm3.0.0_call.bam'
REPO = Path(__file__).resolve().parents[1]
LEGACY = ('ns', 'nl', 'as', 'al', 'nq', 'aq')


def _as_fibertools_ma(src, out, keep_legacy):
    """Copy ``src`` with FiberHMM MA/AQ/AN renamed to fibertools Ma/Aq/An."""
    with pysam.AlignmentFile(str(src)) as bam:
        header = bam.header.to_dict()
        header['CO'] = [c for c in header.get('CO', []) if not c.startswith('MA-TYPES')]
        with pysam.AlignmentFile(str(out), 'wb', header=header) as dst:
            for read in bam:
                if not keep_legacy:
                    for tag in LEGACY:
                        if read.has_tag(tag):
                            read.set_tag(tag, None)
                for old, new in (('MA', 'Ma'), ('AQ', 'Aq'), ('AN', 'An')):
                    if read.has_tag(old):
                        value, kind = read.get_tag(old, with_value_type=True)
                        read.set_tag(old, None)
                        read.set_tag(new, value, value_type=kind if kind == 'Z' else None)
                dst.write(read)
    pysam.index(str(out))
    return out


def _fire_rows(ma, aq):
    """``[(start, length, quals), ...]`` of the fire.Q group of an MA/Ma value."""
    types = parse_ma_tag(ma)['raw_types']
    rows = parse_aq_array(aq, [t[2] for t in types], [len(t[3]) for t in types])
    out, cursor = [], 0
    for name, _strand, _spec, intervals in types:
        for start, length in intervals:
            if name == 'fire':
                out.append((start, length, tuple(rows[cursor])))
            cursor += 1
    return out


# --------------------------------------------------------------------------- #
#  recall-tfs / call: preserve Ma groups, drop the stale Ma                     #
# --------------------------------------------------------------------------- #

def test_recall_tfs_keeps_fibertools_fire_and_drops_stale_ma(tmp_path):
    """recall-tfs regenerates nuc/msp/tf; ft's fire.Q group (with its Aq
    byte) must be carried into the new MA unchanged, in molecular frame on the
    reverse read too, and no Ma may be left beside MA."""
    with pysam.AlignmentFile(str(FT_FIRE)) as bam:
        want = {r.query_name: _fire_rows(r.get_tag('Ma'), r.get_tag('Aq')) for r in bam}
    assert all(want.values())
    out = tmp_path / 'o.bam'
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(REPO), env.get('PYTHONPATH')]))
    proc = subprocess.run(
        [sys.executable, '-m', 'fiberhmm.cli.recall_tfs', '-i', str(FT_FIRE), '-o', str(out),
         '--enzyme', 'hia5', '--seq', 'pacbio', '--force-seq', '-c', '1'],
        capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    with pysam.AlignmentFile(str(out)) as bam:
        reads = list(bam)
    assert {r.is_reverse for r in reads} == {False, True}
    for read in reads:
        assert not any(read.has_tag(t) for t in ('Ma', 'Aq', 'An')), read.query_name
        got = _fire_rows(read.get_tag('MA'), read.get_tag('AQ'))
        assert got == want[read.query_name], read.query_name


def test_preserved_groups_read_fibertools_ma():
    from fiberhmm.inference.tf_recaller import _preserved_ma_groups
    with pysam.AlignmentFile(str(FT_FIRE)) as bam:
        read = next(r for r in bam if r.is_reverse)
    groups, _names, aq = _preserved_ma_groups(read)
    assert [g.partition(':')[0] for g in groups] == ['fire.Q']
    assert aq == list(read.get_tag('Aq'))


def test_strip_stale_call_tags_drops_fibertools_ma():
    """call/apply rewrite the footprints, so ft's Ma/Aq/An (nuc/msp/fire,
    never m5C groups) go, while a kept DddA m5C MA group stays."""
    from fiberhmm.inference.read_filters import strip_stale_call_tags
    with pysam.AlignmentFile(str(FT_FIRE)) as bam:
        read = next(iter(bam))
    read.set_tag('An', ','.join('.' for _ in range(5)))
    strip_stale_call_tags(read)
    assert not any(read.has_tag(t) for t in ('Ma', 'Aq', 'An'))

    with pysam.AlignmentFile(str(FT_FIRE)) as bam:
        read = next(iter(bam))
    read.set_tag('MA', '8000;nuc.Q:101-150;ddda_mcg.:400-20')
    read.set_tag('AQ', array('B', [9]))
    strip_stale_call_tags(read, keep_m5c_groups=True)
    assert read.get_tag('MA') == '8000;ddda_mcg.:400-20'
    assert not read.has_tag('Ma') and not read.has_tag('Aq')


# --------------------------------------------------------------------------- #
#  fiberhmm-qc footprint lengths                                                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('keep_legacy', [False, True])
def test_qc_footprint_lengths_read_fibertools_ma(tmp_path, keep_legacy):
    """Ma-only reads gave 0 nucleosomes/TFs; Ma + legacy gave 0 TFs (nl won)."""
    from fiberhmm.qc.core import _footprint_lengths
    variant = _as_fibertools_ma(FH_CALL, tmp_path / 'ma.bam', keep_legacy)
    with pysam.AlignmentFile(str(FH_CALL)) as bam:
        want = {r.query_name: _footprint_lengths(r) for r in bam}
    with pysam.AlignmentFile(str(variant)) as bam:
        got = {r.query_name: _footprint_lengths(r) for r in bam}
    assert got == want
    assert all(nuc and tf and tagged for nuc, tf, tagged in got.values())


def test_qc_footprint_lengths_on_real_fibertools_output():
    from fiberhmm.qc.core import _footprint_lengths
    with pysam.AlignmentFile(str(FT_FIRE)) as bam:
        for read in bam:
            nuc, tf, tagged = _footprint_lengths(read)
            assert tagged and nuc == [n for _s, n in parse_ma_tag(read.get_tag('Ma'))['nuc']]
            assert tf == []   # FIRE elements are not footprints


# --------------------------------------------------------------------------- #
#  fiberhmm-footprint-model BAM loader                                          #
# --------------------------------------------------------------------------- #

def test_footprint_model_loads_fibertools_ma(tmp_path):
    """The same calls as Ma (molecular by definition, no MA-TYPES line) load
    the same molecules, TF observations and MSPs as FiberHMM's MA (the reverse
    read included); before, every Ma read was skipped as missing MA."""
    from fiberhmm.io.footprint_bam import load_footprint_molecules_from_bam
    variant = _as_fibertools_ma(FH_CALL, tmp_path / 'ma.bam', keep_legacy=False)
    want = load_footprint_molecules_from_bam(FH_CALL)
    got = load_footprint_molecules_from_bam(variant)
    assert got.diagnostics.skipped_missing_ma == 0
    assert got.diagnostics.emitted_molecules == want.diagnostics.emitted_molecules == 2
    assert [(m.molecule_id, m.tfs, m.msps) for m in got.molecules] == \
        [(m.molecule_id, m.tfs, m.msps) for m in want.molecules]
    assert all(m.tfs and m.msps for m in got.molecules)

    native = load_footprint_molecules_from_bam(FT_FIRE)
    assert native.diagnostics.emitted_molecules == 2
    assert all(m.msps and not m.tfs for m in native.molecules)
