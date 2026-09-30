"""Reverse-strand frame of fibertools and FiberHMM footprint tags (real fixtures).

``tests/fixtures/fibertools_frame`` holds one forward and one reverse read over
the same synthetic 8 kb, tagged by the real tools (generator:
FiberBrowser ``tests/fixtures/fibertools_frame/make_fixtures.py``):

* ``ft0.13_addnuc_fire.bam`` -- fibertools-rs 0.13.0 ``add-nucleosomes`` +
  ``fire``: only ``Ma``/``Aq`` (``nuc.``, ``msp.``, ``fire.Q``).
* ``ft0.13_predict_m6a.bam`` -- 0.13.0 ``predict-m6a`` on random kinetics:
  MM/ML plus ``Ma``.
* ``ft0.6.2_addnuc_fire.bam`` -- 0.6.2 ``add-nucleosomes`` + ``fire``: legacy
  ``ns/nl/as/al/aq`` with no frame marker.
* ``fiberhmm3.0.0_call.bam`` -- ``fiberhmm-call --enzyme hia5 --seq pacbio``.

``expected.json`` has fibertools' own reference-coordinate conversion
(``ft extract -r``) of every fixture's nucleosomes and MSPs, so the reference
position of each footprint is known without trusting FiberHMM's frame logic.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pysam
import pytest

from fiberhmm.io.annotation_frame import (
    append_coord_to_ds,
    legacy_tag_frame,
    merged_input_frame,
)
from fiberhmm.io.ma_tags import (
    annotation_tags,
    fibertools_ma_intervals,
    parse_aq_array,
    parse_ma_tag,
)

FX = Path(__file__).parent / 'fixtures' / 'fibertools_frame'
EXPECTED = json.loads((FX / 'expected.json').read_text())
FIXTURES = sorted(EXPECTED['fixtures'])
REPO = Path(__file__).resolve().parents[1]


def _reference_nucs(read):
    """Output nucleosomes on the reference: FiberHMM output is molecular frame
    (it declares coord=molecular) and the fixtures are 8000M at position 0."""
    if read.has_tag('MA'):
        intervals = parse_ma_tag(read.get_tag('MA'))['nuc']
    else:
        intervals = list(zip(read.get_tag('ns'), read.get_tag('nl')))
    length = read.query_length
    out = []
    for start, size in intervals:
        start, size = int(start), int(size)
        if read.is_reverse:
            start = length - (start + size)
        out.append([start + read.reference_start, start + size + read.reference_start])
    return sorted(out)


def _run(tool_args, tmp_path, name, extra=()):
    out = tmp_path / f'{name}.out.bam'
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(REPO), env.get('PYTHONPATH')]))
    proc = subprocess.run(
        [sys.executable, '-m', 'fiberhmm.cli.recall_tfs', '-i', str(FX / name),
         '-o', str(out), '--enzyme', 'hia5', '--seq', 'pacbio', '-c', '1',
         *tool_args, *extra],
        capture_output=True, text=True, env=env)
    return proc, out


# --------------------------------------------------------------------------- #
#  frame resolver                                                               #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name', FIXTURES)
def test_every_fixture_resolves_molecular(name):
    with pysam.AlignmentFile(str(FX / name)) as bam:
        frame, reason = legacy_tag_frame(bam.header)
    assert frame == 'molecular', reason


def _pg(**fields):
    return fields


@pytest.mark.parametrize('pgs, expected', [
    # fibertools-rs nucleosome writers, current and old command names
    ([_pg(ID='ft', PN='fibertools-rs', CL='ft predict-m6a in.bam out.bam')], 'molecular'),
    ([_pg(ID='ft', PN='fibertools-rs', CL='ft m6a -t 8 in.bam out.bam')], 'molecular'),
    ([_pg(ID='ft', PN='fibertools-rs', CL='ft predict in.bam out.bam')], 'molecular'),
    ([_pg(ID='ft', PN='fibertools-rs', CL='ft add in.bam out.bam')], 'molecular'),
    ([_pg(ID='ft', PN='fibertools-rs', CL='/opt/bin/ft add-nucleosomes in.bam out.bam')], 'molecular'),
    ([_pg(ID='ft', PN='fibertools-rs', CL='ft fire in.bam out.bam')], 'molecular'),
    # a fibertools command that does not write ns/nl is not evidence
    ([_pg(ID='ft', PN='fibertools-rs', CL='ft extract in.bam --all x')], None),
    # no provenance at all: FiberHMM <= 2.12 (SEQ) looks exactly like this
    ([_pg(ID='pbmm2', PN='pbmm2', CL='pbmm2 align ref in.bam out.bam')], None),
    ([], None),
    # a FiberHMM pass-through record without a coord token keeps the frame
    ([_pg(ID='ft', PN='fibertools-rs', CL='ft add-nucleosomes a b'),
      _pg(ID='fiberhmm-dedup', PN='fiberhmm-dedup', DS='DAF duplicate marking')], 'molecular'),
    ([_pg(ID='fiberhmm-dedup', PN='fiberhmm-dedup', DS='DAF duplicate marking')], None),
    # explicit FiberHMM declarations
    ([_pg(ID='fiberhmm-dedup', PN='fiberhmm-dedup', DS='x; coord=seq (footprint tags)')], 'seq'),
    ([_pg(ID='ft', PN='fibertools-rs', CL='ft add-nucleosomes a b'),
      _pg(ID='fiberhmm-tag-m5c', PN='fiberhmm-tag-m5c', DS='x; coord=seq')], 'seq'),
    # a path mentioning fiberhmm is not a FiberHMM program record
    ([_pg(ID='ft', PN='fibertools-rs', CL='ft add-nucleosomes a b'),
      _pg(ID='samtools', PN='samtools', CL='samtools sort -o ~/fiberhmm_work/x.bam')], 'molecular'),
])
def test_legacy_tag_frame_rule(pgs, expected):
    assert legacy_tag_frame({'PG': pgs})[0] == expected


def test_coord_molecular_declaration_anywhere_wins():
    assert legacy_tag_frame({'CO': ['fiberhmm:coord=molecular']})[0] == 'molecular'
    assert legacy_tag_frame({'PG': [_pg(ID='x', PN='x', DS='coord=molecular')]})[0] == 'molecular'


def test_pass_through_ds_token_is_honest():
    assert append_coord_to_ds('DAF dedup', 'molecular').startswith('DAF dedup; coord=molecular')
    assert 'coord=seq' in append_coord_to_ds('DAF dedup', 'seq')
    assert append_coord_to_ds('DAF dedup', None) == 'DAF dedup'
    mol = {'CO': ['fiberhmm:coord=molecular']}
    assert merged_input_frame([mol, mol]) == 'molecular'
    assert merged_input_frame([mol, {}]) is None


# --------------------------------------------------------------------------- #
#  Ma grammar: same spec family as MA, one parser                               #
# --------------------------------------------------------------------------- #

def test_fibertools_ma_parses_with_the_ma_parser():
    with pysam.AlignmentFile(str(FX / 'ft0.13_addnuc_fire.bam')) as bam:
        reads = {r.query_name: r for r in bam}
    fwd = reads['read_fwd']
    ma, aq, _an, source = annotation_tags(fwd)
    assert source == 'Ma'
    parsed = parse_ma_tag(ma)
    names = [(name, strand, spec) for name, strand, spec, _ in parsed['raw_types']]
    assert names == [('nuc', '.', ''), ('msp', '.', ''), ('fire', '.', 'Q')]
    # 1-based start + length on the forward read == ft extract's reference
    start, size = parsed['nuc'][0]
    assert [start, start + size] == EXPECTED['fixtures']['ft0.13_addnuc_fire.bam']['nuc']['read_fwd'][0]
    # Aq holds exactly one byte per fire.Q element
    per = parse_aq_array(aq, [t[2] for t in parsed['raw_types']],
                         [len(t[3]) for t in parsed['raw_types']])
    fire_q = [q for q, t in zip(per, [t for t in parsed['raw_types'] for _ in t[3]]) if t[0] == 'fire']
    assert [q[0] for q in fire_q] == list(aq)
    # the helper reports molecular nuc/msp intervals, unchanged by alignment strand
    rev = fibertools_ma_intervals(reads['read_rev'])
    assert rev['nuc'][0] == (391, 155)


def test_fiberhmm_ma_wins_over_fibertools_ma():
    with pysam.AlignmentFile(str(FX / 'fiberhmm3.0.0_call.bam')) as bam:
        read = next(iter(bam))
    assert annotation_tags(read)[3] == 'MA'
    assert fibertools_ma_intervals(read) is None


# --------------------------------------------------------------------------- #
#  recall-tfs / recall-nucs on fibertools and FiberHMM BAMs                      #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name', FIXTURES)
def test_recall_tfs_auto_keeps_every_reverse_nucleosome_in_place(tmp_path, name):
    """recall-tfs keeps >= 90 bp nucleosomes, so its output must reproduce the
    input's nucleosomes exactly where fibertools puts them on the reference.
    Before the fix the unmarked 0.6.2 BAM was read as SEQ frame (reverse read
    mirrored: 391, 546, ... instead of 300, 453, ...) and the 0.13 Ma-only
    BAMs lost every nucleosome."""
    proc, out = _run([], tmp_path, name)
    assert proc.returncode == 0, proc.stderr
    assert 'input frame: molecular' in proc.stderr
    with pysam.AlignmentFile(str(out)) as bam:
        assert 'coord=molecular' in str(bam.header)
        got = {r.query_name: _reference_nucs(r) for r in bam}
    want = EXPECTED['fixtures'][name]['nuc']
    assert set(got) == set(want)
    for read_name, intervals in want.items():
        # Short (< 90 bp) nucleosomes a TF call overlaps are demoted; every
        # other one must come back at fibertools' own reference position.
        kept = [iv for iv in intervals if iv[1] - iv[0] >= 90]
        assert kept and all(iv in got[read_name] for iv in kept), read_name
        assert all(iv in intervals for iv in got[read_name]), read_name


@pytest.mark.parametrize('name', ['ft0.6.2_addnuc_fire.bam', 'ft0.13_addnuc_fire.bam'])
def test_recall_nucs_reverse_read_matches_forward_read(tmp_path, name):
    """Both reads carry identical molecules, so after nucleosome recall the
    reverse read's nucleosomes land on the true m6A-free gaps like the forward
    read's (a mirrored input sends several onto accessible DNA)."""
    proc, out = _run(['--recall-nucs'], tmp_path, name)
    assert proc.returncode == 0, proc.stderr
    gaps = EXPECTED['truth']['gaps']
    with pysam.AlignmentFile(str(out)) as bam:
        for read in bam:
            nucs = _reference_nucs(read)
            on_gap = [any(min(b, g[1]) - max(a, g[0]) > 0.5 * (b - a) for g in gaps) for a, b in nucs]
            assert nucs and all(on_gap), (read.query_name, nucs)


def test_recall_refuses_when_provenance_is_ambiguous(tmp_path):
    """A header with no frame evidence (a FiberHMM <= 2.12 BAM looks like this;
    so does a fibertools BAM whose @PG history was lost) stops the run with a
    message naming both explicit choices, instead of guessing SEQ."""
    src = FX / 'ft0.6.2_addnuc_fire.bam'
    stripped = tmp_path / 'noprov.bam'
    with pysam.AlignmentFile(str(src)) as bam:
        header = bam.header.to_dict()
        header.pop('PG', None)
        with pysam.AlignmentFile(str(stripped), 'wb', header=header) as out:
            for read in bam:
                out.write(read)
    pysam.index(str(stripped))
    out = tmp_path / 'x.bam'
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(REPO), env.get('PYTHONPATH')]))
    base = [sys.executable, '-m', 'fiberhmm.cli.recall_tfs', '-i', str(stripped), '-o', str(out),
            '--enzyme', 'hia5', '--seq', 'pacbio', '-c', '1']
    proc = subprocess.run(base, capture_output=True, text=True, env=env)
    assert proc.returncode != 0
    assert '--input-frame query' in proc.stderr and '--input-frame molecular' in proc.stderr
    assert not out.exists()
    # the explicit choice runs, and molecular reproduces fibertools' placement
    proc = subprocess.run(base + ['--input-frame', 'molecular'], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    with pysam.AlignmentFile(str(out)) as bam:
        got = {r.query_name: _reference_nucs(r) for r in bam}
    assert got == EXPECTED['fixtures']['ft0.6.2_addnuc_fire.bam']['nuc']
