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


def _chain(*pgs):
    """@PG records linked by PP, as every real tool writes them."""
    out = []
    for pg in pgs:
        pg = dict(pg)
        if out:
            pg['PP'] = out[-1]['ID']
        out.append(pg)
    return out


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
    (_chain(_pg(ID='ft', PN='fibertools-rs', CL='ft add-nucleosomes a b'),
      _pg(ID='fiberhmm-dedup', PN='fiberhmm-dedup', DS='DAF duplicate marking')), 'molecular'),
    ([_pg(ID='fiberhmm-dedup', PN='fiberhmm-dedup', DS='DAF duplicate marking')], None),
    # explicit FiberHMM declarations
    ([_pg(ID='fiberhmm-dedup', PN='fiberhmm-dedup', DS='x; coord=seq (footprint tags)')], 'seq'),
    (_chain(_pg(ID='ft', PN='fibertools-rs', CL='ft add-nucleosomes a b'),
      _pg(ID='fiberhmm-tag-m5c', PN='fiberhmm-tag-m5c', DS='x; coord=seq')), 'seq'),
    # a path mentioning fiberhmm is not a FiberHMM program record
    (_chain(_pg(ID='ft', PN='fibertools-rs', CL='ft add-nucleosomes a b'),
      _pg(ID='samtools', PN='samtools', CL='samtools sort -o ~/fiberhmm_work/x.bam')), 'molecular'),
])
def test_legacy_tag_frame_rule(pgs, expected):
    assert legacy_tag_frame({'PG': pgs})[0] == expected


_FT = _pg(ID='ft', PN='fibertools-rs', CL='ft add-nucleosomes a b')
_ALN = _pg(ID='pbmm2', PN='pbmm2', CL='pbmm2 align ref in out')
_MERGE = 'samtools merge -o all.bam a.bam b.bam'


def _merged(branch_a, branch_b):
    """samtools merge of two histories: clashing IDs renamed with a -XXXXXXXX
    suffix, one merge record per chain end (same PN/VN/CL, one PP each)."""
    def rename(pgs, suffix):
        ids = {pg['ID']: pg['ID'] + suffix for pg in pgs}
        return [dict(pg, ID=ids[pg['ID']], **({'PP': ids[pg['PP']]} if 'PP' in pg else {}))
                for pg in pgs]
    a, b = branch_a, rename(branch_b, '-0A1B2C3D')
    ends = [a[-1]['ID'], b[-1]['ID']]
    merge = [_pg(ID='samtools' + ('' if i == 0 else '-5E6F7A8B'), PN='samtools', VN='1.21',
                 CL=_MERGE, PP=end) for i, end in enumerate(ends)]
    return a + b + merge


def test_pp_graph_follows_each_branch_of_a_merge():
    fibertools = _chain(_ALN, _FT)
    old_fiberhmm = _chain(_ALN)                      # FiberHMM <= 2.12: no @PG of its own
    called = _chain(_ALN, _pg(ID='fiberhmm-call', PN='fiberhmm-call', DS='x; coord=molecular'))
    # both branches fibertools: molecular, through the renamed IDs
    assert legacy_tag_frame({'PG': _merged(fibertools, fibertools)})[0] == 'molecular'
    # fibertools merged with an unmarked old BAM: header order would say
    # molecular (the fibertools record is last-but-merge); the branches say unknown
    assert legacy_tag_frame({'PG': _merged(fibertools, old_fiberhmm)})[0] is None
    # an explicit coord= declaration settles a disagreement it does not contradict
    assert legacy_tag_frame({'PG': _merged(fibertools, called)})[0] == 'molecular'
    seq = _chain(_ALN, _pg(ID='fiberhmm-dedup', PN='fiberhmm-dedup', DS='x; coord=seq (c)'))
    assert legacy_tag_frame({'PG': _merged(called, seq)})[0] is None


def test_pp_graph_order_does_not_decide():
    """Header order is not ancestry. A caller's branch listed before a
    fibertools branch would look superseded by fibertools in header order;
    on the PP graph it is still a branch whose MA-less reads the caller
    skipped, so consensus's disabled frame is not inferred."""
    from fiberhmm.io.annotation_frame import resolve_disabled_legacy_frame
    merged = _merged(_chain(_ALN, _pg(ID='fiberhmm-call', PN='fiberhmm-call',
                                      DS='x; coord=molecular')), _chain(_ALN, _FT))
    assert resolve_disabled_legacy_frame({'PG': merged}) is None
    assert resolve_disabled_legacy_frame({'PG': _merged(_chain(_ALN, _FT), _chain(_ALN, _FT))})[0] \
        == 'molecular'


def test_fibertools_without_pn_and_pp_cycles():
    no_pn = [_pg(ID='ft.1', CL='ft predict-m6a a b')]
    assert legacy_tag_frame({'PG': no_pn})[0] == 'molecular'
    by_cl = [_pg(ID='x', CL='/usr/local/bin/fibertools add-nucleosomes a b')]
    assert legacy_tag_frame({'PG': by_cl})[0] == 'molecular'
    # a PP cycle neither loops forever nor invents a frame
    cycle = [dict(_FT, PP='fiberhmm-dedup'),
             _pg(ID='fiberhmm-dedup', PN='fiberhmm-dedup', DS='d', PP='ft')]
    assert legacy_tag_frame({'PG': cycle})[0] in ('molecular', None)


def test_coord_molecular_declaration_anywhere_wins():
    assert legacy_tag_frame({'CO': ['fiberhmm:coord=molecular']})[0] == 'molecular'
    assert legacy_tag_frame({'PG': [_pg(ID='x', PN='x', DS='coord=molecular')]})[0] == 'molecular'


def test_pass_through_ds_token_is_honest():
    assert append_coord_to_ds('DAF dedup', 'molecular').startswith('DAF dedup; coord=molecular')
    assert 'coord=seq' in append_coord_to_ds('DAF dedup', 'seq')
    assert append_coord_to_ds('DAF dedup', None) == 'DAF dedup'


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


# --------------------------------------------------------------------------- #
#  consensus evidence loading and upstream Hia5 nucleosome recall                #
# --------------------------------------------------------------------------- #

def _load(path, **kwargs):
    import numpy as np
    from fiberhmm.inference.strand_rescue import N_CTX, load_region_evidence
    diagnostics = {}
    reads = load_region_evidence(
        str(path), 'chrT', 0, 8000, strand_mode='alignment', mode='pacbio-fiber',
        context_size=3, prob_threshold=125, llr_hit=np.zeros(N_CTX),
        llr_miss=np.zeros(N_CTX), min_mapq=0, ma_annotation_frame='auto',
        load_diagnostics=diagnostics, **kwargs)
    return reads, diagnostics


@pytest.mark.parametrize('name', FIXTURES)
def test_consensus_loader_places_fibertools_and_fiberhmm_footprints(name):
    """Consensus input with the default legacy_hia5_annotation_frame=disabled:
    Ma (0.13) and provenance-molecular ns/nl/as/al (0.6.2) load at fibertools'
    reference positions on both strands. Before the fix the 0.13 BAMs loaded no
    nucleosomes or MSPs and the 0.6.2 BAM raised the explicit-frame error."""
    reads, diagnostics = _load(FX / name, legacy_annotation_frame='disabled')
    want = EXPECTED['fixtures'][name]
    assert sorted(r.name for r in reads) == ['read_fwd', 'read_rev']
    for read in reads:
        assert sorted([c.start, c.end] for c in read.nucs) == want['nuc'][read.name]
        assert sorted([c.start, c.end] for c in read.msps) == want['msp'][read.name]
    provenance = diagnostics.get('legacy_annotation_frame_from_provenance')
    if name.startswith('ft0.6.2'):
        assert provenance['frame'] == 'molecular' and 'fibertools' in provenance['reason']
    else:
        # MA/Ma BAMs never fall back to legacy tags: diagnostics unchanged.
        assert provenance is None


def test_consensus_loader_keeps_the_explicit_error_without_provenance(tmp_path):
    src = FX / 'ft0.6.2_addnuc_fire.bam'
    stripped = tmp_path / 'noprov.bam'
    with pysam.AlignmentFile(str(src)) as bam:
        header = bam.header.to_dict()
        header.pop('PG', None)
        with pysam.AlignmentFile(str(stripped), 'wb', header=header) as out:
            for read in bam:
                out.write(read)
    pysam.index(str(stripped))
    with pytest.raises(ValueError, match='explicit annotation frame'):
        _load(stripped, legacy_annotation_frame='disabled')
    reads, _ = _load(stripped, legacy_annotation_frame='molecular')
    want = EXPECTED['fixtures']['ft0.6.2_addnuc_fire.bam']['nuc']
    assert {r.name: sorted([c.start, c.end] for c in r.nucs) for r in reads} == want


@pytest.mark.parametrize('name', ['ft0.13_addnuc_fire.bam', 'ft0.6.2_addnuc_fire.bam',
                                  'fiberhmm3.0.0_call.bam'])
def test_upstream_hia5_recall_reads_fibertools_scaffolds(name):
    """The staged engine's upstream nucleosome/TF recall re-reads the query
    annotations and checks them against the loaded scaffold; Ma and the
    provenance-resolved legacy frame must agree with it on the reverse read."""
    from fiberhmm.core.model_io import load_model_with_metadata
    from fiberhmm.inference.consensus.adapter import evidence_unit
    from fiberhmm.inference.consensus.upstream_recall import recall_hia5_alignment
    from fiberhmm.inference.strand_rescue import load_region_evidence
    from fiberhmm.inference.tf_recaller import build_llr_tables
    from fiberhmm.io.annotation_frame import resolve_disabled_legacy_frame

    model, k, mode = load_model_with_metadata(str(REPO / 'fiberhmm' / 'models' / 'hia5_pacbio.json'))
    hit, miss = build_llr_tables(model)
    reads = load_region_evidence(
        str(FX / name), 'chrT', 0, 8000, strand_mode='alignment', mode=mode,
        context_size=k, prob_threshold=125, llr_hit=hit, llr_miss=miss, min_mapq=0,
        ma_annotation_frame='auto', legacy_annotation_frame='disabled')
    by_name = {r.name: r for r in reads}
    gaps = EXPECTED['truth']['gaps']
    with pysam.AlignmentFile(str(FX / name)) as bam:
        frame = (resolve_disabled_legacy_frame(bam.header) or ('disabled',))[0]
        for alignment in bam:
            unit = evidence_unit(by_name[alignment.query_name], model, 'd', [], 0, 8000)
            result = recall_hia5_alignment(alignment, unit, model, 'alignment', mode, k,
                                           125, 5.0, legacy_annotation_frame=frame)
            assert result['status'] == 'recalled'
            nucs = result['nucleosomes']
            assert len(nucs) == len(gaps)
            assert all(any(min(b, g[1]) - max(a, g[0]) > 0.5 * (b - a) for g in gaps)
                       for a, b in nucs), alignment.query_name


# --------------------------------------------------------------------------- #
#  pass-through tools record the frame of the tags they carry                  #
# --------------------------------------------------------------------------- #

_BASE = {'HD': {'VN': '1.6', 'SO': 'coordinate'}, 'SQ': [{'SN': 'chr1', 'LN': 1000}]}
_INPUT_HEADERS = {
    # fibertools output: molecular by provenance, no coord token anywhere
    'fibertools': dict(_BASE, PG=[_pg(ID='ft.1', PN='fibertools-rs', VN='0.6.2',
                                      CL='ft add-nucleosomes -t 1 in.bam out.bam')]),
    # FiberHMM <= 2.12 output: no FiberHMM @PG, frame unknown from the header
    'unknown': dict(_BASE, PG=[_pg(ID='pbmm2', PN='pbmm2', CL='pbmm2 align ref in out')]),
    # a previous pass-through tool that resolved SEQ frame
    'seq': dict(_BASE, PG=[_pg(ID='fiberhmm-dedup', PN='fiberhmm-dedup', DS='x; coord=seq')]),
}
_RECORDED = {'fibertools': 'molecular', 'unknown': None, 'seq': 'seq'}


def _pass_through_headers(header):
    """Output header of every FiberHMM tool that copies ns/nl/as/al/MA unchanged."""
    from fiberhmm.cli.dedup import _dedup_output_header
    from fiberhmm.cli.duplex import _header_with_program
    from fiberhmm.cli.merge import _merge_output_header
    from fiberhmm.cli.strand_rescue_annotate import _header_with_provenance
    from fiberhmm.cli.tag_families import _family_header
    header = pysam.AlignmentHeader.from_dict(header)
    return {
        'fiberhmm-dedup': _dedup_output_header(
            header, min_jaccard=0.95, min_deam=10, ignore_strand=False, collapse=False,
            prob_threshold=0, max_end_diff=50),
        'fiberhmm-pair': _header_with_program(header, None, 'hybrid'),
        'fiberhmm-merge': _merge_output_header(
            header, recall=False, enzyme='ddda', prob_threshold=128, pairs_only=False,
            nuc_recall_policy='conservative', phase_nrl=196, cpg_mask_policy=None),
        'fiberhmm-strand-rescue-annotate': _header_with_provenance(
            header, report_sha256='0' * 64, minimum_posterior=0.9, command_line='x'),
        'fiberhmm-tag-consensus': _family_header(
            header, assignment_sha256='0' * 64, command_line='x'),
    }


@pytest.mark.parametrize('kind', sorted(_INPUT_HEADERS))
def test_pass_through_tools_record_the_input_frame(kind):
    """Before the fix these @PG records had no coord token, so a reader that
    lets the latest FiberHMM record decide (FiberBrowser) turned a fibertools
    BAM that went through e.g. fiberhmm-dedup into SEQ frame. Unknown input
    frames stay unrecorded rather than claimed."""
    expected = _RECORDED[kind]
    for program, output in _pass_through_headers(_INPUT_HEADERS[kind]).items():
        record = output.to_dict()['PG'][-1]
        assert record['PN'] == program
        ds = record.get('DS', '')
        if expected is None:
            assert 'coord=' not in ds, program
        else:
            assert f'coord={expected}' in ds, program
            assert 'coord=molecular' not in ds or expected == 'molecular', program
        # a reader sees the same frame after the tool as before it
        assert legacy_tag_frame(output)[0] == legacy_tag_frame(_INPUT_HEADERS[kind])[0]


def test_dedup_run_records_molecular_for_fibertools_input(tmp_path):
    from test_dedup import A_SITES, _make_bam
    from fiberhmm.cli.dedup import run_dedup

    plain = tmp_path / 'plain.bam'
    _make_bam(plain, [(f'A{i}', A_SITES, i % 2 == 1, 60) for i in range(3)])
    source = tmp_path / 'in.bam'
    with pysam.AlignmentFile(str(plain)) as bam:
        header = bam.header.to_dict()
        header['PG'] = _INPUT_HEADERS['fibertools']['PG']
        with pysam.AlignmentFile(str(source), 'wb', header=header) as out:
            for read in bam:
                out.write(read)
    output = tmp_path / 'out.bam'
    run_dedup(str(source), str(output), collapse=False)
    with pysam.AlignmentFile(str(output), check_sq=False) as bam:
        program = bam.header.to_dict()['PG'][-1]
        assert program['PN'] == 'fiberhmm-dedup'
        assert 'coord=molecular' in program['DS']
        assert legacy_tag_frame(bam.header)[0] == 'molecular'


@pytest.mark.parametrize('frame_arg, expected, used', [
    ('auto', 'coord=molecular', True),     # the fixture's fibertools provenance
    ('molecular', 'coord=molecular', True),
    ('query', 'coord=seq', False),         # the user's explicit choice is what was assumed
])
def test_tag_m5c_records_the_frame_it_carried(monkeypatch, tmp_path, frame_arg, expected, used):
    """The recorded frame is the one the tool actually used on the tags."""
    import fiberhmm.cli.tag_m5c as tag_m5c

    captured = {}

    def fake_annotate(*args, header_record=None, **kwargs):
        captured['record'] = header_record
        captured['frame'] = kwargs['input_molecular_frame']
        return {}

    monkeypatch.setattr(tag_m5c, '_preflight_input', lambda path: None)
    monkeypatch.setattr(tag_m5c, 'annotate_bam_per_read_islands', fake_annotate)
    tag_m5c.main(['-i', str(FX / 'ft0.6.2_addnuc_fire.bam'), '-o', str(tmp_path / 'o.bam'),
                  '-r', str(tmp_path / 'ref.fa'), '--enzyme', 'ddda',
                  '--input-frame', frame_arg])
    assert expected in captured['record']['DS']
    assert captured['frame'] is used


@pytest.mark.parametrize('header, expected', [
    ({'PG': [_pg(ID='ft', PN='fibertools-rs', CL='ft add-nucleosomes a b')]}, 'molecular'),
    # a pass-through FiberHMM tool after fibertools keeps it
    ({'PG': _chain(_pg(ID='ft', PN='fibertools-rs', CL='ft add-nucleosomes a b'),
             _pg(ID='fiberhmm-dedup', PN='fiberhmm-dedup',
                 DS='DAF dedup; coord=molecular (footprint tags carried over from the input)'))},
     'molecular'),
    # a FiberHMM caller after fibertools: reads it left without MA are ones it
    # skipped, so their legacy tags are not vouched for
    ({'PG': _chain(_pg(ID='ft', PN='fibertools-rs', CL='ft add-nucleosomes a b'),
             _pg(ID='fiberhmm-call', PN='fiberhmm-call', DS='x; coord=molecular (ns/nl)'))}, None),
    # FiberHMM 2.16.8 run over FiberHMM <= 2.12 output (the real ind Hia5 BAM
    # shape): 150 unrecalled reads keep query-frame ns/nl under coord=molecular
    ({'PG': _chain(_pg(ID='pbmm2', PN='pbmm2', CL='pbmm2 align'),
             _pg(ID='fiberhmm-call', PN='fiberhmm-call', DS='x; coord=molecular (ns/nl)'))}, None),
    ({'CO': ['fiberhmm:coord=molecular']}, None),
    ({}, None),
])
def test_consensus_disabled_frame_resolves_only_from_fibertools(header, expected):
    from fiberhmm.io.annotation_frame import resolve_disabled_legacy_frame
    resolved = resolve_disabled_legacy_frame(header)
    assert (resolved[0] if resolved else None) == expected


def test_recall_runs_ma_only_input_without_provenance(tmp_path):
    """fibertools 0.13 Ma is molecular by definition, so a lost @PG history
    does not block recall of Ma-only reads (only ns/nl/as/al need the frame)."""
    stripped = tmp_path / 'ma_noprov.bam'
    with pysam.AlignmentFile(str(FX / 'ft0.13_addnuc_fire.bam')) as bam:
        header = bam.header.to_dict()
        header.pop('PG', None)
        with pysam.AlignmentFile(str(stripped), 'wb', header=header) as out:
            for read in bam:
                out.write(read)
    pysam.index(str(stripped))
    out = tmp_path / 'o.bam'
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(REPO), env.get('PYTHONPATH')]))
    proc = subprocess.run(
        [sys.executable, '-m', 'fiberhmm.cli.recall_tfs', '-i', str(stripped), '-o', str(out),
         '--enzyme', 'hia5', '--seq', 'pacbio', '-c', '1'], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    with pysam.AlignmentFile(str(out)) as bam:
        got = {r.query_name: _reference_nucs(r) for r in bam}
    assert got == EXPECTED['fixtures']['ft0.13_addnuc_fire.bam']['nuc']


def test_fiberhmm_ma_wins_when_a_read_carries_both():
    """A read with FiberHMM MA and a stale fibertools Ma: every reader uses MA."""
    from array import array
    from fiberhmm.cli.extract_tags import _parse_all_ma_annotations
    with pysam.AlignmentFile(str(FX / 'ft0.13_addnuc_fire.bam')) as bam:
        read = next(r for r in bam if r.query_name == 'read_rev')
    read.set_tag('MA', '8000;nuc.Q:101-150')
    read.set_tag('AQ', array('B', [77]))
    assert annotation_tags(read)[3] == 'MA'
    assert fibertools_ma_intervals(read) is None
    parsed = _parse_all_ma_annotations(read, annotation_frame='molecular')
    assert [(a['start'], a['length'], a['quals']) for a in parsed['nuc']] == [(8000 - 250, 150, [77])]
    assert 'fire' not in parsed and 'msp' not in parsed
    from fiberhmm.inference.legacy_annotations import legacy_annotations
    assert legacy_annotations(read, 'molecular') is None


def test_consensus_export_keeps_fibertools_ma_under_family_layers():
    """Exporting family layers onto a fibertools 0.13 read copies its Ma
    nucleosomes/MSPs/FIRE into the new MA (MA wins over Ma, so they would
    otherwise vanish), and a fibertools source's export declares molecular."""
    from fiberhmm.cli.extract_tags import _parse_all_ma_annotations
    from fiberhmm.inference.consensus.bam_export import (
        _append_annotations, _export_ds)
    with pysam.AlignmentFile(str(FX / 'ft0.13_addnuc_fire.bam')) as bam:
        header = bam.header
        read = next(r for r in bam if r.query_name == 'read_rev')
    rows = [dict(chrom='chrT', interval=[100, 120], layer='fam', token='t1', tq=50, fi=1, op=2)]
    assert _append_annotations(read, rows) == 1
    parsed = _parse_all_ma_annotations(read, annotation_frame='molecular')
    assert len(parsed['nuc']) == 25 and len(parsed['msp']) == 24 and len(parsed['fire']) == 7
    fam = parsed['fam'][0]
    assert (fam['start'], fam['start'] + fam['length']) == (100, 120)  # 8000M at 0: query == reference
    assert [a['start'] for a in parsed['nuc']][:1] == [8000 - 392 - 154]
    from fiberhmm.io.annotation_frame import ma_annotation_frame, resolve_disabled_legacy_frame
    from fiberhmm.io.bam_header import append_pg_record
    exported = append_pg_record(header, dict(PN='fiberhmm-consensus', DS=_export_ds(header)))
    # the export's MA reads as molecular, and a 0.6.2-style export still
    # resolves consensus's disabled legacy frame from its fibertools provenance
    assert ma_annotation_frame(exported) == 'molecular'
    assert resolve_disabled_legacy_frame(exported)[0] == 'molecular'
    with pysam.AlignmentFile(str(FX / 'fiberhmm3.0.0_call.bam')) as bam:
        assert _export_ds(bam.header) == 'Frozen staged family annotations in MA/AQ/AN'
