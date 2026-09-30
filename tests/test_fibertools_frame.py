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

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pysam
import pytest

from fiberhmm.io.annotation_frame import (
    FIBERHMM_FOOTPRINT_PROGRAMS,
    append_coord_to_ds,
    legacy_tag_frame,
    legacy_tag_frame_report,
    pg_family,
    pg_writer_frame,
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


# Synthetic-header cases mirror FiberBrowser's tests/test_footprint_tag_frame.py
# (the shared rule, docs/reference/footprint-tag-frame.md).

REPORT = legacy_tag_frame_report
FT = {"ID": "ft.1", "PN": "fibertools-rs", "CL": "ft predict-m6a in.bam out.bam"}
FHMM_OLD = {"ID": "fiberhmm-apply", "PN": "fiberhmm-apply", "CL": "fiberhmm-apply -i in.bam -o out.bam"}
FHMM_NEW = {"ID": "fiberhmm-call", "PN": "fiberhmm-call", "CL": "fiberhmm-call -i in.bam -o out.bam",
            "DS": "FiberHMM fused apply+recall; coord=molecular"}
SORT = {"ID": "samtools", "PN": "samtools", "CL": "samtools sort -o out.bam"}


def _frame(pgs, co=()):
    return REPORT({"PG": list(pgs), "CO": list(co)})["frame"]


@pytest.mark.parametrize("ran", [(FT, FHMM_OLD, SORT), (FHMM_OLD, FT, SORT)])
def test_every_header_order_of_one_chain_gives_the_same_frame(ran):
    chain = _chain(*ran)
    want = "seq" if ran[1] is FHMM_OLD else "molecular"   # the later writer decides
    for order in itertools.permutations(chain):
        assert _frame(order) == want, [r["ID"] for r in order]


def test_unmarked_fiberhmm_writer_after_fibertools_in_the_chain_wins_even_when_listed_first():
    pgs = _chain(FT, FHMM_OLD)
    assert _frame([pgs[1], pgs[0]]) == "seq"


def test_header_order_is_the_chain_only_when_no_record_has_pp():
    assert _frame([FT, FHMM_OLD]) == "seq"
    assert _frame([FHMM_OLD, FT]) == "molecular"
    report = REPORT({"PG": [FHMM_OLD, FT, dict(SORT, PP="ft.1")]})
    assert report["ambiguous"] and report["votes"] == {"molecular": 1, "seq": 1}


@pytest.mark.parametrize("record,family", [
    ({"ID": "ft.1", "PN": "fibertools-rs"}, "fibertools"),
    ({"ID": "ft.1-61483E4"}, "fibertools"),
    ({"ID": "x", "CL": "/opt/bin/ft add-nuc in.bam out.bam"}, "fibertools"),
    ({"ID": "fiberhmm-call-75E22241"}, "fiberhmm"),
    ({"ID": "fiberhmm-call.2", "PN": "fiberhmm-call"}, "fiberhmm"),
    ({"ID": "samtools-fiberhmm-output", "PN": "samtools"}, None),
    ({"ID": "ft.1", "PN": "samtools"}, None),
    ({"ID": "samtools", "PN": "samtools", "CL": "samtools sort -o ~/fiberhmm_work/x.bam"}, None),
    ({"ID": "whatever", "CL": "python pipeline.py fiberhmm-call"}, None),
])
def test_producer_identity(record, family):
    assert pg_family(record) == family


@pytest.mark.parametrize("cl,writes", [
    ("ft predict-m6a in.bam out.bam", True), ("ft -t 8 m6a in.bam out.bam", True),
    ("ft predict -m 254 in.bam out.bam", True), ("ft add in.bam out.bam", True),
    ("ft add-nuc --min-ml-score 225 in.cram out.bam", True), ("ft fire - -", True),
    ("fibertools-rs fiber-hmm in.bam out.bam", True),
    ("ft extract --all out.tsv in.bam", False), ("ft convert-tags in.bam out.bam", False),
    ("ft pileup in.bam", False), ("ft clear-kinetics in.bam out.bam", False),
])
def test_fibertools_footprint_writers(cl, writes):
    assert (pg_writer_frame({"ID": "ft.1", "PN": "fibertools-rs", "CL": cl}, False) == "molecular") is writes


def test_fiberhmm_passthrough_programs_do_not_reset_the_frame():
    for pn in ("fiberhmm-tag-m5c", "fiberhmm-dedup", "fiberhmm-merge", "fiberhmm-pair", "fiberhmm-tag-consensus",
               "fiberhmm-strand-rescue-annotate", "fiberhmm-pipeline", "fiberhmm-call-m5c"):
        assert _frame(_chain(FT, {"ID": pn, "PN": pn, "CL": f"{pn} -i in.bam -o out.bam"})) == "molecular", pn
    for pn in sorted(FIBERHMM_FOOTPRINT_PROGRAMS):
        assert _frame(_chain(FT, {"ID": pn, "PN": pn, "CL": f"{pn} -i in.bam"})) == "seq", pn


def test_fiberhmm_writer_frame_follows_its_own_or_the_header_declaration():
    assert _frame(_chain(FT, FHMM_NEW)) == "molecular"
    assert _frame(_chain(FT, FHMM_OLD), co=["fiberhmm:coord=molecular"]) == "molecular"
    assert _frame(_chain(FT, {"ID": "fiberhmm", "PN": "fiberhmm", "CL": "fiberhmm-apply -i in.bam"})) == "seq"


def test_pp_cycles_terminate():
    pgs = [dict(FT, PP="fiberhmm-apply"), dict(FHMM_OLD, PP="ft.1")]
    assert REPORT({"PG": pgs})["frame"] in ("seq", "molecular")
    assert _frame([dict(SORT, PP="samtools")]) == "seq"


def test_pp_to_a_missing_id_ends_the_chain():
    assert _frame([FT, dict(SORT, PP="not-there")]) == "molecular"
    assert _frame([dict(FT, PP="gone"), dict(SORT, PP="ft.1")]) == "molecular"


def test_duplicate_ids_link_to_the_nearest_preceding_record():
    pgs = [dict(FHMM_OLD, ID="dup"), dict(SORT, PP="dup"), dict(FT, ID="dup"), dict(SORT, ID="s2", PP="dup")]
    report = REPORT({"PG": pgs})
    assert report["votes"] == {"molecular": 1, "seq": 1} and report["ambiguous"]


def _old_and_ft_branches():
    old_branch = _chain(dict(FHMM_OLD, ID="fiberhmm-apply"), dict(SORT, ID="samtools"))
    ft_branch = _chain(dict(FT, ID="ft.1-570F7E82"), dict(SORT, ID="samtools-39D564CF"))
    return old_branch, ft_branch


def test_merged_branches_that_disagree_are_ambiguous():
    old_branch, ft_branch = _old_and_ft_branches()
    report = REPORT({"PG": old_branch + ft_branch})
    assert report["ambiguous"] and report["frame"] == "seq" and report["source"] == "majority"
    two_ft = ft_branch + _chain(dict(FT, ID="ft.1-60F6C071"), dict(SORT, ID="samtools-11AAC8BA"))
    report = REPORT({"PG": old_branch + two_ft})
    assert report["ambiguous"] and report["frame"] == "molecular"
    report = REPORT({"PG": old_branch + ft_branch, "CO": ["fiberhmm:coord=molecular"]})
    assert report["frame"] == "molecular" and not report["ambiguous"]
    new_branch = _chain(FHMM_NEW, dict(SORT, ID="samtools-7BAE9814"))
    report = REPORT({"PG": old_branch + new_branch})
    assert report["ambiguous"] and report["frame"] == "molecular" and report["source"] == "declared"


def test_no_writer_uses_the_declaration_or_the_seq_default():
    report = REPORT({"PG": [SORT]})
    assert (report["frame"], report["source"], report["ambiguous"], report["chains"]) == ("seq", "default", False, 1)
    assert REPORT({"CO": ["fiberhmm:coord=molecular"]})["frame"] == "molecular"
    assert REPORT(None)["frame"] == "seq"


def test_fiberhmm_refuses_where_fiberbrowser_takes_the_majority():
    """FiberHMM differs from the shared report in one case only: disagreeing
    writers with nothing declared (FiberBrowser shows the majority)."""
    old_branch, ft_branch = _old_and_ft_branches()
    two_ft = ft_branch + _chain(dict(FT, ID="ft.1-60F6C071"), dict(SORT, ID="samtools-11AAC8BA"))
    for pgs in (old_branch + ft_branch, old_branch + two_ft):
        frame, reason = legacy_tag_frame({"PG": pgs})
        assert frame is None and 'disagree' in reason
    # declared: the declaration decides, as in FiberBrowser
    new_branch = _chain(FHMM_NEW, dict(SORT, ID="samtools-7BAE9814"))
    assert legacy_tag_frame({"PG": old_branch + new_branch})[0] == "molecular"
    # every other case is the shared report's frame
    for pgs in (_chain(FT, FHMM_OLD), _chain(FHMM_OLD, FT), [SORT], []):
        assert legacy_tag_frame({"PG": pgs})[0] == REPORT({"PG": pgs})["frame"]


def test_pass_through_ds_token_is_honest():
    assert append_coord_to_ds('DAF dedup', 'molecular').startswith('DAF dedup; coord=molecular')
    # the shared rule has one declaration token: SEQ and unknown are not recorded
    assert append_coord_to_ds('DAF dedup', 'seq') == 'DAF dedup'
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


def test_recall_refuses_when_merged_histories_disagree(tmp_path):
    """Writers that disagree across merged @PG branches, with nothing declaring
    coord=molecular, stop the run with a message naming both explicit
    choices instead of guessing (FiberBrowser would show the majority)."""
    old_branch, ft_branch = _old_and_ft_branches()
    mixed = tmp_path / 'mixed.bam'
    with pysam.AlignmentFile(str(FX / 'ft0.6.2_addnuc_fire.bam')) as bam:
        header = bam.header.to_dict()
        header['PG'] = old_branch + ft_branch
        with pysam.AlignmentFile(str(mixed), 'wb', header=header) as out:
            for read in bam:
                out.write(read)
    pysam.index(str(mixed))
    out = tmp_path / 'x.bam'
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(REPO), env.get('PYTHONPATH')]))
    base = [sys.executable, '-m', 'fiberhmm.cli.recall_tfs', '-i', str(mixed), '-o', str(out),
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
_RECORDED = {'fibertools': 'molecular', 'unknown': None, 'seq': None}


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
    ('query', None, False),                # the user's explicit SEQ choice: nothing declared
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
    if expected is None:
        assert 'coord=' not in captured['record']['DS']
    else:
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
    # Codex round 14: fibertools -> sort, merged with an independent aligner
    # chain that has no footprint writer (its reverse reads carry SEQ-frame
    # ns/nl). One writer vote over two chains is not "every chain".
    ({'PG': _chain(_pg(ID='ft', PN='fibertools-rs', CL='ft add-nucleosomes a b'),
                   _pg(ID='samtools', PN='samtools', CL='samtools sort'))
            + [_pg(ID='pbmm2', PN='pbmm2', CL='pbmm2 align ref in out')]}, None),
    # two fibertools chains merged: both vote fibertools
    ({'PG': _chain(_pg(ID='ft', PN='fibertools-rs', CL='ft add-nucleosomes a b'),
                   _pg(ID='samtools', PN='samtools', CL='samtools sort'))
            + _chain(_pg(ID='ft-0A1B2C3D', PN='fibertools-rs', CL='ft add-nucleosomes c d'),
                     _pg(ID='samtools-5E6F7A8B', PN='samtools', CL='samtools sort'))}, 'molecular'),
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
    otherwise vanish). Codex round 14 [HIGH] 2: the copy must be in the
    file's MA frame (this header declares nothing, so SEQ), while the family
    layer stays molecular as its FIBERHMM-CONSENSUS-MA contract declares."""
    from fiberhmm.cli.extract_tags import _parse_all_ma_annotations
    from fiberhmm.inference.consensus.bam_export import _append_annotations
    from fiberhmm.io.annotation_frame import ma_annotation_frame
    with pysam.AlignmentFile(str(FX / 'ft0.13_addnuc_fire.bam')) as bam:
        frame = ma_annotation_frame(bam.header)
        read = next(r for r in bam if r.query_name == 'read_rev')
    assert frame == 'seq'
    rows = [dict(chrom='chrT', interval=[100, 120], layer='fam', token='t1', tq=50, fi=1, op=2)]
    assert _append_annotations(read, rows, frame) == 1
    parsed = _parse_all_ma_annotations(read, annotation_frame=frame, molecular_layers={'fam'})
    assert len(parsed['nuc']) == 25 and len(parsed['msp']) == 24 and len(parsed['fire']) == 7
    want = EXPECTED['fixtures']['ft0.13_addnuc_fire.bam']
    assert sorted([a['start'], a['start'] + a['length']] for a in parsed['nuc']) == want['nuc']['read_rev']
    assert sorted([a['start'], a['start'] + a['length']] for a in parsed['msp']) == want['msp']['read_rev']
    fire = sorted([a['start'], a['start'] + a['length'], a['quals'][0]] for a in parsed['fire'])
    assert fire == want['fire']['read_rev']          # Aq bytes stay with their elements
    fam = parsed['fam'][0]
    assert (fam['start'], fam['start'] + fam['length']) == (100, 120)  # 8000M at 0: query == reference
    # a declared-molecular file copies Ma verbatim
    with pysam.AlignmentFile(str(FX / 'ft0.13_addnuc_fire.bam')) as bam:
        read = next(r for r in bam if r.query_name == 'read_rev')
    _append_annotations(read, rows, 'molecular')
    assert read.get_tag('MA').startswith(read.get_tag('Ma') + ';fam.')


def test_consensus_export_publisher_keeps_fibertools_footprints_in_place(tmp_path):
    """The full export_bams publisher on a fibertools 0.13 source (Codex's
    shape): read back with the header's MA frame and the contract's molecular
    layers, the copied nucleosomes and the family land where fibertools and
    the family put them, and the export declares no frame it did not write."""
    import hashlib
    from fiberhmm.cli.extract_tags import _parse_all_ma_annotations
    from fiberhmm.inference.consensus.artifacts import digest
    from fiberhmm.inference.consensus.bam_export import export_bams
    from fiberhmm.io.annotation_frame import consensus_molecular_layers, ma_annotation_frame
    import shutil
    src = tmp_path / 'ft013.bam'
    shutil.copy(FX / 'ft0.13_addnuc_fire.bam', src)
    pysam.index(str(src))
    with pysam.AlignmentFile(str(src)) as bam:
        read = next(r for r in bam if r.is_reverse)
    sha = hashlib.sha256(read.to_string().encode()).hexdigest()
    unit = dict(unit_id='u', read_name=read.query_name, positions=list(range(8000)),
                source_members=[dict(read_name=read.query_name, library_id=str(src),
                                     record_sha256=sha, alignment_occurrence=0)],
                native_multi_interval_calls=[dict(interval=[100, 120], llr=8, opportunities=20)])
    payload = dict(region=dict(chrom='chrT', start=0, end=8000),
                   strata=[dict(dataset_id='a', chemistry='hia5-pacbio', units=[unit])],
                   input_files=[dict(dataset_id='a', path=str(src), size=src.stat().st_size,
                                     mtime_ns=src.stat().st_mtime_ns)])
    result = dict(final_stage='resolved',
                  manifest=dict(input_digest=digest(payload), parameters={'cross': {'enabled': False}},
                                display_mode='SR'),
                  datasets={'a': dict(cr=dict(records=[dict(unit_id='a::u', proposals=[
                      dict(source_interval=[100, 120], compatible_families=['compact'])])]))})
    outputs = export_bams([(result, payload)], tmp_path / 'export', scope='full')
    with pysam.AlignmentFile(outputs[0]['bam']) as bam:
        header = bam.header
        read = next(r for r in bam if r.is_reverse)
    assert 'coord=' not in header.to_dict()['PG'][-1].get('DS', '')
    layers = consensus_molecular_layers(header)
    parsed = _parse_all_ma_annotations(read, annotation_frame=ma_annotation_frame(header),
                                       molecular_layers=layers)
    want = EXPECTED['fixtures']['ft0.13_addnuc_fire.bam']['nuc']['read_rev']
    assert sorted([a['start'], a['start'] + a['length']] for a in parsed['nuc']) == want
    family = [(a['start'], a['start'] + a['length']) for name in layers for a in parsed.get(name, [])]
    assert family == [(100, 120)]


