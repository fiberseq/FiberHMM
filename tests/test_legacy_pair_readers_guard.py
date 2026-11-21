"""Every reader of the legacy footprint pairs uses the shared array-pair rule.

``fiberhmm.io.legacy_tags.array_tag_pair`` decides when ``ns``/``nl`` and
``as``/``al`` form a call: both tags present, integer arrays, equal lengths.
A scalar or string tag (Dorado writes ``ns:i``, a signal count) or unequal
lengths make the pair absent. Tools that read the tags raw used to crash or
disagree with each other on such reads (recall-tfs raised a TypeError on a
scalar ``ns`` beside an ``nl`` array). This guard fails when a module outside
``legacy_tags`` reads one of these tags directly.
"""
import re
from pathlib import Path

import fiberhmm

PACKAGE = Path(fiberhmm.__file__).parent
RULE_MODULE = PACKAGE / 'io' / 'legacy_tags.py'
RAW_READ = re.compile(r"""get_tag\(\s*(?:['"](?:ns|nl|as|al)['"]|start_tag|length_tag)\s*[,)]""")


def test_no_module_reads_legacy_pairs_without_the_shared_rule():
    offenders = []
    for path in sorted(PACKAGE.rglob('*.py')):
        if path == RULE_MODULE:
            continue
        for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if RAW_READ.search(line):
                offenders.append(f'{path.relative_to(PACKAGE)}:{lineno}: {line.strip()}')
    assert not offenders, ('read ns/nl/as/al through fiberhmm.io.legacy_tags.array_tag_pair:\n'
                           + '\n'.join(offenders))


def test_guard_pattern_catches_raw_reads():
    for line in ("ns = read.get_tag('ns')", 'x = read.get_tag("al")', 'v = read.get_tag(start_tag)'):
        assert RAW_READ.search(line), line
    for line in ("read.get_tag('nq')", "read.get_tag('MA')", "read.has_tag('ns')"):
        assert not RAW_READ.search(line), line


# ---------------------------------------------------------------------------
# MA first, legacy fallback (3.3.0): fiberhmm.io.footprint_calls
# ---------------------------------------------------------------------------
#
# Nucleosome/MSP calls come from the MA (else fibertools Ma) nuc/msp groups;
# legacy pairs only when the record has none. A module may read a legacy pair
# only where it has checked MA first, or where the read is not a source of
# calls. Each entry says why; a new reader must go through
# fiberhmm.io.footprint_calls.resolve_footprint_tags or be listed here.

PAIR_READ = re.compile(r"\b(?:raw_)?(?:has_)?(?:array_tag_pair|legacy_pair|legacy_calls)\(")

LEGACY_PAIR_READERS = {
    'io/footprint_calls.py': 'the MA-first resolver itself (legacy fallback, matching legacy nq)',
    'daf/m5c.py': 'MA first: legacy only when footprint_annotation_tag(read) is not MA',
    'qc/core.py': 'MA first: MA/Ma footprint groups checked before the legacy pair',
    'qc/states.py': 'MA first: MA/Ma msp/nuc groups checked before the legacy pair',
    'scan/modules/nfr.py': 'MA first: MA/Ma footprint groups checked before ns/nl',
    'nuc/records.py': 'MA first: decode_record tries _footprints_from_ma before legacy',
    'cli/extract_tags.py': 'MA first: _extract_ma_interval_type before legacy; diagnostics counts',
    'inference/consensus/upstream_recall.py': 'MA first: legacy_tag_intervals only without MA/Ma nuc/msp',
    'io/scan_ma.py': 'legacy -> MA conversion for records without MA/Ma (scan annotation)',
    'io/graph_lift.py': 'transport: lifts both tag forms; not a source of calls',
    'io/frame_carry.py': 'transport: carries both tag forms into molecular frame',
    'scan/runner.py': 'transport: carries legacy tags into molecular frame',
    'io/annotation_frame.py': 'frame evidence only (does the read carry frame-dependent tags)',
    'advisories.py': 'presence only (was the read called)',
    'pipeline/runner.py': 'presence only (was the read called)',
    'scan/reads.py': 'presence only (was the read called), MA/Ma checked first',
    'cli/recall_tfs.py': 'presence only: legacy frame needed only without MA/Ma nuc/msp',
    'scan/peaks_reads.py': "FiberBrowser loader parity (its tiles read legacy first); open question",
}


def test_legacy_pair_readers_check_ma_first_or_are_listed():
    found = set()
    for path in sorted(PACKAGE.rglob('*.py')):
        if path == RULE_MODULE:
            continue
        rel = str(path.relative_to(PACKAGE))
        text = path.read_text(encoding='utf-8')
        if any(PAIR_READ.search(line) and not line.lstrip().startswith(('def ', '#'))
               for line in text.splitlines()):
            found.add(rel)
    unlisted = sorted(found - set(LEGACY_PAIR_READERS))
    assert not unlisted, ('these modules read legacy ns/nl/as/al without the MA-first rule; use '
                          'fiberhmm.io.footprint_calls.resolve_footprint_tags or list them with a reason:\n'
                          + '\n'.join(unlisted))
    stale = sorted(set(LEGACY_PAIR_READERS) - found)
    assert not stale, f'listed modules no longer read legacy pairs: {stale}'


def _read(ma=None, legacy=None, reverse=False):
    import array

    import pysam

    header = pysam.AlignmentHeader.from_dict({
        'HD': {'VN': '1.6'}, 'SQ': [{'SN': 'c', 'LN': 10_000}],
        'PG': [{'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call', 'VN': '3.3.0',
                'DS': 'FiberHMM fused apply+recall; coord=molecular'}],
        'CO': ['fiberhmm:coord=molecular']})
    r = pysam.AlignedSegment(header)
    r.query_name, r.flag, r.reference_id, r.reference_start = 'r', 16 if reverse else 0, 0, 0
    r.query_sequence = 'A' * 1000
    r.cigarstring = '1000M'
    if ma:
        r.set_tag('MA', ma)
    if legacy:
        for tag, values in legacy.items():
            r.set_tag(tag, array.array('I', values))
    return r


MA = '1000;nuc.:101-150,401-147;msp.:301-50,601-200'
STALE = {'ns': [700], 'nl': [160], 'as': [10], 'al': [40]}


def _consumers():
    from fiberhmm.cli.utils import _parse_footprints_from_bam_read
    from fiberhmm.daf import m5c
    from fiberhmm.inference.legacy_annotations import legacy_calls_apply
    from fiberhmm.inference.tagging import _retained_seq_msps
    from fiberhmm.inference.tf_recaller import recall_read
    from fiberhmm.qc.core import _footprint_lengths
    from fiberhmm.scan.modules.nfr import nucleosome_runs

    zeros = [0.0] * 4096
    return {
        'recall_read': lambda r: recall_read(r, zeros, zeros, 'pacbio-fiber', 3, 5.0, 3, 90),
        'm5c_nucs': lambda r: m5c._nucleosome_intervals_seq(r, True),
        'm5c_msps': lambda r: m5c._initial_msp_intervals_seq(r, True),
        'qc_lengths': lambda r: _footprint_lengths(r)[0],
        'nfr_runs': nucleosome_runs,
        'probs_mask': lambda r: _parse_footprints_from_bam_read(r, True).nonzero()[0].tolist(),
        'apply_retained_msps': _retained_seq_msps,
        'consensus_legacy_applies': legacy_calls_apply,
    }


def test_changed_consumers_follow_ma_over_stale_legacy():
    for reverse in (False, True):
        for name, consume in _consumers().items():
            ma_only = consume(_read(MA, reverse=reverse))
            assert consume(_read(MA, STALE, reverse=reverse)) == ma_only, name
            if name != 'consensus_legacy_applies':
                # The stale tags alone really give something else.
                assert consume(_read(None, STALE, reverse=reverse)) != ma_only, name
            else:
                assert consume(_read(None, STALE, reverse=reverse)) is True
            # Legacy is the fallback when MA carries no nuc/msp group.
            with_m5c = consume(_read('1000;ddda_mcg.:11-5', STALE, reverse=reverse))
            assert with_m5c == consume(_read(None, STALE, reverse=reverse)), name
