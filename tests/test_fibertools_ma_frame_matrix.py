"""MA vs fibertools Ma: frame and precedence across the readers that fall back
to Ma, plus the tag sniffs that count it.

Matrix: forward/reverse 400M reads at chr1:1000 x

* ``MA_mol`` -- FiberHMM MA under a ``coord=molecular`` header;
* ``MA_seq`` -- FiberHMM MA under an unmarked FiberHMM <= 2.12 header (SEQ);
* ``Ma_seqhdr`` -- fibertools Ma under that SEQ header (Ma stays molecular);
* ``both`` -- MA (molecular header) plus a decoy Ma: MA wins.

Each case stores the same molecular truth in its own frame, so every reader
must place it on the same reference coordinates.
"""
from __future__ import annotations

import array
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pysam
import pytest

FX = Path(__file__).parent / 'fixtures' / 'fibertools_frame'
FT_FIRE = FX / 'ft0.13_addnuc_fire.bam'
REPO = Path(__file__).resolve().parents[1]
L = 400
OLD_CALL = {'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call', 'VN': '2.12.0',
            'CL': 'fiberhmm-call -i in.bam -o out.bam'}
MOL_CO = 'fiberhmm:coord=molecular'

# Molecular truth (0-based start, length) and per-annotation quality rows.
NUC = [((20, 150), [11])]
FIRE = [((175, 30), [21]), ((230, 40), [22])]
MSP = [((172, 100), [])]
EXT = [((280, 10), [31, 32])]          # a 2-byte carried group
TF = [((300, 12), [41, 42, 43])]
GROUPS = [('nuc', 'Q', NUC), ('fire', 'Q', FIRE), ('msp', '', MSP),
          ('ext', 'QQ', EXT), ('tf', 'QQQ', TF)]
DECOY = '400;nuc.Q:301-90;tf.QQQ:51-8'
DECOY_AQ = [99, 98, 97, 96]
CASES = ['MA_mol', 'MA_seq', 'Ma_seqhdr', 'both']


def _flip(start, length):
    return L - (start + length), length


def _stored(reverse, seq_frame):
    """MA string + AQ for the truth, mirrored into SEQ frame when asked."""
    parts, aq = [str(L)], []
    for name, spec, items in GROUPS:
        rows = []
        for (start, length), quals in items:
            if reverse and seq_frame:
                start, length = _flip(start, length)
                if len(quals) >= 3:
                    quals = [quals[0], quals[2], quals[1], *quals[3:]]
            rows.append(((start, length), quals))
        rows.sort()
        parts.append(f'{name}.{spec}:' + ','.join(f'{s + 1}-{n}' for (s, n), _ in rows))
        for _iv, quals in rows:
            aq.extend(quals)
    return ';'.join(parts), aq


def _header(case):
    header = {'HD': {'VN': '1.6', 'SO': 'coordinate'},
              'SQ': [{'SN': 'chr1', 'LN': 10_000}], 'PG': [dict(OLD_CALL)]}
    if case in ('MA_mol', 'both'):
        header['CO'] = [MOL_CO]
    return pysam.AlignmentHeader.from_dict(header)


def _read(case, reverse, header=None, legacy=False):
    header = header or _header(case)
    read = pysam.AlignedSegment(header)
    read.query_name = f'{case}_{"rev" if reverse else "fwd"}'
    read.flag, read.reference_id = (16 if reverse else 0), 0
    read.reference_start, read.mapping_quality = 1000, 60
    read.query_sequence, read.cigarstring = 'A' * L, f'{L}M'
    read.query_qualities = pysam.qualitystring_to_array('I' * L)
    ma, aq = _stored(reverse, seq_frame=(case == 'MA_seq'))
    tags = ('Ma', 'Aq') if case == 'Ma_seqhdr' else ('MA', 'AQ')
    read.set_tag(tags[0], ma, value_type='Z')
    read.set_tag(tags[1], array.array('B', aq))
    if case == 'both':
        read.set_tag('Ma', DECOY, value_type='Z')
        read.set_tag('Aq', array.array('B', DECOY_AQ))
    if legacy:   # stale legacy nucleosomes that MA/Ma must win over
        read.set_tag('ns', array.array('I', [5]))
        read.set_tag('nl', array.array('I', [77]))
    return read


def _ref(interval, reverse):
    start, length = _flip(*interval) if reverse else interval
    return 1000 + start, 1000 + start + length


# --------------------------------------------------------------------------- #
#  recall carry-over (write_ma_tags after the output-frame carry)               #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('case', CASES)
def test_recall_carry_over_matrix(case, reverse):
    """Carried groups (fire.Q, ext.QQ) land molecular, with their own AQ bytes,
    whether quality groups sit before (nuc.Q), between (msp) or after (tf.QQQ)
    them; nuc/msp/tf are regenerated; no Ma is left; MA beats a decoy Ma."""
    from fiberhmm.io.frame_carry import carry_to_molecular
    from fiberhmm.inference.tf_recaller import write_ma_tags

    read = _read(case, reverse)
    carry_to_molecular(read)        # what recall does before writing
    write_ma_tags(read, L, [], [(50, 150)], [(0, 50)], nq_for_kept_nucs=[7],
                  also_write_legacy=False)

    assert not any(read.has_tag(t) for t in ('Ma', 'Aq', 'An'))
    groups = read.get_tag('MA').split(';')[1:]
    heads = [g.partition(':')[0] for g in groups]
    assert heads == ['nuc.Q', 'msp.', 'fire.Q', 'ext.QQ']
    fire = ','.join(f'{s + 1}-{n}' for (s, n), _ in FIRE)
    assert groups[2] == f'fire.Q:{fire}'
    assert groups[3] == f'ext.QQ:{EXT[0][0][0] + 1}-{EXT[0][0][1]}'
    assert list(read.get_tag('AQ')) == [7, 21, 22, 31, 32]


# --------------------------------------------------------------------------- #
#  fiberhmm-qc footprint lengths                                                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('case', CASES)
def test_qc_footprint_lengths_matrix(case, reverse, legacy):
    from fiberhmm.qc.core import _footprint_lengths

    nuc, tf, tagged = _footprint_lengths(_read(case, reverse, legacy=legacy))
    assert tagged
    assert (nuc, tf) == ([n for (_s, n), _q in NUC], [n for (_s, n), _q in TF])


# --------------------------------------------------------------------------- #
#  fiberhmm-footprint-model BAM loader                                          #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('case', CASES)
def test_footprint_model_loader_matrix(tmp_path, case):
    from fiberhmm.io.footprint_bam import load_footprint_molecules_from_bam

    header = _header(case)
    path = tmp_path / f'{case}.bam'
    with pysam.AlignmentFile(str(path), 'wb', header=header) as out:
        for reverse in (False, True):
            out.write(_read(case, reverse, header=header))
    pysam.index(str(path))
    loaded = load_footprint_molecules_from_bam(path)
    assert loaded.diagnostics.skipped_missing_ma == 0
    got = {m.molecule_id: m for m in loaded.molecules}
    assert len(got) == 2
    for reverse in (False, True):
        molecule = got[f'{case}_{"rev" if reverse else "fwd"}']
        assert [(t.start, t.end) for t in molecule.tfs] == [_ref(iv, reverse) for iv, _ in TF]
        assert list(molecule.msps) == [_ref(iv, reverse) for iv, _ in MSP]


# --------------------------------------------------------------------------- #
#  sniffs: extract tag diagnostic, ma-types --scan, check                       #
# --------------------------------------------------------------------------- #

def _ma_only_copy(tmp_path):
    """The real fibertools 0.13 fixture: Ma/Aq only, ft @PG, no MA-TYPES."""
    dst = tmp_path / 'ft.bam'
    shutil.copy(FT_FIRE, dst)
    shutil.copy(str(FT_FIRE) + '.bai', str(dst) + '.bai')
    return dst


def test_extract_tag_sniff_counts_fibertools_ma(capsys):
    from fiberhmm.cli.extract_tags import _print_tag_diagnostic, diagnose_bam_tags

    diag = diagnose_bam_tags(str(FT_FIRE))
    assert diag['has_Ma'] == 2 and diag['has_MA_AQ'] == 0
    assert diag['ma_features'] == ['fire', 'msp', 'nuc']
    assert 'Ma=2/2' in diag['summary']
    _print_tag_diagnostic(diag, ['nucleosome', 'msp', 'tf'])
    out = capsys.readouterr().out
    assert 'nucleosome[Ma nuc. annotations]' in out
    assert 'msp[Ma msp. annotations]' in out
    assert 'not detected in sniff (sparse annotations may still be present): tf' in out


def test_ma_types_scan_counts_fibertools_ma(tmp_path):
    from fiberhmm.cli import utils

    assert utils._scan_bam_ma_types(str(FT_FIRE)) == (['nuc', 'msp', 'fire'], 2, 2)
    bam = _ma_only_copy(tmp_path)
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(REPO), env.get('PYTHONPATH')]))
    proc = subprocess.run(
        [sys.executable, '-c', 'import sys; from fiberhmm.cli._entry import utils_main; '
         'sys.argv[0] = "fiberhmm-utils"; sys.exit(utils_main())',
         'ma-types', str(bam), '--scan'],
        capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    assert '(2 with MA/Ma); observed: nuc,msp,fire' in proc.stdout
    with pysam.AlignmentFile(str(bam)) as handle:
        assert 'MA-TYPES:v1:nuc,msp,fire' in handle.header.get('CO', [])


def test_advisory_scan_counts_fibertools_ma_calls():
    from fiberhmm.advisories import scan_reads

    with pysam.AlignmentFile(str(FT_FIRE)) as bam:
        assert scan_reads(bam).call_tags == 2
    # MA wins: a FiberHMM MA without nuc/msp/tf hides a decoy Ma
    header = _header('both')
    read = _read('both', False, header=header)
    read.set_tag('MA', '400;ddda_mcg.:11-20', value_type='Z')
    read.set_tag('AQ', None)

    class _One:
        def fetch(self, until_eof=True):
            return iter([read])
    assert scan_reads(_One()).call_tags == 0


def test_fibertools_ma_bam_gets_untracked_calls_advisory():
    from fiberhmm.advisories import check_bam

    ids = [advisory.id for advisory in check_bam(str(FT_FIRE), sidecars=False)]
    assert 'untracked-calls' in ids
