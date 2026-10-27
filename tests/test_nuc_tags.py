"""MA groups nucsz/nucseg/nucmol and their lifecycle (NUC P-C).

- writing, reading back and staleness of the groups;
- writers that regenerate calls drop them (write_ma_tags, strip_stale_call_tags,
  and the call / recall / apply CLIs), with their header declarations;
- other groups (m5C islands, other tools' layers) are kept.
"""
from __future__ import annotations

import array
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pysam
import pytest

from nuc_synth import flat_calibration, header, make_record, planted_record_molecule

from fiberhmm.io.bam_header import (MA_TYPES_PREFIX, SCAN_MA_CONTRACT_PREFIX, declared_ma_types,
                                    drop_derived_ma_contracts)
from fiberhmm.nuc.molecule import molecule_outputs
from fiberhmm.nuc.records import DecodeContext, decode_record
from fiberhmm.nuc.tags import (NUC_GROUPS, NUCMOL, NUCMOL_CODECS, NUCSEG, NUCSZ, annotate_header,
                               annotate_record, decode_nucmol, encode_nucmol, encode_scalar,
                               decode_scalar, read_nuc_groups)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_scalar_codec_round_trip_missing_and_clipping():
    for name, nb, sc, off, _u in NUCMOL_CODECS:
        mid = off + sc * (100 if nb == 1 else 1000)
        assert abs(decode_scalar(encode_scalar(mid, nb, sc, off), nb, sc, off) - mid) < 1e-9
        assert decode_scalar(encode_scalar(float('nan'), nb, sc, off), nb, sc, off) is None
        assert decode_scalar(encode_scalar(None, nb, sc, off), nb, sc, off) is None
    assert encode_scalar(1e9, 1, 1.0, 100.0) == [254]
    assert encode_scalar(-5, 1, 1.0, 100.0) == [0]
    q = encode_nucmol({'nrl': 189.0, 'regularity': 0.3, 'lambda_acc': 0.31234, 'rho_read': 0.01612,
                       'n_mono': 300.0}, {'too_short_snr': True, 'low_information': True})
    vals, flags = decode_nucmol(q)
    assert vals['nrl'] == 189.0 and abs(vals['regularity'] - 0.296) < 0.005
    assert abs(vals['rho_read'] - 0.01612) < 2e-5 and abs(vals['lambda_acc'] - 0.31234) < 2e-5
    assert vals['n_mono'] == 300.0 and vals['gap_cv'] is None
    assert flags['too_short_snr'] and flags['low_information'] and not flags['too_short_nrl']


def _annotated_read(tmp_path, reverse=False, extra_groups=()):
    rng = np.random.default_rng(1)
    seq, nucs, msps, marks = planted_record_molecule(rng, n_nuc=14)
    tfs = [(nucs[4][1] + 5, nucs[4][1] + 30)]
    path = str(tmp_path / 'a.bam')
    with pysam.AlignmentFile(path, 'wb', header=header()) as out:
        out.write(make_record(out, 'r1', seq, 1000, nucs, msps, tfs, marks, reverse=reverse,
                              extra_groups=extra_groups))
    bam = pysam.AlignmentFile(path)
    read = next(bam.fetch(until_eof=True))
    mol, _ = decode_record(read, DecodeContext(bam.header))
    res = molecule_outputs(mol, cal=flat_calibration())
    annotate_record(read, res)
    return bam, read, mol, res


@pytest.mark.parametrize('reverse', [False, True])
def test_annotate_and_read_back(tmp_path, reverse):
    keep = (('ddda_ucg.:5-20', ()), ('fire.Q:30-10', (77,)))
    bam, read, mol, res = _annotated_read(tmp_path, reverse, keep)
    ma = read.get_tag('MA')
    heads = [g.partition(':')[0] for g in ma.split(';')[1:]]
    assert heads[:5] == ['nuc.QQQ', 'msp.', 'tf.QQQ', 'ddda_ucg.', 'fire.Q']
    assert {h.split('.')[0] for h in heads[5:]} == set(NUC_GROUPS)
    names = read.get_tag('AN').split(',')
    nuc_names = [n for n in names if n.startswith(('nucsz@', 'nucseg@'))]
    assert len(nuc_names) == len(set(nuc_names))                 # unique per interval
    assert all(f'@{res.digest}' in n for n in nuc_names)
    assert names.count('nucmol') == 1
    g = read_nuc_groups(read)
    assert not any(v['stale'] for v in g.values())
    assert len(g[NUCSZ]['intervals']) == len(mol.nucs) + len(mol.tfs)
    sz = sorted(zip(g[NUCSZ]['intervals'], g[NUCSZ]['q']))
    fp = res.footprints
    for (s, n), q in sz:
        i = int(np.flatnonzero((fp['start'] == s) & (fp['length'] == n))[0])
        assert q[0] & 15 == fp['coarse'][i] and q[2] == round(255 * fp['posterior'][i])
        assert bool(q[0] & 16) == bool(fp['tf_overlap'][i])
    assert len(g[NUCSEG]['intervals']) == len(res.segments)
    assert g[NUCMOL]['intervals'] == [(0, mol.length)]
    assert g[NUCMOL]['scalars']['nrl'] == round(res.scalars['nrl'])
    # AQ stayed aligned: the preserved fire.Q byte is still 77
    from fiberhmm.io.ma_tags import parse_aq_array, parse_ma_tag
    p = parse_ma_tag(ma)
    q = parse_aq_array(read.get_tag('AQ'), [t[2] for t in p['raw_types']], [len(t[3]) for t in p['raw_types']])
    fire_i = sum(len(t[3]) for t in p['raw_types'][:4])
    assert q[fire_i] == [77]
    # re-annotating replaces the groups (idempotent)
    before = (read.get_tag('MA'), list(read.get_tag('AQ')), read.get_tag('AN'))
    annotate_record(read, res)
    assert (read.get_tag('MA'), list(read.get_tag('AQ')), read.get_tag('AN')) == before


def test_changed_calls_make_the_groups_stale(tmp_path):
    bam, read, mol, res = _annotated_read(tmp_path)
    ma = read.get_tag('MA')
    groups = ma.split(';')
    nuc = groups[1]
    head, body = nuc.split(':')
    items = body.split(',')
    s, n = items[2].split('-')
    items[2] = f'{s}-{int(n) + 7}'
    groups[1] = head + ':' + ','.join(items)
    read.set_tag('MA', ';'.join(groups))
    g = read_nuc_groups(read)
    assert all(v['stale'] for v in g.values())


def test_write_ma_tags_drops_call_derived_groups_and_keeps_others(tmp_path):
    from fiberhmm.inference.tf_recaller import write_ma_tags
    keep = (('ddda_ucg.:5-20', ()), ('fire.Q:30-10', (77,)), ('fpcls.QQQ:100-50', (1, 200, 1)))
    bam, read, mol, res = _annotated_read(tmp_path, False, keep)
    L = read.query_length
    write_ma_tags(read, L, [], [(400, 147)], [(0, 400), (547, L - 547)])
    heads = [g.partition(':')[0].split('.')[0] for g in read.get_tag('MA').split(';')[1:]]
    assert heads == ['nuc', 'msp', 'ddda_ucg', 'fire']
    assert not read_nuc_groups(read)
    aq = list(read.get_tag('AQ'))
    assert aq[-1] == 77


@pytest.mark.parametrize('keep_m5c', [True, False])
def test_strip_stale_call_tags_drops_nuc_groups(tmp_path, keep_m5c):
    from fiberhmm.inference.read_filters import strip_stale_call_tags
    bam, read, mol, res = _annotated_read(tmp_path, False, (('ddda_ucg.:5-20', ()),))
    strip_stale_call_tags(read, keep_m5c_groups=keep_m5c)
    if keep_m5c:
        assert read.get_tag('MA').split(';')[1].startswith('ddda_ucg.')
        assert len(read.get_tag('MA').split(';')) == 2
    else:
        assert not read.has_tag('MA')
    assert not read_nuc_groups(read)


def test_header_declarations_are_written_and_dropped():
    h = pysam.AlignmentHeader.from_dict(header(extra_co=[MA_TYPES_PREFIX + 'nuc,msp,tf', 'keep me']))
    h2 = annotate_header(h, calibration_digest='ab' * 32, params_digest='cd' * 32,
                         calls_program='fiberhmm-call')
    co = h2.to_dict()['CO']
    contracts = [json.loads(c[len(SCAN_MA_CONTRACT_PREFIX):]) for c in co if c.startswith(SCAN_MA_CONTRACT_PREFIX)]
    assert sorted(c['group'] for c in contracts) == sorted(NUC_GROUPS)
    assert all(c['coordinates'] == 'molecular' and c['module'] == 'nuc' for c in contracts)
    assert set(NUC_GROUPS) <= set(declared_ma_types(h2))
    # annotating twice keeps one line per group
    h3 = annotate_header(h2)
    assert sum(c.startswith(SCAN_MA_CONTRACT_PREFIX) for c in h3.to_dict()['CO']) == 3
    d = drop_derived_ma_contracts(h3).to_dict()['CO']
    assert not any(c.startswith(SCAN_MA_CONTRACT_PREFIX) for c in d)
    assert 'keep me' in d and MA_TYPES_PREFIX + 'nuc,msp,tf' in d
    assert not (set(NUC_GROUPS) & set(declared_ma_types(drop_derived_ma_contracts(h3))))


# --- CLI lifecycle: call, recall and apply drop the groups --------------------------------

def _run_cli(module, *args, timeout=240):
    env = dict(os.environ)
    env.setdefault('FIBERHMM_NO_UPDATE_CHECK', '1')
    env['PYTHONPATH'] = str(REPO_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    return subprocess.run([sys.executable, '-m', module, *map(str, args)], cwd=REPO_ROOT,
                          capture_output=True, timeout=timeout, env=env)


@pytest.fixture(scope='module')
def annotated_bam(tmp_path_factory):
    """Synthetic FiberHMM-called BAM whose reads carry nuc groups and contracts."""
    tmp = tmp_path_factory.mktemp('nuc_lifecycle')
    rng = np.random.default_rng(5)
    path = str(tmp / 'annotated.bam')
    hdr = pysam.AlignmentHeader.from_dict(header())
    recs = []
    for i in range(4):
        seq, nucs, msps, marks = planted_record_molecule(rng, n_nuc=10, mark_rate=0.4)
        recs.append((1000 + i * 6000, f'read{i}', seq, nucs, msps, marks, i % 2 == 1))
    cal = flat_calibration()
    with pysam.AlignmentFile(path, 'wb', header=annotate_header(hdr, calls_program='fiberhmm-call')) as out:
        ctx = DecodeContext(out.header)
        for pos, name, seq, nucs, msps, marks, rev in recs:
            r = make_record(out, name, seq, pos, nucs, msps, (), marks, reverse=rev,
                            extra_groups=(('fire.Q:30-10', (77,)),))
            r.set_tag('ns', array.array('I', [s for s, _ in nucs]))
            r.set_tag('nl', array.array('I', [e - s for s, e in nucs]))
            mol, _ = decode_record(r, ctx)
            annotate_record(r, molecule_outputs(mol, cal=cal))
            out.write(r)
    pysam.index(path)
    with pysam.AlignmentFile(path) as bam:
        assert all(read_nuc_groups(r) for r in bam.fetch(until_eof=True))
    return path


def _assert_no_nuc_groups(path):
    with pysam.AlignmentFile(str(path), check_sq=False) as bam:
        co = bam.header.to_dict().get('CO', [])
        reads = list(bam.fetch(until_eof=True))
    assert reads
    assert not any(c.startswith(SCAN_MA_CONTRACT_PREFIX) for c in co)
    for r in reads:
        if r.has_tag('MA'):
            heads = {g.partition(':')[0].split('.')[0] for g in r.get_tag('MA').split(';')[1:]}
            assert not heads & set(NUC_GROUPS), heads
        if r.has_tag('AN'):
            assert '@' not in r.get_tag('AN') and 'nucmol' not in r.get_tag('AN')


def test_call_drops_nuc_groups(annotated_bam, tmp_path, benchmark_model_path):
    out = tmp_path / 'called.bam'
    res = _run_cli('fiberhmm.cli.call', '-i', annotated_bam, '-o', out, '-m', benchmark_model_path,
                   '--min-read-length', '0', '--no-qc', '-c', '1', '--io-threads', '1')
    assert res.returncode == 0, res.stderr.decode(errors='replace')
    _assert_no_nuc_groups(out)


def test_recall_drops_nuc_groups(annotated_bam, tmp_path, benchmark_model_path):
    out = tmp_path / 'recalled.bam'
    res = _run_cli('fiberhmm.cli.recall_tfs', '-i', annotated_bam, '-o', out,
                   '--model', benchmark_model_path)
    assert res.returncode == 0, res.stderr.decode(errors='replace')
    _assert_no_nuc_groups(out)
    with pysam.AlignmentFile(str(out), check_sq=False) as bam:
        r = next(bam.fetch(until_eof=True))
        assert 'fire.Q' in r.get_tag('MA')                    # other tools' layers survive


def test_apply_drops_nuc_groups(annotated_bam, tmp_path, benchmark_model_path):
    out = tmp_path / 'applied.bam'
    res = _run_cli('fiberhmm.cli.apply', '-i', annotated_bam, '-o', '-', '-m', benchmark_model_path,
                   '--min-read-length', '0', '-c', '1', '--io-threads', '1')
    assert res.returncode == 0, res.stderr.decode(errors='replace')
    out.write_bytes(res.stdout)
    _assert_no_nuc_groups(out)


def test_msp_only_record_nucmol_is_not_stale(tmp_path):
    rng = np.random.default_rng(4)
    seq, nucs, msps, marks = planted_record_molecule(rng, n_nuc=3)
    path = str(tmp_path / 'msp.bam')
    L = len(seq)
    with pysam.AlignmentFile(path, 'wb', header=header()) as out:
        out.write(make_record(out, 'm', seq, 1000, (), [(0, L)], (), marks))
    with pysam.AlignmentFile(path) as bam:
        r = next(bam.fetch(until_eof=True))
        mol, _ = decode_record(r, DecodeContext(bam.header))
        res = molecule_outputs(mol, cal=flat_calibration())
        annotate_record(r, res)
        g = read_nuc_groups(r)
        assert set(g) == {NUCMOL} and not g[NUCMOL]['stale']
        r.set_tag('MA', f'{L};msp.:1-{L - 10};' + r.get_tag('MA').split(';', 2)[2])
        assert read_nuc_groups(r)[NUCMOL]['stale']


def test_recall_nucs_drops_groups_on_reads_without_calls(tmp_path):
    from fiberhmm.io.ma_tags import drop_derived_groups
    rng = np.random.default_rng(8)
    seq, nucs, msps, marks = planted_record_molecule(rng, n_nuc=6)
    L = len(seq)
    path = str(tmp_path / 'tfonly.bam')
    hdr = annotate_header(pysam.AlignmentHeader.from_dict(header()), calls_program='fiberhmm-call')
    with pysam.AlignmentFile(path, 'wb', header=hdr) as out:
        r = make_record(out, 'tfonly', seq, 1000, (), (), [(400, 430)], marks,
                        extra_groups=(('fire.Q:30-10', (77,)), ('nucsz.QQQQ:401-30', (1, 0, 255, 60))))
        r.set_tag('AN', '.,.,nucsz@deadbeef_0')
        out.write(r)
    pysam.index(path)
    res = _run_cli('fiberhmm.cli.recall_tfs', '-i', path, '-o', tmp_path / 'out.bam', '--recall-nucs',
                   '--no-legacy-tags', '--enzyme', 'hia5', '--seq', 'pacbio', '--force-seq')
    assert res.returncode == 0, res.stderr.decode(errors='replace')
    _assert_no_nuc_groups(tmp_path / 'out.bam')
    with pysam.AlignmentFile(str(tmp_path / 'out.bam'), check_sq=False) as bam:
        assert 'fire.Q' in next(bam.fetch(until_eof=True)).get_tag('MA')


def test_reader_refuses_other_group_versions_and_lengths():
    """A header declaring nucseg version 1, or a 24-byte nucmol, is flagged
    unsupported and stale, never decoded as version 2 (review case)."""
    import array
    import pysam
    from nuc_synth import header
    from fiberhmm.io.scan_ma import scan_contract_line
    from fiberhmm.nuc.tags import read_nuc_groups, record_calls_digest
    h = pysam.AlignmentHeader.from_dict(header(extra_co=[scan_contract_line({'group': 'nucseg', 'group_version': 1})]))
    r = pysam.AlignedSegment(h)
    r.query_sequence = 'A' * 1000
    r.set_tag('MA', '1000;nuc.:101-147;nucseg.QQQ:1-1000')
    r.set_tag('AQ', array.array('B', [2, 200, 0]))
    r.set_tag('AN', '.,nucseg@' + record_calls_digest(r) + '_0')
    g = read_nuc_groups(r)['nucseg']
    assert g['unsupported'] and g['stale'] and g['version'] == 1
    h2 = pysam.AlignmentHeader.from_dict(header())
    r2 = pysam.AlignedSegment(h2)
    r2.query_sequence = 'A' * 1000
    r2.set_tag('MA', '1000;nuc.:101-147;nucmol.' + 'Q' * 24 + ':1-1000')
    r2.set_tag('AQ', array.array('B', [0] * 24))
    r2.set_tag('AN', '.,nucmol')
    g2 = read_nuc_groups(r2)['nucmol']
    assert g2['unsupported'] and g2['stale'] and 'scalars' not in g2
