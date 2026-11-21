"""fiberhmm-call --hmm-only (3.3.0) and the deprecated fiberhmm-apply alias.

--hmm-only writes every HMM call to MA: footprints >= --nuc-min-size bp as
nuc, smaller ones as tf (inside MSPs too), the HMM MSPs as msp. Real CLI runs
on the documented demo data (docs/examples/make_demo_data.py, a subset of
reads) and the fibertools FIRE fixtures (tests/fixtures/fire_contract).
"""
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pysam
import pytest

from fiberhmm.io.ma_tags import parse_ma_tag

REPO = Path(__file__).resolve().parents[1]
FIRE = REPO / 'tests' / 'fixtures' / 'fire_contract'
FT13 = FIRE / 'fiberhmm3.2.1_call_ft0.13_fire.bam'
FT062 = FIRE / 'fiberhmm3.2.1_call_ft0.6.2_fire.bam'
CHEMISTRY = {'hia5_pacbio': ('--enzyme', 'hia5', '--seq', 'pacbio'),
             'ddda': ('--enzyme', 'ddda')}
N_READS = 40


def _run(module, *args, timeout=600):
    env = dict(os.environ, FIBERHMM_NO_UPDATE_CHECK='1', OPENBLAS_NUM_THREADS='1')
    env['PYTHONPATH'] = str(REPO) + os.pathsep + env.get('PYTHONPATH', '')
    return subprocess.run([sys.executable, '-m', module, *map(str, args)],
                          capture_output=True, text=True, cwd=REPO, env=env,
                          timeout=timeout)


def _ok(proc):
    assert proc.returncode == 0, proc.stderr[-3000:]
    return proc


@pytest.fixture(scope='module')
def demo(tmp_path_factory):
    """The first N_READS reads of the demo Hia5 PacBio and DddA BAMs."""
    root = tmp_path_factory.mktemp('hmm_only_demo')
    full = root / 'full'
    runpy.run_path(str(REPO / 'docs' / 'examples' / 'make_demo_data.py'),
                   run_name='demo_data')['main'](str(full))
    out = {}
    for name in CHEMISTRY:
        path = root / f'{name}.bam'
        with pysam.AlignmentFile(str(full / f'{name}.bam')) as src, \
                pysam.AlignmentFile(str(path), 'wb', template=src) as dst:
            for i, read in enumerate(src):
                if i >= N_READS:
                    break
                dst.write(read)
        pysam.index(str(path))
        out[name] = path
    return out


def _calls(path):
    """{read name: {feature: set of (start, length)}} from MA."""
    out = {}
    with pysam.AlignmentFile(str(path), check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            parsed = parse_ma_tag(read.get_tag('MA')) if read.has_tag('MA') else {}
            out[read.query_name] = {f: set(map(tuple, parsed.get(f, [])))
                                    for f in ('nuc', 'msp', 'tf')}
    return out


def _last_pg(path):
    with pysam.AlignmentFile(str(path), check_sq=False) as bam:
        return bam.header.to_dict()['PG'][-1]


def _call(src, out, chem, *extra):
    return _ok(_run('fiberhmm.cli.call', '-i', src, '-o', out, *CHEMISTRY[chem],
                    '-c', '1', '--io-threads', '1', '--no-qc', *extra))


# --------------------------------------------------------------------------- #
#  classing                                                                     #
# --------------------------------------------------------------------------- #

def test_footprints_are_split_by_size_and_none_dropped():
    from fiberhmm.inference.fused_stages import build_hmm_only_result

    # 180-219 is a TF-sized footprint inside the MSP 150-259.
    apply_result = {'ns': np.array([10, 180, 260]), 'nl': np.array([140, 40, 120]),
                    'ns_scores': np.array([0.9, 0.5, 0.8]),
                    'as': np.array([150]), 'al': np.array([110])}
    result = build_hmm_only_result({'query_sequence': 'A' * 400}, apply_result, 85, True)
    assert list(zip(result['ns'], result['nl'])) == [(10, 140), (260, 120)]
    assert [(c.start, c.length) for c in result['tf_calls']] == [(180, 40)]
    assert [(c.llr, c.left_ambiguity >= 30, c.right_ambiguity >= 30)
            for c in result['tf_calls']] == [(0.0, True, True)]     # AQ bytes 0
    assert result['nq_for_kept_nucs'] == [229, 204]                 # HMM posterior bytes
    assert list(result['as']) == [150] and list(result['al']) == [110]
    # The size threshold is --nuc-min-size.
    result = build_hmm_only_result({'query_sequence': 'A' * 400}, apply_result, 130, False)
    assert list(zip(result['ns'], result['nl'])) == [(10, 140)]
    assert [(c.start, c.length) for c in result['tf_calls']] == [(180, 40), (260, 120)]
    assert result['nq_for_kept_nucs'] is None                       # written as 0


def test_circular_footprints_are_split_by_size():
    from fiberhmm.inference.fused_stages import build_hmm_only_result

    apply_result = {'circular': True, 'circular_read_length': 1000,
                    'circular_ns': [(100, 150), (990, 40)],         # tf wraps the origin
                    'circular_as': [(250, 740)], 'ns': [], 'nl': [], 'as': [], 'al': []}
    result = build_hmm_only_result({'query_sequence': 'A' * 1000}, apply_result, 85, False)
    assert result['circular_ns'] == [(100, 150)]
    assert [(c.start, c.length) for c in result['tf_calls']] == [(990, 40)]
    assert result['circular_as'] == [(250, 740)]


@pytest.mark.parametrize('chem', sorted(CHEMISTRY))
def test_hmm_only_ma_is_the_full_hmm_call_set(tmp_path, demo, chem):
    out = tmp_path / 'hmm_only.bam'
    _call(demo[chem], out, chem, '--hmm-only')
    pg = _last_pg(out)
    tokens = pg['DS'].split()
    assert pg['PN'] == 'fiberhmm-call'
    assert {'hmm_only=on', 'recall_nucs=False', 'tf_recall=off', 'nuc_min_size=85',
            'msps=hmm', 'coord=molecular'} <= set(tokens)
    calls = _calls(out)
    assert sum(len(c['tf']) for c in calls.values()) > 0
    for name, call in calls.items():
        assert all(n >= 85 for _s, n in call['nuc']), name
        assert all(n < 85 for _s, n in call['tf']), name
        # nuc and msp tile the analysed span; every tf lies inside an MSP.
        tiles = sorted(call['nuc'] | call['msp'])
        assert all(a[0] + a[1] == b[0] for a, b in zip(tiles, tiles[1:])), name
        for s, n in call['tf']:
            assert any(ms <= s and s + n <= ms + ml for ms, ml in call['msp']), name
    with pysam.AlignmentFile(str(out)) as bam:
        reads = list(bam)
    assert all(not r.has_tag('ns') and not r.has_tag('as') for r in reads)
    # The MA-only default; --legacy-tags adds ns/nl = nuc, as/al = msp.
    legacy = tmp_path / 'legacy.bam'
    _call(demo[chem], legacy, chem, '--hmm-only', '--legacy-tags')
    with pysam.AlignmentFile(str(legacy)) as bam:
        for read in bam:
            parsed = parse_ma_tag(read.get_tag('MA'))
            if parsed['nuc']:
                assert sorted(zip(read.get_tag('ns'), read.get_tag('nl'))) == \
                    sorted(map(tuple, parsed['nuc']))
            assert sorted(zip(read.get_tag('as'), read.get_tag('al'))) == \
                sorted(map(tuple, parsed['msp']))
            assert not read.has_tag('hl')


def test_nothing_lost_against_the_old_apply_legacy_view(tmp_path, demo):
    """Every footprint of apply's legacy ns/nl (the old engine, still used
    internally) is an hmm-only nuc or tf, every as/al an msp, and nothing else."""
    from fiberhmm.inference.parallel import _process_bam_streaming_pipeline
    from fiberhmm.models import get_model_path

    legacy = tmp_path / 'legacy_apply.bam'
    _process_bam_streaming_pipeline(
        input_bam=str(demo['hia5_pacbio']), output_bam=str(legacy),
        model_path=get_model_path('hia5', tool='apply', seq='pacbio'),
        train_rids=set(), edge_trim=10, circular=False, mode='pacbio-fiber',
        context_size=3, msp_min_size=0, nuc_min_size=85, min_mapq=0,
        prob_threshold=128, min_read_length=1000, with_scores=False,
        n_cores=1, chunk_size=10, io_threads=1, write_legacy_tags=True,
        primary_only='primary-supplementary')
    out = tmp_path / 'hmm_only.bam'
    _call(demo['hia5_pacbio'], out, 'hia5_pacbio', '--hmm-only')
    calls = _calls(out)
    compared = 0
    with pysam.AlignmentFile(str(legacy)) as bam:
        for read in bam:
            if not read.has_tag('ns'):
                continue
            compared += 1
            call = calls[read.query_name]
            assert set(zip(read.get_tag('ns'), read.get_tag('nl'))) == call['nuc'] | call['tf']
            assert set(zip(read.get_tag('as'), read.get_tag('al'))) == call['msp']
    assert compared == N_READS


# --------------------------------------------------------------------------- #
#  hmm-only output is recall input                                              #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('chem', sorted(CHEMISTRY))
def test_hmm_only_then_recall_nucs_reproduces_the_integrated_call(tmp_path, demo, chem):
    integrated = tmp_path / 'call.bam'
    _call(demo[chem], integrated, chem)
    hmm_only = tmp_path / 'hmm_only.bam'
    _call(demo[chem], hmm_only, chem, '--hmm-only')
    recalled = tmp_path / 'recalled.bam'
    proc = _ok(_run('fiberhmm.cli.recall_tfs', '-i', hmm_only, '-o', recalled,
                    *CHEMISTRY[chem], '--recall-nucs', '-c', '1', '--io-threads', '1'))
    assert 'fiberhmm-call --hmm-only output' in proc.stderr
    want, got = _calls(integrated), _calls(recalled)
    assert want.keys() == got.keys()
    assert sum(len(c['tf']) for c in want.values()) > 0
    assert got == want


def test_hmm_only_then_recall_tfs_matches_call_without_nucleosome_recall(tmp_path, demo):
    chem = 'hia5_pacbio'
    integrated = tmp_path / 'call.bam'
    _call(demo[chem], integrated, chem, '--no-recall-nucs')
    hmm_only = tmp_path / 'hmm_only.bam'
    _call(demo[chem], hmm_only, chem, '--hmm-only')
    recalled = tmp_path / 'recalled.bam'
    _ok(_run('fiberhmm.cli.recall_tfs', '-i', hmm_only, '-o', recalled,
             *CHEMISTRY[chem], '-c', '1', '--io-threads', '1'))
    assert _calls(recalled) == _calls(integrated)


def test_region_parallel_matches_streaming_and_train_reads_are_left_uncalled(tmp_path, demo):
    chem = 'hia5_pacbio'
    with pysam.AlignmentFile(str(demo[chem])) as bam:
        names = [read.query_name for read in bam][:2]
    train = tmp_path / 'train.tsv'
    train.write_text('rid\n' + '\n'.join(names) + '\n')
    streaming, region = tmp_path / 'streaming.bam', tmp_path / 'region.bam'
    _call(demo[chem], streaming, chem, '--hmm-only', '--train-reads', train)
    _call(demo[chem], region, chem, '--hmm-only', '--train-reads', train,
          '--region-parallel', '--region-size', '5000')
    a, b = _calls(streaming), _calls(region)
    assert a == b
    assert all(not any(a[name].values()) for name in names)
    assert sum(bool(c['nuc']) for c in a.values()) == N_READS - 2


# --------------------------------------------------------------------------- #
#  recorded-call readers                                                        #
# --------------------------------------------------------------------------- #

def _hmm_only_header(*later):
    call = {'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call', 'VN': '3.3.0',
            'CL': 'fiberhmm-call -i in.bam -o out.bam --hmm-only --enzyme ddda '
                  '--prob-threshold 100 --edge-trim 12',
            'DS': 'FiberHMM HMM-only call (no nucleosome or TF recall); coord=molecular '
                  'hmm_only=on mode=daf enzyme=ddda prob_threshold=100 recall_nucs=False '
                  'tf_recall=off nuc_min_size=85 cpg_mask=unmethylated-only'}
    return {'HD': {'VN': '1.6'}, 'PG': [call, *later]}


def test_recorded_call_readers_take_an_hmm_only_call_record():
    from fiberhmm.cli.recall_tfs import _ddda_radial_call_in_history
    from fiberhmm.io.footprint_calls import hmm_only_calls
    from fiberhmm.nuc.records import last_call_record, recorded_call_masks, \
        recorded_call_threshold_trim
    from fiberhmm.nuc.recall import CallerSettings

    header = _hmm_only_header()
    assert last_call_record(header)['DS'].split()[0:3] == ['FiberHMM', 'HMM-only', 'call']
    assert recorded_call_threshold_trim(header) == (100, 12)
    assert recorded_call_masks(header, 'ddda', 'daf')[2] == 'unmethylated-only'
    settings = CallerSettings.from_header(header)
    assert (settings.enzyme, settings.prob_threshold, settings.edge_trim) == ('ddda', 100, 12)
    assert any('--hmm-only' in note for note in settings.notes)
    # No radial recall ran, so recall-tfs does not warn about recalling it.
    assert not _ddda_radial_call_in_history(header)
    assert hmm_only_calls(header)
    # A later recall (or an old apply record) makes the tf group recalled TFs.
    recall = {'ID': 'fiberhmm-recall-nucs', 'PN': 'fiberhmm-recall-nucs', 'PP': 'fiberhmm-call',
              'DS': 'recall_nucs=True'}
    assert not hmm_only_calls(_hmm_only_header(recall))
    sort = {'ID': 'samtools', 'PN': 'samtools', 'PP': 'fiberhmm-call'}
    assert hmm_only_calls(_hmm_only_header(sort))
    assert not hmm_only_calls({'PG': [{'ID': 'fiberhmm-apply', 'PN': 'fiberhmm-apply'}]})


def test_recall_reads_nuc_plus_tf_only_from_hmm_only_files():
    from fiberhmm.io.footprint_calls import resolve_footprint_tags

    header = pysam.AlignmentHeader.from_dict({'HD': {'VN': '1.6'},
                                              'SQ': [{'SN': 'chr1', 'LN': 10_000}]})
    read = pysam.AlignedSegment(header)
    read.query_name, read.flag, read.reference_id, read.reference_start = 'r', 0, 0, 0
    read.query_sequence, read.cigarstring = 'A' * 400, '400M'
    read.set_tag('MA', '400;nuc.Q:11-140,261-120;msp.:151-110;tf+QQQ:181-40', value_type='Z')
    read.set_tag('AQ', [0, 0, 0, 0, 0])
    plain = resolve_footprint_tags(read, True)
    assert list(plain['ns']) == [10, 260] and list(plain['nq']) == [0, 0]
    merged = resolve_footprint_tags(read, True, hmm_footprints=True)
    assert list(zip(merged['ns'], merged['nl'])) == [(10, 140), (180, 40), (260, 120)]
    assert 'nq' not in merged and list(merged['as']) == [150]


# --------------------------------------------------------------------------- #
#  --keep-msps                                                                  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('fixture', [FT13, FT062], ids=['ft0.13', 'ft0.6.2'])
def test_keep_msps_keeps_the_input_msps_and_fibertools_fire(tmp_path, fixture):
    out = tmp_path / 'kept.bam'
    _ok(_run('fiberhmm.cli.call', '-i', fixture, '-o', out, '--enzyme', 'hia5',
             '--seq', 'pacbio', '--force-seq', '--hmm-only', '--keep-msps', '-c', '1',
             '--io-threads', '1', '--no-qc', '--min-read-length', '0'))
    assert 'msps=kept' in _last_pg(out)['DS'].split()
    with pysam.AlignmentFile(str(fixture)) as bam:
        before = {r.query_name: r for r in bam}
    n_fire = 0
    with pysam.AlignmentFile(str(out)) as bam:
        for read in bam:
            orig = before[read.query_name]
            src_ma = str(orig.get_tag('MA') if orig.has_tag('MA') else orig.get_tag('Ma'))
            src_groups = {g.partition(':')[0]: g for g in src_ma.split(';')[1:]}
            groups = {g.partition(':')[0]: g for g in str(read.get_tag('MA')).split(';')[1:]}
            assert groups['msp.'] == src_groups['msp.']                 # input MSPs
            assert any(g.startswith('nuc') for g in groups)             # new HMM nucs
            for name in [g for g in src_groups if g.startswith('fire')]:
                assert groups[name] == src_groups[name]                 # ft >= 0.13 FIRE
                n = len(groups[name].partition(':')[2].split(','))
                src_aq = orig.get_tag('AQ') if orig.has_tag('AQ') else orig.get_tag('Aq')
                assert list(read.get_tag('AQ'))[-n:] == list(src_aq)[-n:]
                n_fire += 1
            # legacy MSPs and fibertools <= 0.12 FIRE aq stay; ns/nl/nq go
            for tag in ('as', 'al', 'aq'):
                assert read.has_tag(tag) == orig.has_tag(tag)
                if orig.has_tag(tag):
                    assert list(read.get_tag(tag)) == list(orig.get_tag(tag))
            assert not any(read.has_tag(t) for t in ('ns', 'nl', 'nq', 'Ma', 'Aq'))
    assert n_fire == (1 if fixture == FT13 else 0)


def test_option_conflicts_are_refused(tmp_path, demo):
    src = demo['hia5_pacbio']
    for extra, message in ((('--keep-msps',), '--keep-msps requires --hmm-only'),
                           (('--hmm-only', '--recall-nucs'), '--recall-nucs conflicts'),
                           (('--hmm-only', '--recall-model', src), '--recall-model has no effect')):
        proc = _run('fiberhmm.cli.call', '-i', src, '-o', tmp_path / 'x.bam',
                    *CHEMISTRY['hia5_pacbio'], *extra)
        assert proc.returncode == 2 and message in proc.stderr, proc.stderr
        assert not (tmp_path / 'x.bam').exists()


# --------------------------------------------------------------------------- #
#  fiberhmm-apply alias                                                         #
# --------------------------------------------------------------------------- #

def test_apply_alias_maps_its_options():
    from fiberhmm.cli.apply import call_argv, parse_args

    argv, notes = call_argv(parse_args([
        '-i', 'in/x.bam', '-o', 'outdir', '--enzyme', 'hia5', '--seq', 'nanopore',
        '--no-msps', '--scores', '-t', 'train.tsv', '--stats', '--stats-sample', '500',
        '--stats-seed', '7', '--harmonize-legacy-tags', '--output-posteriors', 'p.h5',
        '--debug-timing', '--streaming', '--no-process-unmapped', '-r']))
    assert argv[:5] == ['-i', 'in/x.bam', '-o', os.path.join('outdir', 'x_footprints.bam'),
                        '--hmm-only']

    def value(flag):
        return argv[argv.index(flag) + 1]
    assert (value('--enzyme'), value('--seq'), value('--train-reads')) == \
        ('hia5', 'nanopore', 'train.tsv')
    assert (value('--qc-sample-reads'), value('--qc-seed')) == ('500', '7')
    assert (value('--cores'), value('--io-threads')) == ('1', '4')     # apply's defaults
    assert {'--keep-msps', '--scores', '--legacy-tags', '--no-process-unmapped',
            '--circular'} <= set(argv)
    assert '--no-qc' not in argv
    joined = ' '.join(notes)
    for word in ('--harmonize-legacy-tags', '--output-posteriors', 'fiberhmm-posteriors',
                 '--debug-timing', '--streaming', '--stats'):
        assert word in joined
    # Without --stats the alias writes no QC report (apply wrote none).
    argv, _ = call_argv(parse_args(['-i', '-', '-o', '-', '--enzyme', 'dddb']))
    assert argv[argv.index('-o') + 1] == '-' and '--no-qc' in argv


def test_apply_alias_runs_call_hmm_only(tmp_path, demo):
    chem = 'hia5_pacbio'
    proc = _ok(_run('fiberhmm.cli.apply', '-i', demo[chem], '-o', tmp_path / 'apply',
                    *CHEMISTRY[chem], '--harmonize-legacy-tags', '--debug-timing'))
    assert 'fiberhmm-apply is deprecated' in proc.stderr
    assert 'removed in the release after 3.3' in proc.stderr
    assert '--harmonize-legacy-tags is gone' in proc.stderr
    assert '--debug-timing ignored' in proc.stderr
    out = tmp_path / 'apply' / 'hia5_pacbio_footprints.bam'
    pg = _last_pg(out)
    assert pg['PN'] == 'fiberhmm-call' and pg['CL'].startswith('fiberhmm-call -i ')
    assert '--hmm-only' in pg['CL'].split() and 'hmm_only=on' in pg['DS'].split()
    assert not (tmp_path / 'apply' / 'qc').exists()
    direct = tmp_path / 'direct.bam'
    _call(demo[chem], direct, chem, '--hmm-only', '--legacy-tags')
    assert _calls(out) == _calls(direct)
    with pysam.AlignmentFile(str(out)) as bam:
        assert all(r.has_tag('as') for r in bam)                       # --legacy-tags


def test_apply_alias_keeps_rejecting_never_implemented_options(tmp_path):
    proc = _run('fiberhmm.cli.apply', '-i', FT062, '-o', tmp_path, '--enzyme', 'hia5',
                '--chroms', 'chr1')
    assert proc.returncode == 2 and '--chroms' in proc.stderr


def test_apply_alias_entry_point_is_kept():
    from fiberhmm.cli._entry import apply_main
    assert callable(apply_main)
