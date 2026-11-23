"""The DddA CpG mask stays on by default; ``--no-use-m5c`` turns it off (3.3.0).

Owner decision: keep the mask (CpGs neutral unless tag-m5c ``ddda_ucg``
islands say otherwise) as the default everywhere, with one switch for
organisms without CpG methylation (e.g. Drosophila). The switch is the
existing ``--use-m5c/--no-use-m5c`` on every DddA tool, and
``input.use_m5c`` (``--use-m5c/--no-use-m5c``) for consensus and transfer.

A replay of called BAMs (recall-tfs/-nucs, pair/merge, posteriors, qc's light
call, consensus/transfer) inherits ``off`` from the @PG record of the run that
made the calls; an explicit flag always wins.
"""
from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

import pysam
import pytest

from fiberhmm.inference.consensus.bam import check_saved_evidence, resolve_ddda_cpg_policy
from fiberhmm.inference.consensus.parameters import parse_options
from fiberhmm.nuc.records import inherit_use_m5c, recorded_cpg_mask

REPO = Path(__file__).resolve().parents[1]


def _header(*pgs):
    return {'HD': {'VN': '1.6'}, 'SQ': [{'SN': 'chr1', 'LN': 1000}], 'PG': [dict(p) for p in pgs]}


CALL_OFF = {'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call',
            'CL': 'fiberhmm-call -i a.bam -o b.bam --enzyme ddda --no-use-m5c',
            'DS': 'mode=daf enzyme=ddda cpg_mask=off hmm_cpg_mask=off'}
CALL_ON = {'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call',
           'CL': 'fiberhmm-call -i a.bam -o b.bam --enzyme ddda',
           'DS': 'mode=daf enzyme=ddda cpg_mask=unmethylated-only hmm_cpg_mask=on'}


# --------------------------------------------------------------------------- #
#  the recorded setting                                                        #
# --------------------------------------------------------------------------- #

def test_recorded_cpg_mask_reads_the_newest_caller_record():
    assert recorded_cpg_mask(_header()) is None
    assert recorded_cpg_mask(_header(CALL_OFF)) == 'off'
    assert recorded_cpg_mask(_header(CALL_ON)) == 'unmethylated-only'
    # the newest of call / recall-tfs / recall-nucs / merge decides
    recall_off = {'ID': 'fiberhmm-recall-tfs', 'PN': 'fiberhmm-recall-tfs',
                  'CL': 'fiberhmm-recall-tfs -i b.bam -o c.bam --enzyme ddda --no-use-m5c',
                  'DS': 'mode=daf enzyme=ddda cpg_mask=off'}
    assert recorded_cpg_mask(_header(CALL_ON, recall_off)) == 'off'
    recall_on = dict(recall_off, CL='fiberhmm-recall-tfs -i b.bam -o c.bam --use-m5c',
                     DS='mode=daf enzyme=ddda cpg_mask=unmethylated-only')
    assert recorded_cpg_mask(_header(CALL_OFF, recall_on)) == 'unmethylated-only'
    # a merge without --recall made no calls: the call before it decides
    merge = {'ID': 'fiberhmm-merge', 'PN': 'fiberhmm-merge', 'CL': 'fiberhmm-pair -i b.bam -o c.bam',
             'DS': 'both-strand consensus of CT/GA pairs; recall=off enzyme=n/a'}
    assert recorded_cpg_mask(_header(CALL_OFF, merge)) == 'off'
    # other programs do not count
    other = {'ID': 'samtools', 'PN': 'samtools', 'CL': 'samtools view --no-use-m5c'}
    assert recorded_cpg_mask(_header(CALL_OFF, other)) == 'off'
    # an older DddA call without the DS token: its flags, else the default mask
    old = {'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call', 'CL': 'fiberhmm-call --enzyme ddda'}
    assert recorded_cpg_mask(_header(old)) == 'unmethylated-only'
    assert recorded_cpg_mask(_header(dict(old, CL=old['CL'] + ' --no-use-m5c'))) == 'off'
    # a custom-model call was off only because it was not DddA: not a choice
    custom = {'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call', 'CL': 'fiberhmm-call -m my.json',
              'DS': 'mode=daf enzyme=custom cpg_mask=off'}
    assert recorded_cpg_mask(_header(custom)) is None


def test_explicit_flags_beat_the_record(capsys):
    off = pysam.AlignmentHeader.from_dict(_header(CALL_OFF))
    assert inherit_use_m5c(None, None, off, tool='t') is False
    assert 'as recorded' in capsys.readouterr().err
    assert inherit_use_m5c(True, None, off) is True
    assert inherit_use_m5c(None, 'methylated-only', off) is None
    assert inherit_use_m5c(None, None, pysam.AlignmentHeader.from_dict(_header(CALL_ON))) is None
    assert inherit_use_m5c(None, None, None) is None


# --------------------------------------------------------------------------- #
#  consensus / transfer input option                                           #
# --------------------------------------------------------------------------- #

def test_consensus_use_m5c_option():
    default = parse_options({})['input']
    assert default.use_m5c == 'auto'
    assert resolve_ddda_cpg_policy(default) == 'unmethylated-only'
    # true/false are accepted as on/off
    assert parse_options({'input': {'use_m5c': False}})['input'].use_m5c == 'off'
    assert parse_options({'input': {'use_m5c': True}})['input'].use_m5c == 'on'
    off = parse_options({'input': {'use_m5c': 'off'}})['input']
    assert resolve_ddda_cpg_policy(off, ('unmethylated-only',)) == 'off'
    on = parse_options({'input': {'use_m5c': 'on', 'ddda_cpg_mask_policy': 'methylated-only'}})['input']
    assert resolve_ddda_cpg_policy(on, ('off',)) == 'methylated-only'
    # the legacy switch still turns it off
    assert resolve_ddda_cpg_policy(parse_options({'input': {'ddda_m5c_correction': False}})['input']) == 'off'
    with pytest.raises(ValueError, match='contradicts'):
        parse_options({'input': {'use_m5c': 'on', 'ddda_m5c_correction': False}})
    with pytest.raises(ValueError):
        parse_options({'input': {'use_m5c': 'sometimes'}})


def test_consensus_auto_inherits_off_from_the_bams():
    auto = parse_options({})['input']
    assert resolve_ddda_cpg_policy(auto, ('off',)) == 'off'
    assert resolve_ddda_cpg_policy(auto, ('off', 'off')) == 'off'
    assert resolve_ddda_cpg_policy(auto, (None, 'unmethylated-only')) == 'unmethylated-only'
    with pytest.raises(ValueError, match='with and without'):
        resolve_ddda_cpg_policy(auto, ('off', 'unmethylated-only'))


def _saved(*policies):
    return {'strata': [{'dataset_id': f'd{i}', 'chemistry': 'ddda',
                        'model_manifest': {'emissions_sha256': 'x', 'cpg_mask_policy': p}}
                       for i, p in enumerate(policies)]}


def test_saved_evidence_never_mixes_policies():
    auto = parse_options({})
    check_saved_evidence(_saved('off'), auto)                      # recorded --no-use-m5c
    check_saved_evidence(_saved('unmethylated-only'), auto)
    with pytest.raises(ValueError, match='mixes CpG policies'):
        check_saved_evidence(_saved('off', 'unmethylated-only'), auto)
    with pytest.raises(ValueError, match='methylated-only'):
        check_saved_evidence(_saved('methylated-only'), auto)
    off = parse_options({'input': {'use_m5c': 'off'}})
    check_saved_evidence(_saved('off'), off)
    with pytest.raises(ValueError, match='this run asks for off'):
        check_saved_evidence(_saved('unmethylated-only'), off)
    on = parse_options({'input': {'use_m5c': 'on'}})
    with pytest.raises(ValueError, match='this run asks for unmethylated-only'):
        check_saved_evidence(_saved('off'), on)


# --------------------------------------------------------------------------- #
#  every tool accepts the switch                                               #
# --------------------------------------------------------------------------- #

ENTRY_POINTS = ['call_main', 'apply_main', 'recall_tfs_main', 'recall_nucs_main', 'pair_main',
                'merge_main', 'posteriors_main', 'qc_main', 'pipeline_main']


def _help(code):
    env = dict(os.environ, FIBERHMM_NO_UPDATE_CHECK='1', COLUMNS='200')
    env['PYTHONPATH'] = str(REPO) + os.pathsep + env.get('PYTHONPATH', '')
    proc = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                          cwd=REPO, env=env, timeout=300)
    return proc.stdout + proc.stderr


@pytest.mark.parametrize('entry', ENTRY_POINTS)
def test_every_ddda_tool_has_the_switch(entry):
    out = _help("import sys; sys.argv=['tool','--help']\n"
                f"from fiberhmm.cli._entry import {entry}\n"
                f"try:\n    {entry}()\nexcept SystemExit:\n    pass")
    assert '--no-use-m5c' in out, out[-2000:]


@pytest.mark.parametrize('module', ['fiberhmm.inference.consensus.cli',
                                    'fiberhmm.inference.consensus.transfer_cli'])
def test_consensus_and_transfer_have_the_switch(module):
    out = _help("import sys; sys.argv=['tool','--help']\n"
                f"from {module} import main\n"
                "try:\n    main()\nexcept SystemExit:\n    pass")
    assert '--no-use-m5c' in out, out[-2000:]


# --------------------------------------------------------------------------- #
#  end to end on the demo DddA data                                             #
# --------------------------------------------------------------------------- #

N_READS = 40


def _run(module, *args):
    env = dict(os.environ, FIBERHMM_NO_UPDATE_CHECK='1', OPENBLAS_NUM_THREADS='1')
    env['PYTHONPATH'] = str(REPO) + os.pathsep + env.get('PYTHONPATH', '')
    proc = subprocess.run([sys.executable, '-m', module, *map(str, args)], capture_output=True,
                          text=True, cwd=REPO, env=env, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return proc


def _ma(path):
    with pysam.AlignmentFile(str(path), check_sq=False) as bam:
        return {r.query_name: r.get_tag('MA') for r in bam.fetch(until_eof=True) if r.has_tag('MA')}


def _ds(path, program):
    with pysam.AlignmentFile(str(path), check_sq=False) as bam:
        return [pg for pg in bam.header.to_dict()['PG'] if pg.get('PN') == program][-1]['DS']


@pytest.fixture(scope='module')
def called(tmp_path_factory):
    root = tmp_path_factory.mktemp('cpg_off')
    runpy.run_path(str(REPO / 'docs' / 'examples' / 'make_demo_data.py'),
                   run_name='demo_data')['main'](str(root / 'full'))
    src_path = root / 'ddda.bam'
    with pysam.AlignmentFile(str(root / 'full' / 'ddda.bam')) as src, \
            pysam.AlignmentFile(str(src_path), 'wb', template=src) as dst:
        for i, read in enumerate(src):
            if i >= N_READS:
                break
            dst.write(read)
    pysam.index(str(src_path))
    out = {}
    for name, extra in (('on', ()), ('off', ('--no-use-m5c',))):
        path = root / f'{name}.bam'
        _run('fiberhmm.cli.call', '-i', src_path, '-o', path, '--enzyme', 'ddda',
             '-c', '1', '--io-threads', '1', '--no-qc', *extra)
        pysam.index(str(path))
        out[name] = path
    out['root'] = root
    return out


def test_off_changes_the_calls_where_cpgs_matter(called):
    assert 'cpg_mask=unmethylated-only' in _ds(called['on'], 'fiberhmm-call').split()
    assert 'cpg_mask=off' in _ds(called['off'], 'fiberhmm-call').split()
    on, off = _ma(called['on']), _ma(called['off'])
    assert on.keys() == off.keys()
    assert on != off


def test_recall_inherits_the_recorded_off_and_explicit_flags_win(called):
    root = called['root']
    inherited, explicit_off, explicit_on = root / 'r_inh.bam', root / 'r_off.bam', root / 'r_on.bam'
    common = ('--enzyme', 'ddda', '-c', '1', '--io-threads', '1')
    proc = _run('fiberhmm.cli.recall_tfs', '-i', called['off'], '-o', inherited, *common)
    assert 'as recorded by the run' in proc.stderr
    _run('fiberhmm.cli.recall_tfs', '-i', called['off'], '-o', explicit_off, *common, '--no-use-m5c')
    _run('fiberhmm.cli.recall_tfs', '-i', called['off'], '-o', explicit_on, *common, '--use-m5c')
    assert _ma(inherited) == _ma(explicit_off)
    assert _ma(explicit_on) != _ma(inherited)
    assert 'cpg_mask=off' in _ds(inherited, 'fiberhmm-recall-tfs').split()
    assert 'cpg_mask=unmethylated-only' in _ds(explicit_on, 'fiberhmm-recall-tfs').split()
    # a recall of a default call keeps the mask
    default = root / 'r_def.bam'
    _run('fiberhmm.cli.recall_tfs', '-i', called['on'], '-o', default, *common)
    assert 'cpg_mask=unmethylated-only' in _ds(default, 'fiberhmm-recall-tfs').split()


def test_hmm_only_off_then_recall_nucs_reproduces_the_off_call(called):
    root = called['root']
    src = root / 'ddda.bam'
    hmm = root / 'hmm_off.bam'
    _run('fiberhmm.cli.call', '-i', src, '-o', hmm, '--enzyme', 'ddda', '-c', '1',
         '--io-threads', '1', '--no-qc', '--hmm-only', '--no-use-m5c')
    recalled = root / 'hmm_off_recalled.bam'
    proc = _run('fiberhmm.cli.recall_tfs', '-i', hmm, '-o', recalled, '--enzyme', 'ddda',
                '--recall-nucs', '-c', '1', '--io-threads', '1')
    assert 'as recorded by the run' in proc.stderr
    assert _ma(recalled) == _ma(called['off'])


def test_nuc_caller_settings_inherit_off(called):
    from fiberhmm.nuc.recall import CallerSettings
    with pysam.AlignmentFile(str(called['off']), check_sq=False) as bam:
        assert CallerSettings.from_header(bam.header).cpg_mask_policy is None
        assert CallerSettings.from_header(bam.header, cpg_mask_policy='unmethylated-only').cpg_mask_policy \
            == 'unmethylated-only'
    with pysam.AlignmentFile(str(called['on']), check_sq=False) as bam:
        assert CallerSettings.from_header(bam.header).cpg_mask_policy == 'unmethylated-only'


def test_consensus_replay_inherits_off_and_explicit_wins(called):
    from fiberhmm.inference.consensus.bam import load_bam_payload
    region = dict(chrom='chrDemo', start=2000, end=2600)

    def manifest(name, options):
        payload = load_bam_payload([dict(dataset_id='d', paths=[str(called[name])], chemistry='ddda')],
                                   region, parse_options(options))
        return payload['strata'][0]['model_manifest']

    inherited = manifest('off', {})
    assert inherited['cpg_mask_policy'] == 'off'
    assert inherited['cpg_mask_source'] == 'recorded --no-use-m5c'
    assert inherited['m5c_conditioned_opportunities'] == 0
    forced = manifest('off', {'input': {'use_m5c': 'on'}})
    assert forced['cpg_mask_policy'] == 'unmethylated-only' and 'cpg_mask_source' not in forced
    assert forced['m5c_conditioned_opportunities'] > 0
    default = manifest('on', {})
    assert default['cpg_mask_policy'] == 'unmethylated-only' and 'cpg_mask_source' not in default
    assert manifest('on', {'input': {'use_m5c': False}})['cpg_mask_policy'] == 'off'
    # datasets of one run must agree
    with pytest.raises(ValueError, match='one run uses one mask'):
        load_bam_payload([dict(dataset_id='a', paths=[str(called['off'])], chemistry='ddda'),
                          dict(dataset_id='b', paths=[str(called['on'])], chemistry='ddda')],
                         region, parse_options({}))


def test_consensus_precedence_matches_the_command_line_tools():
    # a non-default ddda_cpg_mask_policy is explicit, as --cpg-mask-policy is
    meth = parse_options({'input': {'ddda_cpg_mask_policy': 'methylated-only'}})
    assert resolve_ddda_cpg_policy(meth['input'], ('off',)) == 'methylated-only'
    with pytest.raises(ValueError, match='this run asks for methylated-only'):
        check_saved_evidence(_saved('off'), meth)


def test_frozen_catalog_pins_the_source_mask():
    from fiberhmm.inference.consensus.lattice_recaller.frozen import _pin_ddda_cpg_mask
    auto = {'input': {'use_m5c': 'auto'}, 'cr': {'engine': 'lattice_recaller'}}
    assert _pin_ddda_cpg_mask(auto, _saved('off'))['input']['use_m5c'] == 'off'
    assert _pin_ddda_cpg_mask(auto, _saved('unmethylated-only'))['input']['use_m5c'] == 'on'
    assert auto['input']['use_m5c'] == 'auto'                       # not edited in place
    explicit = {'input': {'use_m5c': 'off'}}
    assert _pin_ddda_cpg_mask(explicit, _saved('off')) is explicit
    hia5 = {'strata': [{'dataset_id': 'h', 'chemistry': 'hia5-pacbio', 'model_manifest': {'emissions_sha256': 'x'}}]}
    assert _pin_ddda_cpg_mask(auto, hia5) is auto
    # a pinned catalog makes transfer strict in both directions
    pinned_on = parse_options(_pin_ddda_cpg_mask(auto, _saved('unmethylated-only')))
    with pytest.raises(ValueError, match='this run asks for unmethylated-only'):
        check_saved_evidence(_saved('off'), pinned_on)
    pinned_off = parse_options(_pin_ddda_cpg_mask(auto, _saved('off')))
    with pytest.raises(ValueError, match='this run asks for off'):
        check_saved_evidence(_saved('unmethylated-only'), pinned_off)


def test_posteriors_inherit_off(called):
    import gzip
    import json
    out = called['root'] / 'post_off.tsv.gz'
    proc = _run('fiberhmm.cli.export_posteriors', '-i', called['off'], '-o', out, '--enzyme', 'ddda',
                '--format', 'tsv', '-c', '1')
    assert 'as recorded by the run' in proc.stderr
    with gzip.open(out, 'rt') as fh:
        meta = json.loads(fh.readline().split(':', 1)[1])
    assert meta['cpg_mask_policy'] == 'off'


def test_qc_light_call_inherits_off(called):
    from fiberhmm.qc.states import compute_state_rates
    for name, want in (('off', 'off'), ('on', 'unmethylated-only')):
        with pysam.AlignmentFile(str(called[name]), check_sq=False) as bam:
            header = bam.header
            reads = [r for _, r in zip(range(5), bam.fetch(until_eof=True))]
        block, _ = compute_state_rates(reads, 'daf', 'ddda', header=header, state_source='light-call',
                                       light_call_reads=2)
        assert block['definition']['hmm_cpg_mask'] == want


# --------------------------------------------------------------------------- #
#  recorded command lines: the resolved DS value, else the last switch          #
# --------------------------------------------------------------------------- #

def test_recorded_mask_prefers_ds_then_the_last_switch():
    from fiberhmm.nuc.records import recorded_call_masks
    # a wrapper appended --use-m5c after --no-use-m5c: the run masked CpGs
    final_on = dict(CALL_ON, CL=CALL_ON['CL'] + ' --no-use-m5c --use-m5c')
    assert recorded_cpg_mask(_header(final_on)) == 'unmethylated-only'
    # older records without the DS token: the last Boolean switch wins
    old = {'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call', 'CL': 'fiberhmm-call --enzyme ddda'}
    for tail, want in ((' --no-use-m5c --use-m5c', 'unmethylated-only'),
                       (' --use-m5c --no-use-m5c', 'off')):
        hd = _header(dict(old, CL=old['CL'] + tail))
        assert recorded_cpg_mask(hd) == want
        assert recorded_call_masks(hd, 'ddda', 'daf')[2] == (None if want == 'off' else want)
    custom = {'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call',
              'CL': 'fiberhmm-call -m my.json --use-m5c --no-use-m5c',
              'DS': 'mode=daf enzyme=custom cpg_mask=off'}
    assert recorded_cpg_mask(_header(custom)) == 'off'


def test_caller_settings_honour_the_last_switch():
    from fiberhmm.nuc.recall import CallerSettings
    for tail, want in ((' --no-use-m5c --use-m5c', 'unmethylated-only'),
                       (' --use-m5c --no-use-m5c', None)):
        pg = {'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call',
              'CL': 'fiberhmm-call -i a.bam -o b.bam --enzyme ddda' + tail}
        settings = CallerSettings.from_header(pysam.AlignmentHeader.from_dict(_header(pg)))
        assert settings.cpg_mask_policy == want


# --------------------------------------------------------------------------- #
#  an inherited / explicit off survives successive recalls                      #
# --------------------------------------------------------------------------- #

def test_inherited_off_is_recorded_and_read_back():
    custom_call = {'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call',
                   'CL': 'fiberhmm-call -i a.bam -o b.bam -m my.json --no-use-m5c',
                   'DS': 'mode=daf enzyme=custom cpg_mask=off cpg_mask_source=explicit'}
    custom_recall = {'ID': 'fiberhmm-recall-tfs', 'PN': 'fiberhmm-recall-tfs',
                     'CL': 'fiberhmm-recall-tfs -i b.bam -o c.bam -m my.json',
                     'DS': 'mode=daf enzyme=custom cpg_mask=off cpg_mask_source=inherited'}
    assert recorded_cpg_mask(_header(custom_call)) == 'off'
    assert recorded_cpg_mask(_header(custom_call, custom_recall)) == 'off'
    second = dict(custom_recall, ID='fiberhmm-recall-tfs.2')
    assert recorded_cpg_mask(_header(custom_call, custom_recall, second)) == 'off'
    # an off that was only the non-DddA default is still undecided
    default = dict(custom_recall, DS='mode=daf enzyme=custom cpg_mask=off')
    assert recorded_cpg_mask(_header(CALL_OFF, default)) is None


def test_off_survives_a_custom_model_recall(called):
    # a custom-model call with --no-use-m5c (no DddA declaration), a custom
    # recall that inherits off, then a DddA replay: still off
    from fiberhmm.models import get_model_path
    root = called['root']
    common = ('-c', '1', '--io-threads', '1')
    call = root / 'c_custom.bam'
    _run('fiberhmm.cli.call', '-i', root / 'ddda.bam', '-o', call, '-m',
         get_model_path('ddda', tool='apply'), '--no-use-m5c', '--no-qc', *common)
    pysam.index(str(call))
    assert 'cpg_mask_source=explicit' in _ds(call, 'fiberhmm-call').split()
    custom = root / 'r_custom.bam'
    proc = _run('fiberhmm.cli.recall_tfs', '-i', call, '-o', custom,
                '-m', get_model_path('ddda', tool='recall'), *common)
    ds = _ds(custom, 'fiberhmm-recall-tfs').split()
    assert 'enzyme=custom' in ds and 'cpg_mask=off' in ds
    assert 'cpg_mask_source=inherited' in ds, proc.stderr[-2000:]
    pysam.index(str(custom))
    again = root / 'r_custom_ddda.bam'
    proc = _run('fiberhmm.cli.recall_tfs', '-i', custom, '-o', again, '--enzyme', 'ddda',
                '--replace-chemistry', *common)
    assert 'as recorded by the run' in proc.stderr
    ds = _ds(again, 'fiberhmm-recall-tfs').split()
    assert 'cpg_mask=off' in ds and 'cpg_mask_source=inherited' in ds
    # the default output records no source token
    assert not any(t.startswith('cpg_mask_source=') for t in _ds(called['on'], 'fiberhmm-call').split())
    assert 'cpg_mask_source=explicit' in _ds(called['off'], 'fiberhmm-call').split()

