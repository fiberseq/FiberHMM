"""--nuc-recall-policy topology (opt-in again in 3.3.0).

Real CLI runs on the documented demo data (docs/examples/make_demo_data.py, a
subset of reads): topology runs for Hia5 Nanopore, Hia5 PacBio and DddB and
changes nucleosome calls on the ONT Hia5 demo; the default is conservative and
its output unchanged; DddA refuses topology (radial recall), whether the
enzyme is given or inherited from the input header; the resolved policy is
recorded in @PG DS and read back by fiberhmm.nuc and the consensus replay.
"""
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pysam
import pytest

from fiberhmm.io.ma_tags import parse_ma_tag

REPO = Path(__file__).resolve().parents[1]
CHEMISTRY = {'hia5_nanopore': ('--enzyme', 'hia5', '--seq', 'nanopore'),
             'hia5_pacbio': ('--enzyme', 'hia5', '--seq', 'pacbio'),
             'dddb': ('--enzyme', 'dddb'),
             'ddda': ('--enzyme', 'ddda')}
N_READS = 40
REFUSAL = 'DddA nucleosome recall uses the radial model, which has no topology policy'


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
    """The first N_READS reads of the demo BAMs."""
    root = tmp_path_factory.mktemp('topology_demo')
    full = root / 'full'
    runpy.run_path(str(REPO / 'docs' / 'examples' / 'make_demo_data.py'),
                   run_name='demo_data')['main'](str(full))
    out = {'root': root}
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


def _call(demo, chem, tag, *extra):
    out = demo['root'] / f'{chem}.{tag}.bam'
    _ok(_run('fiberhmm.cli.call', '-i', demo[chem], '-o', out, *CHEMISTRY[chem],
             '-c', '1', '--io-threads', '1', '--no-qc', *extra))
    return out


def _records(path):
    with pysam.AlignmentFile(str(path), check_sq=False) as bam:
        return [r.to_string() for r in bam.fetch(until_eof=True)]


def _nucs(path):
    out = {}
    with pysam.AlignmentFile(str(path), check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            parsed = parse_ma_tag(read.get_tag('MA')) if read.has_tag('MA') else {}
            out[read.query_name] = sorted(map(tuple, parsed.get('nuc', [])))
    return out


def _header(path):
    with pysam.AlignmentFile(str(path), check_sq=False) as bam:
        return bam.header.to_dict()


def _ds_policy(path):
    ds = str(_header(path)['PG'][-1].get('DS', ''))
    return [t.split('=', 1)[1] for t in ds.split() if t.startswith('nuc_recall_policy=')]


@pytest.fixture(scope='module')
def ont(demo):
    return dict(default=_call(demo, 'hia5_nanopore', 'default'),
                conservative=_call(demo, 'hia5_nanopore', 'conservative',
                                   '--nuc-recall-policy', 'conservative'),
                topology=_call(demo, 'hia5_nanopore', 'topology',
                               '--nuc-recall-policy', 'topology'))


# --------------------------------------------------------------------------- #
#  calls                                                                       #
# --------------------------------------------------------------------------- #

def test_default_is_conservative_and_unchanged(ont):
    assert _records(ont['default']) == _records(ont['conservative'])
    assert _ds_policy(ont['default']) == ['conservative']


def test_topology_calls_ont_hia5_and_differs_from_conservative(ont):
    cons, topo = _nucs(ont['conservative']), _nucs(ont['topology'])
    assert set(cons) == set(topo) and len(topo) == N_READS
    assert all(topo.values()), 'topology left reads without nucleosomes'
    differing = sum(cons[name] != topo[name] for name in cons)
    assert differing > 0
    assert _ds_policy(ont['topology']) == ['topology']


@pytest.mark.parametrize('chem', ['hia5_pacbio', 'dddb'])
def test_topology_runs_for_hia5_pacbio_and_dddb(demo, chem):
    out = _call(demo, chem, 'topology', '--nuc-recall-policy', 'topology')
    assert _ds_policy(out) == ['topology']
    assert any(_nucs(out).values())


def test_recall_nucs_records_topology_and_readers_return_it(demo, ont):
    from fiberhmm.nuc.recall import CallerSettings
    from fiberhmm.nuc.records import recorded_nuc_recall_policy

    assert recorded_nuc_recall_policy(_header(ont['topology'])) == 'topology'
    assert CallerSettings.from_header(_header(ont['topology'])).nuc_recall_policy == 'topology'
    assert CallerSettings.from_header(_header(ont['default'])).nuc_recall_policy == 'conservative'
    # recall-nucs on top of a conservative call: its own record wins
    out = demo['root'] / 'ont.recall_nucs.topology.bam'
    _ok(_run('fiberhmm.cli.recall_tfs',
             '-i', ont['default'], '-o', out, '--recall-nucs', *CHEMISTRY['hia5_nanopore'],
             '--nuc-recall-policy', 'topology', '-c', '1'))
    assert _ds_policy(out) == ['topology']
    assert recorded_nuc_recall_policy(_header(out)) == 'topology'
    assert CallerSettings.from_header(_header(out)).nuc_recall_policy == 'topology'
    # recall-nucs with the default on a topology call says how to reproduce it
    proc = _ok(_run('fiberhmm.cli.recall_tfs', '-i', ont['topology'],
                    '-o', demo['root'] / 'ont.recall_nucs.default.bam', '--recall-nucs',
                    *CHEMISTRY['hia5_nanopore'], '-c', '1'))
    assert '--nuc-recall-policy topology to reproduce' in proc.stderr


# --------------------------------------------------------------------------- #
#  DddA refusal                                                                #
# --------------------------------------------------------------------------- #

def test_ddda_topology_is_refused_by_call(demo):
    proc = _run('fiberhmm.cli.call', '-i', demo['ddda'], '-o', demo['root'] / 'x.bam',
                '--enzyme', 'ddda', '--nuc-recall-policy', 'topology')
    assert proc.returncode == 2
    assert REFUSAL in proc.stderr
    assert not (demo['root'] / 'x.bam').exists()


def test_ddda_topology_is_refused_when_the_enzyme_is_inherited(demo):
    from fiberhmm.models import get_model_path

    called = _call(demo, 'ddda', 'default')
    # a custom --model without --enzyme inherits DddA from the input's declaration
    model = get_model_path('ddda', tool='recall')
    for extra in (('--model', model), ('--enzyme', 'ddda')):
        proc = _run('fiberhmm.cli.recall_tfs', '-i', called, '-o', demo['root'] / 'y.bam',
                    '--recall-nucs', '--nuc-recall-policy', 'topology', *extra)
        assert proc.returncode == 2, proc.stderr[-2000:]
        assert REFUSAL in proc.stderr
    assert not (demo['root'] / 'y.bam').exists()
    # fiberhmm-call: a custom -m without --enzyme on a DddA-declared input
    proc = _run('fiberhmm.cli.call', '-i', called, '-o', demo['root'] / 'z.bam',
                '-m', get_model_path('ddda', tool='apply'), '--nuc-recall-policy', 'topology')
    assert proc.returncode == 2, proc.stderr[-2000:]
    assert REFUSAL in proc.stderr
    assert not (demo['root'] / 'z.bam').exists()


# --------------------------------------------------------------------------- #
#  units                                                                       #
# --------------------------------------------------------------------------- #

def _pg(pid, pn, ds):
    return {'ID': pid, 'PN': pn, 'DS': ds}


def test_recorded_policy_follows_the_latest_nucleosome_recall():
    from fiberhmm.nuc.records import recorded_nuc_recall_policy

    call_topo = _pg('fiberhmm-call', 'fiberhmm-call', 'x recall_nucs=True nuc_recall_policy=topology')
    tfs_only = _pg('fiberhmm-recall-tfs', 'fiberhmm-recall-tfs',
                   'x recall_nucs=False nuc_recall_policy=off')
    nucs_cons = _pg('fiberhmm-recall-nucs', 'fiberhmm-recall-nucs',
                    'x recall_nucs=True nuc_recall_policy=conservative')
    hmm_only = _pg('fiberhmm-call.1', 'fiberhmm-call',
                   'x hmm_only=on recall_nucs=False nuc_recall_policy=conservative')
    assert recorded_nuc_recall_policy({'PG': [call_topo]}) == 'topology'
    assert recorded_nuc_recall_policy({'PG': [call_topo, tfs_only]}) == 'topology'
    assert recorded_nuc_recall_policy({'PG': [call_topo, nucs_cons]}) == 'conservative'
    assert recorded_nuc_recall_policy({'PG': [call_topo, hmm_only]}) is None
    assert recorded_nuc_recall_policy({'PG': [hmm_only, _pg('r', 'fiberhmm-recall-nucs',
                                      'recall_nucs=True nuc_recall_policy=topology')]}) == 'topology'
    assert recorded_nuc_recall_policy({'PG': []}) is None


def test_refusal_is_ddda_only():
    from fiberhmm.inference.fused_stages import nuc_recall_policy_error

    assert REFUSAL in nuc_recall_policy_error('topology', 'ddda')
    for enzyme in ('hia5', 'dddb', None):
        assert nuc_recall_policy_error('topology', enzyme) is None
    for policy in ('auto', 'conservative'):
        assert nuc_recall_policy_error(policy, 'ddda') is None


def test_caller_settings_refuses_an_explicit_ddda_topology():
    from fiberhmm.nuc.recall import CallerSettings

    header = {'HD': {'VN': '1.6'}, 'SQ': [{'SN': 'c', 'LN': 1000}],
              'PG': [_pg('fiberhmm-call', 'fiberhmm-call',
                         'mode=daf enzyme=ddda recall_nucs=True nuc_recall_policy=topology')]}
    header['PG'][0]['CL'] = 'fiberhmm-call -i a.bam -o b.bam --enzyme ddda'
    # an older DddA call that recorded topology (radial recall ignored it)
    settings = CallerSettings.from_header(header)
    assert settings.nuc_recall_policy == 'conservative'
    assert any('radial' in n for n in settings.notes)
    with pytest.raises(ValueError, match='radial model'):
        CallerSettings.from_header(header, nuc_recall_policy='topology')
    hia5 = CallerSettings.from_header(header, enzyme='hia5', seq='nanopore',
                                      nuc_recall_policy='auto')
    assert hia5.nuc_recall_policy == 'conservative'


def test_saved_nanopore_evidence_accepts_a_recorded_topology_call():
    from fiberhmm.inference.consensus.bam import check_saved_evidence
    from fiberhmm.inference.consensus.parameters import parse_options

    def stratum(**manifest):
        return dict(strata=[dict(dataset_id='d', chemistry='hia5-nanopore', units=[],
                                 model_manifest=dict(preset='hia5-nanopore', emissions_sha256='0' * 64,
                                                     upstream_nuc_tf_recall=True, **manifest))])
    options = parse_options({})
    topo = dict(upstream_nuc_tf_settings=dict(nuc_recall_policy='topology'))
    check_saved_evidence(stratum(call_nuc_recall_policy='topology', **topo), options)
    with pytest.raises(ValueError, match='Reload the evidence'):
        check_saved_evidence(stratum(**topo), options)
    with pytest.raises(ValueError, match='Reload the evidence'):
        check_saved_evidence(stratum(call_nuc_recall_policy='topology',
                                     upstream_nuc_tf_settings=dict(nuc_recall_policy='conservative')),
                             options)


# --------------------------------------------------------------------------- #
#  consensus and scan replay                                                   #
# --------------------------------------------------------------------------- #

def _load(paths):
    from fiberhmm.inference.consensus import bam
    from fiberhmm.inference.consensus.parameters import parse_options

    opt = parse_options(dict(cr=dict(engine='lattice_recaller'),
                             recaller=dict(max_discovery_molecules=0, max_quantification_molecules=0)))
    ds = [dict(dataset_id='D', paths=[str(p) for p in paths], chemistry='hia5-nanopore')]
    payload = bam.load_bam_payload(ds, dict(chrom='chrDemo', start=9000, end=11000), opt)
    return payload, opt


def test_consensus_replay_uses_the_recorded_policy(ont):
    from fiberhmm.inference.consensus.bam import check_saved_evidence

    for tag, stamp in (('conservative', None), ('topology', 'topology')):
        payload, opt = _load([ont[tag]])
        manifest = payload['strata'][0]['model_manifest']
        assert manifest['upstream_nuc_tf_recall']
        assert manifest['upstream_nuc_tf_settings']['nuc_recall_policy'] == tag
        assert manifest.get('call_nuc_recall_policy') == stamp
        if stamp is None:
            assert 'call_nuc_recall_policy' not in manifest   # default manifests unchanged
        check_saved_evidence(payload, opt)


def test_consensus_refuses_a_dataset_with_mixed_policies(ont):
    with pytest.raises(ValueError, match='different nucleosome recall policies'):
        _load([ont['conservative'], ont['topology']])


def test_scan_state_replay_uses_the_recorded_policy(ont, monkeypatch):
    from fiberhmm.inference.consensus import upstream_recall
    from fiberhmm.scan.states import StateReader

    seen = []
    real = upstream_recall.recall_hia5_alignment

    def spy(*args, **kwargs):
        seen.append(kwargs.get('nuc_recall_policy'))
        return real(*args, **kwargs)

    monkeypatch.setattr(upstream_recall, 'recall_hia5_alignment', spy)
    for tag, want in (('default', 'conservative'), ('topology', 'topology')):
        seen.clear()
        with pysam.AlignmentFile(str(ont[tag])) as bam:
            reader = StateReader(bam.header)
            assert reader._chemistry_runtime()['replay']
            read = next(bam.fetch('chrDemo', 10_000, 10_001))
            assert reader.state_at(read, 'nfr.region', 'chrDemo', 10_050) in {'0', '1', '.'}
        assert seen and set(seen) == {want}
