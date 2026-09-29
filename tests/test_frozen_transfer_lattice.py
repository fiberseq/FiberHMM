"""Frozen-class transfer for the lattice recaller: freeze -> apply round trips, validation and the CLI."""
import copy
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fiberhmm.inference.consensus.artifacts import read_json, write_json
from fiberhmm.inference.consensus.lattice_recaller import frozen as F
from fiberhmm.inference.consensus.transfer import export_run, load_bundle
from fiberhmm.inference.consensus.workflow import run_workflow
from test_consensus_lattice_recaller import planted_payload

ROOT = Path(__file__).resolve().parents[1]
PARAMS = {'cr': {'engine': 'lattice_recaller'}, 'compute': {'cores': 1}}


def spotted(n=600, seed=7, occupancy=.4, rate=.3, prefix=''):
    """planted_payload plus an interior site (1196 on CT, 1194 on GA) marked in `rate` of bound molecules: a learned spot."""
    p = planted_payload(n=n, occupancy=occupancy, seed=seed); rng = np.random.default_rng(seed + 100)
    for u in p['strata'][0]['units']:
        if u['native_multi_interval_calls']:
            for i, x in enumerate(u['positions']):
                if x in (1194, 1196):
                    u['hits'][i] = int(rng.random() < rate)
        u['unit_id'] = prefix + u['unit_id']; u['read_name'] = prefix + u['read_name']
    return p


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('source')
    result = run_workflow(spotted(), PARAMS, tmp/'run')
    catalog = export_run(tmp/'run', tmp/'frozen_classes.json.gz')
    return tmp, result, catalog


def molecules(path):
    with gzip.open(path, 'rt') as fh:
        return fh.read()


def test_freeze_accepts_default_engine_run_on_oriented_pool(tmp_path):
    """Regression (release audit BLOCKER): freezing a default-engine run failed with FileNotFoundError / KeyError."""
    from fiberhmm.inference.consensus.regions import pool_payloads
    p = planted_payload()
    pooled = pool_payloads([p], [dict(p['region'], strand='+', name='locus1')])
    run_workflow(pooled, dict(PARAMS, recaller={'learned_spots': False}), tmp_path/'run')
    catalog = export_run(tmp_path/'run', tmp_path/'frozen.json.gz')
    assert catalog['schema'] == F.SCHEMA and catalog['frame']['pooled'] and catalog['classes']
    assert catalog['frame']['region']['start'] == 0 and catalog['training_molecule_hashes']


def test_catalog_carries_geometry_spots_and_provenance(source):
    tmp, result, catalog = source
    assert [c['id'] for c in catalog['classes']] == [c['id'] for c in result['recaller']['classes']]
    assert catalog['spot_precision'] == 'full'
    ct = catalog['channels']['planted::CT']
    assert ct['chemistry'] == 'ddda' and ct['strand'] == 'CT'
    spots = ct['classes']['class_001']['spots']
    assert list(spots) == [1196] and 0 < spots[1196] < .35                 # learned internal spot, full precision
    row = next(r for r in result['recaller']['rows'] if r['channel'] == 'planted::CT')
    assert spots[1196] == row['spot_rates']['1196']
    assert ct['classes']['class_001']['L'] == [row['L0'], row['L1']]
    prov = catalog['provenance']
    assert prov['input_digest'] == result['manifest']['input_digest'] and prov['freeze_code']['fiberhmm_version']
    assert catalog['tiles'] == result['manifest']['recaller']['tiles']


def test_self_application_reproduces_the_run(source, tmp_path):
    tmp, result, catalog = source
    evidence = read_json(tmp/'run/evidence.json.gz')
    applied, _ = F.apply_catalog(catalog, evidence, tmp_path/'self', include_training=True, cores=1)
    assert applied['cr_mode'] == 'lattice_recaller' and applied['schema'] == 'fiberhmm.consensus.v1'
    assert applied['transfer']['schema'] == F.TRANSFER_SCHEMA and applied['transfer']['catalog_sha256'] == catalog['content_sha256']
    assert applied['transfer']['channel_map'] == {'planted::CT': 'planted::CT', 'planted::GA': 'planted::GA'}
    assert applied['recaller']['rows'] == result['recaller']['rows']          # every prevalence tier, gain, flag and spot
    assert molecules(tmp_path/'self/molecules.tsv.gz') == molecules(tmp/'run/molecules.tsv.gz')
    assert molecules(tmp_path/'self/broader.tsv.gz') == molecules(tmp/'run/broader.tsv.gz')
    same = lambda r: {d: v['cr']['records'] for d, v in r['datasets'].items()}
    assert same(applied) == same(result)
    for name in ('classes.tsv', 'manifest.json', 'result.json.gz', 'evidence.json.gz', 'report.html'):
        assert (tmp_path/'self'/name).exists(), name
    # By default the training molecules are excluded (as for staged families): nothing is left to score.
    empty, _ = F.apply_catalog(catalog, evidence, tmp_path/'excluded', cores=1)
    assert empty['recaller']['rows'] == [] and empty['transfer']['excluded_training_molecules'] == 600
    assert len(read_json(tmp_path/'excluded/transfer_exclusions.json')) == 600


def test_held_out_sample_recovers_planted_prevalence(source, tmp_path):
    tmp, result, catalog = source
    target = spotted(n=600, seed=11, occupancy=.25, prefix='h')
    target['strata'][0]['dataset_id'] = 'target'
    applied, _ = F.apply_catalog(catalog, target, tmp_path/'heldout', cores=1)
    # A new dataset of the same chemistry takes the frozen channel boxes and spots by chemistry + strand.
    assert applied['transfer']['channel_map'] == {'target::CT': 'planted::CT', 'target::GA': 'planted::GA'}
    assert applied['transfer']['excluded_training_molecules'] == 0
    rows = applied['recaller']['rows']
    assert {r['channel'] for r in rows} == {'target::CT', 'target::GA'}
    assert all(.18 <= r['prevalence'] <= .32 and r['supported'] for r in rows)
    assert all(r['spots'] == next(s['spots'] for s in result['recaller']['rows'] if s['strand'] == r['strand']) for r in rows)
    assert all(r['prevalence'] <= r['prevalence_edge'] <= r['prevalence_loose'] for r in rows)


def test_catalog_validation_and_fail_fast(source, tmp_path):
    tmp, result, catalog = source
    raw = read_json(tmp/'frozen_classes.json.gz')
    wrong = dict(raw, schema='fiberhmm.frozen_classes.lattice_recaller.v0')
    with pytest.raises(ValueError, match='Unsupported frozen-class catalog schema'):
        F.load_catalog(wrong)
    tampered = copy.deepcopy(raw); tampered['classes'][0]['L'][0] -= 1
    with pytest.raises(ValueError, match='digest'):
        F.load_catalog(tampered)
    write_json(tmp_path/'catalog.json', raw)
    with pytest.raises(ValueError, match='lattice-recaller frozen-class catalog'):
        load_bundle(tmp_path/'catalog.json')
    # Evidence in another frame, a fixed parameter group, and ambiguous channel mapping all fail before scoring.
    shifted = spotted(n=40, prefix='s'); shifted['region'] = dict(shifted['region'], start=1000)
    with pytest.raises(ValueError, match='differs from the frozen frame'):
        F.apply_catalog(catalog, shifted, tmp_path/'a')
    with pytest.raises(ValueError, match="'recaller' is fixed"):
        F.apply_catalog(catalog, spotted(n=40, prefix='s'), tmp_path/'b', parameters={'recaller': {'bf_threshold': 10.}})
    two = copy.deepcopy(catalog)
    two['channels']['other::CT'] = dict(copy.deepcopy(two['channels']['planted::CT']), dataset='other')
    target = spotted(n=40, prefix='t'); target['strata'][0]['dataset_id'] = 'target'
    with pytest.raises(ValueError, match='--dataset-map target=SOURCE_DATASET'):
        F.apply_catalog(two, target, tmp_path/'c')
    applied, _ = F.apply_catalog(two, target, tmp_path/'d', dataset_map={'target': 'other'}, cores=1)
    assert applied['transfer']['channel_map']['target::CT'] == 'other::CT'


def test_unsupported_runs_fail_fast(tmp_path):
    (tmp_path/'run').mkdir()
    write_json(tmp_path/'run/manifest.json', dict(cr_mode='call_harmonization', status='complete'))
    with pytest.raises(ValueError, match='cannot freeze a call_harmonization run'):
        export_run(tmp_path/'run', tmp_path/'x.json.gz')
    with pytest.raises(ValueError, match='not a consensus result directory'):
        export_run(tmp_path/'missing', tmp_path/'x.json.gz')
    run_workflow(planted_payload(n=40), dict(PARAMS, recaller={'minimum_core_bp': 1000}), tmp_path/'empty')
    with pytest.raises(ValueError, match='no discovered classes'):
        export_run(tmp_path/'empty', tmp_path/'x.json.gz')


def _cli(*args):
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''),
               OPENBLAS_NUM_THREADS='1', FIBERHMM_NO_UPDATE_CHECK='1')
    return subprocess.run([sys.executable, '-m', *args], capture_output=True, text=True, env=env, cwd=ROOT)


def test_cli_freeze_and_apply_end_to_end(source, tmp_path):
    tmp, result, catalog = source
    done = _cli('fiberhmm.inference.consensus.transfer_cli', '--freeze-run', str(tmp/'run'), '--output', str(tmp_path/'frozen'))
    assert done.returncode == 0, done.stderr
    frozen = F.load_catalog(tmp_path/'frozen/frozen_classes.json.gz')
    assert frozen['classes'] == catalog['classes'] and frozen['channels'] == catalog['channels']
    done = _cli('fiberhmm.inference.consensus.transfer_cli', '--models', str(tmp_path/'frozen'), '--evidence', str(tmp/'run/evidence.json.gz'),
                '--include-training-molecules', '--no-bam', '--cores', '1', '--output', str(tmp_path/'out'))
    assert done.returncode == 0, done.stderr
    out = tmp_path/'out'
    for name in ('classes.tsv', 'molecules.tsv.gz', 'broader.tsv.gz', 'result.json.gz', 'manifest.json', 'transfer_summary.tsv',
                 'transfer_manifest.json', 'report.html'):
        assert (out/name).exists(), name
    applied = read_json(out/'result.json.gz')
    assert applied['cr_mode'] == 'lattice_recaller' and applied['transfer']['catalog_sha256'] == frozen['content_sha256']
    assert applied['recaller']['rows'] == result['recaller']['rows']
    assert read_json(out/'manifest.json')['transfer']['spot_precision'] == 'full'
    manifest = read_json(out/'transfer_manifest.json')
    assert manifest['schema'] == 'fiberhmm.transfer_run.lattice_recaller.v1' and not manifest['refitted'] and not manifest['rediscovered']
    # Staged-only controls are rejected for recaller catalogs and vice versa.
    bad = _cli('fiberhmm.inference.consensus.transfer_cli', '--models', str(tmp_path/'frozen'), '--evidence', str(tmp/'run/evidence.json.gz'),
               '--bed', 'x.bed', '--output', str(tmp_path/'bad'))
    assert bad.returncode != 0 and 'omit --bed' in bad.stderr


def test_bam_windows_map_into_a_genomic_frame_and_export(source, tmp_path):
    """BAM + oriented BED6 into a catalog whose frame starts at 1050 (a non-pooled run): both orientations load,
    are scored in the frozen frame, and the family BAM export maps them back to the reads."""
    import pysam
    from test_consensus_bed_cli import make_bam
    tmp, result, catalog = source
    bam = tmp_path/'reads.bam'; make_bam(bam)
    bed = tmp_path/'windows.bed'; bed.write_text('chr1\t0\t300\tplus\t0\t+\nchr1\t0\t300\tminus\t0\t-\n')
    chip = tmp_path/'chip.bed'; chip.write_text('chr1\t150\t170\tpeak\n')
    from fiberhmm.inference.consensus.transfer_cli import main
    main(['--models', str(tmp/'frozen_classes.json.gz'), '--bam', str(bam), '--bed', str(bed), '--chip-bed', str(chip), '--cores', '1',
          '--output', str(tmp_path/'out')])
    out = tmp_path/'out'
    for w in ('window_000001', 'window_000002'):
        evidence = read_json(out/w/'evidence.json.gz')
        assert evidence['region']['start'] == 1050 and evidence['region']['end'] == 1350
        unit = evidence['strata'][0]['units'][0]
        assert all(1050 <= x < 1350 for x in unit['positions']) and unit['genomic_provenance']['coordinate_origin'] == 1050
        assert read_json(out/w/'result.json.gz')['transfer']['window']['name'] in ('plus', 'minus')
    assert (out/'chip_evaluation.json').exists() and (out/'report.html').exists()
    exported = read_json(out/'bams/bam_exports.json')['outputs']
    with pysam.AlignmentFile(exported[0]['bam'], 'rb') as handle:
        assert [r.query_name for r in handle] == ['molecule1']
