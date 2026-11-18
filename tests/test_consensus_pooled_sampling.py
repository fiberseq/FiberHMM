"""Pooled (--pool-loci) molecule caps limit what is decoded, with the same cap selection as a full load."""
import gzip
from copy import deepcopy

import pytest

from fiberhmm.inference.consensus import bam, loader_sampling as S, pooled_sampling as P
from fiberhmm.inference.consensus.parameters import parse_options
from fiberhmm.inference.consensus.regions import pool_payloads
from fiberhmm.inference.consensus.workflow import run_workflow


def _window_payloads():
    from test_consensus_lattice_recaller import planted_payload
    windows = [dict(chrom='chrA', start=1050, end=1350, name='a', strand='+'),
               dict(chrom='chrB', start=1050, end=1350, name='b', strand='-')]
    payloads = []
    for k, w in enumerate(windows):
        p = planted_payload(240, occupancy=.6, seed=11 + k)
        p['region'] = dict(chrom=w['chrom'], start=w['start'], end=w['end'])
        for u in p['strata'][0]['units']:
            u['read_name'] = f"{w['name']}_{u['read_name']}"; u['unit_id'] = f"{w['name']}_{u['unit_id']}"
        payloads.append(p)
    return windows, payloads


@pytest.mark.parametrize('caps', [(40, 25), (60, 60)])
def test_pooled_caps_decode_fewer_names_and_match_full_load(tmp_path, monkeypatch, caps):
    windows, payloads = _window_payloads()
    rows = {w['chrom']: [(u['read_name'], u['reference_start'], u['reference_end']) for u in p['strata'][0]['units']]
            for w, p in zip(windows, payloads)}
    monkeypatch.setattr(S, 'scan', lambda paths, region, mapq: (len(rows[region['chrom']]), rows[region['chrom']]))
    monkeypatch.setattr(bam, '_dataset_chemistry', lambda item, paths: (item['chemistry'], {}))
    params = dict(compute=dict(cores=1), recaller=dict(max_discovery_molecules=caps[0],
                  max_quantification_molecules=caps[1], learned_spots=False))
    options = parse_options(dict(params, cr=dict(engine='lattice_recaller')))
    decoded = []
    def load_window(i, w, names):
        p = deepcopy(payloads[i]); decoded.append(len(names['planted']))
        p['strata'][0]['units'] = [u for u in p['strata'][0]['units'] if u['read_name'] in names['planted']]
        p['strata'][0]['load'] = {}
        return p
    datasets = [dict(dataset_id='planted', paths=['unused'], chemistry='ddda')]
    pooled = P.load_pooled(datasets, windows, options, lambda *a: None, load_window)
    assert pooled is not None and sum(decoded[:2]) < 480
    baseline = run_workflow(pool_payloads(deepcopy(payloads), windows), params, tmp_path/'full')
    result = run_workflow(pooled, params, tmp_path/'pooled')
    assert result['recaller']['classes'] == baseline['recaller']['classes']
    assert result['recaller']['rows'] == baseline['recaller']['rows']
    assert gzip.decompress((tmp_path/'pooled'/'molecules.tsv.gz').read_bytes()) == gzip.decompress(
        (tmp_path/'full'/'molecules.tsv.gz').read_bytes())
    assert result['manifest']['loader_sampling'][0]['strategy'] == 'pooled_hash_threshold'


def test_uncapped_pooled_runs_load_every_window_in_full():
    options = parse_options(dict(cr=dict(engine='lattice_recaller')))
    assert P.load_pooled([], [], options, lambda *a: None, None) is None
