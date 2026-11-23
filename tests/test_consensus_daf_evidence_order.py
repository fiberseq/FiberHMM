"""Saved DddB consensus evidence must record the 3.3 DAF evidence order.

FiberHMM 3.3.0 reads DAF evidence as fiberhmm-call does (R/Y, then MD, then
MM/ML); earlier loaders read MM/ML before MD and recorded nothing, so their
saved DddB evidence is refused (--evidence, --resume, transfer --evidence).
"""
import copy

import pysam
import pytest

from fiberhmm.inference.consensus.bam import (
    DAF_EVIDENCE_ORDER, check_saved_evidence, load_bam_payload,
)
from fiberhmm.inference.consensus.parameters import parse_options

REF = 'ACGT' * 50


def dddb_bam(tmp_path, enzyme='dddb'):
    from fiberhmm.io.bam_header import append_chemistry
    header = pysam.AlignmentHeader.from_dict(dict(
        HD={'VN': '1.6', 'SO': 'coordinate'}, SQ=[{'SN': 'chr1', 'LN': 1000}]))
    header = append_chemistry(header, dict(assay='daf', enzyme=enzyme, platform='pacbio',
                                           mode='daf'))
    marked = {i for i, b in enumerate(REF) if b == 'C' and i % 8 < 4}
    query = ''.join('T' if i in marked else b for i, b in enumerate(REF))
    pieces, matches = [], 0
    for a, b in zip(REF, query):
        if a == b:
            matches += 1
        else:
            pieces.extend([str(matches), a]); matches = 0
    md = ''.join(pieces) + str(matches)
    path = tmp_path / 'input.bam'
    with pysam.AlignmentFile(str(path), 'wb', header=header) as handle:
        for index in range(3):
            r = pysam.AlignedSegment(header)
            r.query_name = f'r{index}'; r.query_sequence = query
            r.reference_id = 0; r.reference_start = 100; r.mapping_quality = 60
            r.cigarstring = '200M'; r.set_tag('MD', md)
            r.set_tag('MA', '200;msp.:1-200')
            handle.write(r)
    pysam.index(str(path))
    return [dict(dataset_id='d', paths=[str(path)])]


def saved(chemistry, **manifest):
    return dict(strata=[dict(dataset_id='d', chemistry=chemistry, units=[],
                             model_manifest=dict(preset=chemistry, emissions_sha256='0' * 64,
                                                 **manifest))])


def test_new_dddb_evidence_records_the_order_and_is_accepted(tmp_path):
    options = parse_options({})
    payload = load_bam_payload(dddb_bam(tmp_path), dict(chrom='chr1', start=100, end=300),
                               options)
    manifest = payload['strata'][0]['model_manifest']
    assert payload['strata'][0]['chemistry'] == 'dddb'
    assert manifest['daf_evidence_order'] == DAF_EVIDENCE_ORDER == 'ry,md,mm'
    check_saved_evidence(payload, options)
    # The same evidence without the token (as written before 3.3.0) is refused.
    old = copy.deepcopy(payload)
    del old['strata'][0]['model_manifest']['daf_evidence_order']
    with pytest.raises(ValueError, match='recompute the evidence from the BAMs'):
        check_saved_evidence(old, options)


def test_old_dddb_evidence_is_refused_and_hia5_is_not():
    options = parse_options({})
    with pytest.raises(ValueError, match='saved DddB evidence was prepared before FiberHMM 3.3.0'):
        check_saved_evidence(saved('dddb'), options)
    with pytest.raises(ValueError, match='another order'):
        check_saved_evidence(saved('dddb', daf_evidence_order='mm,md'), options)
    check_saved_evidence(saved('dddb', daf_evidence_order=DAF_EVIDENCE_ORDER), options)
    # Hia5 evidence is unaffected (no DAF evidence order).
    check_saved_evidence(saved('hia5-pacbio'), options)
    check_saved_evidence(saved('hia5-nanopore'), options)
    # Hand-built payloads without the loader's receipt are taken as given.
    check_saved_evidence(dict(strata=[dict(dataset_id='d', chemistry='dddb', units=[])]),
                         options)
    # Pooled evidence: every window's preparation is checked.
    pooled = saved('dddb', daf_evidence_order=DAF_EVIDENCE_ORDER)
    pooled['pooling'] = dict(window_preparation=[dict(
        window='w2', dataset_id='d', model=dict(preset='dddb', emissions_sha256='0' * 64))])
    with pytest.raises(ValueError, match='DddB'):
        check_saved_evidence(pooled, options)


def test_hia5_manifest_has_no_evidence_order(tmp_path):
    payload = load_bam_payload(dddb_bam(tmp_path, enzyme='hia5'),
                               dict(chrom='chr1', start=100, end=300),
                               parse_options({}))
    assert [s['chemistry'] for s in payload['strata']] == ['hia5-pacbio']
    for stratum in payload['strata']:
        assert 'daf_evidence_order' not in stratum['model_manifest']


def test_consensus_cli_refuses_old_dddb_evidence_with_exit_2(tmp_path, capsys):
    from fiberhmm.inference.consensus.artifacts import write_json
    from fiberhmm.inference.consensus.cli import main
    path = tmp_path / 'evidence.json.gz'
    payload = saved('dddb')
    payload['region'] = dict(chrom='chr1', start=0, end=100)
    write_json(path, payload)
    with pytest.raises(SystemExit) as stop:
        main(['--evidence', str(path), '--output', str(tmp_path / 'out'), '--no-bam'])
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert 'saved DddB evidence was prepared before FiberHMM 3.3.0' in err
    assert 'recompute the evidence from the BAMs' in err
    assert len(err.strip().splitlines()) == 1
