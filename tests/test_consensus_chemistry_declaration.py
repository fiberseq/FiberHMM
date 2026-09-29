"""Consensus chemistry resolution honours the declared enzyme (release audit 2026-09-29, codex probe_integration.py)."""
from types import SimpleNamespace

import pysam
import pytest

from fiberhmm.cli.provenance import chemistry_declaration
from fiberhmm.io.bam_header import append_chemistry, chemistry_profile_from_declaration, resolve_bam_chemistry


def _bam(tmp_path, name, declaration=None, pg=None):
    header = pysam.AlignmentHeader.from_dict({'HD': {'SO': 'coordinate'}, 'SQ': [{'SN': 'chrT', 'LN': 3000}],
                                              **({'PG': [pg]} if pg else {})})
    if declaration:
        header = append_chemistry(header, declaration)
    path = tmp_path/f'{name}.bam'
    with pysam.AlignmentFile(str(path), 'wb', header=header):
        pass
    return path


def test_ecogii_declaration_is_never_resolved_to_hia5(tmp_path):
    decl = chemistry_declaration(SimpleNamespace(seq='pacbio', enzyme='ecogii'), 'pacbio-fiber', 'ecogii_pacbio.json', None)
    assert decl['enzyme'] == 'ecogii' and decl['assay'] == 'fiber-seq'
    assert chemistry_profile_from_declaration(decl) == 'ecogii-pacbio'
    bam = _bam(tmp_path, 'ecogii', decl)
    with pytest.raises(ValueError, match='consensus has no ecogii profile'):
        resolve_bam_chemistry([bam], None)
    with pytest.raises(ValueError, match='conflicts with BAM chemistry ecogii-pacbio'):
        resolve_bam_chemistry([bam], 'hia5-pacbio')


def test_custom_model_declaration_needs_an_explicit_chemistry(tmp_path):
    decl = chemistry_declaration(SimpleNamespace(seq='pacbio', enzyme=None), 'pacbio-fiber', 'my_model.json', None)
    assert decl['enzyme'] == 'custom' and chemistry_profile_from_declaration(decl) is None
    bam = _bam(tmp_path, 'custom', decl)
    with pytest.raises(ValueError, match='provide --chemistry'):
        resolve_bam_chemistry([bam], None)
    assert resolve_bam_chemistry([bam], 'hia5-pacbio')[0] == 'hia5-pacbio'


def test_supported_declarations_and_legacy_fiber_seq_without_an_enzyme(tmp_path):
    for enzyme, seq, mode, want in (('hia5', 'pacbio', 'pacbio-fiber', 'hia5-pacbio'), ('hia5', 'nanopore', 'nanopore-fiber', 'hia5-nanopore'),
                                    ('ddda', 'pacbio', 'daf', 'ddda'), ('dddb', 'nanopore', 'daf', 'dddb')):
        decl = chemistry_declaration(SimpleNamespace(seq=seq, enzyme=enzyme), mode, f'{enzyme}.json', None)
        assert resolve_bam_chemistry([_bam(tmp_path, enzyme + seq, decl)], None)[0] == want
    # A legacy @PG that never recorded an enzyme keeps the historical Hia5 assumption for Fiber-seq.
    legacy = _bam(tmp_path, 'legacy', pg=dict(ID='fiberhmm-call', PN='fiberhmm-call', DS='mode=pacbio-fiber k=3', CL='fiberhmm-call -i a.bam -o b.bam'))
    profile, records = resolve_bam_chemistry([legacy], None)
    assert profile == 'hia5-pacbio' and records[0]['source'] == 'legacy_pg_inference'
