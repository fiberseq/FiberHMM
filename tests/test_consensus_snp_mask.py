"""SNP exclusions survive consensus preparation, replay, transfer and restart."""
from hashlib import sha256

import numpy as np
import pysam
import pytest

from fiberhmm.inference.consensus.artifacts import read_json
from fiberhmm.inference.consensus.bam import load_bam_payload
from fiberhmm.inference.consensus.parameters import parse_options
from fiberhmm.inference.engine import daf_snp_mask_scope
from fiberhmm.inference.strand_rescue import hard_observations


def read_with_variant(strand, encoding, reverse=False):
    header = pysam.AlignmentHeader.from_dict({'SQ': [{'SN': 'chr1', 'LN': 1000}]})
    read = pysam.AlignedSegment(header)
    ref = 'ACGT' * 50
    target, alternate, code = ('C', 'T', 'Y') if strand == 'CT' else ('G', 'A', 'R')
    marked = {i for i, b in enumerate(ref) if b == target and i % 8 < 4}
    query = ''.join((code if encoding == 'ry' else alternate) if i in marked else b for i, b in enumerate(ref))
    read.query_sequence = query
    read.query_name = 'variant'
    read.reference_id = 0
    read.reference_start = 100
    read.cigarstring = '200M'
    read.mapping_quality = 60
    read.flag = 16 if reverse else 0
    if encoding == 'ry':
        read.set_tag('st', strand)
    else:
        pieces, matches = [], 0
        for a, b in zip(ref, query):
            if a == b:
                matches += 1
            else:
                pieces.extend([str(matches), a]); matches = 0
        read.set_tag('MD', ''.join(pieces) + str(matches))
    offset = 1 if strand == 'CT' else 2
    return read, 64 + offset, 68 + offset  # one hit and one unconverted target


@pytest.mark.parametrize('strand', ['CT', 'GA'])
@pytest.mark.parametrize('encoding', ['ry', 'md'])
@pytest.mark.parametrize('reverse', [False, True])
def test_masked_alleles_are_unknown_in_both_orientations(strand, encoding, reverse):
    read, hit, miss = read_with_variant(strand, encoding, reverse)
    before = read.to_string()
    with daf_snp_mask_scope({}):
        plain, physical = hard_observations(read, 'daf', 'daf', 3, None)
    assert plain[hit] < 4096 and 4097 <= plain[miss] < 8193
    with daf_snp_mask_scope({'chr1': frozenset({100 + hit, 100 + miss})}):
        masked, observed = hard_observations(read, 'daf', 'daf', 3, None)
    assert masked[hit] == masked[miss] == 8193
    assert observed == physical == strand
    assert read.to_string() == before
    restored, _ = hard_observations(read, 'daf', 'daf', 3, None)
    np.testing.assert_array_equal(restored, plain)


def test_duplex_masks_both_opportunity_types(tmp_path):
    from test_consensus_duplex import joint_bam
    read = joint_bam(tmp_path / 'duplex.bam')
    with daf_snp_mask_scope({'chr1': frozenset(range(160, 164))}):
        obs, strand = hard_observations(read, 'daf', 'daf', 3, None)
    assert strand == 'BOTH'
    assert list(obs[60:64]) == [8193] * 4


def test_mm_deamination_and_unconverted_sites_are_both_masked():
    import array
    read, hit, miss = read_with_variant('CT', 'md')
    read.set_tag('MD', None)
    # Encode converted T sites with the SAM dU code, counting only query T's.
    targets = [i for i, b in enumerate(read.query_sequence) if b == 'T']
    marked = {i for i in range(200) if i % 8 == 1}
    skips, skipped = [], 0
    for q in targets:
        if q in marked:
            skips.append(str(skipped)); skipped = 0
        else:
            skipped += 1
    read.set_tag('MM', 'T+u.,' + ','.join(skips) + ';')
    read.set_tag('ML', array.array('B', [255] * len(skips)))
    plain, _ = hard_observations(read, 'daf', 'daf', 3, None)
    assert plain[hit] < 4096 and 4097 <= plain[miss] < 8193
    with daf_snp_mask_scope({'chr1': frozenset({100 + hit, 100 + miss})}):
        masked, _ = hard_observations(read, 'daf', 'daf', 3, None)
    assert masked[hit] == masked[miss] == 8193


def test_mask_coordinates_follow_cigar_not_query_offset():
    read, hit, miss = read_with_variant('CT', 'ry', reverse=True)
    read.cigarstring = '10S90M2D100M'
    ref_sites = {100 + hit - 10, 100 + miss - 10}
    with daf_snp_mask_scope({'chr1': frozenset(ref_sites)}):
        masked, _ = hard_observations(read, 'daf', 'daf', 3, None)
    assert masked[hit] == masked[miss] == 8193


def test_loader_and_native_replay_share_mask_without_mutating_bam(tmp_path, monkeypatch):
    from test_consensus_bed_cli import make_bam
    from fiberhmm.inference import tf_recaller
    bam = tmp_path / 'input.bam'; make_bam(bam)
    # Explicit strand-qualified MA scaffold so replay actually calls the decoder.
    with pysam.AlignmentFile(bam, 'rb') as handle:
        header = handle.header
        records = list(handle)
    records[0].set_tag('MA', '200;msp.:1-200;tf.:61-20')
    with pysam.AlignmentFile(bam, 'wb', header=header) as handle:
        for record in records: handle.write(record)
    pysam.index(str(bam))
    mask = tmp_path / 'snps.bed'; mask.write_text('chr1\t161\t162\n')
    original = sha256(bam.read_bytes()).hexdigest()
    seen = []
    real = tf_recaller.call_tfs_in_interval
    def capture(obs, *args, **kwargs):
        seen.append(np.asarray(obs).copy())
        return real(obs, *args, **kwargs)
    monkeypatch.setattr(tf_recaller, 'call_tfs_in_interval', capture)
    datasets = [dict(dataset_id='d', paths=[str(bam)])]
    region = dict(chrom='chr1', start=120, end=220)
    result = load_bam_payload(datasets, region, parse_options({'input': {'daf_snp_mask': str(mask)}}))
    source, = result['strata']; unit, = source['units']
    assert 161 not in unit['positions'] and 165 in unit['positions']
    assert seen and all(obs[61] == 8193 for obs in seen)
    assert source['model_manifest']['daf_snp_mask']['sha256'] == sha256(mask.read_bytes()).hexdigest()
    assert source['model_manifest']['daf_snp_mask']['sites'] == 1
    assert sha256(bam.read_bytes()).hexdigest() == original
    plain = load_bam_payload(datasets, region, parse_options({}))
    assert 161 in plain['strata'][0]['units'][0]['positions']


def test_invalid_mask_is_rejected(tmp_path):
    from fiberhmm.inference.consensus.snp_mask import read_mask
    mask = tmp_path / 'bad.bed'
    for line in ('chr1\t-1\t2', 'chr1\t5\t5', 'chr1\t5', 'chr1\tx\t6'):
        mask.write_text(line)
        with pytest.raises(ValueError, match='invalid SNP mask BED'):
            read_mask(mask)


def test_mask_content_is_part_of_restart_contract(tmp_path):
    from fiberhmm.inference.consensus.batch import run_record
    mask = tmp_path / 'snps.bed'; mask.write_text('chr1\t161\t162\n')
    values = {'input': {'daf_snp_mask': str(mask)}}
    first = run_record([], [], values, {})
    mask.write_text('chr1\t165\t166\n')
    second = run_record([], [], values, {})
    assert first['options']['daf_snp_mask']['file']['sha256'] != second['options']['daf_snp_mask']['file']['sha256']


def test_saved_evidence_cannot_be_silently_remasked(tmp_path):
    from fiberhmm.inference.consensus.cli import main
    with pytest.raises(SystemExit) as error:
        main(['--evidence', str(tmp_path / 'evidence.json.gz'), '--daf-snp-mask', 'missing.bed',
              '--output', str(tmp_path / 'out')])
    assert error.value.code == 2


def test_frozen_source_mask_is_not_inherited_by_targets():
    from dataclasses import asdict
    from fiberhmm.inference.consensus.lattice_recaller.frozen import transfer_options
    params = {k: asdict(v) for k, v in parse_options({'input': {'daf_snp_mask': '/source/snps.bed'}}).items()}
    catalog = {'parameters': params}
    assert transfer_options(catalog, [], cores=1)['input'].daf_snp_mask == ''
    overrides = {'input': {'daf_snp_mask': '/target/snps.bed'}}
    assert transfer_options(catalog, [], overrides, 1)['input'].daf_snp_mask == '/target/snps.bed'


def test_consensus_freeze_transfer_with_explicit_mask(tmp_path):
    from test_consensus_continue import make_windows_bam
    from fiberhmm.inference.consensus.cli import main as consensus
    from fiberhmm.inference.consensus.transfer_cli import main as transfer
    bam = make_windows_bam(tmp_path / 'reads.bam', n_windows=1)
    mask = tmp_path / 'snps.bed'; mask.write_text('chr1\t1300\t1304\n')
    fit, frozen, target = (tmp_path / n for n in ('fit', 'frozen', 'target'))
    consensus(['--bam', str(bam), '--region', 'chr1:1100-1500', '--daf-snp-mask', str(mask),
               '--cores', '1', '--no-bam', '--output', str(fit)])
    transfer(['--freeze-run', str(fit), '--output', str(frozen)])
    bed = tmp_path / 'windows.bed'; bed.write_text('chr1\t1100\t1500\twindow\t0\t-\n')
    transfer(['--models', str(frozen), '--bam', str(bam), '--bed', str(bed), '--daf-snp-mask', str(mask),
              '--include-training-molecules', '--cores', '1', '--no-bam', '--output', str(target)])
    evidence = read_json(target / 'evidence.json.gz')
    source, = evidence['strata']
    assert source['units'] and all(u['strand'] == 'GA' for u in source['units'])
    # Minus-window orientation into the original frame: 1100 + (1499 - pos).
    assert all(not set(range(1296, 1300)).intersection(u['positions']) for u in source['units'])
    assert source['model_manifest']['daf_snp_mask']['sha256'] == sha256(mask.read_bytes()).hexdigest()
    assert read_json(target / 'transfer_manifest.json')['preparation_parameters']['input']['daf_snp_mask'] == str(mask)
    # The evidence is self-contained: replay does not reopen the original BED.
    mask.unlink()
    replay = tmp_path / 'replay'
    consensus(['--resume', str(fit), '--cores', '1', '--no-bam', '--output', str(replay)])
    assert read_json(replay / 'evidence.json.gz') == read_json(fit / 'evidence.json.gz')


def test_parallel_windows_keep_mask_and_continue_rejects_changed_content(tmp_path, capsys):
    import os
    from test_consensus_continue import make_windows_bam
    from fiberhmm.inference.consensus.cli import main
    bam = make_windows_bam(tmp_path / 'reads.bam', n_windows=2)
    mask = tmp_path / 'mask.bed'
    mask.write_text('chr1\t1300\t1304\nchr1\t3300\t3304\n')
    output = tmp_path / 'parallel'
    args = ['--bam', str(bam), '--region', 'chr1:1100-1500', '--region', 'chr1:3100-3500',
            '--daf-snp-mask', str(mask), '--cores', '2', '--window-jobs', '2', '--no-bam', '--output', str(output)]
    main(args)
    for i, base in enumerate((1300, 3300), 1):
        evidence = read_json(output / f'window_{i:06d}' / 'evidence.json.gz')
        source, = evidence['strata']
        assert source['units'] and all(not set(range(base, base + 4)).intersection(u['positions']) for u in source['units'])
        assert source['model_manifest']['daf_snp_mask']['sha256'] == sha256(mask.read_bytes()).hexdigest()
    stat = mask.stat()
    mask.write_text('chr1\t1310\t1314\nchr1\t3310\t3314\n')
    os.utime(mask, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(SystemExit) as error:
        main(args + ['--continue'])
    assert error.value.code == 2
    assert 'options.daf_snp_mask.file.sha256' in capsys.readouterr().err
