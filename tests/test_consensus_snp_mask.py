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


@pytest.mark.parametrize('strand,sequence,neighbor', [('CT', 'AYCGT', 62), ('GA', 'ACGRT', 62)])
@pytest.mark.parametrize('context_size', [2, 3])
@pytest.mark.parametrize('run_min,policy', [(0, 'keep-one'), (2, 'keep-one'), (2, 'drop')])
def test_duplex_mask_precedes_context_restoration_and_run_thinning(strand, sequence, neighbor, context_size, run_min, policy):
    from fiberhmm.core.bam_reader import ContextEncoder, daf_run_mask_scope
    read, _, _ = read_with_variant(strand, 'ry')
    read.query_sequence = sequence * 40
    read.set_tag('st', strand)
    q = 61 if strand == 'CT' else 63
    with daf_snp_mask_scope({'chr1': {100 + q}}), daf_run_mask_scope(run_min, policy):
        single, _ = hard_observations(read, 'daf', 'daf', context_size, None)
        read.set_tag('cs', 'ct;ga')
        read.set_tag('MA', '200;deam+:1-200;deam-:1-200;msp.:1-200')
        joint, physical = hard_observations(read, 'daf', 'daf', context_size, None)
    unknown = 2 * ContextEncoder.get_n_codes(context_size) + 1
    assert physical == 'BOTH'
    assert single[q] == joint[q] == unknown
    assert single[neighbor] != unknown
    assert joint[neighbor] == single[neighbor]


def test_reference_mask_invalidates_cached_md_strand_result():
    from fiberhmm.daf.encoder import get_daf_positions
    from fiberhmm.inference.tf_recaller import extract_modification_calls
    read, _, _ = read_with_variant('CT', 'md')
    class CachedRead:
        _daf_md_result = get_daf_positions(read)
        def __getattr__(self, name):
            return getattr(read, name)
    cached = CachedRead()
    assert cached._daf_md_result[2] == 'CT'
    # Removing every conversion makes the MD evidence unassignable. A stale
    # cache must not retain the old CT strand and turn the read into all misses.
    excluded = {100 + q for q in cached._daf_md_result[0]}
    assert get_daf_positions(read, excluded_reference_positions=excluded) is None
    assert extract_modification_calls(cached, 'daf', 3, excluded_reference_positions=excluded) is None


@pytest.mark.parametrize('strand,unconverted,converted,q', [('CT', 'ACCGT', 'AYCGT', 61),
                                                         ('GA', 'ACGGT', 'ACGRT', 63)])
@pytest.mark.parametrize('duplex', [False, True])
def test_point_mask_retains_neighbor_context_limitation_and_expanded_mask_removes_it(strand, unconverted, converted, q, duplex):
    from fiberhmm.core.bam_reader import daf_run_mask_scope
    def observations(pattern, padding, joint):
        read, _, _ = read_with_variant(strand, 'ry')
        # Keep independent R/Y evidence elsewhere so the unconverted version
        # still has an encoding and a known chemical strand.
        sequence = list(unconverted * 40)
        if pattern == converted:
            sequence[q] = 'Y' if strand == 'CT' else 'R'
        sequence[11 if strand == 'CT' else 13] = 'Y' if strand == 'CT' else 'R'
        read.query_sequence = ''.join(sequence)
        if joint:
            read.set_tag('cs', 'ct;ga')
            read.set_tag('MA', '200;deam+:1-200;deam-:1-200;msp.:1-200')
        mask = {'chr1': set(range(100 + q - padding, 101 + q + padding))}
        with daf_snp_mask_scope(mask), daf_run_mask_scope(2, 'keep-one'):
            return hard_observations(read, 'daf', 'daf', 3, None)[0]
    # Point masks remove the position, not its neighboring sequence effects.
    # Native unconverted CC/GG can still thin the neighboring target; removing
    # a conversion leaves T/A there instead. Single and duplex paths agree.
    point_ref = observations(unconverted, 0, duplex)
    point_alt = observations(converted, 0, duplex)
    assert point_ref[q] == point_alt[q] == 8193
    assert point_ref[62] == 8193 and point_alt[62] != 8193
    for pattern in (unconverted, converted):
        assert observations(pattern, 0, False)[62] == observations(pattern, 0, True)[62]
    expanded_ref = observations(unconverted, 3, duplex)
    expanded_alt = observations(converted, 3, duplex)
    # This short-run example becomes invariant after masking the affected
    # neighborhood; longer runs can require a wider sensitivity mask.
    np.testing.assert_array_equal(expanded_ref, expanded_alt)
    assert all(expanded_ref[q-3:q+4] == 8193)


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
    # Reference 1279-1283 is CTAC in the fixture: two isolated, non-CpG C targets.
    mask = tmp_path / 'snps.bed'; mask.write_text('chr1\t1279\t1283\n')
    fit, frozen, target, control = (tmp_path / n for n in ('fit', 'frozen', 'target', 'control'))
    consensus(['--bam', str(bam), '--region', 'chr1:1100-1500', '--daf-snp-mask', str(mask),
               '--cores', '1', '--no-bam', '--output', str(fit)])
    transfer(['--freeze-run', str(fit), '--output', str(frozen)])
    bed = tmp_path / 'windows.bed'; bed.write_text('chr1\t1100\t1500\twindow\t0\t-\n')
    common = ['--models', str(frozen), '--bam', str(bam), '--bed', str(bed),
              '--include-training-molecules', '--cores', '1', '--no-bam']
    transfer(common + ['--daf-snp-mask', str(mask), '--output', str(target)])
    transfer(common + ['--output', str(control)])
    # Minus-window orientation into the original frame: 1100 + (1499 - pos).
    masked_sites = {1100 + (1499 - p) for p in (1279, 1282)}
    unmasked, = read_json(control / 'evidence.json.gz')['strata']
    assert masked_sites <= {p for u in unmasked['units'] for p in u['positions']}
    evidence = read_json(target / 'evidence.json.gz')
    source, = evidence['strata']
    assert source['units'] and all(u['strand'] == 'GA' for u in source['units'])
    assert all(not masked_sites.intersection(u['positions']) for u in source['units'])
    assert source['model_manifest']['daf_snp_mask']['sha256'] == sha256(mask.read_bytes()).hexdigest()
    assert 'daf_snp_mask' not in unmasked['model_manifest']
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
    # Both intervals hold C targets (reference CTAC and GCTG in the fixture).
    mask.write_text('chr1\t1279\t1283\nchr1\t3300\t3304\n')
    output = tmp_path / 'parallel'
    args = ['--bam', str(bam), '--region', 'chr1:1100-1500', '--region', 'chr1:3100-3500',
            '--daf-snp-mask', str(mask), '--cores', '2', '--window-jobs', '2', '--no-bam', '--output', str(output)]
    main(args)
    for i, base in enumerate((1279, 3300), 1):
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


def single_mark_read(encoding, marked=True):
    """CT read whose only conversion is at query 61 (reference 161): R/Y without
    an st tag, or MM-tagged; ``marked=False`` gives the same read with no mark."""
    import array
    read, _, _ = read_with_variant('CT', 'md')
    read.set_tag('MD', None)
    ref = 'ACGT' * 50
    q = 61
    if encoding == 'ry':
        read.query_sequence = ref[:q] + ('Y' if marked else 'C') + ref[q + 1:]
    else:
        read.query_sequence = ref[:q] + ('T' if marked else 'C') + ref[q + 1:]
        if marked:
            before = read.query_sequence[:q].count('T')
            read.set_tag('MM', f'T+u.,{before};')
            read.set_tag('ML', array.array('B', [255]))
    return read, q


@pytest.mark.parametrize('encoding', ['ry', 'mm'])
def test_read_whose_only_strand_mark_is_masked_is_skipped_not_called_ga(encoding):
    read, q = single_mark_read(encoding)
    assert not read.has_tag('st')
    plain, strand = hard_observations(read, 'daf', 'daf', 3, None)
    assert strand == 'CT' and plain[q] < 4096
    # A mask that leaves the strand mark keeps the read on its strand.
    with daf_snp_mask_scope({'chr1': frozenset({100 + q + 4})}):
        kept, kept_strand = hard_observations(read, 'daf', 'daf', 3, None)
    assert kept_strand == 'CT' and kept[q + 4] == 8193
    # Masking the only mark leaves no strand evidence: unusable, not GA.
    with daf_snp_mask_scope({'chr1': frozenset({100 + q})}):
        assert hard_observations(read, 'daf', 'daf', 3, None) is None


@pytest.mark.parametrize('encoding', ['ry', 'mm'])
def test_read_undecidable_before_masking_keeps_ga_fallback_under_a_mask(encoding):
    # Strand already undecidable without the mask (R/Y tie without st, or an
    # MM call below the probability threshold): the "." -> GA fallback stays.
    import array
    read, q = single_mark_read(encoding, marked=encoding == 'ry')
    if encoding == 'ry':
        sequence = list(read.query_sequence); sequence[q + 5] = 'R'
        read.query_sequence = ''.join(sequence)
    else:
        read.set_tag('MM', 'T+u.,0;'); read.set_tag('ML', array.array('B', [1]))
    _, strand = hard_observations(read, 'daf', 'daf', 3, None)
    with daf_snp_mask_scope({'chr1': frozenset({100 + q + 8})}):
        masked, masked_strand = hard_observations(read, 'daf', 'daf', 3, None)
    assert strand == masked_strand == 'GA'
    assert masked[q + 8] == 8193


def test_loader_skips_read_whose_strand_the_mask_removes(tmp_path):
    from fiberhmm.io.bam_header import append_chemistry
    header = pysam.AlignmentHeader.from_dict(dict(HD={'VN': '1.6', 'SO': 'coordinate'}, SQ=[{'SN': 'chr1', 'LN': 1000}]))
    header = append_chemistry(header, dict(assay='daf', enzyme='ddda', platform='pacbio', mode='daf'))
    alt, q = single_mark_read('ry')
    other, _, _ = read_with_variant('CT', 'ry')
    bam = tmp_path / 'input.bam'
    with pysam.AlignmentFile(str(bam), 'wb', header=header) as handle:
        for name, source in (('only_masked_mark', alt), ('declared_strand', other)):
            r = pysam.AlignedSegment(header); r.query_name = name; r.query_sequence = source.query_sequence
            r.reference_id = 0; r.reference_start = 100; r.mapping_quality = 60; r.cigarstring = '200M'
            if source.has_tag('st'):
                r.set_tag('st', source.get_tag('st'))
            r.set_tag('MA', '200;msp.:1-200')
            handle.write(r)
    pysam.index(str(bam))
    mask = tmp_path / 'snps.bed'; mask.write_text(f'chr1\t{100 + q}\t{101 + q}\n')
    datasets = [dict(dataset_id='d', paths=[str(bam)])]
    region = dict(chrom='chr1', start=120, end=220)
    strands = lambda payload: {u['read_name']: u['strand'] for u in payload['strata'][0]['units']}
    assert strands(load_bam_payload(datasets, region, parse_options({}))) == {
        'only_masked_mark': 'CT', 'declared_strand': 'CT'}
    assert strands(load_bam_payload(datasets, region, parse_options({'input': {'daf_snp_mask': str(mask)}}))) == {
        'declared_strand': 'CT'}


def test_replaying_masked_evidence_requires_native_replay():
    from test_staged_families import payload, parameters
    from fiberhmm.inference.consensus.harmonized_families.workflow import prepare_sources
    data = payload()
    data['strata'][0]['model_manifest'] = dict(daf_snp_mask=dict(path='/x/snps.bed', sha256='0' * 64, sites=1))
    options = parameters()
    assert options['input']['correct_native'] is False
    with pytest.raises(ValueError, match='DAF SNP mask.*input.correct_native=true'):
        prepare_sources(data, parse_options(options))
    # Unmasked saved evidence still replays with the BAM's own calls.
    assert prepare_sources(payload(), parse_options(options))


@pytest.mark.parametrize('duplex', [False, True])
def test_mask_positions_skip_cigar_padding(duplex):
    # P consumes neither query nor reference; pysam's aligned pairs shift the
    # query index after it, which would mask the neighbouring base instead.
    read, hit, miss = read_with_variant('CT', 'ry')
    read.cigarstring = '60M1P140M'
    if duplex:
        read.set_tag('cs', 'ct;ga')
        read.set_tag('MA', '200;deam+:1-200;deam-:1-200;msp.:1-200')
    with daf_snp_mask_scope({'chr1': frozenset({100 + hit, 100 + miss, 100 + 199})}):
        masked, _ = hard_observations(read, 'daf', 'daf', 3, None)
    with daf_snp_mask_scope({}):
        plain, _ = hard_observations(read, 'daf', 'daf', 3, None)
    unknown = lambda obs: {int(q) for q in np.flatnonzero(np.asarray(obs) == 8193)}
    # Exactly the masked bases become unknown (neighbour contexts may change).
    assert {hit, miss} <= unknown(masked) - unknown(plain) <= {hit, miss, 199}


def test_masked_query_positions_follow_every_cigar_operation():
    from fiberhmm.inference.consensus.snp_mask import masked_query_positions
    read, _, _ = read_with_variant('CT', 'ry')
    read.cigarstring = '5H3S10M2I4M1P3D5N6M5H'
    # Query: 3 S, 10 M (ref 100-109), 2 I, 4 M (ref 110-113), 3 D + 5 N (ref
    # 114-121), 6 M (ref 122-127, query 19-24).
    read.query_sequence = 'A' * 25
    assert masked_query_positions(read, {100, 109, 110, 113, 115, 122, 127}) == {3, 12, 15, 18, 19, 24}
    assert masked_query_positions(read, set()) == set()


def test_staged_transfer_manifest_keeps_mask_digest(tmp_path):
    from pathlib import Path
    from test_consensus_bed_cli import make_bam
    from fiberhmm.inference.consensus.transfer_cli import main as transfer
    model = Path(__file__).parent / 'fixtures/paper_families/gata_tal1/models.json.gz'
    bam = tmp_path / 'reads.bam'; make_bam(bam)
    bed = tmp_path / 'windows.bed'; bed.write_text('chr1\t0\t300\tplus\t0\t+\n')
    mask = tmp_path / 'snps.bed'; mask.write_text('chr1\t161\t162\n')
    transfer(['--models', str(model), '--bam', str(bam), '--bed', str(bed), '--daf-snp-mask', str(mask),
              '--no-bam', '--output', str(tmp_path / 'masked')])
    receipt, = read_json(tmp_path / 'masked' / 'manifest.json')['daf_snp_masks']
    assert receipt['sha256'] == sha256(mask.read_bytes()).hexdigest() and receipt['sites'] == 1
    transfer(['--models', str(model), '--bam', str(bam), '--bed', str(bed), '--no-bam', '--output', str(tmp_path / 'plain')])
    assert 'daf_snp_masks' not in read_json(tmp_path / 'plain' / 'manifest.json')
