"""fiberhmm-call with a DAF SNP mask: a read whose MD evidence the mask
empties is skipped, not re-read from retained MM/ML tags (basecaller 6mA
would count as G->A deamination). The consensus loader has the same rule
(tests/test_consensus_snp_mask.py)."""
import array
import random
import subprocess
import sys

import pysam
import pytest

from fiberhmm.inference import engine

CONTIG = 'chrM'
REF_LEN = 6000


@pytest.fixture(autouse=True)
def policy():
    engine.configure_daf_chimera_filter()
    engine.configure_daf_snp_mask(None)
    engine.configure_daf_insert_evidence(None)
    yield
    engine.configure_daf_chimera_filter()
    engine.configure_daf_snp_mask(None)
    engine.configure_daf_insert_evidence(None)


def reference(seed=5):
    rng = random.Random(seed)
    return ''.join(rng.choice('ACGT') for _ in range(REF_LEN))


def header():
    return pysam.AlignmentHeader.from_dict({
        'HD': {'VN': '1.6', 'SO': 'coordinate'}, 'SQ': [{'SN': CONTIG, 'LN': REF_LEN}]})


def make_read(hdr, ref, name, start, length, converted, *, mm=None):
    """CT read over ``ref[start:start+length]`` with C->T at ``converted``
    (reference positions); MD from the alignment; optional MM/ML tags."""
    seq, md, run = [], [], 0
    for r in range(start, start + length):
        base = ref[r]
        out = 'T' if (r in converted and base == 'C') else base
        if out != base:
            md += [str(run), base]
            run = 0
        else:
            run += 1
        seq.append(out)
    md.append(str(run))
    read = pysam.AlignedSegment(hdr)
    read.query_name = name
    read.reference_id = 0
    read.reference_start = start
    read.mapping_quality = 60
    read.cigartuples = [(0, length)]
    read.query_sequence = ''.join(seq)
    read.query_qualities = pysam.qualitystring_to_array('I' * length)
    read.set_tag('MD', ''.join(md))
    if mm is not None:
        a_positions = [i for i, b in enumerate(seq) if b == 'A']
        chosen = a_positions[::7][:40]
        deltas, last = [], -1
        for q in chosen:
            deltas.append(sum(1 for i in a_positions if last < i < q))
            last = q
        read.set_tag('MM', 'A+a.,' + ','.join(map(str, deltas)) + ';')
        read.set_tag('ML', array.array('B', [250] * len(chosen)))
    return read


def masked_read():
    """A CT read whose only MD conversion is at one reference site, with
    retained 6mA MM/ML tags."""
    ref = reference()
    hdr = header()
    start = 1000
    site = next(r for r in range(start + 400, start + 2000) if ref[r] == 'C')
    return make_read(hdr, ref, 'only_masked', start, 2500, {site}, mm=True), site


def test_live_read_with_masked_md_is_skipped_not_read_from_mm():
    read, site = masked_read()
    plain = engine._extract_fiber_read_from_pysam(read, 'daf', 128)
    assert plain is not None and plain['_daf_strand'] == '+'
    with engine.daf_snp_mask_scope({CONTIG: frozenset({site})}):
        assert engine._extract_fiber_read_from_pysam(read, 'daf', 128) is None
        # Payload (slim-IPC) path: the producer records the decision.
        payload = engine.make_apply_payload(read, 'daf')
        assert payload.get('_daf_mask_emptied') is True
        assert engine.extract_fiber_read_from_payload(payload, 'daf', 128) is None
    # A mask elsewhere keeps the MD reading.
    with engine.daf_snp_mask_scope({CONTIG: frozenset({site + 3})}):
        kept = engine._extract_fiber_read_from_pysam(read, 'daf', 128)
        payload = engine.make_apply_payload(read, 'daf')
    assert kept is not None and kept['_daf_strand'] == '+'
    assert '_daf_mask_emptied' not in payload


def test_unmasked_extraction_is_unchanged():
    read, _ = masked_read()
    payload = engine.make_apply_payload(read, 'daf')
    assert '_daf_mask_emptied' not in payload
    live = engine._extract_fiber_read_from_pysam(read, 'daf', 128)
    stub = engine.extract_fiber_read_from_payload(payload, 'daf', 128)
    assert live['m6a_query_positions'] == stub['m6a_query_positions']


def test_md_without_conversions_still_reads_mm_under_a_mask():
    """No reference evidence before or after masking: MM/ML applies, as
    without a mask."""
    ref = reference()
    read = make_read(header(), ref, 'no_conversions', 1000, 2500, set(), mm=True)
    plain = engine._extract_fiber_read_from_pysam(read, 'daf', 128)
    with engine.daf_snp_mask_scope({CONTIG: frozenset({1500})}):
        masked = engine._extract_fiber_read_from_pysam(read, 'daf', 128)
        payload = engine.make_apply_payload(read, 'daf')
    assert '_daf_mask_emptied' not in payload
    assert (plain is None) == (masked is None)
    if plain is not None:
        assert plain['m6a_query_positions'] == masked['m6a_query_positions']


def _records(path):
    with pysam.AlignmentFile(str(path), 'rb', check_sq=False) as bam:
        return {r.query_name: r for r in bam.fetch(until_eof=True)}


def test_call_cli_skips_the_masked_read_streaming_and_region_parallel(tmp_path):
    ref = reference()
    hdr = header()
    read, site = masked_read()
    rng = random.Random(3)
    reads = [read]
    for i in range(4):
        start = 200 + 600 * i
        sites = {r for r in range(start, start + 2500) if ref[r] == 'C' and rng.random() < 0.3}
        sites.discard(site)
        sites = {r for r in sites if abs(r - site) > 5}
        reads.append(make_read(hdr, ref, f'clean{i}', start, 2500, sites, mm=True))
    reads.sort(key=lambda r: r.reference_start)
    bam = tmp_path / 'in.bam'
    with pysam.AlignmentFile(str(bam), 'wb', header=hdr) as out:
        for r in reads:
            out.write(r)
    pysam.index(str(bam))
    mask = tmp_path / 'mask.bed'
    mask.write_text(f'{CONTIG}\t{site}\t{site + 1}\n')

    def call(target, *extra):
        command = [sys.executable, '-m', 'fiberhmm.cli.call', '-i', str(bam),
                   '-o', str(target), '--enzyme', 'dddb', '-c', '1', '--io-threads', '1',
                   '--no-dedup', '--no-daf-call-snps', '--daf-insert-consensus', 'off',
                   '--no-qc', *extra]
        run = subprocess.run(command, capture_output=True, text=True, timeout=300)
        assert run.returncode == 0, run.stderr
        return _records(target)

    plain = call(tmp_path / 'plain.bam')
    masked = call(tmp_path / 'masked.bam', '--daf-snp-mask', str(mask))
    parallel = call(tmp_path / 'rp.bam', '--daf-snp-mask', str(mask), '--region-parallel')
    called = lambda r: r.has_tag('MA') or r.has_tag('ns') or r.has_tag('as')
    assert called(plain['only_masked'])
    assert not called(masked['only_masked']) and not called(parallel['only_masked'])
    for name in ('clean0', 'clean1', 'clean2', 'clean3'):
        assert called(masked[name])
        assert masked[name].get_tag('MA') == parallel[name].get_tag('MA')
