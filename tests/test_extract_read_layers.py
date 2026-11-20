"""fiberhmm-extract var (--variants) and sv (--sv) layers on synthetic data."""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys

import pysam
import pytest

from fiberhmm.cli.extract_tags import bed_to_bigbed, extract_tags_parallel
from fiberhmm.io.autosql import (
    DUPLICATE_FIELD_COUNT,
    HAPLOTYPE_FIELD_COUNT,
    LAYER_BLOCK_COLUMN_COUNTS,
    LAYER_FIELD_COUNTS,
    get_schema,
    vcf_source_note,
)
from fiberhmm.io.circular_bed import block_column_count, fold_bed12_row
from fiberhmm.io.read_layers import (
    ALLELE_DAF_AMBIGUOUS,
    ALLELE_DELETED,
    ALLELE_OTHER,
    SV_CLIP_LEFT,
    SV_CLIP_RIGHT,
    SV_DEL,
    SV_INS,
    VariantOptions,
    VariantSite,
    bam_looks_daf,
    daf_conversion,
    find_daf_snp_vcf,
    genotype_read,
    pack_blocks,
    prepare_indexed_vcf,
    record_to_site,
    sa_column,
    sv_events,
)

random.seed(7)
REF = ''.join(random.choice('ACGT') for _ in range(6000))


def _set(seq, pos, base):
    return seq[:pos] + base + seq[pos + 1:]


# Fixed bases at the sites used below.
REF = _set(REF, 1000, 'C')
REF = _set(REF, 1100, 'T')
REF = _set(REF, 1200, 'G')
REF = _set(REF, 1300, 'A')
REF = REF[:1400] + 'ATG' + REF[1403:]      # deletion site REF=ATG ALT=A
REF = _set(REF, 1500, 'G')                 # insertion site REF=G ALT=GCC

HEADER = pysam.AlignmentHeader.from_dict({
    'HD': {'VN': '1.6', 'SO': 'coordinate'},
    'SQ': [{'SN': 'chr1', 'LN': len(REF)}, {'SN': 'chr2', 'LN': len(REF)}],
})


def _md(read_seq, ref_seq):
    """MD string for an M-only alignment."""
    out, run = [], 0
    for q, r in zip(read_seq, ref_seq):
        if q == r:
            run += 1
        else:
            out.append(f'{run}{r}')
            run = 0
    out.append(str(run))
    return ''.join(out)


def _read(name, start, seq, cigar=None, *, flag=0, mapq=60, tags=(), qual='5',
          ref_id=0):
    a = pysam.AlignedSegment(HEADER)
    a.query_name = name
    a.query_sequence = seq
    a.flag = flag
    a.reference_id = ref_id
    a.reference_start = start
    a.mapping_quality = mapq
    a.cigartuples = cigar or [(0, len(seq))]
    a.query_qualities = pysam.qualitystring_to_array(qual * len(seq))
    for tag, value in tags:
        a.set_tag(tag, value)
    return a


def _site(start, *alleles):
    ref = alleles[0]
    indel = any(len(a) != len(ref) for a in alleles[1:])
    return VariantSite(start, start + len(ref), tuple(alleles), indel)


# ---------------------------------------------------------------------------
# genotyping
# ---------------------------------------------------------------------------

def test_snv_ref_alt_other_and_multiallelic():
    site = _site(1000, 'C', 'T', 'G')
    for base, code in (('C', 0), ('T', 1), ('G', 2), ('A', ALLELE_OTHER)):
        seq = _set(REF[900:1100], 100, base)
        read = _read('r', 900, seq, qual='?')
        [(s, e, allele, bq)] = genotype_read(read, [site])
        assert (s, e, allele, bq) == (1000, 1001, code, 30)


def test_site_outside_read_is_not_reported():
    read = _read('r', 900, REF[900:1000])
    assert genotype_read(read, [_site(1000, 'C', 'T')]) == []


def test_deletion_and_insertion_alleles():
    del_site = _site(1400, 'ATG', 'A')
    ins_site = _site(1500, 'G', 'GCC')
    ref_read = _read('ref', 1350, REF[1350:1600])
    calls = genotype_read(ref_read, [del_site, ins_site])
    assert [c[2] for c in calls] == [0, 0]
    # read with the 2 bp deletion after the anchor and a CC insertion after 1500
    seq = REF[1350:1401] + REF[1403:1501] + 'CC' + REF[1501:1600]
    cigar = [(0, 51), (2, 2), (0, 98), (1, 2), (0, 99)]
    alt = _read('alt', 1350, seq, cigar)
    calls = genotype_read(alt, [del_site, ins_site])
    assert [(c[0], c[1], c[2]) for c in calls] == [(1400, 1403, 1), (1500, 1501, 1)]
    # a 1 bp insertion matches neither allele
    seq = REF[1350:1501] + 'C' + REF[1501:1600]
    other = _read('other', 1350, seq, [(0, 151), (1, 1), (0, 99)])
    assert genotype_read(other, [ins_site])[0][2] == ALLELE_OTHER


def test_indel_site_at_alignment_end_is_not_covered():
    ins_site = _site(1500, 'G', 'GCC')
    read = _read('r', 1400, REF[1400:1501])   # ends on the anchor base
    assert genotype_read(read, [ins_site]) == []


def test_spanning_deletion():
    snv = _site(1000, 'C', 'T')
    seq = REF[900:990] + REF[1020:1100]
    read = _read('d', 900, seq, [(0, 90), (2, 30), (0, 80)])
    assert genotype_read(read, [snv]) == [(1000, 1001, ALLELE_DELETED, -1)]


def test_missing_qualities_give_minus_one():
    read = _read('r', 900, REF[900:1100])
    read.query_qualities = None
    assert genotype_read(read, [_site(1000, 'C', 'T')])[0][3] == -1


@pytest.mark.parametrize('conversion,base,site,code', [
    # C->T read: a T at a C/T site may be a converted C
    ('CT', 'T', ('C', 'T'), ALLELE_DAF_AMBIGUOUS),
    ('CT', 'C', ('C', 'T'), 0),
    ('CT', 'T', ('T', 'C'), ALLELE_DAF_AMBIGUOUS),
    ('CT', 'C', ('T', 'C'), 1),
    ('CT', 'A', ('G', 'A'), 1),          # G/A sites are informative on C->T reads
    # G->A read
    ('GA', 'A', ('G', 'A'), ALLELE_DAF_AMBIGUOUS),
    ('GA', 'G', ('G', 'A'), 0),
    ('GA', 'T', ('C', 'T'), 1),
    # undecided DAF strand: both conversions possible
    ('both', 'T', ('C', 'T'), ALLELE_DAF_AMBIGUOUS),
    ('both', 'A', ('G', 'A'), ALLELE_DAF_AMBIGUOUS),
])
def test_daf_ambiguity(conversion, base, site, code):
    seq = _set(REF[900:1100], 100, base)
    read = _read('r', 900, seq)
    assert genotype_read(read, [_site(1000, *site)], conversion)[0][2] == code


def _daf_options(daf='auto', bam_is_daf=False):
    return VariantOptions(path='x', source_path='x', daf=daf, bam_is_daf=bam_is_daf)


def test_daf_ry_encoded_reads_on_both_strands():
    # R/Y input (fiberhmm-daf-encode): Y = converted C, R = converted G.
    seq = REF[900:1300]
    ct = ''.join('Y' if b == 'C' and i % 3 == 0 else b for i, b in enumerate(seq))
    ct = _set(ct, 100, 'Y')       # the C/T site at 1000 is converted
    ct_read = _read('ct', 900, ct, tags=[('st', 'CT')])
    assert daf_conversion(ct_read, _daf_options()) == 'CT'
    calls = genotype_read(ct_read, [_site(1000, 'C', 'T'), _site(1200, 'G', 'A')],
                          daf_conversion(ct_read, _daf_options()))
    assert [c[2] for c in calls] == [ALLELE_DAF_AMBIGUOUS, 0]

    ga = ''.join('R' if b == 'G' and i % 3 == 0 else b for i, b in enumerate(seq))
    ga = _set(ga, 300, 'R')       # the G/A site at 1200 is converted
    ga_read = _read('ga', 900, ga, tags=[('st', 'GA')])
    conversion = daf_conversion(ga_read, _daf_options())
    assert conversion == 'GA'
    calls = genotype_read(ga_read, [_site(1000, 'C', 'T'), _site(1200, 'G', 'A')], conversion)
    assert [c[2] for c in calls] == [0, ALLELE_DAF_AMBIGUOUS]


def test_daf_strand_from_md_and_modes():
    seq = REF[900:1300]
    conv = ''.join('T' if b == 'C' and i % 4 == 0 else b for i, b in enumerate(seq))
    read = _read('md', 900, conv, tags=[('MD', _md(conv, seq))])
    # No R/Y, no st: auto needs BAM-level DAF evidence.
    assert daf_conversion(read, _daf_options()) is None
    assert daf_conversion(read, _daf_options(bam_is_daf=True)) == 'CT'
    assert daf_conversion(read, _daf_options(daf='on')) == 'CT'
    assert daf_conversion(read, _daf_options(daf='off', bam_is_daf=True)) is None
    # A DAF read without conversions has no decided strand.
    plain = _read('plain', 900, seq, tags=[('MD', _md(seq, seq))])
    assert daf_conversion(plain, _daf_options(daf='on')) == 'both'


def test_bam_level_daf_evidence_comes_from_the_header():
    def pg(program, ds=''):
        return {'PG': [{'ID': program, 'PN': program, 'DS': ds}]}
    assert bam_looks_daf('x', pg('fiberhmm-call', 'coord=molecular; mode=daf enzyme=ddda'))
    assert bam_looks_daf('x', pg('fiberhmm-daf-encode'))
    assert not bam_looks_daf('x', pg('fiberhmm-call', 'mode=pacbio-fiber'))
    assert not bam_looks_daf('x', pg('fibertools-rs'))   # m6A T-a is not DAF


# ---------------------------------------------------------------------------
# VCF records
# ---------------------------------------------------------------------------

VCF_TEXT = """##fileformat=VCFv4.3
##contig=<ID=chr1,length={n}>
##FILTER=<ID=LowQual,Description="low">
##ALT=<ID=DEL,Description="deletion">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
chr1\t1001\t.\tC\tT,G\t.\tPASS\t.\tGT\t0|1
chr1\t1101\t.\tT\tC\t.\tLowQual\t.\tGT\t0/1
chr1\t1201\t.\tG\tA\t.\t.\t.\tGT\t1/1
chr1\t1301\t.\tA\tG\t.\tPASS\t.\tGT\t0/0
chr1\t1401\t.\tATG\tA\t.\tPASS\t.\tGT\t0|1
chr1\t1501\t.\tG\tGCC\t.\tPASS\t.\tGT\t1|0
chr1\t1601\t.\tA\t<DEL>\t.\tPASS\t.\tGT\t0/1
chr1\t1701\t.\tA\t{long}\t.\tPASS\t.\tGT\t0/1
""".format(n=len(REF), long='A' + 'C' * 60)


@pytest.fixture
def vcf_path(tmp_path):
    path = tmp_path / 'calls.vcf'
    path.write_text(VCF_TEXT)
    return str(path)


def _sites(path, **kwargs):
    options = VariantOptions(path=path, source_path=path, **kwargs)
    with pysam.VariantFile(path) as vcf:
        return [s for s in (record_to_site(r, options) for r in vcf) if s is not None]


def test_record_filters(vcf_path):
    starts = [s.start for s in _sites(vcf_path)]
    # LowQual, symbolic-only and too-long records are dropped; '.' FILTER kept.
    assert starts == [1000, 1200, 1300, 1400, 1500]
    assert [s.start for s in _sites(vcf_path, pass_only=False)] == [1000, 1100, 1200, 1300, 1400, 1500]
    assert [s.start for s in _sites(vcf_path, sample='S1')] == [1000, 1200, 1400, 1500]
    assert [s.start for s in _sites(vcf_path, sample='S1', het_only=True)] == [1000, 1400, 1500]
    assert [s.start for s in _sites(vcf_path, max_allele_len=100)][-1] == 1700


def test_prepare_indexed_vcf_handles_plain_vcf(vcf_path, tmp_path):
    work = tmp_path / 'work'
    work.mkdir()
    indexed = prepare_indexed_vcf(vcf_path, str(work))
    assert indexed != vcf_path
    with pysam.VariantFile(indexed) as vcf:
        assert vcf.index is not None
        assert len(list(vcf.fetch('chr1', 990, 1010))) == 1
    # an already-indexed file is used in place
    assert prepare_indexed_vcf(indexed, str(work)) == indexed


def test_prepare_indexed_vcf_indexes_an_unindexed_bcf_in_a_fresh_process(tmp_path):
    # `pysam.bcftools` is a submodule: a process that never imported it must still index BCF.
    bcf = tmp_path / 'calls.bcf'
    header = pysam.VariantHeader()
    header.contigs.add('chr1', length=5000)
    with pysam.VariantFile(str(bcf), 'wb', header=header) as dst:
        dst.write(dst.new_record(contig='chr1', start=1000, alleles=('A', 'G')))
    work = tmp_path / 'work'
    work.mkdir()
    code = (
        'import sys, pysam\n'
        'from fiberhmm.io.read_layers import prepare_indexed_vcf\n'
        'out = prepare_indexed_vcf(sys.argv[1], sys.argv[2])\n'
        'with pysam.VariantFile(out) as vcf:\n'
        '    assert vcf.index is not None\n'
        '    print(len(list(vcf.fetch("chr1", 990, 1010))))\n'
    )
    result = subprocess.run([sys.executable, '-c', code, str(bcf), str(work)],
                            capture_output=True, text=True, env=os.environ.copy())
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '1'


def test_find_daf_snp_vcf(tmp_path):
    bam = tmp_path / 'run.calls.bam'
    with pysam.AlignmentFile(str(bam), 'wb', header=HEADER):
        pass
    assert find_daf_snp_vcf(str(bam))[0] is None
    qc = tmp_path / 'qc'
    qc.mkdir()
    (qc / 'run.calls.daf_snps.vcf').write_text(VCF_TEXT)
    assert find_daf_snp_vcf(str(bam))[0] == str(qc / 'run.calls.daf_snps.vcf')
    # a prefix recorded on the fiberhmm-call @PG wins
    other = tmp_path / 'elsewhere.vcf'
    other.write_text(VCF_TEXT)
    header = {'PG': [{'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call',
                      'CL': f'fiberhmm-call -i x --daf-snp-output-prefix {tmp_path / "elsewhere"}'}]}
    assert find_daf_snp_vcf(str(bam), header)[0] == str(other)


# ---------------------------------------------------------------------------
# sv events
# ---------------------------------------------------------------------------

def test_sv_events_cigar_and_clips():
    seq = 'A' * 150 + REF[2000:2100] + 'G' * 40 + REF[2100:2200] + REF[2260:2400] + 'T' * 120
    cigar = [(4, 150), (0, 100), (1, 40), (0, 100), (2, 60), (0, 140), (4, 120)]
    read = _read('sv', 2000, seq, cigar)
    events = sv_events(read, 30, 100)
    assert events == [
        (2000, 2001, SV_CLIP_LEFT, 150),
        (2099, 2100, SV_INS, 40),
        (2200, 2260, SV_DEL, 60),
        (2399, 2400, SV_CLIP_RIGHT, 120),
    ]
    assert sv_events(read, 50, 160) == [(2200, 2260, SV_DEL, 60)]


def test_sv_hard_clips_count_and_small_events_skipped():
    read = _read('h', 2000, REF[2000:2100] + REF[2110:2200],
                 [(5, 300), (0, 100), (2, 10), (0, 90), (5, 20)])
    assert sv_events(read, 30, 100) == [(2000, 2001, SV_CLIP_LEFT, 300)]


def test_sa_column():
    read = _read('r', 0, 'ACGT')
    assert sa_column(read) == '.'
    read.set_tag('SA', 'chr2,100,+,50S50M,60,0;')
    assert sa_column(read) == 'chr2,100,+,50S50M,60,0;'
    read.set_tag('SA', 'chr2,100,+,50S50M,60,0;' * 400)
    value = sa_column(read)
    assert len(value) <= 4000 and value.endswith(';...')


def test_pack_blocks_splits_overlaps():
    rows = pack_blocks([(10, 20, 'a'), (15, 16, 'b'), (20, 21, 'c'), (30, 31, 'd')])
    assert rows == [[(10, 20, 'a'), (20, 21, 'c'), (30, 31, 'd')], [(15, 16, 'b')]]


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

def test_layer_schemas():
    var = get_schema('var', haplotype_fields=True)
    assert 'table fiberhmm_var' in var
    assert var.index('blockAllele') < var.index('blockBq') < var.index(' hp;') < var.index('isDuplicate')
    sv = get_schema('sv')
    assert 'table fiberhmm_sv' in sv
    order = [sv.index(f) for f in ('blockType', 'blockLen', 'uint mapq', 'uint readLen',
                                   'uint flag', 'lstring sa', 'isDuplicate')]
    assert order == sorted(order)
    assert LAYER_FIELD_COUNTS == {'var': 2, 'sv': 6}
    assert LAYER_BLOCK_COLUMN_COUNTS == {'var': 2, 'sv': 2}
    assert block_column_count('var', False) == 2
    assert block_column_count('sv', True) == 2
    note = vcf_source_note('/data/my "calls".vcf.gz', 'abc123', None)
    assert note == 'VCF: name=my_calls_.vcf.gz; md5=abc123; sample=*.'
    described = get_schema('var', sample_name='S', source_note=note)
    assert '"Sample: S. VCF: name=my_calls_.vcf.gz; md5=abc123; sample=*. FiberHMM' in described


def test_circular_fold_keeps_layer_columns_with_blocks():
    row = 'chr1\t90\t120\tr\t0\t+\t90\t120\t0\t2\t1,1\t0,29\t1,0\t30,31\t0'
    head, tail = fold_bed12_row(row, 100, 2)
    assert head.split('\t')[12:14] == ['1', '30']
    assert tail.split('\t')[12:14] == ['0', '31']


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def _write_bam(path):
    reads = []
    # alt at 1000 (T), ref elsewhere
    reads.append(_read('alt', 900, _set(REF[900:1600], 100, 'T'),
                       tags=[('HP', 1), ('PS', 7)]))
    reads.append(_read('ref', 950, REF[950:1650]))
    # 2 bp deletion at 1401 + insertion after 1500; both alleles alt
    seq = REF[1350:1401] + REF[1403:1501] + 'CC' + REF[1501:1700]
    reads.append(_read('indel', 1350, seq, [(0, 51), (2, 2), (0, 98), (1, 2), (0, 199)]))
    # SV read: 150 bp clip, 40 bp deletion, SA; plus its supplementary
    seq = 'A' * 150 + REF[3000:3100] + REF[3140:3300]
    reads.append(_read('split', 3000, seq, [(4, 150), (0, 100), (2, 40), (0, 160)],
                       tags=[('SA', 'chr2,501,-,100M260S,60,0;')]))
    reads.append(_read('split', 500, REF[500:600], [(5, 260), (0, 100)],
                       flag=2048 | 16, ref_id=1, tags=[('SA', 'chr1,3001,+,150S260M,60,0;')]))
    # secondary records are ignored
    reads.append(_read('sec', 3000, REF[3000:3300], flag=256))
    reads.sort(key=lambda r: (r.reference_id, r.reference_start))
    with pysam.AlignmentFile(str(path), 'wb', header=HEADER) as out:
        for read in reads:
            out.write(read)
    pysam.index(str(path))


def _rows(path):
    with open(path) as handle:
        return [line.rstrip('\n').split('\t') for line in handle]


def _extract(tmp_path, vcf_path, types=('var', 'sv'), **kwargs):
    bam = tmp_path / 'sample.bam'
    _write_bam(bam)
    beds = {t: str(tmp_path / f'sample_{t}.bed') for t in types}
    options = VariantOptions(path=vcf_path, source_path=vcf_path)
    extract_tags_parallel(str(bam), beds, list(types), n_cores=1,
                          variants=options if 'var' in types else None, **kwargs)
    return bam, beds


def test_extract_var_and_sv_rows(tmp_path, vcf_path):
    _bam, beds = _extract(tmp_path, vcf_path, haplotype_fields=True)
    var = {r[3]: r for r in _rows(beds['var'])}
    assert set(var) == {'alt', 'ref', 'indel'}
    alt = var['alt']
    # sites 1000, 1200, 1300, 1400, 1500 (blocks = REF length)
    assert alt[1] == '1000' and alt[9] == '5'
    assert alt[10] == '1,1,1,3,1'
    assert alt[12] == '1,0,0,0,0'
    assert alt[13] == '20,20,20,20,20'
    assert alt[14:] == ['1', '7', '0']                 # hp, ps, isDuplicate
    assert var['indel'][12] == '1,1'                    # deletion and insertion alts
    assert var['ref'][14:] == ['-1', '-1', '0']

    sv = _rows(beds['sv'])
    assert [(r[0], r[3]) for r in sv] == [('chr1', 'split'), ('chr2', 'split')]
    primary, supp = sv
    assert primary[1:3] == ['3000', '3140']
    assert primary[12] == f'{SV_CLIP_LEFT},{SV_DEL}' and primary[13] == '150,40'
    assert primary[14:18] == ['60', '410', '0', 'chr2,501,-,100M260S,60,0;']
    assert supp[12:18] == [str(SV_CLIP_LEFT), '260', '60', '360', str(2048 | 16),
                           'chr1,3001,+,150S260M,60,0;']
    assert supp[5] == '-'


def test_extract_var_sample_restriction(tmp_path, vcf_path):
    bam = tmp_path / 'sample.bam'
    _write_bam(bam)
    bed = str(tmp_path / 'v.bed')
    options = VariantOptions(path=vcf_path, source_path=vcf_path, sample='S1', het_only=True)
    extract_tags_parallel(str(bam), {'var': bed}, ['var'], variants=options)
    alt = {r[3]: r for r in _rows(bed)}['alt']
    assert alt[10] == '1,3,1'          # 1000, 1400, 1500


def test_var_requires_options(tmp_path):
    bam = tmp_path / 'sample.bam'
    _write_bam(bam)
    with pytest.raises(ValueError, match='variants'):
        extract_tags_parallel(str(bam), {'var': str(tmp_path / 'v.bed')}, ['var'])


@pytest.mark.skipif(shutil.which('bedToBigBed') is None, reason='bedToBigBed not installed')
def test_bigbed_round_trip(tmp_path, vcf_path):
    _bam, beds = _extract(tmp_path, vcf_path)
    sizes = {'chr1': len(REF), 'chr2': len(REF)}
    note = vcf_source_note(vcf_path, 'deadbeef', None)
    for layer in ('var', 'sv'):
        bb = str(tmp_path / f'{layer}.bb')
        assert bed_to_bigbed(beds[layer], bb, sizes, extract_type=layer, sample_name='S',
                             source_note=note if layer == 'var' else None)
        pyBigWig = pytest.importorskip('pyBigWig')
        handle = pyBigWig.open(bb)
        try:
            schema = handle.SQL().decode()
            assert f'table fiberhmm_{layer}' in schema
            if layer == 'var':
                assert 'md5=deadbeef' in schema
                entries = handle.entries('chr1', 0, len(REF))
                rest = {e[2].split('\t')[0]: e[2].split('\t') for e in entries}
                # rest = name, score, strand, thick..., blocks..., extras
                assert rest['alt'][9] == '1,0,0,0,0'
            else:
                entries = handle.entries('chr2', 0, len(REF))
                fields = entries[0][2].split('\t')
                assert fields[-1] == '0' and fields[-2].startswith('chr1,3001')
        finally:
            handle.close()


def test_cli_all_includes_sv_and_skips_var_without_vcf(tmp_path):
    bam = tmp_path / 'sample.bam'
    _write_bam(bam)
    out = tmp_path / 'out'
    result = subprocess.run(
        [sys.executable, '-m', 'fiberhmm.cli.extract_tags', '-i', str(bam), '-o', str(out),
         '--bed-only'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'skipping var' in result.stdout
    assert (out / 'sample_sv.bed').exists()
    assert not (out / 'sample_var.bed').exists()


def test_cli_all_uses_daf_snp_vcf(tmp_path):
    bam = tmp_path / 'sample.bam'
    _write_bam(bam)
    (tmp_path / 'qc').mkdir()
    (tmp_path / 'qc' / 'sample.daf_snps.vcf').write_text(VCF_TEXT)
    out = tmp_path / 'out'
    result = subprocess.run(
        [sys.executable, '-m', 'fiberhmm.cli.extract_tags', '-i', str(bam), '-o', str(out),
         '--all', '--bed-only'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'DAF SNP screen VCF' in result.stdout
    assert (out / 'sample_var.bed').exists()


def test_cli_variants_auto_without_vcf_fails(tmp_path):
    bam = tmp_path / 'sample.bam'
    _write_bam(bam)
    result = subprocess.run(
        [sys.executable, '-m', 'fiberhmm.cli.extract_tags', '-i', str(bam), '-o',
         str(tmp_path / 'out'), '--variants', 'auto'], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'found no DAF SNP VCF' in result.stderr


def test_cli_single_layer_selection(tmp_path, vcf_path):
    bam = tmp_path / 'sample.bam'
    _write_bam(bam)
    out = tmp_path / 'out'
    result = subprocess.run(
        [sys.executable, '-m', 'fiberhmm.cli.extract_tags', '-i', str(bam), '-o', str(out),
         '--variants', vcf_path, '--variants-sample', 'S1', '--bed-only'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert sorted(os.listdir(out)) == ['sample_var.bed']


# ---------------------------------------------------------------------------
# graph lift (feat/pg-lift); skipped where the lift module is not present
# ---------------------------------------------------------------------------

def test_lift_reassembly_keeps_layer_columns_with_blocks():
    graph_lift = pytest.importorskip('fiberhmm.io.graph_lift')
    row = 'chr1\t1000\t1501\talt\t20\t+\t1000\t1501\t0\t3\t1,3,1\t0,400,500\t1,-2,0\t30,-1,25\t-1\t-1\t0'
    bed6, rows, _header = graph_lift.explode_bed12([row])
    # identity lift with the middle block lost
    lifted = [line for line in bed6 if not line.split('\t')[3].endswith('|1')]
    out, stats = graph_lift.reassemble_bed12(lifted, rows, partial='keep')
    fields = out[0].split('\t')
    assert fields[9] == '2' and fields[10] == '1,1'
    assert fields[12:14] == ['1,0', '30,25']
    assert fields[14:] == ['-1', '-1', '0']
    assert stats.blocks_dropped == 1


def test_lift_reassembly_leaves_sa_column_alone():
    graph_lift = pytest.importorskip('fiberhmm.io.graph_lift')
    row = ('chr1\t0\t60\tr\t60\t+\t0\t60\t0\t6\t1,1,1,1,1,1\t0,10,20,30,40,59\t'
           '2,0,0,0,0,3\t100,40,40,40,40,120\t60\t500\t0\tchr2,5,+,10M,60,0;\t0')
    bed6, rows, _header = graph_lift.explode_bed12([row])
    lifted = [line for line in bed6 if not line.split('\t')[3].endswith('|2')]
    out, _stats = graph_lift.reassemble_bed12(lifted, rows, partial='keep')
    assert out[0].split('\t')[17] == 'chr2,5,+,10M,60,0;'


# ---------------------------------------------------------------------------
# 3.2.0 audit fixes (M3, M5, M6)
# ---------------------------------------------------------------------------

def test_longer_deletion_does_not_match_a_shorter_alt_deletion():
    """REF=AC ALT=A read through by 1M3D10M: the read deletes C and the two
    bases after it, which is no allele of the site (audit M3)."""
    site = _site(1400, 'AT', 'A')
    seq = REF[1400:1401] + REF[1404:1414]
    longer = _read('longer', 1400, seq, [(0, 1), (2, 3), (0, 10)])
    assert genotype_read(longer, [site]) == [(1400, 1402, ALLELE_OTHER, -1)]
    # the exact 1 bp deletion is still ALT, and the reference still REF
    seq = REF[1400:1401] + REF[1402:1414]
    exact = _read('exact', 1400, seq, [(0, 1), (2, 1), (0, 12)])
    assert genotype_read(exact, [site])[0][2] == 1
    ref = _read('ref', 1400, REF[1400:1414])
    assert genotype_read(ref, [site])[0][2] == 0


def test_seq_star_and_unmapped_records_are_skipped():
    """SEQ '*' on a mapped record is not genotyped (no IndexError), and
    sv_events handles a record without reference_end (audit M6)."""
    star = pysam.AlignedSegment(HEADER)
    star.query_name = 'star'
    star.flag = 256
    star.reference_id = 0
    star.reference_start = 990
    star.mapping_quality = 0
    star.cigartuples = [(0, 20)]
    assert star.query_sequence is None and star.reference_end == 1010
    assert genotype_read(star, [_site(1000, 'C', 'T')]) == []
    unmapped = pysam.AlignedSegment(HEADER)
    unmapped.query_name = 'u'
    unmapped.flag = 4
    unmapped.query_sequence = 'ACGT'
    assert unmapped.reference_end is None
    assert sv_events(unmapped) == []
    assert genotype_read(unmapped, [_site(1000, 'C', 'T')]) == []


CIRC = REF[:100]
CIRC_HEADER = pysam.AlignmentHeader.from_dict({
    'HD': {'VN': '1.6', 'SO': 'coordinate'},
    'SQ': [{'SN': 'plasmid', 'LN': 100, 'TP': 'circular'}],
})


def _circular_vcf(tmp_path):
    text = ('##fileformat=VCFv4.3\n##contig=<ID=plasmid,length=100>\n'
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
            '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n'
            f'plasmid\t6\t.\t{CIRC[5]}\t{"A" if CIRC[5] != "A" else "C"}\t.\tPASS\t.\tGT\t0/1\n'
            f'plasmid\t96\t.\t{CIRC[95]}\t{"A" if CIRC[95] != "A" else "C"}\t.\tPASS\t.\tGT\t0/1\n')
    path = tmp_path / 'circ.vcf'
    path.write_text(text)
    return prepare_indexed_vcf(str(path), str(tmp_path))


def test_site_window_returns_sites_after_a_circular_origin(tmp_path):
    from fiberhmm.io.read_layers import SiteWindow
    path = _circular_vcf(tmp_path)
    options = VariantOptions(path=path, source_path=path)
    with pysam.VariantFile(path) as vcf:
        linear = SiteWindow(vcf, 'plasmid', options, 0)
        assert [s.start for s in linear.sites_in(90, 120)] == [95]
        circular = SiteWindow(vcf, 'plasmid', options, 0)
        circular.set_circular_from_header(CIRC_HEADER)
        assert circular.circular_length == 100
        assert [(s.start, s.end) for s in circular.sites_in(90, 120)] == [(95, 96), (105, 106)]
        assert [s.start for s in circular.sites_in(92, 104)] == [95]   # 5 not reached
        assert [s.start for s in circular.sites_in(0, 30)] == [5]


def test_extract_var_genotypes_origin_spanning_reads(tmp_path):
    path = _circular_vcf(tmp_path)
    bam = tmp_path / 'circ.bam'
    seq = CIRC[90:] + CIRC[:20]              # 90..120 through the origin
    alt = 'A' if CIRC[5] != 'A' else 'C'
    seq = seq[:15] + alt + seq[16:]          # ALT at site 5 (read offset 15)
    a = pysam.AlignedSegment(CIRC_HEADER)
    a.query_name = 'origin'
    a.query_sequence = seq
    a.flag = 0
    a.reference_id = 0
    a.reference_start = 90
    a.mapping_quality = 60
    a.cigartuples = [(0, 30)]
    a.query_qualities = pysam.qualitystring_to_array('5' * 30)
    with pysam.AlignmentFile(str(bam), 'wb', header=CIRC_HEADER) as out:
        out.write(a)
    pysam.index(str(bam))
    beds = {'var': str(tmp_path / 'circ_var.bed')}
    options = VariantOptions(path=path, source_path=path)
    extract_tags_parallel(str(bam), beds, ['var'], n_cores=1, variants=options)
    (row,) = _rows(beds['var'])
    starts = [int(row[1]) + int(x) for x in row[11].split(',') if x]
    assert starts == [95, 105]
    assert row[12].split(',')[:2] == ['0', '1']
