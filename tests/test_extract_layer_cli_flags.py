"""fiberhmm-extract var/sv (3.2): each command-line flag reaches its option.

tests/test_extract_read_layers.py tests the record filters and SV thresholds
through keyword arguments; here the same BAM and VCF go through the CLI, one
flag at a time, and the sites in the written BED are compared.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from test_extract_read_layers import VCF_TEXT, _rows, _write_bam


@pytest.fixture
def inputs(tmp_path):
    bam = tmp_path / 'sample.bam'
    _write_bam(bam)
    vcf = tmp_path / 'calls.vcf'
    vcf.write_text(VCF_TEXT)
    return tmp_path, bam, vcf


def _extract(tmp_path, bam, *flags, layer):
    out = tmp_path / f'out{abs(hash(flags))}'
    proc = subprocess.run([sys.executable, '-m', 'fiberhmm.cli.extract_tags', '-i', str(bam), '-o', str(out),
                           '--bed-only', *flags], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    rows = _rows(out / f'sample_{layer}.bed')
    sites = sorted({int(f[1]) + int(b) for f in rows for b in f[11].strip(',').split(',')})
    return rows, sites


def test_variant_record_filters_from_the_command_line(inputs):
    tmp_path, bam, vcf = inputs
    v = str(vcf)
    assert _extract(tmp_path, bam, '--variants', v, layer='var')[1] == [1000, 1200, 1300, 1400, 1500]
    assert _extract(tmp_path, bam, '--variants', v, '--no-variants-pass-only', layer='var')[1] == [1000, 1100, 1200, 1300, 1400, 1500]
    assert _extract(tmp_path, bam, '--variants', v, '--variants-sample', 'S1', '--variants-het-only', layer='var')[1] == [1000, 1400, 1500]
    # max allele length 0: the deletion and insertion sites go, SNVs stay
    assert _extract(tmp_path, bam, '--variants', v, '--variants-max-len', '0', layer='var')[1] == [1000, 1200, 1300]


def test_variants_daf_on_marks_conversion_ambiguous_alleles(inputs):
    tmp_path, bam, vcf = inputs
    rows_auto, _ = _extract(tmp_path, bam, '--variants', str(vcf), layer='var')
    rows_on, _ = _extract(tmp_path, bam, '--variants', str(vcf), '--variants-daf', 'on', layer='var')
    alt_auto = next(r for r in rows_auto if r[3] == 'alt')[12]
    alt_on = next(r for r in rows_on if r[3] == 'alt')[12]
    assert alt_auto == '1,0,0,0,0'
    # with DAF handling on, the C>T at 1000 and the G>A at 1200 can be deamination: ambiguous (-3)
    assert alt_on == '-3,0,-3,0,0'


def test_het_only_without_a_sample_is_refused(inputs):
    tmp_path, bam, vcf = inputs
    proc = subprocess.run([sys.executable, '-m', 'fiberhmm.cli.extract_tags', '-i', str(bam), '-o', str(tmp_path / 'x'),
                           '--variants', str(vcf), '--variants-het-only', '--bed-only'], capture_output=True, text=True)
    assert proc.returncode == 2 and '--variants-het-only needs --variants-sample' in proc.stderr


def test_sv_thresholds_from_the_command_line(inputs):
    tmp_path, bam, _vcf = inputs
    # the 150 bp soft clip at 3000, the 40 bp deletion at 3100, the 260 bp hard clip of the supplementary on chr2
    assert _extract(tmp_path, bam, '--sv', layer='sv')[1] == [500, 3000, 3100]
    assert _extract(tmp_path, bam, '--sv', '--sv-min-len', '50', layer='sv')[1] == [500, 3000]
    assert _extract(tmp_path, bam, '--sv', '--sv-min-clip', '200', layer='sv')[1] == [500, 3100]
