"""Re-run v8 (both first-pass options) on all 10 datasets and write
output BAMs with BOTH legacy and MA tags.

Iter-16: delegates to caller_v8.call_read directly so this driver
always matches the CLI behavior — new quality scores (lq/rq/tq/el/er),
TF calls (tn/tl/tq/el/er), and MA tags (nuc+QQQQ, msp+, tf+QQQ) are
all included for free.

Outputs:
    bench/output/bam/<ds>__v8.bam           (protected_runs + v8 merge)
    bench/output/bam/<ds>__v8_gapcdf.bam    (gap_cdf + v8 merge)
"""

from __future__ import annotations

import os
import sys
import time

import pysam

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from bench.run_bench import DATASETS, MIN_FOOTPRINT
from caller_v8 import (
    call_read, clear_all_call_tags, set_array_tag, MA_TAGS, NEW_LEGACY_TAGS,
)
from caller_v7 import STALE_CALL_TAGS
from ma_tags import format_ma_tag, format_aq_array
from enzyme_extractors import DAFExtractor


MAX_READS = 2000


def run_dataset(bam_path, first_pass, out_bam_path):
    extractor = DAFExtractor()
    src = pysam.AlignmentFile(bam_path, 'rb', check_sq=False)
    dst = pysam.AlignmentFile(out_bam_path, 'wb', template=src)
    n = 0
    n_called = 0
    for read in src.fetch(until_eof=True):
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            dst.write(read)
            continue
        n += 1
        if n > MAX_READS:
            break
        # Reconstruct ref_seq from MD tags (avoids external FASTA)
        try:
            ref_seq = read.get_reference_sequence().upper()
        except Exception:
            clear_all_call_tags(read)
            dst.write(read)
            continue
        if not ref_seq or len(ref_seq) != read.reference_end - read.reference_start:
            clear_all_call_tags(read)
            dst.write(read)
            continue
        try:
            result = call_read(
                read, ref_seq, extractor,
                W=40, min_read_rate=0.05, merge_alpha=0.10,
                max_merge_len=250, min_footprint=MIN_FOOTPRINT,
                first_pass=first_pass,
            )
        except Exception as e:
            print(f'  call_read failed on {read.query_name}: {e}',
                  file=sys.stderr)
            clear_all_call_tags(read)
            dst.write(read)
            continue

        clear_all_call_tags(read)

        if result is None:
            dst.write(read)
            continue

        # Legacy tags
        set_array_tag(read, 'ns', result['ns'])
        set_array_tag(read, 'nl', result['nl'])
        set_array_tag(read, 'nq', result['nq'])
        set_array_tag(read, 'lq', result['lq'])
        set_array_tag(read, 'rq', result['rq'])
        set_array_tag(read, 'as', result['as'])
        set_array_tag(read, 'al', result['al'])
        if result['tn']:
            set_array_tag(read, 'tn', result['tn'])
            set_array_tag(read, 'tl', result['tl'])
            set_array_tag(read, 'tq', result['tq'])
            set_array_tag(read, 'el', result['el'])
            set_array_tag(read, 'er', result['er'])

        # MA/AQ tags
        nuc_intervals = list(zip(result['ns'], result['nl']))
        msp_intervals = list(zip(result['as'], result['al']))
        tf_intervals = list(zip(result['tn'], result['tl']))
        ma_str = format_ma_tag(
            result['qlen'],
            nuc_intervals, msp_intervals,
            tf_intervals=tf_intervals,
            nuc_qual_spec='QQQQ',
            tf_qual_spec='QQQ',
        )
        aq_arr = format_aq_array(
            result['nq'], result['mq'],
            lq_values=result['lq'], rq_values=result['rq'],
            tf_q_values=result['tq'],
            tf_lq_values=result['el'],
            tf_rq_values=result['er'],
        )
        read.set_tag('MA', ma_str, value_type='Z')
        if len(aq_arr) > 0:
            read.set_tag('AQ', aq_arr)

        n_called += 1
        dst.write(read)
    src.close()
    dst.close()
    return n, n_called


def main():
    out_dir = os.path.join(HERE, 'output', 'bam')
    os.makedirs(out_dir, exist_ok=True)

    for ds_name, bam_path, tier in DATASETS:
        if not os.path.exists(bam_path):
            print(f'  skip {ds_name}: BAM not found')
            continue
        for fp_name, fp_tag in [('protected_runs', 'v8'),
                                 ('gap_cdf', 'v8_gapcdf')]:
            t0 = time.time()
            out_bam = os.path.join(out_dir, f'{ds_name}__{fp_tag}.bam')
            n, n_called = run_dataset(bam_path, fp_name, out_bam)
            print(f'  {fp_tag:10s} {ds_name:22s}  '
                  f'{n_called}/{n} called  '
                  f'{time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
