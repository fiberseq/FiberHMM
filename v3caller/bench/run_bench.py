"""Run the full nucleosome-caller benchmark across DddA datasets.

For each (dataset × algorithm × {alone, +poisson_merge}) triple, we:
  1. Load up to --max-reads from the BAM
  2. Build ReadBundle (opp/hit/baseline/ref_to_q) via unified extractor
  3. Run first pass → atoms
  4. Optionally apply caller_v7.poisson_merge_atoms
  5. Clip/filter by min_footprint, convert to query coords, compute MSPs
  6. Emit ns/nl/as/al to a caller-specific output BAM
  7. Aggregate metrics

Output layout:
    bench/output/
        bam/
            <dataset>__<caller>.bam     # tagged output
        summary.tsv                    # per (caller, dataset) metrics
        per_read_stats.tsv             # per-read lengths, nucs/read
        progress.log
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from collections import defaultdict

import numpy as np
import pysam

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # phase0

from bench.shared import (
    bundle_from_read, clip_and_filter_atoms, atoms_to_query_intervals,
    query_complement, write_calls_to_bam, BenchResult,
)
from bench.algorithms import ALGORITHMS
from caller_v7 import poisson_merge_atoms
from enzyme_extractors import DAFExtractor


# -------------------------------------------------------------------
# Dataset config
# -------------------------------------------------------------------

DATA_ROOT = os.path.dirname(HERE) + '/data/ddda'
DATASETS = [
    # name, bam_path, tier
    ('scDAF_PS00758',   '/Users/tt7739/Dropbox/Fiber-NET-seq/DAF-seq/Data/bam/PS00758_consensus_GA_HG38_corrected.haplotagged.bam', 'scDAF_whole_genome'),
    ('NAPA_PS00626',    '/Users/tt7739/Dropbox/Fiber-NET-seq/DAF-seq/Data/bam/for eitan/NAPA_PS00626_haplotype_corrected.bam', 'amplicon_showcase'),
    ('UBA1_PS00685',    '/Users/tt7739/Dropbox/Fiber-NET-seq/DAF-seq/Data/bam/for eitan/PS00685_GM12878_UBA1_map-pb_corrected_realigned_UBA1_region.bam', 'amplicon_showcase'),
    ('ENH30',           os.path.join(DATA_ROOT, 'PCR1_ENH30_DMSO16.mapped.consensus.decorated.bam'), 'amplicon_clean'),
    ('GLI2',            os.path.join(DATA_ROOT, 'encoded', 'GLI2.daf.bam'), 'amplicon_clean'),
    ('PS01498',         os.path.join(DATA_ROOT, 'PS01498.daf_encoded.bam'), 'amplicon_mouse'),
    ('PS01530',         os.path.join(DATA_ROOT, 'encoded', 'PS01530.daf.bam'), 'amplicon_multi'),
    ('ftz_22',          os.path.join(DATA_ROOT, 'embryo', 'ftz_22.daf.bam'), 'embryo_fly'),
    ('4RZV9P_6_eve_GA', os.path.join(DATA_ROOT, 'embryo', '4RZV9P_6_eve_GA.daf.bam'), 'embryo_fly'),
    ('4RZV9P_2_sna',    os.path.join(DATA_ROOT, 'embryo', '4RZV9P_2_sna.daf.bam'), 'embryo_fly'),
]


# -------------------------------------------------------------------
# Variant specs: which algorithms, which parameter sets
# -------------------------------------------------------------------

# Parameters tuned roughly to each caller's published defaults, but
# intentionally NOT over-tuned per-dataset so we see each one's raw
# behavior. Calibration sweeps come after the first pass.
ALG_PARAMS = {
    'gap_cdf':        {'gap_radius': 30},
    'xdrop':          {'miss_score': 1.0, 'hit_penalty': 5.0,
                       'x_drop': 10.0, 'min_score': 15.0},
    'hmm2':           {'emit_protected': 0.02, 'expect_prot_tokens': 73,
                       'expect_acc_tokens': 25},
    'core_peaks':     {'W': 40, 'core_ratio': 0.25, 'min_sep': 160,
                       'flank': 80},
    'protected_runs': {'W': 40},
    'profile_guided': {'scan_gap': 20, 'llr_threshold': 0.0},
}


MERGE_PARAMS = {
    'merge_alpha': 0.05,
    'gap_override_length': 30,
    'gap_override_opp': 10,
}

MIN_FOOTPRINT = 80


# -------------------------------------------------------------------
# Per-read caller wrapper
# -------------------------------------------------------------------

def call_bundle(bundle, alg_name, alg_params, apply_merge):
    fn = ALGORITHMS[alg_name]
    atoms = fn(bundle, **alg_params)
    if apply_merge and atoms:
        atoms = poisson_merge_atoms(
            atoms, bundle.opp, bundle.hit, bundle.baseline,
            MERGE_PARAMS['merge_alpha'],
            MERGE_PARAMS['gap_override_length'],
            MERGE_PARAMS['gap_override_opp'],
        )
    atoms = clip_and_filter_atoms(atoms, bundle.L, MIN_FOOTPRINT)
    if not atoms:
        return None
    qinter = atoms_to_query_intervals(atoms, bundle)
    if not qinter:
        return None
    ns = [qs for qs, _ in qinter]
    nl = [qe - qs for qs, qe in qinter]
    comp = query_complement(qinter, bundle.qlen)
    as_ = [qs for qs, _ in comp]
    al = [qe - qs for qs, qe in comp]
    return {'ns': ns, 'nl': nl, 'as': as_, 'al': al}


# -------------------------------------------------------------------
# Main run loop
# -------------------------------------------------------------------

def run_one(caller_tag, alg_name, apply_merge, ds_name, bam_path,
             out_dir, max_reads, extractor, progress_fh):
    t0 = time.time()
    result = BenchResult(caller=caller_tag, dataset=ds_name)
    calls_by_id = {}

    bam = pysam.AlignmentFile(bam_path, 'rb', check_sq=False)
    n_seen = 0
    for read in bam.fetch(until_eof=True):
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        n_seen += 1
        result.n_reads_total += 1
        if n_seen > max_reads:
            result.n_reads_total -= 1
            break
        try:
            bundle = bundle_from_read(read, extractor)
        except Exception:
            continue
        if bundle is None:
            continue
        try:
            call = call_bundle(bundle, alg_name,
                                ALG_PARAMS[alg_name], apply_merge)
        except Exception as e:
            # Algorithm crashed on this read — skip, continue
            continue
        if call is None:
            continue
        result.add_call(call['ns'], call['nl'], call['al'])
        calls_by_id[bundle.read_id] = call
    bam.close()

    # Write output BAM
    out_bam = os.path.join(out_dir, 'bam',
                            f'{ds_name}__{caller_tag}.bam')
    write_calls_to_bam(bam_path, out_bam, calls_by_id,
                        max_reads=max_reads)

    result.runtime_s = time.time() - t0
    msg = (f'  {caller_tag:28s} {ds_name:22s}  '
           f'called={result.n_reads_called}/{result.n_reads_total}  '
           f'median_nl={np.median(result.nl) if result.nl else 0:.0f}  '
           f'nucs/read={np.mean(result.nucs_per_read) if result.nucs_per_read else 0:.1f}  '
           f'{result.runtime_s:.1f}s')
    print(msg)
    progress_fh.write(msg + '\n')
    progress_fh.flush()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'output'))
    ap.add_argument('--max-reads', type=int, default=2000)
    ap.add_argument('--datasets', default=None,
                    help='comma-separated subset; default = all')
    ap.add_argument('--callers', default=None,
                    help='comma-separated subset; default = all')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, 'bam'), exist_ok=True)

    ds_subset = None
    if args.datasets:
        ds_subset = set(args.datasets.split(','))

    caller_subset = None
    if args.callers:
        caller_subset = set(args.callers.split(','))

    extractor = DAFExtractor()

    progress_path = os.path.join(args.out, 'progress.log')
    summary_path = os.path.join(args.out, 'summary.tsv')
    per_read_path = os.path.join(args.out, 'per_read_stats.tsv')
    nl_dump_path = os.path.join(args.out, 'nl_dump.json')

    all_results = []
    nl_dump = {}

    progress_fh = open(progress_path, 'w')
    progress_fh.write(f'# Benchmark started {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
    progress_fh.write(f'# max_reads={args.max_reads}\n')
    progress_fh.flush()

    # Enumerate (caller_tag, alg_name, apply_merge) combos
    combos = []
    for alg_name in ALGORITHMS:
        combos.append((f'{alg_name}', alg_name, False))
        combos.append((f'{alg_name}+merge', alg_name, True))

    if caller_subset:
        combos = [c for c in combos if c[0] in caller_subset]

    datasets = [d for d in DATASETS
                if ds_subset is None or d[0] in ds_subset]

    total = len(combos) * len(datasets)
    print(f'Benchmark: {len(combos)} callers × {len(datasets)} datasets = {total} runs')
    progress_fh.write(f'# {total} runs\n')

    run_idx = 0
    for ds_name, bam_path, tier in datasets:
        if not os.path.exists(bam_path):
            print(f'SKIP missing: {ds_name} {bam_path}')
            continue
        print(f'\n=== {ds_name} [{tier}] ===')
        progress_fh.write(f'\n=== {ds_name} [{tier}] ===\n')
        progress_fh.flush()
        for caller_tag, alg_name, apply_merge in combos:
            run_idx += 1
            try:
                r = run_one(caller_tag, alg_name, apply_merge,
                             ds_name, bam_path, args.out, args.max_reads,
                             extractor, progress_fh)
                all_results.append(r)
                nl_dump.setdefault(caller_tag, {})[ds_name] = r.nl
            except Exception as e:
                msg = f'  FAIL {caller_tag} {ds_name}: {e}'
                print(msg)
                progress_fh.write(msg + '\n')
                traceback.print_exc(file=progress_fh)
                progress_fh.flush()

    # Write summary TSV
    if all_results:
        summary_rows = [r.summary() for r in all_results]
        keys = list(summary_rows[0].keys())
        with open(summary_path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys, delimiter='\t')
            w.writeheader()
            w.writerows(summary_rows)
        print(f'\nWrote summary: {summary_path}')

        with open(nl_dump_path, 'w') as fh:
            json.dump(nl_dump, fh)
        print(f'Wrote nl dump: {nl_dump_path}')

    progress_fh.write(f'\n# Done {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
    progress_fh.close()


if __name__ == '__main__':
    main()
