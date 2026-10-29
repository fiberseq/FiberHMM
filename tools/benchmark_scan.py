#!/usr/bin/env python3
"""Measure one scan module on genome-grid units covering supplied BAM slices.

Run each module in a fresh process to measure its RSS independently. BAMs are
pooled as one dataset, and read only. Outputs, profiles and timing records live
in the requested output directory (use a directory under /tmp for benchmarks).
The timed portion is the scan engine; input surveying to choose grid zones is
excluded. The engine and CLI use the same run_units entry point.
"""
import argparse
import cProfile
import json
from pathlib import Path
import resource
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bam', action='append', required=True)
    p.add_argument('--module', required=True, choices=['fp', 'nfr', 'peaks', 'nuc'])
    p.add_argument('--chemistry', default='hia5-pacbio')
    p.add_argument('--cores', type=int, default=4, choices=range(1, 5))
    p.add_argument('--annotate', action='store_true', help='Default: exact per-molecule tables')
    p.add_argument('-o', '--output', required=True)
    args = p.parse_args()
    import pysam
    from fiberhmm.scan.runner import run_units, resolve_datasets
    from fiberhmm.scan.units import plan
    datasets = [dict(dataset_id='demo', paths=args.bam, chemistry=args.chemistry)]
    zones, aligned = [], 0
    for path in args.bam:
        spans = {}
        with pysam.AlignmentFile(path, 'rb') as bam:
            for read in bam.fetch(until_eof=True):
                if read.is_unmapped:
                    continue
                c, a, b = read.reference_name, read.reference_start, read.reference_end
                lo, hi = spans.get(c, (a, b))
                spans[c] = min(a, lo), max(b, hi)
                if not read.is_secondary and not read.is_duplicate:
                    aligned += sum(e-s for s, e in read.get_blocks())
        zones.extend((c, a, b) for c, (a, b) in spans.items())
    _, reference = resolve_datasets(datasets)
    units = plan(reference, genome=True, zones=zones)
    bp = sum(u.zone_end-u.zone_start for u in units[0])
    out = Path(args.output)
    profile = cProfile.Profile()
    start = time.perf_counter()
    profile.enable()
    from fiberhmm.inference.consensus.execution import single_threaded_blas
    with single_threaded_blas():
        run_units(datasets, units, modules=[args.module], out=out, annotate=args.annotate, cores=args.cores)
    profile.disable()
    elapsed = time.perf_counter()-start
    profile.dump_stats(str(out / 'work' / 'benchmark.prof'))
    scale = 1 if sys.platform == 'darwin' else 1024
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*scale
    child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*scale
    attempt = json.loads((out / 'run.json').read_text())['attempts'][-1]
    calibration = sum(attempt['calibration_seconds'].values())
    # Standard per-process high-water RSS; the largest worker is separate
    # from the coordinator. This is not an aggregate simultaneous RSS.
    record = dict(module=args.module, seconds=elapsed, bp=bp, units=len(units[0]),
                  aligned_bases=aligned, depth=aligned/bp, cores=args.cores,
                  parent_peak_rss_bytes=own, largest_worker_peak_rss_bytes=child,
                  seconds_per_mb=elapsed/(bp/1e6),
                  calibration_seconds=calibration,
                  module_worker_seconds=attempt['module_unit_seconds'][args.module],
                  finalize_seconds=attempt['finalize_seconds'],
                  extrapolated_3gb_30x_seconds=calibration+(elapsed-calibration)*90e9/aligned)
    (out / 'work' / 'benchmark.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
