"""Region-parallel BAM and BED pipeline coordinators."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Set, Tuple

import pysam

from fiberhmm.inference.bam_output import (
    _concatenate_region_bams,
    _sort_and_index_bam,
    atomic_output,
    ensure_parent_dir,
)
from fiberhmm.inference.mp_context import _MP_CONTEXT
from fiberhmm.inference.region_planning import (
    _get_genome_regions,
    plan_region_work,
    require_indexed_bam,
)
from fiberhmm.inference.region_types import (
    RegionBamAggregation,
    RegionBamResult,
    RegionBamWorkItem,
    RegionBedAggregation,
    RegionBedResult,
    RegionBedWorkItem,
)
from fiberhmm.inference.region_workers import (
    _init_fused_region_worker,
    _init_region_worker,
    _process_region_to_bam,
    _process_region_to_bam_fused,
    _process_region_to_bed,
)
from fiberhmm.inference.worker_results import enforce_worker_failure_policy
from fiberhmm.posteriors.region_tsv import (
    merge_region_posteriors_tsv as _merge_region_posteriors_tsv,
)
from fiberhmm.posteriors.region_tsv import (
    region_posteriors_tsv_output_path,
)


def _plan_work_items(input_bam, temp_dir, region_size, skip_scaffolds, chroms,
                     with_tsv=False):
    """Region work items (processing + pass-through) in output order."""
    plan = plan_region_work(input_bam, region_size, skip_scaffolds, chroms)
    work_items = []
    for i, item in enumerate(plan):
        temp_bam = os.path.join(temp_dir, f'region_{i:06d}.bam')
        temp_tsv = (
            os.path.join(temp_dir, f'region_{i:06d}.tsv')
            if with_tsv and not item.passthrough else None
        )
        work_items.append(RegionBamWorkItem(
            item.region, input_bam, temp_bam, temp_tsv, item.passthrough,
        ))
    n_regions = sum(1 for item in plan if not item.passthrough)
    return work_items, n_regions


def _enforce_region_failures(aggregation) -> None:
    failures = int(aggregation.metrics.get('worker_failures', 0))
    enforce_worker_failure_policy(
        failures,
        aggregation.total_reads + failures,
        aggregation.failure_messages,
        log=sys.stdout,
        label='region worker',
    )


def _process_bam_region_parallel(input_bam: str, output_bam: str,
                                   model_path: str, train_rids: Set[str],
                                   edge_trim: int, circular: bool,
                                   mode: str, context_size: int,
                                   msp_min_size: int,
                                   nuc_min_size: int = 85,
                                   min_mapq: int = 0,
                                   prob_threshold: int = 0,
                                   min_read_length: int = 0,
                                   with_scores: bool = False,
                                   n_cores: int = 1,
                                   region_size: int = 10_000_000,
                                   skip_scaffolds: bool = False,
                                   chroms: Optional[Set[str]] = None,
                                   primary_only: bool = False,
                                   output_posteriors: Optional[str] = None,
                                   write_msps: bool = True,
                                   io_threads: int = 4) -> Tuple[int, int]:
    """
    Process BAM using region-based parallelism with indexed access.

    Each worker independently reads from the BAM using the index,
    enabling true parallel I/O. Results are concatenated at the end.

    Args:
        region_size: Size of each region in bp (default 10MB)
        skip_scaffolds: If True, skip scaffold/contig chromosomes
        chroms: If provided, only process these chromosomes
        primary_only: If True, skip secondary/supplementary alignments
        output_posteriors: If provided, write HMM posteriors to this H5 file
        write_msps: If True, write as/al/aq MSP tags to output BAM

    Returns:
        (total_reads_processed, reads_with_footprints)
    """
    start_time = time.time()
    return_posteriors = output_posteriors is not None

    # Check that BAM is indexed
    if not os.path.exists(input_bam + '.bai') and not os.path.exists(input_bam.replace('.bam', '.bai')):
        print("Indexing input BAM for region-parallel processing...")
        pysam.index(input_bam)
    require_indexed_bam(input_bam)

    print(f"Planning regions with {n_cores} cores...")
    if return_posteriors:
        print(f"Posteriors will be written to: {output_posteriors}")
    sys.stdout.flush()

    # Create temp directory in output folder for easier cleanup
    output_dir = ensure_parent_dir(output_bam)
    temp_dir = tempfile.mkdtemp(prefix='.fiberhmm_tmp_', dir=output_dir)

    try:
        # Prepare parameters (will be passed to initializer)
        params = {
            'edge_trim': edge_trim,
            'circular': circular,
            'mode': mode,
            'context_size': context_size,
            'msp_min_size': msp_min_size,
            'nuc_min_size': nuc_min_size,
            'min_mapq': min_mapq,
            'prob_threshold': prob_threshold,
            'min_read_length': min_read_length,
            'with_scores': with_scores,
            'train_rids': train_rids,
            'primary_only': primary_only,
            'return_posteriors': return_posteriors,
            'write_msps': write_msps,
            'io_threads': io_threads,
        }

        # Work items (processing regions plus pass-through contigs and the
        # unplaced unmapped reads) - include temp TSV path if posteriors
        # requested.
        work_items, n_regions = _plan_work_items(
            input_bam, temp_dir, region_size, skip_scaffolds, chroms,
            with_tsv=return_posteriors,
        )
        regions = work_items
        print(f"Processing {n_regions} regions "
              f"(+{len(work_items) - n_regions} pass-through)...")

        # Process regions in parallel
        aggregation = RegionBamAggregation()
        first_result_time = None

        # Use initializer to load model once per worker
        print(f"  Initializing {n_cores} worker processes (loading HMM model in each)...")
        sys.stdout.flush()
        pool_start = time.time()

        with ProcessPoolExecutor(
            max_workers=n_cores,
            mp_context=_MP_CONTEXT,
            initializer=_init_region_worker,
            initargs=(model_path, params)
        ) as executor:
            futures = {executor.submit(_process_region_to_bam, item): i
                      for i, item in enumerate(work_items)}

            for future in as_completed(futures):
                try:
                    result = RegionBamResult.from_value(future.result())
                    include_tsv = bool(
                        result.temp_tsv_path and os.path.exists(result.temp_tsv_path)
                    )
                    aggregation.add_result(futures[future], result, include_tsv=include_tsv)

                    # Track first result
                    if first_result_time is None:
                        first_result_time = time.time()
                        init_time = first_result_time - pool_start
                        print(f"  Workers ready ({init_time:.1f}s). Processing regions...")
                        sys.stdout.flush()

                    elapsed = time.time() - start_time
                    rate = aggregation.total_reads / elapsed if elapsed > 0 else 0
                    print(f"\r  Regions: {aggregation.completed}/{len(regions)} | "
                          f"Reads: {aggregation.total_reads:,} | "
                          f"With footprints: {aggregation.reads_with_footprints:,} | "
                          f"{rate:.1f} reads/s", end='')
                    sys.stdout.flush()

                except Exception as e:
                    print(f"\nError processing region: {e}")
                    raise

        print()  # Newline after progress

        # Print skip reasons summary
        if aggregation.total_skipped > 0:
            total_encountered = aggregation.total_reads + aggregation.total_skipped
            print(
                f"  Processed: {aggregation.total_reads:,} | "
                f"Skipped: {aggregation.total_skipped:,} | "
                f"With footprints: {aggregation.reads_with_footprints:,}"
            )
            print("  Skip reasons:")
            for reason, count in sorted(
                aggregation.skip_reasons.items(), key=lambda x: -x[1]
            ):
                if count > 0:
                    pct = 100 * count / total_encountered
                    print(f"    {reason}: {count:,} ({pct:.1f}%)")

        # Sort temp BAMs by region order and filter to non-empty
        aggregation.temp_bams.sort(key=lambda x: x[0])
        non_empty_bams = [bam for _, bam in aggregation.temp_bams
                         if os.path.exists(bam) and os.path.getsize(bam) > 0]

        _enforce_region_failures(aggregation)

        def _finalize(path):
            # Index (sorting first if needed) the closed temporary; the BAM
            # and its index are published together afterwards.
            sys.stdout.flush()
            output_size_gb = os.path.getsize(path) / (1024**3)
            print(f"Output BAM: {output_size_gb:.2f}GB")
            print("Step: Index/Sort...")
            _sort_and_index_bam(path, threads=n_cores)

        with atomic_output(output_bam, finalize=_finalize) as output_path:
            _concatenate_region_bams(input_bam, output_path, non_empty_bams,
                                     temp_dir)

        # Merge temp TSV files if posteriors were requested
        if return_posteriors and aggregation.temp_tsvs:
            print(f"Merging {len(aggregation.temp_tsvs)} posterior files...")
            merge_start = time.time()
            n_fibers = _merge_region_posteriors_tsv(
                aggregation.temp_tsvs, output_posteriors,
                mode, context_size, edge_trim, input_bam,
            )
            merge_time = time.time() - merge_start

            # Figure out actual output path (always .tsv.gz)
            tsv_path = region_posteriors_tsv_output_path(output_posteriors)

            if os.path.exists(tsv_path):
                file_size = os.path.getsize(tsv_path) / (1024 * 1024)
                print(
                    f"Posteriors: {n_fibers:,} fibers -> {tsv_path} "
                    f"({file_size:.1f} MB, {merge_time:.1f}s)"
                )
                if output_posteriors.endswith('.h5'):
                    print("  HDF5 output needs fiberhmm-posteriors (pip install "
                          "\"fiberhmm[posteriors]\"); the TSV above holds the same data.")

        elapsed = time.time() - start_time
        rate = aggregation.total_reads / elapsed if elapsed > 0 else 0
        print(
            f"Completed: {aggregation.total_reads:,} reads | "
            f"{aggregation.reads_with_footprints:,} with footprints | "
            f"{rate:.1f} reads/s | {elapsed:.1f}s"
        )

        return aggregation.total_reads, aggregation.reads_with_footprints

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def _process_bed_region_parallel(input_bam: str, output_bed: str,
                                  model_path: str, train_rids: Set[str],
                                  edge_trim: int, circular: bool,
                                  mode: str, context_size: int,
                                  msp_min_size: int,
                                  nuc_min_size: int = 85,
                                  min_mapq: int = 0,
                                  prob_threshold: int = 0,
                                  min_read_length: int = 0,
                                  with_scores: bool = False,
                                  n_cores: int = 1,
                                  region_size: int = 10_000_000,
                                  skip_scaffolds: bool = False,
                                  chroms: Optional[Set[str]] = None,
                                  primary_only: bool = False) -> Tuple[int, int]:
    """
    Process BAM using region-based parallelism, writing BED output directly.

    This is more space-efficient than processing to BAM first when only
    BED/bigBed output is needed - no large temp BAMs are created.

    Returns:
        (total_reads_processed, reads_with_footprints)
    """
    start_time = time.time()

    # Check that BAM is indexed
    if not os.path.exists(input_bam + '.bai') and not os.path.exists(input_bam.replace('.bam', '.bai')):
        print("Indexing input BAM for region-parallel processing...")
        pysam.index(input_bam)

    # Get regions
    regions = _get_genome_regions(input_bam, region_size, skip_scaffolds, chroms)
    print(f"Processing {len(regions)} regions with {n_cores} cores (BED output)...")
    sys.stdout.flush()

    # Create temp directory for BED files (small compared to BAMs)
    output_dir = ensure_parent_dir(output_bed)
    temp_dir = tempfile.mkdtemp(prefix='.fiberhmm_bed_tmp_', dir=output_dir)

    try:
        params = {
            'edge_trim': edge_trim,
            'circular': circular,
            'mode': mode,
            'context_size': context_size,
            'msp_min_size': msp_min_size,
            'nuc_min_size': nuc_min_size,
            'min_mapq': min_mapq,
            'prob_threshold': prob_threshold,
            'min_read_length': min_read_length,
            'with_scores': with_scores,
            'train_rids': train_rids,
            'primary_only': primary_only
        }

        # Work items - write temp BED files
        work_items = []
        for i, region in enumerate(regions):
            temp_bed = os.path.join(temp_dir, f'region_{i:06d}.bed')
            work_items.append(RegionBedWorkItem(region, input_bam, temp_bed))

        aggregation = RegionBedAggregation()
        first_result_time = None

        print(f"  Initializing {n_cores} worker processes (loading HMM model in each)...")
        sys.stdout.flush()
        pool_start = time.time()

        with ProcessPoolExecutor(
            max_workers=n_cores,
            mp_context=_MP_CONTEXT,
            initializer=_init_region_worker,
            initargs=(model_path, params)
        ) as executor:
            futures = {executor.submit(_process_region_to_bed, item): i
                      for i, item in enumerate(work_items)}

            for future in as_completed(futures):
                try:
                    result = RegionBedResult.from_value(future.result())
                    aggregation.add_result(futures[future], result)

                    # Track first result
                    if first_result_time is None:
                        first_result_time = time.time()
                        init_time = first_result_time - pool_start
                        print(f"  Workers ready ({init_time:.1f}s). Processing regions...")
                        sys.stdout.flush()

                    elapsed = time.time() - start_time
                    rate = aggregation.total_reads / elapsed if elapsed > 0 else 0
                    print(f"\r  Regions: {aggregation.completed}/{len(regions)} | "
                          f"Reads: {aggregation.total_reads:,} | "
                          f"With footprints: {aggregation.reads_with_footprints:,} | "
                          f"{rate:.1f} reads/s", end='')
                    sys.stdout.flush()

                except Exception as e:
                    print(f"\nError processing region: {e}")
                    raise

        print()  # Newline after progress

        # Sort temp BEDs by region order and concatenate
        aggregation.temp_beds.sort(key=lambda x: x[0])
        non_empty_beds = [bed for _, bed in aggregation.temp_beds
                         if os.path.exists(bed) and os.path.getsize(bed) > 0]

        print(f"Concatenating {len(non_empty_beds)} region BED files...")
        sys.stdout.flush()

        with open(output_bed, 'wb') as fout:
            for bed_path in non_empty_beds:
                with open(bed_path, 'rb') as fin:
                    shutil.copyfileobj(fin, fout)

        elapsed = time.time() - start_time
        rate = aggregation.total_reads / elapsed if elapsed > 0 else 0
        print(
            f"Completed: {aggregation.total_reads:,} reads | "
            f"{aggregation.reads_with_footprints:,} with footprints | "
            f"{rate:.1f} reads/s | {elapsed:.1f}s"
        )

        return aggregation.total_reads, aggregation.reads_with_footprints

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _process_bam_region_parallel_fused(
    input_bam: str, output_bam: str,
    apply_model_path: str, recall_model_path: Optional[str],
    train_rids: Set[str],
    edge_trim: int, circular: bool,
    mode: str, context_size: int,
    msp_min_size: int, nuc_min_size: int,
    min_mapq: int, prob_threshold: int, min_read_length: int,
    with_scores: bool,
    min_llr: float, min_opps: int, unify_threshold: int,
    emission_uplift: float,
    also_write_legacy: bool, downstream_compat: bool,
    n_cores: int, region_size: int, skip_scaffolds: bool,
    chroms: Optional[Set[str]], io_threads: int,
    primary_only: bool = False,
    ref_fasta_path: Optional[str] = None,
    recall_nucs: bool = False,
    split_min_llr: float = 4.0,
    split_min_opps: int = 3,
    nuc_recall_policy: str = "conservative",
    filter_chimeras: bool = True,
    chimera_min_seg: int = 5,
    chimera_purity: float = 0.8,
    phase_nrl: int = 0,
    nuc_profile_path: str = None,
    nuc_model_path: str = None,
    derived_tf_max_edge_ambiguity: int = None,
    pg_record: dict = None,
    ddda_mcg: bool = False,
    daf_snp_mask_path: str = None,
    cpg_mask_policy: Optional[str] = None,
    work_dir: Optional[str] = None,
    resume: bool = False,
    keep_work_dir: bool = False,
    run_identity: Optional[dict] = None,
    progress=None,
):
    """Region-parallel fused apply+recall.

    ``cpg_mask_policy`` enables DddA CpG-aware recall (None = off).

    Splits the BAM into genomic regions, runs fused apply+recall in each
    region as an independent worker, then concatenates sorted temp BAMs
    in region order.  Input BAM must be coordinate-sorted + indexed.

    Output is coordinate-sorted with no sort pass needed.

    ``work_dir`` makes the run resumable (see :mod:`region_resume`): finished
    region BAMs and their markers are kept there (also when the run fails or
    is interrupted), ``resume`` reuses the ones that validate against
    ``run_identity`` plus this function's own parameters, model digests and
    region plan, and the directory is removed after a successful publish
    unless ``keep_work_dir``. Without ``work_dir`` a private temporary
    directory is used and always removed. ``progress(event, **fields)``
    receives machine-readable progress events.
    """
    start_time = time.time()

    require_indexed_bam(input_bam)
    # Validate the plan (unknown --chroms, nothing left to process) before
    # creating any temporary state.
    plan = plan_region_work(input_bam, region_size, skip_scaffolds, chroms)
    emit = progress or (lambda event, **fields: None)
    # A missing output directory is created (as every writer does), for both the
    # resumable work directory and the private temporary directory below.
    ensure_parent_dir(output_bam)

    params = {
        'edge_trim': edge_trim, 'circular': circular,
        'mode': mode, 'context_size': context_size,
        'msp_min_size': msp_min_size, 'nuc_min_size': nuc_min_size,
        'min_mapq': min_mapq, 'prob_threshold': prob_threshold,
        'min_read_length': min_read_length, 'with_scores': with_scores,
        'train_rids': train_rids, 'primary_only': primary_only,
        'io_threads': io_threads,
        'min_llr': min_llr, 'min_opps': min_opps,
        'unify_threshold': unify_threshold,
        'also_write_legacy': also_write_legacy,
        'downstream_compat': downstream_compat,
        'recall_nucs': recall_nucs,
        'split_min_llr': split_min_llr,
        'split_min_opps': split_min_opps,
        'nuc_recall_policy': nuc_recall_policy,
        'filter_chimeras': filter_chimeras,
        'chimera_min_seg': chimera_min_seg,
        'chimera_purity': chimera_purity,
        'phase_nrl': phase_nrl,
        'nuc_profile_path': nuc_profile_path,
        'nuc_model_path': nuc_model_path,
        'derived_tf_max_edge_ambiguity': derived_tf_max_edge_ambiguity,
        'ddda_mcg': ddda_mcg,
        'cpg_mask_policy': cpg_mask_policy,
        'daf_snp_mask_path': daf_snp_mask_path,
        'pg_record': pg_record,
        # Path string, NOT an open handle: pysam.FastaFile is not fork-safe,
        # so each worker opens it lazily in _init_fused_region_worker.
        'ref_fasta_path': ref_fasta_path,
    }

    work = None
    published = False
    if work_dir is not None:
        from fiberhmm.inference.region_resume import (
            WorkDir, file_digest, reference_identity,
        )
        identity = dict(
            run_identity or {},
            region_pipeline=dict(
                {k: v for k, v in params.items()
                 if k not in ('pg_record', 'io_threads')},
                emission_uplift=emission_uplift,
                region_size=region_size, skip_scaffolds=skip_scaffolds,
                chroms=chroms,
            ),
            files=dict(
                apply_model=file_digest(apply_model_path),
                recall_model=file_digest(recall_model_path),
                nuc_profile=file_digest(nuc_profile_path),
                nuc_model=file_digest(nuc_model_path),
                daf_snp_mask=file_digest(daf_snp_mask_path),
                reference=reference_identity(ref_fasta_path),
            ),
            region_plan=[[list(item.region), bool(item.passthrough)]
                         for item in plan],
            pg_record={k: v for k, v in (pg_record or {}).items() if k != 'CL'},
        )
        work = WorkDir(work_dir, identity, pg_record, resume=resume).open()
        params['pg_record'] = work.pg_record
        temp_dir = str(work.path)
    else:
        output_dir = ensure_parent_dir(output_bam)
        temp_dir = tempfile.mkdtemp(prefix='.fiberhmm_call_tmp_', dir=output_dir)

    aggregation = RegionBamAggregation()
    regions = []
    try:
        work_items, n_regions = _plan_work_items(
            input_bam, temp_dir, region_size, skip_scaffolds, chroms,
        )
        regions = work_items
        reused = work.completed(work_items) if work is not None else {}
        for index, result in sorted(reused.items()):
            aggregation.add_result(index, result)
        pending = [(i, item) for i, item in enumerate(work_items) if i not in reused]
        # Remaining work in processed base pairs (pass-through items count as
        # nothing): a steadier ETA basis than the region count.
        def _bp(item):
            return 0 if item.passthrough else max(0, item.region[2] - item.region[1])
        pending_bp = sum(_bp(item) for _, item in pending)
        print(f"Processing {n_regions} regions "
              f"(+{len(work_items) - n_regions} pass-through) with {n_cores} "
              "cores (fused apply+recall)...")
        if reused:
            print(f"  Resuming: {len(reused)}/{len(work_items)} regions already "
                  f"finished in {temp_dir}; {len(pending)} to run.")
        sys.stdout.flush()
        emit('start', regions_total=len(work_items), regions_done=len(reused),
             regions_reused=len(reused), work_dir=temp_dir if work else None,
             output=os.path.abspath(output_bam))

        dispatch_start = time.time()
        new_reads = 0
        done_bp = 0
        if pending:
            print(f"  Initializing {n_cores} workers (loading apply model + LLR tables)...")
            sys.stdout.flush()
            pool_start = time.time()
            first_result = None
            initializer, initargs = _init_fused_region_worker, (
                apply_model_path, recall_model_path, emission_uplift, params)
            if work is not None:
                from fiberhmm.inference.region_resume import worker_initializer
                initializer, initargs = worker_initializer, (initializer, *initargs)
            executor = ProcessPoolExecutor(
                max_workers=n_cores,
                mp_context=_MP_CONTEXT,
                initializer=initializer,
                initargs=initargs,
            )
            try:
                futures = {executor.submit(_process_region_to_bam_fused, item): i
                           for i, item in pending}

                for future in as_completed(futures):
                    index = futures[future]
                    result = RegionBamResult.from_value(future.result())
                    if work is not None:
                        work.mark_done(index, work_items[index], result)
                    aggregation.add_result(index, result)
                    new_reads += result.total_reads
                    done_bp += _bp(work_items[index])
                    if first_result is None:
                        first_result = time.time()
                        print(f"  Workers ready ({first_result - pool_start:.1f}s). Processing...")
                        sys.stdout.flush()
                    elapsed = time.time() - dispatch_start
                    rate = new_reads / elapsed if elapsed > 0 else 0
                    eta = (elapsed * (pending_bp - done_bp) / done_bp
                           if done_bp > 0 else None)
                    print(f"\r  Regions: {aggregation.completed}/{len(regions)} | "
                          f"Reads: {aggregation.total_reads:,} | "
                          f"With FP: {aggregation.reads_with_footprints:,} | "
                          f"{rate:.0f} r/s"
                          f"{f' | ETA {eta / 60:.1f} min' if eta is not None and aggregation.completed < len(regions) else ''}",
                          end='')
                    sys.stdout.flush()
                    emit('region', region=list(work_items[index].region),
                         regions_done=aggregation.completed,
                         regions_total=len(regions), regions_reused=len(reused),
                         reads=aggregation.total_reads,
                         reads_with_footprints=aggregation.reads_with_footprints,
                         reads_per_s=round(rate, 1),
                         elapsed_s=round(elapsed, 1),
                         eta_s=round(eta, 1) if eta is not None else None)
            except BaseException:
                from fiberhmm.inference.region_resume import abort_executor
                abort_executor(executor)
                raise
            else:
                executor.shutdown(wait=True)
            print()

        if aggregation.total_skipped > 0:
            total_enc = aggregation.total_reads + aggregation.total_skipped
            print(
                f"  Processed: {aggregation.total_reads:,} | "
                f"Skipped: {aggregation.total_skipped:,} | "
                f"With FP: {aggregation.reads_with_footprints:,}"
            )
            print("  Skip reasons:")
            for reason, count in sorted(
                aggregation.skip_reasons.items(), key=lambda x: -x[1]
            ):
                if count > 0:
                    print(f"    {reason}: {count:,} ({100*count/total_enc:.1f}%)")

        if ddda_mcg:
            print(
                f"  DddA mCG: {aggregation.metrics.get('ddda_mcg_spans', 0):,} "
                f"spans on {aggregation.metrics.get('ddda_mcg_reads', 0):,} "
                f"reads; per-read failures="
                f"{aggregation.metrics.get('ddda_mcg_failures', 0):,}"
            )

        # Concat region BAMs in region-index order - preserves coord sort.
        # Published atomically, and only if the per-read failure policy holds.
        aggregation.temp_bams.sort(key=lambda x: x[0])
        non_empty = [bam for _, bam in aggregation.temp_bams
                     if os.path.exists(bam) and os.path.getsize(bam) > 0]
        _enforce_region_failures(aggregation)

        def _index_quietly(path):
            # Index the closed temporary directly (input sorted -> each region
            # sorted -> concat sorted); published together with the BAM.
            try:
                pysam.index(path)
            except pysam.SamtoolsError:
                pass

        emit('merge', regions_total=len(regions), bams=len(non_empty))
        with atomic_output(output_bam, finalize=_index_quietly) as output_path:
            _concatenate_region_bams(input_bam, output_path, non_empty, temp_dir)
        published = True

        elapsed = time.time() - start_time
        rate = aggregation.total_reads / elapsed if elapsed > 0 else 0
        print(
            f"  Total: {aggregation.total_reads:,} reads, "
            f"{aggregation.reads_with_footprints:,} with footprints, "
            f"{rate:.1f} r/s"
        )
        emit('done', output=os.path.abspath(output_bam),
             regions_total=len(regions), regions_reused=len(reused),
             reads=aggregation.total_reads,
             reads_with_footprints=aggregation.reads_with_footprints,
             elapsed_s=round(elapsed, 1))
        if work is not None and not keep_work_dir:
            work.remove()

    except BaseException as exc:
        if work is not None and not published:
            # Interrupted or failed: finished regions stay in the work dir.
            done, total = aggregation.completed, len(regions)
            from fiberhmm.inference.worker_results import WorkerFailureError
            if isinstance(exc, KeyboardInterrupt):
                advice = "Rerun the same command with --resume to continue."
            elif isinstance(exc, WorkerFailureError):
                # The failed reads are recorded in the finished regions, so a
                # resume would reuse them and fail the same way.
                advice = (f"The failed reads are part of the finished regions: "
                          f"delete {temp_dir} before rerunning.")
            else:
                advice = ("Fix the cause and rerun with --resume (finished regions "
                          f"are reused), or delete {temp_dir} to start over.")
            print(f"\n  Stopped ({type(exc).__name__}): {done}/{total} regions "
                  f"finished and kept in {temp_dir}; nothing was published. "
                  f"{advice}", file=sys.stderr)
            emit('stopped', reason=type(exc).__name__, regions_done=done,
                 regions_total=total, work_dir=temp_dir)
        raise
    finally:
        if work is None:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return aggregation.total_reads, aggregation.reads_with_footprints
