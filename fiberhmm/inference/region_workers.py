"""Multiprocessing worker entry points for region-parallel inference."""

from __future__ import annotations

from typing import Optional

import pysam

from fiberhmm.core.model_io import freeze_model_for_inference, load_model
from fiberhmm.io.bam_header import append_coord_marker, maybe_append_pg
from fiberhmm.inference.engine import (
    CHIMERA_SKIP,
    _extract_fiber_read_from_pysam,
    _process_single_read,
    configure_daf_chimera_filter,
    extract_fiber_read_from_payload,
    make_apply_payload,
)
from fiberhmm.inference.fused_stages import (
    apply_result_has_footprints,
    build_fused_recall_result,
    run_hmm_apply_stage,
)
from fiberhmm.inference.read_filters import ReadFilterConfig, streaming_skip_reason
from fiberhmm.inference.region_types import (
    RegionBamResult,
    RegionBamWorkItem,
    RegionBedResult,
    RegionBedWorkItem,
)
from fiberhmm.inference.tagging import (
    set_legacy_apply_tags,
    write_fused_recall_tags,
)
from fiberhmm.inference.worker_warmup import (
    disable_numba_cache_locking,
    warm_up_model_predict,
    warm_up_tf_recaller,
)
from fiberhmm.posteriors.region_tsv import format_region_posterior_line

_worker_model = None
_worker_region_params = None
_worker_recall_state = {}

_PRE_OWNERSHIP_SKIP_REASONS = {"unmapped", "secondary_supplementary"}
_REGION_SKIP_REASON_KEYS = (
    'unmapped',
    'secondary_supplementary',
    'low_mapq',
    'too_short',
    'training_excluded',
    'no_modifications',
    'extraction_failed',
    'no_footprints',
    'chimera',
)


def _new_region_skip_reasons() -> dict:
    return {reason: 0 for reason in _REGION_SKIP_REASON_KEYS}


def _write_skipped_region_read(outbam, read, skip_reasons: dict, reason: str) -> int:
    """Pass through a skipped BAM read and count its reason."""
    outbam.write(read)
    skip_reasons[reason] += 1
    return 1


def _read_starts_in_region(read, start: int, end: int) -> bool:
    return int(start) <= int(read.reference_start) < int(end)


def _region_read_filter_config(params: dict, *, require_train_rids: bool) -> ReadFilterConfig:
    train_rids = (
        params['train_rids'] if require_train_rids
        else params.get('train_rids') or set()
    )
    return ReadFilterConfig(
        min_mapq=int(params['min_mapq']),
        min_read_length=int(params['min_read_length']),
        primary_only=params.get('primary_only', False),
        process_unmapped=False,
        train_rids=train_rids,
    )


def _init_region_worker(model_path: str, params: dict):
    """Initialize worker for region-parallel processing."""
    global _worker_model, _worker_region_params

    try:
        # Disable numba caching to avoid file lock contention.
        disable_numba_cache_locking()

        # Load model once per worker.
        _worker_model = freeze_model_for_inference(load_model(model_path))
        _worker_region_params = params

        configure_daf_chimera_filter(
            params.get('filter_chimeras', True),
            params.get('chimera_min_seg', 5),
            params.get('chimera_purity', 0.8),
        )

        # Warm up numba JIT.
        warm_up_model_predict(_worker_model)

    except Exception as e:
        import traceback

        print(f"Region worker init error: {e}")
        traceback.print_exc()
        raise


def _process_region_to_bam(args: RegionBamWorkItem) -> RegionBamResult:
    """
    Worker function: process one genomic region and write to temp BAM.

    Each worker opens its own BAM file handle and uses the index to fetch
    reads from its assigned region. This enables true parallel I/O.

    Uses global _worker_model and _worker_region_params (set by _init_region_worker).

    Args:
        args: RegionBamWorkItem, or the legacy tuple shape.

    Returns:
        RegionBamResult with temp BAM, counts, optional TSV path, and skip reasons.
    """
    import traceback

    global _worker_model, _worker_region_params

    try:
        work_item = RegionBamWorkItem.from_value(args)
        chrom, start, end = work_item.region
        input_bam = work_item.input_bam
        temp_bam_path = work_item.temp_bam_path
        temp_tsv_path = work_item.temp_tsv_path

        # Ensure start/end are Python ints (not numpy).
        start = int(start)
        end = int(end)

        # Use global model and params (loaded once per worker).
        model = _worker_model
        params = _worker_region_params

        # Unpack parameters.
        edge_trim = int(params['edge_trim'])
        circular = params['circular']
        mode = params['mode']
        context_size = int(params['context_size'])
        msp_min_size = int(params['msp_min_size'])
        nuc_min_size = int(params.get('nuc_min_size', 85))
        prob_threshold = int(params['prob_threshold'])
        with_scores = params['with_scores']
        return_posteriors = params.get('return_posteriors', False) and temp_tsv_path is not None
        write_msps = params.get('write_msps', True)
        io_threads = int(params.get('io_threads', 4))

        total_reads = 0
        reads_with_footprints = 0
        written = 0
        skipped = 0
        posteriors_written = 0

        skip_reasons = _new_region_skip_reasons()
        filter_config = _region_read_filter_config(params, require_train_rids=True)

        pysam.set_verbosity(0)

        # Open posteriors TSV file for streaming writes (if requested).
        tsv_file = None
        if return_posteriors and temp_tsv_path:
            try:
                tsv_file = open(temp_tsv_path, 'w')
            except Exception:
                return_posteriors = False  # Can't write, disable.

        try:
            with pysam.AlignmentFile(input_bam, "rb", threads=io_threads, check_sq=False) as inbam:
                with pysam.AlignmentFile(temp_bam_path, "wb",
                                         header=append_coord_marker(inbam.header),
                                         threads=io_threads) as outbam:

                    # Fetch reads from this region using the index.
                    try:
                        read_iter = inbam.fetch(chrom, start, end)
                    except ValueError:
                        # Region not in BAM (e.g., unplaced contigs).
                        if tsv_file:
                            tsv_file.close()
                        return RegionBamResult(temp_bam_path, 0, 0, 0)

                    for read in read_iter:
                        skip_reason = streaming_skip_reason(read, filter_config)
                        if skip_reason in _PRE_OWNERSHIP_SKIP_REASONS:
                            written += _write_skipped_region_read(
                                outbam, read, skip_reasons, skip_reason
                            )
                            skipped += 1
                            continue

                        # Only process reads that START in this region to avoid duplicates.
                        # fetch returns reads that overlap the region.
                        if not _read_starts_in_region(read, start, end):
                            continue

                        if skip_reason:
                            written += _write_skipped_region_read(
                                outbam, read, skip_reasons, skip_reason
                            )
                            skipped += 1
                            continue

                        try:
                            fiber_read = _extract_fiber_read_from_pysam(read, mode, prob_threshold)
                            if fiber_read is CHIMERA_SKIP:
                                written += _write_skipped_region_read(
                                    outbam, read, skip_reasons, 'chimera'
                                )
                                skipped += 1
                                continue
                            if fiber_read is None:
                                written += _write_skipped_region_read(
                                    outbam, read, skip_reasons, 'no_modifications'
                                )
                                skipped += 1
                                continue
                        except Exception:
                            written += _write_skipped_region_read(
                                outbam, read, skip_reasons, 'extraction_failed'
                            )
                            skipped += 1
                            continue

                        total_reads += 1

                        result = _process_single_read(
                            fiber_read, model, edge_trim, circular,
                            mode, context_size, msp_min_size, nuc_min_size=nuc_min_size,
                            with_scores=with_scores,
                            return_posteriors=return_posteriors,
                        )

                        if result is not None:
                            reads_with_footprints += 1

                            set_legacy_apply_tags(read, result, with_scores, write_msps)

                            # Stream posteriors to TSV immediately (no memory accumulation).
                            if tsv_file and result.get('posteriors') is not None:
                                try:
                                    tsv_file.write(
                                        format_region_posterior_line(
                                            read_name=read.query_name,
                                            chrom=read.reference_name,
                                            ref_start=read.reference_start,
                                            ref_end=read.reference_end,
                                            strand=result.get('strand', '.'),
                                            posteriors=result['posteriors'],
                                            footprint_starts=result['ns'],
                                            footprint_sizes=result['nl'],
                                        )
                                    )
                                    posteriors_written += 1
                                except Exception:
                                    pass  # Don't crash on posteriors write failure.
                        else:
                            skip_reasons['no_footprints'] += 1

                        outbam.write(read)
                        written += 1

        finally:
            if tsv_file:
                tsv_file.close()

        # Return TSV path if we wrote any posteriors.
        if return_posteriors and posteriors_written > 0 and temp_tsv_path:
            return RegionBamResult(
                temp_bam_path, total_reads, reads_with_footprints,
                written, temp_tsv_path, skip_reasons,
            )

        return RegionBamResult(
            temp_bam_path, total_reads, reads_with_footprints,
            written, None, skip_reasons,
        )

    except Exception as e:
        print(f"\nWorker error in region {chrom}:{start}-{end}: {e}")
        traceback.print_exc()
        raise


def _process_region_to_bed(args: RegionBedWorkItem) -> RegionBedResult:
    """
    Process a genomic region and write BED output directly (no temp BAM).

    This is more space-efficient than _process_region_to_bam when only
    BED/bigBed output is needed.

    Args:
        args: RegionBedWorkItem, or the legacy tuple shape.

    Returns:
        RegionBedResult with temp BED path and counts.
    """
    work_item = RegionBedWorkItem.from_value(args)
    region = work_item.region
    input_bam = work_item.input_bam
    temp_bed_path = work_item.temp_bed_path
    chrom, start, end = region

    try:
        start = int(start)
        end = int(end)

        model = _worker_model
        params = _worker_region_params

        edge_trim = int(params['edge_trim'])
        circular = params['circular']
        mode = params['mode']
        context_size = int(params['context_size'])
        msp_min_size = int(params['msp_min_size'])
        nuc_min_size = int(params.get('nuc_min_size', 85))
        prob_threshold = int(params['prob_threshold'])
        with_scores = params['with_scores']
        io_threads = int(params.get('io_threads', 4))
        filter_config = ReadFilterConfig(
            min_mapq=int(params['min_mapq']),
            min_read_length=int(params['min_read_length']),
            primary_only=True,
            process_unmapped=False,
            train_rids=params['train_rids'],
        )

        total_reads = 0
        reads_with_footprints = 0

        pysam.set_verbosity(0)

        with pysam.AlignmentFile(input_bam, "rb", threads=io_threads, check_sq=False) as inbam:
            with open(temp_bed_path, 'w') as bed_out:
                try:
                    read_iter = inbam.fetch(chrom, start, end)
                except ValueError:
                    return RegionBedResult(temp_bed_path, 0, 0)

                for read in read_iter:
                    if streaming_skip_reason(read, filter_config):
                        continue

                    if not _read_starts_in_region(read, start, end):
                        continue

                    try:
                        fiber_read = _extract_fiber_read_from_pysam(read, mode, prob_threshold)
                        if fiber_read is None or fiber_read is CHIMERA_SKIP:
                            continue
                    except Exception:
                        continue

                    total_reads += 1

                    result = _process_single_read(
                        fiber_read, model, edge_trim, circular,
                        mode, context_size, msp_min_size, nuc_min_size=nuc_min_size,
                        with_scores=with_scores,
                    )

                    if result is not None and len(result['ns']) > 0:
                        reads_with_footprints += 1

                        ref_name = read.reference_name
                        ref_start = read.reference_start
                        ref_end = read.reference_end
                        strand = '-' if read.is_reverse else '+'
                        read_id = read.query_name
                        read_length = ref_end - ref_start

                        ns = result['ns']
                        nl = result['nl']
                        block_starts_list = [int(s - ref_start) for s in ns]
                        block_sizes_list = [int(length) for length in nl]

                        score_list = None
                        if with_scores and result['ns_scores'] is not None:
                            score_list = [int(s * 1000) for s in result['ns_scores']]

                        # BED12 requires blocks to span chromStart to chromEnd.
                        if block_starts_list[0] != 0:
                            block_starts_list.insert(0, 0)
                            block_sizes_list.insert(0, 1)
                            if score_list is not None:
                                score_list.insert(0, 0)

                        last_end = block_starts_list[-1] + block_sizes_list[-1]
                        if last_end < read_length:
                            block_starts_list.append(read_length - 1)
                            block_sizes_list.append(1)
                            if score_list is not None:
                                score_list.append(0)

                        block_count = len(block_starts_list)
                        block_sizes = ','.join(str(s) for s in block_sizes_list)
                        block_starts = ','.join(str(s) for s in block_starts_list)

                        if score_list is not None:
                            scores = ','.join(str(s) for s in score_list)
                            bed_out.write(f"{ref_name}\t{ref_start}\t{ref_end}\t{read_id}\t0\t{strand}\t"
                                        f"{ref_start}\t{ref_end}\t0,0,0\t{block_count}\t{block_sizes}\t{block_starts}\t{scores}\n")
                        else:
                            bed_out.write(f"{ref_name}\t{ref_start}\t{ref_end}\t{read_id}\t0\t{strand}\t"
                                        f"{ref_start}\t{ref_end}\t0,0,0\t{block_count}\t{block_sizes}\t{block_starts}\n")

        return RegionBedResult(temp_bed_path, total_reads, reads_with_footprints)

    except Exception as e:
        import traceback

        print(f"\nWorker error in region {chrom}:{start}-{end}: {e}")
        traceback.print_exc()
        raise


def _init_fused_region_worker(
    apply_model_path: str,
    recall_model_path: Optional[str],
    emission_uplift: float,
    params: dict,
):
    """Per-worker init for region-parallel fused apply+recall.

    Loads the apply HMM model, builds the TF LLR tables (from the recall
    model or by reusing the apply model), warms up numba JIT, and stashes
    params for the region worker to pick up.
    """
    global _worker_model, _worker_region_params, _worker_recall_state

    disable_numba_cache_locking()

    _worker_model = freeze_model_for_inference(load_model(apply_model_path))
    # Open the reference FASTA after fork: pysam.FastaFile is not fork-safe.
    ref_path = params.get('ref_fasta_path')
    if ref_path:
        import pysam as _pysam

        params = dict(params)   # don't mutate the shared-across-workers dict
        params['ref_fasta'] = _pysam.FastaFile(ref_path)
    _worker_region_params = params

    from fiberhmm.core.model_io import load_model_with_metadata
    from fiberhmm.inference.tf_recaller import apply_emission_uplift, build_llr_tables

    r_path = recall_model_path or apply_model_path
    r_model, _, _ = load_model_with_metadata(r_path)
    llr_hit, llr_miss = build_llr_tables(r_model)
    if abs(emission_uplift - 1.0) > 1e-9:
        llr_hit, llr_miss = apply_emission_uplift(llr_hit, llr_miss, r_model, emission_uplift)
    _worker_recall_state['llr_hit'] = llr_hit
    _worker_recall_state['llr_miss'] = llr_miss

    configure_daf_chimera_filter(
        params.get('filter_chimeras', True),
        params.get('chimera_min_seg', 5),
        params.get('chimera_purity', 0.8),
    )

    warm_up_model_predict(_worker_model)
    warm_up_tf_recaller(llr_hit, llr_miss)


def _process_region_to_bam_fused(args: RegionBamWorkItem) -> RegionBamResult:
    """Region worker: fetch reads in one genomic region, run fused
    apply+recall per read, write in-order to a coordinate-sorted temp BAM.

    Because pysam.fetch(chrom,start,end) yields reads in coordinate order
    AND we only process reads that START in this region (the reference_start
    filter), each temp BAM is coordinate-sorted within itself. Concatenating
    temp BAMs in region order gives a coordinate-sorted final BAM without
    any sort pass.

    Returns a RegionBamResult with temp BAM, counts, and skip reasons.
    """
    import traceback

    global _worker_model, _worker_region_params, _worker_recall_state

    try:
        work_item = RegionBamWorkItem.from_value(args)
        chrom, start, end = work_item.region
        input_bam = work_item.input_bam
        temp_bam_path = work_item.temp_bam_path
        start = int(start)
        end = int(end)

        params = _worker_region_params
        model = _worker_model
        llr_hit = _worker_recall_state['llr_hit']
        llr_miss = _worker_recall_state['llr_miss']

        edge_trim = int(params['edge_trim'])
        circular = params['circular']
        mode = params['mode']
        ref_fasta = params.get('ref_fasta')
        context_size = int(params['context_size'])
        msp_min_size = int(params['msp_min_size'])
        nuc_min_size = int(params.get('nuc_min_size', 85))
        prob_threshold = int(params['prob_threshold'])
        with_scores = params.get('with_scores', False)
        io_threads = int(params.get('io_threads', 4))
        min_llr = float(params['min_llr'])
        min_opps = int(params['min_opps'])
        unify_threshold = int(params['unify_threshold'])
        also_write_legacy = params['also_write_legacy']
        downstream_compat = params['downstream_compat']
        recall_nucs = bool(params.get('recall_nucs', False))
        split_min_llr = float(params.get('split_min_llr', 4.0))
        split_min_opps = int(params.get('split_min_opps', 3))
        phase_nrl = int(params.get('phase_nrl', 0))

        pysam.set_verbosity(0)

        total_reads = 0
        reads_with_fp = 0
        written = 0
        skipped = 0
        skip_reasons = _new_region_skip_reasons()
        filter_config = _region_read_filter_config(params, require_train_rids=False)

        with pysam.AlignmentFile(input_bam, "rb", threads=io_threads, check_sq=False) as inbam:
            with pysam.AlignmentFile(
                    temp_bam_path, "wb",
                    header=maybe_append_pg(inbam.header, params.get('pg_record')),
                    threads=io_threads) as outbam:
                try:
                    read_iter = inbam.fetch(chrom, start, end)
                except ValueError:
                    return RegionBamResult(temp_bam_path, 0, 0, 0)

                for read in read_iter:
                    skip_reason = streaming_skip_reason(read, filter_config)
                    if skip_reason in _PRE_OWNERSHIP_SKIP_REASONS:
                        written += _write_skipped_region_read(
                            outbam, read, skip_reasons, skip_reason
                        )
                        skipped += 1
                        continue
                    if not _read_starts_in_region(read, start, end):
                        continue
                    if skip_reason:
                        written += _write_skipped_region_read(
                            outbam, read, skip_reasons, skip_reason
                        )
                        skipped += 1
                        continue

                    payload = make_apply_payload(read, mode=mode, ref_fasta=ref_fasta)
                    if payload is None:
                        written += _write_skipped_region_read(
                            outbam, read, skip_reasons, 'no_modifications'
                        )
                        skipped += 1
                        continue

                    try:
                        fiber_read = extract_fiber_read_from_payload(payload, mode, prob_threshold)
                        if fiber_read is CHIMERA_SKIP:
                            written += _write_skipped_region_read(
                                outbam, read, skip_reasons, 'chimera'
                            )
                            skipped += 1
                            continue
                        if fiber_read is None:
                            written += _write_skipped_region_read(
                                outbam, read, skip_reasons, 'no_modifications'
                            )
                            skipped += 1
                            continue
                        apply_result = run_hmm_apply_stage(
                            fiber_read,
                            model,
                            edge_trim,
                            circular,
                            mode,
                            context_size,
                            msp_min_size,
                            nuc_min_size,
                            with_scores,
                        )
                    except Exception:
                        written += _write_skipped_region_read(
                            outbam, read, skip_reasons, 'extraction_failed'
                        )
                        skipped += 1
                        continue

                    total_reads += 1

                    if not apply_result_has_footprints(apply_result):
                        outbam.write(read)
                        written += 1
                        skip_reasons['no_footprints'] += 1
                        continue

                    fused_result = build_fused_recall_result(
                        fiber_read,
                        apply_result,
                        llr_hit,
                        llr_miss,
                        min_llr,
                        min_opps,
                        unify_threshold,
                        with_scores,
                        recall_nucs=recall_nucs,
                        split_min_llr=split_min_llr,
                        split_min_opps=split_min_opps,
                        nuc_min_size=nuc_min_size,
                        msp_min_size=msp_min_size,
                        phase_nrl=phase_nrl,
                    )
                    write_fused_recall_tags(
                        read,
                        read_length=len(fiber_read['query_sequence']),
                        result=fused_result,
                        also_write_legacy=also_write_legacy,
                        downstream_compat=downstream_compat,
                    )
                    outbam.write(read)
                    written += 1
                    reads_with_fp += 1

        return RegionBamResult(
            temp_bam_path, total_reads, reads_with_fp,
            written, None, skip_reasons,
        )

    except Exception:
        traceback.print_exc()
        raise
