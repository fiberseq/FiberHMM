"""Multiprocessing worker entry points for region-parallel inference."""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np
import pysam

from fiberhmm.core.model_io import freeze_model_for_inference, load_model
from fiberhmm.inference.engine import (
    CHIMERA_SKIP,
    _extract_fiber_read_from_pysam,
    _process_single_read,
    configure_daf_chimera_filter,
    configure_daf_snp_mask,
    extract_fiber_read_from_payload,
    make_apply_payload,
)
from fiberhmm.inference.fused_stages import (
    apply_result_has_footprints,
    build_fused_recall_result,
    payload_cpg_mask,
    run_ddda_mcg_stage,
    run_hmm_apply_stage,
)
from fiberhmm.inference.read_filters import (
    ReadFilterConfig,
    new_skip_counts,
    streaming_skip_reason,
)
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
from fiberhmm.inference.worker_results import record_failure_message
from fiberhmm.io.bam_header import (
    append_coord_marker,
    append_ma_types,
)
from fiberhmm.posteriors.region_tsv import format_region_posterior_line

_worker_model = None
_worker_region_params = None
_worker_recall_state = {}

_REGION_NUC_PROFILE_CACHE: dict = {}


def _region_nuc_profile(path):
    """Load (and cache per worker) the DddA radial profile, or None."""
    if not path:
        return None
    if path not in _REGION_NUC_PROFILE_CACHE:
        from fiberhmm.inference.nuc_recaller import load_nuc_profile
        _REGION_NUC_PROFILE_CACHE[path] = load_nuc_profile(path)
    return _REGION_NUC_PROFILE_CACHE[path]


def _write_skipped_region_read(outbam, read, skip_reasons: dict, reason: str) -> int:
    """Pass through a skipped BAM read and count its reason."""
    outbam.write(read)
    skip_reasons[reason] += 1
    return 1


def _init_region_worker(model_path: str, params: dict):
    """Initialize worker for region-parallel processing."""
    global _worker_model, _worker_region_params
    import os

    try:
        # Disable numba caching to avoid file lock contention.
        os.environ['NUMBA_CACHE_DIR'] = ''

        # Load model once per worker.
        _worker_model = freeze_model_for_inference(load_model(model_path))
        _worker_region_params = params

        configure_daf_chimera_filter(
            params.get('filter_chimeras', True),
            params.get('chimera_min_seg', 5),
            params.get('chimera_purity', 0.8),
        )
        configure_daf_snp_mask(params.get('daf_snp_mask_path'))

        # Warm up numba JIT.
        from fiberhmm.core.hmm import HAS_NUMBA

        if HAS_NUMBA:
            dummy_obs = np.array([0, 1, 2, 3], dtype=np.int32)
            _ = _worker_model.predict(dummy_obs)

    except Exception as e:
        import traceback

        print(f"Region worker init error: {e}", file=sys.stderr)
        traceback.print_exc()
        raise


def _passthrough_region(inbam, outbam, work_item, skip_reasons, strip) -> int:
    """Copy one pass-through plan item (unselected contig or ``'*'``)."""
    chrom = work_item.region[0]
    written = 0
    for read in inbam.fetch(chrom):
        strip(read)
        outbam.write(read)
        written += 1
        reason = 'unmapped' if read.is_unmapped else 'contig_not_selected'
        skip_reasons[reason] += 1
    return written


def _region_read_owned(read, start: int, end: int) -> bool:
    """A region owns the reads that START in it (fetch returns overlaps)."""
    return start <= read.reference_start < end


def _process_region_to_bam(args: RegionBamWorkItem) -> RegionBamResult:
    """
    Worker function: process one genomic region and write to temp BAM.

    Each worker opens its own BAM file handle and uses the index to fetch
    reads from its assigned region. This enables true parallel I/O. Only
    reads that start inside the region are written (ownership is decided
    before any write), so records spanning a boundary appear exactly once.

    Uses global _worker_model and _worker_region_params (set by _init_region_worker).

    Args:
        args: RegionBamWorkItem, or the legacy tuple shape.

    Returns:
        RegionBamResult with temp BAM, counts, optional TSV path, and skip reasons.
    """
    import traceback

    from fiberhmm.inference.streaming_drain import strip_for_apply

    global _worker_model, _worker_region_params

    chrom, start, end = '?', 0, 0
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
        min_mapq = int(params['min_mapq'])
        prob_threshold = int(params['prob_threshold'])
        min_read_length = int(params['min_read_length'])
        with_scores = params['with_scores']
        train_rids = params['train_rids']
        primary_only = params.get('primary_only', False)
        return_posteriors = params.get('return_posteriors', False) and temp_tsv_path is not None
        write_msps = params.get('write_msps', True)
        io_threads = int(params.get('io_threads', 4))

        def strip(read):
            strip_for_apply(read, write_msps)

        total_reads = 0
        reads_with_footprints = 0
        written = 0
        skipped = 0
        posteriors_written = 0
        failures = 0
        failure_messages = []

        skip_reasons = new_skip_counts()
        filter_config = ReadFilterConfig(
            min_mapq=min_mapq,
            min_read_length=min_read_length,
            primary_only=primary_only,
            process_unmapped=False,
            train_rids=train_rids,
            mode=mode,
        )

        pysam.set_verbosity(0)

        # Open posteriors TSV file for streaming writes (if requested).
        tsv_file = None
        if return_posteriors and temp_tsv_path and not work_item.passthrough:
            try:
                tsv_file = open(temp_tsv_path, 'w')
            except Exception:
                return_posteriors = False  # Can't write, disable.

        try:
            with pysam.AlignmentFile(input_bam, "rb", threads=io_threads, check_sq=False) as inbam:
                with pysam.AlignmentFile(temp_bam_path, "wb",
                                         header=append_coord_marker(inbam.header),
                                         threads=io_threads) as outbam:

                    if work_item.passthrough:
                        written = _passthrough_region(
                            inbam, outbam, work_item, skip_reasons, strip)
                        return RegionBamResult(
                            temp_bam_path, 0, 0, written, None, skip_reasons,
                        )

                    for read in inbam.fetch(chrom, start, end):
                        # Ownership first: a boundary-spanning record is
                        # fetched by every region it overlaps.
                        if not _region_read_owned(read, start, end):
                            continue
                        strip(read)

                        skip_reason = streaming_skip_reason(read, filter_config)
                        if skip_reason:
                            written += _write_skipped_region_read(
                                outbam, read, skip_reasons, skip_reason
                            )
                            skipped += 1
                            continue

                        try:
                            fiber_read = _extract_fiber_read_from_pysam(read, mode, prob_threshold)
                            if fiber_read is CHIMERA_SKIP:
                                outbam.write(read)
                                written += 1
                                skipped += 1
                                skip_reasons['chimera'] += 1
                                continue
                            if fiber_read is None:
                                outbam.write(read)
                                written += 1
                                skipped += 1
                                skip_reasons['no_modifications'] += 1
                                continue
                        except Exception:
                            failures += 1
                            record_failure_message(failure_messages, read.query_name)
                            outbam.write(read)
                            written += 1
                            skipped += 1
                            skip_reasons['extraction_failed'] += 1
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

        metrics = {'worker_failures': failures} if failures else {}
        tsv_out = (
            temp_tsv_path
            if return_posteriors and posteriors_written > 0 and temp_tsv_path
            else None
        )
        return RegionBamResult(
            temp_bam_path, total_reads, reads_with_footprints,
            written, tsv_out, skip_reasons, metrics, tuple(failure_messages),
        )

    except Exception as e:
        print(f"\nWorker error in region {chrom}:{start}-{end}: {e}", file=sys.stderr)
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
        min_mapq = int(params['min_mapq'])
        prob_threshold = int(params['prob_threshold'])
        min_read_length = int(params['min_read_length'])
        with_scores = params['with_scores']
        train_rids = params['train_rids']
        io_threads = int(params.get('io_threads', 4))

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
                    if read.is_unmapped or read.is_secondary or read.is_supplementary:
                        continue

                    if read.reference_start < start or read.reference_start >= end:
                        continue

                    if read.mapping_quality < min_mapq:
                        continue

                    if read.query_alignment_length is None or read.query_alignment_length < min_read_length:
                        continue

                    read_id = read.query_name
                    if train_rids and read_id in train_rids:
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

        print(f"\nWorker error in region {chrom}:{start}-{end}: {e}", file=sys.stderr)
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
    import os

    os.environ['NUMBA_CACHE_DIR'] = ''

    _worker_recall_state = {}
    _worker_model = freeze_model_for_inference(load_model(apply_model_path))
    # Open the reference FASTA after fork: pysam.FastaFile is not fork-safe.
    ref_path = params.get('ref_fasta_path')
    if ref_path:
        import pysam as _pysam

        params = dict(params)   # don't mutate the shared-across-workers dict
        params['ref_fasta'] = _pysam.FastaFile(ref_path)
    _worker_region_params = params

    from fiberhmm.core.model_io import load_model_with_metadata
    from fiberhmm.inference.tf_recaller import (
        apply_emission_uplift,
        build_conditional_hit_tables,
        build_llr_tables,
        build_m5c_llr_tables,
    )

    r_path = recall_model_path or apply_model_path
    r_model, _, _ = load_model_with_metadata(r_path)
    llr_hit, llr_miss = build_llr_tables(r_model)
    if abs(emission_uplift - 1.0) > 1e-9:
        llr_hit, llr_miss = apply_emission_uplift(llr_hit, llr_miss, r_model, emission_uplift)
    _worker_recall_state['llr_hit'] = llr_hit
    _worker_recall_state['llr_miss'] = llr_miss
    n_model = r_model
    nuc_emission_uplift = emission_uplift
    if params.get('recall_nucs', False):
        n_path = params.get('nuc_model_path') or r_path
        nuc_emission_uplift = (
            1.0 if params.get('nuc_model_path') else emission_uplift
        )
        if n_path != r_path:
            n_model, _, _ = load_model_with_metadata(n_path)
        nuc_llr_hit, nuc_llr_miss = build_llr_tables(n_model)
        if abs(nuc_emission_uplift - 1.0) > 1e-9:
            nuc_llr_hit, nuc_llr_miss = apply_emission_uplift(
                nuc_llr_hit, nuc_llr_miss, n_model, nuc_emission_uplift,
            )
        _worker_recall_state['nuc_llr_hit'] = nuc_llr_hit
        _worker_recall_state['nuc_llr_miss'] = nuc_llr_miss
    nuc_profile_path = params.get('nuc_profile_path')
    if nuc_profile_path:
        from fiberhmm.inference.nuc_recaller import (
            attach_nuc_profile_emissions,
            load_nuc_profile,
        )
        protected_hit, accessible_hit = build_conditional_hit_tables(
            n_model, emission_uplift=nuc_emission_uplift,
        )
        _REGION_NUC_PROFILE_CACHE[nuc_profile_path] = (
            attach_nuc_profile_emissions(
                load_nuc_profile(nuc_profile_path),
                protected_hit,
                accessible_hit,
            )
        )
    _worker_recall_state['ddda_mcg'] = bool(params.get('ddda_mcg', False))
    _worker_recall_state['cpg_mask_policy'] = params.get('cpg_mask_policy')
    if _worker_recall_state['ddda_mcg'] or _worker_recall_state['cpg_mask_policy']:
        m5c_hit, m5c_miss = build_m5c_llr_tables(
            r_model, emission_uplift=emission_uplift,
        )
        _worker_recall_state['m5c_llr_hit'] = m5c_hit
        _worker_recall_state['m5c_llr_miss'] = m5c_miss
        if params.get('recall_nucs', False):
            nuc_m5c_hit, nuc_m5c_miss = build_m5c_llr_tables(
                n_model, emission_uplift=nuc_emission_uplift,
            )
            _worker_recall_state['nuc_m5c_llr_hit'] = nuc_m5c_hit
            _worker_recall_state['nuc_m5c_llr_miss'] = nuc_m5c_miss

    configure_daf_chimera_filter(
        params.get('filter_chimeras', True),
        params.get('chimera_min_seg', 5),
        params.get('chimera_purity', 0.8),
    )
    configure_daf_snp_mask(params.get('daf_snp_mask_path'))
    from fiberhmm.inference.engine import configure_daf_insert_evidence
    configure_daf_insert_evidence(params.get('daf_insert_evidence_path'))

    from fiberhmm.core.hmm import HAS_NUMBA

    if HAS_NUMBA:
        dummy_obs = np.array([0, 1, 2, 3], dtype=np.int32)
        _ = _worker_model.predict(dummy_obs)
        from fiberhmm.inference.tf_recaller import call_tfs_in_interval

        _ = call_tfs_in_interval(
            np.zeros(16, dtype=np.int32), 0, 16,
            llr_hit, llr_miss, min_llr=4.0, min_opps=3,
        )


def _process_region_to_bam_fused(args: RegionBamWorkItem) -> RegionBamResult:
    """Region worker: fetch reads in one genomic region, run fused
    apply+recall per read, write in-order to a coordinate-sorted temp BAM.

    Because pysam.fetch(chrom,start,end) yields reads in coordinate order
    AND we only write reads that START in this region (ownership is decided
    before any write), each temp BAM is coordinate-sorted within itself and
    boundary-spanning records appear once. Concatenating temp BAMs in region
    order gives a coordinate-sorted final BAM without any sort pass.

    Per-read exceptions pass the read through unannotated and are counted in
    ``metrics['worker_failures']`` with the first tracebacks, for the run's
    failure policy.

    Returns a RegionBamResult with temp BAM, counts, and skip reasons.
    """
    import traceback

    from fiberhmm.cli.provenance import output_header_with_provenance
    from fiberhmm.inference.streaming_drain import strip_for_fused_call

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
        min_mapq = int(params['min_mapq'])
        prob_threshold = int(params['prob_threshold'])
        min_read_length = int(params['min_read_length'])
        with_scores = params.get('with_scores', False)
        train_rids = params.get('train_rids') or set()
        primary_only = params.get('primary_only', False)
        io_threads = int(params.get('io_threads', 4))
        min_llr = float(params['min_llr'])
        min_opps = int(params['min_opps'])
        unify_threshold = int(params['unify_threshold'])
        also_write_legacy = params['also_write_legacy']
        downstream_compat = params['downstream_compat']
        recall_nucs = bool(params.get('recall_nucs', False))
        split_min_llr = float(params.get('split_min_llr', 4.0))
        split_min_opps = int(params.get('split_min_opps', 3))
        nuc_recall_policy = str(
            params.get('nuc_recall_policy', 'conservative'))
        phase_nrl = int(params.get('phase_nrl', 0))
        nuc_profile = _region_nuc_profile(params.get('nuc_profile_path'))
        derived_tf_max_edge_ambiguity = params.get(
            'derived_tf_max_edge_ambiguity')

        def strip(read):
            strip_for_fused_call(read, also_write_legacy)

        pysam.set_verbosity(0)

        total_reads = 0
        reads_with_fp = 0
        written = 0
        skipped = 0
        failure_messages = []
        skip_reasons = new_skip_counts()
        metrics = {
            'ddda_mcg_reads': 0,
            'ddda_mcg_spans': 0,
            'ddda_mcg_failures': 0,
            'worker_failures': 0,
        }
        filter_config = ReadFilterConfig(
            min_mapq=min_mapq,
            min_read_length=min_read_length,
            primary_only=primary_only,
            process_unmapped=False,
            train_rids=train_rids,
            mode=mode,
            has_reference=ref_fasta is not None,
        )

        def pass_through(read, reason=None):
            nonlocal written, skipped
            outbam.write(read)
            written += 1
            if reason is not None:
                skipped += 1
                skip_reasons[reason] += 1

        with pysam.AlignmentFile(input_bam, "rb", threads=io_threads, check_sq=False) as inbam:
            ma_types = [] if downstream_compat else ["nuc", "msp", "tf"]
            if params.get("ddda_mcg", False):
                ma_types.append("ddda_mcg")
            output_header = append_ma_types(
                output_header_with_provenance(inbam.header, params.get('pg_record')),
                ma_types,
            )
            with pysam.AlignmentFile(
                    temp_bam_path, "wb",
                    header=output_header,
                    threads=io_threads) as outbam:
                if work_item.passthrough:
                    written = _passthrough_region(
                        inbam, outbam, work_item, skip_reasons, strip)
                    return RegionBamResult(
                        temp_bam_path, 0, 0, written, None, skip_reasons,
                    )

                for read in inbam.fetch(chrom, start, end):
                    # Ownership first: a boundary-spanning record is fetched
                    # by every region it overlaps.
                    if not _region_read_owned(read, start, end):
                        continue
                    strip(read)
                    skip_reason = streaming_skip_reason(read, filter_config)
                    if skip_reason:
                        pass_through(read, skip_reason)
                        continue

                    payload = make_apply_payload(
                        read, mode=mode, ref_fasta=ref_fasta,
                        include_ddda_mcg=bool(params.get('ddda_mcg', False)),
                    )
                    if payload is None:
                        pass_through(read, 'no_modifications')
                        continue

                    try:
                        fiber_read = extract_fiber_read_from_payload(payload, mode, prob_threshold)
                        if fiber_read is CHIMERA_SKIP:
                            pass_through(read, 'chimera')
                            continue
                        if fiber_read is None:
                            pass_through(read, 'no_modifications')
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
                        metrics['worker_failures'] += 1
                        record_failure_message(failure_messages, read.query_name)
                        pass_through(read, 'extraction_failed')
                        continue

                    total_reads += 1

                    if not apply_result_has_footprints(apply_result):
                        pass_through(read)
                        skip_reasons['no_footprints'] += 1
                        continue

                    try:
                        m5c_mask = None
                        m5c_spans = []
                        m5c_failed = False
                        if _worker_recall_state.get('ddda_mcg'):
                            try:
                                m5c_mask, m5c_spans = run_ddda_mcg_stage(
                                    payload.get('_ddda_mcg_observations'),
                                    apply_result,
                                    len(fiber_read['query_sequence']),
                                )
                            except Exception:
                                m5c_mask, m5c_spans, m5c_failed = None, [], True
                        elif _worker_recall_state.get('cpg_mask_policy'):
                            m5c_mask = payload_cpg_mask(
                                payload, len(fiber_read['query_sequence']),
                                _worker_recall_state['cpg_mask_policy'],
                            )

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
                            nuc_recall_policy=nuc_recall_policy,
                            nuc_min_size=nuc_min_size,
                            msp_min_size=msp_min_size,
                            phase_nrl=phase_nrl,
                            nuc_profile=nuc_profile,
                            derived_tf_max_edge_ambiguity=(
                                derived_tf_max_edge_ambiguity),
                            m5c_mask=m5c_mask,
                            m5c_llr_hit=_worker_recall_state.get('m5c_llr_hit'),
                            m5c_llr_miss=_worker_recall_state.get('m5c_llr_miss'),
                            nuc_llr_hit=_worker_recall_state.get('nuc_llr_hit'),
                            nuc_llr_miss=_worker_recall_state.get('nuc_llr_miss'),
                            nuc_m5c_llr_hit=_worker_recall_state.get(
                                'nuc_m5c_llr_hit'),
                            nuc_m5c_llr_miss=_worker_recall_state.get(
                                'nuc_m5c_llr_miss'),
                        )
                        if _worker_recall_state.get('ddda_mcg'):
                            fused_result['ddda_mcg_spans'] = m5c_spans
                            fused_result['ddda_mcg_failed'] = m5c_failed
                        write_fused_recall_tags(
                            read,
                            read_length=len(fiber_read['query_sequence']),
                            result=fused_result,
                            also_write_legacy=also_write_legacy,
                            downstream_compat=downstream_compat,
                        )
                    except Exception:
                        # Same contract as the streaming worker: the read is
                        # passed through unannotated and the failure counted.
                        metrics['worker_failures'] += 1
                        record_failure_message(failure_messages, read.query_name)
                        strip(read)
                        pass_through(read)
                        continue
                    if _worker_recall_state.get('ddda_mcg'):
                        metrics['ddda_mcg_reads'] += int(bool(m5c_spans))
                        metrics['ddda_mcg_spans'] += len(m5c_spans)
                        metrics['ddda_mcg_failures'] += int(m5c_failed)
                    pass_through(read)
                    reads_with_fp += 1

        return RegionBamResult(
            temp_bam_path, total_reads, reads_with_fp,
            written, None, skip_reasons, metrics, tuple(failure_messages),
        )

    except Exception:
        traceback.print_exc()
        raise
