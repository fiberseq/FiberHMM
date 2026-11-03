"""DAF strand-swap marks and molecule-wide call preprocessing.

``xc:Z:strand_swap`` is a local SAM tag (lowercase namespace). Together
with 0x200 it identifies FiberHMM chimeras; unrelated QC failures are untouched.
Detection uses the caller's extraction and masking policy, not a second test.
"""
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import pysam

CHIMERA_TAG = 'xc'
CHIMERA_REASON = 'strand_swap'
CHIMERA_QC_TAG = 'xq'  # original QC-fail state before our mark (0 or 1)


def _has_chimera_tag(read):
    return read.has_tag(CHIMERA_TAG) and read.get_tag(CHIMERA_TAG) == CHIMERA_REASON


def is_marked_chimera(read):
    return (bool(getattr(read, 'is_qcfail', False))
            and _has_chimera_tag(read))


def clear_chimera_mark(read):
    """Restore QC-fail only when our exact mark records flag ownership.

    Older marks without ownership metadata cannot prove who set 0x200;
    conservatively preserve the flag in that case.
    """
    if _has_chimera_tag(read):
        if read.has_tag(CHIMERA_QC_TAG):
            if read.get_tag(CHIMERA_QC_TAG) == 0:
                read.is_qcfail = False
            read.set_tag(CHIMERA_QC_TAG, None)
        read.set_tag(CHIMERA_TAG, None)


def mark_chimera(read):
    # Repeated marking must not turn our own QC failure into a prior one.
    if not _has_chimera_tag(read):
        read.set_tag(CHIMERA_QC_TAG, int(read.is_qcfail), value_type='i')
    read.is_qcfail = True
    read.set_tag(CHIMERA_TAG, CHIMERA_REASON, value_type='Z')


def seekable_input(path):
    """Regular local inputs can be reopened; stdin and FIFOs need a spool."""
    return path != '-' and Path(path).is_file()


def _is_chimeric(read, ref):
    """Only the caller's chimera decision, without extracting call evidence."""
    from . import engine
    from fiberhmm.daf.encoder import get_daf_positions, is_daf_chimera

    seq = read.query_sequence
    if not seq:
        return False
    if engine.has_iupac_encoding(seq):
        return engine._is_iupac_daf_chimera(
            read, seq, engine._stub_or_live_excluded(read), ref)
    result = get_daf_positions(
        # Both arrays are populated regardless of the chosen strand. Force
        # collection past the dominant-strand tie check; the spatial test
        # below evaluates both CT->GA and GA->CT swaps.
        read, force_strand='CT', ref_fasta=ref,
        excluded_reference_positions=engine._daf_reference_mask(read))
    return result is not None and is_daf_chimera(
        result[0], result[1], min_seg_events=engine._DAF_CHIMERA_CFG['min_seg'],
        purity=engine._DAF_CHIMERA_CFG['purity'])


def empty_chimera_stats(mode, min_seg=5, purity=0.8):
    return dict(detected=False, mode=mode, n_reads_checked=0, n_chimeric=0,
                fraction=None, min_seg=min_seg, purity=purity,
                n_records_affected=0, n_stale_marks_cleared=0,
                full_run=True, count_unit='unique read names before removal')


def prepare_chimeras(input_bam, output_bam=None, *, mode='flag', daf=True,
                     min_seg=5, purity=0.8, prob_threshold=128,
                     reference=None, snp_mask=None, min_mapq=0,
                     min_read_length=0, process_unmapped=False,
                     io_threads=1, chimeric_names=None, spool_bam=None,
                     alignments='primary-supplementary'):
    """Decide QNAMEs in a read-only pass; spool only non-seekable inputs.

    The CLI passes a mutable ``chimeric_names`` set and applies marks/removal
    in the caller, so file input needs no intermediate BAM. For streams,
    ``spool_bam`` supplies the replay path. ``output_bam`` optionally writes
    a marked/filtered BAM for standalone use. Counts precede removal.
    """
    from . import engine
    from .read_filters import ReadFilterConfig, streaming_skip_reason

    if mode not in ('flag', 'remove', 'keep'):
        raise ValueError(f'Invalid chimera mode: {mode}')
    if min_seg < 1 or not 0.5 < purity <= 1:
        raise ValueError('chimera min_seg must be >= 1 and purity in (0.5, 1]')
    engine.configure_daf_chimera_filter(daf and mode != 'keep', min_seg, purity)
    engine.configure_daf_snp_mask(snp_mask)
    config = ReadFilterConfig(min_mapq=min_mapq, min_read_length=min_read_length,
                              primary_only=alignments,
                              process_unmapped=process_unmapped, mode='daf',
                              has_reference=reference is not None)
    checked, chimeric = set(), set()
    counts = Counter()
    stale = 0
    # The optional standalone writer owns its private spool. The CLI owns
    # its stream spool's lifetime, which extends through calling.
    private_spool = False
    if not seekable_input(input_bam) and spool_bam is None:
        if output_bam is None:
            raise ValueError('Non-seekable input requires spool_bam')
        spool_bam = str(Path(output_bam).with_suffix('.source.bam'))
        private_spool = True
    replay = spool_bam or input_bam
    try:
        with ExitStack() as stack:
            ref = stack.enter_context(pysam.FastaFile(reference)) if reference else None
            bam = stack.enter_context(pysam.AlignmentFile(
                input_bam, 'rb', check_sq=False, threads=io_threads))
            out = (stack.enter_context(pysam.AlignmentFile(
                spool_bam, 'wb', header=bam.header, threads=io_threads))
                   if spool_bam else None)
            for read in bam.fetch(until_eof=True):
                counts[read.query_name] += 1
                stale += int(_has_chimera_tag(read))
                clear_chimera_mark(read)
                if daf and mode != 'keep' and streaming_skip_reason(read, config) is None:
                    checked.add(read.query_name)
                    if read.query_name not in chimeric and _is_chimeric(read, ref):
                        chimeric.add(read.query_name)
                if out is not None:
                    out.write(read)
        if chimeric_names is not None:
            chimeric_names.update(chimeric)
        if output_bam is not None:
            with pysam.AlignmentFile(replay, 'rb', check_sq=False) as bam, \
                    pysam.AlignmentFile(output_bam, 'wb', header=bam.header,
                                        threads=io_threads) as out:
                for read in bam.fetch(until_eof=True):
                    clear_chimera_mark(read)
                    if read.query_name in chimeric:
                        if mode == 'remove':
                            continue
                        mark_chimera(read)
                    out.write(read)
    finally:
        if private_spool:
            Path(spool_bam).unlink(missing_ok=True)
    return dict(detected=bool(daf and mode != 'keep'), mode=mode if daf else 'n/a',
                n_reads_checked=len(checked), n_chimeric=len(chimeric),
                fraction=len(chimeric) / len(checked) if checked else None,
                min_seg=min_seg if daf else None, purity=purity if daf else None,
                n_records_affected=sum(counts[name] for name in chimeric),
                n_stale_marks_cleared=stale, full_run=True,
                count_unit='unique read names before removal')
