"""Native observation transport; no assay-average replacement emissions."""
from __future__ import annotations
import numpy as np
from .. import strand_boundary_normalization as sbn
from ..tf_recaller import build_conditional_hit_tables
from .artifacts import digest
from .geometry import merged


def read_adapter(unit, hits=None):
    lo, hi = unit['_region']
    calls = unit.get('native_multi_interval_tf_intervals', unit['representative_raw_tf_intervals'])
    return sbn.BoundaryRead(unit['unit_id'], unit['strand'], unit['positions'],
        unit['hits'] if hits is None else hits, unit['p_accessible'], unit['p_protected'],
        [tuple(v) for v in calls if v[0] < hi and v[1] > lo],
        unit['raw_nuc_intervals'], unit['aligned_blocks'])


def unit_identity(read, dataset_id, dataset_ordinal=None):
    """The content that names an evidence unit (``unit_id`` and ``fold_group_id``).

    With ``dataset_ordinal`` (the dataset's position in the run) the identity is
    the dataset ordinal, the file's index within that dataset's paths, and the
    record itself: the same BAM bytes give the same units wherever the file
    lives and whatever the dataset is called. unit_id orders the units fed to
    class discovery and keys its folds, so a path or label in it made the
    classes depend on where the BAM was opened from. Without an ordinal (callers
    that predate it) the dataset ID and library ID are kept as before."""
    record = [str(read.name), str(read.strand), str(read.record_sha256 or ''), int(read.alignment_occurrence)]
    if dataset_ordinal is None:
        return [dataset_id, str(read.library_id or ''), *record]
    return [int(dataset_ordinal), int(read.input_index), *record]


def evidence_unit(read, model, dataset_id, source_members, start, end, *, dataset_ordinal=None):
    """Convert a production ReadEvidence and its same-strand duplicate aliases."""
    pp, pa = build_conditional_hit_tables(model)
    pos = np.asarray(read.positions, dtype=np.int64)
    context = np.asarray(read.contexts, dtype=np.int64)
    hit = np.asarray(read.hits, dtype=np.int8)
    keep = (pos >= start) & (pos < end)
    pos, context, hit = pos[keep], context[keep], hit[keep]
    if np.any((context < 0) | (context >= len(pa))):
        raise ValueError('Native model does not cover the observed context encoding')
    members = sorted(source_members, key=lambda v: (v.get('read_name', ''), v.get('strand', '')))
    uid = 'unit_' + digest(unit_identity(read, dataset_id, dataset_ordinal))[:24]
    spans = lambda items: [[int(c.start), int(c.end)] for c in items]
    return dict(unit_id=uid, fold_group_id=uid, read_name=str(read.name), strand=str(read.strand),
        positions=pos.tolist(), hits=hit.tolist(), contexts=context.tolist(),
        p_accessible=pa[context].tolist(), p_protected=pp[context].tolist(),
        representative_raw_tf_intervals=spans(read.tfs), raw_tf_intervals=spans(read.tfs),
        raw_nuc_intervals=spans(read.nucs), msp_intervals=spans(read.msps),
        aligned_blocks=[list(v) for v in (read.alignment_blocks or [(read.ref_start, read.ref_end)])],
        reference_start=int(read.ref_start), reference_end=int(read.ref_end),
        alignment_orientation='reverse' if int(read.alignment_flag) & 16 else 'forward',
        source_members=members, _region=[start, end],
        physical_source_names=list(read.duplex_sources),
        pairing_method=read.pairing_method,pairing_model=read.pairing_model,
        provenance=dict(representative_record_sha256=read.record_sha256,
            alignment_occurrence=int(read.alignment_occurrence), native_emissions_unchanged=True,
            complementary_strands_paired=bool(read.duplex_sources)))


def m5c_query_mask(read, length):
    """The same SEQ-frame DddA mCG annotation used by production TF recall."""
    mask=np.zeros(length,dtype=bool)
    if read.has_tag('MA'):
        from ...daf.m5c import ma_intervals
        from ...io.ma_tags import DDDA_MCG_FEATURE
        for a,b in ma_intervals(read,DDDA_MCG_FEATURE):
            mask[max(0,a):min(length,b)]=True
    return mask


def condition_unit_on_m5c(read,unit):
    """Remove methylated-island CpGs from the native opportunity lattice."""
    from ..strand_rescue import cigar_to_query_ref
    refs=np.asarray(cigar_to_query_ref(read))
    mask=m5c_query_mask(read,len(refs))
    cpg=((np.asarray(unit['contexts'])%64)//16)==3
    observed_mask=np.isin(unit['positions'],refs[mask & (refs>=0)]) & cpg
    if observed_mask.any():
        keep=~observed_mask
        for key in ('positions','hits','contexts','p_accessible','p_protected'):
            if key in unit:
                unit[key]=np.asarray(unit[key])[keep].tolist()
        unit.pop('m5c_observations',None)
        unit['provenance']['native_m5c_excluded_opportunities']=int(observed_mask.sum())
        unit['provenance']['native_emissions_unchanged']=True


def reference_gap_inside(lo, hi, domains):
    """Reference length inside [lo, hi) that no aligned block covers: deletions and skips."""
    covered = 0
    for a, b in domains:
        covered += max(0, min(hi, b)-max(lo, a))
    return int(hi-lo)-int(covered)


def replay_alignment(read, unit, model, strand_mode, mode, context_size,
                     probability_threshold, minimum_llr, minimum_opportunities=3,
                     minimum_nfr_length=0, use_m5c=False, maximum_alignment_gap_bp=0):
    """Replay the installed corrected decoder on the ACTUAL query observations.

    No reconstruction of insertions or missing observations from reference bases.
    MSPs, nucleosomes and alignment are frozen. This is not an upstream HMM refit.
    """
    from ..strand_rescue import hard_observations, cigar_to_query_ref
    from ..tf_recaller import build_llr_tables, build_m5c_llr_tables, call_tfs_in_interval
    obs, _ = hard_observations(read, strand_mode, mode, context_size, probability_threshold)
    obs = np.asarray(obs, dtype=np.int32)
    refs = np.asarray(cigar_to_query_ref(read))
    # The actual query encoder and CIGAR mapper can expose different lengths.
    # Never hand a refs-derived out-of-bounds interval to numba.
    usable=min(len(obs),len(refs));obs=obs[:usable];refs=refs[:usable]
    hit, miss = build_llr_tables(model)
    mask=m5c_query_mask(read,usable) if use_m5c else None
    m5c_hit,m5c_miss=build_m5c_llr_tables(model) if mask is not None else (None,None)
    # An insertion may split get_blocks() into adjacent reference intervals.
    # Their union is continuous reference coverage, not an unobserved gap.
    # Merge adjacency only: deletions/skips still separate these domains.
    reference_domains = merged(unit['aligned_blocks'])
    # Query spans production calling leaves uncalled (long DAF insertions and
    # clips without consensus evidence; a supplementary record's clips): a
    # TF footprint overlapping one is dropped there, so it is here too.
    from ..engine import read_no_call_blocks
    no_call_blocks = read_no_call_blocks(read, mode)
    calls = []
    for a, b in unit['msp_intervals']:
        if b-a < minimum_nfr_length:
            continue
        ix = np.flatnonzero((refs >= a) & (refs < b))
        if not len(ix):
            continue
        result = call_tfs_in_interval(obs, int(ix.min()), int(ix.max()+1),
            hit, miss, float(minimum_llr), int(minimum_opportunities),
            m5c_mask=mask,m5c_llr_hit=m5c_hit,m5c_llr_miss=m5c_miss,decoder='multi_interval')
        for call in result:
            qa,qb=call.start,call.start+call.length
            score,opportunities=call.llr,call.n_opps
            if any(qa < block_end and qb > block_start for block_start, block_end in no_call_blocks):
                continue
            mapped = refs[qa:qb]
            mapped = mapped[mapped >= 0]
            if not len(mapped):
                continue
            lo, hi = int(mapped.min()), int(mapped.max()+1)
            if any(lo < nb and hi > na for na, nb in unit['raw_nuc_intervals']):
                raise ValueError('Corrected native TF crosses a frozen nucleosome')
            gap = 0
            if not any(ba <= lo and hi <= bb for ba, bb in reference_domains):
                # Actual reference gaps remain unavailable, never misses. The frozen rule
                # (allowance 0) discards any call that crosses one, so a one-base deletion
                # inside a footprint vetoes it whatever its evidence. With an allowance, a
                # call may span gaps up to that many unaligned reference bases in total;
                # the bases inside contribute nothing and the record declares the gap.
                gap = reference_gap_inside(lo, hi, reference_domains)
                if gap > int(maximum_alignment_gap_bp) or not any(ba < hi and lo < bb for ba, bb in reference_domains):
                    continue
            calls.append(dict(interval=[lo, hi], query_interval=[int(qa), int(qb)],
                llr=float(score), opportunities=int(opportunities),
                **({'alignment_gap_bp': int(gap)} if gap else {})))
    calls.sort(key=lambda v: v['interval'])
    if any(a['interval'][1] > b['interval'][0] for a, b in zip(calls, calls[1:])):
        raise ValueError('Corrected native replay yielded overlapping calls')
    return calls
