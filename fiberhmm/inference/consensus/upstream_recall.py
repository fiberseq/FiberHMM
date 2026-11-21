"""Explicit, motif-independent nuc/TF replay before family classification.

The native query sequence and annotations are used directly. No family posterior
participates in detection. The caller owns its nuc/MSP topology; the Browser keeps
the original BAM layers and exposes the replay as a separate analysis baseline.
"""
from copy import deepcopy
import numpy as np
from .adapter import reference_gap_inside
from .geometry import merged


# Per-molecule inconsistencies: the loader excludes such a molecule (with a
# receipt) rather than aborting the dataset. Anything else still raises.
MOLECULE_RECALL_FAILURES = ('Upstream TF overlaps a final nucleosome',
                            'Query annotations do not match the loaded original scaffold')


def legacy_tag_intervals(read, start_tag, length_tag, legacy_annotation_frame):
    """SEQ-frame intervals of one legacy tag pair, as the loader reads them.

    A scalar or string tag (Dorado's ``ns:i``) is not a legacy call and counts
    as absent (``legacy_annotations._legacy_tag``); a missing array partner
    still raises.
    """
    from ..legacy_annotations import _legacy_tag
    from ...io.ma_tags import flip_intervals_to_seq
    if legacy_annotation_frame not in ('seq', 'molecular'):
        raise ValueError('Nuc recall needs MA annotations or an explicit legacy tag frame')
    has_start, has_length = _legacy_tag(read, start_tag), _legacy_tag(read, length_tag)
    if not has_start and not has_length:
        return []  # Absent feature; scaffold equality below still rejects any disagreement.
    if not has_start or not has_length:
        raise ValueError('Missing original nuc/MSP annotation tags')
    from ...io.legacy_tags import array_tag_pair
    pair = array_tag_pair(read, start_tag, length_tag)
    if pair is None:
        # Consensus replay stays strict: a present but malformed pair (unequal
        # lengths, non-integer arrays) is an error, not silently truncated.
        raise ValueError('Original nuc/MSP annotation tags are not an integer array pair of equal length')
    starts, lengths = pair
    if legacy_annotation_frame == 'molecular':
        starts, lengths = flip_intervals_to_seq(starts, lengths, read)
    return [(int(a), int(a+b)) for a,b in zip(starts,lengths)]


def recall_hia5_alignment(read, unit, model, strand_mode, mode, context_size,
                          probability_threshold, minimum_llr, *,
                          minimum_opportunities=3, split_minimum_llr=4.,
                          maximum_alignment_gap_bp=0, legacy_annotation_frame='disabled',minimum_nfr_length=0,
                          nuc_recall_policy='auto'):
    from ..strand_rescue import hard_observations, cigar_to_query_ref
    from ..tf_recaller import build_llr_tables
    from ..fused_stages import _build_fused_recall_result_with_nucs, resolve_nuc_recall_policy
    from ...daf.m5c import ma_intervals
    from ...io.footprint_calls import annotation_nuc_msp, footprint_annotation_tag
    from ...io.ma_tags import flip_intervals_to_seq
    if mode not in ('pacbio-fiber', 'nanopore-fiber'):
        raise ValueError('This upstream recall adapter is restricted to Hia5')
    # The nucleosome policy fiberhmm-call resolves for this platform, recorded
    # in the settings so saved evidence says how it was replayed.
    nuc_recall_policy=resolve_nuc_recall_policy(nuc_recall_policy,mode)
    settings=dict(split_minimum_llr=split_minimum_llr,split_minimum_opportunities=3,nuc_recall_policy=nuc_recall_policy,
        nuc_min_size=85,unify_threshold=90,minimum_llr=minimum_llr,minimum_opportunities=minimum_opportunities,
        maximum_alignment_gap_bp=maximum_alignment_gap_bp,minimum_nfr_length=minimum_nfr_length,
        phase_prior=False,query_coordinate_replay=True,motif_or_family_informed=False)
    if not unit['raw_nuc_intervals'] and not unit['msp_intervals']:
        return dict(calls=[],nucleosomes=[],msps=[],excluded_calls=[],settings=settings,
                    status='no_original_scaffold')
    obs, _ = hard_observations(read, strand_mode, mode, context_size, probability_threshold)
    obs = np.asarray(obs, dtype=np.int32)
    refs = np.asarray(cigar_to_query_ref(read), dtype=np.int64)
    if len(obs) != len(refs) or len(obs) != len(read.query_sequence):
        raise ValueError('Complete query observations and CIGAR projection required for nuc recall')

    source = footprint_annotation_tag(read)   # MA first, then Ma; legacy only without nuc/msp groups

    def intervals(feature, start_tag, length_tag):
        if source == 'MA':
            return ma_intervals(read, feature)
        if source == 'Ma':  # fibertools >= 0.13: Ma, always molecular
            fibertools = annotation_nuc_msp(read, 'Ma')
            starts, lengths = flip_intervals_to_seq([a for a, _ in fibertools[feature]],
                                                    [b for _, b in fibertools[feature]], read)
            return sorted((int(a), int(a+b)) for a, b in zip(starts, lengths))
        return legacy_tag_intervals(read, start_tag, length_tag, legacy_annotation_frame)

    nucs=intervals('nuc','ns','nl');msps=intervals('msp','as','al')
    def project(a,b):
        values=refs[int(a):int(b)];values=values[values>=0]
        return [int(values.min()),int(values.max()+1)] if len(values) else None
    def projected(values):
        return sorted(iv for a,b in values if (iv:=project(a,b)) is not None)
    region=unit['_region']
    def regional(values):
        return sorted(iv for iv in values if iv[0]<region[1] and iv[1]>region[0])
    actual_msps={tuple(iv) for iv in regional(projected(msps))}
    loaded_msps={tuple(iv) for iv in regional(unit['msp_intervals'])}
    # Evidence loaders can omit very short/no-opportunity MSPs. Their original
    # query tags still exist. Verify every loaded MSP exactly, and record the
    # extra original tags rather than confusing this with a coordinate mismatch.
    if regional(projected(nucs))!=regional(unit['raw_nuc_intervals']) or not loaded_msps<=actual_msps:
        raise ValueError('Query annotations do not match the loaded original scaffold')
    hit,miss=build_llr_tables(model)
    result=_build_fused_recall_result_with_nucs(dict(query_sequence=read.query_sequence),
        dict(encoded=obs,ns=[a for a,b in nucs],nl=[b-a for a,b in nucs],
             **{'as':[a for a,b in msps],'al':[b-a for a,b in msps]}),
        hit,miss,min_llr=minimum_llr,min_opps=minimum_opportunities,unify_threshold=90,
        split_min_llr=split_minimum_llr,split_min_opps=3,nuc_min_size=85,msp_min_size=0,
        phase_nrl=0,nuc_recall_policy=nuc_recall_policy,nuc_llr_hit=hit,nuc_llr_miss=miss)
    recalled_nucs=projected([(a,a+b) for a,b in zip(result['ns'],result['nl'])])
    recalled_msps=projected([(a,a+b) for a,b in zip(result['as'],result['al'])])
    domains=merged(unit['aligned_blocks']);calls=[];excluded=[]
    for call in result['tf_calls']:
        a,b=call.start,call.start+call.length;iv=project(a,b)
        if iv is None:continue
        gap=reference_gap_inside(*iv,domains)
        record=dict(interval=iv,query_interval=[int(a),int(b)],llr=float(call.llr),
                    opportunities=int(call.n_opps),source='upstream_nuc_tf_recall',alignment_gap_bp=gap,
                    released_from_original_nuc=any(iv[0]<nb and iv[1]>na for na,nb in unit['raw_nuc_intervals']))
        if any(iv[0]<nb and iv[1]>na for na,nb in recalled_nucs):
            raise ValueError('Upstream TF overlaps a final nucleosome')
        if not any(a<=iv[0] and iv[1]<=b and b-a>=minimum_nfr_length for a,b in recalled_msps):
            excluded.append(dict(record,reason='outside_eligible_recalled_MSP'))
        elif gap>maximum_alignment_gap_bp:
            excluded.append(dict(record,reason='alignment_gap_exceeds_input_allowance'))
        else:calls.append(record)
    return dict(calls=sorted(calls,key=lambda c:c['interval']),nucleosomes=recalled_nucs,msps=recalled_msps,
        excluded_calls=excluded,settings=settings,status='recalled',
        original_tag_msps_absent_from_loaded_scaffold=[list(iv) for iv in sorted(actual_msps-loaded_msps)])


def install_recall(unit, result):
    """Install a separate analysis scaffold and preserve all original layers."""
    if 'upstream_nuc_tf_recall' in unit:
        raise ValueError('Do not recursively recall an already recalled evidence unit')
    for original, current in [('original_bam_tf_intervals','raw_tf_intervals'),
                              ('original_bam_nuc_intervals','raw_nuc_intervals'),
                              ('original_bam_msp_intervals','msp_intervals')]:
        unit.setdefault(original,deepcopy(unit[current]))
    unit['upstream_nuc_tf_recall']=deepcopy(result)
    unit['raw_nuc_intervals']=deepcopy(result['nucleosomes'])
    unit['msp_intervals']=deepcopy(result['msps'])
    unit['native_multi_interval_calls']=deepcopy(result['calls'])
    unit['native_multi_interval_tf_intervals']=[list(c['interval']) for c in result['calls']]
