"""Explicit, motif-independent nuc/TF replay before family classification.

The native query sequence and annotations are used directly. No family posterior
participates in detection. The caller owns its nuc/MSP topology; the Browser keeps
the original BAM layers and exposes the replay as a separate analysis baseline.
"""
from bisect import bisect_left, bisect_right
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
    starts, lengths = read.get_tag(start_tag), read.get_tag(length_tag)
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
    from ...io.ma_tags import fibertools_ma_intervals, flip_intervals_to_seq
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

    def intervals(feature, start_tag, length_tag):
        if read.has_tag('MA'):
            return ma_intervals(read, feature)
        fibertools = fibertools_ma_intervals(read)
        if fibertools is not None:  # fibertools >= 0.13: Ma, always molecular
            starts, lengths = flip_intervals_to_seq([a for a, _ in fibertools[feature]],
                                                    [b for _, b in fibertools[feature]], read)
            return sorted((int(a), int(a+b)) for a, b in zip(starts, lengths))
        return legacy_tag_intervals(read, start_tag, length_tag, legacy_annotation_frame)

    nucs=intervals('nuc','ns','nl');msps=intervals('msp','as','al')
    project=_projector(refs)
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
    gap_inside=_gap_counter(domains)
    original_nucs=_Overlaps(unit['raw_nuc_intervals']);final_nucs=_Overlaps(recalled_nucs)
    eligible_msps=[(a,b) for a,b in recalled_msps if b-a>=minimum_nfr_length]
    for call in result['tf_calls']:
        a,b=call.start,call.start+call.length;iv=project(a,b)
        if iv is None:continue
        gap=gap_inside(*iv)
        record=dict(interval=iv,query_interval=[int(a),int(b)],llr=float(call.llr),
                    opportunities=int(call.n_opps),source='upstream_nuc_tf_recall',alignment_gap_bp=gap,
                    released_from_original_nuc=original_nucs.any(iv[0],iv[1]))
        if final_nucs.any(iv[0],iv[1]):
            raise ValueError('Upstream TF overlaps a final nucleosome')
        if not any(a<=iv[0] and iv[1]<=b for a,b in eligible_msps):
            excluded.append(dict(record,reason='outside_eligible_recalled_MSP'))
        elif gap>maximum_alignment_gap_bp:
            excluded.append(dict(record,reason='alignment_gap_exceeds_input_allowance'))
        else:calls.append(record)
    return dict(calls=sorted(calls,key=lambda c:c['interval']),nucleosomes=recalled_nucs,msps=recalled_msps,
        excluded_calls=excluded,settings=settings,status='recalled',
        original_tag_msps_absent_from_loaded_scaffold=[list(iv) for iv in sorted(actual_msps-loaded_msps)])


def _projector(refs):
    """project(a, b): the reference span [min, max + 1) of the aligned query positions in [a, b), or None.

    Equal to slicing ``refs[a:b]``, dropping unaligned (-1) entries and taking min and max. When the aligned reference
    positions never decrease along the query (every linear alignment), the minimum is the first aligned position in
    the range and the maximum the last, found by bisection; otherwise (or for negative bounds) it slices."""
    valid = np.flatnonzero(refs >= 0)
    aligned = refs[valid]
    def sliced(a, b):
        values = refs[int(a):int(b)]; values = values[values >= 0]
        return [int(values.min()), int(values.max()+1)] if len(values) else None
    if len(aligned) > 1 and not bool(np.all(aligned[1:] >= aligned[:-1])):
        return sliced
    index, ref = valid.tolist(), aligned.tolist()
    def project(a, b):
        a, b = int(a), int(b)
        if a < 0 or b < 0:
            return sliced(a, b)
        i, j = bisect_left(index, a), bisect_left(index, b)
        return [ref[i], ref[j-1]+1] if i < j else None
    return project


def _gap_counter(domains):
    """reference_gap_inside(lo, hi, domains) for sorted, disjoint domains (``geometry.merged``), visiting only the
    domains that can overlap [lo, hi): the same integer sum, since every other domain adds max(0, ...) = 0."""
    starts, ends = [d[0] for d in domains], [d[1] for d in domains]
    if any(s1 <= e0 for e0, s1 in zip(ends, starts[1:])):
        return lambda lo, hi: reference_gap_inside(lo, hi, domains)
    def gap(lo, hi):
        covered = 0
        for k in range(bisect_right(ends, lo), len(starts)):
            a = starts[k]
            if a >= hi:
                break
            covered += max(0, min(hi, ends[k])-max(lo, a))
        return int(hi-lo)-int(covered)
    return gap


class _Overlaps:
    """any(lo < b and hi > a for a, b in intervals), answered from the intervals sorted by start with running
    maxima of their ends (intervals with a >= hi can never overlap)."""

    def __init__(self, intervals):
        pairs = sorted((int(a), int(b)) for a, b in intervals)
        self.starts = [a for a, _ in pairs]
        self.max_end = []
        best = None
        for _, b in pairs:
            best = b if best is None or b > best else best
            self.max_end.append(best)

    def any(self, lo, hi):
        k = bisect_left(self.starts, hi)          # intervals 0..k-1 start before hi
        return k > 0 and self.max_end[k-1] > lo


def _copy_tree(value, memo):
    """copy.deepcopy for trees of dicts, lists and scalars (the recall result), keeping deepcopy's sharing of
    repeated containers through ``memo``; any other type goes to deepcopy with the same memo."""
    t = type(value)
    if t in (int, float, str, bool, type(None)):
        return value
    key = id(value)
    if key in memo:
        return memo[key]
    if t is list:
        out = memo[key] = []
        out.extend(_copy_tree(v, memo) for v in value)
        memo.setdefault(id(memo), []).append(value)     # keep the original alive while memo is (as deepcopy does)
        return out
    if t is dict:
        out = memo[key] = {}
        for k, v in value.items():
            out[_copy_tree(k, memo)] = _copy_tree(v, memo)
        memo.setdefault(id(memo), []).append(value)
        return out
    return deepcopy(value, memo)


def _deepcopy(value):
    return _copy_tree(value, {})


def install_recall(unit, result):
    """Install a separate analysis scaffold and preserve all original layers."""
    if 'upstream_nuc_tf_recall' in unit:
        raise ValueError('Do not recursively recall an already recalled evidence unit')
    for original, current in [('original_bam_tf_intervals','raw_tf_intervals'),
                              ('original_bam_nuc_intervals','raw_nuc_intervals'),
                              ('original_bam_msp_intervals','msp_intervals')]:
        if original not in unit:unit[original]=_deepcopy(unit[current])
    unit['upstream_nuc_tf_recall']=_deepcopy(result)
    unit['raw_nuc_intervals']=_deepcopy(result['nucleosomes'])
    unit['msp_intervals']=_deepcopy(result['msps'])
    unit['native_multi_interval_calls']=_deepcopy(result['calls'])
    unit['native_multi_interval_tf_intervals']=[list(c['interval']) for c in result['calls']]
