#!/usr/bin/env python3
"""fiberhmm-recall-tfs CLI -- LLR-based TF footprint recaller.

Runs as a 2nd pass on a BAM already tagged by ``fiberhmm-apply``.
Writes spec-compliant ``MA``/``AQ`` Molecular-annotation tags
(https://github.com/fiberseq/Molecular-annotation-spec) plus refreshed
legacy ``ns``/``nl``/``as``/``al`` tags reflecting the unified call set.

By default, v2 short nuc calls (``nl < --unify-threshold``) that overlap
a recaller TF call are demoted out of the ``nuc+`` annotation -- the
recaller version (with proper LLR + edge-ambiguity scoring) replaces
them in the ``tf+`` annotation. v2 nucs above the threshold and
unmatched short nucs are left in ``nuc+``.

Examples:
  # DddA amplicon BAM, autodetect from --enzyme preset
  fiberhmm-recall-tfs -i tagged.bam -o recalled.bam -m models/dm6_dddb.json --enzyme ddda

  # Hia5 PacBio: defaults work
  fiberhmm-recall-tfs -i tagged.bam -o recalled.bam -m models/hia5_pacbio.json --enzyme hia5

  # Override threshold
  fiberhmm-recall-tfs ... --enzyme dddb --min-llr 5
"""
import argparse
import json
import sys

import numpy as np
import pysam

from fiberhmm.core.model_io import load_model_with_metadata
from fiberhmm.inference.tf_recaller import (
    ENZYME_PRESETS,
    apply_emission_uplift,
    build_llr_tables,
    recall_read,
    write_ma_tags,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('-i', '--in-bam', required=True,
                   help='Input BAM tagged by fiberhmm-apply (has ns/nl/as/al)')
    p.add_argument('-o', '--out-bam', required=True,
                   help='Output BAM with MA/AQ + refreshed legacy tags')
    p.add_argument('-m', '--model', required=True,
                   help='FiberHMM model JSON used by the original apply run '
                        '(or a closely related one for emission-uplift use)')
    p.add_argument('--enzyme', choices=sorted(ENZYME_PRESETS.keys()),
                   default=None,
                   help='Auto-pick min-llr + emission-uplift defaults '
                        f'({", ".join(sorted(ENZYME_PRESETS))}). '
                        'Override individual values with --min-llr / '
                        '--emission-uplift.')
    p.add_argument('--min-llr', type=float, default=None,
                   help='Override min LLR (nats) for emission. Default: '
                        'enzyme preset (5 for hia5, 4 for dddb, 5 for ddda).')
    p.add_argument('--min-opps', type=int, default=3,
                   help='Min informative target positions per call (default 3)')
    p.add_argument('--emission-uplift', type=float, default=None,
                   help='Power transform on emission probabilities. Default: '
                        'enzyme preset (1.0 for hia5/dddb, 2.0 for ddda).')
    p.add_argument('--unify-threshold', type=int, default=90,
                   help='v2 nucs with nl < this are scanned and may be '
                        'demoted out of nuc+ if a TF call overlaps. v2 nucs '
                        '>= this are always kept as nucleosomes. (default 90)')
    p.add_argument('--no-legacy-tags', action='store_true',
                   help='Skip writing refreshed ns/nl/as/al -- emit only '
                        'MA/AQ tags. Use only with MA-aware downstream tools.')
    p.add_argument('--mode', default=None,
                   help='Override observation mode (pacbio-fiber|nanopore-fiber|daf). '
                        'Default: read from model JSON.')
    p.add_argument('--context-size', type=int, default=None,
                   help='Override context size. Default: read from model JSON.')
    p.add_argument('--max-reads', type=int, default=0,
                   help='0 = no limit (default)')
    return p.parse_args()


def _resolve_model_metadata(model_path):
    """Read mode + context_size from the model JSON, with sane fallbacks."""
    mode = 'pacbio-fiber'
    k = 3
    if model_path.endswith('.json'):
        try:
            with open(model_path) as f:
                d = json.load(f)
            mode = d.get('mode', mode)
            k = int(d.get('context_size', k))
        except (OSError, ValueError):
            pass
    return mode, k


def main():
    args = parse_args()

    # Resolve presets + overrides
    preset = ENZYME_PRESETS.get(args.enzyme, {}) if args.enzyme else {}
    min_llr = args.min_llr if args.min_llr is not None else preset.get('min_llr', 5.0)
    uplift = args.emission_uplift if args.emission_uplift is not None \
        else preset.get('emission_uplift', 1.0)

    model, model_k, model_mode = load_model_with_metadata(args.model)
    # Fall back to JSON metadata if loader didn't recover them
    if not model_mode or not model_k:
        fallback_mode, fallback_k = _resolve_model_metadata(args.model)
        model_mode = model_mode or fallback_mode
        model_k = model_k or fallback_k
    mode = args.mode or model_mode
    k = args.context_size or int(model_k)

    llr_hit, llr_miss = build_llr_tables(model)
    if abs(uplift - 1.0) > 1e-9:
        llr_hit, llr_miss = apply_emission_uplift(llr_hit, llr_miss, model, uplift)

    print(
        f"[recall_tfs] enzyme={args.enzyme or 'custom'} mode={mode} k={k} "
        f"min_llr={min_llr:.2f} uplift={uplift:.2f} "
        f"unify_threshold={args.unify_threshold}",
        file=sys.stderr,
    )
    print(
        f"[recall_tfs] llr_miss median={np.median(llr_miss):+.3f}  "
        f"llr_hit median={np.median(llr_hit):+.3f}",
        file=sys.stderr,
    )

    bam_in = pysam.AlignmentFile(args.in_bam, 'rb', check_sq=False)
    bam_out = pysam.AlignmentFile(args.out_bam, 'wb', template=bam_in)

    n_reads = 0
    n_with_v2 = 0
    n_tf_total = 0
    n_dropped_short_nucs = 0
    for read in bam_in:
        if args.max_reads and n_reads >= args.max_reads:
            break
        n_reads += 1

        # Count v2 short nucs to estimate how many we demote
        v2_short_count = 0
        if read.has_tag('ns') and read.has_tag('nl'):
            n_with_v2 += 1
            v2_short_count = sum(
                1 for l in read.get_tag('nl')
                if 0 < int(l) < args.unify_threshold
            )
            # Pull v2 nq for the kept nucs (best-effort)
            v2_nq = list(read.get_tag('nq')) if read.has_tag('nq') else None
        else:
            v2_nq = None

        tf_calls, kept_nucs, msps = recall_read(
            read, llr_hit, llr_miss, mode, k,
            min_llr=min_llr, min_opps=args.min_opps,
            unify_threshold=args.unify_threshold,
        )
        n_tf_total += len(tf_calls)
        # How many of v2's short nucs survived (= kept_nucs with nl < threshold)?
        survived_short = sum(1 for _, l in kept_nucs if l < args.unify_threshold)
        n_dropped_short_nucs += max(0, v2_short_count - survived_short)

        # nq for kept nucs: align to v2's nq array when index ordering preserved
        nq_for_kept = None
        if v2_nq is not None and read.has_tag('ns'):
            try:
                ns_old = list(read.get_tag('ns'))
                nl_old = list(read.get_tag('nl'))
                old_to_nq = {(int(s), int(l)): int(v2_nq[i])
                             for i, (s, l) in enumerate(zip(ns_old, nl_old))
                             if i < len(v2_nq)}
                nq_for_kept = [old_to_nq.get((s, l), 0) for s, l in kept_nucs]
            except Exception:
                nq_for_kept = None

        write_ma_tags(
            read,
            read_length=len(read.query_sequence) if read.query_sequence else 0,
            tf_calls=tf_calls,
            kept_nucs=kept_nucs,
            msps=msps,
            nq_for_kept_nucs=nq_for_kept,
            also_write_legacy=not args.no_legacy_tags,
        )
        bam_out.write(read)

    bam_in.close()
    bam_out.close()

    print(
        f"\n[recall_tfs] processed {n_reads} reads; {n_with_v2} carried v2 "
        f"tags; {n_tf_total} TF calls emitted; "
        f"{n_dropped_short_nucs} v2 short nucs demoted to tf+",
        file=sys.stderr,
    )


if __name__ == '__main__':
    main()
