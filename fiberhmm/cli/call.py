#!/usr/bin/env python3
"""fiberhmm-call — fused apply + recall-tfs in a single Python process.

Runs the 2-state HMM (nucleosome/MSP calls) and native-LLR multi-interval
decoding (TF calls) in the same worker per read, with nucleosome recall in
between. ``--hmm-only`` stops after the HMM and writes every HMM call (the
former ``fiberhmm-apply``); ``fiberhmm-recall-nucs`` finishes such a file.

Output BAM: MA/AQ spec tags (Molecular-annotation, nuc/msp/tf with quality
bytes); --legacy-tags also writes legacy ns/nl/as/al.

Examples:
  # Hia5 PacBio, fused apply+recall (bundled model)
  fiberhmm-call -i aligned.bam -o out.bam --enzyme hia5 --seq pacbio -c 8

  # HMM calls only (nuc >= 85 bp, tf smaller, msp); recall later
  fiberhmm-call -i aligned.bam -o hmm.bam --enzyme hia5 --seq pacbio --hmm-only
  fiberhmm-recall-nucs -i hmm.bam -o out.bam --enzyme hia5 --seq pacbio

  # DddB DAF-seq with custom recall threshold
  fiberhmm-call -i aligned.bam -o out.bam --enzyme dddb --min-llr 6.0 -c 8

  # Stream to stdout for pipe to ft fire or samtools sort
  fiberhmm-call -i in.bam -o - --enzyme hia5 --seq pacbio | ft fire - -
"""
import argparse
import sys

from fiberhmm.cli.common import (
    add_basecaller_provenance_args,
    add_force_seq_arg,
    add_legacy_mode_override,
    add_legacy_tags_args,
    resolve_write_legacy,
    ml_threshold,
    non_negative_int,
    refuse_model_enzyme_assay_conflict,
    refuse_non_bam_output,
    refuse_region_options_without_region_parallel,
    require_model_files,
    resolve_basecaller_provenance,
    resolve_observation_mode,
    resolve_platform_argument,
)
from fiberhmm.cli.provenance import (
    DEFAULTS_RESOLVED_KEY,
    REPLACE_CHEMISTRY_KEY,
    ChemistryConflictError,
    reconcile_chemistry,
    resolve_effective_chemistry,
)
from fiberhmm.cli.provenance import (
    chemistry_declaration as _chemistry_declaration,
)
from fiberhmm.io.provenance import ds_tokens as basecaller_ds_tokens
from fiberhmm.cli.provenance import (
    nuc_profile_identity as _nuc_profile_identity,
)
from fiberhmm.cli.provenance import (
    nuc_profile_sha256 as _nuc_profile_sha256,
)
from fiberhmm.core.model_io import (
    ModelContextError,
    load_model_with_metadata,
    validate_context_size,
)
from fiberhmm.daf.snps import (
    DEFAULT_SNP_MIN_ALT_FIBERS,
    DEFAULT_SNP_MIN_DEPTH,
    DEFAULT_SNP_MIN_FRACTION,
    describe_snp_threshold_policy,
)
from fiberhmm.inference.chimeras import (
    STANDARD_MIN_SEG, STANDARD_PURITY, STRICT_MIN_BASEQ, STRICT_MIN_SEG,
    STRICT_MIN_SPAN, STRICT_PURITY,
)
from fiberhmm.inference.parallel import (
    _process_bam_region_parallel_fused,
    _process_bam_streaming_pipeline_fused,
)
from fiberhmm.inference.tf_recaller import (
    ENZYME_PRESETS,
    TF_DECODER_VERSION,
    resolve_cpg_mask_policy,
)
from fiberhmm.models import (
    SUPPORTED_ENZYMES,
    bundled_models_differ_by_tool,
    get_metadata_mode_aliases,
    get_observation_mode,
)
from fiberhmm.models import get_model_path as _get_bundled_model


from fiberhmm import defaults as _D  # noqa: E402  (shared call defaults)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- I/O ---
    p.add_argument('-i', '--input', required=True,
                   help='Input BAM. Use "-" for stdin (streaming mode).')
    p.add_argument('-o', '--output', required=True,
                   help='Output BAM path or "-" for stdout (unsorted).')
    p.add_argument('-m', '--model', default=None,
                   help='Apply HMM model JSON. If omitted, bundled model for '
                        '--enzyme/--seq is used.')
    p.add_argument('--recall-model', default=None,
                   help='Separate model for TF LLR tables. Default: reuse apply model.')
    p.add_argument('--enzyme', choices=sorted(SUPPORTED_ENZYMES), default=None,
                   help='Bundled enzyme preset.')
    p.add_argument('--seq', choices=['pacbio', 'nanopore'], default=None,
                   help='Sequencing platform. For Hia5 it selects the model; '
                        'when omitted it is detected from the input (MM specs: '
                        'PacBio T-a vs Nanopore A+a only; header records) and '
                        'the run stops if the evidence conflicts. A given '
                        '--seq that the reads contradict is refused (see '
                        '--force-seq). For dddb/ddda it only sets the '
                        'declared platform.')
    add_force_seq_arg(p)
    add_basecaller_provenance_args(p)
    p.add_argument('--replace-chemistry', action='store_true',
                   help='Replace, instead of reconcile with, the input BAM\'s '
                        'FIBERHMM-CHEMISTRY declaration (re-calling a BAM with '
                        'a deliberately different chemistry).')
    p.add_argument('--reference', default=None,
                   help='Reference FASTA for DAF-seq BAMs that lack '
                        'both R/Y IUPAC encoding and MD tags. When present, acts '
                        'as a fallback source for deamination-site detection '
                        '(R/Y codes and a usable MD tag take precedence). Must '
                        'match the BAM\'s assembly and be faidx-indexed.')

    # --- Apply params ---
    add_legacy_mode_override(p)
    p.add_argument('-k', '--context-size', type=int, default=None,
                   help='Context size override. Default: from model.')
    p.add_argument('--edge-trim', type=int, default=_D.EDGE_TRIM,
                   help='Bases to mask at edges (default 10)')
    p.add_argument('--min-mapq', type=non_negative_int, default=_D.MIN_MAPQ,
                   help='Min mapping quality (default 0)')
    p.add_argument('--prob-threshold', type=ml_threshold, default=None,
                   help='Min MM/ML modification probability 0-255. Default: '
                        'chemistry preset -- 248 for Hia5 Nanopore (--seq '
                        'nanopore, given or detected), 128 otherwise. R/Y- and '
                        'MD-encoded DAF input is binary and ignores it.')
    p.add_argument('--min-read-length', type=non_negative_int, default=_D.MIN_READ_LENGTH,
                   help='Min read length: the full SEQ, soft clips included (DAF with '
                        '--daf-mask-unaligned: the aligned length). Default 1000, '
                        'as the HMM step has always used.')
    p.add_argument('--msp-min-size', type=int, default=_D.MSP_MIN_SIZE,
                   help='Min MSP size (default 0)')
    p.add_argument('--nuc-min-size', type=int, default=_D.NUC_MIN_SIZE,
                   help='Min footprint size to count as nucleosome (default 85): '
                        'only footprints this size or larger bound an MSP; with '
                        '--hmm-only it also splits footprints into nuc and tf.')
    p.add_argument('--hmm-only', action='store_true',
                   help='Run the HMM step only (no nucleosome recall, no TF '
                        'recall) and write every HMM call to MA/AQ: footprints '
                        '>= --nuc-min-size bp as nuc (nuc.Q: 0, or the HMM '
                        'posterior-mean byte with --scores), smaller footprints '
                        'as tf (tf+QQQ, bytes 0: no LLR), wherever they lie '
                        '(inside MSPs too), and the HMM MSPs as msp. Nothing is '
                        'dropped, so the output is input for '
                        'fiberhmm-recall-nucs / fiberhmm-recall-tfs (they read '
                        'nuc + tf of an hmm-only call as the HMM footprints). '
                        'Replaces fiberhmm-apply. --legacy-tags adds ns/nl = nuc, '
                        'as/al = msp.')
    p.add_argument('--keep-msps', action='store_true',
                   help='With --hmm-only: keep the input\'s own MSPs (MA/Ma msp, '
                        'legacy as/al/aq), fibertools FIRE and other non-footprint '
                        'groups instead of writing the HMM\'s MSPs (e.g. MSPs from '
                        'fibertools). Replaces fiberhmm-apply --no-msps.')
    p.add_argument('-t', '--train-reads', default=None, metavar='TSV',
                   help='TSV with a "rid" column of read names to leave uncalled '
                        '(passed through), e.g. the reads a custom model was '
                        'trained on.')
    p.add_argument('--with-scores', '--scores', dest='with_scores',
                   action='store_true',
                   help='Write the HMM posterior-mean nq score of baseline '
                        'nucleosomes (with --no-recall-nucs; nucleosome recall '
                        'writes its own LLR-based nq). No aq is written. '
                        'With --hmm-only: the HMM nq of every nuc.')
    p.add_argument('-r', '--circular', action='store_true',
                   help='Enable circular molecule mode (3x tile internally, '
                        'emit wrapped MA/AQ/AN annotations).')
    p.add_argument('--process-unmapped', action=argparse.BooleanOptionalAction,
                   default=None,
                   help='Call unmapped reads that carry SEQ + MM/ML. Default: '
                        'automatic -- on for stdin, unindexed and unaligned '
                        '(uBAM) input, off (pass-through) for indexed aligned '
                        'BAMs. A run that skips >90%% of records as unmapped '
                        'fails unless --no-process-unmapped is given.')
    from fiberhmm.inference.read_filters import add_alignment_args
    add_alignment_args(p)

    # --- Recall params ---
    p.add_argument('--min-llr', type=float, default=None,
                   help='Native LLR cost per TF interval in joint decoding '
                        '(default: enzyme preset; not a calibrated FDR threshold).')
    p.add_argument('--min-opps', type=int, default=_D.MIN_OPPS,
                   help='Min informative target positions per TF call (default 3).')
    p.add_argument('--unify-threshold', type=int, default=_D.UNIFY_THRESHOLD,
                   help='v2 nucs with nl < this may be demoted to tf+ (default 90).')
    p.add_argument('--emission-uplift', type=float, default=None,
                   help='Emission power transform. Default: enzyme preset.')
    p.add_argument('--use-m5c', action=argparse.BooleanOptionalAction,
                   default=None,
                   help='DddA CpG-aware calling: CpG observations are excluded '
                        'from the first-pass HMM and from nucleosome and TF '
                        'recall, except inside confident unmethylated island '
                        'calls (MA ddda_ucg from fiberhmm-tag-m5c) the input '
                        'already carries. The same mask applies in '
                        'fiberhmm-call --hmm-only, -posteriors and -pair/-merge. '
                        'Default: on for --enzyme ddda, off otherwise. '
                        '--no-use-m5c turns the mask off, for organisms without '
                        'CpG methylation (e.g. Drosophila); it is recorded in '
                        '@PG and inherited by recall-tfs/-nucs, pair/merge, '
                        'posteriors and consensus.')
    p.add_argument('--cpg-mask-policy',
                   choices=('unmethylated-only', 'methylated-only'),
                   default=None,
                   help='With CpG-aware calling (HMM and recall): keep CpGs only '
                        'inside ddda_ucg islands (default unmethylated-only), or '
                        'mask only ddda_mcg spans (methylated-only, the former '
                        'behaviour). DddA only; ignored with a warning otherwise.')
    add_legacy_tags_args(p)
    p.add_argument('--downstream-compat', action='store_true',
                   help='Skip MA/AQ; write TF calls into legacy ns/nl track.')

    # --- Nucleosome recall params ---
    p.add_argument('--recall-nucs', action=argparse.BooleanOptionalAction, default=None,
                   help='Split over-merged nucleosomes + resolve platform-aware edges '
                        '(emits nuc.QQQ) and promote nucleosome-sized TF leaks to nuc. '
                        'ON by default for all '
                        'enzymes (DddA uses phase-aware radial inference, others the '
                        'accessible-cut Kadane split). Use --no-recall-nucs for '
                        'baseline HMM nucleosomes (nuc.Q).')
    p.add_argument('--split-min-llr', type=float, default=_D.SPLIT_MIN_LLR,
                   help='Min accessible-run LLR to split a nucleosome; for DddA, '
                        'the molecule-local linker-residue configuration LLR '
                        '(default 4.0).')
    p.add_argument('--split-min-opps', type=int, default=_D.SPLIT_MIN_OPPS,
                   help='Min informative positions in a nucleosome-splitting cut '
                        'or DddA linker residue (default 3).')
    p.add_argument(
        '--ddda-derived-tf-max-edge-gap',
        type=int,
        default=_D.DDDA_DERIVED_TF_MAX_EDGE_GAP,
        metavar='BP',
        help='DddA phase-aware radial recall only: TF calls exposed solely by '
             'nucleosome refinement must have a deamination hit within BP on '
             'both sides (default 12). Original HMM-accessible TF scan space '
             'is unchanged. Use -1 to disable the safeguard.',
    )
    p.add_argument(
        '--nuc-recall-policy',
        choices=['auto', 'conservative', 'topology'],
        default=_D.NUC_RECALL_POLICY,
        help='Nucleosome recall policy: conservative (default; auto resolves '
             'to it) or topology (keeps fragment structure in nucleosome '
             'recall; works well on some ONT Hia5 datasets). Not available '
             'for DddA.')
    p.add_argument('--phase-nrl', default='off',
                   help=argparse.SUPPRESS)  # retired experimental mode (3.0); kept only to reproduce earlier calls

    # --- DAF chimera filter (mode=daf only) ---
    p.add_argument('--chimeras', choices=['flag', 'remove', 'keep', 'split'], default='flag',
                   help='DAF strand-swap chimeras: flag (default) retains reads '
                        'uncalled with SAM 0x200 and xc:Z:strand_swap; remove drops '
                        'all records of the read; keep disables detection and '
                        'clears earlier FiberHMM chimera marks and attempts calls; '
                        'split cuts each chimeric alignment the detector flags at '
                        'its breakpoint gaps (found by the strict segmentation, with '
                        'either detector) into single-strand pieces, '
                        'one per segment (<name>/c0, <name>/c1, ... from left to '
                        'right; a read with several switches gives more than two; '
                        'the gaps are dropped, pieces shorter than --min-read-length '
                        'are dropped, MM/ML are re-indexed to each piece; unmapped '
                        'records stay whole and flagged) that are called separately. '
                        'No strand-swap detection for m6A modes.')
    p.add_argument('--keep-chimeras', action='store_true',
                   help='Deprecated alias for --chimeras keep (DAF only).')
    p.add_argument('--chimera-detector', choices=['standard', 'strict'],
                   default='standard',
                   help='DAF strand-swap chimera detector. standard (default): one '
                        'CT/GA switch with --chimera-min-seg events per segment at '
                        '--chimera-purity, every deamination counted (unchanged '
                        'from 3.2.1). strict (opt-in): min_seg 10 and purity 0.9 '
                        'by default, plus --chimera-min-baseq, --chimera-min-span, '
                        '--chimera-multi-switch and --chimera-error-mask, '
                        'duplicate-flagged reads not counted and a graded chimera '
                        'row in QC (QC grade WARN >= 10%%, high >= 20%%).')
    p.add_argument('--chimera-min-seg', type=int, default=STANDARD_MIN_SEG,
                   action=_GivenOption,
                   help='DAF chimera: min same-strand deamination events per '
                        f'segment to call a swap (default {STANDARD_MIN_SEG}; '
                        f'{STRICT_MIN_SEG} with --chimera-detector strict).')
    p.add_argument('--chimera-purity', type=float, default=STANDARD_PURITY,
                   action=_GivenOption,
                   help='DAF chimera: min same-strand purity per segment '
                        f'(default {STANDARD_PURITY:g}; {STRICT_PURITY:g} with '
                        '--chimera-detector strict).')
    p.add_argument('--chimera-min-baseq', type=int, default=None,
                   help='Strict DAF chimera detector only: deamination events count '
                        'for the chimera test only with base quality >= this '
                        f'(default {STRICT_MIN_BASEQ}; no effect on records without '
                        'QUAL; 0 disables). Also anchors split-mode breakpoints. '
                        'Calling is unaffected.')
    p.add_argument('--chimera-min-span', type=int, default=None,
                   help='Strict DAF chimera detector only: min span in query bp from '
                        'the first to the last same-strand event of each segment '
                        f'(default {STRICT_MIN_SPAN}; 0 disables).')
    p.add_argument('--chimera-multi-switch', action=argparse.BooleanOptionalAction,
                   default=None,
                   help='Strict DAF chimera detector only: also detect reads with '
                        'several switches (for example CT -> GA -> CT, an '
                        'opposite-strand island), each segment passing the same '
                        'rule; split mode cuts them into all their pieces '
                        '(default on).')
    p.add_argument('--chimera-error-mask', action=argparse.BooleanOptionalAction,
                   default=None,
                   help='Strict DAF chimera detector only: build a chimera-only mask '
                        'of recurrent opposite-strand error sites during the DAF SNP '
                        'screen and drop events there from the chimera test (default '
                        'on; written to qc/<stem>.daf_snps.chimera_error_mask.bed/.json, '
                        'never applied to calling). Needs the SNP screen: without it '
                        'there is no error mask.')
    p.add_argument('--daf-mask-runs', type=int, default=None, metavar='N',
                   help='DAF only: thin targets lying in same-strand runs of >= N original C (CT) or G (GA) bases (CC/GG and longer at N=2; see --daf-run-policy). Adjacent conversions are coupled and do not follow the per-site emission model. Default: 2 with keep-one for --enzyme ddda (duplex-validated), off otherwise; 0 disables.')
    p.add_argument('--daf-run-policy', choices=['keep-one', 'drop'], default='keep-one',
                   help='With --daf-mask-runs: keep the 5\'-most target of each run '
                        '(keep-one, default) or remove the whole run (drop).')
    p.add_argument('--daf-mask-unaligned', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='DAF only: treat query bases with no reference counterpart '
                        '(CIGAR insertions and soft clips) as no evidence, like '
                        'SNP-masked sites, and leave unaligned stretches of >= 50 bp '
                        'uncalled (default on). Deaminations are read-versus-reference '
                        'mismatches, so these bases can never carry one; '
                        '--no-daf-mask-unaligned restores the pre-3.0 encoding, which '
                        'counts them as protected.')
    p.add_argument('--daf-insert-consensus', choices=['auto', 'off'], default='auto',
                   help='DAF only: give insertions carried by many reads real evidence. '
                        'A pre-pass clusters CIGAR insertions >= 50 bp by breakpoint, '
                        'builds a deamination-aware consensus of each cluster with '
                        'enough carriers (a column that is sometimes C and sometimes T '
                        'on C->T reads is a C; G/A likewise) and re-encodes each '
                        'carrier\'s inserted bases against it; other insertions stay '
                        'masked. auto (default): on for file input with '
                        '--daf-mask-unaligned; off: mask only.')
    p.add_argument('--daf-insert-min-carriers', type=int, default=_D.DAF_INSERT_MIN_CARRIERS,
                   help='Carriers an insertion needs for --daf-insert-consensus '
                        '(default 20).')
    p.add_argument('--daf-snp-mask', default=None,
                   help='DAF only: 0-based BED of recurrent C>T/G>A SNP sites '
                        'to exclude from deamination observations. MD is preserved.')
    snp_mode = p.add_mutually_exclusive_group()
    snp_mode.add_argument('--daf-call-snps', dest='daf_call_snps', action='store_true',
                          default=None,
                          help='DAF only: force two-pass recurrent opposite-conversion '
                               'SNP masking. By default file-based DddA/DddB runs '
                               'screen automatically after a bounded depth preflight.')
    snp_mode.add_argument('--no-daf-call-snps', dest='daf_call_snps', action='store_false',
                          help='Disable automatic recurrent SNP screening.')
    p.add_argument('--daf-snp-output-prefix', default=None,
                   help='Output prefix for --daf-call-snps (default: '
                        'qc/<output BAM stem>.daf_snps).')
    p.add_argument('--daf-snp-min-fraction', type=float,
                   default=DEFAULT_SNP_MIN_FRACTION,
                   help='Minimum opposite-direction fiber fraction for '
                        f'--daf-call-snps (validated default '
                        f'{DEFAULT_SNP_MIN_FRACTION:g}).')
    p.add_argument('--daf-snp-min-depth', type=int,
                   default=DEFAULT_SNP_MIN_DEPTH,
                   help='Minimum fiber depth in each conversion-direction class '
                        f'for --daf-call-snps (validated default '
                        f'{DEFAULT_SNP_MIN_DEPTH}; all thresholds must '
                        'pass in both classes).')
    p.add_argument('--daf-snp-min-alt-fibers', type=int,
                   default=DEFAULT_SNP_MIN_ALT_FIBERS,
                   help='Minimum alternate fibers in each direction for '
                        f'--daf-call-snps (validated default '
                        f'{DEFAULT_SNP_MIN_ALT_FIBERS}).')
    p.add_argument('--daf-snp-min-dominant-events', type=int, default=5,
                   help='Minimum dominant conversions to classify a fiber for '
                        '--daf-call-snps (default 5).')
    p.add_argument('--daf-snp-min-dominant-purity', type=float, default=0.80,
                   help='Minimum conversion-direction purity for '
                        '--daf-call-snps (default 0.80).')
    p.add_argument('--daf-snp-all-snvs', action='store_true',
                   help='DAF only: the SNP screen also reports the other recurrent '
                        'SNVs (every substitution except C>T/G>A, same thresholds on '
                        'fibers of each conversion direction) in its VCF, so '
                        'fiberhmm-extract --variants auto gets a fuller site list. '
                        'The BED mask is unchanged.')
    p.add_argument('--daf-snp-min-amplicon-reads', type=int, default=20,
                   help='Minimum aligned reads required to discover and plot an '
                        'amplicon consensus in SNP QC (default 20).')
    p.add_argument(
        '--ddda-mcg', action='store_true',
        help='Deprecated integrated per-CpG mode; retained only to emit a clear '
             'migration error. Run fiberhmm-call, then fiberhmm-tag-m5c '
             '(whole CpG islands), then fiberhmm-call again on the tagged BAM.',
    )

    # --- PCR dedup (DAF / ddda|dddb only) ---
    dedup_mode = p.add_mutually_exclusive_group()
    dedup_mode.add_argument('--dedup', dest='dedup', action='store_true',
                   default=None,
                   help='DAF (ddda/dddb) only: force PCR duplicate detection '
                        '(already automatic for file-based DddA/DddB calls). Detect by '
                        'deamination-pattern fingerprint (see fiberhmm-dedup) '
                        'and similar alignment ends BEFORE footprinting. The '
                        'integrated default is nondestructive: retain every read '
                        'and set 0x400 plus di/ds cluster tags. Amplicon/UMI-less DAF libraries can be '
                        'heavily PCR-duplicated and coordinate dedup does not '
                        'apply. Requires a file input (not stdin). Ignored for '
                        'fiber-seq (hia5).')
    dedup_mode.add_argument('--no-dedup', dest='dedup', action='store_false',
                   help='Disable automatic DddA/DddB duplicate marking.')
    p.add_argument('--dedup-min-jaccard', type=float, default=_D.DEDUP_MIN_JACCARD,
                   help='Deamination-set Jaccard threshold for --dedup '
                        '(default 0.95). Use 0.90 for optional more aggressive '
                        'matching; lower thresholds can merge distinct molecules.')
    p.add_argument('--dedup-flag-only', action='store_true',
                   help='Deprecated compatibility spelling for the nondestructive '
                        'integrated default (0x400 + di/ds; retain every read).')
    p.add_argument('--dedup-collapse', action='store_true',
                   help='With --dedup: destructively collapse each duplicate cluster '
                        'to one representative. Default: mark and retain all reads.')
    p.add_argument('--dedup-min-deam', type=int, default=_D.DEDUP_MIN_DEAM,
                   help='With --dedup: reads with fewer deamination calls are '
                        'not fingerprinted and pass through (default 10).')
    p.add_argument('--dedup-prob-threshold', type=int, default=None,
                   help='With --dedup: min ML probability for MM/ML-native dU '
                        'calls, 0-255 (default: the calling --prob-threshold, '
                        '128). R/Y and MD inputs are binary and ignore it.')
    p.add_argument('--dedup-ignore-strand', action='store_true',
                   help='With --dedup: cluster reads across deamination '
                        'flavours (C->T with G->A reads). Default: only reads '
                        'of the same flavour can be duplicates.')
    p.add_argument('--dedup-max-end-diff', type=int, default=_D.DEDUP_MAX_END_DIFF,
                   help='With --dedup: maximum difference at both aligned reference '
                        'ends for duplicate matching (default 50 bp).')
    p.add_argument('--dedup-stats-tsv', default=None,
                   help='With --dedup: write a cluster_id<TAB>n_reads table.')
    # MinHash internals (num-hashes/bands/seed) are intentionally not exposed
    # here; use the standalone fiberhmm-dedup to tune those.

    # --- Parallelism ---
    p.add_argument('-c', '--cores', type=int, default=4,
                   help='Worker processes (0 = all CPUs; default 4).')
    p.add_argument('--chunk-size', type=int, default=500,
                   help='Reads per worker chunk (default 500; streaming mode only).')
    p.add_argument('--io-threads', type=int, default=8,
                   help='htslib I/O threads per stage (default 8).')
    p.add_argument('--max-reads', type=int, default=0,
                   help='0 = no limit (default; streaming mode only)')

    # --- Bounded post-call QC ---
    p.add_argument('--qc', action=argparse.BooleanOptionalAction, default=True,
                   help='Run bounded signal/periodicity/footprint QC after a file '
                        'output (default: on; use --no-qc to disable).')
    p.add_argument('--qc-sample-reads', type=int, default=2000,
                   help='Target random-window QC sample size (default 2000).')
    p.add_argument('--qc-seed', type=int, default=20260824,
                   help='Deterministic QC sampler seed (default 20260824).')
    p.add_argument('--qc-min-mapq', type=int, default=_D.MIN_MAPQ,
                   help='Minimum mapping quality for the QC sample (default: 0, no filtering).')
    p.add_argument('--qc-output-prefix', default=None,
                   help='QC output prefix (default: qc/<BAM stem> beside output BAM).')

    # --- Region-parallel mode ---
    p.add_argument('--region-parallel', action='store_true',
                   help='Process genomic regions in parallel (one worker per region). '
                        'Scales linearly with --cores up to chromosome count. '
                        'Requires coordinate-sorted + indexed input BAM. '
                        'Recommended for full-genome runs; use streaming for stdin/unaligned.')
    p.add_argument('--region-size', type=int, default=10_000_000,
                   help='Region size in bp for --region-parallel (default 10 Mb).')
    p.add_argument('--skip-scaffolds', action='store_true',
                   help='Skip scaffold/contig chromosomes in region-parallel mode.')
    p.add_argument('--chroms', nargs='+', default=None,
                   help='Only process these chromosomes (region-parallel mode).')

    # --- Long runs: resume and progress ---
    p.add_argument('--resume', action='store_true',
                   help='Continue an interrupted --region-parallel run: regions finished '
                        'in the work directory are reused, missing or partial regions are '
                        'rerun, then the output is merged and published as usual. Refused '
                        'if the input BAM or any effective parameter changed. Implies '
                        '--region-parallel for an indexed, aligned input; with no work '
                        'directory present a new run starts. Streaming (stdin, unindexed) '
                        'runs cannot resume.')
    p.add_argument('--work-dir', default=None, metavar='DIR',
                   help='Region-parallel work directory holding finished region BAMs and '
                        'the resume manifest (default: .<output name>.fiberhmm-work beside '
                        'the output). Kept when a run fails or is interrupted; removed '
                        'after a successful publish.')
    p.add_argument('--keep-work-dir', action='store_true',
                   help='Keep the region-parallel work directory after a successful run.')
    p.add_argument('--progress-json', nargs='?', const='-', default=None, metavar='FILE',
                   help='Write machine-readable progress, one JSON object per line, to '
                        'stderr (no value) or append to FILE: start, region (regions '
                        'done/total, reads, reads/s, ETA), merge, done and stopped events '
                        '(schema fiberhmm.progress.v1).')

    from fiberhmm.cli.common import add_version_args
    add_version_args(p)
    args = p.parse_args()
    if args.keep_chimeras:
        if args.chimeras != 'flag' and args.chimeras != 'keep':
            p.error('--keep-chimeras conflicts with --chimeras remove')
        args.chimeras = 'keep'
    _resolve_chimera_detector_args(p, args)
    if args.chimera_min_seg < 1 or not 0.5 < args.chimera_purity <= 1:
        p.error('--chimera-min-seg must be >= 1 and --chimera-purity in (0.5, 1]')
    if args.keep_msps and not args.hmm_only:
        p.error('--keep-msps requires --hmm-only (recall re-derives MSPs from '
                'the recalled nucleosomes)')
    # Explicit --enzyme ddda fails here; an enzyme inherited from the input
    # header is checked after chemistry resolution (_main).
    from fiberhmm.inference.fused_stages import nuc_recall_policy_error
    policy_error = nuc_recall_policy_error(args.nuc_recall_policy, args.enzyme)
    if policy_error:
        p.error(policy_error)
    if args.hmm_only:
        if args.recall_nucs:
            p.error('--recall-nucs conflicts with --hmm-only (no recall runs; '
                    'run fiberhmm-recall-nucs on the output)')
        if args.recall_model:
            p.error('--recall-model has no effect with --hmm-only (no recall runs)')
    return args


class _GivenOption(argparse.Action):
    """Store the value and remember that it was given, so the strict
    detector can apply its own default when it was not."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f'_given_{self.dest}', True)


_STRICT_ONLY_CHIMERA_OPTIONS = (
    ('chimera_min_baseq', '--chimera-min-baseq'),
    ('chimera_min_span', '--chimera-min-span'),
    ('chimera_multi_switch', '--[no-]chimera-multi-switch'),
    ('chimera_error_mask', '--[no-]chimera-error-mask'),
)


def _resolve_chimera_detector_args(parser, args):
    """Fill the chimera detector defaults of ``--chimera-detector``.

    standard: min_seg/purity default 5/0.8; the strict-only options are
    refused (``--chimeras split`` works with both detectors). strict: the
    ``STRICT_*`` defaults.
    """
    from fiberhmm.inference.chimeras import (
        STRICT_ERROR_MASK, STRICT_MULTI_SWITCH,
    )
    if args.chimera_detector == 'strict':
        if not getattr(args, '_given_chimera_min_seg', False):
            args.chimera_min_seg = STRICT_MIN_SEG
        if not getattr(args, '_given_chimera_purity', False):
            args.chimera_purity = STRICT_PURITY
        defaults = dict(chimera_min_baseq=STRICT_MIN_BASEQ,
                        chimera_min_span=STRICT_MIN_SPAN,
                        chimera_multi_switch=STRICT_MULTI_SWITCH,
                        chimera_error_mask=STRICT_ERROR_MASK)
        for key, value in defaults.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
        if args.chimera_min_baseq < 0 or args.chimera_min_span < 0:
            parser.error('--chimera-min-baseq and --chimera-min-span must be >= 0')
        return
    given = [flag for key, flag in _STRICT_ONLY_CHIMERA_OPTIONS
             if getattr(args, key) is not None]
    if given:
        parser.error(f"{', '.join(given)} need --chimera-detector strict")


def _resolve_apply_model(args):
    if args.model:
        return args.model
    if args.enzyme is None:
        print("error: one of --model or --enzyme required.", file=sys.stderr)
        sys.exit(1)
    return _get_bundled_model(args.enzyme, tool='apply', seq=args.seq)


def _resolve_recall_model(args):
    """TF-recall table: --recall-model, else the apply model, else the preset's.

    A custom ``-m`` drives TF recall too (``--recall-model`` help: "Default:
    reuse apply model"), so a refit table is used for both passes. The one
    exception is a preset that ships a separate recall table (DddA:
    ddda_nuc.json for apply, ddda_TF.json for recall), where ``-m`` replaces
    only the apply model.
    """
    if args.recall_model:
        return args.recall_model
    if args.model and not bundled_models_differ_by_tool(args.enzyme, args.seq):
        return None  # reuse the custom apply model
    if args.enzyme:
        try:
            return _get_bundled_model(args.enzyme, tool='recall', seq=args.seq)
        except (KeyError, FileNotFoundError):
            pass
    return None  # reuse apply model


def _resolve_nuc_profile_path(args, recall_nucs: bool):
    """DddA uses phase-aware radial nucleosome inference.

    Other enzymes use the accessible-cut Kadane split and have no radial
    profile.
    """
    if recall_nucs and args.enzyme == 'ddda':
        from fiberhmm.models import _bundled_model_path
        return _bundled_model_path('ddda_nuc_profile.json')
    return None


def _resolve_nuc_model_path(args, recall_nucs: bool):
    """Return the frozen likelihood model for bundled DddA nuc refinement.

    ``--recall-model`` changes TF scoring only. A caller that explicitly
    identifies the chemistry as DddA therefore retains the locked bundled nuc
    refinement model; fully custom workflows without ``--enzyme ddda`` retain
    the compatibility shared-table fallback.
    """
    if (
        recall_nucs
        and args.enzyme == 'ddda'
    ):
        return _get_bundled_model(args.enzyme, tool='nuc_refine', seq=args.seq)
    return None


def _resolve_derived_tf_edge_gap(args, recall_nucs: bool):
    """Resolve the DddA-only radial-recall TF edge safeguard."""
    value = int(args.ddda_derived_tf_max_edge_gap)
    if value < -1:
        print(
            "error: --ddda-derived-tf-max-edge-gap must be -1 or >= 0",
            file=sys.stderr,
        )
        sys.exit(2)
    if recall_nucs and args.enzyme == 'ddda' and value >= 0:
        return value
    return None


def _resolve_phase_nrl(args, apply_model_path, recall_model_path, mode, k,
                       recall_nucs, input_bam, include_unmapped=False,
                       precall_chimeric_names=None) -> int:
    """Resolve --phase-nrl (off / auto / fixed bp) to an int (0 = off).

    ``input_bam`` is the BAM to sample for auto-estimation -- the deduped
    file when --dedup ran, so the NRL isn't biased by PCR duplicates.
    """
    raw = str(args.phase_nrl).strip().lower()
    if raw in ('off', '', '0', 'none'):
        return 0
    if not recall_nucs:
        # phase rides on the nuc recaller; silently off when recall is off.
        return 0
    if raw != 'auto':
        try:
            return max(0, int(raw))
        except ValueError:
            print(f"  WARNING: invalid --phase-nrl {args.phase_nrl!r}; using off.",
                  file=sys.stderr)
            return 0
    # auto-estimate
    if input_bam == '-':
        print("  NOTE: --phase-nrl auto needs a file input to sample; "
              "falling back to anchor 185 bp.", file=sys.stderr)
        return 185
    from fiberhmm.inference.nrl_estimate import estimate_phase_nrl
    res = estimate_phase_nrl(
        input_bam, apply_model_path, recall_model_path,
        mode=mode, context_size=k,
        split_min_llr=args.split_min_llr, split_min_opps=args.split_min_opps,
        nuc_recall_policy=_resolve_nuc_recall_policy(args, mode),
        nuc_min_size=args.nuc_min_size, msp_min_size=args.msp_min_size,
        prob_threshold=args.prob_threshold, edge_trim=args.edge_trim,
        include_unmapped=include_unmapped,
        precall_chimeric_names=precall_chimeric_names,
    )
    ci = res['ci']
    ci_str = f" CI[{ci[0]:.0f}-{ci[1]:.0f}]" if ci else ""
    print(f"  phase NRL: {res['nrl']} bp ({res['source']}, "
          f"{res['n_pairs']:,} pairs from {res['n_reads']:,} reads{ci_str})",
          file=sys.stderr)
    return int(res['nrl'])


def _resolve_nuc_recall_policy(args, mode: str) -> str:
    """Resolve the public auto policy to the core recaller policy."""
    from fiberhmm.inference.fused_stages import resolve_nuc_recall_policy
    return resolve_nuc_recall_policy(getattr(args, 'nuc_recall_policy', 'auto'), mode)


def _refuse_nuc_recall_policy(policy: str, enzyme, prog: str) -> None:
    """Exit 2 (argparse's usage-error status) for a policy the resolved
    enzyme cannot run, e.g. topology on a DddA input whose header names it."""
    from fiberhmm.inference.fused_stages import nuc_recall_policy_error
    message = nuc_recall_policy_error(policy, enzyme)
    if message:
        print(f"{prog}: error: {message}", file=sys.stderr)
        sys.exit(2)


def _resolve_dedup(args, mode: str) -> bool:
    """Resolve automatic, explicit, and compatibility DAF dedup settings."""
    if args.dedup is not None:
        return bool(args.dedup)
    if args.dedup_collapse or args.dedup_flag_only:
        return True
    return mode == 'daf' and args.enzyme in ('ddda', 'dddb')


def _check_daf_inputs(input_bam: str, reference: str = None,
                       n_sniff: int = 10) -> None:
    """Sniff the first ~N mapped reads of a DAF-mode input BAM and confirm
    at least one deamination source is available: R/Y IUPAC in the stored
    sequence, an MD tag, or a user-supplied ``--reference`` FASTA.

    Exits with an actionable error if none is available -- otherwise
    every read would be silently skipped and the run would produce an
    empty output BAM.
    """
    import pysam

    # Reference FASTA is sufficient by itself (we can always reconstruct
    # mismatches from it regardless of MD tag presence).
    has_ref = reference is not None

    from fiberhmm.daf.encoder import md_matches_cigar

    has_ry = False
    has_md = False
    md_bad = 0
    md_total = 0
    checked = 0
    md_missing = 0
    bug_note = ""
    try:
        with pysam.AlignmentFile(input_bam, 'rb', check_sq=False) as bam:
            from fiberhmm.pipeline.md_fill import minimap2_md_bug_note
            bug_note = minimap2_md_bug_note(bam.header)
            for read in bam:
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                seq = read.query_sequence
                if seq and ('R' in seq or 'Y' in seq):
                    has_ry = True
                if read.has_tag('MD'):
                    has_md = True
                    md_total += 1
                    if not md_matches_cigar(read):
                        md_bad += 1
                else:
                    md_missing += 1
                checked += 1
                if checked >= n_sniff:
                    break
    except (ValueError, OSError):
        # Let downstream error handling report a clear message about the BAM.
        return

    if has_ry or has_md or has_ref:
        if bug_note and md_missing and not has_ry and not has_ref:
            # Some reads carry MD (e.g. ones a pipeline rebuilt), the rest do not.
            print(f"  NOTE: {md_missing}/{checked} of the sniffed reads have no MD tag;\n"
                  "  those reads will be skipped in DAF mode.\n" + bug_note,
                  file=sys.stderr)
        # Warn about stale MD only when we'll actually be relying on it
        # (no R/Y fast path available for these reads) AND some were bad.
        if md_bad and not has_ry and not has_ref:
            print(
                f"  NOTE: {md_bad}/{md_total} of the sniffed reads with MD tags\n"
                f"  have MD/CIGAR length mismatches (typical of consensus BAMs\n"
                f"  where MD is stale after CIGAR was recomputed). Those reads\n"
                f"  will be skipped in DAF mode. The reads themselves are fine;\n"
                f"  only the MD annotation is stale. To recover calls on those\n"
                f"  reads, regenerate MD with:\n"
                f"    samtools calmd -b aligned.bam ref.fa > fixed.bam\n"
                f"  or pass --reference ref.fa to fiberhmm-call.",
                file=sys.stderr,
            )
        return

    print(
        "error: DAF-seq calling needs deamination calls, and none of the supported\n"
        f"  sources were found in the first {checked} mapped reads of {input_bam}:\n"
        "    - R/Y IUPAC codes in the stored query sequence\n"
        "      (produced by fiberhmm-daf-encode), or\n"
        "    - MD tags on aligned reads\n"
        "      (set by 'minimap2 --MD' or 'samtools calmd'), or\n"
        "    - a reference FASTA via --reference ref.fa\n"
        "\n"
        "  One of these is required for fiberhmm-call to locate C->T / G->A\n"
        "  deamination sites. Without it every read would be silently skipped.\n"
        + (("\n" + bug_note) if bug_note else ""),
        file=sys.stderr,
    )
    sys.exit(2)


def _configure_ddda_mcg(args, mode: str) -> bool:
    """Reject the retired sub-island mode or print the whole-island hint."""
    enabled = bool(getattr(args, 'ddda_mcg', False))
    enzyme = getattr(args, 'enzyme', None)
    if not enabled:
        if enzyme == 'ddda' and mode == 'daf':
            print(
                "\n  NOTE: To infer DddA mCG, first finish ordinary calling, then run\n"
                "        fiberhmm-tag-m5c -i calls.bam -o mcg.bam "
                "-r ref.fa --enzyme ddda\n"
                "        and call mcg.bam again with fiberhmm-call (its CpG-aware "
                "recall\n"
                "        uses the islands). The m5C caller reports\n"
                "        one state per complete CpG island inferred from the "
                "reference by default.\n",
                file=sys.stderr,
            )
        return False

    print(
        "error: --ddda-mcg used a retired per-CpG integrated caller. "
        "Run fiberhmm-call without this flag, then fiberhmm-tag-m5c "
        "(one state per complete CpG island), then fiberhmm-call again on "
        "the tagged BAM.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _dedup_input_first(input_bam, output_bam, min_jaccard, flag_only, io_threads,
                       region_parallel, min_deam=10, prob_threshold=0,
                       ignore_strand=False, stats_tsv=None, max_end_diff=50):
    """PCR-dedup the input BAM BEFORE footprinting (DAF/deaminase only).

    The default flag-only mode is nondestructive: every read is retained and
    duplicate cluster members receive 0x400 plus di/ds tags. Collapse mode is
    explicit. Pooled downstream analyses can ignore marked duplicates while
    the HMM/recaller still annotates every read. Returns a
    ``(temporary_bam, statistics)`` tuple. The BAM is ``None`` if nothing was
    fingerprintable; the statistics are forwarded to automatic QC so its
    duplication panel reports the full run rather than a bounded estimate.
    """
    import os
    import tempfile

    import pysam

    from fiberhmm.cli.dedup import run_dedup
    print("\n  --dedup: detecting PCR duplicates by deamination fingerprint "
          f"(Jaccard >= {min_jaccard}, ends ±{max_end_diff} bp, "
          f"{'mark/retain' if flag_only else 'collapse'}) "
          "BEFORE footprinting...", file=sys.stderr)
    outdir = os.path.dirname(os.path.abspath(output_bam)) if output_bam != '-' else None
    fd, tmp = tempfile.mkstemp(prefix='.fiberhmm_dedup_', suffix='.bam', dir=outdir)
    os.close(fd)
    try:
        stats = run_dedup(input_bam, tmp, min_jaccard=min_jaccard,
                          collapse=not flag_only, io_threads=io_threads,
                          min_deam=min_deam, prob_threshold=prob_threshold,
                          ignore_strand=ignore_strand, stats_tsv=stats_tsv,
                          max_end_diff=max_end_diff)
        if stats is None:
            _remove_dedup_temp(tmp)
            return None, None
        if region_parallel:
            # region-parallel needs a coordinate index on its input; dedup
            # preserves the input's sort order so this is valid.
            try:
                pysam.index(tmp)
            except pysam.SamtoolsError as exc:
                raise ValueError(
                    f"--region-parallel: could not index the deduplicated copy "
                    f"of {input_bam} ({exc}); the input must be "
                    "coordinate-sorted."
                ) from exc
    except BaseException:
        _remove_dedup_temp(tmp)
        raise
    return tmp, stats


def _remove_dedup_temp(dedup_tmp):
    """Remove the pre-footprinting dedup temp BAM and any index beside it."""
    import os

    if not dedup_tmp:
        return
    dedup_stem = dedup_tmp[:-4] if dedup_tmp.endswith('.bam') else dedup_tmp
    for path in (
        dedup_tmp,
        dedup_tmp + '.bai',
        dedup_tmp + '.csi',
        dedup_stem + '.bai',
        dedup_stem + '.csi',
    ):
        try:
            os.remove(path)
        except OSError:
            pass


def _daf_snp_depth_preflight(
    input_bam,
    min_mapq=0,
    min_local_depth=20,
    max_records=20_000,
):
    """Bounded screen for enough local coverage to justify full SNP passes.

    A bounded interval sweep measures exact physical overlap depth rather than
    counting reads that merely touch the same coarse bin. The exact SNP caller
    still enforces conversion-direction and per-position mismatch support; this
    screen only prevents an obviously sub-threshold BAM from being rescanned.
    """
    from collections import Counter

    import pysam

    events = Counter()
    start_bins = Counter()
    examined = 0
    eligible = 0
    aligned_bases = 0
    mapped_total = None
    reference_bases = 0
    with pysam.AlignmentFile(input_bam, 'rb', check_sq=False) as bam:
        reference_bases = sum(int(length) for length in bam.lengths)
        try:
            mapped_total = sum(item.mapped for item in bam.get_index_statistics())
        except (ValueError, OSError, AttributeError):
            mapped_total = None
        for read in bam.fetch(until_eof=True):
            examined += 1
            if examined > max_records:
                break
            if (
                read.is_unmapped
                or read.is_secondary
                or read.is_supplementary
                or read.is_duplicate
                or read.mapping_quality < min_mapq
                or read.reference_end is None
            ):
                continue
            eligible += 1
            reference_id = int(read.reference_id)
            start = int(read.reference_start)
            end = int(read.reference_end)
            aligned_bases += max(0, end - start)
            start_bins[(reference_id, start // 10_000)] += 1
            events[(reference_id, start)] += 1
            events[(reference_id, end)] -= 1
    maximum = 0
    current_reference = None
    depth = 0
    for (reference_id, _position), change in sorted(events.items()):
        if reference_id != current_reference:
            current_reference = reference_id
            depth = 0
        depth += change
        maximum = max(maximum, depth)
    mean_span = aligned_bases / eligible if eligible else 0.0
    estimated_coverage = (
        mapped_total * mean_span / reference_bases
        if mapped_total is not None and reference_bases
        else aligned_bases / reference_bases if reference_bases else 0.0
    )
    max_start_bin_reads = max(start_bins.values(), default=0)
    supported_start_bin_reads = sum(
        count for count in start_bins.values() if count >= min_local_depth
    )
    supported_start_bin_fraction = (
        supported_start_bin_reads / eligible if eligible else 0.0
    )
    genome_supported = estimated_coverage >= min_local_depth
    targeted_supported = (
        max_start_bin_reads >= min_local_depth
        and supported_start_bin_fraction >= 0.10
    )
    return {
        'run': bool(genome_supported or targeted_supported),
        'reason': (
            'genome_coverage' if genome_supported
            else 'targeted_locus' if targeted_supported
            else 'low_coverage'
        ),
        'records_examined': min(examined, max_records),
        'eligible_reads': eligible,
        'max_local_depth': maximum,
        'max_alignment_start_bin_reads': max_start_bin_reads,
        'supported_start_bin_fraction': float(supported_start_bin_fraction),
        'estimated_genome_coverage': float(estimated_coverage),
        'minimum_depth': min_local_depth,
    }


def _input_is_indexed_and_aligned(input_bam):
    """(indexed, aligned) for a file input; (False, False) if unreadable."""
    import pysam

    try:
        with pysam.AlignmentFile(input_bam, 'rb', check_sq=False) as bam:
            return bool(bam.has_index()), bool(bam.references)
    except (OSError, ValueError):
        return False, False


def _resolve_process_unmapped(args):
    """Auto-enable unmapped calling for stdin, unindexed or unaligned input.

    Returns ``(process_unmapped, reason)``. An explicit
    --process-unmapped/--no-process-unmapped always wins.
    """
    if args.process_unmapped is not None:
        return bool(args.process_unmapped), None
    if args.region_parallel:
        return False, None
    if args.input == '-':
        return True, 'stdin input'
    indexed, aligned = _input_is_indexed_and_aligned(args.input)
    if not aligned:
        return True, 'unaligned input (no @SQ reference sequences)'
    if not indexed:
        return True, 'unindexed input'
    return False, None


def _resolve_call_prob_threshold(args, chemistry=None):
    """Explicit --prob-threshold, else the resolved chemistry's default.

    ``chemistry`` is the run's (input-reconciled) chemistry declaration, so a
    custom ``-m`` that inherits Hia5 Nanopore from the input BAM also gets the
    Nanopore threshold.
    """
    from fiberhmm.models import resolve_prob_threshold

    chemistry = chemistry or {}
    enzyme = args.enzyme or chemistry.get('enzyme')
    seq = args.seq or chemistry.get('platform')
    return resolve_prob_threshold(args.prob_threshold, enzyme, seq)


def _load_train_reads(path):
    """Read names to leave uncalled (``--train-reads``: a TSV with a ``rid``
    column, as fiberhmm-train writes); an empty set without a path."""
    if not path:
        return set()
    import csv
    with open(path, newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if not reader.fieldnames or 'rid' not in reader.fieldnames:
            print(f"error: --train-reads {path} has no 'rid' column", file=sys.stderr)
            sys.exit(2)
        rids = {row['rid'] for row in reader if row.get('rid')}
    print(f"  Excluding {len(rids):,} training reads (--train-reads)", file=sys.stderr)
    return rids


def _validate_model_context(model_paths, k):
    """Exit with a clear message if ``k`` cannot index every model table."""
    seen = set()
    for label, path in model_paths:
        if not path or path in seen:
            continue
        seen.add(path)
        model, _, _ = load_model_with_metadata(path)
        try:
            validate_context_size(model, k, label=f"{label} {path}")
        except ModelContextError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(2)


def _strict_worker_kwargs(args, error_mask_info):
    """Strict-detector worker settings ({} for the standard detector)."""
    if args.chimera_detector != 'strict':
        return {}
    return dict(chimera_detector='strict',
                chimera_min_baseq=args.chimera_min_baseq,
                chimera_min_span=args.chimera_min_span,
                chimera_multi_switch=args.chimera_multi_switch,
                chimera_error_mask_path=(error_mask_info or {}).get('bed'))


def _chimera_detector_args(args):
    """The run's strict chimera detector settings (``detector_config`` keywords)."""
    return dict(min_seg=args.chimera_min_seg, purity=args.chimera_purity,
                min_baseq=args.chimera_min_baseq, min_span=args.chimera_min_span,
                multi_switch=args.chimera_multi_switch)


def _chimera_pass(args, mode, working_input, snp_mask_path, process_unmapped,
                  stdout_mode, error_mask_info=None):
    """DAF chimera decisions (and split pieces): (input, names, stats, tmpdir).

    Only DAF flag/remove/split need a QNAME decision pass. In particular m6A
    and keep retain the original working path without copying/scanning.
    Keep still supplies an authoritative empty decision to consumers, which
    clear old marks as they read each record. ``error_mask_info`` is the
    chimera error-mask block (:func:`fiberhmm.inference.chimeras.error_mask_summary`);
    its ``bed`` is applied to the chimera test.
    """
    chimera_tmpdir = None
    if mode == 'daf' and args.chimera_detector != 'strict':
        return _standard_chimera_pass(args, mode, working_input, snp_mask_path,
                                      process_unmapped, stdout_mode)
    if mode == 'daf' and args.chimeras != 'keep':
        from fiberhmm.inference.chimeras import (
            ChimeraDecisions, chimera_summary_lines, chimera_warning,
            prepare_chimeras, seekable_input,
        )
        spool = None
        # Split mode always writes the pieces to a private spool.
        if args.chimeras == 'split' or not seekable_input(working_input):
            import tempfile
            from pathlib import Path
            chimera_tmpdir = tempfile.TemporaryDirectory(
                prefix='.fiberhmm_chimeras_',
                dir=None if stdout_mode else str(Path(args.output).parent))
            spool = str(Path(chimera_tmpdir.name) / 'source.bam')
        chimeric_names = ChimeraDecisions()
        chimera_stats = prepare_chimeras(
            working_input, mode=args.chimeras, detector='strict',
            **_chimera_detector_args(args),
            error_mask=(error_mask_info or {}).get('bed'),
            error_mask_info=error_mask_info,
            prob_threshold=args.prob_threshold, reference=args.reference,
            snp_mask=snp_mask_path, min_mapq=args.min_mapq,
            min_read_length=args.min_read_length, process_unmapped=process_unmapped,
            io_threads=args.io_threads, chimeric_names=chimeric_names,
            spool_bam=spool, alignments=args.alignments)
        if spool is not None:
            working_input = spool
        print(f"  DAF strand-swap chimeras: {chimera_stats['n_chimeric']:,} / "
              f"{chimera_stats['n_reads_checked']:,} checked reads "
              f"(mode={args.chimeras}; detector min_seg={args.chimera_min_seg}, "
              f"purity={args.chimera_purity}; "
              f"{chimera_stats['n_records_affected']:,} records)", file=sys.stderr)
        for line in chimera_summary_lines(chimera_stats, indent='    ')[1:]:
            print(line, file=sys.stderr)
        warning = chimera_warning(chimera_stats)
        if warning:
            print(f"\n  {warning}\n", file=sys.stderr)
        return working_input, chimeric_names, chimera_stats, chimera_tmpdir
    if mode == 'daf':
        from fiberhmm.inference.chimeras import empty_strict_chimera_stats
        return working_input, set(), empty_strict_chimera_stats(
            'keep', **_chimera_detector_args(args)), None
    return working_input, None, None, None


def _standard_chimera_pass(args, mode, working_input, snp_mask_path,
                           process_unmapped, stdout_mode):
    """The standard detector's decision pass (as in 3.2.1).

    ``--chimeras split``: the standard detector flags, and its flagged
    records are cut into a private spool (breakpoints:
    :func:`fiberhmm.inference.chimeras.standard_breakpoints`).
    """
    chimera_tmpdir = None
    chimeric_names = None
    chimera_stats = None
    # Only DAF flag/remove need a QNAME decision pass. In particular m6A
    # and keep retain the original working path without copying/scanning.
    # Keep still supplies an authoritative empty decision to consumers,
    # which clear old marks as they read each record.
    if mode == 'daf' and args.chimeras != 'keep':
        from fiberhmm.inference.chimeras import ChimeraDecisions, prepare_chimeras, seekable_input
        spool = None
        # Split mode always writes the pieces to a private spool.
        if args.chimeras == 'split' or not seekable_input(working_input):
            import tempfile
            from pathlib import Path
            chimera_tmpdir = tempfile.TemporaryDirectory(
                prefix='.fiberhmm_chimeras_',
                dir=None if stdout_mode else str(Path(args.output).parent))
            spool = str(Path(chimera_tmpdir.name) / 'source.bam')
        chimeric_names = ChimeraDecisions()
        chimera_stats = prepare_chimeras(
            working_input, mode=args.chimeras,
            min_seg=args.chimera_min_seg, purity=args.chimera_purity,
            prob_threshold=args.prob_threshold, reference=args.reference,
            snp_mask=snp_mask_path, min_mapq=args.min_mapq,
            min_read_length=args.min_read_length, process_unmapped=process_unmapped,
            io_threads=args.io_threads, chimeric_names=chimeric_names,
            spool_bam=spool, alignments=args.alignments)
        if spool is not None:
            working_input = spool
        print(f"  DAF strand-swap chimeras: {chimera_stats['n_chimeric']:,} / "
              f"{chimera_stats['n_reads_checked']:,} checked reads "
              f"(mode={args.chimeras}; "
              f"{chimera_stats['n_records_affected']:,} records)", file=sys.stderr)
        if chimera_stats.get('split'):
            from fiberhmm.inference.chimeras import chimera_summary_lines
            for line in chimera_summary_lines(chimera_stats, indent='    ')[1:]:
                print(line, file=sys.stderr)
    elif mode == 'daf':
        from fiberhmm.inference.chimeras import empty_chimera_stats
        chimeric_names = set()
        chimera_stats = empty_chimera_stats('keep', args.chimera_min_seg,
                                           args.chimera_purity)
    return working_input, chimeric_names, chimera_stats, chimera_tmpdir


def main(chemistry_error_hint=None):
    """Console entry point. ``chemistry_error_hint`` rewrites a chemistry
    conflict message for a wrapper whose own options differ (the deprecated
    fiberhmm-apply alias has no --replace-chemistry)."""
    args = parse_args()
    from fiberhmm.inference.read_filters import MostlyUnmappedError
    from fiberhmm.inference.region_planning import RegionPlanError
    from fiberhmm.inference.region_resume import (
        ResumeRefused, RunInterrupted, interrupt_on_terminate,
    )
    from fiberhmm.inference.worker_results import WorkerFailureError

    try:
        # SIGTERM/SIGHUP unwind like Ctrl-C: temporaries are removed, the
        # region-parallel work directory is kept for --resume, and nothing is
        # half-published.
        with interrupt_on_terminate():
            _main(args)
    except KeyboardInterrupt as exc:
        signum = getattr(exc, 'signum', None) if isinstance(exc, RunInterrupted) else None
        print("error: interrupted" + (f" (signal {signum})" if signum else "") +
              "; no output was published", file=sys.stderr)
        sys.exit(128 + (signum or 2))
    except ResumeRefused as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    except ChemistryConflictError as exc:
        message = str(exc)
        if chemistry_error_hint is not None:
            message = chemistry_error_hint(message)
        print(f"error: {message}", file=sys.stderr)
        sys.exit(2)
    except RegionPlanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    except (WorkerFailureError, MostlyUnmappedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


def _refuse_call_path_aliases(args, apply_model_path=None, recall_model_path=None):
    """Refuse outputs that would destroy an input (the resolved models
    included), another output, or be deleted with the region-parallel work
    directory (exit 2).

    The work directory is removed after a successful run unless
    --keep-work-dir is given; then nothing the user keeps may live in it.
    QC and DAF SNP files are written beside their prefixes and are checked as
    outputs too.
    """
    from pathlib import Path

    from fiberhmm.cli.common import refuse_path_aliases

    deleted_dirs = {}
    if args.region_parallel and args.output != '-' and not args.keep_work_dir:
        from fiberhmm.inference.region_resume import default_work_dir
        deleted_dirs['--work-dir'] = (
            args.work_dir or str(default_work_dir(args.output)))
    outputs = [('--output', args.output),
               ('--progress-json', args.progress_json),
               ('--dedup-stats-tsv', args.dedup_stats_tsv)]
    if args.output != '-':
        stem = Path(args.output).parent / 'qc' / Path(args.output).with_suffix('').name
        # Recurrent-SNP files: when forced, or when the automatic DddA/DddB
        # screen may run (see auto_snp in _main).
        if args.daf_call_snps or (args.daf_call_snps is None and args.daf_snp_mask is None
                                  and args.enzyme in ('ddda', 'dddb')):
            snp_prefix = args.daf_snp_output_prefix or f"{stem}.daf_snps"
            suffixes = ('.bed', '.vcf', '.json', '.amplicons.tsv')
            if args.chimera_detector == 'strict':
                suffixes += ('.chimera_error_mask.bed', '.chimera_error_mask.json')
            outputs += [('--daf-snp-output-prefix', snp_prefix + suffix)
                        for suffix in suffixes]
        if args.qc:
            qc_prefix = args.qc_output_prefix or str(stem)
            outputs += [('--qc-output-prefix', qc_prefix + suffix)
                        for suffix in ('.qc.json', '.qc.tsv', '.qc.png', '.qc.pdf',
                                       '.qc.curves.json')]
    refuse_path_aliases(
        'fiberhmm-call',
        inputs=[('--input', args.input), ('--model', args.model),
                ('--model', apply_model_path),
                ('--recall-model', args.recall_model),
                ('--recall-model', recall_model_path),
                ('--reference', args.reference),
                ('--daf-snp-mask', args.daf_snp_mask)],
        outputs=outputs,
        deleted_dirs=deleted_dirs,
    )


def _main(args):
    from fiberhmm.io.annotation_frame import warn_if_frames_unverifiable
    warn_if_frames_unverifiable(str(args.input) if args.input is not None else None)
    stdout_mode = (args.output == '-')
    using_bundled_model = args.model is None

    if stdout_mode:
        sys.stdout = sys.stderr  # informational prints → stderr, BAM → real stdout

    if args.cores == 0:
        import multiprocessing
        args.cores = multiprocessing.cpu_count()
    if args.cores < 0:
        print("error: --cores must be >= 0 (0 = all CPUs)", file=sys.stderr)
        sys.exit(2)

    if args.resume and not args.region_parallel:
        # Only the region-parallel pipeline keeps per-region results; an
        # indexed, aligned file input can always take that path.
        if args.input == '-' or args.output == '-' or \
                _input_is_indexed_and_aligned(args.input) != (True, True):
            print("error: --resume needs the region-parallel pipeline (a "
                  "coordinate-sorted, indexed BAM file in and a BAM file out); "
                  "streaming runs (stdin/stdout, unindexed or unaligned input) "
                  "cannot resume.", file=sys.stderr)
            sys.exit(2)
        print("  NOTE: --resume implies --region-parallel.", file=sys.stderr)
        args.region_parallel = True
    if (args.work_dir or args.keep_work_dir) and not args.region_parallel:
        print("error: --work-dir/--keep-work-dir apply to --region-parallel runs",
              file=sys.stderr)
        sys.exit(2)

    if args.region_parallel:
        if args.input == '-' or args.output == '-':
            print("error: --region-parallel requires file I/O "
                  "(input must be indexed BAM, not stdin; output cannot be stdout).",
                  file=sys.stderr)
            sys.exit(1)
        from fiberhmm.inference.region_planning import (
            plan_region_work,
            require_indexed_bam,
        )
        # Fail in a second, before dedup/SNP/NRL passes, on an unusable input
        # or an empty/unknown region selection.
        require_indexed_bam(args.input)
        plan_region_work(args.input, args.region_size, args.skip_scaffolds,
                         set(args.chroms) if args.chroms else None)

    # A missing --seq is inferred from the input's own evidence (and refused
    # on conflicting evidence) before any model is chosen.
    resolve_platform_argument(args, args.input, tool='fiberhmm-call')
    # Recorded in the output @PG DS (basecaller=... tokens).
    basecaller = resolve_basecaller_provenance(args, args.input, tool='fiberhmm-call')
    require_model_files('fiberhmm-call', ('-m/--model', args.model),
                        ('--recall-model', args.recall_model))
    refuse_region_options_without_region_parallel(args, 'fiberhmm-call')
    refuse_non_bam_output(args.output, 'fiberhmm-call')

    apply_model_path = _resolve_apply_model(args)
    # --hmm-only runs no recall pass: no recall table is read or declared.
    recall_model_path = None if args.hmm_only else _resolve_recall_model(args)
    # Before anything is written (everything above only reads).
    _refuse_call_path_aliases(args, apply_model_path, recall_model_path)

    # Resolve mode/k from model metadata
    _, model_k, model_mode = load_model_with_metadata(apply_model_path)
    inferred_mode = (
        get_observation_mode(
            args.enzyme, args.seq, warn_missing_seq=False
        )
        if using_bundled_model else None
    )
    metadata_mode_aliases = (
        get_metadata_mode_aliases(
            args.enzyme, args.seq, warn_missing_seq=False
        )
        if using_bundled_model else ()
    )
    try:
        mode = resolve_observation_mode(
            model_mode,
            inferred_mode=inferred_mode,
            explicit_mode=args.mode,
            source_label=(
                f"bundled {args.enzyme} model"
                if using_bundled_model else "custom model"
            ),
            metadata_mode_aliases=metadata_mode_aliases,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    if not using_bundled_model:
        refuse_model_enzyme_assay_conflict(
            model_mode, args.enzyme, args.seq, args.model,
            tool='fiberhmm-call', explicit_mode=args.mode)
    k = args.context_size or int(model_k or 3)
    # Chemistry: reconcile with the input's declaration now (file input) so a
    # conflicting re-call fails in a second, before dedup/SNP/NRL passes, and
    # so every enzyme-dependent default below (CpG masking, DAF run mask, ML
    # threshold, dedup/SNP screens, nucleosome policy, TF presets) follows the
    # effective chemistry. A custom -m without --enzyme inherits the input's
    # enzyme/platform when the observation mode matches; the model paths
    # resolved above stay in use. A stdin header cannot be read in advance:
    # the pg_record marker then refuses a late enzyme inheritance.
    input_header = None
    if args.input != '-':
        import pysam
        with pysam.AlignmentFile(args.input, 'rb', check_sq=False) as _bam:
            input_header = _bam.header
    run_chemistry = resolve_effective_chemistry(
        args, mode, input_header, apply_model_path, recall_model_path,
        replace=args.replace_chemistry, tool='fiberhmm-call',
    )
    ddda_mcg = _configure_ddda_mcg(args, mode)
    # DddA CpG-aware recall: the same default policy as recall-tfs/-nucs.
    try:
        cpg_mask_policy = resolve_cpg_mask_policy(
            args.use_m5c, args.cpg_mask_policy, args.enzyme, mode,
            tool='fiberhmm-call')
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    from fiberhmm.nuc.records import cpg_mask_source_ds
    # '' for the default (headers unchanged); explicit when the user set it
    cpg_source_ds = cpg_mask_source_ds(args.use_m5c, args.cpg_mask_policy, False,
                                       cpg_mask_policy)
    _validate_model_context(
        [('apply model', apply_model_path), ('recall model', recall_model_path)],
        k,
    )

    explicit_dedup = args.dedup
    if explicit_dedup is False and (args.dedup_collapse or args.dedup_flag_only):
        print(
            "error: --no-dedup conflicts with --dedup-collapse/--dedup-flag-only",
            file=sys.stderr,
        )
        sys.exit(2)
    args.dedup = _resolve_dedup(args, mode)
    if args.dedup and args.input == '-':
        if explicit_dedup is True or args.dedup_collapse or args.dedup_flag_only:
            print("error: --dedup requires a file input (it two-passes the BAM to "
                  "fingerprint reads); cannot dedup a stdin stream.", file=sys.stderr)
            sys.exit(1)
        print(
            "  automatic DAF dedup skipped for stdin; save the input BAM or use "
            "--no-dedup to silence this note.",
            file=sys.stderr,
        )
        args.dedup = False

    # Enzyme preset for recall params
    preset = ENZYME_PRESETS.get(args.enzyme, {}) if args.enzyme else {}
    min_llr = args.min_llr if args.min_llr is not None else preset.get('min_llr', 5.0)
    uplift = args.emission_uplift if args.emission_uplift is not None \
             else preset.get('emission_uplift', 1.0)

    # Nucleosome recaller: ON by default for all enzymes. DddA uses a dedicated
    # radial dyad model plus a context-aware phase-marginal edge posterior (the
    # accessible-cut Kadane split shatters DddA nucleosomes because DddA
    # deaminates inside them); other enzymes use the Kadane split. Explicit
    # --recall-nucs / --no-recall-nucs wins.
    if args.hmm_only:
        recall_nucs = False
    elif args.recall_nucs is None:
        recall_nucs = True
        if args.enzyme == 'ddda':
            print(
                "  NOTE: DddA nucleosome recall uses phase-aware radial "
                "inference (bundled ddda_nuc_profile.json).",
                file=sys.stderr,
            )
    else:
        recall_nucs = bool(args.recall_nucs)
    nuc_recall_policy = _resolve_nuc_recall_policy(args, mode)
    _refuse_nuc_recall_policy(nuc_recall_policy, args.enzyme, 'fiberhmm-call')

    # Fast-fail sniff for DAF mode BEFORE any BAM scanning (e.g. --phase-nrl
    # auto estimation): the DAF path needs R/Y in the stored sequence, MD tags,
    # or --reference. If none are available every read is silently skipped, so
    # error out in under a second with an actionable message.
    if mode == 'daf' and args.input != '-':
        _check_daf_inputs(args.input, args.reference)

    if args.daf_call_snps and args.daf_snp_mask:
        print(
            "error: use either --daf-call-snps or --daf-snp-mask, not both",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.daf_mask_runs:
        if mode != 'daf':
            print("error: --daf-mask-runs requires --mode daf", file=sys.stderr)
            sys.exit(2)
        if args.daf_mask_runs < 2:
            print("error: --daf-mask-runs must be 0 (off) or >= 2", file=sys.stderr)
            sys.exit(2)
    # Unset means the chemistry default (DddA: keep-one on runs >= 2). Always
    # configure explicitly (also 0) so an inherited environment value cannot leak in.
    from fiberhmm.core.bam_reader import configure_daf_run_mask, resolve_daf_run_mask
    args.daf_mask_runs, args.daf_run_policy = resolve_daf_run_mask(
        args.daf_mask_runs if mode == 'daf' else 0, args.daf_run_policy, args.enzyme)
    configure_daf_run_mask(args.daf_mask_runs, args.daf_run_policy)
    from fiberhmm.inference.engine import configure_daf_unaligned_mask
    configure_daf_unaligned_mask(bool(args.daf_mask_unaligned) and mode == 'daf')
    if (args.daf_call_snps or args.daf_snp_mask) and mode != 'daf':
        print("error: DAF SNP masking requires --mode daf", file=sys.stderr)
        sys.exit(2)
    if args.daf_call_snps and (args.input == '-' or args.output == '-'):
        print(
            "error: --daf-call-snps requires file input and output (two-pass pre-call)",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.dedup_collapse and args.dedup_flag_only:
        print(
            "error: --dedup-collapse conflicts with --dedup-flag-only",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.dedup_max_end_diff < 0:
        print("error: --dedup-max-end-diff must be non-negative", file=sys.stderr)
        sys.exit(2)

    # ML threshold: explicit value, else the resolved chemistry's preset
    # (Hia5 Nanopore 248, otherwise 128). Integrated dedup follows it.
    args.prob_threshold = _resolve_call_prob_threshold(args, run_chemistry)
    if args.dedup_prob_threshold is None:
        args.dedup_prob_threshold = args.prob_threshold

    process_unmapped, unmapped_reason = _resolve_process_unmapped(args)
    if unmapped_reason:
        print(f"  NOTE: calling unmapped reads ({unmapped_reason}); pass "
              "--no-process-unmapped to pass them through instead.",
              file=sys.stderr)
    if args.region_parallel and args.process_unmapped:
        print("  NOTE: --region-parallel passes unmapped reads through uncalled; "
              "use the streaming pipeline to call them.", file=sys.stderr)

    # A missing output directory is created (as apply/extract/recall do)
    # before the dedup, SNP and region-parallel temporaries that live in it.
    if not stdout_mode:
        from fiberhmm.inference.bam_output import ensure_parent_dir
        try:
            ensure_parent_dir(args.output)
        except OSError as exc:
            print(f"error: cannot create the output directory for "
                  f"{args.output}: {exc}", file=sys.stderr)
            sys.exit(2)

    snp_mask_path = args.daf_snp_mask
    snp_report_path = None
    snp_mask_sites = 0

    # PCR dedup runs FIRST (DAF only); everything after it runs under a
    # finally that removes the dedup temp BAM however the run ends.
    working_input = args.input
    dedup_tmp = None
    insert_tmpdir = None
    insert_summary = None
    dedup_stats = None
    chimera_tmpdir = None
    chimera_stats = None
    chimeric_names = None
    chimera_filter_config = None
    error_mask_info = None
    if mode == 'daf':
        from fiberhmm.inference import engine as _engine
        chimera_filter_config = dict(_engine._DAF_CHIMERA_CFG)
    try:
        # PCR dedup runs FIRST (DAF only). The default mark/retain mode preserves
        # every molecule record while flagging non-representatives; explicit
        # collapse mode keeps one representative per cluster. working_input feeds
        # every downstream stage.
        if args.dedup:
            if mode != 'daf':
                print(f"  NOTE: --dedup applies to DAF-seq (ddda/dddb) deamination "
                      f"data; mode is {mode!r} (fiber-seq has no deamination) -- "
                      f"skipping dedup.", file=sys.stderr)
            else:
                dedup_tmp, dedup_stats = _dedup_input_first(
                    args.input, args.output, args.dedup_min_jaccard,
                    not args.dedup_collapse, args.io_threads, args.region_parallel,
                    min_deam=args.dedup_min_deam,
                    prob_threshold=args.dedup_prob_threshold,
                    ignore_strand=args.dedup_ignore_strand,
                    stats_tsv=args.dedup_stats_tsv,
                    max_end_diff=args.dedup_max_end_diff)
                if dedup_tmp is not None:
                    working_input = dedup_tmp

        # Recurrent SNP discovery follows deduplication so PCR copies cannot
        # inflate opposite-direction support. In nondestructive mark/retain mode,
        # the SNP caller ignores 0x400 records while FiberHMM still annotates them.
        run_snp_discovery = args.daf_call_snps is True
        snp_preflight = None
        # Why this run has no chimera error mask (it needs the SNP screen).
        error_mask_reason = None
        if args.daf_snp_mask is not None:
            error_mask_reason = 'no DAF SNP screen (user-supplied --daf-snp-mask)'
        elif args.daf_call_snps is False:
            error_mask_reason = 'no DAF SNP screen (--no-daf-call-snps)'
        elif args.daf_call_snps is None and args.enzyme not in ('ddda', 'dddb'):
            error_mask_reason = 'no automatic DAF SNP screen for this enzyme'
        auto_snp = (
            args.daf_call_snps is None
            and snp_mask_path is None
            and mode == 'daf'
            and args.enzyme in ('ddda', 'dddb')
        )
        if auto_snp and args.input != '-' and args.output != '-':
            snp_preflight = _daf_snp_depth_preflight(
                args.input,
                min_mapq=args.min_mapq,
                min_local_depth=2 * args.daf_snp_min_depth,
            )
            run_snp_discovery = bool(snp_preflight['run'])
            if not run_snp_discovery:
                error_mask_reason = 'DAF SNP screen skipped (low-depth preflight)'
            if run_snp_discovery:
                print(
                    "  automatic DAF SNP screen enabled: bounded preflight found "
                    f"{snp_preflight['reason'].replace('_', ' ')} support "
                    f"(estimated genome coverage {snp_preflight['estimated_genome_coverage']:.2f}x; "
                    f"max start-bin reads {snp_preflight['max_alignment_start_bin_reads']:,}; "
                    f"targeted-bin fraction {100 * snp_preflight['supported_start_bin_fraction']:.1f}%; "
                    f"max sampled depth {snp_preflight['max_local_depth']:,}) "
                    f"within {snp_preflight['records_examined']:,} records",
                    file=sys.stderr,
                )
            else:
                print(
                    "  automatic DAF SNP screen skipped: bounded preflight found "
                    f"estimated genome coverage {snp_preflight['estimated_genome_coverage']:.2f}x; "
                    f"max start-bin reads {snp_preflight['max_alignment_start_bin_reads']:,}; "
                    f"targeted-bin fraction "
                    f"{100 * snp_preflight['supported_start_bin_fraction']:.1f}% "
                    f"(targeted trigger requires >= {snp_preflight['minimum_depth']:,} "
                    "start-bin reads and >= 10% of eligible reads); "
                    "isolated sampled depth peak "
                    f"{snp_preflight['max_local_depth']:,} "
                    f"({snp_preflight['records_examined']:,} records examined)",
                    file=sys.stderr,
                )
        elif auto_snp:
            error_mask_reason = 'DAF SNP screen skipped (stdin/stdout)'
            print(
                "  automatic DAF SNP screen skipped for stdin/stdout; use saved BAMs "
                "or force --daf-call-snps with file input/output",
                file=sys.stderr,
            )

        if run_snp_discovery:
            from pathlib import Path

            from fiberhmm.daf.snps import (
                call_opposite_conversion_snps,
                chimera_error_mask_paths,
                write_snp_outputs,
            )
            from fiberhmm.inference.chimeras import detector_config, error_mask_summary

            strict_chimeras = args.chimera_detector == 'strict'
            build_error_mask = bool(strict_chimeras and args.chimera_error_mask
                                    and args.chimeras != 'keep')

            output_path = Path(args.output)
            snp_prefix = args.daf_snp_output_prefix or str(
                output_path.parent / 'qc' / f"{output_path.with_suffix('').name}.daf_snps"
            )
            snp_policy_name = describe_snp_threshold_policy(
                args.daf_snp_min_fraction,
                args.daf_snp_min_depth,
                args.daf_snp_min_alt_fibers,
            )['name']
            print(
                "\n  --daf-call-snps: two-pass opposite-conversion SNP discovery "
                f"after deduplication (policy={snp_policy_name}; "
                f"fraction >= {args.daf_snp_min_fraction:g}, "
                f"depth >= {args.daf_snp_min_depth}, "
                f"alternate fibers >= {args.daf_snp_min_alt_fibers})...",
                file=sys.stderr,
            )
            snp_payload = call_opposite_conversion_snps(
                working_input,
                min_fraction=args.daf_snp_min_fraction,
                min_depth=args.daf_snp_min_depth,
                min_alt_fibers=args.daf_snp_min_alt_fibers,
                min_dominant_events=args.daf_snp_min_dominant_events,
                min_dominant_purity=args.daf_snp_min_dominant_purity,
                min_mapq=args.min_mapq,
                reference_fasta=args.reference,
                min_amplicon_reads=args.daf_snp_min_amplicon_reads,
                all_snvs=getattr(args, 'daf_snp_all_snvs', False),
                # The chimera error mask is a strict-detector option only.
                **(dict(chimera_error_mask=build_error_mask,
                        chimera_detector=detector_config(**_chimera_detector_args(args)))
                   if strict_chimeras else {}),
            )
            if dedup_tmp is not None:
                snp_payload["input"] = str(Path(args.input).resolve())
                snp_payload["preprocessing"] = {
                    "deduplication": "duplicate-flagged records excluded",
                    "mode": dedup_stats.get("mode") if dedup_stats else None,
                    "n_duplicates": (
                        int(dedup_stats.get("n_duplicates", 0)) if dedup_stats else 0
                    ),
                    "max_end_diff_bp": args.dedup_max_end_diff,
                }
            snp_payload = write_snp_outputs(snp_payload, snp_prefix)
            snp_mask_path = snp_payload['outputs']['bed']
            snp_report_path = snp_payload['outputs']['json']
            snp_mask_sites = snp_payload['n_called_snps']
            print(
                f"  DAF SNP mask: {snp_mask_sites:,} sites -> {snp_mask_path}",
                file=sys.stderr,
            )
            if build_error_mask:
                error_mask_info = error_mask_summary(
                    True, snp_payload['outputs']['chimera_error_mask_bed'],
                    snp_payload['chimera_error_mask']['n_sites'])
                print(
                    f"  DAF chimera error mask (chimera test only): "
                    f"{error_mask_info['n_sites']:,} sites -> {error_mask_info['bed']}",
                    file=sys.stderr,
                )
            elif strict_chimeras:
                # A mask from an earlier run at this prefix must not be
                # mistaken for this run's.
                for stale in chimera_error_mask_paths(snp_prefix):
                    stale.unlink(missing_ok=True)
        elif snp_mask_path:
            from fiberhmm.daf.snps import mask_summary

            snp_mask_sites = mask_summary(snp_mask_path)['n_sites']

        if (mode == 'daf' and args.chimera_detector == 'strict'
                and error_mask_info is None):
            from fiberhmm.inference.chimeras import error_mask_summary
            error_mask_info = error_mask_summary(
                bool(args.chimera_error_mask), None, 0, error_mask_reason)

        # Split mode cuts chimeras before the insert-consensus pre-pass, so
        # the pieces (new records) carry their own insertion evidence.
        chimeras_prepared = False
        if mode == 'daf' and args.chimeras == 'split':
            working_input, chimeric_names, chimera_stats, chimera_tmpdir = \
                _chimera_pass(args, mode, working_input, snp_mask_path,
                              process_unmapped, stdout_mode, error_mask_info)
            chimeras_prepared = True

        # DAF insert consensus pre-pass: evidence inside insertions carried by
        # enough reads (fiberhmm.daf.insert_consensus).
        insert_evidence_path = None
        insert_state = 'n/a' if mode != 'daf' else 'off'
        if (mode == 'daf' and args.daf_insert_consensus == 'auto'
                and args.daf_mask_unaligned
                and args.input != '-' and args.output != '-'):
            import tempfile as _tempfile
            from pathlib import Path as _Path
            from fiberhmm.daf.insert_consensus import run_insert_consensus_prepass
            output_path = _Path(args.output)
            # Evidence goes to a private temp dir (removed in the finally);
            # the report is published with the other outputs once calling
            # has succeeded.
            insert_tmpdir = _tempfile.mkdtemp(prefix='.fiberhmm_insert_')
            insert_report_path = str(
                output_path.parent / 'qc'
                / f"{output_path.with_suffix('').name}.insert_consensus.json")
            insert_summary = run_insert_consensus_prepass(
                working_input, insert_tmpdir,
                min_carriers=args.daf_insert_min_carriers,
                min_mapq=args.min_mapq,
                reference=getattr(args, 'reference', None),
                # The caller's strand inputs, so each carrier's evidence is
                # on the strand it will be called on.
                snp_mask_path=snp_mask_path,
                prob_threshold=args.prob_threshold,
            )
            insert_state = (f"on/{insert_summary['clusters_used']}of"
                            f"{insert_summary['clusters']}/min{args.daf_insert_min_carriers}")
            if insert_summary['clusters_used']:
                insert_evidence_path = insert_summary['evidence_path']
            print(
                f"  DAF insert consensus: {insert_summary['clusters_used']:,} of "
                f"{insert_summary['clusters']:,} insertion clusters (>= 50 bp) with "
                f">= {args.daf_insert_min_carriers} carriers; "
                f"{insert_summary['records_with_evidence']:,} records re-encoded "
                f"(report: {insert_report_path})",
                file=sys.stderr,
            )
        from fiberhmm.inference.engine import configure_daf_insert_evidence
        configure_daf_insert_evidence(insert_evidence_path)

        if not chimeras_prepared:
            working_input, chimeric_names, chimera_stats, chimera_tmpdir = \
                _chimera_pass(args, mode, working_input, snp_mask_path,
                              process_unmapped, stdout_mode, error_mask_info)

        # Pass 1 owns detection. Payload construction and worker extraction
        # must not repeat its mismatch walk on the surviving reads.
        if mode == 'daf':
            from fiberhmm.inference.engine import configure_daf_chimera_filter
            configure_daf_chimera_filter(False)

        # Freeze DddA nucleosome-refinement likelihoods independently of the TF
        # recaller. NRL estimation is part of nuc refinement and uses the same
        # frozen model.
        nuc_model_path = _resolve_nuc_model_path(args, recall_nucs)
        _validate_model_context([('nuc likelihood model', nuc_model_path)], k)

        # Resolve the Pass-2 phase prior: off / auto-estimate / fixed bp.
        phase_nrl = _resolve_phase_nrl(
            args, apply_model_path, nuc_model_path or recall_model_path, mode, k,
            recall_nucs, working_input, include_unmapped=process_unmapped,
            precall_chimeric_names=chimeric_names)

        # DddA uses phase-aware radial nucleosome inference; other enzymes use the
        # accessible-cut Kadane split (no profile).
        nuc_profile_path = _resolve_nuc_profile_path(args, recall_nucs)
        nuc_profile_identity = _nuc_profile_identity(nuc_profile_path)
        nuc_profile_sha256 = _nuc_profile_sha256(nuc_profile_path)
        derived_tf_max_edge_ambiguity = _resolve_derived_tf_edge_gap(
            args, recall_nucs)

        # @PG provenance for the output BAM header. The molecular-frame note is the
        # important bit: it tells downstream tools how to read ns/nl/as/al/MA.
        import fiberhmm as _fh
        chimera_state = args.chimeras
        chimera_filter_state = 'off' if args.chimeras == 'keep' else 'on'
        dedup_state = ('off' if not args.dedup or dedup_tmp is None
                       else f"j{args.dedup_min_jaccard}"
                            f"{'/collapse' if args.dedup_collapse else '/mark'}"
                            f"/ends{args.dedup_max_end_diff}")
        unaligned_state = ('n/a' if mode != 'daf'
                           else ('on' if args.daf_mask_unaligned else 'off'))
        if error_mask_info is None or args.chimeras == 'keep':
            error_mask_state = 'n/a'
        elif error_mask_info['applied']:
            error_mask_state = f"on/{error_mask_info['n_sites']}sites"
        else:
            error_mask_state = 'off' if not error_mask_info['enabled'] else 'unavailable'
        duplicates_state = ((chimera_stats or {}).get('n_duplicates_excluded', 'n/a')
                            if mode == 'daf' and args.chimeras != 'keep' else 'n/a')
        # The strict detector records itself and its settings in DS; a DS
        # without chimera_detector= is the standard detector (as in 3.2.1).
        strict_chimera_ds = (
            f"chimera_detector=strict "
            f"chimera_min_baseq={args.chimera_min_baseq} "
            f"chimera_min_span={args.chimera_min_span} "
            f"chimera_multi_switch={'on' if args.chimera_multi_switch else 'off'} "
            f"chimera_error_mask={error_mask_state} "
            + (f"chimera_error_mask_sha256={error_mask_info['sha256']} "
               if error_mask_state.startswith('on/') and error_mask_info.get('sha256')
               else '')
            + f"chimera_duplicates_excluded={duplicates_state} "
            if args.chimera_detector == 'strict' else '')
        if mode == 'daf' and args.chimeras == 'split':
            # Split mode records how its breakpoints were found (the
            # flagging detector is chimera_detector=, absent = standard).
            from fiberhmm.inference.chimeras import SPLIT_BREAKPOINTS_DS
            strict_chimera_ds += ("chimera_split_breakpoints="
                                  f"{SPLIT_BREAKPOINTS_DS[args.chimera_detector]} ")
        if snp_mask_path:
            snp_state = f"on/{snp_mask_sites}sites"
        elif snp_preflight is not None and not snp_preflight['run']:
            snp_state = f"auto-skip/depth{snp_preflight['max_local_depth']}"
        else:
            snp_state = 'off'
        chemistry = _chemistry_declaration(
            args,
            mode,
            apply_model_path,
            # The recall pass reuses the apply table when no separate recall
            # table is resolved: record the table it actually read.
            # --hmm-only runs no recall pass: no recall table is declared.
            None if args.hmm_only else (recall_model_path or apply_model_path),
            nuc_profile_identity,
            nuc_profile_sha256,
            nuc_model_path=nuc_model_path,
        )
        if args.hmm_only:
            # A different description and token set, so readers (recall,
            # FiberBrowser) can tell an hmm-only call; the default call's DS
            # below is unchanged.
            call_ds = (
                f"FiberHMM HMM-only call (no nucleosome or TF recall: nuc = HMM "
                f"footprints >= {args.nuc_min_size} bp, tf = smaller HMM footprints, "
                f"msp = {'the input MSPs' if args.keep_msps else 'HMM MSPs'}); "
                f"coord=molecular "
                f"(ns/nl/as/al/MA in molecular original-fiber coordinates); "
                f"hmm_only=on mode={mode} enzyme={args.enzyme or 'custom'} "
                f"prob_threshold={args.prob_threshold} "
                f"primary_only={'on' if args.alignments == 'primary' else 'off'} alignments={args.alignments} "
                f"recall_nucs=False tf_recall=off "
                f"nuc_min_size={args.nuc_min_size} msp_min_size={args.msp_min_size} "
                f"msps={'kept' if args.keep_msps else 'hmm'} "
                f"chimera_filter={chimera_filter_state} chimeras={chimera_state} "
                f"chimera_min_seg={args.chimera_min_seg} chimera_purity={args.chimera_purity} "
                + strict_chimera_ds +
                f"dedup={dedup_state} "
                f"daf_snp_mask={snp_state} "
                f"daf_unaligned_mask={unaligned_state} "
                f"daf_insert_consensus={insert_state} "
                f"daf_run_mask={('>=' + str(args.daf_mask_runs) + '/' + args.daf_run_policy) if args.daf_mask_runs else 'off'} "
                f"cpg_mask={cpg_mask_policy or 'off'} "
                f"hmm_cpg_mask={'on' if cpg_mask_policy and not ddda_mcg else 'off'}"
                + cpg_source_ds
                + " " + basecaller_ds_tokens(basecaller))
        pg_record = {
            'PN': 'fiberhmm-call',
            'VN': getattr(_fh, '__version__', 'unknown'),
            'CL': ' '.join(sys.argv),
            # Machine-readable, versioned scientific metadata. Unlike DS/CL this
            # contract is safe for downstream model selection and survives renames.
            'chemistry': (
                reconcile_chemistry(
                    input_header, chemistry, replace=args.replace_chemistry,
                    tool='fiberhmm-call',
                )
                if input_header is not None else chemistry
            ),
            REPLACE_CHEMISTRY_KEY: bool(args.replace_chemistry),
            DEFAULTS_RESOLVED_KEY: True,
            # The `coord=molecular` token is a stable, version-independent contract
            # for downstream consumers (e.g. FiberBrowser) to detect that ns/nl/as/al
            # and MA are in molecular (original-fiber) frame -- keep the exact token.
            'DS': call_ds if args.hmm_only else (f"FiberHMM fused apply+recall; coord=molecular "
                   f"(ns/nl/as/al/MA in molecular original-fiber coordinates); "
                   f"mode={mode} enzyme={args.enzyme or 'custom'} "
                   f"prob_threshold={args.prob_threshold} "
                   f"primary_only={'on' if args.alignments == 'primary' else 'off'} alignments={args.alignments} "
                   f"tf_decoder={TF_DECODER_VERSION} tf_interval_penalty={min_llr} "
                   f"recall_nucs={recall_nucs} "
                   f"nuc_recall_policy={nuc_recall_policy} "
                   f"nuc_profile={nuc_profile_identity or 'off'} "
                   f"nuc_sha256={nuc_profile_sha256 or 'off'} "
                   f"phase_nrl={phase_nrl} "
                   f"ddda_derived_tf_edge_gap="
                   f"{derived_tf_max_edge_ambiguity if derived_tf_max_edge_ambiguity is not None else 'off'} "
                   f"chimera_filter={chimera_filter_state} chimeras={chimera_state} "
                   f"chimera_min_seg={args.chimera_min_seg} chimera_purity={args.chimera_purity} "
                   + strict_chimera_ds +
                   f"dedup={dedup_state} "
                   f"daf_snp_mask={snp_state} "
                   f"daf_unaligned_mask={unaligned_state} "
                   f"daf_insert_consensus={insert_state} "
                   f"daf_run_mask={('>=' + str(args.daf_mask_runs) + '/' + args.daf_run_policy) if args.daf_mask_runs else 'off'} "
                   f"cpg_mask={cpg_mask_policy or 'off'} "
                   f"hmm_cpg_mask={'on' if cpg_mask_policy and not ddda_mcg else 'off'}"
                   + cpg_source_ds
                   # Basecaller provenance (fiberhmm.io.provenance.ds_tokens).
                   + " " + basecaller_ds_tokens(basecaller)),
        }

        if mode == 'daf':
            from fiberhmm.daf.replay import evidence_token
            from fiberhmm.daf.snps import load_snp_mask
            pg_record['DS'] += ' ' + evidence_token(
                load_snp_mask(snp_mask_path) if snp_mask_path else {},
                mask_unaligned=args.daf_mask_unaligned)

        mode_label = 'region-parallel' if args.region_parallel else 'streaming'
        if args.hmm_only:
            print(
                "\n=========================================================================\n"
                f"  fiberhmm-call --hmm-only — HMM calls, no recall ({mode_label})\n"
                f"  apply model:  {apply_model_path}\n"
                f"  mode={mode} k={k} enzyme={args.enzyme or 'custom'} "
                f"prob-threshold={args.prob_threshold} "
                f"alignments={args.alignments}\n"
                f"  nuc >= {args.nuc_min_size} bp, tf < {args.nuc_min_size} bp, "
                f"msp: {'kept from the input' if args.keep_msps else 'HMM'}\n"
                f"  cores={args.cores} io-threads={args.io_threads} "
                f"cpg_mask={cpg_mask_policy or 'off'} "
                f"daf_run_mask={('>=' + str(args.daf_mask_runs) + '/' + args.daf_run_policy) if args.daf_mask_runs else 'off'}"
                f"{' circular=on' if args.circular else ''}\n"
                "=========================================================================\n",
                file=sys.stderr,
            )
        else:
            print(
                "\n=========================================================================\n"
                f"  fiberhmm-call — fused apply + recall-tfs ({mode_label})\n"
                f"  apply model:  {apply_model_path}\n"
                f"  recall model: {recall_model_path or '(reuse apply model)'}\n"
                f"  nuc likelihood model: {nuc_model_path or '(reuse recall model)'}\n"
                f"  mode={mode} k={k} enzyme={args.enzyme or 'custom'} "
                f"prob-threshold={args.prob_threshold} "
                f"alignments={args.alignments}\n"
                f"  min_llr={min_llr} min_opps={args.min_opps} "
                f"unify_threshold={args.unify_threshold} uplift={uplift}\n"
                f"  tf-decoder={TF_DECODER_VERSION} interval-penalty={min_llr}\n"
                + (f"  nuc-recall-policy={nuc_recall_policy} phase-nrl={phase_nrl}\n"
                   if (nuc_recall_policy != 'conservative' or phase_nrl) else '') +
                f"  ddda-derived-tf-edge-gap="
                f"{derived_tf_max_edge_ambiguity if derived_tf_max_edge_ambiguity is not None else 'off'}\n"
                f"  cores={args.cores} io-threads={args.io_threads} "
                f"cpg_mask={cpg_mask_policy or 'off'} "
                f"daf_run_mask={('>=' + str(args.daf_mask_runs) + '/' + args.daf_run_policy) if args.daf_mask_runs else 'off'}"
                f"{' circular=on' if args.circular else ''}\n"
                "=========================================================================\n",
                file=sys.stderr,
            )

        also_write_legacy = resolve_write_legacy(args, 'fiberhmm-call')
        train_rids = _load_train_reads(args.train_reads)

        progress = None
        if args.progress_json:
            from fiberhmm.inference.region_resume import ProgressJSON
            progress = ProgressJSON(args.progress_json, 'fiberhmm-call')
        if args.region_parallel:
            chroms_set = set(args.chroms) if args.chroms else None
            from fiberhmm.inference.region_resume import (
                default_work_dir, input_identity, reference_identity,
            )
            # Everything that decides the called records, beyond what the
            # region pipeline records itself (its parameters, model digests
            # and region plan): --resume refuses a work directory made with
            # anything else.
            def run_identity(memo):
                return {
                    'tool': 'fiberhmm-call',
                    'fiberhmm_version': getattr(_fh, '__version__', 'unknown'),
                    'input': input_identity(args.input, memo),
                    'dedup': {
                        'applied': dedup_tmp is not None,
                        'min_jaccard': args.dedup_min_jaccard,
                        'collapse': bool(args.dedup_collapse),
                        'min_deam': args.dedup_min_deam,
                        'prob_threshold': args.dedup_prob_threshold,
                        'ignore_strand': bool(args.dedup_ignore_strand),
                        'max_end_diff': args.dedup_max_end_diff,
                    },
                    'chimeras': chimera_stats,
                    'daf_run_mask': [args.daf_mask_runs, args.daf_run_policy],
                    'daf_unaligned_mask': unaligned_state,
                    'daf_insert_consensus': [args.daf_insert_consensus,
                                             args.daf_insert_min_carriers],
                    'reference': reference_identity(args.reference, memo),
                    'process_unmapped': process_unmapped,
                }

            n_reads, n_fp = _process_bam_region_parallel_fused(
                input_bam=working_input,
                output_bam=args.output,
                apply_model_path=apply_model_path,
                recall_model_path=recall_model_path,
                train_rids=train_rids,
                edge_trim=args.edge_trim,
                circular=args.circular,
                mode=mode,
                context_size=k,
                msp_min_size=args.msp_min_size,
                nuc_min_size=args.nuc_min_size,
                min_mapq=args.min_mapq,
                prob_threshold=args.prob_threshold,
                min_read_length=args.min_read_length,
                with_scores=args.with_scores,
                min_llr=min_llr,
                min_opps=args.min_opps,
                unify_threshold=args.unify_threshold,
                emission_uplift=uplift,
                also_write_legacy=also_write_legacy,
                downstream_compat=args.downstream_compat,
                n_cores=args.cores,
                region_size=args.region_size,
                skip_scaffolds=args.skip_scaffolds,
                chroms=chroms_set,
                io_threads=args.io_threads,
                primary_only=args.alignments,
                ref_fasta_path=args.reference,
                recall_nucs=recall_nucs,
                split_min_llr=args.split_min_llr,
                split_min_opps=args.split_min_opps,
                nuc_recall_policy=nuc_recall_policy,
                precall_chimeric_names=chimeric_names,
                chimera_mode=args.chimeras,
                filter_chimeras=False,
                chimera_min_seg=args.chimera_min_seg,
                chimera_purity=args.chimera_purity,
                **_strict_worker_kwargs(args, error_mask_info),
                phase_nrl=phase_nrl,
                nuc_profile_path=nuc_profile_path,
                nuc_model_path=nuc_model_path,
                derived_tf_max_edge_ambiguity=derived_tf_max_edge_ambiguity,
                pg_record=pg_record,
                ddda_mcg=ddda_mcg,
                daf_snp_mask_path=snp_mask_path,
                daf_insert_evidence_path=insert_evidence_path,
                cpg_mask_policy=cpg_mask_policy,
                work_dir=args.work_dir or str(default_work_dir(args.output)),
                resume=args.resume,
                keep_work_dir=args.keep_work_dir,
                run_identity=run_identity,
                progress=progress,
                hmm_only=bool(args.hmm_only),
                keep_msps=bool(args.keep_msps),
            )
        else:
            if progress is not None:
                progress('start', output=None if stdout_mode else str(args.output),
                         regions_total=None)
            n_reads, n_fp = _process_bam_streaming_pipeline_fused(
                input_bam=working_input,
                output_bam=args.output,
                model_path=apply_model_path,
                recall_model_path=recall_model_path,
                train_rids=train_rids,
                edge_trim=args.edge_trim,
                circular=args.circular,
                mode=mode,
                context_size=k,
                msp_min_size=args.msp_min_size,
                nuc_min_size=args.nuc_min_size,
                min_mapq=args.min_mapq,
                prob_threshold=args.prob_threshold,
                min_read_length=args.min_read_length,
                with_scores=args.with_scores,
                min_llr=min_llr,
                min_opps=args.min_opps,
                unify_threshold=args.unify_threshold,
                emission_uplift=uplift,
                also_write_legacy=also_write_legacy,
                downstream_compat=args.downstream_compat,
                max_reads=args.max_reads,
                n_cores=args.cores,
                chunk_size=args.chunk_size,
                io_threads=args.io_threads,
                process_unmapped=process_unmapped,
                primary_only=args.alignments,
                ref_fasta_path=args.reference,
                recall_nucs=recall_nucs,
                split_min_llr=args.split_min_llr,
                split_min_opps=args.split_min_opps,
                nuc_recall_policy=nuc_recall_policy,
                precall_chimeric_names=chimeric_names,
                chimera_mode=args.chimeras,
                filter_chimeras=False,
                chimera_min_seg=args.chimera_min_seg,
                chimera_purity=args.chimera_purity,
                **_strict_worker_kwargs(args, error_mask_info),
                phase_nrl=phase_nrl,
                nuc_profile_path=nuc_profile_path,
                nuc_model_path=nuc_model_path,
                derived_tf_max_edge_ambiguity=derived_tf_max_edge_ambiguity,
                pg_record=pg_record,
                ddda_mcg=ddda_mcg,
                daf_snp_mask_path=snp_mask_path,
                cpg_mask_policy=cpg_mask_policy,
                # A run that skipped nearly everything as unmapped is an
                # error unless the user asked for pass-through explicitly.
                fail_on_mostly_unmapped=args.process_unmapped is not False,
                # region-parallel already indexes; streaming indexes the
                # temporary before publishing BAM + index together.
                index_output=not stdout_mode,
                hmm_only=bool(args.hmm_only),
                keep_msps=bool(args.keep_msps),
            )
            if progress is not None:
                progress('done', output=None if stdout_mode else str(args.output),
                         reads=n_reads, reads_with_footprints=n_fp)

        # The pre-footprinting dedup temp is no longer needed; removing it here
        # (as well as in the finally below) keeps the disk footprint low during QC.
        _remove_dedup_temp(dedup_tmp)

        if insert_summary is not None and not stdout_mode:
            from fiberhmm.daf.insert_consensus import write_report
            write_report(insert_summary, insert_report_path)

        # Preserve only aggregate full-run deduplication statistics beside QC.
        # This lets a later standalone/multi-BAM fiberhmm-qc reproduce the exact
        # duplication panel without rescanning or packaging molecule identities.
        if dedup_stats is not None and not stdout_mode:
            import json
            from pathlib import Path

            output_path = Path(args.output)
            dedup_report_path = (
                output_path.parent / "qc" /
                f"{output_path.with_suffix('').name}.dedup.json"
            )
            dedup_report_path.parent.mkdir(parents=True, exist_ok=True)
            dedup_payload = {
                "schema_version": 1,
                "method": "fiberhmm_deamination_fingerprint",
                "input": str(Path(args.input).resolve()),
                "output": str(output_path.resolve()),
                "statistics": dedup_stats,
            }
            temporary = dedup_report_path.with_name(dedup_report_path.name + ".tmp")
            temporary.write_text(json.dumps(dedup_payload, indent=2, sort_keys=True) + "\n")
            temporary.replace(dedup_report_path)
            print(f"  PCR deduplication report: {dedup_report_path}", file=sys.stderr)

        if not stdout_mode:
            import json
            from pathlib import Path
            report = Path(args.output).parent / 'qc' / (Path(args.output).stem + '.chimeras.json')
            if mode != 'daf':
                # Only remove after the new BAM has been published successfully.
                report.unlink(missing_ok=True)
            elif chimera_stats is not None and Path(args.output).is_file():
                from fiberhmm.io.run_state import sha256_file
                report.parent.mkdir(parents=True, exist_ok=True)
                temporary = report.with_suffix('.json.tmp')
                payload = {
                    'schema_version': 1,
                    'assay': {'mode': mode, 'enzyme': args.enzyme or 'custom'},
                    # Bind the report to the published BAM, including its @PG
                    # history, without changing resumable-call provenance.
                    'output_sha256': sha256_file(args.output),
                    'statistics': chimera_stats,
                }
                temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
                temporary.replace(report)

        # Bounded post-call QC. Indexed BAMs are sampled via random genomic
        # windows; unindexed outputs use a capped prefix reservoir. Sampling
        # is bounded; chimera report validation separately hashes the BAM.
        if args.qc:
            if stdout_mode:
                print("  NOTE: automatic QC skipped for stdout BAM output; run "
                      "fiberhmm-qc on the saved BAM.", file=sys.stderr)
            elif mode not in ('daf', 'pacbio-fiber', 'nanopore-fiber'):
                print(f"  NOTE: automatic QC is not defined for mode {mode!r}; "
                      "skipping.", file=sys.stderr)
            else:
                try:
                    from pathlib import Path

                    from fiberhmm.qc.core import run_qc

                    qc_prefix = args.qc_output_prefix
                    if qc_prefix is None:
                        output_path = Path(args.output)
                        qc_prefix = str(
                            output_path.parent / 'qc' / output_path.with_suffix('').name
                        )

                    run_qc(
                        input_path=args.output,
                        output_prefix=qc_prefix,
                        mode=mode,
                        enzyme=args.enzyme or 'auto',
                        # The automatic reference is locked to the resolved
                        # enzyme/platform combination selected for this call.
                        # Explicit profile overrides belong to standalone
                        # fiberhmm-qc, where they cannot silently mislabel a call.
                        reference_profile='auto',
                        reference_fasta=args.reference,
                        sample_reads=args.qc_sample_reads,
                        seed=args.qc_seed,
                        min_mapq=args.qc_min_mapq,
                        prob_threshold=args.prob_threshold,
                        snp_report_path=snp_report_path,
                        snp_mask_path=snp_mask_path,
                        snp_preflight_summary=snp_preflight,
                        dedup_run_summary=dedup_stats,
                        chimera_run_summary=chimera_stats,
                        # QC's light call (sampled reads without calls)
                        # resolves the CpG mask from this run's options.
                        use_m5c=args.use_m5c,
                        cpg_mask_policy=args.cpg_mask_policy,
                    )
                except Exception as exc:
                    # QC must never invalidate a successfully written callset.
                    print(f"  WARNING: automatic FiberHMM QC could not run: {exc}",
                          file=sys.stderr)

    finally:
        # Calling can run in-process beside QC or another caller. Its
        # decision-only pass must not leave detection disabled for them.
        if chimera_filter_config is not None:
            _engine.restore_daf_chimera_filter(chimera_filter_config)
        if chimera_tmpdir is not None:
            chimera_tmpdir.cleanup()
        _remove_dedup_temp(dedup_tmp)
        if insert_tmpdir is not None:
            import shutil as _shutil
            _shutil.rmtree(insert_tmpdir, ignore_errors=True)

if __name__ == '__main__':
    main()
