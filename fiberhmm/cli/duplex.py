#!/usr/bin/env python3
"""Validated sequence-free scorer used by unified ``fiberhmm-pair``.

The tool reads a coordinate-sorted, indexed FiberHMM-called DddA BAM and a
matching reference FASTA.  Within complete local overlap components it scores
CT/GA candidates from nucleosome lattice agreement, alignment geometry, and
raw plus component-residual non-CpG DddA protection profiles.  It never uses
A/T mismatch, haplotype, TF LLR, or a sequence veto.

Output reads retain their original sequence and annotations.  Primary reads
receive the standard pair status/partner tags plus model metadata:

    mt:A  P paired, U unresolved candidate, . no opposite-flavor candidate
    mp:Z  paired mate query name
    pm:A  D (sequence-identity-free duplex model)
    dm:i  standardized duplex-model decision score x1000
    mg:i  two-sided reciprocal margin x1000
    mv:Z  frozen model identifier

Reads flagged as PCR duplicates (0x400) are excluded from pairing and written
through without pair tags; the receipt counts them as ``duplicate_reads``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from itertools import chain
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pysam

from fiberhmm.models import DEFAULT_PROB_THRESHOLD
from fiberhmm import __version__
from fiberhmm.crossstrand.duplex import (
    DuplexModel,
    DuplexParams,
    FLAVOR_CT,
    STATUS_PAIRED,
    assign_duplex_pairs,
    build_pattern_feature,
    build_protection_profile,
    infer_call_layer,
    load_duplex_model,
)
from fiberhmm.crossstrand.pairing import (
    PairParams,
    SequenceScore,
    _sequence_assignment,
    _sequence_signature,
    _sequence_signature_from_md,
    read_flavor,
    score_pair,
)
from fiberhmm.io.bam_header import append_pg_record

_TAG_PARTNER = "mp"
_TAG_STATUS = "mt"
_TAG_METHOD = "pm"
_TAG_SCORE = "dm"
_TAG_MARGIN = "mg"
_TAG_MODEL = "mv"
_PAIR_TAGS = ("mp", "mc", "mg", "mt", "pm", "sb", "sd", "sr", "sg", "pa", "dm", "mv")


def _header_with_program(header, model_id: Optional[str], pairing_mode: str):
    descriptions = {
        "hybrid": "sequence-supported plus high-confidence sequence-free CT/GA pairing",
        "sequence-only": "sequence-supported CT/GA pairing",
        "sequence-free": "sequence-identity-free CT/GA pairing",
    }
    return append_pg_record(header, {
        "PN": "fiberhmm-pair",
        "VN": __version__,
        "CL": descriptions[pairing_mode],
        "DS": (
            f"mode={pairing_mode}; model={model_id or 'none'}; "
            "TF LLR and haplotype unused"
        ),
    })


def _selected_edge(result, left: int, right: int):
    return result.evidence.get((left, right)) or result.evidence.get((right, left))


def run_pairing(in_bam: str, out_bam: str, reference_path: Optional[str],
               params: Optional[DuplexParams] = None,
               sequence_params: Optional[PairParams] = None,
               model: Optional[DuplexModel] = None,
               model_path: Optional[str] = None,
               prob_threshold: int = 0,
               pairs_tsv: Optional[str] = None,
               receipt_json: Optional[str] = None,
               paired_only: bool = False,
               io_threads: int = 4,
               max_component: int = 10000,
               call_layer: str = "auto",
               create_index: bool = True,
               pairing_mode: str = "hybrid"):
    """Run unified CT/GA pairing and return its machine-readable receipt.

    ``hybrid`` (the public default) takes independently sequence-supported
    reciprocal pairs first, then adds non-conflicting pairs that pass the
    externally replicated sequence-free model margin. ``sequence-only`` is
    provided for analyses that require every accepted pair to have direct A/T
    support. ``sequence-free`` is retained internally for validation replay.
    """
    started = time.time()
    if pairing_mode not in {"hybrid", "sequence-only", "sequence-free"}:
        raise ValueError(f"unsupported pairing mode: {pairing_mode!r}")
    use_sequence = pairing_mode != "sequence-free"
    use_model = pairing_mode != "sequence-only"
    if use_model and not reference_path:
        raise ValueError(
            "default pairing requires --reference to enumerate non-CpG DddA "
            "opportunities; use --sequence-only to pair from FASTA or MD+CIGAR"
        )
    params = params or DuplexParams()
    sequence_params = sequence_params or PairParams(
        grid_bp=params.grid_bp,
        sigma_bp=params.dyad_sigma_bp,
        max_lag_bp=params.max_lag_bp,
        min_overlap_bp=params.min_overlap_bp,
        min_nucs=params.min_nucs,
    )
    with pysam.AlignmentFile(in_bam, "rb") as source:
        detected_call_layer = infer_call_layer(source.header)
    resolved_call_layer = detected_call_layer if call_layer == "auto" else call_layer
    if use_model:
        model = model or load_duplex_model(model_path, resolved_call_layer)
    resolved: Dict[Tuple[str, int], dict] = {}
    status_of: Dict[Tuple[str, int], str] = {}
    seen_names: Dict[str, str] = {}
    counts = Counter()
    pair_rows: List[dict] = []

    bam = pysam.AlignmentFile(in_bam, "rb")
    fasta = pysam.FastaFile(reference_path) if reference_path else None
    try:
        if bam.header.get("HD", {}).get("SO") != "coordinate":
            raise ValueError("fiberhmm-pair requires a coordinate-sorted BAM")
        if not bam.has_index():
            raise ValueError("fiberhmm-pair requires an indexed input BAM")
        fasta_names = set(fasta.references) if fasta is not None else set()
        component_number = 0

        for chromosome in bam.references:
            iterator = bam.fetch(chromosome)
            first = next(iterator, None)
            if first is None:
                continue
            if fasta is not None and chromosome not in fasta_names:
                raise ValueError(f"reference FASTA is missing BAM contig {chromosome!r}")
            bam_length = bam.get_reference_length(chromosome)
            fasta_length = fasta.get_reference_length(chromosome) if fasta is not None else None
            if fasta_length is not None and bam_length != fasta_length:
                raise ValueError(
                    f"reference length mismatch for {chromosome}: "
                    f"BAM={bam_length}, FASTA={fasta_length}"
                )
            reference = (np.frombuffer(
                fasta.fetch(chromosome).upper().encode("ascii"), dtype=np.uint8,
            ) if fasta is not None else None)
            component = []
            component_end = -1

            def process_component(records):
                nonlocal component_number
                if not records:
                    return
                component_number += 1
                features = [record[0] for record in records]
                profiles = {
                    record[0].index: record[1] for record in records
                    if record[1] is not None
                }
                by_index = {feature.index: feature for feature in features}
                counts["components"] += 1
                counts["features"] += len(features)
                seq_partner = {}
                seq_scores: Dict[int, SequenceScore] = {}
                seq_margins = {}
                seq_nodes = set()
                seq_kind = {}
                if use_sequence:
                    (seq_partner, seq_scores, seq_margins, _seq_edges,
                     seq_nodes, seq_kind) = _sequence_assignment(
                        features, sequence_params,
                    )

                model_result = None
                model_partner = {}
                if use_model:
                    model_result = assign_duplex_pairs(features, profiles, model, params)
                    counts["geometric_edges"] += model_result.geometric_edges
                    counts["scored_edges"] += model_result.scored_edges
                    # Sequence evidence is independent and takes precedence.
                    # Keep only complete model pairs that do not consume either
                    # member of a sequence-supported pair.
                    for index, mate in model_result.partner.items():
                        if index in seq_partner or mate in seq_partner:
                            continue
                        model_partner[index] = mate

                partner = {**seq_partner, **model_partner}
                status = {}
                for index in by_index:
                    if index in partner:
                        status[index] = STATUS_PAIRED
                    elif (index in seq_nodes or
                          (model_result is not None and
                           model_result.status.get(index) != ".")):
                        status[index] = "U"
                    else:
                        status[index] = "."
                counts["sequence_pairs"] += len(seq_partner) // 2
                counts["sequence_free_pairs"] += len(model_partner) // 2
                counts["unresolved_reads"] += sum(value == "U" for value in status.values())

                for index, value in status.items():
                    feature = by_index[index]
                    status_of[(feature.name, feature.flavor)] = value
                    if value != STATUS_PAIRED:
                        continue
                    mate = by_index[partner[index]]
                    if index in seq_partner:
                        sequence = seq_scores[index]
                        correlation = score_pair(feature, mate, sequence_params)
                        resolved[(feature.name, feature.flavor)] = {
                            "mate": mate.name,
                            "method": "S",
                            "correlation": correlation,
                            "sequence": sequence,
                            "sequence_margin": seq_margins[index],
                            "assignment": seq_kind.get(index),
                        }
                    else:
                        resolved[(feature.name, feature.flavor)] = {
                            "mate": mate.name,
                            "method": "D",
                            "score": model_result.score[index],
                            "margin": model_result.margin[index],
                        }

                for index, mate_index in partner.items():
                    if index > mate_index:
                        continue
                    a, b = by_index[index], by_index[mate_index]
                    ct, ga = (a, b) if a.flavor == FLAVOR_CT else (b, a)
                    if index in seq_partner:
                        sequence = seq_scores[index]
                        pair_rows.append({
                            "chromosome": chromosome,
                            "component": f"{chromosome}:{component_number}",
                            "ct_read": ct.name,
                            "ga_read": ga.name,
                            "method": "S",
                            "assignment": seq_kind.get(index, ""),
                            "seq_bases": sequence.bases,
                            "seq_differences": sequence.mismatches,
                            "seq_rate": f"{sequence.rate:.6f}",
                            "seq_margin": f"{seq_margins[index]:.6f}",
                        })
                        continue
                    edge = _selected_edge(model_result, ct.index, ga.index)
                    if edge is None:
                        raise RuntimeError("selected sequence-free edge lacks evidence")
                    evidence = edge.features
                    pair_rows.append({
                        "chromosome": chromosome,
                        "component": f"{chromosome}:{component_number}",
                        "ct_read": ct.name,
                        "ga_read": ga.name,
                        "method": "D",
                        "model": model.model_id,
                        "call_layer": resolved_call_layer,
                        "score": f"{edge.score:.6f}",
                        "margin": f"{model_result.margin[index]:.6f}",
                        "overlap_bp": int(evidence["overlap_bp"]),
                        "ct_dyads": int(evidence["ct_dyads"]),
                        "ga_dyads": int(evidence["ga_dyads"]),
                        "span_jaccard": f"{evidence['span_jaccard']:.6f}",
                        "raw_protection_corr": f"{evidence['raw_protection_sigma20_lag0']:.6f}",
                        "residual_protection_corr": f"{evidence['residual_all_reads_sigma20_lag0']:.6f}",
                    })

            previous_start = -1
            for read in chain((first,), iterator):
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                counts["primary_reads"] += 1
                if read.reference_start < previous_start:
                    raise ValueError("input BAM is not coordinate sorted")
                previous_start = read.reference_start
                if read.is_duplicate:
                    # 0x400 PCR copies (fiberhmm-dedup / call --dedup mark and
                    # retain them) are not pairing candidates: a copy competes
                    # with its original for the opposite-strand mate and blocks
                    # the true duplex. They pass through untagged in pass 2.
                    counts["duplicate_reads"] += 1
                    continue
                if component and read.reference_start >= component_end:
                    process_component(component)
                    component = []
                    component_end = -1
                feature = build_pattern_feature(
                    read, counts["features_considered"], params, prob_threshold,
                )
                if feature is None:
                    counts["excluded_reads"] += 1
                    continue
                previous = seen_names.get(feature.name)
                if previous is not None:
                    raise ValueError(
                        "duplex BAM requires unique primary query names; "
                        f"{feature.name!r} identifies both {previous} and "
                        f"{chromosome}:{feature.ref_start}-{feature.ref_end}"
                    )
                seen_names[feature.name] = (
                    f"{chromosome}:{feature.ref_start}-{feature.ref_end}"
                )
                counts["features_considered"] += 1
                if use_sequence:
                    if reference is not None:
                        feature.sequence_pos, feature.sequence_base = _sequence_signature(
                            read, reference,
                        )
                    else:
                        feature.sequence_pos, feature.sequence_base = \
                            _sequence_signature_from_md(read)
                profile = (build_protection_profile(
                    read, reference, params, prob_threshold, feature.flavor,
                ) if use_model else None)
                component.append((feature, profile))
                component_end = max(component_end, feature.ref_end)
                if len(component) > max_component:
                    raise RuntimeError(
                        "component safety limit reached; increase --max-component "
                        "only after auditing the input"
                    )
            process_component(component)

        input_header = pysam.AlignmentHeader.from_dict(bam.header.to_dict())
    finally:
        bam.close()
        if fasta is not None:
            fasta.close()

    counts["paired_reads"] = len(resolved)
    counts["pairs"] = len(resolved) // 2
    if pairs_tsv:
        fields = [
            "chromosome", "component", "ct_read", "ga_read", "method",
            "assignment", "model", "call_layer", "score", "margin", "overlap_bp",
            "ct_dyads", "ga_dyads", "span_jaccard",
            "raw_protection_corr", "residual_protection_corr",
            "seq_bases", "seq_differences", "seq_rate", "seq_margin",
        ]
        with open(pairs_tsv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(pair_rows)

    output_header = _header_with_program(
        input_header, model.model_id if model is not None else None, pairing_mode,
    )
    written = 0
    # Published atomically: a failed pass 2 never leaves a partial BAM at
    # out_bam.
    from fiberhmm.inference.bam_output import atomic_output
    with atomic_output(out_bam) as out_path, \
            pysam.AlignmentFile(in_bam, "rb") as bam, pysam.AlignmentFile(
        out_path, "wb", header=output_header, threads=io_threads,
    ) as out:
        for read in bam.fetch(until_eof=True):
            write_record = not paired_only
            if not (read.is_unmapped or read.is_secondary or read.is_supplementary):
                # A rerun replaces an earlier pairing decision as one coherent
                # tag set; stale sequence-route fields must not survive an
                # unresolved duplex-model decision.
                for tag in _PAIR_TAGS:
                    if read.has_tag(tag):
                        read.set_tag(tag, None)
                flavor = read_flavor(read, prob_threshold)
                if flavor is not None:
                    key = (read.query_name, flavor)
                    status = status_of.get(key)
                    if status is not None:
                        read.set_tag(_TAG_STATUS, status, value_type="A")
                        if status == STATUS_PAIRED:
                            write_record = True
                            evidence = resolved[key]
                            read.set_tag(_TAG_PARTNER, evidence["mate"], value_type="Z")
                            read.set_tag(_TAG_METHOD, evidence["method"], value_type="A")
                            if evidence["method"] == "D":
                                read.set_tag(_TAG_SCORE, int(round(1000 * evidence["score"])), value_type="i")
                                read.set_tag(_TAG_MARGIN, int(round(1000 * evidence["margin"])), value_type="i")
                                read.set_tag(_TAG_MODEL, model.model_id, value_type="Z")
                            else:
                                sequence = evidence["sequence"]
                                correlation = evidence.get("correlation")
                                if correlation is not None:
                                    read.set_tag("mc", int(round(1000 * correlation)), value_type="i")
                                read.set_tag("sb", sequence.bases, value_type="i")
                                read.set_tag("sd", sequence.mismatches, value_type="i")
                                read.set_tag("sr", int(round(1_000_000 * sequence.rate)), value_type="i")
                                read.set_tag("sg", int(round(1_000_000 * evidence["sequence_margin"])), value_type="i")
                                if evidence.get("assignment"):
                                    read.set_tag("pa", evidence["assignment"], value_type="A")
            if write_record:
                out.write(read)
                written += 1
    counts["written_reads"] = written
    if create_index:
        pysam.index(out_bam)

    receipt = {
        "status": "complete",
        "tool": "fiberhmm-pair",
        "pairing_mode": pairing_mode,
        "model_id": model.model_id if model is not None else None,
        "model_status": model.metadata.get("status") if model is not None else None,
        "input_bam": str(Path(in_bam).resolve()),
        "output_bam": str(Path(out_bam).resolve()),
        "reference_fasta": str(Path(reference_path).resolve()) if reference_path else None,
        "pairs_tsv": str(Path(pairs_tsv).resolve()) if pairs_tsv else None,
        "call_layer": resolved_call_layer,
        "detected_call_layer": detected_call_layer,
        "selection": {
            "min_margin": params.min_margin,
            "null_floor": params.null_floor,
            "min_overlap_bp": params.min_overlap_bp,
            "min_nucs": params.min_nucs,
        },
        "score_inputs": list(model.feature_names) if model is not None else [],
        "sequence_identity_used": use_sequence,
        "AT_mismatch_used": use_sequence,
        "haplotype_used": False,
        "TF_LLR_used": False,
        "counts": dict(counts),
        "seconds": time.time() - started,
    }
    if receipt_json:
        Path(receipt_json).write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def run_duplex(in_bam: str, out_bam: str, reference_path: str, **kwargs):
    """Compatibility wrapper for validation replay of the sequence-free route."""
    return run_pairing(
        in_bam, out_bam, reference_path,
        pairing_mode="sequence-free", **kwargs,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="python -m fiberhmm.cli.duplex",
        description=(
            "Sequence-identity-free CT/GA duplex pairing from FiberHMM "
            "nucleosome lattices and DddA protection profiles."
        ),
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Coordinate-sorted, indexed FiberHMM-called DddA BAM")
    parser.add_argument("-o", "--output", required=True,
                        help="Output BAM with pair status and mate tags")
    parser.add_argument("-r", "--reference", required=True,
                        help="Matching indexed reference FASTA; used only for non-CpG opportunities")
    parser.add_argument("--pairs-tsv", default=None,
                        help="Optional selected-pair evidence table")
    parser.add_argument("--receipt-json", default=None,
                        help="Optional machine-readable run receipt")
    parser.add_argument("--paired-only", action="store_true",
                        help="Write only primary source records in selected pairs")
    parser.add_argument("--model", default=None,
                        help="Override the bundled frozen duplex model JSON")
    parser.add_argument("--call-layer", choices=["auto", "input-ma", "rotational-recall"],
                        default="auto",
                        help="Nucleosome calibration; auto detects current phase-posterior headers (default auto)")
    parser.add_argument("--min-margin", type=float, default=1.0,
                        help="Two-sided model-score margin (default 1.0; externally replicated)")
    parser.add_argument("--null-floor", type=float, default=0.0,
                        help="Virtual null model score for reads with no second candidate (default 0.0)")
    parser.add_argument("--min-overlap", type=int, default=1500,
                        help="Minimum genomic overlap in bp (default 1500)")
    parser.add_argument("--min-nucs", type=int, default=4,
                        help="Minimum dyads in the overlap on each read (default 4)")
    parser.add_argument("-p", "--prob-threshold", type=int,
                        default=DEFAULT_PROB_THRESHOLD,
                        help="Minimum ML probability for MM/ML-native dU calls "
                             f"(0-255; default {DEFAULT_PROB_THRESHOLD}). R/Y- and "
                             "MD-encoded input is binary and ignores it.")
    parser.add_argument("--max-component", type=int, default=10000,
                        help="Safety ceiling for a complete overlap component (default 10000)")
    parser.add_argument("--io-threads", type=int, default=4,
                        help="htslib output compression threads (default 4)")
    parser.add_argument("--no-index", action="store_true",
                        help="Do not create an index for the output BAM")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        parser.error(f"input not found: {args.input}")
    if not os.path.isfile(args.reference):
        parser.error(f"reference not found: {args.reference}")
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        parser.error("input and output paths must differ")
    params = DuplexParams(
        min_margin=args.min_margin,
        null_floor=args.null_floor,
        min_overlap_bp=args.min_overlap,
        min_nucs=args.min_nucs,
    )
    try:
        receipt = run_duplex(
            args.input, args.output, args.reference,
            params=params, model_path=args.model,
            prob_threshold=args.prob_threshold,
            pairs_tsv=args.pairs_tsv, receipt_json=args.receipt_json,
            paired_only=args.paired_only, io_threads=args.io_threads,
            max_component=args.max_component, call_layer=args.call_layer,
            create_index=not args.no_index,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"fiberhmm-duplex: error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    counts = receipt["counts"]
    print(
        f"fiberhmm-duplex: {counts.get('features', 0):,} feature reads, "
        f"{counts.get('scored_edges', 0):,} scored edges -> "
        f"{counts.get('pairs', 0):,} pairs in {receipt['seconds']:.1f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
