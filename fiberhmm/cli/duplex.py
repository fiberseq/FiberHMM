#!/usr/bin/env python3
"""fiberhmm-duplex -- sequence-identity-free scDAF duplex pairing.

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
from fiberhmm.crossstrand.pairing import read_flavor
from fiberhmm.io.bam_header import append_pg_record

_TAG_PARTNER = "mp"
_TAG_STATUS = "mt"
_TAG_METHOD = "pm"
_TAG_SCORE = "dm"
_TAG_MARGIN = "mg"
_TAG_MODEL = "mv"
_PAIR_TAGS = ("mp", "mc", "mg", "mt", "pm", "sb", "sd", "sr", "sg", "pa", "dm", "mv")


def _header_with_program(header, model_id: str):
    return append_pg_record(header, {
        "PN": "fiberhmm-duplex",
        "VN": __version__,
        "CL": "sequence-identity-free CT/GA duplex pairing",
        "DS": f"model={model_id}; A/T identity, haplotype and TF LLR unused",
    })


def _selected_edge(result, left: int, right: int):
    return result.evidence.get((left, right)) or result.evidence.get((right, left))


def run_duplex(in_bam: str, out_bam: str, reference_path: str,
               params: Optional[DuplexParams] = None,
               model: Optional[DuplexModel] = None,
               model_path: Optional[str] = None,
               prob_threshold: int = 0,
               pairs_tsv: Optional[str] = None,
               receipt_json: Optional[str] = None,
               paired_only: bool = False,
               io_threads: int = 4,
               max_component: int = 10000,
               call_layer: str = "auto",
               create_index: bool = True):
    """Run the two-pass pairing workflow and return its machine-readable receipt."""
    started = time.time()
    params = params or DuplexParams()
    with pysam.AlignmentFile(in_bam, "rb") as source:
        detected_call_layer = infer_call_layer(source.header)
    resolved_call_layer = detected_call_layer if call_layer == "auto" else call_layer
    model = model or load_duplex_model(model_path, resolved_call_layer)
    resolved: Dict[Tuple[str, int], Tuple[str, float, float]] = {}
    status_of: Dict[Tuple[str, int], str] = {}
    seen_names: Dict[str, str] = {}
    counts = Counter()
    pair_rows: List[dict] = []

    with pysam.AlignmentFile(in_bam, "rb") as bam, pysam.FastaFile(reference_path) as fasta:
        if bam.header.get("HD", {}).get("SO") != "coordinate":
            raise ValueError("fiberhmm-duplex requires a coordinate-sorted BAM")
        if not bam.has_index():
            raise ValueError("fiberhmm-duplex requires an indexed input BAM")
        fasta_names = set(fasta.references)
        component_number = 0

        for chromosome in bam.references:
            iterator = bam.fetch(chromosome)
            first = next(iterator, None)
            if first is None:
                continue
            if chromosome not in fasta_names:
                raise ValueError(f"reference FASTA is missing BAM contig {chromosome!r}")
            bam_length = bam.get_reference_length(chromosome)
            fasta_length = fasta.get_reference_length(chromosome)
            if bam_length != fasta_length:
                raise ValueError(
                    f"reference length mismatch for {chromosome}: "
                    f"BAM={bam_length}, FASTA={fasta_length}"
                )
            reference = np.frombuffer(
                fasta.fetch(chromosome).upper().encode("ascii"), dtype=np.uint8,
            )
            component = []
            component_end = -1

            def process_component(records):
                nonlocal component_number
                if not records:
                    return
                component_number += 1
                features = [record[0] for record in records]
                profiles = {record[0].index: record[1] for record in records}
                result = assign_duplex_pairs(features, profiles, model, params)
                by_index = {feature.index: feature for feature in features}
                counts["components"] += 1
                counts["features"] += len(features)
                counts["geometric_edges"] += result.geometric_edges
                counts["scored_edges"] += result.scored_edges
                counts["unresolved_reads"] += sum(
                    status == "U" for status in result.status.values()
                )
                for index, status in result.status.items():
                    feature = by_index[index]
                    status_of[(feature.name, feature.flavor)] = status
                    if status != STATUS_PAIRED:
                        continue
                    mate = by_index[result.partner[index]]
                    resolved[(feature.name, feature.flavor)] = (
                        mate.name, result.score[index], result.margin[index],
                    )
                for index, mate_index in result.partner.items():
                    if index > mate_index:
                        continue
                    a, b = by_index[index], by_index[mate_index]
                    ct, ga = (a, b) if a.flavor == FLAVOR_CT else (b, a)
                    edge = _selected_edge(result, ct.index, ga.index)
                    if edge is None:
                        raise RuntimeError("selected duplex edge lacks evidence")
                    feature = edge.features
                    pair_rows.append({
                        "chromosome": chromosome,
                        "component": f"{chromosome}:{component_number}",
                        "ct_read": ct.name,
                        "ga_read": ga.name,
                        "method": "D",
                        "model": model.model_id,
                        "call_layer": resolved_call_layer,
                        "score": f"{edge.score:.6f}",
                        "margin": f"{result.margin[index]:.6f}",
                        "overlap_bp": int(feature["overlap_bp"]),
                        "ct_dyads": int(feature["ct_dyads"]),
                        "ga_dyads": int(feature["ga_dyads"]),
                        "span_jaccard": f"{feature['span_jaccard']:.6f}",
                        "raw_protection_corr": f"{feature['raw_protection_sigma20_lag0']:.6f}",
                        "residual_protection_corr": f"{feature['residual_all_reads_sigma20_lag0']:.6f}",
                    })

            previous_start = -1
            for read in chain((first,), iterator):
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                counts["primary_reads"] += 1
                if read.reference_start < previous_start:
                    raise ValueError("input BAM is not coordinate sorted")
                previous_start = read.reference_start
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
                profile = build_protection_profile(
                    read, reference, params, prob_threshold, feature.flavor,
                )
                component.append((feature, profile))
                component_end = max(component_end, feature.ref_end)
                if len(component) > max_component:
                    raise RuntimeError(
                        "component safety limit reached; increase --max-component "
                        "only after auditing the input"
                    )
            process_component(component)

        input_header = pysam.AlignmentHeader.from_dict(bam.header.to_dict())

    counts["paired_reads"] = len(resolved)
    counts["pairs"] = len(resolved) // 2
    if pairs_tsv:
        fields = [
            "chromosome", "component", "ct_read", "ga_read", "method",
            "model", "call_layer", "score", "margin", "overlap_bp",
            "ct_dyads", "ga_dyads", "span_jaccard",
            "raw_protection_corr", "residual_protection_corr",
        ]
        with open(pairs_tsv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(pair_rows)

    output_header = _header_with_program(input_header, model.model_id)
    written = 0
    with pysam.AlignmentFile(in_bam, "rb") as bam, pysam.AlignmentFile(
        out_bam, "wb", header=output_header, threads=io_threads,
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
                            mate, score, margin = resolved[key]
                            read.set_tag(_TAG_PARTNER, mate, value_type="Z")
                            read.set_tag(_TAG_METHOD, "D", value_type="A")
                            read.set_tag(_TAG_SCORE, int(round(1000 * score)), value_type="i")
                            read.set_tag(_TAG_MARGIN, int(round(1000 * margin)), value_type="i")
                            read.set_tag(_TAG_MODEL, model.model_id, value_type="Z")
            if write_record:
                out.write(read)
                written += 1
    counts["written_reads"] = written
    if create_index:
        pysam.index(out_bam)

    receipt = {
        "status": "complete",
        "tool": "fiberhmm-duplex",
        "model_id": model.model_id,
        "model_status": model.metadata.get("status"),
        "input_bam": str(Path(in_bam).resolve()),
        "output_bam": str(Path(out_bam).resolve()),
        "reference_fasta": str(Path(reference_path).resolve()),
        "pairs_tsv": str(Path(pairs_tsv).resolve()) if pairs_tsv else None,
        "call_layer": resolved_call_layer,
        "detected_call_layer": detected_call_layer,
        "selection": {
            "min_margin": params.min_margin,
            "null_floor": params.null_floor,
            "min_overlap_bp": params.min_overlap_bp,
            "min_nucs": params.min_nucs,
        },
        "score_inputs": list(model.feature_names),
        "sequence_identity_used": False,
        "AT_mismatch_used": False,
        "haplotype_used": False,
        "TF_LLR_used": False,
        "counts": dict(counts),
        "seconds": time.time() - started,
    }
    if receipt_json:
        Path(receipt_json).write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(
        prog="fiberhmm-duplex",
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
    parser.add_argument("-p", "--prob-threshold", type=int, default=0,
                        help="Minimum ML probability for MM/ML dU calls (default 0)")
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
