#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 INPUT.bam OUTPUT.bam" >&2
    exit 2
fi

input_bam=$1
output_bam=$2

if [[ "$input_bam" == "$output_bam" ]]; then
    echo "input and output BAM paths must differ" >&2
    exit 2
fi

fiberhmm-call \
    --input "$input_bam" \
    --output "$output_bam" \
    --enzyme hia5 \
    --seq nanopore \
    --prob-threshold 248 \
    --recall-nucs \
    --nuc-recall-policy topology \
    --phase-nrl off \
    --msp-min-size 60 \
    --region-parallel \
    --skip-scaffolds \
    --cores 12 \
    --io-threads 8
