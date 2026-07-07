#!/usr/bin/env bash
# samosa_ccs_to_modbam.sh — SAMOSA / any PacBio-EcoGII-m6A raw -> standard-tag modBAM.
#
# SAMOSA (Ramani) deposits EcoGII m6A read out by PacBio kinetics — the SAME mark and
# platform as Fiber-seq on PacBio. So the standard-tag modBAM path reuses the fibertools
# m6A CNN (`ft predict-m6a`), which writes spec MM/ML (A+a). Output feeds FiberHMM's
# existing `pacbio-fiber` mode directly.
#
# Pipeline:
#   subreads.bam --ccs(--hifi-kinetics)--> CCS.bam --ft predict-m6a--> m6A modBAM
#                --pbmm2--> aligned sorted modBAM
#
# DEPS (install before running):
#   ccs (pbccs)   : conda install -c bioconda pbccs      # NOT yet installed
#   ft            : /home/tommytullius/yes/bin/ft        # present
#   pbmm2         : present
# If the SRA object is already CCS-with-kinetics, skip the ccs step (set -C).
#
# SAMOSA raw: BioProject PRJNA681807 / SRP295327 (K562 + in-vitro; 22 PacBio runs).
# Pick a treated (chromatin) run + a naked-DNA control run — sample identity is in the
# SRA/GEO SRX titles (GSE162410), not in the ENA library_name field.
#
# Usage:
#   samosa_ccs_to_modbam.sh -i subreads.bam -o out.aligned.bam -r ref.fa [-C]   # -C: input already CCS
set -euo pipefail
FT=${FT:-/home/tommytullius/yes/bin/ft}
CCS_DONE=0; IN=""; OUT=""; REF=""
while getopts "i:o:r:C" f; do case $f in i) IN=$OPTARG;; o) OUT=$OPTARG;; r) REF=$OPTARG;; C) CCS_DONE=1;; esac; done
[ -z "$IN" ] || [ -z "$OUT" ] || [ -z "$REF" ] && { echo "usage: -i in.bam -o out.bam -r ref.fa [-C]"; exit 2; }
TMP=$(dirname "$OUT"); base=$(basename "$OUT" .bam)

if [ "$CCS_DONE" -eq 0 ]; then
  command -v ccs >/dev/null || { echo "ERROR: ccs (pbccs) not installed — conda install -c bioconda pbccs"; exit 3; }
  echo "[1/3] ccs --hifi-kinetics"; ccs --hifi-kinetics -j 16 "$IN" "$TMP/$base.ccs.bam"
  CCS="$TMP/$base.ccs.bam"
else CCS="$IN"; fi

echo "[2/3] ft predict-m6a (writes standard MM/ML A+a)"
"$FT" predict-m6a "$CCS" "$TMP/$base.m6a.bam"

echo "[3/3] pbmm2 align + sort"
pbmm2 align "$REF" "$TMP/$base.m6a.bam" "$OUT" --sort --preset CCS
samtools index -@4 "$OUT"
echo "[done] $OUT"
