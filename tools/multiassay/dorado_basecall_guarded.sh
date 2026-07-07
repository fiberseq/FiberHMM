#!/usr/bin/env bash
# dorado_basecall_guarded.sh — VRAM-safe dorado modification basecalling.
#
# Produces an (optionally aligned) modBAM with STANDARD MM/ML tags (dorado emits
# ChEBI-standard codes: 6mA=a, 5mC=m), so output feeds directly into FiberHMM.
#
# SAFETY (never crash a shared GPU):
#   * Refuses to start if GPU memory USED >= MAX_USED_GB (default 30) — i.e. another
#     job (e.g. a model training) already owns the card.
#   * Refuses to start if FREE VRAM < MIN_FREE_GB (default 14).
#   * Caps dorado's own footprint with --batchsize (default 96).
#   * Runs a watchdog: if used climbs >= MAX_USED_GB or free drops < YIELD_FREE_GB
#     while we run, it KILLS dorado so the other job keeps the memory. A yielded run
#     leaves a .INCOMPLETE marker and can be re-run later (idempotent-ish).
#
# INPUT: a directory of pod5 (preferred) or fast5. fast5 must first be converted:
#     pod5 convert fast5 raw/*.fast5 --output pod5/   (pod5 pkg; dorado>=0.5 needs pod5)
# NOTE: dorado 0.8 targets R10.4.1. R9.4.1 modification models may be unavailable;
#     if so, basecall R9 with an older dorado (<=0.5) or remora. Check per dataset.
#
# Usage:
#   dorado_basecall_guarded.sh -i POD5_DIR -o OUT.bam [-m "sup,6mA,5mCG_5hmCG"] [-r REF.fa]
set -uo pipefail

DORADO=${DORADO:-/home/tommytullius/tools/dorado-0.8.0-linux-x64/bin/dorado}
MODEL="sup,6mA,5mCG_5hmCG"     # accuracy tier + modification models
REF=""; BATCH=96
MAX_USED_GB=${MAX_USED_GB:-30}     # do not start if >= this many GB already used
MIN_FREE_GB=${MIN_FREE_GB:-14}     # require at least this much free to start
YIELD_FREE_GB=${YIELD_FREE_GB:-4}  # kill our job if free drops below this
IN=""; OUT=""

while getopts "i:o:m:r:b:" f; do case $f in
  i) IN=$OPTARG;; o) OUT=$OPTARG;; m) MODEL=$OPTARG;; r) REF=$OPTARG;; b) BATCH=$OPTARG;; esac; done
[ -z "$IN" ] || [ -z "$OUT" ] && { echo "usage: -i POD5_DIR -o OUT.bam [-m model] [-r ref.fa]"; exit 2; }

used_gb(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}'; }
free_gb(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}'; }

U=$(used_gb); F=$(free_gb)
echo "[guard] GPU used=${U}GB free=${F}GB  (limits: start<${MAX_USED_GB} used, need>=${MIN_FREE_GB} free)"
if [ "$U" -ge "$MAX_USED_GB" ]; then echo "[guard] GPU busy (used ${U}GB >= ${MAX_USED_GB}GB) — DEFERRING, not starting."; exit 3; fi
if [ "$F" -lt "$MIN_FREE_GB" ]; then echo "[guard] only ${F}GB free (< ${MIN_FREE_GB}) — DEFERRING."; exit 3; fi

echo "[run] dorado basecaller $MODEL  batchsize=$BATCH  -> $OUT"
"$DORADO" basecaller "$MODEL" "$IN" --device cuda:0 --batchsize "$BATCH" \
    ${REF:+--reference "$REF"} > "$OUT.tmp" 2> "$OUT.dorado.log" &
DPID=$!

# watchdog: yield the GPU if contention appears
( while kill -0 "$DPID" 2>/dev/null; do
    u=$(used_gb); f=$(free_gb)
    if [ "$u" -ge "$MAX_USED_GB" ] || [ "$f" -lt "$YIELD_FREE_GB" ]; then
      echo "[guard] contention (used=${u}GB free=${f}GB) — KILLING dorado to yield GPU." >> "$OUT.dorado.log"
      kill "$DPID" 2>/dev/null; touch "$OUT.INCOMPLETE"; break
    fi
    sleep 15
  done ) &
WPID=$!

wait "$DPID"; RC=$?
kill "$WPID" 2>/dev/null
if [ -f "$OUT.INCOMPLETE" ]; then echo "[guard] YIELDED before completion — rerun later."; exit 4; fi
[ "$RC" -ne 0 ] && { echo "[run] dorado exited rc=$RC (see $OUT.dorado.log)"; exit "$RC"; }

# sort + index if aligned (reference given), else keep unaligned modBAM
if [ -n "$REF" ]; then
  samtools sort -@4 -o "$OUT" "$OUT.tmp" && rm -f "$OUT.tmp" && samtools index -@4 "$OUT"
else
  mv "$OUT.tmp" "$OUT"
fi
echo "[done] $OUT"
