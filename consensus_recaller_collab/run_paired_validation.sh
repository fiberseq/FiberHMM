#!/usr/bin/env bash
set -euo pipefail

# Reproduce the focal paired-consensus validation panel. All scientific
# artifacts remain under the Dropbox-backed project tree; no report or BAM is
# written to the system temporary directory.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${1:-"${ROOT}/consensus_visualization_outputs/paired_v7"}
REPORTS="${OUT}/reports"
PACBIO=/mnt/g/v3seg_mp
DDDB=/mnt/g/Dropbox/Fiber-NET-seq/Drosophila_phase2/Datasets/DAF-seq/spacetime_updated/fp_update
NANOPORE=/mnt/g/Dropbox/Fiber-NET-seq/Drosophila_phase2/Datasets/Fiber-seq/2-point_timecourse/bam

cd "${ROOT}"
mkdir -p \
  "${REPORTS}" \
  "${OUT}/homie" "${OUT}/nhomie" "${OUT}/sf1" "${OUT}/sf2" \
  "${OUT}/ind_dddb" "${OUT}/ind_nanopore" \
  "${OUT}/uba1_ddda" "${OUT}/napa_ddda"

pacbio_recall() {
  local locus=$1
  local region=$2
  python -m consensus_recaller_collab.revised_prototype \
    -i "${PACBIO}/2-4hr_11.bam" \
    -i "${PACBIO}/2-4hr_4.bam" \
    -i "${PACBIO}/2-4hr_6.bam" \
    -i "${PACBIO}/2-4hr_7.bam" \
    -i "${PACBIO}/2-4hr_9.bam" \
    --preset hia5-pacbio \
    --region "${region}" \
    --min-support 50 \
    --min-config-support 20 \
    --max-proposals 0 \
    --proposal-tsv "${REPORTS}/${locus}.paired-v7.tsv" \
    -o "${REPORTS}/${locus}.paired-v7.json"
}

pacbio_recall homie chr2R:9988750-9989118
pacbio_recall nhomie chr2R:9972790-9973386
pacbio_recall sf1 chr3R:6853644-6855684
pacbio_recall sf2 chr3R:6869630-6871751

python -m consensus_recaller_collab.revised_prototype \
  -i "${DDDB}/1.5-2_recalled.bam" \
  -i "${DDDB}/2-2.5_recalled.bam" \
  -i "${DDDB}/2.5-3_recalled.bam" \
  -i "${DDDB}/3-3.5_recalled.bam" \
  -i "${DDDB}/3.5-4_recalled.bam" \
  -i "${DDDB}/yw_2-4_recalled.bam" \
  --preset dddb \
  --region chr3L:15039880-15040260 \
  --site 15039948-15040022 \
  --site 15040154-15040235 \
  --forced-sites-only \
  --min-support 50 \
  --strand-min-source-support 50 \
  --strand-control-shifts=-200,-100,100,200 \
  --skip-composite-deconvolution \
  --max-proposals 0 \
  --proposal-tsv "${REPORTS}/ind-dddb.paired-v7.tsv" \
  -o "${REPORTS}/ind-dddb.paired-v7.json"

python -m consensus_recaller_collab.revised_prototype \
  -i "${NANOPORE}/siGAF_1.5-3hr.fiberhmm.thr248.bam" \
  -i "${NANOPORE}/siGAF_3-4.5hr.fiberhmm.thr248.bam" \
  --preset hia5-nanopore \
  --prob-threshold 248 \
  --region chr3L:15042180-15042440 \
  --site 15042227-15042243 \
  --site 15042377-15042396 \
  --forced-sites-only \
  --min-support 1 \
  --strand-min-source-support 1 \
  --strand-control-shifts=-200,-100,100,200 \
  --max-proposals 0 \
  --proposal-tsv "${REPORTS}/ind-nanopore.paired-v7.tsv" \
  -o "${REPORTS}/ind-nanopore.paired-v7.json"

python -m consensus_recaller_collab.revised_prototype \
  -i "${ROOT}/ddda_profile/uba1_ddda_recall.bam" \
  --preset ddda \
  --region chrX:47194800-47195020 \
  --site 47194899-47194929 \
  --forced-sites-only \
  --min-support 100 \
  --strand-min-source-support 100 \
  --strand-control-shifts=-1550,-1275,-600,-325,-225 \
  --skip-composite-deconvolution \
  --max-proposals 0 \
  --proposal-tsv "${REPORTS}/uba1-ddda.paired-v7.tsv" \
  -o "${REPORTS}/uba1-ddda.paired-v7.json"

python -m consensus_recaller_collab.revised_prototype \
  -i "${ROOT}/ddda_nuc_output/napa_recaller_TF.bam" \
  --preset ddda \
  --region chr19:47515400-47515620 \
  --site 47515501-47515514 \
  --forced-sites-only \
  --min-support 100 \
  --strand-min-source-support 100 \
  --strand-control-shifts=-500,-350,-200,-100,1850 \
  --skip-composite-deconvolution \
  --max-proposals 0 \
  --proposal-tsv "${REPORTS}/napa-ddda.paired-v7.tsv" \
  -o "${REPORTS}/napa-ddda.paired-v7.json"

for locus in homie nhomie sf1 sf2; do
  python -m consensus_recaller_collab.annotate \
    --report "${REPORTS}/${locus}.paired-v7.json" \
    --paired \
    --output-dir "${OUT}/${locus}"
done

python -m consensus_recaller_collab.annotate \
  --report "${REPORTS}/ind-dddb.paired-v7.json" \
  --paired --output-dir "${OUT}/ind_dddb"
python -m consensus_recaller_collab.annotate \
  --report "${REPORTS}/ind-nanopore.paired-v7.json" \
  --paired --output-dir "${OUT}/ind_nanopore"
python -m consensus_recaller_collab.annotate \
  --report "${REPORTS}/uba1-ddda.paired-v7.json" \
  --paired --output-dir "${OUT}/uba1_ddda"
python -m consensus_recaller_collab.annotate \
  --report "${REPORTS}/napa-ddda.paired-v7.json" \
  --paired --output-dir "${OUT}/napa_ddda"

merge_pacbio() {
  local locus=$1
  samtools merge -f -o "${OUT}/${locus}/${locus}.pooled.paired-v7.bam" \
    "${OUT}/${locus}/2-4hr_11.consensus-overlay.bam" \
    "${OUT}/${locus}/2-4hr_4.consensus-overlay.bam" \
    "${OUT}/${locus}/2-4hr_6.consensus-overlay.bam" \
    "${OUT}/${locus}/2-4hr_7.consensus-overlay.bam" \
    "${OUT}/${locus}/2-4hr_9.consensus-overlay.bam"
  samtools index "${OUT}/${locus}/${locus}.pooled.paired-v7.bam"
  python -m consensus_recaller_collab.audit_paired_bam \
    -i "${OUT}/${locus}/${locus}.pooled.paired-v7.bam" \
    -o "${OUT}/${locus}/${locus}.pooled.paired-v7.audit.json"
}

for locus in homie nhomie sf1 sf2; do
  merge_pacbio "${locus}"
done

samtools merge -f -o "${OUT}/ind_dddb/ind-dddb.pooled.paired-v7.bam" \
  "${OUT}/ind_dddb/1.5-2_recalled.consensus-overlay.bam" \
  "${OUT}/ind_dddb/2-2.5_recalled.consensus-overlay.bam" \
  "${OUT}/ind_dddb/2.5-3_recalled.consensus-overlay.bam" \
  "${OUT}/ind_dddb/3-3.5_recalled.consensus-overlay.bam" \
  "${OUT}/ind_dddb/3.5-4_recalled.consensus-overlay.bam" \
  "${OUT}/ind_dddb/yw_2-4_recalled.consensus-overlay.bam"
samtools index "${OUT}/ind_dddb/ind-dddb.pooled.paired-v7.bam"
python -m consensus_recaller_collab.audit_paired_bam \
  -i "${OUT}/ind_dddb/ind-dddb.pooled.paired-v7.bam" \
  -o "${OUT}/ind_dddb/ind-dddb.pooled.paired-v7.audit.json"

samtools merge -f -o "${OUT}/ind_nanopore/ind-nanopore.pooled.paired-v7.bam" \
  "${OUT}/ind_nanopore/siGAF_1.5-3hr.fiberhmm.thr248.consensus-overlay.bam" \
  "${OUT}/ind_nanopore/siGAF_3-4.5hr.fiberhmm.thr248.consensus-overlay.bam"
samtools index "${OUT}/ind_nanopore/ind-nanopore.pooled.paired-v7.bam"
python -m consensus_recaller_collab.audit_paired_bam \
  -i "${OUT}/ind_nanopore/ind-nanopore.pooled.paired-v7.bam" \
  -o "${OUT}/ind_nanopore/ind-nanopore.pooled.paired-v7.audit.json"

python -m consensus_recaller_collab.audit_paired_bam \
  -i "${OUT}/uba1_ddda/uba1_ddda_recall.consensus-overlay.bam" \
  -o "${OUT}/uba1_ddda/uba1-ddda.paired-v7.audit.json"
python -m consensus_recaller_collab.audit_paired_bam \
  -i "${OUT}/napa_ddda/napa_recaller_TF.consensus-overlay.bam" \
  -o "${OUT}/napa_ddda/napa-ddda.paired-v7.audit.json"

python -m consensus_recaller_collab.audit_paired_bam \
  -i "${OUT}/homie/homie.pooled.paired-v7.bam" \
  -i "${OUT}/nhomie/nhomie.pooled.paired-v7.bam" \
  -i "${OUT}/sf1/sf1.pooled.paired-v7.bam" \
  -i "${OUT}/sf2/sf2.pooled.paired-v7.bam" \
  -i "${OUT}/ind_dddb/ind-dddb.pooled.paired-v7.bam" \
  -i "${OUT}/ind_nanopore/ind-nanopore.pooled.paired-v7.bam" \
  -i "${OUT}/uba1_ddda/uba1_ddda_recall.consensus-overlay.bam" \
  -i "${OUT}/napa_ddda/napa_recaller_TF.consensus-overlay.bam" \
  -o "${OUT}/paired-v7.all-panels.audit.json"
