#!/usr/bin/env python3
"""Sweep FN and breathing for DddA FiberHMM model.

Metrics reported per combination:
  - nucs / read
  - mean nuc size
  - % of nucs in mono-nuc range (90-180 bp)
  - % tri+ (likely overmerge)
  - MSPs / read
  - % MSPs ≥300 bp (likely NFRs / real open regions)
"""
import subprocess, json, os, shutil, sys
import numpy as np
import pysam

BAM_IN = '/tmp/ps01499_small.bam'
# Expected for PS01499 amplicon (~2.9 kb):
#   ~15-16 nucs/read at 180 bp NRL
#   median nuc 147-170 bp
EXPECTED_NUCS_PER_READ = 16

def run_combo(fn, breathing, fp_scale=1.0, sample_reads=2000):
    tag = f'fn{int(fn*100):02d}_br{int(breathing*100):02d}_fpx{int(fp_scale*100):03d}'
    mdl = f'models/legacy/ddda_v3/ddda_{tag}.json'
    out_dir = f'/tmp/ddda_sweep/{tag}/'
    os.makedirs(out_dir, exist_ok=True)

    # Build model
    subprocess.run(['python', 'build_ddda_model.py',
                     '--source-model', 'models/dddb_nanopore.json',
                     '--fp-model', 'phase0/data/fp_models/ct_nanopore_fp_3mer.json',
                     '--fn', str(fn),
                     '--breathing', str(breathing),
                     '--fp-scale', str(fp_scale),
                     '--out-model', mdl], check=True,
                    capture_output=True)

    # Apply
    result = subprocess.run(['python', 'apply_model.py',
                              '-i', BAM_IN, '-m', mdl, '-o', out_dir,
                              '--mode', 'daf', '-k', '3',
                              '--cores', '4'],
                             capture_output=True, text=True)
    # Find output BAM
    out_bam = None
    for f in os.listdir(out_dir):
        if f.endswith('_footprints.bam'):
            out_bam = os.path.join(out_dir, f); break
    if out_bam is None:
        return None

    # Read stats — retry a few times, with time.sleep, since apply_model
    # sometimes returns before BAM is fully flushed
    import time
    for attempt in range(5):
        try:
            time.sleep(0.5)
            b = pysam.AlignmentFile(out_bam, 'rb', check_sq=False,
                                      ignore_truncation=True)
            break
        except Exception as e:
            if attempt == 4:
                print(f'  failed to open {out_bam}: {e}')
                return None
    all_nl = []; all_al = []; n_reads = 0
    for r in b:
        if r.is_unmapped or r.is_secondary or r.is_supplementary: continue
        n_reads += 1
        if r.has_tag('nl'): all_nl.extend(list(r.get_tag('nl')))
        if r.has_tag('al'): all_al.extend(list(r.get_tag('al')))
        if n_reads >= sample_reads: break
    b.close()

    nl = np.array(all_nl) if all_nl else np.array([1])
    al = np.array(all_al) if all_al else np.array([1])

    stats = {
        'fn': fn, 'breathing': breathing, 'fp_scale': fp_scale,
        'n_reads': n_reads,
        'nucs_per_read': len(nl) / max(1, n_reads),
        'msps_per_read': len(al) / max(1, n_reads),
        'nuc_mean': float(nl.mean()),
        'nuc_median': int(np.median(nl)),
        'pct_mono': float(100 * ((nl >= 90) & (nl <= 180)).mean()),
        'pct_di':   float(100 * ((nl > 180) & (nl <= 300)).mean()),
        'pct_tri_plus': float(100 * (nl > 300).mean()),
        'msp_mean': float(al.mean()),
        'msp_median': int(np.median(al)),
        'pct_msp_nfr_sized': float(100 * (al >= 300).mean()),
    }
    # cleanup BAM to save disk
    shutil.rmtree(out_dir, ignore_errors=True)
    return stats


def main():
    fns = [0.2, 0.3, 0.4, 0.5, 0.6]
    breathings = [0.05, 0.10, 0.15, 0.20]
    results = []
    print(f'{"fn":>5s} {"breathing":>10s} {"nucs/rd":>8s} {"mean_nl":>8s} '
          f'{"med_nl":>7s} {"%mono":>6s} {"%di":>6s} {"%tri+":>6s} '
          f'{"msp/rd":>7s} {"%msp≥300":>9s}')
    for fn in fns:
        for br in breathings:
            try:
                s = run_combo(fn, br)
            except subprocess.CalledProcessError as e:
                print(f'ERR fn={fn} br={br}: {e}')
                continue
            if s is None:
                print(f'NO OUTPUT for fn={fn} br={br}'); continue
            print(f'{s["fn"]:>5.2f} {s["breathing"]:>10.2f} '
                  f'{s["nucs_per_read"]:>8.1f} {s["nuc_mean"]:>8.0f} '
                  f'{s["nuc_median"]:>7d} {s["pct_mono"]:>5.1f}% '
                  f'{s["pct_di"]:>5.1f}% {s["pct_tri_plus"]:>5.1f}% '
                  f'{s["msps_per_read"]:>7.1f} {s["pct_msp_nfr_sized"]:>8.1f}%',
                  flush=True)
            results.append(s)

    with open('models/legacy/ddda_v3/sweep_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nWrote models/legacy/ddda_v3/sweep_results.json')


if __name__ == '__main__':
    main()
