#!/usr/bin/env python3
"""Build a DddA-specific FiberHMM model by combining:
  - Transitions from an existing well-trained model (DddB Nanopore default)
  - Nuc-state emissions: per-context FP rate (from DddB control) × FP multiplier,
    plus a fixed breathing rate. Represents "how often a hit appears
    inside a nuc body due to FP + enzyme breathing."
  - Linker-state emissions: (1 - FN) = per-position probability of
    observing a hit in accessible DNA. Fixed across contexts.

No training — just biologically-motivated emission choices that
can be swept until the output looks right on known amplicons
(NAPA NFR, fly nucleosome arrays, etc.).
"""
import argparse, json, copy
import numpy as np


def build_model(source_model_path, fp_rates, fn=0.5, fp_scale=1.0,
                 breathing=0.0):
    """fp_rates: dict context_index → fp rate (1 per 6-mer context, size 4096)
       OR a flat float (same for all contexts)
       fn: false-negative rate in linker state → P(meth|linker) = 1 - fn
       breathing: added to nuc emission — enzyme breathing rate
    """
    m = json.load(open(source_model_path))
    if m['n_states'] != 2:
        raise ValueError('Only 2-state models supported')

    n_contexts = 4096  # 6-mer context
    non_target = n_contexts  # code 4096
    unmeth_offset = n_contexts + 1  # codes 4097..8192 are "unmethylated"
    n_symbols = 2 * (n_contexts + 1)  # 8194

    # Resolve fp_rates
    if isinstance(fp_rates, (int, float)):
        fp_vec = np.full(n_contexts, float(fp_rates))
    else:
        fp_vec = np.array([fp_rates[i] for i in range(n_contexts)])
    nuc_meth_rate = np.clip(fp_vec * fp_scale + breathing, 0.0, 1.0)

    # Linker: fixed (1 - FN)
    linker_meth_rate = float(np.clip(1.0 - fn, 0.0, 1.0))

    emit = np.zeros((2, n_symbols))
    # Convention: state 0 = linker, state 1 = nuc (matches DddB Nanopore model)
    for ctx in range(n_contexts):
        # Linker (state 0)
        emit[0, ctx] = linker_meth_rate              # methylated
        emit[0, ctx + unmeth_offset] = 1 - linker_meth_rate  # unmethylated
        # Nuc (state 1)
        emit[1, ctx] = nuc_meth_rate[ctx]
        emit[1, ctx + unmeth_offset] = 1 - nuc_meth_rate[ctx]
    # Non-target positions: uninformative (same in both states)
    emit[0, non_target] = 0.5
    emit[0, non_target + unmeth_offset] = 0.5
    emit[1, non_target] = 0.5
    emit[1, non_target + unmeth_offset] = 0.5

    m_out = copy.deepcopy(m)
    m_out['emissionprob'] = emit.tolist()
    m_out['mode'] = 'daf'
    m_out['_source_model'] = source_model_path
    m_out['_ddda_params'] = {'fn': fn, 'fp_scale': fp_scale,
                              'breathing': breathing}
    return m_out


def load_fp_model_to_6mer(fp_json_path):
    """Our FP models are 3-mer. We expand to 6-mer (13-mer flank pairs?
    Actually the fiberhmm 6-mer is a specific encoding). For a minimal
    start, just return the global FP rate as a scalar — we can upgrade
    to 6-mer tables later."""
    data = json.load(open(fp_json_path))
    return float(data['global_rate'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-model', default='models/dddb_nanopore.json',
                    help='Transitions sourced from this existing model')
    ap.add_argument('--fp-model', default=None,
                    help='Path to our ct_*_fp_3mer.json. If omitted, '
                         'uses --fp-flat value uniformly')
    ap.add_argument('--fp-flat', type=float, default=0.01,
                    help='Uniform FP rate when no fp-model provided')
    ap.add_argument('--fn', type=float, default=0.5,
                    help='False-negative rate (1 - linker hit prob)')
    ap.add_argument('--fp-scale', type=float, default=1.0,
                    help='Multiplier on FP rate for nuc emission')
    ap.add_argument('--breathing', type=float, default=0.0,
                    help='Additional per-opp breathing rate in nuc state')
    ap.add_argument('--out-model', required=True)
    args = ap.parse_args()

    if args.fp_model:
        fp_rate = load_fp_model_to_6mer(args.fp_model)
    else:
        fp_rate = args.fp_flat

    m = build_model(args.source_model, fp_rate, fn=args.fn,
                    fp_scale=args.fp_scale, breathing=args.breathing)

    with open(args.out_model, 'w') as f:
        json.dump(m, f, indent=2)
    print(f'Wrote {args.out_model}')
    print(f'  transitions from: {args.source_model}')
    print(f'  FN (linker unmeth rate): {args.fn}')
    print(f'  FP scale (nuc meth multiplier): {args.fp_scale}')
    print(f'  breathing (nuc meth addition): {args.breathing}')
    print(f'  linker P(meth) = {1 - args.fn:.3f}')
    print(f'  nuc P(meth) ≈ {fp_rate * args.fp_scale + args.breathing:.3f}')


if __name__ == '__main__':
    main()
