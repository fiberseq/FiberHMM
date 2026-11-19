"""Write deterministic fingerprints for paired checks against an unmodified checkout.

Run this script twice, selecting each checkout through PYTHONPATH, then compare
its JSON outputs. It uses main-test nulls and planted effects, bootstrap
intervals, and the M2 grouped results with only the interaction removed.
"""
import hashlib
import json
import sys

import numpy as np

from fiberhmm.assoc.condition import condition
from fiberhmm.assoc.cooccur import cooccurrence
from fiberhmm.assoc.overlap import overlap_enrichment, simulate_overlap
from fiberhmm.assoc.rate import rate_ratio
from fiberhmm.assoc.simulate import simulate_m1, simulate_m2, simulate_m3


def digest(result):
    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    data = json.dumps(result, default=encode, sort_keys=True).encode()
    return hashlib.sha256(data).hexdigest()


def main():
    fingerprints = {}
    for seed in (11, 29, 47):
        for planted in (False, True):
            s = simulate_m1(seed, effect=0.5 if planted else 0.0)
            result = cooccurrence(s['a'], s['b'], s['molecules'], n_boot=300, per_pair=True, seed=seed)
            fingerprints[f'M1:{seed}:{planted}'] = digest(result)
            for group in (None, 'dataset', 'haplotype'):
                s = simulate_m2(seed, site_effect=0.4 if planted else 0.0, n_datasets=2, unphased_fraction=0.2)
                result = condition(s['app'], s['loci'], s['metric'], n_boot=300, per_anchor=False,
                                   group=group, group_kind='haplotype' if group == 'haplotype' else 'label',
                                   seed=seed, return_null=True)
                for b in result['bins']:
                    b.pop('interaction', None)
                fingerprints[f'M2:{group}:{seed}:{planted}'] = digest(result)
            s = simulate_m3(seed, beta=-0.3 if planted else 0.0)
            result = rate_ratio(s, n_boot=300, seed=seed)
            fingerprints[f'M3:{seed}:{planted}'] = digest(result)
            s = simulate_overlap(seed, n_a=1000, n_b=2000, planted=0.2 if planted else 0.0)
            result = overlap_enrichment(s['a'], s['b'], s['territory'], s['strata'], seed=seed)
            fingerprints[f'M4:{seed}:{planted}'] = digest(result)
            print(f'{seed}, planted={planted} done', flush=True)
    with open(sys.argv[1], 'w') as handle:
        json.dump(fingerprints, handle, indent=2, sort_keys=True)
        handle.write('\n')


if __name__ == '__main__':
    main()
