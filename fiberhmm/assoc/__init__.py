"""``fiberhmm.assoc``: the feature hub's association statistics (feature hub design section 7).

Four modes over plain arrays (states, marks, scalars, informative geometry); no BAM access:

- **M1** :func:`cooccurrence`: per-molecule co-occurrence of two feature types' states at pairs of hits
  (pair-stratified covariate-adjusted score with molecule-clustered variance; conditional logistic OR; per-pair
  exact tests); :func:`combination_patterns` for three or more elements.
- **M2** :func:`condition`: state-conditioned response at anchors (Delta_anchor by within-cell permutation,
  Delta_site by anchor-role permutation within matched sets); :func:`matched_backgrounds`.
- **M3** :func:`rate_ratio`: within-molecule rate of point features inside versus outside interval features
  (conditional binomial, circular-shift null); :func:`edge_association`.
- **M4** :func:`overlap_enrichment`: pooled overlap and distance enrichment of hit sets (cluster-preserving matched
  relocation, max-statistic across distances); :func:`overlap_contrast` between groups.

Nucleosome analyses (NUC 3.8, 3.9): :func:`condition_stats` and :func:`association_stats`
(``fiberhmm.assoc.nucleosome``); ``fiberhmm.nuc`` re-exports both.

Shared machinery: locus groups and the block bootstrap (:mod:`fiberhmm.assoc.resample`), strata
(:mod:`fiberhmm.assoc.strata`), conditional logistic regression (:mod:`fiberhmm.assoc.clogit`), group contrasts,
haplotypes (unphased molecules excluded, D14) and replicates (:mod:`fiberhmm.assoc.groups`), confounded-null
simulators (:mod:`fiberhmm.assoc.simulate`) and scan-directory readers (:mod:`fiberhmm.assoc.scanio`).

``coaccess`` holds the element co-accessibility statistics of ``fiberhmm-nfr`` (moved here from
``fiberhmm.inference.accessibility.coaccess``, which remains an alias of this module).
"""

_EXPORTS = {
    'AssociationCancelled': 'control',
    'cooccurrence': 'cooccur', 'combination_patterns': 'cooccur', 'FeatureStates': 'cooccur',
    'MoleculeTable': 'cooccur', 'covariate_basis': 'cooccur',
    'condition': 'condition', 'Appearances': 'condition', 'LocusTable': 'condition', 'SumMetric': 'condition',
    'mean_metric': 'condition', 'rate_metric': 'condition',
    'matched_backgrounds': 'background', 'candidate_positions': 'background',
    'rate_ratio': 'rate', 'edge_association': 'rate', 'RateData': 'rate', 'compress': 'rate',
    'overlap_enrichment': 'overlap', 'overlap_contrast': 'overlap', 'HitSet': 'overlap',
    'intersect_territories': 'overlap', 'tercile_strata': 'overlap',
    'condition_stats': 'nucleosome', 'association_stats': 'nucleosome',
    'clogit': 'clogit', 'random_effects': 'resample', 'combine_replicates': 'groups',
    'post_selection_check': 'groups', 'group_labels': 'groups',
}

__all__ = sorted(_EXPORTS) + ['coaccess']


def __getattr__(name):
    # lazy exports keep `import fiberhmm.assoc` light
    if name in _EXPORTS:
        import importlib
        return getattr(importlib.import_module(f'fiberhmm.assoc.{_EXPORTS[name]}'), name)
    raise AttributeError(name)
