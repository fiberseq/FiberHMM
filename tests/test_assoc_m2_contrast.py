"""M2 contrasts count measured support and use the full crossed bootstrap."""
import numpy as np
import pytest
from scipy.stats import t as tdist

from fiberhmm.assoc import condition as CD


def _sparse_groups():
    rng = np.random.default_rng(213)
    loc, mol, state, mask, values, labels = [], [], [], [], [], []
    for unit in range(24):
        for role in range(2):
            locus = 2 * unit + role
            for group in (1, 2):
                for st in ((0, 1) if unit < 6 else (1,)):
                    for rep in range(2):
                        # Four molecules connect the six informative matched sets.
                        molecule = 2 * (group - 1) + st if rep == 0 else 4 + 4 * unit + 2 * (group - 1) + st
                        loc.append(locus); mol.append(molecule); state.append(st); mask.append(1)
                        values.append(rng.normal() + 0.2 * group * st * (role == 0))
                        labels.append(group)
    # Unmeasured and unphased appearances must not supply contrast clusters.
    for unit in range(24):
        loc.extend([2 * unit, 2 * unit])
        mol.extend([1000 + 2 * unit, 1001 + 2 * unit])
        state.extend([0, 1]); mask.extend([0, 1]); values.extend([0.0, 0.0]); labels.extend([1, 0])
    y = np.asarray(values)
    app = CD.Appearances(loc, mol, state, np.zeros(len(loc), int), mask,
                         np.stack([y, np.ones_like(y)], -1), labels={'haplotype': labels})
    loci = CD.LocusTable([f'locus:{i}' for i in range(48)], ['c'] * 48, np.arange(48) * 5000,
                         np.tile([True, False], 24), np.repeat(np.arange(24), 2))
    return app, loci


def test_m2_contrast_uses_two_way_covariance_and_only_contributing_clusters(monkeypatch):
    app, loci = _sparse_groups()
    collected = []
    original = CD._boot_bin

    def capture(*args, **kwargs):
        boot = original(*args, **kwargs)
        collected.append(boot[1])
        return boot

    monkeypatch.setattr(CD, '_boot_bin', capture)
    result = CD.condition(app, loci, CD.mean_metric(), n_perm=0, n_role=0, n_boot=400,
                           per_anchor=False, group='haplotype', group_kind='haplotype', seed=19)
    interaction = result['bins'][0]['interaction']
    support = interaction['bootstrap_support']
    assert interaction['variance'] == 'twoway'
    assert support['units'] == 6
    assert support['molecules'] == 28
    assert support['intersections'] == 48
    assert support['clusters'] == 6
    assert support['group_clusters'] == {'1': 6, '2': 6}
    assert interaction['df_clusters'] == 5
    assert result['bootstrap']['units'] == 24
    assert result['unphased']['note'].startswith('descriptive; excluded')

    # Check the crossed covariance directly, including its cross-group terms.
    boot = collected[0]
    names = ['1|delta_site', '2|delta_site']
    covs = {}
    for part, draws in boot.parts.items():
        paired = np.column_stack([draws[name] for name in names])
        paired = paired[np.all(np.isfinite(paired), axis=1)]
        covs[part] = np.cov(paired, rowvar=False)
    covariance = covs['units'] + covs['molecules'] - covs['items']
    assert np.linalg.eigvalsh(covariance).min() > 0
    contrast = np.array([-1.0, 1.0])
    se = np.sqrt((contrast @ covariance @ contrast) * 6 / 5)
    row = interaction['differences'][0]
    assert row['se'] == pytest.approx(se)
    assert interaction['p'] == pytest.approx(2 * tdist.sf(abs(row['difference'] / se), 5))
    molecule_se = np.sqrt((contrast @ covs['molecules'] @ contrast) * 28 / 27)
    assert abs(se - molecule_se) > 0.01


@pytest.mark.parametrize('matched', [False, True])
def test_m2_contrast_support_is_bin_specific_and_handles_anchor_only(matched):
    app, loci = _sparse_groups()
    app.stats = np.repeat(app.stats, 2, axis=1)
    # The second bin measures just three of the six informative matched sets.
    app.mask = np.where((app.locus < 6) & (app.mask == 1), 3, app.mask)
    if not matched:
        loci.set_id[:] = -1
    result = CD.condition(app, loci, CD.mean_metric(), n_perm=0, n_role=0, n_boot=200,
                           per_anchor=False, group='haplotype', group_kind='haplotype', seed=8)
    interactions = [b['interaction'] for b in result['bins']]
    assert [i['bootstrap_support']['units'] for i in interactions] == [6, 3]
    assert [i['df_clusters'] for i in interactions] == [5, 2]
    assert all(i['statistic'] == ('delta_site' if matched else 'delta_anchor') for i in interactions)
    assert all(np.isfinite(i['p']) for i in interactions)


def test_m2_contrast_intersection_draws_keep_a_molecule_together_within_a_set():
    from fiberhmm.assoc.resample import BlockBootstrap, locus_groups

    app, loci = _sparse_groups()
    app = app.subset((app.labels['haplotype'] != 0) & (app.mask == 1))
    # The same state and response at each role: every locus's Delta is the
    # same as its matched partner's. Independent appearance weights break
    # this cancellation; whole (set, molecule) weights must preserve it.
    app.stats[app.locus % 2 == 1] = app.stats[app.locus % 2 == 0]
    unit = loci.set_id[app.locus]
    bb = BlockBootstrap(locus_groups(unit, app.molecule, 24), unit, app.molecule, seed=31)
    metric = CD.mean_metric()
    bd = CD._Bin(app, loci, metric, 0, 'bin', True, 1)
    groups = ['1', '2']
    labels = app.labels['haplotype'].astype(str)
    gbd = {g: CD._Bin(app.subset(labels == g), loci, metric, 0, 'bin', True, 1) for g in groups}
    ordinary, contrast = CD._boot_bin(bd, loci, bb, 100, labels, groups, gbd, return_contrast=True)
    for g in groups:
        name = f'{g}|delta_site'
        original = ordinary.parts['items'][name]
        corrected = contrast.parts['items'][name]
        assert np.nanstd(original) > 0.01
        assert np.nanmax(np.abs(corrected)) < 1e-12
        assert contrast.parts['units'][name] is ordinary.parts['units'][name]
        assert contrast.parts['molecules'][name] is ordinary.parts['molecules'][name]


def test_m2_unequal_group_support_reduces_to_the_welch_test(monkeypatch):
    from scipy.stats import ttest_ind
    from fiberhmm.assoc.groups import group_contrast
    from fiberhmm.assoc.resample import BootDraws

    x = np.array([-3.0, 1.0, 4.0])
    y = np.array([1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    counts = {'a': len(x), 'b': len(y)}
    # Exact covariance of nonparametric bootstrap sample means, before CR1.
    covariance = np.diag([x.var() / len(x), y.var() / len(y)])
    boot = BootDraws('twoway', info={'clusters': len(x) + len(y)})
    monkeypatch.setattr(boot, 'cov', lambda *args: covariance.copy())
    result = group_contrast({'a': x.mean(), 'b': y.mean()}, boot, cluster_counts=counts)
    expected = ttest_ind(x, y, equal_var=False)
    assert result['p'] == pytest.approx(expected.pvalue)
    assert result['df_clusters'] == pytest.approx(expected.df)
    assert result['differences'][0]['se'] == pytest.approx(np.sqrt(x.var(ddof=1) / len(x) +
                                                                 y.var(ddof=1) / len(y)))
    result = group_contrast({'a': x.mean(), 'b': y.mean()}, boot, cluster_counts={'a': 1, 'b': len(y)})
    assert np.isnan(result['p'])
