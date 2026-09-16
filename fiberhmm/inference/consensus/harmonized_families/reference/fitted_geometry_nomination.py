# Extracted reference kernels; see SOURCE_MANIFEST.json.
import numpy as np
from .cross_source_family_consolidation import freeze_children
from fiberhmm.inference.consensus.native_cross import transfer_density
from .native_cell_consolidation import nominate_parents

def density_cell(frozen):
    grid = frozen['grid']
    density = transfer_density(frozen, grid)
    if not np.isfinite(density).any():
        raise ValueError('No finite fitted native density')
    tied = np.flatnonzero(np.isfinite(density) & (density >= np.max(density) - 1e-10))
    mean = np.asarray(frozen['geometry_summary']['mean'])
    center = np.asarray(frozen['model'].get('fitted_distribution_center', mean))
    index = min(tied, key=lambda i: (float(np.sum((grid['coordinates'][i] - center) ** 2)), int(i)))
    p = grid['positions']
    a = grid['starts'][index]
    b = grid['ends'][index]
    cell = [[int(np.r_[grid['domain'][0], p + 1][a]), int(grid['left_hi'][index])], [int(grid['right_lo'][index]), int(np.r_[p, grid['domain'][1]][b])]]
    box = frozen['geometry_summary'].get('credible_boxes', {}).get('0.95')
    return dict(cell=cell, tied_max_density_cells=len(tied), selected_index=int(index), extraction='maximum_fitted_log_density_not_mass_or_raw_profile', normalized_geometry_mean=mean.tolist(), additional_fits=0, fitted_parameter_center=center.tolist(), mode_cell_offset_from_fitted_center_bp=(grid['coordinates'][index] - center).tolist(), mode_cell_touches_domain_limit=cell[0][0] == grid['domain'][0] or cell[1][1] == grid['domain'][1], unconstrained_center_has_positive_width=bool(center[0] < center[1]), mode_cell_within_conditional_95_box=all((box[e][0] <= a <= b <= box[e][1] for (e, (a, b)) in zip(('left', 'right'), cell))) if box else None, tie_break='nearest_fitted_parameter_center_then_grid_index')

def source_density_cells(parts):
    result = {}
    for (channel, part) in sorted(parts.items()):
        for (family, frozen) in freeze_children(part).items():
            result[channel + '::' + family] = density_cell(frozen)
    return result

def nominate_fitted_parents(models, radius, cells=None, method='density_cell'):
    if method not in ('density_cell', 'distribution_box'):
        raise ValueError('Unknown fitted-geometry nomination method')
    proxies = []
    for m in models:
        if method == 'density_cell':
            cell = cells[m['family']]['cell']
        else:
            box = m['normalized_geometry']['credible_boxes']['0.95']
            cell = [box['left'], box['right']]
        proxies.append(dict(family=m['family'], reference_interval=m['reference_interval'], native_projection_cell=cell))
    proposals = nominate_parents(proxies, radius)
    for prop in proposals:
        prop['id'] = prop['id'].replace('P:G', 'P:D' if method == 'density_cell' else 'P:B', 1)
        prop.update(nomination='original_fitted_' + method, additional_initial_fits=0, second_refinement_pass=False, candidate_cells_are_not_confidence_intervals=True)
    return proposals
