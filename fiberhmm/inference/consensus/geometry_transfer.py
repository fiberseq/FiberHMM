"""Transfer source-native geometry distributions without invented point edges."""
from __future__ import annotations
import numpy as np


def integer_aliases(center,ambiguity_bp):
    x=int(ambiguity_bp);a,b=np.meshgrid(np.arange(center[0]-x,center[0]+x+1),
        np.arange(center[1]-x,center[1]+x+1),indexing='ij')
    valid=a<b
    return a[valid],b[valid]


def source_integer_mixture(kernel,family,class_mass,uniform_floor=.05):
    """Uniform conditional density WITHIN every source projection class.

    A fixed uniform floor also covers aliases invisible on the source grid.
    Source-invisible does not mean impossible in a recipient with more sites.
    This is a declared robustness floor, not a learned error probability.
    """
    if not 0<=uniform_floor<=1:raise ValueError('Uniform floor must lie in [0,1]')
    q=np.asarray(class_mass,float);native=kernel.families[family]
    if q.shape!=native['q'].shape or np.any(~np.isfinite(q)) or np.any(q<0) or q.sum()<=0:
        raise ValueError('One finite nonnegative mass per source projection required')
    q=q/q.sum();a,b=integer_aliases(kernel.centers[family],kernel.ambiguity_bp)
    projection=np.c_[np.searchsorted(kernel.positions,a),np.searchsorted(kernel.positions,b)]
    lookup={(int(ga),int(gb)):j for j,(ga,gb) in enumerate(zip(native['starts'],native['ends']))}
    ids=np.array([lookup.get(tuple(v),-1) for v in projection]);visible=ids>=0
    count=np.bincount(ids[visible],minlength=len(q));density=np.zeros(len(a))
    density[visible]=q[ids[visible]]/count[ids[visible]]
    density=(1-uniform_floor)*density+uniform_floor/len(a)
    np.testing.assert_allclose(density.sum(),1.,atol=1e-12)
    c=kernel.centers[family];x=kernel.ambiguity_bp
    shell=(np.abs(a-c[0])==x)|(np.abs(b-c[1])==x)
    return a,b,density,dict(source_invisible_integer_fraction=float((~visible).mean()),
        source_invisible_prior_mass=float(density[~visible].sum()),
        exact_box_shell_mass=float(density[shell].sum()),uniform_box_shell_mass=float(shell.mean()),
        uniform_floor=float(uniform_floor))


def recipient_projection_mixture(kernel,family,a,b,weights):
    """Map all integer aliases onto recipient projections; report lost mass.

    The counterfactual kernel uses its original conditional-visible convention.
    Shape diagnostics separately gate the full retained source mass, so an
    unobserved source mode cannot donate its weight to a favorable sliver.
    """
    q=kernel.families[family];w=np.asarray(weights,float)
    if w.shape!=np.shape(a) or np.shape(a)!=np.shape(b) or np.any(w<0) or np.any(~np.isfinite(w)):
        raise ValueError('Finite source weight per integer interval required')
    lookup={(int(ga),int(gb)):j for j,(ga,gb) in enumerate(zip(q['starts'],q['ends']))}
    out=np.zeros(len(q['q']))
    for ga,gb,weight in zip(np.searchsorted(kernel.positions,a),np.searchsorted(kernel.positions,b),w):
        j=lookup.get((int(ga),int(gb)))
        if j is not None:out[j]+=weight
    retained=float(out.sum())
    if retained:out/=retained
    return out,retained


def completed_source_cells(kernel,family,class_mass,region,uniform_floor=.05,maximum_integer_aliases=200000):
    """Complete each nominated projection's exact SOURCE-lattice boundary cell.

    The population grid chooses projection classes; it supplies no information
    within one class. A nominal +/-bp box must not clip off identical source
    observations during transfer. This extends aliases, not projection support:
    each original visible class keeps exactly its fitted probability, uniform
    within its complete cell. Terminal cells are bounded by the observed region.
    The separately declared floor stays on the original integer box.
    """
    if not 0<=uniform_floor<=1:raise ValueError('Uniform floor must lie in [0,1]')
    lo,hi=map(int,region);pos=kernel.positions;g=kernel.families[family]
    q=np.asarray(class_mass,float)
    if q.shape!=g['q'].shape or np.any(~np.isfinite(q)) or np.any(q<0) or q.sum()<=0:raise ValueError('Valid source projection masses required')
    q=q/q.sum()
    # searchsorted(position,a)=i iff p[i-1]+1 <= a <= p[i].
    lower=np.r_[lo,pos+1];upper=np.r_[pos,hi]
    rectangles=[];total=0
    for ga,gb,weight in zip(g['starts'],g['ends'],q):
        al,ah=max(lo,int(lower[ga])),min(hi,int(upper[ga]));bl,bh=max(lo,int(lower[gb])),min(hi,int(upper[gb]))
        size=max(0,ah-al+1)*max(0,bh-bl+1)
        if size==0:raise ValueError('Source projection outside recorded observation region')
        total+=size;rectangles.append((al,ah,bl,bh,weight,size))
    if total>maximum_integer_aliases:raise MemoryError(f'Complete source cells require {total} integer aliases, budget {maximum_integer_aliases}; none truncated')
    density={}
    for al,ah,bl,bh,weight,size in rectangles:
        for aa in range(al,ah+1):
            for bb in range(bl,bh+1):
                if aa>=bb:raise AssertionError('Visible source projection has invalid-width alias')
                density[(aa,bb)]=(1-uniform_floor)*weight/size
    original_a,original_b=integer_aliases(kernel.centers[family],kernel.ambiguity_bp)
    for aa,bb in zip(original_a,original_b):density[(int(aa),int(bb))]=density.get((int(aa),int(bb)),0.)+uniform_floor/len(original_a)
    pairs=np.array(sorted(density),dtype=np.int64);weights=np.array([density[tuple(v)] for v in pairs])
    np.testing.assert_allclose(weights.sum(),1.,atol=1e-12)
    c=kernel.centers[family];x=kernel.ambiguity_bp
    outside=(np.abs(pairs[:,0]-c[0])>x)|(np.abs(pairs[:,1]-c[1])>x)
    return pairs[:,0],pairs[:,1],weights,dict(completed_integer_aliases=len(pairs),
        mass_outside_nominal_box=float(weights[outside].sum()),uniform_floor=uniform_floor,
        source_projection_support_unchanged=True,region_censors_terminal_cells=True)


def projected_geometry_table(positions,center,a,b,weights):
    """Recipient projection table for exact integer-cell transfer (visible part)."""
    w=np.asarray(weights,float);pairs=np.c_[np.searchsorted(positions,a),np.searchsorted(positions,b)]
    if w.shape!=(len(pairs),) or np.any(~np.isfinite(w)) or np.any(w<0):raise ValueError('Valid alias weights required')
    visible=(pairs[:,0]<pairs[:,1])&(w>0);retained=float(w[visible].sum())
    if not retained:raise ValueError('Family has no visible nonempty opportunity projection')
    unique,inverse=np.unique(pairs[visible],axis=0,return_inverse=True)
    mass=np.bincount(inverse,weights=w[visible],minlength=len(unique))
    means=np.c_[np.bincount(inverse,weights=np.asarray(a)[visible]*w[visible])/mass,
                np.bincount(inverse,weights=np.asarray(b)[visible]*w[visible])/mass]
    return dict(center=list(map(int,center)),starts=unique[:,0],ends=unique[:,1],q=mass/retained,
        mean_integer_edges_given_projection=means,conditional_on_visible_projection=True,
        retained_source_prior_mass=retained,semantics='Source-native distribution over complete source opportunity cells')
