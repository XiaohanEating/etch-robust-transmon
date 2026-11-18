"""Reproducible geometry and paired perturbation-field images for VLM input."""
from pathlib import Path
import hashlib
import numpy as np


def render_triptych(record, output, mesh_h=0.05, side=72):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import shapely
    from shapely.geometry import MultiLineString
    from ..configs import frozen
    from ..evaluator import fast
    from ..evaluator.fields import ElectrostaticField
    from ..perturbation.channels import apply_etch

    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    cache=fast.FieldCache()
    mesh=dict(h_metal=mesh_h,h_max=0.6,grow=3.0)
    base=cache.get_or_solve(record,mesh)
    etch,reason=apply_etch(record,frozen.ETCH_UM)
    if reason: raise ValueError('Paired image requires a legal +etch geometry: '+reason)
    altered=cache.get_or_solve(etch,mesh)
    x=np.linspace(-frozen.FOOTPRINT_UM[0]/2,frozen.FOOTPRINT_UM[0]/2,side)
    y=np.linspace(-frozen.FOOTPRINT_UM[1]/2,frozen.FOOTPRINT_UM[1]/2,round(side*0.70))
    xx,yy=np.meshgrid(x,y)
    points=np.vstack([(xx.ravel()+3)*1e-3,(yy.ravel()+3)*1e-3,
                      np.full(xx.size,frozen.SUBSTRATE_MM+0.0005)])
    e0=ElectrostaticField(base).electric(points)
    e1=ElectrostaticField(altered).electric(points)
    mag0=np.linalg.norm(e0,axis=0).reshape(xx.shape)
    mag1=np.linalg.norm(e1,axis=0).reshape(xx.shape)
    metals=np.zeros(xx.shape,dtype=bool)
    for poly in record.polys:
        metals |= shapely.contains_xy(poly,xx,yy)
    log0=np.log10(np.maximum(mag0,1e-9)); log1=np.log10(np.maximum(mag1,1e-9))
    log0[metals]=np.nan;log1[metals]=np.nan
    diff=log1-log0
    edges=MultiLineString([list(p.exterior.coords) for p in record.polys])
    boundary=shapely.distance(shapely.points(xx,yy),edges)

    fig,axes=plt.subplots(1,3,figsize=(14,4.4),constrained_layout=True)
    fig.patch.set_facecolor('white')
    ax=axes[0]
    for poly,color,label in zip(record.polys,('#0F4D92','#42949E'),('Island 1','Island 2')):
        ax.fill(*poly.exterior.xy,color=color,alpha=.83,label=label)
    ax.set_xlim(x.min(),x.max());ax.set_ylim(y.min(),y.max());ax.set_aspect('equal')
    ax.set_xlabel('x (μm)');ax.set_ylabel('y (μm)')
    ax.set_title('Geometry and footprint')
    ax.text(.03,.97,'30 μm junction gap',transform=ax.transAxes,va='top',
            bbox=dict(facecolor='white',alpha=.82,edgecolor='none'))
    ax.legend(loc='lower right',framealpha=.82,facecolor='white')
    vmin=float(np.nanpercentile(log0,5));vmax=float(np.nanpercentile(log0,99))
    im=axes[1].pcolormesh(xx,yy,log0,cmap='magma',vmin=vmin,vmax=vmax,shading='auto')
    axes[1].set_title('Nominal log electric-field magnitude')
    axes[1].set_xlabel('x (μm)');axes[1].set_ylabel('y (μm)')
    axes[1].set_aspect('equal');fig.colorbar(im,ax=axes[1],label='log₁₀ |E| (V/m)')
    lim=float(np.nanpercentile(np.abs(diff),99));lim=max(lim,.01)
    im2=axes[2].pcolormesh(xx,yy,diff,cmap='coolwarm',vmin=-lim,vmax=lim,shading='auto')
    axes[2].set_title('Field change after +0.25 μm etch')
    axes[2].set_xlabel('x (μm)');axes[2].set_ylabel('y (μm)')
    axes[2].set_aspect('equal');fig.colorbar(im2,ax=axes[2],label='Δ log₁₀ |E|')
    for axis in axes[1:]:
        axis.text(.02,.98,'0.5 μm above substrate',transform=axis.transAxes,va='top',
                  bbox=dict(facecolor='white',alpha=.82,edgecolor='none'))
    fig.savefig(output,dpi=220,facecolor='white')
    plt.close(fig)
    return dict(path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        geometry_hash=record.geometry_hash,etch_um=frozen.ETCH_UM,
        field_offset_um=.5,mesh_h_metal_mm=mesh_h,
        field_log_quantiles=np.quantile(log0[np.isfinite(log0)],[.05,.5,.95]).tolist(),
        difference_log_quantiles=np.quantile(diff[np.isfinite(diff)],[.05,.5,.95]).tolist(),
        min_manufacturing_edge_distance_um=float(boundary.min()))
