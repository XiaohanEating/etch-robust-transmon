"""Shared deterministic boundary operations with a fixed junction region."""
from shapely.affinity import scale
from shapely.geometry import box
from shapely.ops import unary_union
from .clean import clean_polygon
from . import checks, library

VERSION='boundary-operations-v1'


def resize(rec, sx=1.0, sy=1.0):
    if not (0.5 <= sx <= 2.5 and 0.5 <= sy <= 2.5):
        raise ValueError('Resize factor outside allowed operation range')
    polygons=[]
    for poly,side in zip(rec.polys,(-1,1)):
        shaped=scale(poly,xfact=sx,yfact=sy,origin=(side*library.G2,0))
        # Preserve the physical junction and its protected boundary verbatim.
        band=box(-1000,-library.NECK_HALF_Y,1000,library.NECK_HALF_Y)
        result=unary_union([shaped.difference(band),poly.intersection(band)])
        # Reject clipping/disconnection rather than hiding a lost feature.
        if result.geom_type!='Polygon': raise ValueError('Operation disconnects an island')
        polygons.append(clean_polygon(result))
    return rec.with_operation('resize:x='+repr(sx)+',y='+repr(sy),*polygons)


def fill_extent(rec):
    # Stretch inside the existing footprint, leaving 0.25um for etch expansion.
    bound=library.G2+library.W2-0.25
    sx=min((bound-library.G2)/(max(abs(p.bounds[0]),abs(p.bounds[2]))-library.G2)
           for p in rec.polys)
    sy=(library.H2-0.25)/max(max(abs(p.bounds[1]),abs(p.bounds[3])) for p in rec.polys)
    # Fingers entering the centre gap must retain the minimum legal gap.
    intrusion=max(max(0,library.G2-min(abs(x) for x,y in p.exterior.coords)) for p in rec.polys)
    if intrusion:
        sx=min(sx,(library.G2-library.frozen.MIN_GAP_UM/2-0.25)/intrusion)
    return resize(rec,sx=min(2.5,sx),sy=min(2.5,sy))


def apply(rec,operation):
    kind=operation.get('kind')
    if kind=='resize':
        candidate=resize(rec,float(operation.get('sx',1)),float(operation.get('sy',1)))
    elif kind=='round':
        radius=float(operation.get('radius_um',0))
        if not 0.5 <= radius <= 20: raise ValueError('Invalid rounding radius')
        polys=[]
        for poly in rec.polys:
            shaped=poly.buffer(-radius).buffer(radius)
            band=box(-1000,-library.NECK_HALF_Y,1000,library.NECK_HALF_Y)
            result=unary_union([shaped.difference(band),poly.intersection(band)])
            if result.geom_type!='Polygon':raise ValueError('Rounding disconnects an island')
            polys.append(clean_polygon(result))
        candidate=rec.with_operation('round:'+repr(radius),*polys)
    else:
        raise ValueError('Unsupported geometry operation')
    valid,reasons=checks.check_record(candidate)
    if not valid:raise ValueError('; '.join(reasons))
    if candidate.geometry_hash==rec.geometry_hash:raise ValueError('Operation has no geometric effect')
    return candidate
