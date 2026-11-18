"""Common readout coupling geometry; all coordinates are in micrometres."""
from dataclasses import asdict, dataclass
from itertools import combinations

from shapely.geometry import Polygon, box

from .novel_structures import etch_variant, nominal_pocket


@dataclass(frozen=True)
class ReadoutLayout:
    gap_um: float = 30.0
    depth_um: float = 50.0
    height_um: float = 200.0
    feed_width_um: float = 10.0
    slot_gap_um: float = 10.0
    extension_um: float = 100.0


def validate_loaded_geometry(struct):
    """Reject broken conductors, shorts and metals touching the ground."""
    if struct.get("islands"):
        raise ValueError("This pilot requires exactly two qubit islands and one readout")
    metals = {key: struct[key] for key in ("pad_left", "pad_right", "readout")}
    for key, poly in {**metals, "pocket": struct["pocket"],
                      "readout_cutout": struct["readout_cutout"]}.items():
        if not isinstance(poly, Polygon) or poly.is_empty or not poly.is_valid:
            raise ValueError(f"Invalid or disconnected geometry: {key}")
    aperture = struct["pocket"].union(struct["readout_cutout"])
    for key, poly in metals.items():
        if not aperture.contains(poly) or aperture.boundary.distance(poly) <= 1e-6:
            raise ValueError(f"Metal touches ground: {key}")
    gaps = {}
    for (name1, poly1), (name2, poly2) in combinations(metals.items(), 2):
        gap = poly1.distance(poly2)
        if gap <= 1e-6:
            raise ValueError(f"Short between {name1} and {name2}")
        gaps[f"{name1}__{name2}"] = gap
    return gaps


def attach_readout(struct, layout=ReadoutLayout()):
    """Anchor once to nominal geometry; subsequent etching never reanchors it."""
    if "readout" in struct:
        raise ValueError("Readout already attached")
    if any(value <= 0 for value in asdict(layout).values()):
        raise ValueError("Readout dimensions must be positive")
    out = dict(struct)
    pocket = struct.get("pocket")
    out["pocket"] = nominal_pocket(struct) if pocket is None else pocket
    x0 = struct["pad_right"].bounds[2] + layout.gap_um
    x1 = x0 + layout.depth_um
    end = out["pocket"].bounds[2] + layout.extension_um
    if end <= x1:
        raise ValueError("Feed endpoint must extend beyond coupling pad")
    pad = box(x0, -layout.height_um / 2, x1, layout.height_um / 2)
    feed = box(x0, -layout.feed_width_um / 2, end, layout.feed_width_um / 2)
    out["readout"] = pad.union(feed)
    out["readout_cutout"] = out["readout"].buffer(layout.slot_gap_um, join_style=2)
    out["readout_layout"] = asdict(layout)
    validate_loaded_geometry(out)
    return out


def etch_loaded(struct, delta_um):
    """Uniform metal retreat: two pads/readout shrink, ground apertures grow."""
    validate_loaded_geometry(struct)
    out = etch_variant(struct, delta_um)
    out["readout"] = struct["readout"].buffer(-delta_um, quad_segs=8)
    out["readout_cutout"] = struct["readout_cutout"].buffer(delta_um, join_style=2)
    validate_loaded_geometry(out)
    return out
