"""Paired rectangle meshes for isolating remeshing noise in etch differences.

This diagnostic changes coordinates, reassembles, and resolves the physical
problem; it never reuses the old potential as the perturbed solution.
"""

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
from shapely.geometry import box

from ..configs import frozen
from . import finite_film
from .fast import FieldSolution

VERSION = 'rectangle-piecewise-affine-v1'


def transport_nodes(points, source, target):
    """Map each rectangle face exactly and keep all enclosure wall planes."""
    for rec in (source, target):
        for poly in rec.polys:
            if poly.symmetric_difference(box(*poly.bounds)).area > 1e-8:
                raise ValueError('Paired mapping supports only axis-aligned rectangles')
    result = np.asarray(points, dtype=float).copy()
    lengths = frozen.SEARCH_BOX_MM[:2]
    for axis, length in enumerate(lengths):
        source_knots, target_knots = [0.0], [0.0]
        values = sorted(set(value for poly in source.polys
                            for value in (poly.bounds[axis], poly.bounds[axis+2])))
        for value in values:
            correspond = []
            for a, b in zip(source.polys, target.polys):
                for bound in (axis, axis+2):
                    if a.bounds[bound] == value:
                        correspond.append(b.bounds[bound])
            if np.ptp(correspond) > 1e-8:
                raise ValueError('Rectangle faces lack a common coordinate mapping')
            source_knots.append(value*1e-3+length/2)
            target_knots.append(correspond[0]*1e-3+length/2)
        source_knots.append(float(length))
        target_knots.append(float(length))
        if min(np.diff(source_knots)) <= 0 or min(np.diff(target_knots)) <= 0:
            raise ValueError('Coordinate mapping folds or leaves the enclosure')
        if np.min(result[axis]) < -1e-10 or np.max(result[axis]) > length+1e-10:
            raise ValueError('Source mesh lies outside the enclosure')
        result[axis] = np.interp(result[axis], source_knots, target_knots)
    return result


def solve_from_archive(folder, target, settings):
    from qiskit_metal.analyses.fem.solver import Package, PackageMesh
    from ..experiments.field_audit import load_solution, sha256

    start = time.perf_counter()
    folder = Path(folder)
    meta = json.loads((folder/'finite_model.json').read_text(encoding='utf-8'))
    if (meta['identity']['settings'] != asdict(settings) or
            meta['identity']['frozen_config'] != frozen.frozen_hash()):
        raise ValueError('Parent mesh settings or physical parameters differ')
    if sha256(folder/'surfaces.npz') != meta['surfaces_sha256']:
        raise ValueError('Parent interface archive checksum differs')
    old, source, _ = load_solution(folder)
    points = transport_nodes(old.est._p, source, target)
    t = old.est._t.copy()
    before = old.est._p[:, t].transpose(2, 1, 0)
    after = points[:, t].transpose(2, 1, 0)
    det0 = np.linalg.det(before[:, 1:]-before[:, :1])
    det1 = np.linalg.det(after[:, 1:]-after[:, :1])
    ratio = det1/det0
    if not np.all(np.isfinite(ratio)) or np.any(ratio <= 0):
        raise ValueError('Transported mesh contains an inverted or degenerate element')
    lx, ly, lz = map(float, frozen.SEARCH_BOX_MM)
    metal_xy = {name: np.asarray(poly.exterior.coords[:-1])*1e-3+[lx/2, ly/2]
                for name, poly in zip(old.est.names, target.polys)}
    package = Package(lx, ly, lz, frozen.SUBSTRATE_MM, frozen.EPS_R_SUB, metal_xy, {})
    pm = PackageMesh(p=points, t=t, material=old.package_mesh.material.copy(),
                     metal_tris={k: v.copy() for k, v in old.package_mesh.metal_tris.items()},
                     junction_segs={}, probe_segs=None, probe_xy=None, bounds=(0, lx, 0, ly),
                     region='full', package=package, mesh_time=time.perf_counter()-start)
    with np.load(folder/'surfaces.npz', allow_pickle=False) as saved:
        surfaces = {k: saved[k] for k in saved.files}
    names, port_a, port_b = old.est.names, old.port_a, old.port_b
    del old
    print('TRANSPORT', target.geometry_hash, t.shape[1], 'tetrahedra', flush=True)
    est = finite_film.electrostatics(pm)
    if est.names != names:
        raise ValueError('Transported conductor order changed')
    fs = FieldSolution(geometry_hash=target.geometry_hash, C_matrix_fF=est.C*1e15,
                       C_q_fF=float(est.port_capacitance(port_a, port_b)*1e15),
                       origin_xy_mm=(lx/2, ly/2), est=est, port_a=port_a, port_b=port_b,
                       mesh_time_s=pm.mesh_time, solve_time_s=est.time, package_mesh=pm)
    return fs, surfaces
