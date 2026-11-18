"""Read finite-film archives to locate spatial resolution and flux errors.

Quadrature exactly integrates each P2 squared-gradient trace. Distance to its
nearest point describes mesh resolution, not an omitted integration region.
"""

import argparse
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import MultiLineString

from ..configs import frozen
from ..evaluator.fields import ElectrostaticField
from ..runtime.records import provenance, write_json
from .field_audit import load_solution, sha256


def inspect_archive(folder):
    from skfem.quadrature import get_quadrature_tri
    from qiskit_metal.analyses.fem.solver import _facet_index

    folder = Path(folder)
    fs, rec, meta = load_solution(folder)
    with np.load(folder/'surfaces.npz', allow_pickle=False) as saved:
        surfaces = {k: saved[k] for k in saved.files}
    field, mesh = ElectrostaticField(fs), fs.est._mesh
    uv, weights = get_quadrature_tri(2)
    bary = np.vstack((1-uv.sum(axis=0), uv))
    boundary = MultiLineString([list(poly.exterior.coords) for poly in rec.polys])
    rows = {}
    for label, triangles in surfaces.items():
        facets = _facet_index(mesh, triangles)
        vertices = mesh.p[:, mesh.facets[:, facets]]
        cross = np.cross((vertices[:, 1]-vertices[:, 0]).T,
                         (vertices[:, 2]-vertices[:, 0]).T)
        areas = np.linalg.norm(cross, axis=1)/2
        normals = cross/(2*areas[:, None])
        edge = np.linalg.norm(vertices[:, [1, 2, 0]]-vertices[:, [0, 1, 2]], axis=0)
        points = np.einsum('iv,nif->nvf', bary, vertices).reshape(3, -1)
        adjacent = mesh.f2t[:, facets]
        first = np.where(adjacent[0] >= 0, adjacent[0], adjacent[1])
        if label == 'SA':
            if np.any(adjacent < 0):
                raise ValueError('SA is missing a dielectric side')
            first = np.where(fs.package_mesh.material[adjacent[0]] == 1,
                             adjacent[0], adjacent[1])
        e = field.electric_on_elements(points, np.tile(first, len(weights)))
        wm = (weights[:, None]/weights.sum()*areas).ravel()
        density = np.sum(e**2, axis=0)
        if label == 'SA':
            ei = frozen.INTERFACE_EPS_R[label]
            density = ei*np.sum(e[:2]**2, axis=0)+frozen.EPS_R_SUB**2/ei*e[2]**2
        contribution = density*wm
        xy = (points[:2].T-np.asarray(fs.origin_xy_mm))*1000
        distance = shapely.distance(shapely.points(xy), boundary)
        row = dict(surface_triangles=len(facets),
                   surface_edge_um_quantiles=(np.quantile(edge, [0, .1, .5, .9, 1])*1000).tolist(),
                   nearest_quadrature_to_planar_edge_um=float(distance.min()),
                   cumulative_interface_integral_fraction_within_um={
                       str(r): float(contribution[distance <= r].sum()/contribution.sum())
                       for r in (.2, .5, 1, 2, 5, 10, 50)})
        if label == 'SA':
            second = np.where(fs.package_mesh.material[adjacent[0]] == 0,
                              adjacent[0], adjacent[1])
            air = field.electric_on_elements(points, np.tile(second, len(weights)))
            sub_d = frozen.EPS_R_SUB*e[2]
            row['normal_D_jump_relative_l2'] = float(np.sqrt(
                np.dot(wm, (air[2]-sub_d)**2)/np.dot(wm, sub_d**2)))
            row['tangential_E_jump_relative_l2'] = float(np.sqrt(
                np.dot(wm, np.sum((air[:2]-e[:2])**2, axis=0))/
                max(np.dot(wm, np.sum(e[:2]**2, axis=0)), 1e-300)))
        else:
            normal_e = np.sum(e.T*np.tile(normals, (len(weights), 1)), axis=1)
            tangential = e.T-normal_e[:, None]*np.tile(normals, (len(weights), 1))
            row['PEC_tangential_energy_fraction'] = float(np.dot(
                wm, np.sum(tangential**2, axis=1))/np.dot(wm, np.sum(e**2, axis=0)))
        rows[label] = row
    return dict(geometry_hash=rec.geometry_hash, mesh=meta['mesh_kwargs'],
                source_sha256={name: sha256(folder/name) for name in
                               ('solution.json', 'solution.npz', 'stiffness.npz', 'surfaces.npz')},
                rows=rows, G1_passed=False,
                note='Resolution and flux diagnostics; no cutoff, correction, or new gate threshold.',
                provenance=provenance())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    report = inspect_archive(args.input)
    write_json(args.input/'edge_diagnostics.json', report)
    print({label: dict(nearest_quadrature_to_planar_edge_um=row['nearest_quadrature_to_planar_edge_um'],
                       normal_D_jump_relative_l2=row.get('normal_D_jump_relative_l2'),
                       PEC_tangential_energy_fraction=row.get('PEC_tangential_energy_fraction'))
           for label, row in report['rows'].items()}, flush=True)


if __name__ == '__main__':
    main()
