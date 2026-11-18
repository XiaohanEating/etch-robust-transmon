"""Analytic P1/P2 electrostatic gradients; mm, V/mm, joules."""

import numpy as np

LOCAL_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


class ElectrostaticField:
    """Read a solved Metal FEM system without changing Metal source files."""
    def __init__(self, fs, energy=1.0):
        self.fs, self.est = fs, fs.est
        est = self.est
        charge = np.zeros(len(est.names))
        q = np.sqrt(2.0 * energy * est.port_capacitance(fs.port_a, fs.port_b))
        charge[est.index(fs.port_a)], charge[est.index(fs.port_b)] = q, -q
        vm = np.linalg.solve(est.C, charge)
        self.phi = np.asarray(est._P @ vm).ravel()
        self.phi[est._interior] = -est._X @ vm
        self.energy = float(0.5 * self.phi @ (est._K @ self.phi))
        if not np.isfinite(self.energy) or self.energy <= 0:
            raise ValueError('Nonpositive electrostatic mode energy')
        if not np.isclose(self.energy, energy, rtol=1e-7, atol=1e-12):
            raise ValueError('Field and port energy normalizations disagree')
        from qiskit_metal.analyses.fem.solver import _element_edges
        self.edge_map = _element_edges(est._mesh, est._t)

    def electric_on_elements(self, points, elements):
        """Evaluate a one-sided trace in the specified adjacent tetrahedra."""
        points, elements = np.asarray(points, dtype=float), np.asarray(elements, dtype=int)
        if np.any(elements < 0):
            raise ValueError('Missing tetrahedron on requested interface side')
        est = self.est
        tet = est._t[:, elements]
        vertices = est._p[:, tet].transpose(2, 1, 0)
        matrix = (vertices[:, 1:, :] - vertices[:, :1, :]).transpose(0, 2, 1)
        inv = np.linalg.inv(matrix)
        rest = np.einsum('nij,nj->ni', inv, points.T - vertices[:, 0, :])
        lam = np.column_stack((1.0 - rest.sum(axis=1), rest))
        gradients = np.concatenate((-inv.sum(axis=1)[:, None, :], inv), axis=1)
        node_values = self.phi[est._basis.nodal_dofs[0][tet]].T
        if est._order == 2:
            grad_phi = np.einsum('ni,ni,nij->nj', node_values, 4.0 * lam - 1.0, gradients)
            edges = est._basis.edge_dofs[0][self.edge_map[elements]]
            edge_values = self.phi[edges]
            for k, (a, b) in enumerate(LOCAL_EDGES):
                shape_grad = 4.0 * (lam[:, a, None] * gradients[:, b, :]
                                    + lam[:, b, None] * gradients[:, a, :])
                grad_phi += edge_values[:, k, None] * shape_grad
        else:
            grad_phi = np.einsum('ni,nij->nj', node_values, gradients)
        return -grad_phi.T

    def electric(self, points, chunk_size=20000):
        from qiskit_metal.analyses.fem.solver import _Locator
        if not hasattr(self, '_locator'):
            self._locator = _Locator(self.est._p, self.est._t)
        points, chunks = np.asarray(points, dtype=float), []
        for start in range(0, points.shape[1], chunk_size):
            part = points[:, start:start + chunk_size]
            elements, _ = self._locator.locate(part)
            chunks.append(self.electric_on_elements(part, elements))
        return np.concatenate(chunks, axis=1) if chunks else np.empty((3, 0))


def interface_quadrature(fs, rec, order=2, offset_um=0.0, edge_cutoff_um=0.0):
    """Whole-chip interface quadrature with explicit material-sided traces.

    Offset and edge cutoff are numerical diagnostics, not physical thicknesses.
    The degree-two rule exactly integrates squared P2 gradients on a triangle.
    """
    from skfem.quadrature import get_quadrature_tri
    import shapely
    from shapely.geometry import MultiLineString

    mesh, z0 = fs.est._mesh, fs.substrate_mm
    idx = np.nonzero(np.all(np.isclose(mesh.p[2, mesh.facets], z0,
                                      atol=1e-10, rtol=0), axis=0))[0]
    faces, adjacent = mesh.facets[:, idx], mesh.f2t[:, idx]
    if np.any(adjacent < 0):
        raise ValueError('Chip interface is not shared by two material regions')
    centres = mesh.p[2, mesh.t].mean(axis=0)
    e0, e1 = adjacent
    above = centres[e0] > z0
    air_elements, sub_elements = np.where(above, e0, e1), np.where(above, e1, e0)
    if np.any(centres[air_elements] <= z0) or np.any(centres[sub_elements] >= z0):
        raise ValueError('Interface side classification failed')
    verts = mesh.p[:, faces]
    area = np.linalg.norm(np.cross((verts[:, 1] - verts[:, 0]).T,
                                   (verts[:, 2] - verts[:, 0]).T), axis=1) / 2.0
    uv, weights = get_quadrature_tri(order)
    bary = np.vstack((1.0 - uv.sum(axis=0), uv))
    points = np.einsum('iv,nif->nvf', bary, verts).reshape(3, -1)
    weights_m2 = (weights[:, None] / weights.sum() * area[None, :]).ravel() * 1e-6
    air_elements, sub_elements = np.tile(air_elements, len(weights)), np.tile(sub_elements, len(weights))
    x_um = (points[0] - fs.origin_xy_mm[0]) * 1e3
    y_um = (points[1] - fs.origin_xy_mm[1]) * 1e3
    metal = np.zeros(len(x_um), dtype=bool)
    for poly in rec.polys:
        metal |= shapely.contains_xy(poly, x_um, y_um)
    keep = np.ones(len(x_um), dtype=bool)
    if edge_cutoff_um > 0:
        boundary = MultiLineString([list(poly.exterior.coords) for poly in rec.polys])
        keep &= shapely.distance(shapely.points(x_um, y_um), boundary) >= edge_cutoff_um
    field = ElectrostaticField(fs)
    if offset_um:
        displacement = np.array([[0.0], [0.0], [offset_um * 1e-3]])
        e_air, e_sub = field.electric(points + displacement), field.electric(points - displacement)
    else:
        e_air = field.electric_on_elements(points, air_elements)
        e_sub = field.electric_on_elements(points, sub_elements)
    return dict(E_air=e_air * 1e3, E_sub=e_sub * 1e3, weights_m2=weights_m2,
                metal=metal, keep=keep, energy_J=field.energy, n_faces=len(idx),
                points_mm=points, total_area_m2=float(weights_m2.sum()))
