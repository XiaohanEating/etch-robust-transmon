"""Finite PEC islands in the same grounded box as the HFSS reference.

Only dielectric tetrahedra are solved. Metal cavity surfaces are equipotential
electrodes, not grounded exterior walls. No edge cutoff or fitted multiplier.
Lengths in the mesh are mm; exported capacitance and energy are SI.
"""

from dataclasses import dataclass
import time

import numpy as np
import scipy.sparse as sp

from ..configs import frozen
from .fast import FieldSolution
from .fields import ElectrostaticField

VERSION = 'finite-pec-p2-v1'
EPS0 = 8.8541878128e-12


@dataclass(frozen=True)
class MeshSettings:
    thickness_nm: float = 200.0
    h_edge_mm: float = 0.005
    h_max_mm: float = 0.6
    grow_mm: float = 3.0
    threads: int = 1

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in
               (self.thickness_nm, self.h_edge_mm, self.h_max_mm, self.grow_mm)):
            raise ValueError('Finite positive thickness and mesh sizes required')
        if self.threads < 1:
            raise ValueError('At least one mesh thread required')


def mesh_film(rec, settings):
    """Boolean partition of air/substrate and finite metal cavities."""
    import gmsh
    from qiskit_metal.analyses.fem.solver import Package, PackageMesh

    start = time.perf_counter()
    lx, ly, lz = map(float, frozen.SEARCH_BOX_MM)
    z0, thick = float(frozen.SUBSTRATE_MM), settings.thickness_nm * 1e-6
    metal_xy = {'left': np.asarray(rec.pad_left.exterior.coords[:-1]) * 1e-3 + [lx/2, ly/2],
                'right': np.asarray(rec.pad_right.exterior.coords[:-1]) * 1e-3 + [lx/2, ly/2]}
    package = Package(lx, ly, lz, z0, frozen.EPS_R_SUB, metal_xy, {})
    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)
    gmsh.option.setNumber('General.NumThreads', settings.threads)
    gmsh.model.add('finite_pec')
    try:
        occ = gmsh.model.occ
        volumes = [occ.addBox(0, 0, 0, lx, ly, z0),
                   occ.addBox(0, 0, z0, lx, ly, lz-z0)]
        solids = []
        for xy in metal_xy.values():
            vertices = [occ.addPoint(float(x), float(y), z0) for x, y in xy]
            curves = [occ.addLine(vertices[i], vertices[(i+1) % len(vertices)])
                      for i in range(len(vertices))]
            sheet = occ.addPlaneSurface([occ.addCurveLoop(curves)])
            body = occ.extrude([(2, sheet)], 0, 0, thick)
            solids.append(next(tag for dim, tag in body if dim == 3))
        _, mapping = occ.fragment([(3, v) for v in volumes], [(3, v) for v in solids])
        occ.synchronize()
        metal_volumes = {tag for part in mapping[2:] for dim, tag in part if dim == 3}
        metal_faces = {}
        surface_tags = {'SM': [], 'MA': [], 'SA': []}
        for name, part in zip(metal_xy, mapping[2:]):
            faces = {tag for dim, vol in part if dim == 3
                     for _, tag in gmsh.model.getBoundary([(3, vol)], oriented=False)}
            metal_faces[name] = sorted(faces)
            for tag in faces:
                centre_z = occ.getCenterOfMass(2, tag)[2]
                surface_tags['SM' if abs(centre_z-z0) < 1e-9 else 'MA'].append(tag)
        material_volumes = {tag: mat for mat, part in zip((1, 0), mapping[:2])
                            for dim, tag in part if dim == 3 and tag not in metal_volumes}
        subfaces = {tag for vol, mat in material_volumes.items() if mat == 1
                    for _, tag in gmsh.model.getBoundary([(3, vol)], oriented=False)}
        airfaces = {tag for vol, mat in material_volumes.items() if mat == 0
                    for _, tag in gmsh.model.getBoundary([(3, vol)], oriented=False)}
        surface_tags['SA'] = sorted(subfaces & airfaces)
        occ.remove([(3, tag) for tag in metal_volumes], recursive=False)
        occ.synchronize()
        curves = {c for faces in metal_faces.values() for tag in faces
                  for _, c in gmsh.model.getBoundary([(2, tag)], oriented=False)}
        field = gmsh.model.mesh.field
        distance = field.add('Distance')
        field.setNumbers(distance, 'CurvesList', sorted(curves))
        field.setNumber(distance, 'Sampling', 80)
        threshold = field.add('Threshold')
        for key, value in [('InField', distance), ('SizeMin', settings.h_edge_mm),
                           ('SizeMax', settings.h_max_mm), ('DistMin', 0),
                           ('DistMax', settings.grow_mm)]:
            field.setNumber(threshold, key, value)
        field.setAsBackgroundMesh(threshold)
        for key, value in [('Mesh.MeshSizeMax', settings.h_max_mm),
                           ('Mesh.MeshSizeExtendFromBoundary', 0),
                           ('Mesh.MeshSizeFromPoints', 0), ('Mesh.MeshSizeFromCurvature', 0),
                           ('Mesh.LcIntegrationPrecision', 1e-3), ('Mesh.Algorithm3D', 10)]:
            gmsh.option.setNumber(key, value)
        gmsh.model.mesh.generate(3)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        lookup = np.full(int(tags.max())+1, -1, dtype=np.int64)
        lookup[tags.astype(np.int64)] = np.arange(len(tags))

        def elements(dim, tag, width):
            types, _, nodes = gmsh.model.mesh.getElements(dim, tag)
            if len(types) != 1 or int(types[0]) != {2: 2, 3: 4}[dim]:
                raise ValueError('Expected linear triangle/tetrahedron mesh')
            return lookup[nodes[0].astype(np.int64)].reshape(-1, width)

        tets, mats = [], []
        for tag, mat in material_volumes.items():
            t = elements(3, tag, 4)
            tets.append(t)
            mats.append(np.full(len(t), mat, dtype=np.int8))
        def triangles(tags):
            return np.vstack([elements(2, tag, 3) for tag in sorted(set(tags))])
        surface_tris = {name: triangles(tags) for name, tags in surface_tags.items()}
        # Unused vertices in removed solids must not enter a singular system.
        raw_t = np.vstack(tets)
        used = np.unique(raw_t)
        remap = np.full(len(coords)//3, -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        pmesh = PackageMesh(p=coords.reshape(-1, 3)[used].T.copy(),
                    t=remap[raw_t].T.copy(), material=np.concatenate(mats),
                    metal_tris={n: remap[triangles(tags)] for n, tags in metal_faces.items()},
                    junction_segs={}, probe_segs=None, probe_xy=None,
                    bounds=(0, lx, 0, ly), region='full', package=package,
                    mesh_time=time.perf_counter()-start)
        surfaces = {name: remap[tris] for name, tris in surface_tris.items()}
        if any(np.any(tris < 0) for tris in surfaces.values()):
            raise ValueError('An interface has no adjacent dielectric element')
        return pmesh, surfaces
    finally:
        gmsh.model.remove()
        if own:
            gmsh.finalize()


def electrostatics(pmesh):
    """Metal's P2 Schur complement with only physical box walls grounded."""
    from skfem import Basis, BilinearForm, ElementTetP2, MeshTet, asm
    from skfem.helpers import dot, grad
    from qiskit_metal.analyses.fem.solver import Electrostatics, _Factor, _facet_index

    start = time.perf_counter()
    est = Electrostatics.__new__(Electrostatics)
    mesh = MeshTet(pmesh.p, pmesh.t)
    element = ElementTetP2()
    basis = Basis(mesh, element)
    @BilinearForm
    def laplace(u, v, w):
        return dot(grad(u), grad(v))
    K = asm(laplace, Basis(mesh, element, elements=np.nonzero(pmesh.material == 0)[0]))
    K += pmesh.package.eps_r * asm(laplace, Basis(mesh, element,
                                     elements=np.nonzero(pmesh.material == 1)[0]))
    K = (EPS0 * 1e-3 * K).tocsr()
    boundary = mesh.boundary_facets()
    points = mesh.p[:, mesh.facets[:, boundary]]
    box = pmesh.package
    on_wall = np.zeros(len(boundary), dtype=bool)
    for axis, length in enumerate((box.Lx, box.Ly, box.Lz)):
        on_wall |= np.all(np.isclose(points[axis], 0, atol=1e-10, rtol=0), axis=0)
        on_wall |= np.all(np.isclose(points[axis], length, atol=1e-10, rtol=0), axis=0)
    ground = basis.get_dofs(facets=boundary[on_wall]).flatten()
    owner = np.full(basis.N, -1, dtype=np.int64)
    est.names = list(pmesh.metal_tris)
    for k, name in enumerate(est.names):
        dofs = basis.get_dofs(facets=_facet_index(mesh, pmesh.metal_tris[name])).flatten()
        if np.intersect1d(dofs, ground).size:
            raise ValueError('A metal island touches the grounded enclosure')
        owner[dofs] = k
    owner[ground] = -2
    interior = np.nonzero(owner == -1)[0]
    selected = np.nonzero(owner >= 0)[0]
    P = sp.csr_matrix((np.ones(len(selected)), (selected, owner[selected])),
                      shape=(basis.N, len(est.names)))
    Kim = (K[interior] @ P).toarray()
    X = _Factor(K[interior][:, interior]).solve(Kim)
    C = (P.T @ K @ P).toarray() - Kim.T @ X
    est.C = (C+C.T)/2
    est._K, est._P, est._X, est._interior, est._owner = K, P, X, interior, owner
    est._basis, est._mesh, est._p, est._t, est._order = basis, mesh, pmesh.p, pmesh.t, 2
    est.doflocs, est.time = basis.doflocs, time.perf_counter()-start
    return est


def solve_fields(rec, settings):
    pmesh, surfaces = mesh_film(rec, settings)
    print('MESH', rec.geometry_hash, pmesh.t.shape[1], 'tetrahedra', flush=True)
    est = electrostatics(pmesh)
    fs = FieldSolution(geometry_hash=rec.geometry_hash, C_matrix_fF=est.C*1e15,
                C_q_fF=float(est.port_capacitance('left', 'right')*1e15), est=est,
                port_a='left', port_b='right', mesh_time_s=pmesh.mesh_time,
                solve_time_s=est.time, package_mesh=pmesh,
                origin_xy_mm=(pmesh.package.Lx/2, pmesh.package.Ly/2))
    return fs, surfaces


def surface_participation(fs, surfaces, lj_H, ti_scale=1.0):
    from skfem.quadrature import get_quadrature_tri
    from qiskit_metal.analyses.fem.solver import _facet_index
    from .lom import transmon_metrics

    if not np.isfinite(ti_scale) or ti_scale <= 0:
        raise ValueError('Positive interface thickness multiplier required')
    field = ElectrostaticField(fs)
    est, mesh = fs.est, fs.est._mesh
    uv, weights = get_quadrature_tri(2)
    bary = np.vstack((1-uv.sum(axis=0), uv))
    p, areas, residuals = {}, {}, {}
    tm = transmon_metrics(fs.C_q_fF*1e-15, lj_H)
    denominator = 2*field.energy * tm['C_sigma_fF']/fs.C_q_fF
    for label, tris in surfaces.items():
        facets = _facet_index(mesh, tris)
        adjacent = mesh.f2t[:, facets]
        first = adjacent[0]
        if np.any(first < 0):
            first = np.where(first >= 0, first, adjacent[1])
        if label == 'SA':
            if np.any(adjacent < 0):
                raise ValueError('SA must separate substrate and air')
            a, b = adjacent
            first = np.where(fs.package_mesh.material[a] == 1, a, b)
        expected_material = 0 if label == 'MA' else 1
        if np.any(fs.package_mesh.material[first] != expected_material):
            raise ValueError('Wrong dielectric side at '+label)
        verts = mesh.p[:, mesh.facets[:, facets]]
        cross = np.cross((verts[:, 1]-verts[:, 0]).T, (verts[:, 2]-verts[:, 0]).T)
        area = np.linalg.norm(cross, axis=1)/2
        points = np.einsum('iv,nif->nvf', bary, verts).reshape(3, -1)
        wm2 = (weights[:, None]/weights.sum()*area[None, :]).ravel()*1e-6
        E = field.electric_on_elements(points, np.tile(first, len(weights)))*1e3
        eps_bulk = frozen.EPS_R_SUB if label != 'MA' else 1.0
        eps_i = frozen.INTERFACE_EPS_R[label]
        if label == 'SA':
            density = eps_i*np.sum(E[:2]**2, axis=0)+eps_bulk**2/eps_i*E[2]**2
        else:
            density = eps_bulk**2/eps_i*np.sum(E**2, axis=0)
        p['p_'+label] = float(EPS0*frozen.T_INTERFACE_NM[label]*1e-9*ti_scale
                              * np.dot(wm2, density)/denominator)
        areas[label] = float(wm2.sum())
    phi, K = field.phi, est._K
    interior_residual = (K@phi)[est._interior]
    residuals['internal_relative'] = float(np.linalg.norm(interior_residual)/np.linalg.norm(K@phi))
    p.update(tm, C_q_fF=fs.C_q_fF, evaluator_version=VERSION)
    rate = 2*np.pi*tm['f01_GHz']*1e9*sum(p['p_'+k]*frozen.TAN_DELTA[k] for k in ('SM','MA','SA'))
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError('Nonpositive loss rate')
    p.update(T1_us=1e6/rate, areas_m2=areas, energy_J=field.energy, residuals=residuals)
    return p
