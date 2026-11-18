"""Reproduce interface diagnostics on one archived electrostatic solution.

The archive uses NumPy arrays and JSON, without pickle or a new field solve.
This diagnostic does not change the evaluator or the G1 acceptance criteria.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from ..configs import frozen
from ..evaluator.fast import FieldSolution, solve_fields
from ..evaluator.fields import ElectrostaticField, interface_quadrature
from ..geometry import library
from ..runtime.records import ROOT, provenance, write_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_solution(folder, fs, rec, mesh_kwargs):
    """Archive the assembled system and geometry before any postprocessing."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'solution.json').exists():
        raise FileExistsError('Use a new archive directory for a new solve')
    est = fs.est
    arrays = dict(p=est._p, t=est._t, mesh_t=est._mesh.t,
                  material=fs.package_mesh.material, C=est.C, X=est._X,
                  owner=est._owner, interior=est._interior,
                  doflocs=est.doflocs)
    arrays.update({'metal_' + str(k): fs.package_mesh.metal_tris[name]
                   for k, name in enumerate(est.names)})
    np.savez_compressed(folder / 'solution.npz', **arrays)
    sp.save_npz(folder / 'stiffness.npz', est._K)
    write_json(folder / 'solution.json', dict(
        geometry_hash=rec.geometry_hash, category=rec.category,
        polygons_wkt=[p.wkt for p in rec.polys], junction_wkt=rec.junction.wkt,
        names=est.names, order=est._order, port_a=fs.port_a, port_b=fs.port_b,
        C_q_fF=fs.C_q_fF, origin_xy_mm=fs.origin_xy_mm,
        substrate_mm=fs.substrate_mm, eps_r=frozen.EPS_R_SUB,
        mesh_kwargs=mesh_kwargs, mesh_time_s=fs.mesh_time_s,
        solve_time_s=fs.solve_time_s, provenance=provenance(),
        archive_sha256={name: sha256(folder / name)
                        for name in ('solution.npz', 'stiffness.npz')}))


def load_solution(folder):
    """Restore array state and the native basis, without assembling or solving."""
    from skfem import Basis, ElementTetP1, ElementTetP2, MeshTet
    from qiskit_metal.analyses.fem.solver import Electrostatics
    from shapely import wkt
    from ..geometry.record import GeometryRecord

    folder = Path(folder)
    meta = json.loads((folder / 'solution.json').read_text(encoding='utf-8'))
    for name, expected in meta['archive_sha256'].items():
        if sha256(folder / name) != expected:
            raise ValueError('Archive checksum mismatch: ' + name)
    with np.load(folder / 'solution.npz', allow_pickle=False) as saved:
        a = {name: saved[name] for name in saved.files}
    mesh = MeshTet(a['p'], a['t'])
    basis = Basis(mesh, ElementTetP2() if meta['order'] == 2 else ElementTetP1())
    if not np.array_equal(mesh.t, a['mesh_t']) or not np.array_equal(basis.doflocs, a['doflocs']):
        raise ValueError('Native mesh/basis ordering differs from the archive')
    est = Electrostatics.__new__(Electrostatics)
    est.C, est.names = a['C'], meta['names']
    est._mesh, est._basis, est._p, est._t = mesh, basis, a['p'], a['t']
    est._order, est.doflocs = meta['order'], a['doflocs']
    est._K = sp.load_npz(folder / 'stiffness.npz').tocsr()
    est._X, est._owner, est._interior = a['X'], a['owner'], a['interior']
    on_metal = np.flatnonzero(est._owner >= 0)
    est._P = sp.csr_matrix((np.ones(len(on_metal)), (on_metal, est._owner[on_metal])),
                          shape=(basis.N, len(est.names)))
    pmesh = SimpleNamespace(p=a['p'], t=a['t'], material=a['material'],
                           package=SimpleNamespace(eps_r=meta['eps_r']),
                           metal_tris={name: a['metal_' + str(k)]
                                       for k, name in enumerate(est.names)})
    fs = FieldSolution(geometry_hash=meta['geometry_hash'], C_matrix_fF=est.C * 1e15,
                       C_q_fF=meta['C_q_fF'], origin_xy_mm=tuple(meta['origin_xy_mm']),
                       est=est, port_a=meta['port_a'], port_b=meta['port_b'],
                       mesh_time_s=meta['mesh_time_s'], solve_time_s=meta['solve_time_s'],
                       substrate_mm=meta['substrate_mm'], package_mesh=pmesh)
    rec = GeometryRecord(family='two_pad_free', category=meta['category'],
                         category_index=0, seed=0,
                         pad_left=wkt.loads(meta['polygons_wkt'][0]),
                         pad_right=wkt.loads(meta['polygons_wkt'][1]),
                         junction=wkt.loads(meta['junction_wkt']))
    if rec.geometry_hash != meta['geometry_hash']:
        raise ValueError('Archived geometry hash differs')
    return fs, rec, meta


def audit(fs, rec):
    """Check labels, constraints, native gradients, energy, and SA flux scales."""
    import shapely

    est, pm = fs.est, fs.package_mesh
    field = ElectrostaticField(fs)
    phi, K = field.phi, est._K
    z = est._p[2, est._t]
    zmin, zmax = z.min(axis=0), z.max(axis=0)
    z0 = fs.substrate_mm
    inferred = (z.mean(axis=0) < z0).astype(int)
    crosses = (zmin < z0 - 1e-10) & (zmax > z0 + 1e-10)
    materials = dict(tetrahedra=len(inferred),
                     wrong_labels=int(np.sum(pm.material != inferred)),
                     crossing_interface=int(crosses.sum()),
                     substrate_tetrahedra=int(np.sum(pm.material == 1)),
                     air_tetrahedra=int(np.sum(pm.material == 0)))
    volts = np.array([np.mean(phi[est._owner == k]) for k in range(len(est.names))])
    dv = float(np.ptp(volts))
    constraints = dict(metal_potentials_V=volts.tolist(),
                       metal_spread_over_port_voltage=[float(np.ptp(phi[est._owner == k]) / dv)
                                                       for k in range(len(est.names))],
                       ground_max_over_port_voltage=float(np.max(np.abs(phi[est._owner == -2])) / dv))
    wrong_metal = []
    for k, poly in enumerate(rec.polys):
        loc = est.doflocs[:, est._owner == k]
        xy = (loc[:2] - np.asarray(fs.origin_xy_mm)[:, None]) * 1e3
        inside = shapely.covers(poly.buffer(1e-6), shapely.points(xy.T))
        wrong_metal.append(int(np.sum(~inside | (np.abs(loc[2] - z0) > 1e-10))))
    constraints['metal_dofs_outside_expected_pad'] = wrong_metal
    r = K @ phi
    free = est._interior
    lhs = K[free][:, free] @ phi[free]
    rhs = K[free] @ (est._P @ volts)
    equations = dict(interior_residual_relative=float(np.linalg.norm(r[free]) /
                     max(np.linalg.norm(lhs) + np.linalg.norm(rhs), 1e-300)),
                     reaction_charge_C=(est._P.T @ r).tolist(),
                     reaction_sum_with_ground_C=float(r.sum()))

    native_gradient = -est._basis.interpolate(phi).grad
    coords = np.asarray(est._basis.global_coordinates())
    squared_error = squared_native = maximum_error = 0.0
    for start in range(0, est._t.shape[1], 2048):
        stop = min(start + 2048, est._t.shape[1])
        pts = coords[:, start:stop].reshape(3, -1)
        elements = np.repeat(np.arange(start, stop), coords.shape[2])
        values = field.electric_on_elements(pts, elements)
        expected = native_gradient[:, start:stop].reshape(3, -1)
        difference = values - expected
        squared_error += np.sum(difference ** 2)
        squared_native += np.sum(expected ** 2)
        maximum_error = max(maximum_error, float(np.max(np.abs(difference))))
    eps_r = np.where(pm.material == 1, pm.package.eps_r, 1.0)
    energy_native = float(0.5 * 8.8541878128e-12 * 1e-3 * np.sum(
                         np.sum(native_gradient ** 2, axis=0) * est._basis.dx * eps_r[:, None]))
    gradients = dict(relative_l2=float(np.sqrt(squared_error / squared_native)),
                     max_absolute_V_per_mm=maximum_error,
                     native_energy_J=energy_native, matrix_energy_J=field.energy,
                     energy_relative_difference=float(abs(energy_native / field.energy - 1)),
                     mesh_and_original_local_vertex_order_equal=bool(np.array_equal(est._mesh.t, est._t)))
    q = interface_quadrature(fs, rec)
    sa, w = ~q['metal'], q['weights_m2']
    air, sub = q['E_air'][2], pm.package.eps_r * q['E_sub'][2]
    tangential = q['E_air'][:2] - q['E_sub'][:2]
    flux = {}
    xy_um = (q['points_mm'][:2] - np.asarray(fs.origin_xy_mm)[:, None]) * 1e3
    regions = dict(full=sa,
                   junction=sa & (np.abs(xy_um[0]) < 20) & (np.abs(xy_um[1]) < 20))
    for name, mask in regions.items():
        weight = w[mask]
        a, b = air[mask], sub[mask]
        norm = np.sqrt(np.sum(weight * b ** 2))
        flux[name] = dict(area_m2=float(weight.sum()),
                          signed_air_V_m=float(np.sum(weight * a)),
                          signed_substrate_V_m=float(np.sum(weight * b)),
                          absolute_air_V_m=float(np.sum(weight * np.abs(a))),
                          absolute_substrate_V_m=float(np.sum(weight * np.abs(b))),
                          absolute_jump_V_m=float(np.sum(weight * np.abs(a - b))),
                          relative_l2=float(np.sqrt(np.sum(weight * (a - b) ** 2)) / max(norm, 1e-300)),
                          tangential_jump_l2_V=float(np.sqrt(np.sum(weight * np.sum(tangential[:, mask] ** 2, axis=0)))))
    return dict(materials=materials, constraints=constraints, equations=equations,
                gradients=gradients, SA_flux=flux, phi_sha256=hashlib.sha256(phi.tobytes()).hexdigest(),
                G1_passed=False, note='Diagnostics alone cannot establish participation convergence or cross-solver calibration.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--h-metal', type=float, default=0.01)
    args = parser.parse_args()
    folder = args.output
    if not (folder / 'solution.json').exists():
        rec = library.rect_record(200.0, 300.0, 0.0, note='field-audit')
        mesh = dict(h_metal=args.h_metal, h_max=0.6, grow=3.0)
        print('SOLVE one nominal rectangle; archive before diagnostics', flush=True)
        fs = solve_fields(rec, mesh, order=2)
        save_solution(folder, fs, rec, mesh)
        print('ARCHIVED', fs.est._t.shape[1], 'tetrahedra;', round(fs.solve_time_s, 3), 'solve seconds', flush=True)
        del fs
    fs, rec, metadata = load_solution(folder)
    if metadata['mesh_kwargs']['h_metal'] != args.h_metal:
        raise ValueError('Requested mesh size differs from the archived solve')
    report = audit(fs, rec)
    report.update(archive_sha256=metadata['archive_sha256'], provenance=provenance())
    write_json(folder / 'audit.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'provenance'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
